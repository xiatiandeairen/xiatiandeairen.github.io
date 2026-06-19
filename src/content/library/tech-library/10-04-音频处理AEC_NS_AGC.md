---
title: "音频处理：AEC / NS / AGC（WebRTC 域）"
slug: "10-04"
collection: "tech-library"
group: "webrtc"
order: 10004
summary: "前置依赖：第 1 章（信令 & SDP）、第 3 章（RTP/SRTP）。本章假设读者理解音频采样基础（PCM、采样率、帧长）、FFT/频域基本概念，以及 WebRTC 端到端媒体路径（getUserMedia → RTCPeerConnection → 网络）。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-15T14:17:51.000Z"
updatedAt: "2026-06-15T14:17:51.000Z"
---
> **前置依赖**：第 1 章（信令 & SDP）、第 3 章（RTP/SRTP）。本章假设读者理解音频采样基础（PCM、采样率、帧长）、FFT/频域基本概念，以及 WebRTC 端到端媒体路径（getUserMedia → RTCPeerConnection → 网络）。

---

## TL;DR

- **APM（Audio Processing Module）** 是 libwebrtc 的音频处理核心，由 Google 2011 年随 WebRTC 开源，串联 AEC → NS → AGC 三道处理。
- **AEC（Acoustic Echo Cancellation）** 消除扬声器声音被麦克风拾取后的回路：LMS/NLMS 自适应滤波器在频域持续追踪扬声器信号的「反射模型」，从麦克风信号中减去估计的回声。
- **NS（Noise Suppression）** 估计背景噪声的统计特征，用维纳滤波（Wiener filter）在频域抑制平稳噪声；WebRTC APM 用的是基于最小值跟踪的噪声估计器。
- **AGC（Automatic Gain Control）** 把输入音量自动拉到目标电平：数字 AGC 分「固定增益」和「自适应限幅」两模式，确保远端不爆音也不听不到。
- **Jitter Buffer** 是音频送达接收端后的最后一道屏障：重排乱序 RTP 包、平滑网络抖动，代价是引入固定的 playout delay。
- 浏览器通过 `chrome://webrtc-internals` 可实时观测 `audioLevel`、`echoReturnLoss`、`jitterBufferDelay` 等统计量，是调试音频质量的第一工具。

---

## 1 设计考古：APM 的由来与动机

### 1.1 WebRTC APM 的历史

Google 于 2004 年收购 Global IP Solutions（GIPS），后者是当时语音通话质量的技术领先者，其 VoiceEngine 内嵌了工业级 AEC/NS/AGC。2011 年 Google 将 WebRTC 开源，APM 随之对外。

> **出处**：WebRTC 项目首页 [webrtc.org](https://webrtc.org/getting-started/overview)，2011-06-01 首次对外公告。GIPS 收购记录：Google Blog 2010-05-03 "Google acquires GIPS"。

APM 的三道处理有明确的处理顺序约束：

```
麦克风 PCM input
        │
        ▼
   AEC（消回声）   ← 需要参考信号：同时刻的扬声器 PCM（远端来的）
        │
        ▼
   NS（降噪）     ← 处理 AEC 残余噪声 + 环境噪声
        │
        ▼
   AGC（增益控制）← 在 NS 后做，避免把噪声也放大
        │
        ▼
   Opus 编码器 → RTP
```

顺序不可颠倒。若先 AGC 再 AEC，会导致参考信号与麦克风信号电平不匹配，AEC 收敛变慢。

### 1.2 为什么回声是个难题

全双工通话中，扬声器发出的远端声音被麦克风重新拾取，形成回路：

```
远端音频 → 扬声器 → 房间混响 → 麦克风 → 本地 APM → 编码 → 网络 → 远端耳机
                  ↑_____________回声路径_____________________________↑
```

挑战在于：
1. **房间冲激响应（RIR）** 动态变化。人移动、门开关都会改变反射特性。
2. **非线性失真**：扬声器在高音量下产生非线性谐波，线性滤波器无法完全消除。
3. **双讲（double talk）**：本地和远端同时说话时，必须冻结 AEC 自适应更新，否则把本地声音当回声消掉。

> **RFC 出处**：ITU-T G.167（1993）"Acoustic Echo Controllers" 是最早的 AEC 标准；G.168（2015 更新）"Digital Network Echo Cancellers" 定义了测试方法，WebRTC AEC 实现需满足 G.168 测试用例。

### 1.3 自适应滤波器理论基础

AEC 的核心是自适应滤波，用 **NLMS（Normalized Least Mean Squares）** 算法追踪回声路径：

设：
- `x[n]`：参考信号（远端到扬声器的 PCM）
- `d[n]`：麦克风信号（含近端语音 + 回声）
- `ŷ[n] = wᵀ · x[n]`：估计的回声（w 是自适应滤波器权重向量）
- `e[n] = d[n] - ŷ[n]`：误差信号（AEC 输出，即去回声后的信号）

NLMS 权重更新规则：

```
w[n+1] = w[n] + μ · e[n] · x[n] / (||x[n]||² + ε)
```

- `μ`：步长（step size），控制收敛速度 vs 稳定性 trade-off。
- `ε`：正则化项，防止除零。
- 归一化（除以 `||x[n]||²`）使步长对信号能量自适应，比 LMS 更稳定。

> **论文出处**：Widrow & Stearns "Adaptive Signal Processing" (1985)，第 6 章；Haykin "Adaptive Filter Theory" (4th ed. 2002) 第 9 章。

在 libwebrtc 实际实现中，AEC 在 **频域（FDAF，Frequency-Domain Adaptive Filter）** 做，原因：
- 频域卷积 = 时域相关，用 FFT 把 O(N²) 降到 O(N log N)。
- 允许对不同频段设不同的收敛速度（低频回声路径比高频更稳定）。

---

## 2 AEC 深度实现

### 2.1 libwebrtc AEC3 架构

libwebrtc 中的 AEC3（AEC version 3，2017 年引入）由 Per Åhgren（Google）设计，相比旧版 AECM 有三个改进：

1. **Block-based processing**：以 64 sample（1.33 ms @ 48kHz）为 block，而非以帧为单位，降低内部延迟。
2. **Delay estimation**：自动估计扬声器到麦克风的系统延迟（含声卡 buffer + 传播时间），不再需要手动配置。
3. **Non-linear suppressor**：在线性 AEC 后接一个基于机器学习分类的非线性抑制器，处理扬声器失真。

> **源码路径**（待核，libwebrtc 太大未直接 fetch）：`modules/audio_processing/aec3/` — 包含 `echo_canceller3.cc`、`adaptive_fir_filter.cc`、`delay_estimator.cc` 等。

### 2.2 浏览器暴露的 AEC 统计量

WebRTC Stats API（`RTCStatsReport`）暴露的 AEC 相关字段：

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `echoReturnLoss` | dB | 回声抑制量；越大越好，>25 dB 表示 AEC 工作正常 |
| `echoReturnLossEnhancement` | dB | 在 `echoReturnLoss` 基础上的额外增强量 |
| `totalAudioEnergy` | 无单位 | 累积音频能量，用于检测静音检测（VAD）是否正常 |
| `removedSamplesForAcceleration` | count | 用于 playout 加速而丢弃的样本数（jitter buffer 追赶） |
| `insertedSamplesForDeceleration` | count | 用于 playout 减速插入的样本数（jitter buffer 扩展） |

在 `chrome://webrtc-internals` 中选择对应的 `RTCInboundRtpStreamStats` 或 `RTCOutboundRtpStreamStats`，可实时看到上述数据的折线图。

---

## 3 NS（Noise Suppression）深度实现

### 3.1 算法：维纳滤波 + 最小值追踪

NS 的目标：给定带噪信号 `Y(k) = S(k) + N(k)`（k 为频率 bin），估计干净信号 `Ŝ(k)`。

最优维纳滤波器权重：

```
H(k) = SNR(k) / (1 + SNR(k))
```

其中 `SNR(k) = |S(k)|² / |N(k)|²`（先验 SNR）。

问题变成：**如何估计 `|N(k)|²`**（噪声功率谱）？

libwebrtc NS 使用 **Minimum Statistics** 方法（Martin 2001）：

- 在滑动窗口（约 1.5 秒）内跟踪每个频率 bin 的功率最小值。
- 平稳噪声的能量会周期性成为最小值（语音间隙期间）。
- 最小值乘以一个偏置因子（bias factor）得到噪声功率估计。

偏置因子补偿统计偏差：真实最小值在理论上应低于均值，bias factor 把最小值「抬」到正确的均值估计。

> **论文出处**：R. Martin, "Noise power spectral density estimation based on optimal smoothing and minimum statistics", IEEE Trans. Speech Audio Process., 2001。

### 3.2 NS 与语音检测（VAD）的关系

NS 内部维护一个简单的 **VAD（Voice Activity Detection）** 状态，用于：
- 语音段：冻结噪声估计器（不更新最小值），避免把语音误判为噪声。
- 静音段：更新噪声估计器。

WebRTC APM 内的 VAD 基于对数能量 + GMM（Gaussian Mixture Model）分类，不依赖外部 VAD 结果。

---

## 4 AGC（Automatic Gain Control）深度实现

### 4.1 两种 AGC 模式

WebRTC APM 提供两种 AGC 实现，通过 `AudioProcessing::Config` 的 `gain_controller1` / `gain_controller2` 字段选择（待核，libwebrtc API 可能在 2023 后变更）：

#### AGC1（经典模式）

- **kFixedDigital**：固定数字增益，在 PCM 层直接乘以增益系数。
- **kAdaptiveDigital**：自适应模式，基于 RMS 电平测量，通过 I 控制器（积分控制）缓慢调整增益，目标 RMS 电平约为 -18 dBFS。

#### AGC2（2019 引入，默认）

架构分三层：
1. **Input Volume Controller**：控制系统麦克风硬件增益（OS 级别，Android/iOS 不同），避免用软件增益放大硬件噪底。
2. **Adaptive Digital**：软件层自适应增益，基于语音活动度加权的 RMS。
3. **Limiter**：峰值限幅器，防止增益过大导致 clipping（截幅失真）。

> **源码路径**（待核）：`modules/audio_processing/agc2/` — 包含 `adaptive_digital_gain_controller.cc`、`limiter.cc`。

### 4.2 AGC 与回声的交互陷阱

**经典坑**：AGC 会把回声残余放大。场景：
1. 说话人音量较小 → AGC 提高增益。
2. 扬声器音量也被相应放大 → 回声增大。
3. AEC 尚未追上新的回声水平 → 更多残余回声。
4. AGC 再次提高增益 → 正反馈环路。

解决方案：AEC 需要在 AGC 之前感知增益变化，通过 `AudioProcessing::SetRuntimeSetting(GainControlCompressionGain)` 通知 AEC 当前增益量。

---

## 5 Jitter Buffer 深度实现

### 5.1 设计动机

网络包到达时间的方差就是 **抖动（jitter）**。对于 Opus 编码的音频帧（每帧 20 ms），若包到达时间抖动 ±30 ms，直接 playout 会导致音频断裂。

Jitter Buffer 用「以延迟换平滑」的策略：
- 入站包放入缓冲区，等待固定时间窗口。
- 时间窗口结束后按序号顺序取出 playout。
- 代价：引入 `jitterBufferDelay`（典型 20-60 ms）。

### 5.2 Pion JitterBuffer 真实源码精读

【真实源码 pion/interceptor@pkg/jitterbuffer/jitter_buffer.go】

```go
// JitterBuffer 结构体：核心状态
type JitterBuffer struct {
    packets       *PriorityQueue  // 优先级队列，按 seq 排序
    minStartCount uint16          // 开始 playout 前必须积累的最小包数（默认 50）
    overflowLen   uint16          // 队列溢出阈值（默认 100）
    lastSequence  uint16          // 最近收到包的 seq
    playoutHead   uint16          // 下一个要取出的 seq
    playoutReady  bool            // 是否已经完成初始缓冲
    state         State           // Buffering 或 Emitting
    stats         Stats           // outOfOrder/underflow/overflow 计数
    listeners     map[Event][]EventListener
    mutex         sync.Mutex
}

// Push：入站包处理
func (jb *JitterBuffer) Push(packet *rtp.Packet) {
    jb.mutex.Lock()
    defer jb.mutex.Unlock()

    if jb.packets.Length() == 0 {
        jb.emit(StartBuffering)  // 首包到达，发出事件
    }

    if jb.packets.Length() > jb.overflowLen {
        jb.stats.overflowCount++
        jb.emit(BufferOverflow)  // 队列满，通知调用者
    }

    // 首包锚定 playoutHead（只在 !playoutReady 且队列为空时）
    if !jb.playoutReady && jb.packets.Length() == 0 {
        jb.playoutHead = packet.SequenceNumber
    }

    jb.updateStats(packet.SequenceNumber)  // 更新乱序统计
    jb.packets.Push(packet, packet.SequenceNumber)  // 插入优先级队列
    jb.updateState()  // 检查是否可以切换到 Emitting
}

// Pop：取出下一帧（必须在 Emitting 状态）
func (jb *JitterBuffer) Pop() (*rtp.Packet, error) {
    jb.mutex.Lock()
    defer jb.mutex.Unlock()
    if jb.state != Emitting {
        return nil, ErrPopWhileBuffering  // 还在 Buffering，拒绝取包
    }
    packet, err := jb.packets.PopAt(jb.playoutHead)
    if err != nil {
        jb.stats.underflowCount++
        jb.emit(BufferUnderflow)  // 该 seq 包丢失，调用者需做 PLC
        return nil, err
    }
    jb.playoutHead = (jb.playoutHead + 1)  // seq 回绕安全（uint16）
    jb.updateState()
    return packet, nil
}

// updateState：状态机转换（Buffering → Emitting）
func (jb *JitterBuffer) updateState() {
    if jb.packets.Length() >= jb.minStartCount && jb.state == Buffering {
        jb.state = Emitting
        jb.playoutReady = true
        jb.emit(BeginPlayback)  // 通知：可以开始 playout 了
    }
}
```

### 5.3 关键设计决策分析

**minStartCount = 50 的含义**

默认值 50 包 × 20 ms/包 = **1000 ms（1秒）** 的初始缓冲。这对音频来说太大，实际生产场景通常设为 2-5（40-100 ms）。Pion 这里的默认值更适合 VoD 场景而非实时通话。

**seq 回绕处理**

`uint16` 类型自动处理 65535→0 的回绕，`playoutHead = (playoutHead + 1)` 无需显式 mod 操作。PriorityQueue 的比较函数也需要正确处理回绕（RFC 3550 §A.1 定义了序列号比较算法）。

**BufferUnderflow 的含义**

Underflow 意味着期望 seq 的包没来（丢失或严重延迟）。调用者需要：
1. **PLC（Packet Loss Concealment）**：Opus 解码器内置 PLC，只需传入 NULL 帧，解码器自动插值。
2. 跳过该 seq，继续取下一包，避免永久阻塞。

### 5.4 WebRTC NetEQ（实际浏览器实现）

浏览器内实际的 jitter buffer 是 **NetEQ**（Network Equalizer），比 Pion 的实现复杂得多：

- **自适应延迟**：NetEQ 根据网络状态动态调整 buffer size，而非固定 minStartCount。
- **DTMF 处理**：识别并正确处理 RFC 4733 带内 DTMF。
- **加速/减速**：当 buffer 过满时用 WSOLA（Waveform-Similarity Overlap-and-Add）加速；buffer 欠满时插入静音或重复帧。

> **源码路径**（待核）：`modules/audio_coding/neteq/` — 包含 `neteq_impl.cc`、`decision_logic.cc`（playout 策略决策）。

---

## 6 可运行 Demo

### Demo 1：浏览器 JS — 观测 AEC/NS/AGC 前后效果（webrtc-internals）

本 demo 建立本地回环 PeerConnection，开启麦克风，通过 `webrtc-internals` 实时观测音频处理统计量。

**完整代码（index.html，单文件，无需服务器）**

```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>WebRTC APM 观测 Demo</title>
  <style>
    body { font-family: monospace; padding: 20px; background: #1a1a1a; color: #e0e0e0; }
    button { margin: 5px; padding: 10px 20px; cursor: pointer; font-size: 14px; }
    #stats { white-space: pre; background: #2a2a2a; padding: 15px; margin-top: 15px; border-radius: 4px; }
    .label { color: #7ec8e3; }
    .value { color: #90ee90; }
  </style>
</head>
<body>
  <h2>WebRTC APM 统计观测</h2>
  <p>开启后在 <code>chrome://webrtc-internals</code> 观察 echoReturnLoss、audioLevel 等指标</p>
  
  <button id="btnStart">开始（开麦克风）</button>
  <button id="btnStop" disabled>停止</button>
  <button id="btnToggleAEC" disabled>关闭 AEC（对比）</button>
  
  <div id="stats">等待启动...</div>

  <script>
  'use strict';

  let pc1, pc2, localStream, statsInterval;
  let aecEnabled = true;

  const $ = id => document.getElementById(id);

  // 获取麦克风，支持 AEC/NS/AGC 开关
  async function getMic(echoCancellation, noiseSuppression, autoGainControl) {
    return navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation,    // AEC 开关
        noiseSuppression,    // NS 开关
        autoGainControl,     // AGC 开关
        sampleRate: 48000,   // 强制 48kHz（Opus 标准采样率）
        channelCount: 1,
      },
      video: false,
    });
  }

  async function start() {
    // 开启全部 APM 处理
    localStream = await getMic(true, true, true);
    
    const config = { iceServers: [] };  // 本地回环，不需要 STUN/TURN
    pc1 = new RTCPeerConnection(config);
    pc2 = new RTCPeerConnection(config);

    // 本地 ICE candidate 直接给对端
    pc1.onicecandidate = e => e.candidate && pc2.addIceCandidate(e.candidate);
    pc2.onicecandidate = e => e.candidate && pc1.addIceCandidate(e.candidate);

    // pc2 收到远端音频，接入 AudioContext 分析
    pc2.ontrack = e => {
      const audioCtx = new AudioContext();
      const analyser = audioCtx.createAnalyser();
      const source = audioCtx.createMediaStreamSource(e.streams[0]);
      source.connect(analyser);
      // 不接 destination，避免本地播放产生新的回声
    };

    // 把麦克风 track 加入 pc1（发送端）
    localStream.getTracks().forEach(t => pc1.addTrack(t, localStream));

    // SDP offer/answer
    const offer = await pc1.createOffer();
    await pc1.setLocalDescription(offer);
    await pc2.setRemoteDescription(offer);
    const answer = await pc2.createAnswer();
    await pc2.setLocalDescription(answer);
    await pc1.setRemoteDescription(answer);

    // 定时轮询 RTCStatsReport
    statsInterval = setInterval(collectStats, 1000);

    $('btnStart').disabled = true;
    $('btnStop').disabled = false;
    $('btnToggleAEC').disabled = false;
  }

  async function collectStats() {
    if (!pc1) return;
    const reports = await pc1.getStats();
    const output = [];
    
    reports.forEach(report => {
      if (report.type === 'media-source' && report.kind === 'audio') {
        output.push(`[media-source]`);
        output.push(`  audioLevel:          ${(report.audioLevel || 0).toFixed(4)}`);
        output.push(`  totalAudioEnergy:    ${(report.totalAudioEnergy || 0).toFixed(6)}`);
      }
      if (report.type === 'outbound-rtp' && report.kind === 'audio') {
        output.push(`[outbound-rtp]`);
        output.push(`  packetsSent:         ${report.packetsSent || 0}`);
        output.push(`  bytesSent:           ${report.bytesSent || 0}`);
        output.push(`  targetBitrate:       ${report.targetBitrate || 'N/A'}`);
      }
      if (report.type === 'inbound-rtp' && report.kind === 'audio') {
        output.push(`[inbound-rtp]`);
        output.push(`  jitter:              ${(report.jitter || 0).toFixed(4)} s`);
        output.push(`  packetsLost:         ${report.packetsLost || 0}`);
        output.push(`  jitterBufferDelay:   ${(report.jitterBufferDelay || 0).toFixed(4)} s`);
        output.push(`  removedSamplesForAcceleration: ${report.removedSamplesForAcceleration || 0}`);
        output.push(`  insertedSamplesForDeceleration: ${report.insertedSamplesForDeceleration || 0}`);
        // echoReturnLoss 在 Chrome 74+ 实现
        if (report.echoReturnLoss !== undefined) {
          output.push(`  echoReturnLoss:      ${report.echoReturnLoss.toFixed(2)} dB`);
          output.push(`  echoReturnLossEnhancement: ${(report.echoReturnLossEnhancement || 0).toFixed(2)} dB`);
        }
      }
    });
    
    $('stats').textContent = output.length ? output.join('\n') : '统计收集中...';
  }

  async function stop() {
    clearInterval(statsInterval);
    if (pc1) { pc1.close(); pc1 = null; }
    if (pc2) { pc2.close(); pc2 = null; }
    if (localStream) { localStream.getTracks().forEach(t => t.stop()); }
    $('stats').textContent = '已停止。';
    $('btnStart').disabled = false;
    $('btnStop').disabled = true;
    $('btnToggleAEC').disabled = true;
  }

  // 对比实验：关闭 AEC，重建 PeerConnection（注意需要重新获取麦克风）
  async function toggleAEC() {
    aecEnabled = !aecEnabled;
    clearInterval(statsInterval);
    if (pc1) { pc1.close(); }
    if (pc2) { pc2.close(); }
    if (localStream) { localStream.getTracks().forEach(t => t.stop()); }
    
    // 用新的 constraints 重新获取麦克风
    localStream = await getMic(aecEnabled, aecEnabled, aecEnabled);
    
    const config = { iceServers: [] };
    pc1 = new RTCPeerConnection(config);
    pc2 = new RTCPeerConnection(config);
    pc1.onicecandidate = e => e.candidate && pc2.addIceCandidate(e.candidate);
    pc2.onicecandidate = e => e.candidate && pc1.addIceCandidate(e.candidate);
    pc2.ontrack = () => {};
    localStream.getTracks().forEach(t => pc1.addTrack(t, localStream));
    
    const offer = await pc1.createOffer();
    await pc1.setLocalDescription(offer);
    await pc2.setRemoteDescription(offer);
    const answer = await pc2.createAnswer();
    await pc2.setLocalDescription(answer);
    await pc1.setRemoteDescription(answer);
    
    statsInterval = setInterval(collectStats, 1000);
    $('btnToggleAEC').textContent = aecEnabled ? '关闭 AEC（对比）' : '开启 AEC（恢复）';
    $('stats').textContent = `APM 已切换：AEC=${aecEnabled}, NS=${aecEnabled}, AGC=${aecEnabled}`;
  }

  $('btnStart').onclick = start;
  $('btnStop').onclick = stop;
  $('btnToggleAEC').onclick = toggleAEC;
  </script>
</body>
</html>
```

**运行步骤**

1. 将代码保存为 `apm-demo.html`。
2. 用 Chrome 打开（`file://` 协议即可，也可用 `python3 -m http.server 8080` 再访问 `localhost:8080/apm-demo.html`）。
3. 点击「开始」，允许麦克风权限。
4. 新标签页打开 `chrome://webrtc-internals`，展开 `RTCPeerConnection` 条目。
5. 在 `Stats graphs` 中找到 `echoReturnLoss`、`audioLevel`、`jitterBufferDelay` 图表。
6. 点击「关闭 AEC（对比）」，观察 `echoReturnLoss` 数值变化（若有扬声器，应明显下降或消失）。

**预期观测**

- AEC 开启时：`echoReturnLoss` > 20 dB（Chrome 正常范围 25-45 dB）。
- AEC 关闭时：`echoReturnLoss` 消失或为 0（该字段仅在 AEC 启用时有意义）。
- `jitterBufferDelay`：本地回环通常 < 5 ms，真实网络场景下 20-80 ms。
- `jitter`：本地回环接近 0，真实 4G 网络可达 30-80 ms。

---

### Demo 2：Pion Go — Jitter Buffer 行为观测

本 demo 用 Pion 的 JitterBuffer 模拟乱序包到达场景，打印 underflow/overflow 事件，量化 playout delay。

**前置条件**

```bash
go version  # 需要 Go 1.21+
mkdir pion-jitter-demo && cd pion-jitter-demo
go mod init jitter-demo
go get github.com/pion/interceptor@latest
go get github.com/pion/rtp@latest
```

**完整代码（main.go）**

```go
package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/pion/interceptor/pkg/jitterbuffer"
	"github.com/pion/rtp"
)

func main() {
	// 创建 JitterBuffer，minStartCount=5（100 ms），overflowLen=20
	jb := jitterbuffer.New(
		jitterbuffer.WithMinimumPacketCount(5),
		jitterbuffer.WithPacketOverflowSize(20),
	)

	// 注册事件监听
	underflowCount := 0
	overflowCount := 0
	
	jb.Listen(jitterbuffer.BufferUnderflow, func(event jitterbuffer.Event, _ *jitterbuffer.JitterBuffer) {
		underflowCount++
		fmt.Printf("  [事件] BufferUnderflow #%d\n", underflowCount)
	})
	jb.Listen(jitterbuffer.BufferOverflow, func(event jitterbuffer.Event, _ *jitterbuffer.JitterBuffer) {
		overflowCount++
		fmt.Printf("  [事件] BufferOverflow #%d\n", overflowCount)
	})
	jb.Listen(jitterbuffer.BeginPlayback, func(event jitterbuffer.Event, _ *jitterbuffer.JitterBuffer) {
		fmt.Println("  [事件] BeginPlayback — 缓冲区已就绪，开始 playout")
	})

	fmt.Println("=== 场景 1：正常顺序到达 ===")
	for seq := uint16(1); seq <= 8; seq++ {
		pkt := makePacket(seq, uint32(seq)*160)
		jb.Push(pkt)
		fmt.Printf("Push seq=%d\n", seq)
	}
	drainBuffer(jb, "场景 1")

	// 重置（创建新 buffer）
	jb = jitterbuffer.New(
		jitterbuffer.WithMinimumPacketCount(5),
		jitterbuffer.WithPacketOverflowSize(20),
	)
	jb.Listen(jitterbuffer.BufferUnderflow, func(event jitterbuffer.Event, _ *jitterbuffer.JitterBuffer) {
		fmt.Println("  [事件] BufferUnderflow（丢包场景）")
	})
	jb.Listen(jitterbuffer.BeginPlayback, func(event jitterbuffer.Event, _ *jitterbuffer.JitterBuffer) {
		fmt.Println("  [事件] BeginPlayback")
	})

	fmt.Println("\n=== 场景 2：乱序到达（模拟网络重排序）===")
	// 准备 8 个包，随机打乱顺序
	seqs := []uint16{1, 2, 3, 4, 5, 6, 7, 8}
	rand.Shuffle(len(seqs), func(i, j int) { seqs[i], seqs[j] = seqs[j], seqs[i] })
	
	for _, seq := range seqs {
		pkt := makePacket(seq, uint32(seq)*160)
		fmt.Printf("Push seq=%d（乱序）\n", seq)
		jb.Push(pkt)
	}
	drainBuffer(jb, "场景 2")

	// 场景 3：模拟丢包
	jb = jitterbuffer.New(
		jitterbuffer.WithMinimumPacketCount(5),
		jitterbuffer.WithPacketOverflowSize(20),
	)
	jb.Listen(jitterbuffer.BufferUnderflow, func(event jitterbuffer.Event, _ *jitterbuffer.JitterBuffer) {
		fmt.Println("  [事件] BufferUnderflow — 需要 PLC 填补此帧")
	})
	jb.Listen(jitterbuffer.BeginPlayback, func(event jitterbuffer.Event, _ *jitterbuffer.JitterBuffer) {
		fmt.Println("  [事件] BeginPlayback")
	})

	fmt.Println("\n=== 场景 3：丢包（seq 3 缺失）===")
	for _, seq := range []uint16{1, 2, 4, 5, 6} { // 故意跳过 seq=3
		jb.Push(makePacket(seq, uint32(seq)*160))
		fmt.Printf("Push seq=%d\n", seq)
	}
	drainBuffer(jb, "场景 3")
}

// makePacket 构造一个最小 RTP 包
func makePacket(seq uint16, ts uint32) *rtp.Packet {
	return &rtp.Packet{
		Header: rtp.Header{
			Version:        2,
			PayloadType:    111, // Opus payload type
			SequenceNumber: seq,
			Timestamp:      ts,
			SSRC:           0xDEADBEEF,
		},
		Payload: []byte{0x00, 0x01}, // 最小 Opus payload（实际会更长）
	}
}

// drainBuffer 尝试从 buffer 取出所有包
func drainBuffer(jb *jitterbuffer.JitterBuffer, label string) {
	fmt.Printf("--- 开始 playout（%s）---\n", label)
	start := time.Now()
	popped := 0
	for i := 0; i < 10; i++ {
		pkt, err := jb.Pop()
		if err != nil {
			fmt.Printf("  Pop err: %v（可能 underflow 或 buffering）\n", err)
			// 模拟 PLC：跳过此帧，继续取下一帧
			jb.SetPlayoutHead(jb.PlayoutHead() + 1)
			continue
		}
		popped++
		fmt.Printf("  Pop seq=%d ts=%d\n", pkt.Header.SequenceNumber, pkt.Header.Timestamp)
	}
	elapsed := time.Since(start)
	fmt.Printf("--- 取出 %d 包，耗时 %v ---\n\n", popped, elapsed)
}
```

**运行步骤**

```bash
go run main.go
```

**预期输出（示例）**

```
=== 场景 1：正常顺序到达 ===
Push seq=1
...
Push seq=5
  [事件] BeginPlayback — 缓冲区已就绪，开始 playout
Push seq=6
...
--- 开始 playout（场景 1）---
  Pop seq=1 ts=160
  Pop seq=2 ts=320
  ...
  Pop seq=8 ts=1280
--- 取出 8 包，耗时 ...

=== 场景 2：乱序到达（模拟网络重排序）===
Push seq=5（乱序）
Push seq=2（乱序）
...
  [事件] BeginPlayback
--- 开始 playout（场景 2）---
  Pop seq=1 ts=160   ← 不管推入顺序，取出顺序始终按 seq 排列
  Pop seq=2 ts=320
  ...

=== 场景 3：丢包（seq 3 缺失）===
...
  [事件] BeginPlayback
--- 开始 playout（场景 3）---
  Pop seq=1 ts=160
  Pop seq=2 ts=320
  [事件] BufferUnderflow — 需要 PLC 填补此帧  ← seq=3 丢失
  Pop err: ...（underflow）
  Pop seq=4 ts=640   ← SetPlayoutHead 跳到 4，继续取
  ...
```

---

### Demo 3：浏览器 JS — getUserMedia constraints 对比实验

用不同的 audio constraints 获取麦克风流，用 Web Audio API 实时显示频谱，直观对比 NS 效果。

```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>NS 频谱对比 Demo</title>
  <style>
    body { background: #111; color: #eee; font-family: monospace; padding: 20px; }
    canvas { display: block; margin: 10px 0; border: 1px solid #333; }
    button { margin: 5px; padding: 8px 16px; }
    .row { display: flex; gap: 20px; flex-wrap: wrap; }
    .col { flex: 1; min-width: 300px; }
  </style>
</head>
<body>
  <h2>NS 频谱对比：开 vs 关</h2>
  <p>点击两个按钮后，在嘈杂环境（或播放背景音乐）中说话，观察频谱差异</p>
  <div class="row">
    <div class="col">
      <h3>NS 开启</h3>
      <button onclick="startStream('ns-on', true)">开启麦克风（NS=on）</button>
      <canvas id="canvas-ns-on" width="400" height="150"></canvas>
    </div>
    <div class="col">
      <h3>NS 关闭</h3>
      <button onclick="startStream('ns-off', false)">开启麦克风（NS=off）</button>
      <canvas id="canvas-ns-off" width="400" height="150"></canvas>
    </div>
  </div>

  <script>
  'use strict';
  const streams = {};

  async function startStream(id, noiseSuppressionEnabled) {
    if (streams[id]) {
      streams[id].forEach(t => t.stop());
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        noiseSuppression: noiseSuppressionEnabled,
        echoCancellation: false,   // 关闭 AEC 以单独观察 NS 效果
        autoGainControl: false,    // 关闭 AGC 保持电平一致
      },
      video: false,
    });
    streams[id] = stream.getTracks();
    
    const audioCtx = new AudioContext({ sampleRate: 48000 });
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;
    source.connect(analyser);
    
    const canvas = document.getElementById(`canvas-${id}`);
    const ctx = canvas.getContext('2d');
    const bufLen = analyser.frequencyBinCount;
    const dataArr = new Uint8Array(bufLen);
    
    function draw() {
      requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArr);
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      const barW = (canvas.width / bufLen) * 2.5;
      let x = 0;
      for (let i = 0; i < bufLen; i++) {
        const barH = (dataArr[i] / 255) * canvas.height;
        // 低频（<500Hz）用绿色，高频用蓝色
        const hue = (i / bufLen) * 240;
        ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
        ctx.fillRect(x, canvas.height - barH, barW, barH);
        x += barW + 1;
      }
    }
    draw();
  }
  </script>
</body>
</html>
```

**预期观测**

- NS=on 时：背景噪声（风扇声、空调声等平稳噪声）对应的低频频谱柱显著矮化，说话间隙的频谱趋向静默。
- NS=off 时：背景噪声持续显示，频谱基底明显更高。
- NS 对非平稳噪声（键盘声、突发声）效果较弱，可以在背景音乐场景下观察到 NS 的局限性。

---

## 7 方案对比：APM 处理选项

### 7.1 浏览器端 vs 服务端（SFU）处理

| 维度 | 浏览器端 APM（WebRTC 默认） | SFU 端处理（如 mediasoup / Janus） |
|------|---------------------------|-----------------------------------|
| **延迟** | 本地处理，延迟最低（APM 本身 < 5 ms） | 需要将音频解码后处理，增加 20-50 ms |
| **参考信号获取** | AEC 参考信号（扬声器 PCM）直接在系统层获取 | SFU 无法获取用户本地扬声器信号，**无法做 AEC** |
| **计算资源** | 消耗用户设备 CPU（移动端约 3-8%）| 消耗服务器 CPU，用户设备无负担 |
| **效果可控性** | 受浏览器实现限制，无法调参 | 可使用 libwebrtc APM 精细调参 |
| **多路混音** | 不适用 | SFU 在混音后才能做全局 NS/AGC |
| **适用场景** | 所有 WebRTC 通话（默认路径） | 录制、转码、多路汇聚质量控制 |

**关键结论**：AEC 必须在发送端（靠近扬声器的那一侧）做，SFU 无法替代。NS/AGC 可以在任一侧叠加，但双重处理（端侧 + 服务端）可能引入过度处理伪影（musical noise）。

### 7.2 AEC 实现对比

| 实现 | 算法 | 特点 | 适用 |
|------|------|------|------|
| WebRTC AEC3 | FDAF NLMS + 非线性抑制 | 工业级，延迟估计自动 | 所有浏览器 WebRTC |
| Speex AEC | 基于频域 LMS | 开源，可调参，质量略低 | 嵌入式、低功耗设备 |
| RNNoise | 递归神经网络 | 对非平稳噪声效果好 | NS 而非 AEC |
| Krisp | 商业 DNN | 效果最好，需授权 | 企业级会议软件 |
| Apple AVAudioSession | 系统级 | 黑盒，iOS 效果好 | 仅 iOS/macOS |

### 7.3 getUserMedia constraints 对比

```javascript
// 高质量通话（默认推荐）
const highQuality = {
  audio: {
    echoCancellation: true,    // AEC
    noiseSuppression: true,    // NS
    autoGainControl: true,     // AGC
    sampleRate: 48000,
    channelCount: 1,
  }
};

// 音乐场景（关闭处理，保留音质）
const musicMode = {
  audio: {
    echoCancellation: false,   // 音乐需要完整频响，AEC 会破坏低频
    noiseSuppression: false,   // NS 会产生 musical noise
    autoGainControl: false,    // 动态音乐不需要 AGC 压缩
    sampleRate: 48000,
    channelCount: 2,           // 立体声
  }
};

// 录音场景（半双工，无需 AEC）
const recordingMode = {
  audio: {
    echoCancellation: false,   // 录音时不需要 AEC（无远端播放）
    noiseSuppression: true,    // 保留 NS
    autoGainControl: true,     // 录音 AGC 有用
  }
};
```

**注意**：Chrome 实现中，`echoCancellation: false` 会同时禁用 AEC 和部分系统级回声处理；在 macOS 上 Chrome 会忽略部分 constraints（因为 Core Audio 在系统层也做 AEC）。

---

## 8 失败模式与真实坑

### 8.1 AEC 失效场景

**坑 1：播放延迟动态变化导致 AEC 失散**

问题：音视频同步调整（视频 A/V sync）会改变扬声器实际播放时刻，导致 AEC 参考信号时间戳与麦克风信号时间戳不对齐，AEC 重新收敛期间（约 0.5-2 秒）有明显回声。

根因：WebRTC AEC3 的延迟估计器假设系统延迟在短时间内稳定。视频 A/V sync 调整是一个脉冲式延迟变化，超过延迟估计器的追踪能力。

解法：视频 A/V sync 调整尽量平滑（每次调整 < 10 ms），或通知 APM 系统延迟变化（`AudioProcessing::SetRuntimeSetting(PlayoutDelayMs)`）。

**坑 2：音频设备切换后 AEC 失效**

问题：用户插拔耳机时，扬声器设备切换，但 WebRTC 没有检测到设备变化，继续用旧的参考信号做 AEC。

根因：Chrome < 90 的 bug，设备切换后 AEC 参考链路未重建。

解法：监听 `navigator.mediaDevices.ondevicechange`，设备变化时重新 `getUserMedia`。

**坑 3：双讲检测误触发**

问题：本地说话声音过大时，AEC 误认为「本地信号太强，可能是回声」，触发双讲检测冻结，导致本地声音在远端「断字」。

根因：AEC 的双讲检测（Double-talk Detection）基于「若近端能量突然超过参考信号估计，则认为是双讲」，高能量语音容易触发。

解法：调低扬声器音量，或在 AGC 目标电平已稳定时避免突然大声。

### 8.2 NS 失效与 Musical Noise

**坑：Musical Noise（音乐噪声）**

问题：NS 在频域对每个 bin 独立抑制，当噪声估计不准确时，某些 bin 的信号被过度抑制、某些被欠抑制，产生像「音乐」一样的断断续续的残余噪声，比原始噪声更让人烦躁。

根因：维纳滤波器的噪声估计滞后于实际噪声变化；最小值追踪在非平稳噪声（如键盘声）下失效。

解法：
- 降低 NS 强度（若 API 允许）。
- 换用基于神经网络的 NS（RNNoise、Krisp），对非平稳噪声更鲁棒。
- 双重 NS（端侧 + 服务端）时，第二道 NS 会加剧 musical noise。

### 8.3 Jitter Buffer 陷阱

**坑 1：minStartCount 过大导致首帧延迟**

问题：Pion 默认 `minStartCount=50`（1 秒），导致通话建立后 1 秒没声音。

根因：初始缓冲期设计为 VoD 场景，对实时通话太保守。

解法：实时通话场景设 `minStartCount=2`（40 ms）。

**坑 2：网络突发丢包导致 underflow 风暴**

问题：网络切换（WiFi → 4G）瞬间丢 10-20 个包，jitter buffer 触发大量 underflow，PLC 连续填充，音质骤降。

根因：PLC 只能插值 1-2 帧（20-40 ms），连续丢包超过 3 帧，音质无法掩盖。

解法：
- 发送端 FEC（Forward Error Correction）：Opus 内置 INBAND_FEC，丢包率 > 10% 时自动开启，将上一帧数据附在下一帧中冗余传输。
- NACK + RTX：可以重传丢失包，但实时通话场景 RTX 延迟（往返 + 处理）> 50 ms 时无效。

**坑 3：seq 回绕比较错误**

问题：seq 从 65535 回绕到 0 时，简单的 `seq_a > seq_b` 比较失效，导致包被误排序。

根因：uint16 算术回绕。

解法：RFC 3550 §A.1 定义了正确的循环比较算法：

```go
// RFC 3550 §A.1 序列号循环比较
func seqGT(a, b uint16) bool {
    const maxSeqDiff = 0x8000  // 32768
    return (a != b) && (((a - b) & 0xFFFF) < maxSeqDiff)
}
```

### 8.4 AGC 陷阱

**坑：AGC 在会议 unmute 瞬间爆音**

问题：用户静音（mute）后 AGC 持续提高增益（因为静音期间信号为 0，AGC 误认为信号太低）。Unmute 后 AGC 以高增益播放第一帧，产生爆音。

根因：AGC 的状态在 mute 期间继续演化。

解法：Mute 时重置 AGC 状态，或在 mute 期间送入低电平白噪声（-60 dBFS）维持 AGC 平衡。Chrome 的 `RTCRtpSender.replaceTrack()` 在 mute 时会自动处理此问题。

---

## 9 章末五件套

### 9.1 核心概念速查

| 概念 | 一句话定义 |
|------|----------|
| APM | libwebrtc 的音频处理管道，串联 AEC→NS→AGC，处理周期 10 ms |
| AEC | 从麦克风信号中减去扬声器参考信号的估计回声，核心算法 NLMS |
| AEC3 | WebRTC 第三代 AEC，频域自适应滤波 + 自动延迟估计 + 非线性抑制 |
| NS | 基于最小值统计的噪声功率估计 + 维纳滤波频域抑制 |
| AGC | 自动调整增益使输出维持目标 RMS 电平（约 -18 dBFS） |
| Jitter Buffer | RTP 包重排序 + 平滑抖动的缓冲区，代价是固定 playout delay |
| NetEQ | Chrome 内置的自适应 jitter buffer + PLC + 速度调整引擎 |
| PLC | 丢包隐藏，Opus 解码器内置，连续丢失 < 3 帧时效果可接受 |
| Double-talk | 本端和远端同时说话，AEC 需检测并冻结自适应更新 |
| Musical noise | NS 过度处理产生的频域残余噪声伪影，比原始噪声更烦人 |

### 9.2 调试工具清单

1. **`chrome://webrtc-internals`**：最重要的工具。观测 `echoReturnLoss`、`jitter`、`jitterBufferDelay`、`packetsLost`、`audioLevel`。
2. **`chrome://media-internals`**：Chrome 音频设备路由、APM 配置状态。
3. **WebAudio API + AnalyserNode**：实时频谱可视化（见 Demo 3）。
4. **Wireshark + RTP 解析**：捕获 RTP 包，分析 seq 不连续、时间戳抖动。
5. **`rtcdump` / `WebRTC-Internals-Exporter` Chrome 扩展**：导出 stats 数据为 CSV/JSON，做长时间分析。

### 9.3 生产经验 Checklist

```
发布前必查：
□ echoCancellation=true (通话场景) 或 false (音乐场景)
□ 设备切换后重建 getUserMedia（监听 devicechange 事件）
□ Jitter buffer minStartCount 已调至实时通话合适值（2-5）
□ Opus FEC 在丢包率 >5% 时已开启（通过 SDP fmtp inband-fec=1）
□ AGC mute/unmute 爆音问题已测试
□ 移动端 4G 切换场景已测试（模拟抖动 50 ms）

监控告警阈值：
□ jitter > 50 ms → 触发 jitter buffer 扩容告警
□ packetsLost > 5% → 触发 FEC/NACK 策略调整
□ echoReturnLoss < 10 dB → 触发 AEC 异常告警
□ jitterBufferDelay > 150 ms → 触发延迟过高告警
```

### 9.4 延伸阅读

- **ITU-T G.168**：数字网络回声消除器测试标准，AEC 实现的事实验收基准。
- **R. Martin (2001)**："Noise power spectral density estimation based on optimal smoothing and minimum statistics" — WebRTC NS 的直接来源论文。
- **Per Åhgren, AEC3 设计文档**（待核）：WebRTC 官方 blog 2016 年有介绍 AEC3 设计动机的文章。
- **RFC 3550 §A.1**：序列号循环比较算法的权威定义。
- **Haykin "Adaptive Filter Theory" (4th ed.)**：NLMS 算法的教科书级推导。
- **Pion interceptor 仓库**：`pkg/jitterbuffer/` — 可读性最高的 jitter buffer 开源实现之一。

### 9.5 本章知识图谱

```
音频采集（麦克风 PCM）
    │
    ▼
┌───────────────────────────────────────────┐
│  WebRTC APM（Audio Processing Module）     │
│                                           │
│  AEC3 ────── 参考信号（扬声器 PCM）         │
│   │  NLMS 频域自适应滤波                   │
│   │  延迟估计 + 非线性抑制                  │
│   ▼                                       │
│  NS ── 最小值统计噪声估计 + 维纳滤波        │
│   │  VAD 保护语音段                        │
│   ▼                                       │
│  AGC ── 目标 RMS 电平控制                  │
│     数字增益 + 峰值限幅器                   │
└───────────────────────────────────────────┘
    │
    ▼
Opus 编码（20 ms 帧，内置 PLC + FEC）
    │
    ▼
RTP 打包 → SRTP 加密 → ICE/UDP → 网络
    │
    ▼（接收端）
RTP 解包 → Jitter Buffer（NetEQ）
            │  自适应延迟 + 重排序 + PLC
            ▼
        Opus 解码 → 扬声器 PCM 输出
```

---

*本章真实源码取自 `pion/interceptor@pkg/jitterbuffer/jitter_buffer.go`（JitterBuffer struct、Push/Pop/updateState 方法）。AEC/NS/AGC libwebrtc 内部实现因源码体积限制未直接 fetch，架构描述基于 WebRTC 官方文档、RFC 及 Per Åhgren 相关公开资料，标注「待核」处建议直接查阅 `chromium.googlesource.com` 对应路径。*
