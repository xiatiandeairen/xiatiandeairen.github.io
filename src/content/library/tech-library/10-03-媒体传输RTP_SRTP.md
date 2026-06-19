---
title: "媒体传输 RTP / SRTP（WebRTC 域）"
slug: "10-03"
collection: "tech-library"
group: "webrtc"
order: 10003
summary: "前置依赖：第 1 章（信令 & SDP）、第 2 章（ICE/STUN/TURN）。本章假设 ICE 已完成 candidate pair 选定，DTLS 握手结束，读者能理解 Big-Endian 位字段和基本密码学概念（AES-CTR、HMAC-SHA1、AES-GCM）。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-15T14:11:51.000Z"
updatedAt: "2026-06-15T14:11:51.000Z"
---
> **前置依赖**：第 1 章（信令 & SDP）、第 2 章（ICE/STUN/TURN）。本章假设 ICE 已完成 candidate pair 选定，DTLS 握手结束，读者能理解 Big-Endian 位字段和基本密码学概念（AES-CTR、HMAC-SHA1、AES-GCM）。

---

## TL;DR

- **RTP** 是媒体数据的载体。12 字节固定 header（V/P/X/CC/M/PT/SeqNum/Timestamp/SSRC）+ 可选扩展 + payload，不保可靠、不保顺序，刻意如此。
- **SRTP** 在 RTP 之上叠加机密性（AES-CTR 或 AES-GCM）与完整性（HMAC-SHA1 或 GCM tag），密钥材料由 DTLS-SRTP 导出，不单独协商。
- **RTCP** 是 RTP 的控制面：SR/RR 报告丢包 / 抖动 / 时钟同步，NACK / PLI / REMB / Transport-CC 是质量控制回路的反馈信号。
- **WebRTC 完整媒体发送路径**：getUserMedia → MediaStreamTrack → RTCRtpSender → RTP packetizer → SRTP encrypt → ICE/DTLS socket → 网络。

---

## 1 设计考古：为什么要有 RTP

### 1.1 背景与动机

1992 年，IETF Schooler/Schulzrinne/Casner/Frederick 在 RFC 1889 中发布 RTP。核心洞见来自 MBone（Multicast Backbone on the Internet）实验：

- **TCP 不适合实时媒体**。TCP 的重传机制保序但引入不可预测延迟；对于音视频，一个 50 ms 到达的包远优于 200 ms 到达的重传包——宁可丢，不要等。
- **裸 UDP 不够用**。媒体同步（audio/video lip sync）需要时间戳；丢包检测需要序列号；多路复用需要 SSRC；这些统一放到 UDP payload 里比反复发明轮子好。

> **出处**：RFC 3550 §1 "Introduction"（2003，H. Schulzrinne 等，取代 RFC 1889），以及 Schulzrinne 1992 "A Transport Protocol for Audio and Video Conferences"。

RFC 3550 的核心设计哲学总结为一句话："**RTP is intended to be tailored through modifications and/or additions to the headers as needed.**" — 这也是为什么扩展机制（X bit + Extension Profile）存在。

### 1.2 SRTP 的由来

2004 年 RFC 3711（Baugher/McGrew/Naslund/Carrara/Norrman）定义 SRTP，动机来自 VoIP 窃听攻击：裸 RTP 在 IP 网络上明文可嗅探。SRTP 的设计约束是：

1. **不改变 RTP 包结构**（只追加 auth tag，在某些模式下重写部分 header）。
2. **密钥管理由外部解决**（SIP 时代用 SDES/MIKEY；WebRTC 时代用 DTLS-SRTP）。
3. **性能可接受**：AES-CTR 可用硬件加速；HMAC-SHA1 截断为 80 bit 够用。

WebRTC 强制要求 SRTP（RFC 8827），密钥由 DTLS 握手后通过 RFC 5764（DTLS-SRTP）导出，**没有例外**。

---

## 2 RTP Header 逐位精读

### 2.1 固定 Header 结构（12 字节）

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|X|  CC   |M|     PT      |       sequence number         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           timestamp                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           synchronization source (SSRC) identifier           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            contributing source (CSRC) identifiers            |
|                             ....                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

字段解析：

| 字段 | 宽度 | 含义 |
|------|------|------|
| V | 2 bit | 版本号，固定为 2（RFC 3550 起） |
| P | 1 bit | Padding 标志，payload 末尾有填充字节，末字节记填充长度 |
| X | 1 bit | Extension 标志，固定 header 后跟 Extension header |
| CC | 4 bit | CSRC count，0–15 个 Contributing Source SSRC（混流场景） |
| M | 1 bit | Marker bit，语义由 PT 定义；视频通常在帧尾包置 1 |
| PT | 7 bit | Payload Type，SDP 协商的动态编号（96–127 用于动态 codec） |
| SeqNum | 16 bit | 序列号，每包 +1，wraps at 65535→0；用于丢包检测 |
| Timestamp | 32 bit | 采样时钟时间戳；Opus 48 kHz 每包 960 sample → +960 |
| SSRC | 32 bit | 同步源 ID，随机选取，全局唯一 |
| CSRC | 0–60 B | CC 个 32-bit 贡献源 ID（MCU 混流时携带） |

### 2.2 Pion 源码解析 Header.Unmarshal

【真实源码 pion/rtp@master/packet.go】

```go
// Header 结构体字段与 RFC 3550 完全对应
type Header struct {
    Version          uint8    // 2 bit，RFC 强制 = 2
    Padding          bool
    Extension        bool
    Marker           bool
    PayloadType      uint8    // 动态 PT，96-127
    SequenceNumber   uint16   // 丢包检测关键字段
    Timestamp        uint32   // 媒体时钟，不是 wall clock
    SSRC             uint32   // 随机选取，实践中由浏览器/SDK 生成
    CSRC             []uint32
    ExtensionProfile uint16   // 0xBEDE = one-byte ext (RFC 8285)
    Extensions       []Extension
    PaddingSize      byte
    PayloadOffset    int
}
```

Unmarshal 的核心 bit 操作（从 raw bytes 解析第一个字节）：

```go
// 对应 RFC 3550 Figure 1 第一个 octet
// buf[0] layout: [V V P X C C C C]
func (h *Header) Unmarshal(buf []byte) (n int, err error) {
    // V (version): 高 2 bit，shift 6
    h.Version = buf[0] >> versionShift & versionMask   // >> 6 & 0x03
    // P (padding): bit 5
    h.Padding = (buf[0] >> paddingShift & paddingMask) > 0   // >> 5 & 0x01
    // X (extension): bit 4
    h.Extension = (buf[0] >> extensionShift & extensionMask) > 0   // >> 4 & 0x01
    // CC (csrc count): 低 4 bit
    csrcCount := int(buf[0] & ccMask)   // & 0x0F

    // buf[1] layout: [M P P P P P P P]
    h.Marker = (buf[1] >> markerShift & markerMask) > 0   // >> 7 & 0x01
    h.PayloadType = buf[1] & ptMask   // & 0x7F

    // buf[2:4] = SequenceNumber (Big-Endian uint16)
    h.SequenceNumber = binary.BigEndian.Uint16(buf[2:])
    // buf[4:8] = Timestamp
    h.Timestamp = binary.BigEndian.Uint32(buf[4:])
    // buf[8:12] = SSRC
    h.SSRC = binary.BigEndian.Uint32(buf[8:])
    // ... CSRC 列表、Extension 解析
}
```

**关键洞见**：Timestamp 是**媒体采样时钟**，不是 Unix 时间。Opus 在 48 kHz 下每帧 960 samples，所以连续包的 Timestamp 差值固定为 960。不同 codec 有不同时钟速率（G.711 = 8000 Hz，VP8 = 90000 Hz）。这是 A/V 同步的核心。

### 2.3 Extension Header（RFC 8285）

```
 0                   1                   2                   3
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  0xBE |  0xDE |            length                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  ID   |  len  | data                                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Profile `0xBEDE` = One-Byte Header Extension（RFC 8285），每个 extension 1 字节 ID+length。WebRTC 广泛用于：
- **Transport-wide Sequence Number**（twcc ext id）：用于带宽估计
- **Audio Level Indication**（RFC 6464）：VAD 静音检测
- **Abs-Send-Time**：绝对发送时间，用于 GCC 拥塞控制
- **MID**（RFC 8843）：与 SDP `a=mid:` 关联，标识 m-line

```go
// pion/rtp 中的 Extension 类型
type Extension struct {
    id      uint8
    payload []byte
}
// ExtensionProfileOneByte = 0xBEDE  (RFC 8285)
// ExtensionProfileTwoByte = 0x1000  (RFC 8285 §4.3)
// CryptexProfileOneByte   = 0xC0DE  (RFC 9335 — header encryption)
```

---

## 3 SRTP：加密层精读

### 3.1 SRTP 包结构与加密范围

```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         RTP Header                            |  ← 明文（默认）
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         RTP Payload                           |  ← 加密
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Authentication Tag                        |  ← HMAC-SHA1 截断 80bit
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**AEAD_AES_128_GCM** 模式（RFC 7714，WebRTC 推荐）则合并加密+认证：

```
RTP Header（明文） | Encrypted Payload | 16-byte GCM Tag
```

### 3.2 密钥派生链路：DTLS → SRTP

DTLS 握手完成后，双方调用 `TLS-Exporter`（RFC 5705）派生密钥材料：

```
DTLS-SRTP Key Material = PRF(master_secret, "EXTRACTOR-dtls_srtp", ...)
                                        ↓ split 按 Protection Profile
  ClientWriteMasterKey  | ServerWriteMasterKey  | ClientWriteMasterSalt | ServerWriteMasterSalt
```

Pion 中对应 `SessionKeys`：

```go
// 【真实源码 pion/srtp@master/session.go】
type SessionKeys struct {
    LocalMasterKey   []byte  // 己方加密用 master key (16/32 byte)
    LocalMasterSalt  []byte  // 己方 salt (14 byte)
    RemoteMasterKey  []byte  // 对方 master key
    RemoteMasterSalt []byte  // 对方 salt
}
```

### 3.3 SRTP Index 与 ROC（Rollover Counter）

SRTP 用一个 48-bit Index 来保证每包密钥流唯一：

```
SRTP Index = (ROC << 16) | SequenceNumber
```

- SequenceNumber 是 16 bit，回绕 65535→0 时 ROC +1
- ROC 不在包里传输，双端独立维护，通过 "V" 算法（RFC 3711 §3.3.1）同步

【真实源码 pion/srtp@master/srtp.go】

```go
// decryptRTP 内部核心逻辑（简化展示关键步骤）
func (c *Context) decryptRTP(dst, encrypted []byte, header *rtp.Header, ...) ([]byte, error) {
    // 1. 获取该 SSRC 的状态（ROC、index window）
    s := c.getSRTPSSRCState(header.SSRC)

    // 2. 计算 SRTP Index: 利用 ROC 和当前 SeqNum 推断
    //    (roc << 16) | sequenceNumber
    // 注：replay 检测在 authenticate BEFORE decrypt — 优化：拒绝重放包可节省 AES+HMAC 计算
    if err := s.replayDetector.Check(index); err != nil {
        return nil, err   // 已见过此 index → 丢弃（防重放攻击）
    }

    // 3. 找到对应密钥（支持 MKI = Master Key Identifier）
    cipher := c.cipher  // 或通过 MKI 查 cipher 表

    // 4. 执行解密（AES-CTR 模式 or AES-GCM）
    dst, err = cipher.decryptRTP(dst, encrypted, header, ...)

    // 5. 解密成功后提交 replay 状态（不可逆）
    s.replayDetector.Accept(index)
    return dst, err
}
```

**关键设计**：Replay Check **先于** 解密执行。这是刻意的：防止攻击者重放大量包触发 CPU 密集的 AES+HMAC 操作（DoS 防御）。只有认证通过后才 Accept index。

### 3.4 Protection Profile 对比

| Profile | 算法 | 密钥长度 | Tag 长度 | 性能 | WebRTC 推荐度 |
|---------|------|---------|---------|------|-------------|
| SRTP_AES128_CM_SHA1_80 | AES-CTR + HMAC-SHA1 | 128 bit | 80 bit | 高（SIMD 加速） | 遗留兼容 |
| SRTP_AES128_CM_SHA1_32 | AES-CTR + HMAC-SHA1 | 128 bit | 32 bit | 最高 | 低（tag 太短） |
| SRTP_AES256_CM_SHA1_80 | AES-CTR + HMAC-SHA1 | 256 bit | 80 bit | 中 | 高安全场景 |
| SRTP_AEAD_AES_128_GCM | AES-GCM | 128 bit | 128 bit | 高（AES-NI） | **推荐** |
| SRTP_AEAD_AES_256_GCM | AES-GCM | 256 bit | 128 bit | 中 | 高安全推荐 |

Chrome 默认协商顺序（2024+）：AES-128-GCM > AES-128-CM-SHA1-80。

### 3.5 Pion SRTP Context 初始化

【真实源码 pion/srtp@master/context.go】

```go
// Context 持有整个 SRTP 会话的加密状态
type Context struct {
    cipher        srtpCipher         // 多态：GCM 或 CM-HMAC 实现
    srtpSSRCStates map[uint32]*srtpSSRCState  // 每路流独立 ROC + replay window
    srtcpSSRCStates map[uint32]*srtcpSSRCState
    // MKI 支持：允许在会话中切换主密钥（密钥更新场景）
}

// CreateContext 初始化加密上下文
func CreateContext(masterKey, masterSalt []byte,
    profile ProtectionProfile, opts ...ContextOption) (*Context, error) {
    // 应用默认选项后叠加用户选项（ContextOption 模式）
    // 根据 profile 选择 cipher 实现：
    //   SRTP_AEAD_AES_128_GCM → srtpCipherAeadAesGcm{}
    //   SRTP_AES128_CM_SHA1_80 → srtpCipherAesCmHmacSha1{}
}
```

---

## 4 RTCP：控制平面精读

### 4.1 RTCP 类型概览

RTCP 与 RTP 共享同一 5-tuple（DTLS 之上），通过 PT 字段区分：

| PT | 类型 | 用途 |
|----|------|------|
| 200 | SR（Sender Report） | 发送方时钟 + 媒体统计 |
| 201 | RR（Receiver Report） | 接收质量反馈（丢包/抖动） |
| 202 | SDES | Source Description（CNAME）|
| 203 | BYE | 流结束通知 |
| 205 | RTPFB | RTP 层反馈（NACK, TWCC） |
| 206 | PSFB | Payload 层反馈（PLI, FIR, REMB） |

### 4.2 ReceptionReport：丢包与抖动的载体

【真实源码 pion/rtcp@master/reception_report.go】

```go
type ReceptionReport struct {
    SSRC               uint32  // 被报告的发送方 SSRC
    FractionLost       uint8   // 过去间隔内丢包分数 (0=0%, 255=100%)
    TotalLost          uint32  // 累计丢包总数（24 bit 有效）
    LastSequenceNumber uint32  // 收到的最大扩展序列号
    Jitter             uint32  // 到达时间抖动（RTP 时钟单位）
    LastSenderReport   uint32  // 最后收到的 SR 的 NTP 中 32 bit
    Delay              uint32  // 上次 SR 到本 RR 的延迟（1/65536 秒）
}

func (r ReceptionReport) Marshal() ([]byte, error) {
    rawPacket := make([]byte, receptionReportLength)  // 24 bytes
    binary.BigEndian.PutUint32(rawPacket, r.SSRC)
    rawPacket[fractionLostOffset] = r.FractionLost
    // TotalLost 是 24-bit 字段，存 3 字节
    tlBytes := rawPacket[totalLostOffset:]
    tlBytes[0] = byte(r.TotalLost >> 16)
    tlBytes[1] = byte(r.TotalLost >> 8)
    tlBytes[2] = byte(r.TotalLost)
    binary.BigEndian.PutUint32(rawPacket[jitterOffset:], r.Jitter)
    // RTT 计算：Delay + (now - LastSenderReport 对应时刻) = RTT
    binary.BigEndian.PutUint32(rawPacket[lastSROffset:], r.LastSenderReport)
    binary.BigEndian.PutUint32(rawPacket[delayOffset:], r.Delay)
    return rawPacket, nil
}
```

**RTT 计算**（RFC 3550 §6.4.1）：

```
RTT = now_NTP - LSR - DLSR
```
其中 `LSR = LastSenderReport`（SR 的 NTP 中间 32 bit），`DLSR = Delay`。这是 WebRTC 中估算往返延迟的唯一纯 RTCP 途径。

### 4.3 抖动计算算法

RFC 3550 定义的 interarrival jitter 是对到达时间差方差的**指数加权移动平均**：

```
D(i,j) = |( Rj - Ri ) - ( Sj - Si )|   // 到达间隔 vs 发送间隔之差
J(i) = J(i-1) + (|D(i-1,i)| - J(i-1)) / 16
```

这是一个 1/16 权重的 EWMA，不需要保留所有历史记录，单次更新 O(1)。

### 4.4 关键 RTCP 反馈信号

#### NACK（RFC 4585）

```
RTPFB (PT=205), FMT=1
+--------+--------+--------+
| PID (16bit) | BLP (16bit) |
```

- `PID`：丢失包的 SeqNum
- `BLP`：以 PID 为起点后 16 个包的丢失 bitmap

接收方发现 gap（期望 SeqNum N 但收到 N+k）触发 NACK。发送方可决定是否重传（实时性 vs 带宽 tradeoff）。

#### PLI（Picture Loss Indication，RFC 4585）

单一包，请求完整的 I-frame。常见于：
- 解码器检测到解码错误
- 接收方刚加入（需要关键帧起点）
- 丢包率过高导致 P-frame 链断裂

#### REMB（Receiver Estimated Max Bitrate，draft-alvestrand-rmcat-remb）

接收方估计的最大接收带宽，通知发送方降速。**注意**：REMB 已被 Transport-CC 逐步取代（但旧浏览器仍用 REMB）。

#### Transport-CC（draft-holmer-rmcat-transport-wide-cc-extensions）

每个 RTP 包携带 transport-wide seq num（通过 RTP Extension Header），接收方在 RTCP TWCC 中上报所有包的接收时间，发送方侧用 **GCC（Google Congestion Control）** 算法做带宽估计。这是 Chrome 2017 年后的默认拥塞控制方案。

---

## 5 WebRTC 媒体发送完整路径

```
getUserMedia()
    ↓
MediaStreamTrack（PCM/YUV 原始帧）
    ↓
RTCRtpSender
  ├── Encoder（Opus/VP8/VP9/H264/AV1）
  ├── RTP Packetizer（分帧、加 RTP header、填 TS/SeqNum）
  ├── RTP Extensions（MID/TWCC/AbsSendTime）
  ├── SRTP Encrypt（DTLS 导出的密钥）
  └── ICE Socket（UDP，选定的 candidate pair）
    ↓
网络
    ↓
RTCRtpReceiver
  ├── SRTP Decrypt
  ├── RTCP 解析（SR/RR → 统计；NACK/PLI → 重传/I-frame 请求）
  ├── RTP Jitter Buffer（乱序重排、时间戳平滑播放）
  └── Decoder
```

关键时序约束：
- Opus 打包间隔：20 ms（960 samples @ 48 kHz）
- VP8 / VP9 Jitter Buffer：典型 50–150 ms 适配性延迟
- 端到端目标延迟：< 300 ms（ITU G.114 speech threshold）

---

## 6 可运行 Demo

### Demo A：浏览器解析 RTP Header 字段（JS + chrome://webrtc-internals）

**完整可运行代码**——复制到本地 `index.html`，用任意 HTTP server 打开：

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>WebRTC RTP Header Inspector</title>
</head>
<body>
<button id="start">Start & Inspect RTP</button>
<pre id="log" style="background:#1a1a1a;color:#0f0;padding:16px;font-size:13px;
  overflow:auto;height:500px;"></pre>
<script>
const log = (msg) => {
  document.getElementById('log').textContent += msg + '\n';
};

document.getElementById('start').onclick = async () => {
  // 1. 获取本地摄像头+麦克风
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: true, video: { width: 320, height: 240 }
  });

  // 2. 建立本机 loopback PeerConnection 对
  const pc1 = new RTCPeerConnection({
    iceServers: [],
    // WebRTC 默认强制 SRTP，无法关闭；此处仅演示 RTP 层逻辑
  });
  const pc2 = new RTCPeerConnection({ iceServers: [] });

  // ICE candidate 直接转发（本机 loopback，无需 STUN）
  pc1.onicecandidate = e => e.candidate && pc2.addIceCandidate(e.candidate);
  pc2.onicecandidate = e => e.candidate && pc1.addIceCandidate(e.candidate);

  // 3. 添加媒体轨道
  stream.getTracks().forEach(t => pc1.addTrack(t, stream));

  // 4. 在 pc2 上挂 RTCRtpReceiver，监听统计
  pc2.ontrack = async (event) => {
    const receiver = event.receiver;
    const track = event.track;
    log(`[OnTrack] kind=${track.kind} id=${track.id}`);

    // 5. 每 500ms 拉取 RTP 统计（含 timestamp/sequenceNumber 信息）
    const statLoop = setInterval(async () => {
      const stats = await receiver.getStats();
      stats.forEach(report => {
        if (report.type === 'inbound-rtp') {
          log(
            `[inbound-rtp] kind=${report.kind} ` +
            `ssrc=${report.ssrc} ` +
            `pt=${report.payloadType} ` +
            `seq=${report.extendedSequenceNumber ?? 'N/A'} ` +
            `ts=${report.timestamp?.toFixed(0)} ` +
            `jitter=${(report.jitter * 1000).toFixed(1)}ms ` +
            `lost=${report.packetsLost} ` +
            `recv=${report.packetsReceived}`
          );
        }
        if (report.type === 'remote-inbound-rtp') {
          // RTT via RTCP RR (LSR/DLSR 计算)
          log(
            `[remote-inbound] roundTripTime=${
              (report.roundTripTime * 1000).toFixed(1)}ms ` +
            `fractionLost=${report.fractionLost?.toFixed(4)}`
          );
        }
      });
    }, 500);

    // 30s 后清理
    setTimeout(() => {
      clearInterval(statLoop);
      pc1.close(); pc2.close();
      stream.getTracks().forEach(t => t.stop());
      log('[DONE] connections closed');
    }, 30000);
  };

  // 6. SDP Offer/Answer 协商
  const offer = await pc1.createOffer();
  await pc1.setLocalDescription(offer);
  await pc2.setRemoteDescription(offer);
  const answer = await pc2.createAnswer();
  await pc2.setLocalDescription(answer);
  await pc1.setRemoteDescription(answer);

  // 7. 打印 SDP 中的 SSRC 和 PT 分配
  log('\n=== SDP PT/SSRC 分析 ===');
  const sdpLines = answer.sdp.split('\r\n');
  sdpLines.filter(l =>
    l.startsWith('a=rtpmap') ||
    l.startsWith('a=ssrc') ||
    l.startsWith('a=crypto') ||   // 无此行 = 用 DTLS-SRTP（正确）
    l.startsWith('a=fingerprint')
  ).forEach(l => log(l));
};
</script>
</body>
</html>
```

**运行步骤**：

```bash
# 1. 需要 HTTPS 或 localhost（getUserMedia 要求）
python3 -m http.server 8080
# 或
npx serve .

# 2. 浏览器打开 http://localhost:8080
# 3. 点击 "Start & Inspect RTP"，授权摄像头/麦克风
# 4. 同时打开 chrome://webrtc-internals 观察
```

**预期输出**：

```
[OnTrack] kind=audio id=...
[OnTrack] kind=video id=...
=== SDP PT/SSRC 分析 ===
a=rtpmap:111 opus/48000/2
a=rtpmap:96 VP8/90000
a=fingerprint:sha-256 AA:BB:CC:...   ← DTLS 指纹，证明 SRTP 通道
a=ssrc:1234567890 cname:someidentifier
[inbound-rtp] kind=audio ssrc=1234567890 pt=111 jitter=0.5ms lost=0 recv=47
[inbound-rtp] kind=video ssrc=9876543210 pt=96 jitter=3.2ms lost=0 recv=23
[remote-inbound] roundTripTime=1.2ms fractionLost=0.0000
```

**chrome://webrtc-internals 观察要点**：
- `Stats graphs` → `packetsSent` / `packetsReceived`：验证 RTP 包在流动
- `Stats graphs` → `jitterBufferDelay`：观察 jitter buffer 工作
- `Stats graphs` → `roundTripTime`：RTCP RR 计算的 RTT
- SDP 中 `a=fingerprint`：确认 DTLS-SRTP（无 `a=crypto` 说明不是 SDES）

---

### Demo B：Pion Go —— 接收 RTP 并解析 Header 字段

**完整可运行代码**，不依赖浏览器，纯 Go UDP 收发：

```go
// rtp_receiver.go — Pion RTP 包接收与 Header 解析
// 运行: go run rtp_receiver.go
// 搭配: ffmpeg -re -i input.mp4 -vcodec libvp8 -f rtp rtp://127.0.0.1:5004
package main

import (
    "fmt"
    "log"
    "net"

    "github.com/pion/rtp"
)

func main() {
    // 监听 5004 端口（裸 RTP，演示 header 解析；生产环境需 SRTP）
    conn, err := net.ListenUDP("udp", &net.UDPAddr{Port: 5004})
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    fmt.Println("Listening for RTP on :5004 ...")

    buf := make([]byte, 1500)
    var prevSeq uint16
    var prevTS  uint32
    var prevRecvTime int64
    pktCount := 0

    for {
        n, addr, err := conn.ReadFromUDP(buf)
        if err != nil {
            log.Println("read err:", err)
            continue
        }

        // ---- Pion RTP 解析 ----
        packet := &rtp.Packet{}
        if err := packet.Unmarshal(buf[:n]); err != nil {
            log.Println("unmarshal err:", err)
            continue
        }
        h := packet.Header
        now := timeNowMicro()

        fmt.Printf(
            "[#%04d] from=%s V=%d PT=%d SeqNum=%5d TS=%10d SSRC=0x%08X "+
                "M=%v Ext=%v PayloadLen=%d",
            pktCount, addr,
            h.Version, h.PayloadType, h.SequenceNumber,
            h.Timestamp, h.SSRC, h.Marker, h.Extension,
            len(packet.Payload),
        )

        // 计算 SeqNum 连续性（丢包检测）
        if pktCount > 0 {
            seqDelta := int(h.SequenceNumber) - int(prevSeq)
            if seqDelta < 0 {
                seqDelta += 65536 // wrap around
            }
            if seqDelta != 1 {
                fmt.Printf(" ⚠ GAP=%d", seqDelta-1)
            }

            // 计算 interarrival jitter (简化版 RFC 3550 §A.8)
            tsDelta  := int64(h.Timestamp) - int64(prevTS)
            recvDelta := now - prevRecvTime  // microseconds
            // 假设 PT=96 是 VP8，clock=90000Hz → 1μs = 90000/1e6 = 0.09 clock unit
            clockUnitsPerMicro := float64(90000) / 1e6
            tsDeltaInMicro := float64(tsDelta) / clockUnitsPerMicro
            d := recvDelta - int64(tsDeltaInMicro)
            if d < 0 { d = -d }
            fmt.Printf(" jitter_d=%dμs", d)
        }
        fmt.Println()

        // 解析 One-Byte Extension（如有）
        if h.Extension {
            fmt.Printf("  Extensions (profile=0x%04X):\n", h.ExtensionProfile)
            for _, ext := range h.Extensions {
                fmt.Printf("    ID=%d len=%d data=%X\n",
                    ext.ID(), len(ext.Payload()), ext.Payload())
            }
        }

        prevSeq = h.SequenceNumber
        prevTS  = h.Timestamp
        prevRecvTime = now
        pktCount++
    }
}

func timeNowMicro() int64 {
    // 在实际实现中使用 time.Now().UnixMicro()
    // 此处占位，Go 1.17+ 才有 UnixMicro
    return 0  // 替换为 time.Now().UnixMicro()
}
```

**项目初始化**：

```bash
mkdir rtp-demo && cd rtp-demo
go mod init rtpdemo
go get github.com/pion/rtp@latest
# 复制上面代码到 rtp_receiver.go
# 修正 timeNowMicro() 返回 time.Now().UnixMicro()
go run rtp_receiver.go
```

**搭配发送端**（ffmpeg）：

```bash
# 发送 VP8 编码的 RTP 流到本机 5004 端口
ffmpeg -re -f lavfi -i testsrc=duration=30:size=320x240:rate=30 \
       -vcodec libvpx -b:v 500k \
       -f rtp rtp://127.0.0.1:5004
```

**预期输出**：

```
Listening for RTP on :5004 ...
[#0000] from=127.0.0.1:56789 V=2 PT=96 SeqNum=    1 TS=   90000 SSRC=0x12345678 M=true Ext=false PayloadLen=487
[#0001] from=127.0.0.1:56789 V=2 PT=96 SeqNum=    2 TS=   90000 SSRC=0x12345678 M=false Ext=false PayloadLen=1420
[#0002] from=127.0.0.1:56789 V=2 PT=96 SeqNum=    3 TS=   93000 SSRC=0x12345678 M=true Ext=false PayloadLen=312
```

**观察要点**：
- 同一帧视频 TS 相同（VP8 分帧打包），最后一个分片 `M=true`
- TS 差值 = 3000 = 90000/30（30fps 下每帧 3000 clock units）
- GAP 出现时丢包（网络抖动或 ffmpeg 缓冲）

---

### Demo C：Pion Go —— 完整 WebRTC SRTP 接收（Save-to-Disk 模式）

【基于真实源码 pion/webrtc@master/examples/save-to-disk/main.go】

```go
// srtp_receiver.go — Pion WebRTC 接收端，展示 SRTP 透明解密
// 完整示例见: github.com/pion/webrtc/tree/master/examples/save-to-disk
// 运行: go run srtp_receiver.go (然后粘贴浏览器 offer)
package main

import (
    "fmt"
    "strings"

    "github.com/pion/webrtc/v4"
)

func main() {
    pc, _ := webrtc.NewPeerConnection(webrtc.Configuration{
        ICEServers: []webrtc.ICEServer{{URLs: []string{"stun:stun.l.google.com:19302"}}},
    })

    // OnTrack: SRTP 解密对上层透明，直接拿到明文 RTP 包
    pc.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
        codec := track.Codec()
        fmt.Printf("[OnTrack] kind=%s codec=%s PT=%d SSRC=%d\n",
            track.Kind(), codec.MimeType, codec.PayloadType, track.SSRC())

        for {
            // ReadRTP() 内部已完成 SRTP 解密、序列号验证、replay 检测
            pkt, _, err := track.ReadRTP()
            if err != nil {
                fmt.Println("ReadRTP error:", err)
                return
            }
            fmt.Printf("  RTP: SeqNum=%d TS=%d Marker=%v PayloadLen=%d\n",
                pkt.SequenceNumber, pkt.Timestamp,
                pkt.Marker, len(pkt.Payload))

            // 在这里可以写入文件、推送到解码器等
        }
    })

    // 状态变化监听（连接建立 = DTLS + ICE 均完成）
    pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
        fmt.Printf("[State] %s\n", state)
        if state == webrtc.PeerConnectionStateFailed {
            fmt.Println("Connection failed, exiting")
        }
    })

    // 读取浏览器 Offer（base64 JSON）
    fmt.Println("Paste browser SDP offer (then press Enter twice):")
    offer := webrtc.SessionDescription{}
    decode(readInput(), &offer)  // decode = base64 + JSON unmarshal

    pc.SetRemoteDescription(offer)
    answer, _ := pc.CreateAnswer(nil)
    pc.SetLocalDescription(answer)
    <-webrtc.GatheringCompletePromise(pc)

    fmt.Println("=== Answer (paste into browser) ===")
    fmt.Println(encode(pc.LocalDescription()))  // encode = JSON + base64

    select {} // block
}

// encode/decode/readInput 参见 pion examples 公共实现（信号交换）
```

**运行步骤**：

```bash
go mod init srtpdemo
go get github.com/pion/webrtc/v4
go run srtp_receiver.go
# 浏览器端打开 pion save-to-disk 的 HTML sender
# 或用 Demo A 中的 pc1 → pc2 替换为此 Go 接收端
```

---

## 7 失败模式与真实坑

### 7.1 SeqNum Wrap-Around 处理不当

**症状**：SeqNum 从 65535 回绕到 0 后，jitter buffer 误判大量包为"乱序"或"极旧"，触发大量无谓丢弃。

**根因**：比较 SeqNum 时用有符号减法而非 modular arithmetic。RFC 3550 §A.1 明确：

```c
// RFC 3550 Appendix A.1 — 正确的序列号推进判断
#define RTP_SEQ_MOD (1<<16)
// delta > MAX_DROPOUT → 乱序 or 大跳（不是递增）
// delta < RTP_SEQ_MOD - MAX_MISORDER → 太旧
```

**修复**：始终用 `(uint16)(a - b)` 计算差值，或用 extended sequence number（32-bit = ROC<<16 | seq）做比较。

### 7.2 DTLS-SRTP Profile 不匹配

**症状**：`DTLS alert: handshake failure` 或连接建立后无音视频。

**根因**：SDP 中 `a=crypto` 出现（SDES 模式），而对端期望 DTLS-SRTP（`a=fingerprint`），或双端 Protection Profile 列表没有交集。

**排查**：

```bash
# 用 Wireshark 抓包，过滤 DTLS
dtls.handshake.type == 1   # Client Hello
# 检查 use_srtp extension 中的 protection_profiles 字段
```

WebRTC 强制 `a=fingerprint`，若 SDP 有 `a=crypto` 则是 SIP/SDES，与 WebRTC 不兼容。

### 7.3 ROC 不同步（SRTP Key Reset 后）

**症状**：接收端突然全部解密失败（`authentication tag mismatch`），而非单包失败。

**根因**：发送端重启 / 重建 PeerConnection 后，ROC 从 0 重置，但接收端的 SRTP Context 仍持有旧 ROC 值。

**修复**：重建 PeerConnection 必须重新进行 DTLS 握手，接收端创建全新的 SRTP Context。不能复用旧 session。

### 7.4 NAT 映射超时导致 SRTP 包静默丢失

**症状**：通话建立后，约 30–60 秒后音视频突然中断，ICE 状态仍显示 connected。

**根因**：NAT 设备 UDP 超时通常 30s，RTCP 间隔（5–10s）无法保活所有五元组。

**修复**：
- 启用 STUN keepalive（ICE consent freshness，RFC 7675）
- RTCP 最小间隔 < NAT 超时（5s 以下）
- 或强制通过 TURN（代价是延迟增加）

### 7.5 大带宽下 SRTP GCM Tag 验证 CPU 飙升

**症状**：4K 视频流（> 5 Mbps）时 CPU 使用率异常高，SRTP 解密成为瓶颈。

**根因**：未启用 AES-NI 硬件指令，软件 AES-GCM 在高比特率下 CPU 占用不可忽略。

**排查**：

```bash
# 确认 AES-NI 可用
grep aes /proc/cpuinfo | head -1
# Go 在 amd64 上自动使用 AES-NI（通过 crypto/aes 包）
# 但 CGo 禁用或交叉编译到非 native arch 时回退到软件实现
```

### 7.6 Jitter Buffer 过小导致音频断续

**症状**：网络抖动 > 80ms 时音频出现明显断点（glitch）。

**根因**：Jitter Buffer 目标延迟设置过激进（< 50ms），到达晚的包被丢弃而非等待。

**权衡**：
- Buffer 小 → 延迟低但抗抖动弱（适合实时对话）
- Buffer 大 → 抗抖动强但延迟高（适合直播单向推流）

WebRTC 的 Adaptive Jitter Buffer 基于历史抖动自动调整，但初始值设置不当仍会引发早期断续。

### 7.7 RTCP Compound Packet 与带宽误估

**症状**：REMB 或 TWCC 反馈延迟大，带宽估计不准。

**根因**：RFC 3550 要求 RTCP 以**复合包**发送（SR/RR 必须开头），若实现拆分发送或错误合并，导致接收端统计滞后。

**注意**：RFC 5506 放宽了此限制（允许 reduced-size RTCP），但需要在 SDP 中显式协商 `a=rtcp-rsize`。

---

## 8 方案对比

### 8.1 SRTP 密钥协商方式

| 方式 | 使用场景 | 安全性 | 互操作性 |
|------|---------|--------|---------|
| DTLS-SRTP（RFC 5764）| WebRTC 强制 | 高（前向保密可选）| WebRTC 标准 |
| SDES（RFC 4568）| SIP/SDP 内联密钥 | 低（密钥明文在 SDP）| 传统 VoIP |
| MIKEY（RFC 3830）| 企业 SIP 系统 | 高 | 复杂，少用 |
| EKT（RFC 8870）| 多方会议密钥分发 | 高 | 新标准，实现少 |

WebRTC **不允许 SDES**（RFC 8827 §6.5 明确禁止），必须用 DTLS-SRTP。

### 8.2 拥塞控制算法

| 算法 | 信号来源 | 适用场景 | 主要问题 |
|------|---------|---------|---------|
| REMB | 接收端带宽估计 | 旧版 WebRTC | 单向、不适合 bi-dir |
| GCC（Transport-CC）| 发送端基于 TWCC 延迟梯度 | Chrome 默认 | 在高 RTT 下收敛慢 |
| NADA（RFC 8698）| 延迟梯度 + 丢包 | IETF RMCAT 标准 | 实现少 |
| SCReAM（RFC 8298）| 丢包 + 队列延迟 | 低延迟媒体 | 实现少 |

### 8.3 WebRTC 实现对比

| 维度 | libwebrtc（Chrome 内核）| Pion（Go）| mediasoup（Node.js/C++）|
|------|----------------------|-----------|----------------------|
| SRTP | BoringSSL + AES-NI | Go crypto/aes + golang.org/x/crypto | OpenSSL |
| RTCP | 内置 SR/RR/NACK/PLI/REMB/TWCC | pion/rtcp 独立库 | libsrtp2 |
| Jitter Buffer | Adaptive，生产级 | 基础实现，需 NetEQ 插件 | 无（SFU 不解码）|
| 适用场景 | 浏览器 | 自研服务器/SFU | SFU/MCU 生产部署 |

---

## 9 五件套

### ① 关键概念速查

| 术语 | 含义 |
|------|------|
| SSRC | Synchronization Source，每路媒体流的 32-bit 随机 ID |
| PT | Payload Type，7-bit 编码器标识，由 SDP 动态分配（96-127）|
| ROC | Rollover Counter，SRTP 48-bit index 的高 32 位 |
| TWCC | Transport-Wide Congestion Control，发送方侧带宽估计框架 |
| PLI | Picture Loss Indication，请求 I-frame 的 RTCP 信号 |
| DTLS-SRTP | DTLS 握手后导出 SRTP 密钥的标准方法（WebRTC 唯一合法方式）|
| Jitter Buffer | 接收端缓冲，平衡网络抖动与播放延迟 |
| Extended Seq | 32-bit = ROC<<16 + SeqNum，用于全局唯一包标识 |

### ② 常见问题排查路径

```
音视频无数据
  ├── ICE connected? → 否 → 第 2 章 NAT 穿透问题
  ├── DTLS connected? → 否 → 证书指纹不匹配 / Profile 不支持
  ├── OnTrack 触发? → 否 → SDP m-line direction 问题（sendonly/recvonly）
  └── SRTP 解密错? → 是 → ROC 不同步 / 密钥材料错误

音频断续
  ├── jitter > jitter buffer? → 调大 buffer 或 改网络
  ├── 丢包率 > 5%? → 检查 NACK / PLC 配置
  └── CPU 过载? → 检查 SRTP cipher 是否用硬件加速

视频花屏/绿屏
  ├── PLI 发出没有? → RTCP feedback 是否通路
  └── 丢包 P-frame? → 触发 PLI → 等待 I-frame
```

### ③ 必读 RFC/文档

| RFC | 内容 | 重要性 |
|-----|------|--------|
| RFC 3550 | RTP 基础规范 | ★★★★★ |
| RFC 3711 | SRTP | ★★★★★ |
| RFC 5764 | DTLS-SRTP | ★★★★★ |
| RFC 8285 | RTP Header Extensions (one/two-byte) | ★★★★ |
| RFC 4585 | Extended RTP Profile (NACK/PLI) | ★★★★ |
| RFC 8827 | WebRTC Security Architecture | ★★★★ |
| RFC 7714 | AES-GCM for SRTP | ★★★ |
| RFC 8698 | NADA 拥塞控制 | ★★★ |

### ④ 深入阅读路径

1. **libwebrtc 源码**：`modules/rtp_rtcp/source/rtp_packet.cc` — 生产级 RTP 解析
2. **Pion 完整示例**：`github.com/pion/webrtc/tree/master/examples` — 20+ 可运行 demo
3. **mediasoup RTP 处理**：`worker/src/RTC/RtpPacket.cpp` — C++ SFU 侧处理
4. **Chrome WebRTC 内部**：`chrome://webrtc-internals` 的 Stats API 完整字段列表（W3C WebRTC-Stats）
5. **Wireshark RTP 解析**：`Telephony → RTP → RTP Streams` 可视化所有字段

### ⑤ 下章预告

**第 4 章：编解码器与媒体协商（Opus/VP8/VP9/H264/AV1）**

- SDP `a=rtpmap` / `a=fmtp` 详细语义
- Opus CBR/VBR/DTX/FEC 参数影响
- VP9 SVC 层级（spatial/temporal layers）
- H.264 Profile/Level 匹配陷阱
- simulcast 与 SFU 选流

---

## 参考文献与源码索引

| 类型 | 位置 | 注记 |
|------|------|------|
| 真实源码 | pion/rtp@master/packet.go | Header struct + Unmarshal bit ops |
| 真实源码 | pion/srtp@master/session.go | SessionKeys + session.start() |
| 真实源码 | pion/srtp@master/context.go | Context struct + CreateContext |
| 真实源码 | pion/srtp@master/srtp.go | decryptRTP/encryptRTP + ROC logic |
| 真实源码 | pion/rtcp@master/reception_report.go | RR Marshal/Unmarshal + RTT 计算 |
| 真实源码 | pion/webrtc@master/examples/save-to-disk/main.go | OnTrack + ReadRTP loop |
| 规范 | RFC 3550 | RTP 基础（Schulzrinne 等，2003） |
| 规范 | RFC 3711 | SRTP（Baugher 等，2004） |
| 规范 | RFC 5764 | DTLS-SRTP（McGrew & Rescorla，2010） |
| 规范 | RFC 8285 | RTP One/Two-Byte Header Extensions |
| 规范 | RFC 4585 | Extended RTP Profile for RTCP-based Feedback |
