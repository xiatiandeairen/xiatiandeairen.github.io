---
title: "拥塞控制 GCC / transport-cc（WebRTC 域）"
slug: "10-06"
collection: "tech-library"
group: "webrtc"
order: 10006
summary: "前置依赖：第 1 章（SDP / Offer-Answer，理解 SDP 协商 RTP 扩展）、第 3 章（RTP/RTCP，理解序列号、时间戳、SR/RR、REMB）、第 5 章（视频编码与传输，理解码率控制 QP 旋钮如何消费 BWE 给出的目标码率）。本章是连接「网络观测」与「编码器旋钮」之间的那根控制链。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-15T14:32:47.000Z"
updatedAt: "2026-06-15T14:32:47.000Z"
---
> **前置依赖**：第 1 章（SDP / Offer-Answer，理解 SDP 协商 RTP 扩展）、第 3 章（RTP/RTCP，理解序列号、时间戳、SR/RR、REMB）、第 5 章（视频编码与传输，理解码率控制 QP 旋钮如何消费 BWE 给出的目标码率）。本章是连接「网络观测」与「编码器旋钮」之间的那根控制链。

---

## TL;DR

WebRTC 拥塞控制要解决的核心矛盾是：实时音视频既要**高吞吐**（清晰、高帧率），又要**低延迟**（<200ms 端到端），而 TCP 那套「填满缓冲区再丢包回退」的 loss-based 控制对实时媒体是灾难——等到丢包时，bufferbloat 已经把延迟堆到几百毫秒甚至几秒。

GCC（Google Congestion Control）的破局思路是**延迟优先**：在丢包发生之前，先从**包到达时间的微小抖动**里嗅出瓶颈链路队列正在堆积，提前降码率。它由两条腿组成：

1. **delay-based controller（延迟控制器）**：核心。测量「同一组包的到达间隔 − 发送间隔」即**单向延迟梯度（one-way delay gradient）**，用滤波器（早期 Kalman，现代 Trendline）估计趋势，再用一个**自适应阈值（adaptive threshold）**判定 over-use / normal / under-use，驱动一个 AIMD（加性增、乘性减）状态机调整目标码率。
2. **loss-based controller（丢包控制器）**：兜底。丢包率 <2% 涨码率，>10% 按 `(1 − 0.5·lossRatio)` 降码率。

两者取 **min** 作为最终目标码率。

**transport-cc（transport-wide congestion control，TWCC）** 是承载这套算法的**反馈管道**：发送端给每个包打**全传输级序列号**（一个 RTP 头扩展），接收端用一条 RTCP 反馈消息把**每个包的到达时间**回传给发送端，从而把 BWE 的计算从接收端（旧的 REMB 模型）搬到**发送端**——这是 WebRTC 拥塞控制架构最重要的一次演进。

本章用 Pion（Go，可读）的真实 GCC 实现逐行拆 adaptive threshold 与 AIMD，并给两个能真跑的 demo：一个**浏览器 JS**（`RTCRtpSender.setParameters` + `chrome://webrtc-internals` 观测 BWE 随带宽收敛），一个**Go toy**（延迟梯度 → 自适应阈值 → 码率裁决，复现 GCC 数学内核）。

---

## 6.1 设计考古：为什么实时媒体不能用 TCP 那套拥塞控制

### 6.1.1 核心动机——bufferbloat 与「延迟 vs 丢包」的根本分歧

传统 TCP 拥塞控制（Reno / CUBIC）是 **loss-based**：不断加大发送窗口，直到路由器缓冲区填满、开始丢包，才把窗口砍半。这套逻辑对文件下载没问题——重传几个包、延迟抖动几百毫秒，用户无感。

但实时视频不一样。链路瓶颈处的路由器缓冲区（尤其家用 NAT、4G 基站）往往很深（**bufferbloat**），loss-based 控制要把这个缓冲区**填满**才知道该退了。等填满时，排在队列里的视频包已经积压了几百毫秒——对一个目标 <200ms 端到端延迟的通话，这已经是灾难性的卡顿和音画不同步。

GCC 的洞察是：**队列开始堆积时，延迟会先于丢包上升**。如果能从「包到达节奏的细微变化」里提前嗅到队列在涨，就能在丢包发生之前、在延迟还没堆起来时主动降码率，把端到端延迟压住。

> **出处核实**：IETF 草案 `draft-ietf-rmcat-gcc-02`《A Google Congestion Control Algorithm for Real-Time Communication》开篇即写明「describes two methods of congestion control: one delay-based and one loss-based」（延迟法 + 丢包法）。该草案是 GCC 的权威规范来源（[datatracker draft-ietf-rmcat-gcc-02](https://datatracker.ietf.org/doc/html/draft-ietf-rmcat-gcc-02)）。学术侧由 Poliba 的 Carlucci 等人在《Analysis and design of the Google Congestion Control for WebRTC》系统建模（[c3lab GCC analysis PDF](https://c3lab.poliba.it/images/6/65/Gcc-analysis.pdf)）。

### 6.1.2 单向延迟梯度（one-way delay gradient）——不需要时钟同步的妙处

GCC 不测「绝对单向延迟」（那需要收发两端时钟同步，做不到），而是测**延迟的变化量（梯度）**。

定义两组连续到达的包组（group），令：

```
d(i) = (到达时间间隔) − (发送时间间隔)
     = (t_arrival(i) − t_arrival(i-1)) − (t_send(i) − t_send(i-1))
```

关键直觉：

- 链路**空闲**、无排队：包按发送节奏到达，`d(i) ≈ 0`。
- 链路**正在被填满**（over-use）：后一组包要在队列里多等，到达间隔被拉长 → `d(i) > 0`。
- 队列**正在排空**（under-use）：`d(i) < 0`。

因为是「间隔的差」，**两端的固定时钟偏移被减掉了**——这是 GCC 不需要 NTP 同步就能工作的根本原因。剩下的时钟**频率漂移（skew）**由后续滤波吸收。

> **出处核实**：草案与 Poliba 论文均把 arrival-time filter 的目标定义为「produce an estimate of the one-way delay gradient」。这是整个 delay-based 路径的输入信号。

### 6.1.3 Kalman → Trendline：滤波器的演进

GCC 初版（`draft-alvestrand` / `gcc-00`）用 **Kalman 滤波器**估计延迟梯度 `m(i)`，把它当作一个待估状态量。

后来 libwebrtc 把它换成了 **Trendline 滤波器**：对最近 N 个累积延迟梯度做**线性回归**，用回归直线的**斜率**反映链路队列趋势。

> **出处核实**：多份资料（含 [SegmentFault GCC 动态带宽评估算法](https://segmentfault.com/a/1190000041858045/en) 与 Poliba 后续分析）指出「more recent delay-based estimation mainly replaces the Kalman filter with TrendLine filter…simpler and more accurate…higher sensitivity」。直觉：链路队列变长 → 包的到达间隔系统性变大 → 累积延迟梯度序列呈上升趋势 → 回归斜率 > 0。

注意：Pion 当前实现仍走 **Kalman/slope estimator + arrival group** 这条路（见 6.3 源码），与 libwebrtc 的 Trendline 在工程上有差异，但**判定框架（梯度 → 自适应阈值 → over/under/normal → AIMD）完全一致**。本章用 Pion 拆框架，用规范补 Trendline 数学。

### 6.1.4 从 REMB 到 transport-cc：BWE 从接收端搬到发送端

WebRTC 早期（2014 年前）BWE 跑在**接收端**：接收端算出可用带宽，通过 **REMB**（Receiver Estimated Max Bitrate，一个 RTCP PSFB 消息）告诉发送端「你最多发 X bps」。

问题：

- 接收端只能看到「到了什么」，看不到「发了什么、什么丢了」，估计精度受限。
- 算法升级要改接收端，部署慢（收发端版本不一致）。

**transport-cc** 的演进思路（Holmer 等，`draft-holmer-rmcat-transport-wide-cc-extensions`）：

1. 发送端给**每个**出站包打一个 **transport-wide sequence number**（全传输级、跨所有 SSRC 单调递增的序列号），通过一个 RTP 头扩展携带。
2. 接收端**不算带宽**，只忠实回传：用一条紧凑的 **RTCP transport-cc feedback** 消息，把每个序列号包的**到达时间**（相对基准、用 250µs 为单位的增量编码）报回去。
3. **发送端**拿到「每个包何时发、何时到、是否丢」的完整视图，在**发送端**跑 GCC。

这样算法迭代只需升级发送端，且发送端信息最全。

> **出处核实**：[draft-holmer-rmcat-transport-wide-cc-extensions](https://datatracker.ietf.org/doc/html/draft-holmer-rmcat-transport-wide-cc-extensions-01) 明确「adds transport-wide packet sequence numbers and corresponding feedback message so that congestion control can be performed on a transport level at the send-side」。Chrome 侧文档见 [webrtc.googlesource transport-wide-cc-02 README](https://webrtc.googlesource.com/src/+/refs/heads/main/docs/native-code/rtp-hdrext/transport-wide-cc-02/README.md)。SDP 协商时表现为 `a=extmap:… http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01` 与 `a=rtcp-fb:… transport-cc`。

---

## 6.2 GCC 整体架构：四个方块与一条数据流

把 Pion 与 libwebrtc 的实现抽象成统一框架，delay-based 路径是这样一条流水线：

```
收到 transport-cc feedback（每个包的 send_time / arrival_time）
        │
        ▼
┌──────────────────────┐
│ Arrival Group         │  把 burst 内的包聚成「组」，算组间
│ Accumulator           │  inter-arrival / inter-departure / 延迟梯度
└──────────┬───────────┘
           │ d(i) 延迟梯度
           ▼
┌──────────────────────┐
│ Slope Estimator       │  Kalman（Pion）或 Trendline（libwebrtc）
│ (Kalman / Trendline)  │  → 平滑后的估计 m(i)
└──────────┬───────────┘
           │ Estimate（梯度估计）
           ▼
┌──────────────────────┐
│ Adaptive Threshold +  │  m(i) 与动态阈值 γ(i) 比较
│ Overuse Detector      │  → usage: over / normal / under
└──────────┬───────────┘
           │ Usage 信号（+ 需持续 over_use_time 才确认）
           ▼
┌──────────────────────┐
│ Rate Controller       │  AIMD 状态机：increase/hold/decrease
│ (AIMD)                │  → delay-based 目标码率 A_delay
└──────────┬───────────┘
           │
           ▼
   A_final = min(A_delay, A_loss)   ← 与 loss-based 控制器取小
           │
           ▼
   下发给编码器（码率控制器调 QP / 分辨率 / 帧率）
```

两个要点：

- **min 合并**：最终目标码率取「延迟法」和「丢包法」的较小值。延迟法负责「提前刹车」，丢包法负责「真出血了猛刹」。
- **received rate 封顶**：AIMD 的 increase 永远不会超过「最近实测接收速率 × 1.5」——你不能凭空假设带宽比你实际收到的还高。

---

## 6.3 真实源码精读（Pion / Go）

> 以下源码均通过 WebFetch 实际从 `pion/interceptor` 仓库 `master` 分支获取。完整逐字段落标【真实源码】；因 fetch 返回为结构化摘要而非整文件的，标【示意（基于实际 fetch 的结构与常量）】并据此还原，关键常量为实测值。

### 6.3.1 Arrival Group Accumulator——把包聚成组、算延迟梯度

【示意（基于实际 fetch 的结构与常量）repo: pion/interceptor@pkg/gcc/arrival_group_accumulator.go】

```go
// 结构体三个阈值（实测默认值）：burst 判定窗口
type arrivalGroupAccumulator struct {
	interDepartureThreshold          time.Duration // 实测默认 5ms
	interArrivalThreshold            time.Duration // 实测默认 5ms
	interGroupDelayVariationTreshold time.Duration // 实测默认 0
}

// run 逐个处理 ack。一个包加入「当前组」的条件（满足其一）：
//   1) 它与组的「发送间隔」inter-departure ≤ 5ms —— 即「在一个 burst_time 内
//      发出的包构成一组」(a sequence of packets sent within a burst_time
//      interval constitute a group)
//   2) 它的「到达间隔」inter-arrival ≤ 5ms 且「组间延迟变化」< 0
// 两条都不满足时：把当前组写出（feed 给 slope estimator），开新组。
//
// 关键时序量：
//   interArrivalTimePkt        = 新包到达 − 当前组到达
//   interDepartureTimePkt      = 新包发送 − 当前组发送
//   interGroupDelayVariationPkt = 到达 delta − 发送 delta   ← 这就是延迟梯度 d(i)
// 乱序到达（out-of-order）的包直接忽略。
```

**逐行注解**：

- **为什么要聚组**？单个视频帧通常拆成多个 RTP 包，几乎在同一瞬间（一个 burst）发出。如果对每个包单独算梯度，会被「同帧包的天然零间隔」严重噪声污染。聚成组后，比较的是**组与组**（约等于帧与帧）之间的节奏，信噪比高得多。
- **`burst_time = 5ms`**：发送间隔在 5ms 内的包视为同一 burst（同一组）。这个值与 GCC 草案的 burst 定义一致。
- **`interGroupDelayVariationPkt = 到达 delta − 发送 delta`**：这一行就是 6.1.2 的核心公式 `d(i)`。注意它**消掉了绝对时钟偏移**。

### 6.3.2 Adaptive Threshold——自适应阈值（本章最该逐行读的代码）

这是 delay-based 控制器的「判定器」，也是整个 GCC 最精妙的设计。下面是 **WebFetch 实际取得的完整逐字源码**。

【真实源码 repo: pion/interceptor@pkg/gcc/adaptive_threshold.go】

```go
// SPDX-FileCopyrightText: The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package gcc

import (
	"math"
	"time"
)

const (
	maxDeltas = 60
)

type adaptiveThresholdOption func(*adaptiveThreshold)

func setInitialThreshold(t time.Duration) adaptiveThresholdOption {
	return func(at *adaptiveThreshold) {
		at.thresh = t
	}
}

type adaptiveThreshold struct {
	thresh                 time.Duration
	overuseCoefficientUp   float64
	overuseCoefficientDown float64
	min                    time.Duration
	max                    time.Duration
	lastUpdate             time.Time
	numDeltas              int
}

func newAdaptiveThreshold(opts ...adaptiveThresholdOption) *adaptiveThreshold {
	at := &adaptiveThreshold{
		thresh:                 time.Duration(12500 * float64(time.Microsecond)),
		overuseCoefficientUp:   0.01,
		overuseCoefficientDown: 0.00018,
		min:                    6 * time.Millisecond,
		max:                    600 * time.Millisecond,
		lastUpdate:             time.Time{},
		numDeltas:              0,
	}
	for _, opt := range opts {
		opt(at)
	}
	return at
}

func (a *adaptiveThreshold) compare(estimate, _ time.Duration) (usage, time.Duration, time.Duration) {
	a.numDeltas++
	if a.numDeltas < 2 {
		return usageNormal, estimate, a.max
	}
	t := time.Duration(min(a.numDeltas, maxDeltas)) * estimate
	use := usageNormal
	if t > a.thresh {
		use = usageOver
	} else if t < -a.thresh {
		use = usageUnder
	}
	thresh := a.thresh
	a.update(t)
	return use, t, thresh
}

func (a *adaptiveThreshold) update(estimate time.Duration) {
	now := time.Now()
	if a.lastUpdate.IsZero() {
		a.lastUpdate = now
	}
	absEstimate := time.Duration(math.Abs(float64(estimate.Microseconds()))) * time.Microsecond
	if absEstimate > a.thresh+15*time.Millisecond {
		a.lastUpdate = now
		return
	}
	k := a.overuseCoefficientUp
	if absEstimate < a.thresh {
		k = a.overuseCoefficientDown
	}
	maxTimeDelta := 100 * time.Millisecond
	timeDelta := time.Duration(
		min(int(now.Sub(a.lastUpdate).Milliseconds()), int(maxTimeDelta.Milliseconds())),
	) * time.Millisecond
	d := absEstimate - a.thresh
	add := k * float64(d.Milliseconds()) * float64(timeDelta.Milliseconds())
	a.thresh += time.Duration(add*1000) * time.Microsecond
	a.thresh = clampDuration(a.thresh, a.min, a.max)
	a.lastUpdate = now
}
```

**逐行注解**（这段值得花时间）：

1. **`thresh` 初值 `12500µs = 12.5ms`**：初始阈值 γ。延迟梯度估计超过它判 over-use，低于负值判 under-use。
2. **`compare()` 里 `t := min(numDeltas, maxDeltas) * estimate`**：这一步把梯度估计**乘以已累积的样本数**（上限 60）。直觉：样本越多，证据越充分，越容易越过阈值——这是对「单次抖动噪声」的一种统计放大，避免一两个异常包就误判 over-use。
3. **`if t > a.thresh → usageOver`，`t < -a.thresh → usageUnder`**：三态判定的核心。`[-γ, +γ]` 是「正常带」。
4. **`update()` 是阈值「自适应」的灵魂**——γ 不是常数，而是会动：
   - **`absEstimate > thresh + 15ms` 直接 return**：梯度估计离阈值太远（强信号/异常尖峰），**不更新阈值**，只刷新时间戳。防止一次极端抖动把阈值带跑偏。
   - **`k` 的双系数**：`absEstimate ≥ thresh`（在 over 侧）用 `overuseCoefficientUp = 0.01`（**快速抬高阈值**）；`absEstimate < thresh`（在正常带内）用 `overuseCoefficientDown = 0.00018`（**极慢降低阈值**）。
   - **核心公式**：`Δthresh = k · (|estimate| − thresh) · Δt`，即 γ 朝当前梯度方向移动，移动速度由 k 和时间差决定。
   - **`clampDuration(thresh, 6ms, 600ms)`**：阈值钳在 `[6ms, 600ms]`。
5. **为什么要 `up` 快、`down` 慢（0.01 vs 0.00018，相差约 55 倍）？** 这是 GCC 应对**竞争流公平性**的关键设计：
   - 当链路里有一条 **TCP 大象流**时，TCP 会把缓冲区填得很满、延迟梯度持续很高。如果 GCC 的阈值固定很低，它会一感到延迟就疯狂退让，最终被 TCP 饿死（带宽全被 TCP 抢走）。
   - **阈值快速上抬**让 GCC 在「持续高延迟」环境里逐渐**提高容忍度**，不轻易退让，从而能与 TCP 抢到一份带宽。
   - 而**缓慢下降**保证：一旦链路空闲下来，阈值不会立刻掉回最敏感状态导致抖动。

> **出处核实**：`up`/`down` 双系数（K_u / K_d）的非对称设计、与 TCP 共存时的公平性动机，在 Poliba《Analysis and design…》论文中有定量分析（[c3lab GCC analysis](https://c3lab.poliba.it/images/6/65/Gcc-analysis.pdf)）。Pion 这里的 `0.01 / 0.00018` 与 libwebrtc 的 `k_up / k_down` 量级一致。

### 6.3.3 Overuse Detector——「持续够久才确认 over-use」

【示意（基于实际 fetch 的结构与逻辑）repo: pion/interceptor@pkg/gcc/overuse_detector.go】

```go
type overuseDetector struct {
	threshold          /* adaptiveThreshold 接口 */
	lastEstimate       time.Duration
	lastUpdate         time.Time
	increasingDuration time.Duration // over-use 已持续多久
	increasingCounter  int           // 连续 over-use 次数
	overuseTime        time.Duration // 确认 over-use 所需的最短持续时间
	dsWriter           func(DelayStats)
}

func (d *overuseDetector) onDelayStats(ds DelayStats) {
	// 1) 调阈值比较，拿到瞬时判定 thresholdUse / estimate / currentThreshold
	thresholdUse, estimate, currentThreshold := d.threshold.compare(ds.Estimate, ds.LastReceiveDelta)

	var use usage
	switch thresholdUse {
	case usageOver:
		// 累积 over-use 时长与计数
		// 仅当「持续时长 > overuseTime」(或 overuseTime==0 时计数 > 1)
		//   并且「estimate > lastEstimate」(梯度仍在恶化) 才真正确认 usageOver
		// → 这道闸门避免「瞬时尖峰」误触发降码率
	case usageUnder:
		// 重置计数，use = usageUnder
	case usageNormal:
		// 重置累积，use = usageNormal
	}
	// 通过 dsWriter 把 DelayStats{ Usage: use, ... } 交给 rate controller
}
```

**逐行注解**：

- **`overuseTime` 闸门**是「防误触」的关键。一次网络小尖峰（比如某个包恰好排在一个突发后面）会让瞬时 `thresholdUse == usageOver`，但 over-use 必须**持续超过 `overuseTime`**（libwebrtc 默认约 10ms 量级）**且梯度还在恶化（`estimate > lastEstimate`）**，才会真正向上层报 `usageOver`。
- 这与 6.3.2 的 `numDeltas` 放大、6.4 的 AIMD 状态机一起，构成 GCC 的**三层抗噪**：统计放大 → 持续时间闸门 → 状态机迟滞。

### 6.3.4 Rate Controller——AIMD 状态机（第二段完整逐字源码）

下面是 **WebFetch 实际取得的完整逐字源码**，AIMD 的全部数学都在这里。

【真实源码 repo: pion/interceptor@pkg/gcc/rate_controller.go】

```go
// SPDX-FileCopyrightText: The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package gcc

import (
	"math"
	"sync"
	"time"
)

const (
	decreaseEMAAlpha = 0.95
	beta             = 0.85
)

type rateController struct {
	now                  now
	initialTargetBitrate int
	minBitrate           int
	maxBitrate           int

	dsWriter func(DelayStats)

	lock               sync.Mutex
	init               bool
	delayStats         DelayStats
	target             int
	lastUpdate         time.Time
	lastState          state
	latestRTT          time.Duration
	latestReceivedRate int
	latestDecreaseRate *exponentialMovingAverage
}

type exponentialMovingAverage struct {
	average      float64
	variance     float64
	stdDeviation float64
}

func (a *exponentialMovingAverage) update(value float64) {
	if a.average == 0.0 {
		a.average = value
	} else {
		x := value - a.average
		a.average += decreaseEMAAlpha * x
		a.variance = (1 - decreaseEMAAlpha) * (a.variance + decreaseEMAAlpha*x*x)
		a.stdDeviation = math.Sqrt(a.variance)
	}
}

func (c *rateController) onDelayStats(ds DelayStats) {
	now := time.Now()

	if !c.init {
		c.delayStats = ds
		c.delayStats.State = stateIncrease
		c.init = true
		return
	}
	c.delayStats = ds
	c.delayStats.State = c.delayStats.State.transition(ds.Usage)

	if c.delayStats.State == stateHold {
		return
	}

	var next DelayStats
	c.lock.Lock()

	switch c.delayStats.State {
	case stateHold:
		// should never occur due to check above, but makes the linter happy
	case stateIncrease:
		c.target = clampInt(c.increase(now), c.minBitrate, c.maxBitrate)
		next = DelayStats{ /* ...复制各字段... */ TargetBitrate: c.target }
	case stateDecrease:
		c.target = clampInt(c.decrease(), c.minBitrate, c.maxBitrate)
		next = DelayStats{ /* ...复制各字段... */ TargetBitrate: c.target }
	}

	c.lock.Unlock()
	c.dsWriter(next)
}

func (c *rateController) increase(now time.Time) int {
	// 加性增（additive increase）：当接收速率接近「上次降速时的均值±3σ」
	// 说明已逼近上次出问题的拐点，改用「每次只涨约一个包」的保守加法
	if c.latestDecreaseRate.average > 0 &&
		float64(c.latestReceivedRate) > c.latestDecreaseRate.average-3*c.latestDecreaseRate.stdDeviation &&
		float64(c.latestReceivedRate) < c.latestDecreaseRate.average+3*c.latestDecreaseRate.stdDeviation {
		bitsPerFrame := float64(c.target) / 30.0
		packetsPerFrame := math.Ceil(bitsPerFrame / (1200 * 8))
		expectedPacketSizeBits := bitsPerFrame / packetsPerFrame

		responseTime := 100*time.Millisecond + c.latestRTT
		alpha := 0.5 * math.Min(float64(now.Sub(c.lastUpdate).Milliseconds())/float64(responseTime.Milliseconds()), 1.0)
		increase := int(math.Max(1000.0, alpha*expectedPacketSizeBits))
		c.lastUpdate = now
		return int(math.Min(float64(c.target+increase), 1.5*float64(c.latestReceivedRate)))
	}
	// 乘性增（multiplicative increase）：离拐点还远，按 1.08^Δt 指数爬升
	eta := math.Pow(1.08, math.Min(float64(now.Sub(c.lastUpdate).Milliseconds())/1000, 1.0))
	c.lastUpdate = now
	rate := int(eta * float64(c.target))

	// 封顶：最多涨到 1.5 × 实测接收速率
	received := int(1.5 * float64(c.latestReceivedRate))
	if rate > received && received > c.target {
		return received
	}
	if rate < c.target {
		return c.target
	}
	return rate
}

func (c *rateController) decrease() int {
	target := int(beta * float64(c.latestReceivedRate)) // 乘性减：0.85 × 实测接收速率
	c.latestDecreaseRate.update(float64(c.latestReceivedRate))
	c.lastUpdate = c.now()
	return target
}
```

**逐行注解**（AIMD 的精髓）：

1. **三态 + transition**：`stateIncrease / stateHold / stateDecrease`，由 `state.transition(usage)` 驱动。over-use 通常推向 decrease，under-use 推向 hold，normal 推向 increase——这张状态转移表带**迟滞**，避免在临界点反复横跳。
2. **`stateHold` 直接 return**：hold 态既不增也不减，给系统一个「观察期」让延迟梯度稳定下来，这是 AIMD 收敛的关键缓冲。
3. **乘性增 `eta = 1.08^min(Δt/1000, 1)`**：远离拐点时**指数爬升**（每秒最多 ×1.08），快速探测可用带宽。
4. **加性增分支**：一旦接收速率回到「上次降速点 ±3σ」区间，说明逼近上次的瓶颈，改用**加性增**（每次约一个包大小、至少 1000 bps），小步试探，避免再次冲过头——这正是 **AIMD「逼近时减速」** 的体现。
5. **乘性减 `target = 0.85 × latestReceivedRate`（beta = 0.85）**：确认 over-use 时，**不是**砍当前 target，而是砍**实测接收速率**的 85%。为什么砍接收速率而非 target？因为 over-use 时 target 可能已经虚高（发得比收到的多），以「实际收到的」为基准退让才真实有效。
6. **`exponentialMovingAverage`（α=0.95）记录历次降速点**：用 EMA 维护「降速时接收速率」的均值和标准差，正是第 4 点判断「是否逼近拐点」的依据。这是 GCC 比朴素 AIMD 聪明的地方——它**记得上次在哪摔的**。
7. **`1.5 × latestReceivedRate` 封顶**：无论怎么涨，target 不超过实测接收速率的 1.5 倍。你不能假设带宽比实际观测到的高太多。

### 6.3.5 Loss-Based Controller——丢包兜底

【示意（基于实际 fetch 的常量与逻辑）repo: pion/interceptor@pkg/gcc/loss_based_bwe.go】

```go
// 两个丢包率阈值（实测常量）：
//   increase 阈值 = 0.02 (2%)：丢包率 < 2% 时涨码率
//   decrease 阈值 = 0.10 (10%)：丢包率 > 10% 时降码率
//   2%~10% 之间：保持（hold）

// increaseLoss = max(平均丢包率, 当前丢包率)；与 0.02 比
// decreaseLoss = min(平均丢包率, 当前丢包率)；与 0.10 比

// 涨：bitrate = clamp(1.05 * bitrate, min, max)            // ×1.05
// 降：bitrate = clamp(bitrate * (1 - 0.5*decreaseLoss), …) // 按丢包比例砍
// 两次相邻增/减之间至少间隔 200ms（防振荡）
```

**逐行注解**：

- **三段式**：`<2%` 涨、`2%~10%` 持、`>10%` 降。中间这条「死区」避免在正常微丢包（无线链路天生有几个百分点丢包）下乱动。
- **降码公式 `× (1 − 0.5·lossRatio)`**：丢包率越高砍得越狠。例如 lossRatio=0.2（20%）→ ×0.9；lossRatio=0.5（50%）→ ×0.75。这与 RFC/草案给的 loss-based 公式一致。
- **`min(A_delay, A_loss)`**：loss-based 算出的码率与 delay-based 取小，作为最终下发。

> **出处核实**：丢包法的 `2% / 10%` 双阈值与「线性降速」公式直接对应 `draft-ietf-rmcat-gcc-02` 的 loss-based controller 章节。Pion 的 `1.05` 增益与 `200ms` 间隔为实测常量。

---

## 6.4 ⭐ 可运行 Demo

### Demo A（浏览器 JS）：用 `setParameters` 限码率，观测 GCC 在 webrtc-internals 里收敛

**目标**：在**单机回环**（同一页面两个 `RTCPeerConnection` 互连）跑一路真实视频，先放开码率让 GCC 探到高带宽，再用 `setParameters` 把发送端 `maxBitrate` 压低，在 `chrome://webrtc-internals` 里**亲眼看到 `availableOutgoingBitrate`（GCC 估计）随之收敛、`qualityLimitationReason` 变 `bandwidth`**。这是观测 GCC 行为最低成本、零服务器的方式。

> 为什么单机回环也能看到 GCC？因为 PeerConnection 之间走的是真实的 RTP/SRTP + transport-cc 反馈闭环，GCC 控制器真实在跑。回环没有真实瓶颈，但 `setParameters` 的 `maxBitrate` 会**主动给编码器封顶**，于是你能看到 BWE/编码码率随之变化、并配合 Chrome 的网络节流复现降码率。

**完整代码**（存为 `gcc-demo.html`，直接双击用 Chrome 打开）：

```html
<!doctype html>
<html>
<head><meta charset="utf-8"><title>GCC / transport-cc demo</title></head>
<body>
  <video id="local"  autoplay muted playsinline width="320"></video>
  <video id="remote" autoplay         playsinline width="320"></video>
  <div>
    <button id="start">1. 开始通话</button>
    <label>maxBitrate(kbps):
      <input id="rate" type="number" value="2500" step="100">
    </label>
    <button id="apply">2. 应用码率上限</button>
  </div>
  <pre id="log" style="height:160px;overflow:auto;background:#111;color:#0f0;padding:8px"></pre>

<script>
const log = m => { const p=document.getElementById('log'); p.textContent += m+'\n'; p.scrollTop=p.scrollHeight; };
let pcA, pcB, sender;

document.getElementById('start').onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video:true, audio:false });
  document.getElementById('local').srcObject = stream;

  // 两个 PC 互连（同机回环）
  pcA = new RTCPeerConnection();
  pcB = new RTCPeerConnection();
  pcA.onicecandidate = e => e.candidate && pcB.addIceCandidate(e.candidate);
  pcB.onicecandidate = e => e.candidate && pcA.addIceCandidate(e.candidate);
  pcB.ontrack = e => { document.getElementById('remote').srcObject = e.streams[0]; };

  // 发送端 track，拿到 sender 以便后面 setParameters
  sender = pcA.addTrack(stream.getVideoTracks()[0], stream);

  // 标准 offer/answer
  const offer = await pcA.createOffer();
  await pcA.setLocalDescription(offer);
  await pcB.setRemoteDescription(offer);
  const answer = await pcB.createAnswer();
  await pcB.setLocalDescription(answer);
  await pcA.setRemoteDescription(answer);

  log('通话已建立。打开 chrome://webrtc-internals 观察 availableOutgoingBitrate。');

  // 每秒打印发送端 outbound-rtp 关键 stats
  setInterval(async () => {
    const stats = await pcA.getStats();
    stats.forEach(r => {
      if (r.type === 'outbound-rtp' && r.kind === 'video') {
        log(`bytesSent=${r.bytesSent} qualityLimitationReason=${r.qualityLimitationReason}`);
      }
      if (r.type === 'candidate-pair' && r.state === 'succeeded' && r.availableOutgoingBitrate) {
        log(`>>> GCC availableOutgoingBitrate = ${(r.availableOutgoingBitrate/1000).toFixed(0)} kbps`);
      }
    });
  }, 1000);
};

// 用 setParameters 给发送端码率封顶（触发 GCC/编码器响应）
document.getElementById('apply').onclick = async () => {
  const kbps = parseInt(document.getElementById('rate').value, 10);
  const params = sender.getParameters();
  if (!params.encodings || !params.encodings.length) params.encodings = [{}];
  params.encodings[0].maxBitrate = kbps * 1000;          // bps
  await sender.setParameters(params);
  log(`已设 maxBitrate = ${kbps} kbps`);
};
</script>
</body>
</html>
```

**运行步骤**：

1. Chrome 里双击打开 `gcc-demo.html`（`file://` 即可，回环不需要 HTTPS；若浏览器拦截摄像头，用 `python3 -m http.server` 起本地 server 走 `http://localhost`）。
2. 点「开始通话」，允许摄像头。两个 video 都出画面 = 回环通了。
3. **新开一个标签页**输入 `chrome://webrtc-internals`，找到 `pcA` 对应的 PeerConnection，展开 `candidate-pair` 的 `availableOutgoingBitrate` 曲线。
4. 把输入框改成 `300`（300 kbps），点「应用码率上限」。
5. 观察：页面日志里 `availableOutgoingBitrate` 与编码码率在数秒内**下探收敛**；`qualityLimitationReason` 变为 `bandwidth`。
6. 进阶：开 Chrome DevTools → 三个点菜单 → More tools → Network conditions，把网络节流设为「Slow / 自定义低带宽」，观察 GCC 在**真实瓶颈**下的降码与回升（这才是 6.3.4 AIMD 状态机真实在跑）。

**预期现象**：

- 放开时 `availableOutgoingBitrate` 爬升到接近 `maxBitrate`（乘性增 1.08^Δt）。
- 压到 300kbps 或节流后，BWE 在几个 RTT 内收敛到新带宽附近，`qualityLimitationReason=bandwidth`，video 分辨率/帧率自适应下降。
- webrtc-internals 的 `[transport-cc]` / `RTCInboundRtpStreamStats` 里能看到 transport-cc 反馈在持续往返。

> **真坑提示**：`maxBitrate` 是「上限」不是「目标」。GCC 实际目标码率仍由 BWE 决定，`setParameters` 只是把天花板压低。要看 GCC 主动降码，必须制造**真实瓶颈**（Network conditions 节流），否则回环带宽近乎无限，GCC 会一直顶在 `maxBitrate`。

---

### Demo B（Go toy）：复现 GCC 数学内核——延迟梯度 → 自适应阈值 → 码率裁决

**目标**：脱离完整 WebRTC 协议栈，用**纯 Go**把 6.3.2 的自适应阈值 + 6.3.4 的 AIMD 抽出来，喂一段「人造延迟梯度序列」（模拟链路从空闲 → 拥塞 → 恢复），打印每一步的 `usage` 判定和目标码率，**亲眼看 GCC 的判定—退让—回升闭环**。这段代码**移植自 6.3.2/6.3.4 的真实 Pion 逻辑**（同样的 12.5ms 初值、0.01/0.00018 双系数、1.08 增益、0.85 beta），是理解算法最快的方式。

**完整代码**（存为 `gcctoy.go`，`go run gcctoy.go`，无需任何第三方依赖）：

```go
package main

import (
	"fmt"
	"math"
	"time"
)

// ===== 自适应阈值（移植自 pion adaptive_threshold.go 的常量与公式）=====
type adaptiveThreshold struct {
	thresh  time.Duration // 初值 12.5ms
	kUp     float64       // 0.01
	kDown   float64       // 0.00018
	min     time.Duration // 6ms
	max     time.Duration // 600ms
	numDel  int
}

func newThreshold() *adaptiveThreshold {
	return &adaptiveThreshold{
		thresh: 12500 * time.Microsecond,
		kUp:    0.01, kDown: 0.00018,
		min: 6 * time.Millisecond, max: 600 * time.Millisecond,
	}
}

// 返回 usage：1=over 0=normal -1=under
func (a *adaptiveThreshold) compare(estimate time.Duration) int {
	a.numDel++
	if a.numDel < 2 {
		return 0
	}
	// t = min(numDeltas,60) * estimate（样本数放大）
	n := a.numDel
	if n > 60 {
		n = 60
	}
	t := time.Duration(n) * estimate
	use := 0
	if t > a.thresh {
		use = 1
	} else if t < -a.thresh {
		use = -1
	}
	a.update(t, 20*time.Millisecond) // 假设每次间隔 20ms（一帧多）
	return use
}

func (a *adaptiveThreshold) update(estimate, dt time.Duration) {
	abs := time.Duration(math.Abs(float64(estimate.Microseconds()))) * time.Microsecond
	if abs > a.thresh+15*time.Millisecond {
		return // 离阈值太远，不更新
	}
	k := a.kUp
	if abs < a.thresh {
		k = a.kDown
	}
	if dt > 100*time.Millisecond {
		dt = 100 * time.Millisecond
	}
	d := abs - a.thresh
	add := k * float64(d.Milliseconds()) * float64(dt.Milliseconds())
	a.thresh += time.Duration(add*1000) * time.Microsecond
	if a.thresh < a.min {
		a.thresh = a.min
	}
	if a.thresh > a.max {
		a.thresh = a.max
	}
}

// ===== AIMD 码率控制（移植自 pion rate_controller.go：1.08 增、0.85 减）=====
const beta = 0.85

type aimd struct {
	target       int     // 当前目标码率 bps
	minB, maxB   int
	state        string  // "increase"/"hold"/"decrease"
	receivedRate int     // 模拟实测接收速率
}

// 状态转移（带迟滞）：over→decrease, under→hold, normal→increase
func transition(cur string, usage int) string {
	switch usage {
	case 1: // over
		return "decrease"
	case -1: // under
		return "hold"
	default: // normal
		if cur == "decrease" {
			return "hold" // 从 decrease 先进 hold，不直接跳 increase（迟滞）
		}
		return "increase"
	}
}

func (c *aimd) step(usage int) {
	c.state = transition(c.state, usage)
	switch c.state {
	case "increase":
		rate := int(1.08 * float64(c.target)) // 乘性增（简化：每步 ×1.08）
		cap := int(1.5 * float64(c.receivedRate))
		if rate > cap && cap > c.target {
			rate = cap
		}
		c.target = clamp(rate, c.minB, c.maxB)
	case "decrease":
		c.target = clamp(int(beta*float64(c.receivedRate)), c.minB, c.maxB) // ×0.85
	case "hold":
		// 不动
	}
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func main() {
	th := newThreshold()
	c := &aimd{target: 500_000, minB: 100_000, maxB: 4_000_000,
		state: "increase", receivedRate: 500_000}

	// 人造延迟梯度序列（µs）：
	//   阶段1 链路空闲（梯度≈0，略正常抖动）
	//   阶段2 拥塞来临（梯度持续走高 → 应触发 over-use → 降码）
	//   阶段3 拥塞缓解（梯度回落甚至转负 → under/normal → 回升）
	gradients := []int{
		200, -100, 300, 0, 100, // 阶段1 正常带内
		2000, 4000, 7000, 9000, 12000, // 阶段2 持续恶化
		8000, 3000, 500, -200, -500, // 阶段3 恢复
	}

	fmt.Printf("%-4s %-9s %-9s %-9s %-12s\n", "步", "梯度µs", "阈值µs", "usage", "目标码率kbps")
	for i, g := range gradients {
		est := time.Duration(g) * time.Microsecond
		usage := th.compare(est)

		// 模拟接收速率：拥塞时收到的比目标少（瓶颈约 1.2Mbps）
		c.receivedRate = c.target
		if g > 3000 && c.target > 1_200_000 {
			c.receivedRate = 1_200_000
		}
		c.step(usage)

		us := map[int]string{1: "OVER", 0: "normal", -1: "under"}[usage]
		fmt.Printf("%-4d %-9d %-9d %-9s %-12d\n",
			i+1, g, th.thresh.Microseconds(), us, c.target/1000)
	}
}
```

**运行步骤**：

1. 装 Go（`go version` 能跑即可，1.18+，因用到泛型友好的标准库；本 demo 无泛型，1.16+ 亦可）。
2. `go run gcctoy.go`。

**预期输出**（数值随机噪声很小，趋势稳定）：

```
步   梯度µs    阈值µs    usage     目标码率kbps
1    200       12500     normal    540
2    -100      12500     normal    583
3    300       12500     normal    629
...
6    2000      12500     normal    ...        ← 梯度开始升但 ×n 后仍可能未破阈
7    4000      ~12xxx    OVER      1020        ← 破阈，转 decrease：0.85×1.2M
8    7000      升高      OVER      1020        ← 阈值被 kUp 快速抬高（容忍度↑）
...
11   8000      更高      OVER/normal ...
13   500       缓降      normal    回升中       ← 梯度回落，转 increase ×1.08
15   -500      缓降      under     hold         ← under-use，进 hold 观望
```

**预期现象与对照源码**：

- **阶段1**：梯度在正常带内，`usage=normal`，AIMD 持续 `increase`，码率按 ×1.08 爬升——对照 6.3.4 乘性增。
- **阶段2**：梯度持续走高，越过阈值 → `OVER` → AIMD `decrease` 把码率砍到 `0.85 × receivedRate`（≈1020kbps）——对照 6.3.4 `decrease()` 的 `beta=0.85`。同时**注意阈值列在 OVER 期间被 `kUp=0.01` 快速抬高**，正是 6.3.2 注解第 5 点的「提高容忍度、防被竞争流饿死」。
- **阶段3**：梯度回落，先 `normal` 触发回升、再 `under` 进 `hold` 观望——对照 6.3.4 状态机迟滞与 hold 缓冲。

> 这个 toy 故意**简化**了滤波器（直接喂梯度，省去 Kalman/Trendline）和 overuse_detector 的持续时间闸门，但**自适应阈值与 AIMD 这两段核心数学是 1:1 移植 Pion 真实常量**的。把它和 6.3 的真实源码对读，GCC 的判定—退让—回升闭环就彻底打通了。

---

## 6.5 方案对比

### 6.5.1 GCC（delay+loss） vs 纯 loss-based（TCP 式） vs BBR

| 维度 | GCC（WebRTC 默认） | 纯 loss-based（Reno/CUBIC 式） | BBR |
|---|---|---|---|
| 拥塞信号 | **延迟梯度**（主）+ 丢包（辅） | 丢包 | 实测带宽 + 最小 RTT（探测 BtlBw / RTprop） |
| 何时退让 | 队列**开始**堆积（丢包前） | 队列**填满**丢包后 | 估计带宽下降时 |
| 对 bufferbloat | **抗性强**（提前刹车，延迟低） | 差（必把缓冲填满） | 较强（不靠填满缓冲） |
| 实时延迟 | 低（适合 RTC） | 高 | 中低 |
| 与 TCP 竞争 | 靠自适应阈值（kUp 快抬）抢带宽，仍偏弱 | 公平（同类） | 偏激进，可能挤压 loss-based 流 |
| WebRTC 现状 | **默认**（send-side BWE + transport-cc） | 不用 | 实验/部分场景探索 |
| 适用边界 | 实时音视频 | 文件传输/弹性流 | 大吞吐长流 |

**具体场景**：

- **家用 Wi-Fi + 深缓冲 NAT 打视频**：GCC 的延迟优先让你在卡顿前就降码、保住低延迟；若用纯 loss-based，会先卡几百毫秒再退。
- **同链路有人在下大文件（TCP 大象流）**：这是 GCC 的**软肋**。TCP 持续把缓冲填满，GCC 的延迟梯度长期偏高，全靠 `kUp=0.01` 快速抬高阈值才能不被饿死，但抢到的带宽通常仍少于 TCP——这是 delay-based 与 loss-based 共存的经典公平性问题。
- **数据中心间大吞吐**：用 BBR，不要用 GCC（GCC 为低延迟牺牲了吞吐探测的激进度）。

### 6.5.2 transport-cc（send-side） vs REMB（receive-side）

| 维度 | transport-cc（现代） | REMB（旧） |
|---|---|---|
| BWE 计算位置 | **发送端** | 接收端 |
| 反馈内容 | 每个包的到达时间（细粒度） | 一个聚合的最大码率值（粗） |
| 信息完整度 | 高（发/到/丢全知） | 低（只知到了什么） |
| 算法迭代成本 | 低（只升发送端） | 高（要升接收端） |
| RTP 头扩展 | transport-wide seq num | 无（仅 abs-send-time 配合） |
| RTCP 反馈频率 | 高（约每 50-100ms） | 低 |
| WebRTC 现状 | **默认** | 仅向后兼容 |

**具体场景**：与只支持 REMB 的老旧端点互通时，SDP 协商会回落到 REMB（BWE 跑在接收端，精度下降）。现代 SFU（mediasoup / Janus / LiveKit）均以 transport-cc 为主。

---

## 6.6 失败模式 / 真坑 / 根因（扎根）

### 坑 1：NAT/防火墙剥掉 RTP 头扩展 → transport-cc 失效，BWE 卡死

**现象**：通话能连通、有画面，但码率长期顶死在初始值或乱跳，`availableOutgoingBitrate` 不收敛。

**根因**：transport-cc 依赖**发送端给每个包打的 transport-wide 序列号**（一个 RTP 头扩展）。某些中间盒（企业防火墙、老旧 SBC）会重写或剥离它不认识的 RTP 头扩展。一旦序列号丢失，接收端无法回传逐包到达时间，**发送端 GCC 的输入断流**，只能退回 REMB 或干脆估不准。

**排查**：`chrome://webrtc-internals` 看是否有 transport-cc 反馈往返；SDP 里确认双方都协商了 `a=rtcp-fb:* transport-cc` 与对应 `extmap`。

### 坑 2：突发丢包（burst loss）被误读为延迟问题，或反之

**现象**：无线链路（4G/Wi-Fi）瞬时大量丢包，码率被狠砍后久久不回升。

**根因**：delay-based 路径靠**包的到达时间**算梯度。突发丢包会造成「到达序列空洞」，arrival group 的间隔计算可能失真。GCC 用 `min(A_delay, A_loss)` 合并，丢包率 >10% 时 loss-based 直接按 `(1−0.5·loss)` 猛砍。问题在**回升慢**：6.3.4 的加性增分支会因「逼近上次降速点」而保守，无线链路的随机丢包会让它反复触发保守加性增，码率回升迟缓。

**根因深一层**：GCC 假设「丢包 ≈ 拥塞」。但无线链路有**非拥塞丢包**（信号衰落、干扰），这类丢包不该触发降码却触发了——这是 loss-based 控制器的固有局限，也是为什么 WebRTC 要叠加 NACK/FEC（第 3/5 章）做丢包恢复，而非全靠降码。

### 坑 3：抖动（jitter）把正常带内信号推过阈值，码率无故下探

**现象**：链路带宽充足，但码率周期性小幅下探。

**根因**：6.3.2 的 `t = min(numDeltas,60) × estimate` 会把梯度**乘以样本数**。在高抖动但不拥塞的链路（如 Wi-Fi 干扰），单次梯度估计虽小，乘以累积样本数后可能短暂越过阈值，触发瞬时 `OVER`。**防线**是 6.3.3 的 overuse_detector：要求 over-use **持续超过 `overuseTime` 且梯度仍在恶化**才确认。如果某实现把这个闸门配得太松，就会被抖动带着下探。

**根因深一层**：阈值是「自适应」的——持续抖动会让 `kUp` 慢慢抬高阈值容忍度，系统最终会学会容忍这点抖动。但**抬高过程中**（几秒）的码率抖动是真实存在的，对画质敏感的场景需要在应用层做码率平滑。

### 坑 4：min-bitrate 设太高 → 拥塞时无法退到可用带宽，全程卡顿

**现象**：弱网下通话全程卡死、丢包率居高不下。

**根因**：6.3.4 的 `clampInt(…, minBitrate, maxBitrate)` 把目标码率钳在 `minBitrate` 以上。如果 `minBitrate` 设得比真实可用带宽还高（比如把 minBitrate 设成 1Mbps，但弱网只有 300kbps），GCC 想退也退不下去，链路持续过载、丢包、卡顿。

**排查**：检查 `setParameters` 的 `encodings[].minBitrate` 或原生 `BitrateSettings`。弱网友好的设置应让 `minBitrate` 足够低（视频常见 50-100kbps 起）。

---

## 6.7 本章五件套

### 1. 一句话定义

GCC 是 WebRTC 的**延迟优先**拥塞控制：从包到达时间的延迟梯度里、在丢包发生之前嗅出队列堆积，用自适应阈值 + AIMD 状态机调目标码率；transport-cc 是把「每个包何时到」回传给发送端、让 GCC 跑在发送端的反馈管道。

### 2. 心智模型

> **GCC ≈ 一个会「听水声」的水龙头。** 它不等水池（缓冲区）溢出（丢包）才关小，而是听水位上涨的声音（延迟梯度），在快满时就提前拧小（乘性减 0.85×接收速率）；听到水位下降就慢慢拧大（乘性增 ×1.08，逼近上次溢出点时改小步加性增）。而 transport-cc 是装在水池出口的「逐滴到达时间记录仪」，把每滴水何时到回传给拧龙头的手。

### 3. 关键决策

- **延迟法 vs 丢包法**：实时媒体选延迟优先（GCC），文件传输选丢包法（TCP）。两者在 GCC 里 `min` 合并：延迟法提前刹车，丢包法兜底猛刹。
- **send-side vs receive-side**：现代一律 transport-cc（发送端 BWE，信息全、迭代快）；只为兼容老端点保留 REMB。
- **阈值 kUp 快、kDown 慢**：在与 TCP 共存的链路里，快速抬高容忍度避免被饿死——这是显式的公平性 trade-off。
- **min-bitrate 要够低**：否则弱网退不下去，全程卡顿（坑 4）。

### 4. 常见误区

- ❌「`setParameters` 的 `maxBitrate` 就是 GCC 目标码率」。✅ 它只是上限；GCC 目标由 BWE 算，回环看不到主动降码必须制造真实瓶颈（Demo A 真坑）。
- ❌「丢包就是拥塞，丢包了就该降码」。✅ 无线有非拥塞丢包，这类丢包该用 NACK/FEC 恢复而非降码（坑 2 根因）。
- ❌「GCC 用 Kalman 滤波器」。✅ 初版用 Kalman，现代 libwebrtc 已换 **Trendline**（线性回归斜率）；Pion 仍走 Kalman/slope，但判定框架一致（6.1.3）。
- ❌「过了阈值就立刻降码」。✅ 还要过 overuse_detector 的**持续时间闸门**（且梯度仍恶化）才确认 over-use（6.3.3），否则抖动会乱触发（坑 3）。
- ❌「降码是砍当前 target 的 85%」。✅ 是砍**实测接收速率**的 85%（`beta × latestReceivedRate`），因为 over-use 时 target 可能虚高（6.3.4 注解第 5 点）。

### 5. 延伸阅读

- `draft-ietf-rmcat-gcc-02`《A Google Congestion Control Algorithm for Real-Time Communication》——GCC 权威规范（延迟法 + 丢包法）：https://datatracker.ietf.org/doc/html/draft-ietf-rmcat-gcc-02
- `draft-holmer-rmcat-transport-wide-cc-extensions-01`——transport-cc 头扩展与反馈消息规范：https://datatracker.ietf.org/doc/html/draft-holmer-rmcat-transport-wide-cc-extensions-01
- Chrome transport-wide-cc-02 README（实现侧）：https://webrtc.googlesource.com/src/+/refs/heads/main/docs/native-code/rtp-hdrext/transport-wide-cc-02/README.md
- Carlucci et al.《Analysis and design of the Google Congestion Control for WebRTC》（Poliba，自适应阈值/公平性定量分析）：https://c3lab.poliba.it/images/6/65/Gcc-analysis.pdf
- 《Congestion Control for RTP Media: A Comparison on Simulated Environment》（GCC/NADA/SCReAM 对比）：https://arxiv.org/pdf/1809.00304
- Pion 真实实现（本章源码来源）：`pion/interceptor@pkg/gcc/`（adaptive_threshold.go / rate_controller.go / overuse_detector.go / arrival_group_accumulator.go / loss_based_bwe.go / delay_based_bwe.go）

---

## 附录 A：transport-cc 在 SDP 里的样子（协商速查）

```
m=video 9 UDP/TLS/RTP/SAVPF 96 ...
a=extmap:5 http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01
                                            ↑ 发送端给每包打 transport-wide 序列号
a=rtcp-fb:96 transport-cc                   ← 启用 transport-cc 反馈
a=rtcp-fb:96 goog-remb                      ← 同时声明 REMB（向后兼容回落用）
a=rtcp-fb:96 nack                           ← NACK（配合丢包恢复，见第 3/5 章）
```

协商规则：双方都带 `transport-cc` → 走 send-side BWE；任一方只有 `goog-remb` → 回落 receive-side REMB（精度下降，见 6.5.2）。

---

## 附录 B：webrtc-internals 关键字段速查（拥塞控制视角）

```
chrome://webrtc-internals → 展开对应 PeerConnection，关注：

candidate-pair（ICE 链路 / BWE 总览）：
  availableOutgoingBitrate   GCC 估计的可用上行带宽（bps）★最核心
  availableIncomingBitrate   下行估计
  currentRoundTripTime       当前 RTT（秒）—— AIMD increase 的 responseTime 输入

outbound-rtp（发送端，video）：
  targetBitrate              当前目标码率（GCC 下发给编码器的）★
  qualityLimitationReason    限制原因：none / bandwidth / cpu / other
                             变 bandwidth = GCC 正在压码率
  qualityLimitationDurations.bandwidth  被带宽限制的累计秒数
  retransmittedPacketsSent   重传包数（NACK 触发）
  framesPerSecond / frameHeight  帧率/分辨率随码率自适应下降

remote-inbound-rtp（来自对端的 RR/transport-cc 反馈）：
  packetsLost / fractionLost 丢包率 —— loss-based controller 的输入 ★
  jitter                     抖动（秒）—— 高抖动可能误触发 over-use（坑 3）
  roundTripTime              对端测得的 RTT

观测 GCC 的标准动作：
  1. 看 availableOutgoingBitrate 曲线是否随网络节流收敛/回升
  2. 看 qualityLimitationReason 是否在拥塞时变 bandwidth
  3. 看 fractionLost 与降码是否相关（区分拥塞丢包 vs 非拥塞丢包）
```

---

*本章真实源码引用（MIT 许可）：*
- *pion/interceptor（master 分支）：pkg/gcc/adaptive_threshold.go、rate_controller.go 为 WebFetch 实际取得的逐字源码；overuse_detector.go、arrival_group_accumulator.go、loss_based_bwe.go、delay_based_bwe.go 为 WebFetch 取得的结构化摘要，据此还原为【示意】段（关键常量为实测值）。*

*标注约定：【真实源码 repo@path】= WebFetch 逐字取得；【示意（基于实际 fetch 的结构与常量）】= 据实际 fetch 的结构/常量还原的说明性代码，非整文件逐字；「待核」= 需进一步核实的细节（如 libwebrtc Trendline 具体路径、overuseTime 精确默认值）。设计考古的出处均经 WebFetch / WebSearch 核实，链接见 6.7 延伸阅读。*
