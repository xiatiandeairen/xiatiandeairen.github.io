---
title: "SFU / MCU 架构（WebRTC 域）"
slug: "10-07"
collection: "tech-library"
group: "webrtc"
order: 10007
summary: "TL;DR 多人实时音视频的拓扑只有三种：Mesh（全连接 P2P）、MCU（混流）、SFU（选择性转发）。Mesh 上行带宽 O(N) 在 4~5 人就爆；MCU 把 N 路解码混合再编码成 1 路，省带宽但烧 CPU 且引入转码延迟、破坏端到端加密；SFU 既不解码也不混流，只做 RTP 包级转…"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-15T14:41:39.000Z"
updatedAt: "2026-06-15T14:41:39.000Z"
---
> **TL;DR**
> 多人实时音视频的拓扑只有三种：**Mesh（全连接 P2P）**、**MCU（混流）**、**SFU（选择性转发）**。Mesh 上行带宽 O(N) 在 4~5 人就爆；MCU 把 N 路解码混合再编码成 1 路，省带宽但烧 CPU 且引入转码延迟、破坏端到端加密；SFU 既不解码也不混流，只做 **RTP 包级转发 + 头部重写（SSRC/SN/TS）**，是今天 95% 多人会议和直播的事实标准。
> 本章源码级拆 SFU 的核心循环：一个 inbound track（Receiver）如何 fan-out 到 N 个 outbound track（DownTrack），每个 DownTrack 如何把同一份 payload 重写成订阅者认得的 SSRC/序列号/时间戳。配 **两个能真跑的 demo**（浏览器纯 JS Mesh 对照 + Pion 最小 SFU 多人转发），并用 `chrome://webrtc-internals` 观测转发链路。最后扎根到三大真坑：**keyframe 饥饿（新订阅者黑屏/花屏）**、**带宽估计不能复用**、**级联（cascading）与单点瓶颈**。
>
> **前置依赖**：第 1 章（信令/SDP/Offer-Answer）、第 3 章（RTP/SRTP，SSRC/序列号/时间戳语义）、第 5 章（视频编码、keyframe/IDR、simulcast）、第 6 章（GCC/transport-cc 拥塞控制）。本章假设你已经知道「一个 RTP 包长什么样」「SSRC 是流的身份」「keyframe 之后才能解码」。

---

## 1. 设计考古：为什么会有 SFU 这个东西

### 1.1 问题的原点——一对一不需要服务器，多人才需要

WebRTC 的初心是 **P2P**：两个浏览器通过 ICE 打洞（见第 2 章）直接连起来，媒体不经过服务器。一对一通话这套很完美——延迟最低、服务器零成本、天然端到端加密。

问题在 **N 人**。一旦 N≥3，"每个人都要看到/听到其他所有人"这个需求，在不同拓扑下成本天差地别。

### 1.2 三种拓扑的演化

**第一代：Mesh（全连接 P2P）。**
最直接的扩展：N 个人，两两之间建 PeerConnection，共 `N*(N-1)/2` 条连接。每个客户端要 **上行 N-1 份自己的编码流**，同时 **下行 N-1 份别人的流**。

致命问题是 **上行带宽和编码 CPU**。家用网络上行普遍只有几 Mbps，发一路 720p 约 1.5~2.5 Mbps。4 个人时每人要上行 3 路 = 4.5~7.5 Mbps，已经顶满；而且摄像头的同一帧要被编码器编 N-1 次（或一次编码多次打包，但 SRTP 加密仍要做 N-1 次）。所以 Mesh 的现实天花板是 **4~5 人**。

> **史料核实**【bloggeek.me/webrtcglossary/sfu，Tsahi Levent-Levi】给出的拓扑对照表（原文措辞）：
>
> | Aspect | Mesh | SFU | MCU |
> |---|---|---|---|
> | Server Cost | Zero | Low-medium | High |
> | Uplink Bandwidth | High (N-1 streams) | Low (single stream) | Low |
> | Sender CPU | High | Low-medium | Low |
> | Receiver CPU | High | Low-medium | Low |
> | Scalability | 2-5 users | Hundreds-thousands | Low tens |
>
> 原文结论（verbatim）：*"SFUs are the most common media server architecture today when implementing large group meetings and live streaming services. The reason for that is that it gives the best return on investment."*

**第二代：MCU（Multipoint Control Unit，混流）。**
传统视频会议（H.323/SIP 时代，Polycom/Cisco 那套）的方案：服务器把 N 路流 **全部解码**，在像素域/PCM 域 **合成一路**（比如九宫格画面、混音），再 **重新编码** 成 1 路发给每个参会者。

客户端侧极度简单——永远只收 1 路、发 1 路，CPU 和带宽都最低，甚至能兼容只会收单流的老设备/电话网关。代价在服务器：
- **CPU 爆炸**：N 路解码 + 混合 + 重编码，是计算最重的方案。一台服务器撑不了几十路。
- **转码延迟**：decode→mix→encode 这条流水线天然引入几十~上百 ms 延迟。
- **破坏端到端加密**：服务器必须看到明文像素才能混流，E2EE 无从谈起。
- **画面不可定制**：所有人看到同一个混好的版面，客户端没法自己排版（谁大谁小、谁 pin 到主屏）。

MCU 的现实天花板（bloggeek 表）是 **low tens**（几十人）量级，且服务器成本 High。

**第三代：SFU（Selective Forwarding Unit，选择性转发）——本章主角。**
关键洞察：**服务器根本不需要看懂媒体内容，只要做包级转发**。

SFU 的工作是：
1. 每个客户端 **上行 1 路** 自己的流给 SFU（不是 N-1 路！）。
2. SFU 收到这路 RTP 包后，**原样（payload 不动）转发** 给所有订阅了这路流的其他客户端。
3. 客户端 **下行 N-1 路**（每个其他参会者一路），各自独立解码、自己排版。

> **史料核实**【mediasoup.org/documentation，及 bloggeek 同篇】对 SFU 的定义（原文措辞）：*"A media server component that receives multiple streams and decides which streams to forward to which participants, without processing the media itself."* —— 核心是 **decides which to forward** + **without processing the media**。

SFU 的得失：
- **服务器 CPU 低**：不解码、不编码，只做 RTP 头改写 + SRTP 解密/重加密（或 passthrough）。一台机器能撑数百到上千路。
- **上行省**：客户端只上行 1 路，和 MCU 一样省。
- **下行不省**：客户端下行 N-1 路，比 MCU 多——这是 SFU 把 MCU 的「服务器 CPU 成本」换成了「客户端下行带宽成本」。但配合 **simulcast / SVC**（见第 5 章和 §6），SFU 能为不同订阅者选不同清晰度的层，把下行压下来。
- **保住 E2EE 的可能**：SFU 不碰 payload，配合 **Insertable Streams / SFrame** 可以做端到端加密（SFU 只看 RTP 头、看不到明文媒体）。这是 MCU 永远做不到的。
- **客户端可定制版面**：每路流独立，谁大谁小客户端说了算。

**为什么工业界从 MCU 倒向 SFU**：bloggeek 的原话是 SFU "consume considerably less CPU than their MCU alternative" 且 scalability 是 hundreds-thousands。本质是 **服务器算力（贵、难扩）换客户端带宽（便宜、用户自己承担）** 的经济账——加上保留 E2EE 和客户端自由排版这两个 MCU 给不了的东西。Zoom、Google Meet、Teams、几乎所有现代 WebRTC 多人方案的媒体层都是 SFU（或 SFU 为主、MCU 为补充的混合）。

> **本节小结的一句话**：Mesh 把成本压在客户端上行，MCU 把成本压在服务器 CPU，SFU 把成本压在客户端下行——而下行带宽是三者里最容易用 simulcast/SVC 优化、且对用户最便宜的资源。

---

## 2. SFU 的核心抽象：Receiver（入流）↔ DownTrack（出流）

理解 SFU 源码，先建立这张心智图。以 `pion/ion-sfu` 为例，一个房间里：

```
                          ┌─────────────────── SFU (一个进程) ───────────────────┐
  Alice ── PeerConn ──►   │  Receiver(Alice.video)                                │
  (上行1路)               │      │  fan-out                                       │
                          │      ├──► DownTrack(给Bob)   ──► PeerConn ──► Bob 下行 │
                          │      └──► DownTrack(给Carol) ──► PeerConn ──► Carol 下行│
                          │                                                        │
  Bob ──── PeerConn ──►   │  Receiver(Bob.video)                                   │
  (上行1路)               │      ├──► DownTrack(给Alice)                           │
                          │      └──► DownTrack(给Carol)                           │
                          └────────────────────────────────────────────────────────┘
```

两个核心对象：

- **Receiver（接收器）**：对应「一个上行 track」。Alice 推一路视频上来，SFU 为它建一个 Receiver。Receiver 内部有一个 **buffer**（抖动/重排/NACK 缓冲，见第 6 章），它从 buffer 读出有序的 RTP 包，然后 **fan-out 给挂在它身上的所有 DownTrack**。
- **DownTrack（下行轨）**：对应「把某个上行 track 发给某个订阅者」的一条出边。Alice 的视频要发给 Bob 和 Carol，就有两个 DownTrack。**DownTrack 的核心职责是 RTP 头改写**：把 Alice 流的 SSRC/序列号/时间戳改写成「Bob 这条 PeerConnection 上协商出来的、Bob 认得的」那一套。

> **为什么必须改写头部？** 这是 SFU 最反直觉、也最容易踩坑的点。每条 PeerConnection 的每个 track 在 SDP 协商时绑定了自己的 SSRC、自己的序列号空间、自己的时间戳基准。SFU 把 Alice 的包转给 Bob 时，**不能把 Alice 的 SSRC 直接塞给 Bob**——Bob 的 RTP 栈期望的是协商好的那个 SSRC。而且当订阅者从「看 Alice 的高清层」切到「看 Alice 的低清层」（simulcast 切层）时，源 SSRC 变了，但 Bob 那侧必须感知不到中断：**序列号要连续、时间戳要单调递增**，否则 Bob 的解码器/jitter buffer 会卡顿、花屏甚至重置。改写头部就是为了把「源侧的多变」隐藏成「订阅侧的一条连续流」。

### 2.1 真实源码：Router 与 DownTrack 的结构体

【真实源码 pion/ion-sfu@pkg/sfu/router.go】Router 接口与实现结构体（verbatim）：

```go
type Router interface {
	ID() string
	AddReceiver(receiver *webrtc.RTPReceiver, track *webrtc.TrackRemote, trackID, streamID string) (Receiver, bool)
	AddDownTracks(s *Subscriber, r Receiver) error
	SetRTCPWriter(func([]rtcp.Packet) error)
	AddDownTrack(s *Subscriber, r Receiver) (*DownTrack, error)
	Stop()
	GetReceiver() map[string]Receiver
	OnAddReceiverTrack(f func(receiver Receiver))
	OnDelReceiverTrack(f func(receiver Receiver))
}

type router struct {
	sync.RWMutex
	id            string
	twcc          *twcc.Responder
	stats         map[uint32]*stats.Stream
	rtcpCh        chan []rtcp.Packet
	stopCh        chan struct{}
	config        RouterConfig
	session       Session
	receivers     map[string]Receiver
	bufferFactory *buffer.Factory
	writeRTCP     func([]rtcp.Packet) error
	onAddTrack    atomic.Value
	onDelTrack    atomic.Value
}
```

**逐行注解（关键字段）**：
- `receivers map[string]Receiver`：房间里所有上行流，key 是 trackID。这是 fan-out 的「源」集合。
- `twcc *twcc.Responder`：transport-wide congestion control 的响应器（见第 6 章）。SFU 必须代为回 transport-cc 反馈，因为它是 RTP 包的「中转的接收端」。
- `rtcpCh chan []rtcp.Packet` + `writeRTCP`：RTCP（PLI/NACK/REMB/SR-RR）的回传通道。SFU 是双向的——除了转发 RTP，还要把订阅侧的反馈（如 PLI 请求关键帧）路由回发布侧。这条 RTCP 回路是 §7.1「keyframe 饥饿」问题的关键。
- `bufferFactory *buffer.Factory`：为每个上行流创建 jitter/NACK buffer。

DownTrack 的字段（【示意，非逐字】，依据 WebFetch 摘要重述，精确字段名以源码为准）：

```go
// DownTrack：一条「把某上行流发给某订阅者」的出边。
type DownTrack struct {
    id, peerID, streamID string   // 身份
    ssrc        uint32            // 【订阅侧】协商出的 SSRC —— 改写的目标值
    payloadType uint8             // 【订阅侧】协商出的 payload type
    snOffset    uint16            // 序列号偏移：newSN = srcSN - snOffset
    tsOffset    uint32            // 时间戳偏移：newTS = srcTS - tsOffset
    lastSSRC    uint32            // 上一个源 SSRC（检测切层/换源）
    lastSN      uint16            // 上一个改写后的序列号（保证连续）
    lastTS      uint32            // 上一个改写后的时间戳（保证单调）
    enabled, bound, reSync atomicBool // 启用/已绑定/需重新同步
    currentSpatialLayer, targetSpatialLayer int32 // simulcast 当前/目标层
    sequencer   *sequencer        // 源SN → 改写SN 的映射表（用于 NACK 重传时找回原包）
    // ... octetCount/packetCount 用于生成 Sender Report
}
```

> 注：实际 ion-sfu 用 `atomicBool` 封装并发标志，字段顺序/类型可能随版本变化；上面字段名与语义来自 WebFetch 对 `downtrack.go` 的精读摘要，**精确定义以对应 commit 的源码为准（待核到具体行）**。

---

## 3. ⭐ 核心源码精读：fan-out 循环（一进 N 出）

这是 SFU 的心脏。【真实源码 pion/ion-sfu@pkg/sfu/receiver.go】`WebRTCReceiver.writeRTP`（verbatim）：

```go
func (w *WebRTCReceiver) writeRTP(layer int) {
	defer func() {
		w.closeOnce.Do(func() {
			w.closed.set(true)
			w.closeTracks()
		})
	}()

	pli := []rtcp.Packet{
		&rtcp.PictureLossIndication{SenderSSRC: rand.Uint32(), MediaSSRC: w.SSRC(layer)},
	}

	for {
		pkt, err := w.buffers[layer].ReadExtended()
		if err == io.EOF {
			return
		}

		if w.isSimulcast {
			if w.pending[layer].get() {
				if pkt.KeyFrame {
					w.Lock()
					for idx, dt := range w.pendingTracks[layer] {
						w.deleteDownTrack(dt.CurrentSpatialLayer(), dt.id)
						w.storeDownTrack(layer, dt)
						dt.SwitchSpatialLayerDone(int32(layer))
						w.pendingTracks[layer][idx] = nil
					}
					w.pendingTracks[layer] = w.pendingTracks[layer][:0]
					w.pending[layer].set(false)
					w.Unlock()
				} else {
					w.SendRTCP(pli)
				}
			}
		}

		for _, dt := range w.downTracks[layer].Load().([]*DownTrack) {
			if err = dt.WriteRTP(pkt, layer); err != nil {
				if err == io.EOF || err == io.ErrClosedPipe {
					w.Lock()
					w.deleteDownTrack(layer, dt.id)
					w.Unlock()
				}
				Logger.Error(err, "Error writing to down track", "id", dt.id)
			}
		}
	}
}
```

**逐行精读**：

- `pli := []rtcp.Packet{&rtcp.PictureLossIndication{...MediaSSRC: w.SSRC(layer)}}`
  预构造一个 **PLI（Picture Loss Indication）** 包模板。PLI 是 RTCP 反馈，含义是「我（接收端）丢失了关键帧/无法解码，请发送端立刻产生一个 keyframe」。`MediaSSRC` 指向出问题的那一层的源 SSRC。**这是 §7.1 keyframe 饥饿的解药埋点**。

- `pkt, err := w.buffers[layer].ReadExtended()`
  从该层的 buffer 读出 **下一个有序、去重、已补 NACK 的 ExtPacket**。注意：fan-out 是从 buffer 读、不是从网络直接读——所有 jitter/重排/NACK 修复在 buffer 层做掉，fan-out 拿到的已经是干净有序的流。这是「**入流处理一次，出流复用 N 次**」的关键解耦：N 个订阅者不必各自做 NACK。

- `if w.isSimulcast { if w.pending[layer].get() { ... } }`
  **simulcast 切层逻辑**。当某个 DownTrack 请求从低清层切到本层（layer），它被放进 `pendingTracks[layer]`，`pending[layer]` 置位。切层 **必须等到一个 keyframe** 才能完成——因为订阅者在新层上没有参考帧，从非关键帧开始解码就是花屏。
  - `if pkt.KeyFrame { ... 完成切换：把 pending track 移到本层、调 SwitchSpatialLayerDone ... }`
    等到 keyframe，才把这些 track 正式挂到本层，从此它们收本层的包。
  - `else { w.SendRTCP(pli) }`
    还没等到 keyframe？**主动发 PLI 催发布端出关键帧**。这就是 SFU「切层不黑屏」的核心机制：等 keyframe + 催 keyframe 双管齐下。

- `for _, dt := range w.downTracks[layer].Load().([]*DownTrack) { dt.WriteRTP(pkt, layer) }`
  **fan-out 的本体**：遍历挂在本层上的所有 DownTrack，把 **同一个 pkt** 交给每个 DownTrack 去写。注意 `downTracks` 用 `atomic.Value` 存（`.Load()`），读路径无锁——因为 fan-out 是热路径，每个 RTP 包都跑一遍，不能让锁成为瓶颈。增删 DownTrack 时才加锁替换整个 slice（copy-on-write）。
  - 写失败且是 `io.EOF`/`ErrClosedPipe`（订阅者断了）→ 加锁删除该 DownTrack。这是 SFU 的连接生命周期管理：订阅者掉线，对应出边自动清理，不影响其他订阅者。

> **这段代码回答了「SFU 凭什么省」**：发布端的一个 RTP 包，进 buffer 一次（一次 NACK/重排），fan-out 时只是 **遍历指针 + 改头部**，没有任何解码/编码/混流。N 个订阅者 = N 次「读指针 + 写头」，是 O(N) 的轻量操作。对比 MCU 的「N 次解码 + 1 次混合 + 1 次编码」，CPU 差了几个数量级。

---

## 4. ⭐ 核心源码精读：DownTrack 的头部改写（同一份 payload，N 套头）

fan-out 把同一个 `pkt` 交给每个 DownTrack 后，每个 DownTrack 各自改写头部。【真实源码 pion/ion-sfu@pkg/sfu/downtrack.go】`writeSimpleRTP`（verbatim，非 simulcast 的简单转发路径）：

```go
func (d *DownTrack) writeSimpleRTP(extPkt *buffer.ExtPacket) error {
	if d.reSync.get() {
		if d.Kind() == webrtc.RTPCodecTypeVideo {
			if !extPkt.KeyFrame {
				d.receiver.SendRTCP([]rtcp.Packet{
					&rtcp.PictureLossIndication{SenderSSRC: d.ssrc, MediaSSRC: extPkt.Packet.SSRC},
				})
				return nil
			}
		}

		if d.lastSN != 0 {
			d.snOffset = extPkt.Packet.SequenceNumber - d.lastSN - 1
			d.tsOffset = extPkt.Packet.Timestamp - d.lastTS - 1
		}
		atomic.StoreUint32(&d.lastSSRC, extPkt.Packet.SSRC)
		d.reSync.set(false)
	}

	d.UpdateStats(uint32(len(extPkt.Packet.Payload)))

	newSN := extPkt.Packet.SequenceNumber - d.snOffset
	newTS := extPkt.Packet.Timestamp - d.tsOffset
	if d.sequencer != nil {
		d.sequencer.push(extPkt.Packet.SequenceNumber, newSN, newTS, 0, extPkt.Head)
	}
	if extPkt.Head {
		d.lastSN = newSN
		d.lastTS = newTS
	}
	hdr := extPkt.Packet.Header
	hdr.PayloadType = d.payloadType
	hdr.Timestamp = newTS
	hdr.SequenceNumber = newSN
	hdr.SSRC = d.ssrc

	_, err := d.writeStream.WriteRTP(&hdr, extPkt.Packet.Payload)
	return err
}
```

**逐行精读——这是整章最该吃透的 40 行**：

**(A) reSync 块——换源/恢复时的「接缝缝合」**
```go
if d.reSync.get() {
    if d.Kind() == ...Video {
        if !extPkt.KeyFrame {
            d.receiver.SendRTCP([]rtcp.Packet{ &rtcp.PictureLossIndication{...} })
            return nil   // ← 关键：不是 keyframe 就丢弃 + 催关键帧
        }
    }
    ...
}
```
`reSync` 在「新订阅者刚加入」「simulcast 切层」「mute 后恢复」时被置位。语义是：**这条出流刚刚换了源/恢复，必须从一个 keyframe 重新开始，否则订阅者解不了码**。
- 如果是视频且当前包 **不是关键帧** → 立刻 `SendRTCP(PLI)` 催发布端出关键帧，并 `return nil` **直接丢弃这个非关键帧包**（发给订阅者也没用，反而花屏）。
- 这就是「**新人入会先看到几百 ms 黑屏然后画面出现**」的根因（§7.1）：SFU 在等一个 keyframe。
- `SenderSSRC: d.ssrc, MediaSSRC: extPkt.Packet.SSRC`：PLI 里 MediaSSRC 用的是 **源包的 SSRC**——这条 PLI 要顺着 RTCP 回路一直送到发布端的编码器，让它对那路流强制 IDR。

**(B) 计算偏移——把「源侧的多变」吸收掉**
```go
if d.lastSN != 0 {
    d.snOffset = extPkt.Packet.SequenceNumber - d.lastSN - 1
    d.tsOffset = extPkt.Packet.Timestamp - d.lastTS - 1
}
atomic.StoreUint32(&d.lastSSRC, extPkt.Packet.SSRC)
```
等到了 keyframe（能往下走），重新计算 **偏移量**：
- `snOffset = 源SN - 上次发出去的SN - 1`：让「这次改写后的 SN」正好接在「上次发出去的 SN」后面 +1，**序列号连续无跳变**。订阅者的 jitter buffer 看到的是一条连号的流，根本不知道源换过。
- `tsOffset = 源TS - 上次发出去的TS - 1`：同理保证 **时间戳单调递增**。
- `lastSSRC` 记下当前源 SSRC，下次用来检测「源是不是又换了」。

> **这就是 §2.1 说的「把多变隐藏成连续流」的算法实现**。源侧可能是高清层断了切到低清层（SSRC 变、SN 从一个完全不同的值开始），但经过 offset 减法，订阅侧永远看到 `...,1000,1001,1002,...` 这样的连号。

**(C) 改写并发送——payload 一字不动**
```go
newSN := extPkt.Packet.SequenceNumber - d.snOffset
newTS := extPkt.Packet.Timestamp - d.tsOffset
if d.sequencer != nil {
    d.sequencer.push(extPkt.Packet.SequenceNumber, newSN, newTS, 0, extPkt.Head)
}
...
hdr := extPkt.Packet.Header   // ← 拷贝 header（值拷贝，不动原包）
hdr.PayloadType = d.payloadType
hdr.Timestamp = newTS
hdr.SequenceNumber = newSN
hdr.SSRC = d.ssrc
_, err := d.writeStream.WriteRTP(&hdr, extPkt.Packet.Payload)  // ← payload 原样
```
- `newSN/newTS`：套用偏移得到订阅侧的 SN/TS。
- `d.sequencer.push(srcSN, newSN, ...)`：把「源SN → 改写SN」的映射存进 sequencer。**为什么需要？** 当订阅者发 NACK 说「我丢了改写后的 SN=2000 那个包」，SFU 要能反查回「那是源 SN=多少」，从 buffer 里取出原包重新改写重发。没有这张映射表就没法做 SFU 侧重传。
- `hdr := extPkt.Packet.Header`：**值拷贝** header（Go 里 struct 赋值是拷贝），所以改 `hdr` 不污染原 `extPkt`——这点至关重要，因为 **同一个 extPkt 正被 N 个 DownTrack 并发改写**（回看 §3 的 fan-out 循环），如果原地改头，第二个 DownTrack 就拿到被第一个改坏的头了。每个 DownTrack 拷一份 header 各改各的。
- `WriteRTP(&hdr, extPkt.Packet.Payload)`：**payload 直接复用原始切片，零拷贝、零修改**。这就是「SFU 不碰媒体内容」的字面落实——它只换了信封（header），没动信里的内容（payload）。也正因如此，配合 Insertable Streams，payload 可以是端到端加密的，SFU 改头照样工作（§1.2 提到的 E2EE 优势）。

> **一句话总结 §3+§4**：**Receiver 把入流处理干净（一次），fan-out 遍历 N 个 DownTrack（O(N) 指针操作），每个 DownTrack 拷一份 header、用 offset 把 SSRC/SN/TS 改成订阅侧认得的值、payload 原样转发。** 这 ~80 行就是 SFU 的全部核心。其余都是周边：信令、带宽估计、simulcast 选层、级联。

---

## 5. 真实源码：mediasoup 的路由模型（C++ 对照）

Pion 用「Receiver → DownTracks」的对象图。mediasoup（C++ worker）用更显式的 **Producer/Consumer + 路由 map** 模型。【真实源码 versatica/mediasoup@worker/include/RTC/Router.hpp】Router 的成员 map（verbatim）：

```cpp
ankerl::unordered_dense::map<std::string, RTC::Transport*> mapTransports;
ankerl::unordered_dense::map<std::string, RTC::RtpObserver*> mapRtpObservers;
ankerl::unordered_dense::map<RTC::Producer*,
  ankerl::unordered_dense::set<RTC::Consumer*>> mapProducerConsumers;
ankerl::unordered_dense::map<RTC::Consumer*, RTC::Producer*>
  mapConsumerProducer;
ankerl::unordered_dense::map<RTC::Producer*,
  ankerl::unordered_dense::set<RTC::RtpObserver*>> mapProducerRtpObservers;
ankerl::unordered_dense::map<std::string, RTC::Producer*> mapProducers;
ankerl::unordered_dense::map<RTC::DataProducer*,
  ankerl::unordered_dense::set<RTC::DataConsumer*>> mapDataProducerDataConsumers;
ankerl::unordered_dense::map<RTC::DataConsumer*, RTC::DataProducer*>
  mapDataConsumerDataProducer;
ankerl::unordered_dense::map<std::string, RTC::DataProducer*> mapDataProducers;
```

**精读**：
- **`Producer`** = 发布端的一路流（≈ Pion 的 Receiver）。**`Consumer`** = 某订阅者对某 Producer 的一条订阅（≈ Pion 的 DownTrack）。**`Transport`** = 一条 PeerConnection 级别的连接（一个 Transport 上可挂多个 Producer/Consumer）。
- **`mapProducerConsumers: Producer* → set<Consumer*>`**：这就是 fan-out 的路由表——一个 Producer（一路上行）映射到一组 Consumer（多条下行）。当 Producer 收到 RTP 包，遍历这个 set 把包喂给每个 Consumer。**和 Pion 的 `downTracks[layer]` slice 是同一个意思，只是 mediasoup 用 set、Pion 用 atomic slice。**
- **`mapConsumerProducer: Consumer* → Producer*`**：反向索引。Consumer 发来 RTCP（PLI/NACK）时，用它快速找到对应的 Producer，把反馈路由回发布端。**对应 Pion 里 DownTrack 持有的 `d.receiver` 引用**（回看 §4 的 `d.receiver.SendRTCP(...)`）。
- 末尾几条 `DataProducer/DataConsumer`：SCTP DataChannel 的同构路由（mediasoup 把 DataChannel 也纳入同一套 SFU 路由模型）。

> **两个项目的设计同构**：无论 Pion 还是 mediasoup，SFU 的本质数据结构都是 **「一个源 → 一组汇」的 fan-out map + 「汇 → 源」的反向 map（给 RTCP 回路用）**。理解了这对 map，任何 SFU 的代码你都能按图索骥。

> **方案差异（对比见 §8）**：mediasoup 把媒体平面放在 C++ worker（多进程，每核一个 worker），信令/编排放在 Node.js，靠 libuv pipe 通信——这让它能吃满多核、媒体平面零 GC。Pion ion-sfu 全 Go，单进程，靠 goroutine 并发——可读性极高（本章源码就是证据），但媒体热路径要小心 GC 停顿和 channel 开销。

---

## 6. ⭐ Demo 1（对照组）：浏览器纯 JS Mesh，亲手感受「为什么需要 SFU」

> 目标：用纯浏览器 JS 搭一个 **2 人 Mesh**，在 `chrome://webrtc-internals` 里观测「每多一个人，上行多一路」，亲身体会 Mesh 为什么撑不住多人。这是 SFU 的「反面教材对照组」，**不需要任何服务器**（信令用最朴素的「手动复制粘贴 SDP」，零依赖能真跑）。

**为什么先做 Mesh demo**：SFU demo 需要 Go 环境（Demo 2）。但理解 SFU「省在哪」，最快的方式是先亲手摸到 Mesh「贵在哪」——看着 webrtc-internals 里上行 track 数随人数线性涨。

### 6.1 完整代码（单文件 `mesh.html`，两个浏览器标签页对跑）

```html
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Mesh PoC（无信令服务器，手动交换SDP）</title></head>
<body>
<h3>Mesh 1对1 PoC：体会 P2P 直连</h3>
<video id="local" autoplay muted playsinline style="width:240px;border:1px solid #ccc"></video>
<video id="remote" autoplay playsinline style="width:240px;border:1px solid #ccc"></video>
<div>
  <button id="start">1. 开摄像头</button>
  <button id="makeOffer">2A. [发起方] 生成 Offer</button>
  <button id="makeAnswer">2B. [应答方] 粘贴Offer后生成 Answer</button>
  <button id="accept">3. [发起方] 粘贴 Answer 完成连接</button>
</div>
<p>把下面文本框的内容复制到另一个标签页对应的框（这就是「信令」，这里用人肉传递）：</p>
<textarea id="sdp" rows="8" cols="90" placeholder="SDP 会出现在这里"></textarea>

<script>
const pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }] // 公共 STUN，仅打洞用
});
const sdpBox = document.getElementById('sdp');

// 远端轨到达 → 显示
pc.ontrack = (e) => { document.getElementById('remote').srcObject = e.streams[0]; };

// ICE 候选收集完成后，才把完整 SDP 显示出来（避免 trickle 的复杂度，PoC 用 non-trickle）
pc.onicegatheringstatechange = () => {
  if (pc.iceGatheringState === 'complete' && pc.localDescription) {
    sdpBox.value = JSON.stringify(pc.localDescription);
  }
};
pc.oniceconnectionstatechange = () => console.log('ICE:', pc.iceConnectionState);

document.getElementById('start').onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  document.getElementById('local').srcObject = stream;
  // 关键：把本地每条 track 加进 PeerConnection —— Mesh 下「每个对端」都要这样加一遍
  stream.getTracks().forEach(t => pc.addTrack(t, stream));
};

document.getElementById('makeOffer').onclick = async () => {
  await pc.setLocalDescription(await pc.createOffer());
  // SDP 在 onicegatheringstatechange complete 时显示
};

document.getElementById('makeAnswer').onclick = async () => {
  const offer = JSON.parse(sdpBox.value);          // 粘贴对方的 Offer
  await pc.setRemoteDescription(offer);
  await pc.setLocalDescription(await pc.createAnswer());
  // Answer 在 gathering complete 时显示，再复制回发起方
};

document.getElementById('accept').onclick = async () => {
  const answer = JSON.parse(sdpBox.value);          // 粘贴对方的 Answer
  await pc.setRemoteDescription(answer);
};
</script>
</body>
</html>
```

> 上面的 `RTCPeerConnection`/`getUserMedia`/`addTrack`/`createOffer`/`ontrack` 都是 **W3C WebRTC 标准 API**（第 1 章已介绍语义）。代码为本章手写 PoC【示意，非逐字引用某仓库】，但每个 API 调用都是标准、可直接在 Chrome/Edge/Firefox 跑通的。

### 6.2 运行步骤

1. 存为 `mesh.html`。**两个浏览器标签页都打开它**（同机即可，模拟两个人）。
2. 两个标签页各点 **「1. 开摄像头」**，各自允许摄像头权限，能看到自己的画面。
3. 标签页 A 点 **「2A. 生成 Offer」**，等下方文本框出现 JSON SDP，**全选复制**。
4. 标签页 B 把 A 的 SDP **粘贴进文本框**，点 **「2B. 生成 Answer」**，等它产生新的 SDP，复制。
5. 标签页 A 把 B 的 Answer **粘贴进文本框**，点 **「3. 完成连接」**。
6. **预期结果**：两个标签页的 remote `<video>` 互相显示对方画面；Console 里 ICE 状态走到 `connected`/`completed`。

### 6.3 用 webrtc-internals 观测（核心收获）

打开 `chrome://webrtc-internals`，找到这个 PeerConnection，展开 Stats：
- **`outbound-rtp`**：你会看到 **1 个 video + 1 个 audio** 的 outbound（你上行了 1 路给对端）。
- **`inbound-rtp`**：对应 **1 个 video + 1 个 audio** 的 inbound（你下行了对端 1 路）。
- 看 `outbound-rtp` 的 `bytesSent` 曲线，记下单路上行码率（约 1~2 Mbps）。

**关键推演（不用真搭 4 人，算给你看）**：Mesh 下加到 4 人，每个标签页会有 **3 个 PeerConnection**，于是 **outbound-rtp 变成 3 路 video**（上行 ×3）。把 §6.3 看到的单路码率 ×3，就是你上行要扛的总量。**这就是 §1.2 说的「Mesh 上行 O(N)，4~5 人就爆」的亲眼证据**——而 SFU 永远只有 1 路 outbound（Demo 2 验证）。

---

## 7. ⭐ Demo 2（主菜）：Pion 最小 SFU，多人转发真跑

> 目标：用 Pion 搭一个 **最小可跑的 SFU**：客户端推 1 路上去，SFU fan-out 给所有其他客户端。验证 §3/§4 的核心循环，并在 webrtc-internals 看到「上行 1 路、下行 N-1 路」。

SFU 的本体在第 3、4 章的源码里已经讲透。这个 demo 用 Pion 的 **`TrackLocalStaticRTP` + `OnTrack` + 包级转发** 还原最小 SFU 转发链路（ion-sfu 是它的工业级扩展）。

### 7.1 SFU 转发的本质循环（真实源码佐证）

最小 SFU 的核心就一句：从某 inbound track 读 RTP 包，写到所有其他客户端的 outbound track。【真实源码 pion/webrtc@examples/broadcast/main.go】的转发循环（verbatim）正是这个本质：

```go
rtpBuf := make([]byte, 1400)
for {
    i, _, readErr := remoteTrack.Read(rtpBuf)
    if readErr != nil {
        panic(readErr)
    }

    if _, err = localTrack.Write(rtpBuf[:i]); err != nil && !errors.Is(err, io.ErrClosedPipe) {
        panic(err)
    }
}
```

`remoteTrack.Read` 从发布端读出一个 RTP 包（原始字节），`localTrack.Write` 写进一个 `TrackLocalStaticRTP`——这个 local track 被 add 到所有订阅者的 PeerConnection 上，于是一写即广播。**最小 SFU 就是把这个「1→1」扩成「1→N」（一个 remoteTrack 写进 N 个订阅者各自的 local track）。**

> 注意这里 `localTrack.Write(rtpBuf[:i])` 写的是 **原始 RTP 字节**——Pion 的 `TrackLocalStaticRTP` 内部会处理 SSRC 改写（绑定到该订阅 PeerConnection 协商的 SSRC），所以 demo 层不用手算 offset。ion-sfu 的 `writeSimpleRTP`（§4）是这个机制的「手动精细版」，用于支持 simulcast 切层等高级场景。

### 7.2 keyframe 请求（PLI）——新订阅者不黑屏的关键

新订阅者加入时，必须尽快拿到 keyframe。现代 Pion 用 **`intervalpli` interceptor** 周期性发 PLI。【真实源码 pion/webrtc@examples/broadcast/main.go 注释，verbatim】：

> *"This interceptor sends a PLI every 3 seconds. A PLI causes a video keyframe to be generated by the sender."*

即注册一个 `intervalpli.NewReceiverInterceptor()` 到 InterceptorRegistry，Pion 自动每 3 秒向发布端发 PLI 催关键帧。**这呼应 §3/§4 里 ion-sfu 手动 `SendRTCP(pli)` 的逻辑**——都是为了解决 §8.1 的「keyframe 饥饿」，只是 interceptor 把它自动化了。

### 7.3 最小 SFU 完整代码骨架（Go）

下面是把「broadcast 的 1→1」扩成「SFU 的 1→N」的最小骨架。【示意，非逐字 —— 基于 pion/webrtc 标准 API 与 broadcast/sfu 例程的本章组装；`Read/Write/OnTrack/AddTrack/intervalpli` 均为 Pion 真实 API】：

```go
package main

import (
	"errors"
	"io"
	"sync"

	"github.com/pion/interceptor"
	"github.com/pion/interceptor/pkg/intervalpli"
	"github.com/pion/webrtc/v4"
)

// 房间：保存所有订阅者要写入的本地转发轨。
type Room struct {
	mu          sync.RWMutex
	localTracks []*webrtc.TrackLocalStaticRTP // 每个发布流对应一条，被加到所有订阅者 PC 上
}

func (r *Room) addLocalTrack(t *webrtc.TrackLocalStaticRTP) {
	r.mu.Lock()
	r.localTracks = append(r.localTracks, t)
	r.mu.Unlock()
}

// 构造带 intervalpli 的 API：自动每 3s 发 PLI 催关键帧（见 §7.2 真实注释）。
func newAPI() *webrtc.API {
	m := &webrtc.MediaEngine{}
	_ = m.RegisterDefaultCodecs()
	ir := &interceptor.Registry{}
	_ = webrtc.RegisterDefaultInterceptors(m, ir)
	pli, _ := intervalpli.NewReceiverInterceptor() // ← 关键帧请求自动化
	ir.Add(pli)
	return webrtc.NewAPI(webrtc.WithMediaEngine(m), webrtc.WithInterceptorRegistry(ir))
}

// 每个新连接进来调用：建 PC，处理上行(OnTrack→fan-out)与下行(把已有 localTracks 加进来)。
func (r *Room) handlePeer(api *webrtc.API, offer webrtc.SessionDescription) (webrtc.SessionDescription, error) {
	pc, err := api.NewPeerConnection(webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{{URLs: []string{"stun:stun.l.google.com:19302"}}},
	})
	if err != nil {
		return webrtc.SessionDescription{}, err
	}

	// (下行) 把房间里已有的所有发布流加给这个新订阅者
	r.mu.RLock()
	for _, lt := range r.localTracks {
		if _, err := pc.AddTrack(lt); err != nil {
			r.mu.RUnlock()
			return webrtc.SessionDescription{}, err
		}
	}
	r.mu.RUnlock()

	// (上行) 这个 peer 推流上来 → 建一条 local 转发轨，fan-out 给所有人
	pc.OnTrack(func(remote *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		local, err := webrtc.NewTrackLocalStaticRTP(
			remote.Codec().RTPCodecCapability, "video", "pion-sfu")
		if err != nil {
			return
		}
		r.addLocalTrack(local) // 后续新订阅者会拿到它；已在线的需重新协商(本骨架略)

		// ★ SFU 本质循环：读上行包 → 写进本地转发轨(=广播给所有 add 了它的订阅者)
		buf := make([]byte, 1500)
		for {
			n, _, readErr := remote.Read(buf)
			if readErr != nil {
				return
			}
			if _, err := local.Write(buf[:n]); err != nil && !errors.Is(err, io.ErrClosedPipe) {
				return
			}
		}
	})

	if err := pc.SetRemoteDescription(offer); err != nil {
		return webrtc.SessionDescription{}, err
	}
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		return webrtc.SessionDescription{}, err
	}
	gather := webrtc.GatheringCompletePromise(pc) // non-trickle，简化信令
	if err := pc.SetLocalDescription(answer); err != nil {
		return webrtc.SessionDescription{}, err
	}
	<-gather
	return *pc.LocalDescription(), nil
}
```

> **诚实标注**：上面是 **教学骨架**，聚焦「OnTrack→fan-out 转发轨」这条主干。生产级（ion-sfu）还需要：① 新发布流出现时给已在线订阅者 **重新协商**（renegotiation）；② 信令传输层（WebSocket）；③ simulcast 选层、带宽估计、NACK 重传。**真正可直接 `git clone && go run` 跑起来的完整多人 SFU，请用官方 `pion/webrtc/examples/sfu-ws`（一个 HTML 客户端 + Go SFU，开箱即跑）**——本骨架是为了让你看清它内部那条转发循环到底在干嘛。`broadcast` 例程（§7.1，单发布多订阅的直播形态）可直接跑通验证转发循环本身。

### 7.4 运行与观测（用 broadcast 例程验证转发循环）

最省事的「真跑」路径——直接用官方 broadcast 例程验证 §7.1 那条循环：

```bash
go install github.com/pion/webrtc/v4@latest   # 拉依赖
git clone https://github.com/pion/webrtc.git
cd webrtc/examples/broadcast
go run main.go                                 # 启动，监听 :8080
```

1. 浏览器开 `http://localhost:8080`（或例程 README 指定页面），**发布端**点 Publish，把生成的 base64 SDP 按提示交换。
2. 再开一个标签作 **订阅端**，Subscribe。
3. **预期**：订阅端看到发布端的视频；切换/新订阅者加入时，约 ≤3 秒（intervalpli 周期）画面出现——**这 ≤3 秒就是在等 keyframe**（§8.1）。
4. **webrtc-internals 观测**：
   - 发布端 PC：`outbound-rtp` 1 路 video（上行 1 路）。
   - 订阅端 PC：`inbound-rtp` 1 路 video。
   - **和 Demo 1 的 Mesh 对比**：Mesh 加人 = outbound 路数线性涨；SFU 里发布端 **永远 outbound 1 路**，加多少订阅者都不变——这就是 SFU 省上行的铁证。
   - 看订阅端 inbound-rtp 的 `pliCount` / `keyFramesDecoded`：每次新订阅会触发 PLI、随后 keyFramesDecoded +1。

---

## 8. 扎根：失败模式 / 真坑 / 根因

### 8.1 真坑一：keyframe 饥饿——新订阅者黑屏/花屏几秒

**现象**：用户进会，自己画面立刻有，但 **其他人的画面要愣几百 ms 到几秒才出现**；或者 simulcast 切清晰度时花屏一下。

**根因**：视频帧分 **keyframe（I 帧/IDR，可独立解码）** 和 **inter frame（P/B 帧，依赖前帧）**（第 5 章）。新订阅者从「流的中间」接入，它收到的大概率是 inter frame——**没有参考帧，解不了码**。回看 §4 的 `writeSimpleRTP`：`reSync` 期间 `if !extPkt.KeyFrame { SendRTCP(PLI); return nil }` —— SFU **主动丢弃非关键帧并发 PLI 催发布端出 IDR**。从「发 PLI」到「发布端编码器产出 IDR」到「IDR 传到订阅者」这段时间，订阅者就是黑屏。

**为什么不能让 SFU 自己缓存 keyframe？** 因为 SFU 不解码（§1.2），它不知道 payload 里哪个是 keyframe 的完整数据、更没法合成一个。它顶多靠 RTP 扩展头/depacketizer 判断「这个包是不是 keyframe 的一部分」（`extPkt.KeyFrame`），但无法凭空造一个。所以唯一出路是 **催发布端重发 keyframe**。

**缓解**：
- **PLI/FIR 及时回传**（§7.2 的 intervalpli、§4 的 SendRTCP）。这要求 §2.1 那条 `rtcpCh`+`writeRTCP` RTCP 回路畅通——RTCP 回路断了，PLI 到不了发布端，就是永久黑屏。
- **发布端缩短 keyframe 间隔**（GOP 调小），代价是码率上升。
- **keyframe 请求去抖**：N 个订阅者同时入会会同时发 PLI，发布端被 PLI 风暴打爆、疯狂出 IDR 导致码率尖峰。SFU 要 **聚合/限频 PLI**（一段时间内只向发布端转发一个）。

### 8.2 真坑二：带宽估计（BWE）不能跨连接复用——抖动/丢包的不对称

**现象**：Alice 网络很好（推 2 Mbps 上来没问题），但 Carol 的下行只有 500 kbps。如果 SFU 把 Alice 的 2 Mbps 原样转给 Carol，Carol **持续丢包、花屏、卡顿**。

**根因**：SFU 是「**多条独立拥塞域的交汇点**」。发布端→SFU 是一条拥塞域，SFU→每个订阅者各是 **独立的一条拥塞域**（第 6 章 GCC/transport-cc）。Alice 上行链路的带宽估计 **完全不能代表** Carol 下行链路的容量。SFU 必须 **为每条出边独立做带宽估计**（靠 §2.1 的 twcc Responder 收每个订阅者的 transport-cc 反馈），然后据此 **为每个订阅者选择合适的层/码率**。

**缓解（这正是 simulcast/SVC 存在的根本原因）**：
- **Simulcast**：发布端 **同时上行多个清晰度层**（如 180p/360p/720p，不同 SSRC）。SFU 为每个订阅者根据其下行 BWE **选转发哪一层**（回看 §3 的 `downTracks[layer]` 按 layer 组织、`pendingTracks` 切层）。Carol 网差就只转 180p 那层给她。
- **SVC（可伸缩编码）**：单条流内分层（时间/空间/质量），SFU 通过 **丢弃高层包** 就能降码率，无需发布端配合。
- **核心代价**：simulcast 让发布端上行变成「多层之和」（部分抵消 SFU 的省上行优势），但换来 SFU 能精细适配每个异构订阅者——这是 §1 经济账的延续：**用一点发布端上行，换所有订阅者的体验下限**。

### 8.3 真坑三：单点瓶颈与级联（cascading）——SFU 不是银弹

**现象**：单台 SFU 撑到几百路后，**网卡带宽/CPU（SRTP 加解密、RTCP 处理）打满**；或者参会者分布全球，都连同一台 SFU 导致 **远端用户 RTT 极高**。

**根因**：
- SFU 虽不解码，但 **SRTP 解密/重加密 + transport-cc/PLI 处理 + N×N 的转发** 在高并发下仍吃满 CPU/带宽。下行总量是 `Σ(每个订阅者收的路数 × 码率)`，N 人全互看是 O(N²) 量级的总下行带宽（虽然每条客户端只 N-1 路）。
- 单台 SFU 有 **地理位置**，跨洲订阅者 RTT 高。

**缓解**：
- **级联 SFU（cascading / relayed SFU）**：多台 SFU 互联，用户就近接入本地 SFU，SFU 之间只传「跨区需要的流」。这是 Jitsi 的 Octo、Janus 多实例、mediasoup 的 PipeTransport（`mapTransports` 里可以是连到另一台 SFU 的 pipe）干的事。代价：**SFU 间多一跳延迟**、级联拓扑的路由复杂度。
- **房间分片**：大会议把订阅关系裁剪（不是人人互看，只看说话人+主屏），把 O(N²) 压成 O(N)。
- **SFU 选层主动降级**：BWE 紧张时主动给所有人降一层。

### 8.4 真坑四：RTCP 回路与 sequencer——SFU 侧重传的隐藏复杂度

**现象**：订阅者偶发丢包，但 NACK 重传后画面还是有破绽；或日志里 sequencer 找不到原包。

**根因**：回看 §4，DownTrack 把「源SN → 改写SN」存进 `sequencer`。当订阅者发 NACK（要改写后的 SN=X），SFU 要：① 用 sequencer 反查 X 对应的源 SN；② 从 Receiver 的 buffer 里取出那个源包；③ **重新走一遍 offset 改写**（用当时的 offset，不是现在的）；④ 重发。任何一环——buffer 已淘汰原包、offset 因为中途切过层而变了、sequencer 容量不够——都会让重传失败。**SFU 的重传比端到端重传复杂，因为它要在「源 SN 空间」和「每个订阅者各自的改写 SN 空间」之间反复翻译。**

**缓解**：buffer 留足重传窗口；sequencer 容量匹配 RTT×码率；切层时正确处理 offset 重算（§4 的 reSync 块）。

---

## 9. 方案对比

### 9.1 拓扑对比（综合 §1 bloggeek 表 + 工程经验）

| 维度 | Mesh (P2P) | MCU (混流) | SFU (转发) |
|---|---|---|---|
| 服务器成本 | 零 | **高**（解码+混+编码） | 低-中（只改头+SRTP） |
| 服务器 CPU | 零 | **极高** | **低**（不碰媒体内容） |
| 客户端上行 | **高** O(N-1) | 低（1 路） | 低（1 路，simulcast 时多层） |
| 客户端下行 | 高 O(N-1) | **低**（1 路混好的） | 中 O(N-1) |
| 客户端 CPU | **高**（N-1 次编/解码） | **低**（1 收 1 发） | 中（N-1 路解码） |
| 端到端延迟 | **最低**（直连） | 高（转码流水线） | 低（仅转发一跳） |
| 可扩展性 | 2-5 人 | low tens（几十） | **hundreds-thousands** |
| E2EE | 天然支持 | **不可能**（要看明文混流） | **可支持**（Insertable Streams/SFrame） |
| 客户端自定义版面 | 支持 | **不支持**（版面服务器定死） | 支持 |
| 兼容老/弱端 | 差 | **好**（只需收 1 路） | 中 |

### 9.2 场景化选型与不适用边界

- **1 对 1 通话** → **Mesh（直连）**。没有任何理由上服务器，直连延迟最低、零成本、天然 E2EE。*不适用边界*：若两端都在对称 NAT 后打洞失败，需 TURN 中继（第 2 章），但那只是中继字节、仍是 P2P 语义，不是 SFU。
- **3~50 人会议（现代主流）** → **SFU**。配 simulcast 适配异构网络，配 Insertable Streams 做 E2EE。*不适用边界*：当订阅端是 **算力极弱、只能解 1 路** 的设备（如低端机顶盒、PSTN 电话网关接入），SFU 的「下行 N-1 路」扛不住——这时退回 MCU 混成 1 路。
- **超大规模直播（万人观看）** → **SFU 级联 + 边缘分发**（§8.3），或退化为 **单向 RTMP/HLS/LL-HLS CDN**（不再是实时互动语义，延迟从亚秒到数秒，换吞吐）。*不适用边界*：万人 **互动**（人人可发言）任何拓扑都扛不住，必须产品上限制同时发言人数。
- **电话会议网关 / 录制成单文件 / 需要服务端 AI 处理（转写、虚拟背景合成、内容审核）** → **MCU 或 SFU+旁路解码**。需要在服务端看到明文媒体的场景，是 MCU 仍存在的根本理由。*不适用边界*：一旦要服务端解码，E2EE 就放弃了——这是个产品级取舍，不是技术能绕过的。
- **混合架构（工业现实）**：大厂方案常是 **SFU 为主干 + 按需 MCU**——绝大多数参会者走 SFU，对电话接入/录制/AI 分析这些少数需求，旁路一个 MCU/解码 worker。mediasoup 的 `PipeTransport`、给 Producer 旁挂一个解码 Consumer，就是为这种混合留的口子。

---

## 10. 章末五件套

### 10.1 一图流（心智模型）

```
                P2P/Mesh                MCU                    SFU（本章主角）
  上行          ▲▲▲ N-1路               ▲ 1路                  ▲ 1路(simulcast多层)
  服务器        （无）                  解码→混流→编码          只改 SSRC/SN/TS 转发(payload不动)
  下行          ▼▼▼ N-1路               ▼ 1路(混好)            ▼ N-1路(各自解码自排版)
  瓶颈          客户端上行              服务器CPU              客户端下行 / SFU网卡

  SFU 核心循环（Receiver → DownTrack）：
    Receiver.buffer ──读出有序包──► fan-out for-loop ──► DownTrack1: copy header→ssrc/sn/ts改写→Write
                                            │           └► DownTrack2: copy header→ssrc/sn/ts改写→Write
                                            └► (payload 永远零拷贝复用)
    反向 RTCP 回路：订阅者 PLI/NACK ──map(Consumer→Producer)──► 发布端编码器(出IDR)/buffer(重传)
```

### 10.2 决策清单（选拓扑时逐条问）

1. 人数？≤2 → Mesh 收工；3~50 → SFU；上千 → SFU 级联或 CDN。
2. 订阅端能解几路？只能 1 路（弱端/网关）→ 被迫 MCU。
3. 要不要 E2EE？要 → 只能 SFU（+Insertable Streams），MCU 出局。
4. 要不要服务端看明文（录制/AI/转写）？要 → MCU 或 SFU 旁路解码，放弃 E2EE。
5. 网络异构严重吗？是 → SFU + simulcast/SVC（§8.2），否则弱端被强端码率打爆。
6. 跨地域吗？是 → 级联 SFU 就近接入（§8.3）。
7. keyframe 策略定了吗？PLI 回路通不通、要不要去抖（§8.1）？

### 10.3 三个最该背下来的结论

1. **SFU 不碰 payload，只改 header**：`writeSimpleRTP` 拷一份 header、用 `snOffset/tsOffset` 把 SSRC/SN/TS 改成订阅侧的值，payload 零拷贝转发（§4）。这一条解释了 SFU 的省 CPU、可 E2EE、可零拷贝。
2. **fan-out 是 O(N) 指针操作**：Receiver 从 buffer 读一次（NACK/重排做一次），遍历 N 个 DownTrack 各写一份（§3）。对比 MCU 的「N 解码+混+编码」，这是数量级差异。
3. **SFU 是多拥塞域交汇点**：发布端 BWE ≠ 订阅端 BWE，必须每条出边独立估计 + simulcast/SVC 选层（§8.2）。这是 SFU 工程复杂度的主要来源。

### 10.4 自测题（能答上才算吃透）

1. 为什么 §4 的 `writeSimpleRTP` 里要 `hdr := extPkt.Packet.Header`（值拷贝）而不是直接改原包？（提示：回看 §3 fan-out 并发写同一个 extPkt）
2. 新订阅者入会黑屏 2 秒，链路上哪几个环节可能出问题？（PLI 没发？RTCP 回路断？发布端 GOP 太长？PLI 被去抖吞了？）
3. simulcast 切层为什么「必须等 keyframe」？§3 代码里是怎么等的？等的时候为什么还要发 PLI？
4. 订阅者发 NACK 要重传，SFU 凭什么找回原包？（`sequencer` 的「源SN→改写SN」反查，§4+§8.4）
5. 100 人全互看的房间，单台 SFU 下行总带宽是 O(N) 还是 O(N²)？为什么每个客户端只感觉到 O(N-1)？
6. mediasoup 的 `mapProducerConsumers` 和 `mapConsumerProducer` 各对应 Pion 的什么？为什么需要那条反向 map？（§5）

### 10.5 延伸阅读（按可读性排序）

- **Pion ion-sfu**（Go，本章源码主来源，最可读的工业级 SFU）：`pkg/sfu/receiver.go`（fan-out）、`pkg/sfu/downtrack.go`（头部改写）、`pkg/sfu/router.go`（路由+RTCP 回路）。
- **Pion webrtc 官方 examples**（开箱即跑）：`examples/broadcast`（单发布多订阅，本章 Demo 验证）、`examples/sfu-ws`（完整多人 SFU，WS 信令+HTML 客户端）。
- **mediasoup**（C++ worker + Node，工业级多核 SFU）：`worker/include/RTC/Router.hpp`（路由模型）、`mediasoup.org/documentation`（设计文档）。
- **设计史/选型**：bloggeek.me/webrtcglossary/sfu（Tsahi Levent-Levi，SFU/MCU/mesh 对照）；webrtcforthecurious.com（拓扑章节）。
- **RFC**：RFC 3550（RTP/RTCP，SSRC/SN/TS 语义）、RFC 4585（RTP/AVPF，PLI/FIR/NACK 反馈，SFU 回路的协议基础）、RFC 7667（RTP 拓扑学，mesh/translator/mixer 的官方分类——MCU≈mixer、SFU≈selective forwarding middlebox）。

---

> **本章诚实声明**：§3（receiver fan-out）、§4（writeSimpleRTP）、§5（mediasoup Router.hpp maps）、§7.1（broadcast 转发循环）、§7.2（intervalpli 注释）的代码块均为 **WebFetch 实际取得的 verbatim 真实源码**，标注了 repo@path。§2.1 的 DownTrack 字段、§6（Mesh HTML）、§7.3（SFU Go 骨架）为 **基于标准 API 的本章组装/示意**，已逐处标注【示意，非逐字】。Demo 1（Mesh HTML）可直接在浏览器跑；Demo 2 的 **broadcast 例程可 `go run` 直接跑通**验证核心转发循环，§7.3 的多人 SFU 骨架是教学用、生产请用官方 sfu-ws。少数字段精确性标了「待核」。本章未取到 mediasoup 的 Design.md（404）与 sfu-ws/rtp-forwarder 源码（404/被摘要），相关结论用高把握知识补全并标注来源。
