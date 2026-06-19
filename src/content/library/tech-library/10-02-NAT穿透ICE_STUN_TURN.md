---
title: "NAT 穿透:ICE / STUN / TURN(WebRTC 域)"
slug: "10-02"
collection: "tech-library"
group: "webrtc"
order: 10002
summary: "TL;DR - WebRTC 两端几乎从不在同一个公网地址下,中间隔着一层甚至多层 NAT。ICE(RFC 8445)是总框架,它编排两件事:收集候选地址(candidate gathering) 和 连通性检查(connectivity check)。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-15T14:06:02.000Z"
updatedAt: "2026-06-15T14:06:02.000Z"
---
> **TL;DR**
> - WebRTC 两端几乎从不在同一个公网地址下,中间隔着一层甚至多层 NAT。**ICE(RFC 8445)是总框架**,它编排两件事:**收集候选地址(candidate gathering)** 和 **连通性检查(connectivity check)**。
> - **STUN(RFC 8489 / 旧 5389)** 是一把"照镜子"的工具:你向 STUN server 发一个 Binding Request,它把"我从外面看到你的源地址:端口"原样回给你 —— 这就是你的 **server-reflexive candidate(srflx)**,是打洞(hole punching)的弹药。
> - **TURN(RFC 8656 / 旧 5766)** 是兜底中继:当两端任何直连都打不通(典型是 symmetric NAT 对 symmetric NAT),数据绕道 TURN server 转发,得到 **relay candidate**。TURN 永远能通,但贵、慢、烧带宽。
> - ICE 不是"先 STUN 失败再 TURN",而是**所有候选同时收集、配对成 candidate pair、按 priority 排序、并行做连通性检查(STUN Binding over the pair)、用 nomination 选出最终通的那一对**。
> - 本章给两个**真能跑**的 demo:① 浏览器 `RTCPeerConnection` + `webrtc-internals` 实时看候选与 selected pair;② 一段 ~40 行的脚本/Pion 发一个 STUN Binding Request,亲手拿到自己的"公网地址:端口"。
>
> **前置依赖**:第 1 章(信令与 SDP —— 候选最终是塞进 SDP 的 `a=candidate` 行,或走 Trickle ICE 单独传);UDP / TCP / IP 基础;懂"网络字节序 = 大端"。

---

## 6.1 设计考古:为什么需要 ICE 这一整套?

### NAT 的原罪:地址翻译破坏了端到端可达性

IPv4 地址不够用,于是有了 **NAT(Network Address Translation,RFC 3022 / 3489 背景)**。家里、公司、运营商的私网设备共享少量公网 IP,出口路由器在转发时改写 (源 IP, 源端口),并维护一张映射表。问题来了:

- 私网设备 **不知道自己出去后是什么公网 IP:端口**;
- 外部设备 **无法主动连入** 私网设备,因为映射表里没有为它预留的条目(没有"入向引导")。

WebRTC 是 P2P 媒体传输 —— 两个浏览器要直接互发 UDP 包。但双方都在 NAT 后,都不知道对方的公网落点,也都无法被对方主动连入。这就是 NAT 穿透要解决的核心矛盾。

### NAT 的行为分类:决定能不能打洞

经典 RFC 3489 把 NAT 分成四类(Full Cone / Restricted Cone / Port-Restricted Cone / Symmetric)。这套分类**在工程上已经被认为过于粗糙**(RFC 4787 改用"映射行为 + 过滤行为"两个正交维度描述),但作为心智模型仍然好用。关键就一句话:

> **NAT 给同一个内部 (IP:端口) 分配公网映射时,是否依赖目的地址?**

- **Cone 系(Endpoint-Independent Mapping,EIM)**:内部 `10.0.0.5:5000` 出去,不管发给谁,公网映射都是同一个 `203.0.113.7:40000`。→ 这意味着:我先问 STUN server 拿到 `203.0.113.7:40000`,把它告诉对端,对端发过来的包能命中同一个映射 → **可打洞**。
- **Symmetric NAT(Endpoint-Dependent Mapping,EDM)**:内部 `10.0.0.5:5000` 发给 STUN server 用映射 `:40000`,但发给对端 peer 时 NAT **重新分配** 一个 `:40001`。→ 我从 STUN 拿到的 `:40000` 对 peer 完全无效 → **打洞失败,只能走 TURN 中继**。

> ⚠️ 这是整章最重要的因果链:**STUN srflx 能不能用,取决于本端 NAT 的映射行为是不是 endpoint-independent。** symmetric NAT 是 srflx 失效、必须 fallback 到 relay 的头号原因。两端都是 symmetric 时,基本 100% 走 TURN。

### 三个协议的分工(以及历史演进)

| 协议 | RFC(现行 / 旧) | 解决什么 | 产出的 candidate |
|------|------|----------|------|
| **STUN** | RFC 8489 / 5389 / 3489 | "我从公网看是什么地址" + 后续连通性检查的载具 | **server-reflexive(srflx)** |
| **TURN** | RFC 8656 / 5766 | 直连失败时的中继转发 | **relay** |
| **ICE** | RFC 8445 / 5245 | 编排收集 + 配对 + 检查 + 选路的总框架 | 调度以上全部 + **host** / **peer-reflexive(prflx)** |

历史脉络(待核细节,大方向高把握):
- **RFC 3489(2003)** 第一版 STUN,自带 NAT 类型探测,试图让客户端"自己判断 NAT 类型再决定策略"。这条路被证明**不可靠**(多层 NAT、运营商 NAT、行为不一致)。
- **RFC 5389(2008)** 把 STUN 重构为一个**通用工具协议**,砍掉了内置的 NAT 分类逻辑(`MAPPED-ADDRESS` → `XOR-MAPPED-ADDRESS`,引入 magic cookie),"判断怎么连"这件事整体交给 ICE。
- **RFC 5245 → 8445** ICE,确立"收集所有候选 + 并行连通性检查 + 让数据说话"的范式 —— **不预判 NAT 类型,直接试**。这是工程上的关键胜利:与其猜,不如把所有可能的路径都摆出来同时打,谁通用谁。
- **RFC 8838** Trickle ICE,候选边收集边发送(见第 1 章信令),把建连延迟从"等齐所有候选"降到"拿到一个就试一个"。

---

## 6.2 STUN 协议:源码级拆解(打洞的弹药从哪来)

STUN 报文极简,理解它就理解了半个 NAT 穿透。我们直接看 Pion 的实现。

### 6.2.1 20 字节固定头

【真实源码 pion/stun@message.go】(经 WebFetch 核实)

```go
// 关键常量
const (
	magicCookie       = 0x2112A442 // 固定值,用于把 STUN 包从其他协议包里区分出来
	messageHeaderSize = 20         // 头部固定 20 字节
	TransactionIDSize = 12         // transaction ID 12 字节(96 bit)
)

type Message struct {
	Type          MessageType
	Length        uint32 // len(Raw),不含 20 字节头
	TransactionID [TransactionIDSize]byte
	Attributes    Attributes
	Raw           []byte
	// ... logger / strict 省略
}
```

20 字节头的字节布局(由 `WriteHeader()` 写入,WebFetch 核实其顺序):

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|0 0|     STUN Message Type      |         Message Length        |  <- byte 0..3
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Magic Cookie  0x2112A442              |  <- byte 4..7
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                     Transaction ID (96 bits)                  |  <- byte 8..19
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
【示意,非逐字】上图为 RFC 8489 §5 的标准报文头 ASCII 图,字节偏移与 Pion `WriteHeader()` 实际写入顺序一致(type→length→cookie→transaction id)。

几个**容易踩的细节**:
- **最高两位恒为 0**(`0 0`),这是 STUN 与其他协议(如 RTP/DTLS)复用同一个 UDP 端口时做 demultiplex 的依据之一(配合 magic cookie)。
- **magic cookie `0x2112A442` 是定值**,不是随机数。它有两个作用:(1) 让 `XOR-MAPPED-ADDRESS` 的异或有据可依;(2) 让接收方能区分这是 RFC 5389+ 的新 STUN 还是 RFC 3489 老报文(老报文这 4 字节是 transaction ID 的一部分)。
- **Message Length 不含头**,只算 attributes 部分的字节数,且因为 attribute 4 字节对齐,Length 必然是 4 的倍数。

### 6.2.2 Message Type 的 14-bit 交错编码(经典坑)

STUN 的 type 字段 14 个有效 bit,把 **method(12 bit)** 和 **class(2 bit)** 交错(interleave)编码。Binding 这个 method = `0x001`,class 有 Request / Indication / Success Response / Error Response 四种。

> ⚠️ **待核(具体文件路径)**:Pion 把这部分放在 `messagetype.go`/`type.go`,WebFetch 取这两个路径都 404(可能已重命名或内联)。以下编码规则来自 RFC 8489 §5,高把握,但**未能逐字核实 Pion 当前文件**。

【示意,非逐字,依据 RFC 8489 §5】

```
位布局(14 bit type 字段):
M11 M10 M9 M8 M7 C1 M6 M5 M4 C0 M3 M2 M1 M0
                ^              ^
              class 高位      class 低位
```

- `C1 C0 = 00` → Request;`01` → Indication;`10` → Success Response;`11` → Error Response。
- 所以 **Binding Request** 的完整 16-bit type = `0x0001`;**Binding Success Response** = `0x0101`。这两个魔数值你会在后面 demo 的抓包里反复看到。

> 工程提醒:不要手算这两个 bit。任何成熟库(Pion/libwebrtc/coturn)都封装好了 `MethodBinding | ClassRequest`。理解它的意义是为了**抓包时能读懂第一个字节**。

### 6.2.3 属性 TLV 与 4 字节对齐

【真实源码 pion/stun@attributes.go】(经 WebFetch 核实)

```go
type RawAttribute struct {
	Type   AttrType
	Length uint16 // 编码时被忽略(按 Value 实际长度写)
	Value  []byte
}

// 4 字节对齐:不足 4 的倍数则补 1~3 个 padding 字节
const padding = 4

func nearestPaddedValueLength(l int) int {
	n := padding * (l / padding)
	if n < l {
		n += padding
	}
	return n
}
```

常用属性类型(WebFetch 核实其 hex 值):

| 常量 | 值 | 含义 |
|------|------|------|
| `AttrMappedAddress` | `0x0001` | 旧式明文地址(已被 XOR 版取代) |
| `AttrErrorCode` | `0x0009` | 错误码(如 401 需鉴权,用于 TURN) |
| `AttrXORMappedAddress` | `0x0020` | **核心**:异或编码的反射地址 |
| `AttrSoftware` | `0x8022` | 软件标识(`0x8000+` 为 optional,可忽略) |

> **坑**:`Length` 字段是 **value 的真实长度(不含 padding)**,但报文里紧跟其后的字节要按 `nearestPaddedValueLength` 补齐到 4 的倍数。解析时:读 Length → 取 Length 字节作为 value → 但指针要前进 `nearestPaddedValueLength(Length)` 字节才到下一个属性。手写解析器最容易在这里错位。

### 6.2.4 XOR-MAPPED-ADDRESS:为什么要异或?(STUN 的灵魂)

如果直接用明文 `MAPPED-ADDRESS` 回传你的公网地址,会有一个真实问题:**某些老式 NAT / ALG(Application Layer Gateway)会"贴心地"扫描 UDP payload,发现里面有一个 IP 地址长得像内网/外网地址,就顺手把它也改写了** —— 结果客户端拿到一个被二次污染的地址。RFC 5389 的解法:把地址和端口跟 magic cookie 异或后再传,NAT 看不懂就不会乱改。

【真实源码 pion/stun@xoraddr.go】(经 WebFetch 核实)

```go
type XORMappedAddress struct {
	IP   net.IP
	Port int
}

// ---- 编码 AddToAs() ----
// 端口:与 magic cookie 的高 16 位异或
bin.PutUint16(value[2:4], uint16(a.Port^magicCookie>>16))

// 地址:与 (magic cookie ++ transaction ID) 逐字节异或
xorValue := make([]byte, net.IPv6len)        // 16 字节缓冲(兼容 IPv4/IPv6)
bin.PutUint32(xorValue[0:4], magicCookie)    // 前 4 字节 = magic cookie
copy(xorValue[4:], msg.TransactionID[:])     // 后 12 字节 = transaction ID
xor.XorBytes(value[4:4+len(ip)], ip, xorValue)

// ---- 解码 GetFromAs() ----(反向,用相同的异或源)
a.Port = int(bin.Uint16(value[2:4])) ^ (magicCookie >> 16)
xorValue := make([]byte, 4+TransactionIDSize)
bin.PutUint32(xorValue[0:4], magicCookie)
copy(xorValue[4:], msg.TransactionID[:])
xor.XorBytes(a.IP, value[4:], xorValue)
```

逐行要点:
- **端口** 只与 cookie 高 16 位(`0x2112`)异或 —— 端口是 16 bit,cookie 高 16 位刚好够。
- **IPv4 地址**(4 字节)只用到 `xorValue` 的前 4 字节,即只跟 magic cookie 异或(transaction ID 用不上);**IPv6 地址**(16 字节)才会用满 cookie++transactionID 全 16 字节。这就是为什么 demo 解码 IPv4 时,你会看到地址前 4 字节跟 `0x2112A442` 对齐异或。
- **异或是对称的**:编码解码用完全相同的 `xorValue`,`a ^ k ^ k == a`。

> 这一段就是下一节 demo 的全部秘密:**发一个 Binding Request,server 回一个带 XOR-MAPPED-ADDRESS 的 Success Response,你把那 8 字节(family+port+ip)与 magic cookie 异或回去,就是你的公网 (IP:端口)。**

---

## 6.3 ⭐ Demo A:亲手发一个 STUN Binding Request 拿到公网地址

**目标**:不依赖任何浏览器,用一段最小代码向公共 STUN server(`stun.l.google.com:19302`)发一个 Binding Request,解析返回的 XOR-MAPPED-ADDRESS,打印你的 `公网IP:端口`。

### 版本 1:Pion(Go)—— 推荐,最贴近 WebRTC 生产栈

**前置**:`go 1.21+`;`go get github.com/pion/stun/v3`(若用旧版去掉 `/v3`)。

`main.go`:
```go
package main

import (
	"fmt"
	"log"

	"github.com/pion/stun/v3"
)

func main() {
	// 1. 连接公共 STUN server(Google 的,UDP 19302)
	c, err := stun.Dial("udp4", "stun.l.google.com:19302")
	if err != nil {
		log.Fatalf("dial: %v", err)
	}
	defer c.Close()

	// 2. 构造一个 Binding Request,带随机 transaction ID
	message := stun.MustBuild(stun.TransactionID, stun.BindingRequest)

	// 3. 发送并等待响应(回调里解析 XOR-MAPPED-ADDRESS)
	if err := c.Do(message, func(res stun.Event) {
		if res.Error != nil {
			log.Fatalf("do: %v", res.Error)
		}
		var xorAddr stun.XORMappedAddress
		if err := xorAddr.GetFrom(res.Message); err != nil {
			log.Fatalf("getFrom: %v", err)
		}
		// 4. 打印:这就是 NAT 外面看到的你
		fmt.Printf("你的 server-reflexive 地址 (srflx) = %s:%d\n",
			xorAddr.IP, xorAddr.Port)
	}); err != nil {
		log.Fatalf("send: %v", err)
	}
}
```
【示意,非逐字 —— API 名 `stun.Dial` / `MustBuild` / `BindingRequest` / `XORMappedAddress.GetFrom` 是 Pion 公开 API,高把握;但本段未逐行从某单一文件 WebFetch,故标示意。底层 `GetFrom` 调用的就是 6.2.4 那段已核实的异或解码逻辑。】

**运行**:
```bash
go mod init stundemo && go get github.com/pion/stun/v3
go run main.go
```

**预期输出**(地址会是你出口的公网 IP):
```
你的 server-reflexive 地址 (srflx) = 203.0.113.42:54231
```

**怎么验证它是对的**:
1. 浏览器打开 `https://www.whatismyip.com/`,对比 IP 是否一致(端口对不上很正常 —— 浏览器和你 Go 程序用的是不同的本地端口,出口映射端口自然不同)。
2. 连续运行两次,观察端口:
   - 端口**每次都变** → 你大概率在 symmetric NAT 或带端口随机化的 NAT 后(对打洞不友好);
   - 端口**相对稳定**(同一本地端口出去映射一致)→ 偏 cone 型,打洞友好。
   - ⚠️ 严格的 NAT 行为判断要做 RFC 5780 的多 server / 多端口测试,单次 Binding 只能粗判,**不要据此下死结论**。

### 版本 2:浏览器里"白嫖" STUN(无需任何 server 代码)

如果你连 Go 都不想装,可以用浏览器的 `RTCPeerConnection`,让它替你发 STUN 请求,从 ICE candidate 里读出 srflx:

`stun.html`(直接双击或 `python3 -m http.server` 起一个 8000 端口打开):
```html
<!DOCTYPE html>
<html><body>
<pre id="log"></pre>
<script>
const log = (s) => document.getElementById('log').textContent += s + '\n';

// 只配 STUN(Google 公共),不配 TURN
const pc = new RTCPeerConnection({
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
});

// 必须有 transceiver/datachannel 才会触发候选收集
pc.createDataChannel('probe');

pc.onicecandidate = (e) => {
  if (!e.candidate) { log('--- 候选收集结束 ---'); return; }
  const c = e.candidate;
  // candidate 字符串形如:
  // candidate:1 1 udp 1686052607 203.0.113.42 54231 typ srflx raddr ...
  log(`[${c.type}] ${c.address}:${c.port}  (proto=${c.protocol}, prio=${c.priority})`);
};

pc.createOffer().then(o => pc.setLocalDescription(o));
</script>
</body></html>
```

**预期**:控制台/页面打印多行候选:
```
[host] 192.168.1.20:51000  (proto=udp, prio=126...)
[srflx] 203.0.113.42:54231  (proto=udp, prio=100...)   <- 这就是 STUN 反射出来的公网地址
--- 候选收集结束 ---
```
- `typ host` = 本机网卡地址(内网);
- `typ srflx` = STUN server 反射回来的公网地址 —— **和版本 1 的输出应该是同一个公网 IP**;
- 如果**只有 host 没有 srflx**,说明 STUN 请求没出去(防火墙挡了 UDP 19302,或在某些受限网络)。

> 这两个版本一对照,你就彻底打通了"STUN Binding → XOR-MAPPED-ADDRESS → srflx candidate → 进 SDP"这条链路。版本 1 是裸协议,版本 2 是 WebRTC 把它包好了。

---

## 6.4 ICE 框架:候选、配对、连通性检查、选路

STUN/TURN 是弹药,ICE 是指挥官。RFC 8445 的核心流程分四步:**Gathering → Pairing → Connectivity Check → Nomination**。

### 6.4.1 四类候选(candidate type)

【真实源码 pion/ice@candidatetype.go】(经 WebFetch 核实)

```go
type CandidateType byte

const (
	CandidateTypeUnspecified     CandidateType = iota
	CandidateTypeHost                  // 本机网卡地址
	CandidateTypeServerReflexive       // STUN 反射:srflx
	CandidateTypePeerReflexive         // 连通检查中对端"意外"暴露的地址:prflx
	CandidateTypeRelay                 // TURN 中继:relay
)

func (c CandidateType) Preference() uint16 {
	switch c {
	case CandidateTypeHost:
		return 126
	case CandidateTypePeerReflexive:
		return 110
	case CandidateTypeServerReflexive:
		return 100
	case CandidateTypeRelay, CandidateTypeUnspecified:
		return 0
	}
	return 0
}
```

【真实源码 pion/ice@candidate.go】(经 WebFetch 核实,接口节选)

```go
type Candidate interface {
	Foundation() string  // 用于 freezing 算法的分组键
	ID() string
	Component() uint16    // RTP=1, RTCP=2
	Priority() uint32
	Type() CandidateType
	Address() string
	Port() int
	RelatedAddress() *CandidateRelatedAddress // srflx/relay 的 base 地址(诊断用)
	NetworkType() NetworkType
	LastReceived() time.Time
	LastSent() time.Time
	// ...
}
```

四类候选的来历(高把握):

| 类型 | 怎么来的 | 类型偏好(type pref) | 典型可达性 |
|------|----------|------|------|
| **host** | 直接枚举本机网卡 IP | 126(最高) | 同一 LAN / 公网直连机器才通 |
| **srflx** | 向 STUN server 发 Binding,拿反射地址 | 100 | 双方至少一端 cone NAT 时通 |
| **prflx** | 连通检查时,对端从一个你 SDP 里**没列过**的源地址收到了你的包,这个地址被动学习成 prflx | 110 | symmetric NAT 场景下临时冒出来 |
| **relay** | 向 TURN server 申请 Allocation,拿中继地址 | 0(最低) | **永远能通**(兜底) |

> **prflx 是最反直觉的一个**:你不主动收集它。它是在连通性检查过程中,因为 NAT 给"发往 peer"的流量分配了一个新映射端口(symmetric 行为),对端从这个**新端口**收到了你的 STUN check,于是把这个新 (IP:端口) 学习为一个 peer-reflexive 候选。它是 ICE "让数据说话、动态发现路径"的精髓体现。

### 6.4.2 候选优先级(priority)公式

RFC 8445 §5.1.2 的优先级公式(高把握,这是写进协议的硬规定):

```
priority = (2^24) * type-preference
         + (2^8)  * local-preference
         + (2^0)  * (256 - component-ID)
```
- `type-preference`:就是上面 `Preference()` 的 126/110/100/0,**保证 host > prflx > srflx > relay**。
- `local-preference`:同类型候选间的细分(多网卡/多协议时区分,如 IPv6 优于 IPv4、UDP 优于 TCP)。
- `component-ID`:RTP=1 优先于 RTCP=2(`256-1 > 256-2`)。

> ⚠️ **待核(Pion 文件路径)**:Pion 里 `Priority()` 的具体实现文件(应在 `candidatebase.go` 一类)WebFetch 取到 404,未能逐字核实那段 `1<<24 / 1<<8` 的位移代码。但**公式本身来自 RFC 8445,确定无疑**;Pion 的 `Preference()` 返回值(126/110/100/0)已逐字核实(见上)。

补充已核实的一个细节:relay candidate 内部还按转发协议再排优先级。

【真实源码 pion/ice@candidate_relay.go】(经 WebFetch 核实)
```go
preferenceRelayTLS  = 0
preferenceRelayTCP  = 1
preferenceRelayDTLS = 2
preferenceRelayUDP  = 3   // UDP 中继优先级最高(开销最小)
```
即"中继也要尽量用 UDP,TURN-over-TLS 最重所以排最后"——这套常量直接抄自 libwebrtc 的 p2p constants(Pion 注释说明)。

### 6.4.3 candidate pair 与连通性检查

ICE 双方交换各自的候选列表后,**两两配对**成 candidate pair `(local, remote)`,pair priority 用一个对称公式(RFC 8445 §6.1.2.3)合成:

```
pair priority = 2^32 * MIN(G,D) + 2 * MAX(G,D) + (G>D ? 1 : 0)
```
(G = controlling 端候选优先级,D = controlled 端候选优先级。设计意图:两端独立计算出来的 pair 排序完全一致。)

然后按 priority 从高到低,对每个 pair **发 STUN Binding Request(这次不是发给 STUN server,而是直接发给对端候选地址)**,做"连通性检查":

- 检查请求里带 `USERNAME` / `MESSAGE-INTEGRITY`(用 SDP 里交换的 `ice-ufrag` / `ice-pwd` 计算),对端用同一份凭证校验 —— **这把 STUN check 和你这次会话绑定,防止串扰/攻击**。
- 还带 `PRIORITY` 属性(本端给这条候选算的优先级)和角色属性 `ICE-CONTROLLING` / `ICE-CONTROLLED`。
- 对端收到合法 check → 回 Binding Success Response(里面带 XOR-MAPPED-ADDRESS,即"我看到你这个包的源地址")→ 这条 pair 进入 **Succeeded** 状态。
- 同时双方都会**反向**对收到 check 的源地址发起检查(triggered check),这就是 prflx 被发现并加入的时机。

> ICE 的角色:**controlling agent**(通常是 offerer)负责最终拍板用哪条 pair(nomination);**controlled agent** 配合。Full ICE 用 regular nomination(先全检查再提名),早期/精简实现用 aggressive nomination。

### 6.4.4 选路与状态机(connection state)

所有 succeeded pair 里,controlling 端挑一条(通常是优先级最高的 succeeded pair)发**带 `USE-CANDIDATE` 标记的 check**,提名它为 **selected pair** → ICE 进入 `connected`,媒体开始走这条路。后续可能继续检查、若出现更优 pair 还能切换。

ICE 连接状态(浏览器 `pc.iceConnectionState`,你在 demo B 会看到):
```
new → checking → connected → completed
                    ↘ (中途全挂) ↘
                  disconnected → failed
                              ↘ closed
```
- `checking`:正在并行打各 pair;
- `connected`:至少一条 pair 通了,媒体可发;
- `completed`:controlling 端确认检查全部完成;
- `disconnected`:之前通的路暂时收不到包(可能网络抖动,会自愈);
- `failed`:所有 pair 都失败(典型:两端 symmetric NAT 且没配 TURN)。

---

## 6.5 ⭐ Demo B:浏览器里看完整 ICE 过程(webrtc-internals)

**目标**:在**一个浏览器、一个页面内**建立两个 `RTCPeerConnection`(pc1 ↔ pc2)互连,用 `chrome://webrtc-internals` 看候选收集、candidate pair、selected pair、连接状态。这是观察 ICE 全流程最快、最不依赖外部服务的方式。

`ice-loopback.html`:
```html
<!DOCTYPE html>
<html><body>
<button id="go">建立连接</button>
<pre id="log"></pre>
<script>
const log = (s) => { document.getElementById('log').textContent += s + '\n'; };
const ICE = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };

document.getElementById('go').onclick = async () => {
  const pc1 = new RTCPeerConnection(ICE);
  const pc2 = new RTCPeerConnection(ICE);

  // 互相转发 ICE 候选(本页内充当"信令通道")
  pc1.onicecandidate = e => e.candidate && pc2.addIceCandidate(e.candidate);
  pc2.onicecandidate = e => e.candidate && pc1.addIceCandidate(e.candidate);

  // 观察状态机
  pc1.oniceconnectionstatechange =
    () => log('pc1 iceState=' + pc1.iceConnectionState);

  // pc1 建一个 datachannel 触发协商
  const dc = pc1.createDataChannel('chat');
  dc.onopen = () => { log('datachannel OPEN ✅'); dc.send('hello'); };
  pc2.ondatachannel = e =>
    (e.channel.onmessage = m => log('pc2 收到: ' + m.data));

  // 标准 offer/answer 握手
  const offer = await pc1.createOffer();
  await pc1.setLocalDescription(offer);
  await pc2.setRemoteDescription(offer);
  const answer = await pc2.createAnswer();
  await pc2.setLocalDescription(answer);
  await pc1.setRemoteDescription(answer);

  // 5 秒后打印 pc1 选中的 candidate pair
  setTimeout(async () => {
    const stats = await pc1.getStats();
    stats.forEach(r => {
      if (r.type === 'candidate-pair' && r.nominated && r.state === 'succeeded') {
        log(`selected pair: state=${r.state} nominated=${r.nominated} ` +
            `bytesSent=${r.bytesSent} rtt=${r.currentRoundTripTime}`);
      }
    });
  }, 5000);
};
</script>
</body></html>
```

**运行步骤**:
1. `python3 -m http.server 8000`,Chrome 打开 `http://localhost:8000/ice-loopback.html`。
2. **先**新开一个 tab 打开 `chrome://webrtc-internals`(它只记录打开之后创建的 PC)。
3. 回到页面点"建立连接"。

**预期(页面)**:
```
pc1 iceState=checking
datachannel OPEN ✅
pc1 iceState=connected
pc2 收到: hello
selected pair: state=succeeded nominated=true bytesSent=... rtt=0.000x
```

**预期(`webrtc-internals` 里要看的东西)**:
- 展开对应的 PC,看 **`Stats Tables` → `candidate-pair`**:能看到多条 pair,其中一条 `state=succeeded` 且 `nominated=true`,`priority` 最高;
- 看 `local-candidate` / `remote-candidate`:因为是本机环回,会以 `host` 候选(`127.0.0.1`/局域网 IP)直接命中,**不需要走 srflx**(本机互连 host pair 永远最优先且最先通)—— 这正好印证 host(126)优先级最高;
- 看 `ICE connection state` 时间线:`new→checking→connected→completed` 的跳变时间戳;
- 看 `RTCIceCandidatePair` 的 `requestsSent/responsesReceived`:这就是 6.4.3 那些 STUN connectivity check 的计数。

> **想看 srflx/relay 怎么办?** 本机环回只会用 host。要观察真实 NAT 穿透,需要两台不同网络的机器(或一台 + 手机 4G 热点)各跑一端,通过一个真实信令通道交换 SDP。那时 `webrtc-internals` 里才会出现 `srflx`、甚至配了 TURN 后的 `relay` pair。本 demo 的价值是**零依赖看清 ICE 状态机和 candidate-pair 结构**。

---

## 6.6 TURN:兜底中继(什么时候不得不用)

当连通性检查把所有 host/srflx pair 都打成 failed(典型:**symmetric NAT × symmetric NAT**,双方的 srflx 互相无效,prflx 也学不出可用路径),唯一的活路是 **TURN 中继**。

### TURN 的工作模型(高把握,RFC 8656)

1. 客户端向 TURN server 发 **Allocate Request**(STUN method `0x003`,通常**必须鉴权** —— 第一次返回 `401 Unauthorized` + nonce,第二次带 `MESSAGE-INTEGRITY` 重发,这就是 `AttrErrorCode 0x0009` 的用武之地)。
2. TURN server 在自己的公网地址上分配一个 **relayed transport address**(中继地址)给你,返回给客户端。这个 relay 地址就是你的 relay candidate,塞进 SDP 给对端。
3. 对端把媒体发到你的 relay 地址 → TURN server 转发给你;你回的包也经 TURN 中转。**数据全程绕道 server**。
4. 通过 **CreatePermission / ChannelBind** 控制哪些 peer 地址被允许中继(防滥用),`ChannelBind` 还能把每个包的 STUN 封装头从 36 字节压到 4 字节(ChannelData),省带宽。

### TURN 的代价(为什么是最后手段)

- **带宽成本**:所有媒体流量过你的 server,运营 TURN 是真金白银(对视频通话尤其烧)。
- **延迟**:数据绕远路,RTT 增加。
- **单点**:relay 路径依赖 server 可用性。

所以 ICE 给 relay 的 type preference = **0**(最低):**只要有任何直连 pair 能通,绝不用 relay**。生产部署的经验法则:**STUN 免费随便配(白嫖 Google 的也行),TURN 一定要自建/买(coturn 是事实标准),且监控"relay 占比"** —— relay 占比异常高通常意味着大量用户在 symmetric NAT 后,或你的 STUN/打洞配置有问题。

---

## 6.7 方案对比与边界

### STUN vs TURN vs 纯 host 直连

| 维度 | host 直连 | STUN(srflx 打洞) | TURN(relay 中继) |
|------|------|------|------|
| 谁产出候选 | 本机网卡枚举 | STUN Binding 反射 | TURN Allocate |
| 成功条件 | 同 LAN / 公网机器 | ≥1 端 cone NAT | **总是成功** |
| 服务器开销 | 无 | 极小(只回一个包) | **大(转发全部媒体)** |
| 延迟 | 最低 | 低(直连) | 高(绕道) |
| ICE type pref | 126 | 100 | 0 |
| 失败场景 | 跨 NAT 必失败 | symmetric × symmetric 失败 | 几乎不失败(除非 server 挂) |
| 典型占比* | LAN/公司内网 | 大多数家庭宽带 | symmetric NAT、严格企业防火墙 |

\* 占比是经验性大致描述,具体看用户网络分布,**待核**精确数字(各家公开数据不一,常见说法是直连/打洞能覆盖 70%~90%、剩余走 relay,因网络环境差异很大)。

### 什么场景该怎么配

- **内网工具 / 同公司 demo**:host 候选就够,STUN 可省。
- **公网 C 端音视频(主流)**:STUN(可白嫖) + **自建 TURN**(coturn)兜底,这是标配。没有 TURN 的产品,在 symmetric NAT 用户那里就是"有时连不上",体验灾难。
- **企业内网 + 严格防火墙(只放 443)**:很可能 UDP 全被封,需要 **TURN over TCP/TLS(443 端口)** 才能穿,此时 relay 占比会很高,这是正常的。
- **追求极致直连率**:多配几个不同地理位置的 STUN/TURN、开 IPv6(IPv6 常常无 NAT,直接 host 通)、用 Trickle ICE 降延迟。

---

## 6.8 失败模式 / 真坑 / 根因

1. **只有 host 候选,没有 srflx** → STUN 请求没出去。根因:UDP 19302 被防火墙挡 / 网络只放特定端口 / STUN server 地址写错。验证:用 6.3 Demo A 版本 1 单独测 STUN 可达性。

2. **两端都收集到 srflx,但连不上,最终 failed** → 大概率 **symmetric NAT × symmetric NAT**。根因:srflx 地址对 peer 无效(NAT 对每个目的地重分配端口)。解法:**必须上 TURN**。这是"为什么我配了 STUN 还是连不上"的头号答案。

3. **`iceConnectionState` 反复 `disconnected ↔ connected` 抖动** → 网络丢包/切换(WiFi↔4G)/ NAT 映射超时(binding timeout)。根因之一:**NAT 映射有老化时间(常见 30s~数分钟),长时间没流量映射被回收**。解法:开启 **ICE consent freshness(RFC 7675)**,定期发 STUN check 保活;媒体层也靠 RTCP 保活。

4. **建连慢(几秒才 connected)** → 没用 Trickle ICE,等齐所有候选(尤其 TURN Allocate 慢)才发 offer。根因:non-trickle 把"收集"和"发送"串行化。解法:开 Trickle ICE(第 1 章),候选边收边发。

5. **TURN 配了但不生效,relay 候选收不到** → TURN 鉴权失败(username/credential 错、时效凭证过期)、或 TURN server 端口/协议没开对。根因:TURN 几乎总要鉴权(401 + nonce 流程),凭证错就拿不到 Allocation。验证:看 `webrtc-internals` 有没有 `relay` 类型 local-candidate;用 coturn 的 `turnutils_uclient` 单测 TURN server。

6. **手算 STUN type / attribute padding 错位**(自己写解析器时) → 6.2.2 的 14-bit 交错编码、6.2.3 的"Length 不含 padding 但指针要按 4 对齐前进"。根因:协议设计的两个反直觉点。解法:别手撸,用 Pion/libnice/coturn;非要撸就照 `nearestPaddedValueLength` 推进指针。

7. **mDNS 候选 `.local` 看不到真实内网 IP**(隐私特性) → 现代浏览器默认把 host 候选的内网 IP 用 mDNS `xxxx.local` 名字遮蔽,防 IP 泄露指纹。根因:不是 bug,是 privacy 设计。影响:同一 LAN 内两端若都用 mDNS 且不能互相解析 `.local`,host 直连可能退化为走 srflx。

---

## 6.9 章末五件套

### ① 一句话本质
ICE = "**不预判 NAT,把 host/srflx/relay 所有候选摆出来,两两配对并行用带凭证的 STUN check 去打,让能通的那条路自己冒出来,优先直连、relay 兜底**";STUN 是照镜子拿公网落点的工具,TURN 是打不通时的中继。

### ② 必记数字 / 常量
- magic cookie = **`0x2112A442`**;STUN 头 **20 字节**;transaction ID **12 字节**;属性 **4 字节对齐**。
- Binding Request type = **`0x0001`**,Binding Success Response = **`0x0101`**。
- type preference:**host 126 > prflx 110 > srflx 100 > relay 0**。
- priority 公式:`2^24·type + 2^8·local + (256−component)`。
- 常用属性:`XOR-MAPPED-ADDRESS 0x0020`、`ERROR-CODE 0x0009`、`MAPPED-ADDRESS 0x0001`。

### ③ 决策清单(配 ICE server 时)
- [ ] 配了 STUN 吗?(几乎零成本,必配;可白嫖 `stun.l.google.com:19302` 做 demo,生产建议自建)
- [ ] **配了 TURN 吗?**(C 端产品**必须**,否则 symmetric NAT 用户连不上)
- [ ] TURN 支持 TCP/TLS 443 吗?(穿严格企业防火墙)
- [ ] 开了 Trickle ICE 吗?(降建连延迟)
- [ ] 开了 IPv6 吗?(常无 NAT,直接 host 通,提直连率)
- [ ] 监控 relay 占比了吗?(异常高 = NAT 环境差或打洞配置有问题)

### ④ 自测题
1. 为什么 STUN 要用 XOR-MAPPED-ADDRESS 而不是明文 MAPPED-ADDRESS?(答:防 NAT/ALG 扫描 payload 时二次改写内嵌地址)
2. 你配了 STUN 但两端死活连不上,最可能是什么 NAT 组合?怎么救?(答:symmetric × symmetric;上 TURN)
3. peer-reflexive 候选是怎么"长出来"的?为什么收集阶段拿不到它?(答:连通检查时 NAT 给发往 peer 的流量分新端口,对端从未声明过的源地址收到 check 后动态学习而成)
4. relay candidate 的 type preference 为什么是 0?(答:中继烧带宽、增延迟、单点,只做兜底,有直连绝不用)
5. STUN 报文头里"最高两位恒为 0"和 magic cookie 共同实现了什么?(答:与 RTP/DTLS 等复用同一 UDP 端口时的协议 demultiplex)

### ⑤ 延伸阅读(出处)
- **RFC 8445** ICE(现行框架,取代 5245)— 候选/配对/检查/nomination/priority 公式全在这。
- **RFC 8489** STUN(现行,取代 5389;5389 又取代 3489)— 报文格式、XOR-MAPPED-ADDRESS、MESSAGE-INTEGRITY。
- **RFC 8656** TURN(现行,取代 5766)— Allocate / CreatePermission / ChannelBind。
- **RFC 4787** NAT 行为术语(mapping/filtering 两维度,取代 3489 的四分类)。
- **RFC 5780** 通过 STUN 探测 NAT 行为(做严格 NAT 类型判断的正规方法)。
- **RFC 7675** ICE consent freshness(保活,防映射老化)。
- **RFC 8838** Trickle ICE(增量候选)。
- 源码:**pion/stun**(`message.go` / `attributes.go` / `xoraddr.go` 本章已核实)、**pion/ice**(`candidate.go` / `candidatetype.go` / `candidate_relay.go` 本章已核实)、**coturn**(生产级 STUN/TURN server 事实标准)。

---

### 本章源码核实声明
- 【已逐字 WebFetch 核实】pion/stun@message.go(Message struct / magicCookie / 头布局)、pion/stun@attributes.go(RawAttribute / nearestPaddedValueLength / 属性类型值)、pion/stun@xoraddr.go(XOR 编解码)、pion/ice@candidate.go(Candidate 接口)、pion/ice@candidatetype.go(CandidateType + Preference 126/110/100/0)、pion/ice@candidate_relay.go(relay 协议偏好常量)。
- 【示意 / 未逐字】STUN type 14-bit 交错编码图、priority 公式(均来自对应 RFC,高把握,但 Pion 对应文件 WebFetch 取到 404 未能逐行核实)、Demo A 版本 1 的 Pion 高层 API 调用(API 名高把握)。
- 【待核】各 NAT 类型在真实用户中的占比数字;STUN type/priority 在 Pion 当前代码树中的确切文件路径。
