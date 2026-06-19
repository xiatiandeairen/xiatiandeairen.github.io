---
title: "高性能网络：XDP/eBPF 数据面（深化版）"
slug: "9-21"
collection: "tech-library"
group: "linux系统"
order: 9021
summary: "一句话定位：当内核网络栈成为瓶颈，XDP 让你在驱动 NAPI poll 循环里、`alloc_skb()` 之前用一段经 verifier 证明过安全的 eBPF 程序拦截数据包——以接近裸硬件的延迟，却不放弃内核的 RCU、tracing、map、与 TCP/IP 栈的互操作。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T09:29:48.000Z"
updatedAt: "2026-06-12T09:29:48.000Z"
---
> 一句话定位：当内核网络栈成为瓶颈，XDP 让你在驱动 NAPI poll 循环里、`alloc_skb()` 之前用一段经 verifier 证明过安全的 eBPF 程序拦截数据包——以接近裸硬件的延迟，却不放弃内核的 RCU、tracing、map、与 TCP/IP 栈的互操作。

---

## TL;DR — 本章最硬结论

1. **内核网络栈的大头开销不是 TCP/IP 算法，而是"路径长度 × 每跳的 cache/锁成本"**。Cloudflare 实测在同一台机器上、纯丢包这一个动作，从用户态 socket（175 kpps）一路到 XDP_DROP（10.1 Mpps）跨了约 **57 倍**——中间每一层（BPF socket filter 512 kpps、iptables INPUT 608 kpps、iptables raw PREROUTING 1.69 Mpps、tc u32 1.8 Mpps）都在为"包已经被做成 skb、已经走了一段栈"埋单。【史料：Cloudflare "How to drop 10 million packets per second"】
2. **XDP = 在驱动 NAPI poll 里、skb 分配之前插入的 eBPF 钩子**。内核侧 hook 入口是 `bpf_prog_run_xdp()`（`include/net/xdp.h`），驱动在自己的 poll 里直接调它，拿到 `XDP_{ABORTED,DROP,PASS,TX,REDIRECT}` 五个返回码之一就地裁决。单核 XDP_DROP 在 mlx5 上实测可达 **~25.9 Mpps**，4 队列近线性到 ~82 Mpps。【史料：xdp-paper 实验数据 / CoNEXT'18】
3. **AF_XDP 是用户态的快速通道**：XDP 程序把包 `XDP_REDIRECT` 进 `xskmap`，用户态经 UMEM + 四个 ring（FILL/COMPLETION/RX/TX）零拷贝收发。它**不独占 NIC**——同一张卡可以一半流量进 AF_XDP、一半 `XDP_PASS` 给内核 TCP/IP，这是与 DPDK 的根本架构差异。
4. **XDP 的崛起是一场"反 bypass"的设计胜利**：2016 年 Brenden Blanco 的初版 patch 被社区**打回重做**（Alexei 一句"We have to plan the whole project, so we can incrementally add features without breaking abi"定调），最终演化成今天可热更新、可观测、跨驱动的字节码数据面。Cilium、Katran、Cloudflare Gatebot 都建在它上面。【史料：LWN 682538、commit 6a773a15a1e8】
5. **绕过内核（DPDK/RDMA）有真实代价**：DPDK RX-only 峰值确实更高（多核 ~115 Mpps），但代价是 tcpdump/perf/eBPF 全部失明、PMD 与内核版本强绑定、需要专人维护。XDP 用"慢一点点（24 vs 43+ Mpps/core）换回整个内核生态"——这个 trade-off 才是它被工业界选中的真正理由。

---

## 前置依赖章

- 第 18 章（中断与下半部 / NAPI softirq）——理解包从 NIC 进来、`napi->poll()` 在哪个上下文跑。XDP 钩子就活在这个 poll 里。
- 第 20 章（网络协议栈）——理解 skb 的生命周期、netfilter hook 的位置、GRO 在哪一层。XDP 绕过的正是这些。
- 第 23 章（eBPF）——verifier、map、helper、CO-RE 的完整讲解在那里。本章只取"网络数据面"这个切面，需要时引用。

---

## 21.0 本章的"扎根"方法论（先说怎么读）

这一版与浮在表面的讲法的区别，是每个机制都要回答四个问题：

1. **演进**：它解决了上一代什么缺陷？社区辩论过什么？（带 LWN / commit / 邮件出处）
2. **源码**：核心控制流在内核哪个文件、哪个函数？逐行看它到底做了什么。（标【真实源码 v6.12 路径】/【示意代码】/「待核」）
3. **对比**：和更简单/更激进的替代相比，跑一个具体场景看差异，并点出不适用边界。
4. **根因**：为什么快/慢的底层原因是什么（cache、排队、锁、内存序）？失败模式长什么样？

> 源码标注约定（贯穿全章）：
> - **【真实源码 v6.12 路径】**：逐字摘自 https://raw.githubusercontent.com/torvalds/linux/v6.12/ 对应文件，未改动。
> - **【示意代码，非逐字】**：为讲清控制流我自己写/简化的，符号名可能是占位。
> - **「待核」**：我没把握、未取到原文的，明确标出来别当事实背。

---

## 21.1 内核网络栈的开销来源：从"一个包的旅程"到可量化的钱

### 21.1.1 一个包的旅程（复习第 20 章，但这次盯着"花在哪"）

```
NIC DMA → RX ring buffer（硬件写描述符 + payload 进内存）
  ↓  硬中断
NAPI poll（softirq, NET_RX_SOFTIRQ）            ← XDP 钩子就插在这里，下面细讲
  ├─ alloc_skb() / napi_build_skb()             ← slab 分配 + 元数据初始化
  ├─ DMA unmap / dma_sync                       ← 缓存一致性
  └─ eth_type_trans()                           ← 定协议
  ↓
__netif_receive_skb_core()
  ├─ GRO（聚合小包，减少后续遍历次数）
  ├─ tcpdump / packet taps（如果有 AF_PACKET listener）
  └─ packet_type handler（ip_rcv 等）
  ↓
IP 层：ip_rcv → netfilter PREROUTING → 路由查找（FIB）→ netfilter ...
  ↓
TCP 层：tcp_v4_rcv → 查 socket hash → 序号/ACK/拥塞控制 → sk_receive_queue
  ↓
recvmsg() 系统调用 → 拷贝到用户缓冲区
```

每一跳的隐性成本，落到微架构层面是这四类：

- **CPU cache miss**：skb 是 `kmalloc` 出来的堆对象（约 200+ 字节的元数据），路由表、conntrack 表、socket hash 各在不同 cacheline。100Gbps 64B 线速下每包预算只有 **~6.7 ns**（≈ 在 3GHz CPU 上约 20 个周期），一次 LLC miss（~100ns）就够吃掉十几个包的预算。
- **锁竞争**：socket lock、conntrack 的 `nf_conntrack_lock`/bucket lock、iptables 规则在 `ipt_do_table()` 里的线性遍历（O(N) 链规则）。
- **上下文/上半部边界**：softirq → 进程上下文（被 recvmsg 唤醒）→ 用户态，至少两次边界 + 一次调度。
- **内存拷贝**：至少 DMA→skb data（或 DMA map）一次、skb→用户缓冲区一次。

> 根因点睛：网络栈慢的本质不是"算法复杂度高"，而是**指令路径长 + 数据局部性差**。每多走一层，就多碰几条 cacheline、多过一两个锁。XDP 的全部价值就是"在数据还躺在 L1/L2、还没被做成 skb 的那一刻就把判决做了"。

### 21.1.2 量化开销（这次用真实出处的数字，别背量级）

下面这张"丢包能力阶梯"全部来自 **Cloudflare 在单台真实服务器、单 CPU、只做 DROP** 这一个动作的实测（2018，Marek Majkowski）。它比任何"~Xµs"的拍脑袋数字都有说服力，因为它把"路径越短越快"这件事钉死成了一条单调曲线：

| 丢包位置 | 速率（单核，越靠下越靠近驱动） | 为什么这个数 |
|---|---|---|
| 用户态 application（recvmsg 后丢） | **175 kpps** | 走完整栈 + 两次拷贝 + 系统调用 |
| AF_PACKET BPF socket filter | **512 kpps** | 仍在 skb 之后，但省了到用户态那段 |
| iptables INPUT（filter 表） | **608 kpps** | skb 已分配、路由已做、conntrack 已查 |
| iptables PREROUTING（raw 表，notrack） | **1.69 Mpps** | raw 表在 conntrack 之前，省掉 conntrack |
| nftables ingress hook | **1.53 Mpps** | 协议栈很早，但 skb 仍已建 |
| tc ingress + u32 match | **1.8 Mpps** | `sch_handle_ingress`，skb 建好但没进 IP 层 |
| **XDP_DROP（native）** | **10.1 Mpps** | **skb 根本没分配**，驱动 poll 里直接回收页 |

【史料：Cloudflare, "How to drop 10 million packets per second"，blog.cloudflare.com/how-to-drop-10-million-packets】

两个必须读懂的拐点：

- **608 kpps → 1.69 Mpps（iptables filter→raw）**：差距几乎全来自 **conntrack**。raw 表的 `-j NOTRACK` 让包跳过连接跟踪，省掉一次 hash + 锁。这解释了"DDoS 防御为什么要尽量早、尽量不碰 conntrack"。
- **1.8 Mpps（tc）→ 10.1 Mpps（XDP）**：差距全来自 **skb**。tc 已经够早了（在 IP 层之前），但它拿到的是 `struct sk_buff *`；XDP 拿到的是 `struct xdp_buff`——后者根本没经过 `alloc_skb`。这 ~5.6 倍就是"一个 skb 的命"。

> 把它和"转发/REDIRECT"分开看：上面是 DROP（最便宜的动作）。一旦要 `XDP_REDIRECT`（比如送 AF_XDP 或转发到另一网卡），就要把 `xdp_buff` 转成 `xdp_frame`、可能跨 CPU/跨设备入队，单核能力会从 ~25.9 Mpps 掉到 ~8.5 Mpps（见 21.4 的 paper 数据）——**动作越重，每包成本越高，这是后面所有方案对比的基准直觉**。

### 21.1.3 绕过思路的谱系（连续谱，不是二选一）

```
内核全栈        XDP/eBPF         AF_XDP            DPDK / RDMA
  ←────────────────────────────────────────────────────────→
 安全/可观测/通用                                  性能/孤立/裸速
 skb 在,栈全走     无skb,驱动层判决   零拷贝到用户态     PMD 独占 NIC,内核失明
```

关键洞察：XDP 是在**保留内核安全模型（verifier + RCU + map + tracing）**的前提下，把决策点尽量前移。下面整章都在沿这条谱系往右走，每往右一步都要还一笔"可观测性/可维护性"的账。

---

## 21.2 设计考古：XDP 是怎么被"吵"出来的

> 这一节是本章的"史料地基"。不读这段，你只会"知道 XDP 是什么"；读了，你才"知道它为什么长这样、社区否决过哪条更省事的路"。

### 21.2.1 时代背景：2016 年之前，高速包处理只有 DPDK 一条路

2010–2015 年，要在 x86 上跑到 10G+ 线速做包处理，事实标准是 **DPDK**：poll-mode driver（PMD）把 NIC 从内核手里抢过来、用大页 + 用户态轮询彻底绕过内核。它快，但代价是：

- 放弃了内核的 TCP/IP、netfilter、socket——这些得自己在用户态重写。
- 放弃了**全部内核可观测性**：tcpdump、perf、ftrace、后来的 eBPF 都看不见 DPDK 内的包。
- NIC 对内核**完全不可见**，一张卡只能整张给 DPDK。
- PMD 与内核/NIC 固件版本强耦合，维护重。

同期 Cilium/容器网络这边的人（以及 Cloudflare 这种被 DDoS 教育过的人）有个共同诉求：**我想要 DPDK 的速度，但不想丢掉内核**。这就是 XDP 的需求土壤。

### 21.2.2 第一版被打回：LWN 682538 记录的那场辩论（2016-04）

2016 年 4 月，**Brenden Blanco（当时 PLUMgrid）**发出第一版 patch，标题就叫 *"Early packet drop … with BPF"*。Jonathan Corbet 在 LWN 682538 *"Early packet drop — and more — with BPF"*（2016-04-06）里记下了核心争论：

- **目标**：在"最早的时刻——网卡驱动收到包的瞬间——把要丢的包丢掉，最好在做任何协议处理之前"。
- **争论一（skb 是原罪）**：有人指出，哪怕只建一个最小的 skb 也违背初衷，因为"那笔开销里，开头很大一块就是创建 skb 这个结构本身"。→ 这直接决定了 XDP 必须工作在 **skb 之前**，拿一个轻量的 `xdp_buff`。
- **争论二（硬件可移植性）**：如果 BPF 程序被写成"假定有 skb 存在"，那它将来没法下放到硬件（offload）里跑。→ 这决定了 XDP 的 context（`xdp_md`）要**极简、只读、不暴露 skb**。
- **Alexei Starovoitov 定调**：*"We have to plan the whole project, so we can incrementally add features without breaking abi."*（我们得把整个项目规划好，这样才能增量加功能而不破坏 ABI。）

结果：**初版 patch 在当时形态被否，要求先把架构/ABI 规划清楚再来**。这是关键史实——XDP 不是"一发就合入"，而是被社区逼着先想清楚 ABI 边界。性能侧的诱因也写在评论里：当时 `tc + cls_bpf` 能做到 6.5 Mpps，而 XDP 原型能到 **20 Mpps**。

> 面试纠错（高频张冠李戴）：
> - XDP 的**初始作者是 Brenden Blanco**，**David S. Miller（davem）是 merge/committer**，不是作者。很多人把 merge 的人记成作者。
> - Tom Herbert、Alexei Starovoitov 深度参与设计；Jesper Dangaard Brouer（Red Hat）是后续 cpumap/页回收/性能基础设施的核心贡献者（见 21.4 源码里的 Copyright）。

### 21.2.3 第一个真正合入的 commit：`6a773a15a1e8`

打回重做后，落地的奠基 commit 是：

```
commit 6a773a15a1e8874e5eccd2f29190c31085912c95
Author:    Brenden Blanco
Committer: David S. Miller <davem...>
Subject:   bpf: add XDP prog type for early driver filter

    Add a new bpf prog type that is intended to run in early stages of the
    packet rx path. Only minimal packet metadata will be available, hence a
    new context type, struct xdp_md, is exposed to userspace. So far only
    expose the packet start and end pointers, and only in read mode.
    An XDP program must return one of the well known enum values, all other
    return codes are reserved for future use. Unfortunately, this restriction
    is hard to enforce at verification time, so take the approach of warning
    at runtime when such programs are encountered. Out of bounds return codes
    should alias to XDP_ABORTED.
    Acked-by: Alexei Starovoitov ...
```

【史料：github.com/torvalds/linux/commit/6a773a15a1e8874e5eccd2f29190c31085912c95，提交信息逐字引自该 commit 页】

逐句读这段提交信息，能直接推出今天 XDP 的几条铁律：

- *"Only minimal packet metadata … a new context type, struct xdp_md … only … packet start and end pointers, and only in read mode"* → 这就是今天 `xdp_md` 只有 `data`/`data_end`/`data_meta` 那几个 `__u32`、且早期只读的来历（见 21.2.5 源码）。
- *"must return one of the well known enum values … Out of bounds return codes should alias to XDP_ABORTED"* → 解释了为什么"野返回码=XDP_ABORTED"，以及为什么内核要在运行时 `bpf_warn_invalid_xdp_action()` 而不能在 verify 时拦死（返回值是运行期计算出来的，静态证不出来）。下面 21.2.4 的真实驱动代码里你会**逐字**看到这个 `default → bpf_warn_invalid_xdp_action → XDP_ABORTED` 的兜底。

### 21.2.4 演进时间线（节点带出处，细节标「待核」）

| 内核版本（约） | 节点 | 出处/备注 |
|---|---|---|
| 4.8 (2016) | XDP 基础设施 + `xdp_md` 合入，mlx4/ixgbe 初步支持 | commit 6a773a15a1e8（本节） |
| 4.18 (2018) | `XDP_REDIRECT` 成熟，devmap/cpumap | devmap.c©2017 Covalent、cpumap.c©2017 Brouer（21.4 源码） |
| 4.18 (2018) | **AF_XDP 合入**（初版**不含**零拷贝） | LWN 750845（21.4） |
| 5.x | AF_XDP 零拷贝、`bpf_redirect_map`、BTF/CO-RE 起步 | 「待核」具体版本号 |
| 5.18+ | XDP multi-buffer（`XDP_FLAGS_HAS_FRAGS`，支持 jumbo/分片） | 见 21.2.5 `xdp_buff.flags`/`xdp_buff_has_frags()` |
| 6.x | `bpf_loop`、CO-RE 普及、更多 helper/驱动 | 「待核」逐项版本 |

### 21.2.5 核心数据结构：`xdp_buff` vs `xdp_md`（一个内核实体，一个 UAPI 视图）

先看 BPF 程序里你能碰到的 **UAPI context**——它故意极简、只读：

```c
/* 【真实源码 v6.12 路径】include/uapi/linux/bpf.h */
enum xdp_action {
	XDP_ABORTED = 0,
	XDP_DROP,
	XDP_PASS,
	XDP_TX,
	XDP_REDIRECT,
};

/* user accessible metadata for XDP packet hook
 * new fields must be added to the end of this structure
 */
struct xdp_md {
	__u32 data;
	__u32 data_end;
	__u32 data_meta;
	/* Below access go through struct xdp_rxq_info */
	__u32 ingress_ifindex; /* rxq->dev->ifindex */
	__u32 rx_queue_index;  /* rxq->queue_index  */

	__u32 egress_ifindex;  /* txq->dev->ifindex */
};
```

逐行注解：
- `XDP_ABORTED = 0` 排第一且为 0：呼应 21.2.3 commit 的"out-of-bounds 别名到 ABORTED"，也是 `static int act = XDP_DROP/ABORTED` 这类默认初值的语义来源。
- `__u32 data / data_end`：注意是 **`__u32` 而不是指针**！它们是"相对偏移"，verifier 在加载时会把对 `ctx->data` 的访问**改写**成对内核真实 `xdp_buff->data`（真指针）的访问。这就是为什么你在 C 里写 `(void*)(long)ctx->data` 能拿到真实内核地址。
- `data_meta`：XDP **可写**的 metadata 区，默认与 `data` 重合，靠 `bpf_xdp_adjust_meta` 撑开，用来给后续 tc BPF / skb 传递自定义信息。
- 注释 *"new fields must be added to the end"*：这就是 21.2.2 里 Alexei 那句"不破坏 ABI"的代码级落地——`egress_ifindex` 是后加的，只能加在末尾。

再看内核侧的**真实实体**（驱动填充、helper 操作的就是它）：

```c
/* 【真实源码 v6.12 路径】include/net/xdp.h */
struct xdp_buff {
	void *data;
	void *data_end;
	void *data_meta;
	void *data_hard_start;
	struct xdp_rxq_info *rxq;
	struct xdp_txq_info *txq;
	u32 frame_sz;
	u32 flags;
};

struct xdp_rxq_info {
	struct net_device *dev;
	u32 queue_index;
	u32 reg_state;
	struct xdp_mem_info mem;
	unsigned int napi_id;
	u32 frag_size;
} ____cacheline_aligned;
```

逐字段对照（为什么内核侧比 UAPI 多这么多）：
- `data_hard_start`：缓冲区**物理起点**。`data` 可以被 `bpf_xdp_adjust_head` 前后移动（加/剥头部），但释放页、构建 skb 要的是 `data_hard_start`。UAPI 不暴露它，因为程序不该直接碰物理边界。
- `rxq`（`xdp_rxq_info`）：这才是 `xdp_md.ingress_ifindex`/`rx_queue_index` 背后的真身——注释 `rxq->dev->ifindex`/`rxq->queue_index` 与上面 UAPI 的注释一一对应。它 `____cacheline_aligned`，因为每个 RX 队列一个、要避免 false sharing。
- `frame_sz` + `flags`：multi-buffer（5.18+）的产物。`flags & XDP_FLAGS_HAS_FRAGS` 表示这是个**非线性**包（jumbo/分片），此时 `data..data_end` 只是第一段，剩下在 shared info 的 frags 里。对应 helper：

```c
/* 【真实源码 v6.12 路径】include/net/xdp.h（语义摘述，函数体一行） */
static inline bool xdp_buff_has_frags(struct xdp_buff *xdp)
{
	return !!(xdp->flags & XDP_FLAGS_HAS_FRAGS);
}
```

> 坑点（真实踩过）：很多老 XDP 程序假设"包是线性的、`data_end` 就是整包尾"。开了 multi-buffer / jumbo / 某些 GRO-after-XDP 路径后，**`data_end` 只是第一段的尾**。要拿整包长得用 `xdp_get_buff_len()`（它会把 frags 加进来），否则你解析 L4 header 时会莫名其妙读不到。

---

## 21.3 真实源码精读：从驱动 poll 到五个动作的裁决

原版这里给的是我编的 `recycle_rx_page()`/`build_skb_from_xdp()` 占位符。这一版我们读**真实驱动**和**真实内核函数**，逐行看动作是怎么落地的。

### 21.3.1 内核侧 hook 入口：`bpf_prog_run_xdp()`

```c
/* 【真实源码 v6.12 路径】include/net/xdp.h */
static __always_inline u32 bpf_prog_run_xdp(const struct bpf_prog *prog,
					    struct xdp_buff *xdp)
{
	/* Driver XDP hooks are invoked within a single NAPI poll cycle and thus
	 * under local_bh_disable(), which provides the needed RCU protection
	 * for accessing map entries.
	 */
	u32 act = __bpf_prog_run(prog, xdp, BPF_DISPATCHER_FUNC(xdp));

	if (static_branch_unlikely(&bpf_master_redirect_enabled_key)) {
		if (act == XDP_TX && netif_is_bond_slave(xdp->rxq->dev))
			act = xdp_master_redirect(xdp);
	}

	return act;
}
```

逐行注解（这短短十几行藏了三个深点）：
- **注释本身是文档级证据**：*"invoked within a single NAPI poll cycle … under local_bh_disable(), which provides the needed RCU protection for accessing map entries."* —— 这一句话解释了 XDP 程序为什么能**无锁读 map**：它跑在 softirq、`local_bh_disable()` 之下，等价于一个 RCU 读临界区。控制面更新 map 时用 RCU grace period 等老读者退出（21.4 devmap 源码会再现这套）。
- `BPF_DISPATCHER_FUNC(xdp)`：这是 **BPF dispatcher / 静态调用**机制。早年 XDP 调 BPF 程序是**间接调用**（`retpoline` 时代被 Spectre 缓解拖慢得很惨）。dispatcher 把"当前 attach 的程序入口"patch 进一个直接跳转，**消除 indirect call 的分支预测惩罚**。→ 这就是"XDP 在 Spectre 之后还能保持快"的微架构根因。
- `static_branch_unlikely(...)` + `XDP_TX && bond_slave → xdp_master_redirect`：bonding 场景下，从 slave 口 `XDP_TX` 要改写成从 master 口出。用 `static_branch`（静态键/jump label）做到**没开 bond 时这段是个 nop**，零运行期成本。这是"为不常用特性付零代价"的经典内核手法。

### 21.3.2 真实驱动的动作 switch：i40e（不是我编的占位符）

```c
/* 【真实源码 v6.12 路径】drivers/net/ethernet/intel/i40e/i40e_txrx.c
 * i40e_run_xdp - run an XDP program */
static int i40e_run_xdp(struct i40e_ring *rx_ring, struct xdp_buff *xdp,
			struct bpf_prog *xdp_prog)
{
	int err, result = I40E_XDP_PASS;
	struct i40e_ring *xdp_ring;
	u32 act;

	if (!xdp_prog)
		goto xdp_out;

	prefetchw(xdp->data_hard_start); /* xdp_frame write */

	act = bpf_prog_run_xdp(xdp_prog, xdp);
	switch (act) {
	case XDP_PASS:
		break;
	case XDP_TX:
		xdp_ring = rx_ring->vsi->xdp_rings[rx_ring->queue_index];
		result = i40e_xmit_xdp_tx_ring(xdp, xdp_ring);
		if (result == I40E_XDP_CONSUMED)
			goto out_failure;
		break;
	case XDP_REDIRECT:
		err = xdp_do_redirect(rx_ring->netdev, xdp, xdp_prog);
		if (err)
			goto out_failure;
		result = I40E_XDP_REDIR;
		break;
	default:
		bpf_warn_invalid_xdp_action(rx_ring->netdev, xdp_prog, act);
		fallthrough;
	case XDP_ABORTED:
out_failure:
		trace_xdp_exception(rx_ring->netdev, xdp_prog, act);
		fallthrough; /* handle aborts by dropping packet */
	case XDP_DROP:
		result = I40E_XDP_CONSUMED;
		break;
	}
xdp_out:
	return result;
}
```

逐行注解（这才是"XDP 快"的工程实证）：
- `prefetchw(xdp->data_hard_start); /* xdp_frame write */`：在跑 BPF 程序**之前**就预取 `data_hard_start` 并**预取为写**（prefetch-for-**w**rite）。因为不论 `XDP_TX` 还是 `XDP_REDIRECT`，都要把 `xdp_buff` 头部就地改写成 `xdp_frame`（在 `data_hard_start` 那段空间里）。提前把该 cacheline 拉进来、标记为 Exclusive/Modified，省掉真正写时的 RFO（Read-For-Ownership）延迟。→ **这就是 21.1.1 说的"在数据最热时做事"的真实代码体现。**
- `XDP_TX → xdp_rings[rx_ring->queue_index]`：注意它发到的是**专门的 XDP TX 环**，而且**按 RX 队列号取对应的 XDP TX 队列**。这是无锁的关键：RX 队列 i 永远配 XDP-TX 队列 i，同一核处理、不跨核、不抢锁。（dev.c 里 generic 路径的注释明确点了对比："In-driver-XDP use dedicated TX queues, so they do not have this starvation issue."）
- `XDP_REDIRECT → xdp_do_redirect()`：注意驱动**不直接发包**，只是调通用的 `xdp_do_redirect()` 记录"要去哪"，真正的 flush 在 poll 末尾批量做（见 21.3.3 + 21.4 devmap）。返回 `I40E_XDP_REDIR` 让外层知道"这个包交出去了，别回收"。
- `default → bpf_warn_invalid_xdp_action → fallthrough → XDP_ABORTED → trace_xdp_exception → fallthrough → XDP_DROP`：**逐字印证了 21.2.3 commit 信息**——野返回码不会 panic，而是 warn 一下、打 `trace_xdp_exception` tracepoint、当 DROP 处理。这套 `fallthrough` 链非常优雅：ABORTED 比 DROP 多打一个 tracepoint，其余一致。

> 对比原版：我之前用的 `recycle_rx_page()`/`build_skb_from_xdp()`/`driver_xmit_xdp_frame()` 全是**占位名**。真实世界里 DROP 不是"调一个回收函数"，而是简单地 `result = I40E_XDP_CONSUMED` 让调用者走页回收（i40e 用 page reuse/page_pool），`XDP_TX` 是 `i40e_xmit_xdp_tx_ring()`，`XDP_PASS` 才在外层 `i40e_construct_skb()` 建 skb。**别把占位符当真符号背。**

### 21.3.3 `XDP_REDIRECT` 的分派：`xdp_do_redirect()` 怎么知道送哪

`XDP_REDIRECT` 是五个动作里唯一"需要查表"的——程序先调 `bpf_redirect_map(&map, key, flags)` 把目标记在 per-CPU 的 `bpf_redirect_info` 里，驱动再调 `xdp_do_redirect()` 真正分派：

```c
/* 【真实源码 v6.12 路径】net/core/filter.c */
int xdp_do_redirect(struct net_device *dev, struct xdp_buff *xdp,
		    struct bpf_prog *xdp_prog)
{
	struct bpf_redirect_info *ri = bpf_net_ctx_get_ri();
	enum bpf_map_type map_type = ri->map_type;

	if (map_type == BPF_MAP_TYPE_XSKMAP)
		return __xdp_do_redirect_xsk(ri, dev, xdp, xdp_prog);

	return __xdp_do_redirect_frame(ri, dev, xdp_convert_buff_to_frame(xdp),
				       xdp_prog);
}
EXPORT_SYMBOL_GPL(xdp_do_redirect);
```

逐行注解：
- `bpf_net_ctx_get_ri()`：取 **per-CPU** 的 redirect info。为什么 per-CPU？因为 XDP 跑在 softirq、同一核串行、不会重入，per-CPU 就免了锁——又一处"用 NAPI 的串行性换掉锁"。
- **两条岔路**：目标是 `XSKMAP`（AF_XDP socket）走 `__xdp_do_redirect_xsk`（21.4 细讲，包直接进 xsk）；否则 `xdp_convert_buff_to_frame(xdp)` 先把 `xdp_buff` **转成 `xdp_frame`**——因为要离开当前 NAPI 上下文（去别的 CPU/设备），轻量栈上的 `xdp_buff` 活不过这次 poll，必须固化成 `xdp_frame`（它带 `len`/`headroom`/`mem`，能脱离原 NAPI 存活）。

再看非 XSK 的那条，它把 devmap/cpumap/裸 ifindex 三种目标分流：

```c
/* 【真实源码 v6.12 路径】net/core/filter.c —— __xdp_do_redirect_frame 节选 */
	switch (map_type) {
	case BPF_MAP_TYPE_DEVMAP:
		fallthrough;
	case BPF_MAP_TYPE_DEVMAP_HASH:
		if (unlikely(flags & BPF_F_BROADCAST)) {
			...
			err = dev_map_enqueue_multi(xdpf, dev, map,
						    flags & BPF_F_EXCLUDE_INGRESS);
		} else {
			err = dev_map_enqueue(fwd, xdpf, dev);
		}
		break;
	case BPF_MAP_TYPE_CPUMAP:
		err = cpu_map_enqueue(fwd, xdpf, dev);
		break;
	case BPF_MAP_TYPE_UNSPEC:
		if (map_id == INT_MAX) {       /* bpf_redirect(ifindex) 裸重定向 */
			fwd = dev_get_by_index_rcu(dev_net(dev), ri->tgt_index);
			...
			err = dev_xdp_enqueue(fwd, xdpf, dev);
			break;
		}
		fallthrough;
	default:
		err = -EBADRQC;
	}
```

注意全是 `*_enqueue` 而不是 `*_xmit`——**redirect 只入队，不立即发**。原因见下节 devmap 的设计：批量化（bulk）+ 末尾统一 flush，把每包一次的 doorbell/MMIO 写摊薄成每批一次。

---

## 21.4 REDIRECT 的两个后端：devmap / cpumap 的无锁设计（真实源码 + 设计意图）

这是原版完全没讲、但恰恰最能体现 staff 级权衡的地方。两个 map 的源码**头部 DOC 注释本身就是设计文档**。

### 21.4.1 cpumap：把"早期过滤"和"协议栈"拆到不同 CPU

```c
/* 【真实源码 v6.12 路径】kernel/bpf/cpumap.c */
// Copyright (c) 2017 Jesper Dangaard Brouer, Red Hat Inc.
/**
 * DOC: cpu map
 * The 'cpumap' is primarily used as a backend map for XDP BPF helper
 * call bpf_redirect_map() and XDP_REDIRECT action, like 'devmap'.
 *
 * Unlike devmap which redirects XDP frames out to another NIC device,
 * this map type redirects raw XDP frames to another CPU.  The remote
 * CPU will do SKB-allocation and call the normal network stack.
 */
/*
 * This is a scalability and isolation mechanism, that allow
 * separating the early driver network XDP layer, from the rest of the
 * netstack, and assigning dedicated CPUs for this stage.  This
 * basically allows for 10G wirespeed pre-filtering via bpf.
 */
```

读这段设计意图（作者 Brouer，Red Hat）能得到一个非常实战的架构 pattern：

- **场景**：你有 100GbE，单核跑 XDP 解析/过滤已经吃满，但**还想让通过的包走完整内核栈**（要 TCP、要 conntrack）。怎么办？
- **cpumap 的答案**：在收包核上只做最便宜的 XDP 判决，把"PASS 的包"`XDP_REDIRECT` 到 cpumap → 落到**另一组专属 CPU**，由它们做 `alloc_skb` + 走协议栈。
- **根因**：把"驱动 XDP 层"和"netstack 层"**物理隔离到不同核**，各自的工作集（XDP 的 map vs 栈的路由/conntrack/socket 表）不再互相踩 cache，且把 skb 分配这个重活从收包核卸走。注释原话：*"separating the early driver network XDP layer, from the rest of the netstack … assigning dedicated CPUs … 10G wirespeed pre-filtering via bpf."*

> 这就是"软件版的 RPS，但更聪明"：RPS 是把**skb** 散到别的核（skb 已经建了），cpumap 是把**还没建 skb 的 xdp_frame** 散过去、让目标核去建 skb。少做一次、且分配发生在目标核（NUMA/cache 更优）。

### 21.4.2 devmap：为"无锁转发"付出的 RCU + per-cpu flush 设计

```c
/* 【真实源码 v6.12 路径】kernel/bpf/devmap.c */
/* Copyright (c) 2017 Covalent IO, Inc. http://covalent.io */
/* Devmaps primary use is as a backend map for XDP BPF helper call
 * bpf_redirect_map(). Because XDP is mostly concerned with performance we
 * spent some effort to ensure the datapath with redirect maps does not use
 * any locking. This is a quick note on the details.
 *
 * We have three possible paths to get into the devmap control plane bpf
 * syscalls, bpf programs, and driver side xmit/flush operations. ...
 * To ensure updates and deletes appear atomic from the datapath side xchg()
 * is used to modify the netdev_map array. Then because the datapath does a
 * lookup into the netdev_map array (read-only) from an RCU critical section
 * we use call_rcu() to wait for an rcu grace period before free'ing the old
 * data structures. ...
 * the datapath does a "flush" operation that pushes any pending packets in
 * the driver outside the RCU critical section. Each bpf_dtab_netdev tracks
 * these pending operations using a per-cpu flush list. The bpf_dtab_netdev
 * object will not be destroyed until this list is empty ...
 */
```

（Copyright Covalent IO 即后来的 **Isovalent/Cilium 团队**，John Fastabend 等的工作。）

这段把"无锁数据面"的全部代价讲透了，逐点拆：
- **目标**：*"the datapath … does not use any locking"*。转发热路径一把锁都不能有，否则多队列多核会在锁上排队（排队论：到达率逼近服务率时排队延迟爆炸）。
- **更新可见性用 `xchg()`**：控制面改 `netdev_map[i]` 用原子 `xchg`，让数据面要么看到旧的、要么看到新的，不会看到半个——**内存序**层面的原子替换。
- **释放用 `call_rcu()` 等 grace period**：旧的 `bpf_dtab_netdev` 不能立刻 free，因为可能还有数据面读者在 RCU 临界区里持有它。等一个 grace period（所有 CPU 都过了一次 quiescent state）再 free。→ 这正是 21.3.1 那句"NAPI under local_bh_disable 提供 RCU 保护"的另一半：读者侧靠 bh-disable 当 RCU read-side，写者侧靠 call_rcu 等读者退出。
- **per-cpu flush list 防 use-after-free**：redirect 只入队、末尾才 flush，所以"对象正在被某 CPU 的 pending flush 引用"这个窗口必须管理。每个 `bpf_dtab_netdev` 用 per-cpu flush 列表记 pending，**列表非空就不销毁**。

> 失败模式 / 真坑：如果没有这套 per-cpu flush 跟踪，"map 删表 → free 设备对象"可能和"某核还在 flush 队列里的包要发到这个设备"撞上 → **use-after-free / crash**。这就是为什么 devmap 的销毁是"两步 + 等 flush 清空"的笨重流程——不是过度设计，是无锁转发的必然代价。

---

## 21.5 XDP 实战重述：DDoS 过滤与 Katran（修正一个常见错误结论）

### 21.5.1 DDoS 过滤的数据结构与边界检查的"为什么"

数据结构设计（与原版一致，但把"为什么这样选 map 类型"补上）：

```
BPF_MAP_TYPE_LRU_HASH (blocklist)   key: __u32 src_ip   value: __u64 packet_count
  ↑ 选 LRU_HASH：攻击 IP 是动态、海量、会过期的；LRU 自动淘汰老条目，避免表爆。
    普通 HASH 满了会插入失败（新攻击 IP 进不来），LRU 满了淘汰最久没命中的。

BPF_MAP_TYPE_PERCPU_ARRAY (stats)   key: enum xdp_action   value: __u64 count
  ↑ 选 PERCPU：每包都要 ++ 计数。共享 ARRAY 会让多核在同一 cacheline 上原子竞争
    （false sharing + LOCK 前缀 → 几十核时计数本身成瓶颈）。PERCPU 每核独立写，
    用户态读时再求和。这是"per-packet 统计永远用 PERCPU"的铁律。
```

程序逻辑（示意，重点在边界检查）：

```c
/* 【示意代码，非逐字】DDoS filter 核心 */
SEC("xdp")
int xdp_ddos_filter(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_DROP;   /* 必须：否则 verifier 拒绝 */
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_DROP;    /* 必须 */

    __u32 src = ip->saddr;
    __u64 *cnt = bpf_map_lookup_elem(&blocklist, &src);
    if (cnt) { __sync_fetch_and_add(cnt, 1); return XDP_DROP; }
    return XDP_PASS;
}
```

**边界检查为什么是强制的（底层根因）**：`ctx->data_end` 是包尾。XDP 程序处理的包长度是**运行期**才知道的，可能只有 14 字节（残包/攻击构造包）。verifier 在加载时做 **abstract interpretation + range tracking**（追踪每个寄存器的 `[umin_value, umax_value]` 值域和 `tnum` 已知位掩码，见 v6.12 `kernel/bpf/verifier.c` 里大量 `tnum`/`umin_value`/`umax_value`），它必须**证明** `eth+1 <= data_end` 在所有路径成立，才允许你解引用 `eth->h_proto`。少写一行 `if`，verifier 就认为"这次访问可能越界"——而越界读的是 DMA 缓冲区，可能读到别的包甚至内核内存，所以**只能拒绝加载**。

### 21.5.2 Katran：必须修正的"它不用 conntrack"错误

原版（和很多博客）有一个广泛流传的错误结论："Katran 不用 conntrack，全靠一致性哈希"。**这是不准确的**。Katran 官方 README 明确写了：

> *"Modified Maglev hashing for connections"* + *"Fixed size (size is configurable on start) connection tracking table w/ LRU strategy for eviction of old entries."*

【史料：facebookincubator/katran README】

**真相（更有"思考感"的版本）**：Katran 同时用两者，各补对方的短板——

- **Maglev 一致性哈希**：无状态、可由控制面热更新、数据面只读查表。它保证"backend 集合不变时，同一 5-tuple 永远算到同一 backend"。
- **但一致性哈希有个致命窗口**：当 backend **增/删**（扩缩容、健康检查摘除）时，Maglev 会让**一部分**已有 flow 重新映射到不同 backend → 这些 flow 的 TCP 连接会断。
- **LRU connection tracking table 就是来盖这个窗口的**：它记住"这个 flow 上次去了哪个 backend"，即使 Maglev 表变了，**已建立的 flow 仍按 conntrack 表里的旧 backend 走**，新 flow 才用新的 Maglev 结果。LRU 负责淘汰老 flow 控制内存。

所以正确表述是：**Katran 用 Maglev 做"无状态选择 + 最小扰动"，用 LRU conntrack 做"已有连接的粘滞性"**。这比"不用 conntrack"深一层，也正好是 staff 面试想听的 trade-off：一致性哈希降低了 conntrack 的**依赖程度和表压力**（不是消除），LRU 给了**有界内存下的连接亲和**。

其余两个 Katran 设计点（这两个原版对了，保留并加深）：

- **IPIP 封装 + 故意变化的 outer source IP**：README 原话——*"katran crafts a special one, in such a way, that different flows will have different outer (ipip) source IP, but packets in same flow will always have the same."* 目的：让下游（backend 侧）的 **RSS** 能按 outer IP 把不同 flow 散到不同 RX 队列/核。这是"为了配合下游硬件分流，故意设计封装头"的典型权衡——它把"负载均衡的均匀性"从 L7 一路传导到了 backend 的网卡硬件。
- **`XDP_TX` 从原口发出**：流量来自同一网口，IPIP 封装后 `XDP_TX` 原路出去最简单，省掉 devmap 跨设备那套。多 NIC 入向转发才需要 `XDP_REDIRECT + devmap`。

### 21.5.3 Cilium：XDP + tc BPF 的分工（不是"全 XDP"）

新手误区仍然要破：Cilium **不是**"完全用 XDP"。它是 **XDP（L3/L4 早丢，如 DDoS、NodePort 前置过滤）+ tc BPF（ingress/egress，能拿到 skb、做更完整的 L4/L7 策略、conntrack、NAT）** 的组合。根因：XDP 拿的是 `xdp_buff`（无 skb，context 受限、最快），tc BPF 拿的是 `__sk_buff`（有完整 skb 元数据，能做深检查但慢一截）。**把活按"需要多少 skb 信息"分给两层**，是 Cilium 数据面的核心架构决策。

---

## 21.6 AF_XDP：用户态零拷贝快路径（源码级 + 史料）

### 21.6.1 设计动机与"零拷贝是后来才有的"史实

XDP 解决了"内核内"高速处理，但有些应用逻辑（自定义协议、有状态处理、用户态协议栈）**必须在用户态**拿到包。选项：标准 socket（太慢，见 21.1）、DPDK（绕过内核、失明）、或 **AF_XDP**。

史实纠正（很多人以为 AF_XDP 一出生就零拷贝）：LWN 750845 *"Accelerating networking with AF_XDP"*（Corbet, 2018-04-09）记录，**Björn Töpel / Magnus Karlsson（Intel）**的初版 patch set **明确还没实现零拷贝**——原文：*"This whole data structure is designed to enable zero-copy movement … though the current patches do not yet implement that."* 零拷贝是随后逐步加上的（需要驱动配合把 NIC DMA 直接指向 UMEM）。【史料：LWN 750845】

### 21.6.2 架构：UMEM + 四个 ring（用 v6.12 文档原话钉死语义）

```
NIC RX ring → NAPI poll → XDP 程序判断 → XDP_REDIRECT 进 xskmap
   ↓
AF_XDP socket (xsk)  ── 共享内存 UMEM ──  用户态进程 (poll on xsk fd)
```

四个 ring 的生产者/消费者关系，最容易记反。直接引 v6.12 文档：

| Ring | 归属 | 谁生产 → 谁消费 | 语义（文档原话） |
|---|---|---|---|
| **FILL** | UMEM（跨 socket 共享） | 用户 → 内核 | *"transfer ownership of UMEM frames from user-space to kernel-space"*（用户交出空 frame 给内核去 DMA 收包） |
| **RX** | per-socket | 内核 → 用户 | 收到包后，描述符从 FILL 移到 RX，用户来消费 |
| **TX** | per-socket | 用户 → 内核 | *"used to send frames"*，用户填"哪个 UMEM 地址要发" |
| **COMPLETION** | UMEM | 内核 → 用户 | *"transfer ownership of UMEM frames from kernel-space to user-space"*（发完了，frame 还给用户复用） |

文档把规则总结成一句：*"the RX and FILL rings are used for the RX path and the TX and COMPLETION rings are used for the TX path."* 记忆法：**收包用 FILL→RX，发包用 TX→COMPLETION**，UMEM 帧的所有权在用户/内核之间通过这两对 ring 来回交接。

【史料：Documentation/networking/af_xdp.rst @ v6.12】

### 21.6.3 源码精读：包是怎么"零拷贝"进 socket 的，以及 queue_id 为什么必须匹配

```c
/* 【真实源码 v6.12 路径】net/xdp/xsk.c */
static int xsk_rcv_check(struct xdp_sock *xs, struct xdp_buff *xdp, u32 len)
{
	if (!xsk_is_bound(xs))
		return -ENXIO;

	if (xs->dev != xdp->rxq->dev || xs->queue_id != xdp->rxq->queue_index)
		return -EINVAL;                 /* ★ dev 和 queue 必须都对上 */

	if (len > xsk_pool_get_rx_frame_size(xs->pool) && !xs->sg) {
		xs->rx_dropped++;
		return -ENOSPC;                 /* 包比 UMEM frame 还大、又没开 sg → 丢 */
	}

	sk_mark_napi_id_once_xdp(&xs->sk, xdp);
	return 0;
}

static int xsk_rcv(struct xdp_sock *xs, struct xdp_buff *xdp)
{
	u32 len = xdp_get_buff_len(xdp);
	int err;

	err = xsk_rcv_check(xs, xdp, len);
	if (err)
		return err;

	if (xdp->rxq->mem.type == MEM_TYPE_XSK_BUFF_POOL) {
		len = xdp->data_end - xdp->data;
		return xsk_rcv_zc(xs, xdp, len);   /* ★ 零拷贝路径：内存本来就在 UMEM 池里 */
	}

	err = __xsk_rcv(xs, xdp, len);             /* 否则 copy 模式：拷进 UMEM */
	if (!err)
		xdp_return_buff(xdp);
	return err;
}
```

逐行注解（两个高价值结论都在这）：
- `if (xs->dev != xdp->rxq->dev || xs->queue_id != xdp->rxq->queue_index) return -EINVAL;` —— **这就是"AF_XDP socket 必须绑定到产生该包的那个 queue_id"的源码铁证**。文档也写了：*"you specify a specific queue id to bind to and it is only the traffic towards that queue you are going to get on your socket."* 根因：UMEM/ring 是 per-(dev,queue) 资源，跨队列就没有对应的 FILL/RX ring 可投递。**实战部署里，N 个 RX 队列就要 N 个 xsk，一一对应，才能吃到 RSS 的 CPU locality。**
- `mem.type == MEM_TYPE_XSK_BUFF_POOL → xsk_rcv_zc`：**零拷贝的本质是"内存归属"而非"少做一次 memcpy 那么简单"**。当 NIC 的 DMA 缓冲区**本来就分配自 UMEM 池**（`MEM_TYPE_XSK_BUFF_POOL`），包数据落地时就已经在用户可见的 UMEM 里了，`xsk_rcv_zc` 只需把描述符塞进 RX ring（`xskq_prod_*`），**一个字节都不用搬**。否则（普通驱动内存）走 `__xsk_rcv` 把数据 copy 进 UMEM——这就是 XDP_COPY vs XDP_ZEROCOPY 的真实分界。

### 21.6.4 AF_XDP vs DPDK：带真实数据跑一遍

| 维度 | AF_XDP | DPDK |
|---|---|---|
| 内核介入 | 有（NAPI poll + XDP hook） | 无（PMD 独占 NIC，busy poll） |
| NIC 独占 | **否**，可同时 `XDP_PASS` 部分包给内核栈 | 是，NIC 对内核不可见 |
| 可观测性 | 完整（tcpdump/perf/eBPF/ftrace 全在） | 无（包对内核工具不可见） |
| 单核 RX 峰值 | 数 Mpps 级（受 NAPI+XDP+ring 开销，REDIRECT 路径 paper 实测单核 ~8.5 Mpps，多核扩展见下） | 单核 ~43 Mpps（paper 实测 DPDK RX-only Core 1） |
| 多核峰值 | XDP_REDIRECT 6 队列 ~30 Mpps（paper） | RX-only ~115 Mpps 量级（paper，4 核近饱和） |
| 驱动维护 | 随内核升级，无单独负担 | PMD 独立维护，与内核版本绑定 |
| 适用 | 需要混合流量 / 保留可观测性 / 无 DPDK 专项团队 | 极限吞吐 / 完全控制 / NFV / 高频交易 |

【性能数据史料：xdp-paper 实验数据（mlx5）；DPDK RX-only "Cores 1-5 从 ~43.5 Mpps 起"、XDP_REDIRECT 6 队列 ~30 Mpps 见 21.7.1 表】

**跑一个具体场景**：你做一个 IDS（Suricata 风格），需要在用户态深度解析报文，但**只对可疑流量**，正常 web 流量希望仍走内核 TCP 给本机服务。

- 选 DPDK：整张卡被 PMD 拿走，本机的内核服务（ssh、监控 agent、正常 TCP）**全断**——除非再插一张网卡。可观测性也没了，线上抓包只能靠 DPDK 自带工具。
- 选 AF_XDP：XDP 程序里 `if (suspicious) bpf_redirect_map(&xskmap, q, 0); else return XDP_PASS;`。可疑流量零拷贝进用户态 IDS，正常流量继续走内核栈，**同一张卡、同时存在两条路**，tcpdump 照样能抓。
- **不适用边界**：如果你要的是"每一个包都要极致吞吐、且不需要任何内核服务"（纯转发盒子、电信 NFV），AF_XDP 的 NAPI+XDP 那点固定开销和 ~30 Mpps 的 REDIRECT 天花板就是劣势，该上 DPDK。

> 面试加分点（架构本质）：AF_XDP 与 DPDK 的根本差异是**"NIC 共享 vs 独占"**，由此派生出可观测性、混合流量、维护成本的全部不同。说出这一句，比背"AF_XDP 慢一点"有价值得多。

---

## 21.7 方案对比的"思考感"：XDP / DPDK / 内核栈，沿数据走一遍

### 21.7.1 先看一组真实的横向 benchmark（CoNEXT'18 / xdp-paper 实验数据，mlx5）

| RXQs | XDP_DROP (pps) | XDP_REDIRECT (pps) |
|---|---|---|
| 1 | 25,928,270 | 8,461,375 |
| 2 | 51,349,744 | 16,241,020 |
| 3 | 76,578,241 | 18,639,798 |
| 4 | 82,782,450 | 21,417,122 |
| 5 | 82,294,143 | 25,373,567 |
| 6 | 80,444,303 | 29,970,889 |

DPDK RX-only（同 paper）：单核 ~43.5 Mpps 起，多核到 ~100+ Mpps（约 4 核饱和 ~115 Mpps）。
【史料：xdp-paper benchmarks（github.com/xdp-project/xdp-paper）/ CoNEXT'18】

**这张表带读者沿推理走，能读出三件原版没讲的事**：

1. **DROP 近线性、然后撞墙**：1→4 队列从 25.9→82.8 Mpps 几乎线性（每核 ~25 Mpps），但 4→6 队列基本不涨（82→80）。撞的不是 CPU，是 **PCIe / NIC 内部带宽 / 描述符 ring 吞吐**这些共享资源。→ 实战含义：XDP 扩到一定核数后加核无用，要查的是硬件侧。
2. **REDIRECT 比 DROP 贵 ~3 倍**：单核 25.9（DROP）vs 8.5（REDIRECT）。差距就是 21.3.3 讲的"`xdp_buff → xdp_frame` 转换 + 入队 + flush + 可能跨核"。→ 实战含义：能 DROP/TX 解决的别 REDIRECT；AF_XDP 的天花板由 REDIRECT 那条曲线决定（~30 Mpps@6Q），不是 DROP 那条。
3. **XDP 单核 ~25 Mpps < DPDK 单核 ~43 Mpps**：XDP **确实更慢**，因为它没绕过 NAPI、还在内核框架里。但它换回了整个内核生态。→ 这正是工业界"宁可慢 40% 也选 XDP"的量化依据。

### 21.7.2 场景化对比：给"100GbE L4 LB"做技术选型

需求：单机 100GbE，做 L4 负载均衡（类似 Katran），峰值 ~50–100 Mpps，要能热更新 backend、要 metrics、健康检查走内核。

- **路线 A：内核 LVS/IPVS**。优点：成熟、全可观测。但它在 IP 层之后、skb 已建、还有 conntrack——按 21.1.2 的阶梯，单核也就 1–2 Mpps 量级，100Gbps 要堆几十核且 conntrack 锁会成瓶颈。**淘汰**。
- **路线 B：DPDK 自研 LB**。能到 ~100 Mpps，但健康检查/控制面/可观测性全要自己在用户态重写，且 NIC 独占——健康探测想走内核 TCP 都不行。团队得养 DPDK 专家。**只有确实卡在 >50 Mpps/核且团队够强时才考虑**。
- **路线 C：XDP（Katran 模式）**。XDP_TX + Maglev + LRU conntrack（21.5.2）。单核 ~8.5 Mpps（REDIRECT）/ 若用 XDP_TX 更高，多核线性堆到撞 PCIe；backend 表存 map 控制面热更新；PERCPU_ARRAY 出 metrics；健康检查包 `XDP_PASS` 给内核 TCP。**这就是 Meta 的真实选择**——不是因为 XDP 最快，而是它在"够快 + 可热更新 + 可观测 + 不独占卡"四个维度同时及格。

**推理结论**：纯吞吐 DPDK 赢，但把"运维成本 + 可观测性 + 混合流量"算进总账，XDP 在 LB/DDoS/容器网络这类**需要和内核共生**的场景里是更优解。**不适用边界**：纯转发盒子、无任何内核服务依赖、且追求极限 pps 的场景，DPDK 仍是对的。

---

## 21.8 eBPF Verifier：安全数据面的基石（网络视角）

（完整机制见第 23 章，这里只讲"为什么 XDP 程序敢让内核直接跑"。）

### 21.8.1 verifier 给 XDP 的四条硬约束

1. **无越界内存访问**：解引用任何 `data..data_end` 之间的指针前，必须先有比较把范围"夹住"。verifier 用 **range tracking**（每个寄存器维护 `umin_value/umax_value` 区间 + `tnum` 已知位）+ abstract interpretation 模拟所有路径，证明访问在界内才放行（v6.12 `kernel/bpf/verifier.c` 里 `tnum`/`umin_value`/`umax_value` 出现数百处）。
2. **无无限循环**：必须证明所有循环有界。早期完全禁循环；5.3+ 提供 `bpf_loop`/有界循环，但上界必须可证。
3. **helper 白名单**：XDP context 只能调允许的 helper（`bpf_map_lookup_elem`、`bpf_xdp_adjust_head`、`bpf_redirect_map` …），**不能 sleep、不能分配内存、不能调任意内核函数**。
4. **返回值约束**：理想是只允许 5 个 `xdp_action`，但如 21.2.3 commit 所述，返回值是运行期算的、**静态证不死**，所以内核退而求其次：野返回码运行时 warn + 当 ABORTED（21.3.2 驱动代码的 `default` 分支）。

### 21.8.2 一个真实 helper 如何在 verifier 约束下安全改包：`bpf_xdp_adjust_head`

XDP 程序要加/剥包头（如 Katran 加 IPIP 外层），靠 `bpf_xdp_adjust_head` 移动 `data` 指针。看它的真实实现，体会"内核怎么不信任你给的 offset"：

```c
/* 【真实源码 v6.12 路径】net/core/filter.c */
BPF_CALL_2(bpf_xdp_adjust_head, struct xdp_buff *, xdp, int, offset)
{
	void *xdp_frame_end = xdp->data_hard_start + sizeof(struct xdp_frame);
	unsigned long metalen = xdp_get_metalen(xdp);
	void *data_start = xdp_frame_end + metalen;
	void *data = xdp->data + offset;

	if (unlikely(data < data_start ||
		     data > xdp->data_end - ETH_HLEN))
		return -EINVAL;          /* ★ 越过头部预留区 或 短到放不下以太头 → 拒绝 */

	if (metalen)
		memmove(xdp->data_meta + offset,
			xdp->data_meta, metalen);    /* metadata 跟着搬 */
	xdp->data_meta += offset;
	xdp->data = data;

	return 0;
}
```

逐行注解：
- `xdp_frame_end = data_hard_start + sizeof(struct xdp_frame)`：缓冲区最前面要给 `xdp_frame` 留位（21.3.2 那个 `prefetchw(data_hard_start)` 写的就是它），所以 `data` 最多只能往前推到这之后。`data_start` 是合法下界。
- `if (data < data_start || data > data_end - ETH_HLEN) return -EINVAL;`：**内核完全不信任 BPF 传进来的 `offset`**。哪怕 verifier 让程序加载了，运行期 helper 还要再夹一次：往前不能压到 `xdp_frame` 区，往后不能短到连一个以太头都放不下。这是"verifier 静态 + helper 运行期"**双重防御**。
- `memmove(data_meta...)`：调整 head 时 metadata 区要跟着平移，保持 `data_meta` 和 `data` 的相对关系。
- 全程**没有重新分配内存**——只是在 `data_hard_start..data_end` 这块已 DMA 的缓冲区里挪指针。这就是为什么加头是 O(1) 且零分配（前提是 headroom 够，否则 `-EINVAL`，得在 attach 时保证 `XDP_PACKET_HEADROOM` 预留）。

> 坑点：Katran 加 IPIP（+20 字节外层 IP 头）就靠 `bpf_xdp_adjust_head(ctx, -(int)sizeof(struct iphdr))`。如果驱动/配置没给够 headroom（`XDP_PACKET_HEADROOM`，通常 256B），这个调用直接 `-EINVAL`，封装失败、包被丢。生产里"XDP LB 偶发丢包"的一个真因就是 headroom 不足（某些虚拟设备/隧道叠加场景）。

### 21.8.3 verifier 的 sound 性质（面试深入点）

verifier 做的是 **sound（可靠）但不 complete（不完备）** 的分析：**绝不放过非法程序（无 false negative）**，但**会误拒部分合法程序（有 false positive）**。所以你会遇到"我这逻辑明明对，verifier 就是不让过"——通常是它的 range tracking 没能跟上你的等价变换（比如用了它推不出范围的算术）。解法：要么改写让范围显式（多加一个 `if (x < N)`），要么看 `bpftool prog load` 的逐指令报错定位是哪条指令、在哪个寄存器状态下违规。

---

## 21.9 多队列与零拷贝基础设施：RSS/RPS/RFS 和拷贝的消除

（与 XDP 协同的基础设施，保留原版骨架，补"和 XDP 的真实关系 + 根因"。）

### 21.9.1 RSS / RPS / RFS 三层

- **RSS（硬件分流）**：NIC 按 `(sip,dip,sport,dport)` 的 Toeplitz hash 把 flow 散到不同 RX 队列，每队列绑一个 CPU。**最低延迟**，但需 NIC 支持。同一 flow 永远同一核 → 无锁、cache 热。
- **RPS（软件 RSS）**：NIC 不够队列时，内核在 `netif_receive_skb` 早期按 flow hash 把 skb 经 IPI 投到别的核的 backlog。代价：**一次 IPI（跨核中断，~µs 级）+ skb 已建**。
- **RFS（应用感知的 RPS）**：追踪应用最后一次 `recvmsg` 在哪个核，把该 flow 后续包引到同核，减少"包在 A 核处理、应用在 B 核读"的跨核 cache 失效。

配置命令（真实可用）：
```bash
ethtool -l eth0                 # 查 RSS 队列数（combined）
ethtool -L eth0 combined 16     # 设 RSS 队列数（需驱动支持）
ethtool -x eth0                 # 查 RSS hash key 和 indirection table
echo f > /sys/class/net/eth0/queues/rx-0/rps_cpus          # RPS：允许 CPU0-3
echo 32768 > /proc/sys/net/core/rps_sock_flow_entries      # RFS 全局表
echo 2048  > /sys/class/net/eth0/queues/rx-0/rps_flow_cnt  # RFS 每队列表
```

### 21.9.2 XDP 与多队列的真实耦合点（不是泛泛"协同"）

- **XDP 程序是 per-netdev attach、但在每个 RX 队列的 NAPI poll 里各跑一份**（21.3.2 的 `i40e_run_xdp` 就在每队列 poll 中调用）。所以 XDP 天然吃 RSS 的并行——队列越多、跑 XDP 的核越多。
- **AF_XDP 必须 queue_id 一一对应**（21.6.3 `xsk_rcv_check` 的源码铁证）。要利用 RSS 的 CPU locality，就得：RSS 开 N 队列 → 每队列一个 xsk → 每 xsk 绑到对应 queue 所在的核。错配则要么 `-EINVAL`、要么丢掉 RSS 局部性。
- **cpumap 是"XDP 自带的软件 RPS"**（21.4.1）：但它散的是 `xdp_frame`（无 skb），比 RPS 散 skb 更省。延迟敏感优先 RSS（硬件），队列不够再 cpumap/RPS。

```bash
# attach XDP（native，强制：不支持就报错而非退到 generic）
ip link set dev eth0 xdpdrv obj xdp_prog.o sec xdp
```

### 21.9.3 各层零拷贝技术与隐藏成本

| 技术 | 消除哪次拷贝 | 机制 / 隐藏成本 |
|---|---|---|
| GRO | 减少 skb 数量（合并） | 协议栈层聚合，非真零拷贝 |
| splice/sendfile | kernel→user（发送） | page cache 直接映射进 socket buffer |
| MSG_ZEROCOPY | send 路径 | 用户页 pin 住 DMA 直发；**但有记账成本** |
| AF_XDP UMEM | DMA→用户态（接收） | NIC DMA 直写 UMEM，无 skb（21.6.3） |
| io_uring + fixed buffers | 用户态 I/O | 见第 17 章 |

> 坑点（MSG_ZEROCOPY 的反直觉）：它不是"免费的零拷贝"。内核要 **pin 住用户页**，且发送完成后要通过 error queue 的 completion（`recvmsg(MSG_ERRQUEUE)` 拿 `cmsg`）告诉你"这段缓冲区可复用了"。这套"页 pin + 异步记账"在**小包**场景反而比直接 copy 更慢——payload 越大越划算，经验拐点大致在 **~10KB 量级**（依 CPU/NIC/是否同核做 notification 而变，别当硬阈值）。出处：Willem de Bruijn、Eric Dumazet（Google）netdev 2.1（2017）*"sendmsg copy avoidance with MSG_ZEROCOPY"*。

---

## 21.10 何时该绕过内核：决策框架与真实代价

```
你的瓶颈真的在内核网络栈吗？  →  先 perf top / flamegraph / `ethtool -S` 实测，别猜
   ├─ 否 → 回去优化应用层（多半瓶颈在你自己代码 / 锁 / 序列化）
   └─ 是
        需要内核生态（eBPF trace、conntrack、混合流量、本机服务）？
           ├─ 是 → XDP / AF_XDP（现代默认）
           └─ 否 + 追极限 pps + 有专项团队 → DPDK
                                          + 存储/HPC/RDMA 语义 → RDMA(RoCE/IB)
```

**DPDK 的维护现实**：PMD 绑定特定内核版本范围，升级内核可能让 PMD 失效；strace/perf/tcpdump/eBPF 对 DPDK 内的包**全部失明**；需要专职 DPDK 工程师。**RDMA**：适合 NVMe-oF、HPC、大规模 ML 训练（GPUDirect），对通用 Web 服务意义不大。

**结论**：2020 年代的高性能网络默认是 **XDP + AF_XDP**；DPDK 只在"确实卡在 >40 Mpps/核、且不需要内核共生、团队扛得动"时上。判据已被 21.7.1 的 benchmark 量化——不是信仰，是数字。

---

## 21.11 生产调试与调优（命令 + 每条的"为什么"）

### 21.11.1 工具箱
```bash
ip -d link show dev eth0 | grep -o 'xdp[a-z]*'  # ★ 最该核：xdp(native) 还是 xdpgeneric
bpftool prog list                                # 所有加载的 eBPF 程序
bpftool prog show pinned /sys/fs/bpf/xdp_prog    # 看某 prog 的 run_time_ns/run_cnt
bpftool map dump id <map_id>                     # dump map（PERCPU 会自动按 CPU 列出）
ethtool -S eth0 | grep -E 'rx_queue|drop|miss|error'  # 网卡侧丢包计数
ethtool -x eth0                                  # RSS indirection table（是否均衡）
watch -n1 'cat /proc/net/softnet_stat'           # 各 CPU 软中断处理/丢弃/time_squeeze
mpstat -P ALL 1                                  # 各核 %soft，看中断是否挤在少数核
```

### 21.11.2 常见排障场景（带根因）

**场景 1：XDP attach 了但没提速**——头号嫌疑是 **fallback 到 generic XDP**。
- 根因：驱动不支持 native XDP 时，内核默默退到 generic（`do_xdp_generic`，21.3 那段 dev.c 源码），此时 **skb 已分配**，性能等同普通内核路径。你以为上了 native，其实在 generic。
- 排查：`ip -d link show` 看 `xdp` 还是 `xdpgeneric`。强制 native：`ip link set dev eth0 xdpdrv ...`（不支持就报错，不静默退化）。
- 其次查 NUMA：XDP 程序/RX 队列/UMEM 是否在同一 NUMA node，跨 node 访问 DMA 缓冲会吃 QPI/UPI 延迟。

**场景 2：AF_XDP 吞吐低**。
- `bpf_redirect_map` 是否真的命中 xskmap（而不是误 `XDP_PASS` 给了内核栈）。
- queue_id 是否匹配（21.6.3 的 `-EINVAL` 会让包根本进不来）。
- FILL ring 是否经常空（UMEM frame 不够，内核没空 buffer 可 DMA → 丢）。
- 是否开了 `SO_BUSY_POLL` / `XDP_USE_NEED_WAKEUP`，避免 poll() 的唤醒开销。

**场景 3：低 CPU% 却丢包**（经典 10G+ 症状）。详见 21.12 系统题——核心是"忙的只是绑了中断的那几个核，平均下来显得闲"。

### 21.11.3 生产配置清单（每条注明意图）
```bash
service irqbalance stop                          # 停掉自动均衡，手动把 NIC 中断绑核
# （逐 IRQ 写 smp_affinity，把每个队列的中断钉到对应核，配合 RSS 的 CPU locality）

ip -d link show dev eth0 | grep -o 'xdp[a-z]*'   # 确认 native（最该核的一步）
# 注：XDP hw-offload 能力不在 `ethtool -k` 里（没有 xdp-offload feature），
#     它通过 netdev netlink 暴露；只有 Netronome 等极少数 NIC 真支持 offload。

ethtool -G eth0 rx 4096                           # 调大 RX ring，吸收突发、减少 rx_missed
ethtool -L eth0 combined 16                       # 队列数 ≈ 收包核数，摊开软中断
numactl --cpunodebind=0 --membind=0 ./your_app    # 应用与 NIC 同 NUMA node
ethtool -K eth0 gro off                           # 某些 XDP+multi-buffer 场景需关 GRO（视情况）
```

---

## 21.12 系统题预演（量级估算 + 排障，放正文是因为它把 21.1/21.9/21.11 串起来）

**题**：单机 25GbE，压测时 `ethtool -S` 的 `rx_missed_errors` 在涨、丢包 5%，但**整机 CPU 才 30%**。怎么排查？

**量级感知（高把握的线速算术）**：
- 25GbE，64B 包（+preamble+IFG=84B 线速）≈ **37.2 Mpps**。
- 取一个**乐观**假设"纯 RX softirq 每包 ~500 ns"，单核上限 ≈ 2 Mpps。⚠ 这 500 ns 只是收包下半部的精简成本，**不是全栈**——真要 recvmsg 到用户态是 5–20 µs/包，那条路单核远到不了 2 Mpps。这里用 2 Mpps 只为估"需要几个 RX 队列分摊软中断"。
- 37.2 / 2 ≈ 19 核。若 RSS 只开 4 队列：4 核 × 2 Mpps = 8 Mpps < 37.2 Mpps → **瓶颈在队列数/中断只落在少数核**。这正好解释"整机才 30%"：忙的是那几个绑了中断的核（接近 100%），其余核闲着，平均下来就低。

**排查步骤**：
```bash
ethtool -S eth0 | grep -E 'miss|drop|error'
# rx_missed_errors↑ → NIC ring 满、来不及 DMA；rx_no_buffer_count↑ → NAPI 来不及消费

ethtool -l eth0                                   # 当前 combined 队列数
grep eth0 /proc/interrupts                        # 几个中断、各落哪个 CPU 列（看哪几列在涨）

mpstat -P ALL 1                                   # %soft 是否集中在个别 CPU（头号嫌疑）
for irq in $(grep eth0 /proc/interrupts | awk -F: '{print $1}'); do
  printf 'irq %s -> ' "$irq"; cat /proc/irq/$irq/smp_affinity_list
done                                              # 中断亲和是否摊开

watch cat /proc/net/softnet_stat
# 列序（高把握，十六进制）：第1列=已处理包数；第2列=backlog 丢弃；第3列=time_squeeze。
# 第3列(time_squeeze)↑ → NAPI 一次 budget 没收完就被迫让出（处理跟不上，典型 10G+ 症状）；
# 第2列↑ → backlog 溢出（更像 RPS/backlog 配小，与 time_squeeze 是两个病因，别混）。
```

**解决**：
```bash
ethtool -L eth0 combined 16     # 扩队列，把软中断摊到更多核
ethtool -G eth0 rx 4096         # 扩 ring buffer，吸收突发
# 或部署 XDP_DROP 把无效流量在驱动层直接丢，减少 NAPI 消费压力（治本：少做无用功）
```

---

## 21.13 未来演进

- **XDP multi-buffer 普及**（5.18+，21.2.5 的 `flags`/frags）：让 XDP 能处理 jumbo frame、分片、GRO-after-XDP，扩大适用面，但要求程序用 `xdp_get_buff_len`、别再假设线性包。
- **CO-RE / BTF-aware XDP**：Compile Once Run Everywhere，eBPF 程序不绑特定内核头，靠 BTF 重定位字段，跨内核版本可移植（见第 23 章）。
- **BPF token / unprivileged BPF**：降低跑 eBPF 所需权限，让非 root 容器也能用——security 与 usability 的持续博弈，「待核」当前默认策略。
- **Rust for eBPF（Aya）**：aya-rs 让 Rust 写 eBPF 程序，类型安全比 C 强；生态快速成熟。
- **更大图景（eBPF 数据面分层）**：
  ```
  XDP (L2/L3, 驱动层, 最快, 无 skb)
    ↕  tc BPF (L3/L4, 协议栈入口/出口, 有 skb)
    ↕  sk_msg / sk_skb (socket 层, service mesh L7 重定向)
    ↕  sockops (TCP 参数动态调整)
  ```
  Cilium Service Mesh 正用完整 eBPF 数据面替换 sidecar proxy（Envoy），把 L7 策略下沉内核——eBPF 在网络领域影响最大的趋势之一。

---

## 章末心法

> **XDP 的本质不是"绕过内核"，而是"把判决时间点前移到数据还最热、还没被做成 skb 的那一刻"。**
> 配套的一切（per-CPU redirect info、devmap 的 RCU+flush、cpumap 的 CPU 隔离、AF_XDP 的 UMEM 所有权交接、verifier 的 range tracking）都是为了在"前移判决"的同时，**不丢掉内核的安全模型与可观测性**。这个"在最热的点做事、但守住安全边界"的哲学，适用于所有高性能系统设计。

---

## 章末五件套

---

### 一、高频 Staff 级面试题

**Q1：XDP 比 iptables DROP 快的根本原因？给出量化。**
- 路径长度：iptables 在 netfilter hook（IP 层之后），**skb 已分配、路由已查、可能已过 conntrack**；XDP 在 NAPI poll 内、**skb 根本没分配**。
- 数据结构：iptables 线性遍历规则链 O(N)，XDP 可用 hash map O(1)。
- **量化（Cloudflare 实测，单核）**：iptables INPUT 608 kpps、raw PREROUTING 1.69 Mpps、tc 1.8 Mpps，而 XDP_DROP **10.1 Mpps**；从 tc 到 XDP 的 ~5.6× 差距就是"一个 skb 的命"。
- 陷阱纠正：很多人答"XDP 绕过了内核"——**错**。XDP 在内核里执行，只是在更早的位置（且经 verifier 证明安全）。

**Q2：XDP_REDIRECT 与 XDP_TX 的区别，何时用哪个？为什么 REDIRECT 更贵？**
- `XDP_TX`：从**同一** NIC 原路发出，适合回射（DNS 应答、反射型 LB、Katran 的 IPIP 回发）。驱动里就是发到 `xdp_rings[queue_index]` 这个专属 TX 环（21.3.2），无锁。
- `XDP_REDIRECT`：去**另一个** NIC（devmap）/ **另一个 CPU**（cpumap）/ **AF_XDP socket**（xskmap）。
- **为什么贵**：REDIRECT 要 `xdp_convert_buff_to_frame()` 把 `xdp_buff` 固化成 `xdp_frame`（21.3.3），再入队、末尾 flush、可能跨核——paper 实测单核 8.5 vs DROP 25.9 Mpps，约 3×。

**Q3：eBPF verifier 如何保证 XDP 不 panic 内核？sound 还是 complete？**
- 静态 abstract interpretation + **range tracking**（`umin/umax_value` + `tnum`，v6.12 verifier.c），证明每次指针解引用在 `data..data_end` 内；循环必须可证有界；helper 白名单；返回值约束（野返回码运行时 warn+ABORTED，因为返回值静态证不死，见 commit 6a773a15a1e8）。
- **sound 但不 complete**：绝不放过非法程序（无 false negative），但会误拒部分合法程序（有 false positive）——这就是"逻辑明明对却加载不过"的来源。

**Q4：设计 1 亿 pps 的 L4 LB，技术栈怎么选？**
- 算硬件：100M pps × 84B ≈ 67 Gbps 线速 → 100GbE 多队列。
- 数据面：XDP native，Maglev 一致性哈希查表 + **LRU conntrack 做连接粘滞**（不是"不用 conntrack"，见下题），`XDP_TX` 或 `XDP_REDIRECT+devmap` 发包。
- map：backends/hash ring 存 ARRAY（控制面热更新），计数 PERCPU_ARRAY（免 per-packet 锁）。
- 不选 DPDK：健康检查要走内核栈、要 eBPF metrics、不想独占卡。

**Q5（细节杀）：Katran 到底用不用 connection tracking？**
- **用**。官方 README：Maglev 一致性哈希 **+** "Fixed size … connection tracking table w/ LRU"。
- 分工：Maglev 做无状态选择 + backend 变动时最小扰动；**LRU conntrack 盖住 Maglev 重映射会断已有连接的窗口**，给有界内存下的连接亲和。说成"不用 conntrack"是流行的错误。
- 加分：IPIP 故意给不同 flow 不同 outer src IP，让 backend 侧 RSS 能散流。

**Q6：AF_XDP 与 DPDK 的架构本质差异？何时用 AF_XDP？**
- **NIC 共享 vs 独占**：AF_XDP 通过 XDP 选择性 redirect，同卡可同时走内核栈；DPDK 独占、内核失明。源码铁证：`xsk_rcv_check` 要求 socket 的 `queue_id` 匹配产生包的队列（21.6.3）。
- 可观测性：AF_XDP 全在内核框架内（tcpdump/perf/eBPF 可见）；DPDK 不可见。
- 性能：XDP/AF_XDP REDIRECT 单核 ~8.5 Mpps（paper），DPDK RX-only 单核 ~43 Mpps。
- 用 AF_XDP：需要混合流量 / 保留可观测性 / 无 DPDK 专项团队。

**Q7：XDP 何时 fallback 到 generic？危险在哪？怎么强制 native？**
- 驱动不支持 native 时内核静默退到 generic（`do_xdp_generic`/`netif_receive_generic_xdp`，21.3 dev.c 真实源码），此时 skb 已建、无性能优势。
- 危险："上了 XDP 没效果"的困惑，本质是你在跑 generic。
- 排查：`ip -d link show` 看 `xdp` vs `xdpgeneric`。强制：`ip link set dev eth0 xdpdrv ...`（不支持就报错，不退化）。

**Q8（深点）：为什么 XDP 程序能无锁读 map？devmap 删表为什么那么"啰嗦"？**
- 读侧：XDP 跑在 NAPI、`local_bh_disable()` 下，等价 RCU 读临界区（`bpf_prog_run_xdp` 注释原话）。
- 写侧：devmap 更新用 `xchg()` 原子替换、释放用 `call_rcu()` 等 grace period，且用 **per-cpu flush list** 跟踪 pending 转发，列表非空不销毁（devmap.c DOC）。
- 啰嗦的原因：无锁转发下，"删设备" 和 "某核还在 flush 队列里要发到该设备" 会撞 use-after-free，必须靠 RCU + flush 跟踪盖住窗口。

---

### 二、实战项目（P2 增量）

**P2：写一个 XDP/eBPF 程序统计并丢包**

目标：attach 到本地网口，(1) 按源 IP 统计包数（hash map），(2) 对 blocklist 中 IP/端口 DROP，(3) 用户态读统计。

验收：
```bash
make
sudo ip link set dev eth0 xdpdrv obj xdp_filter.o sec xdp   # 强制 native
sudo bpftool map dump name pkt_count
sudo ./xdp_ctl block 10.0.0.2
# 从另一台机器发流 → 该 IP 100% loss，其余正常
sudo ip link set dev eth0 xdp off
```

工具链：libbpf（官方 C）或 aya（Rust 类型安全）；clang+llvm（编 eBPF）；bpftool（调试）。

暗坑：
1. loopback(`lo`) 多数发行版**不支持 native XDP**，测丢包用真实网卡或 veth pair。
2. 编 eBPF 必须 `clang`（gcc 的 bpf target 支持有限/不通用）。
3. 用 `vmlinux.h`（BTF 生成）而非装内核头，走 CO-RE 更可移植。
4. 字节序统一 `bpf_htons`/`bpf_ntohs`（`bpf/bpf_endian.h`），别用 glibc `htons`。

---

### 三、设计题

**题**：流媒体 CDN 节点（单机 100GbE，面向 C 端，要同时处理 HTTP/2 与防 DDoS）的高性能数据面。

考察：是否分离"DDoS 过滤(XDP)"与"正常流量(内核栈)"；XDP 如何在 L3/L4 识别攻击（IP blocklist + SYN rate limit / SYN cookie）；为什么 HTTP/2 不走 XDP bypass（需 TLS+HTTP 解析，用户态逻辑复杂）；RSS 如何按 flow 均匀分核；容量规划（100GbE ≈ 150M pps@64B，几核能扛 XDP）。

参考要点：分层——XDP 做 L3/L4 过滤（blocklist 存 `LRU_HASH`，控制面实时更新，数据面无感）+ `XDP_PASS` 正常流量；内核做 TCP/TLS；用户态做 HTTP/2。不用 DPDK：HTTP/2 逻辑太重、TLS 要自己实现，不值。可叠 **cpumap** 把"过滤核"和"协议栈核"分开（21.4.1），收包核只做最便宜的判决。

---

### 四、系统题

见正文 **21.12**（已把量级估算 + 排障完整展开：25GbE 丢包但 CPU 30% 的"中断挤在少数核"诊断）。

---

### 五、代码题（可运行 XDP 程序骨架）

```c
/* xdp_pkt_counter.bpf.c —— 【示意代码，非逐字】统计每源 IP 入向包数，XDP_PASS 所有包
 * 编译：clang -O2 -g -target bpf -c xdp_pkt_counter.bpf.c -o xdp_pkt_counter.bpf.o
 * 现代 libbpf + CO-RE 写法：只 include vmlinux.h + bpf_helpers.h/bpf_endian.h，
 * 不要混 <linux/*> 与 <arpa/inet.h>（会与 vmlinux.h 的类型打架——新手第一坑）。 */
#include "vmlinux.h"            /* bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h */
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>     /* bpf_htons / bpf_ntohs */

#ifndef ETH_P_IP
#define ETH_P_IP 0x0800
#endif

/* per-CPU 避免每包原子操作（见 21.5.1 的根因） */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __type(key,   __u32);   /* src IPv4 */
    __type(value, __u64);   /* packet count */
    __uint(max_entries, 65536);
} pkt_count SEC(".maps");

SEC("xdp")
int xdp_count_pkts(struct xdp_md *ctx)
{
    void *data_end = (void *)(long)ctx->data_end;
    void *data     = (void *)(long)ctx->data;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)            /* verifier 强制的边界检查 */
        return XDP_DROP;

    if (eth->h_proto != bpf_htons(ETH_P_IP))     /* 统一用 bpf_htons */
        return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)             /* 第二次边界检查 */
        return XDP_DROP;

    __u32 src_ip = ip->saddr;
    __u64 *count = bpf_map_lookup_elem(&pkt_count, &src_ip);
    if (count) {
        (*count)++;                              /* PERCPU：本核独立计数，无需原子 */
    } else {
        __u64 init = 1;
        bpf_map_update_elem(&pkt_count, &src_ip, &init, BPF_ANY);
    }
    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
```

验证：
```bash
clang -O2 -g -target bpf -c xdp_pkt_counter.bpf.c -o xdp_pkt_counter.bpf.o
sudo ip link set dev eth0 xdpdrv obj xdp_pkt_counter.bpf.o sec xdp   # 强制 native
ping 8.8.8.8 -c 10 &
sudo bpftool map list && sudo bpftool map dump id <ID>
sudo ip link set dev eth0 xdp off
```

暗坑：
1. `PERCPU_HASH` 用户态手动读，`bpf_map_lookup_elem` 拿回的是长度 `libbpf_num_possible_cpus()` 的数组，要逐 CPU 求和——是 **possible** 不是 online，否则越界。（bpftool dump 会自动汇总。）
2. 字节序只用 `bpf_htons`/`bpf_ntohs`（`bpf/bpf_endian.h`）：它对编译期常量走 `__builtin_constant_p` 在编译期算好，对运行期值 emit byte-swap，比 glibc `htons` 更可移植。
3. `PERCPU_HASH` 的 `max_entries` 是**每核**表大小，实际内存 ≈ `max_entries × possible_cpus × (key+value 对齐)`，别设太大。
4. generic XDP（如 lo）能跑但性能与 native 完全不同——测性能务必 native。

---

*（本章完）*

---

## 附录：本章真实取材出处（均经实际 WebFetch / curl 取得）

**真实源码（Linux v6.12，raw.githubusercontent.com/torvalds/linux/v6.12/）**
- `include/net/xdp.h` —— `xdp_buff`/`xdp_rxq_info`/`bpf_prog_run_xdp`/`xdp_buff_has_frags`
- `include/uapi/linux/bpf.h` —— `enum xdp_action`/`struct xdp_md`
- `net/core/filter.c` —— `xdp_do_redirect`/`__xdp_do_redirect_frame`/`bpf_xdp_adjust_head`
- `net/core/dev.c` —— `netif_receive_generic_xdp`/`do_xdp_generic`/`generic_xdp_tx`
- `drivers/net/ethernet/intel/i40e/i40e_txrx.c` —— `i40e_run_xdp`（真实驱动动作 switch）
- `net/xdp/xsk.c` —— `xsk_rcv_check`/`xsk_rcv`/`__xsk_map_redirect`（AF_XDP 收包 + queue 校验）
- `kernel/bpf/cpumap.c` —— DOC 注释（Brouer/Red Hat 的设计意图）
- `kernel/bpf/devmap.c` —— DOC 注释（Covalent/Cilium 的无锁 RCU+flush 设计）
- `kernel/bpf/verifier.c` —— 确认 `tnum`/`umin_value`/`umax_value` range tracking
- `Documentation/networking/af_xdp.rst` —— 四个 ring 语义、XDP_ZEROCOPY、queue 绑定

**设计考古 / 史料**
- LWN 682538 *"Early packet drop — and more — with BPF"*（Corbet, 2016-04-06）—— 初版被打回、Alexei 的 ABI 引言、6.5 vs 20 Mpps
- LWN 750845 *"Accelerating networking with AF_XDP"*（Corbet, 2018-04-09）—— Töpel/Karlsson(Intel)、初版无零拷贝、ring 设计
- commit `6a773a15a1e8` *"bpf: add XDP prog type for early driver filter"*（Brenden Blanco，committer davem）—— 奠基提交信息逐字
- Cloudflare *"How to drop 10 million packets per second"* —— 丢包阶梯 175kpps→10.1Mpps
- facebookincubator/katran README —— Maglev + LRU conntrack + IPIP 变 outer-IP（修正"不用 conntrack"）
- xdp-paper benchmarks（github.com/xdp-project/xdp-paper）/ CoNEXT'18 —— XDP_DROP/REDIRECT 多队列扩展表、DPDK RX-only 对比
- iovisor.org/technology/xdp、blogs.igalia.com/dpino XDP 综述 —— 背景佐证

> 凡标「待核」处，是我未能取到一手原文逐字核实的细节（多为某 feature 的精确内核版本号、unprivileged BPF 当前默认策略），请勿当事实背诵。
