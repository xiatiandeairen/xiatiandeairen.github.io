---
title: "调度器：CFS 到 EEVDF（深化版）"
slug: "9-05"
collection: "tech-library"
group: "linux系统"
order: 9005
summary: "调度器是内核最政治敏感的子系统——它决定谁先运行，因此决定谁有资源。理解调度器就是理解内核对\"公平\"这个词的工程定义。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-11T13:31:22.000Z"
updatedAt: "2026-06-11T13:31:22.000Z"
---
> 调度器是内核最政治敏感的子系统——它决定谁先运行，因此决定谁有资源。理解调度器就是理解内核对"公平"这个词的工程定义。
>
> 本章是"扎根版"：每个机制都尽量带**真实源码**（标注版本与文件路径）、**git/社区史料**（带 LWN / commit 出处）、**方案对比**（同一场景在不同调度器下跑一遍）。凡逐字源码标 **【真实源码 vX 路径】**；为讲清算法而简化的标 **【示意代码，非逐字】**；fetch 不到又没把握的符号标「待核」。

---

## TL;DR（本章最硬结论）

1. **调度的核心矛盾是三角不可能**：fairness / latency / throughput 不可同时最优。Linux 调度器二十年迭代史，本质是在这个三角上反复换平衡点——O(1) 押注 throughput + 启发式 latency，CFS 押注严格 fairness，EEVDF 在 fairness 之上补回**可计算的 latency 上界**。
2. **CFS 的 vruntime 是一个天才的近似**：用"虚拟运行时间"把 priority 与已用 CPU 时间折算到同一坐标轴，红黑树取最左节点 O(log n) 选下一任务。核心折算函数就是 `__calc_delta()`（见 5.3.1 真实源码）。它跑了 16 年（2.6.23 → 6.5）。
3. **EEVDF（Earliest Eligible Virtual Deadline First）在 6.6 合入**（Peter Zijlstra，commit `147f3ef`，2023）。它**不是新调度类**——`fair_sched_class` 名字都没变——而是把"取最小 vruntime"的贪心，换成"在 eligible（欠服务）的任务里取最早 virtual deadline"。引入第二个参数 `slice`（请求时间片）：slice 越小，deadline 越近，latency-sensitive 任务越被优先。
4. **EEVDF 删掉了 CFS 时代一批旋钮**：`sched_latency_ns` / `sched_min_granularity_ns` / `sched_wakeup_granularity_ns` 在 6.6+ 上**不存在了**，取而代之的是 per-task slice（经 `sched_setattr` 的 `sched_runtime` 表达，范围 100us–100ms）+ 全局 `sysctl_sched_base_slice`。**旧调优脚本在新内核上会 `No such file`。** 这是当前最高频的生产踩坑。
5. **cgroup CPU 带宽控制是生产 BUG 重灾区**：`cpu.cfs_quota_us` 设太小导致大规模 throttle，表现为 CPU util 不高但 p99 爆炸——大厂 oncall 最常见的调度相关故障。EEVDF 改的是 pick 算法，**没动 bandwidth/throttle 这套**，所以这个坑跨版本依旧。

**前置依赖章**：第 4 章（进程与线程，`task_struct`、进程状态机）；第 6 章（上下文切换）；第 22 章（cgroup，带宽控制全景）；第 8 章（NUMA，跨 node 负载均衡）。

**读前提醒**：本章源码以 **Linux v6.12** 为基准（一个稳定的 EEVDF 已落地的版本）。CFS 时代的源码符号（`sched_slice()`、`check_preempt_wakeup()` 旧逻辑等）在 6.6 后大量改写，遇到"老博客的代码跟现在对不上"是正常的——本章会明确标注谁是 CFS 时代、谁是 EEVDF 时代。

---

## 5.1 背景：调度器要解决什么

### 5.1.1 问题的本质

CPU 是共享资源，N 个可运行任务竞争 M 个核。调度器做三件事：

1. **选哪个任务运行**（`pick_next_task`）
2. **何时切换**（抢占 vs 协作让出）
3. **在哪个核上运行**（负载均衡，SMP/NUMA）

对应三个互相拉扯的目标：

```
┌────────────────┬──────────────────────────────────────────────┐
│ 公平性 fairness │ 每个任务按权重比例获得 CPU，不饿死              │
│ 响应延迟 latency│ 交互/实时任务尽快得到 CPU，尤其 wake-up 时      │
│ 吞吐量 throughput│ CPU 尽量不空转，批量任务跑完快                 │
└────────────────┴──────────────────────────────────────────────┘
```

**不可能三角**：提高公平性 → 频繁切换 → 上下文切换开销增大，吞吐下降；降低延迟 → 给某类任务特权 → 其他任务不公平；提升吞吐 → 减少切换 → 响应变慢。这不是实现质量问题，是物理约束。

**这一节是全章的"为什么"**：后面每次换调度器，你都要能回答"它在三角上往哪个角移了、牺牲了哪个角"。

### 5.1.2 后端工程师的锚点

把调度器类比为 **连接池 + 任务队列**：连接池（CPU 核）是稀缺资源；任务队列（run queue）决定谁先被服务；公平性 ≈ 加权轮询（WRR）；响应延迟 ≈ VIP 队列；吞吐 ≈ 批量聚合减少 per-request overhead。调度器只是这个逻辑的 OS 级实现，只不过"时间片"是微秒级，"切换"要付保存/恢复寄存器 + TLB/cache 冷启动的代价（见第 6 章）。

进一步的精确类比：**CFS = Weighted Fair Queueing (WFQ) 在 CPU 时间域的实现**；**EEVDF = WF2Q 思想 + EDF**。这不是比喻，是同一套排队论。Stoica & Abdel-Wahab 1995 年那篇 *"Earliest Eligible Virtual Deadline First"* 技术报告就是排队/网络调度领域的论文，Zijlstra 把它搬进了 CPU 调度器。

---

## 5.2 设计考古：调度器演进的真实历史与争论

> 这是本章的灵魂小节。调度器的每一次换代都不是"工程师想优化一下"，而是**真实的技术辩论 + 社区政治**的产物。下面每个锚点尽量带出处。

### 5.2.1 时间线总览

| 调度器 | 内核版本 / 年份 | 作者 | 三角上的取舍 | 出处锚点 |
|---|---|---|---|---|
| O(n) scheduler | ≤ 2.4 | 早期内核 | 简单，遍历选 goodness | — |
| **O(1) scheduler** | 2.6（2003）→ 2.6.22 | Ingo Molnar | throughput + 启发式 latency | Wikipedia "O(1) scheduler" |
| RSDL / SD | 2007 未合入 | **Con Kolivas** | 严格 fairness，无启发式 | LWN 230501 / RSDL 公告 2007-03-04 |
| **CFS** | 2.6.23（2007-10）| Ingo Molnar | 严格加权 fairness | LWN 230501，原始 patch 2007-04-13 |
| **EEVDF** | 6.6（2023）| Peter Zijlstra | fairness + latency 上界 | LWN 925371，commit `147f3ef` |
| EEVDF 收尾 | 6.6 → 6.12+ | Peter Zijlstra | lag decay / deferred dequeue | LWN 969062（2024-04） |

> 史料核实说明：上表的 commit `147f3ef`、LWN 文章编号、各版本年份均经实际 WebFetch 核实（详见本章末"史料链接清单"）。git.kernel.org 的 commit 页被反爬挡住，commit 信息改从 GitHub 镜像取得。

### 5.2.2 O(1) 为何不够：启发式的原罪

**O(1) scheduler（Ingo Molnar，2.6，2003）** 的设计很漂亮：

```
两个优先级位图数组 active / expired，每优先级一个 list_head：
  bitmap find_first_set_bit → O(1) 选最高优先级
  时间片耗完 → 任务移入 expired
  active 空 → swap(active, expired) 指针交换，O(1)
  140 个优先级槽：0..99 实时，100..139 对应 nice -20..+19
```

per-CPU run queue + O(1) pick，解决了 O(n) 的遍历和全局锁问题。**但它的死穴是"区分交互任务靠启发式"**：

> （LWN/Wikipedia 核实）"all the calculations to determine the interactivity of tasks are complex and subject to potential miscalculations" —— O(1) 通过分析任务的 sleep pattern 猜"这是不是个等用户输入的交互任务"，猜对了奖励它更高的动态优先级，猜错了就误判。

启发式的根本问题：**它是一个可被博弈、可被对抗的近似**。一个故意频繁短睡的 CPU hog 能骗过 interactivity estimator 拿到交互奖励；而一个真正交互但 sleep pattern 不典型的任务被误判成 batch。内核社区为此贴了海量 band-aid 补丁，越改越复杂。

**记住这个教训**——它是理解 CFS 的钥匙：**CFS 的全部出发点就是"不要猜任务类型"。**

### 5.2.3 Con Kolivas 与那场著名的社区拒绝

这是 Linux 社区史上最著名的技术 + 人情冲突之一，值得工程师认真读。

**Con Kolivas** 是一位麻醉科医生 + 业余内核黑客，长期专注桌面交互性。他先后做了 **Staircase scheduler**，再到 **RSDL（Rotating Staircase Deadline）**，后改名 **SD（Staircase Deadline）**。

RSDL 公告（LWN，**2007-03-04**，标题 *"RSDL completely fair starvation free interactive cpu scheduler"*，核实原文）的核心主张：

> "a starvation free, strict fairness O(1) scalable design with interactivity as good as the above restrictions can provide" —— 不用 interactivity estimator、不测 sleep/run，只用 "simple fixed accounting"，并且 "it is possible to accurately predict the maximum latency that a task may experience"。

注意 Kolivas 的 thesis 跟后来 CFS 几乎一致：**抛弃启发式、用硬性 accounting 实现公平**。可以说他在概念上是对的，而且更早。

**冲突点**：RSDL/SD 在社区获得不少桌面用户好评，但 Linus 和核心 maintainer 对"把它设为默认"犹豫。社区当时还在辩论一个更大的问题——**要不要支持 pluggable scheduler（可插拔调度器框架）**，让用户选调度器。Kolivas 阵营希望走这条路。Linus 明确反对 pluggable scheduler（理由：会让内核为了"可选"而无法对单一调度器做深度集成优化，且把调优负担推给用户）。

**转折**：2007-04-13，**Ingo Molnar 直接甩出 CFS**（LWN 230501，标题 *"[patch] Modular Scheduler Core and Completely Fair Scheduler [CFS]"*，核实原文），距 RSDL 公告仅 5 周。Molnar 在 patch 里**公开致谢 Kolivas**：

> （核实引用）Molnar credited Kolivas for "proving via RSDL/SD that 'fair scheduling' is possible"，并直言目标是 "make CFS's interactivity quality exceed that of RSDL/SD"。

CFS 用红黑树 + 纳秒级 vruntime，Molnar 称设计 "quite radical"，并强调它消除了 O(1) 和 RSDL/SD 都有的 "array switch artifacts"，且因为纳秒精度而 "resistant to attacks that exploited vanilla scheduler heuristics"。

**结局**：CFS 在 **2.6.23（2007-10）** 合入成为默认。RSDL/SD 落选。Con Kolivas 不久后**愤而退出内核开发**，并在多年后（2009-08）以 out-of-tree 的 **BFS（Brain Fuck Scheduler）** 重新出现——他明确表示 BFS **无意进 mainline**，只服务桌面/小核心数场景。

> 这段历史的工程启示（也是社区反复总结的）：**"概念正确 + 更早"不等于"会被采纳"**。CFS 胜出靠的是：①作者是核心 maintainer，集成阻力小；②实现风格契合 Linus 的"单一深度集成调度器、拒绝 pluggable"的方向；③红黑树时间线模型在数学上比楼梯（staircase）更干净可证明。技术辩论从来不只是技术。

### 5.2.4 CFS 跑了 16 年后，为什么还要换 EEVDF

CFS 的 thesis 是"**最终公平**（eventual fairness）"：保证长期每个任务按权重拿到份额。但它**对延迟没有任何显式契约**。Corbet 在 LWN（925371，*"An EEVDF CPU Scheduler for Linux"*，2023-03-09，核实原文）一句话点破：

> "[CFS] does not give processes a way to express their latency requirements."

CFS 时代为了模拟"低延迟"，只能靠两个全局旋钮硬掰：`sched_latency_ns`（一轮调度周期）和 `sched_wakeup_granularity_ns`（唤醒抢占阈值），再加上 `wakeup preemption` 的一堆启发式——**又回到了启发式**，只是换了个地方。这就是 CFS 的结构性缺陷（详见 5.3.5）。

EEVDF 的 thesis 升级为"**最终公平 + 每任务可计算的延迟上界**"。它让任务能通过 `slice`（请求时间片）表达"我需要多急"，并用 EDF 在虚拟时间域给出 deadline 保证。这才是真正"内生地"解决 latency，而不是再加一层启发式旋钮。

---

## 5.3 CFS：完全公平调度器（源码精读）

### 5.3.1 核心抽象：vruntime 与它的真实折算函数

**定义**：`vruntime`（virtual runtime）衡量一个任务"在理想处理器上应该运行了多久"。理想处理器：N 个等权任务各得 1/N CPU。vruntime 把"理想份额"折算成虚拟时间。

**折算公式（概念级）**：

```
delta_vruntime = delta_real_time × (NICE_0_LOAD / task_weight)
```

nice 越低 → `task_weight` 越大 → 同样真实时间累积的 vruntime 越少 → 在红黑树里"显得跑得慢" → 更频繁被选中 → 高优先级多跑。这就是 CFS 的全部魔法。

下面是内核里真正干这件事的代码（不是伪代码）：

**【真实源码 v6.12 kernel/sched/fair.c】 `calc_delta_fair`**
```c
static inline u64 calc_delta_fair(u64 delta, struct sched_entity *se)
{
	if (unlikely(se->load.weight != NICE_0_LOAD))
		delta = __calc_delta(delta, NICE_0_LOAD, &se->load);

	return delta;
}
```
逐行注解：
- 入参 `delta` 是这次跑了多少真实纳秒，`se` 是任务的调度实体。
- **快路径优化**：如果任务就是 nice 0（`weight == NICE_0_LOAD`），那 `NICE_0_LOAD / weight == 1`，vruntime 增量就等于真实时间，**直接返回，连乘法都省了**。`unlikely()` 是给编译器的分支预测提示（多数任务确实是 nice 0）。
- 否则才走 `__calc_delta` 做加权折算。

**【真实源码 v6.12 kernel/sched/fair.c】 `__calc_delta`**
```c
static u64 __calc_delta(u64 delta_exec, unsigned long weight, struct load_weight *lw)
{
	u64 fact = scale_load_down(weight);
	u32 fact_hi = (u32)(fact >> 32);
	int shift = WMULT_SHIFT;
	int fs;

	__update_inv_weight(lw);

	if (unlikely(fact_hi)) {
		fs = fls(fact_hi);
		shift -= fs;
		fact >>= fs;
	}

	fact = mul_u32_u32(fact, lw->inv_weight);

	fact_hi = (u32)(fact >> 32);
	if (fact_hi) {
		fs = fls(fact_hi);
		shift -= fs;
		fact >>= fs;
	}

	return mul_u64_u32_shr(delta_exec, fact, shift);
}
```
逐行注解（这段是热路径定点数学，新读者最容易看懵，慢慢拆）：
- 它算的是 `delta_exec * weight / lw->weight`，但**全程不做除法**。除法在 CPU 上慢，调度器每个 tick 都要算，必须优化掉。
- `lw->inv_weight` 是预先算好的"倒数权重"：`inv_weight ≈ 2^WMULT_SHIFT / lw->weight`。`__update_inv_weight(lw)` 负责在 weight 变了时懒计算它。`WMULT_SHIFT` 是 32（定点小数位数）。这就是 5.3.1 早期版本里提到的 `inv_weight` 字段的真身。
- 于是 `delta * weight / lw->weight ≈ delta * weight * inv_weight >> WMULT_SHIFT`，**两次乘 + 一次右移**代替除法。
- 中间那两段 `if (fact_hi)`：处理**溢出**。`fact = weight * inv_weight` 可能超过 32 位，高 32 位非零（`fact_hi`）就说明会溢出。`fls()` = find-last-set，找最高位；把 `fact` 右移、`shift` 同步减小，**牺牲低位精度换不溢出**。这是定点运算里"宁可丢精度也不能 wrap"的经典手法。
- 最后 `mul_u64_u32_shr(delta_exec, fact, shift)` = `(delta_exec * fact) >> shift`，一条专门的 64×32 带移位乘法。

> 面试杀手锏：被问"CFS 怎么把 nice 折算成 CPU 时间"，能从 `calc_delta_fair` 的 fast path 讲到 `__calc_delta` 的 inv_weight + 溢出处理，就把"背概念"和"读过源码"区分开了。

`NICE_0_LOAD` 不是裸常量 1024，而是：

**【真实源码 v6.12 kernel/sched/sched.h】**
```c
#define NICE_0_LOAD		(1L << NICE_0_LOAD_SHIFT)

#ifdef CONFIG_64BIT
# define NICE_0_LOAD_SHIFT	(SCHED_FIXEDPOINT_SHIFT + SCHED_FIXEDPOINT_SHIFT)
#else
# define NICE_0_LOAD_SHIFT	(SCHED_FIXEDPOINT_SHIFT)
#endif
```
注解：`SCHED_FIXEDPOINT_SHIFT` 是 10（待核：该宏定义在别处未逐字取到，但 `1<<10 = 1024` 与下文权重表 nice 0 = 1024 自洽）。**关键**：64 位下 `NICE_0_LOAD_SHIFT = 10 + 10 = 20`，即 64 位内核的负载精度是 `1<<20`，比 32 位多 10 bit 小数——这是为了 PELT（per-entity load tracking）等更精细的负载计算。日常讲"nice 0 权重 1024"指的是 `scale_load_down` 之后的视图。

### 5.3.2 nice → weight：真实的硬编码查表

nice 到 weight **不是线性公式，是一张 40 项硬编码表**（直接算会错，这是经典暗坑）：

**【真实源码 v6.12 kernel/sched/sched.h，声明】**
```c
extern const int		sched_prio_to_weight[40];
extern const u32		sched_prio_to_wmult[40];
```

数组定义在 `kernel/sched/core.c`。完整 40 个值（经核实，每行注释为对应 nice 值）：

**【真实源码 v6.12 kernel/sched/core.c，sched_prio_to_weight[] 内容】**
```
 /* -20 */  88761,  71755,  56483,  46273,  36291,
 /* -15 */  29154,  23254,  18705,  14949,  11916,
 /* -10 */   9548,   7620,   6100,   4904,   3906,
 /*  -5 */   3121,   2501,   1991,   1586,   1277,
 /*   0 */   1024,    820,    655,    526,    423,
 /*   5 */    335,    272,    215,    172,    137,
 /*  10 */    110,     87,     70,     56,     45,
 /*  15 */     36,     29,     23,     18,     15,
```

关键性质：
- **相邻 nice 的 weight 比 ≈ 1.25**。`1.25^10 ≈ 9.31`，所以"nice 差 10 ≈ 约 10 倍 CPU"这个常用近似来自这里。
- 真正的乘子是**约 1.25/格**，不是"每格 10%"。"nice 每格约 10%"是流传很广但**不严谨**的说法——它只在两任务对跑时近似成立。
- **CPU 时间份额 = 本任务 weight / 所有任务 weight 之和**，是相对量。所以 "nice 降 1 多拿 25%" 只在极简两任务场景下近似。系统里任务一多，单个 nice 的边际影响被稀释。

> 为什么是 1.25 而不是 2？设计目标：让 nice 差 1 对应"用户能感知但不剧烈"的差异，差 10（满量程的一半）对应约 10 倍——一个让桌面用户调起来手感平滑的几何级数。这是个**人因工程**选择，不是数学必然。

`sched_prio_to_wmult[]` 是对应的预算 inv_weight（`2^32 / weight` 量级），就是 5.3.1 里 `__calc_delta` 用的那个倒数权重的查表来源，避免运行时再算除法。

### 5.3.3 数据结构：红黑树（CFS 时代）

CFS 把所有可运行任务放进一棵以 `vruntime` 为 key 的红黑树（`rb_root_cached`，带最左节点缓存指针）：

```
               [vruntime=100]
              /              \
        [50]                [200]
        /  \                /   \
     [30]  [80]          [150]  [350]
   ↑ 最左 = vruntime 最小 = 下一个该跑的
```

- **选下一任务**：取最左节点。有缓存指针 `rb_leftmost`，接近 O(1)。
- **插入（wakeup/fork）**：O(log n)。
- **per-CPU**：每核独立红黑树，跨核走负载均衡（5.7）。

> ⚠ 版本提醒：到 EEVDF（6.6+），**还是这棵红黑树，但被 augment（增广）了**——节点上额外维护子树 `min_vruntime`，pick 不再是"取最左"，而是在树上剪枝找"eligible 且 deadline 最早"。详见 5.4.3 的 `pick_eevdf` 真实源码。**不要被老博客的"两棵树"说法误导，从来都是一棵。**

### 5.3.4 min_vruntime 与新任务防插队

💡 **坑点**：新 fork 或刚唤醒的任务若把 vruntime 初始化为 0，会立刻成最左节点抢 CPU——"新客户不排队直接插队"，破坏公平。

CFS 时代解法：新任务 `vruntime = max(task->vruntime, cfs_rq->min_vruntime - 一个补偿)`，即"至少不比当前最落后的任务领先太多"。`min_vruntime` 是当前 rq 单调推进的基准线，是防插队的锚。

> EEVDF 时代这件事由 `place_entity()` 接管，并改用 **lag** 来做："任务睡了多久、欠/超了多少服务"被记进 `se->vlag`，唤醒时按 lag 重新放置，而不是简单 clamp 到 min_vruntime。这让"睡很久的任务唤醒后该不该立刻抢"有了**数学定义**而非启发式。`place_entity` 完整函数体超出了本次 WebFetch 的可见窗口（fair.c 约 1.3 万行，markdown 转换截断），其内部 `se->vruntime`/`vlag`/`vslice`/`deadline` 的逐字赋值顺序标「待核」；但它"按 lag 放置"的语义由 commit `147f3ef` 与下文 lag 不变量佐证。

### 5.3.5 CFS 的结构性缺陷（EEVDF 要修的就是这些）

```
缺陷 1：wakeup preemption 是启发式，过激或不足两难
  CFS 靠 sched_wakeup_granularity_ns 判断"唤醒的任务该不该抢当前任务"。
  设小 → 交互任务及时但批量任务被频繁打断，吞吐塌。
  设大 → 吞吐好但交互卡。
  → 又回到 O(1) 时代的启发式困境，只是换了旋钮名。

缺陷 2：对 latency-sensitive 任务无显式契约
  CFS 只保证"最终公平"，不保证"我在 X 时间内能上 CPU"。
  web 请求处理、音视频 codec 是软实时，CFS 给不了上界（Corbet 原话见 5.2.4）。

缺陷 3：旋钮是全局的，无法 per-task 表达"我有多急"
  sched_latency_ns 对全系统一刀切。
  一台机器上既有延迟敏感服务又有批量任务时，没有单一旋钮值能两全。
```

这三条共同的病根：**CFS 没有"延迟"这个一等公民，只能靠外挂启发式旋钮模拟。** EEVDF 把延迟做成内生概念（slice → deadline），从根上解决。

---

## 5.4 EEVDF：最早合格虚拟截止时间优先（源码精读）

### 5.4.1 理论来源与正名

EEVDF 算法由 **Ion Stoica 和 Hussein Abdel-Wahab，1995 年技术报告**提出（标题即 *"Earliest Eligible Virtual Deadline First: A Flexible and Accurate Mechanism for Proportional Share Resource Allocation"*）。Peter Zijlstra 工程化后于 **Linux 6.6（2023）** 合入主线，commit **`147f3ef` "sched/fair: Implement an EEVDF-like scheduling policy"**（作者 Peter Zijlstra，committed by Ingo Molnar，核实原文）。

> ⚠ 易混淆（面试常踩）：Luca Abeni / Giorgio Buttazzo 那条线是 **CBS（Constant Bandwidth Server，1998）**，用在 `SCHED_DEADLINE`（EDF + CBS），跟 EEVDF **不是同一套理论**，别张冠李戴。

### 5.4.2 核心概念：lag、eligible、virtual deadline

EEVDF 在 vruntime 基础上加两个量。先看内核里对 **lag** 的官方定义（这是逐字注释，不是我编的）：

**【真实源码 v6.12 kernel/sched/fair.c，注释】**
```
Fair schedulers conserve lag: \Sum lag_i = 0.
Where lag_i is given by:  lag_i = S - s_i = w_i * (V - v_i)
```
逐符号拆解：
- `V` = 系统虚拟时间（virtual time），`v_i` = 任务 i 的 vruntime。
- `w_i` = 任务权重，`S` = 任务"应得"的服务量，`s_i` = 实际拿到的服务量。
- **`lag_i = w_i * (V - v_i)`**：任务的 vruntime 落后系统虚拟时间多少，乘以权重，就是它被"欠"的服务。
- **`\Sum lag_i = 0`**：所有任务 lag 之和恒为 0。这是公平性的**数学不变量**——有人被欠（lag>0），必有人超额（lag<0），加起来为零。CFS 没有这个显式不变量，EEVDF 把它变成可计算、可断言的东西。

**eligible（合格）** 的官方定义：

**【真实源码 v6.12 kernel/sched/fair.c，注释】**
```
Entity is eligible once it received less service than it ought to have,
eg. lag >= 0.   lag_i >= 0 -> V >= v_i
```
即：**lag ≥ 0（被欠服务）⇔ V ≥ v_i ⇔ eligible，有资格上场**。超额了（lag < 0）就暂时按住，等系统虚拟时间 V 追上来。

**virtual deadline**：`vd_i = ve_i + r_i / w_i`（commit `147f3ef` 原文）。`ve_i` 是 eligible time，`r_i` 是请求的 slice，`w_i` 是权重。**slice 越小 → deadline 越近 → 越优先**。这是 latency-sensitive 任务"我很急"的表达方式。

**选任务规则**（官方注释）：

**【真实源码 v6.12 kernel/sched/fair.c，注释】**
```
Earliest Eligible Virtual Deadline First ... selects the best runnable task
from two criteria:
  1) the task must be eligible (must be owed service)
  2) from those tasks that meet 1), we select the one with the earliest
     virtual deadline.
```

> 直觉对比：**CFS 是贪心**（谁 vruntime 最小先跑）。**EEVDF 是"约束优化"**：先用 eligibility 把"已经拿多了的人"挡在门外（保证不让人长期欠债），再在该上场的人里挑 deadline 最紧的（给每个任务可计算的延迟上界）。这正是 EEVDF 能给 latency 上界、CFS 不能的根本原因。内核里这个 lag 体现为 `sched_entity.vlag` 字段。

### 5.4.3 EEVDF 的 pick：真实源码逐行

这是 EEVDF 的心脏——怎么在一棵增广红黑树里 O(log n) 找到"eligible 且 deadline 最早"。

**【真实源码 v6.12 kernel/sched/fair.c】 `pick_eevdf`**
```c
static struct sched_entity *pick_eevdf(struct cfs_rq *cfs_rq)
{
	struct rb_node *node = cfs_rq->tasks_timeline.rb_root.rb_node;
	struct sched_entity *se = __pick_first_entity(cfs_rq);
	struct sched_entity *curr = cfs_rq->curr;
	struct sched_entity *best = NULL;

	if (cfs_rq->nr_running == 1)
		return curr && curr->on_rq ? curr : se;

	if (curr && (!curr->on_rq || !entity_eligible(cfs_rq, curr)))
		curr = NULL;

	if (sched_feat(RUN_TO_PARITY) && curr && curr->vlag == curr->deadline)
		return curr;

	if (se && entity_eligible(cfs_rq, se)) {
		best = se;
		goto found;
	}

	while (node) {
		struct rb_node *left = node->rb_left;

		if (left && vruntime_eligible(cfs_rq,
					__node_2_se(left)->min_vruntime)) {
			node = left;
			continue;
		}

		se = __node_2_se(node);

		if (entity_eligible(cfs_rq, se)) {
			best = se;
			break;
		}

		node = node->rb_right;
	}
found:
	if (!best || (curr && entity_before(curr, best)))
		best = curr;

	return best;
}
```
逐段注解：
1. **单任务快路径**：`nr_running == 1` 直接返回，不查树。
2. **curr 资格检查**：当前任务如果已经下队或不再 eligible（拿够了），置 `curr = NULL`，不让它继续占着。
3. **`RUN_TO_PARITY` 优化**：一个 sched feature。如果开启且 curr 的 `vlag == deadline`（刚好跑到它这一份 slice 的边界），**让它把这一片跑完再切**，减少无谓抢占、保护缓存局部性。这是 EEVDF 在"绝对公平"和"少切换保吞吐"之间的工程妥协旋钮。
4. **最左节点捷径**：`se = __pick_first_entity` 是 vruntime 最小的（最该被欠服务的）。如果它本身 eligible，多半就是答案，`goto found`。
5. **树上剪枝（核心）**：`while(node)` 循环。`if (left && vruntime_eligible(..., __node_2_se(left)->min_vruntime))`——**这就是"augmented 红黑树"的用法**：每个节点存了子树的 `min_vruntime`，如果左子树里存在 eligible 的任务，就往左走（左边 deadline 更早）；否则取当前节点、再往右。整个过程 O(log n)，**在 eligible 约束下找 deadline 最早**，不需要扫全树。
6. **收尾**：`if (!best || (curr && entity_before(curr, best))) best = curr`——curr 如果比树里找到的 best 更该跑（deadline 更早），让 curr 留任。

**eligibility 判定的真实代码**：

**【真实源码 v6.12 kernel/sched/fair.c】 `entity_eligible` / `vruntime_eligible`**
```c
int entity_eligible(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	return vruntime_eligible(cfs_rq, se->vruntime);
}

static int vruntime_eligible(struct cfs_rq *cfs_rq, u64 vruntime)
{
	struct sched_entity *curr = cfs_rq->curr;
	s64 avg = cfs_rq->avg_vruntime;
	long load = cfs_rq->avg_load;

	if (curr && curr->on_rq) {
		unsigned long weight = scale_load_down(curr->load.weight);
		avg += entity_key(cfs_rq, curr) * weight;
		load += weight;
	}

	return avg >= (s64)(vruntime - cfs_rq->min_vruntime) * load;
}
```
注解：这就是 `lag >= 0 ⇔ V >= v_i` 的工程落地。`avg_vruntime` / `avg_load` 是 cfs_rq 维护的**加权平均虚拟时间的分子/分母**（避免每次重算全树）。最后一行 `avg >= (vruntime - min_vruntime) * load` 等价于 `加权平均V >= 该任务相对min的vruntime`，即"系统虚拟时间是否已追上这个任务"。把除法搬成乘法（两边同乘 `load`）又是一处避除法优化。

配套的加权平均维护：

**【真实源码 v6.12 kernel/sched/fair.c】 `avg_vruntime_add`**
```c
static void
avg_vruntime_add(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	unsigned long weight = scale_load_down(se->load.weight);
	s64 key = entity_key(cfs_rq, se);

	cfs_rq->avg_vruntime += key * weight;
	cfs_rq->avg_load += weight;
}
```
注解：任务入队时，把它的 `vruntime`（相对 min 的 key）× weight 累加进 `avg_vruntime`，weight 累加进 `avg_load`。出队对应 `_sub`。**于是系统加权平均虚拟时间 V 是 O(1) 增量维护的**，pick 时直接用，不扫树。这是 EEVDF 能保持 O(log n) 的关键基础设施。

### 5.4.4 deadline 怎么推进：真实源码

**【真实源码 v6.12 kernel/sched/fair.c】 `update_deadline`**
```c
static bool update_deadline(struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	if ((s64)(se->vruntime - se->deadline) < 0)
		return false;

	if (!se->custom_slice)
		se->slice = sysctl_sched_base_slice;

	se->deadline = se->vruntime + calc_delta_fair(se->slice, se);

	return true;
}
```
逐行注解（这函数把 5.4.2 的理论全串起来了）：
- `if ((s64)(se->vruntime - se->deadline) < 0) return false;`——**还没到 deadline，不动**。注意强转 `s64` 是为了正确处理无符号回绕（vruntime 是 u64 单调增，比较差值的符号而非绝对值，是内核时间比较的标准写法）。
- `if (!se->custom_slice) se->slice = sysctl_sched_base_slice;`——如果任务没通过 `sched_setattr` 自定义 slice，就用**全局默认基准 slice** `sysctl_sched_base_slice`。**这就是 5.5 节"旧旋钮去哪了"的答案**：CFS 的 `sched_latency_ns` 没了，取而代之的是这个 base slice + per-task `custom_slice`。
- `se->deadline = se->vruntime + calc_delta_fair(se->slice, se);`——**新 deadline = 当前 vruntime + slice 折算成的虚拟时间**。正好是公式 `vd_i = ve_i + r_i/w_i`：`calc_delta_fair(slice, se)` 就是把请求 slice 按权重折算（注意它复用了 5.3.1 那个 CFS 的折算函数——**EEVDF 大量复用 CFS 基础设施的铁证**）。
- 返回 `true` 表示"这一片跑完了，需要重新评估抢占"。

`update_curr` 里调用它的关键两行（核实）：
```c
	curr->vruntime += calc_delta_fair(delta_exec, curr);   // 累积 vruntime（与 CFS 同）
	resched = update_deadline(cfs_rq, curr);               // 检查/推进 deadline（EEVDF 新增）
```
注解：**vruntime 累积逻辑和 CFS 一模一样**，EEVDF 只是在它旁边多了一行 `update_deadline`。这一行就是 CFS → EEVDF 的核心增量。

### 5.4.5 CFS → EEVDF 过渡总结

EEVDF **不是替换调度类**——`fair_sched_class` 名字都没变——而是替换**任务选择算法**。保留：红黑树、`sched_entity`、vruntime、`calc_delta_fair`、PELT 负载跟踪。变化：

- 红黑树做 **augmented**，节点维护子树 `min_vruntime`，支持 `pick_eevdf` 剪枝。**仍是一棵树**。
- `sched_entity` 新增 `deadline`、`slice`、`custom_slice`、`vlag` 等字段。
- pick 从"取最小 vruntime"变为 `pick_eevdf`（eligible 中取最早 deadline）。
- `update_deadline` 在 `update_curr` 中按 slice 推进 deadline。

**对用户态的影响**：
- 绝大多数应用无感知，行为改进但 syscall 接口（`nice`/`sched_setattr`）不变。
- ⚠ **删了一批旋钮**：`sched_latency_ns`、`sched_min_granularity_ns`、`sched_wakeup_granularity_ns` 在 6.6+ 不存在。取而代之：全局 `sysctl_sched_base_slice` + per-task slice（经 `sched_setattr` 的 `sched_runtime` 表达，范围 **100us–100ms**，核实自 LWN 969062）。**旧调优脚本 echo 这些旋钮会 `No such file`。**

---

## 5.4bis 方案对比：同一个场景，三个调度器各跑一遍

> 硬性要求里的"思考感"。下面用**一个固定场景**，让你跟着推理走，而不是只看结论。

**场景**：1 个 CPU。三个任务，全 nice 0（等权）：
- **A、B**：CPU-bound，永远想跑（计算密集，比如两个编译进程）。
- **C**：交互型，每隔 ~50ms 醒来跑约 1ms（处理一次用户输入/一个网络包），然后睡。

**关心指标**：C 的 **wakeup latency**（醒来到真正上 CPU 的时间，决定交互手感）；A、B 的吞吐与公平。

---

**① O(1) scheduler 下**

C 睡得多 → interactivity estimator **猜**它是交互任务 → 奖励动态优先级 bonus。C 醒来时优先级高于 A/B，**多数情况能较快抢占**，latency 不错。

**失败模式**：估错。若 C 的 sleep pattern 不典型（比如偶尔连续跑 5ms），estimator 可能判它"不够交互"，bonus 缩水，C 醒来排在 A/B 后，latency 突刺到几十 ms。反过来，A 若**伪装**成频繁短睡，能偷到交互 bonus，挤占 C。**latency 取决于启发式猜得准不准——不可控、可被博弈。**

---

**② CFS 下**

C 睡觉时不累积 vruntime；A、B 一直跑，vruntime 蹭蹭涨。C 醒来时 vruntime 远小于 A/B（被 clamp 到 `min_vruntime` 附近，但仍是全场最小）→ **成红黑树最左节点 → 下次调度点被选中**。

**关键细节**：C 醒来**不一定立刻抢占** A——要看 `sched_wakeup_granularity_ns`。CFS 的 wakeup 抢占逻辑是：只有当 C 的 vruntime 比当前 curr 小**超过一个 granularity** 才立刻抢。

- granularity 设大（服务器默认偏吞吐）：C 醒来要等 A 这一片跑完（最多约 `sched_min_granularity`），latency 抖动。
- granularity 设小：C 及时抢占，但 A/B 被频繁打断，**上下文切换变多、吞吐降**。

**失败模式**：单一全局 granularity **无法同时**让 C 低延迟 + A/B 高吞吐。你只能二选一或折中。这正是 5.3.5 缺陷 1。C 的 latency **最终有界但不可显式指定**（界由全局旋钮间接决定）。

---

**③ EEVDF 下**

C 可以（通过 `sched_setattr` 设小 slice，比如请求 1ms）拿到一个**很近的 virtual deadline**。即使不显式设，C 睡醒后 lag 为正（被欠服务）→ eligible，且因为它累积的服务少、deadline 相对早。

`pick_eevdf` 在 eligible 任务里选 deadline 最早的：C 醒来时 deadline 比正在跑的 A（已跑掉自己那片、deadline 较远）更早 → **C 被选中抢占**。而 A、B 之间因为 `RUN_TO_PARITY`，各自跑完自己 slice 才切，**减少无谓抢占、保吞吐**。

**为什么比 CFS 好**：C 的 latency **有可计算上界**——由它的 slice 决定（slice 越小界越紧），而不是由一个对全系统一刀切的全局 granularity 决定。**A/B 的吞吐和 C 的延迟可以同时接近最优**，因为"急不急"是 per-task 表达的，不再是全局二选一。

**对照表**：

| | O(1) | CFS | EEVDF |
|---|---|---|---|
| C 怎么被优先 | 启发式猜它交互、给 bonus | 睡醒 vruntime 最小、成最左 | 睡醒 lag>0 eligible + deadline 早 |
| C 的 latency | 取决于猜得准不准（可博弈）| 最终有界，由**全局** granularity 间接定 | **可计算上界**，由 **per-task** slice 定 |
| A/B 吞吐 vs C 延迟 | estimator 错就两败 | 全局旋钮**二选一/折中** | per-task 表达，**可兼顾** |
| 失败模式 | 误判 / 被对抗 | 单旋钮无法两全 | slice 设太碎→切换多；cgroup 交互仍待完善 |
| 公平性保证 | 弱（可饿死）| 最终公平 | 最终公平 + `\Sum lag=0` 显式不变量 |

> 一句话推理链：**O(1) 用"猜"区分任务（脆弱）→ CFS 用"严格公平"消灭猜（但丢了延迟表达）→ EEVDF 用"per-task slice + deadline"把延迟做成一等公民（公平与延迟兼得）**。三代演进就是把"延迟"这件事从启发式，到没有，再到内生的过程。

---

## 5.5 旧旋钮去哪了：EEVDF 时代的可调参数迁移（生产必读）

这是当前最高频的**升级踩坑**，单独成节。

**CFS 时代（≤ 6.5）的旋钮**（`/proc/sys/kernel/` 或 `/sys/kernel/debug/sched/`）：
```
sched_latency_ns              # 一轮调度周期，N 个任务轮一圈的总时长
sched_min_granularity_ns      # 单任务最短运行，防频繁切换
sched_wakeup_granularity_ns   # 唤醒任务是否立刻抢占的阈值
```

**EEVDF 时代（6.6+）**：上面三个**全部删除**。新模型：
```
sysctl_sched_base_slice       # 全局默认基准 slice（update_deadline 里用，见 5.4.4 真实源码）
                              # debugfs 下名字形如 sched_base_slice_ns（具体路径随版本，待核精确名）
per-task slice                # 经 sched_setattr 的 sched_runtime 字段表达
                              # 范围 100us – 100ms（核实自 LWN 969062，Corbet 2024-04）
```

**踩坑现场**：
```
# 旧 CFS 调优脚本（很多公司 ansible/启动脚本里还有）：
echo 3000000 > /proc/sys/kernel/sched_latency_ns          # 6.6+ 上：No such file or directory
echo 1000000 > /sys/kernel/debug/sched/min_granularity_ns # 6.6+ 上：同样报错

# 后果：脚本静默失败（如果没 set -e）或启动中断；
#       运维以为"调了 latency"，实际内核根本没这个旋钮，调优是幻觉。
```

**迁移方法**：
- 想让某个延迟敏感任务更"急"：**别再调全局**，改用 `sched_setattr(2)` 给它设小 `sched_runtime`（slice）。slice 小 → deadline 近 → 抢占及时。
- 想全局改基准时间片：调 `sysctl_sched_base_slice`（debugfs，具体路径名待核）。但**不要再用 CFS 那套"算 sched_latency / n"的心智**——EEVDF 的延迟是 per-task deadline，不是全局周期。
- **机制理解仍成立**：本章 CFS 模型对"时间片随任务数缩短"的描述帮助理解，但**别在 6.6+ 内核上找那些文件**。

> ⚠ EEVDF 收尾还在进行（LWN 969062，Corbet，2024-04）：lag decay、deferred dequeue（睡眠任务留在队列让 lag 随虚拟时间增长到转正再真正出队）等在 6.6 之后版本陆续完善。**特别注意：该文明确指出"与 cgroup 的交互尚未充分验证、可能不正常工作"**——所以下面 5.8 的 cgroup throttle 行为在 EEVDF 早期版本上可能有细微差异，生产升级前务必在目标内核版本上实测。

---

## 5.6 调度类与策略

### 5.6.1 调度类层次

Linux 用**调度类（sched_class）**做策略分层，优先级严格从高到低：

```
优先级（高→低）   调度类              策略
─────────────────────────────────────────────────
最高              stop_sched_class    内部用（CPU hotplug、migration）
                  dl_sched_class      SCHED_DEADLINE
                  rt_sched_class      SCHED_FIFO, SCHED_RR
                  fair_sched_class    SCHED_NORMAL, SCHED_BATCH（CFS/EEVDF 都在这）
最低              idle_sched_class    SCHED_IDLE
─────────────────────────────────────────────────
```

调度类是**多态策略对象**（类似 vtable），每类实现 `enqueue_task`、`dequeue_task`、`pick_next_task`、`task_tick` 等。`pick_next_task` 按这个顺序逐类问"你有没有任务要跑"，第一个有的就赢。**EEVDF 的全部改动都发生在 `fair_sched_class` 内部**，没碰这个分层。

> 工程美感：新增调度类（如未来的 sched-ext BPF 调度器）不必改核心调度循环，只实现这套接口。开闭原则的教科书案例。

### 5.6.2 实时调度（SCHED_FIFO / SCHED_RR）

```
SCHED_FIFO：同优先级 FIFO，高优先级可无限占 CPU（无时间片耗尽），只有 block/yield 才切。优先级 1-99。
SCHED_RR ：同 FIFO，但有时间片（默认 100ms，可经 sched_rr_timeslice_ms 调；耗完放同优先级队尾。优先级 1-99。

RT 任务可饿死 fair 任务！
/proc/sys/kernel/sched_rt_runtime_us 限制 RT 最多占多少时间。
默认 rt_runtime_us = 950000（即每 sched_rt_period_us=1000000 的周期里最多 95%），保留 5% 给非 RT，防系统完全锁死。
```

🎯 **面试加分点**：为什么 RT 线程跑满 CPU 系统还能 SSH 进去？答案是 `sched_rt_runtime_us` 的 5% 保留。手动设 `-1`（禁用限制），RT 线程真能把系统搞死。

### 5.6.3 SCHED_DEADLINE（EDF + CBS）

```
sched_setattr 设三参（纳秒）：
  runtime ：每 period 内最多跑多少
  period  ：重复周期
  deadline：每 period 内必须完成 runtime 的截止时间

例：视频编解码，每 33ms（30fps）最多需 10ms CPU
  runtime=10ms, period=33ms, deadline=33ms
```

SCHED_DEADLINE 有**准入控制**：设置时内核检查系统总带宽（基于 EDF 可调度性 + CBS），不够返回 `EBUSY`，防 over-commitment。这是 FIFO/RR 没有的保护。

**新手误区**：deadline 不是硬实时保证——Linux 不是 RTOS，关中断临界区、NMI 等会破坏硬实时性。它提供"统计/软实时"。要硬实时上 `PREEMPT_RT`（见 5.7）。

### 5.6.4 nice 值的工程含义与局限

```bash
ps -o pid,ni,comm -p <pid>      # 查 nice
nice -n 10 ./my_batch_job       # 启动时设
renice -n -5 -p <pid>           # 运行中改（降低 nice 需 CAP_SYS_NICE / root）
```

**局限**：nice 只影响 fair 层的 weight（5.3.2 那张表）。**有 RT 任务时 nice 毫无意义**——RT 不看 nice，永远先跑。EEVDF 后 nice 依旧只决定 weight，但现在 weight 既影响 vruntime 累积、也影响 `vd_i = ve_i + r_i/w_i` 里的 deadline 折算。

---

## 5.7 抢占点

### 5.7.1 抢占类型（四档心智 + 动态可调）

```
PREEMPT_NONE（吞吐优先）：内核态不被抢，只在 syscall 返回用户态/主动让出时切。最差延迟最大，切换最少。曾是服务器发行版默认。
PREEMPT_VOLUNTARY：内核里散布显式抢占点（cond_resched() 等）处可让。"到点才让"，latency 比 NONE 好但仍有缺口。桌面/通用折中。
PREEMPT（full）：非持锁/禁抢占/禁中断区，内核态任意点可抢。低延迟桌面/多媒体，调度开销稍高。
PREEMPT_RT：曾长期 out-of-tree，近年已大部分合入主线。把绝大多数 spinlock 转可睡眠 rtmutex、中断线程化，关中断窗口极短。最差延迟最小，吞吐代价最大，工业实时用。
```
> 时效：近年内核推进 **`CONFIG_PREEMPT_DYNAMIC`**——启动参数 `preempt=none|voluntary|full` 或 debugfs 运行时切换，一个内核镜像覆盖多档。还引入 **lazy preemption**（见 5.7.2）。

### 5.7.2 抢占触发与 TIF_NEED_RESCHED

```
触发 reschedule 的时机：
1. timer interrupt（task_tick → EEVDF 下经 update_curr/update_deadline 判 deadline 到没到）
2. wakeup：唤醒任务 deadline 足够早（EEVDF）/ vruntime 足够小（CFS）时抢占当前
3. yield_to_task / sched_yield
4. 任务 block（I/O、mutex）

真正切换的检查点：
- 从 syscall/interrupt 返回用户态（最常见）
- 主动 schedule()
- PREEMPT 下 preempt_enable() 发现 TIF_NEED_RESCHED
```

**TIF_NEED_RESCHED**：当前任务 `thread_info.flags` 的一个标志位（内核经 `resched_curr()` 设置，旧名 `resched_task()`）。置位后**不立刻切**，在最近的安全检查点才落地真正上下文切换。这种"标志 + 延迟落地"是内核抢占的核心机制——把"决定要抢"和"安全地抢"解耦，避免在持锁/临界区被打断。

> 进阶（近年）：为进一步降延迟，引入 `TIF_NEED_RESCHED_LAZY`（lazy preemption），让"立即抢"和"下次返回边界再抢"两档语义共存，是 PREEMPT_RT 长期演进的一部分。具体行为随版本变化，待核。

---

## 5.8 负载均衡

### 5.8.1 SMP 负载均衡

每核独立 run queue，负载均衡把任务从忙核移到闲核：

```
触发：
1. 周期性（tick / 专用 balance 路径）
2. CPU 变 idle 时（idle load balance，最重要——闲着就该拉活）
3. fork/exec 时（新任务可能放到别的核）

策略（调度域 sched_domain，自底向上）：
  SMT（同物理核超线程）→ 同 die 的核 → 跨 NUMA node（代价最高，见第 8 章）
```

**调度域（sched_domain）** 描述 CPU 拓扑 + 每层均衡策略。`taskset -c <cpus> <pid>` 手动 pin 绕过均衡。

> EEVDF 与负载均衡的关系：EEVDF 改的是**单 rq 内**的 pick 算法。跨核负载均衡仍基于 load（PELT）而非 vruntime/deadline，这套大体未变。但任务迁移时 `vlag`/`vruntime` 要按目标 rq 的虚拟时间重新归一（`place_entity` 路径，细节待核）。

### 5.8.2 NUMA 感知调度

```
numa_balancing（/proc/sys/kernel/numa_balancing）：
  周期性把任务迁到其内存所在 NUMA node。
  代价：迁任务 + 迁/重映射内存页，可能短暂 latency 尖刺。

生产实践：
  - 延迟敏感服务：numactl --membind=0 --cpunodebind=0 固定 node
  - 高内存带宽计算：让 numa_balancing 自动
```

---

## 5.9 CFS/EEVDF 带宽控制（cgroup CPU quota）

> 注意：bandwidth/throttle 这套**独立于** CFS→EEVDF 的 pick 算法演进，跨版本基本不变（EEVDF 没动它）。但见 5.5 末尾的 cgroup 交互待完善提醒。

### 5.9.1 机制

```
cgroup v1（两个文件，单位微秒）：
  cpu.cfs_period_us = 100000  # 100ms 配额刷新周期
  cpu.cfs_quota_us  = 50000   # 每周期最多 50ms = 0.5 个 CPU
                    = 200000  # = 2 个 CPU（跨核累加）
                    = -1      # 不限制

cgroup v2（合成一个文件）：
  cpu.max = "200000 100000"   # "<quota> <period>"，微秒
  cpu.max = "max 100000"      # 不限制
```

**机制细节**：每个 cgroup 有 bandwidth pool（配额桶），每核本地有 runtime 余量；任务跑消耗本地 runtime，耗完从全局池补，全局池耗完该 cgroup 被 **throttle** 到下个 period 刷新。

### 5.9.2 生产 BUG：Throttle 导致 P99 爆炸

```
症状：
  CPU util = 30%（看起来很闲），但 p99 = 500ms（本该 10ms）
  kubectl top pod 显示 CPU 在 limit 附近徘徊

根因：
  cfs_quota_us=100ms（1 CPU），但任务突发：
  某 100ms period 内 6 线程同时跑，额度 ~16ms 内耗光，
  剩余 ~84ms 内全部线程被 throttle，即使物理核空闲。

诊断：
  cat /sys/fs/cgroup/.../cpu.stat | grep -E 'nr_throttled|throttled'
  # throttled 时间持续增长 → 确认 throttle
  # Prometheus: container_cpu_cfs_throttled_seconds_total

修复方向：
  1. 提高 cpu.max quota
  2. 只设 request 不设 limit（适合可超卖集群）
  3. 调大 period（减少 throttle 粒度，但延迟精度变差）
```

💡 **坑点**：K8s 的 `resources.limits.cpu` 实际就是设 `cfs_quota_us`。`0.5 CPU` 不是"最多用 50% CPU 时间"，而是"每 100ms 最多用 50ms"——突发场景下两者天差地别。

### 5.9.3 工具诊断链

```bash
cat /proc/<pid>/sched          # vruntime, nr_switches, wait_sum 等（EEVDF 下还能看到 deadline/slice 相关，字段随版本）

perf sched record -g -- sleep 5
perf sched latency             # 看 wakeup→on-cpu 延迟

# bpftrace 统计 runqueue 等待（wakeup→on-cpu），最小可跑版：
bpftrace -e '
  tracepoint:sched:sched_wakeup,
  tracepoint:sched:sched_wakeup_new { @ts[args->pid] = nsecs; }
  tracepoint:sched:sched_switch /@ts[args->next_pid]/ {
    @us[args->next_comm] = hist((nsecs - @ts[args->next_pid]) / 1000);
    delete(@ts[args->next_pid]);
  }'
# 现成工具：bcc 的 runqlat / runqslower 直接给 runqueue latency 直方图

turbostat --interval 1         # CPU 频率/C-state，排除降频干扰
```

---

## 5.10 落地：生产调优与排障

### 5.10.1 延迟敏感服务调优清单

```
1. SCHED_FIFO/RR（需 CAP_SYS_NICE / root）：
   chrt -f 50 <pid>   # FIFO 优先级 50
   chrt -r 50 <pid>   # RR
   ⚠ 确保 sched_rt_runtime_us 留余量，否则整系统卡顿

2. CPU isolation：
   启动参数 isolcpus=2-5 隔离几个核只跑关键任务
   taskset -c 2-5 <pid>

3. IRQ affinity：网络 IRQ 绑非隔离核，防 IRQ 打断关键任务
   /proc/irq/<N>/smp_affinity

4. 关 numa_balancing（已手动固定 node 时）：
   echo 0 > /proc/sys/kernel/numa_balancing

5. 让某任务更"急"：
   ⚠ CFS 内核（≤6.5）：减小 sched_wakeup_granularity_ns
   ✅ EEVDF 内核（6.6+）：该旋钮已删！改用 sched_setattr 给它设小 slice
      （sched_runtime 字段，范围 100us–100ms）。
      不要再去 echo 全局 wakeup granularity——文件不存在。
```

### 5.10.2 常见排障场景

```
场景 A：偶发高延迟，CPU idle 很高
  1. cgroup throttle（首查 cpu.stat 的 nr_throttled/throttled）
  2. 被高优先级抢占（perf sched latency 看 wakeup latency）
  3. NUMA miss（numastat 看 other_node）
  4. 降频（turbostat 看实际频率）

场景 B：批量任务很慢，CPU util=100%
  1. 与在线任务竞争，被低 nice 任务挤
  2. 负载均衡不均（某核 100% 某核空）→ taskset / 查 sched_domain
  3. SMT 假 CPU（每对 HT 共享物理资源）

场景 C：fork 炸弹 / 进程数爆炸
  n_running 巨大时，CFS 时间片缩小、延迟变长；EEVDF 下大量任务竞争同样使
  每任务有效 slice 被压缩、deadline 普遍后移。
  → ulimit -u、cgroup pids.max 限制
```

---

## 5.11 面试视角（Staff 级）

面试官考调度器通常三层：
1. **原理层**：讲清 vruntime / CFS pick / EEVDF eligible+deadline。能背出 `calc_delta_fair` fast path、`pick_eevdf` 树剪枝就拉开差距。
2. **工程层**：用 perf/bpftrace 排过真实调度问题；知道 6.6 旧旋钮被删这种"时效坑"。
3. **权衡层**：能评价不同策略 trade-off，讲清 O(1)→CFS→EEVDF 在三角上各往哪移、为什么 EEVDF 能兼顾延迟与吞吐而 CFS 不能。

---

## 章末五件套

### 一、高频面试题（Staff 级）

**Q1：讲一下 CFS 的核心机制，vruntime 是什么，为什么用红黑树？**

- vruntime = 任务"应得的虚拟 CPU 时间"，经 `delta_real × (NICE_0_LOAD / task_weight)` 折算（真实函数 `calc_delta_fair`/`__calc_delta`，用 inv_weight 把除法变乘法 + 移位）。weight 来自 nice 查表 `sched_prio_to_weight`。
- 红黑树以 vruntime 为 key，最左 = 最小 → pick = 取最左，O(log n)（带 `rb_leftmost` 缓存接近 O(1)）。
- `min_vruntime` 防睡眠唤醒任务以 vruntime=0 插队。

**错误陷阱**：说"CFS 让每个进程完全平等用 CPU"——错，是**加权公平**。"完全公平"的公平 = 按权重成比例，不是人人均等。

---

**Q2：EEVDF 解决了 CFS 的什么问题？怎么解决的？**

- CFS 对 latency-sensitive 任务无 deadline 契约，只选 vruntime 最小，靠全局 `sched_wakeup_granularity_ns` 启发式模拟延迟（Corbet：CFS "does not give processes a way to express their latency requirements"）。
- EEVDF 加 `lag`（`lag_i = w_i*(V-v_i)`，`\Sum lag=0`）定义 eligible，加 `vd_i = ve_i + r_i/w_i` 定义 deadline。`pick_eevdf` 在 eligible 中取最早 deadline。slice 越小 deadline 越近 → latency-sensitive 任务越优先。
- 合入：6.6（2023），Peter Zijlstra，commit `147f3ef`。

**错误陷阱**：说"EEVDF 是全新调度器替换 CFS"——错。`fair_sched_class` 名字没变，复用红黑树/sched_entity/vruntime/calc_delta_fair，是**算法层演进不是架构重写**。`update_curr` 里只比 CFS 多一行 `update_deadline`。

---

**Q3：SCHED_FIFO 和 SCHED_RR 区别？何时用 RT？**

- FIFO：同优先级 FIFO，无时间片耗尽，只 block/yield 才让。
- RR：有时间片（默认 100ms），耗完放同优先级队尾。
- 场景：硬件交互（工业控制、实时音频）、latency 极敏感且可信任的服务。
- 风险：RT 可饿死普通任务，必须配 `sched_rt_runtime_us`（默认 95%）。

---

**Q4：K8s 设 `limits.cpu: 0.5` 会怎样？为什么可能造成 latency 问题？**

- 转化为 `cfs_quota_us=50000, cfs_period_us=100000`，每 100ms 最多 50ms。
- 突发（多 goroutine 并发）50ms 很快耗光，剩余全 throttle。
- CPU util 看着 <50% 但 throttle 时段服务不响应，p99 爆。
- 诊断：`container_cpu_cfs_throttled_seconds_total` / `cpu.stat` 的 `nr_throttled`。
- 补充：这套 throttle 逻辑 EEVDF 没改，跨版本一致（但 EEVDF 早期 cgroup 交互仍在完善，见 LWN 969062）。

---

**Q5：讲讲调度类设计，为什么要这个抽象？**

- 多态策略对象（interface/vtable），每类实现 enqueue/dequeue/pick_next/task_tick。
- 优先级链：stop > deadline > rt > fair > idle。
- 动机：不同负载需完全不同策略，用多态替代 if-else，核心循环只调 sched_class 指针方法。
- 推论：新增调度类（如 sched-ext）不改核心，开闭原则。EEVDF 全部改动局限在 `fair_sched_class` 内。

---

**Q6：服务 p99 高但 CPU util 正常，怎么排查？**

分层：① cgroup throttle（`cpu.stat` 的 `throttled` 是否涨）② wakeup latency（`perf sched latency` / bpftrace `sched:sched_wakeup`）③ NUMA（`numastat` other node）④ IRQ 打断（`/proc/interrupts`、`irqbalance`）⑤ 降频（`turbostat`）。

---

**Q7：什么是负载均衡？跨 NUMA 迁移代价？**

- 负载均衡 = 忙核→闲核迁任务，触发含 idle 时和周期检查。
- 调度域层次：SMT → core → die → NUMA node，代价递增。
- 跨 NUMA：任务迁移（cache 冷启动）+ 内存仍在原 node（remote 访问，带宽降延迟增）。
- 实践：延迟敏感服务 numactl 固定。

---

**Q8：CONFIG_PREEMPT 和 CONFIG_PREEMPT_RT 区别？**

- PREEMPT：非临界区任意点可抢，桌面/低延迟。
- PREEMPT_RT：spinlock 转 rtmutex、中断线程化，关中断窗口极短，工业实时。近年大部分已合入主线。
- 权衡：更强抢占 → 更低最坏延迟，但调度开销略高、吞吐可能降。
- 加分：提 `CONFIG_PREEMPT_DYNAMIC` 运行时切档 + lazy preemption。

---

**Q9：nice -20 和 +19 实际 CPU 时间差多少？**

- weight：nice -20 = **88761**，nice +19 = **15**（精确值，来自 `sched_prio_to_weight`）。
- 比值 88761/15 ≈ **5917:1**。两者对跑时前者约拿 5900 倍 CPU。
- 但这是相对比例；系统只有这两任务时，nice +19 几乎拿不到。
- 常见情况 nice 差 ≤10，约 10 倍（`1.25^10≈9.3`）。

**错误陷阱**：说"nice 每格约 10%"——不严谨，真正乘子是约 1.25/格，"10%"只在两任务对跑时近似。

---

**Q10（设计题）：设计 fair queueing 让多客户端公平共享带宽，类比 CFS？进一步如何给延迟上界（类比 EEVDF）？**

- CFS 类比：每客户端维护 virtual finish time（≈ vruntime），下个服务 = vft 最小者的头包，weight → vt 增量 = 包大小/weight。这就是 **WFQ**。
- EEVDF 类比：再给每个客户端一个基于其请求大小的 **virtual deadline**，只在"被欠服务（eligible）"的客户端里挑 deadline 最早的发 → 给每个客户端**可计算的延迟上界**。这就是 **WF2Q / EEVDF** 思想。能讲到这一步说明你真懂两代的差异。

---

### 二、实战项目（P3 预备）

**目标**：理解后用用户态模拟 CFS-lite，再扩展到 EEVDF-lite，为 P3 打基础。

**项目：用户态 CFS / EEVDF 模拟器**

```c
// sched_lite.c 骨架  —— 验证：观察 vruntime 增长与 weight 成反比；EEVDF 模式下短 slice 任务延迟更低
#include <stdio.h>
#include <stdint.h>

#define NICE_0_WEIGHT 1024  // 与内核 sched_prio_to_weight[nice 0] 一致

// 真实内核 sched_prio_to_weight 的子集（nice -5..5），查表不是公式！
static const int w_tab[] = { 3121,2501,1991,1586,1277, 1024, 820,655,526,423,335 };
int nice_to_weight(int nice) { return w_tab[nice + 5]; }  // 仅演示 -5..5

typedef struct {
    int pid, nice, weight;
    uint64_t vruntime;   // 虚拟运行时间(ns)
    uint64_t deadline;   // EEVDF: 虚拟截止
    uint64_t slice;      // EEVDF: 请求时间片(ns)
} task_t;

// CFS pick：最小 vruntime（线性扫代替红黑树）
task_t *pick_cfs(task_t *t, int n) {
    task_t *m = NULL;
    for (int i = 0; i < n; i++) if (!m || t[i].vruntime < m->vruntime) m = &t[i];
    return m;
}

// EEVDF pick（简化）：在 vruntime<=平均V 的 eligible 任务里取最早 deadline
task_t *pick_eevdf(task_t *t, int n) {
    uint64_t sumv = 0; for (int i=0;i<n;i++) sumv += t[i].vruntime;
    uint64_t avgV = sumv / n;                 // 简化的系统虚拟时间
    task_t *best = NULL;
    for (int i = 0; i < n; i++) {
        if (t[i].vruntime > avgV) continue;   // 不 eligible（拿多了），跳过
        if (!best || t[i].deadline < best->deadline) best = &t[i];
    }
    return best ? best : pick_cfs(t, n);      // 没人 eligible 兜底
}

void run_slice(task_t *x, uint64_t real_ns) {
    x->vruntime += real_ns * NICE_0_WEIGHT / x->weight;   // = calc_delta_fair 的简化
    x->deadline  = x->vruntime + x->slice * NICE_0_WEIGHT / x->weight; // vd = v + r/w
    printf("pid=%d ran %lluns vruntime=%lu deadline=%lu\n",
           x->pid, real_ns, x->vruntime, x->deadline);
}

int main(void) {
    task_t t[] = {
        {1, 0, 1024, 0, 0, 6000000},   // A: CPU-bound, slice 6ms
        {2, 0, 1024, 0, 0, 6000000},   // B: CPU-bound, slice 6ms
        {3, 0, 1024, 0, 0, 1000000},   // C: 交互, slice 1ms → deadline 更近
    };
    for (int r = 0; r < 12; r++) run_slice(pick_eevdf(t, 3), 1000000); // 切 pick_cfs 对比
    return 0;
}
```

**验收**：
- CFS 模式：三任务等权 → run_slice 次数大致相等（公平）。
- EEVDF 模式：C（slice=1ms）的 deadline 始终更近 → 被选中更频繁、相邻两次被选间隔更短（延迟更低），但长期总份额仍接近 1/3（公平不破）。**亲手验证"短 slice 换低延迟但不破公平"**。
- 加 sleep/wakeup：睡眠任务醒来 vruntime 落后 → CFS 立刻插到最前；EEVDF 用 lag 决定（进阶）。

**暗坑**：
- nice→weight 是查表不是线性公式，直接算必错。
- vruntime 用 u64：纳秒计可跑约 584 年不溢出；用 u32 会 wrap。内核比较 vruntime 一律用 `(s64)(a-b)` 看符号，别直接比大小（5.4.4 真实源码就是这么写的）。

---

### 三、设计题

**题目**：在线服务（高并发 HTTP，p99<10ms）+ 批量作业（离线处理）混合部署，要求互不饿死、批量在 idle 多跑。

**要点**：
- 在线：cgroup 高 weight + 用 request 不用 limit（防 throttle）；或 nice -5；EEVDF 内核上额外给它**小 slice**（`sched_setattr`）压低延迟上界。
- 批量：SCHED_BATCH 或 nice +15，idle 才跑。
- CPU isolation：部分核 isolcpus 专供在线，其余共享。
- 监控：`container_cpu_cfs_throttled_seconds_total`、`node_context_switches_total`。
- 权衡：isolation 提升在线稳定性但降批量利用率，按 SLA 定比例。
- **版本意识**：若目标是 6.6+，调优手段从"调全局 sched_latency"改为"per-task slice"，并复核 cgroup 交互（LWN 969062 提醒早期可能不完善）。

---

### 四、系统题（真实排障）

**场景**：K8s Java 微服务，GC 很快（<10ms），但 HTTP p99=300ms，CPU 在 limit 的 80%。如何排查？

```bash
# Step 1: cgroup throttle
# v1: /sys/fs/cgroup/cpu/cpu.stat   字段 nr_throttled / throttled_time(ns)
# v2(现代 K8s 多为此): /sys/fs/cgroup/<slice>/cpu.stat   nr_periods/nr_throttled/throttled_usec(us)
kubectl exec <pod> -- sh -c 'cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null || cat /sys/fs/cgroup/cpu.stat'
# 看 nr_throttled 与 throttled_(time|usec) 是否持续增长（v1 单位 ns，v2 us）

# Step 2: throttle 率 = throttled_periods / total_periods，>5% 值得关注
# Step 3: kubectl top pod --containers  看 burst 峰值
# Step 4: 临时提高 limit，若 p99 立刻改善 → 确认 quota throttle
# Step 5: 永久修复：提高 cpu.max / 改只设 request（集群可超卖时）
```

**量级参考**：cfs_period=100ms，100 并发线程爆发时 50ms quota 在亚毫秒级耗光，剩余 ~99ms 全 throttle → p99 直接打满。

---

### 五、代码题

**题目**：bpftrace 脚本统计每进程 wakeup 延迟（唤醒→上 CPU），输出分桶直方图。

```bash
#!/usr/bin/env bpftrace
// wakeup_latency.bt  验证：另一终端 stress-ng --cpu 4，观察分布变化。需 root；bpftrace>=0.12
tracepoint:sched:sched_wakeup,
tracepoint:sched:sched_wakeup_new { @wake_ts[args->pid] = nsecs; }

tracepoint:sched:sched_switch {
    $pid = args->next_pid;
    if (@wake_ts[$pid]) {
        $lat = (nsecs - @wake_ts[$pid]) / 1000;     // us
        @latency_us = hist($lat);
        @by_comm[$pid, args->next_comm] = hist($lat);
        delete(@wake_ts[$pid]);
    }
}
END {
    printf("\n=== Wakeup Latency (us) ===\n"); print(@latency_us);
    printf("\n=== Top by latency ===\n");      print(@by_comm);
    clear(@wake_ts);
}
```

**暗坑**：
1. `sched_wakeup` 与 `sched_switch` 间可能有别的进程先切，`@wake_ts` 必须 pid 做 key 不能用全局标量。
2. 容器里 tracepoint 需特权 + `hostPID: true`。
3. bpftrace map 有默认大小限制，pid 多时溢出（调 `BPFTRACE_MAP_KEYS_MAX`）。
4. 别依赖 `args->delay`，用 `nsecs` delta 手算更可靠。
5. EEVDF 内核上想进一步看"为什么这个任务延迟高"，可加 `kprobe:pick_eevdf` 或读 `/proc/<pid>/sched` 的 deadline/slice 字段交叉印证。

---

## 章末心法

**调度器不优化 CPU 利用率，它分配 CPU 时间的使用权。** O(1) 用启发式"猜"谁该优先（脆弱、可博弈）；CFS 用严格加权公平消灭了猜，却把"延迟"丢成了只能靠全局旋钮模拟的二等公民；EEVDF 用 per-task slice + virtual deadline 把延迟做成一等公民，在数学上同时给出公平不变量（`\Sum lag=0`）和可计算的延迟上界。**读懂这条从"猜"到"没有"再到"内生"的演进线，你才算把 throttle、vruntime、deadline 这些设计决策背后的哲学吃透。**

---

## 附：本章源码与史料出处

- 真实源码基准：**Linux v6.12**，`kernel/sched/fair.c`、`kernel/sched/sched.h`、`kernel/sched/core.c`（经 raw.githubusercontent.com/torvalds/linux/v6.12 取得逐字片段）。
- EEVDF 实现 commit：`147f3efaa24182a21706bca15eab2f3f4630b5fe` "sched/fair: Implement an EEVDF-like scheduling policy"（Peter Zijlstra，committed by Ingo Molnar）。
- LWN：
  - "An EEVDF CPU Scheduler for Linux"（Jonathan Corbet，2023-03-09，articles/925371）
  - "[patch] Modular Scheduler Core and Completely Fair Scheduler [CFS]"（Ingo Molnar 原始公告，2007-04-13，articles/230501）
  - "RSDL completely fair starvation free interactive cpu scheduler"（Con Kolivas，2007-03-04）
  - "Completing the EEVDF scheduler"（Jonathan Corbet，2024-04-11，articles/969062）
- 理论：Stoica & Abdel-Wahab, *EEVDF: A Flexible and Accurate Mechanism for Proportional Share Resource Allocation*（1995 技术报告）。
- 标「待核」处：`SCHED_FIXEDPOINT_SHIFT` 精确定义、`place_entity` 逐字函数体、`sysctl_sched_base_slice` 的 debugfs 精确路径名、`TIF_NEED_RESCHED_LAZY` 具体行为——这些 fetch 未逐字取到或随版本变动，正文已就地标注。
