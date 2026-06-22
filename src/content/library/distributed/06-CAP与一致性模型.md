---
title: "CAP 与一致性模型：用仿真量化取舍"
slug: "06"
collection: "distributed"
order: 6
summary: "第 5 章证明了「分区下少数派提不了交」是 Raft 的机械事实；本章把它上升成定理：分区一定发生，发生时只能在一致性(C)和可用性(A)里二选一。用同一份分区脚本分别跑 CP（拒写保 C）和 AP（全收保 A），量出两套代价——CP 写成功 10/12 但 0 分歧，AP 写成功 12/12 但 3 个分歧键，外加一个时钟偏移下静默丢写的失败模式。这个丢写的洞正是第 7 章用 CRDT 收拾的烂摊子。"
topics:
  - "分布式系统"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

先把第 5 章的结论摆出来：Raft 里少数派分区上的 leader 收不齐 quorum（多数派投票），所以它的写永远 commit 不了。那一章我们把它当成「Raft 这么设计」的细节。这一章要说的是：**那根本不是 Raft 的选择，是数论强加给所有 quorum 系统的命数**。N=5 的集群被切成 3 和 2，永远只有一侧能凑够多数。两侧都想接受写、还想保持一致？counting 不允许。CAP 定理就是这句话的正式版。

我直接把这条命数做成可跑的二维账单。同一份分区脚本、同一份 12 次写的工作负载，跑两遍：一遍 CP（凑不齐 quorum 就拒写），一遍 AP（两侧无脑全收，事后用 last-write-wins 合并）。然后量两个数：可用性 = 写成功率，一致性 = 分区愈合后还有几个键各副本各执一词。代码在 `examples/distributed-from-scratch/src/stage06-cap-tradeoff.ts`。

## CAP 不是三选二：先把误读拆掉

几乎每篇博客都把 CAP 念成「Consistency、Availability、Partition tolerance 三个里挑两个」。这是错的，而且错得有害——它让人以为「我选 CA，放弃 P」是个合法选项。不是。

**P（partition tolerance，分区容忍）不是一个你能勾掉的特性，它是环境给你的事实**。网线会断，交换机会挂，GC（垃圾回收停顿）会让一个节点几百毫秒不应答，TCP 连接会僵死。只要你的系统跑在多台机器上，分区（partition，节点间消息丢失/延迟到无法通信）就一定会发生。你「选择不容忍分区」的唯一后果是：分区真发生时系统行为未定义——既不保证 C 也不保证 A，比两个都差。所以「CA 系统」在分布式语境下是个空集；单机数据库能叫 CA，但它根本没进 CAP 的讨论范围。

真正的命题是这样的：**既然 P 必然发生，那么在分区发生的那一刻，你只能在 C 和 A 里二选一**。CAP 是一个「分区期间」的局部定理，平时它对你毫无约束。这是第一个专家钩子。✦

第二个更要命：**CAP 里的 C 特指线性一致性（linearizability，每个操作看起来在某个瞬间原子生效、且尊重真实时间先后），不是泛泛的「数据一致」**。很多系统号称「保证一致性」，指的是最终一致或读己之写，那跟 CAP 的 C 是两码事。把 C 念成「数据别乱」，你会以为 AP 系统「牺牲了一致性」是个小事——其实它牺牲的是「任何客户端任何时刻读到的都是最新值」这个最强的保证，代价比直觉大得多。

所以 CAP 的正确读法是一句话：**分区时，要么拒绝一部分请求来维持线性一致（CP），要么响应所有请求但放弃线性一致（AP）。** 下面用仿真把这句话的两边都标上价格。

## 为什么 quorum 让 CP 必然牺牲可用性

CP 那一侧的可用性损失不是「我们决定拒绝少数派」的政策，而是 counting 逼出来的。看 `runCpStrategy` 的注册逻辑——一次写要 commit，coordinator（接收客户端写的那个节点）必须收齐 `WRITE_QUORUM = 3` 个 ack：

```typescript
} else if (msg.kind === "Ack") {
  const { writeId } = msg.payload as { writeId: number };
  if (writeDecided.has(writeId)) return; // already terminal; ignore late acks
  const c = (ackCounts.get(writeId) ?? 0) + 1;
  ackCounts.set(writeId, c);
  if (c >= WRITE_QUORUM) {
    writeDecided.add(writeId);
    committed++;
    // ... 广播 Commit
  }
}
```

集群 N=5，写 quorum W=3，读 quorum R=3。文件头注释把数论讲透了：单键的线性一致读要求 `W + R > N` **且** `W > N/2`。前一个条件（6 > 5）保证读集合和写集合必然相交，读到的至少有一个副本带着最新写；后一个条件（3 > 2.5）才是 CAP 的杠杆——**只有持多数派的那一侧能凑齐 quorum，任何分区里最多一侧能完成写**。

把 5 个节点切成多数派 `n0,n1,n2`（3 个）和少数派 `n3,n4`（2 个），少数派那边的 coordinator 怎么努力都凑不齐 3 个 ack：它自己 1 个，对面 3 个 ack 被网络在分区窗口里丢掉了。于是它超时、拒写：

```typescript
clock.schedule(ACK_TIMEOUT_MS, () => {
  if (writeDecided.has(writeId)) return; // committed in time
  writeDecided.add(writeId);
  rejected++;
  // Abort: drop the buffered (never-committed) write on every replica. This
  // is the CP guarantee made concrete — an unavailable write leaves ZERO
  // residue, so post-heal divergence is 0 by construction.
  for (const id of NODE_IDS) pending.get(id)!.delete(writeId);
});
```

注意这里的关键设计：CP 用两阶段（Prepare 先 buffer 不落地，收齐 quorum 才 Commit 落地）。被拒的写从没在任何副本的 `store` 上留过痕，所以愈合后没有任何东西需要 reconcile——**CP 的 0 分歧是「构造上」成立的，不是「运气好没冲突」**。可用性的损失（少数派写被拒）和一致性的保全（0 残留）是同一枚硬币的两面。

## AP 那一侧：可用性满格，账记在愈合时

AP 反过来：每个 coordinator 无条件接受本地写，立刻返回成功。`runApStrategy` 里没有任何 quorum 检查：

```typescript
for (const w of workload) {
  clock.schedule(w.issuedAtMs, () => {
    // AP accepts unconditionally — that's the availability win. The timestamp
    // is the coordinator's LOCAL physical clock ...
    const skew = MINORITY.includes(w.coordinator) ? MINORITY_CLOCK_SKEW_MS : 0;
    const stampMs = clock.now() + skew;
    const versioned: VersionedValue = { value: w.value, stampMs, realTimeMs: clock.now() };
    applyLww(store.get(w.coordinator)!, w.key, versioned);
    committed++;
    // Best-effort replicate to peers. Cross-partition sends are dropped ...
```

可用性满格的代价在分区期间悄悄累积：两侧对同一个键各写各的，跨分区的复制消息被丢，于是同一个键在多数派和少数派各有一个版本。这笔账在愈合那一刻结清——代码故意在 `HEAL_AT_MS`、跑反熵合并**之前**就量一次分歧键数：

```typescript
// Measure consistency cost AT heal (before reconciliation) ...
clock.schedule(HEAL_AT_MS, () => {
  divergentAtHeal = countDivergentKeys(store);
});
```

为什么要在合并前量？因为合并跑完副本都收敛到 LWW 赢家了，事后再数分歧永远是 0——那会把整章要讲的代价藏起来。真正的账单是「愈合时两侧到底分歧了几个键」，那是一个正确的合并器**必须去 reconcile** 的工作量，也是 LWW 偷懒（有时偷错）的地方。

## 二维账单：把取舍量出来

同一脚本（seed=7，分区窗口 [200, 800]ms，12 次写）跑出来的真实输出：

```
集群 N=5 (多数派 n0,n1,n2 | 少数派 n3,n4), 写 quorum W=3
分区窗口 [200, 800]ms, 工作负载 12 次写, seed=7
二维代价表 (可用性 = 写成功率; 一致性 = 恢复后分歧键数):
strategy                             写成功 (avail)     被拒  恢复后分歧键 (consistency)
-----------------------------------  --------------  --  --------------------
CP (quorum write, minority refuses)  10/12 (83.3%)    2                     0
AP (both sides accept, LWW merge)    12/12 (100.0%)   0                     3
```

这张表就是 CAP 定理的实测形态，两行各占据光谱的一端：

- **CP**：写成功 10/12（83.3%），2 次写被拒（就是落在少数派 coordinator 上、分区期间凑不齐 quorum 的那些），换来愈合后 **0 个分歧键**。它花可用性买了一致性。
- **AP**：写成功 12/12（100%），一次都不拒，但愈合时有 **3 个分歧键**。它保住了可用性，账记在一致性上。

这些是 5 节点 toy 上的仿真数（绝对比例是脚本的产物——多少写落在哪侧、切口何时落下都是 scripted，换 seed 数字会变）。**能迁移的是相对形状**：CP 拿可用性换 0 分歧，AP 保可用性付分歧。不要把 83.3% / 3 当成真实系统的参数,它们是用来演示形状的。

值得停一秒：这两行不是「谁更好」，是「在分区这个不可避免的事件里，你的业务能容忍哪种痛」。订单扣款宁可拒写也不能两个版本（选 CP）；购物车「加入商品」宁可两边各加事后合并也不能让用户点了没反应（选 AP）。CAP 不替你选,它只告诉你：分区时，这两种痛你必挑一个。

## 失败模式：AP 的「合并很简单」是个谎

到这里 AP 看着不赖——12/12 可用，3 个分歧愈合时合并掉就好。问题在「怎么合并」。本章 AP 用的是最常见的天真答案：last-write-wins，按物理时间戳取大的。合并器只有这三行：

```typescript
function applyLww(kv: Map<string, VersionedValue>, key: string, incoming: VersionedValue): void {
  const cur = kv.get(key);
  if (!cur || incoming.stampMs > cur.stampMs) kv.set(key, incoming);
}
```

它信任 `stampMs`（coordinator 写入时的本地物理时钟）是真实时间的全序（total order，任意两个事件都能比出先后）。**它不是**。给少数派 coordinator 注入 -250ms 的时钟偏移（物理时钟落后真实时间，VM 被暂停过或没跑 NTP 的机器漂这么多很常见），一个真实更晚发生的写就会带着更小的时间戳，在合并里输给一个真实更早的写。`detectLostUpdate` 把这个 lost update（丢更新——一次成功的写被另一次静默覆盖）抓出来打印：

```
--- 失败模式: AP 的 last-write-wins 在时钟偏移下丢写 ---
少数派 coordinator 时钟偏移 -250ms (落后)。键 "k2" 真实最后一次写是 "k2=v11" (real t=923ms),
但 LWW 按物理时间戳合并后, 集群收敛到 "k2=v9" —— 真实更晚的写 "k2=v11" 被静默丢弃 (lost update)。
```

键 `k2` 真实的最后一次写是 `k2=v11`（真实虚拟时间 t=923ms），但因为它落在偏移 -250ms 的少数派 coordinator 上，戳出来的时间戳比早先 `k2=v9` 的还小，合并后集群收敛到 `k2=v9`。一次真实成功、真实更晚的写，**没有任何报错，悄悄消失了**。

这里有个比丢写本身更阴险的点，文件注释点破了：合并之后所有副本确实都收敛到 `k2=v9`，事后查分歧键是 0。**收敛 ≠ 正确（convergence is not correctness）**。系统看起来「一致了」，监控也绿，但它一致到了一个错误的值上。LWW-by-wall-clock 是分布式系统里经典的错误答案——它把「最终一致」做到了，却用一个不可信的全序毁了数据。

根因一句话：**物理时钟不是真实时间的全序**。两个节点的 wall clock 之间没有任何保证，偏移、漂移、回拨都合法。任何拿物理时间戳当因果顺序的合并都带着这个洞。

## 怎么补这个洞（第 7 章预告）

补法不是「把时钟调准」——NTP 也有几十毫秒误差，且永远存在窗口。正确的方向是**别用物理时间当全序，用因果关系（causality，谁基于谁产生）**。这正是第 7 章 CRDT（conflict-free replicated data type，无冲突复制数据类型）的活：用第 2 章的 vector clock（向量时钟）或 version vector 记录每个写「见过哪些写」，合并时按因果偏序而非 wall-clock 来定输赢——并发的写不会互相静默覆盖，而是被显式保留/合并。第 7 章的策略是「主动选 AP，然后用 CRDT 把这章暴露的烂摊子收拾干净」，让「可用性满格 + 收敛到正确值」同时成立。

不过要诚实：CRDT 不是银弹。它只对能定义出合法合并语义的数据类型有效（计数器、集合、LWW-register 升级版、序列），而像「银行账户余额必须 ≥ 0」这种需要全局不变量的约束，CRDT 给不了——那种你还是得回 CP。CAP 没消失，只是被推到了「哪些数据能 AP」的边界上。

## 一致性模型谱系：C 和 A 之间不是开关是滑块

前面把 CAP 讲成 C-vs-A 的二选一，是为了把定理讲清楚。但工程上「一致性」不是 0/1，是一条谱系。从强到弱：

- **线性一致（linearizable）**——CAP 的 C。每个操作看起来在调用与返回之间的某一瞬间原子生效，且全局尊重真实时间先后。读永远看到最新写。本章 CP 那侧（W+R>N 的 quorum）逼近的就是它。
- **顺序一致（sequential）**——存在一个所有节点都同意的操作全序，且每个进程自己的操作保持程序顺序，但**不要求**这个全序尊重真实时间。A 在 B 完成后才发起的写，全序里可能排 B 前面。比线性一致弱在「丢掉了 real-time 约束」。
- **因果一致（causal）**——只保证有因果关系的操作（写 X 后基于 X 写 Y）所有人看到的顺序一致；无因果关系的并发写，不同节点可以看到不同顺序。这是不牺牲可用性能拿到的最强模型，CRDT 和第 7 章走的就是这条线。
- **最终一致（eventual）**——只保证「不再有新写、等足够久，所有副本最终收敛到同一个值」。对中间过程零保证，对收敛到**哪个**值也常常零保证——本章 LWW 就是最终一致的一个实现，它收敛了，却收敛错了。

这条谱系解释了为什么「AP 牺牲一致性」这句话太粗。AP 牺牲的是线性一致；它完全可以守住因果一致（这就比 LWW 的最终一致强得多，也正是 LWW 丢写问题的修法）。**选 AP 不等于选最终一致，你还能在 AP 内部继续选一致性档位**——这是第 7 章能把烂摊子收拾干净的前提。

## PACELC：CAP 漏掉了平时 ⚡

CAP 有个大窟窿：它只描述分区**期间**的取舍，对「分区没发生」的 99.9% 时间一字不提。但工程上系统绝大多数时间是不分区的，那段时间的取舍同样真实——**强一致需要跨节点协调（多副本确认），协调要花延迟**。

PACELC 扩展把这补上，读法是：**if Partition, then C-or-A; Else, then Latency-or-Consistency**（分区时在 C/A 里选，否则在延迟/一致性里选）。它把单一的分区取舍升级成两段取舍，能更准地刻画真实系统的工程光谱：

- **Spanner（Google 的全球分布式 SQL）** 是 PC/EC：分区时选 C，平时也选 C——它用 TrueTime（带误差界的物理时钟 API）做外部一致性，代价是每次写要等过 TrueTime 的不确定窗口（commit-wait），**主动牺牲延迟换强一致**。本章那个「物理时钟不可信」的洞，Spanner 不是无视，是花钱买了带误差界的原子钟/GPS 时钟把窗口压到几毫秒再显式等过去。
- **Dynamo（亚马逊的 KV 存储）/ Cassandra** 是 PA/EL：分区时选 A，平时也选低延迟——可调的 R/W quorum，默认偏弱一致，用 vector clock + 应用层合并收拾冲突。本章 AP 那侧就是 Dynamo 谱系的简化版。

光用 CAP 你只能说「Spanner 是 CP、Dynamo 是 AP」，听起来它俩只在分区时有别。PACELC 才点出真正的日常差异：**Spanner 平时就慢（为一致性），Dynamo 平时就快（为可用/延迟）**——这个「平时」的差异，才是你 99.9% 时间真正体验到的东西。

标注一句这是前沿/开放问题：**怎么在不付 Spanner 那种专用硬件（原子钟/GPS）成本的前提下，逼近它那种「平时也强一致还不太慢」的点，目前没有通用解**。一类路线是混合逻辑时钟（HLC，把物理时钟和逻辑时钟揉一起，CockroachDB 在用）去掉 TrueTime 的硬件依赖，但它放松了 Spanner 的外部一致性保证；另一类是 deterministic database（Calvin 那条线，先定全序再执行）绕开协调延迟，但对工作负载形态有强假设。这条「PACELC 光谱上的右上角」——平时既快又强一致——仍在活跃研究中，没有一个方案能在通用工作负载、商用硬件、强一致三者上同时通吃。

---

把本章收束成一句可带走的判断：**CAP 不是让你三选二的选择题，是分区这个必然事件强加的二选一；C 特指线性一致，P 不可弃，所以真问题永远是「分区时拒写还是放弃强一致」。** 仿真把两条路的价签都打出来了——CP 付 2 次拒写换 0 分歧，AP 付 3 个分歧（外加 LWW 的静默丢写）换满格可用。下一章我们主动站到 AP 这边，用 CRDT 把 LWW 留下的丢写洞补上，看看「可用性满格 + 收敛到正确值」能不能同时拿到。
