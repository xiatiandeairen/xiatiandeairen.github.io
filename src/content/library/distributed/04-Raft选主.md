---
title: "Raft 选主：从零实现 Leader Election"
slug: "04"
collection: "distributed"
order: 4
summary: "全书的核心起点。在第 3 章故障检测的超时思想之上，手写 Raft 选主：term、RequestVote、随机化选举超时、心跳维持 leader。用实测证明随机化超时如何打破对称僵局——绕过第 0 章的 FLP 不是靠确定性算法，而是靠随机性买 liveness。本章只做选主；第 5 章接日志复制，二者合起来才是完整 Raft。"
topics:
  - "分布式系统"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

先说一个最容易被教程含糊过去的事实：**Raft 选主能选出唯一 leader，靠的不是某个聪明的确定性算法，而是往超时里掺随机数。** 第 0 章讲过 FLP 不可能性（在完全异步、允许一个节点崩溃的系统里，没有确定性算法能同时保证 safety 和 liveness）。Raft 的回答不是"我找到了 FLP 的漏洞"，而是"我永远保 safety，liveness 用随机化买"。本章配套代码 `examples/distributed-from-scratch/src/stage04-raft-election.ts` 里有一个对照实验直接量化这笔买卖：同一个 seed、同一张零抖动的网络，**固定超时烧掉 14 个 term 一个 leader 都选不出（活锁），随机化超时第 1 个 term、171ms 就收敛**。这一章就围绕这条对照展开。

本章的范围是诚实的：只做选主，不做日志复制。代码里 `RequestVote` 那个"候选人日志要足够新"的检查被省了——因为这一阶段所有节点的日志都是空的，检查恒为真。我在文件头注释里明确标了这一点，免得你把它的缺席当成 bug：

```ts
// Consequence: the RequestVote "candidate's log is at least as up-to-date"
// check is trivially true for everyone (empty logs), so it is omitted — noted
// here so a reader does not mistake its absence for a bug.
```

但这条被省掉的检查恰恰是 Raft 安全性的命门，本章 `## ✦` 那节会专门讲它为什么不能省、省了会丢什么。

> 全书参数都是 toy（故意调小好让 demo 几秒跑完），绝对 ms 偏乐观；能迁移到生产直觉的是**相对趋势**和**分布形状**，不是具体毫秒数。运行输出里那行 `注: 虚拟时间单位为模拟 ms; 绝对值随 toy 参数偏乐观` 就是这个意思。

## 三个角色、一个 term，先把状态机摆清楚

Raft 每个节点在任意时刻是三种角色之一：follower（跟随者，被动收心跳）、candidate（候选人，正在拉票）、leader（领导者，发心跳）。驱动角色转换的核心是 **term**（任期，一个单调递增的整数，相当于逻辑时钟的"届数"）。term 是 Raft 全部一致性论证的锚——几乎每条消息都带 term，每个收到消息的节点第一件事就是比 term。

代码把这三个角色和 term 收敛在一个类里（`stage04-raft-election.ts`），刻意只放协议本身，计时器/发送/崩溃这些管道继承自 `core/node.ts` 的基类 `Node`：

```ts
class RaftElectionNode extends Node {
  // 持久状态（崩溃后仍在，模拟 fsync 到磁盘）
  private get currentTerm(): number { ... }   // 存进基类 persistent Map
  private get votedFor(): string | null { ... }
  // 易失状态（崩溃即丢，onRestart 重建）
  private role: Role = "follower";
  private votesReceived = new Set<string>();
  private electionTimer = 0;
  private heartbeatTimer = 0;
}
```

这里第一个不能含糊的设计决策：**`currentTerm` 和 `votedFor` 必须持久化（落盘），`role` 这些可以易失。** 为什么？因为 Raft 安全性有一条铁律——"一个节点在一个 term 里最多投一票"。这条律靠 `votedFor` 记着"这个 term 我投给谁了"。如果 `votedFor` 是易失的，一个节点崩溃重启后会忘记自己投过票，在同一个 term 里再投一次——于是同一个 term 可能选出两个 leader，safety 当场破。代码注释把这个失败模式写死在字段定义旁：

```ts
// currentTerm and votedFor MUST be persistent: ... If they were volatile, a
// restarted node could re-vote in a term it already voted in, electing two
// leaders.
```

这就是为什么真实 Raft 实现里每次 `votedFor`/`currentTerm` 变化都得 `fsync` 才能回复 RPC——一笔实打实的延迟代价，买的是崩溃重启后不双投。基类 `Node` 用一个 `persistent` Map 模拟这块"落盘后能活过崩溃"的状态，`crash()` 保留它、`onRestart()` 从它重建。

### majority：唯一 leader 的全部数学

```ts
private get majority(): number {
  return Math.floor(this.clusterSize / 2) + 1;  // 5 节点 => 3
}
```

整个"一个 term 最多一个 leader"的安全性，浓缩在这个 `+1` 里。candidate 要当 leader 必须拿到**严格多数**票（含自己）。两个候选人想在同一个 term 都凑齐多数，他们的票池必然有重叠节点——而每个节点一个 term 只投一次，重叠不可能。所以同 term 双 leader 在数学上被排除。这不是工程技巧，是鸽巢原理。代码用一个不变量在**每个事件之后**校验它，下文 `## 安全性怎么验` 会展开。

## 选举怎么打：从 timeout 到 majority

健康集群里 follower 安安静静收 leader 的心跳，每收一次就重置自己的选举计时器。一旦在选举超时内没收到任何心跳，follower 就认定 leader 没了，发起选举：

```ts
private becomeCandidate(): void {
  this.currentTerm = this.currentTerm + 1;   // 开新 term
  this.role = "candidate";
  this.votedFor = this.id;                    // 投自己
  this.votesReceived = new Set([this.id]);    // 自票算进 majority
  for (const peer of this.peers) {
    this.sendTo(peer, "RequestVote", { term: this.currentTerm });
  }
  this.armElectionTimer();   // 重新武装一个随机超时：本轮没选出就更高 term 再来
}
```

收到 `RequestVote` 的节点怎么决定投不投，是协议里信息密度最高的一段：

```ts
private onRequestVote(candidate: string, p: VoteReqPayload): void {
  if (p.term < this.currentTerm) {            // 候选人 term 更旧 => 拒绝
    this.sendTo(candidate, "RequestVoteResp", { term: this.currentTerm, granted: false });
    return;
  }
  if (p.term > this.currentTerm) this.stepDown(p.term);  // 候选人更新 => 先降级再投
  const free = this.votedFor === null || this.votedFor === candidate;
  if (free) {
    this.votedFor = candidate;
    this.armElectionTimer();   // 投了票 = 听到了准 leader，重置计时器别去抢
  }
  this.sendTo(candidate, "RequestVoteResp", { term: this.currentTerm, granted: free });
}
```

三个点值得停下来看。其一，`p.term > this.currentTerm` 时**先 `stepDown` 再决定投票**——必须先采纳新 term 变回 follower，才有资格在这个新 term 里投票。其二，`free` 的判断允许"重复投给同一个候选人"（`votedFor === candidate`），这让投票在消息重复（第 3 章讲过 at-least-once 重试会让同一条消息被收多次）下幂等——重发的 RequestVote 不会让你投第二次也不会改主意。其三，**投票本身被当作"听到了合法的准 leader"，所以要重置自己的选举计时器**，否则刚投完票自己又超时去开一个更高 term 的选举，把刚帮的人废掉。

candidate 这边收票，用 Set 去重防止重复响应灌票：

```ts
private onVoteResponse(voter: string, p: VoteRespPayload): void {
  if (p.term > this.currentTerm) { this.stepDown(p.term); return; }  // 我已过时
  if (this.role !== "candidate" || p.term !== this.currentTerm) return;  // 旧选举的迟到响应
  if (!p.granted) return;
  this.votesReceived.add(voter);   // Set => 重复票不灌水
  if (this.votesReceived.size >= this.majority) this.becomeLeader();
}
```

整个协议能收敛，靠一条被集中到 `stepDown` 的规则——**任何带 term 比我大的消息，都说明我过时了：采纳它、降级 follower、忘掉这个新 term 投过谁**。把它收在一处，每个 handler 第一步调它就行。这是 Raft 比 Paxos 好读的关键之一：term 比较是唯一的"谁更权威"裁决器，没有别的隐藏状态。

### 选主成本实测：12 个 seed 全是 1 轮收敛

光说不练等于 survey。代码实验 A 跑了 12 个独立 seed（随机种子），量"选出第一个 leader 花了几个 term、几毫秒"。结果：

```
rounds(term) to first leader:  min=1  mean=1  max=1  (n=12)
virtual time to first leader:  min=172ms  mean=202.2ms  max=250ms
per-seed: seed1=T1@172ms  seed42=T1@199ms  seed256=T1@227ms
          seed2024=T1@250ms  seed65535=T1@233ms  ...
```

读这组数：**12 个 seed 全部在第 1 个 term 就选出了 leader（min=mean=max=1）**，没有一次 split-vote。时间落在 172–250ms 这个窗口，正好坐在选举超时区间 `[150, 300)ms` 的下半段——这符合直觉：超时抽到最小值的那个节点最先发起、最先拿到多数。每次运行还都带 `safe=true` 和 `dropped=0`，说明零丢包的乐观网络下，随机化几乎一抽就分出胜负。别把 200ms 当生产数字（toy 参数），要带走的是**分布形状**：随机化下"1 轮收敛"是常态而非运气。

## leader crash 与失败检测的复用

leader 维持权威靠周期性心跳：

```ts
const HEARTBEAT_MS = 50;       // 必须 << ELECTION_BASE_MS(150)
```

这个 `<<`（远小于）不是随便定的，是 Raft 调参里**最重要的一条规则**：心跳间隔必须远小于选举超时下界，否则一个健康 leader 的心跳还没到，follower 就先超时去开选举了——会有源源不断的虚假换届。代码把这条规则编码成了构造期约束。这也正是对第 3 章故障检测思想的复用：第 3 章讲"超时是异步网络里区分'慢'和'死'的唯一手段，且必然要在'误判存活'和'反应迟钝'之间权衡"。选举超时就是这个权衡的实例——调短了误判 leader 死（虚假选举），调长了 leader 真死后 failover 慢。

实验 B 直接量这个 failover（故障切换）代价：t=800ms 时把当时的 leader `n0` 崩掉，看多久选出新 leader：

```
old leader (crashed)    n0 @T1 t=800ms
new leader (recovered)  n2 @T2 t=1750ms
failover time           950ms
```

failover 950ms 这个数要拆开理解：它 = follower 等到选举超时（最长 ~300ms）+ 跑一轮选举（~200ms）+ 实验里探针每 50ms 轮询一次的采样粒度。生产 Raft 把超时调到秒级，failover 通常就是几百 ms 到几秒——**这段时间集群对外是不可写的**（没有 leader 没法提交），是 Raft 用强一致换来的可用性空窗。这个空窗是真实代价，不是 bug：第 6 章讲 CAP 权衡时会回到这里。

崩溃节点必须真的"死透"——不收消息、不触发计时器，否则崩溃测试会因为错误的原因通过。基类的不变量写得很硬：

```ts
// Invariant: a crashed node delivers NO messages and fires NO timers until
// restart(). The failure mode we guard against is a "crashed" node that keeps
// participating — that would make crash tests pass for the wrong reason.
```

## 网络分区：少数派选不出，多数派照常活

实验 C 把 5 节点切成少数派 `{n0,n1}` 和多数派 `{n2,n3,n4}`，这是分布式系统最经典的脑裂场景：

```
leaders on minority side (no quorum)      0
leaders on majority side                  1
leaders after heal (cluster-wide)         1
leader term after heal                   11
[msgs] sent=374 delivered=272 dropped=102 (loss=0 partition=102)
```

读这组数能看到 majority 法定人数机制的全部价值：**少数派那侧 0 个 leader**——`n0`/`n1` 不停超时、不停升 term 拉票，但永远凑不齐 3 票，所以一个 leader 都选不出（这正是为什么 term 愈合后高达 11：少数派在隔离期间空转烧了一堆 term）。**多数派那侧 1 个 leader**，照常工作。`dropped=102 (partition=102)` 是被分区吃掉的跨界消息。

关键在愈合之后：term 更高的那一侧会逼另一侧 `stepDown`，全集群收敛回 1 个 leader，**全程"任一 term 至多一个 leader"从未破**。这就是 Raft 在分区下的承诺——**少数派宁可不可用，也绝不产生第二个 leader**。它选了 CAP 里的 C 和 P，牺牲少数派的 A。注意：本章选主阶段空转烧 term 无害，但到了第 5 章有日志后，少数派那个空转升 term 的节点愈合后回来，会触发一个微妙的正确性问题——这正是 `## ✦` 那节要讲的"选举限制"存在的理由。

## ⚡ 失败模式：固定超时的 split-vote 活锁

现在到了本章的压轴对照——**如果不掺随机数会怎样**。实验 D 把所有节点的选举超时设成同一个固定值，并把网络抖动调到零（`defaultJitterMs: 0`），堵死"靠运气的延迟差异让某人侥幸先赢"这条后门，逼出最纯的病理：

```
=== D1.fixed-timeout-livelock (seed=7) ===
leader elected?              NO (livelock)
terms burned with no leader  14
[msgs] sent=524 delivered=520

=== D2.randomized-timeout-converges (seed=7) ===
leader elected?        yes
term of first leader   1
virtual time to elect  171ms
[msgs] sent=164 delivered=164
```

机理：所有节点用同一个超时值，于是**同时超时 → 同时变 candidate → 同时把票投给自己 → 每个 term 每人恰好 1 票 → 没人够 3 票的多数**。没人当选，全体再同时超时，升到下一个 term，重演。term 一路爬到 14 仍然 0 个 leader——这就是 **split-vote 活锁**（活锁：系统一直在动、一直在换 term，但永远不产出结果）。注意它发的消息更多（524 vs 164），是因为它白白打了 14 轮选举。

把随机化打开，同一个 seed、同一张零抖动网络，第 1 个 term、171ms 收敛。代码的结论行说得很直接：

```
=> 随机化把 "无限 split-vote" 变成 "1 轮内收敛".
   这就是 Raft 用随机性买 liveness 的代价与收益.
```

这正是 FLP 在工程里的真实样貌。FLP 说没有确定性算法能在异步下保证 liveness——固定超时就是那个确定性算法，它的活锁就是 FLP 定理在你眼前活生生发生。Raft 没有推翻 FLP，它绕开：**随机化让"所有人同时超时"的概率随时间趋于 0**。理论上随机化仍可能无限 split（每轮都恰好撞车），但那个概率每轮乘一个 <1 的因子，期望几轮内必收敛——12 个 seed 全 1 轮就是这个期望的实测。代价是 liveness 只是**概率性**的、没有硬上界；收益是实践中几乎总在常数轮收敛。

**⚡ 这是仍在研究的前沿**：FLP 的根本约束没有通用解。随机化（Raft）、故障检测器（Chandra-Toueg 的 ◇P，第 3 章碰过）、部分同步假设（Paxos 在 GST 后才保证活性）——都是绕道，没有一个能在纯异步下给出确定性的 liveness 保证。"如何在更弱的同步假设下拿到更强的活性保证"至今没有银弹。

## ✦ 命门：被本章省掉的"选举限制"

回到开头那个被省的检查。完整 Raft 的 `RequestVote` 还有一条：**只有日志至少和投票者一样新的候选人才能拿到票**（比较 `(lastLogTerm, lastLogIndex)`）。本章空日志，这条恒为真，所以省了。但它是 Raft 安全性的真正命门，**很多 Raft 实现的 bug 就出在这条没实现对**。

为什么不能省？想象没有这条限制：一个落后的 follower（缺了几条已提交日志）选举超时、发起选举、靠多数票当选 leader。它的日志缺了已提交的条目，而 Raft 规定 leader 的日志是权威，follower 要向 leader 看齐——于是这个新 leader 会用它残缺的日志**覆盖**掉其他节点上那些已经提交、已经返回给客户端"成功"的条目。**已提交的数据被丢了。** 这违反 Raft 最核心的 State Machine Safety（状态机安全性）：已提交的条目永不丢失、永不被覆盖。

选举限制怎么堵住它：候选人拉票时带上自己日志的 `(lastLogTerm, lastLogIndex)`，投票者只在"候选人的日志不比自己旧"时才投。因为一条日志要被提交必然已经在多数派上，而当选 leader 也要多数票，这两个多数必然有交集——交集里那个节点既见过这条已提交日志、又给候选人投了票，意味着候选人的日志不比它旧、也就含有这条日志。**于是缺已提交日志的候选人永远凑不齐多数，当不上 leader。** 已提交数据安全。

这就是为什么本章必须和第 5 章合起来才是完整 Raft：**选主决定"谁有权写"，选举限制决定"谁有资格被选"，而后者只有在有日志可比时才有意义**。第 5 章一旦引入真实日志，`onRequestVote` 里就要补回这段比较——那时它不再恒为真，会真正开始排除掉日志落后的候选人。把这两章分开讲，恰恰是为了让你看清：Raft 的安全性不在选主本身，而在"选主 + 日志新旧约束"的合谋。

## 安全性怎么验：每个事件后都查一遍

本章每个场景的输出都带 `safe=true` 和结尾的 `[safety] 从未被违反`，这不是嘴上保证。代码注册了一个不变量，在仿真**每个事件之后**都重新校验"任一 term 至多一个 leader"：

```ts
ctx.watch(invariant("at-most-one-leader-per-term", () => {
  for (const n of nodes) {
    if (n.isDown()) continue;       // 崩溃节点的残留 role 不算 leader
    if (n.getRole() === "leader") {
      leaderHistory.get(n.getTerm())?.add(n.id) ?? ...;
    }
  }
  for (const set of leaderHistory.values()) if (set.size > 1) return false;
  return true;
}, ...));
```

为什么是"每个事件之后"而不是"跑完再查一次"？因为同 term 双 leader 可能是一个**转瞬即逝**的中间态——某节点刚 `becomeLeader` 还没收到那条让它 `stepDown` 的高 term 消息。跑完再查会让这种瞬态溜走。逐事件校验才能抓到它，这是写并发/分布式仿真测试和写普通单测的根本区别：**你要断言的是"在任意时刻"成立的不变量，不是"最终"成立的结果**。这套逐事件不变量校验是全书共用的 harness，第 5 章验 State Machine Safety、第 7 章验 CRDT 收敛都靠它。

## 这一章你该带走什么

三件事。**一**，Raft 的唯一 leader 不靠确定性算法，靠 majority 的鸽巢原理保 safety、靠随机化超时买 liveness——固定超时活锁烧 14 个 term、随机化 1 轮 171ms 收敛，是 FLP 在工程里的真实样貌。**二**，`currentTerm`/`votedFor` 必须落盘、心跳必须 `<<` 选举超时、投票必须重置计时器，这三条任何一条错都会让选主坏，且坏法各不相同。**三**，本章最关键的内容恰恰是被它省掉的那条——选举限制，它是已提交日志不丢的命门，下一章引入日志后才补得回来。

最后一个工程判断，回应 `⚡` 钩子开头的问题：单节点 leader 是吞吐瓶颈（所有写都过它），前沿做法是打散 leader 热点——**multi-Raft**（如 TiKV 把 key 空间分成很多 region，每个 region 跑一个独立 Raft 组，leader 分散到不同物理节点）、或 **leaderless 协议**（如 EPaxos 让任意节点都能提交无冲突命令）。但工业界至今大量选 Raft 而非 Paxos/EPaxos，原因朴素得有点反直觉：**可理解性即可维护性**。Paxos 论文出了名地难懂，EPaxos 的冲突恢复路径更绕；Raft 把一致性拆成"选主 + 日志复制 + 安全性"三块各自能讲清的子问题，工程师能读懂、能调试、能在凌晨三点故障时推理它的行为——这个属性在生产系统里值钱过纸面上的吞吐峰值。这也正是本书把 Raft 当核心起点、且坚持手写到能跑的理由。下一章，我们给它接上日志复制（`stage05-raft-log-replication.ts`），让这套选主真正变成能跑银行账本的共识。
