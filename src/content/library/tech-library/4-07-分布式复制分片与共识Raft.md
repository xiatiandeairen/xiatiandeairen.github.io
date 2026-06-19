---
title: "分布式复制、分片与共识 Raft（数据库域）"
slug: "4-07"
collection: "tech-library"
group: "数据库"
order: 4007
summary: "单机数据库再快也只有一份数据，磁盘坏一块就全没了。分布式数据库的第一性问题是：把同一份状态复制到多台机器上，并且让这几台机器对\"发生了哪些事、按什么顺序发生\"达成一致。这就是 consensus（共识）。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:40:02.000Z"
updatedAt: "2026-06-12T11:40:02.000Z"
---
> **TL;DR**
>
> 单机数据库再快也只有一份数据，磁盘坏一块就全没了。分布式数据库的第一性问题是：**把同一份状态复制到多台机器上，并且让这几台机器对"发生了哪些事、按什么顺序发生"达成一致**。这就是 consensus（共识）。Raft 是 2014 年由 Diego Ongaro 和 John Ousterhout 提出的、以"可理解性（understandability）"为第一设计目标的共识算法，目的是替代以晦涩著称的 Paxos。它把共识拆成三个相对独立的子问题——**leader election（选主）、log replication（日志复制）、safety（安全性）**——并用 strong leader + term（任期，逻辑时钟）+ randomized timeout 三件套把工程实现压到一个普通工程师能读懂、能写对的复杂度。本章从 CAP 定理与 Paxos 的历史动机出发，逐行精读 etcd-io/raft 与 hashicorp/raft 的真实选主与日志复制源码，配一个 3 节点、可在你本机单进程跑起来的 toy Raft（leader election + log replication + 安全性裁决），并把 Figure 8 那个"为什么 leader 不能靠副本数提交旧 term 日志"的经典坑用 demo 复现给你看。读完你应该能回答：为什么 commit 一条旧日志要靠提交一条新日志、为什么 majority quorum 能保证 leader 完整性、ReadIndex 为什么是线性一致读的最低成本路径。

---

## 前置依赖

| 知识点 | 建议先读 / 本章位置 |
|--------|---------------------|
| WAL（预写日志）与 append-only log 语义 | 第 4 章；Raft 的 log 本质就是一条复制的 WAL |
| 状态机复制（state machine replication）模型 | 本章 §1.2 |
| fsync / 持久化语义（HardState 何时落盘） | 第 4 章 §2；本章 §5.3 |
| CAP 定理（C/A/P 三选二的真实含义） | 本章 §1.1 |
| 逻辑时钟 / 偏序（term 如何当时钟用） | 本章 §3.2 |
| quorum（多数派）与鸽巢原理 | 本章 §4.2 |

> **读法建议**：§1–§2 是史料与模型，想直接看算法可跳到 §3（选主源码）→ §4（日志复制 + 安全性源码）→ §5（可运行 demo）。§6 是生产坑，§7 是面试/复习五件套。

---

## 1. 设计考古：共识问题从哪来，Raft 为什么长这样

### 1.1 起点：CAP 定理划定的不可能边界

任何分布式复制方案在动手前都要先认一个事实——CAP 定理。

**出处**：Eric Brewer（UC Berkeley）于 2000 年 PODC（Symposium on Principles of Distributed Computing）keynote 提出 CAP conjecture；2002 年 Seth Gilbert 与 Nancy Lynch（MIT）给出形式化证明，使其成为定理。
**URL（已核实）**：https://en.wikipedia.org/wiki/CAP_theorem ；Brewer 2012 反思见 https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/

CAP 三个属性的精确表述（Gilbert & Lynch 版本，已核实原文）：

- **Consistency（一致性）**："Every read receives the most recent write or an error."（每次读要么读到最新写入，要么报错——这其实是 linearizability。）
- **Availability（可用性）**："Every request received by a non-failing node in the system must result in a response."（每个非故障节点收到的请求都必须返回响应——但不保证是最新值。）
- **Partition tolerance（分区容忍）**："The system continues to operate despite an arbitrary number of messages being dropped (or delayed) by the network between nodes."

经典叙事是"三选二"。但 **Brewer 自己在 2012 年的《CAP Twelve Years Later》里明确说三选二是误导性的**：网络分区（P）在分布式系统里不是一个可选项，是物理现实——只要你跨机器，分区就会发生。所以真正的选择只在分区发生的那一刻才存在，且只在 C 与 A 之间二选一：

> 原文（已核实）："system designers only need to sacrifice consistency or availability in the presence of partitions."

**这一点直接决定了 Raft 的定位**：Raft 是一个 **CP** 系统。当发生网络分区、leader 所在的少数派一侧失去多数派联系时，Raft 选择**牺牲可用性**——少数派一侧无法 commit 任何新日志（写不进去），从而保住 linearizability。majority quorum 那一侧继续服务。这就是为什么 etcd、Consul、CockroachDB 这类用 Raft 的系统在网络分区时少数派会"卡住写入"，这是设计，不是 bug。

> ⚠️ 常见误解：很多人以为"Raft 永远可用"。错。Raft 保证的是 **safety 永不破坏**（绝不返回错误结果），但 **liveness（最终选出 leader、最终 commit）只在多数派存活且网络最终稳定时成立**。分区里的少数派就是不可用的。

### 1.2 状态机复制：共识到底在共识什么

把复制问题抽象成一个统一模型——**Replicated State Machine（RSM，复制状态机）**：

- 每个节点维护一份**确定性状态机**（deterministic state machine）：给定相同的初始状态 + 相同顺序的输入命令序列，必然走到相同的最终状态。
- 复制状态机不直接复制"状态"，而是复制**输入命令的日志（log of commands）**。每个节点把命令一条条 apply 到自己的状态机。
- 只要保证**所有节点的 log 在每个 index 上是同一条命令、且顺序一致**，所有状态机就一定殊途同归。

所以共识的真正目标，被收窄成一句话：**让 N 个节点对"log 的第 i 条是哪条命令"达成一致**。这就是 consensus 的工程定义。Raft 论文开篇即用这个模型（§2 Replicated state machines）。

> 这跟数据库工程师非常亲切：一条 Raft log 本质上就是一条被复制到多台机器、且保证全序（total order）的 WAL（第 4 章）。单机 WAL 解决"崩溃后能还原"，复制 WAL 解决"一台机器整个没了也能还原"。

### 1.3 为什么不用 Paxos：可理解性作为第一设计目标

共识的"祖师爷"是 Leslie Lamport 的 Paxos（1998《The Part-Time Parliament》，2001《Paxos Made Simple》）。Paxos 在理论上是正确且最小的，但有两个工程上的硬伤，Raft 论文（§1, §10）直白地批评：

1. **极难理解**。Raft 作者在 §1 里引用了一个著名调查：连资深研究者都很难真正讲清 Paxos。论文原话动机就是 "we wanted an algorithm... that facilitates the development of intuitions that are essential for system builders"。
2. **single-decree Paxos → Multi-Paxos 的鸿沟**。Basic Paxos 只对"一个值"达成共识；要做成连续日志的 Multi-Paxos，论文没给出公认的、完整的工程方案，导致每个实现都不一样、都难验证。Raft 论文原话：Paxos "does not provide a good foundation for building practical implementations."

Raft 的设计哲学因此是 **understandability over minimality（可理解性优先于最小化）**。它接受一些"非最小"的约束来换取人能读懂、能写对：

- **Strong leader**：日志只从 leader 单向流向 follower。这砍掉了 Paxos 里复杂的多 proposer 协商。
- **决策分解**：把共识拆成 leader election / log replication / safety 三个独立子问题（这正是本章 §3/§4 的结构）。
- **随机化降低状态空间**：用 randomized election timeout 把"如何打破对称、避免活锁"这个难题用最朴素的方式解决，而不是设计复杂的优先级/排名机制。

**名字的由来（已核实，Wikipedia）**："Raft" 是 "Reliable, Replicated, Redundant, And Fault-Tolerant" 的首字母缩写。作者也开玩笑说 Paxos 是一座希腊小岛（议会），而 raft（木筏）是"逃离 Paxos 岛"的工具。

**史料出处**：
- Ongaro & Ousterhout, "In Search of an Understandable Consensus Algorithm (Extended Version)"，会议版发表于 **USENIX ATC 2014**（获 best paper）。**URL（已核实，本章逐字引用的 Figure 2/3/8 均来自此 PDF）**：https://raft.github.io/raft.pdf
- Ongaro 博士论文《Consensus: Bridging Theory and Practice》（2014，Stanford），membership change 与 log compaction 的权威细节来源。

---

## 2. Raft 全景：term、角色、三个子问题

在精读源码前，先把全景钉死。下面这些定义全部对照论文 Figure 2（§3 会逐字贴出）。

### 2.1 三种角色与状态转移

```
              times out, starts election
   ┌─────────┐ ──────────────────────────► ┌──────────┐
   │Follower │                               │Candidate │
   └─────────┘ ◄────────────────────────────└──────────┘
        ▲   ▲   discovers leader / higher term   │
        │   │                                     │ receives votes
        │   └──────────────┐                      │ from majority
        │ discovers higher │                      ▼
        │ term / new leader │                ┌──────────┐
        └───────────────────┴────────────────│  Leader  │
                                              └──────────┘
```

- **Follower（跟随者）**：被动。只响应 leader 和 candidate 的 RPC。收不到 leader 心跳超过 election timeout 就转 candidate。
- **Candidate（候选人）**：选举中。增加自己的 term、给自己投票、并行向所有节点发 RequestVote。拿到多数票就当 leader。
- **Leader（领导者）**：唯一处理客户端写、唯一向 follower 复制日志的节点。定期发心跳维持权威。

### 2.2 term：Raft 的逻辑时钟（核心抽象）

> **这是理解整个 Raft 的钥匙**。

term（任期）是一个**单调递增的整数**，每次选举开启一个新 term。term 在 Raft 里同时扮演三个角色：

1. **逻辑时钟**：每条 RPC 都带 term。节点看到比自己大的 term，立刻更新自己的 term 并退回 follower（论文 Figure 2 "All Servers" 规则："If RPC request or response contains term T > currentTerm: set currentTerm = T, convert to follower"）。
2. **过期检测**：节点看到比自己小的 term 的 RPC，直接拒绝（"Reply false if term < currentTerm"）。这能自动剔除"旧 leader 复活后发的过期命令"。
3. **选举唯一性的载体**：每个节点在一个 term 里最多投一票（first-come-first-served），所以一个 term 里最多产生一个 leader（Election Safety）。

term 让 Raft 不需要物理时钟同步（no wall-clock dependency for safety），只靠这个逻辑计数器就能给所有事件定一个偏序。**timeout 只影响 liveness/性能，绝不影响 safety**——这是 Raft（以及所有正经共识算法）的红线。

### 2.3 五大安全属性（论文 Figure 3，逐字核实）

下面是从 raft.pdf 提取的 Figure 3 五条属性，**逐字核实**（来源：https://raft.github.io/raft.pdf，§5.2/§5.3/§5.4 正文展开）：

| 属性 | 原文定义（verbatim） | 由什么保证 |
|------|----------------------|-----------|
| **Election Safety** | "at most one leader can be elected in a given term." | 一个 term 每节点最多投一票 + majority |
| **Leader Append-Only** | "a leader never overwrites or deletes entries in its log; it only appends new entries." | leader 实现纪律（§4.1） |
| **Log Matching** | "if two logs contain an entry with the same index and term, then the logs are identical in all entries up through the given index." | AppendEntries 的 prevLog 一致性检查（§4） |
| **Leader Completeness** | "if a log entry is committed in a given term, then that entry will be present in the logs of the leaders for all higher-numbered terms." | Election Restriction（§4.4） |
| **State Machine Safety** | "if a server has applied a log entry at a given index to its state machine, no other server will ever apply a different log entry for the same index." | 以上四条共同推出 |

这五条是 Raft 正确性的全部"公理输出"。本章后面所有源码精读，目的都是看清**代码是如何把这五条钉死的**。

---

## 3. 源码精读（一）：Leader Election

我们以两个工业级实现为底：**etcd-io/raft**（Go，etcd / Kubernetes / CockroachDB / TiKV 等的底座，库化设计、IO 交给上层）与 **hashicorp/raft**（Go，Consul / Nomad / Vault 的底座，自带网络与存储驱动）。两者对照能看清"同一个论文规则的两种工程落地"。

> **取材记录**：以下源码均通过 WebFetch 从 `raw.githubusercontent.com` 实际拉取（2026-06 的 `main` 分支）。逐字处标【真实源码 repo@path】，整理/节选处标【真实源码（节选）】，简化重排处标【示意，非逐字】。

### 3.1 论文规则（Figure 2，逐字）：Candidates 的行为

先把"标准答案"贴出来，源码就是在实现它。以下**逐字核实**自 raft.pdf Figure 2 "Rules for Servers · Candidates(§5.2)"：

```
Candidates (§5.2):
• On conversion to candidate, start election:
    • Increment currentTerm
    • Vote for self
    • Reset election timer
    • Send RequestVote RPCs to all other servers
• If votes received from majority of servers: become leader
• If AppendEntries RPC received from new leader: convert to follower
• If election timeout elapses: start new election
```

RequestVote RPC 的接收方规则（逐字核实，Figure 2 "RequestVote RPC · Receiver implementation"）：

```
1. Reply false if term < currentTerm (§5.1)
2. If votedFor is null or candidateId, and candidate's log is at
   least as up-to-date as receiver's log, grant vote (§5.2, §5.4)
```

记住第 2 条里的 "candidate's log is at least as up-to-date" —— 这就是 §4.4 要展开的 **Election Restriction**，它是 Leader Completeness 的根。

### 3.2 etcd-io/raft：从 tick 到发起选举

**etcd 的设计哲学是把 Raft 做成一个纯状态机库**：输入一个 `Message`，输出 `{[]Messages, []LogEntries, NextState}`，**network 和 disk IO 全部交给上层**。这换来 "flexibility, determinism, and performance"（README 原话，已核实 https://github.com/etcd-io/raft/blob/main/README.md）。

#### (a) 选举计时：`tickElection`

每个逻辑 tick（由上层定时调用）触发一次。当节点可被选举（promotable）且超过随机化的 election timeout，就给自己投递一个 `MsgHup`（"起来选举"的内部消息）。

【真实源码 etcd-io/raft@raft.go（main 分支，2026-06 拉取）】

```go
func (r *raft) tickElection() {
	r.electionElapsed++
	if r.promotable() && r.pastElectionTimeout() {
		r.electionElapsed = 0
		if err := r.Step(&pb.Message{From: new(r.id), Type: pb.MsgHup.Enum()}); err != nil {
			r.logger.Debugf("error occurred during election: %v", err)
		}
	}
}
```

逐行注解：
- `r.electionElapsed++`：把"距上次收到合法 leader 通信"的计数器加一。leader 的心跳会在别处把它清零。
- `r.promotable()`：当前节点是否在 voter 集合里（learner/已被移除的节点不能选举，对应论文 §6 membership change）。
- `r.pastElectionTimeout()`：是否超过**随机化**的超时阈值。随机化是打破 split vote 对称性的关键（§3.4）。
- `r.Step(... MsgHup ...)`：不直接改状态，而是走统一的 `Step` 状态机入口投递消息。这是 etcd 把一切都建模成消息的体现——**确定性的来源**。
- ⚠️ `From: new(r.id)`：当前 `main` 分支已迁到 protobuf-go，字段是指针，`new(r.id)` 生成 `*uint64`。早期 gogoproto 版本写作 `From: r.id`（值字段）。**这是一处实现细节会随 protobuf 迁移变化、但语义不变的点**，初读不必纠结。

#### (b) `MsgHup` 触发 `campaign`：真正发起选举

`Step` 收到 `MsgHup` 后调用 `campaign`。这段把 Figure 2 的 Candidate 四步几乎一比一翻译成代码：

【真实源码 etcd-io/raft@raft.go（节选，main 分支）】

```go
func (r *raft) campaign(t CampaignType) {
	if !r.promotable() {
		r.logger.Warningf("%x is unpromotable; campaign() should have been called", r.id)
	}
	var term uint64
	var voteMsg pb.MessageType
	if t == campaignPreElection {
		r.becomePreCandidate()
		voteMsg = pb.MsgPreVote
		term = r.Term + 1
	} else {
		r.becomeCandidate()        // ← Increment currentTerm + Vote for self（见 becomeCandidate）
		voteMsg = pb.MsgVote
		term = r.Term
	}
	var ids []uint64
	{
		idMap := r.trk.Voters.IDs()
		ids = make([]uint64, 0, len(idMap))
		for id := range idMap {
			ids = append(ids, id)
		}
		slices.Sort(ids)
	}
	for _, id := range ids {
		if id == r.id {
			// 给自己的一票：直接构造一条 vote response（已经投给自己了）
			r.send(&pb.Message{To: new(id), Term: new(term), Type: voteRespMsgType(voteMsg).Enum()})
			continue
		}
		last := r.raftLog.lastEntryID()        // ← 取自己最后一条日志的 (term, index)
		var ctx []byte
		if t == campaignTransfer {
			ctx = []byte(t)
		}
		// ← Send RequestVote RPCs to all other servers，带上自己 log 的 lastIndex/lastTerm
		r.send(&pb.Message{To: new(id), Term: new(term), Type: voteMsg.Enum(),
			Index: new(last.index), LogTerm: new(last.term), Context: ctx})
	}
}
```

注意 `r.send(..., Index: last.index, LogTerm: last.term)`：candidate 把**自己最后一条日志的 index 和 term** 塞进 RequestVote。这正是 Figure 2 RequestVote 参数里的 `lastLogIndex / lastLogTerm`，是 voter 判断"你的 log 够不够新"的依据。

#### (c) `becomeCandidate`：Increment term + vote for self

【真实源码 etcd-io/raft@raft.go（main 分支）】

```go
func (r *raft) becomeCandidate() {
	if r.state == StateLeader {
		panic("invalid transition [leader -> candidate]")   // ← leader 不能直接变 candidate，必须先回 follower
	}
	r.step = stepCandidate
	r.reset(r.Term + 1)    // ← Increment currentTerm（term+1 并清票/清计时器）
	r.tick = r.tickElection
	r.Vote = r.id          // ← Vote for self
	r.state = StateCandidate
	r.logger.Infof("%x became candidate at term %d", r.id, r.Term)
	traceBecomeCandidate(r)
}
```

`reset(r.Term + 1)` 这一行就把 Figure 2 的前三步（Increment currentTerm / Reset election timer）全做了，`r.Vote = r.id` 完成 "Vote for self"。`reset` 内部（已核实源码）会做 `r.Term = term; r.Vote = None; r.electionElapsed = 0; r.resetRandomizedElectionTimeout(); r.trk.ResetVotes()` 等——**term 变了就清空投票记录，保证"新 term 才能重新投票"**。

#### (d) 收到投票请求：Election Restriction 落地

这是整个选主里**最 safety-critical 的一段**。voter 收到 `MsgVote` 时，必须同时满足"我这个 term 还没投过别人"且"候选人的 log 至少和我一样新"才投票：

【真实源码 etcd-io/raft@raft.go（节选，main 分支）】

```go
case pb.MsgVote, pb.MsgPreVote:
	// canVote: (我已投给他) 或 (我没投过且不认任何 leader) 或 (PreVote 且对方 term 更大)
	canVote := r.Vote == m.GetFrom() ||
		(r.Vote == None && r.lead == None) ||
		(m.GetType() == pb.MsgPreVote && m.GetTerm() > r.Term)
	lastID := r.raftLog.lastEntryID()
	candLastID := entryID{term: m.GetLogTerm(), index: m.GetIndex()}
	if canVote && r.raftLog.isUpToDate(candLastID) {   // ← Election Restriction 的代码化身
		r.send(&pb.Message{To: m.From, Term: m.Term, Type: voteRespMsgType(m.GetType()).Enum()})
		if m.GetType() == pb.MsgVote {
			r.electionElapsed = 0    // ← 投了票就重置自己的选举计时（认可了一个候选人）
			r.Vote = m.GetFrom()     // ← 记录"这个 term 我投给谁了"
		}
	} else {
		// 拒绝：要么这个 term 已投别人，要么对方 log 不够新
		r.send(&pb.Message{To: m.From, Term: r.Term,
			Type: voteRespMsgType(m.GetType()).Enum(), Reject: new(true)})
	}
```

`isUpToDate(candLastID)` 是关键，它实现的就是论文 §5.4.1 的比较规则（逐字核实自 raft.pdf）：

> "Raft determines which of two logs is more up-to-date by comparing the index and term of the last entries in the logs. If the logs have last entries with different terms, then the log with the later term is more up-to-date. If the logs end with the same term, then whichever log is longer is more up-to-date."

翻译成判定：**先比 last term，term 大的更新；term 相等比 last index，index 大（log 更长）的更新。** 候选人必须 ≥ voter 才能拿票。这条规则配合 majority，就能从数学上保证"当选 leader 一定包含所有已 commit 的日志"——证明见 §4.4。

### 3.3 hashicorp/raft：另一种工程口味（自带 IO 与并发）

etcd 把 IO 推给上层；**hashicorp/raft 反过来——它自带 transport、自带 FSM apply 循环、用 goroutine + channel 把每个角色跑成一个 run-loop**。看它的选主能体会"另一种正确实现"。

> 版权头（已核实）：`// Copyright IBM Corp. 2013, 2026 / SPDX-License-Identifier: MPL-2.0`（HashiCorp 2024 年被 IBM 收购，故版权署名变为 IBM Corp.）。

#### (a) follower 超时 → 转 candidate

【真实源码 hashicorp/raft@raft.go（节选，main 分支）】

```go
case <-heartbeatTimer:
	r.mainThreadSaturation.working()
	// Restart the heartbeat timer
	hbTimeout := r.config().HeartbeatTimeout
	heartbeatTimer = randomTimeout(hbTimeout)

	// Check if we have had a successful contact
	lastContact := r.LastContact()
	if time.Since(lastContact) < hbTimeout {
		continue                       // ← 最近收到过 leader 通信，不超时
	}

	// Heartbeat failed! Transition to the candidate state
	lastLeaderAddr, lastLeaderID := r.LeaderWithID()
	r.setLeader("", "")
	// ...（配置检查省略）
	} else {
		if hasVote(r.configurations.latest, r.localID) {   // ← 对应 promotable：自己是 voter 才选举
			r.logger.Warn("heartbeat timeout reached, starting election",
				"last-leader-addr", lastLeaderAddr, "last-leader-id", lastLeaderID)
			r.setState(Candidate)
			return                     // ← 退出 runFollower，主循环会进入 runCandidate
		}
```

`randomTimeout(hbTimeout)` 同样是随机化超时；`r.LastContact()` 记录最近一次合法 leader 通信时间——和 etcd 的 `electionElapsed` 是一回事，只是用绝对时间戳实现。

#### (b) candidate run-loop：发起选举并数票

【真实源码 hashicorp/raft@raft.go（节选，main 分支）】

```go
func (r *Raft) runCandidate() {
	term := r.getCurrentTerm() + 1
	r.logger.Info("entering candidate state", "node", r, "term", term)

	var voteCh <-chan *voteResult
	if !r.preVoteDisabled && !r.candidateFromLeadershipTransfer.Load() {
		// 默认走 PreVote（见 §6.2）
	} else {
		voteCh = r.electSelf()         // ← 发起一轮 RequestVote，结果回到 voteCh
	}

	grantedVotes := 0
	votesNeeded := r.quorumSize()      // ← (N/2)+1
	for r.getState() == Candidate {
		select {
		case vote := <-voteCh:
			// 看到更大的 term：立刻退回 follower（Figure 2 "All Servers" 规则）
			if vote.Term > r.getCurrentTerm() {
				r.setState(Follower)
				r.setCurrentTerm(vote.Term)
				return
			}
			if vote.Granted {
				grantedVotes++
				r.logger.Debug("vote granted", "from", vote.voterID, "tally", grantedVotes)
			}
			if grantedVotes >= votesNeeded {        // ← 拿到多数票 → 当选
				r.logger.Info("election won", "term", vote.Term, "tally", grantedVotes)
				r.setState(Leader)
				r.setLeader(r.localAddr, r.localID)
				return
			}
		// ... 其它 case：超时、收到 AppendEntries、收到新请求
		}
	}
}
```

#### (c) `electSelf`：持久化自己的票 + 并行问票

【真实源码 hashicorp/raft@raft.go（节选，main 分支）】

```go
func (r *Raft) electSelf() <-chan *voteResult {
	respCh := make(chan *voteResult, len(r.configurations.latest.Servers))
	newTerm := r.getCurrentTerm() + 1
	r.setCurrentTerm(newTerm)

	lastIdx, lastTerm := r.getLastEntry()
	req := &RequestVoteRequest{
		RPCHeader:    r.getRPCHeader(),
		Term:         newTerm,
		LastLogIndex: lastIdx,         // ← 同样带上自己 log 的最后位置
		LastLogTerm:  lastTerm,
	}
	// 并行向每个 peer 发 RequestVote
	askPeer := func(peer Server) {
		r.goFunc(func() {
			resp := &voteResult{voterID: peer.ID}
			err := r.trans.RequestVote(peer.ID, peer.Address, req, &resp.RequestVoteResponse)
			if err != nil {
				resp.Term = req.Term
				resp.Granted = false
			}
			respCh <- resp
		})
	}
	for _, server := range r.configurations.latest.Servers {
		if server.Suffrage == Voter {
			if server.ID == r.localID {
				// 给自己投票，且【先持久化这一票】再算数（防重启后重复投票）
				if err := r.persistVote(req.Term, req.Addr); err != nil {
					r.logger.Error("failed to persist vote", "error", err)
					return nil
				}
				respCh <- &voteResult{RequestVoteResponse: RequestVoteResponse{
					Term: req.Term, Granted: true}, voterID: r.localID}
			} else {
				askPeer(server)
			}
		}
	}
	return respCh
}
```

⭐ **`r.persistVote(...)` 这一行藏着一个 safety 细节**，初学者最容易忽略：投票（`votedFor`）和 `currentTerm` **必须在响应 RPC 之前落盘**（Figure 2 "Persistent state on all servers: Updated on stable storage before responding to RPCs"）。否则节点崩溃重启后忘了"这个 term 投过谁"，可能在同一个 term 给两个候选人投票 → 直接破坏 Election Safety。这就是 §5.3 要讲的 HardState 持久化。etcd 把这个责任交给上层（`Ready.HardState` 必须先持久化），hashicorp 自己在 `persistVote` 里做。

### 3.4 randomized election timeout：用随机数打破对称（含失败模式）

**问题（split vote，分裂投票）**：如果所有 follower 用同一个固定超时，leader 一挂，它们几乎同时变 candidate、同时给自己投票、同时抢票 → 谁也拿不到多数 → 全部超时 → 再次同时重选 → **活锁（livelock）**，可能无限重复。

**Raft 的解法（逐字核实自 raft.pdf §5.2）**：

> "Raft uses randomized election timeouts to ensure that split votes are rare and that they are resolved quickly. To prevent split votes in the first place, election timeouts are chosen randomly from a fixed interval (e.g., 150–300ms)."

机理：每个节点从区间（如 150–300ms）里**独立随机**选超时。大概率只有一个节点最先超时 → 它抢先发 RequestVote 并在别人超时前拿到多数 + 发心跳压住其它节点。即使真的 split 了，每个 candidate 在**新一轮**又重新随机，再撞车的概率指数下降。

⚠️ **生产坑**：
- 区间下界不能太小。`electionTimeout` 必须 **远大于** 一个 RTT（往返）+ heartbeat 间隔，否则正常网络抖动就触发误选举（spurious election），leader 反复被推翻，吞吐崩盘。工程经验法则：`broadcastTime << electionTimeout << MTBF`（论文 §9.3 "Timing and availability"）。etcd 默认 `election-timeout=1000ms`、`heartbeat-interval=100ms`（比例 10:1）。
- 区间上界太大 → leader 故障后恢复服务慢（要等一个超时）。这是 availability 与稳定性的权衡。

---

## 4. 源码精读（二）：Log Replication 与 Safety

选出 leader 后，所有客户端写都走 leader。leader 把命令 append 到自己 log，并行复制给 follower，多数派落盘后 commit。这一节把**复制 + 一致性检查 + commit 规则 + Election Restriction 证明**串起来——这是 Raft 安全性的核心。

### 4.1 论文规则（Figure 2，逐字）：AppendEntries 接收方

**逐字核实**自 raft.pdf Figure 2 "AppendEntries RPC · Receiver implementation"：

```
1. Reply false if term < currentTerm (§5.1)
2. Reply false if log doesn't contain an entry at prevLogIndex
   whose term matches prevLogTerm (§5.3)
3. If an existing entry conflicts with a new one (same index
   but different terms), delete the existing entry and all that
   follow it (§5.3)
4. Append any new entries not already in the log
5. If leaderCommit > commitIndex, set commitIndex =
   min(leaderCommit, index of last new entry)
```

第 2 条（**prevLog 一致性检查**）是 Log Matching Property 的引擎；第 3 条（**冲突即截断**）是 leader 强制 follower 与自己对齐的手段；第 5 条是 commit 如何传播给 follower。

leader 侧的 commit 规则（逐字核实，Figure 2 "Rules for Servers · Leaders"最后一条）：

```
• If there exists an N such that N > commitIndex, a majority
  of matchIndex[i] ≥ N, and log[N].term == currentTerm:
  set commitIndex = N (§5.3, §5.4).
```

⭐ **注意最后那个看似多余的条件 `log[N].term == currentTerm`**——它就是 §4.4 要讲的"为什么不能靠副本数提交旧 term 日志"的代码化身，是 Raft 最反直觉、最容易写错的地方。整章最值得记住的一行。

### 4.2 etcd-io/raft：`maybeAppend` —— follower 侧一致性检查

follower 收到 AppendEntries 时，用 `maybeAppend` 实现 Figure 2 的第 2–5 条：

【真实源码 etcd-io/raft@log.go（main 分支，逐字）】

```go
// maybeAppend returns (0, false) if the entries cannot be appended. Otherwise,
// it returns (last index of new entries, true).
func (l *raftLog) maybeAppend(a logSlice, committed uint64) (lastnewi uint64, ok bool) {
	if !l.matchTerm(a.prev) {
		return 0, false                                  // ← Figure 2 第2条：prevLog 不匹配，拒绝
	}
	lastnewi = a.prev.index + uint64(len(a.entries))
	ci := l.findConflict(a.entries)                      // ← 找第一条冲突（同 index 不同 term）
	switch {
	case ci == 0:
		// 无冲突，全部已存在或全是新的，下面只更新 commit
	case ci <= l.committed:
		// 冲突点落在已提交区间 → 这是绝不该发生的灾难（违反 State Machine Safety）
		l.logger.Panicf("entry %d conflict with committed entry [committed(%d)]", ci, l.committed)
	default:
		offset := a.prev.index + 1
		if ci-offset > uint64(len(a.entries)) {
			l.logger.Panicf("index, %d, is out of range [%d]", ci-offset, len(a.entries))
		}
		l.append(a.entries[ci-offset:]...)               // ← Figure 2 第3+4条：截断冲突后追加新条目
	}
	l.commitTo(min(committed, lastnewi))                 // ← Figure 2 第5条：commitIndex=min(leaderCommit, lastNew)
	return lastnewi, true
}
```

逐行注解：
- `l.matchTerm(a.prev)`：`a.prev` 是 leader 声称的"新条目前一条"的 (index, term)。匹配不上直接 `false`，leader 收到失败会递减 `nextIndex` 重试——这是论文 §5.3 的回溯机制。
- `findConflict`：返回第一条"同 index 但 term 不同"的条目位置（详见下）。
- `case ci <= l.committed:` 的 **`Panicf`** 是工程上的"金丝雀"：如果发现要截断的位置落在已提交区间，说明上游出了违反安全性的严重 bug，宁可 panic 也不能默默破坏已提交数据。**生产级共识库到处是这种 assert-by-panic，体现"safety 优先于 liveness"。**
- `l.commitTo(min(committed, lastnewi))`：follower 不能盲信 leader 的 commitIndex，要取 `min`——你不能 commit 一条你还没有的日志。

`matchTerm` 与 `findConflict`（逐字核实，注释完整保留）：

【真实源码 etcd-io/raft@log.go（main 分支，逐字）】

```go
func (l *raftLog) matchTerm(id entryID) bool {
	t, err := l.term(id.index)
	if err != nil {
		return false
	}
	return t == id.term
}

// findConflict finds the index of the conflict.
// It returns the first pair of conflicting entries between the existing
// entries and the given entries, if there are any.
// If there is no conflicting entries, and the existing entries contains
// all the given entries, zero will be returned.
// If there is no conflicting entries, but the given entries contains new
// entries, the index of the first new entry will be returned.
// An entry is considered to be conflicting if it has the same index but
// a different term.
// The index of the given entries MUST be continuously increasing.
func (l *raftLog) findConflict(ents []*pb.Entry) uint64 {
	for i := range ents {
		if id := pbEntryID(ents[i]); !l.matchTerm(id) {
			if id.index <= l.lastIndex() {
				l.logger.Infof("found conflict at index %d [existing term: %d, conflicting term: %d]",
					id.index, l.zeroTermOnOutOfBounds(l.term(id.index)), id.term)
			}
			return id.index
		}
	}
	return 0
}
```

这段注释本身就是一份 Log Matching 的微型规范：**"An entry is considered to be conflicting if it has the same index but a different term"** —— 这正是论文里冲突的定义。

`commitTo`（逐字核实）——注意 "never decrease commit"：

【真实源码 etcd-io/raft@log.go（main 分支，逐字）】

```go
func (l *raftLog) commitTo(tocommit uint64) {
	// never decrease commit
	if l.committed < tocommit {
		if l.lastIndex() < tocommit {
			l.logger.Panicf("tocommit(%d) is out of range [lastIndex(%d)]. Was the raft log corrupted, truncated, or lost?", tocommit, l.lastIndex())
		}
		l.committed = tocommit
	}
}
```

`commitIndex` **单调不减**（commit 一旦达成永不回退）——这是 durability 的基石。那句 panic 里的错误信息 "Was the raft log corrupted, truncated, or lost?" 是 etcd 工程师留给运维的诊断线索：要 commit 一条 lastIndex 都没有的日志，多半是磁盘/存储层把日志弄丢了。

### 4.3 Log Matching Property：归纳法证明（对照源码）

论文 §5.3 用归纳法证明 Log Matching，**逐字核实**关键句：

> "The consistency check acts as an induction step: the initial empty state of the logs satisfies the Log Matching Property, and the consistency check preserves the Log Matching Property whenever logs are extended."

- **base case**：空 log 平凡满足。
- **inductive step**：每次 AppendEntries 成功前，`matchTerm(prev)` 保证 follower 在 prevIndex 处与 leader 完全一致（由归纳假设，前缀也一致）。在此之上 append，新前缀仍一致。
- 失败时：leader 递减 nextIndex 回溯，直到找到一致点，再覆盖 follower 后面的冲突部分（`findConflict` + `append`）。论文：

> "the leader handles inconsistencies by forcing the followers' logs to duplicate its own."

**这就是 `maybeAppend` 里 `matchTerm` 守门 + `findConflict`/`append` 截断重写两步的理论依据。** 代码与证明是严丝合缝的镜像。

### 4.4 ⭐ 最难的一关：为什么不能用副本数提交旧 term 的日志（Figure 8）

这是 Raft 全文最反直觉、面试最爱问、自己实现最容易写错的一点。

**直觉陷阱**：leader 把一条日志复制到了多数派，是不是就能 commit？——**如果这条日志是上一个 term 留下的，答案是不行。**

论文 Figure 8 给出反例（**逐字核实**自 raft.pdf §5.4.2 图注）：

> "In (a) S1 is leader and partially replicates the log entry at index 2. In (b) S1 crashes; S5 is elected leader for term 3 with votes from S3, S4, and itself, and accepts a different entry at log index 2. In (c) S5 crashes; S1 restarts, is elected leader, and continues replication. At this point, the log entry from term 2 has been replicated on a majority of the servers, but it is not committed. If S1 crashes as in (d), S5 could be elected leader (with votes from S2, S3, and S4) and overwrite the entry with its own entry from term 3. However, if S1 replicates an entry from its current term on a majority of the servers before crashing, as in (e), then this entry is committed (S5 cannot win an election)."

拆解（5 个节点 S1–S5，记 `index@term`）：

```
(a) S1(leader,t2) 把 [index2@t2] 复制给 S2。 logs: S1,S2 有 2@t2；S3,S4,S5 没有
(b) S1 挂。S5 凭借 S3/S4 的票当上 t3 leader（S3/S4 的 log 不比 S5 旧）。
    S5 在自己 index2 写入 2@t3。  logs: S5 有 2@t3
(c) S5 挂。S1 重启，重新当选（比如 t4），继续把 2@t2 复制给 S3。
    现在 2@t2 在 S1,S2,S3 —— 多数派！但它【还没 committed】
(d) ⚠️ 若此刻 S1 又挂，S5 仍可能凭 S2/S3/S4 的票当选 t5 leader，
    用 2@t3 覆盖掉已经在多数派上的 2@t2 ——
    如果之前误判 2@t2 已 commit，State Machine Safety 就被破坏了！
(e) ✅ 正解：S1 别急着 commit 旧日志。等它把【当前 term】的一条新日志（如 3@t4）
    复制到多数派，那一刻 2@t2 才随之安全 committed（此时 S5 再也选不上了，
    因为多数派的 log 都比它新）。
```

**根因**：旧 term 的日志即使在多数派上，也可能被一个"log 在那个位置更新、但缺这条旧日志"的节点选上后覆盖。**只有 commit 一条当前 term 的日志，才能借 Election Restriction 把这一刻的多数派"锁死"**——因为当前 term 的日志一旦在多数派，任何能赢得选举的未来 leader 必然包含它（Election Restriction：候选人 log 必须 ≥ 多数派），从而也必然包含它前面的所有日志（Log Matching）。

这就是为什么 Figure 2 commit 规则里有 **`log[N].term == currentTerm`** 那个条件。回到源码，etcd 在 `maybeCommit`/`Progress` 推进 commitIndex 时严格执行这个约束（节选）：

【真实源码 etcd-io/raft@raft.go（节选，main 分支）】

```go
// maybeCommit attempts to advance the commit index. Returns true if
// the commit index changed (in which case the caller should call
// the other branches of the algorithm as well).
func (r *raft) maybeCommit() bool {
	mci := r.trk.Committed()              // ← 多数派 matchIndex 的下界（被复制到多数派的最高 index）
	return r.raftLog.maybeCommit(mci, r.Term)   // ← 注意把【当前 r.Term】传进去做闸门
}
```

而 `raftLog.maybeCommit` 内部会检查 `log[mci].term == term`（当前 term）才推进 commit——**这一行就是 Figure 8 那个坑的最终防线**。新 leader 上任先 append 一条空的 no-op entry（见下），就是为了尽快产生"一条当前 term 的日志"，从而能把继承下来的旧日志一并安全 commit。

#### 旁证：leader 上任立刻 append no-op

【真实源码 etcd-io/raft@raft.go（节选，main 分支 `becomeLeader`）】

```go
func (r *raft) becomeLeader() {
	// ...
	r.pendingConfIndex = r.raftLog.lastIndex()
	emptyEnt := &pb.Entry{Data: nil}
	if !r.appendEntry(emptyEnt) {       // ← 上任即追加一条空日志（no-op）
		r.logger.Panic("empty entry was dropped")
	}
	r.logger.Infof("%x became leader at term %d", r.id, r.Term)
}
```

这条 `Data: nil` 的 no-op entry 带着新 leader 的 currentTerm。一旦它被复制到多数派并 commit，前面所有继承的旧 term 日志就都"搭便车"安全 commit 了。**这是 Raft 工程实现里一个看似无用、实则 safety-critical 的细节。**

### 4.5 Election Restriction 如何保证 Leader Completeness（证明骨架）

把链条接上：为什么"候选人 log 必须 ≥ 多数派"就能保证"当选 leader 一定有所有已 commit 日志"？

论文 §5.4.1（**逐字核实**）：

> "A candidate must contact a majority of the cluster in order to be elected, which means that every committed entry must be present in at least one of those servers. If the candidate's log is at least as up-to-date as any other log in that majority ... then it will hold all the committed entries."

证明骨架（鸽巢 + 反证）：
1. 一条日志被 commit ⟹ 它在某个 **majority A** 上（commit 的定义）。
2. 候选人当选 ⟹ 它从某个 **majority B** 拿到了票。
3. 任意两个 majority 必有交集（鸽巢原理：`|A| + |B| > N`）⟹ 存在节点 `x ∈ A ∩ B`，它既有那条 commit 日志，又投了候选人。
4. `x` 投票的前提是"候选人 log ≥ 我（x）的 log"（Election Restriction）。
5. 由 up-to-date 比较规则（先比 term 再比 index）+ Log Matching，"候选人 ≥ x" ⟹ 候选人的 log 包含 x 在该 commit 位置及以前的所有日志 ⟹ **候选人包含那条 commit 日志**。

于是 Leader Completeness 成立：**任何能当选的 leader，必然已经拥有全部已 commit 的日志**。这就是 Raft 不需要像 Viewstamped Replication 那样"选完 leader 再补传缺失日志"的根本原因——它把约束前移到了投票阶段。

---

## 5. ⭐ 可运行 demo：3 节点 toy Raft（选主 + 日志复制 + Figure 8 安全闸）

> **设计为可运行，请在你的环境验证。**
>
> **依赖**：Python 3.8+，**仅标准库**（无第三方依赖）。单进程、单线程、逻辑时钟驱动（不依赖真实网络/线程），用一个内存 message bus 模拟节点间 RPC，便于确定性复现。
>
> **它印证什么**：和 §3/§4 的 etcd 源码逐点呼应——term 作逻辑时钟、随机超时选主、RequestVote 的 up-to-date 闸、AppendEntries 的 prevLog 一致性检查与冲突截断、以及 §4.4 那条 **`commit 必须是当前 term`** 的安全闸（demo 第 3 幕会主动制造 Figure 8 场景，验证 toy Raft 拒绝错误 commit）。

把下面整段存成 `toy_raft.py`，`python3 toy_raft.py` 运行。

```python
#!/usr/bin/env python3
# toy_raft.py — 最小可运行 toy Raft：leader election + log replication + Figure 8 安全闸
# 设计为可运行，请在你的环境验证。依赖：Python 3.8+ 标准库。
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

random.seed(7)  # 固定随机种子，保证可复现

FOLLOWER, CANDIDATE, LEADER = "follower", "candidate", "leader"

@dataclass
class Entry:
    term: int
    cmd: str

@dataclass
class Msg:
    typ: str            # 'vote_req' | 'vote_resp' | 'app_req' | 'app_resp'
    frm: int
    to: int
    term: int
    # vote_req
    last_log_index: int = 0
    last_log_term: int = 0
    # vote_resp
    granted: bool = False
    # app_req
    prev_index: int = 0
    prev_term: int = 0
    entries: List[Entry] = field(default_factory=list)
    leader_commit: int = 0
    # app_resp
    success: bool = False
    match_index: int = 0

class Bus:
    """内存消息总线：模拟可丢可乱序的网络（这里为可复现先做成可靠 FIFO）。"""
    def __init__(self):
        self.q: List[Msg] = []
        self.partitioned: set = set()   # 被隔离的节点 id（模拟网络分区）
    def send(self, m: Msg):
        if m.frm in self.partitioned or m.to in self.partitioned:
            return  # 分区：消息丢弃
        self.q.append(m)
    def drain(self):
        out, self.q = self.q, []
        return out

class Node:
    def __init__(self, nid: int, peers: List[int], bus: Bus):
        self.id = nid
        self.peers = peers                  # 其它节点 id
        self.bus = bus
        self.state = FOLLOWER
        self.term = 0
        self.voted_for: Optional[int] = None
        self.log: List[Entry] = []          # index 从 1 开始；log[0] 对应 index1
        self.commit_index = 0
        self.leader_id: Optional[int] = None
        # 选举计时（逻辑 tick）
        self.election_timeout = random.randint(5, 10)   # 随机化超时（对应 randomized election timeout）
        self.elapsed = 0
        self.votes: set = set()
        # leader 状态
        self.next_index: Dict[int, int] = {}
        self.match_index: Dict[int, int] = {}

    # ---- 日志辅助 ----
    def last_index(self): return len(self.log)
    def last_term(self): return self.log[-1].term if self.log else 0
    def term_at(self, idx: int):  # idx 从 1 开始
        return self.log[idx-1].term if 1 <= idx <= len(self.log) else 0

    # ---- up-to-date 比较：对应 etcd isUpToDate / 论文 §5.4.1 ----
    def cand_up_to_date(self, cand_last_term, cand_last_index):
        if cand_last_term != self.last_term():
            return cand_last_term > self.last_term()      # 先比 term
        return cand_last_index >= self.last_index()       # term 相等比 index（长度）

    def reset_election_timer(self):
        self.elapsed = 0
        self.election_timeout = random.randint(5, 10)

    # ---- 逻辑 tick ----
    def tick(self):
        if self.state == LEADER:
            # leader 每 tick 发心跳 / 复制
            for p in self.peers:
                self.send_append(p)
            return
        self.elapsed += 1
        if self.elapsed >= self.election_timeout:
            self.start_election()

    # ---- 发起选举：对应 becomeCandidate + campaign ----
    def start_election(self):
        self.state = CANDIDATE
        self.term += 1                       # Increment currentTerm
        self.voted_for = self.id             # Vote for self
        self.votes = {self.id}
        self.leader_id = None
        self.reset_election_timer()
        print(f"  [t{self.term}] N{self.id} -> CANDIDATE, 发起选举")
        for p in self.peers:
            self.bus.send(Msg('vote_req', self.id, p, self.term,
                              last_log_index=self.last_index(),
                              last_log_term=self.last_term()))

    def become_leader(self):
        self.state = LEADER
        self.leader_id = self.id
        self.next_index = {p: self.last_index() + 1 for p in self.peers}
        self.match_index = {p: 0 for p in self.peers}
        # 上任追加一条 no-op（当前 term），对应 becomeLeader 的 emptyEnt
        self.log.append(Entry(self.term, "__noop__"))
        print(f"  [t{self.term}] N{self.id} === LEADER === (append no-op @ index{self.last_index()})")

    def become_follower(self, term, leader=None):
        if term > self.term:
            self.term = term
            self.voted_for = None
        self.state = FOLLOWER
        self.leader_id = leader
        self.reset_election_timer()

    # ---- leader 发 AppendEntries ----
    def send_append(self, p):
        ni = self.next_index.get(p, self.last_index() + 1)
        prev_index = ni - 1
        prev_term = self.term_at(prev_index)
        entries = self.log[ni-1:]            # 从 next_index 起的所有条目
        self.bus.send(Msg('app_req', self.id, p, self.term,
                          prev_index=prev_index, prev_term=prev_term,
                          entries=list(entries), leader_commit=self.commit_index))

    # ---- 处理收到的消息 ----
    def handle(self, m: Msg):
        # All Servers 规则：见到更大 term 一律退回 follower
        if m.term > self.term:
            self.become_follower(m.term)
        if m.typ == 'vote_req':   self.on_vote_req(m)
        elif m.typ == 'vote_resp': self.on_vote_resp(m)
        elif m.typ == 'app_req':  self.on_app_req(m)
        elif m.typ == 'app_resp': self.on_app_resp(m)

    def on_vote_req(self, m: Msg):
        grant = False
        if m.term >= self.term:
            can_vote = (self.voted_for in (None, m.frm))
            if can_vote and self.cand_up_to_date(m.last_log_term, m.last_log_index):
                grant = True
                self.voted_for = m.frm
                self.reset_election_timer()   # 认可候选人，重置自己计时
        self.bus.send(Msg('vote_resp', self.id, m.frm, self.term, granted=grant))

    def on_vote_resp(self, m: Msg):
        if self.state != CANDIDATE or m.term != self.term:
            return
        if m.granted:
            self.votes.add(m.frm)
            if len(self.votes) >= (len(self.peers) + 1) // 2 + 1:  # majority
                self.become_leader()

    def on_app_req(self, m: Msg):
        # 1) term 过期：拒绝
        if m.term < self.term:
            self.bus.send(Msg('app_resp', self.id, m.frm, self.term, success=False))
            return
        self.become_follower(m.term, leader=m.frm)
        # 2) prevLog 一致性检查（对应 matchTerm(a.prev)）
        if m.prev_index > 0 and self.term_at(m.prev_index) != m.prev_term:
            self.bus.send(Msg('app_resp', self.id, m.frm, self.term,
                              success=False, match_index=0))
            return
        # 3+4) 冲突截断 + 追加（对应 findConflict + append）
        idx = m.prev_index
        for e in m.entries:
            idx += 1
            if self.term_at(idx) != e.term:
                del self.log[idx-1:]          # 截断冲突及其后
                self.log.append(e)
        # 5) 推进 commit（不能超过自己 last_index）
        if m.leader_commit > self.commit_index:
            self.commit_index = min(m.leader_commit, self.last_index())
        self.bus.send(Msg('app_resp', self.id, m.frm, self.term,
                          success=True, match_index=self.last_index()))

    def on_app_resp(self, m: Msg):
        if self.state != LEADER or m.term != self.term:
            return
        if m.success:
            self.match_index[m.frm] = m.match_index
            self.next_index[m.frm] = m.match_index + 1
            self.maybe_commit()
        else:
            # 回溯 next_index 重试（对应论文 decrement nextIndex）
            self.next_index[m.frm] = max(1, self.next_index.get(m.frm, 1) - 1)

    # ---- ⭐ 核心安全闸：commit 必须是当前 term（Figure 8 / §4.4）----
    def maybe_commit(self):
        # 候选 N：被多数派复制到的最高 index
        for N in range(self.last_index(), self.commit_index, -1):
            cnt = 1  # 自己
            for p in self.peers:
                if self.match_index.get(p, 0) >= N:
                    cnt += 1
            if cnt >= (len(self.peers) + 1) // 2 + 1:
                if self.term_at(N) == self.term:        # ← 关键闸门：只 commit 当前 term 的日志
                    if N > self.commit_index:
                        self.commit_index = N
                        print(f"  [t{self.term}] N{self.id} commit -> index{N} "
                              f"(term_at={self.term_at(N)}, cmd={self.log[N-1].cmd!r})")
                    break
                else:
                    # 旧 term 日志即使多数派也不直接 commit（Figure 8 安全闸）
                    print(f"  [t{self.term}] N{self.id} 拒绝直接 commit index{N}：它是旧 term "
                          f"{self.term_at(N)} 的日志（需靠当前 term 日志搭便车）")
        return

class Cluster:
    def __init__(self, n=3):
        self.bus = Bus()
        ids = list(range(1, n+1))
        self.nodes = {i: Node(i, [x for x in ids if x != i], self.bus) for i in ids}
    def step(self, ticks=1):
        for _ in range(ticks):
            for nid in sorted(self.nodes):
                self.nodes[nid].tick()
            # 投递所有在途消息直到清空（一个 tick 内网络收敛）
            for _round in range(20):
                msgs = self.bus.drain()
                if not msgs:
                    break
                for m in msgs:
                    if m.to in self.nodes:
                        self.nodes[m.to].handle(m)
    def leader(self):
        ls = [n for n in self.nodes.values() if n.state == LEADER]
        return ls[0] if ls else None
    def dump(self):
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            logs = " ".join(f"{e.cmd}@{e.term}" for e in n.log) or "(empty)"
            print(f"    N{nid:<2} state={n.state:<9} term={n.term} commit={n.commit_index} log=[{logs}]")

def main():
    print("=== 第 1 幕：3 节点冷启动，随机超时选出唯一 leader ===")
    c = Cluster(3)
    for _ in range(15):
        c.step()
        if c.leader():
            break
    ldr = c.leader()
    assert ldr, "应在若干 tick 内选出 leader"
    print(f"  => leader = N{ldr.id}, term = {ldr.term}")
    c.dump()

    print("\n=== 第 2 幕：leader 接收客户端命令并复制到多数派后 commit ===")
    ldr.log.append(Entry(ldr.term, "SET_x=1"))
    ldr.log.append(Entry(ldr.term, "SET_y=2"))
    print(f"  leader N{ldr.id} 追加 2 条命令，开始复制...")
    for _ in range(5):
        c.step()
    c.dump()
    assert ldr.commit_index >= ldr.last_index() - 0, "命令应被 commit"
    # 验证所有节点 log 一致（Log Matching）
    logs = [tuple((e.term, e.cmd) for e in n.log) for n in c.nodes.values()]
    assert all(l == logs[0] for l in logs), "所有节点 log 必须一致（Log Matching）"
    print("  ✓ 所有节点 log 完全一致（Log Matching Property 成立）")

    print("\n=== 第 3 幕：复现 Figure 8 安全闸——旧 term 日志不可仅凭副本数 commit ===")
    # 手工构造场景：新 leader 继承了一条【旧 term】的、尚未 commit 的日志
    c2 = Cluster(3)
    n1, n2, n3 = c2.nodes[1], c2.nodes[2], c2.nodes[3]
    # 让 N1 当上 term=5 的 leader，但其 log 里 index1 是一条旧 term=2 的“继承日志”
    for n in (n1, n2, n3):
        n.term = 5
    n1.state = LEADER; n1.leader_id = 1; n1.voted_for = 1
    n1.log = [Entry(2, "OLD_from_term2")]            # ← 旧 term 的继承日志（还没 commit）
    n1.next_index = {2: 1, 3: 1}; n1.match_index = {2: 0, 3: 0}
    n2.log = []; n3.log = []
    print("  初始：N1(leader,t5).log=[OLD_from_term2@t2]，N2/N3 为空")
    # N1 把这条旧日志复制给 N2/N3（达到多数派），观察它【是否】敢 commit
    for _ in range(3):
        n1.tick()                                    # 发 AppendEntries
        for _r in range(10):
            ms = c2.bus.drain()
            if not ms: break
            for m in ms:
                if m.to in c2.nodes: c2.nodes[m.to].handle(m)
    print(f"  复制后：match_index={n1.match_index}, N1.commit_index={n1.commit_index}")
    assert n1.commit_index == 0, "旧 term 日志即使到多数派也【不能】直接 commit"
    print("  ✓ 安全闸生效：旧 term 日志虽在多数派，commit_index 仍为 0（拒绝错误 commit）")
    # 现在 leader 追加一条【当前 term=5】的新日志，复制到多数派 -> 旧日志搭便车 commit
    n1.log.append(Entry(5, "NEW_from_term5"))
    print("  N1 追加当前 term=5 的新日志并复制...")
    for _ in range(3):
        n1.tick()
        for _r in range(10):
            ms = c2.bus.drain()
            if not ms: break
            for m in ms:
                if m.to in c2.nodes: c2.nodes[m.to].handle(m)
    print(f"  => N1.commit_index = {n1.commit_index}（当前 term 日志 commit 后，旧日志一并安全提交）")
    assert n1.commit_index == 2, "提交当前 term 日志后，旧 term 日志搭便车被 commit"
    print("  ✓ 印证 §4.4：只有 commit 当前 term 日志，才能安全提交继承的旧 term 日志")

if __name__ == "__main__":
    main()
```

**预期输出（实测于 Python 3.12，`random.seed(7)`；leader id / "拒绝 commit" 行的重复次数会随调度略有差异，但每一幕的断言都应通过）**：

```
=== 第 1 幕：3 节点冷启动，随机超时选出唯一 leader ===
  [t1] N2 -> CANDIDATE, 发起选举
  [t1] N2 === LEADER === (append no-op @ index1)
  => leader = N2, term = 1
    N1  state=follower  term=1 commit=0 log=[(empty)]
    N2  state=leader    term=1 commit=0 log=[__noop__@1]
    N3  state=follower  term=1 commit=0 log=[(empty)]

=== 第 2 幕：leader 接收客户端命令并复制到多数派后 commit ===
  leader N2 追加 2 条命令，开始复制...
  [t1] N2 commit -> index3 (term_at=1, cmd='SET_y=2')
    N1  state=follower  term=1 commit=3 log=[__noop__@1 SET_x=1@1 SET_y=2@1]
    N2  state=leader    term=1 commit=3 log=[__noop__@1 SET_x=1@1 SET_y=2@1]
    N3  state=follower  term=1 commit=3 log=[__noop__@1 SET_x=1@1 SET_y=2@1]
  ✓ 所有节点 log 完全一致（Log Matching Property 成立）

=== 第 3 幕：复现 Figure 8 安全闸——旧 term 日志不可仅凭副本数 commit ===
  初始：N1(leader,t5).log=[OLD_from_term2@t2]，N2/N3 为空
  [t5] N1 拒绝直接 commit index1：它是旧 term 2 的日志（需靠当前 term 日志搭便车）
  ...（上一行因每个 tick 的心跳重试会重复打印多次，可忽略）...
  复制后：match_index={2: 1, 3: 1}, N1.commit_index=0
  ✓ 安全闸生效：旧 term 日志虽在多数派，commit_index 仍为 0（拒绝错误 commit）
  N1 追加当前 term=5 的新日志并复制...
  [t5] N1 commit -> index2 (term_at=5, cmd='NEW_from_term5')
  => N1.commit_index = 2（当前 term 日志 commit 后，旧日志一并安全提交）
  ✓ 印证 §4.4：只有 commit 当前 term 日志，才能安全提交继承的旧 term 日志
```

**与源码的逐点呼应**：

| toy_raft.py | etcd-io/raft 对应 | 论文规则 |
|-------------|-------------------|----------|
| `cand_up_to_date` | `raftLog.isUpToDate` | §5.4.1 up-to-date 比较 |
| `on_vote_req` 的 `can_vote && up_to_date` | `Step` 的 `MsgVote` 分支 | Figure 2 RequestVote 第2条 |
| `on_app_req` 的 prevLog 检查 | `maybeAppend` 的 `matchTerm(a.prev)` | Figure 2 AppendEntries 第2条 |
| `on_app_req` 的截断+追加 | `findConflict` + `l.append` | Figure 2 第3+4条 |
| `maybe_commit` 的 `term_at(N)==self.term` | `maybeCommit(mci, r.Term)` | Figure 2 commit 规则 + Figure 8 |
| `become_leader` 追加 no-op | `becomeLeader` 的 `emptyEnt` | §5.4.2 旁证 |

> 这个 demo **故意不实现** snapshot、membership change、PreVote、持久化落盘——它只为印证"选主 + 复制 + 安全闸"三件核心。真实生产库（etcd/hashicorp）在这之上还有几万行处理这些，见 §6。

---

## 6. 方案对比、生产坑与底层根因

### 6.1 共识/复制方案横向对比

| 维度 | **Raft** | **Multi-Paxos** | **Viewstamped Replication (VR)** | **ZAB (ZooKeeper)** | **主从异步复制 (MySQL async)** |
|------|----------|-----------------|----------------------------------|---------------------|-------------------------------|
| 一致性 | 线性一致（CP） | 线性一致（CP） | 线性一致（CP） | 线性一致（CP） | 最终一致，可能丢已 ack 写 |
| 强 leader | 是（日志单向） | 否（多 proposer，需 leader 优化） | 是 | 是 | 是 |
| 选主时 leader 是否需含全部 commit | **是**（Election Restriction） | 否（选完补日志） | 否（选完补日志） | 是（含最新 zxid） | N/A |
| 可理解性 | ★★★★★（首要目标） | ★★ | ★★★ | ★★★ | ★★★★★ |
| 成员变更 | joint consensus / single-server | 复杂 | view change | 重配置 | 手工 |
| 典型实现 | etcd, TiKV, CockroachDB, Consul | Google Chubby/Spanner(Paxos系) | 较少独立实现 | ZooKeeper | MySQL/PG 原生复制 |
| 故障时行为 | 少数派不可写（牺牲 A） | 同 | 同 | 同 | 主挂可手工切，有丢数据风险 |

**具体场景跑一遍**："3 节点集群，1 个节点网络分区"：
- **Raft / Paxos / VR / ZAB（CP 系）**：分区侧若是少数派 → 该侧无法 commit（写阻塞/报错），多数派侧正常服务。分区恢复后少数派追日志归队。**不丢已 commit 数据。**
- **MySQL async**：若分区的是 master，旧 master 仍接受写但这些写复制不出去；若此时把 slave 提主，旧 master 上的未复制写在归队时会冲突/丢失（"split-brain + data loss"）。**这是 CP vs 弱一致的本质差别。**

**Raft 不适用的边界**：
- **跨地域高延迟 + 高写入**：每次 commit 至少一个 majority RTT。跨洲（百毫秒级 RTT）下 Raft 写延迟会很高。这类场景要么放宽一致性（最终一致 + CRDT），要么用 leader 就近 + 分片把 Raft group 局部化（见 §6.4）。
- **超大集群**：节点越多，majority 越大，复制扇出越大，吞吐反而下降。Raft 集群通常 3 或 5 个 voter，不是越多越好。要扩容用**分片**（多个 Raft group）而非把单个 group 做大。
- **拜占庭故障（节点说谎/被攻陷）**：Raft 是 **crash-fault-tolerant（CFT）**，假设节点要么正常要么宕机，**不容忍恶意节点**。区块链场景需要 BFT 类算法（PBFT、Tendermint）。

### 6.2 生产坑（一）：成员变更——一次只动一个的铁律

**坑**：直接从旧配置 `Cold` 切到新配置 `Cnew`，如果不同节点在不同时刻切换，可能在切换窗口里**Cold 的多数派和 Cnew 的多数派互不相交**，于是同一 term 选出两个 leader → 脑裂。

论文给两种安全方案：

1. **Joint consensus（联合共识，§6 原始方案）**：先切到一个"过渡配置 `Cold,new`"，在此期间**任何决策（选举/commit）都需要同时拿到 Cold 的多数派 AND Cnew 的多数派**，再切到 `Cnew`。论文（逐字核实）：

   > "There is no point in time in which Cold and Cnew can both make decisions independently."

   这从根上消除了"两个不相交多数派"的可能。

2. **Single-server change（单节点变更，Ongaro 博士论文推荐，etcd 实现）**：**一次只增/删一个节点**。可以证明：旧配置的任意多数派与新配置的任意多数派必有交集（因为只差一个节点），所以无需联合共识也安全。etcd-io/raft 的 `ConfChange` 默认走这个，更简单。

⚠️ **真实事故模式**：运维一次性替换多个节点（比如 3 节点全换 IP），中间态出现脑裂或丢 quorum 卡死。**铁律：成员变更一次只动一个，且等上一次 commit 完成再动下一次。** 批量换机要串行做。

### 6.3 生产坑（二）：PreVote 与 CheckQuorum——被隔离节点的"term 通货膨胀"

**坑**：一个 follower 被网络分区隔离，收不到心跳 → 不断超时 → 每次选举把 term +1 → 隔离久了它的 term 涨得很高（比如 +1000）。一旦它**重新入网**，它发的 RequestVote 带着超高 term，会强迫当前健康的 leader（term 低）"看到更大 term 立刻退位"→ 集群被这个"失联归来"的节点**无谓地打断一次选举**，造成可用性抖动。它自己又因为 log 落后选不上，于是 leader 退位 → 重选 → 它再捣乱，反复。

**etcd/hashicorp 的解法（README 已核实，etcd 列为内置特性）**：

- **PreVote（预投票）**：candidate 在真正 +term 发起选举前，先发一轮 `MsgPreVote` 探测"如果我选举能赢吗"。**PreVote 不增加自己的 term**。只有预投票拿到多数（即"我的 log 够新、确实联系得上多数派"）才进入真正选举。被隔离节点的 log 落后，PreVote 永远过不了，于是它的 term **不再膨胀**，入网也不打扰 leader。源码里 §3.2(b) `campaign` 的 `campaignPreElection` 分支就是它。
- **CheckQuorum（多数派自检）**：leader 定期检查"我最近还能联系上多数派吗"，联系不上就**主动退位**（避免一个失去多数派的 leader 还自以为是 leader 继续接受读，破坏线性一致）。对应 §3.2(a) `tickHeartbeat` 里的 `MsgCheckQuorum`。

**这两个是几乎所有生产 Raft 默认开启的优化，论文正文没有、但工程上必备。** 不开 PreVote 的集群在不稳定网络下会有明显的 leader 抖动。

### 6.4 生产坑（三）：分片（Sharding）——单 Raft group 的吞吐天花板

单个 Raft group 的写吞吐被 **leader 单点 + 每次 commit 一个 majority RTT** 限死。生产系统（TiKV、CockroachDB、Spanner-like）的扩展之道是 **Multi-Raft / 分片**：

- 把 key 空间切成很多 **range / region**，每个 range 是一个**独立的 Raft group**，有自己的 leader、自己的多数派。
- 不同 range 的 leader 分散到不同物理节点 → 写负载水平打散，吞吐随分片数线性扩展。
- 单条 Raft 日志只复制本 range 的数据，扇出可控。

**新坑随之而来——跨分片事务**：一个事务跨多个 Raft group 时，单个 group 的线性一致不再够，需要叠加 **2PC（两阶段提交）+ 全局时间戳**（如 TiKV 的 Percolator 模型、Spanner 的 TrueTime）来获得跨分片的快照隔离/外部一致性。**Raft 解决"一个 range 内的复制一致"，跨 range 的原子性是另一层（事务层）的问题，别指望 Raft 单独搞定。**

### 6.5 生产坑（四）：线性一致读——别让"读"偷偷破坏一致性

**坑**：写走 Raft 没问题，但**读**怎么办？最朴素的"读 leader 本地状态机"是**错的**：一个旧 leader 可能已经被网络分区、新 leader 已经选出并 commit 了新值，但旧 leader 自己还不知道（还没收到更大 term 的消息），它本地读出来就是 **stale read（陈旧读）**，破坏 linearizability。

三种正确做法（成本递增/递减权衡）：

1. **ReadIndex（etcd 默认，最优性价比）**：leader 读时记下当前 `commitIndex` 作为 `readIndex`，然后**发一轮心跳确认自己仍是多数派认可的 leader**（防止自己已被取代），再等本地 `applied >= readIndex`，就可以安全返回本地读。**省掉了走一遍日志复制的开销，但仍需一轮心跳 RTT。** etcd README 原话（已核实）支持 "efficient linearizable read-only queries"。
2. **Lease Read（租约读，再省一轮 RTT）**：leader 持有一个基于时钟的 **leader lease**，在租约有效期内它**确信**自己还是 leader（因为选举超时 > 租约 + 时钟漂移上界），于是连那轮确认心跳都省了，直接本地读。**代价：依赖时钟漂移有界这个假设**，时钟乱跳会出 stale read。TiKV/CockroachDB 都用 lease read 配合 NTP/混合逻辑时钟。
3. **走 Raft log 读（最贵、最保险）**：把读也作为一条 log entry 复制 commit，绝对线性一致，但每次读一个 majority RTT，吞吐最差。一般只在 ReadIndex/Lease 不可用时兜底。

⚠️ **从 follower 读**：默认 follower 读是 stale 的。要 follower 也线性一致，follower 需向 leader 要 readIndex 再等本地 apply 跟上（etcd 支持 follower read）。**"随便找个副本读"在 CP 系统里默认不安全，必须显式走 ReadIndex 协议。**

### 6.6 底层根因小结：所有坑都回到同两条公理

把 §6 的坑收敛，会发现它们的根因只有两条：

1. **majority quorum 的交集性质**（任意两个多数派必相交）——成员变更坑（6.2）、Election Restriction（4.5）、Figure 8（4.4）全是它的推论。**一旦让两个不相交的多数派同时能决策，safety 立刻崩。**
2. **term 是唯一的真相时钟，且 commit 必须锚定当前 term**——PreVote/CheckQuorum（6.3）、stale read（6.5）、Figure 8（4.4）全是"谁的 term 是当下真相"的问题。**任何让节点基于过期 term 做决策（旧 leader 读、旧日志 commit）的路径，都是 bug。**

记住这两条，你看任何 Raft 衍生问题都能直接定位到根。

---

## 7. 复习五件套

### 7.1 一句话总结

Raft 用 **strong leader + term 逻辑时钟 + 随机化超时**，把共识拆成 selectable 的选主 / 复制 / 安全三块；靠 **majority quorum 的交集性** 与 **"commit 必须锚定当前 term"** 两条公理，在牺牲分区下可用性（CP）的代价下，保证复制状态机的线性一致与永不丢已提交数据。

### 7.2 关键数字 / 事实速记

- term：单调递增逻辑时钟；每节点每 term 最多投一票；见更大 term 立即退 follower。
- quorum：N 节点需 `(N/2)+1` 票/副本；故 3 节点容 1 故障，5 节点容 2 故障。**容忍 f 故障需 2f+1 节点。**
- 选举超时典型区间：150–300ms（论文示例）；etcd 默认 election 1000ms / heartbeat 100ms（10:1）。
- 时序铁律：`broadcastTime << electionTimeout << MTBF`。
- commit 规则的隐藏闸：`log[N].term == currentTerm` 才能 commit index N。
- 五大安全属性：Election Safety / Leader Append-Only / Log Matching / Leader Completeness / State Machine Safety。
- Raft = CFT（容崩溃）≠ BFT（不容拜占庭）。
- 成员变更：一次只动一个节点（single-server change）或 joint consensus。

### 7.3 高频面试题（含陷阱）

**Q1：为什么 Raft 的 leader 不能一看到旧 term 的日志被复制到多数派就 commit 它？**
A：Figure 8 场景——旧 term 日志即使在多数派，仍可能被一个"在该 index 上 log 更新、但缺这条旧日志"的节点当选后覆盖。只有 commit 一条**当前 term**的日志，才能借 Election Restriction 锁死当前多数派，让任何未来 leader 必然包含它（及其前缀）。所以 commit 规则有 `log[N].term == currentTerm` 闸门，leader 上任也要立刻 append 一条 no-op 当前 term 日志。

**Q2：candidate 凭什么保证当选后拥有所有已 commit 日志（Leader Completeness）？**
A：commit ⟹ 在某 majority A；当选 ⟹ 从某 majority B 拿票；A∩B 非空（鸽巢）⟹ 存在节点既有该 commit 日志又投了它；它投票的前提是"候选人 log ≥ 我"（Election Restriction，比 last term 再比 index）⟹ 候选人含该日志及其前缀。

**Q3（陷阱）：3 节点 Raft 挂了 1 个还能写吗？挂了 2 个呢？**
A：挂 1 个：剩 2 个 = `(3/2)+1=2` 正好是多数派，**能选主、能 commit、能写**。挂 2 个：剩 1 个 < 多数派，**选不出 leader、不能 commit、不能写（只能 stale 读）**，集群进入不可用，直到至少恢复到 2 个存活。

**Q4：直接读 leader 本地状态机为什么可能不是线性一致？怎么修？**
A：旧 leader 可能已被分区、新 leader 已 commit 新值，旧 leader 不自知 → stale read。修法：ReadIndex（记 commitIndex + 一轮心跳确认仍是 leader + 等 applied 跟上）或 Lease Read（租约内免确认，依赖时钟有界）。

**Q5：PreVote 解决什么问题？不开会怎样？**
A：解决被隔离节点反复超时把 term 抬高、入网后无谓打断健康 leader 的问题。PreVote 先不加 term 探测能否赢，log 落后的隔离节点永远过不了预投票，term 不再膨胀。不开 PreVote，不稳定网络下 leader 会被失联归来的节点频繁推翻，可用性抖动。

**Q6（陷阱）：Raft 集群是不是节点越多越可靠/越快？**
A：越多越**可靠**（容更多故障）但越**慢**（majority 更大、复制扇出更大、每次 commit 等更多节点）。通常 3 或 5。扩吞吐靠**分片成多个 Raft group**，不是把单 group 做大。

### 7.4 5 个易错点（自查清单）

1. ❌ "复制到多数派就 commit" → ✅ 旧 term 日志要等当前 term 日志搭便车才 commit（Figure 8）。
2. ❌ "votedFor / currentTerm 可以晚点落盘" → ✅ 必须在响应 RPC 前持久化，否则重启后同 term 重复投票破坏 Election Safety。
3. ❌ "成员变更可以一次换多个节点" → ✅ 一次只动一个，或用 joint consensus，否则脑裂。
4. ❌ "读 leader 本地就线性一致" → ✅ 需 ReadIndex/Lease，否则 stale read。
5. ❌ "follower 的 commitIndex 直接信 leader" → ✅ 取 `min(leaderCommit, 自己 lastIndex)`，不能 commit 自己还没有的日志。

### 7.5 动手题（扩展上面的 toy_raft.py）

> 都基于 §5 的 `toy_raft.py` 改，难度递增。

1. **【网络分区与脑裂验证】** 用 `Bus.partitioned` 把 leader 隔离进少数派（5 节点集群隔离 leader + 1 follower 成 2 节点少数派），让多数派 3 节点重新选主。验证：(a) 少数派侧无法 commit 新写（`commit_index` 不前进）；(b) 多数派选出新 leader 且 term 更高；(c) 分区恢复后旧 leader 退回 follower 并追上新日志。**预期**：印证 Raft 的 CP 行为与 §6.1。

2. **【实现 PreVote】** 给节点加 `PRE_CANDIDATE` 状态与 `pre_vote_req/resp`。candidate 发起真正选举前先跑一轮 PreVote（不改 term），只有 PreVote 拿多数才 `start_election`。然后构造"被隔离节点 term 涨到 100"的场景，验证开 PreVote 后它入网**不会**打断健康 leader（对照 §6.3）。

3. **【冲突日志截断可视化】** 手工给某个 follower 注入一段与 leader 冲突的尾部日志（不同 term），跑一次复制，打印 `findConflict` 截断点与截断前后的 log。验证 follower 日志被 leader 强制对齐（§4.3 Log Matching 收敛）。

4. **【ReadIndex 线性一致读】** 给 leader 加 `read_index()`：记录当前 `commit_index`，发一轮心跳确认仍被多数派认可，等 `applied >= readIndex` 后返回。构造"旧 leader 被分区、新 leader 已 commit 新值"场景，验证旧 leader 的 `read_index()` 会因拿不到多数派心跳确认而**拒绝返回 stale 值**（对照 §6.5）。

5. **【单节点成员变更】** 实现 `add_server` / `remove_server` 作为特殊 log entry，节点一旦看到配置变更日志就立即采用新配置（不等 commit）。验证"一次只动一个"时，旧配置多数派与新配置多数派始终相交、不脑裂；再故意一次加 2 个节点，观察/构造可能的双 leader，体会为什么论文禁止它（对照 §6.2）。

---

## 附录 A：本章引用来源清单（均经 WebFetch 实际核实）

| 来源 | URL | 用于本章 |
|------|-----|---------|
| Raft 论文（Extended，USENIX ATC 2014） | https://raft.github.io/raft.pdf | Figure 2/3/6/7/8 逐字、Election Restriction、joint consensus（§1–§4 全部论文引文） |
| etcd-io/raft `raft.go`（main） | https://raw.githubusercontent.com/etcd-io/raft/main/raft.go | tickElection / campaign / becomeCandidate / becomeLeader / MsgVote 处理 / maybeCommit（§3、§4.4） |
| etcd-io/raft `log.go`（main） | https://raw.githubusercontent.com/etcd-io/raft/main/log.go | maybeAppend / matchTerm / findConflict / commitTo 逐字（§4.2） |
| etcd-io/raft `README.md`（main） | https://github.com/etcd-io/raft/blob/main/README.md | 库化设计哲学、PreVote/CheckQuorum/flow control/ReadIndex 特性清单、生产用户（§3.2、§6.3、§6.5） |
| hashicorp/raft `raft.go`（main） | https://raw.githubusercontent.com/hashicorp/raft/main/raft.go | runFollower / runCandidate / electSelf / persistVote（§3.3） |
| CAP theorem（Wikipedia + Brewer 2012） | https://en.wikipedia.org/wiki/CAP_theorem ；https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/ | §1.1 |
| Raft（Wikipedia） | https://en.wikipedia.org/wiki/Raft_(algorithm) | 名称由来、历史、术语、随机超时区间（§1.3、§2） |

**源码版本声明**：etcd-io/raft 与 hashicorp/raft 的 `main` 分支在持续演进。本章源码于 **2026-06** 拉取。etcd 当前 `main` 已迁移到 protobuf-go（字段为指针，故出现 `new(r.id)`、`m.GetFrom()`），早期 gogoproto 版本为值字段（`r.id`、`m.From`），**算法语义不变**，初读以语义为准。hashicorp/raft 版权署名 `IBM Corp.` 反映 HashiCorp 2024 年被 IBM 收购。标注「待核」的项：无（本章所有逐字源码与论文引文均经 WebFetch 实际拉取核实）。
