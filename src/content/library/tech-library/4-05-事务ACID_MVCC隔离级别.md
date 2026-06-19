---
title: "事务 ACID、MVCC 与隔离级别"
slug: "4-05"
collection: "tech-library"
group: "数据库"
order: 4005
summary: "隔离性（Isolation）是 ACID 里最难、也最容易被误解的一个字母。本章的主线论点：“隔离级别”不是一组功能开关，而是一组关于“允许哪些并发异常”的契约——SQL 标准从“禁止哪些 phenomena”反向定义级别，而不是正向定义“保证什么”。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:17:22.000Z"
updatedAt: "2026-06-12T11:17:22.000Z"
---
> **TL;DR**
>
> 隔离性（Isolation）是 ACID 里最难、也最容易被误解的一个字母。本章的主线论点：**“隔离级别”不是一组功能开关，而是一组关于“允许哪些并发异常”的契约**——SQL 标准从“禁止哪些 phenomena”反向定义级别，而不是正向定义“保证什么”。我们以 PostgreSQL 的 heap MVCC 为主线源码（snapshot 的 `xmin/xmax/xip`、`HeapTupleSatisfiesMVCC` 可见性判定、proc array 拍快照），逐行拆解“一个事务为什么看不见另一个事务”。然后回到 1995 年 Berenson/Gray 等人的《A Critique of ANSI SQL Isolation Levels》，理解 Snapshot Isolation 与 write skew（A5B）这个让无数生产系统翻车的异常是怎么被发现的，以及 PostgreSQL 的 SSI（Serializable Snapshot Isolation）如何用“dangerous structure”把 SI 升级成真正的 Serializable。配一个**完整可运行的 toy MVCC 引擎（Python，零依赖）**，复现脏读/不可重复读/幻读/写偏序在不同隔离级别下的真实表现，与上面源码一一印证。

---

## 前置依赖

| 知识点 | 建议先读 | 本章用途 |
|--------|---------|---------|
| WAL / 崩溃恢复（durability 怎么保证） | 第 4 章 | A/C/D 的落地依赖 WAL，本章聚焦 I |
| B-Tree / Heap 物理存储 | 第 3 章 | tuple 在 page 里的物理布局 |
| 事务 XID / CLOG 概念 | 本章 §2 | snapshot 的判定基础 |
| 基础并发：锁 vs 多版本 | 本章 §1.3 | 理解 MVCC 的“无锁读” |

> 阅读约定：源码块逐字标注【真实源码 repo@path】，并给出 WebFetch 实际取过的 URL；非逐字的结构性示意标【示意，非逐字】；个人没有 100% 把握的地方标「待核」。本章所有 PostgreSQL 源码取自 `postgres/postgres` 的 `master` 分支（2026-06 取材），InnoDB / 标准定义来自官方文档与 SIGMOD 1995 论文。

---

## 1. 设计考古：隔离级别为什么“反向”定义

### 1.1 核心问题：可串行化太贵，于是退而求其次

并发控制的“黄金标准”是 **Serializability（可串行化）**：N 个并发事务的执行结果，必须等价于这 N 个事务**某种串行顺序**执行的结果。只要满足这一点，应用开发者就可以**完全不考虑并发**，把每个事务当成独占数据库来写。

问题是：严格做到可串行化，传统手段是 **Strict Two-Phase Locking（S2PL）**——读加共享锁、写加排他锁，全部持有到事务结束。这导致：

- **读阻塞写、写阻塞读**：一个长读事务会把它扫过的行全部锁住，并发写全部排队。
- **吞吐崩塌**：OLTP 高并发下，锁等待 + 死锁检测开销巨大。

于是 ANSI SQL-92 做了一个工程妥协：**定义一个“异常阶梯”，让应用按需放松隔离换取并发性能**。这就是四个隔离级别的由来。关键在于——

> **标准是从“禁止哪些异常（phenomena）”来定义级别的，不是从“保证什么”来定义的。** 这是理解整个隔离体系的钥匙。

### 1.2 三个经典异常（ANSI 原始定义）

ANSI SQL-92 定义了三个 phenomena，用读写操作序列刻画。记号：`w1[x]` = 事务 1 写 x，`r2[x]` = 事务 2 读 x，`c1/a1` = 事务 1 commit/abort。

**出处**：Berenson, Bernstein, Gray, Melton, O'Neil, O'Neil. *A Critique of ANSI SQL Isolation Levels*, SIGMOD 1995. 
URL（已 WebFetch 核实 HTML 版）：https://mwhittaker.github.io/papers/html/berenson1995critique.html 
ACM 收录：https://dl.acm.org/doi/10.1145/223784.223785

论文给出的 phenomena 的“broad interpretation”（宽松解释，记为 P）形式如下【真实论文记号，已核实】：

```
P1 (Dirty Read):        w1[x] ... r2[x] ... (c1 or a1) ... (c2 or a2)
P2 (Non-repeatable Read): r1[x] ... w2[x] ... (c1 or a1) ... (c2 or a2)
P3 (Phantom):           r1[P] ... w2[y in P] ... (c1 or a1) ... (c2 or a2)
```

逐个翻成人话：

- **P1 脏读**：T2 读到了 T1 **尚未提交**的写。如果 T1 随后 abort，T2 就读到了一个“从未真实存在过”的值。
- **P2 不可重复读**：T1 读了 x，T2 改了 x 并提交，T1 **再读同一行** x，值变了。同一个事务里两次读同一行结果不同。
- **P3 幻读**：注意 P3 的对象是**谓词 `[P]`**而不是单行。T1 按条件 `WHERE cls='A'` 读出一批行，T2 **插入**一条满足该条件的新行并提交，T1 **再按同样条件读**，多出来一条“幽灵行”。P2 是“已有行的值变了”，P3 是“满足条件的行集变了”——这是两者的本质区别。

论文一个重要贡献是指出 ANSI 原文的 phenomena 定义有歧义（“strict” A1/A2/A3 vs “broad” P1/P2/P3），并据此重新形式化。这一点我们到 §1.4 讨论 Snapshot Isolation 与 P3 的“不可比”时会再用到。

### 1.3 四个隔离级别 = 异常阶梯

| 隔离级别 | 脏读 P1 | 不可重复读 P2 | 幻读 P3 |
|---------|--------|-------------|--------|
| READ UNCOMMITTED | 允许 | 允许 | 允许 |
| READ COMMITTED | 禁止 | 允许 | 允许 |
| REPEATABLE READ | 禁止 | 禁止 | 允许 |
| SERIALIZABLE | 禁止 | 禁止 | 禁止 |

注意这张表的**思维方式**：级别越高，被禁止的异常越多；标准只规定“某级别**必须**禁止哪些”，**没规定**“某级别**只能**禁止这些”。一个实现完全可以在 REPEATABLE READ 就把幻读也禁了——它**超额**满足了标准。这正是 PostgreSQL 的做法，下一节展开。

**为什么 READ UNCOMMITTED 在很多 MVCC 系统里根本不存在？** 因为 MVCC 的读永远读“某个已提交快照”，从机制上就读不到未提交数据——脏读在 MVCC 架构里**没有发生的物理路径**。所以 PostgreSQL 干脆把 READ UNCOMMITTED 映射成 READ COMMITTED（§3.2 有官方原文）。

### 1.4 Snapshot Isolation：第四种异常的发现

1995 年论文最重要的贡献，是定义了一个 ANSI 标准里**没有**的隔离级别：**Snapshot Isolation（SI，快照隔离）**，并指出 SI 与 ANSI 的级别**不可直接比较**。

SI 的定义（论文原文意译，已 WebFetch 核实）：

> 每个事务开始时拿到一个 **begin timestamp**，之后所有读都从这个时间戳的数据库快照读。提交时拿一个 **commit timestamp**，若本事务的写集与 `[begin, commit]` 区间内任何已提交事务的写集**不相交**，则提交成功（**first-committer-wins**，首提交者胜）。

SI 的性质（论文结论，已核实）：

- SI **排除** A1（脏读）、A2（不可重复读）、A3（部分幻读 anomaly）。
- 但 SI **不排除** A5B —— **Write Skew（写偏序）**，这是论文新引入的异常。
- 论文明确指出：SI 与 REPEATABLE READ **不可比（incomparable）**——SI 能防住一部分 RR 防不住的（如某些 phantom anomaly），但 SI 防不住 write skew，而严格的 RR（基于谓词锁）能防住。

**A5A / A5B 的形式定义**（论文记号，已核实）：

```
A5A (Read Skew):  r1[x] ... w2[x] w2[y] c2 ... r1[y] ... (c1 or a1)
A5B (Write Skew): r1[x] ... r2[y] ... w1[y] ... w2[x] ... (c1 and c2)
```

**Write Skew 的直觉**——这是本章的重点异常，值得反复嚼：

> 两个事务各自读了一组数据，各自基于“自己读到的旧快照”做了一个**单独看完全合法**的写，但两个写**合起来**破坏了一个**跨行约束**。

经典例子（医院排班）：约束是“任意时刻至少 1 名医生 on-call”。当前 A、B 两名医生都 on-call。

- T1 读到“B 还在岗” → 认为自己可以休假 → 把 A 改成休假。
- T2 读到“A 还在岗” → 认为自己可以休假 → 把 B 改成休假。

两个事务的写集（`{A}` 和 `{B}`）**不相交**，SI 的 first-committer-wins 检查**不会拦截**，两个都提交成功。结果：**无人在岗**，约束被破坏。这就是 write skew，§4 的 demo 会把它真实跑出来。

> **为什么 SI 防不住？** 因为 SI 只检查**写-写冲突**（写集相交）。write skew 的两个事务**读了对方要写的数据**，但**没写同一行**——这是**读-写冲突（rw-conflict）**，SI 完全看不见。这个洞，正是后来 SSI 要补的（§5）。

---

## 2. PostgreSQL Heap MVCC：源码级精读

PostgreSQL 的 MVCC 是“**多版本就地存储**”流派的标杆：每次 UPDATE 不原地改，而是在 heap 里**插一个新版本 tuple**，旧版本留在原地（靠后台 VACUUM 回收）。每个 tuple 头部带两个事务 ID：`t_xmin`（谁插入的）、`t_xmax`（谁删除/更新的）。可见性 = 拿你的 snapshot 去判断“插入者对我可见吗？删除者对我可见吗？”。

### 2.1 tuple 头部：版本的物理载体

【真实源码 postgres@src/include/access/htup_details.h】 
URL（已 WebFetch）：https://raw.githubusercontent.com/postgres/postgres/master/src/include/access/htup_details.h

```c
typedef struct HeapTupleFields
{
	TransactionId t_xmin;		/* inserting xact ID */
	TransactionId t_xmax;		/* deleting or locking xact ID */

	union
	{
		CommandId	t_cid;		/* inserting or deleting command ID, or both */
		TransactionId t_xvac;	/* old-style VACUUM FULL xact ID */
	}			t_field3;
} HeapTupleFields;
```

逐行：

- `t_xmin`：**插入**这个版本的事务 ID。
- `t_xmax`：**删除或更新（或加锁）**这个版本的事务 ID；`0`（InvalidTransactionId）表示尚未被删。
- `t_cid`（CommandId）：**同一事务内**的命令序号。这就是 SnapshotData 里 `curcid` 要比对的对象——它让“一个事务内，第 5 条语句看不见第 7 条语句插入的行”成为可能（即“语句级”自我可见性）。

这三个字段藏在每个 heap tuple 头里。再看头部里的 hint bits（性能关键）：

【真实源码 postgres@src/include/access/htup_details.h】

```c
#define HEAP_XMIN_COMMITTED		0x0100	/* t_xmin committed */
#define HEAP_XMIN_INVALID		0x0200	/* t_xmin invalid/aborted */
#define HEAP_XMIN_FROZEN		(HEAP_XMIN_COMMITTED|HEAP_XMIN_INVALID)
#define HEAP_XMAX_COMMITTED		0x0400	/* t_xmax committed */
#define HEAP_XMAX_INVALID		0x0800	/* t_xmax invalid/aborted */
```

**这几个 bit 是 PostgreSQL MVCC 性能的隐藏命脉**。判断一个 XID 是否已提交，本来要去查 CLOG（`pg_xact`，事务状态位图）——那是一次可能 miss cache 的随机访问。第一次访问某 tuple 并确认其 `t_xmin` 已提交后，PG 会把 `HEAP_XMIN_COMMITTED` **写回 tuple 头**（这叫 hint bit），下次直接看 bit、不查 CLOG。

> **生产真坑**：hint bit 是个“写”。这意味着一个**纯 SELECT** 也可能弄脏 page、产生 WAL（full-page write）和磁盘写。很多人困惑“我只读为什么有大量写 I/O / dirty buffer”，根因常是首次扫描冷表批量回写 hint bits。`HEAP_XMIN_FROZEN`（= COMMITTED|INVALID 两位同时置位，一个不可能的组合被复用为“冻结”标记）则是 VACUUM 把老 tuple 标记为“对所有快照永远可见”，用于解决 XID wraparound（32 位 XID 回绕）。

### 2.2 SnapshotData：一次快照到底是什么

【真实源码 postgres@src/include/utils/snapshot.h】 
URL（已 WebFetch）：https://raw.githubusercontent.com/postgres/postgres/master/src/include/utils/snapshot.h

```c
typedef struct SnapshotData
{
	SnapshotType snapshot_type; /* type of snapshot */
	TransactionId xmin;			/* all XID < xmin are visible to me */
	TransactionId xmax;			/* all XID >= xmax are invisible to me */
	TransactionId *xip;
	uint32		xcnt;			/* # of xact ids in xip[] */
	TransactionId *subxip;
	int32		subxcnt;		/* # of xact ids in subxip[] */
	bool		suboverflowed;	/* has the subxip array overflowed? */
	bool		takenDuringRecovery;	/* recovery-shaped snapshot? */
	bool		copied;			/* false if it's a static snapshot */
	CommandId	curcid;			/* in my xact, CID < curcid are visible */
	/* ... 省略 speculativeToken / vistest / 引用计数等字段 ... */
} SnapshotData;
```

**一个 snapshot 就是三件套 `{xmin, xmax, xip[]}`**，加上自我可见用的 `curcid`。它精确刻画了“在我拍快照那一刻，事务世界的状态”：

- `xmin`：**所有 `XID < xmin` 的事务都已结束**（提交或回滚，结局已定，可直接查 CLOG）。
- `xmax`：**所有 `XID >= xmax` 的事务都还没开始**（对我必然不可见）。
- `xip[]`：落在 `[xmin, xmax)` 区间、但**在我拍快照那一刻仍在进行中**的 XID 列表。

判断规则（这就是整个 MVCC 的逻辑核心，记住它）：

```
给定一个 XID：
  XID <  xmin            -> 结局已定，查 CLOG（commit=可见 / abort=不可见）
  XID >= xmax            -> 还没开始 -> 对我不可见
  xmin <= XID < xmax:
        XID in xip[]     -> 拍快照时仍在跑 -> 对我不可见
        XID not in xip[] -> 拍快照时已提交 -> 可见
```

snapshot.h 里对 xmin/xmax 的注释把这点说得很死【真实源码注释】：

```c
/* An MVCC snapshot can never see the effects of XIDs >= xmax.
 * It can see the effects of all older XIDs except those listed in
 * the snapshot. */
```

`subxip[]` 是子事务（SAVEPOINT 产生的 subtransaction）的 in-progress 列表；`suboverflowed` 表示子事务太多、数组溢出（此时退化成查 `pg_subtrans`）。`copied` 区分静态快照（指向全局 `CurrentSnapshot` 那块静态内存）和拷贝出来的快照。

### 2.3 GetSnapshotData：快照是怎么被“拍”出来的

快照不是凭空来的——拍快照要**扫描 proc array**（PGPROC 数组，记录每个活跃 backend 当前的 XID）。

【真实源码 postgres@src/backend/storage/ipc/procarray.c，函数 GetSnapshotData】 
URL（已 WebFetch）：https://raw.githubusercontent.com/postgres/postgres/master/src/backend/storage/ipc/procarray.c

函数头注释（已核实，逐字）：

```c
/*
 * GetSnapshotData -- returns information about running transactions.
 *
 * The returned snapshot includes xmin (lowest still-running xact ID),
 * xmax (highest completed xact ID + 1), and a list of running xact IDs
 * in the range xmin <= xid < xmax.  It is used as follows:
 *		All xact IDs < xmin are considered finished.
 *		All xact IDs >= xmax are considered still running.
 *		For an xact ID xmin <= xid < xmax, consult list to see whether
 *		it is considered running or not.
 * ...
 */
```

核心扫描逻辑【示意，非逐字 —— 真实函数有大量 recovery / 性能优化分支，这里抽取主干以对应 §2.2 的三件套】：

```c
/* xmax = 最近一个已完成事务的 XID + 1 */
xmax = XidFromFullTransactionId(latest_completed);
TransactionIdAdvance(xmax);          /* +1 */

xmin = xmax;                         /* 先设为上界，下面往小修 */

/* 扫描 proc array 里所有 backend 的当前 XID */
for (int pgxactoff = 0; pgxactoff < numProcs; pgxactoff++)
{
    TransactionId xid = UINT32_ACCESS_ONCE(other_xids[pgxactoff]);

    if (!TransactionIdIsNormal(xid))     /* 没有分配 XID 的只读事务跳过 */
        continue;
    if (xid >= xmax)                     /* 超出上界的不收 */
        continue;

    if (NormalTransactionIdPrecedes(xid, xmin))
        xmin = xid;                      /* 维护“最小活跃 XID” */

    xip[count++] = xid;                  /* 收进 in-progress 列表 */
    /* ... 还要收该 backend 的 subtransaction XIDs 到 subxip ... */
}
```

几个**工程要害**：

1. **整个扫描在 `ProcArrayLock`（共享模式）保护下进行**——保证拍快照期间“活跃事务集合”不变（注释原文：*prevents the active transaction set from changing while ProcArrayLock is held*，已核实）。这把锁是 PG 高并发下著名的热点。
2. **只读事务不分配 XID**（`xid` 为 0 被跳过）。这是 PG 的一个关键优化：海量只读连接不会撑大 `xip[]`，也不推高 xmin。
3. **PostgreSQL 14 的著名优化**：历史上 `GetSnapshotData` 在数千连接下因频繁扫 proc array 成为可扩展性瓶颈。PG14（commit `1f51c17c68`，Andres Freund）把“计算 xmin/xmax 所需的 XID”与“完整 `xip[]`”分离，引入 `vistest`（GlobalVisState），让大部分可见性判断不必生成完整 `xip[]`，显著提升高连接数吞吐。这是“snapshot 拍摄成本”被当成一等性能问题对待的实证。「PG14 优化的 commit 号与作者凭记忆，建议核：待核」

### 2.4 HeapTupleSatisfiesMVCC：可见性判定的真身

这是把“tuple 的 `t_xmin/t_xmax`”和“我的 snapshot”碰在一起、产出“可见/不可见”布尔值的函数。它是整个 MVCC 的**终点**。

【真实源码 postgres@src/backend/access/heap/heapam_visibility.c，函数 HeapTupleSatisfiesMVCC】 
URL（已 WebFetch）：https://raw.githubusercontent.com/postgres/postgres/master/src/backend/access/heap/heapam_visibility.c

下面是**真实函数体**（逐字取自上述文件，我在右侧加 `←` 中文注解；注解非源码）：

```c
static inline bool
HeapTupleSatisfiesMVCC(HeapTuple htup, Snapshot snapshot,
					   Buffer buffer, SetHintBitsState *state)
{
	HeapTupleHeader tuple = htup->t_data;

	/* ... 一堆 Assert：要求 snapshot 已注册、tuple 自指针合法 ... */

	if (!HeapTupleHeaderXminCommitted(tuple))          // ← 情况A：xmin 还没被标记“已提交”
	{
		if (HeapTupleHeaderXminInvalid(tuple))
			return false;                              // ← xmin 已标记 abort/invalid -> 不可见

		/* ... HeapTupleCleanMoved 是 VACUUM FULL 旧机制的处理，常态忽略 ... */
		else if (TransactionIdIsCurrentTransactionId(HeapTupleHeaderGetRawXmin(tuple)))
		{
			// ← xmin 是“我自己”：用命令号 cmin 做语句级自我可见性
			if (HeapTupleHeaderGetCmin(tuple) >= snapshot->curcid)
				return false;	/* inserted after scan started */   // ← 本语句之后插的，看不见

			if (tuple->t_infomask & HEAP_XMAX_INVALID)	/* xid invalid */
				return true;                            // ← 没被删 -> 可见

			/* ... 处理 xmax 是 multixact / 自己删自己 的若干分支 ... */

			if (HeapTupleHeaderGetCmax(tuple) >= snapshot->curcid)
				return true;	/* deleted after scan started */    // ← 本语句之后才删，仍可见
			else
				return false;	/* deleted before scan started */
		}
		else if (XidInMVCCSnapshot(HeapTupleHeaderGetRawXmin(tuple), snapshot))
			return false;                               // ← xmin 在快照的 in-progress 集合里 -> 不可见
		else if (TransactionIdDidCommit(HeapTupleHeaderGetRawXmin(tuple)))
			SetHintBitsExt(tuple, buffer, HEAP_XMIN_COMMITTED,
						   HeapTupleHeaderGetRawXmin(tuple), state);  // ← 查 CLOG=已提交，回写 hint bit
		else
		{
			/* it must have aborted or crashed */
			SetHintBitsExt(tuple, buffer, HEAP_XMIN_INVALID,
						   InvalidTransactionId, state);
			return false;                               // ← xmin abort -> 不可见
		}
	}
	else
	{
		/* xmin is committed, but maybe not according to our snapshot */
		if (!HeapTupleHeaderXminFrozen(tuple) &&
			XidInMVCCSnapshot(HeapTupleHeaderGetRawXmin(tuple), snapshot))
			return false;		/* treat as still in progress */
		// ← 即便 CLOG 说已提交，但若在“我快照时刻”它还在跑，对我仍不可见！这是快照隔离的精髓
	}

	/* by here, the inserting transaction has committed */
	// ← 走到这里：插入者对我可见。接下来判断“删除者”

	if (tuple->t_infomask & HEAP_XMAX_INVALID)	/* xid invalid or aborted */
		return true;                                    // ← 没有有效删除者 -> 可见

	if (HEAP_XMAX_IS_LOCKED_ONLY(tuple->t_infomask))
		return true;                                    // ← xmax 只是加锁不是删除 -> 可见

	/* ... multixact 分支略 ... */

	if (!(tuple->t_infomask & HEAP_XMAX_COMMITTED))
	{
		if (TransactionIdIsCurrentTransactionId(HeapTupleHeaderGetRawXmax(tuple)))
		{
			if (HeapTupleHeaderGetCmax(tuple) >= snapshot->curcid)
				return true;	/* deleted after scan started */
			else
				return false;	/* deleted before scan started */
		}

		if (XidInMVCCSnapshot(HeapTupleHeaderGetRawXmax(tuple), snapshot))
			return true;        // ← 删除者在我快照时还在跑 -> 删除对我不可见 -> 行仍可见

		if (!TransactionIdDidCommit(HeapTupleHeaderGetRawXmax(tuple)))
		{
			/* it must have aborted or crashed */
			SetHintBitsExt(tuple, buffer, HEAP_XMAX_INVALID,
						   InvalidTransactionId, state);
			return true;        // ← 删除者 abort -> 行仍可见
		}

		SetHintBitsExt(tuple, buffer, HEAP_XMAX_COMMITTED, ...);  // ← 删除者已提交，回写 hint
	}
	else
	{
		/* xmax is committed, but maybe not according to our snapshot */
		if (XidInMVCCSnapshot(HeapTupleHeaderGetRawXmax(tuple), snapshot))
			return true;		/* treat as still in progress */
	}

	/* xmax transaction committed */
	return false;               // ← 删除者已提交且对我可见 -> 行已被删 -> 不可见
}
```

**把这个函数压缩成两句话**：

1. **看 `t_xmin`**：插入这行的事务，按我的 snapshot 算，**对我可见吗**？不可见就 `false`。
2. **看 `t_xmax`**：删除这行的事务，按我的 snapshot 算，**对我可见吗**？可见（说明在我眼里这行已被删）就 `false`，否则 `true`。

注意那个**反直觉但至关重要**的分支（上面标“快照隔离的精髓”那行）：**即使 CLOG 已经记录某事务提交了，只要它在“我拍快照的那一刻”还在进行中（在 `xip[]` 里），它的效果对我就永远不可见**——哪怕我的事务跑了一个小时、它早就提交了。这就是 Repeatable Read 下“整个事务看到同一个一致快照”的实现根基。

文件头注释还点出一个**只在非 MVCC 快照下才有**的微妙顺序问题（已核实，逐字）：

```c
/* When using a non-MVCC snapshot, we must check TransactionIdIsInProgress
 * before TransactionIdDidCommit ... to avoid race conditions. */
```

即：必须**先**确认“事务还在跑”，**再**去查 CLOG，否则会有竞态——一个事务可能在你“查到它没在跑”和“查 CLOG”之间完成提交。MVCC 快照因为用的是**固定时刻**的 `xip[]`，天然规避了这个竞态（`XidInMVCCSnapshot` 查的是快照里冻结的集合，不是实时状态）。

### 2.5 readers 不阻塞 writers：MVCC 的核心卖点

【官方文档 postgres@doc，mvcc-intro】，已 WebFetch：https://www.postgresql.org/docs/current/mvcc-intro.html

> "in MVCC locks acquired for querying (reading) data do not conflict with locks acquired for writing data, and so **reading never blocks writing and writing never blocks reading**."

这一句是 MVCC 相对 S2PL 的根本优势。因为读走的是“判断旧版本可见性”这条路，根本不需要对数据行加共享锁；写则是“插新版本 + 给旧版本打 xmax”，也不必等读者放锁。代价是：**旧版本堆积**，需要 VACUUM 清理（§6 生产坑）。

---

## 3. 隔离级别在 PostgreSQL 的真实语义

### 3.1 官方异常表：PG 比标准“更严”

【官方文档 postgres@doc，transaction-iso，Table 13.1】，已 WebFetch：https://www.postgresql.org/docs/current/transaction-iso.html

| Isolation Level | Dirty Read | Nonrepeatable Read | Phantom Read | Serialization Anomaly |
|---|---|---|---|---|
| Read uncommitted | Allowed, but not in PG | Possible | Possible | Possible |
| Read committed | Not possible | Possible | Possible | Possible |
| Repeatable read | Not possible | Not possible | **Allowed, but not in PG** | Possible |
| Serializable | Not possible | Not possible | Not possible | Not possible |

注意第 5 列 **Serialization Anomaly**——这是标准三异常之外，PG 文档**额外**列出的一列，它对应的就是 §1.4 的 write skew 一类“没有任何单一串行顺序能解释结果”的异常。**只有 Serializable 能消除它**。这张表把 §1 的理论和 §2 的实现连了起来。

### 3.2 PG 只有三个真实级别

官方原文（已核实，逐字）：

> "In PostgreSQL, you can request any of the four standard transaction isolation levels, but internally only three distinct isolation levels are implemented, i.e., **PostgreSQL's Read Uncommitted mode behaves like Read Committed**. This is because it is the only sensible way to map the standard isolation levels to PostgreSQL's multiversion concurrency control architecture."

呼应 §1.3：MVCC 架构下脏读没有物理路径，READ UNCOMMITTED 退化成 READ COMMITTED 是必然。

### 3.3 Read Committed：每语句一张新快照

官方原文（已核实，逐字）：

> "Because Read Committed mode starts each command with a new snapshot that includes all transactions committed up to that instant, subsequent commands in the same transaction will see the effects of the committed concurrent transaction in any case."

**关键机制**：RC 下，**每条 SQL 语句**都调一次 `GetSnapshotData` 拿新快照。所以同一个事务里，第二条语句能看到第一条语句执行后才提交的并发事务——这正是“不可重复读”在 RC 下发生的根因。

RC 下 UPDATE 的“**EvalPlanQual**”行为（这是个高频面试点 + 生产坑），官方原文（已核实，逐字）：

> "If the first updater commits, the second updater will ignore the row if the first updater deleted it, otherwise it will attempt to apply its operation to the updated version of the row. **The search condition of the command (the `WHERE` clause) is re-evaluated to see if the updated version of the row still matches the search condition.**"

人话：RC 下两个事务改同一行，后者会**等**前者提交，然后**跳到前者写的新版本上**，并**重新评估 WHERE**。这破坏了“UPDATE 看到的是一个一致快照”的错觉——后者实际上对一个**比它快照更新**的版本动了手。由此引出官方文档亲自给的 **lost update** 例子（已核实，逐字）：

```sql
BEGIN;
UPDATE website SET hits = hits + 1;
-- 另一个 session 并发执行： DELETE FROM website WHERE hits = 10;
COMMIT;
```

> "The `DELETE` will have no effect even though there is a `website.hits = 10` row before and after the `UPDATE`. This occurs because the pre-update row value `9` is skipped, and when the `UPDATE` completes and `DELETE` obtains a lock, the new row value is no longer `10` but `11`."

**这就是 RC 的隐藏陷阱**：基于读到的值做条件写（read-modify-write）在 RC 下**不安全**。`hits=10` 这个条件在 DELETE 拍快照时成立，但等它拿到锁、重评估时，值已是 11，条件不再满足，DELETE 默默失效。生产中“计数器/库存扣减偶尔丢更新”十有八九是这个。**正解**：`SELECT ... FOR UPDATE` 显式加锁，或升级到 RR。

### 3.4 Repeatable Read = Snapshot Isolation

官方原文（已核实，逐字）：

> "The Repeatable Read isolation level is implemented using a technique known in academic database literature and in some other database products as **Snapshot Isolation**."
>
> "PostgreSQL's Repeatable Read implementation **does not allow phantom reads**."

**两个要点**：

1. PG 的 RR **就是 SI**：整个事务用**第一条语句**拍的那一张快照（§2.4 “快照隔离的精髓”分支保证了这点）。
2. PG 的 RR **顺手把幻读也防了**——因为快照是固定的，并发插入的新行 `t_xmin` 在我的 `xip` 之后，`HeapTupleSatisfiesMVCC` 直接判不可见。这就是表里“Allowed, but not in PG”的由来：标准**允许** RR 出现幻读，但 PG 的 SI 实现**不会**。

RR 下写冲突的处理（已核实，逐字）：

> "if the first updater commits ... then the repeatable read transaction will be rolled back with the message: `ERROR: could not serialize access due to concurrent update` ... because a repeatable read transaction cannot modify or lock rows changed by other transactions after the repeatable read transaction began."

这就是 SI 的 **first-updater/first-committer-wins**：RR 事务不允许改“它快照之后被别人改过”的行，撞上就回滚。官方明确要求**应用层重试整个事务**（已核实，逐字）：

> "When an application receives this error message, it should **abort the current transaction and retry the whole transaction from the beginning**."

> **工程含义**：用 RR/Serializable 的应用，**必须**实现“捕获序列化失败 → 整事务重试”的循环。这不是可选项，是契约的一部分。很多人把 RR/SER 当“开了就万事大吉”，结果线上偶发 `40001` 直接报错给用户——因为没写重试。

---

## 4. ⭐ 可运行 Demo：toy MVCC 引擎复现四大异常

**这是本章的重中之重。** 下面是一个**零依赖、可直接运行**的 Python toy MVCC 引擎，它把 §2 的源码思想压缩成最小可执行模型：

- 每行存成**版本链**，每个版本带 `(xmin, xmax)` —— 对应 §2.1 的 heap tuple。
- 事务开始拍 `Snapshot(xmin, xmax, xip)` —— 对应 §2.2 / §2.3 的 `GetSnapshotData`。
- `visible(version, snap)` —— 对应 §2.4 的 `HeapTupleSatisfiesMVCC`（极简版）。
- 三种隔离级别：RC（每语句新快照）/ RR（事务级快照 = SI）/ SERIALIZABLE（SI + write-skew 检测）。

> **标注：设计为可运行，请在你的环境验证。** 依赖：仅 Python 3.8+ 标准库（`dataclasses`、`typing`、`itertools`），无任何第三方包。下方输出是我在 Python 3.12.12 实际运行的真实结果。

### 4.1 完整代码

```python
#!/usr/bin/env python3
"""toy MVCC 引擎 —— 印证 PostgreSQL heap MVCC + snapshot 可见性规则。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import itertools

COMMITTED, ABORTED, IN_PROGRESS = "committed", "aborted", "in_progress"
RC, RR, SER = "READ_COMMITTED", "REPEATABLE_READ", "SERIALIZABLE"


@dataclass
class Version:                       # 对应 heap tuple
    value: dict
    xmin: int                       # 插入该版本的 xid（t_xmin）
    xmax: Optional[int] = None      # 删除/更新该版本的 xid（t_xmax）


@dataclass
class Snapshot:                     # 对应 SnapshotData
    xmin: int                       # < xmin 的 xid 全部已结束
    xmax: int                       # >= xmax 的 xid 全部还没开始
    xip: frozenset                  # [xmin,xmax) 区间里仍在进行中的 xid


class Engine:
    def __init__(self):
        self.rows: dict[int, list[Version]] = {}
        self.xstatus: dict[int, str] = {}          # xid -> 状态（CLOG）
        self.next_xid = itertools.count(100)
        self.read_keys: dict[int, set] = {}        # SER 用：读集
        self.write_keys: dict[int, set] = {}       # SER 用：写集

    def begin(self, iso=RC) -> "Txn":
        xid = next(self.next_xid)
        self.xstatus[xid] = IN_PROGRESS
        self.read_keys[xid] = set(); self.write_keys[xid] = set()
        return Txn(self, xid, iso)

    def _take_snapshot(self) -> Snapshot:          # 对应 GetSnapshotData
        in_prog = [x for x, s in self.xstatus.items() if s == IN_PROGRESS]
        xmin = min(in_prog) if in_prog else max(self.xstatus, default=99) + 1
        xmax = max(self.xstatus.keys(), default=99) + 1
        return Snapshot(xmin=xmin, xmax=xmax, xip=frozenset(in_prog))

    def _xid_visible(self, xid, snap, self_xid) -> bool:   # HeapTupleSatisfiesMVCC 核心判定
        if xid == self_xid:        return True             # 自己写的看得见
        if xid >= snap.xmax:       return False            # 快照之后才开始
        if xid in snap.xip:        return False            # 快照时刻仍在跑
        return self.xstatus.get(xid) == COMMITTED          # 查 CLOG

    def visible(self, v, snap, self_xid) -> bool:
        if not self._xid_visible(v.xmin, snap, self_xid):          # 插入者对我可见吗
            return False
        if v.xmax is not None and self._xid_visible(v.xmax, snap, self_xid):  # 删除者对我可见吗
            return False
        return True

    def read(self, txn, pk) -> Optional[dict]:
        snap = txn.stmt_snapshot()
        self.read_keys[txn.xid].add(pk)
        for v in reversed(self.rows.get(pk, [])):
            if self.visible(v, snap, txn.xid):
                return v.value
        return None

    def scan(self, txn, pred) -> list[dict]:        # 谓词扫描，演示幻读
        snap = txn.stmt_snapshot(); out = []
        for pk, chain in self.rows.items():
            for v in reversed(chain):
                if self.visible(v, snap, txn.xid) and pred(v.value):
                    self.read_keys[txn.xid].add(pk); out.append(v.value); break
        return out

    def insert(self, txn, pk, value):
        self.rows.setdefault(pk, []).append(Version(value=value, xmin=txn.xid))
        self.write_keys[txn.xid].add(pk)

    def update(self, txn, pk, value):
        snap = txn.stmt_snapshot(); chain = self.rows.get(pk, [])
        for v in reversed(chain):
            if self.visible(v, snap, txn.xid):
                v.xmax = txn.xid                                   # 旧版本打删除标记
                chain.append(Version(value=value, xmin=txn.xid))   # 追加新版本
                self.write_keys[txn.xid].add(pk); return
        raise RuntimeError(f"update: no visible row pk={pk}")

    def commit(self, txn):
        if txn.iso == SER and self._has_write_skew(txn):
            self.abort(txn)
            raise SerializationFailure(
                "could not serialize access due to read/write dependencies")
        self.xstatus[txn.xid] = COMMITTED

    def abort(self, txn): self.xstatus[txn.xid] = ABORTED

    def _has_write_skew(self, txn) -> bool:
        """极简 write-skew 检测：印证 SSI 的 dangerous structure 思想（非 PG SSI 完整实现）。
        若存在并发已提交事务 T'，本事务读集∩T'写集 且 本事务写集∩T'读集 -> 两条 rw 反依赖 -> 回滚。"""
        my_r, my_w = self.read_keys[txn.xid], self.write_keys[txn.xid]
        for other, st in self.xstatus.items():
            if other == txn.xid or st != COMMITTED or other not in self.read_keys:
                continue
            if (my_r & self.write_keys[other]) and (my_w & self.read_keys[other]):
                return True
        return False


class SerializationFailure(Exception): pass


@dataclass
class Txn:
    engine: Engine; xid: int; iso: str
    _txn_snapshot: Optional[Snapshot] = None
    def stmt_snapshot(self) -> Snapshot:
        if self.iso == RC:                              # RC：每语句新拍
            return self.engine._take_snapshot()
        if self._txn_snapshot is None:                  # RR/SER：首次用时拍一次
            self._txn_snapshot = self.engine._take_snapshot()
        return self._txn_snapshot


def banner(t): print("\n" + "=" * 64 + f"\n {t}\n" + "=" * 64)


def demo_no_dirty_read():
    banner("场景1 脏读：未提交的写对别人不可见")
    e = Engine(); s = e.begin(); e.insert(s, 1, {"bal": 100}); e.commit(s)
    t1 = e.begin(RC); e.update(t1, 1, {"bal": 999})          # 改了未提交
    t2 = e.begin(RC)
    print(" t2 读到 bal =", e.read(t2, 1)["bal"], "（期望 100，看不到未提交的 999）")
    e.commit(t1)
    print(" t1 提交后 t2 再读 =", e.read(t2, 1)["bal"], "（RC 新快照 -> 999）")
    e.commit(t2)


def demo_nonrepeatable_read():
    banner("场景2 不可重复读：RC 出现，RR 消失")
    for iso in (RC, RR):
        e = Engine(); s = e.begin(); e.insert(s, 1, {"bal": 100}); e.commit(s)
        t = e.begin(iso); first = e.read(t, 1)["bal"]
        w = e.begin(); e.update(w, 1, {"bal": 500}); e.commit(w)
        second = e.read(t, 1)["bal"]; e.commit(t)
        v = "一致✓" if first == second else "不一致✗(不可重复读)"
        print(f" {iso:16s} 第一次={first} 第二次={second} -> {v}")


def demo_phantom_read():
    banner("场景3 幻读：谓词 count，RC 出现，RR 消失")
    for iso in (RC, RR):
        e = Engine(); s = e.begin()
        e.insert(s, 1, {"cls": "A"}); e.insert(s, 2, {"cls": "A"}); e.commit(s)
        t = e.begin(iso); n1 = len(e.scan(t, lambda r: r["cls"] == "A"))
        w = e.begin(); e.insert(w, 3, {"cls": "A"}); e.commit(w)
        n2 = len(e.scan(t, lambda r: r["cls"] == "A")); e.commit(t)
        v = "无幻行✓" if n1 == n2 else "出现幻行✗(phantom)"
        print(f" {iso:16s} count1={n1} count2={n2} -> {v}")


def demo_write_skew():
    banner("场景4 写偏序：约束 oncall(A)+oncall(B)>=1，RR 破坏，SER 拦截")
    for iso in (RR, SER):
        e = Engine(); s = e.begin()
        e.insert(s, "A", {"oncall": True}); e.insert(s, "B", {"oncall": True}); e.commit(s)
        t1 = e.begin(iso); t2 = e.begin(iso)
        e.read(t1, "A"); e.read(t1, "B")      # t1 读两行（读集={A,B}）
        e.read(t2, "A"); e.read(t2, "B")      # t2 读两行（读集={A,B}）
        e.update(t1, "A", {"oncall": False})  # t1 看到 B 在岗 -> 让 A 休假
        e.update(t2, "B", {"oncall": False})  # t2 看到 A 在岗 -> 让 B 休假
        out = []
        for tx, name in ((t1, "t1"), (t2, "t2")):
            try: e.commit(tx); out.append(f"{name}=commit")
            except SerializationFailure: out.append(f"{name}=ROLLBACK")
        chk = e.begin(RC); af = e.read(chk, "A")["oncall"]; bf = e.read(chk, "B")["oncall"]; e.commit(chk)
        print(f" {iso:16s} {' '.join(out):24s} A={af} B={bf} "
              f"约束={'保持✓' if (af or bf) else '被破坏✗'}")


if __name__ == "__main__":
    demo_no_dirty_read(); demo_nonrepeatable_read()
    demo_phantom_read(); demo_write_skew()
    print("\n[done] 全部场景执行完毕。")
```

### 4.2 运行步骤

```bash
# 1. 保存上面代码为 mvcc_demo.py（无需 venv，无第三方依赖）
# 2. 直接运行：
python3 mvcc_demo.py
```

### 4.3 真实运行输出（Python 3.12.12 实测）

```
================================================================
 场景1 脏读：未提交的写对别人不可见
================================================================
 t2 读到 bal = 100 （期望 100，看不到未提交的 999）
 t1 提交后 t2 再读 = 999 （RC 新快照 -> 999）

================================================================
 场景2 不可重复读：RC 出现，RR 消失
================================================================
 READ_COMMITTED   第一次=100 第二次=500 -> 不一致✗(不可重复读)
 REPEATABLE_READ  第一次=100 第二次=100 -> 一致✓

================================================================
 场景3 幻读：谓词 count，RC 出现，RR 消失
================================================================
 READ_COMMITTED   count1=2 count2=3 -> 出现幻行✗(phantom)
 REPEATABLE_READ  count1=2 count2=2 -> 无幻行✓

================================================================
 场景4 写偏序：约束 oncall(A)+oncall(B)>=1，RR 破坏，SER 拦截
================================================================
 REPEATABLE_READ  t1=commit t2=commit       A=False B=False 约束=被破坏✗
 SERIALIZABLE     t1=commit t2=ROLLBACK     A=False B=True  约束=保持✓
```

### 4.4 输出与源码/标准的逐项印证

| 场景 | demo 输出 | 印证了什么 |
|------|----------|-----------|
| 1 脏读 | t2 始终读不到未提交的 999 | `_xid_visible` 里 `xid in snap.xip -> False` 对应 §2.4 `XidInMVCCSnapshot -> return false`；MVCC 架构无脏读路径（§3.2） |
| 2 不可重复读 | RC 第二次变 500，RR 不变 | RC **每语句新快照**（§3.3）vs RR **事务级快照**（§3.4）；`Txn.stmt_snapshot` 的两条分支 |
| 3 幻读 | RC count 2→3，RR 恒 2 | 新行 `xmin` 落在 RR 快照 `xip` 之后 → 不可见；印证 PG 表“Phantom: Allowed, but not in PG”（§3.1） |
| 4 写偏序 | RR 两者都提交、约束破坏；SER 拦下一个 | SI 只查写集不相交（`{A}∩{B}=∅`）放行 → write skew（§1.4）；SER 用两条 rw 反依赖（dangerous structure）回滚（§5） |

**这张表是本章的闭环**：理论（§1 论文异常）→ 实现（§2 PG 源码）→ 标准（§3 PG 文档表）→ 可执行验证（§4 demo），四层对上了。

> **demo 的诚实边界**：场景 4 的 `_has_write_skew` 是“危险结构”思想的**极简近似**——它只用“读集/写集相交”判定，**没有**构建真实的 rw-依赖图、没有判定 pivot 的提交顺序，因此会比 PG 的 SSI **更激进**地回滚（可能误杀本可串行的组合，即 false positive）。它能正确**复现** write skew 被拦截这一行为，但**不是** SSI 的精确实现。真实 SSI 见 §5。

---

## 5. 把 SI 升级成真 Serializable：PostgreSQL 的 SSI

PG 9.1 起，`SERIALIZABLE` 不再靠 S2PL，而是 **SSI（Serializable Snapshot Isolation）**：在 SI 的基础上，**运行时检测 rw-依赖构成的“危险结构”**，命中就回滚其中一个事务。它保留了 MVCC“读不加锁”的全部好处，只为可串行化付出“检测 + 偶尔回滚”的代价。

### 5.1 理论根基：dangerous structure

【真实源码 postgres@src/backend/storage/lmgr/README-SSI】，已 WebFetch：https://raw.githubusercontent.com/postgres/postgres/master/src/backend/storage/lmgr/README-SSI

README 的核心论断（已核实，逐字 + ASCII 图逐字）：

> "SSI is based on the observation [2] that each snapshot isolation anomaly corresponds to a cycle that contains a **'dangerous structure' of two adjacent rw-conflict edges**:"

```
      Tin ------> Tpivot ------> Tout
            rw             rw
```

理论来源（已核实）：

- Cahill, Röhm, Fekete. *Serializable Isolation for Snapshot Databases*. SIGMOD 2008.（SSI 算法本体）
- Fekete et al. *Making Snapshot Isolation Serializable*. ACM TODS 2005.（危险结构定理）

**怎么读这张图**：

- `Tin -rw-> Tpivot`：`Tin` **读**了某数据，`Tpivot` 随后**写**了它（`Tin` 的读“先于” `Tpivot` 的写，构成一条 read-write 反依赖）。
- `Tpivot -rw-> Tout`：同理，`Tpivot` 读、`Tout` 写。
- `Tpivot` 是**枢轴**：它既有一条入边、又有一条出边的 rw-反依赖。**任何 SI 下的序列化异常，其依赖环里必然包含这样一个 pivot**。这是 Fekete 2005 的定理。

write skew（§1.4）正是最简单的危险结构：两个事务**互相**读了对方要写的数据，`T1 -rw-> T2` 且 `T2 -rw-> T1`，两条 rw 边形成 2-环，`pivot` 就是它们自己。

### 5.2 PG 的工程优化：不是见到结构就回滚

如果“一出现两条相邻 rw 边就回滚”，会有大量**误杀**（不是所有危险结构都真的导致环）。PG 用两个定理收紧（README 原文已核实，逐字）：

> "Tout must commit before any other transaction in the cycle (see proof of Theorem 2.1 of [2]). **We only roll back a transaction if Tout commits before Tpivot and Tin.**"

> "if Tin is read-only, there can only be an anomaly if **Tout committed before Tin takes its snapshot**."

人话：

1. **提交顺序约束**：环要真正成环，`Tout` 必须**先于**环里其他事务提交。只有满足这个时序，PG 才回滚。这把很多“有危险结构但不成环”的情况放过了。
2. **只读优化（PG 原创贡献）**：若 `Tin` 是只读事务，仅当 `Tout` 在 `Tin` 拍快照**之前**就提交了才可能异常。这让大量只读事务**完全不会**触发序列化回滚——呼应 §3.4 官方那句“read-only transactions will never have serialization conflicts”。

### 5.3 代价与触发条件

SSI 需要追踪每个事务读过哪些“谓词区域”（PostgreSQL 用 **SIReadLock**——一种**不阻塞**任何人的“软锁”，只用于记录读足迹，不是真锁）。代价：

- **内存**：SIReadLock 记录可能很多；超限时退化成 page 级甚至 relation 级粒度（粒度越粗，误杀越多）。
- **回滚**：高冲突负载下序列化失败率上升，**应用必须实现重试**（§3.4）。

> **生产建议**：SERIALIZABLE 不是“无脑最安全选项”。它把“并发正确性”从应用层（手写锁、手写 `SELECT FOR UPDATE`）转移到了数据库层，换来代码简洁，但代价是**序列化失败重试**和**SIReadLock 内存/粒度**。适合“写偏序类约束多、且能接受重试”的场景；不适合“超高并发、长事务、读足迹巨大”的场景。

---

## 6. 方案对比与失败模式

### 6.1 MVCC 两大流派：in-heap 多版本 vs undo-log

PostgreSQL（新版本就地堆放）vs MySQL InnoDB（当前版本就地 + 历史版本在 undo log）。

【官方文档 mysql@dev，innodb-multi-versioning】，已 WebFetch：https://dev.mysql.com/doc/refman/8.0/en/innodb-multi-versioning.html

InnoDB 给每行加隐藏列（原文已核实，逐字）：

> **DB_TRX_ID (6 bytes)**: "indicates the transaction identifier for the last transaction that inserted or updated the row."
> **DB_ROLL_PTR (7 bytes)**: "The roll pointer points to an undo log record ... If the row was updated, the undo log record contains the information necessary to rebuild the content of the row before it was updated."

| 维度 | PostgreSQL（in-heap 多版本） | MySQL InnoDB（undo-log） |
|------|------------------------------|---------------------------|
| 旧版本存哪 | **就在 heap page 里**（新旧 tuple 并存） | 当前版本在聚簇索引；旧版本在 **undo log**，按 `DB_ROLL_PTR` 链上溯重建 |
| UPDATE 成本 | 写新 tuple + 所有索引都要新增条目（除非 HOT） | 原地改 + 写 undo；二级索引未变可不动 |
| 读旧版本 | 直接读 page 里的旧 tuple（无指针追逐） | **指针追逐** undo 链逐版本重建 |
| 表膨胀 | **bloat 严重**，强依赖 VACUUM | 当前数据紧凑；undo 可能膨胀 |
| 长事务伤害 | 老 tuple 无法回收 → 表无限膨胀 | undo 无法 purge → undo 表空间膨胀 + 历史链变长读变慢 |
| XID 回绕 | **有 wraparound 风险**（32 位 XID），需 freeze | 无此问题（实现不同） |

**具体场景跑一遍 —— “一个跑了 6 小时的报表事务”**：

- **PG**：这个老快照的 `xmin` 把全库 VACUUM 的“可回收水位”焊死 6 小时。期间所有被 UPDATE/DELETE 的行，旧版本**一个都不能删**。高写入表会急速 bloat，磁盘暴涨、顺序扫描变慢。`pg_stat_activity` 里这个 `xact_start` 很老的连接就是元凶。
- **InnoDB**：同样，这个 read view 让 purge 线程不敢清理它可能要读的 undo。**History list length** 飙升，二级索引上的“被改过但未 purge”的行让查询要回表追 undo 链，读放大。

**不适用边界**：

- **PG 的 in-heap 多版本** 不适合“超高频小更新 + 容忍长事务”的负载——bloat 会吃掉你。缓解靠 **HOT（Heap-Only Tuple）更新**（更新不涉及索引列时，新版本可复用同 page、不动索引）+ autovacuum 调优。
- **InnoDB 的 undo-log** 不适合“需要频繁读很老历史版本”的负载——undo 链越长，重建越慢。

### 6.2 隔离级别选择对照

| 级别 | 防住 | 防不住 | 适用 | 不适用 |
|------|------|--------|------|--------|
| READ COMMITTED | 脏读 | 不可重复读、幻读、lost update | 绝大多数 OLTP（默认） | 需要事务内一致快照的报表、read-modify-write |
| REPEATABLE READ (SI) | 脏读、不可重复读、幻读(PG) | **write skew**、部分 lost update | 事务内一致读、跨多语句报表 | 有跨行约束的并发写（排班/库存配额类） |
| SERIALIZABLE (SSI) | 全部，包括 write skew | —（但会序列化失败） | 写偏序类约束、要“当独占库写”的逻辑 | 超高并发 + 长事务 + 大读足迹（回滚率/内存爆） |

### 6.3 五个生产真坑（底层根因）

1. **SELECT 产生大量写 I/O**。根因：**hint bit 回写**（§2.1）。首次扫描刚导入的冷表，每个 tuple 确认 xmin 提交后回写 `HEAP_XMIN_COMMITTED`，弄脏 page → 触发 WAL full-page write + 刷脏。缓解：导入后主动 `VACUUM`（顺便冻结 + 写 hint），别让线上首查承受。

2. **表无限膨胀（bloat），VACUUM 追不上**。根因：**长事务**（老快照 `xmin` 卡住回收水位，§6.1）或 autovacuum 配置太保守。诊断：查 `pg_stat_activity` 找最老 `xact_start`；查 `n_dead_tup`。这是 PG MVCC 最高频生产事故。

3. **lost update 静默丢更新**。根因：RC 下 read-modify-write 的 EvalPlanQual 重评估（§3.3 官方例子）。表现：库存/计数器偶发对不上，无报错。正解：`SELECT FOR UPDATE` 或 `UPDATE ... SET x = x - 1 WHERE x >= 1`（让 DB 原子算）或升级 RR/SER。

4. **用了 RR/SERIALIZABLE 但线上偶发 `40001` 直接报错**。根因：**没实现重试循环**（§3.4 是契约要求，不是建议）。`could not serialize access ...` 必须捕获并整事务重试。

5. **XID wraparound 逼近，数据库进入只读保护**。根因：32 位 XID 即将回绕，autovacuum 的 freeze 没跟上（常因长事务 + 大量写阻塞 freeze）。后果极严重——PG 会强制停写以防数据被“未来”XID 误判可见。监控 `age(datfrozenxid)`，别等到 `autovacuum_freeze_max_age`。

---

## 7. 未来演进

- **去 32 位 XID / 64 位 XID**：wraparound 是 PG MVCC 的原罪之一。社区长期讨论 64 位 XID（彻底消除回绕），但牵动磁盘格式与升级路径，尚未落地主干。「具体进展凭记忆，待核」
- **更便宜的快照**：PG14 的 `GetSnapshotData` 重构（§2.3）只是开始；连接数继续上涨的趋势下，proc array 扫描、`ProcArrayLock` 争用仍是热点，可能向无锁/分段方向继续演进。
- **存储与 MVCC 解耦（zheap / pluggable storage）**：PG 的 table AM（Access Method）接口已让“非 heap 的存储引擎”成为可能。`zheap`（undo-log 风格，对标 InnoDB，意在消灭 bloat）是长期探索方向，但成熟度与是否进主干「待核」。
- **分布式下的隔离**：单机 SI/SSI 之外，分布式数据库（CockroachDB、TiDB、Spanner）在 MVCC + 时间戳排序（HLC / TrueTime）上做可串行化，是下一章级别的话题——核心思想仍是本章的“快照 + 冲突检测”，只是时间戳来源和冲突检测跨了节点。

---

## 8. 面试高频

1. **“隔离级别是怎么定义的？”** —— 反向定义：禁止哪些 phenomena（脏读/不可重复读/幻读/序列化异常），不是正向保证。能说出这点直接区分“背过 vs 理解”。
2. **“MVCC 一个事务为什么看不见另一个事务？”** —— snapshot 三件套 `xmin/xmax/xip` + tuple 的 `t_xmin/t_xmax`，可见性 = 插入者可见且删除者不可见（§2.4）。
3. **“RR 和 SI 是什么关系？”** —— PG 的 RR **就是** SI；SI 与标准 RR 不可比；PG 的 SI 实现连幻读都防了（§3.4）。
4. **“写偏序是什么，SI 为什么防不住？”** —— 两事务读对方要写的数据、写不同行、合起来破坏跨行约束；SI 只查写-写冲突，看不见读-写冲突（§1.4 + §4 demo）。
5. **“SERIALIZABLE 怎么实现的、代价是什么？”** —— SSI：检测两条相邻 rw 边的 dangerous structure，命中回滚；代价是序列化失败重试 + SIReadLock 内存（§5）。
6. **“为什么一个 SELECT 会产生写？”** —— hint bit 回写（§2.1），冷表首查的经典坑。
7. **“PG 表为什么会膨胀？”** —— in-heap 多版本 + 长事务卡住 VACUUM 回收水位（§6.1/§6.3）。
8. **“RC 下做 `UPDATE t SET n=n+1` 安全吗？lost update？”** —— `n=n+1` 这种让 DB 原子计算的安全；但“先 SELECT 读出值、应用层算、再写回”不安全（§3.3 官方例子）。

---

## 9. 章末五件套

### ① 一句话总结
隔离级别是“允许哪些并发异常”的契约；MVCC 用 `snapshot{xmin,xmax,xip}` × `tuple{t_xmin,t_xmax}` 实现“读不阻塞写”的快照可见性；SI 防不住 write skew，SSI 靠检测危险结构补上可串行化。

### ② 三个关键结论
- **可见性 = 插入者对我可见 ∧ 删除者对我不可见**，全部判断收敛到 `HeapTupleSatisfiesMVCC`（§2.4）。
- **PG 的 Repeatable Read 就是 Snapshot Isolation**，它超额防了幻读，但**防不住 write skew**——这是 SI 的根本局限（§1.4）。
- **SERIALIZABLE 不是免费**：SSI 用“两条相邻 rw 边”检测异常，代价是序列化失败重试；**用 RR/SER 必须实现重试循环**（§3.4 是契约）。

### ③ 一个反直觉点
**即使 CLOG 已记录某事务提交，只要它在“我拍快照那一刻”还在进行中，它的效果对我就永远不可见**——哪怕我跑了一小时它早提交了。这正是 §2.4 那个 `XidInMVCCSnapshot -> return false` 分支，是 RR“整事务一致快照”的根（很多人以为“提交了就该看见”，错）。

### ④ 一道思考题
> RC 下，事务 T 执行 `UPDATE accounts SET bal = bal - 100 WHERE id = 1`，与此并发另一事务把 `id=1` 的 `bal` 从 1000 改成 50 并提交。T 最终扣的是 1000-100=900 还是 50-100=-50？为什么？这暴露了 EvalPlanQual 的什么行为？如果 `WHERE` 改成 `WHERE id=1 AND bal >= 100` 又会怎样？

<details>
<summary>参考答案</summary>

扣的是 **50-100=-50**（在新版本上算）。RC 下 UPDATE 找到目标行后发现已被并发修改，会**等对方提交**，然后**跳到对方写的新版本**（bal=50）重评估并施加操作（§3.3 EvalPlanQual）。所以扣减基于的是**比 T 快照更新**的版本——这正是“UPDATE 不是在一个一致快照上跑”的体现，也是 lost update 类问题的温床。若 `WHERE` 加上 `bal >= 100`：重评估时新版本 bal=50 **不满足**条件，该行被**跳过**，UPDATE 影响 0 行——这又回到官方 `hits=10` 例子的陷阱（条件在重评估时不再成立而静默失效）。**正确做法**是用 `SELECT FOR UPDATE` 先锁、或直接让 DB 原子算、或升级到 RR 让冲突变成显式 `40001` 再重试。
</details>

### ⑤ 一道代码题（扩展本章 demo）
> 给 §4 的 toy 引擎加一个 **`demo_read_skew`（A5A 读偏序）**：约束“账户 X+Y 总和守恒”，初始 `X=100, Y=100`。事务 T1（RR）先读 X=100；并发事务 T2 把 100 从 X 转到 Y（`X=0, Y=200`）并提交；T1 再读 Y。要求：用 RR 跑出“T1 看到 X=100 且 Y=200（总和=300 ≠ 200）”这个**读偏序矛盾**，再解释**为什么 RR 实际上能避免它、而你的 toy 引擎能否避免**，找出 toy 引擎与真实 PG 的差异点。

<details>
<summary>实现提示 + 关键洞察</summary>

```python
def demo_read_skew():
    banner("扩展 A5A 读偏序：X+Y 守恒")
    e = Engine(); s = e.begin()
    e.insert(s, "X", {"v": 100}); e.insert(s, "Y", {"v": 100}); e.commit(s)
    t1 = e.begin(RR)
    x1 = e.read(t1, "X")["v"]                 # T1 读 X
    t2 = e.begin()                            # 并发转账 X->Y
    e.update(t2, "X", {"v": 0}); e.update(t2, "Y", {"v": 200}); e.commit(t2)
    y1 = e.read(t1, "Y")["v"]                 # T1 再读 Y
    e.commit(t1)
    print(f" RR: T1 见 X={x1} Y={y1} 总和={x1+y1}  期望守恒=200")
```

**关键洞察**：在**正确的 RR（事务级单快照）**下，T1 的快照在 T2 提交**之前**拍定，T2 的 `xmin` 落在 T1 的 `xip` 里，所以 T1 读 Y 时**看不到** T2 的写，仍读到 Y=100，总和=200，**守恒、无读偏序**——这正是 SI 防住 A5A 的机制（§1.4：SI 排除 A1/A2/A3，A5A 也被 SI 的“begin timestamp 单快照”天然防住）。

跑这个 demo 你会发现本章的 toy 引擎**正确地**输出守恒（Y=100），因为 `Txn.stmt_snapshot` 对 RR 返回同一张快照。**toy 与真实 PG 的差异点**留给你挖：(a) toy 的 `_take_snapshot` 用 `min(in_prog)` 近似 xmin，没处理“拍快照与事务注册之间的竞态”（真实 PG 用 `ProcArrayLock`，§2.3）；(b) toy 没有子事务 `subxip`；(c) toy 的 write-skew 检测是读写集相交的**过近似**，没建真实 rw-依赖图、不判 pivot 提交顺序（§5.2），会 false-positive 误杀。把这三点逐一对照 §2、§5 的源码，就完成了从“能跑的玩具”到“理解工业实现”的最后一跃。
</details>

---

## 附录：本章 WebFetch 取材清单（均实际取过）

| # | 来源 | URL | 用于 |
|---|------|-----|------|
| 1 | postgres snapshot.h（SnapshotData 结构） | raw.githubusercontent.com/postgres/postgres/master/src/include/utils/snapshot.h | §2.2 |
| 2 | postgres htup_details.h（tuple 头 + infomask） | raw.githubusercontent.com/postgres/postgres/master/src/include/access/htup_details.h | §2.1 |
| 3 | postgres heapam_visibility.c（HeapTupleSatisfiesMVCC） | raw.githubusercontent.com/postgres/postgres/master/src/backend/access/heap/heapam_visibility.c | §2.4 |
| 4 | postgres procarray.c（GetSnapshotData） | raw.githubusercontent.com/postgres/postgres/master/src/backend/storage/ipc/procarray.c | §2.3 |
| 5 | postgres README-SSI（dangerous structure） | raw.githubusercontent.com/postgres/postgres/master/src/backend/storage/lmgr/README-SSI | §5 |
| 6 | postgres snapmgr.c（snapshot 管理） | raw.githubusercontent.com/postgres/postgres/master/src/backend/utils/time/snapmgr.c | §2.2 |
| 7 | PostgreSQL 官方文档 transaction-iso | postgresql.org/docs/current/transaction-iso.html | §3 |
| 8 | PostgreSQL 官方文档 mvcc-intro | postgresql.org/docs/current/mvcc-intro.html | §2.5 |
| 9 | Berenson et al. 1995 Critique（HTML 版） | mwhittaker.github.io/papers/html/berenson1995critique.html | §1.2/§1.4 |
| 10 | MySQL InnoDB multi-versioning | dev.mysql.com/doc/refman/8.0/en/innodb-multi-versioning.html | §6.1 |

> 取材诚信声明：§2 的 PostgreSQL 源码块（snapshot.h / htup_details.h 的结构与 infomask、heapam_visibility.c 的函数体、README-SSI 的 ASCII 图与定理表述）均逐字取自上述 raw.githubusercontent.com 实际响应；procarray.c 的 `GetSnapshotData` 主循环标【示意，非逐字】（原函数含大量 recovery/优化分支，已抽主干并显式标注）；§4 demo 为本章原创、Python 3.12.12 实测可运行，输出为真实粘贴。凭记忆未独立核实之处（PG14 commit 号、64 位 XID / zheap 进展）已就地标「待核」。
