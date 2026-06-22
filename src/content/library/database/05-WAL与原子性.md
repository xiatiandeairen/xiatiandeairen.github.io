---
title: "WAL 与原子性:先写日志,掉电也不丢已提交的事务"
slug: "05"
collection: "database"
order: 5
summary: "在第 01 章的 Disk 抽象之上构建预写日志(WAL):记录格式(lsn/txid/before-after image)、append-only 顺序写、commit 时强制日志落盘、checkpoint 截断回放范围。本章用一份 1000 事务的 demo 证明『commit 只 fsync 日志、不要求数据页落盘』,并把崩溃后的中间态留给第 07 章的 ARIES 简化版恢复;fsync 时机的讨论复用第 02 章脏页刷盘那一套。"
topics:
  - "数据库"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

先把结论摆在最前面,因为它是这一章唯一要你刻进直觉的东西:**一个事务 commit 返回成功,磁盘上对应的数据页可能一个字节都没改**。它改的只有日志。如果你以为 commit 意味着"数据落盘了",这一章会用一份可跑的 demo 把这个错觉打碎——demo 里崩溃之后,数据页持久态停在初始值 `0`,而日志里已经稳稳记着那条已提交的修改。这不是 bug,这是 WAL(Write-Ahead Logging,预写日志)的设计本身。

为什么要这么反直觉?因为如果 commit 真的要求把每个被改的数据页都 `fsync`(强制刷盘)到磁盘,那事务吞吐会被随机写碾死——你改了 100 行散落在不同页上的数据,就要 100 次随机 IO + N 次 fsync。WAL 的整个经济账就是:**把"很多次散乱的数据页 fsync"换成"一次顺序的日志 append + 一次 fsync"**。代价是数据页落后于日志,这笔代价由崩溃恢复(第 07 章)来还。

本章配套代码在 `examples/database-from-scratch/src/stage05-wal.ts`,建在第 01 章的 `core/disk.ts`(Disk 抽象,带 `writePage`/`fsync` 计数和崩溃注入)和 `core/page.ts`(slotted page,带槽位行布局)之上。下面的所有数字都来自这份代码的真实运行,确定性的(字节数、fsync 次数)和墙钟实测的(吞吐)我会分开标。

## ✦ WAL 的黄金规则:日志先于数据落盘

整章就压在一条规则上,值得用一句大白话说死:

> **一个数据页在变成持久态(写进磁盘的数据文件并 fsync)之前,描述这次修改的日志记录必须已经持久态。** 简称 write-ahead:日志写在前面。

这条规则在代码里不是一句注释,而是 `runTxn` 方法用执行顺序硬性保证的。看 `stage05-wal.ts` 的 `Database.runTxn`:

```ts
runTxn(txnId: number, slot: number, newValue: number): number {
  const off = this.rowOffset(slot);
  const page = this.dataDisk.readPage(this.dataPageId);

  // before image: 即将被覆盖的那段字节(UNDO 的依据)
  const beforeImage = page.slice(off, off + ROW_BYTES);
  // after image: 同一段字节改成新行的样子(REDO 的依据)
  const afterImage = encodeRow([slot, newValue], ROW_SCHEMA);

  // 第 3 步:在碰持久态数据页之前,先把这次修改记进 WAL。
  this.wal.append({ kind: RecordKind.Update, txnId, pageId: this.dataPageId,
                    beforeImage, afterImage });

  // 第 4 步:才在内存里 apply。writePage 只脏页,没 fsync。
  page.set(afterImage, off);
  this.dataDisk.writePage(this.dataPageId, page);

  // 第 5 步:commit 记录 + (可能被 group commit 推迟的)flush
  const commitLsn = this.wal.append({ kind: RecordKind.Commit, txnId, /* ... */ });
  this.commitsSinceFlush++;
  return commitLsn;
}
```

注意顺序:**append 日志(第 3 步) → apply 内存(第 4 步) → commit 记录(第 5 步)**,数据页的 `writePage` 全程没有 `fsync`。数据页此刻是"脏页"——在内存(或 OS page cache)里改了,但磁盘上的数据文件还是旧的。

### 为什么这一条就同时保证了 D 和 A

这是这一章最该想透的地方,很多人会背"WAL 保证持久性"却说不清为什么。

**先看 D(Durability,持久性)。** commit 的语义被重新定义成:**commit = 把日志强制 fsync 落盘**(代码里是 `flushCommits` → `wal.flush` → `disk.fsync`),而**不要求**数据页落盘。一旦日志的 commit 记录落盘成功,就算紧接着掉电、数据页一个字节没写,恢复时也能从日志里读到这条 update 记录的 after-image,把修改重新做一遍(REDO,重做)。所以"已提交"等价于"日志里有它的 commit 记录且日志已落盘"——D 由日志的 fsync 兑现,不由数据页兑现。

**再看 A(Atomicity,原子性)。** 一个事务要么全做要么全不做。崩溃可能停在任意中间点。靠什么回到"全做"或"全不做"两个干净状态?靠日志里成对的 before/after image:

- 事务**已 commit**(日志里有 commit 记录)→ 它的所有修改用 after-image **REDO** 补齐,达到"全做"。
- 事务**未 commit**(崩溃时没写 commit 记录,但部分脏页可能已经溜到磁盘)→ 用 before-image **UNDO** 抹掉,回到"全不做"。

只要 write-ahead 规则成立——日志一定先于数据落盘——恢复就**永远有据可依**。反过来,如果允许数据页先落盘、日志后落盘(违反规则),就会出现"磁盘上有个改动但日志里没有对应记录"的状态:恢复时既不知道该 REDO 还是该 UNDO,这个改动属于哪个事务、该不该保留全都无从判断。这就是黄金规则的全部分量:**它把"崩溃后的任意中间态"约束成了"日志总是恢复的完整真相来源",A 和 D 才得以同时成立**。

before/after image 是物理日志(physical logging,记录字节级的页区域变化)的做法。记录格式定义在同文件的 `encodeRecord`,头部 15 字节固定:

```
u32 lsn        严格递增的日志序列号
u8  kind       记录类型(update / commit / abort / checkpoint)
u32 txnId      所属事务
u32 pageId     这条记录描述的数据页(非 update 类记录为 0)
u16 imageLen   每个 image 的长度 L
[L] beforeImage  改前字节(UNDO 源)
[L] afterImage   改后字节(REDO 源)
```

这里的 `lsn`(Log Sequence Number,日志序列号)和 before/after 语义不是本章自娱自乐——它们是第 07 章崩溃恢复的直接输入,第 07 章的 ARIES 简化版会照着这套格式逐条 redo/undo。

> 一个诚实的简化:这份 demo 记录的是"被改的那个槽位区域"(8 字节),不是整页 4096 字节。记整页也正确,但会让"平均每事务日志字节数"变成一个无意义的常数。记被改区域是真实引擎的做法(physiological/region logging),也让下面的字节数有信息量。

## WAL 的体量:一次事务到底写多少日志

跑 demo 的工作负载:1000 个小事务,每个事务 = 1 条 update 记录(含 before/after image)+ 1 条 commit 记录;数据模型是单个 slotted 数据页、100 行定宽行(每行 8B)。确定性输出(代码实算,可复现):

| metric | value |
|---|---|
| total log bytes | 46000 |
| avg bytes / txn | 46 |
| record header bytes | 15 |
| image bytes / update | 8 |
| durable commits | 1000 |

算一下账:每事务 46 字节 = update 记录(15 头 + 8 before + 8 after = 31)+ commit 记录(15 头 + 0 image = 15)。这就是 WAL 便宜的核心原因——**改一行数据,日志只多写几十字节,而且是顺序 append**;对照之下,数据页一改就是整页 4096 字节、随机位置。日志用"小且顺序"换掉了"大且随机"。

这里也藏着一个真实引擎要面对的 trade-off:image 越大(比如记整页),日志体量和 fsync 时要刷的字节就越多;image 越精细(只记 diff),编码/解码越复杂、变长记录的打包也更麻烦。本 demo 选了定宽区域这个中间点,把复杂度让给了第 07 章。

## ⚡ group commit:把多个事务的 fsync 攒成一次

到这里你该问一个尖锐问题:既然 commit = fsync 日志,那 1000 个事务不就是 1000 次 fsync?在真实磁盘上,一次 fsync 是毫秒级的物理往返(机械盘寻道、SSD 的 flush),1000 次 fsync 就是吞吐的天花板。这正是 WAL 性能工程的主战场。

**group commit(组提交)** 的思路:append 日志记录是廉价的纯内存操作,真正贵的是 `fsync`。那就让多个事务的 commit 记录先攒在内存里,**攒够一批再一次 fsync**——一次 fsync 就把整批 commit 一起兑现持久态。代码里这个开关就是 `maybeFlush(batch)`:

```ts
maybeFlush(batch: number): void {
  if (this.commitsSinceFlush >= batch) {
    this.flushCommits();   // 一次 fsync 覆盖整批
  }
}
```

一批攒 8 个,对照每事务都 fsync(batch=1),确定性输出:

| policy | walFsyncs | walWrites | dataFsyncs | dataWrites |
|---|---|---|---|---|
| no group (batch=1) | 1000 | 1011 | 0 | 1000 |
| group commit (batch=8) | 125 | 136 | 0 | 1000 |

两个数字要盯死:

1. **walFsyncs 从 1000 降到 125,减少 8.00x**。理论上界正好等于 batch=8;实际 grouped fsync = `ceil(1000/8) = 125`(125 个满批,末尾那次 `flushCommits` 把不满一批的尾巴也刷掉)。这个比例是确定性的、可验证的,不是估算。

2. **dataFsyncs 在两种策略下都是 0**。这是整章最该圈出来的一行:持久性**完全**来自强制日志,数据页在整个事务阶段被写脏(`dataWrites=1000`)却**从未** fsync。代码里有一条 invariant 守着这件事,跑挂了就报错:

```ts
invariant(
  noGroup.dataStats.fsyncs === 0 && grouped.dataStats.fsyncs === 0,
  "durability must come from the WAL fsync, not from data-page fsyncs",
);
```

吞吐这块要诚实标注——这是墙钟实测,但 demo 的"磁盘"是 RAM(没有真实 fsync 延迟),所以绝对值偏乐观,只看相对趋势:

| policy | ns/workload (measured) | txns/sec (measured) |
|---|---|---|
| no group (batch=1) | 1134100 | 881756 |
| group commit (batch=8) | 825256 | 1211745 |

吞吐提升 **1.37x (measured)**。这个数字小得有点"令人失望",但它恰恰诚实:RAM 盘上 fsync 几乎免费,group commit 只占到了"少写一个 page + 少调一次 fsync"的便宜。**在真实盘上 fsync 是毫秒级,加速比会远大于此**——这也是为什么 PostgreSQL、MySQL InnoDB 都默认开 group commit,而且把它当核心吞吐杠杆。

group commit 不是没代价:**它拿延迟换吞吐**。一个事务 commit 后要等本批攒满才真正落盘返回,单事务 commit 延迟变高(demo 里是同步填满,真实系统用一个短的等待窗口)。这是个经典取舍——重吞吐的 OLTP 选大 batch,重单事务延迟的场景选小 batch 或自适应。

### 前沿:从"组提交"到"日志即数据库"

group commit 是把 fsync 批处理这件事做到了单机的极致。再往前,有一条更激进的路线,我标成开放/前沿:

**"日志即数据库"(log is the database)。** 既然已提交状态的唯一真相来源就是日志,那为什么还要维护一份独立的数据页文件、还要费劲把脏页刷回去?Amazon Aurora 把这个想法推到了产品级:计算节点**只往存储层发日志记录**,不发数据页;数据页由存储层在后台用日志"自己长出来"。这把传统数据库"写数据页 + 写日志"的双写,简化成了"只追加日志流"。分布式日志系统(如把日志做成多副本 quorum 写)进一步让这条日志流既是持久化、又是复制、又是单一真相来源。

⚡ 这块**没有放之四海皆准的解**,仍是活跃研究/工程权衡区:把存储简化为日志流后,读路径要从日志"重建"页,读延迟和缓存策略怎么设计、日志的 GC(垃圾回收,清理已经物化进数据页的旧日志)节奏怎么定、跨副本的日志 quorum 一致性与延迟如何平衡——这些在不同系统(Aurora / Neon / 各家分布式 WAL)里答案都不一样,远没有收敛成教科书定论。本章只把单机的根(write-ahead + group commit)夯实,这条前沿留作你读完第 07 章后自己延伸的方向。

## checkpoint:别让恢复从盘古开天回放

只有 WAL 还有个隐患:日志会无限增长,崩溃恢复就得从第一条日志开始 redo,数据库跑得越久恢复越慢。**checkpoint(检查点)** 解决这个:周期性地把内存里的脏数据页真正刷到磁盘并 fsync,然后在日志里写一条 checkpoint 记录。恢复时可以从最近的 checkpoint 起回放,之前的日志不用再扫。

代码里 `checkpoint()` 同样守着 write-ahead 的顺序——先日志后数据:

```ts
checkpoint(): void {
  this.wal.append({ kind: RecordKind.Checkpoint, /* ... */ });
  this.wal.flush();        // checkpoint 记录先持久,才是合法的回放边界
  this.dataDisk.fsync();   // 然后才刷它声称"已稳定"的数据页
}
```

demo 跑前 50 个事务 + 1 次 checkpoint,然后从持久态(`reopenFromDurable`)读回采样槽位:

| check | value |
|---|---|
| slot 2 durable after checkpoint | 666110 |
| slot 2 expected committed | 666110 |

checkpoint 后数据页持久态 slot 2 = 666110,正好等于已提交值。对比下一节崩溃 demo 里"持久态停在 0",这里的差别就是 checkpoint 那次 `dataDisk.fsync()` 把脏页推下去了。意义:恢复时可从这个 checkpoint 起回放,无需扫描更早日志——**checkpoint 是用"现在多花一次数据页刷盘"换"将来崩溃恢复少扫一大段日志"**。fsync 时机这套讨论和第 02 章脏页刷盘是同一套权衡:刷太勤浪费 IO,刷太懒恢复变慢。

## 失败模式:崩溃后,数据页落后于日志

前面都在讲机制怎么对,现在看它怎么"坏"——而且这个"坏"恰恰是 WAL 故意允许的中间态。demo 注入崩溃:在 batch=1 下,WAL 第 3 次 fsync(提交事务 #2 时)抛 `CrashError` 模拟掉电。整个事务阶段**故意从不 fsync 数据盘**,所以数据文件的持久态停在初始全 0。崩溃后对比数据页的两个视图:

| view | slot 2 value |
|---|---|
| data page (in-memory / volatile) | 666110 |
| data page (durable after crash) | 0 |
| expected committed value | 666110 |

读这张表:txn #0 早已提交、它的 commit 记录也已 fsync 进日志,但数据页的**持久态 slot 2 = 0(初始值)**。内存里那个 `666110` 随掉电灰飞烟灭。这就是典型的 crash 中间态——**数据页落后于日志**:日志说"改过且已提交",磁盘上的数据却还没改。

代码用一条 invariant 钉死这个"故意的不一致"作为 demo 的前置条件:

```ts
invariant(
  durableValue !== txns[0].value,
  "demo precondition: data page must NOT have the committed value on disk",
);
```

这一步只**展示**中间态,不修复它。修复是第 07 章的活:

- **REDO**:重放日志里已提交事务的 after-image,把 `666110` 补回数据页——D 兑现。
- **UNDO**:回滚未提交事务的 before-image,抹掉那些溜到磁盘但没 commit 的脏改动——A 兑现。

这也是本书刻意维持的章节边界:第 05 章证明"commit 只强制日志、数据页可以落后",把这个崩溃后的不一致原样交出去;第 07 章再用同样的 lsn / before / after 语义把它收拾干净。两章之间的接口就是这份日志格式。

## 一句话收束

commit 不等于"数据落盘",commit 等于"日志落盘"。这个看似偷工减料的定义,靠 write-ahead 黄金规则同时买到了 A 和 D:日志永远是恢复的完整真相来源。group commit 把这套机制的 fsync 成本再摊薄一个数量级(本 demo fsync 减少 8.00x,真实盘上吞吐收益远大于这里的 1.37x),checkpoint 则把恢复的回放范围收住。下一章(第 06 章)在这套持久化地基上叠并发控制,第 07 章把本章故意留下的崩溃中间态用 redo/undo 真正恢复成一致状态。
