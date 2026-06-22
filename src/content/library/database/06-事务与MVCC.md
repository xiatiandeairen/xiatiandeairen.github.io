---
title: "事务与 MVCC:快照、版本链与四种隔离级别下的并发异常"
slug: "06"
collection: "database"
order: 6
summary: "在第 03/04 章的 B+ 树 / LSM 存储和第 05 章 WAL 的事务边界之上,用 core/scheduler 的确定性交错调度,从零搭一个多版本并发控制引擎:每行一条版本链、事务快照、可见性判定,实现 Read Committed / Repeatable Read / Snapshot Isolation 三档,并在 200 个种子上 sweep 调度、真实复现脏读 / 不可重复读 / 幻读 / 丢失更新 / 写偏序五类异常。终点是一张自检真值表和一个铁证:Snapshot Isolation 不等于可串行化——这正是下一章引入 SSI 的动机。"
topics:
  - "数据库"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

先把一个被无数博客文章糊弄过去的问题摆正:**隔离级别(isolation level)不是"锁的强度等级",而是"可见性规则的差异"。** 多数 survey 稿会给你抄一张 SQL 标准的表——"Read Committed 防脏读、Repeatable Read 还防不可重复读"——然后就没了。它不会告诉你:为什么 PostgreSQL 的 Repeatable Read 其实是 Snapshot Isolation,为什么 SI 挡得住幻读却挡不住写偏序(write skew),为什么这两件事的根因是同一个。这一章我们不抄表,我们**把这张表跑出来**:同一个 MVCC 引擎,只换三行可见性策略,在 200 个确定性交错上 sweep,看每个 (异常 × 级别) 单元格到底复现不复现。跑出来的表会和 SQL 标准的表长得不一样——那些不一样的地方,就是这章的全部价值。

本章的事务边界(哪个 txn 在什么时刻提交、提交时间戳是多少)沿用第 05 章 WAL 里的 txid 与 commit 记录概念;底下存的行,就是第 03/04 章那些 B+ 树 / LSM 里的 key。我们不重新发明存储,只在它上面加一层"同一个 key 存多个版本"。

配套代码:`examples/database-from-scratch/src/stage06-mvcc.ts`,以及它复用的 `core/clock.ts`(逻辑时钟)、`core/scheduler.ts`(确定性调度器)、`core/prng.ts`(种子随机数)。下面所有数字都来自这个 stage 的真实运行输出,不是估算。

## 一、MVCC 的一句话定义:读不返回"最新值",返回"对我可见的最新值"

先把 MVCC(multi-version concurrency control,多版本并发控制——同一行同时存多个历史版本)的核心规则说死,后面所有隔离级别都是它的特例。

传统的"读写都加锁"模型里,一个读必须等写锁释放——读写互相阻塞。MVCC 的整个出发点是:**让读永远不阻塞写、写永远不阻塞读。** 实现手段是:写操作不覆盖旧值,而是追加一个新版本;读操作不看"当前最新",而是沿着版本链往回走,返回"对我这个事务的快照可见的那个版本"。

在 `stage06-mvcc.ts` 里,一行就是一条版本链,每个版本带三个字段:

```typescript
interface Version {
  value: number;
  /** Commit timestamp; the version is visible iff committed before the reader's
   *  snapshot ts. undefined until commit() stamps it. */
  commitTs: number | undefined;
  /** Author txn id, so we can show a txn its own uncommitted write (read-your-
   *  own-writes) without exposing it to others. */
  authorTxn: number;
}
```

`commitTs === undefined` 是关键设计:它表示"这个版本被某个事务写了,但还没提交"。**未提交版本只对它自己的作者可见,对任何其他快照永不可见。** 这个状态被显式建模出来,不是为了好看,是为了能"证明"——脏读的定义就是"读到了别人未提交的版本",我们把这个未提交状态摆在那里,然后证明可见性规则永远不返回它。下面会看到真值表怎么验证这一点。

可见性判定的全部逻辑在 `read()` 里,一段循环就讲完了 MVCC:

```typescript
read(txn: Txn, key: Key): number {
  txn.readSet.add(key);
  if (txn.writeSet.has(key)) return txn.writeSet.get(key)!;  // read-your-own-writes

  // RC re-samples the snapshot on EVERY read; RR/SI reuse the frozen one.
  const effectiveTs = txn.level === "RC" ? this.clock.now() : txn.snapshotTs;

  const chain = this.chains.get(key);
  if (!chain) throw new Error(`read of unknown key ${key}`);

  // Walk newest-first; first committed version within the snapshot wins.
  for (let i = chain.length - 1; i >= 0; i--) {
    const v = chain[i];
    if (v.commitTs !== undefined && v.commitTs <= effectiveTs) return v.value;
  }
  throw new Error(`no version of ${key} visible at ts=${effectiveTs}`);
}
```

逐条拆这个规则:

1. **read-your-own-writes(读自己的写)**:事务先看自己缓冲区里的写,所以它能读到自己改过的值。这一步保证一个事务内部"写完能读到"的直觉成立。
2. **只看已提交版本** (`v.commitTs !== undefined`):别人未提交的版本(`commitTs === undefined`)被这个条件直接跳过——这就是脏读在结构上不可能发生的原因,**零成本,不靠任何锁或中止**。
3. **只看快照之前的版本** (`v.commitTs <= effectiveTs`):这是隔离级别的分水岭。`effectiveTs` 取什么,决定了你是哪一档。

时间戳从哪来?`core/clock.ts` 的 `LamportClock`——一个严格递增、不重复、不留空隙的逻辑计数器。为什么不用 `Date.now()`?注释写得很直白:MVCC 可见性是纯粹的**顺序**问题("T 能不能看到版本 V"只取决于谁在前谁在后),而墙上时钟既不单调也不可复现。逻辑时钟给出一个每次运行都完全一致的全序,这样调度器跑出来的并发异常才能被读者逐字节复现。这是整章"可复现"承诺的地基。

> 失败模式预警:`LamportClock` 的不变量是 `tick()` 严格递增、永不重复。如果两个版本拿到同一个时间戳,可见性规则就无法区分它们——这个 bug 的外在表现是"一次丢失更新,但看起来像正常行为"。`clock.ts` 的注释专门标注了这一点。时间戳唯一性不是细节,是 MVCC 正确性的前提条件。

## 二、三档隔离级别 = 同一个引擎换三行策略

现在看分水岭。三个隔离级别共用上面那一个版本库,区别只在两处:**读时 `effectiveTs` 取什么**,以及**提交时做不做写写冲突检测**。

### 2.1 Read Committed:每次读都重新取快照

```typescript
const effectiveTs = txn.level === "RC" ? this.clock.now() : txn.snapshotTs;
```

RC(Read Committed,读已提交)的 `effectiveTs` 是 `this.clock.now()`——**每次读都重新取"当前时刻"**。后果:事务开始后别人提交的新版本,RC 的下一次读就能看到。这一行,就是不可重复读和幻读的全部成因。RC 只保证"不读未提交",不保证"两次读一致"。

### 2.2 Repeatable Read:事务开始时冻结一个快照

RR(Repeatable Read,可重复读)用 `txn.snapshotTs`——这个值在 `begin()` 时冻结一次,整个事务生命周期不变:

```typescript
begin(level: IsolationLevel): Txn {
  return {
    id: this.nextTxnId++,
    level,
    snapshotTs: this.clock.now(),  // frozen at begin for RR/SI
    writeSet: new Map(),
    readSet: new Set(),
    aborted: false,
  };
}
```

后果:事务开始之后别人的提交一律不可见。同一个 key 读多少次都是同一个值——不可重复读消失。**这里有个反直觉但极其重要的事实:RR 用快照冻结同时也挡住了幻读。** 幻读(phantom)按 SQL 标准本该是 RR 挡不住、只有 Serializable 才挡得住的异常,因为它是"谓词范围内冒出新行"。但 MVCC 的快照是按时间戳冻结整个数据库视图的,新行的 `commitTs` 必然晚于我的 `snapshotTs`,所以我根本看不见它——**不靠谓词锁(predicate lock),靠快照就免费挡住了幻读。** 这就是为什么 PostgreSQL 把它的 SI 实现叫做 "Repeatable Read":在 MVCC 下,RR 顺手就比标准要求强了。这章的真值表会把这个"超标"用数据钉死。

### 2.3 Snapshot Isolation:RR 的快照 + 提交时写写冲突检测

SI(Snapshot Isolation,快照隔离)= RR 的可见性 + 一条提交时的检查。可见性和 RR 完全一样(冻结快照),唯一的增量在 `commit()`:

```typescript
commit(txn: Txn): void {
  if (txn.aborted) throw new Error(`commit on already-aborted txn ${txn.id}`);

  if (txn.level === "SI") {
    for (const key of txn.writeSet.keys()) {
      const last = this.lastCommit.get(key);
      // first-committer-wins: if a DIFFERENT txn committed this key after our
      // snapshot was taken (last.ts > snapshotTs), it ran concurrently and
      // touched the same row. Letting both commit would lose one update, so we
      // abort.
      if (last && last.byTxn !== txn.id && last.ts > txn.snapshotTs) {
        txn.aborted = true;
        throw new Error(`SI write-write conflict on ${key}`);
      }
    }
  }
  // ... stamp and append all buffered writes with one fresh commit ts
}
```

这叫 **first-committer-wins(先提交者胜)**:提交时检查我写的每个 key,如果有一个 key 在我取快照之后被别的事务提交过(`last.ts > snapshotTs`),说明有人和我并发改了同一行,放行两个都提交会丢掉一个更新——所以中止我自己。注意比较的是 `snapshotTs` 而不是"链上最新值":在我快照之前提交的版本是我已经看到并基于它工作的,不算冲突。

**这条检查是 SI 和 RR 之间唯一的区别**,也是这章后半部分要拆穿的那个洞的源头——请记住"per-key(逐 key)"这个词:它只在**同一个 key**上检测写写冲突。

### 2.4 一个隐蔽的设计决策:为什么所有 txn 必须提前 begin

代码里有个容易被忽略但极其要命的安排:harness 在任何操作运行之前,先给所有事务调一遍 `begin()`。注释解释了为什么:

```
WHY the harness calls begin() for ALL txns up front, before any operation runs:
a transaction's snapshot must reflect "what was committed when it started", and
concurrency anomalies only exist between txns whose lifetimes OVERLAP. ... the
first version of this stage had exactly that bug: SI "prevented" lost update only
because the second txn wasn't actually concurrent.
```

翻译一下这个**真实踩过的坑**:如果 `begin()` 懒到"生成器第一次被调度才执行",那么被调度器排在最后跑的那个事务,会在别人都提交完之后才取快照——它就成了"串行在后",而不是"并发"。结果是:这个事务的快照已经包含了对方的提交,冲突检测自然不触发,SI 看起来"挡住了"丢失更新——**但这是假象,因为两个事务根本没真正并发**。这个 bug 的可怕之处在于:它让你的测试全绿,却测的是错的东西。修法是提前 begin,把所有事务钉在同一个起始快照上,保证每个交错研究的都是货真价实的并发事务。

> 这个坑对应一条通用方法论:测并发正确性,N=1 的单次运行是不够的——单个种子可能恰好是"安全的那个交错"。`sweepScenario` 在 200 个种子上 sweep,就是为了把"这个级别根本禁止该异常"和"这个种子运气好没撞上"区分开。注释里记录了实测:种子数 = 1 时自检会**失败**(单个交错错过了异常,正是 N=1 陷阱);从 5 个种子起真值表和所有结构断言稳定,一直到 200 不变。代码保留 200 是图个宽裕余量(运行近乎瞬时)。

## 三、把真值表跑出来:五个异常 × 三个级别

理论讲完了,现在看证据。stage06 定义了五个场景,每个场景是一对(或多个)事务体加一个检测器(detector)。关键设计:**检测器检查的是真实运行后的状态,不是硬编码的"RC 允许脏读"这种断言。** 每个单元格是"观察到的",不是"声称的"。在 200 个确定性种子上 sweep,只要有一个种子触发了异常,就记为复现(YES);全部种子都挡住才记 no。运行 stage06 得到这张真值表:

```
anomaly              RC            RR            SI
-------------------  ------------  ------------  ------------
dirty-read           no  (50% ab)  no  (50% ab)  no  (50% ab)
non-repeatable-read  YES (0% ab)   no  (0% ab)   no  (0% ab)
phantom              YES (0% ab)   no  (0% ab)   no  (0% ab)
lost-update          YES (0% ab)   YES (0% ab)   no  (50% ab)
write-skew(on-call)  YES (0% ab)   YES (0% ab)   YES (0% ab)
```

括号里的数字是该级别在该场景下的**中止率(aborts/attempts)**,由调度器真实记录。这一列信息后面会反复用到——因为"挡住一个异常"有两种方式:靠快照结构挡(零中止),和靠中止事务挡(有中止代价)。这两种"挡住"的工程含义天差地别,真值表把它们区分开了。逐行拆。

### 3.1 脏读:三个级别全挡,而且零成本

脏读(dirty read)那一行,RC/RR/SI 全是 no。但注意那个 `50% ab` 的中止率,**它不是隔离级别为了挡脏读而中止**——脏读场景里 T1 故意"主动回滚"(永不提交,以制造一个未提交值),那个中止是场景自己造的,跟隔离级别无关。脏读靠 MVCC 结构挡,**零成本**。

这个场景的设计有个精妙处。看 `dirtyRead` 的 T1:

```typescript
function* T1(ctx) {
  yield "T1 write x=999 (will roll back)";
  ctx.write(h1, "x", 999);
  yield "T1 ... uncommitted, about to abort";
  throw new Error("T1 voluntary rollback");  // 999 never commits
},
detect: (_s, obs) => obs.reads.includes(999),
```

T1 写 `x=999` 然后**主动抛异常回滚**——999 从头到尾没成为任何已提交版本。检测器问:T0 有没有读到 999?因为 999 在任何交错下都没被提交过,obs 里要出现 999 的唯一可能,就是某次读把未提交缓冲暴露了出来——而 MVCC 永不暴露。所以这必须在每个级别都不可能。真值表的 no 在这里不是"声明 RC 禁止脏读",而是**证明了 MVCC 的结构保证**。

注释里还记了一个值得学的反例修法:早先的版本让 T1 最终提交 999,结果"提交后读到 999"是合法的,检测器分不清它和真脏读。改成"中止"才让检测器干净——**这是构造一个能区分真异常和合法行为的测试用例的典型难点**。

### 3.2 不可重复读 / 幻读:RC 复现,RR/SI 因快照冻结而消失

这两行长得一模一样(RC 是 YES,RR/SI 是 no),因为它们是同一个机制的两个版本——一个作用在单行,一个作用在谓词范围。

不可重复读的事务体:T0 读 x 两次,中间 T1 提交 `x=200`。RC 每次读重新取快照,两次读可能不同,异常触发;RR/SI 快照冻结,两次必然一致。

幻读用的是范围扫描。看 `rangeCount`——它就是把单行 `read` 套在一个谓词上数数:

```typescript
rangeCount(txn: Txn, keys: Key[], pred: (v: number) => boolean): number {
  let n = 0;
  for (const k of keys) if (pred(this.read(txn, k))) n++;
  return n;
}
```

场景初始 `a=50, b=0`(b 不在 `>0` 范围内),T0 数两次"余额 >0 的账户数",中间 T1 提交 `b=75` 让 b 进入范围(模拟一个 insert)。RC 第二次数会看到新行,count 变化 = 幻读;RR/SI 的冻结快照把它挡在外面。

这里要把上面 §2.2 那个反直觉点用代码再钉一次:**RR/SI 挡住幻读靠的是快照,不是谓词锁。** 真值表自检里专门有这条断言:

```typescript
invariant(cell("phantom", "RC"), "RC must admit phantom");
invariant(!cell("phantom", "RR"), "RR snapshot must prevent phantom");
```

注释直接写 "RC 复现,RR/SI 快照挡住(注意:靠快照而非谓词锁)"。这是 MVCC 相对于锁实现的免费午餐:谓词锁实现起来极其昂贵(要锁住"还不存在的行"),而 MVCC 用一个时间戳比较就达到了同等效果。代价我们留到第五节讲——天下没有真免费午餐,只是把成本挪到了别处。

### 3.3 丢失更新:RC/RR 都复现,SI 用 50% 中止率挡住

丢失更新(lost update)那行是真值表第一个真正有趣的对比:RC 和 RR 都 YES(都复现异常),只有 SI 是 no,而且代价是 `50% ab` 的中止率。

场景:两个事务都读 x,然后都写 `x = 读到的值 + 1`。串行执行结果应该是 102(两次 +1);丢失更新就是最后只有 101(一个 +1 凭空消失了)。RC/RR 没有写写冲突检测,后提交的覆盖先提交的 = 101,更新丢了。SI 的 first-committer-wins 会中止第二个写者,所以幸存者的 +1 被保留 = 没丢更新,**但付出了一次中止**(真实系统里中止的事务会重试)。

检测器这里有个精细之处,注释标了它是个修过的假阳性:

```typescript
detect: (s, obs) => obs.commits === 2 && s.groundTruth("x") < 102,
```

为什么要 `commits === 2`?因为 SI 下一个事务中止后 x 停在 101——**这是正确行为(中止的事务会重试),不是丢失更新**。只有当两个都成功提交、且 x < 102 时,才是真正丢了更新。要求"两个都提交"才把假阳性排除掉。这又是一个"区分真异常和合法行为"的细节,和脏读那个修法同源。

真值表自检对这行的断言很有意思——它不只检查 SI 挡住了,还检查它是**靠中止**挡住的:

```typescript
invariant(!siLost.everReproduced, "SI must prevent lost update");
invariant(siLost.abortRatePct > 0, "SI must prevent lost update *by aborting*");
```

`abortRatePct > 0` 这条断言把 SI 的工程代价显式化了:它和 RR 挡幻读那种"零成本快照"不是一回事。SI 防丢失更新是要付中止账单的——50% 的尝试被中止。在写冲突密集的负载下,这个中止率会直接变成吞吐崩塌和重试风暴。**选 SI 不是免费的,这个数字就是价签。**

## 四、专家钩子 ✦:为什么 Snapshot Isolation 不等于 Serializable

现在到这章的核心论点。真值表最后一行,write-skew(写偏序)在三个级别全是 YES,而且 SI 那列是 `YES (0% ab)`——**SI 既没挡住它,中止率还是 0**。这一格,就是"SI ≠ 可串行化(serializable)"的铁证。

很多人对 SI 有个错觉:它有快照、有写写冲突检测,听起来很强,是不是约等于最高级别 Serializable 了?**不是,而且差得很本质。** 写偏序是 SI 自己挡不住的一整类异常,根因就在 §2.3 强调过的那个词:**per-key**。

### 4.1 两医生请假:一个最小的反例

`writeSkew` 场景设计得像个谜题。两个值班医生 Alice 和 Bob 都在岗,医院规矩:**至少一人必须在岗**(`alice_on_duty + bob_on_duty >= 1`)。两人各自独立检查"对方还在不在岗?",在各自的快照里看到的都是"在",于是各自给**自己**请假:

```typescript
function* Alice(ctx) {
  yield "Alice read bob_on_duty";
  const bob = ctx.read(hAlice, "bob_on_duty");
  yield "Alice decide";
  if (bob === 1) {  // 我走也安全,因为 Bob 还在
    yield "Alice write alice_on_duty=0";
    ctx.write(hAlice, "alice_on_duty", 0);
  }
  ctx.commit(hAlice);
},
// Bob 对称:读 alice_on_duty,若=1 则给 bob_on_duty 写 0
```

关键:**Alice 写的是 `alice_on_duty`,Bob 写的是 `bob_on_duty`——不同的 key。** SI 的 first-committer-wins 是 per-key 的,它逐个检查"我写的这个 key 有没有被并发改过"。Alice 只写了 `alice_on_duty`,没人和她抢这个 key;Bob 只写了 `bob_on_duty`,也没人抢。两人的写集不相交,SI 看不到任何冲突,**两个都放行提交,0 中止**。

跑出来的真实交错(种子 seed=1,这是触发该异常的最小种子):

```
触发该异常的最小种子:seed=1
交错调度(每行一步,T0=Alice,T1=Bob):
  step  0  T0  start
  step  1  T1  start
  step  2  T1  Bob read alice_on_duty
  step  3  T0  Alice read bob_on_duty
  step  4  T1  Bob decide
  step  5  T1  Bob write bob_on_duty=0
  step  6  T1  Bob commit
  step  7  T0  Alice decide
  step  8  T1  commit
  step  9  T0  Alice write alice_on_duty=0
  step 10  T0  Alice commit
  step 11  T0  commit
两事务结果:commit, commit(无一中止——SI 放行)
```

注意 step 2 和 step 3:Bob 读 alice(=1),Alice 读 bob(=1)——**两人都在对方还没写之前读到了快照里的"在岗"。** 然后各自基于这个已经过时的快照做决策。最终已提交真值:

```
doctor      on_duty
----------  -------
Alice             0
Bob               0
SUM(约束>=1)        0
```

sum = 0,"至少一人在岗"被违反了,**两个事务都成功提交、零中止**。如果这两个事务串行跑——无论谁先——第二个都会在快照里读到对方已经请假(on_duty=0),从而不敢请假,约束就守住了。SI 放行的这个交错,**不等价于任何串行执行顺序**,这正是"非可串行化"的定义。

### 4.2 为什么 SI 结构上挡不住它

把根因说透:写偏序的本质是**两个事务各读了对方将要修改的数据(在快照里是旧值),各自的决策单独看都合法,合起来违反了一个跨行不变量(cross-row invariant)**。

SI 的冲突检测只覆盖"写写冲突"——同一行被两个并发事务写。它完全看不到"读写依赖":Alice 的决策依赖于她读到的 `bob_on_duty`,而 Bob 并发地把这个值改了。这种"我读的东西被你改了,但我读的不是我写的那个 key"的依赖关系,SI 的 per-key 检查盲区正中。代码注释把这点讲得很干脆:

```
this check is PER-KEY. Write skew slips through because the two txns write
DIFFERENT keys — neither sees a conflict on its own key — yet together they
violate a multi-row invariant. That gap is the chapter's punchline, not a bug.
```

真值表自检对这一格的断言,语气都带着强调:

```typescript
invariant(siSkew.everReproduced, "write skew must reproduce under SI");
invariant(siSkew.abortRatePct === 0,
  "write skew under SI must commit with zero aborts (the whole point)");
```

`abortRatePct === 0` 是整个论点的钉子:SI 不是"试图挡但偶尔漏",而是**结构上根本没有任何机制去碰它**,所以中止率精确为 0。这和上一节 SI 挡丢失更新的 50% 中止率形成尖锐对比——同一个 SI,对"同行写写"有 50% 中止的防御,对"跨行读写"是 0% 的完全放行。这个对比把 SI 的能力边界画得清清楚楚。

### 4.3 工程含义:你的应用里有多少个"两医生"

这不是教科书玩具。现实里大量业务约束是跨行不变量:

- **库存**:"已分配数量之和不能超过总库存"——两个订单各读到"还有货"各自下单。
- **排班 / 值班**:就是医生例子本身。
- **会议室 / 资源预订**:两个预订各读到"该时段空闲"各自占用,时间重叠。
- **账户余额联合约束**:"夫妻联名账户两个子账户余额之和 >= 0"——各自取款,单看都没透支,合起来透支。

只要你用了 SI(包括以为自己用了"Repeatable Read"的 PostgreSQL 用户),这些约束**都可能被写偏序悄悄破坏,而且数据库不会报任何错、不会中止任何事务**。这是最危险的一类 bug:它在低并发测试里几乎不出现(需要特定交错),上线后在高并发下偶发,且没有任何错误日志指向它。你只会发现"库存怎么变负了""怎么两个人订了同一间会议室"。

实务上的对策有三档,代价递增:

1. **物化冲突(materialize the conflict)**:把跨行不变量人为变成同行冲突。比如给"值班表"加一行汇总记录,每个请假事务都去写这一行——这样 SI 的 per-key 检测就能抓到冲突了。土,但有效,且不需要换数据库。
2. **显式加锁**:`SELECT ... FOR UPDATE` 把读变成加锁读,手动制造冲突点。要求开发者准确识别出哪些读是"危险读",容易漏。
3. **升级到 Serializable / SSI**:让数据库自动检测。这是下一节的前沿话题。

## 五、版本链的存储代价:快照不是免费的

第三节说 MVCC 挡幻读是"免费午餐",现在还账。免费的是 CPU(不用谓词锁),**不免费的是存储**。MVCC 保留旧版本,版本链会变长。stage06 在写偏序场景结束时打印了版本链长度:

```
版本链长度:alice=2, bob=2 (MVCC 保留旧版本,这是快照的存储代价)
```

每个 key 从初始 1 个版本变成 2 个——因为每次写都追加而不覆盖。`chainLength` 就是数链上有几个版本:

```typescript
chainLength(key: Key): number {
  return this.chains.get(key)?.length ?? 0;
}
```

这是 toy 规模(标注:绝对版本数不是 benchmark,这章测的是隔离正确性,不测吞吐和存储绝对值)。但**结构性结论是真的**:MVCC 的存储占用随"写次数 + 仍被某个活跃快照引用的旧版本数"增长。一个长事务(long-running transaction)持有一个老快照不提交,就会**钉住**所有它那个时间戳之后本可回收的旧版本——这些版本不能删,因为那个老快照还可能要读它们。这是 PostgreSQL 运维里臭名昭著的现象:一个忘了提交的事务,能让整个库的旧版本无限堆积(对应 PostgreSQL 的 dead tuple 膨胀,bloat)。

所以 MVCC 引擎必须有一个**垃圾回收(garbage collection,回收旧版本)**机制,把"已经没有任何活跃快照能看到的旧版本"清掉。PostgreSQL 叫它 `VACUUM`,MySQL InnoDB 叫 purge。回收的判定规则:一个旧版本可删,当且仅当没有任何活跃事务的 `snapshotTs` 落在"该版本被覆盖"之前。

> stage06 没有实现 GC——它的所有场景都是短事务,跑完即弃,版本链不会真的膨胀。这是有意的范围裁剪:这一章的使命是讲清隔离级别的可见性语义,GC 是一个独立的、足够复杂的子系统。但你要知道这个洞在那里:**生产级 MVCC 引擎,可见性规则只是冰山一角,版本回收的工程量和坑往往比可见性本身还大。**

## 六、前沿与开放问题 ⚡:SSI 与 HTAP 下的 GC 成本

这一章到 SI 为止,写偏序的复现正是引入下一步的动机。把当前还没有"通用免费解"的两个方向标出来。

### 6.1 Serializable Snapshot Isolation:能不能既要快照的高并发又要可串行化

写偏序的根因是 SI 看不到读写依赖。**SSI(Serializable Snapshot Isolation,可串行化快照隔离)** 的思路:在 SI 之上额外追踪事务间的**读写反依赖(rw-antidependency)**——"我读了一个值,而另一个并发事务后来改了它"。SSI 把这些依赖建成一张图,检测其中的**危险结构**(具体是"两条 rw-antidependency 首尾相接形成的特定模式",对应理论上必然出现非可串行化执行的拓扑),一旦发现就中止其中一个事务。stage06 的结论一句话点了它:

```
要堵这个洞需要 SSI(serializable snapshot isolation):
额外追踪「读-写依赖」(rw-antidependency),发现危险结构时中止其一。
```

PostgreSQL 9.1 起的 `SERIALIZABLE` 隔离级别就是 SSI 的工业实现(基于 Cahill 等人 2008 年的论文),也是迄今唯一在主流数据库里大规模落地的真·可串行化的乐观方案。它的吸引力在于:**保留了 SI 读不阻塞写的高并发,又给出了可串行化保证**,代价只是一些"误杀"(false positive)——危险结构的检测是保守的,会中止一些其实安全的事务。

**⚡ 为什么这是开放问题而非已解决:** SSI 没有通用最优解,核心矛盾是"追踪精度 vs 开销"。要追踪 rw-antidependency,就得记录每个事务读过哪些行(read set),在高并发下这个 read set 的内存和锁开销可能很大,而且粒度难权衡——按行追踪精确但贵,按页 / 按谓词追踪便宜但误杀率飙升。误杀率高了,SSI 就退化成"一堆事务反复中止重试",吞吐还不如直接上锁。如何在不同负载形态下自适应地选追踪粒度、把误杀率压到可接受范围,**目前仍是活跃研究方向,没有一个对所有 workload 都优的配置**。这也是为什么很多生产系统宁可停在 SI + 手动物化冲突,也不默认开 Serializable——它们怕的不是正确性,是那个不可预测的中止率。

### 6.2 HTAP 引擎里 MVCC 版本链的 GC 成本

第二个开放问题接着第五节的 GC 往深里走。在 **HTAP(Hybrid Transactional/Analytical Processing,事务与分析混合负载)** 引擎里,版本链 GC 成为一个特别尖锐的矛盾。

矛盾在于:HTAP 要在同一份数据上同时跑 OLTP(高频短事务,飞快地产生新版本)和 OLAP(长时间的大范围分析查询,需要一个稳定的老快照扫几分钟甚至几小时)。OLAP 那个长查询持有的老快照,会**钉住海量旧版本不让回收**——正是第五节说的长事务问题,在 HTAP 下被放大到极致。版本链越拖越长,OLTP 这边的读要沿着长链往回走找可见版本,延迟也跟着涨。

**⚡ 为什么这是开放问题:** "什么时候回收、回收谁、回收的开销摊到谁头上"在 HTAP 下没有通用解。回收太激进,会破坏正在跑的 OLAP 查询(它要的老版本被删了);回收太保守,版本链膨胀拖垮 OLTP 读性能和存储。学术界和工业界在试各种方向——把旧版本异步迁到列存做分析、按版本年龄分层存储、用 epoch-based(基于纪元的)批量回收降低单次开销、甚至给 OLAP 单独维护一份物化的历史快照——但每种方案都是在"OLTP 延迟 / OLAP 一致性 / 存储成本 / 回收开销"这个四维空间里挑一个角落,**没有一个方案能同时在四个维度都赢**。这是当前 HTAP 引擎设计里最硬的权衡之一,仍在活跃演进。

## 小结:这章你真正应该带走的

不是那张 SQL 标准隔离级别表——那个到处都能抄到。是这几条用代码和数字钉死的判断:

1. **隔离级别是可见性规则的差异,不是锁强度等级。** 三档级别在 stage06 里就是 `effectiveTs` 取 `now()` 还是 `snapshotTs`,加上提交时做不做 per-key 写写检测——一共改三行。
2. **MVCC 下脏读和幻读都是结构性零成本挡住的**(真值表里 RR/SI 那两行的 0% 中止),靠快照不靠锁。这是 MVCC 对锁实现的核心优势。
3. **"挡住异常"有两种,代价天差地别**:快照结构挡(0% 中止)vs 中止事务挡(SI 防丢失更新的 50% 中止)。真值表的中止率列就是用来区分这两者的。
4. **SI ≠ Serializable,根因是 per-key 写写检测看不到跨行读写依赖。** 两医生 seed=1 的交错,两个事务都提交、0 中止、约束被破坏——这是铁证,不是声称。你的库存 / 排班 / 预订业务里全是潜在的"两医生"。
5. **快照不免费,代价在存储和 GC**。长事务钉住旧版本是 MVCC 的阿喀琉斯之踵,HTAP 下尤其要命。

下一章引入 SSI:在 SI 上追踪读写反依赖、检测危险结构、自动中止——把这章 seed=1 那个写偏序,变成数据库自己抓得住并中止的冲突。我们会把同一个两医生场景再跑一遍,看 SSI 这次会不会中止其中一个,以及它要为此付出多少误杀率。
