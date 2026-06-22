---
title: "最终一致与 CRDT：让并发写自动收敛"
slug: "07"
collection: "distributed"
order: 7
summary: "第 6 章用 LWW 收 AP 留下的并发写冲突时丢了一个写——这章不丢。从零手写 CRDT（无冲突复制数据类型）G-Counter / OR-Set，证明合并满足交换律 / 结合律 / 幂等，于是乱序、重复、分区下副本必然收敛到同一状态，无需共识、无需协调。对照第 5 章 Raft（强一致靠协调），CRDT 是「弱一致但永远可写」的另一条路；它把冲突解决从业务逻辑变成数学保证，但代价是只能表达能构造成半格的数据类型，不是万能。"
topics:
  - "分布式系统"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

第 6 章末尾我们埋了一笔债。当时为了在分区下保持「永远可写」（AP），我们用 LWW（Last-Writer-Wins，谁的时间戳大谁赢）收拾并发写：两个副本同时改同一个 key，合并时按 `(timestamp, replica)` 比大小，**小的那个写直接扔掉**。本章这个 stage 把那笔债的代价量化了出来——同一组并发操作，LWW 终态是 `{memo,report}`，正确的 CRDT 终态是 `{doc,memo,report}`：

> `LWW 用 (ts=100, replica) tiebreak 丢掉了 r1 对 "doc" 的 add（保留 r2 的 remove）；OR-Set 保留 doc=true`

`doc` 这个写凭空蒸发了，而且没有任何报错——用户以为自己保存了文档，系统静默地把它丢进了垃圾桶。这章要回答的就是一个问题：**有没有一种数据类型，并发写永远不丢，且不需要任何协调（无 leader、无锁、无投票）就能让所有副本最终长成一模一样？** 有，叫 CRDT（Conflict-free Replicated Data Type，无冲突复制数据类型）。它不是魔法，背后是一条很硬的数学约束。我们从零写出来，并用上千次随机合并把这条约束证给你看。

配套代码：`examples/distributed-from-scratch/src/stage07-crdt-convergence.ts`，复用 `core/node.ts`（节点基类）、`core/network.ts`（可注入丢包 / 乱序 / 分区的模拟网络）、`core/scenario.ts`（确定性场景驱动）、`core/prng.ts`（种子随机数）。下面所有数字都来自 `seed=7` 的真实运行，可复现。提醒一句：时延数字是 `SimClock` 虚拟毫秒（相对趋势可迁移到真实系统，绝对值不是 wall-clock）；真正承重的是那些**关系型事实**——「所有合并顺序结果相同」（一个布尔值，跑了 500 次随机试验）、「收敛要几轮 gossip」（实测计数）、「LWW 丢一个写而 CRDT 不丢」（两个终态的真实 diff）。

## 一、收敛的代价定理：合并必须是半格上的 join

先把结论摆出来，再倒推为什么。

**CRDT 收敛的充分条件**：如果每个副本的状态构成一个 **join-semilattice（并半格）**，并且 `merge` 是这个半格上的 **join（上确界，least upper bound）**，那么任意副本以任意顺序、任意次数交换状态并合并，只要彼此最终都看过对方的更新，就必然收敛到同一个状态。

半格（semilattice）一句话：一个偏序集合，里面任意两个元素都有一个唯一的「最小上界」。把这个上界运算（join，记作 ⊔）展开，等价于三条代数律：

- **交换律（commutative）**：`a ⊔ b = b ⊔ a`——谁先合并谁后合并无所谓 ⇒ **抗乱序**。
- **结合律（associative）**：`(a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)`——怎么分组合并无所谓 ⇒ **抗任意拓扑**（gossip 的星型、链式、随机图都行）。
- **幂等（idempotent）**：`a ⊔ a = a`——同一个状态合并两次等于一次 ⇒ **抗重复投递**。

✦ 这就是 CRDT 的全部底牌，也是它和「业务层冲突解决」的本质区别。普通的冲突解决（比如「弹个窗让用户选保留哪个版本」「按时间戳挑一个」）是**把语义判断写进业务逻辑**，对错取决于程序员有没有想全所有并发情形——而并发情形是组合爆炸的，想不全就静默出错（LWW 丢写就是这么来的）。CRDT 反过来：它要求你把数据类型设计成一个半格，于是「合并两个分叉的状态」退化成「求两个元素的上确界」——这是个**纯数学运算，没有歧义、没有遗漏**，对错由半格性质保证，不由业务代码保证。**冲突解决从此不是逻辑问题，是代数问题。**

代价也在这里，必须讲清楚：**不是所有数据类型都能塞进半格。** 半格要求 join 满足上面三条律。计数器、集合、寄存器能找到合适的半格表示；但「任意可变结构 + 任意操作」很多时候找不到——尤其是带**顺序**和**唯一性约束**的（一个有序列表、一个不能有重复的购物车数量上限），构造对应的 CRDT 要么极其精巧，要么根本做不到。所以 CRDT 不是「分布式数据的银弹」，它是「**能被建模成半格的那部分数据**的银弹」。这个边界后面（第七节）还要再撞一次。

下面我们把这条定理从抽象变成能跑的代码，从最简单的计数器开始。

## 二、G-Counter：把「+1」拆成「谁 +1」，让 max 成为 join

最朴素的想法：分布式计数器就是一个共享整数，谁要加就 `count++`。这在单机没问题，分区下立刻坏掉。看 `stage07` 注释里点破的失败模式：

```ts
// 引自 src/stage07-crdt-convergence.ts，GCounter 的设计注释
// Why per-replica entries instead of one shared integer: a single `count++`
// applied on two partitioned replicas and naively merged would either double-
// count or lose an increment — there is no way to tell "we both saw 5 then both
// +1 → should be 7" from the scalar 6 alone.
```

**怎么坏的**：两个副本都看到 `count=5`，分区期间各自 `+1`，各自变成 `6`。分区恢复要合并这两个 `6`。你怎么合？取 max 得 `6`（丢了一个加）；相加得 `12`（多算了 5 这个共同基数）；取任意一个还是 `6`（丢一个）。**根因**：一个标量 `6` 里，分不清「这是我俩共同看到的 5 之上各加了 1」还是别的历史——标量丢失了「这次增量是谁贡献的」这个信息，而合并恰恰需要这个信息。

**为什么这么设计**：G-Counter（Grow-only Counter，只增计数器）的修法是把计数**按作者拆开**。状态不是一个整数，是一张 `replica → 该副本的累计贡献` 的表。观测到的总数 = 所有贡献求和。关键不变量：**每个副本只写自己那一格**，所以它自己那格永远是最新的、别人手里的拷贝永远 ≤ 它——于是合并时取 max 就对了（落后的拷贝被领先的真值覆盖，不会丢、不会重）。代码就一行 join：

```ts
// 引自 src/stage07-crdt-convergence.ts，GCounter
/** 本副本只写自己那格的贡献；entry[r] 只由 replica r 写，所以从不需要协调。 */
increment(who: string, by = 1): void {
  if (by < 0) throw new Error("GCounter is grow-only; got negative increment");
  this.contributions.set(who, (this.contributions.get(who) ?? 0) + by);
}

/** 最小上界 = 逐格取 max。max 满足交换 / 结合（max 本身的性质）、
 *  幂等（max(a,a)=a）。这一行就是整个 CRDT。 */
merge(other: GCounter): void {
  for (const [who, count] of other.contributions) {
    this.contributions.set(who, Math.max(this.contributions.get(who) ?? 0, count));
  }
}
```

为什么 max 是这里的 join？因为每个副本的贡献是**单调递增**的（grow-only），单调序列上「最新值」就是「最大值」。逐格取 max = 在「贡献向量」这个偏序（逐分量比较）上求上确界。三条律自动满足，因为它们都是 `Math.max` 自带的。

stage07 的 Demo A 不是嘴上证，是**实测证**：构造 4 个分叉的副本状态（各自 `+1..5` 自己那格），然后把它们丢进 `merge` 跑 **500 次随机顺序 + 随机重复**，看是否所有顺序都收敛到同一个指纹：

```
## A. G-Counter：并发自增，任意顺序合并均收敛到同一和（且不丢不重）
  trials=500  distinctOutcomes=1  converged=true
  收敛后 sum=11  应有总增量=11  ✓ 无丢失/重复计数
```

`distinctOutcomes=1` 是承重数字：500 种随机合并顺序（夹杂随机重复合并），最终指纹**只有 1 种**——交换律和幂等被这 500 次试验当场验证。`sum=11` 等于「实际应有的总增量 11」，正是「不丢不重」的算术证据。这里我特意跑了 500 次而不是 1 次：N=1 的「碰巧对」区分不了「合并真的可交换」和「这次随机顺序刚好没踩到 bug」；500 次全收敛才是 signal。

> PN-Counter（支持减法的计数器）怎么办？grow-only 不能减。标准做法是塞两个 G-Counter，一个记所有增、一个记所有减，观测值 = 增和 − 减和。本 stage 的输出聚焦在 G-Counter 和集合上，PN-Counter 的「双 G-Counter」构造留给你按同样的半格思路推。

## 三、OR-Set：用 dot 和 tombstone 让「并发的加」赢过「并发的删」

计数器只增很好办，集合要支持删就难了。难点是经典的 **add/remove 并发**：副本 A 删 `x` 的同时，副本 B 加 `x`（B 根本没看到 A 的删，A 也没看到 B 的加），合并后 `x` 该在还是不在？

这里没有「客观正确答案」，只有「一致的策略」。OR-Set（Observed-Remove Set，观察删除集合）选 **add-wins（加赢）**：并发的加和删，加保留。直觉理由：删除是「移除一个我观察到的东西」，而并发的加是「一个删除者从没见过的新东西」——删除者无权删一个它没观察到的加。

实现的精髓在于 `remove` **不删元素本身**。它给元素的每次 `add` 打一个全局唯一的 **dot**（点：`replica#seq`，标记「谁在它的第几次本地操作时加的」）；`remove` 只把它**当前观察到的那些 dot** 记进 tombstone（墓碑，标记某个 dot 已死）。一个删除者没见过的并发 add 会产生一个新 dot，不在它的观察范围里，所以合并后那个 dot 仍然活着 ⇒ 元素保留：

```ts
// 引自 src/stage07-crdt-convergence.ts，OrSet
/** 本地删：只 tombstone 本副本此刻观察到的 dot。没见过的（并发远程 add）不动
 *  → 那个 add 在合并时赢。observed-remove 就这一个方法。 */
remove(element: string): void {
  const dots = this.adds.get(element);
  if (!dots) return; // 删一个从没观察到的元素是 no-op，不是错误
  for (const dk of dots) this.removedDots.add(dk);
}

/** merge = add-dots 的并集 ∪ tombstone 的并集。两者都是只增集合，所以是 join：
 *  交换 / 结合 / 幂等。元素是否存在再从「有未被 tombstone 的 dot」推导出来。 */
merge(other: OrSet): void {
  for (const [el, dots] of other.adds) {
    const mine = this.adds.get(el) ?? new Set<string>();
    for (const dk of dots) mine.add(dk);
    this.adds.set(el, mine);
  }
  for (const dk of other.removedDots) this.removedDots.add(dk);
  // ...seqByReplica 取 max，保证本副本未来的 add 仍铸出唯一 dot
}
```

注意这个 merge 为什么是 join：`adds`（dot 集）和 `removedDots`（tombstone 集）**都是只增集合**，合并就是求并集——并集运算天然交换 / 结合 / 幂等。元素的「存在性」不直接存，而是从「有没有未被 tombstone 的活 dot」**派生**出来。这正是 add-wins 自动成立的原因：并发 add 的新 dot 只进 `adds`、从未进 `removedDots`，所以推导时它必然是活的。

stage07 Demo B 的构造就是这个经典 case：r0 加了 `x,y`；r1 在看到 `x` 后删 `x`；r2 **并发**地又加了一个 `x`（新 dot，r1 没见过）；r3 加 `w`。500 次随机顺序合并：

```
## B. OR-Set（add-wins，带 dot/tombstone）：并发增删，任意顺序+重复合并均收敛
  trials=500  distinctOutcomes=1  converged=true
  收敛终态 value={w,x,y,z}  （x 被 r1 删、被 r2 并发重加 → add-wins，x 保留）
```

`distinctOutcomes=1` 再次锁死收敛；终态 `{w,x,y,z}` 里 `x` 活着——r1 的删只杀了它观察到的那个 dot，r2 的新 dot 没被波及。这就是 add-wins 落地的样子。

> OR-Set 的代价不藏着：tombstone 只增不减。删过的 dot 永远留在墓碑里，集合用得越久墓碑越多，状态膨胀。生产级实现要配「因果稳定后回收 tombstone」（确认所有副本都看过这个删，就能安全清掉墓碑）——这本身又是个需要因果追踪的子问题，不在本 stage 的 toy 范围里，知道有这个坑即可。

## 四、失败模式：伪 CRDT 的 remove 不可交换，副本永久发散

前三节都在讲「对的做法」。这节讲**错的做法长什么样、怎么坏、为什么**——这是本书区别于 survey 稿的地方：survey 告诉你 OR-Set 怎么写，不告诉你「少了 tombstone 会发生什么」，于是你照抄一个 happy-path 能跑的版本上线，分区一来副本就裂开。

伪 CRDT 的诱惑很大：「不就是个集合吗，`add` 就 `set.add`，`remove` 就 `set.delete`，merge 就求并集呗。」问题在 `remove` 那个 `delete`——它**按元素名销毁**，不记 tombstone、不管 dot。这就破坏了交换律。

stage07 用一个 **op-based（操作复制，副本之间传操作而不是传状态）** 视角把发散逼出来，这是有意为之的（注释里写明了 why）：如果用 state-based 求并集，伪 CRDT 只会静默收敛到**错误值**（被删的元素复活），两个副本反而看起来「一致」了——掩盖了问题。真正尖锐的失败是**两个副本收敛到不同的集合**，这只在 op-based 下暴露：副本按到达顺序逐个 apply 远程操作，一个没有 tombstone 的破坏性 delete 会产生「依赖到达顺序」的结果。

场景是教科书级的因果切口：共同祖先加了 `x`（dot=`ox`，两副本都见过）。然后并发地——ra 删它观察到的 `x`（操作 = 删 dot `ox`）；rb 又加一次 `x`（操作 = 加新 dot `rx`）。两个操作最终都送达两个副本，但**到达顺序相反**：

```
## C. 失败模式：伪 CRDT（remove 只删元素、不记 tombstone）op 不可交换 → 发散
    op-based 复制：共同祖先 x（dot=ox）；ra 收到顺序[删ox, 加rx]，rb 收到顺序[加rx, 删ox]
  BadOrSet   ra 终态={x}  rb 终态={}  diverged=true
    └─ 「A 删 / B 加」乱序下 ra 有 x、rb 没 x → 永久发散（remove 把 x 整个删掉，与 add 不可交换）
  正确 OR-Set ra 终态={x}  rb 终态={x}  converged=true
    └─ remove 只 tombstone 它观察到的 dot(ox)；rx 未被删 → 两副本均收敛到 {x}（add-wins）
```

把两条路径手算一遍就懂了。伪 CRDT 的 remove 是按名删：

- ra 收到 `[删x, 加x]`：先 `delete("x")`（集合空），再 `add("x")` ⇒ 终态 `{x}`。
- rb 收到 `[加x, 删x]`：先 `add("x")`（已有 x），再 `delete("x")` ⇒ 终态 `{}`。

`delete("x")` 然后 `add("x")` ≠ `add("x")` 然后 `delete("x")`——**操作不可交换**，到达顺序决定结局，两副本 `{x}` vs `{}` 永久分裂（`diverged=true`）。注意这不是「暂时不一致、等会儿就好了」：所有操作都已送达双方，系统已经 quiesce（静止），它们**就是不一样了**，再 gossip 一万轮也不会一样。这是最阴险的分布式 bug：没有报错，数据就是悄悄裂了。

而正确 OR-Set 的 remove 是按 dot 删：删操作只 tombstone `ox`，而 rb 加的是新 dot `rx`，`rx` 从未被 tombstone。无论先 apply 删还是先 apply 加，活 dot 集都是 `{rx}` ⇒ 两副本都收敛到 `{x}`（`converged=true`）。**dot 让 remove 携带了精确的因果上下文，于是操作可交换了。** 这就是那一堆看似啰嗦的 dot / tombstone 记账的全部意义——它不是为了优雅，是为了让 merge 落进半格、让发散在数学上不可能。

教训钉死：**「能跑通 happy path」和「是 CRDT」是两回事。** 判定一个类型是不是 CRDT，不看它演示能不能跑，看它的 merge / op 满不满足交换 + 结合 + 幂等。少一条，分区一来就裂。

## 五、对照第 6 章 LWW：LWW 也是 CRDT，但它的代价是丢写

现在回到开篇那笔债。这里要讲一个反直觉的点：**LWW 其实也是合法的 CRDT。** `(timestamp, replica)` 上的全序构成一个半格，「取较大者」是合法的 join，满足三条律——所以 LWW 副本**确实会收敛**，不会像伪 CRDT 那样发散。

那 LWW 错在哪？它没错在「不收敛」，错在**收敛的方式是扔掉一个并发写**。stage07 Demo D 把 LWW 和 OR-Set 放进同一组并发操作里对照：r1 加 `doc`、r2 并发删 `doc`，时间戳故意撞成相等（`ts=100`，模拟「真并发，无因果先后」）：

```
## D. 对比第 6 章 LWW：并发写「doc」时 LWW 丢一个写，CRDT 不丢
  LWW-Set  终态={memo,report}
  OR-Set   终态={doc,memo,report}
  LWW 丢失的写: LWW 用 (ts=100, replica) tiebreak 丢掉了 r1 对 "doc" 的 add（保留 r2 的 remove）；OR-Set 保留 doc=true
```

LWW 的 tiebreak 逻辑（引自代码）把这点暴露得很清楚：

```ts
// 引自 src/stage07-crdt-convergence.ts，LwwSet
private dominates(a: LwwEntry, b: LwwEntry): boolean {
  // (tsMs, replica) 的全序确定性打破平局——没有 replica tiebreak，两个同时间戳
  // 的写就不可比，LWW 就不是函数了。tiebreak 是任意的但确定的，
  // 这恰恰就是它静默丢掉两个并发写之一的方式。
  if (a.tsMs !== b.tsMs) return a.tsMs > b.tsMs;
  return a.replica > b.replica;
}
```

时间戳相等时，靠 `replica` 字典序硬分高下：`"r2" > "r1"`，于是 r2 的删赢、r1 的加被丢。结果 LWW 终态 `{memo,report}` 缺了 `doc`；OR-Set 终态 `{doc,memo,report}` 完整保留——因为 add-wins 让 r1 的 add-dot 活了下来。

这就是这两条路的取舍，讲到底：

| | Raft（第 5 章） | LWW（第 6 章） | OR-Set / CRDT（本章） |
|---|---|---|---|
| 一致性 | 强一致（线性化） | 最终一致 | 最终一致 |
| 分区下能写吗 | 否（少数派拒写） | 能（永远可写） | 能（永远可写） |
| 并发写冲突怎么收 | 不存在（写要先过协调） | 时间戳挑一个，**丢另一个** | 合并保留，**不丢** |
| 协调成本 | 每次写一轮共识 | 无 | 无 |
| 代价 | 分区下不可用 | 静默丢写 | 状态膨胀（tombstone）+ 只能表达半格类型 |

没有免费的午餐。Raft 用「可用性」换「强一致 + 简单语义」；LWW 用「丢写」换「实现极简、状态极小」；CRDT 用「状态膨胀 + 类型受限」换「永远可写 + 一个写都不丢」。选哪条取决于你的业务：购物车合并适合 OR-Set（少加东西比多加东西更让用户恼火），而「用户最后设置的头像」这种「本来就该后者覆盖前者」的语义，LWW 反而是**正确**的选择——丢掉旧头像正是用户想要的。**别无脑上 CRDT**，先问「我的数据语义到底是 last-write-wins 还是 merge-all」。

## 六、网络层实测：丢包 30% + 乱序 + 重复 + 分区，2 轮 gossip 收敛

前面几节都是纯代数测试（不过网络）——把分叉的状态喂进 merge 看收不收敛。但生产系统的收敛是包在**真实网络故障**里的：包会丢、会乱序、会重复，集群会分区。这节把同一套数学塞进 gossip（流言传播：每个副本周期性把自己整个状态发给同伴）+ 故障注入，看它扛不扛得住。

`stage07` Demo E：4 个副本，每 50ms gossip 一次自己的 OR-Set 全状态（anti-entropy，反熵——周期性对账消除分歧）；网络注入 **30% 丢包 + 乱序 + 重复**；在 `[200, 600]ms` 把集群切成 `{r0,r1} | {r2,r3}` 两半，期间两边各写各的，600ms 愈合。看它多久收敛：

```
## E. 网络层实测：4 副本 gossip，注入丢包30%+乱序+重复，分区[200,600]ms 后 heal
=== Scenario: crdt-gossip-partition-heal (seed=7) ===
at-heal(t=600) r0                      {beta,gamma}
at-heal(t=600) r1                      {beta,gamma}
at-heal(t=600) r2                      {alpha,beta,delta,gamma}
at-heal(t=600) r3                      {alpha,beta,delta,epsilon,gamma}
final r0                               {alpha,beta,delta,epsilon,gamma}
final r1                               {alpha,beta,delta,epsilon,gamma}
final r2                               {alpha,beta,delta,epsilon,gamma}
final r3                               {alpha,beta,delta,epsilon,gamma}
all replicas converged                 yes
convergence after heal (merge rounds)  2
convergence time after heal (sim ms)   100
alpha survived (add-wins vs remove)    yes
[sim] finalTime=2009ms events=589 safe=true
[msgs] sent=616 delivered=416 dropped=187 (loss=123 partition=64)
```

几个承重数字逐个读：

- **愈合瞬间四个副本是真的不一样的**：`at-heal` 那四行各不相同——r0/r1 在 `{beta,gamma}`，r3 在 `{alpha,beta,delta,epsilon,gamma}`。这不是装出来的「演示分歧」，是分区两边真各写各的、彼此看不见对方的结果。
- **愈合后 2 轮 gossip 收敛**：`merge rounds = 2`，`time after heal = 100ms`（= 2 × 50ms gossip 间隔）。这是实测计数，不是假设——代码每个 gossip 间隔采样一次，记录「四个指纹首次全相等」的那一刻。四个 final 完全相同（`all replicas converged = yes`），且都收敛到完整的 5 元素集。
- **add-wins 在网络层也成立**：`alpha survived = yes`。场景里 r1 在分区期间删了 `alpha`，r2 又并发重加 `alpha`——和第三节的纯代数 case 同构，只是这次跑在丢包乱序的真网络上。删没赢，加赢了。
- **丢包是真的丢了**：`dropped=187（loss=123 partition=64）`——187 条 gossip 没送达（123 条随机丢、64 条被分区挡），`delivered=416 / sent=616`。**为什么还能收敛？** 因为 merge 幂等：gossip 是「全状态反熵」，丢了这轮下轮再发就补上，重复送达是 no-op。我们敢对丢包 / 重复这么满不在乎，靠的不是重传记账，是那条幂等律。这正是 CRDT gossip 相比「精确一次投递」协议的工程红利——网络层可以做得极其廉价。

注意代码里有个刻意的设计：它**没有**把「收敛」写成每 tick 都检查的不变量（invariant）。因为收敛只是**最终**保证，不是每时每刻保证——分区期间副本本来就该不一样，那时候断言「全相等」会假阳性报错，而那恰恰是 CRDT 的正常状态。它真正每 tick 守的不变量是 `no-fabricated-elements`：任何副本都不能观测到一个**没人加过**的元素（`safe=true` 说明全程没违反）。这区分很重要——**收敛是 liveness（活性，最终会发生），不是 safety（安全性，时刻成立）**；把 liveness 当 safety 断言是分布式测试的常见错误。

## 七、⚡ 前沿：序列 CRDT 与 δ-CRDT，两个还没有「通用解」的方向

本章手写的 G-Counter / OR-Set 是 CRDT 里最成熟、最好懂的一档。但你日常用的协同编辑工具（Figma 的多人画布、Google Docs 式的实时文档、Yjs / Automerge 这类库）背后的 CRDT 要难得多，而且至今**没有一个被公认为「通用最优」的方案**。两个尖锐的开放问题：

**① 序列 CRDT：文本插入位置怎么定，至今没有干净的标准答案。**

集合无序，所以 OR-Set 好办。但文本是**有序**的：两个人在「ab」中间同一个位置并发插入字符，合并后谁在前？这不能 add-wins（两个字符都得留，但**顺序**唯一），也不能靠时间戳（会插错位置）。主流思路是给每个字符一个**稠密的、可无限细分的位置标识**（在任意两个位置之间总能再生成一个新位置），让插入退化成「在两个相邻标识之间塞一个新标识」——这就是 RGA（Replicated Growable Array）、LSEQ、以及更晚的 Fugue 等算法干的事。

但这条路有个一直没被彻底解决的毛病——**interleaving anomaly（交错异常）**：两个人各自连续输入一整个单词，并发合并后，两个单词的字符可能**交叉穿插**成一团乱码（`hello` + `world` 合成 `hwoelrllod`），而不是两个完整单词二选一排列。RGA 在某些并发模式下会交错；Fugue（2023 年的工作）证明了一类「最大非交错」的保证并给出了构造，但「什么是文本并发编辑的语义上正确的合并」本身就没有唯一定义——**这是个语义问题，不只是算法问题，目前无通用解，仍在研究**。所以 Yjs、Automerge、各家协同编辑产品用的序列 CRDT 各不相同，是工程权衡下的不同选择，不是收敛到了同一个「正确答案」。

**② δ-CRDT：state-based CRDT 传全状态会带宽爆炸，增量传播是缓解但不是终点。**

本章 Demo E 的 gossip 每次发**整个状态**（`this.state.dump()`）。toy 没问题，生产致命：一个用了一年、装了几十万元素 + 一堆 tombstone 的 OR-Set，每 50ms 给每个同伴发一遍全状态——带宽直接爆炸。这是经典 state-based CRDT 的核心工程瓶颈。

**δ-CRDT（delta-state CRDT）** 的思路是：每次本地操作产生一个小小的 **δ（增量）**——只包含「这次变化新增的那点 dot / tombstone」——副本之间只传播这些 δ 而非全状态，δ 之间再用同样的 join 合并。它在「state-based 的简单容错（丢包靠反熵自愈）」和「op-based 的低带宽（只传变化）」之间取了个中间点。但 δ-CRDT 也不是终点：δ 的传播仍要处理「漏了某个 δ 怎么补」（要么退回偶尔发全状态做反熵兜底，要么维护 δ 的因果区间，又把因果追踪的复杂度引回来了），**因果一致的高效 δ 传播 + tombstone 的及时回收，如何在带宽、内存、容错三者间取得通用最优，仍是活跃的研究方向，没有定论**。

把这两点和第一节的「代价定理」接起来收尾：CRDT 把冲突解决变成了数学保证，这是真本事；但它的边界恰恰在「能不能构造出合适的半格」——计数器和集合容易，**有序结构难、状态回收难、带宽优化难**。下一段路（也是分布式数据系统真正的前沿）不在「要不要 CRDT」，而在「**对那些半格不好构造的数据,我们还能把冲突解决推进到多深的数学保证**」。这也呼应全书的基调：分布式系统里几乎每个漂亮结论都拖着一条「但是……」，把那条「但是」讲透，才是专家和 survey 的分界线。

---

**本章小结**：CRDT 的收敛是半格 join 的代数后果——交换律抗乱序、结合律抗任意拓扑、幂等抗重复（第一节）；G-Counter 靠「按作者拆计数 + 逐格 max」把 join 落地（第二节），OR-Set 靠「dot + tombstone + add-wins」把可删集合塞进半格（第三节）；少了 tombstone 的伪 CRDT 操作不可交换，分区下副本永久发散（第四节）；LWW 也是合法 CRDT 但代价是静默丢写，对照出 CRDT「不丢写」的价值与「状态膨胀 + 类型受限」的代价（第五节）；网络层 30% 丢包 + 分区下 2 轮 gossip 收敛，靠的是幂等而非重传（第六节）；序列 CRDT 的交错异常和 δ-CRDT 的高效传播则是仍无通用解的前沿（第七节）。下一章我们离开「弱一致但永远可写」，回到需要全局协调的场景，看时间和顺序在分布式里还能玩出什么花样。
