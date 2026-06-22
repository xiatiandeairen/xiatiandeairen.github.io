---
title: "确定性编排与 Workflow"
slug: "12"
collection: "agent-engineering"
order: 12
summary: "第 0 章给了你一把「控制流之刀」：路径确定用 workflow，需要临场判断才用 agent。这一章把刀刃落到代码上——手写一个不含 LLM 的编排引擎，用 stage12 的真实运行数字讲透并发上限、失败隔离、stage barrier 三个机制怎么用、各自怎么坏。接 11 章的多 agent，下一章进入评测与可观测性。"
topics:
  - "Agent 工程"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

> 阶段四 · 编排与规模 · 第 12 章
>
> 第 0 章给了你一把刀；这一章是亲手把刀刃磨出来。

第 0 章说过 agent 和 workflow 的区别只有一句话：**谁决定下一步**。Workflow 是你把控制流写死在代码里，模型只在每一格填内容；agent 是把控制流交给模型，下一步、做几步、何时停都由它运行时决定。第 1 章把那把「控制流之刀」摆出来当全书取舍。前面十一章我们一直在磨 agent 那一侧——循环、工具、沙箱、权限、上下文、记忆、子 agent、多 agent。这一章回到刀的另一侧，把 workflow 自己实现一遍，因为「写死控制流」听起来简单到不值得讲,但真要在规模下不把机器跑挂、不被 API 限流封号、出错时还能定位到是哪条 case，需要的工程量比想象的多。

配套代码 `examples/agent-from-scratch/src/stage12-workflow.ts`,**整份不含一行 LLM 调用**——这是故意的。叶子任务用 `setTimeout` 假装成「一次 agent 循环」（`src/stage12-workflow.ts:174-190`），因为这一章要讲的是任务**之间**的编排,不是任务**内部**算什么。跑它直接 `npm run stage12`(或 `npx tsx src/stage12-workflow.ts`),下面引用的所有数字都是这份文件真实跑出来的 wall-clock,不是我估的——wall-clock 会随机器和负载有几毫秒抖动,引用值取自一次真实运行,你跑出来的数量级一致即可。

## 「确定性」到底确定什么

先把一个最容易误会的词钉死。这里说的 deterministic(确定性),**不是**指「LLM 两次给一样的文本」——它不会,温度调到 0 也不一定。指的是**编排层是纯代码,控制路径里没有模型**:给同一张任务图,哪些任务跑、怎么 fan out(扇出,一个任务派生多个并行子任务)、在哪里 barrier(屏障,所有并行任务必须在此汇合才能继续)、怎么失败,每次都是同一个形状。非确定性被**关进了叶子任务里**(叶子可以调模型,随它乱),外面那层脚手架是无聊、可读、可复现的代码(`src/stage12-workflow.ts:9-15`)。

这就是 workflow 相对 agent 的全部价值来源。第 1 章列过 agent 的代价:不可预测、难测试、会跑飞。把控制流从模型手里收回到代码里,这三条代价同时消失——你能 review 它、能单测它、能 replay 它,哪怕它调度的活儿本身是模糊的。**便宜也在这里**:编排层一次模型调用都不花,省下的 token 全在叶子。你正在读这段时用的 Claude Code 环境就同时供着两套:一个模型驱动的主循环(agent),和一个 Workflow 工具(把多个子任务用确定性脚本串起来,带 pipeline / parallel / 并发上限)。这一章手写的引擎,就是那个 Workflow 工具的最小内核。

## 何时把模型驱动降级为确定性 pipeline

这是本章最该带走的判断,不是某个 API。

agent 贵在「模型临场决定下一步」。但很多任务的下一步**根本不需要模型决定**——你已经知道顺序了。「把这 200 个 PR 标题各总结成一句话」:步骤是固定的(对每个标题跑一次总结),没有任何分支需要模型判断。这种活儿交给 agent 是三重浪费:每决定一次「下一步做什么」烧一轮 token、引入一次不确定性、还失去可复现。正确做法是**降级**:把控制流从模型手里拿回来,写成 `for` 循环 fan out + 一个总结函数当叶子。模型只在叶子里出现,负责「填内容」,不负责「决定流程」。

降级的判定线和第 1 章那把刀同一把:**路径能不能预先写出穷举的控制流**。能,就是 workflow——哪怕叶子是模型;不能(比如「修复这个 bug」,你没法预先写出所有排查路径),才留给 agent。实战里一个成熟系统几乎总是混合的:外层 workflow 把确定的骨架钉住(DAG,有向无环图——任务依赖关系的图),骨架的某几个节点内部才是 agent 循环。不是非此即彼。

stage12 把这个骨架抽成三个原语,文件头列得很清楚(`src/stage12-workflow.ts:17-25`):**STAGE**(有序的一步,stage N+1 要等 stage N 的 barrier 解开才开始)、**FAN-OUT**(一个 stage 内 N 个 item 并发跑)、**CONCURRENCY CAP**(并发上限,但任何时刻在飞的不超过 `limit` 个)。再加一个不是原语但同样重要的 **FAILURE ISOLATION**(失败隔离)。下面逐个讲——每个先讲怎么用,再讲怎么坏。

## Fan-out 与并发上限:`Promise.all` 是个陷阱

新手写并行 fan out 的第一反应是 `Promise.all(items.map(t => t.run()))`。这行代码有一个致命问题:它**同时启动所有任务**。100 个 item 就是 100 个任务瞬间在飞;如果叶子是模型调用,那就是 100 个并发请求——peak concurrency(峰值并发)等于 N。而 peak concurrency **就是你的资源账单**:N 个并发模型调用 = 打满 socket / 撞 rate limit(限流) / OOM(内存溢出)。

stage12 的核心机制是手写一个 bounded-concurrency map(有上限的并行映射,等价于 `pLimit`/worker-pool)。它不一次启动所有任务,而是开**恰好 `limit` 个 worker**,每个 worker 从一个共享游标 `next` 拉任务,拉一个跑一个,跑完回来再拉(`src/stage12-workflow.ts:74-119`)。结构上的关键一步在末尾那行:

```ts
const laneCount = Math.min(limit, tasks.length);
await Promise.all(Array.from({ length: laneCount }, () => worker()));
```

`Promise.all` 套的是 **lane(车道)**——一个固定、有界的集合,不是 tasks。每条 lane 内部顺序消费许多任务。这就是「有界并行 = 顶层只 await 有界个 promise」的结构技巧(`src/stage12-workflow.ts:113-117`)。

这里有个不写注释会被后人改坏的不变量,文件里特意点了:`next++` 这个共享游标**没有加锁**,安全**只因为** JS 是单线程、读取和自增之间没有抢占(`src/stage12-workflow.ts:60-66`)。换到真正的多线程 runtime,这里必须上原子操作。还有一个反直觉点——`Task.run` 是个 thunk(惰性函数 `() => Promise<T>`),不是已经启动的 promise(`src/stage12-workflow.ts:47-55`)。原因是 promise 一被创建就立刻开始执行,如果传进来的是已启动的 promise,limiter 还没来得及说「等等」它就跑了,cap 直接失效。**让引擎控制何时启动,是 cap 能成立的前提**。

机制讲完,看它真跑出来什么。Demo A 用 12 个任务、`limit=4`(`src/stage12-workflow.ts:196-205`):

```
items        : 12
peak inflight: 4 (cap was 4 — never exceeded)
wall clock   : 154ms  (serial would be ~600ms; cap=4 ⇒ ~3 waves of 50ms)
succeeded    : 12/12
```

注意 `peak inflight: 4` 不是断言出来的——引擎每次启停任务都回调一个 observer 记录当时在飞的数量,这个 4 是**观测到的真实峰值**(`src/stage12-workflow.ts:77-79`、`148-153`)。154ms 也对得上:12 个 50ms 任务,cap=4 ⇒ 3 波,理论 ~150ms,串行要 ~600ms。这就是 cap 的甜区:比串行快得多,又不像无上限那样把峰值顶到 N。

### 失败模式:无并发上限耗尽资源

Demo D2 把代价量化得最直白(`src/stage12-workflow.ts:275-285`):同样 50 个任务,跑两遍,只改 cap。

```
unbounded (limit=N=50): peak inflight = 50  ← N concurrent model calls = OOM / rate-limit ban
capped    (limit=8)    : peak inflight = 8  ← bounded blast radius, predictable resource use
same 50 tasks both runs; the cap changed peak from 50 to 8 without losing work
```

`limit=N` 就是退化成了那行天真的 `Promise.all`——peak 顶到 50。cap=8 把峰值从 50 压到 8,**一件活没少干**。这里有个诚实标注:demo **没有真的开 50 个 socket**(那会伤到跑这本书 CI 的机器),叶子任务只是记录峰值并发(`src/stage12-workflow.ts:253-258`、`276-279`)。危险是**结构上**演示的——证明了 `limit=N` 会把 peak 驱到 N,而 peak 顶到 N 正是真实无上限 fan out 撞资源天花板的那个确切机制。把这里的 50 换成 200 个真模型调用,你就拿到了一次 rate-limit 封禁或一次 OOM。

## 失败隔离:一个坏 item 不该拖垮一个 stage

回到那行 `Promise.all(tasks.map(run))` 的第二个坑:**任何一个任务 reject,整个 `Promise.all` 立刻 reject**。这意味着一个叶子抛错,同 stage 其他已经跑完的任务的结果**全部丢失**——它们的活白干了,outcome 也拿不到。这在 agent 编排里是灾难:你 fan out 100 个文档去总结,第 37 个文档触发了一个工具错误,结果前 36 个的总结连同后面的全没了。

stage12 的处理是不让任务的异常穿过引擎。每个 worker 在 lane **内部** try/catch,把抛错就地转成一个 `{ ok: false }` 的 outcome,然后立刻 loop 去拉下一个任务(`src/stage12-workflow.ts:97-109`)。一个 poison item(毒丸任务)只花掉一个 slot,不花掉整个 stage。这是 `arch-runtime` 那条「业务失败是值,不是异常」的直接落地——引擎的契约是「每个 item 我必返回恰好一个 outcome」,一个逃逸的异常会违反这个契约并带走兄弟任务(`src/stage12-workflow.ts:38-45`)。

Demo B:5 个任务里塞 1 个必抛错的(`src/stage12-workflow.ts:210-224`):

```
✓ ok-0 done (32ms)
✓ ok-1 done (32ms)
✗ boom-2 hit a simulated tool error (32ms)
✓ ok-3 done (32ms)
✓ ok-4 done (32ms)
result       : 4 ok, 1 failed — pipeline kept going
```

`boom-2` 抛了,另外 4 个照常完成,引擎收齐 5 个 outcome,partial failure 变成一个**被报告的结果**而不是一次崩溃。注意 `results[i]` 是按**捕获的下标 `i`** 写的,不是按完成顺序——完成顺序是非确定的,槽位顺序是确定的(`src/stage12-workflow.ts:68-72`)。这又是「确定性」那条线:谁失败了、在第几个位置,每次 replay 都一样。

## Stage barrier:用尾延迟换顺序保证

多 stage pipeline 的语义是:stage 之间有 barrier,stage 2 在 stage 1 **完全排空**之前一个 input 都看不到。实现上 barrier 是隐式但绝对的——`runStage` 那个 `await` 不到所有 lane 退休不返回,所以 stage N+1 可以安全假设 stage N 全完了(`src/stage12-workflow.ts:121-129`、`146-166`)。

Demo C 两个 stage(fetch 6 个 / summarize 4 个)(`src/stage12-workflow.ts:230-244`):

```
stage "fetch": 6 ok, wall=82ms, started ~0ms after t0
stage "summarize": 4 ok, wall=84ms, started ~82ms after t0
total pipeline: 166ms (stages are SEQUENTIAL — sum of stage barriers, not overlapped)
```

summarize 在 ~82ms 才起步——正好压在 fetch 的 barrier 之后。total 166ms = 82 + 84,两个 stage **串行相加,不重叠**。这就是 barrier 买到的东西:顺序保证。fetch 全部完成才开始 summarize,你不用担心 summarize 拿到半成品。

### 失败模式:barrier 等最慢那一个

顺序保证不是白来的。barrier 只能等到 stage 里**最慢**那个 item 完成才打开。Demo D1 量化了这个税:3 个 20ms 的快任务 + 1 个 200ms 的 straggler(掉队者)(`src/stage12-workflow.ts:260-273`):

```
fastest item : 21ms
slowest item : 201ms (the straggler)
stage wall   : 201ms — barrier waited for the slowest, not the average
tax          : 180ms wasted by 3 fast lanes idling at the barrier
```

stage 墙钟 201ms = straggler 的时间,不是平均值。**你的 p50 无关紧要,你的 p100 才是账单。** 180ms 被 3 条快车道在 barrier 前干等浪费掉了。放到真实 agent 编排:一个 stage 里有 99 个文档 2 秒总结完,1 个超长文档卡了 90 秒,整个 stage 就是 90 秒,后面所有 stage 全得等。这是「等所有东西」这个语义的隐藏代价,fan out 越宽、叶子时延方差越大,这个税越重。

缓解手段都有取舍,没有白拿的:给叶子加 timeout(超时直接判失败,但可能丢真该跑完的活)、把 straggler 改成异步追加而非阻塞 barrier(但 stage N+1 就不能再假设 stage N 全完了——你亲手放弃了顺序保证)、或者干脆别在这里设 barrier(改成流式 pipeline,但实现复杂度和错误处理都上一个台阶)。选哪个取决于你更怕「慢」还是更怕「乱序」。

## ⚡ 开放问题:agent 与 workflow 的边界何时该动态切换

到这里你手上有一把清晰的静态刀:任务规划期路径确定 → workflow,需要临场判断 → agent。但有一类问题这把刀切不利索——**边界本身应该在运行时移动**。

设想一个长任务:大部分步骤路径是确定的(适合 workflow 的确定性 pipeline),但中间某一步突然遇到一个没法预先穷举的岔路(这一段该升级成 agent 让模型临场决策);岔路趟完又回到确定的轨道(该降级回 workflow 省 token、恢复可复现)。理想系统应该能**在这两种模式间动态切换**:确定段用便宜的确定性编排,不确定段才烧 agent 的 token 和不确定性。

问题是**谁来判断当前这一步属于哪种、何时该切**。今天没有通用解。你可以让一个外层 agent 来决定何时下放给 workflow——但那个判断本身又是非确定、要烧 token、会判错的,你只是把不确定性往上挪了一层,没消掉。反过来用静态规则(满足条件 X 就切 agent)又回到了「需要预先穷举控制流」,而那恰恰是 agent 存在的理由被否定的情形。这是个真正的循环:**判断边界在哪需要的智能,本身就模糊到只有 agent 能做,而 agent 又是你想限制使用的那个东西**。

目前业界(包括你正在用的这个环境)的实际做法是把切换点交给**人**:工程师在写 workflow 时手动决定哪个节点内部是 agent 循环,哪些段是确定性脚本。自动、运行时、自适应地移动这条边界——让系统自己判断「这一段我该多确定」——据我所知没有成熟方案,**正在研究**。如果哪天有了,它大概率会改写本书第 0 章那把刀:刀刃不再是设计期画死的一条线,而是运行时浮动的一个面。

## 收束

这一章的引擎是无聊、可复现的代码,这正是它的全部价值(`src/stage12-workflow.ts:293-295`):**cap 框住资源账单,isolation 把崩溃变成被报告的结果,barrier 用尾延迟买来顺序。非确定性始终留在叶子任务里。** 三个数字记住就够了——peak 从 50 压到 8(cap),partial failure 4 ok / 1 failed 而非全灭(isolation),201ms stage 墙钟里 180ms 是 barrier 税(顺序的代价)。

第 11 章的多 agent 把活儿分给了多个模型;这一章给了你一层在它们**上面**、不含模型的编排,决定谁并行、并行多宽、在哪汇合、出错怎么办。下一章进入评测与可观测性——当你的 pipeline 跑起来,怎么知道它跑得对、跑得值、哪里慢。
