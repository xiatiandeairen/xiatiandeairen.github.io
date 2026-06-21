---
title: "子 Agent 与上下文隔离"
slug: "10"
collection: "agent-engineering"
order: 10
summary: "子 agent 的真正卖点不是「角色专精」，是 context 隔离——把一坨噪声中间步骤关在子 agent 自己的窗口里，只让结论翻墙回主循环，从源头掐住第 5 章那条 O(T²) 曲线。这一章用 stage10 的真实数字算清这堵墙省了多少（47.3x 更瘦的父 context），再讲它怎么反咬：墙挡住了 agent 间的共享发现、结论太瘦逼出重复追问。结尾留一个今天没有通用解的开放问题：拆分粒度。"
topics:
  - "Agent 工程"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

> 阶段三 · 规模与持久 · 第 10 章
>
> 第 5 章把单个 context 窗口当预算来管，第 6 章把塞不进去的东西搬到磁盘。这一章给第三个答案：开第二个窗口。

绝大多数关于子 agent（sub-agent，主 agent 派出去干一件事的子进程）的说法是「让一个专门的 agent 扮演专门的角色」——research agent、coding agent、review agent，像招了一个团队。这个说法把因果搞反了。你拆出一个子 agent，**不是因为它要扮演不同角色，是因为它的中间过程会污染你的 context**。角色专精是顺带的，隔离才是目的。

把这件事讲清楚需要先回到第 5 章那条曲线。模型 API 是无状态的，每一轮都要把整个 transcript 重发一遍，所以一个跑 T 轮的 agent 总成本是 O(T²) 而不是 O(T)。`stage10-subagents.ts` 开头的注释把这一章的动机直接写成了这条曲线的推论（`examples/agent-from-scratch/src/stage10-subagents.ts:4-8`）：

```ts
// A single agent that does deep work pays the O(T^2) transcript tax from stage01:
// every intermediate observation it generates is re-sent on EVERY subsequent
// turn. A sub-task that produces a lot of noisy middle steps (grep 200 files,
// read 30 of them, reason over each) can blow the parent's context budget even
// though the parent only ever needed the *conclusion*.
```

关键在最后那句：**父 agent 从头到尾只需要结论**。它问「这个 codebase 的认证漏洞在哪、怎么修」，它要的是一句「在 `legacy_cookie.ts`，明文比对，删掉 fallback」。它不需要那 200 行 grep 命中、不需要被读进来的整个文件。但如果调查是 inline（内联，在父循环里直接跑）做的，这些噪声会全部进父 transcript，然后在之后的**每一轮**重发。你为一坨用完就该扔的字节，付了 O(T²) 的钱。

子 agent 就是修这个的：给调查开一个**自己的** context 窗口，让它在自己窗口里 grep、read、推理，烧自己的预算，最后只把结论字符串递回父 agent。中间步骤从来没进父 transcript，所以从来不会被重发。这堵「墙」是子 agent 的全部意义。

**先把诚实标注放最前面**：和第 5、6 章一样，stage10 全程离线、不连真模型，两个 agent loop（父 + 子）都跑真的控制循环，但模型是写死的脚本（scripted MockLLM）。所有 token 数来自 `estimateTokens`（`core/llm.ts` 里「约 4 字符 = 1 token」的启发式），**不是真 BPE tokenizer**，对中文尤其不准；fake 文件仓也是玩具。下面所有 before/after 比例是在同一把尺上做的同口径对比，机制是真的，绝对数字别当生产基准。跑 `cd examples/agent-from-scratch && npx tsx src/stage10-subagents.ts` 当场复现。

---

## 机制：墙——父 agent 吸收结论，不吸收中间步骤

stage10 把「派一个子 agent」建模成一次工具调用。`spawnSubAgent` 跑一个完整的子循环，但**只**返回它的 answer 字符串，子循环的 transcript 被丢掉——这个「丢掉」就是隔离机制本身（`examples/agent-from-scratch/src/stage10-subagents.ts:262-289`）：

```ts
async function spawnSubAgent(subLlm: LLM, goal: string): Promise<SpawnResult> {
  const run = await runLoop(subLlm, SUBAGENT_TOOLS, SUBAGENT_TOOL_IMPLS, goal,
    'You are a focused investigation sub-agent. Do the digging, then report a single self-contained conclusion.',
    /* maxTurns */ 6);

  // The child's full context (grep dumps + file reads) — this is what isolation
  // keeps OUT of the parent. We measure it to quantify the savings.
  const subAgentContextTokens = transcriptTokens(run.transcript);

  // What the parent gets: just the conclusion, wrapped as one tool_result block.
  const returnedBlock: UserBlock = { type: 'tool_result', toolUseId: 'spawn_1', content: run.answer };
  const returnedTokens = estimateTokens(JSON.stringify(returnedBlock));
  // ...
}
```

注意子 agent 的结论是包成一个 `tool_result` block 回来的——父 agent 看「派子 agent」和看任何一次工具调用的返回，是同一回事。这正是你现在用的 Claude Code 环境里 Agent 工具的形状：你（主循环）发起一个子 agent，它在自己的 context 里 grep、读文件、试错，最后**只有它的 final message 回到主循环**，中间那几十步工具调用主循环全程看不见。子 agent 本质上是「把一段会产生大量噪声的工作 offload 到另一个窗口」——它是第 5 章卸载（offloading）的一种特例，只不过卸载的目标不是磁盘，是另一个 agent 的 context。

这个建模也顺带解释了为什么子 agent 必须有硬性的 turn 上限。`runLoop` 的 maxTurns 是第 1 章那个 #1 可靠性旋钮（`examples/agent-from-scratch/src/stage10-subagents.ts:115-117`）：

```ts
// Hard ceiling (stage01's #1 reliability knob). A sub-agent that never stops
// must not bill forever — and crucially must not stall the parent that awaits it.
return { answer: '[sub-agent stopped: max turns reached]', transcript: messages, turns: maxTurns };
```

父 agent 在 `await` 子 agent，子 agent 一旦死循环不只是烧自己的钱，是**卡住整个父循环**。隔离让子 agent 的失败不污染父 context，但也让子 agent 的卡死直接传染父进程——所以子 agent 必须能自己停下来。

### 算这堵墙省了多少

stage10 派一个子 agent 去查认证漏洞，子 agent 跑 3 轮（grep → read → 给结论），实测输出：

```
sub-agent run     : 3 turns (grep → read → conclude), in its OWN context
sub-agent context : 5531 tokens at peak (grep dump + full file read live HERE)
returned to parent: 117 tokens (the conclusion only)
if INLINE in parent : +5531 tokens of parent context (and re-sent every later turn — O(T^2))
with ISOLATION      : +117 tokens of parent context (one conclusion, re-sent cheaply)
parent saved        : 5414 tokens up front (2% of the inline cost crosses the wall; 47.3x leaner parent)
```

子 agent 自己的 context 峰值 **5531 token**（那坨 grep dump 和整个文件读都活在这里），但翻墙回父 agent 的只有 **117 token**——结论那一句。如果这次调查是 inline 在父循环里做的，父 context 会一次性涨 5531 token，而且这 5531 token 会在之后**每一轮**重发。隔离之后父 context 只涨 117 token。**只有 2% 的内联成本翻过了墙，父 context 瘦了 47.3 倍。**

而且这还是**保守下界**。stage10 在注释里特意标了这点（`examples/agent-from-scratch/src/stage10-subagents.ts:307-312`）：5531 这个数字是「假如内联，父 context 一次性涨多少」，但真实账单更大，因为那 5531 token 进了父 transcript 之后会被 O(T²) 地反复重发。隔离省的不是一次 5531，是「5531 × 之后的轮数」。会话越长，这堵墙越值。

这就是为什么子 agent 的真正卖点是隔离不是角色。哪怕子 agent 和父 agent 用完全一样的模型、一样的 system prompt、一样的工具——只要它做的活会产生大量「父 agent 不关心的中间字节」，开一个独立窗口就已经赚了。角色专精（给子 agent 一个更窄的 prompt）是锦上添花，省 token 才是那块饼。

---

## 失败模式 A：墙挡住了 agent 之间的共享发现

隔离不是免费的。你砌起一堵墙挡住噪声往父 agent 流，这堵墙**同时**挡住了子 agent 之间互相看见对方的中间发现。

stage10 派第二个子 agent 去查一个相关的目标（「审一下 session 刷新路径有没有同类漏洞」）。它有自己的 context，看不到第一个子 agent 已经 grep 过、已经知道认证代码在哪——因为 A 的中间发现从来没离开过 A 的窗口。于是 B 把 grep 和 read 从头重做一遍：

```
sub-agent A did   : grep + read (then discarded its context behind the wall)
sub-agent B redid : 2 identical tool observation(s) A already had
wasted re-work    : 5307 tokens of tool output B regenerated from scratch
why               : B's context cannot reach A's intermediate findings — that is the cost of isolation
```

B 重新生成了 **5307 token** 的工具输出——这些 A 早就有了，但 A 把它们连同自己的 context 一起扔在墙后面了。stage10 是直接比对两个子 transcript 里**字节完全相同**的 `tool_result` block 来量化这个浪费的（`examples/agent-from-scratch/src/stage10-subagents.ts:351-361`），不是估的。

这是隔离的本质代价，不是 bug。墙的方向是单向的：它让结论能出、噪声不能进父 agent；但它对**平级的两个子 agent**是双向不透明的。你想要 A 的噪声不进父 context，就得接受 B 看不到 A 的噪声。

缓解有两条路，stage10 都标了「都不免费」（`examples/agent-from-scratch/src/stage10-subagents.ts:367-368`）：一是给两个子 agent 一个**共享的 scratchpad / store**（就是第 5 章的卸载——父 agent 把 A 的发现写进一个共享存储，B 启动时读得到），二是让父 agent 把 A 的结论**揉进 B 的目标**里（「认证代码在 `legacy_cookie.ts`，现在去查 session 刷新路径」）。第一条要你额外维护一个存储和读写协议，第二条要父 agent 自己判断 A 的哪部分结论对 B 有用——本质上是把 A、B 之间本该自动共享的 context，改成由父 agent 手动搬运。隔离把「共享」从默认变成了需要显式付费的动作。

---

## 失败模式 B：结论太瘦逼出重复追问

机制那一节的 47.3x 有个隐含前提：子 agent 返回的结论是**自足的**——它把父 agent 接下来会追问的东西（在哪、为什么、怎么修）提前答了。如果结论太瘦，墙就反过来咬你。

stage10 派一个故意写得很瘦的子 agent，它做了同样昂贵的调查（grep + read），但只回一句废话：

```
spawn #1 (thin)   : "Yes, there is an auth issue somewhere in the codebase."
parent accepts?   : NO (names location=false, names fix=false) → must re-spawn
spawn #2 (re-ask) : 3 turns, FULL re-investigation (the wall discarded #1's work)
two thin spawns   : 9851 tokens of total investigation work
one rich spawn    : 5531 tokens (had it answered well the first time)
thinness overhead : 4320 extra tokens (a thin child is worse than no isolation savings on that task)
```

父 agent 的验收逻辑很简单：一个能用的结论必须点名一个位置、点名一个修法（`examples/agent-from-scratch/src/stage10-subagents.ts:393-395` 用正则检查 answer 里有没有文件路径和 fix 动词）。这句「codebase 里某处有个认证问题」两样都没有，父 agent 拒收，**只能重派**。

而这里墙又咬了一口：因为第一个子 agent 的 context 连同它的调查全被扔在墙后了，这次追问**不是一次便宜的澄清**——父 agent 没法说「你刚才查的那个，具体在哪个文件」，因为「你刚才查的那个」已经不存在了。它只能重派一个全新的子 agent，从头 grep、从头 read。两次瘦派一共烧 **9851 token**，而一次写好的派只要 **5531 token**。瘦结论的额外开销是 **4320 token**——在这个任务上，**一个瘦子 agent 比根本不用隔离还糟**。

教训写在 stage10 最后（`examples/agent-from-scratch/src/stage10-subagents.ts:418`）：隔离只在结论自足时才划算。所以子 agent 的 brief（任务说明）里必须**显式奖励完整性**——告诉它「报一个自足的结论：位置、根因、修法都要有」，而不只是「去查一下」。第 5 章讲过 context 是预算，这里多一层：你给子 agent 的预算只能花一次，因为墙会在它返回的瞬间把找零全部没收。

这也对照出 orchestrator-worker（编排者-工人）模式的真实分工。父 agent（orchestrator）的核心职责不是「派活」那一下，是**两件判断**：派之前把目标写得足够窄、足够带完整性要求；收到之后判断结论够不够自足、要不要重派。Worker 负责烧 context 把脏活干完，orchestrator 负责守住墙的两边——传参时把完整性要求带进去，收口时做验收。stage10 里那个 `isActionable = namesLocation && namesFix` 就是收口验收的最小形态；真实环境里这一步通常交给模型自己判断，但判断这件事本身不能省，省了就退化成「派出去就信」,瘦结论直接进父 context 当事实。

---

## ⚡ 开放问题：该在哪里切一刀，没有通用解

到这里你可能想问一个很自然的问题：**那我到底该把什么拆成子 agent、什么留在主循环？拆多细？** 这个问题今天**没有通用解，仍在研究**。

难点在于拆分的收益和代价是对冲的，而对冲的平衡点依赖你**事前不知道**的东西：

- 拆得太粗——把一大段还掺着「父 agent 其实需要看见的中间发现」的工作整个关进子 agent，墙会把那些本该共享的发现一起没收（失败模式 A）。
- 拆得太细——每个子 agent 只干一丁点活就返回，你付的是一堆 spawn 的固定开销（每次启动的 system prompt、目标传参、结论验收往返），而每次省下的噪声又不够多，固定成本吃掉隔离收益（失败模式 B 的极端化）。
- 而「这段工作会产生多少父 agent 不关心的噪声」「子 agent 一次能不能给出自足结论」——这两个决定该不该拆的关键量，**你在派它之前并不知道**。你得先让它跑才知道它会 grep 出 200 行还是 5 行，先看到结论才知道它够不够自足。

业界现在的做法全是**启发式**，没有一个能形式化的判据：「探索/调查类任务拆出去」（因为这类任务噪声大、结论窄，正好是隔离的甜区）、「需要和主线频繁交换中间状态的别拆」（因为墙会变成累赘）、「拆出去的活要能写清楚验收标准」（呼应失败模式 B）。这些都对，但都是经验法则，给不出「这个具体任务该不该拆」的确定答案。你现在用的 Claude Code 环境里，「什么时候自动派 Agent 子 agent、什么时候在主循环里直接干」同样是模型基于这类启发式当场判断的，不是查一张确定的表——它也会判断错，也会出现「派出去结果太瘦又追问」或者「本该派出去的脏活堆在主 context 里」。

更深一层：拆分粒度本质上是一个 **context 的分区问题**——把一个任务的信息切成若干块，每块装进一个独立窗口，让块内高内聚、块间低耦合，再让必要的信息以最小代价跨块流动。这和数据库分片、和微服务拆分是同一类问题，而那两个领域几十年也没拆出通用解，全是 case-by-case 的工程判断。Agent 的子 agent 拆分多了一重难度：分区边界要在**运行时**、在你看到任务实际产生多少噪声之前就定下来。这是个开放问题，别指望有人给你一个公式。

---

## 小结

子 agent 的真正理由是 context 隔离，不是角色专精：把会产生大量噪声中间步骤的工作关进一个独立窗口，只让自足的结论翻墙回主循环，从源头掐住第 5 章那条 O(T²) 曲线——stage10 实测父 context 瘦 47.3 倍，且这是保守下界。但墙是有代价的：它挡住了平级子 agent 之间的共享发现（B 白白重做 5307 token 的活），结论太瘦时还会逼出全量重派（瘦派多烧 4320 token，比不隔离还糟）。orchestrator 的职责就在墙的两边——传参时把完整性要求带进去、收口时验收结论是否自足。而「该在哪里切这一刀」是个运行时的 context 分区问题，今天没有通用解。

下一章把这条线推到多个子 agent 并行协作（multi-agent），那里墙的代价会从「两个 agent 重做活」放大成「N 个 agent 的协调与一致性」。
