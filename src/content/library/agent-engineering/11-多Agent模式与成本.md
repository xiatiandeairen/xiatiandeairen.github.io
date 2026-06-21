---
title: "多 Agent 模式与成本"
slug: "11"
collection: "agent-engineering"
order: 11
summary: "上一章用 sub-agent 把脏活塞进子上下文、给父上下文减负——那是『多 agent』唯一一个稳赚的用法。这一章把『多 agent』当成一笔账来算：handoff、并行 worker、辩论各是什么、什么时候真省、什么时候只是把第 1 章的 O(T²) token 税乘以 N。配套 stage11 在同一个工单上跑单 agent vs 两种多 agent 编排，实测两种多 agent 都比单 agent贵（1.35× / 1.61×），并把协调失败、context 不同步、成本爆炸三种翻车现场跑给你看。结尾留一个我认为今天没收敛的开放问题：agent 间通信协议。"
topics:
  - "Agent 工程"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

> 阶段四 · 编排与规模 · 第 11 章
>
> 上一章用 sub-agent 给父上下文减负。这一章问一个更难的问题：再加一个 agent，到底是赚还是亏。

「多 agent」听起来像加马力——多一个脑子干活，总该更快更强。账单不这么看。回到第 1 章的结论：一个 agent 的成本是它每一轮**输入 token 的累加**，而整段 transcript 每一轮都要重发一遍，这是那条 O(T²) 曲线。把一个任务拆给 N 个 agent，**不会**把这笔成本除以 N——它会**乘**：每个子 agent 都要重新付自己那份 system prompt、工具规格、和一份共享任务的拷贝。这一章不讲「多 agent 架构图」，讲一笔诚实的账：在同一个任务上，多 agent 到底比单 agent 贵多少，以及什么时候这笔溢价买到了东西、什么时候买了个寂寞。

配套代码 `examples/agent-from-scratch/src/stage11-multiagent.ts`，完全离线、确定性可复现，跑 `npx tsx src/stage11-multiagent.ts` 当场看输出。它在**同一个客服工单**上跑三条臂（单 agent / handoff / 并行 fan-out），再跑三种翻车现场。你正在用的 Claude Code 就是活体参考：它的 Agent 子 agent（独立 context 的探索者）、Workflow（确定性编排）、TodoWrite（把计划写成 artifact）都是这一章谈的编排原语——区别只在于 Claude Code 把「什么时候该多开一个 agent」的判断交给了模型和你，而这一章想给你一个能算的判断规则。

**先把诚实声明摆前面**，免得你拿 toy 数字当生产数据：stage11 的所有 token 计数来自 `estimateTokens`（`core/llm.ts`），一个「约 4 字符 = 1 token」的启发式——**不是真的 BPE 分词器**，对中文尤其不准（文件头第 25-29 行写明了这点）。但下面所有「N× 更贵」的倍数，是同一个估算器**同等地**作用在两条臂上算出来的真实算术。绝对 token 数是近似的，**倍数（ratio）是可信的**——这正是这一章唯一想让你记住的数字。

---

## 一个任务，三种编排：把账摊开

stage11 钉死了一个跨臂不变的任务，这是整笔账成立的前提——任务一变，成本对比就作弊了（`src/stage11-multiagent.ts:76-79`）：

```
Customer ticket: "My order #4471 arrived broken. I want a refund AND I need
to know if you ship replacements to Canada." Decide the refund and answer
the shipping question.
```

一个工单，两个诉求：退款决策 + 加拿大配送政策。一个 agent 能一口气答完；多 agent 版本把它拆开。系统提示也是真有分量的（policy + persona，`src/stage11-multiagent.ts:84-88`）——这很关键，因为**每个跑起来的 agent 都要付一份自己的系统提示拷贝**，这份重复就是后面账单里那个看不见的税。

### 臂 1：单 agent —— 要被打败的基线

最朴素的一臂：一个模型、一段 transcript、一轮答完整个工单（`src/stage11-multiagent.ts:111-130`）。它是每个多 agent 设计**必须打败**才配存在的基线。实测：

```
[ARM 1] SINGLE AGENT
  answered both parts: true
  solo                 in= 118 out=  61 tok
  TOTAL                          179 tok
```

179 token，两个诉求都答了。记住这个数。

### 臂 2：handoff（Swarm 思路）—— 路由是真需要时才划算

handoff（交接）是 OpenAI Swarm 推的模式：一个便宜的 router agent 看一眼工单，把控制权**转交**给专科 agent。Swarm 最核心的洞察在于：转交不是一个特殊 API，它就是一次 `tool_use`——router 不输出散文，它像请求任何工具一样请求一次「transfer」，编排层读出目标再 dispatch（`src/stage11-multiagent.ts:160-168`）。控制流和工具调用走同一条通道，所以不需要给 API 加任何新东西。

为什么这是个**对的**模式：路由是个小而边界清晰的决策；你不想让退款专科的提示里塞满配送政策，反之亦然。handoff 让每个 agent 的 context 保持窄。但实测它仍然比单 agent 贵：

```
[ARM 2] HANDOFF (router -> specialist), happy path
  routed to : refund_agent
  router               in=  86 out=  35 tok
  specialist:refund_agent in=  85 out=  35 tok
  TOTAL                          241 tok
  answered both parts: false (note: refund only — shipping half lost)
```

241 token，1.35× 单 agent。贵在哪：任务文本被付了**两遍**（router 读一次，专科再读一次，`src/stage11-multiagent.ts:200-201` 和 `227-229`），而且现在跑了两份系统提示。更扎心的是 `answered both parts: false`——窄专科只答了退款，**配送那半截被丢了**。这是单轨 handoff 的真实缺陷：你为「关注点分离」付了 token,又因为分离丢了另一半诉求。handoff 买的是质量/隔离,**不是** token——任何宣称 handoff「省钱」的说法都该被这一行打脸。

### 臂 3：并行 fan-out + 综合 —— 你买的是延迟，付的是 token

fan-out（扇出）把工单拆成互相独立的子问题，一个 worker 答一个，**并发跑**（`Promise.all`），最后一个 synthesizer（综合者）把 worker 的输出合成一个答案（`src/stage11-multiagent.ts:282-304`）。

你买到的是 wall-clock 延迟——worker 同时跑，墙上时钟更快。你付出的是：每个 worker 重付自己的系统提示 + 它那片任务；**而且** synthesizer 的输入 = 它的提示 + **每个** worker 输出的拼接。synthesizer 这一轮是 fan-out 成本悄悄爆炸的地方:worker 越多，synthesizer 的输入越胖。实测:

```
[ARM 3] PARALLEL FAN-OUT + SYNTHESIS, happy path
  worker:refund        in=  46 out=  21 tok
  worker:ship          in=  42 out=  21 tok
  synthesizer          in= 120 out=  39 tok
  TOTAL                          289 tok
  answered both parts: true
```

289 token，1.61× 单 agent。它确实两个诉求都答了（比 handoff 强），但用了 1.6 倍的钱。注意一个关键点:**并发不改成本**。你付的是同样多的 token，只是付得更早。延迟下降了,账单没有。

### 三臂对账

stage11 把三个总数并排打出来（`src/stage11-multiagent.ts:410-417`）:

```
[COST] total tokens vs single agent (lower is cheaper)
  single agent      :  179 tok  (1.00x baseline)
  handoff           :  241 tok  (1.35x)
  fan-out + synth   :  289 tok  (1.61x)
  → both multi-agent arms cost MORE for this task. The single agent
    answered both parts in one turn; the multi-agent tax (duplicated
    system prompts + re-read task + synthesizer re-ingest) bought nothing.
```

**对这个任务，两种多 agent 都更贵，而且什么都没多买到。** 单 agent 一轮答完了两半诉求，而多 agent 税（重复系统提示 + 重读任务 + synthesizer 重新摄入）纯属白付。这不是说多 agent 没用——是说**这个任务**不该上多 agent。一个能被单 agent 一轮答完的任务，拆开只会增加成本和失败面。

> **顺带把 Anthropic 那个常被引用的数字接上**：Anthropic 公开的多 agent research 系统实测 token 用量约为单 agent 聊天的 **~15×**。stage11 这里只到 1.35×/1.61× 是因为任务小、worker 少；下一节的 cost blow-up 实测会让你看到倍数怎么随 worker 数往那个量级爬。15× 不是 bug，是这类系统的**正常运行成本**——它意味着只有当任务价值足够高、且单 agent 确实做不动时,多 agent 才划算。

### 什么时候多 agent 不划算 —— 一条能算的规则

把上面的账抽象成判断规则。**默认用单 agent;只有同时满足下面两条才上多 agent**:

1. **单 agent 真的做不动**——不是「拆开更优雅」,是单 agent 会爆 context（参第 10 章 sub-agent 的 O(T²) 论证）、或子任务需要真正不同的工具/权限边界、或并发能省下来的延迟对用户有实打实的价值。
2. **任务价值 > 多 agent 税**——你愿意为这个任务付大约线性增长的 token 溢价（worker 多了会更陡,见下节)**外加**下面三种失败模式的风险。

反过来,**一个能被单 agent 一轮答完的任务,拆成多 agent 就是纯亏**——这正是臂 2/臂 3 演示的。这条规则和我在 Claude Code 里的实践一致:大多数任务我不开 Agent 子 agent,只在「探索一大片代码、中间步骤会把主 context 撑爆、而我只要结论」时才开——也就是第 10 章那个稳赚的用法。

---

## 三种翻车现场:多 agent 独有的失败模式

下面三种是单 agent **根本不可能**有的失败——它们是你多开 agent 的那一刻起买进来的风险。

### 失败 1:协调失败 —— router 转交给一个不存在的 agent

router 决定转交给一个编排层 dispatch 不了的专科,工单掉进黑洞。stage11 故意让 router 转交给 dispatch 表里没有的 `billing_agent`(`src/stage11-multiagent.ts:212-222`):

```
[FAIL 1] coordination failure: router transfers to a missing specialist
  routed to : billing_agent (not in dispatch table)
  answer    : [coordination failure] router transferred to unknown agent "billing_agent"; ticket dropped
  cost paid for ZERO useful output: 121 tok (router ran, ticket dropped)
```

注意两件事。第一,**你为零产出付了 121 token**——router 跑了、决策了、然后掉单。第二,代码**没有静默吞掉**这个失败:如果不显式 surface,循环会以「没答案」收尾却看起来「成功」(`src/stage11-multiagent.ts:213-214` 的注释点破了这点)。这对应一条通用规则:**agent 间的 dispatch 失败必须 fail-fast,不能 silent**——一个静默掉单的多 agent 系统比一个报错的单 agent 危险得多,因为你以为它在工作。

### 失败 2:context 不同步 —— 隔离的 worker 互相矛盾

fan-out 的 worker 各自独立、**互相看不见**对方的输出——这份隔离正是延迟优势的来源,也正是 context desync 的根源。stage11 给两个 worker 同一个配送问题,它们给出相反结论(`src/stage11-multiagent.ts:436-444`):

```
[FAIL 2] context desync: isolated workers contradict each other
  worker A  : Yes, we ship replacements to Canada.
  worker B  : No, we do not ship to Canada.
  merged    : [context desync] workers disagree: "Yes, we ship replacements to Canada." vs "No, we do not ship to Canada.". Neither worker saw the other, so this merge cannot be trusted. Escalating instead of guessing.
```

synthesizer 面对两个对同一事实给出相反裁决的输入,它**没有 ground truth**去裁定谁对。stage11 的 synthesizer 被写成做一件诚实的事:检测到矛盾就**上报冲突,而不是编一个和解出来**(`src/stage11-multiagent.ts:312-334`)。这是 garbage in, garbage out 的多 agent 版本——但更隐蔽,因为如果你的 synthesizer 不做矛盾检测,它会自信地把两个矛盾揉成一个听起来很顺的答案,而你**完全看不出**它在瞎编。

代码里那个矛盾检测器(`detectContradiction`,`src/stage11-multiagent.ts:340-345`)只是个 toy——在「是否发往加拿大」这一根已知轴上做字符串匹配,**不是**通用矛盾引擎。它存在只为让失败模式 2 可观测、确定性复现。生产里没有这个 toy 检测器替你兜底,而通用的「两个 agent 的输出是否矛盾」检测,本身就是个开放难题。

### 失败 3:成本爆炸 —— synthesizer 把所有 worker 重新摄入一遍

把 fan-out 扩到很多 worker,看 synthesizer 输入怎么膨胀。stage11 跑了 1/2/4/8 个 worker(`src/stage11-multiagent.ts:449-463`):

```
[FAIL 3] cost blow-up: token multiple vs worker count (synthesizer re-ingests all)
  workers=1: total= 238 tok  (1.33x solo), synthesizer input= 114 tok
  workers=2: total= 386 tok  (2.16x solo), synthesizer input= 147 tok
  workers=4: total= 681 tok  (3.80x solo), synthesizer input= 212 tok
  workers=8: total=1270 tok  (7.09x solo), synthesizer input= 342 tok
```

8 个 worker 时已经到 **7.09× 单 agent**——还只是个 toy 任务。把这条曲线往真实任务上外推,你就理解了 Anthropic 那个 ~15× 是怎么来的:**synthesizer 输入随每个 worker 增长**(114 → 147 → 212 → 342),它是 fan-out 成本里那部分**并发抹不掉**的钱。worker 之间能并行,但 synthesizer 必须等所有 worker 回来、再把它们全读一遍,这一轮是串行的、且越来越胖。

stage11 给的结论很直接(`src/stage11-multiagent.ts:464-466`):**只在「并发省下的延迟值这笔大致线性的 token 溢价、外加上面的合并风险」时才加 worker**。加 worker 不是免费扩并行度,每加一个,你同时加了 synthesizer 的输入、加了一份矛盾的可能、加了一个掉单的接口。

---

## ⚡ 开放问题:agent 间该怎么通信,还没收敛

前面三种失败模式有一个共同的根:**agent 之间怎么可靠地交换状态和控制权,today 没有公认的解。** 这是我认为这个领域目前**没有通用解、正在研究**的开放问题。

具体没收敛在哪:

- **协调本身。** handoff 用 tool_use 当转交通道(Swarm),fan-out 用「父编排 + 共享任务字符串」当通道(stage11 臂 3),但这两种都是**点对点的临时拼装**——没有一个 agent 群体能据以协商「谁负责哪一半、谁是真相来源、冲突了找谁仲裁」的标准协议。失败 2 的 context desync 本质就是:两个 agent 没有共享状态的契约,各说各话,合并层只能上报或瞎猜。

- **通信协议。** 业界开始有人推 **A2A(Agent-to-Agent)** 这类协议,想给「agent 怎么发现彼此、怎么协商任务、怎么传递结构化结果」立标准——可以把它类比成第 2 章的 MCP(给「agent↔工具」立的标准),A2A 想给「agent↔agent」立同样的东西。但 A2A 远没到 MCP 那种「接上就能用」的成熟度:谁是权威、传输层长什么样、错误和重试语义怎么定、跨厂商互操作能不能成,这些都还在动。我把它放在这里**点到为止**,正是因为现在押注任何一个具体协议都为时过早。

- **辩论(debate)这类「让多个 agent 互相批判以提质量」的模式**,在 paper 里有信号,但什么时候真比单 agent + 一轮自我反思(第 9 章)强、强多少、值不值那 N× 成本,没有能让你直接照搬的结论。它和 fan-out 共享同一个未解的难题:**多个 agent 的输出怎么可靠地聚合**——失败 2 那个「synthesizer 没有 ground truth」的窟窿,辩论模式同样得面对。

务实的态度:在协议收敛之前,把多 agent 的「通信面」尽量做小做显式——窄接口、强制 fail-fast 的 dispatch(对治失败 1)、合并层强制矛盾检测并在冲突时上报而非猜(对治失败 2)、每加一个 worker 前先算清 synthesizer 的增量成本(对治失败 3)。这三条不是协议,是在没有协议的当下能把风险摁住的工程纪律。

---

## 小结

这一章只想钉死一件事:**多 agent 是一笔账,不是一个架构信仰。** stage11 在同一个任务上实测,handoff 1.35×、fan-out 1.61×,两种多 agent 都比单 agent 贵且没多买到东西——因为这个任务单 agent 一轮就答完了。把这条往上外推,8 worker 已经 7.09×,Anthropic 的生产 research 系统约 15×;这些倍数不是浪费,是这类系统的正常成本,意味着**只有任务价值够高、单 agent 确实做不动时,多 agent 才划算**。

判断规则:默认单 agent;只在「单 agent 真做不动 + 任务价值 > 多 agent 税 + 三种失败风险可控」时才拆。三种失败——协调掉单、context 不同步、成本爆炸——是单 agent 不可能有的、你多开 agent 那一刻买进来的风险。而它们共同的根、也是这一章的开放问题,是 agent 间可靠通信与协调的协议尚未收敛(A2A 在路上但远未成熟)。下一章(第 12 章)讲 Workflow——当你确实需要多步多角色、但又想要**确定性**而不是让模型自由编排时,把控制流从模型手里拿回来的那条路。
