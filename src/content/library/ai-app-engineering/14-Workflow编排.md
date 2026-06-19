---
title: "Workflow 编排"
slug: "14"
collection: "ai-app-engineering"
order: 14
summary: "为什么这章排在这：第 12 章你把控制权交给了模型——手写了一个 agent loop，让它自己决定下一步调什么工具、跑几步、什么时候停（L4）。第 13 章你让它跑得久。但你很快会发现一个反直觉的事实：你交出去的控制权里，有一大半根本不需要交。"
topics:
  - "AI 应用工程"
tags: []
createdAt: "2026-06-11T07:39:41.000Z"
updatedAt: "2026-06-11T07:39:41.000Z"
---
> 为什么这章排在这：第 12 章你把控制权交给了模型——手写了一个 agent loop，让它自己决定下一步调什么工具、跑几步、什么时候停（L4）。第 13 章你让它跑得久。但你很快会发现一个反直觉的事实：**你交出去的控制权里，有一大半根本不需要交。** 一个 research agent 真正需要"模型临场判断"的步骤，可能只占 20%；剩下 80% 是"查 → 抽 → 排 → 写"这种你闭着眼都能画出流程图的固定动作。把这 80% 也丢给 agent 自由发挥，你买到的不是智能，是**不确定性、抖动、和多花的 token**。这一章要解决的核心问题是：**怎么判断哪些步骤该焊死成确定性 workflow，哪些该留给 agent；以及当一个流程要跑几小时、中途会崩、必须能续跑时，你怎么让它不从头再来。**
>
> 阶段定位：阶段三 Agent · 第三章。对应主线项目 **P3 增量**：把你在第 12 章手写的 research agent 里**可固化的流程改写成确定性 workflow，只把真正开放的步骤留给 agent**。这一步做完，你的 P3 才从"一个会乱跑的 demo"变成"一个大部分路径可预测、出了事能定位、崩了能续跑的系统"。

一句话先把方向定死：**能用 workflow 别上 agent。** 这是 Anthropic 在《Building Effective Agents》里反复强调的工程纪律，也是这一章所有判断的总纲。新手听到"agent"两眼放光，老手听到"agent"先问一句"这步真的需要模型临场决策吗，不能写死吗"。后者才是面试官想筛出来的人。

---

## 一、背景：为什么需要 Workflow 这一层

### 1.1 从 P3 的一个真实痛点切入

你在第 12 章写完 research agent，跑起来了，但跑着跑着你会观察到三件让你血压升高的事：

1. **同一个问题，跑两次路径不一样。** 这次它先搜 web 再查内部库，下次它先查内部库再搜 web，偶尔还会跳过某一步。结果可能都对，但**你没法复现、没法测试、没法跟产品解释"它为什么这次这么干"**。
2. **它在能写死的地方浪费 token。** "把这 5 个搜索结果各总结一段"——这是个纯并行的固定动作，但 agent 把它放进 loop 里一个一个串行做，每一步都带着全部历史上下文（第 13 章那个 O(n²) 窟窿），慢且贵。
3. **某一步崩了，整个任务从头再来。** 第 7 步调外部 API 超时挂了，前 6 步的搜索、抽取、几千 token 的中间结果全部作废，重跑。

这三件事的根因是同一个：**你把"路径其实是确定的"任务，套进了"路径由模型运行时决定"的执行模型里。** Agent 的自由度在这里不是资产，是负债。

**Workflow 就是为了把这部分自由度收回来。** 它的定义很简单：

> **Workflow = 由代码（而非模型）编排控制流的多步 LLM 应用。** 每一步调不调 LLM、调哪个、下一步去哪，是你在代码里写死的；LLM 只负责填某个节点里的具体内容（生成/抽取/判断），不负责决定"流程往哪走"。

回到第 12 章那张自主性谱系表，Workflow 就是 **L3**：

| 档位 | 谁决定控制流 | 一句话 |
|---|---|---|
| L1 Chain 链式 | 你 | 步骤写死，线性流水线 |
| L2 Router 路由 | 你定分支，模型选 | switch-case，case 由模型选 |
| **L3 Workflow 编排** | **你定图，模型填节点** | **你画好 DAG/状态机，模型执行每个节点的内容** |
| L4 单 Agent | 模型决定控制流 | 你不写循环体的 while 循环 |

L3 和 L4 的分界，就是这一章的全部。

### 1.2 确定性 vs 自主性：这是一道工程权衡，不是技术高低

把这句话刻进脑子：**Workflow 不是"低级的 agent"，agent 也不是"高级的 workflow"。它们在『确定性 ↔ 自主性』这根轴上各占一段，你按任务需求选位置，不按"谁更先进"选。**

| 维度 | Workflow（确定性） | Agent（自主性） |
|---|---|---|
| 控制流 | 代码写死，可读、可测、可复现 | 模型运行时决定，每次可能不同 |
| 可预测性 | 高——同输入大概率同路径 | 低——路径运行时"长"出来 |
| 可调试性 | 高——崩在哪个节点一眼看到 | 低——要重放整条 trace 才知道它为啥那么走 |
| Token 成本 | 可估、可控（步数固定） | 易爆（步数 × 累积上下文，见第 12 章成本模型） |
| 适用任务 | 路径能事先枚举 | 路径无法事先枚举 |
| 失败模式 | 某节点失败，定位清晰 | 不收敛、原地打转、走错路 |
| 你的后端类比 | 一个编排引擎跑你定义的 DAG | 一个你不写循环体的 while 循环 |

**工程推论（每条都能独立成一道面试题）：**

1. **"确定性"在生产里是一种功能，不是一种限制。** 能复现、能写单测、能跟下游解释行为——这些都建立在确定性上。一个每次跑路径都不同的系统，你没法对它做回归测试，这在严肃后端里近乎不可接受。所以默认值应该是 workflow，agent 是"确有必要才升级"的特例。
2. **自主性是有成本的，而且成本不只是 token。** 你每多给模型一份控制权，就要多补一份停机保护、多写一段可观测、多扛一类失败模式。第 12 章那句"让渡多少自主，就要补回多少停机保护"在这里换个说法：**自主性的真实价格 = token 成本 + 可靠性工程成本，别只算前者。**
3. **判断该用哪个，看『路径能否事先枚举』，不看『任务复不复杂』。** 一个很复杂但路径固定的 ETL（十几步、各种分支）该用 workflow；一个看起来简单但"得读了才知道下一步"的任务（如"把这文档改成 PR"）才该用 agent。复杂度和自主性需求是两个正交的维度，新手最爱混。
4. **现实系统几乎都是混合体，不是纯 workflow 或纯 agent。** 最常见的生产形态是：**外层 workflow 把流程焊死，在其中某一两个"真正需要临场判断"的节点里嵌一个小 agent。** 这就是本章 P3 增量的核心姿势——不是"workflow 还是 agent"二选一，是"在 workflow 的骨架上，精准地点几个 agent 节点"。

> 心法（本节）：**先假设不需要 agent，再去证明哪一步非 agent 不可。证明不出来的地方，全部写死。**

---

## 二、演进：以前怎么做，为什么不够

### 2.1 第一阶段——裸 Chain（你在 P0/P1 已经在做了）

最早大家就是手撸：函数 A 调 LLM，把输出喂给函数 B 再调 LLM，串起来。这就是 **prompt chaining（提示链）**，最基础的 workflow 模式。

```python
def summarize_then_translate(doc: str) -> str:
    summary = call_llm(f"用三句话总结：\n{doc}")        # 第一步
    translated = call_llm(f"把这段翻译成英文：\n{summary}")  # 第二步
    return translated
```

够用，但只要流程一长，问题就来了：**没有状态管理、没有错误恢复、没有分支、没有可观测。** 第二步崩了，第一步的结果就丢了；想加个"如果总结质量不行就重新总结"的分支，代码立刻变成意大利面。

### 2.2 第二阶段——框架接管 loop 和 state（LangChain 时代）

于是出现了 LangChain 这类框架，帮你把"调用链 + prompt 模板 + 输出解析"封装起来。它解决了"拼装"的繁琐，但 2024 年大家很快发现一个结构性缺陷：**LangChain 的 `Chain` 抽象本质是『线性的、隐式的』——它假设流程是一条直线，而真实的 agentic 流程是有环、有分支、有条件回退的图。** 你想表达"评审不通过就回到上一步重写"这种循环，在线性 chain 里要么写得极别扭，要么 hack。

更要命的是**状态是隐式的**：数据在链里流动，但"当前整个流程的状态长什么样"没有一个显式的、你能打印出来检查的对象。调试时你看不清"现在到哪了、状态是什么"。

### 2.3 第三阶段——显式状态机/图（LangGraph 时代，当前主流）

LangGraph（LangChain 团队为解决上述问题做的新抽象）把模型换了：**不再是"链"，而是"图（graph）+ 显式状态（state）"。** 这是当前做复杂 workflow/agent 编排的主流范式。它的三个核心概念，你用后端直觉一秒就懂：

- **State（状态）**：一个显式的、贯穿整个流程的数据对象（通常是个 dict/TypedDict）。每个节点读它、改它。**这就是你的"流程上下文"，看得见、可序列化。**
- **Node（节点）**：一个函数，输入 state，输出对 state 的更新。节点里可以调 LLM，也可以纯执行代码。**这就是 DAG 里的一个 task。**
- **Edge（边）**：定义节点之间怎么走。普通边 = 固定下一步；**条件边（conditional edge）= 一个函数看 state 决定下一步去哪个节点**——这就是分支和循环的来源。

```
        ┌─────────┐
        │  START  │
        └────┬────┘
             ▼
       ┌──────────┐
       │  search  │  ← Node：调检索
       └────┬─────┘
            ▼
       ┌──────────┐
       │ evaluate │  ← Node：调 LLM 判断质量
       └────┬─────┘
            │  conditional edge：看 state.quality
      ┌─────┴─────┐
   不合格│           │合格
      ▼           ▼
  ┌────────┐   ┌────────┐
  │ search │   │ write  │  ← 注意这条边形成了【环】
  └────────┘   └───┬────┘
   (回到上一步)      ▼
                ┌──────┐
                │ END  │
                └──────┘
```

**为什么图模型是对的：** 因为 agentic 流程的本质就是一张有向图——有顺序、有分支、有环。用图来表达图，天经地义；用线性 chain 来硬套图，就是阻抗失配。LangGraph 的另一个关键价值是 **state 显式**：任何时刻你都能 dump 出当前 state 检查，调试体验和 chain 不是一个量级。

**工程推论：**

5. **"显式状态"是 LangGraph 相对 LangChain 的根本进步，不是 API 糖。** 隐式状态 = 不可观测 = 不可调试。把流程状态变成一个你能打印、能存盘、能断点续跑的对象，整个可靠性工程才有抓手。这一点直接通向本章后半的 checkpoint。
6. **会用图模型表达『循环』是 L3 的关键能力。** Router（L2）只能在你给的几条直路里选一条，表达不了"不满意就回去重做"。一旦你的流程需要环（重试、迭代优化），你就从 L2 进了 L3。
7. **图模型不等于 agent。** 这是个高频误区：很多人以为"用了 LangGraph = 做 agent 了"。不对。**LangGraph 是编排工具，它既能编排确定性 workflow（边都是你写死的），也能编排 agent（边的走向由模型输出决定）。** 用了 graph 框架，你的系统是 L3 还是 L4，取决于"控制流是代码决定还是模型决定"，不取决于你用了哪个库。

---

## 三、现状：五种主流 Workflow 模式（必须全部掌握）

Anthropic 在《Building Effective Agents》里把生产中反复出现的 workflow 模式归纳成五个。**这五个是面试硬通货，你要能对每一个说出：它解决什么、长什么样、什么时候用、坑在哪。** 下面逐个过。

### 3.1 Prompt Chaining（提示链）

**一句话**：把一个大任务拆成固定的几步，每步一次 LLM 调用，前一步输出喂给后一步。

**形态**：线性。`A → B → C`。可以在步骤之间插**程序化的 gate（门）**——一段代码检查中间结果，不通过就提前退出或回退。

**什么时候用**：任务能被干净地拆成**固定的、顺序的**子步骤，且每步都更简单。典型：先写大纲 → 再按大纲写正文 → 再校对。

**坑**：

- 步数越多，**累计延迟越高**（串行），且**任一步错会污染后面所有步**。所以中间要加 gate 早停。
- 别为了"看起来有设计感"硬拆。如果一次调用就能干好，拆成三步只是徒增延迟和故障点。

### 3.2 Routing（路由）

**一句话**：先用一次 LLM（或分类器）判断输入属于哪一类，再分发到对应的专门处理分支。

**形态**：一个分类器 + N 个下游分支。`classify → {branch_A | branch_B | branch_C}`。

**什么时候用**：输入有**明显不同的类别**，每类适合用不同的 prompt / 模型 / 流程处理。典型：客服分流（退款走 A、技术问题走 B、闲聊走 C）；**按难度路由到不同档位模型**（简单问题甩给 Haiku，难的给 Opus）。

**坑**：

- **分类本身会错。** 路由错了，后面再完美也白搭。要给一个 fallback 分支（"分不清就走通用处理"），别让分类的边界 case 直接崩。
- **分类器的输出必须被『约束死』，不能裸跑自由文本。** 一个高频生产 bug：让模型"回一个类别名"，它回了 `"退款相关"` 或 `"refund / 也可能是物流"`，你的 `switch` 匹配不上，路由直接漏到默认分支甚至抛异常。正确做法是把分类做成**受约束输出**——用 structured outputs 的 `output_config.format`（`enum` 锁死候选集），或用 `strict: true` 的工具参数校验（见第 4 章结构化输出）。让模型**在物理上无法返回菜单外的标签**，比事后写一堆字符串清洗稳得多。这是 routing 能不能上生产的分水岭。
- 这正是第 12 章谱系表里的 **L2**。Routing 是 workflow，不是 agent——分支是你定的，模型只是选。

> **面试加分点**：按难度/成本路由到不同模型，是生产里最实用的省钱手段之一。"简单分类用 Haiku（~$1/$5 每百万 token），复杂推理用 Opus（~$5/$25）"——能把这句话和具体价格量级说出来，比只会说"用 routing 优化"高一档。（价格按 2026 初版图，会变，记量级和"输出价≈输入价 5 倍"这个不对称。）

### 3.3 Parallelization（并行化）

**一句话**：把能同时干的活并行发出去，再聚合结果。两个子类：

- **Sectioning（分片）**：把一个任务切成**互不依赖**的子任务并行做，再合并。例：长文档分块，每块并行总结，最后拼起来。
- **Voting（投票）**：同一个任务**跑多次**，对结果做投票/取多数。例：让模型判断一段代码有没有漏洞，跑 5 次，多数说有就是有——用冗余换可靠性。

**形态**：扇出（fan-out）→ 扇入（fan-in）。

**什么时候用**：子任务独立（sectioning），或你想用多次采样降低单次随机性的影响（voting）。

**坑（务必记住，是高频考点）：**

- **并行请求和 prompt caching 有个时序陷阱。** 缓存条目要等第一个响应**开始返回**后才可读。你同时发 N 个共享前缀的并行请求，它们谁也读不到别人正在写的缓存——**N 个全部按全价付前缀**。正确姿势：**先发 1 个，等它的首个 token 回来（不必等整条响应），再扇出剩下的 N−1 个**，后面这批才能命中第一个写好的缓存。这个细节直接关系到 sectioning 的成本，面试官很爱问。
- Voting 是用钱买确定性。跑 5 次成本约 5×，要算清楚这份可靠性值不值这个价。

### 3.4 Orchestrator-Workers（编排者-工人）

**一句话**：一个 **orchestrator（编排者）LLM** 动态地把任务拆成子任务、分给多个 **worker** 执行，再综合 worker 的结果。

**形态**：中心化。`orchestrator 拆解 → 派发给 workers → orchestrator 综合`。

**和 Parallelization 的关键区别（必考）**：parallelization 里**子任务是你事先切好的**（固定 sectioning）；orchestrator-workers 里**子任务由 orchestrator 在运行时动态决定**——它读了输入才知道该拆成几个、拆成什么。

**什么时候用**：任务需要拆解，但**你没法事先知道要拆成几块、怎么拆**。典型：一个改代码的需求，要改哪几个文件、每个文件派一个 worker——改几个文件得读了代码才知道。

**坑——这是本章最重要的一个边界判断：**

- **Orchestrator-workers 已经踩在 workflow 和 agent 的边界线上了。** 因为"动态拆解"这一步是模型在运行时决定控制流——这带有 agent 的性质。Anthropic 把它归在 workflow，是因为整体骨架（拆解 → 派发 → 综合）是固定的；但那个"拆"的动作是自主的。**实务上你要清楚：用了 orchestrator-workers，你已经引入了一部分不确定性，要按 agent 的标准给它配停机保护和可观测**，不能当纯 workflow 那样裸跑。

### 3.5 Evaluator-Optimizer（评估者-优化者）

**一句话**：一个 LLM 生成结果，另一个 LLM（evaluator）评估并给改进反馈，循环迭代直到达标。

**形态**：**带环。** `generate → evaluate →（不达标）→ generate → ...→（达标）→ END`。

**什么时候用**：有**明确的评估标准**，且迭代确实能改进结果。典型：翻译（译完 → 评 → 按反馈重译）；写代码（写 → 跑测试/评审 → 改）。**前提是你能清晰地说出"什么叫好"——评估标准模糊，这个模式就退化成无意义的空转。**

**坑：**

- **必须有硬性的循环上限。** 这是个环，evaluator 永远不满意它就永远转。`max_iterations` 是刹车，缺了它你就复现了第 12 章那个"不收敛烧钱"的事故——只不过这次是 evaluator 和 optimizer 互相拉扯着烧。
- 评估标准要**可独立判定**。"让 evaluator 看着办"是耍流氓；要给它具体的、能逐条核对的 rubric（评分细则），否则反馈是噪声，迭代是抖动。（这一点和第 11 章 RAG 评测、以及 LLM-as-judge 的设计是同一套方法论。）

### 五种模式速查表

| 模式 | 控制流形态 | 子任务谁定 | 有没有环 | 一句话适用 | 头号坑 |
|---|---|---|---|---|---|
| Prompt Chaining | 线性 | 你（固定） | 无 | 能干净拆成固定顺序步骤 | 串行延迟累积；中间要加 gate 早停 |
| Routing | 分类+分支 | 你（固定分支） | 无 | 输入有明显类别 | 分类会错，要 fallback 分支 |
| Parallelization | 扇出→扇入 | 你（固定切分） | 无 | 子任务独立 / 多次采样投票 | 并行请求击穿 cache（先发 1 个再扇出）|
| Orchestrator-Workers | 中心派发 | **模型（动态）** | 无 | 拆几块要运行时才知道 | 已踩 agent 边界，要配停机保护 |
| Evaluator-Optimizer | **带环迭代** | 你（固定两角色） | **有** | 有明确评估标准、迭代能改进 | 必须有 max_iterations；rubric 要可判定 |

**工程推论：**

8. **这五种是『组合积木』，不是『五选一』。** 真实系统经常套娃：一个 prompt chain 的某一步是个 parallelization，parallelization 的某个分片里又跑了个 evaluator-optimizer。设计时不是"我用哪种"，是"这一段用哪种最贴"。
9. **带环的模式（evaluator-optimizer、以及任何重试/迭代）是事故高发区。** 凡是有环，第一件事就是问"刹车在哪"——上限步数、上限 token、超时。没有刹车的环 = 定时炸弹。
10. **Orchestrator-workers 是『workflow』和『agent』的语义灰区，能讲清这个灰区是高级信号。** 面试时若你能主动指出"orchestrator-workers 名义是 workflow，但动态拆解那步带 agent 性质，所以我会按 agent 标准给它配可观测"，面试官会立刻给你加分——这说明你不是在背模式名，是真懂边界。

---

## 四、落地：生产环境怎么选、大厂怎么做

### 4.1 第一性决策：这个流程，哪些步骤该写死，哪些该留给 agent？

这是 P3 增量要你亲手做的事，也是这一章的实操内核。给你一个可执行的判断流程，逐节点过你的 research agent：

**对流程里的每一个步骤，问三连：**

1. **这步的『下一步去哪』，是固定的，还是要看这步的结果临场决定？**
   - 固定 → 写成 workflow 的普通边。
   - 要看结果在几条已知路里选 → 写成**条件边（routing）**，仍是 workflow。
   - 要看结果"长出"一条事先不知道的路 → 这步才可能需要 agent。
2. **这步本身，是『纯执行/固定模板生成』，还是『需要模型自由发挥规划』？**
   - 查库、调 API、按模板总结、格式化 → 纯执行节点，连 LLM 都未必要调。
   - "读了这堆资料，自己决定还缺什么、再去查什么" → 这才是 agent 节点。
3. **这步崩了，影响面多大？需不需要单独能重试/续跑？**
   - 影响大、易失败（外部 API、长耗时）→ 必须是独立可重试的节点（接下面 4.2 的 checkpoint）。

**过完三连，你的 research agent 会分层成这样（P3 目标态）：**

```
外层：确定性 Workflow（你写死的图）
  ├─ Node: 解析用户问题           ← 纯/固定 prompt
  ├─ Node: 并行检索（web + 内部库）  ← parallelization，固定扇出
  ├─ Node: 各结果并行总结           ← parallelization
  ├─ Node: 评估资料是否足够          ← 条件边（routing）：够→写作 / 不够→进 agent
  │        └─[不够]→ Node: 缺口补查 Agent  ← ★这里才嵌一个真 agent（开放步骤）
  ├─ Node: 综合写作                ← 固定模板 + LLM
  └─ Node: evaluator 校验 →(不达标回写作)  ← evaluator-optimizer，带 max_iterations
```

看清楚这个结构：**80% 是写死的 workflow 节点，只有"缺口补查"这一个真正开放的步骤留给 agent。** 这就是"能用 workflow 别上 agent"落到代码上的样子——不是不用 agent，是把 agent 收缩到它唯一不可替代的那个点。

**工程推论：**

11. **"把 agent 改成 workflow"在简历和 PR 里是强信号。** 第 12 章说过"我们判断这个场景不需要 agent、用 workflow 更稳"才是加分项。P3 增量正是这句话的实践：你不是删功能，你是**用确定性换可靠性，并精确保留了非 agent 不可的那一点自主性**。这种"收缩自主性边界"的判断力，正是高级和资深的分水岭。
12. **每多焊死一个节点，你就少一类线上故障。** 把一个原本由 agent 自由发挥的步骤改成固定节点，等于把"它可能乱走"这个失败模式直接从系统里删掉。可靠性是减出来的，不是加出来的。

### 4.2 Checkpoint 与 Durable Execution：长流程崩了怎么续跑

这是从"能跑"到"能上线"的关键工程门槛，也是这一章后端老手最该出彩的地方——因为它和你熟悉的分布式系统知识直接接轨。

**问题**：一个 workflow 跑了 6 步、几分钟、几千 token，第 7 步调外部 API 超时崩了。默认行为是整个任务作废重跑——前 6 步白干。流程越长、越值钱，这越不可接受。

**解法的核心概念：durable execution（持久化执行）。** 一句话定义：

> **把工作流的执行状态持久化到外部存储，使得进程崩溃、机器重启后，流程能从『上次成功的那一步』继续，而不是从头开始。**

它由两个机制支撑：

- **Checkpoint（检查点）**：每个节点执行完，把当前 state（本章 2.3 那个显式 State 对象）**存盘**。LangGraph 内置了 checkpointer（可落到内存/SQLite/Postgres/Redis）——每步之后自动存。崩溃后用同一个 `thread_id` 恢复，它会从最后一个 checkpoint 加载 state 继续。
- **Replay / 可恢复（重放）**：更彻底的方案（如 **Temporal**）把"执行历史"持久化成一个事件日志。恢复时**重放这段历史**来重建到崩溃前的状态，然后继续往下跑。

**你的后端直觉锚点（用熟悉的东西接住这个新概念）：**

| Durable Execution 概念 | 你已经会的东西 |
|---|---|
| Checkpoint 每步存 state | DB 事务 / 定期 snapshot |
| Replay 事件日志恢复 | 事件溯源（Event Sourcing）/ WAL 重放 |
| `thread_id` 标识一条可恢复的流程 | 分布式任务的幂等 ID / saga ID |
| 节点必须可安全重试 | 接口幂等性（idempotency） |

**Temporal vs LangGraph checkpointer——生产怎么选：**

| 维度 | LangGraph Checkpointer | Temporal |
|---|---|---|
| 定位 | 框架内置的状态持久化 | 独立的 durable execution 平台 |
| 恢复方式 | 从最后一个 checkpoint 加载 state | 重放事件历史重建状态 |
| 重量 | 轻，几行配置，跟 LangGraph 绑定 | 重，要部署 Temporal 集群、写 worker |
| 适合 | 中小流程、Agent state 续跑、人工介入暂停 | 长时间（小时/天）、高可靠、跨服务的关键业务流程 |
| 一句话 | "够用就上它" | "命悬一线的长流程才值得这套基建" |

**坑（这是 durable execution 的头号暗坑，不懂会出数据事故）：**

- **可重放/可恢复的前提是『节点幂等』。** 重放或重试会**重新执行某些步骤**。如果一个节点有副作用（扣款、发邮件、写库），重放时它会**再执行一遍**——重复扣款、重复发邮件。所以：**所有带副作用的节点必须做幂等**（带幂等键、先查后写、或把副作用收敛到"恰好一次"的保证里）。这和第 12 章 agent 工具设计里"难以撤销的操作要门控"是同一个安全直觉，在这里升级成了"可重放系统里副作用必须幂等"。

**工程推论：**

13. **Durable execution 不是新魔法，是把你已会的分布式可靠性套路搬到了 LLM 工作流上。** Checkpoint = snapshot，replay = event sourcing，节点幂等 = 接口幂等。能把这套映射讲出来，面试官立刻知道你不是只会调 API 的人——你是把后端硬功夫迁移过来的人。
14. **选 checkpointer 还是 Temporal，看『流程时长 × 失败代价』。** 几十秒的流程崩了重跑也无所谓，别上重型基建（over-engineering）。跑几小时、中途崩了损失巨大的关键流程，才值得 Temporal 这套重量级可恢复执行。**默认从 LangGraph checkpointer 起，疼了再上 Temporal。**
15. **显式 state（本章 2.3 / 4.1）是 checkpoint 的前提。** 状态如果是隐式的、散在闭包里的，你根本没法存盘和恢复。这就是为什么 LangGraph 的"显式 state"在 2.3 被反复强调——它不只是调试友好，它是 durable execution 的地基。（别和第 13 章混：第 13 章管的是 agent 跨轮的"工作记忆/长期记忆"——往 context 里塞什么、记什么、忘什么；本章这个 state 是"流程执行到哪一步"的可序列化快照。一个是记忆内容，一个是执行进度，两套东西。）

### 4.3 人工介入点（Human-in-the-Loop）设计

不是所有步骤都该让系统自己拍板。**高风险、不可逆、或模型没把握的步骤，要能暂停、等人批准、再继续。** 这在 workflow 里是一等公民设计，不是补丁。

**两种典型介入点：**

1. **审批门（approval gate）**：执行某个危险动作前（发对外邮件、删数据、提交付款、合 PR）暂停，把"我打算这么做"推给人，等 allow/deny。
2. **断点续填（pause & resume）**：流程跑到一半需要外部信息（用户补个参数、上游系统给个值），暂停，挂起 state，拿到输入后从断点继续。

**实现上靠什么？——又回到 checkpoint。** 人工介入的本质是"在某节点把 state 存盘并停住，等一个外部事件（人的决定）再加载 state 继续"。所以 **HITL 和 durable execution 是同一套机制的两个用途**：一个是"崩了续跑"，一个是"主动暂停等人"。LangGraph 的 `interrupt` / 断点能力就是干这个的；Anthropic 的 Managed Agents 里也有对应的 `always_ask` 权限策略——工具调用前触发 idle、等你回一个 `tool_confirmation` 才往下走（见第 12 章托管 agent，落地细节第 18 章可观测/审计会再碰）。

**坑：**

- **暂停期间 state 必须落盘，不能只挂在内存里。** 人可能十分钟后才点"批准"，这期间进程重启了，内存里的挂起状态就没了。HITL 的暂停必须是**持久化的暂停**（存到 checkpointer），否则一重启就丢。
- **审批门要给人足够的上下文做决策。** 别只弹"是否批准 [Y/N]"，要把"我打算干什么、为什么、影响是什么"一起给。门控的价值在于人能做出**知情**判断，信息不够的门控是橡皮图章。

**工程推论：**

16. **人工介入点不是"不信任 AI 的妥协"，是『把不可逆操作的最终决定权留在人手里』的可靠性设计。** 接第 12 章工具设计的原则：可逆的让系统自动跑，难撤销的（删除、付款、对外发送）门控到人。这条线在 agent、workflow、托管 agent 里是一致的。
17. **HITL 复用 durable execution 的基建，不是另起炉灶。** 看懂"暂停等人"和"崩溃续跑"是同一个 checkpoint 机制的两种用法，你的架构就少一套重复轮子——这是"设计可复用"的体现。

### 4.4 编排框架对比与选型（含国内语境）

**先说一个反框架的判断**：你不一定需要一个重型编排框架。**Anthropic 在《Building Effective Agents》里明确建议：先从直接调 API 开始，框架是当复杂度真的撑不住时才引入的，别一上来就套 LangGraph。** 你在 P0/P1 手写 chain 就是这个意思——先理解底层，再决定要不要框架。

什么时候该上框架？当你需要：复杂的图结构（多分支多环）、内置的 checkpoint/续跑、可视化调试、团队协作的统一抽象。下面是主流选型：

| 框架/方案 | 模型 | 强项 | 适合 |
|---|---|---|---|
| **裸 API + 手写编排** | 你自己 | 零黑盒、完全可控、无框架税 | P0/P1 学习期；逻辑简单；想完全掌控 |
| **LangGraph** | 显式图 + state + checkpointer | 复杂 workflow/agent 编排、续跑、HITL | 当前做严肃 agent 编排的主流默认 |
| **Temporal** | 持久化事件历史 + worker | 工业级 durable execution、跨服务长流程 | 跑数小时、高可靠、关键业务流程 |
| **Anthropic Managed Agents** | 托管 agent loop + 沙箱容器 | 平台帮你跑 loop、提供工具执行环境、内置 checkpoint/HITL | 想要"一个有工作区、能续跑、带审批"的托管 agent（见第 12 章）|
| **云厂商编排**（如 AWS Step Functions） | 状态机 DSL | 和云生态深度集成、运维成熟 | 已重度绑定某云、要和现有基建打通 |

**国内语境**：豆包（字节）、Qwen（阿里）、DeepSeek、Kimi（月之暗面）、GLM（智谱）、混元（腾讯）、文心（百度）这些模型，**编排层是与模型解耦的**——LangGraph 这类框架编排的是"流程图"，节点里调哪家模型是可替换的。所以选型上：**编排框架按上面的表选，模型按能力/价格/合规按需接**。各家也有自己的 agent/编排平台（如阿里百炼、字节扣子/Coze），但跨厂、可移植的工程能力，仍建立在"自己懂图模型 + durable execution"这套通用功底上，而不是绑死某个平台的可视化拖拽。（厂商平台清单按 2026 初版图，会变，学方法论别背名单。）

**工程推论：**

18. **框架是来还技术债的，不是来炫技的。** "能用 workflow 别上 agent"还有个孪生兄弟："能不上框架先别上框架"。框架带来抽象红利，也带来黑盒成本和学习曲线。**判断标准：你手写的编排是不是已经痛到（状态难管、续跑难做、分支太乱）值得为框架的抽象付费了？** 没痛到就别引。
19. **编排层和模型层要解耦，这是可移植性的来源。** 把"流程怎么走"（编排）和"某步用哪个模型"（推理）分开，你才能在不重写流程的前提下换模型（成本/合规/能力驱动）。绑死某厂可视化平台的代价，是把这层解耦能力交了出去。

> 心法（本节）：**默认不上 agent、默认不上框架、默认把流程焊死。每一次"上"，都要有一个具体的、说得出口的疼点来支撑。**

---

## 五、面试视角：高频题与考点

面试官在这一章想区分的，**不是你会不会用 LangGraph，是你有没有『确定性优先』的工程品味、能不能在 workflow/agent 边界上做出带 trade-off 的判断。** 会调 API 的人很多，敢说"这里不该上 agent"并扛住追问的人少。

### 5.1 面试官在这一章筛什么（四条判别轴）

这一章是阶段三里"显工程成熟度"的章节，不是考工具熟练度。面试官手上通常有这四把尺子，你答任何一道题都要主动往这四条上靠：

| 判别轴 | 弱信号（被刷） | 强信号（加分） |
|---|---|---|
| **默认值** | 张口就是 agent、上来就 LangGraph | "先假设不需要 agent / 不需要框架，再证明哪步非它不可" |
| **判据** | 用"任务复不复杂"决定上不上 agent | 用"**路径能否事先枚举**"决定，且能说清它和复杂度正交 |
| **边界感** | 把 5 种模式当名词背 | 能主动点破灰区（orchestrator-workers 名义 workflow、实带 agent 性质） |
| **可靠性** | 只谈"功能能跑" | 谈停机保护、checkpoint、幂等、刹车——把后端硬功夫迁过来 |

### 5.2 三个"一票否决"的高频追问

下面这三问是面试官用来**快速证伪**的——答错一个，前面讲得再漂亮也大幅扣分。它们都不考记忆，考的是你有没有真在生产里踩过：

1. **"你这个带环的流程，刹车在哪？"** —— 任何 evaluator-optimizer / 重试 / 迭代，第一反应必须是 `max_iterations` + token 预算 + 超时。答不出刹车 = 没上过线。这是第 12 章"不收敛"事故在 workflow 形态下的复刻，两章共用一个判断。
2. **"重放/重试时，那个发邮件（扣款）的节点会怎样？"** —— 必须立刻接到**节点幂等**。答不到幂等，说明你只把 durable execution 当"能续跑"的好处，没看见它"会重复执行副作用"的代价。
3. **"这个场景，你真的需要 agent 吗？"** —— 这是个**陷阱题**，标准动作是先尝试论证"不需要、用 workflow 更稳"，而不是为了显得高级硬塞一个 agent。敢说"这里不该上 agent"并给出 trade-off，比会搭 agent 更值钱。

### 5.3 一分钟自检：你能不能扛住"为什么"的连环追问

面试官最爱用"为什么"连环钻。把下面这条链子能顺下来，这一章就稳了：

> 为什么默认 workflow？→ 因为确定性能复现、能测、能解释。
> 为什么不用复杂度判断？→ 复杂但路径固定的也该 workflow；复杂度和自主性需求正交。
> 那 agent 什么时候必要？→ 路径"得读了才知道下一步"、无法事先枚举时。
> 上了 agent 多付什么？→ token + 一整套可靠性工程（停机保护、可观测、幂等）。
> 怎么把代价压到最小？→ 外层 workflow 焊死，只在唯一非它不可的节点点一个 agent。

（具体高频题、答题框架、设计题、系统题、代码题，见第六部分"章末五件套"。）

---

## 六、未来演进：未来 3-5 年方向，对应用工程师的含义

1. **模型越强，边界越往 agent 漂，但"确定性优先"的纪律不会过时。** 单 agent 能力持续抬高（第 12 章已提），今天必须焊死的步骤，明天可能模型自己就能稳定走对。但这不意味着"以后全用 agent"——**对可复现、可审计、可测试有硬要求的环节（金融、医疗、合规），确定性 workflow 永远有位置**。趋势是边界移动，不是边界消失。
2. **编排从"采样超参"彻底转向"自然语言 + 预算控制"。** 注意 Anthropic 这代模型（Opus 4.7/4.8、Fable 5）已经移除了 `temperature/top_p/top_k`，行为控制主通道迁移到了 prompt + thinking effort 档位 + task budget。**这对编排的含义：你越来越不是靠调采样参数控制每个节点的行为，而是靠『说清楚 + 给对预算』。** 节点里的控制手段在变，编排骨架的方法论不变。
3. **Durable execution 会成为 agent 基建的标配，而非高级选项。** 随着 agent 跑的任务越来越长（数小时的自主任务正在变常见），"崩了能续跑"从加分项变成入场券。Temporal 这类平台和 LangGraph 的 checkpointer 会越来越被当作"理所当然该有"的东西。**应用工程师的含义：现在就把 durable execution 当基本功练，不是当 niche 技能。**
4. **托管编排（Managed Agents 类）会吃掉一部分自建。** 平台帮你跑 loop、管 state、配沙箱、做审批——对很多团队，自建编排+续跑的性价比在下降。**但平台做不到的（开源可控、跨厂可移植、深度定制的图逻辑），仍是自建的护城河。** 判断"自建还是用托管"会和"自建还是买"一样，成为高级工程师的常规决策题（接第 12 章的 build-vs-buy 思路）。

> **本章总心法：Workflow 是『你把流程焊死、只在非焊不可处留一个 agent 焊点』的艺术。确定性不是落后，是你能对系统负责的前提——能用 workflow 别上 agent，能少给一份自主就少补一份停机保护。**

---

## 七、章末五件套

### 7.1 高频面试题（10 道，附答题框架）

**Q1. workflow 和 agent 的本质区别是什么？什么时候该用哪个？**
- 框架：本质区别在**控制流谁决定**——workflow 是代码决定（确定性），agent 是模型运行时决定（自主性）。判断标准是**『路径能否事先枚举』**：能枚举→workflow，必须运行时一步步长出来→agent。
- 必说的判断：**默认 workflow，agent 是确有必要才升级**（Anthropic "能用 workflow 别上 agent"）。
- 错误答案陷阱：用"任务复不复杂"做判断（复杂但路径固定的也该 workflow）；或把 agent 当"更高级"而无脑选。

**Q2. 讲一下五种 workflow 模式，并各举一个适用场景。**
- 框架：prompt chaining（固定顺序拆步，如大纲→正文）、routing（分类分发，如客服分流/按难度选模型）、parallelization（独立子任务并行 sectioning / 多次采样 voting）、orchestrator-workers（动态拆解派发，如改代码派 worker）、evaluator-optimizer（生成-评估迭代，如翻译/写码带反馈环）。
- 加分：指出 parallelization 的 sectioning（你切）vs orchestrator-workers（模型切）的区别；指出 evaluator-optimizer 必须有 max_iterations。

**Q3. orchestrator-workers 到底算 workflow 还是 agent？**
- 框架：名义是 workflow（骨架固定），但"动态拆解"那步是模型运行时决定怎么拆，**带 agent 性质**，踩在边界上。
- 加分：所以实务上要按 agent 标准给它配停机保护和可观测，不能当纯 workflow 裸跑。
- 这题在筛"能不能讲清灰区"——能主动点破边界模糊性的是高级信号。

**Q4. evaluator-optimizer 这种带环的模式，最大的风险是什么？怎么防？**
- 框架：环不收敛——evaluator 永远不满意就永远转，复现"不收敛烧钱"。
- 防：**硬性 max_iterations**（步数刹车）+ token 预算 + 超时；evaluator 的 rubric 必须**可独立判定**，模糊标准会让迭代退化成抖动。
- 错误答案：只说"加个重试上限"而不提"评估标准要可判定"——漏了一半。

**Q5. 什么是 durable execution / checkpoint？为什么需要？**
- 框架：把工作流执行状态持久化到外部存储，使进程崩溃/重启后能从**上次成功的步骤**续跑，而非从头。机制：checkpoint（每步存 state）+ replay（重放事件历史恢复）。
- 锚点：checkpoint≈DB snapshot，replay≈event sourcing/WAL。
- 加分：点出**前提是节点幂等**——重放/重试会重新执行副作用步骤。

**Q6. 可恢复/可重放的工作流，有什么必须注意的副作用陷阱？**
- 框架：**节点幂等性**。replay/retry 会重新执行步骤，带副作用的节点（扣款、发邮件、写库）会重复执行 → 重复扣款。
- 解法：带副作用的节点做幂等（幂等键、先查后写、恰好一次保证）。
- 这题和"接口幂等性"是同一个后端直觉，能迁移过来就是加分。

**Q7. Temporal 和 LangGraph 的 checkpointer，怎么选？**
- 框架：按"流程时长 × 失败代价"选。中小流程/agent state 续跑/HITL 暂停→LangGraph checkpointer（轻、几行配置）；数小时/高可靠/跨服务关键业务→Temporal（重、独立平台、事件重放）。
- 必说：**默认从 checkpointer 起，疼了再上 Temporal**，别一上来 over-engineer。

**Q8. 人工介入点（HITL）在 workflow 里怎么实现？**
- 框架：本质是"在某节点把 state 持久化并暂停，等外部事件（人的决定）再加载续跑"——**复用 durable execution 的 checkpoint 机制**，和"崩溃续跑"是同一套基建的两种用法。
- 坑：暂停期间 state 必须落盘（人可能十分钟后才批，期间可能重启）；审批门要给足上下文让人做知情判断。
- 原则：可逆操作自动跑，难撤销的（删除/付款/对外发送）门控到人。

**Q9. parallelization 和 prompt caching 一起用，有什么坑？**
- 框架：缓存条目要等第一个响应**开始返回**才可读。同时发 N 个共享前缀的并行请求，谁也读不到别人在写的缓存，**N 个全按全价付前缀**。
- 解法：**先发 1 个，等首 token 回来再扇出剩下 N−1 个**，后面这批才命中缓存。
- 这是个细节考点，答得出说明真在生产里抠过成本。

**Q10. 你会用 LangGraph 来做纯确定性 workflow 吗？用了 graph 框架就是做 agent 了吗？**
- 框架：会，且"用了 LangGraph≠做 agent"。**LangGraph 是编排工具，既能编排确定性 workflow（边你写死）也能编排 agent（边由模型输出决定）。** 是 L3 还是 L4 取决于"控制流谁决定"，不取决于用了哪个库。
- 错误答案陷阱：以为"用了图框架=升级成 agent 了"——混淆了"工具"和"自主性档位"。

### 7.2 实战项目增量（P3：把 agent 收缩成 workflow + agent 混合体）

**任务**：基于第 12 章手写的 research agent，做"确定性化"改造。

**交付物**：一个 LangGraph（或等价手写编排）实现的 research 流程，结构为"确定性 workflow 骨架 + 一个 agent 节点"。

**明确验收标准（缺一不可）：**
1. **可固化步骤已写死**：解析问题、并行检索、并行总结、综合写作至少这几步是确定性节点（普通边/条件边），不再交给 agent 自由决定先后。
2. **agent 节点精确收缩**：整个流程里**只保留一个**真正开放的 agent 节点（如"缺口补查"），并能在 README 里用一段话说清"为什么只有这一步非 agent 不可，其余为什么能写死"。
3. **带环模式有刹车**：写作-校验若用了 evaluator-optimizer，必须有 `max_iterations`，且能演示触发上限时优雅停止（不抛异常、标注"达到迭代上限"）。
4. **checkpoint 续跑可演示**：接入 checkpointer（内存/SQLite 均可），能演示"中途 kill 进程后，用同一 thread_id 恢复，从最后成功节点续跑"，而非从头重跑。
5. **副作用节点幂等**：若流程里有任何带副作用的节点（写文件/调外部 API），说明它如何保证重放安全（幂等键/先查后写），并在 README 标注。
6. **可复现性对比**：同一输入跑两次，记录路径——确定性节点路径应一致；写一段对比说明"相比第 12 章纯 agent，路径可预测性提升在哪"。

**自检红线**：如果你的"workflow"里每一步还是 agent 自己决定先后，那你只是换了个框架，没做这一章的事——重看 4.1 的三连判断。

### 7.3 设计题（开放，考权衡判断）

> 你要给一家电商做"自动客诉处理"系统：用户提交一条投诉，系统要分类（退款/物流/质量/其他）、检索订单和历史、判断是否需要退款、若退款则发起退款并通知用户。请设计这个系统的编排，并明确：**哪些步骤是确定性 workflow，哪些（如果有）需要 agent，哪些需要人工介入门控，哪些节点必须幂等，整体用不用重型编排框架。** 说清每个决策的 trade-off。

考点：能不能识别出"分类=routing""检索=固定节点""退款发起=必须幂等 + 必须人工/规则门控""通知=有副作用"，并判断这个流程其实大部分可写死、agent 需求很弱、是否值得上 Temporal（看退款金额风险）。**好答案的标志是敢说"这个场景几乎不需要 agent"，而不是为了显得高级硬塞一个 agent。**

### 7.4 系统设计题（含量级估算）

> 设计一个"文档批量翻译 + 质检"workflow 平台：每天处理 **10 万篇**文档，每篇平均 **3000 token**，翻译用 evaluator-optimizer（平均迭代 2 轮：译 1 次 + 评 1 次 + 偶尔重译）。请估算：每日 token 量级、用 Batch API 能否降本、哪些环节需要 checkpoint、并发与限流怎么配。

估算要点（给出量级即可，体现思路）：
- **Token 量级**：10 万篇 × 3000 token ≈ 3 亿 input token/天（仅翻译输入）。算上 evaluator 评估读入 + 重译，乘以迭代系数（约 2–3×）→ 单日 input 量级在 6–9 亿 token。输出按"译文≈原文长度"再算一份，且**输出价≈输入价 5 倍**，输出是成本大头。
- **Batch API 能不能用——关键判断**：**翻译+质检这种『离线、单篇可独立完成、不要求实时』的任务，正是 Batch API 的理想场景**（异步、最长 24h、50% 价格）。和第 12 章那个反例对照记牢：**Batch 跑不了『交互式多步 agent loop』，但跑得了『大量可独立离线完成的单次任务』。** 这里每篇翻译就是一个可独立的任务 → 整批塞 Batch，直接省 50%。这正反两个判断一起答，才显出你真懂 Batch 的边界。
- **evaluator-optimizer 的迭代上限**：必须设 max_iterations（如 2–3），否则 10 万篇里只要有一批难翻的卡在环里反复重译，成本和延迟双爆。
- **Checkpoint**：单篇翻译短、崩了重跑无所谓，**不必给单篇上重型 durable execution**；但"整批 10 万篇的处理进度"要 checkpoint（处理到第几篇），批处理中断能从断点续跑，不重头扫。
- **并发/限流**：10 万篇/天 ≈ 平均 1.16 篇/秒，但用 Batch 就不按在线 QPS 算了，按 Batch 的吞吐和配额规划（单批最多 10 万请求/256MB，正好一天一批或拆几批）。
- **caching**：evaluator 的系统 prompt / 评分 rubric 在所有篇目间复用 → 加 `cache_control` 缓存这段共享前缀（注意第 13 章/本章 3.3 的并发缓存时序，Batch 内同理要让共享前缀先写后读）。

**好答案标志**：算得出 token 量级；**正确判断这个场景该上 Batch（和第 12 章 agent loop 不该上 Batch 形成对照）**；分清"单篇不必 checkpoint、整批进度要 checkpoint"的粒度；把 caching 当一等优化。

### 7.5 代码题（带 TODO 骨架 + 测试要求 + 暗坑）

> **要求**：用 LangGraph 实现一个**带 checkpoint 续跑 + evaluator-optimizer 环（有迭代上限）**的最小翻译 workflow。补全 TODO。模型用 `claude-opus-4-8`，adaptive thinking。

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import anthropic

client = anthropic.Anthropic()


# 显式 state：贯穿整个流程、可序列化、checkpoint 存的就是它
class TransState(TypedDict):
    source: str          # 原文
    draft: str           # 当前译文
    feedback: str        # evaluator 的反馈
    iterations: int      # 已迭代次数（刹车计数器）
    done: bool           # 是否通过质检


MAX_ITERS = 3            # ★ evaluator-optimizer 的硬刹车


def translate_node(state: TransState) -> dict:
    """生成/重译节点：有 feedback 就按 feedback 改，没有就首译。"""
    prompt = f"把下面内容翻译成英文：\n{state['source']}"
    if state.get("feedback"):
        prompt += f"\n\n上一版译文有以下问题，请改进：\n{state['feedback']}\n上一版：\n{state['draft']}"
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},   # 暗坑1：新模型必须显式开 adaptive，且不能传 budget_tokens/temperature
        messages=[{"role": "user", "content": prompt}],
    )
    draft = next(b.text for b in resp.content if b.type == "text")
    # TODO 1: 返回对 state 的更新——更新 draft，并把 iterations + 1
    ...


def evaluate_node(state: TransState) -> dict:
    """评估节点：按可判定的 rubric 给出『PASS』或具体改进反馈。"""
    rubric = "评估标准：1.语义准确无遗漏 2.术语一致 3.语法地道。全部满足只回 PASS，否则逐条指出问题。"
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": f"{rubric}\n\n原文：\n{state['source']}\n译文：\n{state['draft']}"}],
    )
    verdict = next(b.text for b in resp.content if b.type == "text")
    # TODO 2: 若 verdict 判为通过 -> 置 done=True、清空 feedback；
    #         否则 -> done=False、feedback=verdict
    ...


def route_after_eval(state: TransState) -> Literal["translate", "__end__"]:
    """条件边：决定迭代环走向。这是【环 + 刹车】的关键。"""
    # TODO 3: 若 state['done'] 为 True -> 返回 END（结束）
    #         若 iterations >= MAX_ITERS -> 也返回 END（撞上限，优雅停止，不再重译）
    #         否则 -> 返回 "translate"（回去按 feedback 重译，形成环）
    ...


# 组装图
builder = StateGraph(TransState)
builder.add_node("translate", translate_node)
builder.add_node("evaluate", evaluate_node)
builder.add_edge(START, "translate")
builder.add_edge("translate", "evaluate")
# TODO 4: 用 add_conditional_edges 把 evaluate 之后的走向交给 route_after_eval
#         （这条条件边既能回 translate 形成环，也能去 END）
...

# 暗坑2：必须传 checkpointer，否则没法续跑；调用时必须给 thread_id
graph = builder.compile(checkpointer=MemorySaver())


def run(source: str, thread_id: str) -> TransState:
    config = {"configurable": {"thread_id": thread_id}}
    init: TransState = {"source": source, "draft": "", "feedback": "", "iterations": 0, "done": False}
    return graph.invoke(init, config)
    # 续跑：用同一个 thread_id 再 invoke(None, config)，会从最后 checkpoint 继续
```

**测试要求（必须覆盖）：**
1. **正常通过**：给一段简单文本，mock evaluator 在第 1 轮就回 PASS，断言 `done=True` 且 `iterations==1`。
2. **撞迭代上限**：mock evaluator **永远不 PASS**，断言流程在 `iterations==MAX_ITERS` 时**停止**（`done` 仍为 False，但**没有无限循环、没抛异常**）——这是验证刹车。
3. **环确实成环**：mock evaluator 第 1 轮给 feedback、第 2 轮 PASS，断言 `translate_node` 被调用了 2 次（验证条件边真的回流形成了环）。
4. **checkpoint 续跑**：跑到一半（mock 在某节点抛一次异常打断），用同一 `thread_id` 恢复 invoke，断言**从中断处继续**而非从头（可通过断言节点调用次数 / state.iterations 不被清零来验证）。
5. **state 可序列化**：断言 `TransState` 里没有不可序列化的对象（checkpoint 要存盘，塞个 socket/连接进去会炸）。

**暗坑提示（不踩会 debug 到怀疑人生）：**
- **暗坑1**：`claude-opus-4-8` 这代**不能传 `budget_tokens`、`temperature/top_p/top_k`**，传了直接 400；adaptive thinking 要显式 `{"type":"adaptive"}`，默认不一定开。（和第 12 章同一个坑，这里再强化一次。）
- **暗坑2（本章核心）**：`compile()` 不传 `checkpointer` 就没有续跑能力；即使传了，**`invoke` 时不给 `thread_id` 照样没法恢复**——`thread_id` 是那条可恢复流程的身份标识，丢了它 checkpoint 形同虚设。
- **暗坑3（致命）**：evaluator-optimizer 是**环**，`route_after_eval` 里**只要漏了 `iterations >= MAX_ITERS` 这条出口**，遇到一段永远不 PASS 的文本就**无限循环烧钱**——这正是第 12 章"不收敛"事故在 workflow 形态下的复刻。刹车不是可选项，是这个模式能不能上线的前提。
- **暗坑4（迭代计数位置）**：`iterations + 1` 要放在 `translate_node`（每次重译才算一轮），别放 evaluate；放错了刹车计数不准，要么提前停要么停不住。
- **暗坑5（state 设计）**：checkpoint 存的是整个 `TransState`，所以 state 里**只能放可序列化的纯数据**。把 LLM client、文件句柄、连接对象塞进 state 会让 checkpoint 序列化直接报错——state 是"流程数据快照"，不是"运行时对象垃圾桶"。
- **暗坑6（rubric 可判定性）**：evaluate 的 rubric 必须能产出**机器可判的"PASS/具体反馈"**，别让它回"还行吧/可以更好"这种模糊话——模糊反馈会让环抖动、判不出该不该停。这条和第 11 章 RAG 评测、LLM-as-judge 是同一套方法论。

**参考实现要点**（TODO 答案方向）：TODO 1 `return {"draft": draft, "iterations": state["iterations"] + 1}`；TODO 2 判断 `verdict.strip().upper().startswith("PASS")` 决定 `done` 与 `feedback`；TODO 3 先判 `done` 再判 `iterations >= MAX_ITERS` 都返回 `END`，否则 `"translate"`；TODO 4 `builder.add_conditional_edges("evaluate", route_after_eval)`。
