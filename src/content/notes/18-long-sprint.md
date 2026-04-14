---
title: "Long Sprint 实践：让 AI 自主完成大型任务"
slug: "long-sprint-practice"
createdAt: "2026-04-14T22:00:00Z"
updatedAt: "2026-04-14T22:00:00Z"
date: "2026-04-14T22:00:00Z"
question:
  type: "系统设计"
  subType: "AI 工程化"
quality:
  overall: 9
  coverage: 9
  depth: 9
  specificity: 9
  reviewer: "human"
analysis:
  objectivity:
    factRatio: 0.6
    inferenceRatio: 0.25
    opinionRatio: 0.15
  assumptions:
    - "读者已了解 Sprint 的基本概念"
    - "有过让 AI 执行多步骤任务的经验"
  limitations:
    - "Long Sprint 是最复杂的 Skill，学习曲线较陡"
    - "自动编排的可靠性与任务拆分质量强相关"
review:
  status: "reviewed"
  reviewedAt: "2026-04-14T23:00:00Z"
tags:
  - name: "AI"
  - name: "agent"
    parent: "AI"
  - name: "engineering"
  - name: "sprint"
    parent: "agent"
topics:
  - name: "AI 工程化"
  - name: "开发工具"
---

# Long Sprint 实践：让 AI 自主完成大型任务

## 当一个 Sprint 放不下

Sprint 设计上是为**单一、聚焦的任务**优化的。添加一个功能、修复一个 bug、重构一个模块——这些任务的边界清晰，一条流水线走完就能交付。

但有些任务天然就很大：

- "把博客从 Next.js 迁移到 Astro"
- "给系统加上完整的标签层级结构"
- "重写搜索功能，从客户端搜索改成基于预生成索引"

这些任务的特点是：拆成一个 Sprint 太大（涉及多个模块、多种类型的变更），但又不能简单地拆成几个独立的 Sprint——子任务之间有依赖关系，前一个的产出影响后一个的输入。

直觉上的解法是"手动拆分 + 逐个执行"：你自己把大任务拆成 5 个小任务，然后一个个用 Sprint 跑。问题是拆分本身就需要大量思考——你需要理解全貌、识别依赖、确定顺序。做完这些，任务本身可能就完成一半了。

Long Sprint 的目标是**把拆分和编排也交给 AI，但不失控**。

## 三个阶段：准备、执行、收敛

Long Sprint 把大任务的生命周期分为三个阶段：

**Phase A — 准备（人工参与）。** 这是唯一需要人类深度参与的阶段。AI 和你一起完成：

1. 理解任务全貌和约束
2. 识别不可变的技术方向
3. 把大任务拆成有序的子 Sprint
4. 锁定方向（Direction Lock）

Phase A 结束后，你会得到一个子 Sprint 列表和一个 Direction Lock 文档。从这一刻起，技术方向不可变——后续的子 Sprint 只能在这个方向上推进，不能推翻。

**Phase B — 执行（自动运行）。** AI 按顺序执行每个子 Sprint，每个子 Sprint 走自己的流水线（plan → execute → quality，或更长）。

Phase B 不需要人类逐步确认（除非单个子 Sprint 内部触发了 user gate）。但有两个自动检查机制在后台运行：Direction Lock 校验和漂移检测。

**Phase C — 收敛。** 所有子 Sprint 完成后，统一审查整体结果、生成复盘报告。

这三个阶段的时间分配大约是 A:B:C = 20%:70%:10%。Phase A 虽然只占 20% 的时间，但决定了整个 Long Sprint 的质量——方向错了，后面 70% 全是浪费。

> **关键洞察：** Long Sprint 的核心矛盾是自动化程度和失控风险。Phase B 之所以能自动运行，完全依赖 Phase A 的质量。如果方向没对齐、子 Sprint 没拆好，自动执行只会让问题更快地累积。这就是为什么 Phase A 必须人类参与——不是因为 AI 做不好拆分，而是因为方向错误的代价太高了。

## Direction Lock：锁住方向

Direction Lock 是 Long Sprint 最重要的机制。

在 Phase A 结束时，你和 AI 确认一份方向文档，包含：

- 选定的技术方案（不可变）
- 排除的替代方案和理由（不可变）
- 约束条件（不可变）
- 成功标准（不可变）

从这一刻起，后续的所有子 Sprint 都受这份文档约束。

为什么需要锁？

假设你的大任务是"把博客迁移到 Astro"。Phase A 确定了方向：使用 Astro 的内容集合（Content Collections）管理文章，而不是手动读 Markdown 文件。

如果没有 Direction Lock，可能会发生这样的事：

子 Sprint 1 按计划使用内容集合，创建了 schema 定义和内容目录。子 Sprint 2 在实现搜索功能时发现内容集合的 API 不太方便，决定"绕过"内容集合直接读 Markdown 文件。子 Sprint 3 在实现标签页面时发现有两套数据源（内容集合 + 直接读文件），开始做兼容层。

方向漂移了。代码里有两套互相矛盾的数据访问方式，复杂度急剧上升，而且不是任何一个子 Sprint 的"错"——每个子 Sprint 在当时的上下文下做的决定看起来都是合理的。

Direction Lock 阻止这一切发生。子 Sprint 2 在试图绕过内容集合时，Direction Lock 校验会失败——"方向文档明确使用 Content Collections，当前变更引入了直接读取 Markdown 的代码"。AI 被迫在内容集合的框架内解决问题，而不是绕过它。

这是否会导致 AI 在"不合理的方向上死磕"？有这个风险。如果 Phase A 的方向本身就是错的，Direction Lock 会放大这个错误。这就是为什么 Phase A 需要人类深度参与——你需要确保锁住的方向是正确的。

> **常见误区：** "Direction Lock 太死板了，项目执行中发现更好的方案怎么办？"答案是：打断 Long Sprint，重新进入 Phase A。Direction Lock 的设计意图不是"永远不能改方向"，而是"改方向必须是一个显式的、有意识的决策，不能是子 Sprint 中悄悄发生的漂移"。偶尔需要重新开始 Phase A 是正常的，频繁需要说明初始方向分析不够充分。

## 漂移检测：发现范围蔓延

Direction Lock 锁住的是方向，但方向正确不代表范围不会膨胀。

一个常见场景：你的大任务是"实现标签层级结构"。Phase A 拆了 5 个子 Sprint。子 Sprint 3 在实现标签页面时，AI "顺便"加了一个标签搜索功能——这不在计划内，但"反正都改这个页面了"。子 Sprint 4 在实现面包屑导航时，AI "顺便"给所有页面加了一个返回顶部按钮。

每个"顺便"都很小，但累积起来：原本 5 个子 Sprint 的任务量膨胀成了 7-8 个的工作量，而且额外的功能没有经过 Phase A 的评审。

漂移检测在第 3 个子 Sprint 之后自动启动。它检查：

**范围蔓延：** 对比当前已完成的工作和 Phase A 的规划。新增了计划外的文件吗？修改了计划外的模块吗？如果偏差超过阈值（比如计划外的变更占总变更的 20% 以上），触发警告。

**进度轨迹：** 已完成 3 个子 Sprint，计划共 5 个。当前的完成度和预期一致吗？如果 3 个子 Sprint 完成后发现整体进度只有 30%（预期应该是 60%），说明子 Sprint 的拆分粒度有问题——每个子 Sprint 实际比预期复杂得多。

漂移检测的结果有三种：

- **绿色**：范围和进度符合预期，继续。
- **黄色**：有轻微偏差，记录在 Journal 中，继续但提高警惕。
- **红色**：显著偏差，暂停 Phase B，需要人类决定是调整计划还是回退。

## Journal：自动决策的黑匣子

Phase B 是自动运行的，这意味着 AI 会做很多决策——选择实现方式、处理边界情况、解决冲突。这些决策没有人类审批，但需要有记录。

Journal 是 Long Sprint 的决策日志。每个子 Sprint 完成后自动写入一条 Journal 条目：

```
子 Sprint #3: 实现标签页面路由
- 决策: 使用动态路由 [tag].astro 而非预生成所有路径
- 理由: 标签列表可能变化，动态路由避免每次加标签都要重新构建
- 结果: 路由正常工作，构建时间减少 40%
- Direction Lock 校验: 通过
- 偏差: 无
```

Journal 的价值在 Phase C 或出问题时体现。如果第 5 个子 Sprint 的结果不符合预期，你可以回溯 Journal 找到是哪个子 Sprint 的决策导致了偏差，而不是从头审查所有代码变更。

Journal 也是 Know 的输入源。Long Sprint 结束后，Journal 中那些有普遍价值的决策（比如"在 Astro 中优先用动态路由"）可以通过 `/know learn` 进入知识库。

> **关键洞察：** Long Sprint 的三个约束机制——Direction Lock、漂移检测、Journal——本质上是在解决同一个问题：自动化带来的不透明性。AI 自主执行时间越长，人类越容易失去对项目状态的把控。这三个机制用不同的方式维持透明度：Lock 确保方向可知，漂移检测确保范围可知，Journal 确保决策可知。

## 什么时候用 Long Sprint，什么时候不用

Long Sprint 是我的 Skill 系统中最复杂的一个，不是所有大任务都需要它。

**适合 Long Sprint 的：**

- 任务有明确的终态（"迁移完成"有清晰的定义）
- 子任务之间有顺序依赖（不能并行做）
- 技术方向可以在开始前确定（不需要边做边探索）
- 你愿意投入 Phase A 的时间做充分准备

**不适合 Long Sprint 的：**

- 探索性任务（"调研一下用什么方案"——方向不确定，没法锁 Lock）
- 高度并行的任务（5 个独立的功能模块——各自用普通 Sprint 就行）
- 需要频繁调整方向的任务（每两天就要改需求——Direction Lock 会频繁失效）
- Phase A 的投入不划算的任务（任务本身只要 2 小时，Phase A 可能就要 1 小时）

一个判断经验：**如果你能一口气写出所有子任务的列表和它们的依赖关系，用 Long Sprint。如果你写到一半发现"后面的取决于前面的结果"，考虑先做几个普通 Sprint 探路，再决定是否需要 Long Sprint。**

> **实践建议：** 第一次使用 Long Sprint，选一个你已经做过类似事情的任务——比如"把另一个项目的标签系统也改成层级结构"。你知道大概要做什么、会遇到什么坑、合理的方向是什么，这让 Phase A 的质量有保障。等你对 Direction Lock 和漂移检测的行为有了直觉之后，再用 Long Sprint 去做全新的大任务。
