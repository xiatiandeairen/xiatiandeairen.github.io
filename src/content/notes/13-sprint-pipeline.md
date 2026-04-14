---
title: "Sprint：给 AI 一条流水线而不是一句话"
slug: "sprint-ai-pipeline"
createdAt: "2026-04-14T12:00:00Z"
updatedAt: "2026-04-14T12:00:00Z"
date: "2026-04-14T12:00:00Z"
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
    - "读者使用过 AI 编程助手完成实际开发任务"
    - "经历过 AI 输出需要反复修正的情况"
  limitations:
    - "Sprint 系统仍在迭代中"
    - "效果数据基于个人项目，样本有限"
review:
  status: "reviewed"
  reviewedAt: "2026-04-14T13:00:00Z"
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

# Sprint：给 AI 一条流水线而不是一句话

## 一个任务，两种执行方式

假设你的任务是：**把博客的标签系统从扁平列表改成层级结构**。标签之间有父子关系，URL 路由要能解析层级路径，页面要展示面包屑导航。

**没有 Sprint 的执行过程：**

你告诉 AI："把标签改成层级结构，支持父子关系。"

AI 立刻开始写代码。它先改了标签的数据结构，在 frontmatter 里加了 `parent` 字段。然后修改了标签页面的路由。然后发现需要一个面包屑组件，又写了一个。然后发现工具函数里有个 `getTagCount` 没有处理子标签的计数，改了。

40 分钟后，AI 告诉你"完成了"。你开始检查：

- 标签页面能显示子标签了 ✓
- 但首页的标签过滤器坏了——它依赖旧的扁平结构，AI 没改 ✗
- 搜索索引里的标签数据也没更新 ✗
- 有两个标签名冲突了（`css` 同时是根标签和 `frontend` 的子标签），没有处理 ✗
- 面包屑组件在标签深度超过 2 层时死循环了 ✗

你花了两小时修这些问题。其中一半时间在定位"AI 还改了哪些文件"。

**有 Sprint 的执行过程：**

同样的任务进入 Sprint。

Sprint 先问 3 个问题：需求清晰吗？——不完全，层级深度、冲突处理策略都没说清。需要技术设计吗？——是，涉及数据模型变更、多个页面联动、工具函数适配。高风险吗？——是，修改核心数据结构，影响全站标签功能。

裁剪出完整流水线：`brainstorm → design → plan → execute → quality → review → insight`。

**Brainstorm** 阶段和你对齐了 3 个关键决策：最大层级深度限制为 2、同名标签通过别名机制解决、父标签页面自动聚合子标签下的文章。

**Design** 阶段确定了技术方案：在 Tag 接口中添加 `parent` 和 `alias` 可选字段，新增 `tags.ts` 工具模块统一处理层级逻辑，修改 `[tag].astro` 路由支持层级查询。

**Plan** 阶段拆出 5 个任务，并定义了 Anchor：

```
MUST_EXIST: src/utils/tags.ts
MUST_IMPORT: src/pages/tags/[tag].astro -> @utils/tags
FILE_NOT_MODIFIED: src/pages/index.astro
FILE_NOT_MODIFIED: src/utils/schema.ts
MUST_BUILD
```

`FILE_NOT_MODIFIED: src/pages/index.astro`——首页不应该被修改。`FILE_NOT_MODIFIED: src/utils/schema.ts`——验证 schema 不能被改，标签验证逻辑应该加在新模块里。

**Execute** 逐个完成任务。**Quality** 运行 Anchor 检查，发现 AI 确实动了 `schema.ts`（想在里面加标签层级校验），被拦截，改为在 `tags.ts` 里实现。**Review** 确认所有变更范围合理。

整个过程多花了约 5 分钟在前期对齐上，但没有后期返工。

> **关键洞察：** Sprint 的价值不是让 AI 执行变慢了，而是把"返工时间"前移到了"规划时间"。5 分钟的 brainstorm + design 替代了 2 小时的 debug + 修复。

## 3 个问题决定流水线形状

Sprint 的核心机制不是 7 个阶段——而是**裁剪**。

面对一个任务，Sprint 问 3 个问题：

| 问题 | 是 | 否 |
|------|----|----|
| 需求是否需要澄清？ | → 加入 brainstorm | → 跳过 |
| 是否需要技术设计？ | → 加入 design | → 跳过 |
| 是否涉及高风险？ | → 加入 review | → 只做 quality |

`plan`、`execute`、`quality`、`insight` 四个阶段始终存在。

这意味着一个简单的 bug 修复——需求清晰、方案唯一、风险低——只走 4 个阶段：`plan → execute → quality → insight`。没有多余的 brainstorm、design、review。

而一个跨模块重构——需求有歧义、方案有多种、影响面大——走全部 7 个阶段。

流水线的长度不是由人决定的，是由任务性质决定的。这消除了两种常见错误：**对简单任务过度设计**（"不就是改个按钮颜色吗，为什么要 design 阶段"）和**对复杂任务准备不足**（"直接开始写就行了，边写边想"）。

三个问题的回答也不完全依赖 AI 的判断。Sprint 内置了关键词检测：任务描述里出现 `delete`、`migrate`、`payment`、`production`、`permission` 时，自动将"高风险"设为 yes，不管 AI 怎么评估。

> **常见误区：** 有人认为"好的流水线就是阶段多的流水线"。错。阶段多意味着开销大、反馈慢。Sprint 的设计目标是**最小充分流水线**——包含所有必要阶段，不包含任何多余阶段。一个 5 分钟能完成的 bug 修复走 7 个阶段是浪费。

## 7 个阶段，每个只做一件事

Sprint 的阶段设计有一个硬性原则：**每个阶段只读上游 handoff，只写自己的 handoff**。不存在某个阶段"回头看两步之前的输出"。

这是刻意的约束。它确保了阶段之间的依赖是单向的、可追溯的。

**Brainstorm：锁定需求。** 输入是用户的原始描述，输出是经过对齐的需求文档——包含目标、成功标准、明确排除项。关键是"排除项"：不做什么和做什么一样重要。没有排除项的需求会不断膨胀。

**Design：确定方案。** 输入是 brainstorm 的 handoff，输出是技术方案——涉及哪些模块、数据如何流转、接口怎么定义。如果有多个可行方案，列出 trade-off 让用户选。这里有一个 `[STOP:choose]` 门——方案选择权在人，不在 AI。

**Plan：拆分任务 + 定义 Anchor。** 输入是 design 的 handoff，输出是有序的任务列表和结构断言。每个任务独立可验证。Plan 不写代码，只定义"做什么"和"做完后怎么验证"。

**Execute：写代码。** 逐个完成 plan 中的任务。每完成一个任务，标记完成状态。如果执行中发现 plan 遗漏了什么，不能自己加——回到 plan 阶段补充（这种情况极少，通常说明 design 阶段有疏漏）。

**Quality：验证。** 运行 Anchor 检查 + 构建/测试。两者独立——测试通过不代表 Anchor 通过，反之亦然。如果 Anchor 失败，AI 在这个阶段内修复并重新检查，不需要回到 execute。

**Review：审查。** 只在高风险任务中启用。检查变更范围是否合理、是否引入了安全问题、是否有未预期的副作用。Review 不重复 quality 的工作——不再跑测试或检查 Anchor，只做人工判断层面的审查。

**Insight：复盘。** 记录本次 Sprint 的经验——什么出了预期、什么值得下次复用。Insight 不是敷衍的"总结"，而是为 Know 系统提供输入。如果没有值得记录的经验，输出"无"就好，不强制产出。

> **关键洞察：** Handoff 机制的核心价值不是"文档"，而是**信息压缩**。每个阶段把上游的大量信息压缩成下游需要的最小集合。Design 不需要知道 brainstorm 讨论了几轮才锁定需求，它只需要最终的需求文档。这让每个阶段都能专注于自己的职责，不被上游的细节淹没。

## Anchor：比测试更底层的验证

测试验证的是**行为**——给定输入 A，是否输出 B。

Anchor 验证的是**约束**——这个文件是否存在、那个模块是否没被修改、这个导入是否在。

两者是互补的，不是替代的。

一个真实的场景：AI 重构标签系统时，需要修改 `tags.ts` 和 `[tag].astro`。Plan 阶段定义了 Anchor：

```
MUST_EXIST: src/utils/tags.ts
MUST_IMPORT: src/pages/tags/[tag].astro -> @utils/tags
FILE_NOT_MODIFIED: src/layouts/BaseLayout.astro
FILE_NOT_MODIFIED: src/components/Pagination.astro
MUST_BUILD
MUST_TEST
```

`MUST_BUILD` 和 `MUST_TEST` 是行为验证——项目能编译、测试能通过。

其他四行是结构验证——确保 AI 只改了该改的文件。`FILE_NOT_MODIFIED` 尤其关键：它防止 AI 在执行时"顺手"改了不在计划内的文件。这正是开头那个搜索模块白屏事故的根因——AI 顺手改了一个工具函数的签名，测试没覆盖到但 Anchor 能拦住。

Anchor 的类型：

| Anchor | 检查内容 | 典型用途 |
|--------|---------|---------|
| `MUST_EXIST` | 文件/目录必须存在 | 确保新文件被创建 |
| `MUST_NOT_EXIST` | 文件/目录不能存在 | 确保旧文件被清理 |
| `MUST_IMPORT` | A 必须导入 B | 确保模块依赖正确 |
| `MUST_NOT_IMPORT` | A 不能导入 B | 防止循环依赖或违禁依赖 |
| `FILE_NOT_MODIFIED` | 文件不能有变更 | 限制修改范围 |
| `MUST_BUILD` | 构建必须通过 | 基础健全性 |
| `MUST_TEST` | 测试必须通过 | 行为正确性 |

Anchor 由 Plan 阶段产出，由 Quality 阶段执行。Execute 阶段不跑 Anchor——它只管写代码。这个分离是有意的：让"做事的人"和"检查的人"不是同一个阶段，避免自己检查自己。

> **实践建议：** 写 Anchor 时，`FILE_NOT_MODIFIED` 是投入产出比最高的。AI 最常犯的错误不是"没做到该做的"，而是"做了不该做的"——修改了计划外的文件。每个 Sprint 至少为核心文件（布局、配置、公共工具函数）加上 `FILE_NOT_MODIFIED`。

## Long Sprint：大任务的自动编排

有些任务太大，一个 Sprint 放不下——比如"把整个博客从 Next.js 迁移到 Astro"或"给系统加上完整的权限模型"。

Long Sprint 把大任务拆成多个有序的子 Sprint，自动编排执行。

Long Sprint 分三个阶段：

**Phase A（人工参与）：** 理解全貌、拆分子任务、锁定技术方向。这个阶段和普通 Sprint 的 brainstorm + design 类似，但产出的不是一个执行计划，而是多个子 Sprint 的编排方案。

Phase A 结束时会设立一个 **Direction Lock**：技术方向锁定，后续子 Sprint 不能推翻。比如确定了"用 Astro 的内容集合而不是手动读 Markdown 文件"，这个决策在后续所有子 Sprint 中不可变。

Direction Lock 的意义在于防止**方向漂移**。大任务执行周期长，没有锁的话，第 5 个子 Sprint 可能因为遇到了困难而试图推翻第 1 个子 Sprint 的架构决策，导致前面的工作全部作废。

**Phase B（自动执行）：** 按顺序执行子 Sprint。每个子 Sprint 是一个完整的 Sprint 流水线——有自己的 plan、execute、quality。

子 Sprint 之间有两个自动检查：

1. **Direction Lock 校验**：确认当前子 Sprint 没有偏离锁定的技术方向。
2. **漂移检测**：在第 3 个子 Sprint 之后启动，检查整体进度是否偏离初始规划、是否出现范围蔓延。

每个子 Sprint 完成后都会写一条 Journal 日志——记录做了什么决策、基于什么理由、结果如何。Journal 是 Long Sprint 的"黑匣子"，出了问题可以回溯。

**Phase C（总结）：** 所有子 Sprint 完成后的统一审查和复盘。

> **关键洞察：** Long Sprint 的核心不是"自动执行多个任务"——那只是机械的串联。核心是 **Direction Lock + 漂移检测** 这对约束机制。自动化程度越高，失控的代价越大。Direction Lock 保证方向不变，漂移检测保证范围不涨。没有这两个机制，Long Sprint 只是一个"更快地走向错误方向"的工具。

## 你的第一个 Sprint

如果你想体验 Sprint，不需要搭建完整的 7 阶段流水线。最小版本只需要 3 样东西：

**1. 一个 plan 模板。** 在 AI 开始写代码之前，要求它先输出：
- 任务列表（有序、每个独立可验证）
- 涉及的文件列表
- 不应被修改的文件列表

**2. Anchor 检查。** 至少写两种：
- `FILE_NOT_MODIFIED`：列出不应被修改的核心文件
- `MUST_BUILD`：修改完成后项目必须能编译

**3. 一个 quality 步骤。** 在 AI 说"完成了"之后，运行 Anchor 检查。失败了就让 AI 修复，不要自己手动改。

这三样东西加起来不超过 10 行配置。但它们解决了最常见的两个问题：AI 改了不该改的文件（`FILE_NOT_MODIFIED` 拦截）、AI 说完成了但其实编译不过（`MUST_BUILD` 拦截）。

用这个最小版本跑两周。如果你发现 AI 经常需求理解偏差——加 brainstorm。如果你发现 AI 的技术方案经常不合理——加 design。如果你在做高风险操作——加 review。

Sprint 的完整版本是迭代出来的，不是一次设计出来的。从你最痛的点开始，逐步扩展。
