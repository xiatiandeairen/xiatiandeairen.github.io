---
name: article
description: 把 AI 对话中的碎片知识转为可发布到个人站的高质量文章。流程：capture 收集素材 → draft 依 blueprint 扩写 → gate 注入发布前 checklist → publish 落到站点 notes/。v1 仅服务 AI skill 经验类文章。
---

# article

`/article {capture|draft|gate|publish}` — 半自动文章生产管道。

## Pipeline

```
capture  → inbox/{ts}-{slug}.md            # 保存对话片段 + 一句话观点
draft    → drafts/{slug}.md                # AI 按 blueprint 扩写正文 + 填齐 frontmatter
gate     → drafts/{slug}.md (追加 checklist) # 人工逐条勾选，通过后改 gate_passed: true
publish  → $ARTICLE_SITE_REPO/src/content/notes/{N}-{slug}.md  # cp + schema 预检
```

## Hard Rules

- 不改 `src/utils/schema.ts`（以其为约束适配，不反向修改）
- 不改 `.github/workflows/*.yml`
- draft 阶段不得留 `pending` / `0` / `""`（空字符串）在必填字段，否则 publish 必然被 schema 预检拒
- 所有具体例子（命令、代码、数字、引用）必须来自 inbox 素材或用户观点；素材缺失时写 `<!-- TODO: 补例子 -->`，不得编造
- publish 后**不自动** `git add/commit/push`，由用户手动执行（MVP-A 阶段约束）

## Sub-commands

### capture `<slug> <tag> <opinion>`

**做**：把当前对话里相关的片段（原话 / 代码 / 截图描述）写入 `inbox/{ts}-{slug}.md` 正文区。

**条件**：
- `slug` 限小写字母、数字、连字符；匹配 `^[a-z0-9-]+$`
- `tag` 单个词；关联后续 draft 筛选素材
- `opinion` 一句话观点，15-80 字；表达"想主张什么"

**落地**：调 `scripts/article-ctl.sh capture <slug> <tag> <opinion>` 创建文件框架；AI 在正文区填充 3-10 条素材条目，每条含 source（哪段对话）+ excerpt（原文片段）+ 1 行意义说明。

**few-shot**：

Good：
```
/article capture github-workflow-tips github "靠 CI 兜底比靠纪律兜底省力"
```

Bad（不满足条件）：
```
/article capture "GitHub Tips" general "我想讲讲 github"
  ↑ slug 含空格 / tag 太泛 / opinion 是意图不是主张
```

### draft `<slug> [--template {skill-experience|tutorial-intro}]`

**做**：按对应 blueprint 五节骨架扩写成完整草稿；frontmatter 全字段填实值。

**Template 选择**：

| Template | 何时选 | Blueprint 文件 | 骨架 |
|----------|--------|---------------|------|
| `skill-experience`（默认）| 讲"我遇到 X 踩了 Y 坑最后靠 Z 解决" | `blueprint.md` | 起源 / 尝试 / 踩坑 / 设计 / 反共识 |
| `tutorial-intro` | 介绍一个工具或能力集合，目标是扩读者工具箱 | `blueprint-tutorial-intro.md` | 这是什么 / 何时用 / 核心能力 / 典型用法 / 进阶建议 |

选错 template 会得到别扭的文章（例："介绍 git 高级用法"用 skill-experience → 起源节变成硬凑的"我学 git 遇到 Y 问题"）。主题是"我的 X 经验"用 experience，主题是"教你用 Y 工具"用 tutorial-intro。

**步骤**（AI 执行）：
1. 读对应 blueprint 文件拿骨架和扩写 prompt
2. 扫 `inbox/*.md`，匹配 slug / tag 相关文件，提取素材条目
3. 按五节顺序填正文：起源 / 尝试 / 踩坑 / 最终设计 / 反共识 takeaway
4. 用户观点落到"反共识 takeaway"节并展开
5. 填齐 frontmatter（下表必填枚举，**不得留 pending / 0 / 空字符串**）
6. 正文长度 1500-3000 中文字符；每节 200-700 字

**frontmatter 必填枚举**（对齐 `src/utils/schema.ts::NoteFrontmatterSchema`）：

| 字段 | 类型 | 合法值示例 | 常见错误 |
|------|------|-----------|---------|
| `title` | string, ≥1 字符 | `"AI 项目里我靠 GitHub 工作流兜底的三件事"` | 空字符串 |
| `slug` | string | 由参数传入，不改 | — |
| `createdAt` / `updatedAt` / `date` | ISO8601 `Z` 结尾 | `2026-04-20T08:00:00Z` | 非 UTC / 缺 Z |
| `question.type` | string | `"方法论"` / `"系统设计"` / `"工程实践"` / `"AI 工程化"` | 空字符串 |
| `question.subType` | string optional | `"CI/CD"` / 省略 | — |
| `quality.overall/coverage/depth/specificity` | 0-10 number | `7` / `8` / `8` / `7`（实话实估）| 留 0（会被 publish 拒）|
| `quality.reviewer` | `"ai" \| "human" \| "hybrid"` | AI 扩写 → `"ai"`；人工改过 → `"hybrid"` | 其他值 |
| `analysis.objectivity.{factRatio,inferenceRatio,opinionRatio}` | number，和必须 = 1.0（允差 0.01）| `0.6 / 0.25 / 0.15` | 和 ≠ 1.0 |
| `analysis.assumptions` | string[] | `["读者用过 git"]` / `[]` | — |
| `analysis.limitations` | string[] | `["仅覆盖 GitHub Actions，未涉及 GitLab"]` / `[]` | — |
| `review.status` | `"draft" \| "reviewed" \| "deprecated"` | draft 阶段固定 `"draft"` | 其他值 |
| `tags` | `Array<{name, parent?, alias?}>` ≥1 | `[{name: "github"}, {name: "ci", parent: "engineering"}]` | 空数组 / 字符串数组 |
| `topics` | `Array<{name, alias?}>` ≥1 | `[{name: "AI 工程化"}, {name: "开发工具"}]` | 空数组 |
| `gate_passed` | bool | draft 生成时固定 `false` | — |

**few-shot**：

Good（节选正文）：
```
## 起源: 遇到的真实问题

我维护一个个人内容站 `xiatiandeairen.github.io`，每次手写 frontmatter
都要手动核对 20+ 字段对不对——漏一个字段、比例和不是 1.0、tag 写成
字符串而不是对象，都会让 build 炸。
```

Bad：
```
## 起源: 遇到的真实问题

在当今快速发展的 AI 领域，我遇到了一些挑战。让我们深入探讨...
  ↑ AI 腔 / 无具体场景 / 无数字
```

**落地**：调 `scripts/article-ctl.sh draft <slug>` 生成文件骨架（frontmatter 已含默认占位），AI 用 Edit 工具改每个字段为实值并填正文。

### gate `<slug>`

**做**：把 `quality-checklist.md` 追加到 `drafts/{slug}.md` 末尾；人工勾选通过后，手改 frontmatter 的 `gate_passed: false` → `true`。

**步骤**：
1. 调 `scripts/article-ctl.sh gate <slug>` 追加 checklist（幂等：已追加则报错 exit 0）
2. AI 逐条自评展示（"这条文章我觉得勾 / 不勾，理由：…"），用户最终决定
3. 全勾 → 用户或 AI 用 Edit 改 `gate_passed: true`
4. 未全勾 → 回 draft 修补

### publish `<slug>`

**做**：cp draft 到站点 `src/content/notes/{N}-{slug}.md`，打印手动 push 命令块。

**前置条件**：
- `drafts/{slug}.md` 存在且 `gate_passed: true`
- schema 预检通过：title 非空 / question.type 非空 / quality 四项非 0 / 无残留 pending
- `$ARTICLE_SITE_REPO` 或 `config.json` 指向站点 repo（脚本会 git 自动探测）

**步骤**：
1. 调 `scripts/article-ctl.sh publish <slug>`
2. 脚本扫 notes/ 现有最大编号 +1 作为新文件前缀
3. 脚本打印完整 `cd / git add / git commit / git push` 命令块
4. 用户**先**跑 `npm run build`，通过后再跑 git 命令组
5. 失败 → 回 draft 修补，用 `git rm` 删 notes/ 新文件

## Files

- `blueprint.md` — v1 唯一骨架 + AI 扩写 prompt
- `quality-checklist.md` — 发布前 4 条硬标准
- `scripts/article-ctl.sh` — bash 总控（capture / draft / gate / publish）
- `config.json` — 站点路径（gitignored；默认空 → 走 git 自动探测）
- `inbox/` — 素材（gitignored）
- `drafts/` — 草稿（进 git）

## Constraints

- 不调 Anthropic API，AI 扩写 = 当前 Claude 主会话生成
- 站点路径不硬编码：`$ARTICLE_SITE_REPO` env var > `config.json` `site_repo` > `git rev-parse --show-toplevel`
- blueprint v1 作用域："AI skill / 工具 / 工程实践经验类"；脱离此范围的主题需先扩 blueprint

## Roadmap

当前：**MVP-A**（本文档描述的流程）— 半自动，publish 后人工 commit+push。

下一阶段演进（非本次实现，仅预留）：

| 阶段 | 能力 | 触发 | 前置依赖 |
|------|------|------|---------|
| B | AI 自评 gate 替代人工勾选；若全勾则自动改 gate_passed | 手动 `/article publish` | MVP-A 稳定 2 周 + 3 篇真实文章发布 |
| C1 | `/article publish --auto-push`：跑 build → 通过则自动 add/commit/push | 手动命令带 flag | B + 写入 build 失败时的自动回滚（`git rm` 新 note）|
| C2 | SessionEnd hook 扫描 transcript 自动挖素材入 inbox/（仅打标，不写 draft）| Claude Code hook | C1 + 素材相关度评分算法（tag/topic 匹配 + 片段长度门槛）|
| C3 | cron / scheduled agent 扫 inbox 自动 draft → gate → publish | 定时触发 | C2 + 质量门控成熟（B 阶段通过率 ≥80%）|

C3 前必须有一个"dry-run only 模式"（生成 draft 但不 publish，等人工验收 N 次后才解锁自动 publish）。

禁止跳阶段（例如跳过 B 直接做 C2），**因为**：每阶段都在确认一个前提假设（B 确认"AI 自评可靠"，C1 确认"push 安全"，C2 确认"素材筛选准"）。
