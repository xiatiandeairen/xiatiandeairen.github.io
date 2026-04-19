---
name: article
description: 把 AI 对话中的碎片知识转为可发布到个人站的高质量文章。流程：capture 收集素材 → draft 依 blueprint 扩写 → gate 注入发布前 checklist → publish 落到站点 notes/。v1 仅服务 AI skill 经验类文章。
---

# article

`/article {capture|draft|gate|publish}` — 半自动文章生产管道。

## Pipeline

```
capture  → inbox/{ts}-{slug}.md      # 即时保存对话片段 + 一句话观点
draft    → drafts/{slug}.md          # 用户给方向+观点 → AI 依 blueprint 扩写
gate     → drafts/{slug}.md (+ checklist appended)
publish  → $ARTICLE_SITE_REPO/src/content/notes/{N}-{slug}.md
```

## Sub-commands

### capture

收集当前对话中的素材片段。

- 用法：`/article capture "{tag}" "{一句话观点}"`
- AI 行为：从当前会话提取相关片段（用户指点或 AI 自选），写入 `inbox/{YYYYMMDD-HHMMSS}-{slug}.md`
- 文件结构：frontmatter（ts/tag/source/opinion）+ 原始片段
- 落地：调 `scripts/article-ctl.sh capture <slug> <tag> <opinion>` 创建文件框架，AI 填充内容

### draft

从 inbox 素材生成草稿。

- 用法：`/article draft "{topic}" "{core opinion}"`
- AI 行为：
  1. 读 `blueprint.md` 拿骨架 + 扩写 prompt
  2. 按 topic/opinion 关键词筛 `inbox/*.md` 相关素材
  3. 依骨架五节（起源/尝试/踩坑/最终设计/反共识）填充，**不编造，只用素材+用户观点**
  4. 写入 `drafts/{slug}.md`，frontmatter 中 quality/analysis 字段填 `pending`
- 落地：调 `scripts/article-ctl.sh draft <slug>` 创建框架，AI 填正文

### gate

在草稿末尾追加发布前 checklist。

- 用法：`/article gate {slug}`
- 行为：调 `scripts/article-ctl.sh gate <slug>`，把 `quality-checklist.md` 内容追加到 `drafts/{slug}.md` 末尾
- 用户：逐条勾选；通过后在 frontmatter 把 `gate_passed` 改为 `true`

### publish

发布到站点仓库。

- 用法：`/article publish {slug}`
- 前置：`drafts/{slug}.md` 中 `gate_passed: true` 且通过 schema 预检
- 行为：调 `scripts/article-ctl.sh publish <slug>`：
  1. 读 `$ARTICLE_SITE_REPO`（env var 优先）或 `config.json` 中 `site_repo`
  2. Schema 预检（拒绝空 title / 空 question.type / quality 全 0 / 残留 `pending`）——防止 build 期 zod 报错
  3. 自动生成编号（扫描 notes/ 现有最大编号 +1）
  4. cp 到 `<site_repo>/src/content/notes/{N}-{slug}.md`
  5. **不**自动 commit/push；提示用户手动 `cd <site_repo> && git add . && git commit`

## Site schema 对齐

`cmd_draft` 生成的 frontmatter 必须覆盖 `src/utils/schema.ts::NoteFrontmatterSchema` 全部必填字段：

| 字段 | 类型 | draft 模板默认 | publish 前须填 |
|------|------|---------------|--------------|
| `title` | string | `""` | ✓ |
| `slug` | string | arg | — |
| `createdAt/updatedAt/date` | ISO8601 | 自动 | — |
| `question.type` | string | `""` | ✓ |
| `quality.{overall,coverage,depth,specificity}` | 0-10 number | 0 | ✓ |
| `quality.reviewer` | `ai\|human\|hybrid` | `ai` | 按实际改 |
| `analysis.objectivity.{factRatio,inferenceRatio,opinionRatio}` | number，和=1.0 | 0.5/0.3/0.2 | 按实际改 |
| `analysis.{assumptions,limitations}` | string[] | `[]` | 可保持空 |
| `review.status` | `draft\|reviewed\|deprecated` | `draft` | 发布前改 `reviewed` |
| `tags` | `[{name, parent?, alias?}]` | `[]` | ✓ 至少 1 条 |
| `topics` | `[{name, alias?}]` | `[]` | ✓ 至少 1 条 |

Skill workflow 专属字段（`template / gate_passed / quality_gate`）用 `.passthrough()` 透传，不会挂 schema；作者可选择在 publish 时清理。

## Files

- `blueprint.md` — v1 唯一骨架 + 扩写 prompt
- `quality-checklist.md` — 发布前 4 条 checklist
- `scripts/article-ctl.sh` — bash 总控
- `config.json` — 站点路径（gitignored）
- `inbox/` — 素材（gitignored）
- `drafts/` — 草稿（进 git）

## Constraints

- 不直接调 Anthropic API，AI 扩写 = 当前 Claude 主会话生成
- `xiatiandeairen.github.io/` 站点代码不动，只在 publish 时写 `src/content/notes/`
- 站点路径不硬编码：`$ARTICLE_SITE_REPO` env var 优先，否则读 `config.json`
