# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 夏天的爱人 — xiatiandeairen.github.io

个人内容站，报纸/杂志风格，Astro + Tailwind CSS 构建，部署于 GitHub Pages。

## Commands

```bash
npm run dev              # Astro dev server，默认 http://localhost:4321
npm run build            # 构建到 dist/（同时跑 zod 校验和搜索索引生成）
npm run preview          # 本地预览构建产物
npm run lint:unused      # 扫描死 CSS class + 未引用组件（CI 强制）
npm run design:view <folder>  # 临时挂载 design-archive/<folder> 到 dev server
```

无 test 框架。质量保证靠：build 时的 zod schema 校验 + `lint:unused` 死代码扫描 + GitHub Actions（`.github/workflows/lint.yml` push/PR 触发）。

调试单个页面：`curl -s http://localhost:4321/notes/<slug>` 配 grep / python 解析；本项目无 e2e 测试基建。

## Architecture

**框架**：Astro 4.x（static output）+ Tailwind + TypeScript。零客户端 framework，交互靠 `<script is:inline>` 原生 JS。

**内容是 Astro Content Collection**（`src/content/notes/*.md`），构建时被 `src/utils/schema.ts` 的 zod schema 校验（slug 唯一、objectivity 比例和=1.0、ISO 日期等）。Frontmatter 全字段定义在 README §Frontmatter Schema。

**双语 i18n**（`src/i18n/{zh,en}.ts`）：默认中文在根路径，英文在 `/en/*`。每个英文页都在 `src/pages/en/` 镜像一份；公用业务逻辑放在 `src/utils/`，组件通过 `lang` prop 切换文案。

**渲染管线**（文章详情页 `src/pages/notes/[slug].astro`）：
1. `getStaticPaths()` 读 `src/content/notes/`，每篇生成 props（包括 toc / related / series / prev/next）
2. Markdown 经 `marked` 渲染 → `addHeadingIds()` 后处理给 h2/h3 注入 id（默认 marked 不加，TOC 锚跳依赖此）
3. `extractToc()` 和 `addHeadingIds()` 用同一个 `slugify` 算法保证 id 一致

**搜索索引**：构建时 `src/integrations/generate-search-index.ts` 生成 `public/search-index.json`（v1.1 含 `series/tagAliases/topicAliases/tagCounts/topicCounts`）；运行时 `src/utils/search.ts::searchNotes(q, index)` 解析字段前缀 `tag:/topic:/type:/series:`，分层打分（title-exact 100 > title-partial 60 > tag/topic-exact 55 > partial 35 > series 30 > alias 25 > content 10），同分 date desc。overlay（BaseLayout）+ `/search` + `/en/search` 三处 inline JS 镜像该逻辑，无服务端。

**文章模式系统**：`<html data-mode="default|reading|book">` + 正交能力属性 `data-chrome / data-typography-controls / data-reading-progress / data-article-layout`。模式只是预设组合，CSS 只读能力属性不读模式名。新增模式只需在 `src/pages/notes/[slug].astro` 的 `MODE_PRESETS` 里加一行 + 相应能力属性的 CSS 规则；不用改 mode-named 选择器。当前有单个浮层按钮 `mode-toggle`（左键循环、右键/长按弹 menu）。

**CSS 分层**（详见 [UI Architecture §2.1](.know/docs/arch/ui-architecture.md#21-style-层细节css-5-层依赖)）：
```
tokens.css → base/layout/masthead/dropdown/search → components.css → pages/<style> → global.css（仅跨切面规则）
```
`src/styles/` 只放 CSS；设计文档族在 `.know/docs/arch/design-*.md` + `ui-architecture.md`，`src/styles/README.md` 是 stub 指路。

## 架构规范（when → action）

所有规则都可查 [Design Governance](.know/docs/arch/design-governance.md)。**当你遇到这些场景时：**

| 场景 | 动作 |
|------|------|
| **入门 / 不熟悉架构** | 读 [UI Architecture](.know/docs/arch/ui-architecture.md)（CSS 5 层 + Component 4 类 + Page 组成） |
| **写新样式** | 默认写在页面 `<style>`；第 3 次被复用才提升到 `components.css` 并登记 [§2.3 Primitive 表](.know/docs/arch/design-governance.md#23-primitive-登记表)。禁止防御性抽象 |
| **改共享层文件**（`tokens.css` / `base.css` / `layout.css` / `components.css` / `global.css` / `components/*.astro`） | 先 `grep -rlE "\b<target>\b" src/` 预判影响面。页面内联样式改动免 grep |
| **写组件查变量** | 读 [Design Tokens](.know/docs/arch/design-tokens.md) |
| **实现新组件找配方** | 读 [Design Patterns](.know/docs/arch/design-patterns.md)，组件头注释列出复用的 patterns + tokens |
| **新建页面** | 同时建 `src/pages/` 中文版 + `src/pages/en/` 英文版；文案通过 `src/i18n/*.ts` 走 |
| **新增 / 删除 components.css 共享 class** | 同步更新 [§2.3 Primitive 登记表](.know/docs/arch/design-governance.md#23-primitive-登记表)；不在表 = 待删 |
| **引入 `global.css` 规则** | 只允许跨切面激活类（`.dark` / `.reading-mode`），不是"公共样式" |
| **探索新设计方向** | 放到 `design-archive/<YYYY-MM-DD>-<topic>/`；用 `./scripts/view-design.sh <folder>` 预览；归档见 [Design Archive](.know/docs/arch/design-archive.md) |
| **提交前 / 清理后** | `npm run build && npm run lint:unused`。死代码要么删，要么登记登记表 |
| **修改首页 hero 布局**（`src/pages/index.astro` 的 `.layout-featured`） | 首屏不变量：hero + sidebar + secondary headline 必须完整落在首屏 viewport。sidebar 想加内容 → 放进 `.layout-featured-side-filler`（`margin-top: auto` 吸底）；不得新增第 3 grid row；不得给 sidebar 加 >3 篇文章或大图使其变 tall |

## 首页首屏不变量

`.layout-featured` 是刚性的 2 行 grid：

```
row 1: [ hero (main)  | divider | sidebar ]
row 2: [         secondary headline (2 cols)        ]
```

**不变量**：桌面默认视窗下，row 1 + row 2 必须一屏看全（secondary 不能滚出屏幕）。

**扩展规则**
- 要在右侧加内容（如热门标签、订阅 CTA）→ 放 `.layout-featured-side-filler`；该块 `margin-top: auto` 会把自己推到 sidebar 单元格的底部，填充 3 篇文章下方的空白，**不增加 row 高度**
- 想加新"报纸头条级"内容块 → 放 `.suggestions-section` 或 `.remaining-section` 下面（first fold 之外）
- 不得将 sidebar 文章数量 >3；不得在 sidebar 加图片/大块文本
- 不得 emit 新的 `grid-column: 1 / -1` 子元素（会成为 row 3 把 secondary 挤出）

参考实现：`src/styles/layout.css::.layout-featured` 注释 + `src/pages/index.astro::layout-featured` 注释。

## Commit 约定

全局 commit-msg hook 强制 `^(feat|fix|chore|refactor|docs|test|build|ci|revert|subtree)(\([\w._-]+\))?: subject`。scope 不允许 `+`、空格等字符。多行 message 中 subject 与 body 之间留空行。

## Know

### 文档索引

#### 项目级

- [夏天的爱人 产品路线图](.know/docs/roadmap.md) | v1-v3, M1-M10
- [Design Tokens 架构](.know/docs/arch/design-tokens.md) | 5 类原子变量 + 12 语义颜色 + 3 质量分级色 + 8 字号 + 9 间距，消费方禁硬编码
- [Design Patterns 架构](.know/docs/arch/design-patterns.md) | 7 个 Patterns + 8 个 Component 状态规范 + 5 条 Don'ts
- [UI Architecture 架构](.know/docs/arch/ui-architecture.md) | CSS 5 层 + Component 4 类 + Page 组成（路由/i18n/Layout/渲染管线/数据流）
- [Design Governance 架构](.know/docs/arch/design-governance.md) | 4 治理机制：Rule of Three / 变更前 grep / Primitive 登记表 / lint:unused CI
- [Design Archive 架构](.know/docs/arch/design-archive.md) | 候选方向 mockup 归档 + view-design.sh 流程 + 生产构建隔离不变量

#### UI

- [页面区域术语表](.know/docs/ui/page-terminology.md) | 术语 + ASCII 结构图
- [Footer 交互设计](.know/docs/ui/footer.md) | 布局 + 状态
- [首页布局 交互设计](.know/docs/ui/homepage-layout.md) | 5 区域 + 4 卡片变体
- [Masthead 交互设计](.know/docs/ui/masthead.md) | 导航 + 工具栏 + dropdown
- [Search Overlay 交互设计](.know/docs/ui/search-overlay.md) | 弹层 + 实时搜索
- [Sidebar Drawer 交互设计](.know/docs/ui/sidebar-drawer.md) | hover 抽屉

#### Requirements

- [M1 基础可用性](.know/docs/requirements/m1-basic-usability/prd.md) | ✅ 响应式 + 暗色模式 + 关于页
- [M2 阅读体验提升](.know/docs/requirements/m2-reading-experience/prd.md) | ✅ TOC + 系列 + 推荐 + 代码块
