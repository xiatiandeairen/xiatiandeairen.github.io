# 夏天的爱人 — xiatiandeairen.github.io

个人内容站，报纸/杂志风格，Astro + Tailwind CSS 构建，部署于 GitHub Pages。

## Design System

修改 `src/styles/` 或新增组件时，**必须先读** [Design System](.know/docs/design-system.md)。所有交互态、颜色、间距、排版必须使用文档中定义的 token 和 pattern。

## 架构治理（细则见 [Design System §6](.know/docs/design-system.md#6-架构治理)）

**R1 — 分层与依赖**：tokens → base/layout → components → pages → global（跨切面）。上游不引用下游，pages 之间不互引。`global.css` 只放激活类规则（`.dark`、`.reading-mode`），不是"公共样式"。

**R2 — Rule of Three**：新样式默认写在页面 `<style>`；第 3 次被复用才提升到 `components.css`，同时登记 [Design System §6.3](.know/docs/design-system.md#63-primitive-登记表)。禁止防御性抽象。

**R3 — 变更前 grep**：改任何 `tokens.css / base.css / layout.css / components.css / global.css / components/*.astro` 前，先 `grep -rlE "\b<target>\b" src/` 了解波及面。页面内联样式改动免 grep。

**R4 — 未使用代码扫描**：清理工作完成后或提交前跑 `npm run lint:unused`（=`./scripts/lint-unused-css.sh` + `./scripts/lint-unused-components.sh`）。报告的死代码要么删除，要么在 [Design System §6.3](.know/docs/design-system.md#63-primitive-登记表) 登记合法用途。

## Know

### 文档索引

#### 项目级

- [夏天的爱人 产品路线图](.know/docs/roadmap.md) | v1-v3, M1-M10
- [Design System](.know/docs/design-system.md) | Tokens + Patterns + Components + 架构治理 §6

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
