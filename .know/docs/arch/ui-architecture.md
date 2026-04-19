# UI Architecture 架构设计

## 1. 定位与边界

### 职责

定义项目 UI 层的静态架构——CSS 怎么分层、组件怎么分类组织、页面怎么组装内容。读者看完应能 5 分钟建立"骨架"心智模型。

### 不负责

- 原子变量内容（→ [design-tokens](design-tokens.md)）
- 视觉/交互 pattern 内容（→ [design-patterns](design-patterns.md)）
- 规则强制与死代码扫描（→ [design-governance](design-governance.md)）
- 单次设计探索归档（→ [design-archive](design-archive.md)）
- 业务逻辑实现细节（→ 各模块 tech 文档）

## 2. 结构与交互

### 组件图

```
┌─────────────────────────────────────────────────────────────────┐
│                       Page 层（src/pages/）                      │
│   index | about | archive | series | search | notes/[slug]      │
│         + en/* 镜像                                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │ 选择
                       ▼
        ┌──────────────┴──────────────┐
        ▼                              ▼
   [Layout 层]                    [Component 层]
   BaseLayout（chrome）             Content / Data / Navigation / Layout
   IndexLayout（首页变体）          ArticleCard, NoteCard, MetaFields...
   NoteLayout（文章 wrapper）
        │                              │
        └──────────┬───────────────────┘
                   │ 通过 class 应用
                   ▼
          [Style 层（src/styles/）]
       tokens.css ─→ base/layout/masthead/dropdown/search
                  ─→ components.css
                  ─→ pages 内联 <style>（scoped）
                  ─→ global.css（仅跨切面激活类 .dark / .reading-mode）

  数据流: src/content/notes/*.md ─→ src/utils/* ─→ getStaticPaths
       ─→ page props ─→ render（marked + addHeadingIds → HTML）
```

### 组件表

| 层 | 职责 | 边界规则 |
|----|------|---------|
| **Style 层** | 5 个 CSS 子层提供 token / base / layout / shared primitive，加 page-private 内联 + global 跨切面 | 上游不引下游；pages 间不互引；global.css 仅放 `.dark` / `.reading-mode` 等激活类 |
| **Component 层** | 4 类可复用 UI 组件 | 每类有职责约定；新组件必须分类；复用 ≥80% HTML 加 variant，否则拆新组件 |
| **Layout 层** | 3 个页面 chrome 模板，注入 SEO / sidebar / masthead / footer | BaseLayout 是基础；IndexLayout / NoteLayout 是变体；不应再增加新 Layout 除非有强分支 |
| **Page 层** | 路由入口 + getStaticPaths 数据源 + 内联 page-private 样式 | i18n 用 `src/pages/en/` 镜像；`/notes/[slug]` 是动态路由；page 私有样式优先于 components.css |

#### 2.1 Style 层细节（CSS 5 层依赖）

```
tokens.css            ← 最稳定（颜色/字号/间距/动效变量）
  │
  ├─ base.css         ← HTML tag 级 reset / 排版默认
  ├─ layout.css       ← 页面容器 / 双栏栅格 / featured 区
  ├─ masthead.css     ← 站头独立模块
  ├─ dropdown.css     ← 通用下拉
  └─ search.css       ← 搜索 overlay
        │
        └─→ components.css      ← 共享 primitive（白名单制，登记 governance §2.3）
              │
              └─→ pages/*.astro 内联 <style>   ← 页面私有，零下游
                    │
                    └─→ global.css           ← 跨切面激活类（不是公共样式）
```

引入顺序在 `src/styles/global.css` 头部固定，不可乱：

```css
@import './tokens.css';
@import './base.css';
@import './layout.css';
@import './masthead.css';
@import './dropdown.css';
@import './search.css';
@import './components.css';
```

#### 2.2 Component 层细节（4 类 + 现有成员）

| 类别 | 职责 | 现有组件 |
|------|------|---------|
| **Content** | 展示文章/笔记数据 | ArticleCard, NoteCard, VersionDiff |
| **Data** | 展示结构化元信息 | MetaFields, TagHierarchy |
| **Navigation** | 页面间导航/筛选 / 快捷键帮助 | Pagination, ShortcutsHelp |
| **Layout** | 区域组合/容器 | Sidebar |

**Props 约定**：
- `lang?: Lang`——所有组件可选，默认 `'zh'`
- `note: Note`——Content 类必传
- `variant?: string`——视觉变体（string union）
- `show*?: boolean`——区块开关，默认 false
- `mode?: string`——场景切换（非视觉变体）
- `class?: string`——允许外部追加

**粒度规则**：
- 与现有组件共享 ≥80% HTML → 加 variant prop
- 共享数据但结构不同 → 拆新组件，同 category
- 数据类型也不同 → 拆新组件，可能新 category

**头注释模板**：
```astro
---
/* Patterns: hover-indicator, hover-shift, card-divider
   Tokens: --font-h3, --ink-muted, --space-3
   See: .know/docs/arch/design-tokens.md + design-patterns.md */
---
```

#### 2.3 Page 层细节

**路由约定**：

| 路径模式 | 文件 | 说明 |
|---------|------|------|
| `/` | `src/pages/index.astro` | 首页 |
| `/about` `/archive` `/series` `/search` `/subscribe` | `src/pages/{name}.astro` | 静态页 |
| `/notes/<slug>` | `src/pages/notes/[slug].astro` | 动态文章详情 |
| `/tags/<name>` `/topics/<name>` `/page/<n>` | `src/pages/{folder}/[param].astro` | 列表分页 |
| `/tags/` `/topics/` | `src/pages/{folder}/index.astro` | 标签 / 主题聚合首页 |
| `/en/*` | `src/pages/en/*.astro` | 英文镜像（5/6 中文页有对等：subscribe 暂无英文版） |
| `/og/<slug>` | `src/pages/og/` | 构建时生成的 OG 图 |
| `/rss.xml` | `src/pages/rss.xml.ts` | RSS 输出端点 |

**i18n**：
- `src/i18n/{zh,en}.ts` 提供翻译 + 路径切换工具
- 默认中文在根；英文在 `/en/*`
- 镜像策略：每个中文页应有英文对等（当前 subscribe 例外，未做英文版）
- 公共业务逻辑放 `src/utils/`，组件通过 `lang` prop 切换文案

**3 个 Layout**：

| Layout | 用途 | 注入内容 |
|--------|------|---------|
| `BaseLayout` | 所有页面默认 chrome | masthead + sidebar + footer + SEO meta + JSON-LD |
| `IndexLayout` | 首页特殊变体 | BaseLayout 的薄包装 |
| `NoteLayout` | 文章详情包装 | BaseLayout + `<article>` 容器 |

**渲染管线（以 `/notes/[slug]` 为例）**：

```
1. getStaticPaths()
     ├─ getAllNotes()                                  → src/utils/notes.ts
     ├─ extractToc(content)                            → 解 ## / ### → {id, text, level}
     ├─ getRelated(note, allNotes)                     → tag overlap 排序取 3
     ├─ marked.parse(contentWithoutH1)                 → HTML
     └─ addHeadingIds(html)                            → 后处理给 h2/h3 注入 id
                                                          （slugify 算法与 extractToc 共用）
2. 返回 props（note, htmlContent, toc, related, prev/next, series...）
3. <NoteLayout> wrap <article> + toc-float + reading-toggle + type-controls + giscus + nav
4. 客户端 inline script 处理：reading mode toggle / type controls / TOC scroll spy / 复制按钮
```

**搜索索引**：构建时 `src/integrations/generate-search-index.ts` 扫描所有 notes → 输出 `public/search-index.json`，客户端 fetch + 内存搜索（无服务端）。

### 数据流

```
src/content/notes/*.md
        │ (zod schema 校验：src/utils/schema.ts)
        ▼
src/utils/notes.ts (getAllNotes / sortNotes)
        │
        ├─→ src/utils/seo.ts (generateSEOMeta)
        ├─→ src/utils/tags.ts (标签层级解析)
        ├─→ src/utils/search.ts (搜索辅助)
        │
        ▼
[Page].getStaticPaths() → props
        │
        ▼
<Layout> → <Component> → CSS（tokens → patterns → primitives → page-style → global）
        │
        ▼
HTML output (dist/) → GitHub Pages
```

| 来源 | 目标 | 数据格式 | 类型 | 说明 |
|------|------|---------|------|------|
| `src/content/notes/*.md` | zod schema → page props | Frontmatter (validated) + Markdown | 强 | schema 失败 → build fail |
| `src/utils/*.ts` | page `getStaticPaths` | TS function exports | 强 | 数据来源单一入口 |
| `src/i18n/*.ts` | components / layouts | translation maps + helpers | 强 | 缺失 key → 编译错误（TS） |
| `src/integrations/generate-search-index.ts` | `public/search-index.json` | JSON 文件 | 弱 | 缺失 → 搜索功能不可用但站点不挂 |
| Layout / Component / Style | HTML output | Astro template + CSS | 强 | — |

## 3. 设计决策

### 驱动因素

| 因素 | 类型 | 对架构的影响 |
|------|------|------------|
| 个人内容站，规模有限 | 业务需求 | 不引入 React/Vue，零客户端 framework；交互用 inline script |
| 部署 GitHub Pages，纯静态 | 技术约束 | output: static；所有数据构建时确定（getStaticPaths） |
| 中英双语 | 业务需求 | 文件镜像 `src/pages/en/`，i18n 用静态 maps，运行时不切语言 |
| 设计漂移成本最高 | 质量要求 | CSS 5 层强约束 + Component 4 类强约束 + governance 自动检测 |
| 文章结构复杂（TOC、系列、相关） | 业务需求 | getStaticPaths 一次性算清所有派生数据；render 时只查 props |

### 关键选择

| 决策 | 选择 | 被拒方案 | 为什么 |
|------|------|---------|--------|
| CSS 组织 | 5 层 + 单向依赖 + Astro scoped 优先 | flat 全部全局 / SCSS 大量嵌套 / CSS-in-JS | 单向依赖让改动半径可预测；Astro scoped 天然隔离页面私有样式 |
| 组件分类 | 4 类（Content/Data/Navigation/Layout） | 按页面拍平 / Atomic Design (atoms/molecules/organisms) | 4 类对小项目够用；Atomic 过度抽象 |
| i18n 实现 | 文件镜像 `pages/en/` + 静态 i18n maps | 运行时切换 / URL 参数 | 静态生成 SEO 友好；镜像简单直接 |
| Markdown 渲染 | marked + 后处理 `addHeadingIds` | marked plugin / 换 markdown-it | 后处理用 regex 简单可控；插件版本 API 不稳定 |
| 搜索 | 构建时索引 + 客户端内存搜索 | 服务端搜索 / Algolia / 客户端 fuzzy | 静态部署无服务端；7 篇文章索引 ~30KB 性能 ok |
| Layout 数量 | 3 个（Base + Index + Note） | 1 个 BaseLayout + slot / 每页一个 | 3 个变体覆盖现有差异；不必要时不再加 |

### 约束

- **禁止**新建 Layout 除非有 ≥2 个页面共用且差异显著（避免 Layout 增殖）
- **禁止**组件直接读 content collection（必须通过 page 注入 props）
- **禁止** i18n 文案散落在组件里（必须走 `src/i18n/*.ts`）
- **必须**新增 page 同时建 zh + en 两份（保持镜像）
- **必须**新增 component 头注释列出 patterns + tokens

## 4. 质量要求

| 属性 | 指标 | 目标 |
|------|------|------|
| CSS 层依赖纯度 | 反向引用次数 | 0（governance lint 检查） |
| 组件分类覆盖 | 全部组件分到 4 类比例 | 100%（8/8 当前） |
| i18n 镜像同步 | zh ↔ en 页面对等率 | 100%（除文章详情）；当前 5/6 中文页有英文镜像 |
| 渲染管线稳定 | TOC 锚点 / heading id 一致率 | 100%（slugify 算法共用） |
| 构建时校验 | content collection schema 通过率 | 100%（schema.ts zod 强制） |
| 搜索索引更新 | 索引覆盖文章数 / 实际文章数 | 100%（构建时全量重生） |
