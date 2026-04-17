# Design System — 夏天的爱人

报纸/杂志风格。衬线标题 + 无衬线 UI。暖色调纸张感。克制装饰，靠排版建立层级。

---

## 1. Tokens

### 1.1 Colors

| Token | Light | Dark | Usage | Don't |
|-------|-------|------|-------|-------|
| `--paper` | `#f4f0e8` | `#211e1b` | 页面背景 | — |
| `--ink` | `#1a1a1a` | `#ede8e2` | 标题、正文、默认文字 | — |
| `--ink-light` | `#2d2926` | `#ccc6be` | 次级正文（摘要长文） | 不用于 UI 控件 |
| `--ink-muted` | `#4a4440` | `#a8a298` | 弱化文字（导航、摘要） | — |
| `--ink-faint` | `#5c5650` | `#8e8880` | 标签、日期、辅助信息 | — |
| `--ink-ghost` | `#6e6860` | `#706a62` | 最弱文字（占位符、图标默认） | 不用于可读正文 |
| `--ink-hover` | `#5a7a6a` | `#8abaa5` | **所有交互态**：链接/标题/指示线 hover | 不用于默认态、装饰、背景 |
| `--accent` | `#8b0000` | `#d4a574` | 仅 reading-progress 进度条 | **不用于 hover** |
| `--rule` | `#c8c0b4` | `#3d3832` | 主分隔线 | — |
| `--rule-light` | `#ddd5ca` | `#302c28` | 次分隔线（边框） | — |
| `--rule-faint` | `#ece7de` | `#292622` | 最弱分隔线（卡片间） | — |
| `--bg-subtle` | `#efebe3` | `#282420` | hover 背景、评论区底色 | — |

### 1.2 Typography

| Role | Token | Font Family | Weight | Usage |
|------|-------|-------------|--------|-------|
| Masthead | `--font-masthead` | Playfair Display | 700 | 站名 |
| H1 | `--font-h1` | Playfair Display | 700 | 无图头条标题 |
| H2 | `--font-h2` | Playfair Display | 700 | 头条标题、列表标题 |
| H3 | `--font-h3` | Noto Serif SC / Georgia | 700 | 次要标题、sidebar 标题 |
| Body | `--font-body` | Noto Serif SC / Georgia | 400 | 正文、头条摘要 |
| Small | `--font-small` | Noto Serif SC / Georgia | 400 | 卡片摘要 |
| Caption | `--font-caption` | Inter | 400-600 | UI 控件、元信息 |
| Label | `--font-label` | Inter | 500-600 | 全大写标签 |

**规则**：标题用 `font-display`（Playfair Display），正文用 `font-serif`，UI 用 `font-sans`（Inter）。

### 1.3 Spacing

4px 基础单位。使用 `--space-{n}`，不硬编码数值。

| Token | Value | Typical Usage |
|-------|-------|---------------|
| `--space-1` | 4px | 行内间隙、tag margin |
| `--space-2` | 8px | 紧凑间距、评论内 padding |
| `--space-3` | 12px | 卡片 padding、组间距 |
| `--space-4` | 16px | 区域间距（title→excerpt） |
| `--space-5` | 24px | 大区域间距 |
| `--space-6` | 32px | 板块间距 |
| `--space-7` | 48px | 页面级间距 |
| `--space-page` | clamp(16px, 2.5vw+4px, 40px) | 页面边距 |

### 1.4 Transitions

| Token | Value | Usage |
|-------|-------|-------|
| `--transition-color` | `0.2s ease` | 颜色变化（hover 文字、背景） |
| `--transition-all` | `0.25s cubic-bezier(0.4,0,0.2,1)` | 位移、尺寸（padding 偏移、线条展开） |
| `--duration-slow` | `0.25s` | 下划线展开 |

**规则**：不在组件内自定义 timing，统一用 token。

### 1.5 Other

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-sm` | 3px | 微圆角（评论区、图片） |
| `--radius-md` | 4px | 按钮、dropdown |
| `--radius-pill` | 10px | 药丸形 badge |
| `--z-sticky` | 10 | dropdown |
| `--z-overlay` | 1000 | search overlay |

---

## 2. Patterns

可复用的交互/排版配方。组件通过组合 pattern 获得一致行为。

### 2.1 hover-indicator

左侧 2px 竖线，hover 时从 0 展开。

```
构成：::before + position:absolute + width:2px + background:var(--ink-hover)
动画：height 0→50-65% | 或 scaleY(0→0.65)
```

| 使用者 | 高度 | 定位 |
|--------|------|------|
| article-card-headline | 65% | top:0, scaleY |
| article-card-sidebar | 50% | top:50%, translateY(-50%) |
| article-card-secondary | 50% | top:50%, translateY(-50%) |
| article-card-list | 60% | top:50%, translateY(-50%) |
| post-card | 60% | top:50%, translateY(-50%) |

### 2.2 hover-shift

hover 时 padding-left 微偏移，暗示可交互。

```
构成：padding-left 0→var(--space-1) + transition:var(--transition-all)
```

使用者：article-card-headline, article-card-list, archive-item, search-result-card

### 2.3 hover-underline

底部 1px 线从左向右展开。

```
构成：::after + position:absolute + bottom:0 + width:0→100% + background:var(--ink-hover)
动画：width var(--duration-slow) ease
```

使用者：masthead-brand, masthead-nav a

### 2.4 text-hierarchy

文字颜色从深到浅，表达信息层级。

```
ink → ink-light → ink-muted → ink-faint → ink-ghost
标题    次级正文    弱化文字    标签/日期    占位/图标
```

### 2.5 card-divider

卡片之间的细分隔线。

```
构成：border-bottom: 1px solid var(--rule-faint)
末项：:last-child { border-bottom: none }
```

使用者：article-card-sidebar, article-card-list, post-card, comment-placeholder

### 2.6 section-break

区域间的强分隔。

| 变体 | 样式 | 使用者 |
|------|------|--------|
| heavy | `3px solid var(--ink)` | masthead-outer, remaining-section |
| labeled | `1.5px solid var(--ink)` + uppercase label | section-label |
| light | `1px solid var(--rule)` | layout-featured-secondary |

---

## 3. Components

每个组件的状态定义。格式遵循 [CSS 状态表模式](.know/entries/concept/css-state-table-pattern.md)。

### 3.1 ArticleCard

4 变体共享 `article-card-title a` 基础样式（hover → `--ink-hover`）。

| 变体 | 标题 | 摘要 | 额外 | Patterns |
|------|------|------|------|----------|
| headline | h2, `--font-h3` | `--font-small`, italic, flex:1 | 评论占位（bg-subtle + ink-hover 左边框） | hover-indicator, hover-shift |
| sidebar | h4, `--font-body` | `--font-caption` | — | hover-indicator, card-divider |
| secondary | h3, `--font-h3` | `--font-small` | dual-column grid 内 | hover-indicator, hover-shift |
| list | h2, `--font-h3` | `--font-small` + meta 行 | 日期、阅读时间、tags | hover-indicator, hover-shift, card-divider |

### 3.2 NoteCard (post-card)

```
States:
  normal    default       → rule-faint bottom border
  hover     mouse enter   → left indicator 60%, padding-left shift, title → ink-hover
```

### 3.3 Masthead

```
States:
  brand:hover       → color:ink-hover, underline expand (hover-underline)
  nav:hover         → color:ink, bg:bg-subtle, underline expand (hover-underline)
  icon-button:hover → color:ink, bg:bg-subtle, border:rule
  expandable:hover  → max-width expand, label fade in (hover:hover only)
```

### 3.4 Dropdown

```
States:
  menu         hidden → hover parent → opacity:1, translateY(0)
  item:normal  ink-muted, transparent bg
  item:hover   ink, rule-faint bg
  item:active  ink-hover, font-weight:600, left indicator 40%
```

### 3.5 Search

```
States:
  overlay     hidden → open → fixed fullscreen, backdrop blur
  result:hover  bg:bg-subtle
```

### 3.6 Sidebar (drawer)

```
States:
  closed      translateX(220px), indicator visible
  open:hover  translateX(0), indicator fade out (hover:hover only)
```

### 3.7 Pagination

```
States:
  link:normal   ink-muted, rule-light border
  link:hover    ink, rule border
  current       ink, font-weight:600
```

### 3.8 Layout Primitives

| Primitive | CSS | Usage |
|-----------|-----|-------|
| `layout-featured` | 3-column grid (1.6fr 1px 1fr) | 首页头条区 |
| `layout-featured-secondary` | 3-column grid (1fr 1px 1fr) inside featured | 次要双栏 |
| `layout-dual` | 2-column grid (1fr 260px) | 文章页 |
| `site-main` | max-width + padding + mx-auto | 内容区居中 |

---

## 4. Don'ts

1. **不用 `--accent` 做 hover 色** — 用 `--ink-hover`。`--accent` 仅用于 reading-progress
2. **不硬编码颜色/间距** — 用 token。`color: #5a7a6a` ✗ → `color: var(--ink-hover)` ✓
3. **不自定义 transition timing** — 用 `--transition-color` 或 `--transition-all`
4. **不用 float 做需要 flex 自适应的布局** — float 和 flex 规范层面不兼容
5. **不在用户未要求时夹带额外样式变更** — 改什么就改什么，不顺手优化

### 例外

- **≤2px 微调允许硬编码** — `1px` 分隔线、`2px` 微间距不需要 token，token 最小单位为 `--space-1`(4px)
- **专用动画允许自定义 timing** — expandable、sidebar-drawer 等特殊交互的 duration/easing 可自定义，需在 CSS 注释中说明理由

---

## 5. Component System

### 5.1 分类

| 类别 | 职责 | 现有组件 |
|------|------|---------|
| **Content** | 展示文章/笔记数据 | ArticleCard, NoteCard, VersionDiff |
| **Data** | 展示结构化元信息 | MetaFields, TagHierarchy |
| **Navigation** | 页面间导航/筛选 | Pagination |
| **Layout** | 区域组合/容器 | Sidebar |

### 5.2 Props 约定

```typescript
lang?: Lang            // 所有组件可选，默认 'zh'
note: Note             // Content 类必传
variant?: string       // 视觉变体（string union），多种展示形态时用
show*?: boolean        // 可选区块开关，默认 false
mode?: string          // 场景切换（非视觉变体）
class?: string         // 允许外部追加 class
```

### 5.3 粒度规则

```
与现有组件共享 80%+ HTML 结构？ → 加 variant prop
共享数据类型但结构不同？       → 拆新组件，同 category
数据类型也不同？               → 拆新组件，可能新 category
```

### 5.4 Design System 衔接

组件文件头注释标注使用的 pattern 和 token：

```astro
---
/* Patterns: hover-indicator, hover-shift, card-divider
   Tokens: --font-h3, --ink-muted, --space-3
   See: src/styles/DESIGN_SYSTEM.md */
---
```

### 5.5 目录结构

`src/components/` 平铺。超过 15 个组件时按类别建子目录。

### 5.6 New Component Checklist

- [ ] 确定类别（Content / Data / Navigation / Layout）
- [ ] Props 遵循约定（lang 可选默认 zh、variant 用 string union）
- [ ] 头注释标注 Patterns + Tokens
- [ ] 判断粒度（80%+ 共享 → 变体，否则拆新组件）
- [ ] 样式使用本文档中的 token 和 pattern
