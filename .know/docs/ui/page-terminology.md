# 页面区域术语表

## What it is
与用户沟通时使用的页面区域统一命名，避免歧义。

## 术语表

| 术语 | CSS 类名 | 说明 |
|------|---------|------|
| Masthead | .site-header | 报头：站名+tagline |
| Nav | .site-nav | 导航栏 |
| Content Area | .site-main | 主内容区 |
| Featured | .layout-featured-main | 首页特稿区（左栏） |
| Sidebar | .layout-featured-side | 首页侧栏（右栏） |
| Divider | .layout-featured-divider | 双栏分隔线 |
| Suggestions | .suggestions-section / .suggestion-card | 首页"猜你想读"客户端推荐 |
| Series Block | .home-series-section / .home-series-card | 首页"系列进行中" |
| Hot Chips | .home-hot-chips / .home-hot-chip | 首页热门标签（圆角）/ 主题（方角）chip |
| Placeholder | .article-card-placeholder / .home-series-card-placeholder | 数据不足时占位 |
| Article Header | header | 文章标题+meta |
| Article Body | #article-body | 正文 prose |
| TOC | .toc-nav | 目录导航 |
| Related | .related-section | 相关推荐 |
| Series Banner / Nav | .series-banner / .series-nav | 系列横幅/导航 |
| Prev/Next | 底部 nav | 时间序前后篇 |
| Footer | .site-footer | 页脚 |
| Section Label | .section-label | 区块标题 |
| Cat Label | .cat-label | 分类标注 |
| Reading Badge | .reading-badge | 阅读时间徽章 |
| Reading Progress | .reading-progress | 滚动进度条 |

## 页面结构图

### 全局框架（所有页面共用）

```
┌─────────────────────────────────┐
│           Masthead              │  .site-header
│    站名 + tagline               │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤  3px 粗线
│             Nav                 │  .site-nav
│   首页 · 归档 · 关于 · 搜索 · ☀  │
├─────────────────────────────────┤  1px 细线
│                                 │
│        Content Area             │  .site-main
│        (各页面内容)              │
│                                 │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤  3px 粗线
│           Footer                │  .site-footer
│    © 2026 · RSS · 基于 Astro    │
└─────────────────────────────────┘
```

### 首页 (index.astro) — see [homepage-layout.md](./homepage-layout.md) for full spec

```
┌────────────────────┬─┬──────────────────┐
│  hero headline     │D│[近期文章]         │  .layout-featured
│                    │i│ 04-14 · 标题…   │
│                    │v│ (SIDEBAR_COUNT 篇)│
│                    │ │                   │
│                    │ │[热门标签 / 主题] │
│                    │ │ [topic chips]     │
│                    │ │ (tag chips)       │
├────────────────────┴─┴──────────────────┤
│  secondary (2 cols)                      │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤
│ [猜你想读] 2×2 cards（客户端）           │  .suggestions-section
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤
│ [系列进行中] 2 cols（不足 placeholder）  │  .home-series-section
├──────────────────────────────────────────┤
│  Pagination                              │
└──────────────────────────────────────────┘
```

### 文章详情页 (notes/[slug].astro)

```
┌─────────────────────────────────┐  .reading-progress (固定顶部 1px)
├─────────────────────────────────┤
│  Series Banner (如有系列)        │  .series-banner
│  系列文章 · AI Skills 系列 · 3/7│
├─────────────────────────────────┤
│  Article Header                 │
│  Cat Label                      │
│  标题 (h1)                      │
│  日期 · Reading Badge · 字数    │
├─────────────────────────────────┤
│                                 │
│  ┌──────────────────┬──────┐   │  .layout-dual
│  │ Article Body     │TOC   │   │
│  │ 正文 (prose)     │目录   │   │  .toc-nav
│  │                  │      │   │
│  │  h2 标题         │Tags  │   │  标签区
│  │  段落...         │标签   │   │
│  │  blockquote      │      │   │
│  │  代码块 [复制]   │Topics│   │  主题区
│  │                  │主题   │   │
│  └──────────────────┴──────┘   │
│                                 │
├─────────────────────────────────┤
│  Related                        │  .related-section
│  ┌─────┐ ┌─────┐ ┌─────┐      │  相关文章 ×2-3
│  │card │ │card │ │card │      │
│  └─────┘ └─────┘ └─────┘      │
├─────────────────────────────────┤
│  Series Nav (如有系列)           │  .series-nav
│  ← 上一篇        下一篇 →      │
├─────────────────────────────────┤
│  Prev/Next                      │  按时间序
│  ← 上一篇        下一篇 →      │
└─────────────────────────────────┘
```

### 归档页 (archive.astro)

```
┌─────────────────────────────────┐
│  标题: 归档                      │
│  副标题: 共 N 篇文章             │
├─────────────────────────────────┤
│  Filter Bar                     │  .filter-bar
│  [全部] [系统设计] [方法论] ...  │  .filter-btn
├─────────────────────────────────┤
│  ┌──────────────────┬──────┐   │  .layout-dual-archive
│  │ 2026             │Stats │   │  年份大字（装饰）
│  │ ┌──────────────┐ │统计  │   │
│  │ │四月 · 7 篇   │ │文章数│   │  月份标签
│  │ │ 文章标题     │ │     │   │
│  │ │ 文章标题     │ │分类  │   │  archive-item
│  │ └──────────────┘ │技术 5│   │
│  │ ┌──────────────┐ │方法 3│   │
│  │ │一月 · 11 篇  │ │     │   │
│  │ │ ...          │ │     │   │
│  │ └──────────────┘ │     │   │
│  └──────────────────┴──────┘   │
└─────────────────────────────────┘
```

### 关于页 (about.astro)

```
┌─────────────────────────────────┐
│  ┌──────────────────┬──────┐   │  .layout-dual-wide
│  │ Main             │Aside │   │
│  │                  │      │   │
│  │ h1: 关于这里     │Avatar│   │  .avatar-box
│  │ Intro (italic)   │      │   │  .about-intro
│  │                  │联系   │   │
│  │ h3: 关于我       │GitHub│   │  .contact-item
│  │ 正文...          │Email │   │
│  │                  │RSS   │   │
│  │ h3: 为什么写     │      │   │
│  │ 正文...          │关于   │   │
│  │                  │本站   │   │  .colophon
│  │ Values Grid      │      │   │  .values-grid
│  │ ┌────┐ ┌────┐   │      │   │  .value-card ×4
│  │ │精确│ │诚实│   │      │   │
│  │ ├────┤ ├────┤   │      │   │
│  │ │留白│ │持久│   │      │   │
│  │ └────┘ └────┘   │      │   │
│  │                  │      │   │
│  │ h3: 版权声明     │      │   │
│  └──────────────────┴──────┘   │
└─────────────────────────────────┘
```

### 搜索页 (search.astro)

```
┌─────────────────────────────────┐
│  标题: 搜索                      │
├─────────────────────────────────┤
│  ┌─────────────────────┬────┐  │
│  │ 搜索框              │搜索│  │  input + button
│  └─────────────────────┴────┘  │
├─────────────────────────────────┤
│  搜索结果 / 空状态提示           │
│  archive-item                   │  复用 archive-item 样式
│  archive-item                   │
│  archive-item                   │
└─────────────────────────────────┘
```

### 系列页 (series.astro)

```
┌─────────────────────────────────┐
│  标题: 系列文章                  │
│  副标题: 按主题组织...           │
├─────────────────────────────────┤
│  h2: AI Skills 系列              │  .series-group
│  共 7 篇                         │
│  ┌───┬──────────────────────┐   │  .series-list
│  │01 │ 文章标题              │   │  .series-item
│  │   │ 约 12 分钟            │   │
│  ├───┼──────────────────────┤   │
│  │02 │ 文章标题              │   │
│  │   │ 约 13 分钟            │   │
│  ├───┼──────────────────────┤   │
│  │...│ ...                   │   │
│  └───┴──────────────────────┘   │
└─────────────────────────────────┘
```

### 标签/主题索引页 (tags/index, topics/index)

```
┌─────────────────────────────────┐
│  标题: 标签索引 / 主题索引       │
├─────────────────────────────────┤
│  标签名              N 篇       │  顶级标签
│    └ 子标签名        N 篇       │  子标签（缩进）
│  标签名              N 篇       │
│  标签名              N 篇       │
└─────────────────────────────────┘
```

## Caveats
术语与 CSS 类名绑定。重命名类名时需同步更新此表和结构图。
移动端 (<768px) 所有双栏布局变为单栏堆叠，Divider 隐藏，TOC 隐藏。
