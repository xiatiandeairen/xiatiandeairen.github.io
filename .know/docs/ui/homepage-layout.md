# 首页布局 交互设计

<!-- 核心问题: 首页长什么样、怎么操作？ -->

## 1. 布局

### 1.1 信息架构

```
吸引（headline）→ 留住（sidebar + secondary）→ 服务（list + drawer）
```

| 区域 | 目的 | 数据切片 |
|------|------|---------|
| headline | 编辑选择，最重要的一篇 | `notes[0]` |
| sidebar | 快速浏览近期内容 | `notes[1:4]`（3 篇）|
| secondary | 编辑推荐，双栏补充 | `notes[4:6]`（2 篇）|
| list | 翻阅剩余文章 | `notes[6:]` |
| drawer | 侧边发现（热门+专栏+RSS）| 独立数据源 |

### 1.2 区域排布

```
┌─────────────────────────────────────────────────────────┐
│                     masthead                             │
│              夏天的爱人（brand）                           │
│     tagline  |  nav(文章/关于)  |  tools(搜索/语言/主题)   │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤
│                                   │                      │
│  headline (1.6fr)                 │ 1px │  sidebar (1fr)  │
│  ┌─────────────────────────┐      │rule │  ┌───────────┐  │
│  │ 标题 (h2, -0.02em)     │      │     │  │ 近期文章   │  │
│  │                         │      │     │  ├───────────┤  │
│  │ 摘要 (斜体, flex:1)    │      │     │  │ card ×3   │  │
│  │                         │      │     │  │ (sidebar)  │  │
│  │ ─── 读者来信 ───        │      │     │  └───────────┘  │
│  │ " 评论... — 读者        │      │     │                 │
│  └─────────────────────────┘      │     │                 │
│                                   │     │                 │
├───────────────────────────────────┴─────┴─────────────────┤
│  secondary (grid: 1fr | 1px rule | 1fr)                   │
│  ┌─────────────────┐  │  ┌─────────────────┐              │
│  │ card (secondary) │  │  │ card (secondary) │              │
│  └─────────────────┘  │  └─────────────────┘              │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤
│  remaining-section (3px solid ink 顶线)                    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ card (list) × N                                       │ │
│  │ card (list)                                           │ │
│  │ card (list)                                           │ │
│  └───────────────────────────────────────────────────────┘ │
├───────────────────────────────────────────────────────────┤
│  pagination                                               │
├───────────────────────────────────────────────────────────┤
│  footer                                                   │
└───────────────────────────────────────────────────────────┘

                                          ┌──┐ ← drawer indicator
                                          │◂ │   (屏幕右边缘)
                                          └──┘
```

### 1.3 Grid 定义

| Primitive | CSS | 用途 |
|-----------|-----|------|
| `layout-featured` | `grid: 1.6fr 1px 1fr` | 头条 + 分隔线 + 侧栏 |
| `layout-featured-secondary` | `grid: 1fr 1px 1fr`, `grid-column: 1/-1` | 双栏推荐，嵌在 featured 内部 |
| `remaining-section` | 单栏 + `border-top: 3px solid ink` | 列表区 |

### 1.4 响应式

| 断点 | 变化 |
|------|------|
| ≤767px | featured → 单栏；分隔线隐藏；secondary → 单栏；sidebar 下移 |

## 2. 交互流程

### 2.1 文章浏览

```
用户到达首页
  → 视线落在 headline 大标题（视觉锚点）
  → 扫描 sidebar 三篇近期文章（快速判断）
  → 下滑看 secondary 双栏推荐
  → 继续下滑翻阅 list 列表
  → 任意卡片点击 → 跳转文章详情页
```

### 2.2 侧边栏发现

```
鼠标移到屏幕右边缘 indicator
  → drawer 滑出（300ms ease-out）
  → 浏览热门文章 / 系列专栏 / RSS
  → 鼠标移开 → drawer 收回（250ms ease-in）
```

### 2.3 全局工具

```
搜索图标 hover → expandable 展开显示快捷键
语言/主题图标 hover → dropdown 弹出选项列表
```

## 3. 状态与样式

### 3.1 ArticleCard — headline

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | h2 标题 `letter-spacing: -0.02em`, `margin-left: -0.04em`; 摘要斜体 `ink-light`; 读者来信区（装饰引号 `"` 3.5rem + 斜体引用 + small-caps 署名） | — |
| hover | mouse enter | 标题 → `ink-hover` + 下划线（`underline-offset: 0.15em`, `thickness: 1px`） | `--transition-color` |
| active | mouse down | `translateY(0.5px)` | 0.05s ease |
| focus-visible | keyboard tab | `outline: 2px solid ink-hover; offset: 3px` | — |

### 3.2 ArticleCard — sidebar

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | 紧凑：`font-body` 标题 + `font-caption` 摘要 `ink-faint`; `rule-faint` 底线分隔 | — |
| hover | mouse enter | 行背景 `rgba(0,0,0,0.02)` + 标题 → `ink-hover` | `--transition-color` |
| active | mouse down | 背景加深 `rgba(0,0,0,0.04)` | `--transition-color` |
| focus-within | keyboard tab | `outline: 2px solid ink-hover; offset: 2px` | — |

### 3.3 ArticleCard — secondary

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | `font-h3` 标题 + 标题下 `1px rule-faint` 分隔线 + 摘要 | — |
| hover | mouse enter | 左线 2px `ink-hover` 50% 高 + `padding-left` 偏移 + 标题下划线 + 标题 → `ink-hover` | `--transition-all` |
| active | mouse down | `translateY(0.5px)` | 0.05s ease |
| focus-within | keyboard tab | `outline: 2px solid ink-hover; offset: 3px` | — |

### 3.4 ArticleCard — list

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | `font-h3` 标题 + 摘要 + meta 行（`letter-spacing: 0.04em`）; `rule-faint` 底线 | — |
| hover | mouse enter | 行背景 `rgba(0,0,0,0.02)` + 左线 2px `ink-hover` 60% 高 + 标题 → `ink-hover` | `--transition-color` + `--transition-all` |
| active | mouse down | 背景加深 `rgba(0,0,0,0.04)` | `--transition-color` |
| focus-within | keyboard tab | `outline: 2px solid ink-hover; offset: 2px` | — |

### 3.5 Hover 不对称设计

| 变体 | 主信号 | 理由 |
|------|--------|------|
| headline | 标题下划线 | 大面积卡片，排版变化比空间变化更优雅 |
| sidebar | 行背景色 | 紧凑项，整行命中反馈 |
| secondary | 左线 + 下划线 | 双栏内方向引导 + 排版变化 |
| list | 行背景 + 左线 | 长列表扫描定位，双信号 |

### 3.6 Sidebar Drawer

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| closed | default | `translateX(220px)`, indicator 可见 | — |
| open | hover indicator | `translateX(0)`, indicator 淡出 | slide-in 300ms ease-out, indicator 200ms fade |
| closing | mouse leave | `translateX(220px)`, indicator 淡入 | slide-out 250ms ease-in |

### 3.7 评论区（读者来信）

| 元素 | 样式 |
|------|------|
| 装饰引号 `"` | Georgia 3.5rem, `rule-light` 色, 绝对定位左上角 |
| 引用文字 | 斜体 serif, `--font-small`, `ink-muted` |
| 署名 | Inter, `font-variant: small-caps`, `letter-spacing: 0.05em`, `ink-ghost`, `— 读者` |
| 上方分隔 | `1px solid rule-light` |
| 评论间分隔 | `1px solid rule-faint` |
