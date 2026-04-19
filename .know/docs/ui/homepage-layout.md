# 首页布局 交互设计

<!-- 核心问题: 首页长什么样、怎么操作？ -->

## 1. 布局

### 1.1 信息架构

```
吸引（headline）→ 留住（sidebar + secondary）→ 发现（suggestions + series）→ 翻页
```

| 区域 | 目的 | 数据来源 |
|------|------|---------|
| **hero headline** | 编辑选择的焦点文章 | `featured` 优先，fallback `notes[0]` |
| **sidebar**（近期文章） | 快速扫读近期 N 篇 | `notes[1 : 1+SIDEBAR_COUNT]`，不足用 placeholder 补齐 |
| **sidebar filler**（热门标签 / 主题） | 站点维度导航入口 | tag/topic 频次 top 3 各一行 |
| **secondary** | 双栏副头条 | `notes[4:6]` |
| **suggestions**（猜你想读） | 个性化推荐（客户端） | `localStorage.recentlyViewed` × 预构 `relatedMap` |
| **series**（系列进行中） | 长线写作入口 | 按 `series.name` 聚合，取 latest 2 |
| pagination | 翻到下一页 | `Pagination` 组件 |
| drawer | 右边缘侧栏发现（与首页独立） | 独立数据 |

### 1.2 区域排布

```
┌────────────────────────────────────────────────────────────┐
│                        masthead                             │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤
│                                   │                         │
│  hero headline (1.6fr)            │1px│ [ 近期文章 ]        │
│  ┌─────────────────────────┐      │rule│ 04-14 · 标题…     │
│  │ 标题 (h2)               │      │    │ 04-14 · 标题…     │
│  │ 摘要 (斜体)             │      │    │ …共 SIDEBAR_COUNT │
│  │ 读者来信 ×3             │      │    │                    │
│  │                         │      │    │ [ 热门标签 / 主题 ]│
│  │                         │      │    │ [AI 工程化][方法论]│
│  │                         │      │    │ (AI)(agent)(…)     │
│  └─────────────────────────┘      │    │                    │
├───────────────────────────────────┴────┴──────────────────┤
│  secondary (grid: 1fr | 1px rule | 1fr)                    │
│  Know: …让 AI 不再重复犯错  │  Sprint: …给 AI 一条流水线   │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤
│  [ 猜你想读 ] （客户端 hydrate，冷启动隐藏）                 │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ 01 kicker    │  │ 02 kicker    │                        │
│  │ 标题         │  │ 标题         │                        │
│  │ 因为你读过《…》│  │ 因为你读过《…》│                        │
│  └──────────────┘  └──────────────┘   …2×2                  │
├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤
│  [ 系列进行中 ]                              全部系列 →      │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ 系列·N 篇    │  │ 系列·筹备中  │ ← placeholder           │
│  │ 系列名       │  │ 敬请期待     │                        │
│  │ 01 最新篇 1  │  │ (虚线 border) │                        │
│  │ 02 最新篇 2  │  │              │                        │
│  └──────────────┘  └──────────────┘                        │
├───────────────────────────────────────────────────────────┤
│  pagination                                                │
│  footer                                                    │
└───────────────────────────────────────────────────────────┘

                                     ┌──┐ ← drawer indicator
                                     │◂ │   (屏幕右边缘)
                                     └──┘
```

### 1.3 Grid 定义

| Primitive | CSS | 用途 |
|-----------|-----|------|
| `layout-featured` | `grid: 1.6fr 1px 1fr; rows: auto auto` | 首屏 2 行（hero+sidebar / secondary） |
| `layout-featured-secondary` | `grid: 1fr 1px 1fr; column: 1/-1` | 双栏副头条（内嵌 row 2） |
| `layout-featured-side-filler` | block in sidebar flex column | 热门标签/主题 block |
| `suggestions-grid` | `grid: repeat(2, 1fr)` | 猜你想读 2×2 |
| `home-series-grid` | `grid: repeat(2, minmax(0,1fr))` | 系列进行中 2 列，不足用 placeholder |

### 1.4 响应式

| 断点 | 变化 |
|------|------|
| ≤767px | `.layout-featured` → 单栏；`.layout-featured-secondary` → 单栏；sidebar 下移 |
| ≤768px | `.home-series-grid` → 1 列 |
| ≤640px | `.suggestions-grid` → 1 列 |

## 2. 交互流程

### 2.1 浏览

```
用户到达首页
  → 视线落在 hero headline 大标题
  → 扫描 sidebar N 篇紧凑列表（快速判断日期+主题）
  → 下滑看 secondary 副头条
  → 进入 suggestions（若个性化触发）/ series（知识线入口）
  → pagination 或 footer
```

### 2.2 个性化推荐（suggestions）

```
首次访问：section[hidden]，不渲染
多次阅读后：
  → 读取 localStorage.recentlyViewed
  → 用 relatedMap 计算 picks
  → 对每个 pick 记录触发源（score 最高的那篇）
  → 展示 2×2 卡，每张底部 "因为你读过《X》"
```

### 2.3 系列入口（series）

```
用户想看完整主题 → 点系列卡名或子条目
系列少于 2 → placeholder 占位（视觉平衡）
点"全部系列 →" → 跳 /series
```

## 3. 状态与样式

### 3.1 ArticleCard — headline

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | h2 `-0.02em`；摘要斜体；读者来信 | — |
| hover | mouse enter | 标题 → `ink-hover` + 下划线 | `--transition-color` |
| active | mouse down | `translateY(0.5px)` | 0.05s ease |
| focus-visible | keyboard | outline 2px ink-hover | — |

### 3.2 ArticleCard — sidebar（紧凑列表）

| state | trigger | visual |
|-------|---------|--------|
| normal | default | grid `auto minmax(0,1fr)`: 日期 (caption, ghost) + 标题 (small, serif, ellipsis)；`1px rule-faint` 下分隔 |
| hover | mouse enter | 标题 `ink-hover` + 下划线 |
| focus | keyboard | outline 2px ink-hover |
| placeholder | 文章不足 | `.article-card-placeholder` opacity 0.4；日期 "—"；标题 "敬请期待" |

### 3.3 ArticleCard — secondary（裸文本）

| state | trigger | visual |
|-------|---------|--------|
| normal | default | h3 标题 + 标题下 `rule-faint` + 摘要 |
| hover | mouse enter | 标题 → `ink-hover` + 下划线（无 padding 位移，无左条） |
| focus-within | keyboard | outline 2px ink-hover |

### 3.4 Hot chips（热门标签 / 主题）

| 变体 | border-radius | 用途 |
|------|---------------|------|
| `.home-hot-chip--tag` | `999px`（圆角胶囊） | 具体 tag |
| `.home-hot-chip--topic` | `var(--radius-sm)`（方角） | 广义 topic |

两者颜色、border 色、字重一致，**仅形状区分**。单行 `flex-wrap: nowrap + overflow: hidden`，超出直接截断。

### 3.5 Suggestion card

| state | trigger | visual |
|-------|---------|--------|
| normal | default | bg-subtle + rule-faint border + radius-md；kicker 在顶（accent），标题 (h3, clamp 2)，底部 hairline + "因为你读过《…》"；右上角序号 `01–04` |
| hover | mouse enter | border → ink-muted；`box-shadow`；`translateY(-1px)`；标题 → ink-hover |

### 3.6 Series card

| state | trigger | visual |
|-------|---------|--------|
| normal | default | bg-subtle + 左 `3px solid accent` + radius-md；顶 meta (accent)，系列名 (h3)，列表 3 篇编号 + 标题 |
| placeholder | 系列不足 | opacity 0.55；左条改为 `dashed rule`；标题 "敬请期待" + 斜体 empty 文案 |

### 3.7 Sidebar Drawer

参见 [sidebar-drawer.md](./sidebar-drawer.md)。

## 4. 固定参数与约束

| 参数 | 值 | 约束 |
|------|-----|------|
| `SIDEBAR_COUNT` | 6 | 改动前必须 playwright 实测首屏 |
| `HOT_CHIP_COUNT` | 3 | 每行 |
| 系列卡数 | 2 | 不足用 placeholder |
| suggestions picks | 4 | 固定 2×2 |
| 基线 viewport | 1280×900 | 首屏不变量基线 |

详见 [CLAUDE.md §首页首屏不变量](../../../CLAUDE.md#首页首屏不变量)。
