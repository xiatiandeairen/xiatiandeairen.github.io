# Sidebar Drawer 交互设计

<!-- 核心问题: 侧边栏抽屉长什么样、怎么操作？ -->

## 1. 布局

### 1.1 区域排布

```
                                    屏幕右边缘
                                         │
  ┌──┬─────────────────────┐             │
  │◂ │  sidebar-panel      │─────────────│
  │  │  ┌───────────────┐  │             │
  │  │  │ 热门文章 (×3)  │  │             │
  │  │  ├───────────────┤  │             │
  │  │  │ 系列专栏 (×3)  │  │             │
  │  │  ├───────────────┤  │             │
  │  │  │ 订阅 RSS       │  │             │
  │  │  └───────────────┘  │             │
  └──┴─────────────────────┘             │

  indicator (20×36)  panel (220px)
```

### 1.2 元素说明

| 元素 | 类名 | 说明 |
|------|------|------|
| 抽屉容器 | `.sidebar-drawer` | fixed, 垂直居中, 包含 indicator + panel |
| 指示器 | `.sidebar-indicator` | 左侧 chevron 图标, 20×36px |
| 面板 | `.sidebar-panel` | 220px 宽, `max-height: 80vh` 可滚动 |

### 1.3 响应式 / 平台

| 条件 | 行为 |
|------|------|
| `@media (hover: hover)` | hover 触发展开/收起 |
| `≤767px` | 隐藏整个 drawer |
| `prefers-reduced-motion` | 禁用动画，瞬时切换 |
| `print` | 隐藏 |

## 2. 交互流程

```
默认状态：drawer 在屏幕右边缘外，仅 indicator 可见
  → hover indicator → indicator 高亮 + nudge 动画
  → 继续 hover → drawer 滑入（indicator 淡出）
  → 浏览面板内容（热门文章 / 系列专栏 / RSS）
  → 鼠标移开 → drawer 滑出回右边缘
```

## 3. 状态与样式

### 3.1 sidebar-drawer

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| closed | default | `translateY(-50%) translateX(220px)`, indicator 可见 | — |
| open | hover drawer (hover:hover) | `translateY(-50%) translateX(0)`, indicator `opacity: 0` | slide-in: 300ms `cubic-bezier(0.16, 1, 0.3, 1)` |
| closing | mouse leave | 回到 closed 状态 | slide-out: 250ms `cubic-bezier(0.5, 0, 1, 0.5)` |

### 3.2 sidebar-indicator

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | `--ink-ghost`, `--paper` bg, `--rule-light` border | — |
| hover | mouse enter | `--ink-muted`, `--rule-faint` bg, `--rule` border + nudge 动画 | color 150ms, nudge 0.8s infinite |
| hidden | drawer open | `opacity: 0` | 200ms ease |

### 3.3 sidebar-panel

| state | trigger | visual |
|-------|---------|--------|
| normal | drawer open | `--paper` bg, `--rule-light` border, left shadow `rgba(0,0,0,0.08)` |
| scroll | 内容超过 80vh | `overflow-y: auto` |
