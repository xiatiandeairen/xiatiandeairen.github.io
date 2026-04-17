# Search Overlay 交互设计

<!-- 核心问题: 搜索弹层长什么样、怎么操作？ -->

## 1. 布局

### 1.1 区域排布

```
┌─────────────────────────────────────────┐
│  backdrop (fixed inset, blur + 50% 黑)   │
│                                          │
│     ┌──────────────────────────┐         │
│     │ search-dialog (520px max) │         │
│     │ ┌──────────────────────┐ │         │
│     │ │ 🔍 输入框     [ESC]  │ │         │
│     │ ├──────────────────────┤ │         │
│     │ │ result-item          │ │         │
│     │ │ result-item          │ │         │
│     │ │ result-item          │ │         │
│     │ │ ...  (max 50vh 滚动) │ │         │
│     │ └──────────────────────┘ │         │
│     └──────────────────────────┘         │
│                                          │
└─────────────────────────────────────────┘
```

### 1.2 元素说明

| 元素 | 类名 | 字号 | 字体 | 颜色 |
|------|------|------|------|------|
| 输入框 | `.search-input` | `--font-body` | Noto Serif SC 500 | `--ink` |
| 占位文字 | `::placeholder` | — | — | `--ink-ghost` |
| ESC 标签 | `.search-esc` | `--font-xs` | Inter | `--ink-ghost` |
| 结果标题 | `.search-result-title` | `--font-small` | — | `--ink` 600 |
| 结果日期 | `.search-result-date` | `--font-caption` | Inter | `--ink-ghost` |
| 空状态 | `.search-empty` | `--font-small` | — | `--ink-muted` |

## 2. 交互流程

```
搜索按钮点击 / 快捷键 ⌘K
  → overlay 出现（backdrop blur + dialog 居中偏上 15vh）
  → 输入框自动聚焦
  → 实时搜索（标题 + 内容 + 标签）
  → 结果列表更新
  → 点击结果 → 跳转文章
  → 点击 backdrop / 按 ESC → overlay 关闭
```

## 3. 状态与样式

### 3.1 Overlay

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| hidden | default | 不在 DOM 中 | — |
| visible | 搜索触发 | `fixed inset:0`, backdrop `rgba(0,0,0,0.5)` + `blur(4px)`, dialog `paper` bg + `radius-lg` + shadow | — |

### 3.2 search-result-item

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | `--ink`, `transparent` bg, `rule-faint` 底线 | — |
| hover | mouse enter | `bg: --bg-subtle` | `--transition-color` |

### 3.3 search-dialog

| state | trigger | visual |
|-------|---------|--------|
| has-results | 搜索有结果 | 结果列表展示，`max-height: 50vh` 滚动 |
| empty | 搜索无结果 | 居中提示文字 `--ink-muted` |
