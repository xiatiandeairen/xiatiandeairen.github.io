# Footer 交互设计

<!-- 核心问题: 页脚长什么样？ -->

## 1. 布局

### 1.1 区域排布

```
┌━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┐
│ 3px solid ink 顶线                                │
│                                                   │
│  © 2026 夏天的爱人              基于 Astro 构建     │
│                                                   │
└───────────────────────────────────────────────────┘
```

### 1.2 元素说明

| 元素 | 类名 | 字号 | 字体 | 颜色 |
|------|------|------|------|------|
| 容器 | `.site-footer` | `--font-label` | Inter | `--ink-faint` |
| 链接 | `.footer-link` | 同容器 | Inter | `--ink-faint` → hover `--ink` |

### 1.3 响应式

| 断点 | 变化 |
|------|------|
| ≤767px | flex → column, 居中对齐, gap 缩小 |

## 2. 交互流程

无复杂交互。链接 hover 变深色。

## 3. 状态与样式

### 3.1 footer-link

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | `--ink-faint` | — |
| hover | mouse enter | `--ink` | `--transition-color` |

功能性链接，hover 变深而非变 `--ink-hover`（有意区别于内容链接）。
