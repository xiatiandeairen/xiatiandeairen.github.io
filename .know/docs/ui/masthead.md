# Masthead 交互设计

<!-- 核心问题: 站点头部长什么样、怎么操作？ -->

## 1. 布局

### 1.1 区域排布

```
┌─────────────────────────────────────────────────────────┐
│ masthead-outer (full viewport width, 3px solid ink 底线) │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ masthead (max-w-content, centered)                   │ │
│ │                                                      │ │
│ │              masthead-title-row                       │ │
│ │              夏天的爱人 (brand)                        │ │
│ │                                                      │ │
│ │ masthead-meta:                                       │ │
│ │ [spacer] [tagline | sep | nav(文章/关于)] [tools]     │ │
│ │                                                      │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 1.2 元素说明

| 元素 | 类名 | 字号 | 字体 | 颜色 |
|------|------|------|------|------|
| 站名 | `.masthead-brand` | `--font-masthead` | Playfair Display 700 | `--ink` |
| 标语 | `.masthead-tagline` | `--font-caption` | Inter 500 | `--ink-muted` |
| 分隔线 | `.masthead-sep` | — | — | `--rule` |
| 导航链接 | `.masthead-nav a` | `--font-caption` | serif 700 | `--ink-muted` |
| 工具按钮 | `.icon-button` | — | — | `--ink-ghost` |

### 1.3 响应式

| 断点 | 变化 |
|------|------|
| ≤767px | meta 行换行居中；spacer 隐藏；tools 不再 flex:1；icon-button 缩至 26px |

## 2. 交互流程

### 2.1 导航

```
用户到达页面 → 看到站名 + 标语 + 导航
  → hover 导航链接 → 底部下划线展开 + 背景色亮化
  → 点击 → 跳转对应页面
```

### 2.2 工具栏

```
搜索按钮 hover → expandable 展开显示快捷键标签 (hover:hover only)
  → 点击 → search overlay 弹出

语言/主题按钮 hover → dropdown 弹出
  → 选择选项 → 切换语言/主题 → dropdown 收起
```

## 3. 状态与样式

### 3.1 masthead-brand（站名）

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | Playfair Display 700, `--ink`, 无下划线 | — |
| hover | mouse enter | `color: --ink-hover` + `::after` 下划线从左展开至 100% | `--transition-color` + width `--duration-slow` ease |
| focus-visible | keyboard | 同 hover | — |

Pattern: **hover-underline**

### 3.2 masthead-nav a（导航链接）

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | serif 700, `--ink-muted`, `transparent` bg | — |
| hover | mouse enter | `color: --ink`, `bg: --bg-subtle`, `::after` 下划线展开 | `--transition-color` + width `--duration-slow` ease |
| active | `.active` class | 同 hover（当前页面） | — |
| focus-visible | keyboard | 同 hover | — |

Pattern: **hover-underline** + 背景色变化

### 3.3 icon-button（工具按钮）

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | 28×28, `--ink-ghost`, `transparent` bg + border | — |
| hover | mouse enter | `color: --ink`, `bg: --bg-subtle`, `border: --rule` | `--transition-color` |
| dropdown-open | parent `.has-dropdown:hover` | 同 hover（持续） | — |

### 3.4 icon-button.expandable（搜索按钮）

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| collapsed | default | 28px 方块, label `max-width:0 opacity:0` | — |
| expanded | hover (hover:hover) | `max-width: 90px`, label 淡入 + `rule-light` 边框 | expand 0.6s ease-default, label opacity 0.5s delay 0.1s |

Platform: 仅 `@media (hover: hover)` 生效

### 3.5 Dropdown

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| hidden | default | `opacity:0`, `visibility:hidden`, `translateY(-4px)` | — |
| visible | parent `.has-dropdown:hover` | `opacity:1`, `visibility:visible`, `translateY(0)` | `--duration-normal` `--ease-default` |

Platform: 仅 `@media (hover: hover)` 生效

### 3.6 Dropdown Item

| state | trigger | visual | timing |
|-------|---------|--------|--------|
| normal | default | `--ink-muted`, `transparent` bg, 无左线 | — |
| hover | mouse enter | `color: --ink`, `bg: --rule-faint` | `--transition-color` |
| active | `.active` class | `color: --ink-hover`, `font-weight:600`, 左线 2px 40% 高 | `--transition-all` |
| active:hover | active + mouse | `color: --ink-hover`, `bg: --rule-faint`, 左线保持 | `--transition-color` |
