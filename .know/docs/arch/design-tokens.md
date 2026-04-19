# Design Tokens 架构设计

## 1. 定位与边界

### 职责

负责设计系统的原子级变量定义（颜色、字号、间距、动效、圆角、z-index），为 base/layout/components/pages 各层提供单一真实数据源。

### 不负责

- 视觉/交互 pattern 的组合方式（→ [design-patterns](design-patterns.md)）
- 共享 class 的命名与登记（→ [design-governance §2.3](design-governance.md#23-primitive-登记表)）
- 组件级 prop 设计（→ 各 component 文件头注释）

## 2. 结构与交互

### 组件图

```
              [src/styles/tokens.css]
                       │
       ┌────────┬──────┴──────┬──────────┬───────┐
       ▼        ▼             ▼          ▼       ▼
   --color-* --font-*     --space-*  --duration --radius/--z-*
    颜色族    字号族         间距族     动效族       几何族
       │        │             │          │       │
       └────────┴──────┬──────┴──────────┴───────┘
                       ▼
         所有 CSS 文件 + Astro <style> 通过 var() 消费
```

### 组件表

| 组件 | 职责 | 边界规则 |
|------|------|---------|
| `--color-*` | paper / ink-* / rule-* / accent / surface 颜色变量 | 必须 light + dark 双套；消费方禁止写 hex |
| `--font-*` | masthead / h1-h3 / body / small / caption / label 8 档字号 | 用 `clamp()` 内置响应式；消费方禁止覆盖 px |
| `--space-*` | 4px 基础的 1-8 阶梯 + `--space-page` 响应式页边距（共 9 档） | 禁止跨阶混合（`var(--space-3)+4px`） |
| `--duration-*` / `--transition-*` | 时长与缓动语义化别名 | 禁止字面量 `transition: all 0.2s ease` |
| `--radius-*` / `--z-*` | 圆角 + z 层级 | z-index 仅用 token，不写裸数字 |

#### 颜色 token 具体值

| Token | Light | Dark | Usage | Don't |
|-------|-------|------|-------|-------|
| `--paper` | `#f4f0e8` | `#211e1b` | 页面背景 | — |
| `--ink` | `#1a1a1a` | `#ede8e2` | 标题、正文、默认文字 | — |
| `--ink-light` | `#2d2926` | `#ccc6be` | 次级正文 | 不用于 UI 控件 |
| `--ink-muted` | `#4a4440` | `#a8a298` | 弱化文字（导航、摘要） | — |
| `--ink-faint` | `#5c5650` | `#8e8880` | 标签、日期、辅助 | — |
| `--ink-ghost` | `#6e6860` | `#706a62` | 占位符、图标默认 | 不用于可读正文 |
| `--ink-hover` | `#5a7a6a` | `#8abaa5` | **所有交互 hover** | 不用于默认/装饰/背景 |
| `--accent` | `#8b0000` | `#d4a574` | 仅 reading-progress 进度条 | **不用于 hover** |
| `--rule` | `#c8c0b4` | `#3d3832` | 主分隔线 | — |
| `--rule-light` | `#ddd5ca` | `#302c28` | 次分隔线（边框） | — |
| `--rule-faint` | `#ece7de` | `#292622` | 最弱分隔线 | — |
| `--bg-subtle` | `#efebe3` | `#282420` | hover 背景、评论区底色 | — |

#### 字号 token

| Role | Token | Font Family | Weight | Usage |
|------|-------|-------------|--------|-------|
| Masthead | `--font-masthead` | Playfair Display | 700 | 站名 |
| H1 | `--font-h1` | Playfair Display | 700 | 无图头条 |
| H2 | `--font-h2` | Playfair Display | 700 | 头条/列表标题 |
| H3 | `--font-h3` | Noto Serif SC / Georgia | 700 | 次要标题 |
| Body | `--font-body` | Noto Serif SC / Georgia | 400 | 正文 |
| Small | `--font-small` | Noto Serif SC / Georgia | 400 | 卡片摘要 |
| Caption | `--font-caption` | Inter | 400-600 | UI 控件、元信息 |
| Label | `--font-label` | Inter | 500-600 | 全大写标签 |

规则：标题用 `font-display`，正文用 `font-serif`，UI 用 `font-sans`。

#### 间距 / 动效 / 几何 token

| Token | Value | Typical |
|-------|-------|---------|
| `--space-1..8` | 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 px | 行内→页面级 |
| `--space-page` | `clamp(16, 2.5vw+4, 40)px` | 页面边距 |
| `--transition-color` | `0.2s ease` | 颜色 hover |
| `--transition-all` | `0.25s cubic-bezier(0.4,0,0.2,1)` | 位移/尺寸 |
| `--duration-slow` | `0.25s` | 下划线展开 |
| `--radius-sm/md/pill` | 3 / 4 / 10 px | 圆角 |
| `--z-sticky` | 10 | dropdown |
| `--z-overlay` | 1000 | search overlay |

### 数据流

```
tokens.css  --:root + .dark--->  全部 CSS 文件 + Astro <style>
                  │
                  └─ 消费方通过 var(--token-name) 引用
```

| 来源 | 目标 | 数据格式 | 类型 | 说明 |
|------|------|---------|------|------|
| `tokens.css` | `base/layout/masthead/dropdown/search/components.css` | CSS custom properties | 强 | 缺失视觉崩坏 |
| `tokens.css` | `pages/*.astro` 内联 `<style>` | 同上 | 强 | 同上 |
| `tokens.css` | `components/*.astro` scoped style | 同上 | 强 | 同上 |

## 3. 设计决策

### 驱动因素

| 因素 | 类型 | 对架构的影响 |
|------|------|------------|
| 报纸/杂志风格定位 | 业务需求 | 颜色取暖色 paper；衬线 display + Inter UI；克制装饰 |
| Light + Dark 双主题 | 业务需求 | 所有颜色 token 必须双套；消费方仅引语义名（如 `--ink`） |
| Tailwind + Astro scoped 混用 | 技术约束 | token 用原生 CSS 变量，两种系统都能访问 |
| 移动 → 桌面 流式响应 | 质量要求 | 字号 `clamp(min, fluid, max)`，免到处写 media query |

### 关键选择

| 决策 | 选择 | 被拒方案 | 为什么 |
|------|------|---------|--------|
| token 载体 | 原生 CSS custom properties | Tailwind theme.extend / SCSS variables | 原生支持运行时主题切换；Astro scoped 与 global 都能用 |
| 颜色命名 | 语义层（`--ink-muted`） | 物理层（`--gray-500`） | 暗黑翻转语义稳定，消费方零修改 |
| 字号策略 | 8 档语义 + clamp 响应 | 固定 px / 4 档简化 | 8 档覆盖 masthead → label；clamp 一次定义跨断点 |

### 约束

- **禁止**消费方硬编码颜色 hex / 间距 px（破坏主题切换 + 设计漂移）
- **禁止**新增颜色 token 不填 dark 值（暗黑模式断裂）
- **必须**通过 `var(--name)` 引用，不复制值
- **必须**新增 token 同步更新本文档"组件表"

## 4. 质量要求

| 属性 | 指标 | 目标 |
|------|------|------|
| 主题完整性 | 颜色 token 双套覆盖率 | 100%（12/12 当前） |
| 引用一致性 | 消费方硬编码 hex / px 出现次数 | 0（grep 强制） |
| 响应式覆盖 | 字号 token 用 clamp 比例 | 100%（7/7 当前） |
| 暗黑切换性能 | 主题 toggle 到视觉稳定 | <50ms（目标值，待验证） |
