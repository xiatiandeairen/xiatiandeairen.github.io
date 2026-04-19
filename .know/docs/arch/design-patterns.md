# Design Patterns 架构设计

## 1. 定位与边界

### 职责

负责沉淀视觉/交互的复用配方（pattern）和组件状态规范，定义"如何把 token 组合成可识别的行为"，是组件实现的契约层。

### 不负责

- 颜色/间距/字号等原子变量（→ [design-tokens](design-tokens.md)）
- 共享 class 的命名/登记/废弃流程（→ [design-governance §2.3](design-governance.md#23-primitive-登记表)）
- 单个组件的业务逻辑/数据流（→ 各 component 文件 + tech 文档）

## 2. 结构与交互

### 组件图

```
[Patterns 层]                  [Components 层]            [Don'ts]
  hover-indicator      ─┐
  hover-shift          ─┤
  hover-underline      ─┼──组合──→  ArticleCard / NoteCard      校验
  text-hierarchy       ─┤            Masthead / Dropdown    ←── 反例规则
  card-divider         ─┤            Search / Sidebar
  section-break        ─┤            Pagination
  layout primitives    ─┘            Layout Primitives
```

### 组件表

| 组件 | 职责 | 边界规则 |
|------|------|---------|
| Patterns | 定义可命名复用的样式配方（清单见 §2 内表） | 必须 ≥3 处使用才能登记；禁止纯外观差异另立 pattern |
| Components | 描述每个组件的状态机和组合用的 pattern 集 | 状态用枚举（normal/hover/active/...）；禁止跨组件直接复用样式选择器 |
| Don'ts | 反模式清单（错误示范 + 正例对照） | 必须配反例；禁止"建议"等模糊语 |

#### Patterns 清单（7 项）

| Pattern | 构成 | 使用方 |
|---------|------|-------|
| hover-indicator | `::before` + `width:2px` + `--ink-hover` 左竖线 height/scaleY 动画 | article-card-headline/sidebar/secondary/list, post-card |
| hover-shift | hover 时 `padding-left 0→var(--space-1)` 偏移 | article-card-headline/list, archive-item, search-result-card |
| hover-underline | `::after` 底部 1px 线 width 0→100% 展开 | masthead-brand, masthead-nav |
| text-hierarchy | ink → ink-light → ink-muted → ink-faint → ink-ghost 5 级灰阶 | 全站文字 |
| card-divider | `border-bottom: 1px solid var(--rule-faint)` + `:last-child` 去线 | article-card-sidebar/list, post-card, comment-placeholder |
| section-break | heavy / labeled / light 三种区域分隔 | masthead-outer, section-label, layout-featured-secondary |
| layout primitives | layout-featured / layout-featured-secondary / layout-dual-* / site-main | 见 [design-governance §2.3](design-governance.md#23-primitive-登记表) |

#### Components 状态规范（8 个）

| 组件 | 关键状态 |
|------|---------|
| ArticleCard | 4 变体（headline/sidebar/secondary/list），共享 `article-card-title a:hover→--ink-hover`；组合 hover-indicator + hover-shift + card-divider |
| NoteCard (post-card) | normal: rule-faint border；hover: indicator 60% + shift + title→ink-hover |
| Masthead | brand/nav: hover-underline；icon-button: bg-subtle + rule；expandable: max-width 展开 + label 渐入（仅 hover:hover） |
| Dropdown | menu: 父级 hover→opacity 1 + translateY 0；item normal/hover/active 三态 |
| Search | overlay: closed/open 切换；result hover→bg-subtle |
| Sidebar (drawer) | closed: translateX(220) + 指示线；open:hover: translateX(0) + 指示线渐出（仅 hover:hover） |
| Pagination | link normal/hover/current 三态 |
| Layout Primitives | featured / featured-secondary / dual-wide / dual-archive / site-main |

#### Don'ts（反模式）

1. 不用 `--accent` 做 hover 色（仅用于 reading-progress；hover 用 `--ink-hover`）
2. 不硬编码颜色/间距数值（必须 `var(--token)`）
3. 不自定义 transition timing（用 `--transition-color` / `--transition-all`）
4. 不用 float 做需 flex 自适应的布局
5. 不在用户未要求时夹带额外样式变更

**例外**：≤2px 微调允许硬编码（token 最小 4px）；专用动画允许自定义 timing 但需 CSS 注释说明理由。

### 数据流

```
design-tokens  --var()-->  Patterns  --组合-->  Components  --应用到-->  Pages/Layouts
                              ↑
                  Rule of Three 验证后才能登记
```

| 来源 | 目标 | 数据格式 | 类型 | 说明 |
|------|------|---------|------|------|
| design-tokens | Patterns | CSS variable 引用 | 强 | pattern 不能硬编码值 |
| Patterns | Components | class 组合 + state 枚举 | 强 | 组件描述必须列出复用了哪些 pattern |
| Don'ts | Patterns + Components | 反例校验 | 弱 | PR review 拦截，无机器强制 |

## 3. 设计决策

### 驱动因素

| 因素 | 类型 | 对架构的影响 |
|------|------|------------|
| 视觉语言一致性 | 业务需求 | "hover 时左线展开"等配方命名沉淀，避免每组件重发明 |
| 组件状态可枚举 | 质量要求 | 用 normal/hover/active/open 状态机替代散落的 `:hover` 样式 |
| 暗示交互的克制风格 | 业务需求 | 选择 hover-indicator/hover-shift/hover-underline 三种轻量提示，不用大色块 |
| 防止过早抽象 | 技术约束 | Patterns 必须 ≥3 处使用才登记 |

### 关键选择

| 决策 | 选择 | 被拒方案 | 为什么 |
|------|------|---------|--------|
| Pattern 抽象层级 | 命名的配方（构成 + 动画 + token 引用） | "Pattern 即组件" | 配方更细粒度，组件可组合多 pattern；后者会重复发明 |
| 状态描述方式 | 枚举状态机 | 散落 `:hover` 选择器 | 可读性强 + 改变行为时一处变更 |
| Don'ts 形式 | 反例 + 正例对照 + 例外清单 | 无 Don'ts / 仅"建议"语气 | 强约束才能阻止旧反模式回归 |

### 约束

- **禁止**新 pattern 不经 Rule of Three 验证就登记（防御性抽象）
- **禁止**组件直接引用其他组件的样式选择器（破坏边界）
- **必须**新 pattern 同步更新本文档 §2 内表
- **必须**新组件标注"复用了哪些 pattern"

## 4. 质量要求

| 属性 | 指标 | 目标 |
|------|------|------|
| Pattern 复用度 | 每个 pattern 平均使用方数 | ≥3（Rule of Three 下限） |
| 组件 pattern 引用率 | 组件描述列出复用 pattern 比例 | 100%（8/8 当前） |
| Don'ts 覆盖 | 已知反模式数 | ≥5（当前 5 项 + 例外清单） |
| 反模式回归 | CI 检查 hex/timing 硬编码 | 0（lint:hardcoded 待加） |
