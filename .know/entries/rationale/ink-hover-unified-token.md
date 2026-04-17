## --ink-hover 统一交互色 token

### 决策

新增 `--ink-hover` token（亮色 `#5a7a6a` 墨绿 / 暗色 `#8abaa5`），作为全站所有交互态的统一颜色。替代原来各处使用的 `--accent`（`#8b0000` 暗红）。

### 覆盖范围

- 全局 `a:hover`（base.css）
- masthead brand hover + 下划线（masthead.css）
- 导航链接下划线（masthead.css）
- 所有 ArticleCard 变体标题 hover（components.css）
- 左侧 hover indicator 线条颜色（components.css）
- dropdown active 指示（dropdown.css）
- 评论区左边框（components.css）

### --accent 现在只用于

`reading-progress` 进度条。不再用于任何 hover/交互态。

### Why

`--accent`（暗红）作为 hover 色与报纸暖色调冲突，且红色在交互语义中容易被理解为"错误/危险"。墨绿色（`#5a7a6a`）低饱和、安静克制，与报纸风格一致。统一为一个 token 避免各处颜色不一致。
