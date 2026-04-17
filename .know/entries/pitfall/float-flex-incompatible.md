## CSS float 与 flexbox 垂直空间分配不兼容

float 文字环绕和 flexbox 高度自适应是 CSS 规范层面互斥的布局模型。float 属于 flow layout，flex 属于 flex formatting context，两者的高度计算模型不兼容。

### 表现

- `flex: 1` 可以撑高容器，但 float 内容不会自动填满撑高后的空间
- float 元素在 flex 容器中失去 float 行为（规范明确：float, clear, vertical-align 对 flex item 无效）

### 已验证的事实

- CSS Exclusions（W3C 设计来解决此问题的规范）已停滞多年，全球覆盖率 0.29%，仅旧版 IE/Edge 支持
- shape-inside（CSS Shapes Level 2）零浏览器实现
- NYT/Guardian/Time.com 均放弃 float 环绕，用 Grid 分区模拟报纸风格
- 嵌套上下文（flex 外壳 + flow-root 内核）能做到评论固定底部，但 float 区域内仍有垂直空白

### 可行方案

1. **Grid 分区**（推荐）— 图文各占 grid area，零 hack
2. **JS 辅助** — ResizeObserver 动态设高度，但有 SSG 首帧闪烁问题
3. **长文本 + overflow:hidden** — 文字故意给多裁切，视觉欺骗

### Why

遇到"图片环绕 + 自适应高度 + 底部固定"需求时，必须在环绕和自适应之间二选一，不要尝试用 CSS 同时满足。
