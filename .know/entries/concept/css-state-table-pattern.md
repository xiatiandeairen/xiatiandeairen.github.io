## CSS 多状态组件状态表注释规范

每个多状态组件的 CSS 块顶部必须有标准格式的状态表注释。

### 格式

```css
/* ── {Component} ──
   States:
     {state-name}        {trigger}         → {visual changes}
   Timing: {property} {duration} {easing}
   Platform: {hover: hover | all}
   a11y: {reduced-motion | focus-visible | aria}
*/
```

### 规则

- 状态命名用 `:` 分隔层级（如 `active:hover`），最多 2 级
- trigger 列写触发条件（如 `mouse enter`），不写选择器
- visual 列只写变化量（如 `color → ink, bg → rule-faint`）
- 复合状态（A+B 视觉 ≠ A + B）必须单独列出
- Timing/Platform/a11y 仅在有非默认行为时才写

### 已应用的组件

- icon-button (3 states)
- expandable icon-button (2 states)
- dropdown-item (4 states)
- sidebar-drawer (3 states)

### Why

状态是组件的接口文档。没有显式状态表时，状态散落在选择器中，新人容易漏改、AI 需要推理才能理解。
