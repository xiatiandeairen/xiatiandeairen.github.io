# icon-button 上用 overflow:hidden+max-width 会裁掉 SVG

## Symptoms
搜索图标在非 hover 态完全消失。按钮变成空白方块。

## Root cause
在 `.icon-button` 自身设置 `width: auto; max-width: 28px; overflow: hidden`，
配合 `gap` 和子元素 `.icon-label`，导致 flex 布局压缩 SVG 的可用空间为 0。

## Lesson
展开/收缩动画应作用于子元素（.icon-label），不是容器本身。
- 容器 `.icon-button` 保持固定 `width: 28px`，hover 时 `width: auto`
- 子元素 `.icon-label` 用 `max-width: 0 → 50px` + `opacity: 0 → 1` 实现展开
- `overflow: hidden` 放在 `.icon-label` 上，不放在按钮上
