# 响应式字号用 clamp(rem+vw)，media query 只管布局切换

## Symptoms
用 @media (max-width: 767px) + !important 覆盖固定字号，768px 附近跳变，
375px 和 768px 体验差异大，且修改不生效需要回退。

## Root cause
固定断点的字号覆盖是离散的，无法适配连续的视口宽度变化。
纯 vw 单位不响应用户缩放，不符合 WCAG。

## Lesson
- 所有字号定义为 CSS 自定义属性 --font-*: clamp(mobile, rem+vw, desktop)
- preferred 值必须包含 rem（保证缩放响应）
- 间距同理: --space-page: clamp(16px, 2.5vw+4px, 40px)
- media query 只做布局切换（双栏→单栏），不碰字号
