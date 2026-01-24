---
title: "快速笔记：CSS 变量使用"
slug: "quick-note-css-variables"
createdAt: "2026-01-21T15:00:00Z"
updatedAt: "2026-01-21T15:00:00Z"
date: "2026-01-21T15:00:00Z"
question:
  type: "前端开发"
  subType: "CSS"
quality:
  overall: 4
  coverage: 3
  depth: 4
  specificity: 5
  reviewer: "ai"
analysis:
  objectivity:
    factRatio: 0.8
    inferenceRatio: 0.15
    opinionRatio: 0.05
  assumptions:
    - "读者了解 CSS 基础"
  limitations:
    - "内容较简略"
    - "缺少实际案例"
    - "未涵盖所有用法"
review:
  status: "reviewed"
  reviewedAt: "2026-01-21T15:30:00Z"
tags:
  - name: "frontend"
  - name: "css"
    parent: "frontend"
topics:
  - name: "前端开发"
---

# CSS 变量使用

CSS 变量（自定义属性）允许在 CSS 中定义可重用的值。

## 定义变量

```css
:root {
  --primary-color: #0066cc;
  --spacing: 1rem;
}
```

## 使用变量

```css
.button {
  background-color: var(--primary-color);
  padding: var(--spacing);
}
```

## 总结

CSS 变量提供了更灵活的样式管理方式。
