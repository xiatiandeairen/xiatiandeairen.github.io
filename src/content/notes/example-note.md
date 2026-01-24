---
title: "示例笔记：Astro 技术笔记系统"
slug: "example-note"
createdAt: "2026-01-24T10:00:00Z"
updatedAt: "2026-01-24T10:00:00Z"
date: "2026-01-24T10:00:00Z"
version:
  current: 1
  history:
    - version: 1
      updatedAt: "2026-01-24T10:00:00Z"
      updatedBy: "human"
      changes:
        - "初始版本创建"
      changesSummary: "创建了示例笔记，展示完整的 Frontmatter 结构"
question:
  type: "技术架构"
  subType: "静态网站生成"
quality:
  overall: 8
  coverage: 8
  depth: 7
  specificity: 9
  reviewer: "human"
analysis:
  objectivity:
    factRatio: 0.7
    inferenceRatio: 0.2
    opinionRatio: 0.1
  assumptions:
    - "使用 Astro 作为静态网站生成器"
    - "笔记内容以 Markdown 格式存储"
  limitations:
    - "当前仅支持静态生成，不支持动态内容"
review:
  status: "reviewed"
  reviewedAt: "2026-01-24T10:30:00Z"
tags:
  - name: "frontend"
  - name: "react"
    parent: "frontend"
  - name: "astro"
    parent: "frontend"
    alias: ["astrojs"]
topics:
  - name: "静态网站生成"
  - name: "技术笔记"
    alias: ["笔记系统"]
---

# 示例笔记

这是一个示例笔记，展示了完整的 Frontmatter 元字段结构。

## 功能特性

1. **完整的元字段支持**
   - 标题、slug、日期
   - 问题分类
   - 质量评分
   - 分析数据
   - 审核状态
   - 标签和主题

2. **标签层级支持**
   - 支持父子标签关系
   - 支持标签别名

3. **版本管理**
   - 版本历史记录
   - 变更摘要

## 代码示例

```typescript
const note = {
  title: "示例笔记",
  slug: "example-note",
  date: "2026-01-24T10:00:00Z"
};
```

## 总结

这个示例展示了技术笔记系统的完整功能，包括元字段、标签层级、版本管理等特性。
