---
title: "完整元字段示例：展示所有功能特性"
slug: "complete-metadata-example-showcase"
createdAt: "2026-01-12T10:00:00Z"
updatedAt: "2026-01-24T16:00:00Z"
date: "2026-01-12T10:00:00Z"
version:
  current: 2
  history:
    - version: 1
      updatedAt: "2026-01-12T10:00:00Z"
      updatedBy: "human"
      changes:
        - "创建初始版本"
        - "添加基础内容"
      changesSummary: "初始版本，包含基础功能说明"
    - version: 2
      updatedAt: "2026-01-24T16:00:00Z"
      updatedBy: "hybrid"
      changes:
        - "添加完整元字段示例"
        - "补充标签别名功能"
        - "完善分析数据"
      changesSummary: "扩展为完整示例，展示所有元字段功能"
question:
  type: "系统文档"
  subType: "功能展示"
quality:
  overall: 8
  coverage: 9
  depth: 8
  specificity: 7
  reviewer: "hybrid"
analysis:
  objectivity:
    factRatio: 0.6
    inferenceRatio: 0.3
    opinionRatio: 0.1
  assumptions:
    - "用户熟悉 Markdown 格式"
    - "了解 Frontmatter 结构"
    - "需要完整的功能参考"
  limitations:
    - "示例数据，非实际应用场景"
    - "部分字段为演示目的"
review:
  status: "reviewed"
  reviewedAt: "2026-01-24T16:30:00Z"
tags:
  - name: "documentation"
  - name: "example"
    parent: "documentation"
  - name: "metadata"
    parent: "example"
    alias: ["元数据", "frontmatter"]
  - name: "showcase"
    parent: "example"
topics:
  - name: "系统文档"
    alias: ["文档", "docs"]
  - name: "功能展示"
---

# 完整元字段示例

本文档展示了技术笔记系统的所有元字段功能，包括版本管理、标签层级、质量评分等。

## 元字段结构

### 基础字段

- **title**: 笔记标题
- **slug**: 唯一标识符
- **date**: 发布日期
- **createdAt/updatedAt**: 创建和更新时间

### 问题分类

```yaml
question:
  type: "系统文档"
  subType: "功能展示"
```

### 质量评分

质量评分系统包含多个维度：

- **overall**: 总体评分 (0-10)
- **coverage**: 覆盖度评分
- **depth**: 深度评分
- **specificity**: 具体性评分
- **reviewer**: 评审者类型 (ai/human/hybrid)

### 分析数据

#### 客观性分析

客观性分析包含三个比例，总和必须为 1.0：

- **factRatio**: 事实比例
- **inferenceRatio**: 推理比例
- **opinionRatio**: 观点比例

#### 假设和局限性

- **assumptions**: 假设条件列表
- **limitations**: 局限性说明

### 审核状态

```yaml
review:
  status: "reviewed"  # draft | reviewed | deprecated
  reviewedAt: "2026-01-24T16:30:00Z"
```

### 版本管理

版本系统支持：

- **current**: 当前版本号
- **history**: 版本历史记录
  - 版本号
  - 更新时间
  - 更新者
  - 变更列表
  - 变更摘要

### 标签系统

标签支持层级关系和别名：

```yaml
tags:
  - name: "documentation"
  - name: "example"
    parent: "documentation"
  - name: "metadata"
    parent: "example"
    alias: ["元数据", "frontmatter"]
```

### 主题系统

主题也支持别名：

```yaml
topics:
  - name: "系统文档"
    alias: ["文档", "docs"]
```

## 使用场景

### Public 模式

在 Public 模式下，只显示基本信息、标签、主题和质量评分。

### Audit 模式

在 Audit 模式下，显示所有元字段，包括：

- 完整的版本历史
- 详细的分析数据
- 所有假设和局限性
- 完整的审核信息

## 总结

这个示例展示了技术笔记系统的完整功能，包括元字段管理、版本控制、质量评估等特性。
