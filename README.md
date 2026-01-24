# 技术笔记网站

基于 Astro 构建的技术笔记静态网站，支持完整的元字段、标签分级索引、版本管理等功能。

## 功能特性

- 📝 完整的 Frontmatter 元字段支持
- 🏷️ 标签分级和别名支持
- 📊 质量评分和分析数据
- 🔍 搜索索引生成（预留客户端搜索接口）
- 📱 响应式设计，移动端优先
- 🚀 静态生成，零运行时 JavaScript
- 🔐 构建时数据校验，确保数据质量

## 技术栈

- **框架**: Astro 4.x
- **样式**: Tailwind CSS
- **类型**: TypeScript
- **部署**: GitHub Pages

## 开发指南

### 安装依赖

```bash
pnpm install
# 或
npm install
```

### 本地开发

```bash
pnpm dev
# 或
npm run dev
```

访问 http://localhost:4321

### 构建

```bash
pnpm build
# 或
npm run build
```

构建输出在 `dist/` 目录。

### 预览构建结果

```bash
pnpm preview
# 或
npm run preview
```

## Frontmatter Schema

每篇笔记必须包含以下 Frontmatter 字段：

```yaml
title: ""                     # 标题
slug: ""                      # 唯一标识
createdAt: ""                 # 创建时间 (ISO 8601 UTC)
updatedAt: ""                 # 最近更新时间 (ISO 8601 UTC)
date: ""                      # 发布日期 (ISO 8601 UTC)
question:
  type: ""                     # 主问题类型
  subType: ""                  # 子问题类型，可选
quality:
  overall: 0                   # 总评分 0–10
  coverage: 0                   # 覆盖度评分
  depth: 0                      # 深度评分
  specificity: 0                # 具体性评分
  reviewer: ""                  # ai | human | hybrid
analysis:
  objectivity:
    factRatio: 0.0
    inferenceRatio: 0.0
    opinionRatio: 0.0
  assumptions: []               # 假设条件
  limitations: []               # 局限性
review:
  status: draft                 # draft | reviewed | deprecated
  reviewedAt: ""                # 审核时间
tags: []                        # 标签（支持层级和别名）
topics: []                      # 主题（支持别名）
version:                        # 可选，版本管理
  current: 1
  history: []
```

### 标签层级示例

```yaml
tags:
  - name: "frontend"
  - name: "react"
    parent: "frontend"
  - name: "astro"
    parent: "frontend"
    alias: ["astrojs"]
```

### 版本历史示例

```yaml
version:
  current: 2
  history:
    - version: 1
      updatedAt: "2026-01-24T10:00:00Z"
      updatedBy: "human"
      changes:
        - "初始版本"
      changesSummary: "创建了初始版本"
```

## 数据校验

构建时会自动校验：

- slug 唯一性
- objectivity 比例和（factRatio + inferenceRatio + opinionRatio === 1.0）
- 日期格式（ISO 8601 UTC）
- 所有必填字段

## 部署

项目配置了 GitHub Actions 工作流，推送到 `main` 分支后自动构建并部署到 GitHub Pages。

## 扩展指南

### 添加新字段

1. 在 `src/utils/schema.ts` 中添加字段定义（使用 `.optional()` 或 `.default()`）
2. 在组件中处理新字段
3. 逐步迁移为必填字段（如需要）

### 添加新功能

- 所有组件独立、可复用
- 统一的 props 接口
- 便于添加新功能而不破坏现有页面

## 目录结构

```
/
├── src/
│   ├── content/notes/        # Markdown 笔记文件
│   ├── layouts/              # 布局组件
│   ├── components/           # 通用组件
│   ├── pages/                # 页面路由
│   ├── utils/                # 工具函数
│   └── styles/               # 全局样式
├── public/                    # 静态资源
└── astro.config.mjs          # Astro 配置
```

## 许可证

MIT
