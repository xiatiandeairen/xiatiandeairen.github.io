# 项目结构文档

## 目录结构

```
/
├── .cursor/                    # Cursor IDE 配置
│   ├── plans/                 # 项目计划
│   ├── rules/                 # 项目规则
│   └── skills/                # Cursor 技能
├── .github/                    # GitHub 配置
│   └── workflows/             # GitHub Actions 工作流
├── docs/                       # 项目文档
│   ├── project-structure.md   # 项目结构（本文档）
│   ├── module-relationships.md # 模块调用关系
│   └── core-logic.md          # 重要逻辑说明
├── public/                     # 静态资源
│   └── search-index.json      # 搜索索引（构建时生成）
├── src/
│   ├── content/               # 内容文件
│   │   └── notes/             # Markdown 笔记文件
│   ├── components/            # 可复用组件
│   │   ├── MetaFields.astro  # 元字段显示组件
│   │   ├── NoteCard.astro    # 笔记卡片组件
│   │   ├── Pagination.astro  # 分页组件
│   │   ├── TagHierarchy.astro # 标签层级组件
│   │   └── VersionDiff.astro # 版本差异组件
│   ├── integrations/         # Astro 集成
│   │   └── generate-search-index.ts # 搜索索引生成集成
│   ├── layouts/               # 页面布局
│   │   ├── BaseLayout.astro  # 基础布局
│   │   ├── IndexLayout.astro # 索引页布局
│   │   └── NoteLayout.astro  # 笔记详情页布局
│   ├── pages/                 # 路由页面
│   │   ├── index.astro        # 首页
│   │   ├── search.astro       # 搜索页
│   │   ├── notes/             # 笔记路由
│   │   │   └── [slug].astro   # 笔记详情页
│   │   ├── page/              # 分页路由
│   │   │   └── [page].astro   # 分页页
│   │   ├── tags/              # 标签路由
│   │   │   ├── index.astro    # 标签索引页
│   │   │   └── [tag].astro    # 标签详情页
│   │   └── topics/            # 主题路由
│   │       ├── index.astro    # 主题索引页
│   │       └── [topic].astro  # 主题详情页
│   ├── styles/                # 全局样式
│   │   └── global.css         # 全局 CSS
│   ├── types/                 # TypeScript 类型定义
│   │   └── note.ts            # 笔记相关类型
│   └── utils/                 # 工具函数
│       ├── constants.ts       # 常量定义
│       ├── notes.ts           # 笔记操作函数
│       ├── schema.ts          # 数据验证 Schema
│       ├── search.ts          # 搜索索引生成
│       ├── seo.ts             # SEO 元数据生成
│       └── tags.ts            # 标签操作函数
├── astro.config.mjs           # Astro 配置
├── tailwind.config.mjs        # Tailwind CSS 配置
├── tsconfig.json              # TypeScript 配置
└── package.json               # 项目依赖
```

## 模块说明

### 核心模块

#### `src/types/note.ts`
- **职责**：定义所有笔记相关的 TypeScript 类型
- **导出**：`NoteFrontmatter`, `Note`, `Tag`, `Topic`, `Version` 等接口
- **依赖**：无

#### `src/utils/schema.ts`
- **职责**：使用 Zod 定义和验证 Frontmatter 数据结构
- **核心功能**：
  - `NoteFrontmatterSchema`: 完整的 Frontmatter 验证规则
  - `validateNoteFrontmatter()`: 验证单个笔记的 Frontmatter
  - `validateSlugUniqueness()`: 验证所有笔记的 slug 唯一性
- **依赖**：`zod`

#### `src/utils/notes.ts`
- **职责**：笔记文件的读取、过滤、排序、分页操作
- **核心函数**：
  - `getAllNotes()`: 读取所有笔记文件并验证
  - `getNotesByTag()`: 按标签过滤笔记
  - `getNotesByTopic()`: 按主题过滤笔记
  - `sortNotes()`: 排序笔记
  - `paginateNotes()`: 分页处理
- **依赖**：`schema.ts`, `gray-matter`, `fs`

#### `src/utils/tags.ts`
- **职责**：标签层级结构构建和操作
- **核心函数**：
  - `buildTagHierarchy()`: 构建标签层级树
  - `getTagPath()`: 获取标签完整路径
  - `getChildTags()`: 获取子标签
  - `resolveTagAlias()`: 解析标签别名
- **依赖**：`notes.ts`

#### `src/utils/search.ts`
- **职责**：搜索索引生成
- **核心函数**：
  - `generateSearchIndex()`: 生成搜索索引对象
  - `writeSearchIndex()`: 写入搜索索引文件
  - `extractTextFromMarkdown()`: 从 Markdown 提取纯文本
- **依赖**：`notes.ts`

#### `src/utils/seo.ts`
- **职责**：SEO 元数据生成
- **核心函数**：
  - `generateSEOMeta()`: 为笔记生成 SEO 元数据
  - `generatePageMeta()`: 为页面生成 SEO 元数据
- **依赖**：`constants.ts`

### 组件模块

#### `src/components/`
所有组件都是 Astro 组件，接收 props 并渲染 HTML。

- **MetaFields.astro**: 显示笔记元字段（支持 public/audit 模式）
- **NoteCard.astro**: 笔记卡片展示
- **Pagination.astro**: 分页导航
- **TagHierarchy.astro**: 标签层级面包屑
- **VersionDiff.astro**: 版本历史显示

### 布局模块

#### `src/layouts/`
- **BaseLayout.astro**: 基础 HTML 结构、SEO 标签、导航、页脚
- **IndexLayout.astro**: 索引页布局（列表页）
- **NoteLayout.astro**: 笔记详情页布局

### 页面模块

#### `src/pages/`
所有页面使用 Astro 的文件系统路由。

- **index.astro**: 首页，显示所有笔记（分页）
- **search.astro**: 搜索页，客户端搜索功能
- **notes/[slug].astro**: 笔记详情页，动态路由
- **page/[page].astro**: 分页页，动态路由
- **tags/index.astro**: 标签索引页
- **tags/[tag].astro**: 标签详情页，动态路由
- **topics/index.astro**: 主题索引页
- **topics/[topic].astro**: 主题详情页，动态路由

### 集成模块

#### `src/integrations/generate-search-index.ts`
- **职责**：Astro 集成，在构建时和开发时生成搜索索引
- **Hook**：
  - `astro:server:setup`: 开发服务器启动时生成索引
  - `astro:build:done`: 构建完成后生成索引到 dist 目录
- **依赖**：`schema.ts`, `search.ts`, `notes.ts`

## 数据流

### 构建时数据流

```
Markdown 文件 (src/content/notes/*.md)
  ↓
gray-matter 解析 Frontmatter
  ↓
schema.ts 验证数据
  ↓
notes.ts 读取并处理
  ↓
pages/*.astro 使用 getStaticPaths 生成静态页面
  ↓
search.ts 生成搜索索引
  ↓
dist/ 输出目录
```

### 运行时数据流（搜索功能）

```
用户输入搜索词
  ↓
fetch('/search-index.json')
  ↓
客户端 JavaScript 过滤匹配
  ↓
动态渲染搜索结果
```

## 文件命名规范

- **组件文件**: PascalCase (如 `NoteCard.astro`)
- **工具文件**: camelCase (如 `notes.ts`)
- **类型文件**: camelCase (如 `note.ts`)
- **配置文件**: kebab-case (如 `astro.config.mjs`)
- **Markdown 文件**: kebab-case (如 `01-basic-note.md`)

## 扩展性设计

### 当前阶段（0-1000 条笔记）
- 扁平结构，所有笔记在 `src/content/notes/`
- 简单组件组织

### 未来扩展（1000+ 条笔记）
- 按年份组织：`notes/2026/`, `notes/2027/`
- 或按主题组织：`notes/frontend/`, `notes/backend/`
- 组件按功能分组：`components/ui/`, `components/features/`
- 工具函数模块化：`utils/notes/`, `utils/tags/`, `utils/search/`
