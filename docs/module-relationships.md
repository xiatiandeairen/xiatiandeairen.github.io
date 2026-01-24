# 模块调用关系文档

## 模块依赖图

```
┌─────────────────┐
│   types/note.ts │ (基础类型定义，无依赖)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  utils/schema.ts│ (依赖: zod)
└────────┬────────┘
         │
         ├──────────────────┐
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│  utils/notes.ts │  │ integrations/   │
│                 │  │ generate-search- │
│ (依赖: schema,  │  │ index.ts        │
│  gray-matter)   │  │                 │
└────────┬────────┘  │ (依赖: schema,   │
         │           │  search, notes)  │
         │           └──────────────────┘
         │
         ├──────────────────┐
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│  utils/tags.ts  │  │ utils/search.ts │
│                 │  │                 │
│ (依赖: notes)   │  │ (依赖: notes)   │
└────────┬────────┘  └─────────────────┘
         │
         │
┌────────┴────────┐
│  utils/seo.ts   │
│                 │
│ (依赖: constants)│
└─────────────────┘
```

## 详细调用关系

### 1. 类型定义层

#### `src/types/note.ts`
- **被依赖**：所有其他模块
- **依赖**：无
- **导出接口**：
  - `NoteFrontmatter`: Frontmatter 结构
  - `Note`: 完整笔记（包含 content）
  - `Tag`, `Topic`, `Version` 等

### 2. 数据验证层

#### `src/utils/schema.ts`
- **依赖**：
  - `zod`: 数据验证库
- **被依赖**：
  - `utils/notes.ts`: 验证 Frontmatter
  - `integrations/generate-search-index.ts`: 验证笔记数据
- **核心导出**：
  ```typescript
  validateNoteFrontmatter(data: unknown): NoteFrontmatter
  validateSlugUniqueness(notes: Array<{slug: string}>): void
  ```

### 3. 数据操作层

#### `src/utils/notes.ts`
- **依赖**：
  - `schema.ts`: 数据验证
  - `gray-matter`: Frontmatter 解析
  - `fs`: 文件系统操作
- **被依赖**：
  - `utils/tags.ts`: 获取笔记列表
  - `utils/search.ts`: 生成搜索索引
  - `pages/*.astro`: 所有页面使用
  - `integrations/generate-search-index.ts`: 构建时生成索引
- **核心导出**：
  ```typescript
  getAllNotes(): Note[]
  getNotesByTag(tagName: string, notes: Note[]): Note[]
  getNotesByTopic(topicName: string, notes: Note[]): Note[]
  sortNotes(notes: Note[], sortBy: SortField, order: SortOrder): Note[]
  paginateNotes(notes: Note[], page: number, perPage: number): PaginatedResult
  ```

#### `src/utils/tags.ts`
- **依赖**：
  - `notes.ts`: 获取笔记数据
  - `constants.ts`: 标签最大深度常量
- **被依赖**：
  - `pages/tags/*.astro`: 标签页面
  - `components/TagHierarchy.astro`: 标签层级显示
- **核心导出**：
  ```typescript
  buildTagHierarchy(notes: Note[]): Map<string, TagNode>
  getTagPath(tagName: string, notes: Note[]): string
  getChildTags(parentTag: string, notes: Note[]): string[]
  resolveTagAlias(tagName: string, notes: Note[]): string
  resolveTopicAlias(topicName: string, notes: Note[]): string
  ```

#### `src/utils/search.ts`
- **依赖**：
  - `notes.ts`: 获取笔记数据
  - `fs`: 文件写入
- **被依赖**：
  - `integrations/generate-search-index.ts`: 构建时调用
- **核心导出**：
  ```typescript
  generateSearchIndex(notes: Note[]): SearchIndex
  writeSearchIndex(notes: Note[], outputPath: string): void
  ```

#### `src/utils/seo.ts`
- **依赖**：
  - `constants.ts`: 站点配置
  - `types/note.ts`: 笔记类型
- **被依赖**：
  - `layouts/*.astro`: 所有布局使用
  - `pages/*.astro`: 所有页面使用
- **核心导出**：
  ```typescript
  generateSEOMeta(note: Note): SEOData
  generatePageMeta(page: PageInfo): SEOData
  ```

### 4. 集成层

#### `src/integrations/generate-search-index.ts`
- **依赖**：
  - `schema.ts`: 验证笔记数据
  - `search.ts`: 生成搜索索引
  - `notes.ts`: 读取笔记（内部实现，避免循环依赖）
- **被依赖**：
  - `astro.config.mjs`: 注册集成
- **Hook 调用**：
  - `astro:server:setup`: 开发时生成 `public/search-index.json`
  - `astro:build:done`: 构建时生成 `dist/search-index.json`

### 5. 组件层

#### `src/components/*.astro`
- **依赖**：
  - `types/note.ts`: 类型定义
  - `utils/tags.ts`: 标签操作（TagHierarchy）
  - `utils/seo.ts`: SEO 数据（部分组件）
- **被依赖**：
  - `layouts/*.astro`: 布局使用组件
  - `pages/*.astro`: 页面直接使用组件

### 6. 布局层

#### `src/layouts/*.astro`
- **依赖**：
  - `components/*.astro`: 使用组件
  - `utils/seo.ts`: SEO 元数据
  - `styles/global.css`: 全局样式
- **被依赖**：
  - `pages/*.astro`: 所有页面使用布局

### 7. 页面层

#### `src/pages/*.astro`
- **依赖**：
  - `layouts/*.astro`: 使用布局
  - `components/*.astro`: 使用组件
  - `utils/notes.ts`: 获取笔记数据
  - `utils/tags.ts`: 标签操作
  - `utils/seo.ts`: SEO 元数据
- **被依赖**：无（顶层模块）

## 调用链示例

### 示例 1: 笔记详情页生成

```
pages/notes/[slug].astro
  ↓ getStaticPaths()
  ↓
utils/notes.ts::getAllNotes()
  ↓
utils/schema.ts::validateNoteFrontmatter()
  ↓
utils/schema.ts::validateSlugUniqueness()
  ↓
返回 Note[] 数组
  ↓
pages/notes/[slug].astro 生成静态路径
  ↓
layouts/NoteLayout.astro
  ↓
components/MetaFields.astro
```

### 示例 2: 搜索索引生成

```
astro.config.mjs
  ↓ 注册集成
integrations/generate-search-index.ts
  ↓ astro:build:done hook
读取 src/content/notes/*.md
  ↓
utils/schema.ts::validateNoteFrontmatter()
  ↓
utils/search.ts::generateSearchIndex()
  ↓
utils/search.ts::writeSearchIndex()
  ↓
生成 dist/search-index.json
```

### 示例 3: 标签页面生成

```
pages/tags/[tag].astro
  ↓ getStaticPaths()
  ↓
utils/notes.ts::getAllNotes()
  ↓
utils/tags.ts::buildTagHierarchy()
  ↓
utils/tags.ts::resolveTagAlias()
  ↓
utils/notes.ts::getNotesByTag()
  ↓
utils/notes.ts::sortNotes()
  ↓
utils/notes.ts::paginateNotes()
  ↓
生成静态页面
```

## 循环依赖预防

项目设计避免了循环依赖：

1. **类型层** (`types/`) 不依赖任何其他模块
2. **工具层** (`utils/`) 只依赖类型和外部库
3. **组件层** 依赖工具层和类型层
4. **页面层** 依赖所有下层模块

## 模块职责分离

| 模块 | 职责 | 依赖层级 |
|------|------|----------|
| `types/` | 类型定义 | L0 (基础层) |
| `utils/schema.ts` | 数据验证 | L1 |
| `utils/notes.ts` | 笔记操作 | L2 |
| `utils/tags.ts` | 标签操作 | L3 |
| `utils/search.ts` | 搜索索引 | L3 |
| `utils/seo.ts` | SEO 生成 | L2 |
| `components/` | UI 组件 | L4 |
| `layouts/` | 页面布局 | L5 |
| `pages/` | 路由页面 | L6 (顶层) |
| `integrations/` | 构建集成 | L2-L3 |

## 数据流向

### 构建时（静态生成）

```
文件系统 (Markdown)
  → utils/notes.ts (读取)
  → utils/schema.ts (验证)
  → pages/*.astro (生成 HTML)
  → dist/ (输出)
```

### 运行时（客户端搜索）

```
用户交互
  → fetch('/search-index.json')
  → 客户端 JavaScript 过滤
  → DOM 更新
```
