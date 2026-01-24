# 目录规范 (Directory Structure)

## 设计原则

1. **清晰分层**：按功能模块组织，职责明确
2. **可扩展性**：支持 3-5 年内容增长（可能达到数千条笔记）
3. **易于维护**：新成员能快速理解项目结构
4. **避免冗余**：移除无用目录，保持结构精简

## 标准目录结构

```
/
├── .cursor/
│   └── rules/                    # Cursor 规则文件
│       ├── commit-convention.md
│       ├── directory-structure.md
│       └── coding-standards.md
├── .github/
│   └── workflows/                # GitHub Actions 工作流
│       └── pages.yml
├── docs/                         # 项目文档（未来扩展）
│   ├── incremental-build.md
│   ├── build-cache.md
│   └── performance-testing.md
├── public/                       # 静态资源
│   └── search-index.json         # 搜索索引（构建时生成）
├── src/
│   ├── content/
│   │   └── notes/                # Markdown 笔记文件
│   │       ├── 01-*.md
│   │       ├── 02-*.md
│   │       └── ...
│   ├── components/               # 可复用组件
│   │   ├── MetaFields.astro
│   │   ├── NoteCard.astro
│   │   ├── Pagination.astro
│   │   ├── TagHierarchy.astro
│   │   └── VersionDiff.astro
│   ├── integrations/             # Astro 集成
│   │   └── generate-search-index.ts
│   ├── layouts/                  # 页面布局
│   │   ├── BaseLayout.astro
│   │   ├── IndexLayout.astro
│   │   └── NoteLayout.astro
│   ├── pages/                    # 路由页面
│   │   ├── index.astro
│   │   ├── search.astro
│   │   ├── notes/
│   │   │   └── [slug].astro
│   │   ├── page/
│   │   │   └── [page].astro
│   │   ├── tags/
│   │   │   └── [tag].astro
│   │   └── topics/
│   │       └── [topic].astro
│   ├── styles/                   # 全局样式
│   │   └── global.css
│   ├── types/                    # TypeScript 类型定义
│   │   └── note.ts
│   ├── utils/                    # 工具函数
│   │   ├── constants.ts
│   │   ├── notes.ts
│   │   ├── schema.ts
│   │   ├── search.ts
│   │   ├── seo.ts
│   │   └── tags.ts
│   └── env.d.ts                  # 环境类型定义
├── astro.config.mjs              # Astro 配置
├── tailwind.config.mjs           # Tailwind CSS 配置
├── tsconfig.json                 # TypeScript 配置
├── package.json                  # 依赖管理
├── pnpm-lock.yaml               # 锁文件
├── .gitignore                    # Git 忽略规则
└── README.md                     # 项目说明
```

## 目录说明

### `/src/content/notes/`

**用途**：存储所有 Markdown 笔记文件

**命名规范**：
- 使用 kebab-case：`react-hooks-guide.md`
- 可添加数字前缀便于排序：`01-basic-note.md`
- 文件名应与 slug 保持一致（推荐）

**扩展性**：
- 支持子目录分类（未来）：`notes/frontend/`, `notes/backend/`
- 支持按年份组织（大规模时）：`notes/2026/`, `notes/2027/`

### `/src/components/`

**用途**：可复用的 Astro 组件

**组织原则**：
- 按功能分组（如需要）：`components/notes/`, `components/navigation/`
- 保持扁平结构，避免过度嵌套
- 组件命名使用 PascalCase

**扩展性**：
- 未来可添加：`components/forms/`, `components/charts/` 等

### `/src/layouts/`

**用途**：页面布局模板

**原则**：
- 保持精简，只包含必要的布局
- 布局之间可以继承组合

### `/src/pages/`

**用途**：Astro 路由页面

**组织原则**：
- 使用 Astro 文件系统路由
- 动态路由使用 `[param]` 格式
- 保持路由结构清晰

**扩展性**：
- 未来可添加：`pages/admin/`, `pages/api/` 等

### `/src/utils/`

**用途**：纯函数工具库

**组织原则**：
- 按功能模块划分文件
- 保持函数纯净（无副作用）
- 便于单元测试

**扩展性**：
- 未来可添加：`utils/validation/`, `utils/formatting/` 等子目录

### `/src/types/`

**用途**：TypeScript 类型定义

**原则**：
- 集中管理类型定义
- 避免类型分散在各文件中

### `/docs/`

**用途**：项目文档（未来扩展）

**内容**：
- 增量构建文档
- 构建缓存策略
- 性能测试指南
- API 文档

## 禁止的目录结构

以下目录不应存在：

- `themes/` - Hexo 主题目录（已移除）
- `scaffolds/` - Hexo 模板目录（已移除）
- `source/` - Hexo 源文件目录（已移除）
- `node_modules/` - 依赖目录（应在 .gitignore）
- `dist/` - 构建输出（应在 .gitignore）
- `.astro/` - Astro 缓存（应在 .gitignore）

## 未来扩展规划

### 3-5 年演进路径

#### 阶段 1：当前（0-1000 条笔记）
- 扁平结构，所有笔记在 `src/content/notes/`
- 简单组件组织

#### 阶段 2：中等规模（1000-5000 条笔记）
- 按年份组织：`notes/2026/`, `notes/2027/`
- 或按主题组织：`notes/frontend/`, `notes/backend/`
- 组件按功能分组

#### 阶段 3：大规模（5000+ 条笔记）
- 多级分类：`notes/2026/frontend/react/`
- 组件库化：`components/ui/`, `components/features/`
- 工具函数模块化：`utils/notes/`, `utils/tags/`, `utils/search/`

## 文件命名规范

### Markdown 文件
- 使用 kebab-case：`react-hooks-guide.md`
- 与 slug 保持一致

### TypeScript/JavaScript 文件
- 使用 camelCase：`notes.ts`, `schema.ts`
- 组件文件使用 PascalCase：`NoteCard.astro`

### 配置文件
- 使用 kebab-case：`astro.config.mjs`, `tailwind.config.mjs`

## 目录维护

### 定期检查
- 每季度检查是否有无用目录
- 清理临时文件和测试文件
- 确保 .gitignore 正确配置

### 重构原则
- 重构前先评估影响范围
- 保持向后兼容（如可能）
- 更新相关文档

## 迁移指南

当需要调整目录结构时：

1. 创建迁移计划文档
2. 更新所有导入路径
3. 更新构建配置
4. 测试验证
5. 更新文档
