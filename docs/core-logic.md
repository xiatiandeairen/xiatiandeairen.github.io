# 重要逻辑说明文档

## 1. 数据验证逻辑

### 1.1 Frontmatter Schema 验证

**位置**: `src/utils/schema.ts`

**核心逻辑**:
```typescript
NoteFrontmatterSchema = z.object({
  // 必填字段
  title: z.string(),
  slug: z.string(),
  createdAt: z.string().regex(iso8601Regex),
  // ... 其他字段
}).passthrough() // 允许未来扩展字段
```

**验证规则**:
1. **日期格式**: 必须符合 ISO 8601 UTC 格式 (`YYYY-MM-DDTHH:mm:ssZ`)
2. **数值范围**: 
   - `quality.*`: 0-10
   - `objectivity.*`: 0-1
3. **比例验证**: `factRatio + inferenceRatio + opinionRatio === 1.0` (误差 < 0.01)
4. **枚举值**: `review.status`, `quality.reviewer`, `version.updatedBy` 等

**错误处理**:
- 验证失败时抛出 Zod 错误，包含详细字段信息
- 构建时失败会阻止构建完成

### 1.2 Slug 唯一性验证

**位置**: `src/utils/schema.ts::validateSlugUniqueness()`

**逻辑**:
```typescript
1. 遍历所有笔记，收集 slug → 文件路径映射
2. 检测重复的 slug
3. 如果发现重复，抛出错误，列出所有重复的文件
```

**目的**: 确保每个笔记有唯一的 URL 路径

## 2. 笔记读取逻辑

### 2.1 文件读取流程

**位置**: `src/utils/notes.ts::getAllNotes()`

**流程**:
```
1. 读取 src/content/notes/ 目录
2. 过滤 .md 文件
3. 对每个文件：
   a. 使用 gray-matter 解析 Frontmatter 和内容
   b. 使用 schema 验证 Frontmatter
   c. 合并为 Note 对象
4. 验证所有笔记的 slug 唯一性
5. 返回 Note[] 数组
```

**错误处理**:
- 目录不存在：返回空数组
- 文件解析失败：记录错误并抛出，阻止构建
- 验证失败：抛出详细错误信息

### 2.2 笔记过滤逻辑

#### 按标签过滤 (`getNotesByTag`)

**逻辑**:
```typescript
1. 遍历笔记的 tags 数组
2. 匹配条件：
   - tag.name === tagName
   - tag.alias?.includes(tagName)
   - 向上查找父标签链，检查是否有匹配
3. 返回所有匹配的笔记
```

**特点**: 支持标签别名和层级关系

#### 按主题过滤 (`getNotesByTopic`)

**逻辑**:
```typescript
1. 遍历笔记的 topics 数组
2. 匹配条件：
   - topic.name === topicName
   - topic.alias?.includes(topicName)
3. 返回所有匹配的笔记
```

**特点**: 支持主题别名

### 2.3 排序逻辑

**位置**: `src/utils/notes.ts::sortNotes()`

**支持的排序字段**:
- `date`: 按发布日期排序
- `question.type`: 按问题类型排序
- `quality.overall`: 按质量评分排序
- `title`: 按标题字母排序

**排序方向**: `asc` (升序) 或 `desc` (降序)

**实现**: 使用 JavaScript `Array.sort()`，返回新数组（不修改原数组）

### 2.4 分页逻辑

**位置**: `src/utils/notes.ts::paginateNotes()`

**逻辑**:
```typescript
1. 计算总页数: Math.ceil(total / perPage)
2. 计算起始索引: (page - 1) * perPage
3. 计算结束索引: startIndex + perPage
4. 切片数组获取当前页数据
5. 返回分页结果和元数据
```

**返回结构**:
```typescript
{
  notes: Note[],           // 当前页笔记
  pagination: {
    currentPage: number,
    totalPages: number,
    perPage: number,
    total: number,
    hasNext: boolean,
    hasPrev: boolean
  }
}
```

## 3. 标签层级逻辑

### 3.1 标签层级构建

**位置**: `src/utils/tags.ts::buildTagHierarchy()`

**逻辑**:
```
1. 收集所有笔记中的所有标签
2. 为每个标签创建 TagNode 对象
3. 计算每个标签的深度（递归向上查找父标签）
4. 建立父子关系（parent.children.push(child)）
5. 返回 Map<tagName, TagNode>
```

**深度计算**:
```typescript
function calculateDepth(tagName: string, visited: Set<string>): number {
  if (visited.has(tagName)) return 0; // 防止循环引用
  visited.add(tagName);
  
  const tag = tagMap.get(tagName);
  if (!tag || !tag.parent) return 1;
  
  return 1 + calculateDepth(tag.parent, visited);
}
```

**特点**:
- 防止循环引用（使用 visited Set）
- 支持多级嵌套
- 自动计算深度

### 3.2 标签别名解析

**位置**: `src/utils/tags.ts::resolveTagAlias()`

**逻辑**:
```
1. 构建标签层级
2. 遍历所有标签：
   - 如果 name === tagName，返回 name
   - 如果 alias?.includes(tagName)，返回 name
3. 如果未找到，返回原始 tagName
```

**用途**: 允许通过别名访问标签，支持标签重命名

### 3.3 子标签收集

**位置**: `src/utils/tags.ts::getChildTags()`

**逻辑**:
```
1. 构建标签层级
2. 找到父标签节点
3. 递归收集所有子标签（包括子标签的子标签）
4. 返回子标签名称数组
```

**用途**: 在标签页面显示时，包含所有子标签的笔记

## 4. 搜索索引生成逻辑

### 4.1 文本提取

**位置**: `src/utils/search.ts::extractTextFromMarkdown()`

**逻辑**:
```typescript
1. 移除代码块 (```...```)
2. 移除行内代码 (`...`)
3. 移除链接，保留文本 ([text](url) → text)
4. 移除 Markdown 标记符号 (#*_~`)
5. 合并多个换行为单个空格
6. 去除首尾空白
```

**目的**: 从 Markdown 中提取纯文本用于搜索

### 4.2 索引生成

**位置**: `src/utils/search.ts::generateSearchIndex()`

**逻辑**:
```
1. 遍历所有笔记
2. 对每个笔记：
   a. 提取纯文本内容
   b. 收集所有标签（包括别名）
   c. 收集所有主题（包括别名）
   d. 构建索引条目
3. 收集所有唯一标签和主题
4. 返回 SearchIndex 对象
```

**索引结构**:
```typescript
{
  version: "1.0",
  notes: [{
    slug: string,
    title: string,
    content: string,      // 纯文本
    tags: string[],
    topics: string[],
    questionType: string,
    date: string
  }],
  tags: string[],         // 所有唯一标签
  topics: string[]        // 所有唯一主题
}
```

### 4.3 构建时生成

**位置**: `src/integrations/generate-search-index.ts`

**Hook 1: `astro:server:setup`** (开发模式)
```
1. 读取所有笔记文件
2. 验证 Frontmatter
3. 生成搜索索引
4. 写入 public/search-index.json
```

**Hook 2: `astro:build:done`** (构建模式)
```
1. 读取所有笔记文件
2. 验证 Frontmatter
3. 生成搜索索引
4. 写入 dist/search-index.json
```

**错误处理**:
- 开发模式：错误不阻止服务器启动，只记录警告
- 构建模式：错误会阻止构建完成

## 5. 客户端搜索逻辑

### 5.1 搜索执行

**位置**: `src/pages/search.astro` (内联脚本)

**流程**:
```
1. 从 URL 获取查询参数 ?q=...
2. 如果查询为空，不执行搜索
3. fetch('/search-index.json')
4. 解析 JSON
5. 过滤笔记：
   - 标题匹配 (title.toLowerCase().includes(term))
   - 内容匹配 (content.toLowerCase().includes(term))
   - 标签匹配 (tags.some(tag => tag.includes(term)))
   - 主题匹配 (topics.some(topic => topic.includes(term)))
6. 动态渲染结果
```

**匹配逻辑**: 使用 `includes()` 进行子字符串匹配（不区分大小写）

### 5.2 结果渲染

**逻辑**:
```
1. 如果无结果：显示"未找到匹配的笔记"
2. 如果有结果：
   a. 显示结果数量
   b. 遍历结果，渲染每个笔记：
      - 标题（链接到详情页）
      - 内容摘要（前150字符）
      - 日期、标签、主题
3. HTML 转义防止 XSS
```

## 6. 静态页面生成逻辑

### 6.1 动态路由生成

**位置**: `src/pages/*/[param].astro`

**流程** (以 `notes/[slug].astro` 为例):
```
1. getStaticPaths() 函数：
   a. 调用 getAllNotes()
   b. 为每个笔记生成路径参数
   c. 返回 { params: { slug }, props: { note, ... } }
2. Astro 为每个路径生成静态 HTML
3. 页面组件使用 props 渲染内容
```

**特点**:
- 构建时生成所有可能的路径
- 每个路径对应一个静态 HTML 文件
- 支持 SEO（每个页面有独立 URL）

### 6.2 分页生成

**位置**: `src/pages/page/[page].astro`

**逻辑**:
```
1. getStaticPaths():
   a. 获取所有笔记
   b. 计算总页数
   c. 为每页生成路径（从第2页开始，第1页是首页）
2. 每个页面：
   a. 根据页码获取对应页的笔记
   b. 渲染分页组件
```

## 7. SEO 元数据生成

### 7.1 笔记 SEO

**位置**: `src/utils/seo.ts::generateSEOMeta()`

**逻辑**:
```
1. 构建标题: `${note.title} | ${siteName}`
2. 使用笔记的标题作为描述（或截取前160字符）
3. 生成 OpenGraph 标签
4. 生成 Canonical URL
5. 返回 SEOData 对象
```

### 7.2 页面 SEO

**位置**: `src/utils/seo.ts::generatePageMeta()`

**逻辑**:
```
1. 使用页面标题和描述
2. 生成完整 URL
3. 生成 OpenGraph 和 Canonical
4. 返回 SEOData 对象
```

## 8. 错误处理策略

### 8.1 构建时错误

- **数据验证失败**: 抛出错误，阻止构建
- **文件读取失败**: 记录错误并抛出
- **Slug 重复**: 抛出详细错误，列出所有重复文件

### 8.2 运行时错误

- **搜索索引加载失败**: 显示错误消息
- **JavaScript 执行错误**: 控制台记录，页面显示友好错误

### 8.3 开发时错误

- **搜索索引生成失败**: 记录警告，不阻止服务器启动
- **文件解析错误**: 记录错误，继续处理其他文件

## 9. 性能优化策略

### 9.1 构建时优化

- **静态生成**: 所有页面在构建时生成，无需运行时渲染
- **数据缓存**: 笔记数据在构建时读取一次，复用多次
- **索引生成**: 搜索索引在构建时生成，客户端直接使用

### 9.2 运行时优化

- **零 JavaScript**: 默认情况下不发送 JavaScript（搜索功能除外）
- **静态资源**: CSS 和图片等静态资源由 CDN 提供
- **客户端搜索**: 搜索在客户端执行，无需服务器请求

### 9.3 未来优化方向

- **增量构建**: 只重新生成变更的页面
- **索引分块**: 当索引文件 > 1MB 时，考虑分块或压缩
- **懒加载**: 对于大量笔记，考虑按需加载
