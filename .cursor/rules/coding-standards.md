# 编程规范 (Coding Standards)

## TypeScript 严格模式

### 配置要求

项目必须启用 TypeScript 严格模式，`tsconfig.json` 应包含：

```json
{
  "extends": "astro/tsconfigs/strict",
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitThis": true,
    "alwaysStrict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true
  }
}
```

### 类型安全原则

1. **避免使用 `any`**
   - 优先使用具体类型
   - 使用 `unknown` 处理不确定类型
   - 使用类型断言时必须有充分理由

2. **明确的类型定义**
   - 函数参数和返回值必须有类型
   - 避免隐式类型推断（复杂场景）
   - 使用接口（interface）定义对象结构

3. **空值处理**
   - 使用 `null` 和 `undefined` 时明确标注
   - 使用可选链操作符 `?.`
   - 使用空值合并操作符 `??`

## 代码风格

### 命名规范

1. **变量和函数**：camelCase
   ```typescript
   const noteCount = 10;
   function getAllNotes() {}
   ```

2. **类型和接口**：PascalCase
   ```typescript
   interface NoteFrontmatter {}
   type SortField = 'date' | 'quality';
   ```

3. **常量**：UPPER_SNAKE_CASE 或 camelCase（根据作用域）
   ```typescript
   const MAX_PAGE_SIZE = 20;
   const siteConfig = { ... };
   ```

4. **组件文件**：PascalCase
   ```
   NoteCard.astro
   MetaFields.astro
   ```

### 函数规范

1. **单一职责**：每个函数只做一件事
2. **纯函数优先**：避免副作用，便于测试
3. **参数限制**：函数参数不超过 5 个，超过时使用对象
4. **明确返回类型**：函数必须有明确的返回类型

```typescript
// Good
function getNotesByTag(tagName: string, notes: Note[]): Note[] {
  return notes.filter(note => /* ... */);
}

// Bad
function getNotes(tag, notes) {
  return notes.filter(/* ... */);
}
```

### 错误处理

1. **明确的错误类型**：使用自定义错误类
2. **及时抛出**：发现问题立即抛出，不要静默失败
3. **错误信息清晰**：错误消息应包含上下文信息

```typescript
// Good
if (duplicates.length > 0) {
  const errorMessages = duplicates.map(
    ({ slug, files }) => `Slug "${slug}" is duplicated in: ${files.join(', ')}`
  );
  throw new Error(`Duplicate slugs found:\n${errorMessages.join('\n')}`);
}

// Bad
if (duplicates.length > 0) {
  throw new Error('Duplicates found');
}
```

## 代码组织

### 导入顺序

1. 外部依赖
2. 内部工具函数
3. 类型定义
4. 组件
5. 样式文件

```typescript
// External dependencies
import { z } from 'zod';
import matter from 'gray-matter';

// Internal utilities
import { validateNoteFrontmatter } from './schema';
import { PAGINATION } from './constants';

// Types
import type { Note } from './notes';

// Components
import NoteCard from '../components/NoteCard.astro';

// Styles
import '../styles/global.css';
```

### 文件组织

1. **单一文件原则**：每个文件一个主要功能
2. **相关代码聚合**：相关函数放在同一文件
3. **避免过长的文件**：超过 300 行考虑拆分

### 注释规范

根据用户规则：**除非明确要求，否则不添加注释**

仅在以下情况添加注释：
- 复杂的算法或业务逻辑
- 非显而易见的实现细节
- TODO 或 FIXME（临时标记）

## Astro 组件规范

### 组件结构

```astro
---
// 1. Imports
import Layout from '../layouts/Layout.astro';
import Component from '../components/Component.astro';

// 2. Type definitions
interface Props {
  title: string;
  description?: string;
}

// 3. Props destructuring
const { title, description } = Astro.props;

// 4. Data fetching/computation
const data = await fetchData();
---

<!-- 5. Template -->
<Layout>
  <Component data={data} />
</Layout>
```

### Props 类型

所有组件必须定义 Props 接口：

```typescript
interface Props {
  note: Note;
  mode?: 'public' | 'audit';
  showQuality?: boolean;
}
```

## 工具函数规范

### 纯函数优先

```typescript
// Good: Pure function
function sortNotes(notes: Note[], sortBy: SortField): Note[] {
  return [...notes].sort(/* ... */);
}

// Bad: Mutates input
function sortNotes(notes: Note[], sortBy: SortField): void {
  notes.sort(/* ... */);
}
```

### 错误处理

```typescript
// Good: Explicit error handling
export function getAllNotes(): Note[] {
  let files: string[];
  try {
    files = readdirSync(notesDir);
  } catch (error) {
    return [];
  }
  // ...
}

// Bad: Silent failure
export function getAllNotes(): Note[] {
  const files = readdirSync(notesDir);
  // ...
}
```

## 性能考虑

### 避免不必要的计算

```typescript
// Good: Memoize expensive operations
const sortedNotes = useMemo(() => sortNotes(notes), [notes]);

// In Astro: Compute once in getStaticPaths
export async function getStaticPaths() {
  const allNotes = getAllNotes(); // Compute once
  return allNotes.map(/* ... */);
}
```

### 文件系统操作

- 在构建时执行，不在运行时
- 使用缓存避免重复读取
- 批量处理，减少 I/O 操作

## 测试要求

### 单元测试

- 工具函数必须有单元测试
- 使用 Jest 或 Vitest
- 测试覆盖率目标：80%+

### 集成测试

- 关键流程需要有集成测试
- 构建流程需要验证

## 代码审查检查清单

提交代码前检查：

- [ ] TypeScript 严格模式通过
- [ ] 无 `any` 类型（除非有充分理由）
- [ ] 所有函数有明确的类型定义
- [ ] 错误处理完善
- [ ] 代码遵循命名规范
- [ ] 无未使用的导入和变量
- [ ] 代码格式统一（使用 Prettier）
- [ ] 无控制台调试代码（console.log 等）

## 工具配置

### Prettier（可选）

如果使用 Prettier，配置应统一：

```json
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100
}
```

### ESLint（可选）

如果使用 ESLint，应配置 TypeScript 规则：

```json
{
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:@typescript-eslint/recommended-requiring-type-checking"
  ],
  "rules": {
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/explicit-function-return-type": "warn"
  }
}
```

## 重构原则

1. **小步重构**：每次重构一个功能点
2. **保持测试**：重构时保持测试通过
3. **向后兼容**：如可能，保持 API 兼容
4. **文档更新**：重构后更新相关文档
