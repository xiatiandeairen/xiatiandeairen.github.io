# 提交规范 (Commit Convention)

## 基本原则

1. **遵循 Conventional Commits 规范**
2. **全英文描述**：所有提交信息使用英文
3. **单一粒度**：每个 commit 只包含一个逻辑变更
4. **清晰明确**：提交信息应清楚描述做了什么和为什么

## 提交格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type（必需）

提交类型，必须是以下之一：

- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档变更
- `style`: 代码格式变更（不影响代码运行）
- `refactor`: 代码重构（既不是新功能也不是 bug 修复）
- `perf`: 性能优化
- `test`: 测试相关变更
- `build`: 构建系统或外部依赖变更
- `ci`: CI 配置文件和脚本变更
- `chore`: 其他变更（如依赖更新、工具配置等）
- `revert`: 回滚之前的提交

### Scope（可选）

影响范围，例如：

- `schema`: Frontmatter Schema 相关
- `pages`: 页面相关
- `components`: 组件相关
- `utils`: 工具函数相关
- `styles`: 样式相关
- `config`: 配置文件相关
- `deps`: 依赖相关

### Subject（必需）

简短描述，不超过 50 个字符：

- 使用祈使句，现在时态（如 "add" 而不是 "added" 或 "adds"）
- 首字母小写
- 不以句号结尾

### Body（可选）

详细描述，说明：

- 变更的动机
- 与之前行为的对比
- 可以多行，每行不超过 72 个字符

### Footer（可选）

- 关闭的 issue：`Closes #123`
- 破坏性变更：`BREAKING CHANGE: <description>`

## 示例

### 新功能

```
feat(notes): add version history support

Implement version management system with history tracking.
Each note can now maintain version history with change summaries.

Closes #45
```

### Bug 修复

```
fix(pages): correct import path in page/[page].astro

The IndexLayout import path was incorrect, causing build failures.
Changed from '../IndexLayout.astro' to '../../layouts/IndexLayout.astro'.
```

### 文档更新

```
docs(readme): update installation instructions

Add pnpm installation steps and clarify dependency requirements.
```

### 代码重构

```
refactor(components): simplify NoteCard component

Remove unnecessary card wrapper and use semantic HTML elements.
Improve accessibility with proper article tags.
```

### 样式变更

```
style(global): update typography for better readability

Increase base font size to 18px and line height to 1.8.
Apply pure style design principles.
```

### 性能优化

```
perf(build): optimize static page generation

Implement incremental build strategy for large note collections.
Reduce build time by 40% for 1000+ notes.
```

### 构建系统

```
build(deps): add @tailwindcss/typography plugin

Add typography plugin to support prose classes for Markdown content.
```

### CI 配置

```
ci(workflows): update GitHub Pages deployment

Configure Astro build output and add build cache for faster CI runs.
```

### 其他变更

```
chore: remove Hexo-related files

Delete _config.yml, themes/, and other Hexo-specific files.
Clean up project structure for Astro migration.
```

## 提交检查清单

在提交前确认：

- [ ] 提交信息遵循 Conventional Commits 格式
- [ ] 使用全英文描述
- [ ] 提交包含单一逻辑变更
- [ ] Type 选择正确
- [ ] Subject 简洁明确（≤50 字符）
- [ ] 如有必要，包含 Body 详细说明
- [ ] 破坏性变更已标注 BREAKING CHANGE

## 工具支持

可以使用以下工具辅助：

- `commitizen`: 交互式提交信息生成
- `commitlint`: 提交信息格式校验
- `husky`: Git hooks 自动化检查

## 注意事项

1. **不要提交**：`node_modules/`, `dist/`, `.astro/` 等构建产物
2. **不要提交**：个人配置、临时文件、敏感信息
3. **及时提交**：完成一个功能点后立即提交，不要累积大量变更
4. **清晰描述**：提交信息应该让其他开发者（包括未来的自己）能理解变更内容
