# 项目文档索引

本文档目录包含项目的技术文档，帮助开发者理解项目结构、模块关系和核心逻辑。

## 文档列表

### 1. [项目结构文档](./project-structure.md)

**内容**:
- 完整的目录结构说明
- 各模块的职责和功能
- 文件命名规范
- 扩展性设计

**适用场景**:
- 新成员了解项目结构
- 查找特定功能的位置
- 规划新功能的位置

### 2. [模块调用关系文档](./module-relationships.md)

**内容**:
- 模块依赖关系图
- 详细的调用链说明
- 数据流向分析
- 循环依赖预防策略

**适用场景**:
- 理解模块间的依赖关系
- 修改代码时评估影响范围
- 重构时的依赖分析

### 3. [重要逻辑说明文档](./core-logic.md)

**内容**:
- 数据验证逻辑
- 笔记读取和过滤逻辑
- 标签层级构建逻辑
- 搜索索引生成逻辑
- 客户端搜索逻辑
- 静态页面生成逻辑
- SEO 元数据生成
- 错误处理策略
- 性能优化策略

**适用场景**:
- 理解核心功能的实现细节
- 调试问题时查找相关逻辑
- 优化性能时的参考
- 扩展功能时的设计参考

## 快速查找

### 按功能查找

- **数据验证**: [核心逻辑 - 数据验证逻辑](./core-logic.md#1-数据验证逻辑)
- **笔记读取**: [核心逻辑 - 笔记读取逻辑](./core-logic.md#2-笔记读取逻辑)
- **标签系统**: [核心逻辑 - 标签层级逻辑](./core-logic.md#3-标签层级逻辑)
- **搜索功能**: [核心逻辑 - 搜索索引生成逻辑](./core-logic.md#4-搜索索引生成逻辑)
- **分页功能**: [核心逻辑 - 分页逻辑](./core-logic.md#24-分页逻辑)
- **SEO 优化**: [核心逻辑 - SEO 元数据生成](./core-logic.md#7-seo-元数据生成)

### 按模块查找

- **类型定义**: [项目结构 - types/note.ts](./project-structure.md#srctypesnotets)
- **数据验证**: [项目结构 - utils/schema.ts](./project-structure.md#srcutilsschematsts)
- **笔记操作**: [项目结构 - utils/notes.ts](./project-structure.md#srcutilsnotests)
- **标签操作**: [项目结构 - utils/tags.ts](./project-structure.md#srcutilstagsts)
- **搜索索引**: [项目结构 - utils/search.ts](./project-structure.md#srcutilssearchts)
- **构建集成**: [项目结构 - integrations/](./project-structure.md#srcintegrations)

### 按问题查找

- **如何添加新字段**: [核心逻辑 - Frontmatter Schema 验证](./core-logic.md#11-frontmatter-schema-验证)
- **如何修改排序逻辑**: [核心逻辑 - 排序逻辑](./core-logic.md#23-排序逻辑)
- **如何扩展标签功能**: [核心逻辑 - 标签层级构建](./core-logic.md#31-标签层级构建)
- **如何优化搜索**: [核心逻辑 - 搜索索引生成逻辑](./core-logic.md#4-搜索索引生成逻辑)
- **如何调试构建错误**: [核心逻辑 - 错误处理策略](./core-logic.md#8-错误处理策略)

## 文档维护

### 更新原则

1. **及时更新**: 代码变更时同步更新相关文档
2. **保持准确**: 确保文档与代码实现一致
3. **清晰明了**: 使用图表和示例说明复杂逻辑
4. **完整覆盖**: 重要功能必须有文档说明

### 文档结构

每个文档应包含：
- 清晰的标题和目录
- 详细的说明和示例
- 代码片段（如适用）
- 相关链接

## 相关资源

- [项目 README](../README.md) - 项目概览和快速开始
- [目录规范](../.cursor/rules/directory-structure.md) - 目录结构规范
- [编程规范](../.cursor/rules/coding-standards.md) - 代码编写规范
- [提交规范](../.cursor/rules/commit-convention.md) - Git 提交规范
