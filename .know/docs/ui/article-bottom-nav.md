# 文章页底部导航 交互设计

<!-- 核心问题: 文章末尾下方应该出现哪些"继续阅读"块？ -->

## 1. 候选块清单

文章页（`src/pages/notes/[slug].astro`）评论区后、回到顶部前，最多有 4 类候选块：

| ID | 选择器 | 渲染条件 | 视觉 |
|----|--------|----------|------|
| `series-continue` | `.series-continue:not(.topic-continue)` | 文章有 `series` frontmatter | 大卡：进度条 + 系列名 + 下一篇/上一篇标题（突出） |
| `topic-continue` | `.series-continue.topic-continue` | 文章 `topics[0]` 存在且有 ≥1 同主题其他文章 | 大卡：主题名 + 同主题 3 篇 + "查看全部"链接 |
| `series-nav` | `.series-nav` | 同 series-continue | 紧凑双列 prev/next（cat-label + 标题） |
| 通用 prev/next | `nav[aria-label="文章导航"]` | **`!series && (prevNote ‖ nextNote)`**（系列文章不渲染） | 紧凑双列 |

`related-section`（相关文章）独立位于评论区**之前**，不属于本文档范围。

## 2. 渲染矩阵

行 = 文章属性，列 = 入口 ctx（来自 `<html data-source-context>`，参见 [arch/entry-context-routing](../arch/entry-context-routing.md)）。

`✓` = 显示，`✗` = `display: none`，`—` = 不渲染（条件不满足）。

| 文章属性 | ctx | series-continue | topic-continue | series-nav | 通用 prev/next |
|----------|-----|----|----|----|----|
| 仅 series | 任意 | ✓ | — | ✓ | — |
| 仅 topic | 任意 | — | ✓ | — | ✓ |
| series + topic | `null`（默认） | ✓ | ✓ | ✓ | — |
| series + topic | `topic` | ✗ | ✓ | ✗ | — |
| series + topic | `series` | ✓ | ✗ | ✓ | — |
| series + topic | `tag` | ✓ | ✓ | ✓ | — |
| 都没有 | 任意 | — | — | — | ✓ |

### 隐藏规则（`<style>` 块）

```css
:root[data-source-context="topic"] .series-continue:not(.topic-continue),
:root[data-source-context="topic"] .series-nav { display: none; }
:root[data-source-context="series"] .topic-continue { display: none; }
```

## 3. 设计原则

### 3.1 系列内文章不渲染通用 prev/next

`series-continue` + `series-nav` 已覆盖前后篇，再加一组通用 prev/next 等于第三次重复同一篇。模板 gate：`!series && (prevNote || nextNote)`。

### 3.2 ctx 决定哪一块"消失"，不决定哪一块"存在"

服务端按数据条件渲染所有块；客户端 JS 按入口 ctx 通过 CSS 隐藏非来源块。**ctx 是裁剪器，不是渲染器**。

理由：SSR HTML 完整 → SEO 看到全部上下文；CSS 隐藏比 JS 移除节点便宜、可逆；无 ctx 时回退默认（全显示）天然合理。

### 3.3 topic 兄弟不去重 series 成员

早期实现把"已在 series 内的"从 topicSiblings 过滤掉，结果当系列全员同主题时 topic 块为空、不渲染 → 从主题入口看不到 topic 上下文。**故意保留重叠**，因为两块的语义不同：series 强调"按顺序"，topic 强调"按归属"。

## 4. 扩展指南

### 新增一个 continue 类型块（如 `tag-continue`）

1. 模板：参照 `topic-continue` 写一个 section，复用 `.series-continue` 视觉 primitive，加自己的 modifier class（如 `tag-continue`）
2. 渲染矩阵：在第 2 节表格里加 `tag-continue` 列 + 隐藏规则行
3. CSS 隐藏：`:root[data-source-context="topic"] .tag-continue { display: none; }` 等
4. 不需要改 dispatcher / reader

### 新增 ctx（如 `archive`）

参见 [arch/entry-context-routing §4.1](../arch/entry-context-routing.md#41-新增上下文页)。新 ctx 上线后，决定本表中每块在新 ctx 下显隐，补一行规则到 `<style>`。
