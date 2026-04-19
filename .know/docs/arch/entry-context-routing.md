# Entry Context Routing 架构

## 1. 定位与边界

### 职责

让"用户从哪个上级页进入文章"成为下游页面可读取的状态，驱动 UI 在不同入口下展示不同内容。提供一套 **dispatcher → storage → reader → CSS** 的 4 段管道，新增上下文页或上下文驱动的 UI 都按同一模式扩展。

### 不负责

- 文章页底部展示哪些块（→ [ui/article-bottom-nav](../ui/article-bottom-nav.md)）
- 推荐排序、related 评分逻辑（→ `src/utils/rank.ts`）
- 跨标签页 / 持久化的"用户偏好"（不在 sessionStorage 范畴）

## 2. 结构与交互

### 数据流

```
[列表页 click]
   │
   ▼
[BaseLayout dispatcher] ──按 location.pathname 派生 ctx──▶ sessionStorage["entryContext"]
   │
   │ (页面 nav)
   ▼
[文章页 reader] ──读 sessionStorage──▶ <html data-source-context="X"> ──+ removeItem
   │
   ▼
[CSS :root[data-source-context="X"] .block { display: none }]
```

### 4 段实现

| 段 | 文件 | 职责 |
|----|------|------|
| dispatcher | `src/layouts/BaseLayout.astro`（body 末） | 全局 click 委托：监听 `a[href^="/notes/"]`，按当前 `location.pathname` 派生 ctx 写入 sessionStorage |
| storage | sessionStorage key `entryContext` | 单次跳转载体，值 ∈ {`topic`, `series`, `tag`}（可扩展） |
| reader | `src/pages/notes/[slug].astro`（body 末 inline script） | 读取 storage（无值则回退 `document.referrer`）→ `<html data-source-context>` → `removeItem` 清除 |
| CSS rule | `src/pages/notes/[slug].astro` `<style>` | `:root[data-source-context="X"] .non-X-block { display: none }` |

## 3. ctx 派生规则

dispatcher 按 `location.pathname` 优先级匹配：

| 当前页路径前缀 | ctx 值 | 备注 |
|---------------|-------|------|
| `/topics/` | `topic` | 涵盖 `/topics/index` 和 `/topics/[topic]` |
| `/series` | `series` | 列表页 `/series` |
| `/tags/` | `tag` | 涵盖 `/tags/[tag]` |
| `/notes/` | 当前 `<html data-source-context>` 值 | **article→article 透传**，让 ctx 在 related / 系列卡片跳转间延续 |
| 其他（home / archive / search / about） | `null` | dispatcher 调用 `removeItem`，确保不残留上轮 stale ctx |

## 4. 扩展点

### 4.1 新增上下文页

只改 dispatcher 一处（`BaseLayout.astro`），加一行 `else if (...)`。**不要**回到具体列表页加 inline script。

### 4.2 新增上下文驱动的 UI 块

在文章页 `<style>` 里加：
```css
:root[data-source-context="新ctx"] .要隐藏的块 { display: none; }
```
不需要改 reader / dispatcher。

### 4.3 新增需要"了解入口"的页面

在该页面 body 末加同样的 reader 脚本（4 行），读 sessionStorage + 设属性 + 清除。

## 5. Trade-offs（为什么是 sessionStorage 而不是 referrer / URL 参数）

| 方案 | 优 | 劣 | 结论 |
|------|----|----|------|
| `document.referrer` | 零侵入，无 storage | **不可靠**：Cmd+click / 严格 referrer 策略 / 地址栏粘贴都会丢；同标签同源左键点击有时也丢 | ❌ 仅作 fallback，不作主路径 |
| URL 参数 `?from=topic` | 100% 可靠，SSR 可读 | 同一文章 N 个 URL，分裂 canonical/SEO；分享 URL 带参数尴尬；构建产物缓存不再是 1:1 | ❌ |
| `sessionStorage` + click handler | URL 干净；同标签 100% 准；新标签退化合理；可在 article 间透传 | 依赖 JS；`document.referrer` 作 fallback 兜底 | ✅ 选用 |

referrer 不可靠的具体证据：用户在普通同标签左键点击场景，console 中 `document.referrer` 实测为空（浏览器/插件策略未知），导致最初基于 referrer 的实现失效。

## 6. 测试要点

改动 dispatcher / reader 后，用 playwright 至少验 3 条路径：

1. **冷启**：`/topics/X` → click 文章 → `<html data-source-context>` 应等于 `topic`，对应非来源块 `display:none`
2. **透传**：上一步基础上，文章 A 内点 related → 文章 B → ctx 仍应等于 `topic`
3. **退出**：home / search 页 → click 文章 → ctx 应为 undefined，sessionStorage 应已清空（dispatcher 的 `removeItem` 路径）

不要肉眼判断；必须 `getComputedStyle(el).display` 取数。
