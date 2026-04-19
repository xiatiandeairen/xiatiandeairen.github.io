# 夏天的爱人 — 用户功能与回归测试场景

> 全站用户可见功能清单 + 每项的回归验收路径。每个功能给出**入口**、**期望行为**、**测试步骤**三段。
>
> 测试基线：`npm run dev` 启动后访问 http://localhost:4321/

---

## 1 内容浏览

### 1.1 首页 hero 头条

- **入口**：`/` 顶部
- **期望**：显示 `featured: true` 的文章；若无则显示最新一篇
- **测试**：访问 `/`，hero 大标题应是某篇被标记为精选的文章；右侧 sidebar 显示 3 篇近期

### 1.2 首页"猜你想读"个性化区块

- **入口**：`/` 中段，介于次级文章网格和剩余列表之间
- **期望**：
  - 首次访问站点（无浏览历史）：**整块隐藏**
  - 访问过 ≥1 篇文章再回首页：显示 4 张卡片
  - 卡片不会出现已读的文章
- **测试**：
  1. 浏览器隐身窗口打开 `/` → 不应看到"猜你想读"标题
  2. 进 `/notes/ai-agent-operating-system` 任意文章
  3. 回 `/` → 看到"猜你想读"区块，4 张相关文章卡片，不含 AI Agent 那篇

### 1.3 归档页

- **入口**：masthead → 归档 → `/archive`
- **期望**：按年/月分组，全部文章按时间倒序；可按 question.type 筛选
- **测试**：进归档页，确认按年聚合；点顶部 type 按钮过滤

### 1.4 系列页

- **入口**：masthead → 系列 → `/series`
- **期望**：每个系列一段，按 series.order 排序
- **测试**：进系列页，确认每个 series 内部文章顺序为 1,2,3...

### 1.5 主题/标签列表页

- **入口**：`/topics` / `/tags`，或文章 header chip 点击
- **期望**：列出全部主题/标签 + 每项的文章数

### 1.6 标签详情页（带桥接卡片）

- **入口**：`/tags/AI`（或任意 tag）
- **期望**：
  - 顶部桥接卡片显示"所属主题"（accent 色 chip）+ "相关标签"（灰色 chip + count）
  - 点击主题/标签 chip 跳转
  - 下方列出该 tag 下的文章
- **测试**：进 `/tags/AI`，应看到 [AI 工程化, 开发工具, 方法论] 主题 + [agent×7, engineering×7, sprint×2, ...] 标签

### 1.7 主题详情页（带桥接卡片）

- **入口**：`/topics/AI%20%E5%B7%A5%E7%A8%8B%E5%8C%96`
- **期望**：顶部桥接卡片显示"包含标签" + "关联主题"
- **测试**：进 `/topics/AI 工程化`，应看到该主题下出现频次最高的 5 个 tag + 共现的其他主题

### 1.8 文章详情页基础

- **入口**：`/notes/<slug>`
- **期望**：标题居中、kicker 类型、阅读时长 + 日期 + topic/tag chip
- **测试**：进任意文章，header 元信息齐全；正文按 prose 样式渲染

### 1.9 文章 TOC 浮层

- **入口**：文章页右下角"≡"按钮
- **期望**：
  - 点击展开 TOC 列表
  - 点击条目 smooth 滚动到对应 heading，heading 闪烁高亮 2.4s
  - book 模式下点条目跳到对应 spread
- **测试**：
  1. 任意长文 → 点 TOC 按钮 → 列表出现
  2. 点任一项 → 滚到 heading，左侧 accent 色短条出现 2.4s
  3. 进 book 模式 → 点 TOC → 翻到含该 heading 的 spread

### 1.10 文章相关推荐

- **入口**：文章正文末尾"相关文章"区
- **期望**：3 张卡片，按 tag(3) + topic(5) + series(4) + type(2) 加权排序
- **测试**：随便看一篇，相关文章应是同主题/系列优先

### 1.11 系列继续阅读专区

- **入口**：系列中的文章末尾，accent 色背景的卡片块
- **期望**：
  - 显示系列名 + 进度（N / 总数）+ 进度条
  - 显示下一篇大卡片（标题 + "第 X 篇"）
  - 系列剩余 ≥2 篇时显示"系列剩余 X 篇 · 查看全部"
- **测试**：进 `/notes/ai-agent-operating-system`（属于 AI Skills 系列 1/7），应看到该卡片，进度 "1 / 7"，下一篇大卡

### 1.12 系列前后导航

- **入口**：上述继续专区下方
- **期望**：左右两栏，上一篇 / 下一篇（系列内）
- **测试**：系列文章应显示双栏

### 1.13 文章前后导航（按时间）

- **入口**：文章页末尾再往下
- **期望**：← 上一篇 / 下一篇 → 按全站日期顺序
- **测试**：第二篇文章往下应能看到双栏跳转

### 1.14 评论（Giscus）

- **入口**：文章末尾"评论"区
- **期望**：Giscus iframe 加载；切换主题时同步主题
- **测试**：滚到底，Giscus 加载（GitHub 登录）

---

## 2 阅读体验

### 2.1 章节进度条

- **入口**：文章页顶部 3px 横条
- **期望**：
  - 每个 h2 一段，宽度按字数比例
  - 已读段填满 accent 色，当前段部分填，未读段透明
  - hover 段显示章节标题 tooltip
  - 点击段跳到该 heading
  - 无 h2 文章降级为单条总进度
- **测试**：长文滚动，顶部条按章节填充；点段跳转

### 2.2 阅读模式（mode = reading）

- **入口**：右下角按钮（书本图标）/ `r` 键
- **期望**：
  - chrome 全部隐藏（masthead/footer/sidebar/related/giscus/series-banner/series-nav/back-to-top/prev-next）
  - 顶部进度条保留
  - Aa 浮层显示
  - 退出按钮 accent 色
- **测试**：按 `r` → 全屏只剩文章 + 进度条 + 右下浮层；再按 `r` 退出

### 2.3 翻页模式（mode = book）

- **入口**：右下角按钮 / `b` 键
- **期望**：
  - 双列布局，固定 100vh，无滚动
  - 底部居中显示 `N / M` 页码
  - `←/→` 翻页，`Space/PgDn` 下一页，`PgUp` 上一页
  - `Home/g` 首页，`End/G` 尾页，`Esc` 退出
  - 触屏左右滑动翻页（>60px 横向，垂直幅度小）
  - 移动端（<768px）单列排版
  - 切换 typography 后自动重算 spread 数
- **测试**：
  1. 按 `b` → 双列出现，"1 / N" 页码
  2. `←/→` 翻页响应
  3. 触屏模拟左滑 → next
  4. 缩放浏览器窗口 → spread 重算

### 2.4 模式切换按钮（统一）

- **入口**：右下角圆形按钮（图标随模式变化：文档/开卷/双列）
- **期望**：
  - **左键**：循环 default → reading → book → default
  - **右键 / 长按**：弹三选一菜单（默认/阅读/翻页）
  - active 模式按钮 accent 色 + 菜单 aria-current
- **测试**：左键点 3 次走完一圈；右键弹菜单，选某项立即切换

### 2.5 排版控制（Aa 浮层）

- **入口**：右下角 "Aa" 按钮（仅在 reading/book 模式可见）
- **期望**：
  - 字号：小（15px）/ 中（16px）/ 大（19px）
  - 字体：衬线（Georgia）/ 无衬线（Inter）
  - 行宽：窄（62ch）/ 默认（820px）/ 宽（≤1100px）
  - 偏好 localStorage 持久化
  - **退出 reading/book 模式时偏好失效**（不影响默认布局）
- **测试**：reading 模式下选字体、字号、行宽，正文跟随变化；切回 default → 还原

### 2.6 阅读位置恢复

- **入口**：自动（每次进文章）
- **期望**：
  - 滚动 >200px 后离开 → 位置存 localStorage
  - 再次打开同篇文章且 scrollY < 100 → 左下角 toast 弹"继续上次阅读（X%）"
  - 点"继续"smooth 滚到保存位置；× 关闭；10s 自动消失
- **测试**：进文章滚到中段，关 tab 重新打开 → toast 出现，点继续

### 2.7 浮层闲置淡出

- **入口**：自动
- **期望**：右下浮层（TOC / Aa / 模式按钮）2 秒无活动 opacity 0.15；鼠标/滚动/键盘动作恢复
- **测试**：进文章静止 2s → 浮层变灰；动鼠标 → 恢复

### 2.8 键盘导航

- **入口**：键盘
- **期望**（flow 模式）：
  - `Space` / `PgDn` 下翻一屏，`PgUp` 上翻
  - `g` / `Home` 顶部，`G` / `End` 底部
  - `n` / `p` 下/上一个 h2/h3
  - `r` 切换阅读，`b` 切换翻页
  - 输入框内禁用
- **测试**：文章页按 Space 跳屏；按 `n` 跳章节；输入框聚焦时按 Space 输入空格而非翻页

### 2.9 代码块复制按钮

- **入口**：文章正文中的 `<pre>` 块右上角
- **期望**：hover 出现"复制"按钮，点击后变"已复制" 2s
- **测试**：含代码的文章 hover 代码块

### 2.10 Tag/Topic chip 样式

- **入口**：文章 header
- **期望**：
  - topic 用 accent 色（红）
  - tag 用 ink-ghost（浅灰）
  - hover：圆角边框 pill 出现
- **测试**：进文章看 header chip 区，颜色区分清晰

---

## 3 搜索

### 3.1 ⌘K 搜索 overlay

- **入口**：masthead 搜索图标 / `⌘K` / `Ctrl+K`
- **期望**：
  - 全屏 overlay，输入框 focus
  - placeholder："搜索 · 试试 tag: topic: type: series:"
  - 实时搜索，debounce ~160ms
  - 结果显示：标题 + matched-field 标签 + 日期 + 高亮 snippet
  - 默认选中第一项 active
  - `↑/↓` 移动选中，`Enter` 跳转
  - 点结果跳转 + 写入"最近搜索"
- **测试**：⌘K → 输 "AI" → 结果列表带高亮；按 ↓ 切换；Enter 跳转

### 3.2 字段前缀语法

- **入口**：搜索输入框
- **期望**：
  - `tag:X` → 仅匹配 tag
  - `topic:X` → 仅匹配 topic
  - `type:X` → 仅匹配 question.type
  - `series:X` → 仅匹配 series
  - `date:Nd` / `Ny` / `YYYY` / `YYYY-MM` → 时间范围过滤
  - 与自由文本可组合
- **测试**：
  1. 输 `tag:sprint` → 仅 2 篇 sprint 标签的结果
  2. 输 `series:Skills 实践` → series 命中 + 自由文本"实践"

### 3.3 搜索结果排序（相关度）

- **期望**：title-exact > title-partial > tag/topic-exact > tag/topic-partial > series > alias > content；同分按 date desc
- **测试**：搜 "AI" → 标题含 AI 的应排前面，正文命中的排后面

### 3.4 搜索空态导航

- **入口**：⌘K 后未输入
- **期望**：依次显示
  - 最近搜索（最多 5 条，× 删除单条）
  - 热门主题（top 3，accent 色 chip）
  - 热门标签（top 5，灰色 chip）
  - **搜索技巧**（4 个 dashed pill：`tag:` / `topic:` / `type:` / `series:`，点击插入到输入框）
- **测试**：⌘K 不输入 → 4 个 section 都在；点 `tag:` chip 看输入框是否填了 "tag:"

### 3.5 时间过滤 chip

- **入口**：⌘K 输入框正下方一行
- **期望**：[全部] [近30天] [近90天] [近1年]，点击 toggle `date:` 前缀；当前选中实心填充
- **测试**：点"近30天" → 输入框追加 `date:30d`，结果立即更新；再点"全部" → 移除

### 3.6 字段前缀 alias 命中

- **期望**：搜 tag/topic 的 alias 也命中主名
- **测试**：若 schema 有定义某个 alias，搜 alias 应返回主名结果

### 3.7 `/search` 页

- **入口**：搜索框 form 提交后跳 `/search?q=...`
- **期望**：完整列表（不限 8 条）+ 标题/snippet 高亮 + matched-field tag
- **测试**：⌘K 输入 → Enter 不点结果跳到 `/search` 页，列表完整显示

### 3.8 `/en/search` 英文版

- **入口**：英文页 ⌘K 或 form 提交
- **期望**：与中文逻辑相同，UI 文案为英文（"results found", "[tag] · ..."）
- **测试**：进 `/en/`，⌘K，搜索行为一致；结果文案英文

---

## 4 站点 Chrome

### 4.1 Masthead 导航

- **入口**：每页顶部
- **期望**：站点标题 + tagline + 主导航（主题/系列/归档/关于）+ 工具区（搜索 / 语言切换 / 主题切换 / 快捷键 ?）
- **测试**：导航链接全部可点；当前页 active 高亮

### 4.2 主题切换（3 主题）

- **入口**：masthead 月亮/太阳图标
- **期望**：
  - 单击循环 light → sepia → dark → light
  - 旁边 dropdown 三选一
  - localStorage 持久化
  - sepia：暖米黄底 + 棕字
- **测试**：循环点 3 次走完一圈；选 dropdown 任一项即时生效

### 4.3 语言切换

- **入口**：masthead 地球图标
- **期望**：当前语言为中 → 鼠标 hover 显 dropdown 含"中文/English"；点击跳到对应语言版本
- **测试**：进 `/`，hover 地球，选 English → 跳 `/en/`

### 4.4 Sidebar drawer

- **入口**：右侧贴边的"<"指示器
- **期望**：hover/触发时滑出右侧 sidebar，离开后收回
- **测试**：鼠标移到右侧 hot-zone，sidebar 滑出

### 4.5 Sidebar 内容

- **入口**：drawer 内
- **期望**：
  - 精选阅读（按 quality.overall + featured 排序，3 篇）
  - 最近浏览（仅当 localStorage 有时显示，最多 5 条）
  - 随机一篇按钮（点击随机跳）
  - 系列专栏（最多 3 + "全部系列 →"）
  - 热门标签（top 5，带 count）
  - 订阅 RSS
- **测试**：sidebar 各 section 都在；按"随机一篇" → 跳到非当前页

### 4.6 Footer

- **入口**：每页底部
- **期望**：版权 + 订阅链接 + RSS + Built with Astro
- **测试**：滚到底可见

### 4.7 Skip-to-content

- **入口**：键盘 Tab 第一下
- **期望**：屏幕左上角浮现"跳到主内容"链接，按 Enter 跳到 `#main-content`
- **测试**：刷新页面，立即 Tab，左上角看到 skip link

---

## 5 快捷键面板

### 5.1 触发与关闭

- **入口**：masthead `?` 图标 / 按 `?` / `Shift+/`
- **期望**：
  - 模态框居中显示，背板模糊
  - 关闭：× 按钮 / 背板点击 / `Esc` 键
  - 打开时焦点跳到关闭按钮
  - Tab 在模态内循环（focus trap）
  - 关闭后焦点恢复到原触发元素
- **测试**：按 ? → 弹出，焦点在 ×；Tab 循环；Esc 关 → 焦点回 trigger

### 5.2 快捷键内容（4 节）

- **期望分组**：
  - 全站：`?` / `⌘K` / `Esc`
  - 模式切换：`r` / `b`
  - 阅读模式：`Space` / `Shift+Space` / `PgDn` / `PgUp` / `n` / `p` / `g` / `G`
  - 翻页模式：`→` / `←` / `Space` / `PgDn` / `PgUp` / `Home` / `End` / `g` / `G`
- **测试**：模态展开 4 卡片 grid（手机 1 列）

### 5.3 i18n

- **期望**：进 `/en/` 任一页 → 模态文案为英文（"Keyboard Shortcuts", "Global", ...）
- **测试**：`/en/index` 按 `?`，标题英文

---

## 6 SEO / 元数据

### 6.1 SEO meta 标签

- **入口**：每页 `<head>`
- **期望**：title / description / canonical / OG / Twitter card 齐全
- **测试**：浏览器 view-source 检查 meta 标签

### 6.2 OG 图 fallback

- **期望**：每页 `og:image` 指向 `/og-default.svg`（不再 404）
- **测试**：访问 `/og-default.svg` → 200，看到品牌 SVG（暖米黄底 + 站点名）

### 6.3 RSS

- **入口**：footer 链接 / `/rss.xml`
- **测试**：访问 `/rss.xml` 应返回有效 RSS XML

### 6.4 Sitemap

- **入口**：`/sitemap-index.xml`
- **测试**：`curl /sitemap-index.xml` 应返回 sitemap 入口

### 6.5 搜索索引

- **入口**：`/search-index.json`
- **期望**：JSON 含 version: "1.1"、notes[]、tags[]、topics[]、tagCounts、topicCounts
- **测试**：`curl /search-index.json | jq '.version, (.notes|length)'`

---

## 7 持久化（localStorage 全清单）

| Key | 内容 | 失效条件 |
|---|---|---|
| `theme` | "light" / "sepia" / "dark" | 用户清缓存 |
| `mode` | "default" / "reading" / "book" | 同上 |
| `read-pos:<pathname>` | 滚动位置（数字） | 同上 |
| `book-pos:<pathname>` | 当前 spread 索引 | 同上 |
| `recentlyViewed` | `[{slug, title, ts}]` 最多 5 | 同上 |
| `search.recentQueries` | `[query, ...]` 最多 5 | 同上 |
| `type.font-size` | "s" / "m" / "l" | 同上 |
| `type.font-family` | "serif" / "sans" | 同上 |
| `type.width` | "narrow" / "default" / "wide" | 同上 |

**测试**：开 DevTools → Application → Local Storage → 应能看到上述 key

---

## 8 多语言

- 中文版 `/`，英文版 `/en/`
- 镜像页面：`/en/index`、`/en/about`、`/en/archive`、`/en/series`、`/en/search`
- 文章 `/notes/<slug>` **仅中文**，无英文镜像（已知设计选择）
- masthead 语言切换在镜像页之间跳转

**测试**：从 `/` 切英文 → 跳 `/en/`；导航链接、search 文案全英；快捷键面板英文

---

## 9 边界与已知限制

| 场景 | 表现 |
|---|---|
| 触屏窄屏（<768px）进入 book 模式 | 自动降级单列；翻页保留 |
| 刷新后浏览器自动 scroll restore（scrollY > 100）| Resume toast 不弹（避免重复打扰） |
| 旧版本 `search-index.json` 缓存 | console.warn，部分 section 静默消失但不崩溃 |
| 无 h2 的文章 | 章节进度条降级为单段全宽 |
| 文章无系列 | 不显示系列继续专区 + 无 series 导航 |

---

## 10 回归测试推荐顺序

1. **首页**：检查 hero / sidebar / "猜你想读"（清缓存测无历史 + 看历史回流）
2. **文章页**：TOC / 章节进度 / 模式切换 / 排版 / Resume toast / 相关 / 系列继续
3. **搜索**：⌘K 空态 / 输入实时 / 字段前缀 / 时间 chip / `↑/↓/Enter`
4. **Tag/Topic 页**：桥接卡片
5. **快捷键面板**：全站可达 / focus trap / i18n
6. **主题切换**：3 主题循环 + dropdown
7. **多语言**：中英镜像 + search 文案
8. **键盘可达性**：Tab 顺序 / skip-link / 全键盘操作
9. **移动端**：<768px book 模式降级 + sidebar drawer + chip wrap
10. **SEO**：view-source 检查 meta + 访问 `/og-default.svg` `/rss.xml` `/sitemap-index.xml`

---

## 11 测试环境快速启动

```bash
npm run dev               # http://localhost:4321/
npm run build             # 验证全站构建（32 pages）
npm run lint:unused       # 死 CSS class + 未引用组件
```

直接 URL 列表（复制进浏览器）：

- http://localhost:4321/
- http://localhost:4321/notes/ai-agent-operating-system
- http://localhost:4321/tags/AI
- http://localhost:4321/topics/AI%20%E5%B7%A5%E7%A8%8B%E5%8C%96
- http://localhost:4321/search?q=tag:sprint
- http://localhost:4321/series
- http://localhost:4321/archive
- http://localhost:4321/en/
- http://localhost:4321/og-default.svg
- http://localhost:4321/search-index.json
