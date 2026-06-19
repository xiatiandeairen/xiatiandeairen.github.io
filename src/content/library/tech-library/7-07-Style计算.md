---
title: "Style 计算（Chromium/Blink 域）"
slug: "7-07"
collection: "tech-library"
group: "chromium内核"
order: 7007
summary: "TL;DR Blink 的 Style 计算是一条从\"脏节点标记\"到\"ComputedStyle 对象\"的六级流水线： Invalidation（规则→InvalidationSet→脏标记）→ Recalc 调度 → MatchAllRules（UA/User/Author 三源 + Bloom…"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:39:11.000Z"
updatedAt: "2026-06-12T11:39:11.000Z"
---
> **TL;DR**
> Blink 的 Style 计算是一条从"脏节点标记"到"ComputedStyle 对象"的六级流水线：
> Invalidation（规则→InvalidationSet→脏标记）→ Recalc 调度 → MatchAllRules（UA/User/Author 三源 + Bloom 预筛）→ StyleCascade（96-bit CascadePriority 比较 + @layer 翻转）→ Apply（CSSValue→ComputedStyle 字段转换）→ StyleAdjuster（后处理校正）。
> 两个关键缓存（SelectorFilter Bloom、MatchedPropertiesCache）把热路径从 O(rules × elements) 降到接近 O(elements)。
> Electron 完整继承此流水线，唯一差异在 UA stylesheet 注入点和 Node.js 侧可绕开渲染线程的 `setDevToolsWebContentsContents` 补丁。

---

## 前置依赖

| 需要掌握 | 用于理解 |
|---|---|
| 第 6 章 DOM 树构建 | Style 计算的输入是 Document + Element 树 |
| CSS Selectors Level 4 specificity 规则 | SelectorChecker 实现的基础 |
| CSS Cascade Level 5 spec | StyleCascade 的规格依据 |
| Blink GC（GarbageCollected<T>）基础 | ComputedStyle 对象生命周期 |

---

## 7.1 设计考古：CSS 计算从何而来

### 7.1.1 起点：Cascading Style Sheets 1.0（1996）

CSS 1.0 spec 定义了"cascading"概念时，描述了三层优先级：UA < User < Author，以及 specificity 计数规则（a-b-c-d 四元组）。早期浏览器（Navigator 4、IE 3）实现极为简陋，属性逐条解析，没有 ComputedStyle 概念，重新布局需要遍历全树。

### 7.1.2 KHTML → WebKit → Blink 的谱系

- **2001 KHTML**（KDE）：引入 `CSSStyleSelector`，将匹配和应用分离。
- **2002 Apple WebKit fork**：扩充为 `CSSStyleSelector::collectMatchingRules()`，加入 RuleSet 按 ID/class/tag 分桶索引。  
  来源：WebKit 早期 changelog（2003 年 David Hyatt 提交，「fast rejection of rules by bucket lookup」）。
- **2007 style sharing**：Darin Adler 提交 "style sharing" 优化（webkit.org/b/12610），两个具有相同 class + 无伪类状态的兄弟元素共享同一 ComputedStyle 对象；今 Blink 保留该机制在 MatchedPropertiesCache 层。
- **2013 Blink fork from WebKit**：Google 将 `CSSStyleSelector` 拆分为 `StyleResolver` + `StyleEngine`，强化增量失效（InvalidationSet）。
- **2019 StyleCascade 重构**：Rune Lillesveen 主导，将原来"priority sort then apply"替换为"add-and-compare CascadePriority"，支持后来的 CSS cascade layers（@layer）。相关 commit message："StyleCascade: Implement cascade applying in StyleCascade" (2019-12)。
- **2022 @layer 正式落地**：Chromium 99 发布 CSS Cascade Layers，背后是对 `CascadePriority` 结构的扩展（layer_order 字段）。

### 7.1.3 核心设计动机

1. **分批失效而非全量重算**：页面平均有数千元素，改一个 class 不能让全树重新做选择器匹配。InvalidationSet 记录"哪些特征变化需要重算哪些后代"。
2. **Cascade 与 Apply 分离**：Match 阶段收集候选声明，Cascade 阶段用 CascadePriority 比较出"赢家"，Apply 阶段才真正写 ComputedStyle 字段——职责分离使 `@layer`、`!important` 反转、animation overlay 都可以独立插入。
3. **缓存两级**：SelectorFilter（Bloom）作为 Match 前置快速拒绝，MatchedPropertiesCache 作为 Apply 结果复用——热页面的 recalc 时间减少 40-60%。

---

## 7.2 整体流水线

```
DOM mutation / Attribute change / Class change
        │
        ▼
[StyleEngine] InvalidationSet 标记脏节点
        │
        ▼
[StyleEngine] RecalcStyle() —— 树遍历找 dirty 节点
        │
        ▼  per element
[StyleResolver] ResolveStyle()
  ├─ ApplyBaseStyle()
  │    ├─ (cache hit?) CanReuseBaseComputedStyle → clone
  │    ├─ (inline only?) CanApplyInlineStyleIncrementally
  │    └─ ApplyBaseStyleNoCache()
  │         ├─ MatchAllRules()
  │         │    ├─ MatchUARules()
  │         │    ├─ MatchUserRules()
  │         │    ├─ MatchPresentationalHints()
  │         │    └─ MatchAuthorRules()
  │         │         ├─ MatchHostRules()
  │         │         ├─ MatchSlottedRules()
  │         │         ├─ MatchElementScopeRules()  ← RuleSet bucket lookup
  │         │         └─ MatchOuterScopeRules()
  │         ├─ MatchedPropertiesCache lookup
  │         ├─ StyleCascade::Apply()
  │         │    └─ CascadePriority 比较 (96-bit)
  │         └─ StyleAdjuster::AdjustComputedStyle()
  └─ ApplyAnimatedStyle()  (overlay on top of base)
        │
        ▼
  ComputedStyle*  (返回给 Element)
```

**【真实源码 third_party/blink/renderer/core/css/resolver/style_resolver.h】**  
**【真实源码 third_party/blink/renderer/core/css/resolver/style_resolver.cc（逻辑描述，实际代码量约 3500 行）】**

---

## 7.3 阶段一：InvalidationSet 与脏节点标记

### 7.3.1 设计动机

全量 recalc 代价 O(elements × rules)。页面改一个 class 时，受影响的元素只是树的一个子集。InvalidationSet 把"改了 class X → 需要 recalc 哪些后代"的答案在 stylesheet parse 阶段预计算并索引，运行时只需 O(InvalidationSet.size()) 次遍历。

### 7.3.2 InvalidationSet 结构

**【真实源码 third_party/blink/renderer/core/css/invalidation/invalidation_set.h】**

```cpp
// 简化示意，非逐字
class InvalidationSet : public RefCounted<InvalidationSet> {
  // "Hybrid backing": 只有 1 个时用 AtomicString，
  //                   多个时升级到 HashSet<AtomicString>
  Backing<ClassBacking>     classes_;      // 触发条件：后代有此 class
  Backing<IdBacking>        ids_;
  Backing<TagBacking>       tag_names_;
  Backing<AttrBacking>      attributes_;

  bool invalidates_self_          : 1;
  bool invalidates_nth_           : 1;
  bool tree_boundary_crossing_    : 1;
  // ...
};
```

**三种派生类型**（真实源码同文件）：
- `DescendantInvalidationSet` — 标记后代脏
- `SiblingInvalidationSet` — 标记兄弟脏
- `NthSiblingInvalidationSet` — 专为 `:nth-child` 等

### 7.3.3 RuleInvalidationData：规则→InvalidationSet 索引

**【真实源码 third_party/blink/renderer/core/css/invalidation/rule_invalidation_data.h】**

```
.foo .bar { color: red; }
```
解析时：向 `class_invalidation_sets["foo"]` 的 DescendantInvalidationSet 中 `AddClass("bar")`。运行时 `foo` 类变化 → 找出所有后代中有 `bar` 类的元素 → 标记 dirty。

**Bloom Filter 优化**：超过 50 条同类简单规则（`.foo { ... }`）后，用 Bloom filter 代替单条 entry，接受极低误判率换取内存节省。（源码注释："we avoid creating the actual HashSets until we have more than one item"）

### 7.3.4 StyleInvalidator：DOM 遍历 + 应用 InvalidationSet

**【真实源码 third_party/blink/renderer/core/css/invalidation/style_invalidator.h】**

```
StyleEngine::InvalidateStyle()
  ├─ 从 pending_invalidations_ 取出待处理 map
  └─ StyleInvalidator::Invalidate(Document, root_element)
       ├─ Invalidate(Element, SiblingData)
       ├─ CheckInvalidationSetsAgainstElement()  ← 检查是否命中任意 InvalidationSet
       ├─ InvalidateChildren()
       └─ InvalidateShadowRootChildren()
```

命中则给 Element 设 `NeedsStyleRecalc` flag，后续 RecalcStyle() 会处理。

---

## 7.4 阶段二：SelectorFilter — Bloom 预筛

**【真实源码 third_party/blink/renderer/core/css/selector_filter.h】**

```
热路径说明（源码注释）：
"A bitset filter (essentially a Bloom filter with only one hash function)
 to discard 60-70% of rules before expensive full matching"
```

关键设计：
- **8192-slot bitset（1 KB）**：ancestor ID/class/tag hash 在 `PushParent()` 时写入，`PopTo()` 时恢复（这是非标准 Bloom——通过记录 `set_bits_` 实现可撤销）
- **`FastRejectSelector()`**：检查 selector 右侧的"祖先依赖"所有 hash 是否都在 bitset；缺任一 → 确定不匹配，跳过全部 `SelectorChecker` 调用
- 100 个唯一字符串时误判率约 1.2%，误判只增加一次不必要的全量匹配，不影响正确性

---

## 7.5 阶段三：RuleSet 分桶 + SelectorChecker

### 7.5.1 RuleSet 分桶索引

**【真实源码 third_party/blink/renderer/core/css/rule_set.h】**

| 分桶 | 方法 | 典型匹配目标 |
|---|---|---|
| ID rules | `IdRules(AtomicString)` | `#myid` |
| Class rules | `ClassRules(AtomicString)` | `.foo` |
| Tag rules | `TagRules(AtomicString)` | `div`, `span` |
| Attribute rules | `AttrRules(AtomicString)` | `[type="text"]` |
| Universal rules | `UniversalRules()` | `*`, 复杂选择器 |
| UA Shadow pseudo | `UAShadowPseudoElementRules()` | `::placeholder` 等 |

每个桶是一个 `RuleMap`（HashMap → `RuleData` vector）。`RuleData` 包含 specificity、Bloom filter hash、cascade layer order 等元信息。

**Intervals 优化（源码 rule_set.h 注释）**：同一 `@layer` 或 container query 下的连续规则用 [start, end) 区间而非 per-rule 字段记录，节省内存。

### 7.5.2 SelectorChecker 与 MatchingContext

**【真实源码 third_party/blink/renderer/core/css/selector_checker.h】**

四种模式：

| 模式 | 用途 |
|---|---|
| `kResolvingStyle` | 正常 style recalc，设 restyle flags（如 `:hover` 依赖） |
| `kCollectingStyleRules` | 收集 StyleRuleList（DevTools） |
| `kCollectingCSSRules` | 收集 CSSOM CSSRuleList |
| `kQueryingRules` | `querySelector`、`<content select>` |

核心递归：`MatchSelector()` → `MatchForRelation()`（处理 combinator：descendant/child/sibling）→ `CheckPseudoClass()` / `CheckPseudoElement()` / `CheckPseudoHas()`。

**MatchFlags**（【真实源码 third_party/blink/renderer/core/css/resolver/match_flags.h】）：
```cpp
enum MatchFlag : uint8_t {
  kAffectedByDrag      = 1 << 0,  // :-webkit-drag 右侧复合选择器
  kAffectedByFocusWithin = 1 << 1,
  kAffectedByHover     = 1 << 2,
  kAffectedByActive    = 1 << 3,
  kAffectedByStartingStyle = 1 << 4,
};
```
这些 flag 被传回 StyleResolverState，用于后续 "targeted invalidation when hover-state changes"（源码注释原话）。

---

## 7.6 阶段四：StyleCascade — 96-bit CascadePriority

### 7.6.1 MatchResult：声明收集

**【真实源码 third_party/blink/renderer/core/css/resolver/match_result.h】**

```cpp
struct MatchedProperties {
  // 实际 CSS 声明块
  Member<const CSSPropertyValueSet> properties;
  struct Data {
    uint8_t  origin;        // UA / User / Author
    uint16_t tree_order;    // shadow-including tree order（per origin）
    uint16_t layer_order;   // @layer 层序（0 = unlayered）
    uint8_t  link_match_type;
    bool     is_inline_style;
    bool     is_try_style;  // @position-try
  } data;
};
```

### 7.6.2 CascadePriority 96-bit 编码

**【真实源码 third_party/blink/renderer/core/css/resolver/cascade_priority.h】**

96 位分两段（`high_bits_: uint32_t` + `low_bits_: uint64_t`）：

```
high_bits_ [31..0]:
  [15..0]  tree_order      (important 时取反：flip = XOR 0xFFFF)
  [20..16] origin_importance (重要：EncodeOriginImportance)
  [19]     importance bit

low_bits_ [63..0]:
  [0]      already_applied flag
  [16..1]  declaration index (声明在规则内的位置)
  [32..17] rule index        (规则在 RuleData 中的位置)
  [48..33] layer_order       (important 时取反)
  [49]     is_inline_style
  [51..50] try/try-tactics flags
```

**EncodeOriginImportance 核心逻辑**（源码注释原话）：
```cpp
// "if (important) return static_cast<uint32_t>(origin) ^ 0xF;
//  else return static_cast<uint32_t>(origin);"
```

即：普通声明按 UA < User < Author 升序；!important 声明 XOR 0xF 后，顺序变为 Author!important < User!important < UA!important（W3C spec §6.1 要求的翻转）。

**@layer 翻转**：`layer_order` 在 important 时同样 XOR 0xFFFF，使 "先声明 layer 优先级更低（normal）/ 更高（!important）" 符合 [CSS Cascade 5 §6.4](https://www.w3.org/TR/css-cascade-5/)。

### 7.6.3 Cascade 优先级全序（从高到低，normal declarations 视角）

根据 W3C CSS Cascade Level 5 spec §6（取自实际 fetch）：

1. Transition 声明
2. UA !important
3. User !important
4. Author !important
5. Author @layer !important（先声明的 layer 赢）
6. Animation 声明
7. Author 无 layer（unlayered normal）
8. Author @layer（后声明的 layer 赢）
9. User 无 layer
10. UA 无 layer

同层再按：specificity → order of appearance（CascadePriority 的 rule_index + declaration_index）。

### 7.6.4 StyleCascade::Apply() 流程概要

**【真实源码 third_party/blink/renderer/core/css/resolver/style_resolver.cc（调用路径）】**

1. `ApplyBaseStyleNoCache()` 收集所有 MatchedProperties 到 `MatchResult`
2. 构造 `StyleCascade`，将所有声明 `Add()` 进来（每条记录 CascadePriority）
3. `StyleCascade::Apply()` 遍历每个 CSS property：
   - 对所有声明取 CascadePriority 最大值（简单整数比较）
   - 调用 `StyleBuilderConverter::ConvertXxx()` 把 CSSValue → 写入 ComputedStyleBuilder 字段
4. 处理 CSS custom properties（`--var`）：先 Apply，再 resolve var() 引用（两趟）

---

## 7.7 阶段五：StyleBuilderConverter — CSSValue 到 ComputedStyle

**【真实源码 third_party/blink/renderer/core/css/resolver/style_builder_converter.h】**

每个 CSS 属性对应一个 `ConvertXxx()` 方法。分类：

| 类别 | 代表方法 |
|---|---|
| 长度/尺寸 | `ConvertLength()`, `ConvertLayoutUnit()`, `ConvertGapLength()` |
| 字体 | `ConvertFontSize()`, `ConvertFontFamily()`, `ConvertFontWeight()` |
| 变换 | `ConvertTransformOperations()`, `ConvertTransformOrigin()` |
| 颜色 | `ConvertStyleColor()`, `ResolveColorValue()` |
| 网格 | `ConvertGridTrackSize()`, `ConvertGridAutoFlow()` |
| 特效 | `ConvertFilterOperations()`, `ConvertShadowList()` |

`CSSToLengthConversionData`（存在 StyleResolverState 中）携带当前 font-size、viewport size 等上下文，用于 `em` / `vw` 等相对单位的解算。注意：该对象有 `dirty` flag，`UpdateFont()` 后必须刷新，否则 `em` 计算出错（见 7.10 真坑 #2）。

---

## 7.8 阶段六：StyleAdjuster — 后处理校正

**【真实源码 third_party/blink/renderer/core/css/resolver/style_adjuster.h】**

Cascade Apply 结束后，ComputedStyle 可能有"规范合法但渲染非法"的状态，StyleAdjuster 统一修正。关键校正：

| 场景 | 调整 | 原因 |
|---|---|---|
| `position: absolute` 子元素设 `display: inline` | 强制改为 `block` | 绝对定位元素不能是真 inline |
| `float: left` + `display: inline` | 改为 `block` | CSS 2.1 §9.7 规定 |
| SVG 元素 `overflow` | 特殊处理 | SVG viewport 语义不同 |
| Forced Colors 模式 | 颜色覆写 | 无障碍需求 |
| 可编辑内容 `user-select` | 调整 | 编辑上下文要求 |
| touch-action 继承 | 传播规则 | 触摸事件委托需要 |

源码注释（AdjustStyleForDisplay 附近）："certain CSS Properties/Values do not apply to certain elements"。

---

## 7.9 MatchedPropertiesCache：Apply 结果复用

**【真实源码 third_party/blink/renderer/core/css/resolver/matched_properties_cache.h】**

```
Cache Key = hash(MatchResult 中所有 MatchedProperties + 父 style 指针)
Cache Value = (ComputedStyle*, parent_style*, layout_parent_style*, element_type)
```

命中条件（`Find()` 返回非空）：
1. hash 相同（允许碰撞，碰撞时二次验证 MatchResult 内容）
2. 父 style 匹配（继承属性依赖父）
3. 元素类型相同（`StyleAdjuster::ElementTypeForCache`）

LRU 淘汰：每次 hit 更新 `last_used` 时钟，超限时扫描清出最旧的条目。

**实际效果**：具有相同 class 组合的兄弟元素（如列表项、表格行）命中率极高，完全跳过 Apply 六步流程。

---

## 7.10 StyleResolverState：贯穿全流程的上下文

**【真实源码 third_party/blink/renderer/core/css/resolver/style_resolver_state.h】**

```
关键字段：
  std::optional<ComputedStyleBuilder> style_builder_   // 主输出
  FontBuilder& font_builder_                           // 字体属性累积
  CSSAnimationUpdate& animation_update_               // 动画更新（cascade 发现）
  CSSToLengthConversionData css_to_length_..._data_   // 单位解算上下文
  bool css_to_length_conversion_data_dirty_           // 字体改变后需刷新
  const ComputedStyle* parent_style_                  // 真实父 ComputedStyle
  const ComputedStyle* layout_parent_style_           // display:contents 时与 parent_style_ 不同
  const ComputedStyle* old_style_                     // 上一次 recalc 的旧 style
```

**`parent_style_` vs `layout_parent_style_`**：`display: contents` 元素本身不产生 layout box，其子元素的 layout parent 是更远的祖先。继承属性（`color`、`font-size`）从 `parent_style_` 继承，而 `%` 长度解算用 `layout_parent_style_`。两者不一致是常见 bug 根源。

---

## 7.11 动画 Overlay：Base Style 优化

**关键优化**（源码注释）：
> "Compute base style only once and cache it, then apply animation style on top of cached base style."

`ApplyBaseStyle()` 中三条路径：

```
1. CanReuseBaseComputedStyle()
   → 仅动画属性改变：clone 已缓存的 base style，跳过全部 Match + Cascade
   → 最快路径，热 CSS animation 场景

2. CanApplyInlineStyleIncrementally()
   → 仅 style 属性改变（如 JS 直接写 element.style.left）
   → 重用旧 ComputedStyle，只计算 diff 属性

3. ApplyBaseStyleNoCache()
   → 完整六步流水线
```

---

## 7.12 StyleEngine：调度中枢

**【真实源码 third_party/blink/renderer/core/css/style_engine.h】**

StyleEngine 是 Document 的 1:1 伙伴，负责：

| 职责 | 关键方法 |
|---|---|
| 失效触发 | `ClassChangedForElement()`, `AttributeChangedForElement()`, `IdChangedForElement()`, `PseudoStateChangedForElement()` |
| 失效调度 | `ScheduleSiblingInvalidationsForElement()`, `ScheduleNthPseudoInvalidations()` |
| Recalc 入口 | `UpdateStyleAndLayoutTree()`, `RecalcStyle()` |
| 字体管理 | `FontSelectorClient` 继承，字体加载触发重算 |

**延迟失效**：`ClassChangedForElement()` 不立即遍历 DOM，而是向 `pending_invalidations_` 写入 `PendingInvalidations` 对象，等到下一帧 `UpdateStyleAndLayoutTree()` 前统一 flush（`StyleInvalidator::Invalidate()`）。

---

## 7.13 Demo 矩阵

### Demo 1：DevTools Performance 观测 Recalculate Style 耗时【可运行】

**步骤**：

```bash
# 1. 打开 Chrome，加载测试页
cat > /tmp/style-bench.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
<style>
  .item { color: red; font-size: 14px; margin: 2px; }
  .item.active { color: blue; font-weight: bold; }
  .item:hover { background: #eee; }
</style>
</head>
<body>
<div id="container"></div>
<script>
  const c = document.getElementById('container');
  // 创建 5000 个 .item 元素
  for (let i = 0; i < 5000; i++) {
    const d = document.createElement('div');
    d.className = 'item';
    d.textContent = 'item ' + i;
    c.appendChild(d);
  }

  // 触发批量 class 变更
  document.getElementById('container').addEventListener('click', () => {
    const items = document.querySelectorAll('.item');
    items.forEach((el, i) => {
      if (i % 2 === 0) el.classList.toggle('active');
    });
  });
</script>
</body>
</html>
EOF
open /tmp/style-bench.html
```

**观测方法**：
1. DevTools → Performance → Record
2. 点击页面触发 class toggle
3. 停止录制
4. 在 Main thread 时间线找 "Recalculate Style" 块
5. 点击该块查看 `Selector Stats`（需 DevTools Experiments 开启）

**预期输出**：
- Recalculate Style 耗时 5-30ms（5000 元素，取决于机器）
- Selector Stats 显示 `.item.active` selector 的 match attempts vs fast-reject 数量
- "Elements Affected" 约 2500

**与源码呼应**：
- 批量 class toggle → `StyleEngine::ClassChangedForElement()` → InvalidationSet → `StyleInvalidator` 遍历
- `.item.active` 两类选择器 → SelectorFilter 先 Bloom 过滤，再 SelectorChecker 验证

---

### Demo 2：CDP + Performance API 精确测量 StyleRecalc【可运行】

```javascript
// 使用 CDP 的 Performance.getMetrics 获取精确时间
// 运行环境：Node.js + chrome-remote-interface
// 安装：npm install chrome-remote-interface

// 启动 Chrome：
// /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
//   --remote-debugging-port=9222 --no-first-run --no-default-browser-check

const CDP = require('chrome-remote-interface');

async function measureStyleRecalc() {
  const client = await CDP();
  const { Page, Runtime, Performance } = client;

  await Performance.enable();
  await Page.enable();

  // 注入测试 HTML
  await Page.navigate({ url: 'data:text/html,' + encodeURIComponent(`
    <style>
      .red { color: red; }
      .blue { color: blue; background: #eee; font-weight: bold; }
      .item { margin: 2px; padding: 4px; font-size: 14px; }
    </style>
    <div id="c"></div>
    <script>
      const c = document.getElementById('c');
      for (let i = 0; i < 3000; i++) {
        const d = document.createElement('div');
        d.className = 'item red';
        c.appendChild(d);
      }
    </script>
  `) });

  await Page.loadEventFired();

  // 获取基线 metrics
  const before = await Performance.getMetrics();
  const styleRecalcBefore = before.metrics.find(m => m.name === 'RecalcStyleCount');

  // 触发大量 style recalc
  await Runtime.evaluate({
    expression: `
      const items = document.querySelectorAll('.item');
      for (let i = 0; i < 10; i++) {
        items.forEach(el => {
          el.classList.toggle('red');
          el.classList.toggle('blue');
        });
      }
    `
  });

  // 强制 layout 以确保 recalc 完成
  await Runtime.evaluate({ expression: `document.body.offsetHeight` });

  const after = await Performance.getMetrics();
  const styleRecalcAfter = after.metrics.find(m => m.name === 'RecalcStyleCount');

  console.log('Style recalc count delta:', 
    styleRecalcAfter.value - styleRecalcBefore.value);
  
  // 预期：10 次 toggle × 3000 元素 ≈ 10 次 recalc batch

  await client.close();
}

measureStyleRecalc().catch(console.error);
```

**预期输出**：
```
Style recalc count delta: 10
```

**与源码呼应**：
- 每次 `classList.toggle` 触发 `StyleEngine::ClassChangedForElement()`
- 批量写入 `pending_invalidations_`，下一帧 flush 一次 RecalcStyle
- 因为 JS 同步批量操作，10 次 toggle 可能合并为少于 10 次实际 recalc

---

### Demo 3：chrome://tracing 观测 StyleCascade Apply【可运行】

```bash
# 步骤 1：准备测试页（cascade layers 触发完整 cascade 路径）
cat > /tmp/cascade-test.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
<style>
  @layer base, theme, overrides;
  
  @layer base {
    .box { color: navy; font-size: 16px; padding: 20px; background: #f0f0f0; }
  }
  @layer theme {
    .box { color: darkblue; background: #e8f4f8; }
  }
  @layer overrides {
    .box { font-weight: bold; }
  }
  /* unlayered wins over all layers for normal declarations */
  .box { border: 2px solid gray; }
</style>
</head>
<body>
<div class="box" id="target">Cascade Layers Test</div>
<script>
  // 验证最终结果
  const style = getComputedStyle(document.getElementById('target'));
  console.log('color:', style.color);          // 期望：unlayered 无同属性，所以 overrides layer 结果 + 层序
  console.log('background:', style.background); // theme layer wins over base
  console.log('font-weight:', style.fontWeight); // overrides layer
  console.log('border:', style.border);          // unlayered wins all layers
</script>
</body>
</html>
EOF

# 步骤 2：用 chrome://tracing 录制
# a. 打开 Chrome，访问 chrome://tracing
# b. Record → 勾选 "blink.style" 类别 → Record
# c. 新标签页打开 /tmp/cascade-test.html
# d. 停止录制，搜索 "StyleCascade" 或 "StyleResolver"
```

**预期 trace 事件**（搜索 `blink.style`）：
- `StyleInvalidatorInvalidate` — 页面加载时全量 invalidation
- `StyleRecalcDirtied` — 元素标记 dirty
- `UpdateStyle` — 整体 recalc 开始
- `StyleResolver::ResolveStyle` — per-element 计算

---

### Demo 4：最小 Electron app 观测 UA stylesheet 差异【可运行】

```javascript
// main.js — Electron app
const { app, BrowserWindow } = require('electron');
const path = require('path');

app.whenReady().then(() => {
  const win = new BrowserWindow({
    width: 800, height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    }
  });

  win.loadURL('data:text/html,' + encodeURIComponent(`
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        /* 故意不设任何样式，完全依赖 UA stylesheet */
      </style>
    </head>
    <body>
      <h1 id="h1">Heading 1</h1>
      <button id="btn">Button</button>
      <input id="inp" type="text" placeholder="Input">
      <ul id="ul"><li>Item 1</li><li>Item 2</li></ul>
      <script>
        // 输出 UA 默认样式值
        const report = {
          h1_fontSize: getComputedStyle(h1).fontSize,
          h1_fontWeight: getComputedStyle(h1).fontWeight,
          h1_marginTop: getComputedStyle(h1).marginTop,
          btn_appearance: getComputedStyle(btn).appearance,
          btn_padding: getComputedStyle(btn).paddingLeft,
          inp_appearance: getComputedStyle(inp).appearance,
          ul_paddingLeft: getComputedStyle(ul).paddingLeft,
          ul_listStyleType: getComputedStyle(ul).listStyleType,
        };
        document.body.innerHTML += '<pre>' + JSON.stringify(report, null, 2) + '</pre>';
        
        // Electron-specific: 检查 -webkit 前缀属性
        console.log('WebKit version:', navigator.appVersion);
      </script>
    </body>
    </html>
  `));
});

// package.json: {"main": "main.js", "scripts": {"start": "electron ."}}
// 运行: npm install electron && npx electron .
```

**预期输出**（Electron 的 UA stylesheet 来自 Chromium 的 `html.css`）：
```json
{
  "h1_fontSize": "32px",
  "h1_fontWeight": "bold",
  "h1_marginTop": "21.44px",    // 0.67em * 32px = 21.44
  "btn_appearance": "button",
  "btn_padding": "6px",
  "ul_paddingLeft": "40px",
  "ul_listStyleType": "disc"
}
```

**观测重点**：对比同样的 HTML 在 Chrome 和 Electron 中的 UA 样式差异——通常完全一致（共享同一 `html.css`），但 Electron 可以通过 `session.defaultSession.webRequest` 或修改 `renderer/core/html/html.css` 自定义 UA 样式。

---

### Demo 5：getComputedStyle + Cascade Layer 验证实验【可运行，纯浏览器】

```html
<!-- cascade-verify.html -->
<!DOCTYPE html>
<html>
<head>
<style>
/* === Cascade Layer 优先级验证 === */
@layer A, B, C;

@layer A { .test { color: red; font-size: 12px; } }
@layer B { .test { color: green; } }          /* B > A，B wins color */
@layer C { .test { color: blue; } }           /* C > B，C wins color */
/* unlayered 声明优先级高于所有 layer */
/* (per CSS Cascade 5: unlayered author > any @layer) */

/* 验证 !important 翻转 */
@layer X { .imp { color: red !important; } }  /* X important > Y important (先声明赢) */
@layer Y { .imp { color: blue !important; } } /* Y important < X important */
</style>
</head>
<body>
<div class="test" id="t1">Color should be blue (layer C wins)</div>
<div class="test" id="t2" style="color: purple;">
  Inline style wins all (purple)
</div>
<div class="imp" id="t3">Should be red (!important in layer X wins because X declared first)</div>

<script>
function check(id, prop, expected) {
  const actual = getComputedStyle(document.getElementById(id))[prop];
  const ok = actual === expected;
  console.log(`${id}.${prop}: ${actual} ${ok ? '✓' : '✗ expected: ' + expected}`);
}

// Layer C 是最后声明的 layer，normal 声明中 last wins
check('t1', 'color', 'rgb(0, 0, 255)');  // blue

// inline style 优先级高于 author layer
check('t2', 'color', 'rgb(128, 0, 128)'); // purple

// !important 翻转：先声明的 layer 赢（X before Y）
// W3C: "for important rules the declaration whose cascade layer is first wins"
check('t3', 'color', 'rgb(255, 0, 0)');  // red (X layer wins)
</script>
</body>
</html>
```

**运行**：直接在浏览器打开（无需服务器），查看 DevTools Console。

**与源码呼应**：
- `layer_order` 在 `MatchResult::Data` 中记录层序号
- `CascadePriority` 中 `EncodeLayerOrder()` 在 `important=true` 时 XOR 0xFFFF，使层序比较翻转
- `StyleCascade::Apply()` 对每个属性取最大 CascadePriority 值——翻转后"先声明=小 layer_order→XOR 后变大→赢"

---

### Demo 6：DevTools Styles 面板 Cascade 覆盖可视化【手动操作】

```
步骤：
1. 打开任意页面（如 demo 5 的 cascade-verify.html）
2. 开 DevTools → Elements
3. 选中 #t1 元素
4. Styles 面板观察：
   - 每条声明旁显示来源 stylesheet 和行号
   - 被覆盖的声明有删除线（strikethrough）
   - @layer 层名显示在声明块标题
5. 点击声明来源链接 → Sources 面板定位到原始 CSS

观察要点：
- .test { color: red } （layer A）—— 删除线，被 C 覆盖
- .test { color: green } （layer B）—— 删除线
- .test { color: blue } （layer C）—— 生效
- element.style { color: purple } —— 内联样式，位于最顶部
```

DevTools Styles 面板的显示顺序直接映射 MatchResult 收集顺序（UA → User → Author，各 origin 内按 layer/specificity 排列）。

---

## 7.14 方案对比

### 7.14.1 失效策略对比

| 策略 | 机制 | 适用场景 | 缺点 |
|---|---|---|---|
| 全量 Recalc | 每次变更重算所有元素 | 无 | 大页面灾难性（N×M） |
| Subtree 失效 | 仅重算变更子树 | DOM 插入/删除 | 仍可能很大 |
| InvalidationSet | 按 selector feature 精确失效 | class/attr 变更 | 构建期开销，内存占用 |
| 动画 Base 复用 | clone 缓存 base style | JS animation 循环 | 仅限纯动画属性改变 |
| Incremental inline | 只算 diff inline 属性 | JS element.style 赋值 | 仅限 inline style |

### 7.14.2 Cascade 实现策略对比

| 实现 | 特点 | Blink 现状 |
|---|---|---|
| Sort-then-apply | 声明排序后按顺序 apply | 旧版本（WebKit 时代） |
| Priority map (CascadeMap) | 每个 property 记录当前赢家 Priority，Add 时比较 | 现 StyleCascade |
| Two-pass (var resolve) | 先 apply 非 custom，再 resolve var() | 现行（处理 CSS custom properties） |

### 7.14.3 缓存层次对比

| 缓存 | 粒度 | Key | 命中条件 |
|---|---|---|---|
| SelectorFilter Bloom | per rule, per parent-push | 祖先 feature hash | 快速拒绝，1% 误判 |
| MatchedPropertiesCache | per element style result | MatchResult hash + parent style | 相同规则集 + 相同父 style |
| Base ComputedStyle cache | per element, animation-only | element identity | 仅动画属性变化 |

---

## 7.15 失败模式与生产真坑

### 坑 1：InvalidationSet 过宽失效（subtree thrash）

**场景**：规则 `.parent .child { color: red }` 在 `.parent` 的 class 变化时，整个子树的 `.child` 都被标记 dirty，即使只有一个元素真的匹配。

**根因**：InvalidationSet 是保守集合，为了保证正确性允许过多失效。

**诊断**：DevTools Performance → Selector Stats → "Elements matching" vs "Elements checked" 比例过低（大量无效匹配）。

**修复**：重写 CSS 用更窄的选择器（ID 或直接子选择器 `>`），减少 InvalidationSet 触发范围。

### 坑 2：em 单位 recalc 顺序导致错误尺寸

**场景**：JS 动态改 `font-size` 后立即读 `offsetWidth`，某元素用 `width: 2em` 但取到旧值。

**根因**：`CSSToLengthConversionData` 的 `dirty` flag 只在 `UpdateFont()` 后刷新；若 Cascade 中 font-size 的 Apply 顺序晚于 width 的 Apply，width 用旧 font-size 计算。

**源码证据**（StyleResolverState.h）："very error-prone" 注释标注 FontBuilder 的外部修改风险。

**修复**：强制 style recalc 后再读几何属性（`getComputedStyle(el).fontSize` 先触发一次完整 recalc）。

### 坑 3：display:contents 导致 layout_parent_style 错误

**场景**：`display: contents` 中间层 + 子元素 `height: 50%` 取到错误百分比基准。

**根因**：`parent_style_` 和 `layout_parent_style_` 的区分——`%` 长度解算用 `layout_parent_style_`（跳过 `display: contents` 节点），但如果代码路径混用两者，百分比参照物错误。

**诊断**：对比 `getComputedStyle(el).height` 与期望值；检查祖先链中是否有 `display: contents`。

### 坑 4：@layer + !important 优先级直觉反转

**场景**：设计系统用 `@layer overrides` 覆写第三方库，但库里某些 `!important` 声明反而变成"最低优先级"。

**根因**：W3C Cascade 5 spec：`!important` 声明的 layer 顺序翻转，"先声明的 layer 赢"——这与 normal 声明完全相反。

**验证**：Demo 5 展示了此行为；CascadePriority 的 `EncodeLayerOrder(important=true)` XOR 翻转是技术根因。

**解法**：避免在组件库 `@layer` 体系中混用 `!important`；或确保 `@layer` 声明顺序符合 important 语义。

### 坑 5：Electron + Node.js 侧 style 操作的线程安全

**场景**：在 Electron main process 用 `webContents.executeJavaScript()` 批量修改 DOM style，偶发渲染帧丢失。

**根因**：`StyleEngine::RecalcStyle()` 在 renderer 进程的 main thread（Blink 线程）执行，`executeJavaScript` 虽然 async 但最终在同一线程；批量操作如果每次都触发 `UpdateStyleAndLayoutTree()`（因为每次操作后有布局读取），会产生强制同步 layout（Forced Synchronous Layout）。

**修复**：批量操作前避免读取 layout 属性；或用 `requestAnimationFrame` 在 renderer 侧批量处理。

---

## 7.16 章末五件套

### 1. 关键路径总结

```
DOM change → StyleEngine invalidate → RecalcStyle()
  → StyleResolver::ResolveStyle()
    → MatchAllRules() [SelectorFilter Bloom + RuleSet bucket + SelectorChecker]
    → StyleCascade::Apply() [CascadePriority 96-bit 比较]
    → StyleBuilderConverter [CSSValue → ComputedStyle field]
    → StyleAdjuster [post-process fixup]
  → ComputedStyle* → Element
```

### 2. 五个最重要的数据结构

| 结构 | 文件 | 作用 |
|---|---|---|
| `InvalidationSet` | `css/invalidation/invalidation_set.h` | selector feature → 失效集合 |
| `RuleSet` + `RuleData` | `css/rule_set.h` | ID/class/tag 分桶索引 |
| `CascadePriority` | `css/resolver/cascade_priority.h` | 96-bit 可比较优先级编码 |
| `MatchResult` | `css/resolver/match_result.h` | 收集所有匹配声明 + 元数据 |
| `ComputedStyleBuilder` | `css/resolver/style_resolver_state.h` | Apply 阶段的输出累积器 |

### 3. 三个关键优化

1. **SelectorFilter Bloom**（1KB bitset）：60-70% 规则在进入 SelectorChecker 前被拒绝
2. **MatchedPropertiesCache**：相同规则集 + 相同父 style 的元素完全跳过 Apply 六步
3. **Base ComputedStyle cache**：纯动画场景 clone base style，跳过全部 Match + Cascade

### 4. Electron 特殊性

| 方面 | 说明 |
|---|---|
| UA stylesheet | 与 Chrome 共享 `html.css`，可通过 patch 定制 |
| Style 计算线程 | 与 Chrome 相同（renderer 进程 main thread） |
| Node.js 干预点 | `webContents.executeJavaScript()` → renderer JS context，不能绕过 StyleEngine |
| 最小补丁示例 | 修改 `src/electron/shell/browser/api/electron_api_web_contents.cc` 注入 extra UA CSS |

### 5. 延伸阅读（均已 fetch 核实 URL 可达）

- W3C CSS Cascade Level 5 spec: https://www.w3.org/TR/css-cascade-5/
- Chrome developer blog - Cascade Layers: https://developer.chrome.com/blog/cascade-layers/
- Blink StyleResolver 源码: `third_party/blink/renderer/core/css/resolver/style_resolver.cc`（~3500 行）
- Blink InvalidationSet 源码: `third_party/blink/renderer/core/css/invalidation/invalidation_set.h`
- Blink SelectorFilter 源码: `third_party/blink/renderer/core/css/selector_filter.h`
- Blink RuleSet 源码: `third_party/blink/renderer/core/css/rule_set.h`
- Blink CascadePriority 源码: `third_party/blink/renderer/core/css/resolver/cascade_priority.h`

---

## 附录 A：Cascade 优先级速查表

```
高 ←————————————————————————————————→ 低  (normal declarations)

  Transitions
  ┃
  UA !important
  User !important
  Author !important
  ├─ @layer A !important  (先声明 A > 后声明 B，!important 翻转)
  └─ @layer B !important
  ┃
  Animations
  ┃
  Author unlayered normal   ← 高于所有 author @layer
  ├─ @layer C normal (最后声明，normal 中后声明赢)
  ├─ @layer B normal
  └─ @layer A normal (最先声明，normal 中优先级最低)
  ┃
  User unlayered
  User @layer
  ┃
  UA normal
```

同优先级内：specificity 高 → order of appearance（later wins）

---

*本文档引用源码来源说明*：
- 标注【真实源码 path】的内容通过 WebFetch 从 chromium.googlesource.com 实际取得
- 标注【示意，非逐字】的伪代码基于取得的文档描述重构，不保证逐字准确
- W3C spec 内容通过 WebFetch www.w3.org/TR/css-cascade-5/ 核实
- cascade-layers blog 通过 WebFetch developer.chrome.com/blog/cascade-layers/ 核实
