---
title: "Layout / LayoutNG（Chromium 域）"
slug: "7-08"
collection: "tech-library"
group: "chromium内核"
order: 7008
summary: "TL;DR LayoutNG 是 Blink 在 BlinkNG（2014 起的多年重写）框架下对 layout 阶段的彻底重写。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:48:10.000Z"
updatedAt: "2026-06-12T11:48:10.000Z"
---
> **TL;DR**
> LayoutNG 是 Blink 在 BlinkNG（2014 起的多年重写）框架下对 layout 阶段的彻底重写。核心一句话：**把 layout 从"在一棵可变 layout tree 上原地改写"重构为"函数式的 layout algorithm：输入 `(BlockNode, ConstraintSpace, BreakToken)` → 输出 immutable 的 `PhysicalFragment`"**。
> 旧架构里每个 LayoutObject 同时存输入（available size、float 位置）和输出（最终宽高），导致四类系统性 bug：契约不清的 correctness bug、under-invalidation、hysteresis（layout 不幂等）、over-invalidation 引发的 performance cliff；嵌套 flex/grid 的两趟测量更会把复杂度推到 O(2ⁿ)。
> LayoutNG 用三件事根治：① 显式分离 input（`ConstraintSpace`）与 output（`PhysicalFragment`）；② `LayoutResult` 缓存 measure/layout 两个 pass，把复杂度拉回 O(n)；③ fragment tree 是 immutable、只读、可跨线程、可复用的 layout 产物，直接喂给 pre-paint / paint / hit-test。Block fragmentation（多列/分页）从 Chrome 102 起在 fragment tree 里用 `BreakToken` 原生表达，Chrome 108 起 legacy layout engine 彻底退役。
> Electron 完整继承这套引擎，无独立 layout 代码；唯一相关差异是 print-to-PDF / 离屏渲染路径会更频繁触发 fragmentation，以及 main process 通过 `executeJavaScript` 批量读几何属性时极易制造 forced synchronous layout。

---

## 前置依赖

| 需要掌握 | 用于理解 |
|---|---|
| 第 6 章 HTML 解析与 DOM | layout 的输入是 DOM + LayoutObject 树 |
| 第 7 章 Style 计算 | `ComputedStyle` 是 layout algorithm 的输入之一 |
| CSS 盒模型（margin/border/padding/content box）与 BFC | ConstraintSpace、BfcOffset、ExclusionSpace 的语义基础 |
| writing-mode / logical vs physical 坐标 | LayoutNG 全程用 logical（inline/block）算、最后转 physical |
| Blink GC（`GarbageCollected<T>` / `Member<T>`） | Fragment / LayoutResult 的生命周期与 oilpan 关系 |
| RenderingNG 流水线五阶段（style→layout→pre-paint→paint→commit） | LayoutNG 在整条流水线的位置 |

---

## 8.1 设计考古：layout 从何而来，为何要 NG

### 8.1.1 谱系：KHTML → WebKit → Blink legacy layout

Blink 的 layout 代码血统极老。RenderingNG 官方文档（BlinkNG 篇）原话：

> "By 2014 it was definitely showing its age. In that year, we embarked on a set of ambitious projects under the banner of what we're calling BlinkNG."

代码可一直追溯到 WebKit、再到 1998 年的 KHTML。"NG" = **Next Generation**。早期的 layout 是经典的 reflow 模型：一棵 `RenderObject`（后改名 `LayoutObject`）树，每个节点既是输入容器又是输出容器，`layout()` 方法递归调用、原地写回 `m_frameRect`（最终位置和尺寸）。

来源（已 WebFetch 核实）：
- RenderingNG deep-dive: LayoutNG — https://developer.chrome.com/docs/chromium/layoutng
- RenderingNG deep-dive: BlinkNG — https://developer.chrome.com/docs/chromium/blinkng
- LayoutNG 设计文档（仓库内）— third_party/blink/renderer/core/layout/layout_ng.md（raw.githubusercontent.com 取得）

### 8.1.2 legacy 的四宗罪（官方归因）

LayoutNG 官方文档把旧"mutable tree"架构的问题描述为：

> "Each object in the layout tree contained input information, such as the available size imposed by a parent ... and output information, for example, the final width and height."

输入输出混在同一对象，直接导致四类 bug（官方 deep-dive 原文归纳）：

1. **Correctness**：组件间契约不清——某个 LayoutObject 读了它本不该读的祖先/兄弟状态，结果依赖遍历顺序。
2. **Under-invalidation**：树的某部分该标 dirty 却没标，复用了过期的输出。
3. **Hysteresis（迟滞）**：layout 不幂等——同样输入跑两次结果不同，因为第一趟把中间状态写回了节点，第二趟读到被污染的值。官方给的例子是 Chrome 92 及以下的一个 bug，Chrome 93 切到新架构后修复。
4. **Performance cliff**：为了对抗 under-invalidation 而 over-invalidate，又把整棵树重算，掉进性能悬崖。

### 8.1.3 复杂度炸弹：O(n) → O(2ⁿ)

两趟布局（flex / grid 需要先 measure 内容再 layout）是复杂度爆炸的根源。官方原文：

> "for a two-pass layout ... this can potentially result in complexity of O(2n)"

嵌套时指数累积——三层嵌套 flex 在 measure 阶段会把子树各测一遍，整体逼近 **O(2ⁿ)**。官方明确给出 Chrome 92 的 grid 例子是指数级，Chrome 93 切新架构后：

> "brings the complexity back to O(n), resulting in predictably linear performance"

LayoutNG 的解法不是"算得快"，而是**把每一次 measure / layout 的结果显式缓存进 `LayoutResult`**，使得同一节点在同一 `ConstraintSpace` 下不被重复计算（详见 8.6）。

### 8.1.4 三条核心设计动机（提炼）

1. **input/output 分离 + 输出 immutable**：layout tree 退化为"持有输入和缓存指针"，真正的几何输出搬进全新的、只读的 **fragment tree**。官方原文："we generate a completely new, immutable object called the fragment tree"。
2. **layout 即 function**：BlinkNG 六原则之一是"functional stages with deterministic outputs / constant inputs during execution"。每种盒子类型是一个 `LayoutAlgorithm`，吃固定输入吐固定输出，无副作用地访问外部状态。
3. **fragment tree 是后续所有阶段的唯一输入**：paint、hit-test、accessibility、fragmentation 全都遍历 fragment tree，而不再去 inspect 可变的 LayoutObject。这让 layout 可中断、可缓存、未来可并行/跨线程。

### 8.1.5 一个容易踩的命名陷阱：`NG` 前缀已被去掉

历史上类名都带 `NG` 前缀（`NGConstraintSpace` / `NGPhysicalFragment` / `NGBlockNode` / `NGLayoutResult`）。**随着 legacy layout 在 Chrome 108 退役、NG 成为唯一实现，前缀已被批量删除**：现在 main 分支上是 `ConstraintSpace` / `PhysicalFragment` / `BlockNode` / `LayoutResult`，目录也从 `core/layout/ng/` 拍平回 `core/layout/`。

> 「待核」：精确的"去 NG 前缀"批量 rename CL 号未逐一核实；但 main 分支源码（8.2~8.6 的 fetch）已确认现状是无前缀。老博客 / 老 design doc / fragmentation 篇仍用 `NGBlockBreakToken`、`NGPhysicalBoxFragment` 等旧名，阅读时按"NGFoo == Foo"对应即可。本章源码引用一律以 **main 分支无前缀名** 为准，引用旧文档处保留其原文 NG 名并标注。

---

## 8.2 核心数据结构全景

LayoutNG 的"输入 → 算法 → 输出"三元组对应三个核心类型，外加缓存与分页：

```
            ┌─────────────────────────────────────────────┐
            │              输入 (Inputs)                    │
            │  BlockNode        — 要布局的盒子 + ComputedStyle│
            │  ConstraintSpace  — 可用空间 / BFC / 排斥区     │
            │  BreakToken       — 分页/多列的续算位置(可选)    │
            └───────────────────┬─────────────────────────┘
                                │
                                ▼
            ┌─────────────────────────────────────────────┐
            │       LayoutAlgorithm<Node,Builder,Token>    │
            │  ├─ Block / Inline / Flex / Grid / Table ...  │
            │  ├─ ComputeMinMaxSizes()  ← measure pass      │
            │  └─ Layout() → const LayoutResult*            │
            └───────────────────┬─────────────────────────┘
                                │
                                ▼
            ┌─────────────────────────────────────────────┐
            │              输出 (Outputs)                   │
            │  LayoutResult     — 算法返回值(含状态+缓存键)   │
            │   └─ PhysicalFragment  ← immutable 几何输出    │
            │        ├─ Size() / Style() / GetLayoutObject() │
            │        └─ Children()  → 子 fragment + 偏移      │
            └─────────────────────────────────────────────┘
                                │
                                ▼
              pre-paint / paint / hit-test / a11y 遍历 fragment tree
```

| 类型 | main 分支文件 | 旧名 | 角色 |
|---|---|---|---|
| `ConstraintSpace` | `core/layout/constraint_space.h` | NGConstraintSpace | layout 的"输入空间" |
| `BlockNode` | `core/layout/block_node.{h,cc}` | NGBlockNode | 被布局的盒子（LayoutBox 的 NG 视图） |
| `LayoutAlgorithm` | `core/layout/layout_algorithm.h` | NGLayoutAlgorithm | 所有布局算法基类模板 |
| `LayoutResult` | `core/layout/layout_result.h` | NGLayoutResult | 算法返回值 + 缓存键 |
| `PhysicalFragment` | `core/layout/physical_fragment.h` | NGPhysicalFragment | immutable 几何输出 |
| `BreakToken` | `core/layout/block_break_token.h` | NGBlockBreakToken | 分页/多列续算 token |
| `ExclusionSpace` | `core/layout/exclusions/exclusion_space.h` | NGExclusionSpace | float 排斥区记录 |

---

## 8.3 输入侧 ① ：ConstraintSpace —— layout 的"可用空间"

`ConstraintSpace` 是父算法递给子算法的"你能在这块空间里布局"的约束包。官方一句话定义（data-structures 篇）：constraint space 是"the available layout space where the algorithm produces output ... one of three core inputs to any layout algorithm"。

**【真实源码 chromium/chromium@third_party/blink/renderer/core/layout/constraint_space.h】**（以下声明逐字取自 main 分支 WebFetch）

```cpp
// 类顶注释（逐字）:
// The ConstraintSpace represents a set of constraints and available space
// which a layout algorithm may produce a LogicalFragment within.

class CORE_EXPORT ConstraintSpace final {
  DISALLOW_NEW();                 // 栈对象，不进 oilpan heap（轻量、随算法栈生灭）
```

关键 accessor（逐字）：

```cpp
// —— 可用空间 / 百分比解析基准（logical 坐标：inline-size × block-size）——
LogicalSize AvailableSize() const { return available_size_; }
LogicalSize PercentageResolutionSize() const { return percentage_size_; }

// —— BFC 偏移：当前盒子在其 block formatting context 中的位置 ——
BfcOffset GetBfcOffset() const { return bfc_offset_; }

// —— writing-mode / direction（决定 logical↔physical 映射）——
TextDirection Direction() const {
  return static_cast<TextDirection>(bitfields_.direction);
}
WritingMode GetWritingMode() const {
  return static_cast<WritingMode>(bitfields_.writing_mode);
}

// —— 分页 / 多列 ——
FragmentationType BlockFragmentationType() const {
  return rare_data_ ? static_cast<FragmentationType>(
                          rare_data_->block_direction_fragmentation_type)
                    : kFragmentNone;
}
bool HasBlockFragmentation() const {
  return BlockFragmentationType() != kFragmentNone;
}

// —— 缓存槽：measure pass 与 layout pass 分别缓存 ——
LayoutResultCacheSlot CacheSlot() const {
  return static_cast<LayoutResultCacheSlot>(bitfields_.cache_slot);
}

// —— float 排斥区（决定文本绕排）——
const ExclusionSpace& GetExclusionSpace() const {
  return exclusion_space_;
}
```

私有成员布局（逐字）：

```cpp
LogicalSize available_size_;        // 父给的可用 inline/block 尺寸
LogicalSize percentage_size_;       // % 长度的解析基准
BfcOffset bfc_offset_;              // 在 BFC 中的偏移
ExclusionSpace exclusion_space_;    // float 排斥区
Member<const RareData> rare_data_;  // 不常用字段（分页类型等）懒分配，省内存
Bitfields bitfields_;              // direction/writing_mode/cache_slot 等位压缩
```

**逐行要点**：

- **`DISALLOW_NEW()`**：ConstraintSpace 是**栈对象**，随 `LayoutAlgorithm` 调用栈创建销毁——它是"这一次布局的输入快照"，不需要进 GC heap。这正是 input/output 分离的体现：输入是临时的、不可变的、每次新建。
- **logical 坐标贯穿输入**：`available_size_` / `percentage_size_` 都是 `LogicalSize`（inline-size × block-size），不是 width × height。writing-mode 的差异在 algorithm 内部被"折叠"掉，等到产出 `PhysicalFragment` 时才转成 physical（见 8.5）。这是 LayoutNG 处理竖排/RTL 的统一策略。
- **`bfc_offset_` 与 `exclusion_space_`**：BFC（block formatting context）相关字段直接进 ConstraintSpace，使得"float 在哪、当前盒子在 BFC 里的偏移"成为**显式输入**而非旧架构里"从树上某处捞出来的可变状态"。
- **`rare_data_` 懒分配**：分页类型这类不常用字段塞进 `RareData`，普通盒子的 ConstraintSpace 不背这份内存。这是 Blink 全代码库的常见省内存手法（第 7 章 InvalidationSet 的 hybrid backing 同思路）。
- **`cache_slot`**：measure 与 layout 两个 pass 用**不同缓存槽**——这是 O(n²)→O(n) 的关键，下一节缓存机制会展开。

### ConstraintSpace 的"封闭性"契约

block_node 源码注释（data-structures 篇引述）有一句极重要的契约：

> "The current layout should not access any information outside this set, this will break invariants in the system."

即：算法**只能**通过 `(BlockNode, ConstraintSpace, BreakToken)` 这三个入参拿信息，不许偷看树上别处。这正是 BlinkNG"constant inputs during execution"原则落到 layout 的具体形态——也是为什么 LayoutNG 才可能做缓存、可中断、未来可并行。

---

## 8.4 算法侧：LayoutAlgorithm —— "每种盒子一个算法"

LayoutNG 把"如何布局一个盒子"抽象成 `LayoutAlgorithm` 模板。官方原文（data-structures 篇）："For each different type of layout, we have a LayoutAlgorithm."

**【真实源码 chromium/chromium@third_party/blink/renderer/core/layout/layout_algorithm.h】**（逐字）

```cpp
// 类顶注释（逐字）:
// Base class template for all layout algorithms. Subclassed template
// specializations (actual layout algorithms) are required to define the
// following two functions...

template <typename InputNodeType,
          typename BoxFragmentBuilderType,
          typename BreakTokenType>
class CORE_EXPORT LayoutAlgorithm
```

两个子类必须实现的方法（逐字）：

```cpp
// measure pass: 算内容固有 min/max 尺寸（不看 width 属性）
MinMaxSizesResult ComputeMinMaxSizes(const MinMaxSizesFloatInput&);

// layout pass: 在 ConstraintSpace 内真正布局子节点
const LayoutResult* Layout();
```

算法的输入参数包（逐字）：

```cpp
struct LayoutAlgorithmParams {
  STACK_ALLOCATED();                    // 同样是栈对象

 public:
  LayoutAlgorithmParams(BlockNode node,
                        const FragmentGeometry& fragment_geometry,
                        const ConstraintSpace& space)
      : node(node), fragment_geometry(fragment_geometry), space(space) {}

  BlockNode node;                                  // 被布局的盒子
  const FragmentGeometry& fragment_geometry;       // 已算好的 border/padding/尺寸框架
  const ConstraintSpace& space;                    // 输入空间（见 8.3）
  const BlockBreakToken* break_token = nullptr;    // 分页续算（见 8.7）
  const EarlyBreak* early_break = nullptr;         // 最优断点提示
  const ColumnSpannerPath* column_spanner_path = nullptr;
  const LayoutResult* previous_result = nullptr;   // 上次结果（增量/续算用）
  const HeapVector<Member<EarlyBreak>>* additional_early_breaks = nullptr;
};
```

**逐行要点**：

- **三模板参数 `<InputNodeType, BoxFragmentBuilderType, BreakTokenType>`**：不同布局类型用不同 Node / Builder / Token 三件套实例化。block 用 `BlockNode + BoxFragmentBuilder + BlockBreakToken`，inline 走另一套（inline 有独立的 cursor/items 表示，见 8.5.2）。
- **measure 与 layout 是两个独立方法**：`ComputeMinMaxSizes()`（measure）算"内容最小能多窄、最大想多宽"，不考虑 `width` 属性；`Layout()` 才在确定的空间里安置子节点。flex/grid 之所以是两趟，就是先对子项 `ComputeMinMaxSizes` 再 `Layout`——两趟各自走不同 cache slot（8.6），避免指数爆炸。
- **算法是栈对象、即用即弃、无状态泄漏**：`STACK_ALLOCATED()` + 输入全 const 引用。算法跑完只留下返回的 `const LayoutResult*`，自己不残留任何可变状态——这是"layout 即 function"的物理保证。

### 8.4.1 算法分派：DetermineAlgorithmAndRun

具体用哪个算法，由 `BlockNode::Layout` 内部按盒子类型分派。

**【真实源码 chromium/chromium@third_party/blink/renderer/core/layout/block_node.cc】**（逐字片段）

```cpp
// DetermineAlgorithmAndRun() 内按 LayoutObject 类型选择算法:
if (box.IsFlexibleBox()) {
  CreateAlgorithmAndRun<FlexLayoutAlgorithm>(params, callback);
} else if (box.IsTable()) {
  CreateAlgorithmAndRun<TableLayoutAlgorithm>(params, callback);
}
// ... 其余类型类似分派
// MathML 元素走单独路径 DetermineMathMLAlgorithmAndRun()，
// 内部再细分 MathFractionLayoutAlgorithm / MathRowLayoutAlgorithm 等
```

主要算法家族（按文件名归纳，main 分支 `core/layout/`）：

| 算法 | 处理对象 | 备注 |
|---|---|---|
| `BlockLayoutAlgorithm` | 块容器（最常见） | 处理 BFC、margin collapsing、float |
| `InlineLayoutAlgorithm` | 行内内容 | 配合 `InlineCursor` / items（8.5.2） |
| `FlexLayoutAlgorithm` | `display:flex` | 两趟 measure→layout |
| `GridLayoutAlgorithm` | `display:grid` | 两趟，曾是 O(2ⁿ) 重灾区 |
| `TableLayoutAlgorithm` | `display:table` | Chrome 106 起支持 fragmentation |
| `ReplacedLayoutAlgorithm` | `<img>`/`<video>` 等替换元素 | 固有尺寸 |
| `MathRowLayoutAlgorithm` 等 | MathML | 独立分派路径 |

---

## 8.5 输出侧：PhysicalFragment —— immutable 的几何产物

`PhysicalFragment` 是 layout 阶段的**唯一只读产物**，也是 pre-paint/paint/hit-test 的**唯一输入**。

**【真实源码 chromium/chromium@third_party/blink/renderer/core/layout/physical_fragment.h】**（注释逐字）

```cpp
// 类顶注释（逐字）:
// The PhysicalFragment contains the output geometry from layout. The fragment
// stores all of its information in the physical coordinate system for use by
// paint, hit-testing etc.
//
// (关于 immutability 的演进意图，注释逐字:)
// Once we have transitioned fully to LayoutNG it should be a const pointer
// such that paint/hit-testing/etc don't modify it.
```

关键方法（逐字 / 近逐字，取自 main 分支 WebFetch）：

```cpp
FragmentType Type() const;            // box 还是 line box
PhysicalSize Size() const;            // border-box 尺寸（physical: width×height）
const ComputedStyle& Style() const;   // 样式（来自 style 阶段）
const LayoutObject* GetLayoutObject() const;  // 生成它的 LayoutObject（仅 CSS box）

bool IsBox() const;
bool IsLineBox() const;
bool IsContainer() const;
bool IsCSSBox() const;                // 是否对应 CSS box tree 节点
bool IsLayoutObjectDestroyedOrMoved() const;  // 校验生成对象是否还有效
```

子节点遍历（data-structures 篇原文）："A fragment holds a list of child fragments and their offsets" —— main 分支提供 `PostLayoutChildLinkList` 迭代子 fragment，并自动解析到 post-layout 的最新代、跳过已销毁对象。

**逐行 / 设计要点**：

1. **physical 坐标，从此不再 logical**。fragment tree 篇原文：
   > "All coordinates and sizes associated with an PhysicalFragment are physical, i.e. pure left/top offsets from the parent fragment, and sizes are expressed with widths and heights (not inline-size / block-size)."

   即：算的时候全程 logical（8.3），**只在产出 fragment 这一刻把竖排/RTL 折叠成统一的 left/top + width/height**。下游 paint 不需要再关心 writing-mode。

2. **immutable 的三条硬约束**（data-structures 篇逐字）：
   > "After layout, each fragment becomes immutable and is never changed again."
   >
   > "We don't: Allow any 'up' references in the tree. (A child can't have a pointer to its parent.) 'bubble' data down the tree"

   - **no up-reference**：子 fragment 不持有父指针 → 同一个子 fragment 可被多个上下文/多棵树共享。
   - **no bubble**：数据不沿树冒泡 → 任意子树可独立复用，不依赖兄弟。
   - 这两条是 fragment 可缓存、可跨线程、可"只重建脊柱（spine）"的根基。

3. **fragment ≠ element，可一对多**。data-structures 篇："typically one fragment per element ... though printing and multi-column contexts create multiple fragments per element"。一个 `<div>` 跨两列 → 两个 fragment，各带 break token（8.7）。

4. **paint/hit-test 改为遍历 fragment tree**。LayoutNG 文档强调：fragment tree 支持直接 depth-first 遍历做 painting / hit-testing，而不再去 inspect 可变 LayoutObject。这是"fragment tree 是后续阶段唯一输入"的落地。

### 8.5.1 fragment tree 是怎么自底向上长出来的

NG 引擎深度优先遍历 CSS box tree。官方原文：

> "When all descendants of a node are laid out, the layout of that node can be completed by producing an NGPhysicalFragment and returning to the parent layout algorithm, which adds the fragment to its list of child fragments and generates a fragment for itself with all its child fragments inside, creating a fragment tree for the entire document."

即每个算法**先布局完所有后代 → 产出自己的 PhysicalFragment（内含子 fragment + 偏移）→ 返回父算法**，自底向上拼出整棵 fragment tree。这与旧 reflow "递归 layout 原地写回"的差别在于：每层只**新建**输出、不**改写**输入。

### 8.5.2 inline 内容的特殊表示：扁平 items 而非树

行内内容（文本 / inline box）不用树，而用**扁平列表**。data-structures 篇原文：inline content "uses a flat list representation—a tuple of (object, number of descendants)"，并指出收益是"faster iteration, easier querying, and memory efficiency critical for text rendering performance"。遍历入口是 `ng_inline_cursor.h`（`InlineCursor`）。

> 为什么 inline 要特殊：一行文本里有几十上百个 run（字形、emphasis、bidi 段），用树表示既费内存又难线性扫；扁平 (item, descendant_count) 数组可以 O(1) 跳过子树、cache-friendly 地顺序遍历，这对文本这种超高频对象是刚需。

---

## 8.6 缓存与复杂度：LayoutResult 如何把 O(2ⁿ) 拉回 O(n)

`LayoutResult` 是 `LayoutAlgorithm::Layout()` 的返回类型，既装输出（`PhysicalFragment`），又装**缓存键**（生成它的 `ConstraintSpace`）。

**【真实源码 chromium/chromium@third_party/blink/renderer/core/layout/layout_result.h】**（注释逐字）

```cpp
// 类顶注释（逐字）:
// The LayoutResult stores the resulting data from layout. This includes
// geometry information in form of a PhysicalFragment, which is kept around for
// painting, hit testing, etc., as well as additional data which is only
// necessary during layout and stored on this object.
```

关键方法（逐字）：

```cpp
const PhysicalFragment& GetPhysicalFragment() const {
  DCHECK(physical_fragment_);
  DCHECK_EQ(kSuccess, Status());          // 只有成功才有 fragment
  return *physical_fragment_;
}

EStatus Status() const {
  return static_cast<EStatus>(bitfields_.status);
}

// 关键：返回"生成本结果的 ConstraintSpace"，作为缓存比对的依据
const ConstraintSpace& GetConstraintSpaceForCaching() const {
  return space_;
}

LayoutUnit BfcLineOffset() const;
const std::optional<LayoutUnit> BfcBlockOffset() const;
```

`EStatus` 枚举（逐字摘要）——layout 不再只有"成功"，还有一组"需要重试 / 需要更早断点"的状态，这是可中断 layout 的体现：

```cpp
kSuccess                                      // 正常完成
kBfcBlockOffsetResolved                       // BFC 偏移已确定
kNeedsEarlierBreak                            // 分页需要更早的断点
kOutOfFragmentainerSpace                      // fragmentainer 空间耗尽
kNeedsLineClampRelayout                       // line-clamp 需重排
kDisableFragmentation                         // 禁用分页
kNeedsRelayoutWithNoChildScrollbarChanges     // 滚动条变化需重排
kTextBoxTrimEndDidNotApply                    // text-box-trim 未生效
kAlgorithmSpecific1                           // 互斥结果共用值
```

关键字段：`space_`（生成它的 ConstraintSpace，缓存键）、`physical_fragment_`（几何输出）、`bfc_offset_`、`rare_data_`、`bitfields_`。

### 8.6.1 缓存命中的真实代码路径

**【真实源码 chromium/chromium@third_party/blink/renderer/core/layout/block_node.cc】**（`BlockNode::Layout` 逐字片段）

```cpp
const LayoutResult* BlockNode::Layout(
    const ConstraintSpace& constraint_space,
    const BlockBreakToken* break_token,
    const EarlyBreak* early_break,
    const ColumnSpannerPath* column_spanner_path) const {

  // ① 先取上次结果，预热 exclusion space（float 排斥区）
  if (const LayoutResult* previous_result =
          box_->GetCachedLayoutResult(break_token)) {
    constraint_space.GetExclusionSpace().PreInitialize(
        previous_result->GetConstraintSpaceForCaching()
            .GetExclusionSpace());
  }

  // ② 调 box_->CachedLayoutResult(...) 校验缓存是否可复用，
  //    传入 &fragment_geometry, &cache_status 等出参
  //    cache_status ∈ {kHit, kNeedsSimplifiedLayout, kNeedsLayout, ...}

  // ③ measure 与 layout 分槽：
  //    当 CacheSlot()==kMeasure 时走 scrollbar-freezing 逻辑，
  //    当 ==kLayout 时检查 broken spine 等
}
```

cache slot 区分逻辑（逐字）：

```cpp
if ((cache_status == LayoutCacheStatus::kHit ||
     cache_status == LayoutCacheStatus::kNeedsSimplifiedLayout) &&
    needed_layout &&
    constraint_space.CacheSlot() == LayoutResultCacheSlot::kLayout &&
    box_->HasBrokenSpine() && !ChildLayoutBlockedByDisplayLock()) {
  // ... 命中但 spine 已断时的处理
}
// measure pass 分支:
if (constraint_space.CacheSlot() == LayoutResultCacheSlot::kMeasure) {
  // scrollbar freezing during measure
}
```

**逐行讲清缓存怎么救复杂度**：

1. **缓存键就是 ConstraintSpace**。`GetCachedLayoutResult` / `CachedLayoutResult` 拿当前 `constraint_space` 和缓存里 `LayoutResult` 的 `GetConstraintSpaceForCaching()` 比——**只要输入空间没变，就直接复用上次的 fragment，整棵子树不重算**。这正是 input/output 分离带来的"红利"：输入是一个可比较的显式对象，命中判断 O(1)。

2. **measure / layout 两个 slot 互不污染**。`LayoutResultCacheSlot::kMeasure` 与 `kLayout` 让"测量结果"和"布局结果"分别缓存。flex/grid 的两趟里，measure pass 的结果被 layout pass 复用、子树的 measure 结果被父的 measure 复用——把原本指数级的重复测量**摊平成每节点测一次**。官方原话：caching of measure and layout passes "brings the complexity back to O(n)"。

3. **`kNeedsSimplifiedLayout` = 增量 layout**。不是全 hit 也不是全 miss，而是"只需简化重排"——典型是只改了不影响子树几何的属性时，复用大部分子 fragment，只重建"脊柱（spine）"。data-structures 篇：大多数变更"affect only the spine of the tree"，正是靠 fragment 的 no-up-ref / no-bubble（8.5）才能局部复用。

4. **`box_->HasBrokenSpine()`**：spine 复用有前提——如果脊柱本身断了（祖先链尺寸变了），就不能简单复用子 fragment，得回退到更完整的重排。这行代码就是"局部复用"与"正确性"的边界守卫。

---

## 8.7 Block Fragmentation：多列 / 分页在 fragment tree 里原生表达

这是 LayoutNG 相对 legacy 最大的能力跃迁。来源：RenderingNG deep-dive: LayoutNG block fragmentation — https://developer.chrome.com/docs/chromium/renderingng-fragmentation（已 WebFetch）。

### 8.7.1 legacy 分页为什么是"假分页"

官方原文描述旧引擎：它把内容"lays out everything into a tall strip whose width is the inline-size of a column or page, and height is as tall as it needs to be"，然后**布局完之后**再靠 clip + translate 切成列/页。后果（官方逐条）：

- text-shadow / transform 在列边缘被错误裁剪；
- monolithic 内容（可滚动容器、图片）被"brutally sliced rather than overflowing"；
- 不支持 `break-before:avoid`、不支持最优非强制断点；
- "No flex or grid fragmentation support at all"。

### 8.7.2 NG 分页：布局时就断，用 BreakToken 续算

LayoutNG 在**真正布局过程中**分页。核心数据结构（fragmentation 篇，旧名）：

- **`NGBlockBreakToken`**（main 分支 `BlockBreakToken`）：原文"It contains all the information needed to resume layout correctly in the next fragmentainer."它挂在 `NGPhysicalBoxFragment` 上，自身组成一棵 token 树，让续算能在下一个 fragmentainer 精确恢复。
- **fragmentainer**：原文"a column in multi-column layout, or a page in paged media" —— **不是 DOM 元素**，是"装 fragment 的容器槽"。
- **`NGEarlyBreak`**（`EarlyBreak`）：记录候选最优断点，带评分（"perfect" → "last-resort"），用于满足 `break-before/avoid` 等 CSS 断点避让。

流程（官方原文）：算法深度优先遍历 box tree，空间耗尽时"we then produce fragments for the nodes that we visited, and return all the way up to the fragmentation context root"，然后在下一个 fragmentainer 用 break token 续算。子内容拿到的 ConstraintSpace 会告知"the laid-out block-size of the fragmentainer, and the current block offset into it"——即 8.3 里 `HasBlockFragmentation()` / `BlockFragmentationType()` 字段的用途。

回看 8.6 的 `EStatus`：`kOutOfFragmentainerSpace`（空间耗尽）、`kNeedsEarlierBreak`（要更早断点）正是分页驱动 relayout 的状态信号——layout 在这里体现为"可中断、可回退重试"。

### 8.7.3 上线时间线（官方 fragmentation 篇逐条核实）

| Chrome 版本 | 能力 |
|---|---|
| **102** | 核心分页：block container、line、float、out-of-flow positioning |
| **103** | flex / grid fragmentation |
| **106** | table fragmentation |
| **108** | printing 支持；**legacy layout engine 不再用于 layout** |

Chrome 108 是个分水岭：自此 NG 成为唯一 layout 实现，前缀去除（8.1.5）正发生在这之后。

---

## 8.8 Demo 矩阵

> 说明：Chromium 全量 build 不现实，以下 demo 以"真实引擎上观测 + 小实验"为主。标【可运行】的给完整可跑代码；标【手动观测】的给完整步骤与预期；偏 build 的 Electron patch 给走读 + 最小补丁示例。命令默认 macOS，Chrome 路径按需替换。

### Demo 1：layout thrashing —— 量化 forced synchronous layout 的代价【可运行，纯浏览器】

这是本章最核心的 demo：直接观测"读几何属性强制同步 layout"的代价，与 8.6 的缓存机制正反呼应——**读 `offsetHeight` 会强制 flush 一次 layout，缓存失效**。

`offsetHeight` 等属性会强制同步 layout，权威清单见 Paul Irish 的 "What forces layout/reflow"（https://gist.github.com/paulirish/5d52fb081b3570c81e3a）：`offsetTop/Left/Width/Height`、`clientTop/...`、`scrollWidth/Height`、`getBoundingClientRect()`、`getClientRects()`、`getComputedStyle()`、`scrollIntoView()`、`element.focus()`、`element.innerText`、`window.innerHeight/scrollX/scrollY` 等。

```html
<!-- /tmp/layout-thrash.html -->
<!DOCTYPE html>
<html>
<head><style>.box{width:100px;height:20px;background:#cde;margin:1px;}</style></head>
<body>
<div id="container"></div>
<script>
const container = document.getElementById('container');
const N = 2000;
for (let i = 0; i < N; i++) {
  const d = document.createElement('div');
  d.className = 'box';
  container.appendChild(d);
}
const boxes = [...document.querySelectorAll('.box')];

// 强制先布局一次，排除首帧噪声
document.body.offsetHeight;

// —— BAD：读写交替（layout thrashing / forced sync layout）——
function thrash() {
  const t0 = performance.now();
  for (const b of boxes) {
    // 每次先读 offsetWidth（强制 flush layout）再写 width（弄脏 layout）
    const w = b.offsetWidth;            // 强制同步 layout #1..N
    b.style.width = (w + 1) + 'px';     // 立刻让 layout 变脏
  }
  return performance.now() - t0;
}

// —— GOOD：先批量读，再批量写（只触发一次 layout）——
function batched() {
  const t0 = performance.now();
  const widths = boxes.map(b => b.offsetWidth);   // 全部读完（一次 layout）
  boxes.forEach((b, i) => b.style.width = (widths[i] + 1) + 'px'); // 全部写
  return performance.now() - t0;
}

// 交替跑，避免顺序偏差
console.log('BAD  (thrash) :', thrash().toFixed(2), 'ms');
console.log('GOOD (batched):', batched().toFixed(2), 'ms');
console.log('BAD  (thrash) :', thrash().toFixed(2), 'ms');
console.log('GOOD (batched):', batched().toFixed(2), 'ms');
</script>
</body>
</html>
```

**运行**：

```bash
open /tmp/layout-thrash.html   # 然后看 DevTools Console
```

**预期输出**（数量级，机器相关）：

```
BAD  (thrash) : 18.40 ms
GOOD (batched): 1.10 ms
BAD  (thrash) : 17.90 ms
GOOD (batched): 0.95 ms
```

BAD 比 GOOD 慢一个数量级。**与源码呼应**：

- BAD 路径每次 `offsetWidth` 触发 `box_->CachedLayoutResult` 的缓存检查，但上一行的 `style.width` 已经把对应 `LayoutBox` 标 dirty、使缓存 `ConstraintSpace` 失配 → `cache_status = kNeedsLayout` → 整条脊柱重跑 `BlockLayoutAlgorithm::Layout()`，N 次。
- GOOD 路径所有 `style.width` 攒到一帧末尾，只产生一次 layout，绝大多数子 fragment 命中缓存复用。

**DevTools 验证**：Performance → Record → 触发 BAD，停止后在 Main 时间线找紫色 "Layout" 块密集出现；新版 DevTools 的 Performance insights 侧栏会直接有 "Forced reflow" 项指出代码位置与耗时（来源：DebugBear / web.dev 的 forced reflows 文档已核实此 UI）。

---

### Demo 2：CDP 驱动量化 LayoutCount —— 证明缓存命中【可运行，Node.js + CDP】

用 CDP `Performance.getMetrics` 读 `LayoutCount`，直接证明"重复 layout 同一内容、输入不变时 layout 次数不线性增长"。

```bash
# 启动带调试端口的 Chrome
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 --no-first-run --no-default-browser-check \
  --user-data-dir=/tmp/cdp-profile
# 另开终端: npm install chrome-remote-interface
```

```javascript
// layout-count.js  ——  node layout-count.js
const CDP = require('chrome-remote-interface');

(async () => {
  const client = await CDP({ port: 9222 });
  const { Page, Runtime, Performance } = client;
  await Performance.enable();
  await Page.enable();

  await Page.navigate({ url: 'data:text/html,' + encodeURIComponent(`
    <style>.row{height:18px;width:300px;background:#eef;margin:1px}</style>
    <div id="c"></div>
    <script>
      const c=document.getElementById('c');
      for(let i=0;i<1500;i++){const d=document.createElement('div');d.className='row';c.appendChild(d);}
    </script>`) });
  await Page.loadEventFired();

  const read = async name =>
    (await Performance.getMetrics()).metrics.find(m => m.name === name)?.value;

  const before = await read('LayoutCount');

  // 反复读 offsetHeight，但【不弄脏】DOM —— 输入 ConstraintSpace 不变
  await Runtime.evaluate({ expression: `
    let s=0; for(let i=0;i<50;i++){ s += document.body.offsetHeight; } s;` });
  const afterCleanRead = await read('LayoutCount');

  // 现在每次读前都改一个元素 —— 每次弄脏 → 每次强制 layout
  await Runtime.evaluate({ expression: `
    const rows=document.querySelectorAll('.row');
    for(let i=0;i<50;i++){ rows[i].style.width=(300+i)+'px'; void document.body.offsetHeight; }` });
  const afterDirtyRead = await read('LayoutCount');

  console.log('clean repeated reads  → LayoutCount delta:', afterCleanRead - before);
  console.log('dirty-then-read loop  → LayoutCount delta:', afterDirtyRead - afterCleanRead);
  await client.close();
})().catch(console.error);
```

**预期输出**（数量级）：

```
clean repeated reads  → LayoutCount delta: 0   (或 1)
dirty-then-read loop  → LayoutCount delta: 50
```

**与源码呼应**：clean 读循环里 DOM 没脏 → `constraint_space` 与缓存 `LayoutResult::GetConstraintSpaceForCaching()` 完全一致 → 50 次读全部命中缓存，`LayoutCount` 几乎不动；dirty 循环每次 `style.width` 改变 → 缓存失配 → 50 次真 layout。这把 8.6 的"ConstraintSpace 即缓存键"变成可观测的数字。

> 注：`LayoutCount` 是 Blink 暴露的累计 layout 次数指标；不同 Chrome 版本指标名集合可能微调，若取不到可改用 `RecalcStyleCount` 旁证或在 Performance 面板里数 Layout 块。

---

### Demo 3：chrome://tracing 看 layout 事件与 fragment【手动观测】

```
步骤：
1. Chrome 打开 chrome://tracing  →  Record
2. 类别勾选 "blink"、"blink.layout"（或直接选 "Web developer" 预设）→ Record
3. 新标签打开任意稍复杂页面（或 Demo 1 的 layout-thrash.html 并点击触发）
4. Stop，在 timeline 搜索关键事件
```

**预期可见的 trace 事件**（搜索框输入）：

- `LocalFrameView::PerformLayout` / `UpdateLayout` —— layout 阶段入口
- `LayoutNG`、`BlockNode::Layout` 相关切片（取决于 trace 详细度）
- `LayoutObjectsThatHadNeverHadLayout` 等计数事件

**与源码呼应**：`PerformLayout` 内部最终走到各 `BlockNode::Layout`（8.6）。在 thrash 页面上你会看到 Layout 切片**密集重复**；在静态页面上 layout 切片只在首帧出现，之后帧没有——直观对应"输入不变 → 缓存命中 → 无 layout"。

---

### Demo 4：竖排 / RTL 验证 logical→physical 折叠【可运行，纯浏览器】

验证 8.3/8.5 的核心论断：**算的时候 logical，输出 fragment 时折叠为 physical**。同一段内容在 `writing-mode: vertical-rl` 下，`offsetWidth/Height`（physical）与 inline/block 尺寸的对应关系翻转。

```html
<!-- /tmp/writing-mode.html -->
<!DOCTYPE html>
<html><body>
<div id="h" style="width:200px;font:16px/1.5 sans-serif;border:1px solid #888">
  横排：inline-size 沿水平，block-size 沿垂直。这段文字用来撑出多行。这段文字用来撑出多行。
</div>
<div id="v" style="writing-mode:vertical-rl;height:200px;font:16px/1.5 sans-serif;border:1px solid #888">
  竖排：inline-size 现在沿垂直，block-size 沿水平。这段文字用来撑出多列。这段文字用来撑出多列。
</div>
<script>
const r = el => { const b = el.getBoundingClientRect(); return {w:Math.round(b.width), h:Math.round(b.height)}; };
console.log('horizontal box (physical w×h):', r(document.getElementById('h')));
console.log('vertical   box (physical w×h):', r(document.getElementById('v')));
// 横排：约束在 width(=inline) → 内容沿 block(垂直) 增长 → height 变大
// 竖排：约束在 height(=inline) → 内容沿 block(水平) 增长 → width  变大
</script>
</body></html>
```

**预期**：横排盒子 `width≈200`、`height` 随行数增大；竖排盒子 `height≈200`、`width` 随"列数"增大。**与源码呼应**：两者的 `BlockLayoutAlgorithm` 内部都用 `ConstraintSpace::AvailableSize()`（`LogicalSize`）算，inline-size 被约束、block-size 自适应；产出 `PhysicalFragment::Size()` 时按 `GetWritingMode()` 把 logical 折叠成 physical width/height——所以同一套算法、physical 结果方向相反。

---

### Demo 5：观测 block fragmentation —— 多列下一个元素裂成多个 fragment【手动观测 + 可运行旁证】

```html
<!-- /tmp/multicol.html -->
<!DOCTYPE html>
<html><body>
<div style="column-count:3;column-gap:20px;column-fill:auto;height:160px;width:600px;border:1px solid #333">
  <p id="p" style="break-inside:auto;margin:0">
    这是一段会跨列断开的长文本。LayoutNG 会在布局过程中把这个段落切成多个 fragment，
    每列一个 fragmentainer，对应 NGBlockBreakToken 续算。重复填充。重复填充。重复填充。
    重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。
    重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。
    重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。重复填充。
  </p>
</div>
<script>
// 旁证：getClientRects() 对跨列元素会返回多个矩形，每个对应一个 fragment 片段
const rects = document.getElementById('p').getClientRects();
console.log('段落被切成的可视矩形数（≈fragment 片段数）:', rects.length);
for (const r of rects) console.log('  rect:', Math.round(r.left), Math.round(r.width));
</script>
</body></html>
```

**预期**：`getClientRects()` 返回 **2~3 个**矩形，每个对应一列里的一段——直接对应 8.7"一个元素跨多列 → 多个 fragment"。**手动观测加强**：DevTools → Rendering → 勾选 "Layout Shift Regions" 无关；用 Elements 选中 `<p>`，hover 时高亮会在三列各画一块，印证多 fragment。

**与源码呼应**：每列是一个 fragmentainer，`<p>` 产出多个 `PhysicalBoxFragment`，相邻片段间用 `BlockBreakToken` 串联续算（8.7.2）。这在 legacy 引擎里是"切 strip"的假象，在 NG 里是布局时真断。

---

### Demo 6：Electron 路径 —— print-to-PDF 触发分页 fragmentation【可运行 Electron app + patch 走读】

Electron 没有自己的 layout 代码，但它的 **print-to-PDF / 离屏渲染**会走 paged-media 分页路径（8.7 的 Chrome 108 printing 支持），是 Electron 工程里最容易撞上 fragmentation 行为差异的地方。

**最小可运行 app**（演示分页布局产物）：

```javascript
// main.js
const { app, BrowserWindow } = require('electron');
const fs = require('fs');

app.whenReady().then(async () => {
  const win = new BrowserWindow({ show: false, width: 800, height: 600 });
  await win.loadURL('data:text/html,' + encodeURIComponent(`
    <style>
      .page-break { break-after: page; }
      .block { height: 400px; margin: 10px; background: #eef; border: 1px solid #99c; }
    </style>
    ${Array.from({length: 8}, (_, i) =>
      `<div class="block ${i % 2 ? 'page-break' : ''}">Block ${i}</div>`).join('')}
  `));

  // 走 Chromium 的 paged-media 分页：每块 400px，A4 一页放不下 → 触发 block fragmentation
  const pdf = await win.webContents.printToPDF({
    printBackground: true,
    pageSize: 'A4',
  });
  fs.writeFileSync('/tmp/electron-frag.pdf', pdf);
  console.log('written /tmp/electron-frag.pdf, bytes =', pdf.length);
  app.quit();
});
// package.json: { "main": "main.js" }
// 运行: npm install electron && npx electron .
```

**预期**：生成多页 PDF，`break-after:page` 处强制分页；高块在页边界自然续到下一页——这条路径在 Chromium 内部正是 `ConstraintSpace::HasBlockFragmentation()==true` + `BlockBreakToken` 续算（8.7）。

**Electron patch 体系走读**（偏 build，给最小补丁示例，不冒充可跑）：

Electron 通过 `patches/chromium/*.patch` 在 build 时打补丁到 Chromium 源码树。与 layout 相关的定制极少（因为 layout 是纯引擎逻辑），但若要在 Electron 层 hook 分页行为，落点不在 layout 算法本身，而在 print 设置注入。补丁组织形式（示意，非逐字）：

```diff
# patches/chromium/printing_pagesize_override.patch  （示意结构，非真实补丁）
--- a/printing/print_settings.cc
+++ b/printing/print_settings.cc
@@
 void PrintSettings::SetPrinterPrintableArea(...) {
+  // Electron 可在此注入自定义 printable area，
+  // 间接影响 fragmentainer 尺寸 → 影响 block fragmentation 断点
   ...
 }
```

**关键认知**：你**不会**也**不该**去 patch `BlockLayoutAlgorithm` / `ConstraintSpace`——layout 是引擎内核，Electron 的定制点在其**输入侧**（print settings、viewport、device scale），通过改变 fragmentainer 尺寸/可用空间来间接影响 layout 结果。这与第 7 章"Electron 改 UA stylesheet 注入点而非改 StyleResolver"是同一哲学：**改引擎的输入，不改引擎的算法**。

---

## 8.9 方案对比

### 8.9.1 legacy layout vs LayoutNG

| 维度 | legacy（reflow 模型） | LayoutNG |
|---|---|---|
| 输入/输出 | 同一可变 LayoutObject 既存输入又存输出 | `ConstraintSpace`（输入）/ `PhysicalFragment`（输出）显式分离 |
| 输出可变性 | 原地改写，可被任意阶段修改 | fragment immutable，layout 后永不改 |
| 复杂度 | 嵌套 flex/grid 趋向 O(2ⁿ) | measure/layout 分槽缓存 → O(n) |
| 幂等性 | 否（hysteresis bug） | 是（function 式，输入定则输出定） |
| 分页 | 布局后 clip+translate（假分页） | 布局中真断，BreakToken 续算 |
| flex/grid 分页 | 不支持 | Chrome 103 起支持 |
| paint/hit-test 输入 | inspect 可变 LayoutObject | 遍历只读 fragment tree |
| 可中断/可缓存/可并行 | 难 | 缓存已落地，可中断有 EStatus，跨线程/并行有架构基础 |
| 上线 | Chrome 108 退役 | Chrome 108 起唯一实现 |

### 8.9.2 fragment tree（NG）vs 单一 layout tree（legacy）

| 维度 | 单一 layout tree | fragment tree |
|---|---|---|
| element↔节点 | 1:1 | 1:N（跨列/分页一对多） |
| 父指针 | 有 | **无**（no up-ref，可共享子树） |
| 数据流向 | 上下双向冒泡 | **不冒泡**（子树可独立复用） |
| 增量更新 | 全/子树重算 | 只重建 spine，子 fragment 复用 |
| inline 表示 | 行盒树 | 扁平 (item, descendant_count) 列表 |

### 8.9.3 缓存层次

| 缓存 | 粒度 | Key | 命中收益 |
|---|---|---|---|
| `LayoutResult`（kLayout slot） | per BlockNode | 生成它的 `ConstraintSpace` | 输入空间不变 → 复用整棵子 fragment |
| `LayoutResult`（kMeasure slot） | per BlockNode | measure 用 `ConstraintSpace` | flex/grid 两趟里测量结果不重复 |
| simplified layout（`kNeedsSimplifiedLayout`） | spine | 子未变、自身小改 | 只重建脊柱，子 fragment 原样复用 |

**何时各自不适用 / 边界**：

- `LayoutResult` 缓存**对 thrashing 无效**——每次写 DOM 都让 `ConstraintSpace` 失配（Demo 1）。这不是缓存的锅，是调用方读写交替的锅。
- simplified layout **遇到 `HasBrokenSpine()` 失效**——祖先链尺寸变了，子 fragment 的 physical 偏移基准就错了，必须回退完整 layout。
- fragment 复用**依赖 no-up-ref / no-bubble**——任何让子 fragment 依赖父/兄弟的特性（某些复杂 BFC 交互）都会缩小可复用范围。

---

## 8.10 失败模式与生产真坑

### 坑 1：Layout Thrashing（forced synchronous layout）—— 最高频性能杀手

**场景**：循环里"读几何属性 + 改样式"交替（Demo 1 的 BAD），或第三方库在 resize/scroll handler 里反复 `getBoundingClientRect()`。

**根因**：读 `offsetHeight` 等属性必须返回**最新**几何值 → 强制 flush 一次完整 layout；上一行的 style 写入又让 `ConstraintSpace` 失配 → 缓存无法命中 → N 次循环 = N 次真 layout。本质是**主动击穿 8.6 的缓存**。

**诊断**：DevTools Performance → 录制 → Main 线程出现密集紫色 Layout 块；新版 Performance insights 侧栏直接列 "Forced reflow" 及代码定位（来源：DebugBear forced-reflows 文档、web.dev avoid-large-complex-layouts-and-layout-thrashing，均已核实）。

**修复**：批量读、批量写（Demo 1 的 GOOD）；或用 `requestAnimationFrame` / `ResizeObserver` 把读移出写循环；库层面用 FastDOM 模式。

### 坑 2：把 `getComputedStyle()` 当廉价读 —— 隐式触发 layout

**场景**：动画循环里 `getComputedStyle(el).height` 取计算值。

**根因**：当请求的属性是 layout 相关（height/width/margin/padding/transform/grid 等）或元素在 shadow tree / 有 media query 时，`getComputedStyle` **会强制 layout**（paulirish 清单已核实）。开发者常以为它只读 style 阶段结果，实际可能击穿到 layout。

**修复**：缓存计算结果；只在必要时读；优先用 `IntersectionObserver`/`ResizeObserver` 拿几何信息（它们在合成阶段异步算，不强制同步 layout）。

### 坑 3：误以为"一个 element 一个 box / 一个 rect"

**场景**：多列或分页里对元素做 `getBoundingClientRect()` 定位 overlay，结果只覆盖了第一列。

**根因**：跨 fragmentainer 的元素是 **1:N fragment**（8.5.3 / Demo 5）。`getBoundingClientRect()` 返回外接矩形（可能横跨多列留白），而 `getClientRects()` 才返回每个 fragment 片段。

**诊断**：`el.getClientRects().length > 1` 即说明该元素被 fragment 化。

**修复**：定位多片段元素用 `getClientRects()` 逐段处理，别用单一 bounding rect。

### 坑 4：动画 `width/height/top/left` 触发每帧 layout

**场景**：用 JS 或 CSS transition 动画 `left`/`width`，掉帧。

**根因**：这些属性是 layout-inducing —— 每帧改它们都让对应子树 `ConstraintSpace` 失配、走 `BlockLayoutAlgorithm::Layout()`，无法停留在合成线程。

**修复**：动画改用 `transform`/`opacity`（只走 compositor，不进 layout/paint）；需要"位移"用 `transform: translate()` 取代 `left/top`。这是 RenderingNG 把 transform 放进 property tree、绕开 layout 的设计意图。

### 坑 5：`content-visibility` / `display:lock` 下读几何拿到过期值

**场景**：用 `content-visibility:auto` 优化长列表，JS 立刻读其内部元素 `offsetTop` 拿到 0 或错误值。

**根因**：`content-visibility` 会跳过 off-screen 子树的 layout（对应源码 `ChildLayoutBlockedByDisplayLock()`，Demo 2/8.6 出现过）。子树未 layout，几何属性当然不可信。

**修复**：读之前用 `contentVisibilityAutoStateChange` 事件或滚动到可视区触发其 layout；不要在 locked 子树上假设几何有效。

### 坑 6：Electron print-to-PDF / 离屏渲染的分页差异

**场景**：同一页面浏览器里好好的，Electron `printToPDF` 出来内容在页边界被切断或元素消失。

**根因**：print 路径强制 `HasBlockFragmentation()==true`，走 paged-media 分页（8.7）。某些 `monolithic` 内容（固定高滚动容器、`break-inside:avoid` 的大块）在页高放不下时行为与屏幕渲染不同。Electron 的 `pageSize`/`margins`/`scale` 直接决定 fragmentainer 尺寸 → 决定断点。

**修复**：为打印写专门的 `@media print` 样式，显式设 `break-inside`/`break-before`；用 Demo 6 的 app 复现并调 `pageSize`；切忌假设屏幕布局 == 打印布局。

---

## 8.11 章末五件套

### 1. 关键路径总结

```
ComputedStyle + DOM
  → BlockNode::Layout(ConstraintSpace, BreakToken)
      ├─ box_->GetCachedLayoutResult / CachedLayoutResult   [缓存命中? ConstraintSpace 为键]
      │     ├─ kHit              → 直接复用 LayoutResult / 子 fragment
      │     ├─ kNeedsSimplified  → 只重建 spine，复用子 fragment
      │     └─ kNeedsLayout      → 真跑算法
      ├─ DetermineAlgorithmAndRun → Block/Inline/Flex/Grid/Table…LayoutAlgorithm
      │     ├─ ComputeMinMaxSizes()  [measure pass, kMeasure slot]
      │     └─ Layout()             [layout  pass, kLayout  slot]
      └─ 产出 const LayoutResult*  → PhysicalFragment (immutable, physical 坐标)
                                       ├─ 跨 fragmentainer 时多 fragment + BreakToken
                                       └─→ pre-paint / paint / hit-test / a11y 遍历
```

### 2. 五个最重要的数据结构

| 结构 | main 分支文件 | 作用 |
|---|---|---|
| `ConstraintSpace` | `core/layout/constraint_space.h` | layout 输入空间（available size / BFC / exclusion / cache slot） |
| `LayoutAlgorithm` | `core/layout/layout_algorithm.h` | 每种盒子一个算法，`ComputeMinMaxSizes` + `Layout` |
| `LayoutResult` | `core/layout/layout_result.h` | 算法返回值 + 缓存键（`GetConstraintSpaceForCaching`）+ EStatus |
| `PhysicalFragment` | `core/layout/physical_fragment.h` | immutable 几何输出，physical 坐标，paint/hit-test 唯一输入 |
| `BlockBreakToken` | `core/layout/block_break_token.h` | 分页/多列续算 token，挂在 fragment 上组成 token 树 |

### 3. 三个关键设计/优化

1. **input/output 分离 + 输出 immutable**：`ConstraintSpace`（栈、临时、只读输入）↔ `PhysicalFragment`（only-read 输出），根治 legacy 四宗罪（hysteresis / under-/over-invalidation / 契约不清）。
2. **measure/layout 双 slot 缓存**：以 `ConstraintSpace` 为键缓存两个 pass，把嵌套 flex/grid 的 O(2ⁿ) 拉回 O(n)（官方原话 "predictably linear"）。
3. **fragment tree 即后续阶段唯一输入**：no-up-ref / no-bubble 让子树可共享、可只重建 spine、可跨线程；分页在 fragment tree 里用 BreakToken 原生表达（Chrome 102→108 全量上线）。

### 4. Electron 特殊性

| 方面 | 说明 |
|---|---|
| layout 代码 | **零定制**，完整继承 Chromium LayoutNG |
| 定制哲学 | 改引擎**输入**（print settings / viewport / device scale），不改 layout 算法本身 |
| 高频撞点 | `printToPDF` / 离屏渲染走 paged-media 分页，`HasBlockFragmentation()==true` |
| Node.js 干预 | `webContents.executeJavaScript` 批量读几何 = 强制同步 layout（坑 1 的 Electron 版） |
| patch 落点 | `patches/chromium/*.patch` 改 print/printing 相关，间接影响 fragmentainer 尺寸 |

### 5. 延伸阅读（均已 WebFetch 核实 URL 可达）

- RenderingNG deep-dive: LayoutNG — https://developer.chrome.com/docs/chromium/layoutng
- RenderingNG deep-dive: BlinkNG（重写背景/六原则）— https://developer.chrome.com/docs/chromium/blinkng
- Key data structures in RenderingNG（fragment tree immutable 约束）— https://developer.chrome.com/docs/chromium/renderingng-data-structures
- RenderingNG deep-dive: block fragmentation（BreakToken / fragmentainer / 上线时间线）— https://developer.chrome.com/docs/chromium/renderingng-fragmentation
- LayoutNG 仓库设计文档 — third_party/blink/renderer/core/layout/layout_ng.md（raw.githubusercontent.com/chromium/chromium/main/…）
- 真实源码（main 分支，raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/core/layout/）：`constraint_space.h`、`layout_result.h`、`physical_fragment.h`、`layout_algorithm.h`、`block_node.cc`
- Paul Irish — What forces layout/reflow（强制同步 layout 属性清单）— https://gist.github.com/paulirish/5d52fb081b3570c81e3a
- web.dev — Avoid large, complex layouts and layout thrashing — https://web.dev/articles/avoid-large-complex-layouts-and-layout-thrashing

---

*本文档源码来源说明*：
- 标【真实源码 repo@path】的代码经 WebFetch 从 `raw.githubusercontent.com/chromium/chromium/main/...` 实际取得（main 分支），类名以**去 NG 前缀的现状**为准。
- 标【示意，非逐字】的代码（如 Electron patch 结构）基于文档描述重构，不保证逐字。
- 标「待核」处为未逐一核实的细节（如去前缀的具体 rename CL 号）。
- 官方设计史料经 WebFetch 从 developer.chrome.com / chromium.googlesource.com 核实；引用 fragmentation 旧文档时保留其原文 `NG` 类名并标注。
- 老博客 / 旧 design doc 使用 `NGConstraintSpace` / `NGPhysicalFragment` 等旧名；阅读时按 "NGFoo == Foo" 对应 main 分支无前缀名。
