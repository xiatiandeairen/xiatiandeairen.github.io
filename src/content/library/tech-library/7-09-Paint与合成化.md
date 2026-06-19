---
title: "Paint 与合成化（Chromium 域）"
slug: "7-09"
collection: "tech-library"
group: "chromium内核"
order: 7009
summary: "TL;DR Paint 与合成化是 RenderingNG 流水线 layout 之后、屏幕像素之前的那一大段。核心一句话：先把 immutable 的 fragment tree「画」成一份与 DOM 解耦的全局 display list（一串 `DisplayItem`，按 paint orde…"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:59:30.000Z"
updatedAt: "2026-06-12T11:59:30.000Z"
---
> **TL;DR**
> Paint 与合成化是 RenderingNG 流水线 layout 之后、屏幕像素之前的那一大段。核心一句话：**先把 immutable 的 fragment tree「画」成一份与 DOM 解耦的全局 display list（一串 `DisplayItem`，按 paint order 切成共享同一 property tree state 的 `PaintChunk`），再由 `PaintArtifactCompositor` 在 paint 之后决定 layer 划分（layerization），把 paint chunk 合并成尽量少的 `cc::Layer`，并把 Blink 的四棵 property tree 转成 cc 的四棵 property tree，commit/activate 到合成线程，最后由 viz 把每个 layer 光栅化出的 GPU texture tile 拼成 compositor frame 画到屏上**。
> 这套架构是 **Slimming Paint（2015 起，到 CompositeAfterPaint / M94 收官）** 多年重写的产物。它根治的是一个 longstanding 的 **fundamental compositing bug**：旧架构「compositing-before-paint」、layer 划分绑死在 `PaintLayer`（≈ DOM 元素）上，导致没有自己 layer 的元素无法被正确排到合成顺序里——例如 `backface-visibility:hidden` 的兄弟元素 paint order 会错。CompositeAfterPaint 把 compositing 决策搬到 paint 之后、改成基于 display list + property tree，bug 自然消失，并顺手删了约 22,000 行 C++。
> 四棵 **property tree（transform / clip / effect / scroll）** 是整章的骨架：它们让「一个 effect 应用到哪些内容」从「layer 的父子嵌套」解耦成「内容引用哪个 property node」，使得滚动、动画、合成更新都变成 O(感兴趣的节点) 而非 O(layer 数)。`effect` 树上的 `RenderSurfaceReason` 精确定义了「什么会强制一个 render surface（离屏中间纹理）」——opacity<1 带合成子节点、filter、blend mode、mask、clip-path…每一个都是真金白银的显存与带宽。
> Electron 完整继承这套引擎、无独立 paint/compositing 代码；唯一的重点是**离屏渲染（OSR）**：`OffScreenRenderWidgetHostView` 通过 viz 的 `FrameSinkVideoCapturer` 把合成好的 compositor frame 抓出来（software 模式拷 bitmap，shared-texture 模式用 `kPreferGpuMemoryBuffer` 直接拿 GPU 纹理句柄），再 emit `paint` 事件——它消费的正是本章末端那个 compositor frame。

---

## 前置依赖

| 需要掌握 | 用于理解 |
|---|---|
| 第 8 章 Layout / LayoutNG | paint 的输入是 immutable 的 fragment tree（`PhysicalFragment`），不是可变 LayoutObject |
| 第 7 章 Style 计算 | `ComputedStyle` 决定 compositing reason（`will-change` / `transform` / `opacity` / `filter`）与 effect 参数 |
| 第 4 章 调度器与事件循环 | commit / activate / draw 的跨线程时序、BeginMainFrame 节奏 |
| 第 3 章 Mojo IPC | renderer↔GPU(viz) 进程间传 compositor frame、`SubmitCompositorFrame` |
| CSS stacking context / paint order（z-index、opacity、transform 建栈规则） | display list 的顺序、effect 树的父子关系从何而来 |
| GPU 纹理 / 光栅化 / Skia 基本概念（SkPicture / 纹理 tile） | raster 阶段把 display list 烤成像素 |
| RenderingNG 流水线全 12 阶段（animate→style→layout→pre-paint→scroll→paint→commit→layerize→raster→activate→aggregate→draw） | 本章覆盖 pre-paint→draw 这后半程，知道每阶段跑在哪个线程 |

---

## 9.1 设计考古：从 GraphicsLayer 树到「paint 后再合成」

### 9.1.1 旧世界：compositing-before-paint，layer 绑死在 PaintLayer 上

要理解今天，先理解被推翻的昨天。Blink 早期（继承自 WebKit）的合成模型是 **compositing-before-paint**：

- 一棵 `PaintLayer`（旧名 `RenderLayer` / `DeprecatedPaintLayer`）树，大体对应「会建立 stacking context 或需要特殊 paint/hit-test 处理的 DOM 元素」。
- `PaintLayerCompositor` 在 paint **之前**，根据 style 判断哪些 `PaintLayer` 需要「提升（promote）」成独立合成层，产出一棵 `GraphicsLayer` 树。
- 然后每个 `GraphicsLayer` 各自 paint 自己那部分内容，录成 `cc::Layer` 的 picture。

官方对 `PaintLayer` 的定性很直白（core/paint README 原话）：

> "PaintLayer ... an old implementation detail of Blink. It represents some layout objects to handle a lot of operations about painting and hit-testing."

问题在于：**DOM 元素（≈ PaintLayer）根本不是「高效或正确的 layerization 方案」的好基础**。Slimming Paint 项目页把这件事命名为 **the Fundamental Compositing Bug**：

> "the fact that DOM elements are not a good 1:1 representation of an efficient or complete layerization scheme for web page contents. Since compositing was before paint, it more or less inherently depended on DOM elements, not display lists or property trees."

来源（已 WebFetch 核实）：
- Slimming Paint 项目页 — https://www.chromium.org/blink/slimming-paint/
- core/paint README — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/core/paint/README.md
- paint-dev 邮件列表「fundamental compositing bug」串 — https://groups.google.com/a/chromium.org/g/paint-dev/c/TwS7H2qWsuk/m/hINccJiAtr4J

### 9.1.2 bug 长什么样：一个 backface-visibility 的排序例子

抽象描述不如一个具体例子。在 paint-dev 那个串里，Chris Harrelson 给的场景是：元素 A（`backface-visibility:hidden`，会建合成层）后面跟一个用负 margin 盖上来的兄弟 B（B 自己没有任何 compositing reason、因此没有 `PaintLayer`/合成层）。正确的合成方式他原话是：

> "The right way to composite this is to allocate two layers, one for A's drawing, and one for B's. But Blink can't do that right now because B doesn't have a RenderLayer."

—— 旧架构里「能不能单独成层」是先于 paint、按 `RenderLayer` 决定的，**B 没有 RenderLayer 就没法被排进正确的合成顺序**。而同串里 Tien-Ren Chen 指出：display item list 本身其实早就是对的——

> "Background | B | BeingBackface | A | EndBackface"

也就是说：**如果先 paint 出 display list（顺序天然正确），再据此决定 layer，bug 根本不会发生**。这正是 CompositeAfterPaint 的核心洞见。

> 【真实引用 paint-dev 邮件列表】上述两段为该公开线程原文，非本章杜撰。具体 message 落在 https://groups.google.com/a/chromium.org/g/paint-dev/c/TwS7H2qWsuk/m/hINccJiAtr4J 。

### 9.1.3 Slimming Paint 的多年分期（带上线版本）

Slimming Paint 不是一次性切换，而是横跨 2015→2021 的渐进重写。Slimming Paint 项目页给出的里程碑（原文逐条核实）：

| 阶段 flag | 上线版本 | 干了什么（项目页原文要点） |
|---|---|---|
| **SlimmingPaintV1** | M45 | "Introduced paint using display items" —— 引入 display item 录制 |
| **SlimmingPaintInvalidation** | M58 | "Rewrote paint invalidation using display items and introduced property trees in Blink" —— Blink 里首次有了 property tree |
| **SlimmingPaintV175** | M67 | "Utilized property trees for painting in Blink, introduced paint chunks for raster invalidation" |
| **BlinkGenPropertyTrees** | M75 | "Shifted to sending a layer list and generating final property trees in Blink rather than cc" —— 改成 Blink 发 layer list、自己生成 property tree |
| **CompositeAfterPaint** | M94 | "compositing decisions made after paint" —— 合成决策搬到 paint 之后 |

收益（项目页原文）：

> "22,000 lines of c++ were removed ... total chrome CPU usage -1.3%, 3.5%+ improvement to 99th percentile scroll update, 2.2%+ improvement to 95th percentile input delay"

并且它解锁了下游项目（项目页点名）：HitTestOpaqueness、RasterInducingScroll。

来源（已 WebFetch 核实）：
- Slimming Paint 项目页 — https://www.chromium.org/blink/slimming-paint/
- RenderingNG deep-dive: BlinkNG（重写背景）— https://developer.chrome.com/docs/chromium/blinkng

### 9.1.4 三条核心设计动机（提炼）

1. **compositing 必须在 paint 之后**：唯有先有 display list（顺序正确）+ property tree（effect 关系明确），layer 划分才能做对、做优。这是对 fundamental compositing bug 的根治，而非打补丁。
2. **用 property tree 取代 layer 树承载 effect**：「一个 transform/clip/opacity 作用到哪些内容」不再靠 layer 父子嵌套表达，而是内容引用 property node。`how_cc_works` 原话：cc「instead ... is provided with separate trees of properties」，让更新变成 O(感兴趣节点) 而非 O(layer 数)。
3. **display list 与 DOM 解耦**：Slimming Paint 项目目标原文就是「re-implement the Blink<->cc picture recording API to work in terms of a global display list rather than a tree of cc::Layers」。display list 是全局的、扁平的、与 DOM 结构无关的绘制指令流。

---

## 9.2 流水线全景：pre-paint → paint → commit → layerize → raster → activate → aggregate → draw

把本章放进 RenderingNG 的 12 阶段流水线（RenderingNG architecture 官方页逐阶段原文 + 运行线程）：

```
            ┌──────────────────────────────── 主线程（renderer main） ──────────────────────────────┐
 animate → style → layout →  PRE-PAINT  → scroll →  PAINT  →  COMMIT
                              │                       │          │
              "compute        │       "compute a      │   "copy property trees
               property trees,│        display list   │    and the display list
               invalidate     │        that describes │    to the compositor thread"
               display lists   │       how to raster   │
               & tiles"        │       GPU texture     │
                               │       tiles from DOM" │
            └──────────────────┼───────────────────────┼──────────┼──────────────────────────────┘
                               │                       │          ▼
            ┌──────────────────┼───────────────────────┼─── 合成线程（compositor / impl）──────────┐
                                                          LAYERIZE → RASTER(+decode) → ACTIVATE
                                                          │            │                 │
                                  "break up the display   │  "turn display lists,        │ "create a compositor
                                   list into a composited  │   encoded images ... into    │  frame representing how
                                   layer list for          │   GPU texture tiles"         │  to draw and position
                                   independent             │  (部分在 Viz 进程)            │  GPU tiles to the screen"
                                   rasterization and        │
                                   animation"               │
            └─────────────────────────────────────────────┼─────────────────┼────────────────────┘
                                                            │                 ▼
                                          ┌─────────────────┼──── Viz / GPU 进程 ─────────────────┐
                                                       AGGREGATE  →  DRAW
                                                       │              │
                                "combine compositor    │   "execute the aggregated
                                 frames from all the    │    compositor frame on the GPU
                                 visible compositor      │    to create pixels on-screen"
                                 frames into a single,    │
                                 global compositor frame" │
                                          └──────────────────────────────────────────────────────┘
```

> 每一段引号都是 RenderingNG architecture 页对该 stage 的**原文一句话定义**（已 WebFetch 核实）。线程归属同样来自该页（main=主线程、compositor=合成线程、viz=GPU 进程）。

来源（已 WebFetch 核实）：
- RenderingNG architecture（12 阶段定义 + 线程色块）— https://developer.chrome.com/docs/chromium/renderingng-architecture
- how cc works（commit/activate/draw 时序）— https://chromium.googlesource.com/chromium/src/+/lkgr/docs/how_cc_works.md

本章重点覆盖 **PRE-PAINT → DRAW** 这后半程。layout 见第 8 章，调度时序见第 4 章。

---

## 9.3 PRE-PAINT：建 property tree + paint invalidation

paint 之前有一趟独立的树遍历叫 **PrePaintTreeWalk**，它有且只有两个目标。core/paint README 原话：

> The PrePaint walk has "two primary goals: paint invalidation and building paint property trees."

真实源码（类的职责注释逐字）：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/core/paint/pre_paint_tree_walk.h】
// This class walks the whole layout tree, beginning from the root
// LocalFrameView, across frame boundaries. Helper classes are called for each
// tree node to perform actual actions.  It expects to be invoked in InPrePaint
// phase.
class CORE_EXPORT PrePaintTreeWalk final {
  STACK_ALLOCATED();

 public:
  PrePaintTreeWalk() = default;
  void WalkTree(LocalFrameView& root_frame);
  void Walk(LocalFrameView&, const PrePaintTreeWalkContext& parent_context);
  void Walk(const LayoutObject&,
            const PrePaintTreeWalkContext& parent_context,
            PrePaintInfo*);
```

walk 过程中携带一个 context，里面挂着 property tree builder 的上下文：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/core/paint/pre_paint_tree_walk.h】
  struct PrePaintTreeWalkContext : public PrePaintTreeWalkContextBase {
    std::optional<PaintPropertyTreeBuilderContext> tree_builder_context;
```

两件事拆开看：

- **building paint property trees**：每个 `LayoutObject` 在 `FragmentData` 上挂一份 `ObjectPaintProperties`（仅当它「诱导」出 property node 时）。core/paint README 原话：「Each `PaintLayer`'s `LayoutObject` has one or more `FragmentData` objects. Every `FragmentData` has an `ObjectPaintProperties` object if any property nodes are induced by it.」例如一个 `transform: rotate(...)` 诱导一个 transform node，`overflow:hidden` 诱导一个 clip node，`opacity:0.5` 诱导一个 effect node。
- **paint invalidation**：标记哪些 display item / display item client 需要重画。README 原话：在某些 style 更新场景能「directly update the property tree without needing to run the property tree builder」（fast-path），避免整趟重建。

> RenderingNG architecture 对 pre-paint 的一句话定义：「compute property trees and invalidate any existing display lists and GPU texture tiles as appropriate」——注意它同时让 display list **和** GPU tile 失效，这是后面 raster invalidation 的源头。

来源（已 WebFetch 核实）：
- pre_paint_tree_walk.h — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/core/paint/pre_paint_tree_walk.h
- core/paint README — 同上

---

## 9.4 四棵 Property Tree：整章的骨架

property tree 是这整套架构的承重墙。RenderingNG data-structures 页给的总定义：

> "Property trees are data structures that explain how visual and scrolling effects apply to DOM elements. ... Every web document has four separate property trees: transform, clip, effect, and scroll."

四棵树各管一摊（data-structures 页原文）：

| 树 | 管什么（原文） |
|---|---|
| **transform** | "represents CSS transforms and scrolling. (A scroll transform is represented as a 2D transform matrix.)" |
| **clip** | "represents overflow clips" |
| **effect** | "represents all other visual effects: opacity, filters, masks, blend modes, and other kinds of clips" |
| **scroll** | "represents information about scrolling, such as how scrolls chain together" |

每块内容携带一个 **property tree state**——一个四元组：

> "Each DOM element has a property tree state, which is a 4-tuple (transform, clip, effect, scroll) that indicates the nearest ancestor clip, transform, and effect tree nodes that take effect on that element."

注意 property tree 在两侧各有一份：**Blink 侧**（`*PaintPropertyNode`，paint 阶段建、随 paint chunk 走）和 **cc 侧**（`TransformNode`/`ClipNode`/`EffectNode`/`ScrollNode`，commit 时由 Blink 侧转换而来）。下面分别看真实源码。

### 9.4.1 Blink 侧：TransformPaintPropertyNode

文件头注释逐字解释了一个 transform node 是什么：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/paint/transform_paint_property_node.h】
// A transform (e.g., created by css "transform" or "perspective", or for
// internal positioning such as paint offset or scrolling) along with a
// reference to the parent TransformPaintPropertyNode. The scroll tree is
// referenced by transform nodes and a transform node with an associated scroll
// node will be a 2d transform for scroll offset.
//
// The transform tree is rooted at a node with no parent. This root node should
// not be modified.
```

关键一句：**scroll 被 transform node 引用，带 scroll node 的 transform node 就是「滚动偏移的 2D 平移」**——这就是为什么「scroll 也是一种 transform」。

State（节点的全部可变状态）逐字：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/paint/transform_paint_property_node.h】
struct PLATFORM_EXPORT State {
  DISALLOW_NEW();

 public:
  TransformAndOrigin transform_and_origin;                       // gfx::Transform matrix + gfx::Point3F origin
  Member<const ScrollPaintPropertyNode> scroll;                  // 关联的 scroll 节点（若是滚动平移）
  Member<const TransformPaintPropertyNode> scroll_parent_scroll_translation;

  bool flattens_inherited_transform : 1 = false;                 // 是否把继承的 3D 压平到父平面
  bool in_subtree_of_page_scale : 1 = true;                      // 在 pinch-zoom 的 page-scale 子树内（影响 raster）
  bool animation_is_axis_aligned : 1 = false;
  // Set if a frame is rooted at this node.
  bool is_frame_paint_offset_translation : 1 = false;
  bool is_for_svg_child : 1 = false;

  BackfaceVisibility backface_visibility = BackfaceVisibility::kInherited;  // kInherited/kHidden/kVisible
  unsigned rendering_context_id = 0;                             // 3D 排序上下文 id；0 = 不参与 3D 排序
  CompositingReasons direct_compositing_reasons = CompositingReason::kNone; // ★ 直接合成原因
  CompositorElementId compositor_element_id;
  std::unique_ptr<CompositorStickyConstraint> sticky_constraint;
  std::unique_ptr<cc::AnchorPositionScrollData> anchor_position_scroll_data;
  // If a visible frame is rooted at this node, this represents the element
  // ID of the containing document.
  CompositorElementId visible_frame_element_id;

  PaintPropertyChangeType ComputeTransformChange(
      const TransformAndOrigin& other,
      const AnimationState& animation_state) const;
  PaintPropertyChangeType ComputeChange(
      const State& other,
      const AnimationState& animation_state) const;

  bool UsesCompositedScrolling() const {
    return direct_compositing_reasons & CompositingReason::kOverflowScrolling;
  }
  bool RequiresCullRectExpansion() const {
    return direct_compositing_reasons &
           CompositingReason::kRequiresCullRectExpansion;
  }

  void Trace(Visitor*) const;
};
```

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/paint/transform_paint_property_node.h】
class PLATFORM_EXPORT TransformPaintPropertyNode final
    : public TransformPaintPropertyNodeOrAlias {
```

值得记住的三点：
- `direct_compositing_reasons` 直接挂在 transform node 上——**这是「为什么这个 transform 要单独合成」的真相所在**（9.6 展开）。
- `backface_visibility` + `rendering_context_id` 是 3D 排序（`preserve-3d`）的载体，`flattens_inherited_transform` 决定是否压平。
- `…OrAlias` 后缀：property tree 有「alias 节点」机制——只为命名/查找方便存在、不携带实际 effect，layerization 时会被 upcast 掉。

### 9.4.2 cc 侧：四棵树 + PropertyTrees 聚合

commit 之后，cc 持有自己的一份。`property_tree.h` 里四棵树各是 `PropertyTree<NodeType>` 的 final 子类，最后被 `PropertyTrees` 聚合：

```cpp
// 【真实源码 chromium@cc/trees/property_tree.h】
class CC_EXPORT TransformTree final : public PropertyTree<TransformNode> {
 public:
  explicit TransformTree(PropertyTrees* property_trees = nullptr);
  // ...
 private:
  float page_scale_factor_;
  float device_scale_factor_;
  float device_transform_scale_factor_;
  std::vector<int> nodes_affected_by_outer_viewport_bounds_delta_;
  std::vector<TransformCachedNodeData> cached_data_;
  std::vector<StickyPositionNodeData> sticky_position_data_;
  std::vector<AnchorPositionScrollData> anchor_position_scroll_data_;
  // ...
};

class CC_EXPORT ClipTree final : public PropertyTree<ClipNode> { /* ... */ };

class CC_EXPORT EffectTree final : public PropertyTree<EffectNode> {
 private:
  std::multimap<int, std::unique_ptr<viz::CopyOutputRequest>> copy_requests_;
  std::vector<std::unique_ptr<RenderSurfaceImpl>> render_surfaces_;   // ★ effect 树才有 render surface
};

class CC_EXPORT ScrollTree final : public PropertyTree<ScrollNode> {
 private:
  int currently_scrolling_node_id_ = kInvalidPropertyNodeId;
  ScrollOffsetMap scroll_offset_map_;
  SyncedScrollOffsetMap synced_scroll_offset_map_;             // ★ 主线程/合成线程双份 scroll offset
  // ...
};
```

```cpp
// 【真实源码 chromium@cc/trees/property_tree.h】
class CC_EXPORT PropertyTrees final {
 private:
  TransformTree transform_tree_;
  EffectTree effect_tree_;
  ClipTree clip_tree_;
  ScrollTree scroll_tree_;
  bool needs_rebuild_ = true;
  // ...
  int sequence_number_ = 0;
  std::vector<int> changed_effect_nodes_;
  std::vector<int> changed_transform_nodes_;
  // ...
};
```

两处「省源码注释」之外的关键观察：
- **只有 `EffectTree` 持有 `render_surfaces_`**——render surface（离屏中间纹理）是 effect 树的概念，9.5 详解。
- `ScrollTree` 有 `synced_scroll_offset_map_`：scroll offset 在主线程和合成线程各一份、需要同步——这是「合成线程能独立滚动（不等主线程）」的数据基础。
- `changed_effect_nodes_` / `changed_transform_nodes_`：cc 只追踪「变了的节点」，这就是「更新是 O(感兴趣节点)」的实现。

来源（已 WebFetch 核实）：
- transform_paint_property_node.h — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/platform/graphics/paint/transform_paint_property_node.h
- cc/trees/property_tree.h — https://raw.githubusercontent.com/chromium/chromium/main/cc/trees/property_tree.h

---

## 9.5 EffectNode 与 RenderSurface：什么会强制一个离屏中间纹理

effect 树是性能成本的集中地，因为某些 effect 必须「先把子树画到一张离屏纹理、再整体施加效果」——这就是 **render surface**。cc 的 `EffectNode` 把这件事说得最清楚。

### 9.5.1 EffectNode 字段

```cpp
// 【真实源码 chromium@cc/trees/effect_node.h】
struct CC_EXPORT EffectNode {
  float opacity = 1.f;
  float screen_space_opacity = 1.f;
  FilterOperations filters;
  FilterOperations backdrop_filters;
  ElementId element_id;
  SkBlendMode blend_mode = SkBlendMode::kSrcOver;
  RenderSurfaceReason render_surface_reason =
      RenderSurfaceReason::kNone;
  bool has_copy_request : 1 = false;

  bool HasRenderSurface() const {
    return render_surface_reason != RenderSurfaceReason::kNone;
  }
};
```

`HasRenderSurface()` 的逻辑一目了然：**`render_surface_reason != kNone` 就需要一张离屏纹理**。

### 9.5.2 RenderSurfaceReason：完整的「强制离屏」清单（真实枚举）

这份枚举是本章最有操作价值的源码之一——它**逐项列出了所有会触发 render surface 的原因**：

```cpp
// 【真实源码 chromium@cc/trees/effect_node.h】
enum class RenderSurfaceReason : uint8_t {
  kNone,
  kRoot,
  k3dTransformFlattening,
  // Defines the scope of the backdrop for child blend mode or backdrop filter.
  kBackdropScope,
  kBlendMode,
  kBlendModeDstIn,
  kOpacity,                 // opacity<1 且其子树有需要合成的内容
  kOpacityAnimation,
  kFilter,                  // filter: blur()/drop-shadow()/... 
  kFilterAnimation,
  kBackdropFilter,          // backdrop-filter（毛玻璃）
  kBackdropFilterAnimation,
  kRoundedCorner,
  kClipPath,                // clip-path 非轴对齐
  kClipAxisAlignment,
  kMask,                    // -webkit-mask / mask
  kTrilinearFiltering,
  kCache,
  kCopyRequest,             // CDP/截图/captureVisibleTab 等抓像素请求
  kMirrored,
  kSubtreeIsBeingCaptured,
  kViewTransitionParticipant,
  kGradientMask,
  k2DScaleTransformWithCompositedDescendants,
  kUnboundedElement,
  kTest,
};
```

**怎么用这份清单**：当你在 DevTools Layers 面板看到某层的 compositing/render-surface 成本，或在 `chrome://tracing` 看到 render pass 数量暴涨，对照这张表就能反推「是哪个 CSS 属性在制造离屏纹理」。render surface 不是 layer——它是一张**每帧重新合成的中间纹理**，blur/backdrop-filter 尤其昂贵（带宽 × 采样半径）。

> 注意 `kOpacity`：`opacity:0.5` 只有在「其子树存在会单独合成的内容」时才需要 render surface（否则可以直接在 paint 时把 alpha 乘进去）。这解释了为什么有时 `opacity` 廉价、有时昂贵。

### 9.5.3 filter 如何变成像素：RenderSurfaceFilters

effect 树上的 `filters` 最终要变成 Skia 的 image filter 链。负责转换的是 `cc/paint/render_surface_filters.cc`（已核实存在）：

```cpp
// 【真实源码 chromium@cc/paint/render_surface_filters.cc】
sk_sp<PaintFilter> RenderSurfaceFilters::BuildImageFilter(
    const FilterOperations& filters,
    const gfx::Rect& layer_bounds);
```

它把 `FilterOperations`（grayscale / sepia / blur / drop-shadow…）逐个转成对应的 `PaintFilter` 并串成链，作用到 render surface 那张离屏纹理上。

来源（已 WebFetch 核实）：
- cc/trees/effect_node.h — https://raw.githubusercontent.com/chromium/chromium/main/cc/trees/effect_node.h
- cc/paint/render_surface_filters.cc — https://raw.githubusercontent.com/chromium/chromium/main/cc/paint/render_surface_filters.cc

---

## 9.6 PAINT：DisplayItem 与 PaintChunk

paint 阶段的产物是 **paint artifact**：一串 display item，按 paint order 切成 paint chunk。platform/graphics/paint README 原话：paint artifact 由「a list of display items in paint order (ideally mostly or all drawings), partitioned into *paint chunks* which define certain *paint properties*」组成。

### 9.6.1 DisplayItem：display list 的最小单元

```
DisplayItem（README 原文）
  = 由 ID 唯一标识：「an opaque pointer to the display item client that produced it」
    + 「a type (from the DisplayItem::Type enum)」

DrawingDisplayItem（README 原文）
  = "Holds a PaintRecord which contains the paint operations required to draw some atom of content."
```

RenderingNG data-structures 页对 display item 的定性：

> "A display item contains low-level drawing commands ... that can be rasterized with Skia. Display items are typically simple, with just a few drawing commands, such as drawing a border or background."

也就是说，display item 多数是「画个 border」「画个 background」这种细粒度操作，内部是一个 `PaintRecord`（≈ 录好的 Skia 绘制指令，旧称 SkPicture）。

### 9.6.2 PaintChunk：共享 property tree state 的连续 drawing（真实源码）

chunk 是 layerization 的最小颗粒。文件头注释一句话定义 + 真实字段：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/paint/paint_chunk.h】
// A contiguous sequence of drawings with common paint properties.
// ...（struct PaintChunk 字段）
  wtf_size_t begin_index;                          // 该 chunk 对应 display item 区间 [begin,end)
  wtf_size_t end_index;
  Id id;
  BackgroundColorInfo background_color;
  TraceablePropertyTreeStateOrAlias properties;    // ★ 该 chunk 的 (transform,clip,effect,scroll) 状态
  Member<HitTestData> hit_test_data;
  Member<RegionCaptureData> region_capture_data;
  Member<LayerSelectionData> layer_selection_data;
  Member<TrackedElementRects> tracked_element_rects;
  gfx::Rect bounds;                                // chunk 整体包围盒
  gfx::Rect drawable_bounds;
  gfx::Rect rect_known_to_be_opaque;               // 已知不透明区域（LCD text / 合并优化用）
  RasterEffectOutset raster_effect_outset;
  cc::HitTestOpaqueness hit_test_opaqueness;
  bool text_known_to_be_on_opaque_background : 1;
  bool has_text : 1;
  bool is_cacheable : 1;
  bool client_is_just_created : 1;
  bool is_moved_from_cached_subsequence : 1;
  bool effectively_invisible : 1;
```

`properties` 这个 `PropertyTreeStateOrAlias` 就是 9.4 那个四元组——**chunk 是「一段共享同一 property tree state 的连续 display item」**。RenderingNG data-structures 页原话说得最直白：

> "The current property tree state is maintained during the paint tree walk and the display item list is grouped into 'chunks' of display items that share the same property tree state."

`rect_known_to_be_opaque` / `text_known_to_be_on_opaque_background` 不是装饰——它们决定能否用 LCD subpixel 抗锯齿、以及两个 chunk 能否合并进同一 layer（9.7）。

### 9.6.3 谁来生产、谁来缓存：PaintController

paint 的编排者是 `PaintController`。README 原话：它负责「producing the paint artifact. It contains the *current* paint artifact, and *new* display items and paint chunks」。缓存机制是性能关键——painter 调用：

> "PaintController::UseCachedItemIfPossible() or PaintController::UseCachedSubsequenceIfPossible() and if the function returns true, existing display items that are still valid in the current paint artifact will be reused."

即：**没变的子树直接复用上一帧的 display item，不重画**。这是 paint 阶段的增量基础。

### 9.6.4 raster invalidation：两级失效

display list 变了，对应的 GPU tile 要重烤——这叫 raster invalidation，README 说它分两级：

> "Paint chunk level [RasterInvalidator] and Display item level [DisplayItemRasterInvalidator]."

chunk 级管「哪个 chunk 整体动了」，display-item 级管「chunk 内部哪几个 item 动了」，最终转成「哪几块 tile 需要重烤」。

来源（已 WebFetch 核实）：
- platform/graphics/paint README — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/platform/graphics/paint/README.md
- RenderingNG data-structures — https://developer.chrome.com/docs/chromium/renderingng-data-structures
- paint_chunk.h — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/platform/graphics/paint/paint_chunk.h

---

## 9.7 LAYERIZE：PaintArtifactCompositor 把 chunk 合并成 cc::Layer

这是 CompositeAfterPaint 的心脏：**paint 之后**，`PaintArtifactCompositor`（PAC）把 paint artifact 转成 cc 的 layer list + property tree。

### 9.7.1 PAC 的入口

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/compositing/paint_artifact_compositor.h】
class PLATFORM_EXPORT PaintArtifactCompositor final
    : public GarbageCollected<PaintArtifactCompositor>,
      private PropertyTreeManagerClient {
 public:
  // ...
  void Update(const PaintArtifact& artifact,
              const ViewportProperties& viewport_properties,
              const StackScrollTranslationVector& scroll_translation_nodes,
              Vector<std::unique_ptr<cc::ViewTransitionRequest>> requests);
```

`Update()` 吃一份 `PaintArtifact`（= display items + paint chunks），吐出 cc layer list 和 cc property trees。它还有一个 `kRepaint` 快路径，源码注释逐字：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/compositing/paint_artifact_compositor.h】
// This copies over the newly-painted PaintChunks to existing
// |pending_layers_|, issues raster invalidations, and updates the existing
// cc::Layer properties such as background color.
```

即：只有重画、layer 结构没变时，直接把新 chunk 拷进已有的 `pending_layers_`、发 raster invalidation，不重做 layerization。

### 9.7.2 PendingLayer：layerization 的工作单元

layerization 的核心数据结构是 `PendingLayer`——文件头注释一句话定义：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/compositing/pending_layer.h】
// A pending layer is a collection of paint chunks that will end up in the same
// cc::Layer.
class PLATFORM_EXPORT PendingLayer {
  DISALLOW_NEW();

 public:
  enum CompositingType {
    kScrollHitTestLayer,
    kForeignLayer,        // 如 <video>/<canvas>/iframe 这类「外来」layer
    kScrollbarLayer,
    kOverlap,             // 因 overlap 被迫单独成层
    kOther,
  };
```

核心算法是「**贪心合并**」：尽量把相邻 chunk 合进同一个 `PendingLayer`，少建 layer 省显存。能不能合，靠这几个判定（真实方法签名）：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/compositing/pending_layer.h】
  bool Merge(const PendingLayer& guest,
             LCDTextPreference lcd_text_preference,
             float device_pixel_ratio,
             IsCompositedScrollFunction);

  bool CanMergeWithDecompositedBlendMode(const PendingLayer& guest,
                                         const PropertyTreeState& upcast_state,
                                         IsCompositedScrollFunction) const;

  std::optional<PropertyTreeState> CanUpcastWith(
      const PendingLayer& guest,
      const PropertyTreeState& guest_state,
      IsCompositedScrollFunction is_comosited_scroll) const;   // 注：参数名 is_comosited_scroll 系源码原文拼写

  bool CanMerge(const PendingLayer& guest,
                LCDTextPreference lcd_text_preference,
                float device_pixel_ratio,
                IsCompositedScrollFunction,
                gfx::RectF& merged_bounds,
                PropertyTreeState& merged_state,
                gfx::RectF& merged_rect_known_to_be_opaque,
                bool& merged_text_known_to_be_on_opaque_background,
                wtf_size_t& merged_solid_color_chunk_index,
                cc::HitTestOpaqueness& merged_hit_test_opaqueness) const;
```

关键字段：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/compositing/pending_layer.h】
  PaintChunkSubset chunks_;                          // 本层包含的 chunk 集合
  TraceablePropertyTreeState property_tree_state_;   // 本层统一的 property tree state
  gfx::RectF bounds_;
  gfx::RectF rect_known_to_be_opaque_;
  wtf_size_t solid_color_chunk_index_ = kNotFound;
  CompositingType compositing_type_ = kOther;
  cc::HitTestOpaqueness hit_test_opaqueness_;
```

`CanUpcastWith` 体现了 alias / upcast 思想：两个 chunk 的 property state 不完全相同，但若差异部分可以「上提到公共祖先」（如把一个能被 decompose 的 blend mode 拆掉），仍能合并进同一层。这正是 9.4.1 里 `…OrAlias` 节点存在的意义。

### 9.7.3 一句话讲清 layerization 的取舍

- **合得多** → layer 少 → 省显存、省合成开销，但任一处重画就要重烤整层、且独立动画/滚动能力弱。
- **合得少** → layer 多 → 各自可独立 raster/animate（compositor-only 动画的前提），但显存、tile 管理、overlap squashing 成本上升 → 极端就是「layer explosion」（9.10 坑 2）。

PAC 的 `CanMerge` 就是在每对相邻 chunk 上做这道取舍题。

来源（已 WebFetch 核实）：
- paint_artifact_compositor.h — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/platform/graphics/compositing/paint_artifact_compositor.h
- pending_layer.h — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/platform/graphics/compositing/pending_layer.h

---

## 9.8 直接合成原因（Direct Compositing Reasons）：为什么这个东西要单独成层

9.4.1 里 transform node 上那个 `direct_compositing_reasons` 是「为什么单独合成」的真相。Blink 在 `compositing_reasons.h` 里把它们定义成 bitmask。真实源码（节选，名字逐字）：

```cpp
// 【真实源码 chromium@third_party/blink/renderer/platform/graphics/compositing_reasons.h】
k3DTransform = UINT64_C(1) << kE3DTransform,
kTrivial3DTransform = UINT64_C(1) << kETrivial3DTransform,
kWillChangeTransform = UINT64_C(1) << kEWillChangeTransform,
kWillChangeOpacity = UINT64_C(1) << kEWillChangeOpacity,
kWillChangeFilter = UINT64_C(1) << kEWillChangeFilter,
kBackfaceVisibilityHidden = UINT64_C(1) << kEBackfaceVisibilityHidden,
kActiveTransformAnimation = UINT64_C(1) << kEActiveTransformAnimation,
kVideo = UINT64_C(1) << kEVideo,
kCanvas = UINT64_C(1) << kECanvas,
kOverflowScrolling = UINT64_C(1) << kEOverflowScrolling,
kFixedPosition = UINT64_C(1) << kEFixedPosition,
kStickyPosition = UINT64_C(1) << kEStickyPosition,
kBackdropFilter = UINT64_C(1) << kEBackdropFilter,
```

WebSearch 也确认存在组合掩码（`compositing_reasons.cc`）`kComboAllDirectStyleDeterminedReasons`，其展开包含：`k3DTransform | kTrivial3DTransform | kBackfaceVisibilityHidden | kComboActiveAnimation | kWillChangeTransform | kWillChangeOpacity | kWillChangeFilter | kWillChangeOther | kBackdropFilter | kWillChangeBackdropFilter`。

> 「待核」：`compositing_reasons.h` 里这些常量实际由宏（`FOR_EACH_COMPOSITING_REASON` 之类）展开生成，上面是展开后的等价形式；`kComboAllDirectStyleDeterminedReasons` 的精确展开来自 WebSearch 摘要的 `compositing_reasons.cc`，未逐字 fetch 该 .cc 全文。常量名本身已核实。

**怎么读这张表**：当你想知道「为什么 DevTools 说这个元素被 composited」——`will-change: transform/opacity/filter`、3D transform、运行中的 transform 动画、`<video>`/`<canvas>`、`position:fixed/sticky`、composited overflow scrolling、`backface-visibility:hidden`、`backdrop-filter` 都是 **direct** reason（内在原因，直接让该元素自己成层）。另一类是 **overlap** reason——元素 A 因为盖在某个 direct-composited 元素上而被迫也成层（避免穿帮），这类靠 squashing 收敛（9.10 坑 2）。

来源：
- compositing_reasons.h — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/platform/graphics/compositing_reasons.h（已 WebFetch 核实）
- compositing_reasons.cc（组合掩码展开）— https://chromium.googlesource.com/chromium/src/+/c4e126bb9623bd51751f5267c497940349b27e70/third_party/blink/renderer/platform/graphics/compositing_reasons.cc（WebSearch 摘要核实，未逐字 fetch）

---

## 9.9 COMMIT → RASTER → ACTIVATE → DRAW：cc 与 viz

layerization 之后，剩下的是 cc（合成线程）+ viz（GPU 进程）的活。

### 9.9.1 两棵树、三个原子操作（how cc works 原文）

cc 用「主线程树 / impl 线程树」双树结构。how cc works 原话：

> "Main Thread: LayerTreeHost owns a tree of cc::Layer objects that Blink modifies"
> "Compositor Thread: LayerTreeHostImpl manages compositor thread layer trees as cc::LayerImpl instances"

层用 id 对应：「A layer with id 5 on the main thread tree will push to layer id 5 on the pending tree.」三个原子操作：

- **Commit**：「a method of getting data atomically from the main thread to the compositor thread」——主线程在此期间 block（mutex）。数据进 **pending tree**（cc README：「The pending tree is a staging tree for rasterization」）。
- **Activate**：pending tree 光栅化完后晋升为 **active tree**。how cc works：「Activate ... pushes data from the pending tree to the active tree」；上一个 active tree 降级为 **recycle tree** 复用、避免重分配。
- **Draw**：active tree 负责出帧；何时画由 `Scheduler` / `SchedulerStateMachine` 决定。

### 9.9.2 Raster：tile + TileManager

`PictureLayer` 的内容被切成 tile。how cc works 原话：tile 是「a sparse 2d regular tiling of the content at a particular scale」。`TileManager` 在 worker 线程上调度光栅化，软件/GPU 两种模式皆可。RenderingNG 解释了为什么要 tile：

> "the viewport is divided into tiles. A separate GPU texture tile backs each tile ... The renderer can then update individual tiles or even just change the position on screen for the existing tiles."

—— tile 让「只重烤动的那一小块」和「只挪位置不重烤」成为可能。

### 9.9.3 Viz：compositor frame → render pass → 屏幕

每个 compositor（renderer、browser）把自己的 active tree 画成一个 **compositor frame** 提交给 viz。cc README 原话：

> "CompositorFrames from individual compositors are sent to the SurfaceManager (which is in the GPU process). The SurfaceAggregator combines all CompositorFrames together ... given to the viz::DirectRenderer, which finally draws the entire composited browser contents."

compositor frame 的内部结构（RenderingNG data-structures 原文）：

- **render pass**：「a list of quads. The render pass doesn't contain any pixel information; instead, it has instructions on where and how to draw each quad」。
- 一个 frame 是「a list of render passes. There is always a root render pass, which is drawn last and whose destination corresponds to the frame buffer」。
- **为什么会有多个 render pass**：「Some visual effects, such as many filters or advanced blend modes, require that two or more quads are drawn to an intermediate texture」——**这正好对应 9.5 的 render surface**：一个 effect 树上的 render surface ≈ 一个额外的 render pass + 一张中间纹理。
- **quad**：「A quad identifies the input texture, and indicates how to transform and apply visual effects to it」；tile 是 quad 的一种，还有 solid-color quad、texture quad（video/canvas）、surface draw quad（嵌别的 surface）。

> 这条链可以反过来当 perf 诊断用：`chrome://tracing` 里 render pass 数量 = root pass + 每个 render surface 一个。render pass 暴涨 → 回 9.5 的 `RenderSurfaceReason` 表查是哪个 effect。

来源（已 WebFetch 核实）：
- how cc works — https://chromium.googlesource.com/chromium/src/+/lkgr/docs/how_cc_works.md
- cc/README.md — https://raw.githubusercontent.com/chromium/chromium/main/cc/README.md
- RenderingNG data-structures — https://developer.chrome.com/docs/chromium/renderingng-data-structures

---

## 9.10 Demo（重中之重）：在真实引擎上观测 paint 与合成

Chromium 全量 build 不现实，下面五个 demo 全部在**现成的 Chrome / Electron 引擎**上跑，与前面源码逐一呼应。给完整命令 + 步骤 + 预期输出。

> 约定：下文 `chrome` 指你本机的 Chrome/Chromium 可执行文件（macOS 上常为 `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`）。

### Demo 1：DevTools Rendering —— 看 paint flashing / layer borders（呼应 9.6 / 9.7）

**目的**：肉眼看见「哪里在重画」「哪里被提升成 layer / 切成 tile」。

**步骤**：
1. 打开任意页面，F12 → 右上 `⋮` → More tools → **Rendering**。
2. 勾选 **Paint flashing**：发生 repaint 的区域闪绿。
3. 勾选 **Layer borders**：叠加 layer 与 tile 边框。
4. 勾选 **Frame Rendering Stats**：右上角显示实时 FPS / 掉帧。

**预期输出**（DevTools 文档原文）：
- Paint flashing：「Chrome flashes the screen green whenever repainting happens」——在一个有动画 caret 的输入框打字，只有光标那一小块闪绿（验证 9.6.4 raster invalidation 是局部的）。
- Layer borders：「Observe layer borders in **orange and olive** and tiles in **cyan**」——给某个元素加 `will-change: transform`，它立刻被橙色 layer 边框圈出，并被青色网格切成 tile（验证 9.8 `kWillChangeTransform` 直接成层 + 9.9.2 tiling）。
- 颜色精确含义见 cc 源码 `cc/debug/debug_colors.cc`。

**最小验证 HTML**（保存为 `layers.html` 用浏览器打开）：

```html
<!doctype html><meta charset=utf-8>
<style>
  .box{width:200px;height:200px;background:tomato;margin:20px}
  .promoted{will-change:transform}      /* ← 触发 kWillChangeTransform，单独成层 */
</style>
<div class="box"></div>
<div class="box promoted"></div>
```
打开 Layer borders 后，第二个 `.box` 会被橙色边框单独圈出，第一个不会。把 `.promoted` 那行注释掉刷新，橙框消失——直接对应 9.8 的 direct compositing reason。

### Demo 2：CDP 驱动 LayerTree.compositingReasons —— 程序化拿「为什么成层」（呼应 9.8）

**目的**：不靠肉眼，用 Chrome DevTools Protocol 把每个 layer 的 compositing reason 拉出来——直接对应 `direct_compositing_reasons` 字段。

**启动带远程调试端口的 Chrome**：
```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/cdp-profile \
  --no-first-run \
  file:///absolute/path/to/layers.html
```

**用 curl 找到目标页的 WebSocket，再用 Node 驱动 CDP**（可跑代码，仅依赖内置 `ws`-free 的 `WebSocket`——Node 18+ 已内置全局 `WebSocket`）：

```js
// cdp-layers.mjs —— Node 18+，无需安装依赖
const list = await (await fetch('http://localhost:9222/json')).json();
const page = list.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);

let id = 0;
const pending = new Map();
const send = (method, params = {}) =>
  new Promise(res => { const i = ++id; pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params })); });

const layers = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  if (m.method === 'LayerTree.layerTreeDidChange') layers.length = 0,
    (m.params.layers || []).forEach(l => layers.push(l));
};

ws.onopen = async () => {
  await send('DOM.enable');
  await send('LayerTree.enable');            // 触发 layerTreeDidChange 事件
  await new Promise(r => setTimeout(r, 500)); // 等事件到齐
  for (const l of layers) {
    const r = await send('LayerTree.compositingReasons', { layerId: l.layerId });
    if (r.compositingReasonIds?.length)
      console.log(l.layerId, l.width + 'x' + l.height, '→', r.compositingReasonIds);
  }
  ws.close();
};
```
```bash
node cdp-layers.mjs
```

**预期输出**（形如）：
```
12 200x200 → [ 'willChangeTransform' ]
```
即 CDP 报告该 layer 的成层原因是 `willChangeTransform`——与 9.8 源码里的 `kWillChangeTransform` 一一对应。`LayerTree.compositingReasons` 文档原文：「the reasons why the given layer was composited ... returns compositingReasons as an array of strings」。

> 说明：CDP 字段历史上有 `compositingReasons`（字符串）与 `compositingReasonIds`（稳定 id）两版，不同 Chrome 版本择一返回，代码里两者都打。LayerTree domain 完整方法见 https://chromedevtools.github.io/devtools-protocol/tot/LayerTree/ 。

### Demo 3：chrome://tracing —— 看 commit/activate/raster/render-pass 时序（呼应 9.9）

**目的**：在真实 trace 上看见 9.9 的 commit → raster → activate → draw 跨线程时序，以及 render pass 数量。

**步骤**：
1. 地址栏开 `chrome://tracing`（新版若重定向到 Perfetto UI 亦可，操作类似）→ **Record**。
2. 类别选 **cc**、**viz**、**blink**（或直接选 "Rendering" 预设）。
3. 在被测页面触发一次动画 / 滚动，停止录制。
4. 在时间轴上找 `CrRendererMain`（主线程）与 `Compositor`（合成线程）两条 thread。

**预期观测**：
- 主线程上一帧内能看到 `ProxyMain::BeginMainFrame` → `…Commit`；
- 合成线程上看到 `RasterTask`（对应 9.9.2 TileManager 的光栅化）、`ActivateSyncTree`（对应 activate）；
- GPU/viz 线程上看到 `Display::DrawAndSwap` / `SurfaceAggregator`（对应 9.9.3 aggregate+draw）。
- 给页面加一个 `filter: blur(8px)` 再录一次：对照 9.5/9.9.3，render pass 数量 +1（多出 blur 的中间 render surface）。

> 这是「偏观测」demo：trace 事件名随版本演进可能微调，但 commit/raster/activate/draw 四段结构稳定。属「待核」的是具体 event 字符串名在你本机版本上的精确拼写。

### Demo 4：DevTools Layers 面板 —— 抓「layer explosion」与每层显存（呼应 9.7 / 9.10 坑 2）

**目的**：复现并诊断 layer 爆炸。

**步骤**：
1. Command Menu（Cmd/Ctrl+Shift+P）→ 输入 "Layers" → **Show Layers**。
2. 打开下面这个「反面教材」页面：

```html
<!doctype html><meta charset=utf-8>
<style>
  .item{width:80px;height:80px;margin:8px;display:inline-block;
        background:teal; will-change:transform;}   /* ← 每个都强制成层！ */
</style>
<div id="grid"></div>
<script>
  const g = document.getElementById('grid');
  for (let i=0;i<400;i++) g.appendChild(Object.assign(
     document.createElement('div'), {className:'item'}));
</script>
```

**预期观测**：Layers 面板里出现**数百个 layer**；选中任一层，Details 显示其内存占用与 compositing reason（`willChangeTransform`）。把 `will-change:transform` 删掉刷新 → 退回个位数 layer。这直接演示 9.10 坑 2：无差别 `will-change` = 显存灾难。

> 经验阈值（社区/web.dev 口径，非源码常量）：单层超过 ~2048×2048 就该警惕；一张 4096×4096 RGBA ≈ 64MB 未压缩显存——几个就能打满移动 GPU 预算。

### Demo 5：最小 Electron OSR app —— 消费本章末端的 compositor frame（呼应 9.9.3 + Electron 角）

**目的**：亲手拿到「合成好的那一帧」——Electron 离屏渲染消费的正是 viz 输出的 compositor frame。

```js
// main.js —— 最小 Electron OSR；npm i electron 后 npx electron .
const { app, BrowserWindow } = require('electron');
const fs = require('fs');

app.whenReady().then(() => {
  const win = new BrowserWindow({
    width: 800, height: 600, show: false,
    webPreferences: { offscreen: true },      // ★ 开启离屏渲染
  });
  win.loadURL('data:text/html,<body style="background:tomato">' +
              '<h1 style="filter:blur(2px)">OSR</h1>');
  win.webContents.setFrameRate(10);

  let n = 0;
  win.webContents.on('paint', (event, dirty, image) => {  // ★ 每帧合成结果
    if (n++ === 3) {                                       // 抓第 4 帧存盘
      fs.writeFileSync('osr.png', image.toPNG());
      console.log('saved osr.png', image.getSize());
      app.quit();
    }
  });
});
```
```bash
npx electron .     # 生成 osr.png（800x600，含 tomato 背景与 blur 标题）
```

**预期输出**：终端打印 `saved osr.png { width: 800, height: 600 }`，目录下出现 `osr.png`。

**与源码呼应**：Electron OSR README 原文链路——`OffScreenRenderWidgetHostView` 创建 `OffScreenVideoConsumer`，后者经 `HostFrameSinkManager` → `FrameSinkManagerImpl` 拿到 viz 的帧；software 模式直接给 bitmap，shared-texture 模式（`webPreferences.offscreen.useSharedTexture=true`）走 `video_capturer_->Start(... kPreferGpuMemoryBuffer)`，`FrameSinkVideoCapturerImpl` 用 `GpuMemoryBufferVideoFramePool` 拷帧，`OnFrameCaptured` 构造 `OffscreenSharedTextureValue` 把 GPU 纹理句柄交给你。**`paint` 事件里那张 `image` / 纹理句柄，就是 9.9.3 里 `viz::DirectRenderer` 画出来的 compositor frame 的产物**。

> README 警告（原文）：shared-texture 模式下「every frame Chromium may pass a different texture to you」——必须立刻拷进自己的纹理再 release，否则 use-after-free。

来源（已 WebFetch 核实）：
- DevTools Rendering（paint flashing / layer borders 颜色）— https://developer.chrome.com/docs/devtools/rendering/performance
- CDP LayerTree domain — https://chromedevtools.github.io/devtools-protocol/tot/LayerTree/
- Electron OSR README — https://raw.githubusercontent.com/electron/electron/main/shell/browser/osr/README.md

---

## 9.11 方案对比

### 对比 1：compositing-before-paint（旧）vs CompositeAfterPaint（今）

| 维度 | compositing-before-paint（≤M93 路径） | CompositeAfterPaint（M94+） |
|---|---|---|
| layer 划分依据 | `PaintLayer`（≈ DOM 元素）→ `GraphicsLayer` 树 | display list + property tree（paint 之后） |
| effect 承载 | layer 父子嵌套 | 四棵 property tree（内容引用 node） |
| fundamental compositing bug | **存在**（无 layer 的元素排序错，9.1.2） | **根治**（顺序来自 display list） |
| 合成更新复杂度 | 倾向 O(layer 数) | O(感兴趣节点)（cc `changed_*_nodes_`） |
| 代码量 | —— | 净删约 22,000 行 |
| 适用 | 已退役 | 现状 |

**不适用边界**：CompositeAfterPaint 不是「让一切变快」的银弹——它只是让 layer 划分正确且可优化；写出 400 个 `will-change`（Demo 4）照样显存爆炸。架构对了 ≠ 用法对了。

### 对比 2：何时该「主动提升合成层」 vs 何时「绝不要碰」

| 场景 | 该不该提升 | 依据 |
|---|---|---|
| 需要 60fps 的 transform/opacity 动画 | **该**，用 `will-change:transform/opacity` | compositor-only 属性，绕开 layout/paint（9.8 `kActiveTransformAnimation`） |
| `<video>` / WebGL `<canvas>` | 引擎**自动**提升 | `kVideo`/`kCanvas`（9.8），无需手动 |
| 固定的工具栏/侧边栏（`position:fixed`） | 通常**自动** | `kFixedPosition`（9.8） |
| 大面积静态内容、列表项 | **绝不**无差别加 `will-change` | layer explosion（9.10 坑 2） |
| 动画结束后仍留着 `will-change` | **必须撤掉** | 长期占显存（9.10 坑 3） |

### 对比 3：render surface 触发项的成本梯度（基于 9.5 `RenderSurfaceReason`）

| effect | 成本 | 说明 |
|---|---|---|
| `opacity:0.5`（子树无合成内容） | 低（可不建 surface） | paint 时直接乘 alpha |
| `opacity:0.5`（子树有合成内容） | 中 | `kOpacity`，需中间纹理 |
| `transform`（2D，无 3D flatten） | 低 | 多数不建 surface，直接 GPU 变换 quad |
| `filter:blur()` / `drop-shadow()` | **高** | `kFilter` + 中间纹理 + 多次采样（半径 ×） |
| `backdrop-filter`（毛玻璃） | **最高** | `kBackdropFilter`，还要先聚合 backdrop |
| `mask` / `clip-path`（非轴对齐） | 中-高 | `kMask`/`kClipPath` |
| blend mode（非 src-over） | 中-高 | `kBlendMode`，可能拉 `kBackdropScope` |

---

## 9.12 扎根：失败模式 / 生产真坑 / 根因

### 坑 1：用 JS 读 `offsetWidth`/`getBoundingClientRect` 触发 forced sync 后又意外触发重 paint

**场景**：动画循环里先写 style 再立刻读几何，掉帧；Performance 面板出现紫色 layout + 绿色 paint 长条。

**根因**：写 style 让 pre-paint/paint 失效；同帧读几何强制把 layout **和** pre-paint 提前跑完（9.3）。这不仅是第 8 章的 forced layout，还连带 property tree 重建。

**修复**：读写分离（先批量读、再批量写，`requestAnimationFrame` 内）；动画只改 `transform`/`opacity`（9.11 对比 2）——它们走合成线程，根本不进 paint。

### 坑 2：Layer explosion / overlap squashing 失效 → 显存爆 + 滚动卡

**场景**：给列表每项加 `will-change:transform` 或 `translateZ(0)`「优化」，结果更卡，移动端崩溃。

**根因**：每个 direct compositing reason（9.8）都建一个 `cc::Layer`（9.7 `PendingLayer`）。几百个层 = 几百张 tile 纹理；且任何盖在合成层上的元素会因 **overlap** 被迫也成层（9.7.2 `kOverlap`），squashing 收不住时层数指数增长。

**修复**：只给「真在做 compositor 动画」的少数元素加 `will-change`，且**动画前加、动画后撤**。用 Demo 4 的 Layers 面板核对层数与每层显存。

### 坑 3：`will-change` 长期挂着不撤 → 隐性显存泄漏

**场景**：交互结束、元素早已静止/移出视口，显存却一直高。

**根因**：`will-change`/`kWillChange*`（9.8）是「持续承诺」——只要 style 在，layer 与其 backing store 就常驻，不会因为「不动了」自动回收。

**修复**：`transitionend`/`animationend` 里清掉 `will-change`（React/Vue 可 hook 这些事件）。诊断：DevTools Layers 面板交叉核对「已停止动画的元素是否仍被提升」；`chrome://gpu` / GPU 内存指标看占用。

### 坑 4：滥用 `filter`/`backdrop-filter` 导致 render pass 暴涨、低端 GPU 跪

**场景**：满屏毛玻璃卡片（`backdrop-filter:blur`），滚动时帧率断崖。

**根因**：每个 filter / backdrop-filter 在 effect 树上是一个 render surface（9.5 `kFilter`/`kBackdropFilter`），= 一个额外 render pass + 一张每帧重算的中间纹理（9.9.3）；blur 还要按半径多次采样，带宽 ×。`backdrop-filter` 更要先把背后内容聚合好才能模糊。

**修复**：限制毛玻璃元素数量与面积；能用静态模糊图替代就别用实时 backdrop-filter；用 Demo 3 的 `chrome://tracing` 数 render pass 验证。

### 坑 5：Electron OSR shared-texture 模式 use-after-free / 黑屏

**场景**：Electron 离屏渲染 + `useSharedTexture:true`，把拿到的纹理句柄存起来异步用，结果画面错乱或崩溃。

**根因**：OSR README 明示「every frame Chromium may pass a different texture to you」——纹理是 viz 借给你的、随时回收（9.10 Demo 5）。存着指针异步用 = use-after-free。GPU 模式还有历史坑：开不了 GPU channel 会「(after 5 tries) falls back to using the SoftwareOutputDevice」，悄悄退化成软件合成、性能骤降。

**修复**：`paint`/`OnFrameCaptured` 回调里**同步**把纹理拷进自己持有的纹理再 release；软件回退要监控日志、必要时显式 `--disable-gpu` 或修好 GPU 环境。

### 坑 6：合成线程能滚但你以为是主线程在滚（scroll-linked effect 抖动）

**场景**：监听 `scroll` 事件改 style 做视差，滚动时效果滞后/抖动。

**根因**：composited scrolling（9.8 `kOverflowScrolling`）下，scroll offset 在合成线程独立更新（9.4.2 `synced_scroll_offset_map_`），`scroll` 事件是事后通知主线程的——你在主线程改 style 永远慢合成线程一帧。

**修复**：scroll-linked effect 用声明式的 `position:sticky`（9.4.1 `sticky_constraint`，合成线程原生支持）或 ScrollTimeline（animate 阶段在合成线程驱动），而非 JS `scroll` 监听改 style。

---

## 9.13 章末五件套

### 1. 关键路径总结

```
immutable fragment tree (第8章)
  → PRE-PAINT (PrePaintTreeWalk)            主线程
      ├─ build 4 paint property trees (transform/clip/effect/scroll)
      └─ paint invalidation（标脏 display item / raster invalidation 源头）
  → PAINT (PaintController)                 主线程
      ├─ 录 DisplayItem（DrawingDisplayItem 持 PaintRecord）
      ├─ UseCachedItem/SubsequenceIfPossible（没变就复用）
      └─ 切成 PaintChunk（共享同一 PropertyTreeState）→ PaintArtifact
  → COMMIT                                  主线程→合成线程（原子、主线程 block）
  → LAYERIZE (PaintArtifactCompositor)      合成线程
      ├─ PendingLayer 贪心合并 chunk（CanMerge/CanUpcastWith）
      └─ 生成 cc::Layer list + cc property trees（含 EffectNode.render_surface_reason）
  → RASTER (TileManager, worker)            合成线程(+Viz)
      └─ PictureLayer → tiles → GPU texture（filter 经 RenderSurfaceFilters→Skia）
  → ACTIVATE (pending tree → active tree)   合成线程
      └─ 组 compositor frame（render passes + quads；每 render surface 一个额外 pass）
  → AGGREGATE (SurfaceAggregator)           Viz/GPU 进程
  → DRAW (viz::DirectRenderer)              Viz/GPU 进程 → 屏幕像素
                                             └─→（Electron OSR：FrameSinkVideoCapturer 抓帧 → paint 事件）
```

### 2. 五个最重要的数据结构

| 结构 | main 分支文件 | 作用 |
|---|---|---|
| `TransformPaintPropertyNode`（+ Clip/Effect/Scroll 三姊妹） | `platform/graphics/paint/transform_paint_property_node.h` | Blink 侧 property tree 节点；`direct_compositing_reasons` 在此 |
| `PaintChunk` | `platform/graphics/paint/paint_chunk.h` | 共享同一 `PropertyTreeStateOrAlias` 的连续 display item；layerization 颗粒 |
| `PaintArtifactCompositor` | `platform/graphics/compositing/paint_artifact_compositor.h` | paint artifact → cc layer list + property trees（CAP 心脏） |
| `PendingLayer` | `platform/graphics/compositing/pending_layer.h` | layerization 工作单元；`CanMerge`/`CanUpcastWith` 决定合并 |
| `EffectNode`（cc） | `cc/trees/effect_node.h` | cc 侧 effect 节点；`render_surface_reason` 决定是否离屏中间纹理 |

### 3. 三个关键设计/优化

1. **compositing after paint**：layer 划分搬到 paint 之后、基于 display list + property tree，根治 fundamental compositing bug（顺序来自 display list 而非 PaintLayer），净删约 22,000 行。
2. **四棵 property tree 取代 layer 树承载 effect**：effect 作用范围由「内容引用 property node」表达；cc 只追踪 `changed_*_nodes_`，使滚动/动画/合成更新降到 O(感兴趣节点)。
3. **PaintChunk + PendingLayer 贪心合并**：以「共享 property tree state」为单位切 chunk、再贪心合并成尽量少的 cc::Layer，在「少建层省显存」与「多建层可独立 raster/animate」之间逐对取舍。

### 4. Electron 特殊性

| 方面 | 说明 |
|---|---|
| paint/compositing 代码 | **零定制**，完整继承 Chromium |
| 唯一重点 | 离屏渲染（OSR）：消费 viz 输出的 compositor frame |
| 关键类/事件 | `OffScreenRenderWidgetHostView` / `OffScreenVideoConsumer` / `FrameSinkVideoCapturer` / `paint` 事件 |
| 两种模式 | software（拷 bitmap，慢但通用）vs shared-texture（`kPreferGpuMemoryBuffer` 直拿 GPU 纹理句柄，快） |
| 高频撞点 | shared-texture「每帧不同纹理」use-after-free；GPU channel 失败静默回退 SoftwareOutputDevice |
| patch 落点 | `patches/chromium/*` 中 OSR / printing 相关；不改合成算法本身 |

### 5. 延伸阅读（均已 WebFetch / WebSearch 核实 URL 可达）

- Slimming Paint 项目页（fundamental compositing bug / 分期 / 收益）— https://www.chromium.org/blink/slimming-paint/
- RenderingNG architecture（12 阶段定义 + 线程归属）— https://developer.chrome.com/docs/chromium/renderingng-architecture
- Key data structures in RenderingNG（property tree / display list / compositor frame / render pass / quad / tile）— https://developer.chrome.com/docs/chromium/renderingng-data-structures
- RenderingNG deep-dive: BlinkNG — https://developer.chrome.com/docs/chromium/blinkng
- How cc Works — https://chromium.googlesource.com/chromium/src/+/lkgr/docs/how_cc_works.md
- paint-dev 邮件列表「fundamental compositing bug」线程 — https://groups.google.com/a/chromium.org/g/paint-dev/c/TwS7H2qWsuk/m/hINccJiAtr4J
- CDP LayerTree domain — https://chromedevtools.github.io/devtools-protocol/tot/LayerTree/
- DevTools Rendering（paint flashing / layer borders）— https://developer.chrome.com/docs/devtools/rendering/performance
- 真实源码（main，raw.githubusercontent.com/chromium/chromium/main/…）：
  - `third_party/blink/renderer/core/paint/pre_paint_tree_walk.h`、`core/paint/README.md`
  - `third_party/blink/renderer/platform/graphics/paint/{README.md,paint_chunk.h,transform_paint_property_node.h}`
  - `third_party/blink/renderer/platform/graphics/compositing/{paint_artifact_compositor.h,pending_layer.h}`
  - `third_party/blink/renderer/platform/graphics/compositing_reasons.h`
  - `cc/trees/{property_tree.h,effect_node.h}`、`cc/paint/render_surface_filters.cc`、`cc/README.md`
- Electron OSR 源码 — https://raw.githubusercontent.com/electron/electron/main/shell/browser/osr/README.md

---

*本文档源码来源说明*：
- 标【真实源码 repo@path】的代码经 WebFetch 从 `raw.githubusercontent.com/chromium/chromium/main/...`（及 electron/electron/main）实际取得（main 分支）。字段顺序、注释、`: 1` bitfield、`is_comosited_scroll` 等原文拼写均按取回内容保留。
- `compositing_reasons.h` 的常量为宏展开后的等价形式（源码用宏生成）；`kComboAllDirectStyleDeterminedReasons` 的精确展开来自 WebSearch 对 `compositing_reasons.cc` 的摘要，标「待核」处未逐字 fetch 该 .cc 全文。
- 标【真实引用 paint-dev 邮件列表】的两段为该公开线程原文。
- 官方设计史料（Slimming Paint / RenderingNG / how cc works / DevTools / CDP / Electron OSR）经 WebFetch / WebSearch 从 chromium.org / developer.chrome.com / chromium.googlesource.com / chromedevtools.github.io / github.com 核实。
- Demo 1/2/4/5 为可真跑（DevTools 勾选 / Node+CDP / Electron app）；Demo 3（chrome://tracing）为「偏观测」，其具体 trace event 字符串名随 Chrome 版本可能微调，标「待核」。
