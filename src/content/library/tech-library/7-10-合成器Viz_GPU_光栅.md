---
title: "合成器 / Viz / GPU / 光栅（Chromium 域）"
slug: "7-10"
collection: "tech-library"
group: "chromium内核"
order: 7010
summary: "TL;DR 第 9 章把内容「画」成了一份与 DOM 解耦的 display list + 四棵 property tree，并 commit 到了合成线程。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T12:11:34.000Z"
updatedAt: "2026-06-12T12:11:34.000Z"
---
> **TL;DR**
> 第 9 章把内容「画」成了一份与 DOM 解耦的 display list + 四棵 property tree，并 commit 到了合成线程。本章接着讲**这份输入怎么变成屏幕上的像素**——也就是 RenderingNG 流水线 commit 之后的后半程：**raster → activate → submit → aggregate → draw → swap → present**。
> 第一条主线是 **cc（`cc/`）在 renderer 里的合成线程**。注意一个反直觉的事实：cc 早就「不是一个 compositor 了」，官方文档原话是 *"It's neither 'the' chrome compositor ... nor a compositor at all any more. danakj suggests 'content collator' as an alternative name."*【真实源码 chromium@docs/how_cc_works.md】。cc 真正干的是：接收 painted input、决定哪些内容可见、把内容**光栅化/解码成 GPU 纹理 tile**、再打包成一个 **compositor frame**（一组 render pass，每个 render pass 是一串 quad）交给 display compositor。它还在合成线程上**独立处理滚动/缩放**，不打扰 Blink。
> 第二条主线是 **多 tree 模型 + Scheduler**。`pending tree`（光栅暂存）/ `active tree`（绘制）/ `recycle tree`（复用缓存）三棵 `LayerTreeImpl`，加上一个由 vsync 驱动的状态机 `cc::Scheduler`，把「main 慢、raster 慢、GPU 慢」三种 jank 解耦开——`commit` 阻塞 main、`activate` 等 raster、`draw` 等 deadline，谁慢只拖谁。`pending tree` 存在的唯一理由是**原子性**：一次 JS callstack 里的多处改动必须一起上屏。
> 第三条主线是 **Viz / GPU 进程**。`SubmitCompositorFrame` 之后帧离开 renderer，到 GPU 进程里的 **`SurfaceManager`**；`SurfaceAggregator` 把所有 frame producer（每个 renderer、browser UI）的 frame **按 SurfaceId 递归展开**拼成一个全局 frame，再交给 **`DirectRenderer`（SkiaRenderer）** 录成 Skia DDL，replay 到 GPU、`SwapBuffers`、`Present`。GPU 进程独立存在是**安全（renderer 在 sandbox 里碰不到 GL/D3D 驱动）+ 健壮（驱动崩溃不拖垮浏览器）**两个铁律。renderer 与 GPU 之间隔着 **command buffer**（一个跨进程共享内存 ring buffer + GL ES 风格命令）。
> 第四条主线是 **资源 / 同步**：`SharedImage` + `Mailbox` + `SyncToken` + `GpuMemoryBuffer`。这是「一块纹理在 renderer 光栅、在 GPU 进程被 display compositor 读」时唯一安全的跨进程引用与时序保证，也是 **Electron OSR shared-texture 模式**直接复用的基础设施。
> **Electron 平台特殊性**：Electron 完整继承这整条链，无独立合成/光栅代码。唯一真正咬到引擎内部的是 **OSR（离屏渲染）的 shared-texture 路径**——它把 viz 输出的那块 `SharedImage` 经 IPC（Windows `DuplicateHandle` / macOS `mach_port`+`IOSurface`）暴露给你，让你零拷贝拿到 GPU 纹理；代价是你要亲手管 `SyncToken` 与 use-after-free。这是全书 Chromium-internals × Electron-platform 结合最紧的一处，本章会拿 Electron 官方那篇 `shared_texture/README.md` 的设计自述逐条对照。

---

## 前置依赖

| 需要掌握 | 用于理解 |
|---|---|
| 第 9 章 Paint 与合成化 | 本章的**输入**就是第 9 章的输出：display list + 四棵 cc property tree + `cc::Layer` 列表，已 commit 到合成线程 |
| 第 4 章 调度器与事件循环 | `cc::Scheduler` 是 Chrome 众多 scheduler 之一；BeginFrame 节奏、deadline、跨线程 hop 都建立在 task runner 模型上 |
| 第 3 章 Mojo IPC | `SubmitCompositorFrame`、FrameSink 的 Mojo 接口、GPU channel 都是 Mojo；privileged/unprivileged viz 接口隔离 |
| 第 1/2 章 多进程 / sandbox | 为什么有独立 GPU 进程、renderer 为何碰不到 GL 驱动、command buffer 为何存在 |
| GPU / 纹理 / 光栅基本概念（纹理、双缓冲、vsync、Skia/Ganesh、`SkDeferredDisplayList`）| raster 把 display list 烤成像素、draw 把 quad 录成 GL 命令 |
| RenderingNG 12 阶段全景（animate→…→paint→commit→raster→activate→aggregate→draw）| 本章覆盖 **commit 之后 → present** 这后半程，须知道每阶段跑在哪个线程/进程 |

> 阅读建议：本章与第 9 章是「同一条流水线的上下半场」。第 9 章止于「renderer 侧 compositor frame 怎么产生」，本章从「这帧怎么被光栅、怎么离开 renderer、怎么在 GPU 进程被拼合上屏」接续。两章共用 property tree / compositor frame / render pass / quad / tile 这套术语，本章不再重复定义，只在用到时给 cc 官方 glossary 的原文锚点。

---

## 10.1 设计考古：从「renderer 画 bitmap 交 browser」到「content collator + Viz」

### 10.1.1 被推翻的昨天：软件渲染路径

要懂今天这条复杂的链，先看它替换掉的那条极简的链。早期 Chrome（GPU 加速之前）的模型，官方 *GPU Accelerated Compositing in Chrome* 设计文档（更新于 2014-05）把它放在 Appendix 里：renderer 进程**把整页所有 layer 顺序画进一块共享 bitmap，再把这块 bitmap 经 IPC 交给 browser 进程显示**。

这个模型的问题，设计文档列得很直接，GPU 合成相对它的三条收益是：

> *"Compositing page layers on the GPU can achieve far better efficiency than the CPU (both in terms of speed and power draw)"*
> *"Expensive readbacks aren't necessary for content already on the GPU"*
> *"Parallelism between the CPU and GPU, which can operate at the same time"*
> —— 【真实引用 chromium.org/developers/design-documents/gpu-accelerated-compositing-in-chrome】

一句话：CPU 串行画 + 全页 readback + CPU/GPU 不能并行。三条全是性能与功耗灾难。于是 Chrome 引入了**合成器（compositor）**与「先 paint 后 composite」的两段式：

> *"Rendering occurs in two phases: first paint, then composite. This allows the compositor to perform additional work on a per-compositing-layer basis."*
> —— 【真实引用 同上】

### 10.1.2 为什么必须有一个独立的 GPU 进程

合成搬到 GPU，立刻撞上 sandbox。renderer 在 sandbox 里**不能直接调用 OS 的 3D API**。设计文档把 GPU 进程的存在理由写死成一句：

> *"The GPU process exists primarily for security reasons."*
> *"Restricted by its sandbox, the Renderer process (which contains an instance of Blink and of cc) cannot directly issue calls to the 3D APIs provided by the OS (GL / D3D)."*
> —— 【真实引用 同上】

三条收益（原文）：

> *"Security: The bulk of the rendering logic remains in the sandboxed Renderer process"*
> *"Robustness: A GPU process crash (e.g. due to faulty drivers) doesn't bring down the browser"*
> *"Uniformity: Standardizing on OpenGL ES 2.0 as the rendering API for the browser"*

于是 renderer 与 GPU 之间需要一个**客户端-服务端**的桥：**command buffer**。

> *"The commands accepted by the GPU process are patterned closely after the GL ES 2.0 API (for example there's a command corresponding to glClear, one to glDrawArrays, etc)."*
> Renderer 把命令 *"serializes [them] and puts them in a ring buffer (the command buffer) residing in memory shared between itself and the server process."*
> —— 【真实引用 同上】

这条「sandbox 隔离 + command buffer 桥接」是理解后面所有「为什么纹理要用 `SharedImage`/`Mailbox` 跨进程引用、为什么要 `SyncToken`」的根。renderer 永远不直接持有 GL 纹理对象，它持有的是一个**能在 GPU 进程里被解析成真纹理的句柄**。

### 10.1.3 为什么合成又搬到了一条独立线程（threaded compositor）

GPU 合成解决了「画得快」，但没解决「main 忙时还能不能动」。设计文档第三部分引入 threaded compositor，动机一句话：

> *"In theory, the threaded compositor is fundamentally tasked with taking enough information from the main thread to produce frames independently in response to future user input, even if the main thread is busy and can't be asked for additional data."*
> —— 【真实引用 同上】

落到结构上，就是 main thread 持 `LayerTreeHost`、impl thread 持 `LayerTreeHostImpl`，两棵 layer 树**概念上完全分离**：

> *"This means the main thread can be busy running JavaScript and the compositor can still ... redraw previously-committed content on the GPU without interruption."*
> —— 【真实引用 同上】

RenderingNG 文档把这条收益升格成设计原则：

> *"Separating the main and compositor threads is critically important for performance isolation of animation and scrolling from main thread work."*
> *"Separating Viz into its own process is good for stability in the face of bugs in GPU drivers or hardware. It's also good for security isolation."*
> —— 【真实引用 developer.chrome.com/docs/chromium/renderingng-architecture】

### 10.1.4 「cc」这个名字的考古：它早就不是 compositor 了

把上面三步合起来，就有了今天 `cc/` 这个目录。但它的名字是历史包袱。官方 `how_cc_works.md` 开篇 tl;dr 原话：

> *"[cc/] is historically but inaccurately called the Chrome Compositor. It's neither 'the' chrome compositor (of course we have many), nor a compositor at all any more. danakj suggests 'content collator' as an alternative name."*
> —— 【真实源码 chromium@docs/how_cc_works.md, tl;dr】

cc 今天的真实职责（原文）：

> *"cc is responsible for taking painted inputs from its embedder, figuring out where and if they appear on screen, rasterizing and decoding and animating images from the painted input into gpu textures, and finally forwarding those textures on to the display compositor in the form of a compositor frame. cc also handles input forwarded from the browser process to handle pinch and scroll gestures responsively without involving Blink."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

注意它**两个 embedder**：browser 进程经 `ui/compositor` 用单线程版 cc；renderer 进程经 Blink/RenderWidget 用多线程版 cc。原文给了选择理由：

> *"the browser uses the single-threaded version as its main thread is cheap and light, whereas the renderer uses the multi-threaded version as its main thread (Blink) can be quite busy on some pages."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

还有一个全章高频出现、必须先解释的术语——**「impl」后缀**。cc 里 `impl` **不是** "implementation" 的常规含义，而是「跑在合成线程上」：

> *"In cc, 'impl' means that the class is used on the compositor thread and not on the main thread."*
> 历史：jamesr@ 咨询 nduca@，后者论证「合成线程上的东西是 main thread 版本的实现细节」，于是 `LayerImpl`；进而 `LayerTreeImpl`、`LayerTreeHostImpl`，「impl thread」「impl-side painting」之名由此而来。
> —— 【真实源码 chromium@docs/how_cc_works.md, "Impl" 节，附 WebKit bug 55013】

**记住这条，否则后面 `LayerTreeHostImpl` / `pending_tree_` / `DrawLayers` 全是「合成线程上的事」会读不顺。**

> 设计考古小结：今天这条「合成线程 cc → command buffer → GPU 进程 Viz」的复杂链，是三次解耦的叠加产物——**画得快**（合成搬 GPU，2010 前后）、**崩不垮 & 安全**（GPU 独立进程 + command buffer）、**main 忙也能动**（合成搬独立线程）。Viz（`components/viz`）则是更晚一步：把「display compositor」从 browser 进程进一步抽到 GPU 进程、统一所有 frame producer 的聚合。理解每一层「在解耦什么」，比记住类名重要得多。

---

## 10.2 多 tree 模型与 commit / activate（合成线程的核心数据流）

### 10.2.1 四棵 tree，永远活着 2~3 棵

cc 在合成线程上维护多棵 `LayerTreeImpl`。`how_cc_works.md` 原文：

> *"There are four types of layer trees, although there always exists 2-3 at any given time:*
> *• Main thread tree (cc::Layers, main thread, always exists)*
> *• Pending tree (cc::LayerImpl, compositor thread, staging for rasterization, optional)*
> *• Active tree (cc::LayerImpl, compositor thread, staging for drawing, always exists)*
> *• Recycle tree (cc::LayerImpl, compositor thread, mutually exclusive with pending tree)"*
> —— 【真实源码 chromium@docs/how_cc_works.md, Trees: commit / activation】

注意一个吐槽点（原文自嘲）——这些「tree」其实早已是 list 不是 tree：

> *"These are called 'trees' as historically they have been trees and they exist in cc/trees/, but they are all lists and not trees (sorry)."*

cc/README.md 的 glossary 给了 pending / active 的精确定义：

> **Pending Tree**: *"The set of layers and property trees that is generated from a main frame (or BeginMainFrame, or commit). The pending tree exists to do raster work in the layer compositor without clobbering the active tree until it is done. This allows the active tree to be used in the meantime."*
> **Active Tree**: *"The set of layers and property trees that was/will be used to submit a CompositorFrame from the layer compositor. Composited effects such as scrolling, pinch, and animations are done by modifying the active tree."*
> —— 【真实源码 chromium@cc/README.md, Glossaries】

### 10.2.2 commit：阻塞 main、原子拷贝，不走 IPC

`commit` 是把数据从 main thread 原子搬到合成线程的手段。关键反直觉点：**它不是 IPC，而是阻塞 main + 直接拷内存**。

> *"Commit is a method of getting data atomically from the main thread to the compositor thread. (Even when running in single threaded mode, this operation occurs to move data into the right data structures.) Rather than sending an ipc, commit is done by blocking the main thread and copying data over."*
> —— 【真实源码 chromium@docs/how_cc_works.md, Commit Flow】

谁来搬？只有一个类能同时碰两边数据结构——`ProxyImpl`：

> *"ProxyImpl is the only class that accesses data structures on both the main thread and the compositor thread. It only accesses the LayerTreeHost and Layers on the main thread when the main thread is blocked and enforces this through accessor DCHECKs."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

`ProxyMain`（main 侧，归 `LayerTreeHost` 所有）发 `NotifyReadyToCommit` 并交出 mutex 阻塞自己；scheduler ready 时回 `ScheduledActionCommit`；`ProxyImpl` 在合成线程把 main（此刻被冻住）的数据拷进合成线程结构，再释放 mutex 放 main 走。单线程下 `SingleThreadProxy` 一肩挑两边。

### 10.2.3 activate：等光栅完成才换上 active tree

`commit` 把数据放进 **pending tree**。但 pending tree 还不能上屏——它得先把内容光栅化完。等光栅好了，`activate` 把 pending → active：

> **Activate** = *"an analogous operation to commit, and pushes data from the pending tree to the active tree."*
> —— 【真实源码 chromium@cc/README.md】

**pending tree 为什么必须存在**？原文给了唯一理由——**原子性**：

> *"The reason the pending tree exists is that if there are multiple changes to webpage content in a single Javascript callstack (e.g. an html canvas has a line drawn on it, while a div moves, and some background-color changes to blue), these all must be presented to the user atomically. ... The pending tree is the staging area to wait until all of the asynchronous rasterization work is complete. While the pending tree is staging all the rasterization work, the active tree can be updated with animations and scrolling to still be responsive to the user."*
> —— 【真实源码 chromium@docs/how_cc_works.md, Trees】

**recycle tree** 纯粹是省分配的优化：pending 一旦 activate，它就变成 recycle tree，作为「上一次 pending 的缓存」复用，避免每帧重建 `LayerImpl` 和重推 property——*"This is merely an optimization."*

单线程版（browser UI）没有 pending tree，直接 commit 到 active tree；代价是 active tree 在 tile 没全 ready 前不能画。原文：*"Single-threaded versions of cc do not have a pending tree and commit directly to the active tree."*

### 10.2.4 tree 同步：靠 layer id 对位 push

commit/activate 时怎么把一棵树「推」到另一棵？靠 **layer id 对位**：

> *"A layer with id 5 on the main thread tree will push to layer id 5 on the pending tree. That pending layer will push to a layer with id 5 on the active tree. If that layer doesn't exist, during the push it will be created. Similarly layers that no longer exist in the source tree are removed from the destination tree. This is all done via the tree synchronization process."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

> **本节小结（一张时序心智图）**：main 改了内容 → 请求 commit → scheduler 安排 `BeginMainFrame`（带上合成线程已经吃掉的 scroll delta）→ Blink 跑完 rAF/lifecycle → `commit`（冻 main、`ProxyImpl` 拷进 **pending tree**）→ 放 main 走 → **raster pending tree 的 tile** → 全 ready → `activate`（pending→active，旧 active 变 recycle）→ active tree `DrawLayers` 产出 compositor frame。**commit 只阻塞 main 一瞬，raster 慢只拖 activate，互不串味**——这就是多 tree 模型的全部价值。

---

## 10.3 Scheduler：用一个状态机把三种 jank 解耦

### 10.3.1 它是谁、吃什么、吐什么

cc 不是被动等调用，它由 `cc::Scheduler` 自驱。原文先把它放进 Chrome「一堆 scheduler」的语境：

> *"cc's actions are driven by a cc::Scheduler. This is one of many schedulers in Chrome, including the Blink scheduler, the viz::DisplayScheduler, the browser UI task scheduler, and the gpu scheduler."*
> —— 【真实源码 chromium@docs/how_cc_works.md, Scheduling】

它的输入/输出（原文）：

> *"It takes various inputs (visibility, begin frame messages, needs redraw, ready to draw, ready to activate, etc). These inputs drive the cc::SchedulerStateMachine, which then determines actions for the SchedulerClient (LayerTreeHostImpl) to take, such as 'Commit' or 'ActivateSyncTree' or 'PrepareTiles'."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

`SchedulerClient` 这套 action 在头文件里是一组纯虚函数（节选）——这就是状态机「命令」合成线程干活的接口：

```cpp
// 【真实源码 chromium@cc/scheduler/scheduler.h, class SchedulerClient（节选）】
virtual void ScheduledActionPrepareTiles() = 0;
virtual void ScheduledActionInvalidateLayerTreeFrameSink(bool needs_redraw) = 0;
virtual void ScheduledActionPerformImplSideInvalidation() = 0;
// 一帧处理完的回调；注意 last_activated_args 未必等于 WillBeginImplFrame 收到的那个
virtual void DidFinishImplFrame(const viz::BeginFrameArgs& last_activated_args) = 0;
virtual void DidNotProduceFrame(const viz::BeginFrameAck& ack,
                                FrameSkippedReason reason) = 0;
virtual void OnBeginImplFrameDeadline() = 0;
// ...
// class CC_EXPORT Scheduler : public viz::BeginFrameObserverBase { ... }
```

`Scheduler` 本身 `public viz::BeginFrameObserverBase`——**它是 viz BeginFrameSource 的观察者**，vsync 节奏由 GPU 进程的 display compositor 经 `viz::BeginFrameSource` 推过来。

### 10.3.2 两种 BeginFrame：Impl vs Main

这是 scheduler 最核心的区分：

> *"cc::Scheduler code differentiates begin frames from the display compositor as BeginImplFrame (i.e. should cc produce a compositor frame) and a begin frame for its embedder as BeginMainFrame (i.e. should cc tell Blink to run requestAnimationFrame and produce a commit ...). The BeginImplFrame is driven by a viz::BeginFrameSource which in turn is driven [by] the display compositor."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

理想的全流水线一次更新，顺序固定（原文，背下来）：

> *"BeginImplFrame -> BeginMainFrame -> Commit -> ReadyToActivate -> Activate -> ReadyToDraw -> Draw."*

### 10.3.3 raster 慢时怎么办：pipelining 与 high-latency mode

关键设计：**raster 慢，main 不用干等**。原文给了一个 raster 慢时的真实交错例子：

> *"BeginImplFrame1 -> BeginMainFrame1 -> Commit1 -> (slow raster) -> BeginImplFrame2 -> BeginMainFrame2 -> ReadyToActivate1 -> Activate1 -> Commit2 -> ReadyToDraw1 -> Draw1."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

机制是：第二个 `BeginMainFrame` 可以在 activation 之前就发出，但它会在 `NotifyReadyToCommit` 处阻塞，直到上一个 pending tree activate 完（状态机不允许同时存在两个未 activate 的 pending tree）。这样 main 能并行做下一帧，代价是 latency。

还有一个 deadline 机制与 **high latency mode**：

> *"The cc::Scheduler maintains a deadline by which it expects its embedder to respond. If the main thread is slow to respond, then the Scheduler may draw without waiting for a commit. If this happens, then Scheduler is considered to be in high latency mode. ... High latency mode trades off latency for throughput by increasing pipelining."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

**这解释了一个生产现象**：页面卡顿时，你的滚动/动画往往还「能动但延迟变大」——这就是 scheduler 进了 high-latency mode，用 throughput 换 latency，宁可帧晚一点也不掉帧。

---

## 10.4 Raster 与 TileManager：把 display list 烤成 GPU 纹理 tile

### 10.4.1 tile 是什么、谁拥有它

`PictureLayerImpl` 把内容按 scale 切成 `PictureLayerTiling`，每格是一个 `cc::Tile`。cc/README.md glossary：

> **Tiles**: *"An abstraction of a piece of content of a Layer. A tile may be rasterized or not. It may be known to be a solid color or not. A PictureLayerImpl indirectly owns a sparse set of Tiles to represent its rasterizable content. When tiles are invalidated, they are replaced with new tiles."*
> —— 【真实源码 chromium@cc/README.md】

tile 尺寸的真实启发式（这条很实用，调试时 layer borders 看到的格子大小就来自这里）：

> *"for software raster tiles are roughly 256x256 px and for gpu raster tiles are roughly viewport width x one quarter viewport height."*
> —— 【真实源码 chromium@docs/how_cc_works.md, PictureLayer】

紧跟着原文有一句活灵活现的警告，关于改 raster scale 启发式：

> *"There are a number of heuristics to determine when and how to change rasterization scales. These aren't perfect, but change them at your own peril. 🐉🐉🐉"*
> —— 【真实源码 chromium@docs/how_cc_works.md】

### 10.4.2 TileManager：调度优先级光栅 + 图像解码

`TileManager` 是「光栅化整个 tile 世界」的总调度。它的类注释把职责和**优先级分层**写得极清楚（这是全章最值得逐字读的一段源码）：

```cpp
// 【真实源码 chromium@cc/tiles/tile_manager.h, class TileManager 注释（逐字）】
// This class manages tiles, deciding which should get rasterized and which
// should no longer have any memory assigned to them. Tile objects are "owned"
// by layers; they automatically register with the manager when they are
// created, and unregister from the manager when they are deleted.
//
// The TileManager coordinates scheduling of prioritized raster and decode work
// across 2 different subsystems, namely the TaskGraphRunner used primarily for
// raster work and images which must be decoded before rasterization of a tile
// can proceed, and the CheckerImageTracker used for images decoded
// asynchronously from raster using the |image_worker_task_runner|. The order in
// which work is scheduled across these systems is as follows:
//
// 1) RequiredForActivation/Draw Tiles: These are the highest priority tiles
// which block scheduling of any decode work for checkered-images.
//
// 2) Pre-paint Tiles: These are offscreen tiles which fall within the
// pre-raster distance. ...
//
// 3) Pre-decode Tiles: These are offscreen tiles which are outside the
// pre-raster distance but have their images pre-decoded and locked. ...
```

`PrepareTiles` 是入口，其头注释精确说明了「何时跑、跑完通知谁」：

```cpp
// 【真实源码 chromium@cc/tiles/tile_manager.h, PrepareTiles 注释（逐字）】
// Assigns tile memory and schedules work to prepare tiles for drawing.
// This step occurs after Commit and at most once per BeginFrame. It can be
// called on its own, that is, outside of Commit.
//
// - Runs client_->NotifyReadyToActivate() when all tiles required for
// activation are prepared, or failed to prepare due to OOM.
// - Runs client_->NotifyReadyToDraw() when all tiles required draw are
// prepared, or failed to prepare due to OOM.
bool PrepareTiles(const GlobalStateThatImpactsTilePriority& state);
```

**这两个回调 `NotifyReadyToActivate` / `NotifyReadyToDraw` 正是上一节 scheduler 状态机的输入**——光栅子系统与调度状态机就是在这里咬合的。

cc/README.md glossary 对 **Prepare Tiles** 的定义补充了「它还顺带踢图像解码」：

> **Prepare Tiles**: *"Prioritize and schedule needed tiles for raster. This is the entry point to a system that converts painting (raster sources / recording sources) into rasterized resources that live on tiles. This also kicks off any dependent image decodes for images that need to be decode[d] for the raster to take place."*
> —— 【真实源码 chromium@cc/README.md】

### 10.4.3 TaskGraph：整图调度、不可动态改、已启动不可取消

`TileManager` 把要做的活生成一张 `TaskGraph` 丢给 worker。这套调度有三条硬约束（背下来，否则会误以为能「中途插队」）：

> *"Once the TileManager decides the set of work to do, it generates a TaskGraph with dependencies and schedules that work across worker threads. TaskGraphs are not updated dynamically, but instead rescheduled as a whole graph. Tasks cannot be cancelled once they have started running. Scheduled tasks that have not yet started are cancelled by submitting another graph that does not include them."*
> —— 【真实源码 chromium@docs/how_cc_works.md, Raster and tile management】

### 10.4.4 两种 raster 模式 × 三种 RasterBufferProvider

cc 的 raster 有两种模式，且**全局非此即彼，切换会销毁所有资源**：

> *"There are currently two modes of raster in cc: software raster ... gpu raster .... It is always in one mode or the other. ... Switching modes destroys all resources. A common reason for switching modes is that the gpu process has crashed too much and all of Chrome switches from gpu to software raster and compositing modes."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

raster 模式 × 合成模式的组合，决定用哪个 `RasterBufferProvider`。原文三选一：

> *"• ZeroCopyRasterBufferProvider: rasters software bitmaps (a) for software compositing into shared memory that is read directly by the software compositor and (b) for gpu compositing directly into a GpuMemoryBuffer (e.g. IOSurface), which is memory that can be mapped by CPU and used by the GPU*
> *• OneCopyRasterBufferProvider: rasters software bitmaps for gpu compositing into shared memory, which are then uploaded to gpu memory in the gpu process*
> *• GpuRasterBufferProvider: rasters gpu textures for gpu compositing over a command buffer via paint commands (for gpu raster)"*
> —— 【真实源码 chromium@docs/how_cc_works.md, Raster Buffer Providers】

一个易踩的真坑（原文）：**gpu raster 受 context 锁限制，一次只能一个 worker 线程**，但图像解码可以并行：

> *"due to locks on the context, gpu raster is limited to one worker thread at a time, although image decoding can proceed in parallel on other threads. This single thread limitation is solved with a lock and not with thread affinity."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

#### 三种 RasterBufferProvider 对照表

| Provider | raster 产物 | 拷贝次数 | 适配的合成模式 | 典型场景 / 边界 |
|---|---|---|---|---|
| `ZeroCopyRasterBufferProvider` | 软件 bitmap → 直写共享内存 / `GpuMemoryBuffer`(IOSurface) | 0 拷贝 | 软件合成；或 GPU 合成但 CPU 光栅 | 直写 IOSurface 这类 CPU 可映射 GPU 可用的内存；省拷贝但依赖平台 GpuMemoryBuffer 支持 |
| `OneCopyRasterBufferProvider` | 软件 bitmap → 共享内存 → 再 upload 到 GPU | 1 拷贝（upload） | GPU 合成 + CPU 光栅 | 没有 zero-copy 路径时的兜底；upload 有带宽成本 |
| `GpuRasterBufferProvider` | GPU 纹理（经 command buffer 发 paint 命令） | 不走 CPU bitmap | GPU 合成 + GPU 光栅（OOP-R） | 现代默认；最快，但 context 锁 → 单 worker 串行光栅 |

> **不适用边界**：Chrome **从不混用「软件合成 + 硬件光栅」**——原文 *"Chrome never mixes software compositing with hardware raster, but the other three combinations of raster mode x compositing mode are valid."*。即合成模式比光栅模式更「保守」：软件合成必配软件光栅；硬件合成可配软/硬光栅。

### 10.4.5 图像解码为何被特殊对待

raster 里最贵的是图像解码，尤其相对快得多的 gpu raster：

> *"Image decoding receives a lot of special care in the TileManager, as they are the most expensive part of raster, especially relative to comparatively speedy gpu raster. Each decode receives its own dependent task in the task graph. There is a separate decode cache for software raster vs gpu raster."*
> —— 【真实源码 chromium@docs/how_cc_works.md, Image Decoding】

还有一个冷知识：**动图 GIF 的每帧都是合成线程自发起一棵新 pending tree**（不经 main）：

> *"When gifs animate, they generate a new pending tree (initiated by the compositor thread instead of the main thread) with some raster invalidations and then re-raster tiles that are covered by that gif."*
> —— 【真实源码 chromium@docs/how_cc_works.md】

> **本节小结**：raster 的关键不是「怎么把矢量画成像素」（那是 Skia 的事），而是 **cc 怎么决定「先光栅哪些 tile、用哪种 buffer、怎么不卡死 worker」**。`TileManager` 的三层优先级（activation/draw 必需 → pre-paint → pre-decode）+ 整图 `TaskGraph` + 单/多线程 raster 锁，构成了「在有限显存与 worker 下，尽量先让能上屏的内容 ready」的调度逻辑。`NotifyReadyToActivate/Draw` 两个回调把它接回 scheduler。

---

## 10.5 跨进程的边界：compositor frame 离开 renderer，进入 Viz

### 10.5.1 「draw」「swap」在 cc 里都是名不副实

active tree ready 后，cc「draw」。但 cc 早就不真画也不真 swap，原文专门澄清这两个词的真实含义：

> *"'Draw' in cc means constructing a compositor frame full of quads and render passes for eventual drawing on screen. 'Swap' in cc means submitting that frame to the display compositor via a CompositorFrameSink. These frames get sent to the viz SurfaceAggregator where compositor frames from all frame producers are aggregated together."*
> —— 【真实源码 chromium@docs/how_cc_works.md, Content Data Flow Overview】

`LayerTreeHostImpl::DrawLayers` 的头注释印证了「draw 只是产 frame」并点出 `PrepareToDraw`/`DidDrawAllLayers` 的配对契约：

```cpp
// 【真实源码 chromium@cc/trees/layer_tree_host_impl.h（逐字节选）】
// Returns `DrawResult::kSuccess` unless problems occurred preparing the
// frame, and we should try to avoid displaying the frame. If
// `PrepareToDraw()` is called, `DidDrawAllLayers()` must also be called,
// regardless of whether `DrawLayers()` is called between the two.
virtual DrawResult PrepareToDraw(FrameData* frame, bool expects_to_draw = false);
// ...
virtual std::optional<SubmitInfo> DrawLayers(FrameData* frame);
```

### 10.5.2 compositor frame / render pass / quad（cc 侧定义）

cc/README.md glossary（这三条第 9 章已用过，此处给 cc 官方原文做跨进程边界的锚点）：

> **CompositorFrame**: *"A set of RenderPasses (which are a list of DrawQuads) along with metadata. Conceptually this is the instructions (transforms, texture ids, etc) for how to draw an entire scene which will be presented in a surface."*
> **Render Pass**: *"A list of DrawQuads which will all be drawn together into the same render target (either a texture or physical output). ... Additional RenderPasses are used for effects that require a set of DrawQuads to be drawn together into a buffer first."*
> **DrawQuad**: *"A unit of work for drawing. Each DrawQuad has its own texture id, transform, offset, etc."*
> **Shared Quad State**: *"A shared set of states used by multiple draw quads."*
> —— 【真实源码 chromium@cc/README.md】

render pass 的**排序规则**（依赖在前、root 在后；pass 内 quad 按画家算法 back-to-front）原文：

> *"If render pass 1 depends on render pass 9 ... then 9 will appear in the list before 1. Therefore, the root render pass is always last in the list. Inside a single render pass, the quads are ordered back to front (Painter's algorithm)."*
> —— 【真实源码 chromium@docs/how_cc_works.md, Compositor frames】

### 10.5.3 跨进程怎么走：CompositorFrameSink → SurfaceManager（在 GPU 进程）

cc/README.md 的「Composite」与跨进程那段，把 renderer→GPU 的交接讲死：

> *"'Drawing' in a compositor consists of LayerImpl::AppendQuads which batches up a set of DrawQuads and RenderPasses into a CompositorFrame which is sent via a CompositorFrameSink. CompositorFrames from individual compositors are sent to the SurfaceManager (which is in the GPU process). The SurfaceAggregator combines all CompositorFrames together when asked to by the Display. These are given to the viz::DirectRenderer, which finally draws the entire composited browser contents."*
> —— 【真实源码 chromium@cc/README.md】

注意 **`SurfaceManager` 明文「在 GPU 进程」**。这就是 renderer 与 Viz 的进程边界：`SubmitCompositorFrame` 是一次 Mojo 调用，frame（含对 GPU 纹理资源的 `Mailbox` 引用，而非纹理本身）越过进程墙进入 GPU 进程。

> **关键澄清——两个「composite」**：cc/README.md glossary 把「合成」一词拆成两层，务必区分：
> **Composite**: *"the layer compositor does raster from recordings and manages memory, performs composited effects such as scrolling, pinch, animations, producing a CompositorFrame. The display compositor does an actual 'composite' to draw the final output into a single physical output."*
> 即 **renderer 侧 cc =「layer compositor」（产 frame）**，**GPU 进程侧 viz =「display compositor」（把多个 frame 拼成一块物理输出）**。本章 10.2~10.5.2 是前者，10.6 起是后者。

---

## 10.6 Viz / GPU 进程：display compositor 把所有 frame 拼成一块屏

### 10.6.1 Viz 是什么、跑在哪、谁能调用它

`components/viz/README.md` 原文给定义：

> *"Viz - short for visuals - is the client library and service implementations for compositing and gpu presentation."*
> *"The display compositor uses Gpu or software to composite a set of frames, from multiple clients, into a single backing store for display to the user."*
> —— 【真实源码 chromium@components/viz/README.md】

`services/viz/README.md` 给了**权限隔离**模型（与第 1/2 章的进程/安全主题呼应）：

> *"Privileged Client: responsible for starting and restarting Viz after a crash and for facilitating connections to Viz from unprivileged clients."*
> 接口分三层 `public/` `private/` `privileged/`，核心原则：*"an unprivileged client cannot be provided interfaces by which it can impact the operation of another client."*
> —— 【真实源码 chromium@services/viz/README.md】

**这条隔离原则的实际意义**：一个 renderer（unprivileged）提交 frame 后，**不能拿到任何能影响别的 renderer 的 viz 接口**。browser 进程作为 privileged client，负责拉起/重启 viz、撮合连接。这是 Site Isolation 的合成侧延伸。

### 10.6.2 Display：一块物理输出的控制器

GPU 进程里，`viz::Display` 是「把若干 surface 的 frame 画到一块物理输出」的控制器。头文件类注释（逐字）：

```cpp
// 【真实源码 chromium@components/viz/service/display/display.h, class Display 注释（逐字）】
// A Display produces a surface that can be used to draw to a physical display
// (OutputSurface). The client is responsible for creating and sizing the
// surface IDs used to draw into the display and deciding when to draw.
class VIZ_SERVICE_EXPORT Display : public DisplaySchedulerClient, ...
```

cc/README.md glossary 补充 `Display` 与 `DirectRenderer` 的分工：

> **Display**: *"A controller class that takes CompositorFrames for each surface and draws them to a physical output."*
> **DirectRenderer**: *"An abstraction that provides an API for the Display to draw a fully-aggregated CompositorFrame to a physical output. Subclasses ... provide implementations for various backends, currently GL, Skia, or Software."*
> —— 【真实源码 chromium@cc/README.md】

### 10.6.3 Surface / SurfaceId：frame producer 之间的间接引用

多个 frame producer（每个 renderer、browser UI、OOPIF）怎么互相嵌套？靠 **Surface + SurfaceId**。RenderingNG data-structures 文档：

> *"When a compositor submits a compositor frame, it is accompanied by an identifier, called a surface ID, allowing other compositor frames to embed it by reference."*
> —— 【真实引用 developer.chrome.com/docs/chromium/renderingng-data-structures】

`how_cc_works.md` 给了它在 layer 层的载体——`SurfaceLayer`：

> *"A surface layer has a surface id, which refers to some other stream of compositor frames in the system. This is a way of having an indirection to other compositor frame producers. ... Blink embeds references to out of process iframes via SurfaceLayer."*
> —— 【真实源码 chromium@docs/how_cc_works.md, SurfaceLayer】

`surface.h` 的类注释揭示了一个**生产级难点：客户端不可信 + 依赖可能缺失 → activation deadline**（逐字节选）：

```cpp
// 【真实源码 chromium@components/viz/service/surfaces/surface.h（逐字节选）】
// This pending+active frame mechanism for managing CompositorFrames from a
// client exists to enable best-effort synchronization across clients. A surface
// subtree will remain pending until all dependencies are resolved: all clients
// have submitted CompositorFrames corresponding to a new property of the
// subtree (e.g. a new size).
//
// Clients are assumed to be untrusted and so a client may not submit a
// CompositorFrame to satisfy the dependency of the parent. Thus, by default, a
// surface has an activation deadline associated with its dependencies. If the
// deadline passes, then the CompositorFrame will activate despite missing
// dependencies.
```

**这解释了一个真实视觉 bug 的根因**：调整窗口大小时，OOPIF（或某些 UI）可能在极短时间内出现「旧尺寸内容 + 新尺寸框」的撕裂——就是父 surface 的 dependency（子 frame 的新尺寸 frame）没在 deadline 内到，viz 选择「带缺失依赖也先 activate」以避免整体卡死。设计上是 best-effort，不是 bug。

### 10.6.4 SurfaceAggregator：递归展开成一个全局 frame

`SurfaceAggregator` 把所有 producer 的 frame 拼成一个。`how_cc_works.md` 与 RenderingNG 都讲了它：

> *"These frames get sent to the viz SurfaceAggregator where compositor frames from all frame producers are aggregated together."* —— 【真实源码 chromium@docs/how_cc_works.md】
> *"Multiple compositor frames are submitted to Viz ... an aggregation phase ... converts them into a single, aggregated compositor frame. Aggregation replaces surface draw quads by the compositor frames they specify."* —— 【真实引用 developer.chrome.com/docs/chromium/renderingng-data-structures】

`life_of_a_frame.md` step [10] 给了**递归展开 SurfaceQuad**的精确描述：

> *"Before the actual draw could happen SurfaceAggregator will recursively walk over compositor frames and replace SurfaceQuads (quads produced by SurfaceLayer) with contents of the embedded compositor frame. This step produces single CompositorFrame in the end that can be drawn by the Display Compositor."*
> —— 【真实源码 chromium@docs/life_of_a_frame.md, step [10]】

### 10.6.5 DirectRenderer / SkiaRenderer：录 DDL → replay → swap → present

聚合后的单一 frame 交给 `DirectRenderer`（现代是 `SkiaRenderer`）。`life_of_a_frame.md` step [11]~[17] 把「draw 之后到像素上屏」拆成 7 步，**这是把抽象 compositor frame 落到真 GPU 命令的关键链**：

> *"[11] Draw Frame: ... go over quads and render passes in the aggregated compositor frame and produce draw commands. For SkiaRenderer it's recording of Deferred Display Lists (DDL).*
> *[12] RequestSwap: ... submitted to the GPU thread to replay along with SwapBuffers request.*
> *[13] Wait until ready to draw: ... Chrome uses SyncTokens to ensure this type of synchronization. GPU Task submitted at step [12] won't be scheduled until all associated SyncTokens will be signaled.*
> *[15] GPU draw: ... Skia will be replaying DDLs and issue commands to the GPU.*
> *[16] Swap: ... Submits commands to request displaying framebuffer and/or overlays.*
> *[17] Presentation: ... the display controller started scanning out the results. The pixels are finally visible on the screen."*
> —— 【真实源码 chromium@docs/life_of_a_frame.md, steps [11]–[17]】

一个**易被忽略的 GPU 进程性质**（step [14]，解释「为什么 GPU 不忙却还有延迟」）：

> *"GPU Main Thread does all the GPU work and by the time display compositor is ready to draw it might still be busy doing other tasks (e.g raster for next frame). gpu::Scheduler uses cooperative multi-tasking and can't preempt the current task unless it yields."*
> —— 【真实源码 chromium@docs/life_of_a_frame.md, step [14]】

### 10.6.6 damage / partial swap：只重画变化的部分

display compositor 不会每帧全画。cc/README.md glossary 的 **Damage** 把「为什么要 damage」讲透：

> **Damage**: *"Damage is the equivalent of invalidation, but for the final display. ... Damage is tracked via the DamageTracker. This allows for partial swap, where only the parts of the final CompositorFrame that touch the screen are drawn, and only that drawn portion is swapped, which saves quite a bit of power for small bits of damage."*
> —— 【真实源码 chromium@cc/README.md】

`how_cc_works.md` 补了 damage 的两类来源与 overlay 收益：

> *"There are two types of damage: invalidation damage and expose damage. ... cc calculates damage in the DamageTracker and forwards it along with a CompositorFrame. One reason damage is needed in the display compositor is to do partial swap ... Another reason is when using hardware overlays, such that the display compositor can know that only an overlay was damaged and not have to re-composite the rest of the scene."*
> —— 【真实源码 chromium@docs/how_cc_works.md, Damage】

> **本节小结（跨进程全景）**：renderer 侧 cc 产 frame（含 `Mailbox` 资源引用）→ Mojo `SubmitCompositorFrame` → GPU 进程 `SurfaceManager` 存进对应 `Surface`（pending/active，带 deadline）→ `Display` 要画时 `SurfaceAggregator` 递归把 `SurfaceQuad` 换成被嵌入 frame、产出单一全局 frame → `SkiaRenderer` 录成 DDL → GPU Main 线程 replay（等 `SyncToken`）→ `SwapBuffers` → `Present`。整条链上 **renderer 永不直接持 GL 对象，全靠 `Mailbox`/`SyncToken` 跨进程引用与定序**——这正是下一节资源模型的来历。

---

## 10.7 资源与同步：SharedImage / Mailbox / SyncToken / GpuMemoryBuffer

这一节既是 Chromium 资源模型的核心，也是下一节 Electron OSR 的直接前置。它回答一个问题：**一块纹理在 renderer 光栅、在 GPU 进程被 display compositor 读、可能还在另一个 renderer 的 WebGL 里用——怎么安全引用、怎么知道 GPU 用完了？**

### 10.7.1 三件套的职责

- **`SharedImage`**：跨进程纹理的「持有者」。它持有一个 `Mailbox`，指向 GPU 进程里真正的 `SharedImageBacking`。`SharedImageInterface::CreateSharedImage` 可以接受一个 `GpuMemoryBufferHandle`，里面装着平台原生句柄。
- **`Mailbox`**：一个轻量、可序列化的「纹理身份」。renderer 传 `Mailbox` 而非纹理对象本身越过进程墙。
- **`SyncToken`**：跨 context/进程的**定序原语**。它回答「GPU 在某 context 上用完这块资源了吗」，是避免「主进程释放了帧，GPU 还在读」这类 use-after-free 的唯一手段。
- **`GpuMemoryBuffer`**：平台原生共享内存的抽象——Windows NT HANDLE、macOS `IOSurfaceRef`、Linux `NativePixmapHandle`（每 plane 一个 fd）。

这些不是我归纳的二手描述，而是 Electron 在 `shared_texture/README.md` 里**对照 Chromium 内部实现写下的设计自述**（基于 Electron 37 / Chromium 137），下一节会逐条引用原文。

### 10.7.2 为什么需要 SyncToken：异步 GPU 调用的定序

`life_of_a_frame.md` step [13] 已经点了 `SyncToken` 在 display 路径里的作用（GPU task 等所有 `SyncToken` signaled 才调度）。Electron 的设计自述把「跨进程多 `SharedImage` 引用同一 `Mailbox`」的危险讲得更具体：

> *"Most GPU calls are asynchronous, sending to the command buffer of the GPU process through the GpuChannel of each client process. When we have two SharedImage instances referencing the same Mailbox in two different processes, we don't know when the GPU has finished using the resources. Typically, this is guaranteed by SyncToken."*
> —— 【真实源码 electron@shell/common/api/shared_texture/README.md】

**这条是整个跨进程 GPU 资源模型的灵魂**：command buffer 是异步的（renderer 发命令进 ring buffer，GPU 进程稍后执行），所以「谁先用完」不能靠 CPU 侧的时间顺序判断，必须靠 GPU 侧 signal 的 `SyncToken`。

---

## 10.8 Electron 平台特殊性：OSR 与 shared-texture 把 Viz 资源暴露给你

### 10.8.1 OSR 是什么、消费的是哪一帧

Electron 没有独立合成/光栅代码，完整继承上述全链。唯一真正咬到引擎内部的是 **OSR（offscreen rendering）**。官方文档定义：

> *"Offscreen rendering lets you obtain the content of a BrowserWindow in a bitmap or a shared GPU texture, so it can be rendered anywhere, for example, on texture in a 3D scene. The offscreen rendering in Electron uses a similar approach to that of the Chromium Embedded Framework project."*
> —— 【真实源码 electron@docs/tutorial/offscreen-rendering.md】

关键：**OSR 消费的正是本章末端 viz 那帧合成结果**。第 9 章讲过 software 模式拷 bitmap；本章重点是 **shared-texture 模式如何直接拿 viz 输出的 GPU 纹理**。

三种 OSR 模式（官方原文）：

> *"1. Use GPU shared texture (useSharedTexture: true) ... The frames are directly copied in GPU textures, thus this mode is very fast because there's no CPU-GPU memory copies overhead, and you can directly import the shared texture to your own rendering program.*
> *2. Use CPU shared memory bitmap (useSharedTexture: false, default) ... The frame has to be copied from the GPU to the CPU bitmap which requires more system resources.*
> *Software output device ... uses a software output device for rendering in the CPU."*
> —— 【真实源码 electron@docs/tutorial/offscreen-rendering.md】

两个易踩的官方注意点（原文）：

> *"When webPreferences.offscreen.useSharedTexture is false, the maximum frame rate is 240 ..."*
> *"When nothing is happening on a webpage, no frames are generated."*
> —— 【真实源码 electron@docs/tutorial/offscreen-rendering.md】

**第二条直接对应 10.3 的 scheduler**：页面无变化→无 BeginMainFrame damage→不产 frame→无 `paint` 事件。OSR 的「省电」本质就是 cc scheduler 的 on-demand 帧驱动。

### 10.8.2 shared-texture：把 Viz 的 SharedImage 经 IPC 交到你手上

`OffscreenSharedTexture` 结构体（官方）暴露的字段，正是 viz/media 那套 `VideoFrame` 元数据：

> `pixelFormat`（`rgba`/`bgra`/`rgbaf16`）、`codedSize`、`colorSpace`、`visibleRect`、`contentRect`（OSR 下即 dirty 区）、`handle`([SharedTextureHandle])、以及 `release` 函数。
> *"Only a limited number of textures can exist at the same time, so it's important that you call texture.release() as soon as you're done with the texture."*
> —— 【真实源码 electron@docs/api/structures/offscreen-shared-texture.md】

为什么 `paint` 事件里能直接拿到一个跨进程可用的句柄？Electron 设计自述给了根因——**Chromium 的 IPC 替你把句柄复制/转移好了**：

> *"In fact, Chromium's IPC internally handles all the concerns about this (duplicate handles for remote processes, passing mach_port through mach_port), which is why the OSR paint event can use the handle directly - because IPC has transparently handled these issues."*
> —— 【真实源码 electron@shell/common/api/shared_texture/README.md】

平台差异（原文，做客户端 staff 必须知道这三套）：

> *"For Windows, a shared D3D11 texture can be created by GetSharedHandle (deprecated) or CreateSharedHandle. The deprecated method generates a non-NT HANDLE ... the newer one generates an NT HANDLE that is local to the current process. To share it with other processes, you need to call DuplicateHandle ...*
> *For macOS, IOSurface can also be global by setting kIOSurfaceIsGlobal ... If you want to share an IOSurface with other processes, you need to create a mach_port from it and pass the mach_port through a previously created IPC ..."*
> —— 【真实源码 electron@shell/common/api/shared_texture/README.md】

### 10.8.3 为什么用 SharedImage 而不是 WebGPU/Dawn 直接导

Electron 作者试过 Dawn 的 `WGPUSharedTextureMemory`，放弃了，理由很能说明 `SharedImage` 在跨进程的不可替代性：

> *"I initially considered using WebGPU Dawn Native API to import the external texture as a WGPUSharedTextureMemory, but encountered more problems. For example, it was unable to export, difficult to manage the lifetime of a frame ... SharedImage has advantages when it comes to sharing across processes because it holds a reference to a Mailbox, which points to the corresponding SharedImageBacking in the GPU process. Therefore, I use SharedImageInterface->ImportSharedImage and ClientSharedImage->Export to serialize sufficient information to retrieve the SharedImage reference in another process."*
> —— 【真实源码 electron@shell/common/api/shared_texture/README.md】

### 10.8.4 真坑：ownership 与 use-after-free（生产级）

这是 OSR shared-texture 最容易炸的地方，作者列了两条具体崩溃模式：

> *"For Windows, calling CloseHandle on a non-NT HANDLE is an invalid operation and will cause Chromium to crash. Therefore, you cannot use a non-NT HANDLE when importing. ... when calling importSharedTexture, an NT HANDLE will be duplicated. For macOS, IOSurface is a reference-counted resource. ... we can simply let Chromium retain this resource and increment the reference count."*
> —— 【真实源码 electron@shell/common/api/shared_texture/README.md】

释放时序的正解（用 `SyncToken` 串起来）：

> *"When you call release() on the imported shared texture object, if you've used VideoFrame and imported it into a WebGPU pipeline, it will wait for WebGPU to finish rendering, then run a callback to notify you to release dependent resources, such as the original imported object in the main process, the source texture, etc."*
> —— 【真实源码 electron@shell/common/api/shared_texture/README.md】

> **Electron 小结**：OSR 是 Electron 唯一深度耦合 Chromium 合成内部的地方。software/CPU-bitmap 模式简单但每帧 GPU→CPU 拷贝；shared-texture 模式零拷贝、直接复用 viz 的 `SharedImage`/`Mailbox`/`GpuMemoryBuffer`/`SyncToken` 全套基础设施，代价是你接管了跨进程纹理的生命周期与定序——平台句柄（NT HANDLE / IOSurface+mach_port）、`release()` 时机、`SyncToken` 依赖，任何一处错都是 crash 或 use-after-free。Electron 没有「绕过」Chromium 这套机制，而是「把它薄薄包了一层暴露给 JS」。

---

## 10.9 五类硬货

### 10.9.1 ⭐Demo（重中之重）

> Chromium 全量 build 不现实，故全部 demo 以「在真实引擎上观测 + 小实验 + 最小 Electron app + CDP 驱动」为主，每个给完整命令/代码 + 步骤 + 预期输出，并与前文源码呼应。**标「可真跑」的均为不依赖 Chromium 源码 build 的可执行步骤；标「偏观测」的依赖具体 Chrome 版本，trace 字符串名可能微调，标「待核」。**

---

#### Demo 1 ·【可真跑】用 layer borders + tile 看「分层 + 256/视口 tile」

**呼应**：10.4.1「software tile≈256×256，gpu tile≈视口宽×1/4 视口高」、第 9 章 layerization。

步骤：
1. 任意 Chrome/Edge/Electron 打开一个有滚动、有 `transform`/`will-change` 的页面（如 `https://www.chromium.org`）。
2. DevTools → 右上 `⋮` → More tools → **Rendering**。
3. 勾选 **Layer borders**。
4. （可选）开 **Frame Rendering Stats** 看 FPS / GPU memory。

预期输出：
- 橙/黄色边框 = 合成 layer 边界（对应 `cc::Layer`，第 9 章 layerization 的产物）。
- 蓝色网格 = **tile 边界**。在 GPU 光栅下，tile 多为「视口宽 × 约 1/4 视口高」的长条；强制软件光栅（见 Demo 5）后会变成接近 256×256 的方格。
- 滚动时观察：只有进入/接近视口的 tile 才被光栅（对应 10.4.2 `TileManager` 的 pre-paint 优先级）。

> 注：这是验证 10.4 tile 启发式最直观的方式。**层多≠好**——第 9 章讲过 layerization 是省显存 vs 可独立 raster 的权衡；这里能直接看到层与 tile 数量。

---

#### Demo 2 ·【可真跑】合成线程滚动 vs 主线程卡死（off-main-thread scroll 对比）

**呼应**：10.1.3 / 10.2 / 10.3——「main 忙时合成线程仍能滚」。

完整可跑 HTML（存为 `scroll.html`，双击用浏览器打开）：

```html
<!doctype html><meta charset=utf8>
<style>body{height:6000px;background:linear-gradient(#eef,#fcc);}
.box{position:fixed;top:10px;left:10px;padding:8px;background:#000;color:#0f0;font:14px monospace}</style>
<div class=box id=s>scroll me; then click to block main for 4s</div>
<script>
document.body.addEventListener('click', () => {
  s.textContent = 'MAIN BLOCKED 4s — try scrolling now';
  const end = performance.now() + 4000;
  while (performance.now() < end) {}      // 同步死循环，卡死 main thread
  s.textContent = 'main unblocked';
});
</script>
```

步骤与预期输出：
1. 打开页面，正常滚动——丝滑。
2. 点击页面触发 4 秒主线程死循环。
3. **立刻尝试滚动**：页面**仍能平滑滚动**（合成线程在 active tree 上独立处理 scroll，10.2.3）。
4. 但此时点击/JS 交互无响应——证明卡的是 main，不是合成线程。
5. 对照：把滚动改成 `onscroll` 里做重活，或给元素加同步 wheel handler，会发现滚动被迫回主线程（对应 10.1 原文「同步 JS touch/wheel handler 的输入必须回 Blink」）。

> 这个实验直接证明 threaded compositor 的核心价值：*"the main thread can be busy running JavaScript and the compositor can still redraw previously-committed content"*（10.1.3 原文）。

---

#### Demo 3 ·【偏观测，待核】chrome://tracing 看 PipelineReporter 全流水线分段

**呼应**：10.3 scheduler 流水线、10.5/10.6 跨进程、`life_of_a_frame.md` 17 步。

步骤：
1. 地址栏开 `chrome://tracing`（新版可能跳 Perfetto UI，用 `https://ui.perfetto.dev` 等价）。
2. Record → 选 categories 至少包含 `cc`、`viz`、`gpu`、`blink`（或直接选 "Rendering" 预设）。
3. 在另一标签滚动/播放动画几秒，回来 Stop。
4. 在结果里搜 **`PipelineReporter`**。

预期输出（与 `life_of_a_frame.md` 原文逐段对应）——每个 PipelineReporter 被切成这些段：

> `BeginImplFrameToSendBeginMainFrame` → `SendBeginMainFrameToCommit` → `Commit` → `EndCommitToActivation` → `Activation` → `EndActivateToSubmitCompositorFrame` → `SubmitCompositorFrameToPresentationCompositorFrame`
> 末段又细分：`SubmitToReceiveCompositorFrame` / `ReceiveCompositorFrameToStartDraw` / `StartDrawToSwapStart` / `Swap` / `SwapEndToPresentationCompositorFrame`
> —— 【真实源码 chromium@docs/life_of_a_frame.md, PipelineReporter trace events】

观察要点：
- 哪一段最长 = 瓶颈在哪。`SendBeginMainFrameToCommit` 长 = main/Blink 慢；`Commit→Activation` 之间长 = raster 慢（10.4）；`StartDrawToSwapStart`/`Swap` 长 = GPU 进程 draw/swap 慢（10.6.5）。
- 多个 PipelineReporter **重叠** = 多帧流水线并行（10.3.3 high-latency/pipelining），`life_of_a_frame.md` 的 Example 1~3 给了三种重叠形态。

> 标「待核」原因：trace event 命名随 Chrome 版本演进可能微调；上面这组名字取自 main 分支 `life_of_a_frame.md`，与你本地 Chrome 版本若不符以本地为准。

---

#### Demo 4 ·【可真跑】CDP（`LayerTree` domain）枚举合成层与 quad

**呼应**：10.2 layer、10.5.2 quad/compositingRect。用 `--remote-debugging-port` + 原生 WebSocket，零依赖第三方库。

完整可跑 Node 脚本（`layers.mjs`，Node ≥18 自带 `WebSocket`/`fetch`）：

```js
// 1) 先起一个带远程调试的浏览器：
//    macOS:  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
//              --remote-debugging-port=9222 --user-data-dir=/tmp/cdp https://www.chromium.org
//    Electron 同理：electron 你的app --remote-debugging-port=9222
// 2) node layers.mjs
const list = await (await fetch('http://127.0.0.1:9222/json')).json();
const page = list.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
const send = (method, params={}) =>
  new Promise(res => { const i = ++id; pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params })); });

ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  if (m.method === 'LayerTree.layerTreeDidChange') {
    const layers = m.params.layers || [];
    console.log(`\n[layerTreeDidChange] 合成层数 = ${layers.length}`);
    for (const L of layers.slice(0, 12))
      console.log(`  layerId=${L.layerId} size=${L.width}x${L.height}`
        + ` paints=${L.paintCount} drawsContent=${L.drawsContent}`);
  }
};
ws.onopen = async () => {
  await send('DOM.enable'); await send('LayerTree.enable');   // 开 LayerTree domain
  console.log('LayerTree enabled; 滚动/交互页面看合成层变化，Ctrl+C 退出');
};
```

预期输出（示例形态，数值依页面而变）：
```
LayerTree enabled; 滚动/交互页面看合成层变化，Ctrl+C 退出
[layerTreeDidChange] 合成层数 = 7
  layerId=21 size=1280x4012 paints=3 drawsContent=true
  layerId=24 size=1280x40   paints=1 drawsContent=true
  ...
```

观察要点：
- `LayerTree.enable` 后每次合成层结构变化都会推 `layerTreeDidChange`，列出的 `layerId`/`size`/`paintCount` 直接对应 10.2 的 `LayerImpl`。
- 进一步可调 `LayerTree.compositingReasons {layerId}` 拿到「这层为什么被提升为合成层」（与第 9 章 compositing reasons 呼应）。
- 这是把 10.2「合成层是什么」从概念变成可枚举数据的最小手段。

---

#### Demo 5 ·【可真跑】强制软件光栅 / 看 GPU 进程状态：chrome://gpu 与 chrome://process-internals

**呼应**：10.1.2 GPU 进程存在理由、10.4.4 raster 模式切换、10.6 viz。

A. **chrome://gpu**（任意 Chromium 内核浏览器）：
- 打开后看 "Graphics Feature Status"。预期：`Canvas / Compositing / Rasterization / WebGL` 多为 *Hardware accelerated*。
- 若某项是 *Software only*，下方 "Problems Detected" / driver bug list 会给原因——这对应 10.4.4「gpu 崩太多次会全局切软件光栅/合成」。

B. **强制软件光栅看 tile 变化**（与 Demo 1 对照）：
```
# 关硬件加速（等价于 app.disableHardwareAcceleration / 全局软件路径）
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --disable-gpu --user-data-dir=/tmp/swr https://www.chromium.org
```
- 再开 Rendering → Layer borders：tile 网格会从「长条」变成接近 **256×256 方格**（10.4.1 软件 tile 启发式）。
- chrome://gpu 里多项会变 *Software only*。

C. **chrome://process-internals**（看进程边界）：
- 可看到 renderer / GPU 进程的关系，印证 10.1.2 / 10.6 的「合成的 display 侧在 GPU 进程」。

> 这组观测把「为什么要 GPU 进程、软件/硬件光栅的真实差异、viz 在哪个进程」三件事一次性落到可见状态。

---

#### Demo 6 ·【可跑 / 偏 build 走读】最小 Electron OSR app（software 模式可真跑；shared-texture 给走读）

**呼应**：10.8 全节。

**software 模式（可真跑，零原生模块）** —— 即官方 fiddle 的最小化：

```js
// main.js  ;  package.json 里 "main":"main.js"，依赖 electron
// 运行： npx electron .
const { app, BrowserWindow } = require('electron')
const fs = require('node:fs')

app.disableHardwareAcceleration()        // 走 Software output device（10.8.1）

app.whenReady().then(() => {
  const win = new BrowserWindow({
    width: 800, height: 600, show: false, // OSR 窗口不显示
    webPreferences: { offscreen: true }    // 开 OSR；默认 useSharedTexture:false
  })
  win.loadURL('https://www.chromium.org')
  let n = 0
  win.webContents.on('paint', (event, dirty, image) => {
    // dirty 即本帧脏矩形（对应 10.6.6 damage / 10.4 局部 raster）
    if (++n === 30) {                      // 抓第 30 帧落盘验证
      fs.writeFileSync('osr.png', image.toPNG())
      console.log('saved osr.png; dirty =', JSON.stringify(dirty))
      app.quit()
    }
  })
  win.webContents.setFrameRate(30)         // 节流帧率（10.3 scheduler on-demand）
})
```

预期输出：当前目录出现 `osr.png`（页面截图），控制台打印每帧 `dirty` 矩形。**注意「页面静止时 `paint` 不触发」**（10.8.1 原文 *"When nothing is happening ... no frames are generated"*）——若页面已静止可能凑不满 30 帧，把阈值调小或 reload 即可。

**shared-texture 模式（偏 build / 原生模块，给体系走读 + 最小补丁式骨架）**：

shared-texture 需要 `webPreferences:{ offscreen:{ useSharedTexture:true } }`，且 `paint` 事件给的是 `texture`（含 `textureInfo.handle` 平台句柄），**不能直接用 JS 处理纹理**——必须把句柄交给原生 node 模块/你的 GPU 程序导入。官方明示：*"This is an advanced feature requiring a native node module to work with your own code."*（10.8.1）。最小骨架（不可纯 JS 跑通，仅示意接线）：

```js
// 【示意，非逐字】shared-texture 接线骨架——真正 import 纹理需原生代码
const win = new BrowserWindow({
  webPreferences: { offscreen: { useSharedTexture: true } }
})
win.webContents.on('paint', (event, dirty, image) => {
  const tex = event.texture                  // OffscreenSharedTexture（10.8.2 结构）
  if (!tex) return
  const info = tex.textureInfo               // pixelFormat/codedSize/colorSpace/handle...
  try {
    // 把 info.handle（SharedTextureHandle）交给你的原生模块：
    //  - Windows: 内部已 DuplicateHandle 成 NT HANDLE（10.8.2）
    //  - macOS:   IOSurface 引用计数 +1（10.8.4）
    nativeImporter.import(info)              // 你的 .node：经 SharedImageInterface 导入
  } finally {
    tex.release()                            // 必须尽快 release，否则纹理配额耗尽（10.8.2）
  }
})
```

走读要点（对应 patch/原生体系，非伪装可跑）：
- `texture.release()` 的时序受 `SyncToken` 约束：若纹理进了 WebGPU/WebGL 管线，要等 GPU 用完才能安全 release（10.8.4 原文）。
- 真正可运行的官方端到端示例在 Electron 仓库 `spec/api-shared-texture-spec.ts` + `spec/fixtures/api/shared-texture/{preload,renderer}.js`（README 指向），涉及 `startTransferSharedTexture` 把 `SharedImage` 转给 renderer 进程——这是 build/原生层，故此处只走读不冒充可跑。

---

### 10.9.2 方案对比

#### 对比 A：renderer 侧 cc（layer compositor）vs GPU 进程 viz（display compositor）

| 维度 | cc（layer compositor，renderer） | viz（display compositor，GPU 进程） |
|---|---|---|
| 进程/线程 | renderer 进程的合成线程（impl thread） | GPU 进程的 display compositor 线程 + GPU main |
| 输入 | display list + property tree（第 9 章 commit 来） | 多个 producer 的 `CompositorFrame`（含 `Mailbox`） |
| 「composite」含义 | raster + 管理显存 + scroll/pinch/动画，**产 1 个 frame** | **聚合多个 frame** 成一块物理输出 |
| 产物 | `CompositorFrame`（quad/render pass，资源是引用） | 屏幕像素 / framebuffer / overlay |
| 核心类 | `LayerTreeHostImpl` / `TileManager` / `cc::Scheduler` | `Display` / `SurfaceAggregator` / `SkiaRenderer` |
| 不适用边界 | 不直接持 GL 对象、不真画屏 | 不做 raster（raster 是 cc/worker 的事，gpu raster 在 GPU 进程但属 cc 链路） |

依据：cc/README.md「Composite」glossary 区分两层 composite（10.5.3）。

#### 对比 B：三种 OSR 模式（Electron）

| 模式 | 触发 | 拷贝 / 路径 | 速度 | 适用场景 | 不适用 / 边界 |
|---|---|---|---|---|---|
| Software output device | `app.disableHardwareAcceleration()` + `offscreen:true` | CPU 软件 output device | 比 CPU-bitmap GPU 模式快 | 无 GPU / 服务器抓帧 / 不需 WebGL | 不支持 WebGL/3D CSS 硬件路径 |
| CPU shared-memory bitmap | `offscreen:{ useSharedTexture:false }`（默认） | GPU→CPU 拷 bitmap，`NativeImage` 取用 | 慢（每帧回拷），≤240fps | 需 GPU 功能但只要 bitmap 结果 | 每帧 GPU→CPU 拷贝是带宽瓶颈 |
| GPU shared texture | `offscreen:{ useSharedTexture:true }` | 零拷贝，直接给 `SharedImage` 句柄 | 最快 | 把页面当 GPU 纹理喂进自己 3D 管线 | 须原生模块；须手管 `SyncToken`/`release`，错则 crash/UAF |

依据：Electron `offscreen-rendering.md` + `shared_texture/README.md`（10.8）。

#### 对比 C：三种 RasterBufferProvider

见 **10.4.4** 的对照表（含「软件合成从不配硬件光栅」的边界），此处不重复。

---

### 10.9.3 失败模式 / 生产真坑 / 根因

| # | 失败模式（现象） | 根因 | 出处锚点 |
|---|---|---|---|
| 1 | 页面卡顿但滚动「能动只是变迟钝」 | scheduler 进 **high-latency mode**，用 throughput 换 latency、宁迟不掉帧 | 10.3.3【how_cc_works】 |
| 2 | 改 raster scale 启发式后大面积重光栅/显存暴涨 | raster scale 启发式牵一发动全身，官方明确警告「🐉🐉🐉 at your own peril」 | 10.4.1【how_cc_works】 |
| 3 | 以为能「中途取消某个 raster 任务」却无效 | `TaskGraph` 整图调度，**已启动任务不可取消**，只能靠提交新图剔除未启动任务 | 10.4.3【how_cc_works】 |
| 4 | GPU 进程反复崩后整页突然变糊/变慢 | gpu 崩太多次 → 全局从 gpu raster/合成**切到软件**，`Switching modes destroys all resources` | 10.4.4【how_cc_works】 |
| 5 | resize 时 OOPIF/UI 出现「旧内容+新框」短暂撕裂 | surface 依赖未在 **activation deadline** 内到，viz 选择带缺失依赖先 activate（best-effort） | 10.6.3【surface.h】 |
| 6 | gpu raster 没跑满多核 worker | gpu raster 受 **context 锁**限制单 worker 串行（解码可并行） | 10.4.4【how_cc_works】 |
| 7 | OSR `paint` 不触发 | 页面无 damage → 不产 frame；与「省电」是同一机制 | 10.8.1【offscreen-rendering】 |
| 8 | OSR shared-texture Windows 崩溃 | 对 non-NT HANDLE 调 `CloseHandle` 非法 → Chromium crash；导入须用 NT HANDLE 并 `DuplicateHandle` | 10.8.4【shared_texture/README】 |
| 9 | OSR shared-texture 纹理「闪烁/错帧/UAF」 | 跨进程多 `SharedImage` 引用同一 `Mailbox`，未用 `SyncToken` 等 GPU 用完就 `release` | 10.7.2 / 10.8.4【shared_texture/README】 |
| 10 | OSR shared-texture「拿不到纹理 / 配额耗尽」 | 同时存活纹理数有限，未及时 `texture.release()` | 10.8.2【offscreen-shared-texture】 |

---

### 10.9.4 五件套

**① 一句话定位**
本章 = RenderingNG「commit 之后到屏幕像素」的后半程：**renderer 合成线程 cc（content collator，产 compositor frame）→ command buffer 跨进程 → GPU 进程 viz（display compositor，聚合+绘制+swap+present）**，外加贯穿其间的 **多 tree 模型 / Scheduler / TileManager / SharedImage 资源模型**，以及 Electron 唯一深耦合点 **OSR shared-texture**。

**② 三个最容易记错的点**
- 「cc」**不是 compositor**，是 content collator；「impl」**不是 implementation**，是「合成线程上」。（10.1.4）
- **commit 不是 IPC**，是「阻塞 main + 直接拷内存」；`ProxyImpl` 是唯一能碰两边的类。（10.2.2）
- **renderer 永不直接持 GL 纹理**；跨进程靠 `Mailbox` 引用 + `SyncToken` 定序 + `GpuMemoryBuffer` 承载平台句柄。（10.7）

**③ 一条主链（背诵版）**
`BeginImplFrame → BeginMainFrame → Commit(冻 main，拷进 pending tree) → raster tiles(TileManager/TaskGraph) → ReadyToActivate → Activate(pending→active) → DrawLayers(产 CompositorFrame) → SubmitCompositorFrame(Mojo 入 GPU 进程 SurfaceManager) → SurfaceAggregator 递归展开 SurfaceQuad → SkiaRenderer 录 DDL → GPU replay(等 SyncToken) → SwapBuffers → Present`。

**④ pending tree / activation deadline / SyncToken 的「存在理由」一句话**
- pending tree：一次 JS callstack 的多处改动**必须原子上屏**，需要一个暂存区等所有异步 raster 完成。（10.2.3）
- activation deadline：客户端**不可信**且依赖可能永不到，必须有 deadline 让 best-effort activate，避免整体卡死。（10.6.3）
- SyncToken：command buffer **异步**，CPU 侧时间顺序不能判断「GPU 用完没」，必须 GPU 侧 signal。（10.7.2）

**⑤ Electron 一句话**
Electron 不改合成/光栅一行核心代码；OSR 把 viz 那帧结果以 bitmap（拷）或 `SharedImage`（零拷贝、复用 `Mailbox`/`SyncToken`/`GpuMemoryBuffer` 全套）暴露给你——shared-texture 的全部复杂度（平台句柄、生命周期、定序）都是 Chromium 跨进程资源模型的复杂度，Electron 只薄薄包了一层。

---

## 延伸阅读（均经 WebFetch / WebSearch 核实 URL 可达）

- How cc Works（本章主干，content collator / 多 tree / scheduler / raster 全景）— https://chromium.googlesource.com/chromium/src/+/main/docs/how_cc_works.md
- Life of a frame（17 步 + PipelineReporter 分段，Demo 3 依据）— https://chromium.googlesource.com/chromium/src/+/HEAD/docs/life_of_a_frame.md
- cc/README.md（权威 glossary：CompositorFrame/RenderPass/Quad/Tile/Damage/Composite…）— https://chromium.googlesource.com/chromium/src/+/main/cc/README.md
- components/viz/README.md（display compositor 定义）— https://chromium.googlesource.com/chromium/src/+/HEAD/components/viz/README.md
- services/viz/README.md（privileged/unprivileged 接口隔离）— https://chromium.googlesource.com/chromium/src/+/HEAD/services/viz/
- GPU Accelerated Compositing in Chrome（2014 设计文档：GPU 进程/command buffer/threaded compositor 的史料）— https://www.chromium.org/developers/design-documents/gpu-accelerated-compositing-in-chrome/
- RenderingNG architecture（线程/进程归属 + 流水线阶段）— https://developer.chrome.com/docs/chromium/renderingng-architecture
- Key data structures in RenderingNG（surface/compositor frame/render pass/quad/tile）— https://developer.chrome.com/docs/chromium/renderingng-data-structures
- Chromium Graphics 设计文档总入口 — https://www.chromium.org/developers/design-documents/chromium-graphics/
- Electron OSR 教程 — https://www.electronjs.org/docs/latest/tutorial/offscreen-rendering
- Electron `shared_texture/README.md`（SharedImage/Mailbox/SyncToken/GpuMemoryBuffer 设计自述）— https://github.com/electron/electron/blob/main/shell/common/api/shared_texture/README.md
- Electron `OffscreenSharedTexture` 结构 — https://www.electronjs.org/docs/latest/api/structures/offscreen-shared-texture
- 真实源码（main 分支，raw.githubusercontent.com/chromium/chromium/main/… 与 electron/electron/main/…）：
  - `cc/README.md`、`docs/how_cc_works.md`、`docs/life_of_a_frame.md`
  - `cc/tiles/tile_manager.h`、`cc/scheduler/scheduler.h`、`cc/trees/layer_tree_host_impl.h`
  - `components/viz/service/display/display.h`、`components/viz/service/surfaces/surface.h`、`components/viz/service/display/surface_aggregator.h`
  - `electron/docs/tutorial/offscreen-rendering.md`、`electron/shell/common/api/shared_texture/README.md`、`electron/docs/api/structures/offscreen-shared-texture.md`

---

*本文档源码来源说明*

- 标【真实源码 repo@path】的代码与注释，均经 WebFetch / curl 从 `raw.githubusercontent.com/chromium/chromium/main/...`（及 `electron/electron/main/...`）与 `chromium.googlesource.com/.../main/...` 实际取得（main / HEAD 分支，2026-06 取材）。注释原文（含 `🐉🐉🐉`、`(sorry)`、`danakj suggests "content collator"` 等口语）按取回内容逐字保留。
- `cc/scheduler/scheduler.h`、`cc/trees/layer_tree_host_impl.h`、`cc/tiles/tile_manager.h`、`components/viz/service/{display/display.h, surfaces/surface.h}` 的代码块为**逐字节选**（截取了 `SchedulerClient` 部分纯虚函数、`PrepareToDraw/DrawLayers` 头注释、`TileManager`/`PrepareTiles` 类注释、`Display`/`Surface` 类注释），省略号处为为聚焦而裁剪，未改写保留行的文字。
- 设计史料（GPU Accelerated Compositing in Chrome / RenderingNG architecture / data structures）经 WebFetch 从 chromium.org 与 developer.chrome.com 核实；引用段落标【真实引用 域名】。GPU 合成设计文档时间戳为其页面标注的 Updated May 2014。
- Demo 1/2/4/5/6(software) 为**可真跑**（DevTools 勾选 / 纯浏览器 HTML / Node 原生 WebSocket+CDP / `chrome://` 页面 / 最小 Electron app，均不依赖 Chromium 源码 build）。Demo 6 的 shared-texture 骨架标【示意，非逐字】，因其必须配原生 node 模块、无法纯 JS 跑通，故只给接线走读 + 指向官方 `spec/api-shared-texture-spec.ts`。
- Demo 3（chrome://tracing PipelineReporter）标「待核」：trace event 字符串名取自 main 分支 `life_of_a_frame.md`，随 Chrome 版本可能微调，以本地实际 trace 为准。
- 全文未对 `surface_aggregator.h` / `display.h` 的**实现 .cc** 做逐行 fetch；聚合/绘制的算法细节以官方文档（`life_of_a_frame.md` step [10][11]）的原文描述为准，未声称逐字源码处即为文档原文引用。
