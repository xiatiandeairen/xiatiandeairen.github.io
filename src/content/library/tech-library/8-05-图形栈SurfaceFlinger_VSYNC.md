---
title: "图形栈 SurfaceFlinger / VSYNC(Android 域)"
slug: "8-05"
collection: "tech-library"
group: "android系统"
order: 8005
summary: "读者画像:你写过多线程、用过 producer/consumer 队列、知道双缓冲/三缓冲是什么、读过一点 OpenGL 或图形管线、能读 C++ 和模板。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:27:02.000Z"
updatedAt: "2026-06-12T11:27:02.000Z"
---
> 读者画像:你写过多线程、用过 producer/consumer 队列、知道双缓冲/三缓冲是什么、读过一点 OpenGL 或图形管线、能读 C++ 和模板。本章不教"怎么 `setContentView` 让 View 显示出来"那一层应用语义,而是把 **一帧画面从 app 的 BufferQueue 生产、到 SurfaceFlinger 合成、再到屏幕刷新,这条流水线如何被 VSYNC 这个"心跳"对齐到一起** 拆到源码级,并给你能在真机/模拟器上亲手抓一帧、看 VSYNC 拍子、读 SurfaceFlinger 内部模型的脚本。
>
> **TL;DR(先给结论,再展开)**
> - 整个 Android 图形栈是 **一条三段流水线**:`app 渲染 → SurfaceFlinger 合成 → HWC/显示控制器扫描出图`。三段各自跑在不同节奏上,**VSYNC 是把三段对齐的同一个时钟拍子**。没有 VSYNC,三段各跑各的,结果是撕裂(tearing)和卡顿(jank)。
> - **生产者/消费者用 BufferQueue 解耦**:app 是 producer,`dequeueBuffer`(拿空 buffer)→ 画 → `queueBuffer`(交回);SurfaceFlinger 是 consumer,`acquireBuffer` → 合成 → `releaseBuffer`。buffer 在 `FREE/DEQUEUED/QUEUED/ACQUIRED` 四态间流转,**两端通过内核 fence 同步、不搬像素**。【真实源码 platform/frameworks/native@libs/gui/BufferQueueProducer.cpp】
> - **三缓冲(triple buffering)的真身**就是 `mMaxDequeuedBufferCount`:producer 最多能同时持有几个 buffer 在画。这个数 = 2 时是双缓冲(producer 画一个、consumer 显示一个,producer 画完得等),= 3 时多一个周转 buffer,**让 producer 偶尔慢一帧也不必整条流水线停摆**。代码里就是一句 `dequeuedCount >= mMaxDequeuedBufferCount` 的判断。【真实源码 …@BufferQueueProducer.cpp:322】
> - **VSYNC 不是直接从硬件 IRQ 喂给 app 的**。SurfaceFlinger 用 **HW_VSYNC 时间戳喂一个软件模型**,模型反过来 **预测** 未来的 VSYNC 时刻,再按 phase offset **分别** 唤醒 app(`VSYNC-app`)和 SF 自己(`VSYNC-sf`)。这层"软件 PLL / 预测器"是 Project Butter(Android 4.1)引入、十年间从 `DispSync`(PLL)演进到 `VSyncPredictor`(最小二乘线性回归)。【真实源码 …@services/surfaceflinger/Scheduler/VSyncPredictor.cpp】
> - **为什么要预测而不是直接用硬件中断?** 三个理由:(1)硬件 VSYNC 中断有抖动和延迟,模型能滤噪、给出比中断本身更稳的拍子;(2)**预测让"提前唤醒"成为可能** —— 要在 VSYNC 时刻交出合成结果,就得在 VSYNC **之前** `workDuration` 纳秒把线程叫醒开始干活,这个减法 `wakeup = predictedVsync - workDuration - readyDuration` 是整个调度的keystone;(3)稳定后可以**关掉硬件 VSYNC 中断省电**,纯靠模型跑。【真实源码 …@VSyncDispatchTimerQueue.cpp:108】
> - 现代 AOSP 把这套拆成 **三个正交组件**:`VSyncTracker`(=`VSyncPredictor`,管"模型/预测,什么时候是下一帧")、`VSyncDispatch`(=`VSyncDispatchTimerQueue`,管"调度,提前多久把谁叫醒")、`VSyncController`(=`VSyncReactor`,管"硬件 VSYNC 开关 / 喂样本")。三者由 `VsyncSchedule` 组合。这是 monolithic `DispSync` 被拆开后的清晰职责划分。【真实源码 …@VsyncSchedule.h】
>
> **前置知识**:producer/consumer 与有界队列、双/三缓冲与撕裂、条件变量(`std::condition_variable`)、fence/同步原语的概念、最小二乘线性回归(`slope`/`intercept`)、相位(phase)与周期(period)、`atrace`/Perfetto 大概是什么。
>
> **本章用到的源码出处(均实际 WebFetch 取过,见章末"取材记录")**
> - 生产者/消费者:`platform/frameworks/native@libs/gui/BufferQueueProducer.cpp`、`libs/gui/include/gui/BufferQueueCore.h`(googlesource `main` 分支,现代)
> - VSYNC 调度:`platform/frameworks/native@services/surfaceflinger/Scheduler/` 下 `VSyncDispatch.h`、`VSyncPredictor.{h,cpp}`、`VSyncDispatchTimerQueue.cpp`、`VsyncSchedule.h`、`VSyncReactor.cpp`(googlesource `main`,现代)
> - 设计考古:legacy `DispSync.{h,cpp}`(`android-7.1.1_r28` tag);AOSP 官方 VSync 文档;Project Butter(Google I/O 2012)二手史料
> - 官方文档:source.android.com `implement-vsync` / `surfaceflinger-props`;perfetto.dev frametimeline

---

## 一、设计考古:从"卡顿的 Android"到 Project Butter

### 1.1 病:Android 4.0 之前为什么"不跟手"

2012 年之前,Android 给人的体感是"卡、不跟手",而同期的 iOS 很顺滑。问题不在 CPU/GPU 不够快,而在 **图形管线三段之间没有共同的拍子**:

- app 主线程在某个时刻渲染、把 buffer 交给系统;
- SurfaceFlinger 在另一个时刻把各 Surface 合成;
- 显示控制器在又一个时刻把帧扫描到屏幕。

三段各跑各的,典型后果是两个:

- **撕裂(tearing)**:屏幕正在扫描上半屏旧帧时,下半屏的 buffer 被换成了新帧,一屏里出现两帧的拼接。
- **卡顿/掉帧(jank)**:某一帧 app 渲染慢了一点点(比如 17ms 而不是 16ms),错过了这一轮显示窗口,就得整整等下一轮,体感是"顿一下"。更糟的是 **没有缓冲余量时,一次掉帧会引发连锁掉帧**。

> 设计动机一句话:**要顺滑,就必须让"app 画"、"SF 合成"、"屏幕刷新"三件事踩在同一个 16.6ms(60Hz)的节拍上,并且给流水线留出缓冲余量,让偶尔的一次慢帧不至于停摆整条线。**

### 1.2 药:Project Butter 的三味(Google I/O 2012)

Android 4.1 Jelly Bean 的 **Project Butter**(黄油计划)正是冲这个去的,它由三块组成【真实史料 二手,WebSearch 命中 AndroidPolice 2012-07-12 深度解析、TechCrunch 2012-06-27 发布报道、多份独立资料一致复述;一手 Google I/O 2012 keynote 视频未直接 WebFetch,关键时间点标「待核」】:

| 组件 | 解决什么 | 本质 |
|---|---|---|
| **VSYNC** | 三段流水线没有共同拍子 | 把 app 渲染、touch、合成、屏幕刷新全部对齐到一个 ~16ms 的"心跳" |
| **Triple Buffering** | 双缓冲下一次慢帧 = 流水线停摆 | 多加一个周转 buffer,让 producer 偶尔慢一帧也能继续往前画 |
| **Choreographer** | app 不知道"该什么时候开始画" | app 注册 VSYNC 回调,在拍子上 `doFrame`(input→animation→traversal),不再随便乱画 |

这三味是配套的,缺一不可:

- 光有 VSYNC 没有 triple buffering,慢帧照样停摆;
- 光有 triple buffering 没有 VSYNC,缓冲是满了,但三段还是各跑各的、照样撕裂;
- 光有前两者没有 Choreographer,app 不知道在拍子上画,VSYNC 这个"指挥棒"指挥不到 app。

> 这一章的主线就是把这三味拆到源码:**BufferQueue + 三缓冲**(第二、三节)是"流水线和缓冲",**VSYNC 模型 + 调度**(第四、五、六节)是"拍子和指挥棒"。

### 1.3 十年演进:从 PLL 到线性回归预测器

VSYNC 这一味,内部实现十年间被重写过。理解这条线索很重要,因为网上大量博客还停在老实现:

**第一代(Android 4.1 ~ 9):`DispSync` —— 软件锁相环(PLL)。**

AOSP 官方文档对它的定义是逐字的:

> 【真实文档 source.android.com/docs/core/graphics/implement-vsync】"DispSync is a software phase-lock loop (PLL) that generates the VSYNC and SF_VSYNC signals used by Choreographer and SurfaceFlinger"
>
> 类注释(legacy):"It maintains a model of the periodic hardware-based vsync events of a display and uses that model to execute period callbacks at specific phase offsets from the hardware vsync events."【真实源码 platform/frameworks/native@services/surfaceflinger/DispSync.h, android-7.1.1_r28】

它的工作方式:`addResyncSample(timestamp)` 喂硬件 VSYNC 时间戳进来,内部用 PLL 的方式拟合周期与相位;`addPresentFence(fence)` 用实际显示完成的 fence 时间戳来 **验证模型有没有漂移**;漂移超过阈值就重新同步。那个阈值是一个写死的魔数:

```cpp
static const nsecs_t kErrorThreshold = 160000000000; // 400 usec squared
```
【真实源码 platform/frameworks/native@services/surfaceflinger/DispSync.cpp, android-7.1.1_r28】—— 注意它是"误差的平方"的阈值(400μs 的平方),代码里用残差平方和判断模型精度。

**第二代(Android 10+):`VSyncReactor` + `VSyncPredictor` + `VSyncDispatch` —— 拆分 + 线性回归。**

Android 10 的 SurfaceFlinger 大重构里,monolithic 的 `DispSync` 被拆成三个职责正交的组件(下文第四节详述),其中"预测下一帧 VSYNC 时刻"的核心从 PLL 换成了 **简单线性回归(simple linear regression / 最小二乘)**:把最近 N 个硬件 VSYNC 时间戳当作 Y、把它们的序号当作 X,拟合一条直线,**斜率 slope 就是 VSYNC 周期、截距 intercept 就是相位**。这段回归代码我们会在第四节逐行读。

> 为什么换?PLL 是连续反馈系统,参数(增益)难调、对突变(比如刷新率从 60Hz 切到 120Hz)响应不够利落;而可变刷新率(VRR)、多刷新率(LTPO 屏 1~120Hz)在 Android 10 后成了刚需。**线性回归对"换一批样本、立刻重算一条新直线"更直接,配合多个 `VsyncTimeline` 还能优雅处理刷新率切换时的相位接续。** 这是从"控制论思路"到"统计拟合思路"的转变。

---

## 二、整体架构:三段流水线 + 一帧的全链路

### 2.1 三段流水线与三个 VSYNC

先建立全局心智模型。一帧画面流经三段,每段被一个 VSYNC 信号"踩点":

```
   VSYNC-app(相位 offset_app)          VSYNC-sf(相位 offset_sf)        HW_VSYNC(硬件原点)
        │                                    │                              │
        ▼                                    ▼                              ▼
 ┌─────────────┐   BufferQueue      ┌─────────────────┐   HWC commit   ┌──────────────┐
 │  app 进程    │  (producer→consumer)│  SurfaceFlinger  │ ─────────────▶ │ 显示控制器/屏  │
 │  渲染一帧     │ ─────────────────▶ │  合成所有 Layer   │                │ 扫描出图       │
 │ (Choreographer│   queueBuffer     │ (HWC or GPU)    │                │              │
 │  doFrame)    │                    │                 │                │              │
 └─────────────┘                    └─────────────────┘                └──────────────┘
   UI/RenderThread                    SF main thread                     panel refresh
        │                                    │                              │
        └──── 提前 workDuration 唤醒 ────────┘                              │
              (因为要在 VSYNC 时刻交出结果,必须提前开始干活)                  │
                                                                            │
   全部时间基准 = 喂给软件模型的 HW_VSYNC 时间戳(VSyncPredictor 拟合) ◀──────┘
```

关键认知:

- **三个 VSYNC 不是三个硬件信号**。硬件只有一个 `HW_VSYNC`(显示控制器每次刷新发的中断)。`VSYNC-app` 和 `VSYNC-sf` 是 SurfaceFlinger 用软件模型 **从 `HW_VSYNC` 派生** 出来的两个"虚拟拍子",各自带一个相位偏移(phase offset)。
- **相位偏移的意义**:让 app 比 SF 早一点醒、SF 比屏幕刷新早一点醒,这样三段工作 **流水线式重叠**,而不是串行排队。偏移配错会增加端到端延迟或引发掉帧(第七节有坑)。
- **fence 同步贯穿全链路**:buffer 在两端之间转移所有权时不拷像素,靠 GPU/显示 fence 告诉对方"这块 buffer 我画完了/显示完了你可以收回了"。

### 2.2 一帧的全链路(60Hz,无 offset 简化版)

把一帧拆成事件序列(忽略相位偏移,假设三段都在 VSYNC 时刻对齐):

```
t=0ms   HW_VSYNC 到达 → SurfaceFlinger::onVsyncReceived(timestamp)
        → VSyncPredictor::addVsyncTimestamp(t)   把样本喂进模型,更新 slope/intercept
        → 模型预测下一个 VSYNC 在 t=16.6ms

t≈0ms   VSYNC-app 触发 → app 进程 Choreographer.doFrame()
        → input → animation → measure/layout → draw(录制 DisplayList)
        → RenderThread: dequeueBuffer() 拿一个 FREE buffer → GPU 画 → queueBuffer() 交回
           (buffer: FREE → DEQUEUED → QUEUED)

t≈0ms   VSYNC-sf 触发 → SurfaceFlinger 主线程被 VSyncDispatch 唤醒
        → 从各 Layer 的 BufferQueue acquireBuffer() 取最新 QUEUED buffer
           (buffer: QUEUED → ACQUIRED)
        → 合成:能交给 HWC 的图层走 HWC overlay,剩下的用 GPU(RenderEngine)合成
        → HWC::presentDisplay() 提交给显示控制器

t=16.6ms 下一个 HW_VSYNC → 显示控制器把合成结果扫描到屏幕
        → 之前 ACQUIRED 的 buffer 显示完,releaseBuffer() 还给 producer
           (buffer: ACQUIRED → FREE,producer 可以再 dequeue)
```

这条链路的"对齐"全靠 VSYNC;"不停摆"全靠 BufferQueue 里有足够的 buffer 周转。下面分别拆。

---

## 三、生产者/消费者:BufferQueue 与三缓冲的真身

### 3.1 buffer 的四态与两端职责

BufferQueue 是 app(producer)和 SurfaceFlinger(consumer)之间的有界缓冲队列。每个 buffer slot 在四个状态间流转:

| 状态 | 谁持有 | 含义 |
|---|---|---|
| **FREE** | 队列 | 空闲,可被 producer dequeue |
| **DEQUEUED** | producer | producer 拿走了,正在往里画 |
| **QUEUED** | 队列 | producer 画完交回,等 consumer 取 |
| **ACQUIRED** | consumer | consumer 取走了,正在合成/显示 |

环路:`FREE →(dequeueBuffer)→ DEQUEUED →(queueBuffer)→ QUEUED →(acquireBuffer)→ ACQUIRED →(releaseBuffer)→ FREE`。

source 里这个状态划分体现在 `BufferQueueCore` 的几个集合上:

```cpp
// mSlots is an array of buffer slots that must be mirrored on the producer
// side. This allows buffer ownership to be transferred between the producer
// and consumer without sending a GraphicBuffer over Binder.
BufferQueueDefs::SlotsType mSlots;

// mQueue is a FIFO of queued buffers used in synchronous mode.
Fifo mQueue;

// mFreeSlots contains all of the slots which are FREE and do not currently
// have a buffer attached.
std::set<int> mFreeSlots;

// mFreeBuffers contains all of the slots which are FREE and currently have
// a buffer attached.
std::list<int> mFreeBuffers;

// mActiveBuffers contains all slots which have a non-FREE buffer attached.
std::set<int> mActiveBuffers;

// mDequeueCondition is a condition variable used for dequeueBuffer in
// synchronous mode.
mutable std::condition_variable mDequeueCondition;
```
【真实源码 platform/frameworks/native@libs/gui/include/gui/BufferQueueCore.h:207-234】

注意第一段注释里的核心设计:**buffer 所有权在 producer/consumer 间转移,但 `GraphicBuffer` 本体不经过 Binder 传**(`mSlots` 两端镜像,只传 slot 索引 + fence)。这就是"不搬像素"的实现根:跨进程传的是"第几号槽位 + 一个 fence",不是几 MB 的 framebuffer。

### 3.2 ⭐核心源码逐行:三缓冲就是这一句判断

整个三缓冲机制的"真身",在 producer 想 dequeue 一个 buffer 时的等待逻辑里。这是本章最该读懂的一段源码 ——

```cpp
status_t BufferQueueProducer::waitForFreeSlotThenRelock(FreeSlotCaller caller,
        std::unique_lock<std::mutex>& lock, int* found) const {
    auto callerString = (caller == FreeSlotCaller::Dequeue) ?
            "dequeueBuffer" : "attachBuffer";
    bool tryAgain = true;
    while (tryAgain) {
        if (mCore->mIsAbandoned) {
            BQ_LOGE("%s: BufferQueue has been abandoned", callerString);
            return NO_INIT;
        }

        int dequeuedCount = 0;
        int acquiredCount = 0;
        for (int s : mCore->mActiveBuffers) {
            if (mSlots[s].mBufferState.isDequeued()) {
                ++dequeuedCount;        // 统计当前 producer 手里握着几个 buffer
            }
            if (mSlots[s].mBufferState.isAcquired()) {
                ++acquiredCount;        // 统计 consumer 手里握着几个
            }
        }

        // Producers are not allowed to dequeue more than
        // mMaxDequeuedBufferCount buffers.
        // This check is only done if a buffer has already been queued
        if (mCore->mBufferHasBeenQueued &&
                dequeuedCount >= mCore->mMaxDequeuedBufferCount) {   // ★ 三缓冲的真身在这里
            // Supress error logs when timeout is non-negative.
            if (mDequeueTimeout < 0) {
                BQ_LOGE("%s: attempting to exceed the max dequeued buffer "
                        "count (%d)", callerString,
                        mCore->mMaxDequeuedBufferCount);
            }
            return INVALID_OPERATION;
        }

        *found = BufferQueueCore::INVALID_BUFFER_SLOT;

        // If we disconnect and reconnect quickly, we can be in a state where
        // our slots are empty but we have many buffers in the queue. ...
        const int maxBufferCount = mCore->getMaxBufferCountLocked();
        bool tooManyBuffers = mCore->mQueue.size()
                            > static_cast<size_t>(maxBufferCount);
        if (tooManyBuffers) {
            BQ_LOGV("%s: queue size is %zu, waiting", callerString,
                    mCore->mQueue.size());
        } else {
            // ... 优先复用已挂 buffer 的 FREE slot(getFreeBufferLocked),
            //     否则取一个空 slot 再分配(getFreeSlotLocked) ...
            if (caller == FreeSlotCaller::Dequeue) {
                int slot = getFreeBufferLocked();
                if (slot != BufferQueueCore::INVALID_BUFFER_SLOT) {
                    *found = slot;
                } else if (mCore->mAllowAllocation) {
                    *found = getFreeSlotLocked();
                }
            } else { /* attach 分支:优先空 slot */ }
        }

        // If no buffer is found, or if the queue has too many buffers
        // outstanding, wait for a buffer to be acquired or released ...
        tryAgain = (*found == BufferQueueCore::INVALID_BUFFER_SLOT) ||
                   tooManyBuffers;
        if (tryAgain) {
            // 非阻塞模式(producer/consumer 都在 app 手里)直接返回 WOULD_BLOCK
            if ((mCore->mDequeueBufferCannotBlock || mCore->mAsyncMode) &&
                    (acquiredCount <= mCore->mMaxAcquiredBufferCount)) {
                return WOULD_BLOCK;
            }
            // 否则:阻塞在条件变量上,等 consumer release / acquire 后唤醒
            if (mDequeueTimeout >= 0) {
                std::cv_status result = mCore->mDequeueCondition.wait_for(lock,
                        std::chrono::nanoseconds(mDequeueTimeout));
                if (result == std::cv_status::timeout) {
                    return TIMED_OUT;
                }
            } else {
                mCore->mDequeueCondition.wait(lock);   // ★ producer 在这里挂起,直到有 buffer 周转出来
            }
        }
    } // while (tryAgain)

    return NO_ERROR;
}
```
【真实源码 platform/frameworks/native@libs/gui/BufferQueueProducer.cpp:297-408】(中间几段注释/分支为可读性做了 `// ...` 省略,关键判断逐字保留)

逐行抓住三件事:

1. **`dequeuedCount >= mCore->mMaxDequeuedBufferCount`(第 322 行)就是双/三缓冲的开关**。`mMaxDequeuedBufferCount` 默认是 1,加上 SurfaceFlinger 那一端正在显示的,有效就是双缓冲;app 申请 triple buffering 时这个数会被抬到 2(producer 可同时握 2 个),总 buffer 数到 3。**producer 手里的 buffer 达到上限,就不许再 dequeue。**

2. **达到上限后 `mDequeueCondition.wait(lock)`(第 401 行)是 producer 阻塞点**。producer 卡在这里,直到 consumer `releaseBuffer` 把一个 buffer 还成 FREE、或 `acquireBuffer` 腾出队列空间,**唤醒** 它。这就是"为什么双缓冲下一次慢帧会停摆":只有 2 个 buffer,producer 画完一个、consumer 还在显示另一个,producer 没第三个可画,只能干等下一个 VSYNC,**白白浪费一整个 16ms**。三缓冲多一个 buffer,producer 就能继续画下一帧,不必等。

3. **非阻塞 vs 阻塞两条路(第 384 / 401 行)**。当 producer 和 consumer 都在同一个 app 进程(比如用 `ImageReader` 自己当 consumer),`mDequeueBufferCannotBlock` 为真,dequeue 不许阻塞、直接返回 `WOULD_BLOCK`,避免自己等自己死锁。普通屏幕渲染(consumer 是 SurfaceFlinger,跨进程)走阻塞路。

> 一句话:**triple buffering 不是"画三帧",是"允许 producer 同时握 2 个 buffer 在手",代价是多一个 buffer 的内存和最多多一帧的延迟,换来流水线在偶尔慢帧时不停摆。**

### 3.3 双缓冲 vs 三缓冲:对比与边界

| 维度 | 双缓冲 | 三缓冲 |
|---|---|---|
| buffer 总数 | 2(1 producer 画 + 1 consumer 显示) | 3(2 producer 可握 + 1 consumer 显示) |
| 一次慢帧后果 | producer 无 buffer 可画,**停摆一整个 VSYNC** | producer 可继续画下一帧,**不停摆** |
| 端到端延迟 | 低(最多滞后 1 帧) | 略高(可能滞后 2 帧) |
| 内存 | 省一个 framebuffer | 多一个 framebuffer(全屏 RGBA 约几 MB) |
| 适用 | 渲染稳定不会慢、对延迟极敏感(如某些游戏/VR) | 通用 UI、渲染偶有抖动(Android 默认倾向) |

**边界与陷阱**:

- **三缓冲不是越多越好**。buffer 越多,一帧从生产到上屏的滞后越大,**触摸延迟(touch-to-display latency)越高**。这就是为什么不是"多多益善"开十缓冲 —— 顺滑和跟手是一对 trade-off。
- **三缓冲治不了"持续慢"**。它只缓冲 **偶尔** 的慢帧。如果 app 每帧都画不完(比如稳定 20ms/帧 @60Hz),三缓冲很快也被填满,producer 照样阻塞,只是把"立刻卡"推迟成"卡得有规律"。根因还得回去优化渲染。
- **buffer 数还受 consumer 的 `mMaxAcquiredBufferCount` 约束**。SurfaceFlinger 至少要能同时持有正在显示 + 正在合成的 buffer,这部分加上 producer 的份额,才是总 buffer 数。

---

## 四、VSYNC 模型:用线性回归预测下一帧

现在进入"拍子"这一半。核心问题:**SurfaceFlinger 怎么知道下一个 VSYNC 在什么时刻?**

答案不是"读硬件中断的时间",而是 **维护一个统计模型,用历史 VSYNC 时间戳预测未来**。

### 4.1 三组件分工:Tracker / Dispatch / Controller

现代 AOSP 把老 `DispSync` 拆成三个正交组件,由 `VsyncSchedule` 组合持有:

```
                      ┌──────────────────────────────────────┐
                      │            VsyncSchedule              │  "synchronizes to hardware
                      │  (一个物理显示一个)                    │   VSYNC of a physical display"
                      └──────────────────────────────────────┘
                         │              │              │
            ┌────────────┘    ┌─────────┘    └──────────────┐
            ▼                 ▼                             ▼
   ┌─────────────────┐ ┌──────────────────┐      ┌────────────────────┐
   │  VSyncTracker    │ │  VSyncDispatch    │      │  VSyncController    │
   │  =VSyncPredictor │ │ =VSyncDispatch-   │      │  =VSyncReactor      │
   │                  │ │   TimerQueue      │      │                    │
   │ 管"模型/预测"     │ │ 管"提前多久叫谁"   │      │ 管"硬件VSYNC开关/   │
   │ addVsyncTimestamp│ │ schedule/cancel   │      │  喂样本/重同步"      │
   │ nextAnticipated  │ │ registerCallback  │      │ addHwVsyncTimestamp │
   │   VSyncTimeFrom  │ │                   │      │                    │
   └─────────────────┘ └──────────────────┘      └────────────────────┘
```

来自 `VsyncSchedule` 的真实成员(consumer 视角验证了这个三分):

> `VsyncSchedule` 持有三个 const 指针:`mTracker`(TrackerPtr)、`mDispatch`(DispatchPtr)、`mController`(ControllerPtr);对外暴露 `addResyncSample(timestamp, hwcVsyncPeriod)`(喂样本、检测周期变化)、`enableHardwareVsync()` / `disableHardwareVsync(disallow)`(硬件 VSYNC 开关)、`getTracker()` / `getDispatch()`(给测试用)。【真实源码 platform/frameworks/native@services/surfaceflinger/Scheduler/VsyncSchedule.h】

这种拆分的价值:**"怎么预测下一帧"(Tracker)和"提前多久把线程叫醒"(Dispatch)是两件正交的事**。老 `DispSync` 把它们糊在一起,导致改预测算法会动到调度、改调度会动到预测。拆开后,`VSyncPredictor`(回归)可以独立替换 PLL,而 `VSyncDispatchTimerQueue` 完全不用改。

### 4.2 ⭐核心源码逐行:最小二乘拟合 VSYNC 周期

`VSyncPredictor` 是 `VSyncTracker` 的实现。它的心脏是 `addVsyncTimestamp` —— 每来一个硬件 VSYNC 时间戳,就用最近 N 个样本 **重新拟合一条直线**。这段是整个 VSYNC 预测的数学核心:

```cpp
bool VSyncPredictor::addVsyncTimestamp(nsecs_t timestamp) {
    SFTRACE_CALL();
    std::lock_guard lock(mMutex);

    if (!validate(timestamp)) {
        // 时间戳不合理(比如比上一个还早、或跳变过大):学习期就清空重来,
        // 稳定期就只更新 mKnownTimestamp,不污染回归样本
        if (mTimestamps.size() < kMinimumSamplesForPrediction) {
            mTimestamps.push_back(timestamp);
            clearTimestamps(/* clearTimelines */ false);
        } else if (!mTimestamps.empty()) {
            mKnownTimestamp =
                    std::max(timestamp, *std::max_element(mTimestamps.begin(), mTimestamps.end()));
        } else {
            mKnownTimestamp = timestamp;
        }
        return false;
    }

    // 把新样本放进固定大小的环形缓冲(满了就覆盖最老的)
    if (mTimestamps.size() != kHistorySize) {
        mTimestamps.push_back(timestamp);
        mLastTimestampIndex = next(mLastTimestampIndex);
    } else {
        mLastTimestampIndex = next(mLastTimestampIndex);
        mTimestamps[mLastTimestampIndex] = timestamp;
    }

    const size_t numSamples = mTimestamps.size();
    if (numSamples < kMinimumSamplesForPrediction) {
        // 样本不够,先用"理想周期"(显示模式标称的刷新周期)兜底,截距设 0
        mRateMap[idealPeriod()] = {idealPeriod(), 0};
        return true;
    }

    // ★★ 这是一段 'simple linear regression':Y=VSYNC 时间戳,X=VSYNC 计数序号
    //    拟合出的 slope 就是 VSYNC 周期,intercept 就是相位
    // 公式(代码注释里逐字给出):
    //         Sigma_i( (X_i - mean(X)) * (Y_i - mean(Y) )
    // slope = -------------------------------------------
    //         Sigma_i ( X_i - mean(X) ) ^ 2
    //
    // intercept = mean(Y) - slope * mean(X)
    std::vector<nsecs_t> vsyncTS(numSamples);
    std::vector<nsecs_t> ordinals(numSamples);

    // 归一化到最老的时间戳,减小 intercept 的数值误差
    const auto oldestTS = *std::min_element(mTimestamps.begin(), mTimestamps.end());
    auto it = mRateMap.find(idealPeriod());
    auto const currentPeriod = it->second.slope;

    // 序号需要高精度,放大 1000 倍做定点运算
    constexpr int64_t kScalingFactor = 1000;

    nsecs_t meanTS = 0;
    nsecs_t meanOrdinal = 0;
    for (size_t i = 0; i < numSamples; i++) {
        const auto timestamp = mTimestamps[i] - oldestTS;     // Y_i(相对最老样本)
        vsyncTS[i] = timestamp;
        meanTS += timestamp;
        // X_i:把时间戳除以当前周期、四舍五入成"第几个 VSYNC"
        const auto ordinal = currentPeriod == 0
                ? 0
                : (vsyncTS[i] + currentPeriod / 2) / currentPeriod * kScalingFactor;
        ordinals[i] = ordinal;
        meanOrdinal += ordinal;
    }
    meanTS /= numSamples;
    meanOrdinal /= numSamples;

    // 去均值
    for (size_t i = 0; i < numSamples; i++) {
        vsyncTS[i] -= meanTS;
        ordinals[i] -= meanOrdinal;
    }

    // 分子 = Σ(ΔX·ΔY),分母 = Σ(ΔX²)
    nsecs_t top = 0;
    nsecs_t bottom = 0;
    for (size_t i = 0; i < numSamples; i++) {
        top += vsyncTS[i] * ordinals[i];
        bottom += ordinals[i] * ordinals[i];
    }
    if (CC_UNLIKELY(bottom == 0)) {     // 退化:所有样本同序号,放弃本次拟合
        it->second = {idealPeriod(), 0};
        clearTimestamps(/* clearTimelines */ true);
        return false;
    }

    nsecs_t const anticipatedPeriod = top * kScalingFactor / bottom;            // slope = 预测周期
    nsecs_t const intercept = meanTS - (anticipatedPeriod * meanOrdinal / kScalingFactor); // 相位

    // ★ 离群保护:拟合出的周期和理想周期偏离超过容忍度,丢弃本次结果、清空重学
    auto const percent = std::abs(anticipatedPeriod - idealPeriod()) * kMaxPercent / idealPeriod();
    if (percent >= kOutlierTolerancePercent) {
        it->second = {idealPeriod(), 0};
        clearTimestamps(/* clearTimelines */ true);
        return false;
    }

    it->second = {anticipatedPeriod, intercept};   // 存进 rateMap:这就是新模型
    return true;
}
```
【真实源码 platform/frameworks/native@services/surfaceflinger/Scheduler/VSyncPredictor.cpp:143-269】(license 头、trace 调用、log 行省略,数学主体逐字)

逐行抓住五件事:

1. **`Model = {slope, intercept}`**(头文件 `struct Model { nsecs_t slope; nsecs_t intercept; }`,【真实源码 …@VSyncPredictor.h:61-64】)。整个"VSYNC 模型"被压缩成 **两个 64 位整数**:周期 + 相位。一条直线,够用。

2. **X 轴是"第几个 VSYNC",不是连续时间**。`ordinal = timestamp / currentPeriod`(四舍五入)。这一步把不规则到达的时间戳"吸附"到整数序号上 —— 即使中间漏采了几个 VSYNC,只要序号对,回归出的斜率还是对的。这是回归比朴素"相邻差分求平均"鲁棒的关键。

3. **归一化到 `oldestTS` 是数值技巧**。VSYNC 时间戳是纳秒级的 `CLOCK_MONOTONIC` 大数(开机以来纳秒,十几位),直接平方求和会溢出/丢精度。减掉最老样本,把数值压到一帧的量级,再算。注释里写得很直白:"Normalizing to the oldest timestamp cuts down on error in calculating the intercept."

4. **`kScalingFactor = 1000` 是定点运算**。没有浮点,序号乘 1000 保留三位小数精度,全程整数算。这是系统级代码对"可预测性/无浮点抖动"的偏好。

5. **`kOutlierTolerancePercent` 离群保护是稳定性的命门**。如果某次拟合出的周期偏离标称周期太多(比如一个伪 VSYNC、或刷新率刚切换还没稳),**直接丢弃、清空样本、退回理想周期重学**。构造参数注释:"a number 0 to 100 that will be used to filter samples that fall outlierTolerancePercent from an anticipated vsync event."【真实源码 …@VSyncPredictor.h:39-42】没有这层,一个坏样本能把整个拍子带歪,引发可见的卡顿。

> 这段代码回答了 TL;DR 里那个问题:**为什么不直接用硬件中断?** 因为硬件中断 = 单个带噪样本;而 `VSyncPredictor` = 对最近 N 个样本做回归,给出一条 **滤过噪声、可外推到未来任意时刻** 的直线。`nextAnticipatedVSyncTimeFrom(t)` 就是把 `t` 代进这条直线、解出"≥t 的下一个 VSYNC 在哪"。

### 4.3 多刷新率怎么办:VsyncTimeline 与相位接续

`VSyncPredictor` 内部还有一个 `std::deque<VsyncTimeline> mTimelines`(【真实源码 …@VSyncPredictor.h:169】)。每条 `VsyncTimeline` 是"某段时间内有效的一条直线"。刷新率切换(60→120Hz)时,旧 timeline 在切换点 `freeze()`、新 timeline 从切换点开始,**相位平滑接续**,避免切换瞬间拍子跳变。`VsyncOnTimeline { Unique, Shared, Outside }` 枚举(【真实源码 …@VSyncPredictor.h:110-114】)用来判断一个预测的 VSYNC 落在哪条 timeline 上。这是 PLL 时代很难优雅做到、而回归 + 多 timeline 能自然处理的场景 —— 也是当年换实现的主要动机之一。

---

## 五、调度:提前多久把谁叫醒

模型能预测"下一个 VSYNC 在 t=16.6ms"了,但 SurfaceFlinger 不能等到 16.6ms 才开始合成 —— 那时已经来不及了。它必须 **在 16.6ms 之前** 把线程叫醒、留出 `workDuration` 干活时间。这就是 `VSyncDispatch` 的活。

### 5.1 ScheduleTiming:工作量预算

注册一个 VSYNC 回调时,要给一份"时间预算":

```cpp
struct ScheduleTiming {
    nsecs_t workDuration = 0;   // 客户端干活需要多久
    nsecs_t readyDuration = 0;  // 干完后下游还需多久(见下)
    nsecs_t lastVsync = 0;      // 目标显示时刻(会吸附到 ≥它的最近预测 VSYNC)
    std::optional<nsecs_t> committedVsyncOpt;  // 已承诺的目标 VSYNC
    ...
};
```
【真实源码 platform/frameworks/native@services/surfaceflinger/Scheduler/VSyncDispatch.h:102-114】

头文件对 `workDuration` / `readyDuration` 的逐字解释非常关键:

> 【真实源码 …@VSyncDispatch.h:85-100】"@workDuration: The time needed for the client to perform its work. @readyDuration: The time needed for the client to be ready before a vsync event. For external (non-SF) clients, not only do we need to account for their workDuration, but we also need to account for the time SF will take to process their buffer/transaction. ... callback will be dispatched at 'workDuration + readyDuration' nanoseconds before a vsync event."

翻译:

- **`workDuration`** = 这个客户端自己干活要多久(app 渲染一帧、或 SF 合成一帧的耗时估计)。
- **`readyDuration`** = 下游还要多久。对 **app**(external client)来说,它画完不算完,SurfaceFlinger 还要花时间合成它的 buffer,所以 app 的回调要 **再提前** 一个 SF 合成的时间量。对 **SF 自己**(internal),下游就是屏幕,通常 `readyDuration=0`。
- **回调时刻 = 目标 VSYNC − workDuration − readyDuration**。

这正是相位偏移(VSYNC-app vs VSYNC-sf)的现代实现:app 比 SF 早醒,早的那部分就是 `readyDuration`(SF 合成耗时)。

### 5.2 ⭐核心源码逐行:wakeup = predictedVsync − work − ready

`VSyncDispatchTimerQueueEntry::schedule` 把上面的减法落地。这是连接"预测的 VSYNC"和"什么时候武装定时器"的keystone:

```cpp
ScheduleResult VSyncDispatchTimerQueueEntry::schedule(VSyncDispatch::ScheduleTiming timing,
                                                      VSyncTracker& tracker, nsecs_t now) {
    SFTRACE_NAME("VSyncDispatchTimerQueueEntry::schedule");
    // ① 问 tracker(=VSyncPredictor):从"现在 + 我要的工作时间"往后,最近的预测 VSYNC 在哪?
    //    max(lastVsync, now+work+ready) 保证不会调度到一个已经来不及的 VSYNC
    auto nextVsyncTime =
            tracker.nextAnticipatedVSyncTimeFrom(std::max(timing.lastVsync,
                                                          now + timing.workDuration +
                                                                  timing.readyDuration),
                                                 timing.committedVsyncOpt.value_or(
                                                         timing.lastVsync));
    // ② keystone:唤醒时刻 = 预测 VSYNC 时刻 − 工作时间 − 下游就绪时间
    auto nextWakeupTime = nextVsyncTime - timing.workDuration - timing.readyDuration;

    // ③ 防抖:如果新算的目标/唤醒时刻只比已武装的稍晚一点(在 mMinVsyncDistance 内),
    //    不要平白跳过一个 VSYNC 目标,沿用已武装的
    bool const wouldSkipAVsyncTarget =
            mArmedInfo && (nextVsyncTime > (mArmedInfo->mActualVsyncTime + mMinVsyncDistance));
    bool const wouldSkipAWakeup =
            mArmedInfo && ((nextWakeupTime > (mArmedInfo->mActualWakeupTime + mMinVsyncDistance)));
    if (FlagManager::getInstance().dont_skip_on_early_ro()) {
        if (wouldSkipAVsyncTarget || wouldSkipAWakeup) {
            nextVsyncTime = mArmedInfo->mActualVsyncTime;
        } else {
            nextVsyncTime = adjustVsyncIfNeeded(tracker, nextVsyncTime);
        }
        nextWakeupTime = std::max(now, nextVsyncTime - timing.workDuration - timing.readyDuration);
    } else {
        if (wouldSkipAVsyncTarget && wouldSkipAWakeup) {
            return getExpectedCallbackTime(nextVsyncTime, timing);
        }
        nextVsyncTime = adjustVsyncIfNeeded(tracker, nextVsyncTime);
        nextWakeupTime = nextVsyncTime - timing.workDuration - timing.readyDuration;
    }

    auto const nextReadyTime = nextVsyncTime - timing.readyDuration;  // 截止线(deadline)
    mScheduleTiming = timing;
    // ④ 记下"武装信息":唤醒时刻 / 目标VSYNC / 就绪截止时刻
    mArmedInfo = {nextWakeupTime, nextVsyncTime, nextReadyTime};
    return ScheduleResult{TimePoint::fromNs(nextWakeupTime), TimePoint::fromNs(nextVsyncTime)};
}
```
【真实源码 platform/frameworks/native@services/surfaceflinger/Scheduler/VSyncDispatchTimerQueue.cpp:99-135】

三层对照,看清这段在做什么:

```
   现在(now)                    nextWakeupTime              nextReadyTime      nextVsyncTime
      │                              │                          │                 │
      ▼                              ▼                          ▼                 ▼
   ───●──────────────────────────────●──────────────────────────●─────────────────●────▶ 时间
                                      │◀──── workDuration ──────▶│◀── readyDuration ─▶│
                                      │                                              │
                                  定时器在这里响,                              预测的 VSYNC
                                  回调被调用,开始干活                          (合成结果要在此刻就绪)
```

- **`nextVsyncTime`** 来自 `tracker.nextAnticipatedVSyncTimeFrom(...)` —— 就是第四节那条回归直线外推的结果。
- **`nextWakeupTime = nextVsyncTime - workDuration - readyDuration`**(第 108 行)—— 定时器实际武装的时刻。`VSyncDispatchTimerQueue` 维护一个按 `nextWakeupTime` 排序的回调队列,用一个 `TimeKeeper`(底层 `timerfd`)武装到"最早的那个唤醒时刻",到点 fire。
- **`getExpectedCallbackTime` 也是同一个减法**(第 41-45 行:`nextVsyncTime - readyDuration - workDuration`),保证返回给调用者的预期唤醒时刻和内部武装的一致。

> 这段把"VSYNC 调度"彻底祛魅:**它不是"等 VSYNC 来了再做事",而是"预测 VSYNC 会来,倒推出该提前多久醒,武装一个普通 timerfd"。** 整个 SurfaceFlinger 的合成节拍,本质是一串被 `VSyncPredictor` 的直线驱动、由 `timerfd` 触发的定时器。

### 5.3 调度的两个边界条件

- **不能调度到过去**:`max(lastVsync, now + work + ready)`(第 103 行)保证目标 VSYNC 至少在"现在 + 干活时间"之后。如果 app 申报的 `workDuration` 超过一个 VSYNC 周期(渲染太慢),目标会被推到下一个甚至下下个 VSYNC —— 这在 trace 上表现为该帧 **直接 miss、滞后一帧**。
- **`mMinVsyncDistance` 防抖**(第 111/113 行):重复 schedule 时,如果新目标只比已武装的晚一丁点,不重新武装、沿用旧的,避免在 VSYNC 边界附近反复横跳、把一个本来赶得上的帧调度丢了。这类边界 bug 在 AOSP 有专门的 flag(`dont_skip_on_early_ro`)做行为切换,可见踩过坑。

---

## 六、⭐可运行 Demo:亲手看 VSYNC 与一帧合成

> 全部 demo **需 Android 真机或模拟器验证**。模拟器(AVD)能跑 `dumpsys` / `gfxinfo` / 大部分 `atrace`;**Perfetto 的 `frametimeline` 和精确 HW_VSYNC 在真机上更可信**(模拟器的"显示"是软件合成,VSYNC 是宿主机模拟的,数值仅供理解机制、不代表真实硬件)。前置:`adb` 已连上设备、`adb root` 可选(部分 service call 需要)。

### Demo A:读 SurfaceFlinger 的 VSYNC 相位配置(最简,30 秒)

**目的**:把第五节的 `workDuration` / phase offset 从抽象变成具体数字。

```bash
# 看 VSYNC 各相位偏移(对应 app phase / SF phase / early phase / GL phase)
adb shell dumpsys SurfaceFlinger | grep -i phase
```

**预期输出**(数值随设备/刷新率不同,形如)【真实样例 二手,CSDN/腾讯云多篇 dumpsys 实录一致,标「二手」】:

```
app phase:      2333334 ns    SF phase:      6166667 ns
early app phase: 833334 ns    early SF phase: 666667 ns
GL early app phase: 15500001 ns  GL early SF phase: 3166667 ns
```

**怎么读**:

- `app phase` / `SF phase` = 正常情况下 app/SF 相对 HW_VSYNC 的提前量。注意 **app phase < SF phase**(app 比 SF 早醒)—— 对应第 5.1 节 app 要多算一个 `readyDuration`(SF 合成时间)。
- `early *` = 刷新率切换瞬间用的相位(过渡期)。
- `GL early *` = SF 走 GPU 合成(而非 HWC overlay)时用的相位 —— GPU 合成更慢,所以 app 要 **更早** 醒(15.5ms!几乎提前一整帧)。

> 把这三组数字和第五节的"wakeup = vsync − work − ready"对上:这些 phase 值就是不同场景下的"提前量"。

### Demo B:实时看一帧的生产/消费节拍(gfxinfo,1 分钟)

**目的**:不用抓 trace,直接看 app 渲染管线每个阶段的耗时,对照 16.6ms 预算。

```bash
# 1. 重置某个 app 的帧统计(以系统设置为例,换成你要观测的包名)
adb shell dumpsys gfxinfo com.android.settings reset

# 2. 在设备上操作该 app(滑动列表制造渲染)

# 3. dump 帧统计
adb shell dumpsys gfxinfo com.android.settings
```

**预期输出**(节选,关键看这几行):

```
Total frames rendered: 257
Janky frames: 12 (4.67%)
50th percentile: 5ms
90th percentile: 11ms
95th percentile: 14ms
99th percentile: 22ms
Number Missed Vsync: 8
Number High input latency: 0
Number Slow UI thread: 5
Number Slow bitmap uploads: 1
Number Slow issue draw commands: 6
...
---PROFILEDATA---
Flags,IntendedVsync,Vsync,...,FrameCompleted,...
0,3502340000000,3502356600000,...   ← 每行一帧,列是各阶段时间戳
```

**怎么读**:

- **`Janky frames` + `Number Missed Vsync`** = 掉帧统计。`Missed Vsync` 直接对应第五节"`workDuration` 超过一个周期 → 目标被推到下一个 VSYNC"。
- **`90th percentile` 接近或超过帧预算**(60Hz=16.6ms,90Hz≈11ms,120Hz≈8.3ms)就是渲染太慢的信号。
- **`---PROFILEDATA---`** 段每行是一帧,`IntendedVsync`(本该哪个 VSYNC)vs `Vsync`(实际对齐的 VSYNC)—— 两者差一个周期就是这帧 miss 了。这就是 `VSyncPredictor` 预测值在 app 侧的落地。

### Demo C:⭐用 Perfetto 抓一帧,亲眼看 VSYNC 拍子与合成(核心 demo)

**目的**:把第二节那张"三段流水线 + 三个 VSYNC"的图,在真实 trace 上一一对上。这是本章最该亲手做的。

**步骤 1 — 抓 trace**(设备端 Perfetto,Android 9+ 自带):

```bash
# 方式一:用 record_android_trace 脚本(推荐,自动拉回 trace 文件)
#   先下载脚本:
curl -O https://raw.githubusercontent.com/google/perfetto/main/tools/record_android_trace
chmod +x record_android_trace

# 抓 5 秒,带上图形栈相关的 atrace 类别 + frametimeline 数据源
./record_android_trace -o trace_file.perfetto-trace -t 5s \
    sched freq idle am wm gfx view sync \
    surfaceflinger hal binder_driver

# 方式二:纯 adb(等价,手动触发)
adb shell perfetto -o /data/misc/perfetto-traces/trace.pb -t 5s \
    sched gfx view sf sync surfaceflinger
adb pull /data/misc/perfetto-traces/trace.pb ./trace.pb
```

> `frametimeline` 数据源(Perfetto config 里 `data_sources { config { name: "android.surfaceflinger.frametimeline" } }`)用于拿"期望 vs 实际"帧时间线与 jank 分类。【真实文档 perfetto.dev/docs/data-sources/frametimeline】

**步骤 2 — 在 ui.perfetto.dev 打开 trace,定位这几条 track**(track 名来自真实 Perfetto UI):

在 `surfaceflinger` 进程下:

- **`VSYNC-app`** —— app VSYNC 信号,值在 0/1 间跳,每次跳变 = 一个 app VSYNC。【真实 track 名,androidperformance.com Perfetto 系列】
- **`VSYNC-sf`** —— SurfaceFlinger VSYNC 信号,无 offset 时和 `VSYNC-app` 同步跳。
- **`VSYNC-appSf`** —— Android 13+ 新增,服务需要和 SF 同步的特殊 Choreographer 客户端。
- **`HW_VSYNC`** —— 硬件 VSYNC 是否开启(1=开 0=关)。**会看到它时开时关** —— 这就是第 4.1 节"稳定后关掉硬件中断省电、需要时再打开重同步"的实证。

在 app 进程下:

- **`Choreographer#doFrame`**(UI Thread)—— app 在 VSYNC 拍子上的一帧工作。
- **`DrawFrame` / `syncAndDrawFrame` / `queueBuffer`**(RenderThread)—— GPU 渲染 + 交 buffer。
- **`FrameDisplayEventReceiver.onVsync`** —— app 收到 VSYNC 信号的时刻。

**步骤 3 — 验证机制**(你应该能在 trace 上亲眼确认):

1. **拍子**:量相邻 `VSYNC-app` 跳变的间隔 ≈ 16.6ms(60Hz)/ 11.1ms(90Hz)/ 8.3ms(120Hz)。这就是 `VSyncPredictor` 拟合出的 `slope`(周期)。
2. **提前量**:`Choreographer#doFrame` 的 **起点** 早于对应的 `VSYNC-sf` 跳变 —— 早的那段就是 `workDuration + readyDuration`(第 5.2 节的减法)。
3. **流水线重叠**:app 在画第 N 帧时,SF 在合成第 N−1 帧 —— 三段不是串行,是错位重叠的。
4. **掉帧**:找一个 jank 帧,`actual frame timeline` 比 `expected frame timeline` 长、越过了下一个 VSYNC —— 对应 `Missed Vsync`。

> 这一步把"源码里的 slope/intercept/workDuration"和"屏幕上看得见的卡顿"接上了。**强烈建议真机跑一遍** —— 看懂这张 trace,VSYNC 机制就内化了。

### Demo D:看 SurfaceFlinger 的 Layer 与合成方式(HWC vs GPU)

**目的**:验证"能交给 HWC 的图层走硬件 overlay、剩下的才用 GPU 合成"。

```bash
# 看当前所有 Layer 及其合成方式
adb shell dumpsys SurfaceFlinger | grep -A 2 -i "Display 0\|Composition\|HWC\|Client\|Device"

# 更直接:看 HWC 把哪些层标成 DEVICE(硬件合成)vs CLIENT(GPU 合成)
adb shell dumpsys SurfaceFlinger | grep -iE "Composition type|DEVICE|CLIENT|SOLID_COLOR"
```

**预期**:每个 Layer 有一个 composition type。`DEVICE` = HWC 硬件 overlay 直接扫描(省 GPU、省电);`CLIENT` = 落到 GPU 用 RenderEngine 合成。**当 Layer 数超过 HWC overlay 通道数(典型 4~8)时,多出来的层只能 `CLIENT` 合成** —— 这是"开太多悬浮窗/图层导致掉帧"的底层原因。

> 对照:把屏幕上的窗口数量减少(关掉通知阴影、悬浮球),再 dump,会看到更多 Layer 变回 `DEVICE`。

### Demo E(可选,进阶):触发一次 SurfaceFlinger 的 Binder 调用

**目的**:确认 SurfaceFlinger 也是个 Binder 服务(和第 2 章呼应)。

```bash
# 列出 SurfaceFlinger 这个 binder 服务
adb shell service list | grep -i SurfaceFlinger

# 发一个真实 transaction(code 1013 在多数版本是 "capture/获取帧号" 类查询,
#  具体 code 随版本变,见 ISurfaceComposer.h;此处仅演示"它确实是 binder 服务")
adb shell service call SurfaceFlinger 1013
```

**预期**:返回一个 `Result: Parcel(...)`,证明 SurfaceFlinger 通过 Binder 暴露接口。**注意 transaction code 随 Android 版本变化**,不要把具体数字当 API。

---

## 七、扎根:失败模式 / 生产真坑 / 根因

### 7.1 失败模式

| 现象 | 直接原因 | 根因 / 源码位置 |
|---|---|---|
| **撕裂(tearing)** | 屏幕扫描中途换了 buffer | 没开 VSYNC 同步,或在非 async 模式强行丢帧。BufferQueue 同步模式 + VSYNC 对齐才能消除 |
| **持续掉帧(steady jank)** | app 每帧 `workDuration` > 帧周期 | 渲染本身慢。三缓冲只缓冲偶发慢帧,治不了持续慢(第 3.3 节) |
| **偶发卡顿 + 双缓冲** | 一次慢帧导致 producer 无 buffer 停摆 | `dequeuedCount >= mMaxDequeuedBufferCount` 阻塞(第 3.2 节)。开三缓冲缓解 |
| **触摸延迟高** | buffer 链路太长,帧从画到上屏滞后多帧 | 三缓冲/多缓冲的代价;phase offset 配得过保守 |
| **刷新率切换瞬间抖一下** | 切换点相位没接续好 | `VsyncTimeline` freeze/接续逻辑;或离群保护把切换样本全丢了重学(第 4.2/4.3 节) |
| **HW_VSYNC 一直开着费电** | 模型一直判定漂移、反复重同步 | 时间戳源抖动大,`kOutlierTolerancePercent` 内反复触发 resync |

### 7.2 三个生产真坑

**坑 1:相位偏移(phase offset)配错,要么延迟高要么掉帧。**

`ro.surface_flinger.vsync_event_phase_offset_ns` / `vsync_sf_event_phase_offset_ns`(【真实属性 source.android.com/docs/core/graphics/surfaceflinger-props】)是 OEM 在 BoardConfig 里调的。

- 偏移 **太小**(app/SF 醒得太晚):干活时间不够,频繁 miss VSYNC,掉帧。
- 偏移 **太大**(醒得太早):每帧白白多等,**触摸到上屏的端到端延迟增加**,体感"跟手但发飘"。

根因:`workDuration` 是 **估计值**,phase offset 是它的静态近似。现代 SurfaceFlinger 用动态 `workDuration`(基于历史合成耗时)替代静态 offset,正是为了自适应这个 trade-off。坑在于老设备/老配置还在用静态 offset,换屏(刷新率变)后没重调。

**坑 2:app 自己当 consumer(ImageReader/SurfaceTexture)忘了及时 release,producer 饿死。**

当 app 用 `ImageReader` 把某个 Surface 的内容读出来时,app 既是间接 producer 又是 consumer。如果 `ImageReader` 拿到 `Image` 后忘了 `close()`(= 不 release buffer),buffer 一直 ACQUIRED,producer 很快 `dequeuedCount` 触顶,`dequeueBuffer` 返回 `WOULD_BLOCK`(第 3.2 节非阻塞路)或干脆挂死。

根因:第 3.2 节那段 —— consumer 不 release,producer 等不到 `mDequeueCondition` 唤醒。**`maxImages` 设小一点 + 每个 `Image` 用完立刻 `close()`** 是纪律。

**坑 3:把 SurfaceFlinger 当成"画图的",其实它主要是"合成的 + 调度的"。**

常见误解:以为掉帧是 SurfaceFlinger 慢。绝大多数掉帧根因在 **app 自己的 `Choreographer#doFrame` 超了预算**(measure/layout/draw 太重),不是 SF。SF 只在 **图层过多被迫 GPU 合成**(Demo D 的 `CLIENT` 路)时才成为瓶颈。

根因:误判方向会浪费时间优化错地方。**先用 Demo C 看 trace 里到底是 `doFrame` 长还是 SF 合成长**,再决定优化 app 渲染还是减图层。

### 7.3 根因思维:这套设计在防什么

回到第一节的病。整套 VSYNC + BufferQueue 设计,本质在用三个手段对抗"三段流水线不同步":

1. **共同时钟(VSYNC 模型)** 防"各跑各的" → 撕裂;
2. **有界缓冲 + 余量(三缓冲)** 防"一处慢全线停" → 偶发掉帧;
3. **提前唤醒(workDuration 倒推)** 防"在 deadline 才开始干活" → 来不及。

理解了这三层,任何图形卡顿问题都能定位到是 **拍子乱了(模型/相位)、缓冲不够(buffer 数)、还是干活太慢(workDuration 超预算)** 这三类之一。

---

## 章末·五件套

### 1. 一句话本质

Android 图形栈 = **用 VSYNC 把"app 渲染 / SF 合成 / 屏幕刷新"三段流水线对齐到同一拍子,用 BufferQueue(三缓冲)给流水线留余量,用"预测 VSYNC + 提前唤醒"让每段在 deadline 之前就开始干活**。VSYNC 不是硬件中断本身,而是 `VSyncPredictor` 对历史 HW_VSYNC 时间戳做 **最小二乘回归** 拟合出的一条直线(slope=周期、intercept=相位),再由 `VSyncDispatch` 按 `wakeup = predictedVsync − workDuration − readyDuration` 倒推、用 `timerfd` 触发。

### 2. 关键源码地图(出处)

| 关注点 | 文件@符号 | 出处标注 |
|---|---|---|
| 三缓冲 / producer 阻塞 | `BufferQueueProducer.cpp` :: `waitForFreeSlotThenRelock`(`dequeuedCount >= mMaxDequeuedBufferCount`、`mDequeueCondition.wait`) | 【真实源码 platform/frameworks/native@libs/gui/】(main) |
| buffer 四态集合 | `BufferQueueCore.h` :: `mFreeSlots`/`mFreeBuffers`/`mActiveBuffers`/`mQueue`/`mDequeueCondition` | 同上 |
| VSYNC 预测(回归) | `VSyncPredictor.cpp` :: `addVsyncTimestamp`(simple linear regression,slope/intercept)、`nextAnticipatedVSyncTimeFrom` | 【真实源码 …@services/surfaceflinger/Scheduler/】(main) |
| 提前唤醒调度 | `VSyncDispatchTimerQueue.cpp` :: `VSyncDispatchTimerQueueEntry::schedule`(`nextWakeupTime = nextVsyncTime − work − ready`) | 同上 |
| 调度接口/预算字段 | `VSyncDispatch.h` :: `ScheduleTiming{workDuration,readyDuration,lastVsync}`、`registerCallback`/`schedule` | 同上 |
| 三组件组合 | `VsyncSchedule.h` :: `mTracker`/`mDispatch`/`mController`、`addResyncSample`/`enableHardwareVsync` | 同上 |
| 设计考古(PLL) | legacy `DispSync.{h,cpp}` :: 类注释("software phase-lock loop")、`addResyncSample`/`addPresentFence`、`kErrorThreshold` | 【真实源码 …@(android-7.1.1_r28)】+【真实文档 source.android.com】 |

### 3. 三个数字记死

- **三缓冲 = `mMaxDequeuedBufferCount` 抬到 2**(producer 可同时握 2 个 buffer,总 3 个);双缓冲是握 1 个、总 2 个。"画三帧"是误解,是"允许握 2 个"。
- **VSYNC 模型 = 2 个整数**:`Model{slope, intercept}`,slope=周期、intercept=相位。一条直线。
- **唤醒时刻 = 预测 VSYNC − workDuration − readyDuration**。整个 SF 合成节拍是被这个减法 + `timerfd` 驱动的。

### 4. 动手验证清单(都能在 emulator 或真机跑)

- [ ] Demo A:`dumpsys SurfaceFlinger | grep phase` 看 app/SF/early/GL 各相位偏移,确认 app phase < SF phase
- [ ] Demo B:`dumpsys gfxinfo <pkg>` 看 Janky frames / Missed Vsync / 90th percentile 对照帧预算
- [ ] **Demo C(核心):Perfetto 抓一帧,在 UI 里找 `VSYNC-app`/`VSYNC-sf`/`HW_VSYNC` track,量拍子间隔、看 `doFrame` 提前量、观察 `HW_VSYNC` 时开时关**
- [ ] Demo D:`dumpsys SurfaceFlinger` 看 Layer 的 `DEVICE`(HWC)vs `CLIENT`(GPU)合成,减窗口数看更多层变 DEVICE
- [ ] Demo E:`service call SurfaceFlinger <code>` 确认 SF 是 binder 服务(呼应第 2 章)

### 5. 最容易讲错/记错的点(自检)

- **VSYNC ≠ 硬件中断喂给 app**。硬件只有一个 `HW_VSYNC`;`VSYNC-app`/`VSYNC-sf` 是软件模型从它派生的两个带相位的虚拟拍子。喂模型的是中断时间戳,**驱动 app 的是模型预测值**。
- **三缓冲不是"快",可能更"慢"**(端到端延迟更高)。它换的是"偶发慢帧不停摆",代价是延迟 + 内存。不是 buffer 越多越好。
- **现代 AOSP 早不是 `DispSync`(PLL)了**。Android 10+ 是 `VSyncPredictor`(最小二乘回归)+ `VSyncDispatch` + `VSyncReactor` 三件套。网上停在 `DispSync.addResyncSample` PLL 描述的博客是老版本。
- **掉帧的锅多半在 app `doFrame`,不在 SurfaceFlinger**。SF 只在图层过多被迫 GPU(`CLIENT`)合成时才是瓶颈。先看 trace 是 `doFrame` 长还是 SF 合成长再下结论。
- **`workDuration`/`readyDuration` 不对称**:app(external)的回调要额外提前一个 `readyDuration`(SF 合成它 buffer 的时间);SF 自己(internal)`readyDuration` 通常为 0。这就是 app 比 SF 早醒的代码根。
- **`HW_VSYNC` 会被关掉**。模型稳定后 SurfaceFlinger 关硬件 VSYNC 中断省电、纯靠预测跑,需要重同步(漂移/唤醒新客户端)时再打开。trace 上看到它 0/1 跳是正常的,不是 bug。

---

## 取材记录(本章实际 WebFetch / WebSearch 的 URL)

> 标注原则:逐字引用标【真实源码 repo@path】;机制性二手交叉核对标【二手】;未取到一手标「待核」。
> 说明:AOSP 历史上的 `aosp-mirror/...` GitHub 镜像在本次取材时已 404(仓库疑似下线),故全部改从 **android.googlesource.com**(AOSP 一手 Git)的 `?format=TEXT`(base64)端点取原始字节,本地 base64 解码后逐行核对。下列源码均为此方式实际取到并解码。

**逐字源码(googlesource `main` 分支,现代 AOSP,WebFetch/curl 成功取到并解码核对)**
- `platform/frameworks/native@libs/gui/BufferQueueProducer.cpp`(`waitForFreeSlotThenRelock` 全文逐行)
  `https://android.googlesource.com/platform/frameworks/native/+/refs/heads/main/libs/gui/BufferQueueProducer.cpp?format=TEXT`
- `platform/frameworks/native@libs/gui/include/gui/BufferQueueCore.h`(buffer 四态集合 + 注释)
  `https://android.googlesource.com/platform/frameworks/native/+/refs/heads/main/libs/gui/include/gui/BufferQueueCore.h?format=TEXT`
- `platform/frameworks/native@services/surfaceflinger/Scheduler/VSyncPredictor.cpp`(`addVsyncTimestamp` 最小二乘回归全文)
  `https://android.googlesource.com/platform/frameworks/native/+/refs/heads/main/services/surfaceflinger/Scheduler/VSyncPredictor.cpp?format=TEXT`
- `platform/frameworks/native@services/surfaceflinger/Scheduler/VSyncPredictor.h`(`struct Model{slope,intercept}`、构造参数注释、`VsyncTimeline`)
  `https://android.googlesource.com/platform/frameworks/native/+/refs/heads/main/services/surfaceflinger/Scheduler/VSyncPredictor.h?format=TEXT`
- `platform/frameworks/native@services/surfaceflinger/Scheduler/VSyncDispatchTimerQueue.cpp`(`VSyncDispatchTimerQueueEntry::schedule`、`getExpectedCallbackTime`)
  `https://android.googlesource.com/platform/frameworks/native/+/refs/heads/main/services/surfaceflinger/Scheduler/VSyncDispatchTimerQueue.cpp?format=TEXT`
- `platform/frameworks/native@services/surfaceflinger/Scheduler/VSyncDispatch.h`(`ScheduleTiming` 结构 + workDuration/readyDuration 注释 + 接口)
  `https://android.googlesource.com/platform/frameworks/native/+/refs/heads/main/services/surfaceflinger/Scheduler/VSyncDispatch.h?format=TEXT`
- (亦取到并交叉参考)`VSyncReactor.cpp`、`VSyncDispatchTimerQueue.h`、`VsyncSchedule.h`、`VSyncTracker.h`(同目录 `?format=TEXT`)

**设计考古 / legacy 源码(WebFetch 成功)**
- legacy `DispSync.{h,cpp}`(类注释 "software phase-lock loop"、`addResyncSample`/`addPresentFence`、`kErrorThreshold = 160000000000`)
  `https://android.googlesource.com/platform/frameworks/native/+/android-7.1.1_r28/services/surfaceflinger/DispSync.cpp`
  `https://android.googlesource.com/platform/frameworks/native/+/android-7.1.1_r28/services/surfaceflinger/DispSync.h`

**官方文档(WebFetch 成功)**
- VSync 实现(DispSync=software PLL、HWC2 vsync 回调签名、phase offset 属性):`https://source.android.com/docs/core/graphics/implement-vsync`
- SurfaceFlinger 系统属性(`vsync_event_phase_offset_ns` 等):`https://source.android.com/docs/core/graphics/surfaceflinger-props`
- Perfetto FrameTimeline 数据源(`android.surfaceflinger.frametimeline`、expected/actual timeline):`https://perfetto.dev/docs/data-sources/frametimeline`

**二手交叉核对(WebSearch / 博客,机制性引用,标【二手】;一手时间点标「待核」)**
- Project Butter 三组件(VSync/Triple Buffering/Choreographer,Google I/O 2012):AndroidPolice 2012-07-12 深度解析、TechCrunch 2012-06-27 发布报道、多份资料一致 —— 一手 keynote 视频未直接取,「待核」
- AOSP VSYNC 模型组件梳理(DispSync/DispSyncSource/EventThread/VSyncReactor):`https://utzcoz.github.io/2020/05/02/Analyze-AOSP-vsync-model.html`【二手】
- Perfetto 看 VSYNC track 名(`VSYNC-app`/`VSYNC-sf`/`VSYNC-appSf`/`HW_VSYNC`、`Choreographer#doFrame`):`https://androidperformance.com/en/2025/08/05/Android-Perfetto-08-Vsync/`【二手】
- `dumpsys SurfaceFlinger | grep phase` 输出样例(app/SF/early/GL phase 数值):CSDN/腾讯云多篇 dumpsys 实录一致【二手】

> **诚实声明**:本章 ⭐demo 的命令与 dumpsys 字段均依据官方文档 + 多篇一致的二手实录撰写,**作者未在本次会话中实际运行**(无设备/模拟器接入)。所有 demo 标注"需 Android 真机/模拟器验证";不同 Android 版本/OEM 的 dumpsys 字段名、`service call` 的 transaction code、Perfetto track 名可能有差异,以你设备实测为准。Perfetto track 名、atrace 类别、frametimeline 数据源名取自官方文档与一致的二手资料,可信度高;具体数值样例标【二手】。
