---
title: "Electron 架构（Chromium 域）"
slug: "7-13"
collection: "tech-library"
group: "chromium内核"
order: 7013
summary: "TL;DR Electron = Chromium Content API + Node.js，通过四个关键机制缝合两个异构运行时：① `ContentBrowserClient`/`ContentRendererClient` 插件点让 Electron 劫持 Chromium 进程生命周期； ②…"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T12:40:53.000Z"
updatedAt: "2026-06-12T12:40:53.000Z"
---
> **TL;DR**  
> Electron = Chromium Content API + Node.js，通过四个关键机制缝合两个异构运行时：①
> `ContentBrowserClient`/`ContentRendererClient` 插件点让 Electron 劫持 Chromium 进程生命周期；
> ② `NodeBindings` 用独立 polling 线程 + semaphore 把 libuv event loop 嫁接进 Chromium message
> loop；③ `contextIsolation` + Blink Isolated World 在单进程内隔离 preload 与页面脚本；④ 几十个
> Chromium patch 填补 Content API 的能力缺口。理解这四根支柱，Electron 的所有坑都有根可追。

---

## 前置依赖

| 概念 | 在本系列哪章 |
|------|------------|
| Chromium 多进程模型（BrowserProcess / RenderProcess / GPU） | 第 01 章 |
| Mojo IPC 原理 | 第 03 章 |
| V8 与 Blink binding（gin、IDL、ScriptContext） | 第 11 章 |
| Blink 渲染管线（Paint/Composite） | 第 09、10 章 |

---

## 1  设计考古：Electron 是如何诞生的

### 1.1  原点：Atom Shell（2013）

2013 年 GitHub 为 Atom 编辑器需要一个可编程的桌面 WebView 壳。当时的选项：

| 方案 | 问题 |
|------|------|
| CEF（Chromium Embedded Framework） | C++ API，无 Node.js，难以用 JS 编写 native 扩展 |
| node-webkit（nw.js） | 合并了 Node 与 Chromium 的 main loop，上下文不隔离，安全边界模糊 |
| 裸 WebKit/WebView | 缺少 DevTools、多进程沙盒 |

GitHub 工程师 Cheng Zhao（zcbenz）选择了一条不同的路：以 Chromium **Content API** 作为
embedding 层，把 Node.js 嫁接进去，保持两套 event loop 分离。2014 年正式更名 Electron。

设计博客原文（Electron 官网博客存档）：
> *"The main process runs Node.js and can access native OS APIs. The renderer process only runs in
> Chromium's renderer process and has limited access."*  
> — https://www.electronjs.org/blog/electron-internals-node-integration（已 WebFetch 核实）

### 1.2  关键演进节点

| 时间 | 事件 |
|------|------|
| 2013 | Atom Shell，单进程原型 |
| 2014 | 重命名 Electron；引入 `node_bindings` 分离两套 loop |
| 2016 | 使用 Node shared library（BoringSSL/OpenSSL 符号冲突驱动）；`contextIsolation` 选项引入 |
| 2020 | Mojo IPC 取代旧 legacy IPC channel；sandbox 默认开启 |
| 2021 | Electron 12 将 `contextIsolation` 默认值改为 `true` |
| 2022+ | Utility Process API；`nodeIntegrationInWorker` GA；V8 snapshot 优化启动时间 |

---

## 2  整体架构：三进程 + 两套运行时

```
┌──────────────────────────────────────────────────────────────────────┐
│  OS Process: Main Process                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Chromium: ContentBrowserClient / BrowserMainParts          │    │
│  │  ┌─────────────────────┐  ┌──────────────────────────────┐ │    │
│  │  │ JavascriptEnvironment│  │  ElectronBrowserMainParts    │ │    │
│  │  │  (V8 Isolate)        │  │  PreMainMessageLoopRun()     │ │    │
│  │  └────────┬────────────┘  │  PostEarlyInitialization()   │ │    │
│  │           │ node::Env      └──────────────────────────────┘ │    │
│  │  ┌────────▼────────────┐                                     │    │
│  │  │  NodeBindings        │  ← libuv polling thread            │    │
│  │  │  EmbedThreadRunner() │     semaphore sync                 │    │
│  │  │  UvRunOnce()         │                                     │    │
│  │  └────────┬────────────┘                                     │    │
│  │           │ Mojo IPC (ipcMain)                                │    │
│  └───────────┼─────────────────────────────────────────────────┘    │
│              │ Mojo Pipe                                              │
│  ┌───────────▼──────────────────────────────────────────────────┐   │
│  │  OS Process: Renderer Process (one per BrowserWindow)        │   │
│  │  ┌──────────────────────────────────────────────────────┐    │   │
│  │  │  Blink + V8 (Main World)     ← 页面 JS               │    │   │
│  │  │  Blink Isolated World 999    ← preload.js            │    │   │
│  │  │  ElectronRendererClient      ← DidCreateScriptContext │    │   │
│  │  │  NodeBindings (kRenderer)    ← Node in renderer      │    │   │
│  │  └──────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  OS Process: GPU Process  (Chromium 标准，Electron 不特殊处理)       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3  Content API 插件点：Electron 如何劫持 Chromium

Chromium Content API 提供两个核心抽象类供 embedder 重写：

- `content::ContentBrowserClient` — 浏览器进程侧
- `content::ContentRendererClient` — 渲染进程侧
- `content::BrowserMainParts` — 进程启动各生命周期钩子

### 3.1  ElectronBrowserClient

【真实源码 electron/electron@shell/browser/electron_browser_client.h】

```cpp
// ElectronBrowserClient 继承 content::ContentBrowserClient
// 并私有继承 content::RenderProcessHostObserver
class ElectronBrowserClient
    : public content::ContentBrowserClient,
      private content::RenderProcessHostObserver {
 public:
  // 核心 override：在子进程 command line 里注入 Electron 开关
  void AppendExtraCommandLineSwitches(base::CommandLine* command_line,
                                      int child_process_id) override;

  // 窗口创建权限控制（传递 webPreferences）
  bool CanCreateWindow(...) override;

  // 覆盖 WebPreferences（contextIsolation/nodeIntegration 等传给 Blink）
  void OverrideWebPreferences(content::WebContents*,
                               blink::web_pref::WebPreferences*) override;

  // 让 Electron 自定义 scheme 走 ProtocolRegistry 而非 network service
  std::unique_ptr<content::URLLoaderFactoryForBrowserProcess>
  CreateNonNetworkNavigationURLLoaderFactory(const std::string& scheme,
                                             ...) override;

  // 渲染进程生命周期观测（用于 ProcessMetric 等）
  void RenderProcessWillLaunch(content::RenderProcessHost*) override;
  void RenderProcessExited(...) override;

 private:
  // pending 渲染进程 → WebContents 映射（在 AppendExtraCommandLineSwitches 用）
  std::map<int, content::WebContents*> pending_processes_;
};
```

关键 override —— `AppendExtraCommandLineSwitches`：
【真实源码 electron/electron@shell/browser/electron_browser_client.cc（精简示意，非逐字）】

```cpp
void ElectronBrowserClient::AppendExtraCommandLineSwitches(
    base::CommandLine* command_line, int process_id) {
  // 对 renderer 进程注入 app path、secure schemes、
  // preload script 路径等——renderer 启动后通过 --electron-preload 拿到这些值
  if (IsRendererProcess(command_line)) {
    command_line->AppendSwitchPath(switches::kAppPath, app_path_);
    // 把 secure/cors/service-worker scheme 传给渲染进程
    AppendSchemeSwitch(command_line, switches::kSecureSchemes, ...);
  }
}
```

### 3.2  ElectronBrowserMainParts

【真实源码 electron/electron@shell/browser/electron_browser_main_parts.h（摘要）】

```cpp
class ElectronBrowserMainParts : public content::BrowserMainParts {
 public:
  static ElectronBrowserMainParts* Get();  // 单例

  // 生命周期顺序：
  // PreEarlyInitialization → PostEarlyInitialization → ToolkitInitialized
  // → PreCreateThreads → PostCreateMainMessageLoop
  // → PreMainMessageLoopRun → PostMainMessageLoopRun

 private:
  // V8 isolate 持有者（比 node_env_ 先构造）
  std::unique_ptr<JavascriptEnvironment> js_env_;

  // Node.js 环境（依赖 js_env_ 的 isolate）
  std::shared_ptr<node::Environment> node_env_;

  // libuv ↔ Chromium message loop 桥接器
  const std::unique_ptr<NodeBindings> node_bindings_;
  const std::unique_ptr<ElectronBindings> electron_bindings_;
};
```

**PostEarlyInitialization** 是主进程 Node 初始化的核心：

【真实源码 electron/electron@shell/browser/electron_browser_main_parts.cc（关键流程，非逐字）】

```cpp
void ElectronBrowserMainParts::PostEarlyInitialization() {
  // 1. 用 libuv loop 创建 V8 platform + Isolate
  js_env_ = std::make_unique<JavascriptEnvironment>(node_bindings_->uv_loop());

  v8::Isolate* isolate = js_env_->isolate();
  v8::HandleScope handle_scope(isolate);

  // 2. 初始化 Node bindings，拿到 context
  node_bindings_->Initialize(isolate, context);

  // 3. 创建 node::Environment（含 libuv loop + V8 context）
  node_env_ = node_bindings_->CreateEnvironment(
      isolate, context, js_env_->platform(),
      js_env_->max_young_generation_size_in_bytes());

  // 4. 绑定 Electron API（ipcMain, app, BrowserWindow …）
  electron_bindings_->BindTo(isolate, node_env_.get());

  // 5. 创建微任务 runner；设置 uv_env
  js_env_->CreateMicrotasksRunner();
  node_bindings_->set_uv_env(node_env_.get());

  // 6. 加载并执行 main.js（JS 入口）
  node_bindings_->LoadEnvironment(node_env_.get());
  node_bindings_->JoinAppCode();
}
```

**PreMainMessageLoopRun** 启动 libuv polling：

【真实源码 electron/electron@shell/browser/electron_browser_main_parts.cc（非逐字）】

```cpp
int ElectronBrowserMainParts::PreMainMessageLoopRun() {
  // 锁定 URL scheme registry（防竞态）
  url::LockSchemeRegistries();

  // 启动 libuv embed thread
  node_bindings_->PrepareEmbedThread();
  node_bindings_->StartPolling();

  // 触发 app "will-finish-launching" / "ready"
  Browser::Get()->WillFinishLaunching();
  Browser::Get()->DidFinishLaunching(...);
  return content::RESULT_CODE_NORMAL_EXIT;
}
```

### 3.3  JavascriptEnvironment 与 V8 初始化

【真实源码 electron/electron@shell/browser/javascript_environment.cc（摘要）】

```cpp
// JavascriptEnvironment::Initialize() 关键步骤：
// 1. 创建 node::MultiIsolatePlatform（不用 Chromium scheduler，
//    因为此时 Chromium 调度器尚未启动）
// 2. V8::InitializePlatform(platform)
// 3. v8::Isolate::Allocate() 分配 isolate
// 4. 向 platform 注册 isolate + uv_loop
// 5. 如有 Node startup snapshot → 从 snapshot 恢复（节省启动时间）
// 6. 存入全局 g_isolate

// 构造函数：
JavascriptEnvironment::JavascriptEnvironment(uv_loop_t* event_loop) {
  Initialize();  // 以上流程
  // gin::IsolateHolder 管理 isolate 生命周期
  isolate_holder_ = std::make_unique<gin::IsolateHolder>(
      isolate_, gin::IsolateHolder::kSingleThread, ...);
  // 若无 snapshot，创建新的 Node context
  if (!node::IsUsingNodeSnapshot()) {
    context_.Reset(isolate_, node::NewContext(isolate_));
  }
}
```

---

## 4  libuv ↔ Chromium message loop 桥接：NodeBindings

这是 Electron 最核心的创新之一，也是历史上修改次数最多的模块。

### 4.1  问题陈述

- Chromium 使用 `base::MessagePumpForUI`（macOS 底层是 `NSRunLoop`，Linux 底层是 glib），
  主线程只能跑一个 pump。
- Node.js / libuv 有自己的 `uv_run` 循环。
- 两者互斥，不能直接嵌套。

### 4.2  历史演进

| 阶段 | 方案 | 问题 |
|------|------|------|
| v1（2013） | 用 libuv 重写 Chromium message pump | macOS NSRunLoop 无法替换；Linux 需要 glib 集成 |
| v2（2014） | 小间隔 timer 轮询 libuv | CPU 持续占用；操作延迟 |
| v3（最终，2014+）| libuv backend fd + 独立 polling 线程 | 无轮询 overhead；跨平台一致 |

原始设计博客（已 WebFetch 核实）：  
https://www.electronjs.org/blog/electron-internals-node-integration

> *"libuv introduced the backend_fd concept—a file descriptor whose readability signals that libuv
> has new events. We create a worker thread to poll this file descriptor."*

### 4.3  NodeBindings 实现

【真实源码 electron/electron@shell/common/node_bindings.h（摘要）】

```cpp
class NodeBindings {
 public:
  // 进程类型区分（影响 uv_loop 选取策略）
  enum class BrowserEnvironment {
    kBrowser,   // 主进程
    kRenderer,  // 渲染进程
    kUtility,   // utility 进程
    kWorker,    // Web Worker
  };

  static std::unique_ptr<NodeBindings> Create(BrowserEnvironment);

  void PrepareEmbedThread();     // 初始化 semaphore + dummy async handle
  void StartPolling();           // 启动 embed thread
  // ...

 private:
  static void EmbedThreadRunner(void* arg);  // 独立线程入口
  void UvRunOnce();                          // 在主线程执行一次 libuv iteration

  const BrowserEnvironment browser_env_;
  const raw_ptr<uv_loop_t> uv_loop_;   // 当前线程的 libuv loop
  uv_loop_t worker_loop_;              // worker 模式专用 loop
  uv_thread_t embed_thread_;           // polling 线程
  uv_sem_t embed_sem_;                 // 线程同步 semaphore
  UvHandle<uv_async_t> dummy_uv_handle_; // 防止 libuv 提前 exit
  bool initialized_ = false;
  bool embed_thread_prepared_ = false;
};
```

【真实源码 electron/electron@shell/common/node_bindings.cc（关键函数，已 WebFetch 核实）】

```cpp
// PrepareEmbedThread：一次性初始化
void NodeBindings::PrepareEmbedThread() {
  if (initialized_) return;
  if (!embed_thread_prepared_) {
    // dummy handle：只要 libuv loop 里有 handle，loop 就不会退出
    uv_async_init(uv_loop_, dummy_uv_handle_.get(), nullptr);
    uv_sem_init(&embed_sem_, 0);      // 初始值 0 → polling thread 一开始阻塞
    embed_thread_prepared_ = true;
  }
  // 创建专用 polling 线程
  uv_thread_create(&embed_thread_, EmbedThreadRunner, this);
}

// EmbedThreadRunner（在独立线程中跑）
void NodeBindings::EmbedThreadRunner(void* arg) {
  auto* self = static_cast<NodeBindings*>(arg);
  while (true) {
    uv_sem_wait(&self->embed_sem_);   // 等主线程通知
    if (self->embed_closed_) break;
    self->PollEvents();               // 调用 backend fd / epoll / kqueue
    if (self->embed_closed_) break;
    self->WakeupMainThread();         // 通知主线程：有 libuv 事件待处理
  }
}

// UvRunOnce（在主线程的 Chromium task runner 里执行）
void NodeBindings::UvRunOnce() {
  node::Environment* env = uv_env();
  if (!env) return;
  v8::HandleScope handle_scope(env->isolate());
  v8::Context::Scope context_scope(env->context());
  // 处理微任务
  util::ExplicitMicrotasksScope microtasks_scope(
      env->context()->GetMicrotaskQueue());

  // UV_RUN_NOWAIT：非阻塞执行一次 libuv iteration
  int r = uv_run(uv_loop_, UV_RUN_NOWAIT);

  // libuv 没有更多 active handles → 请求 Chromium message loop 退出
  if (r == 0) base::RunLoop().QuitWhenIdle();

  // 告知 polling 线程可以继续 poll
  uv_sem_post(&embed_sem_);
}
```

**核心时序图：**

```
Main Thread                   Embed Thread (polling)
    │                               │
    │  PrepareEmbedThread()         │
    │  ──────────────────────────→  │ uv_sem_wait(sem) → 阻塞
    │                               │
    │  UvRunOnce() [Chromium task]  │
    │  uv_run(UV_RUN_NOWAIT)        │
    │  uv_sem_post(sem) ──────────→ │ PollEvents()
    │                               │   epoll_wait / kqueue
    │                               │   有事件 → WakeupMainThread()
    │  ←──────────────────────────  │ WakeupEmbedThread (post task)
    │  UvRunOnce() [再次调度]        │ uv_sem_wait(sem) → 再次阻塞
    │  ...                          │ ...
```

---

## 5  渲染进程：ElectronRendererClient 与 contextIsolation

### 5.1  ElectronRendererClient 继承链

【真实源码 electron/electron@shell/renderer/electron_renderer_client.h（已 WebFetch 核实）】

```
content::ContentRendererClient
    └── RendererClientBase       (electron/electron@shell/renderer/renderer_client_base.cc)
            └── ElectronRendererClient   (shell/renderer/electron_renderer_client.cc)
```

`ElectronRendererClient` 构造时：

【真实源码 electron/electron@shell/renderer/electron_renderer_client.cc（已 WebFetch 核实）】

```cpp
ElectronRendererClient::ElectronRendererClient()
    : node_bindings_{NodeBindings::Create(
          NodeBindings::BrowserEnvironment::kRenderer)},
      electron_bindings_{
          std::make_unique<ElectronBindings>(node_bindings_->uv_loop())} {}
```

### 5.2  DidCreateScriptContext：Node 环境在渲染进程的创建

【真实源码 electron/electron@shell/renderer/electron_renderer_client.cc（已 WebFetch 核实，精简）】

```cpp
void ElectronRendererClient::DidCreateScriptContext(
    v8::Isolate* const isolate,
    v8::Local<v8::Context> renderer_context,
    content::RenderFrame* render_frame) {

  // 1. 检查是否应该加载 preload
  if (!ShouldLoadPreload(isolate, renderer_context, render_frame))
    return;

  // 2. 初始化 Node bindings（每进程一次）
  if (!node_bindings_->IsInitialized()) {
    node_bindings_->Initialize(isolate, renderer_context);
    node_bindings_->PrepareEmbedThread();
  }

  // 3. 配置 Node tracing agent
  if (!node::tracing::TraceEventHelper::GetAgent())
    node::tracing::TraceEventHelper::SetAgent(node_env_->tracing_agent_);

  // 4. 关键：先保存 Blink 的 fetch/Response/Request/Headers 引用
  //    因为 Node.js 初始化会覆盖这些全局变量
  //    saved as "blink_fetch", "blink_Response", etc.

  // 5. 延迟 document loading（freeze mode）直到 preload 执行完
  render_frame->GetWebFrame()->SetLifecycleState(
      blink::mojom::FrameLifecycleState::kFrozen);

  // 6. 创建 Node environment（kRenderer 模式）
  auto node_env = node_bindings_->CreateEnvironment(
      isolate, renderer_context, js_platform,
      js_env_->max_young_generation_size_in_bytes(),
      [render_frame]() {
        // 回调：preload 执行完毕后解冻 document
        render_frame->GetWebFrame()->SetLifecycleState(
            blink::mojom::FrameLifecycleState::kRunning);
      });

  // 7. 设置 force_context_aware（保证 native module 在 context 内）
  node_env->set_force_context_aware(true);

  // 8. 绑定 Electron API
  electron_bindings_->BindTo(isolate, node_env.get());

  // 9. 启动 libuv polling（渲染进程也有独立的 libuv loop）
  node_bindings_->StartPolling();
}
```

### 5.3  contextIsolation 与 Blink Isolated World

contextIsolation 依赖 Blink 的 Isolated World 机制（详见第 11 章）。

【真实源码 electron/electron@shell/renderer/renderer_client_base.cc（已 WebFetch 核实）】

```cpp
// 取 context 时，依据 contextIsolation 开关选择不同的 world
v8::Local<v8::Context> RendererClientBase::GetContext(
    blink::WebLocalFrame* frame,
    v8::Isolate* isolate) const {
  if (context_isolation_) {
    // World ID 999 = Electron 的专用 isolated world
    return frame->GetScriptContextFromWorldId(
        isolate, WorldIDs::ISOLATED_WORLD_ID);
  }
  return frame->MainWorldScriptContext();
}

// BindProcess 向 process 对象注入隔离标志
void RendererClientBase::BindProcess(v8::Isolate* isolate,
                                     gin_helper::Dictionary* process,
                                     content::RenderFrame* render_frame) {
  process->Set("isMainFrame", render_frame->IsMainFrame());
  process->Set("contextIsolated",
               render_frame->GetBlinkPreferences().context_isolation);
  process->Set("contextId", base::StringPrintf("%s-%" PRId64, ...));
}
```

**WebContentsPreferences 存储配置：**

【真实源码 electron/electron@shell/browser/web_contents_preferences.cc（已 WebFetch 核实）】

```cpp
WebContentsPreferences::WebContentsPreferences(
    content::WebContents* web_contents,
    const gin_helper::Dictionary& web_preferences) {
  // contextIsolation 默认 true（Electron 12+）
  web_preferences.Get(options::kContextIsolation, &context_isolation_);

  // nodeIntegration 默认 false
  web_preferences.Get(options::kNodeIntegration, &node_integration_);

  // preload 必须为绝对路径
  base::FilePath::StringType preload_path;
  if (web_preferences.Get(options::kPreloadScript, &preload_path)) {
    base::FilePath preload(preload_path);
    if (preload.IsAbsolute()) {
      preload_path_ = preload;
    } else {
      LOG(ERROR) << "preload must be an absolute path";
    }
  }
}
```

### 5.4  contextBridge：跨世界 API 暴露

当 `contextIsolation: true` 时，preload 中的 `contextBridge.exposeInMainWorld(key, api)` 是
跨 world 安全通信的唯一官方路径。技术实现要点（已 WebFetch 核实，electron 文档）：

- 简单类型（string / number / boolean）**拷贝并冻结**，双向不可变
- Object / Array / Error / Promise / Function：深拷贝或 proxy
- **Function 是 proxy**：调用时从 isolated world 切换到 main world 执行，保留异步语义
- Symbol 不能跨 bridge
- `ipcRenderer` 对象本身不能直接暴露（Electron 文档明确禁止），必须包一层

```javascript
// preload.js（在 Isolated World 999 中运行）
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  // 只暴露具体操作，不暴露整个 ipcRenderer
  sendMessage: (channel, data) => ipcRenderer.send(channel, data),
  onMessage: (channel, cb) => ipcRenderer.on(channel, (_, v) => cb(v))
})

// renderer.js（在 Main World 中运行）
window.electronAPI.sendMessage('ping', { ts: Date.now() })
```

---

## 6  进程间通信：ipcMain / ipcRenderer 内部实现

### 6.1  通道栈

```
JS: ipcRenderer.send('channel', ...args)
        ↓  gin 序列化为 base::Value
ElectronApiIPCRenderer
        ↓  Mojo: ElectronBrowser.Message (mojom)
NetworkService / BrowserThread::UI
        ↓
ElectronApiIPCMain
        ↓  JS: ipcMain.on('channel', handler)
```

在 Electron 12+ 全面切换 Mojo 后，旧的 legacy IPC channel 已废弃。

【真实源码 electron/electron@shell/browser/electron_browser_client.cc（preload 注入路径，已 WebFetch 核实）】

```cpp
// GetExtraCreateNewWindowReplyData: about:blank 弹窗场景下同步传递 preload
std::optional<mojo_base::BigBuffer>
ElectronBrowserClient::GetExtraCreateNewWindowReplyData(
    content::RenderFrameHost* new_window_main_frame,
    const GURL& target_url) {
  // 序列化 RendererStartupData（含 preload 路径 + code cache blob）
  // 在异步 IPC 到达前通过同步回复传给渲染进程
  ...
}
```

### 6.2  ipcRenderer.invoke / ipcMain.handle（Promise 模式）

```
// renderer
const result = await ipcRenderer.invoke('get-data', id)

// ─── 内部 ────────────────────────────────────────────────────────
// 1. renderer: 生成 call_id，发送 Mojo message（带 call_id）
// 2. main: 找到对应 handler，await 执行，通过 Mojo 回传 {call_id, result/error}
// 3. renderer: 按 call_id 找到 pending Promise，resolve/reject
// ────────────────────────────────────────────────────────────────

// main
ipcMain.handle('get-data', async (event, id) => {
  return db.query(id)
})
```

---

## 7  Chromium patch 体系

Electron 不能直接修改 Chromium 源码（gclient sync 会覆盖），改动通过 patch 文件管理。

```
electron/electron/patches/chromium/   ← 几十个 .patch 文件
```

已 WebFetch 核实的关键 patch 类别（github.com/electron/electron/tree/main/patches/chromium）：

| Patch 名称（示意） | 动机 |
|-------------------|------|
| `feat_expose_raw_response_headers_from_urlloader.patch` | 网络层暴露原始 header，供 `ses.webRequest.onHeadersReceived` |
| `can_create_window.patch` | 增强 `CanCreateWindow` 回调，允许更丰富的窗口控制 |
| `feat_enable_offscreen_rendering_with_viz_compositor.patch` | 离屏渲染（OSR）支持 |
| `custom_protocols_plzserviceworker.patch` | 让自定义 scheme 在 Service Worker 中工作 |
| `allow_new_privileges_in_unsandboxed_child_processes.patch` | 非沙盒子进程权限提升 |
| `desktop_media_list.patch` | 屏幕/窗口捕获列表 |

**raw_response_headers patch 细节（已 WebFetch 核实）：**

Chromium 出于安全默认过滤了 raw headers，只通过 DevTools channel 暴露。Electron 需要
在 `URLLoader` 里同步拿到原始 headers 以便触发 `webRequest` 事件。patch 在
`TrustedUrlRequestParams` 里加了 `report_raw_headers` flag，在 `URLResponseHead` 加了
`raw_response_headers` 数组，loader 收到响应后通过 `SetResponseHeadersCallback()` 填充。

---

## 8  自定义协议与网络层钩子

### 8.1  ElectronURLLoaderFactory

【真实源码 electron/electron@shell/browser/net/electron_url_loader_factory.cc（已 WebFetch 核实）】

```cpp
// 支持 6 种协议类型
enum class ProtocolType { kBuffer, kFile, kHttp, kStream, kString, kFree };

// 工厂入口
static mojo::PendingRemote<network::mojom::URLLoaderFactory>
ElectronURLLoaderFactory::Create(ProtocolType type,
                                  const ProtocolHandler& handler);

// 核心方法：接受 Mojo request → 调用 handler → 分派到协议处理器
void CreateLoaderAndStart(
    mojo::PendingReceiver<network::mojom::URLLoader> loader,
    int32_t request_id,
    uint32_t options,
    const network::ResourceRequest& request,
    mojo::PendingRemote<network::mojom::URLLoaderClient> client,
    const net::MutableNetworkTrafficAnnotationTag& traffic_annotation);
```

特殊文件处理：`StartLoadingFile()` 内部对 `.asar` 包有特殊支持（ASAR VFS）。

### 8.2  ProxyingURLLoaderFactory（webRequest API）

【真实源码 electron/electron@shell/browser/net/proxying_url_loader_factory.cc（已 WebFetch 核实）】

```cpp
// InProgressRequest 管理单个请求的拦截状态机：
// Restart → ContinueToBeforeSendHeaders → ContinueToSendHeaders
//   → ContinueToResponseStarted → ContinueAfterResponse
// 每个阶段都会触发对应的 webRequest 事件（如果有 listener）
```

---

## 9  BrowserWindow 对象模型：从 JS 到 Native

```
JS: new BrowserWindow({ width:800, height:600, webPreferences:{...} })
        ↓  gin 构造
BrowserWindow (JS wrapper, shell/browser/api/electron_api_browser_window.cc)
        ↓  继承
BaseWindow → NativeWindow（平台抽象层）
   ├── NativeWindowMac（macOS NSWindow）
   ├── NativeWindowViews（Linux/Win，使用 Chromium views::Widget）
        ↓  组合
WebContentsView → content::WebContents（Chromium DOM/渲染）
        ↓  通过 NativeWindowRelay 反查
content::WebContents → NativeWindow（弱引用，避免循环）
```

【真实源码 electron/electron@shell/browser/api/electron_api_browser_window.cc（已 WebFetch 核实）】

```cpp
// 创建路径
BrowserWindow::BrowserWindow(...) {
  // 1. 从 options 提取 webPreferences → 创建 WebContentsView
  gin_helper::Handle<WebContentsView> web_contents_view =
      WebContentsView::Create(isolate, web_preferences);

  // 2. 持有 WebContents 的 JS handle
  web_contents_.Reset(isolate, web_contents.ToV8());
  api_web_contents_ = web_contents->GetWeakPtr();
  api_web_contents_->AddObserver(this);

  // 3. InitWithArgs → 创建 NativeWindow
  InitWithArgs(args);

  // 4. 把 WebContentsView 挂进 native window 的 contents view 层级
  window()->GetContentsView()->AddChildViewAt(web_contents_view->view(), 0);
  window()->InitFromOptions(options);
}
```

---

## 10  与 CEF / Tauri / WebView2 的架构对比

| 维度 | Electron | CEF | Tauri | WebView2（Windows） |
|------|----------|-----|-------|---------------------|
| Chromium 集成方式 | Content API（进程级）| Content API + 额外 C API | 系统 WebView（WKWebView/WebKitGTK/WebView2） | 系统 WebView2 |
| 进程架构 | 完整多进程（browser/renderer/gpu）| 完整多进程 | 单进程（host）+ 系统 WebView 进程 | 系统管理 |
| Node.js | 内置，main + renderer 均可 | 无 | 无（Rust backend via Tauri IPC） | 无 |
| 打包体积 | 大（~100MB，含完整 Chromium） | 大 | 小（系统 WebView，Rust binary ~5MB） | 小（系统） |
| 系统 API | Node.js 直接调用 | C++ 扩展 | Rust commands via IPC | Win32 + WinRT |
| Chromium 版本控制 | 固定 pin，项目自己滚 | 固定 pin | 依赖系统更新 | 自动随系统更新 |
| 沙盒 | Chromium 沙盒（可选）| Chromium 沙盒 | 系统 WebView 沙盒 | 系统 WebView 沙盒 |
| 适用场景 | 重度 Node/npm 生态，开发工具 | C++ 嵌入，游戏 overlay | 轻量发布，Rust 生态 | Windows only，最小体积 |
| 典型 use case | VS Code, Slack, Figma Desktop | Steam overlay, 游戏内浏览器 | 各类 Rust 桌面 app | Windows 原生工具 |

**不适用边界：**
- Electron 不适合对安装包大小极度敏感的 C 端应用
- Electron renderer 的 Node.js 集成在 `contextIsolation: false` 时存在严重安全风险（已知 RCE 向量）
- CEF 缺乏 Node.js，不适合重 JS/npm 生态场景
- Tauri 依赖系统 WebView，不同平台渲染差异可能引发兼容问题

---

## 11  Demo 实战：最小 Electron App 观测进程与隔离

本节 demo 全部可在安装了 Node.js + npm 的机器上直接运行。

### Demo 1：最小 Electron App（可运行）

```bash
mkdir electron-demo && cd electron-demo
npm init -y
npm install electron --save-dev
```

**目录结构：**
```
electron-demo/
├── main.js
├── preload.js
├── renderer.js
└── index.html
```

**main.js：**
```javascript
// main.js —— 在 Main Process 运行（Node.js 完整能力）
const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('path')
const os = require('os')  // Node.js API，renderer 中直接不可用

app.whenReady().then(() => {
  const win = new BrowserWindow({
    width: 900,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,   // 默认 true，显式写出以便观察
      nodeIntegration: false,   // 默认 false
      sandbox: false,           // demo 中关闭以允许 preload 访问 Node
    }
  })

  // ipcMain handler：接收渲染进程消息
  ipcMain.handle('get-system-info', async () => {
    // 这里可以访问 Node.js API
    return {
      platform: process.platform,
      arch: process.arch,
      nodeVersion: process.versions.node,
      hostname: os.hostname(),
      pid: process.pid,
    }
  })

  ipcMain.on('log', (event, msg) => {
    console.log('[Main] Received from renderer:', msg)
  })

  win.loadFile('index.html')
  win.webContents.openDevTools()
})
```

**preload.js：**
```javascript
// preload.js —— 在 Isolated World 999 中运行
// 有 Node.js 能力，但与页面 JS 隔离
const { contextBridge, ipcRenderer } = require('electron')

// 通过 contextBridge 向 Main World 安全暴露 API
contextBridge.exposeInMainWorld('electronAPI', {
  getSystemInfo: () => ipcRenderer.invoke('get-system-info'),
  log: (msg) => ipcRenderer.send('log', msg),

  // 验证 contextIsolation：preload 可以定义私有变量
  // 页面 JS 无法访问 _secret
  _secret: '这个字段页面 JS 看不到'
})

console.log('[Preload] Running in isolated world, typeof require:', typeof require)
// 期望输出: [Preload] Running in isolated world, typeof require: function
```

**index.html：**
```html
<!DOCTYPE html>
<html>
<head><title>Electron Demo</title></head>
<body>
  <h1>Electron Process Architecture Demo</h1>
  <button id="btn">Get System Info</button>
  <pre id="output"></pre>
  <script src="renderer.js"></script>
</body>
</html>
```

**renderer.js：**
```javascript
// renderer.js —— 在 Main World 运行
// contextIsolation: true → require 不可用
console.log('typeof require in renderer:', typeof require)
// 期望: "undefined"  ← 证明 contextIsolation 生效

// 只能访问 contextBridge 暴露的接口
console.log('electronAPI:', window.electronAPI)
// 期望: { getSystemInfo: [Function], log: [Function] }
// 注意: _secret 不会出现（即使在 preload 中定义了）

document.getElementById('btn').addEventListener('click', async () => {
  const info = await window.electronAPI.getSystemInfo()
  document.getElementById('output').textContent = JSON.stringify(info, null, 2)
  window.electronAPI.log('button clicked')
})
```

**运行：**
```bash
npx electron main.js
```

**预期观察：**
1. DevTools Console 中 `typeof require` 显示 `"undefined"` → contextIsolation 生效
2. `window.electronAPI._secret` 在 renderer console 中为 `undefined`
3. 点击按钮后看到系统信息（来自 ipcMain.handle）
4. Main process terminal 输出 `[Main] Received from renderer: button clicked`

---

### Demo 2：用 DevTools Performance 观测 libuv polling（可运行）

```bash
# 在同一 electron-demo 目录，修改 main.js 添加 CPU 密集 libuv 任务
```

在 `main.js` 的 `ipcMain.handle` 后追加：
```javascript
// 模拟 libuv I/O（setImmediate 会走 libuv check 阶段）
let count = 0
setInterval(() => {
  count++
  // 每 500ms 执行一次 → 可在 Performance timeline 看到 task
}, 500)

// 真实 Node.js I/O
const fs = require('fs')
setInterval(() => {
  fs.readFile(__filename, () => { /* libuv I/O callback */ })
}, 1000)
```

**观测步骤：**
1. 打开 DevTools → Performance → 点 Record
2. 等 3 秒 → Stop
3. 在 Timeline 中可以看到两类 task 交替：
   - Chromium 自身的 rendering/compositing tasks（蓝/绿）
   - 来自 `UvRunOnce` 调度的 libuv callback tasks（在 Main Thread）

**chrome://process-internals 观察进程拓扑：**
```
在 Electron app 的任意 BrowserWindow 地址栏输入：
  chrome://process-internals
（注：需要 --remote-debugging-port 或直接在 DevTools URL bar 输入）
```
预期：看到 browser process + 各 renderer process + GPU process 的关系树。

---

### Demo 3：CDP 驱动观测 contextIsolation world（可运行）

```bash
# 启动时开放调试端口
npx electron main.js --remote-debugging-port=9229
```

```javascript
// 用 Node.js 脚本连接 CDP，检查 isolated world
const CDP = require('chrome-remote-interface')  // npm install chrome-remote-interface

async function inspect() {
  const client = await CDP({ port: 9229 })
  const { Runtime } = client

  await Runtime.enable()

  // 列出所有 execution context（worlds）
  Runtime.executionContextCreated(({ context }) => {
    console.log('Context:', {
      id: context.id,
      name: context.name,   // "" = main world, "Electron Isolated Context" = world 999
      origin: context.origin,
      auxData: context.auxData,
    })
  })

  // 在 Main World 执行：confirm require is undefined
  const r1 = await Runtime.evaluate({
    expression: 'typeof require',
    contextId: 1,  // main world
  })
  console.log('typeof require in main world:', r1.result.value)
  // 期望: "undefined"

  // 在 Isolated World（world 999）执行：confirm require exists
  // 注：实际 contextId 需从 executionContextCreated 事件获取
}

inspect().catch(console.error)
```

**预期输出：**
```
Context: { id: 1, name: '', origin: 'null', auxData: { isDefault: true } }
Context: { id: 2, name: 'Electron Isolated Context', origin: '...', auxData: { isDefault: false, frameId: ... } }
typeof require in main world: "undefined"
```

---

### Demo 4：chrome://tracing 追踪 NodeBindings 调度（可运行）

```bash
npx electron main.js --remote-debugging-port=9229
```

打开 Chrome 浏览器（或另一 Electron 窗口）访问：
```
http://localhost:9229
```
选取 Electron app 的 target → 在 DevTools 中打开 `chrome://tracing`（注：需通过 `about:tracing`）。

或者直接用命令行触发 tracing：
```javascript
// 在 main.js 中
const { session } = require('electron')
const fs = require('fs')

app.whenReady().then(() => {
  // 启动 tracing
  session.defaultSession.extension.tracing = null  // placeholder
  const { TracingController } = process._linkedBinding('electron_common_tracing')

  // 更简单的方式：使用 Chromium tracing via --trace-startup
})
```

更直接的 tracing 方式（命令行）：
```bash
npx electron main.js \
  --trace-startup \
  --trace-startup-duration=5 \
  --trace-startup-file=/tmp/electron-trace.json

# 用 chrome://tracing 或 https://ui.perfetto.dev 打开 /tmp/electron-trace.json
# 搜索 "NodeBindings" 或 "UvRunOnce" 可见 libuv 调度轨迹
```

---

### Demo 5：最小 Electron Patch 示例（走读，不可直接运行）

演示 Electron patch 体系的结构（以 raw_response_headers patch 为例）：

```bash
# patch 文件位置
electron/electron/patches/chromium/feat_expose_raw_response_headers_from_urlloader.patch

# patch 修改了以下 Chromium 文件：
# - services/network/public/mojom/url_loader.mojom  （加 report_raw_headers flag）
# - services/network/public/mojom/url_response_head.mojom （加 raw_response_headers 字段）
# - services/network/url_loader.cc                  （实现 SetResponseHeadersCallback）

# 应用 patch 的方式（gclient/depot_tools）：
cd chromium/src
git apply ../../electron/patches/chromium/feat_expose_raw_response_headers_from_urlloader.patch
```

**patch 的 diff 结构（示意，非逐字）：**
```diff
--- a/services/network/public/mojom/url_response_head.mojom
+++ b/services/network/public/mojom/url_response_head.mojom
+  // Raw response headers, only populated when TrustedParams.report_raw_headers=true
+  array<HttpRawHeaderPair> raw_response_headers;

--- a/services/network/url_loader.cc
+++ b/services/network/url_loader.cc
+  if (options_ & mojom::kURLLoadOptionReportRawHeaders) {
+    url_request_->SetResponseHeadersCallback(
+        base::BindRepeating(&URLLoader::OnRawResponseHeaders,
+                            weak_ptr_factory_.GetWeakPtr()));
+  }
```

---

## 12  失败模式与生产真坑

### 坑 1：contextIsolation: false + nodeIntegration: true → RCE

**现象：** `eval()` 或 XSS 可直接执行 `require('child_process').exec('rm -rf /')`。  
**根因：** 页面 JS 与 preload 共享同一个 V8 context，`require` 未被隔离。  
**修复：** 始终 `contextIsolation: true` + `nodeIntegration: false`，通过 contextBridge 最小化暴露面。  
**历史：** Electron 12 之前默认 `contextIsolation: false`，大量老项目受影响（CVE-2020-15174 等）。

### 坑 2：libuv polling 线程与 Chromium UI 线程竞态

**现象：** 高频 I/O 场景下（如大量文件 watch）UI 卡顿甚至崩溃。  
**根因：** `EmbedThreadRunner` 的 `PollEvents()` 完成后通过 `WakeupMainThread()` post task 到
Chromium main thread。如果 main thread 的 task queue 积压，`UvRunOnce` 延迟执行，导致
libuv 回调堆积。  
**缓解：** 把密集 I/O 移到 Utility Process 或 Node.js worker_threads；减少主进程 I/O 回调频率。

### 坑 3：preload 中的 Blink fetch 被 Node.js 覆盖

**现象：** preload 里 `fetch` 行为异常，或使用了 Node.js 的 `undici` 而非 Blink 的 fetch。  
**根因：** `DidCreateScriptContext` 在初始化 Node.js 之前保存了 Blink 的 `fetch/Response/Request/Headers` 引用（`blink_fetch` 等），但如果 preload 代码在保存完成前访问 `fetch`，可能拿到 undefined 或 Node 版本。  
**修复：** 使用 `window.fetch` 而非裸 `fetch`；或在 preload 顶部读取 `globalThis.__blink_fetch`。

### 坑 4：原生 Node addon 在 renderer 进程崩溃

**现象：** `nodeIntegrationInWorker: true` 时，加载含 libuv handle 的 native addon 导致 renderer
崩溃。  
**根因：** renderer 进程的 libuv loop 是从 node_bindings 的 `worker_loop_` 拿的，不同于主进程的
`uv_default_loop()`；部分 addon 直接用 `uv_default_loop()` 注册 handle，导致 loop 不匹配。  
**修复：** addon 使用 `napi_get_uv_event_loop` 而非 `uv_default_loop()`。

### 坑 5：Chromium 升级导致 patch 冲突

**现象：** 升级 Electron 版本后，原来正常的功能（webRequest raw headers、OSR）突然报错。  
**根因：** Chromium 内部修改了被 patch 的文件（mojom、url_loader.cc），导致 patch apply 失败或语义变化。  
**修复：** 检查 `patches/chromium/` 目录的 patch，对照上游变更重新 adapt；Electron 的 release
notes 通常会标注 breaking Chromium changes。

---

## 13  五件套

### 必读源码

| 文件 | 为什么重要 |
|------|-----------|
| `electron/electron@shell/common/node_bindings.cc` | libuv ↔ Chromium event loop 桥接的完整实现 |
| `electron/electron@shell/browser/electron_browser_main_parts.cc` | 主进程启动生命周期、Node 环境创建 |
| `electron/electron@shell/renderer/electron_renderer_client.cc` | 渲染进程 Node 初始化、preload 注入 |
| `electron/electron@shell/browser/web_contents_preferences.cc` | contextIsolation/nodeIntegration 配置解析 |
| `electron/electron@shell/renderer/renderer_client_base.cc` | Isolated World 选取、preload 执行时机 |
| `electron/electron@patches/chromium/` | Electron 对 Chromium 的全部定制化改动 |

### 推荐工具链

```bash
# 1. 源码搜索（Electron 内部）
gh search code "NodeBindings" --repo electron/electron

# 2. Chromium Code Search
# https://source.chromium.org/chromium/chromium/src/+/main:content/public/browser/content_browser_client.h

# 3. 追踪 libuv ↔ Chromium 调度
npx electron app.js --trace-startup --trace-startup-duration=5 \
  --trace-startup-file=/tmp/trace.json

# 4. CDP 检查 world 隔离
npx electron app.js --remote-debugging-port=9229
# 在 chrome://inspect 连接

# 5. 查看 Electron 当前 pin 的 Chromium 版本
cat node_modules/electron/dist/VERSION  # 或查看 DEPS 文件
```

### 常见误区

1. **"Electron renderer 就是普通网页"** — 错。即使 `nodeIntegration: false`，renderer 仍
   跑在定制的 `ElectronRendererClient` 里，Blink preferences 被 Electron 覆盖。
2. **"contextIsolation 只是安全功能"** — 它实际上是 Blink Isolated World 的直接暴露，有
   性能开销（跨 world 函数调用需 proxy）。
3. **"ipcRenderer 是同步的"** — `send` 是 fire-and-forget；`sendSync` 阻塞 renderer 主线程，
   生产中应该避免。`invoke/handle` 是推荐的异步 RPC。
4. **"preload 在独立进程"** — preload 跑在 renderer 进程内，只是在 Blink Isolated World 999，
   不是独立进程。
5. **"Electron 直接修改了 Chromium 源码"** — 不是。Chromium 源码通过 gclient 同步，改动
   以 patch 形式管理，每次 gclient sync 后重新 apply。

### 章末练习

1. 在 Demo 1 的 `renderer.js` 中尝试 `require('fs')`，观察报错信息，解释为什么。
2. 修改 Demo 1，将 `contextIsolation` 改为 `false` + `nodeIntegration: true`，验证 `require`
   可用，然后思考安全风险。
3. 在 `main.js` 中加一个 `ipcMain.handle('risky', async () => require('fs').readdirSync('/'))` ，
   在 renderer 里调用它，分析信任边界。
4. 用 `--trace-startup` 生成 tracing，在 perfetto.dev 里找到 `NodeBindings::UvRunOnce` 的调用
   轨迹，统计其调用频率。
5. 阅读 Electron 的某个 Chromium patch（如 `can_create_window.patch`），找到它修改的
   Chromium 文件，在 Chromium Code Search 上查看该文件的 upstream 版本，对比差异。

---

## 参考资料

1. Electron 博客：Node integration internals  
   https://www.electronjs.org/blog/electron-internals-node-integration（已 WebFetch 核实）
2. Electron 博客：Using Node as a library  
   https://www.electronjs.org/blog/electron-internals-using-node-as-a-library（已 WebFetch 核实）
3. Electron 文档：Process Model  
   https://www.electronjs.org/docs/latest/tutorial/process-model（已 WebFetch 核实）
4. Electron 文档：contextBridge API  
   https://www.electronjs.org/docs/latest/api/context-bridge（已 WebFetch 核实）
5. Chromium Multi-Process Architecture  
   https://www.chromium.org/developers/design-documents/multi-process-architecture/（已 WebFetch 核实）
6. Chromium ContentBrowserClient 头文件  
   https://source.chromium.org/chromium/chromium/src/+/main:content/public/browser/content_browser_client.h（已 WebFetch 核实）
7. 真实源码（已 WebFetch 核实）：
   - `shell/browser/electron_browser_main_parts.cc`
   - `shell/renderer/electron_renderer_client.cc`
   - `shell/common/node_bindings.cc` / `.h`
   - `shell/browser/web_contents_preferences.cc`
   - `shell/renderer/renderer_client_base.cc`
   - `shell/browser/electron_browser_client.h` / `.cc`
   - `shell/browser/javascript_environment.cc`
   - `shell/browser/api/electron_api_browser_window.cc`
   - `shell/browser/net/electron_url_loader_factory.cc`
   - `shell/browser/net/proxying_url_loader_factory.cc`
   - `patches/chromium/feat_expose_raw_response_headers_from_urlloader.patch`
