---
title: "Electron 改造与定制（Chromium 域）"
slug: "7-14"
collection: "tech-library"
group: "chromium内核"
order: 7014
summary: "TL;DR Electron 不是「写一个 app」，而是「维护一个 Chromium fork」。它从不直接改 Chromium 源码—— `gclient sync` 会无情覆盖——而是把所有改动编码成 patch 文件，每次同步后用 `git am --3way` 重放到指定 repo。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T12:50:14.000Z"
updatedAt: "2026-06-12T12:50:14.000Z"
---
> **TL;DR**
> Electron 不是「写一个 app」，而是「维护一个 Chromium fork」。它从不直接改 Chromium 源码——
> `gclient sync` 会无情覆盖——而是把所有改动编码成 **patch 文件**，每次同步后用 `git am --3way`
> 重放到指定 repo。本章拆四根支柱：① **patch 体系**（`patches/config.json` 把 13 个 patch 目录
> 映射到 chromium/v8/node/boringssl/ffmpeg 等子树，`.patches` manifest 定义 apply 顺序，
> `apply_all_patches.py` 用 ThreadPoolExecutor 并行重放）；② **build 体系**（depot_tools + gclient
> + gn + ninja，`build/args/all.gn` 把 Electron 的全部 GN 决策集中，`is_electron_build=true` /
> `root_extra_deps=["//electron"]` 把 `//electron:electron` 挂进 Chromium 构建图）；③ **版本治理**
> （Chromium 4 周一发，Electron 8 周一个 major 跟版，DEPS 里 `chromium_version` 是单点 pin，
> 升级=patch rebase 战争）；④ **裁剪/native 模块**（`node_module_version` ABI 锁定、proprietary
> codecs、framework 拆分）。理解这四根支柱，Electron 升级地狱与体积地狱的每一个坑都有根可追。
>
> 与第 13 章的分工：第 13 章讲**运行时**（NodeBindings 怎么缝两个 event loop、contextIsolation
> 怎么隔离 world）；本章讲**构建时与维护时**（patch 怎么打、build 怎么配、版本怎么跟、包怎么裁）。
> 第 13 章是「Electron 跑起来之后长什么样」，本章是「Electron 怎么被造出来、怎么不被 Chromium
> 甩下车」。

---

## 前置依赖

| 概念 | 在本系列哪章 |
|------|------------|
| Chromium 多进程模型 / zygote / sandbox | 第 01、02 章 |
| Mojo IPC、.mojom 代码生成 | 第 03 章 |
| V8 snapshot、Blink binding、IDL | 第 11 章 |
| Electron 运行时架构（NodeBindings / ContentBrowserClient override / contextIsolation） | **第 13 章（强依赖）** |
| Viz / GPU 合成（OSR patch 涉及） | 第 10 章 |

> 本章默认你已读完第 13 章。第 13 章解释了 `ElectronBrowserClient`、`NodeBindings`、
> `ElectronRendererClient` 这些**类**做什么；本章解释这些类背后的**代码**是怎么进入 Chromium
> 构建图的、以及为了让它们工作 Electron 在 Chromium 本体上动了哪些刀。

---

## 1  设计考古：为什么 Electron 必须是「patch + 跟版」而不是「fork」

### 1.1  embedding 的三条路，Electron 选了最痛但最可控的一条

第 13 章已交代 Electron 的出身（2013 Atom Shell → 2014 改名）。这里补一个 Chromium-域的关键
决策：**如何与 Chromium 源码共存**。历史上 embed Chromium 有三种姿势：

| 姿势 | 代表 | 与 Chromium 源码的关系 | 致命问题 |
|------|------|----------------------|---------|
| **API 封装层**（不碰源码） | CEF 早期、WebView2 | 只用公开 Content API + 一层 C API | 公开 API 覆盖不了的能力（raw headers、OSR、自定义 scheme 走 SW）做不了 |
| **永久 fork**（独立维护） | 一些魔改浏览器 | 把 Chromium 整个 fork 出来改 | 跟不上上游安全补丁；每次 rebase 是几万行冲突 |
| **patch + 跟版**（改源码但不持有源码） | **Electron** | 源码仍由 `gclient` 从 googlesource 拉取，改动以 patch 重放 | patch 会随上游漂移而冲突；升级成本高 |

Electron 选第三条的根本原因：它**既需要改 Chromium 内部**（API 封装层不够），**又必须吃到
Chromium 的每一个安全补丁**（不能 fork 后落后）。patch + 跟版是这两个矛盾约束下唯一的解。

官方把这个取舍写进了 patch 文档，措辞很重【已 WebFetch 核实 electronjs.org/docs/latest/development/patches】：

> *"Every patch in Electron is a maintenance burden."*（每一个 patch 都是维护负担。）
> *"We should aim to upstream changes whenever we can, and avoid indefinite-lifetime patches."*

这句话是整个本章的「第一性原理」：**patch 越多，跟版越痛；所以 Electron 的工程文化是疯狂
upstream、能不打就不打**。下面所有机制都是围绕「如何让 patch 这件痛事可管理」展开的。

### 1.2  跟版节奏：被 Chromium 的发版日历绑架

Electron 的 major 版本号不是自己定的，是被 Chromium 拖着走的。考古其发版策略演进
【已 WebFetch 核实 electronjs.org/blog/12-week-cadence】：

> *"Every 6 weeks, a new Chromium release comes out with new features, bug fixes / security fixes,
> and V8 improvements."*
> *"Simply put, Chromium doesn't stop shipping so Electron is not going to slow down either."*

| 时间 | Chromium 节奏 | Electron 节奏 | 绑定关系 |
|------|--------------|--------------|---------|
| 2019（v6） | 6 周一个 milestone | 12 周一个 major（隔一个 Chromium 周期跟一次） | *"Electron v6.0.0 will include M76 ... the same release day as Chromium M76."* |
| 2021 Q3+ | **提速到 4 周** | 跟着提速到 **8 周一个 major** | *"In Q3 2021, the Chrome team increased their release cadence ... Electron's releases have followed suit."* |

支持窗口：**只维护最新 3 个 major**（latest three major versions）。一个 major EOL 后不再回合
安全补丁。对生产意味着：**你最多落后上游约 24 周（3×8）就必须升级，否则裸奔在已知 CVE 上**。

**这条考古的工程含义**：Electron 的「跟版痛」不是工程没做好，是商业现实——Chromium 每 4 周
塞进来的安全补丁是必须吃的，吃就得 rebase 全部 patch。第 5 节会看到这个 rebase 的真实代价。

### 1.3  从源码 build 的史料：depot_tools 与 gclient 的由来

Electron build 依赖 Chromium 的 `depot_tools`（`gclient`/`gn`/`ninja`），不是因为偷懒，而是
因为 **Chromium 的依赖图根本无法用 npm/cargo 这类包管理表达**：它有上千个 git 子仓、CIPD
二进制、平台 toolchain，只有 `gclient`（基于 `DEPS` 文件的递归 solution 解析器）能拉全。
Electron 把自己伪装成「Chromium 的一个 unmanaged solution」塞进 gclient——这是第 7 节
build demo 的核心机制。

---

## 2  整体架构：构建时的数据流

运行时架构见第 13 章。本章关心的是**构建时**这条流水线——从「拉源码」到「出 dist」：

```
                       ┌────────────────────────────────────────────────┐
  depot_tools          │  gclient（读 .gclient → 读 electron/DEPS）       │
  (gclient/gn/ninja)   │                                                 │
        │              │  DEPS: chromium_version = 151.0.7873.0          │
        ▼              │        node_version    = v24.16.0               │
  .gclient solution ──▶│  → 从 chromium.googlesource.com 拉 src@151...   │
   name "src/electron" │  → 把 electron 放到 src/electron                 │
   unmanaged           │  → 拉 node 到 src/third_party/electron_node      │
        │              └───────────────────────┬────────────────────────┘
        │                                       │  gclient runhooks
        │                                       ▼
        │              ┌────────────────────────────────────────────────┐
        │              │  hook: patch_chromium                           │
        │              │   python3 .../apply_all_patches.py config.json  │
        │              │                                                 │
        │              │  config.json: 13 个 (patch_dir → repo) 映射      │
        │              │   patches/chromium     → src                    │
        │              │   patches/v8           → src/v8                 │
        │              │   patches/node         → src/third_party/...    │
        │              │   ...                                           │
        │              │  每个目录: 读 .patches manifest → git am --3way  │
        │              └───────────────────────┬────────────────────────┘
        │                                       │  源码 = Chromium + Electron patch
        ▼                                       ▼
  CHROMIUM_BUILDTOOLS_PATH         ┌────────────────────────────────────┐
        │                          │  gn gen out/Release                 │
        └─────────────────────────▶│   --args='import("//electron/       │
                                   │            build/args/release.gn")' │
                                   │                                     │
                                   │  release.gn → import all.gn:        │
                                   │   is_electron_build = true          │
                                   │   root_extra_deps = ["//electron"]  │
                                   │   node_module_version = 148         │
                                   │   is_official_build = true          │
                                   └──────────────┬──────────────────────┘
                                                  │  ninja -C out/Release electron
                                                  ▼
                                   ┌────────────────────────────────────┐
                                   │  electron_lib (source_set)          │
                                   │   deps: //content/public/browser,   │
                                   │         //v8, electron_node:libnode,│
                                   │         chromium_src:chrome ...      │
                                   │  → electron_app / electron_framework│
                                   │  → dist (Electron.app / electron)   │
                                   └─────────────────────────────────────┘
```

四个交付物，逐节拆：patch 体系（§3-4）、build 体系（§6-7）、版本治理（§5）、裁剪/native（§8）。

---

## 3  patch 体系（一）：映射与清单——`config.json` 与 `.patches`

### 3.1  config.json：13 个 patch 目录到子树的映射

这是整个 patch 体系的「路由表」。Electron 改的不只是 Chromium 本体，还有 V8、Node、
BoringSSL、FFmpeg 等十几个组件，每个组件是一个独立 git repo，patch 必须打到对应 repo。

【真实源码 electron/electron@patches/config.json（已 WebFetch 核实，逐字）】

```json
[
  { "patch_dir": "src/electron/patches/chromium",         "repo": "src" },
  { "patch_dir": "src/electron/patches/boringssl",        "repo": "src/third_party/boringssl/src" },
  { "patch_dir": "src/electron/patches/devtools_frontend","repo": "src/third_party/devtools-frontend/src" },
  { "patch_dir": "src/electron/patches/ffmpeg",           "repo": "src/third_party/ffmpeg" },
  { "patch_dir": "src/electron/patches/v8",               "repo": "src/v8" },
  { "patch_dir": "src/electron/patches/node",             "repo": "src/third_party/electron_node" },
  { "patch_dir": "src/electron/patches/nan",              "repo": "src/third_party/nan" },
  { "patch_dir": "src/electron/patches/perfetto",         "repo": "src/third_party/perfetto" },
  { "patch_dir": "src/electron/patches/squirrel.mac",     "repo": "src/third_party/squirrel.mac" },
  { "patch_dir": "src/electron/patches/ReactiveObjC",     "repo": "src/third_party/squirrel.mac/vendor/ReactiveObjC" },
  { "patch_dir": "src/electron/patches/webrtc",           "repo": "src/third_party/webrtc" },
  { "patch_dir": "src/electron/patches/reclient-configs", "repo": "src/third_party/engflow-reclient-configs" },
  { "patch_dir": "src/electron/patches/sqlite",           "repo": "src/third_party/sqlite/src" }
]
```

**读这张表能学到的东西**：

1. **`node` 的 repo 是 `src/third_party/electron_node`，不是 upstream node**。Electron 维护
   一个 Node 的 fork（vendored 进 Chromium 树），因为 Node 默认链 V8 的某个版本，而 Electron
   必须让 Node 用 **Chromium 自带的那个 V8**（否则一个进程里两份 V8 直接爆炸）。这一条是
   Electron 最深的耦合之一——Node 与 Chromium 共享同一个 V8 是通过 patch + build 强行实现的。
2. **BoringSSL 单独一个 patch 目录**：Node 默认用 OpenSSL，Chromium 用 BoringSSL，符号冲突
   （第 13 章 1.1 节提到的 2016 年 shared library 决策的根源）就在这里靠 patch 弥合。
3. **DevTools frontend、FFmpeg、WebRTC** 都有 patch：说明「裁剪/定制」不止 Chromium 本体。

### 3.2  .patches manifest：apply 顺序的真相

每个 patch 目录下有一个 `.patches` 文件，**逐行列出该目录 patch 的 apply 顺序**。为什么需要
顺序？因为 patch 是 `git am` 串行重放成 commit 的，后一个 patch 的 context 行号依赖前一个
patch 已经 apply 完的状态。乱序 = context 对不上 = apply 失败。

`patches/chromium/.patches` 里能读到的真实条目（部分，已 WebFetch 核实，逐字文件名）：

```
build_gn.patch
boringssl_build_gn.patch
build_add_electron_tracing_category.patch
build_libc_as_static_library.patch
build_make_libcxx_abi_unstable_false_for_electron.patch
build_allow_electron_to_use_exec_script.patch
build_allow_electron_mojom_interfaces_to_depend_on_blink.patch
...
can_create_window.patch
support_mixed_sandbox_with_zygote.patch
allow_new_privileges_in_unsandboxed_child_processes.patch
printing.patch
fix_properly_honor_printing_page_ranges.patch
worker_context_will_destroy.patch
feat_plumb_node_integration_in_worker_through_workersettings.patch
...
```

> 注意命名约定本身就是文档：`build_*`（让 Electron 能编译/把 //electron 挂进构建图）、
> `feat_*`（新增能力，第 13 章的 raw headers / OSR 都在这类）、`fix_*`（修上游 bug，通常是
> 临时 patch 等 upstream merge）、`chore_*`、`worker_*`。**前缀直接对应 §4.2 的三类 patch
> 分类**。

### 3.3  为什么用 manifest 而不是给 patch 编号

老式做法是 `0001-xxx.patch`、`0002-xxx.patch` 用数字排序。Electron 放弃数字编号，改用
`.patches` 文本清单，官方解释【已 WebFetch 核实 docs/development/patches】：

> *"The `.patches` manifest approach prevents numbering conflicts across parallel PRs."*

含义：两个 PR 同时新增 patch，如果用数字，都想叫 `0042`，merge 冲突；用「文件名 + 清单追加」
则各自追加一行到清单末尾，冲突面小得多。**这是一个被规模逼出来的工程决策**——Electron 的
Chromium patch 有几百个，几十个 PR 并行飞，编号方案会被并发杀死。

---

## 4  patch 体系（二）：一个真实 patch 的逐行解剖

光看清单是抽象的。我们精读两个**真实存在**的 patch，体会 patch 的解剖结构和它承载的设计信息。

### 4.1  patch 文件的物理结构

一个 Electron patch 文件 = **一封 git format-patch 邮件** = `From/Date/Subject` 头 +
**人写的 rationale 正文** + `diff` hunks。rationale 正文是 Electron patch 文化的灵魂——官方
强制要求每个 patch 解释「为什么存在」。

### 4.2  实例一：`can_create_window.patch`——一个「坏掉但还在」的 patch

【真实源码 electron/electron@patches/chromium/can_create_window.patch（已 WebFetch 核实头部与文件清单）】

```
From: Cheng Zhao <zcbenz@gmail.com>
Date: Thu, 20 Sep 2018 17:45:32 -0700
Subject: can_create_window.patch

This adds a hook to the window creation flow so that Electron can intercede
and potentially prevent a window from being created.

TODO(loc): this patch is currently broken.
```

它改了 **12 个 Chromium 文件**（逐字路径，已核实）：

```
content/browser/renderer_host/render_frame_host_impl.cc
content/browser/web_contents/web_contents_impl.cc
content/common/frame.mojom                                ← 改了 Mojo 接口定义
content/public/browser/content_browser_client.cc
content/public/browser/content_browser_client.h           ← 第 13 章的 CanCreateWindow override 靠它
content/public/browser/web_contents_delegate.cc
content/public/browser/web_contents_delegate.h
content/renderer/render_frame_impl.cc
content/web_test/browser/web_test_content_browser_client.cc
content/web_test/browser/web_test_content_browser_client.h
third_party/blink/public/web/web_window_features.h
third_party/blink/renderer/core/frame/local_dom_window.cc
```

**这个 patch 教会我们三件事**：

1. **patch 会改 `.mojom`**：改了 `content/common/frame.mojom` 意味着 patch 不只是改 C++ 逻辑，
   还改了 **IPC 接口定义**，会触发 Mojo 代码重新生成。这是为什么 patch 冲突时 mojom 改动最难
   rebase——上游也在频繁动 mojom（第 5 节坑点）。
2. **跨越 browser + renderer + blink 三层**：一个看似简单的「拦截窗口创建」需要从 Blink 的
   `local_dom_window.cc`（`window.open` 发起点）一路 plumb 到 `content_browser_client.h`
   （Electron override 点），中间穿过 frame.mojom。**这就是为什么 Content API 不够、必须打
   patch**——公开 API 在这条链路上没有暴露足够的 hook。
3. **`TODO(loc): this patch is currently broken` 是真的**：一个标注「当前是坏的」的 patch
   仍然留在主干清单里。这不是疏忽，而是 patch 体系的现实——某些 patch 处于「上游重构后半残、
   等人修」的状态，但删掉会丢失历史意图，于是带着 TODO 活着。**生产启示：不要假设每个
   upstream patch 都处于完美工作态，读 patch 正文的 TODO/FIXME 是排查诡异行为的一手线索。**

### 4.3  实例二：`support_mixed_sandbox_with_zygote.patch`——rationale 写满设计意图

【真实源码 electron/electron@patches/chromium/support_mixed_sandbox_with_zygote.patch（已 WebFetch 核实）】

```
From: Jeremy Apthorp <nornagon@nornagon.net>
Date: Wed, 28 Nov 2018 13:20:27 -0800
Subject: support_mixed_sandbox_with_zygote.patch
```

rationale 正文（核实要点，非逐字全文）：Linux 上 Chromium 用 **zygote** 进程预初始化 sandbox
能力再 fork renderer；原生 `--no-zygote` 开关会**整体**关掉 zygote，连带关掉**所有** sandbox。
而 Electron 需要 **mixed sandbox**——某些 renderer（开了 `nodeIntegration` 的）必须不沙盒，
其余 renderer 仍要沙盒。这个 patch 把 `--no-zygote` 从「全局开关」改成「**逐进程、在 launch
前一刻按目标进程的命令行决定走不走 zygote**」（case-by-case, checking immediately prior to
launch）。作者明确写：**这个 patch 理论上可 upstream，但会触及 security-sensitive 代码，需要
security team review。**

改的文件（逐字，已核实）：

```
content/browser/renderer_host/render_process_host_impl.cc
content/browser/renderer_host/renderer_sandboxed_process_launcher_delegate.cc
content/browser/renderer_host/renderer_sandboxed_process_launcher_delegate.h
```

新增了一个 `use_zygote_` 成员，由 `--no-zygote` switch 控制。

**这个 patch 是「functional patch」的范本**（§4.4 分类）：它做的事 upstream **不会接受成默认
行为**（mixed sandbox 是 Electron 特有需求，纯 Chromium 浏览器不需要），所以注定是
indefinite-lifetime patch，每次升级都得 rebase。**它解释了第 13 章「坑 4：native addon 在
renderer 崩溃」和沙盒相关的诸多行为差异的源头**——Electron 的沙盒模型是被这个 patch 改写过的，
不等于纯 Chromium 的沙盒模型。

### 4.4  patch 三分类：决定一个 patch 该不该存在

官方把 patch 强制归入三类，每类有不同的 commit message 要求【已 WebFetch 核实 docs/development/patches】：

| 类别 | 定义 | commit message 要求 | 例子 |
|------|------|--------------------|------|
| **Temporary（临时）** | 准备 upstream 或终将删除 | **必须**附 upstream PR 链接 / 移除条件 | `fix_*` 修上游 bug，等 merge 后删 |
| **Compilation（编译）** | 让 Electron 环境能编译，无法 upstream | **必须**解释为什么不能用 subclass / 拷代码替代 | `build_gn.patch` 把 //electron 挂进构建图 |
| **Functional（功能）** | Electron 特有、与 upstream 根本不兼容 | 解释为什么 upstream 不会接受 | `support_mixed_sandbox_with_zygote.patch` |

官方原话：*"Every patch must include a commit message describing why it exists."* 这不是风格要求，
是**维护刚需**——升级时 rebase 一个 patch 冲突，维护者第一件事是读 commit message 判断「这个
patch 还需不需要、上游是不是已经做了同样的事」。**没有 rationale 的 patch 在升级时等于一颗
无法拆除的雷。**

---

## 5  版本治理：升级即 patch rebase 战争

### 5.1  单点 pin：DEPS 里的 chromium_version

Electron 跟哪个 Chromium，由 `DEPS` 文件里**一行**决定。

【真实源码 electron/electron@DEPS（已 WebFetch 核实，逐字值）】

```python
'chromium_version': '151.0.7873.0',
'node_version': 'v24.16.0',
# ...
'nan_version': '675cefebca42410733da8a454c8d9391fcebfbc2',
'squirrel.mac_version': '8d808803bc89ec0e2aa1450474856dfee3b00c6b',
'engflow_reclient_configs_version': '955335c30a752e9ef7bff375baab5e0819b6c00d',
```

deps 段把它拼成 git URL（已核实）：

```python
'src': {
    'url': Var("chromium_git") + '/chromium/src.git@' + Var("chromium_version"),
    # → https://chromium.googlesource.com/chromium/src.git@151.0.7873.0
}
```

并注册 patch hook（已核实）：

```python
hooks = [
  {
    'name': 'patch_chromium',
    'pattern': '.',
    'action': ['python3', 'src/electron/script/apply_all_patches.py',
               'src/electron/patches/config.json'],
  },
  # ...
]
```

**这条机制的全部威力与全部痛苦都在「单点 pin + hook 自动重放」**：改一行 `chromium_version`
→ `gclient sync` 拉新 Chromium → hook 自动 `apply_all_patches.py` → 几百个 patch 撞上一个
已经变了的 Chromium 树。能 clean apply 的皆大欢喜，撞了的就是下面的战争。

> ⚠ **note**：我读到的 DEPS 是 `main` 分支当前值（Chromium 151 / Node v24.16）。这个数字几乎
> 每天变。你看到本章时，去 `electron/electron@main/DEPS` 复核当前 pin。

### 5.2  升级的真实代价：patch rebase

升级流程的本质（结合 §1.2 的发版节奏）：

```
上游 M150 → M151，Chromium 改了 N 个文件
        │
        ▼
bump chromium_version → gclient sync → apply_all_patches.py
        │
        ├── git am --3way 能自动 3way merge 的 patch ── 静默通过
        │
        └── context 彻底对不上 / 被改的代码被上游删了 ── apply 失败
                │
                ▼
        人肉 rebase：git import-patches 到失败那个 commit
        → 手动改 → git export-patches 重写 patch 文件
        → 更新 .patches manifest
```

`git am` 用了 `--3way`（见 §6.2），所以**很多** patch 能靠三方合并自动过。但三类 patch 里
**Functional 与改 mojom 的 patch 最容易爆**：

- **改 `.mojom` 的 patch**（如 §4.2 的 can_create_window）：上游频繁重构 mojom 接口，一旦
  接口签名变了，3way 也救不了。
- **改 V8 / Node 桥接的 patch**：V8 每个 milestone 大改，Node 与 V8 的共享 patch 是重灾区。
- **sandbox 相关 patch**（§4.3）：security 代码上游变动谨慎但一变就全链路影响。

### 5.3  生产真坑：升级后「明明没动代码，功能却坏了」

**现象**：把 Electron 从 vN 升到 vN+1，自己代码一行没改，`session.webRequest.onHeadersReceived`
拿不到 raw headers / OSR 黑屏 / 自定义 scheme 在 Service Worker 里 404。
**根因**：这些能力**全部由 patch 提供**（第 13 章 §7 的 patch 表）。升级时对应 patch 在新
Chromium 上的语义被上游变动稀释了——patch 可能「apply 成功了，但上游把它依赖的回调改了行为」。
`git am --3way` 只保证**文本合并成功**，不保证**语义正确**。
**排查路径**：
1. 去 `electron/electron@v{N+1}/patches/chromium/` 找对应 patch（如
   `feat_expose_raw_response_headers_from_urlloader.patch`）。
2. 读它改的 Chromium 文件，在 Chromium Code Search 上对比 M{N} vs M{N+1} 该文件的 upstream diff。
3. 看 Electron 该版本 release notes 的「Breaking Changes / Chromium」段。
**根本缓解**：**不要在生产里依赖未文档化的 patch 副作用**；依赖的能力优先走 Electron 官方
API（它会随版本维护 patch），少直接戳被 patch 暴露的底层。

---

## 6  build 体系（一）：GN 决策的集中地——`build/args/*.gn`

### 6.1  三层 GN args：testing → release → all

Electron 把所有 GN 决策分层。最终 build 命令只 import 一个文件，它再 import 共享层。

```
build/args/release.gn   ← release 专属（official build / PGO）
        │ import
        ▼
build/args/all.gn       ← 所有 Electron build 共享的核心决策
        │ (testing.gn 同样 import all.gn)
```

【真实源码 electron/electron@build/args/release.gn（已 WebFetch 核实，逐字）】

```gn
import("//electron/build/args/all.gn")
is_component_build = false
is_official_build = true
is_component_ffmpeg = true
v8_builtins_profiling_log_file = "//electron/build/pgo_profiles/electron-v8-builtins.profile"
```

逐行解：

- `import("//electron/build/args/all.gn")` —— 共享决策全在 all.gn。
- `is_component_build = false` —— release **静态链接**成大 binary（component build 是开发期把
  每个 component 编成单独 .so，启动快但产物碎、不可分发）。
- `is_official_build = true` —— 开启全量优化（LTO、符号裁剪），这也是为什么 release 编译极慢。
- `is_component_ffmpeg = true` —— **FFmpeg 单独编成动态库**。这是个精妙裁剪点：把 FFmpeg
  抽成 .so，使得「无专利 codec 的开源 FFmpeg」可以被「带 H.264/AAC 的专利版」替换而不重编整个
  Electron（§8.2 codec 裁剪的基础设施）。
- `v8_builtins_profiling_log_file` —— PGO（profile-guided optimization）档案，用真实 V8
  builtin 调用画像优化代码布局，提升启动/执行性能。

### 6.2  all.gn：Electron 的「身份证」

这是 Electron 改造 Chromium 构建图的核心文件——它告诉 Chromium 的 GN「我不是纯 Chromium，
我要把 //electron 编进来」。

【真实源码 electron/electron@build/args/all.gn（已 WebFetch 核实，逐项）】

```gn
is_electron_build = true                 # 全局身份标记，下游 BUILD.gn 用它做条件编译
root_extra_deps = [ "//electron" ]       # ★把 //electron 挂进 Chromium 根构建图——一切的根

# Node / V8 共享层（呼应 §3.1：Node 必须用 Chromium 的 V8）
node_module_version = 148                 # ★Node ABI 版本，native 模块编译的锚（见 §8.3）
v8_promise_internal_field_count = 1       # 给 Node 的 async_hooks 预留 V8 内部字段
v8_embedder_string = "-electron.0"        # V8 版本串带 electron 后缀
v8_enable_javascript_promise_hooks = true # Node async_hooks 依赖
v8_expose_public_symbols = true           # 让 Node addon 能链到 V8 符号

# 安全/兼容 flag——为了让 Node 共存而关掉的 Chromium 强化
enable_cet_shadow_stack = false           # CET shadow stack 与 V8 JIT 冲突
is_cfi = false                            # CFI 与 Node native 调用不兼容
enable_dangling_raw_ptr_checks = false

# 体积/平台裁剪
enable_pseudolocales = false
enable_linux_installer = false
enable_pdf_save_to_drive = false
enable_resource_allowlist_generation = true   # 配合 build_*.patch 生成资源白名单做裁剪
```

**`root_extra_deps = ["//electron"]` 是整个 build 体系的枢纽**：Chromium 的根 BUILD.gn 本来
不认识 electron，这一行把 `//electron:electron` 这个 target 注入根依赖，于是
`ninja -C out/Release electron` 才能找到目标。**而 `//electron` 这个 GN label 能被解析，靠的是
`build_gn.patch`**（§4.4 表里的 Compilation patch）——patch 体系和 build 体系在这里交汇：
**没有 build_gn.patch 把 electron 接入 Chromium 的 GN 文件，all.gn 里的 root_extra_deps 就会
报「找不到 //electron」**。

> **关键认知**：`is_electron_build` / `node_module_version` / 那一堆 `*=false` 的安全 flag——
> 这些不是「配置」，是 **Electron 与 Chromium 安全模型的差异清单**。每关一个 Chromium 强化
> （CFI、CET shadow stack），都是为了让 Node 的 native 代码能跑而付出的安全代价。**这解释了
> 为什么「Electron app 的攻击面比纯 Chromium 浏览器大」不是 app 写得烂，而是构建层的先天取舍。**

### 6.3  BUILD.gn:`//electron` target 长什么样

【真实源码 electron/electron@BUILD.gn（已 WebFetch 核实结构与关键 deps）】

```gn
# 核心库：Electron 的全部 C++（第 13 章那些 ElectronBrowserClient 等都在这里编）
source_set("electron_lib") {
  deps = [
    "//content/public/browser", "//content/public/child",
    "//content/public/gpu",     "//content/public/renderer",
    "//content/public/utility",                          # ← Content API（embedding 层）
    "//v8", "//v8:v8_libplatform",                       # ← V8
    "//third_party/electron_node:libnode",               # ← Node（vendored fork，§3.1）
    "//third_party/electron_node:node_snapshot",
    "chromium_src:chrome", "chromium_src:chrome_spellchecker",
    ":electron_fuses", ":electron_js2c", ":resources",
  ]
  sources = filenames.lib_sources                        # 源文件清单从 filenames.gni 来
  if (is_win)   { sources += filenames.lib_sources_win }
  if (is_mac)   { sources += filenames.lib_sources_mac }
  if (is_linux) { sources += filenames.lib_sources_linux }
}

# macOS：framework bundle（区别于 Win/Linux 的单 executable）
mac_framework_bundle("electron_framework") {
  output_name = "$electron_product_name Framework"
  public_deps = [ ":electron_lib", ":electron_framework_libraries" ]
  # 打包 resources / helper 子进程
}

# 顶层分发 target
group("electron") {
  public_deps = [ ":electron_app" ]   # → Win/Linux executable 或 mac app bundle
}
```

**读这个 target 学到**：

1. `electron_lib` 把 `//content/public/*`、`//v8`、`electron_node:libnode`、`chromium_src:chrome`
   全部 `deps` 进来——**Electron 是把 Content API + V8 + Node + 部分 chrome 层「拼装」成一个
   source_set**，第 13 章讲的所有运行时类就编在这个 lib 里。
2. macOS 走 `mac_framework_bundle`（Electron.app/Contents/Frameworks/Electron Framework.framework），
   Win/Linux 走单 executable——**这就是为什么 macOS 的 Electron 目录结构和 Win/Linux 截然不同**，
   也是 §8.4 裁剪打包时平台差异的根源。
3. `sources = filenames.lib_sources`：源文件清单抽到独立 `.gni`，**升级时新增/删除源文件改
   filenames.gni 即可，不动 BUILD.gn 主体**——又一个为可维护性做的工程拆分。

### 6.4  方案对比：GN vs 传统 build 系统（对 Electron 改造者的影响）

| 维度 | GN（Chromium/Electron） | CMake（CEF embedding 常见） | npm/node-gyp（纯 JS app 想象） |
|------|------------------------|---------------------------|------------------------------|
| 依赖解析 | gclient + DEPS 递归 solution | find_package / submodule | package.json |
| 增量构建 | ninja，极快 | make/ninja | 不适用 |
| 改造 Chromium 的姿势 | `root_extra_deps` + patch GN 文件 | 链接预编译 CEF 库 | 不可能 |
| 能改 Chromium 内部吗 | **能**（patch + 重编） | 否（只用公开库） | 否 |
| 全量编译耗时 | 数十分钟~数小时（official build） | 分钟级（只编自己） | 秒级 |
| 适用边界 | 需要改 Chromium 内部 / 跟版 | 只用公开 Content/CEF API | 完全不碰内核 |

**不适用边界**：如果你的需求**纯粹**用 Electron 公开 API 能满足（绝大多数 app），**永远不要
自己 build Electron**——直接 `npm i electron` 用官方预编译版。自己 build 的唯一正当理由是：
你必须新增/修改 Chromium patch（如自研 codec、特殊 OSR、定制网络栈），那才进入本章的 build 地狱。

---

## 7  ⭐ Demo：patch 体系走读 + build 配置精读 + 真实观测

> Chromium 全量 build 需要数十 GB 磁盘、数小时编译，本地不现实。本节遵循「能真跑的真跑、
> 偏 build 的给可复核的走读 + 最小补丁」原则。Demo 1-3 **可在装了 git/python/node 的机器直接
> 跑**；Demo 4-5 是 build 体系走读（给完整可复核命令，真 build 选做）；Demo 6 是**最小补丁
> 制作全流程**（patch 工具链可在任意 git repo 上演示，不需要编 Chromium）。

### Demo 1：抓取并统计 Electron 真实 patch 体系（可运行）

无需 build，直接用 git/gh 把 patch 体系拉到本地分析。

```bash
# 浅克隆 Electron（只取 patch 目录所在的 tree，不取全历史）
git clone --depth 1 https://github.com/electron/electron.git /tmp/electron-src
cd /tmp/electron-src

# 1) Chromium patch 总数（衡量「维护负担」的直接指标）
echo "chromium patches:"; ls patches/chromium/*.patch | wc -l

# 2) 各 patch 目录的 patch 数（对照 §3.1 的 13 个目录）
for d in patches/*/; do
  n=$(ls "$d"*.patch 2>/dev/null | wc -l | tr -d ' ')
  echo "$n  $d"
done | sort -rn

# 3) 按前缀分类统计（对照 §4.4 三分类）
echo "--- by prefix ---"
ls patches/chromium/ | grep '\.patch$' \
  | sed -E 's/^(build|feat|fix|chore|worker)_.*/\1/' \
  | grep -E '^(build|feat|fix|chore|worker)$' \
  | sort | uniq -c | sort -rn
```

**预期输出（形态，具体数随版本变）**：

```
chromium patches: 180        # 量级在一两百，逐版增减
80   patches/chromium/       # Chromium 本体最多
20   patches/node/
12   patches/v8/
 6   patches/boringssl/
...
--- by prefix ---
  45 feat                    # 新增能力占大头
  38 build                   # 让 Electron 能编译
  25 fix                     # 临时修上游 bug
  12 chore
   8 worker
```

**与源码呼应**：`feat_*` 多 → Electron 主要在「补 Content API 的能力缺口」；`build_*` 多 →
「让 //electron 接入 Chromium 构建图」本身就需要大量 patch（§6.2 的 root_extra_deps 只是冰山一角）。

### Demo 2：读懂一个真实 patch 的解剖结构（可运行）

```bash
cd /tmp/electron-src
# 看 can_create_window.patch 的头部 rationale + 改了哪些文件
echo "=== rationale (commit message) ==="
grep -m1 -A8 '^Subject:' patches/chromium/can_create_window.patch | head -12

echo "=== files touched ==="
grep '^diff --git' patches/chromium/can_create_window.patch \
  | sed -E 's#^diff --git a/(.*) b/.*#\1#'

echo "=== 改了几个 .mojom（升级最易冲突的信号）==="
grep '^diff --git' patches/chromium/can_create_window.patch | grep -c '\.mojom'
```

**预期输出**：

```
=== rationale (commit message) ===
Subject: can_create_window.patch

This adds a hook to the window creation flow so that Electron can intercede
and potentially prevent a window from being created.

TODO(loc): this patch is currently broken.        ← §4.2 那个真实 TODO
=== files touched ===
content/browser/renderer_host/render_frame_host_impl.cc
content/common/frame.mojom
content/public/browser/content_browser_client.h
... (共 12 个)
=== 改了几个 .mojom ===
1                                                  ← 改了 mojom，升级高危
```

**学到**：用 `grep '\.mojom'` 命中数当「升级风险计」——改 mojom 的 patch 在跟版时最易爆（§5.2）。

### Demo 3：用 CDP 把第 13 章的 isolated world 与本章 patch 效果连起来（可运行）

本 demo 验证「`can_create_window` patch 改写的 `window.open` 行为」+「contextIsolation world」，
把构建层 patch 与运行时表现对上。前置见第 13 章 Demo 1 的最小 app（`main.js`/`preload.js`）。

```bash
# 复用第 13 章 electron-demo 目录
cd electron-demo
npm i -D chrome-remote-interface
npx electron main.js --remote-debugging-port=9229 &
sleep 2
```

```javascript
// inspect-worlds.js —— 观测 world 数量 + window.open 拦截（can_create_window patch 的运行时面）
const CDP = require('chrome-remote-interface')
;(async () => {
  const client = await CDP({ port: 9229 })
  const { Runtime, Page } = client
  await Runtime.enable(); await Page.enable()

  const worlds = []
  Runtime.executionContextCreated(({ context }) => worlds.push(context.name || '(main world)'))
  await new Promise(r => setTimeout(r, 500))
  console.log('execution contexts:', worlds)
  // 期望: ['(main world)', 'Electron Isolated Context']  ← contextIsolation 生效（第 13 章）

  // 触发 window.open —— 主进程的 ElectronBrowserClient::CanCreateWindow 会被调用
  // （正是 can_create_window.patch 在 content_browser_client.h 暴露的 hook）
  await Runtime.evaluate({ expression: `window.open('about:blank')` })
  console.log('window.open dispatched → 看主进程是否触发 "new-window"/setWindowOpenHandler')

  await client.close()
})().catch(console.error)
```

```bash
node inspect-worlds.js
kill %1   # 关掉后台 electron
```

**预期**：`execution contexts: ['(main world)', 'Electron Isolated Context']`；主进程若设了
`webContents.setWindowOpenHandler` 会拦到这次 open——**这就是 §4.2 那个 patch 在
`content_browser_client.h` 加的 hook 在运行时的可观测投影**。构建层的 patch ↔ 运行时的行为，
在这里闭环。

### Demo 4：build 配置精读——不 build 也能复核 GN 决策（可运行）

```bash
cd /tmp/electron-src
echo "=== release.gn（§6.1）==="
cat build/args/release.gn

echo "=== all.gn 里的身份与裁剪决策（§6.2）==="
grep -E 'is_electron_build|root_extra_deps|node_module_version|is_cfi|enable_cet|enable_pdf|enable_linux_installer' build/args/all.gn

echo "=== DEPS 单点 pin（§5.1）==="
grep -E "chromium_version|node_version" DEPS | head -4

echo "=== patch hook（§5.1）==="
grep -A3 'patch_chromium' DEPS
```

**预期**（对照前文逐字源码，值随版本变）：

```
=== release.gn ===
import("//electron/build/args/all.gn")
is_component_build = false
is_official_build = true
is_component_ffmpeg = true
v8_builtins_profiling_log_file = "//electron/build/pgo_profiles/electron-v8-builtins.profile"
=== all.gn ... ===
is_electron_build = true
root_extra_deps = [ "//electron" ]
node_module_version = 148
is_cfi = false
...
=== DEPS 单点 pin ===
  'chromium_version': '151.0.7873.0',
  'node_version': 'v24.16.0',
```

### Demo 5：从源码 build Electron——完整命令走读（真 build 选做）

下面命令**全部来自官方 build-instructions，可复核**【已 WebFetch 核实
electronjs.org/docs/latest/development/build-instructions-gn】。真跑需 ~100GB 磁盘 + 数小时。

```bash
# ① depot_tools（gclient/gn/ninja 工具集）
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
export PATH="$PWD/depot_tools:$PATH"

# ② 把 Electron 配成 gclient 的一个 unmanaged solution（§1.3 的伪装）
mkdir electron && cd electron
gclient config --name "src/electron" --unmanaged https://github.com/electron/electron

# ③ 同步：拉 Chromium(@DEPS pin) + Node + 所有子仓，并自动跑 patch_chromium hook
gclient sync --with_branch_heads --with_tags
#   ↑ 这一步内部就会执行 apply_all_patches.py（§5.1 hook），把全部 patch git am 上去

# ④ buildtools 路径
cd src
export CHROMIUM_BUILDTOOLS_PATH="$PWD/buildtools"

# ⑤ gn gen：只 import 一个 args 文件（§6.1 三层 args 的入口）
gn gen out/Release --args='import("//electron/build/args/release.gn")'

# ⑥ ninja 编译顶层 electron target（§6.3 的 group("electron")）
ninja -C out/Release electron
#   产物: out/Release/Electron.app (mac) | electron (Linux) | electron.exe (Win)
```

**与源码呼应的关键点**：
- 第 ② 步 `--name "src/electron" --unmanaged`：对应 §1.3——Electron 把自己注册成 gclient
  solution，gclient 才会去读 `electron/DEPS` 递归拉 Chromium。
- 第 ③ 步**自动触发 patch**：`gclient sync` 跑完 DEPS 里的 `patch_chromium` hook
  （§5.1），不需要手动 apply。这就是「跟版」自动化的落点。
- 第 ⑤ 步 args **只有一行 import**：所有复杂度被收进 `release.gn → all.gn`（§6），命令行干净。

### Demo 6：最小补丁全流程——在任意 git repo 上演示 patch 工具链（可运行）

不需要 Chromium。用 Electron 的 patch 工具同款机制（`git format-patch` / `git am`）演示
**「打补丁 → 导出 patch + 写 .patches manifest → clean checkout 重放」**全链路，等价于 §3-4 的
Electron patch 体系最小复现。

```bash
# 模拟「上游仓库」
rm -rf /tmp/upstream && mkdir /tmp/upstream && cd /tmp/upstream
git init -q && git config user.email a@b.c && git config user.name a
printf 'int main(){return 0;}\n' > app.cc
git add . && git commit -qm "upstream baseline"   # 这是「Chromium 原始状态」
git tag upstream-head

# ① 像 Electron 一样：在上游之上叠加一个 functional 改动，写满 rationale
sed -i.bak 's/return 0;/return 42;/' app.cc && rm -f app.cc.bak
git commit -qam "feat_change_exit_code.patch

This makes the app exit with 42 instead of 0 so the embedder can detect it.
Category: functional — upstream returns 0 by spec, this is embedder-specific.
No upstream PR (would not be accepted)."

# ② 导出 patch（等价 git-export-patches；Electron 用 --zero-commit --full-index --keep-subject）
mkdir -p patches
git format-patch upstream-head..HEAD \
  --keep-subject --no-stat --no-signature --zero-commit --full-index \
  -o patches
# 写 .patches manifest（apply 顺序，§3.2）
ls patches/*.patch | xargs -n1 basename > patches/.patches
echo "=== patches/.patches manifest ==="; cat patches/.patches
echo "=== 导出的 patch 头部（含 rationale，§4.1）==="; head -12 patches/*.patch

# ③ 模拟「gclient sync 后的 clean Chromium」：回到 baseline，重放 patch（等价 apply_all_patches）
git checkout -q upstream-head
git am --3way --keep-non-patch patches/$(head -1 patches/.patches)   # ★ §6.2 同款 flag
echo "=== 重放后 app.cc ==="; grep return app.cc
# 期望: return 42;  ← patch 在「干净上游」上成功重放，正是 Electron 每次 sync 做的事
```

**预期输出**：

```
=== patches/.patches manifest ===
0001-feat_change_exit_code.patch
=== 导出的 patch 头部（含 rationale）===
From 0000000000000000000000000000000000000000 ...
Subject: feat_change_exit_code.patch

This makes the app exit with 42 instead of 0 so the embedder can detect it.
Category: functional ...
=== 重放后 app.cc ===
  int main(){return 42;}
```

**这个 demo 把本章最核心的机制压成 30 行可跑脚本**：rationale-rich commit（§4.1/4.4）→
`format-patch` 导出（对应 `git.py::format_patch`，§下节源码）→ `.patches` manifest（§3.2）→
`git am --3way` 在 clean 上游重放（对应 `apply_all_patches.py`，§5.1）。**升级冲突的本质**也能
在这里复现：把 baseline 改成「上游也动了 return 那一行」，再 `git am --3way` 就会触发三方合并
甚至冲突——这正是 §5.2 的微缩模型。

```bash
# 升级冲突复现（选做）：让「上游」也改同一行 → 制造 patch 漂移
cd /tmp/upstream && git checkout -q master 2>/dev/null || git checkout -q main 2>/dev/null
# （在 baseline 上模拟上游变更后重放 patch，观察 git am 的 3way / 冲突行为）
```

---

## 8  裁剪、native 模块与打包（Chromium 域）

### 8.1  patch 工具链源码：`script/lib/git.py` 精读

Electron 的 patch 重放不是裸 `git am`，而是包了一层有性能与正确性考量的封装。精读其核心函数。

【真实源码 electron/electron@script/lib/git.py（已 WebFetch 核实，逐字关键片段）】

```python
def import_patches(repo, ref=UPSTREAM_HEAD, **kwargs):
  """same as am(), but we save the upstream HEAD so we can refer to it when we
  later export patches"""
  update_ref(repo=repo, ref=ref, newvalue='HEAD')          # ★保存上游 HEAD
  if ref != _LEGACY_UPSTREAM_HEAD:
    update_ref(repo=repo, ref=_LEGACY_UPSTREAM_HEAD, newvalue='HEAD')
  subprocess.call(
    ['git', '-C', repo, 'update-index', '--index-version', '4'],
    stderr=subprocess.DEVNULL)
  am(repo=repo, **kwargs)

def format_patch(repo, since):
  args = [
    'git', '-C', repo, '-c',
    'core.attributesfile=' + os.path.join(...,'electron.gitattributes'),
    '-c', 'diff.renames=true', '-c',
    'diff.electron.xfuncname=$^', 'format-patch',
    '--keep-subject', '--no-stat', '--stdout',
    '--no-signature', '--zero-commit', '--full-index', since   # ★稳定化输出
  ]
  return subprocess.check_output(args).decode('utf-8')

def split_patches(patch_data):
  """Split concatenated patch series into N separate patches"""
  patches = []
  patch_start = re.compile('^From [0-9a-f]+ ')               # 按 "From <sha> " 切分
  for line in patch_data.splitlines(keep_line_endings=True):
    if patch_start.match(line):
      patches.append([])
    patches[-1].append(line)
  return patches
```

apply 侧（`am()`）的 flag（已核实，逐字）：

```python
args = ['--keep-non-patch']
if threeway:   args += ['--3way']          # ★ 三方合并，缓解 patch 漂移
if directory:  args += ['--directory', directory]
if keep_cr:    args += ['--keep-cr']

root_args += [                              # ★ 大规模 am 的性能优化
  '-c', 'index.skipHash=true',
  '-c', 'index.version=4',
  '-c', 'core.fsync=none',
  '-c', 'gc.auto=0',
]
command = ['git'] + root_args + ['am'] + args
```

**逐项设计意图**：

- `--3way`：patch context 对不上时退回 blob 三方合并——**升级时大量 patch 能自动过的关键**（§5.2）。
- `--zero-commit` / `--full-index`（export 侧）：把 commit sha 归零、用全长 index——**让同一改动
  每次导出的 patch 文件字节一致**，否则 patch 文件天天因 sha 变化产生 noise diff。这是「让
  patch 可 review、git 历史干净」的工程细节。
- `index.skipHash=true` / `core.fsync=none` / `gc.auto=0`：apply 几百个 patch 时关掉 hash 校验、
  fsync、自动 gc——**纯性能**，把 patch 重放从分钟级压到秒级。
- `import_patches` 先 `update_ref(UPSTREAM_HEAD)`：记住「打 patch 前的上游 HEAD」，**export 时
  才知道从哪个点 `format-patch`**（`format_patch(repo, since=UPSTREAM_HEAD)`）。import/export
  对称依赖这个 ref。

> 顶层入口 `apply_all_patches.py` 用 `ThreadPoolExecutor` **并行**处理 13 个 patch 目录（roller
> 分支上改为串行以便冲突输出可读），每个目录调 `git.import_patches(threeway=THREEWAY,...)`
> 【已 WebFetch 核实】。

### 8.2  codec 裁剪：proprietary codecs 与 component ffmpeg

官方预编译 Electron **不带** H.264/AAC 等专利 codec（避免授权费）。需要专利 codec 的场景必须
自己 build，靠两个机制：

- `is_component_ffmpeg = true`（§6.1）：FFmpeg 编成独立 .so，可替换。
- GN args 里开 `proprietary_codecs=true` + `ffmpeg_branding="Chrome"`（指向带专利 codec 的
  FFmpeg 配置）——这两个是 Chromium 原生 GN arg，Electron build 时追加到 args。

> ⚠ 待核：`proprietary_codecs` / `ffmpeg_branding` 我未在本次 fetch 的 all.gn 片段里直接读到
> （它们由 build 命令额外追加，或在未读到的 args 行）。机制方向（component ffmpeg 可替换）已由
> §6.1 的 `is_component_ffmpeg=true` 逐字证实；codec 开关的精确 arg 名以 Chromium GN
> 文档为准，build 前复核。

### 8.3  native 模块的 ABI 锁：`node_module_version`

§6.2 读到 `node_module_version = 148`。这是 **Node 的 ABI（Module）版本号**，决定预编译 native
addon（`.node`）能否在该 Electron 上加载。

- 每个 Electron major 通常对应一个 `node_module_version`。
- 用 N-API（napi）编译的 addon 跨版本稳定（这也是第 13 章「坑 4」推荐 `napi_get_uv_event_loop`
  而非 `uv_default_loop()` 的同源逻辑——napi 是 Electron native 兼容的正道）。
- 用旧 nan/直接 V8 API 编译的 addon 绑死 `node_module_version`，**升级 Electron 必须用
  `electron-rebuild` 重编**，否则 `Error: The module was compiled against a different Node.js
  version`。

**生产坑**：CI 里 `npm i` 装的 native addon 是按**系统 Node** 的 ABI 编的，丢进 Electron
（不同 ABI）直接加载失败。**修复**：用 `@electron/rebuild`（按 Electron 的
`node_module_version` 重编），或全程用 N-API addon。

### 8.4  打包与裁剪的平台差异

§6.3 已揭示：macOS 是 framework bundle，Win/Linux 是单 executable。裁剪打包时：

- **ASAR**：app 代码打成 `.asar` 虚拟归档（第 13 章 §8.1 提过 `ElectronURLLoaderFactory` 对
  `.asar` 有特殊 VFS 支持）——这是「应用层裁剪」，不碴 Chromium。
- **locale 裁剪**：Chromium 自带上百个 `.pak` locale 文件，体积可观。`enable_pseudolocales=false`
  （§6.2 已读到）+ 删除不需要的 `locales/*.pak` 是常见瘦身手段。
- **不可裁的硬底**：Chromium 引擎本体 + V8 + Node 是 ~100MB 量级的硬底（第 13 章 §10 对比表），
  **这是 Electron 体积无法接近 Tauri 的根本原因**——你裁的是边角，砍不动引擎。

---

## 9  失败模式与生产真坑（构建/维护域）

> 运行时坑见第 13 章 §12。这里聚焦**构建时与维护时**的坑。

### 坑 1：`gclient sync` 后 patch apply 失败，卡在某个 commit

**现象**：bump chromium_version 后 `gclient sync`，hook `apply_all_patches.py` 报
`error: patch failed: content/common/frame.mojom`。
**根因**：上游改了该文件，patch context 漂移，连 `--3way` 都救不了（§5.2，改 mojom 的 patch 高危）。
**修复**：手动 `git import-patches` 到失败的 patch → 改 → `git export-patches` 重写 patch 文件 +
更新 `.patches`。读该 patch 的 rationale（§4.4）判断「上游是不是已经做了同样的事，能不能直接删」。

### 坑 2：自己加的 patch，别人 sync 后丢了

**现象**：本地改了 Chromium 源码生效，push 后 CI / 同事拉下来功能没了。
**根因**：直接改 `src/` 下的 Chromium 文件**不会被持久化**——`gclient sync` 会覆盖。改动必须
`git export-patches` 成 patch 文件并提交到 `patches/` + 追加 `.patches`。
**修复**：在 chromium src 里 commit 你的改动 → `e patches chromium`（或 `git export-patches`）
导出 → 把 `patches/chromium/your.patch` 和改动后的 `.patches` 一起提交。**记住：patch 文件
才是真相源，src/ 是临时工作区。**

### 坑 3：patch 文件天天产生无意义 diff

**现象**：明明没改逻辑，`patches/*.patch` 总有 diff（commit sha / index 变化）。
**根因**：没用 Electron 的 export 参数（`--zero-commit --full-index`，§8.1），导致每次导出 sha 不同。
**修复**：始终用 `git export-patches`（封装了稳定化参数），不要手搓 `git format-patch`。

### 坑 4：native addon 升级 Electron 后加载失败

**现象**：`Error: The module was compiled against a different Node.js version`。
**根因**：addon 绑死旧 `node_module_version`（§8.3），新 Electron 的 ABI 变了。
**修复**：`@electron/rebuild`，或迁移到 N-API。

### 坑 5：build 出来体积爆炸 / 编译 OOM

**现象**：official build 占满磁盘、链接阶段 OOM。
**根因**：`is_official_build=true` + `is_component_build=false`（§6.1）开了全量 LTO，内存/磁盘
峰值极高。
**缓解**：开发期用 `testing.gn`（component build，不开 official，编译快、产物碎但能调试）；
只有出 release 才用 `release.gn`。

### 坑 6：依赖了某个 patch 的副作用，升级后静默坏掉

见 §5.3。**根本对策**：把「依赖 patch 暴露的底层能力」收敛到 Electron 官方 API；自研 patch
写满 rationale + 在 CI 加针对该能力的回归测试，升级时第一时间发现语义漂移。

---

## 10  五件套

### 必读源码

| 文件 | 为什么重要 |
|------|-----------|
| `electron/electron@patches/config.json` | patch 路由表：13 个目录到 chromium/v8/node/... 子树的映射（§3.1） |
| `electron/electron@patches/chromium/.patches` | apply 顺序清单，命名前缀=patch 分类（§3.2/4.4） |
| `electron/electron@script/lib/git.py` | patch 工具链核心：`import_patches`/`format_patch`/`am` 的 flag 设计（§8.1） |
| `electron/electron@script/apply_all_patches.py` | 顶层重放入口，ThreadPoolExecutor 并行（§5.1/8.1） |
| `electron/electron@DEPS` | `chromium_version` 单点 pin + `patch_chromium` hook（§5.1） |
| `electron/electron@build/args/all.gn` | Electron 构建身份：`is_electron_build`/`root_extra_deps`/`node_module_version`（§6.2） |
| `electron/electron@build/args/release.gn` | release 决策：official/static/component ffmpeg（§6.1） |
| `electron/electron@BUILD.gn` | `electron_lib` 的 deps 拼装 + mac framework（§6.3） |
| `electron/electron@patches/chromium/can_create_window.patch` | 范本：改 mojom + 跨 browser/renderer/blink 三层 + 真实 TODO（§4.2） |
| `electron/electron@patches/chromium/support_mixed_sandbox_with_zygote.patch` | functional patch 范本，rationale 写满设计意图（§4.3） |

### 推荐工具链

```bash
# 1. 不 build 也能分析 patch 体系（Demo 1/2）
git clone --depth 1 https://github.com/electron/electron.git
ls patches/chromium/*.patch | wc -l

# 2. Electron 自带 patch 工具（在 electron src 内）
e patches chromium                 # = git export-patches，导出 chromium patch
e sync                             # = gclient sync（含 patch hook）
node script/export_all_patches.py patches/config.json   # 批量导出

# 3. 查当前 pin 的 Chromium / Node 版本
grep -E "chromium_version|node_version" DEPS
cat node_modules/electron/dist/version   # 已安装的官方版本

# 4. Chromium Code Search：对比 patch 改的文件在 upstream 的演进
# https://source.chromium.org/chromium/chromium/src/+/main:content/common/frame.mojom

# 5. native addon 跨版本重编
npx @electron/rebuild
```

### 常见误区

1. **"改 Electron 行为 = 改自己的 JS 代码"** —— 涉及 Chromium 内部能力（OSR/raw headers/自定义
   网络栈）必须打 **Chromium patch**，不是 JS 层能解决的（§4.2 的三层 plumb）。
2. **"直接改 src/ 下的 Chromium 文件就行"** —— 会被 `gclient sync` 覆盖，必须 export 成 patch
   提交（坑 2）。**patch 文件才是真相源。**
3. **"Electron 版本号是自己排的"** —— 被 Chromium 4 周发版日历绑架，8 周一个 major 跟版，
   最多支持最新 3 个 major（§1.2）。
4. **"`git am --3way` apply 成功 = patch 正确"** —— 只保证文本合并成功，不保证语义正确；升级后
   功能静默漂移正源于此（§5.3/坑 6）。
5. **"Electron 体积大是没优化"** —— Chromium+V8+Node 是 ~100MB 硬底，能裁的只有 locale/ASAR
   等边角，砍不动引擎（§8.4）。
6. **"patch 越多功能越强"** —— 官方原话「每个 patch 都是维护负担」，工程文化是疯狂 upstream、
   能不打就不打（§1.1/4.4）。

### 章末练习

1. 跑 Demo 1，统计你机器上拉到的 Electron `patches/chromium` 的 patch 总数，按 `feat_/build_/fix_`
   前缀分类。哪类最多？结合 §1.1「能不打就不打」解释为什么 `feat_` 仍占大头。
2. 跑 Demo 2，找一个改了 `.mojom` 的 patch 和一个只改 `.cc` 的 patch，预测哪个升级时更容易冲突，
   说明理由（提示：§5.2 上游对 mojom 的改动频率）。
3. 跑 Demo 6 的「升级冲突复现」部分：让「上游」也改 `return` 那一行，再 `git am --3way` 重放你的
   patch，观察是 clean apply、3way 自动合并、还是冲突。把这个微缩模型映射到 §5.2 的真实升级。
4. 在 `git.py` 源码里找到 `format_patch` 的 `--zero-commit` 与 `--full-index`，去掉它们手动
   `git format-patch` 同一 commit 两次，对比两次输出 diff，解释为什么 Electron 必须用这两个参数（§8.1/坑 3）。
5. 读 `support_mixed_sandbox_with_zygote.patch` 的 rationale，结合第 13 章的 sandbox/zygote，
   解释为什么这个 patch 是 indefinite-lifetime 的 functional patch、upstream 为何不会默认接受。
6. （选做，需 build）按 Demo 5 在一台 ~100GB 磁盘的机器上 `gclient sync` 到 patch hook 那一步
   （可中断在 sync），观察 `apply_all_patches.py` 的并行 apply 日志。

---

## 参考资料

1. Electron patch 体系官方指南（patch 三分类 / 最小化 / upstream 政策）
   https://www.electronjs.org/docs/latest/development/patches （已 WebFetch 核实）
2. Electron build-from-source 官方指南（depot_tools / gclient / gn / ninja 完整命令）
   https://www.electronjs.org/docs/latest/development/build-instructions-gn （已 WebFetch 核实）
3. Electron 发版节奏（与 Chromium milestone 绑定、N-3 支持窗口）
   https://www.electronjs.org/blog/12-week-cadence （已 WebFetch 核实）
4. 真实源码（均已 WebFetch 核实，raw.githubusercontent.com/electron/electron/main/...）：
   - `patches/config.json`（13 目录映射，逐字）
   - `patches/chromium/.patches`（apply 顺序，逐字文件名）
   - `script/lib/git.py`（`import_patches`/`format_patch`/`split_patches`/`am` flag，逐字）
   - `script/apply_all_patches.py`（config 解析 + ThreadPoolExecutor 并行）
   - `DEPS`（`chromium_version=151.0.7873.0`/`node_version=v24.16.0`/`patch_chromium` hook，逐字）
   - `build/args/release.gn`（逐字全文）
   - `build/args/all.gn`（`is_electron_build`/`root_extra_deps`/`node_module_version=148` 等，逐项）
   - `BUILD.gn`（`electron_lib` deps / `mac_framework_bundle` / `group("electron")` 结构）
   - `patches/chromium/can_create_window.patch`（头部 + 12 文件清单 + TODO，已核实）
   - `patches/chromium/support_mixed_sandbox_with_zygote.patch`（头部 + rationale + 3 文件）
5. Chromium Code Search（对比 patch 改动在 upstream 的演进）
   https://source.chromium.org/chromium/chromium/src/
6. 关联章节：第 13 章（Electron 运行时架构，本章强依赖）、第 01/02 章（多进程/sandbox/zygote）、
   第 03 章（Mojo/.mojom）、第 11 章（V8/Node 共享 V8）、第 10 章（Viz/OSR）。
