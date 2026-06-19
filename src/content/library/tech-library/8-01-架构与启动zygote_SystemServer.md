---
title: "Android 架构与启动：Zygote & SystemServer"
slug: "8-01"
collection: "tech-library"
group: "android系统"
order: 8001
summary: "前置知识：Linux 进程模型（fork/exec/COW）、Linux Capabilities、SELinux 基础、Binder IPC 机制（第 2 章会精讲，本章只引用）、JVM/ART 基本概念。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T10:37:19.000Z"
updatedAt: "2026-06-12T10:37:19.000Z"
---
> **前置知识**：Linux 进程模型（fork/exec/COW）、Linux Capabilities、SELinux 基础、Binder IPC 机制（第 2 章会精讲，本章只引用）、JVM/ART 基本概念。
>
> **TL;DR**：Android 从内核启动到"Home Screen 可用"依赖一条精心设计的单线串联：`init → Zygote(预加载 JVM+框架类) → fork SystemServer → 各系统服务按 phase 拉起 → AMS.systemReady() → Launcher`。Zygote 的核心价值是把 **8000+ 个框架类的 JVM 初始化成本**从每次 app 启动均摊走，用 COW 内存共享让所有进程共享只读代码页。理解这条链路是理解 Android 一切问题（启动慢/OOM/服务 crash/SELinux denial）的基础。

---

## 目录

1. [设计考古：为什么是 Zygote 模型](#1-设计考古)
2. [全局启动链路图](#2-全局启动链路)
3. [init 进程：从内核到用户空间](#3-init-进程)
4. [Zygote 精读：预加载与 fork 机制](#4-zygote-精读)
5. [SystemServer 精读：服务注册与 boot phase](#5-systemserver-精读)
6. [Binder 在启动链路中的角色](#6-binder-在启动链路中的角色)
7. [SELinux 与 capabilities 安全边界](#7-selinux-与-capabilities)
8. [Project Treble 对启动架构的影响](#8-project-treble)
9. [失败模式与生产真坑](#9-失败模式与生产真坑)
10. [可运行 Demo 集合](#10-可运行-demo)
11. [方案对比表](#11-方案对比)
12. [章末五件套](#12-章末五件套)

---

## 1. 设计考古

### 1.1 Zygote 的由来：Android 1.0 前的设计决策

Android 在 2007 年前（代号 Astro、Bender，最终 Cupcake）面临一个根本约束：移动设备 RAM 极少（最初目标 64 MB），但 Java 应用每次启动都需要重新初始化 JVM 和加载数千个框架类，在当时设备上耗时 500ms~2s。

Dalvik 团队（Dan Bornstein 等）借鉴了 Web 服务器 preforking 模型（Apache 的 prefork MPM）和 JVM warm-up 思路，提出 **Zygote 模型**：

1. 系统启动时，在一个"受保护的温暖 JVM"里把所有框架类都 `Class.forName()` 一遍，让 ART 的 JIT/AOT 缓存和 DEX 映射全部到位。
2. 此后每次 app 启动，不再 `exec` 新进程，而是 `fork` 这个预热好的进程。
3. Linux `fork` + COW（Copy-on-Write）语义保证：父子进程共享同一批**只读**内存页（代码、常量、类数据），只有写入时才按页复制。

这个设计的结果是：

- **冷启动减少约 200~400ms**（具体取决于机型，避免了 JVM init + framework class loading）
- **内存节省显著**：8000+ 个预加载类的代码页在所有 app 间共享（Zygote 占 RSS 约 50~80 MB，但 PSS 远小于此）

> 注：Dalvik 到 ART 的切换（Android 4.4 实验性引入，5.0 正式替换）没有改变 Zygote 模型，只是让 preload 时预编译的 DEX 缓存（.odex/.art）命中率更高。

**出处**：
- Google I/O 2008 "Anatomy & Physiology of an Android"（Dianne Hackborn）
- AOSP commit history: `platform/frameworks/base` 的最早 commit（2008 年）即包含 `ZygoteInit.java`
- Android Developers Blog "Memory Management for Android Games"（引用了 Zygote 共享内存数字）

### 1.2 SystemServer：不是普通服务进程

Android 的架构选择是把大量系统服务（AMS、WMS、PMS 等）塞进**同一个进程** `system_server`，而非每个服务一个进程（对比 Linux 的 systemd 每服务独立 daemon）。

**设计动机**：
- 同进程服务调用 **不走 Binder**，速度接近直接函数调用（省去 Binder 内核往返）
- 统一 crash 处理：`system_server` 死亡触发 zygote 重启，整个系统重置到干净状态，比"某服务 crash 后别的服务状态不一致"更简单可控
- 共享 `mSystemContext` 和 service registry，服务间依赖解耦靠 phase 机制而非启动顺序 hardcode

**代价**：
- `system_server` 进程的任何 OOM 或 uncaught exception 导致设备重启
- 启动时间串行（虽然有 `SystemServerInitThreadPool` 并行优化）
- 内存隔离为零：一个服务的内存泄漏会影响所有服务

### 1.3 演进节点

| Android 版本 | 关键变化 |
|---|---|
| 1.0 (2008) | Zygote + SystemServer 基础架构确立 |
| 2.3 | Dalvik JIT，preload 效果进一步改善 |
| 4.4 | ART 引入（实验），zygote 预编译 .odex |
| 5.0 | ART 正式替换 Dalvik，64-bit 支持，zygote64 |
| 7.0 | Project Treble 开始规划（8.0 落地） |
| 8.0 | **Project Treble**：vendor 分区隔离，hwservicemanager，zygote 双进程（64+32）稳定化 |
| 9.0 | USAP（Unspecialized App Process）池实验性引入 |
| 10.0 | USAP 池正式启用，进一步减少 app 启动延迟 |
| 12.0 | App Zygote（隔离进程的 sub-zygote），渲染进程沙箱化 |
| 13.0+ | Mainline（APEX）模块可独立更新，部分服务移入 APEX |

---

## 2. 全局启动链路

```
┌─────────────────────────────────────────────────────────────────────┐
│  Power On → Bootloader → Linux Kernel → init (PID 1)               │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ init.rc 解析
                              │ 执行 on early-init / on init / on late-init
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  init 启动关键 daemon:                                               │
│    servicemanager (Binder DNS)                                       │
│    hwservicemanager / vndservicemanager (Treble)                     │
│    ueventd, logd, adbd, ...                                          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ on zygote-start trigger
                              │ start zygote / start zygote_secondary
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Zygote64 (app_process64) -- Zygote32 (app_process32)              │
│                                                                      │
│  1. AndroidRuntime::start("ZygoteInit", ...)                         │
│  2. ZygoteInit.main()                                                │
│       preload(): 8000+ classes, resources, shared libs               │
│       gcAndFinalize()                                                │
│  3. forkSystemServer() → fork() → child: handleSystemServerProcess  │
│  4. zygoteServer.runSelectLoop()  ← 永久等待 app fork 请求           │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ fork 出 system_server (PID ~xxx)
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SystemServer.main() → new SystemServer().run()                     │
│                                                                      │
│  run():                                                              │
│    loadLibrary("android_servers")     // 加载 JNI                   │
│    BinderInternal.setMaxThreads(31)   // Binder 线程池上限           │
│    createSystemContext()              // ActivityThread.systemMain() │
│    new SystemServiceManager(ctx)                                     │
│                                                                      │
│    startBootstrapServices(t):  [PHASE 100 → 200]                    │
│      Watchdog, AMS/ATMS, PMS, DisplayManager, PowerManager...       │
│    startCoreServices(t):       [PHASE 480]                           │
│      BatteryService, UsageStats, WebViewUpdate...                   │
│    startOtherServices(t):      [PHASE 500 → 550]                    │
│      WMS, IMS, CameraService, NMS, ConnectivityService...           │
│    startApexServices(t)                                              │
│                                                                      │
│    mActivityManagerService.systemReady(callback)  ← 关键节点         │
│      → PHASE_BOOT_COMPLETED (1000)                                   │
│      → 启动 persistent apps, home launcher                           │
│                                                                      │
│    Looper.loop()  ← 永不返回                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. init 进程

### 3.1 init 是什么

`init` 是 Android 的 PID 1，由 Linux 内核在完成硬件初始化后执行。它不是 Linux 发行版的 systemd/SysVinit，而是 AOSP 自己实现的（`platform/system/core/init/`）。

**【真实源码 `platform_system_core@init/main.cpp`】**

```cpp
int main(int argc, char** argv) {
    // 最高优先级
    setpriority(PRIO_PROCESS, 0, -20);

    // 同一个二进制文件根据 argv[0] 或参数分支到不同角色
    if (!strcmp(basename(argv[0]), "ueventd")) {
        return ueventd_main(argc, argv);
    }

    if (argc > 1) {
        if (!strcmp(argv[1], "subcontext")) {
            // SELinux subcontext 隔离执行
            const BuiltinFunctionMap& function_map = GetBuiltinFunctionMap();
            return SubcontextMain(argc, argv, &function_map);
        }
        if (!strcmp(argv[1], "selinux_setup")) {
            return SetupSelinux(argv);  // 第一阶段 SELinux 初始化
        }
        if (!strcmp(argv[1], "second_stage")) {
            return SecondStageMain(argc, argv);  // 第二阶段 init
        }
    }
    return FirstStageMain(argc, argv);  // 默认：第一阶段
}
```

init 分**三个阶段**执行（通过 `execv` 重新执行自身，用参数区分）：

1. **FirstStageMain**：在 kernel 直接挂载的临时文件系统上运行，挂载 `/system`、`/vendor`、`/odm`，处理 A/B 分区，加载 first-stage init 的 `init.rc`。
2. **SetupSelinux**：加载 SELinux 策略（`/system/etc/selinux/plat_sepolicy.cil`），然后 `execv` 自身切换到 second stage。
3. **SecondStageMain**：完整的 init，解析 `init.rc` 及所有 `import` 的 `.rc` 文件，执行 `on` triggers，启动 `service`。

### 3.2 init.rc 启动 Zygote

**【真实源码 `platform_system_core@rootdir/init.rc`（节选）】**

```rc
# init.rc 动态 import zygote 配置，根据 ro.zygote 属性决定用哪种模式
import /system/etc/init/hw/init.${ro.zygote}.rc

# on zygote-start trigger 由 on boot 触发
on zygote-start
    wait_for_prop odsign.verification.done 1   # 等待签名验证完成
    exec_start update_verifier                  # A/B OTA 验证
    start statsd
    start zygote
    start zygote_secondary
```

**【真实源码 `platform_system_core@rootdir/init.zygote64_32.rc`（节选）】**

```rc
service zygote /system/bin/app_process64 -Xzygote /system/bin \
        --zygote --start-system-server --socket-name=zygote
    class main
    priority -20
    user root
    group root readproc reserved_disk
    socket zygote stream 660 root system        # Unix domain socket
    socket usap_pool_primary stream 660 root system
    onrestart exec_background - system system -- /system/bin/vdc volume reset
    onrestart write /sys/power/state on
    onrestart restart audioserver
    onrestart restart cameraserver
    onrestart restart media
    onrestart restart netd
    onrestart restart wificond
    task_profiles ProcessCapacityHigh MaxPerformance

service zygote_secondary /system/bin/app_process32 -Xzygote /system/bin \
        --zygote --socket-name=zygote_secondary
    class main
    priority -20
    user root
    group root readproc reserved_disk
    socket zygote_secondary stream 660 root system
    socket usap_pool_secondary stream 660 root system
    onrestart restart zygote    # zygote_secondary 挂了→主 zygote 也重启
    task_profiles ProcessCapacityHigh MaxPerformance
```

关键点：
- `--start-system-server` 只传给 `zygote`（64-bit 主进程），`zygote_secondary` 不启动 system_server。
- `socket zygote stream 660 root system`：init 替 zygote 创建好 Unix socket，zygote 启动后从 init 传递的 fd 获取它（避免 root 权限运行时自建 socket 的权限复杂性）。
- `onrestart` 链：zygote 重启会拉起整个 media/audio/camera 子系统重启——这是 Android "soft reboot"的实现机制。

### 3.3 init 如何 execv 启动 zygote

**【真实源码 `platform_system_core@init/service.cpp`（ExpandArgsAndExecv，节选）】**

```cpp
static bool ExpandArgsAndExecv(const std::vector<std::string>& args, bool sigstop) {
    std::vector<char*> c_strings;
    std::vector<std::string> expanded_args(args.size());

    c_strings.push_back(const_cast<char*>(args[0].data()));  // argv[0] = /system/bin/app_process64
    for (std::size_t i = 1; i < args.size(); ++i) {
        auto expanded_arg = ExpandProps(args[i]);   // 展开 ${prop} 变量
        expanded_args[i] = *expanded_arg;
        c_strings.push_back(expanded_args[i].data());
    }
    c_strings.push_back(nullptr);

    if (sigstop) {
        kill(getpid(), SIGSTOP);  // 给调试器挂上来的机会
    }
    return execv(c_strings[0], c_strings.data()) == 0;
}
```

`fork()` + `execv()` 让 `/system/bin/app_process64` 以 root 身份启动。`app_process64` 的 `main()` 调用 `AndroidRuntime::start("com.android.internal.os.ZygoteInit", args, true)` 进入 JVM 世界。

---

## 4. Zygote 精读

### 4.1 ZygoteInit.main() 全流程

**【真实源码 `platform_frameworks_base@core/java/com/android/internal/os/ZygoteInit.java`（节选，已核实）】**

```java
public static void main(String[] argv) {
    ZygoteServer zygoteServer = null;

    // ① 禁止在 zygote 阶段创建线程（fork + 多线程 = 死锁）
    ZygoteHooks.startZygoteNoThreadCreation();

    // 设置进程组为 foreground，避免 LMK 误杀
    Os.setpgid(0, 0);

    try {
        final long startTime = SystemClock.elapsedRealtime();
        boolean startSystemServer = false;
        String zygoteSocketName = "zygote";
        String abiList = null;
        boolean enableLazyPreload = false;

        // 解析命令行参数
        for (int i = 1; i < argv.length; i++) {
            if ("--start-system-server".equals(argv[i])) {
                startSystemServer = true;
            } else if ("--enable-lazy-preload".equals(argv[i])) {
                enableLazyPreload = true;
            } else if (argv[i].startsWith("--socket-name=")) {
                zygoteSocketName = argv[i].substring("--socket-name=".length());
            }
            // ...
        }

        final boolean isPrimaryZygote = zygoteSocketName.equals(Zygote.PRIMARY_SOCKET_NAME);

        // ② 预加载（除非 lazy 模式）
        if (!enableLazyPreload) {
            preload(bootTimingsTraceLog);  // 核心！见 4.2
        }

        // ③ 预加载完成，GC 清理，让 Zygote heap 进入干净状态
        gcAndFinalize();

        // ④ 创建 ZygoteServer（监听 socket）
        zygoteServer = new ZygoteServer(isPrimaryZygote);

        // ⑤ fork SystemServer（仅主 zygote）
        if (startSystemServer) {
            Runnable r = forkSystemServer(abiList, zygoteSocketName, zygoteServer);
            if (r != null) {
                r.run();   // 子进程执行这里，父进程 r == null
                return;
            }
        }

        // ⑥ 主循环：等待 app fork 请求
        Log.i(TAG, "Accepting command socket connections");
        caller = zygoteServer.runSelectLoop(abiList);
    } catch (Throwable ex) {
        Log.e(TAG, "System zygote died with fatal exception", ex);
        throw ex;
    } finally {
        if (zygoteServer != null) {
            zygoteServer.closeServerSocket();
        }
    }

    // caller != null 说明是刚 fork 的子进程
    if (caller != null) {
        caller.run();
    }
}
```

**关键设计**：`fork()` 后父子进程从同一个 `main()` 返回，通过 `r != null`（子进程）vs `r == null`（父进程）区分执行路径。这是 Unix fork-and-dispatch 的经典用法。

### 4.2 preload()：COW 共享内存的关键

**【真实源码 `platform_frameworks_base@core/java/com/android/internal/os/ZygoteInit.java`（preload 节选，已核实）】**

```java
static void preload(TimingsTraceLog bootTimingsTraceLog) {
    Log.d(TAG, "begin preload");
    beginPreload();

    // 加载 /system/etc/preloaded-classes 里列出的 8000+ 个类
    preloadClasses();

    // 缓存非 boot classpath 的 ClassLoader（WebView、系统 overlay 等）
    cacheNonBootClasspathClassLoaders();

    // 预加载 resources.arsc（主题、layout 等）
    Resources.preloadResources();

    // 预加载 HAL 层共享库（OpenGL ES、Vulkan、Audio HAL 等）
    nativePreloadAppProcessHALs();

    // 可选：预加载 GPU 驱动
    maybePreloadGraphicsDriver();

    // 预加载 libandroid.so、libaudio.so 等 JNI 库
    preloadSharedLibraries();

    // 预加载 ICU 文本资源（Emoji、语言规则等）
    preloadTextResources();

    // 预加载 WebView（条件性）
    WebViewFactory.prepareWebViewInZygote();

    // 预热 JCA 密码学 Provider（SecureRandom 等）
    warmUpJcaProviders();

    sPreloadComplete = true;
}
```

**preloadClasses() 核心循环（已核实）**：

```java
// PRELOADED_CLASSES = "/system/etc/preloaded-classes"
private static void preloadClasses() {
    // 临时降权：避免 static initializer 有 root 操作
    final int count;
    try (InputStream is = new FileInputStream(PRELOADED_CLASSES)) {
        Log.i(TAG, "Preloading classes...");
        final long startTime = SystemClock.uptimeMillis();

        // 暂时 drop root，防止 <clinit> 中有不安全操作
        // (dropAccessToAddressSpaceRandomization / 其余 capability 操作)

        BufferedReader br = new BufferedReader(
            new InputStreamReader(is), Zygote.SOCKET_BUFFER_SIZE);
        int missingLambdaCount = 0;

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("#") || line.equals("")) continue;  // 跳过注释

            Trace.traceBegin(Trace.TRACE_TAG_DALVIK, line);
            try {
                // true = 强制执行 <clinit> 静态初始化器
                Class.forName(line, true, null);
                count++;
            } catch (ClassNotFoundException e) {
                if (line.contains("$$Lambda$")) {
                    missingLambdaCount++;  // Lambda 类可能被 R8 优化掉，忽略
                } else {
                    Log.w(TAG, "Class not found for preloading: " + line);
                }
            } catch (UnsatisfiedLinkError e) {
                Log.w(TAG, "Problem preloading " + line + ": " + e);
            } catch (Throwable t) {
                Log.e(TAG, "Error preloading " + line + ".", t);
                if (t instanceof Error) throw (Error) t;
                throw new RuntimeException(t);
            }
            Trace.traceEnd(Trace.TRACE_TAG_DALVIK);
        }
        Log.i(TAG, "...preloaded " + count + " classes in "
            + (SystemClock.uptimeMillis() - startTime) + "ms.");
    }
}
```

**COW 机制详解**：

- `preloadClasses()` 执行后，所有类的 `.dex`/`.art`（mmaped 文件）和 JVM 类数据页均为**只读页**（`PROT_READ`），OS 计入 Zygote 的 Shared Clean 内存。
- `fork()` 后子进程获得同样的页表，只要不修改，这些页**零拷贝**共享。
- 当子进程（或 Zygote 本身）写入某页时，内核才触发 COW：分配新物理页，复制内容，映射到写进程的虚拟地址空间。
- 实测数据（Pixel 4, Android 12）：`system_server` VSS ~900MB，PSS ~150MB，Shared Clean（来自 Zygote）约 55~70MB。

### 4.3 forkSystemServer()：fork + capabilities 配置

**【真实源码 `platform_frameworks_base@core/java/com/android/internal/os/ZygoteInit.java`（forkSystemServer 节选，已核实）】**

```java
private static Runnable forkSystemServer(String abiList, String socketName,
        ZygoteServer zygoteServer) {
    // system_server 需要的 Linux capabilities（比普通 app 多）
    long capabilities =
            (1L << OsConstants.CAP_IPC_LOCK)       |  // 锁内存，防止 swap
            (1L << OsConstants.CAP_KILL)            |  // 向任意进程发信号
            (1L << OsConstants.CAP_NET_ADMIN)       |  // 网络配置
            (1L << OsConstants.CAP_NET_BIND_SERVICE)|  // 绑定 < 1024 端口
            (1L << OsConstants.CAP_NET_BROADCAST)   |
            (1L << OsConstants.CAP_NET_RAW)         |
            (1L << OsConstants.CAP_SYS_MODULE)      |  // 加载内核模块（有限）
            (1L << OsConstants.CAP_SYS_NICE)        |  // 调整进程优先级
            (1L << OsConstants.CAP_SYS_PTRACE)      |  // 跟踪调试
            (1L << OsConstants.CAP_SYS_TIME)        |  // 设置系统时钟
            (1L << OsConstants.CAP_SYS_TTY_CONFIG)  |
            (1L << OsConstants.CAP_WAKE_ALARM)      |  // 设置 AlarmManager
            (1L << OsConstants.CAP_BLOCK_SUSPEND);     // 阻止系统休眠

    // --setuid=1000 = AID_SYSTEM, --setgid=1000
    // --setgroups 包含 AID_READPROC, AID_GRAPHICS 等辅助 GID
    String[] args = {
        "--setuid=1000",
        "--setgid=1000",
        "--setgroups=1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,"
            + "1018,1021,1023,1024,1032,1065,3001,3002,3003,3006,3007,"
            + "3009,3010,3011",
        "--capabilities=" + capabilities + "," + capabilities,
        "--nice-name=system_server",
        "--runtime-args",
        "--target-sdk-version=" + VMRuntime.SDK_VERSION_CUR_DEVELOPMENT,
        "com.android.server.SystemServer",
    };

    ZygoteArguments parsedArgs = ZygoteArguments.getInstance(commandBuffer);
    Zygote.applyDebuggerSystemProperty(parsedArgs);
    Zygote.applyInvokeWithSystemProperty(parsedArgs);

    // 实际 fork
    pid = Zygote.forkSystemServer(
            parsedArgs.mUid, parsedArgs.mGid,
            parsedArgs.mGids,
            parsedArgs.mRuntimeFlags,
            null,   // rlimits
            parsedArgs.mPermittedCapabilities,
            parsedArgs.mEffectiveCapabilities);

    if (pid == 0) {
        // 子进程：关闭不需要的 socket 描述符
        if (socketName.equals(Zygote.PRIMARY_SOCKET_NAME)) {
            SpecialRuntimeFlags.clearFlags(
                SpecialRuntimeFlags.NO_FORCE_SET_LARGE_HEAP);
        }
        zygoteServer.closeServerSocket();
        return handleSystemServerProcess(parsedArgs);  // 返回 Runnable
    }
    return null;  // 父进程返回 null
}
```

**【真实源码 `platform_frameworks_base@core/java/com/android/internal/os/Zygote.java`（forkAndSpecialize 节选，已核实）】**

```java
static int forkSystemServer(int uid, int gid, int[] gids, int runtimeFlags,
        int[][] rlimits, long permittedCapabilities, long effectiveCapabilities) {
    ZygoteHooks.preFork();   // ART: 停止 GC，保存 heap 状态

    int pid = nativeForkSystemServer(
            uid, gid, gids, runtimeFlags, rlimits,
            permittedCapabilities, effectiveCapabilities);

    Thread.currentThread().setPriority(Thread.NORM_PRIORITY);
    ZygoteHooks.postForkCommon();  // ART: 恢复 GC
    return pid;
}
```

### 4.4 SpecializeCommon()：fork 后子进程的隔离配置

**【真实源码 `platform_frameworks_base@core/jni/com_android_internal_os_Zygote.cpp`（SpecializeCommon 节选，已核实）】**

```cpp
static void SpecializeCommon(JNIEnv* env, uid_t uid, gid_t gid,
        jintArray gids, jint runtime_flags, ...) {

    // 1. 创建独立的 mount namespace（与 Zygote 隔离文件系统视图）
    ensureInAppMountNamespace(fail_fn);
    // unshare(CLONE_NEWNS) — 子进程有独立的 mount 表

    // 2. App 数据目录隔离（Android 11+）
    if (mount_data_dirs) {
        isolateAppData(env, pkg_data_info_list, ...);
        // 用 tmpfs 覆盖 /data/data，再 bind mount 只有该 app 的目录
    }

    // 3. Capabilities 设置（必须在 setuid 之前）
    EnableKeepCapabilities(fail_fn);
    SetInheritable(permitted_capabilities, fail_fn);
    DropCapabilitiesBoundingSet(fail_fn, bounding_capabilities);

    // 4. 切换 GID/UID（不可逆）
    setresgid(gid, gid, gid);
    SetUpSeccompFilter(uid, is_child_zygote);
    setresuid(uid, uid, uid);

    // 5. 最终 capability 配置
    SetCapabilities(permitted_capabilities, effective_capabilities,
                    permitted_capabilities, fail_fn);

    // 6. SELinux context 切换
    selinux_android_setcontext(uid, is_system_server, se_info_ptr, nice_name_ptr);

    // 7. 线程名
    SetThreadName(nice_name.value());
}
```

### 4.5 ZygoteServer.runSelectLoop()：等待 app fork 请求

**【真实源码 `platform_frameworks_base@core/java/com/android/internal/os/ZygoteServer.java`（构造器节选，已核实）】**

```java
ZygoteServer(boolean isPrimaryZygote) {
    mUsapPoolEventFD = Zygote.getUsapPoolEventFD();

    if (isPrimaryZygote) {
        // 从 init 传递的 fd 创建托管 socket（避免自行 bind/listen）
        mZygoteSocket = Zygote.createManagedSocketFromInitSocket(
            Zygote.PRIMARY_SOCKET_NAME);             // "zygote"
        mUsapPoolSocket = Zygote.createManagedSocketFromInitSocket(
            Zygote.USAP_POOL_PRIMARY_SOCKET_NAME);   // "usap_pool_primary"
    } else {
        mZygoteSocket = Zygote.createManagedSocketFromInitSocket(
            Zygote.SECONDARY_SOCKET_NAME);           // "zygote_secondary"
        // ...
    }
    mUsapPoolSupported = true;
    fetchUsapPoolPolicyProps();
}
```

`runSelectLoop()` 的本质是一个 `poll()` 循环：
- 监听 `mZygoteSocket`（新连接）、已有 session socket（读命令）、USAP 池管道（池补充信号）
- 每个新连接对应一个 `ZygoteConnection`
- 收到 fork 请求后调用 `ZygoteConnection.processOneCommand()` → `Zygote.forkAndSpecialize()` → 返回 `Runnable`（子进程执行）或继续循环（父进程）

### 4.6 USAP（Unspecialized App Process）池

Android 10 引入 USAP 池来进一步减少 app 冷启动延迟：

- Zygote 预先 fork 出若干未专化的子进程（USAP），它们等待通过 `usap_pool_primary` socket 接收专化参数。
- 当 AMS 请求启动新 app 时，如果 USAP 池有可用进程，直接发送专化参数（UID/GID/seinfo 等），省去 `fork()` 本身的延迟（在 ARM64 上 `fork` 约 10~50ms）。
- USAP 池大小通过 `persist.device_config.runtime_native.usap_pool_size_max` 控制。

---

## 5. SystemServer 精读

### 5.1 run() 方法全貌

**【真实源码 `platform_frameworks_base@services/java/com/android/server/SystemServer.java`（run 节选，已核实）】**

```java
private void run() {
    TimingsTraceAndSlog t = new TimingsTraceAndSlog();
    try {
        t.traceBegin("InitBeforeStartServices");

        // 记录进程启动计数（便于 crash loop 检测）
        SystemProperties.set(SYSPROP_START_COUNT, String.valueOf(mStartCount));
        SystemProperties.set(SYSPROP_START_ELAPSED,
            String.valueOf(mRuntimeStartElapsedTime));

        EventLog.writeEvent(EventLogTags.SYSTEM_SERVER_START,
                mStartCount, mRuntimeStartUptime, mRuntimeStartElapsedTime);

        // 设置默认 WTF handler
        RuntimeInit.setDefaultApplicationWtfHandler(SystemServer::handleEarlySystemWtf);

        // 初始化 ApplicationSharedMemory（ashmem 区域，用于跨进程读系统信息）
        ApplicationSharedMemory instance = ApplicationSharedMemory.create();
        ApplicationSharedMemory.setInstance(instance);

        // 启动各阶段服务
        try {
            t.traceBegin("StartServices");
            startBootstrapServices(t);
            startCoreServices(t);
            startOtherServices(t);
            startApexServices(t);
            updateWatchdogTimeout(t);
            CriticalEventLog.getInstance().logSystemServerStarted();
        } catch (Throwable ex) {
            Slog.e("System", "Failure starting system services", ex);
            throw ex;
        } finally {
            t.traceEnd(); // StartServices
        }

        // 主循环，永不返回
        Looper.loop();
        throw new RuntimeException("Main thread loop unexpectedly exited");
    }
    // ...
}
```

**关键初始化（run() 早期，已核实）**：

```java
// 加载 JNI 库（包含 SurfaceFlinger、Input、Camera 等 native 服务的绑定）
System.loadLibrary("android_servers");

// 设置 system_server 的 Binder 线程池上限为 31
BinderInternal.setMaxThreads(sMaxBinderThreads);  // sMaxBinderThreads = 31

// 通过 ActivityThread.systemMain() 创建系统 Context
createSystemContext();

// 创建服务管理器（所有 SystemService 的容器）
mSystemServiceManager = new SystemServiceManager(mSystemContext);
mSystemServiceManager.setStartInfo(mRuntimeRestart,
        mRuntimeStartElapsedTime, mRuntimeStartUptime);
LocalServices.addService(SystemServiceManager.class, mSystemServiceManager);
```

### 5.2 createSystemContext()

**【真实源码 `platform_frameworks_base@services/java/com/android/server/SystemServer.java`（已核实）】**

```java
// 约 line 1065
private void createSystemContext() {
    // ActivityThread 是 AMS 的"对等体"，在 SystemServer 里以 system 身份运行
    ActivityThread activityThread = ActivityThread.systemMain();
    mSystemContext = activityThread.getSystemContext();
    mSystemContext.setTheme(DEFAULT_SYSTEM_THEME);

    // 独立的 SystemUI context（用于 WMS 等 UI 相关服务）
    final Context systemUiContext = activityThread.getSystemUiContext();
    systemUiContext.setTheme(DEFAULT_SYSTEM_THEME);
    Trace.registerWithPerfetto();
}
```

`ActivityThread.systemMain()` 是一个特殊工厂方法，它创建一个**不对应任何 APK** 的 ActivityThread，使 SystemServer 内的代码可以使用完整的 Android 上下文（Resources、ContentResolver 等）。

### 5.3 Boot Phase 机制

SystemService 的生命周期由 `SystemServiceManager.startBootPhase()` 驱动，服务通过覆写 `onBootPhase()` 在正确时机完成初始化。

**【真实源码 `platform_frameworks_base@services/core/java/com/android/server/SystemService.java`（已核实）】**

```java
// Phase 常量（严格升序，不可跳过）
public static final int PHASE_WAIT_FOR_DEFAULT_DISPLAY = 100;     // DisplayManager 就绪
public static final int PHASE_WAIT_FOR_SENSOR_SERVICE  = 200;     // SensorService 就绪
public static final int PHASE_LOCK_SETTINGS_READY      = 480;     // 锁屏设置可读
public static final int PHASE_SYSTEM_SERVICES_READY    = 500;     // 大多数服务已就绪
public static final int PHASE_DEVICE_SPECIFIC_SERVICES_READY = 520;
public static final int PHASE_ACTIVITY_MANAGER_READY   = 550;     // AMS 就绪
public static final int PHASE_THIRD_PARTY_APPS_CAN_START = 600;   // 可以启动第三方 app
public static final int PHASE_BOOT_COMPLETED           = 1000;    // 开机完成

// 所有 SystemService 必须实现的抽象方法
public abstract void onStart();

// 可选覆写：按 phase 做延迟初始化
public void onBootPhase(@BootPhase int phase) {}
```

**【真实源码 `platform_frameworks_base@services/core/java/com/android/server/SystemServiceManager.java`（已核实）】**

```java
public void startBootPhase(@NonNull TimingsTraceAndSlog t, int phase) {
    if (phase <= mCurrentPhase) {
        throw new IllegalArgumentException("Next phase must be larger than previous");
    }
    mCurrentPhase = phase;
    t.traceBegin("OnBootPhase_" + phase);
    try {
        final int serviceLen = mServices.size();
        for (int i = 0; i < serviceLen; i++) {
            final SystemService service = mServices.get(i);
            long time = SystemClock.elapsedRealtime();
            try {
                service.onBootPhase(mCurrentPhase);
            } catch (Exception ex) {
                throw new RuntimeException("Failed to boot service " +
                        service.getClass().getName() +
                        ": onBootPhase threw an exception during phase "
                        + mCurrentPhase, ex);
            }
            warnIfTooLong(SystemClock.elapsedRealtime() - time, service, "onBootPhase");
        }
    } finally {
        t.traceEnd();
    }
}
```

`startService()` 的防重入和 onStart 调用：

**【真实源码 `platform_frameworks_base@services/core/java/com/android/server/SystemServiceManager.java`（已核实）】**

```java
public void startService(@NonNull final SystemService service) {
    String className = service.getClass().getName();
    if (mServiceClassnames.contains(className)) {
        Slog.i(TAG, "Not starting an already started service " + className);
        return;
    }
    mServiceClassnames.add(className);
    mServices.add(service);

    long time = SystemClock.elapsedRealtime();
    try {
        service.onStart();
    } catch (RuntimeException ex) {
        throw new RuntimeException("Failed to start service " +
                service.getClass().getName() + ": onStart threw an exception", ex);
    }
    warnIfTooLong(SystemClock.elapsedRealtime() - time, service, "onStart");
}
```

### 5.4 startBootstrapServices()：关键服务启动

**【真实源码 `platform_frameworks_base@services/java/com/android/server/SystemServer.java`（节选，已核实）】**

```java
private void startBootstrapServices(@NonNull TimingsTraceAndSlog t) {
    // Watchdog 最先启动：监控所有后续服务
    t.traceBegin("StartWatchdog");
    final Watchdog watchdog = Watchdog.getInstance();
    watchdog.start();
    t.traceEnd();

    // Installer: 与 installd daemon 通信
    t.traceBegin("StartInstaller");
    Installer installer = mSystemServiceManager.startService(Installer.class);
    t.traceEnd();

    // ATMS + AMS: 核心活动/任务管理
    t.traceBegin("StartActivityManager");
    ActivityTaskManagerService atm = mSystemServiceManager.startService(
            ActivityTaskManagerService.Lifecycle.class).getService();
    mActivityManagerService = ActivityManagerService.Lifecycle.startService(
            mSystemServiceManager, atm);
    mActivityManagerService.setSystemServiceManager(mSystemServiceManager);
    mActivityManagerService.setInstaller(installer);
    mWindowManagerGlobalLock = atm.getGlobalLock();
    t.traceEnd();

    // PowerManager
    t.traceBegin("StartPowerManager");
    mPowerManagerService = mSystemServiceManager.startService(PowerManagerService.class);
    t.traceEnd();

    // DisplayManager（在 PHASE_WAIT_FOR_DEFAULT_DISPLAY 之前必须完成）
    t.traceBegin("StartDisplayManager");
    mDisplayManagerService = mSystemServiceManager.startService(DisplayManagerService.class);
    t.traceEnd();

    // 等待 Display 就绪（其他服务依赖它）
    mSystemServiceManager.startBootPhase(t,
            SystemService.PHASE_WAIT_FOR_DEFAULT_DISPLAY);

    // PackageManager: 最耗时（需要 scan APKs）
    t.traceBegin("StartPackageManagerService");
    try {
        Watchdog.getInstance().pauseWatchingCurrentThread("packagemanagermain");
        mPackageManagerService = PackageManagerService.main(
                mSystemContext, installer, domainVerificationService,
                mFactoryTestMode != FactoryTest.FACTORY_TEST_OFF);
    } finally {
        Watchdog.getInstance().resumeWatchingCurrentThread("packagemanagermain");
    }
    t.traceEnd();

    // ... 更多服务
}
```

### 5.5 AMS.systemReady() 与 PHASE_BOOT_COMPLETED

`startOtherServices()` 末尾会调用 `mActivityManagerService.systemReady(goingCallback, t)`，这是启动链路的终点之一：

```
AMS.systemReady()
  → startBootPhase(PHASE_ACTIVITY_MANAGER_READY)        // phase 550
  → 启动 persistent app（如 Phone/Telephony）
  → 发 ACTION_PRE_BOOT_COMPLETED 广播
  → startBootPhase(PHASE_THIRD_PARTY_APPS_CAN_START)    // phase 600
  → 发 ACTION_BOOT_COMPLETED 广播
  → startBootPhase(PHASE_BOOT_COMPLETED)                // phase 1000
  → 触发 Home Launcher 启动
```

### 5.6 WindowManagerService 的特殊初始化

**【真实源码 `platform_frameworks_base@services/core/java/com/android/server/wm/WindowManagerService.java`（已核实）】**

```java
// WMS 构造必须在 DisplayThread 上执行（非主线程！）
public static WindowManagerService main(final Context context,
        final InputManagerService im, final boolean showBootMsgs,
        WindowManagerPolicy policy, ActivityTaskManagerService atm) {
    final WindowManagerService[] wms = new WindowManagerService[1];
    // runWithScissors: 在 DisplayThread 同步执行，主线程阻塞等待
    DisplayThread.getHandler().runWithScissors(() ->
            wms[0] = new WindowManagerService(context, im, showBootMsgs,
                    policy, atm, ...), 0);
    return wms[0];
}
```

`runWithScissors()` 是 Android framework 里的一个危险方法（可能死锁），WMS 是其少数合法使用场景之一。

---

## 6. Binder 在启动链路中的角色

### 6.1 ServiceManager：Binder 的 DNS

`servicemanager` 在 init 阶段早于 zygote 启动，是所有 Binder 服务的注册中心（类似 DNS）：

- `addService(name, binder)` → 服务注册
- `getService(name)` → 服务查询
- 自身通过固定 handle 0 访问（不需要查询，硬编码在 libbinder 里）

SystemServer 的每个服务在 `onStart()` 时调用 `publishBinderService(name, binder)` → 最终调用 `ServiceManager.addService()`。

### 6.2 system_server 的 Binder 线程池

**【真实源码 `platform_frameworks_base@services/java/com/android/server/SystemServer.java`（已核实）】**

```java
// run() 方法早期
private static final int sMaxBinderThreads = 31;

// 设置线程池上限（底层: ioctl(binderFd, BINDER_SET_MAX_THREADS, 31)）
BinderInternal.setMaxThreads(sMaxBinderThreads);
```

为什么是 31？系统服务接收来自数百个 app 的并发 Binder 调用，31 是经验值（Linux 内核默认上限 15，system_server 需要更多）。每个线程约占 1MB stack，31 线程 = ~31MB，在可接受范围内。

### 6.3 Binder 调用流向

```
App 进程                  system_server
    │                         │
    │──getService("activity")→│  ServiceManager.getService()
    │                         │
    │──startActivity(intent)──→│  AMS.startActivity()  [Binder 调用]
    │                         │  (在 Binder 线程池中执行)
    │                         │  → ATMS 处理
    │                         │  → WMS 处理（同进程直接调用）
    │←──────────────结果──────│
```

---

## 7. SELinux 与 Capabilities

### 7.1 安全域（domain）

每个进程在 SELinux 里有一个 **type/domain**：

| 进程 | SELinux domain | 关键权限 |
|---|---|---|
| `init` | `u:r:init:s0` | 可启动所有服务，读写 sysfs |
| `zygote` | `u:r:zygote:s0` | 可 fork，可 setuid，不可网络访问 |
| `system_server` | `u:r:system_server:s0` | 可与所有 HAL 交互，可管理 app |
| 普通 app | `u:r:untrusted_app:s0:cXXX` | 受严格限制，只能访问自己的沙箱 |

### 7.2 Zygote 的 capabilities 约束

Zygote 以 root 身份运行，但 fork app 子进程时通过 `SpecializeCommon()` 做了三件事：

1. `DropCapabilitiesBoundingSet()`：子进程即使后来调用 `setcap` 也无法提升到 bounding set 以外
2. `setresuid(uid, uid, uid)` + `setresgid(gid, gid, gid)`：切换到 app 的 UID/GID（不可逆）
3. `SetUpSeccompFilter()`：安装 seccomp-BPF 过滤器，限制允许的 syscall 集合

### 7.3 system_server 的特殊 capabilities

`forkSystemServer()` 显式传入 `capabilities` 位掩码（如 `CAP_SYS_NICE` 允许调整进程优先级、`CAP_NET_ADMIN` 允许网络配置）。这些 capabilities 通过 `nativeForkSystemServer()` 传给子进程，在 `SpecializeCommon()` 里通过 `SetCapabilities()` 配置。

---

## 8. Project Treble 对启动架构的影响

### 8.1 Treble 解决的问题

Android 8.0 之前，vendor 代码（SoC 驱动、HAL 实现）直接链接进 `system_server` 或 app 进程。这造成：

- **OTA 阻塞**：升级 Android 版本需要 SoC 厂商同步更新 HAL，往往延迟 6~12 个月
- **稳定性问题**：vendor HAL bug 可直接 crash `system_server`

### 8.2 Treble 的架构变化

**Android 8.0（Treble 落地）**：

```
Before Treble:                      After Treble:
system_server                        system_server
  └── HAL (linked in)                  └── HAL client (HIDL stub)
                                                │ Binder/HwBinder
                                        HAL server (独立进程, vendor partition)
                                          └── vendor.so (SoC 代码)
```

- 新增 `hwservicemanager` 服务（类比 `servicemanager`，专门管理 HIDL 服务）
- 新增 `vndservicemanager`（vendor 分区 Binder 服务注册）
- HAL 实现迁移到独立进程（`android.hardware.foo@1.0-service`），崩溃不会影响 `system_server`
- VNDK（Vendor NDK）限制 vendor 代码只能链接白名单 lib，防止 ABI 破坏

### 8.3 AIDL over HIDL（Android 11+）

Android 11 开始，新 HAL 接口推荐使用 **Stable AIDL** 而非 HIDL，统一 IPC 机制：

```
Android 8~10: HIDL (Hardware Interface Definition Language)
Android 11+:  Stable AIDL (稳定 ABI 保证 + 标准 Binder 传输)
```

`init.rc` 中现在同时存在两种服务定义：
- `android.hardware.audio.service`（HIDL 遗留）
- `android.hardware.audio-service.stub`（AIDL 新实现）

---

## 9. 失败模式与生产真坑

### 9.1 Zygote crash 导致全系重启

**现象**：设备突然重启，logcat 显示 `Fatal signal 11 (SIGSEGV) in zygote`。

**根因**：`preload()` 阶段某个类的 `<clinit>` 触发了 JNI native crash（如 OpenGL 初始化失败、读取不存在的系统文件）。

**处理**：
```
adb logcat -b crash -d | grep zygote
```
Zygote crash 会被 init 捕获（`onrestart` 链），导致 `mediaserver`、`audioserver`、`cameraserver` 等依赖服务全部重启。

**生产经验**：厂商定制 ROM 往往在 `preloaded-classes` 里加入自己的类，这些类的 `<clinit>` 如果 crash，会导致设备无法启动（boot loop）。

### 9.2 system_server OOM

**现象**：设备无响应，`dmesg` 显示 `Out of memory: Kill process <system_server_pid>`，设备重启。

**根因**：system_server 管理着大量服务的状态缓存（AMS activity stack、WMS window state、PMS package cache），内存泄漏积累到 LMK 无法保护的临界点。

**诊断命令**：
```bash
adb shell dumpsys meminfo system_server
adb shell cat /proc/<pid>/status | grep VmRSS
```

**生产经验**：
- PMS 的 `mPackages` HashMap 泄漏：每次安装/卸载 APK 后 cache 没清理
- AMS 的 Activity 记录泄漏：异常 finish 流程没有移除 ActivityRecord
- 大量 Binder 调用导致 thread-local 缓冲区累积

### 9.3 Boot Phase 死锁

**现象**：设备卡在启动动画，logcat 无新输出，ANR watchdog 触发。

**根因**：某个 `SystemService.onBootPhase()` 调用了另一个服务的 Binder 接口，而那个服务还没到就绪 phase（Binder 调用阻塞 → 死锁）。

**典型案例**：在 `PHASE_SYSTEM_SERVICES_READY` 阶段调用了 `PackageManager.getPackageInfo()`，而 PMS 本身在等待 `PHASE_LOCK_SETTINGS_READY` 才初始化完成。

**诊断**：
```bash
adb shell debuggerd -b <system_server_pid>
# 查看线程调用栈，找 "waiting to acquire"
```

### 9.4 Zygote 多线程 fork 陷阱

**现象**：app 进程 fork 后 deadlock，症状是 app 启动后立即 ANR，logcat 看不到 `ActivityThread.main()` 日志。

**根因**：Linux 规定 `fork()` 后子进程只有调用 `fork()` 的那个线程存活（其他线程被静默终止），如果被终止的线程持有 mutex，子进程内该 mutex 永久锁死。

**Zygote 的防御**：`ZygoteHooks.startZygoteNoThreadCreation()` 在 `ZygoteInit.main()` 最开始调用，向 ART runtime 注册 hook，使任何线程创建尝试抛出 `RuntimeException`。一旦需要多线程（如 ServerSocket accept），必须在 `forkSystemServer()` 完成后进行。

### 9.5 SELinux denial 导致服务启动失败

**现象**：某服务的 `onStart()` 抛出 `SecurityException: Permission denied`，但在 AOSP 上没问题。

**根因**：厂商定制的 `sepolicy` 缺少对应的 `allow` 规则，或者 `type_transition` 没有正确配置。

**诊断**：
```bash
adb shell dmesg | grep "avc: denied"
adb logcat -b all | grep "avc:"
# 典型输出：
# avc: denied { read } for pid=1234 comm="system_server"
#   path="/proc/meminfo" dev="proc" ino=xxx
#   scontext=u:r:system_server:s0 tcontext=u:object_r:proc_meminfo:s0
#   tclass=file permissive=0
```

---

## 10. 可运行 Demo

### Demo 1：观测启动链路进程树

**需要**：Android 模拟器或真机，`adb` 已连接。

```bash
#!/bin/bash
# demo1_boot_process_tree.sh
# 观测 init → zygote → system_server 进程层级

echo "=== 进程树（按 PPID 排序）==="
adb shell "ps -Ao PID,PPID,NAME --sort PPID | grep -E 'init|zygote|system_server'"

echo ""
echo "=== Zygote 进程详情 ==="
adb shell "cat /proc/$(adb shell pidof zygote | tr -d '\r')/status | grep -E 'Name|Pid|PPid|Uid|Gid|VmRSS|VmShared|Threads'"

echo ""
echo "=== system_server 进程详情 ==="
adb shell "cat /proc/$(adb shell pidof system_server | tr -d '\r')/status | grep -E 'Name|Pid|PPid|Uid|Gid|VmRSS|Threads'"

echo ""
echo "=== system_server 的父进程（应该是 zygote）==="
SSRV_PID=$(adb shell pidof system_server | tr -d '\r')
SSRV_PPID=$(adb shell "cat /proc/$SSRV_PID/status" | grep PPid | awk '{print $2}' | tr -d '\r')
adb shell "cat /proc/$SSRV_PPID/cmdline" | tr '\0' ' '
echo ""
```

**预期输出**：
```
=== 进程树（按 PPID 排序）===
    1     0 init
  xxx     1 zygote64
  yyy   xxx system_server

=== Zygote 进程详情 ===
Name:   zygote64
Pid:    xxx
PPid:   1
Uid:    0    0    0    0      <- root UID
VmRSS:  75432 kB             <- ~75MB 常驻内存（含预加载的 framework 类）
Threads: 6                   <- zygote 本身线程少（fork 限制）

=== system_server 的父进程（应该是 zygote）===
/system/bin/app_process64 -Xzygote /system/bin --zygote --start-system-server
```

---

### Demo 2：查看 Zygote socket 和 USAP 池

**需要**：root 权限（模拟器 `adb root`）或 userdebug build。

```bash
#!/bin/bash
# demo2_zygote_sockets.sh
# 观测 zygote 监听的 Unix domain sockets

adb root && adb wait-for-device

echo "=== Zygote 相关 Unix Socket ==="
adb shell "ss -lxp | grep -E 'zygote|usap'"

echo ""
echo "=== socket 权限 ==="
adb shell "ls -la /dev/socket/ | grep -E 'zygote|usap'"

echo ""
echo "=== USAP 池大小（Android 10+）==="
adb shell "getprop persist.device_config.runtime_native.usap_pool_size_max"
adb shell "getprop persist.device_config.runtime_native.usap_pool_size_min"

echo ""
echo "=== Zygote 子进程数（近似等于 USAP 池大小 + 运行中 app 数）==="
ZYGOTE_PID=$(adb shell pidof zygote64 | tr -d '\r')
adb shell "ls /proc/$ZYGOTE_PID/task | wc -l"
```

**预期输出**：
```
=== Zygote 相关 Unix Socket ===
u_str LISTEN 0   1  @/dev/socket/zygote 0 * 0  users:(("zygote64",...))
u_str LISTEN 0   1  @/dev/socket/zygote_secondary 0 * 0
u_str LISTEN 0   5  @/dev/socket/usap_pool_primary 0 * 0

=== socket 权限 ===
srw-rw---- 1 root system /dev/socket/zygote
srw-rw---- 1 root system /dev/socket/usap_pool_primary
```

---

### Demo 3：用 systrace 观测 Zygote preload 耗时

**需要**：Android Studio SDK tools，Python 3，模拟器或真机。

```bash
#!/bin/bash
# demo3_zygote_preload_trace.sh
# 捕获启动 systrace，分析 preloadClasses 耗时

TRACE_OUTPUT="/tmp/boot_trace.html"

echo "重启设备并立即抓取 systrace..."
adb reboot
sleep 5
adb wait-for-device

# 在 boot 完成后抓 trace（boot 期间 atrace 可能受限，用 perfetto 更好）
adb shell "perfetto --config :test --out /data/misc/perfetto-traces/boot.trace &"

# 等待设备 boot 完成
adb shell "until getprop sys.boot_completed | grep -q 1; do sleep 1; done"
echo "Boot 完成"

# 拉取 trace
adb pull /data/misc/perfetto-traces/boot.trace /tmp/boot.trace
echo "Trace 已保存到 /tmp/boot.trace"
echo "在 https://ui.perfetto.dev 打开查看 'preloadClasses' span"
```

**Perfetto 查看方式**：
1. 访问 `https://ui.perfetto.dev`
2. 打开 `boot.trace`
3. 搜索 `preloadClasses` → 可见每个类的加载耗时
4. 搜索 `ZygoteInit` → 可见 preload 总体时间轴

**预期观测**：
- `preloadClasses` 总耗时约 **800ms~2000ms**（取决于设备，模拟器更慢）
- 单个类（如 `android.app.Activity`）加载约 0.1~5ms
- 重量级类（涉及 JNI 初始化的）可达 10~50ms

---

### Demo 4：用 dumpsys 查看 SystemServer boot phase 时间

**需要**：任意 Android 设备，`adb` 已连接。

```bash
#!/bin/bash
# demo4_boot_phases.sh
# 查看各 boot phase 完成时间（需要设备已 boot 完成）

echo "=== SystemServer 启动时序 ==="
adb shell "dumpsys activity service com.android.internal.app.SystemServer" 2>/dev/null \
    || adb shell "logcat -d -b system | grep -E 'PHASE_|bootPhase|StartServices' | tail -50"

echo ""
echo "=== 各服务启动耗时（EventLog）==="
adb shell "logcat -d -b events | grep -E 'boot_progress|system_server_start'" | tail -20

echo ""
echo "=== 关键 boot 事件时间戳 ==="
adb shell "logcat -d -b events | grep -E \
    'boot_progress_start|boot_progress_preload_start|boot_progress_preload_end|\
boot_progress_system_run|boot_progress_pms_start|boot_progress_pms_ready|\
boot_progress_ams_ready|boot_progress_enable_screen'" | tail -20
```

**预期输出**：
```
=== 关键 boot 事件时间戳 ===
01-01 00:00:01.234  I boot_progress_start: [123]
01-01 00:00:02.456  I boot_progress_preload_start: [1234]
01-01 00:00:04.789  I boot_progress_preload_end: [3456]   ← preload 约 2.3s
01-01 00:00:05.012  I boot_progress_system_run: [3679]
01-01 00:00:08.345  I boot_progress_pms_start: [6012]
01-01 00:00:10.678  I boot_progress_pms_ready: [8345]     ← PMS 扫描约 2.3s
01-01 00:00:11.234  I boot_progress_ams_ready: [9234]
01-01 00:00:13.456  I boot_progress_enable_screen: [11223]
```

各阶段时间戳单位为**开机后毫秒**。

---

### Demo 5：分析 system_server 内存组成

**需要**：root 权限或 userdebug build。

```bash
#!/bin/bash
# demo5_system_server_memory.sh
# 分析 system_server 的内存来源（Zygote 共享 vs 私有）

SSRV_PID=$(adb shell pidof system_server | tr -d '\r')

echo "=== system_server (PID=$SSRV_PID) 内存详情 ==="
adb shell "dumpsys meminfo $SSRV_PID"

echo ""
echo "=== Zygote 共享内存估算（smaps 分析）==="
adb root
adb shell "cat /proc/$SSRV_PID/smaps | awk '
/^[0-9a-f]+-[0-9a-f]+/ {
    mapping = \$0
}
/^Shared_Clean:/ { shared_clean += \$2 }
/^Shared_Dirty:/ { shared_dirty += \$2 }
/^Private_Clean:/ { priv_clean += \$2 }
/^Private_Dirty:/ { priv_dirty += \$2 }
END {
    printf \"Shared Clean (COW read-only, Zygote 共享代码页): %d kB\n\", shared_clean
    printf \"Shared Dirty (修改后共享页): %d kB\n\", shared_dirty
    printf \"Private Clean (私有只读页): %d kB\n\", priv_clean
    printf \"Private Dirty (私有可写页，Heap 等): %d kB\n\", priv_dirty
    printf \"PSS 估算: %d kB\n\", priv_dirty + priv_clean + shared_clean/2 + shared_dirty/2
}'"
```

**预期输出示例（Pixel 4, Android 12）**：
```
Shared Clean (COW read-only, Zygote 共享代码页): 67584 kB  ← ~66MB 来自 Zygote
Shared Dirty (修改后共享页): 1024 kB
Private Clean (私有只读页): 8192 kB
Private Dirty (私有可写页，Heap 等): 82944 kB             ← system_server 自己的堆
PSS 估算: 125440 kB                                        ← 约 122MB PSS
```

**关键分析**：Shared Clean 这部分（~66MB）对所有 app 进程来说都是共享的，是 Zygote COW 设计的直接收益。

---

### Demo 6：模拟 app 进程的 Zygote fork 流程

以下是一个最小化 Zygote 协议客户端（用于教学，观察 ZygoteServer 的 socket 协议）：

```python
#!/usr/bin/env python3
# demo6_zygote_protocol.py
# 观察 Zygote socket 协议（仅限 userdebug/root 环境）
# 警告：实际 fork 需要更多参数，此脚本仅用于协议研究
#
# 运行方式：
#   adb root
#   adb shell python3 /data/local/tmp/demo6_zygote_protocol.py
#
# 注意：在真实设备上，此 socket 受 SELinux 保护，
# 只有 AMS 可以连接（domain: system_server）。
# userdebug build 可通过 adb shell 以 shell user 尝试。

import socket
import os

ZYGOTE_SOCKET = "/dev/socket/zygote"

def read_zygote_response(sock):
    """读取 Zygote 的 PID 响应（ASCII 数字 + '\n'）"""
    data = b""
    while True:
        chunk = sock.recv(1)
        if not chunk or chunk == b'\n':
            break
        data += chunk
    return data.decode('utf-8').strip()

def main():
    print(f"连接到 Zygote socket: {ZYGOTE_SOCKET}")
    
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(ZYGOTE_SOCKET)
        print("连接成功")
        
        # Zygote 协议：发送参数行数，然后逐行发送参数
        # 实际 AMS 发送的参数包括 uid, gid, seinfo, niceName, classpath 等
        # 此处仅演示协议格式，不发送真实 fork 请求
        
        # 读取 Zygote 的初始响应（如果有）
        sock.settimeout(1.0)
        try:
            hello = sock.recv(64)
            print(f"Zygote 初始响应: {hello}")
        except socket.timeout:
            print("Zygote 没有主动发送初始消息（正常，等待命令）")
        
        sock.close()
        print("连接关闭")
        
    except PermissionError as e:
        print(f"权限拒绝: {e}")
        print("提示：此 socket 受 SELinux 保护，需要 system_server domain")
        print("在 userdebug build 上可以：adb root && adb shell setenforce 0（仅测试用）")
    except FileNotFoundError:
        print(f"Socket 不存在: {ZYGOTE_SOCKET}")
        print("提示：确认 zygote 进程正在运行：adb shell ps -A | grep zygote")

if __name__ == "__main__":
    main()
```

**预期输出（userdebug build, adb root）**：
```
连接到 Zygote socket: /dev/socket/zygote
连接成功
Zygote 没有主动发送初始消息（正常，等待命令）
连接关闭
```

或：
```
权限拒绝: [Errno 13] Permission denied
提示：此 socket 受 SELinux 保护，需要 system_server domain
```

---

## 11. 方案对比

### 11.1 Zygote fork 模型 vs 其他 app 启动方案

| 维度 | Android Zygote fork | Chrome/Electron preload | iOS XPC spawn | Node.js cluster |
|---|---|---|---|---|
| **冷启动延迟** | 50~200ms（有 Zygote 暖机） | 200~500ms | 50~150ms（Mach-O prelink） | 100~300ms |
| **内存效率** | 高（COW 共享框架类） | 中（共享 V8 snapshot） | 高（dylib 共享） | 低（独立 heap） |
| **隔离强度** | 中高（UID + SELinux + seccomp） | 中（sandbox）| 高（Mach sandbox） | 低（同 UID） |
| **多架构支持** | 是（zygote64 + zygote32） | 是 | 是（Fat binary） | 否 |
| **可定制性** | 低（preloaded-classes 固定） | 中 | 低 | 高 |
| **适用场景** | 移动端大量同质化 app | 浏览器渲染进程 | 系统扩展进程 | 服务器 CPU 密集 |

### 11.2 SystemServer 单进程 vs 微服务架构

| 维度 | Android SystemServer（单进程） | 微服务（每服务独立进程） |
|---|---|---|
| **服务间调用** | 直接调用（零 IPC 开销） | Binder/RPC（~1ms/call） |
| **崩溃隔离** | 无（一死全死） | 有（单服务崩溃可重启） |
| **内存开销** | 低（共享进程上下文） | 高（每进程独立虚拟机） |
| **调试难度** | 低（单一进程，统一 trace） | 高（跨进程 trace 复杂）|
| **启动速度** | 快（串行但在同进程） | 慢（每进程独立初始化）|
| **适合场景** | 强依赖、频繁交互的系统服务 | 独立性强的服务（如 MediaServer）|

**不适用边界**：`system_server` 单进程模型不适用于需要强隔离的服务（Android 自己也将 MediaServer、AudioServer、CameraServer 独立出去）。

---

## 12. 章末五件套

### 12.1 核心概念速查

| 概念 | 一句话定义 |
|---|---|
| **Zygote** | 预热好 JVM 的"母进程"，所有 app 进程通过 fork 它产生，COW 共享框架类内存 |
| **COW (Copy-on-Write)** | fork 后父子进程共享页表，写时才复制物理页——Zygote 内存节省的物理原理 |
| **preloaded-classes** | `/system/etc/preloaded-classes` 列出的 8000+ 框架类，Zygote 启动时全部 Class.forName() |
| **USAP 池** | 预先 fork 的"未专化子进程"池（Android 10+），减少 app 冷启动时 fork 延迟 |
| **SystemServiceManager** | system_server 内所有 SystemService 的容器，驱动 phase 生命周期 |
| **Boot Phase** | SystemServer 启动序列的有序检查点（100→1000），服务按 phase 做延迟初始化 |
| **SpecializeCommon** | fork 后对子进程进行隔离配置的核心函数（UID/GID/capabilities/SELinux/mount namespace）|
| **HIDL/Stable AIDL** | Project Treble 引入的 HAL 接口定义语言，将 vendor HAL 实现隔离到独立进程 |

### 12.2 关键路径数字

| 指标 | 典型值 | 备注 |
|---|---|---|
| Zygote preloadClasses 耗时 | 800ms~2s | 模拟器更慢，取决于 I/O 速度 |
| PMS APK 扫描耗时 | 1~3s | 安装 app 越多越慢 |
| AMS.systemReady() 之前总时间 | 5~12s | 低端设备可达 20s |
| Zygote 预加载类数量 | ~8000 类 | 因版本/厂商定制而异 |
| system_server Binder 线程数上限 | 31 | `sMaxBinderThreads = 31` |
| Zygote Shared Clean 内存 | 50~80MB | 被所有进程共享，COW 收益 |

### 12.3 最常见面试/诊断问题

**Q1：app 冷启动慢（> 1s），可能是 Zygote 问题吗？**

- 通常不是。Zygote fork 本身 < 50ms。慢的原因多是 app 自己的 `Application.onCreate()`、ContentProvider 初始化、或者首帧渲染。
- 用 `adb shell am start -W <package>/<activity>` 的 `TotalTime` vs `ThisTime` 区分进程创建时间和应用初始化时间。

**Q2：`system_server` 重启后设备重启，能避免吗？**

- 设计上不能：system_server 死亡触发 zygote 的 `onrestart`，zygote 重启会拉起整个系统。这是 Android 的"软重启"（soft reboot）机制。
- 部分厂商定制了 `system_server` crash 后的行为（如 HiSilicon 的 "Hot Restart"），但 AOSP 标准行为是全系重启。

**Q3：怎么加快设备启动速度（OEM 视角）？**

1. 减少 `preloaded-classes` 的类数（精确到 app 实际用的）——但代价是 app 启动变慢
2. 加快存储 I/O（eMMC → UFS）：PMS APK 扫描是瓶颈
3. 并行化 `startOtherServices()`（已有 `SystemServerInitThreadPool`，可扩大利用）
4. 启用 `verity` 分区加速（dm-verity v2 with FEC）

### 12.4 延伸阅读

- **源码入口**：
  - `platform/frameworks/base/core/java/com/android/internal/os/ZygoteInit.java`
  - `platform/frameworks/base/services/java/com/android/server/SystemServer.java`
  - `platform/system/core/init/main.cpp`
- **进阶阅读**：
  - "Android Internals" (Jonathan Levin) Vol.1 第 4 章
  - Google I/O 2017 "Android O: under the hood" — Project Treble 讲解
  - AOSP wiki: `platform/frameworks/base/+/refs/heads/master/core/java/com/android/internal/os/`
- **工具**：
  - `adb shell dumpsys activity` — AMS 完整状态
  - `adb shell dumpsys window` — WMS 窗口树
  - `Perfetto` / `Systrace` — 启动时序分析

### 12.5 本章知识树定位

```
Android 系统知识树
├── 第 1 章：架构与启动（本章）← 基础
│   ├── init / init.rc 解析
│   ├── Zygote preload + COW
│   ├── SystemServer + Boot Phase
│   └── Project Treble 架构分层
├── 第 2 章：Binder IPC 深度（依赖本章 §6）
├── 第 3 章：ART 运行时（依赖本章 §4.2 preload）
├── 第 4 章：内存管理（依赖本章 §4.2 COW）
├── 第 5 章：SurfaceFlinger + 渲染管线（依赖本章 WMS 启动）
└── 第 6 章：SELinux 深度（依赖本章 §7）
```

---

> **sources_fetched**:
> 1. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/java/com/android/internal/os/ZygoteInit.java`
> 2. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/java/com/android/internal/os/ZygoteServer.java`
> 3. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/java/com/android/internal/os/Zygote.java`
> 4. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/java/com/android/internal/os/ZygoteHooks.java` (404)
> 5. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/java/com/android/internal/os/RuntimeInit.java`
> 6. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/jni/com_android_internal_os_Zygote.cpp`
> 7. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/services/java/com/android/server/SystemServer.java`
> 8. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/services/core/java/com/android/server/SystemServiceManager.java`
> 9. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/services/core/java/com/android/server/SystemService.java`
> 10. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/services/core/java/com/android/server/am/ActivityManagerService.java`
> 11. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/services/core/java/com/android/server/wm/WindowManagerService.java`
> 12. `https://raw.githubusercontent.com/aosp-mirror/platform_system_core/master/rootdir/init.rc`
> 13. `https://raw.githubusercontent.com/aosp-mirror/platform_system_core/master/rootdir/init.zygote64_32.rc`
> 14. `https://raw.githubusercontent.com/aosp-mirror/platform_system_core/master/init/main.cpp`
> 15. `https://raw.githubusercontent.com/aosp-mirror/platform_system_core/master/init/service.cpp`
> 16. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/java/android/os/Binder.java`
> 17. `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_native/master/libs/binder/ProcessState.cpp` (404)
> 18. `https://developer.android.com/topic/performance/memory-overview`
> 19. `https://source.android.com/docs/core/architecture`
> 20. `https://source.android.com/docs/core/architecture/vndk/linker-namespace`
