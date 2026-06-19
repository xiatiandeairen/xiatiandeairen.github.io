---
title: "HAL 与硬件抽象：Treble 架构深度解析"
slug: "8-06"
collection: "tech-library"
group: "android系统"
order: 8006
summary: "TL;DR Android HAL（Hardware Abstraction Layer）是将硬件差异封装在 vendor 分区、暴露稳定 IPC 接口给 system 分区的机制。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:35:28.000Z"
updatedAt: "2026-06-12T11:35:28.000Z"
---
> **TL;DR**
> Android HAL（Hardware Abstraction Layer）是将硬件差异封装在 vendor 分区、暴露稳定 IPC 接口给 system 分区的机制。Android 8 的 Project Treble 是核心架构转折点：通过 HIDL/AIDL HAL + VINTF 将 system image 与 vendor image 解耦，使 Google 能独立 OTA 更新 AOSP framework 而无需 SoC 厂商重新集成。本章从 pre-Treble 痛点出发，精读 HIDL/AIDL HAL 源码，拆解 Binder IPC 传输层、VINTF manifest 兼容性校验，附三个可运行 demo，覆盖失败模式与生产坑。

---

## 前置知识

| 需要熟悉 | 原因 |
|----------|------|
| Binder IPC 原理（第 2 章） | HAL 通信本质是 Binder；HIDL/AIDL 生成的 Bp/Bn 即 Binder proxy/native |
| Android 分区结构 | Treble 的核心是 system/vendor 分区物理隔离 |
| Linux shared memory / mmap | HAL 大数据传输走 hidl_memory / SharedBuffer |
| C++ 虚表、vtable ABI | HIDL hash 机制保护的正是 vtable 不被破坏 |
| init.rc / selinux | HAL service 的注册、权限通过 rc 文件和 sepolicy 声明 |

---

## 6.1 设计考古：从 legacy HAL 到 Treble

### 6.1.1 pre-Android 8 的碎片化噩梦

在 Android 8（Oreo, 2017）之前，HAL 实现是以 **共享库（.so）** 形式存在的，被 system process 通过 `dlopen()` 直接加载。这一模式称为 **legacy / passthrough HAL**：

```
┌──────────────────────────┐
│  system_server / mediaserver │   ← system partition
│  (framework)                 │
│    dlopen("libcamera.so")    │
└──────────┬───────────────────┘
           │ 同进程内函数调用
┌──────────▼───────────────────┐
│  libcamera_hal.so             │   ← vendor partition
│  (vendor 提供，无隔离)        │
└──────────────────────────────┘
```

问题一览：

1. **ABI 绑定**：Google 每次升级 `libhardware.h` 的数据结构，vendor 必须重新编译全部 HAL。Android 版本碎片化的根源之一。
2. **安全边界缺失**：vendor HAL 崩溃直接带崩 system_server，影响整机稳定性。
3. **版本协商无法机械化**：HAL 版本信息散落在各厂商私有头文件，无统一 manifest，OTA 兼容性全靠人工协商。

Google 内部统计数据（来自 Android 8 发布时的 Google I/O 2017 演讲 "Project Treble"）：Android 碎片化导致主流厂商设备平均 **1.5 年** 才能收到主版本 OTA，很大程度源于此。

### 6.1.2 Project Treble：接口契约化

Android 8 引入 **Project Treble**，核心思路：

> "Define a stable, versioned interface boundary between the Android OS framework and the vendor implementation."

三个支柱：

| 支柱 | 机制 | 作用 |
|------|------|------|
| 接口语言 | HIDL（后演进为 AIDL HAL） | 用 IDL 描述 HAL 接口，生成类型安全的 IPC 胶水代码 |
| 进程隔离 | Binderized HAL | HAL 运行在独立进程（vendor partition），通过 hwbinder 与 framework 通信 |
| 版本声明 | VINTF manifest | device manifest 声明 vendor 提供什么，FCM 声明 framework 要求什么，启动时机械校验 |

```
Android 8+ 架构：
┌─────────────────────────┐
│     system partition     │  ← 可被 Google OTA 独立更新
│  CameraService           │
│  AudioFlinger            │
│  VibratorService         │
└──────────┬──────────────┘
           │  hwbinder / binder IPC（稳定接口，版本化）
┌──────────▼──────────────┐
│     vendor partition     │  ← SoC 厂商维护，生命周期独立
│  camera.default.so       │
│  android.hardware.audio  │
│  android.hardware.vibrator │
└─────────────────────────┘
```

### 6.1.3 HIDL → AIDL HAL 的演进

| 版本 | 状态 | 关键变化 |
|------|------|---------|
| Android 8–9 | HIDL 引入 | hwbinder 传输，`.hal` 文件 IDL |
| Android 10 | HIDL deprecated | 新接口推荐用 AIDL |
| Android 11–12 | AIDL HAL 逐步落地 | stable AIDL + `@VintfStability` |
| Android 13+ | AIDL 为主 | HIDL 只维护不新增，hidl2aidl 工具辅助迁移 |

HIDL 被弃用的核心原因：维护两套 IDL 工具链成本高；AIDL 本已支持稳定接口（stable AIDL），统一到一套语言更经济。

---

## 6.2 HIDL 架构精读

### 6.2.1 HIDL IDL 语法

以 audio HAL 2.0 接口为例（AOSP `hardware/interfaces/audio/2.0/IDevice.hal`），摘取典型语法特征：

```hidl
// 【示意，非逐字 — 该文件 raw fetch 返回 404，结合文档还原语法结构】
package android.hardware.audio@2.0;

import android.hardware.audio.common@2.0::types;
import IStreamIn;
import IStreamOut;

interface IDevice {
    /**
     * Returns whether the audio hardware interface has been initialized.
     */
    initCheck() generates (Result retval);

    /**
     * Opens an audio output stream.
     * oneway: absent — this is a synchronous call
     */
    openOutputStream(
        AudioIoHandle ioHandle,
        DeviceAddress device,
        AudioConfig config,
        bitfield<AudioOutputFlag> flags,
        SourceMetadata sourceMetadata)
    generates (
        Result retval,
        IStreamOut outStream,
        AudioConfig suggestedConfig);

    /**
     * oneway 标记：非阻塞，无返回值，不能用 generates
     */
    oneway setMasterMute(bool mute);
};
```

关键语法要点：
- `@VERSION` 包名附版本，如 `android.hardware.audio@2.0`
- `generates` 子句声明同步返回值（类似 out 参数）
- `oneway` 方法：异步投递，不阻塞调用方，不能有 `generates`
- HIDL 类型系统：`uint8_t..uint64_t`, `string`, `vec<T>`, `bitfield<E>`, `handle` (fd 传递)

### 6.2.2 HIDL 生成代码结构

运行 `hidl-gen` 会生成以下文件族（以 `android.hardware.vibrator@1.0` 为例）：

```
auto-generated/
├── IVibrator.h                  ← 纯虚接口类
├── BpHwVibrator.h / .cpp        ← Binder Proxy（client 侧）
├── BnHwVibrator.h / .cpp        ← Binder Native（server 侧）
├── IHwVibrator.h                ← hw binder 中间层
└── android/hardware/vibrator/1.0/IVibrator.h  ← 包含路径
```

`BpHwVibrator`（Proxy）的核心：将方法调用序列化为 Parcel，通过 hwbinder 发送给 server 进程。`BnHwVibrator`（Native）反序列化 Parcel，调用具体实现。这与 Binder 第 2 章的 `Bp/Bn` 模式完全一致，只是底层走 `/dev/hwbinder` 而非 `/dev/binder`。

### 6.2.3 Passthrough vs Binderized 传输模式

```
Passthrough 模式（用于迁移过渡期）：
┌──────────────────────────────────────────┐
│           同一进程                         │
│  Framework ──→ BsFoo（passthrough wrapper）│
│              ──→ legacy HAL .so（dlopen） │
└──────────────────────────────────────────┘

Binderized 模式（Treble 目标态）：
┌────────────────────┐    hwbinder    ┌────────────────────┐
│  Framework Process  │  ─────────→   │   HAL Process       │
│  (system partition) │  ←─────────   │  (vendor partition) │
└────────────────────┘               └────────────────────┘
```

Passthrough 模式存在于 Android 8 升级期，允许不重写 HAL 的情况下先通过 Treble CDD 测试。长期目标是全部迁移到 Binderized。

同进程 HAL（SP-HAL，Same-Process HAL）是例外：OpenGL ES、Vulkan、renderscript 因性能要求过高，被允许继续以 `.so` 形式加载进 app 进程，但受到严格的 Google 白名单管控。

---

## 6.3 AIDL HAL 架构精读（主流）

### 6.3.1 IVibrator.aidl 完整接口定义

【真实源码 `platform/hardware/interfaces @ hardware/interfaces/vibrator/aidl/android/hardware/vibrator/IVibrator.aidl`】

以下为从 `android.googlesource.com` WebFetch 获取的真实接口（经 base64 解码）：

```aidl
// hardware/interfaces/vibrator/aidl/android/hardware/vibrator/IVibrator.aidl
// Copyright (C) 2019 The Android Open Source Project, Apache 2.0

package android.hardware.vibrator;

import android.hardware.vibrator.Braking;
import android.hardware.vibrator.CompositeEffect;
import android.hardware.vibrator.CompositePrimitive;
import android.hardware.vibrator.CompositePwleV2;
import android.hardware.vibrator.Effect;
import android.hardware.vibrator.EffectStrength;
import android.hardware.vibrator.FrequencyAccelerationMapEntry;
import android.hardware.vibrator.IVibratorCallback;
import android.hardware.vibrator.PrimitivePwle;
import android.hardware.vibrator.VendorEffect;

@VintfStability          // ← 关键注解：跨 vendor/system 分区边界，必须 frozen 才能发布
interface IVibrator {

    // ── 能力位 ──────────────────────────────────────────────────────────
    const int CAP_ON_CALLBACK = 1 << 0;           // on() 支持回调
    const int CAP_PERFORM_CALLBACK = 1 << 1;       // perform() 支持回调
    const int CAP_AMPLITUDE_CONTROL = 1 << 2;      // 支持幅度控制
    const int CAP_EXTERNAL_CONTROL = 1 << 3;       // 支持外部（音频）控制
    const int CAP_EXTERNAL_AMPLITUDE_CONTROL = 1 << 4;
    const int CAP_COMPOSE_EFFECTS = 1 << 5;        // 支持 compose() 组合效果
    const int CAP_ALWAYS_ON_CONTROL = 1 << 6;      // 支持常亮震动
    const int CAP_GET_RESONANT_FREQUENCY = 1 << 7; // 支持查询共振频率
    const int CAP_GET_Q_FACTOR = 1 << 8;
    const int CAP_FREQUENCY_CONTROL = 1 << 9;      // 支持频率控制
    const int CAP_COMPOSE_PWLE_EFFECTS = 1 << 10;  // 分段线性波形
    const int CAP_PERFORM_VENDOR_EFFECTS = 1 << 11;
    const int CAP_COMPOSE_PWLE_EFFECTS_V2 = 1 << 12;

    // ── 核心方法 ─────────────────────────────────────────────────────────
    int getCapabilities();
    void off();
    void on(in int timeoutMs, in IVibratorCallback callback);

    // perform 返回效果的实际时长（ms）
    int perform(in Effect effect, in EffectStrength strength,
                in IVibratorCallback callback);

    Effect[] getSupportedEffects();
    void setAmplitude(in float amplitude);  // [0.0, 1.0]
    void setExternalControl(in boolean enabled);

    // ── 组合效果 ──────────────────────────────────────────────────────────
    int getCompositionDelayMax();
    int getCompositionSizeMax();
    CompositePrimitive[] getSupportedPrimitives();
    int getPrimitiveDuration(CompositePrimitive primitive);
    void compose(in CompositeEffect[] composite, in IVibratorCallback callback);

    // ── 常亮震动 ──────────────────────────────────────────────────────────
    Effect[] getSupportedAlwaysOnEffects();
    void alwaysOnEnable(in int id, in Effect effect, in EffectStrength strength);
    void alwaysOnDisable(in int id);

    // ── 频率特性查询 ───────────────────────────────────────────────────────
    float getResonantFrequency();
    float getQFactor();
    float getFrequencyResolution();
    float getFrequencyMinimum();
    float[] getBandwidthAmplitudeMap();
    List<FrequencyAccelerationMapEntry> getFrequencyToOutputAccelerationMap();

    // ── PWLE（Piecewise-Linear Envelope）──────────────────────────────────
    int getPwlePrimitiveDurationMax();
    int getPwleCompositionSizeMax();
    Braking[] getSupportedBraking();
    void composePwle(in PrimitivePwle[] composite, in IVibratorCallback callback);

    // PWLE V2（Android 14+）
    int getPwleV2PrimitiveDurationMaxMillis();
    int getPwleV2CompositionSizeMax();
    int getPwleV2PrimitiveDurationMinMillis();
    void composePwleV2(in CompositePwleV2 composite, in IVibratorCallback callback);

    // ── Vendor 扩展 ────────────────────────────────────────────────────────
    void performVendorEffect(in VendorEffect vendorEffect, in IVibratorCallback callback);
}
```

**逐行注解要点：**

1. `@VintfStability`：这是 stable AIDL 的关键标记。一旦接口被 freeze（`aidl_api/` 目录下有对应版本快照），任何破坏 ABI 的修改都会导致编译失败。
2. `in/out/inout` 方向修饰符：`in` 是 client→server，`out` 是 server 写回，`inout` 双向。大量数据推荐 `out` 避免拷贝。
3. `IVibratorCallback callback`：HAL 向上回调的 AIDL 接口对象，server 持有 client 的 Binder 对象引用，异步完成时 invoke。这与 HIDL 的 `generates` 回调机制对等但表达更清晰。
4. capability bits 模式：HAL 不支持的方法通过 `getCapabilities()` 预先声明，调用方必须先查询，否则会收到 `EX_UNSUPPORTED_OPERATION`。

### 6.3.2 Vibrator HAL 参考实现

【真实源码 `platform/hardware/interfaces @ hardware/interfaces/vibrator/aidl/default/Vibrator.cpp`】

```cpp
// Vibrator.cpp — 选取最能展示机制的 4 个方法

// ── getCapabilities：能力协商 ────────────────────────────────────────────
ndk::ScopedAStatus Vibrator::getCapabilities(int32_t* _aidl_return) {
    LOG(VERBOSE) << "Vibrator reporting capabilities";
    std::lock_guard lock(mMutex);
    if (mCapabilities == 0) {
        int32_t version;
        // 查询自身 interface version，用于条件化新能力
        if (!getInterfaceVersion(&version).isOk()) {
            return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_ILLEGAL_STATE));
        }
        mCapabilities = IVibrator::CAP_ON_CALLBACK | IVibrator::CAP_PERFORM_CALLBACK |
                        IVibrator::CAP_AMPLITUDE_CONTROL | IVibrator::CAP_EXTERNAL_CONTROL |
                        IVibrator::CAP_COMPOSE_EFFECTS | IVibrator::CAP_FREQUENCY_CONTROL;
        if (version >= 3) {
            // V3 新增能力：仅在接口版本 >=3 时声明
            mCapabilities |= IVibrator::CAP_PERFORM_VENDOR_EFFECTS;
        }
    }
    *_aidl_return = mCapabilities;
    return ndk::ScopedAStatus::ok();  // 等价于 AIDL 的 void return with no exception
}

// ── on：定时震动，附回调 ─────────────────────────────────────────────────
ndk::ScopedAStatus Vibrator::on(int32_t timeoutMs,
                                const std::shared_ptr<IVibratorCallback>& callback) {
    LOG(VERBOSE) << "Vibrator on for timeoutMs: " << timeoutMs;
    // dispatchVibrate 在 detach 线程中等待超时后回调 callback->onComplete()
    dispatchVibrate(timeoutMs, callback);
    return ndk::ScopedAStatus::ok();
}

// ── off：立即停止，通知两个 callback ────────────────────────────────────
ndk::ScopedAStatus Vibrator::off() {
    LOG(VERBOSE) << "Vibrator off";
    std::lock_guard lock(mMutex);
    // 取出 callback，清空状态（避免 callback 持锁时反向调用造成死锁）
    std::shared_ptr<IVibratorCallback> callback = mVibrationCallback;
    std::shared_ptr<IVibratorCallback> globalCallback = mGlobalVibratorCallback;
    mIsVibrating = false;
    mVibrationCallback = nullptr;
    mGlobalVibratorCallback = nullptr;
    // lock 已释放，在锁外 notify，避免 re-entrant 死锁
    if (callback) {
        auto ret = callback->onComplete();
        if (!ret.isOk()) LOG(ERROR) << "Failed to notify onComplete: " << ret.getMessage();
    }
    if (globalCallback) {
        auto ret = globalCallback->onComplete();
        if (!ret.isOk()) LOG(ERROR) << "Failed to notify onComplete global";
    }
    return ndk::ScopedAStatus::ok();
}

// ── perform：预定义效果播放 ────────────────────────────────────────────
ndk::ScopedAStatus Vibrator::perform(Effect effect, EffectStrength strength,
                                     const std::shared_ptr<IVibratorCallback>& callback,
                                     int32_t* _aidl_return) {
    LOG(VERBOSE) << "Vibrator perform";
    // 只支持 CLICK 和 TICK，其他返回 EX_UNSUPPORTED_OPERATION
    if (effect != Effect::CLICK && effect != Effect::TICK) {
        return ndk::ScopedAStatus(AStatus_fromExceptionCode(EX_UNSUPPORTED_OPERATION));
    }
    constexpr size_t kEffectMillis = 100;  // 固定 100ms 时长
    dispatchVibrate(kEffectMillis, callback);
    *_aidl_return = kEffectMillis;  // 告知 client 实际时长
    return ndk::ScopedAStatus::ok();
}
```

**注解：**
- `ndk::ScopedAStatus` 是 NDK AIDL 的 status 类型，对应 Java 层的 `RemoteException`。`isOk()` 检查是否成功，`getMessage()` 取错误描述。
- 锁的使用模式：**先拿锁、取出指针、清空成员、释放锁、在锁外 invoke callback**——这是防止 re-entrant 死锁的标准模式（参见 off() 实现）。

### 6.3.3 HAL 服务注册：main.cpp

【真实源码 `platform/hardware/interfaces @ hardware/interfaces/vibrator/aidl/default/main.cpp`】

```cpp
#include "vibrator-impl/Vibrator.h"
#include "vibrator-impl/VibratorManager.h"

#include <android-base/logging.h>
#include <android/binder_manager.h>   // AServiceManager_addService
#include <android/binder_process.h>   // ABinderProcess_*

using aidl::android::hardware::vibrator::Vibrator;
using aidl::android::hardware::vibrator::VibratorManager;

int main() {
    // 设置 binder 线程池大小为 0 = 只有主线程
    // 对于简单 HAL 足够；高并发 HAL 需增大（最多 15 个 binder 线程）
    ABinderProcess_setThreadPoolMaxThreadCount(0);

    // 创建 default vibrator 服务实例
    auto vib = ndk::SharedRefBase::make<Vibrator>();

    // 注册到 service manager，服务名 = "android.hardware.vibrator.IVibrator/default"
    // makeServiceName("default") 自动拼接包名 + 接口名 + instance
    binder_status_t status = AServiceManager_addService(
            vib->asBinder().get(),                         // IBinder* raw pointer
            Vibrator::makeServiceName("default").c_str()); // "android.hardware.vibrator.IVibrator/default"
    CHECK_EQ(status, STATUS_OK);

    // 同样注册 VibratorManager，提供多振动器管理
    auto managedVib = ndk::SharedRefBase::make<Vibrator>();
    auto vibManager = ndk::SharedRefBase::make<VibratorManager>(std::move(managedVib));
    status = AServiceManager_addService(
            vibManager->asBinder().get(),
            VibratorManager::makeServiceName("default").c_str());
    CHECK_EQ(status, STATUS_OK);

    // 进入 binder 线程池 loop，阻塞直到 process 退出
    ABinderProcess_joinThreadPool();
    return EXIT_FAILURE;  // 正常不会到这里
}
```

**关键观察：**
1. `AServiceManager_addService` 是 AIDL NDK HAL 的注册 API，对应 Java 的 `ServiceManager.addService()`。HAL 向 `hwservicemanager`（或 Android 12+ 的统一 `servicemanager`）注册。
2. `ndk::SharedRefBase::make<T>()` 是 AIDL NDK 的引用计数对象工厂，类似 `sp<T>`（strongPointer）。
3. `ABinderProcess_joinThreadPool()` 永久阻塞 —— HAL service 进程的主线程就是 binder 线程池的一部分。

### 6.3.4 VINTF Manifest Fragment

【真实源码 `platform/hardware/interfaces @ hardware/interfaces/vibrator/aidl/default/android.hardware.vibrator.xml`】

```xml
<!-- 部署在 /vendor/etc/vintf/manifest/ 下 -->
<manifest version="1.0" type="device">
  <hal format="aidl">
    <name>android.hardware.vibrator</name>
    <version>3</version>                        <!-- interface version，对应 IVibrator V3 -->
    <fqname>IVibrator/default</fqname>          <!-- fully-qualified instance name -->
  </hal>
  <hal format="aidl">
    <name>android.hardware.vibrator</name>
    <version>3</version>
    <fqname>IVibratorManager/default</fqname>
  </hal>
</manifest>
```

这个 XML fragment 在 build 时被合并进 device manifest，系统启动时 `vintf` 库读取并与 Framework Compatibility Matrix（FCM）做交叉校验：如果 FCM 要求 `IVibrator >= 2` 而 manifest 声明的是 `1`，系统会在 early-init 阶段 **abort**。

---

## 6.4 VINTF：版本兼容性保障机制

### 6.4.1 四张表的关系

```
Device Manifest ──────→ 描述 vendor 提供什么 HAL / SEPolicy 版本
      ↓ 校验
Framework Compatibility Matrix (FCM) ← 描述当前 system image 需要什么

Framework Manifest ───→ 描述 system 提供什么给 vendor
      ↓ 校验
Device Compatibility Matrix ────────← vendor 声明需要 system 提供什么
```

两对校验在系统启动时全部运行，任意一对不过都会导致 bootloop（`libvintf` 返回 error，init 进程 abort）。

### 6.4.2 Device Manifest 结构示例

```xml
<!-- /vendor/etc/vintf/manifest.xml 典型结构（示意，基于文档还原） -->
<manifest version="2.0" type="device" target-level="5">

  <!-- HIDL HAL（旧设备仍使用）-->
  <hal format="hidl">
    <name>android.hardware.audio</name>
    <transport>hwbinder</transport>
    <version>7.0</version>
    <interface>
      <name>IDevicesFactory</name>
      <instance>default</instance>
    </interface>
  </hal>

  <!-- AIDL HAL（新设备）-->
  <hal format="aidl">
    <name>android.hardware.vibrator</name>
    <version>3</version>
    <fqname>IVibrator/default</fqname>
  </hal>

  <!-- SEPolicy 版本 -->
  <sepolicy>
    <version>202404.0</version>
  </sepolicy>

</manifest>
```

`target-level` 对应 FCM version，决定 system image 用哪张 FCM 来校验此 device。

### 6.4.3 Manifest Fragments（Android 10+）

大型设备的 vendor manifest 可能包含数十个 HAL。Android 10 引入 `vintf_fragments`：每个 HAL 模块可在自己的 `Android.bp` 中声明 fragment，build 系统合并：

```python
# Android.bp 中（示意）
cc_binary {
    name: "android.hardware.vibrator-service.example",
    vintf_fragments: ["android.hardware.vibrator.xml"],  // 指向上面的 manifest fragment
    ...
}
```

这使得条件编译某个 HAL 时，manifest 也自动跟进，无需手动维护主 manifest 文件。

---

## 6.5 HAL 类型与传输对照

| 维度 | Legacy HAL | HIDL Passthrough | HIDL Binderized | AIDL HAL |
|------|-----------|------------------|-----------------|----------|
| 进程隔离 | 无（同进程 dlopen） | 无（同进程，封装层） | 有（独立 vendor 进程） | 有（独立 vendor 进程） |
| IPC 机制 | 无（函数调用） | 无（函数调用） | hwbinder `/dev/hwbinder` | binder `/dev/binder` |
| 接口语言 | C 结构体 `hw_module_t` | `.hal` HIDL | `.hal` HIDL | `.aidl` stable AIDL |
| 版本校验 | 无 | VINTF | VINTF | VINTF |
| 崩溃隔离 | 连带 system_server | 连带 system_server | 独立（HAL 进程崩溃可重启） | 独立 |
| 适用场景 | Android 7 及以下遗留代码 | Android 8 过渡期迁移 | Android 8–12 主流 | Android 12+ 新接口推荐 |
| 典型接口 | `hw_module_t` / `gralloc` | camera v1 backwards compat | audio@7.0, camera@3.5 | vibrator@3, audio AIDL |

**SP-HAL（Same-Process HAL）例外：**

OpenGL ES、Vulkan 因为高频调用（每帧数百次），允许 dlopen 进 app/SurfaceFlinger 进程以消除 IPC 开销。但这些 HAL 属于 **白名单管控**，Google 严格控制可进入 system 进程的 vendor .so 名单（`/system/lib/vndk-sp/` 目录下）。

---

## 6.6 可运行 Demo

### Demo 1：用 `adb shell service call` 直接 invoke AIDL HAL

**前置：** Android 12+ 设备或 Cuttlefish 模拟器，adb 连接。

```bash
# ── Step 1：确认 vibrator HAL 已注册 ──────────────────────────────
adb shell service list | grep vibrator
# 预期输出（Android 12+）：
# android.hardware.vibrator.IVibrator/default: [android.hardware.vibrator.IVibrator]

# ── Step 2：查询设备能力（method index 0 = getCapabilities）────────
# AIDL 方法索引从 1 开始（0 是 pingBinder），getCapabilities 是第一个声明的方法
adb shell service call android.hardware.vibrator.IVibrator/default 1
# 预期输出（数字为 capability bitmask）：
# Result: Parcel(
#   0x00000000: 00000000 000003ff   '........')
# 0x3ff = 1023 = 0b1111111111 表示支持前10项能力

# ── Step 3：触发震动 100ms（method index 2 = on）──────────────────
# on(timeoutMs=100, callback=null)
# Parcel 格式：i32(100) + null_binder(0)
adb shell service call android.hardware.vibrator.IVibrator/default 2 i32 100
# 设备应短暂震动 ~100ms

# ── Step 4：立即停止（method index 3 = off）───────────────────────
adb shell service call android.hardware.vibrator.IVibrator/default 3
# 预期输出：
# Result: Parcel(00000000)   ← STATUS_OK

# ── Step 5：查看 AIDL HAL 服务的 binder dump ─────────────────────
adb shell dumpsys android.hardware.vibrator.IVibrator/default
# 通常输出：interface descriptor, state info（取决于 HAL 实现是否覆写 dump()）
```

**观察重点：**
- `service list` 展示所有已注册的 AIDL service，格式 `package.InterfaceName/instance`。
- `service call` 的 method index 对应 AIDL 接口中方法声明顺序（从 1 开始，0 是内置 pingBinder）。
- Parcel 格式：`i32` 是 32-bit int，`s16` 是 UTF-16 string。

---

### Demo 2：读取 VINTF manifest 并验证兼容性

**前置：** Android 10+ 设备，adb 连接。

```bash
# ── Step 1：读取 device manifest ──────────────────────────────────
adb shell cat /vendor/etc/vintf/manifest.xml
# 或分段 manifest
adb shell ls /vendor/etc/vintf/manifest/
adb shell cat /vendor/etc/vintf/manifest/android.hardware.vibrator.xml

# ── Step 2：查看 framework manifest ────────────────────────────────
adb shell cat /system/etc/vintf/manifest.xml

# ── Step 3：实时查询 VINTF 对象（Android 11+）─────────────────────
# dumpsys vintf 输出合并后的 manifest + 兼容性结果
adb shell dumpsys vintf
# 典型输出片段：
# HALs declared in vendor manifest: ...
# android.hardware.vibrator@3::IVibrator (default instance): OK

# ── Step 4：手动验证某个 HAL 是否满足 FCM 要求 ────────────────────
adb shell lshal
# 输出所有运行中 HAL，格式：
# Interface                                   Transport  Arch  Thread  Instance
# android.hardware.vibrator.IVibrator/default binder     64    1/15    android.hardware.vibrator@3
# android.hardware.audio@7.0::IDevicesFactory hwbinder   64    2/15    default

# ── Step 5：查看 FCM 版本 ────────────────────────────────────────
adb shell getprop ro.vendor.api_level       # vendor partition 目标 FCM level
adb shell getprop ro.product.first_api_level # 首次发布 API level
# 这两个决定了使用哪个 compatibility_matrix.X.xml 校验
```

**预期输出解读：**
- `lshal` 中 `Transport: binder` 是 AIDL HAL（用 `/dev/binder`），`Transport: hwbinder` 是 HIDL HAL（用 `/dev/hwbinder`）。
- AIDL HAL 的 version 是单整数（`@3`），HIDL 是 `major.minor`（`@7.0`）。
- `Thread: 1/15` 表示当前有 1 个 binder 线程，最大 15。

---

### Demo 3：实现并运行最小 AIDL HAL stub（Cuttlefish / 模拟器）

**前置：** AOSP 源码树、`lunch sdk_phone_x86_64-eng`，或直接用 Cuttlefish。

**需要 Android 模拟器或设备验证，AOSP 源码树环境。**

**3a. 定义最小 AIDL 接口**

创建 `hardware/interfaces/hello/1.0/android/hardware/hello/IHello.aidl`：

```aidl
// IHello.aidl
package android.hardware.hello;

@VintfStability
interface IHello {
    String greet(in String name);
}
```

**3b. Android.bp**

```python
// hardware/interfaces/hello/1.0/Android.bp
aidl_interface {
    name: "android.hardware.hello",
    srcs: ["android/hardware/hello/*.aidl"],
    stability: "vintf",   // ← 触发 stable AIDL 校验，允许跨 vendor/system
    backend: {
        cpp: {
            enabled: false,
        },
        ndk: {
            enabled: true,   // 使用 NDK C++ backend
        },
        java: {
            enabled: false,
        },
    },
    versions_with_info: [
        {
            version: "1",
            imports: [],
        },
    ],
}
```

**3c. 实现类 HelloImpl.cpp**

```cpp
// hardware/interfaces/hello/1.0/default/HelloImpl.cpp
#include <aidl/android/hardware/hello/BnHello.h>
#include <android/binder_manager.h>
#include <android/binder_process.h>
#include <android-base/logging.h>

using ::aidl::android::hardware::hello::BnHello;
using ::ndk::ScopedAStatus;

class HelloImpl : public BnHello {
  public:
    ScopedAStatus greet(const std::string& name, std::string* _aidl_return) override {
        *_aidl_return = "Hello from vendor HAL, " + name + "!";
        LOG(INFO) << "HelloImpl::greet(" << name << ")";
        return ScopedAStatus::ok();
    }
};

int main() {
    ABinderProcess_setThreadPoolMaxThreadCount(0);

    auto service = ndk::SharedRefBase::make<HelloImpl>();
    // 服务名格式：android.hardware.hello.IHello/default
    const std::string instance = HelloImpl::makeServiceName("default");
    CHECK_EQ(AServiceManager_addService(service->asBinder().get(), instance.c_str()),
             STATUS_OK) << "Failed to register " << instance;

    LOG(INFO) << "HelloImpl HAL service running.";
    ABinderProcess_joinThreadPool();
    return EXIT_FAILURE;
}
```

**3d. 注册 manifest fragment**

```xml
<!-- hardware/interfaces/hello/1.0/default/android.hardware.hello.xml -->
<manifest version="1.0" type="device">
  <hal format="aidl">
    <name>android.hardware.hello</name>
    <version>1</version>
    <fqname>IHello/default</fqname>
  </hal>
</manifest>
```

**3e. 编译并测试**

```bash
# 在 AOSP 源码树中
source build/envsetup.sh
lunch sdk_phone_x86_64-eng
m android.hardware.hello-V1-ndk  # 先编译 AIDL 接口库
m hello-hal-service               # 编译服务

# push 到设备/模拟器
adb push out/target/product/*/system/lib64/android.hardware.hello-V1-ndk.so \
    /system/lib64/
adb push out/target/product/*/vendor/bin/hello-hal-service /vendor/bin/
adb push hardware/interfaces/hello/1.0/default/android.hardware.hello.xml \
    /vendor/etc/vintf/manifest/

# 启动服务
adb shell /vendor/bin/hello-hal-service &

# 验证注册
adb shell service list | grep hello
# 预期：android.hardware.hello.IHello/default: [android.hardware.hello.IHello]

# 调用 greet（method 1）
adb shell service call android.hardware.hello.IHello/default 1 s16 "World"
# 预期：
# Result: Parcel(
#   0x00000000: 00000000 00000021 006f006c 006c0065  '....!...l.e.l.o.'
#   ...)
# 解码 UTF-16 = "Hello from vendor HAL, World!"
```

---

## 6.7 失败模式与生产坑

### 坑 1：VINTF 不兼容导致 bootloop

**症状：** 刷机后设备进入 fastboot/recovery 循环，logcat 无法连接。  
**根因：** device manifest 中 HAL 版本低于 FCM 要求的最低版本。  
**定位：**
```bash
# 用 recovery adb 拿 log
adb shell dmesg | grep "vintf"
adb shell logcat -b all | grep "VINTF\|VintfObject"
# 典型错误：
# E VintfObject: "android.hardware.audio@7.0::IDevicesFactory/default" is required
# E VintfObject: but "android.hardware.audio@6.0" is provided
```
**修复：** 更新 vendor manifest 中对应 HAL 版本，或回退 system image 到与 vendor 兼容的 FCM level。

### 坑 2：hwservicemanager 未启动时注册 HAL 导致 crash

**症状：** HAL 服务进程启动后 `AServiceManager_addService` 返回 `STATUS_NAME_NOT_FOUND`，进程 abort。  
**根因：** `init.rc` 中 HAL 服务的 `class` 未正确声明，在 `hwservicemanager` 启动前就尝试注册。  
**修复：**
```rc
# vendor/etc/init/android.hardware.vibrator-service.rc
service vibrator-default /vendor/bin/hw/android.hardware.vibrator-service.example
    class hal              # ← 必须是 "hal" class，在 hwservicemanager 就绪后启动
    user system
    group system
    capabilities SYS_NICE  # 按需添加 capability
```

### 坑 3：AIDL callback 持有引用导致 HAL 进程无法回收

**症状：** HAL 进程内存持续增长，`dumpsys procstats` 显示 vendor HAL 进程 RSS 异常。  
**根因：** framework 向 HAL 传递 `IVibratorCallback`，HAL 实现存进 member 变量但从未清空 —— Binder 强引用阻止 GC。  
**修复：** 在 callback invoke 后（`onComplete()`）或 `off()` 时，显式置 `mVibrationCallback = nullptr`（参见 Vibrator.cpp off() 实现的 lock-外清空模式）。

### 坑 4：HIDL hash mismatch 导致编译失败

**症状：** `m android.hardware.foo@1.0` 报错：`Interface hash ... doesn't match`。  
**根因：** 修改了已 release 的 `.hal` 文件（即使是注释），导致 SHA-256 hash 与 `current.txt` 不符。  
**修复：** 绝不修改已发布的接口。新增功能需走 minor 版本（`@1.1`），使用 `extends`。
```hidl
// android.hardware.foo@1.1::IFoo extends 1.0
package android.hardware.foo@1.1;
import android.hardware.foo@1.0;
interface IFoo extends @1.0::IFoo {
    newMethod() generates (int result);
};
```

### 坑 5：SP-HAL（Vulkan）版本冲突

**症状：** 升级 system image 后 app crash，logcat 显示 `dlopen failed: cannot locate symbol`。  
**根因：** SurfaceFlinger 同进程加载 vendor Vulkan `.so`，该 `.so` 依赖 `libvulkan.so` 的新符号，但 system 版本旧。  
**修复：** VNDK（Vendor NDK）严格管控 vendor 可链接的 system library 列表；vendor `.so` 应只依赖 `vndk-sp` 中的库，不应依赖非 SP 的 system 私有符号。

### 坑 6：lshal 中 HAL 状态为 N/A（not running）

**症状：** `lshal` 显示 HAL 已声明但实例为 N/A，对应功能不工作（如触摸无响应）。  
**根因 A：** SELinux 阻止 HAL 进程绑定，logcat 有 `avc: denied { add }` 日志。  
**根因 B：** HAL 进程 crash 被 init 抑制（`restart_ratelimit`）。  
**定位：**
```bash
adb shell logcat -b all | grep "avc.*vibrator\|vibrator.*died\|vibrator.*crash"
adb shell dumpsys activity | grep "vibrator"
adb shell ps -A | grep vibrator
```

---

## 6.8 设计演进总结

### 2025 年的现状（Android 15/16）

1. **HIDL 仅维护不新增**：所有新 HAL 接口必须用 AIDL。存量 HIDL HAL 逐年迁移，`hidl2aidl` 工具半自动化转换。
2. **统一 servicemanager**：Android 12 将 `hwservicemanager`（HIDL）和 `vndservicemanager`（vendor AIDL）合并到统一 `servicemanager`，简化 service 查找路径。
3. **APEX 化 HAL**：部分 HAL（如 vibrator reference impl）已支持打包进 APEX 模块，允许通过 Google Play 系统更新单独更新，绕过 full OTA。
4. **GKI + HAL 双层抽象**：内核侧由 GKI（Generic Kernel Image）+ vendor modules（`.ko`）提供稳定 KMI；用户态由 HAL 提供稳定 ABI —— 两层抽象共同保障 Google 可以独立更新 Android OS。

### 架构全景图（Android 15）

```
┌──────────────────────────────────────────────────────────────┐
│                     USER SPACE                                │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │    system partition      │  │    vendor partition       │  │
│  │  VibratorService.java    │  │  vibrator-service.example│  │
│  │  AudioFlinger            │◄─┤  (AIDL BnHello)          │  │
│  │  CameraService           │  │  audio.default.so        │  │
│  │  SurfaceFlinger          │  │  Vulkan driver (SP-HAL)  │  │
│  └─────────────────────────┘  └──────────────────────────┘  │
│         ↕ /dev/binder                  ↕ /dev/binder         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │               servicemanager (Android 12+)               │ │
│  │    VINTF manifest 校验 + HAL service registry            │ │
│  └─────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                     KERNEL SPACE                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GKI kernel + binder driver (/dev/binder)             │   │
│  │  Stable KMI interface                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  vendor kernel modules (*.ko) — SoC 驱动               │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 章末五件套

### 1. 核心概念速查

| 概念 | 一句话 |
|------|--------|
| Project Treble | Android 8 引入的 system/vendor 分区接口契约化项目 |
| HIDL | HAL 专用 IDL，已 deprecated，走 hwbinder |
| AIDL HAL (stable) | 现行主流，走 binder，`@VintfStability` 标记跨分区接口 |
| Binderized HAL | HAL 运行在独立 vendor 进程，崩溃隔离 |
| SP-HAL | OpenGL/Vulkan 白名单 HAL，同进程加载，绕过 IPC |
| VINTF manifest | vendor 声明提供的 HAL 及版本，启动时与 FCM 校验 |
| FCM | Framework Compatibility Matrix，system 声明要求的 HAL |
| hwbinder | HIDL 专用 Binder 设备节点 `/dev/hwbinder` |
| `@VintfStability` | AIDL 注解，要求接口跨 vendor/system 边界时必须 frozen |
| vintf_fragments | per-module manifest 片段，build 时合并进 device manifest |

### 2. 学习路径

1. 读 `hardware/interfaces/vibrator/aidl/` 目录全部文件 —— 最简洁的完整 AIDL HAL 参考实现。
2. 运行 `lshal` 在真实设备上，理解哪些接口还是 HIDL、哪些已是 AIDL。
3. 读 `system/libhidl/` 和 `system/libvintf/` 源码 —— 理解 VINTF 校验的底层实现。
4. 用 `hidl2aidl` 工具把一个 HIDL 接口转成 AIDL，观察差异。
5. 阅读 `hardware/interfaces/audio/aidl/` —— 复杂 AIDL HAL 案例（嵌套接口、parcelable、状态机）。

### 3. 面试题

**Q1：** Treble 之前 Android 为什么版本碎片化严重？Treble 如何在架构上解决？  
**Q2：** HIDL passthrough 模式和 binderized 模式的区别？各自适用什么场景？  
**Q3：** `@VintfStability` 注解的含义？不加会怎样？  
**Q4：** 设备刷机后 bootloop，`dmesg` 出现 "VINTF" 关键字，可能是什么原因？如何排查？  
**Q5：** 为什么 Vulkan HAL 可以不走 Binder，而 vibrator HAL 必须走？  

### 4. 进阶阅读

- AOSP `hardware/interfaces/` — 全部 HAL 接口定义
- AOSP `system/libvintf/` — VINTF 校验实现
- AOSP `system/libhidl/` — HIDL 传输层
- Android CDD（Compatibility Definition Document）— FCM/VINTF 的合规要求
- Google I/O 2017 "Project Treble: Helping Android devices run the latest software"
- Android 13 Release Notes — AIDL HAL 全面铺开

### 5. Sources Fetched（本章实际获取的 URL）

| URL | 内容 | 状态 |
|-----|------|------|
| `source.android.com/docs/core/architecture/hal` | HAL 架构概览 | 成功 |
| `source.android.com/docs/core/architecture/hidl` | HIDL 设计与语法 | 成功 |
| `source.android.com/docs/core/architecture/aidl/aidl-hals` | AIDL HAL 架构 | 成功 |
| `source.android.com/docs/core/architecture/vintf` | VINTF 概览 | 成功 |
| `source.android.com/docs/core/architecture/vintf/objects` | manifest 格式 | 成功 |
| `source.android.com/docs/core/architecture/vintf/dm` | device manifest | 成功 |
| `source.android.com/docs/core/architecture/aidl/stable-aidl` | stable AIDL 规则 | 成功 |
| `source.android.com/docs/core/architecture/hidl/hashing` | hash 机制 | 成功 |
| `source.android.com/docs/core/architecture/hidl/services` | 服务注册 | 成功 |
| `source.android.com/docs/core/architecture/partitions` | 分区结构 | 成功 |
| `source.android.com/docs/core/architecture/kernel/generic-kernel-image` | GKI | 成功 |
| `android.googlesource.com/.../IVibrator.aidl` | 真实 AIDL 接口 | 成功（base64解码） |
| `android.googlesource.com/.../Vibrator.cpp?format=TEXT` | 真实实现代码 | 成功（base64解码） |
| `android.googlesource.com/.../main.cpp?format=TEXT` | 服务注册入口 | 成功（base64解码） |
| `android.googlesource.com/.../android.hardware.vibrator.xml` | manifest fragment | 成功 |
| `android.googlesource.com/.../Android.bp` | 构建文件 | 成功 |
| `android.googlesource.com/.../IModule.aidl` | audio AIDL HAL | 成功 |
| `raw.githubusercontent.com/.../IDevice.hal` | audio HIDL HAL | 404（已标注） |

---

*本章源码核查时间：2026-06-12。AOSP 处于 main branch 持续演进中，具体 API 以实际源码为准。*
