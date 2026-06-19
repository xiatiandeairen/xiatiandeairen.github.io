---
title: "Android 安全模型：SELinux、权限与沙箱"
slug: "8-07"
collection: "tech-library"
group: "android系统"
order: 8007
summary: "TL;DR Android 的安全模型是三层叠加的防御体系：① Linux UID/GID 为每个 app 分配独立数字身份（DAC）；② SELinux 在内核层强制执行类型强制策略（MAC），即使 root 进程也无法越权；③ Runtime Permission + AppOps 在用户层按资…"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:46:05.000Z"
updatedAt: "2026-06-12T11:46:05.000Z"
---
> **TL;DR**
> Android 的安全模型是**三层叠加**的防御体系：① Linux UID/GID 为每个 app 分配独立数字身份（DAC）；② SELinux 在内核层强制执行类型强制策略（MAC），即使 root 进程也无法越权；③ Runtime Permission + AppOps 在用户层按资源粒度授权，并全程审计。理解这三层的**交互点**——Zygote fork 时的 UID 切换 + SELinux context 切换、PackageManager 安装时的 seinfo 计算、AppOps 的 noteOp 调用链——才能在工程实践中准确定位 SecurityException 或 avc: denied 的根因。

---

## 前置知识

| 依赖主题 | 章节 |
|---|---|
| 进程模型 / Zygote fork 机制 | 第 1 章 |
| Binder IPC 与调用者身份校验 | 第 2 章 |
| PackageManager / system_server 架构 | 第 4 章 |
| HAL / Treble 隔离策略 | 第 6 章 |

**工具前置**：adb、`adb root`（需 userdebug 固件）、`adb shell`、`logcat`；SELinux 分析工具 `audit2allow`（可从 `selinux-utils` 包安装）。

---

## 7.1 设计考古：安全模型的三次演进

### 7.1.1 第一代（Android 1.0–4.4）：纯 DAC + 静态权限

Android 从 Linux 继承了 **DAC（Discretionary Access Control）**：每个 App 安装时获得 10000–19999 范围内的唯一 UID（Application ID），进程以此 UID 运行，内核 VFS 层用 uid/gid 做文件访问仲裁。

```
/data/data/com.example.myapp/    → uid=10123, gid=10123, mode=700
/data/data/com.example.other/   → uid=10124, gid=10124, mode=700
```

权限模型是**安装时静态授权**：AndroidManifest.xml 声明 `<uses-permission>`，安装对话框展示全部权限，用户 accept-all 或放弃安装。这就是著名的"一刀切授权"——安装即代表用户同意所有权限，直到 Android 6.0 前都如此。

**痛点**：
1. 权限粒度太粗（READ_EXTERNAL_STORAGE 意味着访问所有 SD 卡）。
2. DAC 无法约束 root 进程。2012 年的 Samsung Exynos 漏洞（CVE-2012-6422）正是利用了 root 权限绕过 `/dev/exynos-mem` DAC 的弱点。
3. 同签名 App 共享 sharedUserId，造成供应链风险。

### 7.1.2 第二代（Android 5.0–8.0）：SELinux Enforcing + Runtime Permission

**Android 5.0（2014）** 是安全史上的分水岭——SELinux 从 Permissive 切到 **全面 Enforcing**。设计动机见 Stephen Smalley 在 Linux Security Summit 2013 上的演讲：核心问题是 DAC 不足以约束 privileged daemon（netd, vold 以 root 运行），一旦被漏洞利用，系统全线失守。SELinux 把每个进程关进独立的"类型域"，即使 root 也必须通过 MAC 校验。

**Android 6.0（2015）** 引入 **Runtime Permission**：dangerous 级权限不再安装时授权，而是使用时弹框请求。这一设计参考了 iOS 8 的细粒度授权模式，但在 Android 上通过 `PackageManagerService` + `AppOpsManager` 在纯软件层实现。

**Android 8.0（2017）** 随 Project Treble 引入 **Vendor SELinux 分离**：platform sepolicy 和 vendor sepolicy 独立版本化，vendor 不能再直接修改核心策略，解决了 OEM 碎片化安全补丁的问题。

### 7.1.3 第三代（Android 9–14+）：纵深防御完善

| 版本 | 关键变化 |
|---|---|
| Android 8.0 | seccomp-bpf syscall 过滤正式对所有 App 启用 |
| Android 9 | 非特权 App（targetSdk≥28）获得**独立 SELinux 沙箱**（per-app MCS category）|
| Android 10 | Scoped Storage：App 不再能通过路径直接访问 `/sdcard/DCIM` 等目录 |
| Android 11 | One-time permission（单次授权）；permission auto-revoke |
| Android 12 | Approximate location permission 独立；Bluetooth 权限重构 |
| Android 13 | Photo picker；细粒度媒体权限（READ_MEDIA_IMAGES 等） |
| Android 14 | Health permissions；partial photo access |

---

## 7.2 Linux UID 沙箱：基础隔离层

### 7.2.1 UID 空间分配

```
0           root（系统）
1000        system（system_server）
1001        radio
1002        bluetooth
1003        graphics
...
9999        （系统服务保留区）
10000–19999  App（single user，userId=0）
20000–29999  App（userId=1，多用户第一个）
99000–99999  isolated_app（独立服务进程）
```

【真实源码 frameworks/base/core/java/android/os/Process.java】中定义了关键常量：

```java
// 【真实源码 aosp/frameworks/base 
//  core/java/android/os/Process.java】
public static final int FIRST_APPLICATION_UID = 10000;
public static final int LAST_APPLICATION_UID  = 19999;
public static final int FIRST_ISOLATED_UID    = 99000;
public static final int LAST_ISOLATED_UID     = 99999;
```

**Application ID** = UID mod 100000。多用户场景下 App 的实际 UID = userId * 100000 + appId。

### 7.2.2 App 数据目录的 DAC

App 安装后，`installd`（native daemon）创建数据目录，权限如下：

```
/data/data/<package>/       uid=10123 gid=10123 mode=0700  (targetSdk≥24)
/data/data/<package>/cache/ uid=10123 gid=10123 mode=0771
/data/data/<package>/files/ uid=10123 gid=10123 mode=0771
```

`targetSdk < 24` 的旧 App 权限是 `0751`，允许 others 的 execute bit（用于目录遍历），Android 6.0 起新安装的 App 改为 `0700`，彻底封闭。这是 DAC 层的第一道防线。

### 7.2.3 supplementary GID 的权限扩展

App 进程启动时，除了 UID/GID 之外还会被加入若干补充 GID，授予对特定设备节点的访问权：

```
gid=1003 (graphics)  → /dev/graphics/
gid=1004 (input)     → /dev/input/
gid=3001 (net_bt_admin)
gid=3002 (net_bt)
gid=3003 (inet)      → socket(AF_INET/AF_INET6)
```

`inet` GID（3003）历史上是控制 App 网络访问的主要机制——早于 SELinux 的网络沙箱。

---

## 7.3 SELinux：强制访问控制层

### 7.3.1 基本概念速览

SELinux 基于 **Type Enforcement（TE）** 模型，所有对象（进程、文件、socket、property 等）都打上 **security context**（安全标签），格式为：

```
user:role:type:sensitivity[:categories]
```

Android 简化了标准 Linux SELinux：
- **user** 固定为 `u`
- **role** 对 subject（进程）是 `r`，对 object（文件等）是 `object_r`  
- **type** 是核心——进程的 type 叫 **domain**
- **sensitivity** 固定 `s0`
- **categories** 用于 MCS per-app 隔离（Android 9+）

策略规则语法：
```
allow   source_domain  target_type:object_class  { permissions };
deny    ...;           # 默认拒绝，无需显式写
neverallow ...;        # 编译时检查，永远不允许
```

### 7.3.2 Android SELinux 域层次

```
domain（所有进程）
├── coredomain（平台核心域：system_server, zygote, init...）
│   ├── system_server
│   ├── zygote
│   ├── installd
│   ├── netd
│   └── ...
└── appdomain（所有 App 域）
    ├── untrusted_app_all（第三方 App 基类）
    │   ├── untrusted_app          (targetSdk ≥ 34)
    │   ├── untrusted_app_32       (targetSdk 32-33)
    │   └── untrusted_app_30       (targetSdk 30-31)
    ├── isolated_app_all（独立服务进程）
    │   └── isolated_app
    ├── platform_app               (平台签名 App)
    ├── system_app                 (system uid)
    ├── priv_app                   (特权 App)
    └── ephemeral_app              (即时 App)
```

### 7.3.3 核心宏定义：app_domain、untrusted_app_domain

【真实源码 android.googlesource.com platform/system/sepolicy main public/te_macros】：

```
# app_domain($1) 宏 —— 注册一个 App 域
define(`app_domain', `
  typeattribute $1 appdomain;          # 打上 appdomain 属性
  type_transition $1 tmpfs:file appdomain_tmpfs;  # tmpfs 文件自动打 appdomain_tmpfs 标签
  userfaultfd_use($1)                  # 允许 userfaultfd（内存管理）
  allow $1 appdomain_tmpfs:file { execute getattr map read write ioctl };
  allowxperm $1 appdomain_tmpfs:file ioctl ashmem_ioctls;

  # 关键 neverallow：App 不能读写其他 App 的文件
  neverallow { $1 -runas_app -shell -simpleperf } { domain -$1 }:file no_rw_file_perms;
  neverallow { appdomain -runas_app -shell -simpleperf -$1 } $1:file no_rw_file_perms;

  # ptrace 隔离：只有 crash_dump、llkd 可 ptrace App（调试除外）
  neverallow { domain -$1 -crash_dump userdebug_or_eng(`-llkd') -runas_app -simpleperf } $1:process ptrace;
')

# untrusted_app_domain($1) 宏 —— 标记为非特权 App
define(`untrusted_app_domain', `
  typeattribute $1 untrusted_app_all;  # 打上 untrusted_app_all 属性
')

# domain_trans($1→$2 via $3) —— 域转换宏
define(`domain_trans', `
  allow $1 $2:file { getattr open read execute map };   # 父进程可读新程序文件
  allow $1 $3:process transition;                       # 父可触发进程转换
  allow $3 $2:file { entrypoint open read execute getattr map }; # 新域以该文件为入口
  ifelse($1, `init', `', `allow $3 $1:process sigchld;')  # 非 init 需能接收 SIGCHLD
  dontaudit $1 $3:process noatsecure;
  allow $1 $3:process { siginh rlimitinh };             # 继承信号和 rlimit
')
```

### 7.3.4 untrusted_app.te 完整策略（SDK 34）

【真实源码 android.googlesource.com platform/system/sepolicy main prebuilts/api/34.0/private/untrusted_app.te】：

```
# targetSdkVersion >= 34 的第三方 App

typeattribute untrusted_app coredomain;   # ⚠ 注意：虽然名字是 untrusted，
                                           # 它仍是 coredomain 的成员

app_domain(untrusted_app)                 # 展开 app_domain 宏
untrusted_app_domain(untrusted_app)       # 标记 untrusted_app_all 属性
net_domain(untrusted_app)                 # 允许网络访问 (typeattribute netdomain)
bluetooth_domain(untrusted_app)           # 允许蓝牙访问

# SDK Sandbox 文件描述符共享（为 WebView 实验数据）
# TODO(b/229249719): Android U 起不再支持
allow untrusted_app sdk_sandbox_data_file:fd use;
allow untrusted_app sdk_sandbox_data_file:file write;
neverallow untrusted_app sdk_sandbox_data_file:file { open create };
# ↑ 只能用收到的 fd，不能自己 open
```

### 7.3.5 seapp_contexts：App 如何获得 SELinux 域

每次 Zygote fork App 进程时，libselinux 读取 `/system/etc/selinux/plat_seapp_contexts`，根据 UID、seinfo、isPrivApp 等字段匹配出目标 domain 和 levelFrom。

关键规则（示意整理，内容来自 android.googlesource.com platform/system/sepolicy main private/seapp_contexts）：

```
# system_server 进程
user=system isSystemServer=true domain=system_server_startup levelFrom=all

# system uid + platform seinfo → system_app
user=system seinfo=platform domain=system_app levelFrom=user

# GMS Core 特殊处理
user=_app seinfo=platform:privapp:targetSdkVersion=33:partition=system \
    name=com.google.android.gms domain=gmscore_app levelFrom=all

# SDK 34+ 第三方 App
user=_app minTargetSdkVersion=34 domain=untrusted_app levelFrom=all

# SDK 32-33
user=_app minTargetSdkVersion=32 domain=untrusted_app_32 levelFrom=all

# SDK 30-31
user=_app minTargetSdkVersion=30 domain=untrusted_app_30 levelFrom=all

# 默认（老 App）
user=_app domain=untrusted_app_25 levelFrom=all

# isolated 服务进程
user=_isolated domain=isolated_app levelFrom=all

# SDK Sandbox
user=_app_zygote domain=sdk_sandbox_next levelFrom=all
```

`levelFrom=all` 表示 MCS category 从 user + app 双维度派生，确保每个 App 的 data 目录都有唯一的 MCS label pair（如 `s0:c14,c259`），防止跨 App 读取。

### 7.3.6 file_contexts：文件系统安全标签

【真实源码（部分）android.googlesource.com platform/system/sepolicy main private/file_contexts】：

```
# 系统分区
/system(/.*)?                     u:object_r:system_file:s0
/system/lib(64)?(/.*)?            u:object_r:system_lib_file:s0
/system/bin/init                  u:object_r:init_exec:s0

# APK 目录
/data/app(/.*)?                   u:object_r:apk_data_file:s0
/data/app/[^/]+/oat(/.*)?         u:object_r:dalvikcache_data_file:s0

# vendor 分区（Treble 隔离）
/(vendor|system/vendor)(/.*)?     u:object_r:vendor_file:s0
/(odm|vendor/odm)(/.*)?           u:object_r:vendor_file:s0

# /data 下文件默认标签（未被更精确规则匹配的）
# 由 seapp_contexts 的 type= 字段决定 App 私有目录标签
# 即 /data/data/<pkg>/ → app_data_file:s0:c14,c259（MCS pair）
```

App 私有目录的 MCS label 不在 file_contexts 里静态定义，而是 `installd` 在安装时调用 `selinux_android_setfilecon()` 动态设置，MCS category pair 由 App UID 派生（deterministic hash），确保唯一性和可重现性。

### 7.3.7 Vendor/Platform 策略分离（Treble）

Android 8.0 起，SELinux 策略被分成两部分独立版本化：

```
/system/etc/selinux/
├── plat_sepolicy.cil        # 平台策略（OTA 更新）
├── plat_file_contexts
├── plat_seapp_contexts
└── plat_mac_permissions.xml

/vendor/etc/selinux/
├── vendor_sepolicy.cil      # vendor 策略（单独更新）
├── vendor_file_contexts
└── vendor_seapp_contexts
```

**核心约束**（来自 private/domain.te neverallow 规则）：

```
# vendor 域不能访问 core data 文件
neverallow { vendor_domain -vendor_init } core_data_file_type:file *;

# vendor 不能执行 /system/bin 下的工具（除白名单外）
neverallow { vendor_domain -init } { system_file vendor_file }:file execute_no_trans;

# core 域不能直接访问 vendor 文件（通过 HAL 接口访问）
neverallow { coredomain -init -recovery } vendor_file:file execute;
```

这套分离机制使 Google 的平台安全补丁可以独立通过 OTA 推送，而不需要等待 OEM 重新集成 vendor 驱动。

---

## 7.4 进程生命周期中的安全切换：Zygote → App

理解安全模型最关键的一条链路是 **Zygote fork 时发生了什么**。

### 7.4.1 SpecializeCommon 的执行顺序

【真实源码 android.googlesource.com platform/frameworks/base main core/jni/com_android_internal_os_Zygote.cpp — SpecializeCommon 函数】：

```cpp
// 1. 切换到子进程 PID namespace（Android 9+ 可选）

// 2. 设置 GID（在 UID 之前！因为 setgid 需要之前的能力）
if (setresgid(gid, gid, gid) == -1) {
    fail_fn(CREATE_ERROR("setresgid(%d) failed: %s", gid, strerror(errno)));
}

// 3. 设置 seccomp BPF 过滤（必须在 setuid 之前，此时还有 CAP_SYS_ADMIN）
SetUpSeccompFilter(uid, is_child_zygote);

// 4. 设置调度策略（需要 CAP_SYS_NICE，还未 drop）
SetSchedulerPolicy(fail_fn, is_top_app);

// 5. 切换 UID（不可逆！之后 capabilities 根据 inheritable set 计算）
if (setresuid(uid, uid, uid) == -1) {
    fail_fn(CREATE_ERROR("setresuid(%d) failed: %s", uid, strerror(errno)));
}

// 6. 设置 SELinux context（调用 libselinux）
const char* se_info_ptr = se_info.has_value() ? se_info.value().c_str() : nullptr;
if (selinux_android_setcontext(uid, is_system_server, se_info_ptr, nice_name_ptr) == -1) {
    fail_fn(CREATE_ERROR("selinux_android_setcontext(...) failed"));
}
// selinux_android_setcontext 内部：
// 1. 读 plat_seapp_contexts（已 mmap 到内存）
// 2. 按 uid/seinfo/pkgname 匹配规则
// 3. 调用 setcon() 切换当前进程的 SELinux context
// 4. 调用 selinux_android_setcondircreatename() 设置新创建文件的 context
```

**顺序的安全含义**：seccomp filter 在 setuid 前设置，因为安装 seccomp filter 需要 `CAP_SYS_ADMIN` 或 `PR_SET_NO_NEW_PRIVS`。一旦 setuid 切到非特权 UID，就无法再修改 filter，实现了过滤器不可绕过。

### 7.4.2 seinfo 的计算：SELinuxMMAC

在 PackageManager 扫描安装包时，`SELinuxMMAC.java` 读取 `mac_permissions.xml` 计算 seinfo 字符串：

【真实源码 android.googlesource.com platform/frameworks/base main services/core/java/com/android/server/pm/SELinuxMMAC.java】：

```java
// getSeInfo() 构建 App 的 seinfo 字符串
public static String getSeInfo(PackageState pkgState, AndroidPackage pkg,
        boolean isPrivileged, int targetSdkVersion) {
    String seInfo = null;

    // 遍历已加载的 mac_permissions.xml 策略（按特异性排序）
    for (Policy policy : sPolicies) {
        seInfo = policy.getMatchedSeInfo(pkg);  // 签名匹配
        if (seInfo != null) break;
    }

    if (seInfo == null) {
        seInfo = DEFAULT_SEINFO;  // "default"
    }

    // 追加修饰词
    if (isPrivileged) {
        seInfo += ":privapp";       // priv_app 目录下的 App
    }
    seInfo += ":targetSdkVersion=" + targetSdkVersion;  // 影响 seapp_contexts 匹配
    seInfo += ":partition=" + getPartition(pkgState);   // system/vendor/product 等

    return seInfo;
    // 示例结果：
    // 平台签名 App  → "platform:privapp:targetSdkVersion=34:partition=system"
    // 第三方 App    → "default:targetSdkVersion=34:partition=data"
}
```

`mac_permissions.xml` 中的签名标签（`@PLATFORM` 宏展开为实际公钥哈希）：

```xml
<!-- plat_mac_permissions.xml（【示意，非逐字】）-->
<policy>
  <signer signature="@PLATFORM">
    <seinfo value="platform"/>     <!-- 平台签名 App 获得 platform seinfo -->
  </signer>
  <signer signature="@BLUETOOTH">
    <seinfo value="bluetooth"/>
  </signer>
  <!-- 无匹配的第三方 App 用 seapp_contexts 中的 default 规则 -->
</policy>
```

### 7.4.3 MCS Category：per-App 文件隔离

Android 9 引入的 per-app SELinux sandbox 依赖 **MCS（Multi-Category Security）**。每个 App 的 data 目录被赋予唯一的 category pair（两个 category bit 的组合），其他 App 的进程没有这两个 category，因此内核拒绝访问。

Category pair 由 UID 通过确定性算法计算：

```
categories = (uid / 1024, uid % 1024)
# uid=10123 → c9, c587（示例，实际算法在 selinux_android_setfilecon 内部）
```

App 进程的 SELinux context：
```
u:r:untrusted_app:s0:c9,c587
```

App data 目录的 label：
```
u:object_r:app_data_file:s0:c9,c587
```

即使两个 App 的 domain type 相同（都是 `untrusted_app`），由于 category 不同，MCS 强制隔离确保 A 无法访问 B 的数据。这就是 `neverallow` 在 te_macros 中对 inter-app 文件访问的强制保证。

---

## 7.5 权限系统：三层架构

### 7.5.1 四级 Protection Level

【真实源码 android.googlesource.com platform/frameworks/base main core/java/android/content/pm/PermissionInfo.java】：

```java
// 基础 protection level（低 8 bit）
public static final int PROTECTION_NORMAL    = 0;  // 安装时自动授予，低风险
public static final int PROTECTION_DANGEROUS = 1;  // Runtime permission，需用户确认
public static final int PROTECTION_SIGNATURE = 2;  // 签名相同才授予
public static final int PROTECTION_INTERNAL  = 4;  // 系统内部，不对外暴露

// 保护标志（高 bit 组合）
public static final int PROTECTION_FLAG_PRIVILEGED    = 0x10;  // 需 priv_app
public static final int PROTECTION_FLAG_DEVELOPMENT   = 0x20;  // 仅 dev 模式
public static final int PROTECTION_FLAG_APPOP         = 0x40;  // 通过 AppOps 控制
public static final int PROTECTION_FLAG_PRE23         = 0x80;  // 兼容 pre-M
public static final int PROTECTION_FLAG_INSTALLER     = 0x100;
public static final int PROTECTION_FLAG_RUNTIME_ONLY  = 0x2000; // 仅 runtime 控制
public static final int PROTECTION_FLAG_ROLE          = 0x4000000; // 角色权限
public static final int PROTECTION_FLAG_KNOWN_SIGNER  = 0x8000000; // 已知证书
```

实际使用中常见组合：
- `signatureOrSystem` = `PROTECTION_SIGNATURE | PROTECTION_FLAG_PRIVILEGED`  
  → 必须是系统签名 **且** 安装在 priv-app 目录
- `dangerous` = `PROTECTION_DANGEROUS`  
  → Camera、Location、Microphone 等

### 7.5.2 Runtime Permission 的授权流程

```
用户点击"Allow" 
  → PermissionController (独立 APK，隔离进程)
  → IPackageManager.grantRuntimePermission() [Binder 调用]
  → PermissionManagerServiceImpl.grantRuntimePermission()
  → UidPermissionState.grantPermission(bp)
  → 写入 /data/system/users/{userId}/runtime-permissions.xml
  → killUidForPermissionChange() 如需要（部分权限授予后需重启 App）
  → 回调通知 AppOpsManager 更新 op mode
```

【真实源码 android.googlesource.com platform/frameworks/base main services/core/java/com/android/server/pm/permission/PermissionManagerServiceImpl.java ~line 1460】：

```java
// grantRuntimePermission 核心逻辑（整理版，标注关键检查点）
public void grantRuntimePermission(String packageName, String permName,
        String deviceId, int userId) {
    // ① 权限门：需要 ADJUST_RUNTIME_PERMISSIONS_POLICY
    final int callingUid = Binder.getCallingUid();
    mContext.enforceCallingOrSelfPermission(
            android.Manifest.permission.GRANT_RUNTIME_PERMISSIONS, null);

    // ② 验证 userId 和 package 存在
    enforceGrantRevokeGetRuntimePermissionPermissions("grantRuntimePermission");
    enforceCrossUserPermission(callingUid, userId, ...);

    final AndroidPackage pkg = getPackageLocked(packageName);
    final BasePermission bp = getPermission(permName);

    // ③ 只允许 Runtime 类型权限
    if (bp.getProtectionFlags() == PermissionInfo.PROTECTION_DANGEROUS) {
        // 检查 SYSTEM_FIXED / POLICY_FIXED 标志
        final int flags = getPermissionFlagsInternal(pkg, permName, userId);
        if ((flags & PackageManager.FLAG_PERMISSION_SYSTEM_FIXED) != 0) {
            throw new SecurityException("Cannot grant system fixed permission");
        }

        // ④ 更新状态
        final UidPermissionState uidState = getUidStateLocked(pkg, userId);
        if (uidState.grantPermission(bp)) {
            // ⑤ GID 变化需要杀进程重启
            if (GIDS_CHANGED) killUidForPermissionChange(uid, userId, reason);
            // ⑥ 回调 PermissionController
            mOnPermissionChangeListeners.onPermissionsChanged(uid);
        }
    }
}
```

### 7.5.3 AppOps：运行时细粒度审计层

**AppOps（Application Operations）** 是权限系统和实际 API 访问之间的**桥梁层**，职责：
1. 细粒度授权（background/foreground location 不同模式）
2. 访问审计（记录 App 何时使用了 camera/mic 等）
3. 权限代理（当 A 代表 B 访问资源时，AppOps 记录实际归因）

【真实源码 android.googlesource.com platform/frameworks/base main core/java/android/app/AppOpsManager.java】：

```java
// OP 常量（共 156 个）
public static final int OP_CAMERA         = 26;  // 对应 CAMERA 权限
public static final int OP_RECORD_AUDIO   = 27;  // 对应 RECORD_AUDIO
public static final int OP_FINE_LOCATION  = 0;   // 对应 ACCESS_FINE_LOCATION
public static final int OP_COARSE_LOCATION= 1;

// 访问模式
public static final int MODE_ALLOWED    = 0;  // 允许，记录访问
public static final int MODE_IGNORED    = 1;  // 静默拒绝，不抛异常
public static final int MODE_ERRORED    = 2;  // 抛出 SecurityException
public static final int MODE_DEFAULT    = 3;  // 默认（查 permission 决定）
public static final int MODE_FOREGROUND = 4;  // 仅前台允许（后台静默拒绝）

// Runtime permission → AppOp 映射（用于访问审计）
// RUNTIME_PERMISSION_OPS 数组定义了 permission string → op code 对应关系
// 如：Manifest.permission.CAMERA → OP_CAMERA
```

每个 API provider（CameraService、AudioFlinger 等）在实际访问资源前必须调用：

```java
// 点操作（一次性）
int result = mAppOps.noteOp(AppOpsManager.OP_CAMERA, uid, packageName, 
                             attributionTag, message);
if (result != AppOpsManager.MODE_ALLOWED) {
    return; // 或抛异常
}

// 持续操作（开始/结束括号）
int result = mAppOps.startOp(AppOpsManager.OP_RECORD_AUDIO, uid, pkgName, ...);
// ... 使用资源 ...
mAppOps.finishOp(AppOpsManager.OP_RECORD_AUDIO, uid, pkgName, ...);
```

`noteOp` 除校验外还记录时间戳和 attribution，Android 12 的隐私指示器（相机/麦克风小图标）就靠这些记录驱动。

### 7.5.4 DefaultPermissionGrantPolicy：系统 App 的自动授权

【真实源码 android.googlesource.com platform/frameworks/base main services/core/java/com/android/server/pm/permission/DefaultPermissionGrantPolicy.java】：

```java
// 各系统角色的默认权限组（整理版）
private static final String[] PHONE_PERMISSIONS = {
    READ_PHONE_STATE, CALL_PHONE, READ_CALL_LOG, WRITE_CALL_LOG, 
    ADD_VOICEMAIL, USE_SIP, PROCESS_OUTGOING_CALLS
};

// 默认拨号器获得：电话+联系人+短信+麦克风+相机
void grantDefaultSystemHandlerPermissions(int userId) {
    // 拨号器
    grantPermissionsToSystemPackage(dialerPkg, userId,
        PHONE_PERMISSIONS, CONTACTS_PERMISSIONS, 
        SMS_PERMISSIONS, MICROPHONE_PERMISSIONS, CAMERA_PERMISSIONS);

    // 位置服务（Google Location Services）
    grantPermissionsToSystemPackage(locationPkg, userId,
        CONTACTS_PERMISSIONS, CALENDAR_PERMISSIONS, 
        MICROPHONE_PERMISSIONS, PHONE_PERMISSIONS, ...);
}

// 安全闸：用户手动设置的权限不可被策略覆盖
if ((flags & (USER_SET | USER_FIXED | POLICY_FIXED)) != 0) {
    return; // 用户意志优先
}
```

---

## 7.6 第三层防线：seccomp-bpf syscall 过滤

### 7.6.1 设计动机

SELinux 在 VFS/Binder/socket 等 hook 点工作，但无法防御**直接 syscall**（如 `ptrace`、`process_vm_readv` 这类不过 VFS 的攻击面）。seccomp-bpf 在 syscall 入口处设置 BPF 程序，对 App 进程的可用 syscall 进行白名单过滤，从内核最底层切断攻击向量。

### 7.6.2 实现架构

【真实源码 android.googlesource.com platform/bionic main libc/seccomp/seccomp_policy.cpp】：

```cpp
// Android 为三种进程类型定义了不同粒度的过滤器
enum FilterType {
    APP,          // 第三方 App（最严格）
    APP_ZYGOTE,   // App Zygote（孵化 WebView/SDK sandbox）
    SYSTEM,       // 系统进程（较宽松）
};

// 过滤器安装（由 Zygote 的 SpecializeCommon 调用）
void SetUpSeccompFilter(uid_t uid, bool is_app_zygote) {
    // 架构检测 + BPF 程序构建
    // BPF 规则：
    // 1. 验证当前指令集架构（防止 seccomp bypass via arch switching）
    // 2. 加载 syscall number
    // 3. 与白名单对比
    // 4. 不匹配 → SECCOMP_RET_TRAP（触发 SIGSYS，进程崩溃）
    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);
}

// 公开接口（在 Zygote fork 后调用）
void set_app_seccomp_filter();           // App 过滤器
void set_app_zygote_seccomp_filter();    // App Zygote 过滤器
void set_system_seccomp_filter();        // 系统进程过滤器
```

**被 App 过滤器阻断的典型 syscall**：
- `ptrace`（防止进程注入）
- `process_vm_readv` / `process_vm_writev`（跨进程内存访问）
- `kexec_load`（内核替换）
- `perf_event_open`（非特权硬件计数器访问）
- `pivot_root`（namespace 根切换）
- `acct`（进程记账）

### 7.6.3 setresuid/setresgid 范围限制

seccomp 还包含一个专门针对 `setresuid`/`setresgid` 的参数检查过滤器，防止 App 将自身 UID 提升到系统保留范围：

```cpp
// install_setuidgid_seccomp_filter 为 setresuid/setresgid 添加参数校验
// 规则：uid 参数必须 >= FIRST_APPLICATION_UID (10000)
// 或 == uid (不变)
// 其他值 → SECCOMP_RET_TRAP
```

---

## 7.7 方案对比：各安全层的能力边界

| 维度 | DAC (UID/GID) | SELinux MAC | Runtime Permission | seccomp-bpf |
|---|---|---|---|---|
| **粒度** | 进程/文件所有权 | 类型域 + 对象类 | 权限组/单个权限 | syscall 级 |
| **工作层** | VFS kernel | LSM hook（kernel） | framework（JVM） | syscall 入口（kernel） |
| **能否约束 root** | 否 | 是 | 否 | 是（部分） |
| **用户可感知** | 否 | 否 | 是（弹框） | 否（SIGSYS 崩溃） |
| **配置位置** | 文件权限 bit | .te / seapp_contexts | AndroidManifest + PackageManager | bionic seccomp 白名单 |
| **绕过难度** | 低（root 可绕） | 高（需改内核/SELinux policy） | 中（需 overlay/欺骗）| 高（需内核漏洞） |
| **主要局限** | 不能约束同 UID 的多组件 | policy 编写错误即失效 | 用户可能滥点 Allow | 无法过滤合法 syscall 的参数 |

### 场景适用分析

**场景 1：读取另一个 App 的私有文件**
- DAC 阻止（不同 UID，mode=0700）
- SELinux 阻止（app_data_file + MCS category 不匹配）
- 两层都阻止，防御纵深

**场景 2：后台静默录音**
- DAC 无法阻止（录音设备 App 有权限打开）
- SELinux 允许（untrusted_app 有 net_domain + bluetooth_domain，但不是录音门控）
- AppOps `OP_RECORD_AUDIO` MODE_FOREGROUND 阻止后台访问
- 单独靠 AppOps，前台录音正常后台被拒

**场景 3：ptrace 注入系统进程**
- DAC：root 进程可 ptrace 任何进程
- SELinux：`neverallow { domain -crash_dump ... } system_server:process ptrace` 阻止
- seccomp：App 进程的 ptrace syscall 直接 SIGSYS
- SELinux 是核心防线

---

## 7.8 可运行 Demo：源码级安全观测

以下所有 demo **需要 Android 模拟器或 userdebug 物理设备**，需 `adb root` 权限（或 `su`）。在 `eng`/`userdebug` 编译的系统上运行。

### Demo 1：观测 App UID 沙箱与 SELinux context

**前置**：已安装一个普通 App（示例用 `com.example.testapp`）。

```bash
# 步骤 1：查询 App 的 UID
adb shell dumpsys package com.example.testapp | grep userId
# 预期输出：userId=10123

# 步骤 2：找到 App 的 PID
adb shell pidof com.example.testapp
# 预期输出：3742

# 步骤 3：查看完整进程安全标签
adb shell ps -AZ | grep com.example.testapp
# 预期输出：
# u:r:untrusted_app:s0:c193,c512  u0_a123  3742  ...  com.example.testapp

# 分解：
# u:r:untrusted_app  - SELinux domain
# s0                 - sensitivity level
# c193,c512          - MCS category pair（由 UID 派生，保证 per-app 唯一）
# u0_a123            - Linux 用户名（uid 10123 = u0_a123）

# 步骤 4：验证 App 数据目录的 SELinux 标签
adb shell ls -laZ /data/data/com.example.testapp/
# 预期输出：
# drwx------  u0_a123 u0_a123 u:object_r:app_data_file:s0:c193,c512 .
# MCS category 与进程的 category 匹配，其他 App 无法访问

# 步骤 5：尝试从另一个 App 访问（演示拒绝）
adb shell run-as com.example.otherapp \
    cat /data/data/com.example.testapp/shared_prefs/prefs.xml
# 预期输出：
# cat: /data/data/com.example.testapp/shared_prefs/prefs.xml: 
# Permission denied
# （同时在 logcat 看到 avc: denied）

# 步骤 6：查看 SELinux enforce 状态
adb shell getenforce
# 预期输出：Enforcing
```

### Demo 2：实时捕获 SELinux denial

```bash
# 方法一：实时 logcat 过滤
adb logcat -b all -d | grep 'avc: denied'

# 预期 avc denied 格式解析：
# type=1400 audit(0.0:123): avc: denied { read } 
#   for pid=3742 comm="testapp" 
#   scontext=u:r:untrusted_app:s0:c193,c512 
#   tcontext=u:object_r:system_data_file:s0 
#   tclass=file
#        ↑操作    ↑进程域              ↑目标标签      ↑对象类

# 解读：
# scontext = 谁在访问（untrusted_app，category c193,c512）
# tcontext = 访问的是什么（system_data_file）
# tclass   = 对象类型（file）
# { read } = 被拒绝的操作

# 方法二：禁用 rate limit 后持续监听
adb shell auditctl -r 0  # 需要 root
adb logcat -s "audit"

# 方法三：提取 policy 并用 audit2allow 分析
adb pull /sys/fs/selinux/policy /tmp/sepolicy
adb logcat -b all -d | grep 'avc: denied' | \
    audit2allow -p /tmp/sepolicy
# 会建议 allow 规则，但需人工审查！
```

### Demo 3：观测运行时权限状态

```bash
# 查看某 App 的所有权限状态
adb shell dumpsys package com.example.testapp | grep -A2 "permission\."

# 预期输出片段：
# android.permission.CAMERA: granted=true, flags=[ USER_SET]
# android.permission.RECORD_AUDIO: granted=false, flags=[ USER_SET]
# android.permission.ACCESS_FINE_LOCATION: granted=true, flags=[ USER_SET]

# 用 shell 命令模拟撤销权限（无需 UI）
adb shell pm revoke com.example.testapp android.permission.CAMERA
# 验证撤销
adb shell dumpsys package com.example.testapp | grep "CAMERA"
# 预期：granted=false

# 重新授予
adb shell pm grant com.example.testapp android.permission.CAMERA

# 查看 AppOps 审计记录（谁最近访问了 camera）
adb shell dumpsys appops | grep -A5 "OP_CAMERA"
# 预期输出：
# OP_CAMERA (26): uid=10123 packageName=com.example.testapp
#   ACCESS: time=2026-06-12 10:23:45 duration=-1ms  ← 一次性访问
#   REJECT: time=...                                 ← 如有拒绝记录
```

### Demo 4：查看 seinfo 和 seapp_contexts 匹配

```bash
# 查看 App 的 seinfo 字符串（PackageManager 存储）
adb shell dumpsys package com.example.testapp | grep seInfo
# 预期：seInfo=default:targetSdkVersion=34:partition=data

# 对比系统 App
adb shell dumpsys package com.android.phone | grep seInfo
# 预期：seInfo=platform:privapp:targetSdkVersion=34:partition=system

# 查看当前 seapp_contexts 规则
adb shell cat /system/etc/selinux/plat_seapp_contexts | grep untrusted_app
# 预期：
# user=_app minTargetSdkVersion=34 domain=untrusted_app levelFrom=all
# user=_app minTargetSdkVersion=32 domain=untrusted_app_32 levelFrom=all
```

### Demo 5：seccomp 阻断 syscall 观测

```bash
# 编写一个尝试调用 ptrace 的小程序（需 NDK 构建）
cat > /tmp/test_ptrace.c << 'EOF'
#include <sys/ptrace.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

int main() {
    // 尝试 ptrace 自身 - 用于测试 seccomp
    long ret = ptrace(PTRACE_TRACEME, 0, 0, 0);
    if (ret == -1) {
        printf("ptrace PTRACE_TRACEME: %s (errno=%d)\n", strerror(errno), errno);
    } else {
        printf("ptrace allowed: ret=%ld\n", ret);
    }
    return 0;
}
EOF

# 用 NDK 编译并 push 到设备
# (假设已配置 NDK 环境)
# $NDK/toolchains/llvm/prebuilt/.../bin/aarch64-linux-android33-clang \
#     -o /tmp/test_ptrace /tmp/test_ptrace.c
# adb push /tmp/test_ptrace /data/local/tmp/
# adb shell chmod +x /data/local/tmp/test_ptrace

# 以 App UID 运行（模拟 App 进程）
# adb shell run-as com.example.testapp /data/local/tmp/test_ptrace
# 预期：PTRACE_TRACEME 在 root 环境允许，但 App 进程由于 seccomp
# 对 ptrace syscall 的限制，在实际 App 进程中会收到 SIGSYS

# 快速验证 seccomp 是否对 App 进程生效
adb shell cat /proc/$(pidof com.example.testapp)/status | grep Seccomp
# 预期输出：Seccomp: 2   （2 = SECCOMP_MODE_FILTER，已启用 BPF 过滤）
# 0 = 未启用，1 = strict mode，2 = filter mode（Android App 都是 2）
```

### Demo 6：Zygote 到 App 的域转换追踪

```bash
# 在启动 App 前开启详细 SELinux 日志
adb shell setenforce 0  # 先切 permissive（不阻断但记录）
adb shell setprop persist.logd.size 64M

# 清除 logcat，启动目标 App
adb logcat -c
adb shell am start -n com.example.testapp/.MainActivity

# 抓取 Zygote 域转换日志
adb logcat -d | grep -E "selinux|Zygote|type_transition"

# 查看 Zygote 的 SELinux 标签（应是 u:r:zygote:s0）
adb shell ps -AZ | grep zygote
# 预期：u:r:zygote:s0   root  ...  zygote
# 注意 zygote 以 root(uid=0) 运行，但 SELinux domain 是 zygote 而非 kernel

# fork 后子进程切换到 App 域
adb shell ps -AZ | grep com.example.testapp
# 预期：u:r:untrusted_app:s0:c193,c512  u0_a123  ...  com.example.testapp

adb shell setenforce 1  # 恢复 enforcing
```

---

## 7.9 失败模式与生产真坑

### 坑 1：SELinux denial 日志被 rate limit 截断

**症状**：avc: denied 每 5 秒最多只记录 5 条，日志不完整，看不到低频但关键的违规。

**根因**：内核 `audit_rate_check` 的 rate limit 默认 `5/5s`。

**解决**：
```bash
adb shell auditctl -r 0   # 关闭 rate limit（仅调试用）
# 或读 /proc/kmsg 绕过 logd
adb shell cat /proc/kmsg | grep 'avc: denied'
```

### 坑 2：vendor App 使用了 platform 签名密钥

**症状**：OEM 预装 App 被 `mac_permissions.xml` 识别为 `seinfo=platform`，自动进入 `platform_app` 或 `system_app` 域，权限过宽。

**根因**：OEM 使用了与 AOSP 相同的 debug 签名 key 打包 vendor App。

**解决**：生产系统必须使用独立的 release key，`@PLATFORM` 宏在 release 构建中展开为不同的公钥哈希。用 `android-build-tools` 的 `apksigner` 验证签名身份。

### 坑 3：Runtime Permission 授予但 AppOps 被独立设为 IGNORED

**症状**：`checkSelfPermission` 返回 `GRANTED`，但实际访问 camera/mic 时资源拿不到，无任何报错。

**根因**：AppOps 层 `MODE_IGNORED` 与权限系统的 granted 状态可以独立。某些安全软件或 DPM 策略会单独设置 AppOps mode。

**定位**：
```bash
adb shell dumpsys appops | grep -B2 "IGNORED\|MODE_IGNORED"
```

**解决**：代码中除了 `checkSelfPermission`，还需检查 `AppOpsManager.noteOpNoThrow()` 的返回值。

### 坑 4：isolated_app 进程通过继承 fd 绕过 SELinux

**症状**：App 将打开的文件 fd 传给 isolated service，service 通过 fd 操作了理论上无权访问的文件。

**根因**：SELinux 中 `allow source_domain target_type:fd use` 允许 fd 继承。isolated_app 的策略：
```
# 只能用收到的 fd，不能自行 open
allow isolated_app sdcard_type:fd use;
neverallow isolated_app sdcard_type:file { open read write };
```

**关键**：fd 传递是合法的设计（IPC），但需确认 fd 来源合法。若 host App 自身被漏洞利用，fd 可能被恶意构造。

### 坑 5：targetSdkVersion 降级绕过 SELinux 沙箱

**症状**：恶意 App 将 targetSdkVersion 设置为 25（低于 28），跳过 per-app SELinux 沙箱，进入宽松的 `untrusted_app_25` 域。

**根因**：seapp_contexts 按 minTargetSdkVersion 匹配，低 targetSdk App 进入较旧、限制较少的域。

**现状缓解**：Google Play 自 2019 年强制要求 targetSdk ≥ 28，2023 年 ≥ 33。但 sideload App 不受此约束。

**检测**：
```bash
adb shell dumpsys package <pkg> | grep targetSdk
# 如果 targetSdk 异常低，结合 seapp_contexts 确认其 domain
```

### 坑 6：自定义 ROM 将 SELinux 设为 permissive

**症状**：`adb shell getenforce` 返回 `Permissive`，所有 SELinux 违规只记录不拦截，安全假设失效。

**根因**：部分 AOSP fork（或厂商调试固件）在生产构建中保留 `androidboot.selinux=permissive`。

**检测**：CTS 测试 `android.security.cts.SELinuxHostTest` 会 fail；生产环境检查：
```bash
adb shell cat /proc/cmdline | grep selinux
# 不应有 androidboot.selinux=permissive
```

---

## 7.10 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        User Space                            │
│                                                              │
│  App Process (uid=10123)                                     │
│  ┌─────────────────────────────────────────────┐            │
│  │  Java/Kotlin Code                           │            │
│  │  checkSelfPermission() → GRANTED/DENIED     │            │
│  │  AppOpsManager.noteOp() → ALLOWED/IGNORED   │            │
│  └──────────────┬──────────────────────────────┘            │
│                 │ system call                                │
│  ┌──────────────▼──────────────────────────────┐            │
│  │  seccomp BPF Filter (bionic)                │            │
│  │  syscall whitelist → ALLOW / SIGSYS          │            │
│  └──────────────┬──────────────────────────────┘            │
└─────────────────┼───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                     Linux Kernel                             │
│                                                              │
│  DAC Check: uid/gid/mode bits                               │
│        ↓ (pass)                                             │
│  LSM (SELinux) Hook:                                        │
│    scontext=u:r:untrusted_app:s0:c193,c512                  │
│    tcontext=u:object_r:system_data_file:s0                  │
│    → ALLOW or DENY (avc: denied 日志)                       │
│                                                              │
│  MCS Check: source categories ⊇ target categories?         │
│        ↓ (pass all checks)                                  │
│  Actual Resource Access                                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         Zygote fork 安全切换时序                              │
│                                                              │
│  Zygote(root)                          Child Process         │
│  ──────────────────────────────────►                        │
│  fork()                                                      │
│  setresgid(gid)     ──────────────►  GID切换                │
│  SetUpSeccompFilter ──────────────►  BPF过滤器安装(需CAP)   │
│  setresuid(uid)     ──────────────►  UID切换(不可逆)        │
│  selinux_android_setcontext ──────►  SELinux域切换           │
│                                       u:r:untrusted_app:s0:cX,cY
└─────────────────────────────────────────────────────────────┘
```

---

## 章末五件套

### 1. 精准定义

**SELinux domain transition**：进程从父域切换到子域的时刻，由 `type_transition` 规则 + `allow X Y:process transition` 规则共同定义，发生在 exec() 或 Zygote fork 后的 `selinux_android_setcontext()` 调用。

**MCS category pair**：SELinux sensitivity label 的 category 字段，Android 中每个 App 的数据目录被分配唯一的两个 category bit（`c_x, c_y`），实现 MAC 级别的 per-app 文件隔离，即使两个 App 的 domain type 相同也无法互相访问。

**seinfo**：PackageManager 计算的字符串（如 `platform:privapp:targetSdkVersion=34:partition=system`），传入 Zygote fork 时作为 `selinux_android_setcontext()` 参数，驱动 seapp_contexts 的域匹配选择。

**AppOps noteOp vs checkOp**：`noteOp` 执行校验**并记录**访问（影响审计 + 隐私指示器），`checkOp`/`unsafeCheckOp` 只校验不记录。API provider **必须**用 `noteOp`/`startOp`，否则系统无法追踪谁实际使用了敏感资源。

### 2. 关键数字

| 项目 | 数值 |
|---|---|
| App UID 范围 | 10000–19999（单用户） |
| isolated_app UID 范围 | 99000–99999 |
| AppOps operation 总数 | 156（Android 14） |
| Android 引入 SELinux Enforcing | 版本 5.0（API 21） |
| targetSdk 触发 per-app SELinux 沙箱 | ≥ 28（Android 9） |
| targetSdk 切换到 700 目录权限 | ≥ 24（Android 7） |
| seccomp Seccomp status（/proc/pid/status） | 0=off, 1=strict, 2=filter |

### 3. 调试工具速查

```bash
# 查进程 SELinux 标签
adb shell ps -AZ | grep <process>

# 查文件 SELinux 标签
adb shell ls -laZ <path>

# 实时 SELinux denial
adb logcat -b all | grep 'avc: denied'

# 关闭 rate limit
adb shell auditctl -r 0

# 查 SELinux enforce 状态
adb shell getenforce

# 查 App 权限状态
adb shell dumpsys package <pkg> | grep "permission\."

# 查 AppOps 记录
adb shell dumpsys appops | grep -A5 <pkg>

# 查 seccomp 状态
adb shell cat /proc/<pid>/status | grep Seccomp

# 临时切换 permissive（调试）
adb shell setenforce 0

# 提取 policy 做 audit2allow 分析
adb pull /sys/fs/selinux/policy /tmp/
adb logcat -d | grep 'avc: denied' | audit2allow -p /tmp/policy
```

### 4. 延伸阅读

- **SELinux for Android**（Google Security Team）：`source.android.com/docs/security/features/selinux`
- **Android Application Sandbox**：`source.android.com/docs/security/app-sandbox`
- SELinux te_macros / global_macros：`android.googlesource.com/platform/system/sepolicy main public/`
- SELinuxMMAC.java 源码：`frameworks/base/services/core/java/com/android/server/pm/SELinuxMMAC.java`
- `com_android_internal_os_Zygote.cpp` SpecializeCommon 函数
- seccomp_policy.cpp：`bionic/libc/seccomp/seccomp_policy.cpp`
- **Stephen Smalley** "Security Enhanced (SE) Android" NDSS 2013（SELinux on Android 的设计论文）

### 5. 思考题

1. `untrusted_app` 是 `coredomain` 的成员，这意味着什么？如果它不是 coredomain，会有什么不同？（提示：coredomain 决定是否可以访问某些 core 资源，也决定 neverallow 中的分类）

2. 一个 App 持有 `ACCESS_FINE_LOCATION` 运行时权限，但 AppOps 的 `OP_FINE_LOCATION` 被设为 `MODE_IGNORED`，`checkSelfPermission` 会返回什么？实际定位请求会成功吗？这种状态是如何出现的？

3. isolated_app 的 UID 在 99000–99999 范围，不同的 isolated service 共享 UID `99000`（虚构例子）的话，MCS category 如何保证它们的数据隔离？还是说 isolated service 天然无持久 data？

4. 如果一个 root exploit 绕过了 SELinux（比如改写了 policy），seccomp-bpf 还能提供什么保护？反过来，如果 seccomp 被绕过，SELinux 的保护还有效吗？

5. `vendor_domain` 不能访问 `core_data_file_type` 文件——但 vendor HAL server 需要读取 App 传入的 camera buffer，这是通过什么机制允许的？（提示：Binder/HwBinder + fd passing over IPC）
