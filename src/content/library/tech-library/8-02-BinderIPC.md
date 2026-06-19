---
title: "Binder IPC(Android 域)"
slug: "8-02"
collection: "tech-library"
group: "android系统"
order: 8002
summary: "读者画像:你写过多进程、用过 socket / pipe / shared memory、知道 `ioctl` 是什么、能读 C 和一点点内核代码。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T10:47:32.000Z"
updatedAt: "2026-06-12T10:47:32.000Z"
---
> 读者画像:你写过多进程、用过 socket / pipe / shared memory、知道 `ioctl` 是什么、能读 C 和一点点内核代码。本章不教"怎么 bindService 调一个 AIDL 方法"那一层应用语义,而是把 **一次跨进程方法调用从用户态 `transact()` 一路打到内核 `binder_transaction()` 再返回** 的整条链路、内核驱动的单拷贝/引用计数/线程池设计,逐层拆到源码级,并给你能在真机/模拟器上亲手观测 transaction 的脚本。
>
> **TL;DR(先给结论,再展开)**
> - Binder 不是"又一个 IPC",它是一个 **面向对象的、内核托管引用计数与生命周期的 RPC 框架**。传统 IPC(pipe/socket/SysV)解决"搬字节",Binder 解决"跨进程持有一个对象、调它的方法、对象死了通知我"。
> - **一次拷贝(single copy)** 是 Binder 区别于 socket(两次拷贝)的核心:内核把 sender 用户态 buffer 用 `copy_from_user`(精确说是 `binder_alloc_copy_user_to_buffer`)**直接写进 target 进程 `mmap` 出来的接收区**,target 读时零拷贝。【真实源码 torvalds/linux@drivers/android/binder.c】
> - 每个用 Binder 的进程在 `ProcessState` 单例里 `open("/dev/binder")` + `mmap` 一块 **约 1MB - 8KB** 的接收区(`BINDER_VM_SIZE`),这就是 `TransactionTooLargeException` 那个"1MB"的真身。【真实源码 cozybit/aosp-frameworks-base@libs/binder/ProcessState.cpp】
> - 用户态和内核之间所有交互都压进 **一个 `ioctl(fd, BINDER_WRITE_READ, &bwr)`**:`bwr.write_buffer` 里是一串 `BC_*` 命令(client→kernel),`bwr.read_buffer` 里读回一串 `BR_*` 命令(kernel→client)。一次 ioctl 可能既写又读。【真实源码 cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp】
> - **ServiceManager 是 handle 0**,是整个体系唯一靠"约定 magic handle"启动的节点,通过 `BINDER_SET_CONTEXT_MGR` 把自己注册成 context manager,其它服务都通过它做名字解析。
> - 服务端不是一个线程死循环 accept,而是一个 **由内核驱动调度的 binder 线程池**(libbinder 默认上限 15,system_server 抬到 31),驱动决定唤醒哪个空闲线程去执行 incoming transaction。
>
> **前置知识**:进程虚拟地址空间与 `mmap`、`ioctl` 系统调用语义、用户态/内核态拷贝(`copy_from_user`/`copy_to_user`)、引用计数(strong/weak ref)、Linux misc device 驱动模型。
>
> **本章用到的源码出处(均实际 WebFetch 取过,见章末"取材记录")**
> - 内核驱动:`torvalds/linux@drivers/android/binder.c`(v6.6 tag 对应行)
> - libbinder C++:`cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp` 与 `ProcessState.cpp`(这是一份 **较老的 AOSP 快照**,约 Android 4.x/Gingerbread 时代;选它是因为它的 transaction 主循环短小、未被后续 scatter-gather/RPC/locking 重构淹没,适合逐行读"骨架"。涉及现代差异处我会明确标注)
> - Java 框架:`aosp-mirror/platform_frameworks_base@core/java/android/os/Binder.java`(master,现代)
> - 官方文档:developer.android.com AIDL、source.android.com binder-ipc / aidl overview

---

## 一、设计考古:Binder 为什么不是"又一个 socket"

### 1.1 血统:BeOS → OpenBinder → Android

Binder 不是 Android 原创。它的血统是一条很长的线【出处:多方二手史料,见下,标「待核」于一手时间点】:

- **2001 前后,Be Inc.**:为"下一代 BeOS"设计了一套面向对象的 IPC,核心思想是"跨进程持有对象引用、调方法"。
- **PalmSource 收购 Be**:这套东西用在 Palm Cobalt 上;Palm 后来转向 Linux,把它移植过去并开源,叫 **OpenBinder**(主设计者 **Dianne Hackborn**)。
- **2006,Google 雇了 Dianne Hackborn**,Android 采用 Binder 作为 IPC 基础。**内核驱动部分**沿用了 OpenBinder 的设计,但 **用户态部分因为 OpenBinder 的 license 与 Android 不兼容被重写**。

> 【真实史料 二手,WebSearch 命中多篇一致复述,关键时间点标「待核」】这条血统在多份独立资料(LinkedIn 技术长文、Vanderbilt CS282 课件 PDF、Marko Gargenta 2013 Android Builders Summit 的 "Deep Dive into Android IPC/Binder Framework" 课件)里互相印证。要追到 **一手** 出处,应查 OpenBinder 官方文档(Hackborn 写的 "The OpenBinder Programming Guide")与早期 AOSP commit history,本章未能直接 WebFetch 到该一手文档,标「待核」。

设计考古的意义不在八卦,而在于:**Binder 的"对象语义 + 引用计数 + 死亡通知"是它娘胎里带的,不是 Android 后加的**。这解释了为什么 Binder 的内核结构里满是 `binder_node`(对象本体)、`binder_ref`(对象引用)、strong/weak count、death notification —— 这些都是"跨进程对象系统"的标配,而不是"消息队列"的标配。

### 1.2 为什么不用传统 IPC:四个硬约束

把 Binder 和传统手段(pipe、SysV message queue、UNIX domain socket、shared memory)对照,Android 的场景有四个传统手段都不舒服的硬约束:

| 约束 | 传统 IPC 的难处 | Binder 怎么解 |
|---|---|---|
| **对象语义 / RPC** | socket/pipe 只搬字节流,"调用远端对象的某个方法"得自己造一套 marshalling + 方法分发 + 句柄管理 | 内核驱动原生理解"对象引用"(`flat_binder_object`),AIDL 自动生成 marshalling,`onTransact` 做方法分发 |
| **拷贝开销** | socket 走内核 = 用户态→内核态→用户态 **两次拷贝** | **一次拷贝**:sender 用户态直接拷进 target 的 `mmap` 接收区 |
| **生命周期 / 死亡通知** | 进程在 lowmemorykiller 下随时被杀,socket 对端死了你只能靠 read 返回 0 兜底,且无法跨进程做"对象引用计数" | 内核给每个 binder 对象维护跨进程引用计数;对端死亡时主动投递 **death notification**(`BR_DEAD_BINDER`) |
| **身份与安全** | socket 拿不到对端可信的 uid/pid(`SO_PEERCRED` 有,但不贯穿调用链) | 内核在每次 transaction 里 **盖上 sender 的真实 pid/euid**(下文 `tr.sender_pid`/`sender_euid`),`Binder.getCallingUid()` 因此可信 |

最后一条尤其关键:**Android 的整个权限模型(permission check)建立在"被调用方能拿到调用方可信 uid"之上**。`checkCallingPermission()` 之所以不能被伪造,是因为 uid 是内核在 transaction 里盖的章,不是用户态传进来的参数。这是 socket 给不了的。

> 设计动机一句话:**Android 要的不是"进程间传消息",而是"进程间共享一个有生命周期、有身份、能被引用计数的对象系统",还要省一次拷贝、还要在内存随时被杀的敌对环境下活下来。** 传统 IPC 没有一个同时满足这四点,所以重造。

---

## 二、整体架构:四层 + 一次调用的全链路

### 2.1 四层栈

一次"client 调 server 的方法"穿过这些层:

```
┌─────────────────────────────────────────────────────────┐
│  Java 应用层    IRemoteService.Stub.asInterface(binder)   │  ← AIDL 生成的 Proxy/Stub
│                 proxy.getPid()                            │
├─────────────────────────────────────────────────────────┤
│  Java 框架层    BinderProxy.transactNative()              │  android_util_Binder.cpp (JNI)
│                 Binder.execTransact()  ← server 侧入口     │
├─────────────────────────────────────────────────────────┤
│  Native libbinder   IPCThreadState::transact()           │  C++:打包 Parcel→binder_transaction_data
│                     ProcessState (单例,持 /dev/binder fd) │      ioctl(BINDER_WRITE_READ)
├─────────────────────────────────────────────────────────┤
│  内核 Binder 驱动   binder_transaction()                  │  drivers/android/binder.c
│                    /dev/binder /dev/hwbinder /dev/vndbinder│      单拷贝 + 对象翻译 + 线程调度
└─────────────────────────────────────────────────────────┘
```

三个 binder context 域(现代 AOSP,Treble 之后)【真实文档 source.android.com/docs/core/architecture/hidl/binder-ipc】:

> - **/dev/binder**:framework / app 进程之间,走 AIDL。
> - **/dev/hwbinder**:framework/vendor 与 vendor/vendor 之间,走 HIDL。
> - **/dev/vndbinder**:vendor/vendor 之间,走 AIDL。

三个域互相隔离,各有独立的 context manager,目的是 **Treble 架构下把 vendor 分区和 system 分区的 IPC 命名空间切开**,避免 vendor 进程直接看到 framework 的服务。

### 2.2 一次同步调用的全链路(client 侧)

这是本章的主轴。下面的真实源码来自 `cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp`(老 AOSP,骨架清晰),逐段读。

**第 1 步:`transact()` —— 入口,决定同步还是 oneway**

```cpp
status_t IPCThreadState::transact(int32_t handle,
                                  uint32_t code, const Parcel& data,
                                  Parcel* reply, uint32_t flags)
{
    status_t err = data.errorCheck();
    flags |= TF_ACCEPT_FDS;                          // 允许这次 transaction 传文件描述符
    // ...
    if (err == NO_ERROR) {
        // 把这次调用编码进 mOut(输出缓冲),命令字是 BC_TRANSACTION
        err = writeTransactionData(BC_TRANSACTION, flags, handle, code, data, NULL);
    }
    if (err != NO_ERROR) {
        if (reply) reply->setError(err);
        return (mLastError = err);
    }

    if ((flags & TF_ONE_WAY) == 0) {                 // 同步调用:必须等 reply
        if (reply) {
            err = waitForResponse(reply);
        } else {
            Parcel fakeReply;
            err = waitForResponse(&fakeReply);
        }
        // ...
    } else {                                         // oneway:不等 reply,fire-and-forget
        err = waitForResponse(NULL, NULL);
    }
    return err;
}
```
【真实源码 cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp,IPCThreadState::transact】

逐行要点:
- `handle` 是 **目标对象在本进程的句柄**(ServiceManager 是 0;其它服务是你 `getService` 拿回来的句柄)。
- `code` 是方法号(AIDL 里 `TRANSACTION_getPid = FIRST_CALL_TRANSACTION + 0` 这种)。
- `flags & TF_ONE_WAY` 决定走 **同步等 reply** 还是 **oneway**。同步调用会 **阻塞调用线程** 直到 `waitForResponse` 拿到 `BR_REPLY`。这就是文档反复警告"别在主线程做远程 IPC"的物理原因 —— 你的 UI 线程会卡在这个 `waitForResponse` 里。
- 注意:**oneway 也会 `waitForResponse`**,但它只是去把驱动里的 `BR_TRANSACTION_COMPLETE` 收掉(确认驱动已受理),不等对端执行完。

**第 2 步:`writeTransactionData()` —— 把调用打包成 `binder_transaction_data`**

```cpp
binder_transaction_data tr;

tr.target.handle = handle;        // 目标对象句柄
tr.code = code;                   // 方法号
tr.flags = binderFlags;
tr.cookie = 0;
tr.sender_pid = 0;                // 注意:用户态填 0!真实 pid 由内核盖章
tr.sender_euid = 0;              // 同上,用户态填 0,内核覆盖

const status_t err = data.errorCheck();
if (err == NO_ERROR) {
    tr.data_size  = data.ipcDataSize();      // Parcel 的数据区大小
    tr.data.ptr.buffer  = data.ipcData();    // 指向用户态 Parcel 数据(还在 sender 进程)
    tr.offsets_size = data.ipcObjectsCount()*sizeof(size_t);
    tr.data.ptr.offsets = data.ipcObjects(); // 指向"哪些位置是 binder 对象"的偏移表
}
// ...
mOut.writeInt32(cmd);             // cmd = BC_TRANSACTION,先写命令字
mOut.write(&tr, sizeof(tr));      // 再把 tr 结构体追加进输出缓冲
```
【真实源码 cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp,IPCThreadState::writeTransactionData】

这里有三个 **必须吃透的点**:

1. **`sender_pid` / `sender_euid` 用户态填 0**。安全性的根:身份不是用户态传的,后面你会在内核侧看到驱动用 `task_tgid_nr(proc->tsk)` 之类把真实身份盖进去。**应用层永远没法伪造 calling uid。**
2. **`tr.data.ptr.buffer` 此刻还指向 sender 自己进程的用户态内存**。数据还没动。真正的拷贝发生在内核 `binder_transaction()` 里(下一节)。
3. **`offsets` 表是 Binder 对象语义的关键**。Parcel 里可能夹着 binder 对象(比如你把一个 callback 接口当参数传过去)。`offsets` 告诉驱动"data buffer 的第 X、Y 字节处是 `flat_binder_object`,你得做对象翻译(本进程对象→句柄)",而不是当普通字节拷过去。这就是 Binder 能"传对象"而 socket 不能的实现机制。

**第 3 步:`talkWithDriver()` —— 唯一的内核交互点,`ioctl(BINDER_WRITE_READ)`**

```cpp
binder_write_read bwr;

// 读缓冲是不是空的?
const bool needRead = mIn.dataPosition() >= mIn.dataSize();
// 如果还在读上次没读完的数据、且调用方要读下一批,就先别写
const size_t outAvail = (!doReceive || needRead) ? mOut.dataSize() : 0;

bwr.write_size   = outAvail;
bwr.write_buffer = (long unsigned int)mOut.data();   // BC_* 命令流:用户→内核

if (doReceive && needRead) {
    bwr.read_size   = mIn.dataCapacity();
    bwr.read_buffer = (long unsigned int)mIn.data();  // BR_* 命令流:内核→用户
} else {
    bwr.read_size = 0;
    bwr.read_buffer = 0;
}

bwr.write_consumed = 0;
bwr.read_consumed  = 0;
status_t err;
do {
#if defined(HAVE_ANDROID_OS)
    if (ioctl(mProcess->mDriverFD, BINDER_WRITE_READ, &bwr) >= 0)   // ← 唯一一次进内核
        err = NO_ERROR;
    else
        err = -errno;
#else
    err = INVALID_OPERATION;
#endif
    // ...
```
【真实源码 cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp,IPCThreadState::talkWithDriver】

这是整个 Binder 用户态最核心的一行:**所有跨进程交互都是 `ioctl(fd, BINDER_WRITE_READ, &bwr)`**。

- `bwr` 是双向的:`write_buffer` 把攒在 `mOut` 里的一串 `BC_*` 命令(这次是 `BC_TRANSACTION` + `tr`)灌进内核;`read_buffer` 让内核把这次的回包(`BR_*` 命令)写回 `mIn`。
- **一次 ioctl 可以同时完成"写出请求"和"读回结果"**,减少 syscall 次数。这是 Binder 协议设计的精妙处:不是"写一个 syscall、读一个 syscall",而是把读写合并。
- `(long unsigned int)` 这种裸 cast 暴露了这是 **32 位时代的老代码**;现代 AOSP 用 `binder_uintptr_t`。语义不变。

**第 4 步:`waitForResponse()` —— 解析内核回包,等 `BR_REPLY`**

```cpp
switch (cmd) {
case BR_TRANSACTION_COMPLETE:
    if (!reply && !acquireResult) goto finish;   // oneway:收到"已受理"就够了,返回
    break;                                        // 同步:继续等真正的 BR_REPLY

case BR_DEAD_REPLY:
    err = DEAD_OBJECT;                            // 对端进程死了
    goto finish;

case BR_FAILED_REPLY:
    err = FAILED_TRANSACTION;
    goto finish;

case BR_REPLY:
    {
        binder_transaction_data tr;
        err = mIn.read(&tr, sizeof(tr));
        // ...
        if (reply) {
            if ((tr.flags & TF_STATUS_CODE) == 0) {
                // 关键:reply 的数据直接引用内核回填的 buffer,零拷贝!
                reply->ipcSetDataReference(
                    reinterpret_cast<const uint8_t*>(tr.data.ptr.buffer),
                    tr.data_size,
                    reinterpret_cast<const size_t*>(tr.data.ptr.offsets),
                    tr.offsets_size/sizeof(size_t),
                    freeBuffer, this);            // freeBuffer:Parcel 析构时通知内核回收
            } else {
                err = *static_cast<const status_t*>(tr.data.ptr.buffer);
                // ...
            }
        }
        // ...
    }
    goto finish;
```
【真实源码 cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp,IPCThreadState::waitForResponse】

要点:
- **`BR_TRANSACTION_COMPLETE` 和 `BR_REPLY` 是两个事件**。同步调用会先收到 `COMPLETE`(驱动已把请求投递给对端)、再收到 `REPLY`(对端执行完回的数据)。oneway 只等 `COMPLETE`。
- **`reply->ipcSetDataReference(...)` 是"读侧零拷贝"的体现**:reply 的数据 `tr.data.ptr.buffer` 指向的是 **内核回填进本进程 `mmap` 接收区的地址**,Parcel 不复制,直接引用。等 Parcel 析构,`freeBuffer` 回调发 `BC_FREE_BUFFER` 通知内核回收这块接收区。**这就是为什么忘记 `reply.recycle()` 会泄漏那 1MB 接收区。**
- `BR_DEAD_REPLY` → `DEAD_OBJECT` → Java 层抛 `DeadObjectException`。这是"对端进程在调用过程中死了"的标准路径。

> **client 侧链路小结**:`transact` → `writeTransactionData`(打包 `binder_transaction_data`,身份留空) → `talkWithDriver`(`ioctl(BINDER_WRITE_READ)`,`BC_TRANSACTION` 进、`BR_*` 出) → `waitForResponse`(等 `BR_REPLY`,数据零拷贝引用接收区)。**整条链路只有一次 syscall 进内核,只有一次数据拷贝(发生在内核内)。**

---

## 三、内核驱动:单拷贝、对象翻译、线程调度

现在进内核。源码来自 `torvalds/linux@drivers/android/binder.c`(现代,v6.6 tag 对应)。

### 3.1 驱动自我描述(顶部注释,逐字)

```c
// SPDX-License-Identifier: GPL-2.0-only
/* binder.c
 *
 * Android IPC Subsystem
 *
 * Copyright (C) 2007-2008 Google, Inc.
 */

/*
 * Locking overview
 *
 * There are 3 main spinlocks which must be acquired in the
 * order shown:
 *
 * 1) proc->outer_lock : protects binder_ref
 *    binder_proc_lock() and binder_proc_unlock() are
 *    used to acq/rel.
 * 2) node->lock : protects most fields of binder_node.
 *    binder_node_lock() and binder_node_unlock() are
 *    used to acq/rel
 * 3) proc->inner_lock : protects the thread and node lists
 *    (proc->threads, proc->waiting_threads, proc->nodes)
 *    and all todo lists associated with the binder_proc
 *    (proc->todo, thread->todo, proc->delivered_death and
 *    node->async_todo), as well as thread->transaction_stack
 *    binder_inner_proc_lock() and binder_inner_proc_unlock()
 *    are used to acq/rel
 *
 * Any lock under procA must never be nested under any lock at the same
 * level or below on procB.
 * ...
 */
```
【真实源码 torvalds/linux@drivers/android/binder.c,文件顶部注释】

读这段注释你能直接读出内核侧的 **核心数据结构骨架**:
- `binder_proc`:一个使用 Binder 的进程在驱动里的代表。持有 `threads`(binder 线程池)、`nodes`(本进程暴露的对象)、`todo`(待处理事务队列)。
- `binder_node`:一个 binder **对象本体**(server 侧)。
- `binder_ref`:一个对象的 **引用**(client 侧持有的句柄在内核里的实体)。
- `thread->transaction_stack`:**同步调用栈**。A 调 B、B 回调 A 时,驱动要保证"B 的回调走回 A 那个正在等待的线程",这个栈就是干这个的(避免死锁、保证调用线程身份连续)。

> **设计动机(锁的演进)**:这套"三把 spinlock + 命名后缀标注持锁"的精细锁,是 **Android 8 引入的 fine-grained locking**(之前是一把 `binder_main_lock` 全局大锁)。官方文档把它和 scatter-gather 一起列为 O 版本的 binder 性能改进。粗锁时代 binder 在高并发下锁竞争严重,细锁化后 system_server 这种 binder 风暴进程受益明显。

### 3.2 单拷贝(single copy)的真身

这是 Binder 最被津津乐道、也最常被讲错的点。先看 **target 接收区怎么来**:每个进程在 `ProcessState` 构造时 `mmap` 一块接收区(下一节细看)。驱动侧用 `binder_alloc` 管理这块区域,在每次 transaction 里 `binder_alloc_new_buf()` 从中切一块给本次事务。

```c
	trace_binder_transaction(reply, t, target_node);

	t->buffer = binder_alloc_new_buf(&target_proc->alloc, tr->data_size,
		tr->offsets_size, extra_buffers_size,
		!reply && (t->flags & TF_ONE_WAY));
	if (IS_ERR(t->buffer)) {
		char *s;
		ret = PTR_ERR(t->buffer);
		s = (ret == -ESRCH) ? ": vma cleared, target dead or dying"
			: (ret == -ENOSPC) ? ": no space left"      // ← 接收区满 = TransactionTooLarge 的内核根
			: (ret == -ENOMEM) ? ": memory allocation failed"
			: "";
		// ...
		return_error = return_error_param == -ESRCH ?
			BR_DEAD_REPLY : BR_FAILED_REPLY;
		// ...
	}
```
【真实源码 torvalds/linux@drivers/android/binder.c,binder_transaction()】

注意 `binder_alloc_new_buf` 的第一个参数是 **`&target_proc->alloc`** —— 在 **目标进程** 的接收区里分配 `t->buffer`。这块 buffer 同时映射在目标进程用户态。所以接下来把数据拷进 `t->buffer`,就等于 **直接拷进了目标进程能直接读的内存**。

然后是 **那次唯一的拷贝**:

```c
	/*
	 * Copy the source user buffer up to the next object
	 * that will be processed.
	 */
	copy_size = object_offset - user_offset;
	if (copy_size && (user_offset > object_offset ||
			binder_alloc_copy_user_to_buffer(
				&target_proc->alloc,
				t->buffer, user_offset,
				user_buffer + user_offset,    // ← 源:sender 用户态 buffer
				copy_size))) {                 // ← 直接拷进 target 的 t->buffer
		binder_user_error("%d:%d got transaction with invalid data ptr\n",
				proc->pid, thread->pid);
```
【真实源码 torvalds/linux@drivers/android/binder.c,binder_transaction(),v6.6 的分段拷贝版本】

以及 offsets 表的拷贝:

```c
	if (binder_alloc_copy_user_to_buffer(
				&target_proc->alloc,
				t->buffer,
				ALIGN(tr->data_size, sizeof(void *)),
				(const void __user *)
					(uintptr_t)tr->data.ptr.offsets,   // 源:sender 用户态 offsets
				tr->offsets_size)) {
		binder_user_error("%d:%d got transaction with invalid offsets ptr\n",
				proc->pid, thread->pid);
```
【真实源码 torvalds/linux@drivers/android/binder.c,binder_transaction()】

**把"一次拷贝"讲准确**(这是面试和实战都容易翻车的点):

- 拷贝的是:**sender 用户态 Parcel buffer → target 进程接收区(`t->buffer`)**,用 `binder_alloc_copy_user_to_buffer`(底层 `copy_from_user`),**1 次**。
- target 读数据:`t->buffer` 已经映射在 target 用户态,target 进程的 Parcel 直接引用,**0 次拷贝**(就是上一节 client 侧 `ipcSetDataReference` 的对端版本)。
- 对比 socket:`write()` 把 sender 用户态拷进内核 socket buffer(1 次)、`read()` 把内核 buffer 拷回 receiver 用户态(1 次),**2 次**。
- **所以"一次拷贝"指的是:全链路只有内核里那一次 `copy_from_user`,sender 写出和 receiver 读入合起来只发生一次内存复制。**

> ⚠ 现代版本细节:v6.6 这里是 **分段拷贝(scatter-gather)**:不是一次性把整个 buffer 拷完,而是"拷到下一个 binder 对象前的那段、翻译对象、再拷下一段",`user_offset`/`object_offset` 就是分段游标。这是 Android 8 "scatter-gather 把拷贝从 3 次降到 1 次" 的内核实现【真实文档 source.android.com:"Android 8 uses scatter-gather optimization to reduce the number of copies from 3 to 1."】。老 AOSP 是一次性整块拷,概念上同样是"一次拷贝"。

> 把官方那句"3 → 1"讲清楚:**没优化前** 的 3 次是 ① 调用方把对象序列化进 Parcel(用户态自拷) ② 内核拷一次 ③ 目标进程反序列化时再拷。scatter-gather 让数据 **保持原始内存布局**,驱动直接从原结构拷进目标,省掉 ①③ 那两次用户态自拷。

### 3.3 对象翻译:Binder 凭什么能"传对象"

`offsets` 表指向的每个位置是一个 `flat_binder_object`。驱动在拷贝过程中对它们做 **翻译**(`binder_translate_binder` / `binder_translate_handle`):

- **sender 传出一个本进程的 binder 对象**(`type = BINDER_TYPE_BINDER`,带本进程虚拟地址):驱动查/建对应的 `binder_node`,增加引用计数,然后把 `flat_binder_object` 改写成 `type = BINDER_TYPE_HANDLE` + 一个 **target 进程视角的句柄号**。target 收到的是句柄,不是 sender 的地址。
- **反向**(`BINDER_TYPE_HANDLE` 传回它的属主进程):驱动还原成 `BINDER_TYPE_BINDER` + 原始地址。

【真实机制 二手交叉核对 synacktiv "Binder transactions in the bowels of the Linux Kernel",字段名 `fp->hdr.type` / `fp->handle` / `fp->binder` / `fp->cookie` 与内核 `flat_binder_object` 一致】

这就是"传对象"的真相:**跨进程传的从来不是对象本体,而是内核做的"地址 ↔ 句柄"翻译 + 引用计数**。client 拿到的句柄,在内核里对应一个 `binder_ref`,指向 server 进程里的 `binder_node`。client 调用时填 `tr.target.handle = 句柄`,驱动据此找到 server 进程、唤醒它的 binder 线程。

> **这也是引用计数的落点**:翻译时 `BC_ACQUIRE`/`BC_INCREFS` 增加 strong/weak count,最后一个引用释放时 `BC_RELEASE`/`BC_DECREFS`。引用计数归零且属主进程在,`binder_node` 才可回收;属主进程先死,则向所有持有 `binder_ref` 的进程投递 `BR_DEAD_BINDER`(death notification)。**这套"内核托管的跨进程对象生命周期"就是 1.1 里 BeOS 血统带来的东西。**

### 3.4 线程模型:server 不是 accept 死循环

server 侧不存在"一个线程 `accept` 然后 dispatch"。server 进程启动时调 `joinThreadPool`,把当前线程作为 binder 线程交给驱动调度:

```cpp
void IPCThreadState::joinThreadPool(bool isMain)
{
    // 告诉驱动:我这条线程加入 looper(主线程 BC_ENTER_LOOPER,新拉起的 BC_REGISTER_LOOPER)
    mOut.writeInt32(isMain ? BC_ENTER_LOOPER : BC_REGISTER_LOOPER);

    status_t result;
    do {
        // ... 处理待 deref 的弱/强引用 ...
        result = talkWithDriver();              // 阻塞在 ioctl(BINDER_WRITE_READ) 上等活干
        if (result >= NO_ERROR) {
            size_t IN = mIn.dataAvail();
            if (IN < sizeof(int32_t)) continue;
            cmd = mIn.readInt32();
            result = executeCommand(cmd);       // 收到 BR_TRANSACTION → 执行
        }
        // 非主线程超时就退出(线程池可收缩)
        if(result == TIMED_OUT && !isMain) {
            break;
        }
    } while (result != -ECONNREFUSED && result != -EBADF);

    mOut.writeInt32(BC_EXIT_LOOPER);
    talkWithDriver(false);
}
```
【真实源码 cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp,IPCThreadState::joinThreadPool】

模型本质:
- **每条 binder 线程都阻塞在自己的 `ioctl(BINDER_WRITE_READ)` 上**,等驱动给它派 `BR_TRANSACTION`。
- **是驱动在调度**:有 incoming transaction 时,驱动从 target 进程的 `waiting_threads` 里挑一条空闲线程唤醒(优先复用刚处理完的线程,缓存热),把事务放进 `thread->todo`。
- **线程池按需伸缩**:不够用时驱动发 `BR_SPAWN_LOOPER`,libbinder 起新线程 `BC_REGISTER_LOOPER`;空闲非主线程超时自行退出(`TIMED_OUT && !isMain`)。
- **上限**:libbinder 默认 15(下一节 `ProcessState`),system_server 显式抬到 31。**这就是 binder 风暴 / 全线程占满会导致系统级卡顿的根**:当一个热门服务(如 AMS)的 31 条 binder 线程全被慢调用占满,新请求只能在驱动 `todo` 队列里排队,表现为"系统假死"。

收到事务后 `executeCommand` 的 `BR_TRANSACTION` 分支真正分发到对象:

```cpp
    case BR_TRANSACTION:
        {
            binder_transaction_data tr;
            result = mIn.read(&tr, sizeof(tr));
            // ...
            Parcel buffer;
            buffer.ipcSetDataReference(            // 又是零拷贝引用接收区
                reinterpret_cast<const uint8_t*>(tr.data.ptr.buffer),
                tr.data_size,
                reinterpret_cast<const size_t*>(tr.data.ptr.offsets),
                tr.offsets_size/sizeof(size_t), freeBuffer, this);

            mCallingPid = tr.sender_pid;           // ← 此刻 sender_pid 是内核盖过章的真实值!
            mCallingUid = tr.sender_euid;          // getCallingUid() 读的就是它

            Parcel reply;
            if (tr.target.ptr) {
                sp<BBinder> b((BBinder*)tr.cookie);
                // 分发到具体对象:BBinder::transact → onTransact → AIDL Stub 的方法分发
                const status_t error = b->transact(tr.code, buffer, &reply, tr.flags);
                if (error < NO_ERROR) reply.setError(error);
            } else {
                // target.ptr == 0 即 handle 0:ServiceManager(context manager)
                const status_t error = the_context_object->transact(tr.code, buffer, &reply, tr.flags);
                // ...
            }

            if ((tr.flags & TF_ONE_WAY) == 0) {
                sendReply(reply, 0);               // 同步调用:把 reply 发回去(BC_REPLY)
            }
            // ...
        }
        break;
```
【真实源码 cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp,IPCThreadState::executeCommand,BR_TRANSACTION 分支】

**两个收尾要点**:
1. `mCallingPid = tr.sender_pid` —— 印证 2.2 的安全性闭环:**用户态发送时填 0,内核盖真实身份,接收侧读到的是可信值**。`IPCThreadState::getCallingUid()` 返回的就是这个 `mCallingUid`。权限检查的整个安全模型在此落地。
2. `target.ptr == 0`(即 handle 0)走 `the_context_object`,这就是 **ServiceManager 作为 context manager** 的特殊路径。其它对象走 `BBinder::transact → onTransact`。

到 Java 层,binder 线程经 JNI 进入 `Binder.execTransact`:

```java
@UnsupportedAppUsage
private boolean execTransact(int code, long dataObj, long replyObj, int flags) {
    Parcel data = Parcel.obtain(dataObj);
    Parcel reply = Parcel.obtain(replyObj);

    final int callingUid = data.isForRpc() ? -1 : Binder.getCallingUid();
    final long origWorkSource = callingUid == -1
            ? -1 : ThreadLocalWorkSource.setUid(callingUid);
    try {
        return execTransactInternal(code, data, reply, flags, callingUid);
    } finally {
        reply.recycle();      // ← 重要:回收 reply Parcel,释放接收区
        data.recycle();
        if (callingUid != -1) {
            ThreadLocalWorkSource.restore(origWorkSource);
        }
    }
}
// execTransactInternal 内部最终: res = onTransact(code, data, reply, flags);
// onTransact 由 AIDL 生成的 Stub override,按 code 分发到具体方法
protected boolean onTransact(int code, @NonNull Parcel data, @Nullable Parcel reply,
        int flags) throws RemoteException
```
【真实源码 aosp-mirror/platform_frameworks_base@core/java/android/os/Binder.java,execTransact / onTransact】

`execTransact` 是 **JNI 从 native binder 线程回调进 Java 的入口**。它 `obtain` 两个 Parcel(包裹 native 的 data/reply),调 `onTransact`(AIDL Stub override 的方法,按 `code` switch 到 `getPid()` 等具体实现),`finally` 里 `recycle` 回收。AIDL 自动生成的 `Stub.onTransact` 就是那个 `switch(code)` 方法分发表。

---

## 四、`ProcessState`:那块 1MB 接收区和 15 个线程从哪来

```cpp
#define BINDER_VM_SIZE ((1*1024*1024) - (4096 *2))     // = 1MB - 8KB ≈ 1040384 字节

static int open_driver()
{
    int fd = open("/dev/binder", O_RDWR);
    if (fd >= 0) {
        fcntl(fd, F_SETFD, FD_CLOEXEC);                // exec 时关闭,防 fd 泄漏给子进程
        int vers;
        status_t result = ioctl(fd, BINDER_VERSION, &vers);   // 校验内核/用户态协议版本一致
        // ... 版本不匹配则 close ...
        size_t maxThreads = 15;                        // ← 默认 binder 线程上限 15
        result = ioctl(fd, BINDER_SET_MAX_THREADS, &maxThreads);
        // ...
    }
    return fd;
}

ProcessState::ProcessState()
    : mDriverFD(open_driver())
    , mVMStart(MAP_FAILED)
    // ...
{
    if (mDriverFD >= 0) {
        // mmap 出接收区:驱动据此分配本进程的 transaction 接收内存
        mVMStart = mmap(0, BINDER_VM_SIZE, PROT_READ,
                        MAP_PRIVATE | MAP_NORESERVE, mDriverFD, 0);
        if (mVMStart == MAP_FAILED) {
            LOGE("Using /dev/binder failed: unable to mmap transaction memory.\n");
            close(mDriverFD);
            mDriverFD = -1;
        }
    }
    LOG_ALWAYS_FATAL_IF(mDriverFD < 0, "Binder driver could not be opened.  Terminating.");
}
```
【真实源码 cozybit/aosp-frameworks-base@libs/binder/ProcessState.cpp,BINDER_VM_SIZE / open_driver / 构造函数】

这一段把几个"江湖传说"钉死成事实:

1. **`BINDER_VM_SIZE = (1*1024*1024) - (4096*2)` ≈ 1MB - 8KB**。这就是 `TransactionTooLargeException` 文档里那个"约 1MB"的物理来源 —— 每个进程的 **binder 接收区总大小**。【关联文档 developer.android.com/reference/android/os/TransactionTooLargeException,原文 "The Binder transaction buffer has a limited fixed size, currently 1Mb" —— 该页 WebFetch 渲染受限未取到逐字,标「待核」,但与本源码常量一致】
2. **`PROT_READ` + `MAP_NORESERVE`**:用户态对接收区 **只读**(数据由内核写入,用户态只读取),`MAP_NORESERVE` 不预留 swap。
3. **`maxThreads = 15`**:`BINDER_SET_MAX_THREADS` 把上限告诉内核。现代 libbinder 把这个常量叫 `DEFAULT_MAX_BINDER_THREADS = 15`,并提供 `setThreadPoolMaxThreadCount()` 让进程改(system_server 改成 31)。【真实事实 WebSearch 命中 frameworks/native ProcessState,DEFAULT_MAX_BINDER_THREADS=15;system_server 抬到 31】
4. **`ProcessState` 是进程单例**:每个进程 **只 open 一次 /dev/binder、只 mmap 一块接收区、只有一个 binder 线程池**。这是为什么"binder fd 泄漏"通常意味着有人手动开了第二个 /dev/binder。

> **把 1MB 的语义讲准(避免常见误区)**:这 1MB 是 **整个进程所有并发 in-flight transaction 共享** 的接收区,**不是单次调用上限**。所以 `TransactionTooLargeException` 的真实触发条件常常是"这一刻多个大事务并发挤爆了接收区"或"你单次塞了个几百 KB 的 Bitmap/大 List",而 **不是** "单次必须 < 1MB" 这么简单。oneway 事务还只能用接收区的一半(驱动对异步事务有额外限额),更容易爆。

---

## 五、⭐ 可运行 demo:亲手发起并观测一次 Binder transaction

> **本节所有命令需 Android 模拟器或真机验证。** 前置:① 装好 `adb`(`platform-tools`)并 `adb devices` 能看到设备;② 推荐用 **可 root 的 emulator**(AVD 选 *不带 Google Play* 的系统镜像,这种镜像 `adb root` 可用),很多观测命令(ftrace、读 `/sys/kernel/debug`)需要 root。③ Android 10+。

下面给 **三个递进的 demo**,从"零代码纯观测"到"自己写 AIDL service 再观测自己的 transaction"。

### Demo A(零代码,5 分钟):用 dumpsys / service 观测系统 Binder 状态

**A1. 列出所有已注册的系统服务(它们都是 binder 对象,挂在 ServiceManager 即 handle 0 下)**

```bash
adb shell service list
```
预期输出(节选,你机器上服务名会更多):
```
Found 200+ services:
0   DisplayFeatureControl: [...]
... 
23  activity: [android.app.IActivityManager]
...
87  package: [android.content.pm.IPackageManager]
...
```
每一行就是一个 binder 服务,中括号里是它的 **AIDL interface descriptor**。`activity` 就是 AMS,`package` 就是 PMS。这些名字到句柄的映射,正是 ServiceManager(handle 0)维护的。

**A2. 直接对一个系统服务发起一次真实 transaction(命令行版 `transact`)**

`service call <name> <code>` 会真的走一遍 Binder:打包 Parcel → `BC_TRANSACTION` → 内核 → 服务 `onTransact` → `BR_REPLY`。

```bash
# 对 SurfaceFlinger 调 code=1000(各服务 code 含义见其 AIDL,这里仅演示链路真实发生)
adb shell service call SurfaceFlinger 1000
```
预期:返回一个 Parcel 的 hex dump,例如
```
Result: Parcel(00000000 00000001   '........')
```
**这就是一次完整的、真实的 Binder 往返。** `service call` 内部就是构造一个 transaction 打给目标 handle。

**A3. 看某个进程的 binder 线程池现状(需 root)**

```bash
adb root
# system_server 的 pid
adb shell pidof system_server
# 读它的 binder 调试信息:线程数、ready_threads、max_threads
adb shell "cat /sys/kernel/debug/binder/proc/$(pidof system_server)"
```
预期输出里能看到类似(字段名以你内核版本为准):
```
proc <pid>
  context binder
  threads: 31            ← 印证 system_server 把上限抬到了 31
  requested_threads: 0 + ...
  ready_threads N
  free_async_space ...   ← 这就是 oneway 事务可用的"半个接收区"剩余量
  ...
```
> 注:`/sys/kernel/debug/binder/` 在部分较新内核/设备上被移除或路径变化(GKI / debugfs 收紧)。取不到时用下面 D 节的 perfetto/atrace。标「待核(取决于设备内核配置)」。

### Demo B(自己写 AIDL,完整可编译):一个跨进程 AIDL service + client

**目标**:写一个跑在独立进程的 service,暴露 `getPid()`,client 调用它,**让 client 和 service 的 pid 不同**,亲眼确认这是跨进程调用。代码基于 **官方 AIDL 文档** 给的骨架【真实文档 developer.android.com/develop/background-work/services/aidl】。

**B1. AIDL 接口** `src/main/aidl/com/example/binderdemo/IRemoteService.aidl`
```java
// IRemoteService.aidl
package com.example.binderdemo;

/** Example service interface */
interface IRemoteService {
    /** Request the process ID of this service, to do evil things with it. */
    int getPid();

    /** 演示基本类型 marshalling(直接抄官方示例) */
    void basicTypes(int anInt, long aLong, boolean aBoolean, float aFloat,
            double aDouble, String aString);
}
```

**B2. service 实现**(Kotlin)`RemoteService.kt`
```kotlin
package com.example.binderdemo

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.Process

class RemoteService : Service() {
    // Stub 就是 BBinder 的 Java 封装,onTransact 由 AIDL 生成
    private val binder = object : IRemoteService.Stub() {
        override fun getPid(): Int = Process.myPid()   // 返回 service 进程的 pid
        override fun basicTypes(
            anInt: Int, aLong: Long, aBoolean: Boolean,
            aFloat: Float, aDouble: Double, aString: String
        ) { /* no-op */ }
    }
    override fun onBind(intent: Intent): IBinder = binder
}
```

**B3. 让 service 跑在独立进程** `AndroidManifest.xml`(关键:`android:process`)
```xml
<service
    android:name=".RemoteService"
    android:process=":remote"          <!-- 冒号:独立私有进程,这样 pid 才会不同 -->
    android:exported="false" />
```

**B4. client 绑定并调用**(Activity 里)
```kotlin
private var remote: IRemoteService? = null

private val conn = object : android.content.ServiceConnection {
    override fun onServiceConnected(name: android.content.ComponentName, service: android.os.IBinder) {
        // asInterface:如果同进程返回本地对象,跨进程返回 BinderProxy 包装
        remote = IRemoteService.Stub.asInterface(service)
        val myPid = android.os.Process.myPid()
        val svcPid = remote?.getPid()                 // ← 一次真实的跨进程 Binder 调用
        android.util.Log.i("BinderDemo", "client pid=$myPid  service pid=$svcPid")
        // 预期:两个 pid 不同 → 证明走了 Binder 跨进程
    }
    override fun onServiceDisconnected(name: android.content.ComponentName) { remote = null }
}

override fun onStart() {
    super.onStart()
    bindService(Intent(this, RemoteService::class.java), conn, BIND_AUTO_CREATE)
}
```

**B5. 编译运行 + 观测预期**
```bash
./gradlew installDebug
adb shell am start -n com.example.binderdemo/.MainActivity
adb logcat -s BinderDemo:I
```
预期 logcat:
```
I BinderDemo: client pid=12345  service pid=12389
```
**两个 pid 不同 = 你刚刚验证了一次真实的跨进程 Binder 调用。** 如果你把 `android:process=":remote"` 删掉(同进程),会看到两个 pid **相同**,且 `asInterface` 返回的是本地对象(不走内核,直接方法调用)—— 这正好演示了 Binder 的"同进程优化:不进内核"。

### Demo C(把 demo 和源码焊死):证明"身份是内核盖的章"

在 B 的 service 方法里加一行,读 **calling uid**:
```kotlin
override fun getPid(): Int {
    val callerUid = android.os.Binder.getCallingUid()   // 读 §3.4 的 mCallingUid
    val callerPid = android.os.Binder.getCallingPid()
    android.util.Log.i("BinderDemo", "called by uid=$callerUid pid=$callerPid")
    return Process.myPid()
}
```
预期 logcat(service 侧):
```
I BinderDemo: called by uid=10234 pid=12345
```
这个 `uid` 不是 client 传进来的参数,而是 **§2.2 里 client 填 0、§3.4 里内核盖章、service 读 `mCallingUid`** 这条链路的产物。**把 client 代码改成试图伪造 uid 是做不到的** —— 你根本没有那个入口,身份在内核里被覆盖。这就是 `checkCallingPermission` 不可伪造的根。

### Demo D(系统级观测,推荐):用 atrace / Perfetto 抓 binder transaction

Android 的 binder 驱动埋了 ftrace tracepoint(`binder_transaction`、`binder_transaction_received` 等),`atrace` 的 `binder_driver` 类别就是它。

**D1. 命令行 atrace 抓 10 秒 binder 活动**
```bash
adb shell atrace -t 10 -b 16000 binder_driver sched > /tmp/binder_trace.html
# 或直接看文本:
adb shell atrace -t 5 binder_driver | head -50
```
预期文本里能看到成对的 tracepoint:
```
binder_transaction:          transaction=123456 dest_node=... dest_proc=12389 dest_thread=...
binder_transaction_received: transaction=123456
```
`transaction=` 后那个号能把 **发起(transaction)** 和 **目标线程收到(transaction_received)** 配对,直接对应 §2.2 `BC_TRANSACTION` → §3.4 `BR_TRANSACTION`。

**D2. Perfetto(更直观,有 UI)**
```bash
adb shell perfetto -o /data/misc/perfetto-traces/trace.pftrace -t 10s \
  -b 32mb sched freq binder
adb pull /data/misc/perfetto-traces/trace.pftrace .
# 把 trace.pftrace 拖进 https://ui.perfetto.dev
```
预期:在 Perfetto UI 里每个进程的 track 上能看到 `binder transaction` slice,点开看 dest process / 是否 oneway / 耗时。**这是定位"哪次 binder 调用慢、阻塞了谁"的生产级手段** —— 比如你能直接看到主线程一个 slice 卡了 200ms 等 AMS 回包。

> D 节命令在标准 emulator(Android 10+)即可跑,`perfetto` 二进制系统自带。`atrace` 输出文件名为 `.html` 但其实是 systrace 格式,直接 `head` 看文本也行。

---

## 六、方案对比:Binder vs 传统 IPC,以及 Binder 内部的同步/oneway

### 6.1 Binder vs 传统 Linux IPC

| 维度 | Binder | UNIX domain socket | SysV shared memory | pipe / FIFO |
|---|---|---|---|---|
| 拷贝次数 | **1** | 2 | 0(但要自己同步) | 2 |
| 对象 / RPC 语义 | **原生**(传 binder 对象、方法分发) | 无,自己造 | 无 | 无 |
| 跨进程引用计数 + 死亡通知 | **内核托管** | 无(只能 read 返回 0 探知断开) | 无 | 无(SIGPIPE) |
| 可信对端身份 | **每次 transaction 盖 pid/euid** | `SO_PEERCRED`(连接级,非调用级) | 无 | 无 |
| 线程模型 | **内核调度的线程池** | 自己写 epoll/线程 | 自己写 | 自己写 |
| 单次/总量大小 | 进程接收区 **~1MB 共享** | socket buffer 可调,无硬 1MB | 受 SHMMAX,可很大 | pipe buffer 64KB 量级 |
| 适用 | **系统服务 RPC、跨进程对象** | 通用流式、Android 内部(如 logd、netd 部分) | **大块数据零拷贝**(如 ashmem/共享 buffer) | 简单父子进程管道 |

**选型边界(什么时候 Binder 不合适)**:
- **传大块连续数据(图像帧、音频缓冲、几 MB 的文件)**:别塞进 Binder transaction(会 `TransactionTooLargeException`)。正确姿势是 **用 Binder 传一个 ashmem / `MemoryFile` 的文件描述符(`ParcelFileDescriptor`)**,大数据走共享内存,Binder 只传"门把手"。这正是 §3.4 里 `TF_ACCEPT_FDS` 和 offsets 里 `BINDER_TYPE_FD` 的用途。
- **高吞吐流式**(持续推送大量小包):Binder 每次都进内核 + 线程池调度,开销不低;纯数据流用 socket/pipe 更省。
- **同机大数据共享**:shared memory 零拷贝完胜,Binder 只用来协调/传 fd。

### 6.2 Binder 内部:同步 transaction vs oneway

| | 同步(默认) | oneway(`oneway` 关键字 / `FLAG_ONEWAY`) |
|---|---|---|
| 调用线程 | **阻塞** 到对端执行完返回 `BR_REPLY` | 收到 `BR_TRANSACTION_COMPLETE` 即返回,不等执行 |
| 接收区限额 | 可用整块(~1MB) | **只能用异步限额(约一半)**,`free_async_space` 见 Demo A3 |
| 顺序保证 | —— | 同一对象的连续 oneway 调用 **按序到达**(官方文档明确) |
| 典型用途 | getter、需要返回值/需要确认完成的调用 | 通知、回调、不关心结果的 fire-and-forget |
| 坑 | 主线程发同步调用 = 可能 ANR | oneway 风暴打爆异步限额 → 后续 oneway `FAILED`;且 **callback 接口默认不是 oneway,容易反向阻塞 server** |

> 实战关键:**给 AIDL 的 callback 接口加 `oneway`**。否则 server 回调 client 时走 **同步** transaction,会阻塞 server 的 binder 线程等 client 处理完 callback —— 一个慢 client 能拖垮整个 server 的线程池(经典生产事故,见 §7)。

---

## 七、扎根:失败模式 / 生产真坑 / 根因

### 坑 1:`TransactionTooLargeException` —— "我单次明明没到 1MB 啊"
- **现象**:偶发 crash,栈里 `TransactionTooLargeException`,数据量看起来远不到 1MB。
- **根因**:那 1MB 是 **进程级、所有并发 in-flight transaction 共享** 的接收区(§4),不是单次上限;oneway 还只能用一半。高并发时多个事务叠加挤爆,或某次塞了大 `Bundle`/大 `List`/`Bitmap`。
- **修**:① 大数据走 `ParcelFileDescriptor` + ashmem,Binder 只传 fd;② 分页拉取替代一次性返回大集合;③ 别往 `Intent`/`Bundle` 塞 `Bitmap`、大 `Parcelable[]`;④ 用 §5 Demo A3 的 `free_async_space` 观测异步剩余量。

### 坑 2:binder 线程池耗尽 → 系统级"假死"
- **现象**:某热门服务(AMS/PMS/WMS)响应骤慢,大量进程调它都卡,看着像系统冻住。
- **根因**:目标进程 31 条(或 15 条)binder 线程 **全被慢调用占满**(比如某个 onTransact 里做了 IO / 锁等待 / 死循环),新请求只能在驱动 `todo` 队列排队。§3.4 的线程池有上限,满了不会无限扩。
- **诊断**:`/sys/kernel/debug/binder/proc/<pid>` 看 `ready_threads`(空闲数,长期 0 = 满载);Perfetto(Demo D2)看该进程 binder track 是否全满、哪个 transaction 长期占用。
- **修根因**:onTransact 里 **不做阻塞 IO / 不持长锁 / 不等其它进程**;真要慢活,转交业务线程异步处理后 oneway 回调。

### 坑 3:callback 没加 `oneway`,被慢 client 反向拖垮 server
- **现象**:server 端 binder 线程被占满,但"凶手"在 client 进程。
- **根因**:server 通过 client 注册的 callback 接口回调,callback 是 **同步** transaction(§6.2),server 的 binder 线程要 **阻塞等 client 执行完 callback**。某个 client 的 callback 实现很慢(或在它自己主线程卡住),就把 server 的线程一条条占死。
- **修**:callback AIDL 接口声明 `oneway`;server 回调即返回,不等 client。

### 坑 4:忘记 `linkToDeath` / 没处理 `DeadObjectException`
- **现象**:对端进程(常被 LMK 杀)死后,client 继续持旧 `BinderProxy` 调用,抛 `DeadObjectException`,或持有已死服务的引用导致逻辑错乱。
- **根因**:§3.3 的 death notification(`BR_DEAD_BINDER`)你没订阅。Binder 给了你"对端死了主动通知"的能力(这是它优于 socket 的设计动机之一),但你得 `linkToDeath` 注册。
- **修**:`binder.linkToDeath(recipient, 0)`,在 `binderDied()` 里清理引用、重连;所有远程调用 try-catch `RemoteException`/`DeadObjectException`。

### 坑 5:`getCallingUid()` 在错误的地方读到了自己的 uid
- **现象**:权限检查"莫名其妙"通过/失败;`getCallingUid()` 返回的是自己进程 uid 而非调用方。
- **根因**:§3.4 里 `mCallingPid/Uid` 是 **per-thread 且只在 onTransact 执行期间有效**。如果你在 onTransact 里 `clearCallingIdentity()` 之后、或异步抛到别的线程后再读,拿到的就不是原始 caller。`clearCallingIdentity()` 会把身份临时切成自己(用于"我代表系统去访问受保护资源"),`restoreCallingIdentity(token)` 还原。
- **修**:**在 onTransact 同步路径、`clearCallingIdentity` 之前** 读 `getCallingUid()`;需要降权访问时 `clear` 前先存好 caller uid。

---

## 章末·五件套

### 1. 一句话本质
Binder 是 **内核托管"跨进程对象引用计数 + 可信身份 + 一次拷贝 RPC"** 的框架:一次同步调用 = `transact` 打包 `binder_transaction_data` → `ioctl(BINDER_WRITE_READ)` 把 `BC_TRANSACTION` 送进内核 → 内核 `binder_alloc_copy_user_to_buffer` 单拷贝进目标进程 mmap 接收区、翻译对象句柄、唤醒目标 binder 线程 → `onTransact` 分发执行 → `BR_REPLY` 原路返回。

### 2. 关键源码地图(出处)
| 关注点 | 文件@符号 | 出处标注 |
|---|---|---|
| client 发起/打包/收包 | `IPCThreadState.cpp` :: `transact` / `writeTransactionData` / `talkWithDriver` / `waitForResponse` | 【真实源码 cozybit/aosp-frameworks-base@libs/binder/】(老 AOSP) |
| server 线程池/分发 | `IPCThreadState.cpp` :: `joinThreadPool` / `executeCommand`(BR_TRANSACTION) | 同上 |
| 接收区 mmap / 线程上限 | `ProcessState.cpp` :: `BINDER_VM_SIZE` / `open_driver` / 构造函数 | 同上 |
| 单拷贝 / 对象翻译 / 锁 | `binder.c` :: `binder_transaction` / `binder_alloc_copy_user_to_buffer` / 顶部 Locking 注释 | 【真实源码 torvalds/linux@drivers/android/binder.c】(v6.6) |
| Java 接收入口 | `Binder.java` :: `execTransact` / `onTransact` | 【真实源码 aosp-mirror/platform_frameworks_base@core/java/android/os/】 |
| AIDL 用法 / scatter-gather / 三域 | developer.android.com AIDL、source.android.com binder-ipc | 【真实文档】 |

### 3. 三个数字记死
- **接收区 ≈ 1MB - 8KB**(`BINDER_VM_SIZE = (1*1024*1024)-(4096*2)`),进程级共享,oneway 只用一半。
- **默认 binder 线程上限 15**(`DEFAULT_MAX_BINDER_THREADS`),system_server 抬到 **31**。
- **拷贝 1 次**(scatter-gather 后从 3 降到 1)。

### 4. 动手验证清单(都能在 emulator 跑)
- [ ] `adb shell service list` 看 handle 0(ServiceManager)挂的所有 binder 服务
- [ ] `adb shell service call SurfaceFlinger 1000` 发一次真实 transaction
- [ ] 写 Demo B 的 `:remote` 进程 AIDL service,确认 client/service **pid 不同**;删掉 `:remote` 确认同进程 pid 相同(本地优化不进内核)
- [ ] Demo C 用 `getCallingUid()` 验证"身份是内核盖的章,不可伪造"
- [ ] Demo D `atrace binder_driver` / Perfetto 抓 `binder_transaction` 成对 tracepoint

### 5. 最容易讲错/记错的点(自检)
- "一次拷贝"≠ 完全零拷贝:**内核里有且仅有一次 `copy_from_user`**;sender 写出 + receiver 读入合计一次,对比 socket 的两次。
- 1MB **不是单次上限**,是进程级共享接收区;`TransactionTooLargeException` 常因并发叠加或 oneway 半区爆,而非单次刚好超 1MB。
- `sender_pid/euid` **用户态填 0**,内核盖真实值 —— 权限模型的安全根在此,不在用户态。
- oneway **也会 ioctl 进内核** 收 `BR_TRANSACTION_COMPLETE`,只是不等 `BR_REPLY`;callback 接口忘加 `oneway` 会反向阻塞 server。
- server **没有 accept 死循环**,是内核驱动调度 binder 线程池;线程池满了系统会"假死",不会自动无限扩容。

---

## 取材记录(本章实际 WebFetch / WebSearch 的 URL)

> 标注原则:逐字引用标【真实源码 repo@path】;机制性二手交叉核对标【二手】;未取到一手标「待核」。

**逐字源码(WebFetch 成功取到并核对)**
- `cozybit/aosp-frameworks-base@libs/binder/IPCThreadState.cpp`
  `https://raw.githubusercontent.com/cozybit/aosp-frameworks-base/master/libs/binder/IPCThreadState.cpp`
  —— transact / writeTransactionData / talkWithDriver / waitForResponse / joinThreadPool / executeCommand(BR_TRANSACTION)。**这是较老 AOSP 快照(约 Android 4.x);现代代码经 scatter-gather/RPC/fine-grained locking 重构,函数更长,但 transaction 主链路语义一致。** 选老版是为了逐行读骨架。
- `cozybit/aosp-frameworks-base@libs/binder/ProcessState.cpp`
  `https://raw.githubusercontent.com/cozybit/aosp-frameworks-base/master/libs/binder/ProcessState.cpp`
  —— BINDER_VM_SIZE / open_driver / 构造函数 mmap。
- `torvalds/linux@drivers/android/binder.c`(顶部 Locking 注释 + binder_transaction 分配/拷贝)
  `https://raw.githubusercontent.com/torvalds/linux/master/drivers/android/binder.c`
  `https://raw.githubusercontent.com/torvalds/linux/v6.6/drivers/android/binder.c`(取到 binder_alloc_copy_user_to_buffer 调用点)
- `aosp-mirror/platform_frameworks_base@core/java/android/os/Binder.java`(execTransact / onTransact)
  `https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/java/android/os/Binder.java`

**官方文档(WebFetch 成功)**
- AIDL 用法:`https://developer.android.com/develop/background-work/services/aidl`
- AIDL overview(AOSP):`https://source.android.com/docs/core/architecture/aidl/overview`
- binder-ipc(三域 + scatter-gather "3→1"):`https://source.android.com/docs/core/architecture/hidl/binder-ipc`

**二手 / 交叉核对(WebSearch + WebFetch)**
- 设计血统(BeOS/OpenBinder/Hackborn):WebSearch 命中多篇一致复述(LinkedIn 技术长文 / Vanderbilt CS282 课件 PDF / Marko Gargenta 2013 ABS "Deep Dive into Android IPC/Binder")——**一手 OpenBinder 文档与早期 AOSP commit 未直接取到,关键时间点标「待核」**。
- 对象翻译字段名 `flat_binder_object` / `fp->hdr.type` / `BINDER_TYPE_BINDER↔HANDLE`:synacktiv "Binder transactions in the bowels of the Linux Kernel" `https://www.synacktiv.com/en/publications/binder-transactions-in-the-bowels-of-the-linux-kernel`,与内核 `binder.h` 一致。
- DEFAULT_MAX_BINDER_THREADS=15 / system_server=31:WebSearch 命中 frameworks/native ProcessState 相关讨论与 diff。

**WebFetch 受限未取到逐字(标「待核」)**
- `developer.android.com/reference/android/os/TransactionTooLargeException` 的逐字 class 描述(页面渲染为导航壳,正文未取到)——但其"约 1MB"与本章 `BINDER_VM_SIZE` 源码常量一致,故结论可信、逐字标「待核」。
- Bootlin elixir 的 binder.c 逐行(返回的是版本索引页而非源码;改用 raw.githubusercontent torvalds/linux 取到等价内容)。
