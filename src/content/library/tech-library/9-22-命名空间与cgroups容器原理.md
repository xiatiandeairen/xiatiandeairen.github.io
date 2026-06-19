---
title: "命名空间与 cgroups（容器原理）· 深化版"
slug: "9-22"
collection: "tech-library"
group: "linux系统"
order: 9022
summary: "一句话定位：容器不是轻量虚拟机，它就是内核里两组正交的开关——namespace 决定一个进程\"能看到哪些内核对象\"，cgroup 决定它\"能用多少资源\"，再加上 rootfs（pivot_root + overlay）与 capabilities/seccomp/LSM，组合出所有容器。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T09:42:37.000Z"
updatedAt: "2026-06-12T09:42:37.000Z"
---
> **一句话定位**：容器不是轻量虚拟机，它就是内核里两组正交的开关——namespace 决定一个进程"能看到哪些内核对象"，cgroup 决定它"能用多少资源"，再加上 rootfs（pivot_root + overlay）与 capabilities/seccomp/LSM，组合出所有容器。本章不停在"它是什么"，而是用 **v6.12 真实源码 + LWN/lore 设计争论 + 生产排障根因** 把这两组开关拆到底。

---

## TL;DR（本章最硬结论）

1. **容器 = namespace + cgroup + rootfs + capabilities/seccomp/LSM**。Docker / containerd / podman / crun 都是这几样的组合封装。源码层面没有"容器"这个对象——内核里根本搜不到 `struct container`，只有 `struct nsproxy`、`struct css_set`、`struct cgroup`。
2. **namespace 隔离视图，cgroup 限制配额，两者正交**。但它们在内核结构上的归属方式完全不同：6 种 namespace 通过 `task_struct->nsproxy` 共享，user namespace 走 `task_struct->cred->user_ns`（凭证体系），cgroup 走 `task_struct->cgroups`（一个 `css_set` 指针）。这三条挂载路径的差异不是历史偶然，是语义决定的（§22.3.1、§22.4.2）。
3. **cgroup v1→v2 不是重构，是一场持续 5 年的设计战争**。核心冲突是 Tejun Heo（cgroup maintainer）坚持的"进程粒度 + no-internal-process"与 Peter Zijlstra/Paul Turner（调度器）坚持的"调度器调度的是线程"之间的根本张力，最终靠 **threaded cgroup**（4.14/4.15）妥协收场。这段历史是理解 v2 一切怪规则的钥匙（§22.4.4，全程有 LWN/lore 出处）。
4. **CFS bandwidth controller 不是在调度器外面包一层，它就长在 CFS 热路径里**。`__account_cfs_rq_runtime()` 在每次 `update_curr` 时扣 `runtime_remaining`，扣穿就 `resched_curr` → `throttle_cfs_rq()` 把整个 cfs_rq 从树上摘下来。slice 默认 **5ms**（`sysctl_sched_cfs_bandwidth_slice = 5000UL` 微秒），这是源码里的常数（§22.8.1，逐行注解真实源码）。
5. **user namespace 是 rootless 容器的基石，也是近十年容器逃逸 CVE 的最大单一入口**。"非特权可创建 user ns" = 把一大票原本要 root 的内核代码路径暴露给普通用户，所以各发行版近年纷纷给它加 sysctl/AppArmor 旋钮收口（§22.6）。
6. **共享内核 = 共享攻击面**。Dirty Pipe（CVE-2022-0847）只需读权限就能改 page cache，read-only 挂载、跨容器统统穿透——这类 bug 把"namespace 是隔离不是沙箱"这句话钉死，也正是 microVM（Kata/Firecracker）存在的理由（§22.1.2、§22.5.3）。

---

## 前置依赖章

- **第 4 章**：进程与线程（`task_struct`、`cred`、fork/clone 路径）——本章大量引用 `task_struct` 的子字段。
- **第 5 章**：调度器（CFS/EEVDF、task group、`cfs_rq`）——CPU controller 直接嵌在这里。
- **第 10 章**：Page Cache 与内存回收（lruvec、`shrink_node`、memcg charge）——memory controller 复用这套机制。
- 选读 **第 17 章**（io_uring）、**第 19 章**（VFS/overlayfs）——rootfs 与 IO controller 涉及。

---

## 22.1 背景：为什么需要隔离与限制？

### 22.1.1 问题溯源：一个内核、多个互不信任的租户

Unix 最初的抽象只有"进程"：独立地址空间，但**共享整个内核命名空间**——同一个 `/proc`、同一个网络栈、同一个 PID 空间、同一套 UID。单租户服务器无所谓；多租户 / CI / FaaS 一上来三个矛盾立刻爆炸：

```
矛盾一：隔离性（视图）
  进程 A 能 ls /proc 看到所有 PID    → 信息泄露 / 误杀
  进程 A 能 bind 0.0.0.0:80          → 端口冲突
  进程 A 能看到宿主 hostname / 挂载   → 环境耦合

矛盾二：资源公平性（配额）
  进程 A 起 1000 线程吃满 CPU        → 进程 B 饿死
  进程 A fork bomb / malloc 狂飙      → 全机 OOM

矛盾三：环境一致性（文件系统）
  A 依赖 glibc 2.17，B 需要 2.35     → 同一 rootfs 无法共存
```

矛盾一、三是"看到什么"的问题 → **namespace**。矛盾二是"用多少"的问题 → **cgroup**。这条对应关系是本章的骨架。

**VM 是一个解**：每个租户一个完整 Guest OS + Hypervisor。代价是固定开销——独立内核常驻内存（数百 MB）、启动秒级、镜像 GB 级。**内核的解是在同一个内核里制造"视图隔离 + 配额控制"**，省掉那块固定开销。省得了多少、安全性差多少，是后面所有 trade-off 的源头。

### 22.1.2 设计哲学差异（含 microVM 第三条路）

| 维度 | 传统 VM | Container | microVM (Firecracker/Kata) |
|---|---|---|---|
| 隔离单元 | 整个 OS | 进程组（ns+cgroup） | 整个 OS（裁剪 guest） |
| 内核数量 | 2（host+guest） | **1（共享）** | 2（host + 极简 guest） |
| 启动时间 | 秒~数十秒 | ~10ms~数百 ms | ~100ms 级（Firecracker 宣称 <125ms） |
| 额外内存 | 数百 MB 级 | ~MB 级（无独立内核固定开销） | ~数 MB~数十 MB（裁剪 guest） |
| 隔离强度 | 强（独立内核+hypervisor） | **弱（共享内核，同 CVE 穿透）** | 强（独立内核，攻击面比传统 VM 还小） |
| 进程可见性 | 完全隔离 | 依赖 PID ns | 完全隔离 |

> 数字都是**量级**，强依赖平台/镜像/配置，别当精确值背。"额外内存 ~MB 级"是说容器**没有独立内核这块固定开销**，不是说一个容器只占 1MB——它实际占多少由你跑的进程决定。

**关键判断：容器的安全边界是宿主内核本身，不是 hypervisor。** 一个内核 LPE 漏洞对容器内进程和宿主同样致命。**这不是抽象担忧——Dirty Pipe（CVE-2022-0847，§22.5.3 详解）只需对一个文件有读权限，就能越过 read-only 挂载改写 page cache，进而改任何容器或宿主的文件**。namespace 把视图挡住了，但底层 page cache 是一份，内核代码是一份。这正是 microVM 这一列存在的全部理由：用一个裁剪到极致的 guest kernel + KVM，换回"独立内核"这道硬边界，同时把启动开销压到接近容器。AWS Lambda（Firecracker）、阿里/字节的不可信多租户负载都走这条路。面试里能把这三列的 trade-off 说清，比只会背"容器共享内核"高一档。

---

## 22.2 演进史：从 chroot 到现代容器（带出处）

| 时间 | 机制 | 意义与考据 |
|---|---|---|
| 1979 | `chroot`（Unix V7；1982 入 4.2BSD） | 最原始的"文件系统根隔离"。**chroot 从来不是安全边界**：有 `CAP_SYS_CHROOT` 或一个残留的目录 fd 就能 double-chroot 逃出。 |
| 2000 | FreeBSD Jail | 第一个把 chroot + 网络/进程隔离捏成"容器"概念的系统。Linux 没直接跟进。 |
| 2002 | mount namespace（2.4.19） | Linux 第一个 namespace。flag 至今叫 `CLONE_NEWNS`——历史包袱，没带 "MNT"，因为当时还没料到会有第二种 namespace。 |
| 2006-06 | nsproxy 框架落地 | 见下方 `kernel/nsproxy.c` 文件头：**作者 Serge Hallyn（IBM）+ Pavel Emelianov（OpenVZ/SWsoft）**，"Jun 2006 - namespaces support"。OpenVZ 这个商业容器方案是把 namespace 推进主线的关键推手。 |
| 2006 | Google 启动 "process containers" | cgroup 前身，**Paul Menage / Rohit Seth** 发起（Wikipedia/cgroups 词条）。 |
| 2008-01 | cgroup v1 合入（2.6.24） | 此时已更名 control groups——故意避开 "container" 一词的歧义（"container" 在内核语境太满）。 |
| 2006-2013 | uts/ipc/pid/net/user ns 陆续合入 | uts/ipc≈2.6.19；pid≈2.6.24；net≈2.6.29；**user ns 到 3.8（2013-02）才算完整可用**（Kerrisk, LWN/532593）。 |
| 2013 | Docker 发布 | 把上述特性 + 镜像分层 + 友好 CLI 封装，引爆容器时代。Docker 没发明任何内核机制。 |
| 2014-11 | cgroup namespace 提案（Jake Edge, LWN/621006） | 解决"容器能透过 `/proc/self/cgroup` 看到宿主完整 cgroup 路径"。Eric Biederman："definitely looks like the right direction… something I had been asking for since cgroups were merged"。`CLONE_NEWCGROUP` 4.6 落地。 |
| 2016-03 | cgroup v2 official（4.5） | 统一层级。Tejun Heo 维护 v1/v2 双版本。 |
| 2016-08 | Tejun 发出 "State of CPU controller in cgroup v2"（LWN/697369） | CPU controller **还进不了 v2**——与调度器阵营僵持。这封文档是理解整场争论的核心史料（§22.4.4）。 |
| 2017-07/09 | threaded cgroup 排队进 4.14/4.15 | Tejun 的 lore patchset 标题就是 `[PATCHSET for-4.14]` 与 `[PATCHSET REPOST for-4.15] cgroup, sched: cgroup2 interface for CPU controller`。Corbet："A milestone for control groups"（LWN/729215，2017-07-31）。 |
| 2019+ | systemd 默认 cgroup v2 | systemd 243 / Fedora 31 先行，主流发行版 2021-2022 跟进（Debian 11、Ubuntu 22.04、RHEL 9 默认 unified v2）。 |

**关键教训（这是"演进式系统"的代价）**：6/7 种 namespace 各自独立进入内核、跨越十余年、由不同人主导（Hallyn、Emelianov、Biederman、Kerrisk 测试…），**没有一个统一设计**。直接后果是某些组合的语义至今别扭——user ns 与其他 ns 的创建顺序约束、`pid_ns_for_children` 这种"为子进程而非自己"的字段（§22.3.1）、user ns 单独挂在 cred 而非 nsproxy。cgroup 则相反：v1 是"先实现后设计"（Tejun 原话 "design followed implementation"，LWN/679786），v2 是一次"推倒重来 + 强行统一"，代价是 5 年社区拉锯。两条线一对照，正好是"自底向上演进"vs"自顶向下重构"的活教材。

---

## 22.3 Namespace：隔离"看到什么"

### 22.3.1 内核数据结构：nsproxy 与三条挂载路径

每个 `task_struct` 持有一个 `struct nsproxy *nsproxy`。下面是 **v6.12 逐字源码**：

【真实源码 v6.12 · `include/linux/nsproxy.h`】
```c
struct nsproxy {
	refcount_t count;
	struct uts_namespace *uts_ns;
	struct ipc_namespace *ipc_ns;
	struct mnt_namespace *mnt_ns;
	struct pid_namespace *pid_ns_for_children;
	struct net 	     *net_ns;
	struct time_namespace *time_ns;
	struct time_namespace *time_ns_for_children;
	struct cgroup_namespace *cgroup_ns;
};
extern struct nsproxy init_nsproxy;
```
头文件注释（逐字）：*"the nsproxy is shared by tasks which share all namespaces. As soon as a single namespace is cloned or unshared, the nsproxy is copied."* ——这一句把 nsproxy 的生命周期模型讲完了：**copy-on-write 式共享**。同一个进程组只要 namespace 全相同，就共用一个 nsproxy（`count` 引用计数）；任何一个 namespace 被 clone/unshare，整个 nsproxy 被复制一份。

**三条挂载路径（这是 namespace 理解到位的分水岭）：**

1. **6 种走 nsproxy**：uts/ipc/mnt/net/time/cgroup。共享/复制由上面的 CoW 规则统一管理。
2. **PID ns 走 `task->pid` 反查，nsproxy 里只存 `pid_ns_for_children`**（见下方高频坑）。
3. **user ns 完全不在 nsproxy 里，挂在 `task->cred->user_ns`**——因为 user ns 决定的是**权限与身份**，归属凭证（credential）体系，不是"命名空间代理"。这也是 7 种 ns 里 user ns 结构上最特殊的根因。

**💡高频面试坑 —— `pid_ns_for_children` 为什么是"for_children"？**
进程**自己所在**的 PID ns 不在 nsproxy 里，而是通过它的 `struct pid` 反查。nsproxy 里存的是"它 fork 出来的子进程将进入哪个 PID ns"。根因：一个进程一旦诞生，它在某个 PID ns 里的 PID 是**终生不可变**的——你不能把一个活着的进程"搬"进另一个 PID ns。所以 `setns(fd, CLONE_NEWPID)` **只影响调用者之后 fork 的子进程，对调用者自己不生效**。time ns 同理有 `time_ns_for_children`。能讲清这个"for_children"的设计动机，就证明你理解了"PID 的 immutability"这个底层约束。

### 22.3.2 namespace 是怎么被批量创建的：create_new_namespaces() 逐行精读

`clone()` / `unshare()` / `setns()` 最终都会走到 `create_new_namespaces()`，它是 namespace 复制的总闸。下面是 **v6.12 逐字源码**（含 2006 年的文件头，本身就是史料）：

【真实源码 v6.12 · `kernel/nsproxy.c`】
```c
// SPDX-License-Identifier: GPL-2.0-only
/*
 *  Copyright (C) 2006 IBM Corporation
 *
 *  Author: Serge Hallyn <serue@us.ibm.com>
 *
 *  Jun 2006 - namespaces support
 *             OpenVZ, SWsoft Inc.
 *             Pavel Emelianov <xemul@openvz.org>
 */

/*
 * Create new nsproxy and all of its the associated namespaces.
 * Return the newly created nsproxy.  Do not attach this to the task,
 * leave it to the caller to do proper locking and attach it to task.
 */
static struct nsproxy *create_new_namespaces(unsigned long flags,
	struct task_struct *tsk, struct user_namespace *user_ns,
	struct fs_struct *new_fs)
{
	struct nsproxy *new_nsp;
	int err;

	new_nsp = create_nsproxy();
	if (!new_nsp)
		return ERR_PTR(-ENOMEM);

	new_nsp->mnt_ns = copy_mnt_ns(flags, tsk->nsproxy->mnt_ns, user_ns, new_fs);
	if (IS_ERR(new_nsp->mnt_ns)) {
		err = PTR_ERR(new_nsp->mnt_ns);
		goto out_ns;
	}

	new_nsp->uts_ns = copy_utsname(flags, user_ns, tsk->nsproxy->uts_ns);
	if (IS_ERR(new_nsp->uts_ns)) {
		err = PTR_ERR(new_nsp->uts_ns);
		goto out_uts;
	}

	new_nsp->ipc_ns = copy_ipcs(flags, user_ns, tsk->nsproxy->ipc_ns);
	if (IS_ERR(new_nsp->ipc_ns)) {
		err = PTR_ERR(new_nsp->ipc_ns);
		goto out_ipc;
	}

	new_nsp->pid_ns_for_children =
		copy_pid_ns(flags, user_ns, tsk->nsproxy->pid_ns_for_children);
	if (IS_ERR(new_nsp->pid_ns_for_children)) {
		err = PTR_ERR(new_nsp->pid_ns_for_children);
		goto out_pid;
	}

	new_nsp->cgroup_ns = copy_cgroup_ns(flags, user_ns,
					    tsk->nsproxy->cgroup_ns);
	if (IS_ERR(new_nsp->cgroup_ns)) {
		err = PTR_ERR(new_nsp->cgroup_ns);
		goto out_cgroup;
	}

	new_nsp->net_ns = copy_net_ns(flags, user_ns, tsk->nsproxy->net_ns);
	if (IS_ERR(new_nsp->net_ns)) {
		err = PTR_ERR(new_nsp->net_ns);
		goto out_net;
	}

	new_nsp->time_ns_for_children = copy_time_ns(flags, user_ns,
					tsk->nsproxy->time_ns_for_children);
	if (IS_ERR(new_nsp->time_ns_for_children)) {
		err = PTR_ERR(new_nsp->time_ns_for_children);
		goto out_time;
	}
	new_nsp->time_ns = get_time_ns(tsk->nsproxy->time_ns);

	return new_nsp;

out_time:
	put_net(new_nsp->net_ns);
out_net:
	put_cgroup_ns(new_nsp->cgroup_ns);
out_cgroup:
	if (new_nsp->pid_ns_for_children)
		put_pid_ns(new_nsp->pid_ns_for_children);
out_pid:
	if (new_nsp->ipc_ns)
		put_ipc_ns(new_nsp->ipc_ns);
out_ipc:
	if (new_nsp->uts_ns)
		put_uts_ns(new_nsp->uts_ns);
out_uts:
	if (new_nsp->mnt_ns)
		put_mnt_ns(new_nsp->mnt_ns);
out_ns:
	kmem_cache_free(nsproxy_cachep, new_nsp);
	return ERR_PTR(err);
}
```

**逐行/逐段注解：**
- **`user_ns` 是第一个参数，且每个 `copy_*_ns` 都把它传进去**：这是 user namespace "是其他 namespace 的权限上下文"的源码证据。在新 user ns 里，`copy_pid_ns`/`copy_net_ns` 之类的权限检查（`ns_capable`）是相对**这个新 user ns** 判定的——这正是"非特权用户先建 user ns，再在里面建别的 ns"得以成立的内核机制（§22.6）。
- **复制顺序 mnt → uts → ipc → pid → cgroup → net → time**：不是随意排的。net ns 放在很后面，因为创建网络 namespace 最重（要初始化 loopback、网络命名空间的各种子系统），失败概率/成本最高，放后面让前面便宜的先做；一旦它失败，前面的回滚代价是确定的。
- **教科书级的 goto 错误处理阶梯（`out_time → out_net → … → out_ns`）**：每个 label 精确地只回滚"已经成功分配的那些"。注意 `out_time:` 第一句是 `put_net(...)`——因为 time 失败时 net 已经成功了，必须放掉。这是 Linux 内核 "goto ladder cleanup" 范式的标准写法，**面试问"内核怎么做错误清理"直接背这段**。
- **`time_ns` 与 `time_ns_for_children` 分开处理**：`time_ns`（自己用的）直接 `get_time_ns`（增引用，不复制），`time_ns_for_children`（给子进程的）才 `copy_time_ns`。又一次印证 §22.3.1 的"自己 vs 子进程"二分。
- **`create_new_namespaces` 不挂到 task 上**（文件头注释明说 "Do not attach this to the task, leave it to the caller"）：分配与挂载分离，挂载由 `switch_task_namespaces()` 在持锁下原子完成。这是为了让 setns/unshare 这些可能失败的路径"先把新东西准备好、最后一步才切换"，避免中途失败留下半挂状态。

### 22.3.3 七种（+time=八种）Namespace 对照表

```
Namespace   Flag（clone/unshare）  隔离内容                       挂载路径         首个可用版本
──────────────────────────────────────────────────────────────────────────────────────
MNT         CLONE_NEWNS            mount point 视图                nsproxy          2.4.19
UTS         CLONE_NEWUTS           hostname, domainname            nsproxy          2.6.19
IPC         CLONE_NEWIPC           System V IPC, POSIX MQ          nsproxy          2.6.19
PID         CLONE_NEWPID           PID 空间（容器内 PID=1）          nsproxy(*)       2.6.24
NET         CLONE_NEWNET           网络设备/路由/iptables/socket     nsproxy          2.6.29
USER        CLONE_NEWUSER          UID/GID 映射                    cred->user_ns    3.8（可用）
CGROUP      CLONE_NEWCGROUP        cgroup 根视图                   nsproxy          4.6
TIME        CLONE_NEWTIME          CLOCK_MONOTONIC/BOOTTIME 偏移    nsproxy          5.6
(*) PID ns：nsproxy 里存的是 pid_ns_for_children，自己所在的 PID ns 由 task->pid 反查
```

### 22.3.4 PID Namespace 深潜：PID=1 的特殊性与信号黑洞

```
宿主机 PID 空间                          容器内 PID 空间（独立）
  PID 1   (systemd)                        PID 1  (bash)  ← 容器内看到自己是 1
  PID 8847 (containerd-shim)
    └── PID 8900 (bash)  ← 同一进程，宿主侧 PID=8900
```

一个进程**同时拥有多个 PID**（每层 PID ns 一个），可在 `/proc/<host_pid>/status` 的 `NSpid:` 行看到全部：
```
NSpid:  8900   1     # 第一列宿主 PID，第二列该进程所在 PID ns 内的 PID
```

**为什么 PID=1 在容器里有"信号黑洞"特性？** 内核对 PID-ns 的 init 进程有特殊保护：**对它发送没有自定义 handler 的信号会被静默丢弃**（和宿主 PID 1 的保护同源，防止意外杀死 init 导致整个 namespace 崩塌）。直接后果：
- 容器里 `kill -9 1` 永远无效，要从宿主 `kill -9 8900`（真实宿主 PID）。
- **`docker stop` 优雅退出的著名坑**：`CMD ["sh","-c","myapp"]` 让 PID 1 是 sh，myapp 是 PID 2。`docker stop` 发 `SIGTERM` 给 PID 1（sh），sh **默认不转发**给子进程 → myapp 收不到 → 10 秒超时后 `SIGKILL` 硬杀 → 数据没 flush。修法：用 exec 形式 `CMD ["myapp"]` 让 myapp 直接当 PID 1，或塞一个 `tini`/`dumb-init` 当 PID 1 专门 reap 僵尸 + 转发信号。
- **僵尸进程收割**：PID ns 内的孤儿进程会被 reparent 到该 ns 的 PID 1，PID 1 有义务 `wait()` 收割。业务进程当 PID 1 又不 reap → 容器里僵尸堆积。这也是 tini 的另一半价值。

### 22.3.5 NET Namespace 深潜：veth pair 与跨 ns 通信

每个 NET ns 拥有完全独立的：网络设备、路由表、iptables/nftables、socket 表、`/proc/net`、端口空间。**这就是两个容器能同时 bind 80 端口、各跑各的网络栈的原理。**

```
宿主网络栈                       容器网络栈（独立 NET ns）
eth0 (物理)                      eth0 (= veth peer，被改名)
docker0 (bridge) ── veth_host ───┄┄ veth_cont
lo                               lo
路由表 A                         路由表 B
iptables A (MASQUERADE 出外网)   iptables B
```

**veth pair** 是一对"虚拟网线两头"：包从一头进、另一头出。建容器网络的标准流程：
```
1. ip link add veth_host type veth peer name veth_cont   # 宿主创建 pair
2. ip link set veth_host master docker0                  # 一头插到 bridge
3. ip link set veth_cont netns <container_pid>           # 另一头移入容器 NET ns
4. (容器内) ip addr add 172.17.0.2/16 dev eth0; ip link set eth0 up
5. (宿主) iptables -t nat -A POSTROUTING -s 172.17.0.0/16 -j MASQUERADE
```
**根因提醒**：veth 数据路径要穿过宿主网络栈两次（进 bridge、过 netfilter），所以高 PPS 场景 veth+bridge 有可观开销——这正是 Cilium 用 eBPF 旁路（直接在 tc/XDP hook 转发，绕开 bridge 和 iptables）、或 macvlan/ipvlan（容器设备直接挂物理网卡）能显著提速的根因。

### 22.3.6 MNT Namespace 深潜：为什么光有 MNT ns 还不够

MNT ns 隔离的是"挂载点视图"。但**新建一个 MNT ns，进程的根目录 `/` 还指向宿主 rootfs**——你只是有了独立的挂载表，并没有换根。容器"有自己文件系统"靠两步：

1. **OverlayFS 叠出 rootfs**：
   ```
   lowerdir  = 镜像层（只读，可多层叠加）
   upperdir  = 容器可写层（写时复制 copy-up）
   workdir   = overlay 内部用的临时目录
   merged    = 容器内看到的 /（读写合并视图）
   ```
   读命中 lower 直接读；写一个 lower 里的文件时，overlay 先把它整份 **copy-up** 到 upper 再改——这就是"容器删了基础镜像里的文件、其他容器不受影响"的原理，也是为什么"在容器里改大文件第一次写很慢"（copy-up 整文件）。

2. **`pivot_root` 换根**（不是 chroot！）：把进程的根挂载点从宿主 rootfs 切到 merged 目录，旧根挂到一个子目录后 `umount -l` 卸掉。**为什么用 pivot_root 不用 chroot？** chroot 只改 `task->fs->root` 指针，旧根的挂载和 fd 还在，配合 `CAP_SYS_CHROOT` 可逃逸（经典 double-chroot escape）；pivot_root 真正把旧根从挂载树上摘除，逃逸面小得多。容器 runtime 一律用 pivot_root。

`unshare --mount` 可在不 fork 的情况下给当前 shell 一个新 MNT ns，是手搓容器第一步。

### 22.3.7 USER Namespace 深潜：rootless 的基石（附真实结构）

USER ns 是 7 种里最复杂、安全性最关键的。**核心能力**：进程在 user ns 内部持有 UID 0（root），但这个 0 映射到宿主一个普通 UID（如 1000）。映射表是 v6.12 里这个结构：

【真实源码 v6.12 · `include/linux/user_namespace.h`】
```c
struct uid_gid_extent {
	u32 first;
	u32 lower_first;
	u32 count;
};

struct uid_gid_map { /* 64 bytes -- 1 cache line */
	union {
		struct {
			struct uid_gid_extent extent[UID_GID_MAP_MAX_BASE_EXTENTS];
			u32 nr_extents;
		};
		struct {
			struct uid_gid_extent *forward;
			struct uid_gid_extent *reverse;
		};
	};
};

struct user_namespace {
	struct uid_gid_map	uid_map;
	struct uid_gid_map	gid_map;
	struct uid_gid_map	projid_map;
	struct user_namespace	*parent;
	int			level;
	kuid_t			owner;
	kgid_t			group;
	struct ns_common	ns;
	unsigned long		flags;
	/* parent_could_setfcap: true if the creator if this ns had CAP_SETFCAP
	 * in its effective capability set at the child ns creation time. */
	bool			parent_could_setfcap;
	/* ... 还有更多字段（keyring、rlimit ucounts 等），此处省略，标【示意-截断】 ... */
} __randomize_layout;
```
> 上方 `struct user_namespace` 末尾我**显式标了截断**：v6.12 真实定义还有 keyring、`ucounts`/rlimit 计数等字段，为聚焦只保留映射相关字段，其余以 `/* ... */` 省略——不是逐字全文，但所列字段逐字真实。`uid_gid_map` 与 `uid_gid_extent` 是逐字全文。

**逐字段读出设计意图：**
- **`union` 的两种布局（小映射内联 vs 大映射指针）**：注释写死 "64 bytes -- 1 cache line"。映射 extent 数 ≤ `UID_GID_MAP_MAX_BASE_EXTENTS`（5）时直接内联进结构体（一个 cache line 装下，查映射零额外解引用）；超过 5 个才退化成 `forward`/`reverse` 两棵排序数组指针。**根因是性能**：UID 翻译在每次权限检查的热路径上，绝大多数容器映射就一两条 extent，内联让常见情况快到无指针追逐。这是内核里"为常见情况优化布局"的典型微架构手法。
- **`parent` + `level`**：user ns 可嵌套（容器套容器），翻译 UID 要沿 parent 链逐层向上换算。`level` 限制嵌套深度（防止无限套娃打爆栈/内存）。
- **`parent_could_setfcap`**：注释解释得很清楚——记录创建本 ns 时父进程是否持有 `CAP_SETFCAP`。这是为了堵一个提权：防止非特权 user ns 内部给文件写 fcaps 然后拿到本不该有的能力。**这种"记录创建时刻的权限快照"字段，全是历史 CVE 打补丁打出来的**。

**两个落地铁律（面试高频）：**
1. **创建 user ns 不需要任何特权**（与其他所有 ns 的根本区别）。`unshare --user --map-root-user` 普通用户即可执行。
2. `/proc/<pid>/uid_map` 格式 `<内部uid起始> <外部uid起始> <count>`，且**写一次后锁定**（write-once）、一个非特权进程只能映射"包含自己 UID 的那一条"、行数上限 5（Kerrisk, LWN/532593 逐字："the number of lines that may be written to the file is limited to five"，"only a single write … to a uid_map file"）。write-once 是安全设计：映射一旦确立不可改，杜绝"先映射成普通用户骗过检查、再改成 root"的 TOCTOU 攻击。

### 22.3.8 Namespace 创建的三种系统调用

```c
// 1. clone()：创建子进程时一并建立新 namespace（容器 runtime 主用）
clone(child_fn, stack, CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS | SIGCHLD, NULL);

// 2. unshare()：当前进程进入新 namespace（不 fork，手搓容器第一步）
unshare(CLONE_NEWNS | CLONE_NEWUTS);

// 3. setns()：通过 fd 加入已存在的 namespace（nsenter / sidecar 共享网络）
int fd = open("/proc/<pid>/ns/net", O_RDONLY);
setns(fd, CLONE_NEWNET);
```
**`/proc/<pid>/ns/` 里每个文件是一个 nsfs 节点**，持有它的 fd 就能让 namespace 在所有进程退出后**继续存活**（`ip netns add` 就是把 net ns bind-mount 到 `/var/run/netns/` 来保活）。这也是 `kubectl exec` / `docker exec` 进容器的底层：拿到目标进程的 ns fd，`setns` 一组，再 exec。

---

## 22.4 cgroup：限制"能用多少"

### 22.4.1 cgroup 的内核骨架：css / css_set / cgroup

理解 cgroup 必须先认识三个核心结构。`task_struct` 里持有的是 `struct css_set *cgroups`（不是直接指向 cgroup！），这个间接层是性能关键。先看 controller 与 cgroup 的连接点 `cgroup_subsys_state`（缩写 css）：

【真实源码 v6.12 · `include/linux/cgroup-defs.h`】
```c
/* define the enumeration of all cgroup subsystems */
#define SUBSYS(_x) _x ## _cgrp_id,
enum cgroup_subsys_id {
#include <linux/cgroup_subsys.h>
	CGROUP_SUBSYS_COUNT,
};
#undef SUBSYS
```
```c
/*
 * Per-subsystem/per-cgroup state maintained by the system.  This is the
 * fundamental structural building block that controllers deal with.
 *
 * Fields marked with "PI:" are public and immutable and may be accessed
 * directly without synchronization.
 */
struct cgroup_subsys_state {
	/* PI: the cgroup that this css is attached to */
	struct cgroup *cgroup;

	/* PI: the cgroup subsystem that this css is attached to */
	struct cgroup_subsys *ss;

	/* reference count - access via css_[try]get() and css_put() */
	struct percpu_ref refcnt;

	/* siblings list anchored at the parent's ->children
	 * linkage is protected by cgroup_mutex or RCU */
	struct list_head sibling;
	struct list_head children;

	/* flush target list anchored at cgrp->rstat_css_list */
	struct list_head rstat_css_node;

	/* PI: Subsys-unique ID.  0 is unused and root is always 1. */
	int id;

	unsigned int flags;

	/* Monotonically increasing unique serial number … defines a
	 * uniform order among all csses … used to allow interrupting
	 * and resuming iterations. */
	u64 serial_nr;

	/* Incremented by online self and children. … parents are not
	 * offlined before their children. */
	atomic_t online_cnt;

	struct work_struct destroy_work;
	struct rcu_work destroy_rwork;

	/* PI: the parent css. */
	struct cgroup_subsys_state *parent;

	/* Keep track of total numbers of visible descendant CSSes. */
	int nr_descendants;
};
```
> 上方 css **所列字段逐字真实**，注释为聚焦做了少量删节（原注释更长），已标。`enum cgroup_subsys_id` 段为逐字全文。

**逐字段读出 cgroup 的核心设计：**
- **`SUBSYS(_x)` 宏 + `#include <linux/cgroup_subsys.h>`**：这是 X-macro 模式。所有 controller（cpu、memory、io、pids…）在一个 `cgroup_subsys.h` 列表里登记一次，这个宏自动展开成 `cpu_cgrp_id`、`memory_cgrp_id`… 的枚举。**新增/删除 controller 只改一处**——这正是本仓库 CLAUDE.md 里"metric registry 单点登记"哲学的内核同款先例。`CGROUP_SUBSYS_COUNT` 自动等于 controller 总数，用作各种 per-subsys 数组的长度。
- **css 是"(cgroup × subsystem) 的交叉点"**：`->cgroup` 指所属 cgroup，`->ss` 指哪个 controller。一个 cgroup 节点对每个使能的 controller 各有一个 css。memory controller 的 `mem_cgroup` 结构就内嵌一个 css 当头部。
- **`struct percpu_ref refcnt`**：注意是 **per-cpu 引用计数**，不是普通 atomic。根因是 cgroup 的 get/put 极度高频（每次内存 charge、每次调度都可能碰），全局 atomic 会成 cache line 争用热点；per-cpu ref 把增减打散到各 CPU 本地，只在"准备销毁"时才汇总（`percpu_ref_kill`）。这是排队论意义上的"消除单点竞争"。
- **`serial_nr` + 可中断迭代**：注释明说用于"interrupting and resuming iterations"。遍历庞大 cgroup 树时要能中途放锁再接着走（否则长时间持 `cgroup_mutex` 卡住别人），单调序号让"从上次停的地方继续"成为可能。
- **`online_cnt` / `nr_descendants`**：保证父 css 不会先于子 css 下线、统计可见子孙数——cgroup 的创建/销毁是并发的，这些计数维持树的生命周期不变量。

**`task → css_set → cgroup` 的间接层为什么存在？** 系统里可能有几万个进程，但**不同进程的"cgroup 组合"种类很少**（同一个容器里所有进程的 cgroup 归属完全一样）。`css_set` 就是"一组 css 归属的去重缓存"：归属相同的进程共享同一个 `css_set`。好处是改 cgroup 配置时按 `css_set` 批量操作，而非逐进程；坏处是这层间接增加了理解成本。**面试问"`task_struct` 怎么连到 cgroup"，答`task->cgroups`(css_set)→`subsys[]`→css→cgroup，并说出 css_set 是去重缓存，就是 staff 级。**

### 22.4.2 cgroup v1 设计与缺陷（"先实现后设计"的代价）

cgroup v1（2.6.24）的架构是**多层级（multiple hierarchies）**：每个 controller 可以挂到一棵独立的树上。

```
hierarchy 1（挂 memory）           hierarchy 2（挂 cpu）
  /sys/fs/cgroup/memory/             /sys/fs/cgroup/cpu/
    ├── container_A/                   ├── high_priority/
    └── container_B/                   └── low_priority/
进程可同时属于 hierarchy1 的 container_A 和 hierarchy2 的 low_priority！
```

**v1 的四宗罪**（Tejun 在 LWN/679786 中自陈 "design followed implementation … different decisions were taken for different controllers … sometimes too much flexibility causes a hindrance"）：
1. **语义混乱**：进程在 memory 树属 container_A，在 cpu 树属 low_priority——**两棵树的边界不重合**，运维心智模型崩塌，更要命的是**跨 controller 协作无从谈起**（见下条）。
2. **controller 无法协作**：page cache writeback 需要同时知道"内存压力"和"IO 能力"才能正确节流脏页回写。但 v1 里 memory 和 io 在不同 hierarchy、边界不一致，**内核没有一个共同的 cgroup 能同时问到这两类信息**。这是 Tejun 推 v2 统一层级最硬的技术理由（§22.4.4 详述）。
3. **线程 vs 进程归属规则不一致**，各 controller 自行其是，bug 频出。
4. **接口各搞各的**：`memory.memsw.limit_in_bytes`、cpuset 的 `clone_children`（一个只跟 cpuset 有关的开关却出现在所有 controller）、freezer/device 的怪癖……缺乏统一约定。

### 22.4.3 cgroup v2 设计：统一层级 + no-internal-process

cgroup v2（4.5 official）的核心变化是**单一统一层级（unified hierarchy）**：所有 controller 共用同一棵树，一个 cgroup 节点同时控制 cpu+memory+io，边界天然一致。

```
/sys/fs/cgroup/                       （唯一一棵树）
  ├── system.slice/
  │   ├── docker-<id>.scope/   ← 这一个节点同时有 cpu/memory/io controller
  │   └── nginx.service/
  └── user.slice/
```

**最反直觉、也最常踩的规则——No Internal Process Constraint**。v6.12 文档逐字：

【真实文档 v6.12 · `Documentation/admin-guide/cgroup-v2.rst`】
> *"Non-root cgroups can distribute domain resources to their children only when they don't have any processes of their own. In other words, only domain cgroups which don't contain any processes can have domain controllers enabled in their `cgroup.subtree_control` files. This guarantees that, when a domain controller is looking at the part of the hierarchy which has it enabled, processes are always only on the leaves. This rules out situations where child cgroups compete against internal processes of the parent."*

**翻译成人话 + 根因**：开了 controller 的非叶子节点**不能直接装进程**——进程只能在叶子上。为什么？因为如果父节点既有子 cgroup 又有自己的进程，调度器/内存控制器就得回答一个没有良定义答案的问题："父节点里这些散进程，和子 cgroup 这个整体，按什么比例分资源？" v1 允许这种"进程 vs cgroup 直接竞争"，语义全靠各 controller 自己瞎定。v2 用 no-internal-process 一刀切死：**资源分配图里，只有叶子是终端消费者，中间节点纯粹是分配域**。代价是你不能再像 v1 那样把进程随手丢在任意节点上——迁移老系统最常被这条规则绊倒（报 `EBUSY`/`ENOTSUP`）。

其余 v2 改进：
- **threaded mode**（4.14+）：专门给"需要线程级 cpu 分配"的场景开的后门，下一节讲它怎么来的。
- **PSI（Pressure Stall Information）**：v2 新增的压力监控（§22.4.7）。
- **接口统一**：`cpu.max`、`memory.max`、`io.max` 命名一致。

### 22.4.4 设计考古：CPU controller 进 v2 的五年战争（核心史料）

这是本章最值得展开的一段——它解释了 v2 几乎所有"怪规则"的来由。**全程有出处。**

**冲突的本质**：Tejun Heo（cgroup maintainer）和调度器阵营（Peter Zijlstra、Paul Turner）在一个根本点上对立——

> Tejun：**cgroup 的资源分配单位必须是"进程"，中间节点不能有进程**（no-internal-process）。
> 调度器：**调度器调度的是"线程"，而且历史上就支持把同进程的不同线程放进不同 cgroup**（v1 行为）。强行进程粒度会砍掉真实在用的能力。

**Tejun 的论据**（LWN/697369，"State of CPU controller in cgroup v2"，2016-08-05，逐点转述其原文）：
1. **内存语义逼出进程粒度**：*"an address space is shared between all threads of a process, the terminal consumer is a process, not a thread."* 一个进程的所有线程共享地址空间，page cache 一旦实例化就**无法再归因到单个线程甚至单个进程**——所以 *"a process can't be a first class object in the resource distribution graph as its total resource consumption can't be described without the containing resource domain."* memory/io 天然是进程粒度，为了**所有 controller 统一组织**，cpu 也得跟。
2. **common resource domain 论**：controller 之间要协作（最硬的例子还是 page cache writeback——*"dirty page cache is regulated through throttling buffered writers based on memory availability, and initiating batched write outs to the disk based on IO capacity"*），必须有一个共同的层级组织，cpu 不能自己搞一套。
3. **接口同步问题**：cgroup 的 vfs 接口是多步操作、无同步原语，允许线程级控制会制造无法解决的竞态。

**Zijlstra 的反击**（同文引用其邮件，附他画的 ASCII 对比图）：
```
        R                    R
      / | \                /   \
     t1 t2 A              L       A
        /   \           /   \   /   \
       t3   t4         t1  t2  t3   t4
   （进程与cgroup直接竞争）  （v2 强制：进程先归到 L）
```
他指出两种结构语义**根本不同**：左图给 R 加一个任务 t5，A 的带宽从 1/3 掉到 1/4；右图 *"A doesn't get any less bandwidth"*。强行禁止左图 = 砍掉一类合法布局。**Paul Turner** 进一步反对用 nice/priority 替代线程级 cgroup 控制——*"priorities were not a suitable solution while cgroups are."*

**Tejun 的让步与僵持**：他承认找不到左图的强 use case（称这类需求 *"super duper fringe"*、*"real world use cases of such layouts could not be established during the discussions"*），但调度器阵营不接受"为了 cgroup 统一性砍掉已有能力"。结果就是 **CPU controller 一度根本进不了 cgroup v2 主线**，在 2016 年僵住。

**破局：threaded cgroup**（Corbet "A milestone for control groups"，LWN/729215，2017-07-31；patchset 见 lore `[PATCHSET for-4.14]` / `[PATCHSET REPOST for-4.15]`）。妥协方案：
- 默认所有 cgroup 是 **domain cgroup**，遵守 no-internal-process + "一个进程的所有线程绑在一起"。
- 往 `cgroup.type` 写 `"threaded"` 把一棵子树标成 **threaded cgroup**，**在这棵子树内部**允许线程分散到不同节点、允许内部进程。
- 关键设计抉择（v3 patch）：Tejun 最终采纳"**逐 cgroup** 标 threaded"而非 Zijlstra 建议的"逐子树标"，换来更灵活、更少接口怪癖。

threaded mode 排队进 4.14、cgroup2 CPU controller 接口随后落地（4.15 一带），**五年拉锯收场**。

**这段历史教你什么（远超容器知识本身）**：
- v2 的 no-internal-process 不是洁癖，是**为了让多个 controller 能在同一棵树上协作**（writeback 那个例子）付出的必要约束。
- threaded cgroup 这种"看着别扭的特例"，几乎都是**两个都有道理的设计在真实约束下妥协的疤痕**。读内核源码遇到别扭设计，先假设它在补某个你还没看到的窟窿，而不是假设作者糊涂。
- "找不到 use case 就该砍" vs "已有能力不能随便砍"——这是系统演进里永恒的张力，Linus 体系下通常后者赢（不破坏现有用户），这也是为什么内核宁可背 threaded 这种复杂度也不肯直接砍线程级控制。

### 22.4.5 CPU 控制器：与调度器的真实耦合（逐行源码）

cgroup v2 CPU controller 两种机制。

**(A) CPU Weight（比例，work-conserving）**
```
cpu.weight（1-10000，默认 100）→ 直接换算成 CFS 的调度权重（见第 5 章）
  A.weight=200, B.weight=100 → CPU 满载时 A:B ≈ 2:1；CPU 空闲时不设限（能抢就抢）
```

**(B) CPU Bandwidth / `cpu.max`（硬配额）**。v6.12 文档逐字：

【真实文档 v6.12 · `Documentation/admin-guide/cgroup-v2.rst`】
> *"A read-write two value file … The default is `max 100000`. The maximum bandwidth limit. It's in the following format: `$MAX $PERIOD` which indicates that the group may consume up to $MAX in each $PERIOD duration. `max` for $MAX indicates no limit."*

即 `echo "50000 100000" > cpu.max` = 每 100ms 周期最多用 50ms CPU（= 0.5 核）。**它怎么在 CFS 里实现的？以下三段是 v6.12 逐字源码。**

第一段：**每次调度推进扣配额的热路径**——
【真实源码 v6.12 · `kernel/sched/fair.c`】
```c
static void __account_cfs_rq_runtime(struct cfs_rq *cfs_rq, u64 delta_exec)
{
	/* dock delta_exec before expiring quota (as it could span periods) */
	cfs_rq->runtime_remaining -= delta_exec;

	if (likely(cfs_rq->runtime_remaining > 0))
		return;

	if (cfs_rq->throttled)
		return;
	/*
	 * if we're unable to extend our runtime we resched so that the active
	 * hierarchy can be throttled
	 */
	if (!assign_cfs_rq_runtime(cfs_rq) && likely(cfs_rq->curr))
		resched_curr(rq_of(cfs_rq));
}
```
逐行注解：
- `cfs_rq->runtime_remaining -= delta_exec;`：`delta_exec` 是这一段刚跑掉的纳秒数（由 `update_curr` 算出）。**配额扣减就是这一行**——它跑在调度器每次更新当前任务运行时间的热路径上，证明 bandwidth controller **不是外层 wrapper，是长在 CFS 里的**。
- `if (likely(runtime_remaining > 0)) return;`：还有余额，啥也不做，零额外开销。`likely()` 提示分支预测器"通常有余额"（微架构层面减少 misprediction）。
- `if (cfs_rq->throttled) return;`：已经被 throttle 了，不重复处理。
- `if (!assign_cfs_rq_runtime(cfs_rq) && cfs_rq->curr) resched_curr(...)`：余额耗尽 → 向全局池**再要一片**；要不到（返回 0）且当前有任务在跑 → **触发重新调度**。注意：**这里并不直接 throttle**，只是 `resched_curr` 打个重调度标记，真正的摘除发生在下一个调度点调 `throttle_cfs_rq()`。这个"延迟到调度点"的设计避免在记账热路径里做重活。
- 注释 `dock delta_exec before expiring quota (as it could span periods)`：先扣再判过期，因为一段执行可能横跨周期边界。

第二段：**向全局池借配额**——
【真实源码 v6.12 · `kernel/sched/fair.c`】
```c
static int __assign_cfs_rq_runtime(struct cfs_bandwidth *cfs_b,
				   struct cfs_rq *cfs_rq, u64 target_runtime)
{
	u64 min_amount, amount = 0;

	lockdep_assert_held(&cfs_b->lock);

	/* note: this is a positive sum as runtime_remaining <= 0 */
	min_amount = target_runtime - cfs_rq->runtime_remaining;

	if (cfs_b->quota == RUNTIME_INF)
		amount = min_amount;
	else {
		start_cfs_bandwidth(cfs_b);

		if (cfs_b->runtime > 0) {
			amount = min(cfs_b->runtime, min_amount);
			cfs_b->runtime -= amount;
			cfs_b->idle = 0;
		}
	}

	cfs_rq->runtime_remaining += amount;

	return cfs_rq->runtime_remaining > 0;
}
```
逐行注解：
- `cfs_b`（`cfs_bandwidth`）是 **task group 级别的全局配额池**，`cfs_rq` 是**每 CPU 的本地队列**。这是个"中央银行 + 各分行"模型：全局池按 period 补满 quota，每个 CPU 从池里借小片用。
- `min_amount = target_runtime - cfs_rq->runtime_remaining`：要借多少（`runtime_remaining` 已 ≤0，所以这是正数，注释专门点明）。
- `if (cfs_b->quota == RUNTIME_INF) amount = min_amount;`：没设限（`cpu.max=max`）就要多少给多少。
- `start_cfs_bandwidth(cfs_b)`：**惰性启动周期 hrtimer**——只有真有 cgroup 在用 bandwidth 时才启动那个每 period 补配额的定时器，没人用就不空转。
- `amount = min(cfs_b->runtime, min_amount); cfs_b->runtime -= amount;`：从全局池有多少借多少（封顶到需求量），**这是临界区，`lockdep_assert_held(&cfs_b->lock)` 强制持锁**——全局池是所有 CPU 争抢的点，必须串行化。这也是 bandwidth controller 在**核很多**的机器上有锁竞争开销的根因：每次本地配额见底都要抢 `cfs_b->lock`。
- 返回 `runtime_remaining > 0`：借到了没有，决定 §1 里要不要 `resched_curr`。

第三段：**真正把整个 cfs_rq 从 CPU 上摘下**（节选前半，核心逻辑完整）——
【真实源码 v6.12 · `kernel/sched/fair.c`】
```c
static bool throttle_cfs_rq(struct cfs_rq *cfs_rq)
{
	struct rq *rq = rq_of(cfs_rq);
	struct cfs_bandwidth *cfs_b = tg_cfs_bandwidth(cfs_rq->tg);
	struct sched_entity *se;
	long task_delta, idle_task_delta, dequeue = 1;
	long rq_h_nr_running = rq->cfs.h_nr_running;

	raw_spin_lock(&cfs_b->lock);
	/* This will start the period timer if necessary */
	if (__assign_cfs_rq_runtime(cfs_b, cfs_rq, 1)) {
		/*
		 * We have raced with bandwidth becoming available, and if we
		 * actually throttled the timer might not unthrottle us for an
		 * entire period. We additionally needed to make sure that any
		 * subsequent check_cfs_rq_runtime calls agree not to throttle
		 * us, as we may commit to do cfs put_prev+pick_next, so we ask
		 * for 1ns of runtime rather than just check cfs_b.
		 */
		dequeue = 0;
	} else {
		list_add_tail_rcu(&cfs_rq->throttled_list,
				  &cfs_b->throttled_cfs_rq);
	}
	raw_spin_unlock(&cfs_b->lock);

	if (!dequeue)
		return false;  /* Throttle no longer required. */

	se = cfs_rq->tg->se[cpu_of(rq_of(cfs_rq))];

	/* freeze hierarchy runnable averages while throttled */
	rcu_read_lock();
	walk_tg_tree_from(cfs_rq->tg, tg_throttle_down, tg_nop, (void *)rq);
	rcu_read_unlock();
	/* …（向下逐层 dequeue_entity 把该组从各级 cfs_rq 摘除，略，标【示意-截断】）… */
}
```
逐行注解：
- **先试 `__assign_cfs_rq_runtime(cfs_b, cfs_rq, 1)` 要 1ns**：真正 throttle 前最后一搏——万一刚好和"周期补满配额"撞上（race），就别 throttle 了（`dequeue=0` 直接返回）。注释解释得很细：如果误 throttle，可能要白等整整一个 period 才被 unthrottle，代价太大，所以宁可多要 1ns 确认。**这就是并发系统里"提交前最后检查"的范式**。
- `list_add_tail_rcu(&cfs_rq->throttled_list, &cfs_b->throttled_cfs_rq);`：确实没配额 → 把这个 cfs_rq 挂进**全局 throttled 链表**，等下个 period 的 timer 来 unthrottle。
- `walk_tg_tree_from(..., tg_throttle_down, ...)`：**冻结整个子树的负载统计**——throttle 是层级的，要从这个 task group 往下递归把每层标记冻结，否则负载均衡会被"明明在队列里却不该跑"的实体误导。
- 截断部分是真正的 `for_each_sched_entity` 逐层 `dequeue_entity`——把这组实体从它在各级 CPU 队列里摘下来，**我已显式标【示意-截断】，未冒充全文**。

**配额片大小是源码常数（佐证原版"5ms slice"说法属实）：**
【真实源码 v6.12 · `kernel/sched/fair.c`】
```c
static unsigned int sysctl_sched_cfs_bandwidth_slice = 5000UL;       /* 微秒 → 5ms */
static const u64 min_cfs_rq_runtime = 1 * NSEC_PER_MSEC;             /* 1ms slack 阈值 */
```
`5000UL` 微秒 = **5ms**，就是每个 CPU 一次从全局池借的默认片大小（`sched_cfs_bandwidth_slice()` 再 ×1000 转纳秒）。`min_cfs_rq_runtime = 1ms` 是"归还本地剩余配额给全局池"的阈值（slack，避免把零碎全留本地导致其他 CPU 饿着）。

**💡生产真坑 —— CPU Throttling 假象（大厂最高频容器调优场景之一）：**
```
现象：容器 CPU 使用率监控显示 40%，但 P99 延迟从 50ms 飙到 500ms，服务无报错
根因链（结合上面源码）：
  cpu.max = "400000 100000"（4 核 / 100ms period）
  请求 burst：某 100ms 窗口内瞬间用满 400ms（4 核全开）→ runtime_remaining 见底
  → __account_cfs_rq_runtime 借不到 → resched_curr → 下个调度点 throttle_cfs_rq
  → 该组被挂 throttled_list，整组按住直到下个 period（最坏等 ~60ms）
  → 期间请求在队列里排队 → P99 跳升
  但跨整个监控窗口平均下来 CPU 利用率只有 40% → "利用率不高却被 throttle"假象
诊断：cat cpu.stat | grep -E 'nr_throttled|throttled_usec'
      nr_throttled = throttle 事件次数；throttled_usec = 累计被夺走的时间（看这个！）
解法（按优先级）：
  1. 加大 period：echo "4000000 1000000" > cpu.max（1s period，同样 4 核，但容忍更长 burst）
  2. 提配额  3. 改 cpu.weight 放弃硬配额（风险：邻居打满时无保证）  4. 业务削峰
```
**为什么加大 period 有效？** 配额总量不变（4 核），但**补配额的频率变低、单次配额池变大**，一次 burst 更不容易在一个 period 内把池抽干。代价是 throttle 一旦发生，要等更久才解除——所以不是无脑调大，要和业务 burst 形状匹配。

### 22.4.6 Memory 控制器：与内存回收的真实耦合

```
memory.max       硬限制（充电超限 → 先 memcg 内 direct reclaim，仍不足 → memcg OOM killer）
memory.high      软限制（超过 → throttle + 重度 reclaim，不 kill）
memory.swap.max  swap 上限
memory.current   当前用量（含 page cache！见下方面试坑）
memory.stat      分类统计（anon / file / slab / kernel …）
```
v6.12 文档逐字定义：
【真实文档 v6.12 · `Documentation/admin-guide/cgroup-v2.rst`】
> `memory.high`：*"Memory usage throttle limit. If a cgroup's usage goes over the high boundary, the processes of the cgroup are throttled and put under heavy reclaim pressure."*
> `memory.max`：*"Memory usage hard limit. This is the main mechanism to limit memory usage of a cgroup."*

**`memory.high` 的"软刹车 + 背压"机制**（近年内核，函数名 `mem_cgroup_handle_over_high()` 一族，「具体调用点细节待核」）：
```
进程申请内存（page fault / charge 路径）
  → memcg 充电时发现 usage > memory.high
  → 当场同步 reclaim（直接在分配路径上回收该 memcg 的 lruvec）
  → 仍超 high → 记一笔"惩罚"，进程下次返回用户态时被强制睡眠一段
    （penalty 时长与超额量成正比）
  → 效果：进程被"减速"制造背压，给 reclaim 争取时间，但永不 kill
```
**关键：`memory.high` 没有硬上界**——一直超就一直被 throttle 到几乎不前进，但**不会 OOM**。它是"软刹车"，不是"墙"。`memory.max` 才是墙：
```
进程 → charge 超 memory.max → 先在该 memcg 内 direct reclaim（复用第 10 章 shrink_lruvec 那套，
                              只是扫描范围从全局 node 收窄到单个 memcg 的 lruvec）
                          → 仍不足 → memcg OOM killer，只在该 cgroup 子树内挑 badness 最高的进程 kill
                          → 不波及宿主其他进程（这是容器内存隔离的关键保证）
```
**💡v2 专属 `memory.oom.group = 1`**：OOM 时把整个 cgroup 当一个单位**整组 kill**，而非只杀一个。对"一个进程死了其他也没意义"的 pod 很有用——避免留下半死不活的残体。

**🎯面试加分点 —— 容器里 `free` 为什么"内存虚高"？** `memory.current` **包含 page cache（file-backed）**。容器跑久了读了很多文件，page cache 全算进 `memory.current`，`free` 看 used 接近 limit 让人以为要 OOM。但 page cache 是**可回收**的——真正回收不掉的是 `memory.stat` 里的 `anon`。诊断内存压力看 `anon` 接近 `memory.max` 的程度，别被含 cache 的总数吓到。

**first-touch charging（多容器共享镜像层的内存计费 nuance）**：file-backed page 由**第一个把它 charge 进来的 cgroup** 记账，不是最后访问者。多容器共享同一镜像层（同一 inode）时，第一个读到该页的容器被算"拥有者"，后来者免费搭便车。这解释了为什么"同样的镜像、不同启动顺序，各容器 `memory.current` 数字不一样"。注意：**first-touch 是 memcg 通用规则（v1/v2 皆然），不是 v2 独有。**

### 22.4.7 IO 控制器与 PSI

**IO controller**：v1 的 blkio 只能控 direct IO（不管 buffered write，因为脏页回写发生在内核 writeback 线程上下文、和发起写的进程脱钩了）。v2 靠 **cgroup writeback** 把脏页回写正确归因回发起写入的 cgroup，从而能限 buffered IO。
```
echo "8:0 rbps=10485760 wbps=10485760" > io.max   # 设备 8:0 读写各限 10MB/s
io.weight                                          # 比例分配
```

**PSI（Pressure Stall Information，v2 独有）**——监控"有多少时间进程卡在等资源"：
```bash
cat /sys/fs/cgroup/system.slice/docker.service/cpu.pressure
# some avg10=0.34 avg60=0.25 avg300=0.10 total=12345678
# full avg10=0.00 ...
# some: 至少一个进程在等该资源   full: 所有进程都在等（严重饥饿）
# avg10/60/300: 10s/60s/5min 滑动平均占比
```
**生产价值（根因层面）**：CPU 利用率回答"CPU 有多忙"，但 P99 延迟的真正来源是"任务排队等了多久"——这是排队论里的等待时间，利用率高不一定等待时间长（取决于到达分布）。PSI `some` **直接测等待时间占比**，是延迟异常的早期信号，比利用率灵敏得多。利用率 60% 但 PSI some=30% → 30% 时间有任务在等 CPU → 该扩容了。

---

## 22.5 容器 = 几样之和（含安全层）

```
┌──────────────────────────── 容器 = 内核已有特性的组合 ────────────────────────────┐
│  Namespaces（看到什么）          cgroup v2（用多少）           rootfs（文件系统）      │
│  PID/NET/MNT/UTS/IPC/USER/...    cpu.max / cpu.weight          OverlayFS lower(镜像)  │
│                                  memory.max / memory.high      OverlayFS upper(可写)  │
│  Capabilities（特权细分）         io.max / io.weight            pivot_root 换根        │
│  drop 绝大多数 cap               pids.max（防 fork bomb）                              │
│                                                                                      │
│  + seccomp（syscall 过滤） + AppArmor/SELinux（MAC） ← 现代 runtime 默认叠加的安全层  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 22.5.1 Capabilities：把 root 拆成 ~40 块

Linux 把 root 的全权拆成约 40 个 capability（`CAP_NET_ADMIN`、`CAP_SYS_ADMIN`…）。runtime 默认 **drop 绝大多数**，只留容器正常运行的最小集：
```
默认保留（Docker 默认，示意，各版本略有差异）：
  CAP_CHOWN, CAP_DAC_OVERRIDE, CAP_FOWNER, CAP_SETUID, CAP_SETGID,
  CAP_NET_BIND_SERVICE（绑 <1024 端口）, CAP_KILL, ...
默认删除：
  CAP_SYS_ADMIN（"新的 root"，权能极广）, CAP_NET_ADMIN, CAP_SYS_PTRACE,
  CAP_SYS_MODULE（加载内核模块）, CAP_SYS_TIME, ...
```
**`CAP_SYS_ADMIN` 为什么被叫"新 root"？** 它管的操作太杂（mount、setns、pivot_root、各种 admin syscall…），授它几乎等于授全权——这是 capability 体系本身的设计缺陷（粒度不够细，太多东西塞进一个 cap），也是为什么 rootless + user ns 比"给几个 cap"更彻底。

**新手致命误区 —— `--privileged` 远不止"给全部 capability"。** 它**同时**：① 给全部 cap；② 关掉 seccomp；③ 关掉 AppArmor/SELinux confinement；④ 放开所有宿主设备节点（`/dev`）；⑤ 去掉 `/proc`、`/sys` 的只读 mask。等于把上图所有防护层一次拆光。**所以 privileged 容器逃逸往往易如反掌**（例如直接 mount 宿主磁盘、写 `/proc/sysrq-trigger`、加载内核模块）。生产铁律：要某能力 `--cap-add` 精确给，要某设备 `--device` 单独挂，**永远别 `--privileged`**。

### 22.5.2 seccomp 与 AppArmor/SELinux

- **seccomp**：syscall 过滤。**方向常被说反**——x86_64 有 350+ syscall，Docker 默认 profile 是 **allowlist 写法、默认放行绝大多数、只拦约 40+ 个高危/历史遗留 syscall**（`keyctl`、`add_key`、`kexec_load`、`reboot`、部分 `clone` flag…）。它收的是"攻击面的尾巴"，不是主干。
- **AppArmor/SELinux**：MAC（强制访问控制），按 path/label 限制文件操作。容器逃逸常见的一环就是绕过/缺失这层。

这两样不是本章核心，但"`--privileged` 到底拆了哪几层"是容器安全高频追问，务必能列全 5 条。

### 22.5.3 安全考古：Dirty Pipe 为什么让"共享内核"成为原罪

**Dirty Pipe（CVE-2022-0847，Max Kellermann 披露）** 是把"namespace 是隔离不是沙箱"钉死的活案例：
- **机制**：pipe buffer 的 `PIPE_BUF_FLAG_CAN_MERGE` 标志在 `splice()` 后可能残留未清零，导致后续对 pipe 的写**直接覆写 page cache 页**，而不是新建 buffer。原文：*"it is possible to overwrite the page cache even in the absence of writers, with no timing constraints, at (almost) arbitrary positions with arbitrary data."*
- **引入/修复**：Linux **5.8**（commit `f6dd975583bd`，2020）引入；**5.16.11 / 5.15.25 / 5.10.102**（2022-02）修复（fix commit `9d2231c5d74e`）。
- **为什么对容器是灾难**：① **只需读权限**就能改文件——read-only 挂载形同虚设；② page cache 是**全机一份**，一个非特权容器里的进程能污染其他容器/宿主缓存的文件（比如改宿主上某个被 root 执行的脚本、或 setuid 程序）；③ namespace 把"视图"挡住了，但底层 page cache 这个共享对象 namespace 根本不管。

**底层根因**：容器的所有进程和宿主**共用同一个内核地址空间、同一份 page cache、同一套内核代码路径**。namespace/cgroup 是内核**自愿提供**的视图/配额隔离，一个内核 bug 可以绕过它们直达共享状态。**这就是 microVM 用独立 guest kernel 把"共享内核"这个根换掉的全部动机**——隔离强度的天花板由"是否共享内核"决定，不由你叠多少 seccomp/AppArmor 决定。面试讲容器安全，能把 Dirty Pipe 这条具体链路 + "共享内核是隔离强度天花板"讲出来，远胜泛泛而谈。

---

## 22.6 rootless 容器（生产态）

### 22.6.1 原理：user namespace 撑起整条生命周期

传统 Docker daemon 以 root 跑，攻破 daemon = 拿宿主 root。**rootless 让整条容器生命周期在普通用户权限下运行**，核心就是 §22.3.7 的 user namespace：
```
普通用户 uid=1000：
  unshare --user --map-root-user      ← 在新 user ns 里把自己映射成 root（uid 0→1000）
  在新 user ns 里再 unshare --pid --net --mount ...
  → 新 ns 里的"root"能操作这些 ns 内资源（§22.3.2 源码里每个 copy_*_ns 都按新 user_ns 判权限）
  → 但这个 root 在宿主眼里始终是 uid=1000，碰不到宿主特权资源
```
**podman 默认 rootless，这是它与 Docker 的主要差异之一。**

### 22.6.2 限制（都源自"宿主侧没真特权"）

- **网络**：建 veth、改 bridge 需要宿主 `CAP_NET_ADMIN`，rootless 没有 → 退化到 **slirp4netns/pasta**（用户态 TCP/IP 栈做 NAT），性能有损（用户态拷贝 + 协议栈开销）。
- **存储**：`mknod` 设备节点需 `CAP_MKNOD`，某些存储驱动不可用，多用 fuse-overlayfs。
- **cgroup**：v1 里非 root 不能建 cgroup；**v2 靠 delegation（委托）**——systemd 把某子树的属主交给普通用户，用户即可在自己子树里建/管 cgroup。这是 rootless + 资源限制能共存的关键。

### 22.6.3 收口：非特权 user ns 是把双刃剑

"非特权可创建 user ns"让 rootless 成立，**也把一大票原本要 root 的内核代码路径暴露给普通用户**——十年来一连串容器逃逸/提权 CVE 的入口。各发行版近年纷纷加旋钮收口（手搓/排障必须分清，**别用错 sysctl**）：
- **上游 / RHEL / Fedora**：`user.max_user_namespaces`（设 0 即禁用，upstream 通用旋钮）。
- **Debian / Ubuntu**：额外带下游补丁 `kernel.unprivileged_userns_clone`（已被官方标 deprecated，Ubuntu 上可能两个旋钮并存）。
- **Ubuntu 23.10+**：再叠 AppArmor 限制 `kernel.apparmor_restrict_unprivileged_userns`——即便 sysctl 放开，默认 profile 仍拦掉大多数非特权 userns。**这是近年容器逃逸收口的真实变化**：从"默认全开"转向"默认收紧、按需放行"。

---

## 22.7 落地：生产调优与排障

### 22.7.1 常用查看命令

```bash
ls -la /proc/<pid>/ns/                     # 看进程各 ns 的 inode（比对是否共享同一 ns）
nsenter -t <pid> -n ip addr                # 进入某进程的 NET ns 执行命令
systemd-cgls                               # 看 cgroup v2 层级树
cat /sys/fs/cgroup/.../memory.stat         # 区分 anon / file / slab
cat /sys/fs/cgroup/.../cpu.stat            # nr_throttled / throttled_usec
cat /sys/fs/cgroup/.../cpu.pressure        # PSI 实时压力
cat /proc/<pid>/cgroup                     # 某进程属于哪个 cgroup（最稳，别在 BPF 里抠结构）
```

### 22.7.2 典型排障场景

**场景一：容器频繁 OOM，但内存看起来够用**
```
1. cat memory.stat → anon 是否接近 memory.max（anon 才是回收不掉的）
2. memory.current - anon ≈ page cache（可回收，别被它吓到）
3. anon 远低于 max 却 OOM → 可能是 slab（内核内存，如大量 dentry/inode、socket buffer）
   → 看 memory.stat 的 slab/kernel 字段；v1 曾有内核 slab 不计入 memcg 的 bug，v2 修复
4. 也可能是 cgroup OOM 由瞬时尖峰触发 → 看 dmesg 的 memcg OOM 记录确认是哪个进程
```

**场景二：延迟毛刺，CPU 利用率看起来正常**（即 §22.4.5 的 throttle 假象，排障路径见那里）
```
1. cat cpu.stat 看 throttled_usec 增速  2. throttle 高 → period 太短，加大 period
3. cat cpu.pressure 看 PSI some  4. PSI some 高 → CPU 真超配，降部署密度
```

**场景三：容器内 `/proc/cpuinfo` 看到宿主所有核**
```
根因：/proc/cpuinfo 是内核级文件，不受 cgroup 控制；PID ns 只隔离 PID，不隔离 CPU 拓扑信息
后果：JVM 按 Runtime.availableProcessors() = 宿主核数起线程池 → 线程爆炸、内存高、上下文切换重
     （Go 的 GOMAXPROCS、各种线程池库同病）
解法：① JDK 8u191+/JDK 11+ 自动读 cgroup cpu.max（首选）
     ② 手动 -XX:ActiveProcessorCount=N  ③ 容器内设 GOMAXPROCS / 用 automaxprocs
错误答案：用 CPU affinity——affinity 不改 /proc 视图，JVM 还是数错核数
```

### 22.7.3 cgroup v1 / v2 共存与判别

```bash
mount | grep cgroup
# cgroup2 on /sys/fs/cgroup type cgroup2  → 纯 v2（unified）
# cgroup on /sys/fs/cgroup/memory type cgroup → v1
stat -fc %T /sys/fs/cgroup/   # cgroup2fs = v2；tmpfs = v1（混合）
# 强制回 v1（内核参数，慎用）：systemd.unified_cgroup_hierarchy=0
```
**关键判断**：走 v1 还是 v2 由**宿主内核 + systemd** 决定，**不是 runtime 自己选**。systemd 247+（Debian 11 / Ubuntu 22.04 / RHEL 9）默认 unified v2，runtime 只是适配宿主当前模式。老 Docker 只支持 v1，跑在纯 v2 宿主上会出问题。

---

## 22.8 深水区：两个耦合的底层根因

### 22.8.1 CPU bandwidth ↔ 调度器（源码已在 §22.4.5）

把 §22.4.5 的源码提炼成三条工程推论：
1. **throttle 判定随调度推进发生**（`update_curr` → `__account_cfs_rq_runtime`），粒度是调度 tick 级（1~4ms，看 `CONFIG_HZ`），不是微秒级。**极短 burst 可能在一个 5ms slice 内跑完，根本没触发 throttle**——所以"偶尔超配额"不一定有惩罚。
2. **`nr_throttled` 是事件次数，`throttled_usec` 才是被夺走的时长**。盯延迟必须看后者；只看前者会把"throttle 了很多次但每次极短"误判成严重。
3. **`cfs_b->lock` 是全局池锁**（源码 `lockdep_assert_held`/`raw_spin_lock`）。核越多、本地配额越容易见底、抢这把锁越频繁——**大核数 + 紧配额的服务，CFS bandwidth 本身会引入可观锁竞争**，这是 throttle 之外的二阶开销。
4. 版本提示：每 tick 入口历史上叫 `scheduler_tick()`，**近年（~6.10）重命名为 `sched_tick()`**——跨版本搜源码名字会对不上，机制不变。

### 22.8.2 Memory cgroup ↔ 内存回收（详见第 10 章）

```
内存 cgroup 树 ≈ 内存回收的组织单元（每个 memcg 有自己的 per-node lruvec）
memory.high 超 → 充电路径上同步 reclaim 该 memcg 的 lruvec → 仍超记惩罚 → 返回用户态 throttle
memory.max 超 → memcg direct reclaim → 不足 → memcg OOM killer（只在该子树内 kill）
（注意：内核另有 cgroup throttle swap rate 一族函数，那是 swap 繁忙时给分配限速，
  与 memory.high 背压是两条独立机制，别混。「该族确切函数名待核」）
```
**核心洞察**：memcg reclaim **不是另写一套回收器**，而是把全局回收的同一套 LRU 扫描（`shrink_node`/`shrink_lruvec`，第 10 章）**把扫描范围从"全局 node"收窄到"单个 memcg 的 lruvec"**。这就是为什么容器内存回收的行为（脏页回写、refault、active/inactive 平衡）和全机回收同源——理解了第 10 章就理解了 memcg reclaim 的一大半。

---

## 22.9 未来演进

- **cgroup v2 IO 精度**：cgroup writeback 对 buffered IO 的归因在高并发下仍有偏差，内核持续改进。
- **memory controller 的 slab 共享计费**：跨 cgroup 共享内核 slab 的归因方式仍在演进（`CONFIG_MEMCG_KMEM`）。
- **time namespace（5.6）**：让容器内 `CLOCK_MONOTONIC`/`BOOTTIME` 有独立起点，解决 CRIU **容器迁移后时钟跳变**——迁到新宿主后单调时钟若突变，依赖它的程序（如 Go runtime、各种超时计算）会错乱。
- **Landlock（5.13）**：用户态可编程的文件系统访问控制，无需 root——可理解为"seccomp 的文件系统版"，给 rootless 沙箱补一块。
- **io_uring × namespace（第 17 章）**：io_uring 的 fixed buffer/file、SQPOLL 内核线程跨 namespace 的行为是当前安全研究热点（多个 io_uring 提权 CVE）。
- **runtime 分化**：crun（C 写、轻量）、kata-containers（硬件虚拟化边界）、gVisor（用户态内核拦 syscall）、**WASM 容器**（wasmtime+containerd，用 WASM sandbox 替代部分 namespace 隔离，更细粒度更低开销，但资源控制仍落回 cgroup）。趋势是**隔离强度光谱化**——从纯 ns/cgroup 到 microVM，按信任度选档。

---

## 心法

> **Namespace 给进程一张地图，cgroup 给进程一个配额；容器是内核已有特性的重新组合，不是新概念。读 v2 的怪规则，先假设它在补某个 controller 协作的窟窿（writeback / no-internal-process / threaded），而不是假设作者糊涂——内核里别扭的设计，多半是两个都有道理的需求在真实约束下妥协的疤痕。**

---

## 章末五件套

### 一、高频面试题（Staff 级）

**Q1：容器和 VM 的根本区别？各适合什么？**
- 核心：VM 有独立内核，容器共享宿主内核。安全边界：VM 靠 hypervisor，容器靠内核（ns+cgroup+seccomp+MAC）。
- 工程推论：容器内核 LPE 直穿宿主（举 Dirty Pipe CVE-2022-0847：只需读权限、跨 read-only 挂载改 page cache）；VM/microVM 适合不信任多租户，容器适合同团队微服务。
- 陷阱答案："容器是轻量级 VM"——架构错误，二者不同维度的东西。加分：说出 microVM（Kata/Firecracker）是第三条路。

**Q2：cgroup v1 的核心缺陷？v2 怎么解？为什么 CPU controller 进 v2 拖了 5 年？**
- v1：多层级 → 进程跨层级归属不一致、语义混乱、**controller 间无法协作**（page cache writeback 需同时知内存压力+IO 能力，v1 给不出共同 cgroup）。
- v2：单一统一层级 + no-internal-process（保证 controller 看到的层级里进程只在叶子）。
- 5 年战争：Tejun 坚持进程粒度（内存语义逼出来的，LWN/697369），Zijlstra/Turner 坚持调度器调度线程、v1 已支持线程级 cgroup 不能砍 → 僵持 → **threaded cgroup**（4.14/4.15）妥协。能讲这段 = 顶配。

**Q3：容器内 /proc/cpuinfo 看到宿主所有核？怎么让 JVM 正确感知配额？**
- 根因：/proc/cpuinfo 是内核级文件，不受 cgroup 控制，PID ns 不隔离 CPU 拓扑。
- 解法：JDK 8u191+/11+ 自动读 cgroup cpu.max；或 -XX:ActiveProcessorCount=N。陷阱："用 affinity"——affinity 不改 /proc 视图。

**Q4：user namespace 怎么撑起 rootless？有什么限制？为什么它是安全双刃剑？**
- 原理：uid_map 把容器内 uid 0 映射到宿主普通 uid，创建 user ns 不需特权；在新 user ns 里再建其他 ns（源码：create_new_namespaces 每个 copy_*_ns 按新 user_ns 判权限）。
- 限制：网络退化到 slirp4netns（性能损），cgroup 靠 v2 delegation，不能 mknod。
- 双刃：非特权 user ns 暴露大量本需 root 的内核路径 → 一连串逃逸 CVE → 发行版加 sysctl/AppArmor 收口（user.max_user_namespaces / apparmor_restrict_unprivileged_userns）。

**Q5：CPU throttle 致 P99 高，怎么诊断和解？**（结合源码答更深）
```
诊断：1. cat cpu.stat 看 throttled_usec 增速（不是 nr_throttled）
     2. 对比 CPU 利用率——低利用率 + 高 throttle = 经典 burst 假象
     3. cat cpu.pressure 看 PSI some
机制：__account_cfs_rq_runtime 扣穿 runtime_remaining → 借不到 → resched_curr
     → throttle_cfs_rq 把整组挂 throttled_list，等下个 period（最坏等近一个 period）
解法（优先级）：加大 period（池更大更耐 burst）> 提 quota > 改 cpu.weight > 业务削峰
```

**Q6：描述容器创建的完整内核序列（ns+网络+fs+cgroup+cap）？**
```
1. runtime clone(CLONE_NEWUSER|NEWPID|NEWNET|NEWNS|NEWUTS|NEWIPC|...) 建进程+ns
   （走到 create_new_namespaces，按 mnt→uts→ipc→pid→cgroup→net→time 顺序复制）
2. 容器进程在新 PID ns 内 PID=1
3. runtime（宿主侧）：建 cgroup 目录、写 cgroup.procs 把进程纳管；
   建 veth pair，一端插 bridge、一端 ip link set netns 移入容器 NET ns；配 IP/路由/MASQUERADE
4. 容器进程内：mount overlay(lower=镜像,upper=可写)；pivot_root 换根；mount /proc /sys /dev
5. drop capabilities 到白名单；apply seccomp profile + AppArmor/SELinux label
6. execve entrypoint
```

**Q7：memory.high vs memory.max？内核怎么实现"软限制"？**
- max：硬限，超 → memcg direct reclaim → 不足 → memcg OOM kill。
- high：软限，超 → 充电路径同步 reclaim + 进程返回用户态时按超额量 throttle（减速不 kill，无硬上界）。
- 用法：high 做早期背压/自动回收，max 做最后防线。加分：oom.group 整组 kill；memory.current 含 page cache 导致 free 虚高。

**Q8：PID=1 在容器里的特殊性？为什么 ENTRYPOINT 用 exec 形式？**
- PID=1：无 handler 的信号被静默丢弃（init 保护）+ 有义务 reap 僵尸。
- shell 形式 `sh -c myapp`：PID 1 是 sh，SIGTERM 发给 sh，sh 默认不转发 → 容器停不下来（10s 后被 SIGKILL）→ 数据丢。
- exec 形式 `["myapp"]`：myapp 直接当 PID 1 收 SIGTERM；或用 tini/dumb-init 当 PID 1 转发信号 + reap。

---

### 二、实战项目（P3 增量）

**P3：用 unshare/cgroup 手搓极简"容器"（不借 Docker）**
```bash
# 验收：① 容器内 PID=1、独立 hostname、独立 NET ns
#       ② memory.max=64MB 验证 OOM   ③ cpu.max=50% 验证 throttle

# A. 准备 rootfs
docker export $(docker create ubuntu) | tar -C /tmp/rootfs -xf -

# B. 建 cgroup v2 并设限
mkdir /sys/fs/cgroup/mycontainer
echo "67108864" > /sys/fs/cgroup/mycontainer/memory.max     # 64MB
echo "50000 100000" > /sys/fs/cgroup/mycontainer/cpu.max    # 0.5 核

# C. 启动隔离进程
unshare --mount --uts --ipc --pid --net --fork bash -c '
  echo $$ > /sys/fs/cgroup/mycontainer/cgroup.procs   # 自己纳入 cgroup
  hostname mycontainer
  mount --bind /tmp/rootfs /tmp/rootfs; cd /tmp/rootfs
  mount -t proc proc proc/; mount -t sysfs sysfs sys/; mount -t tmpfs tmpfs dev/
  pivot_root . old_root/; umount -l /old_root
  hostname; echo $$    # 期望：mycontainer / 1
  bash -c "while true; do dd if=/dev/zero of=/dev/null bs=1M; done"
'
# D. 另开终端观测
cat /sys/fs/cgroup/mycontainer/memory.current
cat /sys/fs/cgroup/mycontainer/cpu.stat   # throttled_usec 应随 CPU 压测增长
# 验收：□ echo $$ =1  □ hostname 与宿主不同  □ 超 64MB 触发 OOM(dmesg)  □ throttled_usec 增长
```
**注意**：
- `pivot_root` 对 new_root 必须是 mount point 等约束跨版本有细微差别，上面是示意，实跑要调。
- 非特权创建 user ns 的开关因发行版而异，**别用错 sysctl**（见 §22.6.3：`user.max_user_namespaces` vs `kernel.unprivileged_userns_clone` vs `kernel.apparmor_restrict_unprivileged_userns`）。

---

### 三、设计题

**为多租户 AI 推理服务设计资源隔离**（共享 GPU，CPU/内存/网络隔离，优先级，内存抖动不拖垮宿主）：
```
隔离层：推理可信 → cgroup v2 + seccomp 够；完全不信任 → Kata（但 GPU passthrough 复杂）
CPU：cpu.weight 按付费等级（免费 100 / 付费 500）；cpu.max 设上限防独占；
     period=1s（推理 burst 明显，短 period 必触发 throttle 假象，见 §22.4.5）
内存：memory.high=配额*0.8（早回收背压）；memory.max=配额（OOM 只在容器内）；
     监控 PSI memory.pressure 超阈预警；oom.group=1 让坏 pod 整组死
IO：io.weight 按优先级；日志/checkpoint IO 限速不与推理竞争
网络：每容器独立 NET ns + veth；veth 上 tc qdisc 限速（补 cgroup io 只管磁盘的空缺）
OOM 防护：memory.max 兜底；宿主预留 reserved memory；关键进程 oom_score_adj 保护
```

---

### 四、系统题（排障）

**生产 K8s，某服务 P99 从 50ms→500ms，CPU 利用率监控仅 40%（配额 2 核），无报错。排查？**
```
第一步：kubectl exec -- cat /sys/fs/cgroup/cpu.stat，看 throttled_usec 增速 → 高则 throttle 是根因
第二步：理解 40% 却 throttle —— burst：某 100ms 窗口用满 200ms 后续 0ms，平均 40% 但 burst 期 throttle
第三步：验证 burst —— bpftrace 看 throttle 事件频率 / 分析 /proc/<pid>/schedstat（等待 vs 运行）
第四步：修复 —— A.加大 period(echo "2000000 1000000">cpu.max) B.提 quota C.改 cpu.weight D.业务削峰
第五步：验证 —— throttled_usec 增速下降 + P99 恢复
```
**真能跑的观测**（近年内核，root）——直接看"CFS 把整组 throttle 掉的那一刻"：
```bash
sudo bpftrace -e '
  kprobe:throttle_cfs_rq { @throttles = count(); }
  interval:s:1 { print(@throttles); clear(@throttles); }
'
# 对照 watch -n1 "grep nr_throttled /sys/fs/cgroup/.../cpu.stat" —— 两边同步增长，
# 即"配额耗尽 → 调度器把整组踢下 CPU"的直接证据。throttle_cfs_rq 是 §22.4.5 那段源码的函数。
```
> 不要在 BPF 里手抠 `task->cgroups->dfl_cgrp->kn->name` 这种跨版本易碎的字段链；查"某 PID 属哪个 cgroup"永远用 `cat /proc/<pid>/cgroup` 最稳。

---

### 五、代码题（可运行片段）

**用 clone() 创建独立 PID namespace 的子进程，验证子进程在 ns 内看到自己是 PID 1：**
```c
/* minimal_pid_ns.c  编译: gcc -o m m.c   运行: 需 root 或外包一层 user ns */
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#define STACK_SIZE (1024 * 1024)

static int child_fn(void *arg) {
    printf("[child] 容器内 PID: %d\n", getpid());      /* 期望 1 */
    printf("[child] 容器内 PPID: %d\n", getppid());     /* 期望 0：父不在同一 PID ns */
    /* 要让容器内 ps/top 正确，还需 CLONE_NEWNS + mount -t proc proc /proc */
    sleep(2);
    return 0;
}

int main(void) {
    char *stack = malloc(STACK_SIZE);
    if (!stack) { perror("malloc"); return 1; }

    /* CLONE_NEWPID 只对子进程生效；父进程仍在原 PID ns（getpid 不变）
     * 无 CAP_SYS_ADMIN 时返回 EPERM → 解法：外包 CLONE_NEWUSER，或以 root 运行 */
    pid_t host_pid = clone(child_fn, stack + STACK_SIZE,
                           CLONE_NEWPID | SIGCHLD, NULL);
    if (host_pid == -1) { perror("clone"); free(stack); return 1; }

    printf("[parent] 宿主侧看子进程 PID: %d\n", host_pid);
    printf("[parent] 自己 PID: %d\n", getpid());

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cat /proc/%d/status | grep NSpid", host_pid);
    system(cmd);   /* NSpid: <host_pid>  1  ← 第二列即 PID ns 内的 1 */

    waitpid(host_pid, NULL, 0);
    free(stack);
    return 0;
}
/* 暗坑：
 * 1. CLONE_NEWPID 只影响子进程，父 getpid() 不变
 * 2. 子进程 /proc 仍是宿主 /proc，需 CLONE_NEWNS + mount proc 才能让 ps 在容器内对
 * 3. 子进程是 PID=1，发它的 SIGTERM 被默认忽略（像真 init）
 * 4. 需 root 或 user ns 外包才能跑 */
```
> 上方 C 程序标【示意-可运行教学代码，非内核源码】。与之对照的**内核侧逐字源码**见 §22.3.2（`create_new_namespaces`，CLONE_NEWPID 最终落到 `copy_pid_ns`）。

---

*本章完*
