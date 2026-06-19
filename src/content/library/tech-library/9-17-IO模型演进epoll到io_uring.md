---
title: "IO 模型演进（epoll → io_uring）【深化版】"
slug: "9-17"
collection: "tech-library"
group: "linux系统"
order: 9017
summary: "一句话定位：IO 模型 25 年的演进只做了一件事——把\"每个事件的固定成本\"逐项消灭：select/poll 时代消灭的是线程，epoll 消灭的是每次调用的全量扫描与注册，io_uring 消灭的是系统调用与拷贝本身。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T05:45:19.000Z"
updatedAt: "2026-06-12T05:45:19.000Z"
---
> **一句话定位**：IO 模型 25 年的演进只做了一件事——把"每个事件的固定成本"逐项消灭：select/poll 时代消灭的是线程，epoll 消灭的是每次调用的全量扫描与注册，io_uring 消灭的是系统调用与拷贝本身。本章用 v6.12 真实源码 + 一手史料（LWN、commit、邮件列表）把这条演进链从 ABI 一直挖到内存序。

---

## TL;DR — 本章最硬结论

1. **select/poll 的 O(n) 不是实现没优化，是 ABI 决定的**：每次调用都要全量传入兴趣集，内核必须每次重建/拆除全部 wait queue（`fs/select.c:do_select` 源码可证）。epoll 的本质创新只有一个——**把兴趣集变成内核里的持久数据结构**，注册一次，事件经由 wait queue 回调主动入队。
2. **epoll 最初是 ET-only 的**，LT 是 2003 年 3 月 Davide Libenzi 被现实逼出来的补丁——他自己的理由是 *"developers not quite understand ET APIs"* 和 *"most existing apps are written using LT APIs"*（LWN Articles/25011）。LT/ET 在今天内核里的全部区别就是 `ep_send_events()` 里的一个 `list_add_tail`：LT 把 epitem 重新挂回就绪链表。
3. **Linux AIO 失败的根因是架构而非态度**：completion-based API 要求内核全路径可挂起重续，而 Linux 同步 IO 路径深处处可睡眠；AIO 社区长期拒绝"线程卸载"方案认为不够纯粹，2016 年 Linus 直接开骂（LWN Articles/671649）。io_uring 的聪明之处是把线程卸载（io-wq）藏成实现细节，先 inline 试一把，不行再丢给 worker。
4. **io_uring = 把 NVMe/NIC 硬件队列模型上移成 syscall ABI**：两个 SPSC ring 共享内存，单生产者单消费者使得 acquire/release 内存序就够了（`io_uring/io_uring.c` 文件头注释逐字在手）；SQPOLL 的睡眠/唤醒是一个教科书级的 Dekker 式协议（置标志 → full barrier → 再检查）。
5. **SQE 在用户态可写的共享内存里**，内核必须 `READ_ONCE` 一次取数、校验后不得重读——double-fetch/TOCTOU 是 io_uring 安全模型的第一性约束，源码注释明说 *"we cannot rely on userspace always being a good citizen"*。
6. **io_uring 是 exploit 富矿是结构性的**：Google kCTF 数据——60% 的内核漏洞提交打的是 io_uring，单这一项赏金约 100 万美元（总盘 180 万）；Google 在 ChromeOS、生产服务器禁用，Android 用 seccomp 挡住 app，结论原话是 *"we currently consider it safe only for use by trusted components"*（oss-security, 2023-07）。
7. **惊群的三次治理各有锚点**：accept 惊群靠 exclusive wait（2.4 时代）；多 epfd 监视同一 listenfd 靠 `EPOLLEXCLUSIVE`（4.5，Jason Baron 的 commit 实测 860s→24s）；根治靠 `SO_REUSEPORT`（3.9，Tom Herbert，commit message 里写明动机：共享 accept 队列下负载不均可达 3:1）。
8. **方案选型不是"新的赢"**：fd 少于十个时 select 仍是最快路径；regular file 上 epoll 语义上无意义（永远 readable）；io_uring 在小并发、低速率、需要安全审计、需要跨平台的场景全部不占优。本章 17.11 用三个场景把账逐笔算给你看。

---

## 前置依赖章

- **第 2 章**（系统调用机制）——syscall 开销量化是本章所有"账本"的基础
- **第 6 章**（调度与上下文切换）——惊群、SQPOLL 烧核的代价计算
- **第 14 章**（VFS）——`file_operations->poll` 是 select/poll/epoll 共同的底座
- **第 16 章**（块 IO 层)——IOPOLL 模式与 NVMe 队列对
- **第 20 章**（网络栈）——`sk_data_ready` 回调是 epoll 就绪通知的起点
- 内存序基础（`smp_load_acquire`/`smp_store_release`、`smp_mb`）——17.9 的硬前提

---

## 17.0 本章路线图

本章按三层组织，可按需取用：

```
第一层（API 与历史）   17.1–17.4, 17.7–17.8   —— 谁、何时、为什么、吵过什么
第二层（内核实现）     17.5–17.6, 17.9–17.10  —— v6.12 真实源码逐行
第三层（工程决策）     17.11–17.13 + 五件套   —— 场景推演、排障、选型边界
```

源码引用约定：

- 【真实源码 v6.12 路径:行号】= 从 `https://raw.githubusercontent.com/torvalds/linux/v6.12/<path>` 逐字取得（本次写作时实际抓取核对），删节处以 `/* ... 中略 ... */` 显式标出；
- 【示意代码，非逐字】= 为讲解而简化；
- 「待核」= 未能从一手来源核实的具体数字/版本。

---

## 17.1 背景：IO 模型是排队论问题，也是微架构问题

### 17.1.1 数量级的暴政

CPU 在纳秒级工作，IO 在微秒到毫秒级：

```
L1 cache hit            ~1 ns
syscall 进出（含 mitigations）  ~100–1000 ns        （第 2 章）
上下文切换               ~1–10 µs                  （第 6 章）
NVMe 4K 随机读           ~10–100 µs
数据中心内 RTT           ~50–500 µs
跨地域 RTT               ~10–100 ms
```

IO 等待比计算慢 3–6 个数量级，所以**任何让 CPU 同步等 IO 的设计都是在烧钱**。问题是：不等，谁来等？等的代价记在哪本账上？

把每种 IO 模型的成本拆成四类固定开销，演进史立刻清晰：

```
成本项               每连接固定成本        每事件固定成本           每轮调用成本
──────────────────────────────────────────────────────────────────────────
thread-per-conn     线程栈+task_struct    2 次上下文切换           —
select/poll         —                    1 次唤醒                O(n) 拷贝+扫描+wait queue 重建
epoll               ~160B epitem         1 次回调+1 次唤醒+拷贝    O(ready) 
io_uring            —（可选 fixed slot）  ~0（批量摊薄+共享内存）   O(batch)/N
```

**每一代模型就是把上一代账单里最贵的那一项消掉。** 这是本章的主线。

### 17.1.2 thread-per-connection 的真实账本

单核时代的解法是"一连接一进程/线程"，Apache prefork 是代表。账本：

1. **虚拟内存**：线程栈默认 8MB（`ulimit -s`），10K 连接 = 80GB **虚拟保留**。注意精度：因 overcommit 这不是 RSS，64 位下真正先压垮你的是页表、栈实际增长和 TLB；32 位下是地址空间本身。Dan Kegel 在 C10K 页面上算的是当年的账：*"if each thread gets a 2MB stack ... you run out of virtual memory at 512 threads on a 32 bit machine with 1GB user-accessible VM"*（kegel.com/c10k.html，逐字）。
2. **内核内存**：每线程内核栈 x86-64 上 16KB（4 页，不可换出）+ `task_struct` 1–2KB 量级。10K 线程仅内核栈 160MB。
3. **调度**：10K 线程全活跃时 runqueue 操作、上下文切换的直接成本（µs 级 × 频次）之外，更隐蔽的是**cache 污染**——每次切换后 L1/L2 对新线程是冷的，TLB 部分失效。这是"上下文切换 5µs"远低估真实代价的原因（间接成本可达直接成本数倍，取决于工作集）。

### 17.1.3 C10K：问题被命名的时刻

1999 年 Dan Kegel 发表著名的 C10K 页面，开篇就是宣战：

> *"It's time for web servers to handle ten thousand clients simultaneously, don't you think? After all, the web is a big place now."*
> *"you can buy a 1000MHz machine with 2 gigabytes of RAM and an 1000Mbit/sec Ethernet card for $1200 or so."*
> —— Dan Kegel, *The C10K problem*（kegel.com/c10k.html，逐字）

硬件已经便宜到单机万连接在物理上成立，软件栈成了瓶颈。Kegel 枚举的策略分类至今仍是标准框架：

1. 单线程多客户端 + 非阻塞 IO + **水平触发**就绪通知（select、poll、Solaris /dev/poll）
2. 单线程多客户端 + 非阻塞 IO + **就绪变化（边沿）**通知（epoll、RT signals、kqueue）
3. 异步 IO（completion 通知，aio_*）
4. thread-per-client
5. 内核态 server（khttpd、TUX——这条路线死了，但它的精神在 io_uring 里复活：把更多工作下沉进内核态一次完成）

C10M（千万并发）的提法来自 Robert Graham 2013 年的 Shmoocon 演讲，核心论点后置到 17.13。

---

## 17.2 五种 IO 模型：两阶段框架与内核机制映射

### 17.2.1 分类矩阵（保留经典框架）

以"读 socket"为例，把一次 IO 拆成两阶段：

- **Phase 1：等待数据就绪**（数据到达内核 socket 接收队列）
- **Phase 2：数据从内核拷贝到用户空间**

```
模型            Phase 1（等待）        Phase 2（拷贝）       内核机制
────────────────────────────────────────────────────────────────────────
阻塞 IO         阻塞                  阻塞                 sk_wq 上睡眠
非阻塞 IO       轮询（EAGAIN）         阻塞                 每次 poll 一遍直接返回
IO 多路复用     阻塞在 select/epoll    阻塞（read 时）       wait queue + 回调
信号驱动 IO     SIGIO 通知            阻塞                 信号队列（会溢出，见下）
异步 IO         不阻塞                不阻塞（内核搬完）     completion 投递
```

内核视角的统一翻译：**Phase 1 的所有变体都是"在哪个 wait queue 上睡、被谁唤醒"的问题；Phase 2 的所有变体都是"copy_to_user 发生在谁的上下文里"的问题。** io_uring 之前的所有模型，Phase 2 都在调用者上下文同步发生；io_uring 第一次让两个阶段都可以不占用调用线程。

### 17.2.2 一个被历史淘汰但必须懂的方案：RT signals

很多人不知道 epoll 之前 Linux 社区押注的 ET 机制是 **realtime signals**（`F_SETSIG` + `SIGIO`）：每个 fd 就绪投递一个携带 fd 信息的 RT 信号。它失败的根因是**信号队列是有限资源**：

> *"If sigwaitinfo returns a traditional SIGIO, the signal queue overflowed."*
> —— C10K 页面（逐字）

队列溢出后内核退化投递普通 SIGIO（不带 fd 信息），应用必须 fallback 到全量 poll() 兜底——也就是说**正确的 RT signals 程序必须同时实现两套事件循环**。一个在过载时把复杂度推回给应用的通知机制，注定出局。这给后续设计者留下的教训直接体现在 epoll 与 io_uring 上：**就绪/完成队列的溢出语义必须由内核兜底**（epoll 的 rdllist 天然不丢事件；io_uring 5.5 起 `IORING_FEAT_NODROP`，CQ 满时内核侧暂存 overflow 链表）。

### 17.2.3 工程推论（修正版）

1. IO 多路复用不是异步 IO：select/poll/epoll 只解决 Phase 1，read/write 仍同步。
2. 非阻塞 IO 单独使用是烧 CPU 在 EAGAIN 上。
3. POSIX AIO（glibc `aio_read`）在 Linux 上是**用户态线程池模拟**，与内核 AIO（`io_submit`）是两套东西——前者从未进过内核。
4. 真正的完成式异步在 Linux 上直到 2019 年 io_uring 才算成立。

---

## 17.3 select/poll：把 O(n) 焊死在 ABI 里

### 17.3.1 ABI 层面的死刑判决

```c
int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
int poll(struct pollfd *fds, nfds_t nfds, int timeout);
```

select 的 `fd_set` 是 1024 bit 的位图（`FD_SETSIZE`，glibc ABI 固化，改了就破坏所有已编译程序）；poll 换成数组去掉了上限。但两者共享同一个 ABI 级缺陷：**兴趣集每次调用整体传入，调用返回即被遗忘**。内核没有任何跨调用的记忆，于是每次都必须：

1. 全量 `copy_from_user` 兴趣集（O(n)）；
2. 遍历每个 fd 调用 `vfs_poll`（即 `file_operations->poll`），顺手把自己挂上每个 file 的 wait queue（O(n) 次内存分配+链表操作）；
3. 无就绪则睡眠；被任一 fd 唤醒后**再全量扫一遍**确认谁就绪（因为唤醒不携带"谁干的"信息——wait queue 的 wakeup 只是把你弄醒）；
4. 返回前拆掉所有 wait queue 挂载，结果 `copy_to_user` 回去（select 还要用户态再扫一遍位图）。

### 17.3.2 真实源码：do_select 的三层循环

【真实源码 v6.12 fs/select.c:505-575，逐字，有删节】

```c
	for (;;) {
		unsigned long *rinp, *routp, *rexp, *inp, *outp, *exp;
		bool can_busy_loop = false;

		inp = fds->in; outp = fds->out; exp = fds->ex;
		rinp = fds->res_in; routp = fds->res_out; rexp = fds->res_ex;

		for (i = 0; i < n; ++rinp, ++routp, ++rexp) {
			unsigned long in, out, ex, all_bits, bit = 1, j;
			unsigned long res_in = 0, res_out = 0, res_ex = 0;
			__poll_t mask;

			in = *inp++; out = *outp++; ex = *exp++;
			all_bits = in | out | ex;
			if (all_bits == 0) {
				i += BITS_PER_LONG;
				continue;
			}

			for (j = 0; j < BITS_PER_LONG; ++j, ++i, bit <<= 1) {
				struct fd f;
				if (i >= n)
					break;
				if (!(bit & all_bits))
					continue;
				mask = EPOLLNVAL;
				f = fdget(i);
				if (fd_file(f)) {
					wait_key_set(wait, in, out, bit,
						     busy_flag);
					mask = vfs_poll(fd_file(f), wait);

					fdput(f);
				}
				if ((mask & POLLIN_SET) && (in & bit)) {
					res_in |= bit;
					retval++;
					wait->_qproc = NULL;
				}
				/* ... 中略：POLLOUT_SET / POLLEX_SET 同构处理 ... */
			}
			/* ... 中略：写回 res_in/res_out/res_ex ... */
			cond_resched();
		}
		wait->_qproc = NULL;
		if (retval || timed_out || signal_pending(current))
			break;
		/* ... 中略：busy poll 处理 ... */
		if (!poll_schedule_timeout(&table, TASK_INTERRUPTIBLE,
					   to, slack))
			timed_out = 1;
	}
```

逐行解剖（行号按上文片段相对位置）：

- **`for(;;)` 外层**：睡醒一次重扫全部 fd——这就是"唤醒不携带信息"的代价。10K fd 中 1 个就绪，也要扫 10K 次。
- **位图跳跃优化**：`all_bits == 0` 时整 long（64 个 fd）跳过——内核工程师在常数项上能省则省，但 O(n) 的形状救不回来。
- **`fdget(i)` / `fdput(f)`**：每个 fd 每轮一次引用计数操作。多线程共享 fdtable 时这是原子操作，意味着 cacheline 在核间弹跳——这个细节后面 io_uring 的 registered files 会专门回收（17.9.6）。
- **`mask = vfs_poll(fd_file(f), wait)`**：统一的 VFS poll 协议。`wait` 里带着 `_qproc` 回调指针：**第一圈循环时**，每个 fd 的 `->poll()` 实现会经由 `_qproc`（这里是 `__pollwait`）把当前线程挂到自家 wait queue 上；
- **`wait->_qproc = NULL`（找到就绪后）**：一旦确认有结果马上要返回了，后续 fd 就不必再挂 wait queue——以及外层循环第二圈起 `_qproc` 恒为 NULL，**只查询不再注册**。也就是说 wait queue 挂载在一次 syscall 内只做一遍；但**下一次 syscall 全部重来**（`poll_initwait`/`poll_freewait` 包着整个函数）。
- **`cond_resched()`**：扫描 10K fd 可能耗时不短，主动让出避免 RT 延迟尖刺——内核对"大 n 的 select 很慢"心知肚明。

**结论性洞见**：select/poll 慢的不是"遍历"本身，而是 ABI 强迫内核把"建立监听关系"这个本可摊销的操作放进每次调用。LWN 在 2002 年报道 epoll 时一句话点破：

> *"The persistent data structure built around the epoll file descriptor is one of the reasons for this scalability: there is no need to set it up and tear it down for every epoll_wait() call."*
> —— LWN, *sys_epoll - making poll fast*（lwn.net/Articles/14168/，2002-10-30，逐字）

### 17.3.3 select/poll 的不适用边界——以及它们仍然正确的场景

别把演进史读成"旧的全错"。fd 数量 ≤ 个位数、调用频率低时，select/poll 是**更快**的：没有 epoll_ctl 的注册成本（红黑树插入 + wait queue 挂载 + 90–160B 内核内存），一次 syscall 全搞定，数据结构在栈上 cache 友好。glibc 内部、各种 CLI 工具、子进程管道收集这类场景至今用 poll 是正确选型。**epoll 的优势从"fd 数量 × 调用频次"足够大、且就绪集稀疏时才开始兑现。**

---

## 17.4 设计考古 I：epoll 的诞生（1999–2003）

### 17.4.1 时间线（带出处）

```
1999        Dan Kegel 发表 C10K（kegel.com/c10k.html）
~1998–2000  Solaris /dev/poll；FreeBSD 2000-07 合入 kqueue（Jonathan Lemon）
2001–2002   Davide Libenzi 开发 /dev/epoll 补丁（字符设备 + ioctl/mmap 接口）
2002-10     /dev/epoll port to 2.5.42（LWN Articles/12772）
2002-10     Linus 要求改成系统调用而非 /dev 设备 → sys_epoll
            （LWN Articles/13176、14168："at Linus Torvalds's request ... converted
             into a new set of system calls"）
2002-10-28  IBM LSE（Hanna Linder）站台 benchmark：
            "epoll is many times better than standard poll"（LWN Articles/13918）
2002-10     epoll 合入 2.5.44（epoll(7) man page："The epoll API was introduced
            in Linux 2.5.44"；glibc 2.3.2 提供封装）
2003-03-10  Libenzi 发布 LT 补丁（针对 2.5.64，LWN Articles/25011）
2003-12     2.6.0 发布：epoll 定型为 LT 默认 + EPOLLET 可选
2016-01     4.5 加入 EPOLLEXCLUSIVE（commit df0108c5da56，Jason Baron）
```

### 17.4.2 辩论一：/dev 设备 vs 系统调用

Libenzi 的原始实现是 `/dev/epoll`：open 一个字符设备，ioctl 注册兴趣，mmap 一块共享内存读事件（注意：**共享内存读事件——这个思路 17 年后在 io_uring 身上还魂了**）。Linus 拒绝设备接口，要求做成正经 syscall。表层理由是接口规范性；深层是 `/dev` 接口绕开了类型检查与权限模型，且 ioctl 多路复用语义是内核社区长期厌恶的 API 形态。于是有了 `epoll_create/epoll_ctl/epoll_wait` 三件套。

代价也真实存在：syscall 化之后每次收事件都要一次 `epoll_wait` 进出内核，mmap 直读事件的零拷贝特性丢了。**2019 年 io_uring 把这个被砍掉的设计捡了回来，并用"两个方向都共享内存"做得更彻底**——历史在这里画了一个完整的圈。

### 17.4.3 辩论二：要不要用户态回调

LWN 14168 的评论区保存了一场典型争论：有人问为什么不把回调直接送进用户态（像 Windows APC / kqueue 的 udata 那样省一层 demux）。社区的回答是：内核打进用户态的回调本质上是信号语义，带来重入、锁、栈切换的全部噩梦。epoll 坚持"内核只交付事件数组，控制流永远属于应用"。这个决定塑造了之后 20 年 Linux 事件驱动程序的形态：**事件循环是用户态的库问题（libevent/libuv/tokio），内核只做最小机制**。

### 17.4.4 辩论三：ET-only 的傲慢与 LT 的回归

最初合入的 epoll 是**纯边沿触发**的。Libenzi 当时的立场是 ET 在机制上更优（少一次重扫）。五个月后他自己提交了 LT 补丁，给出的理由非常诚实：

> *"developers not quite understand ET APIs"*；*"most existing apps are written using LT APIs"*；
> *"The LT epoll is by all means the fastest poll available and can be used wherever poll can be used"*
> —— Davide Libenzi，lt-epoll 补丁邮件，2003-03-10，针对 2.5.64（LWN Articles/25011，逐字）

这是 API 设计史上一次教科书式让步：**机制上更优的接口输给了认知成本**。ET 降格为 `EPOLLET` 标志位，LT 成为默认。2.6.0（2003-12）就是这个形态。深入实现层面你会发现这次让步的代价低得惊人——见 17.5.5：LT 在内核里只是"一行重挂"。

### 17.4.5 为什么不直接抄 kqueue

FreeBSD 的 kqueue（2000）在 API 设计上至今被公认更优雅：单一 `kevent()` 调用同时提交 changelist 和收割 eventlist（io_uring 的"提交+收割一次进出"在这里有先声）；filter 机制统一了 fd、signal、timer、进程状态、文件修改等事件源。Linux 没有采纳，走了"一物一 fd"的路线：epoll 只管 fd，随后用 signalfd、timerfd、eventfd（2.6.22–2.6.27 间陆续合入）把其他事件源**变成 fd** 塞回 epoll。

两条路线的 trade-off 值得认真对待：

- kqueue 路线：一次 syscall 干两件事，API 表面积小，但 `struct kevent` 的 filter/flags/fflags 语义高度复用，扩展靠加 filter 文档迅速复杂化；
- Linux 路线：每个原语独立 fd、独立语义、可独立演进与组合（eventfd 后来成为 io_uring/KVM/vhost 的通知原语，远超 epoll 的用途），代价是事件源转 fd 有额外开销、API 数量膨胀。

**"一切皆 fd"是 Linux 的组合哲学税，也是它的复利。** 评价接口时不要只看单点优雅度，要看十年后的演化路径。

### 17.4.6 IBM 的站台与 benchmark 文化

epoll 能在 2.5 冻结窗口前合入，IBM Linux Scalability Effort 的背书功不可没。Hanna Linder（IBM）2002-10-28 在 lkml 发的测试结论：

> *"The results of our testing show not only does the system call interface to epoll perform as well as the /dev interface but also that epoll is many times better than standard poll."*
> —— LWN Articles/13918（逐字），详细数据当年发布在 lse.sourceforge.net/epoll

注意这个细节：合入主线前先回答"syscall 化有没有损失性能"（对比 /dev 接口），再回答"比现状好多少"（对比 poll）。**两组对照、先证无回退再证收益**——这是内核性能类补丁的标准举证姿势，今天给内核提性能补丁依然如此。

---

## 17.5 epoll 内核实现精读（v6.12 fs/eventpoll.c）

### 17.5.1 三条路径总览

```
注册路径（epoll_ctl ADD）          就绪路径（中断/软中断上下文）        收割路径（epoll_wait）
─────────────────────            ─────────────────────────         ─────────────────────
ep_insert()                       网卡 → NAPI → TCP 栈              ep_poll()
 ├─ 分配 epitem                    → sk->sk_data_ready()             ├─ 有就绪？ep_send_events()
 ├─ ep_rbtree_insert()             → wake_up(sk_wq)                  ├─ 无就绪：挂 ep->wq 睡眠
 └─ ep_ptable_queue_proc()         → ep_poll_callback()  ◄── 关键    └─ 被 ep_poll_callback 唤醒
     把 ep_poll_callback             ├─ epitem 挂入 rdllist
     挂到 socket 的 wait queue       └─ wake_up(ep->wq)
```

epoll 的全部魔法：**把"线程睡在 10000 个 socket 的 wait queue 上"换成"一个回调函数替线程守在 10000 个 wait queue 上，线程只睡自己的 ep->wq"**。

### 17.5.2 数据结构

【真实源码 v6.12 fs/eventpoll.c:131-172，逐字，注释为内核原注释】

```c
struct epitem {
	union {
		/* RB tree node links this structure to the eventpoll RB tree */
		struct rb_node rbn;
		/* Used to free the struct epitem */
		struct rcu_head rcu;
	};

	/* List header used to link this structure to the eventpoll ready list */
	struct list_head rdllink;

	/*
	 * Works together "struct eventpoll"->ovflist in keeping the
	 * single linked chain of items.
	 */
	struct epitem *next;

	/* The file descriptor information this item refers to */
	struct epoll_filefd ffd;

	/*
	 * Protected by file->f_lock, true for to-be-released epitem already
	 * removed from the "struct file" items list; together with
	 * eventpoll->refcount orchestrates "struct eventpoll" disposal
	 */
	bool dying;

	/* List containing poll wait queues */
	struct eppoll_entry *pwqlist;

	/* The "container" of this item */
	struct eventpoll *ep;

	/* List header used to link this item to the "struct file" items list */
	struct hlist_node fllink;

	/* wakeup_source used when EPOLLWAKEUP is set */
	struct wakeup_source __rcu *ws;

	/* The structure that describe the interested events and the source fd */
	struct epoll_event event;
};
```

逐字段解读：

- `rbn` 与 `rcu` 共用 union：epitem 活着时在红黑树里，死了之后 rbn 没用了，原地复用为 RCU 释放头——内核数据结构按生命周期阶段复用内存的惯用法。
- `rdllink`：挂入 `ep->rdllist` 的链节。**一个 epitem 同时活在两个数据结构里**：红黑树（按 {file*, fd} 索引，服务 epoll_ctl 查找）+ 就绪链表（服务 epoll_wait 收割）。
- `next` + `ovflist`：双缓冲机制的栓子，17.5.4 详解。
- `ffd`（`epoll_filefd` = {struct file *, int fd}）：**epoll 的身份键是 (file*, fd) 二元组**，不是 fd——这直接导致 17.5.6 的经典坑。
- `pwqlist`：本 epitem 挂在目标 file 的哪些 wait queue 上（一个 file 的 poll 可能涉及多个队列）。
- `event`：用户注册的兴趣 events + user data 原样存这里。

每个被监视 fd 的内核开销就是一个 epitem + 一个 eppoll_entry，epoll(7) man page 给的账：**32 位内核约 90 字节，64 位约 160 字节**；`/proc/sys/fs/epoll/max_user_watches`（2.6.28 起）默认允许吃掉约 1/25（4%）的低端内存。100 万 watch ≈ 160MB 内核内存——能撑，但要进容量规划。

【真实源码 v6.12 fs/eventpoll.c:179-242，逐字，有删节】

```c
struct eventpoll {
	/*
	 * This mutex is used to ensure that files are not removed
	 * while epoll is using them. This is held during the event
	 * collection loop, the file cleanup path, the epoll file exit
	 * code and the ctl operations.
	 */
	struct mutex mtx;

	/* Wait queue used by sys_epoll_wait() */
	wait_queue_head_t wq;

	/* Wait queue used by file->poll() */
	wait_queue_head_t poll_wait;

	/* List of ready file descriptors */
	struct list_head rdllist;

	/* Lock which protects rdllist and ovflist */
	rwlock_t lock;

	/* RB tree root used to store monitored fd structs */
	struct rb_root_cached rbr;

	/*
	 * This is a single linked list that chains all the "struct epitem" that
	 * happened while transferring ready events to userspace w/out
	 * holding ->lock.
	 */
	struct epitem *ovflist;

	/* wakeup_source used when ep_send_events or __ep_eventpoll_poll is running */
	struct wakeup_source *ws;

	/* The user that created the eventpoll descriptor */
	struct user_struct *user;

	struct file *file;

	/* used to optimize loop detection check */
	u64 gen;
	struct hlist_head refs;
	/* ... 中略：refcount、CONFIG_NET_RX_BUSY_POLL 的 napi busy poll 字段、lockdep nests ... */
};
```

四个核心成员对应四个angles：`rbr`（兴趣集，O(log n) 增删查）、`rdllist`（就绪集，O(1) 入队）、`wq`（epoll_wait 调用者睡这）、`poll_wait`（**epoll fd 本身也可以被另一个 epoll 监视**——嵌套 epoll，`gen`/`refs` 就是为嵌套环路检测服务的）。两把锁分工：`mtx` 保护慢路径（ctl、收割），`rwlock_t lock` 保护快路径（rdllist/ovflist，会在中断上下文被拿）。

### 17.5.3 注册路径：回调是怎么挂上去的

`ep_insert()` 借用 VFS poll 协议完成挂载——和 do_select 用同一个 `->poll()` 接口，但 `_qproc` 换成了 epoll 自己的：

【真实源码 v6.12 fs/eventpoll.c:1413-1438，逐字】

```c
static void ep_ptable_queue_proc(struct file *file, wait_queue_head_t *whead,
				 poll_table *pt)
{
	struct ep_pqueue *epq = container_of(pt, struct ep_pqueue, pt);
	struct epitem *epi = epq->epi;
	struct eppoll_entry *pwq;

	if (unlikely(!epi))	// an earlier allocation has failed
		return;

	pwq = kmem_cache_alloc(pwq_cache, GFP_KERNEL);
	if (unlikely(!pwq)) {
		epq->epi = NULL;
		return;
	}

	init_waitqueue_func_entry(&pwq->wait, ep_poll_callback);
	pwq->whead = whead;
	pwq->base = epi;
	if (epi->event.events & EPOLLEXCLUSIVE)
		add_wait_queue_exclusive(whead, &pwq->wait);
	else
		add_wait_queue(whead, &pwq->wait);
	pwq->next = epi->pwqlist;
	epi->pwqlist = pwq;
}
```

- `init_waitqueue_func_entry(&pwq->wait, ep_poll_callback)`：wait queue entry 的 `func` 不是默认的"唤醒线程"，而是 `ep_poll_callback`。**wait queue 在 Linux 里从来不只是"睡眠队列"，而是泛化的回调注册点**——epoll 整个机制就建立在这个泛化上。
- `EPOLLEXCLUSIVE` 分支：决定挂普通节点还是 exclusive 节点（17.6 惊群治理的机制底座，`__wake_up_common` 遇到 exclusive entry 且回调返回非零时停止继续唤醒）。
- 对比 do_select：同样的挂载动作，select 每次 syscall 重做 n 次，epoll 一生只做一次。**select 与 epoll 的全部复杂度差异，物理上就落在这一处调用频次上。**

### 17.5.4 就绪路径：ep_poll_callback 与 ovflist 双缓冲

数据到达后：网卡中断 → NAPI → TCP 入队 → `sk->sk_data_ready()` → `wake_up(&sk->sk_wq->wait)` → 遍历 wait queue 执行每个 entry 的 func → 进入 `ep_poll_callback`（**注意：此时在软中断上下文，持有 socket 锁，每一纳秒都贵**）：

【真实源码 v6.12 fs/eventpoll.c:1308-1376，逐字，有删节】

```c
static int ep_poll_callback(wait_queue_entry_t *wait, unsigned mode, int sync, void *key)
{
	int pwake = 0;
	struct epitem *epi = ep_item_from_wait(wait);
	struct eventpoll *ep = epi->ep;
	__poll_t pollflags = key_to_poll(key);
	unsigned long flags;
	int ewake = 0;

	read_lock_irqsave(&ep->lock, flags);

	ep_set_busy_poll_napi_id(epi);

	/*
	 * If the event mask does not contain any poll(2) event, we consider the
	 * descriptor to be disabled. This condition is likely the effect of the
	 * EPOLLONESHOT bit that disables the descriptor when an event is received,
	 * until the next EPOLL_CTL_MOD will be issued.
	 */
	if (!(epi->event.events & ~EP_PRIVATE_BITS))
		goto out_unlock;

	/*
	 * Check the events coming with the callback. At this stage, not
	 * every device reports the events in the "key" parameter of the
	 * callback. We need to be able to handle both cases here, hence the
	 * test for "key" != NULL before the event match test.
	 */
	if (pollflags && !(pollflags & epi->event.events))
		goto out_unlock;

	/*
	 * If we are transferring events to userspace, we can hold no locks
	 * (because we're accessing user memory, and because of linux f_op->poll()
	 * semantics). All the events that happen during that period of time are
	 * chained in ep->ovflist and requeued later on.
	 */
	if (READ_ONCE(ep->ovflist) != EP_UNACTIVE_PTR) {
		if (chain_epi_lockless(epi))
			ep_pm_stay_awake_rcu(epi);
	} else if (!ep_is_linked(epi)) {
		/* In the usual case, add event to ready list. */
		if (list_add_tail_lockless(&epi->rdllink, &ep->rdllist))
			ep_pm_stay_awake_rcu(epi);
	}

	/*
	 * Wake up ( if active ) both the eventpoll wait list and the ->poll()
	 * wait list.
	 */
	if (waitqueue_active(&ep->wq)) {
		if ((epi->event.events & EPOLLEXCLUSIVE) &&
					!(pollflags & POLLFREE)) {
			switch (pollflags & EPOLLINOUT_BITS) {
			case EPOLLIN:
				if (epi->event.events & EPOLLIN)
					ewake = 1;
				break;
			case EPOLLOUT:
				if (epi->event.events & EPOLLOUT)
					ewake = 1;
				break;
			case 0:
				ewake = 1;
				break;
			}
		}
		wake_up(&ep->wq);
	}
	if (waitqueue_active(&ep->poll_wait))
		pwake++;

out_unlock:
	read_unlock_irqrestore(&ep->lock, flags);
	/* ... 中略：ep_poll_safewake(嵌套唤醒)、EPOLLEXCLUSIVE 之外 ewake=1、POLLFREE 处理 ... */
	return ewake;
}
```

四个硬核细节：

1. **两道快速滤网**（`EP_PRIVATE_BITS` 检查和事件交集检查）：EPOLLONESHOT 触发过的、事件不匹配的，在软中断里几条指令就弹走。回调的设计纪律：**先便宜地拒绝，再昂贵地接受**。
2. **ovflist 双缓冲**——本函数最精妙的并发设计。问题：`ep_send_events` 要往用户态拷贝事件（`copy_to_user` 可能缺页、可能睡眠），期间不能持有 `ep->lock`（中断上下文要拿它）。不持锁，rdllist 又会被收割逻辑掏空重排，回调此刻并发插入会竞态。解法：收割开始时（`ep_start_scan`）把 rdllist 整体搬到私有 txlist，并把 `ovflist` 从哨兵值 `EP_UNACTIVE_PTR` 置为 NULL（= "双缓冲开启"）；期间所有新就绪事件由本回调链到 `ovflist` 单链表；收割结束（`ep_done_scan`）再把 ovflist 合并回 rdllist。**事件零丢失，且拷贝期间完全无锁。** 这是"用阶段切换替代细粒度锁"的范本。
3. **lockless 入链**：`list_add_tail_lockless` / `chain_epi_lockless` 用 cmpxchg 实现，配合读侧 `read_lock`（多个回调可并发入链，收割侧 `write_lock` 独占）——rwlock 的用法在这里是反直觉的：**回调（写者语义）拿读锁并发跑，收割（读者语义）拿写锁独占**，锁保护的真正对象是"双缓冲模式的切换瞬间"。
4. **EPOLLEXCLUSIVE 的 ewake 裁决**：返回值决定 `__wake_up_common` 是否继续唤醒下一个 exclusive waiter。注意它检查**事件类型匹配**——只关心 EPOLLOUT 的 waiter 不该为 EPOLLIN 消耗一次独占唤醒名额，否则事件会被"唤而不取"地吞掉。这是 EPOLLEXCLUSIVE 语义里最容易被忽略的正确性细节。

### 17.5.5 收割路径与 LT 的"一行实现"

`ep_poll()` 的等待循环（节选）：

【真实源码 v6.12 fs/eventpoll.c:2037-2064，逐字，有删节；函数签名见 1966 行】

```c
		init_wait(&wait);
		wait.func = ep_autoremove_wake_function;

		write_lock_irq(&ep->lock);
		/*
		 * Barrierless variant, waitqueue_active() is called under
		 * the same lock on wakeup ep_poll_callback() side, so it
		 * is safe to avoid an explicit barrier.
		 */
		__set_current_state(TASK_INTERRUPTIBLE);

		/*
		 * Do the final check under the lock. ep_start/done_scan()
		 * plays with two lists (->rdllist and ->ovflist) and there
		 * is always a race when both lists are empty for short
		 * period of time although events are pending, so lock is
		 * important.
		 */
		eavail = ep_events_available(ep);
		if (!eavail)
			__add_wait_queue_exclusive(&ep->wq, &wait);

		write_unlock_irq(&ep->lock);

		if (!eavail)
			timed_out = !schedule_hrtimeout_range(to, slack,
							      HRTIMER_MODE_ABS);
		__set_current_state(TASK_RUNNING);
```

- **`__add_wait_queue_exclusive(&ep->wq, ...)`**：多线程 `epoll_wait` 同一个 epfd，在 v6.12 上**默认就是 wake-one**——很多资料还在说"epoll_wait 多线程会惊群"，对现代内核已不成立（同一 epfd 场景）。真正的惊群残余在"多个独立 epfd 监视同一 fd"场景，见 17.6。
- **先置 TASK_INTERRUPTIBLE、锁内再查一次 eavail、然后才睡**——经典的"check-then-sleep"竞态消除三步曲。注释直说：两个链表短暂同时为空但事件实际 pending 的窗口存在，必须锁内终查。
- `ep_autoremove_wake_function`：唤醒即自动摘队。源码上方一大段注释（1941-1947、2020-2036 行）解释了为什么：若唤醒不摘队，事件可能被一个未及时跑起来的线程占着名额，造成唤醒丢失或延迟——**内核源码里就写着惊群与唤醒公平性的完整讨论，读 fs/eventpoll.c 比读任何博客都值**。

收割本体 `ep_send_events()` 里藏着 LT/ET 的全部真相：

【真实源码 v6.12 fs/eventpoll.c:1872-1901，逐字，有删节】

```c
		revents = ep_item_poll(epi, &pt, 1);
		if (!revents)
			continue;

		events = epoll_put_uevent(revents, epi->event.data, events);
		if (!events) {
			list_add(&epi->rdllink, &txlist);
			ep_pm_stay_awake(epi);
			if (!res)
				res = -EFAULT;
			break;
		}
		res++;
		if (epi->event.events & EPOLLONESHOT)
			epi->event.events &= EP_PRIVATE_BITS;
		else if (!(epi->event.events & EPOLLET)) {
			/*
			 * If this file has been added with Level
			 * Trigger mode, we need to insert back inside
			 * the ready list, so that the next call to
			 * epoll_wait() will check again the events
			 * availability. At this point, no one can insert
			 * into ep->rdllist besides us. The epoll_ctl()
			 * callers are locked out by
			 * ep_send_events() holding "mtx" and the
			 * poll callback will queue them in ep->ovflist.
			 */
			list_add_tail(&epi->rdllink, &ep->rdllist);
			ep_pm_stay_awake(epi);
		}
```

读懂这段，LT/ET 的一切玄学落地为机制：

- **`revents = ep_item_poll(epi, &pt, 1)`**：交付前**重新调用一次 `->poll()` 确认当前状态**。就绪链表只是"候选名单"——epitem 在链上不代表事件还成立（数据可能已被别的线程读走）。这就是为什么 epoll 也会**虚假唤醒**，`epoll_wait` 返回的事件在你 read 时可能已经 EAGAIN（尤其多线程/EPOLLEXCLUSIVE 下），健壮的程序必须容忍 read 返回 EAGAIN。
- **LT = `list_add_tail(&epi->rdllink, &ep->rdllist)`**：水平触发的全部实现就是这一行——交付完把 epitem **重新挂回就绪链表**。下次 epoll_wait 再来时重新 `ep_item_poll` 一次：状态还在（缓冲区还有数据）就再次上报，不在就静默移除。**LT 不是"内核盯着你的缓冲区"，而是"内核反复替你问一遍"**。
- **ET = 不挂回**。事件交付即消失，只有 `ep_poll_callback` 再次被触发（新数据到达产生新的状态变迁）才会重新入链。于是 ET 的经典死锁成立：epoll(7) man page 的 2kB 管道例子——写端写 2kB，读端收到通知只读 1kB，剩余 1kB 永远不会再产生通知，*"the next call to epoll_wait(2) ... will probably hang despite the available data still present"*（man page 原文）。
- **代价对比现在可以精确量化**：LT 多付的是"每轮对每个未读尽 fd 多一次 `ep_item_poll`+重挂"；ET 省掉这个，换来应用必须"读到 EAGAIN"的协议义务。**当绝大多数事件一次读尽时，两者性能差异趋近于零**——选 ET 的真实收益场景是高频小事件+深缓冲（避免同一 fd 反复出现在每轮返回集里）。
- `epoll_put_uevent` 失败时把 epitem 放回 txlist 并返回 -EFAULT——用户给了坏指针，事件也不丢。再次呼应 17.2.2 的设计纪律：通知机制的溢出/失败语义必须内核兜底。

### 17.5.6 失败模式与生产暗坑（源码级根因）

1. **"我明明 close 了，事件还来" / "EPOLL_CTL_DEL 报 EBADF"**：epoll 的键是 `{file*, fd}`（`ep_cmp_ffd`），注册关系挂在 **open file description** 上。`fork()`/`dup()` 之后 close 原 fd，file 引用计数未归零，epitem 仍然存活并继续产生事件（带着旧 user_data），而此时你已没有合法 fd 去 DEL 它。根因：自动清理发生在 `eventpoll_release_file()`——file 最后一个引用释放时。**规约：close 前手动 DEL；fork 后子进程立刻处理继承的 epfd（或 EPOLL_CLOEXEC）。** 排障入口：`/proc/<pid>/fdinfo/<epfd>` 能看到全部注册项（tfd/events/data），strace 看不到这种"幽灵注册"。
2. **ET + 多线程 + 同一 fd**：ET 通知合并特性意味着一次通知可能代表多条消息；两个线程先后在同一 fd 上 read 会交错撕裂消息流。规约是 `EPOLLET|EPOLLONESHOT` + 处理完 `EPOLL_CTL_MOD` 重新武装，或一个 fd 永远只属于一个线程（shared-nothing，nginx/envoy 的做法）。
3. **ET 饥饿**：一个 fd 上持续有数据（如对端狂写），"读到 EAGAIN"的纪律会让你在这个 fd 上循环到天荒地老，饿死同轮其他就绪 fd。规约：每 fd 每轮限额读 N 次，没读完自己记到应用层 ready list 下轮再处理——**注意这等于在用户态重新实现了 LT**。nginx 的事件循环就是这么处理的。
4. **epoll 嵌套**：epfd 可以监视 epfd（`poll_wait` 字段就是为此），但环路与深度由内核检查（`gen`/`loop_check_gen`、PATH_ARR_SIZE 路径数限制），EPOLL_CTL_ADD 时做 DFS——嵌套太深或扇出太大会拿到 ELOOP/ENOSPC。少有人用，但 libuv 在某些场景会踩到。
5. **`epoll_wait` 返回后事件已过期**：见 17.5.5 第一条。永远把 EAGAIN 当正常路径写。

### 17.5.7 复杂度账本（修正流传版本的错误）

```
操作                      select/poll         epoll
───────────────────────────────────────────────────────────────
建立监听（一次性）          —（无此概念）        O(log n) ctl + wait queue 挂载
每次等待调用               O(n) 拷入+扫描+挂载   O(1)（只看 rdllist）
每个就绪事件               O(n) 重扫定位        O(1) 回调入链 + O(ready) 交付
                                              （LT 未读尽 fd 每轮多一次 ->poll）
每事件内核内存             0（栈上）            ~160B（epitem+pwq，64 位）
唤醒携带信息               否（醒后重扫）        是（rdllist 即答案）
```

流传说法"epoll_wait O(1)、epoll 全面碾压"需要修正的点：**epoll 把每次调用的 O(n) 换成了每个 fd 一次性的注册成本 + 每个事件 O(1) 的回调成本**。当就绪密度极高（n 个 fd 每轮几乎全就绪）时，epoll 每轮交付仍是 O(n)，还多付回调、锁、重挂（LT）的常数，**相比 poll 的一把梭扫描并无优势甚至更差**——Libenzi 自己从未宣称 epoll 万能，他的措辞一直是"多连接、稀疏就绪"。

---

## 17.6 惊群考古：三次治理，三个层次

**问题定义**：N 个等待者守同一个事件源，事件到来全部被唤醒，只有一个抢到活，其余 N-1 次唤醒是纯浪费（两次上下文切换 + cache 污染 each）。

### 17.6.1 史前：accept 惊群（≈2.4 时代）

多进程 `accept()` 同一 listen socket 睡在 `sk_wq` 上，连接到来内核全员唤醒。治理：accept 路径改用 exclusive wait（`WQ_FLAG_EXCLUSIVE`，`__wake_up_common` 唤醒第一个 exclusive waiter 即停）。「待核：精确引入版本，2.4.x 时代，未能从一手来源核实具体小版本」。从此**裸 accept 不再惊群**——但事件循环时代大家不裸 accept 了，问题换壳重生。

### 17.6.2 epoll 时代的两种惊群与 EPOLLEXCLUSIVE（4.5, 2016）

形态 A：**多线程 epoll_wait 同一个 epfd**——现代内核默认 exclusive（17.5.5 源码 `__add_wait_queue_exclusive` 实证），wake-one，已非问题。

形态 B：**每 worker 一个独立 epfd，各自 ADD 同一个 listenfd**（nginx 的拓扑）。此时 listen socket 的 wait queue 上挂的是 N 个 epoll 回调节点，`wake_up()` 会逐个执行所有非 exclusive 节点的回调：N 个 epfd 全部把各自的 epitem 入链、N 个 worker 全部唤醒。`EPOLLEXCLUSIVE`（Jason Baron, Akamai, commit df0108c5da56, 合入 4.5）让 epoll 的 wait queue 节点也用 exclusive 模式挂载（17.5.3 源码里的分支），配合 `ep_poll_callback` 返回值裁决（17.5.4）实现 wake-one-ish。commit message 给了实测：

> 一个 listen socket、多 epfd 场景的负载耗时从 **860 秒降到 24 秒**（"This creates thundering herd type behavior" → EPOLLEXCLUSIVE）
> —— commit df0108c5da56（github.com/torvalds/linux，逐字要点）

35 倍。惊群不是理论洁癖，是真金白银的 CPU。

**EPOLLEXCLUSIVE 的边界**（容易答错的细节）：只在 `EPOLL_CTL_ADD` 时可设（MOD 会 EINVAL）；"只唤醒一个"没有硬保证（语义是"至少一个"）；被唤醒的 worker 若不把 accept 队列掏干，事件不会自动转给别人（LT 重挂依然有效，但唤醒时机交给了下次事件）。

### 17.6.3 根治：SO_REUSEPORT（3.9, 2013, Tom Herbert）

换一个问题问法：与其治理"N 个等待者抢一个队列"，为什么不直接给每个 worker 一个独立队列？`SO_REUSEPORT` 允许多个 socket bind 同一 ip:port，内核在 TCP 层按四元组 hash 把新连接分发到组内某一个 socket。Tom Herbert 的 commit message（da5e36308d9f）把动机写得很完整：

> 动机场景是 web server 多线程各持 listener socket；此前两条路都不好——单 listener 线程分发是瓶颈；多线程 accept 同一 socket 则**负载不均，实测倾斜可达 3:1**；SO_REUSEPORT 之后 *"the distribution is uniform"*。
> —— commit da5e36308d9f（github.com/torvalds/linux，逐字要点）

注意他给的第二个理由：惊群之外，**共享 accept 队列本身分发不均**（唤醒顺序与调度的耦合导致某些线程吃撑、某些挨饿）。SO_REUSEPORT 一刀解决两个问题：没有共享队列 → 没有争抢 → 没有惊群也没有倾斜。

后续演化：4.5/4.6 加 `SO_ATTACH_REUSEPORT_EBPF`（eBPF 自定义选 socket，UDP 先行、TCP 跟进）+ reuseport 组的分组查找优化。**面试高频错答：把 SO_REUSEPORT 本体说成 4.5——本体是 3.9，4.5 加的是 eBPF 选择器。**

**SO_REUSEPORT 的代价**（不付钱的方案不存在）：

1. worker 崩溃/退出时，它的 listen socket **accept 队列里已三次握手完成但未 accept 的连接直接丢弃**（客户端表现为连接被 RST）。滚动重启需要先摘流量或配合 eBPF 选择器做 drain。
2. 四元组 hash 分发对长连接负载是均匀的，对**连接数少但每连接代价高度不均**的场景无能为力（hash 均匀 ≠ 负载均匀）。
3. nginx 1.9.1 起 `listen ... reuseport` 可用；开启后 accept_mutex 语义失效（本来就是软件层惊群补丁，可退役）。

### 17.6.4 三种拓扑对照推演

场景：4 worker，1000 新连接/秒，看每秒浪费的唤醒次数：

```
拓扑                            惊群行为                  每秒无效唤醒（量级）
────────────────────────────────────────────────────────────────────────
A. 1 epfd 共享，4 线程 wait      默认 exclusive，wake-one    ~0（但 accept 后处理在哪个核不可控）
B. 4 epfd 各 ADD 同一 listenfd   无 EPOLLEXCLUSIVE：全员醒    3000（=1000×3）
                                加 EPOLLEXCLUSIVE：~0
C. SO_REUSEPORT 4 socket        无共享 → 无惊群             0（且 cache 亲和最好）
```

为什么 C 还优于 B+EPOLLEXCLUSIVE：B 中哪个 worker 被唤醒由 wait queue 顺序决定，连接在 worker 间的分布不保证均匀（Herbert 的 3:1 问题仍部分存在）；C 中连接从 SYN 起就属于确定的 socket/worker/核，配合 RSS/RPS 把软中断也钉在同核，**从网卡队列到应用线程的整条路径 cache-local**。这就是 C10M 路线的雏形（17.13）。

---

## 17.7 Linux AIO 之死（2002–2019）：一个 API 的完整生命周期

### 17.7.1 出生即残疾

Linux native AIO（`io_setup/io_submit/io_getevents`，2.5 开发周期合入，主推者 Benjamin LaHaise，金主是数据库厂商）从第一天起就只对 **O_DIRECT** 真异步。LWN 2016 年的总结性清单（lwn.net/Articles/671649，转述）：

- 复杂的子系统，每种 IO 目标都要显式支持，覆盖面长期残缺（fsync 都没做全）；
- **buffered IO 不支持**；O_DIRECT 路径在特定条件下照样阻塞（元数据未缓存、请求槽满、文件系统持锁路径）；
- API 设计差到 **glibc 拒绝封装**（应用得自己拼 syscall 或用 libaio）；
- 操作类型扩展极难。

最致命的是第二条的隐蔽形态：`io_submit` 这个"提交"动作本身可能同步阻塞数百毫秒（fs 元数据 IO、块层拥塞）。**应用以为自己在异步提交，实际整个事件循环被卡死**——这种"语义上承诺异步、实现上偶发同步"的 API 比纯同步 API 更危险，因为它骗过了架构设计阶段的所有评审。

### 17.7.2 为什么修不好：completion 模型与内核现实的冲突

根因要从执行模型说起。completion-based API 的隐含契约：**任何提交路径都不能阻塞调用者，等待必须发生在"别处"**。两种实现路线：

1. **状态机化**：把整条 IO 路径改写成可挂起/重续的状态机（每个可能睡眠的点都变成"注册回调后返回"）。Linux 的同步路径（page cache 锁、fs 日志、块层）深达几十层调用，全面状态机化等于重写半个 VFS。Suparna Bhattacharya（IBM）2003 年前后的 buffered AIO retry 补丁走的就是这条路，最终没能合入——侵入面太大。
2. **线程卸载**：提交即转交给内核线程同步执行。简单通用，但 AIO 社区长期视之为"假异步"，认为线程开销违背 AIO 的初衷。

2007 年还有过第三波尝试：Zach Brown 的 fibrils、Ingo Molnar 的 syslets/threadlets（内核轻量执行流，可睡眠时自动切换）——概念惊艳，复杂度失控，全部死亡。

2016 年 Benjamin LaHaise 再次尝试用线程模型扩展 AIO（支持 openat、fsync 等），Linus 的回应成为内核史名场面：

> *"If you want to do arbitrary asynchronous system calls, just \*do\* it. But do _that_, not 'let's extend this horrible interface in arbitrary random ways one special system call at a time'."*
> *"In other words, why is the interface not simply: 'do arbitrary system call X with arguments A, B, C, D asynchronously using a kernel thread'."*
> —— Linus Torvalds，2016-01（LWN Articles/671649，逐字）

同场讨论里 Dave Chinner 指出其 fsync 实现没解决真问题，Andy Lutomirski 提出异步系统调用与进程退出语义的安全难题（"exit is bad"）。修补路线在 2016 年事实死亡。

**Linus 那句话几乎就是 io_uring 的需求文档**：通用（任意操作而非逐个 opcode 打补丁）、异步执行机制对用户透明（内核线程与否是实现细节）。三年后 Jens Axboe 交卷。

### 17.7.3 设计教训（值得抽象出来的部分）

1. **"纯粹性"杀死了 AIO**：拒绝线程卸载追求"真异步"，结果 17 年只覆盖 O_DIRECT。io_uring 的答案是**混合执行**：先 inline 非阻塞试一次（大多数 page cache 命中、socket 有数据的场景零线程开销），不行再丢 io-wq worker——线程是 fallback 不是模型。工程上"不纯但全覆盖"碾压"纯但残缺"。
2. **API 的失败会传染**：glibc 拒绝封装 → 应用接入成本高 → 用户少 → 投入修复的动力小 → 更残缺。生态死亡螺旋。io_uring 从第一天就配 liburing（Axboe 本人维护），把 ring 操作、屏障、封装全部代办——**内核 API 的成败一半在用户态库**。

---

## 17.8 设计考古 II：io_uring 的崛起（2019–）

### 17.8.1 从 polled AIO 到推倒重来

直接史料链：Jens Axboe（块层 maintainer、fio 作者）2018 年先做了十余版 **polled AIO** 补丁（给 aio 加 NVMe 轮询完成），越改越发现 aio 的 ABI 没救——`io_getevents` 的拷贝、上下文切换、per-IO 两次 syscall 在百万 IOPS 的 NVMe 面前全是大头。2019-01 他放弃修补，提交全新接口 io_uring。LWN 第一时间报道（Corbet, 2019-01-15）：

> *"io_uring introduces just what the kernel needed more than anything else: yet another ring buffer."*（Corbet 的招牌嘲讽）
> Axboe 自陈动机：*"buffered I/O has always been a bit of a sore spot for Linux AIO."*
> —— lwn.net/Articles/776703（逐字）

合入异常顺利：初始 commit `2b188cc1bb85`（"Add io_uring IO interface"，Reviewed-by: Hannes Reinecke），随 5.1 发布（2019-05）。commit message 要点（github 核实）：SQ/CQ 两个 ring 应用与内核共享消除拷贝；两个新 syscall `io_uring_setup` + `io_uring_enter`；**每个 io_uring 背后有 workqueue 兜底 buffered IO，page cache 命中则 inline 完成**——17.7.3 的两条教训在出生证明上就写着。

为什么 AIO 修不动而 io_uring 三个月合入？因为它绕开了所有历史包袱：不动 aio ABI、不重写同步路径、不发明内核执行流原语，只是**把"共享内存队列 + 混合执行"这两个各自成熟的想法第一次拼对了**。

性能上 Axboe 在 cover letter 报过 polled io_uring 单核 ~1.6–1.7M IOPS vs libaio ~600K 的对比数字「待核：lore.kernel.org 反爬未取到原文，数字凭记忆标注，量级与后续公开 talk 一致」。

### 17.8.2 opcode 大爆炸与"第二个系统调用界面"之争

LWN 2020-01 的追踪报道（*The rapid growth of io_uring*, lwn.net/Articles/810414，逐字要点）给出逐版本时间线：

```
5.1 (2019-05)  NOP, READV/WRITEV, READ_FIXED/WRITE_FIXED, FSYNC, POLL_ADD/REMOVE
5.2            SYNC_FILE_RANGE
5.3            SENDMSG / RECVMSG          ← 网络 IO 进场
5.4            TIMEOUT
5.5            TIMEOUT_REMOVE, ACCEPT, CONNECT, ASYNC_CANCEL, LINK_TIMEOUT
5.6            FALLOCATE, OPENAT(2), CLOSE, FILES_UPDATE, STATX, READ/WRITE,
               FADVISE/MADVISE, SEND/RECV, EPOLL_CTL（io_uring 可以操作 epoll！）
```

到 5.6 已 30 个 opcode。LWN 点出的结构性担忧：opcode 字段 8 位、上限 256，而 Linux syscall 超过 400——**io_uring 正在长成一个平行的、异步的系统调用界面**，且这个界面绕过了既有的 seccomp/审计/可观测性体系（这颗雷在 17.10 爆炸）。Axboe 当时的进一步规划是用 BPF 串联操作链（让前一个 op 的输出喂给后一个），受 unprivileged BPF 被禁的大环境拖累未成主流。

网络侧的成熟标志在 5.19–6.0（man pages / liburing wiki 口径核实）：

```
5.19 (2022)   multishot accept（一次提交持续收连接）、provided buffer ring、
              uring_cmd（NVMe passthrough 配套 SQE128/CQE32）、IORING_CQE_F_SOCK_NONEMPTY
6.0  (2022)   multishot recv、IORING_OP_SEND_ZC（零拷贝发送）
6.1            IORING_SETUP_DEFER_TASKRUN（完成处理推迟到 GETEVENTS 时批量跑）
6.6            kernel.io_uring_disabled sysctl「待核：版本号未从一手来源核实」、
              IORING_SETUP_NO_SQARRAY（v6.12 头文件已有，见 17.9.4）
6.15 前后      零拷贝接收 zcrx「待核」
```

multishot 系列值得单独理解：它把 io_uring 从"异步提交每个操作"升级为"**订阅模式**"——一次 SQE，内核持续投递 CQE（`IORING_CQE_F_MORE` 标志表示订阅仍活着）。accept/recv/poll 都有 multishot 版后，io_uring 在网络场景的语义已经覆盖 epoll：**epoll 是"就绪订阅"，multishot recv + buffer ring 是"数据订阅"——连 read 都替你做了**。

---

## 17.9 io_uring 实现精读（v6.12）

### 17.9.1 内存布局：三块 mmap 与两个 SPSC ring

```
io_uring_setup(entries, &params) → uringfd，随后 mmap 三段（偏移是 ABI 常量）：

  IORING_OFF_SQ_RING (0)         SQ ring 元数据：head/tail/mask/flags/dropped + sq_array[]
  IORING_OFF_CQ_RING (0x8000000) CQ ring 元数据 + cqes[]（CQE 直接内嵌）
  IORING_OFF_SQES    (0x10000000) SQE 数组本体（64B × entries）

           用户态（生产者）                     内核（消费者）
  SQ:  写 sqes[i] → sq_array[tail&mask]=i     读 head..tail 的 SQE
       → smp_store_release(tail)              → 消费后 store_release(head)
  CQ:  读 head..tail 的 CQE                    写 cqes[tail&mask]
       → 消费后 store_release(head)            → smp_store_release(tail)
```

（5.4 起 `IORING_FEAT_SINGLE_MMAP`：SQ/CQ ring 合并一次 mmap；6.x 的 `IORING_SETUP_NO_MMAP` 反向支持由应用提供内存。）

**为什么 SQE 数组与 SQ ring 分离、中间隔一层 `sq_array` 索引？** 初始设计允许应用把 SQE 池当槽位分配器用：固定槽位预填模板（比如槽 0-31 永远是某类读请求），提交时只把槽号丢进 ring，乱序复用。实践证明几乎所有应用都顺序使用，这层间接每次提交多吃一次依赖加载（潜在 cache miss）。于是 v6.6 前后加 `IORING_SETUP_NO_SQARRAY`——uapi 头文件注释只有一句话：*"Removes indirection through the SQ index array."*（v6.12 头文件逐字）。**CQ 从来没有这层间接**：CQE 由内核生产、天然顺序写，应用只读——不对称是因为两个方向的生产者不同。一个 ABI 设计的灵活性预留，十几年后被数据证伪然后退场，这个完整闭环很有教学价值：**为想象中的用法付出的每条指令，最终都会被 perf 找出来清算。**

### 17.9.2 共享数据结构（内核侧定义）

【真实源码 v6.12 include/linux/io_uring_types.h:103-184，逐字，有删节】

```c
struct io_uring {
	u32 head;
	u32 tail;
};

/*
 * This data is shared with the application through the mmap at offsets
 * IORING_OFF_SQ_RING and IORING_OFF_CQ_RING.
 *
 * The offsets to the member fields are published through struct
 * io_sqring_offsets when calling io_uring_setup.
 */
struct io_rings {
	/*
	 * Head and tail offsets into the ring; the offsets need to be
	 * masked to get valid indices.
	 *
	 * The kernel controls head of the sq ring and the tail of the cq ring,
	 * and the application controls tail of the sq ring and the head of the
	 * cq ring.
	 */
	struct io_uring		sq, cq;
	/*
	 * Bitmasks to apply to head and tail offsets (constant, equals
	 * ring_entries - 1)
	 */
	u32			sq_ring_mask, cq_ring_mask;
	/* Ring sizes (constant, power of 2) */
	u32			sq_ring_entries, cq_ring_entries;
	/*
	 * Number of invalid entries dropped by the kernel due to
	 * invalid index stored in array
	 */
	u32			sq_dropped;
	/*
	 * Runtime SQ flags
	 *
	 * The application needs a full memory barrier before checking
	 * for IORING_SQ_NEED_WAKEUP after updating the sq tail.
	 */
	atomic_t		sq_flags;
	/* ... 中略：cq_flags ... */
	/*
	 * Number of completion events lost because the queue was full;
	 * this should be avoided by the application by making sure
	 * there are not more requests pending than there is space in
	 * the completion queue.
	 */
	u32			cq_overflow;
	/*
	 * Ring buffer of completion events.
	 *
	 * The kernel writes completion events fresh every time they are
	 * produced, so the application is allowed to modify pending
	 * entries.
	 */
	struct io_uring_cqe	cqes[] ____cacheline_aligned_in_smp;
};
```

- **所有权矩阵**（内核注释原话）：SQ 的 head 归内核、tail 归应用；CQ 的 tail 归内核、head 归应用。**每个 u32 永远只有一个写者**——SPSC 的精髓，这是 17.9.3 内存序能如此便宜的前提。
- ring 大小是 2 的幂，`mask = entries-1`，下标 = `counter & mask`：head/tail 是**永不回绕的自由计数器**，溢出由无符号回绕自然处理；空 = (head==tail)，满 = (tail-head==entries)。不需要"留一个空位"的传统环形缓冲技巧，也不需要单独的 count 字段——**计数器即状态**。
- `cqes[]` 带 `____cacheline_aligned_in_smp`：CQE 区与前面的元数据隔 cacheline，内核狂写 CQE 不会把应用正在自旋读的 head/tail 行打脏（false sharing 治理）。
- `sq_dropped`/`cq_overflow`：两个方向的异常账本都在共享内存里，应用零成本可见。

### 17.9.3 内存序协议：文件头注释就是规范

【真实源码 v6.12 io_uring/io_uring.c:1-41，逐字，有删节】

```c
/*
 * Shared application/kernel submission and completion ring pairs, for
 * supporting fast/efficient IO.
 *
 * A note on the read/write ordering memory barriers that are matched between
 * the application and kernel side.
 *
 * After the application reads the CQ ring tail, it must use an
 * appropriate smp_rmb() to pair with the smp_wmb() the kernel uses
 * before writing the tail (using smp_load_acquire to read the tail will
 * do). It also needs a smp_mb() before updating CQ head (ordering the
 * entry load(s) with the head store), pairing with an implicit barrier
 * through a control-dependency in io_get_cqe (smp_store_release to
 * store head will do). Failure to do so could lead to reading invalid
 * CQ entries.
 *
 * Likewise, the application must use an appropriate smp_wmb() before
 * writing the SQ tail (ordering SQ entry stores with the tail store),
 * which pairs with smp_load_acquire in io_get_sqring (smp_store_release
 * to store the tail will do). And it needs a barrier ordering the SQ
 * head load before writing new SQ entries (smp_load_acquire to read
 * head will do).
 *
 * When using the SQ poll thread (IORING_SETUP_SQPOLL), the application
 * needs to check the SQ flags for IORING_SQ_NEED_WAKEUP *after*
 * updating the SQ tail; a full memory barrier smp_mb() is needed
 * between.
 *
 * io_uring also uses READ/WRITE_ONCE() for _any_ store or load that happens
 * from data shared between the kernel and application.
 *
 * Copyright (C) 2018-2019 Jens Axboe
 * Copyright (c) 2018-2019 Christoph Hellwig
 */
```

把它翻译成可执行的心智模型：

1. **发布-订阅对**：生产者"先写数据、后 release 写 tail"，消费者"acquire 读 tail、后读数据"。acquire/release 在 x86 上几乎免费（TSO 下 release store/acquire load 就是普通 mov），在 ARM 上是 `stlr/ldar`——**SPSC ring 是为弱内存序硬件设计的最便宜协议**。为什么够：每个变量单写者 → 不存在写-写竞争 → 不需要 RMW 原子操作和全序。
2. **消费侧也要 release 写 head**：直觉上"我只是标记消费完了"，但 head 一前移，生产者就有权覆写那个槽——你对槽的最后一次 load 必须先于 head store 全局可见，否则读到被覆写的数据。"释放槽位"本质也是一次发布。
3. **第三条是唯一需要 full barrier 的地方**（SQPOLL 唤醒协议），单独看 17.9.5——store-then-load 跨变量顺序，acquire/release 罩不住，这是 Dekker/Peterson 的经典形状。
4. **READ_ONCE/WRITE_ONCE 无处不在**：共享内存对面是不可信且并发活动的用户态，编译器合并/重读/撕裂访问都可能变成安全洞（17.9.4）。

### 17.9.4 提交路径：防御性编程的范本

【真实源码 v6.12 io_uring/io_uring.c:2241-2293，逐字（含函数前注释）】

```c
static void io_commit_sqring(struct io_ring_ctx *ctx)
{
	struct io_rings *rings = ctx->rings;

	/*
	 * Ensure any loads from the SQEs are done at this point,
	 * since once we write the new head, the application could
	 * write new data to them.
	 */
	smp_store_release(&rings->sq.head, ctx->cached_sq_head);
}

/*
 * Fetch an sqe, if one is available. Note this returns a pointer to memory
 * that is mapped by userspace. This means that care needs to be taken to
 * ensure that reads are stable, as we cannot rely on userspace always
 * being a good citizen. If members of the sqe are validated and then later
 * used, it's important that those reads are done through READ_ONCE() to
 * prevent a re-load down the line.
 */
static bool io_get_sqe(struct io_ring_ctx *ctx, const struct io_uring_sqe **sqe)
{
	unsigned mask = ctx->sq_entries - 1;
	unsigned head = ctx->cached_sq_head++ & mask;

	if (!(ctx->flags & IORING_SETUP_NO_SQARRAY)) {
		head = READ_ONCE(ctx->sq_array[head]);
		if (unlikely(head >= ctx->sq_entries)) {
			/* drop invalid entries */
			spin_lock(&ctx->completion_lock);
			ctx->cq_extra--;
			spin_unlock(&ctx->completion_lock);
			WRITE_ONCE(ctx->rings->sq_dropped,
				   READ_ONCE(ctx->rings->sq_dropped) + 1);
			return false;
		}
	}

	/*
	 * The cached sq head (or cq tail) serves two purposes:
	 *
	 * 1) allows us to batch the cost of updating the user visible
	 *    head updates.
	 * 2) allows the kernel side to track the head on its own, even
	 *    though the application is the one updating it.
	 */

	/* double index for 128-byte SQEs, twice as long */
	if (ctx->flags & IORING_SETUP_SQE128)
		head <<= 1;
	*sqe = &ctx->sq_sqes[head];
	return true;
}
```

安全视角逐行：

- **函数前注释是 io_uring 威胁模型的官方表述**：*"we cannot rely on userspace always being a good citizen"*。SQE 活在用户态可写内存里，恶意程序可以在内核校验后、使用前改写字段。规则：每字段 `READ_ONCE` 取一次进内核栈、校验、之后**只用栈上副本**。违反 = double-fetch/TOCTOU 漏洞（先用合法 opcode 过校验，再改成越界参数）。这是把"共享内存当 ABI"必须支付的纪律税——对比传统 syscall：`copy_from_user` 天然做了快照，而 io_uring 为了省这次拷贝，把快照纪律分摊进了每一行代码。
- `head = READ_ONCE(ctx->sq_array[head])` + 越界检查：**间接索引来自用户态，永远当敌意输入**。无效就计入 `sq_dropped` 丢弃——不 panic、不 -EINVAL 整批失败，单条静默丢弃 + 账本可见，保持 ring 的流水不断。
- `cached_sq_head`：内核自己记消费位置，**批量消费完才一次 `smp_store_release` 发布**（io_commit_sqring）——共享 cacheline 的写次数从 N 降到 1。注释里"内核侧自主跟踪 head"还有一层安全意义：即使应用乱写共享内存里的 head，也只是骗自己，内核进度不受影响。
- `io_commit_sqring` 的注释精确解释了 release 的方向性：head 一发布，应用即可覆写槽位，所以**对 SQE 的全部 load 必须排在 head store 之前**——release 保证的恰好就是"之前的访存不越过它往后跑"。

外层批量循环：

【真实源码 v6.12 io_uring/io_uring.c:2295-2343，逐字，有删节】

```c
int io_submit_sqes(struct io_ring_ctx *ctx, unsigned int nr)
	__must_hold(&ctx->uring_lock)
{
	unsigned int entries = io_sqring_entries(ctx);
	unsigned int left;
	int ret;

	if (unlikely(!entries))
		return 0;
	/* make sure SQ entry isn't read before tail */
	ret = left = min(nr, entries);
	io_get_task_refs(left);
	io_submit_state_start(&ctx->submit_state, left);

	do {
		const struct io_uring_sqe *sqe;
		struct io_kiocb *req;

		if (unlikely(!io_alloc_req(ctx, &req)))
			break;
		if (unlikely(!io_get_sqe(ctx, &sqe))) {
			io_req_add_to_cache(req, ctx);
			break;
		}

		/*
		 * Continue submitting even for sqe failure if the
		 * ring was setup with IORING_SETUP_SUBMIT_ALL
		 */
		if (unlikely(io_submit_sqe(ctx, req, sqe)) &&
		    !(ctx->flags & IORING_SETUP_SUBMIT_ALL)) {
			left--;
			break;
		}
	} while (--left);
	/* ... 中略：未提交配额归还 ... */
	io_submit_state_end(ctx);
	 /* Commit SQ ring head once we've consumed and submitted all SQEs */
	io_commit_sqring(ctx);
	return ret;
}
```

- `io_sqring_entries` 内部是 `smp_load_acquire(tail)`（与应用的 release-tail 配对），注释"make sure SQ entry isn't read before tail"就是 17.9.3 协议第二条的落地。
- 一次 `io_uring_enter` 里这个 do-while 把全部 pending SQE 转成内核请求对象（`io_kiocb`，对象池分配）、逐条派发，最后一次性 commit——**N 个 IO 的固定成本（syscall 进出、屏障、task ref）被 1/N 摊薄**。这就是"批量"在代码上的样子。
- `io_submit_sqe` 内部走"先 inline 非阻塞尝试，需要等待则注册 poll 驱动（FEAT_FAST_POLL）或丢 io-wq"，buffered/socket IO 的全异步由此达成（17.7.3 的混合执行）。

### 17.9.5 SQPOLL：零 syscall 模式与丢失唤醒的攻防

`IORING_SETUP_SQPOLL` 启动内核线程（`iou-sqp-<pid>`，可 `sq_thread_cpu` 钉核）持续轮询 SQ——应用只写共享内存，**稳态下 IO 路径零 syscall**。代价：一个核被烧掉（idle 自旋），所以有 `sq_thread_idle` 超时：无活可干超过该时长，线程要睡。**"轮询者要睡觉"立刻制造一个经典竞态**：线程刚决定睡、应用恰好写入新 SQE 且没人通知——请求永远没人消费。

看内核怎么解（睡前协议）：

【真实源码 v6.12 io_uring/sqpoll.c:341-375，逐字，有删节】

```c
		prepare_to_wait(&sqd->wait, &wait, TASK_INTERRUPTIBLE);
		if (!io_sqd_events_pending(sqd) && !io_sq_tw_pending(retry_list)) {
			bool needs_sched = true;

			list_for_each_entry(ctx, &sqd->ctx_list, sqd_list) {
				atomic_or(IORING_SQ_NEED_WAKEUP,
						&ctx->rings->sq_flags);
				if ((ctx->flags & IORING_SETUP_IOPOLL) &&
				    !wq_list_empty(&ctx->iopoll_list)) {
					needs_sched = false;
					break;
				}

				/*
				 * Ensure the store of the wakeup flag is not
				 * reordered with the load of the SQ tail
				 */
				smp_mb__after_atomic();

				if (io_sqring_entries(ctx)) {
					needs_sched = false;
					break;
				}
			}

			if (needs_sched) {
				mutex_unlock(&sqd->lock);
				schedule();
				mutex_lock(&sqd->lock);
				sqd->sq_cpu = raw_smp_processor_id();
			}
			list_for_each_entry(ctx, &sqd->ctx_list, sqd_list)
				atomic_andnot(IORING_SQ_NEED_WAKEUP,
						&ctx->rings->sq_flags);
		}
```

竞态推演（两侧各三步，Dekker 形状）：

```
SQ 线程（睡前）                      应用（提交时）
S1: 置 NEED_WAKEUP 标志              A1: 写 SQE、release 写 sq tail
S2: smp_mb__after_atomic()           A2: smp_mb()   ←(io_rings 注释明示应用义务)
S3: 读 sq tail，空才 schedule()       A3: 读 NEED_WAKEUP，置位则 io_uring_enter(SQ_WAKEUP)
```

为什么两边都要 **full barrier**：这是"我先写自己的变量，再读对方的变量"的对称结构。仅有 release/acquire 时，S3 的 load 可以被 CPU 提前到 S1 的 store 之前全局可见（store-load 重排，x86 TSO 也允许这一种！），两边同时"看旧值"：线程看到 tail 空（A1 还没传播过来）安心睡，应用看到 NEED_WAKEUP 未置位（S1 还没传播过来）不调 enter——**双方各自有理，请求永久搁浅**。smp_mb 强制 store 先于后续 load 全局可见后，可以穷举证明至少一方能看到另一方的写：要么线程读到非空 tail 不睡，要么应用读到标志去 wakeup。**这是教科书之外、生产 ABI 里活着的 Dekker 协议**，io_rings 结构注释（17.9.2）专门提醒应用侧义务，liburing 替你做对了——自己裸写 ring 的人十有八九死在这。

`atomic_andnot` 清标志放在醒来之后：宁可多收一次冗余 wakeup syscall，不可丢失一次唤醒——**唤醒协议的不变式永远是"至多冗余、绝不丢失"**。

### 17.9.6 SQE/CQE：ABI 的字节经济学

【真实源码 v6.12 include/uapi/linux/io_uring.h:30-77,423-433，逐字，有删节】

```c
struct io_uring_sqe {
	__u8	opcode;		/* type of operation for this sqe */
	__u8	flags;		/* IOSQE_ flags */
	__u16	ioprio;		/* ioprio for the request */
	__s32	fd;		/* file descriptor to do IO on */
	union {
		__u64	off;	/* offset into file */
		__u64	addr2;
		struct {
			__u32	cmd_op;
			__u32	__pad1;
		};
	};
	union {
		__u64	addr;	/* pointer to buffer or iovecs */
		__u64	splice_off_in;
		/* ... 中略 ... */
	};
	__u32	len;		/* buffer size or number of iovecs */
	union {
		__kernel_rwf_t	rw_flags;
		__u32		fsync_flags;
		__u32		accept_flags;
		/* ... 中略：20+ 个按 opcode 复用的 flags 槽 ... */
	};
	__u64	user_data;	/* data to be passed back at completion time */
	/* ... 中略：buf_index/buf_group、personality、splice_fd_in/file_index、
	        addr3 / SQE128 时的 80 字节 cmd[] 尾巴 ... */
};

struct io_uring_cqe {
	__u64	user_data;	/* sqe->user_data value passed back */
	__s32	res;		/* result code for this event */
	__u32	flags;

	/*
	 * If the ring is initialized with IORING_SETUP_CQE32, then this field
	 * contains 16-bytes of padding, doubling the size of the CQE.
	 */
	__u64 big_cqe[];
};
```

- **SQE = 64 字节 = 一条 cacheline**。这不是巧合是设计目标：填一个请求最多脏一行，内核读一个请求最多 miss 一次。union 嵌 union 的"丑"换来几十种 opcode 共享同一布局——`off/addr/len` 三件套覆盖 90% 操作，flags 槽按 opcode 重释义。版本演化时只能往 union 里加新解释、往保留位下手，**uapi 一旦发布即是合同**：`__pad1`、`resv` 字段就是为未来留的逃生门。NVMe/网络命令直通需要更大负载时不破坏布局，而是整体倍增（SQE128/CQE32，5.19 配合 uring_cmd）。
- **CQE = 16 字节**：`user_data`（请求关联，8 字节透传）+ `res`（= 同步世界的返回值/-errno）+ `flags`。一条 cacheline 装 4 个完成事件，高 IOPS 收割时 cache 行为极佳。对比 epoll：`epoll_event` 12 字节但每次要 syscall + copy_to_user；对比 AIO：`io_event` 32 字节 + io_getevents syscall。**CQE 是三代接口里每事件字节成本与每事件 syscall 成本同时最低的。**
- `user_data` 的地位被低估：它是异步世界的"调用栈替身"——同步编程里"哪个调用返回了"由栈帧天然回答，异步世界必须显式编码（通常放请求对象指针或 {conn_id, op_type} 编码）。**所有 completion 模型的应用复杂度都收敛到 user_data 的设计纪律上。**

### 17.9.7 注册机制：把"每次"变成"一次"

io_uring 的性能哲学一以贯之——凡是每次 IO 都要做的固定动作，都提供"预注册"把它挪到 setup 阶段做一次：

```
每 IO 固定开销                       注册机制（io_uring_register）
─────────────────────────────────────────────────────────────────
fdget/fdput（原子 refcount，         IORING_REGISTER_FILES → sqe->fd 填
 多线程下 cacheline 弹跳）             fixed file 槽号 + IOSQE_FIXED_FILE
get_user_pages / iovec 遍历校验      IORING_REGISTER_BUFFERS → READ_FIXED/
（每次 pin/unpin 用户页）              WRITE_FIXED 直用预 pin 的页
recv 前不知道数据多大、buffer 难复用   provided buffer ring（5.19）：应用预投
                                     一池 buffer，内核完成时挑一个，CQE 带回 buf id
io_uring_enter 本身                  SQPOLL（17.9.5）
完成时的 IPI/中断打断用户态           COOP_TASKRUN/DEFER_TASKRUN（6.1）：完成
                                     处理攒到应用下次主动进内核时批量跑
```

每一行都是"识别一类固定成本 → 提供摊销机制"的重复。**读任何高性能系统的设计，先列它的 per-operation 固定成本清单，再看它提供了哪些摊销手段，框架就清楚了**——这个方法论对 DPDK（UIO+大页+轮询）、对 JIT（inline cache）同样适用。

---

## 17.10 安全争议：强大与危险同源

### 17.10.1 一手数据：Google kCTF 的账单

Google 2023-06 复盘 kCTF VRP 收到的 42 个内核 exploit，随后在 oss-security 邮件列表补充了决策细节（Tamás Koczka，2023-07-19，seclists.org/oss-sec/2023/q3/47，逐字要点）：

- **60% 的提交打的是 io_uring**；总赏金 180 万美元中**约 100 万付给了 io_uring 漏洞**；
- 处置：kernelCTF 环境**禁用 io_uring 和 nftables**；**ChromeOS 禁用**（*"while we explore new ways to sandbox it"*）；**Google 生产服务器已禁用**；**Android** 用 seccomp-bpf 挡住 app，后续 SELinux 策略把 io_uring 限制到 fastbootd 和 snapuserd 两个组件；GKE AutoPilot 评估默认关闭；
- 定性原话：*"it is still affected by severe vulnerabilities and also provides strong exploitation primitives"*，结论 *"we currently consider it safe only for use by trusted components"*。

### 17.10.2 为什么 io_uring 是 exploit 富矿（结构性分析）

1. **攻击面 = 半个内核的异步重述**：60+ opcode 意味着 read/write/socket/fs/xattr/futex 的内核路径都多了一个异步入口，每条路径的锁与生命周期假设都要在"可取消、可链接、可能在 io-wq 线程跑"的新前提下重新成立。同样的代码量，状态空间是同步 syscall 的多倍。
2. **对象生命周期是 UAF 的温床**：请求可被 cancel、可被 link 触发、可在 poll 回调/task_work/io-wq 三种上下文完成——引用计数错一处就是 use-after-free，而 UAF 配上 io_uring 自带的内核内存喷射能力（注册 buffer、msg_ring）即是完整利用链。"strong exploitation primitives" 指的就是这个：**io_uring 不仅 bug 多，还顺手提供了利用 bug 的工具箱**。
3. **共享内存 ABI 的 double-fetch 面**：17.9.4 的 