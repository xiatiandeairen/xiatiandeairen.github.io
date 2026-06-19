---
title: "（深化版）· 内核内存分配 slab / slub"
slug: "9-09"
collection: "tech-library"
group: "linux系统"
order: 9009
summary: "一句话定位：buddy allocator 是内核的\"批发商\"（以页为单位），slab/slub 是内核的\"零售商\"（以字节级对象为单位）。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T07:06:47.000Z"
updatedAt: "2026-06-12T07:06:47.000Z"
---
> **一句话定位**：buddy allocator 是内核的"批发商"（以页为单位），slab/slub 是内核的"零售商"（以字节级对象为单位）。本章不停在"分工"层面，而是把 SLUB 的 lockless fast path 拆到 cmpxchg 指令级、把 `struct slab` 从 `struct page` 拆出来的设计考古讲到 commit、把 GFP 标志的语义钉死到 `gfp_types.h` 的 bit 定义。

---

## TL;DR — 本章最硬结论（每条都能在源码 / 史料里落地）

1. **SLUB fast path 是一条 lockless cmpxchg，不是"加锁取对象"**：分配热路径 `__slab_alloc_node()` 不关中断、不禁抢占、不禁迁移，靠一个 per-cpu `freelist + tid` 的 **double-word `this_cpu_cmpxchg`**（`freelist_aba_t`）来探测"中途是否被抢占/迁移到别的 CPU"。理解这条路径是理解 SLUB 全部性能优势的钥匙。【真实源码 v6.12 `mm/slub.c`，见 §9.3.3】
2. **SLUB 不是 SLAB 的平替，是 2007 年的架构重写，动机是"队列吃内存"**：Christoph Lameter 的核心抱怨不是"SLAB 慢"，而是 SLAB 的 per-node / per-cpu / alien queue 在大 NUMA 机器上"光存引用就吃掉几个 GB"。这是 LWN 229984（2007-04-11）白纸黑字的设计理由。【史料已核实，见 §9.2】
3. **`struct slab` 在 5.17 从 `struct page` 里拆出来，是 folio 大改造的一环**：commit `d122019bf061`（Matthew Wilcox / Oracle，2021）"mm: Split slab into its own type"。拆分的根因是 `struct page` 已变成"一堆 #ifdef 和 union 的烂摊子，没有任何机制保证当前用对了字段"。【commit + LWN 881039 已核实，见 §9.2.4】
4. **kmalloc 档位不全是 2 的幂，96/192 两个填空档写死在 `kmalloc_index()` 里**：`kmalloc(180)` 落到 192 而非 256，省一档。现代内核 kmalloc cache 是二维 `kmalloc_caches[type][index]`（NORMAL / RECLAIM / DMA / CGROUP / 15 份 RANDOM 拷贝）。【真实源码 v6.12 `include/linux/slab.h`，见 §9.3.6】
5. **`GFP_ATOMIC` 与 `GFP_KERNEL` 的区别可以读 bit 读出来**：`GFP_ATOMIC = __GFP_HIGH | __GFP_KSWAPD_RECLAIM`，`GFP_KERNEL = __GFP_RECLAIM | __GFP_IO | __GFP_FS`。区别不在"原子"二字，而在**带不带 `__GFP_DIRECT_RECLAIM`**（自己能不能下场同步回收 = 能不能睡）。【真实源码 v6.12 `include/linux/gfp_types.h`，见 §9.5】
6. **时效（必背）**：SLOB 于 **6.4（2023）** 删除，经典 SLAB 于 **6.8（2024）** 删除（删掉 `mm/slab.c` 共 4026 行，整个 series 净删 5122 行）。**6.8+ 通用对象分配器只剩 SLUB**，boot 参数再也切不回 SLAB。本章一切"现状机制"以 SLUB 为准；SLAB 只作设计对比的历史参照。【LWN 951272 / 932201 已核实，见 §9.2.5】

---

## 前置依赖

- **第 3 章** · 进程地址空间与虚拟内存（vm_area_struct、pgd/pmd/pte、direct map 概念）
- **第 8 章** · buddy allocator 与 zone 模型（页级分配，SLUB 的下游供货商）
- **第 18 章** · 中断上下文 / spinlock（理解 `GFP_ATOMIC` 为什么不能睡、`might_sleep()` 的判定）
- **第 10 章** · OOM killer 与 page reclaim（kmalloc 失败路径、`__GFP_DIRECT_RECLAIM` 的去向）
- 微架构背景：cache line、false sharing、`cmpxchg` 的原子语义与 ABA 问题（本章会现场补，但有底子读起来更快）

> **本章取材边界声明**：所有标【真实源码 v6.12 路径】的代码块，是 2026-06 从 `https://raw.githubusercontent.com/torvalds/linux/v6.12/...` 逐字取得，未改一字（仅在必要处删掉无关行并用 `/* ... */` 标注省略）。标【示意代码，非逐字】的是我为讲解画的伪码 / 简化骨架。标「待核」的是我没取到一手证据、凭经验给的判断。绝不把示意代码冒充真实源码——这是这份文档与"浮在表面"版本的根本区别。

---

## 9.1 背景：为什么 buddy 之上还需要 slab

### 9.1.1 buddy 的物理粒度问题

buddy allocator 以**页（page，x86-64 上 4 KiB）**为最小单位。内核中大量数据结构远小于一页，如果每个都独占整页：

| 数据结构 | 量级（随版本/CONFIG 浮动，待核精确值） | 若独占整页的浪费率 |
|---|---|---|
| `struct task_struct` | 数 KiB 级 | 视页数而定 |
| `struct inode` | 数百字节 | ~90%+ |
| `struct file` | ~256 字节量级 | ~93%+ |
| `struct dentry` | ~192 字节量级 | ~95%+ |

直接用 buddy 分配小对象有两个硬代价：

- **内部碎片（internal fragmentation）**：一个 `struct file` 用一整页，3840+ 字节永久闲置。
- **构造开销**：每次分配后调用者要 `memset` 清零、再逐字段 `spin_lock_init` / `INIT_LIST_HEAD` / `atomic_set`。在每个 `open(2)` 都要造一个 `struct file` 的高频路径上，这笔构造代价不可忽视。

### 9.1.2 Jeff Bonwick 的 slab 洞见（1994，Solaris）

> **对象在"释放"之后保持已构造状态，下次"分配"只需重新激活，不必重新构造。**（Bonwick, USENIX Summer 1994, "The Slab Allocator: An Object-Caching Kernel Memory Allocator"）

这把热对象路径从「构造 + 分配」压成「分配（cache hit）」，顺便用"一页切成 N 个等大对象"解决了小对象碎片。Linux 在 2.x 时代引入 slab，此后三代演进：SLAB（原版）→ SLOB（嵌入式裁剪）→ SLUB（主线重写）。

> **底层根因：为什么"保持已构造态"这么值钱？** 不只是省几条初始化指令。真正贵的是 `spin_lock_init` 之类背后可能触动的 **lockdep 注册、debugobjects 登记**，以及把对象内存第一次拉进 CPU cache 的 **cold miss**。slab 把这些一次性成本摊到"slab page 第一次建立"，热路径只剩"从 freelist 弹一个、cache line 大概率已 warm"。这是 §9.6 ctor 一节的伏笔。

---

## 9.2 设计考古：slab → slob → slub 的真实演进与社区辩论

> 这一节是本章的"史料硬货"。每个论断尽量带出处（LWN 文章号 / commit / 邮件线程），并说清"谁、何时、为什么、解决前一代什么缺陷、社区辩论过什么"。

### 9.2.1 经典 SLAB（Linux 早期 ~ 6.7）的架构与它的"原罪"

SLAB 的形状（设计对比视角，SLAB 已删，以下为历史结构）：

```text
【示意代码，非逐字】经典 SLAB 的三级队列模型
kmem_cache (每种对象类型一个)
  ├── array_cache  (per-CPU 热对象指针栈，避免 lock)
  ├── alien caches (per-CPU × per-remote-node 的跨节点回收队列)  ← 原罪在这
  └── kmem_cache_node[nid]
        ├── slabs_full      ← 全部对象已分配
        ├── slabs_partial   ← 部分空闲
        └── slabs_free      ← 全部空闲（留着不还 buddy）
             每个 slab: [slab mgmt/header] [obj0][obj1]...[objN]
```

SLAB 在单机小系统上工作得很好，但 Christoph Lameter 2007 年提出 SLUB 时，列的核心缺陷是**队列规模随 CPU×Node 爆炸**：

> LWN 229984（2007-04-11，已 WebFetch 核实）记载 Lameter 的论点：SLAB 维护 per-node / per-cpu 的对象队列，外加 **alien cache** 结构，在 "1k nodes / processors" 的大机器上"光是存这些引用就占掉好几个 GB"，并警告极端下"整机内存有朝一日会被这些队列吃光"。此外 SLAB 把元数据塞在 slab 开头，"让对象对齐变难"。

换句话说，**SLUB 的初始动机不是"SLAB 慢"，而是"SLAB 的队列在大 NUMA 上不可扩展、吃内存"**。这是面试里区分"背过对照表"和"读过 changelog"的分水岭。

### 9.2.2 SLUB 的设计主张：用 cache line 友好取代队列（2007）

SLUB 源码文件头把设计哲学写得极干净——这是**真实源码**，不是后人总结：

```c
// SPDX-License-Identifier: GPL-2.0
/*
 * SLUB: A slab allocator that limits cache line use instead of queuing
 * objects in per cpu and per node lists.
 *
 * The allocator synchronizes using per slab locks or atomic operations
 * and only uses a centralized lock to manage a pool of partial slabs.
 *
 * (C) 2007 SGI, Christoph Lameter
 * (C) 2011 Linux Foundation, Christoph Lameter
 */
```
【真实源码 v6.12 `mm/slub.c` 行 1–11】

**逐行读这段哲学：**
- `limits cache line use instead of queuing objects`：SLAB 的队列是"把对象指针排进数组/链表"，每个队列项是一次潜在的 cache miss；SLUB 反其道——尽量不排队，把 free 指针**嵌进空闲对象本身**（§9.3.2），不额外占 cache line。
- `synchronizes using per slab locks or atomic operations`：常态走 atomic（`cmpxchg`），只有碰 partial 列表才上锁。
- `only uses a centralized lock to manage a pool of partial slabs`：唯一的"中心锁"是 per-node 的 `list_lock`，且 SLUB 拼命避免碰它。

SLUB 还顺手做了 **slab merging**：相同 size/对齐/flag 的 cache 合并成一个，LWN 229984 称"caches 数量减约 50%"，但也老实说这"可能暴露此前隐藏的 bug"（不同子系统的对象混在一个 cache 里，越界写会串台）。

### 9.2.3 社区辩论：明知有取舍，为什么还接受 SLUB？

LWN 229984 评论区当年就有 GC vs 手工内存管理之争，但更关键的辩论持续了十几年——**SLAB 在某些 workload 上确实更平滑（SLUB 把空 slab 立刻还 buddy，突发分配/释放会抖），那为什么最终还是删 SLAB？** 见 §9.2.5。

### 9.2.4 设计考古重头戏：`struct slab` 从 `struct page` 拆出来（5.17，2022）

这是近年 mm 子系统最重要的一次"卫生整顿"。

**为什么要拆？** LWN 881039（已核实）记录的根因：

> `struct page` 已经变成 "a complicated mess of #ifdefs and unions with no mechanisms to ensure that the right fields are used at any given time"（一堆 #ifdef 和 union 的烂摊子，没有任何机制保证在某一刻用对了字段）。物理页可能是用户数据、内核结构、DMA buffer……所有用途共用一个臃肿结构，互相踩字段；而 `struct page` 又被全内核引用，导致任何 mm 改动都牵一发动全身。

**谁做的、哪个版本？**
- 初始拆分 commit `d122019bf061`（"mm: Split slab into its own type"，**Matthew Wilcox (Oracle)**，作为 folio 工作的一部分）。【github commit 已 WebFetch 核实】
- Vlastimil Babka 接手打磨并在 **5.17 merge window（2022-01）** 合入主线。【LWN 881039 已核实】
- 关键设计决策：`struct slab` 故意放在 `mm/slab.h`（内部头），**不进 public header**，让 mm 之外的代码碰不到它；并按当前用哪个 allocator 收窄字段，防止跨 allocator 误用。

**拆完长什么样？** 这是**真实源码**，注意它如何"叠"在 `struct page` 上：

```c
/* Reuses the bits in struct page */
struct slab {
	unsigned long __page_flags;

	struct kmem_cache *slab_cache;
	union {
		struct {
			union {
				struct list_head slab_list;
#ifdef CONFIG_SLUB_CPU_PARTIAL
				struct {
					struct slab *next;
					int slabs;	/* Nr of slabs left */
				};
#endif
			};
			/* Double-word boundary */
			union {
				struct {
					void *freelist;		/* first free object */
					union {
						unsigned long counters;
						struct {
							unsigned inuse:16;
							unsigned objects:15;
							unsigned frozen:1;
						};
					};
				};
#ifdef system_has_freelist_aba
				freelist_aba_t freelist_counter;
#endif
			};
		};
		struct rcu_head rcu_head;
	};

	unsigned int __page_type;
	atomic_t __page_refcount;
#ifdef CONFIG_SLAB_OBJ_EXT
	unsigned long obj_exts;
#endif
};

#define SLAB_MATCH(pg, sl)						\
	static_assert(offsetof(struct page, pg) == offsetof(struct slab, sl))
SLAB_MATCH(flags, __page_flags);
SLAB_MATCH(compound_head, slab_cache);	/* Ensure bit 0 is clear */
SLAB_MATCH(_refcount, __page_refcount);
/* ... CONFIG_MEMCG 分支省略 ... */
#undef SLAB_MATCH
static_assert(sizeof(struct slab) <= sizeof(struct page));
#if defined(system_has_freelist_aba)
static_assert(IS_ALIGNED(offsetof(struct slab, freelist), sizeof(freelist_aba_t)));
#endif
```
【真实源码 v6.12 `mm/slab.h` 行 51–109】

**逐行注解（这段是理解"叠加"魔法的核心）：**
- `/* Reuses the bits in struct page */`：`struct slab` 并不另外分配内存。它和 `struct page` 占的是**同一块 `mem_map` 内存**——同一个物理页帧描述符，你既可以当 `struct page` 看，也可以当 `struct slab` 看。拆分只是"换一副类型安全的眼镜"，没换内存。
- `unsigned long __page_flags;` 必须放第一个字段，且 `SLAB_MATCH(flags, __page_flags)` 用 `static_assert` **编译期**强制它和 `struct page.flags` 偏移一致——因为 page flags（如 `PG_locked`、`PG_workingset`）是按固定偏移操作的，两套视图必须对齐。
- `SLAB_MATCH(compound_head, slab_cache); /* Ensure bit 0 is clear */`：`slab_cache` 复用 `compound_head` 的位置，注释提醒 bit 0 必须为 0（`compound_head` 的 bit 0 是"这是不是 tail page"的标志，slab page 不能被误判成复合页 tail）。
- `union { ... struct rcu_head rcu_head; }`：整个 freelist/counters 区域与 `rcu_head` 共用——这是给 `SLAB_TYPESAFE_BY_RCU` 用的，slab page 释放可走 RCU grace period（§9.6.2）。
- `static_assert(sizeof(struct slab) <= sizeof(struct page))`：**编译期防呆**。slab 视图绝不能比 page 大，否则就会越界踩到下一个 `struct page`。
- 最后那个 `IS_ALIGNED(offsetof(..., freelist), sizeof(freelist_aba_t))`：freelist 字段必须按 double-word 对齐——因为 fast path 要对它做 `cmpxchg_double`（§9.3.3），硬件 double-word CAS 有对齐要求。**这行 static_assert 是连接"数据结构布局"和"lockless 算法可行性"的螺栓。**

> **设计感小结**：拆 `struct slab` 没改任何运行时行为，纯粹是把"隐式的、靠注释约定的字段复用"变成"显式的、`static_assert` 编译期保证的类型"。这是大型 C 项目对抗熵增的经典手法——**用编译期断言把口头约定钉成机器可验证的契约**。folio 是同一思想的更大版本。

### 9.2.5 SLOB 删除（6.4）与 SLAB 删除（6.8）：明知有取舍为何还砍

这是 staff 级面试的"时效杀手锏"，必须讲清。

**SLOB（6.4，2023 删除）**：SLOB 是 first-fit 链表分配器，为几十 MB RAM 的嵌入式而生，无 per-CPU cache，省内存但慢、不适合 SMP。LWN 932201（已核实）记录 Vlastimil Babka 在 6.4 merge window 宣布删除"applause（全场鼓掌）"——因为 SLUB 已能通过 `CONFIG_SLUB_TINY`、`slub_max_order` 等调到小内存友好，SLOB 维护成本 > 收益。

**SLAB（6.8，2024 删除）**：这是更有"思考感"的决策。

- **删了多少**：LWN 951272（已核实）——SLAB deprecated 自 **6.5**，6.8 移除；整个 patch series 净删 **5122 行**，其中 `mm/slab.c` 单文件就是 **4026 行**。维护者 Vlastimil Babka。
- **为什么明知 SLAB 在个别 workload 更平滑还要删**：Babka 在 LWN 932201 的理由——SLAB 那 4000 行"不是都被经常或好好测试"；SLAB 和 SLUB 共享一层 common code，还重复实现了 memcg 等特性；**维护两套分配器的长期成本 > 个别场景的微小收益**。而且 SLUB 这些年已补平当年短板（per-cpu partial slab、`slub_max_order` 等）。Google / SUSE 等多家最终"nobody objects（无人反对）"。
- **删除还带来直接技术红利**：Babka 明说删 SLAB 后"不必再把 fastpath 代码在 `slab_common.c` 和 `slub.c` 之间切来切去"——这正是为什么 v6.12 的 `__kvmalloc_node` 等代码现在干净地待在一处（§9.4）。

> **面试答法**：被追问"SLAB 和 SLUB 现在怎么选"，正确回答是"**6.8 之后没得选，SLUB 是唯一通用对象分配器**"；能进一步说出"社区明知有 workload 取舍仍选择移除，因为长期维护成本压倒短期收益，且 SLUB 已补平性能短板"，就比背对照表高一档。

---

## 9.3 现状：SLUB 核心机制与数据结构（全部以 v6.12 源码为准）

### 9.3.1 三个核心结构：cache / per-cpu / slab

**(1) `struct kmem_cache`** —— 每种对象类型一个，是"零售店"的总账本：

```c
/*
 * Slab cache management.
 */
struct kmem_cache {
#ifndef CONFIG_SLUB_TINY
	struct kmem_cache_cpu __percpu *cpu_slab;
#endif
	/* Used for retrieving partial slabs, etc. */
	slab_flags_t flags;
	unsigned long min_partial;
	unsigned int size;		/* Object size including metadata */
	unsigned int object_size;	/* Object size without metadata */
	struct reciprocal_value reciprocal_size;
	unsigned int offset;		/* Free pointer offset */
#ifdef CONFIG_SLUB_CPU_PARTIAL
	/* Number of per cpu partial objects to keep around */
	unsigned int cpu_partial;
	/* Number of per cpu partial slabs to keep around */
	unsigned int cpu_partial_slabs;
#endif
	struct kmem_cache_order_objects oo;
	/* Allocation and freeing of slabs */
	struct kmem_cache_order_objects min;
	gfp_t allocflags;		/* gfp flags to use on each alloc */
	int refcount;			/* Refcount for slab cache destroy */
	void (*ctor)(void *object);	/* Object constructor */
	unsigned int inuse;		/* Offset to metadata */
	unsigned int align;		/* Alignment */
	unsigned int red_left_pad;	/* Left redzone padding size */
	const char *name;		/* Name (only for display!) */
	struct list_head list;		/* List of slab caches */
	/* ... CONFIG_SYSFS / kobj 省略 ... */
#ifdef CONFIG_SLAB_FREELIST_HARDENED
	unsigned long random;
#endif
	/* ... NUMA / KASAN / HARDENED_USERCOPY 等字段省略 ... */
};
```
【真实源码 v6.12 `mm/slab.h` 行 253–307，省略部分已标注】

逐字段挑重点：
- `size` vs `object_size`：`object_size` 是你 `sizeof(struct foo)`，`size` 是**加完对齐 padding / redzone 之后**实际占用。kmalloc(100) 的 `object_size=100` 但落进的 cache `size=128`，碎片就在这两者之差。
- `offset`：**free 指针嵌在对象内部的偏移**。SLUB 不用独立 bitmap，free 链直接写进空闲对象（§9.3.2）。
- `ctor`：构造函数，**没有 dtor**——SLUB 砍掉了经典 SLAB 的析构回调（§9.6.1）。
- `random`（仅 `CONFIG_SLAB_FREELIST_HARDENED`）：per-cache 随机数，用来 XOR 混淆 free 指针，防 heap 利用（§9.3.4）。
- `name` 的注释 `(only for display!)`：提醒你别拿 name 做逻辑判断，它只给 `/proc/slabinfo`、`/sys/kernel/slab/` 看。

**(2) `struct kmem_cache_cpu`** —— per-CPU 热缓存，fast path 的全部战场：

```c
struct kmem_cache_cpu {
	union {
		struct {
			void **freelist;	/* Pointer to next available object */
			unsigned long tid;	/* Globally unique transaction id */
		};
		freelist_aba_t freelist_tid;
	};
	struct slab *slab;	/* The slab from which we are allocating */
#ifdef CONFIG_SLUB_CPU_PARTIAL
	struct slab *partial;	/* Partially allocated slabs */
#endif
	local_lock_t lock;	/* Protects the fields above */
#ifdef CONFIG_SLUB_STATS
	unsigned int stat[NR_SLUB_STAT_ITEMS];
#endif
};
```
【真实源码 v6.12 `mm/slub.c`，`struct kmem_cache_cpu` 定义处】

**这个 union 是整个 SLUB lockless 设计的物理基础**：
- `freelist`（当前 active slab 的空闲对象链头）和 `tid`（全局唯一事务号）被塞进一个 union，对外可以当两个独立字段读，也可以当一个 `freelist_aba_t`（double-word）一次性 `cmpxchg`。
- 为什么要 tid？见 §9.3.3 的 ABA 防护——**没有 tid，单纯 CAS `freelist` 会被 ABA 问题骗过**。
- `slab`：当前正在掏的那个 slab；`partial`：per-cpu 攒的半满 slab（避免每次回 per-node 列表，减少 `list_lock` 争用）。
- `local_lock_t lock`：注意 fast path **不碰它**，它只保护 slow path 改这些字段；在 PREEMPT_RT 上语义不同（见 §9.3.3 末）。

**(3) `struct slab`**：已在 §9.2.4 逐行讲过，它叠在 `struct page` 上，`freelist` 字段是该 slab 的"主"空闲链（区别于 per-cpu 那条"借出去激活"的链）。

### 9.3.2 freelist：把空闲链嵌进对象本身

```text
【示意代码，非逐字】SLUB 对象布局（SLUB_DEBUG 关闭，offset 在对象头部为例）
slab 页内:
  [obj0: 前 8B = freeptr→obj1 | 其余是用户数据(空闲时无意义)]
  [obj1: 前 8B = freeptr→obj2 | ... ]
  [obj2: 前 8B = freeptr→NULL | ... ]   ← 链尾
  [obj3: 已分配，这 8B 现在是用户数据]
              ↑ freeptr 的位置由 kmem_cache.offset 决定
```

**为什么这样设计、为什么不用 bitmap？**
- bitmap 方案：每个 slab 额外维护一个位图标记哪些 obj 空闲。代价是**额外元数据 + 分配时要扫位图找空位**（一次潜在 cache miss + 计算）。
- SLUB 的"嵌入式 freelist"：空闲对象反正没人用，就拿它前 `offset` 字节当 next 指针。**零额外元数据，弹栈 O(1)**。这正是文件头说的 `limits cache line use instead of queuing`。

**新手误区**：以为 freelist 是独立数组/位图。不是——它是一条**穿过所有空闲对象内部**的单向链。

**代价（§9.6.1 会展开）**：对象一旦 free，它内部 `offset` 处那几字节就被覆写成 freeptr——"free 后完整保持已构造态"这个 Bonwick 承诺在 SLUB 下是**打折的**。

### 9.3.3 ⭐ Fast path 源码精读：一条 lockless cmpxchg 是怎么成立的

这是全章最硬的一段。先看**真实源码**，再逐行拆。

```c
static __always_inline void *__slab_alloc_node(struct kmem_cache *s,
		gfp_t gfpflags, int node, unsigned long addr, size_t orig_size)
{
	struct kmem_cache_cpu *c;
	struct slab *slab;
	unsigned long tid;
	void *object;

redo:
	/*
	 * Must read kmem_cache cpu data via this cpu ptr. Preemption is
	 * enabled. We may switch back and forth between cpus while
	 * reading from one cpu area. That does not matter as long
	 * as we end up on the original cpu again when doing the cmpxchg.
	 * ...
	 */
	c = raw_cpu_ptr(s->cpu_slab);
	tid = READ_ONCE(c->tid);

	/*
	 * Irqless object alloc/free algorithm used here depends on sequence
	 * of fetching cpu_slab's data. tid should be fetched before anything
	 * on c to guarantee that object and slab associated with previous tid
	 * won't be used with current tid. ...
	 */
	barrier();

	object = c->freelist;
	slab = c->slab;

	if (!USE_LOCKLESS_FAST_PATH() ||
	    unlikely(!object || !slab || !node_match(slab, node))) {
		object = __slab_alloc(s, gfpflags, node, addr, c, orig_size);
	} else {
		void *next_object = get_freepointer_safe(s, object);

		/*
		 * The cmpxchg will only match if there was no additional
		 * operation and if we are on the right processor.
		 * ...
		 * 1. Relocate first pointer to the current per cpu area.
		 * 2. Verify that tid and freelist have not been changed
		 * 3. If they were not changed replace tid and freelist
		 * ...
		 */
		if (unlikely(!__update_cpu_freelist_fast(s, object, next_object, tid))) {
			note_cmpxchg_failure("slab_alloc", s, tid);
			goto redo;
		}
		prefetch_freepointer(s, next_object);
		stat(s, ALLOC_FASTPATH);
	}

	return object;
}
```
【真实源码 v6.12 `mm/slub.c` 行 3915–3988，注释有删减，逻辑零改动】

配套的 CAS 原语：

```c
static inline bool
__update_cpu_freelist_fast(struct kmem_cache *s,
			   void *freelist_old, void *freelist_new,
			   unsigned long tid)
{
	freelist_aba_t old = { .freelist = freelist_old, .counter = tid };
	freelist_aba_t new = { .freelist = freelist_new, .counter = next_tid(tid) };

	return this_cpu_try_cmpxchg_freelist(s->cpu_slab->freelist_tid.full,
					     &old.full, new.full);
}
```
【真实源码 v6.12 `mm/slub.c` 行 3556–3567】

**逐行精读 + 底层根因：**

1. `c = raw_cpu_ptr(s->cpu_slab); tid = READ_ONCE(c->tid);`
   注意是 `raw_cpu_ptr` 不是 `this_cpu_ptr`——**它不禁抢占**。注释挑明："Preemption is enabled. We may switch back and forth between cpus"。也就是说，读 `c` 和读 `freelist` 之间，本线程可能被抢占、被迁到另一个 CPU、回来时甚至换了 CPU。SLUB **不阻止**这件事发生，而是**用 CAS 在最后一刻检测**它有没有发生。这是整段算法的精髓：**乐观并发（optimistic concurrency），不是悲观加锁。**

2. `barrier();`
   编译器屏障。注释说明：tid 必须**先于** `c->freelist` / `c->slab` 被读取。为什么？因为 tid 是"版本号"。如果先读 freelist 后读 tid，可能读到"旧 freelist + 新 tid"的撕裂组合，CAS 反而误判成功。`barrier()` 钉死读取顺序。**这是内存序（memory ordering）在真实代码里的杀伤力——顺序错了，正确性就错了。**

3. `object = c->freelist; slab = c->slab;`
   乐观地取当前空闲链头和 active slab。

4. `if (!USE_LOCKLESS_FAST_PATH() || unlikely(!object || !slab || !node_match(slab, node)))`
   三种情况掉进 slow path `__slab_alloc`：① 平台/配置不支持 lockless（如 PREEMPT_RT，见下）；② freelist 空了（当前 slab 掏完）；③ NUMA node 不匹配（要的 node 和 active slab 所在 node 不同）。`unlikely()` 是给编译器的分支预测提示——告诉它"正常情况走 else"，让 fast path 排在 CPU 的预测路径上（**微架构层面的分支预测优化**）。

5. `void *next_object = get_freepointer_safe(s, object);`
   读出空闲链的下一个节点（就是 `object` 内部 `offset` 处那个指针）。这一步就是 §9.3.2 "嵌入式 freelist" 的兑现。

6. **核心**：`__update_cpu_freelist_fast(s, object, next_object, tid)` →
   ```
   old = {freelist: object,      counter: tid}
   new = {freelist: next_object, counter: tid+1}
   this_cpu_try_cmpxchg_freelist(...full)   // double-word CAS
   ```
   一条 **double-word `cmpxchg`** 同时干三件事（对应注释里的 1/2/3）：
   - **原子地**比较 `(freelist==object && tid==旧tid)`；
   - 若都没变 → 把 freelist 推进到 `next_object`，并把 tid 自增；
   - 任一不符 → 返回 false。
   这就把"取对象 + 推进链头 + 校验没被抢占"压成**一条原子指令**，全程无锁、无关中断、无禁抢占。

7. **为什么必须把 tid 一起 CAS？——ABA 问题的真实现场。**
   设想没有 tid，只 CAS `freelist`：
   - 本线程读到 `freelist = A`（A 的 next 是 B），准备 CAS 成 B。
   - 此刻被抢占。别的路径在本 CPU 上 alloc 了 A、又 alloc 了 B、然后 free 回 A → `freelist` 又变回 A，但**此时 A 的 next 已经不是 B 了**（可能是 C）。
   - 本线程回来 CAS：`freelist == A`？成立！于是把 freelist 设成 B——**可 B 早被分出去了，双重分配，堆损坏。**
   这就是 ABA：值从 A 变 B 又变回 A，单纯比较值看不出"中间发生过事"。`tid` 是单调递增的版本号，每次操作 +1，CAS 连 tid 一起比，"中间发生过事 → tid 变了 → CAS 失败 → `goto redo`"。**double-word CAS（`freelist_aba_t`）的存在理由就是为了带上这个版本号。** 这也解释了 §9.2.4 那行 `IS_ALIGNED(..., sizeof(freelist_aba_t))` 的 static_assert——硬件 double-word CAS 要求对齐。

8. `note_cmpxchg_failure(...); goto redo;`
   CAS 失败（被抢占/迁移/竞争）→ 记一笔统计 → 重头来。这就是乐观并发的"失败重试"。失败极罕见（`unlikely`），所以摊销下来 fast path 仍是 O(1) 几条指令。

9. `prefetch_freepointer(s, next_object);`
   成功后预取**下一个**空闲对象的 freeptr 到 cache。**这是给下一次 alloc 铺路的微架构优化**——下次进来读 `next_object` 的 freeptr 时大概率已 warm，省一次 cache miss。staff 级的"为什么"就在这种细节里：SLUB 不只让本次快，还主动给下一次预热。

> **PREEMPT_RT 的特殊性**：`USE_LOCKLESS_FAST_PATH()` 在 `CONFIG_PREEMPT_RT` 下为 `false`（见源码行 192/204）。RT 内核里 spinlock 变成可睡眠的 rtmutex，lockless fast path 可能和正在进行的 slow path 互相干扰，所以 RT 改为"总是拿 `local_lock`，但仍复用 freelist 结构"。这是"同一份数据结构，两种并发策略"的工程取舍。【真实源码 v6.12 `mm/slub.c` 行 126–129、189–205】

### 9.3.4 Fast path 的安全加固：`CONFIG_SLAB_FREELIST_HARDENED`

free 指针嵌在对象里，意味着**堆溢出攻击只要能写到空闲对象就能改 free 链，进而控制下一次 alloc 的返回地址**。SLUB 的对抗手段是把 free 指针**加扰存储**：

```c
/*
 * Returns freelist pointer (ptr). With hardening, this is obfuscated
 * with an XOR of the address where the pointer is held and a per-cache
 * random number.
 */
static inline freeptr_t freelist_ptr_encode(const struct kmem_cache *s,
					    void *ptr, unsigned long ptr_addr)
{
	unsigned long encoded;

#ifdef CONFIG_SLAB_FREELIST_HARDENED
	encoded = (unsigned long)ptr ^ s->random ^ swab(ptr_addr);
#else
	encoded = (unsigned long)ptr;
#endif
	return (freeptr_t){.v = encoded};
}

static inline void *freelist_ptr_decode(const struct kmem_cache *s,
					freeptr_t ptr, unsigned long ptr_addr)
{
	void *decoded;

#ifdef CONFIG_SLAB_FREELIST_HARDENED
	decoded = (void *)(ptr.v ^ s->random ^ swab(ptr_addr));
#else
	decoded = (void *)ptr.v;
#endif
	return decoded;
}

static inline void *get_freepointer(struct kmem_cache *s, void *object)
{
	unsigned long ptr_addr;
	freeptr_t p;

	object = kasan_reset_tag(object);
	ptr_addr = (unsigned long)object + s->offset;
	p = *(freeptr_t *)(ptr_addr);
	return freelist_ptr_decode(s, p, ptr_addr);
}

static inline void set_freepointer(struct kmem_cache *s, void *object, void *fp)
{
	unsigned long freeptr_addr = (unsigned long)object + s->offset;

#ifdef CONFIG_SLAB_FREELIST_HARDENED
	BUG_ON(object == fp); /* naive detection of double free or corruption */
#endif

	freeptr_addr = (unsigned long)kasan_reset_tag((void *)freeptr_addr);
	*(freeptr_t *)freeptr_addr = freelist_ptr_encode(s, fp, freeptr_addr);
}
```
【真实源码 v6.12 `mm/slub.c` 行 468–552】

**逐行注解：**
- `encoded = ptr ^ s->random ^ swab(ptr_addr)`：存进对象的不是裸指针，而是"裸指针 XOR per-cache 随机数 XOR 存放地址的字节翻转"。
  - `s->random`：攻击者不知道，泄露一处也只泄露一个 cache 的密钥。
  - `swab(ptr_addr)`（地址字节序翻转）：让"同一个指针存在不同位置时编码不同"，**防止攻击者把别处抄来的合法 encoded 值平移过来复用**（位置相关加扰，类似加盐）。
- `get/set_freepointer` 里的 `kasan_reset_tag`：和 KASAN 的 tag-based 模式协作，先把硬件 tag 抹掉再算地址。这说明 freelist 操作要同时兼容 hardening 和 KASAN 两套机制——**真实内核代码的复杂度往往来自"多个正交特性必须共存"**。
- `BUG_ON(object == fp)`：hardening 下一个**朴素 double-free 探测**——如果有人把对象 free 给自己（free 链自环），立刻 BUG。注释自己都说 `naive`，它挡不住高级攻击，但零成本挡掉最蠢的一类 bug。

> **底层根因（为什么 XOR 而不是加密）**：fast path 每次 alloc/free 都要 encode/decode 一次，**必须是几个时钟周期能完成的操作**。XOR 是单周期、无分支、不污染流水线的理想选择；真正的加密（哪怕轻量 block cipher）会让热路径慢一个数量级。这是"安全 vs 性能"在指令级的精确权衡——**hardening 的全部艺术就是找到"几乎零成本但显著抬高利用门槛"的操作**。

### 9.3.5 Free fast path：对称的另一半

```c
static __always_inline void do_slab_free(struct kmem_cache *s,
				struct slab *slab, void *head, void *tail,
				int cnt, unsigned long addr)
{
	struct kmem_cache_cpu *c;
	unsigned long tid;
	void **freelist;

redo:
	c = raw_cpu_ptr(s->cpu_slab);
	tid = READ_ONCE(c->tid);

	/* Same with comment on barrier() in __slab_alloc_node() */
	barrier();

	if (unlikely(slab != c->slab)) {
		__slab_free(s, slab, head, tail, cnt, addr);
		return;
	}

	if (USE_LOCKLESS_FAST_PATH()) {
		freelist = READ_ONCE(c->freelist);

		set_freepointer(s, tail, freelist);

		if (unlikely(!__update_cpu_freelist_fast(s, freelist, head, tid))) {
			note_cmpxchg_failure("slab_free", s, tid);
			goto redo;
		}
	} else {
		/* Update the free list under the local lock */
		/* ... PREEMPT_RT 分支省略 ... */
	}
}
```
【真实源码 v6.12 `mm/slub.c` 行 4509–4543，RT 分支省略】

**对称之美 + 关键差异：**
- `if (unlikely(slab != c->slab))`：**free 的对象所属 slab，不一定是当前 CPU 的 active slab！** 这是 free 比 alloc 复杂的地方。alloc 总是从"我自己的 active slab"掏；但对象可能在 CPU 0 分配、被 CPU 1 释放。如果 `slab != c->slab`，fast path 走不了，掉进 `__slab_free`（slow path）——它会把对象还回该 slab 的"主" freelist，可能触发 slab 从 full→partial 的列表迁移，要碰 `list_lock`。
- 命中 fast path 时：`set_freepointer(s, tail, freelist)` 把待释放对象的 next 指向当前链头，再一条 `__update_cpu_freelist_fast` 把链头 CAS 成 `head`——**和 alloc 完全对称的乐观并发 + ABA 防护。**
- 支持 bulk（`head/tail/cnt`）：一次还一串对象（`kmem_cache_free_bulk`），把多次 CAS 摊成一次。

> **生产含义 / 真坑**：跨 CPU free 走 slow path 这件事，意味着**"在 CPU A 分配、固定在 CPU B 释放"的 workload（典型如某些 producer-consumer 队列）会持续打 slow path + `list_lock`**，吞吐远不如"谁分配谁释放"。排障时如果 `perf` 显示 `__slab_free` / `list_lock` 热，先怀疑对象的分配/释放是否跨 CPU。缓解手段见 §9.6.2。

### 9.3.6 kmalloc：通用小对象接口，档位与二维查表

`kmalloc(size, gfp)` 不需预建 cache，它从内核启动时预建的固定档位 cache 里挑最小够用的。档位逻辑**写死在编译期函数里**：

```c
static __always_inline unsigned int __kmalloc_index(size_t size,
						    bool size_is_constant)
{
	if (!size)
		return 0;

	if (size <= KMALLOC_MIN_SIZE)
		return KMALLOC_SHIFT_LOW;

	if (KMALLOC_MIN_SIZE <= 32 && size > 64 && size <= 96)
		return 1;
	if (KMALLOC_MIN_SIZE <= 64 && size > 128 && size <= 192)
		return 2;
	if (size <=          8) return 3;
	if (size <=         16) return 4;
	if (size <=         32) return 5;
	if (size <=         64) return 6;
	if (size <=        128) return 7;
	if (size <=        256) return 8;
	if (size <=        512) return 9;
	if (size <=       1024) return 10;
	if (size <=   2 * 1024) return 11;
	if (size <=   4 * 1024) return 12;
	/* ... 一路到 2 MiB ... */
	if (size <=  2 * 1024 * 1024) return 21;
	/* ... BUILD_BUG / BUG 兜底 ... */
}
static_assert(PAGE_SHIFT <= 20);
#define kmalloc_index(s) __kmalloc_index(s, true)
```
【真实源码 v6.12 `include/linux/slab.h` 行 ~648–696】

**逐行读出三个"面试常挖"的事实：**
- `return 1` 和 `return 2` 那两行就是传说中的 **96 / 192 填空档**。序列实际是 8/16/32/64/**96**/128/**192**/256/…。所以 `kmalloc(100)` 落 128（96 不够），但 `kmalloc(180)` 落 **192 而非 256**，省一档、压低中间尺寸碎片。**这两个非 2 幂档位是写死的，不是传说。**
- `size_is_constant` + `BUILD_BUG_ON_MSG`：如果 size 是编译期常量且超出最大档，**编译期就报错**而不是运行时崩——又一次"把错误左移到编译期"。
- `static_assert(PAGE_SHIFT <= 20)`：档位表最大到 2 MiB（`return 21`），断言确保页大小假设成立。

**二维查表**（确认旧版本的"待核"）：现代内核 kmalloc cache 不是一维数组，而是 `kmalloc_caches[type][index]`，type 来自这个 enum：

```c
enum kmalloc_cache_type {
	KMALLOC_NORMAL = 0,
	/* ... 视 CONFIG 把 DMA/CGROUP/RECLAIM 别名到 NORMAL 或独立 ... */
	KMALLOC_RANDOM_START = KMALLOC_NORMAL,
	/* ... RANDOM_KMALLOC_CACHES_NR(=15) 份随机拷贝 ... */
	KMALLOC_RECLAIM,
	KMALLOC_DMA,
	KMALLOC_CGROUP,
	NR_KMALLOC_TYPES
};
extern kmem_buckets kmalloc_caches[NR_KMALLOC_TYPES];
```
【真实源码 v6.12 `include/linux/slab.h` 行 ~573–599，按 CONFIG 化简】

- `KMALLOC_NORMAL / RECLAIM / DMA / CGROUP`：按用途分套——可回收的（如 dentry 名）、DMA 区的、cgroup 记账的，分开放便于回收统计和隔离。
- `KMALLOC_RANDOM_START ... RANDOM_KMALLOC_CACHES_NR(=15)`：**这是 `CONFIG_RANDOM_KMALLOC_CACHES` 安全特性**——把同一档位拆成 15 份随机拷贝，按调用点哈希分散，让攻击者更难预测某个 `kmalloc` 落到哪个 slab、更难做 heap feng shui（堆风水布局）。旧版本"待核"的二维结构，根因之一就是这个安全特性。

> **内部碎片的量化**：`kmalloc(100)` 落 128 档，浪费 28B（22%）。高频固定 size 的 struct 应建专用 `kmem_cache`（`size` 贴合 `object_size`，碎片趋零）。这就是为什么 `struct file`/`dentry`/`inode` 都有自己的 cache 而不走 kmalloc。

---

## 9.4 kmalloc vs vmalloc vs kvmalloc：物理连续性的取舍

### 9.4.1 物理连续性为什么是硬约束

```text
【示意代码，非逐字】
物理内存:  [pfn 100][pfn 101][空洞][pfn 103]...
kmalloc 返回的虚拟地址 → 落在内核 direct map，对应【连续物理页】
vmalloc 返回的虚拟地址 → 连续，但背后物理页可散，靠页表逐页拼

DMA 引擎 / 不带 IOMMU 的设备:
   直接吃物理地址 (bus addr)，无法理解"虚拟连续"
   → 必须物理连续 → 只能 kmalloc / alloc_pages / dma_alloc_coherent
```

### 9.4.2 对照表（已核实关键数字）

| 维度 | kmalloc | vmalloc |
|---|---|---|
| 物理连续 | 是 | 否（散页页表拼接）|
| 虚拟连续 | 是 | 是 |
| 大小上限 | 受 buddy 最大阶限制；`MAX_PAGE_ORDER` 默认 10 → 2¹⁰ 页 = 4 MiB（注：`MAX_ORDER` 语义在 6.4 改为"包含上界"并更名，结果不变） | 受 vmalloc 区大小（x86-64 上 32 TiB 量级）|
| 访问性能 | 快：命中 direct map，可被 2 MiB/1 GiB **大页 TLB 表项**覆盖 | 慢：vmalloc 区只能 PAGE_SIZE 粒度映射，TLB 覆盖差、miss 多 |
| 建/拆成本 | 低 | 高：改页表 + 多核 TLB flush（IPI）|
| 失败概率 | 较高（要连续物理页）| 较低（散页即可）|
| 接口 | `kmalloc`/`kzalloc`/`kmem_cache_alloc` | `vmalloc`/`vzalloc` |

> **🎯 纠高频误区（staff 级信号）**："vmalloc 慢是因为每次访问都软件 walk 页表"——**错**。vmalloc 内存和任何映射一样躺在硬件页表里，page-walker + TLB 一视同仁，没有额外软件遍历。它真正慢在两点：① 只能 PAGE_SIZE 粒度映射，拼不出大页表项，**TLB 覆盖率差**；② 建/拆映射要实改页表并跨核 TLB flush（IPI），**teardown 成本高**。能把"TLB 覆盖率"和"TLB flush 成本"分开讲，是真懂的标志。此外 vmalloc 区是全 CPU 共享的内核地址空间，大量使用会碎片化，`/proc/vmallocinfo` 看占用。

### 9.4.3 ⭐ kvmalloc 源码精读：弹性分配的标准答案

`kvmalloc` 是"先试物理连续（kmalloc），失败降级虚拟连续（vmalloc）"的官方弹性接口。**真实源码**：

```c
static gfp_t kmalloc_gfp_adjust(gfp_t flags, size_t size)
{
	/*
	 * We want to attempt a large physically contiguous block first because
	 * it is less likely to fragment multiple larger blocks and therefore
	 * contribute to a long term fragmentation less than vmalloc fallback.
	 * However make sure that larger requests are not too disruptive - no
	 * OOM killer and no allocation failure warnings as we have a fallback.
	 */
	if (size > PAGE_SIZE) {
		flags |= __GFP_NOWARN;

		if (!(flags & __GFP_RETRY_MAYFAIL))
			flags |= __GFP_NORETRY;

		/* nofail semantic is implemented by the vmalloc fallback */
		flags &= ~__GFP_NOFAIL;
	}

	return flags;
}

void *__kvmalloc_node_noprof(DECL_BUCKET_PARAMS(size, b), gfp_t flags, int node)
{
	void *ret;

	/*
	 * It doesn't really make sense to fallback to vmalloc for sub page
	 * requests
	 */
	ret = __kmalloc_node_noprof(PASS_BUCKET_PARAMS(size, b),
				    kmalloc_gfp_adjust(flags, size),
				    node);
	if (ret || size <= PAGE_SIZE)
		return ret;

	/* non-sleeping allocations are not supported by vmalloc */
	if (!gfpflags_allow_blocking(flags))
		return NULL;

	/* Don't even allow crazy sizes */
	if (unlikely(size > INT_MAX)) {
		WARN_ON_ONCE(!(flags & __GFP_NOWARN));
		return NULL;
	}

	/*
	 * kvmalloc() can always use VM_ALLOW_HUGE_VMAP, ...
	 */
	return __vmalloc_node_range_noprof(size, 1, VMALLOC_START, VMALLOC_END,
			flags, PAGE_KERNEL, VM_ALLOW_HUGE_VMAP,
			node, __builtin_return_address(0));
}
EXPORT_SYMBOL(__kvmalloc_node_noprof);
```
【真实源码 v6.12 `mm/util.c` 行 611–684】

**逐行精读（这段把好几个深坑串起来）：**

1. **为什么先试 kmalloc 而不是直接 vmalloc？** `kmalloc_gfp_adjust` 注释给了根因：先要一大块**物理连续**块，"长期看比 vmalloc fallback 更不容易碎片化"。反直觉但对——vmalloc 把大请求拆成散页，反而加剧物理内存碎片；kmalloc 拿到连续块，释放时整块还回，对 buddy 更友好。

2. **`if (size > PAGE_SIZE)` 才调 gfp**：小于一页根本不该 fallback（vmalloc 最小粒度就是一页，给子页请求用 vmalloc 是浪费 + 慢）。后面 `if (ret || size <= PAGE_SIZE) return ret;` 兜死：子页请求即使 kmalloc 失败也直接返回 NULL，不降级。

3. **`flags |= __GFP_NOWARN; flags |= __GFP_NORETRY;`**：既然有 vmalloc 兜底，这次 kmalloc 就"温柔尝试"——**不打失败告警、不死命重试、不触发 OOM killer**（`&= ~__GFP_NOFAIL`）。这是 kvmalloc 设计的精髓：**第一次尝试故意"软"，把"必须成功"的责任交给 vmalloc fallback。** 反过来，如果调用者显式给了 `__GFP_RETRY_MAYFAIL`（"我宁可 kmalloc 多试也不想要 vmalloc 的性能损失"），就不加 `__GFP_NORETRY`——把决定权留给调用者。

4. **`if (!gfpflags_allow_blocking(flags)) return NULL;`**：**这是 kvmalloc 最重要的使用约束**——vmalloc 路径必然可能睡眠（改页表、可能分配页表页），所以 `GFP_ATOMIC`/`GFP_NOWAIT` 下**根本不能走 fallback**，kmalloc 失败就只能返回 NULL。呼应函数 doc："GFP_NOWAIT and GFP_ATOMIC are not supported"。**在原子上下文里调 kvmalloc 期待它降级 vmalloc，是个 bug。**

5. `VM_ALLOW_HUGE_VMAP`：fallback 时允许 vmalloc 用大页映射，缓解 §9.4.2 说的"vmalloc TLB 覆盖差"——因为 kvmalloc 的调用者本来就不能假设指针的物理性质，给大页没有副作用。**这是把"vmalloc 慢"的根因之一直接在 fallback 里对冲掉。**

> **配套：kvfree 必须用**。kvmalloc 返回的指针你不知道来自 kmalloc 还是 vmalloc，所以释放必须用 `kvfree()`（它内部 `is_vmalloc_addr()` 判断走 `vfree` 还是 `kfree`）。混用 `kfree` 去放 vmalloc 指针会 crash。kvmalloc 自 **~4.12（2017）** 引入，正是为了消灭内核里大量"手写 kmalloc 失败再 vmalloc"的样板代码。

### 9.4.4 场景跑一遍：一个 1.5 MiB 的缓冲区该怎么分配

带读者沿推理走一遍（硬性要求 #3 的"思考感"）：

- **场景**：某子系统要一块 1.5 MiB 的临时缓冲，纯软件用，不做 DMA，进程上下文可睡眠。
- **走 `kmalloc(1.5M, GFP_KERNEL)`？** 1.5 MiB < 4 MiB 上限，理论可行。但要 buddy 拿出 **order-9（512 页）连续块**——系统跑久了高阶 free 常年为 0，大概率失败或触发昂贵的 compaction/reclaim。不适用。
- **走 `vmalloc(1.5M)`？** 散页拼接，几乎必成。但拿到的内存 TLB 覆盖差，如果这块缓冲后续被**高频随机访问**，TLB miss 会拖慢。仅一次性顺序读写则无所谓。
- **走 `kvmalloc(1.5M, GFP_KERNEL)`？** 正解。先温柔试 kmalloc（拿到则赚到，物理连续 + 大页 TLB）；拿不到自动降级 vmalloc（带 `VM_ALLOW_HUGE_VMAP` 缓解 TLB）。**默认就该用它**——除非你明确知道"必须物理连续"（DMA → 用 `dma_alloc_coherent`）或"明确高频随机访问且能接受 kmalloc 重试代价"（加 `__GFP_RETRY_MAYFAIL`）。
- **不适用边界**：原子上下文（中断/持 spinlock）**不能**用 kvmalloc（§9.4.3 第 4 点），那里要么改用 `GFP_ATOMIC` 的小 kmalloc，要么预分配 mempool。

---

## 9.5 GFP 标志：上下文合规，读 bit 就懂

### 9.5.1 为什么不能在原子上下文睡眠（底层根因）

内存分配可能触发 page reclaim，reclaim 可能等 I/O，等 = 让出 CPU = 调度。而中断 handler / spinlock 临界区是**不可抢占、不可调度**的上下文——在这里调度会破坏锁的持有假设、丢失中断状态，直接 BUG。

### 9.5.2 GFP 标志的真相：组合宏，不是魔法常量

**真实源码**，把"能不能睡"钉死到 bit：

```c
#define __GFP_RECLAIM ((__force gfp_t)(___GFP_DIRECT_RECLAIM|___GFP_KSWAPD_RECLAIM))
/* ... */
#define GFP_ATOMIC	(__GFP_HIGH|__GFP_KSWAPD_RECLAIM)
#define GFP_KERNEL	(__GFP_RECLAIM | __GFP_IO | __GFP_FS)
#define GFP_NOWAIT	(__GFP_KSWAPD_RECLAIM | __GFP_NOWARN)
#define GFP_NOIO	(__GFP_RECLAIM)
#define GFP_NOFS	(__GFP_RECLAIM | __GFP_IO)
#define GFP_HIGHUSER	(GFP_USER | __GFP_HIGHMEM)
```
【真实源码 v6.12 `include/linux/gfp_types.h` 行 ~261–386】

**对着 bit 读懂全部区别（这比任何对照表都硬）：**

- **`GFP_KERNEL = __GFP_RECLAIM | __GFP_IO | __GFP_FS`**，而 `__GFP_RECLAIM` = `__GFP_DIRECT_RECLAIM | __GFP_KSWAPD_RECLAIM`。**带 `__GFP_DIRECT_RECLAIM` = 调用者自己能下场同步回收 = 能睡。** 还带 `__GFP_IO`（回收时可发起 I/O）、`__GFP_FS`（可调进文件系统回收，如 shrinker）。
- **`GFP_ATOMIC = __GFP_HIGH | __GFP_KSWAPD_RECLAIM`**。对比 KERNEL，它**少了 `__GFP_DIRECT_RECLAIM`**（自己绝不同步回收 → 不睡），但有 `__GFP_KSWAPD_RECLAIM`（**能踢醒 kswapd 做后台回收**）。多出来的 `__GFP_HIGH` = **可动用 emergency reserve pool**（内存极紧张时的优先通道，代价是 pool 耗尽即失败、无重试）。
  - **关键区分**："能唤醒后台回收（kswapd）"和"能自己睡着等回收（direct reclaim）"是两回事。`GFP_ATOMIC` 有前者、没后者。这是面试区分背书和真懂的点。
- **`GFP_NOIO = __GFP_RECLAIM` 去掉 `__GFP_IO|__GFP_FS`**：能睡能回收，但回收时**不发起 I/O**——用于 block driver 自身路径，防止"回收又回头让本 driver 写盘"的递归死锁。
- **`GFP_NOFS = __GFP_RECLAIM | __GFP_IO`（无 `__GFP_FS`）**：能睡、能 I/O，但回收**不进文件系统**——用于 fs 路径，防止"持有 fs 锁时回收又回调 fs shrinker"的死锁。
- **`GFP_NOWAIT = __GFP_KSWAPD_RECLAIM | __GFP_NOWARN`**：像 ATOMIC 但**连 emergency pool 都不碰**（无 `__GFP_HIGH`），失败更干脆、更安静。

> **💡 把 `GFP_KERNEL | GFP_ATOMIC` 讲准确（别想当然）**：很多人以为"OR 出来就变 atomic 不睡"或"allocator 会 WARN 说你俩冲突"——**都不对**。位运算下，`GFP_KERNEL` 的 `__GFP_DIRECT_RECLAIM` 仍在，这块分配**照样可能进直接回收、照样可能睡**；`GFP_ATOMIC` 的 `__GFP_HIGH` 只是被附加上去（多了动用 reserve 的权限），并不会把"不睡"语义盖回来。真正抓"原子上下文里睡眠"的不是 gfp 校验器，而是分配路径里的 **`might_sleep()`**：开 `CONFIG_DEBUG_ATOMIC_SLEEP` 时，若此刻确实处于 atomic 上下文（持 spinlock / 在中断）就刷 "sleeping function called from invalid context" 的 BUG/告警。另外 **lockdep 的 `fs_reclaim` 标注**能在带 `__GFP_FS` 的分配里发现"回收路径 ↔ 已持锁"的依赖环。一句话：**诊断来自"上下文 + 实际睡眠"，不来自"flag 看着矛盾"。**

### 9.5.3 分配失败的正确处理

```c
/* 错误：忽略返回值（内核空指针解引用比用户态更难调试，可能直接 oops 卡死整机）*/
struct foo *p = kmalloc(sizeof(*p), GFP_KERNEL);
p->field = 1;  /* crash if p == NULL */

/* 正确：向上传播 errno，不要 panic */
struct foo *p = kmalloc(sizeof(*p), GFP_KERNEL);
if (!p)
	return -ENOMEM;
```
【示意代码，非逐字】

---

## 9.6 object cache 与 per-CPU 缓存：构造开销的根本解法

### 9.6.1 构造函数（ctor）的精确语义——别背成"全局只调一次"

`kmem_cache_create` 的 `ctor` 在**某个 slab page 被建立、对该 page 上的对象做初始化时**调用一次；之后对象在 free/alloc 间流转，**alloc 路径不再调 ctor**。这把 `spin_lock_init`/`INIT_LIST_HEAD`/`atomic_set` 等只付一次构造代价（§9.1.2 讲的 cold cost 摊销）。

> **🎯 staff 级精确性（两层都要讲对）**：
>
> **第一层——"per slab-page 一次"≠"系统生命周期一次"**：Bonwick 原版倾向把空 slab 留着、对象长期保持已构造态；但 **SLUB 把空 slab 立刻还 buddy**（文件头 `Slabs are freed when they become empty`）。于是同一段物理内存被 buddy 回收、再分给某个 cache 时，会重新 `setup_object` → **`ctor` 再次被调用**。正确表述："**每次某个 slab page 被实例化时，其上每个对象各调一次 ctor**"，不是"全程仅一次"。
>
> **第二层——"free 后保持完整已构造态"在 SLUB 下打折**：SLUB 把 free 指针**写在空闲对象内部**（`offset` 处，§9.3.2/9.3.4）。对象一旦 free，其 `offset` 处若干字节被覆写成（加扰的）freeptr。**Bonwick 那个"free 后逐字节原样保留"的经典承诺，在 SLUB 下不成立**。如果 `offset` 落在某个被 ctor 初始化过的字段上，下次 alloc 拿回来那几字节已不是 ctor 设的值（allocator 交还前会修好 freelist 区，但你不能假设整个对象逐字节不变）。
>
> **实践结论**：ctor 只放"再幂等也无所谓"的轻量初始化（`spin_lock_init` 之类）；真正依赖精确初值的字段，仍在 alloc 后由调用方显式设置。这就是为什么内核里 ctor 用得其实不多——`kzalloc` + 手动初始化更直白可控。

```c
/* 【示意代码，非逐字】ctor 的典型用法 */
static void my_struct_ctor(void *obj)
{
	struct my_struct *s = obj;
	spin_lock_init(&s->lock);
	INIT_LIST_HEAD(&s->list);
	atomic_set(&s->refcount, 0);
}
my_cache = kmem_cache_create("my_struct", sizeof(struct my_struct),
			      0, SLAB_HWCACHE_ALIGN, my_struct_ctor);
```

> **新手误区**：ctor 不是析构清理，而是"第一次构造"。对象析构要释放内部资源时，应在 `kmem_cache_free` 之前手动做——**SLUB 已彻底去掉经典 SLAB 的 dtor 回调**（看 §9.3.1 `struct kmem_cache` 里只有 `ctor`，没有 dtor 字段）。

### 9.6.2 per-CPU cache 的 NUMA / false sharing 影响与对策

SLUB 的 per-CPU 设计意味着：CPU 0 分配、CPU 1 释放的对象会回到 CPU 1 的 freelist（§9.3.5 那个 `slab != c->slab` 分支），高频跨 CPU 对象会持续打 slow path + `list_lock`，且对象在两个 CPU 的 cache line 间弹跳（false sharing）。对策：

1. **`SLAB_TYPESAFE_BY_RCU`**：允许对象 free 后在一个 RCU grace period 内"类型稳定"——内存可能被复用为**同类型**对象，但不会变成别的类型。配合 RCU read lock，无锁读者能安全地拿着指针而不怕它中途被释放重用成异类。代价：对象不会立刻真正释放。这也是 §9.2.4 `struct slab` 里那个 `struct rcu_head rcu_head` union 的用途。（详见 RCU 章）
2. **`alloc_percpu`**：高频且天然 per-CPU 的数据，直接用 per-CPU 变量，根本不跨 CPU。
3. **`SLAB_HWCACHE_ALIGN`**：让对象按 cache line 对齐，至少保证**不同对象不共享 cache line**（消除"无关对象挤在一条 line 上"的 false sharing）。

---

## 9.7 内核内存泄漏：kmemleak

### 9.7.1 原理：保守式 mark-and-sweep（tri-color 的保守版）

kmemleak（`CONFIG_DEBUG_KMEMLEAK`）是内核内置的、借鉴 tracing GC 思路的**保守式 mark-and-sweep** 泄漏检测器，类比用户态 valgrind memcheck，但实现是"扫内存找疑似指针"而非影子内存。

工作原理：
1. 每次 `kmalloc`/`vmalloc`/`kmem_cache_alloc` 等，kmemleak 把 `[addr, addr+size)` 登记进一棵全局对象树，每对象带引用计数和阈值 `min_count`（普通 kmalloc 默认 `min_count=1`）。
2. 扫描时，从 root（内核 data/bss、各任务栈、per-CPU 区等）出发，按字（指针对齐）逐字读，**只要某字的值落在某已登记对象的地址区间内（含指向对象中间的 interior pointer），就当作一个引用**，对应对象计数 +1。
3. 一轮扫描后，引用计数仍 **< `min_count`** 的对象 → 判"无引用" → 报告到 `/sys/kernel/debug/kmemleak`。

> **⚠️ 颜色语义务必别背反**：这是标准 tri-color marking 的保守版。按 GC 惯例，**白色 = 尚未被触达 = 泄漏候选**，**灰/黑 = 已被触达 = 安全（活）**。新分配对象初始在"待证明"一侧（white），被 root 链路触达后转"安全"侧；**始终没被任何 root 触达的，才是要报的那批**。很多资料把灰/白对调，方向反了全错。

### 9.7.2 保守扫描的两类误差（讲对方向）

保守扫描分不清"真指针"和"恰好长得像地址的整数"，由此两类误差：

- **整数巧合落在某对象地址区间 → false negative（漏报真漏）**：对象其实没人用了，但内存里某无关整数碰巧等于它的地址，被当成"还有引用"，于是不报。**kmemleak 最常见的局限。**
- **指针被变形存储（XOR/加扰/压缩，如 §9.3.4 的 freelist hardening、或某些指针压缩）→ false positive（误报）**：对象其实可达，但指向它的"指针"以变形值存在，保守扫描认不出，把活对象误判为泄漏。

> **💡 坑点**：kmemleak 对 `vmalloc` 区域的扫描在某些内核版本支持不完整（待核，取决于版本）。若泄漏对象来自 vmalloc，可能漏报。

### 9.7.3 使用与生产建议

```bash
# 内核配置 CONFIG_DEBUG_KMEMLEAK=y，启动参数 kmemleak=on（默认 off）
echo scan > /sys/kernel/debug/kmemleak   # 触发一次全量扫描（需 debugfs）
cat /sys/kernel/debug/kmemleak           # 看报告
# 典型输出:
# unreferenced object 0xffff888012345678 (size 128):
#   comm "insmod", pid 1234, jiffies 4294967295
#   backtrace:
#     [<...>] kmalloc+0x...
#     [<...>] my_module_init+0x...
```

生产默认关闭，因为：每次分配维护额外元数据；定期全量扫描有 CPU 开销；对实时系统不可接受。**调试流程**：复现环境开 `CONFIG_DEBUG_KMEMLEAK` → 复现操作（如 insmod/rmmod 模块）→ `echo scan` → 分析 backtrace，重点看模块 init 路径的无主分配。

---

## 9.8 落地：生产调优、排障与工具

### 9.8.1 关键观测命令

```bash
cat /proc/slabinfo                       # 所有 slab cache 原始统计
slabtop -o                               # 更易读（procps）
# 单 cache 详情（SLUB debug 开启时）
cat /sys/kernel/slab/kmalloc-128/alloc_calls
cat /proc/vmallocinfo | head -50         # vmalloc 占用
cat /proc/meminfo | grep -E "Slab|KReclaimable|Unreclaimable"
```

### 9.8.2 三个真实排障场景

**场景 1：内存持续涨，free 减少但进程 RSS 正常**
```bash
watch -n5 'grep Slab /proc/meminfo'      # Slab 行是否异常涨
slabtop -s c                             # 按 cache size 排序找元凶
# dentry/inode_cache 暴涨 → 大量文件创建删除 / 目录遍历未回收
echo 2 > /proc/sys/vm/drop_caches        # 临时释放 slab(dentry/inode)，有 IO 抖动风险
```

**场景 2：order>0 kmalloc 高阶分配失败**
```bash
cat /proc/buddyinfo                      # 高阶 free 是否为 0（碎片严重）
echo 1 > /proc/sys/vm/compact_memory     # 短期：触发 compaction
# 长期：把 order>0 kmalloc 改 vmalloc / kvmalloc（见 §9.4.4 推理）
```

**场景 3：怀疑模块 slab 泄漏（无 kmemleak 时用 tracepoint）**
```bash
# 优先挂 kmem tracepoint 而非裸符号 kprobe：
# 近年内核（分配剖析合入后，约 6.10+）这些函数常被改名/包裹（*_noprof 后缀，
# 见 §9.4 的 __kvmalloc_node_noprof）或内联，直接 kprobe 符号可能挂不上；
# tracepoint ABI 稳定得多。
bpftrace -e 'tracepoint:kmem:kmem_cache_alloc { @[kstack] = count(); }'
bpftrace -e 'tracepoint:kmem:kmem_cache_free  { @[kstack] = count(); }'
# 字段名随版本略变，以 bpftrace -lv 'tracepoint:kmem:*' 实测为准
```

### 9.8.3 生产优化建议

- **高频 struct 用专用 cache**：`kmem_cache_create` + `SLAB_HWCACHE_ALIGN`，碎片率低、cache-line 对齐、消除 false sharing。
- **必须成功的分配用 mempool**：`mempool_create` 预分配对象池，内存紧张时仍有保障（block/bio 路径常用）——是 **reserve 语义**，不是性能优化。
- **大块（≥ 一页、可睡眠）用 kvmalloc**：§9.4.3 已精读，默认弹性接口。

---

## 9.9 面试视角：staff 级高频题

### Q1：SLAB 和 SLUB 核心区别？SLUB 为什么更好？现在还怎么选？
1. 队列模型：SLAB 三队列 + alien cache vs SLUB 单 partial + per-cpu partial（空 slab 立即还 buddy）。
2. per-CPU：SLAB 是对象指针栈（array_cache）vs SLUB 是 `kmem_cache_cpu`（直接持 active slab + freelist + tid，走 lockless cmpxchg）。
3. 元数据：SLAB 嵌 slab 内 vs SLUB 复用 `struct page`/`struct slab` 字段。
4. NUMA：SLUB 原生 per-node partial vs SLAB 后补。
5. 可调试：SLUB 运行时 `/sys/kernel/slab/` 开关 vs SLAB 编译期。
- **陷阱 1**：别说"SLUB 一定更快"——突发分配/释放、空 slab 抖动场景 SLAB 反而平滑；SLUB 卖点是简洁、可维护、运行时可调、原生 NUMA。
- **陷阱 2（时效，能答出加分）**：被追问"现在怎么选"→ **6.8 后没得选，SLAB 已删，SLUB 唯一**。能讲出"社区明知有 workload 取舍仍砍，因长期维护成本(4000+行) > 个别收益，且 SLUB 已补平短板"（§9.2.5），高一档。
- **源码级加分**：能说出 fast path 是"`freelist+tid` 的 double-word `this_cpu_cmpxchg`，tid 防 ABA，不关中断不禁抢占"（§9.3.3），直接到 staff。

### Q2：kmalloc / vmalloc / kvmalloc 怎么选？vmalloc 为什么慢？
1. 物理连续需求 → 只能 kmalloc（DMA、硬件寄存器、`virt_to_phys`）。
2. 大块且无物理连续需求、可睡眠 → kvmalloc（先 kmalloc 后 fallback vmalloc，§9.4.3）。
3. vmalloc 慢分两层：**访问期** PAGE_SIZE 粒度映射 → TLB 覆盖差、miss 多（**不是"软件 walk 页表"**）；**建拆期** 改页表 + 跨核 TLB flush(IPI)。能把"TLB 覆盖率 vs TLB flush 成本"分开是 staff 信号。
4. 原子上下文**不能** kvmalloc（fallback 必睡，§9.4.3 第 4 点）——能答出这条加分。

### Q3：GFP_ATOMIC vs GFP_KERNEL，分别用在哪？
1. 读 bit：`KERNEL=__GFP_RECLAIM|IO|FS`（带 `__GFP_DIRECT_RECLAIM`=能睡）；`ATOMIC=__GFP_HIGH|__GFP_KSWAPD_RECLAIM`（无 direct reclaim=不睡，但能踢 kswapd，且有 emergency reserve）。
2. 为什么不能睡：spinlock/中断不可调度，调度=BUG。
3. **陷阱**：ATOMIC 不是"用于原子操作（atomic_t）"，而是"不可睡眠上下文"。
4. **陷阱**：`KERNEL|ATOMIC` OR 起来**仍可能睡**（direct reclaim 位还在）；抓睡眠的是 `might_sleep()`+`CONFIG_DEBUG_ATOMIC_SLEEP`，不是 gfp 校验（§9.5.2）。

### Q4：内核怎么检测内存泄漏？kmemleak 原理？
1. 保守 mark-and-sweep，登记所有分配，定期从 root 扫内存找疑似指针。
2. 两类误差讲对方向：**整数巧合落地址区间 → false negative（漏报真漏）**；**指针被 XOR/加扰变形 → false positive（误报活对象）**。
3. 颜色别背反：白=未触达=疑似泄漏，灰/黑=已触达=安全。
4. 生产不开（性能），调试开，看 `/sys/kernel/debug/kmemleak`；加分：kmem tracepoint + BPF 做精确 alloc/free 配对。

### Q5：buddy 为什么分大块容易失败？怎么缓解？
1. buddy 要物理连续 2^N 页，跑久了碎片化，大块连续不够。
2. compaction 迁移可移动页整理连续块（`/proc/sys/vm/compact_memory`）。
3. huge pages 预留连续块避免运行时碎片。
4. 根本：避免 order>0 kmalloc，大块用 vmalloc/kvmalloc。

### Q6：SLAB_CACHE_DMA / GFP_DMA 何时需要？
1. 要求对象落在 ISA DMA 可达 zone（x86 < 16 MiB ZONE_DMA）。
2. 只有 legacy ISA DMA 需要；PCIe + IOMMU 可访问任意 PA。
3. 现代系统基本不用；IOMMU 让 DMA 虚拟地址与 PA 解耦。

### Q7：mempool 是什么，解决什么？
1. 预分配对象保障池，内存极紧张（GFP_ATOMIC 都可能失败）时仍能拿到对象。
2. 场景：I/O 路径 bio/request，必须成功不能 -ENOMEM。
3. 机制：预放 N 个对象，`mempool_alloc` 先试 kmalloc 失败取 pool，`mempool_free` 先补 pool 再 kfree。
4. 加分：是 reserve 语义不是性能优化，别当 kmalloc 替代。

### Q8（源码级，区分 principal）：SLUB fast path 为什么需要 tid？去掉会怎样？
1. tid 是单调递增版本号，和 freelist 一起 double-word CAS。
2. 去掉 → 暴露 ABA：freelist 从 A→B→...→A，单 CAS 看不出"中间发生过事"，把已分配对象重复分出，堆损坏（§9.3.3 第 7 点跑过的例子）。
3. 这也解释 `struct slab.freelist` 那行 `IS_ALIGNED(..., sizeof(freelist_aba_t))` static_assert——double-word CAS 的对齐要求。

---

## 9.10 实战项目：P1 · 在模块里正确用 kmalloc/kfree（含验收与暗坑）

### 代码框架（真实可运行骨架）

```c
// slab_demo.c  【示意代码，非逐字：教学骨架，非内核源码】
#include <linux/module.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

struct my_obj { int id; spinlock_t lock; char data[64]; };

static struct kmem_cache *my_cache;
static struct my_obj *obj_a;   /* kmalloc 路径 */
static struct my_obj *obj_b;   /* kmem_cache 路径 */

static void my_obj_ctor(void *obj)
{
	struct my_obj *o = obj;
	spin_lock_init(&o->lock);   /* 只初始化锁这类"幂等也无所谓"的轻量字段 */
}

static int __init slab_demo_init(void)
{
	obj_a = kzalloc(sizeof(*obj_a), GFP_KERNEL);
	if (!obj_a)
		return -ENOMEM;
	spin_lock_init(&obj_a->lock);   /* kzalloc 无 ctor，需手动初始化 */
	obj_a->id = 1;

	my_cache = kmem_cache_create("my_obj_cache", sizeof(struct my_obj),
				     0, SLAB_HWCACHE_ALIGN, my_obj_ctor);
	if (!my_cache) { kfree(obj_a); obj_a = NULL; return -ENOMEM; }

	obj_b = kmem_cache_alloc(my_cache, GFP_KERNEL);
	if (!obj_b) {
		kmem_cache_destroy(my_cache); my_cache = NULL;
		kfree(obj_a); obj_a = NULL;
		return -ENOMEM;
	}
	obj_b->id = 2;   /* spin_lock 已由 ctor 初始化，不必再 init */

	pr_info("slab_demo: obj_a=%px obj_b=%px\n", obj_a, obj_b);
	return 0;
}

static void __exit slab_demo_exit(void)
{
	/* 顺序：先 free 对象，再 destroy cache —— 反序会 crash/WARN */
	if (obj_b) { kmem_cache_free(my_cache, obj_b); obj_b = NULL; }
	if (my_cache) { kmem_cache_destroy(my_cache); my_cache = NULL; }
	if (obj_a) { kfree(obj_a); obj_a = NULL; }
	pr_info("slab_demo: exited cleanly\n");
}
module_init(slab_demo_init);
module_exit(slab_demo_exit);
```

### 验收
```bash
make && sudo insmod slab_demo.ko
dmesg | tail -5                          # 看到 obj_a= obj_b= 地址，无 WARN
grep my_obj_cache /proc/slabinfo         # cache 出现
sudo rmmod slab_demo
dmesg | tail -3                          # "exited cleanly"，无 WARN
echo scan > /sys/kernel/debug/kmemleak   # 若开了 kmemleak
cat /sys/kernel/debug/kmemleak           # 应无 slab_demo 条目
```

### 暗坑
1. **`kmem_cache_destroy` 前必须 free 全部对象**：否则 WARN_ON 并泄漏 cache，高版本可能 BUG。
2. **ctor 里不要做分配**：ctor 在 slab 初始化时调用，可能处于不可分配上下文；且因 §9.6.1 第一层，ctor 会随 slab 重建被多次调用。
3. **卸载反初始化顺序**：永远"先 free 对象，再 destroy cache"。
4. **`%px` 是 review 红线**：自 **4.15（2018）** 起 `%p` 默认对内核指针哈希（防 KASLR 泄露），`%px` 主动绕过这层防护，**只该临时调试用，正式代码别留**。早于 4.15 的 `%p` 还是裸地址。

---

## 9.11 设计题：为高频 HTTP 连接对象设计内核内存分配策略

**背景**：内核态网络代理模块（类 kTLS offload），每连接一个 512B `struct conn`，峰值 100 万并发，建/销 10 万/秒，需支持 DMA 到网卡。

**讨论要点（带本章的源码级理由）**：
1. **专用 `kmem_cache`**：512B 是固定 size，专用 cache 让 `kmem_cache.size≈object_size`，碎片趋零；配 `SLAB_HWCACHE_ALIGN` 防 false sharing。
2. **DMA 分离**：连接对象本身不需 DMA（内核 CPU 访问），但连接内的 tx/rx buffer 要 DMA——**分开分配**：conn 走专用 cache，buffer 走 `dma_alloc_coherent` / DMA pool。别把整个 conn 塞进 DMA zone（浪费稀缺的低端内存）。
3. **跨 CPU 释放的代价**：10 万/秒 建销，若"在网卡软中断 CPU 分配、在应用 CPU 释放"会持续打 §9.3.5 的 slow path + `list_lock`。设计上尽量让**同一连接的分配/释放固定在同一 CPU**（按 RX queue 绑核），或用 `SLAB_TYPESAFE_BY_RCU` 让无锁查表读者安全。
4. **NUMA**：100 万对象、多 node，SLUB per-node partial 自然分摊；按网卡所在 node 分配 conn 减少跨节点访问。
5. **内存压力降级**：mempool 保最小连接数（如 1000），超出宁可拒连（返回错误）也别 OOM——比内核 OOM killer 乱杀进程优雅。

---

## 9.12 系统题：线上 OOM 但进程 RSS 正常，定位 slab 泄漏

**场景**：128 GiB 服务器跑第三方驱动模块 A，两周后 `free -h` available 持续降直至 OOM，`top` 所有进程 RSS 加总远小于 128 GiB。

**排障路径**：
```bash
# Step 1: slab 泄漏 vs 页缓存
cat /proc/meminfo
#   Slab: 很大(>50G) → 基本是 slab 泄漏
#   Cached: 很大 → 页缓存(drop_caches 可暂缓)

# Step 2: 找异常 cache
slabtop -s c | head -20          # 看 CACHE SIZE 列找元凶 cache

# Step 3: 有 kmemleak(理想)
cat /sys/kernel/debug/kmemleak | grep -A10 module_A

# Step 4: 无 kmemleak → kmem tracepoint(比裸 kprobe 稳, §9.8.2)
bpftrace -e 'tracepoint:kmem:kmem_cache_alloc { @[kstack]=count(); }'
bpftrace -e 'tracepoint:kmem:kmem_cache_free  { @[kstack]=count(); }'
# alloc/free 计数差最大的 kstack → 泄漏路径

# Step 5: 量级确认
# 某 cache 1000万对象 × 512B = 5GiB，× 增长速率 → 估 OOM 时点
```
**答案要点**：slab 泄漏不在进程 RSS（RSS 是用户态分配），只在 `/proc/meminfo` 的 `Slab:` 行反映。关键链路：`slabtop` 找异常 cache → kmemleak / kmem tracepoint+BPF 找 backtrace → alloc/free 配对差定位泄漏点。

---

## 章末五件套

### 一、本章心法
> **buddy 是页级批发，SLUB 是对象级零售，GFP 标志是上下文合规证。SLUB 的全部魔法是一条 `freelist+tid` 的 lockless double-word cmpxchg——tid 防 ABA，不关中断不禁抢占，靠"乐观并发 + 失败重试"取代加锁。把这条 fast path 和它背后的 ABA/内存序/cache 预取讲清楚，内核分配路径 80% 的 bug 与性能问题你都能一眼看穿。**

### 二、一句话回答（电梯测试）
- SLAB→SLUB 为什么重写？→ SLAB 的 per-node/cpu/alien 队列在大 NUMA 上吃几个 GB 内存、不可扩展（LWN 229984）。
- SLUB fast path 凭什么无锁？→ per-cpu `freelist+tid` 的 double-word cmpxchg，tid 当版本号防 ABA，被抢占/迁移就 CAS 失败重试。
- 大块内存默认怎么分？→ `kvmalloc`：先温柔试 kmalloc（不 OOM 不重试），失败降级 vmalloc；但原子上下文不能用。
- GFP_ATOMIC 和 KERNEL 差在哪？→ 差一个 `__GFP_DIRECT_RECLAIM`（能不能自己下场同步回收=能不能睡）。

### 三、自测清单（答不上回去重读对应节）
- [ ] 不看源码，画出 `__slab_alloc_node` fast path 的 5 个关键步骤，并说明 `barrier()` 和 tid 各防什么。（§9.3.3）
- [ ] 解释 `struct slab` 为什么能"叠"在 `struct page` 上，`SLAB_MATCH`/`static_assert` 保证了什么。（§9.2.4）
- [ ] `kmalloc(96)`/`kmalloc(180)`/`kmalloc(200)` 各落哪个档，为什么有 96/192 两个非 2 幂档。（§9.3.6）
- [ ] 读 `kvmalloc` 源码说出：为什么先试 kmalloc、为什么加 `__GFP_NORETRY`、为什么原子上下文不能用。（§9.4.3）
- [ ] 写出 `GFP_ATOMIC`/`GFP_KERNEL`/`GFP_NOFS`/`GFP_NOIO` 的 bit 组合，并解释 NOFS/NOIO 防的是什么死锁。（§9.5.2）
- [ ] 说清 kmemleak 颜色语义和两类误差的方向（哪类是 false positive 哪类是 false negative）。（§9.7）
- [ ] SLOB / SLAB 各在哪个版本删除、为什么删（带行数/理由）。（§9.2.5）

### 四、延伸阅读（已核实可达）
- LWN 229984「The SLUB allocator」（2007，Lameter 的原始设计动机）
- LWN 881039「Splitting the page」/ 相关 folio 系列（`struct slab` 拆分背景）
- LWN 932201「Looking ahead to the 6.4 kernel」相关（SLOB 删除、SLAB 删除提案讨论）
- LWN 951272「remove the SLAB allocator」（SLAB 6.8 删除，行数与理由）
- commit `d122019bf061`「mm: Split slab into its own type」（Matthew Wilcox）
- 源码：`mm/slub.c`、`mm/slab.h`、`mm/util.c`、`include/linux/slab.h`、`include/linux/gfp_types.h`（v6.12）
- 经典：Jeff Bonwick「The Slab Allocator」(USENIX 1994)

### 五、与其他章的接口
| 引用点 | 所在章 |
|---|---|
| 虚拟内存地址空间、pgd/pte、direct map | 第 3 章 |
| buddy allocator、zone 模型、`struct page`、compaction | 第 8 章 |
| 中断上下文、spinlock、不可抢占、`might_sleep` | 第 18 章 |
| OOM killer、page reclaim、`__GFP_DIRECT_RECLAIM` 去向 | 第 10 章 |
| RCU、`SLAB_TYPESAFE_BY_RCU`、grace period | RCU 章 |

---

*文档最后更新：2026-06-12。所有【真实源码 v6.12 路径】块取自 `raw.githubusercontent.com/torvalds/linux/v6.12`，逐字未改（省略处已标注）。版本结论已核实：SLOB 于 6.4（2023）删除、SLAB 于 6.8（2024）删除（删 mm/slab.c 4026 行）、`struct slab` 从 `struct page` 拆分于 5.17（commit d122019bf061，Matthew Wilcox）。标「待核」处为未取到一手证据的经验判断。*
