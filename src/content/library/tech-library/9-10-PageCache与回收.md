---
title: "Page Cache 与内存回收（深化版）"
slug: "9-10"
collection: "tech-library"
group: "linux系统"
order: 9010
summary: "一句话定位：Page cache 是 Linux I/O 性能的核心杠杆——读写都经过它；内存回收的本质是在\"保留热数据\"与\"释放给新分配\"之间做持续权衡。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T07:39:57.000Z"
updatedAt: "2026-06-12T07:39:57.000Z"
---
> **一句话定位**：Page cache 是 Linux I/O 性能的核心杠杆——读写都经过它；内存回收的本质是在"保留热数据"与"释放给新分配"之间做持续权衡。这一章的深化目标，是把"page cache + LRU + reclaim + OOM"从一组**概念**，变成你能在 `mm/filemap.c`、`mm/vmscan.c`、`mm/workingset.c`、`mm/oom_kill.c`、`mm/page-writeback.c` 里**逐行指出实现位置、说清为什么这样设计、复述社区当年争论过什么**的硬知识。

---

## TL;DR — 本章最硬结论

1. **Page cache 的物理载体在 v5.16 之后是 folio，不是 page**。`struct folio`（`include/linux/mm_types.h`）保证"绝不是 tail page"，从类型上消灭了 head/tail page 混淆这个十几年的隐患。这是 Matthew Wilcox 主导、社区吵了很久（命名、是否值得、memdesc 路线）才合入的一次基础重构。理解 page cache 现在必须从 folio 讲起。
2. **LRU 是"双链表 + 二次机会 + refault 在线自适应"三件套**：新页先进 inactive，二次访问才晋升 active（防顺序读污染）；驱逐时写 shadow entry 记录 `nonresident_age`，refault 回来时算 `refault_distance` 与 workingset size 比较，决定是否直接复活成 active（`mm/workingset.c`）。
3. **回收有两条路径，压力层级靠 watermark 分层**：`kswapd` 后台异步（跨过 low 唤醒，回收到 high），direct reclaim 在分配进程上下文同步阻塞（跌破 min 才触发）。`pgscand/s > 0` 才是真压力信号，`free` 小不是。
4. **swappiness=0 在 3.5 之后语义彻底变了**（commit fe35004f，Satoru Moriya）：从"小概率扫匿名页"变成 `ap=0` 完全不扫匿名页，代价是 page cache 还在也可能先 OOM。这是"数据库为什么配 1 不配 0"的根因。
5. **OOM `oom_badness()` 是 per-mm 计量**（RSS + MM_SWAPENTS + 页表页 + adj×totalpages/1000），**不累加子进程**（那是 2.6.36 前 `badness()` 的老行为）。`oom_score_adj=-1000` 是直接 `return LONG_MIN` 短路豁免，不是"分很低"。
6. **MGLRU（v6.1 合入）是一整套替换实现，不是双链表的旋钮**。它用多代 generation + page-table walk + bloom filter 取代 rmap 逐页扫描，Google 在 ChromeOS/Android 上量产；社区（Johannes Weiner）当年最大的顾虑是"两套回收代码无法同时维护"，最终走的是替换而非共存。

**前置依赖章**：第 7 章（虚拟内存与分页）、第 8 章（物理内存管理：Buddy / Slab / watermark）。块 I/O 路径下半段（bio、I/O 调度、wbt）在第 16 章。cgroup memory controller 在第 22 章。

**本章源码基线**：Linux **v6.12**（除非显式标注其它版本）。所有标【真实源码 v6.12 路径】的片段均逐字取自 `raw.githubusercontent.com/torvalds/linux/v6.12/...`；标【示意代码，非逐字】的是为讲清逻辑做的简化；个别拿不准的标「待核」。

---

## 10.0 阅读地图：五个文件，一条数据流

在钻细节前先建立"哪段逻辑在哪个文件"的肌肉记忆，否则后面的源码会散成一地：

```
read()/write()/mmap fault
        │
        ▼
  mm/filemap.c        ← page cache 的查找/读取/缺页：__filemap_get_folio,
        │               filemap_read, filemap_get_pages, filemap_fault
        ▼
  (miss) bio 提交 → 第 16 章块层
        │
        ▼ 内存变紧
  mm/vmscan.c         ← 回收主引擎：kswapd(balance_pgdat), direct reclaim
        │               (try_to_free_pages), get_scan_count(swappiness),
        │               shrink_lruvec / shrink_inactive_list / shrink_folio_list
        ▼
  mm/workingset.c     ← refault 检测：workingset_eviction(写shadow),
        │               workingset_refault(读shadow算距离决定是否activate)
        ▼ dirty 页要落盘
  mm/page-writeback.c ← dirty 限速:balance_dirty_pages, domain_dirty_limits
        ▼ 回收彻底失败
  mm/oom_kill.c       ← out_of_memory → select_bad_process → oom_badness
```

记住这张图：本章后面每讲一个机制，都会先告诉你"它在上面哪一行"。

---

## 10.1 背景：为什么 Page Cache 是 Linux I/O 的核心

### 问题的本质（底层根因：存储层级的延迟鸿沟）

现代存储与内存的延迟/带宽差 3–5 个数量级：

| 介质 | 随机读延迟（量级） | 相对内存 |
|------|------|------|
| 内存（DRAM） | ~100 ns | 1× |
| NVMe SSD | ~100 μs | ~1000× |
| SATA SSD | ~500 μs | ~5000× |
| HDD（寻道+旋转） | ~10 ms | ~100000× |

这不是工程偏好问题，是**排队论 + 局部性原理**的硬约束：任何 I/O 密集负载，如果每次访问都打到磁盘，吞吐被设备 IOPS 钉死，延迟被设备 service time 钉死。缓存的唯一作用是把"命中流量"从设备队列上摘走——命中率每提高一点，设备队列长度（Little's Law：`L = λ·W`）就线性下降。

Linux 的解法是把**所有空闲物理内存都当文件缓存**。不是"分配一块固定缓存区"，而是"free 内存就是潜在的 page cache，需要时再吐出来"。这个哲学决定了三条反直觉的运维结论：

- `free` 命令里 `available` 比 `free` 有意义得多；
- 运行越久 buff/cache 越大是**健康态**，不是泄漏；
- 真正的病症是 **cache miss rate / refault rate 上升**，不是 cache 占用多。

### Page Cache 的统一地位与它的真实载体

教科书常画成"`read()` 走 page cache"，但 v5.16 之后**page cache 里装的是 folio，不是 page**。在 `mm/filemap.c` 里，查找缓存项的核心 API 已经是：

```c
struct folio *__filemap_get_folio(struct address_space *mapping,
    pgoff_t index, fgf_t fgp_flags, gfp_t gfp)
```
【真实源码 v6.12 路径 mm/filemap.c —— 函数签名】

- `mapping`：每个被缓存的文件（inode）有一个 `address_space`，它的 `i_pages`（一棵 **XArray**）就是"页索引 → folio"的映射表。
- `index`：以 `PAGE_SIZE` 为单位的文件内偏移。
- 返回 `struct folio *`：命中则拿到现成 folio，未命中按 `fgp_flags`（如 `FGP_CREAT`）决定是否新建。

整条读路径（`filemap_read` → `filemap_get_pages`）的伪流程：

```
filemap_read(iocb, iter)              [mm/filemap.c]
  → filemap_get_pages(iocb, count, fbatch, ...)
      → filemap_get_folio()  命中？
          命中 → 直接 copy_folio_to_iter()（零磁盘 I/O）
          未命中 → page_cache_sync_readahead() / filemap_create_folio()
                   → mapping->a_ops->read_folio() → 提交 bio（第 16 章）
                   → folio_wait_locked() 等数据就绪
```

写路径对称：`write()` 把数据拷进 folio、置 dirty、立即返回；真正落盘由 writeback（10.2）异步完成。由此三条**断电安全**结论：

- `write()` 返回 ≠ 落盘；
- `fsync()` / `fdatasync()` 才是"我要落盘"的正确姿势；
- 绝大多数"断电丢数据"的根因都在这一层（应用以为写完了，其实还在 dirty page 里）。

> **为什么是 XArray 而不是别的？** `i_pages` 在 v4.20 之前是 radix tree，之后内核统一换成 XArray（Wilcox 的另一项工作）。XArray 的关键能力：①支持在 slot 里直接塞"value entry"（最低位打标记的整数），这让 **shadow entry / swap entry** 能和真实 folio 指针共用同一棵树，refault 检测（10.3）正是靠这个；②内建 RCU 读 + 细粒度锁，配合下面的锁序。

### 真坑：page cache 的锁序（为什么 truncate vs fault 不会死锁）

`mm/filemap.c` 文件头有一段**锁序宪法**，是理解 mm 死锁的总纲：

```c
/*
 * Lock ordering:
 *
 *  ->i_mmap_rwsem		(truncate_pagecache)
 *    ->private_lock		(__free_pte->block_dirty_folio)
 *      ->swap_lock		(exclusive_swap_page, others)
 *        ->i_pages lock
 *
 *  ->i_rwsem
 *    ->invalidate_lock		(acquired by fs in truncate path)
 *      ->i_mmap_rwsem		(truncate->unmap_mapping_range)
 *
 *  ->mmap_lock
 *    ->i_mmap_rwsem
 *      ->page_table_lock or pte_lock
 *        ->i_pages lock
 */
```
【真实源码 v6.12 路径 mm/filemap.c —— 文件头锁序注释】

逐层读法（自顶向下 = 加锁顺序，反向 = 必然死锁）：
- **`i_rwsem`（inode 读写锁）在最外层**：truncate/write 这种改变文件大小/内容的操作先拿它。
- **`invalidate_lock`**（v5.15 引入，Jan Kara）专门解决"truncate 把页删了，但缺页正在为同一偏移建页"的竞态——以前用 page lock 兜，覆盖不全，导致过 stale data。现在 truncate 路径和 fault 路径都过这把锁。
- **`i_pages lock` 永远在最内层**：所有人改 XArray 都最后才碰它，且持有时间极短。

**工程意义**：当你在生产里看到 `mmap_lock` 相关的 hung task 或 lockdep splat，对照这张序表就能判断是谁逆序拿锁。这套锁序也是为什么 folio 化要"一个子系统一个子系统改"——锁的语义必须在转换中保持不变。

---

## 10.2 演进：从 bdflush 到 IO-less dirty throttling（设计考古）

writeback（把 dirty page 刷回磁盘）的历史，是一部"如何在不卡死前台的前提下持续吐数据"的演进史。这一节既讲历史也落到 v6.12 源码。

### 2.6 之前：bdflush + kupdate（单线程时代）

早期两个内核线程：

- **bdflush**：被动触发，dirty 占比超阈值才刷；
- **kupdate**：定时（默认 30s）扫 dirty 页。

**缺陷**：单线程、**不区分设备**——一块慢 HDD 的回写会阻塞所有设备；调参空间近乎没有。

### 2.6：pdflush（线程池，但仍全局）

`pdflush`（page dirty flush）引入线程池，多线程并发回写，可同时服务多设备。**但线程池是全局共享的**，高负载下慢设备仍能抢占快设备的回写线程。

### 2.6.32+：per-BDI flusher（每设备一条回写流）

关键演进：每个 **BDI（Backing Device Info）** 拥有自己的回写线程（现代以 `kworker/u*:*+flush-<major>:<minor>` 形式出现）。

```
BDI(sda)    ──▶ writeback ctx A ──▶ bio 队列 ──▶ sda
BDI(nvme0)  ──▶ writeback ctx B ──▶ bio 队列 ──▶ nvme0n1
```

工程推论：①慢设备不再饿死快设备——这是"混合存储"生产环境的基础；②per-BDI 统计让 `/sys/block/<dev>/stat`、`iostat` 更准；③可 per-mount 差异化 `dirty_bytes`。

### 2011 的真正难题与 IO-less throttling（Fengguang Wu / Jan Kara）

per-BDI 解决了"哪条线程刷哪个设备"，但没解决**前台写进程怎么限速**。老做法（`balance_dirty_pages` 直接让写进程自己去发起回写 I/O）有个恶性问题：被限速的进程提交的 I/O 是**小而散**的，反而恶化磁盘 seek，形成"越限速越慢"的正反馈。

2011 年合入的 **IO-less balance_dirty_pages**（Fengguang Wu 主导，Jan Kara 等参与）改成：**前台进程不再自己发 I/O，只是按算出来的速率睡一会儿（throttle by sleeping）**，真正的回写完全交给 flusher。限速量由一个**位置式速率控制器（position-based rate control）**算出——把 dirty 量当作"水位"，用类似控制论的 setpoint 把水位稳在 `dirty_ratio` 附近。

v6.12 的 `mm/page-writeback.c` 仍是这套：

```c
static int dirty_background_ratio = 10;
static unsigned long dirty_background_bytes;
static int vm_dirty_ratio = 20;
static unsigned long vm_dirty_bytes;
```
【真实源码 v6.12 路径 mm/page-writeback.c —— 全局默认值】

注意：`*_ratio`（百分比）和 `*_bytes`（绝对字节）是**互斥**的——设了 `dirty_bytes`，对应的 `dirty_ratio` 会被置 0，反之亦然。大内存机器（512G+）用 `ratio=20` 意味着允许 ~100G dirty，远超任何磁盘的吞吐能力，必然在 fsync 时爆出秒级卡顿，所以生产上大内存机普遍改用 `dirty_bytes` 设绝对上限。

限速核心（睡多久）：

```c
static int balance_dirty_pages(struct bdi_writeback *wb,
			       unsigned long pages_dirtied, unsigned int flags)
```
【真实源码 v6.12 路径 mm/page-writeback.c —— 函数签名】

```c
period = HZ * pages_dirtied / task_ratelimit;
pause = period;
if (current->dirty_paused_when)
	pause -= now - current->dirty_paused_when;
```
【真实源码 v6.12 路径 mm/page-writeback.c —— balance_dirty_pages 的 pause 计算】

逐行解读：
- `task_ratelimit`：内核算出来的"这个写进程当前被允许的 dirty 速率（页/秒）"，由 `dirty_ratelimit * pos_ratio` 推出，`pos_ratio` 是位置控制器输出（dirty 越逼近上限，`pos_ratio` 越小，限速越狠）。
- `period = HZ * pages_dirtied / task_ratelimit`：要把刚弄脏的 `pages_dirtied` 个页"摊"到允许速率上，需要这么多 jiffies。
- `pause -= now - dirty_paused_when`：扣掉上一轮已经睡过的时间，避免重复睡——这让限速是**平滑的**而不是"猛睡一大觉"。

文件里那段 `pos_ratio` 控制线注释（"The wb control line won't drop below pos_ratio=1/4..."）正是这套控制器的设计说明：它保证限速曲线平滑，不会在水位刚过线时就把进程一脚踩死。

> **底层根因（为什么是控制论而非 if-else）**：dirty 速率是一个有惯性的系统（flusher 吞吐、设备带宽都在波动），用硬阈值开关会震荡（写进程一会儿全速一会儿全停，产生延迟毛刺）。位置式速率控制本质是给系统加阻尼，把"开关控制"换成"比例控制"，这和 TCP 拥塞控制、CoDel 处理 bufferbloat 是同一类思路。

### 与块层的衔接：wbt（2016，Jens Axboe）

writeback 限速管"产生多少 dirty"，但**已提交的回写 I/O 仍可能把设备队列灌满**，饿死交互读。2016 年 Jens Axboe 的 **writeback throttling (wbt)**（LWN《Toward less-annoying background writeback》，Corbet，2016-04-13）在块层提交时区分"普通持久化 vs 内存回收回写"，给后台回写**限流**，思路明确借鉴网络的 **CoDel / bufferbloat**。这部分在第 16 章详述，这里只点出：**page cache 的 dirty 限速（mm 层）和回写 I/O 限速（块层）是两套独立闸门，调优时要分清你卡在哪一层**。

### writeback 触发条件（v6.12）

| 触发源 | 条件 | 行为 |
|--------|------|------|
| `balance_dirty_pages()` | 进程 dirty 速率 > 回写速率，dirty 逼近软限 | 按 `pos_ratio` 算 `pause` 睡眠限速 |
| flusher 定时 | `dirty_writeback_centisecs`（默认 500=5s）唤醒 | 刷超过 `dirty_expire_centisecs`（默认 3000=30s）的 dirty 页 |
| flusher 比例 | dirty > `dirty_background_ratio`(10%) | 后台开刷，不阻塞前台 |
| 硬阈值 | dirty > `dirty_ratio`(20%) | 前台写入被同步限速（可见卡顿） |
| `sync`/`fsync`/`msync` | 用户主动 | 强制刷入 |

💡 **真坑**：高吞吐写（日志、流式）若 `dirty_ratio` 太低，进程频繁进 `balance_dirty_pages` 睡眠，表现为**写延迟周期性毛刺**。但盲目调高也错——大内存机调高 `dirty_ratio` 会攒出几十 G dirty，一次 fsync 直接秒级停顿。正解：用 `dirty_background_bytes`/`dirty_bytes` 设**绝对值**（如 background=512M、limit=1G），让水位跟磁盘带宽而非内存总量挂钩。

---

## 10.3 LRU 与 refault：page cache 智能的核心（源码精读）

这是本章最该吃透的机制。先讲清楚"为什么不用简单 LRU"，再逐行读 `mm/workingset.c`。

### 为什么不用简单 LRU（失败模式）

朴素 LRU 的致命伤：**一次大顺序读（tar / 备份 / 全表扫描）把整个热工作集驱逐**。因为顺序读的每一页都瞬间变成"最近访问"，挤到链表头部，把真正反复用的热页顶下去。这在数据库机器上就是"跑了个 `SELECT *` 全表扫，之后所有查询变慢"——buffer pool 外的 page cache 被一次性流量冲走了。

Linux 的解法是**双链表 + 二次机会**（Clock-Pro / 2Q 思路的工程化）。每个 NUMA node 的 `lruvec`（在 memcg 场景下是 per-memcg-per-node）维护：

```
lruvec
├── LRU_INACTIVE_ANON   匿名页（堆/栈/私有 mmap）非活跃
├── LRU_ACTIVE_ANON     匿名页活跃
├── LRU_INACTIVE_FILE   文件页（page cache）非活跃   ← 新读入的页落这里
├── LRU_ACTIVE_FILE     文件页活跃                   ← 二次访问才晋升到这
└── LRU_UNEVICTABLE     mlock / 部分 tmpfs huge 等不可驱逐
```

**二次机会**：新读入的 file folio 进 `INACTIVE_FILE` 尾。若在被驱逐前再次被访问（`PG_referenced`/`folio_test_referenced` 已置位），才晋升 `ACTIVE_FILE`。顺序读的页若不再访问，很快从 inactive 头被驱逐，**碰不到 active list，热工作集得保护**。

### 双链表的盲点 → refault distance（v3.15+，Johannes Weiner）

双链表仍有盲点：**一个页被驱逐后多快又被访问？** 如果驱逐后立刻又要，说明 inactive list 开太小、缓存少了一点点就保不住它。但页被驱逐后元数据就没了，怎么"事后"知道它当初差一点就能留下？

Johannes Weiner 的 **workingset detection / refault distance**（v3.15 合入，LWN 有系列文章）给出了一个精巧答案：**驱逐时别清空 slot，塞一个 shadow entry 记账；refault 回来时算账**。

#### 记账单位：nonresident_age

每个 lruvec 有一个单调递增计数器 `nonresident_age`，每从该 lruvec 移走一页（驱逐或晋升）就加：

```c
void workingset_age_nonresident(struct lruvec *lruvec, unsigned long nr_pages)
{
	do {
		atomic_long_add(nr_pages, &lruvec->nonresident_age);
	} while ((lruvec = parent_lruvec(lruvec)));
}
```
【真实源码 v6.12 路径 mm/workingset.c】

逐行：`atomic_long_add` 把本次移走的页数累加进当前 lruvec 的计数；`while ((lruvec = parent_lruvec(lruvec)))` 沿 memcg 层级**向上一路累加到根**——这样父 cgroup 的计数包含子 cgroup 的活动，refault 距离在层级间可比。

#### 驱逐：把"当时的 age"打包进 shadow entry

```c
void *workingset_eviction(struct folio *folio, struct mem_cgroup *target_memcg)
{
	struct pglist_data *pgdat = folio_pgdat(folio);
	unsigned long eviction;
	struct lruvec *lruvec;
	int memcgid;
	if (lru_gen_enabled())
		return lru_gen_eviction(folio);
	lruvec = mem_cgroup_lruvec(target_memcg, pgdat);
	memcgid = mem_cgroup_id(lruvec_memcg(lruvec));
	eviction = atomic_long_read(&lruvec->nonresident_age);
	eviction >>= bucket_order;
	workingset_age_nonresident(lruvec, folio_nr_pages(folio));
	return pack_shadow(memcgid, pgdat, eviction,
				folio_test_workingset(folio));
}
```
【真实源码 v6.12 路径 mm/workingset.c】

逐行：
- `if (lru_gen_enabled()) return lru_gen_eviction(folio);`：**MGLRU 开启时走另一套**（10.10）——一句话点明两套回收实现的分叉口就在这。
- `eviction = atomic_long_read(&lruvec->nonresident_age)`：快照"现在的 age"。
- `eviction >>= bucket_order`：右移分桶，**丢弃低位精度**。为什么？见下面 shadow 编码——XArray value entry 位宽紧张，装不下完整 age，只能粗粒度。
- `pack_shadow(memcgid, pgdat, eviction, workingset)`：把 memcgid、node、age、是否曾属 workingset 打包成一个 XArray value entry，原地塞回被驱逐 folio 的 slot。

#### shadow 编码：在一个 unsigned long 里塞四样东西

```c
#define WORKINGSET_SHIFT 1
#define EVICTION_SHIFT	((BITS_PER_LONG - BITS_PER_XA_VALUE) +	\
			 WORKINGSET_SHIFT + NODES_SHIFT + \
			 MEM_CGROUP_ID_SHIFT)
#define EVICTION_MASK	(~0UL >> EVICTION_SHIFT)
```
【真实源码 v6.12 路径 mm/workingset.c】

```c
static void *pack_shadow(int memcgid, pg_data_t *pgdat, unsigned long eviction,
			 bool workingset)
{
	eviction &= EVICTION_MASK;
	eviction = (eviction << MEM_CGROUP_ID_SHIFT) | memcgid;
	eviction = (eviction << NODES_SHIFT) | pgdat->node_id;
	eviction = (eviction << WORKINGSET_SHIFT) | workingset;
	return xa_mk_value(eviction);
}
```
【真实源码 v6.12 路径 mm/workingset.c】

逐行：从低到高依次拼 `workingset`(1bit) → `node_id`(NODES_SHIFT bits) → `memcgid` → 剩余高位放 `eviction`（age）；`xa_mk_value` 把它标成 XArray value entry（最低位置 1 表示"这是值不是指针"）。`unpack_shadow` 是镜像的逆操作。**这就是 `bucket_order` 右移的代价来源**：`EVICTION_SHIFT` 把一大截高位让给了 memcg/node/标志位与 XArray tag，留给 age 的位不够表达每一次驱逐，只能分桶——精度换"在一个指针位宽里同时编码租户/节点/时序"。

#### refault：算距离、和 workingset size 比，决定是否复活

```c
void workingset_refault(struct folio *folio, void *shadow)
{
	bool file = folio_is_file_lru(folio);
	...
	if (lru_gen_enabled()) {
		lru_gen_refault(folio, shadow);
		return;
	}
	...
	mod_lruvec_state(lruvec, WORKINGSET_REFAULT_BASE + file, nr);
	if (!workingset_test_recent(shadow, file, &workingset, true))
		return;
	folio_set_active(folio);
	workingset_age_nonresident(lruvec, nr);
	mod_lruvec_state(lruvec, WORKINGSET_ACTIVATE_BASE + file, nr);
	if (workingset) {
		folio_set_workingset(folio);
		lru_note_cost_refault(folio);
		mod_lruvec_state(lruvec, WORKINGSET_RESTORE_BASE + file, nr);
	}
}
```
【真实源码 v6.12 路径 mm/workingset.c —— 关键路径节选】

逐行：
- `mod_lruvec_state(..., WORKINGSET_REFAULT_BASE + file, nr)`：先无条件记一笔"发生了 refault"（这就是 `/proc/vmstat` 里 `workingset_refault_file/anon` 的来源——你的监控应该盯这个）。
- `if (!workingset_test_recent(shadow, ...)) return;`：算距离并判定"这次 refault 是否够近、近到说明缓存只差一点点"。不够近 → 这页确实是冷的，正常放回 inactive，**不复活**。
- `folio_set_active(folio)`：判定够近 → **直接置 active**，绕过"必须二次访问才晋升"的常规门槛。这是 refault 的全部威力所在——它让内核"事后追认"一个本该留住的热页。
- `if (workingset) { folio_set_workingset(...); ... WORKINGSET_RESTORE_BASE ... }`：如果它当初就属于稳定工作集，额外记 `restore` 计数并调整 cost，进一步加固。

判定逻辑的核心（"够近"的定义）：

```c
refault_distance = (refault - eviction) & EVICTION_MASK;
workingset_size = lruvec_page_state(eviction_lruvec, NR_ACTIVE_FILE);
if (!file) {
	workingset_size += lruvec_page_state(eviction_lruvec, NR_INACTIVE_FILE);
}
if (mem_cgroup_get_nr_swap_pages(eviction_memcg) > 0) {
	workingset_size += lruvec_page_state(eviction_lruvec, NR_ACTIVE_ANON);
	if (file) {
		workingset_size += lruvec_page_state(eviction_lruvec, NR_INACTIVE_ANON);
	}
}
return refault_distance <= workingset_size;
```
【真实源码 v6.12 路径 mm/workingset.c —— workingset_test_recent 核心比较】

逐行（这是整章最该背下来的一段推理）：
- `refault_distance = (refault - eviction) & EVICTION_MASK`：refault 时的 age 减去驱逐时记下的 age = **这页离开期间，本 lruvec 又移走了多少页**。它衡量"如果当时缓存大 `refault_distance` 这么多页，这页就不会被踢"。
- `workingset_size`：当前能用来装这页的"竞争容量"。注意它的组合很讲究——对 file refault，基线是 `NR_ACTIVE_FILE`；**只有当还有 swap 空间时**才把 anon 的两条链表算进来。为什么？因为没有 swap，anon 页根本不能被回收去给 file 让位，把它算进"可竞争容量"是骗自己。
- `return refault_distance <= workingset_size`：距离 ≤ 容量 → "再多一点缓存就能保住" → 复活成 active。

> **设计哲学**：这等价于让内核**在线估计"最优缓存该多大"**，无需任何人工调 cache size。高并发随机读 → refault 多 → 自动倾向多留 file 页；容器里 memcg limit 卡得紧 → refault 飙升正是 OOM 前兆。这套自适应正是 10.10 里 MGLRU 想用更便宜的方式重做的东西。

🎯 **面试分水岭**："Linux 如何防止顺序读污染 page cache？"——只答"LRU"是初级；答"新页先进 inactive + 二次机会"是中级；能讲出"refault distance 用 nonresident_age 记账、和 workingset size 比较来事后复活热页"，并指出"没 swap 时 anon 不计入竞争容量"，是 staff 级。

---

## 10.4 kswapd 与 direct reclaim：压力层级与 get_scan_count

### 回收的两条路径（watermark 分层）

```
内存分配请求 (alloc_pages)
    │  zone watermark check
    ├── 高于 high ───────▶ 直接拿页，无回收
    ├── 跌破 low ────────▶ 唤醒 kswapd（后台异步回收到 high）
    └── 跌破 min ────────▶ direct reclaim（分配进程上下文同步阻塞）
```

- **kswapd**（每 NUMA node 一个：`kswapd0/1/...`）：主循环是 `balance_pgdat()`（`mm/vmscan.c`），跌破 low 被唤醒，**目标回收到 high watermark**，回收过程不阻塞业务进程。
- **direct reclaim**：入口 `try_to_free_pages()` → `do_try_to_free_pages()`（`mm/vmscan.c`），**在申请内存的那个进程的上下文里同步跑**。一旦触发，该进程的 `alloc` 延迟从 <1μs 膨胀到 ms 级（若要 swap out 更糟）。

`min_free_kbytes`（`/proc/sys/vm/`）是 min watermark 的基准；调高它让 kswapd **更早**介入，给后台回收更多提前量，降低跌破 min 触发 direct reclaim 的概率——这是高并发服务的常用调优。

### scan_control：一次回收的"上下文对象"

所有回收路径都围绕一个 `struct scan_control`（栈上构造，贯穿整条调用链）展开：

```c
struct scan_control {
	unsigned long nr_to_reclaim;
	nodemask_t *nodemask;
	struct mem_cgroup *target_mem_cgroup;
	unsigned long anon_cost;
	unsigned long file_cost;
	unsigned int may_deactivate:2;
	...
	unsigned int may_writepage:1;
	unsigned int may_unmap:1;
	unsigned int may_swap:1;
	...
	unsigned int proactive:1;
	unsigned int memcg_low_reclaim:1;
	...
	s8 order;
	s8 priority;
	s8 reclaim_idx;
	gfp_t gfp_mask;
	unsigned long nr_scanned;
	unsigned long nr_reclaimed;
	...
};
```
【真实源码 v6.12 路径 mm/vmscan.c —— struct scan_control 节选】

关键字段的工程含义：
- `priority`（s8，从 `DEF_PRIORITY=12` 递减到 0）：扫描激进程度。每轮回收不够就把 priority 减 1，扫描比例翻倍；`priority==0` 意味着"扫了所有页还不够"——通常就是 OOM 前夜。
- `may_swap` / `may_writepage` / `may_unmap`：本轮**允许**做哪些昂贵动作。例如 atomic 上下文的回收 `may_writepage=0`（不能在那里发 I/O）。
- `anon_cost` / `file_cost`：内核**实测**的"回收 anon vs file 的代价"，配合 swappiness 决定扫描配比（见下）。
- `proactive`：是否是 `memory.reclaim`（v5.19+ 主动回收接口）触发的——主动回收和被动回收要区分计量。

回收主循环：`shrink_lruvec()` 对每个 lruvec 先调 `get_scan_count()` 算出"四条 LRU 各扫多少页"，再对 inactive 链表调 `shrink_inactive_list()`：

```c
static unsigned long shrink_inactive_list(unsigned long nr_to_scan,
		struct lruvec *lruvec, struct scan_control *sc,
		enum lru_list lru)
```
【真实源码 v6.12 路径 mm/vmscan.c —— 函数签名】

它内部：`isolate_lru_folios()` 从链表批量摘下候选 → `shrink_folio_list()` 真正处理（尝试 unmap、回写 dirty、丢弃 clean、swap out anon）→ 把没回收成的放回链表。

```c
static unsigned int shrink_folio_list(struct list_head *folio_list,
		struct pglist_data *pgdat, struct scan_control *sc,
		struct reclaim_stat *stat, bool ignore_references)
```
【真实源码 v6.12 路径 mm/vmscan.c —— 函数签名】

这是回收的"脏活车间"：决定每个 folio 是直接丢（clean file）、还是等 writeback（dirty file）、还是 swap out（anon）。

### get_scan_count：swappiness 的真正落点

"扫 anon 还是扫 file、各扫多少"全在 `get_scan_count()` 里，用一个 `enum scan_balance` 表达决策：

```c
enum scan_balance {
	SCAN_EQUAL,   /* 按比例算出的份额，anon/file 平权 */
	SCAN_FRACT,   /* 按 anon_cost/file_cost 比例分配（常态）*/
	SCAN_ANON,    /* 只扫 anon */
	SCAN_FILE,    /* 只扫 file */
};
```
【真实源码 v6.12 路径 mm/vmscan.c】

```c
static void get_scan_count(struct lruvec *lruvec, struct scan_control *sc,
			   unsigned long *nr)
```
【真实源码 v6.12 路径 mm/vmscan.c —— 函数签名】

判定优先级（从源码注释提炼，**顺序即优先级**）：
1. **无 swap 空间** → `SCAN_FILE`（"do not bother scanning anon folios"）：连 swap 都没有，扫 anon 毫无意义。
2. **`cgroup_reclaim(sc) && !swappiness`** → `SCAN_FILE`：cgroup 回收且该 cgroup swappiness=0，只扫 file。
3. **`!sc->priority && swappiness`** → `SCAN_EQUAL`：priority 已经压到 0（命悬一线）且允许 swap，anon/file 平权全力扫。
4. **`sc->file_is_tiny`** → `SCAN_ANON`：file 页已经很少了，再扫 file 榨不出油，转去扫 anon。
5. **`sc->cache_trim_mode`** → `SCAN_FILE`：专门修剪 cache 的模式。
6. **默认** → `SCAN_FRACT`：按实测 `anon_cost`/`file_cost` 比例分配扫描份额。

### swappiness=0 的语义巨变（commit fe35004f，3.5-rc1，Satoru Moriya）

这是高频踩坑点，必须讲清"考古级"细节。**3.5 之前**，`get_scan_count` 算 anon/file 扫描份额是：

```
ap = (1 + anon_prio) * (reclaim_stat->recent_scanned[0] + 1);
fp = (1 + file_prio) * (reclaim_stat->recent_scanned[1] + 1);
```
【示意代码，非逐字 —— 3.5 之前的旧 get_scan_count，据 commit fe35004f 与 eklitzke.org 还原】

注意那个 **`1 +`**：即便 `anon_prio=0`（swappiness=0），`ap` 仍是个小正数 → anon 页**仍会被少量扫描回收**。

**commit fe35004f（Satoru Moriya, 3.5-rc1）** 改成：

```
anon_prio = swappiness;
file_prio = 200 - anon_prio;
ap = anon_prio * (reclaim_stat->recent_scanned[0] + 1);
fp = file_prio * (reclaim_stat->recent_scanned[1] + 1);
```
【示意代码，非逐字 —— fe35004f 之后】

去掉了 `1 +`：当 `swappiness=0`，`anon_prio=0` → **`ap` 恒为 0** → 全局回收下**完全不扫匿名页**，直到 `nr_free + nr_filebacked < high_watermark`（file 页快被榨干）才动 anon。**副作用**：哪怕还有大把可回收 page cache，只要它们暂时挪不动，内核也可能**宁愿 OOM kill 也不 swap**。

> **生产结论（为什么 DB 配 1 不配 0）**：正因为 `=0` 在 ≥3.5 内核下把你推向 OOM，社区普遍改用 `swappiness=1`——"几乎不 swap"但保留一条活路。Percona、MySQL 调优文档明确写 `1` 而非 `0`，根因就是这个 commit。能在面试里讲出"`1 +` 被去掉导致 ap 从小正数变成 0"，是这题的天花板答案。

### direct reclaim 的代价与观测

1. 分配延迟从 <1μs → ms 级（要 swap out 更糟）；
2. 高并发下多进程同时 direct reclaim = **回收风暴**（都在抢同一批 LRU、互相 isolate 失败重试）；
3. **观测信号**：`/proc/vmstat` 的 `pgscan_kswapd`（kswapd 扫描）vs `pgscan_direct`（直接回收扫描），或 `sar -B` 的 `pgscank/s` vs `pgscand/s`。**`pgscand/s` 持续 > 0 才是真压力**。

💡 **真坑**：大量团队调优只盯 `free`，不看 `pgscand`。`free` 小是常态（page cache 占着），**direct reclaim 频率**才是病灶。监控里 `pgscan_direct` 上扬 + `workingset_refault` 上扬同时出现，基本可断定"内存不够，正在抖"。

### 回收优先级：谁先死

| 页类型 | 回收代价 | 优先级 |
|--------|----------|--------|
| Clean file page（page cache 未改） | 极低，直接丢（数据还在磁盘） | 最优先 |
| Dirty file page | 要等 writeback 落盘 | 次优先 |
| Anonymous page（无 swap） | 不可回收，只能 OOM | 无 swap 时不可回收 |
| Anonymous page（有 swap） | 需 swap I/O | 低优先（受 swappiness 调） |
| Slab（dentry/inode 等内核缓存） | 可 shrink，但有锁竞争 | 按各 shrinker 优先级 |

工程推论：①DB 机常 `swappiness=1`（不是 0，见上）避免 buffer pool 被换出引发尾延迟；②批量写场景 dirty file 多，回收速度被磁盘带宽钉死，易形成正反馈压力；③`find`/`ls` 大量遍历会撑大 dentry/inode slab，回收主要来源是 slab shrinker（`echo 2 > drop_caches` 清的就是这部分）。

---

## 10.5 Swap 与 Swappiness

### swappiness 的真实语义（接 10.4）

`/proc/sys/vm/swappiness` 默认 **60**。它控制的是 **anon 回收 vs file 回收的相对倾向**（落点就是上面 `get_scan_count` 的 `anon_prio=swappiness`）：

- 值越高 → 越倾向 swap out anon、保留 file cache；
- 值越低 → 越倾向保留 anon、回收 file cache；
- 默认 60 是通用 workload 的历史经验值。

**两个误区**：
- `swappiness=0` ≠ 禁止 swap（禁止用 `swapoff -a`）；且 ≥3.5 内核下 `=0` 反而**更容易 OOM**（见 fe35004f）。
- v5.8（commit c843966c，Johannes Weiner）后 swappiness 上限从 100 提到 **200**：`file_prio = 200 - anon_prio`，所以 swappiness 可以取到大于 100 的值表达"比 file 更优先回收 anon"，给 zram/zswap 场景更大调节空间。

| 场景 | 推荐值 | 原因 |
|------|--------|------|
| 数据库（MySQL/PG） | **1**（不是 0） | 保留 buffer pool，避免 swap 尾延迟；用 1 而非 0 留 OOM 活路 |
| Web（PHP/Node） | 10–30 | 匿名内存轻，可适当换 file cache |
| 桌面 | 60（默认） | 响应优先，换出冷进程 |
| 容器 host | 10–30（需实测） | cgroup 下 swappiness 传导复杂 |
| 配了 zram | 100–200 | 压缩 RAM swap 极快，鼓励 swap anon |

### Swap 的演进

- **swap partition**：固定大小、扩容麻烦；
- **swap file**：灵活可动态扩容（`mkswap` + `swapon`），现代已成熟；
- **zswap / zram**：内存压缩型伪 swap。`zram` 把 swap 放进压缩 RAM，避免真实磁盘 I/O；内存充足但偶发峰值的场景（嵌入式/移动）非常实用。

🎯 **面试加分**：`zram` + 合适压缩算法（lzo-rle 低延迟 / zstd 高压缩比）是移动/嵌入式 Linux 内存管理的标准方案；配上 swappiness=100~200 让内核乐于把 anon 压进 zram，说明有生产视野。

---

## 10.6 OOM Killer：打分、短路与争议（源码精读）

### 触发链路

direct reclaim 彻底失败（无可回收页、swap 也满）后进入 OOM：

```
out_of_memory(oc)             [mm/oom_kill.c]
    → select_bad_process(oc)  遍历所有 task，逐个 oom_badness()，选 score 最高者
    → oom_kill_process()
        → __oom_kill_process()  对目标发 SIGKILL；并由 oom_reaper 异步回收其内存
```

```c
bool out_of_memory(struct oom_control *oc)
static void select_bad_process(struct oom_control *oc)
```
【真实源码 v6.12 路径 mm/oom_kill.c —— 函数签名】

### oom_badness：逐行精读（这是高频错点重灾区）

```c
long oom_badness(struct task_struct *p, unsigned long totalpages)
{
	long points;
	long adj;

	if (oom_unkillable_task(p))
		return LONG_MIN;

	p = find_lock_task_mm(p);
	if (!p)
		return LONG_MIN;

	/*
	 * Do not even consider tasks which are explicitly marked oom
	 * unkillable or have been already oom reaped or the are in
	 * the middle of vfork
	 */
	adj = (long)p->signal->oom_score_adj;
	if (adj == OOM_SCORE_ADJ_MIN ||
			test_bit(MMF_OOM_SKIP, &p->mm->flags) ||
			in_vfork(p)) {
		task_unlock(p);
		return LONG_MIN;
	}

	/*
	 * The baseline for the badness score is the proportion of RAM that each
	 * task's rss, pagetable and swap space use.
	 */
	points = get_mm_rss(p->mm) + get_mm_counter(p->mm, MM_SWAPENTS) +
		mm_pgtables_bytes(p->mm) / PAGE_SIZE;
	task_unlock(p);

	/* Normalize to oom_score_adj units */
	adj *= totalpages / 1000;
	points += adj;

	return points;
}
```
【真实源码 v6.12 路径 mm/oom_kill.c —— 完整函数】

逐行：
- `if (oom_unkillable_task(p)) return LONG_MIN;`：内核线程、init(pid 1) 等不可杀，直接给最小分。
- `p = find_lock_task_mm(p)`：找到这个进程里持有 `mm` 的线程并加锁（多线程共享 mm，只统计一次）。
- **三条短路豁免**：`adj == OOM_SCORE_ADJ_MIN`（即 -1000）/ `MMF_OOM_SKIP`（已被 reaper 收过）/ `in_vfork(p)`（vfork 中，杀了会连累父进程地址空间）→ **直接 `return LONG_MIN`**。这就是 `oom_score_adj=-1000` 的真实机制：**短路，不是"分很低"**。面试若答"-1000 是把分压到很低"是错的，正确是"直接返回 LONG_MIN 无条件出局"。
- `points = get_mm_rss + get_mm_counter(MM_SWAPENTS) + mm_pgtables_bytes/PAGE_SIZE`：**per-mm 三项之和**——常驻内存(RSS) + **被换出的页(MM_SWAPENTS)** + 页表自身占用。
  - **swap 被计入**意味着"换出去就安全了"是错觉——进程被大量换出，OOM 打分里仍背着完整工作集。
  - **页表计入**是为了惩罚"映射极大稀疏地址空间"的进程（页表本身能吃掉可观内存）。
  - **注意它只看 `p->mm`，不递归子进程**。"杀父进程把整棵子树带走"的子树求和逻辑是 **2.6.36 之前老 `badness()`** 的行为，现代内核早改成 per-mm。面试答"会累加子进程内存"会被资深面试官当场纠正。
- `adj *= totalpages / 1000; points += adj;`：把 `oom_score_adj` 归一化到"系统总页数"量纲再叠加。所以 `+1000` ≈ 把"整台机器内存"加到原始分上（几乎必中），`-1000` 走上面的短路（豁免）。范围 -1000 ~ +1000。

观测与实践：

```bash
cat /proc/<pid>/oom_score        # 内核当前算出的分（越高越可能被杀）
cat /proc/<pid>/oom_score_adj    # 可写权重 -1000~+1000
echo -1000 > /proc/<pid>/oom_score_adj   # 关键进程无条件豁免
echo  500  > /proc/<pid>/oom_score_adj   # batch/爬虫优先被杀
dmesg -T | grep -A20 "Out of memory"     # 复盘 OOM（含被杀进程的内存账单）
journalctl -k | grep -i oom
```

### OOM 是权衡决策，不是 bug

| 立场 | 观点 | 适用 |
|------|------|------|
| 拥抱 OOM kill | 快速失败比卡死好；容器化后 kill→restart 幂等 | 无状态服务、微服务 |
| 避免 OOM kill | kill 导致数据不一致；应事前限内存 | 数据库、有状态服务 |

**overcommit 与 OOM 的关系**：Linux 默认 `vm.overcommit_memory=0`（启发式超卖）——`malloc` 通常不真分配物理页（写时 CoW 才分），所以 2GB 机器上 `malloc(4GB)` 能成功返回，**写到临界点才 OOM**。三种模式：

| `vm.overcommit_memory` | 语义 |
|---|---|
| 0（默认） | 启发式，允许合理超卖 |
| 1 | 永远允许（危险，内核保证申请成功） |
| 2 | 严格：超过 `swap + RAM × overcommit_ratio` 即 ENOMEM |

> **争议的历史回响**：OOM killer 的"启发式选择"一直被诟病"杀错进程"（经典抱怨：内存压力来自 A，却把无辜的大进程 B 杀了）。这也是 cgroup v2 引入 `memory.oom.group`（把一个 cgroup 当整体一起杀）和用户态 OOM daemon（如 `systemd-oomd`、Facebook 的 `oomd`，基于 PSI 提前干预）兴起的根因——把"杀谁"的策略从内核启发式上移到用户态可编程策略。

---

## 10.7 设计考古：page → folio（本章最大的一次基础重构）

这一节是为"扎根"专设的——page cache 现在的载体是 folio，不讲清这次转换就只是停在表面。

### 病根：head page / tail page 的类型混淆

`struct page`（4.0 时代）有个长期隐患：一个 **compound page**（如 THP、阶 >0 的高阶页）由一个 **head page** + 若干 **tail page** 组成。很多函数收到 `struct page *` 时，**不知道你给的是 head 还是 tail**，语义全靠口头约定。Matthew Wilcox 的原话（LWN 849538）：

> "A function which has a struct page argument might be expecting a head or base page and will BUG if given a tail page."

代码里到处是 `compound_head(page)` 把 tail 转 head 的防御性调用——既是性能损耗，又是 bug 温床（漏调一次就可能对 tail page 做了只该对 head 做的事）。

### 药方：folio = "保证不是 tail page" 的类型

Wilcox 引入 `struct folio`：一个**编译期保证永远指向 head/base page**的新类型。`mm/mm_types.h` 里 folio 的定义和 page 共享内存布局（union），但类型系统会拦住"把 tail page 当 folio 用"：

```c
struct folio {
	union {
		struct {
			unsigned long flags;
			union {
				struct list_head lru;
				...
			};
			struct address_space *mapping;
			pgoff_t index;
			union {
				void *private;
				swp_entry_t swap;
			};
			atomic_t _mapcount;
			atomic_t _refcount;
			...
		};
		struct page page;          /* 与 struct page 的过渡性 union */
	};
	union {
		struct {
			unsigned long _flags_1;
			unsigned long _head_1;
			atomic_t _large_mapcount;
			atomic_t _entire_mapcount;
			atomic_t _nr_pages_mapped;
			atomic_t _pincount;
#ifdef CONFIG_64BIT
			unsigned int _folio_nr_pages;
#endif
		};
		struct page __page_1;
	};
	...
};
```
【真实源码 v6.12 路径 include/linux/mm_types.h —— struct folio 节选】

文档注释定义（同文件）：

> "A folio is a physically, virtually and logically contiguous set of bytes. It is a power-of-two in size, and it is aligned to that same power-of-two."
【真实源码 v6.12 路径 include/linux/mm_types.h —— folio 注释】

布局兼容靠一组编译期断言保证：

```c
#define FOLIO_MATCH(pg, fl)						\
	static_assert(offsetof(struct page, pg) == offsetof(struct folio, fl))
FOLIO_MATCH(flags, flags);
FOLIO_MATCH(lru, lru);
FOLIO_MATCH(mapping, mapping);
FOLIO_MATCH(compound_head, lru);
FOLIO_MATCH(index, index);
FOLIO_MATCH(private, private);
FOLIO_MATCH(_mapcount, _mapcount);
FOLIO_MATCH(_refcount, _refcount);
...
#undef FOLIO_MATCH
```
【真实源码 v6.12 路径 include/linux/mm_types.h —— FOLIO_MATCH】

逐行意义：`static_assert(offsetof(page,X)==offsetof(folio,Y))` 在**编译期**强制 folio 的字段和 page 对应字段**同偏移**——这样 folio 和 page 可以零成本互转（`&folio->page`），转换期间老代码（吃 page）和新代码（吃 folio）能共存于同一内存对象。注意 `FOLIO_MATCH(compound_head, lru)`：page 的 `compound_head` 和 folio 的 `lru` 同位置——因为 folio 一定是 head，那个本来存"我的 head 是谁"的字段在 folio 里腾出来当 `lru` 用了。

### 社区辩论（设计考古的核心）

合入过程不是一帆风顺（LWN 849538《Clarifying memory management with page folios》, Corbet, 2021-03-18）：

- **支持**：Dave Chinner（XFS maintainer）称这个抽象对文件系统开发者"absolutely necessary"；Kirill Shutemov、Michal Hocko 支持。
- **质疑**：Andrew Morton 担心"a lot of noise"、质疑收益；Hugh Dickins 表达过 skepticism。
- **命名 bikeshed**：连"folio"这个名字都吵过——Wilcox 在定名前试过 `ream / sheaf / quarto / aigle` 等一堆词。这是内核社区著名的"bikeshedding"案例。

合入时间线（据 LWN / kernel.org git / Phoronix）：
- **v5.16**：`struct folio` 类型本体合入（`Merge tag 'folio-5.16'`，commit 49f8275c…，从 Wilcox 的 pagecache 树拉取）。
- **v5.17**：把大量 **page cache / filemap 代码**转成 folio。
- 之后逐子系统推进（writeback、各文件系统的 large folio 支持等）。Wilcox 公布过某些路径 folio 化带来约 **80% 的吞吐提升**（LKML，2021），并报告 page cache 用 large folio 可观降低软件开销。

### 终局：memdesc（为什么这只是中途站）

folio 不是终点。`struct folio` 里那些 `/* the union with struct page is transitional */` 注释泄露了长期目标：**memory descriptors (memdescs)**——把臃肿的 `struct page`（每个物理页都要一个，占内存可观）拆成按用途分化的小描述符（folio 给文件/匿名页、slab 给内核对象、各自只带自己需要的字段）。folio 是通往"瘦身 struct page"的第一步。理解这点，才知道 folio 转换为什么值得社区花数年、改动整个 mm。

> **为什么不能一步到位？** 因为 `struct page` 被几千处代码直接使用，含文件系统、驱动、各 arch 代码。一次性替换是不可能完成的 patch。folio 的 union+FOLIO_MATCH 设计正是为了**让转换可以增量进行、随时可编译、老新代码共存**——这是大型代码库重构的教科书手法（strangler fig 模式的内核版）。

---

## 10.8 cgroup 内存限制与回收（page cache 视角，详见第 22 章）

### v1 vs v2 关键差异

| 项目 | cgroup v1 memory | cgroup v2 memory |
|------|-----------------|-----------------|
| 统计 | `cache`/`rss` 分开 | `memory.current` 统一 |
| 回收触发 | 超 `memory.limit_in_bytes` | 超 `memory.max`（另有 `memory.high` 软限做提前限速回收） |
| OOM | per-cgroup OOM | per-cgroup OOM + `memory.oom.group` 整组杀 |
| page cache 归属 | 可被多 cgroup 共享计入（复杂） | 归属明确，隔离更好 |
| 压力信号 | 无标准接口 | `memory.pressure`（PSI，low/some/full） |

回收路径上，memcg 回收同样走 `shrink_lruvec`，只是 `scan_control.target_mem_cgroup` 被设上，扫描限定在该 memcg 的 lruvec。**refault 检测（10.3）天然 per-memcg**——`nonresident_age` 沿层级累加、`workingset_test_recent` 里用 `eviction_memcg` 取该 memcg 的链表大小做容量，正是为容器隔离设计的。

### 容器三坑

- **坑 1：容器内 `free` 看到 host 内存**（读 `/proc/meminfo`，不感知 cgroup limit）。解法：读 `memory.current`（v2）/ `memory.usage_in_bytes`（v1）。
- **坑 2：page cache 占满 cgroup limit 触发频繁回收**。现象：容器内读文件第一次慢（cache 被本 cgroup 回收），水位逼近 limit 时 CPU 升高（per-cgroup kswapd 式回收在转）。监控：`cat /sys/fs/cgroup/<path>/memory.pressure` 看 `some`/`full`。
- **坑 3：memcg swappiness 行为**（v1 支持 per-cgroup `memory.swappiness`；v2 行为有变化，且依赖 host swap account 配置）「待核：具体取决于内核版本与 swap account 开关」。

> **PSI 是这章和第 22 章的桥**：`memory.pressure` 的 `some`/`full`（v4.20+，Johannes Weiner）直接量化"有多少时间进程因等内存而 stall"，比"水位接近 limit"提前得多，是 `systemd-oomd`/`oomd` 在 OOM 之前主动干预的依据。

---

## 10.9 为什么"free 内存少"通常不是问题

最经典的"理解 Linux 内存哲学"考点。

```
物理 RAM
┌─────────────────────────────────────────────────────┐
│ kernel + slab   │  进程 RSS    │  page cache (file)  │
│   （固定开销）   │  （工作集）  │  （可随时回收）      │
└─────────────────────────────────────────────────────┘
              ↑ 这里"空闲"=0，但 page cache 随时可回收给新分配

free:
              total   used    free   shared  buff/cache  available
Mem:          16384M  8192M   512M    256M     7680M       7424M
                              ↑不重要          ↑可回收      ↑真实可用
```

- **`available` ≈ free + 可回收 buff/cache + 可回收 slab**（内核 `MemAvailable` 是带启发式的估算，不是简单相加）。
- 判断内存紧张：**`available` 趋近 0 且 `pgscan_direct/s` 走高**，才是真信号；只看 `free` 必误判。

---

## 10.10 未来演进：MGLRU（v6.1，一整套替换实现）

### 双链表的根本代价：rmap 逆向扫描

传统 active/inactive 回收要判断"一个页最近有没有被访问"，得通过 **rmap（reverse mapping）** 从物理页反查所有映射它的 PTE，逐个查 access bit。问题：**rmap 逆向遍历在共享页/大内存下极贵**（一个 libc 页可能被上千进程映射，逐个反查 PTE 是 O(映射数)）。refault distance 那套（10.3）虽聪明但实现复杂、维护成本高。

### MGLRU 的思路（Yu Zhao @ Google）

LWN 856931《Multi-generational LRU: the next generation》(Corbet, 2021-05-24) 讲清了核心：

- **多代 generation 取代双链表**：页按"代龄"分多代，最老的代优先回收；访问把页升到最年轻代。"active/inactive"只是"两代"的特例，MGLRU 推广成多代。
- **正向 page-table walk + bloom filter 取代 rmap 逆扫**：不再从页反查 PTE，而是**正向遍历进程页表**批量采集 access bit；用 bloom filter 记住"哪些 mm 最近活跃过"，跳过没必要扫的地址空间。把"逐页鼠标式反查"换成"按页表批量正扫"，在大内存/多共享场景代价低得多。
- **tier（层）+ 类 PID 控制器**：generation 内再按访问频度分 tier，按 per-tier refault 率决定保护谁——这是 10.3 那套 refault 自适应的"更便宜的重做"。

源码层面，MGLRU 和老路径的分叉口在前面已两次看到（`mm/workingset.c`）：

```c
if (lru_gen_enabled())
	return lru_gen_eviction(folio);     // workingset_eviction 内
...
if (lru_gen_enabled()) {
	lru_gen_refault(folio, shadow);     // workingset_refault 内
	return;
}
```
【真实源码 v6.12 路径 mm/workingset.c】

这两处 `lru_gen_enabled()` 分支证明：**MGLRU 与传统双链表是互斥的两套实现**，编译期/启动期二选一，不是运行时随便 A/B 的旋钮。

### 社区辩论（这是 MGLRU 故事的灵魂）

最大顾虑来自 **Johannes Weiner**（现有回收代码的主要维护者，refault detection 的作者），原话（LWN 856931）：

> "It would be impossible to maintain both, focus development and testing resources, and provide a reasonably stable experience with both systems tugging at a complicated shared code base."

他的核心论点：现有回收代码有"billions of hours of production testing and tuning"，新实现必须**替换而非共存**，且要小步演进而不是整块换。这场辩论的结果是 MGLRU 走了**替换**路线（默认仍是传统 LRU，MGLRU 作为可配置开关 `CONFIG_LRU_GEN`），并经过多轮 revision 才在 **v6.1** 由 Andrew Morton 的 mm pull 合入主线（与 Maple Tree 同窗口）。Google 在 ChromeOS 和 Android 上已量产（厂商公布的量级：kswapd CPU 显著下降、低内存 kill 大幅减少、交互延迟改善——具体数字依 workload 而定，不宜当通用承诺）。

```bash
# 查看 / 切换 MGLRU（v6.1+，需 CONFIG_LRU_GEN）
cat /sys/kernel/mm/lru_gen/enabled
# 写入的是 feature bitmask（如 0x0007 全开），不是单纯 0/1
# echo 7 > ... 全开；echo 0 关闭。默认取决于 CONFIG_LRU_GEN_ENABLED
```

> **取舍总结**：MGLRU 把 refault 那套在线自适应换成更便宜的多代+页表扫描，但代价是它是**整套替换**——切换是重编/重启级别的开关，不是生产环境随手调的旋钮。这正印证 know.md 的一条教训："基础设施的'更优实现'往往不是叠加而是替换，迁移成本和共存成本是真实约束。"

### 旁支：THP / File THP 与 io_uring

- **THP（Transparent Hugepage）**：主要服务 anon 页。**file-backed THP / large folio** 在近年随 folio 化逐步完善（folio 天然支持 >PAGE_SIZE 的块）——对 DB 顺序大块读有潜在收益，但 fragmentation 和 compaction 代价要权衡「具体成熟度待核，进展快」。
- **io_uring**：提供 async I/O，但 **buffered I/O 的 page cache 路径本就 async-friendly**。io_uring 的真正收益是减少 syscall 开销（批量提交）+ fixed buffer（减少拷贝），**并不绕过 page cache**（除非配 `O_DIRECT`）。

---

## 10.11 落地：生产排障与调优

### 核心观测命令（每条都标"看什么信号"）

```bash
free -h                          # available 是真实可用；free 小别慌
vmstat 1                         # si/so（swap 出入）、bi/bo（块 I/O）、r/b（运行/阻塞队列）
cat /proc/meminfo | grep -E "Cached|Buffers|Dirty|Writeback|Active|Inactive"
cat /proc/vmstat | grep -E "pgscan_direct|pgscan_kswapd|workingset_refault|workingset_restore|nr_dirty|nr_writeback"
                                 # ↑ 本章最该盯的一行：direct 扫描 + refault 同涨 = 真在抖
sar -B 1                         # pgscank/s(kswapd) vs pgscand/s(direct)；后者>0 是压力
ps aux --sort=-%mem | head -20   # RSS/VSZ 大户
slabtop -o                       # slab（dentry/inode）占用，find/ls 后会涨
iostat -x 1                      # 设备 %util、await，判断回写是否打满设备
```

判读口诀：**`free` 看心情，`available` 看死活，`pgscan_direct` + `workingset_refault` 看是不是真在抖，`iostat await` 看回写卡没卡设备。**

### vmtouch：手动验证/控制 page cache

```bash
apt install vmtouch                 # Debian/Ubuntu（macOS 无 page cache 调优语义，仅演示）
vmtouch /path/to/bigfile            # 查看文件多少在 cache
vmtouch -t /path/to/hotfile         # 预热（读入 cache）
vmtouch -e /path/to/file            # 驱逐（测 cold start）
vmtouch -l /path/to/critical_file   # 锁定不被回收（需 CAP_IPC_LOCK）
```

### 典型排障四例

**例 1：服务重启后首批请求延迟高（cold page cache）**
```bash
vmtouch -t /data/db/hot_table.ibd   # 启动前预热 InnoDB 文件
# 或应用层 posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED)
```

**例 2：内存告警但 `available` 充足（监控读错指标）**
```bash
awk '/^MemAvailable/ {print $2}' /proc/meminfo   # 正确读 MemAvailable，别读 MemFree
```

**例 3：写入周期性卡顿（dirty throttle）**
```bash
bpftrace -e 'tracepoint:writeback:writeback_start { @[comm]=count(); }'
# 确认是 balance_dirty_pages 限速；解法：dirty_background_bytes/dirty_bytes 设绝对值
# 同时 iostat 看设备是否被回写打满（区分卡在 mm 层还是块层 wbt）
```

**例 4：OOM 定位**
```bash
dmesg -T | grep -A20 "Out of memory"   # 含每进程内存账单 + 被选中者 oom_score
dmesg -T | grep "Killed process"
journalctl -f -k | grep -i oom
# 复盘要点：被杀进程的 total_vm/rss/swap 列 + oom_score_adj，对照 oom_badness 公式核验
```

### eBPF/bpftrace 观测（注意函数名随版本漂移）

```bash
# page cache 缺页热点
bpftrace -e 'kprobe:filemap_fault { @[comm]=count(); }
             interval:s:5 { print(@); clear(@); }'

# writeback 回写量
bpftrace -e 'tracepoint:writeback:writeback_pages_written {
               printf("written:%lu by %s\n", args->pages, comm); }'
```
⚠ **暗坑**：`filemap_get_pages`/`filemap_fault` 等是近年内核名，旧版可能不同；上线前 `bpftrace -l 'kprobe:filemap*'` 确认。bpftrace 需 `CAP_BPF`/`CAP_SYS_ADMIN`。`drop_caches` **禁用于生产**（会清掉所有进程的命中率）。

---

## 10.12 面试视角：Staff 级高频题

**Q1：解释 page cache 工作原理，write() 返回后数据是否一定落盘？**
- 框架：write() 写入 folio（v5.16 后载体是 folio 不是 page）→ 置 dirty → 立即返回；flusher 异步回写（定时/比例/expire）；返回 ≠ 落盘，`fsync/fdatasync` 才保证。
- 加分：WAL、`O_SYNC`/`O_DSYNC` 语义差异；`fdatasync` 不刷 inode 元数据（少一次 I/O）。
- 常见不完整答案："write 到 buffer，buffer 定时 flush"——没讲清 page cache 统一地位和 fsync 必要性。

**Q2：free 只剩 200MB、还有 14GB buff/cache，内存紧张吗？怎么判断？**
- 框架：看 `MemAvailable` 不看 free；`vmstat` 的 si/so；`/proc/vmstat` 的 `pgscan_direct`（高=真压力）+ `workingset_refault`（高=缓存不够在抖）；swap 趋势。
- 加分：refault distance 原理、`memory.pressure`(PSI)。

**Q3：kswapd vs direct reclaim？后者为何危险？**
- 框架：kswapd 后台异步（`balance_pgdat`，跨 low 唤醒回收到 high）；direct reclaim 在分配进程上下文同步（`try_to_free_pages`，跌破 min）；后者把 alloc 延迟从 ns→ms，高并发引发回收风暴。
- 加分：调高 `min_free_kbytes` 让 kswapd 提前介入；`pgscan_direct` 观测。

**Q4：OOM 如何选目标？如何保护关键进程？**
- 框架：`oom_badness` = RSS + MM_SWAPENTS + 页表页 + `adj×totalpages/1000`，**per-mm 不累加子进程**（2.6.36 前老 `badness` 才累加，易答错）；`oom_score_adj=-1000` 是 `return LONG_MIN` 短路豁免（不是分很低）。
- 加分：overcommit 模式；cgroup v2 `memory.oom.group`、`systemd-oomd` 基于 PSI 提前干预。

**Q5：swappiness=0 能禁 swap 吗？数据库怎么配？**
- 框架：=0 不禁 swap（禁用 swapoff）；DB 推荐 **1 而非 0**——commit fe35004f（3.5）去掉 `1 +` 后 `ap=0`，page cache 还在也可能先 OOM；=1 几乎不 swap 又留活路。
- 加分：v5.8 后 swappiness 上限提到 200；NUMA 下 `zone_reclaim_mode=0` 避免本地不足时频繁回收。

**Q6：容器内 free 显示充足但应用 OOM crash？**
- 框架：容器内 free 读 `/proc/meminfo`（host 视图）不感知 cgroup；实际限制在 `memory.max`/`limit_in_bytes`。
- 加分：cgroup v2 `memory.pressure`（low/some/full）提前预警；refault 在 memcg 内统计。

**Q7：大文件顺序读如何影响 page cache？内核如何缓解？**
- 框架：顺序读新页进 inactive，不直接污染 active；`posix_fadvise(SEQUENTIAL)` 触发 readahead，`POSIX_FADV_DONTNEED` 读完主动驱逐（rsync/备份）。
- 加分：refault distance 如何事后保护真热页；MGLRU 用页表扫描降低判断成本。

**Q8（进阶）：folio 是什么？为什么内核要从 page 转到 folio？**
- 框架：folio = 保证非 tail page 的类型，消灭 head/tail 混淆与满屏 `compound_head`；union+FOLIO_MATCH 保证零成本互转、增量迁移；v5.16 合入类型、v5.17 转 page cache；终极目标是 memdesc 给 struct page 瘦身。
- 加分：能讲社区辩论（Wilcox 推动、Morton/Dickins 质疑、命名 bikeshed）和"strangler fig"式增量重构。

### 设计题：高吞吐日志写入系统，平衡"写延迟可预测"与"磁盘 I/O 最优"
- buffered write 吞吐最高但延迟不确定（受 `balance_dirty_pages` 限速）；`O_SYNC` 每写 fsync，延迟稳但吞吐低；`O_DIRECT` 绕 page cache，适合自管 buffer（DB buffer pool）。
- 生产方案：buffered write + 周期 fsync（如 Kafka `log.flush.interval`）+ 用 `dirty_background_bytes`/`dirty_bytes` 设**绝对水位**（避免大内存机攒出几十 G dirty 引发秒级 fsync 停顿）+ 块层 wbt 兜底交互读。

### 系统题：内存每小时涨 ~50MB，3 天后 OOM，是泄漏还是 cache 正常增长？
- 步骤：①`free` + `watch awk /proc/meminfo` 区分 RSS 涨 vs buff/cache 涨；②RSS 涨=泄漏，`pmap -x` / `heaptrack` / `/proc/<pid>/smaps_rollup` 定位；③buff/cache 涨=正常，确认 `MemAvailable` 是否同步降；④`pgscan_direct`/`workingset_refault` 是否升（真压力）。
- 量级：50MB/h × 72h = 3.6GB，若 RSS 泄漏 `ps` 能直接看到异常增长。

### 代码题：bpftrace 统计 page cache miss
```bash
sudo bpftrace -e '
kretprobe:filemap_get_pages /comm == "your_process"/ { @miss[comm]=count(); }
kprobe:filemap_get_pages    /comm == "your_process"/ { @total[comm]=count(); }
interval:s:5 { printf("miss=%d total=%d\n", @miss["your_process"], @total["your_process"]); }'
```
暗坑：①`filemap_get_pages` 是近年名，`bpftrace -l 'kprobe:filemap*'` 先确认；②`drop_caches` 勿用于生产；③需 `CAP_BPF`。

---

## 本章心法

> **Linux 把空闲内存都用作 page cache，不是浪费，是主动投资。**
> **page cache 的现代载体是 folio（不是 page）；回收的智能在 refault distance（用 nonresident_age 记账、和 workingset size 比、事后复活热页）；真正的内存压力看 `available` + `pgscan_direct` + `workingset_refault`，不看 `free`；swappiness=0 在 ≥3.5 内核会把你推向 OOM（所以 DB 配 1）；OOM 打分是 per-mm 不累加子进程，-1000 是短路豁免；MGLRU 是替换不是旋钮。**

---

## 本章参考与考据（真正取材到的来源）

源码（逐字取自 raw.githubusercontent.com/torvalds/linux/**v6.12**）：
- `mm/filemap.c`（锁序、`__filemap_get_folio`/`filemap_read`/`filemap_get_pages`/`filemap_fault`）
- `mm/vmscan.c`（`struct scan_control`、`get_scan_count`/`scan_balance`、`shrink_inactive_list`/`shrink_folio_list`）
- `mm/workingset.c`（`nonresident_age`、`workingset_eviction`/`pack_shadow`/`EVICTION_SHIFT`、`workingset_refault`/`workingset_test_recent`）
- `mm/oom_kill.c`（`oom_badness` 完整函数、`out_of_memory`/`select_bad_process` 签名）
- `mm/page-writeback.c`（dirty 默认值、`balance_dirty_pages` 签名与 pause 计算）
- `include/linux/mm_types.h`（`struct folio` 定义、注释、`FOLIO_MATCH`）

史料 / 设计考古：
- LWN 849538《Clarifying memory management with page folios》(Corbet, 2021-03-18)
- LWN 856931《Multi-generational LRU: the next generation》(Corbet, 2021-05-24)
- LWN 682582《Toward less-annoying background writeback》(Corbet, 2016-04-13，wbt/bufferbloat)
- commit **fe35004f**（Satoru Moriya, 3.5-rc1，swappiness=0 不扫 anon）；commit **c843966c**（Johannes Weiner, v5.8，swappiness 上限 200）
- 折叠合入：`Merge tag 'folio-5.16'`（commit 49f8275c，folio 类型合入 v5.16）；MGLRU 合入 v6.1（Andrew Morton mm pull，Phoronix/Kernel.org 记录）
- 旁证：eklitzke.org/swappiness（fe35004f 的 `ap`/`fp` 代码还原）；Percona/MySQL 调优文档（swappiness=1 实践）

---

*交叉引用：块 I/O 路径（bio 提交、I/O 调度、wbt 限流）见第 16 章；虚拟内存地址空间与 mmap、rmap 见第 7 章；物理内存分配（Buddy、Slab、watermark）见第 8 章；cgroup memory controller 与 PSI、per-cgroup 回收见第 22 章。*
