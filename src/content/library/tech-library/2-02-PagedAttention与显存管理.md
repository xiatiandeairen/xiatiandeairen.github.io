---
title: "PagedAttention 与显存管理"
slug: "2-02"
collection: "tech-library"
group: "推理引擎"
order: 2002
summary: "版本：2026-06-12 · 源码取材：vLLM v0.2.0（PagedAttention 首个稳定形态，逐字可读）+ vLLM v1 prefix caching 设计文档 · 论文：Kwon et al., SOSP 2023"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T10:33:13.000Z"
updatedAt: "2026-06-12T10:33:13.000Z"
---
> 版本：2026-06-12 · 源码取材：vLLM v0.2.0（PagedAttention 首个稳定形态，逐字可读）+ vLLM v1 prefix caching 设计文档 · 论文：Kwon et al., SOSP 2023

---

## TL;DR

KV cache 是 LLM serving 的真正瓶颈——不是算力，是显存。第 1 章讲了 KV cache 为什么必须存、它怎么随序列增长。本章解决的是**怎么存**：在一个有几十上百条并发请求、每条长度不可预测、随时生成随时结束的工作负载下，如何把有限的 HBM 切得既不浪费又不卡吞吐。

vLLM 之前的主流做法（Orca、FasterTransformer、HF generate）都给每条请求**预留一段连续显存**到 `max_seq_len`。论文实测：这种做法下**只有 20.4%–38.2% 的 KV cache 显存真正存了 token 状态**，其余 60%–80% 被三种浪费吃掉——reserved（为未来 token 预留但还没用）、internal fragmentation（请求实际长度远短于 max_len）、external fragmentation（不同请求预留块大小不一导致的空洞）。

PagedAttention 的核心洞察来自操作系统：**把 KV cache 像虚拟内存一样分页管理**。逻辑上连续的 token 序列，物理上存在一堆不连续的、固定大小（默认 16 token）的 block 里，靠一张 **block table**（页表）做逻辑→物理映射。这一步把 external fragmentation 干到 0，把 internal fragmentation 限制在「最后一个块」之内（最多浪费 15 个 token 的空间），显存利用率从 ~30% 拉到 ~96%，吞吐提升 2–4×。

更关键的副产品：block 是带 `ref_count` 的共享单位。parallel sampling、beam search、共享 system prompt 的多请求，可以让多条逻辑序列**指向同一批物理 block**，只在某条要写入时才 copy-on-write 复制——这是 prefix caching 和后续 RadixAttention 的地基。

本章按「源码 × demo 双轮」推进：每读一段 vLLM 真实源码，就用一个 numpy/纯 Python toy 把它的算法印证一遍。读完你应该能**徒手写一个能跑的 KV block manager**，并讲清它每一个设计决策的代价。

---

## 前置依赖

读本章你需要先有：
- **第 1 章**的全部内容：Prefill/Decode 两阶段、KV cache 是什么、为什么 Decode 是 memory-bandwidth-bound、KV cache 的形状 `[num_layers, 2, num_heads, seq_len, head_dim]`。
- **操作系统虚拟内存**的基本概念：虚拟地址 vs 物理地址、页表（page table）、分页（paging）、内部/外部碎片。PagedAttention 几乎是这套东西在 GPU 上的逐字搬运，OS 概念越熟读得越快。
- **引用计数（reference counting）**与 copy-on-write 的基本语义（Linux `fork()` 的 COW 是同一个 idea）。
- 一点 CUDA 心智模型（grid/block/warp/thread）足够读懂 kernel 怎么用 block table 寻址；不写 kernel 也能看懂主线。

本章在推理引擎知识树的位置：

```
推理引擎知识树
├── 推理两阶段与 KV 缓存（第 1 章，基础）
├── 【本章】PagedAttention 与显存管理  ← KV cache 的物理管理层，吞吐的地基
│     ├── block 分配/释放/共享（block_manager）
│     ├── COW（parallel sampling / beam search 共享）
│     ├── 抢占：recompute vs swap（scheduler）
│     └── 自动前缀缓存 / RadixAttention（共享的演进）
├── Continuous Batching（调度层，和 PagedAttention 是孪生兄弟，靠它才能动态拼 batch）
├── FlashAttention / 内核优化（Attention 的计算优化，和显存管理正交）
└── Quantization（压缩 KV cache 与 weight，和分页管理叠加）
```

一句话定位：**第 1 章告诉你 KV cache 是什么，本章告诉你它在 HBM 里怎么放才不浪费、怎么共享才省钱。** Continuous batching（调度）和 PagedAttention（显存）是一对孪生兄弟——没有分页管理，调度器没法动态往 batch 里塞新请求；没有调度器，分页省下来的显存也用不上。本章聚焦显存侧，调度侧的连续批处理留给后续章节。

---

## 一、设计考古：为什么会有 PagedAttention

### 1.1 谁先做的、论文怎么来的

PagedAttention 出自论文 **《Efficient Memory Management for Large Language Model Serving with PagedAttention》**，Woosuk Kwon、Zhuohan Li、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica（UC Berkeley + Stanford + UCSD），发表在 **SOSP 2023**（arXiv:2309.06180）。配套开源系统是 **vLLM**，2023 年 6 月 20 日发布首个公开版本。

注意一个史料细节：vLLM 这个**系统**和 PagedAttention 这个**算法**几乎同时出生。论文摘要原话：

> "the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size."
> 【真实摘录 arXiv:2309.06180 abstract】

> "vLLM improves the throughput of popular LLMs by 2-4× with the same level of latency compared to the state-of-the-art systems."
> 【真实摘录 arXiv:2309.06180 abstract】

### 1.2 它解决的瓶颈：KV cache 显存浪费

把背景讲透。LLM serving 的吞吐上限几乎完全由 **能并发多少条请求** 决定，而并发数被 **显存** 卡死。显存里两块大头：

1. **模型权重**：固定的，13B FP16 约 26GB，启动后不变。
2. **KV cache**：随请求数 × 每请求长度线性增长，**这是唯一可优化的变量**。

在一张 40GB A100 上跑 13B 模型，权重吃掉 ~26GB，剩 ~14GB 全给 KV cache。13B 模型每个 token 的 KV cache 约 **800KB**（论文给的数）。14GB / 800KB ≈ 1.8 万 token——这就是你的全部预算，要在所有并发请求间分。预算用得越省，并发越高，吞吐越高。

问题是 vLLM 之前的系统**极度浪费**这块预算。它们沿用「张量必须连续」的思维：给每条请求按 `max_seq_len`（比如 2048）一次性预留一段连续显存。论文 Section 3 给出了关键实测数据：

> "only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems."
> 【真实摘录 arXiv:2309.06180】

vLLM 官方博客说得更直白：

> "We find that existing systems waste **60% – 80%** of memory due to fragmentation and over-reservation."
> 【真实摘录 vLLM blog 2023-06-20】

浪费被论文拆成三类，这三类对应 OS 内存管理里的经典名词，记住它们：

| 浪费类型 | OS 对应 | 在 KV cache 里的含义 | 论文原文 |
|---|---|---|---|
| **Reserved（预留浪费）** | 预分配未使用 | 给请求预留到 max_len，但当前才生成到第 100 token，剩下 1948 个 slot 占着显存却空着 | "pre-allocates memory for maximum possible sequence length" |
| **Internal fragmentation（内部碎片）** | 页内未用空间 | 请求实际只生成 200 token 就 EOS 了，预留的 2048 里 1848 永远用不上 | "severe internal fragmentation, since the request's actual length can be much shorter than its maximum length" |
| **External fragmentation（外部碎片）** | 块间空洞 | 不同请求预留块大小不一（有的 max=512 有的 max=2048），释放后留下大小不一的空洞，新请求拼不进去 | "the pre-allocated size can be different for each request" |

> Figure 2 caption：**"Average percentage of memory wastes in different LLM serving systems during the experiment"**【真实摘录 arXiv:2309.06180】

这三类里，**reserved 是最大头**——它是「连续预留」这个设计的直接后果。只要你坚持「一条请求的 KV cache 必须是一段连续显存」，你就必须按最坏情况（max_len）预留，就必然 reserved 浪费。

### 1.3 关键洞察：借 OS 分页

论文的洞察是一句话级别的优雅：**KV cache 不需要物理连续，只需要逻辑连续。** Attention 计算时我们需要「序列里所有历史 token 的 K/V」，但这些 K/V 在物理显存里在哪、连不连续，计算本身并不关心——只要我们有一张表能找到它们。

这正是 OS 解决进程内存碎片的方式：**虚拟内存 + 分页**。进程看到的是连续的虚拟地址空间，物理上 OS 把它切成固定大小的 page，散布在物理内存各处，靠 page table 做映射。这样进程不需要一整段连续物理内存，碎片问题消失。

vLLM 博客把这个类比讲到了字面对应：

> "One can think of blocks as pages, tokens as bytes, and sequences as processes. The contiguous _logical blocks_ of a sequence are mapped to non-contiguous _physical blocks_ via a block table."
> 【真实摘录 vLLM blog 2023-06-20】

记住这个对应表，本章后面所有代码都是它的实现：

| OS 虚拟内存 | PagedAttention |
|---|---|
| process（进程） | sequence（一条请求/一个序列） |
| page（页，固定大小） | block（KV cache 物理块，默认 16 token） |
| byte（字节） | token 的 K/V |
| page table（页表） | block table |
| virtual address | (logical block idx, offset) |
| physical address | (physical block number, offset) |
| page fault → 分配新页 | 序列增长 → `append_slot` 分配新 block |
| COW fork | parallel sampling / beam search 共享 block |
| swap to disk | swap KV block to CPU memory |

这个借用之所以漂亮，是因为它**一次性**消灭了两类碎片：
- **external fragmentation → 0**：所有 block 大小相同，free list 里任何 block 都能给任何请求用，不存在「拼不进去的空洞」。
- **internal fragmentation → 限制在最后一块**：序列按 token 增长，只有最后那个没填满的 block 有浪费，最多 `block_size - 1 = 15` 个 token 的空间。

代价是引入了一层间接寻址（block table lookup），但这个代价在 GPU 上可以做得很轻——后面读 CUDA kernel 时你会看到它只是多一次 `block_table[block_idx]` 的访存。

---

## 二、真实源码精读：vLLM 的 block 管理

下面读 vLLM **v0.2.0** 的源码。选 v0.2.0 是因为它是 PagedAttention 第一个稳定、逐字可读、没被后来 v1 架构重写搅乱的版本——核心算法和论文一一对应，最适合精读。读懂它，再读 v1（§五）就只是「同一思想的工程加强版」。

数据结构分三层，自底向上：`block.py`（block 本身）→ `block_manager.py`（分配器 + 页表管理）→ `scheduler.py`（抢占策略）。

### 2.1 `block.py`：逻辑块与物理块

这是整个机制的原子单位。**逐字真实源码**：

```python
"""Token blocks."""
from typing import List

from vllm.utils import Device

_BLANK_TOKEN_ID = -1


class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size

        self.token_ids = [_BLANK_TOKEN_ID] * block_size
        self.num_tokens = 0

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def append_tokens(self, token_ids: List[int]) -> None:
        assert len(token_ids) <= self.get_num_empty_slots()
        curr_idx = self.num_tokens
        self.token_ids[curr_idx:curr_idx + len(token_ids)] = token_ids
        self.num_tokens += len(token_ids)

    def get_token_ids(self) -> List[int]:
        return self.token_ids[:self.num_tokens]

    def get_last_token_id(self) -> int:
        assert self.num_tokens > 0
        return self.token_ids[self.num_tokens - 1]


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size

        self.ref_count = 0

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'ref_count={self.ref_count})')
```
【真实源码 vllm-project/vllm@v0.2.0:vllm/block.py】

逐行注解关键点：

- **`LogicalTokenBlock` 不存任何 K/V tensor，只存 `token_ids`**。这是个常被误解的点：逻辑块是「序列结构的影子」，它记录「这一段是哪些 token、填到第几个了」，但真正的 K/V 浮点数据全在物理块对应的那块 HBM 里。逻辑块归 `Sequence` 对象持有（CPU 上的 Python 对象，几乎不占显存）。
- **`token_ids = [_BLANK_TOKEN_ID] * block_size`**（第 30 行）：逻辑块一创建就预分配满 `block_size` 个槽位，用 `-1` 填占位，`num_tokens` 记真实填了几个。`append_tokens` 就是往这个定长数组里顺序写、推进 `num_tokens`。这对应「页大小固定」。
- **`PhysicalTokenBlock` 只有四个字段：device / block_number / block_size / ref_count**（第 60–66 行）。注意它**也不直接持有 tensor**——`block_number` 是它的身份证，真正的 KV tensor 是一个巨大的预分配 pool（形状大致 `[num_blocks, block_size, num_heads, head_dim]`），物理块通过 `block_number` 索引进去（kernel 那一节会看到寻址）。
- **`ref_count` 是整个共享机制的命根子**（第 66 行）。初始 0，被分配时置为正数，被多条序列共享时累加。它就是 OS COW 页的引用计数。后面 `fork` / `free` / COW 全靠它。

> 设计取舍：为什么物理块和逻辑块要分开两个类、各自带 `block_number`？因为**多条逻辑块可以映射到同一个物理块**（共享），也可以一条逻辑块在不同时刻映射到不同物理块（swap/COW 后重映射）。两者生命周期解耦，才有后面所有的灵活性。

### 2.2 `block_manager.py`：分配器 + 页表

`BlockAllocator` 是最纯粹的「物理块 free list」，**逐字真实源码**：

```python
class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)
```
【真实源码 vllm-project/vllm@v0.2.0:vllm/core/block_manager.py】

注解：

- **`allocate` 就是 `free_blocks.pop()` + `ref_count = 1`**——O(1)，从尾部弹一个块出去。`free` 是 `ref_count -= 1`，**减到 0 才真正还回 free list**。这两个方法合起来就是引用计数内存管理的最小实现，和 OS 的物理页分配器是一个模子。
- **OOM 是显式 raise**：free list 空了直接报「Out of memory! No free blocks」。在 serving 里这个异常不会真的抛给用户——调度器会在分配前用 `can_allocate` 预判，预判不过就排队或抢占（§2.4）。
- 注意 GPU 和 CPU 各有一个 `BlockAllocator` 实例，CPU 那个是 swap 的落脚地（§2.4）。

`BlockSpaceManager` 是页表管理器，持有 `block_tables: Dict[int, BlockTable]`（`seq_id → 物理块列表`），这就是页表本体。**逐字真实源码**（核心方法）：

```python
class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        seq = seq_group.get_seqs()[0]
        num_required_blocks = len(seq.logical_token_blocks)
        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return (num_free_gpu_blocks - num_required_blocks >=
                self.watermark_blocks)

    def allocate(self, seq_group: SequenceGroup) -> None:
        seq = seq_group.get_seqs()[0]

        block_table: BlockTable = []
        for logical_idx in range(len(seq.logical_token_blocks)):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
            else:
                block = self.gpu_allocator.allocate()
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                block = self.gpu_allocator.allocate()
                block_table.append(block)
                return None

        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            return None
        else:
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]
```
【真实源码 vllm-project/vllm@v0.2.0:vllm/core/block_manager.py】

这是本章最重要的一段，逐方法拆：

**`can_allocate` —— 准入判断（带 watermark）**。新请求进来先算它要几个 block（`len(seq.logical_token_blocks)`，即 prompt 长度向上取整到 block），再看 GPU free block 够不够，**并且要留 `watermark_blocks` 余量**（默认 1%）。watermark 的作用很微妙：它防止「刚好把显存填满、下一步 decode 连一个 append_slot 的块都分不出来、被迫立刻抢占」的颠簸。这是生产系统才会有的防抖细节，论文里没有。

**`allocate` —— 为 prompt 一次性铺页**。注意第二个 for 循环：`seq_group` 里**所有序列共享同一份 `block_table.copy()`**，而每个物理块的 `ref_count = seq_group.num_seqs()`。这就是 parallel sampling 的起点——同一个 prompt 要采样 N 个不同续写，prompt 部分的 KV 完全相同，于是 N 条序列**共享 prompt 的物理块**，ref_count = N。省下来的就是 (N-1)× prompt 的 KV 显存。

**`append_slot` —— decode 每步的核心，也是 COW 触发点**。每生成一个 token 调一次。三种情况：
1. **逻辑块多于物理块**（`len(block_table) < len(logical_blocks)`）：说明刚跨进一个新 block（前一个填满了），分配一个新物理块 append 进去，返回 `None`（无需 copy）。
2. **最后一块 `ref_count == 1`**（独占）：直接返回 `None`，原地往这个块写新 token 的 KV 即可，零拷贝。
3. **最后一块 `ref_count > 1`**（被共享）：**触发 copy-on-write**。不能在共享块上写（会污染别的序列），于是分配 `new_block`，把 `block_table[-1]` 指向它，对旧块 `free`（ref_count--），返回 `(old_block_number, new_block_number)`——这个 tuple 会被调度器传给 GPU，让它把旧块的内容**物理拷贝**到新块，然后这条序列在新块上写自己的 token。这就是 OS COW 页错误的逐字对应。

**`fork` —— 共享的显式入口**。beam search 里一个 beam 分裂成多个 candidate，或 parallel sampling 派生子序列时调用：child 直接 `copy()` 父序列的 block_table（**共享同一批物理块**），并把每个块 ref_count++。注意 `copy()` 是浅拷贝列表——拷贝的是「指针列表」，物理块本体不动。后续谁要写谁触发 COW。

**`free` —— 归还**。序列结束（EOS）时把它 block_table 里每个块 `free` 一遍（ref_count--，减到 0 才真还），删页表项。

把这几个方法连起来看，你会发现**整个 PagedAttention 的显存管理就是「带引用计数的页表 + COW」**，和 1960 年代的 OS 虚拟内存是同一套数学，只是搬到了 GPU 上、单位从 4KB page 变成 16-token block。

### 2.3 CUDA kernel：block table 怎么在 GPU 上寻址

逻辑→物理映射在 Python 侧建好后，真正算 attention 时 kernel 必须**在 GPU 上**用 block table 把分散的物理块「拼」成一条逻辑序列来算 softmax。这是 PagedAttention 名字里 "Attention" 的部分——它是个**定制的 attention kernel**，不是普通 attention 能直接吃分页 KV 的。

**逐字真实源码**（kernel 签名 + 寻址核心）：

```cuda
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(
  scalar_t* __restrict__ out,
  const scalar_t* __restrict__ q,
  const scalar_t* __restrict__ k_cache,
  const scalar_t* __restrict__ v_cache,
  const int* __restrict__ head_mapping,
  const float scale,
  const int* __restrict__ block_tables,
  const int* __restrict__ context_lens,
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes,
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride)
```
【真实源码 vllm-project/vllm@v0.2.0:csrc/attention/attention_kernels.cu】

寻址核心（从 block table 取物理块号，再算出 K/V 在 pool 里的地址）：

```cuda
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
// ...
const int physical_block_number = block_table[block_idx];
// ...
const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                                + kv_head_idx * kv_head_stride
                                + physical_block_offset * x;
// ...
const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride
                                + kv_head_idx * kv_head_stride;
```
【真实源码 vllm-project/vllm@v0.2.0:csrc/attention/attention_kernels.cu】

主循环（每个 warp 处理一批逻辑块）：

```cuda
for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
  const int physical_block_number = block_table[block_idx];
  // Process tokens within this block...
}
```
【真实源码 vllm-project/vllm@v0.2.0:csrc/attention/attention_kernels.cu】

注解要点：

- **`single_query_cached_kv_attention`** 是 **decode** 阶段的 kernel——一个 query token（刚生成的那个）对全部历史 K/V 做 attention。名字里 "single_query" + "cached_kv" 就是这个意思。（prefill 阶段是 `multi_query`，用 xformers/FlashAttention，见 §2.5 的 `forward`。）
- **`block_tables` 是个扁平的 int 数组**，按 `seq_idx * max_num_blocks_per_seq` 切片定位到某条序列的页表行。这正是 §2.2 那个 Python `Dict[seq_id, BlockTable]` 被「打平成 tensor 传给 GPU」的形态。
- **寻址公式 `k_ptr = k_cache + physical_block_number * kv_block_stride + ...`** 就是页表翻译的硬件版：拿到逻辑位置 → 查 `block_table[block_idx]` 得物理块号 → 乘 stride 算出在 KV pool 里的字节地址。多出来的成本仅仅是一次 `block_table[block_idx]` 的访存 + 一次乘加，**O(1)**。这就是「间接寻址代价很轻」的实锤。
- **warp 切分**：`for (block_idx = warp_idx; ...; block_idx += NUM_WARPS)`——不同 warp 负责不同逻辑块，块内 token 再分给 thread。这种「按 block 并行」的结构正是论文选 `block_size=16` 的原因之一：块太小则 warp 利用不满、并行度低；块太大则 internal fragmentation 涨。16 是 GPU 友好（够一个 warp 干活）和省显存（最多浪费 15 token）之间的平衡点。

> 一个常见误解：「PagedAttention 只是个内存管理技巧」。不对。它是**内存管理 + 配套定制 kernel** 的组合——因为分页破坏了 KV 的物理连续性，标准 attention kernel（假设 K/V 是连续张量）直接用不了，必须重写 kernel 让它能按 block table 跳着读。这也是为什么不是「在任何引擎上加个 block manager 就有 PagedAttention」。

### 2.4 scheduler：显存不够时怎么办——recompute vs swap

分页解决了「怎么省」，但当并发请求的 KV 总量**仍然超过 GPU 显存**时，必须有请求被「踢下车」（抢占 preemption）。vLLM 给了两种策略，**逐字真实源码**：

```python
class PreemptionMode(enum.Enum):
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()

def _preempt(
    self,
    seq_group: SequenceGroup,
    blocks_to_swap_out: Dict[int, int],
    preemption_mode: Optional[PreemptionMode] = None,
) -> None:
    if preemption_mode is None:
        if seq_group.get_max_num_running_seqs() == 1:
            preemption_mode = PreemptionMode.RECOMPUTE
        else:
            preemption_mode = PreemptionMode.SWAP
    if preemption_mode == PreemptionMode.RECOMPUTE:
        self._preempt_by_recompute(seq_group)
    elif preemption_mode == PreemptionMode.SWAP:
        self._preempt_by_swap(seq_group, blocks_to_swap_out)

def _preempt_by_recompute(self, seq_group: SequenceGroup) -> None:
    seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
    assert len(seqs) == 1
    for seq in seqs:
        seq.status = SequenceStatus.WAITING
        self.block_manager.free(seq)
    self.waiting.insert(0, seq_group)
```
【真实源码 vllm-project/vllm@v0.2.0:vllm/core/scheduler.py】

swap 路径靠 block_manager 在 GPU/CPU 两个 allocator 之间搬块，**逐字真实源码**：

```python
def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
    mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
    for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
        new_block_table: BlockTable = []
        block_table = self.block_tables[seq.seq_id]
        for gpu_block in block_table:
            if gpu_block in mapping:
                cpu_block = mapping[gpu_block]
                cpu_block.ref_count += 1
            else:
                cpu_block = self.cpu_allocator.allocate()
                mapping[gpu_block] = cpu_block
            new_block_table.append(cpu_block)
            self.gpu_allocator.free(gpu_block)
        self.block_tables[seq.seq_id] = new_block_table
    block_number_mapping = {
        gpu_block.block_number: cpu_block.block_number
        for gpu_block, cpu_block in mapping.items()
    }
    return block_number_mapping
```
【真实源码 vllm-project/vllm@v0.2.0:vllm/core/block_manager.py】

注解：

- **默认策略选择很关键**：`get_max_num_running_seqs() == 1`（单序列，普通请求）→ **RECOMPUTE**；多序列（beam search / parallel sampling，序列间共享 block，swap 能摊薄成本）→ **SWAP**。
- **RECOMPUTE**：直接 `free` 掉这条序列在 GPU 的所有块，状态打回 `WAITING`，插到 waiting 队列**最前面**（优先恢复）。恢复时把它当新 prompt 重新 prefill。代价是**重算**——但 prefill 是 compute-bound、一次并行，往往比 swap 的来回搬运还快，对单序列尤其划算（没有共享块可摊销）。
- **SWAP**：把 GPU 块的内容拷到 CPU pinned memory 的对应块，GPU 块 `free` 还回 free list，序列状态转 `SWAPPED`。`swap_out` 返回 `{gpu_block_number: cpu_block_number}` 映射，调度器据此发起实际的 D2H 拷贝。恢复时 `swap_in` 反向搬回。代价是 **PCIe 带宽** 的来回搬运 + 占 CPU 内存。
- 注意 `swap_out` 里的 `if gpu_block in mapping` 去重——多条共享同一物理块的序列，只搬一次、ref_count 累加。这是共享语义在 swap 路径上的延续。

这一层是「continuous batching 调度器」的一部分，本章只取它与显存直接相关的抢占/swap 子集；完整的调度循环（怎么决定先跑谁、怎么动态拼 batch）留到调度章。

---

## 三、⭐ 配套可运行 demo：手写一个 KV block manager

> 本域重中之重。下面这个 demo **设计为可运行**，纯 Python + numpy，**无需 GPU**。请在你的环境实际跑一遍。它把 §2.1–2.2 的 vLLM 源码算法（block 分配、append_slot、fork、COW、free、引用计数）用最小代码印证一遍——读完源码你可能「觉得懂了」，跑完 demo 你才**真的**懂 ref_count 在 COW 那一刻是怎么变的。

依赖：`numpy`（`pip install numpy`）。Python ≥ 3.8。无 GPU、无 vLLM、无 CUDA。

```python
# kv_block_manager_demo.py
# 一个最小可运行的 KV block manager toy，印证 vLLM v0.2.0 的分页 KV 管理算法。
# 设计为可运行：python kv_block_manager_demo.py
import numpy as np

BLOCK_SIZE = 4          # 故意调小（vLLM 默认 16），方便观察跨块/碎片
NUM_GPU_BLOCKS = 8      # 整个"显存"只有 8 个物理块
HEAD_DIM = 2            # 玩具维度，真实是 128
NUM_HEADS = 1


# ---------- 物理块：对应 vllm/block.py PhysicalTokenBlock ----------
class PhysicalBlock:
    def __init__(self, block_number):
        self.block_number = block_number
        self.ref_count = 0

    def __repr__(self):
        return f"PB(#{self.block_number}, ref={self.ref_count})"


# ---------- 分配器：对应 vllm BlockAllocator ----------
class BlockAllocator:
    def __init__(self, num_blocks):
        # 真实 KV pool：一大块连续显存，物理块靠 block_number 索引进来
        # 形状 [num_blocks, 2(K/V), block_size, num_heads, head_dim]
        self.kv_pool = np.zeros((num_blocks, 2, BLOCK_SIZE, NUM_HEADS, HEAD_DIM),
                                dtype=np.float32)
        self.free_blocks = [PhysicalBlock(i) for i in range(num_blocks)]

    def allocate(self):
        if not self.free_blocks:
            raise MemoryError("Out of memory! No free blocks.")  # 对应 vLLM 的 OOM
        blk = self.free_blocks.pop()
        blk.ref_count = 1
        return blk

    def free(self, blk):
        assert blk.ref_count > 0, f"Double free! {blk}"
        blk.ref_count -= 1
        if blk.ref_count == 0:           # 减到 0 才真正还回 free list
            self.free_blocks.append(blk)

    def num_free(self):
        return len(self.free_blocks)


# ---------- 页表管理器：对应 vllm BlockSpaceManager ----------
class BlockManager:
    def __init__(self):
        self.alloc = BlockAllocator(NUM_GPU_BLOCKS)
        self.block_tables = {}      # seq_id -> [PhysicalBlock, ...]  这就是"页表"
        self.seq_len = {}           # seq_id -> 已生成 token 数（用于算最后一块填到第几）

    # 新序列：按 prompt 长度一次铺页（对应 allocate）
    def add_seq(self, seq_id, prompt_len):
        n_blocks = (prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # 向上取整
        table = []
        for _ in range(n_blocks):
            table.append(self.alloc.allocate())
        self.block_tables[seq_id] = table
        self.seq_len[seq_id] = prompt_len
        return n_blocks

    # 共享：派生子序列，共享父的物理块（对应 fork）—— parallel sampling / beam search 的起点
    def fork(self, parent_id, child_id):
        src = self.block_tables[parent_id]
        self.block_tables[child_id] = list(src)   # 浅拷贝指针列表，物理块不动
        for blk in src:
            blk.ref_count += 1                     # 每个被共享的块 ref++
        self.seq_len[child_id] = self.seq_len[parent_id]

    # decode 一步：给序列追加一个 token 的 slot（对应 append_slot，含 COW）
    # 返回 None 表示零拷贝；返回 (old#, new#) 表示触发了 COW，需要物理拷贝旧块内容
    def append_slot(self, seq_id):
        table = self.block_tables[seq_id]
        cur_len = self.seq_len[seq_id]
        offset_in_block = cur_len % BLOCK_SIZE

        if offset_in_block == 0:
            # 情况1：刚好跨进新块，需要新分配一个物理块
            new_blk = self.alloc.allocate()
            table.append(new_blk)
            self.seq_len[seq_id] += 1
            return None

        last = table[-1]
        if last.ref_count == 1:
            # 情况2：最后一块独占，原地写，零拷贝
            self.seq_len[seq_id] += 1
            return None
        else:
            # 情况3：最后一块被共享 -> Copy-On-Write
            new_blk = self.alloc.allocate()
            # 物理拷贝：把旧块的 KV 内容复制到新块（真实系统由 GPU kernel 做）
            self.alloc.kv_pool[new_blk.block_number] = \
                self.alloc.kv_pool[last.block_number].copy()
            table[-1] = new_blk            # 这条序列改指向私有新块
            self.alloc.free(last)          # 旧块 ref--（别的序列还在用）
            self.seq_len[seq_id] += 1
            return (last.block_number, new_blk.block_number)

    # 写入某个 token 的 KV（模拟 reshape_and_cache：按页表寻址写进 pool）
    def write_kv(self, seq_id, k_vec, v_vec):
        table = self.block_tables[seq_id]
        pos = self.seq_len[seq_id] - 1     # 刚追加的那个 token
        blk = table[pos // BLOCK_SIZE]
        off = pos % BLOCK_SIZE
        self.alloc.kv_pool[blk.block_number, 0, off, 0] = k_vec   # K
        self.alloc.kv_pool[blk.block_number, 1, off, 0] = v_vec   # V

    # 序列结束：归还所有块（对应 free）
    def free_seq(self, seq_id):
        for blk in set(self.block_tables[seq_id]):   # set 去重：共享块只 free 一次
            self.alloc.free(blk)
        del self.block_tables[seq_id]
        del self.seq_len[seq_id]

    def snapshot(self, tag):
        print(f"\n[{tag}] free_blocks={self.alloc.num_free()}/{NUM_GPU_BLOCKS}")
        for sid, table in self.block_tables.items():
            phys = [b.block_number for b in table]
            print(f"  seq {sid}: len={self.seq_len[sid]:2d}  "
                  f"block_table(logical->physical)={phys}")


# ============================ 跑一遍 ============================
if __name__ == "__main__":
    bm = BlockManager()

    # 1) 新请求 A：prompt 6 个 token -> 需要 ceil(6/4)=2 个块
    bm.add_seq("A", prompt_len=6)
    bm.snapshot("A 入场, prompt=6")

    # 2) parallel sampling：从 A 派生 A1（共享 prompt 的 KV）
    bm.fork("A", "A1")
    bm.snapshot("fork A->A1 (共享 prompt 块, 注意 free 没减少)")
    print("  -> 验证共享: 物理块 ref_count =",
          [b.ref_count for b in bm.block_tables["A"]])

    # 3) A 继续 decode 2 步。第7个 token offset=6%4=2，落在共享的最后一块 -> COW！
    r1 = bm.append_slot("A"); bm.write_kv("A", [1, 1], [1, 1])
    print("\n  A decode 第7个 token, append_slot 返回:", r1, "(非 None = 发生了 COW)")
    r2 = bm.append_slot("A"); bm.write_kv("A", [2, 2], [2, 2])
    print("  A decode 第8个 token, append_slot 返回:", r2, "(此块已私有, None = 零拷贝)")
    bm.snapshot("A 走了2步 (跨到新块前先 COW 了共享块)")

    # 4) A1 也 decode 一步：它的最后一块现在 ref_count==1 了(A 已 COW 走) -> 零拷贝
    r3 = bm.append_slot("A1"); bm.write_kv("A1", [9, 9], [9, 9])
    print("\n  A1 decode 一步, append_slot 返回:", r3, "(A 走后 A1 独占, 零拷贝)")
    bm.snapshot("A1 走了1步")

    # 5) 压测 OOM：不断加新请求直到显存耗尽
    print("\n--- 压测 OOM ---")
    try:
        for i in range(10):
            bm.add_seq(f"X{i}", prompt_len=BLOCK_SIZE)  # 每个吃 1 块
            print(f"  X{i} 入场ok, free={bm.alloc.num_free()}")
    except MemoryError as e:
        print(f"  触发 OOM: {e}  <-- 真实系统这里会改为抢占/排队, 而非抛异常")

    # 6) 释放 A，块回到 free list
    bm.free_seq("A")
    bm.snapshot("free A 之后 (块回收)")
```

**运行步骤**：
```bash
pip install numpy
python kv_block_manager_demo.py
```

**预期输出**（数值里 free_blocks 计数、ref_count、COW 触发点是关键，物理块号因 `pop()` 从尾部取所以是 7,6,5… 递减，属正常）：

```
[A 入场, prompt=6] free_blocks=6/8
  seq A: len= 6  block_table(logical->physical)=[7, 6]

[fork A->A1 (共享 prompt 块, 注意 free 没减少)] free_blocks=6/8
  seq A: len= 6  block_table(logical->physical)=[7, 6]
  seq A1: len= 6  block_table(logical->physical)=[7, 6]
  -> 验证共享: 物理块 ref_count = [2, 2]

  A decode 第7个 token, append_slot 返回: (6, 5) (非 None = 发生了 COW)
  A decode 第8个 token, append_slot 返回: None (此块已私有, None = 零拷贝)

[A 走了2步 (跨到新块前先 COW 了共享块)] free_blocks=5/8
  seq A: len= 8  block_table(logical->physical)=[7, 5]
  seq A1: len= 6  block_table(logical->physical)=[7, 6]

  A1 decode 一步, append_slot 返回: None (A 走后 A1 独占, 零拷贝)
[A1 走了1步] free_blocks=5/8
  seq A: len= 8  block_table(logical->physical)=[7, 5]
  seq A1: len= 7  block_table(logical->physical)=[7, 6]

--- 压测 OOM ---
  X0 入场ok, free=4
  X1 入场ok, free=3
  X2 入场ok, free=2
  X3 入场ok, free=1
  X4 入场ok, free=0
  触发 OOM: Out of memory! No free blocks.  <-- 真实系统这里会改为抢占/排队, 而非抛异常
```

**demo 与源码的对应（这才是 demo 的价值）**：

- `append_slot` 返回 `(6, 5)` 那一刻——**这就是 §2.2 vLLM 源码里 `last_block.ref_count != 1` 分支返回 `(last_block.block_number, new_block.block_number)` 的本体**。A 和 A1 fork 后共享物理块 6/7（ref_count=2），A 要往 block 6 写第 7 个 token，但 block 6 被 A1 也指着——不能脏写，于是 COW：分配新块 5、拷贝 6 的内容、A 改指 5、block 6 的 ref_count 从 2 降到 1（A1 独占）。
- 下一步 A 的 `append_slot` 返回 `None`——因为 block 5 现在 ref_count=1，独占，零拷贝。**印证了「COW 只发生在第一次写共享块，之后该序列就私有了」**。
- A1 后来 decode 返回 `None`——因为 A 已经把 block 6 让出来了，A1 此刻 ref_count=1。**印证了引用计数让共享在「有人写」时优雅退化为独占。**
- OOM 那段印证 `BlockAllocator.allocate` 的 `raise`，并点出真实系统用 §2.4 的抢占替代裸抛异常。

> 把这个 demo 跑通、并能解释「为什么 A 的第 7 个 token 触发 COW 而第 8 个不触发」，你就真正吃透了 PagedAttention 的共享语义。这比读十遍源码有用。

---

## 四、方案对比：naive / paged / 各家共享

### 4.1 显存利用率对照表

| 方案 | KV 显存布局 | external 碎片 | internal 碎片 | 跨请求共享 | 实测 KV 利用率 | 代表系统 |
|---|---|---|---|---|---|---|
| **Naive 连续预留** | 每请求一段连续显存到 max_len | 严重（块大小不一） | 严重（reserved + 实际短于 max） | 无 | **20.4%–38.2%**（论文）/ ~30%（博客） | Orca, FasterTransformer, HF generate |
| **PagedAttention** | 固定大小 block + 页表 | **0** | 仅最后一块（≤15 token） | block 级 COW | **~96%** | vLLM v0.2 |
| **PagedAttention + 自动前缀缓存** | 同上 + 内容哈希复用 block | 0 | 同上 | 跨请求按内容哈希自动复用 | ~96% + 前缀命中省 prefill | vLLM v1, TensorRT-LLM |
| **RadixAttention** | KV 存 radix tree 节点 | 0 | 节点级 | 自动、细粒度（前缀树） | 高 + 多轮对话/共享前缀场景更优 | SGLang |
| **vAttention** | 用 CUDA VMM 在驱动层做连续虚拟地址 + 动态物理页 | 0 | 页级 | 依赖底层 | 接近 paged，但 kernel 无需改写 | vAttention（研究系统） |

数据出处：利用率 20.4%–38.2% 与 ~96% 来自论文 arXiv:2309.06180 与配套实测综述；~30% 与 60%–80% 浪费来自 vLLM blog 2023-06-20。

### 4.2 具体场景跑一遍：parallel sampling 省多少

设一个真实场景：**一个 prompt 采样 N=4 个续写**（OpenAI API 的 `n=4`），prompt 长 1000 token，每个续写 200 token，13B 模型（每 token KV ≈ 800KB）。

- **Naive（无共享）**：4 条序列各存完整 (1000+200) token 的 KV = 4 × 1200 × 800KB ≈ **3.84 GB**。prompt 部分被复制了 4 份。
- **PagedAttention（共享 prompt）**：1000 个 prompt token 只存一份（4 条序列共享，ref_count=4），只有各自的 200 个续写 token 独立。= (1000 + 4×200) × 800KB ≈ **1.44 GB**。
- **省了 ~62%**。与博客「parallel sampling / beam search 省显存 up to 55%、吞吐 up to 2.2×」的量级一致（具体百分比随 prompt/续写比例变化）。

> 这就是 §2.2 `allocate` 里 `block.ref_count = seq_group.num_seqs()` 那一行在生产里的现金价值。prompt 越长、采样数越多、续写越短，省得越多。

### 4.3 不适用边界（点出来才有思考感）

PagedAttention 不是银弹，几个真实的「不划算/不适用」边界：

1. **极短序列 + 极小 batch**：分页的间接寻址和页表维护有固定开销。如果你只跑单条短请求（比如 chatbot 单轮、序列几十 token），naive 连续反而更简单更快——这也是为什么 llama.cpp 这类**端侧单用户**场景长期没上 PagedAttention（它的瓶颈不是并发显存，是单序列延迟）。
2. **block_size 选错会反噬**：太小 → warp 利用不满、kernel 慢、页表长；太大 → internal fragmentation 回归。16 是 GPU serving 的甜点，但长上下文 + 小模型场景可能要调。
3. **需要定制 kernel**：分页破坏 KV 物理连续性，标准 attention kernel 用不了，必须有配套 paged kernel。这是 vAttention 这类研究的动机——它想**保住分页的显存收益，但不改 attention kernel**，办法是用 CUDA 的 Virtual Memory Management（VMM）在**驱动层**给每条序列一段连续的虚拟地址、底下挂动态物理页。代价是依赖较新的 CUDA 驱动特性。
4. **COW 拷贝本身有成本**：高频写共享块的场景（很多 beam 频繁分裂）会触发大量 COW 物理拷贝，吃 GPU 带宽。共享省的是「存」，但 COW 花的是「拷」，要算净账。

---

## 五、演进与现状：从 COW 到自动前缀缓存到 RadixAttention

### 5.1 共享的三个台阶

vLLM v0.2 的共享是**显式**的：你得 `fork`（beam search/parallel sampling 内部调）才共享。它只覆盖「同一个 `seq_group` 内部」的共享。但生产里有个更大的金矿没挖：**不同请求之间的共享**——比如成千上万个请求都带同一段 system prompt、同一份 few-shot 示例、同一个长文档。这些请求的前缀 KV 完全相同，却各算各的、各存各的。

于是演进出**自动前缀缓存（Automatic Prefix Caching, APC）**：不靠显式 fork，而是**按内容自动发现可复用的 block**。vLLM v1 的设计文档原话：

> "we hash each kv-cache block by the tokens in the block and the tokens in the prefix before the block."
> 【真实摘录 vLLM v1 prefix_caching design doc】

机制要点（摘自 v1 设计文档）：
- **链式哈希**：每个 block 的 hash = `hash(父块hash, 本块token_ids, 额外项如 LoRA id / 多模态 hash / cache salt)`。这样 block 被「它自己 + 它之前的全部前缀」唯一标识——两个请求只有前缀完全一致，hash 才相同，才敢复用。
- **复用而非拷贝**：和 COW 的「先共享、写时拷贝」不同，APC 是「**算之前先查哈希表，命中就直接让新请求的 block_table 指向已有 block**，连 prefill 都省了」。
- **LRU 驱逐**：空闲块进一个双向链表 free queue，`"added to the tail of the free queue in the reverse order"`——后面的块哈希了更多 token、更不可能被别人复用，所以先驱逐。驱逐时从头部弹（最久未用），`"Remove[s] the block ID from the cache block"`。【真实摘录 vLLM v1 prefix_caching design doc】

一句话区分两者：**COW 是「同 group 内、写时分裂」，APC 是「跨 group、读时按内容命中」。** 前者省 parallel sampling 的显存，后者省共享前缀的 prefill + 显存。两者叠加。

### 5.2 RadixAttention：把共享做到极致

SGLang 把「跨请求共享」推到了另一种数据结构：**RadixAttention**——KV cache 不再是「一堆固定块 + 哈希表」，而是存进一棵 **radix tree（压缩前缀树）**。

和 PagedAttention/APC 的关键区别（综合多篇技术综述）：
- **vLLM**：固定大小 block + block table（页表思路），APC 在其上加哈希复用，块粒度。
- **SGLang RadixAttention**：用 radix tree 做**自动、细粒度的前缀共享**，共享的前缀（如 system prompt）只算一次，跨 session 复用；树结构天然表达「多请求共享一段前缀、再各自分叉」的拓扑。
- **适用场景**：多轮对话、不可预测的对话分支、高前缀重叠（60%+）的工作负载，RadixAttention 的 TTFT 优势明显；模板化批处理、高并发下 vLLM 把 paged kernel 下沉到 C++/CUDA 扩展、受 Python GIL 影响小，是它的强项。

> 演进主线一句话：**PagedAttention 解决「单请求显存怎么省」→ COW 解决「同 group 内怎么共享」→ APC 解决「跨请求按内容怎么自动复用」→ RadixAttention 把跨请求共享做成一等公民的树结构。** 四者是同一条「KV 复用」脉络上越走越远的台阶，地基都是「KV 不必物理连续，靠映射表找」。

### 5.3 现状（2026）

- **vLLM**：v1 架构（KVCacheManager + KVCacheBlocks，见第 1 章）把分页、前缀缓存、调度统一重构，prefix caching 默认开启；block manager 从 v0.2 的「单一全局」演化为支持多种 KV 布局（含 sliding window、MLA 等不同 attention 变体的差异化分页）。
- **PagedAttention 已成事实标准**：TensorRT-LLM、TGI、LMDeploy 等主流引擎都实现了分页 KV，名字各异（paged KV cache / block KV），思想同源。
- **反思声音**：vAttention（arXiv:2405.04437）论文标题直接是 "Serving LLMs **without** PagedAttention"——它质疑「为了分页而改写每一个 attention kernel」的工程负担，主张用 CUDA VMM 在底层拿到分页收益、上层 kernel 不动。这代表了「PagedAttention 是否是唯一正确路径」的健康争论。结论未定，但说明这个领域仍在演进。

---

## 六、扎根：失败模式 / 生产真坑 / 底层根因

把生产里真会咬人的列出来，每条带根因。

1. **`--gpu-memory-utilization` 调太高 → 频繁 preemption 抖动**。
   - 现象：吞吐忽高忽低，日志刷 "Sequence group ... is preempted"。
   - 根因：vLLM 启动时按这个比例把「权重之外的显存」全切成 KV block。调到 0.95，看似榨干显存，但留给突发长请求的 free block 太少，一有长序列就触发 §2.4 的抢占，recompute/swap 的开销反噬吞吐。`watermark`（§2.2，默认 1%）只能防最后一格的抖动，防不住整体过满。**根因是「静态预切 KV pool 大小」与「动态变化的请求长度分布」之间的张力**。生产里要按 P99 序列长度留余量，而非按平均。

2. **`block_size` 与硬件/模型不匹配 → kernel 慢或碎片回归**。
   - 根因：§2.3 的 warp 切分。block_size=16 是为典型 head_dim=128、A100/H100 调的。换到 head_dim 很小的模型、或长上下文场景，默认值未必最优。改 `--block-size` 后必须重测，不能拍脑袋。

3. **COW 风暴**：beam search width 很大、频繁分叉 → 海量 COW 物理拷贝吃带宽。
   - 根因：§2.2 `append_slot` 情况 3。共享省「存」，但每次写共享块要拷一整块。beam 多、分叉密时，COW 拷贝量可能盖过共享收益。这是「共享不是免费」的具体形态。

4. **前缀缓存的隐形正确性坑：缓存污染**。
   - 根因：§5.1 链式哈希必须把「所有影响 KV 的因素」都纳入 hash。早期实现若漏掉 LoRA id / 多模态输入 / 采样无关的 metadata，会让「看起来前缀相同但实际 KV 不同」的请求错误命中缓存，**输出悄悄错掉且极难复现**。这就是 v1 文档里 `extra hashes`（LoRA id、multimodal hash、cache salt）存在的原因——它们是正确性补丁，不是性能优化。多租户场景尤其要用 cache salt 做隔离，否则租户 A 的缓存可能被租户 B 命中，**是安全问题**。

5. **swap 把延迟尾巴拉长**。
   - 根因：§2.4 SWAP 走 PCIe（D2H + H2D）。被 swap 的请求恢复时要等搬运，TTFT/TPOT 出现长尾。这也是为什么单序列默认 RECOMPUTE 而非 SWAP——重算虽费算力但走的是 GPU 内部，没有 PCIe 往返的尾延迟。

6. **「以为加个 block manager 就有 PagedAttention」**（架构误区）。
   - 根因：§2.3。分页破坏 KV 物理连续，**必须配套 paged attention kernel**。只搬 Python 侧的 block 管理、不改 kernel，attention 根本读不了分散的 KV。这是移植 PagedAttention 到新框架时最常见的低估点。

---

## 七、面试视角：高频问题与答法

- **Q：PagedAttention 解决了什么问题？为什么不是算力问题？**
  A：解决 KV cache 显存浪费。LLM serving 吞吐由并发数决定、并发由显存卡死、显存里唯一可优化的是 KV cache。旧系统连续预留到 max_len，只有 20.4%–38.2% 显存真存了 token，其余被 reserved/internal/external 三类碎片吃掉。PagedAttention 把利用率拉到 ~96%，吞吐 2–4×。

- **Q：它和 OS 虚拟内存的对应关系？**
  A：sequence↔process、block↔page、token↔byte、block table↔page table、append_slot↔page fault、共享↔COW fork、swap↔swap to disk。外部碎片归零（块等大），内部碎片限制在最后一块（≤block_size-1 token）。

- **Q：parallel sampling / beam search 怎么省显存？COW 何时触发？**
  A：同 prompt 的多条序列共享 prompt 的物理块（ref_count=N），只在某条要**写**共享块时触发 COW——分配新块、拷贝旧块内容、改指向、旧块 ref_count--。能讲清「读共享、写分裂」就到位。（可现场用 §三 demo 的 `(6,5)` 那个例子讲。）

- **Q：显存不够时怎么办？recompute 和 swap 怎么选？**
  A：抢占。单序列默认 recompute（重 prefill，走 GPU 内部、无 PCIe 尾延迟、prefill compute-bound 往往更快）；多序列默认 swap（有共享块可摊销搬运成本，搬到 CPU pinned memory）。

- **Q：block_size 为什么默认 16？调大调小的代价？**
  A：GPU 友好（一个 warp 够干活、并行度足）和省显存（最多浪费 15 token）的平衡。太小则 warp 利用不满、kernel 慢、页表长；太大则 internal fragmentation 回归。

- **Q：PagedAttention 和自动前缀缓存、RadixAttention 的关系？**
  A：同一条「KV 复用」脉络的台阶。PagedAttention=单请求省显存；COW=同 group 写时分裂；APC=跨请求按内容哈希自动命中（连 prefill 都省）；RadixAttention=把跨请求共享做成 radix tree。地基都是「KV 不必物理连续、靠映射表找」。

- **Q（进阶）：PagedAttention 一定要改 attention kernel 吗？有没有别的路？**
  A：经典实现要——分页破坏物理连续，标准 kernel 读不了，必须重写 paged kernel。vAttention 提出另一条路：用 CUDA VMM 在驱动层给每条序列连续虚拟地址、底下挂动态物理页，从而拿到分页的显存收益但 kernel 不用改。代价是依赖较新驱动特性。能答到这层说明你跟了前沿。

---

## 八、未来方向

- **底层分页下沉到驱动（vAttention 路线）**：把「分页」从应用层（改 kernel）移到 CUDA VMM 层，让显存管理和 attention kernel 解耦。若成熟，新 attention 变体（各种 sparse/linear attention）就不必各自重写 paged 版本。
- **KV cache 卸载分层化**：GPU HBM → CPU DRAM → NVMe/远端，按访问热度分层（类似 OS 多级 swap）。长上下文、超长会话场景的刚需，已有引擎在做 KV offload 到 CPU/SSD。
- **共享 + 压缩叠加**：分页管理 + KV 量化（FP8/INT4）+ 前缀复用三者组合。本章是「怎么放」，量化是「放多大」，正交可叠。
- **跨实例 KV 复用**：把前缀缓存从单机推到集群（prefix-aware routing），让同前缀请求路由到已缓存该前缀的实例——SGLang/分布式 serving 正在这个方向。
- **PagedAttention 是否被取代**：随着 attention 形态多样化（MLA、sliding window、linear attention），「固定块 + 页表」是否仍是最优 KV 容器，是开放问题。本章给的是「2023–2026 的事实标准」，不是终点。

---

## 五件套

### 1. 一句话总结
PagedAttention 把 KV cache 当虚拟内存分页管理——逻辑连续、物理分散、靠带引用计数的 block table 映射，从而把显存利用率从 ~30% 拉到 ~96%、消灭外部碎片、把内部碎片关进最后一块，并让 parallel sampling/beam search 靠 COW 共享 prompt KV。

### 2. 必记数字
- 旧系统 KV 利用率 **20.4%–38.2%**，浪费 **60%–80%**（reserved/internal/external 三类）。
- PagedAttention 利用率 **~96%**，吞吐 **2–4×**。
- 默认 **block_size=16** token；13B 每 token KV ≈ **800KB**；最后一块最多浪费 **15** token。
- parallel sampling/beam search 省显存 **up to 55%**、吞吐 **up to 2.2×**。
- 来源：论文 arXiv:2309.06180（SOSP 2023）、vLLM blog 2023-06-20。

### 3. 一张图（文字版）
```
逻辑视图(连续)          block table(页表)         物理视图(分散, 可共享)
seq A: [tok0..tok5][tok6..]  A: [#7, #6, #5]   ┐    KV pool (HBM):
seq A1:[tok0..tok5]          A1:[#7, #6]       ├──> #5 #6 #7 ... (固定大小块)
                                   ↑ ↑              ref_count: #7=2(A,A1共享 prompt)
                              共享 prompt 块            #6=1, #5=1(A 写时 COW 出来)
写共享块 → COW: 分配新块+拷贝+改指向+旧块ref--
显存不够 → 抢占: 单序列 recompute / 多序列 swap 到 CPU
```

### 4. 自测题
1. 为什么「连续预留到 max_len」会同时造成 reserved、internal、external 三种浪费？哪种最大？
2. `append_slot` 的三个返回分支分别在什么情况触发？哪个返回非 None、为什么？
3. fork 之后两条序列共享物理块，此时其中一条 decode 写到共享块，ref_count 怎么变？画出来。
4. 单序列抢占为什么默认 recompute 而不是 swap？从延迟和算力两个角度答。
5. block_size 从 16 调到 256，显存和 kernel 各会怎样？调到 1 呢？
6. 自动前缀缓存（APC）和 COW 的本质区别是什么？为什么 APC 的 block hash 必须包含 LoRA id？
7. 为什么说「光加个 block manager 不等于实现了 PagedAttention」？缺了什么？

### 5. 代码题（= 扩展 demo / 实现挑战）
1. **【印证源码】** 给 §三 的 demo 加 `swap_out(seq_id)` / `swap_in(seq_id)`：实现一个 `cpu_allocator`，把某序列的 block 从 GPU 搬到 CPU（拷贝 kv_pool 数据）、GPU 块归还，再搬回。对照 §2.4 的 vLLM `swap_out` 源码，验证你的 `{gpu_block#: cpu_block#}` 映射和共享块去重逻辑一致。
2. **【watermark 防抖】** 给 demo 加 `can_allocate`（带 watermark）和一个简单调度循环：当 `can_append_slot` 不满足时，按「单序列 recompute、多序列 swap」抢占一条最年轻的序列。复现 §六坑 #1 的抖动：把 watermark 设 0 vs 设 0.1，观察抢占频率差异。
3. **【自动前缀缓存】** 给 demo 加一个 `block_hash -> PhysicalBlock` 的哈希表：新序列 add 时，对每个满块算链式 hash（`hash(父hash, 本块token_ids)`），命中则直接共享已有块（ref++）而不分配新块。喂两个共享前 8 token 的序列，验证第二个序列的前两块直接命中、free_blocks 不减少。这就是把 vLLM v1 的 APC 思想在 toy 上实现一遍。
4. **【paged attention kernel toy】** 用 numpy 写一个「按 block table 寻址的 single-query attention」：给定 query 向量、block_table、kv_pool，按 §2.3 的寻址公式（`physical_block_number * block_stride + offset`）把分散的 K/V 拼出来算 softmax(qKᵀV)，和「连续 KV 直接算」的结果做 `np.allclose` 对比，验证分页不改变数学结果、只改变寻址。
5. **【碎片可视化】** 模拟 naive 连续预留 vs paged：随机生成 100 个不同 max_len/实际 len 的请求，分别用「连续预留」和「分页」两种分配器跑一遍，统计两者的真实利用率，复现论文「~30% vs ~96%」的量级差距，并画出 external fragmentation 在 naive 下随请求增删如何累积。

---

> 取材来源（本章实际 WebFetch/WebSearch 成功取得）：
> - 论文 arXiv:2309.06180（abstract + 正文 HTML 版 ar5iv）
> - vLLM 源码 v0.2.0：`vllm/block.py`、`vllm/core/block_manager.py`、`vllm/core/scheduler.py`、`vllm/model_executor/layers/attention.py`、`csrc/attention/attention_kernels.cu`（均经 raw.githubusercontent.com 逐字取得）
> - vLLM 官方博客 2023-06-20
> - vLLM v1 prefix caching 设计文档（docs.vllm.ai）
> - RadixAttention / block_size 取舍 / vAttention 综述（WebSearch）
