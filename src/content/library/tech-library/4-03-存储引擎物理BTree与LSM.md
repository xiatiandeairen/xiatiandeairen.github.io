---
title: "存储引擎物理层：B-Tree 与 LSM"
slug: "4-03"
collection: "tech-library"
group: "数据库"
order: 4003
summary: "TL;DR 数据库的\"存储引擎物理层\"回答一个问题：键值对在磁盘上怎么摆、怎么改、怎么读。两条主线统治了 50 年： - B-Tree（Bayer & McCreight, 1972）：原地更新（update-in-place）的页式平衡树。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T10:55:45.000Z"
updatedAt: "2026-06-12T10:55:45.000Z"
---
> **TL;DR**
> 数据库的"存储引擎物理层"回答一个问题：键值对在磁盘上怎么摆、怎么改、怎么读。两条主线统治了 50 年：
> - **B-Tree（Bayer & McCreight, 1972）**：原地更新（update-in-place）的页式平衡树。读路径短（一次点查 = 树高次 I/O），但每次写都要把整个磁盘页改写回去，随机写工作负载下**写放大严重**，且页内碎片化。代表：PostgreSQL、MySQL InnoDB、几乎所有 SQL 引擎。
> - **LSM-Tree（O'Neil et al., 1996）**：追加写（append-only）+ 后台归并（compaction）。把随机写转成顺序写、批量落盘，**写放大可控、写吞吐极高**；代价是读要查多层、空间有冗余、后台 compaction 抢 I/O。代表：LevelDB、RocksDB、Cassandra、HBase、ScyllaDB。
>
> 核心权衡是 **RUM 猜想**里的 Read / Write / Space 三角：你不可能同时把读放大、写放大、空间放大都压到最低，B-Tree 偏读、LSM 偏写。本章用 LevelDB **真实源码**逐行拆 LSM 的 memtable / SSTable / compaction，并给两个**可运行的 Python toy demo**（手写 toy LSM + B-Tree vs LSM 写放大对照），让你把"写放大"从一个名词变成你能在自己机器上跑出来的数字。
>
> **前置依赖**
> - 你知道 page / block、顺序 I/O vs 随机 I/O 的量级差异（HDD 随机 IOPS ~100，顺序 ~100MB/s；SSD 随机也远慢于顺序且有写寿命问题）。
> - 你写过或读过任意一个 KV 接口（put/get/delete）。
> - Demo 需要 **Python 3.8+，纯标准库**，无需第三方依赖。
> - 看源码片段需要能读 C++（LevelDB 是 C++17），但本章对每段都做了逐行中文注解。

---

## 0. 本章地图

```
                       存储引擎物理层
                            │
        ┌───────────────────┴───────────────────┐
   原地更新 (in-place)                     追加写 (out-of-place)
        │                                       │
     B-Tree / B+Tree                        LSM-Tree
        │                                       │
  ┌─────┴─────┐                    ┌────────────┼────────────┐
 节点=页    分裂/合并            memtable    SSTable      compaction
 树高=I/O次数  WAL              (skiplist)  (有序不可变)  (leveled/tiered)
        │                                       │
  PostgreSQL / InnoDB              LevelDB / RocksDB / Cassandra
```

阅读顺序建议：先看 §1（为什么会有这两条路），再看 §3 的真实源码（LSM 怎么实现的），然后**务必动手跑 §4 的 demo**，最后用 §5 的对照表和 §6 的生产坑收口。

---

## 1. 背景与设计考古：两次"I/O 成本"的论证

### 1.1 B-Tree：为机械磁盘的随机访问而生（1972）

B-Tree 由 **Rudolf Bayer 和 Edward M. McCreight** 在 **Boeing Research Labs** 发明，论文 *"Organization and Maintenance of Large Ordered Indexes"* 1970 年 7 月首次内部流传，1972 年正式发表于 *Acta Informatica* 1, 173–189（已 WebFetch 核实，见文末出处）。

它要解决的核心问题，论文原话（已核实大意）：为一个**动态随机访问文件**维护索引，且**索引大到必须放在磁盘/磁鼓这类伪随机访问的后备存储上**，要让检索、插入、删除的代价正比于 `log_k(I)`（`I` 是索引大小，`k` 是设备相关的自然数 = 一页能放多少键）。

设计动机的本质是 **磁盘的访问粒度是页（block），不是字节**：

- 一次磁盘 I/O 读一整页（典型 4KB～16KB），CPU 读一个字节和读一整页的磁盘成本几乎一样。
- 所以应该让**一个节点 = 一个页**，节点里塞尽可能多的键（fanout 大到几百上千），让树尽可能**矮而宽**。
- 树高 `h ≈ log_fanout(N)`。fanout=500、十亿条记录时 `h≈4`，意味着**任意一条记录最多 4 次磁盘 I/O 就能定位**。这就是 B-Tree 统治 OLTP 点查的根本原因。

> 关于命名：Bayer 和 McCreight 从未解释 "B" 代表什么——Boeing / Balanced / Bayer / Broad / Bushy 都有人猜过（Wikipedia 已核实）。

**B+Tree** 是后来的工业改良（数据只存叶子、叶子用链表串起来便于范围扫描），PostgreSQL、InnoDB 用的都是 B+Tree。本章谈"B-Tree"时默认指这一族。

### 1.2 LSM-Tree：为高插入率工作负载翻盘（1996）

26 年后，**Patrick O'Neil、Edward Cheng、Dieter Gawlick、Elizabeth O'Neil** 在 *Acta Informatica* 33(4): 351–385 (1996) 发表 *"The Log-Structured Merge-Tree (LSM-Tree)"*（已 WebFetch 核实 PDF：cs.umb.edu/~poneil/lsmtree.pdf）。

它对 B-Tree 的攻击点非常具体——论文摘要原话（已核实）：

> "Standard disk-based index structures such as the B-tree will effectively **double the I/O cost** of the transaction to maintain an index such as this in real time, increasing the total system cost up to **fifty percent**."

翻译：对一个持续高速插入的文件，B-Tree 实时维护索引会让事务的 I/O 成本**翻倍**，整体系统成本上升 50%。

为什么会翻倍？这就是本章最重要的概念之一——**写放大（write amplification）**。当你向 B-Tree 插入一条 100 字节的记录：

1. 先得**随机读**出目标叶子页（4KB）——因为要保持有序、要找到插入位置。
2. 改完再把**整页 4KB 随机写**回去。
3. 如果页满了还要分裂，可能再写一两个页 + 更新父节点。

100 字节的逻辑写，落盘可能是几 KB 的物理写，而且是**随机** I/O。在机械盘时代随机 I/O 是稀缺资源（~100 IOPS），这直接卡死了高插入吞吐。

LSM 的破局思路是 **out-of-place / append-only**：

- **C₀ 组件**：内存里的有序结构（论文里是平衡树，工业界常用 skiplist），吸收所有新写。
- **C₁, C₂, ... Cₖ 组件**：磁盘上逐级变大的有序结构。
- **rolling merge（滚动归并）**：C₀ 满了不是逐条写盘，而是和 C₁ **批量顺序归并**，写出新的 C₁。随机写被转成了**顺序写**。

> 论文用 **five-minute rule**（Gray & Putzolu 的经济学分析）论证 C₀ 该多大：如果一条数据 5 分钟内会被再次访问，放内存比反复磁盘 I/O 更划算。这把"内存该买多大"变成了一道**成本最优化**题。这一点在今天 RAM 便宜、SSD 普及后阈值变了，但"用内存缓冲把随机写转顺序写"的内核思想完全成立。

### 1.3 一句话史观

- **1972 B-Tree**：磁盘随机 I/O 贵 → 矮宽树减少 I/O **次数** → 读快，写放大被接受。
- **1996 LSM**：插入吞吐是瓶颈 → 缓冲 + 顺序批写减少**随机** I/O → 写快，读放大/空间放大被接受。
- **2010s SSD 时代**：随机读不再那么贵，但 **SSD 写寿命有限 + 顺序写更友好**，LSM 的"少写、顺序写"反而更契合闪存（这正是 WiscKey 2016 进一步优化的前提，见 §7）。

两者不是谁淘汰谁，而是 **RUM 三角**（Read / Write / Space amplification，Athanassoulis et al. 2016 提出的 RUM Conjecture）上的两个不同站位。后文 §5 用数据说话。

---

## 2. 演进：从论文到 LevelDB 的工程形态

LevelDB（Google，Jeff Dean & Sanjay Ghemawat，2011 开源）是 LSM 思想最干净、最值得精读的开源实现，RocksDB（Facebook fork）、众多 KV 引擎都源于它。它把论文里抽象的 C₀/C₁/.../Cₖ 落成了一套具体机制：

| 论文概念 | LevelDB 工程形态 | 源码位置 |
|---|---|---|
| C₀（内存组件） | **MemTable**（skiplist）+ 不可变 immutable memtable | `db/memtable.cc`, `db/skiplist.h` |
| 落盘的有序文件 | **SSTable**（`.ldb` 文件，block 化 + index + footer） | `table/`, `doc/table_format.md` |
| C₁...Cₖ 的分层 | **Levels**（L0..L6，共 7 层），每层比上层大 10 倍 | `db/version_set.cc`, `db/dbformat.h` |
| rolling merge | **Compaction**（size-tiered 触发 + seek 触发） | `db/version_set.cc` |
| 崩溃恢复 | **WAL**（write-ahead log，先写 log 再写 memtable） | `db/log_writer.cc` |

下面进入真实源码。

---

## 3. 真实源码精读：LevelDB 的写/读/归并三条路径

> 所有代码块标注【真实源码 repo@path】的均已通过 WebFetch 从 `raw.githubusercontent.com/google/leveldb/main/...` 取回，逐字核实。简化或我自己写的标【示意，非逐字】。

### 3.1 写路径起点：MemTable::Add —— 内部 key 编码是灵魂

LSM 的删除不是真删（不能去磁盘上改不可变文件），而是写一条**墓碑（tombstone）**；更新也不是覆盖，而是写一条更新版本。如何区分新旧、如何区分"值"和"删除"？答案在每条记录的 **internal key 编码**里。

【真实源码 google/leveldb@db/memtable.cc】

```cpp
void MemTable::Add(SequenceNumber s, ValueType type, const Slice& key,
                   const Slice& value) {
  // Format of an entry is concatenation of:
  //  key_size     : varint32 of internal_key.size()
  //  key bytes    : char[internal_key.size()]
  //  tag          : uint64((sequence << 8) | type)
  //  value_size   : varint32 of value.size()
  //  value bytes  : char[value.size()]
  size_t key_size = key.size();
  size_t val_size = value.size();
  size_t internal_key_size = key_size + 8;
  const size_t encoded_len = VarintLength(internal_key_size) +
                             internal_key_size + VarintLength(val_size) +
                             val_size;
  char* buf = arena_.Allocate(encoded_len);
  char* p = EncodeVarint32(buf, internal_key_size);
  std::memcpy(p, key.data(), key_size);
  p += key_size;
  EncodeFixed64(p, (s << 8) | type);
  p += 8;
  p = EncodeVarint32(p, val_size);
  std::memcpy(p, value.data(), val_size);
  assert(p + val_size == buf + encoded_len);
  table_.Insert(buf);
}
```

**逐行注解：**

- `SequenceNumber s`：全局单调递增的序列号。**这是 MVCC 和"新值压旧值"的关键**——同一个 user key 的不同版本靠 sequence 排序。
- `ValueType type`：取值见 `db/dbformat.h`【真实源码 google/leveldb@db/dbformat.h】：
  ```cpp
  enum ValueType { kTypeDeletion = 0x0, kTypeValue = 0x1 };
  ```
  `kTypeDeletion` 就是墓碑。删除一个 key = `Add(seq, kTypeDeletion, key, "")`。
- `internal_key_size = key_size + 8`：internal key = user key + 8 字节 tag。
- `EncodeFixed64(p, (s << 8) | type)`：**把 56-bit 的 sequence 左移 8 位，低 8 位塞 type**，拼成一个 64-bit tag。注意 sequence 在高位——这样 internal key 比较时，**同一 user key 下 sequence 大的（更新的）排在前面**（配合比较器对 tag 做降序）。
- `table_.Insert(buf)`：插进 skiplist。整条记录是一块连续内存（arena 分配），skiplist 节点只存这块内存的指针。

**这段代码隐含了 LSM 的两个本质设计：**
1. **写永远是 append**：put / delete / update 在内存层面都是一条新记录，从不原地改。这就是为什么 LSM 写放大低——内存写没有"先读后写整页"。
2. **版本靠 sequence 排序，读时取最新**：见下面的 Get。

### 3.2 读路径：MemTable::Get —— 为什么 Seek 能直接拿到最新版本

【真实源码 google/leveldb@db/memtable.cc】

```cpp
bool MemTable::Get(const LookupKey& key, std::string* value, Status* s) {
  Slice memkey = key.memtable_key();
  Table::Iterator iter(&table_);
  iter.Seek(memkey.data());
  if (iter.Valid()) {
    // entry format is:
    //    klength  varint32
    //    userkey  char[klength]
    //    tag      uint64
    //    vlength  varint32
    //    value    char[vlength]
    // Check that it belongs to same user key.  We do not check the
    // sequence number since the Seek() call above should have skipped
    // all entries with overly large sequence numbers.
    const char* entry = iter.key();
    uint32_t key_length;
    const char* key_ptr = GetVarint32Ptr(entry, entry + 5, &key_length);
    if (comparator_.comparator.user_comparator()->Compare(
            Slice(key_ptr, key_length - 8), key.user_key()) == 0) {
      // Correct user key
      const uint64_t tag = DecodeFixed64(key_ptr + key_length - 8);
      switch (static_cast<ValueType>(tag & 0xff)) {
        case kTypeValue: {
          Slice v = GetLengthPrefixedSlice(key_ptr + key_length);
          value->assign(v.data(), v.size());
          return true;
        }
        case kTypeDeletion:
          *s = Status::NotFound(Slice());
          return true;
      }
    }
  }
  return false;
}
```

**逐行注解（关键在那两段注释）：**

- `LookupKey` 内部把 user key 和**当前 snapshot 的 sequence** 编码成一个 lookup key，且故意用一个比所有真实 type 都大的 type 占位（`kValueTypeForSeek`）。
- `iter.Seek(memkey.data())`：在 skiplist 里找 `>=` lookup key 的第一个节点。因为同一 user key 下 sequence 降序排，**Seek 落点天然是"不超过我快照、且最新"的那个版本**。这就是注释说的 "we do not check the sequence number since the Seek() call above should have skipped all entries with overly large sequence numbers"——**版本可见性靠排序+一次 Seek 解决，零额外比较**。
- `tag & 0xff` 取出 type：
  - `kTypeValue` → 返回值。
  - `kTypeDeletion` → 返回 `NotFound`。**注意：墓碑被命中也算"查到了"（return true），只是结果是 NotFound**。这正确地遮蔽了更老 level 里同 key 的旧值——这是 LSM 删除语义的精髓。

> **生产坑预告**：墓碑必须保留到比它老的所有数据都被 compaction 清理掉之后才能丢，否则旧值会"复活"。删除密集的工作负载会堆积大量墓碑，拖慢范围扫描（每次 scan 都要跳过墓碑）。Cassandra 的 `tombstone_failure_threshold` 就是为此而设。详见 §6.3。

### 3.3 C₀ 的实现：SkipList::Insert —— 为什么不用平衡树

论文里 C₀ 是"平衡树"，但 LevelDB 用 **skiplist**。原因：skiplist 实现简单、**插入无需旋转**、且能做**无锁并发读**（单写多读）。

【真实源码 google/leveldb@db/skiplist.h】

```cpp
template <typename Key, class Comparator>
int SkipList<Key, Comparator>::RandomHeight() {
  // Increase height with probability 1 in kBranching
  static const unsigned int kBranching = 4;
  int height = 1;
  while (height < kMaxHeight && rnd_.OneIn(kBranching)) {
    height++;
  }
  assert(height > 0);
  assert(height <= kMaxHeight);
  return height;
}
```

```cpp
template <typename Key, class Comparator>
void SkipList<Key, Comparator>::Insert(const Key& key) {
  // TODO(opt): We can use a barrier-free variant of FindGreaterOrEqual()
  // here since Insert() is externally synchronized.
  Node* prev[kMaxHeight];
  Node* x = FindGreaterOrEqual(key, prev);

  // Our data structure does not allow duplicate insertion
  assert(x == nullptr || !Equal(key, x->key));

  int height = RandomHeight();
  if (height > GetMaxHeight()) {
    for (int i = GetMaxHeight(); i < height; i++) {
      prev[i] = head_;
    }
    // It is ok to mutate max_height_ without any synchronization
    // with concurrent readers. ...
    max_height_.store(height, std::memory_order_relaxed);
  }

  x = NewNode(key, height);
  for (int i = 0; i < height; i++) {
    // NoBarrier_SetNext() suffices since we will add a barrier when
    // we publish a pointer to "x" in prev[i].
    x->NoBarrier_SetNext(i, prev[i]->NoBarrier_Next(i));
    prev[i]->SetNext(i, x);
  }
}
```

**逐行注解：**

- `kBranching = 4`：每升一层概率 1/4，期望每 4 个节点提升一层。这决定了 skiplist 的"fanout"，对应期望查找 `O(log_4 N)`。`kMaxHeight = 12`（`db/skiplist.h` 里 `enum { kMaxHeight = 12 };`，已核实），所以这棵 skiplist 设计上撑约 `4^12 ≈ 1600 万` 个节点而层数不爆。
- `FindGreaterOrEqual(key, prev)`：找插入位置，同时把每一层的前驱记到 `prev[]`。
- `assert(... !Equal(key, x->key))`：**skiplist 本身不允许重复 key**。但 LSM 明明要存同一 user key 的多个版本？——因为这里的 Key 是 **internal key（含 sequence）**，不同版本 sequence 不同，internal key 就不同，不冲突。设计闭环。
- 插入循环里的 `NoBarrier_SetNext` + 注释：先把新节点的 next 指针设好（无内存屏障，因为还没人能看到它），最后 `prev[i]->SetNext(i, x)`（带屏障）才"发布"这个节点。**这是单写多读无锁的经典手法**：读者要么看不到新节点（看到旧链），要么看到完整的新节点，绝不会看到半个。

> **为什么 memtable 用 skiplist 而不是红黑树**：(1) 插入只改指针不旋转，写路径短；(2) 上面这套无屏障发布让**读不用加锁**，而 LSM 写是串行化的（外部已同步），完美契合"单写多读"；(3) 实现量小、缓存友好性可接受。RocksDB 也提供 hash-skiplist、vector 等可插拔 memtable，但默认仍是 skiplist。

### 3.4 落盘形态：SSTable 的物理布局

memtable 满了（默认 `write_buffer_size = 4 * 1024 * 1024` = 4MB，见 `include/leveldb/options.h`，已核实）就冻结成 immutable memtable，后台线程把它写成一个 **SSTable 文件**。

SSTable 的 on-disk 布局【真实源码 google/leveldb@doc/table_format.md，已核实】：

```
[data block 1]      <- 一批有序 KV，默认 block_size = 4KB
[data block 2]
...
[data block N]
[meta block 1]      <- 例如 filter block（bloom filter）
...
[meta block K]
[metaindex block]   <- 指向各 meta block
[index block]       <- 每个 data block 一条：(>=该block最后key 的分隔key, BlockHandle)
[Footer]            <- 固定 48 字节，含 metaindex/index 的 handle + magic
```

Footer 结构【真实源码 google/leveldb@doc/table_format.md】：

```
metaindex_handle: char[p];     // Block handle for metaindex
index_handle:     char[q];     // Block handle for index
padding:          char[40-p-q];// zeroed bytes to make fixed length
magic:            fixed64;     // == 0xdb4775248b80fb57 (little-endian)
```

**为什么这么设计：**
- **block 化 + 压缩**：每个 data block 独立 Snappy 压缩（默认 `compression = kSnappyCompression`，已核实），读时只解压需要的 block。
- **index block 常驻/可缓存**：点查一个 key，先在 index block 二分定位到唯一可能的 data block，再读那一个 block。**一个 SSTable 内点查 ≈ 1 次 I/O**（index 缓存命中时）。
- **bloom filter（meta block）**：`include/leveldb/filter_policy.h` 提供 `NewBloomFilterPolicy(bits_per_key)`。点查时先问 bloom：**"这个 key 肯定不在这个 SSTable"** 能直接跳过，省掉一次 data block I/O。这是 LSM 控制读放大的最关键武器（见 §6.1）。

SSTable **写一次、永不修改**（immutable）。这个不变性是 LSM 一切并发简单性的根基：读者读旧文件，compaction 写新文件，互不阻塞，旧文件靠引用计数 + version 机制安全回收。

### 3.5 归并：Compaction —— LSM 的心脏与代价

写一直 append，文件会越堆越多：读要查的文件越来越多（读放大↑）、空间冗余越来越大（同一 key 多版本散在多文件，空间放大↑）。**compaction** 就是后台把多个 SSTable 归并去重、重新分层的过程。它是 LSM 把"写得爽"的债**还回去**的地方。

LevelDB 用 **leveled compaction**：

- 共 7 层（`kNumLevels = 7`，已核实），L0 特殊，L1..L6 每层是上层的 ~10 倍。
- 每层（L1+）内的 SSTable **key 范围互不重叠、全局有序**。
- **L0 例外：文件之间 key 范围可以重叠**（因为是 memtable 直接 flush 下来的，谁也没归并谁）。

**层大小规则**【真实源码 google/leveldb@db/version_set.cc】：

```cpp
static double MaxBytesForLevel(const Options* options, int level) {
  // Note: the result for level zero is not really used since we set
  // the level-0 compaction threshold based on number of files.

  // Result for both level-0 and level-1
  double result = 10. * 1048576.0;
  while (level > 1) {
    result *= 10;
    level--;
  }
  return result;
}
```

逐行：L1 上限 10MB，之后每层 ×10——L2=100MB，L3=1GB，L4=10GB……**这就是论文 rolling merge 的"逐级 10 倍放大"在工程上的落点**。

**何时触发 compaction —— compaction score**【真实源码 google/leveldb@db/version_set.cc】：

```cpp
void VersionSet::Finalize(Version* v) {
  // Precomputed best level for next compaction
  int best_level = -1;
  double best_score = -1;

  for (int level = 0; level < config::kNumLevels - 1; level++) {
    double score;
    if (level == 0) {
      // We treat level-0 specially by bounding the number of files
      // instead of number of bytes for two reasons:
      //
      // (1) With larger write-buffer sizes, it is nice not to do too
      // many level-0 compactions.
      //
      // (2) The files in level-0 are merged on every read and
      // therefore we wish to avoid too many files when the individual
      // file size is small ...
      score = v->files_[level].size() /
              static_cast<double>(config::kL0_CompactionTrigger);
    } else {
      // Compute the ratio of current size to size limit.
      const uint64_t level_bytes = TotalFileSize(v->files_[level]);
      score =
          static_cast<double>(level_bytes) / MaxBytesForLevel(options_, level);
    }

    if (score > best_score) {
      best_level = level;
      best_score = score;
    }
  }

  v->compaction_level_ = best_level;
  v->compaction_score_ = best_score;
}
```

逐行注解：
- **L0 按文件数算 score**：`文件数 / kL0_CompactionTrigger`（`kL0_CompactionTrigger = 4`，已核实）。注释 (2) 说出了关键——**L0 文件每次读都要全部检查**（因为它们 key 范围重叠，不能靠二分排除），所以 L0 文件数必须卡死，否则读放大爆炸。
- **L1+ 按字节数算 score**：`当前层总字节 / 该层上限`。超过 1 就该 compact。
- 取 score 最大的层作为下次 compaction 目标。**score ≥ 1 才真正触发**（见下）。

**优先级：size 触发 > seek 触发**【真实源码 google/leveldb@db/version_set.cc，`PickCompaction` 片段】：

```cpp
  // We prefer compactions triggered by too much data in a level over
  // the compactions triggered by seeks.
  const bool size_compaction = (current_->compaction_score_ >= 1);
  const bool seek_compaction = (current_->file_to_compact_ != nullptr);
```

这里出现了 LevelDB 一个精巧设计——**seek-triggered compaction**：如果某个文件被反复 seek 却总是 miss（说明它和上层有大量 key 重叠、白白增加读放大），就主动把它 compact 下去。计数逻辑【真实源码 google/leveldb@db/version_set.cc】：

```cpp
f->allowed_seeks = static_cast<int>((f->file_size / 16384U));
if (f->allowed_seeks < 100) f->allowed_seeks = 100;
```

```cpp
bool Version::UpdateStats(const GetStats& stats) {
  FileMetaData* f = stats.seek_file;
  if (f != nullptr) {
    f->allowed_seeks--;
    if (f->allowed_seeks <= 0 && file_to_compact_ == nullptr) {
      file_to_compact_ = f;
      file_to_compact_level_ = stats.seek_file_level;
      return true;
    }
  }
  return false;
}
```

逐行：每 16KB 文件给 1 次 allowed_seeks（最少 100 次）。每次一个 Get 穿过该文件却没命中、最终在更低层命中，就给它 `allowed_seeks--`；扣到 0 就标记 `file_to_compact_`。**直觉**：一次 seek（随机 I/O ~10ms 量级）和归并 16KB 数据（~40MB/s 顺序）的成本量级相当，所以"被无效 seek 这么多次，不如直接 compact 掉"。这是把"读放大代价"反馈到"写/compaction 决策"的经典 self-tuning。

**L0→L1 的特殊性**【真实源码 google/leveldb@db/version_set.cc，`PickCompaction` 片段】：

```cpp
  // Level-0 requires merging overlapping files
  if (level == 0) {
    InternalKey smallest, largest;
    GetRange(c->inputs_[0], &smallest, &largest);
    current_->GetOverlappingInputs(0, &smallest, &largest, &c->inputs_[0]);
  }
```

因为 L0 文件互相重叠，挑一个文件做 compaction 必须把所有和它 key 范围重叠的 L0 文件**一起拉进来**，否则归并结果不可能保证 L1 的"不重叠"不变性。

**新 memtable 的快捷下推**【真实源码 google/leveldb@db/dbformat.h】：

```cpp
static const int kMaxMemCompactLevel = 2;
```

注释说：一个新 compact 出来的 memtable，如果和下层没有 key 重叠，**最多可以直接推到 L2**（跳过 L0/L1），以避免昂贵的 L0⇒L1 compaction 和 manifest 操作；但不会一路推到最底层，否则同一 key 空间被反复覆写时会浪费大量磁盘。这是个很实战的小优化。

> **写放大的来源现在很清楚了**：一条数据从 L0 活到 L6，最坏要被 compaction 重写 6 次，每次和下层 ~10 倍体量的数据归并。RocksDB 文档直言 leveled compaction 的写放大"often larger than 10"（已核实）。这就是 LSM 用读/写/空间放大换写吞吐的账本。下一节用真实数字把它跑出来。

---

## 4. ⭐可运行 Demo（重中之重）

> **声明：以下两个 demo 设计为可运行，请在你的环境验证。** 依赖：**Python 3.8+，纯标准库（json / os / bisect / glob / random / shutil），无第三方包**。我已在 Python 3 实测通过，下方"预期输出"是真实运行结果（随机种子固定，可复现）。

这两个 demo 的目标，是把 §3 的 LevelDB 机制用最小代码**亲手实现并印证**：
- Demo A：手写 toy LSM（memtable + SSTable + compaction + 墓碑删除 + 版本覆盖），印证 §3.1/3.2/3.5。
- Demo B：toy B-Tree vs toy LSM 的**写放大对照**，把 §1.2、§3.5 那句"写放大 often > 10"变成你机器上的数字。

### 4.1 Demo A：手写 toy LSM —— memtable / SSTable / compaction

**机制对应关系：**
- `MemTable` ↔ LevelDB MemTable（这里用 dict + 排序代替 skiplist，语义等价：内存有序缓冲）。
- `delete()` 写 `None` ↔ LevelDB 的 `kTypeDeletion` 墓碑。
- `SSTable` 一行一条 JSON、写一次不可变 ↔ LevelDB `.ldb` 文件。
- `get()` 先查 memtable 再**从新到旧**查 SSTable ↔ LevelDB 的版本可见性（新值/墓碑遮蔽旧值）。
- `compact()` 多文件归并去重、丢墓碑 ↔ LevelDB compaction。

把下面存成 `toy_lsm.py`：

```python
"""toy_lsm.py —— 最小可运行 LSM-tree，印证 LevelDB 的 memtable/SSTable/compaction。
设计为可运行，请在你的环境验证。依赖：Python 3.8+ 标准库。"""
import os, json, bisect, glob, shutil

MISS = "__MISS__"  # 区分"没查到"和"查到的值恰好是 None"

class MemTable:
    """内存有序缓冲。对应 LevelDB MemTable（这里用 dict，flush 时排序）。"""
    def __init__(self):
        self.d = {}
    def put(self, k, v):
        self.d[k] = v
    def delete(self, k):
        self.d[k] = None          # None = 墓碑(tombstone)，对应 kTypeDeletion
    def get(self, k):
        return self.d.get(k, MISS)
    def __len__(self):
        return len(self.d)
    def items_sorted(self):
        return sorted(self.d.items())

class SSTable:
    """不可变有序文件。一行一条 JSON: [key, value]。对应 LevelDB 的 .ldb。
    内存里只保留一份 key 索引(对应 SSTable 的 index block)，点查靠二分。"""
    def __init__(self, path):
        self.path = path
        self.index = []           # 有序 key 列表，模拟 index block
        with open(self.path) as f:
            for line in f:
                self.index.append(json.loads(line)[0])

    @staticmethod
    def write(path, items):       # items: 已排序的 (k, v) 列表
        with open(path, "w") as f:
            for k, v in items:
                f.write(json.dumps([k, v]) + "\n")
        return SSTable(path)

    def get(self, k):
        i = bisect.bisect_left(self.index, k)   # index block 二分定位
        if i < len(self.index) and self.index[i] == k:
            with open(self.path) as f:          # 命中才读文件这一行
                for j, line in enumerate(f):
                    if j == i:
                        return json.loads(line)[1]
        return MISS

class ToyLSM:
    def __init__(self, dirpath, flush_threshold=4):
        self.dir = dirpath
        self.flush = flush_threshold
        os.makedirs(self.dir, exist_ok=True)
        self.mem = MemTable()
        self.seq = 0
        self.write_bytes = 0      # 累计物理落盘字节，用于观察写放大
        self.ssts = [SSTable(p) for p in sorted(glob.glob(self.dir + "/sst-*.json"))]

    def put(self, k, v):
        self.mem.put(k, v)
        if len(self.mem) >= self.flush:
            self._flush()

    def delete(self, k):
        self.mem.delete(k)
        if len(self.mem) >= self.flush:
            self._flush()

    def _flush(self):
        """memtable 满 -> 冻结 -> 写成一个新 SSTable。对应 LevelDB 的 minor compaction。"""
        if len(self.mem) == 0:
            return
        path = f"{self.dir}/sst-{self.seq:04d}.json"
        self.seq += 1
        sst = SSTable.write(path, self.mem.items_sorted())
        self.write_bytes += os.path.getsize(path)
        self.ssts.append(sst)
        self.mem = MemTable()

    def get(self, k):
        """读路径：先查 memtable，再从最新到最旧查 SSTable。
        命中墓碑(None) => 返回 None(已删除)，并遮蔽更旧的值。"""
        v = self.mem.get(k)
        if v != MISS:
            return None if v is None else v
        for sst in reversed(self.ssts):     # reversed = 从新到旧
            v = sst.get(k)
            if v != MISS:
                return None if v is None else v
        return None

    def compact(self):
        """把所有 SSTable 归并成一个：同 key 取最新、丢弃墓碑。
        对应 LevelDB 的 (full) compaction。"""
        self._flush()
        merged = {}
        for sst in self.ssts:               # 旧->新顺序覆盖，后写的赢
            with open(sst.path) as f:
                for line in f:
                    k, v = json.loads(line)
                    merged[k] = v
        live = {k: v for k, v in merged.items() if v is not None}  # 丢墓碑
        for sst in self.ssts:
            os.remove(sst.path)
        path = f"{self.dir}/sst-{self.seq:04d}.json"
        self.seq += 1
        sst = SSTable.write(path, sorted(live.items()))
        self.write_bytes += os.path.getsize(path)
        self.ssts = [sst]


if __name__ == "__main__":
    shutil.rmtree("/tmp/toy_lsm_db", ignore_errors=True)
    lsm = ToyLSM("/tmp/toy_lsm_db", flush_threshold=4)

    for i in range(20):                 # 写 20 条，flush_threshold=4 => 触发 5 次 flush
        lsm.put(f"k{i:03d}", f"v{i}")
    lsm.put("k005", "UPDATED")           # 更新：写新版本(append)，不原地改
    lsm.delete("k010")                   # 删除：写墓碑

    print("get k005          :", lsm.get("k005"))   # 期望 UPDATED(新版本遮蔽旧值)
    print("get k010          :", lsm.get("k010"))   # 期望 None(墓碑遮蔽旧值)
    print("get k019          :", lsm.get("k019"))   # 期望 v19
    print("ssts  before cmpct:", len(lsm.ssts))     # 期望 5
    print("bytes before cmpct:", lsm.write_bytes)

    lsm.compact()
    print("ssts  after  cmpct:", len(lsm.ssts))     # 期望 1(归并成一个)
    print("get k005 after    :", lsm.get("k005"))   # 期望 UPDATED(归并后仍正确)
    print("get k010 after    :", lsm.get("k010"))   # 期望 None(墓碑被清理，但 key 不在 = None)
```

**运行步骤：**
```bash
python3 toy_lsm.py
```

**预期输出（实测，固定逻辑可复现）：**
```
get k005          : UPDATED
get k010          : None
get k019          : v19
ssts  before cmpct: 5
bytes before cmpct: 310
ssts  after  cmpct: 1
get k005 after    : UPDATED
get k010 after    : None
```

**这个 demo 印证了什么：**
1. **更新不原地改**：`k005` 先写 `v5`、后写 `UPDATED`，两条都落了盘，读时 `reversed(ssts)` 从新到旧取到 `UPDATED`——精确对应 §3.2 的 sequence 排序 + "取最新"。
2. **删除是墓碑**：`k010` 的墓碑遮蔽了它的旧值 `v10`，对应 §3.2 的 `kTypeDeletion` 命中也 return true（结果是"已删除"）。
3. **compaction 去重+清墓碑**：5 个文件归并成 1 个，读语义不变——对应 §3.5。
4. **读放大可观察**：compact 前查一个老 key 最坏要 reverse 扫多个 SSTable（这里用 index 二分缩小，但文件数本身就是读放大来源），compact 后只剩 1 个文件。把 `flush_threshold` 调大或写更多数据，能直观看到文件数=读放大的关系。

> **进阶玩法**：把 `compact()` 改成"只归并相邻两个最老的 SSTable"，就从 full compaction 退化成 tiered；再给每个 SSTable 标 level、按 §3.5 的 10 倍规则触发，就接近 leveled。这正是 §8 代码题的方向。

### 4.2 Demo B：B-Tree vs LSM 写放大对照 —— 把"often > 10"跑出来

**思路**：用两个最小模型，对**同一批随机插入**统计**物理落盘字节**，再除以逻辑用户字节，得到写放大比。
- toy B-Tree：固定容量的"页"数组，每次插入命中哪个页就**把整页改写回盘**（模拟 in-place 的 page-granular 写）；页满分裂时改写两个页。
- toy LSM：内存缓冲攒够 batch 就**顺序写一个 run**，每条只写一次（模拟 L0 落盘，不含后续 compaction 重写）。

把下面存成 `wa_compare.py`：

```python
"""wa_compare.py —— B-Tree(原地改页) vs LSM(顺序批写) 写放大对照。
设计为可运行，请在你的环境验证。依赖：Python 3.8+ 标准库。"""
import bisect, random

PAGE_BYTES = 256   # 每页字节(故意调小以放大分裂效应)
REC_BYTES  = 32    # 每条记录逻辑大小

class CountingBTree:
    """页式 B-Tree 近似：每次插入改写命中的整页，页满则分裂改写两页。
    统计 bytes_written = 物理落盘量。"""
    def __init__(self, page_capacity=8):
        self.cap = page_capacity
        self.pages = [[]]            # 每页是一个有序 (k,v) 列表
        self.page_lo = [""]          # 每页最小 key，用于路由
        self.bytes_written = 0

    def _find_page(self, k):
        i = bisect.bisect_right(self.page_lo, k) - 1
        return max(0, i)

    def put(self, k, v):
        pi = self._find_page(k)
        page = self.pages[pi]
        ks = [x[0] for x in page]
        idx = bisect.bisect_left(ks, k)
        if idx < len(page) and page[idx][0] == k:
            page[idx] = (k, v)        # 更新：仍要改写整页
        else:
            page.insert(idx, (k, v))  # 插入：仍要改写整页(且先得随机读它)
        if len(page) > self.cap:      # 页满 -> 分裂 -> 改写两个页
            mid = len(page) // 2
            left, right = page[:mid], page[mid:]
            self.pages[pi] = left
            self.pages.insert(pi + 1, right)
            self.page_lo[pi] = left[0][0]
            self.page_lo.insert(pi + 1, right[0][0])
            self.bytes_written += 2 * PAGE_BYTES
        else:
            self.bytes_written += PAGE_BYTES   # 普通插入改写一个页

class CountingLSM:
    """LSM 的 L0 落盘近似：内存攒够 flush 条就顺序写一个 run，每条写一次。"""
    def __init__(self, flush=64):
        self.flush = flush
        self.mem = {}
        self.bytes_written = 0
        self.runs = 0
    def put(self, k, v):
        self.mem[k] = v
        if len(self.mem) >= self.flush:
            self._flush()
    def _flush(self):
        self.bytes_written += len(self.mem) * REC_BYTES
        self.runs += 1
        self.mem = {}
    def close(self):
        if self.mem:
            self._flush()


if __name__ == "__main__":
    random.seed(42)
    N = 50000
    keys = [f"k{random.randint(0, 10**9):010d}" for _ in range(N)]  # 随机 key

    bt = CountingBTree(page_capacity=8)
    for k in keys:
        bt.put(k, "v")

    lsm = CountingLSM(flush=64)
    for k in keys:
        lsm.put(k, "v")
    lsm.close()

    user_bytes = N * REC_BYTES
    print(f"records inserted     : {N}")
    print(f"logical user bytes   : {user_bytes}")
    print(f"B-tree bytes written : {bt.bytes_written:>12}  (pages={len(bt.pages)})")
    print(f"LSM    bytes written : {lsm.bytes_written:>12}  (runs={lsm.runs})")
    print(f"B-tree write amp     : {bt.bytes_written / user_bytes:6.2f}x")
    print(f"LSM    write amp (L0): {lsm.bytes_written / user_bytes:6.2f}x")
```

**运行步骤：**
```bash
python3 wa_compare.py
```

**预期输出（实测，seed=42 可复现）：**
```
records inserted     : 50000
logical user bytes   : 1600000
B-tree bytes written :     15047680  (pages=8781)
LSM    bytes written :      1600000  (runs=782)
B-tree write amp     :   9.40x
LSM    write amp (L0):   1.00x
```

**这个 demo 印证了什么：**
- **B-Tree 随机插入写放大 ~9.4x**：每条 32 字节的逻辑写，因为要"读出整页 256 字节、改完写回"，物理上写了 ~300 字节，随机插入还频繁触发分裂（最终 8781 个页）。这就是 §1.2 论文那句"B-tree double the I/O cost"在小规模、page-granular 下的放大版——**真实数据库里 page=4KB/16KB，单条小记录的写放大只会更夸张**。
- **LSM L0 落盘写放大 ~1.0x**：顺序批写，每条只写一次。这是 LSM 写吞吐高的根因。
- **但这只是 L0**：真实 LSM 还有 compaction 把数据从 L0 重写到 L6（§3.5），加上 compaction 的总写放大会爬到 10+（RocksDB 文档原话，已核实 §5）。**所以"LSM 写放大低"的准确说法是：前台写放大极低，总写放大被推迟到后台 compaction、且可通过 compaction 策略调节**。

> **把 demo 推进一步**：给 `CountingLSM` 加一个 `compact()` 模拟 leveled——每攒 10 个 run 归并成 1 个更大的 run 推到下一层，统计含 compaction 的总 bytes_written，你就能亲手把 LSM 写放大从 1.0x 推到 ~10x，完整复现 §3.5 的账本。这是 §8 代码题之一。

---

## 5. 方案对比：B-Tree vs LSM，跑一遍具体场景

### 5.1 RUM 三角对照表

| 维度 | B-Tree (B+Tree) | LSM-Tree (leveled) | 出处/依据 |
|---|---|---|---|
| **写放大** | 高（每条小写改写整页 + 分裂；本章 demo ~9.4x） | 前台极低(~1)，含 compaction 总写放大常 >10 | RocksDB wiki："often larger than 10"（已核实） |
| **写吞吐** | 受随机 I/O 限制 | 高（顺序批写） | LSM 论文核心论点（已核实） |
| **点查读放大** | 低（树高次 I/O，~3-4 次） | 中（查 memtable + 多层 SSTable，靠 bloom filter 压低） | §3.4 |
| **范围扫描** | **优**（B+Tree 叶子链表天然顺序） | 中（要多路归并多层 + 跳墓碑） | §6.3 |
| **空间放大** | 低-中（页有 ~1/3 碎片，fill factor 问题） | 中-高（多版本冗余；leveled ~1.1x，tiered 可达 2x） | RocksDB universal："200 means triple"（已核实） |
| **删除成本** | 即时（原地标删/合并） | 延迟（墓碑要等 compaction 才真清） | §3.2, §6.3 |
| **SSD 友好度** | 一般（随机写、写放大伤寿命） | 好（顺序写、少擦除） | WiscKey 前提（已核实 §7） |
| **崩溃恢复** | WAL/redo | WAL replay memtable | 通用 |
| **典型代表** | PostgreSQL, MySQL InnoDB, SQLite | LevelDB, RocksDB, Cassandra, HBase, ScyllaDB | — |

### 5.2 具体场景跑一遍

**场景 A：OLTP 点查为主（用户表按主键查，读多写少，记录小）**
- 选 **B-Tree**。点查 3-4 次 I/O，范围查（`WHERE id BETWEEN`）走叶子链表极快，没有 compaction 抢 I/O、没有读多层的 tail latency。LSM 在这里的多层查询 + compaction 抖动是净负担。
- **不适用边界**：如果点查为主但写也非常高（如计数器、状态频繁更新），InnoDB 的 change buffer 能缓解随机写，但极端高写仍会被 B-Tree 写放大拖垮——这时该考虑 LSM。

**场景 B：时序/日志/监控写入（每秒百万条 append，少更新，范围按时间扫）**
- 选 **LSM**。顺序批写吃满磁盘带宽，写放大前台 ~1，bloom filter 让点查可接受。Cassandra/ScyllaDB/InfluxDB(部分) 走这条。
- **不适用边界**：如果是**删除/TTL 密集**的时序（大量过期数据删除），墓碑会堆积、范围扫描被拖慢（§6.3），需要专门的 tombstone GC / TTL compaction 策略。

**场景 C：KV 缓存持久化 / 大 value（如对象元数据 + 大 blob）**
- LSM 但**警惕大 value 的写放大**：value 大时 compaction 反复重写 value 极其浪费——这正是 WiscKey 要 key-value 分离的场景（§7）。

**场景 D：内存型（数据集装得下内存）**
- 物理层选择变弱（不怎么碰盘）。Redis 用纯内存结构（hash/skiplist/ziplist），它的"持久化"才碰到 LSM 类思想（AOF append-only ≈ WAL，RDB ≈ snapshot）。**注意 Redis 本身不是 LSM 也不是 B-Tree，它是内存数据结构服务器**——别把它归错类，这是面试常见混淆点。

---

## 6. 失败模式、生产真坑与底层根因

### 6.1 读放大失控：bloom filter 没配好 / L0 文件堆积

- **现象**：LSM 点查 p99 突然飙高。
- **根因 1**：bloom filter 的 `bits_per_key` 太小（假阳性率高），导致大量"bloom 说可能在、实际不在"的无效 data block I/O。`NewBloomFilterPolicy(10)`（10 bits/key，假阳性 ~1%）是常见起点；调到 10-16。
- **根因 2**：写太猛，L0 文件堆到 `kL0_SlowdownWritesTrigger = 8`（开始限速）、`kL0_StopWritesTrigger = 12`（停写）（均已核实 `db/dbformat.h`）。L0 文件 key 重叠，每次点查都要全查（§3.5 Finalize 注释），文件一多读放大爆炸。
- **底层根因**：写入速率 > compaction 消化速率，债还不过来。解法：增大 `write_buffer_size`、提升 compaction 并发、或换 universal/tiered 减少 compaction 量（但牺牲空间，见 §6.4）。

### 6.2 写停顿（write stall）：compaction 抢 I/O 抖动

- **现象**：写吞吐周期性掉零，p99/p999 长尾。
- **根因**：后台 compaction 是重 I/O 操作，和前台写抢磁盘带宽；当 L0 触发 stop trigger，前台写被**强制阻塞**等 compaction 追上。这是 LSM 把"前台写得爽"的债集中偿还时的副作用。
- **解法**：rate limiter（RocksDB `RateLimiter` 限 compaction I/O）、subcompaction 并行（RocksDB `max_subcompactions`，已核实 L0→L1 默认不并行是个常见瓶颈）、调大 L0 触发阈值平滑。
- **底层根因**：LSM 的成本是**延迟且突发**的（compaction 批量发生），不像 B-Tree 那样均摊到每次写。这是用"平均写吞吐高"换"写延迟方差大"。

### 6.3 墓碑堆积：删除密集工作负载的隐形杀手

- **现象**：明明 `DELETE` 了大量数据，磁盘没降、范围扫描越来越慢，甚至被删的数据"复活"。
- **根因**：墓碑（§3.2）必须保留到比它**老**的所有同 key 数据都被 compact 掉才能丢。如果这些旧数据在很深的 level、迟迟不参与 compaction，墓碑就一直堆着。范围扫描每次都要读过并跳过这些墓碑（read 端必须看到墓碑才能正确遮蔽旧值）。
- **真实案例**：Cassandra 经典坑——对一个分区反复"插入再删除"，`SELECT` 时扫过成千上万墓碑触发 `tombstone_warn_threshold`(1000) / `tombstone_failure_threshold`(100000) 直接查询失败。
- **底层根因**：LSM 的删除是"逻辑标记 + 延迟物理清理"，删除的真实成本被推迟到 compaction，工作负载若让旧数据躲在深层不参与 compaction，墓碑就 GC 不掉。
- **解法**：调 compaction 策略让旧数据参与归并（如 TTL/time-window compaction）、避免"insert-then-delete"反模式、监控墓碑指标。

### 6.4 空间放大：tiered/universal 的"双倍磁盘"陷阱

- **现象**：换 universal compaction 后写吞吐上去了，但磁盘用量接近**数据真实量的 2 倍**。
- **根因**：universal/tiered compaction 用"少归并"换"低写放大"，代价是同一 key 的旧版本在多个 sorted run 里冗余更久。RocksDB 文档（已核实）：space amplification "200 means triple"，默认 `max_size_amplification_percent` 控制上限。
- **底层根因**：RUM 三角——universal 把指针从 Write 放大移向 Space 放大。
- **解法**：用 leveled（空间放大低 ~1.1x，`level_compaction_dynamic_level_bytes=true` 保证 90% 数据在最底层，已核实）；磁盘紧张选 leveled，写吞吐紧张选 universal。

### 6.5 B-Tree 侧的坑：页分裂风暴与索引膨胀

- **现象**：PostgreSQL 随机 UUID 主键插入慢、索引体积虚高。
- **根因**：随机插入让 B+Tree 频繁页分裂（本章 demo B 已复现：8781 个页、9.4x 写放大），且分裂后页只半满（碎片）。顺序自增主键则几乎不分裂（总在最右叶子追加）——**这就是"用自增 ID 不用随机 UUID 做主键"的物理层原因**。
- **底层根因**：B-Tree 要维持页内有序 + 平衡，随机 key 打散到全树各页，每页都要读-改-写。
- **解法**：顺序主键、或用 UUIDv7（时间有序）、或 `FILLFACTOR` 预留空间减少分裂、定期 REINDEX。

---

## 7. 未来与前沿：key-value 分离与硬件协同

### 7.1 WiscKey：把 value 从 LSM 里拎出来（FAST 2016）

LSM 的 compaction 把 key 和 value **一起**反复重写。当 value 远大于 key 时，绝大部分 compaction I/O 浪费在搬运 value 上。**WiscKey**（Lu et al., FAST '16，UW-Madison，已 WebFetch/搜索核实）提出 **key-value separation**：

- value 顺序追加到独立的 **Value Log (vLog)**，LSM 里只存 `(key, value的指针)`。
- compaction 只排序/重写**很小的 key+指针**，value 原地不动 → **写放大大幅下降**。
- 论文实测数字（已核实）：加载数据库比 LevelDB 快 **2.5×～111×**，随机查快 **1.6×～14×**；1KB/4KB value 下吞吐是 LevelDB 的 **46×/111×**。

**代价（trade-off）**：
- **范围扫描变差**：key 有序但 value 散在 vLog 各处，scan 要随机读 vLog（SSD 上可用并行预取缓解，这也是它叫 "SSD-conscious" 的原因）。
- **vLog 的垃圾回收**：被覆盖/删除的旧 value 在 vLog 里成垃圾，需要独立 GC（扫 vLog、检查 key 是否还指向它、有效的搬到 vLog 头部）。GC 本身又是一笔 I/O 成本。

工业落地：**Badger**（Dgraph，Go）、TiKV 的 **Titan** 插件都是 key-value 分离的产品化。这是 LSM 在大 value 场景的主流演进方向。

### 7.2 其他前沿方向（点到为止）

- **Learned Index / Bourbon**（arXiv 2005.14213）：用学习模型替代/辅助 SSTable 的 index block 做定位，理论上读更快。仍在研究-早期产品化阶段。
- **B-ε tree / Bw-tree**：B-Tree 家族吸收 LSM 的"缓冲写"思想（节点里缓冲更新批量下推），试图在写放大上逼近 LSM 而保留 B-Tree 的范围扫描优势。Microsoft 的 Bw-tree（无锁、append-delta）是代表。
- **硬件协同**：ZNS SSD（分区命名空间）让 LSM 的 SSTable 直接对齐 SSD 的擦除单元，进一步降写放大、延寿命；持久内存（PMEM）改写 memtable/WAL 设计。

> **趋势收口**：物理层的演进主线是**模糊 B-Tree 和 LSM 的边界**——B-Tree 学缓冲写降写放大，LSM 学 key-value 分离/learned index 降读放大与空间放大。RUM 三角永远成立，工程上是不断在三角内找更优的 Pareto 点。

---

## 8. 面试 / 实战五件套

### 8.1 概念题
**Q：解释写放大、读放大、空间放大，并说明 B-Tree 和 LSM 各自在哪个维度吃亏、为什么。**
A：写放大 = 物理落盘字节 / 逻辑写字节；读放大 = 一次逻辑读触发的物理读次数/字节；空间放大 = 物理占用 / 逻辑数据量。B-Tree 吃亏在**写放大**（每条小写要读-改-写整页 + 分裂，本章 demo 实测 ~9.4x）和随机写吞吐；LSM 吃亏在**读放大**（查 memtable + 多层 SSTable，靠 bloom filter 压）和**空间放大**（多版本冗余）+ compaction 抖动。根因是 RUM 三角：B-Tree 原地更新偏读优、LSM 追加写偏写优，二者不可兼得。

### 8.2 排查题
**Q：线上一个 RocksDB 服务写吞吐周期性掉零、p999 飙到秒级，怎么排查？**
A：(1) 看 L0 文件数是否撞 `slowdown/stop trigger`（8/12）→ 写速 > compaction 速。(2) 看 compaction 是否抢满磁盘 I/O（write stall 统计、`rocksdb.stall.micros`）。(3) 检查 L0→L1 是否单线程瓶颈（默认不并行，已核实），开 `max_subcompactions`。(4) 加 `RateLimiter` 限 compaction、增大 `write_buffer_size` 减少 flush 频率。底层根因是 LSM 成本延迟突发，需让 compaction 消化速率匹配写入速率（§6.2）。

### 8.3 设计题
**Q：设计一个 KV 引擎，value 平均 64KB（大 value），写多读少偶尔范围扫，怎么选物理结构？**
A：选 LSM 但做 **key-value 分离（WiscKey 式）**：value 进 vLog 顺序追加，LSM 存 key+指针，让 compaction 只搬小 key、避免反复重写 64KB value（否则写放大爆炸，§7.1）。范围扫描少 → 能接受 vLog 随机读的代价，SSD 上加并行预取。需配套 vLog GC。若范围扫描其实很重，则重新评估——key-value 分离的 scan 是短板。

### 8.4 权衡题
**Q：什么时候宁可用 B-Tree 不用 LSM？给两个理由 + 一个反例边界。**
A：理由 1——**范围扫描 + 顺序读为主**（OLTP report、`ORDER BY` 分页），B+Tree 叶子链表天然顺序，LSM 要多路归并跳墓碑。理由 2——**删除密集**，B-Tree 即时回收，LSM 墓碑堆积拖慢 scan（§6.3）。反例边界：如果同时写入吞吐极高（>单盘随机 IOPS 能撑），B-Tree 写放大会卡死，这时即便有上述需求也得上 LSM + 优化（如 key-value 分离 + time-window compaction 控墓碑）。

### 8.5 代码题（扩展本章 demo）
**题 1（扩 Demo A → leveled compaction）**：给 `ToyLSM` 的每个 SSTable 加一个 `level` 字段，实现：(a) flush 进 L0；(b) 当某层文件数/总大小超过阈值（L0 按文件数、L1+ 按字节 ×10 规则，照搬 §3.5 的 `MaxBytesForLevel` 与 `Finalize` 逻辑），挑一个文件 + 下层所有重叠文件归并下推；(c) 维护 L1+ "范围不重叠"不变性。验收：(i) 点查/范围查结果与未分层版本一致；(ii) 打印每层文件数，写入 10 万条后层级分布接近 1:10:100。

**题 2（扩 Demo B → 含 compaction 的总写放大）**：给 `CountingLSM` 加 `compact()`——每攒 N 个 run 归并成一个推到下层，累计 `bytes_written`。跑 50 万随机插入，画出"层数 vs 总写放大"，验证总写放大随层数线性增长、量级到 ~10x（复现 RocksDB "often > 10"，§5）。对照 `CountingBTree` 的 ~9.4x，讨论：为什么 LSM 总写放大和 B-Tree 量级相近，但 LSM 仍快很多？（提示：顺序 vs 随机 I/O + 前台 vs 后台）。

**题 3（墓碑 GC，扩 Demo A）**：构造"插入 K 条 → 删除其中一半 → 反复"的工作负载，统计 compact 前后墓碑数量与文件大小，复现 §6.3 的墓碑堆积；再实现"墓碑只在最底层 compaction 时才物理删除"的规则，观察墓碑被正确清理的条件。

---

## 附录：本章核实过的出处（均已 WebFetch / WebSearch 实际取过）

**真实源码（raw.githubusercontent.com/google/leveldb/main/...，逐字核实）**
- `db/memtable.cc` — `MemTable::Add` / `MemTable::Get` / `EncodeKey`
- `db/skiplist.h` — `RandomHeight` / `Insert` / `kMaxHeight=12` / `kBranching=4`
- `db/version_set.cc` — `MaxBytesForLevel` / `Finalize` / `PickCompaction`(size vs seek) / `allowed_seeks` / `UpdateStats`
- `db/dbformat.h` — `kNumLevels=7` / `kL0_CompactionTrigger=4` / `kL0_SlowdownWritesTrigger=8` / `kL0_StopWritesTrigger=12` / `kMaxMemCompactLevel=2` / `ValueType{kTypeDeletion,kTypeValue}`
- `include/leveldb/options.h` — `write_buffer_size=4MB` / `max_file_size=2MB` / `block_size=4KB` / `compression=kSnappy`
- `doc/table_format.md` — SSTable 布局 / Footer 结构 / magic `0xdb4775248b80fb57`
- `doc/impl.md` — 层 10MB 规则 / 2MB 文件切分 / L0 重叠原因

**论文与文档（核实）**
- Bayer & McCreight (1972), *Organization and Maintenance of Large Ordered Indexes*, Acta Informatica 1:173–189（B-Tree 起源、Boeing、命名）— Springer / Wikipedia 核实
- O'Neil, Cheng, Gawlick, O'Neil (1996), *The Log-Structured Merge-Tree*, Acta Informatica 33(4):351–385 — cs.umb.edu/~poneil/lsmtree.pdf 核实（"double the I/O cost / fifty percent"、C0/C1 rolling merge、five-minute rule）
- RocksDB wiki: Leveled Compaction（multiplier=10、写放大 "often > 10"）/ Universal Compaction（space amp "200=triple"、`max_size_amplification_percent`）— github.com/facebook/rocksdb/wiki 核实
- WiscKey (Lu et al., FAST '16) — key-value 分离、2.5×–111× 加载、46×/111× 吞吐 — usenix/搜索核实

**注**：本章所有 demo 输出为本地 Python 3 实测结果，随机种子固定（seed=42）可复现。RocksDB 的 `max_subcompactions`/`RateLimiter` 行为、Cassandra 墓碑阈值（1000 / 100000）依据官方文档常见默认值，未单独逐条 WebFetch 核实，标记为常识性配置；若用于生产决策请以你所用版本的官方文档为准。
