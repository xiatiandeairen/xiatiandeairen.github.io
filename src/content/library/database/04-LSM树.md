---
title: "LSM 树:把随机写变顺序写,memtable、SSTable 与合并的代价"
slug: "04"
collection: "database"
order: 4
summary: "第 03 章的 B+树把每次写改成磁盘上某一页的原地更新,随机键写就是随机页写;这章反过来,把写攒在内存里排好序再整批顺序刷盘——这就是 LSM 树。代价是一个键散落在多层 SSTable 里,读要归并多个 run,我们用布隆过滤器和 compaction 去压这个代价,并实测写放大/读放大/空间放大三者怎么互相挤。SSTable 的不可变性也为第 05 章 WAL 的顺序写哲学埋了同一个伏笔。"
topics:
  - "数据库"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

第 03 章那棵 B+树有个你当时可能没在意的性质:`insert` 找到目标 leaf,就地改那一页,然后把这一页写回。键是随机的,leaf 就是随机的,于是磁盘看到的是一连串随机位置的 4KB 页写。机械盘上每次随机写是一次寻道(磁头移到目标磁道),SSD 上是一次读-改-写整个擦除块。B+树读快(三层定位十亿行,见上一章),但它的写本质上是随机 IO,这是它的硬伤。

LSM 树(Log-Structured Merge tree,日志结构合并树)的整个设计就是一句话:**别原地改,把写攒起来,排好序,整批顺序刷下去**。顺序写在任何介质上都比随机写快一个数量级。但天下没有白拿的顺序写——你把"在哪儿改"这个问题从写时推迟到了读时。一个键的最新值现在可能在内存里,也可能在上周刷下去的某个文件里,读的时候你得去好几个地方找。这一章就是把这套机器搭出来,然后**量出**这笔账到底多大。

配套代码在 `examples/database-from-scratch/src/stage04-lsm-tree.ts`,跑 `npx tsx src/stage04-lsm-tree.ts`。它复用第 01 章的 `core/disk.ts`(带计数的页式磁盘)和 `core/page.ts`(slotted page,槽式页),所以下面所有"写了多少字节"都是真从那个 Disk 上数出来的,不是估的。

## memtable:写先落在内存里的有序结构

LSM 的写路径第一站是 memtable——内存里的一张有序表,缓冲还没刷盘的写。代码里它就是一个 Map:

```ts
// stage04-lsm-tree.ts
private memtable = new Map<string, Entry>();

put(keyValue: Value, value: Uint8Array): void {
  const key = encodeKey(keyValue, KEY_TYPE);
  this.stats.logicalBytesPut += key.length + value.length;
  this.memtable.set(hex(key), { key, payload: value });
  if (this.memtable.size >= MEMTABLE_FLUSH_THRESHOLD) this.flush();
}
```

注意两件事。第一,key 用 hex 字符串做 Map key——因为 `Uint8Array` 在 JS 里按引用比较,不能直接当 Map 的键用,hex 能无损往返且让 memtable 成为真正的 last-writer-wins(后写覆盖先写)。第二,`put` 不碰磁盘,只改内存,所以单次 put 是 O(1) 且零 IO。这是 LSM 写快的第一层原因:**绝大多数写根本没落盘**,它们攒在 memtable 里,等着被后写覆盖或被整批刷走。

生产里 memtable 通常是跳表(skip list,一种有序链表)或红黑树,因为刷盘时需要按键有序输出,而且要支持范围扫描。这里用 Map + 刷盘时一次性排序,是因为本章要量的是磁盘 IO,memtable 内部数据结构对那些数字没影响——刷出去的 SSTable 字节数一样:

```ts
private flush(): void {
  if (this.memtable.size === 0) return;
  const entries = [...this.memtable.values()].sort((a, b) => compareKeys(a.key, b.key));
  const meta = flushEntriesToSSTable(this.disk, 0, entries);
  this.levels[0].push(meta);
  this.memtable.clear();
  this.stats.flushes++;
  this.maybeCompact();
}
```

**怎么坏 / 为什么这么设计**:memtable 是纯内存的,进程崩了它就没了。本章的 LSM 没有崩溃保护——这不是疏忽,是分工。崩溃恢复靠的是先把每条写记进一个顺序追加的日志(WAL,预写日志),那是第 05 章的主题。这里你只需要记住一个张力:memtable 阈值越大,攒的写越多,刷盘越少(写放大越低),但崩溃丢的未刷数据越多、占的内存越大。本章把阈值设成很小的 `MEMTABLE_FLUSH_THRESHOLD = 1000` 条,纯粹是为了让 5 万次写触发几十次刷盘、看得见 compaction;真实系统这个值是几十上百 MB。

## SSTable:刷出去的那一刻,它就不可变了

memtable 攒满,`flush` 把它排好序,整批写成一个 SSTable(Sorted String Table,有序字符串表)——一个不可变、按键排好序、只追加的磁盘文件。"不可变"是 LSM 的命门:**一旦写下去就再不修改**。这正是顺序写的来源——`flushEntriesToSSTable` 只往新页追加,从不回头改已写的页:

```ts
// 当前页满了:把它持久化,开新页。这就是顺序写模式——只追加,从不 seek 回去。
if (off + recordSize > PAGE_SIZE) {
  flushPage();
  page = new Uint8Array(PAGE_SIZE);
  writer = new PageWriter(page);
  writer.initSlotted(PageType.LEAF);
  pageId = disk.allocPage();
  off = SLOTTED_HEADER.HEADER_SIZE;
}
```

每条记录的磁盘布局是 `[keyLen:u16][key 字节][valLen:u32][val 字节]`,记录在页内按键升序连续摆放,不跨页。`PAGE_SIZE` 是 4096(见 `core/disk.ts`),`valLen` 用一个哨兵值 `0xffff_ffff` 表示墓碑(tombstone)。

墓碑值得停一下。删除在 LSM 里是个反直觉的操作:你**没法**伸手进一个不可变的老 SSTable 把某个键抠掉。所以删除被记成一个标记——墓碑——它遮蔽(shadow)所有更老的同键值:

```ts
delete(keyValue: Value): void {
  const key = encodeKey(keyValue, KEY_TYPE);
  // 删除也是一次逻辑写:它仍占 key 字节,且在 compaction 收走它之前一直占着 run 里的空间。
  this.stats.logicalBytesPut += key.length;
  this.memtable.set(hex(key), { key, payload: TOMBSTONE });
  if (this.memtable.size >= MEMTABLE_FLUSH_THRESHOLD) this.flush();
}
```

**怎么坏**:墓碑不是免费的。一个被删的键,它的墓碑会一路待在 SSTable 里,直到某次 compaction 把它和它要遮蔽的所有老版本一起合并掉才能真正消失。如果你删了一大批键又从不触发对底层的 compaction,这些墓碑会一直占空间、一直拖慢读(读到墓碑才知道"哦这个被删了")。这是 LSM 的一个经典坑:**大量删除后磁盘不降反升**,因为墓碑本身要占空间,被删数据要等 compaction 才腾出来。RocksDB 里专门有"墓碑太多触发 compaction"的策略来对付这个。

为什么 SSTable 一定要真落盘、不能图省事留个 JS 数组?因为本章要报的写放大是"物理写盘字节",这数字只有当字节真的过了那个计数的 Disk 才是真的。内存里的假 SSTable 会让头条数字变成虚构——这本书不干这个。

## 读路径:一个键散落多层,你得挨个找

写的便宜是欠的债,读时还。`get` 的逻辑是从最新往最旧找,第一个命中的 run 赢(新的遮蔽旧的):

```ts
get(keyValue: Value): Uint8Array | undefined {
  const key = encodeKey(keyValue, KEY_TYPE);
  this.stats.getCount++;

  // 1) memtable 是最新数据,免费查(无 IO)。
  const memHit = this.memtable.get(hex(key));
  if (memHit) return memHit.payload === TOMBSTONE ? undefined : memHit.payload;

  // 2) 然后逐层,最新的层(L0)优先,L0 内最新的 run 优先。
  for (let lvl = 0; lvl < this.levels.length; lvl++) {
    const runs = this.levels[lvl];
    for (let i = runs.length - 1; i >= 0; i--) {
      const meta = runs[i];
      // 键范围裁剪:免费且精确。键不在 [min,max] 里这个 run 不可能有它。
      if (compareKeys(key, meta.minKey) < 0 || compareKeys(key, meta.maxKey) > 0) continue;
      // 布隆裁剪:概率性,但说"没有"一定对。失败模式演示关的就是这一行。
      if (this.bloomEnabled && !meta.bloom.mightContain(key)) {
        this.stats.bloomSkips++;
        continue;
      }
      this.stats.ssTablesProbed++;
      const { payload } = scanSSTable(this.disk, meta, key);
      if (payload !== undefined) {
        return payload === TOMBSTONE ? undefined : payload;
      }
    }
  }
  return undefined;
}
```

这就是 **✦ 为什么 LSM 写快读慢、B+树读快写慢** 的全部答案,摆在你眼前:B+树读一个键,从根到 leaf 三次页定位就到位,因为它把每个键放在唯一确定的位置(原地更新的代价就是写时要去那个位置,随机)。LSM 把随机写攒成顺序刷,代价是**它放弃了"每个键有唯一位置"这个性质**——同一个键的多个历史版本散落在 memtable、L0 的几个 run、L1 的大 run 里,读必须从新到旧把这些地方都问一遍,直到命中。写时省下的随机 IO,变成了读时的多 run 归并。这不是哪个实现没优化好,是两种数据结构在"写位置确定性"上的根本取舍。

注意上面有两道**免费**的剪枝在帮读:键范围裁剪(`minKey/maxKey` 直接排除)和布隆过滤器。下一节说后者。

## 布隆过滤器:用一点内存换掉大量无效查表

布隆过滤器(Bloom filter)是一个概率性的集合判断:它对"在不在这个集合里"的回答有个不对称性——**说"一定不在"永远对(无假阴性),说"可能在"有时错(有假阳性)**。对 LSM 读路径这正好是要的:一个"一定不在"让我们零风险地跳过一个 SSTable 的整次扫描 IO。

实现是每个 SSTable 在刷盘时从它的键建一个位集:

```ts
mightContain(key: Uint8Array): boolean {
  const [h1, h2] = this.hashPair(key);
  for (let i = 0; i < this.numHashes; i++) {
    const bit = ((h1 + Math.imul(i, h2)) >>> 0) % this.numBits;
    if ((this.bits[bit >>> 3] & (1 << (bit & 7))) === 0) return false;
  }
  return true;
}
```

代码用 Kirsch-Mitzenmacher 双哈希技巧 `h_i = h1 + i*h2` 从两个基哈希派生出 k 个哈希,比算 k 个真哈希便宜且统计上够用。位数按目标约 1% 假阳性率定(每键约 10 bit,7 个哈希)。

实测它有多值:跑 stage04,2 万次查询(70% 命中、30% 必然落空),布隆开启时:

```
读放大 (bloom 开启)  gets=20000  sstables_probed=12536  bloom_skips=23001  probes_per_get=0.627
  bloom 过滤掉的无效查表 = 23001 次
```

布隆替我们跳掉了 **23001 次** 本来要做的 SSTable 扫描。平均每次 get 只真正扫了 0.627 个 SSTable。

### 失败模式:把布隆关掉,读放大灾难

现在做这章最该看的实验——**同样的数据、同样的查询,只把布隆那一行关掉**(`bloomEnabled = false`,键范围裁剪还留着),重新量:

```
config      sstables_probed  bloom_skips  probes_per_get
bloom 开启            12536        23001           0.627
bloom 关闭            35537            0           1.777
  查表次数膨胀 = 2.8x
```

查表次数翻了 **2.8 倍**。平均每次 get 从扫 0.627 个 SSTable 涨到 1.777 个。再看墙钟延迟(真测的,机器相关):

```
config      ns_per_get_measured  gets_per_sec_measured
bloom 开启              84038.1                  11899
bloom 关闭             108349.2                   9229
  延迟膨胀 = 1.29x (measured, 机器相关)
```

这里有个**必须诚实交代的标注**:这是 RAM 盘,没有寻道延迟,所以延迟只涨了 1.29 倍。**这个绝对数字别往真实系统上搬**。真实磁盘/SSD 上,每多扫一个 SSTable = 一次随机 IO——机械盘上是几毫秒一次寻道,SSD 上是一次页错误。在那种介质上,"查表次数 2.8 倍"会直接变成"延迟 2.8 倍量级"甚至更糟,而不是这里的 1.29 倍。**可迁移的是 2.8x 这个相对趋势**(关掉布隆 → 读放大灾难),不是 1.29x 这个 ns 比值。我把这两个数分开报,就是要你别把 toy 的绝对值当真。

为什么不能没有布隆?因为 L0 的多个 run 键范围是重叠的——你查一个键,光靠 `minKey/maxKey` 裁剪挡不掉那些范围覆盖了它但其实没有它的 run,你得真扫进去才知道扑空。布隆就是在"扫进去"之前先问一句"你到底有没有",大部分时候它能替你省掉这次扑空的 IO。

## compaction:用更多写放大,买更少读放大

布隆压的是单 run 的扫描成本,但 run 的**数量**还在涨——每刷一次 memtable 就多一个 L0 run,读放大的天花板是 run 总数。compaction(合并)就是来管这个的:把一层里太多的小 run 归并成下一层一个大 run,run 少了,读要问的地方就少了。

触发条件:L0 用 run **数量**阈值(L0 的 run 重叠,数量直接伤读),更深的层用按 fanout 增长的数量预算保持树浅:

```ts
private maybeCompact(): void {
  let lvl = 0;
  for (;;) {
    const runs = this.levels[lvl];
    const overLimit = lvl === 0
      ? runs.length > L0_COMPACTION_TRIGGER
      : runs.length > levelRunBudget(lvl);
    if (!overLimit) break;
    // ... 读出本层 + 下一层所有 run(键范围重叠),最新优先归并 ...
    const merged = mergeRuns(runsNewestFirst, isBottom);
    // 用合并出的单个 run 替换两层 ...
    this.stats.compactions++;
    lvl = nextLvl; // 级联:这次合并可能把下一层也撑过预算
  }
}
```

归并的核心是 k 路归并,新 run 在键相同时赢(它的值/墓碑遮蔽老的),这就是 LSM 把散落各 run 的多个版本"对账"成单一当前值的地方:

```ts
// mergeRuns: 在所有 run 头里找最小键;键相同时下标最小(调用方传"最新优先")的 run 赢,
// 推进所有持有该键的 run——把重复版本塌缩掉。
if (winner.payload === TOMBSTONE && dropTombstones) continue;
out.push(winner);
```

`dropTombstones` 只在归并到最底层时为真——只有那时下面没有更老的 run 还需要被这个墓碑遮蔽,墓碑才能安全丢掉。这呼应前面说的墓碑生命周期。

**这就是 ✦ 那笔账的另一半**:compaction 要把它合并的 run 全部**重新读出来再重新写下去**。同一份数据,刷进 L0 写一遍,合进 L1 又写一遍,合进 L2 再写一遍……每下一层就重写一次。这个重复写就是写放大的主要来源。所以 LSM 的内部逻辑是个三方拉锯:

- **布隆**:几乎免费地压读放大(只花一点内存)。
- **compaction**:用**更多写放大**去买**更少读放大**——合得越勤,run 越少读越快,但重写越多写越费。
- **memtable 阈值 / 层 fanout**:调这俩等于在三个放大之间挪。

跑出来的写放大:

```
写放大 (write amplification)  logical_bytes=2014527  physical_bytes=4612096  ratio_x=2.29
flush 次数=47, compaction 次数=5
各层 run 数 [L0..]: [2, 1]
```

物理写了 **2.29 倍** 于逻辑 put 的字节。超过 1 倍的部分,一是 compaction 的重写,二是——这里有个老实话——4KB 页内没填满的空间也算进物理字节了(每个 SSTable 的最后一页几乎都有空尾)。真实 LSM 同样如此,所以我没把它从分子里抠掉。47 次刷盘最后被 5 次 compaction 收成 L0 两个、L1 一个 run 的形状。

至此第三种放大也露面了:**空间放大**(space amplification)。同一个键的多个版本、还没被 compaction 收走的墓碑、页内碎片,都让磁盘上的物理字节多于逻辑数据。本章的 Disk 故意没有 free-list(空闲页回收链表,见 `core/disk.ts` 注释:回收是后续 stage 的练习),compaction 写新 run 后旧页并不归还——这些孤儿页仍计入物理写,这是诚实的:真实 LSM 也是先写完新 run、旧页才可回收,中间那段时间空间放大是实打实存在的。

写放大、读放大、空间放大——你**没法三个同时压到最低**,这是 LSM 的不可能三角。compaction 合得勤,读放大和空间放大降,写放大升;合得懒,写放大降,另两个升。所有 LSM 引擎的调参,本质都是在这个三角里挑你的工作负载能接受的那个点。

最后,正确性不是靠看上去对——代码用一个朴素 Map 重放整个工作负载当 ground-truth,抽样对比 LSM 的 get 结果:

```
正确性校验: LSM get 结果与逐操作 ground-truth 全部一致. ✓
```

刷盘、compaction、墓碑这套机器但凡有个版本对账 bug,这里会炸成不一致,而不是上面那些"看着合理其实错了"的数字。

## ⚡ 前沿:这个三角现在被怎么撬

LSM 的三角不是定死的,过去十年一直有人从不同角度撬它。三条值得知道,且**都还没有通用解**:

**leveled vs tiered compaction。** RocksDB 同时支持两种 compaction 策略。leveled(分层,本章实现的方向)每层维护非重叠的大 run,读放大低、空间放大低,但写放大高(每次合并重写整层)。tiered(分级,Cassandra/ScyllaDB 默认)攒够几个同样大小的 run 才合,写放大低,但读放大和空间放大高(一层里多个 run 重叠)。**没有哪个对所有负载都好**:写多读少选 tiered,读多写多选 leveled,RocksDB 甚至支持混合(hybrid,L0 用 tiered、深层用 leveled)。怎么根据负载自动选/切换,至今是调优工程师的手艺活,没有自动最优解。

**WiscKey 的键值分离。** 写放大的大头是 compaction 反复重写**整条记录**——可值(value)往往比键大得多,而归并其实只需要键有序。WiscKey(2016)的想法:把 value 挪出 LSM,单独存在一个只追加的 value log 里,LSM 里只留键 + 指向 value 的指针。compaction 只重写键和指针,大幅砍写放大。代价是范围扫描时 value 不再连续(要按指针去 value log 随机读),以及 value log 自己需要垃圾回收。RocksDB 的 BlobDB、TiKV 的 Titan 都是这条路的工程化,但**值多大才值得分离、value log GC 怎么不抵消掉收益,仍是 per-workload 的开放问题**。

**云上 LSM 与对象存储解耦。** 本章的 SSTable 落在本地块设备上。云原生数据库(如 Neon、各种 serverless 引擎)正在把 SSTable 直接放进对象存储(S3 这类)——因为 SSTable **不可变**,天然适配对象存储"写一次、不可改"的模型(这正是本章一开始强调的不可变性,在云上又变成了一个解耦的支点)。这样计算和存储分离,compaction 可以扔给独立的无状态节点跑,存储无限弹性。但对象存储的高延迟(几十毫秒一次 GET)让读路径必须叠厚厚的本地缓存,**缓存失效、compaction 的网络成本、对象存储的请求计费怎么平衡,目前各家方案分歧很大,远没收敛**。

---

这章把第 03 章那棵读快写慢的 B+树翻了个面,做出一棵写快读慢的 LSM,并量出了"快"和"慢"各自的代价单位:写放大 2.29 倍、布隆开关之间读放大差 2.8 倍。贯穿全章的不可变 SSTable——写下去就不改、只能靠新 run 遮蔽和 compaction 合并——不是某个实现的偶然,它和第 05 章 WAL 的"先把变更顺序追加到日志再谈持久化"是同一个哲学:**顺序写是地基,随机改是要竭力避开的东西**。下一章我们就去看,当 memtable 还没刷、进程就崩了,那条顺序追加的日志怎么把数据从断电里救回来。
