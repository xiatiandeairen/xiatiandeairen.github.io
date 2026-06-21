---
title: "PagedAttention：像操作系统管内存一样管 KV 缓存"
slug: "05"
collection: "llm-inference"
order: 5
summary: "第 04 章的连续批让 GPU 不再空转，但它把每个请求都按 maxSeq 预留连续 KV 空间，实测内部碎片高达 62.8%。这章借操作系统的虚拟内存思路把 KV cache 切成定长 block、用 block table 做逻辑到物理的映射，消除碎片、把固定显存下的并发数翻 4 倍，并证明分页 bit-for-bit 不改变输出（drift=0）。最后把 block 共享推到 prefix caching 雏形，接上第 06 章投机解码对吞吐的下一刀。"
topics:
  - "LLM 推理"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

第 04 章的连续批（continuous batching，一有请求结束就立刻补新请求进 batch）把 GPU 的空转填满了，但它有一个被我们刻意放过去的账没算：每个请求的 KV cache（注意力的键值缓存，自回归解码时缓存历史 token 的 K/V 避免重算）到底占多少显存。答案很难看。跑一下本章的配套程序你会看到这一行：

```
连续预留 (每请求占 maxSeq=256) = 1.50 MiB (est.), 内部碎片 = 62.8%
```

三个请求，实际只用了 572 KiB 的 KV，却预留了 1.50 MiB，**62.8% 的显存是预留了从没碰过的**。在生产里 KV cache 是显存的头号消费者，连续批能塞多少请求几乎完全由 KV 显存决定——不是由算力决定。所以这一章不解决"算得快不快"，解决"**同样的卡能不能多塞几倍请求**"。这是把第 04 章调度器的内存利用率从"够用"推到"榨干"的关键一跳。

本章配套代码：`examples/llm-inference-from-scratch/src/stage05-paged-kv.ts`。下文所有数字都来自它的真实运行输出，估算值标 `(est.)`。

## 一、连续 KV 缓存怎么坏的：内部碎片

### 为什么连续缓存必须预留 maxSeq

先回到第 02 章建立的连续 KV cache。它的物理形态是一段连续数组，按 `[maxSeq, kvDim]` 布局，解码时一个位置一个位置往里写。问题出在"它得提前知道要多长"。

自回归解码是一边生成一边追加的：你发起请求时根本不知道模型会吐 3 个 token 还是 235 个 token。连续数组要支持"再追加一个位置"有两条路：要么每加一个 token 就 realloc + 拷贝整段历史（O(n²) 的拷贝量，生产里不可接受），要么一次性按最坏情况 `maxSeq` 预留到顶、之后只往里填。生产引擎选后者——**为每个请求预留 maxSeq 个连续槽位**。

代价就是上面那 62.8%。配套程序 (a) 段模拟了一个真实并发批：三个请求的实际长度分别是 3、48、235 token（来自 `core/tokenizer.PROMPTS`），但连续缓存对每个都按 `maxSeq=256` 预留：

```
[a] 内存占用: 连续预留 vs 分页 (同一批并发请求)
    并发请求实际长度 = [3, 48, 235] tokens
    实际用到的 KV (3 请求合计) = 572.00 KiB (est.)
    连续预留 (每请求占 maxSeq=256) = 1.50 MiB (est.), 内部碎片 = 62.8%
```

那个 3-token 的请求最惨：用了 3 个槽，占着 256 个。这种"预留了但没用到的尾巴"在操作系统里有个名字——**内部碎片（internal fragmentation，已分配但用不上的空间）**。

### 这是设计缺陷不是参数没调好

有人第一反应是"把 maxSeq 调小点不就行了"。不行。`maxSeq` 是你愿意支持的最长对话上限，调小等于砍功能；而且只要 `maxSeq` 远大于"典型请求实际长度"——这在真实流量里永远成立，长尾请求少、短请求多——碎片就消不掉。**碎片的根因是"用连续数组表示一个长度未知的序列"这个数据结构选择本身**，不是某个旋钮。

这正是 2023 年 vLLM 那篇 PagedAttention 论文的出发点：他们测真实工作负载，发现连续 KV cache 的有效利用率常常只有 20%-40%，剩下全是这种碎片加上"为了对齐预留的整块"。要根治，得换数据结构。

## 二、把 OS 虚拟内存搬过来

### ✦ 真正的洞见：打破"KV 是连续数组"这个假设

操作系统几十年前就解决过一模一样的问题。进程要一段"看起来连续"的内存，但物理内存被切成定长的页（page），进程拿到的是逻辑地址，靠一张**页表（page table）**把逻辑页映射到任意物理页帧——物理上根本不连续。这样物理内存按页粒度按需分配，碎片被限制在"最后半页"。

PagedAttention 就是把这套搬到 KV cache 上。这里要点破一件事：**PagedAttention 真正的洞见不是"分页"这个机械动作，而是它敢于打破"KV cache 必须是一段连续数组"这个被所有人默认接受的前提**。一旦你接受"KV 可以非连续寻址、靠一张表去查",batch 能塞的请求数就能翻数倍——后面 (b) 段会给出 4 倍的实测。分页只是兑现这个洞见的手段;洞见本身是"质疑那个没人质疑的假设"。

映射关系是这样的（注意力机制术语对操作系统术语）：

| 操作系统 | PagedAttention | 在本章代码里 |
|---|---|---|
| 物理页帧 | KV block（定长，存 `blockSize` 个位置的 K/V） | `PagedKVPool.blocksK/blocksV` |
| 页表 | block table（逻辑块号 → 物理块号） | `PagedSeq.blockTable` |
| 缺页 → 分配页帧 | 写到新逻辑块 → 分配物理 block | `pagedStep` 里的 `allocBlock` |
| 空闲页链表 | free list（回收的 block 等待复用） | `PagedKVPool.freeList` |

### block size：唯一的调优旋钮

block 多大是核心权衡，所以代码里它是个命名常量不是字面量：

```ts
// examples/llm-inference-from-scratch/src/stage05-paged-kv.ts
const BLOCK_SIZE = 8;
```

block 越小，最后半块的浪费越少（内部碎片上限就是 `blockSize`，与 `maxSeq` 无关了），但 block table 条目越多、间接寻址开销越大;block 越大反过来。真实引擎一般用 16。这里用 8，是为了让长度 3/48/235 的 toy 请求横跨好几个 block,让"最后那个没填满的块"在数字上看得见,而不是被取整抹平。

`PagedKVPool` 是一个所有序列共享的物理 block 池。每个 block 对**一层**存 `blockSize` 个位置的 K 和 V,行主序平铺成 `[blockSize, kvDim]`——这个布局是刻意选的,它和第 02 章连续 cache 打包 K/V 的布局**字节兼容**,所以把一串 block 收集起来,就能直接喂给原来的注意力 kernel。这个细节是第三节"输出不变"能成立的全部秘密。

`allocBlock` 是分配器,优先从 free list 拿回收块,拿不到才向池子申请新块——和 OS 优先复用空闲页帧一个套路:

```ts
// examples/llm-inference-from-scratch/src/stage05-paged-kv.ts — allocBlock
function allocBlock(pool: PagedKVPool): number {
  const idx = pool.freeList.length > 0 ? pool.freeList.pop()! : pool.nextFresh++;
  // ...为每一层把该物理块的 K/V 数组按需分配并清零...
  return idx;
}
```

每个序列自己持有一张 block table 加一个已填位置数 `len`:

```ts
type PagedSeq = {
  blockTable: number[]; // 逻辑块 i 存在物理块 blockTable[i]
  len: number;          // 当前已缓存的位置数
};
```

### 一步解码:gather → 算 → scatter

`pagedStep` 是分页版的单步解码,对照第 02 章的 `forwardStep` 看,它和连续路径**唯一的区别就在 K/V 历史怎么取、怎么存**:

```ts
// examples/llm-inference-from-scratch/src/stage05-paged-kv.ts — pagedStep 核心
// 缺页式分配:这个 token 要落的逻辑块还没有物理块,就分一个
const logicalBlock = Math.floor(pos / BLOCK_SIZE);
if (seq.blockTable[logicalBlock] === undefined) {
  seq.blockTable[logicalBlock] = allocBlock(pool);
}
// ...每层:
//  GATHER —— 把分散在各 block 里的 pos 行历史,收集进一段连续 scratch,
//            正好是 blockStep 期望的 [pos, kvDim] 布局
const kHist = new Float64Array(pos * kvDim);
for (let p = 0; p < pos; p++) {
  const phys = seq.blockTable[Math.floor(p / BLOCK_SIZE)];
  kHist.set(pool.blocksK[l][phys].subarray(/* 该位置在块内的偏移 */), p * kvDim);
}
const r = blockStep(h, w, pos, kHist, vHist, pos, cfg); // 完全相同的注意力运算
//  SCATTER —— 把刚算出的这一行 K/V 写回它对应的物理块槽位
pool.blocksK[l][phys].set(r.k, slotInBlock * kvDim);
```

三步:**gather**(把非连续的历史块收拢成连续 scratch)→ 跑**和连续路径逐字节相同的** `blockStep` → **scatter**(把新 K/V 散写回物理块)。这个 gather 是分页用运行时换显存付的成本。在 toy 里我们把它显式写出来让机制可见;真实引擎会把 gather 融进注意力 kernel(所谓"paged attention kernel"),让它近乎免费——这是工业实现和教学实现的关键差距,值得记一笔。

## 三、最该守的不变量:分页不准改变输出

分页只是搬字节,**数学一个比特都不能动**。这是整章承重的不变量:如果分页悄悄改了 logits,那它就不是"优化"而是"引入了静默的精度 bug",再省显存也得回滚。

配套 (c) 段把分页路径的最后一个 token 的 logits 和两个参照逐位对拍:一个是 O(seq²) 的无缓存参照 `forwardNoCache`,一个是第 02 章的连续缓存 `forwardStep`。三个长度都测——这是刻意避开 N=1:分页 bug 常常藏到"序列跨过 block 边界"才暴露,48 和 235 都跨了多块,3 也跨了不止一块(blockSize=8 下 3 token 占 1 块,48 占 6 块,235 占 30 块):

```
[c] 正确性: 分页注意力 logits 逐位对拍 (drift 必须 ≈0)
    len=  3 (跨1块): drift vs reference=0.00e+0, vs contiguous=0.00e+0, argmax match=true
    len= 48 (跨6块): drift vs reference=0.00e+0, vs contiguous=0.00e+0, argmax match=true
    len=235 (跨30块): drift vs reference=0.00e+0, vs contiguous=0.00e+0, argmax match=true
    最坏 drift = 0.00e+0 -> 分页不改变任何输出 (bit-for-bit)
```

最坏 drift = `0.00e+0`,bit-for-bit 一致。为什么能做到零漂移、连浮点最后一位都不差?因为 gather 把分散的块收拢成的那段 scratch,布局和连续 cache 完全一样,送进的是**同一个** `blockStep` 函数。运算顺序不变、加法结合性不变,浮点结果就一位不差。**正确性不是靠运气,是靠"复用同一个运算核"这个设计决策买来的**——这也是为什么 `pagedStep` 宁可显式 gather 也不另写一套注意力:另写一套,迟早因为求和顺序不同引入 1e-7 级别的漂移,然后你要花一周去查"为什么分页后第 200 个 token 偶尔不一样"。

## 四、固定显存预算下,并发数翻 4 倍

碎片消掉了,直接收益是同样的显存能塞更多请求。(b) 段在一个固定预算下对比两种分配器:

```
[b] 固定显存预算下的最大并发请求数
    预算 = 64.00 MiB (est.), 代表性工作长度 = 64 tokens
    连续: 每请求预留 maxSeq=256 -> 512.00 KiB/请求 -> 最多 128 并发
    分页: 每请求只占 8 块 -> 128.00 KiB/请求 -> 最多 512 并发
    分页并发能力 = 连续的 4.0x
```

连续路径不管请求实际多长,都按 `maxSeq=256` 预留 512 KiB,64 MiB 只够 128 个;分页只按工作长度 64 token 占的 8 个 block(128 KiB)算,塞得下 512 个。**4 倍**。

把它接回第 04 章:连续批的吞吐上限本质是"batch 里能同时 in-flight 多少请求",而这个数被 KV 显存卡死。分页把每请求的显存占用从"按最坏预留"降到"按实际占用",直接把这个上限抬高数倍。所以 PagedAttention 不是和连续批二选一,它是连续批的**地基**——vLLM 把两者一起上,才有了那条出名的吞吐曲线。

> 诚实标注:64 MiB 预算和这些绝对 KiB 数是 toy 配置(blockSize=8,kvDim=32,nLayers=4)下的精确浮点估算,标 `(est.)`,不是真实显卡的 RSS 读数。会迁移到生产的不是这些绝对数,是那个 **4.0x 的比值**——它只取决于 `maxSeq / 工作长度` 的比例,真实场景里(maxSeq 几千、典型对话几百)这个比值往往更夸张。

## 五、block 共享:prefix caching 的雏形

分页还顺手解锁了一个连续 cache 根本做不到的能力。block table 是一层间接——既然逻辑块映射到物理块,那**两个序列的 block table 完全可以指向同一个物理块**。当多个请求共享同一段前缀(最典型:同一个 system prompt、多轮对话的历史),这段前缀的 KV block 在物理上只存一份,所有请求的 block table 都指过去。

(d) 段模拟了 4 个请求共用一个 system prompt:

```
[d] prefix 共享: 多个请求共享同一 system prompt 前缀
    system prompt = 63 tokens -> 可共享 7 个整块 (56 tokens), 余 7 tokens 不满整块
    4 个请求, 每个别名同 7 个物理块 (共 28 次别名, 0 拷贝)
    内存: 共享 21 块 vs 朴素各存一份 42 块 -> 省 336.00 KiB (est.)
    prefill 时间: 共享 = 63.33ms (前缀1次 25.03ms + 4尾巴 38.30ms)
                  朴素 ≈ 138.42ms (est.: 前缀重算4次), 加速 = 2.19x
```

两笔账都省了。**显存**:21 块 vs 朴素的 42 块,省 336 KiB。**prefill 计算**(把整段 prompt 喂进去填满 KV 的过程):前缀只算一次,加速 `2.19x`——别小看这个,prefill 是计算密集的,把 N 份重复 prefill 砍成 1 份,直接降低首 token 延迟(TTFT)和总算力。

这里有个真实约束要点破:**只能共享整块**。代码里 63 token 的 prompt 只能共享前 `floor(63/8)=7` 个整块(56 token),余下 7 个 token 不满一块,各请求自己存:

```ts
// examples/llm-inference-from-scratch/src/stage05-paged-kv.ts
// 共享前缀必须落在 block 边界上才能整块共享 —— vLLM 也只共享满块
const sharedBlocks = Math.floor(sysIds.length / BLOCK_SIZE);
```

共享是只读别名(本例 0 拷贝)。但一旦某个请求要在共享区上写——比如各自的对话分叉了——就必须先复制出自己的私有块再写,这就是 **copy-on-write(写时复制,共享只读、写时才分裂)**,和 OS 进程 fork 共享内存页是同一个机制。本章 demo 是纯只读共享,把 CoW 的触发点点到为止。

### ⚡ 前沿:从 block 共享到工业级 prefix caching

block 共享在 2024-2025 长成了一个吞吐利器,但**它仍是活跃研究区,没有放之四海的通用解**:

- **vLLM automatic prefix caching**:自动检测请求间的公共前缀并复用其 KV block,对"固定 system prompt + 变化 user 输入"的场景近乎白嫖。
- **SGLang RadixAttention**:用基数树(radix tree,按公共前缀压缩的字典树)组织所有请求的 KV,前缀共享从"线性扫描找公共块"升级成树上的前缀匹配,多轮对话和共享 prompt 场景吞吐能翻倍。

为什么说没通用解?难点全在**缓存策略**而非机制本身:缓存的 prefix block 什么时候淘汰(LRU?按命中率?)、显存吃紧时优先保谁、被复用的块被原请求改写时的 CoW 时机、跨请求的引用计数怎么在高并发下不出竞态——这些是典型的缓存管理难题,工作负载一变最优策略就变。机制(block 可共享)是确定的,**策略仍在研究**。把这一节和第 04 章连起来看:连续批解决"GPU 别空转",分页解决"显存别浪费",prefix caching 解决"重复计算别重复算"——一层一层往上榨。

## 六、失败模式:一个错条目 = 静默串话

分页用"一层间接"换了效率,代价是引入了一个连续 cache 根本不存在的新危险:**block table 必须永远正确,一个条目错了不会崩,只会静默地让你读到别人的 KV**。

(e) 段故意制造这个 bug:两个独立请求 A、B 在同一个池里 prefill 完,然后把 B 的逻辑块 0 改指向 A 的物理块 0——这正是 block 分配器 off-by-one,或者引用计数竞态会产生的错误:

```ts
// examples/llm-inference-from-scratch/src/stage05-paged-kv.ts
const goodPhys = seqB.blockTable[0];
seqB.blockTable[0] = seqA.blockTable[0]; // <-- 制造错位:B 的第 0 块指向了 A 的
```

结果:

```
[e] 失败模式: block table 映射错位 -> 读到别的请求的 KV (logits 污染)
    请求 B 正确 argmax = 255
    block table 错位后 B 的 argmax = 255 (本例恰好未翻转, 但 logits 已偏移)
    logits 漂移 = 8.04e-1 (≫0: 读到了 A 的 KV, 输出被污染)
    教训: 分页用一层间接换效率, 代价是 block table 必须永远正确 — 一个错条目=静默串话, 不崩溃, 最难查
```

注意这个失败有多阴险:argmax 这次**恰好没翻**(还是 255),但 logits 已经漂了 `8.04e-1`——比第三节那个 `0.00e+0` 的不变量差了十几个数量级。它不抛异常、不崩溃、有时连最终 token 都看着正常,但 B 的注意力实实在在读到了 A 的历史。在生产里这就是**跨请求串话**:用户 B 的回答里混进了用户 A 的上下文。这类 bug 没有栈回溯、复现还看运气(取决于 argmax 翻没翻),是分页引擎里最难查的一类。

这也解释了为什么真实引擎在 block 分配/释放/引用计数这套上极度谨慎、测试覆盖拉满:第三节的 `drift=0` 是分页的承诺,而这个承诺**完全建立在 block table 永不出错之上**。间接寻址是把双刃剑——它给了你共享和按需分配,也给了你一个"错了不报警"的攻击面。

## 小结与承接

这一章做的事:把 KV cache 从"每请求一段按 maxSeq 预留的连续数组"换成"定长 block + block table 映射",拿操作系统虚拟内存的成熟思路,消掉了 62.8% 的内部碎片,在固定显存下把并发从 128 推到 512(4.0x),还证明了这一切 bit-for-bit 不改变输出(drift=0)。顺带靠 block table 的间接性解锁了 prefix 共享,省显存又省重复 prefill(2.19x),这就是工业级 prefix caching 的雏形。代价是认清了一个新危险:间接寻址下,一个错的 block table 条目会静默串话。

到这里,第 04 章(GPU 别空转)+ 第 05 章(显存别浪费)已经把"把现有算力和显存榨干"这条线走完。第 06 章换思路——不再榨现有资源,而是**用小模型猜、大模型验**的投机解码(speculative decoding)去攻"自回归一次只能出一个 token"这个延迟根因,从另一个维度再要一刀吞吐。
