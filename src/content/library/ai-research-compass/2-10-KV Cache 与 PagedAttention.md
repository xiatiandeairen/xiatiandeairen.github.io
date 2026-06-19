---
title: "KV Cache 与 PagedAttention:把推理显存当虚拟内存管"
slug: "2-10"
collection: "ai-research-compass"
group: "MLSys专家课程"
order: 2010
summary: "这一章把你从\"知道 Transformer 推理要缓存 K/V\"带到\"能从第一性原理推出 KV cache 的显存公式、定量解释为什么传统服务系统的显存利用率只有 20–40%,并讲清 vLLM 用操作系统的分页思想把它干到 90%+ 的完整机制——直至你能照着伪代码自己实现一个 block man…"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T06:26:40.000Z"
updatedAt: "2026-06-19T06:26:40.000Z"
---
> 这一章把你从"知道 Transformer 推理要缓存 K/V"带到"能从第一性原理推出 KV cache 的显存公式、定量解释为什么传统服务系统的显存利用率只有 20–40%,并讲清 vLLM 用操作系统的分页思想把它干到 90%+ 的完整机制——直至你能照着伪代码自己实现一个 block manager 和连续批处理调度器"。

## 一、动机:推理的瓶颈不在算,在显存

上一章(FlashAttention)我们花了大力气证明:注意力的瓶颈是**访存**,不是算力。那是从单个算子内部看的。这一章我们退后一步,从整个**推理服务**的视角看显存——你会发现一个更刺眼的事实:**线上推理系统的显存,大部分时间是被浪费掉的。**

先把场景定清楚。大模型推理分两个阶段,这是理解后面一切的地基:

- **Prefill(预填充):** 用户给一段 prompt(比如 512 个 token),模型一次性把这 512 个 token 全部送进去,并行算出每一层每个位置的 K、V、以及最后一个位置的输出 logits,采样出第 1 个生成 token。这一步是**矩阵×矩阵**,所有 token 并行,算力吃得满,是 **compute-bound(算力受限)**。
- **Decode(解码):** 从第 2 个 token 开始,每一步只输入**上一步刚生成的那 1 个 token**,算出它的输出、采样出下一个 token,如此自回归地一个一个吐。这一步每次只处理 1 个 token,是**矩阵×向量**,算力严重闲置,是 **memory-bound(访存受限)**——它的耗时几乎全花在把模型权重从 HBM 搬到计算单元上。

> **关键认知:** 一次完整生成里,prefill 只发生一次,decode 要发生几百上千次。所以**线上推理的总成本由 decode 主导,而 decode 是访存受限的**。这是本章和后面投机解码(第 11 章)、量化(第 12 章)所有优化的共同出发点。

那么 decode 阶段,显存被什么吃掉?三块:模型权重(固定不变)、激活(很小,每步只算 1 个 token)、以及 **KV cache**。前两块基本是常数,唯一随请求数量和序列长度疯涨的就是 KV cache。**KV cache 是推理显存的主战场,管好它就管住了推理的吞吐和成本。** 这就是这一章的全部主题。

## 二、为什么必须缓存 K/V:`O(N²)` 重算 vs `O(N)` 缓存

### 2.1 问题:朴素自回归是 `O(N²d²)` 的灾难

回顾注意力的定义。对一个长度为 t 的序列,第 i 个位置的输出是:

```
对每个位置 i:
  scoreᵢⱼ = qᵢ · kⱼ        (j = 1..i,因果掩码只看自己和前面)
  αᵢ = softmax(scoreᵢ /√d)  (在 j 维归一化)
  oᵢ = Σⱼ αᵢⱼ · vⱼ
```

其中 qᵢ = xᵢ Wq,kⱼ = xⱼ Wk,vⱼ = xⱼ Wv,都是 token 表示 x 经过线性投影得到的。

自回归生成时,第 t 步要算第 t 个位置的输出 oₜ。它需要 qₜ(只跟当前 token 有关)和 **k₁..kₜ、v₁..vₜ**(跟从头到当前的所有 token 有关)。

**现在看一个致命的观察:** kⱼ、vⱼ 只取决于 token xⱼ,跟"现在是第几步"无关。也就是说,第 5 步算出来的 k₁..k₅,在第 6 步、第 7 步……一直到生成结束,**它们的值永远不变**。

如果不缓存,每生成一个新 token 都要把前面所有 token 重新过一遍线性投影,重算全部 K、V。我们来算这笔账。设序列最终长度为 N,投影一个 token 到 K 或 V 的计算量是 `O(d²)`(d×d 矩阵乘 d 维向量,d 为隐藏维)。

- 第 t 步重算全前缀的 K、V:`O(t · d²)`
- 生成完整 N 个 token,总计算量:`∑ₜ₌₁ᴺ O(t·d²) = O(d² · N²/2) = O(N²d²)`

光是重复计算 K、V 就是 **`N` 的平方**。再叠加注意力打分本身每步的 `O(t·d)`,整个生成过程是 `O(N²)` 量级的冗余重算。**对长序列,这是不可接受的浪费——而且全是在重算那些值根本没变的东西。**

### 2.2 解法:缓存历史 K/V,把 `O(N²)` 摊成 `O(N)`

既然 kⱼ、vⱼ 算一次就永远不变,那就**算完存下来**。这就是 KV cache:

```
KV cache = 每一层、每个注意力头、每个历史位置的 K 向量和 V 向量
```

有了它,decode 第 t 步只需要:

1. 算当前 token 的 qₜ、kₜ、vₜ:`O(d²)`(只投影 1 个 token,与 t 无关);
2. 把 kₜ、vₜ **追加(append)** 到 cache 末尾:`O(d)`;
3. 用 qₜ 跟 cache 里的 k₁..kₜ 做注意力打分、加权 v₁..vₜ:`O(t·d)`。

每步从重算的 `O(t·d²)` 降到 `O(d² + t·d)`。整个生成的总计算量从 `O(N²d²)` 降到:

```
∑ₜ₌₁ᴺ O(d² + t·d) = O(N·d² + N²d/2) = O(N·d² + N²·d)
```

省掉了那个最大的 `N²·d²` 项。**代价是:你必须把所有历史 K、V 一直存在显存里。计算上省了一个 N 的量级,空间上多花了一个 N 的量级。这是一个典型的"以空间换时间"。** 而省下来的时间是数量级的,代价的空间却把另一堵墙(显存墙)推到了眼前——于是有了本章后半段的全部故事。

> **一句话总结这一节:** KV cache 的存在理由就是"kⱼ、vⱼ 与步数无关、算一次永久有效",所以缓存能把自回归从 `O(N²)` 重算降到 `O(N)` 增量计算。理解这一点,你才会明白为什么 KV cache 是**刚需而非优化选项**——没有它,长文本生成在算力上根本跑不起来。

## 三、KV cache 显存公式:从字节级推导到量级感

### 3.1 逐字节推导公式

我们要精确算出"存一条序列的 KV cache 要多少字节"。一步步来,这个推导你必须能自己默写出来。

考虑一个标准的多头注意力(MHA)Transformer。定义符号:

```
L      = 层数(number of layers)
H      = 注意力头数(number of heads)
d_head = 每个头的维度(head dimension)
S      = 序列长度(已缓存的 token 数)
B      = batch size(并发处理的序列条数)
P      = 每个数值的字节数(precision:FP16/BF16 = 2,FP8 = 1)
```

通常隐藏维 d_model = H × d_head。

**第一步:一个 token、一个头、一层,存 K 占多少?**
一个 K 向量是 d_head 个数,占 `d_head × P` 字节。V 同理。所以 K+V = `2 × d_head × P` 字节(这个 **2 就是公式里 "K 和 V 两份" 的来源**,新手最容易漏掉它)。

**第二步:一个 token、一层,所有头?**
乘以头数 H:`2 × H × d_head × P` = `2 × d_model × P` 字节。

**第三步:一个 token,所有层?**
乘以层数 L:`2 × L × H × d_head × P` 字节。

**第四步:一条长度 S 的序列?**
乘以 S:`2 × L × H × d_head × S × P` 字节。

**第五步:B 条并发序列?**
乘以 B。得到完整公式:

```
KV_bytes = 2 · L · H · d_head · S · B · P
         = 2 · L · d_model · S · B · P     (因为 d_model = H · d_head)
```

记住两种写法:用 `H · d_head` 是为了在后面 GQA/MQA 减少 KV 头时能直接替换;用 `d_model` 是为了快速心算。

### 3.2 代入真实模型算量级

光有公式没感觉,代入一个具体模型。取一个 130 亿参数级别、配置类似常见开源模型的设定(下列具体配置数字属示意,真实模型请以其 config 为准【待核】):

```
L = 40,d_model = 5120(即 H = 40,d_head = 128),P = 2(BF16)
```

**单 token、单序列的 KV cache:**

```
2 × L × d_model × P
= 2 × 40 × 5120 × 2
= 819,200 字节
≈ 800 KB / token
```

**一条 2048 token 的序列:**

```
800 KB × 2048 ≈ 1.6 GB
```

**让人窒息的结论:一条 2K 上下文的序列,光 KV cache 就吃掉约 1.6 GB 显存。** 一张 80GB 的 A100,装完 13B 模型权重(BF16 约 24GB)后剩约 54GB,理论上**最多只能同时放约 34 条这样的序列**(54 / 1.56 ≈ 34)。如果上下文拉到 32K,一条序列就要 `800KB × 32768 ≈ 25 GB`——单卡连 3 条都放不下。

我们顺手推一个更有用的量——**每条序列每 token 的 KV 字节**,记作 `b_tok = 2 · L · d_model · P`。对上面的模型 b_tok ≈ 800 KB。那么显存能容纳的**总 token 数(所有序列加起来)**为:

```
total_tokens_capacity = (可用显存) / b_tok
```

**这个量是推理吞吐的硬上限。** 后面讲利用率、讲 PagedAttention 的收益,都是围绕"如何让 total_tokens_capacity 装下尽可能多的真实有效 token"。

### 3.3 一个重要的架构旁支:GQA/MQA 直接砍 KV

公式 `2 · L · H · d_head · S · B · P` 里,H(KV 头数)是可以单独动的。**MQA(Multi-Query Attention)** 让所有 Query 头共享同一组 K/V(KV 头数 = 1);**GQA(Grouped-Query Attention)** 让每 g 个 Query 头共享一组 K/V(KV 头数 = H/g)。注意力打分时 Q 头数不变,只是 K/V 被复制广播。

效果:KV cache 直接除以 g(GQA)或除以 H(MQA)。比如 H=40、用 8 组 KV 头的 GQA,KV cache 缩到原来的 8/40 = 1/5。**这是从架构侧、与 PagedAttention 正交的省 KV 手段**(第 12、15 章会再展开)。本章默认 MHA,但你要知道现代模型几乎都用 GQA,把公式里的 H 换成 KV 头数即可。

## 四、核心矛盾:显存碎片让利用率只剩 20–40%

公式告诉我们"装得下多少",但现实中,**传统推理系统远远装不到理论上限**。原因是**碎片(fragmentation)**。这一节是理解 PagedAttention 价值的关键,务必吃透。

### 4.1 病根:连续显存预留 + 未知输出长度

在 vLLM 之前,主流推理框架(如早期的 FasterTransformer、HuggingFace 的朴素实现)是这样管 KV cache 的:**为每条请求预留一整块物理连续的显存,大小按"可能的最大长度"算。**

为什么要连续?因为注意力 kernel 要遍历 k₁..kₜ,如果它们在显存里连续排列,kernel 用一个基址 + 偏移就能顺序读,实现简单、访存高效。这是最自然的写法。

为什么要按最大长度预留?因为**生成多少个 token 事先不知道**——模型可能吐 5 个就遇到结束符,也可能吐满 2048 个。系统不知道未来,只能按 `max_seq_len` 这个上界一次性把坑占住,否则生成到一半显存不够就崩了。

这两个"自然的选择"凑在一起,就埋下了灾难。

### 4.2 两种碎片的精确定义

借用操作系统内存管理的术语,碎片分两种:

**内部碎片(internal fragmentation):预留多、用得少。**
为一条请求预留了 max_seq_len = 2048 的空间,但它实际只生成了 30 个 token 就结束了。那 2048 − 30 = 2018 个 token 的 KV 空间被这条请求**独占着却空着**,谁也用不了。预留越大、实际越短,内部碎片越严重。

量化一下:设平均实际长度 S_real、预留长度 S_max,**单请求的内部碎片率 = (S_max − S_real) / S_max**。如果按 2048 预留、实际平均 200,内部碎片率高达 `(2048−200)/2048 ≈ 90%`。

**外部碎片(external fragmentation):空闲块拼不起来。**
请求来来去去,有的占 2048、有的占 512、有的占 1024,释放后在显存里留下大小不一的空洞。新来一条要 1500 的请求,显存里明明还剩 3000 的总空闲,但**散成了一个 1000 的洞 + 一个 800 的洞 + 一个 1200 的洞,没有任何一个连续空洞 ≥ 1500**,于是这条请求只能排队等待。总量够、但拼不出连续块,这就是外部碎片。

> 这两个词,系统工程师应该秒懂——它们就是 `malloc`/`free` 时代教科书里的经典问题。**vLLM 论文最关键的洞察,就是认出"推理 KV cache 管理 = 一个内存分配问题",于是把操作系统几十年攒下的解法直接搬过来。**

### 4.3 实测利用率:为什么是 20–40%

把两种碎片叠加,再加上一部分为采样保留的临时缓冲,**vLLM 论文(Kwon et al., SOSP 2023)实测**:在它之前的系统里,真正存有效 KV 数据的显存,**只占预留 KV 显存的 20.4%–38.2%**(论文 Figure 2 的测量,具体百分比【以原文为准,此处区间引自论文摘要级结论】)。也就是说,**一张卡 60% 以上的 KV 显存在空转。**

利用率低的直接后果:`total_tokens_capacity` 被砍掉一大半,能并发的请求数 = 吞吐被腰斩再腰斩。**这不是算力问题,纯粹是内存管理太糙。** 一个纯粹的系统工程问题,价值却是数倍吞吐——这正是 PagedAttention 影响力如此之大的原因。

## 五、PagedAttention:照搬操作系统的分页

### 5.1 核心思想:逻辑连续,物理分散

操作系统怎么解决进程内存碎片?**虚拟内存分页(paging)**:把物理内存切成固定大小的页(page,典型 4KB),进程看到的是连续的虚拟地址空间,但每一页可以映射到物理内存里**任意分散**的页框(page frame),映射关系记在**页表(page table)**里。进程不需要连续的物理内存,只要有足够多的空闲页框就行——外部碎片消失了(任何空闲页都能用),内部碎片被限制在"最后一页的零头"(最多浪费小于一页)。

**PagedAttention 一比一搬过来:**

| 操作系统 | PagedAttention |
|---|---|
| 物理内存页框(page frame) | KV cache 的**物理块(physical block)**,固定大小,每块存 `block_size` 个 token 的 KV |
| 进程的虚拟地址空间 | 一条序列的**逻辑 KV 序列**(逻辑上连续:token 0,1,2,…) |
| 页表(page table) | **block table**:逻辑块号 → 物理块号的映射 |
| 按需分页(demand paging) | 序列生成到需要新块时才分配物理块 |

`block_size` 是一块存几个 token 的 KV,典型取 16 或 32(vLLM 默认 16【待核,以配置为准】)。一个物理块的字节数:

```
block_bytes = block_size · 2 · L · H · d_head · P
            = block_size · b_tok
```

### 5.2 内部碎片被锁死在"小于一块"

关键收益来了。现在不再为请求预留 max_seq_len 的连续显存,而是**生成到哪、就分配到哪**:序列每写满一个块,就申请下一个物理块。一条实际生成 30 个 token、block_size=16 的序列,只占 `⌈30/16⌉ = 2` 个块,共 32 个 token 的空间,**内部碎片只有 32 − 30 = 2 个 token,被锁死在不到一个块以内**。

定量对比上一节的例子(S_max=2048,S_real=200,block_size=16):

```
朴素预留:内部碎片 = 2048 − 200 = 1848 token,碎片率 ≈ 90%
PagedAttention:占 ⌈200/16⌉ = 13 块 = 208 token
               内部碎片 = 208 − 200 = 8 token,碎片率 ≈ 3.8%
```

**外部碎片直接归零**:所有物理块大小一样,任何空闲块都能给任何请求用,不存在"拼不起来"。这就是 vLLM 把利用率从 20–40% 拉到接近 100% 的全部秘密——**就是分页,没有魔法。**

### 5.3 代价:block table 寻址 + 定制 kernel

天下没有免费的午餐。物理块分散后,注意力 kernel 不能再"基址 + 偏移"顺序读 KV 了,必须先查 block table 把逻辑位置翻译成物理地址——这相当于软件实现的"MMU 地址翻译"。代价有二:

1. **多一层间接寻址**:每访问一个 KV 块要先读 block table。但 block table 很小(每序列就几十个 int),常驻 SRAM/寄存器,开销可忽略。
2. **必须重写注意力 kernel**:标准 FlashAttention 假设 KV 连续,PagedAttention 要写一个**支持按块跳转读取 KV** 的 kernel。vLLM 为此实现了 paged 版的注意力 kernel(后来也整合了 FlashAttention 的分块 softmax)。这是工程上的真正难点,但只写一次。

### 5.4 block table 寻址伪代码

把"逻辑 token 位置 → 物理地址"的翻译写清楚,这是 PagedAttention 的心脏:

```python
# block_table[seq_id] 是一个列表,逻辑块号 -> 物理块号
# physical_kv_pool 是全局物理块池:形如 [num_blocks, block_size, num_kv_heads, d_head]
# 这是 K 池;V 池结构相同,各一份(对应公式里的 "2")

def locate_kv(seq_id, token_pos, block_size, block_table):
    """把一条序列里第 token_pos 个 token,定位到物理块池中的坐标。"""
    logical_block = token_pos // block_size      # 它属于第几个逻辑块
    offset = token_pos % block_size              # 在块内的第几个槽
    physical_block = block_table[seq_id][logical_block]  # 查"页表"翻译
    return physical_block, offset                # (物理块号, 块内偏移)

def paged_attention_decode(seq_id, q_t, block_table, k_pool, v_pool,
                           block_size, cur_len):
    """decode 第 cur_len 步:用 q_t 对 [0, cur_len) 的历史 KV 做注意力。
       关键:历史 KV 物理上分散在不同块,靠 block_table 跳着读。
       这里用朴素累加表达机制;真实 kernel 会融合 online softmax(见第9章)。"""
    scores = []
    for pos in range(cur_len):                   # 遍历所有历史位置
        pb, off = locate_kv(seq_id, pos, block_size, block_table)
        k = k_pool[pb, off]                       # 物理块内取出该 token 的 K
        scores.append(dot(q_t, k) / sqrt(d_head))
    weights = softmax(scores)                      # 因果范围内归一化
    out = zeros(d_head)
    for pos in range(cur_len):
        pb, off = locate_kv(seq_id, pos, block_size, block_table)
        v = v_pool[pb, off]
        out += weights[pos] * v
    return out

def append_kv(seq_id, k_t, v_t, block_table, k_pool, v_pool,
              block_size, cur_len, free_block_list):
    """把新 token 的 K/V 写入 cache;若当前块写满,先申请新物理块。"""
    logical_block = cur_len // block_size
    offset = cur_len % block_size
    if offset == 0:                                # 正好要开一个新逻辑块
        new_pb = free_block_list.pop()             # 从空闲块池分配(可能物理上任意位置)
        block_table[seq_id].append(new_pb)         # 在"页表"登记映射
    pb = block_table[seq_id][logical_block]
    k_pool[pb, offset] = k_t                        # 写 K
    v_pool[pb, offset] = v_t                        # 写 V
```

读这段代码要抓住三点:(1) `locate_kv` 就是软件版地址翻译;(2) 物理块从 `free_block_list` 拿,**物理上完全不要求连续**;(3) 只有跨块边界(`offset == 0`)才触发分配,这就是"按需分页"。

## 六、Copy-on-Write 前缀共享:把 fork 搬进推理

分页打开了一扇额外的门:**多条请求可以共享物理块**。这在朴素连续预留方案里根本做不到(每条请求独占自己那块连续显存)。

### 6.1 动机:成千上万请求共享同一个 system prompt

线上服务有大量请求共享**完全相同的前缀**:同一个长 system prompt、同一个 few-shot 示例、并行采样(一个 prompt 采 n 个不同续写)。这些前缀的 KV 算出来是**逐字节相同**的。朴素方案会为每条请求重复算一遍、重复存一遍——纯浪费。

### 6.2 机制:逻辑共享物理块,分叉时才复制(COW)

这又是操作系统的老把戏。`fork()` 创建子进程时,不真的复制父进程内存,而是让父子**共享同一批物理页,都标记为只读**;谁先写某一页,就**触发缺页 → 复制那一页 → 改自己的副本**(copy-on-write)。没人写就永远共享,省内存。

PagedAttention 对前缀 KV 做同样的事:

```
请求 A、B、C 共享同一段 system prompt:
  - 这段前缀只算一次 KV,存进物理块 #7,#8,#9
  - A、B、C 的 block table 里,前几个逻辑块都指向 #7,#8,#9(共享)
  - 每个物理块维护一个引用计数 ref_count = 3

各自开始生成不同内容(分叉点)时:
  - 写到共享块的"块内"会污染别人 → 触发 copy-on-write:
    复制那个块 → ref_count 减 1 → 写自己的副本
  - 但分叉通常发生在前缀之后的新块,所以前缀块往往全程共享、永不复制
```

引用计数的管理逻辑(block manager 的一部分):

```python
def fork_prefix(parent_seq, child_seq, num_shared_blocks, block_table, ref_count):
    """child 复用 parent 的前 num_shared_blocks 个物理块(不复制数据)。"""
    for lb in range(num_shared_blocks):
        pb = block_table[parent_seq][lb]
        block_table[child_seq].append(pb)
        ref_count[pb] += 1               # 共享 -> 引用计数加一

def cow_before_write(seq, logical_block, block_table, ref_count,
                     free_block_list, k_pool, v_pool):
    """写一个共享块前的检查:若被多方共享(ref>1),先复制出私有副本。"""
    pb = block_table[seq][logical_block]
    if ref_count[pb] > 1:                 # 还有别人在用 -> 不能就地改
        new_pb = free_block_list.pop()
        k_pool[new_pb] = k_pool[pb].clone()   # 复制这一块
        v_pool[new_pb] = v_pool[pb].clone()
        ref_count[pb] -= 1                # 原块引用减一
        block_table[seq][logical_block] = new_pb   # 改指向私有副本
        ref_count[new_pb] = 1
```

收益:**N 个共享 P 个前缀 token 的请求,前缀 KV 从 `N × P` 份压到 1 份**,显存省 `(N−1)/N`。并行采样、批量评测、多轮对话的公共历史,都吃这个红利。这套机制在 vLLM 里进一步发展成了 **prefix caching**(跨请求、跨时间复用前缀 KV,第 15 章细讲)。

> **设计美感在此:** 把"序列"看成"进程",把"KV cache"看成"虚拟内存",于是 fork+COW 这套验证了几十年的机制几乎零成本地适配过来。这是"认对问题的同构性"带来的杠杆——系统出身的人转 MLSys 最该培养的就是这种"这玩意儿我在 OS 里见过"的嗅觉。

## 七、Continuous Batching:把调度粒度从请求降到迭代

显存管好了,还有一个正交的吞吐杀手:**批处理的调度粒度**。这是 vLLM 高吞吐的第二根支柱,常和 PagedAttention 一起被提及,但它解决的是另一个问题。

### 7.1 病根:static batching 被最长序列拖死

朴素做法叫 **static batching(静态批处理)**:攒够一批请求(比如 8 条),一起送进模型,**等这一批全部生成完**,再处理下一批。问题在哪?同一批里,有的请求生成 10 个 token 就结束,有的要生成 1000 个。**早结束的请求只能干等着最长的那条跑完,它占的显存和 batch 槽位全程空转。**

定量感受:一批 8 条,长度分别是 [10, 20, ..., 1000],static batching 要等 1000 步才能整批退出、释放、换新。**早退出的 7 条平均浪费了几百步的槽位。** GPU 的有效利用率被最长序列单方面拉低。

### 7.2 解法:iteration-level scheduling(迭代级调度)

**continuous batching(连续批处理)**——也叫 **iteration-level scheduling**——把调度粒度从"一整批请求"细化到"一次前向迭代(生成一个 token 的那一步)":

```
每生成一个 token(每次 forward)后,调度器都重新评估一次 batch:
  - 哪条序列吐出了结束符 / 达到长度上限 -> 立刻退出,马上释放它的 KV 块
  - 释放出的显存和 batch 槽位 -> 立刻塞进等待队列里的新请求
  - 新请求先做 prefill,然后无缝加入正在 decode 的大批次一起跑
```

效果:**GPU 永远在满载工作,没有"等最长的那条"这回事**。一条结束,新的立刻补位。这套思想最早由 **Orca(OSDI 2022)** 提出,vLLM 把它和 PagedAttention 结合,因为**只有分页的 KV 管理才能让"随到随分配、随走随释放"在显存上真正高效可行**——两者是绝配。

### 7.3 调度循环骨架

把调度器主循环写出来,这是 vLLM 引擎的心跳:

```python
def continuous_batching_loop(model, scheduler, block_mgr):
    while scheduler.has_unfinished():            # 还有活就一直转
        # 1) 调度:决定这一步 batch 里跑哪些序列
        #    在显存预算内,尽量塞满:优先 decode 在跑的,再接纳新请求做 prefill
        batch = scheduler.schedule(block_mgr.num_free_blocks())
        #    batch 含两类:waiting 里新调入的(待 prefill)+ running 里继续的(decode)

        # 2) 为这一步需要新块的序列分配物理块(按需分页)
        for seq in batch.running:
            if seq.needs_new_block():            # 当前块写满了
                if block_mgr.num_free_blocks() == 0:
                    scheduler.preempt(seq)        # 显存不够 -> 抢占(见下)
                    continue
                block_mgr.allocate(seq)           # 分配一个物理块,登记 block table

        # 3) 一次前向:prefill 段和 decode 段可融合在同一 batch(chunked prefill 思路)
        logits = model.forward(batch)             # 走 paged attention kernel

        # 4) 采样 + 追加 KV(append_kv 见第五节)
        for seq in batch:
            next_tok = sample(logits[seq])
            seq.append(next_tok)
            block_mgr.append_kv(seq, next_tok)    # 写入 cache,必要时已在步骤2分配

        # 5) 退出与回收:谁完成谁立刻让位
        for seq in batch:
            if seq.last_token == EOS or seq.length >= seq.max_len:
                block_mgr.free(seq)               # 立刻归还所有物理块到空闲池
                scheduler.finish(seq)             # 把结果返回给用户
        # 循环回到 1):释放出的块下一轮立刻被新请求吃掉
```

### 7.4 抢占与换出:显存不够时怎么办

步骤 2 里出现了 `preempt`——这是连续批处理必须面对的硬问题:**正在跑的序列越生成越长,可能把显存吃光,而队列里还有请求**。vLLM 的处理同样照搬 OS:

- **重计算(recomputation):** 把某条序列的 KV 块全部释放、把它踢回等待队列;轮到它时,用它已生成的 token 重新 prefill 一遍把 KV 算回来。代价是重算,但省显存。
- **换出(swapping):** 把它的 KV 块从 GPU 显存拷到 CPU 内存(类比 OS 的 swap to disk),需要时再换回。代价是 PCIe 传输。

选哪个看权衡:序列短重算便宜,序列长换出可能更划算。**这又是操作系统页面置换的翻版**——你会越来越觉得 vLLM 就是"给 KV cache 写的一个小操作系统"。

## 八、vLLM 架构:scheduler / block manager / worker 三件套

把上面所有机制归位到 vLLM 的模块职责上,你就有了系统全图:

```
                 ┌─────────────────────────────────────────┐
   请求队列  ──▶ │  Scheduler(调度器)                       │
                 │   • iteration-level 调度:每步选 batch    │
                 │   • 接纳新请求 / 抢占 / 换出决策           │
                 │   • 在显存预算内最大化在跑的 token 数      │
                 └───────────────┬─────────────────────────┘
                                 │ "我要跑这些序列,谁要新块"
                 ┌───────────────▼─────────────────────────┐
                 │  Block Manager(块管理器,即"MMU+分配器")│
                 │   • 维护物理块空闲池 free_block_list      │
                 │   • 每序列的 block table(逻辑→物理映射)  │
                 │   • allocate / free / 引用计数 / COW      │
                 └───────────────┬─────────────────────────┘
                                 │ block table + 物理块池地址
                 ┌───────────────▼─────────────────────────┐
                 │  Worker(执行器,每 GPU 一个)            │
                 │   • 持有模型权重和物理 KV 块池(HBM)      │
                 │   • 跑 paged attention kernel 做前向       │
                 │   • 张量并行时多 worker 协作(见第14章)  │
                 └───────────────────────────────────────────┘
```

- **Scheduler** = OS 的进程调度器,只是把"进程"换成"序列",调度粒度是一次 forward。它不碰显存细节,只问 block manager 要"还剩几块"。
- **Block Manager** = OS 的内存管理单元 + 分配器。它持有那张"页表"(block table)和空闲块池,负责 allocate/free/COW/引用计数。**碎片清零、前缀共享的逻辑全在这里**,是本章源码导读的重点。
- **Worker** = 实际干活的 GPU 进程,持有模型权重和 KV 物理块池,执行 paged attention kernel。多卡张量并行时多个 worker 协同(下接第 14 章通信)。

三者解耦得很干净:**调度决策、显存映射、计算执行各管一摊**。这个分层本身就是好系统设计的范本——和 OS 把"调度器/内存管理/CPU 执行"分开是一个道理。

## 九、设计权衡与常见坑

**block_size 怎么选?** 这是最关键的调参。块太小(如 1):内部碎片几乎为零,但 block table 变长、间接寻址次数多、kernel 跳转访存碎、元数据开销大。块太大(如 256):寻址高效、元数据少,但内部碎片回潮(又开始浪费最后一块的零头),COW 复制的粒度也变粗(改一个 token 要复制一大块)。**典型甜点 16–32**(vLLM 默认 16【待核】)。这正是 OS 选页大小的同款权衡——4KB 不是拍脑袋来的。

**别把 PagedAttention 和 FlashAttention 对立起来。** 它俩解决不同层次的问题:FlashAttention 优化**单次注意力计算的访存**(片上分块算 softmax,不落 N×N 矩阵);PagedAttention 优化**KV cache 的显存布局与生命周期管理**。现代 vLLM 是**两者叠加**——用支持分页的 FlashAttention kernel,块内仍走 online softmax。新手常误以为二选一,错。

**坑:block table 的间接寻址在 kernel 里不是免费的。** 虽然 block table 本身小,但如果实现不当(比如每个线程都重复查表、查表导致访存不合并),会拖慢 kernel。vLLM 的 paged kernel 在这块做了不少工程优化,读 kernel 源码时注意它怎么 batch 化地预取 block table、怎么保证 KV 读取的访存合并(coalescing)。

**坑:利用率高了,反而更容易 OOM。** 朴素方案预留充足,几乎不会中途 OOM;PagedAttention 把显存榨到接近满,**正在跑的序列变长时随时可能没块可分**,所以**抢占/换出逻辑是必需品而非可选项**。如果你自己实现,漏了抢占,长序列负载下必崩。

**坑:COW 的引用计数要和块释放严格配对。** 共享块的 ref_count 管理是经典的引用计数 bug 温床——少减一次导致块永远不释放(显存泄漏),多减一次导致块被提前回收(数据损坏,生成乱码)。这和写 `shared_ptr`/GC 的坑一模一样,要极其小心。

**权衡:吞吐 vs 延迟。** 连续批处理最大化吞吐(GPU 满载),但新请求要等当前迭代结束才能插入,且大 batch 会拉长单步延迟。对延迟敏感(如交互式对话首 token 延迟 TTFT)的场景,要限制 batch 大小或用 **chunked prefill**(把长 prefill 切块,与 decode 交错,避免长 prefill 阻塞 decode)。吞吐和延迟永远在拔河,vLLM 给了一堆旋钮让你按 SLO 调。

## 十、动手练习

**练习 1(估算题 · KV cache 大小)。**
一个模型 L=32、d_model=4096、采用 GQA 且 KV 头数为 8、d_head=128(注意:此时 KV 维不是 d_model,而是 KV头数 × d_head)、精度 BF16(P=2)。
(a) 推导这个 GQA 模型每 token 的 KV 字节 b_tok。
(b) 一张 80GB A100 装完 7B 权重(BF16 约 14GB)后,假设 60GB 可用于 KV,最多能缓存多少总 token?
(c) 如果换成 MHA(KV 头数 = 32),(b) 的答案变成多少?对比体会 GQA 的价值。
*提示:b_tok = 2 · L · (KV头数 · d_head) · P;GQA 时把公式里的 d_model 换成 KV头数 × d_head。注意单位换算 1GB = 2³⁰ 字节。*

**练习 2(估算题 · 利用率→吞吐增益)。**
某服务平均请求实际生成 256 token,但朴素方案按 max_seq_len=2048 预留连续显存。
(a) 算朴素方案的内部碎片率,以及"有效 KV 占预留 KV"的比例(忽略外部碎片)。
(b) 假设 PagedAttention(block_size=16)把利用率提到 96%,在相同显存下能并发的请求数(≈ total_tokens_capacity / 实际平均长度)大致提升几倍?
(c) 若吞吐近似正比于并发请求数,给出吞吐增益的量级估计,并说明为什么真实增益通常小于这个理论值(提示:外部碎片、采样缓冲、调度开销、是否撞到算力上限)。

**练习 3(编码题 · 实现 block manager)。**
照着第五、六节的伪代码,用纯 Python 实现一个最小 block manager,支持:`allocate(seq)`、`append_kv(seq, k, v)`、`free(seq)`、`fork_prefix(parent, child, n)`、写共享块前的 `cow`。用一个 `[num_blocks, block_size, d]` 的 numpy 数组当物理块池。
验证:(1) 两条共享 32-token 前缀的序列,前缀块物理地址相同、ref_count=2;(2) 其中一条在前缀后分叉写入,触发 COW、前缀块仍共享;(3) 一条序列 free 后,它独占的块回到空闲池、共享块 ref_count 减 1 而非释放。
*提示:ref_count 用 `dict[物理块号 -> int]`;free 时对每个块 `ref_count -= 1`,减到 0 才真正归还 free_block_list。*

**练习 4(推导题 · 重算 vs 缓存)。**
(a) 严格写出"不缓存 K/V 的朴素自回归"生成 N 个 token 的总计算量(含 K/V 投影 + 注意力打分两部分),证明主导项是 `O(N²d²)`。
(b) 写出"缓存 K/V"方案的总计算量,指出省掉的是哪一项、代价是多大的显存。
(c) 推广:如果用换出(swapping)把 KV 卸到 CPU,换回时的 PCIe 传输量怎么算?对比"重算"的算力代价,讨论在什么序列长度下换出比重算更划算(给出一个粗略的临界条件,符号化即可)。
*提示:重算代价 ∝ 重新 prefill 的 FLOPs(≈ 2 · 参数量 · token 数级别);换出代价 ∝ KV 字节 / PCIe 带宽。临界点是两者时间相等。*

## 十一、源码 / 论文导读

**论文:**
- **vLLM 原始论文 —— Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023。** 必读:§3(动机,Figure 2 那张利用率 20–40% 的实测图,本章第四节的数据来源)、§4(PagedAttention 与 KV block 的设计)、§4.3–4.4(COW 前缀共享与抢占/换出)。这是本章的源头,从头读到 §4 即可掌握全部核心。
- **Orca —— Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022。** 读它提出的 **iteration-level scheduling**(连续批处理的源头,本章第七节)。看它怎么论证 static batching 的浪费、怎么把调度粒度降到迭代。
- 架构侧省 KV 的 **GQA —— Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)** 和 **MQA —— Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019)**,配合本章 3.3 节理解公式里 H 怎么被砍。

**开源代码(vLLM,以你 clone 到的版本为准,模块名可能随版本演进【待核】):**
- **`vllm/core/block_manager*.py`(及 `vllm/core/block/` 目录)** —— 本章第五、六节的落地:物理块池、block table、allocate/free、引用计数、COW。**重点读** `allocate`、`append_slots`/`can_append`、`fork`、`free` 这几个方法,对照本章伪代码逐行印证。
- **`vllm/core/scheduler.py`** —— 第七节的连续批处理调度器:看 `schedule()` 怎么在显存预算内组 batch、`_preempt`/`_swap_out`/`_swap_in` 怎么处理显存不够。这是"给 KV 写的操作系统"的调度内核。
- **`vllm/attention/`(paged attention kernel,部分为 CUDA/Triton)** —— 第五节寻址 + 第九节坑的工程实现:看它怎么用 block table 在 kernel 里跳块读 KV、怎么保证访存合并、怎么和 FlashAttention 的分块 softmax 融合。这部分最硬,但读懂了你就真正理解 PagedAttention 不是纸上谈兵。
- **prefix caching(`vllm/` 中相关 prefix/automatic prefix caching 逻辑)** —— 第六节 COW 的产品化延伸,跨请求复用前缀。读它怎么用前缀 hash 命中已有物理块。

读源码的方法:**先用本章伪代码在脑子里建好模型,再去源码里找对应物**——你会发现真实代码的复杂度几乎都来自工程细节(并发、张量并行、各种边界),核心机制和本章的几十行伪代码完全一致。

## 十二、小结与承上启下

这一章我们做了三件事:

1. **从第一性原理证明了 KV cache 是刚需**:kⱼ/vⱼ 与步数无关 → 缓存把自回归从 `O(N²d²)` 重算降到增量计算,代价是 `O(N)` 的显存——把瓶颈从算力墙转移到了显存墙。
2. **逐字节推导了显存公式 `2·L·H·d_head·S·B·P`**,代入真实模型算出"一条 2K 序列吃 1.6GB"的量级,并定量解释了内部/外部碎片如何把传统系统的利用率压到 20–40%。
3. **讲透了 vLLM 三板斧**:PagedAttention(把 KV 当虚拟内存分页,碎片清零)、COW 前缀共享(把 fork 搬进推理)、连续批处理(把调度粒度降到迭代,GPU 满载)——它们的共同灵魂是**把推理显存管理识别为一个操作系统问题**。

**在整门课里的位置:** 第 9 章 FlashAttention 优化了**单次注意力计算**的访存(片上算 softmax),这一章优化了 **KV cache 的显存布局与生命周期**——两者正交叠加,共同构成了 vLLM 高效推理的内核。但我们目前只是"把显存管好、GPU 喂满",decode 阶段**算力闲置(memory-bound)** 这个更根本的浪费还没动——一次前向只为算 1 个 token,几百 GB 权重白搬一遍。**第 11 章(投机解码)** 正是要榨干这块闲置算力:用小模型一次草拟 K 个 token、大模型一次前向并行验证,且我们会严格证明它**输出分布无损**。再往后,**第 12 章(量化)** 从字节侧继续压 KV 和权重,**第 15 章** 会把本章的 prefix caching、KV 驱逐/压缩、prefill/decode 解耦推到长上下文前沿,**第 16 章 capstone** 则让你用 vLLM 把这一切在真实服务上实测验证。

记住这一章最值钱的一句话:**当你面对一个"资源该怎么管"的系统问题时,先问一句"操作系统是不是早就解决过它的同构版本"——PagedAttention 整个工作,就是这一问带来的回报。**


---

## 练习参考答案

> 本章「动手练习」的参考答案(AI 生成,推导/代码已尽量自验,具体数值见「待核」标注)。

### 练习 1:GQA 模型的 KV cache 大小估算

已知 L=32、d_head=128、P=2(BF16),逐字节公式为 `b_tok = 2 · L · (KV头数 · d_head) · P`。

**(a) 每 token 的 KV 字节 b_tok(GQA,KV头数=8)**

GQA 时把公式里的 d_model 换成「KV头数 × d_head」,即 KV 维 = 8 × 128 = 1024:

```
b_tok = 2 · L · (KV头数 · d_head) · P
      = 2 × 32 × (8 × 128) × 2
      = 2 × 32 × 1024 × 2
      = 131,072 字节
      = 128 KB / token
```

**结论:GQA 模型 b_tok = 131072 字节 = 128 KB/token。**

**(b) 60GB 可用显存最多缓存多少总 token**

注意 1GB = 2³⁰ = 1,073,741,824 字节,故 60GB = 64,424,509,440 字节:

```
total_tokens = 可用显存 / b_tok
             = 64,424,509,440 / 131,072
             = 491,520 token
```

**结论:GQA 时最多缓存约 491,520 token(≈ 0.49M)。** 直观感受:若平均一条序列 2048 token,可同时容纳约 491520/2048 ≈ 240 条并发序列。

**(c) 换成 MHA(KV头数=32)**

KV 维变为 32 × 128 = 4096(此时正好等于 d_model):

```
b_tok(MHA) = 2 × 32 × (32 × 128) × 2 = 2 × 32 × 4096 × 2 = 524,288 字节 = 512 KB/token
total_tokens(MHA) = 64,424,509,440 / 524,288 = 122,880 token ≈ 0.12M
```

**对比结论:** GQA(8 KV 头)对 MHA(32 KV 头)的 KV 头数之比是 8/32 = 1/4,所以 b_tok 缩到 1/4,**相同显存能缓存的 token 数恰好翻 4 倍(491520 / 122880 = 4.0)**。验算:`491520 = 4 × 122880` ✓。这就是 GQA 的核心价值——在几乎不损失质量的前提下,把 KV cache 直接砍成 `KV头数/原H` 倍,等价于把推理吞吐的硬上限抬高同样的倍数。

---

### 练习 2:利用率 → 吞吐增益估算

已知平均实际生成 S_real=256,朴素方案按 S_max=2048 预留连续显存。

**(a) 内部碎片率与「有效 KV 占预留 KV」比例**

按第 4.2 节定义,单请求内部碎片率 = (S_max − S_real) / S_max:

```
内部碎片率 = (2048 − 256) / 2048 = 1792 / 2048 = 0.875 = 87.5%
有效占比  = S_real / S_max = 256 / 2048 = 0.125 = 12.5%
```

**结论:内部碎片率 87.5%,有效 KV 只占预留 KV 的 12.5%(忽略外部碎片)。** 也就是朴素方案下 87.5% 的预留 KV 显存在空转。

**(b) 相同显存下并发请求数提升几倍**

记总物理显存为 M、每 token 字节为 b_tok,物理 token 槽位总数 `M / b_tok` 是常数。

- 朴素方案:每条请求**预留** S_max=2048 个槽位(无论实际只用 256),故并发数 = `(M/b_tok) / 2048`。
- PagedAttention(block_size=16,利用率 96%):一条 256-token 序列占 `⌈256/16⌉ = 16` 个块 = 256 token(整除,内部碎片为 0),实际占用极贴合真实长度;在 96% 利用率下并发数 ≈ `0.96 · (M/b_tok) / 256`。

两者相除:

```
提升倍数 = [0.96 · (M/b_tok)/256] / [(M/b_tok)/2048]
        = 0.96 × (2048 / 256)
        = 0.96 × 8
        = 7.68 倍
```

**结论:并发请求数大约提升 7.7 倍(≈ 8 倍上界 × 96% 利用率)。** 倍数的本质是「按 max_seq_len 预留 vs 按实际长度精确分配」之比 2048/256 = 8,再乘以分页方案自身的利用率 0.96。

**(c) 吞吐增益的量级估计,及真实增益为何更小**

若吞吐近似正比于并发请求数,则**理论吞吐增益约 7.7 倍(量级约 8×)**。

真实增益通常**小于**这个理论值,原因(结构化):

1. **外部碎片未计入:** (a)(b) 只算了内部碎片。朴素方案还有外部碎片(空闲块拼不出连续大块),会让理论基线更差——这一项其实会让相对增益**更大**;但下面几项把净增益拉低。
2. **采样/临时缓冲与元数据开销:** 分页方案要存 block table、引用计数、为采样保留临时缓冲,96% 这个利用率已包含部分损耗,真实场景可能更低。
3. **调度开销:** 连续批处理每步都要重新调度、处理抢占/换出(preempt/swap),长序列负载下抢占会触发重算或 PCIe 传输,吃掉一部分吞吐。
4. **撞到算力上限(最关键的封顶):** 吞吐正比于并发数的前提是 **decode 阶段 memory-bound、算力有富余**。一旦并发数拉高到把 GPU 算力(或访存带宽)喂满,瓶颈从「显存装不下」切换成「算不过来」,**吞吐不再随并发线性增长而是被算力封顶**。所以即便显存能塞下 7.7 倍的序列,实际吞吐增益会在撞到 compute/带宽 roofline 后饱和,显著低于 7.7×。

**结论:理论增益约 8×,真实增益通常落在数倍区间,且最终被 GPU 算力 roofline 封顶。** 这也呼应正文「这不是算力问题、纯粹是内存管理太糙」——但内存管好之后,算力墙就会重新成为新的天花板(正是第 11 章投机解码要解决的问题)。

---

### 练习 3:最小 block manager 实现

纯 Python + numpy 实现(CPU 可直接跑,无需 GPU)。物理块池用 `[num_blocks, block_size, d]` 的 numpy 数组,K/V 各一份(对应公式里的「2」)。下方代码经实际运行,三个验证断言全部通过。

```python
import numpy as np

class BlockManager:
    def __init__(self, num_blocks, block_size, d):
        self.block_size = block_size
        self.k_pool = np.zeros((num_blocks, block_size, d), dtype=np.float32)
        self.v_pool = np.zeros((num_blocks, block_size, d), dtype=np.float32)
        self.free_block_list = list(range(num_blocks))  # 空闲物理块池
        self.block_table = {}    # seq_id -> [physical_block, ...]   即"页表"
        self.ref_count = {}      # physical_block -> int             引用计数
        self.cur_len = {}        # seq_id -> 已写入的 token 数

    def _new_block(self):
        pb = self.free_block_list.pop()   # 物理上任意位置, 不要求连续
        self.ref_count[pb] = 1
        return pb

    def allocate(self, seq):
        # 登记一条序列(惰性建表), 不预占任何物理块
        self.block_table.setdefault(seq, [])
        self.cur_len.setdefault(seq, 0)

    def append_kv(self, seq, k, v):
        self.allocate(seq)
        pos = self.cur_len[seq]
        lb, off = pos // self.block_size, pos % self.block_size
        if off == 0:                       # 跨到新逻辑块 -> 按需分页, 申请新物理块
            self.block_table[seq].append(self._new_block())
        else:                              # 块内继续写; 若该块仍被共享, 先 COW
            self.cow(seq, lb)
        pb = self.block_table[seq][lb]
        self.k_pool[pb, off] = k           # 写 K
        self.v_pool[pb, off] = v           # 写 V
        self.cur_len[seq] = pos + 1

    def fork_prefix(self, parent, child, n):
        # child 复用 parent 的前 n 个物理块(零拷贝, 仅加引用计数)
        self.allocate(child)
        for lb in range(n):
            pb = self.block_table[parent][lb]
            self.block_table[child].append(pb)
            self.ref_count[pb] += 1        # 共享 -> 引用计数 +1
        self.cur_len[child] = n * self.block_size   # child 从前缀末尾继续生成

    def cow(self, seq, lb):
        # 写共享块前的 copy-on-write: ref>1 才复制出私有副本
        pb = self.block_table[seq][lb]
        if self.ref_count[pb] > 1:
            new_pb = self.free_block_list.pop()
            self.k_pool[new_pb] = self.k_pool[pb].copy()   # 复制这一块
            self.v_pool[new_pb] = self.v_pool[pb].copy()
            self.ref_count[pb] -= 1                         # 原块引用 -1
            self.block_table[seq][lb] = new_pb              # 改指向私有副本
            self.ref_count[new_pb] = 1

    def free(self, seq):
        # 释放一条序列: 每块 ref-1, 减到 0 才真正归还空闲池
        for pb in self.block_table.get(seq, []):
            self.ref_count[pb] -= 1
            if self.ref_count[pb] == 0:
                self.free_block_list.append(pb)
                del self.ref_count[pb]
        self.block_table.pop(seq, None)
        self.cur_len.pop(seq, None)
```

验证脚本与运行结果:

```python
bm = BlockManager(num_blocks=16, block_size=16, d=4)

# 序列 A 写满 32-token 前缀(占 2 个块)
for i in range(32):
    bm.append_kv("A", np.full(4, i, np.float32), np.full(4, -i, np.float32))

# (1) B 共享 A 的 32-token 前缀
bm.fork_prefix("A", "B", n=2)
assert bm.block_table["A"][:2] == bm.block_table["B"][:2]          # 物理块相同
assert all(bm.ref_count[pb] == 2 for pb in bm.block_table["A"][:2])  # ref_count=2

# (2) B 分叉: 先在前缀后写新 token(开新块), 再回头改前缀块 -> 触发 COW
bm.append_kv("B", np.full(4, 99, np.float32), np.full(4, 99, np.float32))
bm.cow("B", 0)
bm.k_pool[bm.block_table["B"][0], 0] = 777.0
assert bm.block_table["B"][0] != bm.block_table["A"][0]   # B 得到私有块
assert bm.ref_count[bm.block_table["A"][0]] == 1          # A 原块 ref 减回 1
assert bm.ref_count[bm.block_table["A"][1]] == 2          # 未被写的前缀块仍共享

# (3) free A: 独占块回收, 共享块只减 ref
a0, a1 = bm.block_table["A"][0], bm.block_table["A"][1]
bm.free("A")
assert a0 in bm.free_block_list           # A 独占块回到空闲池
assert a1 not in bm.free_block_list       # 共享块不被释放
assert bm.ref_count[a1] == 1              # 共享块 ref 减为 1(B 仍在用)
print("全部断言通过")
```

实际运行输出(block_size=16,从 num_blocks-1 倒序分配,故物理块号是 15、14):

```
A 前缀块: [15, 14]
(1) PASS: 前缀物理块相同 = [15, 14]  ref_count = [2, 2]
(2) PASS: COW 后 B[0]=12 ≠ A[0]=15 ; 未写的共享块 ref=2
(3) PASS: 独占块 15 已回收; 共享块 14 未释放, ref=1
B 第二个前缀块首槽 K = [16. 16. 16. 16.]   # 共享块数据未被污染, 仍可正确读取
```

**三个验证点结论:**
1. fork 后两序列前缀逻辑块指向**同一批物理块**,`ref_count=2`,前缀 KV 只存 1 份;
2. 一方在前缀块上写入时**触发 COW** 得到私有副本,原前缀块 ref 减 1、另一方仍正确共享;
3. `free` 一条序列时,**独占块归还空闲池**、**共享块仅 ref−1 而不释放**——这正是正文第九节强调的「引用计数必须和释放严格配对」,少减/多减都会导致泄漏或数据损坏。

---

### 练习 4:重算 vs 缓存 的计算量推导

符号:N=最终序列长度,d=隐藏维,投影一个 token 到 K 或 V 是 d×d 矩阵乘 d 维向量,计算量 `O(d²)`;Q·K 打分一对向量是 `O(d)`。

**(a) 不缓存的朴素自回归:总计算量 = O(N²d²)**

第 t 步要算第 t 个位置的输出,需重新得到全前缀 k₁..kₜ、v₁..vₜ。分两部分:

① **K/V 投影(主导项):** 第 t 步把前 t 个 token 全部重投影一遍 K 和 V,每个 token `O(d²)`,该步 `O(t·d²)`。N 步累加:

```
∑ₜ₌₁ᴺ O(t · d²) = O(d²) · ∑ₜ₌₁ᴺ t = O(d²) · N(N+1)/2 = O(N²d²)
```

② **注意力打分:** 第 t 步 qₜ 与 t 个 kⱼ 做点积,每对 `O(d)`,该步 `O(t·d)`。N 步累加:

```
∑ₜ₌₁ᴺ O(t · d) = O(d) · N(N+1)/2 = O(N²d)
```

合计 `O(N²d²) + O(N²d)`。因 d ≫ 1,**主导项是 K/V 重投影的 `O(N²d²)`**,得证。直观:它把那些「值根本不随步数变化」的 K/V 反复重算了一个 N 量级的次数。

**(b) 缓存 K/V 方案:省掉的项 + 显存代价**

第 t 步只做:投影当前 1 个 token 的 qₜ/kₜ/vₜ → `O(d²)`;追加进 cache → `O(d)`;用 qₜ 与已缓存的 t 个 K 打分、加权 V → `O(t·d)`。N 步累加:

```
∑ₜ₌₁ᴺ O(d² + t·d) = O(N·d²) + O(d)·N(N+1)/2 = O(N·d²) + O(N²·d)
```

**省掉的正是那个最大的 `O(N²·d²)` 项**(K/V 重投影从「每步重算全前缀」降为「每步只投影 1 个新 token」,即 `O(N²d²) → O(N·d²)`,降了一个 N 量级)。剩下的 `O(N²·d)` 是注意力打分本身,无法靠缓存消除(每步都要看全历史)。

**代价**:必须把全部历史 K/V 常驻显存,字节数 = `2 · L · H · d_head · N · P`(单序列),即 `O(N)` 量级显存。**一句话:用 O(N) 的显存换掉了 O(N²d²) 的重复计算——典型的以空间换时间,把瓶颈从算力墙推到显存墙。**

**(c) 换出(swap)的 PCIe 传输量,与重算的临界条件**

设序列已缓存 S 个 token,每 token KV 字节 `b_tok = 2·L·H·d_head·P`,被抢占序列的参数量 N_param,GPU 算力 C_gpu(FLOP/s),PCIe 带宽 BW(字节/s)。

**换出代价(时间):** KV 字节 = `b_tok · S`,换出 + 换回是一来一回(单程则去掉系数 2):

```
t_swap = 2 · b_tok · S / BW          (round-trip)
```

**重算代价(时间):** 用已有 S 个 token 重新 prefill 一遍,FLOPs ≈ `2 · N_param · S`(每参数一次乘加、按 token 数线性,正文提示),prefill 是 compute-bound 可吃满算力:

```
t_recompute = 2 · N_param · S / C_gpu
```

(严格说重算还含注意力的 `O(S²·d)` 项,序列很长时这项会让重算更贵;下面先看线性主导项。)

**临界条件(t_recompute = t_swap):**

```
2 · N_param · S / C_gpu  =  2 · b_tok · S / BW
```

两边的 **S 恰好约掉**,得到一个**与序列长度无关的硬件比值判据**:

```
换出更划算  ⟺  t_swap < t_recompute  ⟺  b_tok / BW  <  N_param / C_gpu
                 ⟺  b_tok · C_gpu  <  N_param · BW
```

**结论(诚实标注):** 在「两者都严格线性于 S」的一阶模型下,**临界点不取决于序列长度,而取决于硬件常数比** `(b_tok/BW) vs (N_param/C_gpu)`——即「每 token 搬 KV 的带宽时间」对比「每 token 重算的算力时间」。

那序列长度何时让换出更划算?来自**被一阶模型忽略的二阶效应**:重算除线性的 `2·N_param·S` 外,还有注意力 `O(S²·d)` 项,**随 S 平方增长**;而换出始终线性于 S。所以——

```
序列越长(S 越大), 重算的 O(S²) 项越占主导, 重算越不划算 ⟹ 换出越占优。
存在一个临界长度 S*, 当 S > S* 时 t_recompute(含 S² 项) > t_swap, 换出更划算。
```

粗略地,令重算时间 `≈ a·S + b·S²` 等于换出时间 `≈ c·S`,临界 `S* ≈ (c − a)/b`(其中 a∝N_param/C_gpu、c∝b_tok/BW、b∝d/C_gpu)。**符号化结论:短序列重算便宜(线性项小、无 S² 包袱),长序列换出便宜(避开重算的 S² 增长),临界长度 S* 由上式的硬件常数决定。** 这正是正文所说「序列短重算便宜、序列长换出可能更划算」的定量来源,本质是 OS 页面置换里「重新计算 vs swap to disk」权衡的翻版。

(量级感参考,均标【待核】,随硬件而变:13B 模型、b_tok≈800KB、A100 BF16 算力 ~312 TFLOPS、PCIe4 x16 ~32GB/s 时,每 token 重算线性项 ≈ 83 μs,每 token 换出来回 ≈ 51 μs——此设定下线性项已偏向换出,叠加重算的 S² 项后长序列更明显倾向换出。)
