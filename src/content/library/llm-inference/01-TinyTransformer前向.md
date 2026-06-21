---
title: "Tiny Transformer 前向：从 token 到 logits，手算每一步"
slug: "01"
collection: "llm-inference"
order: 1
summary: "全书第一章，建立后续所有优化的'正确性基准'：用纯 TypeScript 手写一个 4 层 GQA decoder 的完整前向，embedding 查表到 vocab logits 全程不调矩阵库。逐算子追踪激活尺度证明 RMSNorm 的不变量、实测注意力的 O(seq²) 增长曲线（这是第 2 章 KV 缓存要消灭的成本）、并跑出 softmax 不减 max 时的 NaN 失败。后续每一章（KV 缓存、采样、连续批处理、分页、投机、量化）都要跟本章的 golden logits 对拍，才算正确。"
topics:
  - "LLM 推理"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

要给一个推理引擎做加速，你得先有一个慢得诚实、对得无可争议的东西作为对照。这一章就是造那个东西：一个 4 层、64 维、用 GQA（分组查询注意力，多个查询头共享一组 KV 头）的 decoder，从 token id 算到 256 维 vocab logits，全程手写。不调 BLAS（高性能矩阵库），不调任何 tensor 框架，matmul / softmax / RoPE / RMSNorm 全是 for 循环。

为什么要这么自虐？因为后面六章每一个加速——KV 缓存、连续批处理、分页 KV、投机解码、量化——本质都是「换一种更快的方式算出同一个 logits」。「同一个」三个字需要一个 ground truth（真值基准）来界定。本章末尾产出的 golden logits 就是它：后续任何优化跑完，最后一个 token 的 logits 必须跟本章对得上（float64 噪声内），对不上就是 bug，不管它跑得多快。

先说清楚一件容易误会的事。本章的模型权重是用固定种子的 PRNG 随机生成的合成权重（`buildModel(cfg, SEED)`），模型**没训练过**，所以输出的 logits 是「结构正确的乱码」——argmax 出来的 token 没有任何语言意义。这没关系。本章讲的是前向**机制**，不是输出质量。机制对不对，看的是激活尺度的不变量、计算量的增长形状、数值稳定性这些可验证的东西，跟权重训没训练无关。

配套代码在 `examples/llm-inference-from-scratch/src/core/`（被复用的算子）和 `examples/llm-inference-from-scratch/src/stage01-forward.ts`（本章的可跑实验）。基线配置写死在 `core/model.ts`：

```ts
export const DEFAULT_CONFIG: ModelConfig = {
  dModel: 64,
  nLayers: 4,
  nHeads: 4,
  dHead: 16,
  dFF: 256,
  vocabSize: 256, // byte-level, matches core/tokenizer
  maxSeq: 256,
  nKVHeads: 2,
};
```

注意 `nKVHeads: 2` 小于 `nHeads: 4`——默认就开了 GQA，4 个查询头分 2 组，每组共享一个 KV 头。这不是凑数，是为了让第 2 章的 KV 缓存差异（GQA 把 KV 缓存砍一半）从第一章起就在数据里可见。

## 一次前向到底在算什么

整条管线是一个固定的序列。先把 token id 当下标去 embedding 表里查出一行向量（embedding 就是「token → 向量」的查表），然后这行向量过 `nLayers` 个完全相同结构的 block，每个 block 内部是：

```
RMSNorm → 多头自注意力(带因果掩码) → 残差加 → RMSNorm → SwiGLU FFN → 残差加
```

最后过一次末层 RMSNorm，再用 lm_head（输出投影，本书用 weight tying 复用 embedding 矩阵）投到 vocab 维度，得到每个候选 token 的 logit。

`core/model.ts` 里的 `blockStep` 就是一个 block 的全部，结构和上面这行一字不差：

```ts
function blockStep(h, w, pos, kHist, vHist, histLen, cfg) {
  // pre-norm
  const normed = rmsNorm(tensor(h, [1, cfg.dModel]), w.normAtt).data;
  const { q, k, v } = projectQKV(normed, w, cfg);
  applyRope(q, k, pos, cfg);
  // ... 拼接 KV 历史，attendToken 算注意力 ...
  const attnOut = matmul(tensor(attn, [1, cfg.nHeads * cfg.dHead]), w.wO).data;
  // residual
  const h1 = new Float64Array(cfg.dModel);
  for (let i = 0; i < cfg.dModel; i++) h1[i] = h[i] + attnOut[i];
  // FFN sub-block, pre-norm + residual
  const normed2 = rmsNorm(tensor(h1, [1, cfg.dModel]), w.normFFN).data;
  const ff = ffn(normed2, w, cfg);
  const h2 = new Float64Array(cfg.dModel);
  for (let i = 0; i < cfg.dModel; i++) h2[i] = h1[i] + ff[i];
  return { h: h2, k, v };
}
```

这是「pre-norm」结构（先 norm 再进子层），现代 decoder 几乎都这么做：norm 放在残差分支里面，主干（residual stream）始终是一条没被 norm 动过的「干净加法线」。下面追踪激活尺度时你会看到，主干的幅度随层累加越长越大，而每次喂给子层之前都被 RMSNorm 拉回单位尺度——这个拉回是深层网络不爆炸的核心机制。

### 每个 Float64Array 都能指着说出物理含义

本章的第一个实验 `[a]` 就是把上面这个 block 拆成单算子，逐个打印激活的统计量（mean / std / rms / |max|）。目的很朴素：让你能指着每个 `Float64Array` 说出它是什么、尺度多大。下面是真实运行输出（layer 0，一个 235-token prompt 的最后一个 token，pos=234）：

```
config: dModel=64 nHeads=4 nKVHeads=2 dHead=16 dFF=256
embed (block input)    mean=   0.0080  std=   0.0778  rms=   0.0782  |max|=   0.1232
RMSNorm(att)           mean=   0.1018  std=   0.9948  rms=   1.0000  |max|=   1.5678
Q proj                 mean=   0.0568  std=   0.6143  rms=   0.6170  |max|=   1.5224
K proj                 mean=  -0.1001  std=   0.4745  rms=   0.4849  |max|=   0.9852
V proj                 mean=   0.0798  std=   0.5434  rms=   0.5492  |max|=   1.1607
Q after RoPE           mean=   0.1273  std=   0.6037  rms=   0.6170  |max|=   1.5693
attention output       mean=  -0.0277  std=   0.1169  rms=   0.1202  |max|=   0.2307
attn out-proj (wO)     mean=   0.0100  std=   0.0639  rms=   0.0647  |max|=   0.1615
after attn residual    mean=   0.0180  std=   0.1029  rms=   0.1045  |max|=   0.2795
RMSNorm(ffn)           mean=   0.1715  std=   0.9844  rms=   0.9993  |max|=   2.6601
SiLU-gated hidden      mean=  -0.0167  std=   0.1614  rms=   0.1622  |max|=   1.0244
FFN down-proj          mean=  -0.0014  std=   0.0840  rms=   0.0840  |max|=   0.2341
block output           mean=   0.0166  std=   0.1344  rms=   0.1355  |max|=   0.3652
=> RMSNorm invariant: both norm outputs have rms ~ 1 (1.000, 0.999) — HOLDS
```

盯着 `rms` 这一列看。embed 进来时 rms 是 0.0782（一个很小的数，因为权重按 `1/sqrt(fanIn)` 初始化），过一次 `RMSNorm(att)` 立刻变成 1.0000。这就是 RMSNorm 的全部工作：不管输入幅度是多少，输出的均方根永远是 1（乘上每维的 gain，gain 初始化在 1 附近）。

`core/tensor.ts` 的 `rmsNorm` 干的就是这件事——没有减均值，没有 bias，只有「除以自己的 rms」：

```ts
export function rmsNorm(x, weight, eps = 1e-6) {
  // ...
  const inv = 1 / Math.sqrt(ss / n + eps);
  for (let j = 0; j < n; j++) out[row + j] = x.data[row + j] * inv * weight.data[j];
  // ...
}
```

再看 `after attn residual` 那行：rms 从注意力输出的 0.0647（attn out-proj）跳到 0.1045。这是残差加法——`h1 = h + attnOut`，主干把子层的输出**加**了进来，所以幅度比任一加数都大。然后下一行 `RMSNorm(ffn)` 又把它拉回 0.9993。这一升一降在每个 block、每一层反复发生：残差让主干单调增长，RMSNorm 在每个子层入口把它归一。一个深网络能堆 4 层、40 层、100 层而不在第三层就 overflow 成 NaN，靠的就是这个不变量。

代码末尾把它写成了一个被检查的断言，不是凭感觉说「看起来 norm 了」：

```ts
const ok = Math.abs(normedRms - 1) < 0.1 && Math.abs(normed2Rms - 1) < 0.1;
```

这是本书的一个基本态度：每个声称的不变量都要有一行代码去验证它，输出里打 `HOLDS` 或 `VIOLATED`，不留「我觉得对」的空间。

### RoPE 是一次旋转，所以它不改 rms

顺手看一个能当 sanity check（健全性检查）的细节。`Q after RoPE` 那行的 rms 是 0.6170，跟 RoPE 之前 `Q proj` 的 0.6170 **完全相等**。这不是巧合：RoPE（旋转位置编码，用「转角度」的方式把位置信息塞进 Q/K）的本质是对每一对维度做二维旋转，而旋转保模长。`core/tensor.ts` 的 `rope` 就是标准的二维旋转矩阵：

```ts
const cos = Math.cos(angle);
const sin = Math.sin(angle);
q[i] = q0 * cos - q1 * sin;
q[i + 1] = q0 * sin + q1 * cos;
```

所以如果你 RoPE 实现错了（比如转角符号搞反、配对维度配错），rms 会偏移。「RoPE 后 rms 不变」是一个免费的正确性探针。

RoPE 还藏着第 2 章的伏笔：它编码的是**相对**位置——位置 i 的 query 和位置 j 的 key 之间的注意力分数只依赖 `(i - j)`。这意味着一个在位置 5 算出来的 key，它的旋转是「位置 5 的绝对旋转」，永远冻结、永远有效，不会因为后面又来了 100 个 token 就失效。这正是 KV 缓存能成立的几何理由：缓存的 K/V 不需要重算。本书把 RoPE 放在 `core/` 而不是某个 stage 里，就是因为它是缓存正确性的根。

## 注意力为什么是 O(seq²)，以及它为什么必须

现在讲这一章最重要的成本论点。注意力要为**每一对** (query, key) 算一个分数。一个 query token 看 `len` 个 key，就是 `len` 次点积；而一次完整前向里有 seq 个 query token，因果约束下第 i 个 token 看 0..i 共 i+1 个 key，加起来是 `seq*(seq+1)/2` 对——这就是平方。

`core/model.ts` 的 `attendToken` 是注意力的核心，那个 `for p` 循环就是平方的来源：

```ts
for (let hh = 0; hh < cfg.nHeads; hh++) {
  const kvHead = Math.floor(hh / groupSize); // GQA: 这个 query 头读哪个 KV 头
  const scores = new Float64Array(len);
  for (let p = 0; p < len; p++) {          // <- 对每个缓存位置算一个分数
    const kOff = p * (cfg.nKVHeads * cfg.dHead) + kvHead * cfg.dHead;
    let dotv = 0;
    for (let d = 0; d < cfg.dHead; d++) dotv += q[qOff + d] * kAll[kOff + d];
    scores[p] = dotv * invSqrtD;
  }
  const probs = softmax(tensor(scores, [1, len])).data;
  // ... 用 probs 加权求和 V ...
}
```

本章实验 `[b]` 把这条「无缓存参考路径」（`forwardNoCache`）在 seq = 16/32/64/128 上跑，打印点积数和墙钟时间。真实输出：

```
seq | qk-dot-products | wall-clock ms | ms/seq^2 (x1e-6) | growth vs prev
 16 |            2176 |         5.049 |        19723.242 | —
 32 |            8448 |        10.169 |         9930.892 | 2.01x for 2x seq
 64 |           33280 |        21.136 |         5160.205 | 2.08x for 2x seq
128 |          132096 |        45.000 |         2746.606 | 2.13x for 2x seq
```

先看 `qk-dot-products` 列：2176 → 8448 → 33280 → 132096。seq 翻倍，点积数恰好翻 4 倍（精确平方，因为它就是 `seq*(seq+1)/2 * nHeads * nLayers` 算出来的解析值，不是测量值）。这是确定的、无可争议的平方。

墙钟那列要看得更仔细，这里有个**实测的陷阱**。`growth vs prev` 是 2.01x → 2.08x → 2.13x。如果整个前向是纯 O(seq)，每次 seq 翻倍时间应该恰好翻 2.0 倍。实际比值在 2.0 之上爬，但爬得很慢——为什么不是 4 倍？

因为在这个**玩具尺寸**下，每个 token 的 O(seq) 工作（FFN、Q/K/V 投影，这些跟序列长度无关、只跟 token 数成正比）才是耗时大头，注意力的 O(seq²) 项还很小。总时间 = O(seq) 的线性大头 + O(seq²) 的平方小头，所以归一化的 `ms/seq²` 那列在玩具尺寸下反而随 seq 下降（线性项被 seq² 一除越来越小）。平方的签名不在绝对数里，而在**增长比超过 2.0 这件事**上——2.01 → 2.08 → 2.13 就是平方项在线性大头之上开始冒头。模型更大、序列更长时，注意力会反超，这列才会压平。

这是本书反复强调的诚实：**绝对毫秒数是悲观的**（toy float64 循环，没做任何 kernel 优化），能迁移到真实引擎的是**形状**（平方）和**不变量**，不是这几个 ms。把玩具的绝对耗时当真，是读这类「从零手写」材料最常见的误读。

### 因果掩码为什么必须

`attendToken` 上面那段代码里其实没有显式的掩码——注释写着「the cache *is* the mask」（缓存本身就是掩码）。这是缓存路径的取巧：调用方只缓存「过去 + 当前」这些 token，未来的 token 根本没进 `kAll`，所以不需要额外屏蔽。但「无缓存参考路径」`forwardNoCache` 是逐 token 推进、每个 token 只拿到 `histLen+1` 个 K/V，效果一样：token i 物理上拿不到 i 之后的 key。

为什么必须？因为这是 decoder 的因果性（causality）：训练时模型要预测下一个 token，如果它能「看到」未来的答案，训练就是作弊，推理时也会行为错乱。`core/tensor.ts` 提供了一个独立的 `causalMask`，用来给一次性算整个 [seq, seq] 分数矩阵的实现加掩码，它的注释点出了一个真实的暗坑：

```ts
// 0 on/below the diagonal, -Infinity above it.
out[i * seq + j] = j > i ? -Infinity : 0;
```

注意是 `-Infinity`，不是 `-1e9`。为什么不用一个大负数？因为 `exp(-Infinity) = 0`，masked 位置的 softmax 权重**精确**为 0；而 `exp(-1e9)` 在 float64 里也是 0，看起来没差，但如果有人图省事写成 `-1e9`、再被某个量化或低精度路径放大，就会留下 ~1e-9 的「泄漏」——token 偷看到了一点点未来。一两层看不出来，叠 40 层后是一个真实存在、极难定位的正确性 bug。用 `-Infinity` 是把这个漏洞从源头堵死。

> ✦ **面试高频暗坑。** 能背出注意力公式 `softmax(QKᵀ/√d)V` 的人很多，能说清「为什么 decode 阶段每一步的 Q 只有 1 行、KV 却是整段」的人少。看 `attendToken`：query `q` 是单个 token 的向量（`qOff = hh * dHead`，一个头一行），但 `kAll` / `vAll` 是从位置 0 到当前的**整段**。生成第 N 个 token 时，新 token 的 Q 只跟它自己有关（1 行），但它要去注意前面所有 N-1 个 token 的 K/V（N 行）。这一行 vs 整段的不对称，就是 KV 缓存存在的全部理由：Q 每步都新算无所谓（就 1 行），但 K/V 如果每步都把整段重算，就是 `forwardNoCache` 的 O(seq²) 灾难。把已经算过的 K/V 存下来复用，每步就只多算 1 行 K/V，前向从 O(seq²) 降到 O(seq)。这是第 2 章的全部内容，而它的正确性证明，就是缓存路径的 last-token logits 必须等于本章 `forwardNoCache` 的 golden logits。

> ⚡ **开放问题：那堵平方的墙。** 注意力的 O(seq²) 不是实现笨，是数学定义如此——每对 token 都要交互。当 context 从 4K 涨到 128K、1M，平方项就从「玩具里看不见的小头」变成「吃掉一切的主项」。2024–2025 的一大批架构就是冲着绕开这堵墙去的：线性注意力、状态空间模型（SSM，如 Mamba-2）、RWKV，以及把它们和标准注意力混搭的混合架构。它们的共同思路是用一个**固定大小**的状态（而不是随 seq 线性增长的 KV 缓存）来概括历史，把每步成本压成 O(1)、整体压成 O(seq)。代价是这个固定状态是有损压缩，长程精确召回（比如「大海捞针」式的精确检索）通常打不过老老实实存全部 KV 的标准注意力。**目前没有通用解**：哪种架构能在「线性成本」和「不丢长程信息」之间取得对所有任务都更优的平衡，仍在研究中，工业界主流模型 2025 年仍以标准注意力 + KV 缓存优化（分页、量化，本书后面几章）为主。理解了本章这个精确的平方代价，你才知道那些新架构到底在省什么、又拿什么换。

## 失败模式：softmax 不减 max 会 NaN

注意力的最后一步是 softmax，把分数变成一个概率分布。这里有推理数值学里**最经典的一课**：永远不要直接对原始 logit 取 exp。

`core/tensor.ts` 的 `softmax` 在 exp 之前先减去了每行的最大值：

```ts
let max = -Infinity;
for (let j = 0; j < cols; j++) if (x.data[row + j] > max) max = x.data[row + j];
let sum = 0;
for (let j = 0; j < cols; j++) {
  const e = Math.exp(x.data[row + j] - max);  // <- 减 max 是关键
  out[row + j] = e;
  sum += e;
}
```

减 max 在数学上不改结果（分子分母同乘一个常数），但让最大的那个指数恰好是 `exp(0) = 1`，其余都 ≤ 1，永不溢出。不减会怎样？本章实验 `[c]` 直接证给你看。

它先用本章 `[a]` 真实捕获的那一行注意力分数（235 个位置，max logit = 1.103）跑「减 max」和「不减 max」两个版本：

```
real attention scores (len=235, max logit=1.103): naive vs stable max prob drift = 4.34e-18 (agree — toy logits are too small to overflow)
```

两者一致，drift 是 4.34e-18（浮点噪声级别）。这恰恰是**危险**的地方：在玩具的小 logit 下，错误的实现**看起来完全正确**。它不是「总是坏」，是「有条件地坏」——这种 bug 最难抓，因为你的测试可能永远命中不了触发条件。

然后实验把同一行真实分数线性放大，把最大 logit 推到 800（`exp(800)` 远超 float64 的 ~1.8e308 上限）。注意这不是凭空捏造一个 NaN，是拿真实数据做了一个变换。结果：

```
amplified scores (max logit=800.0, exp(800) overflows float64):
  naive softmax  -> hasNaN=true  sum=NaN  [first 3: 0.00e+0, 0.00e+0, 0.00e+0]
  stable softmax -> hasNaN=false  sum=1.000000  [first 3: 0.00e+0, 0.00e+0, 0.00e+0]
=> PROVEN: max-subtraction is load-bearing — naive path NaNs, stable path stays a valid distribution (sum=1).
```

不减 max 的版本：`exp(800)` = Infinity，sum = Infinity，`e / Infinity` 产生 NaN，整行分布污染成 NaN，hasNaN=true。减 max 的版本：sum = 1.000000，仍是一个合法概率分布。这就是「减 max 是 load-bearing（承重的）」的实测证据——它不是装饰，拆了就塌。

这里要诚实标注一个边界：在本章的 toy 配置下，float64（约 1.8e308 上限）其实**扛得住**真实模型那种 +40..+80 的注意力 logit（`exp(80) ≈ 5.5e34`，没溢出）。所以 `[c]` 是把分数**人为放大到 800** 才触发的失败。但 `core/tensor.ts::softmax` 的注释点出了为什么这一课在真实引擎里不是玩具问题：float32（`exp(89) = inf`）扛不住，第 7 章的量化路径更扛不住。减 max 让结果在「精确算术下完全相同」但「任何精度下都不溢出」——这正是为什么所有真实 kernel（包括 FlashAttention 那套在线 softmax）都把减 max 当成不可省的第一步。本书把减 max 写成无条件的，就是为了让这一课在每一行 softmax 上都成立、行为在不同精度间一致。

## Golden logits：后续所有章节的对拍基准

最后，实验 `[d]` 产出本章的交付物：三个不同长度 prompt 的 last-token logits 的指纹（argmax token + argmax 的 logit 值 + 平均 logit + 一个 order-sensitive 的 checksum）。三个 prompt 来自 `core/tokenizer.ts::PROMPTS`，长度故意拉开（避免 N=1 的运气）。真实输出：

```
prompt(len) | argmax tok | logit[argmax] | mean logit | checksum
p0(  3)     |        112 |        1.9430 |    -0.0338 | -1.473068e+3
p1( 48)     |          2 |        1.5833 |     0.0045 | 3.655462e+2
p2(235)     |         46 |        1.5365 |    -0.0408 | -2.149947e+3
=> determinism: two independent builds drift by 0.00e+0 (must be 0 for golden to be a valid reference)
```

最后那行是 golden 能成立的前提：用同一个种子独立 build 两次模型、各跑一次前向，两次的 logits 逐元素 drift 是 **0.00e+0**——bit-for-bit 一致。这不是运气，是设计。`core/model.ts` 用的 `mulberry32` 是一个可种子化的 PRNG，整个模型 `buildModel(cfg, seed)` 是 `(cfg, seed)` 的纯函数，加上前向全是确定性 float64 运算，所以输出可完全复现。

为什么这件事是地基级的？因为后续每一章的正确性证明都是「我的新路径 reproduce 了本章的 golden」。如果 golden 自己都不确定（两次 build 漂移非零），那所有跨章节的对比都建在沙子上。所以代码把它写成一个 loud assertion，drift 必须是 0，不是「差不多」。

checksum 用的是一个 order-sensitive 的滚动求和，不是密码学哈希，只是一个便宜的、确定的指纹——任何一个 logit 变了它就变。256 个 float 全打出来没法读，打一个 checksum + argmax + 几个统计量，足够检测任何漂移，又小到能一眼扫。

## 小结：这一章给后面留下了什么

你现在手里有一个慢得诚实的前向。它建立了三件后面要反复引用的东西：

1. **RMSNorm 不变量**——每个子层入口 rms ≈ 1，是深层不爆炸的机制（实测 1.000 / 0.999 HOLDS）。后面任何改动若让某层 rms 跑飞，第一个该怀疑的就是它。
2. **注意力的 O(seq²)**——精确平方的点积数（2176 → 132096）和「增长比超过 2.0」的墙钟签名。第 2 章的 KV 缓存就是来把它降到 O(seq) 的，理由是那个「Q 一行、KV 整段」的不对称。
3. **golden logits**——三个 prompt 的确定性指纹（drift = 0），后续 KV 缓存、采样、连续批处理、分页、投机、量化每一章都要对着它对拍。

下一章（第 2 章）正式动手做 KV 缓存：把本章 `forwardNoCache` 里每步重算的整段 K/V 存下来复用，并用本章的 golden 证明它「更快但算出同一个东西」。
