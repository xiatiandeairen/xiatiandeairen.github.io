---
title: "量化 int8/int4：用精度换显存和带宽"
slug: "07"
collection: "llm-inference"
order: 7
summary: "第 02 / 05 章把 KV cache 的显存账算到了 token 级，第 06 章的投机解码靠多搬一遍小模型换吞吐——这章直接砍掉搬运量本身：把权重和 KV 从 float 压到 int8/int4，8x~16x 省显存、省带宽。我们实现对称量化的 per-tensor 与 per-channel，用 maxLogitDrift / perplexity 把精度损失钉在数字上，并复现量化的头号失败模式——单个离群值通道毁掉整个 tensor 的 scale。这是吞吐与显存的收尾章，把'精度-显存-速度'三角落到可跑代码上。"
topics:
  - "LLM 推理"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

decode 阶段（一次生成一个 token 的阶段）是 memory-bound（瓶颈在访存，不在算力）——这件事我们在第 01 章就量过，第 06 章讲投机解码时又用过一次：GPU 的算力远远跑赢它的内存带宽，所以每生成一个 token，瓶颈不是乘法做得快不快，而是把几个 G 的权重从显存搬到计算单元要多久。这一章的全部杠杆就压在这句话上：**如果一个权重不需要 64 位来表示，那它从显存到计算单元的搬运量也能等比例砍掉。** 把权重从 float 压到 int8，搬运量降到 1/8;压到 int4，降到 1/16。在 decode 这个访存瓶颈上，搬运量降多少，速度就近似涨多少。这通常是整个推理引擎里单点收益最高的一个优化——比第 06 章的投机解码更直接，因为它砍的是瓶颈本身而不是绕过它。

代价是精度。一个权重压成整数再还原回来，会有舍入误差;这个误差会顺着前向传播放大成 logit 漂移，最坏情况翻转 argmax（采样选中的那个 token 变了）。这一章要做的，就是把这笔"精度换显存"的账用真实数字算清楚:省多少、错多少、什么时候会崩。

> 本章配套代码 `examples/llm-inference-from-scratch/src/stage07-quantization.ts`，`VOCAB_SIZE=256 | seed=42`，deterministic（重跑数字完全一致）。下文所有数字来自这份代码的真实运行输出。**重要 caveat 先说**:配套用的是合成的近均匀随机权重，这是量化的**最好情形**——真实 LLM 权重是 heavy-tailed（重尾，少数权重幅值远大于多数），绝对漂移会更大。可迁移的是**相对结论**,不是绝对数值。后文反复会标。

## 量化到底在做什么:一个数的对称映射

先把机制讲透，再讲它怎么坏。

量化的核心是一个线性映射:把一段浮点区间均匀地铺到整数网格上。配套用的是 symmetric quantization(对称量化，零点固定在 0)——这是权重量化的标准做法，因为权重大致以 0 为中心,花一个 zero-point(零点偏移)去平移区间几乎不赚,还把反量化(dequant)搞复杂。

对称量化只存一个元数据:scale(缩放因子)。算法三步:

```
scale = absmax / qmax        // absmax = 这组数的绝对值最大值
q     = round(w / scale)     // 量化:浮点 -> 整数网格
w'    = q * scale            // 反量化:整数网格 -> 近似浮点
```

`qmax` 是 b-bit 有符号整数的正向上限:int8 是 127,int4 是 7。看配套代码 `stage07-quantization.ts`:

```typescript
// qmax for a b-bit SIGNED integer. int8 -> 127, int4 -> 7. We use the symmetric
// range [-qmax, qmax] (i.e. 255 of the 256 int8 codes, dropping -128) because a
// symmetric range keeps round(0)=0 exact and avoids a lopsided quantization grid.
function qmaxForBits(bits: number): number {
  return (1 << (bits - 1)) - 1;
}
```

这里有个容易被忽略的取舍:int8 明明有 256 个码位,我们只用 255 个(丢掉 -128)。为什么?因为对称区间 `[-127, 127]` 才能让 `round(0) = 0` 严格成立——0 是权重里最高频的值附近,让它精确映射到 0 比多榨一个码位重要。这是"对称"二字真正买到的东西。

### 为什么用 absmax 而不是均值/标准差

量化函数第一遍扫的是 absmax,不是 mean 或 std:

```typescript
// pass 1: absmax over the slice. absmax (not std/mean) because the grid must span
// the most extreme value or that value clips — and a clipped weight is an
// unbounded error, the worst kind. This is precisely why ONE outlier poisons a
// whole per-tensor group: it stretches absmax so every normal weight gets a coarse
// grid, while per-channel confines the damage to the outlier's own channel.
let absmax = 0;
for (let i = 0; i < count; i++) {
  const v = Math.abs(src[start + i * stride]);
  if (v > absmax) absmax = v;
}
const scale = Math.max(absmax / qmax, SCALE_EPS);
```

这个选择是整章的伏笔。网格必须覆盖最极端的那个值,否则它会被 clip(截断到 qmax),而 clip 是**无界误差**——最坏的一种误差。但反过来:**absmax 由最大值决定,意味着一个离群值就能把整个网格撑粗。** 这正是后面 outlier 灾难的根。

注意 `Math.max(..., SCALE_EPS)` 这个地板。`SCALE_EPS = 1e-12`。这不是洁癖:

```typescript
// Invariant: scale > 0 always. An all-zero group has absmax 0, which would make
// scale 0 and produce 0/0 = NaN on dequant. We floor scale at a tiny epsilon so a
// dead channel dequantizes to exactly 0 instead of NaN — this is a real failure
// mode in sparse/pruned models, not a hypothetical.
const SCALE_EPS = 1e-12;
```

**失败模式 #0(最朴素的那个)**:一个全零的 channel(剪枝/稀疏模型里很常见)absmax = 0 → scale = 0 → 反量化时 `0/0 = NaN`,一个 NaN 顺着矩阵乘法污染整层 logit,整个 batch 的输出全废。地板把死 channel 安全地反量化成精确的 0。这种"看似冗余的防御性代码"是顺序敏感且必须的,不是可删的洁癖。

## 第一笔账:省多少显存

`stage07` 的 `[a]` 段直接报内存占用(精确算术,标 est. 因为是按矩阵尺寸算出来的,不是测出来的):

```
quantizable weights = 262144 params
float64 (baseline) =   2.00 MiB (est.)  1.0x smaller
float16            = 512.00 KiB (est.)  4.0x smaller
int8               = 256.00 KiB (est.)  8.0x smaller
int4               = 128.00 KiB (est.)  16.0x smaller
```

线性关系一目了然:bit 数砍一半,体积砍一半。int8 相对 float64 是 8x,int4 是 16x。这就是搬运量的减少倍数——decode 既然 memory-bound,这个倍数近似就是带宽收益的上限。

但权重不是唯一的显存大户。**第 02 / 05 章我们反复算过 KV cache 的账**:它随 `seq × batch` 线性涨,长上下文下经常比权重还大。所以 KV 也要独立量化(实践中常见组合是权重走 int4、KV 走 int8):

```
KV cache @ seq=512 batch=16:
float64 (baseline) =  16.00 MiB (est.)  1.0x smaller
int8               =   2.00 MiB (est.)  8.0x smaller
int4               =   1.00 MiB (est.)  16.0x smaller
```

这里有个关键判断:KV 量化和权重量化是**两条独立的主线**,因为它们卡在不同地方。

- **weight-only 量化**省的是模型常驻显存 + decode 时每个 token 都要重搬的权重带宽。这条线最成熟,int4 早已是生产标配。
- **KV cache 量化**直接砍第 02 / 05 章那本越写越厚的 KV 账,让同样的显存能塞更长的 context 或更大的 batch。这条线难一点,因为 KV 是 activation(激活值,前向算出来的中间结果)的缓存,而 activation 的离群值问题比权重严重得多——后面会讲为什么这是真正的战场。

配套对哪些权重量化、哪些保留高精度,也是有讲究的:

```typescript
// Build a model whose every weight matrix is fake-quantized. RMSNorm gains are left
// in float64 on purpose: they are O(dModel) tiny vectors (negligible memory) but
// multiply every activation, so quantizing them buys nothing and hurts a lot — real
// engines keep norm/scale params and often the embedding in higher precision for
// exactly this reason. We quantize the big matmul weights, which are >99% of params.
```

RMSNorm 的 gain 是 `O(dModel)` 的小向量,占显存可忽略,但它乘到每一个 activation 上——量化它省不下什么、却毁掉一切。真实引擎也是这么干的:量化占 99%+ 参数的大矩阵,把 norm / scale / 经常还有 embedding 留在高精度。**量化不是越激进越好,是挑对地方激进。**

## 第二笔账:错多少——fake-quant 与漂移度量

怎么测精度损失?配套用的是 fake-quant(伪量化):量化到整数网格后,**立刻反量化回 float64**,再跑正常的浮点 kernel。

```typescript
// What we ACTUALLY do here (and what every "fake-quant" / simulated-quant path in a
// real framework also does): quantize a weight to the integer grid, then immediately
// DEQUANTIZE back to float64 and run the normal float kernels. The arithmetic the
// model sees is the *rounded* weight. ... The drift we measure is exactly the drift
// a true int8 kernel would produce, because both compute on the same rounded values.
```

为什么这样测是诚实的?因为模型看到的是**舍入后的权重**,而真正的 int8 kernel 算的也是同一批舍入后的值——两者数值上等价。fake-quant 把"量化注入的数值误差"这唯一一个我们关心的东西,从"写 int8 矩阵乘法 kernel"这件正交的工程里隔离出来。这一章只谈误差,不谈 kernel。

度量用两个,缺一不可(代码 `core/metrics.ts`,在 `stage07` 里组合使用):

- `maxLogitDrift` —— L∞ 范数(取最大单点偏差),抓那个把 argmax 翻掉的灾难 logit。
- `perplexity` 漂移 —— 抓 argmax 侥幸没翻、但整个分布已经偏掉的情况。

```typescript
// maxLogitDrift is L∞ (catches the one catastrophic logit that flips an argmax);
// perplexity drift catches a distribution skew that argmax happens to survive. We
// need both: a scheme can ace one and fail the other.
```

跑出来的精度损失曲线(`[b]` 段,baseline `argmax=[2,46] PPL=300.37`):

```
int8 per-tensor   drift=2.60e-2  PPL=299.54 (Δ-0.83)  argmax kept
int8 per-channel  drift=3.43e-2  PPL=300.55 (Δ+0.18)  argmax FLIPPED
int4 per-tensor   drift=3.71e-1  PPL=294.58 (Δ-5.80)  argmax FLIPPED
int4 per-channel  drift=4.22e-1  PPL=295.02 (Δ-5.35)  argmax kept
```

第一个硬结论:**int4 的漂移比 int8 大约 10 倍**(`3.71e-1` vs `2.60e-2`)。原因是网格密度:int8 有 255 个网格档位,int4 只有 15 个。bit 砍一半不是误差翻倍,是误差量级跳一档。这就是"精度-显存"三角里最直白的一条边:int4 省一倍显存,买单的是一个数量级的精度。

第二个结论会让人意外:**在这个 no-outlier 场景里,per-channel 并没有更好,甚至略差。** per-channel 是给每个输出 channel(矩阵的每一列)单独算 scale,而 per-tensor 是整个矩阵共用一个 scale。直觉上 per-channel 更精细应该更好,但这里它的 PPL 反而略升:

```
-> per-channel here is NOT better (slightly worse): with near-uniform
   weights and no outliers, every column's absmax ≈ the tensor absmax, so
   per-channel's extra scales just add rounding noise. Its payoff is
   outlier-specific.
```

道理很干净:权重近均匀、没有离群值时,每一列的 absmax 都约等于整个 tensor 的 absmax——per-channel 多算的那一堆 scale 没带来更细的网格,只多引入了一点舍入噪声。**per-channel 不是无条件更好,它的回报是 outlier-specific 的。** 这一步是下一节的反衬:你得先看到它在好情形下不赚,才理解它在坏情形下为什么是救命的。

> ⚠ 别过度解读这些绝对数:`Δ-0.83` 这种 PPL 居然下降,纯属合成数据的噪声(toy 模型对舍入有偶然的正反馈),换真实重尾权重会一致变差。可迁移的是**排序**:int4 漂移 > int8;无 outlier 时 per-channel ≈ per-tensor。

## ✦ 真正的战场:离群值

到这里铺垫完了。现在讲这一章——也是整个 LLM 量化领域——真正的难点。

**weight-only int4 早就是成熟技术。难的从来不是权重,是 activation 的离群值。** 这句话值得拆开说。Transformer 的激活值里存在所谓 emergent outlier features(涌现离群特征):模型规模过了某个阈值(经验上 ~6.7B)后,会出现少数几个维度,其幅值比其他维度大几个数量级,而且这些大值是模型正常工作所**依赖**的,不是噪声。一旦这些离群通道存在,naive 的 per-tensor int8 就崩了——因为前面讲过,scale 由 absmax 决定,一个离群值把 scale 撑粗,所有正常通道被挤进网格底部的几个档位里,整个 tensor 的有效精度被这一个值毁掉。

这正是 SmoothQuant、AWQ 这一系列方法存在的全部理由:它们做的不是"更聪明地舍入",而是"想办法让那几个离群通道不要污染其他人的 scale"。

配套用一个植入的离群通道把这件事复现出来(`[c]` 段):把第 0 层 `w1`(FFN 的 gate 投影矩阵,`[dModel=64, dFF=256]`)的某一列放大 50 倍,模拟真实模型里的离群特征。

```typescript
const OUTLIER_COL = 3;
const OUTLIER_FACTOR = 50; // 50x the surrounding weights — deliberately brutal
function withOutlier(m: Model): Model {
  const w0 = m.layers[0].w1;
  const data = Float64Array.from(w0.data);
  const cols = w0.shape[1];
  for (let r = 0; r < w0.shape[0]; r++) data[r * cols + OUTLIER_COL] *= OUTLIER_FACTOR;
  // ...
}
```

为什么挑 `w1` 而不是 Q 投影?代码注释说得很到位:`w1` 的输出 channel 经 SiLU → 门控 → 下投影 → 残差 → logit,一路畅通地到达输出;而 Q 投影的离群值会被注意力分数的 softmax 大量冲刷掉。**要演示灾难,得让离群值真的能传到输出。** 这是诊断思路:不是随便挑个矩阵注入,是挑那个误差能传播到可观测处的矩阵。

植入后,同样是 int8,per-tensor 和 per-channel 的命运彻底分叉:

```
int8 per-tensor   drift=2.59e-1  argmax FLIPPED
int8 per-channel  drift=3.39e-2  argmax kept
-> per-channel confines the outlier to its own 1/256 scale and keeps argmax;
   per-tensor smears it over all 256 channels and flips the output. ~7-8x gap.
```

对照上一节 no-outlier 时 per-channel `3.43e-2` ——它几乎没变(`3.39e-2`),因为它把那个离群值锁进了它自己那 1/256 的 scale 里,其余 255 列毫发无伤。而 per-tensor 从无 outlier 的 `2.60e-2` 暴涨到 `2.59e-1`(约 10 倍),argmax 翻了。**这就是 per-channel 的全部价值:它不让一个 channel 的灾难外溢成整个 tensor 的灾难。** 上一节它在好情形下"白交了 scale 的成本",在这里连本带利赚回来。

## ✦ 失败模式:per-tensor int4 撞上离群值

把上面这件事推到 int4,就是这一章最该被记住的崩溃(`[d]` 段)。int4 只有 15 个网格档位,当 per-tensor 的单一 scale 被一个 50x 离群值撑到极致,每个正常权重都塌进 0 附近的少数几档——矩阵实际上被摧毁了:

```
per-tensor  int4 drift = 1.35e+0   argmax FLIPPED (was [2,122], now [212,23])
per-channel int4 drift = 4.65e-1   argmax FLIPPED (now [2,46])
blow-up ratio (per-tensor / per-channel) = 2.9x worse drift
```

注意 per-tensor int4 的漂移 `1.35e+0` ——超过 1.0(logit 尺度),argmax 不是简单翻一下,而是落到了**完全无关**的 token(`[2,122]` → `[212,23]`,两个位置全错)。这不是"答案变差了",是"输出彻底乱码"。

```typescript
// int4 has only 15 grid levels, so when per-tensor's single scale is stretched to
// cover a 50x outlier, every NORMAL weight collapses onto a handful of levels near
// zero — the matrix is effectively destroyed. ... This is the concrete reason
// production int4 is NEVER naive per-tensor: it is always per-channel/per-group,
// plus dedicated outlier handling.
```

但请注意一个诚实的细节:**per-channel int4 在这里也翻了 argmax**(`now [2,46]`)。它把漂移压到了 per-tensor 的约 1/3(`blow-up ratio = 2.9x`),损害被遏制了,但 4-bit 本身就是真有损的,toy 模型又没有容错冗余,一个 token 还是翻了。代码注释把结论钉死:

```
-> a SINGLE outlier channel + naive per-tensor int4 corrupts the whole layer.
   per-channel softens it ~3x but 4-bit still flips a token here: at int4 you
   need per-group + outlier handling, not just per-channel.
   This is why outliers are the #1 problem in LLM quantization.
```

这就是为什么生产级 int4 **从来不是** naive per-tensor。它的标配是:per-channel / per-group(按更细的组分别算 scale)打底,再叠专门的离群值处理——给离群通道单独留高精度(mixed-precision,混合精度),或者用旋转/平滑的技巧(SmoothQuant 把离群值从 activation"挪"一部分到权重,AWQ 按激活幅值给重要通道保护)把离群能量摊开。**展示崩溃本身就是目的:只跑 happy path 的 demo 会让你完全不理解这一整套机器为什么存在。**

这也回答了为什么 KV cache 量化比 weight-only 更棘手:KV 缓存的是 activation,而离群值问题在 activation 上比在权重上严重得多。weight-only int4 早就稳了,activation / KV 的低 bit 量化才是 SmoothQuant / AWQ 这一代工作真正在啃的骨头。

## ⚡ 开放问题:activation 量化与极低 bit 的前沿

**2024-2025 的生产现状**已经很清楚,几条主线并行:

- **FP8** 成了 H100 / B200 的原生支持格式(8-bit 浮点,有指数位,对离群值比 int8 友好得多)。DeepSeek-V3 直接用 FP8 **训练**,等于把推理量化的友好性做进了模型前端——训练时就让权重/激活适配低精度,而不是事后硬压。这是一个方向性的转变:量化不再只是推理期的后处理,开始往训练期前移。
- **per-group int4**(GPTQ / AWQ 这一系)是 weight-only 的成熟答案,把本章的 per-channel 进一步细化到组级 + 校准数据驱动的舍入。
- **KV cache 量化**(int8 / fp8 KV)已是长上下文服务的标配,直接兑现第 02 / 05 章的 KV 显存账。

但有一块**目前没有通用解,仍在活跃研究**:**activation 离群值的低 bit 量化,以及 int4 以下(int3 / int2 / 1.58-bit)的极低 bit 量化。** 本章 `[d]` 已经演示了,即使 per-channel,int4 在有离群值时仍可能翻 token——这不是实现没做好,是 4-bit 的信息容量在离群值面前本就吃紧。SmoothQuant 用平滑、AWQ 用激活感知保护、QuaRot / SpinQuant 用旋转矩阵把离群值"转散",各有适用边界,没有一个方法对所有模型/所有层都最优;什么时候该混合精度、什么时候该旋转、离群值在不同架构(MoE、超长上下文)里怎么迁移,都还在 per-model 调。再往下到 1.58-bit(BitNet 那条线),则需要从训练阶段就改架构,不是推理期能后压出来的。**"用更低 bit 而不崩"这件事的通用配方,到 2026 年仍然没有定论。**

## 把三角钉在数字上

这是吞吐与显存的收尾章,回头看"精度-显存-速度"三角,现在每条边都有数字:

| 维度 | int8 | int4 |
|---|---|---|
| 显存 / 带宽(省) | 8x | 16x |
| logit 漂移(无 outlier) | `2.60e-2` | `3.71e-1`(~10x 于 int8) |
| 有 outlier 时(per-tensor) | argmax FLIPPED | drift `1.35e+0`,输出乱码 |
| 有 outlier 时(per-channel) | argmax kept | argmax 仍可能翻,但 drift ~1/3 |

三角的取舍法则:**bit 越低,省得越多,但对离群值越脆;per-channel/per-group 是把离群值伤害局部化的最低成本手段,int4 还要再叠离群值处理。** 至于"省了带宽到底快了多少"——把量化接进真实推理路径、配上前几章的 KV cache 与连续批处理,在第 08 章的基准测试里用真实 tok/s 收口。

> 再强调一次 caveat:本章数字来自近均匀合成权重(量化的最好情形),真实重尾权重会更差。请只迁移**相对排序与失败模式**:int4 漂移 > int8;per-channel 仅在 outlier 下胜出;per-tensor int4 + outlier 必崩。
