---
title: "nanoGPT 真训练：在玩具语料上看它收敛"
slug: "08"
collection: "tiny-dl"
order: 8
summary: "终章合龙：把前七章手写的 autograd、层、优化器、注意力拼成一个微型 GPT，在 core/data 的确定性小语料上做字符级语言建模，端到端训练到收敛再自回归采样。讲清为什么先在小数据上「过拟合到背诵」是验证实现正确的金标准、温度如何调节确定性 vs 多样性、val loss 何时该早停，以及四种把它训坏的方式。读完你手里这条 char-GPT 训练链与真实 nanoGPT/GPT-2 在结构上同构，差的只是规模与 kernel。"
topics:
  - "深度学习"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

前七章你逐件验证过零件：autograd 的 adjoint 过了数值梯度检查，Linear/LayerNorm/Embedding 单测过，AdamW 在凸面上收敛，注意力的 mask 和 softmax 数值对得上。现在做这本书唯一真正重要的实验——把这些零件接成一个 transformer，用梯度下降喂它一段文本，看它**到底学不学得会**。这一章是集成测试，也是终点：如果布线里藏着一条断掉的梯度路径或一个错位的 mask,模型不会报错,它只是**学不会**——loss 卡在某个平台下不来,采样吐垃圾。所以这章每个数字都同时是一个断言。

先把判据钉死。`examples/tiny-dl-from-scratch/src/stage08-nanogpt.ts` 跑出来的真实头部输出:

```
vocab size: 24   train tokens: 1788   val tokens: 199
uniform-guess baseline cross-entropy = ln(24) = 3.1781
init train loss: 3.6072   final train loss: 0.2834
baseline 3.1781 -> final 0.2834  (dropped below baseline: true)
```

`ln(24) = 3.1781` 是**均匀瞎猜基线**（uniform-guess baseline,模型对 24 个字符等概率乱猜时的交叉熵）。任何比这个数小的 loss 才意味着模型学到了一点点东西。从 3.6072(初始,比基线还高,因为权重随机)压到 0.2834,这就是「它在学」的硬证据。

## 验证训练框架对不对,看它能不能背书

这是本章的专家钩子,也是整本书的方法论顶点:**验证一个训练框架实现是否正确,标准不是"在大数据上涨点",而是"能不能在一小段文本上把 train loss 压到接近 0 并逐字背出来"。** ✦

为什么是这个标准,而不是看泛化?因为泛化能力受数据、正则、架构超参一堆因素影响,信号脏;而"能否过拟合一小段数据"几乎只取决于一件事——**梯度有没有正确地流过每一个参数**。一个有足够容量的模型,如果 autograd 把所有 adjoint 都算对了、训练循环里 `zeroGrad → forward → backward → step` 的顺序没错,它**必然**能把一段固定文本背下来,因为这本质上是一个记忆任务,不需要任何"智能",只需要梯度能把损失推到 0。

反过来:**背不出来,一定是 backward 或循环有 bug。** 这是个极强的契约。如果你某条注意力分支忘了接进计算图、某个权重的梯度被覆盖而不是累加、或者你在 backward 前忘了 `zeroGrad` 导致梯度脏累加,模型就会卡在一个下不去的平台。它不会抛异常——这正是深度学习 debug 最阴险的地方:**错误是静默的,只表现为"学得慢"或"学不会"**。把"过拟合小数据"当冒烟测试,等于给这条静默失败装了一个会响的警报。

这就是为什么 stage08 故意选了一个**重复设计的玩具语料**(repeating-by-design corpus)。代码顶部的诚实声明把这件事说死了:

```typescript
// HONESTY BOUNDARY (inherited from data.ts, restated because it governs every number below):
//   The corpus is a tiny repeating "song" CHOSEN so a few-thousand-param model can overfit
//   it on a CPU in minutes. Absolute loss/accuracy here are OPTIMISTIC and do NOT generalize
//   to real text. What transfers is the SHAPE of the story: loss descends monotone-ish below
//   the log(vocab) baseline; train loss -> ~0 (memorization) is the signature of a correct
//   implementation;
```

注意这里的取舍:**绝对 loss 是乐观的、不可迁移的;可迁移的是"形状"(SHAPE)**——loss 单调地降到基线以下、train 远小于 val(记住了)、温度换多样性、容量不足的模型证明性地背不下来。这些关系在任何规模都成立,绝对的毫秒数和 loss 值不成立。这是一条诚实的边界,也是 toy 实现唯一能给你的真东西。本书前面讲 autograd 数值检查时立的规矩——只在能验证的地方声明结论——在这里收口。

## 收敛曲线:形状比数字重要

看真实的 train/val 探针(probe,训练中途的抽样测量):

```
train/val loss probes (val = held-out tail; train<<val later = memorization):
  step   0  train 3.6072  val 3.6716
  step  27  train 2.1271  val 2.1441
  step  54  train 1.6591  val 1.6547
  step  81  train 1.1198  val 1.1522
  step 108  train 0.6850  val 0.8089
  step 135  train 0.5352  val 0.4997
  step 162  train 0.3646  val 0.3512
  step 189  train 0.2859  val 0.3484
  step 216  train 0.2658  val 0.3285
  step 219  train 0.2834  val 0.2771
```

读这张表要读三个东西。

**第一,初期 train ≈ val。** step 0 到 step 81,两条线几乎贴着走(3.61/3.67、2.13/2.14、1.12/1.15)。这阶段模型在学**通用结构**——哪些字符常接哪些,空格在哪——这种知识对训练集和留出集(held-out,没参与训练的尾部文本)同样有用,所以两边一起降。

**第二,后期 train 开始低于 val。** 到 step 189,train 0.2859 但 val 0.3484。这个 gap 就是**记忆化(memorization)的指纹**:模型开始记住训练串的具体细节,这些细节对留出集没用,所以 train 继续降而 val 摆动。在真实大模型训练里,这个 gap 张开就是该早停(early stopping)的信号——继续训只会过拟合。但**在这个玩具任务里我们故意不早停**,因为我们的目标恰恰就是过拟合到背诵,以此验证布线正确。这是 toy 与生产的关键语境差:同一条 train/val gap,在生产里是"停手"信号,在这里是"成功"信号。

⚡ **早停到底停在哪,目前没有通用解。** 实践里靠 val loss 的耐心计数器(patience,连续 N 次不降就停)是个粗糙的启发式,它假设 val loss 单调可信。但现代大模型训练里出现了反直觉现象:grokking(模型在 train loss 早已到 0、val 长期不动之后,某一刻 val 突然断崖式下降到泛化)、以及 double descent(test loss 随模型规模/训练步数先降后升再降)。这意味着"val 一抬头就停"可能把你停在 grokking 发生**之前**,错过真正的泛化。什么时候该相信 val loss、什么时候该再忍一会,这是开放问题,正在研究中,没有适用所有规模和数据的判据。

**第三,看 ASCII 收敛曲线的形状:**

```
*                                                   3.5265 (max)
 **
   ****
       ******
             ******
                   **********
                             *********************  0.2739 (min)
```

这是 `lossCurve()` 画的(代码 532 行起,把 220 步的 loss 历史分桶取均值)。陡降在前、平台在后——典型的健康下降。如果你看到的是**水平直线**(根本不降)或**锯齿乱跳**(learning rate 太大或梯度爆炸),那是病。注意这条曲线靠 `clipGradNorm(params, 1.0)`(梯度范数裁剪,代码 382 行)护着,把偶发的梯度尖峰压住,免得一个坏 batch 把整轮训练 NaN 掉——这也是为什么曲线平滑而不是间歇性炸刺。

训练循环本身在 `train()` 里(代码 374-390 行),核心五步:

```typescript
const { x, y } = ds.getBatch("train", cfg.blockSize, batchSize, rng);
const logits = model.forwardBatch(x, batchSize, cfg.blockSize); // (batch*T, vocab)
const loss = crossEntropy(logits, y);
model.zeroGrad(); // INVARIANT: clear before backward — autograd accumulates on purpose.
loss.backward();
clipGradNorm(params, 1.0); // cap rare grad spikes so one bad batch can't NaN the run.
opt.step();
```

那行 `zeroGrad()` 的注释是整本书的一个回响:**autograd 是故意累加梯度的**(同一参数多条路径的梯度要相加才对)。代价是每步必须手动清零,忘了清就是脏累加——这正是上一节说的"静默 bug"的经典来源之一。

## 采样:让它把语料背给你听

训练完拿"to be"当 prompt,贪心(greedy,每步取概率最大的字符)续写,真实输出:

```
prompt: "to be"
greedy (temp=0) continuation:
    to be that is the question
    whether tis nobler in the mind to to suffer
    the slings and arrows of outrageous fortune
    or to take
```

它把语料背出来了(那句"to to suffer"的小重复暴露了它是背诵而非真懂,这正符合 toy 的预期)。这是上一节"能背书=布线对"判据的视觉确认:loss 降到 0.28 不是数字游戏,它真的记住了文本结构。

采样代码在 `sample()`(311 行起)。贪心分支很朴素——就是 argmax:

```typescript
if (temperature <= 1e-6) {
  // greedy: argmax. Deterministic regardless of rng.
  nextId = 0;
  let best = -Infinity;
  for (let j = 0; j < logits.length; j++)
    if (logits[j] > best) { best = logits[j]; nextId = j; }
}
```

注意 `temperature <= 1e-6` 这个判断:**temperature=0 在数学上会让 `logits / temperature` 除零**,所以代码不真的除,而是把"温度趋于 0"等价实现为 argmax。这是个必须显式处理的边界,不是可以偷懒的细节。

## 温度:确定性与多样性的旋钮

同一个 prompt、同一个随机种子,只换温度,真实对比:

```
temp=0.0:
    to be that is the question
    whether tis nobler in the mind to to suffer
    the slings and
temp=0.5:
    to be thand be ie the the thousatusand natural shocks
    that flesh is heir tis he ho co
temp=1.0:
    to be tha io the the the themm queandaionles sarrleal
    ry oppppososing osind
```

温度(temperature)就是 softmax 前给 logits 做的除法因子,大白话:**调模型有多敢冒险**。看这三段的退化梯度:

- **temp=0**:贪心,逐字背诵,零冒险也零多样性。
- **temp=0.5**:开始出现真词的碎片("natural shocks""flesh is heir")但拼写崩坏("thousatusand"),分布被略微抹平,模型偶尔不取最大概率字符。
- **temp=1.0**:从模型原始分布采样,多样性最高,错误也最多("oppppososing"那串重复字母是低概率尾巴被采中了)。

机制在 `sample()` 的非贪心分支(322-339 行):logits 先除以温度再 softmax,然后用一次随机数做**逆 CDF 采样**(inverse-CDF,按累积概率切一刀):

```typescript
const e = Math.exp((logits[j] - max) / temperature);
// ...
const r = rng();
let cum = 0;
nextId = logits.length - 1; // fallback guards float rounding leaving cum < r
for (let j = 0; j < probs.length; j++) {
  cum += probs[j] / denom;
  if (r < cum) { nextId = j; break; }
}
```

温度小于 1,指数被放大,分布变尖,接近 argmax;温度大于 1,分布变平,长尾字符更容易被采中。那行 `nextId = logits.length - 1` 的 fallback 不是摆设——浮点累加可能让 `cum` 最终差一点点不到 1,如果 `r` 恰好落在那个缝里,循环不 break,fallback 兜住,免得返回未初始化的索引。这是 §security 那条"边界 case 要显式兜"的微缩版。

**温度的失败模式:temp=0 退化成贪心重复。** 在真实模型上,贪心解码会陷入复读循环("the the the the"),因为一旦进入一个高概率的自指环,argmax 永远选同一个字符。这就是为什么生产推理几乎从不用纯贪心,而是配 top-k / top-p / repetition penalty。本书姊妹篇 llm-inference 的「采样策略」章把这些生产级解码器讲透了——这里你看到的是它们要解决的**根问题的最小复现**。

## 失败模式一:容量不足,loss 卡平台,背不出来

这是把判据反过来用——**故意造一个学不会的模型,看失败长什么样**。stage08 训了一个微型版(embed=4、heads=1、layers=1),真实输出:

```
tiny model params=476 (vs capable 71832)
tiny init loss 3.2514 -> final 2.4595  (capable final was 0.2834)
tiny stays near/above baseline 3.1781: true
tiny greedy continuation (gibberish / stuck loops — no memorization):
    to be the the the the the the he the the the the the the the the
```

476 个参数 vs 能干活的 71832 个。tiny 版从 3.25 只降到 2.46,**始终徘徊在 ln(24)=3.18 基线附近降不下去**,贪心续写卡在"the the the"的复读环。

为什么?因为**记忆一段文本需要最低限度的参数容量**,476 个参数装不下 1788 个 token 的训练串的全部细节。loss 卡在平台不是 bug,是物理极限——模型再怎么训也压缩不进去。

这里有个**关键的诊断价值**:它告诉你怎么区分两种"loss 下不去"。

- **容量不足**:loss 平台**接近 ln(vocab) 基线**(这里 2.46,离 3.18 不远),且 train 和 val 都下不去。处方是**加容量**(embed/layers/heads)。
- **布线 bug**:loss 平台可能在**任意位置**,常常 train 也卡住——但区别是,一个足够大的模型本该轻松过拟合却没有,这才是 bug 信号。

换句话说,你得**先确认模型容量足够(大到本该能背)**,才能把"背不出来"归因到 backward bug。这就是为什么主实验用的是 71832 参数的"capable"模型而不是 tiny——只有容量足够时,"背不出来=有 bug"这个推断才成立。诊断顺序很重要:容量是前提,不是变量。

## 失败模式二:context 超过 blockSize,显式 shape 报错

最后一个失败模式是工程性的,但极其常见——**采样时上下文长度超过模型训练的 blockSize**(块大小,模型一次能看的最大 token 数)。真实输出:

```
feeding a 37-token context to a blockSize=32 model via the RAW forward...
caught (as designed):
    GPT.forward: context length 37 exceeds blockSize 32; position embedding has no row
    for index 32..36. Crop the context to the last 32 tokens before sampling.
```

根因在位置嵌入(position embedding)。这个 GPT 用的是**学习式位置嵌入**——`posEmb` 是一个 `Embedding(blockSize, embedDim)`,只有 blockSize=32 行,索引 0 到 31。喂进 37 个 token,位置 32-36 **没有对应的嵌入行**。代码在 `forwardSeqIds()`(229 行)把这个失败**主动做响**:

```typescript
if (T > this.blockSize)
  throw new Error(
    `GPT.forward: context length ${T} exceeds blockSize ${this.blockSize}; ` +
      `position embedding has no row for index ${this.blockSize}..${T - 1}. ` +
      `Crop the context to the last ${this.blockSize} tokens before sampling.`,
  );
```

为什么要主动抛、还带上这么具体的消息?因为如果不抛,`Embedding` 的 OOB(越界)检查也会抛,**但它的报错会怪到错误的地方**(说某个 id 越界,而真正的病因是 blockSize 不匹配)。这是 §log 那条"错误消息写根因,不写技术症状"的实践:报"context 37 > blockSize 32,去裁剪"比报"index 32 out of bounds"对调用者有用得多。

正确的对策不是把 blockSize 调大,而是**采样时把上下文裁剪到最后 blockSize 个 token**。安全采样路径 `logitsForContext()`(269 行)就这么做:

```typescript
const cropped =
  ids.length <= this.blockSize ? ids : ids.subarray(ids.length - this.blockSize);
```

真实输出确认:

```
The SAFE sampler avoids this by cropping context to the last blockSize tokens:
    logitsForContext cropped & returned 24 logits (= vocab 24), no error.
```

这个机制有个直接的生产对应:**这就是为什么模型有"上下文窗口"上限,以及超长输入要靠滑动窗口/裁剪/外推**。本书这版用的是最朴素的"裁掉前面",真实系统会用 RoPE 之类的相对位置编码来缓解外推问题——但**问题的根都在这里**:位置信息的表示能力是有限的。

顺带一个边界 bug 的预警:`forwardBatch` 接 `(xFlat, batch, T)` 三个参数,如果你传进去的 `T` 和实际每条序列的真实长度对不上,切片 `xFlat.subarray(b * T, (b + 1) * T)`(256 行)会悄悄切错位置,喂给模型形状对但内容错的数据——这是另一类不抛异常的静默 bug。batch/blockSize 的形状契约要在调用处守住。

## 你已经拥有一个能改的科研 baseline

收束到本章的前沿钩子。把这条链拆开看:token 嵌入 + 学习位置嵌入 → N 层 pre-norm transformer block(每层 = 多头因果自注意力 + 残差 + MLP + 残差)→ 末层 LayerNorm → LM head 投影到 vocab logits → 交叉熵 loss → AdamW + cosine warmup + 梯度裁剪 → 自回归温度采样。

⚡ **这条手写的 char-GPT 训练链,与真实 nanoGPT / GPT-2 的训练管线在结构上是同构的。** 差别只有三处,而且都不是"概念"差别:**数据规模**(你 1788 个 token,GPT-2 几百 GB 文本)、**kernel 优化**(你的注意力是 JS 里 per-sequence 的二维循环,生产是 FlashAttention 之类的融合 kernel)、**并行**(你单 CPU,生产是数据/张量/流水线并行跨上千 GPU)。

代码 26-30 行把第一处差别(为什么慢)说得很诚实:

```typescript
// CORE-OP CONSTRAINT that shapes the design: core matmul/softmax/transpose are 2-D ONLY.
//   So attention is computed PER SEQUENCE (one (T, d) matrix at a time) inside a JS loop over
//   the batch. This is slower than a batched kernel but keeps the autograd graph dead-simple
//   and 100% inside the verified 2-D ops.
```

这个取舍很关键:**用速度换正确性的可验证性**。每条梯度路径都是前几章数值检查过的二维 core op,代价是 O(batch) 的 JS 循环。生产系统反过来——为了速度把这些 op 融成定制 kernel,代价是正确性更难验证(所以才有 FlashAttention 的 reference 实现专门用来对数值)。你这版的价值不在快,在于**每一步都可信、可读、可改**。

这意味着什么?**你现在手里有一个完整的、可改的科研 baseline**。想试个新的位置编码?改 `posEmb`。想验证一个新优化器?换掉 `AdamW`。想看某个 attention 变体在小数据上的收敛形状?改 `MultiHeadSelfAttention.forwardSeq`。所有这些实验,你都能在 CPU 上几十秒跑完一轮(真实计时:`wall-clock: 40.9s for 220 steps`,约 186 ms/step,机器相关,看比例不看绝对值),拿"能不能过拟合小语料"当快速冒烟测试,确认你的改动没有悄悄打断梯度流。

把整本书的弧线连起来:第一章手写标量 autograd 的那个加法节点的 adjoint,到这一章,正在驱动一个会背莎士比亚片段的 transformer 收敛。中间没有任何一处是黑盒——你验证过每一环。这就是这本书想给你的,也是它作为"AI 批量生成的 survey 稿"的反面所在:**survey 告诉你 GPT 是什么,这本书让你拥有一个你能拆开、能改、能验证、能继续往上做研究的 GPT。** 至于接下来怎么让它跑得快、怎么部署、怎么量化压缩——那是姊妹篇 llm-inference 的事;你已经把"它为什么能学"这件事,从骨头里弄明白了。
