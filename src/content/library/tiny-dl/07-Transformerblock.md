---
title: "Transformer block：残差、LayerNorm 与前馈的装配"
slug: "07"
collection: "tiny-dl"
order: 7
summary: "第 06 章把单头 attention 跑通了，但 attention 不是一座可以独立站立的塔——它要靠残差、LayerNorm、前馈组装成可堆叠的 block 才能训练。本章用配套 stage07 实测四件事：残差让 12 层网络底层梯度比无残差强 13 倍、去掉 position embedding 模型对顺序完全无感（差异 1e-17）、Pre-LN vs Post-LN 早期稳定性差距、FFN 的 4x 升维为什么吃掉 59% 参数。第 08 章把这个 block 堆成完整 nanoGPT 训练。"
topics:
  - "深度学习"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

第 06 章的单头 attention 能跑、梯度也对，但你不能把 50 层 attention 直接摞起来就指望它训练——它会原地爆炸或者底层完全不学。Transformer block 不是一个新算子，它是一组**接线决策**：在 attention 和 FFN 周围怎么接残差、LayerNorm 放在加法的哪一边。这组接线几乎承担了"深层网络能不能训"的全部重量。本章不讲"它能 work"，本章用配套代码量出每个接线选择坏掉时具体坏成什么样。

配套代码在 `examples/tiny-dl-from-scratch/src/stage07-transformer-block.ts`，一个 24 维、2 层的 mini-GPT,单序列处理(`batch=1`)。下面所有数字都是这个文件 `npx tsx` 跑出来的真实输出，toy 规模,绝对值偏乐观,可迁移的是相对趋势——我会逐处标注。

## block 不是新积木，是一种接线

先看完整骨架。一个 mini-GPT 的数据流是 token embedding + position embedding → N 个 block → 最终 LayerNorm → 输出投影到 vocab logits。代码里 `MiniGPT.forwardIds` 就是这条线（`stage07-transformer-block.ts:227`）：

```ts
forwardIds(ids: Int32Array): Tensor {
  const T = ids.length;
  let h = this.tokEmb.forward(new Tensor(Float64Array.from(ids), [T])); // (T, d)
  if (this.posEmb) {
    const pos = new Float64Array(T);
    for (let i = 0; i < T; i++) pos[i] = i; // positions 0..T-1
    h = h.add(this.posEmb.forward(new Tensor(pos, [T]))); // 注入可学习位置信号
  }
  for (const b of this.blocks) h = b.forward(h);
  h = this.lnFinal.forward(h);
  return this.head.forward(h); // (T, vocab)
}
```

注意这里没有任何新的可微算子。token embedding(`Embedding`,token id → 向量的查表)、`Linear`、`LayerNorm`、attention 用的 matmul/softmax——全是前六章建好并 gradCheck 过的东西。block 的全部价值在 `b.forward(h)` 里那几行加法和归一化的**摆放顺序**。这点很重要:它意味着如果你前六章的算子都对,block 几乎不可能引入新的数值 bug,唯一会坏的是接线逻辑本身。

stage07 第一步就验证了这点。整网做一次 gradCheck(解析梯度 vs 数值梯度对拍),29 个张量、CE loss 3.9149,做 eps 扫描:

```
eps=1e-2  maxRelError=1.000e+0   (checked 87)
eps=1e-3  maxRelError=2.326e-2   (checked 87)
eps=1e-4  maxRelError=6.521e-4   (checked 87)
eps=1e-5  maxRelError=2.220e-3   (checked 87)
eps=1e-6  maxRelError=2.220e-2   (checked 87)
U 形曲线最低点 maxRelError: 6.521e-4   PASS
```

这里判据不是单算子那种 1e-6 绝对容差,而是 **U 形深谷**(随 eps 减小误差先降后升的形状)。深栈 softmax+LN+matmul 叠在一起,有限差分(用差商近似导数)的曲率天然更大,1e-4 已经是这种深度能达到的可达精度。一个**坏掉的 adjoint** 会在所有 eps 下都停在 O(1)——所以最低点比 1.0 低 3 个数量级,就证明接线没破坏任何梯度通路。如果你自己接 block 时漏了一个 `.add`,这个测试会立刻在 maxRelError 上炸出来,不用等训练发散才发现。

## 残差不是锦上添花，是梯度高速公路

这是本章最关键的判断,我要直接说清楚:**残差(skip connection,跳过子层把输入直接加到输出)不是性能优化,它是深 transformer 能训练的根因。** 它把每个子层从"学一个完整映射"变成"学一个增量"——LayerNorm 之间的那段从 `x → f(x)` 变成 `x → x + f(x)`。这个改写让 100 层网络的梯度幅度跨层基本恒定,这才是深度能 scale 的物理原因。

为什么?反向传播算梯度是连乘。`d/dx[f(x)]` 在纯子层里是 `f'(x)`,tanh/sigmoid 这类的局部导数 `< 1`,逐层连乘就指数衰减,底层(靠近输入)收到的信号被压到几乎为零——这就是经典的梯度消失。残差把它变成 `d/dx[x + f(x)] = 1 + f'(x)`,那个 `1` 是 identity 直通项,连乘里永远有一条 `1×1×1×...` 的路把梯度原样送到底层。

stage07 把这件事量出来了。这里有个诚实的方法论选择值得讲:**不能在 mini-GPT 的 block 上测残差效果**,因为 block 里有 LayerNorm,它会重新归一化每个子层输出、把梯度流"救回来",即使没残差也看不到干净的消失。为了隔离残差本身,stage07 用 12 层纯 tanh 子层(无 LayerNorm)对照测梯度范数(`stage07-transformer-block.ts:385` 起):

```
layer (0=底/近输入) | grad-norm(有残差) | grad-norm(无残差)
  layer  0          | 2.448e+0      | 1.861e-1
  layer  3          | 1.105e+0      | 2.208e-1
  layer  6          | 6.938e-1      | 2.160e-1
  layer  9          | 5.263e-1      | 2.265e-1
  layer 11          | 3.906e-1      | 2.600e-1
底层 (layer 0) grad-norm: 有残差 2.448e+0 vs 无残差 1.861e-1
底/顶 grad-norm 比值 (>1 表示底层信号未被衰减): 有残差 6.27x   无残差 0.72x
```

读这两列。**无残差**那列底层 grad-norm 0.186,顶层 0.260,底/顶比值 0.72——底层信号比顶层还弱,梯度被 tanh 连乘压在底部。**有残差**那列底层反而最强(2.448),底/顶比值 6.27,梯度一路畅通到最底层。底层信号有残差是无残差的约 13 倍(2.448 / 0.186)。

把这个机制刻进直觉:**没有残差,你加的层越深,底层越学不动,深度变成负资产**。这就是 2015 年 ResNet 之前 CNN 卡在 20 几层、之后能上千层的原因,也是 transformer 敢叠几十上百层的前提。注意一个细节:有残差那列从底到顶是缓慢**递减**的(2.448 → 0.391),这是正常的——梯度从 loss 端(顶)往输入端(底)累积,identity 通路保证它不衰减到零,但每经过一个残差块的 `f'(x)` 项仍会有正常的衰减叠加,这跟"消失"是两回事。深度越大,有无残差的差距越夸张。

## position embedding：忘了它，模型对顺序失明

attention 有一个反直觉的性质:它本身**对位置无感**。`attn = softmax(QK^T)·V` 这套运算是 permutation-equivariant 的——你把输入序列的 token 顺序打乱,输出只是跟着同样打乱,attention 分不出"我"在第 3 位还是第 7 位。对语言模型这是致命的:"狗咬人"和"人咬狗"在没有位置信号时是同一个东西。

所以 mini-GPT 在 token embedding 上**加**了一个可学习的 position embedding(`posEmb`,每个位置 0..T-1 一个独立向量)。stage07 第 4 个实验做消融:建一个 `usePos: false` 的模型,喂同一组 token 的两种顺序,看池化后(对序列维度求和,这个操作本身与顺序无关)的表示差异:

```
原顺序 ids:    [3, 1, 4, 1, 5, 9, 2, 6]
打乱同集合:    [6, 2, 9, 5, 1, 4, 1, 3]
无位置时, 顺序无关的池化表示最大差异: 1.388e-17 (期望 ~0)
位置不变性确认 (diff < 1e-12): true
```

`1.388e-17` 就是浮点舍入噪声,实质是 0——两种顺序在无位置信号下产生**比特级相同**的集合表示。模型完全失明。这个测试故意用顺序无关的池化做 witness(见证),是因为带 causal mask(因果掩码,位置 i 只能看 0..i)时顺序本身会改变"谁能看见谁",会污染纯位置消融的结论;池化绕开了这个干扰,把"无位置 → 顺序不可见"这条因果钉死。

实战里这个 bug 很阴险:你忘了加 position embedding,模型**照样训练、loss 照样下降**(它还能从 token 共现学到很多),只是永远学不会任何依赖语序的东西,你可能要到评估时才发现它分不清主谓宾。加上 position embedding 后,相同 token 在不同位置注入不同向量,顺序差异立刻非零,模型才"睁眼"。

## FFN 的 4x 升维：表达力都在这里

block 里除了 attention,还有一个 position-wise FFN(逐位置前馈,对每个 token 独立做两层 MLP)。它的标准接法是 `d_model → 4*d_model → d_model`,中间夹 ReLU。那个 **4x** 不是随便定的——它是 block 表达力的主要来源,也是参数大头。

为什么是升维?attention 负责"在序列里搬运和混合信息",但它每个位置的变换是线性的(QKV 投影 + 加权和);真正的非线性"思考"发生在 FFN。中间那层升到 4 倍宽,给了网络一个高维空间去做特征组合,再压回来。窄的 FFN(比如 1x)会成为信息瓶颈,表达力被掐死。

stage07 第 1 个实验把参数账算给你看:

```
总参数: 16032
  embedding (token+pos): 960   (6.0%)
  attention (qkv+proj) : 4800  (29.9%)
  FFN (4x 扩展)         : 9456  (59.0%)
  norm + head + 其它    : 816   (5.1%)
```

FFN 吃掉 **59%** 的参数,接近 attention(29.9%)的两倍。这里有个 toy 偏差要标注:这个尺寸下 embedding 占比(6%)偏高,因为 vocab 维度是固定开销,真实 GPT vocab 几万、d_model 几千后 embedding 占比会被稀释。**可迁移的不是这些绝对百分比,是 FFN > attention 这个相对关系**——在真实 GPT 里这个关系依然成立,大模型的参数主体也是 FFN。这也解释了为什么很多 LLM 推理优化(以及 MoE,把 FFN 拆成多个专家按需激活)都盯着 FFN 下手:它是参数和算力的最大块。

代码里 FFN 还有个初始化细节(`stage07-transformer-block.ts:122`):两层 Linear 用 **kaiming** 初始化而非 xavier,因为中间夹的是 ReLU。ReLU 砍掉一半激活,xavier 的方差目标会让隐藏激活逐层衰减;kaiming 专门补偿了这个 factor 2。而 attention 的投影喂的是 softmax/linear 路径,用 xavier。初始化选错不会立刻崩,但会让训练更慢、更不稳——这是那种"能跑但跑不好"的隐性坑。

## ⚡ Pre-LN vs Post-LN：现代 LLM 默认的由来

这是本章的前沿钩子。LayerNorm(层归一化,把每个 token 的向量归一到零均值单位方差)放在残差加法的哪一边,有两种接法,它们的早期训练稳定性差别很大——**而"放哪边"目前没有一个所有场景通用的最优解,仍是活跃研究方向**。

两种接法(`stage07-transformer-block.ts:152`):

```ts
// Pre-LN（现代默认）：LN 在残差分支内部，跳连保留未归一化的原始信号
//   x = x + Attn(LN(x));  x = x + FFN(LN(x))

// Post-LN（原始 2017 接法）：LN 在加法之后，残差高速公路每块都穿过一次归一化
//   x = LN(x + Attn(x));  x = LN(x + FFN(x))
```

差别在于残差直通路上有没有 LayerNorm。Pre-LN 把 LN 塞进残差**分支**,跳连那条 identity 路径保留未归一化的信号,梯度高速公路是干净的;Post-LN 让残差路径**每一块都穿过一次归一化**,早期参数还没稳定时,这层归一化会放大扰动。

stage07 用同 seed、同输入、故意偏高的 LR=0.05(不做 warmup,为了逼出 Post-LN 的敏感性)跑前 50 step:

```
step  |  Pre-LN loss  |  Post-LN loss
    0 |    3.9149    |    3.5087
   10 |    1.8721    |    1.4775
   20 |    0.8682    |    0.5253
   49 |    0.1215    |    0.0799
前 10 step 最大单步 loss 上升 (越大越不稳):  Pre-LN 0.0000   Post-LN 0.5718
```

看最后一行的稳定性指标:前 10 step 内**最大单步 loss 上升**。Pre-LN 是 **0.0000**——loss 单调下降,一步都没回弹;Post-LN 是 **0.5718**——某一步 loss 反而涨了大半,这就是早期不稳的信号(高 LR 下梯度把参数推过头)。

诚实标注:这个 toy 里 Post-LN 末 loss(0.0799)反而比 Pre-LN(0.1215)低,因为单序列玩具数据太容易记住、绝对值偏乐观。**可迁移的不是谁末 loss 低,是 Pre-LN 早期更平稳这个相对趋势**。在真实大模型上,Post-LN 不做 LR warmup 经常直接早期发散,这就是为什么 GPT-2 之后几乎所有主流 LLM 都默认 Pre-LN——你用 Pre-LN 可以无脑上较大 LR、省掉精细的 warmup 调度。

⚡ 但这件事**没有定论**。Post-LN 训得起来的话,最终性能往往略好(它的归一化更"强"),所以后来有 DeepNorm、ResiDual 这类工作专门去修 Post-LN 的早期发散,想两头通吃;也有 normalization-free(完全去掉 LN,靠初始化和缩放)的探索路线。"LN 放哪、要不要 LN、怎么 scale 残差分支"在超深(几百到上千层)网络上仍是 open question。本章实测的 Pre-LN 早期更稳,只是给了你这个工业默认背后最直观的一个证据。

## 失败模式速查

把本章的坑收成一张表,接你自己的 block 时对照:

- **去掉残差** → 深层底部梯度被连乘压到接近零(实测无残差底层只有有残差的 ~0.076 倍),底层不学,深度变负资产。修法:每个子层都包 `x + f(x)`。
- **忘了 position embedding** → 模型对语序失明(实测顺序差异 1e-17),loss 照降但学不会任何依赖顺序的东西。修法:token embedding 上加 position embedding。
- **Post-LN 不做 warmup + 高 LR** → 早期发散或剧烈抖动(实测单步 loss 回弹 0.57)。修法:换 Pre-LN,或 Post-LN 配 LR warmup。
- **FFN 不升维 / 升维不够** → 信息瓶颈,表达力被掐死。修法:守住 4x(或按预算调,但别低于 ~2x)。
- **FFN 用了 xavier 而非 kaiming** → ReLU 后隐藏激活逐层衰减,训练慢且不稳。修法:ReLU 路径用 kaiming。

## 小结与下一章

这一章的核心判断只有一句:**Transformer block 的难点不在算子,在接线**。残差、LayerNorm 位置、position embedding、FFN 升维——这四个接线决策每一个都能用一行代码改对或改错,改错了不会立刻报错,而是让模型"能跑但学不好"或者"深了就训不动"。stage07 的价值就是把每个错误的代价量成数字,让你接线前心里有数。

第 08 章把这个 block 堆成完整的 nanoGPT,接上第 04 章的优化器和第 05 章的训练循环,在真实(虽然仍小)的语料上从头训练,看它从随机权重一路把 loss 降下来、最后能生成像样的文本。本章 gradCheck 过的这个 block,就是第 08 章训练的那个 block——前面六章的每个算子、这一章的每根接线,到那时会第一次作为一个整体跑起来。
