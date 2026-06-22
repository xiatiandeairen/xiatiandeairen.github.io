---
title: "top-k 路由:门控网络如何只点亮 k 个专家"
slug: "02"
collection: "moe"
order: 2
summary: "第 01 章把一个 FFN 拆成 E 个同质专家,但全选——每个 token 跑遍所有专家,没省一分算力。本章引入门控网络(一个 Linear → softmax 给出 E 维分布),实现 top-k 选择:每个 token 只送进得分最高的 k 个专家,输出按门控权重加权求和。实测激活专家数从 8 降到 1/2 而准确率不掉、loss 只付一点点溢价——这就是稀疏激活省算力的来源。本章刻意回避 top-k 的『取最大』不可微问题:梯度怎么回到门控?留给第 03 章。"
topics:
  - "混合专家"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

第 01 章的 MoE 层是个对照组:E=8 个专家,每个 token 跑遍全部 8 个,然后把 8 份输出加权求和。它确立了「多专家 = 容量变大」,但一分算力都没省——激活量就是 dense 的 100%。那其实不是稀疏 MoE,是「带学习权重的 ensemble(集成,多模型加权投票)」,算力跟一个 E 倍宽的稠密 FFN(前馈网络)一模一样。这一章只改一件事,而这件事是整个 MoE 价值主张的全部:**让每个 token 不再跑 8 个专家,只跑得分最高的 k 个**(k 通常是 1 或 2)。改完之后,实测算力降到约 k/E,而准确率不动、loss 只付极小溢价。下面把这个 juxtaposition(并置对比)用配套仓 `examples/moe-from-scratch/src/stage02-topk-routing.ts` 的真实运行数据钉死,再讲两个最容易栽进去的坑。

引用的所有数字来自 `npx tsx src/stage02-topk-routing.ts` 的真实输出。数据是 6 个高斯簇的 toy 任务(可分),所以三种 k 配置准确率都钉在 100%——绝对值偏乐观,但**可迁移的是两条比例:sparse 与 dense 的 loss 溢价、激活算力 ≈ k/E**。凡 toy 上的乐观处我都标注。

## 门控网络:一个 Linear 加一个 softmax,就这么朴素

先把「门控」(gating,决定每个 token 走哪个专家的小网络)祛魅。它不是什么独立的决策大脑,就是**一个把输入投影到 E 维、再过 softmax 的线性层**。`TopKMoELayer` 的构造函数里它和专家平起平坐:

```ts
this.gate = new Linear(DIM, NUM_EXPERTS, rngSrc);
this.experts = [];
for (let e = 0; e < NUM_EXPERTS; e++) {
  this.experts.push(new Expert(DIM, HIDDEN, NUM_CLASSES, rngSrc));
}
```

forward 第一步就是把 token 喂进 gate 拿到一个长度 E 的概率分布:

```ts
const gateProbs = this.gate.forward(x).softmaxRow(); // [1,E], differentiable gate
```

`gateProbs` 是个 8 维向量,每个分量是「这个 token 该有多少比例交给第 e 个专家」。注意它是**可微的**(`Value` 节点,带梯度)——这一点后面救命,先记住。

为什么门控这么简单还能工作?因为它**不需要理解任务**,只需要学会把不同的 token 推向不同的专家。复杂的门控(多层 MLP、注意力门控)在大规模实验里收益微薄甚至有害——门控越复杂,它越容易过拟合到「永远选同几个专家」,反而加剧后面要讲的负载不均。工业主力(Switch Transformer、Mixtral)的门控都是这种单层线性门控。第 01 章我们花一整章证明 autograd 引擎正确(实测 gradCheck 误差 ~1e-10),就是为了能在这里放心地让梯度从 `gateProbs` 自动流回去而不必手写反向。

## top-k 选择:argmax 挑 k 个,然后做一件不能忘的事

拿到 8 维分布后,top-k 就是「排序取前 k 个」。代码直白:

```ts
const idx = Array.from({ length: NUM_EXPERTS }, (_, e) => e);
idx.sort((a, b) => gateProbs.data[b] - gateProbs.data[a]);
const chosen = idx.slice(0, this.k);
```

这里有个容易被忽略的设计细节:排序读的是 `gateProbs.data`(数值快照),**只用来决定跑哪几个专家**;真正承载梯度的是后面 `gather` 取出的 `Value`。换句话说,「选谁」这个离散决策本身是从数值里数出来的、不可导,但「选中之后乘多少权重」走的是可微路径。这道裂缝——选择不可导、权重可导——就是第 03 章整章要处理的命门,本章先按住不展开。

选完 k 个之后,有一步**最容易忘、忘了不报错但慢性中毒**的操作:把这 k 个被选中的门控权重**重新归一化**,让它们重新加和为 1。代码:

```ts
let selMass = 0;
for (const e of chosen) selMass += gateProbs.data[e];

let combined = Value.zeros(1, NUM_CLASSES);
for (const e of chosen) {
  const w = gateProbs.gather(e);                              // 标量 Value, 梯度回流门控
  const weight = this.renormalize ? w.div(Value.scalar(selMass)) : w;
  const expertLogits = this.experts[e].forward(x);            // 只有 k 个专家真正前向
  combined = combined.add(expertLogits.mul(weight));
}
```

`selMass`(被选中权重之和,selected mass)是关键变量。softmax 给出的 8 个权重本来加和为 1;你只取前 k 个,就丢掉了 `1 - selMass` 那部分质量。`renormalize` 打开时每个权重除以 `selMass`,把幸存者重新拉回「partition of unity」(单位分拆,即权重和为 1 的凸组合)。这件事的代价就是失败模式 A,后面专门拆。

还要注意 `this.experts[e].forward(x)` 在循环里——**只有被选中的 k 个专家真的执行前向**。这不是细节,是 MoE 省算力的物理来源。如果你图省事跑遍 8 个专家再把没选中的乘 0 屏蔽,算力一分不省,整章白做。

## 实测:k=1 和 dense 同样准,只用 15.6% 的算力

把 k=1、k=2、k=8(后者每个 token 跑全部 8 个,就是 dense 基线)在**同一份数据、同一初始化种子**下各训 250 步,只让 k 变。真实输出:

```text
k            finalLoss   acc      avgActivated   FLOPs(MAC)    vs dense
k=1          0.0149      100.0%    1.00              48,000    15.6%
k=2          0.0080      100.0%    2.00              86,400    28.1%
k=8 (dense)  0.0039      100.0%    8.00             316,800    103.1%
```

这张表是 MoE 的整个 thesis(核心论点),浓缩成三行:

- **准确率三者全是 100.0%**——稀疏没掉质量。这是 toy 数据可分带来的,绝对值乐观,但它真实地说明了「sparse == dense quality」这件事至少在可分任务上成立。
- **loss 溢价只有 0.0110**(k=1 的 0.0149 减 dense 的 0.0039)。稀疏付了代价,但极小。
- **算力是真省**:k=1 只用 15.6% 的 FLOPs。这个比例约等于 k/E = 1/8 = 0.125。

为什么是 15.6% 而不是干净的 12.5%?因为 FLOP 里除了 k 个专家,还有门控本身的 `in×E` 开销——每个 token 不管选几个专家,都得先过一遍 gate。看 `src/core/metrics.ts` 的 `activatedFLOPs`:

```ts
const gate = inDim * E;                              // 门控 logits, 每 token 必付
const oneExpert = inDim * hidden + hidden * outDim;  // 一个专家两次 matmul
return tokens * (gate + k * oneExpert);
```

门控那项 `inDim * E` 是固定开销,在 k 很小时占比相对变大,所以实测比值略高于纯 k/E。这是个诚实的细节:**k/E 是理想上限,真实激活比会因为门控、负载不均的 padding 等略高**。诚实声明:这里的 FLOP 是从矩阵维度数出来的真实 MAC(乘加)数,不是 wall-clock(墙钟时间)。toy 规模下 wall-clock 被 JS 解释器开销主导,会误导,所以只报算法比例。

✦ **专家配方**:这张表解释了工业界的选择。**Switch Transformer 用 top-1**(k=1,最省),**Mixtral 8×7B 用 top-2**。k 越小越省算力,但路由噪声越大、越容易坍塌(collapse,某几个专家被冷落到训不动,第 06 章解析)——因为 k=1 时一个 token 的命运完全压在单个 argmax 决策上,门控稍微抖一下就换专家,梯度信号更稀疏。top-2 用一点算力换路由稳定性,是个常见的折中点。

第二组输出验证了「avgActivated 真的等于 k」——这个数不是读配置读出来的,是训练完跑一遍、把每个 token 真实跑了几个专家**数出来**的:

```text
   k=1: avgActivated=1.00 ✓
   k=2: avgActivated=2.00 ✓
   k=8: avgActivated=8.00 ✓
```

为什么要数出来而不信配置?因为 MoE 实现里「配置说激活 k 个」和「实际激活了几个」常常对不上——比如容量溢出后某些 token 被 drop(第 05 章),实际激活就不足 k。养成「测量而非断言」的习惯,这本书每个 stage 都这么干。

## 门控权重的第二重身份:它还是缩放因子(梯度的逃生通道)

现在讲本章最该带走的一个直觉,它直接接到第 03 章。

回看 combine 那一步:`combined += expertLogits * weight`。`weight` 是门控给被选中专家的那个连续权重。它在这里有**两重身份**:

1. **选谁**:谁的权重高,谁被 top-k 选中(离散,不可导)。
2. **缩放因子**:被选中之后,它的连续值还决定这个专家的输出乘多少(连续,可导)。

第二重身份是梯度的逃生通道。`argmax` 那个「选谁」的决策确实是堵墙,梯度过不去。但 `weight` 作为乘法因子留在了前向图里——loss 对 `weight` 是有梯度的(`d(loss)/d(weight) = d(loss)/d(combined) · expertLogits`),这个梯度顺着 `gather` 流回 `gateProbs`,再流回 gate 的线性层参数。**门控不是靠「我选对了专家」学习的,是靠「我给对了被选专家多少权重」学习的。** 这就是为什么单层线性门控配 top-k 能训起来——梯度从权重这条侧门溜回去,绕开了 argmax 那堵墙。第 03 章会把这条侧门完整拆开,这里你只需建立这个直觉:✦ softmax 权重不只是「选谁」的依据,它的连续值充当缩放因子,让梯度有路可走。

## 失败模式 A:忘记重新归一化 → 输出尺度漂移

这是 top-k 最容易漏、漏了还能跑、跑着跑着出怪事的坑。

回到 `selMass`。你选了 k 个专家,它们的门控权重之和是 `selMass < 1`(因为 softmax 总和是 1,你扔掉了一部分)。如果**不**重新归一化,combine 出来的输出整体被缩放了 `selMass` 倍——而且这个缩放是 **k 越小越严重**的。在未训练的初始层上直接测「不归一化 / 归一化」的输出范数比:

```text
   k=1: 平均 selMass=0.839  ‖out‖_OFF/‖out‖_ON=0.837
   k=2: 平均 selMass=0.972  ‖out‖_OFF/‖out‖_ON=0.980
   k=8: 平均 selMass=1.000  ‖out‖_OFF/‖out‖_ON=1.000  (k=E: selMass→1, 无漂移)
```

数据说得很干净:输出范数比几乎精确等于 `selMass`。k=1 时输出只有应有尺度的 84%;k=8(全选)时 `selMass→1`,没有漂移——因为全选时 softmax 权重一个没丢,本来就加和为 1。**漂移程度和 k 耦合**,这是最阴的地方:你在 k=2 上调好的超参,换到 k=1 尺度就变了。

现在讲一个诱人但错误的故事,以及它为什么错。你可能以为「不归一化会让训练炸掉(loss 变 NaN)」——**不会**。实测不归一化训练 k=1 照样收敛:

```text
   注: renorm OFF 训练 k=1 仍能收敛 (final loss=0.0128, acc=100.0%) — Adam 会学出更大 logits 吸收这个常数尺度。
```

为什么?因为 `selMass` 对每个 token 是个差不多的常数缩放,Adam(自适应优化器)会让专家学出更大的 logits 把这个常数尺度吸收掉。所以**这不是一个会自爆的 bug,是一个慢性中毒的 bug**——它真实有害的地方在别处:

- combine 不再是 convex combination(凸组合),权重和 ≠ 1,「加权平均」的语义破了。
- 输出尺度和 k 耦合,在真实 transformer 里 MoE 层有 residual(残差连接)和 LayerNorm(层归一化)在下游,它们假设输入尺度稳定。尺度被 `selMass` 悄悄改变会破坏这些假设,在更深的网络里以你想不到的方式表现出来。
- 它悄悄改变了有效门控信号——你以为门控学的是 A,实际学的是 A 乘了个尺度。

教训一句话:**被选专家的门控权重必须 renormalize 成 partition of unity**。这个 bug 不会给你一个干脆的崩溃让你 debug,它给你一个「能跑但说不清哪里不对」的模型。在 toy 上看不出,在生产规模上是几天 debug。

## 失败模式 B:k=0 是空选择,一个伪装成模型的 no-op 层

第二个坑更直接。k=0 意味着不选任何专家:combine 的分子是空的,归一化的分母 `selMass` 是 0,要么除零、要么输出恒为零。这不是「更省的路由」,是**一个把自己伪装成『死掉的模型』的 no-op 层**——你会看到 loss 不动,然后去 debug 数据、debug 学习率,根本想不到是 k 配错了。

处方是在构造时就拒绝,响亮地报错而不是悄悄返回零:

```ts
if (k < 1 || k > NUM_EXPERTS) {
  throw new Error(`TopKMoELayer: k must be in [1,${NUM_EXPERTS}], got ${k}`);
}
```

实测它确实拦住了两端:

```text
   ✓ 构造时拒绝 k=0: TopKMoELayer: k must be in [1,8], got 0
   ✓ 构造时拒绝 k>E: TopKMoELayer: k must be in [1,8], got 9
```

这是个小检查,但体现一条原则:**让非法配置在构造时炸,而不是在训练第 200 步用一个安静的零输出骗你。** k>E(选的比有的还多)同理拦掉。

## 顺手暴露:没有均衡损失时,专家利用率已经歪了

最后看一组本章故意不修、只诚实暴露的数据。同样的 k=2 训练,数一数 8 个专家各自被多少 token 当成 top-1 选中:

```text
E0 │██████······················│ 0.0700
E1 │██████████████████··········│ 0.1950
E2 │██████████████··············│ 0.1500
E3 │████████····················│ 0.0850
E4 │████████████████████████████│ 0.3050
E5 │██████······················│ 0.0650
E6 │████························│ 0.0450
E7 │████████····················│ 0.0850
```

8 个专家都被选中过(没有完全饿死的),但占比从最忙的 E4(30.5%)到最闲的 E6(4.5%),**最忙/最闲 ≈ 6.8 倍**。这是**没加任何负载均衡损失**时的常态:门控没有任何压力去均匀使用专家,它只要 loss 降就行,自然会偏爱几个早期表现好的专家,形成「富者愈富」的马太效应。第 04 章加 auxiliary loss(辅助均衡损失)把这条曲线压平,第 06 章解析当这个不均衡恶化到极端时的专家坍塌。这里先让你看清:**稀疏路由开箱即来的副作用就是负载不均,它不是 bug,是默认行为,必须主动对治。**

## ⚡ 前沿:多专家细粒度 + 低激活比,以及它的开放问题

本章的 top-k 代码,内核就是当前 SOTA 配方的雏形。**DeepSeek-V3 用的是细粒度路由(fine-grained routing):把专家切得更细,256 个专家里选 8 个,激活比低至约 3%。** 对比一下本章 E=8 选 2(激活比 25%),DeepSeek 把同样的「选 k 个」推到了极致——专家更多、更小,每个 token 激活的比例更低,从而在同等激活算力下塞进更大的总参数量。再加上若干「共享专家」(shared experts,每个 token 都过的固定专家)兜底通用能力。「多专家细粒度 + 低激活比」是 2024–2025 年开源 MoE 的主流方向,而它的算法内核,就是你这章写的这几十行 top-k。

⚡ 但这里有个**目前没有通用解、仍在研究**的问题:**k 和 E 到底怎么选,没有可迁移的理论,基本靠经验和 scaling law 实验试出来。** 给定算力预算,你应该用「少而大的专家、高激活比」(如 8 选 2)还是「多而小的专家、低激活比」(如 256 选 8)?细粒度看起来更优,但专家切太细会让单个专家容量不足、路由噪声放大、通信开销(专家分布在不同设备时的 all-to-all 通信)爆炸。不同团队给出的最优 (k, E) 配置差异很大(Mixtral 的 8/2 和 DeepSeek 的 256/8 是两个极端),且高度依赖模型规模、数据、硬件拓扑。学界有一些 scaling law 尝试刻画「专家粒度」这个维度,但还没有像 Chinchilla 之于 dense 模型那样被广泛接受、能直接外推的公式。换句话说:**本章给了你 top-k 的引擎,但「拨到几档」目前还是一门手艺,不是科学。**

## 本章小结

1. **门控 = 一个 Linear + softmax**,给出 E 维可微分布;不需要复杂,复杂反而加剧不均(`src/stage02-topk-routing.ts::TopKMoELayer`)。
2. **top-k = 排序取前 k + 只跑这 k 个专家**;实测 k=1 与 dense 同准(100%),loss 溢价仅 0.0110,算力降到 15.6% ≈ k/E。激活数要数出来验证,不信配置。
3. **门控权重的双重身份(✦)**:既是「选谁」的离散依据,又是被选专家输出的连续缩放因子——后者是梯度回流门控的侧门,接第 03 章。
4. **失败模式 A**:忘记 renormalize → 输出尺度与 k 耦合漂移(实测范数比 ≈ selMass),不会自爆但慢性破坏下游 residual/LayerNorm。**B**:k=0/k>E 是 no-op,必须构造时拒绝。
5. **稀疏的默认副作用是负载不均**(实测最忙/最闲 6.8×),本章只暴露不治,留给第 04/06 章。
6. **⚡ (k, E) 怎么选目前无通用解**:细粒度低激活比(DeepSeek 256 选 8)是当前 SOTA 方向,但最优粒度仍靠经验。

留给第 03 章的那堵墙:`argmax` 选专家是离散的,不可导。本章靠「权重作缩放因子」的侧门让梯度溜了回去,但这个机制为什么成立、它和 Gumbel-softmax/REINFORCE 这些「正经」可微化技术的关系、工业界为什么偏偏选了这条朴素的侧门——下一章正面拆开。
