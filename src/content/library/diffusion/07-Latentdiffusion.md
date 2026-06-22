---
title: "Latent diffusion:为什么不在像素上扩散,在哪里省下算力"
slug: "07"
collection: "diffusion"
order: 7
summary: "承接前六章在 2D 数据空间跑通的完整 DDPM,本章回答工业级 diffusion 的关键一跃:Stable Diffusion 为什么不在 512×512 像素上直接扩散,而是先用自编码器把图压成小得多的 latent、在 latent 上跑 diffusion、再解码回像素。核心是『感知压缩与生成解耦』——autoencoder 砍掉人眼无关的高频冗余,diffusion 只在语义紧凑的低维空间工作。本书在 2D toy 上做最小复刻:训一个 2D→1D 的小自编码器,在 1D latent 上跑第 04 章的训练管线 + 第 05 章的采样器,再解码回 2D,把前向(01)、反向(02)、score(03)、训练(04)、采样(05)、guidance(06)收口进一个降维管线。"
topics:
  - "生成模型"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

前六章我们在原始 2D 数据空间(双月、混合高斯,本章换成 swiss roll)上把 DDPM 从头跑通了:第 01 章往数据里加噪、第 02 章学反向去噪、第 03 章把它认成 score、第 04 章训练 ε-预测网络、第 05 章换采样器、第 06 章加 guidance。整条链路有一个从没被质疑的前提——**diffusion 直接在你要生成的那个空间里跑**。2D 数据,denoiser 就吃 2D;真要换成 512×512×3 的图,这个 denoiser 每一步就要在 786432 维上做一次完整前向。Stable Diffusion 的第一条工程决策就是把这个前提推翻:**不在像素上扩散**。它先用一个自编码器(autoencoder,把数据压缩再还原的网络)把 512×512 压成 64×64 的 latent,在这个小 64 倍的空间里跑 diffusion,采样完再解码回像素。

这一章用本书的 toy 把这个决策做最小复刻,代码在 `examples/diffusion-from-scratch/src/stage07-latent.ts`。它复用了 `core/nn.ts` 的 `MLP` / `SinusoidalEmbedding`、`core/schedule.ts` 的 cosine schedule、`core/optim.ts` 的 Adam——和前几章是同一套零件,只是把 diffusion 搬进了一个降维后的空间。下面所有数字都来自 `npx tsx src/stage07-latent.ts` 在 seed 1337 下的真实运行,确定性的(参数量、维度、Chamfer 距离)我直接引用,toy 性质导致的乐观偏差我会单独标。

## 为什么不直接在像素上扩散:算力账

先把经济账算清楚,这是整个 latent diffusion 的出发点。DDPM 采样的主导成本是 **NFE × 每步 denoiser 的成本**(NFE = number of function evaluations,采样要调用 denoiser 几次)。第 05 章我们已经知道 NFE 是采样的硬成本中心,DDIM 把它从 1000 砍到 50 是那一章的主线。而每步 denoiser 成本里,**输入维度直接决定第一层 matmul 的宽度**。在像素上扩散,这个维度是百万级;在 64× 下采样的 latent 上,是它的 1/48 左右(8× 下采样在 H、W 两个方向各砍 8 倍,再算上通道数变化,真实 SD 的 VAE 把 512×512×3 压到 64×64×4,latent 元素数约为像素的 1/48)。

关键洞察是 Rombach et al. 2022(就是 Stable Diffusion 内核那篇)的核心论点:**真实图像绝大部分是人眼无关的高频细节**。直接在像素上训 diffusion,网络容量和采样的 NFE×dim 成本几乎全花在建模没人在乎的噪声纹理上。把这部分先用一个 autoencoder 一次性砍掉,diffusion 就只在语义紧凑的低维空间工作。

本书 toy 上无法复刻百万维,但能复刻这个**形状**。stage07 的 head-to-head 表(section 4 实测输出)是这样的:

```
metric                            direct 2-D DDPM   latent-DDPM
----------------------------------------------------------------
denoiser params                   5506              1409  (3.91x smaller)
+ one-time autoencoder params     0                 195  (paid ONCE, not per sample)
sampler NFE                       100               100
sampler compute (NFE × dim)       200               100  (2.00x less)
Chamfer to data (lower=better)    0.0468            0.2134
```

读这张表:latent 路线的 denoiser 小 3.91 倍(5506 → 1409 个可训练标量),每个样本的采样成本(NFE×dim)少 2.00 倍(200 → 100,因为 dim 从 2 降到 1)。代价是 Chamfer 距离(两个点云之间互相找最近邻的平均平方距离,越低越像)从 0.0468 涨到 0.2134。

这里必须诚实标注一笔 toy 偏差:**我们的压缩只是 2D→1D(维度 2×)**。真实 LDM 是 8× 下采样、约 48× 更少的 latent 元素,那里同样的架构买到的是一个数量级的成本降低,而不是上面这个温和的 2 倍。stage07 自己在输出里也把这点喊了出来:

```
HONEST: this is a 2→1 (2×) squeeze; a real LDM uses an 8×-downsampled VAE (~48× fewer latent elements),
where the same architecture buys an order-of-magnitude more, not the modest factors above.
```

把 toy 当成证明"成本能降"是错的;它证明的是**架构决策的方向和 trade-off 的形状**——成本掉、质量掉一点、且(见后文)一个坏的 autoencoder 会封死一切。比例可信,绝对值不可信。

## 同一套 DDPM 数学,只换 dim

latent diffusion 在工程上最舒服的一点:**diffusion 那一半完全不用改**。前向加噪、反向采样的数学是维度无关的,你给它 1D latent 还是 2D 数据,公式一字不差。stage07 把这点直接写进了 `Denoiser` 的设计——一个 `dim` 参数让同一个类同时服务 1D latent 模型和 2D 数据模型:

```ts
class Denoiser {
  constructor(dim: number, embDim: number, hidden: number[], rng: RNG) {
    this.dim = dim;
    this.emb = new SinusoidalEmbedding(embDim);
    this.net = new MLP([dim + embDim, ...hidden, dim], "silu", rng);
  }
  forward(xt: Tensor, t: number): Tensor {
    const te = this.emb.forward(new Array(xt.shape[0]).fill(t)); // [n, embDim]
    return this.net.forward(Tensor.concatCols([xt, te]));
  }
}
```

输入 `[x_t ⊕ time-embed]`、输出 ε̂,这就是第 04 章那个 ε-预测 denoiser 的形状(参第 04 章训练目标 MSE(ε̂, ε))。`trainDdpm` 和 `sampleDdpm` 也都吃 `dim` 参数,没有任何一行针对 latent 的特判。这不是巧合,是 latent diffusion 能成立的前提:**diffusion 不关心它建模的是不是像素**,它只建模一个分布。把数据空间换成 latent 空间,对 diffusion 来说只是换了一份输入。

唯一需要额外操心的,是 latent 不是单位尺度。第 01 章的前向过程假设数据大致是 N(0,I) 尺度(这样 x_T 才约等于纯高斯噪声),而 stage07 实测冻结后的 1D latent 范围是 `[-9.214, 17.714]`——远不是单位尺度。所以在喂给 diffusion 前必须标准化,采样出来后再逆标准化才能解码:

```ts
const { standardized: latentsStd, mean: lMean, std: lStd } = standardize(latents);
// ... train + sample in standardized space ...
const genLatent = unstandardize(genLatentStd, lMean, lStd);
const genFromLatent = ae.decode(new Tensor(genLatent, [N_DATA, 1]));
```

这是一个真实但隐蔽的失败点:**如果忘了用同一组 mean/std 逆标准化,decoder 看到的是离分布的输入,swiss roll 永远不会重现**。这条不变量在真实 LDM 里对应 SD 的 VAE latent 要乘一个 scaling factor(0.18215)再进 diffusion——同一个道理的工业版。

## ✦ 最被低估的设计点:分工,而且分开训

面试问 latent diffusion 最容易暴露理解深度的一个问题:**为什么不端到端一起训 autoencoder 和 diffusion?** 表面看,联合训练让 latent 空间为 diffusion 量身定制,应该更好才对。

答案是**解耦(decoupling)带来的分工**。autoencoder 和 diffusion 负责两件正交的事:

- **autoencoder 负责感知保真**:它的目标是 `dec(enc(x)) ≈ x`,在真实 LDM 里是重建损失 + 感知损失(perceptual loss,用预训练网络的特征比对)+ 轻量对抗损失(让重建更锐利)。它学的是"怎么把像素无损地压进 latent 又还原回来"。
- **diffusion 负责在干净的 latent 上建模分布**:它根本不碰像素,只学"latent 长什么样、怎么从噪声采出一个合法 latent"。

两者**分开训练**:先把 autoencoder 训死、冻结,再在冻结的 latent 上训 diffusion。这么做有两个硬收益:

1. **diffusion 不必浪费容量学像素级细节**。像素到 latent 的还原责任全在 autoencoder 那边;diffusion 拿到的已经是干净、紧凑的语义空间。
2. **同一个 autoencoder 可复用给多个 diffusion 模型**。SD 的 VAE 训一次,文生图、图生图、inpainting 共用同一个。联合训练就把这个复用性焊死了。

stage07 用两个独立的 `MLP`(而不是一个带瓶颈的网络)把"冻结后复用"这件事在代码里说清楚:

```ts
class Autoencoder {
  constructor(latentDim: number, hidden: number, rng: RNG) {
    this.enc = new MLP([2, hidden, latentDim], "silu", rng); // 2 -> latentDim
    this.dec = new MLP([latentDim, hidden, 2], "silu", rng); // latentDim -> 2
  }
}
```

训练流程严格复刻 LDM 的两阶段。先把 autoencoder 训 1500 步,recon MSE 从 0.99731 降到 0.10531(实测,9.5× 更低),解码出来的点云仍能读成 swiss roll。然后**冻结**——把整个数据集编码成 latent 一次,从此 enc/dec 权重再不动:

```ts
const latents = ae.encode(data).data; // 冻结:从这里起 diffusion 只见 latent
```

之后才训那个小 3.91 倍的 latent denoiser(latent ε-MSE 从 1.10546 降到 0.05036)。这个先后顺序不是实现细节,**它就是 LDM 架构本身**。

## ⚡ latent 空间的质量是隐形天花板

stage07 的 section 5 是整章最该让你记住的实验,因为它揭示 latent diffusion 一个反直觉的脆弱点:**autoencoder 的质量是 diffusion 永远无法突破的上限**。

实验设计干净到极致——**唯一变量是 autoencoder 训了多久**。好的版本训 1500 步,坏的版本只训 100 步;之后压在上面的 diffusion 模型完全相同:同 seed、同架构、同训练预算。结果(实测):

```
weak autoencoder recon MSE after 100 steps: 0.50909  (vs 0.10531 for the 1500-step one)
latent-DDPM ON TOP of the weak autoencoder, Chamfer to data = 1.0644
(good autoencoder + same diffusion gave 0.2134 — 5.0x worse)
```

弱 autoencoder 的重建里,swiss roll 的卷曲已经没了——曲线塌成一团。**注意:塌的不是 diffusion 的输出,是 autoencoder 的重建**。结构在 diffusion 还没开始前就已经被 latent 丢掉了。压在上面那个一模一样的好 diffusion 模型,生成的点云 Chamfer 距离 1.0644,比好 autoencoder 上的 0.2134 差 5.0 倍。

为什么必然如此?因为 `dec(latent)` **永远不可能还原出 latent 没编码的结构**。diffusion 再强,它生成的也只是一个"合法的 latent";而合法 latent 经过一个把卷曲丢光了的 decoder,出来的还是一团塌掉的云。autoencoder 是地基,地基塌了,上面的 diffusion 盖得再好也救不回来。stage07 把这句话直接写进了输出:

```
THE LESSON: ... Latent quality is the ceiling: dec(latent) can never show structure the
latent threw away. The autoencoder is the foundation — a great diffusion model on a
collapsed latent still produces a collapsed cloud.
```

这正是 LDM 真实张力的 toy 版:**压缩比和重建质量是对立的**。压得越狠,diffusion 越省,但重建上限越低。SD 的 8× VAE 至今留着可见伪影——文字、人脸、规则纹理这些高频细节,bottleneck 一旦封顶,后面再好的 U-Net 也补不回来。这是 LDM 把"质量交给 autoencoder、生成交给 diffusion"这套分工的必然代价。

**这条「压多狠才不掉质量」目前没有通用解,是当前生成模型架构竞争的主战场。** 几个正在博弈的方向:更高压缩比的 tokenizer(把图压得更狠又不掉重建);连续 latent vs 离散 latent(VQ,把 latent 量化成码本里的离散 token);把 diffusion 直接搬进 Transformer 的 latent(DiT,用 Transformer 替掉 U-Net,扩展性更好);乃至端到端可学的 latent(放弃"先冻结 autoencoder"这个前提)。每个方向都在同一道题上下注——**bottleneck 能压多紧而重建不崩**——而这道题没有公认答案。你在 stage07 的 100 步 vs 1500 步对照里看到的,正是这道题最小、最干净的一个切片。

## 收口:六章在一个降维管线里串起来

把 stage07 拆开看,它其实是前六章的一次合奏。autoencoder 把 2D 折进 1D,然后:第 01 章的前向过程往 latent 加噪、第 02 章的反向过程从噪声里还原 latent、第 03 章告诉我们 denoiser 学的是 score、第 04 章的训练管线在 latent 上训 ε-预测、第 05 章的 ancestral 采样器在 latent 上采样、第 06 章的 guidance(本 toy 未接,但 `Denoiser` 的 time-embedding 接口就是它的挂载点)。换掉的只有一件事:diffusion 工作的空间从数据降到了 latent。

这就是从 toy 到 Stable Diffusion 的那一跃的全部本质——**不是更复杂的 diffusion,而是更聪明的工作空间**。diffusion 的数学一字未改,省下的算力全部来自"在哪里扩散"这个决策。而你为这个决策付的代价,是把生成质量的天花板交给了一个你必须先训死、再也不能动的 autoencoder。这笔 trade-off,是整本书最后想让你刻进直觉的东西。
