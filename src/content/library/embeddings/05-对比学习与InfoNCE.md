---
title: "对比学习与 InfoNCE:负采样的现代化身"
slug: "05"
collection: "embeddings"
order: 5
summary: "第 04 章把 skip-gram 的全 softmax 换成「一个正样本 + k 个负样本」的二分类,这章把那个具体技巧抽象成通用框架:拉近正对、推远负对。InfoNCE 就是 04 负采样损失的连续温度化版本,SimCLR 把它从 NLP 搬到视觉。这章实测对比学习的隐式目标 alignment 与 uniformity 如何拉锯,为第 06 章调温度与负样本数量铺好地基。"
topics:
  - "表示学习"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

第 04 章我们干了一件很具体的事:把 skip-gram 的全词表 softmax 拆成「1 个正样本 + k 个负样本」的二分类,代价从 `O(V)` 降到 `O(k)`。当时那是个为了让 word2vec 跑得动的工程妥协。但如果你退一步看,会发现那个妥协里藏着一个比「加速」大得多的思想——而且这个思想后来几乎重写了整个表示学习领域,从 NLP 走到视觉,又走回通用 embedding。这章就是把那个思想拎出来、讲透,并用一段能跑的代码让你亲眼看到它怎么坏。

先把结论摆这儿,后面逐条拆:

> ✦ **对比学习一句话**:好的表示让「相似的更近、不相似的更远」。InfoNCE 就是第 04 章负采样损失的连续温度化版本——同一思想,从 NLP 走到视觉再回到通用表示。

这不是修辞。SGNS(skip-gram negative sampling)和 InfoNCE 在数学上同源,差别只是「怎么算那个把负样本推开的力」。本章配套代码 `examples/embeddings-from-scratch/src/stage05-infonce.ts` 用一组合成簇把这件事跑出来:32 个 anchor(锚点,就是「要被学表示的样本」)分成 4 簇,每个 anchor 加噪声生成 1 个正样本视图,**批内其余 31 个样本直接当负样本**。训完 80 个 epoch,模型在从没见过簇标签的情况下,自己把同簇的点聚到一起。

## 从「负采样」到「对比」:同一个力,换了说法

回顾第 04 章 SGNS 干的事:给定中心词,正样本是真实上下文词,负样本是按 unigram^0.75 分布采样的随机词。损失是对每个 pair 做一次 logistic 二分类——正 pair 输出趋近 1,负 pair 趋近 0。

对比学习(contrastive learning)把这件事重新表述了一遍,而且表述得更通用:

- 不再区分「中心词 / 上下文词」这种 NLP 专有概念。换成**抽象的 anchor 和它的正样本视图**。anchor 是任意一个样本,正样本是「同一个东西的另一个版本」。
- 在 NLP 里,「同一个东西的另一个版本」是上下文共现;在视觉里(SimCLR),是**同一张图的两次随机增强**(裁剪 + 颜色抖动);在句子 embedding 里,是**同一句话过两次 dropout**。

> ⚡ 这里给术语补一句大白话:**视图(view)= 同一样本的另一个噪声版本**。

配套代码里这个「另一个版本」就是加高斯噪声。看 `stage05-infonce.ts` 的 `augment`:

```ts
function augment(rng: Rng, feat: number[]): number[] {
  return feat.map((x) => x + gaussian(rng, 0, AUG_NOISE));
}
```

`AUG_NOISE = 0.25`,而簇内抖动 `CLUSTER_SPREAD = 0.35`、簇间距离约 2~4。所以加噪后的正样本和它的 anchor 依然明显属于同一簇——这是构造数据时刻意保证的「无歧义的同一性」。模型要学的不变量(invariant,就是「该被忽略的扰动」)是:**一个视图和它的源样本是同一个实例,哪怕加了噪声**。

为什么这个抽象重要?因为它把「负样本从哪来」这个第 04 章最费劲的问题(还记得吗,raw unigram 会让负样本几乎全是「the / a」这种高频填充词,模型把容量浪费在推开它早就忽略的停用词上)直接绕过去了。对比学习的答案简单粗暴:

> **batch 即负样本集**。对 anchor `i` 来说,同一批里其余所有样本的视图就是它的负样本。

看代码里训练循环怎么构图(`train` 函数内):

```ts
const candByIndex: Vec[] = positives.map((p) => project(W, p));
const anchorVec = project(W, anchors[i].feat);
loss = infoNceLoss(anchorVec, candByIndex, i, opts.temperature);
```

候选池 `candByIndex` 是**整批 32 个样本的投影**,anchor `i` 的正样本是其中第 `i` 个,其余 31 个自动成为负样本。不需要独立的负采样器,不需要 unigram^0.75,不需要负样本分布调参——batch 本身就是。这就是为什么对比学习能 scale 到亿级数据:负样本是「免费」从 batch 里捡的,你只要把 batch 调大,负样本数量自动跟着涨(这个杠杆第 06 章会专门拆)。

## InfoNCE:带温度的 softmax-over-负样本

现在看核心损失。InfoNCE 的定义是:

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(a, a^+)/\tau)}{\sum_j \exp(\text{sim}(a, x_j)/\tau)}
$$

拆开读:分子是 anchor 和它正样本的相似度;分母是 anchor 和**所有候选(含正样本)**的相似度求和。这整个就是一个 **softmax**——它想把概率质量全压到正样本那一项上。`τ`(temperature,温度)是除在每个相似度上的缩放,它决定 softmax 有多「尖」。

代码里实现得很直白(`infoNceLoss`):

```ts
function infoNceLoss(anchor: Vec, candidates: Vec[], positiveIdx: number, temperature: number): Value {
  const logits: Value[] = candidates.map((c) => cosineSim(anchor, c).mul(1 / temperature));
  const posLogit = logits[positiveIdx];
  return logSumExp(logits).sub(posLogit); // -(pos - lse) = lse - pos
}
```

注意三个设计决策,每一个都是「为什么这么写」而不是「随便写写」:

**第一,相似度用 cosine(余弦)不用裸点积。** 看 `cosineSim`,它先归一化再算。为什么?裸点积允许模型作弊——它可以靠**放大向量模长**来拉高正对的点积,而不去学真正有用的角度结构。归一化强制模型只能用方向(角度),而方向正是我们最后评估时关心的东西。这是个反复在工程里咬人的坑:用了未归一化的相似度,损失看着在降,表示却没学到结构。

**第二,分母里用 logSumExp 而不是直接 `sum(exp(...))`。** 这不是数学洁癖,是防崩。看注释里写得很清楚:

```ts
// Numerically stable logsumexp over a list of Values. exp() of large positive
// logits overflows to Infinity → loss = NaN, the #1 InfoNCE training crash.
```

`τ` 很小的时候(比如 0.07,CLIP 的初始值),相似度被除以 `τ` 放大成很大的 logit,`exp()` 直接溢出成 `Infinity`,loss 变 NaN——这是 InfoNCE 训练最常见的崩法。标准修法是减去最大 logit 再 exp、log 之后加回来(`logSumExp` 函数),数学上完全等价(减的是常数,梯度里抵消),但数值上稳。如果你之后自己实现对比损失,这是必须抄的一行。

**第三,温度 `τ < 1` 会让 softmax 变尖。** 健康运行用 `T = 0.2`(SimCLR 量级)。`τ` 越小,损失越聚焦在「最难的那个负样本」上;`τ` 越大,推力越均匀地摊在所有负样本上。这个旋钮极其敏感,小到一定程度会直接触发本章后半讲的坍缩。第 06 章会专门把 `τ` 当一个连续旋钮扫一遍,看它怎么影响几何。

### 跟 SGNS 到底哪里同源

把上面的 softmax 和第 04 章的二分类对一下:

- SGNS 对每个 pair 算 `σ(sim)` 做独立二分类,负样本之间互不影响。
- InfoNCE 把所有候选放进**一个 softmax** 分母里,负样本之间通过归一化项耦合——推开一个负样本会间接影响其他负样本的梯度分配。

但本质的「力」是一样的:**正对的相似度往上拉(分子),负对的相似度往下压(分母)**。SGNS 是 InfoNCE 在「每个负样本独立、无温度」时的离散特例;InfoNCE 是 SGNS 的连续、温度化、批内耦合版本。Mikolov 2013 的 word2vec 和后来 van den Oord 2018 的 CPC、Chen 2020 的 SimCLR,优化的是同一个东西——这就是开头那句「✦ 同一思想,从 NLP 走到视觉再回到通用表示」的字面意思。

## 训练实测:模型自己学出了簇

跑健康配置(`T=0.2, epochs=80, lr=0.05, Adam`)的真实输出:

```
损失曲线: loss 2.1539 -> 1.8762   (训练 wall-clock 674 ms, 真测)
```

674 毫秒,32 样本 × 80 epoch,这是真测的 wall-clock,不是估算。loss 从 2.15 降到 1.88,降幅看着不大,但别只看 loss——对比学习的 loss 绝对值几乎没有可解释性(它依赖 batch 大小、温度、负样本数),**真正该看的是几何诊断**。

训练后的 2D 散点(代码里 `asciiScatter` 直接打出来,因为投影维度 `D_OUT=2`,不用 PCA):

```
+──────────────────────────────────────────────────+
│                  3   3 3                          │
│     1                   3  33                     │
│   1    1  1               3                       │
│1                                             2    │
│       1                                       2 2 │
│            0           0                  2       │
│                0    0                             │
│                  0                                │
+──────────────────────────────────────────────────+
```

相同数字(真实簇 id)聚成了 4 团。关键在于:**模型从头到尾只见过加噪视图,从没见过簇标签**。聚簇完全是对比学习自己学出来的——它只被告知「这两个视图是同一个东西」,就足以把整个空间组织成有意义的结构。这正是对比学习被叫做「自监督」(self-supervised,不用人工标签的监督)的原因:监督信号是数据自己的结构造出来的。

## ⚡ 脏秘密:alignment 与 uniformity 的拉锯

现在到本章最该记住的部分。对比学习的损失看起来只是「拉近正对、推远负对」,但 Wang & Isola(2020)证明:**InfoNCE 在隐式优化两个正交目标**,而它们之间存在张力。

> ⚡ **对比学习的脏秘密**:它优化的隐式目标是 alignment(正对靠近)与 uniformity(表示在球面铺开),两者拉锯——只追 alignment 会表示坍缩到一点。这是目前**没有通用解**的开放区:如何在不靠大量负样本/不靠工程 trick 的前提下稳定地平衡两者,仍在研究中(BYOL/SimSiam 用 stop-gradient 绕开负样本、DINO 用 centering+sharpening,都是缓解而非通解)。

两个量的大白话定义:

- **alignment(对齐度)**:正对之间的平均距离,**越低越好**——意思是「同一个东西的两个视图应该挨得近」。
- **uniformity(均匀度)**:所有表示在单位球面上铺得多开,**越负越好**(它是负的对数高斯势能)——意思是「不同的东西应该尽量占满整个空间,别挤成一团」。

配套代码用 `core/eval.ts` 里的 `alignment` 和 `uniformity` 实测训练前后:

```
几何诊断 (Wang-Isola, 训练前 vs 训练后):
  alignment  (正对平均距离, 越低越好):  0.0736 -> 0.0234
  uniformity (分布散开程度, 越负越好):  -1.3510 -> -1.6512
```

读这两个数:alignment 下降 0.0502 = 正对被拉近了;uniformity 从 -1.35 变到 -1.65、**保持负值且更负** = 表示没坍缩、反而铺得更开。这是健康的标志——两个目标同时往好的方向走。

为什么说它们「拉锯」?因为只追 alignment 有一个平凡解:**把所有输入映射到同一个向量**。这样正对距离完美为 0(alignment 完美),但整个空间挤成一个点——这就是 **representation collapse(表示坍缩)**。uniformity 就是专门用来抓这个的:坍缩时它会趋近 0。

## 失败模式实测:去掉负样本 → 坍缩

代码用一个对照实验把坍缩跑出来。`collapseLoss` 故意只优化正对、**分母里不放负样本**:

```ts
function collapseLoss(anchor: Vec, positive: Vec): Value {
  return cosineSim(anchor, positive).mul(-1);
}
```

注释把根因写死了:「Minimizing -sim(anchor,pos) alone has a trivial global optimum: map EVERY input to the same vector」。也就是说,坍缩不是 bug,是这个损失函数的**全局最优解**——优化器只是忠实地走到了那里。

实测对比(健康运行 vs 坍缩运行,同数据同优化器,唯一变量是损失函数):

```
坍缩后几何诊断 (对比健康运行):
  alignment:   健康 0.0234   坍缩 0.0000  (坍缩的 alignment 极小 — 但这是陷阱)
  uniformity:  健康 -1.6512   坍缩 -0.4893  (坍缩 uniformity 趋近 0 = 所有向量挤成一点)
```

看出陷阱了吗:**坍缩的 alignment = 0.0000,比健康的 0.0234 还「完美」**。如果你只盯着 alignment(或者只盯着 loss),你会以为坍缩的模型训得更好。这就是为什么必须同时看 uniformity——它从 -1.65 退化到 -0.49,趋近 0,直接暴露了「所有向量挤成一团」。

代码还给了一个更直接的「方向坍缩」度量——所有 embedding 两两的平均余弦相似度:

```
直接度量方向坍缩 (所有 embedding 两两平均余弦相似度, ∈[-1,1]):
  健康: -0.0286   坍缩: 0.2258   (坍缩→1.0 = 所有向量指向同一方向)
```

健康时平均余弦 ≈ 0(向量方向各异、铺开);坍缩时升到 0.23 并往 1.0 走(所有向量越来越指向同一方向)。注意这里度量的是**方向坍缩**而非位置坍缩,所以用 cosine 而不是 L2 距离——因为损失优化的就是 cosine,度量必须和损失对齐,否则你量的是错的东西。

**这个失败模式的工程教训**:InfoNCE 分母里的负样本不是可有可无的配菜,它们提供「推开」的力,是阻止坍缩的唯一机制。一旦你的负样本太少(batch 太小)、或温度太小到只有一个负样本主导梯度、或不小心写成了只优化正对的损失——空间就会塌。第 06 章会让你看到:负样本数量太少时 uniformity 怎么逐步恶化,温度太小时怎么训练抖动、过度分离少数方向。

## 诚实声明与边界

必须说清楚这章数字的可信边界(代码自己也声明了):

```
诚实声明: 以上为 toy 合成簇 + D_out=2, 绝对数值偏乐观。可迁移的是
(1) 训练前后 alignment/uniformity 的变化方向, (2) 健康 vs 坍缩的对比。
```

也就是说:**别把 alignment=0.0234 这种绝对值搬到真实数据上**——合成簇结构干净、输出只有 2 维,数值天然乐观。能迁移的是两件事:(1) 训练应该让 alignment 降、uniformity 更负这个**方向**;(2) 健康运行和坍缩运行之间的**对比形状**。这两点在 ImageNet 上跑 SimCLR、在十亿句对上训 sentence embedding,都成立。

## 小结与往下走

把这章压成几句能带走的判断:

1. 对比学习是第 04 章负采样的通用化:**拉近正对、推远负对**,InfoNCE 是它带温度的 softmax 形式,和 SGNS 数学同源。
2. **batch 即负样本集**——这是对比学习能 scale 的关键工程技巧,负样本「免费」从 batch 里捡。
3. InfoNCE 隐式优化 **alignment + uniformity** 两个拉锯的目标;只看 loss 或只看 alignment 会被坍缩骗。
4. **负样本是阻止坍缩的力**,去掉它,「全输入映射到一点」就是损失的全局最优。
5. cosine 归一化 + logSumExp 数值稳定,是实现 InfoNCE 不崩的两个必备工程细节。

第 06 章接着把这章埋下的两个旋钮——**温度 `τ` 和负样本(数量、难度)**——当连续变量扫一遍,你会看到 `τ` 有个内部最优(太小训练抖、太大趋向坍缩)、负样本数量收益递减(log 级)、hard negative(难负样本)先升后降(过头会引入假负样本)。这章给了你判断「几何好不好」的两把尺(alignment / uniformity),第 06 章就是拿这两把尺去量旋钮怎么调。
