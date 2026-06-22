---
title: "Activation Patching：从相关到因果，定位哪个组件真负责"
slug: "05"
collection: "interpretability"
order: 5
summary: "第 02 章的注意力图、第 03 章的探针、第 04 章的 logit lens 都只证明了「信息在那里」，没证明「答案用了它」——这笔相关≠因果的债，本章用 activation patching 一次还清。做一次干净 run、一次破坏 run，把破坏 run 的某个命名激活替换进干净 run，看 logit 恢复多少，恢复越多该组件越负责。本章扫遍所有 (层, 位置, 组件) 画因果热图，把责任收敛到少数几个点；这套定位法是第 06 章复现 induction head、第 07 章验证 SAE 特征因果性的共同武器。"
topics:
  - "可解释性"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

前四章我们攒下了一堆「看起来很负责」的嫌疑组件，但没有一个被定罪。第 02 章那个在 `=` 位把注意力几乎全压到第一个操作数上的头，第 03 章那个线性探针（linear probe，用一个线性分类器去读某层激活里是否编码了某信息）能从 L0 残差里 90%+ 解码出操作数的层，第 04 章 logit lens（把中间层残差直接乘 unembedding 当成「此刻的预测」）里逐层逼近正确答案的那条曲线——它们全是**相关证据**：信息确实在那、模型确实在看那。但「在那」「在看」不等于「答案依赖它」。一个头可以死死盯着第一个操作数，而它的输出对最终 logit 毫无贡献——下游根本不读它。要把嫌疑变成定罪，唯一的办法是做手术：动它一下，看答案塌不塌。这就是 activation patching（激活替换，又叫 causal tracing）。

本章对应 `examples/interpretability-from-scratch/src/stage05-activation-patching.ts`，研究对象刻意复用 stage01 的同一个 checkpoint（模 7 加法，`modAdd(7)`），这样本章的因果结论能和前几章的相关结论直接对账。

## 因果实验的最小结构：clean / corrupt 一对

patching 的核心是一个对照实验，结构小得令人发指：你需要两个输入。一个 **clean**（干净，模型答对），一个 **corrupt**（破坏，模型答错或答成别的）。然后你跑 clean，但在某个命名激活点上，把它的值偷偷换成 corrupt run 在同一点的值。如果这一次替换就把 clean 的正确答案打塌了，那这个点就**因果地**参与了计算；如果答案纹丝不动，它在这个任务上没有因果责任——无论它的注意力图多漂亮。

corrupt 的设计是大多数人翻车的地方，stage05 在 `makePair` 里把约束写死了：

```typescript
function makePair(a: number, aCorrupt: number, b: number): PatchPair {
  if (a === aCorrupt) throw new Error("makePair: clean and corrupt operand must differ");
  const cleanAnswer = (a + b) % P;
  const corruptAnswer = (aCorrupt + b) % P;
  if (cleanAnswer === corruptAnswer) throw new Error("makePair: answers must differ for a usable gap");
  return {
    cleanIds: [a, b, P], // P is the "=" delimiter token id
    corruptIds: [aCorrupt, b, P],
    cleanAnswer,
    corruptAnswer,
  };
}
```

注意两条硬约束。**第一，clean 和 corrupt 必须结构相同、只差一个语义因子**：这里 `clean=[2,1,=]`、`corrupt=[5,1,=]`，第二个操作数 `b=1` 和 `=` 完全不动，只改第一个操作数 `a`。为什么？因为 patching 定位的精度上限就是你 corruption 的精度。你一次只破坏「第一个操作数的值」这一个因子，那么任何能恢复 logit 的组件，它恢复的就是「搬运第一个操作数」这一个功能。如果你同时改两个操作数，patching 会把责任摊到一团组件上，什么都定位不到——这是设计层面的失败，不是工具的锅。

**第二，`cleanAnswer != corruptAnswer`**。recovery 的分母就是这两个答案的 logit 差（gap），如果两个答案一样，gap≈0，recovery 是 0 除以 0,整张热图全是噪音。`makePair` 直接把这个不变量抛异常拦在门口。

然后是一道闸门：模型必须真的答对 clean、答错 corrupt，否则你 patching 的是「两个错误答案之间的搬运」，毫无因果意义。stage05 在 `assertSolves` 里查这件事，跑出来：

```
[1] 构造 clean/corrupt 对 (只改第一个操作数):
    clean   = [2,1,7]  期望答案 3  (= (2+1)%7)
    corrupt = [5,1,7]  期望答案 6  (= (5+1)%7)
    模型解对 clean? 是   解对 corrupt? 是
```

两个基线都对，实验可信。这是第 01 章「它到底学会没有」那道闸门在第 05 章的翻版——不先确认模型解对，后面所有数字都是空中楼阁。

## recovery 怎么算：一行公式，注意分母

patching 的引擎在 `examples/interpretability-from-scratch/src/core/interp.ts` 的 `patch` 函数里。它做三件事：缓存 corrupt 在目标点的激活值，记下 clean / corrupt 两个 run 在「答案 token」上的 logit，然后用一个 hook 重跑 clean、把目标点的输出强行替换成 corrupt 值：

```typescript
const hooks: Hooks = {
  [point]: () => corruptAct,
};
const patchedLogits = noGrad(() => model.forward(cleanIds, hooks));
const patchedLogit = patchedLogits.data[position * model.cfg.vocab + answerToken];

// Recovery: fraction of the clean->corrupt drop reproduced by this single patch.
const denom = corruptLogit - cleanLogit;
const recovery = Math.abs(denom) < 1e-9 ? 0 : (patchedLogit - cleanLogit) / denom;
```

hook 的语义很干净：它无视该点观测到的真实值，直接返回我们准备好的 corrupt 激活——那一次返回**就是**干预本身。recovery 是个比例：patch 让答案 logit 朝 corrupt 方向移动了多少，归一化到「clean 到 corrupt 的完整落差」上。recovery≈1 = 这一个点的替换就足以把答案从 clean 拽到 corrupt，它单独就能决定答案；recovery≈0 = patch 它对答案毫无影响。分母那个 `1e-9` 守卫就是上一节那条「答案必须不同」不变量的最后一道代码防线。

## 全组件因果热图：责任收敛到几个点

有了 `patch`，剩下的就是遍历。stage05 的 `scanComponents` 对每层的五个残差流地标——`resid_pre`（块输入）/ `attn_out`（注意力输出）/ `resid_mid`（注意力加回残差后）/ `mlp_out`（MLP 输出）/ `resid_post`（块输出）——各做一次 patch，全打在 `=` 位（`EQ_POS`，唯一可评分的位置）。术语一句话：残差流（residual stream）就是从输入一路加到输出的那条主干向量，每个组件往里读、往里写。

跑出来的热图（行=层，列=组件）：

```
[2] 因果热图: 对每个 (layer, 组件) 在 "=" 位做 patch 的恢复率 recovery
   rarmr
L0 @@@%@
L1 @=@+@
  scale: ' .:-=+*#%@'  vmin=0.000 vmax=1.000
    负责电路 (recovery ≥ 0.5):
      L0.attn_out      recovery=1.004
      L0.resid_pre     recovery=1.000
      L0.resid_mid     recovery=1.000
      L0.resid_post    recovery=1.000
      L1.resid_pre     recovery=1.000
      L1.resid_mid     recovery=1.000
      L1.resid_post    recovery=1.000
      L0.mlp_out       recovery=0.941
      L1.mlp_out       recovery=0.535
```

读这张图：残差流的所有地标 recovery≈1.0，这是预期的——残差流是主干，第一个操作数的信息从 L0 一路流到输出，patch 残差任意一节都会把下游切断。真正有信息量的是**支路组件**：`L0.attn_out` recovery=1.004，说明 L0 的注意力是把操作数搬到 `=` 位的关键；`L0.mlp_out`=0.941 也高度参与；到了 L1，`mlp_out` 只剩 0.535、`attn_out` 已经掉出 ≥0.5 名单。结论形状很清楚：**这个任务的因果责任集中在 L0 的注意力 + MLP，L1 主要是收尾**。这正是我们想从 patching 拿到的东西——不是「哪里有信息」（探针能给），而是「哪里被用了」。

注意 `L0.attn_out` 的 recovery=1.004 略超 1。这不是 bug：recovery 是 logit 比例，patch 单点偶尔会把 logit 推得比完整 corrupt run 还过一点点，超出 [0,1] 是正常的浮点现象，不必修。

## 逐头扫描：把责任从「层」收到「头」

层级太粗。stage05 的 `scanHeads` 进一步 patch 每个注意力头的 `head_z`（单个头在做完注意力、还没经过输出投影时的输出），把因果责任从层收敛到具体的头：

```
[3] 逐头 patch (head_z) 的恢复率:
L0.h0 | ################# 0.388
L0.h1 | #################################### 0.824
L0.h2 | ####################### 0.523
L0.h3 | ### 0.079
L1.h0 | ##### 0.114
L1.h1 | ## 0.049
L1.h2 | ### 0.067
L1.h3 | ##### 0.125
```

责任高度集中在 L0：`L0.h1`=0.824 是主力，`L0.h2`=0.523 是帮手，`L0.h0`=0.388 有部分贡献，`L0.h3`=0.079 基本无关。L1 四个头全在 0.05~0.13 的噪音区。把这个和上节的 `L0.attn_out`=1.004 对上：L0 注意力的因果作用，主要由 h1 和 h2 两个头扛着。这就是「定位」的全部意义——我们从「模型会做模加法」一路收敛到了「L0 的 h1/h2 头 + L0 MLP 是计算操作数搬运的电路」。

## ✦ 两个方向：denoising 找充分，noising 找必要

这里是 patching 的精微之处，也是很多人做 patching 翻车的地方。上面所有的扫描都是同一个方向：跑 **clean**、注入 **corrupt** 值，问「破坏这个点会不会毁掉正确答案」。这个方向叫 **noising**（加噪），它测的是**必要性**——这个组件在原位是不是答案所必需的。

但还有反方向：跑 **corrupt**、注入 **clean** 值，问「单独修复这一个点能不能把答案救回正确」。这叫 **denoising**（去噪），它测的是**充分性**——这个组件单独是否足以承载信号。stage05 的 `patchDirections` 把两个方向都跑了，对焦在 recovery 最高的那个点 `L0.attn_out`：

```typescript
function patchDirections(model, pair, point) {
  // noising: clean run, overwrite point with corrupt value, measure "necessity"
  const noising = patch(model, pair.cleanIds, pair.corruptIds, point, EQ_POS, pair.cleanAnswer).recovery;
  // denoising: swap roles — corrupt run, overwrite with clean value, measure "sufficiency"
  const denoising = patch(model, pair.corruptIds, pair.cleanIds, point, EQ_POS, pair.cleanAnswer).recovery;
  return { noising, denoising };
}
```

注意实现上两个方向就是同一个 `patch` 调用，clean/corrupt 角色互换、目标答案都保持 `cleanAnswer`。结果：

```
[4] 同一点 (L0.attn_out) 的两个方向:
    noising  (clean 注入 corrupt, 测"必要性"): recovery=1.004
    denoising (corrupt 注入 clean, 测"充分性"): recovery=1.000
    两方向一致 => 干净的局部电路
```

两个方向都≈1.0，说明 `L0.attn_out` 既必要又充分——这是一个**干净的局部电路**的标志。

**为什么必须两个方向都跑？** 因为它们能分歧，而分歧本身是结论。一个组件可以「在原位必要」（noising 高：拿掉它答案就坏）却「单独不充分」（denoising 低：只修它救不回答案，因为它还依赖 corrupt run 里缺失的上游 context）。如果你只跑 denoising 去找电路，你会漏掉那些「必要但不自足」的组件，画出的因果图缺一块;如果你只跑 noising,你又会把那些「单独足以决定但实际有冗余备份」的组件误判。**denoising 找的是「足够」组件，noising 找的是「必要」组件，两者不是一回事**。toy 任务因为电路太干净，两个方向恰好重合;真模型里它们经常分叉,只报一个方向的 patching 论文,结论往往是片面的。

## ✦ 失败模式：注意力很尖锐，patch 证明它根本没参与

现在兑现第 02 章欠下的债。第 02 章我们用「注意力锐度」（sharpness，1 − 归一化熵，越高表示注意力越集中在单个 key 上）当作「这个头很重要」的信号。stage05 的 `headSharpness` 把锐度算出来，和 patch recovery 并排放，让两个信号当面对质：

```
[5] 失败模式: 注意力很"尖锐"但 patch 证明它无因果作用
    head        sharpness   recovery   判定
    L0.h0     0.796      0.388      有因果责任
    L0.h3     0.640      0.079      尖锐但无因果 ← 注意力图骗了你
    L0.h1     0.630      0.824      有因果责任
    L0.h2     0.532      0.523      有因果责任
    L1.h2     0.183      0.067      既不尖锐也无责任
    L1.h3     0.031      0.125      有因果责任
    L1.h0     0.018      0.114      有因果责任
    L1.h1     0.008      0.049      既不尖锐也无责任
```

看 `L0.h3`：**锐度 0.640，全场第二高**——它的注意力图看起来极其决定性，第 02 章那套方法会把它列为头号嫌疑。但它的 patch recovery=0.079≈0。把它的输出替换成 corrupt 值，答案纹丝不动。它确实在死死盯着某个位置，但下游**根本不读它的输出**——它在看，但答案不依赖它。注意力图骗了你。

反过来 `L1.h0`/`L1.h3` 锐度接近 0（注意力几乎均匀摊开，第 02 章会直接忽略），recovery 却有 0.11~0.13,不是零。锐度和因果责任不是同一个维度。

这就是本章最该刻进脑子的一句话：**注意力图是「它在看哪」的相关证据，patch 才是「答案是否依赖它」的因果裁判。** 锐度高 + recovery≈0 = 图在撒谎,这个组合的判别力,是整套 patching 方法存在的理由。

```
    ⇒ L0.h3: 注意力 sharpness=0.640 (看起来很决定性), 但 patch recovery=0.079 ≈ 0。
      把它的输出替换成 corrupt 值, 答案纹丝不动 => 它根本没参与计算答案。
```

## 诚实边界：toy 任务的电路太干净了

得把这套结论的适用范围讲清楚，否则你拿到真模型上会被狠狠教训。stage05 结尾自己点破了：

> 诚实边界: toy 任务因果高度集中, 热图干净、单点 recovery 常接近 1.0。真模型存在分布式计算与 backup 电路 (敲掉主组件, 备用组件顶上), 单点 patch 会 LOW-estimate 责任, 需配合 path patching / 多点联合 ablation。

关键词是 **backup 电路**（备份电路）。真模型——比如 GPT-2 small 的 indirect object identification（IOI）任务，Wang et al. 2022 的经典分析——里有个反直觉现象：你把主力 name mover 头 ablate（消融，把它的输出清零）掉，性能只掉一点点,因为有一批「backup name mover」头会顶上来接管。后果是:**单点 patch 会系统性低估责任**。你 patch 主力头,recovery 不到 1,你以为它不重要,其实是备份在替它干活。本章 toy 任务因为电路高度集中、没有冗余,单点 recovery 才能干净地逼近 1.0——这是 toy 的奢侈,不是常态。

**能迁移到真模型的不是具体数字,是方法和形状**:recovery 集中在少数 (层, 位置, 组件) = 局部电路;尖锐注意力 + ~0 recovery = 图在撒谎。这两条形状判据跨模型成立,单点 recovery 的绝对值不成立。

## ⚡ 前沿：从单点 patch 到 path patching 与自动电路发现

本章手写的单点 patch,是一整族因果方法的**原子操作**。把它组合起来,就长成了当前机制可解释性的主力工具:

- **path patching**(路径替换):本章 patch 的是「整层 / 整个组件的输出」,等于切断了这个组件流向**所有**下游的边。path patching 更精细——只沿**特定一条边**干预,比如「只把 L0.h1 流向 L1.h2 这条路径上的激活换掉,L0.h1 流向其他头的不动」。这能区分「L0.h1 重要」和「L0.h1 → L1.h2 这条具体的边重要」,把因果图从「点」画到「边」。第 06 章复现 induction head 时,我们就需要这种边级别的精度去验证「previous-token 头 → induction 头」那条具体通路。
- **causal scrubbing**(因果擦洗,Chan et al. 2022):一套把「电路假设」形式化成「哪些激活可以互换而不改变行为」的检验框架,底层操作还是 patch。
- **ACDC**(Automatic Circuit DisCovery,自动电路发现,Conmy et al. 2023):把「对每条边做 path patching、按贡献阈值剪枝」这个过程自动化,让算法自己搜出电路图,而不是人手一个个 patch。它的内核,就是你本章亲手写的那个 `patch` 函数,套在边的枚举循环里。

⚡ **但这里有个目前无通用解的开放问题**:这些方法全都依赖**人来设计 clean/corrupt 对**——你破坏哪个语义因子,决定了你能定位到什么电路。怎么**自动地、无监督地**找到「正确的破坏维度」,目前没有通用方案;corruption 设计本身仍是一门靠研究者直觉的手艺,corrupt 设计错了,后面再自动化的 ACDC 也只会精确地定位到一个错误的电路。此外,backup 电路导致的「单点 patch 低估 + 多点联合爆炸式组合」如何高效搜索,也仍是正在研究的问题——多点联合 ablation 的组合数随组件数指数增长,暴力搜不动。

下一章(第 06 章),我们用本章这套定位法去复现机制可解释性里最著名的电路之一——induction head(归纳头,负责「看到 AB...A 就预测 B」的 in-context 复制机制),并用 path patching 验证它内部「previous-token 头 → induction 头」的两段式通路。第 07 章则用同一个 `patch` 函数,去检验 SAE(稀疏自编码器)拆出来的特征是不是真的有因果作用,还是又一张漂亮但撒谎的图。patching 是这两章共同的因果武器。
