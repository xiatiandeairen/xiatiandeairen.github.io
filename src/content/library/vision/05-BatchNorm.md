---
title: "BatchNorm：训练统计、推理切换与那个最经典的坑"
slug: "05"
collection: "vision"
order: 5
summary: "第 04 章把网络堆深后，深层激活的分布会随训练剧烈漂移，训练变慢甚至发散。本章实现 batchnorm2d：训练时用当前 batch 的均值方差归一化并学习 scale/shift，同时维护 running 统计量供推理用；推理时切换成 running 统计。重点踩两个真实坑——训练/推理模式切换，以及 BN 反向那个耦合整个 batch 的梯度。BN 是第 06-08 章能稳定训练较深网络的前提。"
topics:
  - "计算机视觉"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

先把这一章最该刻进直觉的事实摆在最前面：**BatchNorm 是这本书里第一个「前向不是纯函数」的层，而且它有两套不同的前向公式，靠一个 mode flag 选哪套跑**。这两件事各自孵化一类 bug，而且都是梯度检查抓不到的——你的数学是对的，只是因为层处在错误的模式，跑的是错误的那套数学。第 04 章我们把网络堆深、发现深层激活分布会随训练漂移（先把 internal covariate shift（内部协变量偏移，指层输入分布在训练中不断变）这个学术争议按下，它是不是 BN 起效的真因至今没定论）；BN 的工程价值不需要那个争议来背书：它把每个通道的激活在前向里强行拉回 mean≈0、var≈1，让后面第 06-08 章堆更深、用更大学习率成为可能。

本章配套代码在 `examples/vision-from-scratch/src/stage05-batchnorm.ts`，核心层实现在 `examples/vision-from-scratch/src/core/nn.ts` 的 `BatchNorm2d`。下面所有数字都来自 `npx tsx src/stage05-batchnorm.ts` 的真实运行（seed=0xb47c，纯 CPU、无网络）。确定性的（NaN 与否、梯度相对误差）和依赖注入分布的具体小数，我会分开标——后者换 seed 会变，可迁移的是定性行为，不是那几位小数。

## 前向：把偏移分布拉回 (mean≈0, var≈1)

raw 卷积激活不是零中心、不是单位方差的——这正是 BN 存在的理由。demo 里我故意往输入注入 `N(mean=5, var=9)`（std=3）的偏移：

```
输入实测 per-channel: mean≈4.8905  var≈8.6129  (跨 4 个通道平均)
输出实测 per-channel: mean≈0.0000  var≈1.0000  (gamma=1,beta=0)
```

输入实测的 4.89/8.61 不是正好 5/9，因为是有限样本（N=8,H=W=6，每通道 288 个数），抽样噪声。过一遍 BN 后均值方差被精准拉到 0/1。看 `core/nn.ts::BatchNorm2d.forward` 怎么算的：

```ts
const perChan = N * H * W; // population size per channel
// ... 对每个通道 c：
m /= perChan;                       // batch 均值
v /= perChan;                       // batch 方差（population variance，除以 count）
mean[c] = m;
invStd[c] = 1 / Math.sqrt(v + this.eps);
// ... 然后逐元素：
const xh = (x.data[idx] - mean[c]) * invStd[c];  // 归一化
out[idx] = xh * g + b;                            // 再 scale(gamma)/shift(beta)
```

两个细节值得停一下。

**第一，统计量是「per-channel」而不是 per-element。** 对 (N,C,H,W) 的张量，每个通道 c 把 `N*H*W` 个数当成一个群体算一个均值一个方差。这是 BN 区别于其他归一化的命门——它沿着 batch 维度做统计，所以同一个样本归一化成什么样，取决于同 batch 里的其他样本。这是后面所有麻烦的根源。

**第二，输出 var 是 0.9999... 不是严格 1，因为除的是 `sqrt(var + eps)`。** eps（这里 1e-5）防止方差为 0 时除零。它让输出方差比 1 略小，偏差量级 `~eps/var ~ 1e-6`，显示精度下看不见，所以打印成 `1.0000`。这不是 bug——记住 eps 这个角色，Part 5 它会从「无害的平滑项」变成「决定崩 NaN 还是静默归零」的开关。

`gamma`（scale）和 `beta`（shift）是可学习参数（init 1 和 0，所以此刻是纯归一化）。它们的存在是为了让网络在「需要时」把归一化撤销掉——如果某层其实更适合非零均值的分布，网络可以学出 gamma/beta 把分布推回去。强制 mean0/var1 是默认，不是枷锁。

## running 统计：推理时要用的冻结统计量从哪来

训练时用 batch 统计，但推理时你可能只喂一张图——一张图算不出有意义的「batch 均值方差」。BN 的解法是：训练过程中用 EMA（指数移动平均，新值 = 0.9×旧值 + 0.1×本批值）悄悄攒一份「整个训练集的均值方差估计」，推理时冻结它来用。看 `core/nn.ts` 的 EMA 更新（在前向里、作为副作用）：

```ts
// EMA update of running stats for inference. Mutating buffers here (a side effect in
// forward) is the standard BN design; it's why BN forward is NOT a pure function.
this.runningMean[c] = (1 - this.momentum) * this.runningMean[c] + this.momentum * m;
this.runningVar[c]  = (1 - this.momentum) * this.runningVar[c]  + this.momentum * v;
```

注意这行在 `forward` 里改 `this.runningMean`——**这就是「前向不是纯函数」的字面证据**。同样的输入，第二次 forward 会让 running 统计往前挪一步。这个设计选择（把统计累积塞进前向的副作用）后面会反咬一口：Part 5 里一次错误的前向会污染 running 统计，连累之后所有推理。

momentum=0.1 意味着每个 batch 只补上当前 gap 的约 10%，所以收敛是几何级的。demo Part 2 让 running 从恒等初值 (0,1) 去逼近真实的 `N(5,9)`：

```
batch |  runningMean  runningVar |  mean_gap   var_gap
   1  |     0.5177     1.8365 |   4.4823    7.1635
   5  |     2.0760     4.4442 |   2.9240    4.5558
  10  |     3.2732     6.2576 |   1.7268    2.7424
  20  |     4.3808     8.0246 |   0.6192    0.9754
  40  |     4.8918     8.8007 |   0.1082    0.1993
  60  |     4.9575     8.9586 |   0.0425    0.0414
```

gap 单调收缩、永不过冲——这是 EMA 的几何收敛特性。这里有个常被忽略的工程后果：**训练初期 running 统计是错的**（batch 1 时 runningMean 才 0.52，真值是 5）。如果你训了几十步就切 eval 评估，eval 输出会因为 running 还没收敛而偏掉。这不是 bug，是 EMA 的暖机代价——所以「训练曲线好看但 eval 抖」在训练早期常常只是 running 统计没暖热，不必急着改模型。

## eval 切换：两套前向，输出真的不一样

为什么要费这个劲维护两套统计？因为推理数据通常**不是**训练分布。demo Part 3 把 running 统计在 `N(5,9)` 上喂熟，然后拿一个偏移过的测试 batch `N(8,4)` 走两条路径：

```
train 路径输出: mean≈-0.0000 var≈1.0000  (用本 batch 统计 -> 又被拉回 ~0/1)
eval  路径输出: mean≈0.9518 var≈0.4837  (用冻结 running -> 保留 test 与 train 分布的差异)
两模式逐元素最大差异: 1.9023  (>0 即证明 eval 切换真实改变了前向)
```

train 路径用测试 batch 自己的统计，又把它拉回 0/1——但这意味着输出依赖「这一批里恰好有哪些图」。eval 路径用冻结的训练统计，于是「test 比 train 偏移了」这个真实信息被保留了下来（输出 mean 0.95 而非 0）。逐元素最大差 1.9023 是「两套前向不等价」的硬证据。

代码里靠什么选路径？看 `core/nn.ts::forward`：

```ts
const useBatch = this.training && !noGradActive();
```

两个开关任一关掉就走 eval 分支。`setTraining(false)` 关 `training`，`noGrad(() => ...)` 关 `noGradActive`——后者更稳，因为推理本来就该在 noGrad 下跑（不建梯度图），顺手就把 BN 切对了。这是有意的双保险：**就算你忘了 setTraining(false)，只要推理裹在 noGrad 里，BN 仍走 eval**。

## ✦ BN 反向：为什么梯度耦合了整个 batch

这是 BN 真正难的地方，也是它在小 batch 下崩坏的根因。普通逐元素层（ReLU、加偏置）的反向是「逐元素」的——`dx[i]` 只依赖 `dout[i]`。BN 不是。因为归一化用的均值方差是**整个 batch 的统计**，所以改动任何一个输入元素都会动到均值、动到方差、进而动到同通道**所有**元素的输出。反向必须把这层耦合还回去。看 `core/nn.ts` 的 backward（手写融合，而非靠 sub/div 算子自动微分组合）：

```ts
// dx = (gamma * invStd / M) * (M*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
// where M = perChan and dxhat = dout.
const M = perChan;
for (let c = 0; c < C; c++) {
  let sumDxhat = 0, sumDxhatXhat = 0;
  // 第一遍：扫整个通道群体，攒两个 reduction
  for (/* 所有 n,i */) {
    const dxhat = t.grad[idx] * gscale;
    sumDxhat += dxhat;
    sumDxhatXhat += dxhat * xhat[idx];
    this.gamma.grad[c] += t.grad[idx] * xhat[idx];  // gamma/beta 梯度也是整通道求和
    this.beta.grad[c]  += t.grad[idx];
  }
  // 第二遍：用攒好的 sum 才能算每个 dx
  for (/* 所有 n,i */)
    x.grad[idx] += (inv / M) * (M * dxhat - sumDxhat - xhat[idx] * sumDxhatXhat);
}
```

公式 `dx = (γ·invStd/M)·(M·dxhat − Σdxhat − xhat·Σ(dxhat·xhat))` 手推出来正好**三项**：

1. `M·dxhat`——这个元素自身上游梯度的直接贡献；
2. `−Σdxhat`——「我改了这个元素 → 均值动了 → 全通道都受影响」的回流（来自对均值的偏导）；
3. `−xhat·Σ(dxhat·xhat)`——「我改了这个元素 → 方差动了 → 全通道缩放都变了」的回流（来自对方差的偏导）。

注意代码必须**扫两遍**：第一遍才能攒出 `sumDxhat` 和 `sumDxhatXhat` 这两个跨整个通道的 reduction，第二遍才能算单个 `dx`。这就是「梯度耦合整个 batch」在代码里的字面形态——你没法逐元素算完一个就走，必须先看完全部。`gamma`/`beta` 的梯度同理，是整通道求和。

手写而不是靠 `(x-mean)/sqrt(var)` 的算子自动组合，一是性能（一趟融合 vs 多个临时张量），二是教学——让这三项显式可见。它对不对？demo Part 4 拿解析梯度对数值有限差分：

```
checked 15 个参数分量
max relative error = 2.79e-6  -> PASS (<1e-5)
```

2.79e-6 这个数是**确定性的**（不依赖注入分布的随机性，是 finite-difference 的固有量级），它实证了上面那三项的解析推导没写错。

现在把这个耦合和「小 batch」连起来——**这就是 ✦ 专家钩子的落点**。均值方差是 batch 统计，batch 越小，这个统计的方差越大、噪声越重；极端到 batch=1 时，单样本算不出可靠的群体统计。检测、分割这类任务因为图大、显存吃紧，batch 常常只有 1-2，BN 在这里效果直接崩坏。这正是 GroupNorm / LayerNorm 被发明出来的根因：它们沿通道分组或沿特征维做统计，**不沿 batch 维**，于是天然不依赖 batch 大小。BN 不是「归一化的标准答案」，它是「假设你有足够大 batch」的那个答案。

## 失败模式：推理忘切 eval + 小 batch

把上面所有伏笔收口。demo Part 5 复现工业界最常报的 BN bug：推理忘了 `setTraining(false)`，且 batch 很小。先看正确基线：

```
5a. 正确 (eval, 用 running 统计):
    输出 mean≈1.0359 var≈0.4393  NaN=false  (有限, 保留输入信息)
```

然后是三档逐渐恶化的错误。

**5b. 忘切 eval，但单图还有空间方差（H*W=36）：**

```
输出 mean≈0.0000 var≈1.0000  NaN=false
-> 不崩, 但用了"单张图自己的统计"归一化: 预测变得依赖这一张图的内部分布, 与训练分布脱节 (静默错)。
副作用: running 统计被这次错误前向污染 (EMA 更新), runningMean[0] 现在 = 5.2973 (被往单图统计拉偏)。
```

这是最阴险的——**不崩、不报错**。因为单图有 36 个空间像素，方差不为 0，前向能跑完。但它用「这一张图自己的内部分布」做归一化，而不是训练集分布。后果：你训练时准确率漂亮，一上线（一张张推理）就掉点，因为每张图被它自己的统计归一化，和训练时见过的分布脱节了。这就是 ⚡ 钩子说的「训练准确率高、上线掉点」头号嫌疑，根因往往就一行：忘了切 eval。更糟的是那个副作用——这次错误前向触发了 EMA 更新，`runningMean[0]` 从训练时学的 5 被往单图统计拉到了 5.2973。**bug 污染了 running 统计，之后即便切回 eval 也带着伤。** 这正是前面「前向不是纯函数」埋的雷：副作用让一次推理错误产生了持久后果。

**5c. 极端：population size = 1（N=1, H=W=1，方差恒为 0）：**

```
eps=1e-5: 输出 = 0.000000, 0.000000, 0.000000, 0.000000  NaN=false
  -> 数值不崩, 但每个通道 x-mean≡0, 输出恒等于 beta(=0): 输入信息被彻底抹掉。
eps=0  : 输出 = NaN, NaN, NaN, NaN  NaN=true
  -> sqrt(0)=0 -> 1/0=Inf, 且 (x-mean)=0 -> 0*Inf = NaN。
  下游传播验证: sum(输出) = NaN  -> 一旦 NaN 进入, 后续每一层、loss、梯度全部变 NaN, 整个网络静默失效。
```

这里 eps 从前面那个「看不见的平滑项」变成了**决定崩法的开关**。当群体只有 1 个元素，方差严格为 0：

- **eps=1e-5**：除的是 `sqrt(0+1e-5)`，数值不崩；但 `x - mean ≡ 0`（唯一的元素就等于它自己的均值），归一化结果恒为 0，输出恒等于 `beta`（默认 0）。**输入信息被彻底抹掉**——网络对任何输入都输出同一个常数，无声地废了。
- **eps=0**：`sqrt(0)=0 → 1/0=Inf`，又 `(x-mean)=0`，`0×Inf=NaN`。而 NaN 会**向下游传播**——demo 验证 `sum(输出)=NaN`，一旦 NaN 进网络，后续每一层、loss、梯度全变 NaN，整个网络静默失效，且 loss 曲线会突然变成一条 NaN 直线。

两种崩法都不抛异常。这是 BN 这类层最难调的地方：**错的不是数学，是数学跑在了错误的前提（小 batch / 错误模式）上**。结论钉死：BN 推理必须 `setTraining(false)` 或裹 `noGrad`；小 batch 训练要么换 GroupNorm，要么确保 batch 够大让 batch 统计可信。

## ⚡ 前沿：能不能干脆把 BN 去掉

BN 的根本别扭在于那个**训练-推理不对称**：训练用 batch 统计、推理用 running 统计，两套前向。它带来三个一直没根治的麻烦——忘切模式就掉点（5b）、小 batch 就崩（✦ 那节）、以及分布式训练时 batch 统计要不要跨卡同步（SyncBN 的额外通信开销）。

一个自然的问题：能不能干脆不要 BN？2020 年起 DeepMind 的 NFNet（Normalizer-Free Networks）这条线就在试。思路不是「换个归一化层」，而是**彻底拿掉所有 normalization 层**，改用 scaled weight standardization（对卷积权重本身做标准化）加上精心设计的残差缩放，把 BN 原本提供的「让深层可训练」那个效果，从「对激活做统计」搬到「对权重做约束」上——于是不再依赖 batch、也不再有训练/推理两套前向。NFNet 在 ImageNet 上做到了媲美甚至超过带 BN 的同级网络。

但这**不是已解决的问题**。去 BN 的方案目前需要更精细的初始化、自适应梯度裁剪（adaptive gradient clipping）等一整套配套技巧，调参更娇气，没有 BN「插上就 work」的鲁棒性，也还没成为视觉骨干的默认选择。「如何在完全不依赖 batch 统计的前提下，既稳定训练超深网络、又保持 BN 那种开箱即用的鲁棒性」——**目前没有公认的通用解，仍是活跃的研究方向**。本书第 06 章会在残差块里用上 BN（先用成熟方案把深网络跑稳），这个「能不能去掉它」的问题，留给你读完整本书后自己掂量。
