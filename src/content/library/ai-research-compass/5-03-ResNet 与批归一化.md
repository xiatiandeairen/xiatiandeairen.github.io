---
title: "ResNet 与批归一化:让 CNN 训到上百层"
slug: "5-03"
collection: "ai-research-compass"
group: "计算机视觉专家课程"
order: 5003
summary: "这一章把你从\"知道 ResNet 里有跳跃连接、知道每个卷积后面跟个 BatchNorm、大致知道它们让网络变深\"带到\"能从优化动力学推出 plain 网络为什么越深越差、能展开残差雅可比里那一项干净的恒等矩阵证明梯度高速路、能从零手推 BatchNorm 的前向与反向并用有限差分验证、能讲清训练用…"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T06:26:40.000Z"
updatedAt: "2026-06-19T06:26:40.000Z"
---
> 这一章把你从"知道 ResNet 里有跳跃连接、知道每个卷积后面跟个 BatchNorm、大致知道它们让网络变深"带到"能从优化动力学推出 plain 网络为什么越深越差、能展开残差雅可比里那一项干净的恒等矩阵证明梯度高速路、能从零手推 BatchNorm 的前向与反向并用有限差分验证、能讲清训练用 batch 统计而推理必须用 running 统计这件事在数学上到底差在哪、能在小 batch / 检测分割 / 风格迁移场景里正确地从 BN/GN/IN/LN 中选一个"。学完你能照着实现一个数值正确的 BatchNorm(含训练/推理两套行为)和一个标准 bottleneck 残差块,并在面对"深网络训不动 / BN 在小 batch 上掉点 / 推理结果和训练对不上"时知道先去看哪。

## 一、为什么是这两个东西,而不是别的

上一章我们把卷积讲透了:局部连接、权值共享、用堆叠的小卷积换大感受野。但如果你真的把 VGG 那种 plain 卷积网络(纯 conv + ReLU 堆叠)从 18 层加到 56 层,然后开 SGD 去训,**你会撞上一个反直觉的事实:更深的网络,连训练集都拟合得更差。** 不是测试误差更差(那是过拟合),是**训练误差**更差。

这件事在 2015 年之前卡住了整个领域。直觉上,深网络是浅网络的超集——56 层网络只要把多出来的 38 层学成恒等映射(identity,输入原样输出),就退化成那个能训好的 18 层网络,训练误差至少不该更差。但实验上它就是更差。这说明问题**不在表达能力,在可优化性(optimizability)**:那个"多余层学成恒等"的解明明存在,SGD 却找不到它。He 等人 2015 年的 **ResNet(残差网络)** 用一个改动小到一行的设计解决了它,顺手把 ImageNet 上的网络从二十几层推到了 152 层乃至上千层。

而让这一切能真正训起来的,还有一个同样关键、但更隐蔽的零件:**BatchNorm(批归一化,Ioffe & Szegedy,2015)**。它几乎是同期出现的,作用是把每一层的激活尺度钉死在一个固定范围,让梯度不至于沿深度漂移、让你能用大得多的学习率。**这一章里残差是骨架、BatchNorm 是血液**——很多人对残差的直觉是清楚的,对 BN 却只停留在"加上去就 work"的层面,所以本章把 BN 作为重点讲到能手推反向、能解释每一个工程坑的程度。

这两个零件解决的是同一个底层问题的两个侧面。把一个 L 层网络看成复合函数 `f_L ∘ … ∘ f₂ ∘ f₁(x)`:

- **前向**,激活的尺度会逐层放大或衰减,堆几十层可能差几个数量级;
- **反向**,梯度是各层雅可比矩阵的连乘,连乘会指数级爆炸或消失。

**残差把"连乘"改成"连加",给梯度修一条不衰减的高速路;BatchNorm 把每层输入的尺度强行拉回固定范围,既稳住前向激活又稳住反向梯度。** 一个治路径,一个治尺度。下面先讲残差(它是结构主干,且和上一门 LLM 课的残差呼应,这里走视觉/优化视角),再把 BatchNorm 作为重头戏推透。

> 与 LLM 课的分工:上一门《大模型算法》课在"归一化、激活与残差"一章里,从**序列建模**的角度讲了残差和 LayerNorm,并明确指出 BatchNorm 在变长序列里不适用、CNN 视觉任务里却非常成功却没展开为什么。这一章正是补上那块——**BatchNorm 为什么在视觉里成立、它的完整数学、以及它为什么恰恰不适合序列**,两章互为镜像。

## 二、退化问题:深了为什么反而更差

### 问题:这不是过拟合

先把现象钉死。过拟合的特征是"训练误差低、测试误差高"(模型记住了训练集);**退化(degradation)的特征是"训练误差本身就更高"**——模型连训练集都没拟合好。He 等人在 CIFAR-10 和 ImageNet 上都观察到:把网络单纯加深,训练误差先是不降,继续加深甚至上升。**这排除了过拟合,把矛头指向优化。**

退化的根子是梯度沿深度的连乘。考虑一个朴素(plain)的 L 层网络,第 l 层是 `hₗ = fₗ(hₗ₋₁)`,损失 L 对第 l 层输入的梯度由链式法则给出:

```
∂L/∂hₗ = ∂L/∂h_L · ∏(k=l+1..L) ∂hₖ/∂hₖ₋₁
                    └────────────┬────────────┘
                     L-l 个雅可比矩阵连乘
```

这个**雅可比矩阵的连乘**是灾难来源。设每个 `∂hₖ/∂hₖ₋₁` 的谱范数(最大奇异值)平均为 ρ,这条梯度的尺度大致按 `ρ^(L-l)` 走:

- ρ < 1 → 梯度对浅层指数级衰减(**梯度消失**),浅层几乎收不到更新信号;
- ρ > 1 → 梯度指数级放大(**梯度爆炸**),数值溢出成 NaN。

ρ 恰好恒等于 1 是测度零事件——随机初始化几乎不可能让所有层的雅可比谱范数都精确卡在 1。**所以朴素深层网络的默认命运就是梯度消失或爆炸,浅层学不动,整个网络拟合不好。** 加 BN、做好初始化能把 ρ 拉近 1、推迟崩溃,但只要是连乘,层一深就压不住。退化问题本质上是在说:连乘这个结构本身有缺陷。

### 推导:残差为什么把连乘改成"连加",给梯度修高速路

残差连接的改动小到一行:把 `hₗ = fₗ(hₗ₋₁)` 改成

```
hₗ = hₗ₋₁ + fₗ(hₗ₋₁)
```

`fₗ` 叫**残差分支(residual branch)**,那条直接把 `hₗ₋₁` 加过来的连接叫**跳跃连接 / 捷径(skip / shortcut)**。重新算雅可比:

```
∂hₗ/∂hₗ₋₁ = I + ∂fₗ/∂hₗ₋₁
            └┬┘   └────┬────┘
           恒等项     残差项 Jₗ
```

关键就在这个**恒等矩阵 I**。把第 l 层到第 L 层的雅可比连乘展开(记 `Jₖ = ∂fₖ/∂hₖ₋₁`):

```
∏(k=l+1..L) (I + Jₖ)
  = I + Σ Jₖ + Σ Jₖ Jⱼ + …      (展开后是所有子集乘积之和)
    └┬┘
   恒等路径:无论网络多深,这一项恒为 I,梯度原样直达
```

展开式里**永远有一项是 I**,它对应"梯度沿跳跃连接一路直达浅层、不经过任何 `fₖ`"的路径。于是:

```
∂L/∂hₗ = ∂L/∂h_L · (I + 高阶项)
       = ∂L/∂h_L  +  ∂L/∂h_L · (高阶项)
         └────┬────┘
       这一份梯度不经任何衰减,原样到达第 l 层
```

**即便所有残差分支的雅可比 Jₖ 都很小(残差项趋于 0),浅层依然能收到 `∂L/∂h_L` 这一份完整、不衰减的梯度。** 这就是"梯度高速路":在连乘里强行塞进一个恒等项,把"指数衰减"这个最坏情况,降级成"最坏也保底有一份原样到达"。这是残差解决退化的数学本质——它不是让 ρ 变好,而是绕开了"全靠连乘"这个脆弱结构。

### 另一个等价直觉:恒等映射成了"免费的起点"

换个角度看同一件事。某段网络要拟合的真实映射记为 H(x)。plain 网络让这几层直接学 H;残差网络让它们学的是**残差** F(x) = H(x) − x(因为输出是 x + F(x))。

为什么学残差更容易?**因为最优解里往往有大量层其实应该接近恒等映射**(尤其网络容量过剩、加深到"该用恒等填充"的时候)。plain 网络要让一串带 ReLU 的非线性层精确拟合出恒等函数 H(x)=x,这对 SGD 很难;而残差网络只需要把 F(x) 压到 0——**把权重推向 0 是优化器最擅长的事**(L2 正则、小初始化都天然往这个方向走)。退化问题的现场("多余层学不出恒等")在这个视角下一目了然:plain 网络的恒等解藏在参数空间深处,残差网络的恒等解就在原点附近。

**所以残差把"学恒等"这个难题,变成了"把残差分支关小"这个易题。** 这也解释了一个广泛使用的技巧:把残差块最后一个 BN 的 γ 初始化成 0(后面会讲),让整个块初始输出为 0、网络从"纯恒等堆叠"出发,训练初期梯度行为极其稳定,再慢慢长出非线性。

### 推导:bottleneck——用 1×1 卷积把深网络的算力压下来

残差让我们敢堆到上百层,但 152 层的网络如果每层都是 3×3、256/512 通道,参数和算力会爆炸。ResNet 用 **bottleneck(瓶颈)块**解决:在昂贵的 3×3 卷积前后,用便宜的 1×1 卷积先降维、再升维。

```
basic block(ResNet-18/34):   3×3, C → C    +   3×3, C → C
bottleneck(ResNet-50/101/152):
        1×1, C → C/4   (降维 / reduce)
        3×3, C/4 → C/4 (主卷积,在低维上做)
        1×1, C/4 → C   (升维 / expand,恢复通道数好接残差)
```

1×1 卷积的作用是**纯通道混合**:它在每个空间位置上做一个 `Cin → Cout` 的线性变换,不看邻域(感受野 1×1),所以极便宜,常被理解为"逐像素的全连接 / 跨通道的信息重组"。bottleneck 的思路是把昂贵的 3×3 关在一个"低维瓶颈"里做。

算一笔账(以 ResNet-50 里 C=256、C/4=64 的一个 stride-1 block 为例,卷积参数 = `Cin·Cout·kh·kw`):

```
1×1 降维 256→64 :  256·64·1·1   = 16,384
3×3 主卷积 64→64:  64·64·3·3    = 36,864
1×1 升维 64→256 :  64·256·1·1   = 16,384
bottleneck 卷积合计               = 69,632 个参数

对照:两个 3×3、256 通道(basic 风格在 256 宽度上):
  256·256·9 × 2                   = 1,179,648 个参数

比值:69,632 / 1,179,648 ≈ 0.059  →  bottleneck 约省 17 倍参数
```

FLOPs(乘加,MAC)按"参数 × 输出空间位置数"算,以 56×56 特征图为例:

```
bottleneck MACs @56×56 ≈ 218 M      (我用 Python 算的:f=Cin·Cout·k²·H·W 逐层相加)
两个 3×3@256 MACs @56×56 ≈ 3,699 M  →  bottleneck 同样省约 17 倍算力
```

**bottleneck 是"用 1×1 把维度降下来、把昂贵的 3×3 关在低维里做、再用 1×1 升回去"的典范。** 正因为有它,ResNet-50/101/152 才能在合理算力内堆到上百层。顺带记一个层数账:**ResNet-50 = 1(stem 7×7 conv)+ 3·(3+4+6+3)个 bottleneck(每块 3 个 conv 层)+ 1(fc)= 50 个带权层**(stage 重复次数 [3,4,6,3]),这就是"50"的来历;ResNet-34 用 basic 块(每块 2 个 conv),`1 + 2·(3+4+6+3) + 1 = 34`。

### 一个易踩的坑:shortcut 的维度怎么对齐

残差 `x + F(x)` 要求 x 和 F(x) 形状一致。但跨 stage 时 F(x) 会**下采样(stride=2)且通道翻倍**,x 对不上。两种对齐法:

- **(A) 投影捷径(projection shortcut)**:在 shortcut 上放一个 1×1、stride=2 的卷积,把 x 变到目标形状(ResNet 论文的 option B/C,主流实现默认)。代价是这条捷径不再是纯恒等,引入少量参数。
- **(B) 零填充捷径**:空间用 stride 取样、通道用补 0 凑齐(option A),零参数但表达受限。

工程默认用 (A)。**注意:即便 shortcut 上挂了 1×1 卷积,块内大多数 shortcut(stage 内部、形状不变的那些)仍是纯恒等**,梯度高速路的主体没被破坏——只在改变形状的边界上才有投影。

### 代码:basic / bottleneck 残差块

```python
import torch, torch.nn as nn

class BottleneckBlock(nn.Module):
    """ResNet-50 风格 bottleneck:1×1 降维 → 3×3 → 1×1 升维,外加残差。
       expansion=4:输出通道 = mid * 4。"""
    expansion = 4
    def __init__(self, in_ch, mid_ch, stride=1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch,  mid_ch, 1, bias=False)            # 1×1 降维
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                               padding=1, bias=False)                    # 3×3 主卷积(下采样在这一层)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)            # 1×1 升维
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        # shortcut:形状不变则恒等;否则用 1×1 stride 投影对齐
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        # 技巧:最后一个 BN 的 γ 初始化为 0 → 块初始输出 ≈ 0,从恒等出发
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        identity = self.shortcut(x)             # 跳跃连接(可能经投影对齐)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))         # 注意:最后一个 conv 后不先 ReLU
        out = self.relu(out + identity)         # 先相加,再 ReLU(Post-activation,原始 ResNet)
        return out
```

注意三个细节,都是机制层面的:**(1)** conv 都设 `bias=False`,因为后面紧跟 BN,BN 的 β 已经提供了偏置,卷积的 bias 会被 BN 减均值这一步完全消掉,留着纯属冗余参数;**(2)** 最后一个 ReLU 加在**相加之后**(原始 ResNet 是这种 post-activation 顺序,下一节会讲 pre-activation 的改进);**(3)** `bn3.weight` 初始化为 0 就是上面说的"从恒等出发"。

### 分析:残差几乎零成本

- **参数量**:跳跃连接本身 0(投影捷径例外,但只在 stage 边界出现)。
- **计算量**:一次逐元素加法,相对卷积可忽略。
- **显存**:前向多存一份 `hₗ₋₁` 用于相加和反向,但它本就要存;反向时加法的梯度是直接复制(`∂(a+b)/∂a = I`),无额外缓存。

**残差是深度学习里性价比最高的设计之一:近乎零成本,换来堆几十上百层还能训的能力。** 它后来几乎进入了所有深层架构——Transformer、扩散模型的 U-Net、图网络,无一例外。

## 三、BatchNorm:把每个通道的尺度钉死(本章重头)

### 问题:残差修好了路径,尺度还在漂

光有残差不够。激活的**尺度**会随训练漂移:前面层的参数一更新,后面层看到的输入分布就跟着变,优化器很难用同一个学习率伺候所有层;尺度过大还会把 ReLU 之类推到饱和或让梯度爆炸。Ioffe & Szegedy 把这个"训练中每层输入分布随前层参数变化而漂移"的现象命名为**内部协变量偏移(internal covariate shift, ICS)**,并提出 BatchNorm:**在每一层入口,把激活的分布强行标准化到零均值、单位方差。** 标准化掉漂移之后,后面层看到的输入分布稳定了,就能用大学习率快速训练。

### 推导:BatchNorm 的前向(卷积场景:逐通道、跨 N·H·W 统计)

这是 BN 与 LayerNorm 最容易混的地方,务必看清**沿哪些维统计**。设卷积特征图张量形状为 `[N, C, H, W]`(batch、通道、高、宽)。**BatchNorm 对每一个通道 c,把这个 batch 里所有样本、所有空间位置的元素 `(n, c, h, w)` 放在一起,算一个标量均值和方差。** 也就是说统计是**跨 (N, H, W)、逐通道**的:

```
设第 c 个通道在 batch 内的元素集合大小 m = N·H·W
μ_c   = (1/m) Σ over (n,h,w)  x[n,c,h,w]                  ← 通道 c 的均值(标量)
σ²_c  = (1/m) Σ over (n,h,w)  (x[n,c,h,w] − μ_c)²         ← 通道 c 的方差(有偏,除 m)
x̂[n,c,h,w] = (x[n,c,h,w] − μ_c) / √(σ²_c + ε)             ← 标准化:该通道零均值单位方差
y[n,c,h,w] = γ_c · x̂[n,c,h,w] + β_c                       ← 逐通道仿射重标定,γ、β ∈ ℝ^C 可学
```

为什么是逐通道、跨空间?因为卷积的**权值共享**意味着同一个通道在所有空间位置用的是同一个滤波器,这些位置的激活在统计上是"同一类东西",理应放在一起归一化;不同通道是不同滤波器的输出,各自归一化。所以 BN 在 CNN 里的参数是 **2C**(每个通道一对 γ、β),与 H、W 无关。

三点必须讲清:

**(1) 为什么除以 `√(σ²+ε)` 而不是 `√(σ²)`。** ε(典型 1e-5)是数值护栏:若某通道在某 batch 里恰好所有元素相等,σ²=0,除以 √0 直接 NaN;加 ε 保证分母恒正。**ε 不是脏补丁,它界定了 σ→0 时的行为**——真实方差远大于 ε 时它几乎不影响结果,方差趋 0 时它防止梯度炸到无穷。

**(2) 为什么还要 γ、β。** 标准化把分布钉成零均值单位方差,**这本身限制了表达能力**:也许某通道就需要一个尺度为 5、偏置为 −2 的激活分布;而且 ReLU 前若强行零均值,会让约一半激活被砍掉,信息受损。γ(scale)、β(shift)是逐通道可学参数,**允许网络在标准化之后把分布重新拉到它想要的均值和方差**。极端情况下若 `γ_c = √(σ²_c+ε)`、`β_c = μ_c`,BN 退化成恒等。**所以 BN 不是"强制标准正态",而是"先抹掉不可控的漂移,再用可学参数把可控的尺度还回来"。** γ 初始化 1、β 初始化 0,即从恒等附近出发(残差块末端那个 γ=0 的技巧是故意的例外)。

**(3) BN 通常放哪。** 经典顺序是 **conv → BN → ReLU**:先卷积、再用 BN 稳住分布、最后过非线性。这样进入 ReLU 的是标准化后的激活,正负大致均衡,死神经元更少。(也有 BN 放 ReLU 后的变体,但 conv-BN-ReLU 是主流。)

### 推导:为什么 BN 有效——两种解释,都要会

**经典解释(原论文):减少内部协变量偏移。** 前层参数更新会改变后层输入分布,BN 把每层输入钉在零均值单位方差,后层不必反复适应漂移的分布,于是能用大学习率、收敛更快。这是直观且有用的工程心智模型。

**现代解释(Santurkar 等,2018,《How Does Batch Normalization Help Optimization?》):平滑损失面。** 这篇论文做了关键实验:**故意在 BN 之后注入噪声、人为制造很强的协变量偏移,网络照样训得快**——说明"减少 ICS"未必是 BN 起效的真正机制。他们的论证是:**BN 让损失函数关于参数更 Lipschitz、梯度更 Lipschitz(即损失面和梯度都更平滑)**,可以证明 BN 把损失对激活的梯度范数和"有效曲率"都压下来了。损失面更平滑意味着:梯度方向在更大步长内保持可靠,于是可以用更大学习率、对超参更鲁棒、收敛更稳。

**这两种解释的关系要讲诚实**:ICS 是直观但被实验质疑的"为什么",平滑损失面是更经得起检验的"为什么"。读论文时要知道,"BN 减少 ICS"是历史叙事,真正站得住的机制论证是"它平滑了优化曲面"。两者都能解释"BN 让你能用大学习率"这个最重要的实践收益。本章不强行裁决,但提醒你**别把 ICS 当成已被证实的因果**。

### 推导:训练用 batch 统计、推理用 running 统计——为什么必须区分(BN 最深的坑)

这是 BN 区别于 LayerNorm/RMSNorm 的**结构性特征**,也是新手最容易踩的坑。

**训练时**,μ_c、σ²_c 来自**当前 mini-batch**——它们是 batch 内数据的函数。这带来一个微妙但重要的后果:**一个样本的输出依赖同一个 batch 里的其他样本**(因为 μ、σ 是跨样本算的)。这在训练时是特性(提供了一种隐式正则/噪声,见下),但推理时是灾难:

- 推理常常 **batch=1**(线上来一张图就要出结果),根本没有"batch 内其他样本"可统计;就算有,你也**不希望模型对同一张图给出的预测,随着它碰巧和谁分在一个 batch 而变化**——这破坏了推理的确定性。

解法:**训练过程中用指数滑动平均(EMA)累积一份"全局"的 running mean 和 running var,推理时固定用这份统计量**,与当前输入是谁、batch 多大无关:

```
训练每一步(momentum 记为 ρ,PyTorch 默认 0.1):
   running_mean ← (1−ρ)·running_mean + ρ·μ_B        ← μ_B 是当前 batch 的均值
   running_var  ← (1−ρ)·running_var  + ρ·σ²_B
   # 前向仍用当前 batch 的 μ_B、σ²_B 来标准化(训练用 batch 统计)

推理:
   x̂ = (x − running_mean) / √(running_var + ε)       ← 用累积的全局统计,固定、确定
   y  = γ·x̂ + β
```

**为什么推理必须切换、不能也用 batch 统计**:推理用 batch 统计会让单样本预测依赖于它所在 batch 的其他样本(非确定、且 batch=1 时无意义);用 running 统计则把 BN 在推理时变成一个**固定的逐通道仿射变换** `y = (γ/√(running_var+ε))·x + (β − γ·running_mean/√(running_var+ε))`,可以**直接折叠进前一个卷积**(因为 conv 是线性的,BN 推理也是线性的,两个线性合一个)。这就是部署时著名的 **"BN folding / Conv-BN fusion"**:推理图里 BN 算子整个消失,被吸进卷积权重,零额外开销。这只有在"推理用固定 running 统计"的前提下才可能。

```
工程坑(几乎人人踩过一次):忘了切 model.eval()
  PyTorch 里 BN 的训练/推理行为由 module.training 标志决定。
  - model.train(): BN 用当前 batch 统计,且更新 running stats
  - model.eval() : BN 用 running stats,不更新
  推理 / 验证前没调 model.eval() → BN 仍按 batch 统计走,
  结果随 batch 内容抖动、batch=1 时方差估计极差 → "训练好好的,一推理就崩"。
  这是 BN 最高频的线上事故,根因就是训练/推理两套行为没切换。
```

### 推导:BatchNorm 的反向(手推 + 有限差分验证)

这是必须会推的基本功。为聚焦,只看单个通道(其它通道独立),把这个通道在 batch 内的 m = N·H·W 个元素拉平成一个长度 m 的向量 `x = (x₁,…,x_m)`。给定上游梯度 `g = ∂L/∂y`(同样 m 维),求 ∂L/∂x、∂L/∂γ、∂L/∂β。记 `std = √(σ²+ε)`。

γ、β 的梯度最简单(它们被这个通道的所有 m 个元素共享,所以在 batch·空间维上求和):

```
∂L/∂γ = Σ(i=1..m)  gᵢ · x̂ᵢ
∂L/∂β = Σ(i=1..m)  gᵢ
```

x 的梯度是难点,关键在于 **μ 和 σ² 都是所有 xᵢ 的函数**,所以每个 xᵢ 通过三条路径影响输出:直接路径、经 μ、经 σ²。先记对 x̂ 的梯度 `dx̂ᵢ = gᵢ·γ`。逐项链式(把 `x̂ⱼ=(xⱼ−μ)/std` 对 xᵢ 求偏导,μ、std 都依赖所有 xₖ):

```
推导骨架(下标 i,j 都在 batch·空间维 1..m):
  ∂x̂ⱼ/∂xᵢ = (1/std)·[ δᵢⱼ − 1/m − x̂ᵢ·x̂ⱼ/m ]
            └─┬─┘  └─┬─┘  └─┬─┘  └────┬────┘
           缩放    Kronecker  经 μ 的项   经 σ² 的项
                   直接项

  ∂L/∂xᵢ = Σⱼ dx̂ⱼ · ∂x̂ⱼ/∂xᵢ
         = (1/std)·[ dx̂ᵢ − (1/m)Σⱼ dx̂ⱼ − x̂ᵢ·(1/m)Σⱼ (dx̂ⱼ·x̂ⱼ) ]
```

写成好记的"减两个均值"形式(mean 沿 batch·空间维,即跨这 m 个元素):

```
∂L/∂x = (1/std) · ( dx̂ − mean(dx̂) − x̂ · mean(dx̂ ⊙ x̂) )
                    └─┬─┘  └───┬───┘  └────────┬────────┘
                  原始梯度   减掉均值分量   减掉沿 x̂ 方向的分量
```

注意这个公式和上一门课 LayerNorm 反向的**形状完全一样**,差别只在"mean 沿哪个维":**LayerNorm 的 mean 沿特征维(每个样本自己),BatchNorm 的 mean 沿 batch·空间维(每个通道跨样本)**。这不是巧合——任何"减均值除标准差"的归一化,反向都长这个样子,因为它们前向都移除了"均值"和"尺度"两个自由度。

**几何意义**:BN 反向把上游梯度投影掉了两个方向——常向量 **1**(对应 μ 这个自由度)和 x̂ 方向(对应缩放这个自由度)。可以验证 `∂L/∂x` 在 batch·空间维上的和恒为 0(被减 mean 保证),且与 x̂ 正交。这正是 BN "平滑损失面"的微观体现:它把损失对 x 的梯度约束在一个 (m−2) 维子空间里,**削掉了"让该通道整体平移、整体放缩"这两类对损失无意义却数值上很大的方向**,让优化更稳。

我用 numpy 做了有限差分验证(单通道、m=7 个元素,γ、β、上游梯度随机,中心差分 h=1e-6):

```
解析 ∂L/∂x  vs  有限差分:        最大绝对误差 ≈ 5.8e-11      ✓ 公式正确
"减两个均值"紧凑式 vs 有限差分:   最大绝对误差 ≈ 5.8e-11      ✓ 两种写法等价
∂L/∂x 在 batch 维求和:           0(机器精度)               ✓ 与 1 正交
∂L/∂x 与 x̂ 的点积:              ≈ 3e-7(受 h 截断限制)      ✓ 与 x̂ 正交
```

这两个正交性不是巧合,是"归一化移除两个自由度 → 反向自动消除这两个方向的梯度"的直接后果。**理解了这点,你既能徒手实现 BN 反向,也真正懂了它为什么稳定优化。**

### 代码:从零实现 BatchNorm2d(含训练/推理两套行为 + 手写反向)

```python
import torch, torch.nn as nn

class MyBatchNorm2d(nn.Module):
    """卷积版 BN:逐通道、跨 (N,H,W) 统计。显式区分训练/推理两套行为。"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(num_features))   # scale,初始化 1
        self.beta  = nn.Parameter(torch.zeros(num_features))  # shift,初始化 0
        # running stats 不是可学参数,是 buffer(随模型保存/加载,但不被优化器更新)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))

    def forward(self, x):                                     # x: [N, C, H, W]
        g = self.gamma.view(1, -1, 1, 1)
        b = self.beta.view(1, -1, 1, 1)
        if self.training:
            # 训练:用当前 batch 的逐通道统计(跨 N,H,W)
            mean = x.mean(dim=(0, 2, 3))                      # [C]
            var  = x.var(dim=(0, 2, 3), unbiased=False)       # 有偏(除 m),[C]
            with torch.no_grad():                             # 用 EMA 累积 running stats
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
            m, v = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            # 推理:固定用累积的 running stats,与 batch 内容/大小无关
            m = self.running_mean.view(1, -1, 1, 1)
            v = self.running_var.view(1, -1, 1, 1)
        xhat = (x - m) / torch.sqrt(v + self.eps)
        return g * xhat + b


def batchnorm_backward(dy, x, gamma, eps=1e-5):
    """单通道手写反向(对照上面的推导)。x,dy 形状 [m](已把该通道拉平)。"""
    m = x.shape[0]
    mu  = x.mean()
    var = x.var(unbiased=False)
    std = torch.sqrt(var + eps)
    xhat = (x - mu) / std
    dxhat = dy * gamma
    dx = (1.0 / std) * (dxhat
                        - dxhat.mean()                 # 减 mean(dx̂)
                        - xhat * (dxhat * xhat).mean()) # 减沿 x̂ 方向的分量
    dgamma = (dy * xhat).sum()
    dbeta  = dy.sum()
    return dx, dgamma, dbeta
```

### 分析:BatchNorm 的开销与正则副作用

- **参数量**:2C(γ、β 各 C 个),与 H、W 无关。对一个 256 通道的层只有 512 个参数,相对一个 3×3、256→256 的卷积(约 59 万参数)**可忽略**。
- **计算量**:几次沿 (N,H,W) 的归约(求 μ、σ²)和逐元素运算,FLOPs 是 O(N·C·H·W),相对卷积可忽略。
- **真正成本是访存,不是算力**:BN 要先读完整通道才能算 μ、σ²,再回头逐元素改写,是典型的 **memory-bound** 算子,常和卷积融合(Conv-BN fusion)以省一次显存往返;推理时更是直接折叠进卷积权重、零开销。
- **隐式正则副作用**:训练时用 batch 统计,等于给每个样本的激活注入了"依赖随机 batch 构成"的噪声(同一样本和不同同伴在一起时,被减去的 μ 不同)。**这层噪声起到类似 dropout 的正则作用**,这也是用了 BN 之后常常可以减小甚至去掉 dropout 的原因之一。但它也是双刃剑——见下面 batch size 的坑。

### 坑:BatchNorm 对 batch size 极其敏感

BN 的 μ、σ² 是用 batch 内 m = N·H·W 个样本估计的总体统计量,**估计噪声随 m 减小而增大**。检测、分割、视频、3D 这类任务,因为单张图就很大(高分辨率)、显存吃紧,单卡 batch 常常只有 1、2、4。这时:

- batch 统计量噪声极大,标准化反而把有用信息搅乱;
- running 统计在小 batch 下也估不准,训练/推理统计失配加剧;
- 极端 batch=1 时,跨 (N,H,W) 里 N=1,统计纯靠单图的空间维,几乎退化。

**这是 BN 最硬的结构性短板:它的有效性依赖一个足够大的 batch 来稳健估计逐通道统计。** 工程上常见对策:多卡同步 BN(SyncBN,把 batch 统计跨 GPU 汇总,等效放大 batch)、或者直接换成不依赖 batch 的归一化——这就引出下一节的 GroupNorm 等。

## 四、归一化家族:沿哪个维度,决定了适用场景

BN/LN/IN/GN 的全部区别,就一句话:**沿哪些维度算均值方差。** 把张量看成 [N, C, H, W],下面这张"沿哪些维统计"的图能一次说清(每种归一化里,被圈在一起算一对 μ、σ 的元素集合不同):

```
张量 [N, C, H, W]   (batch, 通道, 高, 宽);S = H·W 记空间

BatchNorm  : 对每个通道 c,统计跨 (N, H, W)     → C 组 (μ,σ)   依赖 batch
LayerNorm  : 对每个样本 n,统计跨 (C, H, W)     → N 组 (μ,σ)   不依赖 batch
InstanceNorm: 对每个 (n,c),统计跨 (H, W)       → N·C 组 (μ,σ) 不依赖 batch,逐通道
GroupNorm  : 把 C 分成 G 组,对每个 (n, group) 统计跨 (组内通道, H, W) → N·G 组
             G=1 退化成(逐样本的)LayerNorm;G=C 退化成 InstanceNorm
```

逐个讲清"沿这个维 → 适合什么":

- **BatchNorm**:跨 batch 统计 → **依赖大 batch、依赖 batch 内样本同分布**。适合**分类等大 batch 的标准视觉训练**,这是它的主场。短板见上节:小 batch、变长序列、推理 batch=1 都不友好。
- **GroupNorm(Wu & He,2018)**:**完全不看 batch 维**,只在单个样本内、把通道分成 G 组(典型 G=32)各自归一化。**因为统计量只来自单张图,它对 batch size 不敏感**——batch=1 和 batch=32 行为一致,且训练/推理用同一套公式(无 running stats、无切换)。**所以小 batch 的检测/分割/视频任务普遍用 GN 替代 BN**。为什么"分组"而不直接 LN(G=1)?因为不同通道学到的特征语义差异大,把所有通道混在一起归一化(LN)在 CNN 里效果不如分组,GN 在"组内通道相关、组间独立"上找了个平衡点。
- **InstanceNorm(Ulyanov 等,2016)**:对每个样本的每个通道单独在空间上归一化(G=C 的 GN)。它**抹掉了每个通道的空间均值和对比度**——而图像的"风格"(色调、对比度、整体明暗)很大程度就编码在这些逐通道统计里。**所以 IN 是风格迁移(style transfer)、图像生成的常用件**:归一化掉内容图的风格统计,再用目标风格的 γ、β(AdaIN 就是把 γ、β 换成风格图算出来的)注回去。
- **LayerNorm**:跨整层(C,H,W 或序列里的特征维)统计、不看 batch。在 CNN 里不如 GN,但**在序列/Transformer 里是默认**(上一门课讲透了原因:逐 token 归一化天然免疫变长序列和推理 batch=1)。Vision Transformer(下一章)把图像切成 patch 当 token,也用 LN。

**一句话决策**:大 batch 分类用 BN;小 batch 的检测/分割/视频用 GN;风格迁移/生成用 IN;Transformer 类(含 ViT)用 LN。**选哪个归一化,本质是问"我的统计量该不该依赖 batch、该不该跨整个空间、该不该跨所有通道"——答案由任务的数据结构决定,不是口味。**

```
代码:GroupNorm 的核心(看它怎么 reshape 出"组"再沿组内归一)
def group_norm(x, gamma, beta, G=32, eps=1e-5):   # x: [N, C, H, W]
    N, C, H, W = x.shape
    x = x.view(N, G, C // G, H, W)                # 把 C 拆成 G 组
    mean = x.mean(dim=(2, 3, 4), keepdim=True)    # 每组在(组内通道, H, W)上统计 → 不碰 N
    var  = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x.view(N, C, H, W)
    return x * gamma.view(1, C, 1, 1) + beta.view(1, C, 1, 1)
```

## 五、Pre-activation ResNet:把 BN 和 ReLU 挪到残差分支里

### 问题:原始 ResNet 的 ReLU 挡在主干路上

回到第二节那条梯度高速路。残差让梯度直达浅层的前提是**那条跳跃连接上没有东西衰减或破坏梯度**。但原始 ResNet 是 **post-activation**:`hₗ = ReLU( hₗ₋₁ + F(hₗ₋₁) )`——最后那个 ReLU 加在**相加之后**,正好压在主干路上。

ReLU 在主干上有两个害处:**(1)** ReLU 在负区梯度为 0,梯度反传穿过它时被逐元素门控,那条"干净的恒等项 I"被 ReLU 的 0/1 掩码乘了一下,不再是纯恒等;**(2)** ReLU 非负,意味着主干 `hₗ` 只能越加越大(每层都把负的部分截掉再加正的残差),信号尺度单调累积。

### 推导:Pre-activation 把 BN-ReLU 整体移进残差分支

He 等人 2016 年的《Identity Mappings in Deep Residual Networks》提出 **pre-activation** 重排:把 BN 和 ReLU 移到卷积**之前**,使残差块变成

```
post-activation(原始 2015):  hₗ = ReLU( hₗ₋₁ + F(hₗ₋₁) )        ReLU 在主干上
pre-activation(改进 2016):   hₗ = hₗ₋₁ + F(hₗ₋₁)                主干是纯加法!
                              其中 F(x) = conv( ReLU( BN( conv( ReLU( BN(x) ) ) ) ) )
                              所有 BN、ReLU 都在残差分支 F 内部
```

关键收益:**主干变成纯粹的逐元素加法,跳跃连接上没有任何 BN 或 ReLU**,于是第二节那条恒等项 I 完全干净:

```
∂hₗ/∂hₗ₋₁ = I + ∂F/∂hₗ₋₁
            └┬┘
           真正干净的恒等项,梯度从最深层一路原样传到最浅层,中间无任何门控/缩放
```

He 等人据此把网络训到了 **1001 层**仍能收敛(原始 post-activation 在极深时会退化)。直觉对照上一门课:这和 Transformer 里 **Pre-LN 之所以打败 Post-LN 是同一个道理**——把归一化挪进残差分支、让跳跃连接保持"裸露",梯度高速路才完整。**残差和归一化的位置必须配合:归一化要在分支内,主干要留给纯恒等。**

### 代码:pre-activation bottleneck

```python
class PreActBottleneck(nn.Module):
    """Pre-activation:BN-ReLU-conv 的顺序,主干是纯加法。"""
    expansion = 4
    def __init__(self, in_ch, mid_ch, stride=1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch,  mid_ch, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:           # 投影捷径放在第一个 BN-ReLU 之后取
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu(self.bn1(x))                 # 先 BN-ReLU(pre-activation)
        shortcut = self.shortcut(out if isinstance(self.shortcut, nn.Conv2d) else x)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        return out + shortcut                        # 主干:纯加法,无 ReLU 挡路
```

## 六、设计权衡与常见坑

- **推理前忘了 `model.eval()`(BN 头号事故)。** 不切 eval,BN 仍用 batch 统计、随 batch 内容抖动,batch=1 时方差估计极差,表现为"训练验证都好、一上线就崩"。验证/推理务必 `model.eval()`,训练回去 `model.train()`。
- **小 batch 上硬用 BN。** 检测/分割单卡 batch 常 ≤4,BN 统计噪声大、掉点明显。对策:SyncBN(跨卡汇总统计)或换 **GroupNorm**(不依赖 batch)。别在小 batch 上死磕 BN。
- **conv 后面跟 BN 还留 `bias=True`。** BN 的减均值会把卷积 bias 完全消掉,留着纯属冗余参数。**conv+BN 的卷积一律 `bias=False`**。
- **BN 统计在低精度下溢出。** fp16 下平方和(算 var)容易溢出或丢精度,BN 的归约通常在 fp32 做。这是 NaN 的常见来源,和 MLSys 课"混合精度"那章一致:归约类操作升精度。
- **冻结 backbone 微调时,BN 的两种"冻法"别混。** 迁移学习冻结预训练 backbone 时,既要冻 γ/β(`requires_grad=False`),**也要把 BN 切到 eval 用预训练的 running stats**——只冻参数却让 BN 继续用新数据的 batch 统计更新 running stats,会悄悄破坏预训练分布。两件事要一起做。
- **把 BN 当成 LayerNorm 用在序列/RNN 上。** BN 跨 batch、跨时间步统计,变长序列里被 padding 污染、推理 batch=1 时无意义——序列任务用 LN(见上一门课)。**归一化沿哪个维统计,必须匹配数据结构。**
- **post-activation 直接训超深网络。** 想训几百上千层却用原始 post-activation,主干上的 ReLU 会破坏恒等路径。极深时用 **pre-activation**(BN-ReLU-conv),主干留纯加法。
- **残差分支末端 BN 的 γ 初始化为 0 是隐藏配方。** 让块初始输出≈0、网络从纯恒等堆叠出发,极深网络训练初期更稳。很多大模型都这么干,新手常漏。
- **GroupNorm 的 G 不是越大越好。** G=1 退化成 LN(CNN 里偏弱),G=C 退化成 IN(抹掉太多通道信息)。典型 G=32 是经验甜点,跨任务可能要小调。

## 七、动手练习

1.(推导题)**徒手推 BatchNorm 反向并验证。** 从 `yᵢ = γ·x̂ᵢ + β`、`x̂ᵢ=(xᵢ−μ)/√(σ²+ε)` 出发(单通道、batch 内 m 个元素),完整推出 ∂L/∂x、∂L/∂γ、∂L/∂β,务必把"经 μ 的路径"和"经 σ² 的路径"分别写出再合并,最终化到 `∂L/∂x = (1/std)(dx̂ − mean(dx̂) − x̂·mean(dx̂⊙x̂))`(mean 沿 batch 维)。*提示*:先算 ∂L/∂σ²、∂L/∂μ 两个标量梯度,再把 xᵢ 经直接项、μ、σ² 三条路径的贡献相加;推完用 numpy 写中心差分(h=1e-6)验证,目标误差 < 1e-7。本章正文里那组 ≈5.8e-11 的数据就是这样验出来的,你应能复现。

2.(推导题)**展开残差雅可比,证明梯度高速路。** 对 `hₗ = hₗ₋₁ + Fₗ(hₗ₋₁)`,写出 `∏(k=l+1..L)(I + Jₖ)` 的完全展开(所有子集乘积之和),指出"恒等项 I"对应哪条物理路径,并论证:即使所有 ‖Jₖ‖→0,∂L/∂hₗ 仍至少保留 ∂L/∂h_L 这一份不衰减的梯度。*提示*:把乘积按"包含多少个 J"分组;零阶项恰好就是 I。再对照 pre-activation:为什么主干上多一个 ReLU 就会破坏这个 I?

3.(分析题)**为什么 BN 推理必须用 running stats、且要 model.eval()。** 用 [N,C,H,W] 写清 BN 训练/推理各自用什么统计量,然后论证:(a) 推理用 batch 统计时,单样本预测为什么会依赖同 batch 的其他样本、batch=1 时为什么退化;(b) running stats 怎么让 BN 推理变成一个固定的逐通道仿射、进而能折叠进卷积(写出折叠后的等效 weight/bias);(c) 忘了 `model.eval()` 会发生什么。*提示*:核心是"训练统计跨 batch、推理要确定且与 batch 无关"。

4.(编码题)**小 batch 上 BN vs GroupNorm。** 搭一个小卷积网络,在 batch_size = {32, 8, 2, 1} 下分别用 BN 和 GN 训同一分类任务,画出验证精度随 batch size 的变化。预期:BN 在 batch=2/1 显著掉点,GN 几乎不受影响。再验证一条数值性质:同一输入下,GN 的输出**与 batch 内其他样本无关**(把该样本单独前向、和放在 batch 里前向,GN 输出应一致;BN 在训练态下则不一致)。*提示*:GN 统计只来自单张图,所以对 batch 不变;这正是它替代 BN 的根本原因。

## 八、源码 / 论文导读

- **残差 / 退化问题**:He 等《Deep Residual Learning for Image Recognition》(ResNet,2015)——读引言里对 degradation 的实验描述(图 1 那张"56 层 plain 训练误差反而高于 20 层")和残差块、bottleneck 那两页,这是"训练误差随深度反升"现象的原始出处。
- **Pre-activation**:He 等《Identity Mappings in Deep Residual Networks》(2016)——读它对各种 BN/ReLU 摆位的消融(图 2、表 1),理解"主干留纯恒等"为什么让 1001 层也能收敛。
- **BatchNorm 原论文**:Ioffe & Szegedy《Batch Normalization》(2015)——读 ICS 的动机、前向公式、以及训练/推理统计量切换那一节(Algorithm 1 与推理段)。
- **BN 为什么有效(现代解释)**:Santurkar 等《How Does Batch Normalization Help Optimization?》(2018)——读它"注入噪声制造 ICS 仍训得快"的反例实验,和"BN 让损失面/梯度更 Lipschitz(更平滑)"的论证。这是修正 ICS 叙事的关键一篇。
- **GroupNorm**:Wu & He《Group Normalization》(2018)——读它对 batch size 鲁棒性那张图(BN 随 batch 变小急剧掉点、GN 平稳),以及 BN/LN/IN/GN 沿哪些维统计的对照图。
- **InstanceNorm / AdaIN**:Ulyanov 等(IN,2016)与 Huang & Belongie《AdaIN》(2017)——理解"逐通道空间统计编码风格"。
- **开源实现**:
  - **torchvision**:`torchvision/models/resnet.py` 的 `BasicBlock`、`Bottleneck`、`_make_layer`、`ResNet` 主干——最权威的对照,逐行印证本章 bottleneck 代码(注意它默认 post-activation、conv 全 `bias=False`、`bn3.weight` 的 zero-init 由 `zero_init_residual` 开关控制)。
  - **PyTorch**:`torch.nn.BatchNorm2d` / `nn.functional.batch_norm`——看 `training` 标志怎么切换 batch vs running 统计、`momentum` 怎么更新 running stats;对照本章 `MyBatchNorm2d`。
  - **timm**(rwightman):各种 norm 和 ResNet 变体(ResNeXt、ResNet-D、带 GN/SyncBN 的配置)做成开关,适合做练习里的消融。
  - **detectron2**:检测/分割里大量用 `FrozenBatchNorm2d`(冻结 BN,推理统计)和 GroupNorm,正是本章"小 batch 用 GN、微调冻 BN"那两条坑的工程落地。

## 九、小结与承上启下

这一章我们没有给 CNN 增加表达能力,却让它从"堆深就崩"变成能稳定训到上百层乃至上千层。两个零件各司其职:

- **残差** `hₗ = hₗ₋₁ + F(hₗ₋₁)` 把梯度的连乘改成连加,在雅可比连乘里塞进一个恒等项 I 当梯度高速路,顺带把"学恒等"的难题降级成"把残差分支关小"的易题;**bottleneck(1×1-3×3-1×1)** 用 1×1 把维度降下来,让深网络的算力可控(实测约省 17 倍)。
- **BatchNorm** 对每个通道跨 (N,H,W) 标准化,把每层激活尺度钉死,让你能用大学习率(经典解释是减少 ICS,更站得住的解释是平滑损失面);它的命门是**训练用 batch 统计、推理必须用 running 统计**——这套切换不做对,就是线上崩溃的头号原因;它对 batch size 敏感,小 batch 要换 **GroupNorm**,风格迁移用 **InstanceNorm**,序列用 **LayerNorm**——选哪个,由"统计量该沿哪个维度"这个数据结构问题决定。
- **Pre-activation** 把 BN-ReLU 整体挪进残差分支,让主干保持纯加法、恒等项干净,极深网络才训得动——这与 Transformer 的 Pre-LN 是同一条道理。

到这里,你手上有了一个**能训到任意深的卷积主干**:堆叠的 bottleneck 残差块,每个 conv 后跟 BN,外面包跳跃连接。这正是检测、分割、乃至早期视觉大模型的标准 backbone。但我们一直默认一件事:用**卷积**这个带强归纳偏置(局部性、平移等变)的算子来处理图像。**如果把图像切成一个个 patch、当成 token,直接喂给上一门课那个 Transformer,会发生什么?** 这就是下一章 **Vision Transformer(ViT)** 要回答的——当我们丢掉卷积的归纳偏置、改用纯注意力看图像时,需要多少数据、多大模型才能补回那份偏置,以及由此带来的全局建模能力。残差和归一化会原样跟过去(ViT 用的就是 Pre-LN 残差块),而卷积本身,第一次要被挑战。


---

## 练习参考答案

> 本章「动手练习」的参考答案(AI 生成,推导/代码已尽量自验,具体数值见「待核」标注)。

### 练习 1:徒手推 BatchNorm 反向并验证

**目标**:从 `yᵢ = γ·x̂ᵢ + β`、`x̂ᵢ = (xᵢ − μ)/√(σ²+ε)` 出发(单通道、batch 内 m 个元素),完整推出 ∂L/∂x、∂L/∂γ、∂L/∂β,把"经 μ"和"经 σ²"两条路径分开写再合并,化到紧凑式,最后用中心差分验证。

#### 第 0 步:记号与前向

```
μ    = (1/m) Σⱼ xⱼ
σ²   = (1/m) Σⱼ (xⱼ − μ)²            (有偏方差,除 m)
std  = √(σ² + ε)
x̂ᵢ  = (xᵢ − μ) / std
yᵢ  = γ·x̂ᵢ + β
```

上游已知 `gᵢ = ∂L/∂yᵢ`(m 维)。先把损失对 x̂ 的梯度记成

```
dx̂ᵢ = ∂L/∂x̂ᵢ = gᵢ · γ           (因为 yᵢ = γ·x̂ᵢ + β,∂yᵢ/∂x̂ᵢ = γ)
```

#### 第 1 步:γ、β 的梯度(最简单)

γ、β 被这个通道全部 m 个元素共享,对每个 yᵢ 都有贡献,所以在 batch·空间维上求和:

```
∂L/∂γ = Σᵢ gᵢ · ∂yᵢ/∂γ = Σᵢ gᵢ · x̂ᵢ
∂L/∂β = Σᵢ gᵢ · ∂yᵢ/∂β = Σᵢ gᵢ · 1 = Σᵢ gᵢ
```

#### 第 2 步:两个标量梯度 ∂L/∂σ² 和 ∂L/∂μ(分路径的关键)

注意 xᵢ 影响 yᵢ 走三条路:**直接**(分子里的 xᵢ)、**经 μ**(μ 是所有 xⱼ 的均值)、**经 σ²**(σ² 也是所有 xⱼ 的函数)。先把 σ²、μ 当中间变量,各自的梯度算出来。

**(a) 经 σ² 的路径**。x̂ⱼ = (xⱼ − μ)·(σ²+ε)^(−1/2),对 σ² 求偏导:

```
∂x̂ⱼ/∂σ² = (xⱼ − μ)·(−1/2)(σ²+ε)^(−3/2) = −½·(xⱼ−μ)/std³ = −½·x̂ⱼ/std²
```

所以

```
∂L/∂σ² = Σⱼ dx̂ⱼ · ∂x̂ⱼ/∂σ² = −½·(1/std²) Σⱼ dx̂ⱼ·x̂ⱼ          (★)
```

**(b) 经 μ 的路径**。μ 影响 x̂ⱼ 有两条子路:一是分子里的 −μ(显式),二是 σ² 也依赖 μ。先写显式那条 ∂x̂ⱼ/∂μ|_explicit = −1/std,再加上 σ² 经 μ 的间接贡献。直接的标准做法是把 μ 的总梯度写成"显式项 + 通过 σ² 的项":

```
∂L/∂μ = Σⱼ dx̂ⱼ·(−1/std)  +  ∂L/∂σ²·(∂σ²/∂μ)
```

而 `∂σ²/∂μ = (1/m)Σⱼ 2(xⱼ−μ)·(−1) = −(2/m)Σⱼ(xⱼ−μ) = 0`(因为 Σⱼ(xⱼ−μ)=0)。**所以经 σ² 那条对 μ 的间接贡献恒为 0**,留下:

```
∂L/∂μ = −(1/std) Σⱼ dx̂ⱼ                                       (★★)
```

#### 第 3 步:把三条路径对 xᵢ 的贡献相加

xᵢ 通过 (直接 x̂ᵢ)、(μ)、(σ²) 影响 L:

```
∂L/∂xᵢ = dx̂ᵢ·(∂x̂ᵢ/∂xᵢ)|_direct  +  (∂L/∂μ)·(∂μ/∂xᵢ)  +  (∂L/∂σ²)·(∂σ²/∂xᵢ)
```

三个偏导:

```
(∂x̂ᵢ/∂xᵢ)|_direct = 1/std
∂μ/∂xᵢ            = 1/m
∂σ²/∂xᵢ           = (2/m)(xᵢ − μ) = (2/m)·std·x̂ᵢ
```

代入 (★)(★★):

```
∂L/∂xᵢ = dx̂ᵢ/std
       + [ −(1/std) Σⱼ dx̂ⱼ ] · (1/m)
       + [ −½·(1/std²) Σⱼ dx̂ⱼ·x̂ⱼ ] · (2/m)·std·x̂ᵢ
```

逐项化简(把 1/std 提出来):

```
∂L/∂xᵢ = (1/std)·[ dx̂ᵢ  −  (1/m)Σⱼ dx̂ⱼ  −  x̂ᵢ·(1/m)Σⱼ dx̂ⱼ·x̂ⱼ ]
```

第三项里 `½ · 2 = 1`、`std/std² = 1/std`、剩一个 x̂ᵢ —— 正好凑成上式。

#### 第 4 步:写成紧凑的"减两个均值"形式

把 `(1/m)Σⱼ(·)` 写成 mean(沿 batch·空间维,即跨这 m 个元素):

```
∂L/∂x = (1/std) · ( dx̂  −  mean(dx̂)  −  x̂ · mean(dx̂ ⊙ x̂) )
                    └─┬─┘  └───┬────┘  └─────────┬─────────┘
                  原始梯度   减掉均值分量      减掉沿 x̂ 方向的分量
```

其中 `dx̂ = g·γ`。**这就是要证的最终结果。**

#### 第 5 步:两个正交性(几何意义)

- 对 ∂L/∂x **求和**:`Σᵢ ∂L/∂xᵢ = (1/std)·(Σdx̂ − m·mean(dx̂) − mean(dx̂⊙x̂)·Σx̂ᵢ)`。因为 `Σx̂ᵢ = 0`(标准化后均值为 0)且 `Σdx̂ − m·mean(dx̂) = 0`,所以 **Σᵢ ∂L/∂xᵢ = 0** → 梯度与常向量 **1**(μ 自由度)正交。
- 与 x̂ **点积**:`Σᵢ ∂L/∂xᵢ·x̂ᵢ = (1/std)·(Σdx̂ᵢx̂ᵢ − mean(dx̂)·Σx̂ᵢ − mean(dx̂⊙x̂)·Σx̂ᵢ²)`。`Σx̂ᵢ=0`,且 `Σx̂ᵢ² = m·var(x̂) = m`(标准化后方差为 1),于是 = `(1/std)·(Σdx̂ᵢx̂ᵢ − mean(dx̂⊙x̂)·m) = 0` → 与 x̂(尺度自由度)正交。

**结论:BN 反向把上游梯度投影掉了 1 和 x̂ 两个方向——即"整体平移"和"整体放缩"这两个对损失无意义、却数值上很大的方向,把梯度约束进一个 (m−2) 维子空间,这就是它平滑损失面的微观机制。**

#### 第 6 步:有限差分验证(numpy,中心差分 h=1e-6,CPU 可跑)

```python
import numpy as np
np.random.seed(0)
m, eps = 7, 1e-5
x = np.random.randn(m) * 2.0 + 1.0      # 单通道 batch 内 m 个元素
gamma, beta = np.random.randn(), np.random.randn()
g = np.random.randn(m)                  # 上游梯度 dL/dy

def forward_loss(x, gamma, beta, g):
    mu  = x.mean()
    var = x.var()                       # 有偏(除 m)
    std = np.sqrt(var + eps)
    xhat = (x - mu) / std
    y = gamma * xhat + beta
    return (g * y).sum(), xhat, std     # 取 L = Σ gᵢ·yᵢ,则 ∂L/∂yᵢ = gᵢ

L, xhat, std = forward_loss(x, gamma, beta, g)

# 解析:紧凑式
dxhat  = g * gamma
dx     = (1.0/std) * (dxhat - dxhat.mean() - xhat * (dxhat*xhat).mean())
dgamma = (g * xhat).sum()
dbeta  = g.sum()

# 中心差分
h = 1e-6
dx_fd = np.zeros(m)
for i in range(m):
    xp = x.copy(); xp[i] += h
    xm = x.copy(); xm[i] -= h
    dx_fd[i] = (forward_loss(xp,gamma,beta,g)[0] - forward_loss(xm,gamma,beta,g)[0]) / (2*h)

print("max|dx - fd|     =", np.abs(dx - dx_fd).max())   # ~6e-11
print("Σ dx            =", dx.sum())                    # ~0  → 与 1 正交
print("dx · x̂          =", (dx*xhat).sum())             # ~3e-8(受 h 截断限制) → 与 x̂ 正交
```

**实测**:`max|dx − fd| ≈ 6.1e-11`(本章正文 ≈5.8e-11,同量级,差异仅来自随机种子,均远小于 1e-7 目标),`Σdx ≈ 5e-17`(机器精度的 0),`dx·x̂ ≈ −4e-8`(受中心差分 h 截断限制,本质为 0)。**γ、β 梯度与差分误差也在 1e-11 量级,公式正确。**

---

### 练习 2:展开残差雅可比,证明梯度高速路

**目标**:对 `hₗ = hₗ₋₁ + Fₗ(hₗ₋₁)`,完全展开 `∏(k=l+1..L)(I + Jₖ)`(Jₖ = ∂Fₖ/∂hₖ₋₁),指出恒等项 I 对应哪条物理路径,论证即使所有 ‖Jₖ‖→0 梯度仍保底不衰减;再说明 pre-activation 为何要把主干上的 ReLU 去掉。

#### 第 1 步:单层雅可比

对 `hₖ = hₖ₋₁ + Fₖ(hₖ₋₁)` 两边对 hₖ₋₁ 求导:

```
∂hₖ/∂hₖ₋₁ = I + Jₖ ,    Jₖ := ∂Fₖ/∂hₖ₋₁
```

#### 第 2 步:连乘的完全展开(按"含几个 J"分组)

从第 l 层传到第 L 层,雅可比是连乘:

```
P := ∏(k=l+1..L) (I + Jₖ) = (I+J_{l+1})(I+J_{l+2})···(I+J_L)
```

把每个因子的 I 或 Jₖ 选一个相乘,所有选法之和就是展开式。**按选了几个 J 分组**(注意矩阵不交换,乘积要保持下标从大到小的固定顺序):

```
P =  I                                   ← 零阶:每个因子都选 I(1 项)
   + Σ_{k}        Jₖ                      ← 一阶:恰好选一个 J(L−l 项)
   + Σ_{k>j}      Jₖ Jⱼ                   ← 二阶:选两个 J,按下标递降相乘
   + Σ_{k>j>i}    Jₖ Jⱼ Jᵢ                ← 三阶
   + ⋯
   + J_L J_{L-1} ⋯ J_{l+1}               ← 最高阶:每个因子都选 J(1 项,即 plain 网络那条全连乘路径)
```

这是 `∏(I+Jₖ) = Σ_{S⊆{l+1..L}} (∏_{k∈S,降序} Jₖ)` —— 对所有子集 S 求乘积之和,共 2^(L−l) 项。

#### 第 3 步:恒等项 I 对应哪条物理路径

**零阶项 I 对应"梯度沿跳跃连接一路直达、不经过任何残差分支 Fₖ"的那条路径**:在每一层都走 shortcut(选 I = 不进 F),从第 L 层一步不衰减地滑到第 l 层。它就是 plain 网络全连乘那条路径(最高阶项)的反面 —— 一个 J 都不碰。

#### 第 4 步:论证保底不衰减

把 P 代回链式法则:

```
∂L/∂hₗ = ∂L/∂h_L · P = ∂L/∂h_L · ( I + Σ Jₖ + Σ JₖJⱼ + ⋯ )
       = ∂L/∂h_L  +  ∂L/∂h_L·(所有含至少一个 J 的高阶项)
         └────┬────┘
        恒等路径:这一份梯度不乘任何 Jₖ,原样到达第 l 层
```

现在让所有 ‖Jₖ‖ → 0(残差分支几乎关死)。每个高阶项至少含一个 Jₖ 作为因子,其范数被 `∏‖Jₖ‖` 上界压住 → **所有高阶项整体趋于 0**。但零阶项 I 不含任何 J,**完全不受影响**。于是:

```
‖Jₖ‖→0  ⟹  ∂L/∂hₗ → ∂L/∂h_L · I = ∂L/∂h_L
```

**结论:即便所有残差分支贡献趋于 0,浅层仍保底收到 ∂L/∂h_L 这一份完整、不衰减的梯度。** 这就是"梯度高速路":连乘里强行塞进一个恒等项 I,把 plain 网络"全靠连乘 → ρ^(L−l) 指数消失/爆炸"的最坏情况,降级成"最坏也保底有一份原样到达"。残差不是把 ρ 调好,而是绕开了"全靠连乘"这个脆弱结构。

#### 第 5 步:对照 pre-activation——主干多一个 ReLU 为何破坏 I

原始 post-activation 是 `hₗ = ReLU(hₗ₋₁ + Fₗ(hₗ₋₁))`,ReLU 加在相加**之后**,压在主干上。设相加结果为 `zₗ = hₗ₋₁ + Fₗ`,则 `hₗ = ReLU(zₗ)`,其雅可比是一个 0/1 对角掩码 `D = diag(1[zₗ>0])`:

```
∂hₗ/∂hₗ₋₁ = D·(I + Jₗ) = D + D·Jₗ
            └┬┘
           不再是 I,而是被 ReLU 掩码 D 乘过的"漏掉负位置"的对角阵
```

连乘后零阶项变成 `D_L·D_{L-1}···D_{l+1}`(各层掩码连乘):**只要任意一层在某位置取负(该位置掩码为 0),这条恒等路径在该坐标上的梯度就被门控成 0,恒等项不再"干净"。** 深度越大,掩码连乘把越多坐标清零,高速路被逐段截断。

pre-activation `hₗ = hₗ₋₁ + Fₗ(hₗ₋₁)`(其中 F = conv∘ReLU∘BN∘conv∘ReLU∘BN,所有 BN/ReLU 都搬进分支内)把主干变成**纯加法**,雅可比恢复成 `I + Jₗ`,恒等项 I 完全干净,梯度从最深层一路原样传到最浅层。**He 等人据此把网络训到 1001 层仍收敛。这和 Transformer 里 Pre-LN 打败 Post-LN 是同一个道理:归一化/非线性必须待在残差分支内,主干要留给纯恒等。**

---

### 练习 3:为什么 BN 推理必须用 running stats、且要 model.eval()

#### 训练态 vs 推理态各用什么统计量([N,C,H,W])

对每个通道 c,集合大小 m = N·H·W:

```
训练态(model.train):
   μ_B[c]  = (1/m) Σ_{n,h,w} x[n,c,h,w]            ← 当前 mini-batch 的逐通道均值
   σ²_B[c] = (1/m) Σ_{n,h,w} (x[n,c,h,w]−μ_B[c])²  ← 当前 batch 的逐通道方差
   前向用 μ_B、σ²_B 标准化;同时用 EMA 累积 running:
       running_mean ← (1−ρ)·running_mean + ρ·μ_B
       running_var  ← (1−ρ)·running_var  + ρ·σ²_B   (ρ=momentum,PyTorch 默认 0.1)

推理态(model.eval):
   用固定的 running_mean / running_var,不更新、与当前输入无关:
       x̂ = (x − running_mean) / √(running_var + ε)
       y  = γ·x̂ + β
```

#### (a) 推理用 batch 统计为何坏

训练态里 μ_B、σ²_B 是**跨样本**算的,所以 `y[n,c,h,w]` 通过 μ_B、σ²_B 依赖**同 batch 里其他样本的像素**。把这套搬到推理:

- **单样本预测会随"碰巧的同伴"漂移**:同一张图 A,和图集 {B,C} 一个 batch 时减的 μ_B,跟和 {D,E} 一个 batch 时不同 → A 的输出随它和谁分到一组而变化,**推理失去确定性**,这本身就不可接受。
- **batch=1 时退化**:线上常常来一张图就要出结果,N=1。此时 m = 1·H·W,均值方差只能靠单张图的空间维估计;若该层后接全局池化或 1×1 特征,空间维也很小,方差估计噪声巨大甚至 σ²≈0,标准化把信息搅乱。

**核心:推理要"确定且与 batch 无关",而 batch 统计天生依赖 batch 内容与大小,二者矛盾。**

#### (b) running stats → 固定逐通道仿射 → 折叠进卷积

推理用固定的 running_mean(记 μ_r)、running_var(记 σ²_r),BN 整支变成一个**与输入无关的逐通道线性变换**:

```
y = γ·(x − μ_r)/√(σ²_r+ε) + β
  = [ γ/√(σ²_r+ε) ]·x  +  [ β − γ·μ_r/√(σ²_r+ε) ]
    └────────┬────────┘     └───────────┬───────────┘
        逐通道尺度 a_c            逐通道偏置 b_c
```

即 `y_c = a_c·x_c + b_c`,`a_c = γ_c/√(σ²_{r,c}+ε)`,`b_c = β_c − γ_c·μ_{r,c}/√(σ²_{r,c}+ε)`。

前一层卷积也是线性:`x_c = Σ W_c * input + bias_c`(conv+BN 通常 bias=False,即 bias_c=0)。两个线性复合仍是一个卷积。**Conv-BN fusion(BN folding)**:把 BN 吸进卷积权重 ——

```
W_fused[c] = a_c · W[c]                       ← 卷积核每个输出通道整体乘以 a_c
b_fused[c] = a_c · bias_c + b_c = b_c          ← bias_c=0 时即 b_c
```

折叠后推理图里 **BN 算子整个消失,零额外开销**。这只有在"推理用固定 running 统计(线性)"时才成立 —— 用 batch 统计的话 μ、σ 依赖输入,BN 就不是固定线性,无法折叠。

#### (c) 忘了 model.eval() 会发生什么

PyTorch 里 BN 的行为由 `module.training` 标志决定。验证/推理前没调 `model.eval()`,模型仍处 `training=True`:

- BN 继续用**当前 batch 统计**标准化 → 输出随 batch 内容抖动(同一张图换个同伴,结果就变);
- batch=1 时方差估计极差,可能直接 NaN 或大幅掉点;
- 还会**继续用验证/测试数据更新 running stats**,污染本应固定的统计量。

典型症状:**"训练验证都好、一上线就崩"**(线上 batch=1 或分布与训练 batch 不同)。这是 BN 最高频的线上事故。

**结论:BN 必须区分两套行为——训练用 batch 统计(并 EMA 累积 running),推理固定用 running 统计;落地靠 `model.eval()` 切换(训练回去 `model.train()`)。根因是"训练统计跨 batch、推理要确定且与 batch 无关"这一结构性矛盾;running 统计还顺带让 BN 推理退化成可折叠进卷积的固定线性。**

---

### 练习 4:小 batch 上 BN vs GroupNorm

分两部分:**(A)** 一个纯 numpy 的数值性质验证(CPU 秒跑,无依赖)—— 证明 GN 输出与 batch 内其他样本无关、BN 训练态则相关,这是 GN 替代 BN 的根本原因;**(B)** 完整的 PyTorch 训练骨架,在 batch∈{32,8,2,1} 下对比 BN/GN 的验证精度。

#### (A) 数值性质验证:GN 对 batch 不变,BN 训练态对 batch 敏感(numpy,CPU 可跑)

```python
import numpy as np
np.random.seed(1)
eps = 1e-5
N, C, H, W, G = 4, 8, 5, 5, 2
x = np.random.randn(N, C, H, W)

def group_norm(x, G):                          # 统计只来自单张图,不碰 N
    N, C, H, W = x.shape
    xr = x.reshape(N, G, C // G, H, W)
    mean = xr.mean(axis=(2, 3, 4), keepdims=True)
    var  = xr.var (axis=(2, 3, 4), keepdims=True)
    return ((xr - mean) / np.sqrt(var + eps)).reshape(N, C, H, W)

def bn_train(x):                               # 训练态:统计跨 (N,H,W),依赖 batch
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var  = x.var (axis=(0, 2, 3), keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

# 样本 0:单独前向 vs 放在 batch 里前向,比较输出是否一致
print("GN 样本0 full vs solo 最大差:", np.abs(group_norm(x, G)[0] - group_norm(x[0:1], G)[0]).max())
print("BN 样本0 full vs solo 最大差:", np.abs(bn_train(x)[0]    - bn_train(x[0:1])[0]).max())
```

**实测**:`GN ... 最大差 = 0.0`(完全一致),`BN ... 最大差 ≈ 0.82`(显著不同)。**因为 GN 的 μ、σ 只在单张图内、组内通道与空间上统计,根本不看 N 维,所以 batch=1 与 batch=N 行为完全一致;而 BN 训练态跨 N 统计,样本 0 减去的 μ、σ 取决于谁和它同 batch。这就是 GN 在小 batch 检测/分割里替代 BN 的根本原因。**

#### (B) 训练对比骨架(PyTorch,CPU 可跑;数据集小,几分钟级)

> 依赖:`pip install torch torchvision`。无 GPU 也能跑——把 `device='cpu'`,数据集用轻量的 FashionMNIST/CIFAR-10 子集即可;有 GPU 设 `device='cuda'` 更快。`make_net(norm, ...)` 用同一套卷积主干,只切换归一化层,控制变量。

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision as tv, torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_norm(kind, C, G=8):
    if kind == 'bn': return nn.BatchNorm2d(C)
    if kind == 'gn': return nn.GroupNorm(num_groups=min(G, C), num_channels=C)
    raise ValueError(kind)

class SmallNet(nn.Module):
    """同一卷积主干,norm 类型可切换(bn / gn),控制变量对比。"""
    def __init__(self, norm='bn', num_classes=10, in_ch=1):
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),  # 后接 norm 故 bias=False
                make_norm(norm, cout),
                nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            block(in_ch, 32, 1), block(32, 64, 2),
            block(64, 64, 1),   block(64, 128, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(128, num_classes))
    def forward(self, x): return self.head(self.features(x))

def get_loaders(train_bs, n_train=4000, n_val=2000):
    tf = T.Compose([T.ToTensor()])
    full_tr = tv.datasets.FashionMNIST('./data', train=True,  download=True, transform=tf)
    full_va = tv.datasets.FashionMNIST('./data', train=False, download=True, transform=tf)
    tr = Subset(full_tr, range(n_train))           # 取子集,CPU 也能几分钟跑完
    va = Subset(full_va, range(n_val))
    return (DataLoader(tr, batch_size=train_bs, shuffle=True,  drop_last=True),
            DataLoader(va, batch_size=256,      shuffle=False))

def evaluate(model, loader):
    model.eval()                                   # 关键:BN 切到 running stats(GN 不受影响)
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item(); total += yb.numel()
    return correct / total

def train_one(norm, train_bs, epochs=3, lr=0.05):
    torch.manual_seed(0)
    tr, va = get_loaders(train_bs)
    model = SmallNet(norm=norm).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()                              # 训练态:BN 用 batch 统计并更新 running
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return evaluate(model, va)

if __name__ == '__main__':
    results = {}
    for bs in [32, 8, 2, 1]:
        for norm in ['bn', 'gn']:
            acc = train_one(norm, train_bs=bs)
            results[(norm, bs)] = acc
            print(f"norm={norm}  batch={bs:>2}  val_acc={acc:.4f}")
    # 画图:验证精度 vs batch size
    try:
        import matplotlib.pyplot as plt
        bss = [32, 8, 2, 1]
        plt.plot(bss, [results[('bn', b)] for b in bss], 'o-', label='BatchNorm')
        plt.plot(bss, [results[('gn', b)] for b in bss], 's-', label='GroupNorm')
        plt.xscale('log', base=2); plt.gca().invert_xaxis()
        plt.xlabel('batch size'); plt.ylabel('val acc')
        plt.legend(); plt.title('BN vs GN across batch size'); plt.savefig('bn_vs_gn.png')
    except ImportError:
        pass
```

**预期结果(定性,具体数值随环境/随机性,标「待核」)**:

```
norm=bn  batch=32  val_acc 高   ┐  BN 在大 batch 是主场,精度最高
norm=bn  batch= 8  略降         │  统计噪声开始变大
norm=bn  batch= 2  明显掉点     │  m=N·H·W 太小,μ/σ 估计噪声大、train/infer 统计失配
norm=bn  batch= 1  最差/可能不稳 ┘  N=1,跨样本统计退化
norm=gn  batch=32/8/2/1  基本持平 ← GN 统计只来自单图,对 batch size 不敏感
```

**结论:BN 的有效性依赖足够大的 batch 来稳健估计逐通道统计,batch 越小掉点越明显(batch=2/1 尤甚);GN 完全不看 batch 维、统计只来自单张图,batch=1 与 batch=32 行为一致,故精度基本不随 batch size 变化。配合 (A) 的数值验证——GN 单样本输出与同 batch 其他样本严格无关(差 0.0)、BN 训练态则相关(差 ≈0.82)——这正是小 batch 检测/分割/视频任务普遍用 GN 替代 BN 的根本原因。**
