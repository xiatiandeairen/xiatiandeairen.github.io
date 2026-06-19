---
title: "Vision Transformer:当注意力遇上图像"
slug: "5-07"
collection: "ai-research-compass"
group: "计算机视觉专家课程"
order: 5007
summary: "这一章把你从\"知道 Transformer 在 NLP 很能打、也大概知道注意力公式\"带到\"能从零写出一个 ViT 分类器、能定量解释它为什么在小数据上打不过 ResNet 而在 JFT-300M 上反超、能讲清 Swin 的窗口注意力为什么把 O(n²) 压到线性\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-18T13:38:05.000Z"
updatedAt: "2026-06-18T13:38:05.000Z"
---
> 这一章把你从"知道 Transformer 在 NLP 很能打、也大概知道注意力公式"带到"能从零写出一个 ViT 分类器、能定量解释它为什么在小数据上打不过 ResNet 而在 JFT-300M 上反超、能讲清 Swin 的窗口注意力为什么把 O(n²) 压到线性"。本章不重讲 Transformer 内部机制——多头注意力、QKᵀ/√dₖ、残差+LayerNorm 这些请看大模型算法课 02/03 章,这里默认你会。本章只回答一个问题:**把这套为一维 token 序列设计的架构搬到二维图像上,缺了什么、补了什么、代价是什么。** 核心是一场"归纳偏置 vs 数据量"的对决,以及它对整个深度学习方法论的启示——**结构先验能省数据,但数据足够多时,结构先验反而是天花板**。

## 一、动机:为什么要把 Transformer 搬到视觉

在 2020 年之前,视觉的统治架构是 CNN(卷积神经网络)。CNN 在结构里硬编码了两个关于图像的**强先验(inductive bias,归纳偏置)**:

- **局部性(locality)**:一个 3×3 卷积核只看 9 个相邻像素。它假设"相关的信息在空间上是邻近的"——边缘、纹理这些低级特征确实如此。
- **平移等变(translation equivariance)**:同一个卷积核在整张图上滑动,权重共享。一只猫不管出现在左上角还是右下角,激活的是同一组滤波器。这等价于假设"图像的统计性质在空间上是平稳的"。

这两个先验**极其符合自然图像的真实结构**,所以 CNN 在中等规模数据(ImageNet 的 128 万张)上学得又快又好。**但先验是一把双刃剑:它是假设,假设就有不成立的时候。** 卷积的局部性意味着要建立"图像左上角的物体"和"右下角的物体"之间的长程关系,必须靠堆很多层把感受野(receptive field)慢慢撑大——一个像素要"看到"整张图,可能要十几二十层。全局关系在 CNN 里是**间接、低效**地建模的。

Transformer 的自注意力恰恰相反:**第一层每个位置就能直接看到所有其他位置**(全局感受野),长程依赖是 O(1) 跳数,不需要堆深。这正是 CNN 的短板。于是一个自然的问题浮出水面:**能不能干脆扔掉卷积,直接把图像喂给 Transformer?**

难点在于:Transformer 吃的是一串 token(序列),图像是 H×W×3 的二维网格。最朴素的想法——**把每个像素当一个 token**——立刻被复杂度劝退。自注意力是 O(n²) 的(n 是 token 数):一张 224×224 的图有 n = 50176 个像素,n² ≈ 2.5×10⁹,光一层注意力矩阵就要 25 亿个元素。**不可行。** ViT 的全部巧思,就是用一个极简的办法解决"图像怎么变成一个长度可控的 token 序列"。

## 二、ViT 的机制:图像如何变成 token 序列

ViT(Vision Transformer,Dosovitskiy 等人,ICLR 2021,论文《An Image Is Worth 16x16 Words》)的流水线只有五步,每一步都有明确的理由:

```
图像 (H×W×C)
  → ① 切成不重叠的 patch (每块 P×P)         —— 把 50176 个像素压成 196 个 patch
  → ② 每个 patch 拉平 + 线性投影成 D 维向量   —— "patch embedding",等价于一个大卷积
  → ③ 前面拼一个可学习的 [CLS] token         —— 用来聚合全图、做分类
  → ④ 加上位置编码                            —— 把"二维网格里的位置"信息补回去
  → ⑤ 标准 Transformer Encoder × L 层 → 取 [CLS] 的输出 → 线性分类头
```

下面逐步拆。

### 2.1 Patch 切分:把 O(n²) 拉回可行区间

设输入图像 x ∈ ℝ^{H×W×C}(C 是通道数,RGB 则 C=3),patch 边长 P(ViT-Base 用 P=16)。把图切成 `N = (H/P)·(W/P)` 个不重叠的方块。对 224×224 的图、P=16:

```
N = (224/16) · (224/16) = 14 · 14 = 196 个 patch
```

**这一步是 ViT 能跑起来的关键。** token 数从像素级的 50176 直降到 196,自注意力的 196² ≈ 38416 完全可接受。代价是:**patch 内部的空间结构被一次性拉平丢掉了**——16×16×3=768 个数被拍扁成一个向量,模型不再知道"这个 patch 里左上角那个像素和右下角那个像素是邻居"。这是 ViT 主动放弃的第一份归纳偏置。

每个 patch 拉平成一个 `P²·C = 16²·3 = 768` 维的向量,N 个 patch 排成矩阵 `xₚ ∈ ℝ^{N × (P²·C)} = ℝ^{196×768}`。

### 2.2 Patch Embedding:它本质上是一个步长等于核大小的卷积

把每个拉平的 patch(P²·C 维)线性投影到模型隐藏维度 D(ViT-Base 的 D=768):

```
z₀ⁱ = xₚⁱ · E        E ∈ ℝ^{(P²·C) × D}        i = 1..N
```

E 是可学习的投影矩阵。这一步叫 **patch embedding**。

**这里有一个必须讲透的等价关系:patch embedding 完全等价于一个卷积层,核大小 = 步长 = P。** 想清楚这件事,你才真正理解 ViT 和 CNN 的连续性而非对立。

逐步论证。一个卷积层做的事是:用一个 K×K 的核,在输入上以步长 S 滑动,每个落点上做"核与对应感受野的逐元素乘加"。现在设 **K = P, S = P**(核大小等于步长):

```
- 步长 = 核大小  ⟹  相邻落点的感受野不重叠、刚好平铺整张图
- 每个落点的感受野 = 一个 P×P×C 的块 = 恰好一个 patch
- 该落点的输出 = 把这个 P×P×C 块展平后与卷积核(也是 P×P×C)做内积
- 用 D 个不同的卷积核 ⟹ 每个落点输出 D 维向量
```

把 D 个核各自展平成 P²·C 维行向量、堆成矩阵,它就是上面的 E(转置关系)。每个落点的 P×P×C 感受野展平就是 xₚⁱ。所以"对每个 patch 做线性投影 xₚⁱ·E"和"用 D 个 P×P×C、步长 P 的卷积核扫一遍"算的是**同一件事**。torchvision 和 timm 的 ViT 实现里,patch embedding 这一行**就是直接写 `nn.Conv2d(C, D, kernel_size=P, stride=P)`**——不是巧合,是数学上等价后选了工程上更顺手的写法(省去手动切块+拉平)。

这个等价告诉我们一件深刻的事:**ViT 并不是"完全没有卷积"。它的第一层就是一个 16×16、步长 16 的大卷积。** ViT 真正抛弃的不是"卷积"这个操作,而是 CNN 那种**层层堆叠的小卷积所携带的局部性+多尺度先验**。第一层之后的所有混合,全部交给没有空间先验的全局注意力。

### 2.3 [CLS] token 与位置编码:补回被丢掉的东西

**[CLS] token。** 沿用 BERT 的设计:在 N 个 patch embedding 前面拼一个**可学习的向量** `z_cls ∈ ℝ^D`(它不来自任何 patch,是一个独立的模型参数)。经过 L 层 Transformer 后,**只取这个 [CLS] 位置的输出向量**送进分类头。

为什么需要它?因为分类要一个**全图级**的单一向量,而 Transformer 输出的是 N+1 个 token 向量。需要一种"池化"。[CLS] 的思路是:让模型**自己学**怎么从所有 patch 聚合信息——[CLS] 通过注意力主动去"查询"各个 patch,把分类需要的信息汇聚到自己身上。这比硬性地对所有 patch 做平均(mean pooling)更灵活。**(实践补充:后续研究发现直接对所有 patch token 做全局平均池化,分类效果与 [CLS] 相当甚至略好,很多现代实现两者皆可。这说明 [CLS] 是个可行设计而非必需品。)**

序列长度因此变成 **N+1 = 197**。

**位置编码(positional encoding)。** 这是 ViT 必须补回的第二份先验。注意力对输入顺序是**置换等变(permutation equivariant)**的:打乱 token 顺序,输出只是跟着打乱,注意力本身分不清谁前谁后。但图像 patch 的**空间位置至关重要**——一个 patch 在左上还是右下,语义完全不同。所以必须把位置信息显式加进去:

```
z₀ = [z_cls; xₚ¹E; xₚ²E; …; xₚᴺE] + Eₚₒₛ
                                       Eₚₒₛ ∈ ℝ^{(N+1)×D}  (可学习)
```

`Eₚₒₛ` 是 N+1 个可学习的位置向量,直接**加**到对应 token 上。ViT 用的是**可学习的一维位置编码**——注意,是"一维"。它把 196 个 patch 当成一个长度 196 的序列编号 0,1,...,195,而**没有显式告诉模型"第 14 号和第 0 号 patch 在二维网格里其实是上下相邻的"**。论文做过消融:用二维位置编码(分别编码行、列)相比一维**几乎没有提升**。这是个反直觉但重要的发现:**给足数据后,模型能从一维编码 + 数据里自己学出二维空间关系,不需要你把它硬编码进去。** 这正是后面"归纳偏置之争"的一个伏笔。

(关于绝对 vs 相对、可学习 vs 正弦位置编码的深入讨论见大模型算法课 04 章;这里只需知道原版 ViT 用可学习绝对一维编码。)

### 2.4 Transformer Encoder 与分类头

剩下的就是标准 Transformer encoder,**与 NLP 完全相同**,堆 L 层。每层(用 Pre-LN,即 LayerNorm 放在子层之前——为什么 Pre-LN 见 03 章梯度分析):

```
对 ℓ = 1..L:
    z'ℓ = MSA(LN(zℓ₋₁)) + zℓ₋₁          # 多头自注意力子层 + 残差
    zℓ  = MLP(LN(z'ℓ))  + z'ℓ           # 前馈子层 + 残差
最终:y = LN(zᴸ)
分类:logits = y[0] · W_head            # 只取 [CLS] 位置(下标 0)
```

MSA 是多头自注意力,MLP 是两层前馈(中间维度通常 4D,GELU 激活)。注意一个关键事实:**Transformer encoder 的每一层都是全局的、各向同性的**——每个 token 都能看到所有 token,没有任何"邻近 patch 更重要"的偏好。这与 CNN 逐层扩大感受野形成鲜明对比,是理解 ViT 行为的钥匙。

ViT-Base/16 的标准配置:**L=12, D=768, 头数 h=12, MLP 中间维 3072, patch 16×16,参数量约 86M(待核:常引用值,不同实现的分类头/embedding 略有出入)。**

### 2.5 完整代码:从图像到 logits

下面是一个**可运行、能体现全部机制**的 ViT 分类器最小实现。注意力子层直接用 PyTorch 自带 `nn.MultiheadAttention`(其内部就是 03 章讲的 MHA),把笔墨集中在"视觉适配"的部分。

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """把图像切 patch 并投影成 token —— 本质是一个 kernel=stride=P 的卷积"""
    def __init__(self, img_size=224, patch_size=16, in_ch=3, dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2      # N = (H/P)*(W/P)
        # 关键:kernel=stride=P 的 Conv2d 严格等价于"切块+拉平+线性投影"
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                  # x: (B, C, H, W)
        x = self.proj(x)                   # (B, dim, H/P, W/P) —— 每个落点一个 patch token
        x = x.flatten(2).transpose(1, 2)   # (B, N, dim) —— 拉成 token 序列
        return x

class ViTBlock(nn.Module):
    """标准 Pre-LN Transformer encoder 层(与 NLP 完全相同)"""
    def __init__(self, dim, n_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]   # 自注意力:Q=K=V=x
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, n_classes=1000,
                 dim=768, depth=12, n_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, dim)
        N = self.patch_embed.n_patches
        # [CLS] token:一个独立的可学习参数,不来自任何 patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # 可学习的绝对位置编码,长度 N+1(含 CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):                          # x: (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)                    # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)             # (B, N+1, dim) —— CLS 拼在最前
        x = x + self.pos_embed                     # 加位置编码
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                  # 只取 CLS 位置 → logits
```

读这段代码时盯住三个"视觉适配"动作:`Conv2d(kernel=stride=P)` 完成切块投影、`cls_token` 是个凭空学出的参数、`pos_embed` 把丢失的位置补回来。除此之外,`ViTBlock` 与任何一个文本 Transformer 一模一样。**这正是 ViT 论文的姿态:尽量不改 Transformer,看纯架构能走多远。**

### 2.6 定量分析:复杂度、参数量、感受野

**自注意力的复杂度。** 序列长 n = N+1 ≈ 197,隐藏维 D。一层多头自注意力的计算量主要来自两块(详见 02 章):QKV/输出投影是 O(n·D²),注意力矩阵 (QKᵀ)·V 是 O(n²·D)。

```
单层 MSA FLOPs ≈ 4·n·D²  (Q,K,V,O 四个投影)  +  2·n²·D  (QKᵀ 与 ·V)
对 ViT-B/16:n=197, D=768
  投影项 ≈ 4 · 197 · 768²       ≈ 4.6×10⁸
  注意力项 ≈ 2 · 197² · 768     ≈ 6.0×10⁷
```

**关键观察:在 ViT-B/16 这个配置下,n=197 还很小,所以 O(n²) 的注意力项(6×10⁷)远小于 O(n·D²) 的投影项(4.6×10⁸)——计算瓶颈其实在 MLP 和投影,不在注意力本身。** 但 O(n²) 是一颗定时炸弹:它随分辨率**平方级**增长。

```
分辨率翻倍(224→448),patch 数 N 翻 4 倍(196→784):
  注意力项 ∝ n² → 涨 16 倍
  投影项   ∝ n  → 涨 4 倍
```

所以**一旦做高分辨率密集预测(检测、分割),n 会大到让 O(n²) 彻底主导,ViT 直接爆显存**。这正是 Swin 要解决的核心问题(见第四节)。

**感受野(receptive field)。** 这是 ViT vs CNN 最锋利的对比点。CNN 第 ℓ 层一个神经元的感受野随层数**线性增长**(L 层 3×3 卷积 ≈ (2L+1)×(2L+1)),要覆盖全图需要很多层。ViT 则:

```
ViT 第 1 层之后,每个 token 的"感受野"= 整张图(全局)
理由:第一层自注意力里,每个 token 都对所有 token 做了加权
```

**ViT 从第一层起就是全局感受野。** 这是它建模长程依赖的天然优势,也是它"没有局部性先验"的直接体现——它不假设邻近 patch 更相关,一上来对远近一视同仁。

**参数量。** 主要来自 L 层 encoder。每层 = MSA(4 个 D×D 投影 = 4D²)+ MLP(D×4D 和 4D×D 两个矩阵 = 8D²)≈ 12D² 参数(忽略 LayerNorm 和 bias 等小项)。

```
encoder 参数 ≈ L · 12 · D²
ViT-B/16:12 · 12 · 768² ≈ 8.5×10⁷ ≈ 85M  —— 与常引用的 ~86M 吻合
```

patch embedding(P²·C·D ≈ 768·768 ≈ 0.6M)、位置编码(197·768 ≈ 0.15M)、分类头(768·1000 ≈ 0.77M)都是零头。**99% 的参数在 encoder 里**,与文本 Transformer 一致。

## 三、本章核心:归纳偏置 vs 数据量之争

现在进入本章最重要的一节。前面反复强调 ViT "抛弃了 CNN 的局部性和平移等变先验"。**抛弃先验有什么后果?** 答案不是简单的"更好"或"更差",而是一条**随数据量翻转的曲线**——这是整个深度学习方法论里最有启发性的结论之一。

### 3.1 先验是什么:用偏差-方差重新理解

把归纳偏置放到统计学习的偏差-方差(bias-variance)框架里,一切就清楚了。

```
泛化误差 ≈ 偏差²(bias) + 方差(variance) + 不可约噪声
```

- **强归纳偏置(CNN)** = 把假设空间**收窄**到"符合局部性、平移等变的函数"。如果真实世界确实满足这些假设(自然图像基本满足),那么:**方差小**(假设空间小,有限数据就能定位到好解,不容易过拟合),代价是**偏差可能略大**(如果有些关系不满足局部性,CNN 学不出来或学得费劲)。
- **弱归纳偏置(ViT)** = 假设空间**极大**(几乎任意 token 间的关系都能表达)。后果:**偏差小**(表达力强,能学出 CNN 学不出的全局/非平稳关系),代价是**方差大**(假设空间太大,需要海量数据才能在里面定位到正确的解,否则严重过拟合)。

**一句话总结:归纳偏置是"用偏差换方差"。先验越强,越省数据但天花板越低;先验越弱,越费数据但天花板越高。** 这不是 ViT 独有的现象,是机器学习的普遍规律(著名的"没有免费午餐"定理的一个具体体现:任何让你在某类问题上更省数据的先验,必然在不符合该先验的问题上拖你后腿)。

### 3.2 实验事实:翻转的曲线

ViT 论文最重要的实验,就是把训练数据量从小到大扫一遍,对比 ViT 和 ResNet(BiT,Big Transfer 版的 ResNet):

```
预训练数据规模              谁赢                  原因
─────────────────────────────────────────────────────────────────
ImageNet-1k (~1.3M 图)      ResNet (CNN) 明显胜    数据不足以喂饱无先验的 ViT,ViT 过拟合
ImageNet-21k (~14M 图)      两者接近,ViT 追上来    数据量开始补偿先验的缺失
JFT-300M (~300M 图)         ViT 反超 ResNet         数据足够,ViT 的高表达力天花板兑现
```

**(具体的 top-1 准确率数字此处不逐一列举以免记错——待核;但"小数据 CNN 胜、大数据 ViT 反超"这个定性的交叉(crossover)曲线是论文的核心结论,反复被复现。)**

这条曲线长这样(示意):

```
准确率
  │                                    ╱─── ViT (无先验,天花板高,起步低)
  │                              ╱────
  │                    ╱────────╳ ← 交叉点(~ImageNet-21k 量级)
  │          ╱────────╱
  │  ───────╱  ResNet (强先验,起步高,但很快饱和)
  │
  └──────────────────────────────────────────→ 预训练数据量(对数轴)
     1M        14M              300M
```

**怎么解读这张图:**

1. **小数据区(左)**:ResNet 在上面。ViT 没有局部性先验,只能从数据里"重新发现"卷积早就内置好的东西(邻近像素相关、平移不变)——在只有百万张图时,它学不充分,而且因为假设空间大,它会去拟合训练集的噪声,过拟合。这就是高方差的代价。

2. **交叉点(中)**:大约在千万级数据。先验的"省数据"优势被数据量抵消。

3. **大数据区(右)**:ViT 反超并拉开。ResNet 受限于它的强先验——局部性这个假设虽然大体对,但也排除了一些有用的全局/非局部关系,**它的偏差成了天花板**,再多数据也突破不了。ViT 没有这个枷锁,数据越多,它越能逼近数据里真实存在的复杂函数。

### 3.3 启示:scale 压倒结构先验

这条曲线给出的方法论启示,影响了 2021 年之后整个领域的走向:

**"当数据和算力足够多时,通用的、弱先验的架构 + 大规模数据,会击败精心设计的、强先验的架构。"**

这就是常说的 **"the bitter lesson"(苦涩的教训,Rich Sutton)** 在视觉上的一次具体兑现:几十年里研究者往视觉模型里塞了大量手工设计的结构先验(SIFT、HOG、卷积的各种变体、特征金字塔……),而 ViT 证明,**这些先验在足够数据下不仅没必要,反而是限制**。模型应该从数据里**学**出这些结构,而不是被我们硬塞进去。

但要把这个启示讲诚实,必须立刻补三个限定条件,否则就成了空洞的口号:

- **"足够数据"是个苛刻前提。** JFT-300M 是 Google 内部的私有数据集,3 亿张图。绝大多数团队、绝大多数任务**根本拿不到这个量级的数据**。对他们来说,CNN 的先验仍然是实实在在的优势。"scale 压倒先验"成立的前提是你**付得起 scale**。
- **先验不是消失了,是"溶解"进了数据和训练里。** ViT 在大数据上其实**学出了类似卷积的局部注意力模式**(论文分析过:浅层的注意力头确实倾向于关注邻近 patch)。先验没被消灭,只是从"架构硬编码"变成了"数据中习得"。
- **完全无先验也不是最优。** 后来的 Swin、ConvNeXt 恰恰是把"适量的局部性先验"重新加回 Transformer(或反过来),在中等数据上取得了比纯 ViT 更好的结果。**真正的答案是"先验的量要和数据量匹配",而非"先验越少越好"。**

下一节的三个改进,正好对应这三个限定:DeiT 解决"拿不到 JFT 怎么办",Swin 解决"先验如何适量地加回来以适配密集任务",ConvNeXt 反过来质问"我们以为的架构收益,有多少其实来自训练配方"。

## 四、三大改进:把 ViT 拉回现实

### 4.1 DeiT:在 ImageNet 上就能训出好 ViT

**问题。** ViT 论文的结论很震撼,但有个让人沮丧的前提:**必须有 JFT-300M。** 如果只有 ImageNet-1k 这 128 万张图,ViT 训出来打不过 ResNet。这几乎把 ViT 锁死在了有私有大数据的大厂手里。**DeiT(Data-efficient image Transformer,Touvron 等人,2021)的目标:不用任何外部数据,只用 ImageNet-1k,把 ViT 训到能打。**

**它靠两件事做到——本质都是"在数据有限时,人为往训练里注入正则化和先验":**

**(1) 强数据增强 + 强正则化。** 既然 ViT 因高方差而过拟合,那就用力压方差。DeiT 把一整套当时最猛的增强和正则堆上去:RandAugment、Mixup、CutMix、Random Erasing、Stochastic Depth、强 weight decay、Label Smoothing 等。**直觉:数据增强本身就是在"伪造"更多数据 + 注入不变性先验**——比如随机裁剪/翻转就是在告诉模型"平移、镜像后还是同一类",这恰好把 CNN 内置、ViT 缺失的那些不变性,用数据的方式喂回去。这印证了 3.2 节那条曲线:**你要么用海量真实数据,要么用强增强"伪造"出等效的数据多样性。**

**(2) 蒸馏 token(distillation token)—— DeiT 的真正创新。** 普通知识蒸馏(knowledge distillation)是让学生网络去拟合教师网络的输出软标签。DeiT 把它**架构化**了:除了 [CLS] token,**再加一个独立的 [distillation] token**,与 [CLS] 并列输入,经过同样的 Transformer。训练时:

```
序列 = [CLS] + [DIST] + patch tokens          # 两个特殊 token
[CLS]  的输出 → 与真实标签 y 算交叉熵           # 学 ground truth
[DIST] 的输出 → 与教师(一个 CNN,如 RegNet)的预测算损失   # 学教师
总损失 = ½·L_CE(CLS, y) + ½·L_distill(DIST, teacher)
推理时:把 CLS 和 DIST 两个头的预测平均(或择一)
```

**精妙之处:教师是一个 CNN。** 这意味着 ViT 通过蒸馏 token,**间接吸收了 CNN 的归纳偏置**——CNN 已经把局部性先验学好了,ViT 不用从零在小数据上苦学,直接"抄"CNN 的答案。论文还发现一个有意思的现象:用 CNN 当教师比用另一个 ViT 当教师效果更好,而且硬蒸馏(用教师的 argmax 类别当硬标签)比软蒸馏(拟合 logits 分布)更有效。**这从侧面再次印证:ViT 缺的就是 CNN 的那点先验。**

```python
# DeiT 蒸馏 token 的核心改动(在 2.5 节 ViT 基础上)
class DeiT(ViT):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        dim = self.head.in_features
        # 多加一个蒸馏 token,与 cls_token 并列
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        # 位置编码长度 +1(因为多了一个 token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 2, dim))
        self.head_dist = nn.Linear(dim, self.head.out_features)  # 蒸馏头
        nn.init.trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls  = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls, dist, x], dim=1) + self.pos_embed   # [CLS][DIST] + patches
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        out_cls  = self.head(x[:, 0])        # 用 CLS  → 学真实标签
        out_dist = self.head_dist(x[:, 1])   # 用 DIST → 学 CNN 教师
        if self.training:
            return out_cls, out_dist          # 训练:两个损失分别算
        return (out_cls + out_dist) / 2       # 推理:平均
```

**DeiT 的意义:它把 ViT 从"大厂专属"拉到"人人可训"。** 从此 ViT 进入主流。

### 4.2 Swin Transformer:窗口注意力 + 层次金字塔,适配密集预测

**问题。** ViT 有两个硬伤,让它没法直接用于检测、分割这类**密集预测(dense prediction)**任务:

1. **O(n²) 复杂度**:第二节算过,n 随分辨率平方增长。检测分割要高分辨率,n 巨大,全局注意力直接爆显存。
2. **单一尺度,无层次结构**:ViT 全程在一个固定的低分辨率(14×14)上工作,所有层 token 数不变。但检测分割需要**多尺度特征**——大物体看粗粒度、小物体看细粒度。CNN 天生有特征金字塔(逐层下采样),ViT 没有。

**Swin Transformer(Liu 等人,ICCV 2021 最佳论文)用两个设计同时解决:**

**(1) 窗口注意力(Window-based MSA, W-MSA):把全局注意力关进局部窗口,复杂度从 O(n²) 降到线性。** 这是 Swin 最核心、也是面试/考试最常问的点,必须算清楚。

不再让每个 token 看全图,而是把特征图划分成不重叠的 **M×M 的窗口(Swin 默认 M=7)**,**注意力只在每个窗口内部计算**。

```
设特征图有 h×w 个 token(总 n = h·w),窗口大小 M×M。
全局 MSA 的注意力计算量(只算 QKᵀ 与 ·V 那部分,详见 4.3):
    Ω(MSA) = 2 · n² · D = 2 · (hw)² · D          ← 对 n 是平方

窗口 MSA:每个窗口内 M² 个 token,做窗口内全局注意力;共 (hw)/(M²) 个窗口:
    每个窗口的注意力计算 = 2 · (M²)² · D = 2 · M⁴ · D
    窗口总数 = hw / M²
    Ω(W-MSA) = (hw/M²) · 2·M⁴·D = 2 · hw · M² · D     ← 对 n=hw 是线性!
```

**逐步对比这两个式子,看清"线性"是怎么来的:**

```
Ω(MSA)   = 2 · (hw)² · D          —— 正比于 (hw)²,即 n²
Ω(W-MSA) = 2 · (hw) · M² · D      —— 正比于 (hw)·M²,即 n·(常数)

比值 = Ω(MSA) / Ω(W-MSA) = (hw)² / (hw·M²) = hw / M²
```

**关键洞察:M(窗口大小)是固定常数(7),不随图变大。** 全局注意力里"每个 token 看的对象数"= n,随图增长;窗口注意力里"每个 token 看的对象数"= M²= 49,**永远是常数**。所以总量从 n×n 变成 n×(常数),**复杂度从 O(n²) 降到 O(n)——这就是"线性"的精确含义:对 token 总数 n 线性。** 分辨率翻倍 n 涨 4 倍,W-MSA 计算量只涨 4 倍而非 16 倍。这让 Swin 能吃高分辨率输入。

**(2) 移位窗口(Shifted Window, SW-MSA):让信息跨窗口流动。** 窗口注意力有个明显缺陷:**窗口之间老死不相往来**——一个物体如果横跨两个窗口的边界,模型永远没法把它的左右两半联系起来。这就丢回了 ViT 引以为傲的全局建模能力。

Swin 的解法很巧:**相邻两层用不同的窗口划分。** 第 ℓ 层用规则划分(W-MSA);第 ℓ+1 层把整个窗口网格**向右下平移 ⌊M/2⌋ 个 patch**(SW-MSA),再划分。平移后,新窗口会**横跨**上一层窗口的边界,于是上一层被切开的相邻 token 这一层落进了同一个窗口,信息就流通了。

```
第 ℓ 层(W-MSA):  规则窗口,窗口内注意力,窗口间隔绝
第 ℓ+1 层(SW-MSA):窗口整体右下移 M/2,新窗口跨越旧边界 → 跨窗信息交流
两层一组,交替进行:    ... → W-MSA → SW-MSA → W-MSA → SW-MSA → ...
```

**直觉:每隔一层"错动"一次窗口边界,信息就能像接力一样从一个窗口逐步传遍全图。** 用 L 层,经过若干次错动,感受野就能覆盖整张图——既保住了线性复杂度,又恢复了全局建模能力。(实现上移位会让边缘窗口大小不齐,Swin 用了一个叫 "cyclic shift + masked attention" 的循环移位技巧来高效处理,这里不展开,见论文 3.2 节。)

**(3) 层次化金字塔(hierarchical):像 CNN 一样逐级下采样。** Swin 分 4 个 stage,每过一个 stage 用 **patch merging** 把相邻 2×2 个 token 拼起来做下采样:token 数变 1/4,通道数翻倍。于是特征图分辨率逐级减半:

```
stage 1: H/4  × W/4,  C            (patch embed 用 4×4 patch)
stage 2: H/8  × W/8,  2C           ← patch merging:2×2 合并,通道×2
stage 3: H/16 × W/16, 4C
stage 4: H/32 × W/32, 8C
```

**这给出了和 ResNet 完全一致的多尺度特征金字塔**(1/4, 1/8, 1/16, 1/32),可以无缝接到 FPN、Mask R-CNN 等检测分割框架上。**Swin 的本质:把 CNN 的两个核心先验——局部性(窗口注意力)和多尺度层次(patch merging 金字塔)——重新塞回 Transformer。** 这正是 3.3 节说的"先验的量要和数据量匹配":在拿不到 JFT、又要做密集任务的现实里,适量加回先验是对的。Swin 成了之后几年检测分割的主力骨干网络。

```python
def window_partition(x, M):
    """把 (B, H, W, C) 划分成 (num_windows*B, M, M, C) 的窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // M, M, W // M, M, C)
    # 重排:把窗口维度提到 batch 维,每个窗口独立做注意力
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, M, M, C)
    return windows                                 # (B*num_win, M, M, C)

# Swin block 的骨架(W-MSA 与 SW-MSA 交替)
class SwinBlock(nn.Module):
    def __init__(self, dim, n_heads, window_size=7, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size      # 0 → W-MSA;M/2 → SW-MSA
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x, H, W):            # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:            # SW-MSA:循环移位,让窗口跨越旧边界
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        win = window_partition(x, self.window_size)        # (nW*B, M, M, C)
        win = win.view(-1, self.window_size**2, C)         # (nW*B, M*M, C)
        win = self.attn(win, win, win, need_weights=False)[0]   # 窗口内注意力
        # ...(窗口还原 + 反向移位 roll back,此处略)...
        x = shortcut + x_back
        x = x + self.mlp(self.norm2(x))
        return x
```

### 4.3 ConvNeXt:用 ViT 的训练配方反哺纯 CNN

**问题——也是对前面所有内容的一次"打假"。** ViT/Swin 在 2021 年的成功,被普遍归因于"注意力 / Transformer 架构的优越性"。但 ConvNeXt(Liu 等人,CVPR 2022)提出一个尖锐的怀疑:**Swin 打赢 ResNet,真的是因为'注意力'这个架构吗?还是因为 ViT 时代一起带来的那套现代训练配方(强增强、AdamW、长训练、各种正则)?换句话说,我们把"训练方法的进步"误记成了"架构的进步"。**

ConvNeXt 的做法堪称深度学习领域最干净漂亮的**控制变量实验**之一:**拿一个标准 ResNet-50,什么注意力都不加,一步步把它'现代化',每步只改一处,看准确率怎么变。** 改动分两类:

**(A) 训练配方现代化(不改架构,只改怎么训):** 把 ResNet 的老式训练换成 DeiT/Swin 同款——训 300 epoch(而非 90)、用 AdamW、加 Mixup/CutMix/RandAugment/Random Erasing、Stochastic Depth、Label Smoothing 等。**光是这一步,ResNet-50 的 ImageNet top-1 就从约 76% 提升到约 78%(待核具体数值)——架构一个字没改。** 这是全文最有冲击力的发现:**之前归因于"Transformer 架构"的相当一部分收益,其实来自训练方法。**

**(B) 架构微调(把 ViT/Swin 的宏观/微观设计搬到 CNN 上,但全程仍是纯卷积):**

```
- 调整 stage 计算量比例(仿 Swin 的 1:1:3:1)
- "patchify" stem:把 ResNet 的 7×7 stride-2 卷积换成 4×4 stride-4(就是 ViT 的 patch embed!)
- 用深度可分离卷积(depthwise conv)—— 类比注意力的"按通道/空间分离混合"
- 增大卷积核到 7×7(向窗口注意力的 7×7 感受野看齐)
- 倒置瓶颈(inverted bottleneck,仿 Transformer FFN 的"宽中间层")
- 把 BatchNorm 换 LayerNorm、ReLU 换 GELU、减少归一化和激活的数量(仿 Transformer)
```

每一步加一点点,最终得到的 **ConvNeXt 是一个纯卷积网络**,却在 ImageNet、检测、分割上**全面追平甚至略超同量级的 Swin(待核具体对比数值)**。

**ConvNeXt 的三个结论,把本章的方法论收口:**

1. **"很多收益来自训练而非结构"** ——这是对 3.3 节"scale 压倒先验"的精确化:不光是数据 scale,**训练配方(增强+优化器+正则+训练时长)的进步,贡献被严重低估,常被错记到架构头上。** 做架构对比实验时,**必须对齐训练配方,否则比的是配方不是架构。**

2. **架构选择(卷积 vs 注意力)在现代配方下没那么关键** ——只要训练对了,精心设计的纯 CNN 能打平 Transformer。"注意力是不是必需的"答案是:**对分类这类任务,不是必需的;真正必需的是足够的容量 + 对的训练。**

3. **卷积没有过时** ——卷积的局部性先验在中等数据上仍是优势(省数据、推理高效),ConvNeXt 证明把现代设计回填进 CNN 一样能到 SOTA 附近。

(更深一层:ConvNeXt 揭示了科学方法论的陷阱——**当多个变量同时改变,把效果归因给最显眼的那个变量是很危险的。** ViT 同时带来了"新架构"和"新训练法",社区下意识把功劳全给了架构。ConvNeXt 用控制变量把这笔账重新算清楚了。这个教训远超视觉本身。)

## 五、设计权衡与常见坑

**为什么 patch 大小是 16 而不是 8 或 32?** 纯粹的复杂度-精度权衡。P 越小 → patch 越多 → n 越大 → token 更细粒度、精度通常更高,但 O(n²) 计算量暴涨。P=8 时 n=784,注意力计算量是 P=16 的约 16 倍。P=32 时 n=49,快但太粗,小物体细节全丢。16 是分类任务上精度/算力的甜点。**坑:换分辨率时 patch 数变了,预训练的位置编码长度对不上,必须对 `pos_embed` 做二维插值(把 14×14 的位置编码插值成新尺寸),否则直接报错或精度崩。** timm/torchvision 都内置了 `interpolate_pos_encoding`,自己实现时极易漏掉。

**坑:ViT 在小数据集上从头训,几乎必崩。** 直接拿 ViT 在 CIFAR 或几万张的自定义数据集上 from scratch 训练,会严重过拟合,远不如随手一个 ResNet。这不是 bug,正是 3.2 节那条曲线的左端。**正确姿势:几乎总是用 ImageNet-21k 或更大数据预训练好的权重做微调(fine-tune),不要 from scratch。** 这是 ViT 落地的第一铁律。

**坑:位置编码必须加,且加错会静默降点。** 忘了加 `pos_embed`,模型变成置换不变,理论上分类还能跑(因为是全局池化式的判别),但精度明显下降且你很难一眼看出原因——因为它不报错。**自注意力对顺序不敏感,位置编码是唯一的位置信息来源,必须显式验证它确实被加上、且长度匹配(含 [CLS] 那个 +1)。**

**权衡:[CLS] token vs 全局平均池化。** 如 2.3 节所述,二者分类效果相当。但若要把 ViT 当**特征提取器**用于下游密集任务(检测/分割),你需要的是**每个 patch 的 token**,这时 [CLS] 反而用不上,全局池化或直接取 patch token 更自然。Swin 干脆没有 [CLS]。**选择取决于下游任务是"全图一个标签"还是"每个位置一个预测"。**

**坑:把"ViT 全面优于 CNN"当结论。** 这是本章最想破除的误解。正确的图景是:**没有放之四海皆准的赢家,只有"数据量 × 任务 × 算力预算"决定的局部最优。** 小数据用 CNN 或带先验的 Swin;大数据预训练 + 微调可上 ViT;密集预测用 Swin/层次结构;在乎推理效率和工程简洁性,ConvNeXt 这类现代 CNN 仍极有竞争力。**面试若被问"ViT 比 CNN 好在哪",标准的成熟回答是先反问"多大数据、什么任务",再给翻转曲线,而不是无脑站队。**

## 六、动手练习

**练习 1(推导/计算题):patch embedding 的等价卷积。** 给定输入 224×224×3、P=16、D=768。(a) 写出与该 patch embedding 完全等价的 `nn.Conv2d` 的全部超参(in_channels, out_channels, kernel_size, stride)。(b) 算出这个卷积层的参数量(含 bias),并验证它远小于 encoder 总参数。(c) 若把 P 改成 8,token 数 N 变成多少?一层自注意力的注意力矩阵 (QKᵀ) 那部分计算量相对 P=16 涨几倍?
*提示:(a) 回顾 2.2 节 K=S=P 的等价。(b) 卷积参数 = out·(in·k·k) + out(bias)。(c) N=(224/8)²=784,注意力项 ∝ N²。*

**练习 2(分析题):画出并解释翻转曲线。** 不查资料,凭 3.1 的偏差-方差框架,自己画出"ViT 与 ResNet 准确率随预训练数据量变化"的两条曲线,标出交叉点的大致数据量级,并用三句话分别解释:小数据为何 CNN 赢、大数据为何 ViT 赢、交叉点的本质是什么。再追问自己:如果你的任务只有 5 万张标注图,你选哪个?为什么?
*提示:小数据=高方差主导(ViT 过拟合);大数据=偏差主导(CNN 先验成天花板);交叉点=先验省的数据量 ≈ 已有数据量。5 万张属于曲线最左端。*

**练习 3(推导题):证明窗口注意力的线性复杂度。** 设特征图 h×w 个 token,窗口 M×M。(a) 从"每个窗口内做全局注意力 + 共有 hw/M² 个窗口"出发,推导 W-MSA 中注意力计算量正比于 hw·M²,而全局 MSA 正比于 (hw)²。(b) 用一句话说清"为什么 M 是常数就让它对 n=hw 线性"。(c) 解释移位窗口(SW-MSA)为什么不增加复杂度量级却能恢复跨窗信息流。
*提示:(a) 照搬 4.2 的两个式子,关键是"每窗 M⁴ × 窗口数 hw/M² = hw·M²"。(b) 每个 token 看的对象数从 n 变成常数 M²。(c) 移位只改窗口边界位置,不改窗口大小,故每窗仍 M² 个 token;靠相邻层错动接力传播感受野。*

**练习 4(编码/复现题):给 2.5 节的 ViT 加全局平均池化分支。** 修改 `forward`,同时返回"[CLS] 头"和"对所有 patch token 做平均后过同一分类头"两种预测,在一个小数据集(如 CIFAR-10,记得把 ViT 输入 resize 到 224 或改小 patch)上微调一个预训练 ViT,对比两种池化的准确率。观察是否如 2.3 节所说"二者相当"。
*提示:平均池化 = `x[:, 1:].mean(dim=1)`(跳过下标 0 的 CLS)。务必用预训练权重微调,不要 from scratch(否则触发练习 2 的左端陷阱)。位置编码若尺寸不符记得插值。*

## 七、源码 / 论文导读

- **ViT 原论文**《An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale》(Dosovitskiy 等,ICLR 2021)。重点读:**Section 3.1**(方法,patch embedding + [CLS] + 位置编码的精确定义,对应本章第二节);**Figure 3 / 数据规模消融**(那条翻转曲线,本章第三节的实验来源,务必亲眼看一遍);**附录里关于一维 vs 二维位置编码、不同 pooling 的消融**。
- **DeiT**《Training data-efficient image transformers & distillation through attention》(Touvron 等,2021)。重点:**蒸馏 token 那一节**(本章 4.1 代码对应处),以及"用 CNN 当教师为何更好""硬蒸馏 vs 软蒸馏"的对比。
- **Swin**《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》(Liu 等,ICCV 2021)。重点:**Section 3.2**(W-MSA/SW-MSA 与复杂度公式,本章 4.2 的两个式子直接来自这里;论文给的就是 `Ω(MSA)=4hwC²+2(hw)²C` 和 `Ω(W-MSA)=4hwC²+2M²hwC`——注意它把投影项也写进去了,本章为聚焦把投影项单列了,核心的 (hw)² vs M²hw 对比一致);**循环移位 + masked attention 的高效实现**。
- **ConvNeXt**《A ConvNet for the 2020s》(Liu 等,CVPR 2022)。重点:**那张"从 ResNet 一步步现代化到 ConvNeXt"的路线图(roadmap)图**——每一步加什么、涨多少点,本章 4.3 全部基于它;尤其注意"仅训练配方现代化"那一步的增益。
- **开源实现**:
  - **timm**(`pytorch-image-models`)的 `models/vision_transformer.py`——工业级 ViT 实现,看它的 `PatchEmbed`(确认就是 Conv2d)、`_pos_embed`(看 [CLS] 拼接 + 位置编码插值 `resample_abs_pos_embed`)。`models/swin_transformer.py` 看窗口划分与移位的真实实现。`models/convnext.py` 对照 4.3 的架构改动。
  - **torchvision** `torchvision.models.vision_transformer` 提供更精简的官方 ViT,适合初次通读。
  - **官方 Swin 仓库** `microsoft/Swin-Transformer`,`models/swin_transformer.py` 里的 `WindowAttention`、`cyclic shift` 和 attention mask 是 4.2 那段省略部分的标准答案。

## 八、小结与承上启下

这一章做完一件事:**把 Transformer 搬上图像,看清"放弃结构先验"的代价与回报。**

- **机制**:图像 → 切 patch → 线性投影(本质是 kernel=stride=P 的大卷积)→ 加 [CLS] + 位置编码 → 标准 Transformer encoder → [CLS] 分类。ViT 主动丢掉了 CNN 的局部性、平移等变、多尺度先验,只保留第一层那个 patch 卷积。
- **核心之争**:归纳偏置 = 用偏差换方差。强先验(CNN)省数据但有天花板,弱先验(ViT)费数据但天花板高。实验给出一条**随数据量翻转的曲线**:小数据 CNN 胜,JFT-300M 级 ViT 反超。启示是"scale 在足够数据下压倒结构先验",但必须配三个限定:足够数据是苛刻前提、先验只是溶解进了数据、完全无先验也非最优。
- **三大改进各对应一个现实约束**:DeiT(强增强+蒸馏 token,在 ImageNet 规模就能训,把 ViT 平民化);Swin(窗口注意力把 O(n²) 降到线性 + 层次金字塔 + 移位窗口跨窗交流,适配密集预测——本质是把局部性和多尺度先验适量加回);ConvNeXt(用控制变量证明很多收益来自训练配方而非架构,纯 CNN 现代化后照样 SOTA)。

**在整门课里的位置**:第二节用到的多头注意力、QKᵀ/√dₖ、Pre-LN 残差块,全部来自大模型算法课 02/03 章——本章是它们在视觉的第一次落地。**往后看**:ViT 不只是个分类器,它是**视觉与语言走向统一架构**的起点。后续章节会看到——掩码自编码(MAE)如何让 ViT 像 BERT 一样自监督预训练;CLIP 如何用 ViT 把图像和文本对齐到同一空间,开启开放词表识别;DETR 如何用 Transformer 把检测变成集合预测;以及多模态大模型如何直接把 ViT 当作"视觉的分词器(visual tokenizer)",把图像 patch 当 token 喂进语言模型。**"把图像变成 token 序列"这一步,正是后面所有视觉-语言融合的地基。** 理解了本章的 patch embedding 与归纳偏置之争,你就握住了读懂这条主线的钥匙。
