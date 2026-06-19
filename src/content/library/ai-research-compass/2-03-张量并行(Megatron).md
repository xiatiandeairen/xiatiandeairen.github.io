---
title: "张量并行(Megatron):把一层切到多卡的艺术"
slug: "2-03"
collection: "ai-research-compass"
group: "MLSys专家课程"
order: 2003
summary: "这章把你从\"知道大模型要并行\"带到\"能亲手把一个 Transformer 层的每个矩阵乘法切到 N 张卡上,并精确算出它每一步要通信多少字节、为什么必须锁死在单机内\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T06:26:40.000Z"
updatedAt: "2026-06-19T06:26:40.000Z"
---
> 这章把你从"知道大模型要并行"带到"能亲手把一个 Transformer 层的每个矩阵乘法切到 N 张卡上,并精确算出它每一步要通信多少字节、为什么必须锁死在单机内"。学完你应该能默写 Megatron 的 f/g 共轭算子、独立推导任意 batch/隐藏维下的 all-reduce 通信量,并看懂 `megatron.core.tensor_parallel` 的列并行/行并行实现。

## 一、动机:数据并行的尽头,与"切层"的必然

上一章我们把**数据并行(data parallelism)**讲透了:每张卡放一份完整模型,各吃一片数据,反向后用 Ring All-Reduce 同步梯度。它的前提是**一份完整模型能塞进一张卡**。

这个前提在百亿参数尺度就破了。复习一下显存账本(第 01 章):用 Adam + 混合精度训练,**每个参数约占 16 字节**(FP16 参数 2 + FP16 梯度 2 + FP32 参数副本 4 + 一阶动量 4 + 二阶动量 4)。一个 175B 的 GPT-3,光参数+梯度+优化器状态这三大类(共 16 字节/参数)就是 175 × 10⁹ × 16 ≈ **2.8 TB**。一张 A100 只有 80 GB。**不是慢的问题,是根本放不下。**

放不下,办法只有两类:

- **把模型状态切开存,但每张卡仍算完整的层** —— 这是 ZeRO / FSDP 的路子(第 05 章)。它切的是"存储",计算前临时把分片 all-gather 回来拼成完整权重再算。
- **把单个算子(单个矩阵乘法)本身切到多张卡上,谁都不持有完整权重,也谁都不单独算出完整结果** —— 这就是**张量并行(tensor parallelism,TP)**,也叫**层内并行(intra-layer parallelism)**。

张量并行是 Megatron-LM(NVIDIA,2019)的标志性贡献。它的母题用一句话概括:

> **一个矩阵乘法 Y = X · A,把权重 A 沿某个维度切成 N 片分给 N 张卡,让每张卡只算一部分,再用尽可能少的通信把结果拼回去。**

听起来直白,难点全在最后半句——**"尽可能少的通信"**。矩阵乘法可以沿行切也可以沿列切,切法不同,中间结果的形状不同,需要的通信原语和通信次数也完全不同。Megatron 的精妙之处,是**把一个 Transformer 层里两个相邻的矩阵乘法配对设计切法,让中间结果天然对齐、无需通信,只在层的边界做一次同步**。这就是本章要讲透的"艺术"。

先建立一个贯穿全章的直觉锚点:**张量并行省的是显存,花的是通信,而且花的是最贵的那种通信。** 数据并行一个 step 只在反向末尾通信一次梯度;张量并行**在前向和反向的每一层都要通信激活值**。后面会算出:这通信量大到只能在单机 8 卡的 NVLink 上跑,一旦跨机走 InfiniBand 就会被带宽掐死。理解这一点,你才理解为什么工业界的 3D 并行(第 06 章)里,TP 永远被锁在最内层、不跨节点。

预备知识就两条,系统出身的人都是现成的:**矩阵分块乘法**(把大矩阵切成块,块之间做乘加),和**集合通信原语**(all-reduce / all-gather / reduce-scatter,上一章已讲)。我们从分块乘法的代数开始。

## 二、可分性:矩阵乘法到底能怎么切

### 问题:Y = X · A 有几种切法

设输入激活 `X` 形状是 `[b, k]`(b 是 batch×seq 展平后的 token 数,k 是输入维度),权重 `A` 形状 `[k, n]`,输出 `Y = X · A` 形状 `[b, n]`。我们手上有 N 张卡(TP 组大小,记作 `tp`,这里先设 N=2 方便推导)。

权重矩阵 A 有两个维度可以切:**列切(column)** 和 **行切(row)**。两种切法的代数后果完全不同,这是整章的基石,必须从分块乘法定义推一遍。

### 推导一:列并行(沿输出维 n 切)

把 A 沿**列方向**(输出维 n)切成两块:`A = [A₁ , A₂]`,其中 A₁、A₂ 各是 `[k, n/2]`。卡 1 拿 A₁,卡 2 拿 A₂。每张卡都持有完整的输入 X(这是前提,记住)。各自计算:

```
卡1:  Y₁ = X · A₁     形状 [b, n/2]
卡2:  Y₂ = X · A₂     形状 [b, n/2]

拼接: Y = [Y₁ , Y₂]   形状 [b, n]   （沿列方向并排）
```

为什么这成立?展开矩阵乘法的列定义:`Y` 的第 j 列 = X 与 A 的第 j 列做乘加。A 的前 n/2 列只决定 Y 的前 n/2 列,后 n/2 列只决定后 n/2 列——**输出的不同列之间互不耦合**。所以把 A 按列分给两张卡,各算各的输出列,**计算阶段零通信**。结果 Y 的两半分别躺在两张卡上,**沿列方向被切开**(术语:Y 是"列分片"的,column-parallel)。

关键性质:**列并行的输入是完整的(replicated),输出是切开的(sharded along n)。** 要想得到完整的 Y,需要一次 all-gather 把 Y₁、Y₂ 拼起来——但 Megatron 的精妙在于**故意不拼**,后面解释。

### 推导二:行并行(沿输入维 k 切)

把 A 沿**行方向**(输入维 k)切成两块,上下叠放:`A = [A₁ ; A₂]`(分号表示竖向堆叠),A₁、A₂ 各是 `[k/2, n]`。要让维度对得上,**输入 X 也必须沿列(k 维)切**成 `X = [X₁ , X₂]`,X₁、X₂ 各是 `[b, k/2]`。卡 1 拿 X₁ 和 A₁,卡 2 拿 X₂ 和 A₂。各自计算:

```
卡1:  Y₁ = X₁ · A₁     形状 [b, n]   （完整形状,但只是部分和）
卡2:  Y₂ = X₂ · A₂     形状 [b, n]   （完整形状,但只是部分和）

求和:  Y = Y₁ + Y₂     形状 [b, n]
```

为什么是相加?这是分块乘法的"内积沿 k 维分段求和"性质。展开 Y 的第 (i,j) 个元素:

```
Y[i,j] = Σ_{t=1..k}  X[i,t] · A[t,j]
       = Σ_{t=1..k/2} X[i,t]·A[t,j]  +  Σ_{t=k/2+1..k} X[i,t]·A[t,j]
       =        (X₁·A₁)[i,j]          +        (X₂·A₂)[i,j]
```

也就是**沿收缩维(contraction dimension,这里是 k)切开后,每张卡算出的是完整尺寸 `[b,n]` 的一个"部分和(partial sum)",真正的结果要把所有部分和加起来**。这个加法跨越了两张卡,**必须用 all-reduce(sum)**。

关键性质:**行并行的输入是切开的(sharded along k),输出是完整的(经 all-reduce 后 replicated),代价是一次 all-reduce。**

### 把两个推导并排:这是 Megatron 的全部秘密

```
                输入             权重切法          每卡输出          得到完整输出的代价
列并行(column)   完整 X          A=[A₁,A₂] 按列     Y_i=[b, n/2] 分片   all-gather(沿 n 拼)
行并行(row)      分片 X=[X₁,X₂]   A=[A₁;A₂] 按行     Y_i=[b, n] 部分和   all-reduce(求和)
```

盯着这张表看十秒,你会发现一个**完美的接力**:

> **列并行的输出形状是"沿 n 维分片",而行并行需要的输入形状恰好是"沿 k 维分片"。如果让第二个矩阵的输入维 k = 第一个矩阵的输出维 n,那么列并行的输出可以直接当行并行的输入,中间不需要任何通信!**

Megatron 把 MLP 的两个线性层、注意力的 QKV 投影与输出投影,全都按这个接力设计。下面具体推。

## 三、MLP block:一次前向一个 all-reduce 的推导

### 问题:Transformer MLP 的结构

标准 Transformer 的 MLP(也叫 FFN,feed-forward network)是两层线性夹一个非线性:

```
Z = GELU(X · A)          A: [h, 4h]    第一层,把维度 h 升到 4h
Y = Z · B                B: [4h, h]    第二层,把 4h 降回 h
```

h 是隐藏维(hidden size),中间维通常是 4h。GELU 是逐元素(element-wise)激活函数。我们要把 A、B 都切到 N 张卡上,目标:**整个 MLP block 前向只通信一次。**

### 推导:第一层列切,第二层行切

**第一层 A 按列切**:`A = [A₁ , A₂]`。每张卡持有完整输入 X(replicated),算 `Z_i = X · A_i`,得到沿中间维(4h)分片的 Z。

这里有个**非平凡的关键点,新手最容易在这里出错**:Z 算完要过 GELU,GELU 能不能各卡各算?**能,且仅仅因为 GELU 是逐元素函数。**

```
GELU 逐元素:  GELU(Z)[i,j] 只依赖 Z[i,j] 一个标量,不跨列耦合
所以:        GELU([Z₁, Z₂]) = [GELU(Z₁), GELU(Z₂)]    ✅ 各卡独立算,零通信
```

**反例警告**:如果中间的非线性是 **softmax 或 LayerNorm 这类沿着被切维度做归约的算子,就不能各算各的**——softmax 要对整行求和,LayerNorm 要对整个特征维求均值方差,而那个维度恰好被切开了,各卡只有一部分,算出来的归一化是错的。Megatron 能把 MLP 切得这么干净,**本质前提是 GELU 的逐元素性**。这个 trade-off 后面"常见坑"还会强调。

GELU 之后,中间激活 `Z' = GELU(X·A)` 仍然沿 4h 维分片,记 `Z' = [Z'₁ , Z'₂]`。

**第二层 B 按行切**:`B = [B₁ ; B₂]`,B₁、B₂ 各 `[2h, h]`。它需要的输入恰好是沿 4h(收缩维)分片的——而 Z' 正好就是这个形状!**接力成功,中间零通信。** 各卡算部分和:

```
卡1:  Y₁ = Z'₁ · B₁    形状 [b, h]   部分和
卡2:  Y₂ = Z'₂ · B₂    形状 [b, h]   部分和
最终:  Y = Y₁ + Y₂                    需要 all-reduce
```

整个 MLP block 前向,从输入完整的 X 到输出完整的 Y,**只在最后做了一次 all-reduce**。中间 X·A、GELU、Z'·B 三步全部零通信。这就是论文里那张图的精髓。

### f 与 g:一对共轭算子(本章最该背下来的概念)

Megatron 把这个通信结构抽象成两个算子,插在 MLP 的首尾:

```
X ──[ f ]──> 各卡列并行 A、GELU、行并行 B ──[ g ]──> Y
```

- **f**:前向是**恒等(identity)**——直接把完整的 X 广播给所有卡(它们本来就该持有完整 X);反向是 **all-reduce**——梯度从各卡汇总。
- **g**:前向是 **all-reduce**——把各卡的部分和 Y_i 求和成完整 Y;反向是**恒等(identity)**——把完整的输出梯度直接发给各卡。

为什么 f 和 g 的前向/反向是**反过来**的?这不是巧合,是**自动微分的对偶性(duality)**的必然结果。给一个严格的小推导:

```
设前向某算子对张量做线性映射:  Y = M · X    (M 是某个通信/复制操作)
反向(链式法则)传播梯度:        ∂L/∂X = Mᵀ · (∂L/∂Y)

关键事实:复制(broadcast)与求和(all-reduce/sum)互为转置 (transpose)。

直觉证明(以复制为例):
  复制算子:把 1 份 X 变成 N 份相同副本,矩阵形式 M = [I; I; ...; I] (N 个 I 竖叠)
  其转置:   Mᵀ = [I, I, ..., I] (横排),作用在 N 份梯度上 = 把 N 份梯度相加
  => 复制的转置 = 求和。 反之求和的转置 = 复制。    ∎

所以:
  f 前向 = identity(复制/不动) ⟹ f 反向 = all-reduce(求和)
  g 前向 = all-reduce(求和)   ⟹ g 反向 = identity(复制/不动)
```

**这对算子是张量并行通信结构的"原子"**。整个 Megatron 框架,无论 MLP 还是 attention,都是用 f、g 这对共轭算子在层边界缝合的。记住这张图,你就抓住了 TP 的骨架:

```
            前向          反向
  f         identity      all-reduce
  g         all-reduce    identity      （恰好与 f 镜像对称）
```

**结论:一个 MLP block,前向 1 次 all-reduce(g 的前向),反向 1 次 all-reduce(f 的反向)。** 注意 f 前向和 g 反向都是 identity,不产生通信。所以一个 MLP block 一次完整训练迭代(前向+反向)共 **2 次 all-reduce**。

### 代码:列并行 / 行并行 Linear 的机制实现

下面是能体现机制的核心伪代码(PyTorch 风格,省去 init 噪声)。重点看 `forward` 里的通信原语和 `backward` 里的镜像通信。Megatron 实际是用 `torch.autograd.Function` 自定义前后向,这里把前后向都显式写出来便于看清 f/g。

```python
# ---- 通信原语(假设 tp_group 是张量并行通信组)----
def all_reduce(x):      # 各卡张量求和,结果广播回所有卡(原地)
    dist.all_reduce(x, op=SUM, group=tp_group); return x
def identity(x):
    return x            # 不做通信

# ---- f 算子:前向 identity,反向 all-reduce ----
class CopyToTPRegion(autograd.Function):
    @staticmethod
    def forward(ctx, x):   return identity(x)        # 前向:不动(X 本就 replicated)
    @staticmethod
    def backward(ctx, g):  return all_reduce(g)      # 反向:汇总各卡梯度

# ---- g 算子:前向 all-reduce,反向 identity ----
class ReduceFromTPRegion(autograd.Function):
    @staticmethod
    def forward(ctx, x):   return all_reduce(x)      # 前向:部分和求和成完整结果
    @staticmethod
    def backward(ctx, g):  return identity(g)        # 反向:不动

# ---- 列并行 Linear:输入完整,输出沿 n 维分片 ----
class ColumnParallelLinear(nn.Module):
    def __init__(self, k, n, tp):
        # 只持有自己那一片权重 A_i,形状 [k, n/tp]
        self.A_i = nn.Parameter(empty(k, n // tp))
    def forward(self, X):
        X = CopyToTPRegion.apply(X)   # f:进入并行区,前向 identity
        Y_i = X @ self.A_i            # 本卡部分输出 [b, n/tp],计算无通信
        return Y_i                    # 不 gather,保持分片(交给下游行并行接力)

# ---- 行并行 Linear:输入沿 k 维分片,输出完整 ----
class RowParallelLinear(nn.Module):
    def __init__(self, k, n, tp):
        # 只持有自己那一片权重 B_i,形状 [k/tp, n]
        self.B_i = nn.Parameter(empty(k // tp, n))
    def forward(self, Z_i):           # 输入已是沿 k 分片(来自上游列并行)
        Y_partial = Z_i @ self.B_i    # 本卡部分和 [b, n],计算无通信
        Y = ReduceFromTPRegion.apply(Y_partial)  # g:all-reduce 求和
        return Y                      # 完整输出

# ---- 组装 MLP:列并行 -> GELU(逐元素,零通信) -> 行并行 ----
class TPMLP(nn.Module):
    def __init__(self, h, tp):
        self.fc1 = ColumnParallelLinear(h, 4*h, tp)   # A 按列切
        self.fc2 = RowParallelLinear(4*h, h, tp)      # B 按行切
    def forward(self, X):
        Z  = self.fc1(X)              # [b, 4h/tp] 分片
        Zp = gelu(Z)                  # 逐元素,各卡独立,零通信 ★
        Y  = self.fc2(Zp)             # 内部一次 all-reduce -> [b, h] 完整
        return Y                      # 整个 block:前向仅 1 次 all-reduce
```

照着这段代码,你能在一个 8 卡机器上跑通最小 TP-MLP。注意三个工程要点:(1) `ColumnParallelLinear` **故意不 all-gather**,把分片直接喂给下游;(2) `gelu` 那行**绝对不能换成会跨被切维归约的算子**;(3) f/g 用 `autograd.Function` 实现,保证反向自动镜像。

## 四、注意力层:按 head 切的天然并行

### 问题:多头注意力为什么适合 TP

多头注意力(multi-head attention,MHA)的结构:输入 X 经三个投影得到 Q、K、V,拆成 `a` 个头(head),每个头独立做 scaled-dot-product attention,各头输出拼接后过一个输出投影 O。

```
Q = X·W_Q,  K = X·W_K,  V = X·W_V      W_*: [h, h],拆成 a 个头,每头维度 d = h/a
对每个头 i:  head_i = softmax(Q_i·K_iᵀ / √d) · V_i
拼接:        Concat(head_1, ..., head_a)
输出:        Y = Concat(...) · W_O       W_O: [h, h]
```

**关键观察:不同的 head 之间完全独立。** head_i 的计算只用到 Q_i、K_i、V_i,从不与别的头交叉(softmax 是在每个头内部、沿 seq 维做的,不跨头)。这意味着**头(head)就是天然的并行切分维度**——比 MLP 还干净,因为连"逐元素"的假设都不需要,头本身就是结构性独立的。

### 推导:QKV 投影列切,输出投影行切

把 `a` 个头按 TP 组大小 `tp` 均分,每张卡负责 `a/tp` 个头。

- **W_Q、W_K、W_V 按列切**:列方向就是头的方向(每 d 列是一个头)。卡 i 持有它负责那些头对应的 Q/K/V 投影权重列。算出的 Q_i、K_i、V_i 只含本卡的头。这是**列并行**(输入完整 X,输出按头分片)。
- **本卡独立完成它那几个头的完整 attention**(softmax 在头内沿 seq 做,本卡有完整的 seq,所以 softmax 可以本地算,零通信)。这一点和 MLP 不同也更优:**attention 的 softmax 不跨被切的维度(头维),所以不破坏可分性**。
- **W_O 按行切**:输出投影的输入维(收缩维)正好是拼接后的头维度,而各卡持有的正是分片的头输出——又一次接力。这是**行并行**(输出部分和,需 all-reduce)。

于是 attention block 的通信结构和 MLP **完全同构**:

```
X ──[ f ]──> 列并行 QKV ──> 本卡各头 attention(含 softmax,本地)──> 行并行 W_O ──[ g ]──> Y
```

**结论:一个 self-attention block,前向 1 次 all-reduce,反向 1 次 all-reduce,与 MLP 一致。**

### 工程约束:head 数必须能被 tp 整除

`a / tp` 必须是整数,否则没法均分头。这是个**硬约束,在选择 tp 时第一个要检查**。比如 LLaMA-2 70B 有 `a=64` 个头(待核:具体头数以官方配置为准),那么 `tp ∈ {1,2,4,8,...}` 都能整除;但如果某模型 `a=12`,则 `tp=8` 就非法。GQA/MQA(分组查询注意力)下 K/V 的头数更少,切分要分别处理 Q 头数和 KV 头数,这是 LLaMA 系工程里的常见坑(细节待核,见各模型 attention 实现)。

```python
class TPSelfAttention(nn.Module):
    def __init__(self, h, num_heads, tp):
        assert num_heads % tp == 0, "head 数必须能被 tp 整除"  # ★ 硬约束
        self.local_heads = num_heads // tp
        self.d = h // num_heads
        # QKV 合成一个列并行 Linear(实现里常融合成一个 [h, 3h] 再列切)
        self.qkv = ColumnParallelLinear(h, 3 * h, tp)   # 输出按头分片
        self.o   = RowParallelLinear(h, h, tp)          # 行并行,内含 all-reduce
    def forward(self, X):
        qkv = self.qkv(X)                 # [b, 3h/tp] 分片(只含本卡的头)
        q, k, v = split_heads(qkv, self.local_heads, self.d)
        # 本卡独立算它负责的头,softmax 沿 seq 维(本地完整),零通信
        out = flash_attention(q, k, v)    # [b, local_heads*d]
        Y = self.o(out)                   # 一次 all-reduce -> [b, h] 完整
        return Y
```

## 五、量化分析:通信量为什么大到只能待在单机

这是本章的"硬核账本"。前面定性说"TP 通信贵",现在精确算。

### 单次 all-reduce 的通信量

设隐藏维 `h`,一个 microbatch 的 token 总数 `s = b × L`(b 是 batch size,L 是序列长度,展平后 token 数)。激活张量是 `[s, h]`。Megatron 通常用 FP16/BF16,**每个元素 2 字节**。

上一章证过:Ring All-Reduce 对一个总大小为 `V` 字节的张量,**每张卡的总收发量约 `2(N-1)/N · V ≈ 2V`(N 较大时)**。这里"通信量"我们统一指**单卡的链路传输字节数**(因为瓶颈是单卡的链路带宽)。

一个激活张量 `[s, h]` 的字节数:

```
V_act = s × h × 2 字节 = b × L × h × 2
单次 all-reduce 单卡传输量 ≈ 2 × V_act = 4 · b · L · h  字节   (N 较大近似)
```

### 一个 Transformer 层、一次迭代的总通信量

每个 Transformer 层 = 1 个 attention block + 1 个 MLP block。每个 block 前向 1 次、反向 1 次 all-reduce。所以:

```
每层 all-reduce 次数 = (attention: 前1 + 反1) + (MLP: 前1 + 反1) = 4 次
```

整个模型 `Lyr` 层,一次训练迭代(前向+反向)的 all-reduce 次数:

```
总 all-reduce 次数 = 4 × Lyr
```

单卡总通信字节(每次 all-reduce 都搬一个 `[s,h]` 激活):

```
C_TP = 4 × Lyr × (2 × V_act)
     = 4 × Lyr × 2 × (b·L·h·2)
     = 16 · Lyr · b · L · h   字节   （单卡,每次迭代,FP16,N 较大近似）
```

**代入 GPT-3 175B 量级感受一下(数量级估算)**:`Lyr=96, h=12288, b=1, L=2048`:

```
C_TP ≈ 16 × 96 × 1 × 2048 × 12288
     ≈ 16 × 96 × 2.5×10⁷
     ≈ 3.9 × 10¹⁰ 字节 ≈ 39 GB   （单卡,每次迭代,前向+反向合计）
```

**39 GB 的通信量,每个 step 都要搬,而且分散在 4×96=384 次 all-reduce 里**,每次还卡在计算的关键路径上(必须等通信完成才能进下一层)。

### 与数据并行对比:差几个数量级

数据并行一个 step 只在反向末尾 all-reduce **一次梯度**,通信量 ≈ `2 × 模型参数字节`。对 175B、梯度 FP16(2 字节):`2 × 175×10⁹ × 2 ≈ 700 GB`?——看起来比 TP 大,但**注意两个本质区别**:

1. **频率与位置**:DP 一个 step 通信 1 次,且可以**与反向计算重叠(overlap)**(梯度算完一层就能开始传那层,见第 02、14 章);TP 一个 step 通信 384 次,且**几乎无法重叠**——下一层的输入依赖上一层 all-reduce 的完整输出,是**串行依赖**。
2. **可扩展性**:DP 的通信量与 batch 无关、与卡数无关(Ring 算法);TP 的通信量正比于 batch×seq,且每次都堵在关键路径。

**真正致命的是延迟(latency),不是总字节。** TP 的 all-reduce 是**小消息、高频率、关键路径**,对**带宽和延迟双敏感**。这就引出本章最重要的工程结论。

### 为什么必须锁在单机 NVLink 内(带宽推导)

做一个**关键路径占比估算**,看通信时间能不能被容忍。

```
一次 all-reduce 的耗时 ≈ 通信量 / 链路带宽   （忽略延迟项,只看带宽项)
单次通信量(单卡)≈ 2 × V_act = 4·b·L·h 字节

NVLink(节点内,A100 用 NVSwitch 全互联):带宽量级 ~数百 GB/s
  例:单向有效带宽约 300 GB/s 量级 (NVLink3/A100,确切数值待核)
InfiniBand(跨节点):带宽量级 ~数十 GB/s
  例:200 Gb/s ≈ 25 GB/s (HDR IB,确切数值待核)

二者相差约 10× 量级。
```

把单次 all-reduce 时间和单层计算时间比一比(数量级)。单层 MLP 的主要 FLOPs ≈ `2 × 2 × s × h × 4h = 16·s·h²`(两个矩阵乘,前向;`s=b·L`),A100 FP16 算力量级几百 TFLOPS。当 `h` 很大时计算时间 ∝ h²,通信时间 ∝ h(单次 `4·b·L·h`)。**h 越大,计算越能摊薄通信,TP 越划算**——但前提是用 NVLink。

定性结论(这是工业界铁律,务必记住):

> **张量并行的频繁 all-reduce 在 NVLink(节点内)上勉强能被计算掩盖一部分,但一旦跨节点走 InfiniBand,带宽掉一个数量级、延迟涨好几倍,通信会彻底主宰耗时,TP 的扩展效率断崖式下跌。因此 TP 度数(tp)几乎永远 ≤ 单节点 GPU 数(典型是 8),绝不跨机。** 跨机的并行交给流水线并行(第 04 章,通信稀疏)和数据并行(第 02 章,可重叠)。

这条结论是 3D 并行(第 06 章)拓扑设计的根基:**TP 放最内层(节点内),PP 放中层(节点间但通信稀疏),DP 放最外层(节点间但可重叠)**。

### 显存账:TP 到底省了多少

TP 把每个被切的权重矩阵的参数、梯度、优化器状态都切成 `1/tp`。但要注意**不是所有东西都被切**:

```
被 tp 切分(变成 1/tp):
  - MLP 的 A、B 权重及其梯度、优化器状态
  - attention 的 W_Q/K/V、W_O 权重及其梯度、优化器状态
  - 这些是参数的大头,所以参数显存 ≈ 原来的 1/tp

不被 tp 切分(每卡仍是完整一份,replicated):
  - LayerNorm 的 γ、β(很小,但要注意)
  - embedding 视实现而定(Megatron 也对 vocab 维做并行,vocab-parallel,细节待核)
  - 激活值:列并行的中间激活 [b,4h/tp] 被切了,但进出层边界的 [b,h] 激活是完整的
```

所以**参数/优化器显存基本按 1/tp 缩小,但激活显存只是部分缩小**(层边界激活仍完整)。这也是为什么 TP 单独用还不够,要配合激活重计算(第 08 章)和 ZeRO/FSDP(第 05 章)对 DP 维度的优化器状态再分片。

## 六、设计权衡与常见坑

**权衡 1:省显存 vs 烧通信带宽。** TP 是用最贵的通信(高频、关键路径、跨卡 all-reduce)换显存。**只在"模型一层都放不下单卡"或"要降低单卡激活压力"时才上,且度数尽量小**(够放下即可)。能用 ZeRO 解决的显存问题,优先用 ZeRO(通信更可重叠)。

**权衡 2:tp 越大,矩阵越小,GPU 利用率越低。** 把 `[h, 4h]` 切成 `tp` 份后,每卡算的是 `[h, 4h/tp]`。`tp` 太大时单卡矩阵太"瘦",触发不了 Tensor Core 的高效区间(矩阵维度要够大、对齐 16/128 的倍数),算力利用率(MFU)反而下降。**tp 不是越大越好,8 通常是节点内上限也是经验甜点。**

**坑 1(最致命):在被切的维度上放了会"跨维归约"的算子。** 重申第三节:GELU 能各卡各算**仅因为它逐元素**;一旦在 4h(被切维)上做 softmax/LayerNorm/任何 reduction,各卡只有分片就会算错。**自己写 TP 层时,任何沿被切维的归约都必须先 all-gather 或换成不切那一维**。Megatron 的 LayerNorm 故意放在不被切的 h 维上、且 LN 权重不切,就是为了规避。

**坑 2:Dropout 的随机性必须对齐。** TP 区内各卡处理同一份数据的不同分片,如果某些地方(比如 attention 的 dropout)各卡用了不同随机种子,会导致**逻辑上同一个张量的不同分片用了不一致的 mask**,结果错误。Megatron 用 **tensor-parallel 专用的 RNG state**,区分"各卡应相同的随机性"和"各卡应不同的随机性"(`model_parallel_cuda_manual_seed` 之类机制,确切 API 待核)。这是新手自己实现 TP 时极易忽略的正确性 bug。

**坑 3:权重初始化与 checkpoint 的切分一致性。** 因为每卡只存权重的一片,**初始化时要保证"切开后再拼起来"等价于"完整矩阵的初始化"**(否则数值分布变了);存 checkpoint 时也要约定切分/合并规则(load 时按 tp 重新切)。Megatron 有专门的 checkpoint 切分/合并工具,迁移到不同 tp 度数要做 reshard。

**坑 4:误以为 TP 像 DP 一样能线性加速吞吐。** TP **不增加有效 batch**(同一份数据被多卡协作处理一层),它换来的是"放得下"和"单卡激活/计算量下降",throughput 受通信拖累常常**亚线性**。要扩 batch 提吞吐是 DP 的活。

**坑 5:忘了 all-reduce 在反向也有。** 新手常只数前向那次 all-reduce,漏掉 f 在反向、g 配套的通信。**正确计数:每个 block 前向 1 次 + 反向 1 次 = 2 次;每层 2 个 block = 4 次。**

**序列并行(sequence parallelism)补充(进阶,Megatron 后续工作)**:注意到层边界那些"完整的 [b,h] 激活"和 LayerNorm/Dropout 是 replicated 的,Megatron-LM 的后续工作(Korthikanti et al. 2022,待核)把这部分沿 **序列维(L)** 也切开,与 TP 配合,进一步省激活显存,且把部分 all-reduce 换成 all-gather + reduce-scatter(通信总量不变但更省激活)。这是第 08 章激活优化的伏笔。

## 七、动手练习

**练习 1(推导题,核心)。** 一个 GPT 模型有 `Lyr` 个 Transformer 层,每层含 1 个 attention block 和 1 个 MLP block,用张量并行度 `tp` 训练。
(a) 推导一次训练迭代(前向 + 反向)总共触发多少次 all-reduce,写出与 `Lyr` 的关系。
(b) 设隐藏维 `h`、batch `b`、序列长 `L`、激活 FP16(2 字节),用 Ring All-Reduce 近似(单卡传输 ≈ 2V),推导单卡每次迭代的总通信字节数 `C_TP` 的闭式表达式。
(c) 代入 `Lyr=96, h=12288, b=1, L=2048`,估算 `C_TP`(GB),并说明为什么这个量级决定了 tp 不能跨机。
*提示*:每个 block 前向 1 次 all-reduce(g 前向)、反向 1 次(f 反向);f 前向和 g 反向是 identity 不通信。一个激活张量是 `[b·L, h]`。

**练习 2(编码题)。** 基于第三节的 `ColumnParallelLinear` / `RowParallelLinear` 伪代码,在单机 2 GPU 上(`torch.distributed`,`nccl` 后端)实现一个最小可跑的 `TPMLP`,并验证:对同一份输入,TP 版输出与"单卡完整权重直接算"的输出在数值上一致(误差 < 1e-3,FP32 下)。
*提示*:关键是 (1) 把完整权重 A 按列、B 按行切给两张卡(初始化时用同一个 full 权重切,保证可比);(2) f/g 用 `autograd.Function` 显式实现;(3) 用 `dist.all_reduce` 做 g 的前向;(4) 对比时 gather 一下结果。先不验证反向,跑通前向数值一致即可。

**练习 3(估算题)。** 接练习 1 的模型与配置。假设 NVLink 单向有效带宽 300 GB/s(待核)、InfiniBand 25 GB/s(待核),忽略延迟项,只用"通信时间 = 通信量 / 带宽"。
(a) 估算单次 all-reduce(单卡传输 ≈ 2 × `b·L·h·2` 字节)在 NVLink 上和 IB 上各耗时多少微秒/毫秒。
(b) 单层 MLP 前向计算量约 `16·b·L·h²` FLOPs,设 A100 FP16 有效算力 150 TFLOPS(待核),估算单层 MLP 前向计算时间。
(c) 比较 (a)(b):在 NVLink 下通信时间占计算时间的比例,在 IB 下又是多少?用这个比例解释"TP 跨机不可行"。
*提示*:注意通信 ∝ h,计算 ∝ h²,所以比值 ∝ 1/h——h 越大 TP 越划算,这正是大模型才适合 TP 的原因。

**练习 4(概念题)。** 解释为什么把 MLP 设计成"第一层列切、第二层行切",而不是"第一层行切、第二层列切"。后者会发生什么?需要几次通信?
*提示*:行切的第一层要求输入 X 沿 k 维分片,但 MLP 的输入 X 是从上一层 attention 出来的完整激活——它是 replicated 的,不是分片的。强行行切第一层,你得先把 X 切开(reduce-scatter 或本地切),最后还要把列切的输出 gather,数一数总通信次数,看是否比"列→行"多。结论:**"列→行"让中间分片天然衔接,通信最少;反过来要在层内额外通信。**

## 八、源码 / 论文导读

**论文(精读)**：
- **Shoeybi et al., 2019, "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"**。重点读 **Section 3(Model Parallel Transformers)**:MLP 的列/行切分图、attention 按 head 切分图、f 与 g 算子的定义(论文用 f、g 记号)。这一节就是本章第三、四节的来源,**对照着把那两张通信结构图默画出来**,你就掌握了。
- **Korthikanti et al., 2022, "Reducing Activation Recomputation in Large Transformer Models"**(序列并行 + 选择性重计算,确切标题/作者待核)。读它如何在 TP 基础上沿序列维进一步切,把 replicated 的 LayerNorm/Dropout 激活也分片,以及 all-reduce 如何分解为 all-gather + reduce-scatter。配合第 08 章读。

**开源代码(对照阅读)**：
- **NVIDIA/Megatron-LM**,模块 **`megatron/core/tensor_parallel/`**(确切路径以仓库当前结构为准,待核):
  - `layers.py`:`ColumnParallelLinear`、`RowParallelLinear` 的真实实现——看它怎么用 `autograd.Function` 实现 f/g、怎么处理 bias、怎么做权重初始化的切分。把它和本章伪代码逐行对照。
  - `mappings.py`(或类似):`copy_to_tensor_model_parallel_region`、`reduce_from_tensor_model_parallel_region`、`scatter`/`gather` 等通信原语——这就是 f、g 的工业实现。
  - `random.py`:TP 专用 RNG(对应坑 2 的 Dropout 一致性),看它怎么区分"各卡同/异"的随机性。
- **关注点清单**(读源码时带着这些问题):(1) 列并行为什么默认 `gather_output=False`?(就是本章说的"故意不 gather,留给下游接力");(2) `RowParallelLinear` 的 `input_is_parallel` 参数何时为 True;(3) bias 在列并行/行并行里分别怎么处理(行并行的 bias 只在 all-reduce 后加一次,避免重复)。

## 九、小结与承上启下

把这章浓缩成几条**必须刻进肌肉记忆**的结论:

1. **张量并行 = 把单个矩阵乘法切到多卡**,省的是显存(权重/优化器按 1/tp 缩小),花的是高频关键路径通信。
2. **MLP 切法:第一层列切、GELU 各算、第二层行切**,前向仅 1 次 all-reduce。能这么干**全靠 GELU 逐元素**;遇到沿被切维的归约(softmax/LN)会出错。
3. **attention 切法:按 head 切**,QKV 列并行、O 行并行,与 MLP 同构,前向 1 次 all-reduce。head 数必须被 tp 整除。
4. **f / g 是一对共轭算子**:f 前向 identity / 反向 all-reduce,g 前向 all-reduce / 反向 identity,镜像对称(根源是"复制与求和互为转置")。一层 4 次 all-reduce。
5. **通信量 ∝ Lyr·b·L·h,高频、串行、不可重叠**,所以 **tp 必须锁在单机 NVLink 内,典型 ≤ 8,绝不跨机**。

这章在整门课里的位置:第 02 章(数据并行)解决"扩 batch、扩吞吐",但要求模型放得下;**本章(张量并行)解决"一层都放不下"**;第 04 章(流水线并行)解决"层数太多、跨机怎么并"(通信稀疏、能跨节点);第 05 章(ZeRO/FSDP)从另一个角度——不切计算只切存储——省显存。这四把刀各有适用边界,**第 06 章(3D 并行)会教你把它们正交组合**:TP 锁节点内、PP 跨节点流水、DP 最外层扩规模,并搜索最优并行度配置。本章打下的"f/g 通信原子"和"通信量正比于 b·L·h"两个量化直觉,是第 06 章拓扑设计和第 14 章通信-计算重叠优化的直接前置。下一章我们换一种切法:不切单层,而是把不同层放到不同卡上,看流水线并行如何用 1F1B 调度把"气泡(bubble)"压到最小。


---

## 练习参考答案

> 本章「动手练习」的参考答案(AI 生成,推导/代码已尽量自验,具体数值见「待核」标注)。

### 练习 1:推导一次迭代的 all-reduce 次数与单卡通信字节闭式

**(a) 总 all-reduce 次数**

逐层数清楚通信发生在哪里。每个 block(attention / MLP)用一对共轭算子 f、g 缝合在层边界:

```
        前向          反向
  f     identity      all-reduce      ← 反向才通信
  g     all-reduce    identity        ← 前向才通信
```

所以单个 block 一次完整迭代(前向 + 反向)的 all-reduce 次数:

```
g 前向 1 次(部分和求和) + f 反向 1 次(梯度汇总) = 2 次
（f 前向、g 反向都是 identity,不通信)
```

一个 Transformer 层 = 1 个 attention block + 1 个 MLP block:

```
每层 = 2 个 block × 2 次 = 4 次 all-reduce
```

整个模型 `Lyr` 层:

```
总 all-reduce 次数 = 4 · Lyr
```

**结论:一次训练迭代共 `4·Lyr` 次 all-reduce(前向 `2·Lyr` 次 + 反向 `2·Lyr` 次),与层数线性相关。**

**(b) 单卡每次迭代总通信字节 `C_TP` 的闭式**

第一步,单个激活张量的字节数。一个 microbatch 展平后 token 数 `s = b·L`,激活形状 `[s, h] = [b·L, h]`,FP16 每元素 2 字节:

```
V_act = s · h · 2 = b · L · h · 2   字节
```

第二步,单次 all-reduce 的单卡传输量。Ring All-Reduce 对总大小 `V` 的张量,单卡收发量 ≈ `2(N-1)/N · V`,N 较大时近似 `2V`:

```
单次 all-reduce 单卡传输 ≈ 2 · V_act = 2 · (b·L·h·2) = 4·b·L·h   字节
```

第三步,乘以总次数 `4·Lyr`:

```
C_TP = (4·Lyr) × (2·V_act)
     = 4·Lyr × 4·b·L·h
     = 16 · Lyr · b · L · h   字节
```

**结论(单卡 / 每次迭代 / FP16 / Ring 近似):**

```
C_TP = 16 · Lyr · b · L · h   字节
```

注意它**正比于 `b·L·h` 而非参数量**——这是与数据并行(∝ 参数量)的本质差异,也是 TP 通信特性的根。

**(c) 代入 GPT-3 175B 量级**

`Lyr=96, h=12288, b=1, L=2048`:

```
C_TP = 16 × 96 × 1 × 2048 × 12288
     = 38 654 705 664 字节
     ≈ 3.87 × 10¹⁰ 字节
     ≈ 38.7 GB（十进制,10⁹）  ≈ 36 GiB（二进制,2³⁰）
```

(验算:16×96 = 1536;2048×12288 = 2.516×10⁷;1536×2.516×10⁷ ≈ 3.865×10¹⁰ ✓)

**为什么这个量级决定 tp 不能跨机:**

- 这 ≈ 39 GB **不是一次性大块传输**,而是被切碎成 `4×96 = 384` 次小 all-reduce,每次只搬一个 `[b·L,h]` 激活(本例单次单卡 ≈ 100 MB)。
- 每一次 all-reduce 都**卡在计算关键路径上**:下一层的输入依赖上一层 all-reduce 出的完整输出,是**串行依赖,几乎无法与计算重叠**。
- 因此 TP 对**带宽和延迟双敏感**。NVLink/NVSwitch(节点内,~数百 GB/s)勉强能让通信被计算掩盖一部分;一旦跨节点走 InfiniBand(~数十 GB/s),带宽掉一个数量级、延迟涨数倍,384 次串行通信会彻底主宰耗时。

**结论:正因为 `C_TP` 高频(384 次)、串行、不可重叠,tp 几乎永远 ≤ 单节点 GPU 数(典型 8),绝不跨机。**

---

### 练习 2:单机 2 GPU 实现最小 TPMLP 并验证前向数值一致

思路:用同一个 full 权重 `A[h,4h]`、`B[4h,h]` 切给两卡(A 按列、B 按行),保证可与单卡完整计算对比;f/g 用 `autograd.Function` 显式实现;g 前向用 `dist.all_reduce` 求和;最后比对 TP 输出与单卡完整输出。

依赖与运行:需 2 块 GPU + `torch`(`nccl` 后端)。保存为 `tp_mlp.py`,运行 `torchrun --nproc_per_node=2 tp_mlp.py`。验证 FP32 下误差 < 1e-3。

```python
# tp_mlp.py  —— 单机 2 GPU,验证 TPMLP 前向与单卡完整计算数值一致
# 运行: torchrun --nproc_per_node=2 tp_mlp.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


# ---- f / g 共轭算子(显式前后向)----
class CopyToTPRegion(Function):           # f: 前向 identity, 反向 all-reduce
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)   # 汇总各卡梯度
        return grad


class ReduceFromTPRegion(Function):       # g: 前向 all-reduce, 反向 identity
    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)      # 部分和求和成完整结果
        return x
    @staticmethod
    def backward(ctx, grad):
        return grad


# ---- 列并行 Linear: 输入完整, 输出沿 n 维分片 ----
class ColumnParallelLinear(nn.Module):
    def __init__(self, weight_shard):     # weight_shard: [k, n/tp](从 full 切来)
        super().__init__()
        self.A_i = nn.Parameter(weight_shard)
    def forward(self, X):
        X = CopyToTPRegion.apply(X)       # f
        return X @ self.A_i               # [b, n/tp] 分片, 不 gather


# ---- 行并行 Linear: 输入沿 k 维分片, 输出完整 ----
class RowParallelLinear(nn.Module):
    def __init__(self, weight_shard):     # weight_shard: [k/tp, n]
        super().__init__()
        self.B_i = nn.Parameter(weight_shard)
    def forward(self, Z_i):
        Y_partial = Z_i @ self.B_i        # [b, n] 部分和
        return ReduceFromTPRegion.apply(Y_partial)    # g: all-reduce -> 完整


class TPMLP(nn.Module):
    def __init__(self, A_shard, B_shard):
        super().__init__()
        self.fc1 = ColumnParallelLinear(A_shard)
        self.fc2 = RowParallelLinear(B_shard)
    def forward(self, X):
        Z = self.fc1(X)                   # [b, 4h/tp] 分片
        Zp = torch.nn.functional.gelu(Z)  # ★ 逐元素, 各卡独立, 零通信
        return self.fc2(Zp)               # 内含 1 次 all-reduce -> [b, h] 完整


def main():
    rank, world = setup()
    torch.manual_seed(0)                  # 同种子 -> 各卡造出相同的 full 权重和输入
    dtype = torch.float32                 # FP32 验证数值一致
    b, h = 4, 128
    hidden = 4 * h

    # 完整权重 / 输入(各卡用同种子生成,保证完全相同的"参照系")
    A_full = torch.randn(h, hidden, dtype=dtype, device="cuda")   # [h, 4h]
    B_full = torch.randn(hidden, h, dtype=dtype, device="cuda")   # [4h, h]
    X      = torch.randn(b, h, dtype=dtype, device="cuda")        # [b, h] replicated

    # 切分: A 按列(dim=1), B 按行(dim=0)
    A_shard = A_full.chunk(world, dim=1)[rank].contiguous()       # [h, 4h/tp]
    B_shard = B_full.chunk(world, dim=0)[rank].contiguous()       # [4h/tp, h]

    tp_mlp = TPMLP(A_shard, B_shard).cuda()
    with torch.no_grad():
        y_tp = tp_mlp(X)                  # 各卡得到相同的完整 [b, h]

        # 单卡完整参照: Y = GELU(X @ A_full) @ B_full
        y_ref = torch.nn.functional.gelu(X @ A_full) @ B_full

    max_err = (y_tp - y_ref).abs().max().item()
    if rank == 0:
        print(f"max abs error = {max_err:.3e}  -> {'PASS' if max_err < 1e-3 else 'FAIL'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

要点说明:

- **可比性的关键**:各卡用相同的 `manual_seed` 生成同一份 `A_full / B_full / X`,再按 rank 各取一片,等价于"把同一个完整权重切开",这样 TP 结果才能和 `y_ref` 严格对齐。
- **GELU 必须逐元素**(代码里 `gelu(Z)`):它作用在沿 4h 分片的 `Z` 上各卡独立算,正确性来自逐元素性;换成 softmax/LayerNorm 这类沿被切维归约的算子,结果会错(见练习 4 与"坑 1")。
- **g 的 all-reduce 是原地操作**;为简洁用了同一份 `A_full/B_full` 不切作 reference,FP32 下两条路径应满足误差 < 1e-3。

**无 GPU 时的最小可跑骨架(CPU 单进程,验证切分代数正确)**:不依赖 `torch.distributed`,把 all-reduce 退化为对各卡部分和的显式求和,可在普通 CPU 上验证"列切→行切"的代数等价:

```python
import torch
torch.manual_seed(0)
b, h, hidden, tp = 4, 128, 512, 2
A = torch.randn(h, hidden); B = torch.randn(hidden, h); X = torch.randn(b, h)
y_ref = torch.nn.functional.gelu(X @ A) @ B            # 单卡完整

A_sh = A.chunk(tp, dim=1)                               # 列切
B_sh = B.chunk(tp, dim=0)                               # 行切
parts = [torch.nn.functional.gelu(X @ A_sh[i]) @ B_sh[i] for i in range(tp)]
y_tp = sum(parts)                                       # all-reduce 的语义 = 求和
print("max err =", (y_tp - y_ref).abs().max().item())  # ~1e-5 量级
```

---

### 练习 3:NVLink vs InfiniBand 的通信/计算比例,解释 TP 跨机不可行

沿用 `Lyr=96, h=12288, b=1, L=2048`,带宽 NVLink 300 GB/s、IB 25 GB/s(均待核),A100 FP16 有效算力 150 TFLOPS(待核),忽略延迟项。

**(a) 单次 all-reduce 耗时**

单次 all-reduce 单卡传输量:

```
2 × V_act = 2 × (b·L·h·2) = 4·b·L·h
          = 4 × 1 × 2048 × 12288
          = 100 663 296 字节 ≈ 100.7 MB
```

时间 = 传输量 / 带宽:

```
NVLink:  100.7 MB / 300 GB/s ≈ 3.36 × 10⁻⁴ s ≈ 336 μs（约 0.34 ms）
IB:      100.7 MB /  25 GB/s ≈ 4.03 × 10⁻³ s ≈ 4 027 μs（约 4.03 ms）
```

(IB 正好是 NVLink 的 12 倍,因为带宽差 12 倍。)

**(b) 单层 MLP 前向计算时间**

计算量 `16·b·L·h²`:

```
16 × 1 × 2048 × 12288² = 4.948 × 10¹² FLOPs ≈ 4.95 TFLOP
```

时间 = FLOPs / 算力:

```
4.948 × 10¹² / 150 × 10¹² ≈ 3.30 × 10⁻² s ≈ 33.0 ms
```

**(c) 通信/计算比例,与跨机结论**

单层 MLP 前向恰好含 1 次 all-reduce(g 前向),直接对比:

```
            单次 all-reduce    占 MLP 前向计算(33.0 ms)的比例
  NVLink    0.336 ms           ≈ 1.0%      ← 可被计算轻松掩盖
  IB        4.03 ms            ≈ 12.2%     ← 已开始显著拖累
```

**比例为何能解释"TP 跨机不可行":**

1. **1/h 标度直觉**(对应提示):通信 ∝ h(单次 `4·b·L·h`),计算 ∝ h²(`16·b·L·h²`),比值 ∝ `通信/计算 ∝ 1/h`。h 越大,计算越能摊薄通信——**这正是只有大模型才适合 TP 的原因**。验算:h 砍半(6144)时 NVLink 占比从 1.0% 翻到 2.0%、IB 从 12.2% 翻到 24.4%,精确翻倍 ✓。
2. **上面 12.2% 还是乐观估计**(用了 full-matrix 计算量、且单卡)。真实 TP 中权重已被切,**单卡计算量是 1/tp,但 all-reduce 的单卡传输量基本不随 tp 变**。所以 tp 越大比例越糟。取实战 tp=8:

```
  tp=8 时单卡 MLP 前向计算 ≈ 33.0/8 ≈ 4.12 ms
  NVLink AR 0.336 ms -> 占比 ≈ 8.1%   (仍可接受)
  IB     AR 4.03 ms  -> 占比 ≈ 97.7%  (通信几乎吃掉全部计算时间!)
```

3. 再叠加 IB 的**延迟项**(本题忽略,但实际 384 次小消息每次都付一份延迟)和**串行不可重叠**(下一层等本层 all-reduce 完成),跨机的实际损耗比上面更严重。

**结论:NVLink 下通信仅占个位数百分比,能被计算掩盖;跨机 IB 下(尤其 tp=8)通信占比逼近 100%,扩展效率断崖式下跌。因此 TP 必须锁在单机 NVLink 内,跨机的并行交给通信稀疏的流水线并行和可重叠的数据并行。**

---

### 练习 4:为什么 MLP 是"列切→行切"而非"行切→列切"

**先回到 MLP 的输入条件。** MLP 的输入 X 来自上一层 attention 的 g 算子输出,经过 all-reduce 后是**完整的(replicated)** `[b, h]`,每张卡都持有同一份完整 X。这个"输入是完整的"是整个切法选择的出发点。

**正确方案:列切 → 行切(为何零中间通信)**

```
第一层列切:  输入要求 = 完整 X  ✓(恰好满足!直接用,不通信)
              输出 Z = [b, 4h/tp] 沿中间维分片
GELU:         逐元素,各卡独立,零通信
第二层行切:  输入要求 = 沿收缩维(4h)分片  ✓(Z 正好是这个形状,接力成功!)
              输出 = [b, h] 部分和 -> g 做 1 次 all-reduce
```

中间 X·A、GELU、Z·B 三步**全部零通信**,只在 block 末尾 1 次 all-reduce。两层的"输出分片形状"与"输入需求形状"天然咬合,这正是 Megatron 的核心设计。

**反向方案:行切 → 列切(会发生什么)**

```
第一层行切:  输入要求 = 沿 k(=h)维分片的 X
              但实际 X 是完整 replicated 的 ✗ 不匹配!
              => 必须先把完整 X 切开:本地 slice(便宜)或 reduce-scatter
              输出 = [b, 4h] 部分和 -> 需要 all-reduce(① 第 1 次通信)
GELU:         必须作用在"完整的 4h 激活"上才正确(逐元素无所谓维度,
              但下一步列切需要完整输入,所以 GELU 前必须先把部分和 all-reduce 成完整)
第二层列切:  输入要求 = 完整激活(上一步 all-reduce 后才完整)
              输出 = [b, h/tp] 沿输出维分片
              但 MLP 的最终输出必须交给下一层(下一层 attention 的输入需完整)
              => 必须 all-gather 拼回完整 [b, h](② 第 2 次通信)
```

**通信次数对比(前向):**

```
列切→行切:  1 次 all-reduce(block 末)
行切→列切:  ≥ 2 次(中间 all-reduce 把部分和拼完整 + 末尾 all-gather 拼输出),
            外加把输入 X 切开的额外操作
```

**根因:**

- "列→行"让前一层输出的**分片形状**正好等于后一层需要的**输入分片形状**(都在中间维 4h 上),衔接处零通信;同时列切的输入需求(完整)与行切的输出形态(完整)恰好分别匹配 MLP 的"完整输入/完整输出"边界条件。
- "行→列"两头都拧着:开头要把完整 X 拆成分片,中间要把部分和拼完整,结尾还要把列切的分片输出 gather 回完整,**每一处错配都换来一次额外的层内通信**。

**结论:"列切→行切"使中间激活的分片在层内天然接力衔接,前向只需 1 次 all-reduce;反过来"行切→列切"会在层内额外引入至少 1 次通信(总计 ≥ 2 次),且需要额外切分输入。Megatron 选择"列→行"正是为了把通信压到层边界的唯一一次。**
