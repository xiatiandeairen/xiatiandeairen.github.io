---
title: "FlashAttention 从零:IO 感知注意力的完整推导"
slug: "2-09"
collection: "ai-research-compass"
group: "MLSys专家课程"
order: 2009
summary: "这一章把你从\"知道 attention = softmax(QKᵀ)V\"带到\"能算清它为什么慢、能从访存第一性原理推出 FlashAttention、能照着伪代码写出一个前向 kernel\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T06:26:40.000Z"
updatedAt: "2026-06-19T06:26:40.000Z"
---
> 这一章把你从"知道 attention = softmax(QKᵀ)V"带到"能算清它为什么慢、能从访存第一性原理推出 FlashAttention、能照着伪代码写出一个前向 kernel"。读完你应当能向别人证明:标准注意力是 memory-bound 的,而 FlashAttention 没有改变 FLOPs,只是把 HBM 往返次数砍掉一个量级。

## 为什么需要这一章:慢在哪不是显然的

很多人对 attention 的第一直觉是"它是 O(N²) 的,序列越长越慢,因为算得多"。这个直觉**只对了一半,而且错的那一半恰恰是 FlashAttention 的全部立足点**。

真相是:在 2020 年前后的 GPU 上,长序列注意力的瓶颈不在"算 N² 次乘法",而在"把那个 N×N 的中间矩阵在显存(HBM)和芯片之间来回搬"。算力(FLOP/s)的增长速度,在过去十几年里远远超过了显存带宽(byte/s)的增长。结果是:**对很多算子,GPU 大部分时间在等数据,而不是在算**。注意力就是典型受害者。

FlashAttention 的洞察因此不是一个数值技巧,而是一个系统观察:**如果我们能不把那个 N×N 矩阵写回 HBM,就能省掉绝大部分访存,从而直接变快——即使总的乘加次数一点没少。** 这就是"IO-aware"(IO 感知)的含义:把算法围绕"数据在存储层级之间怎么流动"重新设计。

要把这件事讲透,我们必须先学会一件基本功:**给一个 GPU 算子做访存预算**。这是 MLSys 工程师区别于"会调 API 的人"的分水岭。所以本章的推进路线是:

1. 给标准注意力精确算 FLOPs 和 HBM 字节数,用 Roofline 证明它 memory-bound;
2. 推导 online softmax——分块后看不到整行还能正确归一化的数学核心;
3. 用 tiling 把整个"打分-softmax-加权"过程压在片上 SRAM 里完成;
4. 推导 FlashAttention 的 IO 复杂度 Θ(N²d²/M),解释为什么省;
5. 讲反向为什么靠"重算"省显存;
6. 梳理 FA-2 / FA-3 的演进:它们在解决什么新瓶颈。

### 预备:GPU 的存储层级与两个数

后面所有推导都建立在两个硬件事实上,先建立直觉。

**存储层级。** 现代 NVIDIA GPU 大致分三层(自上而下越来越快、越来越小):

- **HBM(High Bandwidth Memory)**:就是俗称的"显存",几十 GB,带宽以 TB/s 计。慢、大。
- **SRAM / on-chip memory**:每个 SM(流多处理器)上的片上内存,包括 shared memory 和 register file,总量以 MB 计(单个 SM 上的 shared memory 是几十到几百 KB 量级)。快、极小。
- **寄存器**:线程私有,最快。

关键比例:**SRAM 带宽比 HBM 带宽高一个数量级,但容量小三四个数量级**。具体数字(如 A100 SRAM 总量、各级带宽)随架构变化,见文末「待核」清单,但"快小 vs 慢大"这个结构是稳定的。FlashAttention 的全部努力,就是让数据尽量待在 SRAM 里被反复用,少碰 HBM。

**两个我们要算的量:**

- **FLOPs**:浮点运算次数。一次乘加(multiply-add)算 2 FLOP(一乘一加)。
- **HBM 访存字节数**:从 HBM 读、往 HBM 写的总字节。这是本章的主角。

判断一个算子是 compute-bound 还是 memory-bound,用**算术强度(arithmetic intensity)**:

```
算术强度 I = FLOPs / HBM访存字节数   (单位:FLOP/byte)
```

它的物理意义是"每从 HBM 搬 1 个字节,能喂多少次浮点运算"。GPU 有一个由硬件决定的**临界算术强度 I_crit = 峰值算力 / 峰值带宽**(单位也是 FLOP/byte)。判据是:

- 若 I < I_crit:算子 **memory-bound**,实际吞吐被带宽卡住,堆算力没用;
- 若 I > I_crit:算子 **compute-bound**,实际吞吐被算力卡住。

这就是 **Roofline 模型**的核心:横轴算术强度,纵轴可达性能,左边是带宽决定的斜坡(性能 = I × 带宽),右边是算力决定的水平天花板,拐点在 I_crit。现代数据中心 GPU 的 I_crit 通常在**一两百 FLOP/byte 量级**(具体值随架构,见「待核」)。记住这个数量级,下面会用它给注意力定性。

## 第一步:给标准注意力做访存预算

### 问题设定与记号

单头注意力(single-head),前向。记:

- N:序列长度(token 数)
- d:每个 head 的维度(head dimension),常见 64 / 128
- Q, K, V ∈ ℝ^{N×d}:query / key / value 矩阵

标准三步:

```
S = Q Kᵀ          # S ∈ ℝ^{N×N},打分矩阵(scores)
P = softmax(S)    # 按行做 softmax,P ∈ ℝ^{N×N},注意力权重
O = P V           # O ∈ ℝ^{N×d},输出
```

(为聚焦访存,这里省掉 1/√d 缩放和因果 mask,它们不改变量级结论。)

### FLOPs 计算

逐步数乘加。一次矩阵乘 A_{m×k} · B_{k×n} 需要 m·n·k 次乘加 = 2·m·n·k FLOP。

```
S = Q Kᵀ :  Q 是 N×d, Kᵀ 是 d×N        → 乘加 N·N·d, FLOP = 2N²d
P = softmax(S) : 每个元素几次 exp/加/除   → O(N²),常数小,量级 N²
O = P V :  P 是 N×N, V 是 N×d           → 乘加 N·N·d, FLOP = 2N²d
─────────────────────────────────────────────────────────
总 FLOP ≈ 4N²d  (softmax 的 N² 项相对 N²d 可忽略,因为 d≫1)
```

**结论一:标准注意力前向约 4N²d FLOP,主导项来自两次矩阵乘 QKᵀ 和 PV。** 这个数字 FlashAttention 一分不少——它不优化 FLOPs。

### HBM 访存字节数计算

这才是关键。设每个浮点数 b 字节(FP16/BF16 是 2,FP32 是 4)。我们老老实实追踪每个张量进出 HBM 的字节(标准实现把 S、P 都物化到 HBM):

```
读 Q, K, V                : 3 · N·d · b
写 S = QKᵀ 到 HBM          : N² · b
读 S 回来做 softmax        : N² · b
写 P = softmax(S) 到 HBM   : N² · b
读 P 回来做 PV             : N² · b
读 V(做 PV 时再读一次)    : N·d · b   (实现相关,可能已缓存,量级不变)
写 O                       : N·d · b
─────────────────────────────────────────────────
HBM 字节 ≈ (4N² + 5Nd) · b  ≈ 4N²·b   (当 N ≫ d 时 N² 项主导)
```

不同框架对 S/P 的物化次数略有差异(有的把 softmax 拆成 max、sub、exp、sum、div 多个 kernel,每个 kernel 都要把 N×N 矩阵读出再写回,N² 往返次数更多)。**但结论稳定:标准注意力的 HBM 访存被 Θ(N²) 项主导,而且系数不止 1——那个 N×N 矩阵要在 HBM 里被来回搬好几趟。**

**结论二:标准注意力 HBM 访存 = Θ(N²·b),N×N 中间矩阵的反复读写是访存大头。**

### 算术强度与 Roofline 判定

把两个数一除:

```
I = FLOPs / HBM字节
  ≈ 4N²d / (4N²·b)
  = d / b
```

代入常见值:d = 128,FP16 即 b = 2:

```
I ≈ 128 / 2 = 64  FLOP/byte
```

**结论三(本节高潮):标准注意力的算术强度约为 d/b,与序列长度 N 无关,典型值在几十 FLOP/byte 量级,显著低于 GPU 的临界强度 I_crit(一两百 FLOP/byte 量级)。因此标准注意力是 memory-bound——它实际跑多快由 HBM 带宽决定,瓶颈在搬那个 N×N 矩阵,不在算。**

这个结论有两个让人不舒服但极其重要的推论:

1. **堆更多 Tensor Core(算力)对标准注意力几乎无用**,因为算力本来就没吃满。
2. **既然瓶颈是搬 N×N 矩阵,那么"不搬它"就能直接变快**——这正是 FlashAttention 要做的,而且它不需要任何新硬件。

注意 I ≈ d/b 与 N 无关这一点很微妙:它说明无论序列多长,标准注意力的"效率"都卡在同一个 memory-bound 点上,长序列只是把绝对的访存量 N² 推得更大、把问题暴露得更明显(还会因为 N² 显存直接 OOM)。

## 第二步:online softmax 的完整推导

FlashAttention 想分块计算,但马上撞到一堵墙:**softmax 是按整行归一化的,而分块后我们一次只能看到一行的一段。** 这一节解决这堵墙,它是整个算法的数学心脏。

### 问题:为什么 softmax 不能简单分块

对一行打分向量 x = (x₁, …, x_N),softmax 第 i 个分量是:

```
softmax(x)_i = exp(xᵢ) / Σⱼ exp(xⱼ)
```

分母 Σⱼ exp(xⱼ) 要看**整行**才能算出来。如果我们把这一行切成两块 x = [x^(1), x^(2)],在只看到 x^(1) 时,分母还缺 x^(2) 的贡献,无法给出最终权重。更糟的是数值问题:exp(xⱼ) 在 xⱼ 稍大时就溢出(exp(89) 在 FP32 已 overflow)。

### 安全 softmax:先减最大值

实践中从不直接算 exp(xⱼ),而是减去该行最大值 m = maxⱼ xⱼ,因为:

```
exp(xᵢ) / Σⱼ exp(xⱼ)
 = exp(xᵢ - m) / Σⱼ exp(xⱼ - m)      # 分子分母同乘 exp(-m),恒等
```

减完之后所有指数参数 ≤ 0,exp 落在 (0,1],绝不溢出。**这一步是恒等变形,不改变结果,只为数值安全。** 但它把"需要全行信息"的问题加重了:现在连 m 都要看全行。

### 核心:running max / running sum 与重缩放

思路:**边扫描边维护两个标量统计量,每来一个新块就把旧的累加结果"修正"成仿佛一开始就用新最大值算的样子。**

定义在"已经看过前 t 个块"时的两个量(对某一固定行):

- mₜ:已见元素的最大值(running max)
- ℓₜ:已见元素在"减去 mₜ 之后"的 exp 之和(running sum / 归一化分母)

来一个新块,块内最大值记为 m̃,块内"减块内最大值"的 exp 之和记为 ℓ̃。新的全局最大值:

```
m_new = max(m_old, m̃)
```

现在的麻烦:ℓ_old 是"减 m_old"算出来的,ℓ̃ 是"减 m̃"算出来的,**两者基准不同,不能直接相加**。解决办法是用一个**重缩放因子**把它们拉到同一个基准 m_new。

对任意一项 exp(xⱼ - m_old),要换算成 exp(xⱼ - m_new),只需乘 exp(m_old - m_new):

```
exp(xⱼ - m_new) = exp(xⱼ - m_old) · exp(m_old - m_new)
```

所以整段旧和乘同一个因子即可。于是**核心递推式**:

```
m_new = max(m_old, m̃)
ℓ_new = exp(m_old - m_new) · ℓ_old  +  exp(m̃ - m_new) · ℓ̃
```

第一项:把旧的归一化和重缩放到新基准;第二项:把新块的和重缩放到新基准后并入。因为 m_old ≤ m_new 且 m̃ ≤ m_new,两个 exp 因子都 ≤ 1,数值安全。

### 把输出 O 也一起在线更新

光有分母不够,我们要的是输出 O = P V,即加权的 V 之和。定义"未归一化的输出累加":

```
Õₜ = Σ_{j 在前 t 块}  exp(xⱼ - mₜ) · v_j      # 一个 d 维向量,还没除以 ℓ
```

它和 ℓ 一样依赖当前基准 mₜ,所以**每次换基准时要同样重缩放**。来新块时(块内 exp 用块内最大值算,得到块内贡献 Õ̃ = Σ exp(xⱼ - m̃)·v_j):

```
Õ_new = exp(m_old - m_new) · Õ_old  +  exp(m̃ - m_new) · Õ̃
```

最后扫完所有块,做一次除法得到真正的输出:

```
O = Õ_final / ℓ_final
```

**这就是 online softmax 的全部:三个递推量 (m, ℓ, Õ),每步用 exp(m_old - m_new) 重缩放旧值再并入新块。** 注意 Õ 是 d 维向量、m 和 ℓ 是标量,所以维护它们只占很小的片上空间,与 N 无关——这正是能 tiling 的前提。

### 证明:逐块更新与一次性 softmax 数学等价

要让人放心,得证明这套递推扫完整行后给出的 O,与"先看全行再 softmax 再乘 V"逐位相等。用归纳法。

**命题。** 设把整行切成块 1…T。定义 Mₜ = max{所有前 t 块的 xⱼ},以及

```
Lₜ = Σ_{j∈前t块} exp(xⱼ - Mₜ)
Sₜ = Σ_{j∈前t块} exp(xⱼ - Mₜ) · v_j
```

则上面的递推满足 mₜ = Mₜ、ℓₜ = Lₜ、Õₜ = Sₜ 对所有 t 成立。

**归纳基(t=1)。** 第一块直接初始化:m₁ = M₁(块内即全局)、ℓ₁ = L₁、Õ₁ = S₁。成立。

**归纳步。** 假设 t-1 时 mₜ₋₁ = Mₜ₋₁、ℓₜ₋₁ = Lₜ₋₁、Õₜ₋₁ = Sₜ₋₁。来第 t 块,块内 m̃ = max(第 t 块)、ℓ̃ = Σ_{j∈块t} exp(xⱼ - m̃)、Õ̃ = Σ_{j∈块t} exp(xⱼ - m̃)·v_j。则:

```
mₜ = max(mₜ₋₁, m̃) = max(Mₜ₋₁, max(块t)) = Mₜ                     ✓(max 的结合性)

ℓₜ = exp(mₜ₋₁ - mₜ)·ℓₜ₋₁ + exp(m̃ - mₜ)·ℓ̃
   = exp(Mₜ₋₁ - Mₜ)·Σ_{前t-1块} exp(xⱼ - Mₜ₋₁)
        + exp(m̃ - Mₜ)·Σ_{块t} exp(xⱼ - m̃)
   = Σ_{前t-1块} exp(xⱼ - Mₜ)                # 指数相乘:exp(a-b)exp(x-a)=exp(x-b)
        + Σ_{块t} exp(xⱼ - Mₜ)
   = Σ_{前t块} exp(xⱼ - Mₜ) = Lₜ                                     ✓

Õₜ 同理,把上式每一项末尾乘 v_j 即得 Õₜ = Sₜ                        ✓
```

归纳完成。扫到 t = T 时 ℓ_T = L_T = Σ_全行 exp(xⱼ - M)、Õ_T = S_T = Σ_全行 exp(xⱼ - M)·v_j,于是

```
O = Õ_T / ℓ_T = ( Σ_j exp(xⱼ - M)·v_j ) / ( Σ_j exp(xⱼ - M) )
  = Σ_j softmax(x)_j · v_j
```

**这正是一次性 softmax 再乘 V 的结果,逐位相等(浮点舍入误差除外)。** 所以 FlashAttention 是**精确注意力**,不是近似——这点和各种线性 attention / 稀疏 attention 有本质区别,值得反复强调。

> online softmax 的最初出处是 NVIDIA 的 Maxim Milakov & Natalia Gimelshein 的 "Online normalizer calculation for softmax"(2018)。FlashAttention 的贡献是把它和 tiling、IO 分析、kernel fusion 接到一起。

## 第三步:tiling——把全过程压在片上

有了 online softmax,分块就只剩工程。核心思想:**把 Q、K、V 切成能塞进 SRAM 的小块,在片上完成"打分→softmax→加权"的整条链,中间的 N×N 块算完即弃,绝不写回 HBM。**

### 分块布局

外层按 **Q 的行块**切,内层按 **K/V 的行块**切。设:

- Q 切成 T_r = ⌈N / B_r⌉ 个行块,每块 B_r 行(B_r × d)
- K、V 各切成 T_c = ⌈N / B_c⌉ 个行块,每块 B_c 行(B_c × d)

对每一个 Q 块 Qᵢ(B_r × d),我们要算出对应的输出块 Oᵢ(B_r × d)。算法是:**把 Qᵢ 常驻 SRAM,依次把每个 Kⱼ、Vⱼ 块载入 SRAM,算出局部打分块 Sᵢⱼ = Qᵢ Kⱼᵀ(B_r × B_c),对它做"块内 softmax 统计 + online 合并",更新 Oᵢ、mᵢ、ℓᵢ。** 内层扫完所有 j,Oᵢ 就是最终输出块,一次性写回 HBM。

注意现在最大的中间张量是 Sᵢⱼ,大小 B_r × B_c —— **与 N 无关,只和块大小有关**。这就是省 HBM 的根源:完整的 N×N 从不存在于 HBM,只有一个个 B_r×B_c 的小块在 SRAM 里转瞬即逝。

### SRAM 容量约束如何决定块大小

片上要同时容纳哪些东西?在内层一次迭代里,SRAM 里大致有:Qᵢ(B_r×d)、Kⱼ(B_c×d)、Vⱼ(B_c×d)、打分块 Sᵢⱼ(B_r×B_c)、输出累加 Oᵢ(B_r×d),外加 mᵢ、ℓᵢ(B_r 个标量,小)。总量(以元素计,乘 b 字节):

```
SRAM 占用 ≈ B_r·d + B_c·d + B_c·d + B_r·B_c + B_r·d
         = O( (B_r + B_c)·d + B_r·B_c )   个元素
```

设 SRAM 可用容量为 M 字节,约束是上式 × b ≤ M。FlashAttention 论文里取的策略是让块大小和 d 挂钩,典型地令 **B_c ≈ Θ(M / (b·d))**,并对 B_r 取 min(同样的量, d) 之类的上界。直觉:**d 越大,每行越"重",同样的 SRAM 只能放下越少的行**,所以块的行数反比于 d。具体 kernel 里 B_r、B_c 还要凑成 16/32/64 的整数倍以对齐 Tensor Core 和 warp,见后文坑点。

**一句话:块大小由 SRAM 容量 M 和头维 d 共同卡死,大致 B ∝ M/(b·d);M 越大、d 越小,能用的块越大,内外层循环次数越少,HBM 往返越少。** 这条 B ∝ M 的关系是下一步 IO 复杂度推导的关键输入。

## 第四步:IO 复杂度推导 Θ(N²d²/M)

现在算 FlashAttention 前向到底碰多少次 HBM,并和标准注意力的 Θ(N²·b) 对比。

### 数 HBM 访存

HBM 访存发生在:加载 Q/K/V 块、写出 O 块。结构是双重循环:外层遍历 T_r 个 Q 块,内层遍历 T_c 个 K/V 块。

```
- Q 和 O:外层每个 Q 块 Qᵢ 载入一次、Oᵢ 写出一次。
  整个 Q、O 各被完整读/写一遍 → 访存 Θ(N·d) 元素。
- K 和 V:对每个 Q 块,内层都要把全部 K、V 块扫一遍。
  K、V 各被完整读 T_r 遍 → 访存 Θ(T_r · N·d) = Θ( (N/B_r) · N·d ) 元素。
```

K/V 的重复读是大头(被读了 T_r 遍)。总 HBM 元素数:

```
HBM元素 ≈ Θ(N·d)  +  Θ( (N / B_r) · N·d )
        ≈ Θ( N²·d / B_r )        # 后项主导
```

代入块大小约束 B_r = Θ(M / (b·d))(把字节换成元素时 b 抵消进 M 的单位里,这里把 M 记成"能放多少个元素"的容量更干净,即 B_r = Θ(M/d)):

```
HBM元素 ≈ Θ( N²·d / (M/d) ) = Θ( N²·d² / M )
```

**结论四:FlashAttention 前向 HBM 访存 = Θ(N²d²/M),其中 M 是 SRAM 容量(以元素计)。** 写成字节再乘 b,量级不变。

### 为什么这就更快

把两者并排:

```
标准注意力 HBM      : Θ(N²)           （还带 >1 的系数:N×N 被搬好几趟）
FlashAttention HBM  : Θ(N² d² / M)
比值(FA / 标准)   : Θ(d² / M)
```

只要 **M ≫ d²**,FlashAttention 的 HBM 访存就显著小于标准注意力。d 常见 64/128,d² = 4096 ~ 16384 个元素;而 SRAM 容量 M 以"几万到几十万个元素"计(几十~上百 KB / 2 字节)。所以 d²/M 是一个**远小于 1 的因子**——典型能把 HBM 访存降低近一个数量级(论文报告的实测 HBM 访存下降和端到端加速,具体倍数见「待核」)。

关键再强调一遍:**FLOPs 没变(仍 ~4N²d),变的只是 HBM 字节数从 Θ(N²) 降到 Θ(N²d²/M)。** 因为算子本来 memory-bound,访存降一个数量级就直接换来接近一个数量级的墙钟时间(wall-clock)收益,把算子从带宽屋顶往算力屋顶推。这就是"不改变计算量、只改变数据流动"能加速的全部道理。

### 一个反直觉的点:M 越大越好,但有上限

公式说 HBM ∝ 1/M,所以 SRAM 越大越省。但 M 是硬件给定的,而且**块太大会降低并行度**(同时能跑的块变少,SM 占用率下降),还可能挤爆寄存器导致 register spilling(寄存器溢出到本地内存,反而慢)。所以实际块大小是"省 HBM"与"够并行 + 不 spill"之间的折中,不是越大越好。这是从渐进复杂度走到真实 kernel 必须补的一课。

## 第五步:反向——用重算换显存

前向我们成功地**没有把 N×N 的 P 写回 HBM**。但反向传播按理需要 P:

```
已知上游梯度 dO,要求 dQ, dK, dV。
反向涉及:dV = Pᵀ dO,  dP = dO Vᵀ,  dS = (softmax 的雅可比作用于 dP),  dQ = dS K,  dK = dSᵀ Q
```

这里处处要用到 P(N×N)和 S(N×N)。如果像标准实现那样在前向把它们存下来,反向直接读——但那等于把我们省下的 N² 显存又吐回去了,长序列照样 OOM。

**FlashAttention 的选择:前向不存 P/S,反向时按需重算(recomputation,也叫 gradient checkpointing 的极端形式)。** 反向同样分块,对每个块用前向存下的少量信息(Q、K、V,以及每行的归一化统计——具体是 logsumexp Lᵢ = mᵢ + log ℓᵢ,一个 N 维向量)重新算出该块的 Sᵢⱼ 和 Pᵢⱼ,在片上算完局部梯度即弃。

要点拆开讲:

1. **前向额外存什么?** 只存 O(N×d,本来就要输出)和每行的 logsumexp 统计 L ∈ ℝ^N(很小)。**不存任何 N×N 张量。** 反向靠 L 和 Q/K/V 就能精确重建每个 Pᵢⱼ 块——因为 Pᵢⱼ 的元素 = exp(Sᵢⱼ - Lᵢ),Lᵢ 已知,Sᵢⱼ = Qᵢ Kⱼᵀ 可重算。
2. **重算贵不贵?** 反向重算 S/P 多花的是 FLOPs(又来一遍 QKᵀ 量级的乘加),但**省掉了把 N×N 矩阵从 HBM 读回的访存**。因为算子 memory-bound,多算的 FLOP 几乎免费(算力本来闲着),省下的访存却是真金白银——**这是一笔在 memory-bound 体制下稳赚的交易**。
3. **数值一致性。** 重算用的是和前向完全相同的公式与基准(同样减 Lᵢ),所以重建的 P 与前向一致,梯度精确,不是近似梯度。
4. **softmax 雅可比别物化。** dS 那一步涉及 softmax 的雅可比矩阵(逐行是 diag(p) - ppᵀ),千万不要真的构造这个 N×N(乃至 N×N×... )的雅可比。利用恒等式:对某一行,dS = P ⊙ (dP - rowsum(P ⊙ dP)),其中 ⊙ 是逐元素乘、rowsum 是行内求和得到一个标量再广播。这样反向也只在 B_r×B_c 块上操作,和前向对称。

**一句话:反向用"重算 N×N 而非存储 N×N"把激活显存从 O(N²) 压到 O(N) ,代价是多一遍矩阵乘的 FLOPs;在 memory-bound 体制下这笔代价几乎不影响墙钟时间,却换来能训更长序列。**

## 第六步:演进——FA-2 与 FA-3 在解决什么

FlashAttention-1 把"该不该 memory-bound"这件事解决了。但把 kernel 写到逼近硬件峰值,还有两层硬骨头:**循环并行的划分**(FA-2)和**新硬件的异步特性**(FA-3)。

### FlashAttention-2:并行划分与减少非矩阵乘运算

FA-1 的内外层循环和 GPU 并行映射并不最优。FA-2 的几个关键改动(按重要性):

1. **沿序列维(行)并行,而不是只沿 batch×head 并行。** FA-1 主要把不同 (batch, head) 分给不同 thread block,当 batch×head 数少(比如长序列、小 batch 推理)时 GPU 占用率不足。FA-2 额外**沿 Q 的行块维度并行**,让序列长时也有足够多的并行 block 喂满 SM。
2. **调整循环顺序,让 Q 块在外层、K/V 在内层,且每个 thread block 独立负责一段 Q 行**,使得**输出 Oᵢ 的写出和重缩放更集中**,减少跨 block 通信和原子操作。
3. **减少非矩阵乘(non-matmul)FLOPs。** Tensor Core 算矩阵乘极快,但 online softmax 里那些 exp、重缩放、逐元素乘是普通 CUDA core 在跑,相对慢。FA-2 重排了重缩放的时机——**把每一步都做的"除以 ℓ"推迟到内层循环结束只做一次**(中间只维护未归一化的 Õ 和 ℓ,最后一次性 O = Õ/ℓ),省掉大量中间的标量除法和重缩放,因为非矩阵乘运算虽然 FLOP 占比小,但吞吐低,容易成为新瓶颈。
4. **更好的 warp 内工作划分**,减少 warp 之间通过 shared memory 的同步。

FA-2 不改变 IO 复杂度的量级(还是 Θ(N²d²/M)),它榨的是**把理论访存优势真正转成接近峰值的 Tensor Core 利用率**——把 GPU 占用率和 non-matmul 开销这两个二阶瓶颈打掉(论文报告相对 FA-1 的加速倍数与可达峰值比例见「待核」)。

### FlashAttention-3:吃透 Hopper(H100)的异步

FA-3 是为 NVIDIA Hopper 架构(H100)量身做的,核心是利用该代新硬件特性:

1. **Warp specialization(warp 专门化)+ 异步流水。** H100 的 TMA(Tensor Memory Accelerator)能让数据从 HBM 到 SRAM 的搬运**异步**进行,Tensor Core(WGMMA 指令)也能异步发射。FA-3 把不同 warp 分工:一部分 warp 专门当"生产者"用 TMA 搬下一块数据,另一部分当"消费者"用 Tensor Core 算当前块,**用软件流水线把访存延迟藏到计算背后**(producer-consumer pipeline)。这样 Tensor Core 不再等数据。
2. **矩阵乘与 softmax 重叠(overlap)。** 把第二个矩阵乘(PV)和下一块的 softmax 指数运算在指令级交错,让慢的 non-matmul(exp)和快的 matmul 同时进行,进一步压住 softmax 暴露出来的时间。
3. **FP8 低精度支持。** 利用 H100 的 FP8 Tensor Core 把吞吐再翻一档,同时用一些技巧(如 block-wise 缩放、把 incoherent 处理 / 误差较大的部分保留高精度)控制 FP8 带来的数值误差。FP8 注意力的精度-速度权衡是 FA-3 的工程重点(具体精度损失与提速见「待核」)。

**一句话总结演进脉络:FA-1 解决"是否 memory-bound"(算法层,IO 复杂度);FA-2 解决"并行划分与 non-matmul 开销"(把理论优势转成实际 Tensor Core 利用率);FA-3 解决"如何吃干榨净新硬件的异步与低精度"(架构特化)。三代都不改变 4N²d 的 FLOPs 本质,都在数据流动和硬件映射上做文章。**

## 前向 kernel:Triton 风格伪代码

下面给一个能体现机制的前向伪代码,Triton 风格(省略了边界判断、mask、缩放等工程细节,聚焦 online softmax + tiling 主干)。读它的时候对照第二步的递推式和第三步的循环结构。

```python
# 前向:O = softmax(Q Kᵀ / √d) V，精确，分块在片上完成
# Q,K,V: [N, d]  在 HBM；O: [N, d] 输出到 HBM
# 块大小 BLOCK_M（Q 行块 B_r）, BLOCK_N（K/V 行块 B_c）受 SRAM 容量约束
# 网格：每个 program 负责一个 Q 行块 i（沿序列维并行，这是 FA-2 风格）

@triton.jit
def flash_attn_fwd(Q, K, V, O, L, sm_scale, N, d,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    i = tl.program_id(0)                      # 当前 Q 行块编号
    # --- 把这一块 Q 载入 SRAM，整段内层循环常驻不动 ---
    q = load_block(Q, row=i*BLOCK_M, shape=(BLOCK_M, d))   # SRAM: BLOCK_M×d

    # --- 初始化 online softmax 的三个统计量（都在 SRAM/寄存器里）---
    m = full((BLOCK_M,), -inf)                # running max，每行一个标量
    l = zeros((BLOCK_M,))                     # running sum（归一化分母）
    acc = zeros((BLOCK_M, d))                 # 未归一化输出累加 Õ

    # --- 内层：遍历所有 K/V 块。完整 N×N 从不出现，只有 BLOCK_M×BLOCK_N 的小块 ---
    for j in range(0, N, BLOCK_N):
        k = load_block(K, row=j, shape=(BLOCK_N, d))       # SRAM: BLOCK_N×d
        v = load_block(V, row=j, shape=(BLOCK_N, d))       # SRAM: BLOCK_N×d

        # 1) 局部打分块 Sᵢⱼ = Qᵢ Kⱼᵀ，Tensor Core 干这步，最大中间张量 BLOCK_M×BLOCK_N
        s = tl.dot(q, k.T) * sm_scale                      # [BLOCK_M, BLOCK_N]

        # 2) 本块的行最大值，并更新全局 running max
        m_blk = tl.max(s, axis=1)                          # 块内每行最大
        m_new = tl.maximum(m, m_blk)                       # m_new = max(m_old, m̃)

        # 3) 关键：重缩放因子，把旧的 l/acc 拉到新基准 m_new
        alpha = tl.exp(m - m_new)                          # exp(m_old - m_new) ≤ 1

        # 4) 本块在新基准下的概率（未归一化）：p = exp(s - m_new)
        p = tl.exp(s - m_new[:, None])                     # [BLOCK_M, BLOCK_N]

        # 5) online 更新：先重缩放旧值，再并入本块
        l = alpha * l + tl.sum(p, axis=1)                  # ℓ_new（见第二步递推）
        acc = alpha[:, None] * acc + tl.dot(p, v)          # Õ_new = α·Õ_old + p·V
        m = m_new

    # --- 内层结束，最后一次性归一化（FA-2：除法只做一次，省 non-matmul）---
    o = acc / l[:, None]                                   # O = Õ / ℓ
    store_block(O, row=i*BLOCK_M, value=o)
    # 存每行 logsumexp 供反向重算用：L = m + log(l)，只有 O(N)，不存任何 N×N
    store_block(L, row=i*BLOCK_M, value=m + tl.log(l))
```

把这段和前面对齐着看,三个机制都在:**(a) tiling**——外层 program 切 Q、内层 for 切 K/V,SRAM 里最大就是 `s` 的 BLOCK_M×BLOCK_N;**(b) online softmax**——`m_new`、`alpha = exp(m - m_new)`、`l = alpha*l + ...`、`acc = alpha*acc + ...` 就是第二步递推式的逐行实现;**(c) 反向准备**——只存 `L = m + log(l)`(每行一个标量,logsumexp),前向绝不写 N×N。

一个纯 PyTorch 的等价参考实现(慢,但用来对拍正确性,验证"分块结果 == 一次性 softmax"):

```python
def flash_attn_reference(Q, K, V, sm_scale, BLOCK_N):
    N, d = Q.shape
    O = torch.zeros_like(Q)
    for i in range(N):                      # 每一行(为简洁逐行,真实 kernel 是块)
        m = torch.tensor(float('-inf'))
        l = torch.tensor(0.0)
        acc = torch.zeros(d)
        for j0 in range(0, N, BLOCK_N):     # 分块扫 K/V
            kblk = K[j0:j0+BLOCK_N]; vblk = V[j0:j0+BLOCK_N]
            s = (Q[i] @ kblk.T) * sm_scale  # [BLOCK_N]
            m_blk = s.max()
            m_new = torch.maximum(m, m_blk)
            alpha = torch.exp(m - m_new)    # 重缩放旧累加
            p = torch.exp(s - m_new)        # [BLOCK_N]
            l = alpha * l + p.sum()
            acc = alpha * acc + p @ vblk    # Õ_new
            m = m_new
        O[i] = acc / l                      # 归一化
    return O
# 断言:torch.allclose(flash_attn_reference(...), torch.softmax(Q@K.T*scale,-1)@V) 应为 True
```

**这段参考实现的价值在于:它能用 `allclose` 当场验证第二步证明的"逐位等价"结论——这是把数学推导落到代码的最短闭环。**

## 设计权衡与常见坑

- **FlashAttention 是精确的,不是近似。** 别和线性 attention、稀疏 attention 混为一谈——后两者改变了计算复杂度(往往降到次二次)但牺牲精度;FlashAttention 仍是 O(N²) FLOP 的精确 softmax 注意力,只优化访存。**面试和设计评审里把这点说错会很尴尬。**
- **省的是访存和显存,不是 FLOPs。** "FlashAttention 让 attention 变成线性复杂度"是常见误解。FLOPs 还是 Θ(N²d);线性的是**显存**(从 O(N²) 降到 O(N))和被砍掉的**HBM 访存系数**。
- **块大小不是越大越好。** 渐进式 Θ(N²d²/M) 鼓励大 M,但真实 kernel 受寄存器数量、shared memory 容量、SM 占用率三重夹击。块太大→并行 block 太少 / 寄存器溢出,反而变慢。块大小要凑 16/32/64 的整数倍以喂满 Tensor Core 和对齐 warp。
- **重算不是浪费。** 新手看到反向"又算一遍 QKᵀ"会觉得亏。在 memory-bound 体制下,多花的 FLOP 几乎免费,省下的 N² 访存才是收益。这是"用计算换访存"的典范,和直觉相反。
- **数值:必须减 running max。** 跳过 `m` 直接 exp 会在 FP16 下迅速溢出(FP16 最大约 65504,exp 参数超过约 11 就炸)。重缩放因子 exp(m_old - m_new) 永远 ≤ 1 也是为了不溢出——这是算法能在低精度下稳定的前提。
- **因果 mask 的负载不均。** 带 causal mask 时,下三角才有效,上三角块可整块跳过。但若简单按块切,不同 Q 块的有效 K 块数不同(后面的行要扫更多列),导致**负载不均衡**;成熟实现会做块级的 mask 跳过 + 负载均衡,否则尾部块拖慢整体。
- **head dim 太大吃不下。** d 很大(如 256)时,B ∝ M/d 让块变得很小,内外层循环次数暴增,优势缩水;某些实现对大 d 有专门 kernel 或不支持,选模型结构时要留意。
- **别自己手撸生产 kernel。** 理解机制要从零推一遍;但生产环境直接用 `flash-attn` 库 / PyTorch SDPA 后端,它们处理了无数边界、精度、架构分支。自撸 kernel 漏掉一个边界判断就是静默的数值错误。

## 动手练习

1.(推导题,核心)**亲手推导 online softmax 三量递推。** 不看本章,从"安全 softmax 要减全行最大值"出发,定义 m、ℓ、Õ,推出来新块时三者各自的更新式,并用归纳法证明扫完整行后 O 与一次性 softmax 逐位相等。*提示:难点全在 ℓ 和 Õ 换基准时那个重缩放因子 exp(m_old − m_new),先确认它对单独一项 exp(xⱼ − m_old)→exp(xⱼ − m_new) 成立,再整段乘。*

2.(估算题)**算 N=8192、d=128、FP16(b=2)下,标准注意力 vs FlashAttention 的前向 HBM 访存,给出比值数量级。** *提示:标准 ≈ 4N²·b 字节(N×N 往返主导);FlashAttention ≈ Θ(N²d²/M),取一个合理的 SRAM 容量 M(以字节,自己设并说明,如几十~上百 KB),换成元素数时除以 b。比较两者,解释为什么比值与 N 无关、只和 d²/M 有关。再算两者的 FLOPs(都应是 ~4N²d),确认 FLOPs 相等——这能直接戳破"FA 是线性复杂度"的误解。*

3.(编码题)**实现本章的 `flash_attn_reference` 并对拍。** 用 PyTorch 写出分块版,和 `torch.softmax(Q@K.T*scale, -1) @ V` 用 `torch.allclose(rtol=1e-3, atol=1e-3)` 对比,在随机 Q/K/V(含故意放大的极端值,如某些行乘 50)下验证不溢出且数值一致。*提示:故意制造大值是为了暴露"忘记减 max"的 bug——正确实现应纹丝不动,错误实现会出 nan。*

4.(分析题)**讨论块大小 B 的选择。** 写出 SRAM 占用随 B_r、B_c、d 的表达式,说明给定 M 时如何取 B;再论证为什么"B 取到 SRAM 上限"未必最快(从 SM 占用率、寄存器溢出、并行 block 数三个角度)。*提示:IO 复杂度 Θ(N²d²/M) 只看渐进访存,真实墙钟时间还受并行度和寄存器压力影响,这是渐进分析和工程的鸿沟。*

## 源码 / 论文导读

- **FlashAttention(v1)论文**:Dao et al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*(NeurIPS 2022)。重点读 **Section 3.1(算法,Algorithm 1 前向伪代码)** 和 **Section 3.1 的 IO 复杂度定理(给出 Θ(N²d²/M) 与标准 Θ(Nd + N²) 的对比)及其证明**——和本章第四步逐行对得上;附录有反向重算的完整推导。
- **online softmax 原始出处**:Milakov & Gimelshein, *"Online normalizer calculation for softmax"*(2018)。读它的 Algorithm 3,就是本章第二步的递推骨架。
- **FlashAttention-2 论文**:Dao, *"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"*(2023)。重点看**并行策略(沿序列维并行)**与 **减少 non-matmul FLOP / 推迟归一化**那两节。
- **FlashAttention-3 论文/技术报告**:Shah et al.(2024,Hopper 特化)。重点看 **warp specialization + producer-consumer 异步流水**和 **FP8** 两部分,对照 H100 的 TMA / WGMMA 指令理解。
- **开源代码**:
  - `Dao-AILab/flash-attention` 仓库:`csrc/` 下的 CUDA kernel 是生产实现(C++/CUDA,读起来重,先读 Python 接口与文档);
  - **Triton 官方 tutorial 里的 fused attention 示例**(`python/tutorials/06-fused-attention.py` 一类文件,路径以仓库为准「待核」):这是最适合学习的版本,本章前向伪代码即按它的结构组织,强烈建议把它跑起来 + 单步读懂;
  - PyTorch 的 `scaled_dot_product_attention`(SDPA)及其 flash 后端:生产里实际调用入口,理解它如何在 flash / mem-efficient / math 三个后端间分派。
- 建议路径:先读 Milakov(理解 online softmax)→ 读 FA-1 Section 3 + Triton tutorial(把前向跑通)→ 推一遍反向 → 再看 FA-2/3 的工程演进。

## 小结与承上启下

这一章我们做了一件具体的事:**用访存第一性原理,把"注意力为什么慢"从一句口号变成一组能验算的数。** 你现在应当能独立完成这条推理链:

```
标准注意力 FLOP ≈ 4N²d，HBM ≈ Θ(N²·b)，算术强度 I ≈ d/b（与 N 无关）
  → 低于 I_crit → memory-bound → 瓶颈在搬 N×N 矩阵
  → online softmax（m/ℓ/Õ 三量递推 + exp(m_old−m_new) 重缩放，可证逐位等价）
  → tiling（Q/K/V 切块进 SRAM，N×N 块算完即弃，块大小 B ∝ M/d）
  → HBM 降到 Θ(N²d²/M)，FLOPs 不变 → 直接换来近一个量级的墙钟收益
  → 反向用重算（不存 N×N，只存 O(N) 的 logsumexp）换 O(N) 显存
  → FA-2 修并行划分 + 砍 non-matmul；FA-3 吃 Hopper 异步 + FP8
```

**最该带走的一句话:FlashAttention 没有发明更快的算法,它发明了更省的数据流动——这是 memory-bound 时代 MLSys 的通用方法论。** "FLOPs 不是瓶颈,访存才是"这个视角,会在本课后续反复出现:KV cache、量化、算子融合、通信优化,本质都是同一件事的不同侧面。下一步,我们会把这套 IO 感知的思路从单算子推广到**推理时的 KV cache 与显存管理**——那里序列在增长、batch 在变化,访存预算的算法会变得更动态、更有意思。

---

**本章「待核」清单(数值/API,凭记忆易错,落地前请核对):**
- A100 / H100 的 SRAM(shared memory)单 SM 容量与片上总量的具体 KB / MB 数;
- 各架构 HBM 带宽、峰值 FP16/BF16 算力,以及由此算出的临界算术强度 I_crit 精确值;
- FlashAttention-1/2/3 论文报告的 HBM 访存下降倍数与端到端加速倍数(随序列长度、d、硬件而变);
- FA-2 相对 FA-1 的加速比、可达 Tensor Core 峰值百分比;
- FA-3 在 FP8 下的提速倍数与精度损失数据;
- Triton 官方 fused attention tutorial 的确切文件路径/文件名;
- FlashAttention 论文里块大小取值的精确表达式(B_c、B_r 的确切系数与上界形式)。


---

## 练习参考答案

> 本章「动手练习」的参考答案(AI 生成,推导/代码已尽量自验,具体数值见「待核」标注)。

### 练习 1:亲手推导 online softmax 三量递推并证明等价

**目标。** 在分块、一次只能看到一行的一段的约束下,维护三个统计量 (m, ℓ, Õ),使得扫完整行后得到的 O 与"先看全行再 softmax 再乘 V"逐位相等。

**第一步:为什么必须减最大值(数值动机)。**
softmax 第 i 项是 exp(xᵢ)/Σⱼexp(xⱼ)。直接算 exp(xⱼ) 在 xⱼ 稍大时溢出(FP32 下 exp(89) 已 overflow,FP16 下指数参数超过约 11 就炸)。利用恒等式分子分母同乘 exp(−m)(m 为该行最大值):

```
exp(xᵢ) / Σⱼ exp(xⱼ) = exp(xᵢ − m) / Σⱼ exp(xⱼ − m)
```

减完后所有指数参数 ≤ 0,exp 落在 (0,1],绝不溢出。这是恒等变形,不改结果。难点在于:现在连 m 都要看全行,而分块时看不到全行——这正是要 online 维护 m 的原因。

**第二步:定义三个 running 统计量(看过前 t 个块时,针对某一固定行)。**

```
mₜ = max{前 t 块的所有 xⱼ}                          # running max（标量）
ℓₜ = Σ_{j∈前t块} exp(xⱼ − mₜ)                        # running sum / 归一化分母（标量）
Õₜ = Σ_{j∈前t块} exp(xⱼ − mₜ) · v_j                  # 未归一化输出累加（d 维向量）
```

注意 ℓₜ 和 Õₜ 的"基准"都是 mₜ:它们里每一项的指数都是减当前的 running max。基准会随新块到来而变,这是后面要重缩放的根源。

**第三步:重缩放因子的推导(整个递推的关键)。**
来一个新块,记块内最大值 m̃ = max(块内 xⱼ),块内基准下的和与加权和:

```
ℓ̃ = Σ_{j∈新块} exp(xⱼ − m̃)
Õ̃ = Σ_{j∈新块} exp(xⱼ − m̃) · v_j
```

新的全局最大值 m_new = max(m_old, m̃)。问题:ℓ_old 以 m_old 为基准,ℓ̃ 以 m̃ 为基准,**基准不同不能直接相加**。先在单独一项上确认换基准成立——这是提示要求先验证的:

```
exp(xⱼ − m_new) = exp(xⱼ − m_old) · exp(m_old − m_new)        ……（★）
```

验证:右边 = exp((xⱼ − m_old) + (m_old − m_new)) = exp(xⱼ − m_new),成立。
关键观察:因子 exp(m_old − m_new) 与下标 j 无关——它对旧基准下的每一项都是同一个常数。所以整段旧和只需乘这一个因子。于是把 (★) 对前 (t−1) 块所有项求和:

```
Σ_{前t-1块} exp(xⱼ − m_new) = exp(m_old − m_new) · Σ_{前t-1块} exp(xⱼ − m_old)
                            = exp(m_old − m_new) · ℓ_old
```

新块的 ℓ̃ 同理从 m̃ 换到 m_new,乘 exp(m̃ − m_new)。两段现在同基准,可以相加。

**第四步:三量递推式(结论)。**

```
m_new = max(m_old, m̃)
ℓ_new = exp(m_old − m_new) · ℓ_old  +  exp(m̃ − m_new) · ℓ̃
Õ_new = exp(m_old − m_new) · Õ_old  +  exp(m̃ − m_new) · Õ̃
```

(Õ 的两项就是把 ℓ 的对应项每一加数末尾乘上 v_j,因为换基准因子与 j 无关,可提到求和外。)
因为 m_old ≤ m_new 且 m̃ ≤ m_new,两个 exp 因子都 ≤ 1,数值安全(永不溢出)。
扫完所有 T 块后做一次除法:

```
O = Õ_final / ℓ_final
```

三个量里 m、ℓ 是标量、Õ 是 d 维向量,占用与 N 无关——这是能 tiling 的前提。

**第五步:归纳法证明逐位等价。**
**命题。** 设 Mₜ = max{前 t 块的 xⱼ},Lₜ = Σ_{j∈前t块} exp(xⱼ − Mₜ),Sₜ = Σ_{j∈前t块} exp(xⱼ − Mₜ)·v_j。则递推满足 mₜ = Mₜ、ℓₜ = Lₜ、Õₜ = Sₜ 对所有 t 成立。

**归纳基 (t=1)。** 第一块初始化 m₁ = m̃₁(块内即全局最大)= M₁,ℓ₁ = ℓ̃₁ = L₁,Õ₁ = Õ̃₁ = S₁。成立。

**归纳步。** 设 t−1 时 mₜ₋₁ = Mₜ₋₁、ℓₜ₋₁ = Lₜ₋₁、Õₜ₋₁ = Sₜ₋₁。来第 t 块:

```
mₜ = max(mₜ₋₁, m̃) = max(Mₜ₋₁, max(块t)) = Mₜ                       ✓（max 结合性）

ℓₜ = exp(mₜ₋₁ − mₜ)·ℓₜ₋₁ + exp(m̃ − mₜ)·ℓ̃
   = exp(Mₜ₋₁ − Mₜ)·Σ_{前t-1块} exp(xⱼ − Mₜ₋₁) + exp(m̃ − Mₜ)·Σ_{块t} exp(xⱼ − m̃)
   = Σ_{前t-1块} exp(xⱼ − Mₜ) + Σ_{块t} exp(xⱼ − Mₜ)     # 用 (★)：exp(a−b)·exp(x−a)=exp(x−b)
   = Σ_{前t块} exp(xⱼ − Mₜ) = Lₜ                                      ✓

Õₜ：把上式每一加数末尾乘 v_j（换基准因子与 j 无关，可提到 Σ 外），同理得 Õₜ = Sₜ   ✓
```

归纳完成。扫到 t = T:ℓ_T = L_T = Σ_全行 exp(xⱼ − M),Õ_T = S_T = Σ_全行 exp(xⱼ − M)·v_j(M 为全行最大),于是

```
O = Õ_T / ℓ_T = [ Σⱼ exp(xⱼ − M)·v_j ] / [ Σⱼ exp(xⱼ − M) ]
              = Σⱼ softmax(x)_j · v_j
```

**结论:online 三量递推扫完整行的输出 O,与一次性"全行 softmax 再乘 V"逐位相等(仅差浮点舍入)。因此 FlashAttention 是精确注意力,不是近似——这与线性 / 稀疏 attention 有本质区别。**

---

### 练习 2:N=8192、d=128、FP16 下标准 vs FlashAttention 的 HBM 访存与 FLOPs

**参数。** N = 8192,d = 128,b = 2 字节(FP16)。

**第一步:标准注意力 HBM(N×N 往返主导)。**

```
标准 HBM ≈ 4·N²·b = 4 × 8192² × 2 字节
        = 4 × 67,108,864 × 2 = 536,870,912 字节 ≈ 512 MB
```

(系数 4 来自把 N×N 的 S/P 在 HBM 里来回搬约 4 趟:写 S、读 S、写 P、读 P;不同框架略有差异,量级稳定。Q/K/V/O 的 Θ(Nd) 项相对 N² 可忽略:Nd·b = 8192×128×2 ≈ 2 MB,远小于 512 MB。)

**第二步:FlashAttention HBM = Θ(N²d²/M),自设 M 并说明。**
设单 SM 可用 SRAM 容量 M = 100 KB = 102400 字节(几十~上百 KB 量级,合理;A100/H100 单 SM shared memory 在此量级,精确值待核)。换成"能放多少个元素":

```
M_elem = M / b = 102400 / 2 = 51,200 个元素
FA HBM(元素) ≈ N²·d² / M_elem = 8192² × 128² / 51200
            = 67,108,864 × 16,384 / 51,200 ≈ 2.147×10¹⁰ 个元素
FA HBM(字节) = ×b ≈ 4.295×10⁷ 字节 ≈ 41 MB
```

换不同 M 复核(同一组数):

```
M = 48 KB  → FA ≈ 85 MB，ratio 标准/FA ≈ 6×
M = 100 KB → FA ≈ 41 MB，ratio ≈ 12.5×
M = 192 KB → FA ≈ 21 MB，ratio ≈ 24×
```

**第三步:比值与 N 无关、只看 d²/M。**
取比值(写成元素):

```
FA / 标准 = (N²d²/M_elem) / (N²·系数) = d² / (系数·M_elem) ∝ d²/M
```

**N² 在分子分母同时出现,直接约掉**,所以比值与序列长度 N 完全无关——这是核心:无论序列多长,FA 相对标准省的倍数恒定,由 d²/M 决定。代入 d²=16384、M_elem=51200:d²/M ≈ 0.32,即 FA 访存约为标准的 1/3 量级(再叠加标准那 ~4 倍的 N×N 往返系数,综合约一个数量级,与"省近一个数量级"的论文结论一致)。

**第四步:两者 FLOPs(戳破"FA 是线性复杂度"误解)。**

```
标准 FLOPs ≈ 4N²d = 4 × 8192² × 128 ≈ 3.436×10¹⁰ ≈ 34.4 GFLOP
FA    FLOPs ≈ 4N²d = 同上 = 34.4 GFLOP
```

**结论:标准与 FlashAttention 的 FLOPs 完全相等(均 ~4N²d ≈ 34.4 GFLOP,仍是 Θ(N²d) 的二次复杂度);FlashAttention 不优化 FLOPs,只把 HBM 访存从 ~512 MB 降到几十 MB(本例约 6~24×,取决于 M),且这个倍数只依赖 d²/M、与 N 无关。所谓"FA 把注意力变成线性复杂度"是误解——线性的是显存(O(N²)→O(N)),不是计算量。** 因为算子本来 memory-bound,访存降一个量级直接换来接近一个量级的墙钟收益。

---

### 练习 3:实现 flash_attn_reference 并对拍(含极端值不溢出)

下面给一个 CPU 即可跑的版本。两份实现:numpy 版(无任何额外依赖,已在本机验证)和 PyTorch 版(章节原型,CPU 可跑,`pip install torch` 即可)。两者数学一致。

**numpy 版(已验证:max diff ≈ 2.2e-15,无 NaN):**

```python
import numpy as np

def flash_attn_reference_np(Q, K, V, sm_scale, BLOCK_N):
    """分块 online-softmax 注意力，逐行扫描，精确等价于全行 softmax。"""
    N, d = Q.shape
    O = np.zeros_like(Q)
    for i in range(N):                       # 逐行（真实 kernel 是块，机制相同）
        m = -np.inf                          # running max
        l = 0.0                              # running sum（归一化分母）
        acc = np.zeros(d)                    # 未归一化输出累加 Õ
        for j0 in range(0, N, BLOCK_N):      # 分块扫 K/V
            kb = K[j0:j0+BLOCK_N]; vb = V[j0:j0+BLOCK_N]
            s = (Q[i] @ kb.T) * sm_scale     # 局部打分 [BLOCK_N]
            m_blk = s.max()                  # 块内最大 m̃
            m_new = max(m, m_blk)            # m_new = max(m_old, m̃)
            alpha = np.exp(m - m_new)        # 重缩放因子 exp(m_old-m_new) ≤ 1
            p = np.exp(s - m_new)            # 新基准下未归一化概率，永不溢出
            l = alpha * l + p.sum()          # ℓ_new
            acc = alpha * acc + p @ vb       # Õ_new = α·Õ_old + p·V
            m = m_new
        O[i] = acc / l                       # 最后一次性归一化
    return O

def softmax_gold(Q, K, V, scale):
    S = Q @ K.T * scale
    S = S - S.max(axis=1, keepdims=True)     # gold 同样减 max（数值安全）
    P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
    return P @ V

# ---- 对拍：含故意放大的极端值 ----
np.random.seed(0)
N, d = 64, 16
Q = np.random.randn(N, d); K = np.random.randn(N, d); V = np.random.randn(N, d)
Q[3] *= 50; K[7] *= 50                        # 制造大 logits，暴露"忘减 max"的 bug
scale = 1.0 / np.sqrt(d)

ref  = flash_attn_reference_np(Q, K, V, scale, BLOCK_N=16)
gold = softmax_gold(Q, K, V, scale)
assert np.allclose(ref, gold, rtol=1e-3, atol=1e-3)   # 通过
assert not np.isnan(ref).any()                        # 无 NaN
print("max abs diff:", np.abs(ref - gold).max())      # ≈ 2.2e-15
```

实测输出:`allclose: True`,`any nan: False`,`max diff ≈ 2.2e-15`。

**PyTorch 版(章节原型,CPU 可跑;依赖:`pip install torch`):**

```python
import torch

def flash_attn_reference(Q, K, V, sm_scale, BLOCK_N):
    N, d = Q.shape
    O = torch.zeros_like(Q)
    for i in range(N):
        m = torch.tensor(float('-inf'))
        l = torch.tensor(0.0)
        acc = torch.zeros(d)
        for j0 in range(0, N, BLOCK_N):
            kblk = K[j0:j0+BLOCK_N]; vblk = V[j0:j0+BLOCK_N]
            s = (Q[i] @ kblk.T) * sm_scale          # [BLOCK_N]
            m_blk = s.max()
            m_new = torch.maximum(m, m_blk)
            alpha = torch.exp(m - m_new)            # 重缩放旧累加
            p = torch.exp(s - m_new)                # 减 max，不溢出
            l = alpha * l + p.sum()
            acc = alpha * acc + p @ vblk            # Õ_new
            m = m_new
        O[i] = acc / l
    return O

torch.manual_seed(0)
N, d = 64, 16
Q = torch.randn(N, d); K = torch.randn(N, d); V = torch.randn(N, d)
Q[3] *= 50; K[7] *= 50
scale = 1.0 / d ** 0.5
ref  = flash_attn_reference(Q, K, V, scale, BLOCK_N=16)
gold = torch.softmax(Q @ K.T * scale, dim=-1) @ V
assert torch.allclose(ref, gold, rtol=1e-3, atol=1e-3)
assert not torch.isnan(ref).any()
```

**对照实验:为什么必须减 max。** 把 `p = exp(s - m_new)` 改成 `p = exp(s)`(去掉减 max)、并在 FP16 下跑同一组带 ×50 大值的输入,`exp(50/√16)=exp(12.5)≈2.7×10⁵` 已超 FP16 上限 65504,累加溢出为 inf,最终 `acc/l` 出现 **NaN**(本机用 float16 复现确认 `nan present: True`)。

**结论:正确实现(每块减 running max、用 exp(m_old−m_new) ≤ 1 重缩放)在极端大 logits 下数值纹丝不动,与全行 softmax 逐位一致(max diff ~1e-15);去掉减 max 的实现在低精度下直接溢出成 NaN。这就是 online softmax 必须维护并减去 running max 的工程原因。**

---

### 练习 4:块大小 B 的选择——SRAM 占用表达式与"取满未必最快"

**第一步:SRAM 占用表达式。**
内层一次迭代,片上需同时容纳:Qᵢ(B_r×d)、Kⱼ(B_c×d)、Vⱼ(B_c×d)、打分块 Sᵢⱼ(B_r×B_c)、输出累加 Oᵢ/Õᵢ(B_r×d),外加 mᵢ、ℓᵢ(各 B_r 个标量,可忽略)。以元素计:

```
SRAM占用(元素) ≈ B_r·d + B_c·d + B_c·d + B_r·B_c + B_r·d
              = (2B_r + 2B_c)·d + B_r·B_c
              = O( (B_r + B_c)·d + B_r·B_c )
```

乘字节宽 b 后约束为:`[(2B_r+2B_c)·d + B_r·B_c] · b ≤ M`(M 为可用 SRAM 字节)。

**第二步:给定 M 如何取 B。**
两类项:与 d 挂钩的"载入项" (B_r+B_c)·d,和打分块项 B_r·B_c。FlashAttention 的策略是让块大小随 d 反比缩放:令

```
B_c ≈ Θ( M / (b·d) )，B_r 取 min( 同量级, d ) 之类上界
```

直觉:**d 越大每行越"重",同样的 SRAM 只能放下越少的行,所以块的行数反比于 d**(B ∝ M/(b·d))。M 越大、d 越小,块越大,内外层循环次数越少,K/V 被重复读的遍数 T_r = ⌈N/B_r⌉ 越小,HBM 往返越少——这正是 IO 复杂度 Θ(N²d²/M) 里 1/M 的来源。此外 B_r、B_c 实际还要凑成 16/32/64 的整数倍,以对齐 Tensor Core 的 MMA 形状和 warp。

**第三步:为什么"B 取到 SRAM 上限"未必最快(三个角度)。**
Θ(N²d²/M) 只是渐进访存,鼓励大 M/大块;但真实墙钟时间还受下面三重夹击:

1. **SM 占用率(occupancy)下降。** GPU 靠在一个 SM 上同时驻留多个 thread block / warp,用计算掩盖访存延迟。块越大,单块吃的 shared memory 和寄存器越多,**单 SM 能同时驻留的块数越少**,占用率下降,延迟无法被掩盖,内存延迟重新暴露——访存量虽降,但延迟没藏住,反而慢。
2. **寄存器溢出(register spilling)。** 块越大,累加器 acc(B_r×d)、中间 p 等占的寄存器越多。一旦超过每线程寄存器上限,编译器把寄存器溢出到 local memory(物理上在 HBM),**本想省的 HBM 访存又以 spill 形式吐回来**,且是慢路径,得不偿失。
3. **并行 block 数不足。** 总并行块数 ∝ (序列块数 × batch × head)。块越大,Q 行块数 T_r 越小,**可供调度的 block 越少**;当 batch×head 本就小(长序列、小 batch 推理)时,块再放大会让全 GPU 的 SM 喂不满,算力闲置。

**结论:块大小是"省 HBM(要大块)"与"够并行 + 不 spill + 高占用(要小块)"之间的折中,存在一个最优中段而非越大越好。渐进式 Θ(N²d²/M) 只回答"访存随 M 怎么变",回答不了寄存器压力、占用率、并行度这些一阶决定墙钟时间的工程因素——这正是渐进分析与真实 kernel 调优之间的鸿沟。实践中块大小由 SRAM 容量、寄存器上限、SM 占用率三者共同卡死,并凑成 16/32/64 的整数倍,通常靠 autotune 在候选集上实测选优。**
