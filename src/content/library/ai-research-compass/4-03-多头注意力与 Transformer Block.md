---
title: "多头注意力与 Transformer Block:把组件拼成一层"
slug: "4-03"
collection: "ai-research-compass"
group: "大模型算法专家课程"
order: 4003
summary: "这一章把你从\"会写一个 scaled dot-product 注意力函数\"带到\"能从零搭出一整层 Transformer,并讲清楚每个零件为什么在那里\"。上一章(02)我们把单头注意力 `Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V` 从动机推到精确公式,也算清了它 O(n²d) 的复杂度。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T05:57:03.000Z"
updatedAt: "2026-06-19T05:57:03.000Z"
---
> 这一章把你从"会写一个 scaled dot-product 注意力函数"带到"能从零搭出一整层 Transformer,并讲清楚每个零件为什么在那里"。上一章(02)我们把单头注意力 `Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V` 从动机推到精确公式,也算清了它 O(n²d) 的复杂度。但单头注意力本身**不是一层网络**——它甚至不含一个可学习的非线性。这一章做三件事:(1) 把单头扩展成**多头(multi-head)**,讲清楚"切成 h 份低维子空间"为什么几乎不加参数却显著加表达力;(2) 把注意力子层和**前馈子层(FFN)** 用**残差 + LayerNorm** 包成一个标准 block,讲清 FFN 不是可有可无的配菜而是承载模型大部分参数与"记忆"的主体;(3) 把 **Pre-LN 与 Post-LN** 的梯度量级差异推到能解释"为什么原版 Transformer 要 warmup、为什么现代模型几乎都用 Pre-LN"。读完你能默写出一个 Pre-LN block 的前向、能徒手算出 MHA 与 FFN 各自的参数量、能用一句梯度分析说清两种 LN 摆放的本质区别。这是后面位置编码(04)、归一化与优化(05)、scaling law(06)全部建立其上的"一层"的精确定义。

## 一、动机:单头注意力差在哪,缺了什么

先把上一章的结论钉死,作为这一章的起点。给定查询矩阵 Q ∈ ℝ^{n×dₖ}、键 K ∈ ℝ^{n×dₖ}、值 V ∈ ℝ^{n×dᵥ}(n 是序列长度),单头注意力是:

```
Attn(Q,K,V) = softmax(QKᵀ / √dₖ) · V        ∈ ℝ^{n×dᵥ}
```

这是 Transformer 的心脏没错,但**单独拿它当一层网络,有三个结构性缺陷**,而这一章的每个组件都是来补这三个缺陷的:

**缺陷一:注意力对 value 是线性的,没有逐位置的非线性变换。** 注意一件容易被忽略的事:固定住注意力权重矩阵 `A = softmax(QKᵀ/√dₖ)`(它确实依赖输入,是数据相关的),那么输出 `A·V` 对 V 而言是**纯线性**的——它只是把各个位置的 value **加权平均**。换句话说,注意力做的是"信息的混合与路由(routing)",回答"每个位置该从哪些位置取信息"。但混合完之后,对每个位置自己的特征做一次非线性加工(比如"如果这个位置同时具备特征 X 和 Y,就激活特征 Z")这件事,注意力一点都没做。一个只堆注意力、不加 FFN 的网络,表达力会严重受限——这是 FFN 子层存在的根本理由。

**缺陷二:单头只能学一种"匹配模式"。** 单头注意力里,Q 和 K 的内积定义了**唯一一种**相似度度量。但语言里的依赖关系是多样的:有的头需要捕捉**语法**(动词找它的主语)、有的需要**指代**(代词 "it" 找它指向的名词)、有的需要**位置**(关注前一个 token)、有的需要**主题相关性**。这些是不同的子空间、不同的匹配函数。一个单头被迫把所有这些关系塞进一个 softmax 里,会互相打架——增大权重去捕捉语法,可能就破坏了指代。**多头的核心动机:让不同的头在不同的低维子空间里各学各的关系,最后汇总。**

**缺陷三:没有深度。** 一次注意力 + 一次 FFN 只是"一层"。真正的能力来自把这样的层**堆几十上百层**。而一旦要堆深,"梯度能不能干净地传到底层"就成了生死问题——这就是残差连接和 LayerNorm 摆放(Pre/Post)要解决的。

把这三点连起来,一个 **Transformer block** 的设计逻辑就清楚了:

```
block(x) = x
           ├─ 子层1:多头自注意力   → 解决缺陷二(多种关系)+ 缺陷一的"混合"部分
           ├─ 子层2:前馈 FFN       → 解决缺陷一的"逐位置非线性"部分
           └─ 残差 + LayerNorm 包裹每个子层 → 解决缺陷三(可堆深)
```

下面逐个推。

## 二、多头注意力:把 d 维切成 h 份子空间

### 2.1 机制:投影 → 切头 → 各自注意 → 拼接 → 输出投影

设模型隐藏维度为 d(原论文 d=512,GPT-2 small d=768)。多头注意力(Multi-Head Attention, MHA)的完整定义如下。先定义每个头的工作维度:

```
dₖ = dᵥ = d / h        (h 是头数,要求 h 能整除 d)
```

对输入序列 X ∈ ℝ^{n×d}(n 个 token,每个 d 维),做四组线性投影,得到 h 组各自的 Q/K/V:

```
对第 i 个头 (i = 1..h):
    Qᵢ = X · W_Qⁱ        W_Qⁱ ∈ ℝ^{d×dₖ}
    Kᵢ = X · W_Kⁱ        W_Kⁱ ∈ ℝ^{d×dₖ}
    Vᵢ = X · W_Vⁱ        W_Vⁱ ∈ ℝ^{d×dᵥ}

    headᵢ = Attn(Qᵢ, Kᵢ, Vᵢ) = softmax(Qᵢ Kᵢᵀ / √dₖ) · Vᵢ   ∈ ℝ^{n×dᵥ}
```

每个头独立地在自己的 dₖ 维子空间里做一遍上一章那个完整的注意力。然后**把 h 个头的输出在特征维上拼接(concatenate)**,再过一个输出投影 W_O 把维度变回 d:

```
MultiHead(X) = Concat(head₁, …, head_h) · W_O
               其中 Concat(...) ∈ ℝ^{n × (h·dᵥ)} = ℝ^{n×d}
               W_O ∈ ℝ^{(h·dᵥ) × d} = ℝ^{d×d}
```

因为 h·dᵥ = h·(d/h) = d,拼接后正好是 d 维,W_O 是一个 d×d 的方阵投影。

**关键:为什么这么切,参数量几乎不变?** 直觉上"切成 h 个头"听起来像把模型放大了 h 倍。其实不是。把所有头的投影矩阵在列方向拼起来:`W_Q = [W_Q¹ | W_Q² | … | W_Q^h] ∈ ℝ^{d×(h·dₖ)} = ℝ^{d×d}`。所以**整个 MHA 的 Q 投影合起来就是一个 d×d 矩阵**,和"单头但 dₖ=d"用的投影矩阵一样大。我们只是把这个 d×d 投影的输出**在语义上分组**成 h 块,每块单独做注意力——而不是让整个 d 维一起做一个大注意力。下一节把参数量算到精确数字。

### 2.2 实际实现:不是真的 h 个矩阵,而是一个大矩阵 reshape

工程上没人真的开 h 个独立的 `W_Qⁱ`。标准做法是:用**一个** d×d 的 `W_Q` 一次性把 X 投到 d 维,然后把这 d 维 **reshape 成 (h, dₖ)** 来切头。这两种写法数学上等价(因为分块矩阵乘法 = 各块分别乘),但合成一个大矩阵能让 GPU 一次大 matmul 吃满算力,远快于 h 次小 matmul。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads          # dₖ = dᵥ = d/h
        # 把 Q/K/V 三个 d×d 投影合并成一个 d×3d,一次 matmul 出 Q,K,V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)  # W_O
        self.causal = causal
        self.dropout = dropout

    def forward(self, x):
        # x: (B, n, d)   B=batch, n=序列长, d=d_model
        B, n, d = x.shape
        # 一次投影出 Q,K,V,各 (B, n, d)
        qkv = self.qkv_proj(x)                              # (B, n, 3d)
        q, k, v = qkv.chunk(3, dim=-1)                      # 各 (B, n, d)
        # 切头:(B, n, d) -> (B, n, h, dₖ) -> (B, h, n, dₖ)
        # 把 head 维提到前面,这样后面对每个 head 独立做注意力
        def split_heads(t):
            return t.view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (B, h, n, dₖ)

        # 注意力分数:Qᵢ Kᵢᵀ / √dₖ,在最后两维做矩阵乘 -> (B, h, n, n)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if self.causal:
            # 上三角(未来位置)置 -inf,softmax 后权重=0,见第四节
            mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)                    # (B, h, n, n) 每行和为1
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = attn @ v                                      # (B, h, n, dₖ)
        # 合头:(B, h, n, dₖ) -> (B, n, h, dₖ) -> (B, n, d)
        out = out.transpose(1, 2).contiguous().view(B, n, d)
        return self.out_proj(out)                           # 过 W_O,(B, n, d)
```

注意几个实现要点,都是新手常踩的坑:

- **`transpose` 后必须 `.contiguous()` 再 `view`**:`transpose` 只改 stride 不改内存布局,`view` 要求连续内存,否则报错。
- **缩放用 `d_head ** 0.5` 不是 `d_model ** 0.5`**:每个头工作在 dₖ=d/h 维,上一章的方差推导针对的是单个头的内积维度,缩放因子必须是 √dₖ。用错成 √d 是常见 bug。
- **softmax 在最后一维(`dim=-1`)**:scores 的形状是 (B,h,n,n),最后一维是"被关注的 key 位置",对它归一化才对。

### 2.3 为什么多头 > 单头:表达力来源的三个层次

"多头更好"经常被一句"不同头关注不同东西"带过,但这背后有三个可以说清楚的层次:

**层次一:子空间分解(subspace)。** 单头的相似度 `qᵀk` 是在整个 d 维里算内积。多头先用 `W_Qⁱ, W_Kⁱ` 把 d 维**投影到不同的 dₖ 维子空间**,在各子空间里独立算内积。这等于让模型学 h 套不同的相似度度量。形式上,第 i 个头算的是 `(W_Qⁱ x)ᵀ(W_Kⁱ y) = xᵀ (W_Qⁱᵀ W_Kⁱ) y`,这里 `W_Qⁱᵀ W_Kⁱ` 是一个 d×d 的、秩至多为 dₖ 的双线性形式(bilinear form)。**单头只有一个秩 ≤ d 的双线性形式;多头是 h 个秩 ≤ dₖ 的双线性形式之"集合"**,各自能聚焦于一类关系而不互相污染。

**层次二:多个独立的 attention 模式并行(类似集成)。** 每个头算出一个完全独立的 n×n 注意力图 `Aᵢ`。一个头可以是"指向前一个 token"的对角带状图,另一个可以是"每个动词指向其主语"的稀疏图。如果只有单头,这些模式必须在**同一张**注意力图里折中;多头让它们**各占一张图**。这有点像集成学习里多个弱学习器各管一摊,最后由 W_O 加权融合。

**层次三:W_O 的可学习融合。** 拼接后的 `[head₁;…;head_h]` 过 W_O,这一步不是简单求和:W_O 可以学到"对位置 t 的最终表示,头 3(指代头)的贡献占多少、头 7(语法头)占多少"。**W_O 把"h 个子空间的并行结论"重新组合回 d 维主干**,是多头能真正发挥作用的关键——没有它,各头的输出只是被生硬拼在一起。

**一个重要的诚实补充:** 经验研究发现训练好的模型里**很多头是冗余的**,可以剪掉相当比例的头而几乎不掉性能(见第七节导读 Michel et al. 2019、Voita et al. 2019)。这说明"多头"的收益不全在"每个头都学到独特关系",还在于**训练初期提供了更丰富的优化路径**(更容易找到好解),以及子空间分解带来的优化便利。这不矛盾:多头让训练更容易找到好模型,即便最终很多头可以被裁掉。

### 2.4 复杂度与参数量:精确算一遍

**参数量(只算投影,不含 bias)。** MHA 有四个投影:Q、K、V 各一个 d×d,加 W_O 一个 d×d:

```
MHA 参数 = 4 · d²          (W_Q, W_K, W_V, W_O 各 d×d)
```

注意这个结果**与头数 h 无关**——这就是 2.1 节"参数量几乎不变"的精确版本。切成多少头,投影矩阵总大小都是 4d²(每个头 dₖ=d/h,h 个头合起来还是 d)。

**计算复杂度。** 分两块:

```
(1) 四次投影:X(n×d) 乘 d×d 矩阵,共 4 次     → O(n · d²)
(2) 注意力本身(所有头加总):
    QKᵀ:  h 个头,每个 (n×dₖ)·(dₖ×n) = n²dₖ   → h·n²·dₖ = n²·d
    A·V:  同理                                  → n²·d
    合计                                         → O(n²·d)

总计:O(n·d² + n²·d)
```

**这两项的此消彼长,是后面所有长上下文工作的根:** 当 n ≪ d(短序列、大模型),投影项 n·d² 主导,瓶颈是矩阵乘;当 n ≫ d(长序列),注意力项 n²·d 主导,**n² 的显存(要存 n×n 的注意力矩阵)和算力成为墙**。这正是 09 章高效注意力、FlashAttention(MLSys 课)要打破的对象。

**显存的隐形大头:** 注意力矩阵 A 的形状是 (B, h, n, n)。它要在前向中保存以供反向传播,显存是 O(B·h·n²)。h 在参数量里不出现,但在**激活显存**里实打实地乘了进去——这是为什么长序列下显存爆炸,以及 FlashAttention 通过"不显式存 A"来省显存的动机。

## 三、前馈子层 FFN:逐位置的非线性与"记忆"

### 3.1 为什么必须有 FFN:补上缺陷一

回到第一节的缺陷一:注意力是线性混合。FFN(Feed-Forward Network,也叫 position-wise FFN)就是来提供**逐位置的非线性变换**。它的定义朴素得令人意外——就是一个两层 MLP,但有两个关键设计:

```
FFN(x) = W₂ · σ(W₁ · x + b₁) + b₂
         W₁ ∈ ℝ^{d_ff × d}      (升维:d → d_ff)
         W₂ ∈ ℝ^{d × d_ff}      (降维:d_ff → d)
         σ = 非线性激活(ReLU / GELU / SwiGLU)
         通常 d_ff = 4 · d
```

**设计点一:"position-wise"——对序列里每个 token 独立、同一套权重地作用。** FFN 不跨位置混信息(那是注意力的活),它只对每个位置的 d 维向量做同一个 MLP。所以它的输入输出形状都是 (B, n, d),在 n 这一维上是"广播"同一个 MLP。

**设计点二:中间放大到 4×(d_ff = 4d),再降回来。** 这个"先胖后瘦"的瓶颈结构是表达力的来源。直觉:`W₁` 把 d 维特征投到一个 4d 维的高维空间,在那里每个神经元对应一个"特征探测器",`σ` 做非线性筛选(ReLU 把没激活的清零),`W₂` 再把激活后的高维表示压回 d 维。**没有中间放大,两层线性夹一个非线性的表达力很弱;放大到 4d 给了足够多的"中间特征槽位"。** 4× 是经验上的甜点(原论文 d=512, d_ff=2048),不是定理。

**为什么说 FFN 是"记忆"?** 有一个深刻的视角(Geva et al. 2021,见导读):把 FFN 看成 **key-value 记忆**。`W₁` 的每一行是一个 "key"(模式),输入 x 与它点积大表示"匹配上了这个模式";过激活后,`W₂` 的对应列是这个 key 关联的 "value"(要写回主干的信息)。于是 FFN ≈ "查一张存在权重里的关联记忆表":输入触发若干 key,把它们的 value 加权写回。**模型存储的大量事实性知识("巴黎是法国首都")主要就压在 FFN 的权重里**,而不是注意力里。这也解释了下一节为什么 FFN 是参数大户。

### 3.2 激活函数:从 ReLU 到 GELU 到 SwiGLU

原版用 ReLU(`σ(z)=max(0,z)`)。现代模型几乎都换成更平滑的:

- **GELU**(Gaussian Error Linear Unit, BERT/GPT-2 起):`GELU(z) = z · Φ(z)`,Φ 是标准正态 CDF。直觉:不是硬性把负值清零(ReLU),而是按"这个值有多大概率应该通过"来软门控。处处可导、负区间有非零小梯度,优化更顺。
- **SwiGLU**(PaLM/LLaMA 起,见导读 Shazeer 2020):这是个**门控**变体,改变了 FFN 的结构:

```
SwiGLU-FFN(x) = W₂ · ( Swish(W₁·x) ⊙ (V·x) )
                Swish(z) = z · sigmoid(z)
                ⊙ 是逐元素乘
```

注意它有**三个**矩阵 W₁、V、W₂(多了个门控分支 V)。一个分支算"内容"`V·x`,另一个分支算"门"`Swish(W₁·x)`,逐元素相乘做选择性放行。为了**保持总参数量与原 FFN 相当**,SwiGLU 通常把每个矩阵的中间维从 4d 缩到约 **8d/3 ≈ 2.67d**(三个矩阵 × (8/3)d ≈ 8d² ≈ 原来两矩阵 × 4d = 8d²)。LLaMA 就是这么做的(待核:LLaMA 具体取 d_ff,各模型尺寸不同)。

### 3.3 参数量:FFN 是大户

标准 FFN(两矩阵,d_ff=4d):

```
FFN 参数 = W₁(d_ff × d) + W₂(d × d_ff) = 2 · d · d_ff = 2 · d · 4d = 8 · d²
```

对比上一节:**MHA 是 4d²,FFN 是 8d²。一个 block 里 FFN 占了约 2/3 的参数。** 这呼应 3.1 节"知识压在 FFN 里"的说法——参数多的地方就是存东西多的地方。

把两者加起来,**一个 Transformer block(不含 LN 和 bias)的参数量**:

```
block 参数 ≈ 4d²(MHA) + 8d²(FFN) = 12 · d²
```

这是个特别有用的随手估算公式。比如 GPT-2 small:d=768,L=12 层,主干参数 ≈ 12 × 768² × 12 ≈ 0.85 亿(再加 embedding 等约凑到 1.24 亿,即著名的 124M)。**记住 "每层 ≈ 12d²、参数主要在主干 block" 这个量级感,06 章 scaling law 会反复用到。**(待核:GPT-2 124M 的精确分解,这里是量级估算。)

## 四、因果掩码:用一个上三角 −∞ 实现自回归

GPT 这类**自回归(autoregressive)** 模型,训练时要求位置 t 的预测**只能看见 ≤ t 的 token**,不能偷看未来(否则就是抄答案,推理时根本没有未来 token 可看)。在注意力里实现这一点极其简单且优雅:

```
做 softmax 之前,把 scores 矩阵的"未来位置"(列 j > 行 i)置为 −∞:
    scores[i, j] = −∞   当 j > i

softmax(−∞) = 0   →   位置 i 对所有未来位置 j 的注意力权重恰好为 0
```

为什么用 −∞ 而不是直接把权重清零?因为 softmax 是**先指数化再归一化**的:`exp(−∞)=0`,这个 0 会让该位置**不参与归一化分母**,从而保证剩下的(过去 + 当前)位置权重之和仍为 1。如果在 softmax 之后再清零未来位置,那些位置的权重已经被算进分母了,剩下的权重和会 < 1,语义就错了。**所以 mask 必须加在 softmax 之前的 logits 上。** 这是新手常见错误。

掩码矩阵就是一个 n×n 的上三角(不含对角线)布尔阵,代码里就是 `torch.triu(..., diagonal=1)`(见 2.2 节代码)。

一个值得注意的细节:**双向模型(BERT 这类编码器)不加因果掩码**,每个位置能看全句,适合"理解"任务(分类、抽取);自回归模型加因果掩码,适合"生成"。同一套注意力机制,有没有这个上三角 mask,就分出了两大类模型范式。(实际训练里还有 padding mask 处理变长 batch,机制相同——把 padding 位置的 logits 置 −∞,这里不展开。)

## 五、残差 + LayerNorm:Pre-LN vs Post-LN 的梯度分析

这是本章数学含金量最高、也最区分"会用"和"讲透"的一节。残差连接(`x + sublayer(x)`)和 LayerNorm(对每个 token 的 d 维特征做标准化)是让 Transformer 能堆几十上百层的两个稳定器。问题不在"用不用",而在 **LayerNorm 放在残差的哪一侧**。这一个摆放位置的差别,决定了能不能稳定训练深层网络。

### 5.1 两种摆放

```
Post-LN(原版 Transformer, 2017):
    x_out = LayerNorm( x + Sublayer(x) )        # LN 在残差相加之后

Pre-LN(GPT-2 起的现代默认):
    x_out = x + Sublayer( LayerNorm(x) )        # LN 在子层输入处,残差通路上没有 LN
```

LayerNorm 本身:对每个 token 的 d 维向量,减均值除标准差,再放缩平移:`LN(x) = γ ⊙ (x−μ)/√(σ²+ε) + β`(μ、σ² 是这 d 维上算的)。它的作用是把每层激活的尺度拉回可控范围,防止逐层漂移。

### 5.2 核心区别:残差通路上有没有"挡路的非线性归一化"

把一个 L 层网络的前向沿残差展开。**Pre-LN** 的递推是 `xₗ₊₁ = xₗ + Fₗ(xₗ)`(Fₗ 是"LN + 子层"的合成),从输入 x₀ 展开:

```
x_L = x₀ + Σ_{l=0}^{L-1} Fₗ(xₗ)
```

**残差通路是一条从 x₀ 直达 x_L 的"恒等高速路",中间没有任何东西挡着。** 反向传播时,损失对底层 x₀ 的梯度:

```
∂L/∂x₀ = ∂L/∂x_L · ( I + Σ ∂Fₗ/∂x₀ )
```

那个 **`I`(恒等项)保证了无论网络多深,顶层梯度都能原封不动地直达底层**——这条路不衰减。即使所有 `∂Fₗ` 都很小,梯度也有 `I` 这条底,不会消失。

**Post-LN** 不同:`xₗ₊₁ = LN(xₗ + Fₗ(xₗ))`。**LayerNorm 横在残差通路的正中间。** 反向传播经过每一层都要乘上一个 LayerNorm 的雅可比 `∂LN/∂·`。LayerNorm 的雅可比不是恒等,它的作用之一是把输出方差归一化到 1——这意味着当输入方差很大时,它的雅可比会把梯度按比例**缩小**。L 层叠起来,梯度要连乘 L 个这样的雅可比,**量级会随深度漂移**(放大或缩小),没有那条干净的 `I` 高速路兜底。

### 5.3 量级推导:为什么 Post-LN 深层会炸,需要 warmup

给一个能算的粗略模型(沿 Xiong et al. 2020 "On Layer Normalization in the Transformer Architecture" 的分析思路,见导读)。

**Post-LN 残差累加导致输入方差随层数线性增长。** 假设每个子层输出 `Fₗ(xₗ)` 的各分量近似独立、方差约为常数 σ_F²,残差相加 `xₗ + Fₗ(xₗ)`:若不归一化,方差会逐层累加,到第 l 层方差 ≈ l·σ_F²(线性增长)。Post-LN 每层用 LN 把它压回 O(1),但**LN 的雅可比量级 ≈ 1/√(输入标准差) ≈ 1/√l**。反向连乘这些雅可比:

```
顶层(第 L 层)附近的参数梯度 ~ O(1)
底层(第 1 层)附近,经过 L 层 LN 雅可比连乘,梯度量级被一路放大/失衡
→ 结论:Post-LN 中,靠近输出层的梯度远大于靠近输入层(或反之失衡),
   且这种失衡随 L 增大而恶化。
```

Xiong 等人的具体结论是:**Post-LN 中最后一层附近的梯度量级约为 O(d·√(ln d)) 级别,与层数耦合,初始即很大;** 在训练最开始(参数随机、激活尺度未校准)这个大梯度直接把训练打飞(loss 发散或 spike)。**这就是原版 Transformer 必须配 learning-rate warmup(前几千步把 lr 从 0 缓慢线性升上去)的根本原因**——用极小的 lr 熬过最初梯度量级失控的阶段,等激活尺度被慢慢校准好。去掉 warmup,Post-LN 大概率训练不起来。(待核:O(d·√(ln d)) 的精确系数,这里给的是该文结论的量级形式。)

**Pre-LN 则把这个量级失衡基本消掉。** 因为残差通路上的那个 `I` 提供了与深度无关的梯度直达路径,各层梯度量级大致均衡、不随 L 爆炸。Xiong 等人证明 **Pre-LN 在初始化时各层梯度量级是良态(well-behaved)、不依赖于 warmup**。所以 Pre-LN **可以不用 warmup、直接上较大 lr、稳定训练几十上百层**——这正是它成为现代默认的原因(GPT-2/3、LLaMA、几乎所有大模型)。

### 5.4 Pre-LN 的代价与一个补丁

天下没有免费午餐。Pre-LN 的代价是:**残差主干 `x_L = x₀ + Σ Fₗ(xₗ)` 的方差随层数累加而增长**(因为残差上没有 LN 来压它)。到很深时,主干的"信号尺度"越来越大,而每个子层新加的增量相对越来越小——这会让**深层的有效更新变弱**,理论上对最终表达力略有损失(有研究认为这是 Pre-LN 在同等深度下有时略逊于精调好的 Post-LN 的原因)。工程上的常见补丁:

- 在最后一层之后**补一个 final LayerNorm**(`x_final = LN(x_L)`),把累加方差拉回来——几乎所有 Pre-LN 实现(含 nanoGPT)都这么做,见下一节代码。
- 用 **DeepNorm**(对残差分支乘放大系数 α 并相应缩小部分权重初始化)或**残差/输出投影按 1/√(2L) 缩放**等方案,既保留 Post-LN 表达力又获得稳定性——这是 05 章初始化部分的内容(那里会讲为什么按 1/√(2·层数) 缩放残差)。

**一句话总结这一节:** Post-LN 把 LN 放在残差通路上 → 梯度被 L 个 LN 雅可比连乘、初始量级失衡 → 需 warmup。Pre-LN 把 LN 移到子层输入、残差通路保留恒等 `I` → 梯度直达底层、量级与深度解耦 → 免 warmup、能堆深,代价是主干方差膨胀,用 final LN 补救。

## 六、把它们拼成一个完整 Block(Pre-LN)的 PyTorch 实现

现在把 MHA(第二节)、FFN(第三节)、因果掩码(第四节)、Pre-LN 残差(第五节)拼成一个标准的、可直接堆叠的 Transformer block,再堆成一个小 GPT。这段代码与 nanoGPT 的结构基本一致(见导读),去掉了无关样板。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """ position-wise FFN:升维 4× -> GELU -> 降维 """
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        d_ff = mult * d_model
        self.fc1 = nn.Linear(d_model, d_ff)      # W₁: d -> 4d
        self.fc2 = nn.Linear(d_ff, d_model)      # W₂: 4d -> d
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))   # 逐位置作用

class TransformerBlock(nn.Module):
    """ Pre-LN Transformer block:残差通路干净,梯度直达 """
    def __init__(self, d_model, n_heads, causal=True, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)         # 注意力子层的输入 LN
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)         # FFN 子层的输入 LN
        self.ffn = FeedForward(d_model, mult=4, dropout=dropout)

    def forward(self, x):
        # Pre-LN:LN 在子层输入处;残差 x + ... 上没有 LN(恒等高速路)
        x = x + self.attn(self.ln1(x))           # 子层1:自注意力
        x = x + self.ffn(self.ln2(x))            # 子层2:前馈
        return x
        # 对比 Post-LN 写法(原版):
        #   x = self.ln1(x + self.attn(x))
        #   x = self.ln2(x + self.ffn(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)   # token 嵌入
        self.pos_emb = nn.Embedding(max_len, d_model)      # 位置嵌入(04 章会换成 RoPE)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, causal=True, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)                  # final LN:补救 Pre-LN 主干方差膨胀(5.4 节)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # 输出到词表 logits
        self.max_len = max_len

    def forward(self, idx, targets=None):
        # idx: (B, n) 的 token id
        B, n = idx.shape
        pos = torch.arange(n, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]  # (B, n, d)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)                                       # 最后一层后的 LN
        logits = self.head(x)                                  # (B, n, vocab)
        if targets is None:
            return logits, None
        # 自回归交叉熵:位置 t 预测 t+1(因果 mask 保证没偷看未来)
        # 注意:targets 需由调用方右移一位传入(targets[t]=idx[t+1]),本函数内部不做 shift
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

**几个把整层串起来的关键点:**

- **嵌入相加**:token 嵌入 + 位置嵌入,逐位置相加进主干。这里用可学习位置嵌入(简单但不能外推到 max_len 外),04 章会讲为什么以及怎么换成 RoPE。
- **block 顺序**:先注意力(跨位置混信息)后 FFN(逐位置非线性),两个子层各自被 Pre-LN 残差包裹。堆 n_layers 次。
- **`ln_f`(final LN)缺它会怎样**:去掉它,Pre-LN 主干在深层方差膨胀,输出 logits 尺度失控,训练会变差——这是 5.4 节理论的工程体现。
- **训练目标**:`cross_entropy` 在每个位置预测下一个 token,因果 mask(在 MHA 里)保证位置 t 的表示没用到 t 之后的信息,所以这个"预测下一个"是诚实的(推理时确实只有过去)。

## 七、设计权衡与常见坑

**头数 h 怎么选——一个真实的权衡。** h 越大,每个头维度 dₖ=d/h 越小。dₖ 太小(比如 d=768、h=96 → dₖ=8)会让每个头的子空间表达力不足、注意力分辨率差;h 太小又回到"单头打架"。实践常取 dₖ ∈ [64, 128](所以 d=768 配 h=12 → dₖ=64,GPT-2 就这么配)。**记住:增 h 不增参数(总是 4d²),但增 h 会增激活显存(n×n 矩阵多了 h 这个倍数,见 2.4 节)。** 还有推理侧:KV cache 显存 ∝ h·dₖ = d,但 09 章的 MQA/GQA 会让多个 Q 头共享 K/V 头来省这部分——那是另一个权衡。

**坑一:缩放因子用错维度。** 反复强调:多头里缩放是 √dₖ(=√(d/h))不是 √d。我见过的最隐蔽 bug 之一。

**坑二:mask 加在 softmax 之后。** 必须加在 logits 上(置 −∞),不能 softmax 后清零(破坏归一化,见第四节)。

**坑三:LayerNorm 的归一化维度搞错。** LN 是对**特征维 d**(每个 token 自己的 d 维向量)做,不是对序列维 n、也不是对 batch 维。`nn.LayerNorm(d_model)` 是对的。和 BatchNorm(对 batch 维统计)的区别——Transformer 用 LN 不用 BN,因为序列长度可变、batch 内 token 不独立同分布,BN 的 batch 统计在这里不稳定且不适合自回归推理(逐 token 生成时没有 batch 统计)。05 章会展开 LN vs BN vs RMSNorm。

**坑四:Pre-LN 忘了 final LN。** 见 5.4 和第六节,深层会出问题。

**坑五:把 FFN 当配菜删掉或缩得太小。** FFN 是 2/3 的参数和大部分知识所在(第三节)。把 d_ff 从 4d 砍到 d 省参数,会显著掉性能——省错了地方。

**坑六:残差相加前后形状/dtype 不一致。** 残差要求 `x` 和 `sublayer(x)` 同形状同 dtype,混精度训练里尤其注意 LN 通常要在 fp32 下算(数值稳定),这是 05/混精度相关内容。

**权衡总览:Pre-LN(稳、免 warmup、能堆深、略损表达力)vs Post-LN(表达力上限可能略高、但需 warmup 且深层难训)。** 除非你在复现老论文或有特殊理由,**现代默认无脑选 Pre-LN + final LN**。

## 八、动手练习

**练习 1(参数量推导,基础)。** 设 d_model=d,头数 h,FFN 放大倍数 m(默认 4),不计 bias 与 LayerNorm 参数。
(a) 推导 MHA 的总参数量,并解释为什么它与 h 无关。
(b) 推导标准 FFN 的参数量(用 d 和 m 表示)。
(c) 写出一个 block 的总参数量(用 d、m 表示),代入 m=4 验证 ≈ 12d²。
(d) 进阶:LayerNorm 的参数量是多少(每个 LN 有 γ、β 各 d 维)?一个 Pre-LN block 有几个 LN?说明为什么 LN 参数相对 12d² 可忽略。
*提示:MHA 四个 d×d 投影;FFN 两个 d×(md) 投影;别忘了 W_O。*

**练习 2(多头表达力,分析)。** 用 2.3 节的双线性形式视角回答:
(a) 写出第 i 个头计算的相似度 `qᵀ(·)k` 中那个 d×d 矩阵的表达式,并说明它的秩上界是多少、为什么。
(b) 论证:h 个秩 ≤ dₖ 的双线性形式之集合,在表达"多种不同关系"上为什么优于单个秩 ≤ d 的双线性形式(提示:从"互不干扰地聚焦不同关系"和"W_O 的可学习融合"两个角度)。
(c) 结合 2.3 节末尾的"很多头可被剪枝"事实,说明"多头有用"和"多头冗余"为什么不矛盾。

**练习 3(Pre/Post-LN 梯度路径,核心)。**
(a) 写出 Pre-LN 下 `x_L` 关于 `x₀` 的展开式,指出反向传播中那个保证梯度不消失的恒等项是怎么来的。
(b) 写出 Post-LN 下从 `x_{l+1}` 到 `x_l` 的反向传播会乘上什么雅可比,定性说明为什么 L 层连乘后梯度量级会随深度失衡。
(c) 用一句话解释:为什么 Post-LN 需要 learning-rate warmup 而 Pre-LN 基本不需要。
(d) 编码验证:用第六节代码,分别搭一个 24 层的 Pre-LN 和 Post-LN MiniGPT(随机初始化、不训练),前向一个 batch 后做一次 `loss.backward()`,打印第 1 层和第 24 层 `attn.qkv_proj.weight.grad` 的范数(`.norm()`)。观察 Post-LN 两层梯度范数的比值是否远偏离 1、而 Pre-LN 是否接近。
*提示:`for name, p in model.named_parameters(): if p.grad is not None: print(name, p.grad.norm().item())`。这个实验能让你亲眼看到第五节的理论。*

**练习 4(因果掩码,编码 + 数值)。**
(a) 给一个 4×4 的随机 scores 矩阵,手动加因果 mask(上三角置 −∞),手算每行 softmax,验证:第 0 行只有第 0 个位置权重为 1,第 3 行四个位置权重和为 1。
(b) 故意写一个"错误版":先 softmax 再把未来位置权重清零(不重新归一化),对比第 2 行的权重和,说明它为什么 < 1、错在哪。
(c) 把 2.2 节 MHA 的 `causal=True`,喂同一个序列两次:第二次把序列**末尾 token 改掉**,检查**第 0 个位置**的输出是否完全不变。解释为什么因果性保证了这一点(它对训练并行有什么意义?)。

## 九、源码 / 论文导读

按"先读哪篇的哪部分"给:

- **Vaswani et al. 2017, "Attention Is All You Need"**:读 §3.1(scaled dot-product,上一章已透)、**§3.2.2(Multi-Head Attention 的定义,本章第二节对应)、§3.3(Position-wise FFN,本章第三节)、§5.3(为什么要 warmup——这正是 Post-LN 的症状,虽然原文没用"Post-LN"这个词)**。原版 Transformer 是 Post-LN,这点要带着第五节的眼光去读。
- **Xiong et al. 2020, "On Layer Normalization in the Transformer Architecture"**:本章第五节梯度量级分析的来源。重点读它对 Post-LN 末层梯度量级与层数耦合、Pre-LN 梯度良态的定理与图(看那张"Post-LN 需 warmup、Pre-LN 不需要"的曲线对比)。
- **Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories"**:本章 3.1 节"FFN 是记忆"的来源。读它把 W₁ 行当 key、W₂ 列当 value 的论证。
- **Shazeer 2020, "GLU Variants Improve Transformer"**:本章 3.2 节 SwiGLU 的来源。读它对比 ReLU/GELU/GLU/SwiGLU 的那张表,以及"为何把 d_ff 调到 ~8d/3 保参数量"。
- **Michel et al. 2019 "Are Sixteen Heads Really Better than One?" / Voita et al. 2019 "Analyzing Multi-Head Self-Attention"**:本章 2.3 节"头冗余/可剪枝"以及"不同头学不同关系(语法头、指代头)"的实证来源。
- **开源实现首选 nanoGPT(karpathy/nanoGPT)**:看 `model.py` 里的 `CausalSelfAttention`(对应本章 MHA,注意它怎么用一个 `c_attn` 大 Linear 出 QKV、怎么 `view`+`transpose` 切头、怎么用 `F.scaled_dot_product_attention` 或手写 mask)、`MLP`(对应 FFN)、`Block`(对应本章第六节的 Pre-LN block,看它 `x = x + self.attn(self.ln_1(x))` 的 Pre-LN 写法)、以及 `GPT.__init__` 里那个 `ln_f`(对应 5.4 节 final LN)。这是和本章代码最接近、最值得逐行对照的实现。
- **HF transformers**:看 `GPT2Block` / `LlamaDecoderLayer`。注意 **GPT-2 用 Post-LN 风格?——实则 GPT-2 已是 Pre-LN(它把 LN 移到了子层前)**,而 LLaMA 是 Pre-LN + RMSNorm + SwiGLU,可作为"现代 block"的对照样本。(待核:逐版本的 LN 位置以源码为准,各实现命名易混。)

## 十、小结与承上启下

这一章把上一章那个孤立的注意力函数,补齐成了一个能堆叠、能训练的**完整 Transformer 层**:

- **多头**:把 d 维切成 h 个 dₖ=d/h 的子空间,各自做注意力再用 W_O 融合。**参数量恒为 4d²、与 h 无关**;收益来自子空间分解、并行多种关系、可学习融合三个层次(也来自更易优化,即便很多头最终冗余)。
- **FFN**:补上注意力缺的逐位置非线性,先升维 4× 再降回。**参数量 8d²、占一个 block 的 2/3,是模型知识/记忆的主体**。
- **因果掩码**:softmax 前对未来位置置 −∞,一行实现自回归。
- **残差 + LayerNorm 摆放**:**Pre-LN 让残差通路保留恒等 `I`、梯度直达底层、免 warmup、能堆深**(代价是主干方差膨胀,用 final LN 补救);Post-LN 梯度量级随深度失衡、需 warmup。这是从"会用"到"会调深层模型"的分水岭。
- **随手估算**:一个 block ≈ 12d² 参数;注意力 O(n²d) 时间 + O(n²) 显存——记住这两个数,后面全程要用。

**承上**:本章用的还是可学习的**位置嵌入**(`pos_emb`),它简单但不能外推到训练长度之外,而且把"位置"硬加在输入上不够优雅。**启下**:04 章会先证明"自注意力本身对输入置换不变(打乱 token 顺序,输出只跟着置换)",所以位置信息必须显式注入;然后从正弦绝对编码一路推到 **RoPE**(把位置编进 Q/K 的旋转,使内积只依赖相对距离),那是现代主流。05 章会回到本章第五节埋的伏笔——把 LayerNorm 换成 RMSNorm、把残差缩放(1/√(2L))讲透、把 Adam/AdamW 与初始化补全,真正回答"这一层堆几十上百层后怎么稳定训起来"。06 章的 scaling law 则会用本章的 12d² 参数估算去推"模型多大、数据多少、算力多少最优配比"。

到这里,你手里已经有了一块**完整的、可堆叠的乐高积木**。Transformer 的宏伟之处恰恰在于:整个 GPT/LLaMA 就是把这块积木**几乎不变地**叠几十上百次,再配好位置编码、归一化和优化——后面的章节都是在打磨这块积木的某个面,以及研究"叠多少块、怎么叠"的规律。
