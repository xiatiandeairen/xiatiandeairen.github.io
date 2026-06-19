---
title: "Capstone:从零手写一个小 GPT 并训练"
slug: "4-16"
collection: "ai-research-compass"
group: "大模型算法专家课程"
order: 4016
summary: "这一章把你从\"分头读懂了注意力、block、位置、归一化、解码这些零件\"带到\"把它们焊成一个能在你笔记本上跑起来、loss 真的会降、能吐出像样文本的完整 GPT,并且每一行你都知道在算什么、为什么这么写\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T05:58:01.000Z"
updatedAt: "2026-06-19T05:58:01.000Z"
---
> 这一章把你从"分头读懂了注意力、block、位置、归一化、解码这些零件"带到"把它们焊成一个能在你笔记本上跑起来、loss 真的会降、能吐出像样文本的完整 GPT,并且每一行你都知道在算什么、为什么这么写"。前面 15 章拆开讲的每个组件,这里要按正确的接线方式串成一个系统,而"接线方式"本身——维度怎么对齐、梯度怎么从 logits 流回 embedding、第一个 batch 该看到什么数字——往往是把"懂原理"变成"会做模型"的最后一公里,也是踩坑最密集的一公里。读完你会有一份能直接 `python train.py` 的最小 GPT,和一套"先 overfit 一个 batch 再放数据"的验证方法论,这套方法论比任何单个组件都更值钱。

## 一、动机:为什么"会拆"不等于"会装"

到这一章为止,你已经分别掌握了:语言建模为何是 next-token 预测(00)、token 怎么变成向量(01)、`softmax(QKᵀ/√dₖ)V` 怎么从动机推到公式(02)、多头与一整层 block 怎么拼(03)、位置信息怎么注入(04)、归一化/激活/残差怎么让深层训得动(05)、自回归交叉熵目标怎么用一次前向产生 T 个监督信号(06)。这些是零件。

但**一个能跑的 GPT 不是零件的简单堆叠,它是一组必须精确对齐的接口**。新手把组件分别实现对了,装起来却跑不通或者跑出垃圾,几乎总是栽在下面这些"装配"问题上,而这些问题在单独讲某个组件时根本不会出现:

- **维度在长链条里悄悄错位。** embedding 出来是 (B,T,d),过 N 层 block,到输出投影,再到 cross_entropy,任何一步把 d 和 V 搞混、把 T 和 B 转置反了,要么报错要么静默地学错东西。
- **梯度路径断了你看不见。** tie weights(输出投影与 embedding 共享权重)接错、norm 放错位置、residual 漏加,模型不报错,只是 loss 卡着不降——这是最折磨人的一类 bug。
- **训练循环里的"暗坑"。** 没做 warmup 导致开局发散、没裁剪梯度导致偶发 NaN、学习率和 batch 不匹配、loss 该看的是 per-token 平均还是 sum——这些不在任何"组件"章节里,但决定你能不能训出东西。
- **生成时的状态管理。** 训练是并行喂整段序列,生成是一个 token 一个 token 地往外吐,两种模式共用一套 forward,接口怎么设计才不别扭。

**这一章的核心价值,是把"系统装配"本身当成一门要讲透的手艺。** 我们的策略是:先给出一个最小但完整、每个组件都用现代 LLM 标准形态(Pre-LN + RMSNorm + SwiGLU + 因果 MHA + tied embedding)的 GPT 实现;然后给一套**逐组件验证 + 先 overfit 一个 batch** 的调试方法论,让你能确信"实现是对的"再去烧 GPU;最后讨论参数量/显存/训练曲线怎么读,以及怎么扩展和做消融。代码以 nanoGPT(Karpathy)为蓝本——它是把这套数学落成可运行系统的公认最佳起点——但我们会把每个设计选择回扣到前面的章节,讲清"为什么是这样,不是别样"。

**约定**:全程用 PyTorch。记号沿用前面章节——B=batch、T=序列长(time)、d=d_model(隐藏维)、V=词表大小(vocab)、h=头数、L=层数、d_ff=FFN 隐层维。

## 二、数据与分词:字符级起步,理解 token→id→batch 的全链路

### 2.1 问题:训练的第一个输入张量长什么样

在写模型之前,先把"数据怎么变成喂进模型的张量"这条链打通。这一步看着简单,却定义了 V(词表大小,决定 embedding 和输出层的尺寸)和训练样本的构造方式。我们用**字符级分词(character-level tokenization)**:把每个不同字符当作一个 token。这是最简单、零依赖、最适合 capstone 验证的选择——它跳过了 BPE 的复杂度(01 章已讲透 BPE),让你能把注意力全放在模型和训练上。

字符级的代价我们心里要有数(回扣 01):序列会很长(一个英文单词 ≈ 4-5 个 token 而非 1 个),语义粒度细,模型得花容量去学"字母怎么拼成词";好处是 V 极小(英文语料约 65-100 个不同字符,**待核**,取决于具体语料),embedding 和输出层都很小,适合小模型快速 overfit 看效果。等你验证完实现正确,把 tokenizer 换成小 BPE(用 `tiktoken` 的 `gpt2` 编码,V≈50257)就能直接处理真实语料,模型代码一行不用改——这正是"tokenizer 与模型解耦"的好处。

### 2.2 机制:从原始文本到 (X, Y) 训练对

字符级 tokenizer 只是一个双向字典 + 两个函数:

```python
import torch

class CharTokenizer:
    """字符级分词器:每个不同字符一个整数 id。"""
    def __init__(self, text):
        chars = sorted(list(set(text)))      # 语料里出现的所有不同字符,排序保证可复现
        self.vocab_size = len(chars)         # 这就是 V
        self.stoi = {c: i for i, c in enumerate(chars)}   # char -> id
        self.itos = {i: c for i, c in enumerate(chars)}   # id -> char

    def encode(self, s):                     # 字符串 -> id 列表
        return [self.stoi[c] for c in s]

    def decode(self, ids):                   # id 列表 -> 字符串
        return ''.join(self.itos[i] for i in ids)
```

把整个语料 encode 成一条长 id 序列,再切成训练样本。**关键是 X 和 Y 的错位关系**——这正是 06 章自回归目标的数据侧体现:位置 t 的输入要预测位置 t+1 的 token,所以 Y 就是 X 整体右移一位。

```python
def get_batch(data, block_size, batch_size, device):
    """
    data:       一维 LongTensor,整个语料的 token id 序列。
    block_size: 上下文长度 T(模型一次能看多少 token)。
    返回 X,Y:   各 (B, T)。Y 是 X 右移一位 —— 位置 t 的目标是 X 的第 t+1 个 token。
    """
    # 随机取 B 个起点,每个起点截 T+1 个 token(多取 1 个用来做 Y 的最后一位)
    ix = torch.randint(len(data) - block_size, (batch_size,))     # B 个起点
    x = torch.stack([data[i     : i + block_size]     for i in ix])  # (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # (B, T) 右移一位
    return x.to(device), y.to(device)
```

举个具体例子彻底钉死错位关系。设语料片段是字符 `"hello"`,block_size=4:

```
原始 id 序列:  h  e  l  l  o          (假设 id 分别是 1 2 3 3 4)
X (输入):     [h, e, l, l]  = [1,2,3,3]   位置 0,1,2,3
Y (目标):     [e, l, l, o]  = [2,3,3,4]   位置 0,1,2,3
                                          ↑
含义:看到 "h" 要预测 "e";看到 "he" 要预测 "l";看到 "hel" 预测 "l";看到 "hell" 预测 "o"。
一个长度 T 的样本,因此产生 T 个 (上下文 → 下一字符) 监督信号 —— 这就是 06 章说的"信号密度 ≈100%"。
```

**新手坑(第一个)**:很多人把 Y 也做成 (B,T) 但忘了右移,或者在模型里又 shift 一次,导致"双重 shift"或"看到答案"。本章的约定是:**get_batch 直接产出对齐好的 X,Y,模型 forward 拿到 X 算出每个位置的 logits,直接和 Y 算 cross_entropy,模型内部不再 shift。**(这和 06 章 §3.4 里"logits 和 input_ids 自己 shift"是两种等价写法,二选一,别两个都做。)我们这里采用"数据侧 shift",更直观。

### 2.3 分析:这一步决定了什么数字

- **V = tokenizer.vocab_size**:直接决定 embedding 表 (V×d) 和输出投影 (d×V) 的大小。字符级 V≈65,这两个层都很小;BPE 的 V≈50k,它俩会成为小模型里的参数大头(下面算参数量时会看到)。
- **block_size = T**:决定注意力的 O(T²) 开销和位置编码的范围。capstone 取 T=128 或 256 即可。
- **一个 epoch 的样本数**:语料 N 个 token,理论上能取 N−T 个重叠样本,数据利用率极高。小语料(如莎士比亚约 1.1M 字符,**待核**)足够把小模型训出明显效果。

## 三、模型骨架:把六个组件焊成一个 GPT

现在搭模型。整体数据流是一条直线,我先用一张图把维度标全,然后逐块给代码——**装配的本质就是维度对齐**,把图刻进脑子,接线就不会错:

```
input_ids (B, T)                                        ← 整数 token id
  │
  ├─ token embedding:    查表 (V, d)        → (B, T, d)  ← 01 章:id 变向量
  ├─ position embedding: 查表 (T_max, d)     → (B, T, d)  ← 04 章:注入顺序信息
  │   x = tok_emb + pos_emb                  → (B, T, d)  ← 两者相加
  │
  ├─ block × L (每块都是 Pre-LN 残差):        → (B, T, d)  ← 03/05 章
  │     x = x + Attn(RMSNorm(x))   (因果 MHA)
  │     x = x + FFN(RMSNorm(x))    (SwiGLU)
  │
  ├─ final RMSNorm:                          → (B, T, d)  ← 输出前再归一化一次(现代标配)
  │
  └─ lm_head (输出投影, d→V, 与 token emb tie)→ (B, T, V)  ← 每个位置对全词表的打分(logits)
        │
        └─ cross_entropy(logits, targets)    → 标量 loss   ← 06 章自回归目标
```

注意三个"全局"设计,它们贯穿整个模型,先点明:

1. **位置编码这里用可学习绝对位置嵌入(learned absolute, 04 章 §3),不用 RoPE。** 为什么 capstone 选它?因为它最简单、最直观(就是再加一张查找表),足以让小模型工作,且让你看清"位置信息=往 token 向量上加一个依赖位置的向量"这件事。RoPE 更强(尤其外推),我们把"换成 RoPE"留作练习——这正是消融的好题目。
2. **输出投影与 token embedding 共享权重(weight tying)。** lm_head 的权重矩阵 (V×d) 与 token embedding 表 (V×d) 同形,直接共享同一张表(下面 §3.4 专门推)。这不是省事,是有原理的。
3. **每个 block 用 Pre-LN(05 章结论):norm 在残差分支内,跳跃连接全程裸露。** 加上输出前的 final norm,这是现代 decoder-only LLM 的标准形态。

### 3.1 复用前面章节的组件(RMSNorm / 因果 MHA / SwiGLU-FFN)

这三个组件 05 章和 03 章已经从零推导并实现过,这里直接给出与之一致的精简版,把它们当作"已验证的零件"装进来。每个零件后面括号标注它来自哪一章、解决什么。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """05 章:只 re-scale 不 re-center。比 LayerNorm 省一遍归约,现代 LLM 默认。"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 只有 γ,没有 β(05 章:赌掉 re-centering)
    def forward(self, x):
        # 统计量(平方均值)务必在 fp32 算,避免低精度下溢出/丢精度(05 章常见坑)
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight   # 归一化后乘可学习 γ


class CausalSelfAttention(nn.Module):
    """02/03 章:多头因果自注意力。QKV 合并投影 + 因果 mask + 输出投影 W_O。"""
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)    # 一次 matmul 出 Q,K,V(03 章工程要点)
        self.proj = nn.Linear(dim, dim, bias=False)       # W_O
        self.dropout = dropout

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)            # 各 (B,T,d)
        # 切头:(B,T,d) -> (B,h,T,d_head),把 head 维提前以便各头独立注意
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # 用 PyTorch 2.0 内置融合算子:数学等价于 softmax(QKᵀ/√d_head + causal_mask)V(02 章)
        # is_causal=True 自动加下三角因果 mask,底层走 FlashAttention,省显存又快
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                 # (B,h,T,d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, d)   # 合头回 (B,T,d)
        return self.proj(out)                              # 过 W_O


class SwiGLU_FFN(nn.Module):
    """05 章:门控 FFN。FFN(x)=(Swish(xW₁)⊙(xV))W₂,隐层缩到 ⅔×4d 补偿多出的矩阵。"""
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        hidden = int(2 / 3 * mult * dim)                  # ⅔ 缩放(05 章 §3:对齐参数量)
        hidden = 32 * ((hidden + 31) // 32)               # 取整到 32 倍数,GPU 友好
        self.w1 = nn.Linear(dim, hidden, bias=False)      # 升维(门 gate)
        self.v  = nn.Linear(dim, hidden, bias=False)      # 升维(值 value)
        self.w2 = nn.Linear(hidden, dim, bias=False)      # 降维
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.v(x)))   # Swish=SiLU,⊙ 逐元素乘
```

`F.scaled_dot_product_attention` 是 02 章末尾提到的"那一行"。capstone 里我们直接用它:它和手写的 `softmax(scores)·V` 数学完全等价,但底层是融合 kernel(FlashAttention),不显式构造 (B,h,T,T) 的注意力矩阵,省下 O(T²) 显存。如果你想看手写版,02/03 章有完整实现——**建议先手写跑通一次,确认你懂这一行在算什么,再换成内置算子。**

### 3.2 一个 Transformer block(Pre-LN 残差)

把注意力子层和 FFN 子层用 Pre-LN 残差包起来,这就是 05 章 §五的标准块,原样搬来:

```python
class Block(nn.Module):
    """Pre-LN + 残差。norm 在残差分支内部,跳跃连接 x 全程裸露(05 章:梯度高速路)。"""
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = CausalSelfAttention(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn   = SwiGLU_FFN(dim, dropout=dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # 子层1:先 norm 再注意力,结果加回裸 x
        x = x + self.ffn(self.norm2(x))     # 子层2:先 norm 再 FFN,结果加回裸 x
        return x
```

**为什么这个写法的梯度是干净的(回扣 05)**:展开整个网络,任意一层的输入 x 都以 `x + (分支)` 的形式直达下一层。把 L 层串起来,从输出对第 ℓ 层输入求导,链式法则里会有一项是"所有 `∂(x+F)/∂x = I + ∂F/∂x` 连乘",其中那个**恒等项 I 保证至少有一条不衰减的梯度通路**直通底层。Pre-LN 把 norm 塞进 F 内部(`F = Attn(RMSNorm(·))`),所以跳跃路径上没有任何 norm 去缩放梯度,这条 I-高速路完整保留——这就是 Pre-LN 不需要繁琐 warmup 也能训深的根本原因。如果改成 Post-LN(`x = RMSNorm(x + Attn(x))`),norm 跨在跳跃连接上,梯度高速路被 norm 的 Jacobian 反复缩放,深层就容易梯度不稳。

### 3.3 完整 GPT:嵌入 → L 层 → 输出

把所有零件组装成完整模型。这是本章的核心代码,逐行注释:

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65          # V,由 tokenizer 决定
    block_size: int = 256         # T,最大上下文长度
    n_layer: int = 6              # L,Transformer block 层数
    n_head: int = 6               # h,注意力头数
    n_embd: int = 384             # d,隐藏维(须能被 n_head 整除:384/6=64)
    dropout: float = 0.1

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # —— 嵌入层(01 + 04 章)——
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token 查表 (V,d)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # 可学习绝对位置 (T_max,d)
        self.drop = nn.Dropout(cfg.dropout)
        # —— L 层 block(03 + 05 章)——
        self.blocks = nn.ModuleList(
            [Block(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)]
        )
        self.norm_f = RMSNorm(cfg.n_embd)                          # 输出前 final norm(现代标配)
        # —— 输出投影(06 章)——
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # d -> V
        # —— 权重共享(weight tying):lm_head 和 token embedding 用同一张权重(§3.4 推导)——
        self.lm_head.weight = self.tok_emb.weight    # 二者共享内存,梯度自动累加
        # —— 初始化 ——
        self.apply(self._init_weights)
        # GPT-2 的残差投影缩放初始化:深层残差累加导致方差随层数膨胀,
        # 把每个残差分支末端的投影按 1/√(2·L) 缩小,抵消 L 层累加的方差增长(§5 数值分析)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('w2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)   # GPT-2 默认 std=0.02
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T) token id。targets: (B, T) 或 None。
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长 {T} 超过 block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)                 # (T,) 位置索引 0..T-1
        x = self.tok_emb(idx) + self.pos_emb(pos)                # (B,T,d):token 向量 + 位置向量
        x = self.drop(x)
        for block in self.blocks:                                # 过 L 层
            x = block(x)
        x = self.norm_f(x)                                       # (B,T,d) final norm
        logits = self.lm_head(x)                                 # (B,T,V) 每个位置对全词表打分

        if targets is None:                                      # 推理/生成模式:不算 loss
            return logits, None
        # 训练模式:每个位置一个 V 类分类,展平后算交叉熵(06 章 §3.4)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),    # (B*T, V)
            targets.view(-1),                     # (B*T,)
            ignore_index=-100,                    # padding 位不计入(本例无 pad 可忽略)
        )
        return logits, loss
```

几个**装配级的关键点**,每个都是新手会卡住的地方:

- **`pos_emb(pos)` 广播加到 `tok_emb(idx)` 上。** tok_emb 是 (B,T,d),pos_emb(pos) 是 (T,d),PyTorch 自动在 batch 维广播相加。位置向量和 batch 无关(同一个位置不管哪个样本都加同一个向量),这正是绝对位置编码的语义。
- **forward 同时服务训练和生成。** 给 targets 就返回 loss(训练),不给就只返回 logits(生成时只关心最后一个位置的 logits 用来采样)。一个 forward 两用,接口干净。
- **整个 forward 没有任何显式 shift。** 因为我们在 get_batch 里已经把 Y 右移好了(§2.2 的数据侧 shift)。logits[:, t] 是模型看了 idx[:, :t+1] 后对"下一个 token"的预测,正好对应 targets[:, t]=idx 的第 t+1 个。维度天然对齐,直接 cross_entropy。
- **因果性在哪里保证的?** 在 `CausalSelfAttention` 的 `is_causal=True`。没有它,位置 t 能看到 t 右边的 token,就等于"预测时看到了答案",loss 会异常地低、生成时却完全废掉——这是仅次于 shift 错位的第二大经典 bug。

### 3.4 推导:weight tying 为什么成立,省多少参数

`self.lm_head.weight = self.tok_emb.weight` 这一行让输入嵌入和输出投影共享同一张 (V×d) 矩阵。这不是工程偷懒,是有清楚的数学动机,讲透它:

**动机一:输入和输出生活在同一个"token↔向量"空间,理应用同一组基。** token embedding 做的是 `id i → 向量 E_i`(第 i 行)。输出投影做的是 `隐藏向量 z → 各 token 的分数`,第 i 个 token 的 logit 是 `z · W_i`(W 的第 i 行/列)。如果我们要求"隐藏向量 z 和 token i 的嵌入 E_i 越接近,token i 的 logit 越高",那最自然的打分就是 `logit_i = z · E_i`——也就是**直接拿嵌入矩阵当输出投影**。这把"编码一个 token"和"预测一个 token"约束成同一套表示,语义上自洽。

**动机二:省下一大块参数,且经验上略涨点。** 两个层各自独立时,合计 2·V·d 个参数;共享后只有 V·d。在小模型 + 大词表时这块占比惊人。算笔账,用 BPE 的 GPT-2 small 量级:V=50257,d=768。

```
token embedding:  V·d = 50257 × 768 ≈ 38.6 M 参数
lm_head 独立:     V·d ≈ 38.6 M 参数
若不 tie,光这两层就 ≈ 77.2 M;tie 之后只剩 ≈ 38.6 M,省了 ≈ 38.6 M。
GPT-2 small 总参数 ≈ 124 M(待核),省下的 38.6 M 约占总量的 31%!
```

**这就是 tie weights 在大词表小模型上最实在的收益:把"输入嵌入 + 输出投影"两块大表压成一块。** Press & Wolf(2017)的论文还报告了它通常带来困惑度的小幅改善(共享表示有正则化效果),但对 capstone 而言,"省 30% 参数还不掉点"已经足够说服力。

**梯度怎么处理?** 共享后,这张矩阵在前向里被用了两次(一次查表、一次投影),反向时两条路径的梯度**自动累加**到同一个 `.grad` 上(PyTorch 的 autograd 对共享参数天然如此)。你什么都不用做,这正是 `self.lm_head.weight = self.tok_emb.weight` 这种"让两个 Module 指向同一个 Parameter"写法的妙处。

**坑(第三个)**:如果你给 lm_head 加了 bias,tie 就只 tie 了 weight,bias 仍是独立的——这通常无害,但要清楚。另外,tie 要求 lm_head 和 tok_emb 的形状转置匹配(nn.Linear 的 weight 是 (out, in)=(V, d),nn.Embedding 是 (V, d),正好同形,直接赋值即可,不需要手动转置)。

## 四、训练循环:AdamW + warmup-cosine + 梯度裁剪

模型搭好了,现在让它学。训练循环里每一个部件——优化器选 AdamW、学习率为什么要 warmup 再 cosine 衰减、为什么必须裁梯度——都不是惯例,而是有原因的(对应 07 章优化主题)。我先给完整循环,再逐个拆解为什么。

### 4.1 完整训练循环代码

```python
import math
import torch

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """warmup-cosine 学习率调度:先线性升,再余弦降到 min_lr。"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps          # 阶段1:线性 warmup(0 -> max_lr)
    if step > max_steps:
        return min_lr                                      # 阶段3:训练后期保持 min_lr
    # 阶段2:余弦退火,decay_ratio 从 0 平滑到 1,系数从 1 余弦降到 0
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ∈ [0,1]
    return min_lr + coeff * (max_lr - min_lr)

def train(model, train_data, val_data, cfg, device,
          max_steps=5000, warmup_steps=100, batch_size=64,
          max_lr=3e-4, min_lr=3e-5, weight_decay=0.1, grad_clip=1.0):
    # —— AdamW,且只对 2D 权重做 weight decay,不对 norm 的 γ / embedding 做(§4.3)——
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,   'weight_decay': weight_decay},
         {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=max_lr, betas=(0.9, 0.95), eps=1e-8,            # GPT 类常用 β2=0.95(比默认 0.999 小)
    )
    model.train()
    for step in range(max_steps):
        # 1) 取一个 batch(数据侧已 shift 好,§2.2)
        X, Y = get_batch(train_data, cfg.block_size, batch_size, device)
        # 2) 前向 + 算 loss
        logits, loss = model(X, Y)
        # 3) 反向
        optimizer.zero_grad(set_to_none=True)              # set_to_none 比置 0 省内存且更快
        loss.backward()
        # 4) 梯度裁剪:把全局梯度范数限制在 grad_clip 以内,防偶发大梯度炸成 NaN(§4.4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # 5) 按调度设置本步学习率,再 step
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        # 6) 周期性看验证 loss(下面 §4.5)
        if step % 500 == 0:
            val = estimate_loss(model, val_data, cfg, device)
            print(f"step {step}: train {loss.item():.4f}  val {val:.4f}  lr {lr:.2e}  "
                  f"ppl {math.exp(val):.2f}")
    return model

@torch.no_grad()
def estimate_loss(model, data, cfg, device, iters=50):
    """多个 batch 平均,降低验证 loss 的随机波动。"""
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        X, Y = get_batch(data, cfg.block_size, batch_size=32, device=device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()
```

### 4.2 为什么是 AdamW 而不是 SGD

Transformer 几乎一律用 Adam 系优化器,不是迷信。**根本原因:Transformer 的损失面在不同参数方向上曲率差异极大(病态/ill-conditioned),且不同层、不同参数的合适步长差几个数量级。** SGD 用单一全局学习率,要么对陡方向太大(发散)要么对平方向太小(不动);Adam 用每个参数各自的二阶矩 `v = E[g²]` 做自适应缩放,等效于给每个参数一个自己的学习率,对这种病态、稀疏梯度的目标鲁棒得多。

Adam 的更新(为完整性写出,07 章有详推):

```
mₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ              一阶矩(梯度的滑动平均,方向)
vₜ = β₂·vₜ₋₁ + (1−β₂)·gₜ²             二阶矩(梯度平方的滑动平均,尺度)
m̂ₜ = mₜ/(1−β₁ᵗ),  v̂ₜ = vₜ/(1−β₂ᵗ)   偏差校正(开局 m,v 从 0 起,要除以 1−βᵗ 补偿)
θₜ = θₜ₋₁ − lr · m̂ₜ / (√v̂ₜ + ε)      每个参数除以自己的 √v̂,实现逐参数自适应步长
```

**W(AdamW)的关键修正**:把 weight decay 从梯度里**解耦**出来,直接作用在参数上,而不是混进 gₜ 里。原始 Adam 把 L2 正则当成梯度的一部分(`g += λθ`),但这会被 `√v̂` 那个自适应分母缩放,导致"正则强度被 Adam 的自适应步长扭曲"。AdamW 改成:

```
θₜ = θₜ₋₁ − lr · ( m̂ₜ/(√v̂ₜ+ε) + λ·θₜ₋₁ )    ← weight decay 项 λ·θ 独立,不过 √v̂
```

这样 weight decay 的强度和梯度自适应解耦,正则化行为可预测。**Loshchilov & Hutter(2017)指出这个修正在 Transformer 上稳定且有效,现在是标配。** PyTorch 的 `torch.optim.AdamW` 已经实现了解耦版,直接用即可。

**β₂ 为什么常取 0.95 而非默认 0.999?** 0.999 意味着二阶矩用约 1/(1−0.999)=1000 步的窗口做平均,对梯度尺度的变化反应迟钝;语言模型训练里梯度尺度会随训练阶段变化,0.95(窗口约 20 步)更灵敏、不易在尺度突变时失稳。这是 GPT 系常见配置(**待核**,具体值各家略有差异)。

### 4.3 为什么 weight decay 不作用于 norm 和 embedding

代码里把参数分成两组:`p.dim()>=2` 的(各种权重矩阵)做 weight decay,`p.dim()<2` 的(RMSNorm 的 γ、各种 bias、一维参数)**不做**。原因:

- **weight decay 的本意是约束权重矩阵的规模、防过拟合**,对承担主要表达的 2D 矩阵有意义。
- **对 RMSNorm 的 γ 做 decay 是有害的**:γ 是控制每个特征尺度的增益,把它往 0 拉等于无故压缩归一化后的尺度,破坏 norm 的作用。bias 同理,把它往 0 拉没有正则收益却干扰拟合。

这是个细节,但漏了它(对所有参数无差别 decay)会让训练略变差且难诊断。nanoGPT、HF 的实现都这么分组。

### 4.4 为什么必须 warmup,以及梯度裁剪救什么命

**warmup(开头若干步学习率从 0 线性升到 max_lr)解决"开局梯度不可信"的问题。** 训练最开始,参数是随机初始化的,模型对数据一无所知,头几步的梯度方向噪声极大、尺度也可能异常。此时若直接上大学习率,很容易一步迈到损失面的坏区域,触发发散(loss 飙到 NaN)。warmup 让模型在小步长下先"稳一稳",等 Adam 的二阶矩 `v` 累积出可靠的尺度估计、参数走到合理区域,再放大学习率全速学。

- Pre-LN 架构(我们用的)对 warmup 的依赖比 Post-LN 弱(05 章:Pre-LN 梯度高速路干净),但**短 warmup(几十到几百步)仍是免费的保险**,几乎没人省。
- warmup 之后用 **cosine 衰减**把学习率平滑降到 min_lr:训练后期需要小步长精细收敛,余弦曲线比阶梯式 decay 更平滑,经验上效果好且超参少。

**梯度裁剪(`clip_grad_norm_`)防的是"偶发大梯度炸 NaN"。** 即便有 warmup,训练中途偶尔会遇到一个"坏 batch"(比如罕见 token 组合)产生异常大的梯度,一步就把参数推飞、loss 变 NaN,前功尽弃。梯度裁剪把所有参数梯度的**全局 L2 范数**限制在阈值(典型 1.0)内:

```
g_norm = √(Σ_all_params ‖g‖²)                      所有参数梯度拼起来的总范数
若 g_norm > clip:  所有 g ← g · (clip / g_norm)    整体等比缩小,保持方向不变
```

注意它是按**全局范数**等比缩放,不是逐参数截断——保持梯度方向不变,只压总尺度。这是训练稳定性的"保险丝",成本几乎为零,**永远开着**。

### 4.5 验证 loss 与困惑度(PPL):怎么读训练曲线

`estimate_loss` 多取几个 batch 平均,因为单 batch 的 loss 抖动大,平均后才能看清趋势。报告时除了 loss,还应换算成**困惑度(perplexity, PPL)**——它是 loss 的指数,语义更直观(00 章已讲):

```
PPL = exp(loss)        loss 是 per-token 平均交叉熵(自然对数)
```

PPL 的含义是"模型在每个位置平均把概率质量摊在多少个候选 token 上"——**PPL=10 约等于模型每步在 10 个等可能选项里猜**。两个标尺:

- **随机基线**:完全不学时,模型对 V 个 token 均匀分布,loss=ln(V),PPL=V。字符级 V≈65,初始 loss 应在 ln(65)≈4.17 附近,PPL≈65。**第一步打印的 loss 如果不接近 ln(V),说明初始化或实现有问题——这是第一个该检查的数字(下面 §5.1 详述)。**
- **下降目标**:字符级莎士比亚上,一个小 GPT 训练几千步后 val loss 通常能降到 1.4-1.6 量级(**待核**,随模型大小/语料/超参变化),对应 PPL≈4-5,此时生成的文本已经有莎士比亚的句法和词形了。

**怎么判断健康**:train loss 持续下降、val loss 跟着降且与 train 差距不大 = 健康;val loss 先降后升而 train 仍降 = 过拟合(小数据 + 大模型常见,加 dropout 或减小模型);两者都卡着不降 = 实现有 bug 或学习率不对(回到 overfit-one-batch 排查)。

## 五、调试方法论:先 overfit 一个 batch,再烧 GPU

**这一节是整章最该刻进肌肉记忆的内容。** 前面代码看着完整,但你第一次写的版本几乎一定有 bug——维度错、shift 错、mask 漏、tie 接反。怎么在烧几小时 GPU 之前就发现?答案是一套**自底向上、每步可证伪**的验证流程。这套方法论的价值高于任何单个组件,因为它让你能独立调试任何模型。

### 5.1 第一关:形状与初始 loss(花 30 秒,挡掉一半 bug)

写完模型,第一件事不是训练,是跑一个 batch 检查两个数字:

```python
model = GPT(GPTConfig(vocab_size=65)).to(device)
X, Y = get_batch(train_data, 256, 4, device)        # 取 4 个样本
logits, loss = model(X, Y)
print(logits.shape)                                  # 必须是 (4, 256, 65) = (B,T,V)
print(loss.item())                                   # 必须 ≈ ln(65) ≈ 4.17
print(sum(p.numel() for p in model.parameters()))    # 打印总参数量,和你手算的对一下(§6)
```

- **logits 形状不对** → 维度接线错了,回去对 §3 那张数据流图逐层查。
- **初始 loss 远不等于 ln(V)** → 这是个极强的信号。loss=ln(V) 意味着"模型初始输出接近均匀分布",这正是随机初始化该有的状态。如果初始 loss 明显大于 ln(V)(比如 10+),通常是**初始化方差太大**(logits 尺度失控);如果明显小于 ln(V),通常是**信息泄漏**(模型不知怎么看到了答案,十有八九是因果 mask 没生效或 shift 错位让它看到了当前/未来 token)。这一个数字能挡掉初始化和 mask 两大类 bug。

**为什么 ln(V) 是对的?** 初始随机权重下,每个 token 的 logit 近似独立同分布、均值 0,softmax 后接近均匀分布 1/V。交叉熵 = −ln(预测给正确类的概率) ≈ −ln(1/V) = ln(V)。这是个能精确验算的理论值,务必记住。

### 5.2 第二关:overfit 一个 batch(花几分钟,确认能学)

**取一个小 batch,反复在它上面训练,看 loss 能不能降到接近 0。** 这是验证"实现整体正确、梯度能正常流动"的黄金测试:

```python
model = GPT(GPTConfig(vocab_size=65)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
X, Y = get_batch(train_data, 64, 4, device)          # 固定一个小 batch,不再换
for step in range(200):
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step}: loss {loss.item():.4f}")
# 期望:loss 从 ≈4.17 稳步降到接近 0(几个样本 × 几百 token,模型容量足够死记)
```

逻辑很硬:一个只有几个样本的 batch,数据量远小于模型参数量,一个**正确实现且梯度通畅**的模型必然能把它**死记硬背**到 loss≈0(纯记忆,不需要泛化)。

- **loss 降到接近 0** → 恭喜,前向、反向、梯度路径、优化器全都通了。可以放心上全量数据。
- **loss 卡在某个值不动**(比如卡在 4.17 或 2.x) → 梯度没有正常流动。最常见原因:某处 `detach()`/`no_grad` 用错切断了梯度、tie weights 接错、残差漏加、norm 把梯度缩没了。用 `for n,p in model.named_parameters(): print(n, p.grad.norm() if p.grad is not None else None)` 看哪些参数 grad 是 None 或 0,顺藤摸瓜。
- **loss 降但降得极慢/震荡** → 学习率问题,先调 lr 再说。

**没通过这一关,绝不要去跑全量训练。** 全量训练的 loss 曲线噪声大、反馈慢,bug 藏在里面极难定位;overfit-one-batch 是确定性的、快速的、二元的(过/不过),是性价比最高的验证。

### 5.3 第三关:小规模全量训练(花几十分钟,看泛化)

overfit 通过后,上全量数据小跑(几千步),确认 val loss 也在降(§4.5 的健康判断)。这一关验证的是"模型不仅能记,还能学到泛化的规律"。到这里再去调模型大小、上长训练、做消融,就都是在一个已知正确的基座上调了。

### 5.4 这套方法论为什么通用

**"形状/初始 loss → overfit 一个 batch → 小规模全量"这三关,本质是把'整体能不能 work'这个模糊问题,拆成三个互相独立、各自可证伪的子命题:** ① 接线对不对(形状 + 初始 loss)、② 能不能学(overfit)、③ 能不能泛化(全量)。每一关失败都精确指向一类 bug,不会让你面对"loss 不降"这种信息量为零的现象抓瞎。这套思路适用于任何你从零实现的模型,记住它比记住 GPT 的任何细节都重要。

## 六、参数量、显存与计算量:把规模算清楚

做模型必须能徒手估算资源,否则配出来的模型要么显存爆要么大材小用。以本章默认配置(d=384, L=6, h=6, V=65, T=256)为例,把三笔账算清。

### 6.1 参数量:逐部件加总

```
设 d=n_embd, L=n_layer, V=vocab, T_max=block_size, d_ff=SwiGLU 隐层维(≈⅔·4d 取整)

token embedding:   V·d                                   ← 与 lm_head tie,只算一次
position embedding: T_max·d
每层 block:
  注意力:  qkv (d×3d) + proj (d×d)          = 4d²
  FFN:     w1 (d×d_ff) + v (d×d_ff) + w2 (d_ff×d) = 3·d·d_ff
  norm:    2 个 RMSNorm,各 d 个 γ            = 2d         (相对可忽略)
  每层小计 ≈ 4d² + 3·d·d_ff
L 层:      L·(4d² + 3·d·d_ff)
final norm: d
lm_head:    0(与 token emb 共享,不重复计)

总参数 ≈ V·d + T_max·d + L·(4d² + 3·d·d_ff) + 小项
```

代入默认值(d=384, L=6, V=65, T_max=256;d_ff 取 ⅔·4·384≈1024,再取整到 1024):

```
token emb:    65 × 384            ≈ 0.025 M
pos emb:      256 × 384           ≈ 0.098 M
每层注意力:   4 × 384²            ≈ 0.59 M
每层 FFN:     3 × 384 × 1024      ≈ 1.18 M
每层小计:     ≈ 1.77 M
6 层:         6 × 1.77 M          ≈ 10.6 M
─────────────────────────────────
总计 ≈ 10.7 M 参数(约一千万,小模型,单卡甚至 CPU 可训)
```

**两个一般性结论从这笔账里直接看出来**:

1. **block 参数主导 ≈ L·(4d² + 3·d·d_ff),且随 d² 增长。** 把 d 翻倍,block 参数翻 4 倍;把层数 L 翻倍,线性翻倍。这就是 scaling(08 章)里"加宽 vs 加深"的参数代价差异。
2. **FFN 通常比注意力参数更多。** 每层 FFN(3·d·d_ff≈3·d·(8/3 d)=8d²)比注意力(4d²)大一倍——这印证 03 章说的"FFN 是模型参数的主体,不是配菜"。
3. **小词表(字符级 V=65)时,embedding 占比微不足道(0.025M/10.7M<1%);但换 BPE(V=50257)后,光 token embedding 就 ≈19M,在这个 10M 量级的小模型里直接翻倍**——这就是 §3.4 weight tying 在大词表小模型上特别值钱的原因。

### 6.2 显存:训练为什么比推理吃几倍显存

训练显存四大块,缺一不可:

```
① 参数本身:        P 个参数 × 每参数字节数(fp32=4B, bf16=2B)
② 梯度:           和参数同形,= P × 字节数
③ 优化器状态:      AdamW 要存 m 和 v 两份,各和参数同形 = 2 × P × 4B(优化器态常用 fp32)
④ 激活(activations):前向每层的中间结果,反向要用来算梯度。∝ B·T·d·L,且随 batch 和序列长线性涨
```

**关键认知:①②③ 加起来,每个参数在训练时大约占 fp32 下 4(参数)+4(梯度)+8(Adam 的 m,v)=16 字节(混合精度下参数/梯度可降到 2B,但 Adam 态常保 fp32,总量约 12-16B/参数,待核)。** 所以 10.7M 参数光"参数+梯度+优化器态"就要 ≈170MB,对小模型无压力。

**但激活(第④块)才是大模型显存的真正主角**,而且它和 batch_size·block_size 成正比、和参数量无关。注意力若显式构造 (B,h,T,T) 的分数矩阵,这一块随 T² 爆炸——这正是 §3.1 用 `scaled_dot_product_attention`(FlashAttention)的理由:它不物化那个 T×T 矩阵,把注意力激活从 O(T²) 降到 O(T)。capstone 小模型用不上,但你扩到长上下文(10 章)时,激活显存就是你最先撞的墙,**省显存的手段(activation checkpointing、FlashAttention)优先级高于省参数。**

### 6.3 计算量:一次前向的 FLOPs 怎么估

一个粗略但好用的经验法则(Kaplan 等的近似,08 章会用):**一次前向每个 token 的计算量 ≈ 2·P FLOPs(P 为参数量),反向约为前向 2 倍,所以训练一个 token 总计 ≈ 6·P FLOPs。** 来源:每个权重在前向里参与一次乘加(2 FLOPs)。所以训练总算力 ≈ 6 · P · (总训练 token 数)。

```
本例:P≈10.7M。训练 5000 步 × batch 64 × T 256 ≈ 8.2×10⁷ token。
训练总 FLOPs ≈ 6 × 10.7e6 × 8.2e7 ≈ 5.3×10¹⁵ FLOPs ≈ 5.3 PetaFLOP。
现代单卡(几十 TFLOPS 量级)跑这点量是分钟到小时级 —— capstone 完全可行。
```

这个 `6·P·tokens` 公式是 scaling law 章节(08)估算训练成本、推 compute-optimal 配比(Chinchilla)的基石,这里先建立量感。

## 七、采样生成:temperature + top-p 把 logits 变成文本

模型训好了,怎么让它"写字"?生成是**自回归地一个 token 一个 token 往外吐**:把已生成的序列喂进 forward 拿到最后一个位置的 logits,从中采样出下一个 token,拼回序列,再喂进去,循环。核心问题是"怎么从 logits 这个概率分布里挑下一个 token",这就是解码策略(对应 13 章,这里讲透 capstone 必需的两个旋钮)。

### 7.1 机制:三种基础采样及其问题

forward 输出最后一位的 logits ∈ ℝ^V,过 softmax 变成下一 token 的概率分布。怎么从中选?

- **贪心(greedy):永远取概率最大的 token(argmax)。** 确定性,但生成会**重复、单调**——它锁死在局部最优,写不出多样的文本。
- **纯采样(按 softmax 概率直接抽):** 多样性够,但**长尾问题严重**——V 个 token 里那几万个低概率 token 加起来仍有不小的总概率质量,偶尔抽中一个就让句子突然崩坏("胡言乱语")。
- 我们要的是中间地带:**既有多样性,又不让低质长尾 token 有机会出来。** 两个旋钮 temperature 和 top-p 正是干这个的。

### 7.2 temperature:调节分布的"尖锐度"

temperature(温度 τ)在 softmax 之前给 logits 除以 τ:

```
p_i = softmax(logits / τ)_i = exp(z_i/τ) / Σ_j exp(z_j/τ)
```

推导它的作用:τ 越小,logits 被放大,分布越**尖锐**(高分 token 概率被进一步拉高,趋向 argmax);τ 越大,logits 被压缩,分布越**平坦**(趋向均匀,更随机)。两个极限直接验算:

```
τ → 0⁺:  logits/τ → ±∞,最大那个 logit 主导,softmax → one-hot(退化成贪心)
τ → ∞:   logits/τ → 0,所有 exp(0)=1,softmax → 均匀分布(完全随机)
τ = 1:   不改变原分布
```

所以 **τ 是"保守(接近贪心) ↔ 冒险(接近随机)"的连续旋钮**,典型取 0.7-1.0。τ<1 让生成更稳更可预测,τ>1 更天马行空但易跑偏。

### 7.3 top-p(nucleus sampling):动态截断长尾

光有 temperature 不够——它缩放整个分布,但低概率长尾 token 仍有非零概率被抽中。**top-p(核采样,nucleus sampling,Holtzman 等 2019)直接砍掉长尾:只在"累积概率达到 p 的最小 token 集合"里采样。**

机制:把 token 按概率从高到低排序,从最高的开始往下累加,直到累计概率 ≥ p,这个**最小集合**就是"核(nucleus)",只在核内重新归一化后采样,核外 token 概率清零。

```
为什么 top-p 优于固定取 top-k?核的大小随分布形状动态变化:
  分布很尖(模型很确定下一个词):  几个 token 就累计到 p,核很小 → 输出几乎确定
  分布很平(模型不确定):          要很多 token 才累计到 p,核很大 → 保留更多可能
top-k 固定取前 k 个,在"模型很确定"时反而强行塞进低质 token,在"模型很不确定"时又可能砍掉合理选项。
top-p 自适应,这就是它成为主流的原因。
```

### 7.4 代码:带 temperature + top-p 的自回归生成

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_p=0.9):
    """
    idx: (B, T0) 起始上下文(prompt 的 token id)。返回 (B, T0+max_new_tokens)。
    每步:forward 取最后一位 logits → temperature 缩放 → top-p 截断 → 采样 → 拼回。
    """
    model.eval()
    for _ in range(max_new_tokens):
        # 1) 上下文超过 block_size 就只保留最后 block_size 个(模型只能看这么长)
        idx_cond = idx[:, -model.cfg.block_size:]
        # 2) forward,只取最后一个位置的 logits(我们只关心"下一个" token)
        logits, _ = model(idx_cond)             # (B, T, V)
        logits = logits[:, -1, :]               # (B, V) 最后一位
        # 3) temperature 缩放
        logits = logits / temperature
        # 4) top-p:对每个样本,保留累计概率达到 top_p 的最小 token 集合
        probs = F.softmax(logits, dim=-1)                          # (B, V)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)                   # 累计概率
        # 标记要移除的:累计已超 p 的(保留刚好达到 p 的那个,移除其后所有)
        remove = cum - sorted_probs > top_p     # exclusive 前缀和 > p 的位置移除:即保留"累计首次达到/超过 p"的那个 token 及其之前的全部,移除其后所有
        sorted_probs[remove] = 0.0              # 核外清零
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)     # 核内重新归一化
        # 5) 在核内按概率采样,再映射回原始 token id
        choice = torch.multinomial(sorted_probs, num_samples=1)    # (B,1) 在排序空间里的位置
        next_id = torch.gather(sorted_idx, -1, choice)             # (B,1) 映射回真实 token id
        # 6) 拼回序列,进入下一步
        idx = torch.cat([idx, next_id], dim=1)
    return idx

# 用法:从一个起始字符开始生成 500 个字符
# start = torch.tensor([[tokenizer.stoi['\n']]], device=device)   # (1,1)
# out = generate(model, start, max_new_tokens=500, temperature=0.8, top_p=0.9)
# print(tokenizer.decode(out[0].tolist()))
```

**生成里的关键工程点**:

- **每步重新喂整个(截断后的)上下文做一次完整 forward。** 这是最朴素的实现,O(T²) 每步、总 O(T³) 生成一段——对 capstone 完全够用。生产系统用 **KV cache**(缓存历史 K,V,每步只算新 token 的 Q 并复用缓存,把每步降到 O(T)),那是 10 章/推理优化的主题,这里先用朴素版把机制看清。
- **`idx[:, -block_size:]` 必须截断。** 位置嵌入只有 block_size 个,上下文超长会越界报错;截断保留最近的 block_size 个 token(滑动窗口)。
- **`torch.multinomial` 按给定概率抽样**,配合 `gather` 把"排序空间里的位置"映射回"真实 token id"——这个两步映射是 top-p 实现里最容易写错的地方,务必看懂 `sorted_idx` 的作用。
- **`@torch.no_grad()` + `model.eval()`**:生成不需要梯度(省显存),eval() 关掉 dropout(生成时要确定的网络行为)。

## 八、设计权衡与常见坑

- **位置嵌入用 learned-absolute 的代价:不能外推。** 我们图省事用了可学习绝对位置嵌入(只有 block_size 个),**生成时上下文长度硬上限就是 block_size,一个 token 都超不了**。这是 capstone 的已知妥协。要外推就上 RoPE(04 章),把"换 RoPE"当练习正好体会差异。
- **字符级 vs BPE 的取舍。** 字符级零依赖、V 小、适合验证,但序列长、模型要花容量学拼写、最终质量上限低。验证完务必换 BPE(`tiktoken`)再追质量,模型代码不用改一行——这是 tokenizer 解耦的红利。
- **小数据上大模型必过拟合。** 莎士比亚才 1M 字符,你要是配个几千万参数的模型,val loss 会先降后升。对策:加 dropout、减小模型、或用更大语料。capstone 阶段 10M 参数 + 1M 字符是 OK 的平衡点。
- **千万别跳过 overfit-one-batch 直接全量训练。** 这是本章反复强调的。全量 loss 噪声大反馈慢,bug 极难定位;overfit 是确定性的快速二元测试。省这一步通常要花十倍时间在玄学调试上。
- **初始 loss 不等于 ln(V) 是最强的早期 bug 信号,别放过。** 大于 ln(V)→初始化方差太大;小于 ln(V)→信息泄漏(mask/shift 错)。30 秒能查的事,别等训练几小时才发现。
- **学习率 × batch_size 要协同。** 大 batch 梯度更稳,可配更大 lr(经验上 lr 随 batch 线性或平方根缩放,**待核**,具体规律见优化章)。照搬别人 lr 但 batch 差很多,可能不收敛或发散。
- **梯度裁剪和 warmup 是"保险",永远开着。** 成本几乎为零,救命概率不低。尤其复现别人工作不收敛时,先确认这两个开了、值对不对。
- **tie weights 后别忘了它影响初始化。** 共享的那张表既当 embedding 又当输出投影,std=0.02 的初始化对两个角色都要合适;GPT-2 的经验值在这上面是验证过的,别乱改。
- **eval/train 模式切换别漏。** 生成和验证前 `model.eval()`(关 dropout),训练前切回 `model.train()`。忘了切会让生成带随机 dropout、验证 loss 偏高,诊断时被误导。

## 九、动手练习

1. **(编码题 · 必做)把整套跑通并训练。** 用上面的 tokenizer/get_batch/GPT/train/generate,在莎士比亚字符级语料(nanoGPT 的 `data/shakespeare_char`)上完整跑通。**严格按 §5 三关推进**:先验证 logits 形状是 (B,T,V) 且初始 loss≈ln(V)≈4.17;再 overfit 一个小 batch 看 loss 是否降到接近 0;最后小规模全量训练几千步,记录 train/val loss 和 PPL 曲线,生成 500 字符贴出来。*提示*:第一关任一项不对都不要往下走;d=384/L=6/T=256/lr=3e-4 是稳妥起点。把结果填进下面 §十的报告模板。

2. **(消融题)去掉位置编码 vs 换成 RoPE。** 做两个变体:(a) 把 `pos_emb` 那一项删掉(模型完全不知道顺序);(b) 换成 RoPE(04 章实现,作用在 Q/K 上)。各训同样步数,对比最终 val loss 和生成质量。**预测并验证**:(a) 应明显变差(注意力对排列等变,无位置信息时模型分不清词序,04 章 §1 的结论);(b) 应与 learned-absolute 相当或略好,且理论上能外推超过训练长度——试着让 (b) 生成比 block_size 更长的序列看会不会崩。*提示*:这道题让你亲眼看到位置编码不是可选项,以及不同方案的外推差异。

3. **(消融题)RMSNorm vs LayerNorm vs 无归一化。** 把 `RMSNorm` 分别换成 `nn.LayerNorm` 和"直接去掉 norm",其余不变,各训同样步数。观察:无归一化时深层是否难收敛甚至发散(05 章结论);LayerNorm vs RMSNorm 效果是否接近(应接近,RMSNorm 主要省计算不掉点)。*提示*:无 norm 变体可能需要把学习率调小才不发散,这本身就印证了 norm 的稳定作用。

4. **(分析题 · 推导)推导并验证初始 loss = ln(V)。** 写清楚:随机初始化下,为什么每个位置的预测分布近似均匀、交叉熵期望为 ln(V)。然后用代码验证:新建一个未训练的 GPT,在一个 batch 上算 loss,确认它落在 ln(V) 附近(允许小偏差,因为 logits 不是严格 0)。再做一个对照实验:故意把 `_init_weights` 的 std 从 0.02 调到 1.0,观察初始 loss 是否显著偏离 ln(V),解释为什么(logits 尺度变大 → softmax 变尖 → 偶尔押错代价高 → loss 偏大)。*提示*:这道题让你彻底理解 §5.1 那个"30 秒挡一半 bug"的检查为什么成立。

## 十、小 GPT 实现 + 训练报告模板

做完练习 1,用这份模板写一页报告——**它逼你把"跑通了"量化成"对在哪、数字是多少"**,是把 capstone 沉淀成可复用经验的关键:

```
# 小 GPT 训练报告

## 配置
- tokenizer: 字符级 / BPE,  V = ___
- 模型: d=___, L=___, h=___, d_ff=___, block_size=___, 总参数=___M(手算 vs torch 实测对齐?)
- 训练: max_steps=___, batch=___, max_lr=___, warmup=___, weight_decay=___, grad_clip=___

## 验证三关(§5)
- [ ] logits 形状 = (B,T,V) ?
- [ ] 初始 loss ≈ ln(V) = ___ ?(实测 ___)
- [ ] overfit 一个 batch:loss 降到 ___(应接近 0)
- [ ] 小规模全量:val loss 是否跟随 train 下降?

## 训练曲线
- 初始 train/val loss: ___ / ___    (PPL ___ / ___)
- 最终 train/val loss: ___ / ___    (PPL ___ / ___)
- 是否过拟合(val 先降后升)? ___
- 资源:峰值显存 ___ MB,训练耗时 ___,估算 FLOPs ≈ 6·P·tokens = ___

## 生成样例(temperature=__, top_p=__)
（贴 200-500 字符生成结果）

## 消融结论(练习 2/3)
- 去位置编码: val loss ___ → ___,生成质量变化: ___
- LayerNorm vs RMSNorm: ___
- 一句话结论: ___
```

## 十一、源码 / 论文导读

- **首选实现:nanoGPT(Karpathy)。** 本章代码以它为蓝本,务必对照通读:`model.py` 看 `GPT`(嵌入 + Block + lm_head + weight tying + GPT-2 残差缩放初始化,与本章 §3 一一对应)、`Block`/`CausalSelfAttention`/`MLP`;`train.py` 看训练循环(AdamW 参数分组、warmup-cosine、grad clip,对应本章 §4)、`get_batch`(对应 §2)、`configure_optimizers`(decay/nodecay 分组,对应 §4.3);`sample.py`/`generate` 看 top-k 采样(本章用 top-p,机制相通)。**nanoGPT 是把这门课从数学变成可运行系统的最佳单一入口。**
- **工业级对照:HF transformers `GPT2LMHeadModel`。** `modeling_gpt2.py` 看真实工程实现里 embedding/block/lm_head 的组织、`shift_logits`/`shift_labels`(模型侧 shift,和本章数据侧 shift 等价)、`tie_weights`。看完能理解生产代码和教学代码的差距在哪(主要是 KV cache、混合精度、并行)。
- **可配置消融:x-transformers(lucidrains)。** 把位置编码(absolute/RoPE/ALiBi)、norm 位置(Pre/Post)、激活变体都做成开关,做练习 2/3 的消融对比直接用它最省事。
- **关键论文(读指定部分即可)**:
  - **GPT-2**:Radford 等《Language Models are Unsupervised Multitask Learners》(2019)——看模型结构那节(Pre-LN、初始化、tie weights)和 scaling 描述。
  - **weight tying**:Press & Wolf《Using the Output Embedding to Tie Word Vectors》(2017)——看它对"输入输出嵌入共享"的论证和 PPL 改善,印证本章 §3.4。
  - **AdamW**:Loshchilov & Hutter《Decoupled Weight Decay Regularization》(2017)——看 weight decay 与 Adam 解耦那一节,理解 §4.2 的修正动机。
  - **nucleus sampling**:Holtzman 等《The Curious Case of Neural Text Degeneration》(2019)——看 top-p 的提出和"为什么纯采样/贪心都不好",印证本章 §7。
  - **训练成本估算**:Kaplan 等《Scaling Laws for Neural Language Models》(2020)——看 `C ≈ 6·N·D` 那个估算(N 参数、D token),本章 §6.3 的来源,也是下一章(08 scaling law)的入口。

## 十二、小结与承上启下

这一章我们没有引入任何新机制,而是干了一件同样难、甚至更难的事:**把前面拆开讲的所有零件,按正确的接线方式焊成一个真正能跑、loss 真会降、能生成文本的完整 GPT。** 收尾点三件事:

- **系统装配本身是门手艺。** 维度对齐、梯度路径、训练循环里的 warmup/裁剪、生成时的状态管理——这些"零件之间的接口"不在任何单个组件章节里,却是"懂原理"到"会做模型"的最后一公里。本章把这一公里走完了。每个组件都精确回扣了出处(02 注意力、03 block、04 位置、05 归一化、06 自回归目标、07 优化、13 解码),你现在该能从 input_ids 一路讲到生成文本,中间每一步的维度和动机都说得清。
- **"先 overfit 一个 batch"是本章最该带走的东西。** 形状/初始 loss → overfit 一个 batch → 小规模全量,这三关把"能不能 work"拆成三个独立可证伪的子命题,适用于你将来从零实现的任何模型。它比 GPT 的任何具体细节都更通用、更值钱。
- **你现在手里有一个可改的基座。** 它是后面所有进阶的实验台:08 章讲 scaling law,你可以在这个基座上画自己的"参数/数据/算力 vs loss"小曲线;10 章讲长上下文,你可以把 learned-absolute 换成 RoPE/ALiBi 看外推;11 章讲 PEFT,你可以在这个小 GPT 上挂 LoRA 跑微调;13 章讲解码,你可以扩展 §7 的采样器加 beam search、对比各种策略。

**至此,你不再是"读过 Transformer 论文的人",而是"从零造过一个 GPT 并训出文本的人"。** 这两者的差距,正是这门课从"懂"到"专家"的分水岭。下一章(08 Scaling Laws 与涌现)会告诉你:把这个 10M 的小家伙放大到 10B、100B 时,会发生什么可预测的(loss 沿幂律下降)和不可预测的(能力涌现)事情——而你现在有了一个能亲手验证那些规律起点的实验台。
