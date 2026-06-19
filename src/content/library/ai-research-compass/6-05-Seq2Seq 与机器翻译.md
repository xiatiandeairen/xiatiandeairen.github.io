---
title: "Seq2Seq 与机器翻译:注意力的诞生地(NLP 视角)"
slug: "6-05"
collection: "ai-research-compass"
group: "自然语言处理专家课程"
order: 6005
summary: "这一章把你从\"会用 RNN/LSTM 处理单个序列\"带到\"能从任务角度讲清 seq2seq 为什么是翻译/摘要/对话的统一框架、信息瓶颈如何逼出注意力、神经机器翻译相对统计机器翻译变了什么、以及能徒手实现并讲透 BLEU 的每一项\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T05:58:48.000Z"
updatedAt: "2026-06-19T05:58:48.000Z"
---
> 这一章把你从"会用 RNN/LSTM 处理单个序列"带到"能从任务角度讲清 seq2seq 为什么是翻译/摘要/对话的统一框架、信息瓶颈如何逼出注意力、神经机器翻译相对统计机器翻译变了什么、以及能徒手实现并讲透 BLEU 的每一项"。注意力的内部数学(`softmax(QKᵀ/√dₖ)V`、缩放因子、多头)在算法课 02/03 已经推透,本章只讲它对翻译这个任务意味着什么。

## 一句话引言

机器翻译是 NLP 的"果蝇":几乎每一个序列到序列的核心思想——编码器-解码器、注意力、beam search、自动评测——都是先在翻译上被逼出来,再扩散到摘要、对话、代码生成的。本章从"变长输入映射到变长输出"这个根本问题讲起,推出信息瓶颈,讲注意力如何把翻译还原成"软对齐",再把 SMT→NMT 的范式转变、解码时的覆盖/重复问题、以及 BLEU 的几何平均 × 简短惩罚全部讲到能复现。

## 1. 动机:什么任务需要 seq2seq

### 1.1 一类被长期忽视的任务结构

回顾前面几章的任务结构。文本分类是 `序列 → 单个标签`;序列标注(分词、POS、NER,见第 4 章)是 `长度 n 的序列 → 长度 n 的标签序列`,**输入输出一一对齐、等长**。这两类都可以用"一个编码器 + 一个分类头"解决。

但有一大类任务不满足等长、不满足对齐:

- **机器翻译**:`英文 n 个词 → 中文 m 个词`,n ≠ m,而且词序可能整体重排(英语 SVO、日语 SOV)。
- **摘要**:`原文 500 词 → 摘要 30 词`,输出远短于输入,且不是抽取而是改写。
- **对话**:`用户一句话 → 系统一句回复`,输出长度完全不由输入长度决定。
- **语音识别、问答、代码生成、语法纠错**……

这些任务的共同结构是:**变长输入 → 变长输出,且输出长度不由输入长度决定,输出 token 与输入 token 之间是多对多的软对应**。这就是 sequence-to-sequence(seq2seq)问题。

**关键认识**:seq2seq 不是某个模型,而是一个任务族的抽象。一旦你能把翻译、摘要、对话都写成"读入一个序列、生成另一个序列"的条件概率 `P(y₁...yₘ | x₁...xₙ)`,它们就能共享同一套架构、同一套解码算法、同一套评测思路。这个统一视角是 2014 年之后整个 NLP 的组织原则,也是后来 T5"把一切都变成 text-to-text"(第 7 章)的思想源头。

### 1.2 概率分解:为什么是自回归

seq2seq 要建模的是条件分布 `P(y | x)`,其中 x、y 都是序列。直接建模一个变长序列的联合分布不可行(组合爆炸),标准做法是**链式分解成自回归形式**:

```
P(y₁ y₂ ... yₘ | x) = ∏ₜ₌₁ᵐ  P(yₜ | y₁ ... yₜ₋₁ , x)
```

逐行读这个式子:

- 第 t 个目标词的概率,**条件于已经生成的所有前缀 `y<ₜ` 和整个源序列 x**。
- 模型的工作变成:给定 x 和已生成前缀,预测下一个词的分布(一个 softmax over 词表)。这与第 2 章的语言模型完全同构,**区别只在于多了一个条件 x**。所以 seq2seq = "条件语言模型"。
- 训练时用 teacher forcing:把真实前缀 `y<ₜ` 喂进去算 `P(yₜ|·)`,损失是每一步的交叉熵之和(下面会写)。推理时没有真实前缀,只能用模型自己上一步的输出,这就引出了**曝光偏差(exposure bias)**——训练见的是真前缀、推理喂的是自己生成的可能带错的前缀,分布不一致。这是 seq2seq 的一个根本性裂缝,后面解码和评测都和它有关。

> 自回归分解是无损的(概率论恒等式,任何联合分布都能这么拆),代价是**推理必须顺序进行**:第 t 步要等第 t-1 步的结果。这是 NMT 推理慢的根源,也是后来非自回归翻译(NAT)试图打破的点。

## 2. 经典 RNN seq2seq 与信息瓶颈

### 2.1 编码器-解码器结构

最早把 seq2seq 用神经网络端到端做出来的是两篇 2014 年的工作:Sutskever, Vinyals & Le(2014,Google,"Sequence to Sequence Learning with Neural Networks")和 Cho et al.(2014,提出 GRU 的同一篇)。结构极简:

```
编码器(RNN/LSTM):
  读入 x₁ x₂ ... xₙ , 顺序更新隐藏状态
  h₁ = f(x₁, h₀);  h₂ = f(x₂, h₁);  ... ;  hₙ = f(xₙ, hₙ₋₁)
  取最后一个隐藏状态  c = hₙ   ← 把整句压成一个固定维度向量(context vector)

解码器(RNN/LSTM):
  用 c 初始化解码器隐藏状态  s₀ = c
  s₁ = g(y₀, s₀);  生成 y₁
  s₂ = g(y₁, s₁);  生成 y₂   ← 每一步输出一个词,直到生成 <eos>
```

(RNN/LSTM 的门控与梯度细节见第 3 章,这里把它当成"一个能吃序列、吐隐藏状态的黑箱"。)

注意这个结构里 `c = hₙ` 是**唯一**连接源句和译文的通道。解码器全程只能看着这一个固定维度(比如 512 维)的 c。这就埋下了灾难。

### 2.2 信息瓶颈:长句为什么崩

把整个源句子压进一个固定向量 c,有三个结构性问题(算法课 02 从注意力角度推过,这里从翻译任务角度讲后果):

1. **容量不随输入增长**。c 是固定 512 维。源句 5 个词还是 50 个词都塞进同一个 512 维向量。句子越长,平均每个词分到的"带宽"越窄。实测现象:经典 RNN seq2seq 的 BLEU 随源句长度增加而**显著下滑**,长句翻译质量崩塌(Bahdanau et al. 2015 的图就是为了展示这一点)。

2. **远距离信息衰减**。RNN 顺序更新,`xₙ` 离 c 近、`x₁` 离 c 远。即使 LSTM 缓解了梯度消失,早期 token 抵达 c 时已被反复覆写。Sutskever 那篇当年的一个"黑魔法"是**把源句反转输入**(读 `xₙ...x₁`),让源句开头离 c 更近、和译文开头对齐更好——这个 trick 能涨好几个 BLEU 点,**它能 work 本身就是瓶颈存在的铁证**。

3. **解码每步的信息需求不同,c 却恒定**。生成译文第 1 个词时你最想看源句开头;生成第 5 个词时想看源句对应的那一段。但固定的 c 对每一步提供的信息**完全一样**,无法按需供给。

**一句话总结矛盾**:翻译本质是 token 级的局部对齐问题(目标词 ↔ 源词的对应),却被强行用一个全局固定向量承载。瓶颈不是工程调参问题,是结构错配。

### 2.3 代码:最小可跑的 RNN seq2seq(无注意力)

下面是体现机制的核心骨架(PyTorch),刻意保留信息瓶颈,好和下一节的注意力版对照:

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_src, emb=256, hid=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_src, emb)
        self.rnn = nn.GRU(emb, hid, batch_first=True)
    def forward(self, src):                 # src: [B, n]
        e = self.embed(src)                 # [B, n, emb]
        outputs, h = self.rnn(e)            # outputs:[B,n,hid]  h:[1,B,hid]
        return h                            # 只返回最后隐藏状态 = 信息瓶颈 c

class Decoder(nn.Module):
    def __init__(self, vocab_tgt, emb=256, hid=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_tgt, emb)
        self.rnn = nn.GRU(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab_tgt)
    def forward(self, prev_tok, h):         # 单步: prev_tok [B,1], h [1,B,hid]
        e = self.embed(prev_tok)            # [B,1,emb]
        o, h = self.rnn(e, h)               # o:[B,1,hid]
        logits = self.out(o.squeeze(1))     # [B, vocab_tgt]
        return logits, h

# 训练(teacher forcing)的核心循环:
def train_step(enc, dec, src, tgt, criterion):
    h = enc(src)                            # c = hₙ,整句被压成 h
    loss = 0.0
    prev = tgt[:, :1]                       # 起始 <bos>
    for t in range(1, tgt.size(1)):         # 自回归逐步
        logits, h = dec(prev, h)
        loss = loss + criterion(logits, tgt[:, t])   # 每步交叉熵累加
        prev = tgt[:, t:t+1]                # teacher forcing: 喂真实前缀
    return loss / (tgt.size(1) - 1)
```

注意 `Encoder.forward` 把 `outputs`(每个位置的隐藏状态)**丢掉了**,只返回 `h`。下一节的全部改动,本质就是"别丢 outputs"。

### 2.4 损失函数与量化

训练损失是序列上每一步交叉熵的平均(等价于负对数似然 NLL):

```
L(θ) = − (1/m) Σₜ₌₁ᵐ  log P(yₜ* | y<ₜ* , x ; θ)
```

其中 yₜ* 是参考译文第 t 个真实词。读这个式子:

- 它就是把自回归分解 `∏ P(yₜ|·)` 取负对数变成求和(乘积取 log 变求和),再对长度归一。
- **困惑度(perplexity)** PPL = exp(L),是这个损失的指数;PPL=20 直观理解为"模型在每步平均有效地在 20 个词里犹豫"。
- 复杂度:RNN 解码每步是 O(hid²)(GRU 的矩阵乘),序列长 m 则训练一句 O(m·hid²),且**时间上无法并行**(第 t 步依赖第 t-1 步)。这是 RNN 相对 Transformer 的致命伤——后者训练时整个序列并行(算法课 03)。

## 3. 注意力:把翻译还原成软对齐

### 3.1 核心思想(机制细节指向算法课 02)

Bahdanau, Cho & Bengio(2015,ICLR,"Neural Machine Translation by Jointly Learning to Align and Translate")的破局点朴素到位:**别再把源句压成一个向量**。保留编码器每个位置的隐藏状态 `h₁...hₙ`;解码到第 t 步时,根据当前解码状态 `sₜ₋₁`,**动态算出一个针对这一步定制的上下文向量 cₜ**,只看源句里此刻该看的部分。

形式上(加性/Bahdanau 注意力):

```
对每个源位置 j:  eₜⱼ = vᵀ tanh(W_s sₜ₋₁ + W_h hⱼ)      # 打分:当前解码状态和源位置 hⱼ 有多相关
归一化:          αₜⱼ = softmax_j(eₜⱼ) = exp(eₜⱼ) / Σ_k exp(eₜₖ)   # Σⱼ αₜⱼ = 1
上下文向量:      cₜ = Σⱼ αₜⱼ · hⱼ                       # 对源隐藏状态按权重加权求和
解码:            sₜ = g(yₜ₋₁, sₜ₋₁, cₜ);  P(yₜ|·) = softmax(W [sₜ; cₜ])
```

- `αₜⱼ` 是第 t 个目标词分给第 j 个源词的**注意力权重**,所有 j 上加和为 1——这正是一个**软对齐(soft alignment)**:不是硬性"目标词 t 对应源词 j",而是一个概率分布。
- `cₜ` 随 t 变化,每步都不同,彻底破掉了 §2.2 的"c 恒定"问题;且 cₜ 直接由 `h₁...hₙ` 加权而来,源句开头的信息不必"穿越"整个 RNN 才能用到,破掉了远距离衰减。
- **缩放点积版本** `softmax(QKᵀ/√dₖ)V`、为什么除以 √dₖ、softmax 的雅可比、O(n²) 复杂度——**全部在算法课 02 推过,这里不重复**。从 NLP 角度只需记住:Bahdanau 用的是加性打分(一个小 MLP),Luong et al.(2015)提出更省的乘性/点积打分,Transformer 用的是缩放点积。三者只是打分函数 `eₜⱼ` 的不同写法,软对齐的思想一致。

### 3.2 注意力 = 可视化的词对齐(对翻译任务的独特意义)

注意力对翻译的意义远不止涨点。把 `αₜⱼ` 排成一个 m×n 矩阵(行=目标词,列=源词),它就是一张**可解释的对齐热力图**:

```
        the   cat   sat   on    mat
le      ▓▓    ░     ░     ░     ░
chat    ░     ▓▓    ░     ░     ░
était   ░     ░     ▓     ░     ░
assis   ░     ░     ▓▓    ░     ░
sur     ░     ░     ░     ▓▓    ░
le      ░     ░     ░     ▓     ▓
tapis   ░     ░     ░     ░     ▓▓
```

亮的格子告诉你"模型生成这个法语词时主要在看哪个英语词"。Bahdanau 论文里展示了英法翻译中 `European Economic Area` 被翻成 `zone économique européenne` 时,注意力矩阵呈现**反对角线**——精确捕捉到了形容词后置带来的语序翻转。

这件事的分量:

- **可解释性**。NMT 在它之前是个黑箱端到端系统;注意力第一次让人能"看见"模型在对齐什么,这在 SMT 时代是显式建模、在早期 NMT 是丢失的。第 13 章讲忠实性评测、第 14 章讲可解释性时,这是起点。
- **它逼近了一个隐变量**。SMT 里"对齐"是显式的隐变量(§4),要用 IBM 模型 + EM 专门估计;注意力让对齐变成网络里**可微、可端到端学**的副产品,不用单独训练对齐模型。
- **坑**:注意力权重 ≠ 真实的因果解释。后续研究(Jain & Wallace 2019)指出注意力分布可被扰动而不改变预测,**不能把高注意力权重直接当成"模型因为这个词才这么译"的证据**。可视化是诊断工具,不是因果证明。这个 caveat 第 13/14 章会反复出现。

### 3.3 代码:在 §2.3 上加注意力(关键改动)

只改两处:编码器返回全部 outputs;解码器每步算 cₜ。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnDecoder(nn.Module):
    def __init__(self, vocab_tgt, emb=256, hid=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_tgt, emb)
        self.rnn   = nn.GRU(emb + hid, hid, batch_first=True)  # 输入拼上 cₜ
        self.W_s   = nn.Linear(hid, hid, bias=False)
        self.W_h   = nn.Linear(hid, hid, bias=False)
        self.v     = nn.Linear(hid, 1, bias=False)
        self.out   = nn.Linear(hid * 2, vocab_tgt)             # [sₜ ; cₜ]

    def forward(self, prev_tok, s_prev, enc_outputs):
        # prev_tok:[B,1]  s_prev:[1,B,hid]  enc_outputs:[B,n,hid]  ← 别丢的 outputs
        # ---- 加性注意力打分 ----
        s = s_prev.permute(1, 0, 2)                     # [B,1,hid]
        score = self.v(torch.tanh(self.W_s(s) + self.W_h(enc_outputs)))  # [B,n,1]
        alpha = F.softmax(score, dim=1)                 # [B,n,1]  Σⱼ αⱼ = 1  ← 软对齐
        c_t   = (alpha * enc_outputs).sum(dim=1, keepdim=True)            # [B,1,hid]
        # ---- 解码一步 ----
        e = self.embed(prev_tok)                        # [B,1,emb]
        o, s_new = self.rnn(torch.cat([e, c_t], dim=-1), s_prev)
        logits = self.out(torch.cat([o, c_t], dim=-1).squeeze(1))
        return logits, s_new, alpha.squeeze(-1)         # 返回 alpha 用于可视化
```

`alpha` 就是 §3.2 那张热力图的一行。把每步的 `alpha` 堆叠成 [m, n] 矩阵,`matplotlib` 一个 `imshow` 就能画出对齐图。代价:每步要对全部 n 个源位置打分,解码复杂度从 O(hid²) 变 O(n·hid + hid²),长源句更贵——但相比质量提升完全值得。

> 从这里到 Transformer 只差一步:把"解码器对编码器的注意力"叫 cross-attention,再加上"序列对自身的注意力"叫 self-attention,去掉 RNN 的顺序递归、全靠注意力建模依赖——就是算法课 03 的内容。**注意力诞生于翻译的信息瓶颈,Transformer 是把它推到极致。**

## 4. 范式转变:从 SMT 到 NMT

### 4.1 统计机器翻译(SMT)在做什么

在神经网络之前(约 1990s–2014),主流是统计机器翻译。经典的**噪声信道模型**(IBM,Brown et al. 1993)把翻译写成贝叶斯解码:

```
ŷ = argmax_y  P(y | x) = argmax_y  P(x | y) · P(y)
                                    └翻译模型┘  └语言模型┘
```

读这个分解(它是 SMT 的灵魂):

- **P(y)** 是目标语言的语言模型(通常 n-gram,见第 2 章),保证译文**流畅**(像句人话)。
- **P(x|y)** 是翻译模型,保证译文**忠实**(信息没丢)。注意方向是反的(给定译文 y 生成源 x),这是噪声信道的形式技巧:把"x 是 y 经过含噪信道传过来的"反推回去。
- 翻译模型的核心是**词对齐(word alignment)**:源词和目标词的对应关系,用 IBM Model 1–5 逐步加复杂度(词翻译概率→绝对位置→相对位置/fertility),用 **EM 算法**在大规模双语语料上无监督估计对齐。

实际工程化的是**基于短语的 SMT(phrase-based SMT)**(Koehn et al. 2003):不再逐词,而是抽取**短语对**(phrase pairs)构成短语表(phrase table),每个短语对带多个特征分(翻译概率双向、词汇化概率、短语惩罚……),再加一个**重排序模型(reordering)**处理语序,最后用**对数线性模型**把这些特征加权组合,权重用 MERT 在开发集上调:

```
score(y, x) = Σᵢ λᵢ · featureᵢ(y, x)     # 特征:短语翻译分、LM 分、重排分、长度罚……
```

SMT 的画像:**一堆独立训练的组件拼装**(对齐模型、短语抽取、语言模型、重排序、特征权重调优),每个组件单独优化、用启发式拼起来,流水线长、特征工程重,但可解释、可控、对小语料相对鲁棒。

### 4.2 NMT 改变了什么(范式层面)

神经机器翻译(2014 起,§2/§3)是一次**范式转变**,不是渐进改良:

| 维度 | SMT(短语) | NMT(seq2seq+attn) |
|---|---|---|
| 建模 | 多组件流水线拼装 | 单个端到端神经网络 |
| 翻译单元 | 短语(离散符号) | 连续向量(子词嵌入) |
| 对齐 | 显式隐变量,EM 单独训 | 注意力,端到端可微学出 |
| 目标 | 各组件分别优化 + 对数线性组合 | 统一最大似然 `P(y|x)` |
| 语序 | 显式重排序模型 | 注意力 + 解码器隐式建模 |
| 流畅度 | 受 n-gram LM 局部窗口限制 | 解码器是神经 LM,长程流畅 |
| 泛化 | 短语表是查表,稀疏 | 嵌入空间平滑,相似词共享 |

**为什么 NMT 赢了**(这是范式转变的实质,不是调参):

1. **联合优化 vs 拼装**。SMT 各组件的局部最优拼起来不是全局最优;NMT 用一个损失端到端反传,所有参数为同一目标协同。
2. **连续表示破解稀疏**。短语表是离散查表,没见过的短语就是 0(数据稀疏);嵌入让"沙发"和"长椅"在向量空间相近,泛化天然更好。
3. **流畅度碾压**。SMT 的 n-gram LM 只看局部窗口,译文常局部通顺、整体松散;NMT 解码器是带长程记忆的神经 LM,译文流畅度肉眼可见地好。这是 NMT 最先被用户感知到的优势(Google 2016 上线 GNMT 时的主要卖点)。

代价与新坑(诚实地说):NMT 需要**大双语语料**(几百万句对起),低资源语言下未必赢 SMT;NMT 是黑箱,出错难定位、难打补丁(SMT 可以直接改短语表);NMT 会**流畅地胡说**(译文通顺但意思错或漏译),而 SMT 出错往往"磕巴"反而容易被发现——这个"流畅的错误"问题直接催生了第 13 章的忠实性评测。

## 5. 解码:beam search 与覆盖/重复问题

### 5.1 为什么不能贪心

训练完得到 `P(yₜ|y<ₜ,x)`,推理要找 `ŷ = argmax_y P(y|x)`。在变长序列上精确 argmax 是指数搜索(词表大小 V 的 m 次方),不可行。两个实用方案:

- **贪心(greedy)**:每步取概率最高的词。问题:**局部最优 ≠ 全局最优**。第 1 步选了个看似最优的词,可能把后面整句带歪,且不可回头。
- **beam search(束搜索)**:每步保留概率最高的 k 个**部分序列**(beam,束宽 k),下一步对每个束扩展全部词、再剪枝回 k 个。k=1 退化成贪心,k 越大越接近精确搜索但越慢。这是 NMT 的事实标准解码。

> beam search 的具体算法、长度归一化(length normalization,防止长序列因连乘更多概率而被系统性惩罚)、`length_penalty`、重复惩罚等细节在**算法课 13(解码策略)**讲透,这里只讲它在翻译任务上引出的两个 NLP 特有问题。

### 5.2 翻译特有的两个坑:覆盖不足与重复

注意力 NMT 有两个臭名昭著的失败模式,SMT 时代不存在(因为 SMT 显式追踪每个源词被翻译几次):

- **过翻/漏翻(over/under-translation,覆盖问题)**:源句某些词被翻了好几遍,另一些被完全漏掉。根源:解码每步独立算注意力,**没有任何机制记录"这个源词我已经翻过了"**。
- **重复(repetition)**:译文里同一个词组循环输出(`the the the` / `经济经济发展发展`)。beam search 倾向高概率路径,而模型一旦进入某种局部高概率循环就出不来,束宽越大有时越严重(高概率短而重复的序列被偏好)。

机制级缓解(任务特有,值得记):

1. **覆盖机制(coverage)**(Tu et al. 2016):维护一个累积覆盖向量 `covⱼ = Σ_{t'<t} αₜ'ⱼ`(源词 j 到目前累计被注意了多少),把它喂进注意力打分,**让模型知道哪些源词已被充分翻译、抑制重复关注**。

```
打分加一项:  eₜⱼ = vᵀ tanh(W_s sₜ₋₁ + W_h hⱼ + W_c covₜⱼ)
覆盖损失:    L_cov = Σₜ Σⱼ min(αₜⱼ , covₜⱼ)    # 惩罚对已覆盖源词的再关注(See et al. 2017 的常用形式)
```

2. **解码端禁止重复 n-gram**(`no_repeat_ngram_size`,工程常用)、覆盖惩罚加进 beam 打分、长度归一化(防漏翻导致译文过短)。

3. 这些在 Transformer 时代被部分缓解(self-attention + 更好训练)但**没有根除**——LLM 生成里"复读机"现象仍然存在,缓解手段(repetition penalty、采样)见算法课 13。

## 6. 评测:BLEU 完整推导与实现

翻译评测的根本难题:一个源句有**多种正确译文**(同义改写、语序差异),没有唯一标准答案,人工评测又贵又慢。BLEU(Papineni et al. 2002,IBM)是第一个被广泛接受、与人工判断有不错相关性的自动指标,统治翻译评测二十年。讲透它,既是为了会用,更是为了第 13 章批判它。

### 6.1 直觉与设计目标

BLEU 的核心直觉:**好译文应该和参考译文共享大量 n-gram(连续词片段)**。但只看 n-gram 重叠有两个明显漏洞,BLEU 用两个设计精确堵住:

- 漏洞 A:**重复刷分**。候选译文 `the the the the the` 对参考 `the cat sat`,unigram 命中 `the` 看似很高。→ 用**截断(clipping)** 堵住。
- 漏洞 B:**过短刷分**。候选只输出一个最有把握的词 `the`,精确率 100%。→ 用**简短惩罚(brevity penalty)** 堵住。

### 6.2 修正 n-gram 精确率(modified n-gram precision)

对每个 n(取 n=1,2,3,4),计算**截断后的** n-gram 精确率 pₙ:

```
对候选里每个 n-gram g:
  Count(g)          = g 在候选里出现的次数
  Count_clip(g)     = min( Count(g) ,  maxRef(g) )   # 截断到它在(任一)参考里的最大出现次数
pₙ = ( Σ_g Count_clip(g) ) / ( Σ_g Count(g) )
   = 命中的 n-gram 数(截断) / 候选 n-gram 总数
```

逐步理解截断:候选 `the the the`,参考 `the cat`。`the` 的 `Count`=3,但参考里 `the` 最多出现 1 次,所以 `Count_clip`=min(3,1)=1。于是 p₁ = 1/3,而非 3/3。**漏洞 A 被堵死**:重复一个词不能无限刷分,顶多算它在参考里出现的次数。

为什么取到 4-gram:1-gram 测词汇是否对,高阶 n-gram(2/3/4)测**局部语序和流畅度**是否对。4-gram 是流畅度和数据稀疏(高阶 n-gram 命中率本就低)之间的经验折中。

### 6.3 几何平均:为什么不是算术平均

把 p₁…p₄ 合成一个分数,BLEU 用**几何平均**(通常等权 wₙ=1/4):

```
几何平均部分 = exp( Σₙ₌₁⁴ wₙ · log pₙ )  =  (p₁ · p₂ · p₃ · p₄)^(1/4)     (当 wₙ=1/4)
```

为什么几何平均而非算术平均?**几何平均对"短板"极度敏感**:只要某个 pₙ=0(比如完全没有命中任何 4-gram,译文毫无连贯短语),log pₙ=−∞,整个几何平均=0。算术平均做不到这点(一项为 0 只是拉低不会归零)。这正是想要的:**一个连贯短语都对不上的译文,不该因为单词碰对几个就拿高分**。

> 实践坑:句子级 BLEU 上高阶 pₙ 经常恰好=0(短句很难命中 4-gram),导致整句 BLEU=0,不可用。所以 BLEU **设计上是语料级(corpus-level)指标**:n-gram 计数在整个测试集上累加后再算 pₙ。句子级评测要加平滑(smoothing,如给 0 计数加个 ε),见 6.6。

### 6.4 简短惩罚 BP:完整推导

只有 n-gram 精确率,模型可以靠**输出极短译文**作弊:译文越短,分母越小,越容易全命中(漏洞 B)。注意 BLEU **没有显式的召回项**(因为多参考下召回难定义),所以用简短惩罚(brevity penalty)间接惩罚过短。

设候选总长 c(corpus 级是所有候选长度之和),参考总长 r(每句取与候选长度最接近的那条参考的长度,称 effective reference length,再求和)。BP 定义:

```
        ┌ 1                    , 若 c > r      (译文不短于参考,不罚)
  BP =  ┤
        └ exp(1 − r/c)         , 若 c ≤ r      (译文偏短,指数衰减惩罚)
```

合并写成 `BP = min(1, exp(1 − r/c))`。**推导这个形式为什么合理**,逐条:

1. **边界**:c=r 时,1−r/c=0,exp(0)=1,不罚——长度匹配满分,合理。
2. **单调递减**:c 越小于 r(译文越短),r/c 越大,1−r/c 越负,exp(·) 越小(趋近 0)——越短罚越狠,合理。
3. **为什么用 exp 而不是直接线性 c/r?** 设计者要的是**平滑、温和、且永不为负**的惩罚,且在 c 略小于 r 时惩罚很轻(允许译文比参考略短一点是正常的),在 c 远小于 r 时惩罚急剧加重。exp(1−r/c) 恰好:c/r 从 1 降到 0.5(译文砍半)时,1−r/c 从 0 降到 −1,BP 从 1 降到 exp(−1)≈0.368,惩罚显著但不归零。线性 c/r 在 c 略小时惩罚过重、且缺乏这种"先松后紧"的曲率。
4. **min(1, ·) 的作用**:c>r 时 exp(1−r/c)>1,但**译文偏长不该奖励**(长译文已经因为分母大而 pₙ 偏低被隐式惩罚了),所以截到 1。

把 c=r 取个数验算:c=10, r=12 → exp(1−1.2)=exp(−0.2)≈0.819;c=6, r=12 → exp(1−2)=exp(−1)≈0.368。译文从略短到砍半,BP 从 0.82 掉到 0.37,符合"越短罚越狠"的设计意图。

### 6.5 完整 BLEU 公式与代码

```
BLEU = BP · exp( Σₙ₌₁⁴ wₙ · log pₙ )      其中 wₙ = 1/4
```

完整可跑实现(corpus 级,核心机制全在,加了 6.3/6.4 的截断与 BP):

```python
import math
from collections import Counter

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def modified_precision(cands, refs_list, n):
    """corpus 级修正 n-gram 精确率。cands: list[list[str]]; refs_list: list[list[list[str]]]"""
    num = den = 0
    for cand, refs in zip(cands, refs_list):
        cand_ng = Counter(ngrams(cand, n))
        # 每个 n-gram 在“任一参考”里的最大出现次数,用于截断
        max_ref = Counter()
        for ref in refs:
            ref_ng = Counter(ngrams(ref, n))
            for g, c in ref_ng.items():
                max_ref[g] = max(max_ref[g], c)
        for g, c in cand_ng.items():
            num += min(c, max_ref[g])      # ← clipping,堵漏洞 A
        den += max(sum(cand_ng.values()), 1)
    return num, den                         # 返回分子分母,corpus 级累加在外层已完成

def brevity_penalty(cands, refs_list):
    c = sum(len(x) for x in cands)          # 候选总长
    r = 0
    for cand, refs in zip(cands, refs_list):
        # effective reference length: 取与候选长度最接近的那条参考
        r += min((abs(len(ref) - len(cand)), len(ref)) for ref in refs)[1]
    if c > r:
        return 1.0
    return math.exp(1 - r / c) if c > 0 else 0.0   # ← BP = min(1, exp(1-r/c)),堵漏洞 B

def corpus_bleu(cands, refs_list, max_n=4):
    log_p_sum = 0.0
    w = 1.0 / max_n
    for n in range(1, max_n + 1):
        num, den = modified_precision(cands, refs_list, n)
        if num == 0:
            return 0.0                       # 几何平均:任一 pₙ=0 → BLEU=0
        log_p_sum += w * math.log(num / den)
    bp = brevity_penalty(cands, refs_list)
    return bp * math.exp(log_p_sum)
```

> 复现纪律:不要自己手写 BLEU 报论文分数。学术界用 **sacreBLEU**(Post 2018),它固定了 tokenization、大小写、n-gram 实现,带版本签名(signature),保证跨论文可比。手写 BLEU 因 tokenization 差异能差好几个点,**不可比**。自己实现仅用于理解机制。

### 6.6 BLEU 的局限(给第 13 章埋线)

BLEU 是 NLP 自动评测的奠基石,但它的缺陷恰恰是后续二十年评测演进的动机,务必讲清:

1. **只看 n-gram 表面重叠,完全不懂语义**。`The cat is on the mat` vs 同义改写 `A cat sits on a mat`,人觉得是好译文,BLEU 因词面不同而**狠狠扣分**。同义词、释义、语序合法变体一律被惩罚——这对开放式生成(摘要、对话、创意写作)是致命的。
2. **对单参考过于苛刻**。一个源句有多种正确译法,但测试集通常只给 1 条参考,候选稍微换个合法说法就掉分。多参考能缓解但贵。
3. **句子级不可靠**。如 6.3,短句高阶 n-gram 命中为 0 导致 BLEU=0,句子级 BLEU 噪声极大,**只适合 corpus 级**。需要平滑(Chen & Cherry 2014)才勉强能做句子级。
4. **不可解释、对小差异不敏感**。BLEU 涨 0.5 分是真进步还是噪声?分数本身不告诉你错在哪。
5. **可被针对性攻击**。知道 BLEU 机制就能构造 n-gram 重叠高但实际很烂的译文。

**埋线**:正因为 BLEU 只懂表面、不懂语义,催生了三条改进路线,第 13 章详讲——

- **字符级重叠**:**chrF / chrF++**(Popović 2015),用字符 n-gram 的 F 值,对形态丰富语言(德语、芬兰语)和拼写小差异更鲁棒,且不依赖 tokenization。
- **加入同义/词序/召回**:**METEOR**(Banerjee & Lavie 2005),做词干化 + 同义词(WordNet)对齐 + 显式召回 + 语序惩罚,与人工相关性比 BLEU 高,但需语言资源、慢。
- **基于神经表示**:**BERTScore**(Zhang et al. 2020,用上下文嵌入做 token 软匹配,解决同义改写)、**COMET**(Rei et al. 2020,在人工评分数据上训练的回归模型,翻译评测当前 SOTA 级,与人工相关性显著高于 BLEU)。这些把"懂语义"真正带进了评测——细节、它们各自的偏差与陷阱(尤其 LLM-as-judge 的位置/长度/自我偏好)全在第 13 章。

## 7. 设计权衡与常见坑

- **信息瓶颈 vs 注意力开销**。无注意力 seq2seq 解码 O(hid²)/步、内存省;注意力 O(n·hid)/步、要存全部编码器输出。短序列差别不大,长序列注意力开销显著但质量必需。**坑**:别在已经用注意力/Transformer 时还纠结"压成一个向量更省"——质量差距是数量级的,这个权衡在翻译上早已定论。
- **teacher forcing 的曝光偏差**。训练全程喂真前缀,推理喂自己的输出,分布不一致,错误会沿序列累积。缓解:scheduled sampling(训练时以一定概率喂模型自己的输出)、序列级训练(直接优化 BLEU 的强化学习,如 MIXER/最小风险训练 MRT)。**坑**:scheduled sampling 有理论争议(改变了目标分布),不是免费午餐。
- **beam 宽不是越大越好**。NMT 里束宽 k 增大,BLEU 常先升后**降**(beam search curse):大 beam 偏好高概率的短/重复序列。实践中 k=4~8 是甜区,必须配长度归一化。**坑**:新手以为 k=50 更准,结果译文变短变烂。
- **BLEU 不能跨配置比较**。不同 tokenization、大小写、是否分词的 BLEU 不可比。**坑**:看论文 BLEU 41 就以为比你 BLEU 39 强——先确认是不是 sacreBLEU 同签名,否则毫无意义。
- **用 BLEU 选模型 vs 用 BLEU 报告**。BLEU 做训练中早停/选 checkpoint 大体可用(同一套 tokenization 内部比较);但**不要只靠 BLEU 下"模型 A 比 B 好"的产品结论**,务必配人工评测或 COMET——BLEU 高的译文可能流畅地漏译(§4.2 的"流畅的错误")。
- **低资源场景**。NMT 吃数据,几十万句对以下时,SMT 或预训练多语言模型(第 8 章子词 + 跨语言迁移)往往更稳。**坑**:别无脑上 NMT。

## 8. 动手练习

1. **手算一个小例子的 BLEU(必做,验算 §6)**
   候选 C = `the cat the cat on the mat`(7 词),参考 R = `the cat is on the mat`(6 词)。
   - 提示:先算 p₁。`the` 在候选出现 3 次,参考最多 2 次 → clip 到 2;`cat` 候选 2 次,参考 1 次 → clip 1;`on/mat` 各命中。逐个 n-gram 列表算 `Count_clip` 之和 / 候选 n-gram 总数。
   - 算 p₂(bigram):候选 bigram 有 `the cat, cat the, the cat, cat on, on the, the mat`,逐个对照参考 bigram `the cat, cat is, is on, on the, the mat` 截断。
   - c=7, r=6 → c>r,BP=1。把 p₁p₂(题目只要算到 bigram,wₙ=1/2)代入几何平均。**自检**:p₁ 你应得到 (2+1+1+1+...)/7,亲手把每个词列全。
2. **推导并验算简短惩罚(必做,巩固 §6.4)**
   - 证明 BP=exp(1−r/c) 在 c→r⁻ 时连续地趋于 1(求 c=r 处左极限),在 c→0⁺ 时趋于 0;并求 dBP/dc,说明它对 c 单调递增(译文越长越接近参考、惩罚越轻)。
   - 验算:r=20 固定,分别取 c=20/16/10/5,算 BP,画出"译文越短惩罚越狠"的曲线,确认 c=10(砍半)时 BP≈0.368=1/e。
3. **讨论题:BLEU 为何不足以评测生成质量(对接 §6.6/第 13 章)**
   - 提示:构造一对例子——(a) 一个 n-gram 重叠高但事实错误/漏译的译文(说明 BLEU 高 ≠ 忠实),(b) 一个高质量同义改写但 BLEU 低的译文(说明 BLEU 低 ≠ 差)。
   - 进一步:对开放式任务(对话、摘要),"唯一参考"假设为什么崩?BERTScore/COMET 各自解决了上面哪个具体缺陷、又引入了什么新偏差(神经指标也会偏向训练分布、可被攻击)?
4. **(选做,动手)注意力对齐可视化**
   - 提示:用 §3.3 的 `AttnDecoder`,在一个玩具英→法平行语料(或直接用 HF 的小翻译模型)上跑推理,收集每步 `alpha` 堆成 [m,n] 矩阵,`plt.imshow` 画热力图,纵轴目标词、横轴源词。观察语序翻转(如形容词后置)时是否出现反对角线。**思考**:你看到的高权重格子,能当成"模型因为这个源词才这么译"的因果证据吗?(回顾 §3.2 的 caveat)

## 9. 源码 / 论文导读

- **必读论文**:
  - Sutskever, Vinyals & Le 2014,"Sequence to Sequence Learning with Neural Networks"——读第 2–3 节(编码器-解码器、源句反转 trick),体会信息瓶颈。
  - Bahdanau, Cho & Bengio 2015(ICLR),"Neural Machine Translation by Jointly Learning to Align and Translate"——读第 3 节(对齐模型公式)和附录的注意力对齐可视化图,这是注意力的原点。
  - Luong, Pham & Manning 2015,"Effective Approaches to Attention-based NMT"——读 global vs local attention、乘性打分,对比 Bahdanau 加性打分。
  - Papineni et al. 2002,"BLEU: a Method for Automatic Evaluation of MT"——读修正精确率与 BP 推导(§2–§3),和本章 §6 对照验算。
  - Koehn, Och & Marcu 2003,"Statistical Phrase-Based Translation"——理解 SMT 短语表/对数线性模型(§4.1)。
- **进阶**:Tu et al. 2016 覆盖机制(§5.2);Post 2018 "A Call for Clarity in Reporting BLEU"(为什么必须用 sacreBLEU);Vaswani et al. 2017 "Attention Is All You Need"(承接到算法课 03)。
- **开源实现**:
  - **HF transformers**:`MarianMTModel` / `T5ForConditionalGeneration` 是 encoder-decoder seq2seq 的标准实现;`generate()` 里 `num_beams`、`length_penalty`、`no_repeat_ngram_size`、`early_stopping` 就是 §5 讲的解码参数。读 `modeling_t5.py` 的 `T5Stack`(encoder/decoder 复用)和 cross-attention。
  - **sacreBLEU**(`pip install sacrebleu`):看 `sacrebleu/metrics/bleu.py` 的 `BLEU` 类如何实现 clipping、effective ref length、BP——和本章 §6.5 代码对照。`chrf.py`、`COMET`(`unbabel-comet`)对接第 13 章。
  - **fairseq** / **OpenNMT**:更贴近研究的 seq2seq 训练框架,fairseq 的 `sequence_generator.py` 是 beam search + 长度归一化的工业级参考实现(配合算法课 13 读)。
- **数据/基准**:WMT(Conference on Machine Translation)年度评测集是 NMT 的标准 benchmark;IWSLT(口语翻译)、FLORES-200(低资源/多语言)。具体某年某语向的 SOTA BLEU/COMET 分数请查当年榜单,本章不引具体数值(避免过期)。

## 10. 小结与承上启下

这一章把 seq2seq 从任务角度讲透了:

- **统一框架**:翻译、摘要、对话都是"变长输入→变长输出"的条件语言模型 `P(y|x)=∏ P(yₜ|y<ₜ,x)`,自回归分解 + teacher forcing 训练,这是后面 T5"text-to-text"的思想根。
- **信息瓶颈→注意力**:把整句压成一个向量在长句上崩溃;注意力让解码每步对源序列**软对齐**、算出定制上下文 cₜ,既破瓶颈又给出**可视化词对齐**(但权重 ≠ 因果)。机制数学在算法课 02/03。
- **SMT→NMT 范式转变**:从"多组件流水线 + 显式对齐 + 短语表 + 对数线性"到"单个端到端网络 + 连续表示 + 注意力对齐 + 最大似然",赢在联合优化、连续泛化、长程流畅;代价是吃数据、黑箱、会"流畅地胡说"。
- **解码与坑**:beam search(细节见算法课 13)及翻译特有的覆盖/重复问题与缓解(coverage 机制、禁重复 n-gram)。
- **BLEU 讲透**:截断修正 n-gram 精确率(堵重复刷分)× 几何平均(对短板敏感、一项为 0 即归零)× 简短惩罚 `BP=min(1,exp(1−r/c))`(堵过短刷分,推导了 exp 形式的合理性),以及它"只懂表面不懂语义"的根本局限。

**承上启下**:第 6/7 章会看到 seq2seq 架构被预训练接管——BERT(判别式,理解任务)与 T5/BART(生成式 seq2seq 预训练)把本章的编码器-解码器放大到海量无标注数据上预训练;第 8 章讲子词切分如何让多语言翻译共享一个词表、实现跨语言迁移(终结 SMT 时代的数据稀疏);而本章 §6.6 留下的"BLEU 不懂语义"这条线,会在**第 13 章(生成评测与忠实性)**被彻底展开——BERTScore、COMET、LLM-as-judge 如何把"懂语义"带进评测,又各自带来什么新偏差。注意力诞生在翻译这块土壤,它长成的 Transformer 则重写了整个 NLP。
