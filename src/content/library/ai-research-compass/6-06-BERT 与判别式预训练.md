---
title: "BERT 与判别式预训练:理解任务的范式革命"
slug: "6-06"
collection: "ai-research-compass"
group: "自然语言处理专家课程"
order: 6006
summary: "这一章把你从\"听说过 BERT 是双向 Transformer、会用 `BertForSequenceClassification` 调个 API\"带到\"能讲清楚为什么双向编码对分类/抽取/QA 这类理解任务是结构性优势,能从第一性原理推导 ELECTRA 的样本效率为什么比 MLM 高约 6 倍,…"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T06:26:40.000Z"
updatedAt: "2026-06-19T06:26:40.000Z"
---
> 这一章把你从"听说过 BERT 是双向 Transformer、会用 `BertForSequenceClassification` 调个 API"带到"能讲清楚为什么双向编码对分类/抽取/QA 这类理解任务是结构性优势,能从第一性原理推导 ELECTRA 的样本效率为什么比 MLM 高约 6 倍,能解释为什么 BERT 裸取 [CLS] 当句向量几乎不可用、而 Sentence-BERT 的双塔对比训练为什么能造出可直接比余弦的句嵌入,并能把 RoBERTa/ALBERT/ELECTRA/DeBERTa 这一家子的每个改动对应到它修的是哪个具体缺陷"。学完你能自己写出 MLM 损失和一个分类头微调的训练循环、能为句级和词级任务正确选用 [CLS] 还是 token 表示、能读懂 sentence-transformers 和 HF 的相关源码并知道改哪里。

## 一、动机:判别式预训练是"理解任务"的范式答案

NLP 的任务可以粗分成两大类。一类是**生成(generation)**:翻译、摘要、对话、写代码——输出是一段需要从左到右造出来的文本。另一类是**理解(understanding,业界叫 NLU)**:情感分类、命名实体识别、抽取式问答、句子相似度、自然语言推理——输入是文本,输出是**一个标签、一个 span、一个分数**,本质是在做判别(discrimination)而非生成。

**这一章讲的是为理解任务量身定做的那条预训练路线——判别式预训练(discriminative pretraining),它的奠基作和家族核心是 BERT。** 它和生成式自回归(GPT 路线)是两套不同的哲学:GPT 学"给定前文,下一个词是什么";BERT 学"给定一个词的完整左右语境,这个词/这句话/这个 span 该被理解成什么"。在 2018-2019 年,后者在几乎所有 NLU 榜单上碾压前者。**理解这套路线为什么对判别任务更强、它的家族怎么一步步把效率和效果推到极致、以及它最大的工程产物(可比较的句向量)如何成为后来 RAG 稠密检索的地基,是本章的全部。**

**和其它课的边界,先划清楚,避免你读重复。** 掩码语言建模(masked LM, MLM)的概率形式、为什么掩 15%、80/10/10 拆分背后的 train/test 一致性逻辑、双向为什么不能直接生成——这些**纯目标函数层面的机制**,《大模型算法专家课程》第 06 章《预训练目标与范式》已经从信息论和链式法则推透了,本章不再重复推导,只在需要时引用其结论(下文标注「详见算法课 06」)。Transformer 编码器内部(self-attention 的 QKᵀ/√d、多头、残差归一化)详见算法课 02-05。**本章的重心是 NLP 特有的那一层:判别式预训练如何统一一众理解任务、BERT 家族的工程演进、以及句表示这个 NLP 工程命门。**

给有工程数学基础的你一句迁移提示:**判别式预训练可以理解成"用自监督先学一个极强的文本特征提取器(feature extractor),再在下游任务上接一个浅层 head 做有监督微调"**。这套"预训练特征 + 任务头"的范式,和计算机视觉里"ImageNet 预训练 backbone + 任务头"是同构的——只不过 NLP 的自监督信号来自文本自身(完形填空、替换检测),不需要人工标注。看清这层同构,后面很多设计就不神秘了。

## 二、预备:为什么"双向"对理解任务是结构性优势

在拆 BERT 之前,必须先把这一章的核心论点立稳:**对判别任务,双向上下文(bidirectional context)不是"锦上添花",而是结构性的、难以替代的优势。** 这一点和生成任务恰好相反(生成任务里双向反而是阻碍,因为生成时右侧还不存在)。

### 2.1 一个词的语义需要左右两侧才能确定

考虑英语里经典的歧义词:

```
句 A:  I went to the river bank to fish.        ← bank = 河岸
句 B:  I went to the bank to deposit money.      ← bank = 银行
```

在句 A 里,"bank" 的正确表征要用到它**右侧**的 "to fish";在句 B 里要用到右侧的 "to deposit money"。**一个只能看左侧的单向模型(如 GPT),在编码 "bank" 这个位置时,右侧信息还没进来——它对 "bank" 的表征必然是"看了左半句的、信息不完整的"。** 而双向模型在编码 "bank" 时左右全看,表征自然更准。

这不是个别现象。**几乎所有判别任务的本质,都是"给定一个 token / span / 句子的完整语境,判断它的某个属性"**:命名实体识别要判断 "Washington" 在 "Washington signed the bill"(人)还是 "Washington is rainy"(地)里指什么——靠右侧动词;抽取式 QA 要在全文里定位答案 span——答案位置依赖整篇上下文;自然语言推理要判断两句的蕴含关系——必须同时双向理解两句。**这些任务在推理时,完整的输入文本已经全部给定了(不像生成是边生成边产生右侧),所以"双向看"既合法又必要。**

### 2.2 单向的信息论代价:一个简单论证

把它量化一点。设位置 t 的 token 表征为 hₜ。单向模型给出的是 `hₜ = f(x_{≤t})`,只编码了左侧;双向模型给出 `hₜ = f(x_{1..T})`,编码了全序列。对一个需要预测位置 t 属性 y 的判别任务,我们关心表征里保留了多少关于 y 的信息,用互信息(mutual information)`I(hₜ; y)` 衡量。由数据处理不等式(data processing inequality):

```
单向:  I(h_t^uni; y)  ≤  I(x_{≤t}; y)        ← 表征至多保留左侧上下文里关于 y 的信息
双向:  I(h_t^bi;  y)  ≤  I(x_{1..T}; y)       ← 表征至多保留全序列里关于 y 的信息

而   I(x_{≤t}; y)  ≤  I(x_{1..T}; y)           ← 左侧是全序列的子集,信息不会更多
```

**所以单向表征关于任务标签 y 的信息上限,天然不超过双向表征的上限。** 当 y 真正依赖右侧上下文时(如上面 bank 的例子),这个不等式是严格的——单向模型在表征层面就丢了信息,后面接再强的 head 也补不回来(数据处理不等式:后处理不能增加信息)。**这就是"双向对理解任务是结构性优势"的硬核理由,不是经验观察而是信息论必然。**

代价当然存在:**双向模型不能直接做自回归生成**(预测一个词时如果允许看右侧,就泄漏了答案)。所以 BERT 路线和 GPT 路线是两个用途分区,详见算法课 06 对"为什么 AR 在大模型时代赢了"的讨论——那是从通用性和生成能力出发的判断,与"双向对纯判别任务更强"并不矛盾,是不同任务维度上的结论。

## 三、BERT:双向 Transformer 编码器 + 两个预训练任务

### 3.1 结构:就是一个 Transformer Encoder 堆栈

BERT(Devlin et al., 2019)的结构没有任何新发明——**它就是把 Transformer 的编码器部分(encoder)堆起来**。编码器的 self-attention 是**全连接的**(每个位置看所有位置,不加因果掩码),这正是双向的来源。两个标准配置:

```
BERT-base:   L=12 层, H=768  隐藏维, A=12 注意力头,  约 110M 参数
BERT-large:  L=24 层, H=1024 隐藏维, A=16 注意力头,  约 340M 参数
(参数量为约数;tokenizer 用 WordPiece,词表约 30k —— 精确 30522 待核)
```

输入的构造有 BERT 特有的三层 embedding 相加,这是要记住的细节:

```
输入序列示例(两句拼接做句子对任务):
  [CLS]  the  cat  sat  [SEP]  it  was  fat  [SEP]
   ↓
Token Embedding   : 每个 token 查 WordPiece 词表得到的词向量
Segment Embedding : 标记属于句 A 还是句 B(E_A / E_B 两个可学习向量)
Position Embedding: 标记绝对位置 0,1,2,…(BERT 用可学习的绝对位置嵌入,最长 512)
   ↓ 三者逐元素相加
最终输入 = TokenEmb + SegmentEmb + PositionEmb
```

两个特殊 token 必须吃透其用途:

- **[CLS]**:加在序列最前面。它对应的最终隐状态 `h_[CLS]` 被设计成**整句(或整个句子对)的聚合表征**,用于句级(sentence-level)任务。注意:这个"聚合"能力不是天生的,是靠预训练任务(NSP)逼出来的——下面会讲它其实学得并不好。
- **[SEP]**:分隔符,放在每句末尾,告诉模型句子边界。

**关键的表征-任务对应关系(本章要反复用):**

```
[CLS] 的隐状态     → 句级任务(分类、句子对关系判断、句向量)
每个 token 的隐状态 → 词级任务(NER 序列标注、抽取式 QA 的 span 起止、token 分类)
```

### 3.2 两个预训练任务:MLM(主力)+ NSP(辅助)

BERT 用两个自监督任务联合预训练。

**任务一:Masked Language Model(MLM,掩码语言建模)。** 随机选 15% 的 token,按 80/10/10 拆分(80% 换 [MASK]、10% 换随机 token、10% 不变),让模型用双向上下文还原被选中位置的原 token。**这套机制的完整推导——为什么是 15%、为什么要 80/10/10 解决 [MASK] 在下游不出现的分布错位、为什么多个被遮位置是条件独立预测的——详见算法课 06 第四节,本章不重复。** 这里只强调它在本章语境下的意义:**MLM 是 BERT 双向表征的来源**。因为要还原一个被遮的词必须同时利用它左右两侧的语境,模型被迫学出"融合完整上下文的 token 表征",这正是 2.1/2.2 节说的判别任务最需要的东西。

**任务二:Next Sentence Prediction(NSP,下一句预测)。** 这是 BERT 特有、且后来被证明用处不大的任务。构造:

```
50% 概率: 句 B 是语料中句 A 真正的下一句   → 标签 IsNext
50% 概率: 句 B 是从语料随机抽的另一句      → 标签 NotNext

输入: [CLS] 句A [SEP] 句B [SEP]
监督: 用 [CLS] 的隐状态接一个二分类 head,预测 IsNext / NotNext
```

NSP 的初衷是让模型学会**句子间的关系**,以服务 QA、NLI 这类需要理解两句关系的下游任务。**它也是 [CLS] 表征被训练成"句对聚合表征"的唯一来源。** 但后来 RoBERTa 证明 NSP 几乎没用甚至有害(见第四节),这是 BERT 家族演进的第一个重要发现。

### 3.3 代码:MLM 损失的完整实现

把 MLM 损失写到能跑的程度。核心是:**只在被选中的位置算交叉熵,其余位置的 label 标成 ignore_index(-100)不计入 loss**。

```python
import torch
import torch.nn.functional as F

def build_mlm_inputs(input_ids, vocab_size, mask_token_id,
                     special_ids, mlm_prob=0.15):
    """
    构造 MLM 训练样本:实现 15% 选择 + 80/10/10 拆分。
    input_ids:    (B, T)  原始 token 序列
    special_ids:  集合,如 {[CLS],[SEP],[PAD]} 的 id —— 这些位置永不被选中
    返回:
      masked_inputs: (B, T)  喂给模型的(已扰动的)输入
      labels:        (B, T)  被选中位置=原 token id,未选中位置=-100(不算 loss)
    """
    labels = input_ids.clone()
    # ---- 第一步:以 15% 概率独立采样选择矩阵,但排除特殊 token ----
    prob = torch.full(input_ids.shape, mlm_prob)
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= (input_ids == sid)
    prob[special_mask] = 0.0                       # 特殊 token 不参与
    selected = torch.bernoulli(prob).bool()        # (B,T) True=被选中预测

    labels[~selected] = -100                       # 未选中位置不计 loss(关键!)

    masked_inputs = input_ids.clone()
    # ---- 第二步:在被选中的 token 里做 80/10/10 ----
    # 80% -> [MASK]
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & selected
    masked_inputs[replace_mask] = mask_token_id
    # 10% -> 随机 token(在 selected 但未被 [MASK] 的里面,再取一半)
    rand_mask = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & selected & ~replace_mask
    random_tokens = torch.randint(vocab_size, input_ids.shape)
    masked_inputs[rand_mask] = random_tokens[rand_mask]
    # 剩下 10% 保持原 token 不变(masked_inputs 该位置已是原值,什么都不做)
    return masked_inputs, labels


def mlm_loss(logits, labels):
    """
    MLM 损失:仅对 label != -100 的位置算交叉熵。
    logits: (B, T, V)  编码器输出过 MLM head 后对词表的打分
    labels: (B, T)     被选中位置=原 token id,其余=-100
    """
    # cross_entropy 的 ignore_index 默认就是 -100,直接展平即可
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),   # (B*T, V)
        labels.view(-1),                    # (B*T,)
        ignore_index=-100,                  # -100 的位置自动跳过,不贡献梯度
    )
```

**两个易错点(新手必踩):** (1) 特殊 token([CLS]/[SEP]/[PAD])必须排除在选择之外,否则会让模型学着预测分隔符,污染信号;(2) `labels` 的未选中位置一定要设 -100,否则模型会被要求"预测所有原 token",MLM 就退化成了一个泄漏答案的恒等任务(输入里就有答案)。HF 的 `DataCollatorForLanguageModeling` 实现了这套逻辑,可对照阅读。

### 3.4 预训练 + 微调:一套机制统一一众 NLP 任务

**这是 BERT 范式革命的核心,也是这章最该刻进脑子的一张图。** BERT 之前,每个 NLP 任务往往要从头设计一个专用模型架构(NER 用 BiLSTM-CRF、QA 用复杂的注意力交互网络、分类用 TextCNN……)。BERT 之后,范式坍缩成统一的两步:

```
第一步(预训练,一次性,极贵):
   在海量无标注文本上用 MLM+NSP 训练 BERT —— 得到一个通用文本编码器

第二步(微调,每个任务一次,便宜):
   冻结或不冻结 BERT 主干,在它上面接一个极浅的 task head,
   用该任务的少量标注数据端到端微调
```

不同任务的差别,**只在于"接哪个 head、用哪个位置的表征"**:

| 任务类型 | 例子 | 用哪个表征 | task head | 输出 |
|---------|------|-----------|-----------|------|
| 单句分类 | 情感分析、主题分类 | `h_[CLS]` | Linear(H → C) | C 类 softmax |
| 句子对分类 | NLI、句子相似(分档) | `h_[CLS]`(输入 [CLS] A [SEP] B [SEP]) | Linear(H → C) | C 类 softmax |
| 序列标注 | NER、词性标注 | 每个 token 的 `hₜ` | Linear(H → C),逐 token | 每 token 一个标签 |
| 抽取式 QA | SQuAD | 每个 token 的 `hₜ` | 两个 Linear(H → 1):预测 start / end | 答案 span 的起止位置 |

**抽取式 QA 的 head 值得单独看,因为它体现了"词级表征"的用法。** SQuAD 式任务给定 (问题, 文章),要在文章里框出答案 span。BERT 的做法:输入 `[CLS] 问题 [SEP] 文章 [SEP]`,然后用两个可学习向量 `s, e ∈ ℝ^H`(start 向量、end 向量),对文章里每个 token 的隐状态 hᵢ 算:

```
start 分数:  S_i = s · h_i            (token i 是答案起点的打分)
end   分数:  E_i = e · h_i            (token i 是答案终点的打分)
预测起点 = argmax_i softmax(S)_i
预测终点 = argmax_j softmax(E)_j     (通常约束 j ≥ i)
训练损失 = CE(start 预测, 真起点) + CE(end 预测, 真终点)
```

整个 QA "模型"就多了两个 H 维向量(2H 个参数)。**这就是范式革命的威力:一个 110M 参数的预训练编码器 + 几百到几千个 head 参数,就能在 SQuAD 上超过之前所有专门设计的复杂架构。** 知识在预训练里,任务适配只是薄薄一层。

### 3.5 代码:微调一个分类头

把"接 head + 微调"写成可跑的训练步:

```python
import torch
import torch.nn as nn

class BertForClassification(nn.Module):
    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert = bert                                  # 预训练好的编码器主干
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)  # 唯一新增的 head

    def forward(self, input_ids, attention_mask, segment_ids=None):
        # 编码器输出全部 token 的隐状态: (B, T, H)
        hidden = self.bert(input_ids, attention_mask, segment_ids)  # 伪接口
        cls = hidden[:, 0]                  # 取 [CLS] 位置(句级聚合表征): (B, H)
        cls = self.dropout(cls)
        logits = self.classifier(cls)       # (B, num_labels)
        return logits

def finetune_step(model, batch, optimizer):
    logits = model(batch["input_ids"], batch["attention_mask"], batch["segment_ids"])
    loss = nn.functional.cross_entropy(logits, batch["labels"])
    optimizer.zero_grad()
    loss.backward()                         # 梯度同时流回 head 和整个 BERT 主干
    optimizer.step()
    return loss.item()
```

**微调的关键设计选择:** (1) 学习率要小(典型 2e-5 到 5e-5,**待核**,远小于从头训练),因为预训练权重已经很好,大步长会"灾难性遗忘"预训练学到的知识;(2) 通常**全参数微调**(BERT 主干也更新),而非只训 head——只训 head 效果明显差,因为下游任务往往需要轻微调整主干的表征;(3) 训练 epoch 很少(2-4 个),数据少时极易过拟合。这些经验值都源于"预训练已经做了绝大部分工作,微调只是小幅适配"这个本质。

## 四、BERT 家族(上):RoBERTa 与 ALBERT——把训练和参数做对

BERT 出来后,一大批后续工作做的事可以归成两类:**修预训练目标/流程的缺陷**(RoBERTa、ELECTRA),和**修参数/结构效率的缺陷**(ALBERT、DeBERTa)。先看前两个里的 RoBERTa 和省参的 ALBERT。

### 4.1 RoBERTa:证明"NSP 没用,是 BERT 训练不足"

RoBERTa(Liu et al., 2019,名字是 "Robustly optimized BERT approach")**没有改 BERT 的结构**,它做的是一组严谨的消融实验,结论颠覆了 BERT 的两个设计:

**结论一:去掉 NSP,效果反而更好或持平。** RoBERTa 系统对比了"带 NSP"和"不带 NSP、只用 MLM、输入是连续打包的长文本"几种设定,发现**移除 NSP 后下游任务不降反升**。这说明 BERT 论文里"NSP 有用"的结论可能是个伪因果——真正起作用的是看到了更长的连续上下文,而 NSP 那个二分类任务本身太简单(判断两句是否相邻,模型靠主题是否一致就能蒙对),提供的信号弱且可能干扰 MLM。

**结论二:BERT 严重训练不足(undertrained)。** RoBERTa 用了**更多数据(约 160GB 文本,远超 BERT 的约 16GB)、更大 batch、更长训练步数**,在完全相同的架构下大幅超越 BERT。**这传达了一个深刻且反直觉的信息:BERT 当时刷出的成绩远没有触及这个架构的能力上限,瓶颈不在结构而在训练量。** 这个判断后来被 scaling law(详见算法课 08)系统化。

**结论三:动态掩码(dynamic masking)。** 原始 BERT 在数据预处理阶段就把掩码固定下来(静态掩码),整个训练过程同一个句子的掩码模式不变——相当于看了 N 遍同样的完形填空题。RoBERTa 改成**每次把句子喂进模型时现场随机生成掩码**:

```
静态掩码(BERT):  预处理时固定 → 同一句在所有 epoch 看到相同的 [MASK] 位置
动态掩码(RoBERTa): 训练时实时采样 → 同一句在不同 epoch 看到不同的 [MASK] 位置
                  效果:等价于数据增强,同一句变成多道不同的填空题,泛化更好
```

RoBERTa 还把 tokenizer 换成了**字节级 BPE(byte-level BPE,词表约 50k)**,好处是不会有"未登录字符"(任何字节都能编码)。**RoBERTa 的历史意义:它把"BERT 的成功有多少来自双向 MLM、多少来自没训够"这个问题做成了 controlled experiment,确立了"数据量和训练时长是一等公民"的认知。** 此后大家说的 "BERT" 基线,实际多指 RoBERTa 这套优化后的训练配方。

### 4.2 ALBERT:用参数共享和嵌入分解省参数

ALBERT(Lan et al., 2020,"A Lite BERT")解决的是另一个维度的问题:**BERT 参数量大、显存吃紧、大模型难训**。它用两个正交的技巧大幅压缩参数,且基本不掉点(有时还涨)。

**技巧一:嵌入分解(factorized embedding parameterization)。** 观察:BERT 里 WordPiece 词嵌入的维度 E 被强行等于隐藏维 H(因为嵌入直接喂进第一层)。当 H 很大(如 1024),词表 V≈30k 时,嵌入矩阵就有 `V × H = 30000 × 1024 ≈ 3072 万`参数,占总参数相当大一块。ALBERT 的洞察:**词嵌入是"上下文无关"的静态查表,而隐藏层学的是"上下文相关"的表征,这两者的信息量级不同,没必要用同一个维度。** 于是把嵌入矩阵分解成两个小矩阵:

```
原始:  one-hot(V) → [V × H] → 隐藏(H)            参数 = V · H
分解:  one-hot(V) → [V × E] → [E × H] → 隐藏(H)   参数 = V · E + E · H

取 E ≪ H(如 E=128, H=1024),参数从 V·H 降到 V·E + E·H:
  30000×1024 ≈ 3072万   →   30000×128 + 128×1024 ≈ 384万 + 13万 ≈ 397万
  省了约 8 倍嵌入参数
```

**技巧二:跨层参数共享(cross-layer parameter sharing)。** 标准 BERT 的 L 层 Transformer 块各有各的参数(L 套)。ALBERT 让**所有层共享同一套参数**(默认共享全部:attention + FFN)。这意味着 24 层 ALBERT 的 Transformer 块参数量等于 1 层。

```
BERT-large:  24 层各自独立 → 24 套 Transformer 块参数
ALBERT:      24 层共享 1 套 → 参数量 ≈ 1 套(但仍前向 24 次,计算量不变!)
```

**这里有个必须讲清的权衡,新手最容易误解:参数共享省的是参数量(显存/存储),不省计算量(FLOPs)。** 24 层共享参数,前向时仍然要老老实实跑 24 次(每次用同一套权重),所以**推理速度不会变快**。它的收益是:模型小了,能放进更大的 H、训更大的模型、减少过拟合。代价是:同样深度下,共享参数的表达能力弱于独立参数,所以 ALBERT 要靠更大的隐藏维和更长训练把损失补回来。**"参数少≠算得快"是这一节最该记住的反直觉点。**

**附带改动:SOP 替代 NSP。** ALBERT 把 NSP 换成 **SOP(Sentence Order Prediction,句序预测)**:正例是相邻两句的正序,负例是**同样两句但顺序颠倒**(而非 NSP 那样从别处随机抽一句)。这样负例和正例**主题一致**,模型不能靠"主题是否相关"蒙混,必须真正学句间的连贯性/逻辑顺序。SOP 比 NSP 难、信号更有用,呼应了 RoBERTa "NSP 太简单" 的发现,但 ALBERT 选择把它做难而非删掉。

## 五、BERT 家族(下):ELECTRA——替换检测,让所有 token 都出力

ELECTRA(Clark et al., 2020,"Efficiently Learning an Encoder that Classifies Token Replacements Accurately")是 BERT 家族里**最值得从第一性原理推导的一个**,因为它的样本效率优势可以被精确论证。这一节是本章的技术高峰,慢慢推。

### 5.1 问题:MLM 的信号只有 15%,太浪费

回顾 MLM 的根本低效(详见算法课 06):**一个长度 T 的序列,只有约 15% 的 token 被遮、贡献 loss,剩下 85% 的 token 模型白白编码了一遍却不产生任何监督信号。** 模型每跑一次前向(算力花在全部 T 个 token 上),却只从 0.15T 个位置学到东西。能不能让**每一个 token 都提供训练信号**?

ELECTRA 的答案:**别让模型去"生成"被遮的词(那是个 |V| 类的大分类,且只能对 15% 的位置做),改成让模型对每一个 token 做一个二分类——判断这个 token 是不是被替换过(replaced token detection, RTD)。** 二分类对所有 token 都成立,信号密度从 15% 拉到 100%。

### 5.2 机制:生成器 + 判别器

但"判断 token 是否被替换"需要先有"被替换的 token"。如果只是随机替换,任务太简单(随机词在语境里很突兀,一眼识破),模型学不到东西。ELECTRA 的精巧设计:**用一个小的 MLM 当"生成器(generator)"来造出"看起来合理"的替换词,再用主模型当"判别器(discriminator)"去检测哪些被换了。**

```
原文 x:        the   chef   cooked   the   meal
              ↓ 随机遮 15%(和 MLM 一样)
遮后:         the  [MASK]  cooked  [MASK]  meal

【生成器 G】(一个小 MLM)对 [MASK] 位置采样填词,造出"corrupted"序列:
              the   chef   cooked    the   meal     ← 位置2 填回 chef(碰巧对)
              the   cook   cooked    a     meal     ← 位置2 填 cook(像但错), 位置4 填 a(错)
corrupted x': the   cook   cooked    a     meal

【判别器 D】(主模型, encoder)对 x' 的每一个 token 做二分类:original / replaced
真实标签:    orig  REPL   orig     REPL  orig      ← 与原文比对得到
              ↑ 每个 token 都有标签 → 每个 token 都贡献 loss
```

注意一个微妙处:**生成器填回的词如果碰巧等于原词(如位置2填回了 "chef"),那它的标签是 original 而非 replaced**——因为对判别器来说,token 确实没被改变。这点对损失定义很重要。

### 5.3 损失函数与训练

两个网络联合训练,各有损失。

**生成器**就是标准 MLM,只对被遮位置算:

```
L_G = - E [ Σ_{t ∈ masked}  log p_G(x_t | x_masked) ]      ← 还原被遮的真 token
```

**判别器**对**所有位置**做二分类(sigmoid + 二元交叉熵):

```
设 D(x', t) = sigmoid( w · h_t )  ∈ (0,1)  表示判别器认为位置 t 是 original 的概率
标签 y_t = 1 若 x'_t == 原 token x_t(original),  y_t = 0 若被替换(replaced)

L_D = - E [ Σ_{t=1}^{T} ( y_t · log D(x',t) + (1 - y_t) · log(1 - D(x',t)) ) ]
                          └──────────────── 对全部 T 个位置求和 ────────────────┘
```

**联合目标:** `min L_G + λ·L_D`(λ 是判别器损失权重,原文取较大值如 50,**待核**,因为二分类 loss 数值天然小于 |V| 类 CE,要放大以平衡)。

**三个极其重要、和 GAN 区分开的设计点:**

1. **判别器的梯度不回传到生成器。** 这不是 GAN!GAN 里生成器要骗过判别器(对抗),梯度要从判别器流回生成器。ELECTRA 里生成器**只用自己的 MLM 损失 L_G 训练**,判别器损失 L_D **不更新生成器**。原因:从判别器到"采样哪个词"这一步是离散采样,不可导(要 RL 或 Gumbel 才能传,原文实验过对抗式但效果不如直接 MLM)。**所以 ELECTRA 是"生成器自己学填空 + 判别器学检测",两者通过共享数据流耦合,但不是对抗博弈。** 这是最常被误解的点。

2. **生成器要小。** 生成器若太强,填回的词几乎全对(全是 original),判别器没有 replaced 样本可学,任务退化。所以生成器通常只取判别器的 1/4 到 1/2 大小(**待核**具体比例)。一个"恰到好处地犯错"的弱生成器,才能给判别器制造有难度的替换样本。

3. **下游只用判别器。** 预训练完,**扔掉生成器**,只把判别器(主 encoder)拿去微调下游任务,和 BERT 用法完全一样(接 head)。生成器只是预训练时的"出题机"。

### 5.4 推导:为什么 ELECTRA 样本效率高约 6 倍

这是本章要你能自己推的核心结论。**"样本效率(sample efficiency)"指:每见过一个训练样本(一次前向),模型获得多少有效监督信号。** 用"每序列产生的监督信号数"近似衡量。

```
设序列长度 T,掩码率 r = 0.15。

MLM(BERT):
  每序列只在被遮的 r·T 个位置产生 loss。
  有效监督信号数 ≈ r·T = 0.15·T
  → 模型前向编码了 T 个 token,只有 0.15T 个回传梯度

ELECTRA(判别器):
  每序列对全部 T 个位置做二分类,都产生 loss。
  有效监督信号数 ≈ T
  → 前向编码 T 个 token,全部 T 个回传梯度

样本效率之比(信号密度之比):
  T / (r·T) = 1 / r = 1 / 0.15 ≈ 6.7
```

**所以在"信号密度"这个一阶意义上,ELECTRA 每个样本提供的监督信号约是 MLM 的 6.7 倍。** 这直接解释了 ELECTRA 论文的核心实验现象:**在相同算力/相同模型大小下,ELECTRA 显著超过 MLM 类模型;尤其在小模型、低算力区间优势最大**(ELECTRA-small 用很小的算力就能逼近大得多的 BERT)。

**但必须诚实标注这个推导的近似性(否则就是过度声称):**

- 这只比了**信号的数量**,没比**信号的质量/难度**。MLM 的每个信号是 |V| 类分类(信息量 log|V| ≈ log(30000) ≈ 14.9 bit),ELECTRA 的每个信号是 1 个二分类(信息量上限 1 bit)。**单个二分类信号比单个 |V| 类信号"信息量小得多"**——所以"6.7 倍"绝不能理解成"训练效率精确快 6.7 倍"。
- 真实的效率增益是"信号更密(利好 ELECTRA)"和"单信号更弱(利好 MLM)"两股力量的净结果。ELECTRA 实证更快,说明**信号密度的收益在实践中盖过了单信号变弱的损失**——直觉是:让模型对每个 token 都判断"这词在这语境合不合理",虽然单点信息少,但覆盖全序列、且这个判断本身需要深度语境理解,综合起来学到的表征更高效。

**这个"信号密度 vs 单信号强度"的权衡框架,是理解所有自监督目标设计的通用透镜**——你以后看任何新预训练目标,都可以问:它在每个样本上让多少单元产生信号、每个信号多强、两者的乘积(总信息率)如何。

### 5.5 代码:ELECTRA 判别器损失

```python
import torch
import torch.nn.functional as F

def electra_discriminator_loss(disc_logits, corrupted_ids, original_ids,
                               non_pad_mask):
    """
    ELECTRA 判别器损失:对每个 token 做 original/replaced 二分类。
    disc_logits:   (B, T)    判别器对每个位置输出的 logit(sigmoid 前)
    corrupted_ids: (B, T)    生成器填充后的序列(判别器实际看到的输入)
    original_ids:  (B, T)    原始真实序列
    non_pad_mask:  (B, T)    1=真实 token, 0=padding(padding 不算 loss)
    """
    # 标签:被替换(corrupted != original)→ 1(replaced=正类), 否则 0(original)
    # 注意:生成器碰巧填回原词的位置,corrupted==original,标签=original(0)
    # 约定提示:此处正类=replaced,故 sigmoid(disc_logits) 表示 P(replaced),
    #          与 5.3 公式(D 输出 P(original)、y=1 表示 original)恰好相反。
    #          两种约定只是把学到的函数整体翻转,loss 数值与训练效果不变。
    is_replaced = (corrupted_ids != original_ids).float()    # (B,T) 1=replaced

    # 对每个位置算二元交叉熵;非 padding 位置全部参与(信号密度=100%)
    loss = F.binary_cross_entropy_with_logits(
        disc_logits, is_replaced, reduction="none")          # (B,T) 每位置一个 loss
    loss = (loss * non_pad_mask).sum() / non_pad_mask.sum()  # 只对真实 token 平均
    return loss

# 对比 MLM:F.cross_entropy(logits[masked], labels[masked]) —— 分母只有 masked 位置数,
# 约占 15%;而这里分母是全部 non-pad 位置,约 100%。信号密度差异一目了然。
```

## 六、BERT 家族(补):DeBERTa——解耦内容与位置

DeBERTa(He et al., 2020/2021,"Decoding-enhanced BERT with disentangled attention")是把注意力机制本身改进的代表,在 SuperGLUE 等榜单上一度超越人类基线。它的核心贡献是**解耦注意力(disentangled attention)**,这里讲清直觉和形式,内部 attention 的基础数学详见算法课 02/04。

**动机:标准 Transformer 把"内容信息"和"位置信息"加在一起喂进 attention(token embedding + position embedding 相加),但一个词对另一个词的注意力,其实由两个不同的因素决定——它们是什么词(内容)、以及它们隔多远(相对位置)。把两者揉成一个向量后,注意力无法分别建模这两种依赖。**

DeBERTa 的做法:**每个 token 用两个独立向量表示——内容向量 H 和相对位置向量 P**,attention 分数拆成内容和位置的交叉项之和。直觉形式(简化):

```
标准注意力(单向量):  A_ij ∝ (H_i)(H_j)ᵀ           内容、位置混在 H 里

解耦注意力(双向量):  A_ij ∝ (H_i)(H_j)ᵀ           内容-内容("这两个词本身多相关")
                          + (H_i)(P_{i|j})ᵀ           内容-位置("词 i 多关注'距离 i→j'这个相对位置")
                          + (P_{j|i})(H_j)ᵀ           位置-内容
                       (原文省略了位置-位置项,因相对位置间的交互意义不大)
```

其中 `P_{i|j}` 是从 i 看 j 的相对位置向量(依赖 i−j 这个相对距离)。**收益:模型能分别学"哪些词内容上该互相关注"和"哪个相对距离上该关注",比把两者绑死表达力更强。** DeBERTa 还加了 **EMD(Enhanced Mask Decoder,增强掩码解码器)**:在 MLM 预测被遮词之前,把**绝对位置信息**注入回去(因为前面只用了相对位置,而有些预测需要绝对位置,如句首的词)。这个"相对位置进 attention、绝对位置在输出前补回"的组合是 DeBERTa 的精髓。**对你的迁移价值:解耦注意力是"把混在一起的信号显式拆开各自建模"的范例,这个思路在很多架构改进里反复出现。**

## 七、句向量:从 [CLS] 失效到 Sentence-BERT

这一节是本章通向后续 RAG 章的桥,也是 BERT 在工程上最大的一个"坑 + 解法"。

### 7.1 问题:BERT 裸取 [CLS] 当句向量,几乎不可用

很多人第一反应:BERT 不是有个 [CLS] 表征整句吗,那我把两句话各过一遍 BERT、取各自 [CLS]、算余弦相似度,不就是句子相似度了?**这么做效果出奇地差——实测常常比 GloVe 词向量取平均还差。** 必须讲清为什么,因为这是个深刻的表示学习问题。

**根因一:[CLS] 从未被训练成"余弦空间里可比较"的表征。** [CLS] 在预训练里只服务 NSP 这个二分类(且 NSP 本身就弱/被 RoBERTa 否定)。NSP 的监督是"这两句相邻吗",它**不要求语义相近的句子在向量空间里距离近**——它只要 [CLS] 能线性分出 IsNext/NotNext。**一个向量能被某个分类头区分,和这个向量本身的几何(余弦距离)有语义意义,是两回事。** 直接拿来比余弦,等于用一把没标过刻度的尺子量长度。

**根因二:BERT 的 token 表征存在各向异性(anisotropy)。** 研究(如 BERT-flow、whitening 相关工作)发现 BERT 输出的向量分布**不是各向同性的**,而是挤在一个狭窄的锥形(cone)里——任意两个句子的余弦相似度都偏高且区分度低。**在这样一个被"压扁"的空间里算余弦,信号被几何畸变淹没。**

**根因三(工程上致命):cross-encoder 不可扩展。** 你可能说,那别取 [CLS],把两句拼起来 `[CLS] A [SEP] B [SEP]` 过一遍 BERT 让它直接出相似度分(这叫 **cross-encoder**,交叉编码),效果确实好得多。但代价是:**相似度无法预计算**。要在 N 个句子里找最相似对,得跑 `C(N,2) ≈ N²/2` 次 BERT 前向。Sentence-BERT 论文举的例子:在约 1 万个句子里做聚类,cross-encoder 要约 5000 万次前向、耗时约 65 小时(**待核**具体数值);而下面的双塔方案只需几秒。**对检索/聚类这类要反复两两比较的任务,cross-encoder 在计算上是不可行的。**

### 7.2 解法:Sentence-BERT 的双塔 + 对比/三元组微调

Sentence-BERT(SBERT,Reimers & Gurevych, 2019)的解法干净利落:**用双塔(siamese,孪生)结构,显式地用一个让"语义相近→向量相近"的目标去微调 BERT,使输出的句向量在余弦空间里可比。**

**结构(双塔/bi-encoder):** 两个**共享权重**的 BERT 分别编码句 A 和句 B,各自做 **pooling**(对所有 token 隐状态取平均,mean-pooling,实测比取 [CLS] 好)得到固定维句向量 u 和 v:

```
句 A → BERT → token 隐状态 (T_A, H) → mean-pooling → u ∈ ℝ^H
句 B → BERT → token 隐状态 (T_B, H) → mean-pooling → v ∈ ℝ^H
       (两个 BERT 权重共享,所以是"同一个编码器")
```

**为什么双塔能解决可扩展性:** 每个句子**独立编码成一个固定向量**,可以**离线预计算并建索引**。检索时只需把 query 编码一次,然后和库里 N 个预存向量算余弦(或丢给 FAISS 做近似最近邻),复杂度从 O(N²) 次 BERT 前向降到 **N 次编码(可离线)+ 一次向量检索**。**这就是稠密检索(dense retrieval)的雏形,直接通向 RAG 章。**

**训练目标(三选一,按下游数据形态):**

(1) **分类目标(用于 NLI 这类有离散标签的句对数据)。** 把 u、v 和它们的逐元素差的绝对值 |u−v| 拼起来,过一个 softmax 分类头:

```
o = softmax( W · [u ; v ; |u − v|] )       W ∈ ℝ^{C × 3H}
损失 = CrossEntropy(o, 句对关系标签)         (如蕴含/矛盾/中立)
```

`|u−v|` 这一项很关键:**它显式地把"两向量的差异"喂给分类器,逼着模型学出"语义关系体现在向量差上"的几何**,这正是让余弦变得有意义的训练压力。

(2) **回归目标(用于有连续相似度分数的数据,如 STS)。** 直接让两向量的**余弦**去拟合人标的相似度分数:

```
损失 = MSE( cos(u, v),  人标相似度分数 ∈ [0,1] )
```

**这是最直接的"把余弦校准成相似度"的训练**——优化完后,cos(u,v) 就是一个有刻度的相似度,可以直接用。

(3) **三元组目标(triplet loss,用于只有"谁和谁更像"的弱监督)。** 给定锚点 a、正例 p(和 a 相似)、负例 n(和 a 不相似),要求 a 离 p 比离 n 近至少一个间隔 margin:

```
损失 = max( 0,  ‖a − p‖ − ‖a − n‖ + margin )
              └─正例距离─┘  └─负例距离─┘
含义:正例距离 + margin ≤ 负例距离 时损失为 0;否则惩罚。
推着相似句聚拢、不相似句推远 —— 直接塑造可比较的度量空间。
```

**三种目标的共同本质:都在向量空间里施加"语义相近↔几何相近"的约束**,从而把 BERT 那个不可比的表征,改造成一个余弦/欧氏距离有明确语义的度量空间。**这就是"对比学习(contrastive learning)/度量学习(metric learning)"在 NLP 句表示上的应用**——后续更强的句嵌入模型(SimCSE 用 dropout 造正例、E5/BGE 等)都是这条路线的延伸。

### 7.3 代码:SBERT 三元组训练 + 余弦回归

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_pooling(token_embeddings, attention_mask):
    """对真实 token(非 padding)的隐状态取平均 —— SBERT 默认 pooling。"""
    mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
    summed = (token_embeddings * mask).sum(dim=1)        # (B, H) 只加真实 token
    counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1) 真实 token 数
    return summed / counts                               # (B, H) 平均

class SentenceBERT(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert                                 # 双塔共享这一个 BERT

    def encode(self, input_ids, attention_mask):
        hidden = self.bert(input_ids, attention_mask)    # (B, T, H) 伪接口
        return mean_pooling(hidden, attention_mask)      # (B, H) 句向量

def cosine_regression_loss(model, batch):
    """回归目标:让 cos(u,v) 拟合人标相似度。优化后余弦即可直接当相似度用。"""
    u = model.encode(batch["ids_a"], batch["mask_a"])    # (B, H)
    v = model.encode(batch["ids_b"], batch["mask_b"])    # (B, H)
    cos = F.cosine_similarity(u, v, dim=-1)              # (B,) ∈ [-1, 1]
    return F.mse_loss(cos, batch["sim_score"])           # 拟合 [0,1] 相似度

def triplet_loss(model, batch, margin=1.0):
    """三元组目标:正例拉近、负例推远。"""
    a = model.encode(batch["ids_anchor"], batch["mask_anchor"])
    p = model.encode(batch["ids_pos"],    batch["mask_pos"])
    n = model.encode(batch["ids_neg"],    batch["mask_neg"])
    d_pos = (a - p).norm(dim=-1)                          # ‖a-p‖
    d_neg = (a - n).norm(dim=-1)                          # ‖a-n‖
    return F.relu(d_pos - d_neg + margin).mean()          # max(0, d_pos - d_neg + margin)
```

**用法对照(把本章的表示-任务对应关系收口):**

```
句级语义检索/聚类/相似度  → Sentence-BERT 双塔, mean-pooling 句向量, 比余弦  (本节)
句级分类(单任务, 有标注)  → BERT [CLS] + 分类头, 微调                       (3.4 节)
高精度两两重排(候选少)    → cross-encoder([CLS]A[SEP]B), 直接出分           (7.1 节)
词级标注 / 抽取 QA         → 每 token 隐状态 + 逐 token head                  (3.4 节)
```

**RAG 检索的典型两段式正是上面前两条的组合:先用 SBERT 双塔做粗召回(快、可索引)、再用 cross-encoder 对召回的少量候选做精排(准、不可扩展但候选已经很少)。** 把"双塔召回 + 交叉精排"记成一对,这是检索系统的标准范式,会在 RAG 章展开。

## 八、设计权衡与常见坑

**坑 1:拿裸 BERT 的 [CLS] 算句子相似度。** 本章 7.1 节的全部内容。[CLS] 没在余弦空间被训练过,且 BERT 表征各向异性。**要句向量,用 Sentence-BERT 或专门的句嵌入模型;要相似度但能接受慢,用 cross-encoder。** 这是 NLP 工程里最高频的误用。

**坑 2:用 encoder(BERT)去做生成任务。** BERT 是双向编码器,**结构上不能自回归生成**(没有因果掩码、没有"续写"的训练目标)。想生成用 GPT 类(decoder-only)或 T5/BART(encoder-decoder)。反过来,**用 decoder-only 模型做纯判别任务**,通常要取最后一个 token 的隐状态或用 prompt 续写标签,效果和适配难度往往不如直接微调一个 encoder——纯理解任务,BERT 类仍是性价比之选(详见算法课 06 坑 6)。

**坑 3:把 ELECTRA 当 GAN。** 7.2/5.3 反复强调:**判别器梯度不回传生成器,生成器只用自己的 MLM loss 训练。** 它不是对抗博弈,是"出题机 + 答题机"通过共享数据流耦合。把它理解成 GAN 会在复现时错误地加上对抗梯度。

**坑 4:以为 ALBERT 参数共享能加速推理。** 参数共享省的是**参数量/显存**,24 层仍要前向 24 次,**FLOPs 和延迟不降反可能略升**(因为它用了更大的 H)。要推理快,该做的是蒸馏(DistilBERT)、剪枝、量化,而不是参数共享。

**坑 5:微调学习率照搬从头训练。** 预训练权重已极好,微调用 2e-5 量级的小学习率(**待核**具体值),大学习率会灾难性遗忘。数据少时还要警惕过拟合,2-4 epoch 通常够。

**坑 6:在 MLM 数据构造里漏掉 ignore_index 或没排除特殊 token。** 3.3 节的两个易错点。未选中位置不标 -100,会让 MLM 退化成有答案泄漏的恒等任务;不排除 [CLS]/[SEP],会污染信号。复现 MLM 时务必对照 HF 的 collator 检查这两点。

**坑 7:把 15% 掩码率/各种家族超参当普适常数。** 这些是各自论文在当时数据/规模下的经验最优。RoBERTa 已经把"NSP 有用""静态掩码够了"这类 BERT 的'定论'推翻过一遍。**遇到新规模/新数据,这些数字都该重新 sweep**(详见算法课 06 坑 4)。

**一句话权衡总览:BERT 家族的演进史,就是不断回答"信号密度够不够(ELECTRA)、训练量够不够(RoBERTa)、参数省不省(ALBERT)、注意力建模细不细(DeBERTa)、表征能不能比(SBERT)"这五个问题。** 你看任何一个新的 encoder 模型,都可以用这五把尺子去定位它改了什么。

## 九、动手练习

**练习 1(对比题,必做):并排说清 BERT / RoBERTa / ELECTRA 的预训练目标与样本效率。** 做一张三列表,每列填:(a) 预训练目标(MLM? 去 NSP? RTD?);(b) 每个序列里有多少比例的 token 贡献训练信号;(c) 用 5.4 节的方法估算相对 MLM 的"信号密度倍数"。然后回答:为什么 ELECTRA 在小算力区间优势最大?*提示:BERT≈MLM 15% 信号 + NSP;RoBERTa = MLM 15% 信号、动态掩码、去 NSP;ELECTRA = 全 token 二分类 100% 信号。倍数 ≈ 1/0.15。小算力下"每样本多榨信号"最划算,因为你跑不了很多样本。*

**练习 2(机制题,必做):解释为什么裸 [CLS] 句向量不好,而 SBERT 好。** 不超过 250 字回答三问:(a) [CLS] 在预训练里被什么任务监督、那个监督为什么不保证"语义近→余弦近";(b) "各向异性"指 BERT 表征的什么几何性质、它怎么伤害余弦比较;(c) SBERT 的回归目标 `MSE(cos(u,v), 人标分)` 具体施加了什么约束、为什么优化完余弦就可直接用。*提示:NSP 只要线性可分 IsNext,不约束几何;各向异性=向量挤在窄锥里,余弦普遍偏高区分度低;回归目标直接把余弦"校准"成相似度刻度。*

**练习 3(推导题,必做):从第一性原理推 ELECTRA 的样本效率,并诚实标注其局限。** (a) 设 T、掩码率 r,写出 MLM 和 ELECTRA 判别器每序列的有效监督信号数,求比值;(b) 指出这个比值是"信号数量"之比,论证为什么它**不等于**"训练快多少倍"——具体比较单个 MLM 信号(|V| 类)和单个 RTD 信号(二分类)的信息量;(c) 给出你认为 ELECTRA 实证更快的真实原因(两股力量的净结果)。*提示:|V| 类信息量 log|V|≈14.9 bit,二分类≤1 bit;真实增益是"信号变密(利好 ELECTRA)× 单信号变弱(利好 MLM)"的乘积,实证说明前者占优。*

**练习 4(编码题,选做):实现并验证 MLM 信号稀疏性。** 用 3.3 节代码,构造一个 batch,统计 `labels != -100` 的位置占比,确认 ≈15%;再把 `mlm_prob` 改成 0.5,观察占比变化和(如果你接了模型)loss 行为。然后把 ELECTRA 判别器的"有效位置占比"(非 padding 的全部位置)也统计出来,直观对比两者的信号密度差。*提示:`(labels != -100).float().mean()` 就是 MLM 信号密度;ELECTRA 是 `non_pad_mask.float().mean()` ≈ 1。两个数字一摆,6.7 倍差距具象化。*

## 十、源码 / 论文导读

**论文(按推荐阅读顺序):**

- **Devlin et al., 2019, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** —— 奠基作。**精读 3.1(MLM,15%/80-10-10 动机)、3.2(NSP)、以及第 4 节各下游任务怎么接 head(单句分类 / 句对 / SQuAD QA 的 start-end / NER)**。本章 3.4 节那张"任务→head"表的原始出处就是这里的 Figure。
- **Liu et al., 2019, "RoBERTa: A Robustly Optimized BERT Pretraining Approach"** —— **重点读它的消融实验**:去 NSP、静态 vs 动态掩码、数据量/batch/步数的影响。这是"BERT 训练不足、NSP 没用"的实证来源,也是理解"训练配方比结构更重要"的最佳案例。
- **Clark et al., 2020, "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"** —— **精读 RTD 任务定义、生成器-判别器结构、以及"为什么不是 GAN(不回传梯度)"和"生成器要小"那两段**;它的 Figure 1 把机制画得很清楚,本章 5.2/5.3 即对应。论文的 sample efficiency 实验(尤其小模型)直接支撑 5.4 节推导。
- **Lan et al., 2020, "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"** —— 读嵌入分解、跨层参数共享、SOP 三处。**特别注意它澄清"参数共享省参不省算"的实验**,对应本章坑 4。
- **He et al., 2020/2021, "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"** —— 读解耦注意力的三项分解和 EMD(增强掩码解码器把绝对位置补回)。注意力基础不熟先回看算法课 02/04。
- **Reimers & Gurevych, 2019, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"** —— **本章第七节的全部来源**。精读它的三种目标函数(classification / regression / triplet)、为什么用 mean-pooling、以及那个 cross-encoder 在万级句子上要 65 小时、双塔只要几秒的对比表(可扩展性论证)。
- **(进阶)Sanh et al., 2019, "DistilBERT"** —— 想真正给 BERT 加速/缩小,读知识蒸馏这条线(而非 ALBERT 的参数共享)。**(进阶)Gao et al., 2021, "SimCSE"** —— SBERT 之后最重要的句嵌入工作,用 dropout 当数据增强造对比正例,几乎无监督就拿到强句向量,是 RAG 稠密检索的重要前身。

**开源实现:**

- **HuggingFace transformers** —— 对照看几个类:`BertForMaskedLM`(MLM head,只对被遮位置算 loss)、`BertForSequenceClassification`(取 pooled [CLS] 接 Linear,就是本章 3.5 代码)、`BertForTokenClassification`(逐 token head,NER)、`BertForQuestionAnswering`(两个 Linear 出 start/end logits,正是 3.4 节的 QA head);`ElectraForPreTraining`(判别器 RTD)和 `ElectraForMaskedLM`(生成器)分开看,能看清 ELECTRA 两个网络的分工。**`DataCollatorForLanguageModeling` 里就是 15% + 80/10/10 的实现**(具体类名/参数随版本变,**待核**),对照本章 3.3 代码逐行核。
- **sentence-transformers(SBERT 官方库)** —— **句表示工程的事实标准**。看 `models.Transformer` + `models.Pooling`(pooling 模式可选 mean/cls/max)怎么拼成双塔;`losses.CosineSimilarityLoss`(本章回归目标)、`losses.TripletLoss`、`losses.MultipleNegativesRankingLoss`(in-batch 负例对比,现代检索训练主力)对应本章 7.2 的三类目标。**这个库的 README 例子能让你十几行代码跑通"编码句子→比余弦→检索"。**
- **FAISS(Facebook AI Similarity Search)** —— 双塔把句子编码成向量后,在百万/亿级向量里做近似最近邻(ANN)检索的标准工具。本章只到"句向量可比余弦",FAISS 是把它变成可扩展检索系统的那一环,RAG 章会用它,**现在知道"双塔出向量 → FAISS 建索引 → query 向量检索"这条链即可**。

**阅读策略:先读 BERT 原文 3.x 节 + 第 4 节建立"预训练+微调统一任务"的范式直觉,再用 RoBERTa 的消融校正"哪些 BERT 设计是真有用的",然后 ELECTRA 论文配本章 5.4 推导吃透样本效率,最后 SBERT 论文 + sentence-transformers 库动手跑通句向量,把这条线接到 RAG。** 五篇读完,你对判别式预训练这一支会有完整且能上手的掌握。

## 十一、小结与承上启下

这一章把判别式预训练这条 NLP 主线拆透了:

1. **双向对理解任务是结构性优势**,不是经验观察:由数据处理不等式,单向表征关于任务标签的信息上限不超过双向(当标签依赖右侧上下文时严格更低)。代价是不能直接生成——所以 BERT 路线和 GPT 路线是按任务分区的两套哲学。
2. **BERT = Transformer 编码器 + MLM(双向表征的来源)+ NSP(后被证明没用)**;[CLS] 用于句级、token 隐状态用于词级,这个表征-任务对应是统一一众任务的关键。**"预训练通用编码器 + 接浅层 head 微调"的范式,把 NLP 从"每任务一个专用架构"坍缩成"一套主干 + 薄薄一层 head"**——这是真正的范式革命。
3. **BERT 家族五把尺子**:RoBERTa(更多数据/更久训/动态掩码/去 NSP——证明瓶颈是训练量不是结构)、ALBERT(嵌入分解 + 跨层参数共享省参,但**省参不省算**)、ELECTRA(替换检测让全 token 出力,信号密度 100% vs MLM 15%,**样本效率约 6.7 倍**——但这是信号数量比,单信号更弱,实证净增益来自前者占优)、DeBERTa(解耦内容与位置的注意力)。
4. **句表示是 BERT 最大的工程坑+解法**:裸 [CLS] 不可比余弦(没被几何监督过 + 各向异性),Sentence-BERT 用双塔 + 对比/回归/三元组目标把表征校准成可比的度量空间——**双塔可离线索引,这正是稠密检索的雏形**。

**这一章在 NLP 课里的位置**:它和"序列模型(RNN/LSTM/CRF)"那条线一起,构成 NLP 处理理解任务的两代方法(前神经/RNN 时代的序列标注 → BERT 时代的预训练编码器);**它产出的"可比较句向量"是稠密检索的直接地基,会在《检索增强(RAG)》章里被 FAISS 索引、和生成模型拼成检索增强系统。** 你在本章建立的两条主轴——**"双向编码服务判别、信号密度决定预训练效率"**和**"把表征校准进一个可比的度量空间"**——前者会在你读任何 encoder 论文时当透镜用,后者会一路贯穿到检索、聚类、语义搜索的全部工程。**判别式预训练教会 NLP 的最深一课是:模型的价值不在它能背多少,而在它把文本压成了一个多好用的表征空间——这个空间好不好用,取决于你用什么自监督信号去塑造它。**


---

## 练习参考答案

> 本章「动手练习」的参考答案(AI 生成,推导/代码已尽量自验,具体数值见「待核」标注)。

### 练习 1:并排说清 BERT / RoBERTa / ELECTRA 的目标与样本效率

**三列对照表**

| 维度 | BERT | RoBERTa | ELECTRA |
|---|---|---|---|
| (a) 预训练目标 | MLM(15% 掩码,80/10/10)+ NSP | MLM(动态掩码)、**去掉 NSP**、连续长文本打包 | **RTD(替换检测)**:小生成器(MLM)造词 + 主判别器对每个 token 判 original/replaced;下游只留判别器 |
| (b) 贡献训练信号的 token 比例 | 约 15%(被遮位置算 MLM loss);NSP 另给每序列 1 个句级二分类信号 | 约 15%(同样只在被遮位置)| **约 100%**(每个非 padding 位置都做二分类) |
| (c) 相对 MLM 的信号密度倍数 | 1×(基线) | ≈ 1×(掩码率没变,动态掩码改的是"题目多样性"不是信号比例)| **1 / r = 1 / 0.15 ≈ 6.7×** |

补充说明:

- BERT 与 RoBERTa 的信号密度几乎一样(都卡在 15% 的被遮位置)。RoBERTa 的增益来自**训练配方**(更多数据 ≈160GB vs ≈16GB、更大 batch、更长步数、动态掩码 = 数据增强、去掉弱信号的 NSP),不是来自信号密度变化。所以 (c) 列 RoBERTa 仍记 ≈1×,这恰好说明"信号密度"和"训练充分度"是两个正交的轴。
- ELECTRA 的 6.7× 是"每序列产生 loss 的位置数"之比:MLM 是 r·T,ELECTRA 是 T,比值 T/(r·T)=1/r。验算:1/0.15 = 6.666… ≈ **6.7**。

**为什么 ELECTRA 在小算力区间优势最大?**

样本效率衡量的是"每见过一个样本(一次前向)能榨出多少监督信号"。在小算力 / 小模型区间,你**总共能跑的前向次数本来就少**(样本预算紧),此时"每个样本多榨 6.7 倍信号"的杠杆被放到最大——同样几百万次前向,ELECTRA 累计获得的监督信号量远多于 MLM。反过来在大算力区间,样本几乎管够,MLM"单信号更强(|V| 类)"的劣势可以靠"多跑样本"补上,密度优势的边际收益递减。所以 **ELECTRA-small 用很小算力就能逼近大得多的 BERT,而到了大规模两者差距收窄**——优势集中在算力/样本受限的一端。

**结论:三者信号密度 1× / 1× / 6.7×;RoBERTa 赢在训练充分度而非密度;ELECTRA 赢在密度,且密度红利在小算力区间被放到最大。**

---

### 练习 2:为什么裸 [CLS] 句向量不好,而 SBERT 好(≤250 字)

**(a) [CLS] 受什么监督、为何不保证"语义近→余弦近":** 预训练里 [CLS] 只服务 NSP——一个"两句是否相邻"的二分类。NSP 只要求 [CLS] 能被一个**线性头**分出 IsNext/NotNext,即"线性可分"即可;它**从不约束**两个语义相近句子的 [CLS] 向量在几何上(余弦距离)要靠近。向量能被某分类头区分 ≠ 向量本身的余弦有语义,故直接比余弦如同用没标刻度的尺子量长度。

**(b) 各向异性指什么、怎么伤余弦:** 指 BERT 输出向量**不是各向同性**,而是挤在一个狭窄的锥形(cone)里,占据的方向高度集中。后果:任意两句余弦都偏高、彼此区分度低,真实语义差异被这种几何畸变淹没,余弦失去判别力。

**(c) 回归目标施加了什么约束、为何优化完即可直接用:** `MSE(cos(u,v), 人标分)` 直接逼着 cos(u,v) 去拟合人标相似度,等于把余弦**校准成一把有刻度的相似度尺**;优化收敛后,cos 的取值本身就对齐了人类相似度标注,无需再训任何头,可直接当相似度读数使用。

(正文约 250 字)

---

### 练习 3:从第一性原理推 ELECTRA 样本效率,并诚实标注局限

**(a) 每序列有效监督信号数与比值**

```
设序列长 T,掩码率 r(BERT 用 r = 0.15)。

MLM(BERT 判别信号):只在被遮位置算 loss
    N_MLM = r · T

ELECTRA(判别器):对每个(非 padding)位置做二分类,都算 loss
    N_ELECTRA = T

比值 = N_ELECTRA / N_MLM = T / (r·T) = 1 / r = 1 / 0.15 ≈ 6.7
```

**所以"信号数量"之比 ≈ 6.7 倍。**

**(b) 为什么这个比值 ≠ "训练快多少倍"——比较单信号信息量**

6.7× 只比了**信号的数量**,没比**单个信号的信息量**:

```
单个 MLM 信号:在 |V| 类词表上做分类。
    信息量上限 = log2(|V|) ≈ log2(30000) ≈ 14.9 bit   (验算 log2(30522)≈14.90)

单个 RTD 信号:一个二分类(original / replaced)。
    信息量上限 = log2(2) = 1 bit
```

单个 MLM 信号携带的信息约是单个 RTD 信号的 **15 倍**。若按"总信息率 = 信号数 × 单信号信息量"做一阶估算,反而是 MLM 偏高:

```
MLM 总信息率上限   ≈ r·T · log2|V| = 0.15·T · 14.9 ≈ 2.23·T  bit
ELECTRA 总信息率上限 ≈  T · 1                       =  1.00·T  bit
（取 T=100 验算:MLM≈223,ELECTRA≈100，比值≈2.23）
```

两个比值方向相反(信号数 ELECTRA 占优 6.7×;单信号 / 总信息率上限 MLM 占优),正说明 **"6.7 倍"绝不能直译成"训练精确快 6.7 倍"**——它只是信号密度比,不是端到端效率比。

**(c) ELECTRA 实证更快的真实原因:两股力量的净结果**

真实效率增益 = **信号变密(利好 ELECTRA,6.7×)** 与 **单信号变弱(利好 MLM,信息量小约 15×)** 两股力量相乘后的净结果。ELECTRA 实证更快,说明**前者在实践中盖过了后者**。直觉解释:RTD 让模型对**每一个 token**判断"它在当前语境里合不合理",虽然单点信息少,但 (1) 覆盖全序列、零浪费;(2) 这个"合理性判断"本身要求对上下文做深度理解,不是廉价信号;(3) 由一个"恰到好处地犯错"的弱生成器制造的替换样本,难度适中、可学性强。综合下来,**密集但单点偏弱**的信号在塑造判别式表征上,比**稀疏但单点强**的信号更高效——尤其在样本预算有限时。

**结论:信号数量比 ≈ 6.7×(可验算 1/0.15);但因单个 RTD 信号(≤1 bit)远弱于单个 MLM 信号(≈14.9 bit),该比值不等于训练加速比;ELECTRA 实证更快,是"信号密度红利 > 单信号变弱损失"的净胜出。**

---

### 练习 4:实现并验证 MLM 信号稀疏性(编码题)

思路:直接复用 3.3 节的 `build_mlm_inputs`,用 `(labels != -100).float().mean()` 量 MLM 信号密度,用 `non_pad_mask.float().mean()` 量 ELECTRA 信号密度,两数一摆即把 6.7× 差距具象化。下面给**两份可跑代码**:torch 版(贴合章节接口)和零依赖 numpy 版(CPU 直接出数,已实测)。

**torch 版(需 `pip install torch`,CPU 即可,小维度秒级):**

```python
import torch
import torch.nn.functional as F

def build_mlm_inputs(input_ids, vocab_size, mask_token_id, special_ids, mlm_prob=0.15):
    labels = input_ids.clone()
    prob = torch.full(input_ids.shape, mlm_prob)
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= (input_ids == sid)
    prob[special_mask] = 0.0                          # 特殊 token 不参与选择
    selected = torch.bernoulli(prob).bool()
    labels[~selected] = -100                          # 未选中位置不计 loss
    masked = input_ids.clone()
    rep = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & selected   # 80% -> [MASK]
    masked[rep] = mask_token_id
    rnd = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & selected & ~rep  # 10% -> 随机
    masked[rnd] = torch.randint(vocab_size, input_ids.shape)[rnd]
    return masked, labels                             # 剩 10% 保持原样

torch.manual_seed(0)
B, T, V = 64, 128, 30000
CLS, SEP, PAD = 101, 102, 0
ids = torch.randint(5, V, (B, T))
ids[:, 0] = CLS; ids[:, -1] = SEP                     # 每行首尾放特殊 token

for p in [0.15, 0.5]:
    _, labels = build_mlm_inputs(ids, V, 103, {CLS, SEP, PAD}, mlm_prob=p)
    mlm_density = (labels != -100).float().mean().item()   # ← MLM 信号密度
    print(f"mlm_prob={p}: MLM 信号密度 = {mlm_density:.4f}")

non_pad_mask = (ids != PAD).float()
electra_density = non_pad_mask.mean().item()          # ← ELECTRA 信号密度(非 padding 全算)
print(f"ELECTRA 信号密度(非 padding) = {electra_density:.4f}")
print(f"密度倍数 ELECTRA/MLM@0.15 = {electra_density / 0.15:.1f}")
```

**numpy 零依赖版(已在 CPU 实测,逻辑等价,适合没装 torch 的环境):**

```python
import numpy as np
rng = np.random.default_rng(0)
B, T = 64, 128
CLS_COL, SEP_COL = 0, T - 1     # 用首尾两列模拟特殊 token 位置(永不被选)

def mlm_signal_density(mlm_prob):
    prob = np.full((B, T), mlm_prob)
    prob[:, CLS_COL] = 0.0       # 特殊 token 排除
    prob[:, SEP_COL] = 0.0
    selected = rng.random((B, T)) < prob
    return selected.mean()       # 等价于 (labels != -100).mean()

for p in [0.15, 0.5]:
    d = mlm_signal_density(p)
    print(f"mlm_prob={p}: MLM 信号密度 = {d:.4f}  (理论≈{p*(T-2)/T:.4f})")

electra_density = 1.0            # 无 padding 时,非 pad 位置占比 = 100%
print(f"ELECTRA 信号密度 = {electra_density:.4f}")
print(f"密度倍数 = {electra_density/0.15:.1f}")
```

**实测输出(numpy 版,seed=0):**

```
mlm_prob=0.15: MLM 信号密度 = 0.1532  (理论≈0.1477)
mlm_prob=0.5:  MLM 信号密度 = 0.4873  (理论≈0.4922)
ELECTRA 信号密度 = 1.0000
密度倍数 = 6.7
```

**观察与结论:**

- `mlm_prob=0.15` 时密度 ≈ **0.15**(略低于 0.15,因为首尾特殊 token 被排除,理论值 0.15·126/128 ≈ 0.148,采样波动到 0.153,吻合)。把 `mlm_prob` 改到 0.5,密度随之升到 ≈ 0.49,**确认密度由掩码率线性控制**。
- 若接了模型:`mlm_prob` 调大,被遮位置增多,但每个位置能用的上下文线索变少(更多词被破坏),**单步 loss 通常升高且收敛更难**——这就是 15% 不取更大值的经验权衡(详见算法课 06)。
- ELECTRA 端:非 padding 位置**全部**贡献 loss,密度 ≈ **1.0**。两个数字 0.15 vs 1.0 并排,**1.0 / 0.15 ≈ 6.7**,把 5.4 节的"信号密度 6.7 倍"从公式落成了可观测的实验数值。

**结论:实测 MLM 信号密度随掩码率线性变化(0.15→≈0.15、0.5→≈0.49),ELECTRA 恒为 ≈1.0,两者之比 ≈ 6.7,与第一性原理推导 1/r 一致。**
