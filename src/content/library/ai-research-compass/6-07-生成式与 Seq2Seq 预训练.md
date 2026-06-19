---
title: "生成式与 Seq2Seq 预训练:GPT、T5 与 BART"
slug: "6-07"
collection: "ai-research-compass"
group: "自然语言处理专家课程"
order: 6007
summary: "这章把你从\"会用 BERT 做判别式微调\"带到能讲清生成式/编码-解码预训练的内部机理,能自己构造 T5 的 span corruption 训练样本、复现 text-to-text 的任务统一接口,并能在编码器-only / 解码器-only / 编码-解码三种架构之间为一个具体任务做出有依据的选型。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T05:59:08.000Z"
updatedAt: "2026-06-19T05:59:08.000Z"
---
> 这章把你从"会用 BERT 做判别式微调"带到能讲清生成式/编码-解码预训练的内部机理,能自己构造 T5 的 span corruption 训练样本、复现 text-to-text 的任务统一接口,并能在编码器-only / 解码器-only / 编码-解码三种架构之间为一个具体任务做出有依据的选型。

## 一句话定位

上一章(06,BERT)讲的是**判别式预训练**:用双向编码器把句子压成向量,再接一个分类头去做理解任务。这一章讲的是另外两条路线——**自回归生成式**(GPT 系)和**去噪式编码-解码**(T5/BART)。它们的共同点是:预训练目标本身就是"生成文本",所以下游任务不必再设计任务专属的输出头,而是把任务**也写成文本**。这条路最终通向了"一个模型干所有 NLP 任务",也通向了 ChatGPT。

本章不重复推导 Transformer 内部结构、注意力、解码采样的细节——那些请看《大模型算法专家课程》:多头注意力详见**算法课 03**,自回归/掩码/去噪三种预训练目标的横向对比与梯度分析详见**算法课 06**,零样本/少样本能力为何随规模"涌现"详见**算法课 08**,解码采样(greedy/beam/top-p)详见**算法课 13**,in-context learning 机制详见**算法课 14**。RLHF/对齐的损失推导详见**强化学习课 08(RLHF 全流程)/09(DPO)**。本章重心在 NLP 层面:这三种范式各自的**预训练目标怎么构造、为什么有效、适配什么任务、统一接口的工程价值**。

## 1. 预备:三种架构,一张注意力掩码图

要理解后面所有内容,先把一个常被含糊带过的点钉死:**编码器-only、解码器-only、编码-解码,本质区别只在"谁能看见谁"——也就是注意力掩码(attention mask)的形状**。结构组件(注意力、FFN、LayerNorm)是同一套。

设输入序列长度为 n,注意力得分矩阵 A ∈ ℝ^{n×n},A[i,j] 表示位置 i 关注位置 j 的程度。掩码就是在 softmax 之前给某些 (i,j) 加 −∞,使其权重归零。

```
编码器-only(BERT):全可见(双向)
  允许 i 看 j 对所有 (i,j) —— mask 全 0
  → 位置 i 能用左右全部上下文,适合"理解/编码"

解码器-only(GPT):因果掩码(causal / 下三角)
  允许 i 看 j 当且仅当 j ≤ i —— 上三角置 −∞
  → 位置 i 只能看自己和左边,保证"预测下一个词时没偷看答案"

编码-解码(T5/BART):两段
  encoder 内部:全可见(双向),把源序列编码成记忆 H
  decoder 自注意力:因果掩码(只看已生成部分)
  decoder→encoder 交叉注意力(cross-attention):全可见 H
  → 既能双向理解输入,又能自回归生成输出
```

**记住这张图,后面 GPT/T5/BART 的差异都能还原成"掩码形状 + 训练目标"两个旋钮。** 编码器-only 为什么不能直接做生成?因为它没有因果掩码,训练时每个位置都看到了未来,推理时却没有未来可看,训练-推理不一致。这正是生成式必须用因果掩码的根本原因。

## 2. GPT 系:自回归解码器预训练

### 2.1 问题:把"理解语言"压缩成"预测下一个词"

GPT(Generative Pre-Training,Radford 等,2018)的核心赌注是:**只要能足够好地预测下一个词,就必然学到了语法、语义、事实乃至推理**。直觉:要预测 "三十六计走为______" 的下一个词,模型得知道成语;要预测 "2+3=______",得会算术;要续写一段论证,得理解逻辑。预测下一个词这个目标看似简单,却把无监督学习和语言理解绑在了一起。

### 2.2 推导:自回归分解与训练损失

给定一段文本 token 序列 x = (x₁, x₂, …, x_T)。任何联合分布都可以用链式法则**精确**分解为条件分布之积(这是概率论恒等式,无近似):

```
P(x₁,…,x_T) = ∏_{t=1}^{T} P(x_t | x_1, …, x_{t-1})
```

GPT 用一个带因果掩码的 Transformer 来参数化每个条件 `P(x_t | x_<t)`:把前缀 x_<t 编码成隐状态 h_t ∈ ℝ^d,再经过输出投影(通常与输入词嵌入共享权重,叫 weight tying,省一份 |V|×d 参数)得到 logits,过 softmax:

```
h_t      = Transformer_causal(x_1, …, x_{t-1})          # h_t 只依赖左侧
logits_t = h_t · Wᵀ + b              ,  W ∈ ℝ^{|V|×d}    # 与词嵌入共享 W
P(· | x_<t) = softmax(logits_t)
```

训练目标是**最大化数据的对数似然**,等价于**最小化逐 token 的交叉熵**:

```
L(θ) = − (1/T) · Σ_{t=1}^{T} log P_θ(x_t | x_<t)
```

注意一个工程上极重要的事实:**因为有因果掩码,长度为 T 的一个序列同时提供了 T 个训练信号**(每个位置都在预测它的下一个词),而且这 T 个预测可以在一次前向中**并行**算出来。这就是 GPT 训练高效的原因——它不是"一次只学一个词",而是"一个序列里 T 个位置同时学,且互不偷看"。这一点是与 RNN 语言模型(NLP 课 03)的关键分水岭:RNN 也做下一个词预测,但隐状态必须按时间步串行递推,无法在序列维度并行。

把这个损失和 BERT(06)对比:BERT 的 MLM 只对被掩盖的约 15% token 算损失(其余位置不产生预测目标),而 GPT 对**每个位置**都算损失。所以同样一批 token,GPT 的"有效监督信号"更密,但每个信号更弱(只用了单向上下文)。这是"密但单向" vs "稀但双向"的权衡,**算法课 06** 对此有专门的梯度/样本效率分析。

### 2.3 从微调到零样本/少样本:范式如何变化

- **GPT-1(2018)**:预训练 + 任务专属微调。下游仍要为每个任务加一个线性头并改写输入(比如蕴含任务把前提和假设用分隔符拼起来)。这一步还很像 BERT 的用法。
- **GPT-2(2019)**:核心发现是**零样本(zero-shot)**——把任务用自然语言描述进 prompt,不微调也能做。比如翻译写成 "English: cheese\nFrench:",让模型续写。背后逻辑:如果训练语料足够大且多样,各种任务的"自然出现形式"本来就在语料里,语言模型在学下一个词时顺带学会了它们。
- **GPT-3(2020)**:**少样本/in-context learning**——在 prompt 里给几个示例(demonstration),模型不更新任何权重就能模仿。这种"给例子就会"的能力随模型规模急剧变强,被称为**涌现(emergence)**。

**这里只点到为止:为什么规模一大零样本/少样本能力会涌现、in-context learning 的内部机制是什么,详见《算法课 08(Scaling Laws 与涌现)/14(ICL 与提示)》。** 本章要你记住的 NLP 层面结论是:**生成式范式让"任务"从"改模型结构"退化成了"改输入文本"**。这是后面 T5/指令微调的思想源头。

### 2.4 为什么生成式更"通用"

一句话:**任何 NLP 任务的输出都可以表示成一串 token,而判别式的输出头是任务专属的**。情感分类的输出头是 2 维,NER 是 BIO 标签序列,翻译是另一个词表上的序列——判别式要为每种输出形态设计不同的头与损失。而生成式只有一个输出形态:文本。于是:

- 分类 → 生成类别词("正面"/"负面")
- 抽取 → 生成被抽取的片段
- 翻译/摘要 → 生成目标文本
- 多步推理 → 生成中间推理链(chain-of-thought)

判别式做不了开放式生成(它没有逐步生成机制),生成式却能覆盖判别式的几乎所有任务——**这种"输出空间的统一"就是通用性的根源**,代价是分类这种本来一步能出结果的任务,现在要逐 token 解码,且可能生成词表里不该出现的答案(需要约束解码或后处理)。

## 3. T5:把一切 NLP 任务统一成 Text-to-Text

### 3.1 问题:接口碎片化

到 2019 年,迁移学习已被验证有效,但**接口五花八门**:BERT 加分类头、加 span 头、加序列标注头……每种任务要写一套适配代码、调一套超参。T5(Text-to-Text Transfer Transformer,Raffel 等,2019,论文俗称 "Colossal Clean Crawled Corpus / C4" 那篇)的提案极简单也极有影响力:**把每个任务都建模成"输入一段文本 → 输出一段文本",用任务前缀(task prefix)告诉模型在做什么任务**。

### 3.2 机制:统一接口

同一个编码-解码模型、同一个交叉熵损失、同一套解码,跑所有任务。任务靠输入里的前缀字符串区分:

```
翻译:    "translate English to German: That is good."   → "Das ist gut."
摘要:    "summarize: <一篇长文档>"                        → "<摘要>"
分类:    "cola sentence: The course is jumping well."    → "acceptable"
相似度:  "stsb sentence1: ... sentence2: ..."            → "3.8"     (回归值也当字符串生成)
蕴含:    "mnli premise: ... hypothesis: ..."             → "entailment"
```

注意几个非平凡的设计:回归任务(语义相似度,连续分数)被处理成**生成一个数字字符串**(如 "3.8"),离散化到固定刻度;分类输出是**类别词本身**而非整数 id。这意味着模型在"用语言表达答案",而不是"输出一个抽象类别"。**这个看似随意的决定,正是后来 instruction tuning 和 ChatGPT 把所有交互都做成自然语言对话的雏形。**

### 3.3 预训练目标:Span Corruption(片段破坏)

T5 没有用 GPT 的纯自回归,也没有用 BERT 的单 token MLM,而是用一种适配编码-解码结构的**去噪目标**,叫 span corruption(也叫 replace-spans)。机制:

1. 在输入里随机选若干 token,目标**总破坏率约 15%**(与 BERT 同量级)。
2. 把**连续被选中的 token 合并成一个 span**,用一个**唯一的哨兵 token**(sentinel,如 `<X>`、`<Y>`、`<Z>`)替换整段。
3. 模型的目标(target)是**按顺序输出"每个哨兵 + 它对应的被删片段",最后接一个结束哨兵**。

举例(原句:`Thank you for inviting me to your party last week .`):

```
原始:   Thank you for inviting me to your party last week .
选中:                 [inviting me]        [last]
输入:   Thank you for <X> to your party <Y> week .
目标:   <X> inviting me <Y> last <Z>
        (<Z> 是收尾哨兵,标记"输出结束")
```

为什么这样设计,而不是直接学 BERT 在原位填空?三个理由,逐条想清楚:

- **匹配编码-解码结构**:BERT 是在编码器原位预测被掩盖 token,而编码-解码的解码器是从左到右生成一个**新序列**。span corruption 把任务变成"给定带洞的输入,生成洞里的内容"——天然是个 seq2seq 任务,正好喂给编码-解码架构。
- **目标序列短**:只生成被破坏的片段(约 15%),而不是重建整句。若让 decoder 重建整句,目标长度≈输入长度,训练成本翻倍且大部分是 copy 无信息量的内容。span corruption 的**目标长度 ≈ 0.15·n + 哨兵数**,大幅省算力。这是 T5 论文实验比较后选定的方案。
- **span 而非单 token**:破坏连续片段(而非孤立 token)逼模型学**更长程的依赖与短语级语义**,而不是靠局部线索补一个词。span 长度通常按平均长度(论文设定约 3)的几何/泊松采样。

下面给出**可照着实现**的 span corruption 构造代码——这是本章的核心动手片段:

```python
import random

def build_span_corruption(tokens, corrupt_rate=0.15, mean_span_len=3.0):
    """T5 span corruption: 返回 (encoder_input, decoder_target)。
    tokens: 已分词的 token id 列表(此处用占位 token 演示机制)。
    哨兵用字符串 '<extra_id_0>', '<extra_id_1>', ... (HF T5 的真实哨兵命名)。
    """
    n = len(tokens)
    n_to_corrupt = max(1, round(n * corrupt_rate))      # 总破坏 token 数(约 15%)

    # 1) 真实 T5 按 span 采样;此处为演示先随机选孤立点再聚合,平均 span 长≈1,
    #    mean_span_len 在本简化版中不生效(见下方坑点)
    # 2) 随机选 span 的起点,保证 span 之间不重叠、不相邻(相邻会合并成一个哨兵)
    #    这里用"先随机打标记再聚合连续段"的等价做法,逻辑更清晰:
    corrupt_mask = [False] * n
    chosen = random.sample(range(n), n_to_corrupt)       # 随机选要破坏的位置
    for i in chosen:
        corrupt_mask[i] = True

    enc_input, dec_target = [], []
    sentinel_id = 0
    i = 0
    while i < n:
        if corrupt_mask[i]:
            sent = f"<extra_id_{sentinel_id}>"
            enc_input.append(sent)                       # 输入里:整段 span 换成一个哨兵
            dec_target.append(sent)                      # 目标里:哨兵 + 这段被删的原 token
            while i < n and corrupt_mask[i]:             # 合并连续被破坏的 token 成一个 span
                dec_target.append(tokens[i])
                i += 1
            sentinel_id += 1
        else:
            enc_input.append(tokens[i])                  # 未破坏的原样保留
            i += 1
    dec_target.append(f"<extra_id_{sentinel_id}>")       # 收尾哨兵,标记输出结束
    return enc_input, dec_target

# demo
toks = "Thank you for inviting me to your party last week .".split()
enc, dec = build_span_corruption(toks, corrupt_rate=0.3, mean_span_len=2.0)
print("ENC:", " ".join(map(str, enc)))
print("DEC:", " ".join(map(str, dec)))
```

注意**坑点**:哨兵在 encoder 输入和 decoder 目标里必须**严格一一对应、顺序一致**;HF 的 T5 真实实现里哨兵 `<extra_id_0>` 对应词表**末尾**的特殊 token(id 从大到小排),自己造数据时若哨兵 id 错位,模型会学成乱码。另外被破坏的位置在论文里是按 span 采样而非我演示的"先选孤立点再聚合",我这里为讲清机制做了简化(标了一句),严格复现请对照 `t5x` 或 HF 的 `transformers/models/t5` 数据处理。

### 3.4 text-to-text 任务格式化(下游使用)

预训练完后,下游任务的代码长这样——**注意没有任务专属的 head,只有不同的前缀字符串和不同的目标字符串**:

```python
# 用 HuggingFace transformers(伪示意,API 以版本为准)
from transformers import T5Tokenizer, T5ForConditionalGeneration
tok = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def format_task(task, **kw):
    if task == "summarize":
        return f"summarize: {kw['document']}"
    if task == "translate_en_de":
        return f"translate English to German: {kw['text']}"
    if task == "sentiment":                   # 分类也写成 text→text
        return f"sst2 sentence: {kw['text']}"
    raise ValueError(task)

src = format_task("sentiment", text="The plot was dull and predictable.")
ids = tok(src, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=8)
print(tok.decode(out[0], skip_special_tokens=True))   # 期望生成 "negative"

# 训练时:label 也是文本 "negative" 的 token id,损失就是普通的逐 token 交叉熵
# loss = model(input_ids=ids, labels=tok("negative", return_tensors="pt").input_ids).loss
```

**统一接口的工程价值有多大?** 它把"为每个任务写适配层 + 调头"这件事彻底消除了。多任务训练只需把不同任务的样本(已格式化成 text→text)**混在一起喂**,模型自己靠前缀区分。新增一个任务=新增一个前缀+一批样本,零代码改动。这种"接口即文本"的思想,是从 T5 到 FLAN 再到 ChatGPT 这条线的主干。

## 4. BART:去噪自编码器,理解与生成兼得

### 4.1 问题:GPT 强生成弱理解,BERT 强理解不能生成

GPT(因果解码器)擅长生成但只有单向上下文,理解类任务略吃亏;BERT(双向编码器)理解强但结构上不能生成。BART(Lewis 等,2019)的思路:**用一个完整的编码-解码 Transformer,encoder 双向(像 BERT)、decoder 自回归(像 GPT),预训练目标是"把被任意噪声破坏的文本重建回原文"**——所以它是个**去噪自编码器(denoising autoencoder)**。

### 4.2 机制:多种噪声 + 重建整句

与 T5 只用 span corruption 不同,BART 用一组**任意噪声变换**破坏输入,decoder 目标是**重建完整原文**(不是只补片段):

```
原句:  A B C . D E .
- Token Masking      : A _ C . D E .          (单 token 换 [MASK],同 BERT)
- Token Deletion     : A C . D E .            (直接删掉,模型还得学"哪里少了")
- Text Infilling     : A _ . D E .            (一段 span 换成单个 [MASK];输入不暴露被删 token 数,模型必须自己推断长度——这正是它比 T5 多区分哨兵的方式更难、也更贴近真实文本修复的原因。BART 论文:text infilling 训练模型预测一个 span 缺了几个 token)
- Sentence Permutation: D E . A B C .         (句子顺序打乱,学篇章顺序)
- Document Rotation  : C . D E . A B          (从随机 token 起循环旋转,学"找开头")
重建目标(decoder 输出): A B C . D E .         (永远是完整原文)
```

**关键对比 T5**:T5 的 text infilling 用**多个、可区分的哨兵**且只生成被删片段;BART 的 text infilling 用**单个不可区分的 `[MASK]` 且不告诉模型删了几个 token**,decoder 要重建整句——更难、更接近真实的"修复一段残缺文本"。论文实验中,**text infilling + sentence permutation 的组合效果最好**(具体相对增益数值此处不复述,见论文表格)。

### 4.3 为什么 BART 摘要/翻译特别强

- **重建整句**这个目标天然就是 seq2seq 形态,微调到摘要/翻译几乎是"同构迁移":摘要 = 输入长文档、输出短文本;翻译 = 输入源语、输出目标语。decoder 在预训练时已经练满了"生成流畅完整文本"。
- **encoder 双向**保证对输入的理解不弱于 BERT,所以判别任务(如 GLUE)也能打。
- **mBART(多语言 BART)**:在多语言语料上做同样的去噪预训练,得到的模型对**机器翻译(尤其低资源、无监督/半监督方向)**特别有用——decoder 已经会生成多种语言的流畅文本,只需少量平行语料对齐。多语言/子词切分的工程细节见 **NLP 课 08**。

### 4.4 BART vs T5,一句话

两者都是"编码-解码 + 去噪",**差别在去噪目标的形态和下游接口哲学**:T5 押注"**统一成 text-to-text、用前缀区分任务**",目标是只生成被破坏片段;BART 押注"**多种噪声 + 重建整句**",更偏向把模型当成"会修文本/会生成"的通用 seq2seq backbone,下游仍按任务接不同用法。工程上 T5 的接口统一性影响更深远,BART 在摘要这类生成任务上长期是强 baseline。

## 5. 指令微调的开端:FLAN / T0——通向 ChatGPT 的桥

### 5.1 问题:零样本能力其实没被"对齐"到指令上

GPT-3 证明了大模型有零样本能力,但**直接问它一个任务,它经常答非所问**——因为预训练目标是"续写网上的文本",不是"听从指令完成任务"。比如你问 "把这句话翻成法语:...",它可能继续编一段类似的英文句子,而不是翻译。**模型有能力,但没被引导到"指令→执行"这个模式上。**

### 5.2 机制:多任务指令微调(instruction tuning)

FLAN(Finetuned Language Net,Wei 等,2021)和 T0(Sanh 等,2021)的做法:**把大量已有 NLP 任务,各自用自然语言"指令模板(prompt template)"改写成"指令 + 输入 → 输出"的形式,然后在这些任务上做有监督微调**。

```
情感任务样本(原本是 (text, label))改写成多个指令模板:
  "Review: {text}\nIs this review positive or negative?"        → "Positive"
  "{text}\nThe sentiment of the above review is:"               → "positive"
NER、QA、摘要、NLI ... 每个任务都配几个不同措辞的模板,几十~上千个任务混在一起训练
```

**核心实验结论(被反复验证)**:在足够多样的任务上做指令微调后,模型对**训练时没见过的新任务**也能零样本跟随指令——即**指令跟随能力本身是可以被"学会"并泛化**的,关键是**任务的多样性(任务数量、模板多样性)**而非单个任务的数据量。这就把 GPT-3 那种"会但不听话"的能力,转成了"听到指令就执行"。

### 5.3 与普通微调的本质差别(本章必须讲清的对比)

| 维度 | 普通微调(fine-tuning,如 BERT) | 指令微调(instruction tuning) |
|---|---|---|
| 训练数据 | **单个任务**的 (输入, 标签) | **多个任务**,且都改写成 (自然语言指令+输入, 文本输出) |
| 优化目标 | 让模型在**这个任务**上最好 | 让模型学会**"按指令做任意任务"这个元能力** |
| 泛化方向 | 泛化到同任务的新样本 | 泛化到**训练没见过的新任务**(zero-shot 跨任务) |
| 输出形态 | 任务专属 head 的离散标签 | 统一的**自然语言文本** |
| 部署后用法 | 一个微调权重对一个任务 | 一个模型,靠改 prompt 切换任务 |

一句话:**普通微调是"把模型调成某个任务的专家";指令微调是"把模型调成一个会读指令、按指令干活的通才"。** 前者优化"任务表现",后者优化"指令跟随这个元技能"。

### 5.4 它为什么是通向 ChatGPT 的桥

指令微调解决了"听不听指令";但它仍只优化"格式上像答案",**不优化"答案是否有用、诚实、无害"**——因为这些目标写不出明确的损失函数(你没法对"有用"求导)。补上这最后一块的,是用**人类偏好**来训练:RLHF(SFT → 训奖励模型 → PPO 优化)和它的简化路线 DPO。**这部分的完整推导(Bradley-Terry 偏好建模、奖励模型损失、KL 惩罚为何必不可少、DPO 闭式解)请看《强化学习课 08(RLHF 全流程)/09(DPO)》,本章不重复。** NLP 层面你只需记住这条演化链:

```
自回归预训练(GPT,会续写)
  → 规模放大 + 零样本/少样本(GPT-2/3,会但不听话)        [算法课 08/14]
  → 指令微调(FLAN/T0,听指令、跨任务泛化)                [本章 §5]
  → 人类偏好对齐(RLHF/DPO,答得有用/诚实/无害)           [强化学习课 08/09]
  = ChatGPT 式的对话助手                                  [NLP 课 12]
```

## 6. 关键对照表:三种架构怎么选

这是本章的"压舱"结论。给定一个任务,先问"输出是不是要逐步生成新文本",再问"输入是否需要双向理解":

| | 编码器-only(BERT 系) | 解码器-only(GPT 系) | 编码-解码(T5/BART) |
|---|---|---|---|
| 注意力 | 双向全可见 | 因果(单向) | enc 双向 + dec 因果 + cross-attn |
| 预训练目标 | MLM(掩码填空) | 自回归(预测下一词) | 去噪/span corruption(破坏再重建) |
| 天生擅长 | 分类、序列标注、抽取、句向量、检索 | 开放式生成、续写、对话、few-shot、代码 | 有明确"源→目标"映射的转换任务 |
| 典型任务 | 情感分类、NER、句子相似度、retrieval | 文本续写、对话、CoT 推理、通用助手 | 翻译、摘要、改写、数据到文本、风格转换 |
| 能否开放生成 | 否(无逐步生成机制) | 强 | 强 |
| 参数效率(同等理解任务) | 高(只需 encoder) | 中(要 decode) | 较低(两套栈,但表达力强) |

实战经验法则(标为经验,非定理):
- **纯判别/检索/打标签**且不需要生成 → 编码器-only 最省最稳(如 NER、reranker、embedding 模型仍大量用 BERT 家族)。
- **要做通用助手、对话、few-shot、不想为每个任务训模型** → 解码器-only(这是当前 LLM 的绝对主流,GPT/Llama/Qwen 全是)。
- **有清晰的"输入文本→输出文本"且方向固定**(翻译、摘要) → 编码-解码常更高效、更易训(decoder 的负担被 encoder 分担,且 cross-attention 直接对齐源序列)。

**为什么如今 decoder-only 成了绝对主流(一个常考的设计权衡)?** 不是因为它在每个任务上都最优,而是因为:(1) 它能用同一个模型 + 改 prompt 覆盖所有任务,包括编码-解码本来擅长的翻译/摘要(把源文本放进 prompt 即可),工程上极简;(2) 自回归目标对每个 token 都产生监督,训练信号最密、scaling 最干净;(3) in-context learning 这个杀手级能力主要在 decoder-only 上涌现。代价是:纯理解/检索任务上它比专门的 encoder 更费算力,且没有显式的双向编码。这也是为什么 embedding/retrieval 领域(NLP 课 10 RAG 会用到)至今仍大量保留 encoder 结构。

## 7. 设计权衡与常见坑

- **生成式做分类的隐患**:让模型生成 "positive"/"negative",它**可能生成词表里的其他词**(如 "neutral"、空串、甚至复读 prompt)。生产中要么**约束解码**(只在候选标签集合上做 argmax,即比较各标签序列的似然),要么对输出做严格后处理。直接 `generate` 后字符串匹配是脆弱的。
- **T5 哨兵错位**:自造 span corruption 数据时,encoder 哨兵与 decoder 哨兵必须严格对应、id 与词表特殊 token 一致。HF T5 哨兵 id 是从词表末尾倒着分配的,接错必崩。务必对照官方数据处理代码。
- **破坏率/平均 span 长是超参,不是随便填**:破坏率太低→监督信号太稀、训得慢;太高→输入残缺到无法重建、退化成乱猜。T5 在约 15% 破坏率、平均 span≈3 上做过消融,照搬即可,改动要有依据。
- **指令微调的"模板多样性"比"任务数据量"更重要**:只用一个措辞模板会让模型过拟合到措辞而非任务本身,换个问法就崩。每个任务配多个改写模板是关键。
- **编码-解码的双倍栈成本**:它有 encoder 和 decoder 两套层,同参数预算下"每层"更薄。别默认"编码-解码一定比 decoder-only 强"——在通用生成与 scaling 上,decoder-only 用同样的参数往往更划算。
- **把"指令微调"当成"对齐"**:指令微调只让模型**听指令**,不保证答案**有用/诚实/无害**。误以为指令微调=对齐完成,会跳过 RLHF/DPO 这步,产出"格式对但内容可能有害"的模型。

## 8. 动手练习

1. **(实现)手写 span corruption 并验证一致性。** 用上面的 `build_span_corruption` 为 20 句话生成 (enc, dec)。写一个**逆向重建函数**:用 enc 里的哨兵把 dec 里的片段填回去,断言能还原出原句的全部 token。提示:遍历 enc,遇到哨兵 `<extra_id_k>` 就去 dec 里取该哨兵后、下一个哨兵前的片段插入——这一步能逼你彻底理解哨兵的对应关系,也是定位"哨兵错位"bug 的标准手段。

2. **(对比)同一任务,三种架构怎么做。** 任选"情感分类"和"英译德"两个任务,各写出:(a) 编码器-only 的做法(输入/输出头/损失);(b) 解码器-only 的做法(prompt 写法/输出约束);(c) 编码-解码的做法(text-to-text 前缀/目标)。提示:重点说清三者的**输出空间**与**损失**差异,并指出每个任务在哪种架构上"最自然"、代价是什么。

3. **(论证)解释 text-to-text 统一接口的价值与代价。** 用不超过 300 字论证:为什么"把所有任务写成文本→文本"能让多任务训练和新增任务变得零成本?它的代价是什么(至少举两点:如分类的解码开销、可能生成非法标签)?提示:对照"判别式要为每个任务设计 head 和损失"来写。

4. **(辨析)指令微调 vs 普通微调,设计一个能区分两者的实验。** 描述一个实验:训练集只含任务 A、B、C 的指令样本,测试集是**完全没见过**的任务 D 的指令。说明"普通微调出来的模型"和"指令微调出来的模型"在任务 D 上的预期表现差异,以及这个差异说明了什么。提示:落点在"指令微调泛化的是**指令跟随这个元能力**,而非某个任务"。

## 9. 源码 / 论文导读

- **GPT-1**:Radford et al., 2018, *Improving Language Understanding by Generative Pre-Training*。重点读"预训练目标(式)+ 下游任务输入改写"那两节,体会"还在用任务专属微调"。
- **GPT-2**:Radford et al., 2019, *Language Models are Unsupervised Multitask Learners*。重点读零样本任务设定与 WebText 数据动机。
- **GPT-3**:Brown et al., 2020, *Language Models are Few-Shot Learners*。重点读 few-shot/one-shot/zero-shot 的形式化定义与随规模的趋势图。
- **T5**:Raffel et al., 2019/2020, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*。**必读**:text-to-text 框架定义、span corruption 目标、以及那张对比各种破坏目标/破坏率的大消融表(理解"为什么选这套配置")。
- **BART**:Lewis et al., 2019, *BART: Denoising Sequence-to-Sequence Pre-training...*。重点读 5 种噪声变换的定义,以及噪声组合的消融(text infilling + sentence permutation 最优)。
- **FLAN**:Wei et al., 2021, *Finetuned Language Models Are Zero-Shot Learners*。重点读"指令微调如何提升对**未见任务**的零样本"的实验设计。
- **T0**:Sanh et al., 2021, *Multitask Prompted Training Enables Zero-Shot Task Generalization*。重点读 PromptSource 的多模板设计思路。
- **开源实现**:
  - HF `transformers/models/t5/modeling_t5.py`(`T5ForConditionalGeneration` 的 encoder/decoder/cross-attention 接线)与 T5 的数据预处理(span corruption / 哨兵分配),配 Google 的 `t5x` 仓库看原始数据管线。
  - HF `transformers/models/bart/modeling_bart.py`(对照 T5 看 encoder-decoder 结构的差异)。
  - HF `transformers/models/gpt2/modeling_gpt2.py`(看因果掩码 `causal_mask` 如何在注意力里实现单向)。

## 10. 小结与承上启下

这一章把生成式与编码-解码两条预训练路线讲透了:

- **GPT 系**用因果掩码 + 自回归目标 `∏ P(x_t|x_<t)`,把"理解语言"压缩成"预测下一个词";每个位置都产生密集监督且可并行,随规模放大涌现出零样本/少样本能力,把"任务"从"改结构"退化成"改输入文本"。
- **T5** 提出 text-to-text 统一接口(任务=前缀+文本),预训练用 span corruption(破坏 15%、合并成 span、用哨兵、只生成被破坏片段),让多任务训练和新增任务零成本——这是接口统一思想的源头。
- **BART** 用多种噪声破坏 + 重建整句的去噪自编码器,encoder 双向 + decoder 自回归,兼顾理解与生成,摘要/翻译尤强,mBART 延伸到多语言。
- **指令微调(FLAN/T0)** 用多任务、多模板的指令数据微调,让模型学会"按指令做任意任务"的**元能力**并泛化到未见任务——这是从"会但不听话"到 ChatGPT 的关键一跳,而最后的"有用/诚实/无害"由 RLHF/DPO 补上(详见强化学习课)。
- **架构选型**:判别/检索用 encoder-only,通用生成/对话用 decoder-only(当前主流),固定方向的转换任务用 encoder-decoder。

**承上**:本章的因果掩码、密集监督与 BERT 的双向 MLM(06)正好成对,三种范式的横向梯度/效率对比在算法课 06。**启下**:下一章(08,子词切分与多语言)会补上一个被我们一路默认的前提——这些模型吃的"token"到底怎么来的、跨语言迁移在工程上怎么做;再往后,LLM 时代经典 NLP 任务如何被 prompt 统一(09)、检索增强如何给生成式模型接上外部知识(10),都建立在本章"生成式 + 统一接口"的地基上。
