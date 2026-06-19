---
title: "Capstone:做一个 RAG 问答系统"
slug: "6-16"
collection: "ai-research-compass"
group: "自然语言处理专家课程"
order: 6016
summary: "这一章把你从\"读懂了 RAG 的每个零件\"带到\"亲手把检索 + 生成 + 评测串成一个能跑、能引用、能评测、能迭代的问答系统\"。前面各章讲透了原理(01 表示、06 句向量、10 RAG、13 评测、14 忠实性);这一章只做一件事——把原理落成代码,并用评测逼着它一步步变好。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T05:59:28.000Z"
updatedAt: "2026-06-19T05:59:28.000Z"
---
> 这一章把你从"读懂了 RAG 的每个零件"带到"亲手把检索 + 生成 + 评测串成一个能跑、能引用、能评测、能迭代的问答系统"。前面各章讲透了原理(01 表示、06 句向量、10 RAG、13 评测、14 忠实性);这一章只做一件事——**把原理落成代码,并用评测逼着它一步步变好**。读完你应该能照着搭通一个小 RAG、跑出 Recall@k 与忠实性数字、做"纯稠密 vs 混合 vs 加 reranker"的消融,并写出一份能说服别人的系统报告。如果你更想练经典监督管线,本章末尾给了一条"微调 NER / 文本分类小模型"的备选主线。

---

## 1. 这一章和前面有什么不同:从"懂原理"到"能交付"

前面的章节是**讲透机制**:BM25 每一项为什么长那样、InfoNCE 怎么推、HNSW 为什么对数级。这一章是**收官实战**,目标不是再推一遍公式,而是**把这些零件拼成一个真能跑的系统,并建立"评测驱动迭代"的工程方法论**。所以本章的写法变了:每一节是"我要解决工程上的什么问题 → 给可运行的代码骨架 → 指出这里的权衡与坑",原理细节一律回扣对应章节,不重复推导。

先把整个系统的形状画清楚。一个最小可用的 RAG 问答系统有五个阶段,数据像流水线一样从左到右流过:

```
        ┌─────────── 离线(建库,一次性 / 文档更新时重建)───────────┐
文档库 → ① chunking 切块 → ② 句向量编码(sentence-transformers)→ 建向量索引(FAISS)
                                                                        │
        ┌─────────────────── 在线(每来一个 query)──────────────────────┤
query → ③ 检索 top-k(稠密;可加 BM25 混合 + reranker 精排)──────────────┘
      → ④ 把证据拼进 prompt,让 LLM 基于证据作答(带引用)
      → ⑤ 评测:检索 Recall@k + 答案质量 + 忠实性(答案是否被证据支撑)
```

**离线/在线的切分是 RAG 工程的第一性事实**:`②` 把所有 chunk 编码成向量、建好索引,这一步贵(要过一遍编码模型、建图),但**只做一次**(或文档更新时重建);`③④` 每个 query 都要跑,必须快。第 10 章讲的双塔(bi-encoder)能预计算正是为这条切分服务的——passage 向量离线算好存进索引,在线只编码 query 一次。把哪步放离线、哪步放在线,直接决定系统能不能上线。

**贯穿全章的方法论铁律,先立在这里**:

1. **先建最简基线,再逐步加复杂度。** 第一版只用"稠密检索 + 拼 prompt",跑通、出数字。之后每加一个零件(混合检索、reranker、查询改写),都要**和基线比、用评测证明它真带来增益**。不做对照就加功能,是 RAG 工程最常见的自欺。
2. **评测先行,而不是事后补。** 在写检索之前就要准备好评测集(一批 query + 已知相关文档 + 参考答案)。没有评测,你对系统好坏的判断全是错觉。
3. **分层定位,别只看最终答案。** 答案错了,先问是检索没召回(检索的锅)还是召回了没用好(生成的锅)。Recall@k 是答案质量的天花板——证据没进 top-k,生成再强也无米下炊。

这三条会在后面每一节反复兑现。

---

## 2. 阶段 ①:文档分块(chunking)——最被低估、最影响效果的一步

### 问题:为什么不能整篇文档丢进检索?

两个硬约束。其一,**一个向量的语义容量有限**:把一整篇几千字、跨多个话题的文档压成一个 768 维向量,细节全糊掉,检索时既不精准、命中了也夹带大量无关内容。其二,**LLM 上下文窗口有限**:整篇拼进 prompt 要么超窗口,要么挤掉别的证据。所以必须把文档切成 **chunk(块)**——一个语义相对完整、大小适中的片段,作为检索和编码的基本单位。

chunk 的大小与重叠是一组**核心权衡**,没有万能值,但权衡方向是确定的:

```
切太大(如 2000 token):  一个 chunk 混多个话题 → 向量语义被稀释 → 检索准确率↓
                          命中了也带入大量无关文本 → 浪费上下文窗口、稀释相关信号
切太小(如 64 token):    答案被切碎到跨多个 chunk → 任一 chunk 都不完整 → 召回不全
                          chunk 数量暴增 → 索引变大、检索变慢
重叠(overlap):          相邻 chunk 留一段重叠(常 10~20%)→ 防答案恰好被切在边界两边都不完整
按结构切 > 按字数硬切:    沿段落/标题/句子边界切,尊重文档自然语义单元
                          硬按固定 token 数切常把一句话拦腰斩断,语义破碎
```

**为什么"按结构切"更好的直觉**:文档的段落、标题、列表项本身就是作者划分好的语义单元;沿这些边界切,每个 chunk 内部话题集中、向量语义干净。硬按 token 数切会无视这些边界,把"问题的答案"切在一句话中间,两个 chunk 谁都不完整。代价是按结构切要解析文档格式(Markdown 标题、HTML 标签、PDF 段落),工程更重;字数切实现简单但质量差。**实践折中:先按结构(段落/标题)粗切,再对超长块按 token 数 + 重叠二次切。**

### 代码:一个带重叠、尊重句边界的 chunker

```python
import re
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    doc_id: str          # 来源文档(用于引用归因)
    chunk_id: int        # 块在文档内的序号
    # 可继续挂元数据:URL、标题、页码、字符 offset

def sentence_split(text: str) -> list[str]:
    """极简句切分:中英文句末标点。生产用 spaCy / nltk 更稳。"""
    parts = re.split(r'(?<=[。!?.!?])\s*', text)
    return [p for p in parts if p.strip()]

def chunk_document(text: str, doc_id: str,
                   target_tokens=256, overlap_tokens=40,
                   tok_len=lambda s: len(s)//2) -> list[Chunk]:
    """
    按句子聚合到目标长度,块间留重叠。
    target_tokens: 每块目标 token 数;overlap_tokens: 相邻块重叠量。
    tok_len: 估算 token 数的函数(此处用 字符数//2 粗估;生产应用真实 tokenizer)。
    """
    sents = sentence_split(text)
    chunks, cur, cur_len, cid = [], [], 0, 0
    i = 0
    while i < len(sents):
        s = sents[i]
        cur.append(s); cur_len += tok_len(s); i += 1
        if cur_len >= target_tokens:
            chunks.append(Chunk(" ".join(cur), doc_id, cid)); cid += 1
            # 回退若干句作为下一块的重叠前缀
            back, blen = [], 0
            for s_prev in reversed(cur):
                if blen >= overlap_tokens: break
                back.insert(0, s_prev); blen += tok_len(s_prev)
            cur, cur_len = back, blen
    if cur:                                   # 收尾残块
        chunks.append(Chunk(" ".join(cur), doc_id, cid))
    return chunks
```

**坑**:`//2` 这种粗估对中英文都不准——中文约 1 token/字符会被系统性低估,英文一个 token 常含 3~4 字符会被高估;无论中英文上线前都应换成检索/生成模型真实的 tokenizer 量长度。**chunking 几乎一定要进入第 8 节的消融**——它是后面"评测驱动迭代"最该先调的旋钮。

---

## 3. 阶段 ②:编码 + 建索引——把 chunk 变成可检索的向量

### 问题:用什么编码、怎么建索引

编码用**句向量模型**(第 06 章的 Sentence-BERT 一脉,双塔结构、可预计算)。直接用 `sentence-transformers` 的现成模型,**不要一上来自己训**——先用通用预训练句向量建基线,确认整条管线跑通、出了数字,再判断要不要针对领域微调(这正是"先简单基线再加复杂度"的体现)。

索引用 **FAISS**。小规模(几千~几十万 chunk)直接用精确内积索引(`IndexFlatIP`)就够了——精确、零近似误差、实现最简;只有到了百万级以上、精确 KNN 的 `O(N·d)` 扫描扛不住时,才换 HNSW 这类近似最近邻(ANN)索引(原理见第 10 章第 5 节)。**Capstone 默认从精确索引起步**,这是"基线优先"在索引选型上的体现:先别为还没遇到的规模问题付近似误差的代价。

一个**关键且高频的 bug**:要用**余弦相似度**就必须先把向量 **L2 归一化**,再用内积索引——归一化后的内积等于余弦。直接拿未归一化向量算内积当余弦,是经典错误,会让"长向量"莫名其妙地总排前面。

### 代码:编码 chunk 并建 FAISS 索引

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class DenseIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):  # 模型名待核
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)   # 精确内积;归一化后 = 余弦
        self.chunks: list[Chunk] = []              # 行号 i ↔ 第 i 个 chunk(对齐!)

    def build(self, chunks: list[Chunk], batch_size=64):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vecs = self.model.encode(texts, batch_size=batch_size,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True)   # ← 关键:L2 归一化
        self.index.add(vecs.astype('float32'))
        # 规模上来后改:faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)

    def search(self, query: str, k=5):
        q = self.model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True).astype('float32')
        scores, ids = self.index.search(q, k)      # ids: [[i1, i2, ...]]
        return [(self.chunks[i], float(s))         # 取回 chunk 本体 + 相似度
                for i, s in zip(ids[0], scores[0]) if i != -1]
```

**实现要点**:索引里存的是向量,**chunk 本体单独存一个 `list`,靠行号对齐**——FAISS 只还你向量的整数 id,你得拿 id 回查原文(以及 `doc_id` 等元数据,引用归因要用)。这个 id↔chunk 的对齐一旦错位,整个系统会返回"分数对、内容张冠李戴"的诡异结果,且很难 debug。**离线 `build` 后应把 `index` 和 `chunks` 一起持久化**(`faiss.write_index` + chunks 序列化),在线服务直接加载,不要每次启动重编码。

---

## 4. 阶段 ③:检索——从稠密基线到混合 + 重排

这一节是"基线优先 + 逐步加复杂度"方法论最集中的体现。我们分三档,**每升一档都要在第 8 节的评测里证明它真带来增益**。

### 4.1 第一档(基线):纯稠密检索

就是 §3 的 `DenseIndex.search`。**先把它当唯一检索器,跑通整条管线、出 Recall@k 和答案数字。** 这是你的对照基准——后面任何复杂化都要打败它才有意义。

### 4.2 第二档:稀疏 × 稠密混合检索

稠密强在语义(跨越"心脏病发作"↔"心肌梗死"的词汇鸿沟),但**对精确实体、罕见词、错误码、代码标识符反而常不如 BM25**——这些东西压进向量容易糊掉,而 BM25 的词面精确匹配天然无敌(为什么如此,见第 10 章 §2~§3)。二者强弱区互补,所以**两路都跑、再融合**。

融合用 **RRF(Reciprocal Rank Fusion,倒数排名融合)**:它**只用排名、不用原始分数**,天然规避了"BM25 分数无上界、向量内积量纲完全不同、不可直接相加"的麻烦,且无需调权重,是工业 RAG 的默认融合法之一。文档 d 在第 r 路检索里排名 `rank_r(d)`(从 1 起),融合分:

```
            Σ            1
RRF(d) =   r∈检索器们  ────────────
                       K + rank_r(d)
```

`K` 是平滑常数(原论文用 60):某文档只要在任一路里排得高就拿高分,两路都高则叠加。直觉:`K` 越大,排名靠前的优势越被"压平"(头部和中部差距变小),融合越平均。

```python
from collections import defaultdict

def rrf_fuse(rankings: list[list[str]], K=60, top_k=5) -> list[str]:
    """
    rankings: 多路检索结果,每路是按相关性降序的 chunk 唯一键列表
              (chunk 唯一键 = f"{doc_id}#{chunk_id}",别用文本本身当键)
    返回融合后的 top_k 唯一键。
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):   # rank 从 1 开始
            fused[key] += 1.0 / (K + rank)
    return sorted(fused, key=lambda kkey: -fused[kkey])[:top_k]

# 用法:dense_keys / bm25_keys 各是一路 top-N 的 chunk 唯一键列表
hybrid_top5 = rrf_fuse([dense_keys, bm25_keys], K=60, top_k=5)
```

BM25 一路用第 10 章 §2.3 那个从零实现的 `BM25`,或直接用 `rank_bm25` 库。**注意两路要先各自取较大的 top-N(如各 50)再融合**,只取各自 top-5 再融合会丢掉"被一路埋没但另一路救得回来"的文档。

### 4.3 第三档:加 cross-encoder reranker 精排

混合检索给出的仍是**粗排**——bi-encoder/BM25 在打分时 query 和文档"互不见面"。**cross-encoder** 把 query 和文档**拼起来一起过模型**,每层互相注意,精度高得多,但慢、不能预计算,所以只能用在**小候选集**上。这就是经典的**召回-精排两阶段**:

```
阶段一(召回,要快要广):  混合检索从全库捞 top-50            # 快,排序粗
阶段二(精排,要准量小):  cross-encoder 给这 50 个逐对精打分,重排出 top-5  # 慢,但只算 50 次
```

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, name="cross-encoder/ms-marco-MiniLM-L-6-v2"):  # 模型名待核
        self.model = CrossEncoder(name)

    def rerank(self, query: str, candidates: list[Chunk], top_k=5) -> list[Chunk]:
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)          # 逐 (query, doc) 对打分
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [candidates[i] for i in order[:top_k]]
```

reranker 常是工业 RAG **性价比最高**的一招:只对几十个候选精排,代价可控,却把高精度用在最关键的最终排序上。但它**不是免费**——多一次模型前向,在线延迟增加;且**它救不了召回阶段就漏掉的文档**(阶段一没捞进 top-50 的,阶段二再准也排不出来)。**所以顺序永远是:先把召回 Recall@50 拉高,再上 reranker 提精排;反过来无意义。**

### 4.4 把三档统一成一个可切换的检索器

为了第 8 节做消融,把三档做成一个**配置可切换**的检索器——消融实验的本质就是"只改一个开关,其余不变,看指标怎么动"。

```python
class Retriever:
    def __init__(self, dense: DenseIndex, bm25=None, reranker: Reranker=None):
        self.dense, self.bm25, self.reranker = dense, bm25, reranker

    def retrieve(self, query, k=5, use_hybrid=False, use_rerank=False, recall_n=50):
        n = recall_n if (use_hybrid or use_rerank) else k     # 要融合/精排就多召回
        dense_hits = self.dense.search(query, k=n)            # [(chunk, score)]
        pool = {chunk_key(c): c for c, _ in dense_hits}       # 候选池(去重)

        if use_hybrid and self.bm25 is not None:
            bm25_hits = self.bm25.search(query, k=n)          # 返回 [(chunk, score)]
            for c, _ in bm25_hits: pool[chunk_key(c)] = c
            dense_keys = [chunk_key(c) for c, _ in dense_hits]
            bm25_keys  = [chunk_key(c) for c, _ in bm25_hits]
            ordered_keys = rrf_fuse([dense_keys, bm25_keys], top_k=n)
            cands = [pool[k_] for k_ in ordered_keys]
        else:
            cands = [c for c, _ in dense_hits]

        if use_rerank and self.reranker is not None:
            return self.reranker.rerank(query, cands, top_k=k)
        return cands[:k]

def chunk_key(c: Chunk) -> str:
    return f"{c.doc_id}#{c.chunk_id}"
```

---

## 5. 阶段 ④:基于证据的生成(带引用)

### 问题:检索拿到 top-k chunk,怎么让 LLM"基于证据"作答并标引用

朴素 RAG 的做法(也是今天绝大多数生产系统的形态):把 top-k chunk **编号后拼进 prompt**,连同问题一起交给指令微调过的 LLM,要求它**只用给定证据作答、不知道就说不知道、并标注引用编号**。生成器内部的注意力/解码细节属于算法课,这里只关心"信息怎么进 prompt、答案怎么带回引用"这一层。

**Prompt 的三个要件,缺一不可**:

1. **角色与约束**:明确"只根据资料回答,资料没有就说不知道"。这是压制"检索了仍幻觉"的第一道闸——给模型一个明确的拒答出口,它才不至于硬编。
2. **编号证据**:每个 chunk 前加 `[1] [2]` 编号,且编号要能映射回 `doc_id`(引用归因要用)。
3. **引用要求**:要求答案里对每个论断标注来源编号 `[i]`,便于溯源和后续忠实性校验。

```python
SYSTEM = (
    "你是严谨的问答助手。只能依据【资料】回答;资料中没有的信息,"
    "明确回答'根据现有资料无法确定',绝不编造。每个论断后用 [编号] 标注其来源。"
)

def build_prompt(question: str, chunks: list[Chunk]) -> tuple[str, dict]:
    lines, id2doc = [], {}
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c.text}")
        id2doc[i] = chunk_key(c)               # 引用编号 → 来源 chunk,供归因校验
    context = "\n".join(lines)
    user = f"【资料】\n{context}\n\n【问题】{question}\n\n【回答(含引用编号)】"
    return user, id2doc

def answer(question, retriever, llm, **retrieval_kwargs):
    chunks = retriever.retrieve(question, **retrieval_kwargs)
    user, id2doc = build_prompt(question, chunks)
    resp = llm.chat(system=SYSTEM, user=user)  # 任意 LLM API/本地模型,占位
    return resp, chunks, id2doc                # 把证据与编号映射一并返回,评测要用
```

**为什么要把 `chunks` 和 `id2doc` 一起返回**:评测(尤其忠实性)需要"答案 + 它实际用到的证据"配对。把生成和它的证据绑定返回,是让系统**可评测**的工程前提——否则你事后无从判断答案到底有没有据。

**坑(贯穿第 10 章 §7.3 与第 14 章)**:模型即使被要求引用,也可能**乱标**——标了 `[2]` 但那句话其实来自参数记忆,`[2]` 根本没这信息。**引用的存在 ≠ 引用的正确**。这只能靠下一节的忠实性评测来抓,不能靠"看到答案里有方括号"就放心。另一个真实风险是**提示注入**:检索内容来自外部文档,可能被植入"忽略以上指令"之类的恶意文本劫持生成(详见第 14 章)——把外部证据和系统指令在 prompt 里清晰分区、对证据内容做必要过滤,是基本防线。

---

## 6. 阶段 ⑤:评测——检索 / 答案 / 忠实性三维,缺一不可

### 6.1 为什么必须分层评测

RAG 评测最大的陷阱是**只看最终答案对不对**。这会让你彻底无法定位问题:答案错了,是检索没召回证据(检索的锅),还是召回了但模型没用好(生成的锅)?二者的修法完全相反——前者要去调 chunking/混合/reranker,后者要去调 prompt/换生成模型。**不分层,你连往哪个方向使劲都不知道。** 所以三个轴必须分开测(原理见第 13、14 章,这里讲怎么把它们算出来):

```
检索质量    Recall@k / MRR   —— RAG 质量的天花板:证据没进 top-k,生成再强也白搭
答案质量    EM/F1 或 LLM-judge —— 答案本身对不对、好不好
忠实性      答案是否被检索证据支撑 —— RAG 命门,抓"检索了仍幻觉"和"蒙对"
```

### 6.2 检索质量:Recall@k 与 MRR

需要一份评测集:每个 query 标注了**已知相关的 chunk 集合**(gold)。

```
                # (相关 chunk ∩ top-k 检索结果)
Recall@k(q) =   ─────────────────────────────────
                       # 相关 chunk
```

对所有 query 取平均。**Recall@k 是答案质量的天花板**——这是 RAG 工程最该刻在脑子里的一句话:调 RAG 第一件事永远是把 Recall@k 拉上去,因为证据根本没进 top-k 时,后面再优化生成都是徒劳。

**MRR(Mean Reciprocal Rank,平均倒数排名)**不只看"在不在 top-k",还看"排多前"。第一个相关 chunk 出现在排名 `rankᵢ`:

```
        1   Σ      1
MRR =  ──        ──────        相关 chunk 排第1→贡献1;第2→0.5;第5→0.2
        Q   i     rankᵢ
```

MRR 高说明相关 chunk 不仅被召回、还排得靠前——这对 RAG 重要,因为靠前的 chunk 更可能被模型真正读进去,且在上下文里位置更优(算法课讲的 "lost in the middle":过长上下文里中间位置的信息易被忽略)。

### 6.3 忠实性:RAG 的命门(为什么必须单独测)

忠实性(faithfulness / groundedness)问的是:**答案是否真的由检索到的证据所支撑?** 它和"答案正确"是**两个不同的轴**,组合成一个关键的四象限:

```
                 答案正确              答案错误
证据支撑      ✅ 理想(对且有据)      证据本身就错/过时(检索的锅)
证据不支撑    ⚠ 蒙对(危险!)         ❌ 幻觉(检索了仍编)
```

**"蒙对"那一格特别危险**:答案碰巧对,但模型其实没看证据、是按参数记忆答的。这种系统在你部署到新领域、参数记忆失效时会**突然崩**,而你之前只看"答案正确率"的评测**完全看不出这个隐患**。这正是第 10、14 章反复强调的"检索了但仍幻觉"的量化体现——证据明明放进了上下文,生成器却无视它、按旧记忆作答。**只有把"答案正确"和"忠实"两轴同时测,才能把"蒙对"和"真有据"区分开。**

怎么度量忠实性,两条主流路线:

- **NLI / 蕴含检验**:把"答案的每个陈述"作为假设(hypothesis)、"检索到的证据"作为前提(premise),送进一个自然语言推理(NLI)模型,判断证据是否**蕴含(entail)**该陈述。有陈述不被任何证据蕴含 → 该陈述未被支撑(疑似幻觉)。
- **LLM-as-judge**:给一个强模型"问题 + 证据 + 答案",让它按 rubric 判断答案是否完全由证据支撑、有无证据外的断言(RAGAS 等框架把这类思路工程化)。要警惕评判模型自身的偏置与不稳定(详见第 13 章)。

```python
# 忠实性的 NLI 路线骨架:逐句核对答案陈述是否被证据蕴含
def split_claims(answer: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[。!?.!?])\s*', answer) if s.strip()]

def faithfulness_nli(answer: str, evidence: str, nli) -> float:
    """
    nli(premise, hypothesis) -> {'entail':p, 'neutral':p, 'contradict':p}
    返回:被证据蕴含的句子占比(越高越忠实)。
    """
    claims = split_claims(answer)
    if not claims: return 1.0
    supported = sum(1 for c in claims
                    if nli(premise=evidence, hypothesis=c)['entail'] > 0.5)
    return supported / len(claims)
```

**坑**:把 k 个 chunk 直接拼成一个长 premise 送 NLI,长前提会稀释判断、且多数 NLI 模型有长度上限;更稳的做法是**对每个 claim 在各 chunk 上分别判蕴含,取最大**(只要有一篇证据蕴含它就算被支撑)。

### 6.4 端到端评测骨架

```python
def evaluate_rag(eval_set, retriever, llm, nli, k=5, **retrieval_kwargs):
    """eval_set: List[{query, gold_chunk_keys: set, gold_answer}]"""
    R, MRR, ACC, FAITH = [], [], [], []
    for ex in eval_set:
        chunks = retriever.retrieve(ex['query'], k=k, **retrieval_kwargs)
        keys = [chunk_key(c) for c in chunks]

        # ① 检索 Recall@k
        gold = ex['gold_chunk_keys']
        hit = set(keys) & gold
        R.append(len(hit) / max(len(gold), 1))
        # ② 检索 MRR(第一个相关 chunk 的倒数排名)
        rank = next((i+1 for i, kk in enumerate(keys) if kk in gold), None)
        MRR.append(1.0/rank if rank else 0.0)

        # 生成
        user, _ = build_prompt(ex['query'], chunks)
        ans = llm.chat(system=SYSTEM, user=user)

        # ③ 答案质量(此处用简化匹配;开放式答案改用 LLM-judge)
        ACC.append(answer_match(ans, ex['gold_answer']))
        # ④ 忠实性(答案是否被检索证据蕴含)
        evidence = "\n".join(c.text for c in chunks)
        FAITH.append(faithfulness_nli(ans, evidence, nli))

    return {f"Recall@{k}": mean(R), "MRR": mean(MRR),
            "AnswerAcc": mean(ACC), "Faithful": mean(FAITH)}
```

**读数指南(决定下一步往哪修)**:

```
Recall@k 低           → 先修检索:调 chunking → 开混合 → 加 reranker。此时修生成无意义
Recall@k 高、Acc 低    → 修生成:改 prompt、换更强 LLM、检查证据是否拼对
Acc 高、Faithful 低    → 危险信号:系统在"蒙对",换领域必崩 → 强化"只依据证据"约束、收紧拒答
```

这张表就是"评测驱动迭代"的操作手册——**指标告诉你下一步该动哪个零件,而不是凭感觉乱试**。

---

## 7. 防"检索到了但仍幻觉":单独拎出来讲

这是 RAG 最隐蔽、也最该重点防的失败模式:**正确证据已经被检索进上下文,生成器却仍然给出无据/错误的答案**。它为什么发生、怎么防,串起了前面几乎所有零件:

**成因(至少四类)**:

1. **模型偏信参数记忆**,尤其当证据与模型"成见"冲突时——它宁可信自己背过的旧知识,也不信眼前的新证据。
2. **证据被淹没在过长上下文中间**(lost in the middle):top-k 太大、无关 chunk 太多,真正有用的那篇被挤到中段被忽略。
3. **chunk 切碎使证据不完整**:答案需要的信息被 chunking 切到两个 chunk,单个 chunk 看起来都"半截",模型据此脑补。
4. **被要求引用却乱标**:见 §5,引用存在不等于引用正确。

**防御手段(分别对应上面成因)**:

- 对 (1):prompt 里强约束"只依据资料、冲突时以资料为准、资料没有就拒答";严重场景可加**归因校验**——生成后用 §6.3 的 NLI 逐句核对,未被证据蕴含的句子标红或触发重答。
- 对 (2):**控制 top-k**,别贪多;上 reranker 把最相关证据顶到最前(位置更优);必要时把最相关 chunk 放在 prompt 末尾(贴近问题)。
- 对 (3):回到 chunking,增大块或增大重叠,用评测验证。
- 对 (4):把忠实性纳入常规评测,别只看答案正确率。

**核心认知**:防幻觉不是单点修一个地方,而是 **chunking → 检索/重排 → prompt 约束 → 忠实性评测** 一整条链路的协同。`Acc 高但 Faithful 低` 是它的体检指标——一旦出现,优先级最高。

---

## 8. 把方法论跑起来:消融实验(纯稠密 vs 混合 vs +reranker)

前面反复说"加复杂度要用评测证明增益"。这一节给出**怎么证明**——做**消融(ablation)**:固定其余一切,只改一个开关,看指标怎么动。我们已把检索器做成了配置可切换(§4.4),消融就是遍历这几种配置跑同一份评测集:

```python
configs = {
    "A. 纯稠密(基线)": dict(use_hybrid=False, use_rerank=False),
    "B. 混合检索":      dict(use_hybrid=True,  use_rerank=False),
    "C. 混合 + 重排":   dict(use_hybrid=True,  use_rerank=True),
}
for name, cfg in configs.items():
    m = evaluate_rag(eval_set, retriever, llm, nli, k=5, **cfg)
    print(f"{name:16s}  R@5={m['Recall@5']:.3f}  MRR={m['MRR']:.3f}  "
          f"Acc={m['AnswerAcc']:.3f}  Faith={m['Faithful']:.3f}")
```

**怎么读这张消融表**(典型预期方向,具体数值随数据而变):

- **A→B(加混合)**:在含罕见实体/错误码/精确术语的 query 上,Recall@k 应明显上升(BM25 补了稠密的词面短板);在纯同义改写的 query 上提升有限。**如果混合没带来提升甚至下降**,要查:BM25 一路是否实现/分词正确、RRF 融合前两路是否各取了足够大的 top-N。
- **B→C(加重排)**:MRR 通常涨得比 Recall@k 多——reranker 主要在**已召回的候选里把相关项往前提**,改善的是排序而非召回。若 Recall@k 本来就低,C 也救不回来(reranker 救不了召回阶段漏掉的)。
- **盯住 Faithful**:别只看 Acc 涨了就高兴。如果某配置 Acc 高但 Faithful 没跟上,说明增益里混了"蒙对"。

**这套消融就是 Capstone 的灵魂**:它把"我觉得混合检索更好"变成"在这份评测集上,混合把 Recall@5 从 X 提到 Y、reranker 把 MRR 从 P 提到 Q,而 Faithful 没有退化"——**可复现、可争论、可决策**。脱离评测谈 RAG 优化,都是玄学。

---

## 9. 备选主线:微调一个 NER / 文本分类小模型

如果你更想练**经典监督管线**(而非检索 + 生成),这条线同样能收官——它把第 04 章(序列标注、CRF)、第 07 章(BERT 等预训练模型微调)落成一个完整的"数据 → 微调 → 评估"流程。两条主线择一即可。

**任务设定**:用预训练 BERT 微调做一个序列标注 NER(命名实体识别)或句子级文本分类。NER 的标注用 **BIO 方案**(B-PER 实体开头、I-PER 实体内部、O 非实体);分类则每句一个标签。

**管线四步**:

1. **数据**:NER 用 CoNLL-2003 / 中文 MSRA 这类公开标注集(格式待核),或自标小数据;分类用任意带标签语料。注意 **token 与 label 对齐**——这是 NER 微调最大的坑(见下)。
2. **微调**:BERT + 一个线性分类头(token 级 → 每个 token 输出标签分布;句级 → 取 `[CLS]` 输出句标签)。是否在 BERT 上加 **CRF 层**取决于标签间约束强不强:NER 的标签转移有硬约束(`I-PER` 不能紧跟 `O`、不能跨实体类型),CRF 能在解码时全局排除非法序列,通常比逐位置独立 softmax 更好(为什么严格更优,见第 04 章)。
3. **评估**:NER 用**实体级 F1**(不是 token 级!——必须整个实体边界和类型都对才算命中,token 级会虚高);分类用 accuracy / macro-F1。
4. **错误分析**:看混淆矩阵、看边界错误(实体多切一字少切一字)、看类别不均衡。

```python
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer)

model_name = "bert-base-cased"                       # 中文用 bert-base-chinese
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)  # BIO 标签数

def tokenize_and_align(example):
    """子词切分把一个词拆成多片,label 必须随之对齐——NER 头号坑。"""
    enc = tok(example["tokens"], is_split_into_words=True, truncation=True)
    word_ids = enc.word_ids()                         # 每个子词对应原词的下标
    labels, prev = [], None
    for wid in word_ids:
        if wid is None:                               # 特殊符 [CLS]/[SEP]
            labels.append(-100)                       # -100 = 交叉熵忽略
        elif wid != prev:                             # 词的第一个子词:用原 label
            labels.append(example["ner_tags"][wid])
        else:                                         # 同词后续子词:忽略(或用 I-)
            labels.append(-100)
        prev = wid
    enc["labels"] = labels
    return enc

# Trainer 配置略;评估务必用 seqeval 算实体级 F1,而非 token 级 accuracy
```

**这条线的核心坑**:① **子词对齐**——BERT 的 tokenizer 把一个词切成多个子词,原始 word-level 标签必须重新对齐到子词级(上面代码做的事),对齐错了 F1 会莫名很低且查不出。② **`-100` 忽略**——特殊符和子词续片的位置要置 `-100`,让交叉熵跳过它们。③ **评估粒度**——NER 一定用**实体级 F1**(seqeval),token 级 accuracy 因为 `O` 占绝大多数会虚高得离谱。

无论走 RAG 主线还是这条监督主线,**方法论是一致的**:先建可跑的基线,用对的指标(RAG 看 Recall@k/忠实性,NER 看实体级 F1)量化,再做消融/错误分析逐步改进。

---

## 10. 设计权衡与常见坑(Capstone 总清单)

把全章踩点汇总成一张可对照的清单:

- **先建简单基线,别一上来堆全套。** 纯稠密 + 拼 prompt 先跑通出数字,后续每个零件(混合/重排/查询改写)都要用消融证明增益。不做对照的优化是自欺。
- **chunking 是头号隐形变量。** 检索效果差,十有八九先查 chunk 切法,而不是急着换更大的 embedding 模型。务必让它进消融。
- **chunk 大小与重叠是权衡,不是越大/越小越好。** 大→语义稀释、占窗口;小→证据切碎、召回不全。按结构切优于按字数硬切。
- **embedding/生成模型的领域匹配 > 模型大小。** 通用句向量放专业领域(法律/医疗/代码)可能水土不服,优先选领域贴近的,必要时微调,别盲目堆参数。
- **要余弦必先 L2 归一化再用内积。** 未归一化向量算内积当余弦是经典 bug,会让长向量莫名总排前。
- **id ↔ chunk 对齐别错位。** FAISS 只还整数 id,chunk 本体与元数据要靠 id 严格对齐,错位会产生"分数对、内容张冠李戴"的诡异 bug。
- **top-k 不是越大越好。** k 太大→无关文档涌入,既稀释相关信号(lost in the middle)又烧 token、增延迟。k 按 Recall@k 曲线和上下文预算一起定。
- **reranker 救不了召回漏掉的。** 顺序永远是先拉高召回(Recall@N),再上 reranker 提精排;反过来无意义。
- **引用 ≠ 忠实。** 模型会乱标引用,务必单独测忠实性,别被答案里的方括号骗了。
- **`Acc 高但 Faithful 低` 是最危险的体检结果。** 系统在"蒙对",换领域必崩。优先级最高,立刻收紧"只依据证据"约束。
- **离线/在线切分要清楚。** 编码 + 建索引是离线一次性(或文档更新时)的重活,在线只编码 query + 检索 + 生成。别在线重编码全库。
- **RAG 补知识不补推理。** 多跳逻辑/计算光堆检索文档常无效,需多步检索、查询改写或工具调用(Agent 方向)。
- **提示注入是真实威胁。** 外部检索内容可能含恶意指令劫持生成,证据与系统指令要分区、必要时过滤(详见第 14 章)。

---

## 11. 动手练习

1. **照着搭通一个最小 RAG(必做)。** 取一小批文档(如 20~50 篇你熟悉的技术博客/手册),按 §2~§5 跑通:chunking → `all-MiniLM` 编码 → `IndexFlatIP` → 纯稠密检索 top-5 → 拼 prompt 让任意 LLM 带引用作答。**先不追求好,只追求端到端能跑、能返回带 `[i]` 引用的答案。** *提示:先用精确索引别上 HNSW;chunk 先用 256 token + 40 重叠;手动准备 10 条 query 并标出每条的相关 chunk,这就是你下一题要用的评测集。*

2. **做检索消融:纯稠密 vs 混合 vs +reranker。** 用第 1 题的评测集,按 §8 跑 A/B/C 三档,报告各档的 Recall@5、MRR、AnswerAcc、Faithful。*提示:刻意在评测集里放几条"含罕见实体/错误码/精确术语"的 query 和几条"纯同义改写"的 query,观察混合检索是否在前者上提升更大、在后者上提升有限——这能让你亲眼看到 BM25 与稠密的互补。务必先报告纯稠密基线。*

3. **抓"检索到了仍幻觉"。** 构造 3~5 条"证据正确、但与模型常识/先验冲突"的 query(比如把文档里某个数字/结论改成反直觉的值),看 RAG 答案是站证据还是站先验;再用 §6.3 的 NLI 蕴含检验逐句核对答案与证据,统计 Faithful。*提示:这道题专门暴露"蒙对"——一个 Acc 看着还行的系统,可能 Faithful 很低。把你抓到的案例写进报告。*

4. **(进阶)加一个查询改写步骤并验证它值不值。** 在检索前用 LLM 把口语化/有歧义的 query 改写成更利于检索的形式(或扩展成多个子查询分别检索再合并),作为第四档 D 加入消融,看 Recall@k 是否提升、是否引入噪声。*提示:查询改写不一定总赢——改写可能引入幻觉词或漂移原意。用评测决定它去留,这正是"评测驱动迭代"的精神。*

---

## 12. "RAG 系统报告"模板

交付一个 RAG 系统,光给代码不够,要给一份**让别人能判断它好在哪、差在哪、为什么这么选**的报告。下面是模板,照填即可:

```
# RAG 系统报告:<系统名 / 领域>

## 1. 任务与数据
- 回答什么领域的什么问题;文档来源、规模(文档数 / chunk 数)。
- 评测集:多少条 query,如何标注相关 chunk 与参考答案,标注者/标注规则。

## 2. 系统配置(每项写清"选了什么 + 为什么")
- Chunking:策略(结构/字数)、chunk 大小、重叠;为何这么定。
- 编码模型:哪个句向量模型,为何(领域匹配?速度?)。
- 索引:FAISS 索引类型(Flat/HNSW),规模与延迟;若 HNSW 写 efSearch 及实测召回。
- 检索:稠密 / 混合(RRF,K=?) / 是否 reranker(哪个模型,召回 N→精排 k)。
- 生成:哪个 LLM,prompt 关键约束(只依据证据 / 拒答 / 引用),top-k。

## 3. 评测结果(核心,必须有消融表)
| 配置            | Recall@5 | MRR | AnswerAcc | Faithful |
|-----------------|----------|-----|-----------|----------|
| A 纯稠密(基线) |          |     |           |          |
| B 混合           |          |     |           |          |
| C 混合+重排      |          |     |           |          |
- 每一档相对基线的增益与代价(延迟、成本)。
- 重点解读:Recall@k 天花板在哪;Acc 与 Faithful 是否背离(有无"蒙对")。

## 4. 失败案例分析(至少 3 例,这部分最有价值)
- 检索失败(证据没进 top-k):为什么?chunking?词汇鸿沟?
- 生成失败(召回了没用好):lost in the middle?证据冲突先验?
- 忠实性失败(蒙对/乱标引用):贴出案例 + NLI 判定。

## 5. 已知局限与下一步
- 当前系统不擅长的 query 类型(多跳推理 / 计算 / 跨文档综合)。
- 下一步迭代项及其预期收益(基于评测,而非拍脑袋)。
```

**这份报告的灵魂在第 3、4 节**:消融表证明"每个复杂度都挣到了它的位置",失败案例证明"你真的理解系统为什么会错"。一份没有消融表、没有失败分析的 RAG 报告,等于没说清这系统到底行不行。

---

## 13. 源码 / 论文导读

- **句向量与双塔 / cross-encoder**:`sentence-transformers`——`SentenceTransformer`(编码 chunk 与 query)、`CrossEncoder`(reranker)。读其 `encode` 的 `normalize_embeddings` 参数实现,印证 §3 的归一化要点。
- **向量索引**:`FAISS`——小规模 `IndexFlatIP`(精确),大规模 `IndexHNSWFlat` / `IndexIVFPQ`。读 `faiss` 的 `write_index/read_index` 做离线持久化。
- **BM25**:`rank_bm25`(快速起步)或直接读第 10 章 §2.3 的从零实现;工程级实现读 Lucene/Elasticsearch 的 `BM25Similarity`。
- **混合检索 / RRF**:RRF 出自 Cormack et al., 2009, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*——读它为何只用排名就能稳健融合异构检索器。
- **RAG 评测框架**:RAGAS 把忠实性、答案相关性、上下文相关性做成可调用指标(具体指标定义对照其文档,本章只讲清思想);NLI 蕴含模型可用 HF 上的 `nli`/`mnli` 微调模型(具体模型名待核)。
- **NER / 分类微调(备选主线)**:HF `transformers` 的 `AutoModelForTokenClassification` / `AutoModelForSequenceClassification`、`Trainer`;评估用 `seqeval` 算实体级 F1。子词对齐对照 HF token-classification 官方示例。
- **回扣的章节**:稠密检索原理与 BM25/DPR/HNSW 推导(第 10 章)、句向量与对比学习(第 06 章)、文本表示(第 01 章)、序列标注与 CRF(第 04 章,备选主线)、BERT 微调(第 07 章)、生成与忠实性评测(第 13、14 章)。

---

## 14. 小结:你现在能做什么

走完这一章,你不只是"读懂了 RAG",而是**能交付一个 RAG 问答系统并讲清它每个设计为什么这么选**:

- **能把五个阶段串成可运行的管线**:chunking(带重叠、尊重句边界)→ sentence-transformers 编码 + FAISS 索引(离线一次性)→ 检索(稠密基线 / RRF 混合 / cross-encoder 重排,可切换)→ 拼证据 prompt 让 LLM 带引用作答 → 三维评测。
- **掌握了 RAG 工程的方法论**:先建简单基线再用消融逐步加复杂度;评测先行、分层定位;Recall@k 是答案质量的天花板;`Acc 高但 Faithful 低` 是"蒙对"的危险信号。
- **能防 RAG 最隐蔽的失败**:"检索到了仍幻觉"不是单点 bug,而是 chunking → 检索/重排 → prompt 约束 → 忠实性评测整条链的协同,忠实性必须单独测。
- **若走备选主线**,你能把"数据 → BERT 微调 → 实体级 F1 评估"这条经典监督管线跑通,并避开子词对齐、`-100` 忽略、评估粒度三个坑。

这是整个 NLP 专家课程的收官:前面每一章给你的零件——词向量、语言模型、RNN/CRF、Seq2Seq、子词、预训练、LLM、RAG、评测、忠实性——在这里第一次被你**亲手拼成一个完整、能跑、能评、能迭代的系统**。从这里出发,无论是接 Agent(多步检索、工具调用)、上多模态检索、还是把这套评测方法论用到你自己的生产系统,你都已经具备了"读懂前沿论文、自己实现并改进 NLP 系统"的专家底座。
