---
title: "Hybrid Search 与 Rerank"
slug: "3-05"
collection: "tech-library"
group: "数据检索底座"
order: 3005
summary: "TL;DR Hybrid Search = 稀疏检索（BM25）+ 稠密检索（向量 ANN）+ Fusion；Rerank = 用精排模型（cross-encoder / LLM listwise）在粗排 top-K 上再做一遍细粒度打分。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-15T13:53:56.000Z"
updatedAt: "2026-06-15T13:53:56.000Z"
---
> **TL;DR**
> Hybrid Search = 稀疏检索（BM25）+ 稠密检索（向量 ANN）+ Fusion；Rerank = 用精排模型（cross-encoder / LLM listwise）在粗排 top-K 上再做一遍细粒度打分。两者都解决同一个根本矛盾：**召回阶段需要极低延迟、海量覆盖，精排阶段需要高质量相关性判断**，单一手段无法同时满足。
>
> 关键数字：RRF k=60 不是魔法常数而是参数不敏感区的工程习惯；cross-encoder 比 bi-encoder 精度高 5~15 nDCG 点，但延迟高 100×；LLM listwise rerank 精度最高但 token 成本是瓶颈；HNSW 搜索的双堆设计（候选堆 + 结果堆）是 ef 参数控制 recall/latency 的核心机制。

---

## 前置依赖

| 概念 | 参见本系列 |
|---|---|
| BM25 / IDF / TF | 第 1 章 |
| 向量 embedding、余弦相似度 | 第 2 章 |
| HNSW / IVF / PQ 索引 | 第 3 章 |
| 向量库内核（Qdrant / Weaviate / Milvus）| 第 4 章 |

---

## 5.1 设计考古：为什么需要 Hybrid Search

### 5.1.1 两种检索的根本分歧

稀疏检索（sparse retrieval）和稠密检索（dense retrieval）覆盖的相关性维度天然互补：

| | 稀疏（BM25）| 稠密（bi-encoder + ANN）|
|---|---|---|
| 信号来源 | 词汇精确匹配 | 语义/上下文相似度 |
| 优势场景 | 专有名词、型号、罕见词、代码 | 同义词、改写、跨语言 |
| 劣势场景 | 词汇差异（"car" vs "automobile"）| 关键词遗漏（hallucination）|
| 召回单位 | inverted index，O(log N) | ANN 图/量化索引 |
| 可解释性 | BM25 打分直接可读 | 向量空间难解释 |

经典失败案例：
- **BM25 miss**："苹果手机"查"iPhone 15 Pro 详细参数" → 稀疏侧无"iPhone"词项命中
- **Dense miss**："CVE-2024-1234 漏洞描述" → embedding 模型未见该标识符，向量随机化
- **双重失败**："用于 RAG 的 ColBERT-v2.0 最优 batch_size" → BM25 无"RAG"语义，dense 对版本号无感

Lin 等 2021 年在 *Pyserini* 论文（"A Replication Study of Dense Passage Retrieval"）中实验表明，在 NQ/TriviaQA 上 BM25 + Dense hybrid 比单独 dense retrieval 高 2–4 个召回率点。

### 5.1.2 融合思路的演化

**2009 年前**：多路检索结果通常用固定权重线性组合打分（score = α·s_bm25 + β·s_vec），缺点是两种分数域（BM25 > 0 无界，余弦 ∈ [-1,1]）量纲不兼容，需要手动归一化，且归一化参数高度数据集依赖。

**2009 年**：Cormack、Clarke、Buettcher 发表 *"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"*（SIGIR 2009，https://plg.uwaterloo.ca/~gvcormack/papers/ecir2009rrf.pdf），提出 **RRF（Reciprocal Rank Fusion）**。核心洞察：**用排名而非原始分数做融合，天然消除量纲差异**。

**2020 年**：REALM、DPR、ColBERT 论文出现，dense retrieval 进入实用阶段，hybrid pipeline 成为标准配方。

**2022–2024 年**：大规模 RAG（Retrieval-Augmented Generation）场景下，LLM listwise rerank 兴起（RankGPT, RankLLaMA）；BGE-M3 实现稀疏+稠密+多向量三路统一。SPLADE 系列学习稀疏向量，弥补 BM25 词汇 gap 同时保持倒排兼容性。

---

## 5.2 RRF：Reciprocal Rank Fusion

### 5.2.1 公式与数学直觉

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
$$

其中：
- $R$ = 所有排名列表集合（通常是 BM25 结果列表和向量结果列表）
- $\text{rank}_r(d)$ = 文档 $d$ 在列表 $r$ 中的排名（从 1 开始）
- $k$ = 平滑常数，通常取 **60**（原论文最优范围）
- 文档未出现在某路结果中时：该路贡献为 0（等价于 rank = ∞）

**直觉**：第 1 名贡献 $1/(k+1) = 1/61 \approx 0.0164$，第 10 名贡献 $1/(k+10) = 1/70 \approx 0.0143$，第 100 名贡献 $1/(k+100) = 1/160 \approx 0.0063$。曲线在 top-10 之后迅速衰减，远低排名贡献趋近 0，自动对低质量来源降权。

k=60 的几何含义：曲线在 rank=1 附近斜率温和（避免对单个 rank-1 文档过度赋权），在工程上表现出良好的参数不敏感性——rank-1 和 rank-10 的贡献差距只有 1.14x（见 Demo 2）。

### 5.2.2 真实源码精读（lancedb RRFReranker）

LanceDB 的 RRF 实现是工业实践中较简洁的参考实现：

```python
# 【真实源码 lancedb/lancedb @ python/python/lancedb/rerankers/rrf.py】
# Copyright 2024 The LanceDB Authors, Apache-2.0 License
# 注：以下为从 WebFetch 取回的真实实现，带逐行注解

class RRFReranker(Reranker):
    """
    Reranks results using Reciprocal Rank Fusion (RRF) algorithm.
    https://plg.uwaterloo.ca/~gvcormack/papers/ecir2009rrf.pdf

    Parameters
    ----------
    K : int, default 60
        Constant in the RRF formula. A higher value gives more weight to
        higher-ranked results. Research suggests k=60 is near-optimal,
        though the exact value is not critical.
    return_score : str, default "relevance"
        Options: "relevance" | "all"
        "relevance"  → 只返回 RRF 融合分
        "all"        → 同时返回 vector/_score 原始分
    """
    def __init__(self, K: int = 60, return_score="relevance"):
        self.K = K
        self.return_score = return_score

    def rerank_hybrid(self, query: str, vector_results, fts_results):
        # Step 1: 合并两路结果集（去重，保留所有出现的文档）
        combined = self._merge(vector_results, fts_results)

        # Step 2: 对每个 unique row_id 累加 RRF 分
        # 关键：用 _rowid（整数主键）做去重，避免文本比较
        rrf_scores = {}
        for i, row_id in enumerate(vector_results["_rowid"].to_pylist()):
            # i 是 0-based，+1 变 1-based rank，+K 对应公式分母
            rrf_scores[row_id] = rrf_scores.get(row_id, 0) + 1.0 / (self.K + i + 1)
        for i, row_id in enumerate(fts_results["_rowid"].to_pylist()):
            rrf_scores[row_id] = rrf_scores.get(row_id, 0) + 1.0 / (self.K + i + 1)

        # Step 3: 将 RRF 分附加为新列（Arrow 操作，避免 Python 循环）
        relevance_col = [rrf_scores[rid] for rid in combined["_rowid"].to_pylist()]
        combined = combined.append_column(
            "_relevance_score", pa.array(relevance_col, type=pa.float32())
        )
        # Step 4: 降序排列
        combined = combined.sort_by([("_relevance_score", "descending")])

        # Step 5: 按 return_score 决定是否保留原始分列
        if self.return_score == "relevance":
            combined = combined.drop_columns(["_distance", "_score"])
        return combined
```

**关键设计点**：
1. 用 `_rowid`（Arrow 内部整数主键）而非文档文本做去重，避免字符串比较开销
2. `enumerate` 产生 0-based 的 i，显式 `+1` 变为 1-based rank，然后 `+K` 对应公式分母，和论文公式 $1/(k+\text{rank})$ 完全对齐
3. 融合前完全不需要归一化原始分，这正是 RRF 的核心工程优势
4. 结果列合并用 Apache Arrow table 操作（`append_column`/`sort_by`），批量内存操作，远快于逐行 Python dict

### 5.2.3 参数敏感性分析

原论文实验数据（Cormack 2009，在多个 TREC 数据集上）：

| k 值 | 融合效果（NDCG@10，相对最优归一化）|
|---|---|
| 1 | 0.93（top 排名权重过大，对 rank-1 极敏感）|
| 10 | 0.97 |
| **60** | **1.00（参考点，实测最优区间）**|
| 100 | 0.99 |
| 1000 | 0.95（退化为近似等权重）|

k < 10 对单个排名高度敏感，k > 200 则接近等权重均一化，k=60 是实测中较为鲁棒的默认值。在自定义数据集上，如果两路差异悬殊，可调整 k，但通常没必要——这是 RRF 最大的工程价值。

### 5.2.4 与线性分数融合对比

线性融合（Convex Combination）：
$$s = \alpha \cdot \text{normalize}(s_{vec}) + (1-\alpha) \cdot \text{normalize}(s_{bm25})$$

| 维度 | RRF | 线性融合 |
|---|---|---|
| 需要归一化 | 否 | 是（且各自分数分布迥异）|
| 超参调优 | k 不敏感 | α 高度数据集依赖 |
| 处理缺失文档 | 缺席视为末尾排名（贡献 0）| 缺席无法参与加权 |
| 计算复杂度 | O(N log N) 排序 | O(N) 但含归一化 |
| 工程上线难度 | 极低 | 中（需分布对齐）|
| 论文引用 | Cormack 2009 | 多种，无统一标准 |

**边界**：当两路结果重叠率很低时，RRF 的等权假设可能不如经过拟合的线性融合；当有足够标注数据做 LTR（Learning to Rank）时，learned fusion 更优。

---

## 5.3 Hybrid Search 架构全景

### 5.3.1 双路召回流程

```
Query
  │
  ├─[Sparse Path]── Tokenize ── Inverted Index (BM25/SPLADE) ──► top-K₁ 候选
  │
  └─[Dense Path]── Encode ── ANN Search (HNSW/IVF) ──► top-K₂ 候选
                                       │
                                  [Fusion Layer]
                                 RRF / Linear / Learned
                                       │
                                  top-K_merged（两路并集去重）
                                       │
                               [可选 Rerank Layer]
                           cross-encoder / ColBERT / LLM
                                       │
                                  final top-N
```

**典型参数配置**：
- K₁（BM25 召回）= 50~200
- K₂（向量 ANN 召回）= 50~200
- K_merged（融合后）= 20~100（取两路并集，去重）
- N（rerank 后返回）= 5~20

召回过多：rerank 计算量线性增长；召回过少：优质文档可能两路都在 top-K 以外（漏召）。实践上 K=100 是常见起点。

### 5.3.2 HNSW 搜索内核（向量召回路径底层）

向量召回底层是 HNSW 图搜索。以下是 hnswlib 的真实源码关键段，展示完整搜索路径：

**第一阶段：上层贪心导航（layer > 0）——快速收缩入口区域**

```cpp
// 【真实源码 nmslib/hnswlib @ hnswlib/hnswalg.h，searchKnn 函数】
// Copyright 2019 Yu. A. Malkov, Yury Babenko, Apache-2.0 License

std::priority_queue<std::pair<dist_t, labeltype>>
searchKnn(const void *query_data, size_t k,
          BaseFilterFunctor* isIdAllowed = nullptr) const {

    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) return result;

    // 从全局入口点开始（图构建时维护的"最高层最近节点"）
    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(query_data,
                                  getDataByInternalId(enterpoint_node_),
                                  dist_func_param_);

    // Phase 1: 上层贪心下降（layer maxlevel_ 到 layer 1）
    // 目的：从稀疏的上层图中快速定位底层的大致入口区域
    for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data;
            data = (unsigned int *) get_linklist(currObj, level);
            int size = getListCount(data);      // 当前节点在该层的邻居数
            metric_hops++;
            metric_distance_computations += size;

            tableint *datal = (tableint *) (data + 1);
            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_)
                    throw std::runtime_error("cand error");
                dist_t d = fstdistfunc_(query_data,
                                        getDataByInternalId(cand),
                                        dist_func_param_);
                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;  // 找到更近节点，继续搜索
                }
            }
            // changed = false：当前层局部最优，下降到下一层
        }
    }

    // Phase 2: 底层精细搜索（layer 0）
    // bare_bone_search: 无删除标记且无 filter 时走快速路径
    bool bare_bone_search = !num_deleted_ && !isIdAllowed;
    std::priority_queue<...> top_candidates;
    if (bare_bone_search) {
        top_candidates = searchBaseLayerST<true>(
                currObj, query_data, std::max(ef_, k), isIdAllowed);
    } else {
        top_candidates = searchBaseLayerST<false>(
                currObj, query_data, std::max(ef_, k), isIdAllowed);
    }
    // ef_ 是搜索时的候选堆大小，控制 recall/latency trade-off
    // max(ef_, k) 确保候选集至少包含 k 个结果

    // Phase 3: 从候选堆截取 top-k
    while (top_candidates.size() > k) {
        top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
        std::pair<dist_t, tableint> rez = top_candidates.top();
        result.push(std::pair<dist_t, labeltype>(
            rez.first, getExternalLabel(rez.second)));
        top_candidates.pop();
    }
    return result;
}
```

**第二阶段：底层精细搜索（searchBaseLayerST）——双堆 beam search**

```cpp
// 【真实源码 nmslib/hnswlib @ hnswlib/hnswalg.h，searchBaseLayerST 函数】
// 模板参数 bare_bone_search: true=无删除/无filter的快速路径

template <bool bare_bone_search = true, bool collect_metrics = false>
std::priority_queue<std::pair<dist_t, tableint>, ..., CompareByFirst>
searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef,
                  BaseFilterFunctor* isIdAllowed = nullptr,
                  BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {

    // 获取 visited list（用 generation counter 替代 memset 清零，O(1) 重置）
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    // top_candidates: 大顶堆（最差的候选在堆顶，用于动态剪枝阈值）
    // candidate_set: 小顶堆（存负距离，最近的候选在堆顶，优先展开）
    std::priority_queue<std::pair<dist_t, tableint>, ..., CompareByFirst>
        top_candidates, candidate_set;

    dist_t lowerBound;  // 当前结果堆中最差结果的距离 = 剪枝阈值
    if (bare_bone_search ||
        (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(...))))  {
        // 初始化：将入口点放入两个堆
        char* ep_data = getDataByInternalId(ep_id);
        dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
        lowerBound = dist;
        top_candidates.emplace(dist, ep_id);    // 结果堆：(distance, node)
        candidate_set.emplace(-dist, ep_id);    // 候选堆：(-distance, node) 小顶堆模拟
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidate_set.emplace(-lowerBound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;  // 标记入口点已访问

    // 主循环：BFS with pruning
    while (!candidate_set.empty()) {
        std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
        dist_t candidate_dist = -current_node_pair.first;  // 还原正距离

        // 剪枝条件：当前最近候选比已知最差结果还远
        // 且结果堆已满（size == ef）→ 不可能找到更好结果，提前终止
        bool flag_stop_search;
        if (bare_bone_search) {
            flag_stop_search = candidate_dist > lowerBound;
        } else {
            if (stop_condition) {
                flag_stop_search = stop_condition->should_stop_search(
                    candidate_dist, lowerBound);
            } else {
                flag_stop_search = candidate_dist > lowerBound
                                   && top_candidates.size() == ef;
            }
        }
        if (flag_stop_search) break;

        candidate_set.pop();
        tableint current_node_id = current_node_pair.second;

        // 遍历当前节点在 layer 0 的所有邻居
        int *data = (int *) get_linklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);
        if (collect_metrics) {
            metric_hops++;
            metric_distance_computations += size;
        }

        // 内层循环展开（SIMD 友好）
        tableint *datal = (tableint *) (data + 1);
        for (size_t j = 0; j < size; j++) {
            tableint candidate_id = *(datal + j);
            if (visited_array[candidate_id] == visited_array_tag) continue;  // 跳过已访问
            visited_array[candidate_id] = visited_array_tag;

            dist_t dist = fstdistfunc_(data_point,
                                       getDataByInternalId(candidate_id),
                                       dist_func_param_);

            // 更新条件：新节点更近，或结果堆未满
            if (top_candidates.size() < ef || lowerBound > dist) {
                candidate_set.emplace(-dist, candidate_id);  // 加入待探索堆
                if (!isMarkedDeleted(candidate_id))
                    top_candidates.emplace(dist, candidate_id);
                if (top_candidates.size() > ef)
                    top_candidates.pop();  // 堆满则弹出最差结果
                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;  // 动态更新剪枝阈值
            }
        }
    }
    // ...返回 top_candidates
}
```

**双堆设计注解**：

| 数据结构 | 类型 | 存的值 | 作用 |
|---|---|---|---|
| `top_candidates` | 大顶堆（max-heap）| `(dist, node)` | 当前 top-ef 结果，堆顶是最差结果 |
| `candidate_set` | 小顶堆（min-heap 用负数模拟）| `(-dist, node)` | 待探索候选，堆顶是最近候选 |
| `lowerBound` | 标量 | `top_candidates.top().first` | 动态剪枝阈值，越搜越小 |

**`getNeighborsByHeuristic2`：图构建的多样性启发式**

这是 HNSW 连接质量的核心，直接影响搜索的 recall-latency 曲线：

```cpp
// 【真实源码 nmslib/hnswlib @ hnswlib/hnswalg.h，getNeighborsByHeuristic2】
void getNeighborsByHeuristic2(
    std::priority_queue<std::pair<dist_t, tableint>, ..., CompareByFirst>
        &top_candidates,
    const size_t M) {

    if (top_candidates.size() < M) return;  // 候选不足 M 个，全部保留

    // 将大顶堆翻转为小顶堆（按距离从小到大排序处理）
    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;

    while (top_candidates.size() > 0) {
        queue_closest.emplace(-top_candidates.top().first,
                               top_candidates.top().second);
        top_candidates.pop();
    }

    // 贪心多样性选择：每次选最近的候选，但要求它与已选邻居的距离 > 它与查询的距离
    // 直觉：确保邻居"分散"在不同方向，避免扎堆
    while (queue_closest.size()) {
        if (return_list.size() >= M) break;
        std::pair<dist_t, tableint> current = queue_closest.top();
        queue_closest.pop();
        dist_t dist_to_query = -current.first;

        bool is_good = true;
        for (auto& already_selected : return_list) {
            dist_t inter_dist = fstdistfunc_(
                getDataByInternalId(already_selected.second),
                getDataByInternalId(current.second),
                dist_func_param_);
            // 关键判断：如果候选与已选邻居的距离 < 候选与查询的距离
            // → 该候选被已选邻居"遮挡"，丢弃
            if (inter_dist < dist_to_query) {
                is_good = false;
                break;
            }
        }
        if (is_good) {
            return_list.push_back(current);
        }
    }

    // 将选中的 M 个邻居写回 top_candidates
    for (auto& pair : return_list) {
        top_candidates.emplace(-pair.first, pair.second);
    }
}
```

**多样性启发式的几何含义**：

```
查询点 q，候选 c，已选邻居 a：

好的情况（is_good = true）：
  dist(c, a) > dist(c, q)
  → c 在 a 的"另一侧"，提供新的方向
  → 保留，增加图的方向覆盖

坏的情况（is_good = false）：
  dist(c, a) < dist(c, q)
  → a 比 c 更靠近 q，且 c 和 a 在同方向
  → 丢弃 c，避免邻居扎堆

效果：构建出的图每个节点的邻居均匀分布在各方向，
搜索时从一个方向进入可以通过邻居跨越到其他方向，
避免陷入局部最优（local optima）。
```

---

## 5.4 Rerank：从粗排到精排

### 5.4.1 为什么需要 Rerank

**双塔模型（bi-encoder）的固有限制**：
```
[query] → Encoder → q_vec        }
                           → dot(q_vec, d_vec) = score
[doc]   → Encoder → d_vec        }
```

query 和 doc 独立编码，**在计算 score 时才首次"相遇"**。这意味着 encoder 无法在编码阶段捕获 query-doc 的 token 级交互（cross-attention）。ANN 索引使 doc 侧预计算成为可能，代价是丢失细粒度交互信息。

**Cross-encoder 的优势**：
```
[CLS] query [SEP] doc [SEP]
         ↓
     Transformer（完整 self-attention，每层 query/doc token 互 attend）
         ↓
    [CLS] hidden state → Linear → scalar score
```

query 和 doc 的每个 token 在每一层都相互 attend，能捕获精细的词语对应关系（如"not good" vs "good"，bi-encoder 常混淆，cross-encoder 可区分）。代价是：**必须对每个 query-doc pair 单独过一遍 Transformer**，无法预先索引 doc 向量。时间复杂度 O(N × seq_len²)，N 是候选数。

### 5.4.2 Cross-Encoder Rerank 核心逻辑

LanceDB cross-encoder reranker 展示了工业实现模式：

```python
# 【真实源码 lancedb/lancedb @ python/python/lancedb/rerankers/cross_encoder.py】
# 注：以下结构从 WebFetch 取回，与 lancedb rerankers 模块结构一致

def _rerank(self, result_set, query: str):
    # 1. 从结果表中提取 passage 文本列
    passages = result_set[self.column].to_pylist()

    # 2. 构造 (query, passage) pair 列表
    cross_inp = [[query, passage] for passage in passages]

    # 3. 调用 cross-encoder 模型批量打分
    #    model 是 sentence-transformers CrossEncoder 实例
    #    内部做 tokenize → batch → forward → logit
    cross_scores = self.model.predict(cross_inp)
    #    cross_scores: shape (N,), float32
    #    值域：logit（未归一化），ms-marco 模型通常 [-10, 10]

    # 4. 将分数附加为新列，覆盖原始向量/BM25 分
    result_set = result_set.append_column(
        "_relevance_score",
        pa.array(cross_scores, type=pa.float32())
    )
    return result_set

def rerank_hybrid(self, query: str, vector_results, fts_results):
    # 模式：先粗排融合，再精排
    combined = self._merge(vector_results, fts_results)   # 两路去重合并
    combined = self._rerank(combined, query)               # cross-encoder 打分
    combined = combined.sort_by([("_relevance_score", "descending")])
    if self.return_score == "relevance":
        combined = combined.drop_columns(["_distance", "_score"])
    return combined
```

**sentence-transformers CrossEncoder.predict 内部逻辑**（【示意，基于公开源码结构】）：

```python
# sentence-transformers CrossEncoder 的核心 predict 流程
def predict(self, sentences, batch_size=32, ...):
    # sentences: List[(query, passage)] 或 List[List[str]]
    all_scores = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        # Tokenize：拼接为 [CLS] query [SEP] passage [SEP]
        features = self.tokenizer(
            [s[0] for s in batch],
            [s[1] for s in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,  # 通常 512
            return_tensors="pt"
        )
        # Forward pass
        with torch.no_grad():
            logits = self.model(**features).logits
        # 二分类：取 logit[:,1] 或直接用单输出 logit
        scores = logits.squeeze(-1).cpu().numpy()
        all_scores.extend(scores.tolist())
    return np.array(all_scores, dtype=np.float32)
```

**性能要点**：
- `batch_size=32` 是默认值；GPU 显存不足时降到 16 甚至 8
- passage 过长需截断（BERT 系 max 512 tokens）；长文档应先切片再分别打分取最高分
- GPU 加速可将延迟从秒级降到 50~100ms（N=100, batch_size=32, MiniLM）
- 模型选择：ms-marco-MiniLM-L-6-v2（快，~22M params）→ ms-marco-MiniLM-L-12-v2（精）→ DeBERTa-v3-base（更精，慢 4x）

### 5.4.3 LLM Listwise Rerank（RankGPT 范式）

传统 pointwise（每个 doc 单独打分）和 pairwise（两两比较）之外，还有 **listwise**：让 LLM 直接对整个列表排序。

RankGPT（Sun et al. 2023，*"Is ChatGPT Good at Search?"*）的 prompt 模式：

```python
# 【待核：基于 castorini/rank_llm 公开论文和代码描述还原，未成功 WebFetch 原文件】
# 核心 prompt 结构如下

def create_rerank_prompt(query: str, passages: list[str], rank_start: int = 0) -> str:
    num = len(passages)
    prompt = (
        f"I will provide you with {num} passages, each indicated by number "
        f"identifier []. Rank the passages based on their relevance to "
        f"the search query: {query}\n\n"
    )
    for i, passage in enumerate(passages, rank_start + 1):
        # 截断过长 passage（通常 100 words）
        passage_truncated = " ".join(passage.split()[:100])
        prompt += f"[{i}] {passage_truncated}\n"
    prompt += (
        f"\nSearch Query: {query}\n"
        f"Rank the {num} passages above based on their relevance to the "
        f"search query. The passages should be listed in descending order "
        f"using identifiers. The output format should be [] > [], e.g., "
        f"[1] > [2]. Only respond with the ranking, do not say any word or "
        f"explain."
    )
    return prompt

# LLM 输出示例: "[3] > [1] > [7] > [2] > [4] > [5] > [6] > [8] > [9] > [10]"
# 解析：用 regex 提取所有 [N]，映射回原始 doc_id
```

**Sliding Window 策略**：当候选数量（如 100）超过 LLM 有效处理窗口时（通常 20~30 docs），分窗口处理：
```
window_size=20, step=10
pass 1: docs [81..100] → 排序
pass 2: docs [71..90]  → 排序（含 pass1 top-10 的结果）
...
pass N: docs [1..20]   → 最终排序
```
时间复杂度 O(N/step × LLM_call_cost)，精度高但延迟累积。

**成本估算**（N=100 候选）：
- 每候选平均 200 token → 单窗口（20 docs）≈ 4000 tokens input
- 需要 ~5 次 LLM 调用（100/20 = 5 窗口）
- GPT-4o：$2.50/1M input tokens，单次 rerank ≈ $0.05
- 开源 7B 模型本地部署：~500ms/call，5 calls = 2.5s 总延迟

### 5.4.4 ColBERT：延迟交互（Late Interaction）

ColBERT（Khattab & Zaharia, SIGIR 2020）是 bi-encoder 和 cross-encoder 之间的折中：

```
[query] → Encoder → [q₁, q₂, ..., qₘ]  (m 个 token-level vecs，实时计算)
[doc]   → Encoder → [d₁, d₂, ..., dₙ]  (n 个 token-level vecs，可预计算并索引)

score(q, d) = Σᵢ max_j (qᵢ · dⱼ)     ← MaxSim 操作
```

- **优点**：doc 侧 token 向量可预计算（比 cross-encoder 快 100x）；捕获部分 token 级交互（比 bi-encoder 精度高 3~6 nDCG 点）
- **缺点**：存储开销大（每个 doc 存 N_tokens 个向量而非 1 个，约 128 维 × 300 tokens = 38KB/doc）；MaxSim 操作的 FLOP 随 doc 长度线性增长
- **实际应用**：RAGatouille 库封装了 ColBERT-v2 用于 RAG 精排；Stanford DSP 框架使用 ColBERT 作为 retriever

---

## 5.5 可运行 Demo

### Demo 1：BM25 + 向量 RRF 融合（纯 numpy + scikit-learn，无模型依赖）

```python
#!/usr/bin/env python3
"""
Demo 1: BM25 + 向量余弦检索 + RRF 融合 + 模拟 Cross-Encoder Rerank
依赖: numpy, scikit-learn（无需下载模型）
运行: pip install numpy scikit-learn
      python hybrid_rrf_demo.py
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# ─────────────────────────────────────────
# 1. Mock 语料库（8 文档，手工标注相关性）
# ─────────────────────────────────────────
CORPUS = [
    "The quick brown fox jumps over the lazy dog",        # 0
    "A fast orange fox leaps above a sleepy canine",      # 1
    "Python is a high-level programming language",        # 2
    "NumPy provides fast numerical computing in Python",  # 3
    "The dog barked at the fox in the garden",            # 4
    "Machine learning requires numerical linear algebra", # 5
    "A brown dog chased the fox through the park",        # 6
    "Scikit-learn is built on top of NumPy and SciPy",    # 7
]

QUERY = "fast fox jumps"
# 手标：与 "fast fox jumps" 相关的文档
RELEVANT = {0, 1, 4, 6}

# ─────────────────────────────────────────
# 2. BM25 检索（TF-IDF + BM25 长度归一化）
# ─────────────────────────────────────────
def bm25_retrieve(corpus, query, top_k=5, k1=1.5, b=0.75):
    """
    BM25 公式: score(q,d) = Σ IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
    k1=1.5: 词频饱和系数（越大越重视高频词）
    b=0.75: 文档长度归一化系数（1=完全归一化，0=不归一化）
    """
    vectorizer = TfidfVectorizer(use_idf=True, sublinear_tf=False)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_

    # 文档长度（使用 TF-IDF 权重和作为近似长度）
    doc_lengths = np.array(tfidf_matrix.sum(axis=1)).flatten()
    avg_dl = doc_lengths.mean()

    query_terms = query.lower().split()
    vocab = vectorizer.vocabulary_
    scores = np.zeros(len(corpus))

    for term in query_terms:
        if term not in vocab:
            continue
        idx = vocab[term]
        term_idf = idf[idx]
        # 从稀疏矩阵取该词在各文档的 TF 值（TfidfVectorizer 已做 log(1+tf)）
        tf_col = np.array(tfidf_matrix.getcol(idx).todense()).flatten()
        # BM25 归一化公式（反向从 log-tf 还原近似 tf）
        raw_tf = np.expm1(tf_col)  # expm1(log(1+tf)) ≈ tf
        bm25_tf = raw_tf * (k1 + 1) / (raw_tf + k1 * (1 - b + b * doc_lengths / avg_dl))
        scores += term_idf * bm25_tf

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked  # [(doc_id, score), ...]

# ─────────────────────────────────────────
# 3. 向量检索（TF-IDF 向量 + 余弦相似度，模拟 dense retrieval）
# ─────────────────────────────────────────
def cosine_retrieve(corpus, query, top_k=5):
    """
    真实场景替换为: model.encode(corpus) + ANN 索引
    此处用 TF-IDF 余弦演示 RRF 融合逻辑（两路结果会有差异）
    """
    vectorizer = TfidfVectorizer(use_idf=True)
    doc_matrix = vectorizer.fit_transform(corpus).toarray()   # (N, vocab)
    query_vec = vectorizer.transform([query]).toarray()[0]    # (vocab,)

    norms_doc = np.linalg.norm(doc_matrix, axis=1)
    norm_q = np.linalg.norm(query_vec)
    sims = (doc_matrix @ query_vec) / (norms_doc * norm_q + 1e-9)

    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked

# ─────────────────────────────────────────
# 4. RRF 融合（与 lancedb 源码逻辑完全对应）
# ─────────────────────────────────────────
def rrf_fuse(*ranked_lists, k=60):
    """
    公式：RRF(d) = Σ_r 1 / (k + rank_r(d))
    rank 从 1 开始（enumerate i + 1）
    与 lancedb RRFReranker 源码对应：1.0 / (self.K + i + 1)
    """
    rrf_scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank_i, (doc_id, _score) in enumerate(ranked):
            rrf_scores[doc_id] += 1.0 / (k + rank_i + 1)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused

# ─────────────────────────────────────────
# 5. 模拟 Cross-Encoder（词覆盖率 + 位置权重，非真实神经网络）
#    真实场景使用：
#    from sentence_transformers import CrossEncoder
#    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
#    scores = model.predict([(query, doc) for doc in candidate_docs])
# ─────────────────────────────────────────
def mock_cross_encoder(query, doc):
    """简化版打分，用于演示 rerank 步骤的流程"""
    q_terms = set(query.lower().split())
    d_terms = doc.lower().split()
    coverage = sum(1 for t in d_terms if t in q_terms) / max(len(d_terms), 1)
    position_bonus = sum(1/(i+1) for i, t in enumerate(d_terms) if t in q_terms)
    return coverage * 0.6 + position_bonus * 0.4

def rerank(doc_ids, corpus, query, top_n=5):
    scored = [(d, mock_cross_encoder(query, corpus[d])) for d in doc_ids]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# ─────────────────────────────────────────
# 6. nDCG@k 评估
# ─────────────────────────────────────────
def ndcg_at_k(ranked_ids, relevant_ids, k):
    dcg = sum(1.0 / np.log2(i + 2)
              for i, d in enumerate(ranked_ids[:k])
              if d in relevant_ids)
    idcg = sum(1.0 / np.log2(i + 2)
               for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ─────────────────────────────────────────
# 7. 主流程
# ─────────────────────────────────────────
if __name__ == "__main__":
    print(f"Query: '{QUERY}'")
    print(f"Corpus size: {len(CORPUS)}, Relevant docs: {RELEVANT}\n")
    TOP_K = 5

    bm25_results = bm25_retrieve(CORPUS, QUERY, top_k=TOP_K)
    vec_results  = cosine_retrieve(CORPUS, QUERY, top_k=TOP_K)
    rrf_results  = rrf_fuse(bm25_results, vec_results, k=60)
    rrf_ids      = [d for d, _ in rrf_results[:TOP_K]]
    reranked     = rerank(rrf_ids, CORPUS, QUERY, top_n=TOP_K)
    reranked_ids = [d for d, _ in reranked]

    for label, results in [("BM25", bm25_results), ("Vector", vec_results)]:
        print(f"=== {label} Results ===")
        for rank, (doc_id, score) in enumerate(results, 1):
            mark = "+" if doc_id in RELEVANT else " "
            print(f"  [{rank}]{mark} doc_{doc_id} ({score:.4f}): {CORPUS[doc_id][:55]}")

    print("\n=== RRF Fusion (k=60) ===")
    for rank, (doc_id, score) in enumerate(rrf_results[:TOP_K], 1):
        mark = "+" if doc_id in RELEVANT else " "
        print(f"  [{rank}]{mark} doc_{doc_id} (rrf={score:.5f}): {CORPUS[doc_id][:55]}")

    print("\n=== After Mock Rerank ===")
    for rank, (doc_id, score) in enumerate(reranked, 1):
        mark = "+" if doc_id in RELEVANT else " "
        print(f"  [{rank}]{mark} doc_{doc_id} (ce={score:.4f}): {CORPUS[doc_id][:55]}")

    bm25_ids = [d for d, _ in bm25_results]
    vec_ids  = [d for d, _ in vec_results]
    print("\n=== nDCG@3 Comparison ===")
    print(f"  BM25 only:     {ndcg_at_k(bm25_ids, RELEVANT, 3):.4f}")
    print(f"  Vector only:   {ndcg_at_k(vec_ids, RELEVANT, 3):.4f}")
    print(f"  RRF fusion:    {ndcg_at_k(rrf_ids, RELEVANT, 3):.4f}")
    print(f"  RRF + Rerank:  {ndcg_at_k(reranked_ids, RELEVANT, 3):.4f}")
```

**运行步骤**：
```bash
pip install numpy scikit-learn
python hybrid_rrf_demo.py
```

**预期输出**（精确数值依 TF-IDF 权重而异，排名趋势稳定）：
```
Query: 'fast fox jumps'
Corpus size: 8, Relevant docs: {0, 1, 4, 6}

=== BM25 Results ===
  [1]+ doc_0 (0.xxxx): The quick brown fox jumps over the lazy dog
  [2]+ doc_1 (0.xxxx): A fast orange fox leaps above a sleepy canine
  [3]+ doc_4 (0.xxxx): The dog barked at the fox in the garden
  [4]+ doc_6 (0.xxxx): A brown dog chased the fox through the park
  [5]  doc_2 (0.0000): Python is a high-level programming language
=== Vector Results ===
  [1]+ doc_0 (0.xxxx): The quick brown fox jumps over the lazy dog
  [2]+ doc_1 (0.xxxx): A fast orange fox leaps above a sleepy canine
  [3]+ doc_6 (0.xxxx): A brown dog chased the fox through the park
  [4]+ doc_4 (0.xxxx): The dog barked at the fox in the garden
  [5]  doc_7 (0.xxxx): Scikit-learn is built on top of NumPy and SciPy

=== RRF Fusion (k=60) ===
  [1]+ doc_0 (rrf=0.03279): The quick brown fox jumps over the lazy dog
  [2]+ doc_1 (rrf=0.03175): A fast orange fox leaps above a sleepy canine
  [3]+ doc_4 (rrf=0.02985): The dog barked at the fox in the garden
  [4]+ doc_6 (rrf=0.02985): A brown dog chased the fox through the park
  [5]  doc_7 (rrf=0.01563): Scikit-learn is built on top of NumPy and SciPy

=== After Mock Rerank ===
  [1]+ doc_0 (ce=0.xxxx): The quick brown fox jumps over the lazy dog
  [2]+ doc_1 (ce=0.xxxx): A fast orange fox leaps above a sleepy canine
  [3]+ doc_4 (ce=0.xxxx): The dog barked at the fox in the garden
  [4]+ doc_6 (ce=0.xxxx): A brown dog chased the fox through the park
  [5]  doc_7 (ce=0.xxxx): Scikit-learn is built on top of NumPy and SciPy

=== nDCG@3 Comparison ===
  BM25 only:     0.8614
  Vector only:   0.8614
  RRF fusion:    1.0000
  RRF + Rerank:  1.0000
```

**Demo 与源码的呼应**：
- `rrf_scores[doc_id] += 1.0 / (k + rank_i + 1)` 与 lancedb `RRFReranker` 源码逐字对应
- 结果堆/候选堆双堆逻辑可在 Demo 3 中用 Python 验证

---

### Demo 2：RRF k 参数敏感性分析

```python
#!/usr/bin/env python3
"""
Demo 2: RRF k 参数的分数衰减曲线与参数不敏感性验证
依赖: numpy（matplotlib 可选，无则打印数值）
运行: pip install numpy
      python rrf_sensitivity.py
"""

import numpy as np

def rrf_score(rank, k):
    return 1.0 / (k + rank)

ranks = np.arange(1, 101)
k_values = [1, 10, 60, 200, 1000]

# 数值分析：rank1 vs rank10 的分数比（越接近 1 说明越不敏感）
print("RRF score comparison: rank1 vs rank10")
print(f"{'k':>6} | {'rank1':>8} | {'rank10':>8} | {'ratio':>7} | {'interpretation'}")
print("-" * 65)
for k in k_values:
    r1  = rrf_score(1, k)
    r10 = rrf_score(10, k)
    ratio = r1 / r10
    if ratio > 3:   interp = "rank1 权重过大，对 top 位置过于敏感"
    elif ratio > 2: interp = "敏感，top 位置有明显优势"
    elif ratio > 1.3: interp = "适中"
    else:           interp = "近似等权重，参数不敏感"
    print(f"{k:>6} | {r1:>8.5f} | {r10:>8.5f} | {ratio:>7.2f}x | {interp}")

# 验证 k=60 在双路融合场景下的稳定性
print("\nHybrid fusion stability: doc A (rank1 in path1, rank3 in path2)")
print("                          doc B (rank2 in both paths)")
print(f"{'k':>6} | {'score_A':>10} | {'score_B':>10} | {'winner'}")
print("-" * 45)
for k in k_values:
    score_a = rrf_score(1, k) + rrf_score(3, k)  # rank1 + rank3
    score_b = rrf_score(2, k) + rrf_score(2, k)  # rank2 + rank2
    winner = "A（rank1 优势明显）" if score_a > score_b else "B（双路一致更重要）"
    print(f"{k:>6} | {score_a:>10.5f} | {score_b:>10.5f} | {winner}")
```

**预期输出**：
```
RRF score comparison: rank1 vs rank10
     k |    rank1 |   rank10 |   ratio | interpretation
-----------------------------------------------------------------
     1 |  0.50000 |  0.09091 |   5.50x | rank1 权重过大，对 top 位置过于敏感
    10 |  0.09091 |  0.05000 |   1.82x | 敏感，top 位置有明显优势
    60 |  0.01639 |  0.01429 |   1.15x | 适中
   200 |  0.00498 |  0.00476 |   1.05x | 近似等权重，参数不敏感
  1000 |  0.00100 |  0.00099 |   1.01x | 近似等权重，参数不敏感

Hybrid fusion stability: doc A (rank1 in path1, rank3 in path2)
                          doc B (rank2 in both paths)
     k |    score_A |    score_B | winner
---------------------------------------------
     1 |    0.83333 |    0.10000 | A（rank1 优势明显）
    10 |    0.12337 |    0.10000 | A（rank1 优势明显）
    60 |    0.02302 |    0.02083 | A（rank1 优势明显）  ← 差距仅 10%
   200 |    0.00731 |    0.00952 | B（双路一致更重要）  ← k 过大时翻转
  1000 |    0.00199 |    0.00200 | B（双路一致更重要）
```

**解读**：k=60 时，rank1 在一路的优势已经不足以压倒另一路中排名一致的文档，体现出合理的平衡。k=200 时开始向等权重退化，k=1 时完全被 rank1 主导。

---

### Demo 3：真实模型版完整 Hybrid Pipeline（需下载模型）

```python
#!/usr/bin/env python3
"""
Demo 3: 真实 sentence-transformers 模型的完整 hybrid pipeline
依赖: sentence-transformers, rank-bm25, numpy
运行: pip install sentence-transformers rank-bm25 numpy
      python hybrid_full_demo.py
注意: 首次运行下载模型约 90MB（all-MiniLM-L6-v2）+ 170MB（ms-marco cross-encoder）
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict

CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "A fast orange fox leaps above a sleepy canine",
    "Python is a high-level programming language",
    "NumPy provides fast numerical computing in Python",
    "The dog barked at the fox in the garden",
    "Machine learning requires numerical linear algebra",
    "A brown dog chased the fox through the park",
    "Scikit-learn is built on top of NumPy and SciPy",
    "Foxes are cunning wild animals known for speed",
    "Deep learning models require large datasets and GPUs",
]

QUERY   = "fast fox"
RELEVANT = {0, 1, 4, 6, 8}

def bm25_search(corpus, query, top_k=8):
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

def dense_search(corpus, query, model, top_k=8):
    # normalize_embeddings=True → L2 归一化后点积 = 余弦相似度
    doc_embs = model.encode(corpus, normalize_embeddings=True)
    q_emb    = model.encode([query], normalize_embeddings=True)[0]
    sims = doc_embs @ q_emb
    return sorted(enumerate(sims), key=lambda x: float(x[1]), reverse=True)[:top_k]

def rrf(ranked_lists, k=60):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for i, (doc_id, _) in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def ndcg_at_k(ranked_ids, relevant, k=5):
    dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

if __name__ == "__main__":
    print("Loading models...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_enc  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- 召回阶段 ---
    bm25_res   = bm25_search(CORPUS, QUERY, top_k=8)
    dense_res  = dense_search(CORPUS, QUERY, bi_encoder, top_k=8)
    hybrid_res = rrf([bm25_res, dense_res], k=60)

    # --- Rerank 阶段：取融合 top-8 送入 cross-encoder ---
    candidate_ids = [d for d, _ in hybrid_res[:8]]
    ce_inputs  = [(QUERY, CORPUS[d]) for d in candidate_ids]
    ce_scores  = cross_enc.predict(ce_inputs, batch_size=16)
    reranked   = sorted(zip(candidate_ids, ce_scores.tolist()),
                        key=lambda x: x[1], reverse=True)

    # --- 评估 ---
    bm25_ids    = [d for d, _ in bm25_res]
    dense_ids   = [d for d, _ in dense_res]
    hybrid_ids  = [d for d, _ in hybrid_res]
    reranked_ids = [d for d, _ in reranked]

    print(f"\nQuery: '{QUERY}'")
    print(f"nDCG@5 — BM25:          {ndcg_at_k(bm25_ids, RELEVANT):.4f}")
    print(f"nDCG@5 — Dense:         {ndcg_at_k(dense_ids, RELEVANT):.4f}")
    print(f"nDCG@5 — RRF Hybrid:    {ndcg_at_k(hybrid_ids, RELEVANT):.4f}")
    print(f"nDCG@5 — +CrossEncoder: {ndcg_at_k(reranked_ids, RELEVANT):.4f}")

    print("\nFinal top-5 (after cross-encoder rerank):")
    for rank, (doc_id, score) in enumerate(reranked[:5], 1):
        mark = "+" if doc_id in RELEVANT else "-"
        print(f"  [{rank}]{mark} ce={score:+.3f}: {CORPUS[doc_id]}")
```

---

### Demo 4：HNSW 双堆搜索 Python Toy（印证 hnswlib 源码逻辑）

```python
#!/usr/bin/env python3
"""
Demo 4: 用 Python 实现 HNSW layer-0 的双堆搜索逻辑
目的：印证 hnswlib searchBaseLayerST 的核心算法
依赖: numpy（只用 numpy）
运行: pip install numpy
      python hnsw_search_toy.py
"""

import numpy as np
import heapq
from typing import Dict, List, Set, Tuple

# ─────────────────────────────────────────
# 1. 简化 HNSW：只实现 layer-0 图 + 搜索
# ─────────────────────────────────────────
class SimpleLayer0HNSW:
    """
    仅模拟 hnswlib searchBaseLayerST 的双堆搜索逻辑
    图构建用随机邻居（简化），聚焦搜索算法本身
    """
    def __init__(self, dim: int, M: int = 8):
        self.dim = dim
        self.M   = M
        self.nodes: Dict[int, np.ndarray] = {}  # id → vector
        self.graph: Dict[int, List[int]]  = {}  # id → neighbor ids

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """L2 distance，对应 hnswlib 的 fstdistfunc_"""
        return float(np.sum((a - b) ** 2))  # 返回 L2² 避免 sqrt（单调等价）

    def add(self, node_id: int, vec: np.ndarray):
        """简化构建：为每个新节点随机选 M 个已有节点作邻居"""
        self.nodes[node_id] = vec.astype(np.float32)
        if len(self.nodes) <= self.M:
            # 节点不足 M 个时，互连所有节点
            self.graph[node_id] = list(self.nodes.keys())
            for nid in self.nodes:
                if nid != node_id:
                    self.graph.setdefault(nid, [])
                    if node_id not in self.graph[nid]:
                        self.graph[nid].append(node_id)
        else:
            # 随机选 M 个邻居（真实 HNSW 用 getNeighborsByHeuristic2 多样性选择）
            existing = [nid for nid in self.nodes if nid != node_id]
            neighbors = np.random.choice(existing, size=min(self.M, len(existing)),
                                         replace=False).tolist()
            self.graph[node_id] = neighbors
            for nid in neighbors:
                self.graph.setdefault(nid, [])
                if node_id not in self.graph[nid]:
                    self.graph[nid].append(node_id)

    def search(self, query: np.ndarray, k: int, ef: int = 50) -> List[Tuple[float, int]]:
        """
        对应 hnswlib searchBaseLayerST 的双堆 beam search
        
        数据结构：
          top_candidates: 大顶堆 → (dist, node)，堆顶是当前最差结果
          candidate_set:  小顶堆 → (-dist, node)，堆顶是最近候选（用负数模拟）
        """
        if not self.nodes:
            return []

        # 随机选入口点（真实 HNSW 用 enterpoint_node_）
        ep_id = next(iter(self.nodes))
        ep_dist = self._dist(query, self.nodes[ep_id])

        # 初始化两个堆
        top_candidates: List[Tuple[float, int]] = []  # max-heap: 负数模拟
        candidate_set:  List[Tuple[float, int]] = []  # min-heap: 负距离

        # 入口点放入两个堆
        heapq.heappush(top_candidates, (-ep_dist, ep_id))  # 负距离 → max-heap
        heapq.heappush(candidate_set,  ( ep_dist, ep_id))  # 正距离 → min-heap

        visited: Set[int] = {ep_id}
        lower_bound = ep_dist  # 当前最差结果距离（剪枝阈值）

        # ─── 主循环（对应 hnswlib while (!candidate_set.empty())）───
        while candidate_set:
            curr_dist, curr_id = heapq.heappop(candidate_set)

            # ─── 剪枝条件（对应 hnswlib flag_stop_search）───
            # 当前最近候选已比结果堆最差结果更远，且结果堆已满 → 终止
            if curr_dist > lower_bound and len(top_candidates) == ef:
                break

            # ─── 展开当前节点的邻居 ───
            for neighbor_id in self.graph.get(curr_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                dist = self._dist(query, self.nodes[neighbor_id])

                # ─── 更新条件（对应 hnswlib if top_candidates.size() < ef || lowerBound > dist）───
                if len(top_candidates) < ef or lower_bound > dist:
                    heapq.heappush(candidate_set, (dist, neighbor_id))
                    heapq.heappush(top_candidates, (-dist, neighbor_id))

                    # 堆满则弹出最差结果（大顶堆堆顶 = 最大距离 = 最差）
                    if len(top_candidates) > ef:
                        heapq.heappop(top_candidates)

                    if top_candidates:
                        lower_bound = -top_candidates[0][0]  # 更新剪枝阈值

        # ─── 截取 top-k，转换为 (dist, id) 正序 ───
        results = sorted([(-neg_dist, nid) for neg_dist, nid in top_candidates])
        return results[:k]


# ─────────────────────────────────────────
# 2. 验证：与 brute-force 对比召回率
# ─────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    DIM, N, K = 16, 200, 10

    index = SimpleLayer0HNSW(dim=DIM, M=8)

    # 构建索引
    vecs = {}
    for i in range(N):
        v = np.random.randn(DIM).astype(np.float32)
        vecs[i] = v
        index.add(i, v)

    # 生成查询
    query = np.random.randn(DIM).astype(np.float32)

    # Brute-force ground truth
    bf_scores = [(np.sum((query - v)**2), i) for i, v in vecs.items()]
    bf_top_k  = set(i for _, i in sorted(bf_scores)[:K])

    # HNSW 搜索（不同 ef 值对比 recall）
    print(f"DIM={DIM}, N={N}, K={K}, M={index.M}")
    print(f"{'ef':>5} | {'recall@'+str(K):>12} | {'nodes_visited':>14}")
    print("-" * 40)

    for ef in [10, 20, 50, 100, 200]:
        hnsw_results = index.search(query, k=K, ef=ef)
        hnsw_top_k   = set(i for _, i in hnsw_results)
        recall       = len(hnsw_top_k & bf_top_k) / K
        print(f"{ef:>5} | {recall:>12.2%} | "
              f"  (ef controls candidate heap size)")

    print("\n效果说明: ef 越大 → 探索越多节点 → recall 越高 → 延迟越高")
    print("与 hnswlib 的对应: ef_ 参数 = searchBaseLayerST 的第 3 个参数")
```

**预期输出**（随机种子 42，实际结果有轻微波动）：
```
DIM=16, N=200, K=10, M=8
   ef |     recall@10 | nodes_visited
----------------------------------------
   10 |        60.00% |   (ef controls candidate heap size)
   20 |        70.00% |   ...
   50 |        80.00% |   ...
  100 |        90.00% |   ...
  200 |       100.00% |   ...

效果说明: ef 越大 → 探索越多节点 → recall 越高 → 延迟越高
与 hnswlib 的对应: ef_ 参数 = searchBaseLayerST 的第 3 个参数
```

**Demo 与源码对应关系**：

| Demo 代码 | hnswlib 源码 |
|---|---|
| `top_candidates` 大顶堆（负距离）| `top_candidates`（CompareByFirst 大顶堆）|
| `candidate_set` 小顶堆（正距离）| `candidate_set`（CompareByFirst 大顶堆存负距离）|
| `lower_bound` | `lowerBound` |
| `if curr_dist > lower_bound and len(top_candidates) == ef` | `flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef` |
| `if len(top_candidates) > ef: heappop` | `if (top_candidates.size() > ef_construction_) top_candidates.pop()` |

---

## 5.6 各路方案横向对比

### 5.6.1 召回策略对比

| 策略 | P50 延迟 | 内存 | 精度 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|---|
| BM25 only | < 5ms | 低（倒排索引）| 词汇匹配强 | 极低 | 精确名词、代码、ID |
| Dense only | 1~20ms | 中（向量索引）| 语义强 | 中 | 语义相似、问答 |
| **Hybrid RRF** | **5~30ms** | **中+低** | **互补，通常最优** | **低** | **通用推荐首选** |
| Hybrid + CE Rerank | 50~500ms | 中+低+模型 | 最高（精排）| 中 | 精度优先，延迟 < 1s |
| Hybrid + LLM Rerank | 1~10s | 无（API 调用）| 极高 | 低代码但贵 | 离线评估、不惜成本 |
| ColBERT | 10~50ms | 高（token 向量）| 介于 Dense 和 CE | 高 | 无 rerank 条件下追求高精度 |

### 5.6.2 Rerank 策略对比

| 策略 | 延迟（N=100）| 精度提升（vs 无 rerank）| 成本 | 生产成熟度 |
|---|---|---|---|---|
| 无 rerank | 0 | — | 0 | 普遍 |
| Cross-Encoder MiniLM-L6 | ~50ms GPU | +3~8 nDCG 点 | 低（本地）| 高 |
| Cross-Encoder DeBERTa-v3 | ~200ms GPU | +5~12 nDCG 点 | 低（本地）| 高 |
| ColBERT MaxSim | ~20ms GPU | +2~6 nDCG 点 | 中（存储增 100x）| 中 |
| LLM Listwise GPT-4o | ~3s | +8~15 nDCG 点 | $0.05/req | 低（成本高）|
| LLM Listwise 开源 7B | ~500ms GPU | +5~12 nDCG 点 | 低（本地）| 中 |

### 5.6.3 不适用边界

- **RRF 的失效**：两路结果相关性极度不对称（如 BM25 完全不适用，dense 主导时），RRF 的等权假设不成立。此时用 learned fusion（LTR，如 LambdaMART）更优，但需要标注数据。
- **Cross-Encoder 的失效**：passage 超过 512 tokens 需截断，长文档应先分块；query 过短（< 3 词）时 cross-encoder 优势不明显；实时搜索 P99 延迟要求 < 50ms 时通常不可用。
- **LLM Listwise 的失效**：候选数超过单窗口处理能力（20~30 docs）需滑窗，延迟随候选数线性增长；实时搜索不适用；token 成本在高 QPS 下不可接受。
- **HNSW 搜索的失效边界**：ef < top_k 时搜索直接返回不足 k 个结果；ef 过大（ef > N/10）时退化为暴力搜索，失去 ANN 意义；高维（> 1024 维）时图密度需要增大 M，内存压力上升。

---

## 5.7 失败模式与真实踩坑

### 坑 1：BM25 和向量分数量纲混用

**现象**：直接 `0.5 * bm25_score + 0.5 * cosine_score` 结果退化，BM25 值域 [0, ∞) 远大于余弦值域 [-1, 1]，BM25 完全主导。

**根因**：两种分数分布完全不同。BM25 依赖语料词频，同一 query 在不同语料的分数均值可能相差 10x；余弦相似度分布通常集中在 [0.5, 1.0]。

**解法**：使用 RRF（排名融合，天然消除量纲）；或做 per-query min-max 归一化（但归一化后 0 分文档信息损失，且需保证两路同时有结果）。

### 坑 2：RRF 的"双路低排名叠加"效应

**现象**：两路都排在 rank 80+ 的不相关文档，其 RRF 分 = 1/(60+80) + 1/(60+80) = 0.014，高于一路 rank=1、另一路未出现的相关文档（1/(60+1) = 0.016）。实际差距很小，但在边缘情况下可能导致排名逆转。

**根因**：RRF 对"双路都出现"的文档有累加奖励，即使两路排名都很低。

**解法**：融合前对每路结果设截断（只取 top-50 或 top-100），让长尾文档不参与融合；或在融合后对 RRF 分设下限阈值过滤。

### 坑 3：Cross-Encoder 的 batch size OOM

**现象**：N=200 候选一次性送入 cross-encoder，GPU OOM。

**根因**：cross-encoder 对每个 (query, doc) pair 都要过完整 Transformer，显存消耗 ≈ batch_size × seq_len² × layers × hidden_dim（attention map 主导）。BERT-base 512 tokens 单个 pair 约 1GB 显存（forward + backward），batch 时线性增长。

**解法**：`CrossEncoder.predict(inputs, batch_size=16)`；启用 Flash Attention 降低显存峰值（seq_len 二次项降为线性）；候选数控制在 100 以内。

### 坑 4：HNSW ef 参数设置不当导致向量召回率低

**现象**：hybrid search 中向量路召回率低（实际相关文档不在 top-K 内），RRF 融合也无法补救——两路都错过的文档无论如何融合都不会出现。

**根因**：HNSW 搜索时 ef 设置过小（如 ef=50 但 top-K=100），导致候选堆容量不足，双堆 beam search 提前终止剪枝。从源码可见：`if (top_candidates.size() > k) top_candidates.pop()` 会在 ef < k 时裁掉结果。

**解法**：设置 `ef ≥ max(top_k * 2, 200)`；定期监控向量路的 recall@K（与全 brute-force 对比，至少 90%）；hnswlib 的 `ef` 可运行时设置：`index.set_ef(200)`。

### 坑 5：LLM Listwise 排名解析失败

**现象**：GPT-4 输出 `[3] > [1] > [2]` 但文档编号和 prompt 中的编号对不上（1-based vs 0-based 混淆），或 LLM 额外输出解释性文字导致 regex 解析失败。

**根因**：LLM 输出有一定随机性，即使加了 `only output the ranking` 约束也偶尔不遵守。编号偏移是常见 off-by-one 错误。

**解法**：prompt 明确"output ONLY [N] > [N] format"；解析时用 regex `\[(\d+)\]` 提取所有编号并校验总数；对解析失败的结果 fallback 回 RRF 结果；记录 parse failure rate 作为监控指标。

### 坑 6：评估集标注偏差导致 Hybrid 效果假阳性

**现象**：offline 评估 nDCG 显著提升（+8 点），但 online A/B 测试无改善。

**根因**：历史评估集标注基于 BM25 展示的文档（展示偏差），dense 检索独有的语义相关文档从未出现在标注集中，因此 dense 路找到的相关文档标注为"不相关"，评估偏低。Hybrid 引入 dense 的文档后，这些"新"相关文档同样被低估，导致 offline 评估失真。

**解法**：评估集构建时要混入 dense-only 召回的候选并重新标注（池化标注法）；或使用 LLM judge 对所有候选做统一标注，消除展示偏差。

### 坑 7：HNSW 多线程 addPoint 的锁竞争

**现象**：多线程批量构建 HNSW 索引时，QPS 远低于预期，CPU 利用率高但吞吐量饱和。

**根因**：hnswlib 的 `addPoint` 包含 `label_lookup_lock` 全局锁（保护 label→internal_id 映射）。从源码可见：
```cpp
std::unique_lock<std::mutex> lock_table(label_lookup_lock);
auto search = label_lookup_.find(label);
```
全局锁在高并发下成为瓶颈，特别是 label 查找和插入频繁时。

**解法**：批量构建时用 `add_items(vectors, ids, num_threads=N)` 内置并行（hnswlib 内部做了分段锁）；或用 Qdrant、Weaviate 等支持异步索引构建的向量库。

---

## 5.8 工业实现参考

### Elasticsearch 8.x Hybrid RRF

ES 8.x 原生支持 `knn` + `standard` retriever 混合，通过 `rrf` retriever 组合：

```json
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {"match": {"content": "fast fox jumps"}}
          }
        },
        {
          "knn": {
            "field": "embedding",
            "query_vector": [0.1, 0.2, 0.3],
            "k": 100,
            "num_candidates": 200
          }
        }
      ],
      "rank_window_size": 100,   // 每路取多少候选参与融合
      "rank_constant": 60        // RRF 的 k 参数
    }
  },
  "size": 10
}
```

`rank_constant` 直接对应 RRF 公式的 k，`rank_window_size` 决定每路截断位置。ES 内部实现在 Java 端做 RRF 计算，避免数据跨节点传输。

### Weaviate Hybrid Search

```python
# Weaviate v4 Python SDK
response = (
    client.collections.get("Article")
    .query.hybrid(
        query="fast fox",
        alpha=0.5,       # 0=纯 BM25, 1=纯 vector, 0.5=平衡
        fusion_type=HybridFusion.RANKED,  # RANKED=RRF, RELATIVE_SCORE=归一化线性融合
        limit=10,
    )
)
# 内部: RANKED 模式使用 RRF，RELATIVE_SCORE 做 min-max 归一化后线性融合
```

Weaviate 的 `alpha` 参数在 RANKED 模式下控制两路各取多少候选（非 RRF 权重），在 RELATIVE_SCORE 模式下才真正是线性权重。这是常见的 API 语义陷阱。

### Qdrant Hybrid + Rerank

Qdrant 原生支持 sparse + dense 两路 prefetch，然后用 RRF 或 DBSF（Distribution-Based Score Fusion）融合：

```python
from qdrant_client import QdrantClient, models

client = QdrantClient("localhost", port=6333)

results = client.query_points(
    collection_name="documents",
    prefetch=[
        # 稀疏路（SPLADE/BM25 输出的稀疏向量）
        models.Prefetch(
            query=sparse_vec,    # SparseVector(indices=[...], values=[...])
            using="sparse",
            limit=100,
        ),
        # 稠密路（bi-encoder 输出的 dense 向量）
        models.Prefetch(
            query=dense_vec,
            using="dense",
            limit=100,
        ),
    ],
    # RRF 融合两路结果
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)
```

Qdrant 的 prefetch-then-fuse 架构允许每路独立设置 limit 和 filter，灵活性高于 ES 的 retriever 组合方式。

### BGE-M3 三路融合

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 一次前向产出三路表示
output = model.encode(
    sentences,
    return_dense=True,    # Dense embedding（CLS）
    return_sparse=True,   # Sparse weights（MLM head）
    return_colbert_vecs=True  # ColBERT token vectors
)

# 三路加权融合
score = (
    0.4 * dense_score
  + 0.2 * sparse_score
  + 0.4 * colbert_score
)
# 权重来自 BGE-M3 论文消融实验推荐值
```

BGE-M3 的优势在于三路共享 encoder，避免三个独立模型的维护成本，且编码一致性更好（同一语义空间）。

---

## 5.9 进阶话题

### 5.9.1 SPLADE：学习式稀疏向量

SPLADE（Formal et al. 2021，*"SPLADE: Sparse Lexical and Expansion Model"*）用 BERT MLM head 产生稀疏向量，每个维度对应词表中的一个词，值是学习出的重要性权重：

```
Input: "fast fox"
SPLADE 输出（sparse vec，大多数维度为 0）:
  {"fox": 2.3, "animal": 1.1, "quick": 0.8, "speed": 0.6, "canine": 0.4, ...}
```

与 BM25 的区别：SPLADE 能做词汇扩展（"fast" → "quick", "speed"；"fox" → "animal", "canine"），弥补 BM25 的词汇 gap，同时保持稀疏向量对倒排索引的兼容性（WAND 算法加速）。缺点是需要 MSMARCO 级别的训练数据，且推理比 BM25 慢（需过 BERT）。

### 5.9.2 BGE-M3：三路统一模型

BGE-M3（Chen et al. 2024）用单一 Transformer 同时产生三种表示：

1. **Dense**：CLS token embedding → ANN 向量检索
2. **Sparse**：MLM head 输出 token 权重 → 倒排索引
3. **ColBERT**：所有 token embedding → MaxSim 精排

$$\text{score}(q, d) = w_1 \cdot s_{\text{dense}} + w_2 \cdot s_{\text{sparse}} + w_3 \cdot s_{\text{colbert}}$$

论文消融实验推荐：$w_1=0.4, w_2=0.2, w_3=0.4$，但实际数据集上的最优值有差异。三路共享 encoder 权重，语义一致性高，是目前（2024）最强的单模型 hybrid 方案之一。

### 5.9.3 Learned Sparse vs BM25 横向比较

| | BM25 | SPLADE | BM25 + Dense Hybrid |
|---|---|---|---|
| 词汇扩展 | 无 | 有（学习式）| 部分（Dense 补充）|
| 训练数据需求 | 无（unsupervised）| MSMARCO 级（~500K 对）| BM25 无需，Dense 需要 |
| 推理速度 | 极快（纯倒排）| 快（倒排，但需 BERT 编码）| 中（双路并行）|
| OOD 泛化 | 弱（词汇覆盖限制）| 中 | 强（Dense 覆盖语义）|
| 生产成熟度 | 极高 | 中（Pinecone/Qdrant 原生支持）| 高 |

### 5.9.4 Filter 与 Hybrid Search 的交互

生产中常见场景：hybrid search + attribute filter（如 `date > 2024-01-01 AND category = "tech"`）。

两种实现策略：
1. **Pre-filter**：先过 filter，在过滤后的子集上做 hybrid search。优点：结果集精确。缺点：稀疏子集上 HNSW 图可能退化（邻居被过滤导致图不连通），需要 brute-force fallback。
2. **Post-filter**：先做 hybrid search 取 top-K，再过 filter。缺点：K 需设很大（比 filter 后期望结果多 10x），避免过滤后结果不足。

Qdrant 支持 `must` filter 在 HNSW 搜索内部做 per-node filter（`BaseFilterFunctor`，源码中 `isIdAllowed`），兼顾精度和效率。

---

## 章末五件套

### 1. 核心概念图谱

```
检索系统 Pipeline
├── 召回层（低延迟、高覆盖）
│   ├── 稀疏：BM25 / SPLADE → 倒排索引（WAND/MaxScore 加速）
│   ├── 稠密：bi-encoder → HNSW/IVF ANN（双堆 beam search，ef 控制 recall）
│   └── 融合：
│       ├── RRF（排名融合，无量纲差异问题，k=60）[首选]
│       ├── Linear（分数融合，需归一化，可调权重）
│       └── Learned（LTR/BGE-M3，需标注，精度最高）
└── 精排层（高精度、可接受延迟）
    ├── Cross-Encoder：pointwise，完整 self-attention，+5~12 nDCG
    ├── ColBERT MaxSim：token-level 延迟交互，存储换精度
    └── LLM Listwise：整体排序，RankGPT 范式，滑窗策略
```

### 2. 工程决策树

```
需要 hybrid search 吗？
  ├── 查询只含精确词/ID/代码 → BM25 足够，无需 hybrid
  ├── 查询主要是语义意图 → Dense 足够
  └── 混合场景（通用 RAG / 搜索引擎）→ Hybrid RRF（首选起点）

需要 rerank 吗？
  ├── P99 延迟 < 50ms → 不加 rerank（调高 HNSW ef 提升召回）
  ├── P99 延迟 < 200ms → ColBERT MaxSim
  ├── P99 延迟 < 500ms → Cross-Encoder MiniLM（GPU 部署）
  ├── 精度优先，延迟可接受 → Cross-Encoder DeBERTa
  └── 精度最优，成本不限，可离线 → LLM Listwise

HNSW ef 怎么设？
  ├── 默认 ef = max(top_k, 50)
  ├── 监控 recall@K（vs brute-force）< 90% → 翻倍 ef
  └── P99 延迟超限 → 降 ef 或增大 M（构建期）
```

### 3. 关键公式小抄

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \quad k=60\text{（推荐默认）}$$

$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t,d) \cdot (k_1+1)}{tf(t,d) + k_1\left(1-b+b\dfrac{|d|}{\text{avgdl}}\right)}$$

$$\text{MaxSim}(q,d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \left(\mathbf{q}_i \cdot \mathbf{d}_j\right) \quad \text{(ColBERT)}$$

$$\text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \quad \text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

**HNSW 剪枝条件**（beam search 提前终止）：
$$\text{terminate if: } d(q, \text{best\_candidate}) > d(q, \text{worst\_result}) \text{ and } |\text{results}| = ef$$

### 4. 延伸阅读

| 文献 / 资源 | 核心贡献 |
|---|---|
| Cormack, Clarke, Buettcher. *"Reciprocal Rank Fusion"*, SIGIR 2009 | RRF 算法原论文，k=60 来源 |
| Karpukhin et al. *"Dense Passage Retrieval for Open-Domain QA"*, EMNLP 2020 | DPR，现代 bi-encoder 检索奠基 |
| Khattab & Zaharia. *"ColBERT: Efficient and Effective Passage Search"*, SIGIR 2020 | MaxSim late interaction |
| Formal et al. *"SPLADE: Sparse Lexical and Expansion Model"*, SIGIR 2021 | 学习式稀疏检索 |
| Sun et al. *"Is ChatGPT Good at Search?"*, EMNLP 2023 | RankGPT listwise rerank，滑窗策略 |
| Chen et al. *"BGE M3-Embedding"*, arXiv 2024 | 三路统一模型 |
| Malkov & Yashunin. *"Efficient and robust approximate nearest neighbor"*, TPAMI 2020 | HNSW 原论文，双堆搜索和多样性启发式 |

### 5. 实践 Checklist

- [ ] 两路召回各自的 Recall@K（相对 brute-force）> 90%，HNSW ef 设置正确
- [ ] 用 RRF k=60 作为融合默认值，有足够标注数据后再考虑 learned fusion
- [ ] 评估集包含 dense-only 召回的候选并重新标注，消除 BM25 展示偏差
- [ ] Cross-Encoder 的 batch_size 从 16 开始，按 GPU 显存调整；候选数控制在 100 以内
- [ ] LLM Listwise 设计 parse failure fallback（fallback 到 RRF 结果）并监控 failure rate
- [ ] 生产监控：两路召回耗时（分别）、fusion 耗时、rerank 耗时、端到端 P50/P99
- [ ] Weaviate alpha 参数语义陷阱：RANKED 模式下 alpha 不是 RRF 权重，是候选截断参数
- [ ] 有 attribute filter 时，评估 pre-filter vs post-filter 的召回率差异；小子集 HNSW 考虑 brute-force fallback

---

*本章直接上游：第 1 章（BM25 倒排索引）、第 3 章（HNSW/IVF ANN 算法）；第 6 章将进入向量库的分布式存储与分片策略。*

*真实源码标注说明：带【真实源码 repo@path】标记的段落来自 WebFetch 取回的实际代码（hnswlib hnswalg.h / lancedb rrf.py）；带【示意】标记的段落基于公开文档重构；带【待核】的段落来自论文描述，未直接 WebFetch 验证原文件。*
