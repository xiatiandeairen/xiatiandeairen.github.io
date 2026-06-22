# embeddings-from-scratch — 《表示学习与 Embedding 从零》配套代码

从零用纯 TypeScript 手写表示学习与 embedding 的核心机制：从 one-hot 的维度灾难，到共现/PMI，到 skip-gram + 负采样，到对比学习（InfoNCE / 温度 / 负样本数），再到评估（类比题）与降维可视化（PCA / t-SNE）。

**特性**

- **纯算法、离线、CPU 可跑**：自带微型自动微分引擎（`src/core/autograd.ts`），不依赖 PyTorch / 任何 ML 框架，不需要 LLM / API key / 网络。
- **确定性**：所有随机来自种子 PRNG（`src/core/rng.ts`），全程禁用 `Math.random`，同 seed bit-for-bit 可复现。
- **诚实数字**：每个 stage 真训出 loss 下降、真测指标并画成 console 图。toy 词表规模下绝对值偏乐观，可迁移的是机制与曲线形状——书中每处标注。
- **demo 失败模式**：不只 happy path，专门演示 collapse / NaN / 死单元 / 学不出不存在的结构等。

## 运行

```bash
npm install          # 仅 devDependencies（tsx / typescript / @types/node）
npm run typecheck    # tsc --noEmit，校验类型干净

npm run stage01      # one-hot 与维度灾难：为什么需要稠密表示
npm run stage02      # 共现矩阵与 PMI：第一个能学出语义的 count-based 表示
npm run stage03      # skip-gram + softmax：用自动微分真训词向量
npm run stage04      # 负采样：把 O(V) softmax 降为 O(k)，unigram^0.75 采样
npm run stage05      # InfoNCE：对比学习的统一视角
npm run stage06      # 温度与负样本数：alignment / uniformity 与 collapse
npm run stage07      # 评估：最近邻 + 类比题准确率（king-man+woman）
npm run stage08      # 降维可视化：PCA 与 t-SNE，console 散点
```

每个 stage 文件独立可跑（`npx tsx src/stageNN-*.ts`），import `src/core/*` 共享底座，互不 import（避免 main() 连环触发）。

## 共享底座 `src/core/`

| 文件 | 职责 |
|------|------|
| `rng.ts` | 种子 PRNG + uniform/gaussian/categorical/shuffle，全书唯一随机源 |
| `autograd.ts` | 标量反向自动微分（Value 节点 + 拓扑 backward）+ Vec/Mat 薄壳 |
| `optim.ts` | SGD / Adam，作用在 Value 参数上，带 zeroGrad/step |
| `plot.ts` | asciiLine / asciiBar / asciiHeatmap / asciiScatter |
| `text.ts` | 带语义聚类的 toy 语料生成 + 分词 + 共现窗口枚举 |
| `eval.ts` | cosine/euclidean + nearestNeighbors + analogy + alignment/uniformity |
