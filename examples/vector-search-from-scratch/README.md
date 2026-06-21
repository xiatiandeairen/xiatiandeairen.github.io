# vector-search-from-scratch

从零用 TypeScript 手写一个向量检索引擎，配套《向量检索引擎》书库合集。

**特点**：

- **纯算法，离线可跑**。零运行时依赖，不需要 LLM、API key 或网络。数据由种子 PRNG 合成（`mulberry32` + Box–Muller 聚类高斯），同一台机器、同一次运行、不同次运行都 bit-for-bit 可复现。
- **诚实数字**。recall@k / QPS / 内存全部由代码真算、真测、真打印出来。内存是按 IEEE-754 double payload 估的下界（标注 `(est.)`，不是测出来的 RSS）。
- **反 survey**。每个 stage 不只跑 happy path，还 demo 失败模式（index 漏召回、PQ 量化误差、过度剪枝塌方等）。
- 注释讲 why / 不变量 / 失败模式，不复述 what。

## 跑法

先安装（一次）：

```bash
npm install
npm run typecheck   # 确认类型干净
```

各 stage（每个文件可独立 `npx tsx` 直接跑，加载即跑 main）：

```bash
npm run stage01   # 暴力线性扫描：建立 recall=1.0 / QPS 基线，理解为什么需要 ANN
npm run stage02   # 距离度量与归一化：cosine vs L2，normalize 一次省两个 sqrt
npm run stage03   # IVF 倒排：聚类分桶 + nprobe，recall/QPS 权衡曲线
npm run stage04   # PQ 乘积量化：码本压缩，内存换 recall
npm run stage05   # HNSW 图：可导航小世界，对数级查询
npm run stage06   # 混合过滤：标量过滤 + 向量检索，pre/post-filter 取舍
npm run stage07   # 全量基准：所有索引同台对比 recall/QPS/内存
```

> 说明：合成数据是球状、良分离的聚类高斯，比真实 embedding **更容易**。这里的 recall 绝对值偏乐观；可迁移的是**相对趋势**（nprobe↑ ⇒ recall↑/QPS↓，PQ 用 recall 换内存）。

## core 模块

后续 stage 复用，不重造：

- `src/core/vec.ts` — `dot` / `cosineSim` / `l2dist` / `normalize`
- `src/core/dataset.ts` — `mulberry32` / `makeDataset` / `makeQueries` / `computeGroundTruth`
- `src/core/metrics.ts` — `recallAtK` / `timeIt` / `qps` / `estimateBytes` / `formatBytes`
