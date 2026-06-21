// stage03-bruteforce.ts — the brute-force KNN baseline that every later index
// is judged against.
//
// What this stage teaches:
//   1. Brute force is the *definition* of correct: it scans all N vectors, so its
//      recall@k against the same-metric ground truth is exactly 1.0 — there is
//      nowhere for a true neighbor to hide. This makes it the recall *ceiling*
//      and the speed *floor* simultaneously.
//   2. Its cost is O(N · dim) distance computations *per query*. There is no
//      index, no pruning, no precomputation — so QPS falls ~linearly as N grows.
//      We prove that empirically by timing the SAME query batch against datasets
//      of 1k / 5k / 20k / 50k and printing the QPS curve.
//   3. Why every approximate method (IVF, PQ, HNSW in later stages) exists: the
//      linear scan that is fine at N=1k is unacceptable at N=10M. The approximate
//      methods buy sub-linear query cost by giving up some of this stage's
//      guaranteed recall=1.0. You cannot judge that trade without this baseline.
//
// Honesty note on the numbers: QPS here is single-machine, single-thread,
// plain `number[]` (not Float32Array), and the dataset is synthetic clustered
// Gaussians (easier than real embeddings). Absolute QPS will differ on the
// reader's machine; the *shape* of the curve (QPS ∝ 1/N) is the transferable
// lesson, and we print the measured per-N ratio so you can check linearity.

import { l2dist } from './core/vec.js';
import {
  makeDataset,
  makeQueries,
  computeGroundTruth,
  type Metric,
} from './core/dataset.js';
import {
  recallAtK,
  timeIt,
  qps,
  estimateBytes,
  formatBytes,
} from './core/metrics.js';

// Fixed experiment knobs. dim/clusters/seeds are constants (not magic numbers in
// the loop) so every printed number is reproducible and the only thing varying
// across runs is N — that is what isolates the "QPS vs N" relationship.
const DIM = 16;
const CLUSTERS = 8;
const DATASET_SEED = 42;
const QUERY_SEED = 7;
const NUM_QUERIES = 50;
const K = 10;
const METRIC: Metric = 'l2';

// Brute-force top-k for one query: score every dataset vector, keep the k best.
//
// Why we sort all N instead of a bounded heap: at the N in this stage (≤50k) the
// full sort is simpler and the constant factor is dwarfed by the N distance
// computations, which dominate. A real engine would use a partial selection
// (quickselect / bounded max-heap) to get top-k in O(N log k) instead of
// O(N log N) — we leave that on the table on purpose so the *mechanism* (one
// distance per vector, then pick smallest) is visible. The distance count, not
// the sort, is what later indexes attack.
//
// Tie-break (smaller index wins on equal distance) is kept identical to
// computeGroundTruth's, so a correct brute-force scan reproduces the ground
// truth *exactly* — including ties. If we tie-broke differently, recall could
// dip below 1.0 purely from ordering, which would be a measurement artifact, not
// a real miss. That would defeat the whole point of the baseline.
function bruteForceTopK(
  dataset: number[][],
  query: number[],
  k: number,
): number[] {
  const scored = dataset.map((vec, idx) => ({ idx, dist: l2dist(query, vec) }));
  scored.sort((a, b) => a.dist - b.dist || a.idx - b.idx);
  return scored.slice(0, k).map((e) => e.idx);
}

// Run the whole query batch and collect each query's approximate (here: exact)
// top-k. Pulled into its own function so we can hand exactly this closure to
// timeIt — the timed region is "answer all NUM_QUERIES queries", nothing else
// (no allocation of the dataset, no ground-truth build), so the QPS reflects
// query work alone.
function searchAll(
  dataset: number[][],
  queries: number[][],
  k: number,
): number[][] {
  return queries.map((q) => bruteForceTopK(dataset, q, k));
}

function runForSize(n: number): {
  n: number;
  recall: number;
  qpsValue: number;
  msPerBatch: number;
  bytesEst: number;
} {
  const dataset = makeDataset(n, DIM, CLUSTERS, DATASET_SEED);
  const queries = makeQueries(NUM_QUERIES, DIM, QUERY_SEED);

  // Ground truth is itself a brute-force scan (see core/dataset.ts). We compute
  // it separately and OUTSIDE the timed region: recall is a correctness check,
  // not part of the throughput we are measuring.
  const truth = computeGroundTruth(dataset, queries, K, METRIC);

  // Warmup pass (result discarded): the first run pays V8 JIT compilation of the
  // hot distance loop. Without this, the smallest-N timing is inflated by
  // warmup, which would falsely flatten the QPS-vs-N curve at the low end.
  searchAll(dataset, queries, K);

  let results: number[][] = [];
  const ms = timeIt(() => {
    results = searchAll(dataset, queries, K);
  });

  const recall = recallAtK(results, truth, K);
  const qpsValue = qps(NUM_QUERIES, ms);
  const bytesEst = estimateBytes(n, DIM);

  return { n, recall, qpsValue, msPerBatch: ms, bytesEst };
}

// --- demo 1: the QPS-vs-N curve (the headline of this stage) -----------------
const SIZES = [1000, 5000, 20000, 50000];

console.log('=== Stage 03: 暴力 KNN 基线 (brute-force baseline) ===');
console.log(
  `配置: dim=${DIM}, clusters=${CLUSTERS}, queries=${NUM_QUERIES}, k=${K}, metric=${METRIC}`,
);
console.log(
  '不变量: 暴力扫全量 ⇒ recall@k 必为 1.0000 (与同度量真值无差); 这是召回上限、速度下限。\n',
);

console.log(
  'N\trecall@10\tbatch(ms)\tQPS\t\t内存(est.)',
);
console.log('-'.repeat(72));

const rows = SIZES.map(runForSize);
for (const r of rows) {
  console.log(
    `${r.n}\t${r.recall.toFixed(4)}\t\t${r.msPerBatch.toFixed(1)}\t\t${r.qpsValue.toFixed(0)}\t\t${formatBytes(r.bytesEst)}`,
  );
}

// --- linearity check: QPS should scale ~1/N --------------------------------
// We don't just claim "linear", we print the evidence. If brute force is O(N),
// then QPS(N) · N should be roughly constant (it's queries · 1000 / (N · perVec
// cost)). We normalize against the N=1000 row and show how close QPS·N stays to
// constant. Deviation comes from cache effects, GC, and the O(N log N) sort tail
// — honest, not hidden.
console.log('\n--- 线性性验证: 若 O(N), 则 QPS×N 应近似常数 ---');
const base = rows[0];
const baseProduct = base.qpsValue * base.n;
console.log('N\tQPS×N\t\t相对 N=1000 (理想=1.00)');
for (const r of rows) {
  const product = r.qpsValue * r.n;
  const ratio = product / baseProduct;
  console.log(
    `${r.n}\t${(product / 1e6).toFixed(2)}M\t\t${ratio.toFixed(2)}`,
  );
}
console.log(
  '注: QPS 从 N=1000 到 N=50000 (50×) 的下降倍数 = ' +
    `${(base.qpsValue / rows[rows.length - 1].qpsValue).toFixed(1)}× (理想线性应≈50×)`,
);

// --- demo 2: failure mode (NOT a happy-path-only demo) ----------------------
// Brute force never under-recalls — so what *is* its failure? It fails on cost.
// Below we make that concrete two ways:
//
// (a) Latency blowup: per-query latency at the largest N, contrasted with a
//     realistic SLA. Even a tiny synthetic 50k set already costs real
//     milliseconds per query single-threaded; multiply by real dim (768) and
//     real N (10M) and brute force is hopeless. This is *why* the rest of the
//     book exists.
console.log('\n--- 失败模式 (a): 成本爆炸, 不是召回 ---');
const largest = rows[rows.length - 1];
const msPerQuery = largest.msPerBatch / NUM_QUERIES;
console.log(
  `N=${largest.n} 时每查询 ${msPerQuery.toFixed(3)} ms ⇒ 单线程 QPS≈${largest.qpsValue.toFixed(0)}`,
);
// Extrapolate to production scale. Clearly labeled as an ESTIMATE (linear
// extrapolation of measured per-vector cost), not a measurement — we are honest
// about which numbers are real and which are projected.
const measuredPerVecMs = msPerQuery / largest.n; // ms to score one vector
const PROD_N = 10_000_000;
const PROD_DIM = 768;
const dimScale = PROD_DIM / DIM; // distance cost ∝ dim
const projMsPerQuery = measuredPerVecMs * PROD_N * dimScale;
console.log(
  `外推 (估算, 线性): N=10M, dim=768 ⇒ 每查询 ≈ ${projMsPerQuery.toFixed(0)} ms ` +
    `(≈${(projMsPerQuery / 1000).toFixed(1)} s/查询), QPS≈${(1000 / projMsPerQuery).toFixed(2)}`,
);
console.log(
  '  ⇒ 暴力法在生产规模下单查询要数秒。这就是 IVF/PQ/HNSW 存在的理由 (后续 stage)。',
);

// (b) Metric-mismatch failure: brute force is only "exact" w.r.t. the metric it
//     scans with. If you build ground truth under cosine but query under l2 (a
//     real bug when a team swaps metrics without re-indexing), recall collapses
//     even though every vector was scanned. Demonstrates that "scanned all" ≠
//     "correct" — the metric must match. This is a silent, dangerous failure: no
//     error is thrown, results just quietly become wrong.
console.log('\n--- 失败模式 (b): 度量不匹配 ⇒ 扫全量也错 (静默!) ---');
const smallDs = makeDataset(2000, DIM, CLUSTERS, DATASET_SEED);
const smallQ = makeQueries(NUM_QUERIES, DIM, QUERY_SEED);
const cosineTruth = computeGroundTruth(smallDs, smallQ, K, 'cosine');
const l2Results = searchAll(smallDs, smallQ, K); // brute force, but under l2
const mismatchRecall = recallAtK(l2Results, cosineTruth, K);
const matchedTruth = computeGroundTruth(smallDs, smallQ, K, 'l2');
const matchedRecall = recallAtK(searchAll(smallDs, smallQ, K), matchedTruth, K);
console.log(
  `同一份 l2 暴力结果: vs l2 真值 recall=${matchedRecall.toFixed(4)} (匹配, 满分) | ` +
    `vs cosine 真值 recall=${mismatchRecall.toFixed(4)} (度量不匹配, 暴跌)`,
);
console.log(
  '  ⇒ 暴力的 recall=1.0 只对它扫描所用度量成立; 换度量不重建索引会静默返回错答案。',
);

console.log('\n=== 基线就绪: recall 上限=1.0, 速度随 N 线性下降, 后续近似法以此为对照 ===');
