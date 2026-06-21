// core/dataset.ts — deterministic synthetic data so every number in the book
// reproduces bit-for-bit.
//
// Why synthetic instead of a real embedding dump (SIFT/GloVe/etc.): a book that
// claims "run this and you'll see recall 0.94" must produce *exactly* 0.94 on
// the reader's laptop. Real datasets are large, license-encumbered, and need a
// download step — the opposite of "纯算法离线可跑". So we generate clustered
// Gaussian data from a seeded PRNG. The clustering matters: uniformly random
// vectors have no neighborhood structure, which would make every ANN index look
// equally (badly) like brute force. Real embeddings cluster by topic; our
// clusters stand in for that, so IVF/HNSW have something real to exploit.
//
// The honesty caveat (stated plainly in stage 07): synthetic clustered Gaussians
// are *easier* than real embeddings — clusters are spherical and well-separated.
// Absolute recall numbers here are optimistic vs. production data; the *relative*
// trends (IVF nprobe↑ ⇒ recall↑ ⇒ QPS↓, PQ trades recall for memory) are what
// transfers.

import { l2dist, cosineSim } from './vec.js';

export type Metric = 'l2' | 'cosine';

// mulberry32: a tiny, fast, well-distributed 32-bit PRNG. We need our *own*
// seeded generator because Math.random() is unseeded and would break
// reproducibility. mulberry32 is ~10 lines, passes basic statistical tests, and
// is the standard choice for deterministic demos. Returns a function yielding
// floats in [0, 1).
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0; // force uint32; negative/float seeds would corrupt the state
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box–Muller: turn two uniforms into one standard-normal sample. We need
// Gaussian noise around cluster centers; uniform noise would make square-ish
// clusters with hard edges, unlike the soft, roughly-Gaussian blobs real
// embeddings form. We only keep one of the two values Box–Muller produces —
// simpler, and the PRNG is cheap.
function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng(); // log(0) is -Infinity; resample the (measure-zero) 0
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Build `n` vectors of dimension `dim` grouped into `clusters` Gaussian blobs.
// Construction: pick `clusters` random centers in a unit-ish box, then scatter
// points around them with a spread small relative to inter-center distance so the
// clusters are recoverable but overlap at the edges (realistic — neighborhoods
// blur). Cluster assignment is round-robin so cluster sizes are balanced and
// deterministic regardless of n.
export function makeDataset(
  n: number,
  dim: number,
  clusters: number,
  seed: number,
): number[][] {
  const rng = mulberry32(seed);
  // Spread the centers across a 10-wide box; spread=1.0 keeps blobs ~10x tighter
  // than center separation, so a point's true nearest neighbors are almost always
  // in its own cluster — exactly the structure IVF/HNSW are designed to exploit.
  const centerScale = 10;
  const spread = 1.0;
  const centers: number[][] = [];
  for (let c = 0; c < clusters; c++) {
    const center = new Array<number>(dim);
    for (let d = 0; d < dim; d++) center[d] = rng() * centerScale;
    centers.push(center);
  }
  const data: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const center = centers[i % clusters]; // round-robin ⇒ balanced, deterministic
    const vec = new Array<number>(dim);
    for (let d = 0; d < dim; d++) vec[d] = center[d] + gaussian(rng) * spread;
    data[i] = vec;
  }
  return data;
}

// Query vectors drawn from the *same distribution shape* but a different seed,
// so queries land near (but not on top of) dataset points — the realistic case
// where the answer is "close to several candidates" rather than an exact hit.
// We reuse the same center layout by seeding centers identically to makeDataset's
// default, then jitter more (spread 1.5) so queries sit between clusters often
// enough to make the index work for its recall.
export function makeQueries(m: number, dim: number, seed: number): number[][] {
  const rng = mulberry32(seed);
  // Queries roam the same box but with wider spread, so some fall in cluster
  // overlap zones — these are the queries that expose an index's recall ceiling.
  const centerScale = 10;
  const spread = 1.5;
  const queryClusters = 8;
  const centers: number[][] = [];
  for (let c = 0; c < queryClusters; c++) {
    const center = new Array<number>(dim);
    for (let d = 0; d < dim; d++) center[d] = rng() * centerScale;
    centers.push(center);
  }
  const queries: number[][] = new Array(m);
  for (let i = 0; i < m; i++) {
    const center = centers[i % queryClusters];
    const q = new Array<number>(dim);
    for (let d = 0; d < dim; d++) q[d] = center[d] + gaussian(rng) * spread;
    queries[i] = q;
  }
  return queries;
}

// Exact top-k by brute force — the *ground truth* against which every ANN index's
// recall is measured. This is O(n·m·dim) and deliberately slow; it is the thing
// the rest of the book tries to avoid at query time, but at *eval* time we accept
// the cost because correctness of the yardstick is non-negotiable.
//
// Returns, per query, the indices of the k nearest dataset vectors, best first.
// Tie-breaking: stable on index (lower index wins ties) so ground truth is fully
// deterministic; this matters because float distances do collide on synthetic data.
export function computeGroundTruth(
  dataset: number[][],
  queries: number[][],
  k: number,
  metric: Metric,
): number[][] {
  // For cosine, larger = better; for l2, smaller = better. We unify by computing
  // a "score" where larger is always better, so the same sort works for both.
  const score = (a: number[], b: number[]): number =>
    metric === 'cosine' ? cosineSim(a, b) : -l2dist(a, b);

  return queries.map((q) => {
    const scored = dataset.map((v, idx) => ({ idx, s: score(q, v) }));
    // Sort by score desc, then index asc for deterministic tie-breaking.
    scored.sort((x, y) => (y.s - x.s) || (x.idx - y.idx));
    return scored.slice(0, k).map((e) => e.idx);
  });
}
