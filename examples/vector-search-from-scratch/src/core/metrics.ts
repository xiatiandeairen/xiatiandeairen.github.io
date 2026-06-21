// core/metrics.ts — how we score an index honestly: recall, throughput, memory.
//
// The whole book lives or dies on these three numbers being *measured*, not
// asserted. An ANN index is a deliberate trade: it gives up exactness to win
// speed and/or memory. You cannot reason about that trade without quantifying all
// three axes, so every stage prints recall@k, QPS, and an (estimated) byte cost.

// recall@k: of the k *true* nearest neighbors, what fraction did the approximate
// index actually return (anywhere in its own top-k)?
//
//   recall@k = |approx_topk ∩ truth_topk| / k     (averaged over queries)
//
// Why intersection-over-truth and not order-sensitive: for retrieval, surfacing
// the right *set* of neighbors is what matters downstream (a reranker fixes
// order). recall@10 = 0.9 means "on average the index found 9 of the 10 true
// neighbors." It is NOT precision and NOT NDCG — we measure recall because the
// failure we care about is *missing* good candidates, which an index silently
// does when it prunes too aggressively.
//
// Invariant guarded: if a query's truth list is shorter than k (tiny datasets),
// we divide by that query's actual truth count, not k, so recall stays in [0,1]
// instead of being artificially capped below 1.
export function recallAtK(
  approxIdsPerQuery: number[][],
  truthIdsPerQuery: number[][],
  k: number,
): number {
  if (approxIdsPerQuery.length !== truthIdsPerQuery.length) {
    throw new Error(
      `recallAtK: query count mismatch approx=${approxIdsPerQuery.length} truth=${truthIdsPerQuery.length}`,
    );
  }
  let totalRecall = 0;
  for (let q = 0; q < truthIdsPerQuery.length; q++) {
    const truth = truthIdsPerQuery[q].slice(0, k);
    if (truth.length === 0) continue;
    const approx = new Set(approxIdsPerQuery[q].slice(0, k));
    let hit = 0;
    for (const t of truth) if (approx.has(t)) hit++;
    totalRecall += hit / truth.length; // per-query denom = its own truth size
  }
  return totalRecall / truthIdsPerQuery.length;
}

// Wall-clock a function, returning milliseconds. Uses performance.now() (monotonic,
// sub-ms) rather than Date.now(). Single-shot: callers that need stable numbers
// should run enough work inside `fn` (e.g. all queries) to dwarf JIT warmup and
// GC jitter — stage 07 does a warmup pass before the timed pass for this reason.
export function timeIt(fn: () => void): number {
  const start = performance.now();
  fn();
  return performance.now() - start;
}

// Queries per second from a query count and the elapsed ms it took. The headline
// throughput number. Guards ms===0 (work too fast to measure at ms resolution)
// by returning Infinity rather than dividing by zero — a signal to the caller
// that the batch was too small to time meaningfully.
export function qps(count: number, ms: number): number {
  if (ms === 0) return Infinity;
  return (count / ms) * 1000;
}

// ESTIMATE ONLY — JS object overhead makes true heap cost unknowable without a
// profiler. We model the *payload*: each number is 8 bytes (IEEE-754 double, the
// only number type in JS). A dataset of n vectors × dim is n·dim·8 bytes of raw
// coordinates. Real V8 heap is larger (array headers, boxing, the backing store's
// growth slack) — typically 1.5–3x this for `number[]`. We report the payload
// floor so stages can compare *relative* memory (e.g. PQ codes vs. full vectors)
// apples-to-apples, and we always print "(est.)" next to it so no reader mistakes
// it for a measured RSS.
//
// Overload-ish API: pass either (count, dim) for full float vectors, or
// (count, dim, bytesPerComponent) to model compressed codes (PQ uses 1 byte/code).
export function estimateBytes(
  count: number,
  dim: number,
  bytesPerComponent = 8, // default: JS double
): number {
  return count * dim * bytesPerComponent;
}

// Human-readable byte formatter for the "(est.)" prints. Pure presentation.
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
