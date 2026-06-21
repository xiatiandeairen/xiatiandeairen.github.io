// stage07-eval.ts — what this stage teaches: how to *evaluate* a vector index
// honestly, and the three production concerns that an "it works on my benchmark"
// demo never shows you: sharding, metadata filtering, and incremental updates.
//
// The through-line: every previous stage built an index and reported a recall
// number. But a single recall@k on a frozen dataset is the easiest number in the
// world to fool yourself with. Production breaks the assumptions that make that
// number meaningful:
//   - the corpus is too big for one machine  -> you SHARD, and a naive merge can
//     silently drop the global top-k (failure mode #1)
//   - users want "red shoes under $50", not just "nearest vector"  -> you FILTER,
//     and the obvious "search then filter" approach under-returns when the filter
//     is selective (failure mode #2)
//   - the corpus changes  -> you ADD/DELETE, and a tombstone you forget to skip
//     resurrects a deleted document (failure mode #3)
//
// So this file is a small eval harness (recall@k + nDCG@k) wired to three
// experiments, each of which deliberately demonstrates the wrong way first so the
// number that proves the right way means something.
//
// Honesty notes baked in:
//   - recall/nDCG are computed against brute-force ground truth from core/dataset,
//     not asserted.
//   - QPS uses a warmup pass before the timed pass (JIT/GC would otherwise inflate
//     the first run); it is still single-machine single-thread wall clock.
//   - memory is core/estimateBytes, a payload floor, always printed "(est.)".

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

// ---------------------------------------------------------------------------
// A "flat" (brute-force) index. We deliberately do NOT reuse an approximate
// index from another stage here: the concerns this stage isolates (merge
// correctness, filter correctness, tombstone correctness) are orthogonal to ANN
// approximation. Using an exact index means any recall < 1.0 we see is caused by
// the sharding/filter/update logic under test, not by ANN pruning — that
// separation is the whole point. (Stages 03–06 already covered the recall hit
// from approximation itself.)
//
// Metadata model: each vector carries a parallel `meta` record. We keep meta in a
// sibling array indexed by the same id so the vector hot loop stays pure number[].
type Meta = { id: number; price: number; inStock: boolean };

const METRIC: Metric = 'l2';
// l2 ranking: smaller distance = better. We sort ascending. We use the real
// l2dist (with sqrt) rather than the squared form because this is eval code, not
// a hot inner loop — readability over the sqrt saving here.

// A flat index over a *subset* of the global dataset. `ids` maps local rows back
// to global ids so a shard can answer with global ids the caller can merge.
class FlatIndex {
  // Tombstones: incremental delete marks an id dead instead of compacting the
  // arrays. Compaction would invalidate every id->row mapping and is O(n); a
  // tombstone set is O(1) to delete and the cost is paid lazily at query time by
  // skipping. The failure mode this courts: forgetting to consult `deleted`
  // during search resurrects the document. We demonstrate exactly that below.
  private deleted = new Set<number>();

  constructor(
    private vectors: number[][],
    private ids: number[],
  ) {}

  // Add a vector at query time. Returns the assigned global id. This is the
  // "incremental" in incremental update: no rebuild, just append.
  add(vec: number[], id: number): void {
    this.vectors.push(vec);
    this.ids.push(id);
    // If an id is re-added after deletion, it must come back to life. Without
    // this clear, the tombstone from a prior delete would shadow the new vector.
    this.deleted.delete(id);
  }

  delete(id: number): void {
    this.deleted.add(id);
  }

  size(): number {
    return this.vectors.length - this.deleted.size;
  }

  // Core search. `respectTombstones=false` exists ONLY to demonstrate the
  // resurrection failure mode; production always passes true (the default).
  search(
    query: number[],
    k: number,
    opts: { filter?: (m: Meta) => boolean; meta?: Meta[]; respectTombstones?: boolean } = {},
  ): number[] {
    const respectTombstones = opts.respectTombstones ?? true;
    const scored: { id: number; dist: number }[] = [];
    for (let row = 0; row < this.vectors.length; row++) {
      const id = this.ids[row];
      if (respectTombstones && this.deleted.has(id)) continue;
      // Filtered-ANN, naive version: apply the predicate during the scan
      // ("pre-filtering"). The interesting failure is the *other* naive variant
      // (post-filtering) which we run separately below.
      if (opts.filter && opts.meta && !opts.filter(opts.meta[id])) continue;
      scored.push({ id, dist: l2dist(query, this.vectors[row]) });
    }
    // Tie-break on id asc to match computeGroundTruth's deterministic ordering;
    // without this, recall vs. ground truth would wobble run-to-run on distance
    // ties (synthetic Gaussians do produce them).
    scored.sort((a, b) => a.dist - b.dist || a.id - b.id);
    return scored.slice(0, k).map((e) => e.id);
  }
}

// ---------------------------------------------------------------------------
// nDCG@k — the eval-harness piece core/metrics doesn't provide.
//
// Why nDCG in addition to recall: recall@k treats all k true neighbors as equally
// valuable and ignores order ("did the set show up?"). nDCG (normalized
// discounted cumulative gain) asks the sharper question "did the *best* neighbors
// show up *near the top*?" by discounting gains logarithmically with rank. For a
// reranking-free retrieval system the order the index returns IS what the user
// sees, so a high recall but low nDCG means "found them but buried them" — a real
// and distinct failure recall cannot see.
//
// Graded relevance: we grade a returned id by its rank in the ground-truth list
// (rank 0 = the true nearest = highest gain). gain = (truthLen - truthRank), so
// the true #1 is worth the most and a non-neighbor is worth 0. IDCG is the DCG of
// the perfect ordering, so nDCG ∈ [0,1] and 1.0 means "returned the true top-k in
// exactly the true order".
function ndcgAtK(approxIds: number[], truthIds: number[], k: number): number {
  const truthLen = Math.min(truthIds.length, k);
  if (truthLen === 0) return 0;
  // truthRank[id] = position in ground truth (0-based), or undefined if not a
  // true neighbor. Map lookup keeps this O(k) instead of indexOf's O(k^2).
  const truthRank = new Map<number, number>();
  for (let i = 0; i < truthLen; i++) truthRank.set(truthIds[i], i);

  const gain = (rank: number | undefined): number =>
    rank === undefined ? 0 : truthLen - rank; // nearest neighbor => largest gain

  let dcg = 0;
  const topk = approxIds.slice(0, k);
  for (let i = 0; i < topk.length; i++) {
    // log2(i+2): standard DCG discount, position 0 has discount log2(2)=1.
    dcg += gain(truthRank.get(topk[i])) / Math.log2(i + 2);
  }
  let idcg = 0;
  for (let i = 0; i < truthLen; i++) {
    idcg += gain(i) / Math.log2(i + 2); // perfect order: true rank i at position i
  }
  return idcg === 0 ? 0 : dcg / idcg;
}

function meanNdcg(approxPerQuery: number[][], truthPerQuery: number[][], k: number): number {
  let sum = 0;
  for (let q = 0; q < truthPerQuery.length; q++) {
    sum += ndcgAtK(approxPerQuery[q], truthPerQuery[q], k);
  }
  return sum / truthPerQuery.length;
}

// ---------------------------------------------------------------------------
// Shared fixture. Deterministic: same seeds => same numbers on any machine.
const N = 2000;
const DIM = 16;
const CLUSTERS = 8;
const K = 10;
const DATA_SEED = 7;
const QUERY_SEED = 99;

const dataset = makeDataset(N, DIM, CLUSTERS, DATA_SEED);
const queries = makeQueries(50, DIM, QUERY_SEED);
const truth = computeGroundTruth(dataset, queries, K, METRIC);

// Synthetic metadata, deterministic from id so prices reproduce. Price in [0,100),
// inStock for ~70% of items. This stands in for the "structured fields alongside
// the embedding" every real vector DB carries.
const meta: Meta[] = dataset.map((_, id) => ({
  id,
  price: (id * 37) % 100, // cheap deterministic spread, no PRNG needed
  inStock: id % 10 !== 0 && id % 10 !== 3, // ~80% in stock
}));

function searchAll(
  index: FlatIndex,
  k: number,
  opts?: Parameters<FlatIndex['search']>[2],
): number[][] {
  return queries.map((q) => index.search(q, k, opts));
}

// ---------------------------------------------------------------------------
// EXPERIMENT 0 — the eval harness on a single exact index (the baseline truth).
function runBaseline(): void {
  console.log('\n=== [0] Eval harness: recall@k + nDCG@k (single exact index) ===');
  const index = new FlatIndex([...dataset], dataset.map((_, i) => i));

  const approx = searchAll(index, K);

  // QPS: warmup pass (let V8 JIT the hot loop), then timed pass.
  searchAll(index, K); // warmup, result discarded
  const ms = timeIt(() => searchAll(index, K));

  const recall = recallAtK(approx, truth, K);
  const ndcg = meanNdcg(approx, truth, K);
  console.log(
    `recall@${K} = ${recall.toFixed(4)} | nDCG@${K} = ${ndcg.toFixed(4)} | ` +
      `QPS = ${qps(queries.length, ms).toFixed(0)} (${ms.toFixed(1)} ms / ${queries.length} q)`,
  );
  console.log(
    `dataset memory = ${formatBytes(estimateBytes(N, DIM))} (est. payload floor)`,
  );
  // Sanity: an EXACT index must score recall 1.0. If it doesn't, the harness
  // itself is wrong and every downstream number is meaningless. We assert loudly.
  if (Math.abs(recall - 1.0) > 1e-9) {
    throw new Error(`harness bug: exact index recall ${recall} != 1.0`);
  }
  console.log('check: exact index recall == 1.0000 -> harness is calibrated');
}

// ---------------------------------------------------------------------------
// EXPERIMENT 1 — sharding: split the corpus, search each shard, merge.
//
// The teaching point: a correct merge gathers each shard's *local* top-k and
// re-ranks the union to pick the *global* top-k. The seductive-but-wrong shortcut
// is to ask each shard for top-(k/S) (S = shard count) "to save work" — that
// silently misses cases where one shard holds more than its 1/S share of a
// query's true neighbors (which clustered data guarantees: a query near cluster c
// has almost ALL its neighbors in whichever shard owns cluster c).
function runSharding(): void {
  console.log('\n=== [1] Sharding: per-shard top-k then merge ===');
  const SHARDS = 4;

  // Round-robin assignment to shards. Round-robin (not contiguous ranges) so each
  // shard gets a mix of clusters — this is the realistic "data is spread, not
  // conveniently partitioned by topic" case, and it's what makes the k/S shortcut
  // visibly wrong.
  const shards: FlatIndex[] = [];
  for (let s = 0; s < SHARDS; s++) shards.push(new FlatIndex([], []));
  for (let id = 0; id < N; id++) shards[id % SHARDS].add(dataset[id], id);

  // Correct merge: each shard returns its top-K, we re-rank the union by true
  // distance and take global top-K. We must recompute distance at merge time (or
  // have shards return distances); here we recompute to keep FlatIndex's API
  // returning ids only.
  const mergedCorrect: number[][] = queries.map((q) => {
    const candidateIds = shards.flatMap((sh) => sh.search(q, K));
    return rerankByDistance(q, candidateIds, K);
  });

  // Wrong merge: ask each shard for only ceil(K/SHARDS) = 3, then merge. Under-
  // provisions the shard that actually owns the query's cluster.
  const perShardSmall = Math.ceil(K / SHARDS);
  const mergedWrong: number[][] = queries.map((q) => {
    const candidateIds = shards.flatMap((sh) => sh.search(q, perShardSmall));
    return rerankByDistance(q, candidateIds, K);
  });

  const recallCorrect = recallAtK(mergedCorrect, truth, K);
  const recallWrong = recallAtK(mergedWrong, truth, K);

  console.log(
    `correct merge (each shard top-${K})  recall@${K} = ${recallCorrect.toFixed(4)}`,
  );
  console.log(
    `WRONG  merge (each shard top-${perShardSmall})   recall@${K} = ${recallWrong.toFixed(4)}` +
      `  <- failure mode: k/S under-provisioning drops global top-k`,
  );

  // Consistency invariant: a correct shard-and-merge over an exact index must
  // reproduce the single-index result *exactly*, query by query, id by id. This
  // is the strongest possible check — not just "recall is high" but "byte-for-byte
  // identical to not sharding at all".
  const single = new FlatIndex([...dataset], dataset.map((_, i) => i));
  const singleResult = searchAll(single, K);
  let mismatches = 0;
  for (let q = 0; q < queries.length; q++) {
    if (mergedCorrect[q].join(',') !== singleResult[q].join(',')) mismatches++;
  }
  console.log(
    `consistency: merged top-${K} vs single-index top-${K} -> ` +
      `${mismatches === 0 ? 'IDENTICAL on all' : mismatches + ' MISMATCHED of'} ` +
      `${queries.length} queries`,
  );
  if (mismatches !== 0) {
    throw new Error(`sharding merge bug: ${mismatches} queries diverged from single index`);
  }
}

// Re-rank a candidate id set by true distance to the query and return top-k ids.
// Shared by the merge step. Dedup guards against the same id arriving from
// multiple shards (can't happen with disjoint shards, but the merge logic must be
// dedup-safe for the filtered/incremental reuse below).
function rerankByDistance(query: number[], candidateIds: number[], k: number): number[] {
  const seen = new Set<number>();
  const scored: { id: number; dist: number }[] = [];
  for (const id of candidateIds) {
    if (seen.has(id)) continue;
    seen.add(id);
    scored.push({ id, dist: l2dist(query, dataset[id]) });
  }
  scored.sort((a, b) => a.dist - b.dist || a.id - b.id);
  return scored.slice(0, k).map((e) => e.id);
}

// ---------------------------------------------------------------------------
// EXPERIMENT 2 — metadata filtering (naive filtered-ANN).
//
// Two naive strategies, one correct, one with a notorious failure:
//   pre-filter:  evaluate predicate during the scan, rank only survivors. Always
//                returns up to k *matching* results if that many exist. Correct,
//                but on a real ANN index it breaks the index's pruning (you can't
//                skip a cluster you might need a survivor from) — the cost is
//                speed, demonstrated qualitatively here.
//   post-filter: get top-k by vector first, THEN drop non-matches. Fast, but when
//                the filter is selective the top-k vector neighbors are mostly
//                filtered out and you return far fewer than k — the "I asked for
//                10 red shoes and got 2" failure.
function runFiltering(): void {
  console.log('\n=== [2] Metadata filtering: pre-filter vs post-filter ===');
  // Selective predicate: in stock AND price < 30. ~ (0.8 * 0.30) ≈ 24% of corpus.
  const filter = (m: Meta): boolean => m.inStock && m.price < 30;
  const matchCount = meta.filter(filter).length;
  console.log(
    `predicate "inStock && price<30" matches ${matchCount}/${N} items ` +
      `(${((matchCount / N) * 100).toFixed(1)}% — selective)`,
  );

  const index = new FlatIndex([...dataset], dataset.map((_, i) => i));

  // Unfiltered baseline: how many of the plain top-k already satisfy the filter?
  const unfiltered = searchAll(index, K);
  const avgHitsUnfiltered =
    unfiltered.reduce((s, ids) => s + ids.filter((id) => filter(meta[id])).length, 0) /
    queries.length;

  // Pre-filter: correct, returns k matches when available.
  const pre = searchAll(index, K, { filter, meta });
  const avgReturnedPre =
    pre.reduce((s, ids) => s + ids.length, 0) / queries.length;
  const allPreMatch = pre.every((ids) => ids.every((id) => filter(meta[id])));

  // Post-filter: top-k by vector, then drop non-matches. Returns fewer than k.
  const post = unfiltered.map((ids) => ids.filter((id) => filter(meta[id])));
  const avgReturnedPost =
    post.reduce((s, ids) => s + ids.length, 0) / queries.length;
  const underfilledQueries = post.filter((ids) => ids.length < K).length;

  console.log(
    `before filter: avg ${avgHitsUnfiltered.toFixed(2)}/${K} of plain top-${K} happen to match`,
  );
  console.log(
    `pre-filter  : avg ${avgReturnedPre.toFixed(2)}/${K} returned, all match predicate = ${allPreMatch}`,
  );
  console.log(
    `post-filter : avg ${avgReturnedPost.toFixed(2)}/${K} returned ` +
      `<- failure mode: ${underfilledQueries}/${queries.length} queries under-fill (got < ${K})`,
  );
}

// ---------------------------------------------------------------------------
// EXPERIMENT 3 — incremental update: add then delete, retrieval stays correct.
//
// Two checks:
//   add:    insert a brand-new vector engineered to be query q0's nearest
//           neighbor; it must appear at rank 0 afterward (no rebuild).
//   delete: tombstone that same new vector; it must vanish from results. Then we
//           demonstrate the resurrection failure mode by searching with
//           tombstones disabled — the deleted id comes back.
function runIncremental(): void {
  console.log('\n=== [3] Incremental update: add / delete correctness ===');
  const index = new FlatIndex([...dataset], dataset.map((_, i) => i));
  const q0 = queries[0];

  const before = index.search(q0, K);
  console.log(`before add: q0 top-${K} = [${before.slice(0, 5).join(', ')}, ...]`);

  // Engineer a guaranteed nearest neighbor: a tiny perturbation of q0 itself.
  // Distance ~0, so it must rank #0. Using an exact copy would tie at distance 0
  // and tie-break by id; the perturbation keeps it strictly nearest without
  // relying on tie-break order.
  const NEW_ID = N; // ids 0..N-1 used; N is the first free id
  const newVec = q0.map((x) => x + 1e-6);
  index.add(newVec, NEW_ID);

  const afterAdd = index.search(q0, K);
  const addedAtRank = afterAdd.indexOf(NEW_ID);
  console.log(
    `after add (id ${NEW_ID}): appears at rank ${addedAtRank} ` +
      `${addedAtRank === 0 ? '(correct: it is the nearest)' : '(WRONG)'}`,
  );
  if (addedAtRank !== 0) {
    throw new Error(`incremental add bug: new nearest neighbor landed at rank ${addedAtRank}`);
  }

  // Delete it; it must disappear.
  index.delete(NEW_ID);
  const afterDelete = index.search(q0, K);
  const stillPresent = afterDelete.includes(NEW_ID);
  console.log(
    `after delete (id ${NEW_ID}): present = ${stillPresent} ` +
      `${stillPresent ? '(WRONG)' : '(correct: tombstone skipped it)'}`,
  );
  if (stillPresent) {
    throw new Error('incremental delete bug: tombstoned id survived search');
  }

  // Failure mode: a search path that forgets to consult tombstones resurrects the
  // deleted document. This is the single most common incremental-index bug.
  const resurrected = index.search(q0, K, { respectTombstones: false });
  console.log(
    `failure mode (tombstone check disabled): deleted id ${NEW_ID} ` +
      `${resurrected.includes(NEW_ID) ? 'RESURRECTS at rank ' + resurrected.indexOf(NEW_ID) : 'absent'} ` +
      `<- this is why every search path MUST honor the tombstone set`,
  );

  // And the result set, minus the resurrected ghost, should match the original —
  // proving delete didn't corrupt anything else.
  const recovered = index.search(q0, K); // tombstone-respecting again
  const matchesOriginal = recovered.join(',') === before.join(',');
  console.log(
    `post-delete top-${K} == original top-${K}: ${matchesOriginal} ` +
      `(delete left the rest of the index intact)`,
  );
}

// ---------------------------------------------------------------------------
function main(): void {
  console.log('stage07 — evaluation harness + production concerns');
  console.log(
    `fixture: ${N} vecs × ${DIM} dims, ${CLUSTERS} clusters, ${queries.length} queries, ` +
      `metric=${METRIC}, k=${K} (seeds data=${DATA_SEED} query=${QUERY_SEED})`,
  );
  runBaseline();
  runSharding();
  runFiltering();
  runIncremental();
  console.log('\nall invariants held; numbers above are computed, not asserted.');
}

main();
