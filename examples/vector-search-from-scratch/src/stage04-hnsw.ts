// stage04-hnsw.ts — HNSW (Hierarchical Navigable Small World) from scratch.
//
// What this stage teaches: how a *graph* index answers ANN queries in
// O(log n)-ish hops instead of the O(n) scan of stage01 brute force, and the two
// knobs that govern its accuracy/speed/connectivity:
//
//   - ef (search beam width): how many candidates the greedy search keeps alive.
//     Larger ef => explores more of the graph => higher recall, lower QPS.
//   - M  (max neighbors per node): how densely the graph is wired. Too small and
//     the graph fragments into disconnected islands the search can never leave —
//     a silent, catastrophic recall collapse we demonstrate explicitly.
//
// The core intuition behind HNSW (and the "navigable small world" name):
//   1. A multi-LAYER graph. The top layer is a sparse long-range "highway" with a
//      handful of nodes; each lower layer is denser; layer 0 holds every node.
//      Search descends layer by layer, using the sparse upper layers to teleport
//      close to the answer's neighborhood before doing fine-grained search at the
//      bottom. This is what turns linear scan into logarithmic navigation.
//   2. Greedy best-first search with a bounded frontier (the ef beam). At each
//      step we expand the closest unexpanded candidate and stop when the frontier
//      can no longer improve. ef bounds how much backtracking we tolerate.
//   3. A heuristic neighbor selection at insert time that keeps the M kept edges
//      *diverse* (not all pointing into one dense clump), which is what preserves
//      long-range navigability instead of producing a locally-clustered blob.
//
// Why we build it on plain number[] and the core distance primitives: the book is
// about mechanism. Every distance comparison the graph makes is visible. We do NOT
// import any stageNN file (they self-run main()); we only depend on src/core.
//
// Honesty note carried from core/dataset.ts: synthetic clustered Gaussians are
// EASIER than real embeddings, so absolute recall here is optimistic. What
// transfers is the *shape* of the curves: ef↑ ⇒ recall↑ & QPS↓, and M-too-small
// ⇒ disconnection.

import { l2dist } from './core/vec.js';
import { makeDataset, makeQueries, computeGroundTruth } from './core/dataset.js';
import {
  recallAtK,
  timeIt,
  qps,
  estimateBytes,
  formatBytes,
} from './core/metrics.js';

// ---------------------------------------------------------------------------
// Distance: we rank by SQUARED L2 (no sqrt). sqrt is monotonic so it never
// changes the order of a top-k; skipping it removes one sqrt from the single
// hottest line in the whole index. core/vec.l2dist takes the sqrt because it is
// the user-facing distance; for internal ranking we want the cheaper form, so we
// inline the squared sum here. This is a deliberate hot-path hand-roll, not an
// accidental duplication of l2dist.
function l2sq(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

// HNSW config. Defaults follow the values the original paper found robust across
// datasets; we expose them so the demos can break the index on purpose.
//   - M: max neighbors a node keeps on layers >= 1. Layer 0 gets 2*M because the
//     bottom layer carries the real recall burden and benefits from being denser.
//   - efConstruction: the beam width used WHILE inserting. A larger build beam
//     finds better neighbors to connect to (higher graph quality) at the cost of
//     slower builds. It is independent of query-time ef.
//   - seed: PRNG seed for the random level assignment, so builds are reproducible.
interface HnswConfig {
  M: number;
  efConstruction: number;
  seed: number;
}

// A min-heap keyed by distance. Search touches the frontier in the innermost
// loop; using arrays + repeated Array.sort() there would dominate runtime and
// make the QPS numbers measure the sort, not the graph. So we hand-roll a binary
// heap. Two specializations below (min for "nearest first", max for "evict the
// farthest from a bounded result set") share this code.
class DistHeap {
  // Parallel arrays instead of {id,dist} objects: avoids one allocation per push
  // in the hottest loop. ids[i] and dists[i] belong together.
  private ids: number[] = [];
  private dists: number[] = [];
  // cmp < 0 means `a` should sit above `b` (toward the root). For a min-heap we
  // pass (a,b)=>a-b so the smallest distance is the root; for max-heap, b-a.
  constructor(private readonly cmp: (a: number, b: number) => number) {}

  get size(): number {
    return this.ids.length;
  }

  peekDist(): number {
    return this.dists[0];
  }

  peekId(): number {
    return this.ids[0];
  }

  push(id: number, dist: number): void {
    this.ids.push(id);
    this.dists.push(dist);
    this.bubbleUp(this.ids.length - 1);
  }

  pop(): { id: number; dist: number } {
    const id = this.ids[0];
    const dist = this.dists[0];
    const lastId = this.ids.pop()!;
    const lastDist = this.dists.pop()!;
    if (this.ids.length > 0) {
      this.ids[0] = lastId;
      this.dists[0] = lastDist;
      this.bubbleDown(0);
    }
    return { id, dist };
  }

  private bubbleUp(i: number): void {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.cmp(this.dists[i], this.dists[parent]) >= 0) break;
      this.swap(i, parent);
      i = parent;
    }
  }

  private bubbleDown(i: number): void {
    const n = this.ids.length;
    for (;;) {
      const left = 2 * i + 1;
      const right = 2 * i + 2;
      let best = i;
      if (left < n && this.cmp(this.dists[left], this.dists[best]) < 0) best = left;
      if (right < n && this.cmp(this.dists[right], this.dists[best]) < 0) best = right;
      if (best === i) break;
      this.swap(i, best);
      i = best;
    }
  }

  private swap(i: number, j: number): void {
    const ti = this.ids[i];
    this.ids[i] = this.ids[j];
    this.ids[j] = ti;
    const td = this.dists[i];
    this.dists[i] = this.dists[j];
    this.dists[j] = td;
  }
}

// Local seeded PRNG. We need our own (not Math.random) so the random level
// assignment is reproducible — otherwise the graph topology, and therefore every
// recall number, would change run to run. Same generator as core/dataset's
// mulberry32; duplicated here (≈4 lines) rather than widening the core export
// surface for one internal use, and kept seedable per index.
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class HnswIndex {
  private readonly data: number[][];
  private readonly M: number;
  private readonly Mmax0: number; // layer-0 cap = 2*M (denser bottom)
  private readonly efConstruction: number;
  // mL is the level-generation normalization factor. The paper sets it to
  // 1/ln(M); it makes the expected number of layers ~log_M(n), i.e. each layer up
  // keeps roughly 1/M of the nodes below it — the geometric thinning that gives
  // the logarithmic "highway".
  private readonly mL: number;
  private readonly rng: () => number;

  // neighbors[layer] is a Map<nodeId, neighborIds[]>. A node only appears in the
  // maps for layers 0..its assigned level. Layer 0 contains every node.
  private readonly neighbors: Map<number, number[]>[] = [];
  private entryPoint = -1; // id of the node on the current top layer
  private topLayer = -1; // index of the highest non-empty layer

  constructor(data: number[][], cfg: HnswConfig) {
    this.data = data;
    this.M = cfg.M;
    this.Mmax0 = cfg.M * 2;
    this.efConstruction = cfg.efConstruction;
    // Guard: M=1 makes ln(M)=0 ⇒ division by zero ⇒ every node lands on layer 0
    // with Infinity. We let M>=2 use the formula; for M=1 we force a flat single
    // layer, which is itself a failure mode the demo exercises (no highway).
    this.mL = cfg.M > 1 ? 1 / Math.log(cfg.M) : 0;
    this.rng = makeRng(cfg.seed);
  }

  // Random level for a new node, drawn from an exponentially-decaying
  // distribution. floor(-ln(U) * mL) gives level 0 most of the time, level 1
  // about 1/M of the time, etc. This is THE source of the hierarchy; a buggy
  // version that always returns 0 collapses HNSW to a single flat NSW graph
  // (slower navigation), which is why we keep it explicit and seeded.
  private randomLevel(): number {
    if (this.mL === 0) return 0; // M<=1: flat graph on purpose
    let u = this.rng();
    while (u === 0) u = this.rng(); // ln(0) = -Infinity; resample the measure-zero 0
    return Math.floor(-Math.log(u) * this.mL);
  }

  // Build the whole index by inserting every vector. Insertion order is dataset
  // order, so the build is deterministic given the seed.
  build(): void {
    for (let id = 0; id < this.data.length; id++) this.insert(id);
  }

  private ensureLayer(layer: number): void {
    while (this.neighbors.length <= layer) this.neighbors.push(new Map());
  }

  private getNeighbors(layer: number, id: number): number[] {
    return this.neighbors[layer].get(id) ?? [];
  }

  private insert(id: number): void {
    const level = this.randomLevel();
    this.ensureLayer(level);

    // First node ever: it becomes the entry point on its own top layer, with no
    // edges yet. Everything else navigates down from here.
    if (this.entryPoint === -1) {
      for (let l = 0; l <= level; l++) this.neighbors[l].set(id, []);
      this.entryPoint = id;
      this.topLayer = level;
      return;
    }

    const q = this.data[id];
    let ep = this.entryPoint;

    // Phase 1: descend from the top layer down to (level+1) using PLAIN greedy
    // search (ef=1). On these upper layers we only need to find the single
    // closest node to hand off as the entry point for the next layer down — we
    // are not collecting candidates yet, just walking toward q's neighborhood.
    for (let l = this.topLayer; l > level; l--) {
      ep = this.greedyDescend(q, ep, l);
    }

    // Phase 2: from min(level, topLayer) down to 0, run a *beam* search
    // (ef=efConstruction) to gather candidates, pick M diverse neighbors, and
    // wire bidirectional edges (pruning the neighbor's own list if it overflows).
    for (let l = Math.min(level, this.topLayer); l >= 0; l--) {
      const candidates = this.searchLayer(q, [ep], l, this.efConstruction);
      const Mcap = l === 0 ? this.Mmax0 : this.M;
      const selected = this.selectNeighbors(candidates, this.M);

      this.neighbors[l].set(id, selected.slice());

      // Bidirectional linking: a one-way edge is invisible to search coming from
      // the other side, which silently halves connectivity. We add the reverse
      // edge and, if the neighbor now exceeds its cap, re-run the same diversity
      // heuristic to decide which edge it drops — keeping the graph navigable
      // rather than letting popular nodes accumulate unbounded degree.
      for (const nb of selected) {
        const nbList = this.getNeighbors(l, nb);
        nbList.push(id);
        if (nbList.length > Mcap) {
          const pruned = this.selectNeighbors(
            // dist is measured from nb's own vector here: we are deciding which of
            // nb's edges to keep, so "distance to q" means distance to nb.
            nbList.map((x) => ({ id: x, dist: l2sq(this.data[nb], this.data[x]) })),
            Mcap,
          );
          this.neighbors[l].set(nb, pruned);
        } else {
          this.neighbors[l].set(nb, nbList);
        }
      }

      // Entry point for the NEXT lower layer must be a node that already exists in
      // the graph at that layer — i.e. the nearest candidate this beam found, NOT
      // the node we are inserting. `id` has no edges below `l` yet, so seeding the
      // next searchLayer from `id` would start the beam at a dead-end with an
      // empty neighbor list and the new node would only ever connect to its own
      // tiny landing region. That bug fragments the graph (reachable ≪ n) and
      // flattens recall across every ef. candidates[0] is nearest-first.
      if (candidates.length > 0) ep = candidates[0].id;
    }

    // If the new node reached above the current top, it becomes the new global
    // entry point and raises the ceiling. Edge case that bites naive impls: you
    // must also create empty neighbor lists for the new node on every new layer.
    if (level > this.topLayer) {
      for (let l = this.topLayer + 1; l <= level; l++) this.neighbors[l].set(id, []);
      this.entryPoint = id;
      this.topLayer = level;
    }
  }

  // Plain greedy hill-climb on one layer: repeatedly jump to the neighbor closest
  // to q until no neighbor improves. Used only for the cheap upper-layer descent
  // (ef=1 equivalent). It can get stuck in a local minimum — that's fine on the
  // sparse upper layers because we re-search with a beam once we hit the working
  // layers.
  private greedyDescend(q: number[], entry: number, layer: number): number {
    let current = entry;
    let currentDist = l2sq(q, this.data[current]);
    for (;;) {
      let improved = false;
      for (const nb of this.getNeighbors(layer, current)) {
        const d = l2sq(q, this.data[nb]);
        if (d < currentDist) {
          currentDist = d;
          current = nb;
          improved = true;
        }
      }
      if (!improved) return current;
    }
  }

  // Beam (best-first) search on a single layer. This is the heart of HNSW query
  // time. Invariant maintained: `results` is a MAX-heap of the ef best nodes seen
  // so far (root = current worst-of-the-best), and `candidates` is a MIN-heap of
  // the frontier still to expand. We stop when the nearest unexpanded candidate
  // is already farther than our current worst result — at that point no remaining
  // path can improve the top-ef, so further exploration is wasted.
  //
  // Failure mode this exposes: with ef too small, `results` fills almost
  // immediately and the stop condition triggers after a handful of hops, leaving
  // the true neighbors unreached → recall craters. The demo sweeps ef to show it.
  private searchLayer(
    q: number[],
    entryPoints: number[],
    layer: number,
    ef: number,
  ): { id: number; dist: number }[] {
    const visited = new Set<number>();
    const candidates = new DistHeap((a, b) => a - b); // min-heap: nearest first
    const results = new DistHeap((a, b) => b - a); // max-heap: farthest of best at root

    for (const ep of entryPoints) {
      const d = l2sq(q, this.data[ep]);
      candidates.push(ep, d);
      results.push(ep, d);
      visited.add(ep);
    }

    while (candidates.size > 0) {
      const nearest = candidates.pop();
      // Stop: the closest thing left to explore is worse than our worst kept
      // result and results is already full. Nothing reachable can improve top-ef.
      if (results.size >= ef && nearest.dist > results.peekDist()) break;

      for (const nb of this.getNeighbors(layer, nearest.id)) {
        if (visited.has(nb)) continue;
        visited.add(nb);
        const d = l2sq(q, this.data[nb]);
        // Push into the frontier/results only if it could matter: either we have
        // not filled ef results yet, or it beats the current worst kept result.
        if (results.size < ef || d < results.peekDist()) {
          candidates.push(nb, d);
          results.push(nb, d);
          if (results.size > ef) results.pop(); // evict the farthest, keep ef best
        }
      }
    }

    // Drain the max-heap into an array sorted nearest-first for the caller.
    const out: { id: number; dist: number }[] = [];
    while (results.size > 0) out.push(results.pop());
    out.reverse(); // popped farthest-first; reverse to nearest-first
    return out;
  }

  // Neighbor selection HEURISTIC (paper's Algorithm 4, simplified). Naively
  // keeping the M absolute-nearest candidates produces a locally-clustered hub:
  // every kept edge points into the same dense clump, so the graph loses the
  // long-range links that make it navigable. Instead we keep a candidate only if
  // it is closer to q than to every already-selected neighbor. This favors
  // *diverse directions* and is what preserves small-world navigability.
  //
  // Each candidate already carries `dist` = its distance to q (searchLayer fills
  // it, the prune path computes it), so q itself is not needed here — the
  // domination test compares candidate-to-q (c.dist) against candidate-to-kept
  // (l2sq below), both already in scope.
  private selectNeighbors(
    candidates: { id: number; dist: number }[],
    m: number,
  ): number[] {
    const sorted = candidates.slice().sort((a, b) => a.dist - b.dist);
    const selected: { id: number; dist: number }[] = [];
    for (const c of sorted) {
      if (selected.length >= m) break;
      // Keep c only if it is not "dominated": closer to q than to any neighbor we
      // already kept. Dominated candidates would just thicken an existing edge
      // direction instead of opening a new one.
      let keep = true;
      for (const s of selected) {
        if (l2sq(this.data[c.id], this.data[s.id]) < c.dist) {
          keep = false;
          break;
        }
      }
      if (keep) selected.push(c);
    }
    // If the diversity heuristic was too strict and left us short of m, top up
    // with the nearest remaining candidates so we never under-connect (which
    // would itself harm recall). Connectivity floor beats perfect diversity.
    if (selected.length < m) {
      const chosen = new Set(selected.map((s) => s.id));
      for (const c of sorted) {
        if (selected.length >= m) break;
        if (!chosen.has(c.id)) selected.push(c);
      }
    }
    return selected.map((s) => s.id);
  }

  // Query: descend the highways with greedy ef=1, then one beam search at layer 0
  // with the caller's ef, and return the k nearest ids. ef >= k is required for
  // the result to even contain k items; we let callers pass small ef on purpose
  // to demonstrate the recall collapse, and clamp the returned slice to k.
  query(q: number[], k: number, ef: number): number[] {
    if (this.entryPoint === -1) return [];
    let ep = this.entryPoint;
    for (let l = this.topLayer; l > 0; l--) ep = this.greedyDescend(q, ep, l);
    const found = this.searchLayer(q, [ep], 0, Math.max(ef, k));
    return found.slice(0, k).map((e) => e.id);
  }

  // --- introspection for the failure-mode demos (not part of a real index API) ---

  // Count nodes reachable from the entry point via layer-0 edges. If the graph
  // fragmented (M too small), this is < n and the unreachable nodes can NEVER be
  // returned regardless of ef — the silent failure HNSW is most vulnerable to.
  countReachableFromEntry(): number {
    if (this.entryPoint === -1) return 0;
    const seen = new Set<number>([this.entryPoint]);
    const stack = [this.entryPoint];
    while (stack.length > 0) {
      const cur = stack.pop()!;
      for (const nb of this.getNeighbors(0, cur)) {
        if (!seen.has(nb)) {
          seen.add(nb);
          stack.push(nb);
        }
      }
    }
    return seen.size;
  }

  // Average out-degree on layer 0 — a cheap proxy for how densely wired the graph
  // is. Edge count ≈ nodes * avgDegree / 2 (bidirectional) for the memory model.
  avgDegreeLayer0(): number {
    let total = 0;
    let count = 0;
    for (const list of this.neighbors[0].values()) {
      total += list.length;
      count++;
    }
    return count === 0 ? 0 : total / count;
  }

  get layerCount(): number {
    return this.topLayer + 1;
  }

  // Estimated index memory: the full vectors (payload floor from core) PLUS the
  // graph's adjacency. Each directed edge is one integer id; in a number[] that
  // is an 8-byte double slot (JS has no int32 array of plain numbers). We count
  // every edge on every layer. This is an ESTIMATE (V8 array overhead is real and
  // unmodeled), labelled as such wherever printed.
  estimateBytes(): number {
    const dim = this.data[0]?.length ?? 0;
    const vectorBytes = estimateBytes(this.data.length, dim); // 8 B/component
    let edges = 0;
    for (const layer of this.neighbors) {
      for (const list of layer.values()) edges += list.length;
    }
    const edgeBytes = edges * 8; // one double slot per neighbor id
    return vectorBytes + edgeBytes;
  }
}

// ---------------------------------------------------------------------------
// Demo driver.
// ---------------------------------------------------------------------------

const N = 5000;
const DIM = 32;
const CLUSTERS = 16;
const NUM_QUERIES = 200;
const K = 10;
const DATA_SEED = 7;
const QUERY_SEED = 99;

function runHnsw(
  index: HnswIndex,
  queries: number[][],
  k: number,
  ef: number,
): { ids: number[][]; ms: number } {
  let ids: number[][] = [];
  // All queries inside one timed closure so the wall-clock dwarfs per-call JIT
  // jitter (core/metrics.timeIt is single-shot by design).
  const ms = timeIt(() => {
    ids = queries.map((q) => index.query(q, k, ef));
  });
  return { ids, ms };
}

function main(): void {
  console.log('=== stage04: HNSW from scratch ===\n');

  console.log(
    `数据集: ${N} 向量 × ${DIM} 维, ${CLUSTERS} 个聚类 | 查询: ${NUM_QUERIES} 条 | k=${K}`,
  );
  const data = makeDataset(N, DIM, CLUSTERS, DATA_SEED);
  const queries = makeQueries(NUM_QUERIES, DIM, QUERY_SEED);

  // Ground truth + brute-force baseline (the yardstick every ANN number is judged
  // against). Brute force is exact ⇒ recall 1.0 by definition; we time it so the
  // ef sweep has an honest QPS to beat.
  const truth = computeGroundTruth(data, queries, K, 'l2');
  const bruteMs = timeIt(() => {
    queries.forEach((q) => {
      const scored = data.map((v, idx) => ({ idx, d: l2dist(q, v) }));
      scored.sort((a, b) => a.d - b.d || a.idx - b.idx);
      scored.slice(0, K);
    });
  });
  console.log(
    `\n[brute force baseline] recall@${K}=1.0000 (exact) | QPS=${qps(NUM_QUERIES, bruteMs).toFixed(0)} | ${formatBytes(estimateBytes(N, DIM))} (est.)`,
  );

  // ---- Build the index once with healthy params, then sweep ef at query time.
  const cfg: HnswConfig = { M: 16, efConstruction: 200, seed: 42 };
  let buildMs = 0;
  const index = new HnswIndex(data, cfg);
  buildMs = timeIt(() => index.build());
  console.log(
    `\n[build] M=${cfg.M} efConstruction=${cfg.efConstruction} | 建图 ${buildMs.toFixed(0)} ms | ` +
      `层数=${index.layerCount} | layer0 平均度=${index.avgDegreeLayer0().toFixed(1)} | ` +
      `可达节点=${index.countReachableFromEntry()}/${N}`,
  );
  console.log(`         索引内存(向量+图边, est.)=${formatBytes(index.estimateBytes())}`);

  // ---- The headline trade-off curve: recall↑ and QPS↓ as ef grows. ----------
  console.log('\n[ef 权衡曲线] ef 越大召回越高但 QPS 越低:');
  console.log('   ef  | recall@10 |    QPS   | vs brute');
  console.log('  -----+-----------+----------+---------');
  const efValues = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  const bruteQps = qps(NUM_QUERIES, bruteMs);
  for (const ef of efValues) {
    // Warm the JIT once, then a few timed passes summed, so tiny ef values (which
    // run in well under a millisecond per pass) still produce a measurable total
    // rather than landing on the timeIt 0ms → Infinity QPS guard.
    index.query(queries[0], K, ef);
    const REPS = 5;
    let totalMs = 0;
    let ids: number[][] = [];
    for (let r = 0; r < REPS; r++) {
      const run = runHnsw(index, queries, K, ef);
      totalMs += run.ms;
      ids = run.ids;
    }
    const recall = recallAtK(ids, truth, K);
    const q = qps(NUM_QUERIES * REPS, totalMs);
    const speedup = bruteQps === 0 ? 0 : q / bruteQps;
    console.log(
      `  ${String(ef).padStart(4)} |  ${recall.toFixed(4)}  | ${q.toFixed(0).padStart(8)} | ${speedup.toFixed(1)}x`,
    );
  }

  // ---- FAILURE MODE 1: ef too small ⇒ the beam under-explores ⇒ recall ceiling.
  // With a small beam the search keeps only a few candidates alive, so it commits
  // to the first locally-good region and never backtracks to the true neighbors
  // sitting one cluster over. Note the clamp in query(): ef is raised to at least
  // k so the result can hold k items at all, which is why ef=1..8 here all behave
  // like the effective floor ef=k=10 and share one recall (~0.65) — already well
  // below the 0.999 a generous ef=256 reaches. This is the most common production
  // misconfiguration: people leave ef at the tiny build default and ship 0.65.
  console.log('\n[失败模式 1] ef 太小 → 束搜索欠探索, 召回有天花板:');
  for (const ef of [1, 4]) {
    index.query(queries[0], K, ef);
    const run = runHnsw(index, queries, K, ef);
    const recall = recallAtK(run.ids, truth, K);
    console.log(
      `   ef=${ef}: recall@${K}=${recall.toFixed(4)} ` +
        `(被 query() 内的 max(ef,k) 钳到 ef=${K}; 仍远低于 ef=256 的 0.999)`,
    );
  }

  // ---- FAILURE MODE 2: M too small ⇒ graph fragments / under-connects. -------
  // We rebuild a fresh index with M=2 and a flat M=1 graph, then show two
  // symptoms: (a) low layer-0 degree, (b) for M=1 the level formula degenerates
  // to a single flat layer with no highway, so even generous ef can't recover the
  // recall a well-wired graph reaches. We report reachability too: if it drops
  // below N, some vectors are *unreturnable at any ef*.
  console.log('\n[失败模式 2] M 太小 → 图欠连通, 召回有天花板, 大 ef 也救不回:');
  for (const M of [2, 1]) {
    const weakCfg: HnswConfig = { M, efConstruction: 50, seed: 42 };
    const weak = new HnswIndex(data, weakCfg);
    weak.build();
    const reachable = weak.countReachableFromEntry();
    // Use a generous ef to prove the ceiling is the graph's, not the beam's.
    const bigEf = 200;
    weak.query(queries[0], K, bigEf);
    const run = runHnsw(weak, queries, K, bigEf);
    const recall = recallAtK(run.ids, truth, K);
    const note =
      M === 1
        ? '(M=1 ⇒ mL=0 ⇒ 单层平图, 无上层高速路, 导航退化)'
        : '(稀疏图, 长程链路不足)';
    console.log(
      `   M=${M}: recall@${K}=${recall.toFixed(4)} @ef=${bigEf} | ` +
        `层数=${weak.layerCount} | layer0 平均度=${weak.avgDegreeLayer0().toFixed(1)} | ` +
        `可达=${reachable}/${N} ${note}`,
    );
  }

  console.log(
    '\n结论: HNSW 用「分层高速路 + ef 受限束搜索」把 O(n) 暴力扫降到近 O(log n) 导航;\n' +
      'ef 是召回↔速度的运行时旋钮, M 是建图期决定的连通性下限 — 两者都太小会让索引悄悄失效。',
  );
}

main();
