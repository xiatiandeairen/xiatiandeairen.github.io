// stage05-discrete.ts — Discrete math & logic: the skeleton every algorithm is built on.
//
// SCOPE (this stage): four pillars of discrete structures, each VERIFIED by code, not asserted:
//   1. Graphs   — Dijkstra shortest path + a bipartite matching (augmenting paths).
//   2. Counting — permutation/combination/inclusion-exclusion, brute force vs closed form.
//   3. Induction— prove ∑1..n = n(n+1)/2 etc. by checking the closed form against the loop
//                 for every n up to N (induction's base+step made empirically falsifiable).
//   4. Complexity — wall-clock O(n)/O(n log n)/O(n²)/O(2ⁿ) on growing n, printed as a table.
//
// WHY brute-vs-formula instead of just printing the formula: a formula you can't reproduce
//   from first principles is folklore. Counting each arrangement by hand (small n) and
//   matching nPr/nCr is the only honest proof that the closed form is the SAME object.
//
// HONEST-NUMBER NOTE: the complexity table uses real performance.now() wall-clock on THIS
//   machine. Absolute ms are host-specific and noisy at tiny n (sub-ms work is dominated by
//   loop overhead); what transfers is the GROWTH SHAPE — how ms multiplies as n doubles.
//
// FAILURE MODES demoed (not just happy path):
//   - Dijkstra on a NEGATIVE edge: its greedy "finalized node is optimal" invariant breaks,
//     and it silently returns a too-large distance (no crash — the dangerous kind).
//   - Brute-forcing an exponential problem (subset enumeration): n=20 already costs ~1e6
//     operations; we measure the doubling and extrapolate why n=60 is hopeless.
//
// CONTRACT: reuse core/rng.js to build the seeded graph; core/plot.js for the timing chart.

import { mulberry32, sampleUniform, type Rng } from "./core/rng.js";
import { asciiLine } from "./core/plot.js";

// ─────────────────────────────────────────────────────────────────────────────
// Graph representation: adjacency list of {to, weight}. We keep it explicit (not a
// matrix) because Dijkstra's cost is O(E log V) on a list, and a list makes the
// "which edges does a node have" question obvious to the reader.
// ─────────────────────────────────────────────────────────────────────────────
interface Edge {
  to: number;
  weight: number;
}
type Graph = Edge[][]; // graph[u] = outgoing edges of node u

/**
 * Build a seeded, connected-ish weighted DAG-like graph for reproducibility.
 * INVARIANT: every node i (except the last) gets at least one forward edge i→j>i, so a path
 *   from 0 to n-1 always exists — otherwise "shortest path" could be vacuously Infinity and
 *   the demo would teach nothing. Weights are positive (Dijkstra's precondition).
 */
function buildSeededGraph(n: number, rng: Rng): Graph {
  const graph: Graph = Array.from({ length: n }, () => []);
  for (let u = 0; u < n - 1; u++) {
    // Guaranteed forward edge keeps the graph traversable end-to-end.
    graph[u].push({ to: u + 1, weight: Math.round(sampleUniform(rng, 1, 9)) });
    // A few extra random forward edges to create genuine alternative routes.
    for (let v = u + 2; v < n; v++) {
      if (rng() < 0.4) {
        graph[u].push({ to: v, weight: Math.round(sampleUniform(rng, 1, 9)) });
      }
    }
  }
  return graph;
}

/**
 * Dijkstra shortest path from `src`. Returns dist[] and prev[] (for path reconstruction).
 * WHY a plain array scan for the min instead of a binary heap: V is tiny here (teaching), and
 *   an O(V²) scan is obviously-correct; the heap is an optimization, not the algorithm's idea.
 * CORE INVARIANT (the thing that breaks on negative edges): once a node is popped as the
 *   current minimum among unvisited, its distance is FINAL — because every remaining path to
 *   it must go through some unvisited node whose tentative distance is already ≥ this one, and
 *   adding non-negative edges can only increase it. Negative edges violate "can only increase".
 */
function dijkstra(graph: Graph, src: number): { dist: number[]; prev: number[] } {
  const n = graph.length;
  const dist = new Array<number>(n).fill(Infinity);
  const prev = new Array<number>(n).fill(-1);
  const visited = new Array<boolean>(n).fill(false);
  dist[src] = 0;
  for (let iter = 0; iter < n; iter++) {
    // Pick the unvisited node with the smallest tentative distance.
    let u = -1;
    let best = Infinity;
    for (let i = 0; i < n; i++) {
      if (!visited[i] && dist[i] < best) {
        best = dist[i];
        u = i;
      }
    }
    if (u === -1) break; // remaining nodes unreachable
    visited[u] = true;
    // Relax: a finalized node never gets relaxed again (that's the invariant we rely on).
    for (const { to, weight } of graph[u]) {
      if (dist[u] + weight < dist[to]) {
        dist[to] = dist[u] + weight;
        prev[to] = u;
      }
    }
  }
  return { dist, prev };
}

function reconstructPath(prev: number[], target: number): number[] {
  const path: number[] = [];
  for (let at = target; at !== -1; at = prev[at]) path.unshift(at);
  return path;
}

/**
 * Brute-force shortest path by enumerating ALL simple paths (DFS). Ground truth for cross-
 * checking Dijkstra AND for exposing its negative-edge bug: brute force is correct regardless
 * of edge sign (it just tries everything), so when the two disagree we know Dijkstra is wrong.
 * INVARIANT: only valid because graph is small + acyclic-forward; exponential in general.
 */
function bruteForceShortest(graph: Graph, src: number, target: number): number {
  let best = Infinity;
  const dfs = (u: number, cost: number): void => {
    if (u === target) {
      best = Math.min(best, cost);
      return;
    }
    for (const { to, weight } of graph[u]) dfs(to, cost + weight);
  };
  dfs(src, 0);
  return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bipartite matching via augmenting paths (the Hungarian-algorithm kernel, Kuhn's variant).
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Maximum bipartite matching. `adj[l]` lists right-side nodes acceptable to left node l.
 * matchR[r] = which left node is matched to right node r, or -1.
 * WHY augmenting paths: a matching is maximum iff no augmenting path exists (Berge's theorem).
 *   Each left node tries to claim a partner; if its preferred right node is taken, we try to
 *   re-route the current occupant. This is the same idea as max-flow on a unit-capacity graph.
 * INVARIANT: `seen` must reset per left node — it prevents infinite re-routing within ONE
 *   augmenting search, but stale `seen` across nodes would block legitimate re-routes.
 */
function maxBipartiteMatching(adj: number[][], numRight: number): number[] {
  const matchR = new Array<number>(numRight).fill(-1);
  const tryAugment = (l: number, seen: boolean[]): boolean => {
    for (const r of adj[l]) {
      if (seen[r]) continue;
      seen[r] = true;
      // Right node r is free, OR its current owner can be re-routed elsewhere.
      if (matchR[r] === -1 || tryAugment(matchR[r], seen)) {
        matchR[r] = l;
        return true;
      }
    }
    return false;
  };
  for (let l = 0; l < adj.length; l++) {
    const seen = new Array<boolean>(numRight).fill(false); // fresh per search (see INVARIANT)
    tryAugment(l, seen);
  }
  return matchR;
}

// ─────────────────────────────────────────────────────────────────────────────
// Combinatorics: closed forms, each verified by brute-force counting at small n.
// ─────────────────────────────────────────────────────────────────────────────
function factorial(n: number): number {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
}

// nPr = n!/(n-r)!  — ordered arrangements.
function permutations(n: number, r: number): number {
  return factorial(n) / factorial(n - r);
}

// nCr = n!/(r!(n-r)!) — unordered selections.
function combinations(n: number, r: number): number {
  return factorial(n) / (factorial(r) * factorial(n - r));
}

/**
 * Brute-force count of size-r subsets of {0..n-1} by enumerating all 2ⁿ bitmasks and keeping
 * those with exactly r bits set. This is the GROUND TRUTH that nCr must equal — proving the
 * closed form counts the same objects, not just produces a plausible number.
 */
function bruteForceCountSubsets(n: number, r: number): number {
  let count = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    let bits = 0;
    for (let b = 0; b < n; b++) bits += (mask >> b) & 1;
    if (bits === r) count++;
  }
  return count;
}

/**
 * Inclusion-exclusion: count integers in [1, limit] divisible by 2, 3, or 5.
 * Closed form: |A∪B∪C| = Σ|A| − Σ|A∩B| + |A∩B∩C|. We verify it against a brute scan.
 * WHY this matters: naive "add up each set" double-counts overlaps; IE is the systematic fix
 *   and is the backbone of probability unions, derangements, and the sieve.
 */
function inclusionExclusionDivisible(limit: number): number {
  const div = (k: number) => Math.floor(limit / k);
  // +singles − pairs(lcm) + triple(lcm). All inputs coprime-ish so lcm = product here.
  return div(2) + div(3) + div(5) - div(6) - div(10) - div(15) + div(30);
}

function bruteForceCountDivisible(limit: number): number {
  let count = 0;
  for (let i = 1; i <= limit; i++) {
    if (i % 2 === 0 || i % 3 === 0 || i % 5 === 0) count++;
  }
  return count;
}

// ─────────────────────────────────────────────────────────────────────────────
// Induction made empirical: a closed form is "proven for all n ≤ N" if it matches the
// step-by-step accumulation at every n. Base case (n=1) + every step = the induction chain.
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Verify ∑(1..n) = n(n+1)/2 and ∑(i²) = n(n+1)(2n+1)/6 for all n up to maxN by comparing the
 * closed form against an explicit running sum. Returns the first n where they DISAGREE, or -1
 * if the identity held throughout (which is the empirical evidence the induction is sound).
 */
function verifyInductiveSums(maxN: number): { firstFailSum: number; firstFailSq: number } {
  let runningSum = 0;
  let runningSq = 0;
  let firstFailSum = -1;
  let firstFailSq = -1;
  for (let n = 1; n <= maxN; n++) {
    runningSum += n;
    runningSq += n * n;
    const closedSum = (n * (n + 1)) / 2;
    const closedSq = (n * (n + 1) * (2 * n + 1)) / 6;
    if (firstFailSum === -1 && runningSum !== closedSum) firstFailSum = n;
    if (firstFailSq === -1 && runningSq !== closedSq) firstFailSq = n;
  }
  return { firstFailSum, firstFailSq };
}

// ─────────────────────────────────────────────────────────────────────────────
// Complexity benchmark: one representative algorithm per class, timed on growing n.
// Each does REAL work and returns a sink value so the JIT can't elide it as dead code
// (the same honesty trap stage12 warns about).
// ─────────────────────────────────────────────────────────────────────────────

// O(n): single linear pass (sum).
function linearWork(n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += i & 7;
  return s;
}

// O(n log n): merge sort on a deterministic pseudo-random array (return checksum).
function nLogNWork(n: number, rng: Rng): number {
  const arr = Array.from({ length: n }, () => (rng() * n) | 0);
  const sorted = mergeSort(arr);
  // Checksum touches every element so the sort can't be optimized away.
  let s = 0;
  for (let i = 0; i < sorted.length; i++) s += sorted[i] & 7;
  return s;
}

function mergeSort(arr: number[]): number[] {
  if (arr.length <= 1) return arr;
  const mid = arr.length >> 1;
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  const out: number[] = [];
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    out.push(left[i] <= right[j] ? left[i++] : right[j++]);
  }
  while (i < left.length) out.push(left[i++]);
  while (j < right.length) out.push(right[j++]);
  return out;
}

// O(n²): all-pairs distance sum (the classic double loop).
function quadraticWork(n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) s += (i - j) & 7;
  }
  return s;
}

// O(2ⁿ): enumerate every subset of an n-element set (count those summing to even).
// THE failure mode: this is the wall every brute-force-on-exponential hits.
function exponentialWork(n: number): number {
  let evenSubsets = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    let sum = 0;
    for (let b = 0; b < n; b++) if ((mask >> b) & 1) sum += b;
    if ((sum & 1) === 0) evenSubsets++;
  }
  return evenSubsets;
}

/** Median of repeated timings — single runs at sub-ms scale are dominated by noise/JIT warmup. */
function timeMedianMs(fn: () => number, repeats: number): { ms: number; sink: number } {
  const samples: number[] = [];
  let sink = 0;
  for (let r = 0; r < repeats; r++) {
    const t0 = performance.now();
    sink += fn(); // accumulate result so the work is observable (anti dead-code-elimination)
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  return { ms: samples[samples.length >> 1], sink };
}

function main(): void {
  console.log("=== Stage 05 · 离散数学与逻辑：算法的骨架 ===\n");

  // ── 1. Graphs: Dijkstra + path, cross-checked against brute force ──────────
  const graphRng = mulberry32(5);
  const N_GRAPH = 8;
  const graph = buildSeededGraph(N_GRAPH, graphRng);
  const { dist, prev } = dijkstra(graph, 0);
  const target = N_GRAPH - 1;
  const path = reconstructPath(prev, target);
  const brute = bruteForceShortest(graph, 0, target);
  console.log("① 图：Dijkstra 最短路 (种子图, 8 节点)");
  console.log(`   0 → ${target} 最短距离  = ${dist[target]}`);
  console.log(`   最短路径          = ${path.join(" → ")}`);
  console.log(`   暴力枚举对拍      = ${brute} (${brute === dist[target] ? "一致 ✓" : "不一致 ✗"})`);

  // Bipartite matching: 4 workers × 4 tasks, seeded eligibility.
  const matchRng = mulberry32(11);
  const NUM_LEFT = 4;
  const NUM_RIGHT = 4;
  const eligibility: number[][] = Array.from({ length: NUM_LEFT }, () => {
    const list: number[] = [];
    for (let r = 0; r < NUM_RIGHT; r++) if (matchRng() < 0.55) list.push(r);
    return list;
  });
  const matchR = maxBipartiteMatching(eligibility, NUM_RIGHT);
  const matchedCount = matchR.filter((l) => l !== -1).length;
  console.log("\n   二分匹配 (4 工人 × 4 任务, 种子可行表):");
  for (let l = 0; l < NUM_LEFT; l++) {
    console.log(`     工人${l} 可做任务 = {${eligibility[l].join(",")}}`);
  }
  const assignment = matchR
    .map((l, r) => (l === -1 ? null : `工人${l}→任务${r}`))
    .filter((x): x is string => x !== null);
  console.log(`   最大匹配 = ${matchedCount} 对：${assignment.join(", ")}`);

  // ── 2. Combinatorics: closed form vs brute force ──────────────────────────
  console.log("\n② 组合计数：闭式 vs 暴力对拍");
  const n = 6;
  const r = 3;
  const nPr = permutations(n, r);
  const nCr = combinations(n, r);
  const bruteSubsets = bruteForceCountSubsets(n, r);
  console.log(`   P(${n},${r}) = ${nPr}   C(${n},${r}) = ${nCr}`);
  console.log(
    `   暴力数 ${n} 元集的 ${r}-子集 = ${bruteSubsets} ` +
      `(${bruteSubsets === nCr ? "= C(n,r) ✓" : "≠ C(n,r) ✗"})`,
  );
  const ieLimit = 1000;
  const ie = inclusionExclusionDivisible(ieLimit);
  const ieBrute = bruteForceCountDivisible(ieLimit);
  console.log(
    `   容斥：[1,${ieLimit}] 中能被 2/3/5 整除的数 = ${ie} ` +
      `(暴力 ${ieBrute}, ${ie === ieBrute ? "一致 ✓" : "不一致 ✗"})`,
  );

  // ── 3. Induction: closed form holds for all n up to N ─────────────────────
  console.log("\n③ 归纳证明的代码验证 (闭式 vs 逐步累加, n=1..10000)");
  const { firstFailSum, firstFailSq } = verifyInductiveSums(10000);
  console.log(
    `   ∑(1..n)=n(n+1)/2        : ${firstFailSum === -1 ? "前 10000 个 n 全部成立 ✓" : `n=${firstFailSum} 失败 ✗`}`,
  );
  console.log(
    `   ∑(i²)=n(n+1)(2n+1)/6    : ${firstFailSq === -1 ? "前 10000 个 n 全部成立 ✓" : `n=${firstFailSq} 失败 ✗`}`,
  );

  // ── 4. Complexity: wall-clock growth table ────────────────────────────────
  console.log("\n④ 复杂度实测 (本机 wall-clock, 取多次中位数)");
  console.log("   n      O(n)      O(n log n)   O(n²)       O(2ⁿ)");
  const ns = [4, 8, 12, 16, 20];
  const quadCurve: number[] = [];
  const expCurve: number[] = [];
  let guardSink = 0;
  for (const sz of ns) {
    // More repeats for tiny/fast work to lift its signal above timer granularity.
    const lin = timeMedianMs(() => linearWork(sz * 1000), 50);
    const nlogn = timeMedianMs(() => nLogNWork(sz * 1000, mulberry32(sz)), 20);
    const quad = timeMedianMs(() => quadraticWork(sz * 100), 20);
    const expo = timeMedianMs(() => exponentialWork(sz), 5);
    guardSink += lin.sink + nlogn.sink + quad.sink + expo.sink;
    quadCurve.push(quad.ms);
    expCurve.push(expo.ms);
    const fmt = (x: number) => x.toFixed(4).padStart(8);
    console.log(
      `   n=${String(sz).padEnd(3)} ${fmt(lin.ms)}  ${fmt(nlogn.ms)}    ${fmt(quad.ms)}   ${fmt(expo.ms)}`,
    );
  }
  console.log(`   (校验和 ${guardSink}, 防 JIT 把循环当死代码删掉)`);
  console.log("   注：n 列对各算法量纲不同 (O(n) 用 n×1000, O(2ⁿ) 用 n 本身)，看每列随 n 的增长形状，不横向比绝对值。");

  // ── Failure mode A: exponential growth, why brute force on n=60 is hopeless ─
  console.log("\n--- 失败模式 A：指数级问题上暴力 ---");
  const expFirst = expCurve.find((ms) => ms > 0) ?? 0;
  const expLast = expCurve[expCurve.length - 1];
  console.log(`   O(2ⁿ) 从 n=4 到 n=20：实测 ${expFirst.toFixed(4)}ms → ${expLast.toFixed(4)}ms`);
  console.log(`   n 每 +1 工作量翻倍：2²⁰≈1e6 子集尚可，2⁶⁰≈1.15e18 子集。`);
  // Honest extrapolation, clearly labelled (est.): per-subset cost from the n=20 measurement.
  const perSubsetMs = expLast / Math.pow(2, 20);
  const yearsForN60 = (perSubsetMs * Math.pow(2, 60)) / (1000 * 3600 * 24 * 365);
  console.log(`   按本机 n=20 单子集耗时外推 (est.)：n=60 需约 ${yearsForN60.toExponential(2)} 年 —— 必须换多项式算法或剪枝。`);
  console.log("   O(2ⁿ) 耗时随 n 的增长曲线：");
  console.log(asciiLine(expCurve, 40, 6));

  // ── Failure mode B: Dijkstra silently wrong on a negative edge ─────────────
  console.log("\n--- 失败模式 B：Dijkstra 遇负权悄悄给错答案 ---");
  // Crafted so the bug actually manifests: node 1 is FINALIZED at dist=1 and immediately used
  // to relax target node 3 (→ 2) BEFORE the negative edge 2→1 (w=−5) lowers node 1 to −3.
  // Node 3 is never re-relaxed, so Dijkstra reports 0→3 = 2 while the true cost is −3+1 = −2.
  //   edges: 0→1 (1), 0→2 (2), 1→3 (1), 2→1 (−5);  target = 3.
  const negGraph: Graph = [
    [
      { to: 1, weight: 1 },
      { to: 2, weight: 2 },
    ],
    [{ to: 3, weight: 1 }],
    [{ to: 1, weight: -5 }],
    [],
  ];
  const negTarget = 3;
  const dj = dijkstra(negGraph, 0).dist[negTarget];
  const truth = bruteForceShortest(negGraph, 0, negTarget);
  console.log(`   Dijkstra 给出 0→${negTarget} = ${dj}`);
  console.log(`   暴力(正确) 0→${negTarget}   = ${truth}`);
  console.log(
    `   ${dj === truth ? "一致" : `相差 ${dj - truth} (偏大)`}：节点1 被"定终"在 dist=1 并据此松弛了节点3，` +
      `之后负边 2→1 把节点1 降到 −3 时节点3 已不再回头更新。`,
  );
  console.log(
    "   Dijkstra 不报错、不崩溃，直接返回偏大值 —— 这种\"沉默的错\"最危险，负权要用 Bellman-Ford。",
  );
}

main();
