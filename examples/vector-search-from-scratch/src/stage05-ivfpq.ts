// stage05-ivfpq.ts — IVFPQ: combine coarse pruning (IVF) with code compression (PQ).
//
// What this stage teaches: the two ANN ideas from the previous stages are
// orthogonal and *compose*. IVF (inverted file) prunes the search to a few
// nearby cells so you touch O(nprobe/nlist · n) vectors instead of all n. PQ
// (product quantization) shrinks each stored vector from dim·8 bytes to
// nsub bytes, so a billion-scale index fits in RAM. IVFPQ is the workhorse of
// FAISS-era large-scale search precisely because it attacks *both* costs at
// once: IVF cuts the number of comparisons, PQ cuts the cost (and memory) of
// each comparison.
//
// The non-obvious mechanics this file makes concrete:
//  1. IVF and PQ each lose recall independently, and the losses *stack*. The
//     headline IVFPQ recall is below both standalone IVF and standalone PQ.
//  2. PQ is trained on *residuals* (vector minus its cell centroid), not on the
//     raw vectors. Residuals are smaller and more uniform across cells, so a
//     shared codebook quantizes them far better than raw coordinates. Skipping
//     this (PQ on raw vectors) is the classic IVFPQ implementation bug — we
//     demo both so the recall gap is visible, not asserted.
//  3. The PQ distance is computed via a per-query lookup table (ADC,
//     asymmetric distance computation): precompute distance from the query's
//     residual to every centroid in every subspace once, then each candidate's
//     distance is nsub table reads + adds. No per-candidate vector math.
//
// Honesty notes: all recall/QPS are measured on the deterministic dataset; the
// memory figure for PQ codes is the same payload-floor *estimate* core uses
// (real V8 heap is larger), clearly labelled. Synthetic clustered Gaussians are
// kinder than real embeddings (stage 07 caveat applies), so absolute recall is
// optimistic; the *trends* (nprobe↑⇒recall↑, nsub↑⇒recall↑⇒memory↑, residual
// vs raw gap) are what transfer.

import { l2dist } from './core/vec.js';
import {
  makeDataset,
  makeQueries,
  computeGroundTruth,
  mulberry32,
} from './core/dataset.js';
import {
  recallAtK,
  qps,
  timeIt,
  estimateBytes,
  formatBytes,
} from './core/metrics.js';

// ---------------------------------------------------------------------------
// k-means: shared by both quantizers (IVF's coarse centroids and PQ's per-
// subspace codebooks are both just k-means results). Lloyd's algorithm with
// k-means++ seeding.
//
// Why k-means++ and not random init: random centroids on clustered data often
// drop two seeds in the same blob and none in another, producing a degenerate
// codebook (one centroid serves three real clusters → huge quantization error).
// k-means++ spreads seeds proportional to squared distance, which on our
// well-separated blobs reliably lands one seed per cluster. This is *the*
// difference between an IVF that prunes correctly and one that scatters a
// cluster's points across cells (a failure mode we'd otherwise have to fake).
// ---------------------------------------------------------------------------
interface KMeansResult {
  centroids: number[][];
  // assignment[i] = index of the centroid that point i belongs to. The IVF
  // inverted lists and the PQ codes are both derived from this.
  assignment: number[];
}

function l2sq(a: number[], b: number[]): number {
  // Squared L2. Used everywhere assignment/ranking happens because sqrt is
  // monotonic and never changes which centroid is nearest — paying for it in
  // the innermost loop (run millions of times during k-means) is pure waste.
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

function kmeansPlusPlusInit(
  points: number[][],
  k: number,
  rng: () => number,
): number[][] {
  const centroids: number[][] = [];
  // First centroid: uniform random point. Subsequent: sample with probability
  // proportional to D(x)^2 (distance to nearest chosen centroid).
  centroids.push(points[Math.floor(rng() * points.length)].slice());
  const d2 = new Array<number>(points.length).fill(Infinity);
  for (let c = 1; c < k; c++) {
    let sum = 0;
    for (let i = 0; i < points.length; i++) {
      const dist = l2sq(points[i], centroids[c - 1]);
      if (dist < d2[i]) d2[i] = dist; // running nearest-centroid distance
      sum += d2[i];
    }
    // Weighted pick. sum===0 means all points coincide with a centroid
    // (degenerate, e.g. k > distinct points): fall back to a random point so we
    // never divide by zero or loop forever.
    if (sum === 0) {
      centroids.push(points[Math.floor(rng() * points.length)].slice());
      continue;
    }
    let target = rng() * sum;
    let chosen = points.length - 1;
    for (let i = 0; i < points.length; i++) {
      target -= d2[i];
      if (target <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.push(points[chosen].slice());
  }
  return centroids;
}

function kmeans(
  points: number[][],
  k: number,
  maxIters: number,
  seed: number,
): KMeansResult {
  const dim = points[0].length;
  const rng = mulberry32(seed);
  let centroids = kmeansPlusPlusInit(points, k, rng);
  const assignment = new Array<number>(points.length).fill(-1);

  for (let iter = 0; iter < maxIters; iter++) {
    let moved = 0;
    // Assignment step: each point to its nearest centroid.
    for (let i = 0; i < points.length; i++) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const d = l2sq(points[i], centroids[c]);
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      if (assignment[i] !== best) moved++;
      assignment[i] = best;
    }
    // Convergence: no point changed cluster. Lloyd's is guaranteed to reach a
    // fixed point; we stop early once it does so we don't burn iters spinning.
    if (moved === 0) break;

    // Update step: each centroid becomes the mean of its members.
    const sums = Array.from({ length: k }, () => new Array<number>(dim).fill(0));
    const counts = new Array<number>(k).fill(0);
    for (let i = 0; i < points.length; i++) {
      const c = assignment[i];
      counts[c]++;
      const p = points[i];
      const s = sums[c];
      for (let d = 0; d < dim; d++) s[d] += p[d];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        // Empty cluster: a centroid that won no points. Re-seed it to a random
        // point instead of leaving it stranded (a dead centroid wastes a cell
        // and can never recover on its own). This keeps all k cells live.
        centroids[c] = points[Math.floor(rng() * points.length)].slice();
        continue;
      }
      for (let d = 0; d < dim; d++) centroids[c][d] = sums[c][d] / counts[c];
    }
  }
  return { centroids, assignment };
}

// ---------------------------------------------------------------------------
// Product Quantizer. Splits a dim-vector into `nsub` contiguous sub-vectors and
// k-means-quantizes each sub-space independently into `ksub` (=256) centroids.
// A vector is then stored as `nsub` bytes (one centroid id per sub-space).
//
// Why per-subspace codebooks multiply expressiveness: nsub codebooks of 256
// each represent 256^nsub distinct points using only nsub·256·(dim/nsub) floats
// of codebook. With nsub=4 that's 256^4 ≈ 4.3 billion reconstructions from a
// 4 KB-ish codebook. A single flat codebook would need 4.3B centroids to match.
// ---------------------------------------------------------------------------
const KSUB = 256; // 256 ⇒ exactly one byte per sub-code; the canonical choice.

interface ProductQuantizer {
  nsub: number;
  subDim: number;
  // codebooks[m][c] = centroid c (length subDim) of sub-space m.
  codebooks: number[][][];
}

function subSlice(v: number[], m: number, subDim: number): number[] {
  // Sub-vector m is the contiguous block [m·subDim, (m+1)·subDim). Contiguous
  // (not strided) so that on real embeddings, correlated adjacent dims stay in
  // the same sub-space and quantize together.
  return v.slice(m * subDim, (m + 1) * subDim);
}

function trainProductQuantizer(
  trainVecs: number[][],
  nsub: number,
  seed: number,
): ProductQuantizer {
  const dim = trainVecs[0].length;
  if (dim % nsub !== 0) {
    // Hard failure, not a silent floor-division: an uneven split would drop the
    // tail dimensions from quantization entirely, silently degrading recall.
    throw new Error(`PQ: dim ${dim} not divisible by nsub ${nsub}`);
  }
  const subDim = dim / nsub;
  const codebooks: number[][][] = [];
  for (let m = 0; m < nsub; m++) {
    const subVecs = trainVecs.map((v) => subSlice(v, m, subDim));
    // Distinct seed per sub-space so the nsub codebooks don't share k-means++
    // randomness (which would correlate their failures).
    const km = kmeans(subVecs, KSUB, 25, seed + m * 9973);
    codebooks.push(km.centroids);
  }
  return { nsub, subDim, codebooks };
}

function encode(pq: ProductQuantizer, v: number[]): Uint8Array {
  // Map a vector to its PQ code: nearest centroid id in each sub-space.
  const code = new Uint8Array(pq.nsub);
  for (let m = 0; m < pq.nsub; m++) {
    const sub = subSlice(v, m, pq.subDim);
    let best = 0;
    let bestD = Infinity;
    const book = pq.codebooks[m];
    for (let c = 0; c < KSUB; c++) {
      const d = l2sq(sub, book[c]);
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    code[m] = best;
  }
  return code;
}

// ADC (asymmetric distance computation) lookup table. For one query residual,
// precompute squared distance from each sub-vector of the query to every
// centroid in that sub-space's codebook. Then a candidate's approximate squared
// distance is sum over m of table[m][code[m]] — nsub adds, zero multiplies.
//
// "Asymmetric" because the query stays full-precision while the database side is
// quantized; this loses less than quantizing both (SDC). The whole point of PQ
// at query time is that this table is built once per query and amortized over
// thousands of candidates in the probed cells.
function buildAdcTable(pq: ProductQuantizer, queryResidual: number[]): number[][] {
  const table: number[][] = [];
  for (let m = 0; m < pq.nsub; m++) {
    const qsub = subSlice(queryResidual, m, pq.subDim);
    const row = new Array<number>(KSUB);
    const book = pq.codebooks[m];
    for (let c = 0; c < KSUB; c++) row[c] = l2sq(qsub, book[c]);
    table.push(row);
  }
  return table;
}

function adcDistance(table: number[][], code: Uint8Array): number {
  let sum = 0;
  for (let m = 0; m < table.length; m++) sum += table[m][code[m]];
  return sum; // squared distance estimate; monotonic, fine for top-k ranking
}

// ---------------------------------------------------------------------------
// The IVFPQ index proper.
// ---------------------------------------------------------------------------
interface IvfpqIndex {
  coarseCentroids: number[][]; // nlist IVF cell centroids
  invLists: number[][]; // invLists[cell] = dataset indices assigned to that cell
  codes: Uint8Array[]; // codes[i] = PQ code of vector i (residual-encoded)
  pq: ProductQuantizer;
  useResiduals: boolean; // whether codes encode residuals (correct) or raw (bug demo)
  nlist: number;
}

function buildIvfpq(
  dataset: number[][],
  nlist: number,
  nsub: number,
  useResiduals: boolean,
  seed: number,
): IvfpqIndex {
  // Step 1: IVF coarse quantizer. k-means over the raw vectors gives nlist cells.
  const coarse = kmeans(dataset, nlist, 25, seed);
  const invLists: number[][] = Array.from({ length: nlist }, () => []);
  for (let i = 0; i < dataset.length; i++) invLists[coarse.assignment[i]].push(i);

  // Step 2: build PQ training set. The residual = vector - its cell centroid.
  // Training PQ on residuals (not raw vectors) is the key IVFPQ trick: residuals
  // are centered near the origin and have similar scale across all cells, so a
  // single shared codebook fits them tightly. Raw vectors span the whole 10-wide
  // box, forcing the codebook to waste resolution on inter-cluster offsets it
  // could have gotten for free from the coarse centroid.
  const trainVecs = dataset.map((v, i) => {
    if (!useResiduals) return v;
    const centroid = coarse.centroids[coarse.assignment[i]];
    return v.map((x, d) => x - centroid[d]);
  });
  const pq = trainProductQuantizer(trainVecs, nsub, seed + 12345);

  // Step 3: encode every vector (residual or raw, matching training).
  const codes: Uint8Array[] = dataset.map((v, i) => {
    if (!useResiduals) return encode(pq, v);
    const centroid = coarse.centroids[coarse.assignment[i]];
    return encode(pq, v.map((x, d) => x - centroid[d]));
  });

  return {
    coarseCentroids: coarse.centroids,
    invLists,
    codes,
    pq,
    useResiduals,
    nlist,
  };
}

// Search: find nprobe nearest cells, then ADC-rank all their members.
//
// Failure mode this exposes when nprobe is small: a query that sits in cluster
// overlap can have its true nearest neighbor in an adjacent cell whose centroid
// is the 2nd or 3rd closest. With nprobe=1 we never open that cell, so the true
// neighbor is invisible regardless of how good PQ is — recall is capped by IVF's
// pruning, not by quantization. This is why IVFPQ recall plots have two knobs.
function searchIvfpq(
  index: IvfpqIndex,
  query: number[],
  k: number,
  nprobe: number,
): number[] {
  // Rank cells by distance from query to coarse centroid.
  const cellDists = index.coarseCentroids.map((c, idx) => ({
    idx,
    d: l2sq(query, c),
  }));
  cellDists.sort((a, b) => a.d - b.d);
  const probeCells = cellDists.slice(0, nprobe);

  const candidates: { idx: number; d: number }[] = [];
  for (const cell of probeCells) {
    // The ADC table is residual-aware: when residuals are used, the query must
    // also be expressed relative to *this cell's* centroid before the lookup,
    // because the codes live in residual space. Getting this wrong (using the
    // raw query against residual codes) is a subtle correctness bug; we compute
    // the per-cell query residual so the geometry matches.
    const queryForCell = index.useResiduals
      ? query.map((x, d) => x - index.coarseCentroids[cell.idx][d])
      : query;
    const table = buildAdcTable(index.pq, queryForCell);
    for (const i of index.invLists[cell.idx]) {
      candidates.push({ idx: i, d: adcDistance(table, index.codes[i]) });
    }
  }
  // Top-k by approximate distance, index asc as deterministic tie-break (mirrors
  // computeGroundTruth so ties don't spuriously cost recall).
  candidates.sort((a, b) => a.d - b.d || a.idx - b.idx);
  return candidates.slice(0, k).map((e) => e.idx);
}

function searchAll(
  index: IvfpqIndex,
  queries: number[][],
  k: number,
  nprobe: number,
): number[][] {
  return queries.map((q) => searchIvfpq(index, q, k, nprobe));
}

// ---------------------------------------------------------------------------
// Demo driver.
// ---------------------------------------------------------------------------
function main(): void {
  const N = 5000;
  const DIM = 16;
  const CLUSTERS = 16;
  const K = 10;
  const NLIST = 64; // IVF cells. Rule of thumb ~sqrt(N); 64 ≈ sqrt(5000)·0.9.
  const M_QUERIES = 100;

  console.log('=== Stage 05: IVFPQ (IVF coarse pruning + PQ compression) ===\n');

  const dataset = makeDataset(N, DIM, CLUSTERS, 1);
  const queries = makeQueries(M_QUERIES, DIM, 999);
  const truth = computeGroundTruth(dataset, queries, K, 'l2');
  console.log(
    `dataset: ${N} vecs × ${DIM} dims, ${CLUSTERS} clusters | ${M_QUERIES} queries | k=${K}`,
  );
  console.log(`IVF: nlist=${NLIST} cells\n`);

  // --- Build the canonical residual-trained IVFPQ with nsub=4 ---
  const NSUB = 4; // 16 / 4 = 4 dims per sub-space.
  const index = buildIvfpq(dataset, NLIST, NSUB, true, 7);

  // Memory accounting. Full vectors vs PQ codes — the headline win.
  const fullBytes = estimateBytes(N, DIM); // 8 bytes/component (double)
  const codeBytes = estimateBytes(N, NSUB, 1); // 1 byte per sub-code
  // Codebook overhead is fixed (independent of N): nsub · 256 · subDim doubles.
  const codebookBytes = estimateBytes(NSUB * KSUB, DIM / NSUB);
  console.log('--- memory (payload floor, est.; real V8 heap ~1.5-3x) ---');
  console.log(`full float vectors : ${formatBytes(fullBytes)}`);
  console.log(
    `PQ codes (nsub=${NSUB}) : ${formatBytes(codeBytes)}  ` +
      `(+${formatBytes(codebookBytes)} fixed codebook)`,
  );
  console.log(
    `compression ratio  : ${(fullBytes / codeBytes).toFixed(1)}x ` +
      `(codes only; ${(fullBytes / (codeBytes + codebookBytes)).toFixed(1)}x incl. codebook)\n`,
  );

  // --- nprobe sweep: the IVF recall/speed knob ---
  console.log('--- nprobe sweep (nsub=4, residual-trained) ---');
  console.log('nprobe  recall@10   QPS      cells scanned');
  for (const nprobe of [1, 2, 4, 8, 16, NLIST]) {
    // Warmup pass (JIT) so the timed pass measures steady-state, per metrics.ts.
    searchAll(index, queries.slice(0, 10), K, nprobe);
    let approx: number[][] = [];
    const ms = timeIt(() => {
      approx = searchAll(index, queries, K, nprobe);
    });
    const recall = recallAtK(approx, truth, K);
    const label = nprobe === NLIST ? `${nprobe}*` : `${nprobe}`;
    console.log(
      `${label.padEnd(7)} ${recall.toFixed(4).padStart(8)}   ` +
        `${qps(M_QUERIES, ms).toFixed(0).padStart(6)}   ` +
        `${nprobe}/${NLIST}${nprobe === NLIST ? ' (exhaustive IVF)' : ''}`,
    );
  }
  console.log(
    '  * nprobe=nlist = no IVF pruning: recall ceiling is now pure PQ loss.\n',
  );

  // --- nsub sweep: the PQ recall/memory knob (fixed generous nprobe) ---
  // Higher nsub = finer quantization = more bytes per code but less error.
  console.log('--- nsub sweep (nprobe=8, residual-trained) ---');
  console.log('nsub  bytes/vec  recall@10   compression');
  const FIXED_NPROBE = 8;
  for (const nsub of [1, 2, 4, 8]) {
    const idx = buildIvfpq(dataset, NLIST, nsub, true, 7);
    const approx = searchAll(idx, queries, K, FIXED_NPROBE);
    const recall = recallAtK(approx, truth, K);
    const ratio = (DIM * 8) / nsub; // per-vector compression vs full double vec
    console.log(
      `${String(nsub).padEnd(5)} ${String(nsub).padStart(8)}  ` +
        `${recall.toFixed(4).padStart(8)}   ${ratio.toFixed(1)}x`,
    );
  }
  console.log(
    '  nsub=1 = one byte for the whole 16-dim vector: maximal compression,\n' +
      '  maximal quantization error → recall collapses (under-coded failure mode).\n',
  );

  // --- FAILURE MODE A: PQ on raw vectors instead of residuals ---
  // Same nlist/nsub/nprobe, only the residual trick toggled off. The gap is the
  // value of training PQ in residual space — usually several recall points.
  console.log('--- failure mode A: residual vs raw PQ training (nsub=4, nprobe=8) ---');
  const rawIndex = buildIvfpq(dataset, NLIST, NSUB, false, 7);
  const rawApprox = searchAll(rawIndex, queries, K, FIXED_NPROBE);
  const residApprox = searchAll(index, queries, K, FIXED_NPROBE);
  const rawRecall = recallAtK(rawApprox, truth, K);
  const residRecall = recallAtK(residApprox, truth, K);
  console.log(`  residual-trained PQ : recall@10 = ${residRecall.toFixed(4)}`);
  console.log(`  raw-trained PQ      : recall@10 = ${rawRecall.toFixed(4)}`);
  console.log(
    `  gap                 : ${(residRecall - rawRecall).toFixed(4)} ` +
      `(residual training recovers this much recall for free)\n`,
  );

  // --- FAILURE MODE B: nprobe=1 misses neighbors in adjacent cells ---
  // Quantify how many queries lose their true #1 neighbor purely to IVF pruning
  // (not PQ): a true-neighbor-in-unprobed-cell count, independent of codes.
  console.log('--- failure mode B: nprobe=1 cell-miss on true #1 neighbor ---');
  // nprobe=1: each query opens exactly the single nearest coarse cell (below).
  let missedByPruning = 0;
  for (let q = 0; q < queries.length; q++) {
    const trueTop1 = truth[q][0];
    // Which cell does the true #1 neighbor live in?
    const trueCell = index.invLists.findIndex((list) => list.includes(trueTop1));
    // Which single cell does nprobe=1 open for this query?
    const cellDists = index.coarseCentroids.map((c, idx) => ({
      idx,
      d: l2sq(queries[q], c),
    }));
    cellDists.sort((a, b) => a.d - b.d);
    const probedCell = cellDists[0].idx;
    if (trueCell !== probedCell) missedByPruning++;
  }
  console.log(
    `  ${missedByPruning}/${M_QUERIES} queries have their true #1 neighbor in an UNPROBED cell.`,
  );
  console.log(
    '  These are unrecoverable at nprobe=1 no matter how good PQ is — the fix is\n' +
      '  more nprobe (see sweep above), trading QPS for recall.\n',
  );

  // --- Reference: exact brute-force baseline for the trade context ---
  let bruteApprox: number[][] = [];
  const bruteMs = timeIt(() => {
    bruteApprox = queries.map((q) => {
      const scored = dataset.map((v, idx) => ({ idx, d: l2dist(q, v) }));
      scored.sort((a, b) => a.d - b.d || a.idx - b.idx);
      return scored.slice(0, K).map((e) => e.idx);
    });
  });
  console.log('--- baseline: exact brute force ---');
  console.log(
    `  recall@10 = ${recallAtK(bruteApprox, truth, K).toFixed(4)} (by definition 1.0) | ` +
      `QPS = ${qps(M_QUERIES, bruteMs).toFixed(0)} | memory = ${formatBytes(fullBytes)}`,
  );
  console.log(
    '\nTakeaway: IVFPQ trades two independent recall losses (cell pruning + code\n' +
      'quantization) for a large memory cut and a QPS win. Tune nprobe for recall,\n' +
      'nsub for the memory/recall point, and always train PQ on residuals.',
  );
}

main();
