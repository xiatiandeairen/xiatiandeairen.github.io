// stage02-similarity.ts — what "similar" even means, and why high dimensions
// quietly break the whole idea of a "nearest" neighbor.
//
// This stage teaches three things, in order:
//   1. The three metrics (dot / cosine / l2) are not interchangeable — dot
//      rewards magnitude, cosine ignores it, l2 measures displacement. Picking
//      the wrong one silently returns the wrong neighbors.
//   2. The single most useful identity in this book: for *unit* vectors,
//      dot(a,b) === cosine(a,b). This is why every index normalizes once at
//      insert time and then answers cosine queries with a bare dot product
//      (one multiply-add per dim, no per-query sqrt).
//   3. The concentration of distances ("curse of dimensionality"): as dim grows,
//      the farthest and nearest points in a random set become almost
//      equidistant, so the ratio farthest/nearest → 1. When that ratio is ~1,
//      "the nearest neighbor" is barely distinguishable from a random point —
//      the premise the rest of the book is built on starts to wobble, and only
//      data that *clusters* (not uniform noise) rescues it.
//
// Everything here is deterministic: same seed ⇒ same numbers on any machine.
// Run: npx tsx src/stage02-similarity.ts

import { dot, cosineSim, l2dist, normalize } from './core/vec.js';
import { mulberry32 } from './core/dataset.js';

// ---------------------------------------------------------------------------
// Part 1 — the three metrics disagree, on purpose.
// ---------------------------------------------------------------------------
//
// We use one query and three candidates chosen so that *each metric picks a
// different winner*. That divergence is the whole point: if you grab "the
// similarity function" without thinking, you get whichever neighbor that
// function happens to favor, not the one your task wants.
function demoMetricsDisagree(): void {
  console.log('=== Part 1: the three metrics rank differently ===\n');

  const query = [1, 1];

  // Chosen so all three metrics pick a DIFFERENT winner:
  //  a: huge magnitude, off-direction → wins dot (magnitude dominates the product).
  //  b: identical direction, modest length → wins cosine (perfect angle).
  //  c: spatially nearest to the query → wins l2 (smallest displacement).
  const candidates: Record<string, number[]> = {
    a_huge_off_dir: [9, 1], // dot=10 here, but angle is off
    b_same_dir: [3, 3], // cosine=1 (exact direction match)
    c_spatially_near: [1.5, 1.2], // closest point in space
  };

  console.log(`query = [${query}]`);
  console.log('name              dot      cosine    l2dist');
  for (const [name, v] of Object.entries(candidates)) {
    const d = dot(query, v);
    const c = cosineSim(query, v);
    const l = l2dist(query, v);
    console.log(
      `${name.padEnd(16)} ${d.toFixed(3).padStart(7)} ${c
        .toFixed(4)
        .padStart(8)} ${l.toFixed(4).padStart(9)}`,
    );
  }

  // Read the winners off the table programmatically so the prose can't drift
  // from the numbers if the candidates ever change.
  const names = Object.keys(candidates);
  const byDot = [...names].sort(
    (x, y) => dot(query, candidates[y]) - dot(query, candidates[x]),
  )[0];
  const byCos = [...names].sort(
    (x, y) => cosineSim(query, candidates[y]) - cosineSim(query, candidates[x]),
  )[0];
  const byL2 = [...names].sort(
    (x, y) => l2dist(query, candidates[x]) - l2dist(query, candidates[y]),
  )[0];

  console.log(`\ndot   picks: ${byDot}   (rewards raw magnitude)`);
  console.log(`cosine picks: ${byCos}   (rewards direction, ignores length)`);
  console.log(`l2    picks: ${byL2}   (rewards spatial closeness)`);
  console.log(
    '\nLesson: three metrics, three different winners on the SAME query. dot crowned\n' +
      'the longest vector despite its bad angle; cosine crowned the exact-direction\n' +
      'one ignoring length; l2 crowned the spatially closest. The metric is a\n' +
      'modeling choice, not a detail.\n',
  );
}

// ---------------------------------------------------------------------------
// Part 2 — normalize once, then cosine IS a dot product.
// ---------------------------------------------------------------------------
//
// The identity: cosine(a,b) = dot(a,b) / (||a||·||b||). If ||a||=||b||=1, the
// denominator is 1, so cosine(a,b) = dot(â,b̂). We verify it on random vectors
// (so it's not cherry-picked) and report the worst floating-point gap, which
// must be ~1e-15, not zero — proving the equality holds up to double precision,
// not by luck.
function demoNormalizedDotEqualsCosine(): void {
  console.log('=== Part 2: after normalize, dot == cosine ===\n');

  const rng = mulberry32(7);
  const dim = 8;
  const trials = 1000;
  let maxAbsErr = 0;

  for (let t = 0; t < trials; t++) {
    const a = Array.from({ length: dim }, () => rng() * 2 - 1);
    const b = Array.from({ length: dim }, () => rng() * 2 - 1);
    const cosRaw = cosineSim(a, b);
    const dotUnit = dot(normalize(a), normalize(b));
    const err = Math.abs(cosRaw - dotUnit);
    if (err > maxAbsErr) maxAbsErr = err;
  }

  console.log(`trials: ${trials} random pairs, dim ${dim}`);
  console.log(`max |cosineSim(a,b) - dot(normalize(a),normalize(b))| = ${maxAbsErr.toExponential(3)}`);
  console.log(
    maxAbsErr < 1e-12
      ? 'OK: equal to double precision (the ~1e-16..1e-15 residue is float rounding, not a bug).'
      : 'UNEXPECTED: gap too large — investigate.',
  );

  // Why this matters operationally, with a concrete count: cosineSim pays two
  // sqrt per call (the two norms); a pre-normalized dot pays zero. On a 1M-vector
  // scan that is 2M sqrt avoided per query. We don't time it here (stage 07 does
  // the throughput work) — the point is the *algebraic* license to skip them.
  console.log(
    'Operational payoff: store vectors normalized once, answer cosine queries with\n' +
      'a bare dot product — 0 sqrt per comparison instead of 2.\n',
  );
}

// ---------------------------------------------------------------------------
// Part 2b — the failure mode normalize-once invites: zero vectors.
// ---------------------------------------------------------------------------
//
// "Normalize then dot" has a trap: a zero vector has no direction. A naive
// normalize divides by 0 and poisons every score with NaN, which then
// propagates through any sum/sort and silently corrupts a whole top-k. core/vec
// defends against this (returns 0 = orthogonal). We DEMONSTRATE both the danger
// and the guard so the reader sees why the convention exists.
function demoZeroVectorFailure(): void {
  console.log('=== Part 2b: failure mode — the zero vector ===\n');

  const zero = [0, 0, 0];
  const real = [1, 2, 3];

  // What a naive hand-rolled normalize (divide by norm, no guard) would produce:
  const naiveNorm = (v: number[]): number[] => {
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return v.map((x) => x / n); // n===0 ⇒ x/0 ⇒ NaN for every component
  };
  const naive = dot(naiveNorm(zero), normalize(real));

  // What the library's guarded path produces:
  const guarded = cosineSim(zero, real);

  console.log(`naive normalize(zero) = [${naiveNorm(zero)}]  ⇒ dot = ${naive}`);
  console.log(`cosineSim(zero, real) = ${guarded}   (guarded: zero treated as orthogonal)`);
  console.log(
    `\nWhy it bites: a single NaN in a candidate score makes NaN-vs-anything\n` +
      `comparisons false, so the bad vector neither sorts high nor low — it can\n` +
      `silently displace a real neighbor from top-k. The fix is the boring guard\n` +
      `(norm===0 ⇒ score 0), not "hope no zero vectors show up".\n`,
  );
}

// ---------------------------------------------------------------------------
// Part 3 — the curse of dimensionality, measured.
// ---------------------------------------------------------------------------
//
// Claim under test: for N points drawn UNIFORMLY at random in [0,1]^dim, pick a
// query and look at its nearest and farthest of the N. As dim grows, those two
// distances converge — the contrast (farthest/nearest) collapses toward 1.
//
// We measure it directly (no formula): for each dim, average the ratio over
// several independent queries (≥ a handful, not N=1, so we separate signal from
// PRNG noise). We print the curve so a chapter can quote real numbers.
//
// Honesty note baked into the design: we ALSO run the same measurement on
// *clustered* data (the kind core/dataset makes), to show the curse is a
// property of *unstructured* high-dim data — real embeddings cluster, which is
// exactly why ANN search is possible at all despite the curse.
type ContrastRow = {
  dim: number;
  meanRatio: number; // farthest / nearest, averaged over queries
  meanNearest: number;
  meanFarthest: number;
};

function measureContrast(
  makePoint: (rng: () => number, dim: number) => number[],
  dim: number,
  n: number,
  queries: number,
  rng: () => number,
): ContrastRow {
  let ratioSum = 0;
  let nearSum = 0;
  let farSum = 0;

  for (let q = 0; q < queries; q++) {
    const query = makePoint(rng, dim);
    const points = Array.from({ length: n }, () => makePoint(rng, dim));

    let nearest = Infinity;
    let farthest = 0;
    for (const p of points) {
      const d = l2dist(query, p);
      if (d < nearest) nearest = d;
      if (d > farthest) farthest = d;
    }
    // nearest can be ~0 only if a point coincides with the query; with continuous
    // random coords that has measure zero, but guard anyway so one unlucky draw
    // can't produce Infinity and wreck the average.
    if (nearest === 0) continue;

    ratioSum += farthest / nearest;
    nearSum += nearest;
    farSum += farthest;
  }

  return {
    dim,
    meanRatio: ratioSum / queries,
    meanNearest: nearSum / queries,
    meanFarthest: farSum / queries,
  };
}

function demoCurseOfDimensionality(): void {
  console.log('=== Part 3: curse of dimensionality (measured, not asserted) ===\n');

  const dims = [2, 16, 128, 1024];
  const n = 1000; // points per trial
  const queries = 20; // independent queries averaged ⇒ ratio is signal, not 1 lucky draw

  // Uniform point in [0,1]^dim: maximally unstructured, the worst case for ANN.
  const uniformPoint = (rng: () => number, dim: number): number[] =>
    Array.from({ length: dim }, () => rng());

  // Clustered point: pick one of 8 fixed centers and add small Gaussian-ish
  // noise. Same shape as core/dataset's clusters; stands in for "real embeddings
  // have topical structure". We inline a tiny version rather than import a stageNN.
  const clusterCount = 8;
  const makeClusteredPoint = (centerSeed: number) => {
    return (rng: () => number, dim: number): number[] => {
      // Deterministic center per (cluster index) so all points share the same
      // 8 centers across calls within a dim — that shared structure is what
      // creates a *real* nearest neighbor to find.
      const c = Math.floor(rng() * clusterCount);
      const centerRng = mulberry32(centerSeed + c);
      return Array.from({ length: dim }, () => centerRng() + (rng() - 0.5) * 0.1);
    };
  };

  console.log('UNIFORM random data (no structure — the curse in full force):');
  console.log('dim    nearest   farthest   ratio(far/near)');
  const rngU = mulberry32(101);
  const uniformRows: ContrastRow[] = [];
  for (const dim of dims) {
    const row = measureContrast(uniformPoint, dim, n, queries, rngU);
    uniformRows.push(row);
    console.log(
      `${String(dim).padStart(4)}  ${row.meanNearest.toFixed(4).padStart(8)}  ${row.meanFarthest
        .toFixed(4)
        .padStart(9)}  ${row.meanRatio.toFixed(4).padStart(10)}`,
    );
  }

  console.log('\nCLUSTERED data (8 centers — the structure real embeddings have):');
  console.log('dim    nearest   farthest   ratio(far/near)');
  const rngC = mulberry32(202);
  const clusteredPoint = makeClusteredPoint(303);
  const clusteredRows: ContrastRow[] = [];
  for (const dim of dims) {
    const row = measureContrast(clusteredPoint, dim, n, queries, rngC);
    clusteredRows.push(row);
    console.log(
      `${String(dim).padStart(4)}  ${row.meanNearest.toFixed(4).padStart(8)}  ${row.meanFarthest
        .toFixed(4)
        .padStart(9)}  ${row.meanRatio.toFixed(4).padStart(10)}`,
    );
  }

  const uHi = uniformRows[uniformRows.length - 1];
  const uLo = uniformRows[0];
  const cHi = clusteredRows[clusteredRows.length - 1];
  console.log(
    `\nReading the curves:\n` +
      `  Uniform: ratio falls ${uLo.meanRatio.toFixed(2)} (dim ${uLo.dim}) → ` +
      `${uHi.meanRatio.toFixed(2)} (dim ${uHi.dim}). As ratio → 1, "nearest" and\n` +
      `  "farthest" become nearly the same distance — a nearest-neighbor query in\n` +
      `  high-dim uniform noise returns something barely closer than a random point.\n` +
      `  Clustered: ratio stays ${cHi.meanRatio.toFixed(2)} at dim ${cHi.dim} — much\n` +
      `  larger than uniform's ${uHi.meanRatio.toFixed(2)}. Structure preserves contrast.\n` +
      `\nWhy the book works anyway: ANN search exploits exactly this gap. On uniform\n` +
      `noise there is no neighborhood to find and every index degrades to brute\n` +
      `force; on clustered (real) data the contrast survives, so pruning is safe.\n`,
  );
}

function main(): void {
  demoMetricsDisagree();
  demoNormalizedDotEqualsCosine();
  demoZeroVectorFailure();
  demoCurseOfDimensionality();
}

main();
