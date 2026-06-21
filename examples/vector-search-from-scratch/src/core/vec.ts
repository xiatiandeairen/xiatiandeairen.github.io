// core/vec.ts — the four primitives every index in this book is built on.
//
// Why these four and nothing else: a vector index is, at its heart, a strategy
// for *avoiding* distance computations. But it can never avoid all of them — at
// the leaves it must still compare two vectors. So the entire engine bottoms out
// in dot / cosine / l2. Getting these right (and fast) matters more than any
// clever index, because they run in the innermost loop.
//
// Design choices:
//  - Plain `number[]`, not Float32Array. Slower, but the book is about *mechanism*,
//    not micro-optimization; readers can see every multiply. Stage 07 discusses
//    the Float32Array / SIMD gap as a "what we left on the table" note.
//  - No length guards in the hot path on purpose: a dimension mismatch is a bug
//    in the caller's index construction, not a runtime condition to handle. We
//    assert in debug-style helpers (normalize) but keep dot/l2 branch-free.

// Dot product. The single most important operation in the library: cosine and
// (for normalized vectors) nearest-neighbor both reduce to it. NaN propagates if
// either input contains NaN — we rely on that as a loud failure signal rather
// than silently producing 0.
export function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// Squared L2 norm helper. Squared (no sqrt) because callers that only need to
// *compare* magnitudes (e.g. normalize's zero check, ranking) never need the
// actual length, and sqrt is the expensive part.
function normSq(a: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return s;
}

// Cosine similarity in [-1, 1]: 1 = same direction, 0 = orthogonal, -1 = opposite.
// Failure mode this guards: a zero vector has no direction, so cosine is
// undefined. Returning 0 (orthogonal) is the least-harmful convention — it keeps
// the vector out of any top-k instead of poisoning the ranking with NaN. Stage 02
// demonstrates why you should normalize *once at insert time* instead of paying
// for these two sqrts on every query.
export function cosineSim(a: number[], b: number[]): number {
  const denom = Math.sqrt(normSq(a)) * Math.sqrt(normSq(b));
  if (denom === 0) return 0;
  return dot(a, b) / denom;
}

// Euclidean (L2) distance. Smaller = closer (opposite ordering from cosine).
// We take the sqrt here because some metrics (PQ reconstruction error in stage 04)
// report human-readable distances; index *ranking* code should prefer the squared
// form to skip the sqrt, since sqrt is monotonic and never changes the order.
export function l2dist(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

// Return a unit-length copy. Invariant on the output: ||normalize(v)|| === 1
// for any non-zero v (up to float error ~1e-15). This is the trick that lets an
// index store vectors once and answer cosine queries with a plain dot product:
// cosine(a,b) === dot(normalize(a), normalize(b)). Zero vectors can't be
// normalized — we return a fresh zero copy rather than dividing by 0, so the
// invariant "output norm is 0 or 1, never NaN" holds.
export function normalize(v: number[]): number[] {
  const n = Math.sqrt(normSq(v));
  if (n === 0) return v.slice();
  const out = new Array<number>(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}
