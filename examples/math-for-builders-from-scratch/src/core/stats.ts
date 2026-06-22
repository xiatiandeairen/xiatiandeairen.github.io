// core/stats.ts — Descriptive statistics shared by probability, statistics, info-theory.
//
// WHY hand-rolled instead of a stats package:
//   The whole point of this book is that "mean", "variance", "quantile" are three lines
//   of arithmetic, not magic. A reader who has seen the loop will never again be confused
//   about why sample variance divides by n-1, or why a quantile needs a sorted copy.
//
// DESIGN: every function takes readonly number[] and returns a number (or a Histogram).
//   No mutation of inputs — stages reuse the same dataset for multiple summaries.
//
// FAILURE MODE (shared): an empty array has no mean/variance. We throw with a clear
//   message rather than return NaN, because NaN propagates silently and corrupts a whole
//   downstream pipeline before anyone notices.

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`[stats] ${msg}`);
}

/** Arithmetic mean. INVARIANT: xs.length > 0. */
export function mean(xs: readonly number[]): number {
  assert(xs.length > 0, "mean: empty input");
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/**
 * Variance. `sample=true` (default) divides by n-1 (Bessel's correction); `sample=false`
 * divides by n (population variance).
 * WHY n-1 for a sample: dividing by n systematically UNDER-estimates spread because the
 *   mean is itself estimated from the same data, so the deviations are artificially small.
 *   n-1 is the unbiased correction. Stages that have the full population (not a sample)
 *   pass sample=false on purpose.
 * FAILURE MODE: sample variance of a single point is undefined (n-1 = 0); we throw.
 */
export function variance(xs: readonly number[], sample = true): number {
  const n = xs.length;
  assert(n > (sample ? 1 : 0), "variance: need >1 point for sample variance");
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return s / (sample ? n - 1 : n);
}

/** Standard deviation = sqrt(variance). Same sample flag semantics. */
export function std(xs: readonly number[], sample = true): number {
  return Math.sqrt(variance(xs, sample));
}

/**
 * Quantile q in [0,1] via linear interpolation between order statistics (the common
 * "type 7" / numpy-default method).
 * WHY interpolate: with few points the exact q-th value usually falls between two samples;
 *   interpolation gives a smooth, monotone-in-q estimate instead of a jumpy step function.
 * INVARIANT: q clamped to [0,1]; input is copied before sorting (caller's order preserved).
 */
export function quantile(xs: readonly number[], q: number): number {
  assert(xs.length > 0, "quantile: empty input");
  const qq = Math.min(1, Math.max(0, q));
  const s = xs.slice().sort((a, b) => a - b);
  const pos = qq * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return s[lo];
  const frac = pos - lo;
  return s[lo] * (1 - frac) + s[hi] * frac;
}

export interface Histogram {
  /** Bin edges, length = bins + 1 (left-closed, right-open except the last bin). */
  edges: number[];
  /** Count per bin, length = bins. INVARIANT: sum(counts) === xs.length. */
  counts: number[];
}

/**
 * Equal-width histogram over [min, max] of the data.
 * WHY include the max in the last bin: a strict right-open scheme would drop the single
 *   largest value into "no bin", losing a count and breaking sum(counts) === n. We special-
 *   case the maximum into the final bin so the invariant holds — a classic off-by-one.
 * FAILURE MODE: all values equal => zero-width range. We widen by 1 so every value lands
 *   in bin 0 instead of dividing by a zero range (which produces NaN bin indices).
 */
export function histogram(xs: readonly number[], bins: number): Histogram {
  assert(xs.length > 0, "histogram: empty input");
  assert(bins > 0, "histogram: need >=1 bin");
  let lo = Math.min(...xs);
  let hi = Math.max(...xs);
  if (lo === hi) hi = lo + 1; // degenerate range guard
  const width = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const x of xs) {
    let idx = Math.floor((x - lo) / width);
    if (idx >= bins) idx = bins - 1; // fold the max into the last bin
    if (idx < 0) idx = 0;
    counts[idx]++;
  }
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + i * width);
  return { edges, counts };
}
