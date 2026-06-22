// core/rng.ts — The single source of randomness for the whole book.
//
// WHY a hand-rolled seeded PRNG instead of Math.random():
//   Math.random() has no seed API in JS, so nothing is reproducible across runs. Every
//   honest number in this book — a probe accuracy, a patch recovery %, an SAE feature's
//   activation — must be bit-for-bit reproducible: same seed => same model => same
//   circuit => same measurement. Interpretability claims that don't reproduce are not
//   claims, they're noise. mulberry32 is a tiny deterministic 32-bit generator with
//   good-enough statistics for teaching (NOT cryptographic).
//
// INVARIANT: nothing anywhere may call Math.random(). All stochastic ops (weight init,
//   batch sampling, probe init, SAE init, tie-breaking) thread an Rng made here.
//
// FAILURE MODE: sharing ONE Rng instance across two logically-independent streams couples
//   them — drawing from one shifts the other, so "fix the model seed, vary the data seed"
//   silently fails. Make a fresh mulberry32(seed) per independent stream, or couple on
//   purpose and say so.

export type Rng = () => number; // returns a float in [0, 1)

/**
 * mulberry32: 32-bit state, one multiply + xorshift per draw. The returned closure holds
 * mutable state; each call advances the stream. INVARIANT: state stays an unsigned 32-bit
 * int via `>>> 0` — drop that and the stream diverges across engines.
 */
export function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // /2^32 -> [0,1)
  };
}

/**
 * Standard normal N(0,1) via Box–Muller.
 * WHY Box–Muller: exact transform of two uniforms into one normal, no rejection loop, so
 *   the draw count per call is CONSTANT (=2). Constant draw count keeps stream alignment
 *   predictable — if init silently used a variable number of draws, swapping one layer's
 *   size would desync every later draw and break reproducibility of unrelated components.
 * FAILURE MODE: u1 == 0 -> log(0) = -Inf. We clamp u1 away from 0. (We discard the second
 *   Box–Muller output for code simplicity; the cost is one extra draw, paid consistently.)
 */
export function gaussian(rng: Rng, mean = 0, std = 1): number {
  let u1 = rng();
  const u2 = rng();
  if (u1 < 1e-12) u1 = 1e-12;
  return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Sample an index from a categorical distribution given by `probs` (need not be exactly
 * normalized; we divide by the running total). One rng draw. Used for sampling model
 * outputs when we want stochastic decoding in a reproducible way.
 * FAILURE MODE: all-zero probs -> division by zero. We guard and fall back to argmax-ish
 *   last index, which is wrong-but-loud rather than a silent NaN.
 */
export function sampleCategorical(probs: number[] | Float64Array, rng: Rng): number {
  let total = 0;
  for (let i = 0; i < probs.length; i++) total += probs[i];
  if (total <= 0) return probs.length - 1; // degenerate input, fail loud-ish
  const r = rng() * total;
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1; // float rounding fallthrough
}

/**
 * argmax with deterministic FIRST-wins tie-breaking. WHY deterministic ties matter in
 * interpretability: accuracy and "did the model predict the oracle answer?" hinge on the
 * argmax; a nondeterministic tie-break would make borderline cases flicker run-to-run and
 * pollute every downstream metric. Use this when you want reproducibility over fairness.
 */
export function argmax(xs: number[] | Float64Array): number {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] > bestVal) {
      bestVal = xs[i];
      best = i;
    }
  }
  return best;
}

/**
 * argmax with RANDOM tie-breaking among the maxima. WHY a separate function: at model
 * init, logits are near-uniform and many classes tie; first-wins argmax would bias every
 * prediction toward low indices and make "untrained accuracy" look structured when it's
 * not. Use this for honest random baselines; use argmax() for reproducible eval.
 */
export function argmaxRandomTie(xs: number[] | Float64Array, rng: Rng): number {
  let bestVal = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > bestVal) bestVal = xs[i];
  const winners: number[] = [];
  for (let i = 0; i < xs.length; i++) if (xs[i] === bestVal) winners.push(i);
  return winners[Math.floor(rng() * winners.length)];
}

/**
 * In-place Fisher–Yates shuffle. INVARIANT: draw j in [0, i] INCLUSIVE; the off-by-one
 * (drawing [0, i)) is a classic bug that biases the permutation distribution. Returns the
 * same array for chaining. Used for train/val splits and probe data shuffling.
 */
export function shuffle<T>(arr: T[], rng: Rng): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}
