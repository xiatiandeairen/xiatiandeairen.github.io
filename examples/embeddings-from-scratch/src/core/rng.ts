// core/rng.ts — the single source of randomness for the whole book.
//
// Why a hand-rolled PRNG instead of Math.random: reproducibility is a hard
// requirement here. Every claim in the book ("loss dropped from 4.1 to 0.9",
// "king-man+woman lands on queen") must be regenerable bit-for-bit by a reader
// on a cold machine. Math.random has no seed, so a single call anywhere would
// make a run non-deterministic. Rule for every stage: NEVER call Math.random.
// Anything stochastic — weight init, negative sampling, dropout order, toy
// corpus generation, shuffling — pulls from an Rng created with a fixed seed.
//
// mulberry32 is chosen on purpose: 32-bit state, ~10 lines, passes basic
// statistical sanity for toy ML, and is trivial to audit. It is NOT
// cryptographic and NOT for Monte-Carlo where period matters — but the book is
// about mechanism, and a reader can read the whole generator in one screen.
//
// Failure mode it guards against: two stages sharing one global generator would
// couple their results (running stage A first changes stage B's "random"
// numbers). So Rng is an *object* you instantiate per concern; there is no
// module-level singleton on purpose.

export class Rng {
  // 32-bit unsigned state. We keep it as a number and mask with >>> 0 after
  // every update so the arithmetic stays in the unsigned 32-bit domain even
  // though JS numbers are doubles.
  private state: number;

  constructor(seed: number) {
    // Force seed into uint32. Seed 0 is fine for mulberry32 (unlike some LCGs
    // that get stuck at 0), so we do not special-case it.
    this.state = seed >>> 0;
  }

  // Core step: returns a float in [0, 1). This is the only place the bit-mixing
  // lives; every other method is a transform on top of it, which is what keeps
  // the stream deterministic and auditable.
  nextFloat(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

// Uniform float in [min, max). Default [0,1). Used for small symmetric weight
// init when we want a bounded range rather than a Gaussian tail.
export function sampleUniform(rng: Rng, min = 0, max = 1): number {
  return min + (max - min) * rng.nextFloat();
}

// Standard-normal sample via Box–Muller. Why Gaussian init at all: for embedding
// matrices, zero-mean small-variance Gaussian breaks symmetry between rows while
// keeping the initial dot products near zero, so the first softmax/sigmoid is
// near-uniform and gradients are well-scaled. We discard the second Box–Muller
// value (slightly wasteful) to keep the call site one-number-in-one-number-out;
// performance is irrelevant at toy scale.
export function gaussian(rng: Rng, mean = 0, std = 1): number {
  // u1 must be > 0 because we take log(u1); nextFloat() returns [0,1) so 0 is
  // possible. Clamp to a tiny epsilon to avoid log(0) = -Infinity → NaN weights,
  // which would silently poison the entire model.
  let u1 = rng.nextFloat();
  if (u1 < 1e-12) u1 = 1e-12;
  const u2 = rng.nextFloat();
  const mag = Math.sqrt(-2 * Math.log(u1));
  return mean + std * mag * Math.cos(2 * Math.PI * u2);
}

// Sample an index from a categorical distribution given by (unnormalized)
// weights. This is the workhorse for negative sampling: word2vec draws negatives
// from the unigram^0.75 distribution, which is exactly "weights → index". We
// accept unnormalized weights and normalize internally so callers can pass raw
// counts.
//
// Invariant: weights must be non-negative and have positive sum. A zero-sum or
// all-negative array is a caller bug (e.g. empty vocab) — we throw rather than
// return a misleading 0, because a silent 0 here would make every negative
// sample the same token and quietly destroy training signal.
export function sampleCategorical(rng: Rng, weights: number[]): number {
  let total = 0;
  for (const w of weights) {
    if (w < 0) throw new Error("sampleCategorical: negative weight");
    total += w;
  }
  if (total <= 0) throw new Error("sampleCategorical: non-positive weight sum");
  let r = rng.nextFloat() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r < 0) return i;
  }
  // Floating-point drift can leave r >= 0 after the loop on the last bucket.
  // Returning the last index is the correct fallback, not an error.
  return weights.length - 1;
}

// Fisher–Yates shuffle, seeded and in-place. Used to randomize training-pair
// order each epoch. In-place return-the-same-array keeps allocation out of the
// epoch loop. Determinism comes entirely from the passed-in rng, so two runs
// with the same seed visit pairs in the same order — important because SGD's
// path depends on order.
export function shuffle<T>(rng: Rng, arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng.nextFloat() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}
