// core/rng.ts — Seeded PRNG + samplers, the single source of all randomness in this book.
//
// WHY a hand-rolled PRNG instead of Math.random():
//   Math.random() has no seed API in JS, so results are not reproducible across runs.
//   Every "honest number" claim in this book (loss curves, grad-check error, speedups)
//   depends on bit-for-bit reproducibility: same machine + same seed => same output.
//   mulberry32 is a tiny 32-bit generator with good-enough statistical quality for
//   teaching (NOT cryptographic). It is deterministic and self-contained.
//
// INVARIANT: nothing in stages may call Math.random() directly. All stochastic ops
//   (weight init, shuffling, batch sampling, augmentation jitter) must thread an Rng
//   made here.
//
// FAILURE MODE: reusing the SAME Rng instance in two places couples their streams —
//   advancing one perturbs the other. Make a fresh rng(seed) per logically-independent
//   stream, or accept the coupling deliberately.

export type Rng = () => number; // returns a float in [0, 1)

/**
 * mulberry32: 32-bit state, one multiply + xorshift per draw.
 * Returns a closure holding mutable state — calling it advances the stream.
 * INVARIANT: state is kept as an unsigned 32-bit int via `>>> 0`.
 */
export function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // divide by 2^32 -> [0,1)
  };
}

/**
 * Standard normal sample via Box–Muller.
 * WHY Box–Muller: turns two uniforms into one N(0,1) sample with exact math, no
 *   rejection loop, so the number of rng draws per call is constant (=2) — keeps
 *   the stream alignment predictable across runs.
 * FAILURE MODE: u1 can be exactly 0 -> log(0) = -Inf. We clamp u1 away from 0.
 *   (We discard the second Box–Muller output for simplicity; cost is one extra draw.)
 */
export function randn(rng: Rng): number {
  let u1 = rng();
  const u2 = rng();
  if (u1 < 1e-12) u1 = 1e-12;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Uniform sample in [lo, hi). Convenience for augmentation jitter / random positions. */
export function uniform(rng: Rng, lo: number, hi: number): number {
  return lo + (hi - lo) * rng();
}

/** Uniform integer in [lo, hi] inclusive. Used for random shape placement / crop offsets. */
export function randint(rng: Rng, lo: number, hi: number): number {
  // +1 because hi is inclusive; off-by-one here would never place at the last position.
  return lo + Math.floor(rng() * (hi - lo + 1));
}

/**
 * In-place Fisher–Yates shuffle.
 * INVARIANT: every permutation is equally likely given a uniform rng; we draw j in
 *   [0, i] (inclusive) — off-by-one here (drawing [0, i)) biases the distribution.
 * Returns the same array for convenient chaining.
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

/**
 * Kaiming/He init for ReLU-family layers: N(0, 2/fanIn).
 * WHY 2/fanIn: ReLU zeroes ~half its inputs, halving variance; the factor 2 restores
 *   forward-activation variance so deep stacks neither explode nor vanish.
 * FAILURE MODE: using xavier (1/fanIn) before ReLU makes deep nets' activations decay.
 *
 * For Conv2d the convention is fanIn = inChannels * kH * kW (the number of input
 *   weights feeding one output unit) — the caller passes that product, not just channels.
 */
export function kaiming(fanIn: number, rng: Rng): number {
  const std = Math.sqrt(2 / fanIn);
  return randn(rng) * std;
}

/**
 * Xavier/Glorot init for tanh/linear: N(0, 2/(fanIn+fanOut)).
 * WHY both fans: keeps variance of activations AND gradients roughly constant, which
 *   matters for symmetric saturating nonlinearities (tanh) where ReLU's logic fails.
 */
export function xavier(fanIn: number, fanOut: number, rng: Rng): number {
  const std = Math.sqrt(2 / (fanIn + fanOut));
  return randn(rng) * std;
}
