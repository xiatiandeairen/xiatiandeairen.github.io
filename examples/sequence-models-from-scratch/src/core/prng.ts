// core/prng.ts — Seeded PRNG, the single source of all randomness in this book.
//
// WHY a hand-rolled PRNG instead of Math.random():
//   Math.random() has no seed API in JS, so runs are not reproducible. Every honest
//   number in this book (loss curves, grad norms, O(n) vs O(n^2) timings) is only
//   meaningful if "same seed + same machine => bit-identical output". A seeded
//   generator is the precondition for that claim.
//
// INVARIANT: nothing else in the book may call Math.random(). All stochastic ops
//   (param init, data generation, dropout masks, batch shuffling) thread an Rng made
//   here. The autograd/data layers take an Rng object so the random stream is explicit.
//
// FAILURE MODE: sharing one Rng instance across two logically-independent streams
//   couples them — drawing from one perturbs the other's sequence. If you need
//   independent reproducibility (e.g. data vs init), make two makeRng() with distinct
//   seeds rather than reusing one.

export interface Rng {
  /** Next uniform float in [0, 1). Advances the stream by one draw. */
  next(): number;
  /** Standard normal N(0,1) via Box-Muller. Costs 2 uniform draws. */
  normal(): number;
  /** Uniform integer in [a, b) (b exclusive). FAILS LOUD if b <= a. */
  randint(a: number, b: number): number;
  /** In-place Fisher-Yates shuffle; returns the same array for chaining. */
  shuffle<T>(arr: T[]): T[];
}

/**
 * mulberry32: 32-bit state, one multiply + a couple xorshifts per draw.
 * Good-enough statistical quality for teaching (NOT cryptographic).
 * INVARIANT: state stays an unsigned 32-bit int via `>>> 0`; drop that and the
 *   stream diverges from the reference sequence.
 */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // / 2^32 -> [0,1)
  };
}

/**
 * Construct an Rng object bound to one seed.
 * WHY return an object (not a bare function): stages need normal()/randint()/shuffle()
 *   to share ONE underlying stream so the whole run is deterministic from a single seed.
 */
export function makeRng(seed: number): Rng {
  const next = mulberry32(seed);
  return {
    next,
    normal(): number {
      // Box-Muller: two uniforms -> one N(0,1). Constant 2 draws keeps stream
      // alignment predictable across runs (no rejection loop of variable length).
      // FAILURE MODE: u1 == 0 -> log(0) = -Inf; clamp away from 0.
      let u1 = next();
      const u2 = next();
      if (u1 < 1e-12) u1 = 1e-12;
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    },
    randint(a: number, b: number): number {
      if (b <= a) throw new Error(`randint: empty range [${a}, ${b})`);
      return a + Math.floor(next() * (b - a));
    },
    shuffle<T>(arr: T[]): T[] {
      // Fisher-Yates. INVARIANT: draw j in [0, i] inclusive; drawing [0, i) biases
      //   the permutation distribution (classic off-by-one).
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(next() * (i + 1));
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
      }
      return arr;
    },
  };
}
