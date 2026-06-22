// core/prng.ts — Seeded PRNG, the single source of all randomness in this book.
//
// WHY a hand-rolled PRNG instead of Math.random():
//   Math.random() has no seed API, so runs are not reproducible. Every "honest number"
//   in this book (loss curves, expert-utilization CV, routing entropy, sparse-vs-dense
//   FLOP ratios) is a claim that only holds if the run is bit-for-bit reproducible:
//   same machine + same seed => same output. mulberry32 is a tiny 32-bit generator with
//   good-enough statistical quality for teaching (NOT cryptographic).
//
// INVARIANT: nothing anywhere may call Math.random() directly. Init, shuffling, batch
//   sampling, tie-breaking in routing — all must thread an Rng created here. A grep for
//   Math.random in src/ should return zero hits.
//
// FAILURE MODE: sharing ONE Rng across two logically-independent streams couples them —
//   drawing from one perturbs the other, so a change in (say) data sampling silently
//   shifts model init. Create a fresh rng(seed) per independent stream, or accept the
//   coupling on purpose. Stages document which seed feeds which stream.

export interface Rng {
  /** Uniform float in [0, 1). Advances the stream by one draw. */
  next(): number;
  /** Standard normal N(0,1) via Box–Muller. Costs TWO uniform draws per call. */
  normal(): number;
  /** Uniform integer in [0, n). Used for index sampling / picking. */
  int(n: number): number;
  /** In-place Fisher–Yates shuffle; returns the same array for chaining. */
  shuffle<T>(arr: T[]): T[];
  /** Uniform random element of arr (does not mutate). */
  pick<T>(arr: T[]): T;
}

/**
 * Build a seeded Rng. The closure `state` is the entire generator state (one u32).
 * INVARIANT: state is kept as an unsigned 32-bit int via `>>> 0`; signed overflow would
 *   corrupt the sequence and break reproducibility across V8 versions.
 */
export function rng(seed: number): Rng {
  let state = seed >>> 0;

  function next(): number {
    // mulberry32: one add + one imul + xorshifts per draw.
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // / 2^32 -> [0,1)
  }

  function normal(): number {
    // Box–Muller: two uniforms -> one N(0,1). Constant 2 draws keeps stream alignment
    // predictable across runs (we discard the sin() partner for simplicity).
    // FAILURE MODE: u1 == 0 -> log(0) = -Inf. Clamp u1 off zero.
    let u1 = next();
    const u2 = next();
    if (u1 < 1e-12) u1 = 1e-12;
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function int(n: number): number {
    // floor(uniform * n) is uniform over [0,n) given a uniform in [0,1).
    if (n <= 0) throw new Error(`rng.int(n): n must be > 0, got ${n}`);
    return Math.floor(next() * n);
  }

  function shuffle<T>(arr: T[]): T[] {
    // INVARIANT: draw j in [0, i] INCLUSIVE; drawing [0, i) biases the permutation.
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(next() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  function pick<T>(arr: T[]): T {
    if (arr.length === 0) throw new Error("rng.pick: empty array");
    return arr[int(arr.length)];
  }

  return { next, normal, int, shuffle, pick };
}

/**
 * Kaiming/He std for ReLU-family fan-in: sqrt(2/fanIn).
 * WHY 2/fanIn: ReLU zeroes ~half its inputs, halving variance; the 2 restores
 *   forward-activation variance so deep stacks neither explode nor vanish.
 * Exposed as a helper so nn.ts and stages share ONE init convention.
 */
export function kaimingStd(fanIn: number): number {
  return Math.sqrt(2 / fanIn);
}
