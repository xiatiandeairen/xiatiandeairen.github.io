// core/rng.ts — Seeded PRNG + samplers, the single source of all randomness in this book.
//
// WHY a hand-rolled PRNG instead of Math.random():
//   Math.random() has no seed API in JS, so results are not reproducible across runs.
//   Every "honest number" claim in this book (Monte-Carlo estimates, sampled means,
//   shuffled splits, randomized primality witnesses) depends on bit-for-bit
//   reproducibility: same machine + same seed => identical output. That is the only
//   way a reader can rerun a stage and confirm the printed numbers are not cherry-picked.
//   mulberry32 is a tiny 32-bit generator with good-enough statistical quality for
//   teaching (NOT cryptographic — see stage08 for why we never use it for keys).
//
// INVARIANT: nothing in stages may call Math.random() directly. All stochastic ops
//   must thread an Rng created here. (Crypto stages additionally must NOT use this for
//   secrets — it is fully predictable from its 32-bit state.)
//
// FAILURE MODE: sharing ONE Rng instance across two logically-independent streams
//   couples them — advancing one perturbs the other, so a refactor that reorders calls
//   silently changes "unrelated" results. Make a fresh mulberry32(seed) per independent
//   stream, or accept the coupling on purpose and document it.

export type Rng = () => number; // returns a float in [0, 1)

/**
 * mulberry32: 32-bit state, one multiply + xorshift per draw.
 * Returns a closure holding mutable state — calling it advances the stream.
 * INVARIANT: state stays an unsigned 32-bit int via `>>> 0`; without that the
 *   arithmetic would drift into float land and lose the exact-repro guarantee.
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
 * Uniform sample in [lo, hi). Default [0,1) so callers can omit args.
 * FAILURE MODE: if hi < lo the range is negative and you get values below lo —
 *   we do not guard, callers are expected to pass lo <= hi.
 */
export function sampleUniform(rng: Rng, lo = 0, hi = 1): number {
  return lo + (hi - lo) * rng();
}

/**
 * Standard-normal sample via Box–Muller (one of two values; we discard the second).
 * WHY not cache the second value: caching makes the stream stateful in a way that is
 *   easy to get wrong under reseeding; for teaching we trade a 2x draw cost for a
 *   stateless, obviously-correct transform.
 * INVARIANT: u1 is pulled away from exactly 0 (1 - rng() in (0,1]) so log() is finite.
 */
export function sampleNormal(rng: Rng, mean = 0, std = 1): number {
  const u1 = 1 - rng(); // (0, 1], avoids log(0) = -Infinity
  const u2 = rng();
  const mag = Math.sqrt(-2 * Math.log(u1));
  return mean + std * (mag * Math.cos(2 * Math.PI * u2));
}

/**
 * Fisher–Yates shuffle returning a NEW array (input untouched).
 * WHY return a copy: stages often shuffle a dataset and still need the original order
 *   for an honest "before/after" comparison; mutating in place would destroy that.
 * INVARIANT: each permutation is equally likely given a uniform rng; the j in [0, i]
 *   range (inclusive of i) is what makes it unbiased — j in [0, i) is the classic bug.
 */
export function shuffle<T>(arr: readonly T[], rng: Rng): T[] {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1)); // inclusive upper bound -> unbiased
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
