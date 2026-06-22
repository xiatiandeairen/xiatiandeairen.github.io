// core/prng.ts — the book's single source of randomness.
//
// Why a hand-rolled PRNG instead of Math.random: every number this book prints
// (IO counts, throughput, recovery outcomes) must be byte-identical across runs
// and machines, otherwise readers can't reproduce — and "trust me it works" is
// exactly the survey-grade hand-waving this book rejects. Math.random is
// unseedable, so it is banned everywhere in the codebase.
//
// Algorithm: mulberry32. A 32-bit state, single multiply + xorshift per step.
// Not cryptographically secure (irrelevant here) but has good distribution for
// generating workloads and tie-breaking interleavings. The whole point is that
// `createRng(42)` produces the same stream forever.
//
// Invariant: state is kept as a uint32 (`>>> 0` everywhere). Drift into float /
// signed territory silently changes the stream and breaks reproducibility — the
// failure mode is "numbers stop matching the book and nobody knows why", so we
// are paranoid about the `>>> 0` masks.

export interface Rng {
  /** Raw 32-bit unsigned step. All other generators are derived from this. */
  nextU32(): number;
  /** Uniform float in [0, 1). 53-bit-ish; built from one u32 for determinism. */
  nextFloat(): number;
  /** Uniform integer in [lo, hi). Half-open so it composes like array indices. */
  nextInt(lo: number, hi: number): number;
  /** In-place Fisher-Yates shuffle. Mutates and returns the same array. */
  shuffle<T>(arr: T[]): T[];
}

export function createRng(seed: number): Rng {
  // Force the seed into uint32 space up front; a float or negative seed would
  // otherwise produce a different first step than an integer with the same bits.
  let state = seed >>> 0;

  function nextU32(): number {
    // mulberry32 core. The constants are the published magic numbers; changing
    // them is not "tuning", it is forking the entire book's reproducibility.
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0);
  }

  function nextFloat(): number {
    // Divide by 2^32 so the result is in [0, 1). Using the full u32 keeps the
    // distribution uniform; a modulo-based float would bias low bits.
    return nextU32() / 0x100000000;
  }

  function nextInt(lo: number, hi: number): number {
    // Half-open [lo, hi). We assume hi > lo; an empty range is a caller bug, not
    // a runtime case to paper over — returning lo silently would hide it.
    if (hi <= lo) throw new Error(`nextInt: empty range [${lo}, ${hi})`);
    return lo + Math.floor(nextFloat() * (hi - lo));
  }

  function shuffle<T>(arr: T[]): T[] {
    // Fisher-Yates from the top. Drawing j in [0, i] (inclusive) is the correct
    // version; the common off-by-one (j in [0, i)) skews the permutation.
    for (let i = arr.length - 1; i > 0; i--) {
      const j = nextInt(0, i + 1);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  return { nextU32, nextFloat, nextInt, shuffle };
}
