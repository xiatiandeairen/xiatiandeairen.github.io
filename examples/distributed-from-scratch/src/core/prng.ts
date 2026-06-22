// core/prng.ts — the book's single source of randomness.
//
// Why a hand-rolled, seeded PRNG instead of Math.random: every number this book
// prints (message counts, election rounds, convergence steps, partition
// outcomes) must be byte-identical across runs and machines. A distributed bug
// is only useful to a reader if it REPLAYS — "the leader flapped once on my
// machine" is the survey-grade hand-waving this book rejects. Math.random is
// unseedable, so it is banned everywhere; all nondeterminism (latency jitter,
// drop coin-flips, election timeouts, message reordering) is derived from here.
//
// Algorithm: mulberry32. 32-bit state, one multiply + xorshift per step. Not
// cryptographic (irrelevant) but well-distributed for generating workloads and
// breaking interleaving ties. `seededRng(42)` produces the same stream forever.
//
// Invariant: state stays in uint32 space (`>>> 0` everywhere). Drift into
// float / signed territory silently forks the stream — the failure mode is
// "numbers stop matching the book and nobody can tell why", so we are paranoid
// about the masks.

export interface Rng {
  /** Raw 32-bit unsigned step. Every other generator is derived from this. */
  nextU32(): number;
  /** Uniform float in [0, 1), built from one u32 step. Core primitive for
   *  probabilities (drop rate, partition coin-flips). */
  next(): number;
  /** Uniform integer in [0, n). Half-open so it composes like array indices.
   *  Used for picking nodes, jittered timeouts, message indices. */
  int(n: number): number;
  /** Bernoulli trial: true with probability p. The honest way to express
   *  "this message drops 10% of the time" — drives Network.dropRate. */
  bool(p: number): boolean;
  /** In-place Fisher-Yates shuffle. Mutates AND returns the same array. Used to
   *  randomize message delivery order under the `reorder` fault. */
  shuffle<T>(arr: T[]): T[];
}

export function seededRng(seed: number): Rng {
  // Force the seed into uint32 up front; a float / negative seed would otherwise
  // produce a different first step than an integer with the same bit pattern.
  let state = seed >>> 0;

  function nextU32(): number {
    // mulberry32 core. The constants are the published magic numbers; changing
    // them is not "tuning" — it forks the entire book's reproducibility.
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  }

  function next(): number {
    // Divide by 2^32 → [0, 1). Using the full u32 keeps it uniform; a
    // modulo-based float would bias the low bits.
    return nextU32() / 0x100000000;
  }

  function int(n: number): number {
    // Half-open [0, n). An empty range is a caller bug, not a runtime case to
    // paper over — returning 0 silently would hide it.
    if (n <= 0) throw new Error(`int: n must be positive, got ${n}`);
    return Math.floor(next() * n);
  }

  function bool(p: number): boolean {
    // Strict `<` so p=0 never fires and p=1 always fires — boundary correctness
    // matters because dropRate(0) MUST mean a perfectly reliable link.
    return next() < p;
  }

  function shuffle<T>(arr: T[]): T[] {
    // Fisher-Yates from the top. j in [0, i] inclusive is the correct version;
    // the common off-by-one (j in [0, i)) skews the permutation distribution.
    for (let i = arr.length - 1; i > 0; i--) {
      const j = int(i + 1);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  return { nextU32, next, int, bool, shuffle };
}
