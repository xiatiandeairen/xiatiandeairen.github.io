// core/rng.ts — the single random source for the entire book.
//
// Why one PRNG, exported and threaded by hand everywhere: RL is the most
// replication-hostile subfield of ML. A policy-gradient curve that "works" can
// be pure seed luck; the same code with a different seed can collapse. The only
// defense a teaching repo has is *bit-for-bit reproducibility* — same seed,
// same numbers, every run, on every machine. That is impossible if any code path
// touches Math.random (it seeds from wall-clock and is engine-defined).
//
// Invariant enforced by convention (not by the type system, sadly): no file in
// this repo may call Math.random. Every stochastic decision — sampling an arm,
// sampling an action from a softmax policy, generating a preference pair, adding
// label noise — pulls from a `Rng` produced here. If you see a curve that
// changes run-to-run, the first suspect is a stray Math.random.
//
// Failure mode this guards against: "it worked on my laptop." Without a fixed
// source, two readers comparing the same stage see different KL / win-rate
// numbers and conclude the lesson is wrong, when it is only noise.

// A pulled-from-here random number generator: zero-arg, returns a float in
// [0, 1). Named type so signatures read as `rng: Rng`, not `() => number`,
// which silently invites any nullary number function.
export type Rng = () => number;

// mulberry32: a tiny, fast, well-distributed 32-bit PRNG, identical to the one
// used in the vector-search book so the whole library has ONE random source the
// reader can reason about. ~10 lines, passes basic statistical tests, period
// 2^32 (plenty for toy experiments). We force the seed to uint32 because a
// negative or float seed would corrupt the integer state arithmetic below.
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return function (): number {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box–Muller: two uniforms -> one standard-normal sample (mean 0, std 1).
// Why we need it: bandit arm payoffs and reward-model label noise are Gaussian;
// uniform noise would give the wrong tail behavior and make "regret" and
// "accuracy degrades with noise" lessons quantitatively misleading. We discard
// the second normal Box–Muller produces (caching it would make output depend on
// call parity, a subtle reproducibility hazard when callers interleave draws).
export function gaussian(rng: Rng, mean = 0, std = 1): number {
  // u1 in (0,1]: guard against log(0) = -Infinity. u1 === 0 has probability ~0
  // but over millions of draws it WILL happen; an un-guarded NaN then silently
  // poisons a reward and the bug surfaces 1000 steps later as a flat curve.
  let u1 = rng();
  if (u1 <= 0) u1 = Number.MIN_VALUE;
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

// sampleCategorical: draw an index from a discrete distribution `probs`.
// This is the beating heart of every stochastic policy in the book: the policy
// emits a probability over actions, and the agent must *commit* to one. Using a
// single rng() draw + cumulative walk keeps the consumed-randomness count equal
// to 1 per decision, which is what makes seeded runs reproducible.
//
// Invariant: probs should sum to ~1. We do NOT renormalize (that would hide a
// caller bug — a non-normalized "distribution" usually means a softmax was
// skipped). We DO clamp the fallback to the last index to defend against
// floating-point drift where the cumulative sum ends at 0.9999999 < r.
export function sampleCategorical(probs: number[], rng: Rng): number {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  // Reached only via rounding: r was just above the true total. Returning the
  // last index is correct in the limit and avoids an out-of-range -1.
  return probs.length - 1;
}

// argmax: index of the largest element. First-max wins on ties — deterministic
// but BIASED toward low indices. In RL that bias is real: greedy evaluation of a
// freshly-initialized policy will always pick action 0, masking whether the
// policy actually learned anything. Use argmaxRandomTie when ties carry meaning.
export function argmax(arr: number[]): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

// argmaxRandomTie: like argmax but breaks exact ties uniformly at random.
// Why it matters: an untrained policy has near-equal logits; deterministic
// argmax would report a fake "preference" for action 0 and make exploration look
// broken. Random tie-breaking surfaces the true uncertainty. Uses reservoir-
// style replacement so each tied index has equal probability with one rng draw
// per tie encountered (still fully reproducible under a fixed seed).
export function argmaxRandomTie(arr: number[], rng: Rng): number {
  let best = 0;
  let nTies = 1;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[best]) {
      best = i;
      nTies = 1;
    } else if (arr[i] === arr[best]) {
      nTies++;
      // Replace current winner with probability 1/nTies -> uniform over ties.
      if (rng() < 1 / nTies) best = i;
    }
  }
  return best;
}
