// core/prng.ts — The single, global source of randomness for the whole PEFT book.
//
// WHY a hand-rolled, GLOBAL, stateful PRNG (not Math.random, not a threaded Rng object):
//   Every "honest number" in this book — loss curves, grad-check error, the claim that
//   "LoRA reaches the same loss with 0.5% of the params" — only means something if it is
//   bit-for-bit reproducible. Math.random() has no seed in JS, so it is disqualified.
//   We expose a *global* generator with seed()/uniform()/normal()/randint() because the
//   book's pedagogy is "put seed(1234) at the top of a stage, get the same output to the
//   last decimal". A single global stream makes that one-liner enough; threading an Rng
//   instance through every Linear/Embedding/dropout call would bury the lesson in plumbing.
//
// INVARIANT: nothing in core or stages may call Math.random() directly. All stochastic
//   ops (weight init, batch shuffling, dropout masks, prefix init) draw from here.
//
// FAILURE MODE this guards against: forgetting to re-seed between two experiments in the
//   same process. Because the generator is global and stateful, experiment B inherits
//   experiment A's stream position and prints different numbers than when run alone.
//   => Each stage must call seed() exactly once at the top, before any draw.

// mulberry32 state. Kept as an unsigned 32-bit int (the `>>> 0` everywhere enforces it).
// 32 bits of state is plenty for teaching; this is NOT cryptographically secure.
let _state = 1234 >>> 0;

/**
 * Re-seed the global stream. Call once at the top of every stage.
 * INVARIANT: same seed => identical sequence of all subsequent draws.
 */
export function seed(n: number): void {
  _state = n >>> 0;
}

/**
 * Core mulberry32 step: one add + xorshift-multiply chain per draw.
 * Returns a float in [0, 1). Advancing the stream mutates module state.
 */
function nextU32Float(): number {
  _state = (_state + 0x6d2b79f5) >>> 0;
  let t = _state;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // / 2^32 -> [0,1)
}

/** Uniform float in [0, 1). The primitive every other sampler is built from. */
export function uniform(): number {
  return nextU32Float();
}

/** Uniform float in [lo, hi). */
export function uniformRange(lo: number, hi: number): number {
  return lo + (hi - lo) * nextU32Float();
}

/**
 * Standard-normal-derived sample via Box–Muller, scaled to N(mean, std).
 * WHY Box–Muller: exact transform of two uniforms into a Gaussian with a CONSTANT
 *   number of draws (=2). Constant draw count keeps stream alignment predictable —
 *   a rejection sampler would consume a data-dependent number of draws and break
 *   "same seed => same sequence downstream".
 * FAILURE MODE: u1 == 0 -> log(0) = -Inf. We clamp u1 away from zero.
 *   (We discard the sine output for simplicity; cost is one extra draw per call.)
 */
export function normal(mean = 0, std = 1): number {
  let u1 = nextU32Float();
  const u2 = nextU32Float();
  if (u1 < 1e-12) u1 = 1e-12;
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

/**
 * Integer in [lo, hi) (hi exclusive).
 * INVARIANT: uniform over the half-open range; using Math.round instead of floor would
 *   make the endpoints half as likely as interior values (classic off-by-one bias).
 */
export function randint(lo: number, hi: number): number {
  return lo + Math.floor(nextU32Float() * (hi - lo));
}

/**
 * In-place Fisher–Yates shuffle using the global stream. Returns the array for chaining.
 * INVARIANT: draw j in [0, i] inclusive; drawing [0, i) instead biases the permutation.
 */
export function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(nextU32Float() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

/**
 * Kaiming/He std for ReLU/GELU-family layers: sqrt(2 / fanIn).
 * WHY: GELU/ReLU roughly halve forward variance; the factor 2 restores it so a stack of
 *   toy Transformer blocks neither saturates to zero nor explodes at init.
 */
export function kaimingStd(fanIn: number): number {
  return Math.sqrt(2 / fanIn);
}

/** Xavier/Glorot std for linear/attention projections: sqrt(2 / (fanIn + fanOut)). */
export function xavierStd(fanIn: number, fanOut: number): number {
  return Math.sqrt(2 / (fanIn + fanOut));
}
