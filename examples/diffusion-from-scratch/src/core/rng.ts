// core/rng.ts — The single source of all randomness in this book.
//
// WHY a class wrapping a seeded PRNG instead of Math.random():
//   Diffusion IS noise. Every claim this book prints — loss curves, grad-check error,
//   sampler quality vs steps, guidance vs diversity — only means something if it is
//   bit-for-bit reproducible: same machine + same seed => same bytes. Math.random() has
//   no seed API in JS, so it cannot back a reproducible claim. mulberry32 is a tiny
//   32-bit generator with good-enough statistical quality for teaching (NOT crypto).
//
// READ THIS ONCE: any time the word "noise" appears in this book — the forward q(x_t|x_0)
//   perturbation, the reverse-step injected z, the model weight init, the toy-data
//   jitter — it comes from THIS generator with THIS seed. Reproducibility is the whole
//   point. A stage that calls Math.random() directly silently breaks every reported number.
//
// INVARIANT: nothing in stages may call Math.random(). Each stage opens with
//   `const rng = new RNG(1337)` and threads it everywhere randomness is needed.
//
// FAILURE MODE: sharing one RNG between two logically-independent streams couples them —
//   drawing from one advances the other, so reordering code changes results. When you need
//   independence (e.g. fixed eval batch vs training noise) make a fresh RNG(seed) per stream.

const TWO_PI = 2 * Math.PI;

/**
 * Seeded PRNG + samplers. Construct once per logically-independent random stream.
 * INVARIANT: state is held as an unsigned 32-bit int (`>>> 0`) so the stream is identical
 *   across platforms (no 53-bit float drift).
 */
export class RNG {
  private state: number;
  // Box–Muller produces TWO independent normals per pair of uniforms; we cache the second
  // so randn() costs ~1 uniform amortized AND the draw count stays predictable. The cache
  // is part of stream state — two runs with the same seed consume uniforms identically.
  private spareGaussian: number | null = null;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Next uniform in [0, 1). Calling it advances the stream (mutable state). */
  uniform(): number {
    // mulberry32: one add + multiply + xorshift per draw.
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // divide by 2^32 -> [0,1)
  }

  /**
   * One standard-normal sample N(0,1) via Box–Muller.
   * WHY Box–Muller: exact math, no rejection loop, so draw count per call is deterministic.
   * FAILURE MODE: u1 == 0 -> log(0) = -Inf -> NaN sample. We clamp u1 away from 0.
   * The second output of each transform is cached (spareGaussian) so we don't waste it.
   */
  gaussian(): number {
    if (this.spareGaussian !== null) {
      const g = this.spareGaussian;
      this.spareGaussian = null;
      return g;
    }
    let u1 = this.uniform();
    const u2 = this.uniform();
    if (u1 < 1e-12) u1 = 1e-12;
    const r = Math.sqrt(-2 * Math.log(u1));
    this.spareGaussian = r * Math.sin(TWO_PI * u2);
    return r * Math.cos(TWO_PI * u2);
  }

  /** Integer in [0, n). FAILURE MODE: n <= 0 has no valid value -> throw, don't return -1. */
  choice(n: number): number {
    if (n <= 0) throw new Error(`RNG.choice: n must be > 0, got ${n}`);
    return Math.floor(this.uniform() * n);
  }

  /**
   * In-place Fisher–Yates shuffle. INVARIANT: draw j in [0, i] INCLUSIVE — drawing [0, i)
   *   biases the permutation distribution (a classic off-by-one). Returns arr for chaining.
   */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.uniform() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  /** Fill a fresh Float64Array of `n` standard normals. Caller owns the buffer. */
  randn(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.gaussian();
    return out;
  }
}
