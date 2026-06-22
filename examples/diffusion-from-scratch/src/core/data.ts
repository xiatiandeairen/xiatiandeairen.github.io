// core/data.ts — Toy 2-D data generators (the "images" of this book).
//
// WHY 2-D toys instead of pictures: a diffusion model's job is to learn a data distribution
//   and sample from it. On 2-D point clouds you can SEE the whole distribution at once (an
//   ASCII scatter, plot.ts) and judge sample quality by eye — does the model put points
//   where the data is? Real images need a U-Net + millions of params to even start; the
//   DDPM mechanism (forward noising, score learning, reverse sampling) is identical and is
//   what this book teaches. The honesty note: absolute loss/quality numbers here are
//   optimistic vs real image diffusion; the transferable signal is the mechanism and the
//   shape of the trade-off curves.
//
// SHARED CONTRACT (every generator):
//   - takes an RNG so the dataset is byte-reproducible
//   - returns a Tensor of shape [n, 2]
//   - is NORMALIZED to zero mean and unit variance per coordinate (see normalize())
//
// WHY normalize: the forward process assumes data lives near the unit scale (√ᾱ·x_0 with
//   ᾱ→0 must land in N(0,I) territory). Un-normalized data (e.g. a swiss roll spanning
//   [-15, 15]) would need a wildly different β schedule; normalizing lets ONE schedule work
//   for every dataset. FAILURE MODE if you skip it: x_T is not ~N(0,I), so sampling from
//   N(0,I) starts off-distribution and the model never recovers.

import { RNG } from "./rng.js";
import { Tensor } from "./tensor.js";

/** Normalize a flat [n,2] buffer to per-coordinate zero mean, unit variance, in place,
 *  then wrap as a Tensor. Uses population variance (÷n); the +1e-8 guards a degenerate
 *  constant coordinate from producing a divide-by-zero. */
function normalize(buf: Float64Array, n: number): Tensor {
  for (let c = 0; c < 2; c++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += buf[i * 2 + c];
    mean /= n;
    let varc = 0;
    for (let i = 0; i < n; i++) {
      const d = buf[i * 2 + c] - mean;
      varc += d * d;
    }
    varc /= n;
    const std = Math.sqrt(varc) + 1e-8;
    for (let i = 0; i < n; i++) buf[i * 2 + c] = (buf[i * 2 + c] - mean) / std;
  }
  return new Tensor(buf, [n, 2]);
}

/**
 * Two interleaving half-circles ("two moons"). The classic non-linearly-separable toy: two
 * crescent clusters that a model must learn as two distinct curved modes. `noise` is the
 * std of Gaussian jitter added to each point before normalization.
 */
export function twoMoons(n: number, noise: number, rng: RNG): Tensor {
  const buf = new Float64Array(n * 2);
  const nPerMoon = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const upper = i < nPerMoon;
    // Spread the angle deterministically across the arc, jitter the radius via the RNG.
    const idx = upper ? i : i - nPerMoon;
    const count = upper ? nPerMoon : n - nPerMoon;
    const theta = Math.PI * (idx / Math.max(count - 1, 1));
    let x: number;
    let y: number;
    if (upper) {
      x = Math.cos(theta);
      y = Math.sin(theta);
    } else {
      // Second moon: flipped and offset so the two crescents interlock.
      x = 1 - Math.cos(theta);
      y = 0.5 - Math.sin(theta);
    }
    buf[i * 2] = x + rng.gaussian() * noise;
    buf[i * 2 + 1] = y + rng.gaussian() * noise;
  }
  return normalize(buf, n);
}

/**
 * Mixture of isotropic Gaussians at the given centers (a multi-modal distribution). This is
 * the cleanest test of mode coverage: the model should place mass at EVERY center, not just
 * the easiest one. Mode collapse (a known generative failure) shows up here as missing blobs.
 * Each point: pick a center uniformly, add N(0, 0.15²) jitter.
 */
export function mixtureOfGaussians(centers: Array<[number, number]>, n: number, rng: RNG): Tensor {
  if (centers.length === 0) throw new Error("mixtureOfGaussians: need >= 1 center");
  const buf = new Float64Array(n * 2);
  const std = 0.15;
  for (let i = 0; i < n; i++) {
    const c = centers[rng.choice(centers.length)];
    buf[i * 2] = c[0] + rng.gaussian() * std;
    buf[i * 2 + 1] = c[1] + rng.gaussian() * std;
  }
  return normalize(buf, n);
}

/**
 * 2-D Swiss roll: a spiral where radius grows with angle. A single connected 1-D manifold
 * embedded in 2-D — tests whether the model captures a thin curved structure rather than
 * smearing probability across the plane.
 */
export function swissRoll2D(n: number, rng: RNG): Tensor {
  const buf = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    // t in [1.5π, 4.5π]: three turns of the spiral. Slight RNG jitter off the curve.
    const t = 1.5 * Math.PI * (1 + 2 * (i / Math.max(n - 1, 1)));
    buf[i * 2] = t * Math.cos(t) + rng.gaussian() * 0.5;
    buf[i * 2 + 1] = t * Math.sin(t) + rng.gaussian() * 0.5;
  }
  return normalize(buf, n);
}

/**
 * Two-armed spiral (the galaxy toy). Two interleaved spiral arms — like swiss roll but with
 * two modes that wind around each other, a harder coverage test.
 */
export function spiral(n: number, rng: RNG): Tensor {
  const buf = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    const arm = i % 2; // alternate arms so both are equally represented
    const frac = i / Math.max(n - 1, 1);
    const r = frac; // radius grows from center outward
    const theta = frac * 3 * Math.PI + arm * Math.PI; // second arm offset by π
    buf[i * 2] = r * Math.cos(theta) + rng.gaussian() * 0.03;
    buf[i * 2 + 1] = r * Math.sin(theta) + rng.gaussian() * 0.03;
  }
  return normalize(buf, n);
}
