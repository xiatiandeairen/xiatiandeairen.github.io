// core/data.ts — Toy, fully-reproducible image datasets. Offline, no downloads, no network.
//
// WHY synthetic images: this book teaches the MACHINE (convolution, pooling, BN, residuals),
//   not SOTA accuracy. Procedurally-drawn shapes let every example converge on a CPU in
//   seconds and stay bit-for-bit reproducible (all randomness threads a seeded Rng).
//
// HONESTY BOUNDARY (stated once, applies to every stage that trains):
//   Absolute accuracies here are OPTIMISTIC — the shape classes (circle/square/cross/
//   triangle) are clean, centered-ish, low-noise, and far more separable than real photos.
//   A 2-layer CNN can hit very high accuracy because the task is easy BY CONSTRUCTION.
//   What transfers is the TREND: train loss decreasing, grad-check error < 1e-6, BN
//   stabilizing training, residual converging faster than plain, augmentation shrinking the
//   train/test gap. Do NOT quote these accuracies as if they generalize to natural images.
//
// LAYOUT: every image is a single-channel grayscale Float64Array of length H*W, values in
//   [0,1]. Batches are NCHW with C=1. The dataset stores flat per-image buffers; the batch
//   iterator stacks them into one NCHW Tensor-ready Float64Array.

import { randn, randint, uniform, shuffle, type Rng } from "./rng.js";

export type ShapeLabel = 0 | 1 | 2 | 3; // 0=circle 1=square 2=cross 3=triangle
export const SHAPE_NAMES = ["circle", "square", "cross", "triangle"] as const;

export interface ImageSample {
  pixels: Float64Array; // length H*W, row-major, values in [0,1]
  label: number;
  H: number;
  W: number;
}

export interface ImageDataset {
  samples: ImageSample[];
  classes: number;
  H: number;
  W: number;
}

/** Set one pixel, clamping the additive ink to [0,1]. Out-of-bounds writes are ignored
 *  (shapes near the border just get clipped) — a no-op rather than an error keeps the
 *  drawing routines simple and robust to jittered positions. */
function ink(img: Float64Array, H: number, W: number, x: number, y: number, value: number): void {
  if (x < 0 || x >= W || y < 0 || y >= H) return;
  const idx = y * W + x;
  img[idx] = Math.min(1, img[idx] + value);
}

/** Draw a filled-ish ring (circle outline). thickness controls stroke width. */
function drawCircle(img: Float64Array, H: number, W: number, cx: number, cy: number, r: number, thickness: number): void {
  // Rasterize by testing each pixel's distance to the radius — clearer than Bresenham and
  // we don't care about speed for tiny images. A band |dist - r| < thickness is the stroke.
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const d = Math.hypot(x - cx, y - cy);
      if (Math.abs(d - r) <= thickness) ink(img, H, W, x, y, 1);
    }
}

function drawSquare(img: Float64Array, H: number, W: number, cx: number, cy: number, half: number): void {
  for (let t = -half; t <= half; t++) {
    ink(img, H, W, cx - half, cy + t, 1); // left edge
    ink(img, H, W, cx + half, cy + t, 1); // right edge
    ink(img, H, W, cx + t, cy - half, 1); // top edge
    ink(img, H, W, cx + t, cy + half, 1); // bottom edge
  }
}

function drawCross(img: Float64Array, H: number, W: number, cx: number, cy: number, half: number): void {
  for (let t = -half; t <= half; t++) {
    ink(img, H, W, cx + t, cy, 1); // horizontal bar
    ink(img, H, W, cx, cy + t, 1); // vertical bar
  }
}

function drawTriangle(img: Float64Array, H: number, W: number, cx: number, cy: number, half: number): void {
  // Isoceles triangle: apex at top, base at bottom. Draw the two slanted sides + base.
  for (let t = 0; t <= 2 * half; t++) {
    const y = cy - half + t;
    const spread = Math.round((t / (2 * half)) * half); // widen toward the base
    ink(img, H, W, cx - spread, y, 1);
    ink(img, H, W, cx + spread, y, 1);
  }
  for (let x = -half; x <= half; x++) ink(img, H, W, cx + x, cy + half, 1); // base
}

/** Add i.i.d. Gaussian noise then clamp to [0,1]. noise=0 leaves the image untouched. */
function addNoise(img: Float64Array, rng: Rng, noise: number): void {
  if (noise <= 0) return;
  for (let i = 0; i < img.length; i++) img[i] = Math.min(1, Math.max(0, img[i] + randn(rng) * noise));
}

/**
 * Generate one labelled shape image. Position and size are jittered (deterministically via
 * rng) so the classifier can't cheat on a fixed pixel — it must learn the shape, which is
 * exactly the translation-(in)variance story convolution is about.
 */
export function makeShapeImage(label: ShapeLabel, H: number, W: number, rng: Rng, noise = 0): ImageSample {
  const img = new Float64Array(H * W);
  const margin = Math.floor(Math.min(H, W) * 0.28); // keep shapes mostly in-frame
  const cx = randint(rng, margin, W - 1 - margin);
  const cy = randint(rng, margin, H - 1 - margin);
  const size = randint(rng, Math.floor(margin * 0.6), margin); // half-extent / radius
  switch (label) {
    case 0:
      drawCircle(img, H, W, cx, cy, size, 1.2);
      break;
    case 1:
      drawSquare(img, H, W, cx, cy, size);
      break;
    case 2:
      drawCross(img, H, W, cx, cy, size);
      break;
    case 3:
      drawTriangle(img, H, W, cx, cy, size);
      break;
  }
  addNoise(img, rng, noise);
  return { pixels: img, label, H, W };
}

/**
 * Build a balanced dataset of geometric shapes. perClass images of each of the 4 classes.
 * The class loop order is fixed; shuffling (deterministic via rng) follows so batches mix
 * classes. WHY balanced: keeps accuracy interpretable (chance = 1/classes) without
 * needing class-weighting machinery the book doesn't teach.
 */
export function makeShapeDataset(perClass: number, H: number, W: number, rng: Rng, noise = 0.05): ImageDataset {
  const samples: ImageSample[] = [];
  for (let c = 0; c < 4; c++)
    for (let i = 0; i < perClass; i++) samples.push(makeShapeImage(c as ShapeLabel, H, W, rng, noise));
  shuffle(samples, rng);
  return { samples, classes: 4, H, W };
}

/**
 * MNIST-like single-stroke digit-ish glyphs. NOT real MNIST — a tiny set of 3 stroke
 * patterns (vertical "1", horizontal-top "7"-ish, diagonal "/"). Purpose: a second toy
 * task with thinner, more translation-sensitive structure to contrast with solid shapes.
 * Kept deliberately minimal; absolute accuracy here is even more optimistic than shapes.
 */
export function makeStrokeDataset(perClass: number, H: number, W: number, rng: Rng, noise = 0.05): ImageDataset {
  const classes = 3;
  const samples: ImageSample[] = [];
  for (let c = 0; c < classes; c++)
    for (let i = 0; i < perClass; i++) {
      const img = new Float64Array(H * W);
      const margin = Math.floor(Math.min(H, W) * 0.25);
      const cx = randint(rng, margin, W - 1 - margin);
      const len = randint(rng, Math.floor(H * 0.4), Math.floor(H * 0.7));
      const top = randint(rng, margin, H - 1 - len);
      if (c === 0) {
        for (let t = 0; t < len; t++) ink(img, H, W, cx, top + t, 1); // vertical bar
      } else if (c === 1) {
        for (let t = 0; t < len; t++) ink(img, H, W, cx - Math.floor(len / 2) + t, top, 1); // horizontal bar
      } else {
        for (let t = 0; t < len; t++) ink(img, H, W, cx - Math.floor(len / 2) + t, top + t, 1); // diagonal
      }
      addNoise(img, rng, noise);
      samples.push({ pixels: img, label: c, H, W });
    }
  shuffle(samples, rng);
  return { samples, classes, H, W };
}

/**
 * Split into train/test by ratio (test fraction). Shuffling already happened at generation;
 * we slice deterministically. WHY a held-out test set at all: the augmentation stage needs
 * to show the train/test GAP — without a disjoint test set "accuracy" is meaningless.
 */
export function trainTestSplit(ds: ImageDataset, testRatio: number): { train: ImageDataset; test: ImageDataset } {
  const nTest = Math.floor(ds.samples.length * testRatio);
  const test = ds.samples.slice(0, nTest);
  const train = ds.samples.slice(nTest);
  return {
    train: { samples: train, classes: ds.classes, H: ds.H, W: ds.W },
    test: { samples: test, classes: ds.classes, H: ds.H, W: ds.W },
  };
}

export interface ImageBatch {
  /** flat NCHW data, ready for `new Tensor(data, [N,1,H,W])`. */
  data: Float64Array;
  labels: Int32Array;
  N: number;
  C: number;
  H: number;
  W: number;
}

/**
 * Stack a slice of samples into one NCHW batch (C=1). Pure given its inputs — the caller
 * decides which indices form the batch, so this stays deterministic and easy to test.
 */
export function stackBatch(samples: ImageSample[], H: number, W: number): ImageBatch {
  const N = samples.length;
  const data = new Float64Array(N * H * W); // C=1
  const labels = new Int32Array(N);
  for (let n = 0; n < N; n++) {
    data.set(samples[n].pixels, n * H * W);
    labels[n] = samples[n].label;
  }
  return { data, labels, N, C: 1, H, W };
}

/**
 * Yield mini-batches over a dataset, optionally reshuffling each epoch. The last batch may
 * be smaller than batchSize (we do NOT drop it — on tiny datasets dropping the tail wastes
 * scarce data). Shuffling threads the rng so epoch order is reproducible.
 */
export function* iterBatches(ds: ImageDataset, batchSize: number, rng: Rng, shuffleEach = true): Generator<ImageBatch> {
  const order = Array.from({ length: ds.samples.length }, (_, i) => i);
  if (shuffleEach) shuffle(order, rng);
  for (let start = 0; start < order.length; start += batchSize) {
    const idx = order.slice(start, start + batchSize);
    yield stackBatch(idx.map((i) => ds.samples[i]), ds.H, ds.W);
  }
}

// ----------------------------------------------------------------------------
// Augmentation (stage08). Pure transforms over a single ImageSample.
// ----------------------------------------------------------------------------

/** Shift the image by (dx, dy), filling vacated pixels with 0. Translation invariance is
 *  the property convolution is supposed to provide; augmenting with shifts tests/improves
 *  it. Out-of-frame content is dropped (not wrapped) — wrapping would teach a false torus
 *  topology the conv never sees at test time. */
export function shiftImage(s: ImageSample, dx: number, dy: number): ImageSample {
  const { H, W } = s;
  const out = new Float64Array(H * W);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const sx = x - dx;
      const sy = y - dy;
      if (sx >= 0 && sx < W && sy >= 0 && sy < H) out[y * W + x] = s.pixels[sy * W + sx];
    }
  return { pixels: out, label: s.label, H, W };
}

/** Horizontal flip. NOTE: only label-preserving for symmetric classes; the book uses it on
 *  shapes (all mirror-symmetric except triangle's apex, which stays a triangle) — a
 *  deliberate teaching point about when flips are safe vs label-changing. */
export function flipImageH(s: ImageSample): ImageSample {
  const { H, W } = s;
  const out = new Float64Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = s.pixels[y * W + (W - 1 - x)];
  return { pixels: out, label: s.label, H, W };
}

/** Random small shift in [-maxShift, maxShift] on each axis (deterministic via rng). The
 *  cheap, always-safe augmentation; used by stage08 to demonstrate the generalization gain. */
export function randomShift(s: ImageSample, maxShift: number, rng: Rng): ImageSample {
  const dx = Math.round(uniform(rng, -maxShift, maxShift));
  const dy = Math.round(uniform(rng, -maxShift, maxShift));
  return shiftImage(s, dx, dy);
}
