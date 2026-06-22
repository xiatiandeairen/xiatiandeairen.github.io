// core/metrics.ts — "Honest numbers" toolbox: losses, accuracy, grad-check, timing, plots.
//
// WHY this file is separate from autograd: these are the instruments that let us make
//   verifiable claims. gradCheck is the keystone — it independently verifies the analytic
//   grads in autograd.ts against numerical finite differences. Every "the gradient is
//   right" claim in this book (including the conv/pool/BN gradients) is backed by it.
//
// The ASCII plotters render to the TERMINAL because this book is offline/CPU and has no
//   browser. They show SHAPE (loss descent, a kernel's structure, a feature map's
//   activation, a receptive field's footprint) — not pixel-perfect images. Magnitudes are
//   always labelled so the reader can read values off the axis.

import { Tensor } from "./autograd.js";

/**
 * Cross-entropy loss for (batch, classes) logits and integer targets. Returns a SCALAR
 * Tensor with a correct backward, so callers just do loss.backward().
 * NUMERICAL: softmax inside (stable, subtracts row-max) then -log(prob of target). The
 * fused backward (softmax - onehot)/batch is far more stable than composing log(softmax).
 */
export function crossEntropy(logits: Tensor, targets: Int32Array | number[]): Tensor {
  if (logits.shape.length !== 2) throw new Error(`crossEntropy: expected 2-D logits, got ${logits.shape}`);
  const [batch, classes] = logits.shape;
  if (targets.length !== batch) throw new Error(`crossEntropy: targets ${targets.length} != batch ${batch}`);
  const probs = new Float64Array(batch * classes);
  let loss = 0;
  for (let b = 0; b < batch; b++) {
    const base = b * classes;
    let max = -Infinity;
    for (let j = 0; j < classes; j++) max = Math.max(max, logits.data[base + j]);
    let denom = 0;
    for (let j = 0; j < classes; j++) {
      const e = Math.exp(logits.data[base + j] - max);
      probs[base + j] = e;
      denom += e;
    }
    for (let j = 0; j < classes; j++) probs[base + j] /= denom;
    const tgt = targets[b];
    if (tgt < 0 || tgt >= classes) throw new Error(`crossEntropy: target ${tgt} out of range`);
    loss += -Math.log(probs[base + tgt] + 1e-12); // eps guards log(0)
  }
  loss /= batch;
  const out = new Tensor([loss], [1], [logits], "cross_entropy");
  out._backward = () => {
    const g = out.grad[0] / batch; // chain rule through the 1/batch mean
    for (let b = 0; b < batch; b++) {
      const base = b * classes;
      const tgt = targets[b];
      for (let j = 0; j < classes; j++) logits.grad[base + j] += g * (probs[base + j] - (j === tgt ? 1 : 0));
    }
  };
  return out;
}

/** Mean squared error between pred and target tensors of identical shape. Scalar out. */
export function mseLoss(pred: Tensor, target: Tensor): Tensor {
  if (pred.size !== target.size) throw new Error(`mseLoss: size mismatch ${pred.size} vs ${target.size}`);
  const n = pred.size;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = pred.data[i] - target.data[i];
    s += d * d;
  }
  const out = new Tensor([s / n], [1], [pred], "mse");
  out._backward = () => {
    const g = out.grad[0];
    for (let i = 0; i < n; i++) pred.grad[i] += (2 / n) * (pred.data[i] - target.data[i]) * g;
  };
  return out;
}

/** Top-1 accuracy over (batch, classes) logits vs integer targets. */
export function accuracy(logits: Tensor, targets: Int32Array | number[]): number {
  const [batch, classes] = logits.shape;
  let correct = 0;
  for (let b = 0; b < batch; b++) {
    const base = b * classes;
    let best = 0;
    let bestVal = -Infinity;
    for (let j = 0; j < classes; j++) {
      if (logits.data[base + j] > bestVal) {
        bestVal = logits.data[base + j];
        best = j;
      }
    }
    if (best === targets[b]) correct++;
  }
  return correct / batch;
}

/**
 * Confusion matrix M[true][pred] over predictions vs targets. Rows = true class, cols =
 * predicted. The diagonal is correct; off-diagonal mass tells you WHICH classes get
 * confused (e.g. cross vs square), which a single accuracy number hides.
 */
export function confusionMatrix(logits: Tensor, targets: Int32Array | number[], classes: number): number[][] {
  const M = Array.from({ length: classes }, () => new Array(classes).fill(0));
  const [batch] = logits.shape;
  for (let b = 0; b < batch; b++) {
    const base = b * classes;
    let best = 0;
    let bestVal = -Infinity;
    for (let j = 0; j < classes; j++)
      if (logits.data[base + j] > bestVal) {
        bestVal = logits.data[base + j];
        best = j;
      }
    M[targets[b]][best]++;
  }
  return M;
}

/** Render a confusion matrix with row/col labels. */
export function confusionAscii(M: number[][], names: readonly string[]): string {
  const w = Math.max(6, ...names.map((n) => n.length));
  const pad = (s: string) => s.padStart(w);
  const header = pad("t\\p") + " " + names.map((n) => pad(n)).join(" ");
  const rows = M.map((row, i) => pad(names[i]) + " " + row.map((v) => pad(String(v))).join(" "));
  return [header, ...rows].join("\n");
}

/**
 * Numerical gradient check — the correctness keystone.
 * For each sampled param element, perturb by ±eps, recompute the scalar loss via f(), form
 * the central finite-difference, compare to the analytic grad already in param.grad. Return
 * the MAX RELATIVE error.
 *
 * USAGE CONTRACT: caller must run forward+backward ONCE before calling so param.grad holds
 *   the analytic gradient; f() must return the SAME scalar loss recomputed from current
 *   param.data (a fresh forward each call — central differences need clean forwards).
 * WHY central differences: O(eps^2) error, so f64 gives < 1e-6 for a correct op; a broken
 *   op shows rel error ~O(1) — unmistakable.
 * CAVEAT: at kinks (relu / maxpool tie at exactly 0) finite differences disagree with the
 *   subgradient; keep test inputs away from exact kinks (the stages do, by using noisy
 *   random inputs).
 */
export function gradCheck(
  f: () => number,
  params: Tensor[],
  eps = 1e-5,
  samplePerParam = 8,
): { maxRelError: number; checked: number } {
  let maxRel = 0;
  let checked = 0;
  for (const p of params) {
    const stride = Math.max(1, Math.floor(p.size / samplePerParam)); // subsample big params
    for (let i = 0; i < p.size; i += stride) {
      const orig = p.data[i];
      p.data[i] = orig + eps;
      const lp = f();
      p.data[i] = orig - eps;
      const lm = f();
      p.data[i] = orig; // restore — leaving it perturbed would corrupt later checks
      const numeric = (lp - lm) / (2 * eps);
      const analytic = p.grad[i];
      const denom = Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
      const rel = Math.abs(numeric - analytic) / denom;
      if (rel > maxRel) maxRel = rel;
      checked++;
    }
  }
  return { maxRelError: maxRel, checked };
}

/**
 * Wall-clock timing. Runs fn `iters` times, returns total + per-iter ms.
 * HONESTY: REAL wall-clock (performance.now), single-threaded f64. Absolute ms vary by
 *   machine; report RELATIVE numbers (ratios), and warm up once so JIT compilation isn't
 *   counted as runtime.
 */
export function timeIt(fn: () => void, iters = 1, warmup = 0): { totalMs: number; perIterMs: number; iters: number } {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const totalMs = performance.now() - t0;
  return { totalMs, perIterMs: totalMs / iters, iters };
}

/** Count trainable parameters in a module-like object exposing parameters(). */
export function paramCount(module: { parameters: () => Tensor[] }): number {
  return module.parameters().reduce((acc, p) => acc + p.size, 0);
}

// ============================================================================
// ASCII plotting
// ============================================================================

// Grayscale ramp from empty to full. INVARIANT: index 0 = lowest intensity, last = highest.
// Using a fixed ramp (not terminal colors) keeps output copy-pasteable into a book / log.
const GRAY_RAMP = " .:-=+*#%@";

/** Map a value in [lo,hi] to a ramp char. Values outside the range clamp to the ends. */
function rampChar(v: number, lo: number, hi: number): string {
  const span = hi - lo || 1;
  const norm = Math.min(1, Math.max(0, (v - lo) / span));
  return GRAY_RAMP[Math.min(GRAY_RAMP.length - 1, Math.round(norm * (GRAY_RAMP.length - 1)))];
}

/**
 * ASCII loss curve. Downsamples history to `width` columns (mean per bucket), scales to
 * `height` rows. Shows relative shape (monotone descent / plateaus); min/max are labelled.
 */
export function lossCurveAscii(history: number[], width = 50, height = 8): string {
  if (history.length === 0) return "(no data)";
  const buckets: number[] = [];
  const per = history.length / width;
  for (let c = 0; c < width; c++) {
    const lo = Math.floor(c * per);
    const hi = Math.max(lo + 1, Math.floor((c + 1) * per));
    let s = 0;
    let n = 0;
    for (let i = lo; i < hi && i < history.length; i++) {
      s += history[i];
      n++;
    }
    buckets.push(n > 0 ? s / n : history[history.length - 1]);
  }
  const min = Math.min(...buckets);
  const max = Math.max(...buckets);
  const span = max - min || 1;
  const grid: string[][] = Array.from({ length: height }, () => Array(width).fill(" "));
  for (let c = 0; c < buckets.length; c++) {
    const norm = (buckets[c] - min) / span; // 0..1, 1 = max
    const row = Math.min(height - 1, Math.round((1 - norm) * (height - 1)));
    grid[row][c] = "*";
  }
  const lines = grid.map((r) => r.join(""));
  lines[0] = lines[0] + `  ${max.toFixed(4)} (max)`;
  lines[height - 1] = lines[height - 1] + `  ${min.toFixed(4)} (min)`;
  return lines.join("\n");
}

/**
 * Histogram of pixel (or any) values into `bins` buckets over [lo,hi], rendered as
 * horizontal bars. Used to SEE the effect of normalization / BatchNorm on an activation
 * distribution (e.g. "before BN: skewed; after BN: centered at 0").
 */
export function histogramAscii(values: ArrayLike<number>, bins = 10, lo?: number, hi?: number, barWidth = 30): string {
  let mn = lo ?? Infinity;
  let mx = hi ?? -Infinity;
  if (lo === undefined || hi === undefined)
    for (let i = 0; i < values.length; i++) {
      if (values[i] < mn) mn = values[i];
      if (values[i] > mx) mx = values[i];
    }
  const span = mx - mn || 1;
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < values.length; i++) {
    const b = Math.min(bins - 1, Math.floor(((values[i] - mn) / span) * bins));
    if (b >= 0) counts[b]++;
  }
  const peak = Math.max(...counts, 1);
  const lines = counts.map((c, i) => {
    const edge = mn + (i / bins) * span;
    const bar = "#".repeat(Math.round((c / peak) * barWidth));
    return `${edge.toFixed(2).padStart(7)} | ${bar} ${c}`;
  });
  return lines.join("\n");
}

/**
 * Heatmap of a single 2-D map (kernel or feature map) as grayscale ramp chars. Each value
 * becomes one char (doubled horizontally so cells look roughly square in a terminal).
 * Auto-scales to the map's own min/max so faint structure is still visible; the range is
 * printed so the reader knows the absolute scale.
 */
export function heatmapAscii(map: ArrayLike<number>, h: number, w: number): string {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < h * w; i++) {
    if (map[i] < lo) lo = map[i];
    if (map[i] > hi) hi = map[i];
  }
  const rows: string[] = [];
  for (let y = 0; y < h; y++) {
    let line = "";
    for (let x = 0; x < w; x++) {
      const ch = rampChar(map[y * w + x], lo, hi);
      line += ch + ch; // double width for aspect ratio
    }
    rows.push(line);
  }
  rows.push(`  range [${lo.toFixed(3)}, ${hi.toFixed(3)}]`);
  return rows.join("\n");
}

/**
 * Receptive-field visualization: given an HxW grid, mark the rectangular region [y0,y1)×
 * [x0,x1) that a chosen deep-layer unit "sees" in the input. '#' = inside the field, '.' =
 * outside. This makes the abstract "receptive field grows with depth/stride" claim of
 * stage04 concrete and checkable by eye.
 */
export function receptiveFieldAscii(H: number, W: number, y0: number, y1: number, x0: number, x1: number): string {
  const rows: string[] = [];
  for (let y = 0; y < H; y++) {
    let line = "";
    for (let x = 0; x < W; x++) {
      const inside = y >= y0 && y < y1 && x >= x0 && x < x1;
      const ch = inside ? "#" : ".";
      line += ch;
    }
    rows.push(line);
  }
  return rows.join("\n");
}
