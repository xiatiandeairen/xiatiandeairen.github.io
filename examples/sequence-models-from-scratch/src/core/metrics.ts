// core/metrics.ts — Evaluation metrics + honest timing/complexity instrumentation.
//
// WHY a dedicated metrics module: the book's central claims are quantitative ("LSTM
//   plateaus where the RNN cliffs", "Mamba scales O(n) where attention is O(n^2)").
//   Those claims need (a) consistent metric definitions and (b) HONEST measurement —
//   wall-clock that is really measured, MAC counts that match the math. Defining them
//   once here keeps every chapter's numbers comparable.
//
// HONESTY CONTRACT:
//   - timeit() returns REAL elapsed milliseconds (performance.now), not an estimate.
//   - countMACs() is a deliberately COARSE analytic count (multiply-accumulates implied
//     by the math), labeled as such; it argues complexity SHAPE, not silicon cycles.
//   - perplexity is exp(mean cross-entropy) in NATS-consistent units (we use natural log
//     throughout, so report it as such; do not silently mix log bases).

/** Fraction of predictions equal to targets. Both arrays must align 1:1. */
export function accuracy(preds: number[], targets: number[]): number {
  if (preds.length !== targets.length) throw new Error(`accuracy: length ${preds.length} != ${targets.length}`);
  if (preds.length === 0) return 0;
  let hit = 0;
  for (let i = 0; i < preds.length; i++) if (preds[i] === targets[i]) hit++;
  return hit / preds.length;
}

/**
 * Perplexity = exp(mean cross-entropy). INPUT is mean CE in NATS (natural log), matching
 * tensor.crossEntropy which uses Math.log. Lower is better; perplexity 1 = perfect, and
 * perplexity == vocabSize = uniform-random guessing (a useful sanity floor to print).
 */
export function perplexity(meanCrossEntropy: number): number {
  return Math.exp(meanCrossEntropy);
}

/**
 * timeit: run fn `reps` times, return total + per-rep milliseconds (REAL wall clock).
 * WHY a warmup rep: V8 JITs hot code; the first call pays compile cost that would skew a
 *   single-shot measurement. We discard one warmup before timing. For O(n) vs O(n^2)
 *   plots, call timeit at several n and compare the slopes — absolute ms is machine- and
 *   JIT-dependent and only meaningful relative to itself.
 */
export function timeit(fn: () => void, reps = 1): { totalMs: number; perRepMs: number } {
  fn(); // warmup (JIT), discarded
  const start = performance.now();
  for (let i = 0; i < reps; i++) fn();
  const totalMs = performance.now() - start;
  return { totalMs, perRepMs: totalMs / reps };
}

/**
 * countMACs: coarse multiply-accumulate count for common sequence ops, to CORROBORATE
 * complexity claims with arithmetic (not to predict runtime). Each helper returns the
 * dominant-term MAC count implied by the math; constants/elementwise ops are ignored.
 *
 * The point: side-by-side, dense-attention MACs grow ~n^2 while a linear SSM scan grows
 * ~n. Printing both next to the measured wall-clock makes the "shape matches the math"
 * argument concrete and honest about being an approximation.
 */
export const countMACs = {
  /** y = x[B,in] @ W[in,out] : B*in*out MACs. */
  linear(batch: number, inDim: number, outDim: number): number {
    return batch * inDim * outDim;
  },
  /** Dense self-attention scores QK^T over seqLen n, model dim d: n*n*d (+ same for AV). */
  attention(seqLen: number, dim: number): number {
    return 2 * seqLen * seqLen * dim; // QK^T then (softmax)·V, both ~n^2*d
  },
  /** Recurrent/SSM scan: per-step O(state*dim) work over n steps: n*state*dim. */
  scan(seqLen: number, stateDim: number, dim: number): number {
    return seqLen * stateDim * dim;
  },
};

/**
 * argmax over a flat logits row. Returned index is the predicted class. Ties resolve to
 * the FIRST max (deterministic — important for reproducible accuracy numbers).
 */
export function argmax(row: ArrayLike<number>): number {
  let best = 0;
  let bestVal = row[0];
  for (let i = 1; i < row.length; i++) {
    if (row[i] > bestVal) {
      bestVal = row[i];
      best = i;
    }
  }
  return best;
}

/**
 * Mean squared error over flat arrays (for the regression adding-problem). Reported
 * alongside a baseline (variance of the target) so the chapter can state how much of the
 * target variance the model actually explains, not just a bare MSE.
 */
export function mse(preds: number[], targets: number[]): number {
  if (preds.length !== targets.length) throw new Error("mse: length mismatch");
  if (preds.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < preds.length; i++) {
    const d = preds[i] - targets[i];
    s += d * d;
  }
  return s / preds.length;
}
