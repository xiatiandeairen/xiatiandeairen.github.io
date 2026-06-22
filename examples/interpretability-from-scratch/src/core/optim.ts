// core/optim.ts — Optimizers + grad clipping + LR schedule + cross-entropy loss.
//
// These exist only to TRAIN the research object (the TinyTransformer) into a state worth
// dissecting. The interpretability happens after training; here we just need convergence
// to be reproducible and honest.
//
// CONTRACT every optimizer obeys:
//   - holds a fixed param list (taken once; order frozen so per-param state stays aligned).
//   - step(): read each param's .grad (filled by backward), mutate .data in place.
//   - zeroGrad(): clear all grads. You MUST call this each iteration — autograd ACCUMULATES
//     grads on purpose, so a missing zeroGrad sums grads across steps -> "loss explodes
//     after a few iterations", the classic bug.
//
// HONESTY: textbook update rules, no fused kernels. On toy data the curves are real but
//   absolute steps-to-converge are optimistic; the transferable signal is RELATIVE (AdamW
//   beats SGD, warmup stabilizes early steps), not the step counts.

import { Tensor } from "./autograd.js";

export interface Optimizer {
  step(): void;
  zeroGrad(): void;
  setLr(lr: number): void;
}

export class SGD implements Optimizer {
  private params: Tensor[];
  private lr: number;
  private momentum: number;
  private velocity: Float64Array[];
  constructor(params: Tensor[], opts: { lr: number; momentum?: number }) {
    this.params = params;
    this.lr = opts.lr;
    this.momentum = opts.momentum ?? 0;
    this.velocity = params.map((p) => new Float64Array(p.size));
  }
  step(): void {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const v = this.velocity[i];
      for (let j = 0; j < p.size; j++) {
        let g = p.grad[j];
        if (this.momentum !== 0) {
          v[j] = this.momentum * v[j] + g;
          g = v[j];
        }
        p.data[j] -= this.lr * g;
      }
    }
  }
  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
  setLr(lr: number): void {
    this.lr = lr;
  }
}

/**
 * AdamW (decoupled weight decay).
 * Bias correction (divide m,v by 1-beta^t) un-biases the zero-initialized moments; skipping
 *   it makes the first ~1/(1-beta) steps take tiny effective steps — an observable slow
 *   start. DECOUPLED decay applies directly to the param, NOT folded into the gradient, so
 *   the adaptive denominator doesn't scale the decay (the AdamW correction over Adam that
 *   matters for transformers).
 */
export class AdamW implements Optimizer {
  private params: Tensor[];
  private lr: number;
  private beta1: number;
  private beta2: number;
  private eps: number;
  private wd: number;
  private m: Float64Array[];
  private v: Float64Array[];
  private t = 0;
  constructor(
    params: Tensor[],
    opts: { lr: number; beta1?: number; beta2?: number; eps?: number; weightDecay?: number },
  ) {
    this.params = params;
    this.lr = opts.lr;
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.999;
    this.eps = opts.eps ?? 1e-8;
    this.wd = opts.weightDecay ?? 0;
    this.m = params.map((p) => new Float64Array(p.size));
    this.v = params.map((p) => new Float64Array(p.size));
  }
  step(): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const m = this.m[i];
      const v = this.v[i];
      for (let j = 0; j < p.size; j++) {
        const g = p.grad[j];
        m[j] = this.beta1 * m[j] + (1 - this.beta1) * g;
        v[j] = this.beta2 * v[j] + (1 - this.beta2) * g * g;
        const mhat = m[j] / bc1;
        const vhat = v[j] / bc2;
        let update = (this.lr * mhat) / (Math.sqrt(vhat) + this.eps);
        if (this.wd !== 0) update += this.lr * this.wd * p.data[j]; // decoupled
        p.data[j] -= update;
      }
    }
  }
  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
  setLr(lr: number): void {
    this.lr = lr;
  }
}

/**
 * Global-norm gradient clipping. Compute the L2 norm over ALL grads concatenated; if it
 * exceeds maxNorm, scale every grad by maxNorm/norm — preserving the overall DIRECTION
 * (per-param clipping would distort it). Returns the pre-clip norm so training can log
 * spikes (the early-warning sign of instability). Without it, one bad batch -> huge grad ->
 * overshoot -> NaN.
 */
export function clipGradNorm(params: Tensor[], maxNorm: number): number {
  let sq = 0;
  for (const p of params) for (let i = 0; i < p.size; i++) sq += p.grad[i] * p.grad[i];
  const norm = Math.sqrt(sq);
  if (norm > maxNorm && norm > 0) {
    const scale = maxNorm / norm;
    for (const p of params) for (let i = 0; i < p.size; i++) p.grad[i] *= scale;
  }
  return norm;
}

/**
 * Cosine LR with linear warmup. step<warmup: linear 0->base; warmup..total: cosine base->0;
 * past total: 0. Warmup avoids the first-step overshoot while Adam moments settle; cosine
 * decays smoothly into a minimum. Pure function of step => fully reproducible.
 */
export function cosineWarmup(step: number, warmup: number, total: number, base: number): number {
  if (step < warmup) return (base * step) / Math.max(1, warmup);
  if (step > total) return 0;
  const progress = (step - warmup) / Math.max(1, total - warmup);
  return 0.5 * base * (1 + Math.cos(Math.PI * progress));
}

/**
 * Cross-entropy over (rows, classes) logits and integer targets. Returns a SCALAR Tensor
 * with correct backward, so callers just do loss.backward(). Softmax is fused in (stable,
 * subtracts row-max); fused backward is (softmax - onehot)/rows — far more stable than
 * composing log(softmax) op-by-op (avoids log of a near-zero prob blowing up).
 */
export function crossEntropy(logits: Tensor, targets: number[] | Int32Array): Tensor {
  if (logits.shape.length !== 2) throw new Error(`crossEntropy: expected 2-D logits, got ${logits.shape}`);
  const [rows, classes] = logits.shape;
  if (targets.length !== rows) throw new Error(`crossEntropy: targets ${targets.length} != rows ${rows}`);
  const probs = new Float64Array(rows * classes);
  let loss = 0;
  for (let r = 0; r < rows; r++) {
    const base = r * classes;
    let max = -Infinity;
    for (let j = 0; j < classes; j++) max = Math.max(max, logits.data[base + j]);
    let denom = 0;
    for (let j = 0; j < classes; j++) {
      const e = Math.exp(logits.data[base + j] - max);
      probs[base + j] = e;
      denom += e;
    }
    for (let j = 0; j < classes; j++) probs[base + j] /= denom;
    const tgt = targets[r];
    if (tgt < 0 || tgt >= classes) throw new Error(`crossEntropy: target ${tgt} out of range`);
    loss += -Math.log(probs[base + tgt] + 1e-12);
  }
  loss /= rows;
  const out = new Tensor([loss], [1], [logits], "cross_entropy");
  out._backward = () => {
    const g = out.grad[0] / rows;
    for (let r = 0; r < rows; r++) {
      const base = r * classes;
      const tgt = targets[r];
      for (let j = 0; j < classes; j++) logits.grad[base + j] += g * (probs[base + j] - (j === tgt ? 1 : 0));
    }
  };
  return out;
}
