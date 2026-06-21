// core/optim.ts — Optimizers + grad clipping + LR schedule.
//
// CONTRACT every optimizer obeys:
//   - holds a fixed list of parameter Tensors (taken once at construction; order frozen
//     so per-param state arrays stay index-aligned across steps).
//   - step(): read each param's .grad (already filled by backward), mutate .data in place.
//   - zeroGrad(): clear all .grad (you MUST call this each iteration — autograd ACCUMULATES
//     grads on purpose, so a missing zeroGrad silently sums grads across steps => the
//     classic "loss explodes after a few iterations" bug).
//
// HONESTY NOTE: these update rules are the textbook ones; the only deliberate omission is
//   sparse/fused kernels. On toy data the convergence curves are real but optimistic in
//   absolute steps-to-converge; the transferable signal is RELATIVE (Adam beats SGD here,
//   warmup stabilizes early steps), not the absolute step counts.

import { Tensor } from "./autograd.js";

export interface Optimizer {
  step(): void;
  zeroGrad(): void;
}

export class SGD implements Optimizer {
  private params: Tensor[];
  private lr: number;
  private momentum: number;
  private weightDecay: number;
  private velocity: Float64Array[]; // per-param momentum buffer, lazily all-zero

  constructor(params: Tensor[], opts: { lr: number; momentum?: number; weightDecay?: number }) {
    this.params = params;
    this.lr = opts.lr;
    this.momentum = opts.momentum ?? 0;
    this.weightDecay = opts.weightDecay ?? 0;
    this.velocity = params.map((p) => new Float64Array(p.size));
  }

  /** v = momentum*v + grad(+wd*param); param -= lr*v.
   *  WHY decoupled order matters less for SGD than Adam, but we add weight decay INTO the
   *  gradient here (classic L2) — note this is the "coupled" form; AdamW below shows the
   *  decoupled form and why it differs. */
  step(): void {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const v = this.velocity[i];
      for (let j = 0; j < p.size; j++) {
        let g = p.grad[j];
        if (this.weightDecay !== 0) g += this.weightDecay * p.data[j];
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
 * Adam / AdamW.
 * WHY bias correction: m and v are initialized at 0, so early estimates are biased toward
 *   0; dividing by (1 - beta^t) un-biases them. Skipping it makes the first ~1/(1-beta)
 *   steps take tiny effective steps — a real, observable slow start.
 * WHY a `decoupled` flag (AdamW vs Adam): coupled weight decay (Adam) folds wd into the
 *   gradient, so the adaptive denominator ALSO scales the decay — large-grad params get
 *   decayed less, which is not what "weight decay" should mean. AdamW applies decay
 *   directly to params, independent of the adaptive scaling. This matters for transformers.
 */
export class Adam implements Optimizer {
  private params: Tensor[];
  private lr: number;
  private beta1: number;
  private beta2: number;
  private eps: number;
  private weightDecay: number;
  private decoupled: boolean;
  private m: Float64Array[];
  private v: Float64Array[];
  private t = 0; // global step, drives bias correction

  constructor(
    params: Tensor[],
    opts: {
      lr: number;
      beta1?: number;
      beta2?: number;
      eps?: number;
      weightDecay?: number;
      decoupled?: boolean; // true => AdamW
    },
  ) {
    this.params = params;
    this.lr = opts.lr;
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.999;
    this.eps = opts.eps ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0;
    this.decoupled = opts.decoupled ?? false;
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
        let g = p.grad[j];
        // Coupled decay folds into grad BEFORE the moment updates...
        if (this.weightDecay !== 0 && !this.decoupled) g += this.weightDecay * p.data[j];
        m[j] = this.beta1 * m[j] + (1 - this.beta1) * g;
        v[j] = this.beta2 * v[j] + (1 - this.beta2) * g * g;
        const mhat = m[j] / bc1;
        const vhat = v[j] / bc2;
        let update = (this.lr * mhat) / (Math.sqrt(vhat) + this.eps);
        // ...decoupled (AdamW) decay applies directly to the param, untouched by vhat.
        if (this.weightDecay !== 0 && this.decoupled) update += this.lr * this.weightDecay * p.data[j];
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

/** Convenience: AdamW is just Adam with decoupled=true. Named for discoverability. */
export class AdamW extends Adam {
  constructor(
    params: Tensor[],
    opts: { lr: number; beta1?: number; beta2?: number; eps?: number; weightDecay?: number },
  ) {
    super(params, { ...opts, decoupled: true });
  }
}

/**
 * Global-norm gradient clipping. Compute the L2 norm of ALL grads concatenated; if it
 * exceeds maxNorm, scale every grad by maxNorm/norm.
 * WHY global (not per-param): preserves the DIRECTION of the overall gradient while
 *   capping its magnitude — clipping each param independently would distort direction.
 * Returns the PRE-clip norm so the training loop can log it (spikes in this number are
 *   the early-warning sign of instability). FAILURE MODE without clipping in deep nets:
 *   one bad batch produces a huge grad, the step overshoots, loss -> NaN, unrecoverable.
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
 * Cosine LR schedule with linear warmup.
 *   step < warmup:           linear ramp 0 -> base
 *   warmup <= step <= total: cosine decay base -> 0
 *   step > total:            0
 * WHY warmup: early steps have noisy/large grads (Adam moments not yet settled); ramping
 *   the LR avoids the first-step overshoot. WHY cosine: smooth decay to ~0 lets the model
 *   settle into a minimum without the abrupt drops of step schedules. Pure function of
 *   step => fully reproducible.
 */
export function cosineWarmup(step: number, warmup: number, total: number, base: number): number {
  if (step < warmup) return (base * step) / Math.max(1, warmup);
  if (step > total) return 0;
  const progress = (step - warmup) / Math.max(1, total - warmup);
  return 0.5 * base * (1 + Math.cos(Math.PI * progress));
}
