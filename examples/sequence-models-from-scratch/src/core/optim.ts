// core/optim.ts — Optimizers + gradient-norm clipping over a flat param list.
//
// WHY optimizers take a flat Tensor[] (not a Module): a single training loop often
//   updates params from several cells/layers at once (embedding + RNN + readout). The
//   optimizer should not know the module tree; it owns exactly the leaves you hand it.
//
// CORE INVARIANT — step() reads p.grad, never recomputes it. The caller MUST have run
//   loss.backward() since the last zeroGrad(). step() does NOT zero grads; call
//   zeroGrad() explicitly at the top of each iteration so the accumulate-not-overwrite
//   contract (tensor.ts) holds.
//
// FAILURE MODE clipGradNorm exists for: stage02 shows RNN gradients exploding. Without
//   clipping, one bad step sends params to NaN and the run is dead. Clipping rescales
//   the WHOLE gradient vector (preserving direction) when its global L2 norm exceeds a
//   threshold — and returns the PRE-clip norm so the chapter can print/plot it.

import type { Tensor } from "./tensor.js";

export interface Optimizer {
  step(): void;
  zeroGrad(): void;
}

/**
 * SGD with optional (heavy-ball) momentum.
 * v = mu*v + g ; p -= lr * v.  mu=0 recovers plain SGD.
 */
export class SGD implements Optimizer {
  private params: Tensor[];
  private lr: number;
  private mu: number;
  private vel: Float64Array[];
  constructor(params: Tensor[], opts: { lr?: number; momentum?: number } = {}) {
    this.params = params;
    this.lr = opts.lr ?? 0.1;
    this.mu = opts.momentum ?? 0;
    this.vel = params.map((p) => new Float64Array(p.size));
  }
  step(): void {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const v = this.vel[i];
      for (let j = 0; j < p.size; j++) {
        v[j] = this.mu * v[j] + p.grad[j];
        p.data[j] -= this.lr * v[j];
      }
    }
  }
  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
}

/**
 * Adam (Kingma & Ba). Bias-corrected first/second moment estimates.
 * WHY Adam is the default for the toy models here: tanh-saturating RNN/LSTM grads have
 *   wildly varying per-parameter scale; Adam's per-coordinate normalization lets one
 *   lr work across the whole net so the chapters don't need per-layer tuning.
 * FAILURE MODE the eps guards against: a parameter that has seen ~zero gradient has
 *   v≈0; without eps in the denominator the update divides by ~0 and explodes.
 */
export class Adam implements Optimizer {
  private params: Tensor[];
  private lr: number;
  private b1: number;
  private b2: number;
  private eps: number;
  private m: Float64Array[];
  private v: Float64Array[];
  private t = 0; // timestep, for bias correction
  constructor(
    params: Tensor[],
    opts: { lr?: number; betas?: [number, number]; eps?: number } = {},
  ) {
    this.params = params;
    this.lr = opts.lr ?? 1e-3;
    const [b1, b2] = opts.betas ?? [0.9, 0.999];
    this.b1 = b1;
    this.b2 = b2;
    this.eps = opts.eps ?? 1e-8;
    this.m = params.map((p) => new Float64Array(p.size));
    this.v = params.map((p) => new Float64Array(p.size));
  }
  step(): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.b1, this.t); // bias-correction denominators
    const bc2 = 1 - Math.pow(this.b2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const m = this.m[i];
      const v = this.v[i];
      for (let j = 0; j < p.size; j++) {
        const g = p.grad[j];
        m[j] = this.b1 * m[j] + (1 - this.b1) * g;
        v[j] = this.b2 * v[j] + (1 - this.b2) * g * g;
        const mHat = m[j] / bc1;
        const vHat = v[j] / bc2;
        p.data[j] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
}

/**
 * Clip the GLOBAL L2 norm of all param grads to `maxNorm`, in place.
 * Returns the PRE-clip global norm (so the exploding-gradient chapter can plot it).
 * INVARIANT: scales every grad by the SAME factor (maxNorm/norm) so gradient DIRECTION
 *   is preserved — clipping per-parameter would distort the descent direction.
 */
export function clipGradNorm(params: Tensor[], maxNorm: number): number {
  let sq = 0;
  for (const p of params) for (let j = 0; j < p.size; j++) sq += p.grad[j] * p.grad[j];
  const norm = Math.sqrt(sq);
  if (norm > maxNorm && norm > 0) {
    const scale = maxNorm / norm;
    for (const p of params) for (let j = 0; j < p.size; j++) p.grad[j] *= scale;
  }
  return norm;
}
