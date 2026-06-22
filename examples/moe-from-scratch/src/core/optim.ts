// core/optim.ts — SGD(+momentum) and Adam, operating on a parameters() list.
//
// WHY decouple optimizer from model: the optimizer only needs the flat list of leaf
//   Values (their .data to update, their .grad to read). This is the same parameters()
//   contract nn.ts guarantees — one list, mutated in place. No model coupling.
//
// INVARIANT — zeroGrad() before every backward(): tensor.backward() ACCUMULATES grads.
//   If you step() then backward() again without zeroGrad(), the second step uses
//   stale + new grad and training diverges in a way that looks like "bad LR". The
//   stages call optim.zeroGrad() at the top of each step to make this impossible to forget.
//
// FAILURE MODE Adam guards against: a fresh Adam with bias-uncorrected moments takes huge
//   first steps (m,v ~ 0 -> m̂ explodes). The 1-b1^t / 1-b2^t bias correction below fixes
//   the cold-start; dropping it is a subtle bug that only shows up in the first ~20 steps.

import type { Value } from "./tensor.js";

export interface Optimizer {
  step(): void;
  zeroGrad(): void;
}

/** Vanilla SGD with optional heavy-ball momentum. velocity = mu*velocity - lr*grad. */
export class SGD implements Optimizer {
  private params: Value[];
  private lr: number;
  private momentum: number;
  private velocity: Float64Array[];
  constructor(params: Value[], lr: number, momentum = 0) {
    this.params = params;
    this.lr = lr;
    this.momentum = momentum;
    this.velocity = params.map((p) => new Float64Array(p.data.length));
  }
  step(): void {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const v = this.velocity[i];
      for (let k = 0; k < p.data.length; k++) {
        v[k] = this.momentum * v[k] - this.lr * p.grad[k];
        p.data[k] += v[k];
      }
    }
  }
  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
}

/** Adam with bias correction. The default optimizer for the MoE stages (gating nets are
 *  ill-conditioned; per-parameter adaptive LR keeps experts and the router moving together). */
export class Adam implements Optimizer {
  private params: Value[];
  private lr: number;
  private b1: number;
  private b2: number;
  private eps: number;
  private m: Float64Array[];
  private v: Float64Array[];
  private t: number;
  constructor(params: Value[], lr = 1e-2, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.b1 = b1;
    this.b2 = b2;
    this.eps = eps;
    this.m = params.map((p) => new Float64Array(p.data.length));
    this.v = params.map((p) => new Float64Array(p.data.length));
    this.t = 0;
  }
  step(): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.b1, this.t); // bias-correction denominators
    const bc2 = 1 - Math.pow(this.b2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const m = this.m[i];
      const v = this.v[i];
      for (let k = 0; k < p.data.length; k++) {
        const g = p.grad[k];
        m[k] = this.b1 * m[k] + (1 - this.b1) * g;
        v[k] = this.b2 * v[k] + (1 - this.b2) * g * g;
        const mhat = m[k] / bc1;
        const vhat = v[k] / bc2;
        p.data[k] -= (this.lr * mhat) / (Math.sqrt(vhat) + this.eps);
      }
    }
  }
  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
}
