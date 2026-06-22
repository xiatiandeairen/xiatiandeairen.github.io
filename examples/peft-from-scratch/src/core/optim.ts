// core/optim.ts — SGD(momentum) and Adam, with two PEFT-shaped guarantees.
//
// GUARANTEE 1 — only trainable params move: step() skips any param whose requires_grad is
//   false. This is the optimizer-side half of freezing. Even if a frozen base tensor
//   somehow accumulated grad, the optimizer would not move it. Belt-and-suspenders with
//   tensor.ts (which already declines to write grad into frozen leaves).
//
// GUARANTEE 2 — paramGroups: different param groups can carry different lr. PEFT recipes
//   routinely train, say, LoRA matrices at one lr and a trainable LayerNorm at another;
//   QLoRA does similar. Groups make that a one-liner instead of two optimizers.
//
// MEMORY NOTE (the whole point of PEFT): Adam keeps TWO state buffers (m, v) PER param.
//   So a fully-trained model's optimizer state is ~2x the param count in floats. PEFT
//   freezes 99%+ of params, so those buffers are allocated ONLY for the tiny trainable set
//   — which is most of the real-world memory saving. We allocate state lazily, keyed by the
//   param tensor, so frozen params cost zero optimizer memory here too (mirrors reality).
//
// FAILURE MODE guarded: forgetting to zeroGrad between steps. Grads accumulate (tensor.ts),
//   so without zeroGrad the effective grad is the running sum and the model diverges. We
//   expose zeroGrad() on the optimizer so the training loop has one obvious call.

import { Tensor } from "./tensor.js";

export interface ParamGroup {
  params: Tensor[];
  lr?: number; // overrides the optimizer default lr for this group
}

abstract class Optimizer {
  protected groups: ParamGroup[];
  protected defaultLr: number;
  constructor(params: Tensor[] | ParamGroup[], defaultLr: number) {
    this.defaultLr = defaultLr;
    // Accept a flat param list OR explicit groups. Normalize to groups internally.
    if (params.length > 0 && (params[0] as ParamGroup).params !== undefined) {
      this.groups = params as ParamGroup[];
    } else {
      this.groups = [{ params: params as Tensor[] }];
    }
  }
  /** Zero grads on every param the optimizer manages (trainable or not — frozen are no-ops). */
  zeroGrad(): void {
    for (const g of this.groups) for (const p of g.params) p.zeroGrad();
  }
  abstract step(): void;
  protected lrOf(g: ParamGroup): number {
    return g.lr ?? this.defaultLr;
  }
}

/**
 * SGD with classic (heavy-ball) momentum: v = mu*v + grad ; p -= lr*v.
 * WHY momentum: even at toy scale it visibly smooths the loss curve, which is one of the
 *   "transferable shape" lessons the book leans on.
 */
export class SGD extends Optimizer {
  private mu: number;
  private vel = new Map<Tensor, Float64Array>();
  constructor(params: Tensor[] | ParamGroup[], lr = 0.1, momentum = 0.9) {
    super(params, lr);
    this.mu = momentum;
  }
  override step(): void {
    for (const g of this.groups) {
      const lr = this.lrOf(g);
      for (const p of g.params) {
        if (!p.requires_grad) continue; // GUARANTEE 1
        let v = this.vel.get(p);
        if (!v) {
          v = new Float64Array(p.size); // lazy: frozen params never allocate state
          this.vel.set(p, v);
        }
        for (let i = 0; i < p.size; i++) {
          v[i] = this.mu * v[i] + p.grad[i];
          p.data[i] -= lr * v[i];
        }
      }
    }
  }
}

/**
 * Adam: per-parameter adaptive lr from running 1st/2nd grad moments, with bias correction.
 * WHY bias correction: m,v start at 0, so early steps are biased toward 0; the /(1-β^t)
 *   terms undo that. Skipping correction makes the first ~dozen steps crawl, which would
 *   distort the toy convergence-step numbers the book reports.
 * INVARIANT: t is incremented ONCE per step() (global), not per-param, so all params share
 *   the same correction schedule.
 */
export class Adam extends Optimizer {
  private b1: number;
  private b2: number;
  private eps: number;
  private t = 0;
  private m = new Map<Tensor, Float64Array>();
  private v = new Map<Tensor, Float64Array>();
  constructor(params: Tensor[] | ParamGroup[], lr = 1e-2, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    super(params, lr);
    this.b1 = beta1;
    this.b2 = beta2;
    this.eps = eps;
  }
  override step(): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.b1, this.t);
    const bc2 = 1 - Math.pow(this.b2, this.t);
    for (const g of this.groups) {
      const lr = this.lrOf(g);
      for (const p of g.params) {
        if (!p.requires_grad) continue; // GUARANTEE 1
        let mt = this.m.get(p);
        let vt = this.v.get(p);
        if (!mt || !vt) {
          mt = new Float64Array(p.size);
          vt = new Float64Array(p.size);
          this.m.set(p, mt);
          this.v.set(p, vt);
        }
        for (let i = 0; i < p.size; i++) {
          const grad = p.grad[i];
          mt[i] = this.b1 * mt[i] + (1 - this.b1) * grad;
          vt[i] = this.b2 * vt[i] + (1 - this.b2) * grad * grad;
          const mHat = mt[i] / bc1;
          const vHat = vt[i] / bc2;
          p.data[i] -= (lr * mHat) / (Math.sqrt(vHat) + this.eps);
        }
      }
    }
  }
  /** Number of float slots Adam currently holds as state (for honest memory accounting). */
  stateFloats(): number {
    let n = 0;
    for (const buf of this.m.values()) n += buf.length;
    for (const buf of this.v.values()) n += buf.length;
    return n;
  }
}
