// core/optim.ts — SGD and Adam over a fixed parameter list.
//
// CONTRACT: an optimizer owns a snapshot of the parameter Tensors (the array from
//   Module.parameters()). Each training step is exactly: forward -> loss.backward()
//   (fills .grad) -> opt.step() (reads .grad, writes .data) -> opt.zeroGrad().
//
// FAILURE MODE this guards against: forgetting zeroGrad(). Grads ACCUMULATE (+=) in the
//   autograd engine, so a skipped zeroGrad means step N sees the SUM of grads from steps
//   1..N — the effective learning rate balloons and training diverges. We expose zeroGrad
//   on the optimizer so it sits right next to step() and is hard to forget.

import { Tensor } from "./tensor.js";

export interface Optimizer {
  step(): void;
  zeroGrad(): void;
}

/**
 * SGD with optional heavy-ball momentum: v = μv + g ; θ = θ - lr·v.
 * WHY momentum: plain SGD on the noisy diffusion loss zig-zags across narrow valleys;
 *   momentum averages successive grads, damping the zig-zag and speeding descent. μ=0
 *   recovers vanilla SGD.
 */
export class SGD implements Optimizer {
  private params: Tensor[];
  private lr: number;
  private momentum: number;
  private velocity: Float64Array[]; // one buffer per param, persists across steps

  constructor(params: Tensor[], lr: number, momentum = 0) {
    this.params = params;
    this.lr = lr;
    this.momentum = momentum;
    this.velocity = params.map((p) => new Float64Array(p.size));
  }

  step(): void {
    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi];
      const v = this.velocity[pi];
      for (let i = 0; i < p.size; i++) {
        v[i] = this.momentum * v[i] + p.grad[i];
        p.data[i] -= this.lr * v[i];
      }
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
}

/**
 * Adam — DDPM's default optimizer. Keeps per-parameter running estimates of the grad's
 * 1st moment (m, mean) and 2nd moment (v, uncentered variance), bias-corrects them for the
 * cold-start zeros, and scales each step by 1/sqrt(v): big-variance params take small steps,
 * stable params take large ones. This per-parameter adaptivity is why Adam trains diffusion
 * nets reliably where a single global SGD lr is fragile.
 *
 * INVARIANT: t (step counter) starts at 0 and increments BEFORE bias correction, so the
 *   first step uses t=1. Off-by-one here makes the first few steps explode or stall.
 * FAILURE MODE: eps too small (or 0) -> division blowup when v≈0 early on; 1e-8 is standard.
 */
export class Adam implements Optimizer {
  private params: Tensor[];
  private lr: number;
  private b1: number;
  private b2: number;
  private eps: number;
  private m: Float64Array[];
  private v: Float64Array[];
  private t: number; // step count, for bias correction

  constructor(params: Tensor[], lr: number, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.b1 = b1;
    this.b2 = b2;
    this.eps = eps;
    this.m = params.map((p) => new Float64Array(p.size));
    this.v = params.map((p) => new Float64Array(p.size));
    this.t = 0;
  }

  step(): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.b1, this.t); // bias-correction denominators
    const bc2 = 1 - Math.pow(this.b2, this.t);
    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi];
      const m = this.m[pi];
      const v = this.v[pi];
      for (let i = 0; i < p.size; i++) {
        const g = p.grad[i];
        m[i] = this.b1 * m[i] + (1 - this.b1) * g;
        v[i] = this.b2 * v[i] + (1 - this.b2) * g * g;
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        p.data[i] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }
}
