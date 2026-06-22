// core/optim.ts — the optimizers that turn gradients into learning.
//
// Both optimizers operate on a flat list of Value parameters (collected once at
// model build time). Why a flat param list rather than walking the graph: the
// graph is rebuilt every forward pass (it's dynamic), but the PARAMETERS persist
// across steps. So the model owns a stable Value[] and hands it to the optimizer;
// the optimizer never touches the transient graph, only the leaves.
//
// The mandatory loop contract every training stage follows:
//   for each step:
//     opt.zeroGrad()          // 1. clear last step's grads (autograd accumulates!)
//     const loss = forward()  // 2. build graph, get scalar loss
//     loss.backward()         // 3. fill param.grad
//     opt.step()              // 4. move params down the gradient
// Skipping zeroGrad is the canonical failure: grads sum across steps, effective
// lr explodes, loss → NaN. We expose zeroGrad ON the optimizer so it's impossible
// to forget which params to clear.

import { Value } from "./autograd.js";

// Common interface so stages can swap SGD <-> Adam without touching the loop.
export interface Optimizer {
  zeroGrad(): void;
  step(): void;
}

// Plain SGD: theta <- theta - lr * grad. The honest baseline. Its weakness —
// one global lr for every parameter regardless of gradient scale — is exactly
// what motivates Adam, and the book shows the convergence-speed gap between them
// on the same skip-gram task. We keep SGD so that contrast is reproducible.
export class SGD implements Optimizer {
  private params: Value[];
  private lr: number;

  constructor(params: Value[], lr: number) {
    this.params = params;
    this.lr = lr;
  }

  zeroGrad(): void {
    for (const p of this.params) p.grad = 0;
  }

  step(): void {
    for (const p of this.params) p.data -= this.lr * p.grad;
  }
}

// Adam: per-parameter adaptive lr from running estimates of the 1st moment (mean,
// m) and 2nd moment (uncentered variance, v) of the gradient, with bias
// correction for the cold-start zeros.
//
// Why it wins on embeddings: token frequencies are wildly skewed, so rare-word
// gradients are tiny and frequent-word gradients are huge. SGD's single lr either
// crawls on rare words or overshoots frequent ones. Adam normalizes each param by
// its own gradient magnitude (m/sqrt(v)), giving every word a comparable effective
// step. This is the practical reason real word2vec-style training uses adaptive
// optimizers.
//
// Failure mode it can hide: because the effective step is ~lr regardless of true
// gradient scale, a too-large lr still diverges but more sneakily; the book
// demonstrates this in the negative-sampling stage.
export class Adam implements Optimizer {
  private params: Value[];
  private lr: number;
  private beta1: number;
  private beta2: number;
  private eps: number;
  private m: number[];
  private v: number[];
  private t: number; // timestep, drives bias correction

  constructor(params: Value[], lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    // One moment slot per parameter, indexed parallel to `params`. Initialized to
    // zero — the bias correction below is precisely what compensates for this
    // cold start so early steps aren't artificially tiny.
    this.m = new Array(params.length).fill(0);
    this.v = new Array(params.length).fill(0);
    this.t = 0;
  }

  zeroGrad(): void {
    for (const p of this.params) p.grad = 0;
  }

  step(): void {
    this.t += 1;
    // Precompute bias-correction denominators once per step (same for all params).
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      const g = this.params[i].grad;
      // Exponential moving averages of grad and grad^2.
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * g;
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * g * g;
      const mHat = this.m[i] / bc1;
      const vHat = this.v[i] / bc2;
      // eps inside the sqrt vs outside is a known subtlety; outside (as here)
      // matches the original Adam paper and is robust enough at toy scale.
      this.params[i].data -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
    }
  }
}

// Helper: flatten model parameter structures (Vec / Mat / nested) into the single
// Value[] both optimizers expect. Stages build embedding matrices as Value[][];
// this collects every leaf so none is accidentally left untrained — a silent bug
// where some rows never update.
export function collectParams(...groups: Value[][][] | Value[][] | Value[]): Value[] {
  const out: Value[] = [];
  const walk = (x: unknown): void => {
    if (x instanceof Value) {
      out.push(x);
    } else if (Array.isArray(x)) {
      for (const el of x) walk(el);
    } else {
      throw new Error("collectParams: expected Value or nested arrays of Value");
    }
  };
  walk(groups);
  return out;
}
