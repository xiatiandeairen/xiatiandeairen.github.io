// core/nn.ts — Module base + the handful of layers the whole book is assembled from.
//
// WHY a Module base at all: training needs three cross-cutting operations on a tree of
//   layers — collect all trainable Tensors (for the optimizer), zero their grads, and
//   run forward. Encoding these as a base contract means the transformer in chapter 8
//   is "just" Sequential(Embedding, Block, Block, ..., LayerNorm, Linear) and the
//   training loop never special-cases any layer.
//
// INVARIANT: parameters() returns LEAF tensors only (the actual learnables), recursively.
//   A tensor produced by an op (it has _prev) is NOT a parameter — it's recomputed each
//   forward. Returning intermediates would make the optimizer "update" throwaway nodes.
//
// FAILURE MODE this guards: a sub-module stored in a plain field that parameters()
//   forgets to recurse into => its weights never get grads applied => that layer silently
//   never trains. We make registration explicit via _modules / _params to avoid relying
//   on reflection over arbitrary fields.

import { Tensor } from "./autograd.js";
import { kaiming, xavier, type Rng } from "./rng.js";
import { noGradActive } from "./autograd.js";

export abstract class Module {
  // Explicit registries beat reflecting over `this` fields: order is stable (matters for
  // reproducible optimizer state) and there is no ambiguity about what counts as a param.
  protected _params: Tensor[] = [];
  protected _modules: Module[] = [];
  training = true;

  /** Register a leaf parameter and return it for convenient assignment. */
  protected param(t: Tensor): Tensor {
    this._params.push(t);
    return t;
  }

  /** Register a child module and return it. */
  protected child<M extends Module>(m: M): M {
    this._modules.push(m);
    return m;
  }

  /** Recursively collect all leaf parameters. Order = this layer's params, then children
   *  in registration order. Stable order is what lets Adam's per-param state stay aligned
   *  across steps. */
  parameters(): Tensor[] {
    const out = [...this._params];
    for (const m of this._modules) out.push(...m.parameters());
    return out;
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }

  /** Propagate train/eval mode down the tree. Dropout/LayerNorm read this. */
  setTraining(mode: boolean): void {
    this.training = mode;
    for (const m of this._modules) m.setTraining(mode);
  }

  abstract forward(x: Tensor): Tensor;
}

// ----------------------------------------------------------------------------
// Linear: y = x @ W + b , x:(batch, inF) W:(inF, outF) b:(1, outF)
// ----------------------------------------------------------------------------
export class Linear extends Module {
  W: Tensor;
  b: Tensor | null;

  constructor(inF: number, outF: number, rng: Rng, opts: { bias?: boolean; init?: "kaiming" | "xavier" } = {}) {
    super();
    const { bias = true, init = "kaiming" } = opts;
    // WHY init choice matters: see rng.ts. Default kaiming because most stacks here are
    // ReLU; pass init:'xavier' for tanh heads. Wrong init => vanishing/exploding even
    // with perfect grads.
    const gen = init === "kaiming" ? () => kaiming(inF, rng) : () => xavier(inF, outF, rng);
    this.W = this.param(Tensor.from([inF, outF], gen));
    this.b = bias ? this.param(Tensor.zeros([1, outF])) : null; // bias starts at 0 by convention
  }

  override forward(x: Tensor): Tensor {
    const out = x.matmul(this.W);
    if (!this.b) return out;
    // Broadcast bias row over the batch via the explicit broadcast op so its adjoint
    // (sum grads back to one row) is handled in autograd, not re-derived here.
    return out.add(this.b.broadcastRow(out.shape[0]));
  }
}

// ----------------------------------------------------------------------------
// Embedding: integer ids -> rows of a (vocab, dim) table.
// ----------------------------------------------------------------------------
export class Embedding extends Module {
  weight: Tensor; // (vocab, dim)
  vocab: number;
  dim: number;

  constructor(vocab: number, dim: number, rng: Rng) {
    super();
    this.vocab = vocab;
    this.dim = dim;
    // Small normal init; embeddings are not followed by a fan-in nonlinearity so a plain
    // ~N(0, 0.02^2) (GPT convention) keeps initial logits small and softmax near-uniform.
    this.weight = this.param(Tensor.from([vocab, dim], () => 0.02 * randnLike(rng)));
  }

  /** forward takes a Tensor of integer ids with shape (batch,) stored as floats.
   *  Output: (batch, dim). Adjoint: scatter each row-grad back to its source vocab row
   *  (and ACCUMULATE — the same token id appearing twice in a batch must sum its grads). */
  override forward(ids: Tensor): Tensor {
    const batch = ids.size;
    const out = new Float64Array(batch * this.dim);
    const idx = new Int32Array(batch);
    for (let i = 0; i < batch; i++) {
      const id = Math.round(ids.data[i]);
      if (id < 0 || id >= this.vocab) throw new Error(`Embedding: id ${id} out of range [0,${this.vocab})`);
      idx[i] = id;
      out.set(this.weight.data.subarray(id * this.dim, (id + 1) * this.dim), i * this.dim);
    }
    const t = new Tensor(out, [batch, this.dim], [this.weight], "embedding");
    t._backward = () => {
      for (let i = 0; i < batch; i++) {
        const src = idx[i] * this.dim;
        const dst = i * this.dim;
        for (let d = 0; d < this.dim; d++) this.weight.grad[src + d] += t.grad[dst + d];
      }
    };
    return t;
  }
}

// ----------------------------------------------------------------------------
// LayerNorm over the last axis: normalize each row to mean 0 / var 1, then scale+shift.
// ----------------------------------------------------------------------------
export class LayerNorm extends Module {
  gamma: Tensor; // (1, dim) learnable scale, init 1
  beta: Tensor; // (1, dim) learnable shift, init 0
  dim: number;
  eps: number;

  constructor(dim: number, eps = 1e-5) {
    super();
    this.dim = dim;
    this.eps = eps;
    this.gamma = this.param(Tensor.fill([1, dim], 1));
    this.beta = this.param(Tensor.zeros([1, dim]));
  }

  /** WHY a hand-written fused backward instead of composing ops: LayerNorm's adjoint
   *  couples every element in a row (the mean/var depend on all of them). Composing
   *  sub/mean/div would be correct but allocates many intermediate graphs; more
   *  importantly writing the closed form here lets the comment teach the actual gradient.
   *  EPS lives INSIDE the sqrt: var+eps prevents div-by-0 when a row is constant. */
  override forward(x: Tensor): Tensor {
    if (x.shape.length !== 2 || x.shape[1] !== this.dim)
      throw new Error(`LayerNorm: expected (batch, ${this.dim}), got ${x.shape}`);
    const [rows, dim] = x.shape;
    const out = new Float64Array(rows * dim);
    const mean = new Float64Array(rows);
    const invStd = new Float64Array(rows);
    const xhat = new Float64Array(rows * dim);
    for (let r = 0; r < rows; r++) {
      const base = r * dim;
      let m = 0;
      for (let j = 0; j < dim; j++) m += x.data[base + j];
      m /= dim;
      let v = 0;
      for (let j = 0; j < dim; j++) {
        const d = x.data[base + j] - m;
        v += d * d;
      }
      v /= dim;
      const inv = 1 / Math.sqrt(v + this.eps);
      mean[r] = m;
      invStd[r] = inv;
      for (let j = 0; j < dim; j++) {
        const xh = (x.data[base + j] - m) * inv;
        xhat[base + j] = xh;
        out[base + j] = xh * this.gamma.data[j] + this.beta.data[j];
      }
    }
    const t = new Tensor(out, x.shape, [x, this.gamma, this.beta], "layernorm");
    t._backward = () => {
      for (let r = 0; r < rows; r++) {
        const base = r * dim;
        const inv = invStd[r];
        // grads wrt xhat, then the standard LN reduction to grad wrt x:
        // dx = (1/std) * (dxhat - mean(dxhat) - xhat*mean(dxhat*xhat))
        let sumDxhat = 0;
        let sumDxhatXhat = 0;
        const dxhat = new Float64Array(dim);
        for (let j = 0; j < dim; j++) {
          const g = t.grad[base + j];
          const dx = g * this.gamma.data[j];
          dxhat[j] = dx;
          sumDxhat += dx;
          sumDxhatXhat += dx * xhat[base + j];
          // gamma/beta grads accumulate across rows (they are shared (1,dim) params).
          this.gamma.grad[j] += g * xhat[base + j];
          this.beta.grad[j] += g;
        }
        const mDxhat = sumDxhat / dim;
        const mDxhatXhat = sumDxhatXhat / dim;
        for (let j = 0; j < dim; j++)
          x.grad[base + j] += inv * (dxhat[j] - mDxhat - xhat[base + j] * mDxhatXhat);
      }
    };
    return t;
  }
}

// ----------------------------------------------------------------------------
// Dropout: training-time random zeroing with inverted scaling.
// ----------------------------------------------------------------------------
export class Dropout extends Module {
  p: number;
  rng: Rng;

  constructor(p: number, rng: Rng) {
    super();
    if (p < 0 || p >= 1) throw new Error(`Dropout: p must be in [0,1), got ${p}`);
    this.p = p;
    this.rng = rng;
  }

  /** INVERTED dropout: scale kept activations by 1/(1-p) at TRAIN time so that the
   *  expected activation is unchanged and inference needs NO rescaling (it's identity).
   *  This is why eval mode / noGrad just returns x untouched. The classic bug is scaling
   *  at test time instead — here test time is a literal no-op, which is the point. */
  override forward(x: Tensor): Tensor {
    if (!this.training || noGradActive() || this.p === 0) return x;
    const n = x.size;
    const keep = 1 - this.p;
    const scale = 1 / keep;
    const mask = new Float64Array(n);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const m = this.rng() < keep ? scale : 0;
      mask[i] = m;
      out[i] = x.data[i] * m;
    }
    const t = new Tensor(out, x.shape, [x], "dropout");
    t._backward = () => {
      for (let i = 0; i < n; i++) x.grad[i] += mask[i] * t.grad[i];
    };
    return t;
  }
}

// ----------------------------------------------------------------------------
// Sequential: run modules in order. The composition primitive for everything.
// ----------------------------------------------------------------------------
export class Sequential extends Module {
  constructor(mods: Module[]) {
    super();
    for (const m of mods) this.child(m);
  }
  override forward(x: Tensor): Tensor {
    let h = x;
    for (const m of this._modules) h = m.forward(h);
    return h;
  }
}

// Small helper kept local to avoid a circular import dance with rng for one call site.
function randnLike(rng: Rng): number {
  // reuse Box–Muller via rng draws; duplicated tiny bit rather than import randn to keep
  // this file's import surface minimal. Two draws => stream-aligned with rng.randn.
  let u1 = rng();
  const u2 = rng();
  if (u1 < 1e-12) u1 = 1e-12;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
