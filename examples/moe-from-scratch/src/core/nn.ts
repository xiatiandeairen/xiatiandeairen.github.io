// core/nn.ts — Layer primitives, all built ONLY from tensor.ts operators.
//
// WHY thin layers over the engine: a Linear is just matmul + row-broadcast add; an Expert
//   is two Linears + a nonlinearity. Keeping layers as thin compositions means gradCheck
//   on the engine transitively validates the layers — we never write a custom backward
//   at this level, so there is no second place for an adjoint bug to hide.
//
// INVARIANT: parameters() returns the SAME Value objects the forward pass uses (not
//   copies). The optimizer mutates their .data in place and reads their .grad; if a layer
//   ever returned copies here, training would update phantom params and the live ones
//   would never move. Each module that holds params MUST surface them via parameters().
//
// FAILURE MODE this guards: forgetting to register a sub-module's params in a container's
//   parameters() -> that sub-module silently never trains. Sequential/Expert explicitly
//   concat children's parameters() to make this hard to get wrong.

import { Value, type Shape } from "./tensor.js";
import { rng as makeRng, kaimingStd, type Rng } from "./prng.js";

export interface Module {
  forward(x: Value): Value;
  parameters(): Value[];
}

/**
 * Affine layer: y = x @ W + b, with x shape (batch, in), W (in, out), b (1, out).
 * Init: Kaiming on W (good default for ReLU/GELU experts); b starts at 0.
 * The bias add relies on tensor.add's row-broadcast adjoint — that's the only "magic".
 */
export class Linear implements Module {
  W: Value;
  b: Value;
  constructor(inDim: number, outDim: number, rng: Rng, bias = true) {
    const std = kaimingStd(inDim);
    const w = new Float64Array(inDim * outDim);
    for (let k = 0; k < w.length; k++) w[k] = rng.normal() * std;
    this.W = new Value(w, [inDim, outDim], [], "leaf");
    this.b = bias ? Value.zeros(1, outDim) : Value.zeros(1, outDim);
    this._hasBias = bias;
  }
  private _hasBias: boolean;
  forward(x: Value): Value {
    const z = x.matmul(this.W);
    return this._hasBias ? z.add(this.b) : z;
  }
  parameters(): Value[] {
    return this._hasBias ? [this.W, this.b] : [this.W];
  }
}

/**
 * Embedding table: integer ids -> dense rows. Used by makeTokenStream routing demos.
 * lookup(id) returns a [1,dim] Value sharing grad with the table row (so token grads
 * accumulate into the right row across a batch). INVARIANT: the returned slice writes
 * grad back into the SAME table.grad buffer via the gather-row adjoint below.
 */
export class Embedding implements Module {
  table: Value;
  dim: number;
  constructor(vocab: number, dim: number, rng: Rng) {
    const d = new Float64Array(vocab * dim);
    // Small init: embeddings feed into normed layers; large init drowns the signal.
    for (let k = 0; k < d.length; k++) d[k] = rng.normal() * 0.02;
    this.table = new Value(d, [vocab, dim], [], "leaf");
    this.dim = dim;
  }
  lookup(id: number): Value {
    const out = new Value(new Float64Array(this.dim), [1, this.dim], [this.table], `embed[${id}]`);
    const base = id * this.dim;
    for (let j = 0; j < this.dim; j++) out.data[j] = this.table.data[base + j];
    out._backward = () => {
      for (let j = 0; j < this.dim; j++) this.table.grad[base + j] += out.grad[j];
    };
    return out;
  }
  forward(x: Value): Value {
    // Embedding has no batch forward in this book; lookup() is the real entry point.
    return x;
  }
  parameters(): Value[] {
    return [this.table];
  }
}

/**
 * LayerNorm over the feature axis of a [1,dim] row. Stabilizes gating/expert inputs so
 * one feature can't dominate the route. Built from engine ops only (mean/sub/pow/...).
 * gamma/beta are learnable per-feature scale/shift, init to 1/0 (identity at start).
 */
export class LayerNorm implements Module {
  gamma: Value;
  beta: Value;
  eps: number;
  constructor(dim: number, eps = 1e-5) {
    this.gamma = Value.zeros(1, dim, 1); // ones
    this.beta = Value.zeros(1, dim, 0);
    this.eps = eps;
  }
  forward(x: Value): Value {
    if (x.rows !== 1) throw new Error("LayerNorm expects a [1,dim] row");
    const mu = x.mean(); // scalar
    const centered = x.sub(mu); // broadcast scalar -> row via mul path? use add of -mu
    // mean of squares:
    const variance = centered.pow(2).mean(); // scalar
    const denom = variance.add(Value.scalar(this.eps)).pow(0.5); // scalar std
    const normed = centered.div(denom); // [1,dim]
    return normed.mul(this.gamma).add(this.beta);
  }
  parameters(): Value[] {
    return [this.gamma, this.beta];
  }
}

/** Sequential container: forwards in order; params concatenated in order. */
export class Sequential implements Module {
  layers: Module[];
  constructor(...layers: Module[]) {
    this.layers = layers;
  }
  forward(x: Value): Value {
    let h = x;
    for (const l of this.layers) h = l.forward(h);
    return h;
  }
  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }
}

export type Activation = "relu" | "gelu" | "tanh";

/**
 * Expert = a 2-layer MLP (in -> hidden -> out). The atom an MoE layer routes tokens to.
 * Each expert is an independent parameter set; the whole point of MoE is that only a
 * FEW experts fire per token, so total params >> activated params.
 * GELU default (smooth, transformer-standard); ReLU/tanh available for ablations.
 */
export class Expert implements Module {
  fc1: Linear;
  fc2: Linear;
  act: Activation;
  constructor(inDim: number, hidden: number, outDim: number, rng: Rng, act: Activation = "gelu") {
    this.fc1 = new Linear(inDim, hidden, rng);
    this.fc2 = new Linear(hidden, outDim, rng);
    this.act = act;
  }
  forward(x: Value): Value {
    const h = this.fc1.forward(x);
    const a = this._activate(h);
    return this.fc2.forward(a);
  }
  private _activate(h: Value): Value {
    if (this.act === "relu") return h.relu();
    if (this.act === "tanh") return h.tanh();
    return gelu(h);
  }
  parameters(): Value[] {
    return [...this.fc1.parameters(), ...this.fc2.parameters()];
  }
}

/**
 * GELU via the tanh approximation: 0.5x(1 + tanh(√(2/π)(x + 0.044715 x³))).
 * WHY the approximation: the exact GELU needs erf, which our engine doesn't expose; the
 *   tanh form is what GPT-2/most transformers actually ship and is composed purely from
 *   engine ops, so its backward is auto-derived (no custom adjoint to get wrong).
 */
export function gelu(x: Value): Value {
  const c = Math.sqrt(2 / Math.PI);
  const x3 = x.mul(x).mul(x); // elementwise cube (same-shape mul)
  const inner = x.add(x3.mul(Value.scalar(0.044715))).mul(Value.scalar(c));
  const t = inner.tanh().add(Value.scalar(1)); // (1 + tanh(...)) needs scalar broadcast
  return x.mul(t).mul(Value.scalar(0.5));
}

/**
 * Cross-entropy for ONE example: logits is [1,C], target is the class index.
 * Implemented as logsumexp(logits) - logits[target] for numerical stability — never
 * materializes softmax then logs it (that path loses precision and can log(0)).
 * INVARIANT: returns a scalar loss; average across a batch outside.
 */
export function crossEntropy(logits: Value, target: number): Value {
  if (logits.rows !== 1) throw new Error("crossEntropy expects [1,C] logits");
  const lse = logits.logsumexpRow();
  const picked = logits.gather(target);
  return lse.sub(picked);
}

// Re-export so stages can build their own rng/modules from one import surface.
export { makeRng as rng };
export type { Rng, Shape };
