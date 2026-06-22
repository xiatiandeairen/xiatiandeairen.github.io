// core/nn.ts — Layers + parameter containers, with FREEZE as a first-class concept.
//
// WHY freeze is first-class here (not bolted on per-stage): PEFT is, mechanically, "freeze
//   almost everything, train a tiny add-on". So the Module base must answer two questions
//   cheaply and correctly for every stage:
//     1. parameters()  — every leaf Tensor in this subtree (for checkpointing / grad-check)
//     2. trainable()   — the subset with requires_grad=true (what the optimizer steps)
//   numParams({trainableOnly}) is built on these so a stage can print the REAL trainable
//   ratio (e.g. "LoRA: 0.5% of base") instead of a number quoted from a paper.
//
// INVARIANT: a frozen param is frozen by flipping its Tensor.requires_grad=false. There is
//   no separate "frozen list" to drift out of sync — the single source of truth is the
//   flag on the tensor, which the autodiff engine already honors (see tensor.ts header).
//
// FAILURE MODE this guards against: a stage thinking it froze the base but the optimizer
//   still moving base weights because freeze only updated a bookkeeping array, not the
//   tensors. Here freeze() mutates the tensors themselves, so it cannot disagree.

import { Tensor } from "./tensor.js";
import { normal, kaimingStd, xavierStd } from "./prng.js";

export abstract class Module {
  /** Return every parameter leaf in this module and its children. Override-friendly:
   *   default scans own enumerable Tensor fields + any child Module fields. */
  parameters(): Tensor[] {
    const out: Tensor[] = [];
    for (const key of Object.keys(this)) {
      const v = (this as Record<string, unknown>)[key];
      if (v instanceof Tensor) out.push(v);
      else if (v instanceof Module) out.push(...v.parameters());
      else if (Array.isArray(v)) {
        for (const item of v) {
          if (item instanceof Tensor) out.push(item);
          else if (item instanceof Module) out.push(...item.parameters());
        }
      }
    }
    return out;
  }

  /** Only the leaves the optimizer is allowed to move. */
  trainable(): Tensor[] {
    return this.parameters().filter((p) => p.requires_grad);
  }

  /** Only the frozen leaves (the PEFT "base"). */
  frozen(): Tensor[] {
    return this.parameters().filter((p) => !p.requires_grad);
  }

  /** Count params; trainableOnly drives the "X% trainable" honesty metric. */
  numParams(opts: { trainableOnly?: boolean } = {}): number {
    const ps = opts.trainableOnly ? this.trainable() : this.parameters();
    return ps.reduce((acc, p) => acc + p.size, 0);
  }

  /** Freeze the whole subtree (set requires_grad=false on every leaf). PEFT base setup. */
  freeze(): void {
    for (const p of this.parameters()) p.requires_grad = false;
  }

  /** Zero grads on all parameters. Call before each backward (accumulation invariant). */
  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }

  abstract forward(x: Tensor): Tensor;
}

/**
 * Linear: y = x @ Wᵀ + b, with W shape (out, in) (PyTorch convention).
 * WHY Wᵀ convention: it makes W.shape readable as (out_features, in_features) and matches
 *   how LoRA papers state ΔW shape, so stage code lines up with the math.
 */
export class Linear extends Module {
  W: Tensor; // (out, in)
  b: Tensor; // (1, out) — row-broadcast bias
  constructor(inDim: number, outDim: number, bias = true) {
    super();
    const std = xavierStd(inDim, outDim);
    const w = new Float64Array(outDim * inDim);
    for (let i = 0; i < w.length; i++) w[i] = normal(0, std);
    this.W = new Tensor(w, [outDim, inDim], true, [], "Linear.W");
    this.b = new Tensor(new Float64Array(outDim), [1, outDim], bias, [], "Linear.b");
  }
  override forward(x: Tensor): Tensor {
    // x: (m, in) ; Wᵀ: (in, out) -> (m, out) ; + bias broadcast
    return x.matmul(this.W.transpose()).add(this.b);
  }
}

/**
 * Embedding: integer ids -> rows of a table. Forward gathers rows; backward scatters grad
 *   back into the gathered rows only (the rest stay zero).
 * WHY a dedicated op rather than one-hot @ table: at toy scale either works, but gather
 *   makes the "only touched rows get grad" behavior explicit, which mirrors how prefix/
 *   prompt tuning touches only a few virtual-token rows.
 */
export class Embedding extends Module {
  table: Tensor; // (vocab, d)
  constructor(vocab: number, d: number) {
    super();
    const t = new Float64Array(vocab * d);
    const std = kaimingStd(d);
    for (let i = 0; i < t.length; i++) t[i] = normal(0, std);
    this.table = new Tensor(t, [vocab, d], true, [], "Embedding.table");
  }
  /** ids: number[] of length seq -> (seq, d). */
  lookup(ids: number[]): Tensor {
    const [, d] = this.table.shape;
    const out = new Tensor(new Float64Array(ids.length * d), [ids.length, d], this.table.requires_grad, [this.table], "embed");
    for (let i = 0; i < ids.length; i++) {
      const row = ids[i] * d;
      for (let j = 0; j < d; j++) out.data[i * d + j] = this.table.data[row + j];
    }
    out._backward = () => {
      if (!this.table.requires_grad) return;
      for (let i = 0; i < ids.length; i++) {
        const row = ids[i] * d;
        for (let j = 0; j < d; j++) this.table.grad[row + j] += out.grad[i * d + j];
      }
    };
    return out;
  }
  override forward(_x: Tensor): Tensor {
    throw new Error("Embedding: use lookup(ids), not forward(Tensor)");
  }
}

/**
 * LayerNorm over the last axis (per row). Normalizes then applies learnable gain/bias.
 * WHY LN matters for PEFT: it is a tiny-param layer that adapter/prefix methods often
 *   leave trainable; isolating it as a Module lets stages choose to freeze it or not.
 * INVARIANT: eps inside the sqrt prevents div-by-zero on constant rows.
 */
export class LayerNorm extends Module {
  gamma: Tensor; // (1, d)
  beta: Tensor; // (1, d)
  eps: number;
  constructor(d: number, eps = 1e-5) {
    super();
    this.gamma = new Tensor(new Float64Array(d).fill(1), [1, d], true, [], "LN.gamma");
    this.beta = new Tensor(new Float64Array(d), [1, d], true, [], "LN.beta");
    this.eps = eps;
  }
  override forward(x: Tensor): Tensor {
    // Implemented as a custom node so the (mean/var) backward stays exact & auditable.
    const [m, d] = x.shape;
    const out = new Tensor(new Float64Array(x.size), x.shape, x.requires_grad || this.gamma.requires_grad || this.beta.requires_grad, [x, this.gamma, this.beta], "layernorm");
    const xhat = new Float64Array(x.size); // cache normalized values for backward
    const invstd = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let mean = 0;
      for (let j = 0; j < d; j++) mean += x.data[i * d + j];
      mean /= d;
      let varr = 0;
      for (let j = 0; j < d; j++) {
        const c = x.data[i * d + j] - mean;
        varr += c * c;
      }
      varr /= d;
      const is = 1 / Math.sqrt(varr + this.eps);
      invstd[i] = is;
      for (let j = 0; j < d; j++) {
        const xh = (x.data[i * d + j] - mean) * is;
        xhat[i * d + j] = xh;
        out.data[i * d + j] = xh * this.gamma.data[j] + this.beta.data[j];
      }
    }
    out._backward = () => {
      for (let i = 0; i < m; i++) {
        // grads wrt gamma/beta (summed over rows)
        for (let j = 0; j < d; j++) {
          const g = out.grad[i * d + j];
          if (this.gamma.requires_grad) this.gamma.grad[j] += g * xhat[i * d + j];
          if (this.beta.requires_grad) this.beta.grad[j] += g;
        }
        if (!x.requires_grad) continue;
        // grad wrt input: standard LN backward (see Ba et al.); derived per row.
        let dxhatDot = 0; // sum_j dxhat_j
        let dxhatXhatDot = 0; // sum_j dxhat_j * xhat_j
        for (let j = 0; j < d; j++) {
          const dxhat = out.grad[i * d + j] * this.gamma.data[j];
          dxhatDot += dxhat;
          dxhatXhatDot += dxhat * xhat[i * d + j];
        }
        for (let j = 0; j < d; j++) {
          const dxhat = out.grad[i * d + j] * this.gamma.data[j];
          x.grad[i * d + j] += (invstd[i] / d) * (d * dxhat - dxhatDot - xhat[i * d + j] * dxhatXhatDot);
        }
      }
    };
    return out;
  }
}

/** Sequential chain of single-input/single-output modules. */
export class Sequential extends Module {
  layers: Module[];
  constructor(...layers: Module[]) {
    super();
    this.layers = layers;
  }
  override forward(x: Tensor): Tensor {
    let h = x;
    for (const l of this.layers) h = l.forward(h);
    return h;
  }
}

/**
 * Multi-head self-attention over a (seq, d_model) input.
 * WHY this is the PEFT injection site: LoRA/adapters most commonly attach to the q/v
 *   projections here. Keeping q/k/v/o as separate Linear modules means a stage can wrap
 *   exactly Wq and Wv with a LoRA delta and leave the rest frozen — the realistic recipe.
 * INVARIANT: causal masking is OFF by default (these toy tasks are not autoregressive
 *   left-to-right in a way that needs it); turn it on per-stage if modeling LM.
 */
export class MultiHeadAttention extends Module {
  q: Linear;
  k: Linear;
  v: Linear;
  o: Linear;
  heads: number;
  dHead: number;
  constructor(dModel: number, heads: number) {
    super();
    if (dModel % heads !== 0) throw new Error("MHA: dModel must be divisible by heads");
    this.heads = heads;
    this.dHead = dModel / heads;
    this.q = new Linear(dModel, dModel);
    this.k = new Linear(dModel, dModel);
    this.v = new Linear(dModel, dModel);
    this.o = new Linear(dModel, dModel);
  }
  override forward(x: Tensor): Tensor {
    const [seq, dModel] = x.shape;
    const Q = this.q.forward(x); // (seq, dModel)
    const K = this.k.forward(x);
    const V = this.v.forward(x);
    const scale = 1 / Math.sqrt(this.dHead);
    // Process each head by slicing its block out of the d_model axis, running scaled
    // dot-product attention, then concatenating the per-head contexts back to (seq,dModel).
    // Explicit slicing (vs reshaping to a 3-D (head,seq,dHead) view) keeps the engine 2-D.
    const heads: Tensor[] = [];
    for (let h = 0; h < this.heads; h++) {
      const Qh = sliceCols(Q, h * this.dHead, this.dHead); // (seq, dHead)
      const Kh = sliceCols(K, h * this.dHead, this.dHead);
      const Vh = sliceCols(V, h * this.dHead, this.dHead);
      const scores = Qh.matmul(Kh.transpose()).scale(scale); // (seq, seq)
      const attn = scores.softmax(); // (seq, seq)
      const ctx = attn.matmul(Vh); // (seq, dHead)
      heads.push(ctx);
    }
    const concat = concatCols(heads, [seq, dModel]); // (seq, dModel)
    return this.o.forward(concat);
  }
}

/** Slice a contiguous block of columns [start, start+width) from a 2-D tensor, with grad. */
function sliceCols(x: Tensor, start: number, width: number): Tensor {
  const [m, n] = x.shape;
  const out = new Tensor(new Float64Array(m * width), [m, width], x.requires_grad, [x], "slice");
  for (let i = 0; i < m; i++) for (let j = 0; j < width; j++) out.data[i * width + j] = x.data[i * n + start + j];
  out._backward = () => {
    if (!x.requires_grad) return;
    for (let i = 0; i < m; i++) for (let j = 0; j < width; j++) x.grad[i * n + start + j] += out.grad[i * width + j];
  };
  return out;
}

/** Concatenate equal-height tensors along the column axis, with grad routed to each part. */
function concatCols(parts: Tensor[], outShape: number[]): Tensor {
  const [m, n] = outShape;
  const out = new Tensor(new Float64Array(m * n), outShape, parts.some((p) => p.requires_grad), parts, "concat");
  let off = 0;
  const offsets: number[] = [];
  for (const p of parts) {
    offsets.push(off);
    const w = p.shape[1];
    for (let i = 0; i < m; i++) for (let j = 0; j < w; j++) out.data[i * n + off + j] = p.data[i * w + j];
    off += w;
  }
  out._backward = () => {
    parts.forEach((p, idx) => {
      if (!p.requires_grad) return;
      const w = p.shape[1];
      const o = offsets[idx];
      for (let i = 0; i < m; i++) for (let j = 0; j < w; j++) p.grad[i * w + j] += out.grad[i * n + o + j];
    });
  };
  return out;
}

/**
 * Toy Transformer block: x -> LN -> MHA -> +residual -> LN -> FFN(gelu) -> +residual.
 * WHY pre-LN: more stable for tiny depths and matches modern GPT blocks.
 * Default toy dims (d_model=32, heads=2, ffn=64, seq<=16) keep param count in the low
 *   thousands so a stage trains in seconds yet the trainable-ratio arithmetic is real.
 */
export class TransformerBlock extends Module {
  ln1: LayerNorm;
  attn: MultiHeadAttention;
  ln2: LayerNorm;
  ff1: Linear;
  ff2: Linear;
  constructor(dModel = 32, heads = 2, dFF = 64) {
    super();
    this.ln1 = new LayerNorm(dModel);
    this.attn = new MultiHeadAttention(dModel, heads);
    this.ln2 = new LayerNorm(dModel);
    this.ff1 = new Linear(dModel, dFF);
    this.ff2 = new Linear(dFF, dModel);
  }
  override forward(x: Tensor): Tensor {
    const a = this.attn.forward(this.ln1.forward(x)).add(x); // residual
    const f = this.ff2.forward(this.ff1.forward(this.ln2.forward(a)).gelu()).add(a); // residual
    return f;
  }
}
