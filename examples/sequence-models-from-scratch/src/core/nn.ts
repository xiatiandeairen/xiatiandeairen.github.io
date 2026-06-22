// core/nn.ts — Layer primitives + parameter management on top of tensor.ts.
//
// WHY only primitives live here (Linear/Embedding/LayerNorm) and NOT the recurrent
//   cells: RNNCell/LSTMCell/GRUCell/Attention/SSMBlock are the PEDAGOGICAL payload of
//   their stages. Each stage builds its cell from these primitives in its own file and
//   exports it, so later stages import the cell from the stage that taught it. Putting
//   the cells here would hide the very thing each chapter is about.
//
// CORE INVARIANT — params() is the contract with optim.ts:
//   Anything returned by params() will be updated by the optimizer and zeroed each step.
//   A Tensor created inside forward() (an activation) MUST NOT appear in params() — only
//   persistent learnable leaves do. Leaking an activation into params() = updating a
//   throwaway tensor and never updating the real weight.
//
// FAILURE MODE init guards against: zero-init weight matrices make every hidden unit
//   identical (symmetry never breaks, the net can't learn). All weights get random
//   init; only biases start at zero.

import { Tensor } from "./tensor.js";
import type { Rng } from "./prng.js";

export abstract class Module {
  /** All learnable leaves. Subclasses override; composites concat children's params. */
  abstract params(): Tensor[];
  /** Zero every param's grad. Call BEFORE each backward (grads accumulate, see tensor.ts). */
  zeroGrad(): void {
    for (const p of this.params()) p.zeroGrad();
  }
}

export type InitKind = "xavier" | "orthogonal" | "kaiming";

// Standard deviation for a given init scheme (orthogonal handled specially below).
function initStd(kind: InitKind, fanIn: number, fanOut: number): number {
  switch (kind) {
    case "xavier":
      return Math.sqrt(2 / (fanIn + fanOut)); // tanh/linear: balances fwd & bwd variance
    case "kaiming":
      return Math.sqrt(2 / fanIn); // ReLU: compensates the half of inputs it zeros
    case "orthogonal":
      return 1; // overwritten by the orthogonalization pass
  }
}

/**
 * Build a [rows, cols] weight matrix with the requested init.
 * WHY orthogonal matters in THIS book: an orthogonal recurrent matrix has all singular
 *   values = 1, so repeated multiplication (the RNN unroll) neither shrinks nor grows
 *   the hidden state norm — the cleanest defense against vanishing/exploding gradients
 *   that stage02 demonstrates. We approximate it via Gram-Schmidt on a Gaussian matrix.
 */
function initMatrix(rows: number, cols: number, kind: InitKind, rng: Rng): Float64Array {
  const data = new Float64Array(rows * cols);
  if (kind !== "orthogonal") {
    const std = initStd(kind, cols, rows); // fanIn=cols, fanOut=rows for y = x W^T convention
    for (let i = 0; i < data.length; i++) data[i] = rng.normal() * std;
    return data;
  }
  // Orthogonal: fill Gaussian, then Gram-Schmidt the rows (works for square or tall).
  // FAILURE MODE: rows > cols can't be fully orthogonal (more vectors than dims); we
  //   orthogonalize as far as possible — fine for square recurrent matrices, which is
  //   the case that matters here.
  for (let i = 0; i < data.length; i++) data[i] = rng.normal();
  const row = (r: number) => data.subarray(r * cols, r * cols + cols);
  for (let r = 0; r < rows; r++) {
    const v = row(r);
    for (let q = 0; q < r; q++) {
      const u = row(q);
      let dot = 0;
      for (let c = 0; c < cols; c++) dot += v[c] * u[c];
      for (let c = 0; c < cols; c++) v[c] -= dot * u[c];
    }
    let norm = 0;
    for (let c = 0; c < cols; c++) norm += v[c] * v[c];
    norm = Math.sqrt(norm) || 1;
    for (let c = 0; c < cols; c++) v[c] /= norm;
  }
  return data;
}

/**
 * Linear layer: y = x @ W + b, with W shaped [inDim, outDim].
 * INVARIANT: input x is 2D [batch, inDim]; b broadcasts over the batch row axis.
 *   (Stages with a time axis reshape [T,B,D] -> [T*B,D] before calling.)
 */
export class Linear extends Module {
  W: Tensor;
  b: Tensor | null;
  constructor(inDim: number, outDim: number, rng: Rng, opts: { init?: InitKind; bias?: boolean } = {}) {
    super();
    const init = opts.init ?? "xavier";
    this.W = new Tensor(initMatrix(inDim, outDim, init, rng), [inDim, outDim]);
    this.b = opts.bias === false ? null : Tensor.zeros([1, outDim]);
  }
  forward(x: Tensor): Tensor {
    const y = x.matmul(this.W);
    return this.b ? y.add(this.b) : y; // b [1,out] broadcasts over batch rows
  }
  override params(): Tensor[] {
    return this.b ? [this.W, this.b] : [this.W];
  }
}

/**
 * Embedding: integer token id -> learnable vector. Forward is a gather of rows;
 * backward is a SCATTER-ADD (a token appearing k times in the batch accumulates k
 * gradient contributions into its one row). Getting scatter-add wrong (overwrite, or
 * forgetting repeats) is the classic embedding bug.
 */
export class Embedding extends Module {
  weight: Tensor; // [vocab, dim]
  constructor(vocab: number, dim: number, rng: Rng, scale = 0.1) {
    super();
    // Small scale: embeddings feed directly into the model; large init destabilizes early training.
    this.weight = Tensor.randn([vocab, dim], rng, scale);
  }
  /** ids: flat list of token indices, length = rows. Returns [rows, dim]. */
  forward(ids: number[]): Tensor {
    const [vocab, dim] = this.weight.shape;
    const out = new Float64Array(ids.length * dim);
    for (let r = 0; r < ids.length; r++) {
      const id = ids[r];
      if (id < 0 || id >= vocab) throw new Error(`Embedding id ${id} out of [0,${vocab})`);
      out.set(this.weight.data.subarray(id * dim, id * dim + dim), r * dim);
    }
    const t = new Tensor(out, [ids.length, dim], [this.weight], "embed");
    t._backward = () => {
      for (let r = 0; r < ids.length; r++) {
        const id = ids[r];
        const woff = id * dim;
        const goff = r * dim;
        for (let c = 0; c < dim; c++) this.weight.grad[woff + c] += t.grad[goff + c]; // scatter-ADD
      }
    };
    return t;
  }
  override params(): Tensor[] {
    return [this.weight];
  }
}

/**
 * LayerNorm over the last dim: normalize each row to zero mean / unit var, then scale
 * (gamma) and shift (beta). WHY in a sequence book: it is the stabilizer that makes
 * deep attention/SSM stacks trainable; we keep a hand-written backward so the chapter
 * can show that the normalization's Jacobian (the (1 - 1/N - x̂_i x̂/N) term) is real,
 * not free.
 */
export class LayerNorm extends Module {
  gamma: Tensor; // [1, dim]
  beta: Tensor; // [1, dim]
  eps: number;
  constructor(dim: number, eps = 1e-5) {
    super();
    this.gamma = Tensor.ones([1, dim]);
    this.beta = Tensor.zeros([1, dim]);
    this.eps = eps;
  }
  forward(x: Tensor): Tensor {
    if (x.shape.length !== 2) throw new Error("LayerNorm: expects 2D [rows, dim]");
    const [rows, dim] = x.shape;
    const out = new Float64Array(rows * dim);
    const norm = new Float64Array(rows * dim); // cache x̂ for backward
    const invStd = new Float64Array(rows);
    const g = this.gamma.data;
    const b = this.beta.data;
    for (let r = 0; r < rows; r++) {
      const off = r * dim;
      let mean = 0;
      for (let c = 0; c < dim; c++) mean += x.data[off + c];
      mean /= dim;
      let varr = 0;
      for (let c = 0; c < dim; c++) {
        const d = x.data[off + c] - mean;
        varr += d * d;
      }
      varr /= dim;
      const inv = 1 / Math.sqrt(varr + this.eps);
      invStd[r] = inv;
      for (let c = 0; c < dim; c++) {
        const xn = (x.data[off + c] - mean) * inv;
        norm[off + c] = xn;
        out[off + c] = xn * g[c] + b[c];
      }
    }
    const t = new Tensor(out, x.shape.slice(), [x, this.gamma, this.beta], "layernorm");
    t._backward = () => {
      for (let r = 0; r < rows; r++) {
        const off = r * dim;
        const inv = invStd[r];
        // accumulate gamma/beta grads (shared across rows -> broadcast adjoint = sum)
        let sumDy = 0; // sum over c of dy * gamma
        let sumDyXn = 0; // sum over c of dy * gamma * x̂
        for (let c = 0; c < dim; c++) {
          const dy = t.grad[off + c];
          this.beta.grad[c] += dy;
          this.gamma.grad[c] += dy * norm[off + c];
          const dyg = dy * g[c];
          sumDy += dyg;
          sumDyXn += dyg * norm[off + c];
        }
        for (let c = 0; c < dim; c++) {
          const dyg = t.grad[off + c] * g[c];
          // dx_i = inv/N * (N*dyg - sumDy - x̂_i * sumDyXn)
          x.grad[off + c] += (inv / dim) * (dim * dyg - sumDy - norm[off + c] * sumDyXn);
        }
      }
    };
    return t;
  }
  override params(): Tensor[] {
    return [this.gamma, this.beta];
  }
}

/** Collect params from a heterogeneous list of modules (composite helper for cells). */
export function collectParams(modules: Module[]): Tensor[] {
  const out: Tensor[] = [];
  for (const m of modules) out.push(...m.params());
  return out;
}
