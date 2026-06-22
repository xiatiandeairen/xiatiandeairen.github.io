// core/tensor.ts — Minimal reverse-mode autodiff over small n-d tensors.
//
// WHY this engine exists and is frozen early: every PEFT mechanism in this book (LoRA's
//   BA decomposition, Adapter bottlenecks, prefix KV, QLoRA's dequant path) is expressed
//   as a tiny graph of the operators below. If this contract shifts, every downstream
//   "honest number" silently changes. So it is built once, grad-checked, then trusted.
//
// THE ONE IDEA (reverse-mode autodiff): each Tensor caches a `_backward` closure that,
//   given its own grad, scatters grad into its parents. backward() walks the graph in
//   reverse-topological order so a node's grad is fully accumulated before it is consumed.
//
// CORE INVARIANT — grads ACCUMULATE (+=), never overwrite (=):
//   A parameter feeds many consumers (a frozen base weight is read by LoRA AND the
//   residual path; broadcasting fans one row into many rows). Reverse-mode must SUM every
//   path's contribution. Overwriting keeps only the last path and silently corrupts
//   training. Corollary: callers MUST zeroGrad() between optimizer steps.
//
// PEFT-CRITICAL INVARIANT — requires_grad gates the backward closure's WRITES, not reads:
//   A frozen base weight (requires_grad=false) still participates in the forward pass and
//   still propagates grad to ITS inputs, but its OWN .grad must stay zero so the optimizer
//   never moves it. We enforce this where grad is written into a leaf: skip the write if
//   the leaf does not require grad. This is exactly what makes PEFT cheap — 99%+ of the
//   weights are frozen leaves whose grad we never even compute storage for.
//
// FAILURE MODE this design defends against: forgetting broadcast has a real adjoint. The
//   adjoint of "broadcast a (1,n) bias across (m,n)" is "sum grad back down the broadcast
//   axis". Skip that sum and you get wrong-magnitude grads — numericalGradCheck catches it.
//
// SCOPE: tensors are 1-D or 2-D only. That covers every layer in a toy Transformer
//   (tokens are (seq, d_model) matrices) and keeps the backward math auditable by hand.
//   Higher rank is deliberately out of scope — it would add stride bookkeeping that buys
//   no pedagogical insight for PEFT.

import { uniform } from "./prng.js";

type Backward = () => void;

function prod(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function shapeEq(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export class Tensor {
  data: Float64Array;
  grad: Float64Array;
  shape: number[];
  // requires_grad=false means: this is a FROZEN leaf. Its grad buffer exists (so shared
  //   code paths don't branch) but backward() never writes into it. See header invariant.
  requires_grad: boolean;

  // Graph bookkeeping. _prev are parents (for topo discovery), _backward scatters grad.
  _prev: Tensor[];
  _backward: Backward;
  op: string; // label, debugging only

  constructor(
    data: Float64Array | number[],
    shape: number[],
    requires_grad = false,
    prev: Tensor[] = [],
    op = "",
  ) {
    const flat = data instanceof Float64Array ? data : Float64Array.from(data);
    if (flat.length !== prod(shape)) {
      // Loud failure: a shape/data mismatch here would otherwise corrupt every op silently.
      throw new Error(`Tensor: data length ${flat.length} != prod(shape) ${prod(shape)} for shape [${shape}]`);
    }
    this.data = flat;
    this.grad = new Float64Array(flat.length); // starts at 0; accumulated during backward
    this.shape = shape.slice();
    this.requires_grad = requires_grad;
    this._prev = prev;
    this._backward = () => {};
    this.op = op;
  }

  get size(): number {
    return this.data.length;
  }

  /** Zero this node's grad. Optimizers/training loops call this on every trainable param. */
  zeroGrad(): void {
    this.grad.fill(0);
  }

  // --------------------------------------------------------------------------
  // Leaf constructors
  // --------------------------------------------------------------------------

  static zeros(shape: number[], requires_grad = false): Tensor {
    return new Tensor(new Float64Array(prod(shape)), shape, requires_grad, [], "zeros");
  }

  static from(data: number[], shape: number[], requires_grad = false): Tensor {
    return new Tensor(data, shape, requires_grad, [], "from");
  }

  // --------------------------------------------------------------------------
  // Graph plumbing helpers
  // --------------------------------------------------------------------------

  /**
   * An output node must carry grad iff ANY input does. This is the fix for the subtle bug
   *   where intermediate results default to requires_grad=false and silently block grad
   *   from reaching trainable leaves several ops upstream. The PER-LEAF write guards in
   *   each _backward still gate writes to FROZEN leaves (those keep requires_grad=false),
   *   so frozen base weights stay at zero grad while intermediates correctly propagate.
   */
  private static needsGrad(...inputs: Tensor[]): boolean {
    return inputs.some((t) => t.requires_grad);
  }

  /** True when `small` (shape (1,n) or (n,)) row-broadcasts across `full` (m,n). */
  private _isRowBroadcast(full: number[], small: number[]): boolean {
    if (full.length !== 2) return false;
    const cols = full[1];
    if (small.length === 1) return small[0] === cols;
    if (small.length === 2) return small[0] === 1 && small[1] === cols;
    return false;
  }

  // --------------------------------------------------------------------------
  // Elementwise binary ops (with row-broadcast for bias-style (1,n) over (m,n))
  // --------------------------------------------------------------------------

  /**
   * this + other. Supports exact-shape, and row-broadcast where `other` is (1, n) or (n,)
   *   added to every row of a (m, n) `this`. Row-broadcast is the bias pattern.
   * ADJOINT of broadcast: sum the upstream grad back down the broadcast axis (see header).
   */
  add(other: Tensor): Tensor {
    const [a, b] = [this, other];
    const broadcast = !shapeEq(a.shape, b.shape) && this._isRowBroadcast(a.shape, b.shape);
    if (!shapeEq(a.shape, b.shape) && !broadcast) {
      throw new Error(`add: incompatible shapes [${a.shape}] + [${b.shape}]`);
    }
    const out = new Tensor(new Float64Array(a.size), a.shape, Tensor.needsGrad(a, b), [a, b], "+");
    const cols = a.shape[a.shape.length - 1];
    for (let i = 0; i < a.size; i++) {
      out.data[i] = a.data[i] + (broadcast ? b.data[i % cols] : b.data[i]);
    }
    out._backward = () => {
      if (a.requires_grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i];
      if (b.requires_grad) {
        if (broadcast) {
          // sum grad down the broadcast (row) axis: every output row contributed.
          for (let i = 0; i < a.size; i++) b.grad[i % cols] += out.grad[i];
        } else {
          for (let i = 0; i < b.size; i++) b.grad[i] += out.grad[i];
        }
      }
    };
    return out;
  }

  /** this - other (exact shape only; subtraction broadcasting is unused in this book). */
  sub(other: Tensor): Tensor {
    if (!shapeEq(this.shape, other.shape)) throw new Error(`sub: shape mismatch [${this.shape}] vs [${other.shape}]`);
    const out = new Tensor(new Float64Array(this.size), this.shape, Tensor.needsGrad(this, other), [this, other], "-");
    for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] - other.data[i];
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < this.size; i++) this.grad[i] += out.grad[i];
      if (other.requires_grad) for (let i = 0; i < this.size; i++) other.grad[i] -= out.grad[i];
    };
    return out;
  }

  /** Elementwise this * other (exact shape). d/da = b, d/db = a. */
  mul(other: Tensor): Tensor {
    if (!shapeEq(this.shape, other.shape)) throw new Error(`mul: shape mismatch [${this.shape}] vs [${other.shape}]`);
    const out = new Tensor(new Float64Array(this.size), this.shape, Tensor.needsGrad(this, other), [this, other], "*");
    for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] * other.data[i];
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < this.size; i++) this.grad[i] += other.data[i] * out.grad[i];
      if (other.requires_grad) for (let i = 0; i < this.size; i++) other.grad[i] += this.data[i] * out.grad[i];
    };
    return out;
  }

  /** Scale by a python-side scalar constant. WHY a dedicated op: LoRA's α/r scaling and
   *   QLoRA's dequant blockwise scales are scalar multiplies whose grad is just scaled. */
  scale(k: number): Tensor {
    const out = new Tensor(new Float64Array(this.size), this.shape, Tensor.needsGrad(this), [this], `scale(${k})`);
    for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] * k;
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < this.size; i++) this.grad[i] += k * out.grad[i];
    };
    return out;
  }

  // --------------------------------------------------------------------------
  // Matmul — the workhorse. (m,k) @ (k,n) -> (m,n).
  // --------------------------------------------------------------------------

  /**
   * 2-D matmul. ADJOINTS: dA = dC @ Bᵀ ; dB = Aᵀ @ dC. These two transposed products are
   *   the entire reason LoRA's two skinny matrices get correct grads "for free" — B and A
   *   are just two matmul leaves and the chain rule routes grad through both.
   */
  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) throw new Error("matmul: both operands must be 2-D");
    const [m, k] = this.shape;
    const [k2, n] = other.shape;
    if (k !== k2) throw new Error(`matmul: inner dims disagree ${k} vs ${k2}`);
    const out = new Tensor(new Float64Array(m * n), [m, n], Tensor.needsGrad(this, other), [this, other], "matmul");
    const A = this.data, B = other.data, C = out.data;
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const aip = A[i * k + p];
        if (aip === 0) continue; // tiny speedup; also keeps zero-init B rows cheap (LoRA B=0)
        for (let j = 0; j < n; j++) C[i * n + j] += aip * B[p * n + j];
      }
    }
    out._backward = () => {
      const dC = out.grad;
      if (this.requires_grad) {
        // dA[i,p] = sum_j dC[i,j] * B[p,j]
        for (let i = 0; i < m; i++)
          for (let p = 0; p < k; p++) {
            let s = 0;
            for (let j = 0; j < n; j++) s += dC[i * n + j] * B[p * n + j];
            this.grad[i * k + p] += s;
          }
      }
      if (other.requires_grad) {
        // dB[p,j] = sum_i A[i,p] * dC[i,j]
        for (let p = 0; p < k; p++)
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let i = 0; i < m; i++) s += A[i * k + p] * dC[i * n + j];
            other.grad[p * n + j] += s;
          }
      }
    };
    return out;
  }

  /** Transpose a 2-D tensor. Adjoint of transpose is transpose. */
  transpose(): Tensor {
    if (this.shape.length !== 2) throw new Error("transpose: 2-D only");
    const [m, n] = this.shape;
    const out = new Tensor(new Float64Array(this.size), [n, m], Tensor.needsGrad(this), [this], "T");
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out.data[j * m + i] = this.data[i * n + j];
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) this.grad[i * n + j] += out.grad[j * m + i];
    };
    return out;
  }

  /** Reshape (view with same element count). Grad just flows back unreshaped. */
  reshape(shape: number[]): Tensor {
    if (prod(shape) !== this.size) throw new Error(`reshape: ${prod(shape)} != ${this.size}`);
    const out = new Tensor(this.data.slice(), shape, Tensor.needsGrad(this), [this], "reshape");
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < this.size; i++) this.grad[i] += out.grad[i];
    };
    return out;
  }

  // --------------------------------------------------------------------------
  // Reductions
  // --------------------------------------------------------------------------

  /** Sum all elements to a scalar (1-element tensor). Adjoint broadcasts grad to all. */
  sum(): Tensor {
    let s = 0;
    for (let i = 0; i < this.size; i++) s += this.data[i];
    const out = new Tensor([s], [1], Tensor.needsGrad(this), [this], "sum");
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < this.size; i++) this.grad[i] += out.grad[0];
    };
    return out;
  }

  /** Mean of all elements. Adjoint broadcasts grad/N. */
  mean(): Tensor {
    const n = this.size;
    let s = 0;
    for (let i = 0; i < n; i++) s += this.data[i];
    const out = new Tensor([s / n], [1], Tensor.needsGrad(this), [this], "mean");
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < n; i++) this.grad[i] += out.grad[0] / n;
    };
    return out;
  }

  // --------------------------------------------------------------------------
  // Nonlinearities
  // --------------------------------------------------------------------------

  /** ReLU. Adjoint passes grad only where input was positive. */
  relu(): Tensor {
    const out = new Tensor(new Float64Array(this.size), this.shape, Tensor.needsGrad(this), [this], "relu");
    for (let i = 0; i < this.size; i++) out.data[i] = this.data[i] > 0 ? this.data[i] : 0;
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < this.size; i++) this.grad[i] += (this.data[i] > 0 ? 1 : 0) * out.grad[i];
    };
    return out;
  }

  /**
   * GELU (tanh approximation). WHY tanh-approx not erf: it is the form used by GPT-2 and
   *   has a closed-form derivative we can write and grad-check by hand.
   */
  gelu(): Tensor {
    const out = new Tensor(new Float64Array(this.size), this.shape, Tensor.needsGrad(this), [this], "gelu");
    const c = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < this.size; i++) {
      const x = this.data[i];
      out.data[i] = 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    }
    out._backward = () => {
      if (!this.requires_grad) return;
      for (let i = 0; i < this.size; i++) {
        const x = this.data[i];
        const inner = c * (x + 0.044715 * x * x * x);
        const t = Math.tanh(inner);
        const dInner = c * (1 + 3 * 0.044715 * x * x);
        // d/dx [0.5 x (1+tanh(inner))] = 0.5(1+t) + 0.5 x (1 - t^2) dInner
        const dgelu = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner;
        this.grad[i] += dgelu * out.grad[i];
      }
    };
    return out;
  }

  /**
   * Row-wise softmax over the last axis of a 2-D tensor (each row sums to 1).
   * INVARIANT: subtract row max before exp (numerical stability) — without it, large
   *   logits overflow to Inf and the whole row becomes NaN.
   * ADJOINT (per row): dx_i = y_i * (dy_i - sum_j y_j dy_j). The coupling term is why
   *   softmax grad is not elementwise — every input affects every output in the row.
   */
  softmax(): Tensor {
    if (this.shape.length !== 2) throw new Error("softmax: 2-D only (row-wise)");
    const [m, n] = this.shape;
    const out = new Tensor(new Float64Array(this.size), this.shape, Tensor.needsGrad(this), [this], "softmax");
    for (let i = 0; i < m; i++) {
      let mx = -Infinity;
      for (let j = 0; j < n; j++) mx = Math.max(mx, this.data[i * n + j]);
      let sum = 0;
      for (let j = 0; j < n; j++) {
        const e = Math.exp(this.data[i * n + j] - mx);
        out.data[i * n + j] = e;
        sum += e;
      }
      for (let j = 0; j < n; j++) out.data[i * n + j] /= sum;
    }
    out._backward = () => {
      if (!this.requires_grad) return;
      for (let i = 0; i < m; i++) {
        let dot = 0;
        for (let j = 0; j < n; j++) dot += out.data[i * n + j] * out.grad[i * n + j];
        for (let j = 0; j < n; j++) {
          const y = out.data[i * n + j];
          this.grad[i * n + j] += y * (out.grad[i * n + j] - dot);
        }
      }
    };
    return out;
  }

  /**
   * Inverted dropout, deterministic via the global PRNG. p is the drop probability.
   * WHY inverted (scale survivors by 1/(1-p)): keeps expected activation magnitude
   *   constant so inference (no dropout) sees the same scale as training.
   * INVARIANT: when train=false this is identity (mask all-ones) — eval must be
   *   deterministic and full-strength. FAILURE MODE: applying dropout at eval makes
   *   the loss curve noisy and the merged-vs-unmerged equivalence check (stage08) fail.
   */
  dropout(p: number, train: boolean): Tensor {
    if (!train || p <= 0) {
      // identity, but keep it a graph node so callers can use it unconditionally
      const id = new Tensor(this.data.slice(), this.shape, Tensor.needsGrad(this), [this], "dropout(eval)");
      id._backward = () => {
        if (this.requires_grad) for (let i = 0; i < this.size; i++) this.grad[i] += id.grad[i];
      };
      return id;
    }
    const keep = 1 - p;
    const mask = new Float64Array(this.size);
    const out = new Tensor(new Float64Array(this.size), this.shape, Tensor.needsGrad(this), [this], "dropout");
    for (let i = 0; i < this.size; i++) {
      mask[i] = uniform() < keep ? 1 / keep : 0;
      out.data[i] = this.data[i] * mask[i];
    }
    out._backward = () => {
      if (this.requires_grad) for (let i = 0; i < this.size; i++) this.grad[i] += mask[i] * out.grad[i];
    };
    return out;
  }

  // --------------------------------------------------------------------------
  // Backward driver
  // --------------------------------------------------------------------------

  /**
   * Run reverse-mode autodiff from THIS node (must be scalar / 1-element).
   * Builds reverse-topological order then fires each node's _backward once.
   * INVARIANT: caller seeds this node's grad to 1 (we do it here) and must have zeroed
   *   all leaf grads beforehand — see header on accumulation.
   */
  backward(): void {
    if (this.size !== 1) throw new Error("backward: can only start from a scalar (1-element) tensor");
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const build = (t: Tensor): void => {
      if (visited.has(t)) return;
      visited.add(t);
      for (const p of t._prev) build(p);
      topo.push(t);
    };
    build(this);
    this.grad[0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

/**
 * numericalGradCheck — finite-difference verification of any scalar-output function of one
 *   input tensor. Returns the max relative error between analytic and numeric grad.
 *
 * WHY every stage can assert err < 1e-4: this is the book's safety net. A subtle adjoint
 *   bug (forgotten broadcast sum, wrong transpose in matmul backward) produces grads that
 *   look plausible but fail here. Central differences give O(h^2) accuracy, good to ~1e-6
 *   for f64 with h=1e-5, leaving wide margin under the 1e-4 assertion.
 *
 * FAILURE MODE it surfaces: if you wire LoRA's B/A grads wrong, this fires immediately
 *   instead of you debugging a wrong-but-converging loss curve for an hour.
 *
 * NOTE: requires `x.requires_grad = true` so analytic grad is actually populated.
 */
export function numericalGradCheck(
  f: (x: Tensor) => Tensor,
  x: Tensor,
  h = 1e-5,
): number {
  if (!x.requires_grad) throw new Error("numericalGradCheck: x.requires_grad must be true");
  // analytic
  x.zeroGrad();
  const y = f(x);
  if (y.size !== 1) throw new Error("numericalGradCheck: f must return a scalar tensor");
  y.backward();
  const analytic = x.grad.slice();
  // numeric (central difference), recomputing f fresh each perturbation
  const numeric = new Float64Array(x.size);
  for (let i = 0; i < x.size; i++) {
    const orig = x.data[i];
    x.data[i] = orig + h;
    const yp = f(x).data[0];
    x.data[i] = orig - h;
    const ym = f(x).data[0];
    x.data[i] = orig;
    numeric[i] = (yp - ym) / (2 * h);
  }
  // max relative error
  let maxRel = 0;
  for (let i = 0; i < x.size; i++) {
    const denom = Math.max(1e-8, Math.abs(analytic[i]) + Math.abs(numeric[i]));
    const rel = Math.abs(analytic[i] - numeric[i]) / denom;
    if (rel > maxRel) maxRel = rel;
  }
  return maxRel;
}
