// core/tensor.ts — A minimal Tensor with reverse-mode autograd (the book's whole engine).
//
// This is NOT a fast tensor lib. It exists so a diffusion model is ~200 lines of readable
// code with REAL gradients, not a PyTorch black box. If it were a framework you could not
// see WHY the score network learns ∇log p(x): the gradient that flows back from the
// (ε_pred - ε)² loss into the network weights is computed RIGHT HERE, by these closures,
// and you can read every step.
//
// SCOPE: scalars to small 2-D matrices. The book's models are <50K params on 2-D data, so
//   plain JS Float64Array is fast enough (a stage runs in seconds). No GPU, no SIMD, no
//   batched/strided tricks — clarity over speed.
//
// CORE INVARIANT — grads ACCUMULATE (+=), never overwrite (=):
//   A tensor can feed multiple consumers (a shared weight, a value broadcast into many
//   outputs). Reverse-mode requires SUMMING the contribution from every path. Overwriting
//   keeps only the last path and silently corrupts training. Corollary: callers MUST zero
//   grads between optimizer steps (see optim.zeroGrad).
//
// PRECISION: Float64 everywhere, deliberately. f32 is faster/more realistic but introduces
//   "is this a bug or just f32 noise?" ambiguity. f64 lets numericalGradCheck pass < 1e-5,
//   so a failing check is a REAL adjoint bug. Honesty cost: our wall-clock is pessimistic
//   vs a real f32/SIMD framework — the transferable signal is relative trends, not abs ms.

function computeStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

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
  strides: number[];
  // _backward: scatter THIS node's grad into its parents' grads. No-op for leaves.
  _backward: () => void;
  // _prev: parents, used only to discover the graph for topo sort.
  _prev: Tensor[];
  op: string; // debug label only

  constructor(data: Float64Array | number[], shape: number[], prev: Tensor[] = [], op = "") {
    const flat = data instanceof Float64Array ? data : Float64Array.from(data);
    if (flat.length !== prod(shape)) {
      // Catch shape/data mismatch early — turns a silent NaN cascade into a clear error.
      throw new Error(`Tensor: data length ${flat.length} != prod(shape ${shape}) = ${prod(shape)}`);
    }
    this.data = flat;
    this.grad = new Float64Array(flat.length); // grads start at 0; accumulate over paths
    this.shape = shape.slice();
    this.strides = computeStrides(shape);
    this._backward = () => {};
    this._prev = prev;
    this.op = op;
  }

  get size(): number {
    return this.data.length;
  }

  static zeros(shape: number[]): Tensor {
    return new Tensor(new Float64Array(prod(shape)), shape);
  }

  static fill(shape: number[], value: number): Tensor {
    const d = new Float64Array(prod(shape));
    d.fill(value);
    return new Tensor(d, shape);
  }

  /** Build from a generator. INVARIANT: gen() is called exactly prod(shape) times in
   *  row-major order — stages rely on this draw count for reproducible weight init. */
  static from(shape: number[], gen: () => number): Tensor {
    const n = prod(shape);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = gen();
    return new Tensor(d, shape);
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  // ============================ elementwise (broadcast) ============================
  // Supports (a) identical shapes and (b) `other` broadcastable INTO this shape by tiling
  // its flat buffer (other.size divides this.size). That covers row-bias and per-timestep
  // scalar broadcast, which is everything the book needs. The adjoint of broadcast is
  // SUM-over-tiled-positions — handled by += into b.grad[i % bsize] below.
  private static elementwise(
    a: Tensor,
    b: Tensor,
    fwd: (x: number, y: number) => number,
    back: (x: number, y: number, g: number) => [number, number],
    op: string,
  ): Tensor {
    const broadcasting = !shapeEq(a.shape, b.shape);
    if (broadcasting && (b.size > a.size || a.size % b.size !== 0)) {
      throw new Error(`elementwise(${op}): cannot broadcast ${b.shape} into ${a.shape}`);
    }
    const n = a.size;
    const bsize = b.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = fwd(a.data[i], b.data[broadcasting ? i % bsize : i]);
    const t = new Tensor(out, a.shape, [a, b], op);
    t._backward = () => {
      for (let i = 0; i < n; i++) {
        const bi = broadcasting ? i % bsize : i;
        const [da, db] = back(a.data[i], b.data[bi], t.grad[i]);
        a.grad[i] += da;
        // FAILURE MODE guarded: many i map to one bi when broadcasting, so we must SUM
        // their grads into b.grad[bi] (the broadcast adjoint). += does exactly that.
        b.grad[bi] += db;
      }
    };
    return t;
  }

  add(other: Tensor): Tensor {
    return Tensor.elementwise(this, other, (x, y) => x + y, (_x, _y, g) => [g, g], "+");
  }
  sub(other: Tensor): Tensor {
    return Tensor.elementwise(this, other, (x, y) => x - y, (_x, _y, g) => [g, -g], "-");
  }
  mul(other: Tensor): Tensor {
    return Tensor.elementwise(this, other, (x, y) => x * y, (x, y, g) => [g * y, g * x], "*");
  }
  div(other: Tensor): Tensor {
    // Guard adjoint denom: y==0 already yields Inf in forward; we don't paper over it.
    return Tensor.elementwise(this, other, (x, y) => x / y, (x, y, g) => [g / y, (-g * x) / (y * y)], "/");
  }

  /** Multiply by a plain scalar (no graph node for the scalar). Fast common path. */
  mulScalar(s: number): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.data[i] * s;
    const t = new Tensor(out, this.shape, [this], `*${s}`);
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[i] += s * t.grad[i];
    };
    return t;
  }

  addScalar(s: number): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.data[i] + s;
    const t = new Tensor(out, this.shape, [this], `+${s}`);
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[i] += t.grad[i];
    };
    return t;
  }

  // ============================ matmul (m,k)@(k,n) -> (m,n) ============================
  // WHY only 2-D: the book reshapes any batched matmul into 2-D, so the core op stays 2-D
  //   and its adjoint is the dead-simple chain rule: dA = dC @ B^T , dB = A^T @ dC.
  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(`matmul: both operands must be 2-D, got ${this.shape} @ ${other.shape}`);
    }
    const [m, k] = this.shape;
    const [k2, n] = other.shape;
    if (k !== k2) throw new Error(`matmul: inner dims mismatch ${k} != ${k2}`);
    const out = new Float64Array(m * n);
    const A = this.data;
    const B = other.data;
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const a = A[i * k + p];
        if (a === 0) continue; // sparsity shortcut; correctness-neutral
        const bRow = p * n;
        const oRow = i * n;
        for (let j = 0; j < n; j++) out[oRow + j] += a * B[bRow + j];
      }
    }
    const t = new Tensor(out, [m, n], [this, other], "matmul");
    t._backward = () => {
      const dC = t.grad;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          const g = dC[i * n + j];
          if (g === 0) continue;
          for (let p = 0; p < k; p++) {
            this.grad[i * k + p] += g * B[p * n + j]; // dA = dC @ B^T
            other.grad[p * n + j] += g * A[i * k + p]; // dB = A^T @ dC
          }
        }
      }
    };
    return t;
  }

  // ============================ unary nonlinearities ============================
  relu(): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.data[i] > 0 ? this.data[i] : 0;
    const t = new Tensor(out, this.shape, [this], "relu");
    // Subgradient at 0 taken as 0 (standard convention).
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[i] += (out[i] > 0 ? 1 : 0) * t.grad[i];
    };
    return t;
  }

  tanh(): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.tanh(this.data[i]);
    const t = new Tensor(out, this.shape, [this], "tanh");
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[i] += (1 - out[i] * out[i]) * t.grad[i];
    };
    return t;
  }

  sigmoid(): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    // Numerically stable logistic: avoid exp of large positive arg (overflow -> Inf).
    for (let i = 0; i < n; i++) {
      const x = this.data[i];
      out[i] = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
    }
    const t = new Tensor(out, this.shape, [this], "sigmoid");
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[i] += out[i] * (1 - out[i]) * t.grad[i];
    };
    return t;
  }

  /** SiLU / swish: x * sigmoid(x). The default activation for diffusion time-conditioned
   *  MLPs — smooth and non-monotone, empirically better than ReLU for these nets. */
  silu(): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    const sig = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const x = this.data[i];
      const s = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
      sig[i] = s;
      out[i] = x * s;
    }
    const t = new Tensor(out, this.shape, [this], "silu");
    // d/dx [x*σ(x)] = σ(x) + x*σ(x)*(1-σ(x)) = σ(x)*(1 + x*(1-σ(x)))
    t._backward = () => {
      for (let i = 0; i < n; i++) {
        const x = this.data[i];
        const s = sig[i];
        this.grad[i] += s * (1 + x * (1 - s)) * t.grad[i];
      }
    };
    return t;
  }

  exp(): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.exp(this.data[i]);
    const t = new Tensor(out, this.shape, [this], "exp");
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[i] += out[i] * t.grad[i];
    };
    return t;
  }

  log(): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.log(this.data[i]);
    const t = new Tensor(out, this.shape, [this], "log");
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[i] += (1 / this.data[i]) * t.grad[i];
    };
    return t;
  }

  sqrt(): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.sqrt(this.data[i]);
    const t = new Tensor(out, this.shape, [this], "sqrt");
    // d/dx sqrt(x) = 1/(2*sqrt(x)). FAILURE MODE: x==0 -> Inf grad; callers must keep
    // sqrt arguments strictly positive (e.g. variance + eps), never exactly 0.
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[i] += (0.5 / out[i]) * t.grad[i];
    };
    return t;
  }

  // ============================ reductions ============================
  /** Sum. axis omitted -> scalar [1]. axis given (0 or 1, 2-D only) -> reduce that axis,
   *  keeping the other (result is [1,cols] for axis 0, [rows,1] for axis 1). Axis reduction
   *  is what lets a per-row loss collapse to a scalar while keeping the broadcast adjoint
   *  in one place. */
  sum(axis?: number): Tensor {
    if (axis === undefined) {
      let s = 0;
      for (let i = 0; i < this.size; i++) s += this.data[i];
      const t = new Tensor([s], [1], [this], "sum");
      t._backward = () => {
        const g = t.grad[0];
        for (let i = 0; i < this.size; i++) this.grad[i] += g; // broadcast grad back
      };
      return t;
    }
    return this.reduceAxis(axis, "sum");
  }

  /** Mean. Same axis semantics as sum; adjoint divides the broadcast grad by the count. */
  mean(axis?: number): Tensor {
    if (axis === undefined) {
      const n = this.size;
      let s = 0;
      for (let i = 0; i < n; i++) s += this.data[i];
      const t = new Tensor([s / n], [1], [this], "mean");
      t._backward = () => {
        const g = t.grad[0] / n;
        for (let i = 0; i < n; i++) this.grad[i] += g;
      };
      return t;
    }
    return this.reduceAxis(axis, "mean");
  }

  private reduceAxis(axis: number, kind: "sum" | "mean"): Tensor {
    if (this.shape.length !== 2) throw new Error(`${kind}(axis): expected 2-D, got ${this.shape}`);
    if (axis !== 0 && axis !== 1) throw new Error(`${kind}(axis): axis must be 0 or 1, got ${axis}`);
    const [rows, cols] = this.shape;
    const outShape = axis === 0 ? [1, cols] : [rows, 1];
    const out = new Float64Array(prod(outShape));
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const oi = axis === 0 ? c : r;
        out[oi] += this.data[r * cols + c];
      }
    }
    const denom = kind === "mean" ? (axis === 0 ? rows : cols) : 1;
    if (kind === "mean") for (let i = 0; i < out.length; i++) out[i] /= denom;
    const t = new Tensor(out, outShape, [this], `${kind}@${axis}`);
    t._backward = () => {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const oi = axis === 0 ? c : r;
          this.grad[r * cols + c] += t.grad[oi] / denom; // each input contributed 1/denom
        }
      }
    };
    return t;
  }

  // ============================ shape ops ============================
  /** Reshape, same element count. Data buffer is shared (we never mutate data in place),
   *  grads flow straight through 1:1. */
  reshape(newShape: number[]): Tensor {
    if (prod(newShape) !== this.size) {
      throw new Error(`reshape: ${this.shape} -> ${newShape} changes element count`);
    }
    const t = new Tensor(this.data, newShape, [this], "reshape");
    t._backward = () => {
      for (let i = 0; i < this.size; i++) this.grad[i] += t.grad[i];
    };
    return t;
  }

  /** Transpose a 2-D tensor (materializes a fresh buffer). Adjoint transposes grad back. */
  transpose(): Tensor {
    if (this.shape.length !== 2) throw new Error(`transpose: expected 2-D, got ${this.shape}`);
    const [r, c] = this.shape;
    const out = new Float64Array(r * c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = this.data[i * c + j];
    const t = new Tensor(out, [c, r], [this], "transpose");
    t._backward = () => {
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) this.grad[i * c + j] += t.grad[j * r + i];
    };
    return t;
  }

  /** Concatenate 2-D tensors along axis 1 (columns). This is how the denoiser fuses
   *  [x ⊕ time-embed ⊕ cond] into one input row. INVARIANT: all parts share row count.
   *  Adjoint slices each part's grad back out of the concatenated grad. */
  static concatCols(parts: Tensor[]): Tensor {
    if (parts.length === 0) throw new Error("concatCols: need at least one tensor");
    const rows = parts[0].shape[0];
    for (const p of parts) {
      if (p.shape.length !== 2) throw new Error(`concatCols: expected 2-D, got ${p.shape}`);
      if (p.shape[0] !== rows) throw new Error(`concatCols: row mismatch ${p.shape[0]} != ${rows}`);
    }
    const totalCols = parts.reduce((a, p) => a + p.shape[1], 0);
    const out = new Float64Array(rows * totalCols);
    for (let r = 0; r < rows; r++) {
      let colOffset = 0;
      for (const p of parts) {
        const pc = p.shape[1];
        for (let c = 0; c < pc; c++) out[r * totalCols + colOffset + c] = p.data[r * pc + c];
        colOffset += pc;
      }
    }
    const t = new Tensor(out, [rows, totalCols], parts, "concatCols");
    t._backward = () => {
      for (let r = 0; r < rows; r++) {
        let colOffset = 0;
        for (const p of parts) {
          const pc = p.shape[1];
          for (let c = 0; c < pc; c++) p.grad[r * pc + c] += t.grad[r * totalCols + colOffset + c];
          colOffset += pc;
        }
      }
    };
    return t;
  }

  /** Slice rows [start, end) of a 2-D tensor (used to take minibatches with grad). */
  sliceRows(start: number, end: number): Tensor {
    if (this.shape.length !== 2) throw new Error(`sliceRows: expected 2-D, got ${this.shape}`);
    const cols = this.shape[1];
    if (start < 0 || end > this.shape[0] || start >= end) {
      throw new Error(`sliceRows: bad range [${start}, ${end}) for ${this.shape[0]} rows`);
    }
    const n = (end - start) * cols;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.data[start * cols + i];
    const t = new Tensor(out, [end - start, cols], [this], "sliceRows");
    t._backward = () => {
      for (let i = 0; i < n; i++) this.grad[start * cols + i] += t.grad[i];
    };
    return t;
  }

  // ============================ backward ============================
  /**
   * Reverse-topological backward. Builds topo order (parents before children), walks it
   * reversed so each consumer's grad is final before a node scatters into its parents.
   * ENFORCED: backward() may only start from a scalar loss (shape [1]) — d(vector)/d(?)
   * has no canonical seed. For a non-scalar output, set .grad yourself and drive topo.
   * The loss is always scalar, which is the only call site that matters.
   */
  backward(): void {
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const build = (v: Tensor) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);
    if (this.size !== 1) {
      throw new Error(`backward(): can only start from a scalar loss [1]; got ${this.shape}`);
    }
    this.grad[0] = 1; // d(loss)/d(loss) = 1
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

/**
 * Finite-difference gradient check. WHY it exists: an autograd adjoint bug is invisible —
 * the model still trains, just toward the wrong thing. This perturbs each input element by
 * ±eps, measures the numerical slope of a scalar fn, and compares to the analytic grad.
 *
 * Returns the max absolute error across elements. The book's core ops keep this < 1e-5
 * (f64 + central difference). A spike to ~1e-2 or NaN means an adjoint is wrong, not noise.
 * INVARIANT: fn must return a SCALAR tensor (shape [1]); it is rebuilt fresh each probe so
 *   the graph doesn't accumulate stale grads.
 */
export function numericalGradCheck(fn: (t: Tensor) => Tensor, t: Tensor, eps = 1e-4): number {
  // Analytic grad: one backward pass on the original tensor.
  const probe = fn(t);
  if (probe.size !== 1) throw new Error(`numericalGradCheck: fn must return scalar, got ${probe.shape}`);
  t.zeroGrad();
  probe.backward();
  const analytic = Float64Array.from(t.grad);

  let maxErr = 0;
  for (let i = 0; i < t.size; i++) {
    const orig = t.data[i];
    // Central difference: (f(x+eps) - f(x-eps)) / 2eps — O(eps²) accurate, beats forward diff.
    t.data[i] = orig + eps;
    const fPlus = fn(t).data[0];
    t.data[i] = orig - eps;
    const fMinus = fn(t).data[0];
    t.data[i] = orig; // restore — perturbation must not leak into the next probe
    const numeric = (fPlus - fMinus) / (2 * eps);
    maxErr = Math.max(maxErr, Math.abs(numeric - analytic[i]));
  }
  return maxErr;
}

/** Allocate a Tensor of standard normals from an RNG-like draw fn. The book's "noise"
 *  tensors (forward perturbation, sampler z) are all born here so they stay reproducible. */
export function randn(shape: number[], gaussian: () => number): Tensor {
  return Tensor.from(shape, gaussian);
}

/** Same shape as `t`, fresh standard normals. */
export function randnLike(t: Tensor, gaussian: () => number): Tensor {
  return Tensor.from(t.shape, gaussian);
}
