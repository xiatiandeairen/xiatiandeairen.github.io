// core/autograd.ts — The minimal reverse-mode autodiff engine the whole book stands on.
//
// WHY this is the foundation, frozen early: every interpretability experiment later
//   (logit lens, activation patching, SAE training) reads or rewrites intermediate
//   Tensors and trusts that grads are correct. If an adjoint here is wrong, a probe might
//   "learn" from a corrupted signal and we'd publish a fake circuit. So we build it once,
//   carefully, and prove it with gradCheck() before trusting any measurement.
//
// CORE INVARIANT — grads ACCUMULATE (+=), never overwrite (=):
//   A tensor can feed many consumers (weight tying between embedding and unembedding,
//   a residual stream read by attention AND the skip connection, broadcasting a bias).
//   Reverse-mode requires SUMMING contributions over every path. Overwriting keeps only
//   the last path and silently corrupts training. Corollary: callers must zero grads
//   between steps.
//
// LAYOUT: row-major contiguous Float64Array. PRECISION is f64 on purpose — slower and less
//   realistic than a real f32/SIMD framework, but it makes gradCheck pass at < 1e-6 and
//   removes "real bug or just f32 noise?" ambiguity. HONESTY: our wall-clock timings are
//   therefore pessimistic vs production; the transferable signal is relative trends.

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
  _backward: () => void;
  _prev: Tensor[];
  op: string;

  constructor(data: Float64Array | number[], shape: number[], prev: Tensor[] = [], op = "") {
    const flat = data instanceof Float64Array ? data : Float64Array.from(data);
    if (flat.length !== prod(shape)) {
      // Catch shape bugs here, where the message is clear, not 5 ops later as a NaN cascade.
      throw new Error(`Tensor: data length ${flat.length} != prod(${shape}) = ${prod(shape)}`);
    }
    this.data = flat;
    this.grad = new Float64Array(flat.length);
    this.shape = shape.slice();
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
   *  row-major order — reproducible init depends on this draw count and order. */
  static from(shape: number[], gen: () => number): Tensor {
    const n = prod(shape);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = gen();
    return new Tensor(d, shape);
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  /** Detached copy of the data as a fresh leaf Tensor. WHY interp needs this: patching and
   *  SAE work on captured activations as inputs to NEW graphs; we must not let grads leak
   *  back into the original model's graph through them. detach() severs the link. */
  detach(): Tensor {
    return new Tensor(this.data.slice(), this.shape);
  }

  // ---- elementwise add/sub/mul with bias-style broadcast (other broadcast INTO this) ----
  private static ew(
    a: Tensor,
    b: Tensor,
    fwd: (x: number, y: number) => number,
    back: (x: number, y: number, g: number) => [number, number],
    op: string,
  ): Tensor {
    const broadcasting = !shapeEq(a.shape, b.shape);
    if (broadcasting && (b.size > a.size || a.size % b.size !== 0)) {
      throw new Error(`ew(${op}): cannot broadcast ${b.shape} into ${a.shape}`);
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
        // FAILURE MODE guarded: with broadcast, many i map to one bi, so b.grad[bi] must
        // SUM all their contributions — that sum IS the broadcast adjoint. += does it.
        b.grad[bi] += db;
      }
    };
    return t;
  }

  add(o: Tensor): Tensor {
    return Tensor.ew(this, o, (x, y) => x + y, (_x, _y, g) => [g, g], "+");
  }
  sub(o: Tensor): Tensor {
    return Tensor.ew(this, o, (x, y) => x - y, (_x, _y, g) => [g, -g], "-");
  }
  mul(o: Tensor): Tensor {
    return Tensor.ew(this, o, (x, y) => x * y, (x, y, g) => [g * y, g * x], "*");
  }

  mulScalar(s: number): Tensor {
    const out = new Float64Array(this.size);
    for (let i = 0; i < this.size; i++) out[i] = this.data[i] * s;
    const t = new Tensor(out, this.shape, [this], `*${s}`);
    t._backward = () => {
      for (let i = 0; i < this.size; i++) this.grad[i] += s * t.grad[i];
    };
    return t;
  }

  // ---- matmul: (m,k) @ (k,n) -> (m,n) ----
  // WHY only 2-D: every multi-head / batched matmul in this book is expressed by slicing
  //   into 2-D matrices, which keeps THIS adjoint dead simple and provably right:
  //     dA = dC @ B^T,  dB = A^T @ dC. Those two identities are the chain rule for matmul.
  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2)
      throw new Error(`matmul: operands must be 2-D, got ${this.shape} @ ${other.shape}`);
    const [m, k] = this.shape;
    const [k2, n] = other.shape;
    if (k !== k2) throw new Error(`matmul: inner dims ${k} != ${k2}`);
    const A = this.data;
    const B = other.data;
    const out = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const a = A[i * k + p];
        if (a === 0) continue; // correctness-neutral sparsity shortcut
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
            this.grad[i * k + p] += g * B[p * n + j];
            other.grad[p * n + j] += g * A[i * k + p];
          }
        }
      }
    };
    return t;
  }

  /** Transpose a 2-D tensor (materialized, not a view, for clarity). Adjoint: transpose
   *  the grad back. */
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

  /** Sum all -> scalar [1]. Adjoint: broadcast the single grad to every input. */
  sum(): Tensor {
    let s = 0;
    for (let i = 0; i < this.size; i++) s += this.data[i];
    const t = new Tensor([s], [1], [this], "sum");
    t._backward = () => {
      const g = t.grad[0];
      for (let i = 0; i < this.size; i++) this.grad[i] += g;
    };
    return t;
  }
  /** Mean of all -> scalar [1]. Adjoint divides the broadcast grad by N. */
  mean(): Tensor {
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

  relu(): Tensor {
    const out = new Float64Array(this.size);
    for (let i = 0; i < this.size; i++) out[i] = this.data[i] > 0 ? this.data[i] : 0;
    const t = new Tensor(out, this.shape, [this], "relu");
    t._backward = () => {
      for (let i = 0; i < this.size; i++) this.grad[i] += (out[i] > 0 ? 1 : 0) * t.grad[i];
    };
    return t;
  }

  /** GELU (tanh approximation). WHY this matters for interp: GELU is the nonlinearity that
   *  lets MLP neurons act as soft gates / feature detectors — the things SAEs and probes
   *  hunt for. We use the tanh approx (what GPT-2 ships) so behavior matches the literature.
   *  Adjoint is the analytic derivative of that approximation; gradCheck verifies it. */
  gelu(): Tensor {
    const c = Math.sqrt(2 / Math.PI);
    const out = new Float64Array(this.size);
    for (let i = 0; i < this.size; i++) {
      const x = this.data[i];
      const inner = c * (x + 0.044715 * x * x * x);
      out[i] = 0.5 * x * (1 + Math.tanh(inner));
    }
    const t = new Tensor(out, this.shape, [this], "gelu");
    t._backward = () => {
      for (let i = 0; i < this.size; i++) {
        const x = this.data[i];
        const x3 = x * x * x;
        const inner = c * (x + 0.044715 * x3);
        const tanh = Math.tanh(inner);
        const sech2 = 1 - tanh * tanh;
        const dInner = c * (1 + 3 * 0.044715 * x * x);
        const dgelu = 0.5 * (1 + tanh) + 0.5 * x * sech2 * dInner;
        this.grad[i] += dgelu * t.grad[i];
      }
    };
    return t;
  }

  /** Row-wise softmax over last axis of a 2-D (rows, cols) tensor.
   *  STABILITY: subtract per-row max before exp, else large logits overflow to Inf -> NaN
   *  row (the canonical softmax bug). Adjoint uses the compact form
   *    dx_i = s_i * (g_i - sum_j s_j g_j)
   *  avoiding the full (n x n) Jacobian. This op IS the attention pattern in interp. */
  softmax(): Tensor {
    if (this.shape.length !== 2) throw new Error(`softmax: expected 2-D, got ${this.shape}`);
    const [rows, cols] = this.shape;
    const out = new Float64Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const base = r * cols;
      let max = -Infinity;
      for (let j = 0; j < cols; j++) max = Math.max(max, this.data[base + j]);
      let denom = 0;
      for (let j = 0; j < cols; j++) {
        const e = Math.exp(this.data[base + j] - max);
        out[base + j] = e;
        denom += e;
      }
      for (let j = 0; j < cols; j++) out[base + j] /= denom;
    }
    const t = new Tensor(out, this.shape, [this], "softmax");
    t._backward = () => {
      for (let r = 0; r < rows; r++) {
        const base = r * cols;
        let dot = 0;
        for (let j = 0; j < cols; j++) dot += out[base + j] * t.grad[base + j];
        for (let j = 0; j < cols; j++) this.grad[base + j] += out[base + j] * (t.grad[base + j] - dot);
      }
    };
    return t;
  }

  /** Row-wise log-softmax. WHY a dedicated op rather than log(softmax): composing them
   *  computes exp then log of a possibly-tiny prob and loses precision; the fused form
   *  x - max - log(sum exp(x-max)) is stable. Used for NLL / logit-lens KL. */
  logSoftmax(): Tensor {
    if (this.shape.length !== 2) throw new Error(`logSoftmax: expected 2-D, got ${this.shape}`);
    const [rows, cols] = this.shape;
    const out = new Float64Array(rows * cols);
    const sm = new Float64Array(rows * cols); // cache softmax probs for the adjoint
    for (let r = 0; r < rows; r++) {
      const base = r * cols;
      let max = -Infinity;
      for (let j = 0; j < cols; j++) max = Math.max(max, this.data[base + j]);
      let denom = 0;
      for (let j = 0; j < cols; j++) denom += Math.exp(this.data[base + j] - max);
      const logDenom = Math.log(denom);
      for (let j = 0; j < cols; j++) {
        const shifted = this.data[base + j] - max;
        out[base + j] = shifted - logDenom;
        sm[base + j] = Math.exp(out[base + j]);
      }
    }
    const t = new Tensor(out, this.shape, [this], "logSoftmax");
    // d logsoftmax_i / d x_k = delta_ik - softmax_k ; so dx_k = g_k - softmax_k * sum_j g_j
    t._backward = () => {
      for (let r = 0; r < rows; r++) {
        const base = r * cols;
        let sumG = 0;
        for (let j = 0; j < cols; j++) sumG += t.grad[base + j];
        for (let j = 0; j < cols; j++) this.grad[base + j] += t.grad[base + j] - sm[base + j] * sumG;
      }
    };
    return t;
  }

  /** Broadcast a (1, n) row up to (rows, n). Explicit op so its adjoint (sum grad back to
   *  one row) lives in one tested place. FAILURE MODE without it: bias grads end up rows×
   *  too large. */
  broadcastRow(rows: number): Tensor {
    if (this.shape.length !== 2 || this.shape[0] !== 1)
      throw new Error(`broadcastRow: expected [1, n], got ${this.shape}`);
    const n = this.shape[1];
    const out = new Float64Array(rows * n);
    for (let r = 0; r < rows; r++) out.set(this.data, r * n);
    const t = new Tensor(out, [rows, n], [this], "broadcastRow");
    t._backward = () => {
      for (let r = 0; r < rows; r++) for (let j = 0; j < n; j++) this.grad[j] += t.grad[r * n + j];
    };
    return t;
  }

  /**
   * Reverse-topological backward. Build topo order (parents before children) then walk
   * REVERSED so each consumer's grad is final before a node's _backward runs. Only valid
   * starting from a SCALAR loss ([1]) — loss is always scalar, and "d a vector wrt nothing"
   * is undefined; we enforce this rather than silently seed garbage.
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
    if (this.size !== 1)
      throw new Error(`backward(): start from a scalar loss [1], got ${this.shape}`);
    this.grad[0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

// ----------------------------------------------------------------------------
// LayerNorm as a standalone op (used directly + inside nn.LayerNorm).
// ----------------------------------------------------------------------------
//
// WHY a fused op instead of composing sub/mean/div: LN's adjoint couples every element in
//   a row (mean and var depend on all of them). The closed form is both faster and the
//   place to TEACH the gradient. Affine (gamma/beta) is applied by the nn layer, not here,
//   so this op is the pure normalization whose Jacobian interp tooling can reason about.
// EPS inside the sqrt: var+eps avoids div-by-0 when a row is constant (a real case for
//   degenerate residual streams early in training).
export function layerNorm(x: Tensor, eps = 1e-5): Tensor {
  if (x.shape.length !== 2) throw new Error(`layerNorm: expected 2-D, got ${x.shape}`);
  const [rows, dim] = x.shape;
  const out = new Float64Array(rows * dim);
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
    const inv = 1 / Math.sqrt(v + eps);
    invStd[r] = inv;
    for (let j = 0; j < dim; j++) {
      const xh = (x.data[base + j] - m) * inv;
      xhat[base + j] = xh;
      out[base + j] = xh;
    }
  }
  const t = new Tensor(out, x.shape, [x], "layernorm");
  t._backward = () => {
    for (let r = 0; r < rows; r++) {
      const base = r * dim;
      const inv = invStd[r];
      let sumD = 0;
      let sumDX = 0;
      for (let j = 0; j < dim; j++) {
        const g = t.grad[base + j];
        sumD += g;
        sumDX += g * xhat[base + j];
      }
      const mD = sumD / dim;
      const mDX = sumDX / dim;
      // dx = (1/std) * (g - mean(g) - xhat*mean(g*xhat))
      for (let j = 0; j < dim; j++)
        x.grad[base + j] += inv * (t.grad[base + j] - mD - xhat[base + j] * mDX);
    }
  };
  return t;
}

// ----------------------------------------------------------------------------
// embeddingLookup: integer ids -> rows of a (vocab, dim) table.
// ----------------------------------------------------------------------------
//
// Adjoint scatters each output row-grad back to its source vocab row AND ACCUMULATES: the
// same token id appearing twice in a sequence must SUM its grads, else shared tokens train
// at the wrong rate. ids are passed as a plain number[] (not a Tensor) because token ids
// are discrete — there is no gradient wrt an index.
export function embeddingLookup(weight: Tensor, ids: number[]): Tensor {
  if (weight.shape.length !== 2) throw new Error(`embeddingLookup: weight must be 2-D`);
  const [vocab, dim] = weight.shape;
  const n = ids.length;
  const out = new Float64Array(n * dim);
  for (let i = 0; i < n; i++) {
    const id = ids[i];
    if (id < 0 || id >= vocab) throw new Error(`embeddingLookup: id ${id} out of [0,${vocab})`);
    out.set(weight.data.subarray(id * dim, (id + 1) * dim), i * dim);
  }
  const t = new Tensor(out, [n, dim], [weight], "embedding");
  t._backward = () => {
    for (let i = 0; i < n; i++) {
      const src = ids[i] * dim;
      const dst = i * dim;
      for (let d = 0; d < dim; d++) weight.grad[src + d] += t.grad[dst + d];
    }
  };
  return t;
}

// ----------------------------------------------------------------------------
// noGrad — inference switch.
// ----------------------------------------------------------------------------
//
// WHY: at inference (every interp measurement is inference on a frozen model) we want no
//   accidental grad accumulation polluting a later training step. It's a flag, not a
//   separate Tensor path, to keep ONE Tensor class — the teaching simplification. Honest
//   cost: forward still allocates output buffers either way. try/finally restores the flag
//   even on throw so an error can't leak the inference flag into training.
let _noGrad = false;
export function noGradActive(): boolean {
  return _noGrad;
}
export function noGrad<T>(fn: () => T): T {
  const prev = _noGrad;
  _noGrad = true;
  try {
    return fn();
  } finally {
    _noGrad = prev;
  }
}

// ----------------------------------------------------------------------------
// gradCheck — the correctness keystone.
// ----------------------------------------------------------------------------
//
// For each sampled param element, perturb ±eps, recompute the scalar loss via f(), and form
// the CENTRAL finite difference (f(x+eps)-f(x-eps))/(2eps); compare to the analytic grad in
// param.grad. Returns max relative error over checked elements.
//
// CONTRACT: caller runs forward+backward ONCE first so param.grad holds the analytic grad;
//   f() recomputes the SAME scalar loss from current param.data with a FRESH forward (no
//   stale graph). WHY central differences: O(eps^2) error, so f64 gives < 1e-6 for a
//   correct op while a buggy op shows O(1) — unmistakable. Catches missing broadcast sums,
//   sign errors, forgotten +=. Caveat: at kinks (relu at exactly 0) FD disagrees with the
//   subgradient — keep test inputs off exact kinks.
export function gradCheck(
  f: () => number,
  params: Tensor[],
  eps = 1e-5,
  samplePerParam = 8,
): { maxRelError: number; checked: number } {
  let maxRel = 0;
  let checked = 0;
  for (const p of params) {
    const stride = Math.max(1, Math.floor(p.size / samplePerParam));
    for (let i = 0; i < p.size; i += stride) {
      const orig = p.data[i];
      p.data[i] = orig + eps;
      const lp = f();
      p.data[i] = orig - eps;
      const lm = f();
      p.data[i] = orig;
      const numeric = (lp - lm) / (2 * eps);
      const analytic = p.grad[i];
      const denom = Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
      const rel = Math.abs(numeric - analytic) / denom;
      if (rel > maxRel) maxRel = rel;
      checked++;
    }
  }
  return { maxRelError: maxRel, checked };
}
