// core/autograd.ts — The single source of truth for the whole book's compute engine.
//
// WHY this file is frozen after chapter 3: every later stage (layers, optimizers,
//   attention, the GPT) is built ONLY from the operators below. If the contract here
//   shifts, every downstream "honest number" silently changes. So we build it carefully
//   once and never special-case it later.
//
// TWO LEVELS, ONE IDEA (reverse-mode autodiff):
//   - Value: a scalar node. Pedagogical (chapter 1) — easy to see the graph by hand.
//   - Tensor: an n-d node over a flat Float64Array. The real workhorse (chapter 2+).
//   Both record a `_backward` closure that, given the node's own grad, scatters grad
//   into its parents. backward() runs reverse-topological order so each node's grad is
//   fully accumulated before it is used.
//
// CORE INVARIANT — grads ACCUMULATE (+=), never overwrite (=):
//   A parameter can feed multiple consumers (weight sharing, broadcasting fans a value
//   into many outputs). Reverse-mode requires summing contributions from every path.
//   Overwriting would keep only the last path's grad and silently corrupt training.
//   Corollary: callers MUST zero grads between steps (see nn.zeroGrad / optim.zeroGrad).
//
// FAILURE MODE this design defends against: forgetting that broadcast is a real op with
//   a real adjoint. The adjoint of "broadcast a (1,n) row across (m,n)" is "sum the
//   grad back down over the broadcast axis". Skipping that sum gives wrong-shaped or
//   wrong-magnitude grads that gradCheck (metrics.ts) will catch.

// ============================================================================
// Value — scalar reverse-mode autodiff (chapter 1, "see the graph by hand")
// ============================================================================

export class Value {
  data: number;
  grad: number;
  // _backward: pushes THIS node's grad into its parents' grads. No-op for leaves.
  _backward: () => void;
  // _prev: parents, used only to discover the graph for topo sort. A Set dedupes the
  //   case where the same node is used twice in one op (e.g. x*x), which would otherwise
  //   visit it twice in topo and is harmless, but the Set keeps intent clear.
  _prev: Set<Value>;
  op: string; // label for debugging/graph printing only

  constructor(data: number, prev: Value[] = [], op = "") {
    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = new Set(prev);
    this.op = op;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o], "+");
    // d(out)/d(self) = 1, d(out)/d(o) = 1. Note += for the x+x case.
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, [this, o], "*");
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.mul(-1));
  }

  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  pow(exp: number): Value {
    // Only constant exponents — variable exponents need d/dexp = out*ln(self).
    const out = new Value(Math.pow(this.data, exp), [this], `**${exp}`);
    out._backward = () => {
      this.grad += exp * Math.pow(this.data, exp - 1) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data > 0 ? this.data : 0, [this], "relu");
    // Subgradient at 0 is taken as 0 (the standard convention).
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], "tanh");
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad; // d/dx tanh = 1 - tanh^2
    };
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], "exp");
    out._backward = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  log(): Value {
    const out = new Value(Math.log(this.data), [this], "log");
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  /**
   * Reverse-topological backward pass.
   * Build a topo order (parents before children in the list), then walk it REVERSED so
   * every consumer's grad is final before we call a node's _backward. Seed self.grad=1
   * because d(self)/d(self)=1.
   */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

// ============================================================================
// Tensor — n-d reverse-mode autodiff over a flat Float64Array (chapter 2+)
// ============================================================================
//
// LAYOUT: data is row-major contiguous. We keep an explicit `strides` array so future
//   ops (transpose/reshape) can be views in principle; for teaching clarity most ops
//   here materialize fresh contiguous buffers (correctness over zero-copy).
// PRECISION: Float64 everywhere — deliberately. f32 would be faster and more realistic,
//   but f64 makes gradCheck pass at < 1e-6 and removes "is this a real bug or just f32
//   noise?" ambiguity for the reader. The honesty trade-off: our timings are pessimistic
//   vs a real f32/SIMD framework; the transferable signal is relative trends, not abs ms.

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
  _backward: () => void;
  _prev: Tensor[];
  op: string;
  requiresGrad: boolean;

  constructor(data: Float64Array | number[], shape: number[], prev: Tensor[] = [], op = "") {
    const flat = data instanceof Float64Array ? data : Float64Array.from(data);
    if (flat.length !== prod(shape)) {
      // Catching this early turns a silent NaN-cascade into a clear error.
      throw new Error(`Tensor: data length ${flat.length} != prod(shape ${shape}) = ${prod(shape)}`);
    }
    this.data = flat;
    this.grad = new Float64Array(flat.length); // grads start at 0; accumulate over paths
    this.shape = shape.slice();
    this.strides = computeStrides(shape);
    this._backward = () => {};
    this._prev = prev;
    this.op = op;
    this.requiresGrad = true;
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

  /** Build from a generator (e.g. an Rng-backed init). INVARIANT: gen() is called
   *  exactly prod(shape) times in row-major order — stages rely on this draw count
   *  for reproducible init. */
  static from(shape: number[], gen: () => number): Tensor {
    const n = prod(shape);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = gen();
    return new Tensor(d, shape);
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  // ---- elementwise binary with broadcasting ----
  // We support the two shapes the book actually needs:
  //   (a) identical shapes, and
  //   (b) bias-style broadcast where `other` is (1, ..., n) broadcast over a leading
  //       batch dim. The adjoint of broadcast is sum-over-broadcast-axis (see below).
  private static elementwise(
    a: Tensor,
    b: Tensor,
    fwd: (x: number, y: number) => number,
    // dx, dy: local partials wrt a-elem and b-elem given the upstream grad g
    back: (x: number, y: number, g: number) => [number, number],
    op: string,
  ): Tensor {
    const broadcasting = !shapeEq(a.shape, b.shape);
    if (broadcasting) {
      // Require b to be broadcastable INTO a's shape (trailing-dim match, b smaller).
      // This narrow contract keeps the adjoint simple and is all bias/scale needs.
      if (b.size > a.size || a.size % b.size !== 0) {
        throw new Error(`elementwise(${op}): cannot broadcast shape ${b.shape} into ${a.shape}`);
      }
    }
    const n = a.size;
    const out = new Float64Array(n);
    const bsize = b.size;
    for (let i = 0; i < n; i++) {
      out[i] = fwd(a.data[i], b.data[broadcasting ? i % bsize : i]);
    }
    const t = new Tensor(out, a.shape, [a, b], op);
    t._backward = () => {
      for (let i = 0; i < n; i++) {
        const bi = broadcasting ? i % bsize : i;
        const [da, db] = back(a.data[i], b.data[bi], t.grad[i]);
        a.grad[i] += da;
        // FAILURE MODE guarded here: when b is broadcast, MANY i map to one bi, so we
        // must SUM all their grads into b.grad[bi] (the broadcast adjoint). += does it.
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
    return Tensor.elementwise(
      this,
      other,
      (x, y) => x / y,
      (x, y, g) => [g / y, (-g * x) / (y * y)],
      "/",
    );
  }

  /** Multiply by a Python-style scalar. Common enough to deserve its own fast path. */
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

  // ---- matmul: (m,k) @ (k,n) -> (m,n) ----
  // WHY only 2-D: batched/strided matmul is a generalization the book introduces later
  //   by RESHAPING into 2-D, so keeping the core op 2-D keeps its adjoint dead-simple:
  //   dA = dC @ B^T ,  dB = A^T @ dC . Those two identities ARE the chain rule for matmul.
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
        if (a === 0) continue; // tiny sparsity shortcut; correctness-neutral
        const bRow = p * n;
        const oRow = i * n;
        for (let j = 0; j < n; j++) out[oRow + j] += a * B[bRow + j];
      }
    }
    const t = new Tensor(out, [m, n], [this, other], "matmul");
    t._backward = () => {
      const dC = t.grad;
      // dA[i,p] = sum_j dC[i,j] * B[p,j]
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

  // ---- reductions ----
  /** Sum all elements -> scalar tensor shape [1]. Adjoint: broadcast the single grad
   *  back to every input element (each contributed 1). */
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

  /** Mean of all elements. Adjoint divides the broadcast grad by N (chain rule on 1/N). */
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

  // ---- unary nonlinearities ----
  relu(): Tensor {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.data[i] > 0 ? this.data[i] : 0;
    const t = new Tensor(out, this.shape, [this], "relu");
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

  // ---- shape ops ----
  /** Transpose a 2-D tensor. Materializes a fresh buffer (no view) for clarity.
   *  Adjoint: transpose the grad back. */
  transpose(): Tensor {
    if (this.shape.length !== 2) throw new Error(`transpose: expected 2-D, got ${this.shape}`);
    const [r, c] = this.shape;
    const out = new Float64Array(r * c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = this.data[i * c + j];
    const t = new Tensor(out, [c, r], [this], "transpose");
    t._backward = () => {
      for (let i = 0; i < r; i++)
        for (let j = 0; j < c; j++) this.grad[i * c + j] += t.grad[j * r + i];
    };
    return t;
  }

  /** Reshape with the same element count. Buffer is SHARED in forward (same data ref is
   *  fine because we don't mutate data), grads flow straight through 1:1. */
  reshape(newShape: number[]): Tensor {
    if (prod(newShape) !== this.size)
      throw new Error(`reshape: ${this.shape} -> ${newShape} changes element count`);
    const t = new Tensor(this.data, newShape, [this], "reshape");
    t._backward = () => {
      for (let i = 0; i < this.size; i++) this.grad[i] += t.grad[i];
    };
    return t;
  }

  /** Row-wise softmax over the LAST axis of a 2-D (rows, classes) tensor.
   *  NUMERICAL STABILITY: subtract per-row max before exp — without it, large logits
   *  overflow to Inf and the whole row becomes NaN. This is the canonical softmax bug.
   *  Adjoint uses the compact Jacobian-vector form:
   *    dL/dx_i = s_i * (g_i - sum_j s_j g_j)
   *  which avoids materializing the full (n x n) softmax Jacobian. */
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
        for (let j = 0; j < cols; j++)
          this.grad[base + j] += out[base + j] * (t.grad[base + j] - dot);
      }
    };
    return t;
  }

  /**
   * Broadcast a (1, n) row tensor up to (rows, n). Explicit op so its adjoint
   * (sum the grad back down to one row) lives in one tested place rather than being
   * re-derived ad hoc per layer. FAILURE MODE without this: bias grads end up rows× too
   * large or wrong-shaped.
   */
  broadcastRow(rows: number): Tensor {
    if (this.shape.length !== 2 || this.shape[0] !== 1)
      throw new Error(`broadcastRow: expected shape [1, n], got ${this.shape}`);
    const n = this.shape[1];
    const out = new Float64Array(rows * n);
    for (let r = 0; r < rows; r++) out.set(this.data, r * n);
    const t = new Tensor(out, [rows, n], [this], "broadcastRow");
    t._backward = () => {
      for (let r = 0; r < rows; r++)
        for (let j = 0; j < n; j++) this.grad[j] += t.grad[r * n + j];
    };
    return t;
  }

  /**
   * Reverse-topological backward, same algorithm as Value.backward but over Tensors.
   * Seeds grad to all-ones ONLY if this is a scalar ([1]); otherwise the caller must
   * have set t.grad already (you cannot d a vector wrt nothing). We enforce the scalar
   * convention because loss is always a scalar — backward() is called on the loss.
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
      throw new Error(
        `backward(): can only start from a scalar loss (shape [1]); got ${this.shape}. ` +
          `For non-scalar outputs, set .grad manually and call _backward via topo yourself.`,
      );
    }
    this.grad[0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

// ============================================================================
// noGrad — inference switch
// ============================================================================
//
// WHY: at inference we still call the same forward ops, but we want NO graph bookkeeping
//   cost and NO accidental grad accumulation. This is a lightweight flag rather than a
//   separate no-tracking Tensor path: stages check noGradActive() to skip building
//   _backward closures / cheap-out where it matters (e.g. dropout becomes identity).
//   Keeping ONE Tensor class (vs train/eval variants) is the teaching simplification;
//   the honest cost is that forward still allocates an output buffer either way.

let _noGrad = false;
export function noGradActive(): boolean {
  return _noGrad;
}

/** Run fn with grad tracking semantically disabled; restores the previous flag even on
 *  throw (try/finally) so a thrown error can't leak the inference flag into training. */
export function noGrad<T>(fn: () => T): T {
  const prev = _noGrad;
  _noGrad = true;
  try {
    return fn();
  } finally {
    _noGrad = prev;
  }
}
