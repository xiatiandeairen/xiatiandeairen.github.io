// core/tensor.ts — Minimal reverse-mode autograd over small tensors (micrograd-style,
//   but vector/matrix capable). The single compute engine for the whole MoE book.
//
// WHY one frozen engine: every stage (gating net, experts, load-balance loss) is built
//   ONLY from the operators below. If this contract shifts, every downstream "honest
//   number" silently changes. So we build it once, carefully, and never special-case.
//
// DESIGN: a Value is an n-d node backed by a flat Float64Array `data` + matching `grad`,
//   plus a `shape` ([rows, cols]; scalars are [1,1], vectors are [n,1] or [1,n]) and a
//   `_backward` closure. backward() runs reverse-topological order so each node's grad is
//   fully accumulated before it is consumed by its parents.
//
// CORE INVARIANT — grads ACCUMULATE (+=), never overwrite (=):
//   A parameter can feed many consumers (an expert reused across tokens, a broadcast row
//   fanned across a batch). Reverse-mode requires SUMMING every path's contribution.
//   Overwriting keeps only the last path and silently corrupts training.
//   Corollary: callers MUST zeroGrad() between optimizer steps.
//
// FAILURE MODE this defends against: treating broadcast/sum as grad-free. The adjoint of
//   "broadcast a (1,c) row across (r,c)" is "sum grad back down the broadcast axis"; the
//   adjoint of sum/mean is "scatter grad back up". gradCheck() at the bottom is the test
//   that any new operator's _backward matches its forward — first chapter uses it to
//   PROVE the engine, not assert it.
//
// SCALE NOTE: pure-JS flat arrays, models kept to a few thousand params => each stage
//   converges in <few seconds on one CPU core. This is a teaching engine, not BLAS.

export type Shape = [number, number]; // [rows, cols]; always 2-D for simplicity.

let _idCounter = 0;

export class Value {
  data: Float64Array;
  grad: Float64Array;
  shape: Shape;
  _backward: () => void;
  _prev: Value[];
  op: string;
  readonly id: number;

  constructor(data: Float64Array | number[], shape: Shape, prev: Value[] = [], op = "") {
    const flat = data instanceof Float64Array ? data : Float64Array.from(data);
    if (flat.length !== shape[0] * shape[1]) {
      throw new Error(`Value: data length ${flat.length} != shape ${shape[0]}x${shape[1]}`);
    }
    this.data = flat;
    this.grad = new Float64Array(flat.length); // starts at 0; accumulates in backward()
    this.shape = shape;
    this._backward = () => {};
    this._prev = prev;
    this.op = op;
    this.id = _idCounter++;
  }

  get rows(): number {
    return this.shape[0];
  }
  get cols(): number {
    return this.shape[1];
  }
  /** Flat index for (i,j) in row-major layout. INVARIANT: callers never bypass this. */
  idx(i: number, j: number): number {
    return i * this.shape[1] + j;
  }

  // ---- constructors ---------------------------------------------------------

  static scalar(x: number): Value {
    return new Value([x], [1, 1], [], "scalar");
  }
  /** A leaf filled with `fill`. Leaves have a no-op _backward (no parents). */
  static zeros(rows: number, cols: number, fill = 0): Value {
    const d = new Float64Array(rows * cols);
    if (fill !== 0) d.fill(fill);
    return new Value(d, [rows, cols], [], "leaf");
  }
  static from(arr: number[], shape: Shape): Value {
    return new Value(arr, shape, [], "leaf");
  }

  // ---- elementwise ----------------------------------------------------------

  /**
   * Elementwise add. Supports row-broadcast: if `other` is (1,c) it is added to every
   * row of (r,c). The adjoint of broadcast is "sum grad down the broadcast axis".
   */
  add(other: Value): Value {
    const bcast = this._broadcastRow(other);
    const out = new Value(new Float64Array(this.data.length), this.shape, [this, bcast.node], "+");
    for (let k = 0; k < this.data.length; k++) out.data[k] = this.data[k] + bcast.read(k);
    out._backward = () => {
      for (let k = 0; k < this.data.length; k++) this.grad[k] += out.grad[k];
      bcast.accum(out.grad);
    };
    return out;
  }

  sub(other: Value): Value {
    return this.add(other.mul(Value.scalar(-1)));
  }

  /** Elementwise multiply with the SAME row-broadcast rule as add. */
  mul(other: Value): Value {
    // Scalar fast path keeps the common "scale by a learnable/constant scalar" cheap.
    if (other.shape[0] === 1 && other.shape[1] === 1) {
      const s = other.data[0];
      const out = new Value(new Float64Array(this.data.length), this.shape, [this, other], "*scalar");
      for (let k = 0; k < this.data.length; k++) out.data[k] = this.data[k] * s;
      out._backward = () => {
        for (let k = 0; k < this.data.length; k++) {
          this.grad[k] += s * out.grad[k];
          other.grad[0] += this.data[k] * out.grad[k];
        }
      };
      return out;
    }
    const bcast = this._broadcastRow(other);
    const out = new Value(new Float64Array(this.data.length), this.shape, [this, bcast.node], "*");
    for (let k = 0; k < this.data.length; k++) out.data[k] = this.data[k] * bcast.read(k);
    out._backward = () => {
      for (let k = 0; k < this.data.length; k++) this.grad[k] += bcast.read(k) * out.grad[k];
      const tmp = new Float64Array(this.data.length);
      for (let k = 0; k < this.data.length; k++) tmp[k] = this.data[k] * out.grad[k];
      bcast.accum(tmp);
    };
    return out;
  }

  div(other: Value): Value {
    return this.mul(other.pow(-1));
  }

  /** Elementwise power by a CONSTANT exponent (variable exponents need ln self). */
  pow(exp: number): Value {
    const out = new Value(new Float64Array(this.data.length), this.shape, [this], `**${exp}`);
    for (let k = 0; k < this.data.length; k++) out.data[k] = Math.pow(this.data[k], exp);
    out._backward = () => {
      for (let k = 0; k < this.data.length; k++) {
        this.grad[k] += exp * Math.pow(this.data[k], exp - 1) * out.grad[k];
      }
    };
    return out;
  }

  relu(): Value {
    const out = new Value(new Float64Array(this.data.length), this.shape, [this], "relu");
    for (let k = 0; k < this.data.length; k++) out.data[k] = this.data[k] > 0 ? this.data[k] : 0;
    out._backward = () => {
      // Subgradient at 0 is taken as 0 — standard convention, matches PyTorch.
      for (let k = 0; k < this.data.length; k++) this.grad[k] += (out.data[k] > 0 ? 1 : 0) * out.grad[k];
    };
    return out;
  }

  tanh(): Value {
    const out = new Value(new Float64Array(this.data.length), this.shape, [this], "tanh");
    for (let k = 0; k < this.data.length; k++) out.data[k] = Math.tanh(this.data[k]);
    out._backward = () => {
      for (let k = 0; k < this.data.length; k++) this.grad[k] += (1 - out.data[k] * out.data[k]) * out.grad[k];
    };
    return out;
  }

  exp(): Value {
    const out = new Value(new Float64Array(this.data.length), this.shape, [this], "exp");
    for (let k = 0; k < this.data.length; k++) out.data[k] = Math.exp(this.data[k]);
    out._backward = () => {
      for (let k = 0; k < this.data.length; k++) this.grad[k] += out.data[k] * out.grad[k];
    };
    return out;
  }

  log(): Value {
    const out = new Value(new Float64Array(this.data.length), this.shape, [this], "log");
    for (let k = 0; k < this.data.length; k++) out.data[k] = Math.log(this.data[k]);
    out._backward = () => {
      for (let k = 0; k < this.data.length; k++) this.grad[k] += (1 / this.data[k]) * out.grad[k];
    };
    return out;
  }

  // ---- matmul ---------------------------------------------------------------

  /**
   * Matrix multiply (r,k) x (k,c) -> (r,c). The workhorse of every linear layer.
   * ADJOINTS: dA = dOut @ B^T, dB = A^T @ dOut. Getting these transposes wrong is the
   *   classic MoE-from-scratch bug; gradCheck() below is how we keep ourselves honest.
   */
  matmul(other: Value): Value {
    const [r, k] = this.shape;
    const [k2, c] = other.shape;
    if (k !== k2) throw new Error(`matmul shape mismatch: (${r},${k}) x (${k2},${c})`);
    const out = new Value(new Float64Array(r * c), [r, c], [this, other], "matmul");
    const A = this.data,
      B = other.data;
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) {
        let acc = 0;
        for (let p = 0; p < k; p++) acc += A[i * k + p] * B[p * c + j];
        out.data[i * c + j] = acc;
      }
    }
    out._backward = () => {
      const dO = out.grad;
      // dA[i,p] = sum_j dO[i,j] * B[p,j]
      for (let i = 0; i < r; i++) {
        for (let p = 0; p < k; p++) {
          let acc = 0;
          for (let j = 0; j < c; j++) acc += dO[i * c + j] * B[p * c + j];
          this.grad[i * k + p] += acc;
        }
      }
      // dB[p,j] = sum_i A[i,p] * dO[i,j]
      for (let p = 0; p < k; p++) {
        for (let j = 0; j < c; j++) {
          let acc = 0;
          for (let i = 0; i < r; i++) acc += A[i * k + p] * dO[i * c + j];
          other.grad[p * c + j] += acc;
        }
      }
    };
    return out;
  }

  // ---- reductions -----------------------------------------------------------

  /** Sum ALL elements to a scalar. Adjoint scatters the scalar grad to every element. */
  sum(): Value {
    let s = 0;
    for (let k = 0; k < this.data.length; k++) s += this.data[k];
    const out = new Value([s], [1, 1], [this], "sum");
    out._backward = () => {
      for (let k = 0; k < this.data.length; k++) this.grad[k] += out.grad[0];
    };
    return out;
  }

  /** Mean of ALL elements to a scalar. Adjoint is sum's adjoint / N. */
  mean(): Value {
    const n = this.data.length;
    return this.sum().mul(Value.scalar(1 / n));
  }

  /**
   * Max over a single ROW vector (shape [1,c]) -> scalar. Used for numerically-stable
   * softmax/logsumexp. Adjoint routes grad only to the argmax element (subgradient).
   * INVARIANT: caller passes a row vector; max over a matrix is ambiguous here on purpose.
   */
  maxRow(): Value {
    if (this.rows !== 1) throw new Error("maxRow expects a [1,c] row vector");
    let m = this.data[0],
      arg = 0;
    for (let j = 1; j < this.cols; j++)
      if (this.data[j] > m) {
        m = this.data[j];
        arg = j;
      }
    const out = new Value([m], [1, 1], [this], "maxRow");
    out._backward = () => {
      this.grad[arg] += out.grad[0];
    };
    return out;
  }

  // ---- routing / classification primitives ----------------------------------

  /**
   * Row-wise softmax over a [1,c] logit row -> [1,c] probability row.
   * WHY subtract max: exp(large) overflows; shifting by the row max is the standard
   *   numerically-stable softmax and leaves probabilities unchanged.
   * This is THE gating operator: gate logits -> per-expert probabilities.
   */
  softmaxRow(): Value {
    if (this.rows !== 1) throw new Error("softmaxRow expects a [1,c] row vector");
    const c = this.cols;
    let m = this.data[0];
    for (let j = 1; j < c; j++) if (this.data[j] > m) m = this.data[j];
    const exps = new Float64Array(c);
    let denom = 0;
    for (let j = 0; j < c; j++) {
      exps[j] = Math.exp(this.data[j] - m);
      denom += exps[j];
    }
    const out = new Value(new Float64Array(c), [1, c], [this], "softmaxRow");
    for (let j = 0; j < c; j++) out.data[j] = exps[j] / denom;
    out._backward = () => {
      // Jacobian: ds_i/dz_j = s_i(δ_ij - s_j). dz_j = s_j (dOut_j - sum_i dOut_i s_i).
      const s = out.data,
        dO = out.grad;
      let dot = 0;
      for (let i = 0; i < c; i++) dot += dO[i] * s[i];
      for (let j = 0; j < c; j++) this.grad[j] += s[j] * (dO[j] - dot);
    };
    return out;
  }

  /**
   * logsumexp over a [1,c] row -> scalar. Building block for stable cross-entropy.
   * Adjoint is softmax(row) scaled by out.grad — the classic "lse grad is softmax".
   */
  logsumexpRow(): Value {
    if (this.rows !== 1) throw new Error("logsumexpRow expects a [1,c] row vector");
    const c = this.cols;
    let m = this.data[0];
    for (let j = 1; j < c; j++) if (this.data[j] > m) m = this.data[j];
    let denom = 0;
    for (let j = 0; j < c; j++) denom += Math.exp(this.data[j] - m);
    const lse = m + Math.log(denom);
    const out = new Value([lse], [1, 1], [this], "logsumexpRow");
    out._backward = () => {
      for (let j = 0; j < c; j++) this.grad[j] += (Math.exp(this.data[j] - m) / denom) * out.grad[0];
    };
    return out;
  }

  /**
   * Gather one element at column `j` from a [1,c] row -> scalar. The differentiable form
   * of "pick the probability/logit of the chosen expert/class". Adjoint routes grad to
   * the single picked index. (one-hot @ row, done without materializing the one-hot.)
   */
  gather(j: number): Value {
    if (this.rows !== 1) throw new Error("gather expects a [1,c] row vector");
    if (j < 0 || j >= this.cols) throw new Error(`gather index ${j} out of range [0,${this.cols})`);
    const out = new Value([this.data[j]], [1, 1], [this], `gather[${j}]`);
    out._backward = () => {
      this.grad[j] += out.grad[0];
    };
    return out;
  }

  // ---- broadcast helper -----------------------------------------------------

  /**
   * Internal: lets add/mul accept either a same-shape Value or a [1,c] row to broadcast
   * across this Value's rows. Returns read(k) for the forward and accum(grad) that sums
   * incoming grad back DOWN the broadcast axis into the row's grad — the broadcast adjoint.
   */
  private _broadcastRow(other: Value): {
    node: Value;
    read: (k: number) => number;
    accum: (g: Float64Array) => void;
  } {
    const [r, c] = this.shape;
    if (other.shape[0] === r && other.shape[1] === c) {
      return {
        node: other,
        read: (k) => other.data[k],
        accum: (g) => {
          for (let k = 0; k < g.length; k++) other.grad[k] += g[k];
        },
      };
    }
    if (other.shape[0] === 1 && other.shape[1] === c) {
      return {
        node: other,
        read: (k) => other.data[k % c],
        accum: (g) => {
          // adjoint of row-broadcast: sum each column's incoming grad into the row.
          for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) other.grad[j] += g[i * c + j];
        },
      };
    }
    if (other.shape[0] === 1 && other.shape[1] === 1) {
      // scalar broadcast across the whole tensor; adjoint sums ALL grad into the scalar.
      return {
        node: other,
        read: () => other.data[0],
        accum: (g) => {
          let s = 0;
          for (let k = 0; k < g.length; k++) s += g[k];
          other.grad[0] += s;
        },
      };
    }
    throw new Error(`broadcast: cannot align (${r},${c}) with (${other.shape[0]},${other.shape[1]})`);
  }

  // ---- backward -------------------------------------------------------------

  /**
   * Reverse-mode autodiff. Builds reverse-topo order, seeds this node's grad with 1
   * (d(self)/d(self) = 1), then runs each _backward once.
   * INVARIANT: backward() does NOT zero grads first — grads accumulate by design, so the
   *   caller owns zeroing. Calling backward() twice without zeroGrad DOUBLES the grads
   *   (a real, easy-to-make bug this comment exists to flag).
   */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<number>();
    const build = (v: Value) => {
      if (visited.has(v.id)) return;
      visited.add(v.id);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);
    this.grad.fill(0);
    this.grad[0] = 1; // seed: d(scalar loss)/d(itself) = 1
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

/**
 * Numerical gradient check: compares analytic grad (from backward) against central
 * finite differences for every input scalar. Returns the max relative error.
 * WHY central differences: O(eps^2) accuracy vs O(eps) one-sided, so eps=1e-5 gives
 *   ~1e-7 error for a correct op — well below the ~1e-4 pass threshold stages use.
 * USAGE: stage01 calls this to PROVE the engine before trusting any training number.
 *
 * @param f maps the input Value to a SCALAR loss Value (shape [1,1]).
 * @param x the input leaf whose grad is checked. f must rebuild the graph from x.
 */
export function gradCheck(f: (x: Value) => Value, x: Value, eps = 1e-5): number {
  // Analytic grad.
  x.grad.fill(0);
  const out = f(x);
  if (out.shape[0] !== 1 || out.shape[1] !== 1) throw new Error("gradCheck: f must return a scalar");
  out.backward();
  const analytic = Float64Array.from(x.grad);

  // Numerical grad, element by element.
  let maxRel = 0;
  for (let k = 0; k < x.data.length; k++) {
    const orig = x.data[k];
    x.data[k] = orig + eps;
    const plus = f(x).data[0];
    x.data[k] = orig - eps;
    const minus = f(x).data[0];
    x.data[k] = orig;
    const num = (plus - minus) / (2 * eps);
    const denom = Math.max(1e-8, Math.abs(num) + Math.abs(analytic[k]));
    const rel = Math.abs(num - analytic[k]) / denom;
    if (rel > maxRel) maxRel = rel;
  }
  return maxRel;
}
