// core/tensor.ts — Minimal dynamic-graph reverse-mode autodiff over Float64Array.
//
// WHY this engine exists (and why it is deliberately small):
//   This book trains TOY sequence models in seconds on a CPU. We do NOT need SIMD,
//   no GPU, no fused kernels. We need (a) correct gradients so loss curves are real,
//   and (b) enough operators to build RNN/LSTM/GRU/attention/SSM cells. Correctness
//   over speed. Default dtype is Float64 — f32 rounding would make grad-check noisy.
//
// CORE INVARIANT — grads ACCUMULATE (+=), never overwrite (=):
//   A node can feed many consumers (shared weights, broadcasting fans one value into
//   many outputs, an RNN reuses the same W at every timestep). Reverse-mode requires
//   summing every path's contribution. Overwrite keeps only the last path => silently
//   wrong training. Corollary: callers MUST zero grads between steps (nn.zeroGrad).
//
// THE ADJOINT OF BROADCAST IS A SUM:
//   Forward broadcasts a small shape up to a big one. Backward must sum the big grad
//   back DOWN over every axis that was broadcast (size-1 -> N), else grads are the
//   wrong shape/magnitude. `unbroadcast()` below is where that happens; most autograd
//   bugs in a from-scratch engine live in exactly this function.
//
// FAILURE MODE the topo sort defends against: visiting a node before all its grad
//   contributions are in. backward() runs strict reverse-topological order so each
//   node's grad is complete before its _backward fires.

export type Shape = number[];

function numel(shape: Shape): number {
  return shape.reduce((a, b) => a * b, 1);
}

function shapeEq(a: Shape, b: Shape): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

// Row-major strides for a shape. strides[i] = product of dims after i.
function strides(shape: Shape): number[] {
  const s = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) s[i] = s[i + 1] * shape[i + 1];
  return s;
}

// Broadcast two shapes NumPy-style (align right, dims must be equal or one is 1).
// FAILURE MODE: incompatible dims throw here rather than producing garbage later.
function broadcastShapes(a: Shape, b: Shape): Shape {
  const n = Math.max(a.length, b.length);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`cannot broadcast shapes [${a}] and [${b}]`);
    }
    out[n - 1 - i] = Math.max(da, db);
  }
  return out;
}

export class Tensor {
  data: Float64Array;
  shape: Shape;
  grad: Float64Array;
  _backward: () => void;
  _prev: Tensor[];
  op: string; // debug label only

  constructor(data: Float64Array, shape: Shape, prev: Tensor[] = [], op = "") {
    if (data.length !== numel(shape)) {
      throw new Error(`data length ${data.length} != numel([${shape}])`);
    }
    this.data = data;
    this.shape = shape;
    this.grad = new Float64Array(data.length); // zeros
    this._backward = () => {};
    this._prev = prev;
    this.op = op;
  }

  get size(): number {
    return this.data.length;
  }

  // ---- constructors -------------------------------------------------------
  static zeros(shape: Shape): Tensor {
    return new Tensor(new Float64Array(numel(shape)), shape.slice());
  }
  static ones(shape: Shape): Tensor {
    const d = new Float64Array(numel(shape));
    d.fill(1);
    return new Tensor(d, shape.slice());
  }
  static from(arr: number[], shape: Shape): Tensor {
    return new Tensor(Float64Array.from(arr), shape.slice());
  }
  /**
   * Gaussian init. `scale` multiplies each N(0,1) draw (pass xavier/orthogonal std
   * from nn.ts). INVARIANT: draws thread the shared Rng so init is reproducible.
   */
  static randn(shape: Shape, rng: { normal(): number }, scale = 1): Tensor {
    const n = numel(shape);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = rng.normal() * scale;
    return new Tensor(d, shape.slice());
  }

  // ---- grad accumulation helper for a same-shaped contribution ------------
  private accGrad(g: Float64Array): void {
    const gr = this.grad;
    for (let i = 0; i < gr.length; i++) gr[i] += g[i];
  }

  // Sum a (possibly broadcast) grad `g` of shape `gShape` back down to THIS tensor's
  // shape, then accumulate. This is the adjoint of broadcasting (see header).
  private accUnbroadcast(g: Float64Array, gShape: Shape): void {
    const reduced = unbroadcast(g, gShape, this.shape);
    this.accGrad(reduced);
  }

  // ---- elementwise binary with broadcasting -------------------------------
  private static binary(
    a: Tensor,
    b: Tensor,
    fwd: (x: number, y: number) => number,
    // local partials d out / d a, d out / d b at (x,y)
    da: (x: number, y: number, o: number) => number,
    db: (x: number, y: number, o: number) => number,
    op: string,
  ): Tensor {
    const outShape = broadcastShapes(a.shape, b.shape);
    const n = numel(outShape);
    const out = new Float64Array(n);
    const oStr = strides(outShape);
    const ai = bIndexer(a.shape, outShape);
    const bi = bIndexer(b.shape, outShape);
    for (let i = 0; i < n; i++) {
      const ia = ai(i, oStr, outShape);
      const ib = bi(i, oStr, outShape);
      out[i] = fwd(a.data[ia], b.data[ib]);
    }
    const t = new Tensor(out, outShape, [a, b], op);
    t._backward = () => {
      const ga = new Float64Array(n);
      const gb = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const ia = ai(i, oStr, outShape);
        const ib = bi(i, oStr, outShape);
        const x = a.data[ia];
        const y = b.data[ib];
        const o = t.data[i];
        ga[i] = da(x, y, o) * t.grad[i];
        gb[i] = db(x, y, o) * t.grad[i];
      }
      a.accUnbroadcast(ga, outShape);
      b.accUnbroadcast(gb, outShape);
    };
    return t;
  }

  add(o: Tensor): Tensor {
    return Tensor.binary(this, o, (x, y) => x + y, () => 1, () => 1, "+");
  }
  sub(o: Tensor): Tensor {
    return Tensor.binary(this, o, (x, y) => x - y, () => 1, () => -1, "-");
  }
  mul(o: Tensor): Tensor {
    return Tensor.binary(this, o, (x, y) => x * y, (_x, y) => y, (x) => x, "*");
  }
  // scalar convenience (no graph node for the scalar — it's a constant)
  scale(k: number): Tensor {
    const out = new Float64Array(this.size);
    for (let i = 0; i < out.length; i++) out[i] = this.data[i] * k;
    const t = new Tensor(out, this.shape.slice(), [this], `*${k}`);
    t._backward = () => {
      for (let i = 0; i < out.length; i++) this.grad[i] += k * t.grad[i];
    };
    return t;
  }
  addScalar(k: number): Tensor {
    const out = new Float64Array(this.size);
    for (let i = 0; i < out.length; i++) out[i] = this.data[i] + k;
    const t = new Tensor(out, this.shape.slice(), [this], `+${k}`);
    t._backward = () => this.accGrad(t.grad);
    return t;
  }

  // ---- 2D matmul ----------------------------------------------------------
  // INVARIANT: both operands are exactly rank-2. Higher-rank "batched matmul" is
  //   intentionally NOT supported — stages reshape to 2D first. Keeps the adjoint
  //   (dA = dO @ B^T, dB = A^T @ dO) trivially correct.
  matmul(o: Tensor): Tensor {
    if (this.shape.length !== 2 || o.shape.length !== 2) {
      throw new Error(`matmul needs 2D; got [${this.shape}] x [${o.shape}]`);
    }
    const [m, k] = this.shape;
    const [k2, p] = o.shape;
    if (k !== k2) throw new Error(`matmul inner dim ${k} != ${k2}`);
    const out = new Float64Array(m * p);
    const A = this.data;
    const B = o.data;
    for (let i = 0; i < m; i++) {
      for (let t = 0; t < k; t++) {
        const a = A[i * k + t];
        if (a === 0) continue;
        const boff = t * p;
        const ooff = i * p;
        for (let j = 0; j < p; j++) out[ooff + j] += a * B[boff + j];
      }
    }
    const res = new Tensor(out, [m, p], [this, o], "matmul");
    res._backward = () => {
      const dO = res.grad;
      // dA[i,t] = sum_j dO[i,j] * B[t,j]
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < p; j++) {
          const g = dO[i * p + j];
          if (g === 0) continue;
          for (let t = 0; t < k; t++) {
            this.grad[i * k + t] += g * B[t * p + j];
            o.grad[t * p + j] += g * A[i * k + t];
          }
        }
      }
    };
    return res;
  }

  // ---- transpose (2D only) ------------------------------------------------
  transpose(): Tensor {
    if (this.shape.length !== 2) throw new Error("transpose: 2D only");
    const [r, c] = this.shape;
    const out = new Float64Array(r * c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = this.data[i * c + j];
    const t = new Tensor(out, [c, r], [this], "T");
    t._backward = () => {
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) this.grad[i * c + j] += t.grad[j * r + i];
    };
    return t;
  }

  // ---- reshape (view; same element order) ---------------------------------
  reshape(shape: Shape): Tensor {
    let inferred = shape.slice();
    const neg = inferred.indexOf(-1);
    if (neg >= 0) {
      const rest = inferred.filter((v) => v !== -1).reduce((a, b) => a * b, 1);
      inferred[neg] = this.size / rest;
    }
    if (numel(inferred) !== this.size) throw new Error(`reshape [${this.shape}]->[${shape}] size mismatch`);
    const t = new Tensor(this.data.slice(), inferred, [this], "reshape");
    // reshape preserves element order: grad maps 1:1.
    t._backward = () => this.accGrad(t.grad);
    return t;
  }

  // ---- slice along one axis ([start,end)) ---------------------------------
  // Backward SCATTERS the slice grad back into the parent's full grad buffer at the
  // sliced positions (the un-sliced positions get 0). This is the adjoint of a gather.
  slice(axis: number, start: number, end: number): Tensor {
    const ax = axis < 0 ? this.shape.length + axis : axis;
    const len = end - start;
    if (len <= 0 || end > this.shape[ax]) throw new Error(`slice out of range`);
    const outShape = this.shape.slice();
    outShape[ax] = len;
    const inStr = strides(this.shape);
    const outN = numel(outShape);
    const outStr = strides(outShape);
    const out = new Float64Array(outN);
    const map = new Int32Array(outN); // out flat idx -> in flat idx, reused in backward
    for (let oi = 0; oi < outN; oi++) {
      // decode multi-index of out, shift axis by start, encode into in.
      let rem = oi;
      let inFlat = 0;
      for (let d = 0; d < outShape.length; d++) {
        const coord = Math.floor(rem / outStr[d]) % outShape[d];
        const inCoord = d === ax ? coord + start : coord;
        inFlat += inCoord * inStr[d];
        rem -= coord * outStr[d];
      }
      out[oi] = this.data[inFlat];
      map[oi] = inFlat;
    }
    const t = new Tensor(out, outShape, [this], "slice");
    t._backward = () => {
      for (let oi = 0; oi < outN; oi++) this.grad[map[oi]] += t.grad[oi];
    };
    return t;
  }

  // ---- concat along one axis ----------------------------------------------
  // Forward gathers; backward scatters each piece's grad back to its source tensor.
  static concat(parts: Tensor[], axis: number): Tensor {
    if (parts.length === 0) throw new Error("concat: empty list");
    const ax = axis < 0 ? parts[0].shape.length + axis : axis;
    const base = parts[0].shape;
    let axTotal = 0;
    for (const p of parts) {
      if (p.shape.length !== base.length) throw new Error("concat: rank mismatch");
      for (let d = 0; d < base.length; d++) {
        if (d !== ax && p.shape[d] !== base[d]) throw new Error(`concat: dim ${d} mismatch`);
      }
      axTotal += p.shape[ax];
    }
    const outShape = base.slice();
    outShape[ax] = axTotal;
    const outN = numel(outShape);
    const outStr = strides(outShape);
    const out = new Float64Array(outN);
    // For each part, record (outFlat, inFlat) mapping for backward scatter.
    const maps: { part: Tensor; pairs: Int32Array }[] = [];
    let axOffset = 0;
    for (const p of parts) {
      const pStr = strides(p.shape);
      const pn = p.size;
      const pairs = new Int32Array(pn * 2);
      for (let pi = 0; pi < pn; pi++) {
        let rem = pi;
        let outFlat = 0;
        for (let d = 0; d < p.shape.length; d++) {
          const coord = Math.floor(rem / pStr[d]) % p.shape[d];
          const outCoord = d === ax ? coord + axOffset : coord;
          outFlat += outCoord * outStr[d];
          rem -= coord * pStr[d];
        }
        out[outFlat] = p.data[pi];
        pairs[pi * 2] = outFlat;
        pairs[pi * 2 + 1] = pi;
      }
      maps.push({ part: p, pairs });
      axOffset += p.shape[ax];
    }
    const t = new Tensor(out, outShape, parts.slice(), "concat");
    t._backward = () => {
      for (const { part, pairs } of maps) {
        for (let i = 0; i < pairs.length; i += 2) part.grad[pairs[i + 1]] += t.grad[pairs[i]];
      }
    };
    return t;
  }

  // ---- unary elementwise activations --------------------------------------
  private unary(fwd: (x: number) => number, back: (x: number, o: number) => number, op: string): Tensor {
    const out = new Float64Array(this.size);
    for (let i = 0; i < out.length; i++) out[i] = fwd(this.data[i]);
    const t = new Tensor(out, this.shape.slice(), [this], op);
    t._backward = () => {
      for (let i = 0; i < out.length; i++) this.grad[i] += back(this.data[i], out[i]) * t.grad[i];
    };
    return t;
  }
  tanh(): Tensor {
    return this.unary(Math.tanh, (_x, o) => 1 - o * o, "tanh");
  }
  sigmoid(): Tensor {
    return this.unary((x) => 1 / (1 + Math.exp(-x)), (_x, o) => o * (1 - o), "sigmoid");
  }
  relu(): Tensor {
    return this.unary((x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0), "relu");
  }
  exp(): Tensor {
    return this.unary(Math.exp, (_x, o) => o, "exp");
  }
  log(): Tensor {
    // FAILURE MODE: log(<=0) -> NaN/-Inf. Stages must pass positive inputs (softmax
    //   outputs, abs values). We do not clamp here so bugs surface loudly.
    return this.unary(Math.log, (x) => 1 / x, "log");
  }

  // ---- reductions (sum / mean over one axis, or all) ----------------------
  // axis === undefined: reduce everything to a scalar Tensor (shape [1]).
  sum(axis?: number): Tensor {
    if (axis === undefined) {
      let s = 0;
      for (let i = 0; i < this.size; i++) s += this.data[i];
      const t = new Tensor(Float64Array.from([s]), [1], [this], "sum");
      t._backward = () => {
        const g = t.grad[0];
        for (let i = 0; i < this.size; i++) this.grad[i] += g;
      };
      return t;
    }
    return this.reduceAxis(axis, false);
  }
  mean(axis?: number): Tensor {
    if (axis === undefined) {
      const n = this.size;
      return this.sum().scale(1 / n);
    }
    return this.reduceAxis(axis, true);
  }

  private reduceAxis(axis: number, asMean: boolean): Tensor {
    const ax = axis < 0 ? this.shape.length + axis : axis;
    const outShape = this.shape.slice();
    const axDim = outShape[ax];
    outShape.splice(ax, 1); // dropped axis (keepdim=false)
    const outN = Math.max(1, numel(outShape));
    const inStr = strides(this.shape);
    const out = new Float64Array(outN);
    const outStrFull = strides(outShape.length ? outShape : [1]);
    // For each input element, map to its output bucket (decode in-index, drop ax).
    const bucketOf = (inFlat: number): number => {
      let rem = inFlat;
      let outFlat = 0;
      let od = 0;
      for (let d = 0; d < this.shape.length; d++) {
        const coord = Math.floor(rem / inStr[d]) % this.shape[d];
        rem -= coord * inStr[d];
        if (d === ax) continue;
        outFlat += coord * (outShape.length ? outStrFull[od] : 0);
        od++;
      }
      return outFlat;
    };
    for (let i = 0; i < this.size; i++) out[bucketOf(i)] += this.data[i];
    if (asMean) for (let i = 0; i < outN; i++) out[i] /= axDim;
    const t = new Tensor(out, outShape.length ? outShape : [1], [this], asMean ? "mean" : "sum");
    t._backward = () => {
      const scale = asMean ? 1 / axDim : 1;
      for (let i = 0; i < this.size; i++) this.grad[i] += t.grad[bucketOf(i)] * scale;
    };
    return t;
  }

  // ---- softmax over last axis (numerically stable: subtract row max) -------
  // INVARIANT: subtracting a per-row constant does not change softmax output but keeps
  //   exp() arguments <= 0, avoiding overflow. The Jacobian-vector product is folded
  //   into one closure: dx_i = y_i (g_i - sum_j y_j g_j).
  softmax(): Tensor {
    const rank = this.shape.length;
    if (rank < 1) throw new Error("softmax needs >=1D");
    const lastDim = this.shape[rank - 1];
    const rows = this.size / lastDim;
    const out = new Float64Array(this.size);
    for (let r = 0; r < rows; r++) {
      const off = r * lastDim;
      let max = -Infinity;
      for (let j = 0; j < lastDim; j++) if (this.data[off + j] > max) max = this.data[off + j];
      let denom = 0;
      for (let j = 0; j < lastDim; j++) {
        const e = Math.exp(this.data[off + j] - max);
        out[off + j] = e;
        denom += e;
      }
      for (let j = 0; j < lastDim; j++) out[off + j] /= denom;
    }
    const t = new Tensor(out, this.shape.slice(), [this], "softmax");
    t._backward = () => {
      for (let r = 0; r < rows; r++) {
        const off = r * lastDim;
        let dot = 0;
        for (let j = 0; j < lastDim; j++) dot += out[off + j] * t.grad[off + j];
        for (let j = 0; j < lastDim; j++) this.grad[off + j] += out[off + j] * (t.grad[off + j] - dot);
      }
    };
    return t;
  }

  // ---- cross-entropy from LOGITS over last axis ---------------------------
  // WHY logits version (not log(softmax) composed): fusing log-sum-exp into one node is
  //   numerically stable AND gives the clean adjoint dlogits = softmax - onehot, which
  //   is the single most important gradient in the book. Returns mean CE over rows.
  // `targets` are integer class indices, length = number of rows.
  crossEntropy(targets: number[]): Tensor {
    const rank = this.shape.length;
    const C = this.shape[rank - 1];
    const rows = this.size / C;
    if (targets.length !== rows) throw new Error(`CE: ${targets.length} targets != ${rows} rows`);
    const probs = new Float64Array(this.size); // cache softmax for backward
    let lossSum = 0;
    for (let r = 0; r < rows; r++) {
      const off = r * C;
      let max = -Infinity;
      for (let j = 0; j < C; j++) if (this.data[off + j] > max) max = this.data[off + j];
      let denom = 0;
      for (let j = 0; j < C; j++) {
        const e = Math.exp(this.data[off + j] - max);
        probs[off + j] = e;
        denom += e;
      }
      for (let j = 0; j < C; j++) probs[off + j] /= denom;
      const tgt = targets[r];
      if (tgt < 0 || tgt >= C) throw new Error(`CE: target ${tgt} out of [0,${C})`);
      lossSum += -Math.log(probs[off + tgt] + 1e-12);
    }
    const t = new Tensor(Float64Array.from([lossSum / rows]), [1], [this], "ce");
    t._backward = () => {
      const g = t.grad[0] / rows; // mean -> divide by rows
      for (let r = 0; r < rows; r++) {
        const off = r * C;
        const tgt = targets[r];
        for (let j = 0; j < C; j++) this.grad[off + j] += (probs[off + j] - (j === tgt ? 1 : 0)) * g;
      }
    };
    return t;
  }

  // ---- reverse-mode backward ----------------------------------------------
  // Build reverse-topological order from THIS node, seed its grad = 1 (scalar), then
  // fire each node's _backward exactly once in that order.
  // INVARIANT: only call on a scalar (size 1) loss; otherwise the seed is ambiguous.
  backward(): void {
    if (this.size !== 1) throw new Error(`backward() expects scalar loss; got shape [${this.shape}]`);
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const build = (v: Tensor): void => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);
    this.grad[0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }

  // zero this node's grad in place (used by nn.zeroGrad over params).
  zeroGrad(): void {
    this.grad.fill(0);
  }
}

// Indexer factory: given a (possibly smaller) operand shape and the broadcast output
// shape, return a function mapping an output flat index -> the operand flat index,
// honoring size-1 broadcast dims (which contribute stride 0).
function bIndexer(opShape: Shape, outShape: Shape): (oi: number, oStr: number[], oShape: Shape) => number {
  if (shapeEq(opShape, outShape)) return (oi) => oi; // fast path: identical shape
  const opStr = strides(opShape);
  const off = outShape.length - opShape.length;
  return (oi, oStr, oShape) => {
    let rem = oi;
    let idx = 0;
    for (let d = 0; d < oShape.length; d++) {
      const coord = Math.floor(rem / oStr[d]) % oShape[d];
      rem -= coord * oStr[d];
      const od = d - off;
      if (od < 0) continue; // dim only exists in output -> operand has none
      const dim = opShape[od];
      idx += (dim === 1 ? 0 : coord) * opStr[od]; // size-1 -> stride 0 (broadcast)
    }
    return idx;
  };
}

// Sum grad `g` (shape gShape) down to `targetShape` (adjoint of broadcast).
function unbroadcast(g: Float64Array, gShape: Shape, targetShape: Shape): Float64Array {
  if (shapeEq(gShape, targetShape)) return g;
  const out = new Float64Array(numel(targetShape));
  const gStr = strides(gShape);
  const tStr = strides(targetShape);
  const off = gShape.length - targetShape.length;
  for (let gi = 0; gi < g.length; gi++) {
    let rem = gi;
    let ti = 0;
    for (let d = 0; d < gShape.length; d++) {
      const coord = Math.floor(rem / gStr[d]) % gShape[d];
      rem -= coord * gStr[d];
      const td = d - off;
      if (td < 0) continue; // extra leading dim in g -> collapses into target[0..]
      const dim = targetShape[td];
      ti += (dim === 1 ? 0 : coord) * tStr[td]; // broadcast dim -> all map to index 0
    }
    out[ti] += g[gi];
  }
  return out;
}
