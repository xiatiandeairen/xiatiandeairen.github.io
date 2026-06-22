// core/autograd.ts — The single source of truth for the whole book's compute engine.
//
// WHY this file is frozen after the early chapters: every later stage (pooling, batchnorm,
//   residual, the tiny-CNN) is built ONLY from the operators below. If the contract here
//   shifts, every downstream "honest number" silently changes. So we build it carefully
//   once and never special-case it later.
//
// TWO LEVELS, ONE IDEA (reverse-mode autodiff):
//   - Value: a scalar node. Pedagogical — easy to see the graph by hand.
//   - Tensor: an n-d node over a flat Float64Array. The real workhorse.
//   Both record a `_backward` closure that, given the node's own grad, scatters grad
//   into its parents. backward() runs reverse-topological order so each node's grad is
//   fully accumulated before it is used.
//
// VISION EXTENSION vs the tiny-dl sibling: this file adds conv2d / maxpool2d / avgpool2d /
//   flatten. conv2d is implemented via IM2COL: it unfolds each sliding window into a row
//   so the convolution becomes one matmul. We do this ON PURPOSE — it lets the conv
//   gradient REUSE the already-tested matmul adjoint (dA = dC@B^T, dB = A^T@dC) instead
//   of hand-deriving a 4-D conv backward. The only conv-specific adjoint we hand-write is
//   col2im (the transpose of im2col), which is small and isolated.
//
// CORE INVARIANT — grads ACCUMULATE (+=), never overwrite (=):
//   A parameter can feed multiple consumers (weight sharing; broadcasting / im2col fan one
//   value into many outputs). Reverse-mode requires summing contributions from every path.
//   col2im SUMS overlapping window contributions back to the same pixel — skipping that
//   sum is the classic conv-backward bug.

// ============================================================================
// Value — scalar reverse-mode autodiff ("see the graph by hand")
// ============================================================================

export class Value {
  data: number;
  grad: number;
  _backward: () => void;
  _prev: Set<Value>;
  op: string;

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

  /** Reverse-topological backward pass. Seed self.grad=1 since d(self)/d(self)=1. */
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
// Tensor — n-d reverse-mode autodiff over a flat Float64Array
// ============================================================================
//
// LAYOUT: data is row-major contiguous. We keep an explicit `strides` array so future
//   ops could be views in principle; for teaching clarity most ops here materialize fresh
//   contiguous buffers (correctness over zero-copy).
// IMAGE LAYOUT CONVENTION: images are NCHW (batch, channels, height, width) and conv
//   weights are OIHW (outCh, inCh, kH, kW) — the same layout PyTorch uses, so the reader's
//   mental model transfers.
// PRECISION: Float64 everywhere — deliberately. f32 would be faster and more realistic,
//   but f64 makes gradCheck pass at < 1e-6 and removes "is this a real bug or just f32
//   noise?" ambiguity. The honesty trade-off: timings are pessimistic vs a real f32/SIMD
//   framework; the transferable signal is relative trends, not absolute ms.

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
  // Supports (a) identical shapes and (b) bias-style broadcast where `other` is a smaller
  // buffer tiled across `this` (e.g. a per-channel (1,C,1,1) bias broadcast over NCHW).
  // The adjoint of broadcast is sum-over-broadcast-positions (the += into b.grad[bi]).
  private static elementwise(
    a: Tensor,
    b: Tensor,
    fwd: (x: number, y: number) => number,
    back: (x: number, y: number, g: number) => [number, number],
    op: string,
  ): Tensor {
    const broadcasting = !shapeEq(a.shape, b.shape);
    if (broadcasting) {
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

  /** Multiply by a JS scalar. Common enough to deserve its own fast path. */
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

  // ---- matmul: (m,k) @ (k,n) -> (m,n) ----
  // WHY only 2-D: batched/strided matmul is a generalization we reach by RESHAPING into
  //   2-D, so keeping the core op 2-D keeps its adjoint dead-simple:
  //   dA = dC @ B^T ,  dB = A^T @ dC . Those two identities ARE the chain rule for matmul.
  //   conv2d below reshapes its problem into exactly this op.
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

  // ---- shape ops ----
  /** Transpose a 2-D tensor. Materializes a fresh buffer (no view) for clarity.
   *  Adjoint: transpose the grad back. Used by im2col's matmul plumbing. */
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

  /** Reshape with the same element count. Data ref is shared in forward (we never mutate
   *  data), grads flow straight through 1:1. flatten() builds on this. */
  reshape(newShape: number[]): Tensor {
    if (prod(newShape) !== this.size)
      throw new Error(`reshape: ${this.shape} -> ${newShape} changes element count`);
    const t = new Tensor(this.data, newShape, [this], "reshape");
    t._backward = () => {
      for (let i = 0; i < this.size; i++) this.grad[i] += t.grad[i];
    };
    return t;
  }

  /** Flatten NCHW -> (N, C*H*W). The bridge between the conv stack and the Linear head.
   *  It's just a reshape; grads flow 1:1. We keep it named because it's a conceptual
   *  boundary the book talks about ("where the spatial structure is discarded"). */
  flatten(): Tensor {
    if (this.shape.length !== 4) throw new Error(`flatten: expected NCHW 4-D, got ${this.shape}`);
    const [n, c, h, w] = this.shape;
    return this.reshape([n, c * h * w]);
  }

  /**
   * Broadcast a (1, n) row tensor up to (rows, n). Explicit op so its adjoint
   * (sum the grad back down to one row) lives in one tested place. Used by Linear's bias.
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
   * Enforces the scalar-loss convention: loss is always a scalar, backward() is called on
   * it. For non-scalar outputs the caller must seed .grad manually and drive topo itself.
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
        `backward(): can only start from a scalar loss (shape [1]); got ${this.shape}.`,
      );
    }
    this.grad[0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

// ============================================================================
// im2col / col2im — the workhorse pair that turns convolution into matmul
// ============================================================================
//
// WHY im2col at all: a direct 7-loop conv is correct but its backward must be hand-derived
//   over 4 dims. Instead we UNFOLD: for every output position we copy the inputs that the
//   kernel would touch into one row of a 2-D matrix. Then conv = (cols matrix) @ (weights
//   reshaped to a matrix). The matmul adjoint is already tested, so the ONLY new gradient
//   is col2im = the transpose of the unfold, which is small and local.
//
// SHAPES (single image, generalized to batch by stacking rows):
//   input  N×C×H×W
//   cols   (N·outH·outW) × (C·kH·kW)     -- one row per output pixel per image
//   wMat   (C·kH·kW) × outC               -- weights flattened, transposed to (in, out)
//   out    (N·outH·outW) × outC -> reshape -> N×outC×outH×outW
//
// FAILURE MODE col2im guards: overlapping windows touch the same input pixel many times;
//   their grads must SUM (+=). A plain assignment would keep only the last window and
//   silently halve/quarter input grads where receptive fields overlap.

export interface Conv2dParams {
  stride: number;
  padding: number;
}

/** Output spatial size for one axis. Floor division is the standard "valid + pad" formula;
 *  a non-integer result means the kernel doesn't tile the padded input evenly. */
export function convOutSize(inSize: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((inSize + 2 * padding - kernel) / stride) + 1;
}

/**
 * im2col with autodiff. Input NCHW -> cols (N*outH*outW, C*kH*kW).
 * The returned Tensor's backward routes col-grads back to the padded input via col2im
 * (summing overlaps), then strips the padding region. Zero-padded reads contribute no
 * grad to any real pixel (they read a constant 0), which is automatically handled because
 * we only scatter grads for in-bounds source pixels.
 */
export function im2col(input: Tensor, kH: number, kW: number, p: Conv2dParams): Tensor {
  if (input.shape.length !== 4) throw new Error(`im2col: expected NCHW, got ${input.shape}`);
  const [N, C, H, W] = input.shape;
  const { stride, padding } = p;
  const outH = convOutSize(H, kH, stride, padding);
  const outW = convOutSize(W, kW, stride, padding);
  if (outH <= 0 || outW <= 0) {
    throw new Error(`im2col: non-positive output ${outH}x${outW} for input ${H}x${W} k=${kH}x${kW}`);
  }
  const patch = C * kH * kW;
  const rows = N * outH * outW;
  const cols = new Float64Array(rows * patch);

  // Forward unfold: for each (n, oy, ox) row, walk (c, ky, kx) columns and copy the
  // source pixel, treating out-of-bounds (padding) reads as 0.
  for (let n = 0; n < N; n++) {
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        const row = (n * outH + oy) * outW + ox;
        let col = 0;
        for (let c = 0; c < C; c++) {
          for (let ky = 0; ky < kH; ky++) {
            const iy = oy * stride + ky - padding;
            for (let kx = 0; kx < kW; kx++) {
              const ix = ox * stride + kx - padding;
              let v = 0;
              if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                v = input.data[((n * C + c) * H + iy) * W + ix];
              }
              cols[row * patch + col] = v;
              col++;
            }
          }
        }
      }
    }
  }

  const t = new Tensor(cols, [rows, patch], [input], "im2col");
  t._backward = () => {
    // col2im: scatter each col-grad back to its source pixel, SUMMING overlaps.
    for (let n = 0; n < N; n++) {
      for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
          const row = (n * outH + oy) * outW + ox;
          let col = 0;
          for (let c = 0; c < C; c++) {
            for (let ky = 0; ky < kH; ky++) {
              const iy = oy * stride + ky - padding;
              for (let kx = 0; kx < kW; kx++) {
                const ix = ox * stride + kx - padding;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                  input.grad[((n * C + c) * H + iy) * W + ix] += t.grad[row * patch + col];
                }
                col++;
              }
            }
          }
        }
      }
    }
  };
  return t;
}

/**
 * conv2d via im2col. input NCHW, weight OIHW, optional bias (outC,).
 * Returns NCHW output. The whole forward+backward is COMPOSED from im2col + matmul +
 * (a tiny per-channel bias add), so its gradient is correct by construction provided
 * those two ops are correct (and gradCheck verifies they are).
 *
 * INVARIANT: weight.shape[1] (inCh of the kernel) MUST equal input.shape[1] (image
 *   channels), else the kernel can't see every input channel — we throw rather than
 *   silently truncating.
 */
export function conv2d(input: Tensor, weight: Tensor, bias: Tensor | null, p: Conv2dParams): Tensor {
  if (weight.shape.length !== 4) throw new Error(`conv2d: weight must be OIHW, got ${weight.shape}`);
  const [outC, inC, kH, kW] = weight.shape;
  const [N, C, H, W] = input.shape;
  if (C !== inC) throw new Error(`conv2d: input channels ${C} != weight inCh ${inC}`);
  const outH = convOutSize(H, kH, p.stride, p.padding);
  const outW = convOutSize(W, kW, p.stride, p.padding);

  // cols: (N*outH*outW, inC*kH*kW). wMat: (inC*kH*kW, outC) = weight reshaped+transposed.
  const cols = im2col(input, kH, kW, p);
  const wMat = weight.reshape([outC, inC * kH * kW]).transpose(); // (patch, outC)
  let outMat = cols.matmul(wMat); // (N*outH*outW, outC)

  if (bias) {
    if (bias.shape.length !== 1 || bias.shape[0] !== outC)
      throw new Error(`conv2d: bias must be (outC=${outC},), got ${bias.shape}`);
    // bias broadcasts over every output pixel: reshape to a (1, outC) row and tile.
    outMat = outMat.add(bias.reshape([1, outC]).broadcastRow(outMat.shape[0]));
  }

  // (N*outH*outW, outC) -> N×outC×outH×outW. We need a channel-major layout, but outMat is
  // pixel-major (channel is the fast axis). So transpose the per-image block: do it via a
  // reshape to (N, outH*outW, outC) then a manual permute. Simplest correct route: reshape
  // to (N*outH*outW, outC), then scatter. We use a small explicit permute op below.
  return permutePixelMajorToNCHW(outMat, N, outC, outH, outW);
}

/**
 * Permute a pixel-major matrix (N*outH*outW, outC) into NCHW. Hand-written tiny op with a
 * matching adjoint. WHY not compose from transpose/reshape: those are 2-D only and this is
 * a 3-axis permute per image; one explicit loop is clearer and keeps the adjoint obvious
 * (the inverse index map). Both directions are pure index gymnastics, no arithmetic, so
 * there's nothing to get numerically wrong — only the index map.
 */
function permutePixelMajorToNCHW(src: Tensor, N: number, outC: number, outH: number, outW: number): Tensor {
  const out = new Float64Array(N * outC * outH * outW);
  const HW = outH * outW;
  for (let n = 0; n < N; n++) {
    for (let s = 0; s < HW; s++) {
      for (let c = 0; c < outC; c++) {
        // src row = n*HW + s, col = c. dst = ((n*outC + c)*outH ... ) flattened = n,c,s.
        out[(n * outC + c) * HW + s] = src.data[(n * HW + s) * outC + c];
      }
    }
  }
  const t = new Tensor(out, [N, outC, outH, outW], [src], "permute_nchw");
  t._backward = () => {
    for (let n = 0; n < N; n++) {
      for (let s = 0; s < HW; s++) {
        for (let c = 0; c < outC; c++) {
          src.grad[(n * HW + s) * outC + c] += t.grad[(n * outC + c) * HW + s];
        }
      }
    }
  };
  return t;
}

// ============================================================================
// Pooling — maxpool2d / avgpool2d with autodiff
// ============================================================================
//
// WHY pooling lives in autograd not nn: it's an op with a non-trivial adjoint, same tier
//   as conv. nn.MaxPool2d is a thin Module wrapper over these.
// MAXPOOL ADJOINT: route the whole output grad to the SINGLE input that was the argmax of
//   that window (ties -> first). Every non-max input in the window gets 0 — this is why
//   maxpool is "gradient-sparse" and a talking point in the book.
// AVGPOOL ADJOINT: spread the output grad EQUALLY (÷ poolH·poolW) to every input in the
//   window. Overlapping windows (stride < kernel) accumulate — the += handles it.

export interface Pool2dParams {
  kernel: number;
  stride: number;
}

export function maxpool2d(input: Tensor, p: Pool2dParams): Tensor {
  if (input.shape.length !== 4) throw new Error(`maxpool2d: expected NCHW, got ${input.shape}`);
  const [N, C, H, W] = input.shape;
  const { kernel, stride } = p;
  const outH = convOutSize(H, kernel, stride, 0);
  const outW = convOutSize(W, kernel, stride, 0);
  const out = new Float64Array(N * C * outH * outW);
  // store the flat input index of each window's argmax, so backward routes grad there.
  const argmax = new Int32Array(N * C * outH * outW);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      const inBase = (n * C + c) * H * W;
      for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
          let best = -Infinity;
          let bestIdx = -1;
          for (let ky = 0; ky < kernel; ky++) {
            const iy = oy * stride + ky;
            for (let kx = 0; kx < kernel; kx++) {
              const ix = ox * stride + kx;
              const idx = inBase + iy * W + ix;
              if (input.data[idx] > best) {
                best = input.data[idx];
                bestIdx = idx;
              }
            }
          }
          const o = ((n * C + c) * outH + oy) * outW + ox;
          out[o] = best;
          argmax[o] = bestIdx;
        }
      }
    }
  }
  const t = new Tensor(out, [N, C, outH, outW], [input], "maxpool2d");
  t._backward = () => {
    for (let o = 0; o < t.size; o++) input.grad[argmax[o]] += t.grad[o];
  };
  return t;
}

export function avgpool2d(input: Tensor, p: Pool2dParams): Tensor {
  if (input.shape.length !== 4) throw new Error(`avgpool2d: expected NCHW, got ${input.shape}`);
  const [N, C, H, W] = input.shape;
  const { kernel, stride } = p;
  const outH = convOutSize(H, kernel, stride, 0);
  const outW = convOutSize(W, kernel, stride, 0);
  const out = new Float64Array(N * C * outH * outW);
  const area = kernel * kernel;
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      const inBase = (n * C + c) * H * W;
      for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
          let s = 0;
          for (let ky = 0; ky < kernel; ky++) {
            const iy = oy * stride + ky;
            for (let kx = 0; kx < kernel; kx++) s += input.data[inBase + iy * W + (ox * stride + kx)];
          }
          out[((n * C + c) * outH + oy) * outW + ox] = s / area;
        }
      }
    }
  }
  const t = new Tensor(out, [N, C, outH, outW], [input], "avgpool2d");
  t._backward = () => {
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        const inBase = (n * C + c) * H * W;
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            const g = t.grad[((n * C + c) * outH + oy) * outW + ox] / area;
            for (let ky = 0; ky < kernel; ky++) {
              const iy = oy * stride + ky;
              for (let kx = 0; kx < kernel; kx++) input.grad[inBase + iy * W + (ox * stride + kx)] += g;
            }
          }
        }
      }
    }
  };
  return t;
}

// ============================================================================
// noGrad — inference switch
// ============================================================================
//
// WHY: at inference we still call the same forward ops, but BatchNorm must switch from
//   batch statistics to its running estimates. Stages/Modules check noGradActive() (or
//   their own training flag) to pick the inference path. Keeping ONE Tensor class (vs
//   train/eval variants) is the teaching simplification.

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
