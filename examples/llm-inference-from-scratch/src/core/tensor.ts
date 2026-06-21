// core/tensor.ts — the toy tensor kernels every stage in this book bottoms out in.
//
// Why hand-rolled Float64Array kernels instead of a BLAS / ONNX / wasm backend:
// this book is about the *inference engine* (KV cache, batching, paging,
// speculation, quantization), not about kernel micro-optimization. A reader must
// be able to (a) read every multiply, (b) count FLOPs by eye, and (c) trust that
// two runs produce bit-for-bit identical numbers. A BLAS backend would give us
// speed but steal all three. So every op here is an explicit loop. It is slow on
// purpose; absolute tok/s is therefore pessimistic, but every *relative* speedup
// a later stage reports (cache vs no-cache, batched vs serial) is real and
// transfers to a real engine.
//
// Layout invariant: every Tensor is row-major. A [m, n] matrix stores element
// (i, j) at data[i * n + j]. Every kernel below assumes and preserves this.
// Violating it is the single most common bug when extending these kernels —
// there are no shape-broadcasting niceties to paper over a transposed input.

export type Tensor = { data: Float64Array; shape: number[] };

// Allocate a zeroed tensor. Float64 (not Float32) so that quantization stages
// have a clean high-precision reference to compare against: the whole point of
// stage07 is measuring the L∞ drift float32/int8 introduces, which only means
// something if the baseline is meaningfully more precise.
export function zeros(shape: number[]): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  return { data: new Float64Array(n), shape: [...shape] };
}

export function tensor(data: number[] | Float64Array, shape: number[]): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  if (data.length !== n) {
    // Loud failure: a length/shape mismatch is a construction bug, not a runtime
    // condition. Catching it here saves hours of debugging silently-misread rows.
    throw new Error(`tensor: data length ${data.length} != shape product ${n} (${shape})`);
  }
  return { data: data instanceof Float64Array ? data : Float64Array.from(data), shape: [...shape] };
}

// Matrix multiply: A[m,k] @ B[k,n] -> C[m,n], all row-major.
//
// This is the engine's hot loop — every attention projection and FFN layer is a
// matmul, and in a real model it is >90% of the FLOPs. We keep the textbook ikj
// loop order (not ijk): ikj walks B and C contiguously in the inner loop, which
// is markedly cache-friendlier than ijk even in JS. We do NOT tile or unroll —
// readability wins, and the book's speedups never come from kernel tricks.
//
// Failure mode: inner dimensions must match. A mismatch here is the classic
// "I forgot to transpose the weight" bug; we throw rather than read garbage.
export function matmul(a: Tensor, b: Tensor): Tensor {
  const [m, k] = a.shape;
  const [k2, n] = b.shape;
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error(`matmul: both args must be 2-D, got ${a.shape} and ${b.shape}`);
  }
  if (k !== k2) {
    throw new Error(`matmul: inner dims disagree: ${a.shape} @ ${b.shape}`);
  }
  const out = new Float64Array(m * n);
  const A = a.data;
  const B = b.data;
  for (let i = 0; i < m; i++) {
    const aRow = i * k;
    const cRow = i * n;
    for (let p = 0; p < k; p++) {
      const aip = A[aRow + p];
      // Skipping the multiply when aip === 0 is NOT done on purpose: it would make
      // timing data-dependent and break the "every run costs the same" promise the
      // benchmark stage relies on.
      const bRow = p * n;
      for (let j = 0; j < n; j++) {
        out[cRow + j] += aip * B[bRow + j];
      }
    }
  }
  return { data: out, shape: [m, n] };
}

// Add a per-column bias vector to every row of x[m, n]. In-place on a copy.
// Invariant: bias.length === n. Used after every projection.
export function addBias(x: Tensor, bias: Tensor): Tensor {
  const [m, n] = x.shape;
  if (bias.data.length !== n) {
    throw new Error(`addBias: bias length ${bias.data.length} != cols ${n}`);
  }
  const out = new Float64Array(x.data.length);
  for (let i = 0; i < m; i++) {
    const row = i * n;
    for (let j = 0; j < n; j++) out[row + j] = x.data[row + j] + bias.data[j];
  }
  return { data: out, shape: [m, n] };
}

// RMSNorm (the LLaMA-family normalizer): x_i * w_i / sqrt(mean(x^2) + eps).
//
// We call the export `layerNorm` to match the coreSpec name, but it is RMSNorm:
// no mean-subtraction, no bias. Modern decoder LLMs use RMSNorm because it is
// cheaper and empirically as good; matching that keeps the toy model honest.
//
// Why eps matters (a real failure mode): the +eps inside the sqrt is the only
// thing standing between you and a division by zero on an all-zero row (which the
// synthetic model can produce after an unlucky init). Set eps too small (1e-12)
// and a near-zero row still blows the result up to ~1e6 and then NaNs downstream;
// 1e-6..1e-5 is the safe band. We expose it so a stage can demo the NaN.
export function rmsNorm(x: Tensor, weight: Tensor, eps = 1e-6): Tensor {
  const [m, n] = x.shape;
  if (weight.data.length !== n) {
    throw new Error(`rmsNorm: weight length ${weight.data.length} != cols ${n}`);
  }
  const out = new Float64Array(x.data.length);
  for (let i = 0; i < m; i++) {
    const row = i * n;
    let ss = 0;
    for (let j = 0; j < n; j++) {
      const v = x.data[row + j];
      ss += v * v;
    }
    const inv = 1 / Math.sqrt(ss / n + eps);
    for (let j = 0; j < n; j++) out[row + j] = x.data[row + j] * inv * weight.data[j];
  }
  return { data: out, shape: [m, n] };
}

// alias kept for the coreSpec name; RMSNorm is the layer norm this book uses.
export const layerNorm = rmsNorm;

// Numerically-stable softmax over the last axis of a [rows, cols] tensor.
//
// THE canonical inference numerics lesson: never exponentiate raw logits. Real
// attention logits routinely reach +40..+80; exp(80) ≈ 5.5e34, well under
// float64's ~1.8e308 ceiling so float64 survives — but float32 (exp(89) = inf)
// and especially the quantized paths in stage07 do NOT. Subtracting the row max
// first makes the largest exponent exactly exp(0)=1, so the result is identical
// in exact arithmetic but never overflows. We keep the subtraction unconditional
// so the lesson is visible and the behavior is the same precision-to-precision.
export function softmax(x: Tensor): Tensor {
  const [rows, cols] = x.shape;
  const out = new Float64Array(x.data.length);
  for (let i = 0; i < rows; i++) {
    const row = i * cols;
    let max = -Infinity;
    for (let j = 0; j < cols; j++) if (x.data[row + j] > max) max = x.data[row + j];
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      const e = Math.exp(x.data[row + j] - max);
      out[row + j] = e;
      sum += e;
    }
    // sum >= 1 always (the max term contributes exactly 1), so this division is
    // safe even for an all -Infinity row (fully-masked) — that case yields NaN,
    // which is the correct loud signal that you masked away every position.
    const inv = 1 / sum;
    for (let j = 0; j < cols; j++) out[row + j] *= inv;
  }
  return { data: out, shape: [rows, cols] };
}

// SiLU / swish activation: x * sigmoid(x). The FFN nonlinearity in LLaMA-style
// gated MLPs. Element-wise, shape-preserving. sigmoid is written as the numerically
// stable two-branch form so a large negative x cannot exp-overflow into inf.
export function silu(x: Tensor): Tensor {
  const out = new Float64Array(x.data.length);
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i];
    const s = v >= 0 ? 1 / (1 + Math.exp(-v)) : (() => { const e = Math.exp(v); return e / (1 + e); })();
    out[i] = v * s;
  }
  return { data: out, shape: [...x.shape] };
}

// Rotary Position Embedding (RoPE), applied in-place to a query and key vector
// for ONE position. q and k are length dHead; pos is the absolute token index.
//
// Why RoPE and not learned/absolute embeddings: RoPE encodes position by *rotating*
// query/key pairs, so the attention score between positions i and j depends only
// on (i - j). That relative property is exactly what makes the KV cache work —
// a cached key computed at position 5 stays valid forever, because its rotation
// is absolute and frozen. This is the geometric reason stage02's cache is correct,
// so it lives in core, not in a stage.
//
// Invariant: dHead must be even (we rotate (2i, 2i+1) pairs). theta=10000 is the
// standard base. Failure mode: pass an odd dHead and the last dim is silently
// dropped — we guard against it.
export function rope(q: Float64Array, k: Float64Array, pos: number, theta = 10000): void {
  const d = q.length;
  if (d !== k.length) throw new Error(`rope: q/k length mismatch ${d} != ${k.length}`);
  if (d % 2 !== 0) throw new Error(`rope: dHead must be even, got ${d}`);
  for (let i = 0; i < d; i += 2) {
    const freq = 1 / Math.pow(theta, i / d);
    const angle = pos * freq;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const q0 = q[i];
    const q1 = q[i + 1];
    q[i] = q0 * cos - q1 * sin;
    q[i + 1] = q0 * sin + q1 * cos;
    const k0 = k[i];
    const k1 = k[i + 1];
    k[i] = k0 * cos - k1 * sin;
    k[i + 1] = k0 * sin + k1 * cos;
  }
}

// Build an additive causal mask of shape [seq, seq]: 0 on/below the diagonal,
// -Infinity above it. Added to attention logits BEFORE softmax so a token can
// never attend to its future. Using -Infinity (not a big negative like -1e9)
// makes the masked softmax weight exactly 0; -1e9 leaves a ~1e-9 leak that, over
// many layers, is a real (and maddening to find) correctness bug.
export function causalMask(seq: number): Tensor {
  const out = new Float64Array(seq * seq);
  for (let i = 0; i < seq; i++) {
    for (let j = 0; j < seq; j++) {
      out[i * seq + j] = j > i ? -Infinity : 0;
    }
  }
  return { data: out, shape: [seq, seq] };
}
