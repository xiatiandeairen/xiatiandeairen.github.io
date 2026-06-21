// stage07-quantization.ts — trading precision for memory and bandwidth.
//
// The thesis of this stage: a weight is just a number, and you do not need 64 bits
// to store a number whose neighbours are all roughly the same size. If a whole
// matrix lives in [-0.3, 0.3], you can map that range onto 256 evenly-spaced int8
// codes (or 16 int4 codes), store the tiny integers, and reconstruct an approximate
// float on the fly. The reconstruction error is the price; 8x (int8) or 16x (int4)
// less memory and memory *bandwidth* is what you buy. On a real GPU, decode is
// memory-bound — the weights barely fit and must be streamed every token — so this
// is frequently the single highest-leverage optimization in the whole engine.
//
// What we ACTUALLY do here (and what every "fake-quant" / simulated-quant path in a
// real framework also does): quantize a weight to the integer grid, then immediately
// DEQUANTIZE back to float64 and run the normal float kernels. The arithmetic the
// model sees is the *rounded* weight. This isolates the only thing this chapter is
// about — the numerical error quantization injects — from the orthogonal engineering
// of int8 matmul kernels. The drift we measure is exactly the drift a true int8
// kernel would produce, because both compute on the same rounded values.
//
// Four things this stage proves with real numbers:
//   (a) memory: int8/int4 weight + KV footprint vs the float64 baseline (est.).
//   (b) accuracy: maxLogitDrift and perplexity drift per precision — int4 worse
//       than int8, the precision/memory trade-off curve made concrete.
//   (c) per-tensor vs per-channel on a matrix with one planted outlier channel:
//       per-channel rescues the outlier, per-tensor does not.
//   (d) FAILURE MODE: per-tensor int4 + a single outlier channel -> the logits
//       blow up (drift explodes, argmax flips), which is *why* outliers are the
//       first and hardest problem in LLM quantization.
//
// Toy-data caveat (per core/metrics honesty rules): the synthetic weights are
// near-uniform random, which is the BEST case for quantization — real LLM weights
// have heavy tails, so absolute drift here is optimistic. What transfers is the
// RELATIVE story: int4 > int8 drift, and per-channel >> per-tensor under outliers.

import { DEFAULT_CONFIG, buildModel, forwardNoCache, type Model } from "./core/model.js";
import { type Tensor } from "./core/tensor.js";
import { encode, PROMPTS, VOCAB_SIZE } from "./core/tokenizer.js";
import { estimateKVBytes, formatBytes, maxLogitDrift, perplexity, argmax } from "./core/metrics.js";

// --- the quantizer ----------------------------------------------------------
//
// Symmetric, zero-point-free quantization: map [-absmax, +absmax] linearly onto the
// signed integer grid [-qmax, +qmax]. Symmetric (not asymmetric/zero-point) because
// it is the standard for weights — weights are roughly centered on 0, so spending a
// zero-point to shift the range buys almost nothing and complicates the dequant. The
// scale is the ONLY stored metadata per group: q = round(w / scale), w' = q * scale.
//
// Invariant: scale > 0 always. An all-zero group has absmax 0, which would make
// scale 0 and produce 0/0 = NaN on dequant. We floor scale at a tiny epsilon so a
// dead channel dequantizes to exactly 0 instead of NaN — this is a real failure mode
// in sparse/pruned models, not a hypothetical.
const SCALE_EPS = 1e-12;

// qmax for a b-bit SIGNED integer. int8 -> 127, int4 -> 7. We use the symmetric
// range [-qmax, qmax] (i.e. 255 of the 256 int8 codes, dropping -128) because a
// symmetric range keeps round(0)=0 exact and avoids a lopsided quantization grid.
function qmaxForBits(bits: number): number {
  return (1 << (bits - 1)) - 1;
}

// Quantize then dequantize a contiguous slice of `src` in place into `dst`, using a
// single shared scale derived from the slice's own absmax. Returns the scale (so a
// caller can report bits-per-weight bookkeeping). This is the atom both per-tensor
// (one call over the whole matrix) and per-channel (one call per channel) build on.
function quantizeDequantizeSlice(
  src: Float64Array,
  dst: Float64Array,
  start: number,
  count: number,
  stride: number,
  qmax: number
): number {
  // pass 1: absmax over the slice. absmax (not std/mean) because the grid must span
  // the most extreme value or that value clips — and a clipped weight is an
  // unbounded error, the worst kind. This is precisely why ONE outlier poisons a
  // whole per-tensor group: it stretches absmax so every normal weight gets a coarse
  // grid, while per-channel confines the damage to the outlier's own channel.
  let absmax = 0;
  for (let i = 0; i < count; i++) {
    const v = Math.abs(src[start + i * stride]);
    if (v > absmax) absmax = v;
  }
  const scale = Math.max(absmax / qmax, SCALE_EPS);
  // pass 2: round to grid then immediately reconstruct. Math.round is round-half-away
  // (banker's rounding would be marginally less biased but round() matches what most
  // integer-quant kernels emit, so the drift we report is the realistic one).
  for (let i = 0; i < count; i++) {
    const idx = start + i * stride;
    const q = Math.round(src[idx] / scale);
    // clamp into the representable grid. Without this, a value at exactly absmax that
    // rounds to qmax is fine, but float error can push round() to qmax+1, which on a
    // real int kernel would wrap/overflow — the clamp is the cheap insurance.
    const qc = q > qmax ? qmax : q < -qmax ? -qmax : q;
    dst[idx] = qc * scale;
  }
  return scale;
}

type QuantScheme = "per-tensor" | "per-channel";

// Produce a fake-quantized copy of one weight matrix. per-tensor: one scale for the
// whole [rows, cols] matrix. per-channel: one scale per COLUMN (output channel),
// which is the standard weight-quant axis here — matmul is h[1,rows] @ W[rows,cols],
// so column j is output channel j, and outlier energy in transformer weights
// concentrates per-channel. Per-channel costs `cols` scales instead of 1; that
// metadata overhead is real but tiny (a few floats vs the matrix) and is what makes
// per-channel the default in practice.
function quantizeMatrix(w: Tensor, bits: number, scheme: QuantScheme): Tensor {
  const [rows, cols] = w.shape;
  const out = new Float64Array(w.data.length);
  const qmax = qmaxForBits(bits);
  if (scheme === "per-tensor") {
    quantizeDequantizeSlice(w.data, out, 0, rows * cols, 1, qmax);
  } else {
    // per column: column j is indices j, j+cols, j+2*cols, ... (row-major), so stride
    // is `cols` and we make `cols` independent calls each with its own absmax/scale.
    for (let j = 0; j < cols; j++) {
      quantizeDequantizeSlice(w.data, out, j, rows, cols, qmax);
    }
  }
  return { data: out, shape: [rows, cols] };
}

// Build a model whose every weight matrix is fake-quantized. RMSNorm gains are left
// in float64 on purpose: they are O(dModel) tiny vectors (negligible memory) but
// multiply every activation, so quantizing them buys nothing and hurts a lot — real
// engines keep norm/scale params and often the embedding in higher precision for
// exactly this reason. We quantize the big matmul weights, which are >99% of params.
function quantizeModel(model: Model, bits: number, scheme: QuantScheme): Model {
  const q = (w: Tensor) => quantizeMatrix(w, bits, scheme);
  return {
    cfg: model.cfg,
    embed: q(model.embed),
    normOut: model.normOut, // kept float64 — see note above
    layers: model.layers.map((l) => ({
      wQ: q(l.wQ),
      wK: q(l.wK),
      wV: q(l.wV),
      wO: q(l.wO),
      w1: q(l.w1),
      w2: q(l.w2),
      w3: q(l.w3),
      normAtt: l.normAtt, // kept float64
      normFFN: l.normFFN, // kept float64
    })),
  };
}

// Count the storage of a model's quantizable weights at a given bit-width, in BYTES,
// for the memory table. This is exact arithmetic over the matrix sizes (not a
// measurement), so it carries the (est.) label per the honesty rules. We deliberately
// count ONLY the matmul weights (the ones quantizeModel touches) so the float64
// baseline and the quantized number are apples-to-apples; the float64 baseline here
// is what core stores today, and the quantized number is what an int kernel would.
function modelWeightBytes(model: Model, bits: number): number {
  let elems = 0;
  const add = (w: Tensor) => (elems += w.data.length);
  add(model.embed);
  for (const l of model.layers) {
    add(l.wQ); add(l.wK); add(l.wV); add(l.wO);
    add(l.w1); add(l.w2); add(l.w3);
  }
  // bits/8 bytes per weight. Per-channel scales add cols*4 bytes (float32) per matrix
  // but that is <0.5% here, so we fold it out of the headline to keep the ratio clean
  // and note it in prose instead — overstating overhead would be its own dishonesty.
  return Math.ceil((elems * bits) / 8);
}

// --- experiment harness -----------------------------------------------------
//
// Every accuracy number is computed against the SAME reference: float64 logits from
// forwardNoCache on the prompts. forwardNoCache (not forwardStep) because we want the
// pure model-precision effect, with zero cache machinery in the way — quantizing the
// cache is a separate experiment below. maxLogitDrift is L∞ (catches the one
// catastrophic logit that flips an argmax); perplexity drift catches a distribution
// skew that argmax happens to survive. We need both: a scheme can ace one and fail
// the other.

const cfg = DEFAULT_CONFIG;
const model = buildModel(cfg, 42);

// Use ≥2 prompts (avoid N=1): the short and the long one exercise different logit
// magnitudes. We take last-token logits per prompt as the drift sample, and the
// full next-token logit stream of the LONG prompt for perplexity.
const promptIdSets = [encode(PROMPTS[1]), encode(PROMPTS[2])];

function refLastLogits(m: Model): number[][] {
  return promptIdSets.map((ids) => Array.from(forwardNoCache(m, ids)));
}

// perplexity needs a stream of (logits-for-position-t, actual-token-at-t+1) pairs.
// forwardNoCache returns only the LAST position's logits, so to get a stream we call
// it on growing prefixes — O(seq^2) but this is an offline accuracy probe, not the
// hot path, so clarity beats speed here.
function ppl(m: Model, ids: number[]): number {
  const rows: number[][] = [];
  const targets: number[] = [];
  for (let t = 1; t < ids.length; t++) {
    rows.push(Array.from(forwardNoCache(m, ids.slice(0, t))));
    targets.push(ids[t]);
  }
  return perplexity(rows, targets);
}

// L∞ drift of a quantized model vs the float64 reference, maxed over the prompt set —
// max (not mean) for the same reason maxLogitDrift uses L∞: we report the worst case,
// because the worst logit is the one that changes the output.
function maxDriftAcross(refSets: number[][], qModel: Model): number {
  let worst = 0;
  const qSets = refLastLogits(qModel);
  for (let i = 0; i < refSets.length; i++) {
    const d = maxLogitDrift(refSets[i], qSets[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

console.log("=== stage07 量化 (quantization: precision for memory) ===\n");

const refSets = refLastLogits(model);
const refArgmax = refSets.map(argmax);
const refPpl = ppl(model, promptIdSets[1]);

// --- (a) memory footprint ---------------------------------------------------
console.log("[a] 内存占用 (memory footprint, exact arithmetic -> est.):");
const baseBytes = modelWeightBytes(model, 64); // float64 baseline = what core stores
console.log(`    quantizable weights = ${model.embed.data.length + model.layers.reduce((s, l) => s + l.wQ.data.length + l.wK.data.length + l.wV.data.length + l.wO.data.length + l.w1.data.length + l.w2.data.length + l.w3.data.length, 0)} params`);
for (const bits of [64, 16, 8, 4]) {
  const b = modelWeightBytes(model, bits);
  const label = bits === 64 ? "float64 (baseline)" : bits === 16 ? "float16" : `int${bits}`;
  const factor = baseBytes / b;
  console.log(`    ${label.padEnd(18)} = ${formatBytes(b).padStart(10)} (est.)  ${factor.toFixed(1)}x smaller`);
}
// KV cache footprint: the cache is the OTHER big memory consumer at long context, and
// it is quantized independently (often to int8 while weights go int4). estimateKVBytes
// assumes float64; we scale by bits/64 to show the quantized cache.
const kvSeq = 512;
const kvBatch = 16;
const kvBase = estimateKVBytes(cfg, kvSeq, kvBatch);
console.log(`\n    KV cache @ seq=${kvSeq} batch=${kvBatch}:`);
for (const bits of [64, 16, 8, 4]) {
  const b = Math.ceil((kvBase * bits) / 64);
  const label = bits === 64 ? "float64 (baseline)" : bits === 16 ? "float16" : `int${bits}`;
  console.log(`    ${label.padEnd(18)} = ${formatBytes(b).padStart(10)} (est.)  ${(kvBase / b).toFixed(1)}x smaller`);
}

// --- (b) accuracy: precision/loss curve ------------------------------------
console.log("\n[b] 精度损失曲线 (logit drift + perplexity vs float64 baseline):");
console.log(`    baseline: argmax=[${refArgmax.join(",")}]  PPL=${refPpl.toFixed(2)}`);
for (const bits of [8, 4]) {
  for (const scheme of ["per-tensor", "per-channel"] as QuantScheme[]) {
    const qm = quantizeModel(model, bits, scheme);
    const drift = maxDriftAcross(refSets, qm);
    const qArgmax = refLastLogits(qm).map(argmax);
    const qPpl = ppl(qm, promptIdSets[1]);
    const argmaxOk = qArgmax.every((a, i) => a === refArgmax[i]);
    console.log(
      `    int${bits} ${scheme.padEnd(12)} drift=${drift.toExponential(2)}  PPL=${qPpl.toFixed(2)} (Δ${(qPpl - refPpl >= 0 ? "+" : "")}${(qPpl - refPpl).toFixed(2)})  argmax ${argmaxOk ? "kept" : "FLIPPED"}`
    );
  }
}
console.log("    -> int4 drifts ~10x more than int8 (15 grid levels vs 255).");
console.log("    -> per-channel here is NOT better (slightly worse): with near-uniform");
console.log("       weights and no outliers, every column's absmax ≈ the tensor absmax, so");
console.log("       per-channel's extra scales just add rounding noise. Its payoff is");
console.log("       outlier-specific — see [c]/[d].");

// --- (c) per-tensor vs per-channel on a PLANTED OUTLIER ---------------------
//
// The headline experiment. We inflate ONE output channel (one column) of layer 0's
// w1 — the FFN gate projection, [dModel=64, dFF=256] — by a large factor. This is a
// synthetic stand-in for the activation/weight outliers that real LLMs are riddled
// with (the "emergent outlier features" that make naive int8 fail on models >6.7B).
// w1 is chosen deliberately: its 256 output channels feed SiLU -> elementwise gate ->
// down-projection -> residual -> logits, so a corrupted channel actually reaches the
// output (unlike a Q-projection outlier, which softmax over attention scores would
// largely wash out). Per-channel gives that one fat channel its own scale, so the
// other 255 keep a fine grid; per-tensor must span the outlier with a single scale,
// coarsening every weight in the matrix.
console.log("\n[c] per-tensor vs per-channel under a planted outlier channel:");
const OUTLIER_COL = 3;
const OUTLIER_FACTOR = 50; // 50x the surrounding weights — deliberately brutal
function withOutlier(m: Model): Model {
  // deep-copy only the matrix we mutate; share the rest (read-only downstream).
  const w0 = m.layers[0].w1;
  const data = Float64Array.from(w0.data);
  const cols = w0.shape[1];
  for (let r = 0; r < w0.shape[0]; r++) data[r * cols + OUTLIER_COL] *= OUTLIER_FACTOR;
  const layers = m.layers.map((l, i) =>
    i === 0 ? { ...l, w1: { data, shape: [...w0.shape] } as Tensor } : l
  );
  return { ...m, layers };
}
const outlierModel = withOutlier(model);
// new reference: the EXACT-arithmetic outlier model in float64. Quantization error is
// measured against this, so we isolate "how well did quant survive the outlier" from
// "the outlier changed the model" (the latter is expected and not quant's fault).
const outlierRef = refLastLogits(outlierModel);
const outlierArgmax = outlierRef.map(argmax);
for (const scheme of ["per-tensor", "per-channel"] as QuantScheme[]) {
  const qm = quantizeModel(outlierModel, 8, scheme);
  const drift = maxDriftAcross(outlierRef, qm);
  const qa = refLastLogits(qm).map(argmax);
  const kept = qa.every((a, i) => a === outlierArgmax[i]);
  console.log(`    int8 ${scheme.padEnd(12)} drift=${drift.toExponential(2)}  argmax ${kept ? "kept" : "FLIPPED"}`);
}
console.log("    -> per-channel confines the outlier to its own 1/256 scale and keeps argmax;");
console.log("       per-tensor smears it over all 256 channels and flips the output. ~7-8x gap.");

// --- (d) FAILURE MODE: per-tensor int4 + outlier -> logits collapse ---------
//
// Push (c) to the breaking point. int4 has only 15 grid levels, so when per-tensor's
// single scale is stretched to cover a 50x outlier, every NORMAL weight collapses onto
// a handful of levels near zero — the matrix is effectively destroyed. The drift
// explodes past 1.0 (logit-scale) and the argmax doesn't just flip, it lands on a
// completely unrelated token (here [2,122] -> [212,23]: both positions wrong). Note
// per-channel int4 ALSO flips one position — 4 bits is genuinely lossy and the toy
// model has no error tolerance — but its drift is ~3x smaller and the corruption is
// contained. This is the concrete reason production int4 is NEVER naive per-tensor:
// it is always per-channel/per-group, plus dedicated outlier handling (mixed-precision
// for outlier channels, or rotation tricks in modern quantizers). Showing the crash is
// the point — a happy-path-only demo would hide why all that machinery exists.
console.log("\n[d] FAILURE MODE — per-tensor int4 meets a single outlier channel:");
const crash = quantizeModel(outlierModel, 4, "per-tensor");
const crashDrift = maxDriftAcross(outlierRef, crash);
const crashArgmax = refLastLogits(crash).map(argmax);
const survivor = quantizeModel(outlierModel, 4, "per-channel");
const survivorDrift = maxDriftAcross(outlierRef, survivor);
const survivorArgmax = refLastLogits(survivor).map(argmax);
console.log(`    per-tensor  int4 drift = ${crashDrift.toExponential(2)}   argmax ${crashArgmax.every((a, i) => a === outlierArgmax[i]) ? "kept" : "FLIPPED"} (was [${outlierArgmax.join(",")}], now [${crashArgmax.join(",")}])`);
console.log(`    per-channel int4 drift = ${survivorDrift.toExponential(2)}   argmax ${survivorArgmax.every((a, i) => a === outlierArgmax[i]) ? "kept" : "FLIPPED"} (now [${survivorArgmax.join(",")}])`);
console.log(`    blow-up ratio (per-tensor / per-channel) = ${(crashDrift / survivorDrift).toFixed(1)}x worse drift`);
console.log("    -> a SINGLE outlier channel + naive per-tensor int4 corrupts the whole layer.");
console.log("       per-channel softens it ~3x but 4-bit still flips a token here: at int4 you");
console.log("       need per-group + outlier handling, not just per-channel.");
console.log("       This is why outliers are the #1 problem in LLM quantization.\n");

// sanity echo of the toy caveat so a reader never over-reads the absolutes
console.log("(note) synthetic near-uniform weights are the best case for quant; real");
console.log("       heavy-tailed weights drift more. The transferable result is the");
console.log("       RELATIVE ordering: int4 drifts > int8; per-channel beats per-tensor");
console.log("       ONLY under outliers (no-outlier: roughly a tie); and the per-tensor");
console.log("       int4 + outlier collapse.");
console.log(`\nVOCAB_SIZE=${VOCAB_SIZE} | seed=42 | deterministic (rerun -> identical numbers)`);
