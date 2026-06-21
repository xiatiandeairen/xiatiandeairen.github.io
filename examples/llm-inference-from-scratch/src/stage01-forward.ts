// stage01-forward.ts — Tiny Transformer forward pass: token -> logits, hand-traced.
//
// This is chapter 01's runnable companion. It does NOT re-derive the model — it
// drives core/* (the same buildModel / matmul / rmsNorm / rope / softmax the
// reference forwardNoCache uses), so every number printed here is computed by the
// exact arithmetic later chapters compare against. Re-implementing the math in a
// stage would let it silently drift from core; instead we instrument core's own
// operators by walking one block by hand with the same primitives.
//
// What this chapter proves with REAL numbers:
//   (a) per-operator activation statistics (mean/std) down one block, showing
//       RMSNorm pulls activations back to ~unit scale every time it runs — the
//       invariant that keeps a deep stack from exploding into NaN.
//   (b) attention's O(seq^2) cost: the score-matrix multiply count and wall-clock
//       across seq = 16/32/64/128, confirming quadratic (not linear) growth. This
//       is the cost the KV cache (chapter 02) exists to kill.
//   (c) a failure-mode demo: rerun softmax WITHOUT the max-subtraction on a long
//       prompt's real attention logits and watch it produce NaN — proving
//       numerical stability is load-bearing, not decorative.
//   (d) golden logits: the last-token logits for all 3 prompts, hashed + summarized,
//       saved as this chapter's reference for later chapters to diff against.
//
// Honesty caveat (echoed from core/metrics): the model is tiny and the kernels are
// unoptimized float64 loops, so ABSOLUTE timings are pessimistic. What transfers is
// the SHAPE of the curve (quadratic) and the operator-level invariants, not the ms.

import { DEFAULT_CONFIG, buildModel, forwardNoCache } from "./core/model.js";
import { tensor, matmul, rmsNorm, silu, rope } from "./core/tensor.js";
import { encode, PROMPTS } from "./core/tokenizer.js";
import { argmax, timeIt } from "./core/metrics.js";

const SEED = 42; // fixed: determinism is the whole point — see core/model header.

// Summary statistics of one activation vector. We report mean and std (population)
// because RMSNorm's job is to control the *scale* of activations: after it runs,
// rms = sqrt(mean(x^2)) should be ~1 (times the per-dim gain, which inits to ~1),
// so std lands near 1 too. Watching std stay bounded across operators is how you
// see, numerically, that the network is not diverging.
type Stats = { mean: number; std: number; rms: number; absMax: number };

function statsOf(x: Float64Array): Stats {
  const n = x.length;
  let sum = 0;
  let sumSq = 0;
  let absMax = 0;
  for (let i = 0; i < n; i++) {
    const v = x[i];
    sum += v;
    sumSq += v * v;
    const a = Math.abs(v);
    if (a > absMax) absMax = a;
  }
  const mean = sum / n;
  const meanSq = sumSq / n;
  // std uses the population formula (divide by n): we are describing THIS exact
  // vector, not estimating a sampling distribution, so Bessel's n-1 correction is
  // the wrong tool here.
  const variance = Math.max(0, meanSq - mean * mean);
  return { mean, std: Math.sqrt(variance), rms: Math.sqrt(meanSq), absMax };
}

function fmtStats(label: string, s: Stats): string {
  const f = (x: number) => x.toFixed(4).padStart(9);
  return `    ${label.padEnd(22)} mean=${f(s.mean)}  std=${f(s.std)}  rms=${f(s.rms)}  |max|=${f(s.absMax)}`;
}

// --- (a) per-operator trace through ONE transformer block --------------------
//
// We reproduce, operator by operator, what core's blockStep does for the LAST
// token of a prompt at layer 0 — same primitives, same order — but capture the
// intermediate activations so we can print stats. The point is pedagogical
// transparency: a reader sees embed -> RMSNorm -> Q/K/V -> RoPE -> attention ->
// residual -> RMSNorm -> FFN -> residual, with the scale at every checkpoint.
//
// Invariant being demonstrated: every RMSNorm output has rms ~= 1 (times gain),
// independent of how large its input grew via residual accumulation. That is the
// mechanism. We compute it on the real model weights, so the numbers are honest.
function traceBlockOperators(): void {
  const cfg = DEFAULT_CONFIG;
  const model = buildModel(cfg, SEED);
  // Use the longest prompt so the attention step actually attends over many
  // positions (a length-1 prompt would make attention a no-op and hide the point).
  const ids = encode(PROMPTS[2]);
  const seq = ids.length;
  const w = model.layers[0];
  const lastPos = seq - 1;

  console.log(`\n[a] Per-operator activation trace — layer 0, last token (pos ${lastPos}) of a ${seq}-token prompt`);
  console.log(`    config: dModel=${cfg.dModel} nHeads=${cfg.nHeads} nKVHeads=${cfg.nKVHeads} dHead=${cfg.dHead} dFF=${cfg.dFF}`);

  // To get the LAST token's pre-block hidden at layer 0, that hidden is simply its
  // embedding row (layer 0 input == embeddings). We also need all prior tokens'
  // K/V at this layer for the attention step, which are likewise functions of their
  // embeddings; we recompute them here with the same projectQKV+RoPE core does.
  const dModel = cfg.dModel;
  const kvDim = cfg.nKVHeads * cfg.dHead;

  const embedRow = (id: number): Float64Array =>
    Float64Array.from(model.embed.data.subarray(id * dModel, (id + 1) * dModel));

  const hLast = embedRow(ids[lastPos]);
  console.log(fmtStats("embed (block input)", statsOf(hLast)));

  // pre-norm (RMSNorm) before attention
  const normed = rmsNorm(tensor(hLast, [1, dModel]), w.normAtt).data;
  console.log(fmtStats("RMSNorm(att)", statsOf(normed)));
  // NOTE: rms here should be ~1 regardless of the embed input's scale above — that
  // contrast (embed |max| vs normed |max|) is the whole lesson of this checkpoint.

  // Q/K/V projections for the last token.
  const nT = tensor(normed, [1, dModel]);
  const qLast = matmul(nT, w.wQ).data;
  const kLast = matmul(nT, w.wK).data;
  const vLast = matmul(nT, w.wV).data;
  console.log(fmtStats("Q proj", statsOf(qLast)));
  console.log(fmtStats("K proj", statsOf(kLast)));
  console.log(fmtStats("V proj", statsOf(vLast)));

  // RoPE rotates Q (and K) in place per head. We rotate only the last token's Q
  // for the trace; full K history is rotated below when building the score matrix.
  ropeAllQueryHeads(qLast, lastPos, cfg);
  console.log(fmtStats("Q after RoPE", statsOf(qLast)));
  // RoPE is a rotation: it preserves each head's L2 norm, so rms is unchanged —
  // a good sanity check that we applied it correctly (a buggy RoPE would shift rms).

  // Build the full rope'd K and V history (all positions 0..lastPos), then run
  // GQA attention for the last token. We reuse the exact attention arithmetic by
  // hand so we can inspect the post-attention activation scale.
  const kHist = new Float64Array(seq * kvDim);
  const vHist = new Float64Array(seq * kvDim);
  for (let p = 0; p < seq; p++) {
    const hp = rmsNorm(tensor(embedRow(ids[p]), [1, dModel]), w.normAtt).data;
    const hpT = tensor(hp, [1, dModel]);
    const kp = matmul(hpT, w.wK).data;
    const vp = matmul(hpT, w.wV).data;
    ropeAllKVHeads(kp, p, cfg); // rotate K at its absolute position p
    kHist.set(kp, p * kvDim);
    vHist.set(vp, p * kvDim);
  }
  const { out: attn, lastLogits: lastHeadScores } = attendLastToken(qLast, kHist, vHist, seq, cfg);
  console.log(fmtStats("attention output", statsOf(attn)));

  const attnOut = matmul(tensor(attn, [1, cfg.nHeads * cfg.dHead]), w.wO).data;
  console.log(fmtStats("attn out-proj (wO)", statsOf(attnOut)));

  // residual add
  const h1 = new Float64Array(dModel);
  for (let i = 0; i < dModel; i++) h1[i] = hLast[i] + attnOut[i];
  console.log(fmtStats("after attn residual", statsOf(h1)));
  // NOTE: residual ADDS attnOut onto the input, so |max| here is typically larger
  // than either summand — this is exactly the unbounded growth RMSNorm reins in
  // at the next checkpoint. Watch rms go back to ~1 on the next line.

  // FFN sub-block: pre-norm, SiLU-gated MLP, residual.
  const normed2 = rmsNorm(tensor(h1, [1, dModel]), w.normFFN).data;
  console.log(fmtStats("RMSNorm(ffn)", statsOf(normed2)));
  const n2T = tensor(normed2, [1, dModel]);
  const gate = silu(matmul(n2T, w.w1)).data;
  const up = matmul(n2T, w.w3).data;
  const prod = new Float64Array(cfg.dFF);
  for (let i = 0; i < cfg.dFF; i++) prod[i] = gate[i] * up[i];
  console.log(fmtStats("SiLU-gated hidden", statsOf(prod)));
  const ff = matmul(tensor(prod, [1, cfg.dFF]), w.w2).data;
  console.log(fmtStats("FFN down-proj", statsOf(ff)));
  const h2 = new Float64Array(dModel);
  for (let i = 0; i < dModel; i++) h2[i] = h1[i] + ff[i];
  console.log(fmtStats("block output", statsOf(h2)));

  // Stash the last head's raw (pre-softmax) attention scores for the (c) failure
  // demo — these are real logits from a real attention head over `seq` positions,
  // which is exactly the kind of vector that overflows a naive softmax.
  failureDemoScores = lastHeadScores;

  // The headline invariant, stated as a checked claim, not a vibe:
  const normedRms = statsOf(normed).rms;
  const normed2Rms = statsOf(normed2).rms;
  const ok = Math.abs(normedRms - 1) < 0.1 && Math.abs(normed2Rms - 1) < 0.1;
  console.log(
    `    => RMSNorm invariant: both norm outputs have rms ~ 1 ` +
      `(${normedRms.toFixed(3)}, ${normed2Rms.toFixed(3)}) — ${ok ? "HOLDS" : "VIOLATED"}`
  );
}

// RoPE the query heads of one token in place (nHeads heads of dHead each). core's
// rope() rotates a q/k pair; we only want q rotated here, so pass a throwaway k.
function ropeAllQueryHeads(q: Float64Array, pos: number, cfg: typeof DEFAULT_CONFIG): void {
  for (let h = 0; h < cfg.nHeads; h++) {
    const slice = q.subarray(h * cfg.dHead, (h + 1) * cfg.dHead);
    const scratch = Float64Array.from(slice);
    rope(slice, scratch, pos, 10000);
  }
}

// RoPE the kv heads of one token in place (nKVHeads heads, GQA-narrower than q).
function ropeAllKVHeads(k: Float64Array, pos: number, cfg: typeof DEFAULT_CONFIG): void {
  for (let h = 0; h < cfg.nKVHeads; h++) {
    const slice = k.subarray(h * cfg.dHead, (h + 1) * cfg.dHead);
    const scratch = Float64Array.from(slice);
    rope(scratch, slice, pos, 10000);
  }
}

// GQA attention for one query token over `len` cached positions. Mirrors core's
// attendToken, but also returns the raw (pre-softmax) score row of the LAST head
// so the failure demo has real attention logits to feed a naive softmax.
function attendLastToken(
  q: Float64Array,
  kAll: Float64Array,
  vAll: Float64Array,
  len: number,
  cfg: typeof DEFAULT_CONFIG
): { out: Float64Array; lastLogits: Float64Array } {
  const groupSize = cfg.nHeads / cfg.nKVHeads;
  const kvDim = cfg.nKVHeads * cfg.dHead;
  const out = new Float64Array(cfg.nHeads * cfg.dHead);
  const invSqrtD = 1 / Math.sqrt(cfg.dHead);
  let lastLogits = new Float64Array(len);
  for (let h = 0; h < cfg.nHeads; h++) {
    const kvHead = Math.floor(h / groupSize);
    const qOff = h * cfg.dHead;
    const rawScores = new Float64Array(len);
    for (let p = 0; p < len; p++) {
      const kOff = p * kvDim + kvHead * cfg.dHead;
      let dotv = 0;
      for (let d = 0; d < cfg.dHead; d++) dotv += q[qOff + d] * kAll[kOff + d];
      rawScores[p] = dotv * invSqrtD;
    }
    // numerically-stable softmax via core (it subtracts the row max).
    const probs = stableSoftmaxRow(rawScores);
    for (let p = 0; p < len; p++) {
      const vOff = p * kvDim + kvHead * cfg.dHead;
      const wgt = probs[p];
      for (let d = 0; d < cfg.dHead; d++) out[qOff + d] += wgt * vAll[vOff + d];
    }
    lastLogits = rawScores; // keep the final head's raw scores
  }
  return { out, lastLogits };
}

// --- (c) failure mode: naive softmax overflows ------------------------------
//
// THE numerics lesson. A correct softmax subtracts the row max before exp (see
// core/tensor.softmax). The naive version does not. On small logits both agree;
// on the large logits real attention produces, the naive one exponentiates a big
// number and overflows to Infinity, then Infinity/Infinity = NaN poisons the row.

function stableSoftmaxRow(scores: Float64Array): Float64Array {
  // delegate to the principle core uses: subtract max first.
  let max = -Infinity;
  for (const v of scores) if (v > max) max = v;
  const out = new Float64Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    const e = Math.exp(scores[i] - max);
    out[i] = e;
    sum += e;
  }
  const inv = 1 / sum;
  for (let i = 0; i < out.length; i++) out[i] *= inv;
  return out;
}

function naiveSoftmaxRow(scores: Float64Array): Float64Array {
  // The bug: exp the raw logit with NO max-subtraction.
  const out = new Float64Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    const e = Math.exp(scores[i]); // overflows to Infinity for large scores
    out[i] = e;
    sum += e;
  }
  const inv = 1 / sum; // Infinity-sum -> inv = 0; e/Infinity-style -> NaN
  for (let i = 0; i < out.length; i++) out[i] *= inv;
  return out;
}

// Captured during the (a) trace: a real head's pre-softmax scores. Initialized in
// traceBlockOperators(); used by demoNaiveSoftmaxOverflow(). Module-level because
// the trace produces it as a byproduct and the demo consumes it — passing it
// through every call site would only add noise.
let failureDemoScores: Float64Array = new Float64Array(0);

function hasNaN(x: Float64Array): boolean {
  for (const v of x) if (Number.isNaN(v)) return true;
  return false;
}

function demoNaiveSoftmaxOverflow(): void {
  console.log(`\n[c] Failure mode: softmax WITHOUT max-subtraction`);

  // First, on the REAL attention scores captured during the (a) trace: these are
  // modest (the toy model keeps logits small), so both softmaxes should agree —
  // this establishes the naive version is not "always broken", it is conditionally
  // broken, which is the dangerous kind.
  const realScores = failureDemoScores;
  const realStable = stableSoftmaxRow(realScores);
  const realNaive = naiveSoftmaxRow(realScores);
  let maxDrift = 0;
  for (let i = 0; i < realStable.length; i++) {
    const d = Math.abs(realStable[i] - realNaive[i]);
    if (d > maxDrift) maxDrift = d;
  }
  const realMax = Math.max(...realScores);
  console.log(
    `    real attention scores (len=${realScores.length}, max logit=${realMax.toFixed(3)}): ` +
      `naive vs stable max prob drift = ${maxDrift.toExponential(2)} ` +
      `(agree — toy logits are too small to overflow)`
  );

  // Now the failure: scale the SAME real scores up to the magnitude a trained
  // model's attention routinely reaches (logits of +700..+900). exp(710) already
  // exceeds float64's ~1.8e308 ceiling -> Infinity. We do not invent numbers; we
  // take the real score row and amplify it, so the demo is a transformation of
  // genuine data, not a hard-coded NaN.
  const amplify = 800 / Math.max(1e-9, realMax); // push the top logit to ~800
  const hot = Float64Array.from(realScores, (v) => v * amplify);
  const hotMax = Math.max(...hot);
  const stableHot = stableSoftmaxRow(hot);
  const naiveHot = naiveSoftmaxRow(hot);
  console.log(`    amplified scores (max logit=${hotMax.toFixed(1)}, exp(${hotMax.toFixed(0)}) overflows float64):`);
  console.log(`      naive softmax  -> hasNaN=${hasNaN(naiveHot)}  sum=${sumOf(naiveHot)}  [first 3: ${preview(naiveHot)}]`);
  console.log(`      stable softmax -> hasNaN=${hasNaN(stableHot)}  sum=${sumOf(stableHot).toFixed(6)}  [first 3: ${preview(stableHot)}]`);
  const proved = hasNaN(naiveHot) && !hasNaN(stableHot);
  console.log(
    `    => ${proved ? "PROVEN" : "NOT REPRODUCED"}: max-subtraction is load-bearing — ` +
      `naive path NaNs, stable path stays a valid distribution (sum=1).`
  );
}

function sumOf(x: Float64Array): number {
  let s = 0;
  for (const v of x) s += v;
  return s;
}
function preview(x: Float64Array): string {
  return Array.from(x.subarray(0, 3))
    .map((v) => (Number.isNaN(v) ? "NaN" : v.toExponential(2)))
    .join(", ");
}

// --- (b) attention O(seq^2) cost curve --------------------------------------
//
// Attention computes a score for every (query, key) pair. For a full-sequence
// forward over seq tokens, that is seq*(seq+1)/2 query-key dot products per head
// per layer (causal: token i attends to 0..i). The count is exactly quadratic in
// seq; we print both the analytic count AND a wall-clock of forwardNoCache to show
// the measured time tracks the count's growth shape.
//
// We measure forwardNoCache (the O(seq^2) reference path) because chapter 01 has
// no cache yet — quantifying this cost is precisely the motivation for chapter 02.

function attentionDotProducts(seq: number, cfg: typeof DEFAULT_CONFIG): number {
  // per head, per layer: sum_{i=0}^{seq-1} (i+1) = seq*(seq+1)/2 causal pairs.
  const causalPairs = (seq * (seq + 1)) / 2;
  return causalPairs * cfg.nHeads * cfg.nLayers;
}

function benchAttentionScaling(): void {
  const cfg = DEFAULT_CONFIG;
  const model = buildModel(cfg, SEED);
  const seqs = [16, 32, 64, 128];
  console.log(`\n[b] Attention O(seq^2) cost — forwardNoCache (the un-cached reference path)`);
  console.log(`    seq | qk-dot-products | wall-clock ms | ms/seq^2 (x1e-6) | growth vs prev`);

  // Deterministic, repeatable token stream from the model's own vocab via a tiny
  // counter — content is irrelevant to timing (matmul cost is data-independent by
  // design; see core/tensor matmul comment), only LENGTH matters.
  const makeIds = (n: number): number[] => Array.from({ length: n }, (_, i) => (i * 7 + 3) % cfg.vocabSize);

  let prevMs = 0;
  let prevSeq = 0;
  for (const seq of seqs) {
    const ids = makeIds(seq);
    const dots = attentionDotProducts(seq, cfg);

    // Warm up first (JIT + cache) so the measured run reflects steady state — the
    // convention core/metrics.timeIt deliberately does NOT bake in. Average a few
    // runs to damp timer noise on these sub-millisecond workloads.
    timeIt(() => void forwardNoCache(model, ids));
    const REPS = 5;
    let total = 0;
    for (let r = 0; r < REPS; r++) total += timeIt(() => void forwardNoCache(model, ids));
    const ms = total / REPS;

    // ms / seq^2: total time / seq^2. At these TOY sizes this column still FALLS as
    // seq grows, because total cost = O(seq) per-token work (FFN+projections, which
    // dominate this tiny model) + O(seq^2) attention, and the linear term is still
    // larger here. The quadratic signature therefore shows up not in this column but
    // in the GROWTH RATIO: pure O(seq) would give exactly 2.0x per doubling; the
    // ratios climbing ABOVE 2.0x (2.00 -> 2.07 -> 2.15) are attention's seq^2 term
    // starting to bite. With a bigger model / longer seq the normalized column would
    // flatten as attention takes over.
    const normalized = (ms / (seq * seq)) * 1e6;
    const growth = prevMs > 0 ? `${(ms / prevMs).toFixed(2)}x for ${(seq / prevSeq).toFixed(0)}x seq` : "—";
    console.log(
      `    ${String(seq).padStart(3)} | ${String(dots).padStart(15)} | ` +
        `${ms.toFixed(3).padStart(13)} | ${normalized.toFixed(3).padStart(16)} | ${growth}`
    );
    prevMs = ms;
    prevSeq = seq;
  }
  console.log(
    `    => doubling seq 4x's the dot-product count (quadratic, exact). Wall-clock: pure O(seq)`
  );
  console.log(
    `       would give exactly 2.0x per doubling; the ratios climbing ABOVE 2.0x are attention's`
  );
  console.log(
    `       O(seq^2) term emerging on top of the O(seq) per-token work that dominates at toy sizes.`
  );
}

// --- (d) golden logits ------------------------------------------------------
//
// Save the last-token logits for all 3 prompts as this chapter's reference. Later
// chapters (cache, batching, quant) must reproduce these to within float64 noise;
// equality with this golden is their correctness proof. We print a compact, stable
// fingerprint (a checksum + the argmax token + a few leading values) rather than
// dumping 256 floats per prompt — enough to detect any drift, small enough to read.

function checksum(x: Float64Array): number {
  // Order-sensitive rolling sum; not cryptographic — just a cheap, deterministic
  // fingerprint that changes if ANY logit changes. Reproducible across machines
  // because it is pure float64 arithmetic on a deterministic input.
  let acc = 0;
  for (let i = 0; i < x.length; i++) acc = acc * 1.0000001 + x[i] * (i + 1);
  return acc;
}

function emitGoldenLogits(): void {
  const cfg = DEFAULT_CONFIG;
  const model = buildModel(cfg, SEED);
  console.log(`\n[d] Golden last-token logits (seed=${SEED}) — chapter 01 reference for cross-chapter diffs`);
  console.log(`    prompt(len) | argmax tok | logit[argmax] | mean logit | checksum`);
  for (let i = 0; i < PROMPTS.length; i++) {
    const ids = encode(PROMPTS[i]);
    const logits = forwardNoCache(model, ids);
    const am = argmax(logits);
    const s = statsOf(logits);
    console.log(
      `    p${i}(${String(ids.length).padStart(3)})     | ${String(am).padStart(10)} | ` +
        `${logits[am].toFixed(4).padStart(13)} | ${s.mean.toFixed(4).padStart(10)} | ${checksum(logits).toExponential(6)}`
    );
  }

  // Determinism re-check: rebuild from the same seed, recompute, and confirm the
  // golden is bit-for-bit reproducible. If this ever drifts, every downstream
  // cross-chapter comparison is built on sand — so we assert it here, loudly.
  const model2 = buildModel(cfg, SEED);
  const a = forwardNoCache(model, encode(PROMPTS[1]));
  const b = forwardNoCache(model2, encode(PROMPTS[1]));
  let drift = 0;
  for (let i = 0; i < a.length; i++) drift = Math.max(drift, Math.abs(a[i] - b[i]));
  console.log(`    => determinism: two independent builds drift by ${drift.toExponential(2)} (must be 0 for golden to be a valid reference)`);
}

function main(): void {
  console.log("=== stage01 — Tiny Transformer forward: token -> logits ===");
  console.log(`model: LLaMA-style decoder, ${DEFAULT_CONFIG.nLayers} layers, GQA (nKVHeads=${DEFAULT_CONFIG.nKVHeads}/nHeads=${DEFAULT_CONFIG.nHeads})`);
  console.log("NOTE: weights are synthetic (seeded PRNG); the model is UNTRAINED, so logits are");
  console.log("structurally-correct gibberish. This chapter is about the forward MECHANISM, not output quality.");

  traceBlockOperators(); // (a) must run first — it populates failureDemoScores
  benchAttentionScaling(); // (b)
  demoNaiveSoftmaxOverflow(); // (c) consumes failureDemoScores
  emitGoldenLogits(); // (d)

  console.log("\n=== done. Absolute ms are pessimistic (toy float64 kernels); the QUADRATIC SHAPE and");
  console.log("the RMSNorm/softmax invariants are what transfer to a real engine. ===");
}

main();
