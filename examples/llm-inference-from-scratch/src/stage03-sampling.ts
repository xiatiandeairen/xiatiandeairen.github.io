// stage03-sampling.ts — from a logit row to the next token: the sampler.
//
// The model gives you a vocab-wide logit vector per step. Picking the next token
// from it is NOT a detail — it is the single biggest lever on output behavior the
// engine exposes, and it sits on the decode hot path (it runs once per generated
// token, forever). This stage implements the four knobs every real engine ships —
// temperature, top-k, top-p (nucleus), repetition penalty — and MEASURES what each
// one actually does to the distribution and to throughput, instead of reciting the
// folklore ("higher temperature = more creative").
//
// What is real here vs what is toy:
//   - The samplers are the real algorithms (same math vLLM / llama.cpp run).
//   - Determinism is real: every sample uses mulberry32(seed), so "same seed =>
//     same token sequence" is a bit-for-bit guarantee we PROVE below, not assert.
//   - The per-step overhead numbers (d) are real wall-clock from performance.now()
//     over the actual core forwardStep decode loop. Absolute tok/s is pessimistic
//     (toy float64 kernels); the RELATIVE story — sampling is cheap next to the
//     forward pass — is what transfers to a real engine.
//   - The logit ROWS used for the distribution experiments (a,b,c,e) are fixed
//     synthetic vectors chosen to expose the sharp-vs-flat behavior split, NOT model
//     output. They are labeled as such. Using a crafted distribution is the honest
//     way to show "top-k and top-p diverge when the distribution is flat" — a real
//     model row would only sometimes be flat, hiding the effect behind luck (N=1).
//
// We do NOT import any stageNN file (they auto-run). We reuse core for everything
// that already exists (softmax, mulberry32, the model + forwardStep, metrics).

import { softmax, tensor } from "./core/tensor.js";
import {
  DEFAULT_CONFIG,
  buildModel,
  newCache,
  forwardStep,
  mulberry32,
} from "./core/model.js";
import { encode, PROMPTS } from "./core/tokenizer.js";
import {
  timeIt,
  tokensPerSecond,
  interTokenLatency,
  argmax,
} from "./core/metrics.js";

// ---------------------------------------------------------------------------
// Sampler primitives. Each takes a *copy-safe* logit row and returns a token id.
// They are pure functions of (logits, params, rng) — no global state — which is
// exactly what makes the determinism proof below possible.
// ---------------------------------------------------------------------------

// Convert logits to a probability distribution at a given temperature.
//
// Temperature T divides the logits before softmax. The invariant worth burning in:
// softmax(z / T). T<1 sharpens (the top logit's lead grows), T>1 flattens (toward
// uniform), T->0 is the limit of "all mass on the argmax" = greedy. We handle T==0
// as a SPECIAL CASE (not 1/0): dividing by zero gives Infinity and softmax returns
// NaN, so greedy is implemented as a one-hot on argmax. This is the first failure
// mode the naive "just divide" code hits.
function softmaxWithTemperature(logits: Float64Array, temperature: number): Float64Array {
  if (temperature <= 0) {
    // Greedy limit: one-hot on the argmax. Doing this explicitly (instead of
    // softmax(z/0) -> NaN) is why T=0 "just works" as deterministic greedy.
    const out = new Float64Array(logits.length);
    out[argmax(logits)] = 1;
    return out;
  }
  const scaled = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
  return softmax(tensor(scaled, [1, scaled.length])).data;
}

// Shannon entropy (in bits) of a probability distribution. This is our scalar
// proxy for "diversity": maximal (log2 V) for uniform, 0 for a one-hot. We report
// it because "temperature up => more diverse" is only credible if measured, and
// entropy is the standard measure of how spread the next-token mass is.
function entropyBits(probs: Float64Array): number {
  let h = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 0) h -= p * Math.log2(p); // 0*log0 := 0, skip to avoid NaN
  }
  return h;
}

// Sample an index from a distribution using one draw of the seeded rng. Inverse-CDF
// (walk the cumulative sum until it exceeds u). The final-index fallback guards the
// case where floating-point rounding makes the cumulative sum end at 0.9999999 < u.
function sampleFromDist(probs: Float64Array, rng: () => number): number {
  const u = rng();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (u < cum) return i;
  }
  return probs.length - 1; // fp-rounding fallback; mass should already sum to ~1
}

// Build the top-k candidate set: keep the k highest-logit tokens, renormalize over
// just those, zero the rest. Returns both the truncated distribution and how many
// tokens survived (== k here, but we return it to compare against top-p, where the
// count is data-dependent — that comparison is the point of experiment (c)).
function topKDist(probs: Float64Array, k: number): { dist: Float64Array; kept: number } {
  // index sort by probability desc; stable enough for our deterministic tie-break
  // since argmax/first-max ordering is not needed here (we keep a *set*).
  const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
  const keep = Math.min(k, probs.length);
  const out = new Float64Array(probs.length);
  let sum = 0;
  for (let i = 0; i < keep; i++) sum += probs[idx[i]];
  // Renormalize over the kept set so the truncated vector is a valid distribution.
  // If sum is 0 (degenerate all-zero input) we leave out as zeros; sampleFromDist's
  // fallback then returns the last index — a loud-ish "your distribution is broken".
  for (let i = 0; i < keep; i++) out[idx[i]] = sum > 0 ? probs[idx[i]] / sum : 0;
  return { dist: out, kept: keep };
}

// Build the top-p (nucleus) candidate set: keep the smallest set of highest-prob
// tokens whose cumulative mass >= p, renormalize over them. The candidate COUNT is
// data-dependent — that is the entire difference from top-k. On a sharp distribution
// a few tokens already cover p (small set); on a flat one it takes many (large set).
function topPDist(probs: Float64Array, p: number): { dist: Float64Array; kept: number } {
  const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
  const out = new Float64Array(probs.length);
  let cum = 0;
  let kept = 0;
  for (let i = 0; i < idx.length; i++) {
    out[idx[i]] = probs[idx[i]];
    cum += probs[idx[i]];
    kept++;
    // Stop once we've covered p. The >= means the token that crosses the threshold
    // is included — nucleus sampling keeps the boundary token, it does not drop it.
    if (cum >= p) break;
  }
  let sum = 0;
  for (let i = 0; i < out.length; i++) sum += out[i];
  for (let i = 0; i < out.length; i++) out[i] = sum > 0 ? out[i] / sum : 0;
  return { dist: out, kept };
}

// Apply a repetition penalty to raw LOGITS (before temperature), the way the
// original CTRL / HuggingFace formulation does: a token that already appeared has
// its logit divided by penalty if positive, multiplied if negative. penalty>1
// discourages repeats. The failure mode in (e): too large a penalty crushes even
// the legitimately-correct next token, so we apply it to logits and let the caller
// observe the argmax flip.
function applyRepetitionPenalty(
  logits: Float64Array,
  seen: Set<number>,
  penalty: number
): Float64Array {
  const out = Float64Array.from(logits);
  for (const id of seen) {
    if (id < 0 || id >= out.length) continue;
    // The asymmetric positive/negative branch is the standard formulation: dividing
    // a positive logit lowers it, but dividing a NEGATIVE logit would *raise* it
    // (toward 0), which is backwards — so negatives are multiplied instead.
    out[id] = out[id] > 0 ? out[id] / penalty : out[id] * penalty;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The fixed synthetic logit rows for the distribution experiments. NOT model
// output — crafted to expose behavior. SHARP: one clear winner. FLAT: near-uniform
// with a faint gradient. Vocab kept tiny (10) so the printed candidate counts are
// readable; the algorithms are vocab-size-agnostic.
// ---------------------------------------------------------------------------
const SHARP_LOGITS = new Float64Array([8.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0, -0.5, -1.0]);
const FLAT_LOGITS = new Float64Array([1.2, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]);

function fmt(n: number, d = 3): string {
  return n.toFixed(d);
}

function main(): void {
  console.log("=== stage03: 采样策略 — 从 logits 到下一个 token ===\n");
  console.log("注：(a)(b)(c)(e) 用固定合成 logit 行（非模型输出，便于暴露行为分叉）；");
  console.log("    (d) 用真实 core 模型 + forwardStep 解码热路径实测 wall-clock。\n");

  // ---- (a) determinism: same seed => same token, across strategies & repeats ----
  console.log("--- (a) 可复现性：同 seed 下每种策略重复运行结果一致 ---");
  const SEED = 42;
  const REPEATS = 5;
  // Use a non-degenerate distribution: a peaked row (SHARP at T=1) puts ~0.99 mass
  // on tok0, so EVERY seed lands on tok0 and the cross-seed control below could not
  // actually demonstrate variation (it would look like greedy). FLAT at T=1.2 is
  // spread enough that different seeds genuinely pick different tokens — which is
  // what makes "same seed identical, different seed differs" a real two-sided proof.
  const baseProbs = softmaxWithTemperature(FLAT_LOGITS, 1.2);
  const strategies: { name: string; draw: (rng: () => number) => number }[] = [
    { name: "temperature(T=1.0)", draw: (rng) => sampleFromDist(baseProbs, rng) },
    { name: "top-k(k=3)", draw: (rng) => sampleFromDist(topKDist(baseProbs, 3).dist, rng) },
    { name: "top-p(p=0.9)", draw: (rng) => sampleFromDist(topPDist(baseProbs, 0.9).dist, rng) },
    { name: "greedy(T=0)", draw: () => argmax(FLAT_LOGITS) },
  ];
  for (const s of strategies) {
    const runs: number[] = [];
    for (let r = 0; r < REPEATS; r++) {
      // Fresh PRNG per run from the SAME seed: this is what "reproducible" means —
      // re-seeding resets the stream, so run r always sees the same u draws.
      runs.push(s.draw(mulberry32(SEED)));
    }
    const allSame = runs.every((t) => t === runs[0]);
    console.log(
      `  ${s.name.padEnd(20)} ${REPEATS} 次 -> [${runs.join(", ")}]  identical=${allSame}`
    );
  }
  // Cross-check: a DIFFERENT seed should be able to pick a different token (else the
  // sampler is just disguised greedy). We show the token varies across seeds.
  const seedTokens = [1, 2, 3, 4, 5, 6].map((sd) => sampleFromDist(baseProbs, mulberry32(sd)));
  console.log(
    `  控制对照 top-1 sampler 跨 6 个不同 seed -> [${seedTokens.join(", ")}] (应出现不同 token，证明不是伪随机退化为 greedy)\n`
  );

  // ---- (b) temperature -> entropy monotonicity ----
  console.log("--- (b) 温度 T ↑ ⇒ 分布熵 ↑（多样性单调上升），熵单位 bit ---");
  console.log(`    （SHARP 合成行，vocab=${SHARP_LOGITS.length}，均匀分布上界 = log2(${SHARP_LOGITS.length}) = ${fmt(Math.log2(SHARP_LOGITS.length))} bit）`);
  const temps = [0.0, 0.5, 1.0, 1.5];
  let prevEntropy = -1;
  let monotonic = true;
  for (const t of temps) {
    const probs = softmaxWithTemperature(SHARP_LOGITS, t);
    const h = entropyBits(probs);
    const top = argmax(probs);
    const topP = probs[top];
    if (t > 0 && h < prevEntropy - 1e-12) monotonic = false; // T=0 is the one-hot floor
    console.log(
      `  T=${fmt(t, 1)}  entropy=${fmt(h)} bit   argmax=tok${top} p(argmax)=${fmt(topP)}`
    );
    prevEntropy = h;
  }
  console.log(`  熵随 T 单调非降 = ${monotonic}（T=0 为 one-hot 下界 0 bit）\n`);

  // ---- (c) top-k vs top-p candidate-count divergence on sharp vs flat ----
  console.log("--- (c) 同一 logits 下 top-k vs top-p 实际候选集大小对比 ---");
  console.log("    top-k 候选数恒定；top-p 候选数随分布尖/平变化 —— 这是两者行为分叉点");
  for (const [label, logits] of [
    ["SHARP", SHARP_LOGITS],
    ["FLAT", FLAT_LOGITS],
  ] as const) {
    const probs = softmaxWithTemperature(logits, 1.0);
    const h = entropyBits(probs);
    const k = 5;
    const p = 0.9;
    const kCount = topKDist(probs, k).kept;
    const pCount = topPDist(probs, p).kept;
    console.log(
      `  ${label.padEnd(5)} (entropy=${fmt(h)} bit)  top-k(k=${k}) 候选=${kCount}   top-p(p=${p}) 候选=${pCount}`
    );
  }
  console.log(
    "  解读：SHARP 时 top-p 只需极少数 token 即覆盖 0.9（< k）；FLAT 时需要更多（≈ 或 > k）。\n"
  );

  // ---- (d) per-step sampling overhead on the REAL decode hot path ----
  console.log("--- (d) 采样在 decode 热路径的 per-step 开销实测（真实 wall-clock）---");
  const model = buildModel(DEFAULT_CONFIG, SEED);
  const prompt = encode(PROMPTS[1]); // medium-length prompt (avoid N=1: see tokenizer)
  const N_DECODE = 64;

  // Helper: prefill then decode N tokens with a given next-token picker, returning
  // per-step decode times. Prefill is excluded from the per-step series — we are
  // isolating decode, where sampling actually lives.
  function decodeWith(pick: (logits: Float64Array, seen: Set<number>, rng: () => number) => number): number[] {
    const cache = newCache(DEFAULT_CONFIG);
    let logits!: Float64Array;
    for (const id of prompt) logits = forwardStep(model, id, cache); // prefill
    const seen = new Set<number>(prompt);
    const rng = mulberry32(SEED);
    const perStep: number[] = [];
    let next = pick(logits, seen, rng);
    for (let i = 0; i < N_DECODE && cache.len < DEFAULT_CONFIG.maxSeq; i++) {
      const ms = timeIt(() => {
        logits = forwardStep(model, next, cache); // forward pass (the heavy part)
      });
      // Time the sampling decision SEPARATELY so we can attribute cost. We measure it
      // outside the forward timer; the printed split is forward-ms vs sample-ms.
      perStep.push(ms);
      seen.add(next);
      next = pick(logits, seen, rng);
    }
    return perStep;
  }

  // Measure the sampling decision alone, amortized, on a real logit row.
  function timeSamplerOnly(pick: (logits: Float64Array, seen: Set<number>, rng: () => number) => number): number {
    const cache = newCache(DEFAULT_CONFIG);
    let logits!: Float64Array;
    for (const id of prompt) logits = forwardStep(model, id, cache);
    const seen = new Set<number>(prompt);
    const rng = mulberry32(SEED);
    const ITERS = 2000;
    timeIt(() => { for (let i = 0; i < 50; i++) pick(logits, seen, rng); }); // warmup
    const ms = timeIt(() => { for (let i = 0; i < ITERS; i++) pick(logits, seen, rng); });
    return ms / ITERS;
  }

  const pickers: { name: string; pick: (l: Float64Array, s: Set<number>, r: () => number) => number }[] = [
    { name: "greedy", pick: (l) => argmax(l) },
    { name: "temperature(T=1.0)", pick: (l, _s, r) => sampleFromDist(softmaxWithTemperature(l, 1.0), r) },
    { name: "top-k(k=40)", pick: (l, _s, r) => sampleFromDist(topKDist(softmaxWithTemperature(l, 1.0), 40).dist, r) },
    { name: "top-p(p=0.9)", pick: (l, _s, r) => sampleFromDist(topPDist(softmaxWithTemperature(l, 1.0), 0.9).dist, r) },
  ];

  // Warmup the whole decode path once (JIT) before measuring, per metrics convention.
  decodeWith((l) => argmax(l));

  for (const { name, pick } of pickers) {
    const perStep = decodeWith(pick);
    const itl = interTokenLatency(perStep);
    const totalMs = perStep.reduce((a, b) => a + b, 0);
    const tps = tokensPerSecond(perStep.length, totalMs);
    const samplerMs = timeSamplerOnly(pick);
    const samplerPct = (samplerMs / (itl.mean + samplerMs)) * 100;
    console.log(
      `  ${name.padEnd(20)} forward ITL mean=${fmt(itl.mean)}ms p95=${fmt(itl.p95)}ms  ` +
        `${fmt(tps, 1)} tok/s | sampler alone=${fmt(samplerMs, 4)}ms/tok (~${fmt(samplerPct, 2)}% of step)`
    );
  }
  console.log(
    "  解读：采样开销 << 前向传播。toy 绝对值偏乐观；可迁移的是“采样在热路径里占比极小”这一相对结论。\n"
  );

  // ---- (e) two failure modes ----
  console.log("--- (e) 失败模式 ---");

  // (e1) top-p=1.0 + high temperature collapses to near-uniform sampling.
  console.log("  [e1] top-p=1.0 + 温度过高 ⇒ 近均匀采样（模型偏好被抹平）");
  const uniformBound = Math.log2(SHARP_LOGITS.length);
  for (const t of [1.0, 5.0, 50.0]) {
    const probs = topPDist(softmaxWithTemperature(SHARP_LOGITS, t), 1.0).dist;
    const h = entropyBits(probs);
    const pct = (h / uniformBound) * 100;
    console.log(
      `       T=${fmt(t, 1)} top-p=1.0  entropy=${fmt(h)} bit (= ${fmt(pct, 1)}% of uniform ${fmt(uniformBound)})`
    );
  }
  // Show the actual draws degrade to "could be anything": at T=50 the sampled token
  // across seeds spreads across the vocab instead of favoring the strong tok0.
  const hotDraws = [1, 2, 3, 4, 5, 6, 7, 8].map((sd) =>
    sampleFromDist(topPDist(softmaxWithTemperature(SHARP_LOGITS, 50.0), 1.0).dist, mulberry32(sd))
  );
  console.log(
    `       T=50 跨 8 seed 采样 -> [${hotDraws.join(", ")}]  (强势 tok0 不再主导，退化为掷骰子)\n`
  );

  // (e2) repetition penalty too large suppresses even the correct token.
  console.log("  [e2] repetition penalty 过大 ⇒ 把正确 token 也压没");
  // Craft a row where tok0 is the clearly-correct next token AND has already been
  // seen (e.g. a legitimately repeated word). A sane penalty keeps it; a huge one
  // flips the argmax to a wrong token. We print the flip.
  const repLogits = new Float64Array([6.0, 5.0, 1.0, 0.5, 0.0]); // tok0 correct, tok1 close runner-up
  const seenRep = new Set<number>([0]); // tok0 already appeared
  for (const penalty of [1.0, 1.2, 10.0]) {
    const penalized = applyRepetitionPenalty(repLogits, seenRep, penalty);
    const before = argmax(repLogits);
    const after = argmax(penalized);
    const flipped = before !== after;
    console.log(
      `       penalty=${fmt(penalty, 1)}  logit[tok0] ${fmt(repLogits[0], 2)} -> ${fmt(penalized[0], 2)}  ` +
        `argmax ${before} -> ${after}  ${flipped ? "⚠ 正确 token 被压没" : "OK"}`
    );
  }
  console.log(
    "       解读：penalty 是钝器。轻度抑制重复无害；过大时把 logit 压到次优 token 之下，正确答案被牺牲。"
  );
}

main();
