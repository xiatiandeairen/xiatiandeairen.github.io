// core/_smoke.ts — core底座自检. NOT a book stage; it is the scaffold's own
// acceptance test, kept in core so it can import internals freely. Run with
// `npx tsx src/core/_smoke.ts`. It proves three load-bearing invariants the whole
// book depends on, with REAL numbers (no asserts-only):
//   1. determinism: same (cfg, seed) -> bit-for-bit identical logits.
//   2. cache correctness: forwardStep (cached) == forwardNoCache (reference).
//   3. honest metrics + a real FAILURE MODE (softmax overflow without max-subtract).

import { DEFAULT_CONFIG, buildModel, forwardNoCache, forwardStep, newCache } from "./model.js";
import { encode, decode, PROMPTS, VOCAB_SIZE } from "./tokenizer.js";
import {
  timeIt,
  tokensPerSecond,
  maxLogitDrift,
  estimateKVBytes,
  formatBytes,
  speedup,
  perplexity,
  argmax,
  interTokenLatency,
} from "./metrics.js";

function softmaxNaive(row: number[]): number[] {
  // deliberately WRONG: no max-subtract. Demonstrates the overflow failure mode
  // that core/tensor.softmax guards against.
  const exps = row.map((v) => Math.exp(v));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

console.log("=== core 自检 (core smoke test) ===\n");

const model = buildModel(DEFAULT_CONFIG, 42);
const model2 = buildModel(DEFAULT_CONFIG, 42);

// --- 1. tokenizer round-trip --------------------------------------------------
console.log("[1] tokenizer round-trip (decode∘encode === id):");
for (const p of PROMPTS) {
  const ids = encode(p);
  const ok = decode(ids) === p;
  console.log(`    len=${String(ids.length).padStart(3)} roundtrip=${ok ? "OK" : "FAIL"}  "${p.slice(0, 32)}${p.length > 32 ? "…" : ""}"`);
}
console.log(`    vocabSize=${VOCAB_SIZE} (byte-level)\n`);

// --- 2. determinism -----------------------------------------------------------
const promptIds = encode(PROMPTS[1]);
const logitsA = Array.from(forwardNoCache(model, promptIds));
const logitsB = Array.from(forwardNoCache(model2, promptIds));
console.log("[2] determinism (two builds, seed=42):");
console.log(`    maxLogitDrift = ${maxLogitDrift(logitsA, logitsB).toExponential(2)} (must be 0)\n`);

// --- 3. cache correctness: cached path must equal reference path ---------------
const cache = newCache(DEFAULT_CONFIG);
let cachedLast: Float64Array = new Float64Array(VOCAB_SIZE);
for (const id of promptIds) cachedLast = forwardStep(model, id, cache);
const drift = maxLogitDrift(logitsA, Array.from(cachedLast));
console.log("[3] cache correctness (forwardStep vs forwardNoCache, last-token logits):");
console.log(`    maxLogitDrift = ${drift.toExponential(2)}  (≈0 means cache is exact)`);
console.log(`    argmax cached=${argmax(cachedLast)} ref=${argmax(logitsA)} match=${argmax(cachedLast) === argmax(logitsA)}\n`);

// --- 4. honest timing: prefill (TTFT) vs decode -------------------------------
const N_DECODE = 16;
// warmup (see metrics.timeIt convention)
{
  const c = newCache(DEFAULT_CONFIG);
  for (const id of promptIds) forwardStep(model, id, c);
}
const c2 = newCache(DEFAULT_CONFIG);
const prefillMs = timeIt(() => {
  for (const id of promptIds) forwardStep(model, id, c2);
});
let next = argmax(forwardStep(model, argmax(cachedLast), c2));
const stepMs: number[] = [];
for (let i = 0; i < N_DECODE; i++) {
  const t = timeIt(() => {
    next = argmax(forwardStep(model, next, c2));
  });
  stepMs.push(t);
}
const itl = interTokenLatency(stepMs);
console.log("[4] honest two-phase timing:");
console.log(`    prefill (TTFT) over ${promptIds.length} tokens = ${prefillMs.toFixed(2)} ms`);
console.log(`    decode inter-token latency: mean=${itl.mean.toFixed(3)} p95=${itl.p95.toFixed(3)} ms/tok`);
console.log(`    decode throughput = ${tokensPerSecond(N_DECODE, stepMs.reduce((a, b) => a + b, 0)).toFixed(1)} tok/s\n`);

// --- 5. KV memory estimate (est.) ---------------------------------------------
const kvOne = estimateKVBytes(DEFAULT_CONFIG, 128, 1);
const kvBatch = estimateKVBytes(DEFAULT_CONFIG, 128, 32);
console.log("[5] KV cache footprint (exact arithmetic, labeled est. when printed):");
console.log(`    seq=128, 1 seq   = ${formatBytes(kvOne)} (est.)`);
console.log(`    seq=128, 32 seqs = ${formatBytes(kvBatch)} (est.)`);
console.log(`    GQA factor nHeads/nKVHeads = ${DEFAULT_CONFIG.nHeads / DEFAULT_CONFIG.nKVHeads}x smaller than MHA\n`);

// --- 6. equivalence judge sanity: perplexity ----------------------------------
const logitRows: number[][] = [];
const targets: number[] = [];
const c3 = newCache(DEFAULT_CONFIG);
for (let i = 0; i < promptIds.length - 1; i++) {
  const lg = forwardStep(model, promptIds[i], c3);
  logitRows.push(Array.from(lg));
  targets.push(promptIds[i + 1]);
}
console.log("[6] perplexity (equivalence judge):");
console.log(`    PPL of prompt under untrained model = ${perplexity(logitRows, targets).toFixed(1)} (~vocab size, expected for untrained)\n`);

// --- 7. FAILURE MODE: softmax overflow without max-subtract -------------------
console.log("[7] FAILURE MODE — softmax must subtract max:");
const bigLogits = [800, 801, 802]; // realistic large attention logits
const naive = softmaxNaive(bigLogits);
console.log(`    naive softmax([800,801,802]) = [${naive.map((x) => x.toFixed(3)).join(", ")}]  <- NaN: exp(800)=Inf`);
console.log(`    (core/tensor.softmax subtracts row max first, so it stays finite)\n`);

// --- 8. speedup helper demo ---------------------------------------------------
console.log("[8] speedup helper:");
console.log(`    speedup(baseline=100ms, optimized=25ms) = ${speedup(100, 25).toFixed(2)}x\n`);

console.log("=== all core invariants checked ===");
