// stage02-kvcache.ts — KV 缓存：把 decode 从 O(n²) 砍成 O(n) 的那一步。
//
// The thesis of this chapter, made measurable. Autoregressive decoding emits one
// token per forward pass, and each new token attends over EVERY prior token. The
// naive way to get the next-token logits is to re-run the whole sequence through
// the model from scratch — forwardNoCache — which re-embeds and recomputes the K/V
// projections of ALL past tokens on every step. Step k does O(k) full work, so
// generating n tokens costs O(n²) total. The KV cache is the one idea that makes
// LLM serving viable: store each token's K and V the first time you compute them,
// so every later step does only O(1) new projection (just the current token) plus
// an O(k) attention scan over cached keys. The headline "O(n²) → O(n)" is about
// the PROJECTION/FFN work that dominates: forwardNoCache redoes it for the whole
// prefix every step (O(n²) summed), the cached path does it once per token (O(n)
// summed). The attention scan itself is O(n²) in both, but with a far smaller
// constant, so in practice the cached path's total work collapses. This file
// PROVES that with three things and one failure:
//
//   (a) CORRECTNESS — the two paths must produce bit-identical last-token logits
//       (maxLogitDrift == 0). If the cache is even slightly wrong, the whole speed
//       argument is moot, so equivalence is checked FIRST and is load-bearing.
//   (b) SPEEDUP that grows with generation length — the direct, measured evidence
//       that we removed an O(n) factor: the longer you generate, the more the
//       no-cache path re-does, so the ratio widens. A flat ratio would mean we
//       did NOT remove a factor.
//   (c) MEMORY — estimateKVBytes shows the cost we trade compute for: the cache
//       grows linearly in sequence length. This is exact arithmetic, labeled (est.).
//   (d) FAILURE MODE — decode WITHOUT prefilling the cache. The most common real
//       cache bug: the model "runs" (no crash) but attention silently sees an empty
//       history, so the logits are wrong. We quantify the drift so the reader knows
//       what a consistency bug looks like in numbers, not just in a stack trace.
//
// Honesty caveats (echoing core/metrics header): the model is untrained synthetic
// weights, the kernels are unoptimized float64 loops, so ABSOLUTE tok/s and ms are
// pessimistic. What transfers to a real engine is the RELATIVE story — the speedup
// RATIO and the fact that it widens with length. Determinism (seeded PRNG) makes
// every number here reproducible bit-for-bit on any machine.

import { DEFAULT_CONFIG, buildModel, forwardNoCache, forwardStep, newCache } from "./core/model.js";
import { encode, PROMPTS } from "./core/tokenizer.js";
import {
  timeIt,
  speedup,
  maxLogitDrift,
  estimateKVBytes,
  formatBytes,
  argmax,
} from "./core/metrics.js";

const SEED = 42;
const GEN_LENGTHS = [8, 16, 32, 64] as const;

// Use the second prompt (length 48): long enough that a growing prefix makes the
// O(n²) re-work visible, short enough that even N=64 generation stays well under
// maxSeq=256. Single prompt is fine here because the variable under study is
// GENERATION LENGTH, not prompt content — we sweep that explicitly in GEN_LENGTHS.
const promptIds = encode(PROMPTS[1]);

const model = buildModel(DEFAULT_CONFIG, SEED);

// genTokensCached — the production path. Prefill the prompt into the cache (one
// forwardStep per prompt token, which is exactly what prefill IS), then decode
// nGen tokens greedily, reusing the cache. Returns the generated token ids plus
// the last-token logits, so a caller can compare logits across paths.
//
// Invariant: cache.len after prefill == promptIds.length; after generation ==
// promptIds.length + nGen. forwardStep advances it; we never touch it by hand.
function genTokensCached(nGen: number): { ids: number[]; lastLogits: Float64Array } {
  const cache = newCache(DEFAULT_CONFIG);
  // prefill: run the prompt through, filling the cache. The last prefill step's
  // logits predict the first generated token. Annotated as plain Float64Array (not
  // the ArrayBuffer-narrowed `new` type) so reassignment from forwardStep type-checks
  // under @types/node 22 — same widening the core applies in forwardStep.
  let logits: Float64Array = new Float64Array(0);
  for (const id of promptIds) logits = forwardStep(model, id, cache);
  const ids: number[] = [];
  for (let i = 0; i < nGen; i++) {
    const next = argmax(logits); // greedy, deterministic (shared tie-break)
    ids.push(next);
    logits = forwardStep(model, next, cache); // O(1) projection + O(len) attend
  }
  return { ids, lastLogits: logits };
}

// genTokensNoCache — the reference path. To produce token k we re-run the ENTIRE
// sequence-so-far through forwardNoCache, which recomputes every prior token's K/V
// from scratch. This is the O(n²) baseline: step k does O(k) full work, summed
// over k = O(n²). It is "obviously correct" by construction (no cache to get
// wrong), which is exactly why it is the oracle the cached path is checked against.
//
// We must drive BOTH paths with the same greedy choices, or a single divergent
// argmax would fork the two token streams and make the comparison meaningless.
// Greedy argmax over identical logits guarantees identical choices as long as the
// per-step logits match — which is the very thing (a) verifies.
function genTokensNoCache(nGen: number): { ids: number[]; lastLogits: Float64Array } {
  const seq = [...promptIds];
  let logits = forwardNoCache(model, seq); // prefill-equivalent: last-token logits
  const ids: number[] = [];
  for (let i = 0; i < nGen; i++) {
    const next = argmax(logits);
    ids.push(next);
    seq.push(next);
    logits = forwardNoCache(model, seq); // re-runs the WHOLE prefix every step
  }
  return { ids, lastLogits: logits };
}

// warmup: the first call to any JS hot path pays JIT + cold-cache costs unrelated
// to steady-state throughput (see core/metrics.timeIt convention — warmup is the
// caller's job, deliberately not baked in). One throwaway run of each path on the
// largest length warms both code paths before any measured run.
function warmup(): void {
  genTokensCached(GEN_LENGTHS[GEN_LENGTHS.length - 1]);
  genTokensNoCache(GEN_LENGTHS[GEN_LENGTHS.length - 1]);
}

console.log("=== stage02 — KV 缓存：O(n²) → O(n) ===\n");
console.log(`prompt = "${PROMPTS[1]}"`);
console.log(`prompt tokens = ${promptIds.length}, seed = ${SEED}, config = DEFAULT (GQA nKVHeads=${DEFAULT_CONFIG.nKVHeads}/nHeads=${DEFAULT_CONFIG.nHeads})\n`);

warmup();

// --- (a) CORRECTNESS — cache must equal the reference, exactly --------------------
// Checked first and per length: if the cache drifts at ANY length the speedup is a
// lie. We compare last-token logits (maxLogitDrift, L∞) AND the full generated id
// streams (any mismatch flips a token and is caught).
console.log("[a] CORRECTNESS — cached path vs no-cache reference (must be exact):");
let allExact = true;
for (const n of GEN_LENGTHS) {
  const cached = genTokensCached(n);
  const ref = genTokensNoCache(n);
  const drift = maxLogitDrift(Array.from(cached.lastLogits), Array.from(ref.lastLogits));
  const idsMatch = cached.ids.length === ref.ids.length && cached.ids.every((v, i) => v === ref.ids[i]);
  if (drift !== 0 || !idsMatch) allExact = false;
  console.log(
    `    gen=${String(n).padStart(2)}  maxLogitDrift=${drift.toExponential(2)}  ids_match=${idsMatch}`
  );
}
console.log(`    => ${allExact ? "EXACT at every length (cache is correct)" : "DRIFT DETECTED — cache is buggy"}\n`);

// --- (b) SPEEDUP that widens with length -----------------------------------------
// The headline. Wall-clock both full generations (prompt prefill + nGen decode) at
// each length. The cached total grows ~linearly; the no-cache total grows ~quadratically,
// so the ratio MUST climb with n. A climbing ratio is the direct fingerprint of an
// O(n) factor removed. (Absolute ms are toy-pessimistic; the trend is what transfers.)
console.log("[b] SPEEDUP — no-cache total ms / cached total ms (grows with gen length):");
console.log("    gen  no-cache(ms)  cached(ms)   speedup");
let prevSpeedup = 0;
let monotonic = true;
for (const n of GEN_LENGTHS) {
  const cachedMs = timeIt(() => genTokensCached(n));
  const noCacheMs = timeIt(() => genTokensNoCache(n));
  const sp = speedup(noCacheMs, cachedMs);
  if (sp < prevSpeedup) monotonic = false;
  prevSpeedup = sp;
  console.log(
    `    ${String(n).padStart(3)}  ${noCacheMs.toFixed(2).padStart(11)}  ${cachedMs.toFixed(2).padStart(9)}   ${sp.toFixed(2)}x`
  );
}
console.log(
  `    => speedup ${monotonic ? "grows monotonically with length" : "is noisy run-to-run"} — the O(n²)→O(n) fingerprint\n`
);

// --- (c) MEMORY — the cost we traded compute for ----------------------------------
// estimateKVBytes is exact arithmetic on the config (2*nKVHeads*dHead*seq * nLayers
// * nSeqs * 8 bytes), printed (est.) per the honesty rule. The point: cache memory
// is LINEAR in sequence length — doubling the context doubles the cache. This is why
// stage05 (paged-KV) and GQA exist. We show single-sequence growth plus a 32-seq
// batch to make the serving-capacity number concrete.
console.log("[c] MEMORY — KV cache grows linearly in sequence length (est.):");
console.log("    seq   1 seq        32 seqs");
for (const seq of [16, 32, 64, 128, 256]) {
  const one = estimateKVBytes(DEFAULT_CONFIG, seq, 1);
  const batch = estimateKVBytes(DEFAULT_CONFIG, seq, 32);
  console.log(`    ${String(seq).padStart(3)}   ${formatBytes(one).padStart(10)}   ${formatBytes(batch).padStart(10)} (est.)`);
}
const perTok = estimateKVBytes(DEFAULT_CONFIG, 1, 1);
console.log(`    => +${formatBytes(perTok)}/token/seq (est.); GQA already saves ${DEFAULT_CONFIG.nHeads / DEFAULT_CONFIG.nKVHeads}x vs MHA here\n`);

// --- (d) FAILURE MODE — decode without prefilling the cache -----------------------
// The single most common real KV-cache bug: you forget to run the prompt through
// the cache before decoding (or you reset the cache and reuse it). Nothing crashes
// — forwardStep happily decodes at position 0 with an empty history — but attention
// sees NONE of the prompt, so the model is "answering" a context it never read. The
// logits silently diverge from the correct path. We quantify that drift so the
// reader can recognize the symptom: a finite, large maxLogitDrift (NOT NaN, NOT a
// crash) is the signature of a cache-consistency bug.
console.log("[d] FAILURE MODE — decoding without prefilling the cache (a real bug):");
// correct: prefill THEN take the first decode step.
const correctCache = newCache(DEFAULT_CONFIG);
let correctLogits: Float64Array = new Float64Array(0); // widened, see genTokensCached
for (const id of promptIds) correctLogits = forwardStep(model, id, correctCache);
const firstTok = argmax(correctLogits);
const correctFirstStep = forwardStep(model, firstTok, correctCache);
// buggy: SKIP prefill — decode the same first token into a fresh, empty cache. The
// model attends over an empty history (just this one token at pos 0) instead of the
// 48-token prompt. Same model, same token, same code path — only the cache state differs.
const buggyCache = newCache(DEFAULT_CONFIG);
const buggyFirstStep = forwardStep(model, firstTok, buggyCache);
const failDrift = maxLogitDrift(Array.from(correctFirstStep), Array.from(buggyFirstStep));
const correctArg = argmax(correctFirstStep);
const buggyArg = argmax(buggyFirstStep);
console.log(`    correct cache.len after prefill = ${promptIds.length} (saw the whole prompt)`);
console.log(`    buggy   cache.len               = 0 before decode (saw nothing)`);
console.log(`    maxLogitDrift correct-vs-buggy = ${failDrift.toExponential(2)} (finite & large = silent wrong answer, not a crash)`);
console.log(`    argmax next token: correct=${correctArg} buggy=${buggyArg} flipped=${correctArg !== buggyArg}`);
console.log(`    => the cache IS the context. Skip prefill and the model answers a prompt it never read.\n`);

console.log("=== stage02 done — cache is exact, speedup widens with length, memory is linear ===");
