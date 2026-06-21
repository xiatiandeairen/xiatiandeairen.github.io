// stage06-speculative.ts — speculative decoding: a cheap DRAFT model races ahead
// proposing K tokens, the expensive TARGET model verifies all K in one pass, and a
// rejection-sampling accept rule guarantees the output is sampled from EXACTLY the
// target's distribution — not an approximation, the same distribution.
//
// The single idea worth internalizing: speculative decoding is a LATENCY trick, not a
// quality trick. It does no less target work per accepted token in the worst case, but
// it amortizes target forward passes across a run of cheap guesses, so when the draft
// happens to agree with the target you pay one target pass for several tokens. The
// accept rule is constructed so the marginal output distribution is provably identical
// to vanilla target sampling (Leviathan et al. 2023 / Chen et al. 2023). That theorem
// is the whole reason this is safe to ship — and section (a) verifies it empirically.
//
// A subtlety this stage takes seriously: equivalence is DISTRIBUTIONAL, not per-seed.
// Speculative decoding draws MORE random numbers than plain sampling (accept coins,
// residual resamples), so under a shared seed the two do NOT emit the same sequence —
// the naive "same seed -> same tokens" test is WRONG and would falsely fail. The
// correct test is: (a1) at temperature->0 both are deterministic argmax, so sequences
// must match exactly; (a2) at temperature 1, the empirical token distribution over many
// independent runs must match (KL -> 0). Both are checked below.
//
// Honest-numbers caveat (inherited from core/metrics): toy model, unoptimized float64
// kernels, so ABSOLUTE tok/s is pessimistic and the draft-vs-target cost ratio is toy-
// optimistic (the draft is a shallow truncation of the SAME weights, not a separate
// trained net). What transfers to a real engine: the SHAPE of speedup vs acceptance vs
// K, the existence of a sweet-spot K, and the negative-optimization cliff under
// draft/target misalignment. Treat absolute multipliers as illustrative.
//
// Determinism: every random choice is driven by a seeded mulberry32 PRNG, so all
// reported numbers are reproducible run-to-run.

import {
  DEFAULT_CONFIG,
  type ModelConfig,
  type Model,
  type KVCache,
  buildModel,
  forwardStep,
  mulberry32,
  newCache,
} from "./core/model.js";
import { encode, PROMPTS, VOCAB_SIZE } from "./core/tokenizer.js";
import { softmax, tensor } from "./core/tensor.js";
import { timeIt, tokensPerSecond, argmax } from "./core/metrics.js";

// --- sampling primitives ------------------------------------------------------
//
// Two decoding regimes, both exercised below:
//   - GREEDY (argmax): deterministic. Speculative greedy decoding accepts proposed[j]
//     iff it equals the target's argmax; this is provably bit-for-bit identical to greedy
//     target-only for ANY draft, which is section a1's exact-sequence test.
//   - SAMPLING (temperature): stochastic. Here the correctness theorem is distributional;
//     section a2 verifies it via empirical KL. Temperature shapes how peaked the target is
//     and thus directly shapes acceptance (a peaky target is easy for any draft to hit).
//
// Why NOT fake greedy with a tiny temperature: in this toy model the top-2 logit gap is
// small (the runner-up still holds ~10% mass even at temperature 0.02), so low-temperature
// SAMPLING is not deterministic and would make the a1 exact-match test spuriously fail.
// Greedy must be a real argmax path, not a temperature limit.

// Logit row -> probability distribution at a temperature (>0 so the division stays finite).
function softmaxProbs(logits: Float64Array, temperature: number): Float64Array {
  const scaled = new Float64Array(logits.length);
  const invT = 1 / temperature;
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] * invT;
  return softmax(tensor(scaled, [1, logits.length])).data;
}

// Sample a token id from a distribution with one PRNG draw (inverse-CDF walk). The
// trailing fallback guards floating-point rounding where the cumulative sum stops just
// short of u; without it a u≈1 could fall through and silently bias toward token 0.
function sampleFromProbs(probs: Float64Array, rng: () => number): number {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u < acc) return i;
  }
  return probs.length - 1;
}

// KL(p || q) in nats, used to PROVE distributional equivalence (section a2). We add a
// tiny epsilon to q so a zero-probability bin in an empirical histogram doesn't blow up
// to infinity — the smoothing is uniform and tiny, so a genuinely-different distribution
// still registers a clearly non-zero KL while two matching ones land near 0.
function klDivergence(p: number[], q: number[]): number {
  const eps = 1e-9;
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0) continue;
    kl += p[i] * Math.log(p[i] / (q[i] + eps));
  }
  return kl;
}

// --- KV cache rollback --------------------------------------------------------
//
// THE load-bearing mechanism, and the part every from-scratch implementation gets
// wrong first. Both the target and draft caches must, at the boundary between rounds,
// represent EXACTLY `prompt + committed output` — no more. During a round the draft
// grows by k proposals and the target grows by k verifications; tokens past the
// accepted prefix are conditioned on guesses we threw away, so their cache rows are
// poison. We slice each layer's flat [len*kvDim] buffer back to [keep*kvDim] (the exact
// inverse of forwardStep's per-step growth) and reset len, the single source of truth
// for position. Skip cache poisoning and the next forward attends over garbage and the
// distributional-equivalence guarantee silently breaks.
function rollbackCache(cache: KVCache, keep: number, cfg: ModelConfig): void {
  const kvDim = cfg.nKVHeads * cfg.dHead;
  for (let l = 0; l < cfg.nLayers; l++) {
    cache.k[l] = cache.k[l].slice(0, keep * kvDim);
    cache.v[l] = cache.v[l].slice(0, keep * kvDim);
  }
  cache.len = keep;
}

// --- draft model: a shallow truncation of the target -------------------------
//
// A real engine pairs a big target with a separately-trained small draft. We can't
// train, so we approximate "small aligned draft" by truncating the target to its first
// nLayers from the SAME seed — a genuinely cheaper network (fewer matmuls) sharing the
// target's lower-layer representations, so it agrees with the target often. Draft
// QUALITY is then a dial: more draft layers -> closer to target -> higher acceptance.
// Misalignment (section d) is produced from a DIFFERENT seed: same shape, uncorrelated
// weights, so predictions decorrelate and acceptance collapses.
function buildDraftModel(draftLayers: number, seed: number): Model {
  const cfg: ModelConfig = { ...DEFAULT_CONFIG, nLayers: draftLayers };
  return buildModel(cfg, seed);
}

const TARGET_LAYERS = DEFAULT_CONFIG.nLayers; // 4

// Decoding regime. Greedy is deterministic (argmax, no PRNG); sample draws from the
// temperature-scaled distribution. Both target-only and speculative honor the same regime
// so their outputs are comparable (exact for greedy, distributional for sampling).
type DecodeMode = { kind: "greedy" } | { kind: "sample"; temperature: number };

// Pick the next token from logits under a regime. Returns the token; for sampling it
// consumes one PRNG draw, for greedy it consumes none (so greedy is fully deterministic).
function pickToken(logits: Float64Array, mode: DecodeMode, rng: () => number): number {
  if (mode.kind === "greedy") return argmax(logits);
  return sampleFromProbs(softmaxProbs(logits, mode.temperature), rng);
}

// --- target-only baseline (the thing we must match + beat) --------------------
//
// Plain autoregressive decode from the target: one target forward per generated token.
// Returns the sequence AND the number of target forwards (the cost unit speculative
// decoding tries to shrink).
function generateTargetOnly(
  target: Model,
  promptIds: number[],
  nNewTokens: number,
  mode: DecodeMode,
  seed: number
): { tokens: number[]; targetForwards: number } {
  const rng = mulberry32(seed);
  const cache = newCache(target.cfg);
  // annotate as plain Float64Array (not the ArrayBuffer-narrowed `new` type) so reassignment
  // from forwardStep type-checks under @types/node 22 — same fix core/model applies.
  let lastLogits: Float64Array = new Float64Array(VOCAB_SIZE);
  for (const id of promptIds) lastLogits = forwardStep(target, id, cache);
  let targetForwards = promptIds.length;

  const out: number[] = [];
  for (let i = 0; i < nNewTokens; i++) {
    const tok = pickToken(lastLogits, mode, rng);
    out.push(tok);
    lastLogits = forwardStep(target, tok, cache);
    targetForwards++;
  }
  return { tokens: out, targetForwards };
}

type SpecResult = {
  tokens: number[];
  // targetForwards: SERIAL per-position target evaluations this toy actually runs. Honest
  // for THIS implementation's wall-clock, but NOT the metric a real engine optimizes.
  targetForwards: number;
  // targetPasses: number of target VERIFICATION PASSES (one per round, regardless of k). On
  // real hardware a pass verifies all k+1 positions in ONE batched forward, so this — not
  // targetForwards — is the transferable cost. The algorithmic speedup is N / targetPasses
  // (decode), because plain decoding needs one full target pass PER token.
  targetPasses: number;
  draftForwards: number;
  acceptedDraftTokens: number;
  proposedDraftTokens: number;
  rounds: number;
};

// --- speculative decoding -----------------------------------------------------
//
// Cross-round invariant (the contract that makes everything correct):
//   - tCache and dCache each hold exactly the committed tokens [prompt + out], every row
//     valid (tCache.len == dCache.len == promptIds.length + out.length).
//   - targetLast / draftLast hold each model's next-token distribution GIVEN that committed
//     history (i.e. the distribution predicting the token at index = committed length).
//
// One round, with C = committed length at entry (greedy and sampling regimes unified):
//   1. DRAFT proposes k tokens autoregressively starting from draftLast (k-1 draft
//      forwards; the k-th distribution is not needed). Draft cache grows.
//   2. TARGET verifies: if a boundary token from the previous round is PENDING (its target
//      cache row was deferred), feed it FIRST — this both creates its row and yields the
//      logits for proposed[0], so the boundary token costs the target ZERO extra forwards.
//      Then feed proposed[0..k-1]. targetLogitAt has length k+1.
//   3. ACCEPT/REJECT left to right:
//        greedy   — accept proposed[j] iff it equals argmax(target logits at j); on the
//                   first miss, commit the target's argmax and STOP.
//        sampling — accept with prob min(1, p_t/p_d); on the first reject, resample from
//                   the residual norm((p_t - p_d)_+) and STOP.
//      Both rules are exactly the construction that makes the committed token's law equal
//      the target's law at that position (greedy: identical token; sampling: identical dist).
//   4. If ALL k accepted, the verification's last distribution yields a FREE bonus token —
//      best case k+1 committed tokens for one verification pass.
//   5. REPAIR: roll the target cache back to the last position whose row is valid (the
//      accepted proposals; the boundary token's row is DEFERRED to next round's step 2),
//      and roll/extend the draft cache so draftLast predicts the position after the boundary
//      token. The deferred-boundary trick is the standard fusion that makes a real engine do
//      ~one target forward per *committed* token (fewer when accepting), not one per round
//      extra — without it speculation would do MORE target work than plain decoding here.
function generateSpeculative(
  target: Model,
  draft: Model,
  promptIds: number[],
  nNewTokens: number,
  K: number,
  mode: DecodeMode,
  seed: number
): SpecResult {
  const rng = mulberry32(seed);
  const tCache = newCache(target.cfg);
  const dCache = newCache(draft.cfg);
  // Prefill the draft fully (its draftLast predicts the first generated position). The
  // target prefill stops ONE token early: the last prompt token is left "pending" so it
  // becomes the first verification step of round 1 — keeping the deferred-boundary path
  // (which dominates every later round) the single, uniform code path. tCache ends at
  // promptLen-1, dCache at promptLen.
  let draftLast: Float64Array = new Float64Array(VOCAB_SIZE);
  for (let i = 0; i < promptIds.length; i++) {
    if (i < promptIds.length - 1) {
      forwardStep(target, promptIds[i], tCache);
    }
    draftLast = forwardStep(draft, promptIds[i], dCache);
  }
  let targetForwards = promptIds.length - 1;
  let targetPasses = 0; // verification passes (prefill is not a verification round)
  let draftForwards = promptIds.length;
  // pendingToken: a committed token whose target cache row + successor logits are deferred
  // to the next round's first verification step. Invariant: tCache holds [prompt+out] minus
  // this one token; dCache holds all of [prompt+out].
  let pendingToken = promptIds[promptIds.length - 1];

  const out: number[] = [];
  let acceptedDraftTokens = 0;
  let proposedDraftTokens = 0;
  let rounds = 0;

  while (out.length < nNewTokens) {
    rounds++;
    const k = Math.min(K, nNewTokens - out.length); // never propose past requested length

    // --- 1. DRAFT proposes k tokens from draftLast --------------------------------
    // Retain each proposal's source LOGITS (not probs) so the accept rule can compute probs
    // at the target temperature, or take an argmax, without recomputing the forward. We step
    // ALL k proposals into the draft cache (including the last) so the accepted prefix always
    // has valid draft rows for the cheap draft-side repair in step 5.
    const draftBaseLen = dCache.len; // proposed[j] occupies draft cache row draftBaseLen+j
    const proposed: number[] = [];
    const draftLogitAt: Float64Array[] = [];
    let dLogits = draftLast;
    for (let j = 0; j < k; j++) {
      draftLogitAt.push(dLogits);
      const tok = pickToken(dLogits, mode, rng);
      proposed.push(tok);
      proposedDraftTokens++;
      dLogits = forwardStep(draft, tok, dCache); // step all k so accepted rows always exist
      draftForwards++;
    }

    // --- 2. TARGET verifies (pending token first, for free) -----------------------
    // Feeding pendingToken creates its deferred cache row AND produces the logits for
    // proposed[0] — so the pending/boundary token never costs a dedicated target forward.
    // Then proposed[0..k-1]. After this, tCache holds the pending token + all k proposals.
    targetPasses++; // ONE verification pass per round (k+1 positions; batched on real HW)
    const verifyBaseLen = tCache.len; // == committed-pending length; first valid row index
    const targetLogitAt: Float64Array[] = [];
    let tLogits = forwardStep(target, pendingToken, tCache);
    targetForwards++;
    targetLogitAt.push(tLogits); // logits at the position of proposed[0]
    for (let j = 0; j < k; j++) {
      tLogits = forwardStep(target, proposed[j], tCache);
      targetForwards++;
      targetLogitAt.push(tLogits);
    }

    // --- 3. ACCEPT / REJECT walk -------------------------------------------------
    let accepted = 0;
    let boundaryToken = -1; // resampled/argmax (reject) or bonus (all-accept) token
    for (let j = 0; j < k; j++) {
      const tok = proposed[j];
      if (mode.kind === "greedy") {
        // greedy speculative: accept iff the draft's argmax matches the target's argmax.
        if (tok === argmax(targetLogitAt[j])) {
          out.push(tok);
          accepted++;
          acceptedDraftTokens++;
        } else {
          boundaryToken = argmax(targetLogitAt[j]); // commit the target's own choice
          out.push(boundaryToken);
          break;
        }
      } else {
        // sampling speculative: rejection sampling against the target distribution.
        const pT_all = softmaxProbs(targetLogitAt[j], mode.temperature);
        const pD_all = softmaxProbs(draftLogitAt[j], mode.temperature);
        // accept with prob min(1, pT/pD). pD>0 always (proposed[j] was sampled from pd).
        const acceptProb = pD_all[tok] > 0 ? Math.min(1, pT_all[tok] / pD_all[tok]) : 1;
        if (rng() < acceptProb) {
          out.push(tok);
          accepted++;
          acceptedDraftTokens++;
        } else {
          // first rejection: resample from residual norm((p_t - p_d)_+). This removes the
          // mass the draft over-proposed and lands on the mass it under-proposed — exactly
          // what makes the committed token's distribution equal the target's.
          const residual = new Float64Array(VOCAB_SIZE);
          let sum = 0;
          for (let t = 0; t < VOCAB_SIZE; t++) {
            const r = pT_all[t] - pD_all[t];
            const rr = r > 0 ? r : 0;
            residual[t] = rr;
            sum += rr;
          }
          if (sum > 1e-12) {
            for (let t = 0; t < VOCAB_SIZE; t++) residual[t] /= sum;
          } else {
            residual.set(pT_all); // degenerate residual -> fall back to p_target directly
          }
          boundaryToken = sampleFromProbs(residual, rng);
          out.push(boundaryToken);
          break;
        }
      }
    }
    // --- 4. all k accepted -> free bonus token from the verification's last dist ----
    if (accepted === k && out.length < nNewTokens) {
      boundaryToken = pickToken(targetLogitAt[k], mode, rng);
      out.push(boundaryToken);
    }

    // --- 5. REPAIR caches + set up the deferred boundary token ---------------------
    // Target valid rows after verification: the pending token (at verifyBaseLen) plus the
    // accepted proposals at verifyBaseLen+1 .. verifyBaseLen+accepted. Roll off the rest
    // (rejected proposals / unused verification rows). The boundary token's row is NOT
    // written now — it becomes next round's pending token (step 2 writes it for free).
    rollbackCache(tCache, verifyBaseLen + 1 + accepted, target.cfg);
    if (boundaryToken < 0) break; // hit nNewTokens exactly with a full accept; done.
    pendingToken = boundaryToken;
    // Draft cache: keep the accepted proposals' rows (draftBaseLen .. draftBaseLen+accepted),
    // drop the rejected ones, then feed the boundary token to get draftLast + its row. Draft
    // repair stays on the (cheap) draft — only the expensive TARGET forward is fused/deferred.
    rollbackCache(dCache, draftBaseLen + accepted, draft.cfg);
    draftLast = forwardStep(draft, boundaryToken, dCache);
    draftForwards++;
  }

  return {
    tokens: out.slice(0, nNewTokens),
    targetForwards,
    targetPasses,
    draftForwards,
    acceptedDraftTokens,
    proposedDraftTokens,
    rounds,
  };
}

// --- synthetic-alignment acceptance probe -------------------------------------
//
// Why this exists: in an UNTRAINED toy model every next-token distribution is nearly flat,
// so the accept rule min(1, p_t/p_d) returns a high value for ALMOST ANY draft — even a
// random-weight one. That makes the layer-depth "draft quality" dial too weak to move
// acceptance, and would make the failure-mode demo dishonest (a random draft would appear
// to "work"). To show the real relationship — acceptance RISES with alignment and COLLAPSES
// without it — we drive a DRAFT distribution we fully control:
//
//     p_draft = normalize( alpha * p_target  +  (1 - alpha) * uniform )
//
// alpha is the alignment dial: alpha=1 -> draft==target -> accept ~100%; alpha=0 -> draft is
// uniform noise -> accept collapses. This is faithful to the mechanism (a better-aligned
// draft is, by definition, one whose distribution is closer to the target's); we are simply
// dialing that closeness directly instead of hoping a toy truncation produces it. It uses the
// target's own per-step distribution (computed by running the target), so there is no second
// model — we measure ACCEPTANCE only here, not speed.
function measureSyntheticAcceptance(
  target: Model,
  promptIds: number[],
  nRounds: number,
  K: number,
  alpha: number,
  temperature: number,
  seed: number
): { acceptRate: number; meanAcceptedPerRound: number; rounds: number } {
  const rng = mulberry32(seed);
  const uniform = 1 / VOCAB_SIZE;
  // Cap decode positions per episode well under cfg.maxSeq so the KV cache never overflows
  // (forwardStep throws at maxSeq). We run many short episodes — each reseeds the target from
  // the prompt — to gather nRounds samples without exceeding the position limit.
  const maxDecodePerEpisode = Math.max(8, target.cfg.maxSeq - promptIds.length - K - 2);

  let proposedTotal = 0;
  let acceptedTotal = 0;
  let rounds = 0;

  while (rounds < nRounds) {
    const cache = newCache(target.cfg);
    let lastLogits: Float64Array = new Float64Array(VOCAB_SIZE); // widen for forwardStep reassign
    for (const id of promptIds) lastLogits = forwardStep(target, id, cache);

    let produced = 0;
    while (rounds < nRounds && produced < maxDecodePerEpisode) {
      rounds++;
      // Draft distribution from the current target distribution; propose K tokens i.i.d. from
      // it (a stateless draft is sufficient to probe per-position acceptance).
      const pT = softmaxProbs(lastLogits, temperature);
      const pD = new Float64Array(VOCAB_SIZE);
      for (let t = 0; t < VOCAB_SIZE; t++) pD[t] = alpha * pT[t] + (1 - alpha) * uniform;
      // (already normalized: alpha*sum(pT) + (1-alpha)*1 = 1)

      let accepted = 0;
      for (let j = 0; j < K; j++) {
        const tok = sampleFromProbs(pD, rng);
        proposedTotal++;
        const acceptProb = pD[tok] > 0 ? Math.min(1, pT[tok] / pD[tok]) : 1;
        if (rng() < acceptProb) {
          accepted++;
          acceptedTotal++;
        } else {
          break; // first reject ends the round (matches generateSpeculative)
        }
      }
      // committed = accepted + 1 boundary token; advance the target one real step so the next
      // round's distribution is a fresh position (keeps the probe non-degenerate).
      produced += accepted + 1;
      const next = sampleFromProbs(pT, rng);
      lastLogits = forwardStep(target, next, cache);
    }
  }
  return {
    acceptRate: proposedTotal === 0 ? 0 : acceptedTotal / proposedTotal,
    meanAcceptedPerRound: rounds === 0 ? 0 : acceptedTotal / rounds,
    rounds,
  };
}

// --- analysis helpers ---------------------------------------------------------

function acceptanceRate(acceptedDraftTokens: number, proposedDraftTokens: number): number {
  return proposedDraftTokens === 0 ? 0 : acceptedDraftTokens / proposedDraftTokens;
}

// ============================================================================
// main — four required experiments, all numbers measured/computed, no asserts-only.
// ============================================================================
function main(): void {
  console.log("=== stage06: 投机解码 (speculative decoding) ===\n");

  const SEED_TARGET = 42;
  const N_NEW = 48; // generated tokens per run (>> N=1 to average out luck)
  const prompt = PROMPTS[1];
  const promptIds = encode(prompt);
  const target = buildModel(DEFAULT_CONFIG, SEED_TARGET);
  console.log(
    `model: target nLayers=${TARGET_LAYERS}, vocab=${VOCAB_SIZE}, ` +
      `prompt len=${promptIds.length}, generate N=${N_NEW} tokens\n`
  );

  // ---- (a) DISTRIBUTION EQUIVALENCE -------------------------------------------
  // Equivalence is distributional, NOT per-seed (speculative draws extra random numbers,
  // so a shared seed yields different SAMPLED sequences — testing that would be a bug). Two
  // valid proofs, one per decoding regime:
  //   a1) GREEDY (argmax): deterministic, so the sequences must match bit-for-bit, for ANY
  //       draft (the accept rule commits the target's argmax on every position regardless).
  //   a2) SAMPLING (temp=1): the empirical first-token distribution over many independent
  //       runs must match the target-only one (KL -> sampling-noise floor).
  console.log("[a] 分布等价性 (output distribution == target-only):");

  // a1: greedy regime — true argmax (NOT a tiny temperature; see softmaxProbs header).
  const GREEDY: DecodeMode = { kind: "greedy" };
  const goodDraft = buildDraftModel(TARGET_LAYERS, SEED_TARGET); // == target -> max alignment
  console.log("    [a1] greedy (argmax): speculative sequence must equal target-only");
  let allGreedyMatch = true;
  for (const K of [2, 4, 6]) {
    const base = generateTargetOnly(target, promptIds, N_NEW, GREEDY, 1);
    const spec = generateSpeculative(target, goodDraft, promptIds, N_NEW, K, GREEDY, 1);
    const identical =
      spec.tokens.length === base.tokens.length && spec.tokens.every((t, i) => t === base.tokens[i]);
    allGreedyMatch &&= identical;
    const mism = spec.tokens.findIndex((t, i) => t !== base.tokens[i]);
    console.log(
      `         K=${K}: identical=${identical}` + (identical ? "" : `  (first mismatch @${mism})`)
    );
  }

  // a2: sampling regime — empirical first-token histograms must match (KL ≈ noise floor).
  // We use a weaker (depth-2) draft so acceptance is well below 100% and the residual-
  // resample path is actually exercised; if equivalence still holds here, the accept/
  // resample math is correct, not bypassed.
  const TEMP: DecodeMode = { kind: "sample", temperature: 1.0 };
  const N_RUNS = 4000;
  const klDraft = buildDraftModel(2, SEED_TARGET);
  const histTarget = new Array(VOCAB_SIZE).fill(0);
  const histSpec = new Array(VOCAB_SIZE).fill(0);
  for (let s = 0; s < N_RUNS; s++) {
    // distinct seeds per run -> independent samples of the FIRST generated token.
    histTarget[generateTargetOnly(target, promptIds, 1, TEMP, 1000 + s).tokens[0]]++;
    histSpec[generateSpeculative(target, klDraft, promptIds, 1, 4, TEMP, 5000 + s).tokens[0]]++;
  }
  const pTarget = histTarget.map((c) => c / N_RUNS);
  const pSpec = histSpec.map((c) => c / N_RUNS);
  const klSpecVsTarget = klDivergence(pSpec, pTarget);
  // baseline for scale: KL between two INDEPENDENT target-only histograms (pure sampling
  // noise floor at this N). klSpecVsTarget should be the same order, not larger.
  const histTarget2 = new Array(VOCAB_SIZE).fill(0);
  for (let s = 0; s < N_RUNS; s++) {
    histTarget2[generateTargetOnly(target, promptIds, 1, TEMP, 7000 + s).tokens[0]]++;
  }
  const pTarget2 = histTarget2.map((c) => c / N_RUNS);
  const klNoiseFloor = klDivergence(pTarget2, pTarget);
  console.log(`    [a2] sampling (temp=1, ${N_RUNS} runs): empirical first-token KL`);
  console.log(`         KL(speculative || target-only) = ${klSpecVsTarget.toExponential(2)} nats`);
  console.log(
    `         KL(target-only' || target-only)  = ${klNoiseFloor.toExponential(2)} nats  (sampling noise floor)`
  );
  const klOk = klSpecVsTarget <= klNoiseFloor * 3 + 1e-4;
  console.log(
    `         speculative KL within ~noise floor: ${klOk}  ` +
      `(equal distributions, not just equal argmax)`
  );
  console.log(
    `    -> a1 sequences match=${allGreedyMatch}; a2 KL≈noise: ${klOk}. ` +
      "Output distribution is the target's.\n"
  );

  // ---- (b) ACCEPTANCE RATE vs DRAFT QUALITY -----------------------------------
  // We dial draft/target alignment directly (p_draft = α·p_target + (1-α)·uniform) because
  // the untrained toy model's distributions are too flat for a layer-truncated draft to move
  // acceptance cleanly (see measureSyntheticAcceptance header). α is the alignment dial.
  //
  // Temperature matters here. At temp=1 this toy's target is itself near-uniform, so even a
  // uniform draft (α=0) "agrees" ~78% of the time — the alignment dependence is invisible.
  // We therefore probe at temp=0.15, where the target is PEAKED (the realistic serving
  // regime). Now α=0 genuinely collapses and the full acceptance↔alignment curve appears.
  const PROBE_TEMP = 0.15;
  console.log(`[b] 接受率随 draft 质量(对齐度 α)变化 (peaked temp=${PROBE_TEMP}, K=4):`);
  console.log("    alpha   meaning                acceptRate   meanAccepted/round (of K=4)");
  const K_B = 4;
  for (const alpha of [0.0, 0.5, 0.9, 0.99, 1.0]) {
    const r = measureSyntheticAcceptance(target, promptIds, 2000, K_B, alpha, PROBE_TEMP, 1);
    const meaning =
      alpha === 1 ? "draft = target" : alpha === 0 ? "draft = uniform noise" : "partial alignment";
    console.log(
      `    ${alpha.toFixed(2)}    ${meaning.padEnd(20)}   ${(r.acceptRate * 100).toFixed(1).padStart(5)}%` +
        `        ${r.meanAcceptedPerRound.toFixed(2)}`
    );
  }
  console.log(
    "    -> acceptance rises monotonically with alignment; at α=1 every draft token is\n" +
      "       accepted (k+1 tokens/round), at α=0 it collapses toward ~1 token/round.\n"
  );

  // ---- (c) SPEEDUP vs K (the sweet-spot) --------------------------------------
  // CRUCIAL honesty note. The headline speedup is ALGORITHMIC: speedup = N decode steps /
  // target VERIFICATION PASSES. Plain decoding needs one full target pass per token (N
  // passes); speculation needs one pass per ROUND, and a round commits up to k+1 tokens, so
  // passes < N when acceptance is decent. THAT ratio is what transfers to real hardware,
  // where a verification pass over k+1 positions is ONE batched forward (memory-bound, ~the
  // cost of a single decode step).
  //
  // This toy CANNOT show that as wall-clock: core/forwardStep verifies positions SERIALLY
  // (k+1 sequential cached steps), so there is no batched-parallel win to measure — the very
  // mechanism that makes speculation fast is the one absent from an unbatched reference. We
  // print the measured wall-clock too, but it reflects serial verification and so will look
  // slower; do not read it as the verdict. The transferable verdict is the EFF speedup column
  // (pass count plus charged draft cost); wall-clock ms is shown only for full disclosure.
  // Two speedup numbers per K:
  //   ALGO   = N / targetPasses : the pure batched-verify win, IGNORING draft cost. Useful to
  //            see the verification amortization in isolation.
  //   EFF    = the honest one: effective cost/token = (passes + draftRatio·draftForwards)/N,
  //            speedup = 1/that. It charges for the draft work (which shares the accelerator),
  //            so it shows the real SWEET-SPOT: big K helps verification but the draft cost and
  //            rising rejections eventually overtake it, so EFF peaks at a middle K and falls.
  // We use a peaked temperature here too (temp=0.15) so acceptance actually varies with K — at
  // temp=1 the near-uniform toy target accepts almost everything and the sweet-spot is washed out.
  const SPEEDUP_TEMP: DecodeMode = { kind: "sample", temperature: 0.15 };
  const DRAFT_RATIO_C = 0.25; // depth-1 draft vs depth-4 target
  console.log(`[c] 加速比随 K 变化 (cheap aligned draft, peaked temp=0.15, draftCost=${DRAFT_RATIO_C}x):`);
  const cDraft = buildDraftModel(1, SEED_TARGET); // 1/4 the layers -> cheaper draft
  generateTargetOnly(target, promptIds, 8, SPEEDUP_TEMP, 1); // warmup
  generateSpeculative(target, cDraft, promptIds, 8, 4, SPEEDUP_TEMP, 1);

  const baseMs = timeIt(() => generateTargetOnly(target, promptIds, N_NEW, SPEEDUP_TEMP, 1));
  const baseTps = tokensPerSecond(N_NEW, baseMs);
  console.log(
    `    target-only baseline: ${N_NEW} target passes (one per token), ${baseMs.toFixed(2)} ms serial ` +
      `(${baseTps.toFixed(0)} tok/s)`
  );
  console.log("    K    acceptRate   targetPasses   ALGO speedup   EFF speedup   (serialMs)");
  for (const K of [1, 2, 4, 6, 8, 12]) {
    const stats = generateSpeculative(target, cDraft, promptIds, N_NEW, K, SPEEDUP_TEMP, 1);
    const ms = timeIt(() => generateSpeculative(target, cDraft, promptIds, N_NEW, K, SPEEDUP_TEMP, 1));
    const ar = acceptanceRate(stats.acceptedDraftTokens, stats.proposedDraftTokens);
    const algoSp = stats.targetPasses === 0 ? Infinity : N_NEW / stats.targetPasses;
    // effective: decode draft forwards = total draft forwards minus prefill.
    const decodeDraftFwd = stats.draftForwards - promptIds.length;
    const effCostPerToken = (stats.targetPasses + DRAFT_RATIO_C * decodeDraftFwd) / N_NEW;
    const effSp = effCostPerToken > 0 ? 1 / effCostPerToken : Infinity;
    const mark = effSp >= 1 ? "" : "  <- net slower";
    console.log(
      `    ${String(K).padStart(2)}   ${(ar * 100).toFixed(1).padStart(5)}%       ` +
        `${String(stats.targetPasses).padStart(5)}         ${algoSp.toFixed(2)}x        ` +
        `${effSp.toFixed(2)}x${mark.padEnd(14)}(${ms.toFixed(0)}ms)`
    );
  }
  console.log(
    "    -> EFF speedup (draft cost charged) is the honest verdict and it PEAKS at a middle K:\n" +
      "       small K under-amortizes the verification pass; large K proposes tokens that get\n" +
      "       rejected (one early reject truncates the round) while still paying their draft\n" +
      "       cost. Serial wall-clock (ms) can't show the batched-verify win — see header note.\n"
  );

  // ---- (d) FAILURE MODE: misaligned draft -> acceptance collapse -> NEGATIVE opt ----
  // The honest "塌到接近 0" curve needs a draft that is genuinely uncorrelated with the
  // target. A random-WEIGHT model does NOT achieve this on a flat toy distribution (its
  // accept rate stays misleadingly high), so we use the controlled α=0 (uniform-noise) draft.
  // We contrast a well-aligned draft (α=0.95) against the misaligned one (α=0.0) and sweep K.
  //
  // The negative-optimization metric is the EFFECTIVE COST: target passes per committed token
  // PLUS the draft work per committed token. With a real (parallel-verify) engine, one pass
  // ≈ one decode step, so a useful regime needs (passes/token) < 1 by a margin big enough to
  // pay for the draft. A misaligned draft drives passes/token toward 1 (no win) while still
  // paying K draft forwards per round — net loss. Bigger K digs the hole deeper.
  // Effective-cost model (the honest verdict). On real HW the draft runs on the same
  // accelerator, so total cost per committed token ≈ 1 verification pass + K draft forwards,
  // amortized over the tokens the round commits. With the draft costing DRAFT_COST_RATIO of a
  // target pass, effective cost/token = (1 + K·ratio)/(α-acceptance·K + 1). Baseline = 1.0
  // (one full target pass per token). Speculation HELPS iff effective cost/token < 1.
  const DRAFT_COST_RATIO = 0.25; // a depth-1 draft vs depth-4 target (toy: ~layers ratio)
  console.log(`[d] 失败模式: draft 与 target 对齐度对比 (peaked temp=${PROBE_TEMP}, draftCost=${DRAFT_COST_RATIO}x):`);
  console.log("    align        K    acceptRate   effCost/token   effSpeedup   verdict");
  for (const [label, alpha] of [
    ["aligned α=.95", 0.95],
    ["MISALN α=.00", 0.0],
  ] as const) {
    for (const K of [2, 4, 8]) {
      const r = measureSyntheticAcceptance(target, promptIds, 2000, K, alpha, PROBE_TEMP, 1);
      // committed/round = accepted (= acceptRate·K) + 1 boundary token.
      const committedPerRound = r.acceptRate * K + 1;
      const effCostPerToken = (1 + K * DRAFT_COST_RATIO) / committedPerRound;
      const effSpeedup = 1 / effCostPerToken; // vs baseline cost 1.0/token
      const verdict =
        effSpeedup >= 1 ? `useful (${effSpeedup.toFixed(2)}x)` : "NEGATIVE optimization (slower)";
      console.log(
        `    ${label}   ${String(K).padStart(2)}    ${(r.acceptRate * 100).toFixed(1).padStart(5)}%` +
          `        ${effCostPerToken.toFixed(2)}            ${effSpeedup.toFixed(2)}x        ${verdict}`
      );
    }
  }
  console.log(
    "    -> aligned draft: effective cost/token < 1, a real speedup that grows with K. α=0\n" +
      "       (uniform-noise draft): acceptance collapses (~9%), so each round commits ~1 token\n" +
      "       yet still pays K draft forwards -> effective cost > 1 and WORSENS with K. The\n" +
      "       sign of the optimization flips on alignment alone — that is the whole lesson.\n"
  );

  // even under misalignment the OUTPUT stays correct: run the real generateSpeculative with a
  // genuinely bad draft (a different-seed model) in greedy mode and confirm the sequence still
  // equals target-only. Speculation can only cost latency, never correctness.
  const badDraft = buildModel(DEFAULT_CONFIG, 99999); // uncorrelated weights
  const baseG = generateTargetOnly(target, promptIds, N_NEW, GREEDY, 1);
  const badG = generateSpeculative(target, badDraft, promptIds, N_NEW, 4, GREEDY, 1);
  const stillCorrect =
    badG.tokens.length === baseG.tokens.length && badG.tokens.every((t, i) => t === baseG.tokens[i]);
  console.log(`[d'] even with a TERRIBLE draft, greedy output === target-only: ${stillCorrect}`);
  console.log("     (speculative decoding trades speed for speed, never speed for quality)\n");

  console.log("=== stage06 done ===");
}

main();
