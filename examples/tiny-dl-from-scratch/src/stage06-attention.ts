// stage06-attention.ts — Single-head causal self-attention, built ONLY from core/autograd
// Tensor ops, with a grad-check to prove the hand-derived backward is correct.
//
// WHY this stage exists: attention looks exotic but it is just three linear projections,
//   one scaled matmul, a masked row-wise softmax, and one more matmul. Every piece already
//   has a tested adjoint in core/autograd.ts, so if we assemble them faithfully, autograd
//   gives us the attention gradient for free — and gradCheck confirms we wired it right.
//
// SHAPE CONVENTION (the load-bearing simplification): core's matmul/transpose/softmax are
//   2-D ONLY. So we model ONE sequence at a time as a 2-D (T, d_model) matrix — batch=1.
//   A real GPT folds (B, T) into (B*T, d) for the projections and loops heads; here we keep
//   B=1 and a single head to make the attention matrix (T, T) directly printable and the
//   causal structure visible by eye. The mechanism is identical; only the bookkeeping grows.
//
// THREE THINGS THIS FILE PROVES WITH REAL NUMBERS:
//   1. gradCheck on the whole head (Q/K/V/out projections) < 1e-6 — the backward is honest.
//   2. Causal mask => the attention matrix is strictly lower-triangular (future weight = 0).
//   3. The 1/sqrt(d_k) scale is not cosmetic: without it, as d_k grows the softmax collapses
//      toward one-hot (entropy -> 0), which is exactly where its gradient vanishes.
//   Plus a FAILURE demo: masking AFTER softmax (instead of before) leaks attention to the
//   future — we measure the leaked probability mass so the bug is a number, not a vibe.
//
// HONESTY ON NUMBERS: weights/inputs come from a seeded PRNG, so every printed value is
//   reproducible run-to-run. The entropy/collapse trend is a real measurement of the softmax
//   on Gaussian logits; absolute entropies depend on the toy variance we feed, but the
//   monotone collapse as d_k grows is the transferable signal (same reason GPT scales by
//   1/sqrt(d_k)). No wall-clock claims are made here — this stage is about correctness, not
//   speed.

import { mulberry32, randn, type Rng } from "./core/rng.js";
import { Tensor } from "./core/autograd.js";
import { Linear, Module } from "./core/nn.js";
import { gradCheck } from "./core/metrics.js";

// ============================================================================
// Causal mask
// ============================================================================

// A constant additive mask: 0 on/below the diagonal, NEG on the strict upper triangle.
// WHY additive-before-softmax (not multiplicative-after): softmax must NEVER see the future
//   logits at all. Adding a large negative pushes exp(...) to ~0 BEFORE normalization, so the
//   denominator excludes future positions and the surviving weights still sum to 1 over the
//   allowed (past+self) positions. Zeroing AFTER softmax instead would renormalize wrongly —
//   the future already contributed to the denominator (this is the bug we demo at the end).
// WHY -1e9 and not -Infinity: softmax subtracts the row max before exp; a row that is ALL
//   future-masked never happens here (the diagonal is always allowed), but -1e9 keeps the
//   arithmetic finite (Infinity - Infinity = NaN) while still giving exp() an effective 0.
const MASK_NEG = -1e9;

function causalMask(seqLen: number): Tensor {
  const m = new Float64Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      // position i may attend to j only if j <= i (no peeking at the future).
      m[i * seqLen + j] = j <= i ? 0 : MASK_NEG;
    }
  }
  return new Tensor(m, [seqLen, seqLen]);
}

// ============================================================================
// Single-head self-attention
// ============================================================================

// Scaled dot-product attention for ONE sequence x:(T, dModel).
//   scores = (Q @ K^T) * (1/sqrt(dK))   -> (T, T)
//   weights = softmax(scores + mask)     -> (T, T), row-wise
//   context = weights @ V                -> (T, dV)
// `causal` toggles the mask; `scale` toggles the 1/sqrt(dK) factor so we can A/B it.
// Returns both the context AND the weights so callers can inspect/print the attention matrix.
class SelfAttentionHead extends Module {
  wq: Linear;
  wk: Linear;
  wv: Linear;
  wo: Linear;
  dK: number;

  constructor(dModel: number, dK: number, rng: Rng) {
    super();
    // No bias on projections — standard for attention; biases add nothing the LN/residual
    // around the block can't, and omitting them keeps the grad-check surface minimal.
    this.wq = this.child(new Linear(dModel, dK, rng, { bias: false, init: "xavier" }));
    this.wk = this.child(new Linear(dModel, dK, rng, { bias: false, init: "xavier" }));
    this.wv = this.child(new Linear(dModel, dK, rng, { bias: false, init: "xavier" }));
    this.wo = this.child(new Linear(dK, dModel, rng, { bias: false, init: "xavier" }));
    this.dK = dK;
  }

  // INVARIANT: x is 2-D (T, dModel). Returns the projected context (T, dModel).
  override forward(x: Tensor): Tensor {
    return this.attend(x, true, true).context;
  }

  // The explicit variant used by the experiments: exposes the weight matrix and lets us
  // turn the scale / mask on and off. `maskAfter=true` deliberately reproduces the bug
  // (mask the WEIGHTS after softmax) so we can measure the leak.
  attend(
    x: Tensor,
    causal: boolean,
    scale: boolean,
    maskAfter = false,
  ): { context: Tensor; weights: Tensor } {
    const T = x.shape[0];
    const q = this.wq.forward(x); // (T, dK)
    const k = this.wk.forward(x); // (T, dK)
    const v = this.wv.forward(x); // (T, dV=dK)

    let scores = q.matmul(k.transpose()); // (T, T): scores[i,j] = q_i · k_j
    if (scale) scores = scores.mulScalar(1 / Math.sqrt(this.dK));

    let weights: Tensor;
    if (causal && !maskAfter) {
      // CORRECT path: bias the logits before softmax so the future is never normalized in.
      weights = scores.add(causalMask(T)).softmax();
    } else if (causal && maskAfter) {
      // BUGGY path (on purpose): softmax over ALL positions, THEN zero the future. The
      // denominator already counted future logits, so surviving weights don't sum to 1 and
      // information about how "confident" the model was in the future leaks into the scale
      // of the past weights. We multiply by the 0/1 lower-triangular keep-mask.
      const full = scores.softmax();
      const keep = lowerTriangularKeep(T);
      weights = full.mul(keep);
    } else {
      weights = scores.softmax();
    }

    const context = weights.matmul(v); // (T, dV)
    const out = this.wo.forward(context); // (T, dModel)
    return { context: out, weights };
  }
}

// 0/1 mask: 1 on/below diagonal, 0 above. Used ONLY by the buggy "mask after softmax" path.
function lowerTriangularKeep(seqLen: number): Tensor {
  const m = new Float64Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) for (let j = 0; j <= i; j++) m[i * seqLen + j] = 1;
  return new Tensor(m, [seqLen, seqLen]);
}

// ============================================================================
// helpers for the experiments
// ============================================================================

function formatMatrix(t: Tensor, rows: number, cols: number): string {
  const lines: string[] = [];
  for (let i = 0; i < rows; i++) {
    const cells: string[] = [];
    for (let j = 0; j < cols; j++) cells.push(t.data[i * cols + j].toFixed(3));
    lines.push("  [ " + cells.join("  ") + " ]");
  }
  return lines.join("\n");
}

function section(title: string): void {
  console.log("\n" + "=".repeat(64) + "\n" + title + "\n" + "=".repeat(64));
}

// ============================================================================
// main
// ============================================================================

function main(): void {
  // ----------------------------------------------------------------------
  section("1) gradCheck the whole attention head (analytic vs numerical < 1e-6)");
  // If this passes, every adjoint in the head (3+1 projections, scaled matmul, masked
  // softmax, weighted-sum matmul) composes correctly — we did NOT hand-derive a single
  // gradient, autograd did, and the numbers agree.
  const gcRng = mulberry32(42);
  const Tgc = 4;
  const dModelGc = 6;
  const head = new SelfAttentionHead(dModelGc, 5, gcRng);
  const xGc = Tensor.from([Tgc, dModelGc], () => randn(gcRng));
  // Scalar loss = sum of squared context (any scalar reduction lets backward seed grad=1).
  const lossFn = () => head.attend(xGc, true, true).context.mul(head.attend(xGc, true, true).context).sum().data[0];
  const lossT = (() => {
    const c = head.attend(xGc, true, true).context;
    return c.mul(c).sum();
  })();
  head.zeroGrad();
  lossT.backward();
  const params = head.parameters();
  const gc = gradCheck(lossFn, params, 1e-5, 6);
  console.log(`params checked: ${params.length} tensors, ${gc.checked} elements sampled`);
  console.log(`scalar loss: ${lossT.data[0].toFixed(4)}`);
  console.log(`max relative error: ${gc.maxRelError.toExponential(3)}`);
  console.log(`PASS (< 1e-6): ${gc.maxRelError < 1e-6}`);

  // ----------------------------------------------------------------------
  section("2) causal mask => strictly lower-triangular attention (future weight = 0)");
  // Feed a deterministic toy sequence and print the (T,T) weight matrix. The contract:
  // every entry strictly ABOVE the diagonal must be exactly 0 — position i cannot see j>i.
  const seqRng = mulberry32(7);
  const T = 6;
  const dModel = 8;
  const headC = new SelfAttentionHead(dModel, 8, seqRng);
  const x = Tensor.from([T, dModel], () => randn(seqRng));
  const { weights } = headC.attend(x, true, true);

  console.log("attention weight matrix (row = query position, col = key position):");
  console.log(formatMatrix(weights, T, T));

  let maxUpper = 0; // largest weight anywhere in the strict upper triangle (must stay 0)
  let rowSumsOk = true;
  for (let i = 0; i < T; i++) {
    let rowSum = 0;
    for (let j = 0; j < T; j++) {
      const w = weights.data[i * T + j];
      if (j > i) maxUpper = Math.max(maxUpper, w);
      rowSum += w;
    }
    // Each row must still be a valid distribution over the allowed (past+self) keys.
    if (Math.abs(rowSum - 1) > 1e-9) rowSumsOk = false;
  }
  console.log(`max weight in strict upper triangle (future): ${maxUpper.toExponential(3)}`);
  console.log(`strictly lower-triangular (future == 0): ${maxUpper === 0}`);
  console.log(`every row sums to 1 over allowed keys: ${rowSumsOk}`);

  // ----------------------------------------------------------------------
  section("3) scaling vs no-scaling: does softmax collapse as d_k grows?");
  // The mechanism the 1/sqrt(d_k) factor defends against: a score q·k is a sum of d_k
  // products of (roughly) unit-variance numbers, so Var(q·k) ~ d_k and the logits spread
  // wider as d_k grows. A softmax over wider-spread logits saturates toward one-hot, whose
  // entropy -> 0 AND whose Jacobian -> 0 (vanishing gradient). Dividing by sqrt(d_k) rescales
  // the logit std back to ~1 regardless of d_k, keeping the softmax in its responsive regime.
  //
  // We measure this DIRECTLY on attention logits: draw q,k ~ N(0,1)^{d_k} (exactly the
  // distribution scaled-dot-product attention assumes after well-conditioned projections),
  // form one query's scores against Tctx keys, softmax, and report the entropy. Averaging
  // over many random queries removes single-draw luck (N=1 would be noise, not signal).
  const dKs = [4, 16, 64, 256, 1024];
  const Tctx = 8; // a query attends over this many keys
  const trials = 2000; // many independent (q, K) draws so the entropy is a stable average
  const lnT = Math.log(Tctx); // entropy of a uniform distribution over Tctx keys (the ceiling)
  const entRng = mulberry32(2024);

  // Entropy of softmax over `scores` (length n), in nats.
  function softmaxEntropy(scores: number[]): number {
    let max = -Infinity;
    for (const s of scores) max = Math.max(max, s);
    let denom = 0;
    const p = scores.map((s) => {
      const e = Math.exp(s - max);
      denom += e;
      return e;
    });
    let h = 0;
    for (const e of p) {
      const pr = e / denom;
      if (pr > 1e-12) h -= pr * Math.log(pr);
    }
    return h;
  }

  console.log(`(reference: uniform-over-${Tctx} entropy = ln(${Tctx}) = ${lnT.toFixed(4)} nats)`);
  console.log(`(averaged over ${trials} random (query, keys) draws per d_k)`);
  console.log("d_k     H(no-scale)   H(scaled)    note");
  for (const dK of dKs) {
    let sumHun = 0;
    let sumHsc = 0;
    const invSqrt = 1 / Math.sqrt(dK);
    for (let t = 0; t < trials; t++) {
      // one query vector vs Tctx key vectors, all iid N(0,1) — the canonical assumption.
      const q = new Float64Array(dK);
      for (let d = 0; d < dK; d++) q[d] = randn(entRng);
      const scoresUn: number[] = [];
      for (let j = 0; j < Tctx; j++) {
        let dot = 0;
        for (let d = 0; d < dK; d++) dot += q[d] * randn(entRng);
        scoresUn.push(dot);
      }
      sumHun += softmaxEntropy(scoresUn);
      sumHsc += softmaxEntropy(scoresUn.map((s) => s * invSqrt)); // same logits, just scaled
    }
    const hUn = sumHun / trials;
    const hSc = sumHsc / trials;
    const collapsing = hUn < hSc * 0.6 ? "<< unscaled collapsing toward one-hot" : "";
    console.log(
      `${String(dK).padEnd(7)} ${hUn.toFixed(4).padEnd(13)} ${hSc.toFixed(4).padEnd(12)} ${collapsing}`,
    );
  }
  console.log(
    "Reading: UNSCALED entropy falls toward 0 as d_k grows (softmax -> one-hot, dead gradient);",
  );
  console.log("SCALED entropy is pinned near the uniform ceiling for every d_k. Hence /sqrt(d_k).");

  // Make the gradient consequence concrete. The softmax of one row has Jacobian-vector
  // backward dL/dx_i = s_i (g_i - sum_j s_j g_j). A near-one-hot s (unscaled, high d_k) drives
  // every dx_i toward 0 (the saturated entry has s_i≈1 but g_i-dot≈0; the rest have s_i≈0),
  // so almost no signal reaches the logits — that is the vanishing gradient. We feed a
  // NON-uniform upstream g (g_j = j) on purpose: a constant g would give 0 grad for ANY
  // softmax (it is shift-invariant), which would hide the effect rather than reveal it.
  const dKbig = 1024;
  function softmaxBackwardNorm(scale: boolean): number {
    const rng = mulberry32(7777);
    const q = new Float64Array(dKbig);
    for (let d = 0; d < dKbig; d++) q[d] = randn(rng);
    const raw: number[] = [];
    for (let j = 0; j < Tctx; j++) {
      let dot = 0;
      for (let d = 0; d < dKbig; d++) dot += q[d] * randn(rng);
      raw.push(dot);
    }
    const logits = scale ? raw.map((s) => s / Math.sqrt(dKbig)) : raw;
    // softmax forward
    let max = -Infinity;
    for (const s of logits) max = Math.max(max, s);
    let denom = 0;
    const s = logits.map((v) => {
      const e = Math.exp(v - max);
      denom += e;
      return e;
    });
    for (let i = 0; i < s.length; i++) s[i] /= denom;
    // backward with a NON-uniform fixed upstream grad (g_j = j) so the Jacobian magnitude
    // shows; a constant g cancels out by softmax shift-invariance (g_i - dot == 0).
    const g = s.map((_, j) => j);
    let dot = 0;
    for (let i = 0; i < s.length; i++) dot += s[i] * g[i];
    let nrm = 0;
    for (let i = 0; i < s.length; i++) {
      const dx = s[i] * (g[i] - dot); // softmax adjoint, same formula core uses
      nrm += dx * dx;
    }
    return Math.sqrt(nrm);
  }
  const gnUnscaled = softmaxBackwardNorm(false);
  const gnScaled = softmaxBackwardNorm(true);
  console.log(`\nsoftmax input-grad norm for one row @ d_k=${dKbig} (non-uniform upstream g_j=j):`);
  console.log(`  unscaled: ${gnUnscaled.toExponential(3)}   scaled: ${gnScaled.toExponential(3)}`);
  console.log(
    `  scaled passes ~${(gnScaled / (gnUnscaled || 1e-30)).toFixed(0)}x more gradient through the softmax`,
  );

  // ----------------------------------------------------------------------
  section("4) FAILURE demo: masking AFTER softmax leaks attention to the future");
  // The seductive-but-wrong implementation: softmax over the FULL score row, then multiply
  // by a 0/1 lower-triangular keep-mask. Two observable defects:
  //   (a) the kept (past) weights no longer sum to 1 — the missing mass is exactly what the
  //       future positions stole from the denominator;
  //   (b) equivalently, the future absorbed real probability mass that should never have
  //       existed. We report that stolen mass per row.
  const bugRng = mulberry32(7);
  const headBug = new SelfAttentionHead(dModel, 8, bugRng);
  const xbRng = mulberry32(7); // single rng instance -> distinct draws (NOT one seed per cell)
  const xb = Tensor.from([T, dModel], () => randn(xbRng)); // a real (non-constant) toy seq
  const correct = headBug.attend(xb, true, true).weights; // mask-before (right)
  const buggy = headBug.attend(xb, true, true, true).weights; // mask-after  (wrong)

  console.log("correct (mask-before) row sums vs buggy (mask-after) row sums:");
  console.log("row   correct_sum   buggy_sum   leaked_mass(=1 - buggy_sum)");
  let totalLeak = 0;
  for (let i = 0; i < T; i++) {
    let cSum = 0;
    let bSum = 0;
    let futureMassBeforeZeroing = 0;
    for (let j = 0; j < T; j++) {
      cSum += correct.data[i * T + j];
      bSum += buggy.data[i * T + j];
    }
    // The leaked mass is what the future grabbed in the (wrong) full softmax denominator:
    // it equals 1 - (sum of kept weights), because the full softmax summed to 1 and we then
    // deleted the future entries without renormalizing.
    futureMassBeforeZeroing = 1 - bSum;
    totalLeak += futureMassBeforeZeroing;
    console.log(
      `${String(i).padEnd(5)} ${cSum.toFixed(4).padEnd(13)} ${bSum.toFixed(4).padEnd(11)} ${futureMassBeforeZeroing.toFixed(4)}`,
    );
  }
  console.log(`\nrow 0 leaks the most (it can see 5 future tokens); the last row leaks ~0.`);
  console.log(`total probability mass stolen by the future across rows: ${totalLeak.toFixed(4)}`);
  console.log(
    `>> Lesson: add the mask to the LOGITS before softmax. Masking after softmax silently`,
  );
  console.log(
    `   reweights the past (each row no longer sums to 1) and is a real, measurable bug.`,
  );

  console.log("\nALL STAGE-06 ATTENTION CHECKS DONE.\n");
}

main();
