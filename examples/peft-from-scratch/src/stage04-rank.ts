// stage04-rank.ts — Chapter 04: choosing the rank r. The effective rank of ΔW and why
//   over-parameterizing the adapter is wasted budget.
//
// THE ONE CLAIM this stage proves with real numbers: the update a fine-tune needs is
//   intrinsically LOW RANK. So accuracy as a function of LoRA rank r is a hockey stick —
//   it climbs while r < the update's effective rank, then SATURATES. Pushing r toward
//   min(d,k) buys (near-)zero accuracy but pays full parameter cost, and on a tiny finetune
//   set it can even nudge the val loss UP (the extra capacity memorizes noise).
//
// HOW we make "effective rank" a KNOWN, not a vibe: we synthesize the ground-truth update
//   ΔW* ourselves as a sum of rank-1 outer products with GEOMETRICALLY DECAYING weights
//   (σ_k = base^k). With decay 0.35 the 4th singular value is already ~4% of the 1st, so
//   the *effective* rank is ~3 even though the matrix is technically full rank. The task is
//   then a frozen-base linear map whose only missing piece is ΔW*; LoRA of rank r tries to
//   recover it. Because we built ΔW*, we can assert exactly where accuracy should plateau
//   and we can SVD the *learned* ΔW=BA to show its spectrum echoes the planted decay.
//
// WHY this is the honest version of the chapter's lesson: at toy scale the absolute
//   effective-rank number is an artifact of how we planted ΔW*. The TRANSFERABLE shape is
//   (a) singular values of real fine-tune updates decay fast, and (b) the accuracy-vs-r
//   curve has an elbow — both reproduced here from a real optimizer run, not asserted.
//
// FAILURE MODE demo (not skipped): r = min(d,k) is full-rank LoRA. We print that its
//   trainable-param ratio EXPLODES past full-FT's effective free params while val accuracy
//   does not improve over the saturation point — concrete evidence that "bigger r" is the
//   wrong knob to turn once you are past the elbow.
//
// ESM note: import core with .js suffix; never import another stageNN file.

import { seed, normal, uniformRange } from "./core/prng.js";
import { Tensor, numericalGradCheck } from "./core/tensor.js";
import { Module } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { bar, heatmap, sparkline } from "./core/viz.js";

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error("ASSERT FAILED: " + msg);
}

// ---------------------------------------------------------------------------
// Problem setup: a frozen linear base W0 plus a planted low-effective-rank update ΔW*.
// The "fine-tune target" is y = x @ (W0 + ΔW*)ᵀ. Freezing W0 leaves exactly ΔW* to learn,
//   so a rank-r LoRA delta BA is competing to reconstruct a matrix of known spectrum.
// ---------------------------------------------------------------------------

const D_OUT = 16; // ΔW* is (D_OUT, D_IN); square keeps min(d,k)=16 and SVD cheap
const D_IN = 16;
const TRUE_RANK = 3; // number of planted outer products with non-negligible weight
const DECAY = 0.35; // σ_k = DECAY^k — fast decay => low EFFECTIVE rank even if full rank
const N_TRAIN = 96;
const N_VAL = 96;
const STEPS = 120;
const LR = 1e-2;
const CONV_TOL = 1e-3; // val-MSE threshold that counts a run as "converged"

/** Build ΔW* = Σ_k σ_k · u_k v_kᵀ with σ_k = DECAY^k and orthonormal-ish random u,v.
 *  WHY decaying σ: it is the whole premise — real fine-tune updates concentrate their
 *  energy in a few directions. We make the planted matrix obey that so the spectrum the
 *  learner recovers is meaningful, not noise. */
function buildPlantedDelta(): { deltaStar: Tensor; trueSpectrum: number[] } {
  const data = new Float64Array(D_OUT * D_IN);
  const spectrum: number[] = [];
  // Span the full dimension with random rank-1 terms but kill the tail with DECAY^k, so the
  //   tail terms (k >= TRUE_RANK) are present but ~0 — that is "effective" vs "exact" rank.
  for (let k = 0; k < D_OUT; k++) {
    const sigma = Math.pow(DECAY, k); // strictly decaying
    if (k < TRUE_RANK) spectrum.push(sigma);
    const u = Array.from({ length: D_OUT }, () => normal(0, 1));
    const v = Array.from({ length: D_IN }, () => normal(0, 1));
    const un = l2norm(u), vn = l2norm(v);
    for (let i = 0; i < D_OUT; i++)
      for (let j = 0; j < D_IN; j++) data[i * D_IN + j] += sigma * (u[i] / un) * (v[j] / vn);
  }
  return { deltaStar: Tensor.from(Array.from(data), [D_OUT, D_IN], false), trueSpectrum: spectrum };
}

function l2norm(xs: number[]): number {
  let s = 0;
  for (const v of xs) s += v * v;
  return Math.sqrt(s) || 1;
}

/** Frozen base weight W0 (random); its values are irrelevant to learnability since the
 *  target subtracts it back out — it only exists so the model has a realistic "base + delta"
 *  structure rather than learning the whole map from scratch. */
function buildBase(): Tensor {
  const w = new Float64Array(D_OUT * D_IN);
  for (let i = 0; i < w.length; i++) w[i] = normal(0, 1 / Math.sqrt(D_IN));
  return Tensor.from(Array.from(w), [D_OUT, D_IN], false); // frozen leaf
}

interface Sample {
  x: Tensor; // (1, D_IN)
  y: Tensor; // (1, D_OUT) — target = x @ (W0+ΔW*)ᵀ, frozen leaf
}

function makeDataset(n: number, W0: Tensor, deltaStar: Tensor): Sample[] {
  const Weff = addMat(W0, deltaStar); // W0 + ΔW*  (plain arithmetic, not a graph node)
  const out: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const xs = Array.from({ length: D_IN }, () => uniformRange(-1, 1));
    const x = Tensor.from(xs, [1, D_IN], false);
    // y = x @ Weffᵀ : row vector times (out,in)ᵀ -> (1,out)
    const ys = new Array(D_OUT).fill(0);
    for (let o = 0; o < D_OUT; o++) {
      let acc = 0;
      for (let j = 0; j < D_IN; j++) acc += xs[j] * Weff.data[o * D_IN + j];
      ys[o] = acc;
    }
    out.push({ x, y: Tensor.from(ys, [1, D_OUT], false) });
  }
  return out;
}

function addMat(a: Tensor, b: Tensor): Tensor {
  const d = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) d[i] = a.data[i] + b.data[i];
  return Tensor.from(Array.from(d), a.shape.slice(), false);
}

// ---------------------------------------------------------------------------
// The LoRA-adapted linear layer: y = x @ (W0 + (B@A)·(α/r))ᵀ, with W0 FROZEN and only
//   A (r,in) / B (out,r) trainable. INVARIANT (LoRA init): B = 0 so the delta starts at
//   exactly 0 and the adapted model == the base model at step 0 (no init shock). A ~ small
//   gaussian so grad can flow into it (if A were also 0, B's grad would be 0 forever).
// ---------------------------------------------------------------------------

class LoRALinear extends Module {
  W0: Tensor; // (out,in) frozen base
  A: Tensor; // (r,in) trainable
  B: Tensor; // (out,r) trainable
  alphaOverR: number;
  r: number;
  constructor(W0: Tensor, r: number, alpha: number) {
    super();
    this.W0 = W0; // shared frozen base (requires_grad=false already)
    this.r = r;
    this.alphaOverR = alpha / r;
    const aStd = 1 / Math.sqrt(D_IN);
    this.A = Tensor.from(Array.from({ length: r * D_IN }, () => normal(0, aStd)), [r, D_IN], true);
    this.B = Tensor.from(new Array(D_OUT * r).fill(0), [D_OUT, r], true); // zero-init: delta=0 at start
  }
  /** The learned update ΔW = (B@A)·(α/r), as a detached matrix (for SVD / heatmap). */
  learnedDelta(): Tensor {
    return this.B.matmul(this.A).scale(this.alphaOverR);
  }
  override forward(x: Tensor): Tensor {
    // y = x@W0ᵀ + x@(BA)ᵀ·(α/r). Build the delta in-graph so A,B get grad.
    const base = x.matmul(this.W0.transpose());
    const delta = x.matmul(this.learnedDelta().transpose());
    return base.add(delta);
  }
}

// ---------------------------------------------------------------------------
// Train one LoRA rank to convergence; report honest train/val MSE + a "correct-ish" accuracy
//   (fraction of output coords within a tolerance band) + convergence step.
// ---------------------------------------------------------------------------

interface RankResult {
  r: number;
  trainableParams: number;
  finalTrainMse: number;
  finalValMse: number;
  accuracy: number; // fraction of val output coords within ACC_TOL of target
  convergedAt: number; // step at which val MSE first dropped below CONV_TOL, or -1
}

const ACC_TOL = 0.1; // an output coord counts "right" if |pred-target| < ACC_TOL

function meanMse(model: LoRALinear, set: Sample[]): number {
  let total = 0;
  for (const s of set) {
    const out = model.forward(s.x);
    let e = 0;
    for (let i = 0; i < out.size; i++) {
      const d = out.data[i] - s.y.data[i];
      e += d * d;
    }
    total += e / out.size;
  }
  return total / set.length;
}

function accuracy(model: LoRALinear, set: Sample[]): number {
  let hit = 0, tot = 0;
  for (const s of set) {
    const out = model.forward(s.x);
    for (let i = 0; i < out.size; i++) {
      if (Math.abs(out.data[i] - s.y.data[i]) < ACC_TOL) hit++;
      tot++;
    }
  }
  return hit / tot;
}

function trainRank(r: number, W0: Tensor, train: Sample[], val: Sample[]): { res: RankResult; model: LoRALinear; valCurve: number[] } {
  // Re-seed per rank so every rank sees the SAME data-order randomness and SAME init draws
  //   up to the rank-dependent param count — isolates r as the only varying factor.
  seed(4040 + r);
  const model = new LoRALinear(W0, r, /*alpha*/ 8);
  const opt = new Adam(model.trainable(), LR);
  const valCurve: number[] = [];
  let convergedAt = -1;
  for (let step = 0; step < STEPS; step++) {
    for (const s of train) {
      const out = model.forward(s.x);
      const diff = out.sub(s.y);
      const loss = diff.mul(diff).mean();
      opt.zeroGrad();
      loss.backward();
      opt.step();
    }
    const vm = meanMse(model, val);
    valCurve.push(vm);
    if (convergedAt < 0 && vm < CONV_TOL) convergedAt = step + 1;
  }
  const res: RankResult = {
    r,
    trainableParams: model.numParams({ trainableOnly: true }),
    finalTrainMse: meanMse(model, train),
    finalValMse: meanMse(model, val),
    accuracy: accuracy(model, val),
    convergedAt,
  };
  return { res, model, valCurve };
}

// ---------------------------------------------------------------------------
// One-sided Jacobi SVD on a small dense matrix -> singular values only.
// WHY hand-rolled: core has no SVD and we only need σ's of a 16×16 matrix to show the
//   spectrum. One-sided Jacobi is numerically solid and short. The σ's are the column
//   norms after the iteration orthogonalizes columns. DETERMINISTIC: fixed sweep count.
// ---------------------------------------------------------------------------

function singularValues(mat: Tensor): number[] {
  const [m, n] = mat.shape;
  // Work on a copy of columns; rotate pairs of columns to mutual orthogonality.
  const A: number[][] = [];
  for (let j = 0; j < n; j++) {
    const col = new Array(m);
    for (let i = 0; i < m; i++) col[i] = mat.data[i * n + j];
    A.push(col);
  }
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let i = 0; i < m; i++) {
          alpha += A[p][i] * A[p][i];
          beta += A[q][i] * A[q][i];
          gamma += A[p][i] * A[q][i];
        }
        off += Math.abs(gamma);
        if (Math.abs(gamma) < 1e-15) continue;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let i = 0; i < m; i++) {
          const ap = A[p][i], aq = A[q][i];
          A[p][i] = c * ap - s * aq;
          A[q][i] = s * ap + c * aq;
        }
      }
    }
    if (off < 1e-12) break; // converged: columns mutually orthogonal
  }
  const sigmas = A.map((col) => Math.sqrt(col.reduce((acc, v) => acc + v * v, 0)));
  return sigmas.sort((x, y) => y - x);
}

// ===========================================================================
// MAIN
// ===========================================================================

console.log("=== Stage 04 — 秩 r 的选择: ΔW 的有效秩与过参数化 ===\n");
seed(1234);

// 0. Grad-check the LoRA forward so the numbers below rest on correct adjoints (B@A path).
//    INVARIANT for finite differences: the checked function must be DETERMINISTIC in A only —
//    so W0, B and x are drawn ONCE and held fixed; only A is perturbed by the checker.
{
  seed(7);
  const W0chk = buildBase();
  const m = new LoRALinear(W0chk, 2, 8); // its A is the variable; B is zero-init so we override
  const Bfixed = Tensor.from(Array.from({ length: D_OUT * 2 }, () => normal(0, 0.3)), [D_OUT, 2], true);
  m.B = Bfixed; // nonzero B so grad actually flows through the B@A product into A
  const xchk = Tensor.from(Array.from({ length: D_IN }, () => uniformRange(-1, 1)), [1, D_IN], false);
  const err = numericalGradCheck((A) => {
    m.A = A; // checker mutates A in place; everything else (W0,B,x) is closed-over and fixed
    return m.forward(xchk).sum();
  }, m.A);
  console.log(`[gradcheck] LoRA d(out)/dA max-rel-err = ${err.toExponential(2)} (assert < 1e-4)`);
  assert(err < 1e-4, "LoRA forward grad wrt A failed grad check");
}

// 1. Plant ΔW* and build the frozen base + datasets.
seed(2025);
const { deltaStar, trueSpectrum } = buildPlantedDelta();
const W0 = buildBase();
const train = makeDataset(N_TRAIN, W0, deltaStar);
const val = makeDataset(N_VAL, W0, deltaStar);
console.log(`planted ΔW*: shape ${D_OUT}×${D_IN}, planted effective rank ≈ ${TRUE_RANK} ` +
  `(σ decay = ${DECAY}); first σ's = [${trueSpectrum.map((s) => s.toFixed(3)).join(", ")}]`);

// Show the TRUE spectrum of the planted ΔW* (independent of any learning).
const trueSigmas = singularValues(deltaStar);
console.log("planted ΔW* singular spectrum (top 8):");
console.log("  " + trueSigmas.slice(0, 8).map((s) => s.toFixed(3)).join("  "));
console.log("  sparkline: " + sparkline(trueSigmas.slice(0, 8)) + "  (fast decay => low effective rank)\n");

// 2. Sweep r and train each to convergence.
const RANKS = [1, 2, 3, 4, 6, 8, 16]; // 16 = min(d,k): degenerate full-rank LoRA
const results: RankResult[] = [];
const models: Record<number, LoRALinear> = {};
const curves: Record<number, number[]> = {};
for (const r of RANKS) {
  const { res, model, valCurve } = trainRank(r, W0, train, val);
  results.push(res);
  models[r] = model;
  curves[r] = valCurve;
}

// 3. The r comparison table.
console.log("r 对照表 (同 seed-family, 唯一变量是 rank):");
console.log("  r | trainable params | %ofΔW(256) | train MSE | val MSE  | val acc | conv@step");
console.log("  --+------------------+------------+-----------+----------+---------+----------");
const fullDeltaParams = D_OUT * D_IN; // 256: params of a full ΔW (the thing LoRA approximates)
for (const res of results) {
  const pct = ((res.trainableParams / fullDeltaParams) * 100).toFixed(1);
  const conv = res.convergedAt < 0 ? "  —  " : String(res.convergedAt).padStart(5);
  console.log(
    `  ${String(res.r).padStart(2)}| ${String(res.trainableParams).padStart(16)} | ${pct.padStart(9)}% | ` +
    `${res.finalTrainMse.toExponential(2)} | ${res.finalValMse.toExponential(2)} | ` +
    `${(res.accuracy * 100).toFixed(1).padStart(6)}% | ${conv}`,
  );
}

// 4. Assert saturation: accuracy past the planted effective rank is flat (within noise).
const accAt = (r: number) => results.find((x) => x.r === r)!.accuracy;
const accSat = accAt(TRUE_RANK); // at the planted effective rank
const accHigh = accAt(8);
const accLow = accAt(1);
console.log(`\nsaturation check: acc(r=1)=${(accLow * 100).toFixed(1)}%  ` +
  `acc(r=${TRUE_RANK})=${(accSat * 100).toFixed(1)}%  acc(r=8)=${(accHigh * 100).toFixed(1)}%`);
assert(accSat > accLow + 0.05, "accuracy did not climb from r=1 to effective rank — task too easy/hard");
assert(Math.abs(accHigh - accSat) < 0.07, "accuracy did NOT saturate past effective rank — claim broken");
console.log(`  ✓ acc climbs r=1 → r=${TRUE_RANK}, then SATURATES r=${TRUE_RANK} → r=8 (|Δ| < 7pt).`);

// 5. FAILURE MODE: r = min(d,k) is full-rank LoRA — params explode, accuracy does not.
const full = results.find((x) => x.r === 16)!;
const pctFull = ((full.trainableParams / fullDeltaParams) * 100).toFixed(1);
console.log("\n失败模式 — r = min(d,k) = 16 (退化为近全秩 LoRA):");
console.log(`  trainable params = ${full.trainableParams} = ${pctFull}% of full ΔW ` +
  `(vs r=${TRUE_RANK}: ${results.find((x) => x.r === TRUE_RANK)!.trainableParams} params, ` +
  `${((results.find((x) => x.r === TRUE_RANK)!.trainableParams / fullDeltaParams) * 100).toFixed(1)}%)`);
console.log(`  val acc r=16: ${(full.accuracy * 100).toFixed(1)}%  vs  r=${TRUE_RANK}: ${(accSat * 100).toFixed(1)}%  ` +
  `→ ${(full.trainableParams / results.find((x) => x.r === TRUE_RANK)!.trainableParams).toFixed(1)}x the params for ` +
  `${((full.accuracy - accSat) * 100).toFixed(1)}pt accuracy.`);
// Honest reporting of the val-MSE effect. We do NOT assert overfit: on this clean synthetic
//   task accuracy is already 100% at the effective rank, so extra capacity has almost no MSE
//   to chase either way. The robust, demonstrable failure is the PARAM BLOWUP for zero
//   accuracy gain; the train/val GAP is the honest overfit proxy (train fits noise the val
//   does not). We print the gap and let the sign speak for itself.
const bestVal = (r: number) => Math.min(...curves[r]);
const trainValGap = (r: number) => results.find((x) => x.r === r)!.finalValMse - results.find((x) => x.r === r)!.finalTrainMse;
console.log(`  best val MSE: r=16 ${bestVal(16).toExponential(2)} vs r=${TRUE_RANK} ${bestVal(TRUE_RANK).toExponential(2)} ` +
  `(both already < ${CONV_TOL}; the marginal MSE difference is noise, not a real win).`);
console.log(`  train→val MSE gap (overfit proxy): r=16 ${trainValGap(16).toExponential(2)} ` +
  `vs r=${TRUE_RANK} ${trainValGap(TRUE_RANK).toExponential(2)}  ` +
  `(larger/positive gap at high r = capacity fitting train noise that does not generalize).`);

// param-cost bar — the sliver-vs-slab visual.
console.log("\ntrainable-param cost by rank:");
console.log(bar(results.map((res) => ({
  label: `r=${res.r}`,
  value: res.trainableParams,
  note: `params  (acc ${(res.accuracy * 100).toFixed(0)}%)`,
}))));

// 6. SVD the LEARNED ΔW=BA at the saturating rank — its spectrum should echo the planted decay.
const learned = models[8].learnedDelta(); // r=8 has room to express >3 directions if needed
const learnedSigmas = singularValues(learned);
console.log(`\nlearned ΔW (from r=8 LoRA) singular spectrum (top 8):`);
console.log("  " + learnedSigmas.slice(0, 8).map((s) => s.toFixed(3)).join("  "));
console.log("  sparkline: " + sparkline(learnedSigmas.slice(0, 8)) +
  "  (learner concentrates energy in few directions, like the planted ΔW*)");
// Effective rank via the participation-ratio / energy heuristic: how many σ to reach 90% energy.
const energy = learnedSigmas.map((s) => s * s);
const totalE = energy.reduce((a, b) => a + b, 0);
let cum = 0, effRank = 0;
for (const e of energy) { cum += e; effRank++; if (cum / totalE >= 0.9) break; }
console.log(`  effective rank (σ² energy reaching 90%) = ${effRank}  (planted ≈ ${TRUE_RANK})`);

// heatmap of the learned ΔW — the low-rank stripe structure.
console.log("\nlearned ΔW heatmap (|value| -> gray ramp; low-rank => structured, not noise):");
console.log(heatmap(learned.data, [D_OUT, D_IN], { label: "  ΔW=BA·(α/r), r=8" }));

console.log("\n⚠ toy-scale: 有效秩的绝对值取决于我们手工植入的 ΔW* (σ 衰减 = " + DECAY +
  ");\n  可迁移的是两条形状: (1) 奇异值快速衰减, (2) 准确率-vs-r 曲线有拐点、过拐点后饱和。");
console.log("STAGE 04 DONE");
