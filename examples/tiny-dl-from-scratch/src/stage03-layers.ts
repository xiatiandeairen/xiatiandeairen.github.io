// stage03-layers.ts — Chapter 3: layers & activations as Modules, and the SHAPES their
//   gradients flow through. Everything here leans on core/* so we only assemble, never
//   re-derive: Linear/ReLU/softmax/crossEntropy already carry tested adjoints; this stage's
//   job is to PROVE they compose correctly (per-layer gradCheck), to SHOW why init choice is
//   not cosmetic (zero-init kills a net before training even starts), and to MAKE the classic
//   softmax-overflow bug actually print NaN so the fix is not taken on faith.
//
// WHY per-layer gradCheck instead of one end-to-end check: a single net-wide check can pass
//   while an individual layer's adjoint is subtly wrong if errors cancel. Isolating each layer
//   (feed it a tiny input, backprop a scalar, compare analytic vs central-difference grads)
//   pins the failure to one op. This is the book's correctness keystone applied layer-by-layer.
//
// HONESTY: numbers below are real (computed/measured this run, deterministic via seeded RNG).
//   Toy data => absolute losses/accuracies are optimistic; what transfers is the structure:
//   untrained CE ≈ ln(C), zero-init breaks symmetry-breaking, no-max softmax overflows.

import { mulberry32, type Rng } from "./core/rng.js";
import { Tensor } from "./core/autograd.js";
import { Linear, Module } from "./core/nn.js";
import { crossEntropy, gradCheck, accuracy, paramCount } from "./core/metrics.js";
import { makeMoons } from "./core/data.js";

const SEED = 1337;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/** Wrap a 2-D number[][] into a Tensor (row-major). Used to lift toy datasets into the
 *  autograd world. Not in core because only stages build input tensors from raw arrays. */
function tensorFrom2D(rows: number[][]): Tensor {
  const r = rows.length;
  const c = rows[0].length;
  const flat = new Float64Array(r * c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) flat[i * c + j] = rows[i][j];
  return new Tensor(flat, [r, c]);
}

/** Mean and variance of a flat Float64Array. Plain stats, used to inspect activations.
 *  WHY population variance (divide by n): we want the actual spread of THIS activation
 *  tensor, not an unbiased estimator of a larger population. */
function meanVar(a: Float64Array): { mean: number; var: number } {
  let m = 0;
  for (let i = 0; i < a.length; i++) m += a[i];
  m /= a.length;
  let v = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - m;
    v += d * d;
  }
  v /= a.length;
  return { mean: m, var: v };
}

function fmt(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return x > 0 ? "+Inf" : Number.isNaN(x) ? "NaN" : "-Inf";
  return x.toFixed(digits);
}

// ----------------------------------------------------------------------------
// A tiny MLP for the moons problem, built from core layers via Sequential.
// We override forward to interleave ReLU because Sequential only chains Modules and
// ReLU is a Tensor METHOD, not a Module here — so the activation lives in forward().
// ----------------------------------------------------------------------------
class MoonsMlp extends Module {
  private fc1: Linear;
  private fc2: Linear;

  /** init lets section 3 reuse the exact same architecture with a different W init so the
   *  zero-vs-kaiming comparison changes ONLY the thing under test. */
  constructor(rng: Rng, init: "kaiming" | "zero", hidden = 8) {
    super();
    // Default-kaiming Linear, then optionally stomp W to all-zeros to demonstrate the
    // symmetry-breaking failure. We keep biases (zero by core convention) either way.
    this.fc1 = this.child(new Linear(2, hidden, rng, { init: "kaiming" }));
    this.fc2 = this.child(new Linear(hidden, 2, rng, { init: "kaiming" }));
    if (init === "zero") {
      this.fc1.W.data.fill(0);
      this.fc2.W.data.fill(0);
    }
  }

  /** Expose the hidden pre-activation path for inspection in the init comparison. */
  hidden(x: Tensor): Tensor {
    return this.fc1.forward(x).relu();
  }

  /** Zero ONLY the readout weights (keep hidden Kaiming). Result: initial logits == bias
   *  == 0 => softmax exactly uniform => CE == ln(C). NOT a symmetry problem here: the
   *  HIDDEN layer is still randomly initialized, so once training starts the readout
   *  immediately receives distinct gradients per class and breaks out of zero. */
  zeroOutput(): void {
    this.fc2.W.data.fill(0);
  }

  override forward(x: Tensor): Tensor {
    const h = this.fc1.forward(x).relu();
    return this.fc2.forward(h); // logits; CE applies softmax internally
  }
}

// ----------------------------------------------------------------------------
// 1) Per-layer gradCheck: Linear, ReLU, softmax, and Linear→ReLU→Linear→CE end to end.
//    Each block builds a fresh scalar loss closure so gradCheck's central differences see
//    a clean forward (no stale graph). Expect max relative error < 1e-6 everywhere.
// ----------------------------------------------------------------------------
function section1_perLayerGradCheck(rng: Rng): void {
  console.log("=".repeat(64));
  console.log("1) Per-layer gradCheck (analytic vs numerical, expect < 1e-6)");
  console.log("=".repeat(64));

  const batch = 4;
  // Fixed small input kept away from the ReLU kink (no values near 0) so finite differences
  // and the subgradient agree — see metrics.gradCheck caveat.
  const xData = [
    [0.7, -0.4],
    [-0.9, 0.3],
    [0.2, 0.8],
    [-0.5, -0.6],
  ];

  const rows: Array<{ layer: string; maxRelError: number; checked: number; pass: boolean }> = [];

  // --- Linear alone: loss = sum(Linear(x)) , check W and b grads ---
  {
    const x = tensorFrom2D(xData);
    const lin = new Linear(2, 3, rng, { init: "kaiming" });
    const params = lin.parameters();
    const loss = lin.forward(x).sum();
    params.forEach((p) => p.zeroGrad());
    x.zeroGrad();
    loss.backward();
    const f = () => lin.forward(x).sum().data[0];
    const r = gradCheck(f, params);
    rows.push({ layer: "Linear (W,b)", ...r, pass: r.maxRelError < 1e-6 });
  }

  // --- ReLU: loss = sum(relu(Linear(x))) , check the Linear params THROUGH the ReLU ---
  // We grad-check params (not x) because gradCheck perturbs Tensors in `params`; routing the
  // signal through relu proves the ReLU adjoint (gate = 1{out>0}) composes correctly.
  {
    const x = tensorFrom2D(xData);
    const lin = new Linear(2, 5, rng, { init: "kaiming" });
    const params = lin.parameters();
    const loss = lin.forward(x).relu().sum();
    params.forEach((p) => p.zeroGrad());
    loss.backward();
    const f = () => lin.forward(x).relu().sum().data[0];
    const r = gradCheck(f, params);
    rows.push({ layer: "ReLU (via Linear)", ...r, pass: r.maxRelError < 1e-6 });
  }

  // --- softmax: loss = sum(softmax(Linear(x))) ; note this sum is ~constant (rows sum to 1)
  // so we instead weight by a fixed vector to get a non-trivial gradient signal. ---
  {
    const x = tensorFrom2D(xData);
    const lin = new Linear(2, 3, rng, { init: "kaiming" });
    const params = lin.parameters();
    // weight rows by [1,2,3] so d(loss)/d(probs) is not uniform -> exercises the Jacobian.
    const w = new Tensor([1, 2, 3], [1, 3]);
    const build = () => lin.forward(x).softmax().mul(w.broadcastRow(batch)).sum();
    const loss = build();
    params.forEach((p) => p.zeroGrad());
    loss.backward();
    const f = () => build().data[0];
    const r = gradCheck(f, params);
    rows.push({ layer: "softmax (weighted)", ...r, pass: r.maxRelError < 1e-6 });
  }

  // --- full MLP + crossEntropy end to end ---
  {
    const x = tensorFrom2D(xData);
    const targets = Int32Array.from([0, 1, 0, 1]);
    const net = new MoonsMlp(rng, "kaiming");
    const params = net.parameters();
    const build = () => crossEntropy(net.forward(x), targets);
    const loss = build();
    net.zeroGrad();
    loss.backward();
    const f = () => build().data[0];
    const r = gradCheck(f, params);
    rows.push({ layer: "MLP+CE (end to end)", ...r, pass: r.maxRelError < 1e-6 });
  }

  const pad = (s: string, n: number) => s.padEnd(n);
  console.log(pad("layer", 22) + pad("maxRelError", 14) + pad("checked", 9) + "pass");
  for (const r of rows)
    console.log(pad(r.layer, 22) + pad(r.maxRelError.toExponential(3), 14) + pad(String(r.checked), 9) + r.pass);
  const allPass = rows.every((r) => r.pass);
  console.log(`ALL layers pass (< 1e-6): ${allPass}`);
}

// ----------------------------------------------------------------------------
// 2) Untrained baseline: a fresh net's CE on a balanced 2-class problem must sit near
//    ln(2) ≈ 0.6931 — BUT ONLY IF the OUTPUT logits start near uniform. This is the subtle
//    part worth teaching: ln(C) is the loss of a UNINFORMED predictor (uniform softmax),
//    and you only get uniform softmax when the final-layer logits are ~0.
//
//    A Kaiming-initialized OUTPUT layer does NOT give that: its weights have variance 2/fanIn,
//    so initial logits spread out, softmax is over-confident in random directions, and CE
//    sits ABOVE ln(2) (over-confident wrong guesses are penalized harder than uniform). We
//    print BOTH to make the mechanism visible, then use the small-output-init net (a real,
//    standard convention — e.g. GPT near-zero-inits its final projection) to recover the
//    textbook ln(2) baseline. Hidden layers stay Kaiming; only the readout starts small.
// ----------------------------------------------------------------------------
function section2_untrainedBaseline(rng: Rng): void {
  console.log("=".repeat(64));
  console.log("2) Untrained baseline: CE should be ~ln(2) when output logits start uniform");
  console.log("=".repeat(64));

  const ds = makeMoons(200, rng);
  const x = tensorFrom2D(ds.X);
  const targets = Int32Array.from(ds.y);
  const ln2 = Math.log(2);

  // (a) full Kaiming, including the readout: logits NOT near 0 => CE drifts above ln(2).
  const kaimingNet = new MoonsMlp(rng, "kaiming");
  const kaimingLogits = kaimingNet.forward(x);
  const kaimingLoss = crossEntropy(kaimingLogits, targets).data[0];

  // (b) hidden Kaiming, OUTPUT layer zeroed: logits == bias == 0 => softmax exactly uniform
  //     => CE == ln(2) up to float rounding. This is the configuration that makes the
  //     "untrained loss ≈ ln(C)" sanity check actually hold.
  const baselineNet = new MoonsMlp(rng, "kaiming");
  baselineNet.zeroOutput();
  const baselineLogits = baselineNet.forward(x);
  const baselineLoss = crossEntropy(baselineLogits, targets).data[0];
  const acc = accuracy(baselineLogits, targets);

  console.log(`paramCount(net): ${paramCount(baselineNet)}`);
  console.log(`(a) Kaiming readout  CE = ${fmt(kaimingLoss)}   |CE - ln2| = ${fmt(Math.abs(kaimingLoss - ln2))}` +
    `  <- above ln(2): random over-confidence is penalized`);
  console.log(`(b) zeroed readout   CE = ${fmt(baselineLoss)}   |CE - ln2| = ${fmt(Math.abs(baselineLoss - ln2))}` +
    `  <- uniform softmax`);
  console.log(`ln(2) = ${fmt(ln2)}`);
  console.log(`untrained accuracy (zeroed readout): ${(acc * 100).toFixed(1)}%  (chance = 50.0%)`);
  // With a zeroed readout, logits are exactly 0 so the match is tight (float rounding only).
  console.log(`baseline ≈ ln(2) (within 1e-9): ${Math.abs(baselineLoss - ln2) < 1e-9}`);
}

// ----------------------------------------------------------------------------
// 3) Init matters: zero-init vs Kaiming. With W=0 every hidden unit computes the SAME
//    function of the input (identical weights => identical pre-activations => identical
//    outputs), so the layer has effectively ONE unit no matter how wide. Gradients are
//    likewise identical across units, so SGD can never differentiate them — symmetry is
//    never broken and the extra width is dead. We PROVE it by printing per-unit hidden
//    activations: zero-init gives variance ~0 ACROSS units; Kaiming gives real spread.
// ----------------------------------------------------------------------------
function section3_initComparison(rng: Rng): void {
  console.log("=".repeat(64));
  console.log("3) Init comparison: zero-init vs Kaiming (symmetry breaking)");
  console.log("=".repeat(64));

  const ds = makeMoons(64, rng);
  const x = tensorFrom2D(ds.X);

  for (const init of ["zero", "kaiming"] as const) {
    // Fresh rng per net so the only difference is the W-stomp, not consumed draws.
    const net = new MoonsMlp(mulberry32(SEED + 7), init);
    const h = net.hidden(x); // (batch, hidden) post-ReLU activations
    const hidden = h.shape[1];
    const batch = h.shape[0];

    const all = meanVar(h.data);

    // Variance ACROSS units within a row: are the hidden units actually different from
    // each other? Average that across rows. Zero-init => ~0 (all units identical).
    let acrossUnitVar = 0;
    for (let r = 0; r < batch; r++) {
      const row = h.data.subarray(r * hidden, (r + 1) * hidden);
      acrossUnitVar += meanVar(row).var;
    }
    acrossUnitVar /= batch;

    // Sanity: row 0's first 4 hidden units, to literally SEE them be equal under zero-init.
    const sample = Array.from(h.data.subarray(0, Math.min(4, hidden)))
      .map((v) => fmt(v, 3))
      .join(", ");

    console.log(`[${init.padEnd(8)}] activation mean=${fmt(all.mean)} var=${fmt(all.var)}` +
      `  mean across-unit var=${fmt(acrossUnitVar, 6)}`);
    console.log(`[${init.padEnd(8)}]   row0 first units: [${sample}]` +
      (init === "zero" ? "  <- all identical: symmetry NOT broken" : "  <- distinct: symmetry broken"));
  }
  console.log(">> Lesson: zero-init makes every hidden unit a clone; width is wasted and");
  console.log("   SGD can't break the tie. Random (Kaiming) init is what gives units identity.");
}

// ----------------------------------------------------------------------------
// 4) FAILURE DEMO: softmax WITHOUT subtracting the row max overflows on large logits.
//    We hand-roll the naive version (exp then normalize, NO max subtraction) and feed it a
//    growing logit until exp() hits Inf and the normalized row becomes NaN (Inf/Inf). Then
//    we run the SAME logits through core's max-subtracting softmax and show it stays finite
//    and correct. This is the single most common numerical bug in a from-scratch stack.
// ----------------------------------------------------------------------------
function naiveSoftmaxRow(logits: number[]): number[] {
  // Deliberately UNSTABLE: no max subtraction. exp(large) -> Inf, Inf/Inf -> NaN.
  const exps = logits.map((v) => Math.exp(v));
  const denom = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / denom);
}

function section4_softmaxOverflow(): void {
  console.log("=".repeat(64));
  console.log("4) FAILURE MODE: naive softmax (no max-subtract) overflows to NaN");
  console.log("=".repeat(64));

  // Walk the max logit up; the others stay at 0. Find the first magnitude where naive breaks.
  const probes = [10, 100, 500, 700, 710, 800, 1000];
  let firstNaN = -1;
  console.log("logit z fed as row [z, 0, 0]:");
  console.log("   z      exp(z)        naive p[0]   coreSoftmax p[0]");
  for (const z of probes) {
    const row = [z, 0, 0];
    const naive = naiveSoftmaxRow(row);
    // core softmax via Tensor (subtracts row max internally).
    const core = new Tensor([z, 0, 0], [1, 3]).softmax().data;
    const naiveBroken = !Number.isFinite(naive[0]);
    if (naiveBroken && firstNaN < 0) firstNaN = z;
    console.log(
      `  ${String(z).padStart(4)}  ${fmt(Math.exp(z), 3).padStart(12)}` +
      `  ${fmt(naive[0], 6).padStart(11)}  ${fmt(core[0], 6).padStart(14)}`,
    );
  }
  console.log(
    firstNaN >= 0
      ? `naive softmax first produced non-finite p at z=${firstNaN} (exp(z) overflowed Double)`
      : "naive softmax did not overflow in probe range (unexpected)",
  );
  console.log(">> Lesson: exp() overflows ~z=710 in f64; subtract row max FIRST so the");
  console.log("   largest exponent is exp(0)=1. Core's softmax/crossEntropy already do this.");
}

// ----------------------------------------------------------------------------
function main(): void {
  const rng = mulberry32(SEED);
  console.log(`stage03-layers — seed=${SEED} (deterministic)\n`);
  section1_perLayerGradCheck(rng);
  section2_untrainedBaseline(rng);
  section3_initComparison(rng);
  section4_softmaxOverflow();
  console.log("\nstage03 done.");
}

main();
