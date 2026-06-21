// stage05-training-loop.ts — Chapter 5: the training loop, and how to READ it when loss
// won't drop. A loop is trivial to write and easy to write WRONG; the value here is the
// diagnostics that turn "loss is bad" into "loss is bad BECAUSE x". We build one reusable
// train() and then run three experiments that each isolate one failure signature:
//   (A) memorization vs generalization — train loss -> ~0 while val loss bottoms out and
//       climbs. The canonical overfitting fingerprint; you read it off TWO curves, never one.
//   (B) gradient clipping on an UNBOUNDED-gradient problem (MSE regression). With a large lr,
//       clip OFF => grad-norm explodes, one step overshoots, loss -> Inf/NaN, unrecoverable;
//       clip ON => the same spike gets capped and the run survives. The early-warning
//       instrument is the PRE-clip grad-norm, which blows up a step or two BEFORE the loss.
//   (C) FAILURE MODE: omit zeroGrad(). Autograd ACCUMULATES grads on purpose; skipping the
//       reset sums grads across steps, the effective step grows, loss lurches. We reproduce
//       it, then fix it, on the same seed for a clean A/B.
//
// WHY experiment B uses MSE regression, not the spiral classifier: a softmax+cross-entropy
// gradient is BOUNDED (softmax-onehot lives in [-1,1], divided by batch), so even an absurd
// lr only makes the classifier oscillate — it never reaches NaN, and clipping shows no clean
// win. That is itself an honest lesson: clipping earns its keep where gradients are UNBOUNDED
// (regression targets, autoregressive LM logits) — which is exactly why transformers ship it
// on by default. We pick the problem that actually exhibits the failure instead of faking it.
//
// HONESTY: toy CPU data. Absolute losses/accuracies are optimistic and do NOT generalize —
// what transfers is the SHAPES (train/val divergence, the norm spike before the loss spike,
// the accumulation lurch) and RELATIVE comparisons (clip-on survives where clip-off dies).
// Every printed number is computed live from this run; nothing is hand-tuned to look good.
//
// Run:  npx tsx src/stage05-training-loop.ts

import { mulberry32, randn, type Rng } from "./core/rng.js";
import { Tensor, noGrad } from "./core/autograd.js";
import { Linear, Sequential, Module } from "./core/nn.js";
import { SGD, clipGradNorm } from "./core/optim.js";
import { makeSpiral, trainValSplit, type Dataset2D } from "./core/data.js";
import { crossEntropy, mseLoss, accuracy, lossCurveAscii } from "./core/metrics.js";

function section(title: string): void {
  console.log("\n" + "=".repeat(64) + "\n" + title + "\n" + "=".repeat(64));
}

// Smallest net that can bend a spiral boundary (in -> hidden -> hidden -> out, ReLU between).
// outF is the only thing that changes between classifier (outF=classes) and regressor (outF=1).
function buildMlp(inF: number, hidden: number, outF: number, rng: Rng): Module {
  // Relu is a 1-line param-less Module so activations can sit inside Sequential. No params =>
  // it never touches the optimizer.
  class Relu extends Module {
    override forward(x: Tensor): Tensor {
      return x.relu();
    }
  }
  return new Sequential([
    new Linear(inF, hidden, rng),
    new Relu(),
    new Linear(hidden, hidden, rng),
    new Relu(),
    new Linear(hidden, outF, rng),
  ]);
}

// Pack a Dataset2D into a single (n, 2) feature Tensor + Int32Array targets, once.
// WHY precompute: the loop re-runs forward every step; rebuilding the input each step would
// conflate data-prep with compute and add timing noise.
function toTensors(ds: Dataset2D): { x: Tensor; y: Int32Array } {
  const n = ds.y.length;
  const flat = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    flat[i * 2] = ds.X[i][0];
    flat[i * 2 + 1] = ds.X[i][1];
  }
  return { x: new Tensor(flat, [n, 2]), y: Int32Array.from(ds.y) };
}

interface TrainConfig {
  steps: number;
  lr: number;
  logEvery: number;
  clipMaxNorm?: number; // undefined => no clipping
  zeroGradEachStep?: boolean; // default true; set false to DEMO the accumulation bug
  label: string;
}

interface TrainHistory {
  trainLoss: number[];
  valLoss: number[]; // recorded only on logEvery steps (the val pass is the extra cost)
  gradNorm: number[]; // PRE-clip global grad norm per step — the instability instrument
  diverged: boolean; // hit NaN/Inf and bailed
}

// The reusable loop. Full-batch GD on toy data (no minibatching) so the curves are clean and
// the only moving parts are the diagnostics. Loss is injected as a closure so the SAME loop
// drives both the spiral classifier (cross-entropy) and the regressor (MSE) — the loop must
// not know which task it is running, exactly as a real trainer is task-agnostic.
//
// INVARIANT: a step is  zeroGrad -> forward+loss -> backward -> [clip] -> opt.step. Grads are
// zeroed at the TOP of each step unless we are deliberately demoing the bug. Mutates `model`
// in place (its params ARE the trained weights afterward), so callers can eval it post-train.
function train(
  model: Module,
  trainLossFn: () => Tensor, // builds the scalar train loss from the (current) model
  valLossFn: () => number, // scalar val loss; caller wraps in noGrad
  cfg: TrainConfig,
): TrainHistory {
  const params = model.parameters();
  const opt = new SGD(params, { lr: cfg.lr, momentum: 0.9 });
  const hist: TrainHistory = { trainLoss: [], valLoss: [], gradNorm: [], diverged: false };
  const zeroEach = cfg.zeroGradEachStep ?? true;

  for (let step = 0; step < cfg.steps; step++) {
    if (zeroEach) opt.zeroGrad(); // <-- the line whose ABSENCE is experiment (C)

    const loss = trainLossFn();
    const lossVal = loss.data[0];

    // Divergence guard: once loss is NaN/Inf there is no recovering — grads are NaN and every
    // future step propagates it. Bail and record, so the experiment reports it honestly
    // instead of printing a screen of NaN.
    if (!Number.isFinite(lossVal)) {
      hist.diverged = true;
      hist.trainLoss.push(lossVal);
      hist.gradNorm.push(NaN);
      console.log(`[${cfg.label}] step ${String(step).padStart(4)} | train ${lossVal} | DIVERGED (non-finite loss), stopping`);
      break;
    }

    loss.backward();

    // clipGradNorm returns the PRE-clip norm whether or not it clipped — that is the number
    // worth logging: a spike here is the leading indicator of the loss blowing up next step.
    const preClipNorm = cfg.clipMaxNorm !== undefined ? clipGradNorm(params, cfg.clipMaxNorm) : globalGradNorm(params);

    opt.step();

    hist.trainLoss.push(lossVal);
    hist.gradNorm.push(preClipNorm);

    if (step % cfg.logEvery === 0 || step === cfg.steps - 1) {
      // Val pass under noGrad: no graph built, so it cannot leak grads into params and is
      // cheaper. Exactly how you'd gate eval in a real loop.
      const valLoss = noGrad(valLossFn);
      hist.valLoss.push(valLoss);
      console.log(
        `[${cfg.label}] step ${String(step).padStart(4)} | ` +
          `train ${lossVal.toFixed(4)} | val ${valLoss.toFixed(4)} | gradNorm ${preClipNorm.toExponential(2)}`,
      );
    }
  }
  return hist;
}

// Global L2 norm of all grads, read-only (clipGradNorm mutates; we want the same number when
// clipping is OFF without scaling anything). Tiny + local; not worth a core export.
function globalGradNorm(params: Tensor[]): number {
  let sq = 0;
  for (const p of params) for (let i = 0; i < p.size; i++) sq += p.grad[i] * p.grad[i];
  return Math.sqrt(sq);
}

// ---------------------------------------------------------------------------
// Experiment A: memorization vs generalization.
// We starve the model of data (tiny train split) so a comfortably-sized net memorizes it.
// The signature to READ: train loss marches toward ~0 while val loss bottoms out early then
// turns UP. One curve can't show this; you need both.
// ---------------------------------------------------------------------------
section("A) Overfitting: train loss -> ~0 while val loss bottoms then RISES");

const overfitRng = mulberry32(0xa11ce);
const spiral = makeSpiral(40, 3, overfitRng, 0.25); // 120 pts, 3 arms, noisy enough to punish memorizers
const split = trainValSplit(spiral, 0.6, overfitRng); // 60% held out => only ~48 train pts
const ofTrain = toTensors(split.train);
const ofVal = toTensors(split.val);

const overfitModel = buildMlp(2, 64, 3, mulberry32(123)); // 64 hidden = plenty to memorize ~48 pts
const ofHist = train(
  overfitModel,
  () => crossEntropy(overfitModel.forward(ofTrain.x), ofTrain.y),
  () => crossEntropy(overfitModel.forward(ofVal.x), ofVal.y).data[0],
  { steps: 400, lr: 0.2, logEvery: 50, label: "overfit" },
);

// Read the divergence quantitatively: locate the val-loss minimum and show it rose afterward.
const valMinIdx = ofHist.valLoss.indexOf(Math.min(...ofHist.valLoss));
const valMin = ofHist.valLoss[valMinIdx];
const valFinal = ofHist.valLoss[ofHist.valLoss.length - 1];
const trainFinal = ofHist.trainLoss[ofHist.trainLoss.length - 1];
console.log(`\ntrain loss start ${ofHist.trainLoss[0].toFixed(4)} -> final ${trainFinal.toFixed(4)} (memorizing)`);
console.log(`val loss min ${valMin.toFixed(4)} (around log-point ${valMinIdx}) -> final ${valFinal.toFixed(4)}`);
console.log(`generalization gap (val_final - train_final): ${(valFinal - trainFinal).toFixed(4)}`);
console.log(
  `overfitting confirmed (val rose past its min AND train kept dropping): ${valFinal > valMin + 0.02 && trainFinal < ofHist.trainLoss[0]}`,
);
console.log("\ntrain-loss curve (monotone descent = the model fitting the train set):");
console.log(lossCurveAscii(ofHist.trainLoss, 50, 7));
console.log("\nval-loss curve (the U-turn is overfitting — descent then climb):");
console.log(lossCurveAscii(ofHist.valLoss, ofHist.valLoss.length, 7));

// ---------------------------------------------------------------------------
// Experiment B: gradient clipping on an UNBOUNDED-gradient problem (MSE regression).
// Synthetic target y = 5 * sum(x) with x ~ N(0,1): MSE grads scale with the error, so a too-
// large lr makes one step overshoot, which makes the next error (and grad) larger, a positive
// feedback loop that runs away to Inf/NaN in a handful of steps. Same data, same seed, same
// lr for both runs — the ONLY difference is clipping.
// ---------------------------------------------------------------------------
section("B) Gradient clipping (MSE regression): clip OFF runs away to NaN, clip ON survives");

// Build the regression problem once. 4 inputs -> 1 output. Targets scaled up (×5) so squared
// errors — and thus gradients — are large enough to blow up under an aggressive lr.
const regRng = mulberry32(0xbeef);
const REG_N = 64;
const regX = new Float64Array(REG_N * 4);
const regT = new Float64Array(REG_N * 1);
for (let i = 0; i < REG_N; i++) {
  let s = 0;
  for (let k = 0; k < 4; k++) {
    const v = randn(regRng);
    regX[i * 4 + k] = v;
    s += v;
  }
  regT[i] = s * 5; // unbounded target => unbounded MSE gradient (the whole point)
}
const regXt = new Tensor(regX, [REG_N, 4]);
const regTt = new Tensor(regT, [REG_N, 1]);

const UNSTABLE_LR = 0.1; // empirically: clip-OFF diverges by ~step 5, clip-ON survives, on this seed

console.log("--- clip OFF (expect pre-clip grad-norm to explode, then loss -> NaN -> dead) ---");
const noClipModel = buildMlp(4, 32, 1, mulberry32(777));
const noClipHist = train(
  noClipModel,
  () => mseLoss(noClipModel.forward(regXt), regTt),
  () => mseLoss(noClipModel.forward(regXt), regTt).data[0], // no separate val set needed for this point
  { steps: 60, lr: UNSTABLE_LR, logEvery: 1, label: "no-clip" },
);

console.log("\n--- clip ON, maxNorm=1.0 (same seed, same lr) ---");
const clipModel = buildMlp(4, 32, 1, mulberry32(777));
const clipHist = train(
  clipModel,
  () => mseLoss(clipModel.forward(regXt), regTt),
  () => mseLoss(clipModel.forward(regXt), regTt).data[0],
  { steps: 60, lr: UNSTABLE_LR, logEvery: 10, clipMaxNorm: 1.0, label: "clip-1.0" },
);

const maxNoClipNorm = Math.max(...noClipHist.gradNorm.filter(Number.isFinite));
const maxClipNorm = Math.max(...clipHist.gradNorm.filter(Number.isFinite)); // PRE-clip, so the spike still shows
console.log(`\nclip OFF diverged: ${noClipHist.diverged}  (peak pre-clip gradNorm ${maxNoClipNorm.toExponential(2)} before NaN)`);
console.log(`clip ON  diverged: ${clipHist.diverged}  (peak pre-clip gradNorm ${maxClipNorm.toExponential(2)}, each step capped to 1.0)`);
const clipFinal = clipHist.trainLoss[clipHist.trainLoss.length - 1];
console.log(`clip ON final train loss: ${clipFinal.toFixed(4)} (finite => the clip absorbed the spike)`);
console.log(`lesson holds (clip OFF dies, clip ON lives): ${noClipHist.diverged && !clipHist.diverged}`);
console.log("\nclip-OFF pre-clip grad-norm curve (note: y-axis is the SPIKE before it goes NaN):");
console.log(lossCurveAscii(noClipHist.gradNorm.filter(Number.isFinite), Math.max(2, noClipHist.gradNorm.filter(Number.isFinite).length), 7));
console.log("\nclip-ON train-loss curve (bounded steps => actually converges):");
console.log(lossCurveAscii(clipHist.trainLoss, Math.min(50, clipHist.trainLoss.length), 7));

// ---------------------------------------------------------------------------
// Experiment C: FAILURE MODE — forgetting zeroGrad().
// Autograd accumulates grads by += into param.grad. The loop MUST clear them each step; if
// not, step t's update uses the SUM of grads from steps 0..t, so the effective step grows and
// the loss lurches / blows up. Buggy vs fixed on the SAME seed and data => the difference is
// purely the missing line. Back on the (bounded-grad) spiral classifier, so the contrast is a
// clean "smooth descent vs lurch" rather than a NaN.
// ---------------------------------------------------------------------------
section("C) FAILURE MODE: omitting zeroGrad() makes grads accumulate -> loss lurches");

const bugRng = mulberry32(0xc0de);
const bugSpiral = makeSpiral(60, 3, bugRng, 0.2);
const bugSplit = trainValSplit(bugSpiral, 0.2, bugRng);
const bugTrain = toTensors(bugSplit.train);
const bugVal = toTensors(bugSplit.val);

// MODEST lr so the FIXED run is calm — any wildness in the buggy run is then attributable to
// accumulation, not to an aggressive lr.
const SAFE_LR = 0.3;

console.log("--- BUGGY: zeroGrad() omitted (grads pile up across steps) ---");
const buggyModel = buildMlp(2, 24, 3, mulberry32(999));
const buggyHist = train(
  buggyModel,
  () => crossEntropy(buggyModel.forward(bugTrain.x), bugTrain.y),
  () => crossEntropy(buggyModel.forward(bugVal.x), bugVal.y).data[0],
  { steps: 40, lr: SAFE_LR, logEvery: 4, zeroGradEachStep: false, label: "no-zerograd" }, // the bug
);

console.log("\n--- FIXED: zeroGrad() each step (same seed, same lr) ---");
const fixedModel = buildMlp(2, 24, 3, mulberry32(999));
const fixedHist = train(
  fixedModel,
  () => crossEntropy(fixedModel.forward(bugTrain.x), bugTrain.y),
  () => crossEntropy(fixedModel.forward(bugVal.x), bugVal.y).data[0],
  { steps: 40, lr: SAFE_LR, logEvery: 4, zeroGradEachStep: true, label: "zerograd-ok" }, // the fix
);

// Quantify the lurch: the BUGGY grad-norm grows because each step's grad is the running SUM;
// the fixed run's stays bounded. Compare last/first grad-norm as a crisp, run-computed signal.
const buggyNorms = buggyHist.gradNorm.filter(Number.isFinite);
const fixedNorms = fixedHist.gradNorm.filter(Number.isFinite);
const buggyGrowth = buggyNorms[buggyNorms.length - 1] / buggyNorms[0];
const fixedGrowth = fixedNorms[fixedNorms.length - 1] / fixedNorms[0];
console.log(`\nbuggy grad-norm growth (last/first): ${buggyGrowth.toExponential(2)}  (accumulation inflates it)`);
console.log(`fixed grad-norm growth (last/first): ${fixedGrowth.toExponential(2)}  (bounded, healthy)`);
console.log(`buggy final train loss: ${buggyHist.trainLoss[buggyHist.trainLoss.length - 1].toFixed(4)} (diverged=${buggyHist.diverged})`);
console.log(`fixed final train loss: ${fixedHist.trainLoss[fixedHist.trainLoss.length - 1].toFixed(4)} (diverged=${fixedHist.diverged})`);
const fixedIsBetter =
  (buggyHist.diverged && !fixedHist.diverged) ||
  buggyGrowth > fixedGrowth * 5 ||
  buggyHist.trainLoss[buggyHist.trainLoss.length - 1] > fixedHist.trainLoss[fixedHist.trainLoss.length - 1] + 0.05;
console.log(`zeroGrad matters (buggy measurably worse than fixed): ${fixedIsBetter}`);
console.log("\nbuggy train-loss curve (lurchy / non-monotone from accumulating grads):");
console.log(lossCurveAscii(buggyHist.trainLoss, Math.min(50, buggyHist.trainLoss.length), 7));
console.log("\nfixed train-loss curve (smooth descent — this is what a healthy loop looks like):");
console.log(lossCurveAscii(fixedHist.trainLoss, Math.min(50, fixedHist.trainLoss.length), 7));

// Sanity: the FIXED model (mutated in place by train()) actually learned — accuracy well above
// chance (33%). No retrain needed; train() left the trained weights in fixedModel.
const fixedAcc = noGrad(() => accuracy(fixedModel.forward(bugVal.x), bugVal.y));
console.log(`\nfixed model val accuracy: ${(fixedAcc * 100).toFixed(1)}%  (chance = 33.3%)`);

console.log("\n" + "=".repeat(64));
console.log("Takeaways: (A) read TWO curves to see overfitting; (B) on UNBOUNDED-grad problems");
console.log("watch the PRE-clip grad norm as the divergence early-warning, and clip; (C) call");
console.log("zeroGrad() every step — no exceptions.");
console.log("=".repeat(64));
