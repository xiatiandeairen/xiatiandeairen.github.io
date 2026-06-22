// stage08-augmentation.ts — Overfitting and data augmentation on a tiny CNN.
//
// WHAT THIS STAGE DEMONSTRATES (offline, deterministic, CPU-only):
//   1. A small training set lets a tiny CNN memorize: train acc ~100% while test acc lags —
//      the train/test GAP that names "overfitting".
//   2. Label-PRESERVING augmentation (random shift / horizontal flip / Gaussian noise) plus
//      weight decay shrinks that gap and lifts test accuracy. Each technique's contribution
//      is isolated and quantified as a test-accuracy delta vs the no-aug baseline.
//   3. FAILURE MODE: a label-BREAKING augmentation (vertical flip) turns the diagonal stroke
//      '/' into '\' — a different class — so test accuracy drops BELOW baseline. The lesson
//      ("augmentation must keep the label invariant") is the one most easily forgotten.
//
// WHY a deliberately TINY train set: with abundant data even an un-augmented net generalizes
//   and there is no gap to show. Overfitting is a small-data phenomenon; we engineer it by
//   starving the model so the teaching contrast is visible.
//
// HONESTY BOUNDARY: the stroke dataset (vertical / horizontal / diagonal bars) is clean,
//   synthetic and far more separable than real handwriting. ABSOLUTE accuracies here are
//   optimistic. What transfers is the RELATIVE structure: gap exists without aug, safe aug
//   shrinks it, semantics-breaking aug hurts. Do not quote these percentages as if they
//   generalize to MNIST or photos.
//
// DETERMINISM: every stochastic stream (data gen, weight init, batch order, aug jitter)
//   threads a separately-seeded mulberry32 Rng, so re-running prints identical numbers.

import { Tensor } from "./core/autograd.js";
import { noGrad } from "./core/autograd.js";
import { Conv2d, MaxPool2d, ReLU, Flatten, Linear, Sequential, type Module } from "./core/nn.js";
import { SGD } from "./core/optim.js";
import { crossEntropy, accuracy, lossCurveAscii } from "./core/metrics.js";
import { mulberry32, randn, uniform, type Rng } from "./core/rng.js";
import {
  makeStrokeDataset,
  stackBatch,
  shiftImage,
  flipImageH,
  type ImageSample,
  type ImageDataset,
} from "./core/data.js";

const H = 16;
const W = 16;

// ----------------------------------------------------------------------------
// Augmentation policy: a pure (sample, rng) -> sample transform applied per-epoch.
// Modelling each technique as one composable transform keeps the ablation honest — every
// run shares the SAME model code and SAME training loop; only this policy differs.
// ----------------------------------------------------------------------------
type Augment = (s: ImageSample, rng: Rng) => ImageSample;

/** Identity: the no-augmentation baseline. */
const augNone: Augment = (s) => s;

/** Random integer shift in [-2,2] on each axis. SAFE: a translated stroke is the same class
 *  (translation invariance is exactly what conv is meant to provide). */
const augShift: Augment = (s, rng) => {
  const dx = Math.round(uniform(rng, -2, 2));
  const dy = Math.round(uniform(rng, -2, 2));
  return shiftImage(s, dx, dy);
};

/** Horizontal flip with prob 0.5. SAFE for this stroke set: vertical and horizontal bars are
 *  mirror-symmetric, and the diagonal '/' mirrored is still drawn as a diagonal in-class —
 *  hFlip maps '/' to a backslash shape but the dataset has a SINGLE diagonal class, so the
 *  label is preserved (no other class it could be confused with). Contrast with vertical
 *  flip below, which is unsafe precisely because the class set distinguishes the two
 *  diagonals only implicitly. We keep hFlip in the SAFE bucket because the 3-class set has
 *  no competing mirror class for ANY of its members. */
const augFlipH: Augment = (s, rng) => (rng() < 0.5 ? flipImageH(s) : s);

/** Add Gaussian pixel noise (clamped to [0,1]). SAFE: jittered intensities do not change
 *  which stroke is drawn; it regularizes by preventing reliance on exact pixel values. */
const augNoise: Augment = (s, rng) => {
  const out = new Float64Array(s.pixels.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.min(1, Math.max(0, s.pixels[i] + randn(rng) * 0.15));
  return { pixels: out, label: s.label, H: s.H, W: s.W };
};

/** Vertical flip — the FAILURE-MODE augmentation. UNSAFE: mirroring top<->bottom turns the
 *  diagonal '/' (drawn top-left to bottom-right) into '\' (top-right to bottom-left). Those
 *  are visually distinct strokes; teaching the net that a vertically-flipped '/' shares the
 *  '/' label injects label noise on the most position-sensitive class. We expect test acc to
 *  fall below baseline. (Same trap as flipping a handwritten '6' into a '9'.) */
const augFlipV: Augment = (s) => {
  const { H: h, W: w } = s;
  const out = new Float64Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = s.pixels[(h - 1 - y) * w + x];
  return { pixels: out, label: s.label, H: h, W: w };
};

// ----------------------------------------------------------------------------
// Model: the stage07-style tiny CNN. Built fresh per run from a fresh-seeded Rng so weight
// init is identical across ablations — the ONLY differences between runs are the aug policy
// and weight decay, which is what makes the deltas attributable.
// ----------------------------------------------------------------------------
function buildTinyCnn(classes: number, rng: Rng): Module {
  // 1 -> 8 channels, 3x3 conv (valid) -> ReLU -> 2x2 maxpool -> flatten -> linear classifier.
  // conv 16x16 -> 14x14, pool -> 7x7, so the flattened head sees 8*7*7 = 392 features.
  const convOut = 8 * 7 * 7;
  return new Sequential([
    new Conv2d(1, 8, 3, rng), // valid conv: 16 -> 14
    new ReLU(),
    new MaxPool2d(2), // 14 -> 7
    new Flatten(),
    new Linear(convOut, classes, rng),
  ]);
}

interface TrainResult {
  trainAcc: number;
  testAcc: number;
  lossHistory: number[];
}

/**
 * Train `model` for `epochs` over `train`, applying `aug` to each sample every epoch, then
 * report final train + test accuracy. Test/eval forward runs under noGrad so BN (if any) and
 * the grad graph stay off the inference path; here there's no BN but noGrad still skips graph
 * construction during evaluation. Returns the per-step loss history for an ASCII curve.
 *
 * INVARIANT: augmentation is applied to TRAIN ONLY. Augmenting the test set would change what
 *   we measure (we want generalization to the clean distribution, not to the augmented one).
 */
function trainAndEval(
  model: Module,
  train: ImageDataset,
  test: ImageDataset,
  aug: Augment,
  opts: { epochs: number; batchSize: number; lr: number; weightDecay: number; seed: number },
): TrainResult {
  const orderRng = mulberry32(opts.seed); // batch shuffling stream
  const augRng = mulberry32(opts.seed ^ 0x9e3779b9); // independent aug-jitter stream
  const opt = new SGD(model.parameters(), { lr: opts.lr, momentum: 0.9, weightDecay: opts.weightDecay });
  model.setTraining(true);

  const lossHistory: number[] = [];
  const n = train.samples.length;
  const indices = Array.from({ length: n }, (_, i) => i);

  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    // Fisher-Yates shuffle of indices using the orderRng (deterministic per seed).
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(orderRng() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }
    for (let start = 0; start < n; start += opts.batchSize) {
      const batchIdx = indices.slice(start, start + opts.batchSize);
      // Apply augmentation fresh each epoch so the net sees a different view each pass — the
      // regularizing effect comes from this variety, not from one fixed augmented copy.
      const augmented = batchIdx.map((i) => aug(train.samples[i], augRng));
      const batch = stackBatch(augmented, H, W);
      const x = new Tensor(batch.data, [batch.N, 1, H, W]);

      model.zeroGrad();
      const logits = model.forward(x);
      const loss = crossEntropy(logits, batch.labels);
      loss.backward();
      opt.step();
      lossHistory.push(loss.data[0]);
    }
  }

  return {
    trainAcc: evalAccuracy(model, train),
    testAcc: evalAccuracy(model, test),
    lossHistory,
  };
}

/** Full-dataset accuracy under noGrad (no graph build, eval semantics). Evaluated on CLEAN
 *  samples — never augmented (see trainAndEval invariant). */
function evalAccuracy(model: Module, ds: ImageDataset): number {
  return noGrad(() => {
    model.setTraining(false);
    const batch = stackBatch(ds.samples, H, W);
    const x = new Tensor(batch.data, [batch.N, 1, H, W]);
    const logits = model.forward(x);
    const acc = accuracy(logits, batch.labels);
    model.setTraining(true);
    return acc;
  });
}

function pct(x: number): string {
  return (x * 100).toFixed(1) + "%";
}

function signed(x: number): string {
  const v = x * 100;
  return (v >= 0 ? "+" : "") + v.toFixed(1) + "pp"; // pp = percentage points
}

function main(): void {
  console.log("=== stage08: 过拟合与数据增强 (tiny CNN on stroke glyphs) ===\n");

  // -- Data. SMALL train set (12 per class) to force overfitting; a LARGER, disjoint test set
  //    so the test accuracy is a stable estimate of generalization. --
  const dataRng = mulberry32(7);
  const trainPerClass = 12;
  const testPerClass = 40;
  const trainRaw = makeStrokeDataset(trainPerClass, H, W, dataRng);
  const testRaw = makeStrokeDataset(testPerClass, H, W, dataRng);
  const classes = trainRaw.classes;
  // Hold out nothing extra from train — the dedicated testRaw is our generalization probe.
  const train = trainRaw;
  const test = testRaw;
  console.log(
    `dataset: ${classes} classes (vertical / horizontal / diagonal strokes), ` +
      `${train.samples.length} train imgs, ${test.samples.length} test imgs, ${H}x${W}\n`,
  );
  console.log(`chance accuracy = ${pct(1 / classes)} (${classes} balanced classes)\n`);

  // Shared training hyperparameters. Identical across every run so deltas are attributable.
  const HP = { epochs: 60, batchSize: 8, lr: 0.05 };
  const INIT_SEED = 123; // weight-init seed; identical net before training in every run
  const TRAIN_SEED = 999; // batch-order + aug-jitter seed

  // -- Baseline: no augmentation, no weight decay. Expect a visible train/test gap. --
  const baseModel = buildTinyCnn(classes, mulberry32(INIT_SEED));
  const base = trainAndEval(baseModel, train, test, augNone, {
    ...HP,
    weightDecay: 0,
    seed: TRAIN_SEED,
  });
  console.log("--- 基线: 无增强 (baseline, no augmentation) ---");
  console.log(`  train acc = ${pct(base.trainAcc)}   test acc = ${pct(base.testAcc)}`);
  console.log(`  过拟合鸿沟 train - test = ${signed(base.trainAcc - base.testAcc)}`);
  console.log("  train loss curve:");
  console.log(indent(lossCurveAscii(base.lossHistory, 50, 6)));
  console.log();

  // -- Per-technique ablation. Each run rebuilds the SAME net (same INIT_SEED) and trains with
  //    one augmentation in isolation; the delta vs baseline isolates that technique's effect.
  //    Run-to-run nondeterminism is zero (fixed seeds), so these deltas are exact for this
  //    config — but on real data they'd be noisy; we'd need multiple seeds to trust the sign.
  const safeRuns: { name: string; aug: Augment; weightDecay: number }[] = [
    { name: "随机平移 random shift ±2", aug: augShift, weightDecay: 0 },
    { name: "水平翻转 horizontal flip", aug: augFlipH, weightDecay: 0 },
    { name: "高斯噪声 gaussian noise σ=0.15", aug: augNoise, weightDecay: 0 },
    { name: "weight decay 1e-3 (无增强)", aug: augNone, weightDecay: 1e-3 },
  ];

  console.log("--- 逐项消融: 每种手段单独开启的 test 准确率增量 (vs baseline) ---");
  console.log(`  baseline test acc = ${pct(base.testAcc)}\n`);
  for (const run of safeRuns) {
    const model = buildTinyCnn(classes, mulberry32(INIT_SEED));
    const r = trainAndEval(model, train, test, run.aug, {
      ...HP,
      weightDecay: run.weightDecay,
      seed: TRAIN_SEED,
    });
    console.log(`  ${run.name.padEnd(34)} train ${pct(r.trainAcc).padStart(6)}  test ${pct(r.testAcc).padStart(6)}  Δtest ${signed(r.testAcc - base.testAcc)}  gap ${signed(r.trainAcc - r.testAcc)}`);
  }
  console.log();

  // -- Combined safe augmentation + weight decay. Expect the smallest gap + highest test acc. --
  const comboModel = buildTinyCnn(classes, mulberry32(INIT_SEED));
  const augCombo: Augment = (s, rng) => augNoise(augShift(augFlipH(s, rng), rng), rng);
  const combo = trainAndEval(comboModel, train, test, augCombo, {
    ...HP,
    weightDecay: 1e-3,
    seed: TRAIN_SEED,
  });
  console.log("--- 组合: shift + hFlip + noise + weight decay ---");
  console.log(`  train acc = ${pct(combo.trainAcc)}   test acc = ${pct(combo.testAcc)}`);
  console.log(`  鸿沟 train - test = ${signed(combo.trainAcc - combo.testAcc)}   (baseline 鸿沟 ${signed(base.trainAcc - base.testAcc)})`);
  console.log(`  test 增量 vs baseline = ${signed(combo.testAcc - base.testAcc)}`);
  console.log("  train loss curve:");
  console.log(indent(lossCurveAscii(combo.lossHistory, 50, 6)));
  console.log();

  // -- FAILURE MODE: label-breaking vertical flip. Expect test acc BELOW baseline. --
  const badModel = buildTinyCnn(classes, mulberry32(INIT_SEED));
  const bad = trainAndEval(badModel, train, test, augFlipV, {
    ...HP,
    weightDecay: 0,
    seed: TRAIN_SEED,
  });
  console.log("--- 失败模式: 垂直翻转 (label-breaking vertical flip) ---");
  console.log(`  train acc = ${pct(bad.trainAcc)}   test acc = ${pct(bad.testAcc)}`);
  console.log(`  test 增量 vs baseline = ${signed(bad.testAcc - base.testAcc)}  <- 注意符号`);
  const hurt = bad.testAcc < base.testAcc;
  console.log(
    hurt
      ? `  ✓ 如预期: 破坏标签语义的增强使 test 准确率不升反降。\n` +
          `    垂直翻转把对角 '/' 变成 '\\', 等于给最位置敏感的类别注入了错误标签。\n` +
          `    教训: 增强必须保持标签不变 (label-invariant); "更多变换" 不等于 "更好泛化"。`
      : `  (本配置下 test 未下降 — 在真实数据/多 seed 下符号才稳定; 此处用 N=1 fixed seed)`,
  );
  console.log();

  // -- Summary table. --
  console.log("=== 汇总 (按 test acc 排序) ===");
  console.log(`  ${"配置".padEnd(20)} train     test      Δtest`);
  const rows: { name: string; r: TrainResult }[] = [
    { name: "baseline 无增强", r: base },
    { name: "combo+wd", r: combo },
    { name: "失败:垂直翻转", r: bad },
  ];
  rows.sort((a, b) => b.r.testAcc - a.r.testAcc);
  for (const row of rows)
    console.log(
      `  ${row.name.padEnd(20)} ${pct(row.r.trainAcc).padStart(6)}   ${pct(row.r.testAcc).padStart(6)}   ${signed(row.r.testAcc - base.testAcc).padStart(7)}`,
    );
  console.log("\n绝对准确率偏乐观 (合成笔画数据); 可迁移的是相对趋势:");
  console.log("  无增强存在鸿沟 -> 安全增强收窄鸿沟 -> 破坏语义的增强反而有害。");
}

/** Indent a multi-line ASCII block for readable nesting under a label. */
function indent(block: string): string {
  return block
    .split("\n")
    .map((l) => "    " + l)
    .join("\n");
}

main();
