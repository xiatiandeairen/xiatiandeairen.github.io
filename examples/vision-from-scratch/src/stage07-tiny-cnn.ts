// stage07-tiny-cnn.ts — Train a real (tiny) CNN end-to-end on synthetic geometric shapes.
//
// WHAT this stage proves: every machine the book built so far (conv / BN / maxpool / linear
//   + Adam + cross-entropy) actually composes into a classifier that LEARNS. Earlier stages
//   verified gradients in isolation (core/_smoke); this is the integration test — does the
//   loss go down, does test accuracy beat chance, and WHICH classes get confused.
//
// WHY the failure-mode half matters more than the happy path: a beginner who only ever sees
//   converging runs can't recognize a broken one. The folklore failure is "lr too high =>
//   loss NaN". We tried to reproduce that and DID NOT — and the reason is itself the lesson:
//
//   1) core/metrics.crossEntropy is numerically stabilized (softmax subtracts the row max,
//      and log() has a +1e-12 guard), so even absurd logits never overflow to Inf. The loss
//      therefore PLATEAUS at a high value instead of going NaN — it looks "merely bad", not
//      "broken". This is a TRAP: a stable loss can hide a diverging model.
//   2) The real, measurable collapse is in the WEIGHTS. With a too-hot lr the first-layer
//      kernels blow up by orders of magnitude (we measure peak |w| going from ~1 at init to
//      ~7e3), and their heatmap range explodes. The kernels are the true tell, not the loss
//      — exactly the diagnostic skill this stage teaches.
//
//   So the demo deliberately uses a BN-FREE net + SGD at a large lr to surface the blow-up,
//   and reports honestly that the loss does NOT reach NaN here (it would on a naive,
//   non-stabilized loss; our core is too well-built for that).
//
//   (Aside, deliberately NOT used as the lesson: with BN the conv weights can drift even
//   MORE freely — BN makes a preceding conv's scale irrelevant to the forward, so SGD pushes
//   those weights to astronomically large values while the loss stays bounded. So "watch the
//   weight magnitude" is the robust signal; "BN bounds the weights" is FALSE and we don't
//   claim it.)
//
// HONESTY (inherited from core/data.ts): absolute accuracies are OPTIMISTIC — the 4 shape
//   classes are clean, low-noise and far more separable than natural images. What transfers
//   is the TREND (loss descends, test > chance, confusion concentrates on visually-similar
//   classes) and the failure SIGNATURE (plateaued loss + exploding kernels), not the accuracy
//   number. All printed numbers are computed at runtime; wall-clock is real (performance.now).
//
// DETERMINISM: a single seeded mulberry32 threads dataset generation, weight init and
//   batch shuffling, so re-running reproduces every number bit-for-bit.

import { Tensor, noGrad } from "./core/autograd.js";
import { Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear, Sequential, Module } from "./core/nn.js";
import { Adam, SGD, type Optimizer } from "./core/optim.js";
import { mulberry32 } from "./core/rng.js";
import { makeShapeDataset, trainTestSplit, iterBatches, SHAPE_NAMES, type ImageDataset } from "./core/data.js";
import {
  crossEntropy,
  accuracy,
  confusionMatrix,
  confusionAscii,
  lossCurveAscii,
  heatmapAscii,
  paramCount,
} from "./core/metrics.js";

// ---------------------------------------------------------------------------
// Hyperparameters. Small on purpose: the whole point is "converges on a CPU in seconds".
// ---------------------------------------------------------------------------
const SEED = 1234;
const IMG = 16; // H = W = 16; spatial flow below depends on this exact size
const PER_CLASS = 120; // 4 classes * 120 = 480 images before the split
const TEST_RATIO = 0.25;
const BATCH = 32;
const EPOCHS = 12;
const BASE_LR = 3e-3; // healthy lr for Adam on this toy task
// "Too hot" lr for the failure demo. Applied to a BN-FREE net with SGD+momentum, this lr is
// large enough to make the first-layer kernels explode from O(1) to ~7e3 while the loss
// thrashes high and never settles. (We measured: Adam's bounded per-step update absorbs even
// lr=3.0 here without exploding, so a convincing collapse needs SGD on the BN-free path.)
const DIVERGE_LR = 5.0;
const DIVERGE_EPOCHS = 20; // a few more epochs so the blow-up is unmistakable in the curve

// Channel widths for the two conv blocks. Kept tiny; bigger doesn't help a 4-class toy.
const C1 = 8;
const C2 = 16;

/**
 * Build the tiny CNN. Spatial bookkeeping (the part beginners get wrong and then hit a shape
 * error at the Linear): with 16x16 input and k=3,pad=1 convs (size-preserving) each followed
 * by a 2x2 maxpool that halves H and W:
 *     16x16 --conv,pool--> 8x8 --conv,pool--> 4x4
 * so the flattened feature vector feeding the classifier is C2 * 4 * 4. Getting this wrong is
 * the canonical "expected (N, X) got (N, Y)" matmul failure — we compute it, not eyeball it.
 *
 * INVARIANT: the Linear's inF MUST equal C2 * finalH * finalW or forward throws at matmul.
 *
 * useBatchNorm toggles the BN layers. The happy path uses BN (true). The failure demo uses
 * the BN-free variant (false) because BN re-normalizes activations every forward and thereby
 * SUPPRESSES the lr-driven blow-up we want to exhibit — see the file header note (3).
 */
function buildTinyCnn(
  rng: () => number,
  classes: number,
  useBatchNorm: boolean,
): { net: Sequential; flatFeatures: number } {
  const afterPools = IMG >> 2; // two 2x2 pools => divide by 4 => 4
  const flatFeatures = C2 * afterPools * afterPools;
  const block = (inC: number, outC: number): Module[] => {
    const mods: Module[] = [new Conv2d(inC, outC, 3, rng, { padding: 1 })];
    if (useBatchNorm) mods.push(new BatchNorm2d(outC)); // BN stabilizes the activation dist
    mods.push(new ReLU(), new MaxPool2d(2)); // ReLU then 2x2 pool halves H,W
    return mods;
  };
  const net = new Sequential([
    ...block(1, C1), // 16x16 -> 8x8
    ...block(C1, C2), // 8x8 -> 4x4
    new Flatten(), // -> (N, C2*4*4)
    new Linear(flatFeatures, classes, rng),
  ]);
  return { net, flatFeatures };
}

/**
 * Evaluate mean loss + accuracy + confusion over a whole dataset, in eval mode.
 * WHY setTraining(false) + noGrad: BatchNorm must use FROZEN running stats at eval, otherwise
 *   a single image's prediction would depend on whatever else shares its batch (the canonical
 *   BN bug). noGrad also skips building the autograd graph — pure inference, no backward.
 * Returns null-safe aggregate numbers; confusion is summed over batches.
 */
function evalDataset(net: Module, ds: ImageDataset): { loss: number; acc: number; confusion: number[][] } {
  net.setTraining(false);
  const M = Array.from({ length: ds.classes }, () => new Array<number>(ds.classes).fill(0));
  let lossSum = 0;
  let correct = 0;
  let total = 0;
  const result = noGrad(() => {
    // Fixed order (no shuffle) so eval is deterministic and the confusion matrix is stable.
    for (const batch of iterBatches(ds, BATCH, () => 0, false)) {
      const x = new Tensor(batch.data, [batch.N, batch.C, batch.H, batch.W]);
      const logits = net.forward(x);
      const loss = crossEntropy(logits, batch.labels);
      lossSum += loss.data[0] * batch.N; // weight by batch size; last batch may be partial
      correct += accuracy(logits, batch.labels) * batch.N;
      total += batch.N;
      const bm = confusionMatrix(logits, batch.labels, ds.classes);
      for (let i = 0; i < ds.classes; i++) for (let j = 0; j < ds.classes; j++) M[i][j] += bm[i][j];
    }
    return { loss: lossSum / total, acc: correct / total, confusion: M };
  });
  net.setTraining(true);
  return result;
}

/**
 * Snapshot the first-layer kernels into a (C1, 3, 3) plain array we can heatmap. We read
 * Conv2d.weight (layout OIHW, here inC=1 so O*1*3*3). Copying decouples the snapshot from
 * later in-place optimizer updates — important for the before/after-divergence comparison.
 */
function snapshotFirstKernels(net: Sequential): Float64Array {
  const conv1 = net.parameters()[0]; // first registered param = block-1 conv weight
  return Float64Array.from(conv1.data);
}

/** Fraction of kernel elements that are non-finite (NaN/Inf). A blunt but unambiguous
 *  "have the weights been destroyed" probe used in the failure demo. */
function nonFiniteFraction(arr: Float64Array): number {
  let bad = 0;
  for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) bad++;
  return bad / arr.length;
}

/** Largest absolute, finite value across a param tensor's data. The blunt "how inflated are
 *  the weights" probe — a healthy first-layer kernel stays O(1); a diverging one balloons. */
function peakFiniteAbs(t: Tensor): number {
  let mx = 0;
  for (let i = 0; i < t.size; i++) {
    const a = Math.abs(t.data[i]);
    if (Number.isFinite(a) && a > mx) mx = a;
  }
  return mx;
}

/**
 * Run a full training loop and return per-step train-loss history + the trained net.
 * The optimizer is injected (makeOpt) so the happy path (Adam) and the failure demo (SGD)
 * share ONE loop. We also track the peak first-layer-kernel magnitude over training: that
 * trace is what exposes divergence the (numerically stable) loss can hide.
 * We record NaN if it ever appears — we do NOT sanitize it away — but on this stabilized
 * core it won't (see header); the kernel-magnitude trace is the real divergence signal.
 */
function trainTinyCnn(
  net: Sequential,
  train: ImageDataset,
  rng: () => number,
  makeOpt: (params: Tensor[]) => Optimizer,
  epochs: number,
): { history: number[]; sawNaN: boolean; peakKernelAbs: number } {
  const opt = makeOpt(net.parameters());
  const conv1 = net.parameters()[0]; // first-layer conv weight, watched for blow-up
  const history: number[] = [];
  let sawNaN = false;
  let peakKernelAbs = peakFiniteAbs(conv1);
  net.setTraining(true);
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const batch of iterBatches(train, BATCH, rng, true)) {
      const x = new Tensor(batch.data, [batch.N, batch.C, batch.H, batch.W]);
      const logits = net.forward(x);
      const loss = crossEntropy(logits, batch.labels);
      history.push(loss.data[0]);
      if (!Number.isFinite(loss.data[0])) sawNaN = true;
      net.zeroGrad();
      loss.backward();
      opt.step();
      const m = peakFiniteAbs(conv1);
      if (m > peakKernelAbs) peakKernelAbs = m;
    }
  }
  return { history, sawNaN, peakKernelAbs };
}

/** Render the C1 first-layer 3x3 kernels side by side as small heatmaps with a shared note. */
function renderKernels(kernels: Float64Array, count: number): string {
  const lines: string[] = [];
  for (let k = 0; k < count; k++) {
    const single = kernels.subarray(k * 9, k * 9 + 9); // 3x3 = 9 elems per kernel (inC=1)
    lines.push(`  kernel #${k}:`);
    for (const row of heatmapAscii(single, 3, 3).split("\n")) lines.push("    " + row);
  }
  return lines.join("\n");
}

function main(): void {
  console.log("=== stage07: 训练一个真正的 tiny CNN（合成几何图，4 类） ===\n");

  // --- Data (deterministic) -------------------------------------------------
  const rng = mulberry32(SEED);
  const full = makeShapeDataset(PER_CLASS, IMG, IMG, rng, 0.08);
  const { train, test } = trainTestSplit(full, TEST_RATIO);
  console.log(
    `数据集: ${full.samples.length} 张 ${IMG}x${IMG} 图, ${full.classes} 类 (${SHAPE_NAMES.join("/")}); ` +
      `train=${train.samples.length}, test=${test.samples.length}`,
  );

  // --- Build + size sanity --------------------------------------------------
  const { net, flatFeatures } = buildTinyCnn(rng, full.classes, true);
  console.log(
    `网络: 2x[Conv-BN-ReLU-MaxPool] + Flatten(${flatFeatures}) + Linear -> ${full.classes}; ` +
      `参数量 = ${paramCount(net)}\n`,
  );

  // --- Baseline (untrained) accuracy: should sit near chance (1/4 = 25%) ----
  const before = evalDataset(net, test);
  console.log(
    `训练前 test: loss=${before.loss.toFixed(4)}, acc=${(before.acc * 100).toFixed(1)}% ` +
      `(随机基线 ~${(100 / full.classes).toFixed(0)}%)`,
  );

  // ==========================================================================
  // HAPPY PATH: healthy lr, expect convergence.
  // ==========================================================================
  const t0 = performance.now();
  const { history } = trainTinyCnn(net, train, rng, (p) => new Adam(p, { lr: BASE_LR }), EPOCHS);
  const trainMs = performance.now() - t0;

  const after = evalDataset(net, test);
  const trainEval = evalDataset(net, train);
  console.log(
    `\n训练 ${EPOCHS} epochs (lr=${BASE_LR}), ${history.length} steps, ` +
      `真实墙钟 ${trainMs.toFixed(0)}ms (~${(trainMs / history.length).toFixed(2)}ms/step)`,
  );
  console.log("\n训练 loss 曲线 (按 step 下采样):");
  console.log(lossCurveAscii(history, 56, 8));

  console.log(
    `\n最终: train acc=${(trainEval.acc * 100).toFixed(1)}% (loss=${trainEval.loss.toFixed(4)}), ` +
      `test acc=${(after.acc * 100).toFixed(1)}% (loss=${after.loss.toFixed(4)})`,
  );
  // The train/test gap is the honest signal of overfit vs generalization; print it explicitly.
  console.log(
    `train-test acc gap = ${((trainEval.acc - after.acc) * 100).toFixed(1)} 个百分点 ` +
      `(gap 越大越过拟合; 增广见 stage08)`,
  );

  console.log("\n混淆矩阵 M[真][预测] (对角=正确, 非对角=哪两类相混):");
  console.log(confusionAscii(after.confusion, SHAPE_NAMES));
  // Programmatically surface the single worst confusion so the reader doesn't have to scan.
  let worst = { i: -1, j: -1, n: 0 };
  for (let i = 0; i < full.classes; i++)
    for (let j = 0; j < full.classes; j++)
      if (i !== j && after.confusion[i][j] > worst.n) worst = { i, j, n: after.confusion[i][j] };
  if (worst.n > 0)
    console.log(
      `最易混: 真实 "${SHAPE_NAMES[worst.i]}" 被预测成 "${SHAPE_NAMES[worst.j]}" 共 ${worst.n} 次 ` +
        `(几何上这两类局部边缘最相似)`,
    );
  else console.log("无错分 (toy 任务过于可分; 见 honesty 说明, 绝对值偏乐观)");

  // --- Visualize learned first-layer kernels --------------------------------
  const goodKernels = snapshotFirstKernels(net);
  console.log("\n第一层卷积核 (训练后, 前 4 个 3x3; 应呈现结构化的边缘/角点模式):");
  console.log(renderKernels(goodKernels, 4));

  // ==========================================================================
  // FAILURE MODE: a too-hot learning rate. The folklore is "loss -> NaN"; the honest
  //   reality on this stabilized core is "loss thrashes high while the kernels explode".
  //   We surface it on a BN-FREE net trained with SGD at a large lr, and report honestly
  //   that the loss never NaNs — the WEIGHT magnitude is the real divergence signal.
  // ==========================================================================
  console.log("\n" + "=".repeat(64));
  console.log(`失败模式演示: lr 调到 ${DIVERGE_LR} (健康值 ${BASE_LR} 的 ~${Math.round(DIVERGE_LR / BASE_LR)}x)`);
  console.log("=".repeat(64));

  const l2 = (a: Float64Array): number => {
    let s = 0;
    for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) s += a[i] * a[i];
    return Math.sqrt(s);
  };
  const peakAbs = (a: Float64Array): number => {
    let mx = 0;
    for (let i = 0; i < a.length; i++) {
      const v = Math.abs(a[i]);
      if (Number.isFinite(v) && v > mx) mx = v;
    }
    return mx;
  };

  // --- BN-FREE net at the hot lr: this is the one that visibly diverges ------
  const rngBad = mulberry32(SEED);
  const fullBad = makeShapeDataset(PER_CLASS, IMG, IMG, rngBad, 0.08);
  const { train: trainBad } = trainTestSplit(fullBad, TEST_RATIO);
  const { net: badNet } = buildTinyCnn(rngBad, fullBad.classes, false); // NO BatchNorm
  const initKernels = snapshotFirstKernels(badNet); // finite, O(1) Kaiming init
  const { history: badHistory, sawNaN, peakKernelAbs } = trainTinyCnn(
    badNet,
    trainBad,
    rngBad,
    (p) => new SGD(p, { lr: DIVERGE_LR, momentum: 0.9 }),
    DIVERGE_EPOCHS,
  );
  const badKernels = snapshotFirstKernels(badNet);

  let lossPeak = 0;
  for (const v of badHistory) if (Number.isFinite(v) && v > lossPeak) lossPeak = v;
  console.log(
    `\n无 BN 网络 + SGD(lr=${DIVERGE_LR}): loss 峰值 ${lossPeak.toFixed(2)}, 出现 NaN? ${sawNaN ? "是" : "否"}`,
  );
  console.log(
    `  -> loss 没有 NaN 反而是个陷阱: crossEntropy 数值稳定 (减 max + log 加 1e-12), ` +
      `所以 loss 封顶在 ~${lossPeak.toFixed(1)} 而非发散; 单看 loss 你会以为只是『没学好』。`,
  );
  console.log("\n发散过程的 loss 曲线 (高位震荡/平台, 不收敛):");
  console.log(lossCurveAscii(badHistory, 56, 8));

  // --- The teaching payload: the kernels are the real tell ------------------
  console.log("\n真正的崩溃信号在权重, 不在 loss —— 看第一层卷积核的量级:");
  console.log(`  初始化第一层核:  L2=${l2(initKernels).toFixed(3)}, peak|w|=${peakAbs(initKernels).toExponential(2)}`);
  console.log(`  发散后第一层核:  L2=${l2(badKernels).toExponential(2)}, peak|w|=${peakAbs(badKernels).toExponential(2)}`);
  console.log(`  训练全程峰值 peak|w| = ${peakKernelAbs.toExponential(2)} (从 O(1) 暴涨, 数量级即崩溃强度)`);
  console.log(`  核 non-finite 比例: 初始 ${(nonFiniteFraction(initKernels) * 100).toFixed(0)}% -> 发散后 ${(nonFiniteFraction(badKernels) * 100).toFixed(0)}%`);

  console.log("\n初始化时的第一层核 (随机但 O(1), 结构平滑):");
  console.log(renderKernels(initKernels, 2));
  console.log("\n发散后的第一层核 (heatmap 的 range 暴露了异常量级 —— 这是肉眼可读的崩溃):");
  console.log(renderKernels(badKernels, 2));

  console.log(
    "\n结论: 数值稳定的 loss 会掩盖发散 (封顶而非 NaN); 第一层核的 peak|w| / heatmap range " +
      "暴涨才是肉眼可读的崩溃信号。排查 lr/init 问题时先看核是否仍 O(1) 有结构, 比反复盯 loss 更快定位。",
  );
}

main();
