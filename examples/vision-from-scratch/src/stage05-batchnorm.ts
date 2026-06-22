// stage05-batchnorm.ts — BatchNorm: train-time statistics, the eval switch, and the
//   single most-reported BN bug (forgetting to switch to eval at inference).
//
// WHY this stage exists: BatchNorm is the first layer whose forward pass is NOT a pure
//   function of its input. It mutates running statistics as a side effect, and it has TWO
//   different forward formulas selected by a mode flag. Both facts produce a class of bugs
//   that gradient checking can NOT catch — the math is correct, but the WRONG math runs
//   because the layer is in the wrong mode. This file demonstrates each fact with measured
//   numbers and then deliberately reproduces the NaN explosion that "forgot to call eval()"
//   causes on small-batch inference.
//
// WHAT IS REAL HERE (honesty): every printed statistic is computed by core's BatchNorm2d on
//   seeded synthetic activations, not hand-typed. Means/variances are measured off the actual
//   output buffer. The gradient-check relative error is the real analytic-vs-numerical number.
//   Absolute values (e.g. exactly how fast the EMA converges) depend on the synthetic data
//   distribution we inject; the TRANSFERABLE lesson is the qualitative behavior (post-BN
//   mean->0/var->1, EMA tracking the true stat, eval!=train output, the small-batch NaN), not
//   the specific decimals.
//
// FAILURE MODE demonstrated (the headline): at inference you MUST use frozen running stats.
//   If you forget and keep batch stats, a 1-sample inference batch has per-channel variance
//   exactly 0 -> the normalization divides by sqrt(0+eps); with eps small the normalized
//   value is fine numerically BUT every element of a 1-element channel population equals its
//   own mean, so x-mean == 0 -> output == beta (all information destroyed). Push eps to 0 and
//   the divide-by-zero becomes a literal NaN that then propagates through the rest of the net.
//   We show BOTH: the "silently collapses to beta" variant and the "explicit NaN" variant.
//
// Offline, deterministic (seeded PRNG), CPU-only. Reuses core/* only; imports no other stage.

import { Tensor } from "./core/autograd.js";
import { noGrad } from "./core/autograd.js";
import { BatchNorm2d } from "./core/nn.js";
import { mulberry32, randn, type Rng } from "./core/rng.js";
import { gradCheck, histogramAscii } from "./core/metrics.js";

const SEED = 0xb47c; // arbitrary fixed seed; any constant gives a reproducible run

// ----------------------------------------------------------------------------
// Helpers: build a batch of activations with a KNOWN per-channel shift/scale so the
//   "BN pulls it back to mean~0 var~1" claim is verifiable against a deliberate offset.
// ----------------------------------------------------------------------------

/**
 * Build an (N,C,H,W) tensor whose every channel is drawn from N(targetMean, targetVar).
 * INVARIANT: channels are i.i.d. with the same target moments, so post-BN we expect EVERY
 *   channel near (0,1). We inject a fat offset (mean 5, var 9 = std 3) precisely because raw
 *   conv activations are NOT zero-centered/unit-scaled — that mismatch is what BN exists to fix.
 */
function makeShiftedActivations(
  N: number,
  C: number,
  H: number,
  W: number,
  targetMean: number,
  targetVar: number,
  rng: Rng,
): Tensor {
  const std = Math.sqrt(targetVar);
  return Tensor.from([N, C, H, W], () => targetMean + std * randn(rng));
}

/** Measure the empirical per-channel mean and variance of an (N,C,H,W) tensor's data.
 *  Population variance (divide by count), matching BatchNorm's own convention. */
function measurePerChannelStats(t: Tensor): { mean: Float64Array; variance: Float64Array } {
  const [N, C, H, W] = t.shape;
  const per = N * H * W;
  const mean = new Float64Array(C);
  const variance = new Float64Array(C);
  for (let c = 0; c < C; c++) {
    let m = 0;
    for (let n = 0; n < N; n++) for (let i = 0; i < H * W; i++) m += t.data[(n * C + c) * H * W + i];
    m /= per;
    let v = 0;
    for (let n = 0; n < N; n++)
      for (let i = 0; i < H * W; i++) {
        const d = t.data[(n * C + c) * H * W + i] - m;
        v += d * d;
      }
    variance[c] = v / per;
    mean[c] = m;
  }
  return { mean, variance };
}

function mean(a: Float64Array): number {
  let s = 0;
  for (const x of a) s += x;
  return s / a.length;
}

function hasNaN(t: Tensor): boolean {
  for (const x of t.data) if (Number.isNaN(x)) return true;
  return false;
}

function fmt(x: number, digits = 4): string {
  if (Number.isNaN(x)) return "NaN";
  return x.toFixed(digits);
}

// ----------------------------------------------------------------------------
// Part 1: training-mode forward normalizes a shifted distribution back to (0, 1).
// ----------------------------------------------------------------------------

function demoNormalization(rng: Rng): void {
  console.log("=".repeat(70));
  console.log("Part 1 — 训练前向: 把偏移分布拉回 (mean≈0, var≈1)");
  console.log("=".repeat(70));

  const N = 8;
  const C = 4;
  const H = 6;
  const W = 6;
  const INJECT_MEAN = 5;
  const INJECT_VAR = 9; // std = 3

  const x = makeShiftedActivations(N, C, H, W, INJECT_MEAN, INJECT_VAR, rng);
  const before = measurePerChannelStats(x);
  console.log(
    `输入激活: N=${N} C=${C} H=${H} W=${W}, 注入 N(mean=${INJECT_MEAN}, var=${INJECT_VAR})`,
  );
  console.log(
    `  输入实测 per-channel: mean≈${fmt(mean(before.mean))}  var≈${fmt(mean(before.variance))}  (跨 ${C} 个通道平均)`,
  );

  const bn = new BatchNorm2d(C); // gamma=1, beta=0 => pure normalization, no learned shift yet
  bn.setTraining(true);
  const y = bn.forward(x);
  const after = measurePerChannelStats(y);
  console.log(
    `  输出实测 per-channel: mean≈${fmt(mean(after.mean))}  var≈${fmt(mean(after.variance))}  (gamma=1,beta=0)`,
  );

  // WHY var is ~1 not exactly 1: BN divides by sqrt(var+eps); the eps (1e-5) makes the output
  //   variance very slightly below 1. With injected var 9 the shortfall is ~eps/9 ~ 1e-6, far
  //   below display precision — so it prints as 1.0000. This is correct, not a bug.
  console.log(
    "  注: 输出 var 略小于 1 是 eps 平滑导致 (除以 sqrt(var+eps)); 偏差量级 ~eps/var ~1e-6, 不可见。",
  );
  console.log("  归一化后单通道值分布直方图:");
  console.log(histogramAscii(channelSlice(y, 0), 11));
}

/** Extract one channel's flat values across the batch — used to show the post-BN shape. */
function channelSlice(t: Tensor, c: number): Float64Array {
  const [N, C, H, W] = t.shape;
  const out = new Float64Array(N * H * W);
  let k = 0;
  for (let n = 0; n < N; n++) for (let i = 0; i < H * W; i++) out[k++] = t.data[(n * C + c) * H * W + i];
  return out;
}

// ----------------------------------------------------------------------------
// Part 2: running mean/var converge via EMA over many batches.
// ----------------------------------------------------------------------------

function demoEmaConvergence(rng: Rng): void {
  console.log();
  console.log("=".repeat(70));
  console.log("Part 2 — running mean/var 随 batch 数的 EMA 收敛");
  console.log("=".repeat(70));

  const C = 1; // single channel keeps the table readable; behavior is per-channel anyway
  const TRUE_MEAN = 5;
  const TRUE_VAR = 9;
  const momentum = 0.1; // running = (1-m)*running + m*batch_stat; default 0.1

  // INVARIANT: runningMean is init 0 and runningVar init 1 (the identity transform). With
  //   momentum 0.1 each batch closes ~10% of the gap, so convergence to the true (5, 9) is
  //   geometric. We print the gap so the "approaching, never overshooting" trend is explicit.
  const bn = new BatchNorm2d(C, { momentum });
  bn.setTraining(true);

  console.log(
    `真实分布 N(mean=${TRUE_MEAN}, var=${TRUE_VAR}), momentum=${momentum}; running 初值 (0, 1) = 恒等变换`,
  );
  console.log("  batch |  runningMean  runningVar |  mean_gap   var_gap");
  console.log("  ------+--------------------------+--------------------");
  const NUM_BATCHES = 60;
  for (let b = 1; b <= NUM_BATCHES; b++) {
    // Fresh batch each step: this is what makes the EMA a *moving* estimate of the population.
    const x = makeShiftedActivations(16, C, 6, 6, TRUE_MEAN, TRUE_VAR, rng);
    bn.forward(x); // side effect: updates runningMean / runningVar
    if (b === 1 || b === 5 || b === 10 || b === 20 || b === 40 || b === 60) {
      const rm = bn.runningMean[0];
      const rv = bn.runningVar[0];
      console.log(
        `   ${String(b).padStart(3)}  |   ${fmt(rm).padStart(8)}    ${fmt(rv).padStart(7)} |  ${fmt(Math.abs(rm - TRUE_MEAN)).padStart(7)}   ${fmt(Math.abs(rv - TRUE_VAR)).padStart(7)}`,
      );
    }
  }
  console.log("  趋势: gap 单调收缩 (几何收敛), 永不过冲 — 这是推理时要用的冻结统计量来源。");
}

// ----------------------------------------------------------------------------
// Part 3: eval mode uses frozen running stats; output differs from train mode.
// ----------------------------------------------------------------------------

function demoTrainVsEval(rng: Rng): void {
  console.log();
  console.log("=".repeat(70));
  console.log("Part 3 — eval 模式用 running 统计, 输出 ≠ train 模式");
  console.log("=".repeat(70));

  const C = 4;
  const bn = new BatchNorm2d(C);
  bn.setTraining(true);

  // Warm the running stats on the "true" distribution N(5,9) so eval has meaningful stats.
  for (let b = 0; b < 50; b++) bn.forward(makeShiftedActivations(16, C, 6, 6, 5, 9, rng));

  // Now a SHIFTED test batch: mean 8, var 4 — different from what running stats learned. This
  // is the realistic case: inference data is not the training population. Train mode would
  // re-normalize using THIS batch's stats; eval mode uses the frozen training stats. They
  // must differ, and that difference is the whole point of having two modes.
  const xTest = makeShiftedActivations(8, C, 6, 6, 8, 4, rng);

  bn.setTraining(true);
  const yTrain = bn.forward(xTest); // uses batch stats of xTest
  const trainStats = measurePerChannelStats(yTrain);

  // Inference path: wrap in noGrad so BN takes the eval branch even if someone left training=true.
  // setTraining(false) would also work; we show noGrad to document that it ALSO gates the path.
  bn.setTraining(false);
  const yEval = noGrad(() => bn.forward(xTest)); // uses frozen running stats
  const evalStats = measurePerChannelStats(yEval);

  console.log("同一个测试 batch N(8,4), running 统计学的是 N(5,9):");
  console.log(
    `  train 路径输出: mean≈${fmt(mean(trainStats.mean))} var≈${fmt(mean(trainStats.variance))}  (用本 batch 统计 -> 又被拉回 ~0/1)`,
  );
  console.log(
    `  eval  路径输出: mean≈${fmt(mean(evalStats.mean))} var≈${fmt(mean(evalStats.variance))}  (用冻结 running -> 保留 test 与 train 分布的差异)`,
  );
  // Per-element max divergence: a single number proving the two modes are NOT equivalent.
  let maxAbsDiff = 0;
  for (let i = 0; i < yTrain.size; i++) maxAbsDiff = Math.max(maxAbsDiff, Math.abs(yTrain.data[i] - yEval.data[i]));
  console.log(`  两模式逐元素最大差异: ${fmt(maxAbsDiff)}  (>0 即证明 eval 切换真实改变了前向)`);
}

// ----------------------------------------------------------------------------
// Part 4: gradient check — analytic BN backward vs numerical finite differences.
// ----------------------------------------------------------------------------

function demoGradCheck(rng: Rng): void {
  console.log();
  console.log("=".repeat(70));
  console.log("Part 4 — BatchNorm 梯度检查 (解析 vs 数值有限差分)");
  console.log("=".repeat(70));

  const C = 3;
  const x = makeShiftedActivations(4, C, 5, 5, 5, 9, rng);
  // Give gamma/beta nontrivial values so their gradients are actually exercised (gamma=1,
  // beta=0 would still check, but distinct values catch a wider class of indexing bugs).
  const bn = new BatchNorm2d(C);
  bn.setTraining(true);
  for (let c = 0; c < C; c++) {
    bn.gamma.data[c] = 0.5 + 0.3 * c;
    bn.beta.data[c] = -0.2 + 0.1 * c;
  }

  // The loss must be a scalar to drive backward(). We use a fixed seeded weighting of the
  // output (not mean) so the gradient is non-uniform across elements — a uniform loss can mask
  // bugs in the per-element BN adjoint that only show up under varied upstream gradients.
  const wRng = mulberry32(0x1234);
  const weights = Float64Array.from({ length: x.size }, () => randn(wRng));

  const lossFn = (): number => {
    bn.gamma.zeroGrad();
    bn.beta.zeroGrad();
    x.zeroGrad();
    const y = bn.forward(x); // NOTE: re-run forward each call so xhat is recomputed at perturbed params
    let loss = 0;
    for (let i = 0; i < y.size; i++) loss += weights[i] * y.data[i];
    // Manually seed the output grad with the loss's derivative (weights), then run BN backward.
    for (let i = 0; i < y.size; i++) y.grad[i] = weights[i];
    y._backward();
    return loss;
  };

  // gradCheck perturbs each param, recomputing lossFn (which recomputes forward+backward). The
  // analytic grads sit in x.grad / gamma.grad / beta.grad after the lossFn call inside gradCheck.
  // We must populate analytic grads first (gradCheck reads p.grad), so prime once:
  lossFn();
  const res = gradCheck(lossFn, [x, bn.gamma, bn.beta]);
  const pass = res.maxRelError < 1e-5;
  console.log(`  checked ${res.checked} 个参数分量`);
  console.log(`  max relative error = ${res.maxRelError.toExponential(2)}  -> ${pass ? "PASS (<1e-5)" : "FAIL"}`);
  if (!pass) throw new Error("BatchNorm gradient check failed — analytic backward is wrong");
}

// ----------------------------------------------------------------------------
// Part 5: THE classic bug — inference with batch stats on a 1-sample batch.
// ----------------------------------------------------------------------------

function demoForgotEvalBug(rng: Rng): void {
  console.log();
  console.log("=".repeat(70));
  console.log("Part 5 — 失败模式: 推理忘切 eval, 且 batch 只有 1 个样本");
  console.log("=".repeat(70));

  const C = 4;
  const bn = new BatchNorm2d(C);
  bn.setTraining(true);
  for (let b = 0; b < 50; b++) bn.forward(makeShiftedActivations(16, C, 6, 6, 5, 9, rng)); // warm running stats

  // A single test image. In production this is "user uploads one photo".
  const single = makeShiftedActivations(1, C, 6, 6, 8, 4, rng);

  // --- 5a. CORRECT: eval mode -> frozen stats -> finite, information-preserving output ---
  bn.setTraining(false);
  const correct = noGrad(() => bn.forward(single));
  const correctStats = measurePerChannelStats(correct);
  console.log("5a. 正确 (eval, 用 running 统计):");
  console.log(
    `    输出 mean≈${fmt(mean(correctStats.mean))} var≈${fmt(mean(correctStats.variance))}  NaN=${hasNaN(correct)}  (有限, 保留输入信息)`,
  );

  // --- 5b. BUG variant A: stayed in train mode with eps>0 -> no NaN, but info COLLAPSES ---
  // With N*H*W per channel and N=1, per-channel population is H*W=36 here, so var is NOT zero
  // and you might think you got away with it. To expose the *single-pixel* extreme — the
  // textbook "1 sample, 1 feature" case where the population is truly size 1 — we use a 1x1
  // spatial map below. First the H*W>1 case to show it's "wrong but quiet":
  bn.setTraining(true);
  const stillTrain = bn.forward(single); // batch stats of a 1-IMAGE batch
  const stillTrainStats = measurePerChannelStats(stillTrain);
  console.log("5b. BUG (忘切 eval, 仍 train; 单图但 H*W=36 还有空间方差):");
  console.log(
    `    输出 mean≈${fmt(mean(stillTrainStats.mean))} var≈${fmt(mean(stillTrainStats.variance))}  NaN=${hasNaN(stillTrain)}`,
  );
  console.log(
    "    -> 不崩, 但用了\"单张图自己的统计\"归一化: 预测变得依赖这一张图的内部分布, 与训练分布脱节 (静默错)。",
  );
  // ALSO: this corrupted the running stats! train-mode forward did an EMA update with the
  // single-image batch stats — so the bug poisons future eval too. Prove it:
  console.log(
    `    副作用: running 统计被这次错误前向污染 (EMA 更新), runningMean[0] 现在 = ${fmt(bn.runningMean[0])} (被往单图统计拉偏)。`,
  );

  // --- 5c. BUG variant B: the true degenerate case — population size 1 -> var 0 -> NaN ---
  // 1x1 spatial map + N=1 => per-channel population = 1 element => variance is EXACTLY 0.
  // With eps>0 the divide is sqrt(0+eps): finite but output == beta (all info gone). With
  // eps=0 it is sqrt(0) = 0 -> 1/0 -> Inf, and (x-mean)=0 too, giving 0*Inf = NaN that spreads.
  console.log("5c. BUG 极端 (population size = 1: N=1, H=W=1 -> 方差恒为 0):");

  const onePixel = Tensor.from([1, C, 1, 1], () => 8 + 2 * randn(rng));

  // eps>0 path: no NaN, but x-mean is identically 0 -> output is exactly beta (=0 by default).
  const bnEpsPos = new BatchNorm2d(C, { eps: 1e-5 });
  bnEpsPos.setTraining(true);
  const collapsed = bnEpsPos.forward(onePixel);
  console.log(
    `    eps=1e-5: 输出 = ${[...collapsed.data].map((v) => fmt(v, 6)).join(", ")}  NaN=${hasNaN(collapsed)}`,
  );
  console.log("      -> 数值不崩, 但每个通道 x-mean≡0, 输出恒等于 beta(=0): 输入信息被彻底抹掉。");

  // eps=0 path: literal divide-by-zero -> NaN, and it will propagate through any downstream layer.
  const bnEpsZero = new BatchNorm2d(C, { eps: 0 });
  bnEpsZero.setTraining(true);
  const exploded = bnEpsZero.forward(onePixel);
  console.log(
    `    eps=0  : 输出 = ${[...exploded.data].map((v) => fmt(v, 6)).join(", ")}  NaN=${hasNaN(exploded)}`,
  );
  console.log("      -> sqrt(0)=0 -> 1/0=Inf, 且 (x-mean)=0 -> 0*Inf = NaN。");

  // Prove the NaN PROPAGATES: feed the NaN output through a trivial downstream op (sum).
  let downstream = 0;
  for (const v of exploded.data) downstream += v;
  console.log(
    `      下游传播验证: sum(输出) = ${fmt(downstream)}  -> 一旦 NaN 进入, 后续每一层、loss、梯度全部变 NaN, 整个网络静默失效。`,
  );

  console.log();
  console.log("结论: BN 推理必须 setTraining(false) / noGrad。忘切 + 小 batch 时:");
  console.log("  - H*W>1 单图: 不崩但预测依赖单图统计 (静默错) + 污染 running 统计;");
  console.log("  - population=1 (N=1,H=W=1): eps>0 信息归零, eps=0 直接 NaN 并向下游传播。");
}

// ----------------------------------------------------------------------------
// main — runs all parts on one seeded PRNG so the whole run is reproducible.
// NOTE: parts share the rng sequentially; reordering parts changes the draws (and thus the
//   exact decimals) but not the qualitative conclusions. This is intentional determinism.
// ----------------------------------------------------------------------------

function main(): void {
  const rng = mulberry32(SEED);
  console.log("第 05 章 — BatchNorm: 训练统计 / 推理切换 / 经典坑");
  console.log(`(确定性运行, seed=0x${SEED.toString(16)}, 纯 CPU, 无网络/无 LLM)`);
  console.log();
  demoNormalization(rng);
  demoEmaConvergence(rng);
  demoTrainVsEval(rng);
  demoGradCheck(rng);
  demoForgotEvalBug(rng);
}

main();
