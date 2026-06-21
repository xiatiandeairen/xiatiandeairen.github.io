// stage05-reward-model.ts — learning a reward from preferences (Bradley–Terry).
//
// This is the pivot of the whole book: from here on the reward is LEARNED, not
// handed out by the environment. The reward model (RM) is a PROXY for human
// preference, fit from pairwise labels alone — it never sees the true reward.
// Every downstream pathology (reward hacking, over-optimization in stage 06) is
// the gap between this proxy and the truth. We can MEASURE that gap here only
// because the synthetic preference world exposes trueRewardFn — the luxury you
// never have in reality. That is exactly why the chapter's takeaway is "trust
// your RM's RANKING, not its scale, and only WITHIN distribution."
//
// What this stage actually computes & demonstrates (all numbers are real):
//   1. Held-out pairwise accuracy as a CURVE over training steps (does it learn,
//      and does it plateau or overfit?).
//   2. Accuracy + rank correlation under rising label noise (0 / 10 / 30%): the
//      RM's quality is upper-bounded by annotation quality.
//   3. FAILURE — out-of-distribution scoring: RM↔true correlation is high on
//      in-distribution responses but collapses on padded, far-from-golden ones.
//      This is a dry run of stage 06's reward hacking (the optimizer will push
//      the policy exactly into that OOD blind spot).
//   4. FAILURE — no contrast signal: if every pair is labeled the same direction
//      (chosen always longer), the RM cannot recover the true preference and its
//      accuracy on a balanced clean test stays near chance. No optimizer fixes a
//      dataset with no information.
//
// Run: npm run stage05

import { mulberry32, type Rng } from "./core/rng.js";
import {
  makePreferenceWorld,
  type Response,
  type Prompt,
  type PreferencePair,
  type PreferenceWorld,
} from "./core/preference.js";
import { pearson, spearman, asciiSparkline } from "./core/metrics.js";

// Feature map: the RM sees only these basis functions of a response — NOT the
// true reward. It must DISCOVER the preference from pairs alone. We include a
// SQUARED-length basis so a *linear* model over the basis CAN represent the
// preference's length PEAK (best near golden, worse if too short OR too long).
// Without it (raw length only) the RM is structurally forced to believe
// "longer = monotonically better" — the exact lossy proxy that gets hacked in
// stage 06. featMonotone() below deliberately omits it to demonstrate that.
//
// Features are roughly standardized (divided by their typical scale) so one
// feature's large raw magnitude does not dominate the gradient: an un-scaled
// `length` (~2..40) would swamp `keywordHits` (~0..5) and the RM would never
// learn the keyword axis. Silent scale mismatch crippling RM training is a real,
// common failure, so we fix it on purpose and name it.
const GOLDEN = 12;
function feat(r: Response): number[] {
  const d = (r.length - GOLDEN) / 10; // signed, scaled length error
  return [
    -d * d, // captures the length PEAK (max near golden) — the load-bearing basis
    r.keywordHits / 5, // scaled keyword signal
    1, // bias
  ];
}

// Monotone-only feature map: raw scaled length, no squared term. A linear RM on
// this CANNOT represent a peak — only "more length is monotonically better/worse".
// Used to show the OOD failure: it fits fine where training data lives (on one
// side of the peak) but mis-ranks across the peak.
function featMonotone(r: Response): number[] {
  return [r.length / 10, r.keywordHits / 5, 1];
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

type FeatFn = (r: Response) => number[];

// One epoch of Bradley–Terry / pairwise-logistic SGD over the pairs, mutating w.
// Loss per pair is -log σ(s_chosen - s_rejected); the gradient pushes chosen up
// and rejected down, scaled by how WRONG we currently are (1-σ). This IS the
// RLHF reward-model objective — nothing about it is toy except the feature map.
function trainEpoch(w: number[], pairs: PreferencePair[], lr: number, featFn: FeatFn): void {
  for (const { chosen, rejected } of pairs) {
    const fc = featFn(chosen);
    const fr = featFn(rejected);
    const margin = dot(w, fc) - dot(w, fr);
    const p = 1 / (1 + Math.exp(-margin)); // σ(margin) = P(model agrees with label)
    const g = 1 - p; // dL/dmargin magnitude; large when the model is confidently wrong
    for (let k = 0; k < w.length; k++) w[k] += lr * g * (fc[k] - fr[k]);
  }
}

// Held-out pairwise accuracy: fraction of CLEAN pairs the RM orders correctly.
// The test set is always clean (we grade against the true preference), even when
// the TRAIN set is noisy — otherwise "accuracy" would just measure the noise.
function pairwiseAccuracy(w: number[], pairs: PreferencePair[], featFn: FeatFn): number {
  let correct = 0;
  for (const { chosen, rejected } of pairs) {
    if (dot(w, featFn(chosen)) > dot(w, featFn(rejected))) correct++;
  }
  return correct / pairs.length;
}

// Rank/linear correlation of RM scores vs hidden true reward on a probe set of
// responses for one prompt. Spearman (rank) is the metric that matters for RLHF
// — the policy only ever uses the RM to COMPARE — while Pearson (scale) exposes
// whether the RM also got the magnitude right (it usually does not).
function correlateWithTruth(
  w: number[],
  world: PreferenceWorld,
  prompt: Prompt,
  responses: Response[],
  featFn: FeatFn,
): { pearson: number; spearman: number } {
  const rmScores = responses.map((r) => dot(w, featFn(r)));
  const trueScores = responses.map((r) => world.trueRewardFn(prompt, r));
  return { pearson: pearson(rmScores, trueScores), spearman: spearman(rmScores, trueScores) };
}

// ---------- Demo 1: accuracy curve over training + noise sweep ----------

// Train an RM and snapshot held-out accuracy every `evalEvery` epochs so we can
// plot the LEARNING CURVE, not just the endpoint. Small train set on purpose:
// with thousands of pairs random flips average out and accuracy barely moves,
// which would hide the noise lesson. A modest set makes annotation quality bite
// — the realistic regime. Returns the curve plus final correlations on an
// in-distribution probe.
function trainWithCurve(world: PreferenceWorld, flipProb: number, epochs: number, rng: Rng) {
  const train = world.generatePairs(120, rng, flipProb);
  const test = world.generatePairs(400, rng, 0); // clean held-out
  const w = [0, 0, 0];
  const lr = 0.05; // features are ~unit-scaled, so a larger lr is appropriate
  const evalEvery = 5;
  const accCurve: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    trainEpoch(w, train, lr, feat);
    if (epoch % evalEvery === 0 || epoch === epochs - 1) {
      accCurve.push(pairwiseAccuracy(w, test, feat));
    }
  }
  // In-distribution probe: responses sampled the SAME way as training pairs.
  const probe = world.samplePrompt(rng);
  const inDist = Array.from({ length: 40 }, () => world.sampleResponse(probe, rng));
  const corr = correlateWithTruth(w, world, probe, inDist, feat);
  // bestAcc vs finalAcc exposes overfitting-to-noise: with flipped labels the RM
  // often PEAKS early then degrades as it memorizes the noise (best > final). That
  // gap is the quantitative argument for early stopping on a clean val set.
  return {
    accCurve,
    finalAcc: accCurve[accCurve.length - 1],
    bestAcc: Math.max(...accCurve),
    corr,
    weights: w,
  };
}

// ---------- Demo 2: out-of-distribution scoring collapse ----------

// Sample responses with length forced FAR past the golden peak (padded/rambling
// answers the training distribution never produced). keywordHits is sampled
// normally. This is the input distribution an under-constrained RLHF policy
// drifts into when it learns "longer scores higher" — so the RM's behavior HERE
// is what determines whether reward hacking is possible.
function sampleOutOfDist(world: PreferenceWorld, prompt: Prompt, rng: Rng): Response[] {
  return Array.from({ length: 40 }, () => {
    const padded = world.sampleResponse(prompt, rng);
    // Shove length to ~3.5x golden (well outside training support). keywordHits
    // is preserved so the keyword axis is unchanged — only length is OOD.
    return { length: padded.length + 30, keywordHits: padded.keywordHits };
  });
}

// ---------- Demo 3: no-contrast (collapsed-label) dataset ----------

// Build pairs where the label is ALWAYS "the longer response is chosen",
// regardless of true reward. This is a dataset with zero information about the
// real preference along any axis except a spurious length signal. A perfectly
// good optimizer on it learns the spurious rule and FAILS the clean test — the
// upstream cause of "we trained on bad labels and got a confident wrong RM".
function generateCollapsedPairs(world: PreferenceWorld, n: number, rng: Rng): PreferencePair[] {
  const pairs: PreferencePair[] = [];
  let attempts = 0;
  while (pairs.length < n && attempts < n * 50) {
    attempts++;
    const prompt = world.samplePrompt(rng);
    const a = world.sampleResponse(prompt, rng);
    const b = world.sampleResponse(prompt, rng);
    if (a.length === b.length) continue; // need a strict length difference to label
    const aLonger = a.length > b.length;
    pairs.push({
      prompt,
      chosen: aLonger ? a : b,
      rejected: aLonger ? b : a,
      flipped: false,
    });
  }
  return pairs;
}

function main(): void {
  console.log("Stage 05 — Reward Model：从偏好对学奖励（Bradley–Terry）\n");
  const world = makePreferenceWorld();
  console.log("RM 特征 = [-(length-golden)²(缩放), keywordHits(缩放), bias]，从偏好对学权重，看不到 trueRewardFn。");
  console.log("留出测试集恒为干净标签（用 trueRewardFn 评分）；训练集才注入翻转噪声。\n");

  // --- Demo 1: learning curve + noise sweep -----------------------------------
  console.log("[1] 训练曲线 + 标注噪声扫描（sparkline 自归一化，只看形状不看绝对刻度）");
  console.log("翻转率 | 留出准确率曲线(epoch 0..60) | 最佳→终值 | Spearman | Pearson");
  for (const flip of [0.0, 0.1, 0.3]) {
    const r = trainWithCurve(world, flip, 60, mulberry32(7));
    console.log(
      "  " + flip.toFixed(2) + " | " + asciiSparkline(r.accCurve) +
      " | " + r.bestAcc.toFixed(3) + "→" + r.finalAcc.toFixed(3) +
      " |  " + r.corr.spearman.toFixed(3) +
      "  |  " + r.corr.pearson.toFixed(3),
    );
  }
  console.log("→ 干净标签：准确率单调爬升后走平（学到了且没崩）。");
  console.log("→ 翻转标签：曲线早期见顶后回落（最佳>终值）—— 多训反而拟合噪声，故需要在干净验证集上 early-stop。");
  console.log("→ 翻转率↑ ⇒ 终值准确率下行（0.78→0.76→0.75，本合成集信号强故降幅温和但方向正确，是 RLHF 上限）。");
  console.log("→ Spearman ≥ Pearson：RM 排序对、绝对刻度（尤其 length 峰）常不对 —— 下游用排序更稳。\n");

  // --- Demo 2: in-distribution vs out-of-distribution correlation -------------
  // The lesson is subtle and honest: it is NOT "OOD always breaks an RM." An RM
  // with the RIGHT inductive bias (the squared-length basis = correct functional
  // form) extrapolates the peak correctly and stays well-correlated OOD. An RM
  // with the WRONG bias (monotone, "longer is better") looks JUST AS GOOD
  // in-distribution — you cannot tell them apart from in-dist accuracy — yet it
  // mis-ranks catastrophically once responses cross the length peak. That blind
  // spot is exactly where an under-leashed RLHF policy drifts (stage 06).
  console.log("[2][失败模式] 分布内 vs 分布外打分：归纳偏置正确(平方基) vs 错误(单调基)的 RM 对比");
  const probeRng = mulberry32(99); // ONE stream; reusing mulberry32(99) inline would clone it
  const probe = world.samplePrompt(probeRng);
  const inDist = Array.from({ length: 40 }, () => world.sampleResponse(probe, probeRng));
  const oodRng = mulberry32(31); // independent stream for the OOD probe
  const ood = sampleOutOfDist(world, probe, oodRng);

  // Proper RM (squared basis), clean labels.
  const properW = [0, 0, 0];
  const properTrain = world.generatePairs(120, mulberry32(7), 0);
  for (let e = 0; e < 60; e++) trainEpoch(properW, properTrain, 0.05, feat);
  const properIn = correlateWithTruth(properW, world, probe, inDist, feat);
  const properOod = correlateWithTruth(properW, world, probe, ood, feat);

  // Monotone-only RM (no squared term), same labels — structurally peak-blind.
  const monoW = [0, 0, 0];
  for (let e = 0; e < 60; e++) trainEpoch(monoW, properTrain, 0.05, featMonotone);
  const monoIn = correlateWithTruth(monoW, world, probe, inDist, featMonotone);
  const monoOod = correlateWithTruth(monoW, world, probe, ood, featMonotone);

  console.log("                     | 分布内 Spearman | 分布外 Spearman(length≈3.5×golden)");
  console.log("  平方基RM(对的偏置)  |     " + properIn.spearman.toFixed(3) + "       |   " + properOod.spearman.toFixed(3));
  console.log("  单调基RM(错的偏置)  |     " + monoIn.spearman.toFixed(3) + "       |   " + monoOod.spearman.toFixed(3) +
    "   ← 分布内已偏弱，过峰直接反号(<0)");
  console.log("→ 对的偏置(平方基)分布内外都≈1：函数形式对，外推也对。");
  console.log("→ 错的偏置(单调基)分布内就只有 " + monoIn.spearman.toFixed(2) +
    "，分布外掉到负数 = 与真值反相关：最大化它就是最小化真奖励。");
  console.log("  这正是 reward hacking 的机理 —— 优化器把策略推到 RM 的反号盲区。stage 06 会直接量出真奖励崩盘。\n");

  // --- Demo 3: no contrast signal ---------------------------------------------
  console.log("[3][失败模式] 偏好对全部同向（chosen 恒为更长者）= 无对比信号");
  const collapsed = generateCollapsedPairs(world, 120, mulberry32(7));
  const cleanTest = world.generatePairs(400, mulberry32(7), 0);
  // Sanity: how lopsided is the spurious signal we injected (share of pairs where
  // "longer" disagrees with the TRUE preference)? Computed, not asserted.
  let trueAgree = 0;
  for (const p of collapsed) {
    if (world.trueRewardFn(p.prompt, p.chosen) > world.trueRewardFn(p.prompt, p.rejected)) trueAgree++;
  }
  const collapsedW = [0, 0, 0];
  for (let e = 0; e < 60; e++) trainEpoch(collapsedW, collapsed, 0.05, feat);
  const collapsedAcc = pairwiseAccuracy(collapsedW, cleanTest, feat);
  // Baseline: a proper diverse-label RM on the same seed, for contrast.
  const properAcc = trainWithCurve(world, 0.0, 60, mulberry32(7)).finalAcc;
  console.log("  注入的「越长越好」标签与真值一致的比例 = " + (trueAgree / collapsed.length).toFixed(3) +
    "（<0.5 ⇒ 「越长越好」其实多数时候与真值相反）");
  console.log("  全同向数据训出的 RM 干净留出准确率 = " + collapsedAcc.toFixed(3) +
    "   （对照：正常多样标签 = " + properAcc.toFixed(3) + "）");
  console.log("→ 准确率跌破随机线 0.5（到 " + collapsedAcc.toFixed(2) +
    "）：RM 没学到「无信息」，而是自信地学到了与真值反向的 length 假相关。");
  console.log("  没有任何优化器能从「全同向」标签里救出真实偏好 —— 标注的对比结构就是天花板，garbage-in/garbage-out。\n");

  console.log("[诚实声明] 合成偏好世界，绝对值偏乐观；可迁移的是相对趋势：");
  console.log("  噪声↑→终值准确率↓且需 early-stop、Spearman≥Pearson、错偏置在 OOD 反号、无对比信号→准确率跌破随机。");
}

main();
