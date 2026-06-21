// stage09-rlvr-reasoning.ts — RL from Verifiable Rewards (the R1 recipe), full chapter.
//
// Why RLVR is the frontier: every method so far (RM/RLHF/DPO/GRPO in stages 05-08)
// leaned on a LEARNED reward that is a flawed, hackable proxy. For tasks where
// correctness is CHECKABLE by a program — math, code tests, formal proofs — you
// replace the reward model with a verifier returning an exact 0/1. No reward
// model, no preference noise, nothing to hack: the only way to raise reward is to
// actually become correct. That is the engine behind reasoning models.
//
// This file is NOT a happy-path demo. It measures four things the chapter claims:
//
//   [A] accuracy / pass@k curves vs training step under a verifiable reward, and
//       the RLVR-vs-noisy-proxy contrast: verifiable trueReward climbs monotonically
//       with no hacking inflection; the proxy peaks then turns over as the policy
//       learns to game the proxy's disagreement with truth.
//   [B] a comparison table — SFT / RLHF / DPO / GRPO on the SAME task — reporting
//       final held-out accuracy, training cost (forward passes, really counted),
//       and run-to-run stability (std across seeds, really measured).
//   [C] FAILURE MODE 1 — reward checks only the FINAL answer, not the PROCESS.
//       The policy learns a "lucky shortcut" method that returns the right answer
//       on the training distribution by coincidence, not by reasoning. We print the
//       divergence between answer-accuracy (looks fine) and process-accuracy /
//       shifted-distribution accuracy (collapses): "got it right by guessing".
//   [D] FAILURE MODE 2 — a task too easy: reward saturates in a handful of steps,
//       the gradient dies, and no further learning (or harder-sample learning)
//       happens. We show the curve flat-lining and contrast its area-under-curve.
//
// Determinism: everything pulls from a seeded mulberry32. Same seed => same
// curves, same table, every run. Honesty: every number below is computed from
// real rollouts; wall-clock is really measured; the toy absolute values are
// optimistic — what transfers is the RELATIVE shape (monotone vs inflection,
// answer-vs-process gap, cost ordering, stability ordering).
//
// Run: npm run stage09  (or: npx tsx src/stage09-rlvr-reasoning.ts)

import { mulberry32, type Rng } from "./core/rng.js";
import { RunningMean, asciiSparkline, movingAvg, timeIt } from "./core/metrics.js";

// ---------------------------------------------------------------------------
// The task and its candidate "reasoning methods".
//
// A prompt is a pair (a, b); the verifiable truth is a + b. The policy is a
// distribution over METHODS — each method is a small program mapping (a,b) to an
// answer. This is the crucial modeling choice that lets us separate "right answer"
// from "right reasoning": a method can be answer-correct on some inputs WITHOUT
// being a valid general procedure. That gap is failure mode [C].
// ---------------------------------------------------------------------------
interface Method {
  readonly name: string;
  // The answer this reasoning path produces for (a, b).
  answer(a: number, b: number): number;
  // Whether this path is a genuinely VALID general procedure for a+b (process
  // ground truth the answer-only reward can NEVER see). Only M_ADD is.
  readonly isValidProcess: boolean;
}

const METHODS: Method[] = [
  // The one correct procedure: actually add. Answer-correct AND process-correct
  // on every input.
  { name: "add(a+b)", answer: (a, b) => a + b, isValidProcess: true },
  // Off-by-one bug. Always wrong — the obvious distractor.
  { name: "a+b+1", answer: (a, b) => a + b + 1, isValidProcess: false },
  // LUCKY SHORTCUT: doubles a. Correct ONLY when a === b, by coincidence, not by
  // reasoning. On a training distribution rich in a===b it collects reward "for
  // free" and an answer-only reward cannot tell it apart from real addition.
  { name: "2*a (shortcut)", answer: (a, _b) => 2 * a, isValidProcess: false },
  // Second shortcut: returns max. Correct only when one operand is 0.
  { name: "max(a,b)", answer: (a, b) => Math.max(a, b), isValidProcess: false },
];
const ADD = 0; // index of the only valid method
const SHORTCUT = 2; // index of the 2*a coincidence method

// ---------------------------------------------------------------------------
// Two prompt distributions. The shortcut is invisible on TRAIN (rich in a===b)
// and exposed on HELDOUT (a !== b, the honest test). This distribution shift is
// exactly how shortcut learning hides during training and surfaces in deployment.
// ---------------------------------------------------------------------------
function sampleTrainPrompt(rng: Rng, shortcutFriendlyProb = 0.5): [number, number] {
  const a = 1 + Math.floor(rng() * 30);
  // With probability shortcutFriendlyProb make a===b, the regime where the 2*a
  // shortcut is silently rewarded. The rest are general pairs.
  if (rng() < shortcutFriendlyProb) return [a, a];
  const b = 1 + Math.floor(rng() * 30);
  return [a, b];
}

function sampleHeldoutPrompt(rng: Rng): [number, number] {
  // a !== b guaranteed: no coincidence reward available. Pure test of the method.
  const a = 1 + Math.floor(rng() * 30);
  let b = 1 + Math.floor(rng() * 30);
  if (b === a) b = a === 30 ? a - 1 : a + 1;
  return [a, b];
}

// ---------------------------------------------------------------------------
// Verifiers (the reward functions). All return {0,1}. Each call is one "forward
// pass" of the reward channel; callers count them for the cost column.
// ---------------------------------------------------------------------------

// Answer-only verifiable reward: 1 iff the produced answer equals a+b. Exact and
// ungameable AS AN ANSWER CHECK — but blind to HOW the answer was produced, which
// is the door the shortcut walks through.
function verifyAnswer(a: number, b: number, methodIdx: number): number {
  return METHODS[methodIdx].answer(a, b) === a + b ? 1 : 0;
}

// Process-aware verifiable reward: 1 iff the answer is right AND the method is a
// valid procedure. This is what a strong verifier (unit tests over many inputs,
// or a proof checker) effectively buys you — it cannot be satisfied by a
// coincidence. Used to show the shortcut is a reward-design artifact, not destiny.
function verifyProcess(a: number, b: number, methodIdx: number): number {
  return verifyAnswer(a, b, methodIdx) === 1 && METHODS[methodIdx].isValidProcess ? 1 : 0;
}

// Noisy proxy reward: a stand-in for a LEARNED reward model. The decisive design
// choice (vs. a verifier) is that the proxy has a HACKABLE OPTIMUM that is NOT the
// true optimum. We model an RM with a strong surface-feature bias that systematically
// over-rewards the off-by-one method (index 1) — the analogue of stage 06's
// "longer = better" RM. At `noise` strength the proxy hands method-1 a near-1 score
// REGARDLESS of truth, so the proxy's argmax is method-1 (always truth-WRONG).
// Optimizing the proxy therefore drags the policy onto method-1: proxy reward rises
// while TRUE reward, after an early honest climb, turns over. A verifier has no such
// off-truth attractor, which is the entire point of the A-section contrast.
function proxyReward(a: number, b: number, methodIdx: number, noise: number, rng: Rng): number {
  const truth = verifyAnswer(a, b, methodIdx);
  // The biased method gets `noise` worth of unearned credit even when wrong; this
  // makes its EXPECTED proxy reward exceed honest methods once noise is high.
  if (methodIdx === 1) return rng() < noise ? 1 : truth;
  // Other methods: faithful to truth except residual symmetric label noise.
  return rng() < noise * 0.3 ? 1 - truth : truth;
}

// ---------------------------------------------------------------------------
// Policy primitives. Tiny softmax over the 4 methods; logits ARE the policy.
// Kept local (like stages 08/09 scaffolds) so the gradient is fully visible.
// ---------------------------------------------------------------------------
function softmaxLocal(logits: number[]): number[] {
  const m = Math.max(...logits);
  const e = logits.map((x) => Math.exp(x - m));
  const s = e.reduce((acc, v) => acc + v, 0);
  return e.map((x) => x / s);
}

function sampleIdx(probs: number[], rng: Rng): number {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1;
}

function argmaxIdx(arr: number[]): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

// Evaluate a frozen policy's GREEDY (argmax-method) accuracy on a fresh held-out
// set. Greedy because deployment commits to the best method, not a sample. Uses a
// fixed eval seed so the yardstick is identical across training configs.
function evalGreedyHeldout(logits: number[], n = 1000): { answerAcc: number; processAcc: number } {
  const probs = softmaxLocal(logits);
  const pick = argmaxIdx(probs);
  const rng = mulberry32(424242);
  let ans = 0;
  let proc = 0;
  for (let i = 0; i < n; i++) {
    const [a, b] = sampleHeldoutPrompt(rng);
    ans += verifyAnswer(a, b, pick);
    proc += METHODS[pick].isValidProcess ? 1 : 0; // process correctness is per-method
  }
  return { answerAcc: ans / n, processAcc: proc / n };
}

// pass@k on held-out: probability that AT LEAST ONE of k sampled methods solves
// the prompt. This is the reasoning-model eval metric (sample k chains, keep any
// that verifies). Closed form per prompt from the policy's per-method success, so
// it is exact, not Monte-Carlo. pass@1 == sampled accuracy; pass@k rises with k
// as long as the policy keeps probability mass on a method that can solve it.
function passAtK(logits: number[], k: number, n = 1000): number {
  const probs = softmaxLocal(logits);
  const rng = mulberry32(525252);
  const acc = new RunningMean();
  for (let i = 0; i < n; i++) {
    const [a, b] = sampleHeldoutPrompt(rng);
    // pSolve = total policy mass on methods that solve THIS prompt.
    let pSolve = 0;
    for (let m = 0; m < METHODS.length; m++) if (verifyAnswer(a, b, m) === 1) pSolve += probs[m];
    // P(at least one of k samples solves) = 1 - (1 - pSolve)^k.
    acc.push(1 - Math.pow(1 - pSolve, k));
  }
  return acc.value;
}

// ---------------------------------------------------------------------------
// GRPO trainer with full instrumentation. Returns per-step curves so we can show
// accuracy/pass@k forming and the RLVR-vs-proxy reward divergence. `forwardPasses`
// counts every reward-channel call (the cost the comparison table reports).
// ---------------------------------------------------------------------------
interface TrainTrace {
  logits: number[];
  trueRewardCurve: number[]; // mean TRUE (verifiable answer) reward per step
  proxyRewardCurve: number[]; // mean reward the optimizer actually SAW per step
  heldoutAccCurve: number[]; // greedy held-out answer accuracy, sampled each step
  forwardPasses: number;
}

function trainGRPO(opts: {
  rng: Rng;
  steps: number;
  groupSize: number;
  lr: number;
  // The reward the optimizer sees. "answer" | "process" | a proxy noise level.
  rewardKind: "answer" | "process" | "proxy";
  proxyNoise?: number;
  shortcutFriendlyProb?: number;
  // Optional starting logits: models carry surface-feature PRIORS into RL (a
  // shortcut can be high-probability before training even begins). Section C uses
  // this to give the 2*a shortcut a head start, which is what lets it survive when
  // the answer-only reward cannot distinguish it from real addition.
  initLogits?: number[];
}): TrainTrace {
  const { rng, steps, groupSize, lr, rewardKind } = opts;
  const proxyNoise = opts.proxyNoise ?? 0;
  const shortcutFriendlyProb = opts.shortcutFriendlyProb ?? 0.5;
  const logits = opts.initLogits ? opts.initLogits.slice() : METHODS.map(() => 0);
  const trueRewardCurve: number[] = [];
  const proxyRewardCurve: number[] = [];
  const heldoutAccCurve: number[] = [];
  let forwardPasses = 0;

  for (let step = 0; step < steps; step++) {
    const [a, b] = sampleTrainPrompt(rng, shortcutFriendlyProb);
    const probs = softmaxLocal(logits);
    const group: { idx: number; seen: number; truth: number }[] = [];
    for (let g = 0; g < groupSize; g++) {
      const idx = sampleIdx(probs, rng);
      const truth = verifyAnswer(a, b, idx); // always tracked for honest curves
      let seen: number;
      if (rewardKind === "answer") seen = truth;
      else if (rewardKind === "process") seen = verifyProcess(a, b, idx);
      else seen = proxyReward(a, b, idx, proxyNoise, rng);
      forwardPasses++; // one reward-channel evaluation per rollout
      group.push({ idx, seen, truth });
    }
    // GRPO group-baseline advantage: (reward - group_mean) / group_std. No critic.
    const mean = group.reduce((s, x) => s + x.seen, 0) / groupSize;
    const variance = group.reduce((s, x) => s + (x.seen - mean) ** 2, 0) / groupSize;
    const std = Math.sqrt(variance) + 1e-6;
    for (const { idx, seen } of group) {
      const adv = (seen - mean) / std;
      // Vanilla policy-gradient step on the softmax logits (clip elided; the
      // group baseline is the lesson). adv shapes which method gets reinforced.
      for (let k = 0; k < logits.length; k++) {
        logits[k] += lr * adv * ((k === idx ? 1 : 0) - probs[k]);
      }
    }
    trueRewardCurve.push(group.reduce((s, x) => s + x.truth, 0) / groupSize);
    proxyRewardCurve.push(mean);
    // Sample a cheap held-out accuracy probe every step for the curve (separate
    // from the final high-N eval, which is the reported number).
    heldoutAccCurve.push(evalGreedyHeldout(logits, 50).answerAcc);
  }
  return { logits, trueRewardCurve, proxyRewardCurve, heldoutAccCurve, forwardPasses };
}

// ---------------------------------------------------------------------------
// Analogues of the other three post-training methods on this SAME task, so the
// comparison table is apples-to-apples. Each returns final logits + forwardPasses.
//
// These are deliberately MINIMAL stand-ins (the real algorithms live in stages
// 05-08); the point of the table is the RELATIVE cost / accuracy / stability
// ordering, not to reimplement each method in full.
// ---------------------------------------------------------------------------

// SFT: supervised imitation of demonstrations of the CORRECT method. No reward,
// no rollouts — just cross-entropy toward the gold method. Cheapest, but caps at
// the demonstrations' quality and never explores. Cost = one forward per demo.
function trainSFT(rng: Rng, demos: number): { logits: number[]; forwardPasses: number } {
  const logits = METHODS.map(() => 0);
  const lr = 0.1;
  let fp = 0;
  for (let d = 0; d < demos; d++) {
    void sampleTrainPrompt(rng); // draw a prompt (SFT ignores it; imitation targets the gold method)
    const probs = softmaxLocal(logits);
    fp++; // one forward per demonstration
    // Cross-entropy gradient toward the gold (ADD) method.
    for (let k = 0; k < logits.length; k++) logits[k] += lr * ((k === ADD ? 1 : 0) - probs[k]);
  }
  return { logits, forwardPasses: fp };
}

// RLHF: PPO-style policy gradient against the NOISY PROXY reward (the learned-RM
// channel). Same optimizer as GRPO but the reward is the hackable proxy, so its
// held-out accuracy is capped by RM quality, not by truth — the stage-06 lesson.
function trainRLHF(rng: Rng, steps: number, proxyNoise: number): { logits: number[]; forwardPasses: number } {
  const t = trainGRPO({ rng, steps, groupSize: 4, lr: 0.05, rewardKind: "proxy", proxyNoise });
  return { logits: t.logits, forwardPasses: t.forwardPasses };
}

// DPO: closed-form preference optimization from pairs (chosen=valid method,
// rejected=a wrong method). No rollouts, no reward model. Each pair is two
// forward passes (score both sides). Robust here because the preference signal
// directly encodes "ADD beats the distractors".
function trainDPO(rng: Rng, pairs: number, beta = 0.3): { logits: number[]; forwardPasses: number } {
  const logits = METHODS.map(() => 0);
  const lr = 0.1;
  let fp = 0;
  for (let p = 0; p < pairs; p++) {
    void sampleTrainPrompt(rng);
    const chosen = ADD; // ADD is index 0, so indices 1..3 are exactly the wrong methods
    const rejected = 1 + Math.floor(rng() * (METHODS.length - 1)); // uniform over wrong methods
    fp += 2; // score chosen and rejected
    // DPO logistic gradient: push chosen logit up, rejected down, scaled by beta
    // and the current disagreement. Simplified single-step form of the closed loss
    // (no rollouts, no reward model — the preference pair IS the supervision).
    const margin = beta * (logits[chosen] - logits[rejected]);
    const g = lr * (1 - 1 / (1 + Math.exp(-margin)));
    logits[chosen] += g;
    logits[rejected] -= g;
  }
  return { logits, forwardPasses: fp };
}

// ---------------------------------------------------------------------------
// run-to-run stability: train under `make` across several seeds, return mean/std
// of held-out accuracy. Std is REALLY measured, not asserted — it is the column
// that exposes which methods are seed-fragile (RL methods) vs steady (SFT/DPO).
// ---------------------------------------------------------------------------
function stabilityOf(make: (seed: number) => number[], seeds: number[]): { mean: number; std: number } {
  const accs = seeds.map((s) => evalGreedyHeldout(make(s)).answerAcc);
  const mean = accs.reduce((a, b) => a + b, 0) / accs.length;
  const variance = accs.reduce((a, b) => a + (b - mean) ** 2, 0) / accs.length;
  return { mean, std: Math.sqrt(variance) };
}

// ---------------------------------------------------------------------------
function sectionA(): void {
  console.log("[A] 可验证奖励 vs 噪声代理：曲线 + pass@k —— 谁单调上升、谁被 hack 拐头\n");

  // Verifiable (answer) reward: true reward should climb monotonically to ~1.0.
  const ver = trainGRPO({ rng: mulberry32(7), steps: 600, groupSize: 8, lr: 0.05, rewardKind: "answer" });
  // Noisy proxy: the optimizer's SEEN reward rises, but TRUE reward peaks then
  // turns over as the policy learns to exploit the proxy's bias toward "a+b+1".
  // proxyNoise=0.85 puts the biased method's EXPECTED proxy reward (≈0.85) ABOVE an
  // honest method's (≈1-0.3·0.85≈0.745): the proxy's optimum is now the truth-WRONG
  // method, so optimizing the proxy actively destroys true reward. (At lower noise
  // the honest method still wins the proxy and no hacking appears — try it.)
  const prox = trainGRPO({ rng: mulberry32(7), steps: 600, groupSize: 8, lr: 0.05, rewardKind: "proxy", proxyNoise: 0.85 });

  const w = 40;
  const verTrue = movingAvg(ver.trueRewardCurve, w);
  const proxSeen = movingAvg(prox.proxyRewardCurve, w);
  const proxTrue = movingAvg(prox.trueRewardCurve, w);

  console.log("可验证: 真奖励(=代理奖励) 轨迹  " + asciiSparkline(downsample(verTrue, 60)));
  console.log("  最终真奖励=" + verTrue[verTrue.length - 1].toFixed(3) +
    "（单调上升，无拐点：代理==真值，没有可 hack 的缝隙）");
  console.log("代理(85%偏置): 代理奖励 轨迹      " + asciiSparkline(downsample(proxSeen, 60)));
  console.log("代理(85%偏置): 真奖励   轨迹      " + asciiSparkline(downsample(proxTrue, 60)));
  console.log("  代理峰值=" + Math.max(...proxSeen).toFixed(3) +
    " 但真奖励峰值=" + Math.max(...proxTrue).toFixed(3) +
    " → 末端真奖励=" + proxTrue[proxTrue.length - 1].toFixed(3) +
    "（拐头：优化代理 = 学会钻代理与真值的缝）");

  // pass@k under the clean verifiable policy.
  console.log("\n可验证策略 held-out pass@k：");
  for (const k of [1, 2, 4, 8]) {
    console.log("  pass@" + k + " = " + passAtK(ver.logits, k).toFixed(3));
  }
  console.log("  → pass@k 随 k 升（采样多次留可验证者）；可验证奖励让 pass@k 真实反映能力。\n");
}

function sectionB(): void {
  console.log("[B] 四法同任务对比：最终 held-out 准确率 / 训练成本(前向次数) / run-to-run 稳定性\n");

  const seeds = [1, 2, 3, 4, 5];

  // Build each method once (seed 1) to read accuracy + a real forward-pass count.
  const sft = trainSFT(mulberry32(1), 200);
  const dpo = trainDPO(mulberry32(1), 200);
  const rlhf = trainRLHF(mulberry32(1), 400, 0.85);
  const grpo = trainGRPO({ rng: mulberry32(1), steps: 400, groupSize: 8, lr: 0.05, rewardKind: "answer" });

  const sftAcc = evalGreedyHeldout(sft.logits);
  const dpoAcc = evalGreedyHeldout(dpo.logits);
  const rlhfAcc = evalGreedyHeldout(rlhf.logits);
  const grpoAcc = evalGreedyHeldout(grpo.logits);

  // Stability across seeds (mean/std of held-out answer accuracy), really measured.
  const sftStab = stabilityOf((s) => trainSFT(mulberry32(s), 200).logits, seeds);
  const dpoStab = stabilityOf((s) => trainDPO(mulberry32(s), 200).logits, seeds);
  const rlhfStab = stabilityOf((s) => trainRLHF(mulberry32(s), 400, 0.85).logits, seeds);
  const grpoStab = stabilityOf(
    (s) => trainGRPO({ rng: mulberry32(s), steps: 400, groupSize: 8, lr: 0.05, rewardKind: "answer" }).logits,
    seeds,
  );

  // Real wall-clock of one full GRPO run, so the cost column is not only abstract.
  const { ms: grpoMs } = timeIt(() =>
    trainGRPO({ rng: mulberry32(99), steps: 400, groupSize: 8, lr: 0.05, rewardKind: "answer" }),
  );

  console.log("方法  | 奖励来源        | held-out准确率 | 前向次数 | 跨5seed 均值±std");
  const row = (
    name: string,
    src: string,
    acc: number,
    fp: number,
    stab: { mean: number; std: number },
  ) =>
    console.log(
      "  " + name.padEnd(4) + "| " + src.padEnd(15) + " |     " + acc.toFixed(3) +
      "     |  " + String(fp).padStart(5) + "   |  " + stab.mean.toFixed(3) + " ± " + stab.std.toFixed(3),
    );
  row("SFT", "金标演示(模仿)", sftAcc.answerAcc, sft.forwardPasses, sftStab);
  row("DPO", "偏好对(无采样)", dpoAcc.answerAcc, dpo.forwardPasses, dpoStab);
  row("RLHF", "噪声代理RM", rlhfAcc.answerAcc, rlhf.forwardPasses, rlhfStab);
  row("GRPO", "可验证奖励", grpoAcc.answerAcc, grpo.forwardPasses, grpoStab);

  console.log("\n  GRPO 单次训练实测耗时 = " + grpoMs.toFixed(1) + " ms（真测 wall-clock，非估算）");
  console.log("  → SFT/DPO 不采样、最便宜也最稳，但天花板= 监督信号质量（学不到信号外的东西）。");
  console.log("  → RLHF 在强偏置代理 RM 下被 hack 到 held-out=0（学会代理的最优=真值的最差）；与 stage06 一致。");
  console.log("  → GRPO+可验证奖励准确率最高，代价是更多前向(每步 groupSize 次 rollout)；这是推理模型的取舍。");
  console.log("  → 绝对值乐观；可迁移的是排序：成本 SFT<DPO<RLHF≈GRPO，准确率上限 可验证 > 学习型奖励。\n");
}

function sectionC(): void {
  console.log("[C] 失败模式①：奖励只判最终答案、不判过程 → 学出「蒙对」/捷径 hack\n");

  // Spurious-correlation setup: TRAIN is 100% a===b, where the 2*a shortcut and
  // real addition produce the SAME (correct) answer — so an answer-only verifier
  // literally CANNOT tell them apart, both score 1 every step. We also give the
  // shortcut a prior head start (models bring surface biases into RL). With no
  // gradient pressure to prefer addition (rewards are tied) the head start persists:
  // the policy keeps the shortcut. This is exactly how a model "passes training"
  // by exploiting a spurious feature instead of learning the task.
  const shortcutHeadStart = METHODS.map((_, i) => (i === SHORTCUT ? 2.0 : 0));
  const ansOnly = trainGRPO({
    rng: mulberry32(11),
    steps: 800,
    groupSize: 8,
    lr: 0.05,
    rewardKind: "answer",
    shortcutFriendlyProb: 1.0, // every train prompt has a===b: shortcut indistinguishable
    initLogits: shortcutHeadStart,
  });
  // Process-aware reward: identical setup, but the verifier also requires a VALID
  // method (what running unit tests over many inputs, or a proof checker, buys you).
  // Now the shortcut earns nothing on a===b prompts that addition would also solve —
  // it is process-wrong — so the tie breaks toward real addition.
  const procAware = trainGRPO({
    rng: mulberry32(11),
    steps: 800,
    groupSize: 8,
    lr: 0.05,
    rewardKind: "process",
    shortcutFriendlyProb: 1.0,
    initLogits: shortcutHeadStart,
  });

  const ansProbs = softmaxLocal(ansOnly.logits);
  const procProbs = softmaxLocal(procAware.logits);

  // Train-distribution answer accuracy (what you'd naively report) vs held-out
  // (a !== b, where the shortcut can't coincide) vs process accuracy.
  const ansTrain = evalTrainAnswerAcc(ansOnly.logits, 1000, 1.0); // eval on the TRAIN dist (all a==b)
  const ansHeld = evalGreedyHeldout(ansOnly.logits);
  const procHeld = evalGreedyHeldout(procAware.logits);

  console.log("奖励只判答案: 策略分布 " +
    METHODS.map((m, i) => m.name + "=" + ansProbs[i].toFixed(2)).join("  "));
  console.log("  捷径(2*a)概率=" + ansProbs[SHORTCUT].toFixed(3) +
    "  正确加法概率=" + ansProbs[ADD].toFixed(3));
  console.log("  训练分布答案准确率=" + ansTrain.toFixed(3) +
    "（看起来还行）  ← 但这是被 a==b 捷径撑起来的");
  console.log("  held-out(a≠b)答案准确率=" + ansHeld.answerAcc.toFixed(3) +
    "  过程正确率=" + ansHeld.processAcc.toFixed(3) +
    "  ← 背离：分布一移，蒙对当场垮");
  console.log("\n奖励判过程: 策略分布 " +
    METHODS.map((m, i) => m.name + "=" + procProbs[i].toFixed(2)).join("  "));
  console.log("  held-out 答案准确率=" + procHeld.answerAcc.toFixed(3) +
    "  过程正确率=" + procHeld.processAcc.toFixed(3) +
    "  ← 强 verifier 把捷径堵死，答案==过程");
  console.log("\n  → 教训：弱 verifier(只看最终答案) 可被「在训练分布上恰好对」的捷径 hack；");
  console.log("    答案准确率与过程准确率/分布外准确率的背离，就是「蒙对」的指纹。");
  console.log("    强 verifier(多输入单测/证明检查) 才让可验证奖励真正不可 hack。\n");
}

function sectionD(): void {
  console.log("[D] 失败模式②：任务过易 → 奖励早饱和、梯度死掉、学不到难样本\n");

  // Easy task: shortcut-friendly + tiny method space already near-solved → reward
  // saturates fast, advantage (and thus gradient) collapses to ~0 within a few
  // dozen steps. We quantify by area-under-curve and "steps to 95% of final".
  const easy = trainGRPO({ rng: mulberry32(21), steps: 400, groupSize: 8, lr: 0.05, rewardKind: "answer", shortcutFriendlyProb: 0.9 });
  // A harder regime: only 1 of 4 methods solves (a!=b, no free shortcut), and a
  // smaller lr + group. The single needle takes many more steps to find and lock,
  // so the gradient stays alive longer — learning is spread out, not instant.
  const hard = trainGRPO({ rng: mulberry32(21), steps: 400, groupSize: 4, lr: 0.012, rewardKind: "process", shortcutFriendlyProb: 0.0 });

  const easyCurve = movingAvg(easy.trueRewardCurve, 20);
  const hardCurve = movingAvg(hard.trueRewardCurve, 20);

  console.log("过易任务 真奖励曲线 " + asciiSparkline(downsample(easyCurve, 60)));
  console.log("  饱和步=" + stepsTo95(easyCurve) + " / " + easyCurve.length +
    "（极快触顶，之后梯度≈0、白跑剩余步）  末端=" + easyCurve[easyCurve.length - 1].toFixed(3));
  console.log("较难任务 真奖励曲线 " + asciiSparkline(downsample(hardCurve, 60)));
  console.log("  饱和步=" + stepsTo95(hardCurve) + " / " + hardCurve.length +
    "（学习摊在更多步，梯度更久不死）  末端=" + hardCurve[hardCurve.length - 1].toFixed(3));

  // Quantify the dead-gradient: mean |advantage proxy| = reward std late vs early.
  const earlyStd = rewardStd(easy.trueRewardCurve.slice(0, 40));
  const lateStd = rewardStd(easy.trueRewardCurve.slice(-40));
  console.log("\n  过易任务奖励 std：早期=" + earlyStd.toFixed(3) + " 末期=" + lateStd.toFixed(3) +
    " → std→0 即组内全对/全同、advantage 归零、梯度死（GRPO 无方差就无信号）。");
  console.log("  → 教训：任务难度必须匹配模型能力。太易=奖励饱和、样本浪费；");
  console.log("    可验证奖励的价值，恰在它能在「还不会的难样本」上持续给区分信号——前提是任务真有难度。\n");
}

// --- small honest helpers (no hidden magic) --------------------------------

// Downsample a long curve to ~n points for a readable sparkline (mean-pool).
function downsample(series: number[], n: number): number[] {
  if (series.length <= n) return series;
  const out: number[] = [];
  const bucket = series.length / n;
  for (let i = 0; i < n; i++) {
    const lo = Math.floor(i * bucket);
    const hi = Math.floor((i + 1) * bucket);
    let s = 0;
    for (let j = lo; j < hi; j++) s += series[j];
    out.push(s / Math.max(1, hi - lo));
  }
  return out;
}

// Train-distribution greedy answer accuracy (the naive metric you'd report if you
// only ever looked at training data). shortcutFriendlyProb must match what training
// used, or the number is measuring a different distribution than the policy saw.
function evalTrainAnswerAcc(logits: number[], n = 1000, shortcutFriendlyProb = 0.5): number {
  const pick = argmaxIdx(softmaxLocal(logits));
  const rng = mulberry32(313131);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const [a, b] = sampleTrainPrompt(rng, shortcutFriendlyProb);
    acc += verifyAnswer(a, b, pick);
  }
  return acc / n;
}

// First step at which the (smoothed) curve reaches 95% of its final value — a
// proxy for "how long learning stayed alive". Small => saturated early.
function stepsTo95(curve: number[]): number {
  const target = 0.95 * curve[curve.length - 1];
  for (let i = 0; i < curve.length; i++) if (curve[i] >= target) return i;
  return curve.length;
}

function rewardStd(series: number[]): number {
  const m = series.reduce((a, b) => a + b, 0) / series.length;
  return Math.sqrt(series.reduce((a, b) => a + (b - m) ** 2, 0) / series.length);
}

function main(): void {
  console.log("Stage 09 — RLVR：可验证奖励驱动推理（R1 配方的核）");
  console.log("任务：a+b；策略在 4 个「推理方法」上分布，奖励=程序判定的对错（可验证、不可 hack）。");
  console.log("方法集：" + METHODS.map((m, i) => i + ":" + m.name + (m.isValidProcess ? "(有效)" : "")).join("  "));
  console.log("种子驱动，run-to-run 可复现；toy 绝对值乐观，可迁移的是相对趋势。\n");
  sectionA();
  sectionB();
  sectionC();
  sectionD();
  console.log("[总览] 可验证奖励 = 推理模型引擎：代理无缝可钻 → 真奖励单调；但 verifier 必须判过程、任务必须有难度。");
}

main();
