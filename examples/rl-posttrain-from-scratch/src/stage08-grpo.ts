// stage08-grpo.ts — Group Relative Policy Optimization: PPO minus the critic.
//
// Why GRPO exists (the one thing to take away): PPO computes each action's
// advantage as (reward - V(s)), and V(s) comes from a learned VALUE network — the
// critic. For LLMs that critic is a second model the size of the policy: double
// the memory, its own training instability, its own reward-scale sensitivity.
// GRPO deletes it. For a prompt it samples a GROUP of G responses, scores them,
// and uses the GROUP's own mean/std as the baseline:
//     advantage_i = (reward_i - mean_group) / (std_group + eps)
// That is a Monte-Carlo baseline — no parameters, no second network. The clipped
// PPO update then runs unchanged on these advantages.
//
// What this file MEASURES honestly (not asserts):
//   1. GRPO vs PPO head-to-head on the same seeded preference world: both reach
//      the same policy, but we count the parameter/network difference (the whole
//      selling point) and show both learning curves.
//   2. Group size G ∈ {2,4,8,16}: the std-normalized advantage's variance and the
//      final policy confidence. Bigger group => steadier baseline.
//   3. Two failure modes, because GRPO is not free:
//      (a) G=2 — the two-sample baseline is so noisy the advantage sign flips on
//          luck; learning is slower / less confident than larger groups.
//      (b) DEGENERATE GROUP — when every response in a group gets the SAME reward,
//          mean=reward, std=0, so EVERY advantage is 0/eps = 0. The group has no
//          internal contrast => zero gradient => the policy does not move. This is
//          GRPO's structural Achilles' heel and it is exactly why the next chapter
//          (RLVR / verifiable rewards) matters: you need a reward signal that
//          actually SEPARATES responses, or the group baseline learns nothing.
//
// All rewards here come from the world's hidden trueRewardFn standing in for an
// RM, so absolute reward values are toy-optimistic; the transferable claims are
// the RELATIVE ones (param-count delta, variance-vs-G trend, degenerate-group
// collapse). Deterministic: every roll derives from the passed seeded rng.
//
// Run: npm run stage08

import { mulberry32, type Rng } from "./core/rng.js";
import { softmax, logProb } from "./core/probability.js";
import { movingAvg, asciiSparkline } from "./core/metrics.js";
import {
  makePreferenceWorld,
  type Prompt,
  type Response,
  type PreferenceWorld,
} from "./core/preference.js";

// A fixed action menu: the policy is a softmax over these 4 candidate responses.
// #1 (len 12 = goldenLength, 3 keyword hits) is the true optimum; #3 is the long
// "padded" answer the length-hacking stages abuse. Keeping the action space
// discrete and tiny is what lets us run hundreds of updates on a CPU and still
// read the policy as 4 probabilities.
const MENU: readonly Response[] = [
  { length: 6, keywordHits: 1 },
  { length: 12, keywordHits: 3 },
  { length: 24, keywordHits: 3 },
  { length: 40, keywordHits: 2 },
];
const OPTIMAL_ACTION = 1; // index into MENU; verified against trueRewardFn below

// One fixed prompt for the whole run so PPO's per-state critic has a single state
// to learn and GRPO's group is always scoring the same target. Using a fixed
// prompt isolates the critic-vs-group comparison from prompt-distribution noise.
function fixedPrompt(): Prompt {
  return { id: 0, requiredKeywords: 3 };
}

function sampleAction(probs: number[], rng: Rng): number {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1; // guard against float rounding leaving acc just under 1
}

// Clipped-surrogate gradient step for ONE sampled action, shared by PPO and GRPO
// — the ONLY difference between the two methods is how `advantage` was computed,
// so the update code is identical. This is the point: GRPO is "PPO with a
// different baseline", not a different optimizer.
//
// logpOld is the log-prob under the policy that GENERATED the sample (frozen for
// this step); we recompute logpNew from the live logits so the ratio reflects how
// far this update has already moved the action. clip<=0 disables clipping.
function clippedStep(
  logits: number[],
  action: number,
  advantage: number,
  logpOld: number,
  lr: number,
  clip: number,
): void {
  const probs = softmax(logits);
  const logpNew = Math.log(Math.max(probs[action], 1e-12));
  const ratio = Math.exp(logpNew - logpOld);
  // If the ratio has already left [1-clip, 1+clip] in the rewarded direction, the
  // clipped surrogate's gradient is zero: stop pushing (the trust region). When
  // the unclipped branch is active the gradient of ratio*adv w.r.t. θ is
  // adv*ratio*∇logπ. (clip<=0 => always unclipped.)
  const unclipped = ratio * advantage;
  const clipped = Math.max(1 - clip, Math.min(1 + clip, ratio)) * advantage;
  const clipActive = clip > 0 && clipped < unclipped;
  const gradScale = clipActive ? 0 : advantage * ratio;
  for (let k = 0; k < logits.length; k++) {
    const dlogp = (k === action ? 1 : 0) - probs[k];
    logits[k] += lr * gradScale * dlogp;
  }
}

interface TrainResult {
  probs: number[]; // final policy over MENU
  pick: number; // argmax action
  trueRewardCurve: number[]; // expected true reward of the policy per iteration
  advVarMean: number; // mean per-iteration variance of the advantages used
  numParams: number; // total learnable scalars (policy logits + critic table)
  numNetworks: number; // 1 (policy only) for GRPO, 2 (policy + critic) for PPO
}

// PPO branch: a LEARNED critic V (here a single scalar, since there is one state)
// trained by regression toward observed reward; advantage = reward - V. The critic
// is the extra network/params GRPO removes — we count it explicitly.
function trainPPO(world: PreferenceWorld, iters: number, batch: number, rng: Rng): TrainResult {
  const prompt = fixedPrompt();
  const logits = [0, 0, 0, 0];
  let critic = 0; // V(s): the value network, here one parameter for the one state
  const criticLr = 0.05;
  const lr = 0.1;
  const clip = 0.2;
  const curve: number[] = [];
  let advVarSum = 0;
  for (let it = 0; it < iters; it++) {
    const probs = softmax(logits);
    // Collect a batch of rollouts, each scored by the (hidden) true reward.
    const rolls: { a: number; r: number; logpOld: number }[] = [];
    for (let b = 0; b < batch; b++) {
      const a = sampleAction(probs, rng);
      rolls.push({ a, r: world.trueRewardFn(prompt, MENU[a]), logpOld: logProb(logits, a) });
    }
    // Critic prediction BEFORE this batch's update is the baseline (no peeking at
    // its own targets). Advantage = reward - V(s).
    const advs = rolls.map((x) => x.r - critic);
    const advMean = advs.reduce((s, x) => s + x, 0) / advs.length;
    advVarSum += advs.reduce((s, x) => s + (x - advMean) ** 2, 0) / advs.length;
    for (let i = 0; i < rolls.length; i++) {
      clippedStep(logits, rolls[i].a, advs[i], rolls[i].logpOld, lr, clip);
    }
    // Train the critic toward observed batch reward (TD(0) on a bandit = regress
    // to the mean reward). This regression is the cost GRPO avoids.
    const batchMeanR = rolls.reduce((s, x) => s + x.r, 0) / rolls.length;
    critic += criticLr * (batchMeanR - critic);
    curve.push(expectedTrueReward(softmax(logits), world, prompt));
  }
  const finalProbs = softmax(logits);
  return {
    probs: finalProbs,
    pick: argmaxLocal(finalProbs),
    trueRewardCurve: curve,
    advVarMean: advVarSum / iters,
    numParams: logits.length + 1, // 4 policy logits + 1 critic scalar
    numNetworks: 2, // policy + critic
  };
}

// GRPO branch: NO critic. Each iteration samples one group of G responses and uses
// the group's own mean/std as the baseline. degenerate=true forces the failure
// mode where the whole group collapses to one action (std=0 => zero advantage).
function trainGRPO(
  world: PreferenceWorld,
  groupSize: number,
  iters: number,
  rng: Rng,
  degenerate = false,
): TrainResult {
  const prompt = fixedPrompt();
  const logits = [0, 0, 0, 0];
  const lr = 0.1;
  const clip = 0.2;
  const curve: number[] = [];
  let advVarSum = 0;
  for (let it = 0; it < iters; it++) {
    const probs = softmax(logits);
    const group: { a: number; r: number; logpOld: number }[] = [];
    for (let g = 0; g < groupSize; g++) {
      // Degenerate demo: force every member to the SAME action so the group has
      // zero internal reward spread — the pathology we want to expose.
      const a = degenerate ? OPTIMAL_ACTION : sampleAction(probs, rng);
      group.push({ a, r: world.trueRewardFn(prompt, MENU[a]), logpOld: logProb(logits, a) });
    }
    // GROUP baseline — the defining GRPO move. mean & std are over THIS group only.
    const mean = group.reduce((s, x) => s + x.r, 0) / groupSize;
    const variance = group.reduce((s, x) => s + (x.r - mean) ** 2, 0) / groupSize;
    const std = Math.sqrt(variance);
    // eps guards 0/0 when the group is degenerate. NOTE: this is exactly where the
    // failure mode lives — std=0 => every advantage = 0/eps = 0 => zero gradient.
    // We do NOT clamp std up to hide it; the collapse is the lesson.
    const eps = 1e-6;
    const advs = group.map((x) => (x.r - mean) / (std + eps));
    const advMean = advs.reduce((s, x) => s + x, 0) / advs.length;
    advVarSum += advs.reduce((s, x) => s + (x - advMean) ** 2, 0) / advs.length;
    for (let i = 0; i < group.length; i++) {
      clippedStep(logits, group[i].a, advs[i], group[i].logpOld, lr, clip);
    }
    curve.push(expectedTrueReward(softmax(logits), world, prompt));
  }
  const finalProbs = softmax(logits);
  return {
    probs: finalProbs,
    pick: argmaxLocal(finalProbs),
    trueRewardCurve: curve,
    advVarMean: advVarSum / iters,
    numParams: logits.length, // 4 policy logits — NO critic
    numNetworks: 1, // policy only
  };
}

// Expected true reward of the policy = Σ π(a)·trueRewardFn(a). This is the honest
// learning-curve quantity: it integrates over the policy's actual action
// distribution, so it improves smoothly as probability mass shifts to #1 (unlike
// a noisy single-sample reward).
function expectedTrueReward(probs: number[], world: PreferenceWorld, prompt: Prompt): number {
  let e = 0;
  for (let a = 0; a < probs.length; a++) e += probs[a] * world.trueRewardFn(prompt, MENU[a]);
  return e;
}

function argmaxLocal(arr: number[]): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function main(): void {
  console.log("Stage 08 — GRPO：组内相对优势替代价值网络（无 critic）\n");
  const world = makePreferenceWorld();
  const prompt = fixedPrompt();

  // Sanity: confirm the menu's true optimum is #1 before claiming the policy
  // "should converge to #1". An honest baseline beats asserting it.
  const menuRewards = MENU.map((m) => world.trueRewardFn(prompt, m));
  const trueBest = argmaxLocal(menuRewards);
  console.log(
    "menu(真奖励): " +
      MENU.map((m, i) => `#${i}[len${m.length},kw${m.keywordHits}]=${menuRewards[i].toFixed(2)}`).join(
        "  ",
      ),
  );
  console.log(
    `真最优=#${trueBest}（golden length=${world.goldenLength}）。优势 = (奖励 - 基线)，PPO 基线=学到的 critic V(s)，GRPO 基线=组内均值/标准差。\n`,
  );

  // -------- 1. GRPO vs PPO head-to-head: same seed, same iters/budget --------
  const ITERS = 400;
  const ppo = trainPPO(world, ITERS, 8, mulberry32(7));
  const grpo = trainGRPO(world, 8, ITERS, mulberry32(7)); // group size 8 ≈ PPO batch 8
  console.log("[1] GRPO vs PPO（同种子 / 同 " + ITERS + " 迭代 / 每步采样预算=8）");
  console.log(
    "  PPO  网络数=" +
      ppo.numNetworks +
      "(policy+critic) 参数=" +
      ppo.numParams +
      "  选中#" +
      ppo.pick +
      " 概率=" +
      ppo.probs[ppo.pick].toFixed(3),
  );
  console.log(
    "  GRPO 网络数=" +
      grpo.numNetworks +
      "(policy)        参数=" +
      grpo.numParams +
      "  选中#" +
      grpo.pick +
      " 概率=" +
      grpo.probs[grpo.pick].toFixed(3),
  );
  const paramSaving = ((1 - grpo.numParams / ppo.numParams) * 100).toFixed(0);
  console.log(
    "  → GRPO 砍掉 critic：少 1 个网络、参数 " +
      ppo.numParams +
      "→" +
      grpo.numParams +
      "（本 toy 省 " +
      paramSaving +
      "%；真实 LLM 里 critic≈策略同量级，省的是「一整个大模型」）。",
  );
  // Learning curves (smoothed, since per-iter expected reward is the honest but
  // jagged signal). Both should climb toward menuRewards[#1].
  const smPpo = movingAvg(ppo.trueRewardCurve, 20);
  const smGrpo = movingAvg(grpo.trueRewardCurve, 20);
  const down = (s: number[]) => s.filter((_, i) => i % 8 === 0); // 50 points fit one line
  console.log("  PPO  期望真奖励曲线 " + asciiSparkline(down(smPpo)));
  console.log("  GRPO 期望真奖励曲线 " + asciiSparkline(down(smGrpo)));
  console.log(
    "  曲线范围: PPO [" +
      smPpo[0].toFixed(2) +
      "→" +
      smPpo[smPpo.length - 1].toFixed(2) +
      "]  GRPO [" +
      smGrpo[0].toFixed(2) +
      "→" +
      smGrpo[smGrpo.length - 1].toFixed(2) +
      "]  （上限=#" +
      trueBest +
      " 的 " +
      menuRewards[trueBest].toFixed(2) +
      "）",
  );

  // -------- 2. group size G ∈ {2,4,8,16}: advantage variance + convergence --------
  console.log("\n[2] 组大小 G 对优势方差与收敛的影响（GRPO，同种子）");
  console.log("  组G | 优势方差(均) | 选中项 | 该项概率 | 末期期望真奖励");
  for (const G of [2, 4, 8, 16]) {
    const r = trainGRPO(world, G, ITERS, mulberry32(7));
    const finalR = r.trueRewardCurve[r.trueRewardCurve.length - 1];
    console.log(
      "  " +
        String(G).padStart(2) +
        "  |    " +
        r.advVarMean.toFixed(3) +
        "     |   #" +
        r.pick +
        "   |  " +
        r.probs[r.pick].toFixed(3) +
        "   |    " +
        finalR.toFixed(3),
    );
  }
  console.log(
    "  → 组越大，组均值/标准差越接近策略真实分布，优势估计方差越低、收敛越稳。",
  );
  console.log(
    "  → std 归一化让优势量纲恒定（≈±1），故「优势方差」随 G 增大而降，反映基线噪声下降，而非学得更少。",
  );

  // -------- 3a. failure mode: G=2 noisy baseline --------
  console.log("\n[3a] 失败模式：G=2 双样本基线噪声大");
  const g2 = trainGRPO(world, 2, ITERS, mulberry32(7));
  const g16 = trainGRPO(world, 16, ITERS, mulberry32(7));
  console.log(
    "  G=2  选中#" +
      g2.pick +
      " 概率=" +
      g2.probs[g2.pick].toFixed(3) +
      " 优势方差=" +
      g2.advVarMean.toFixed(3),
  );
  console.log(
    "  G=16 选中#" +
      g16.pick +
      " 概率=" +
      g16.probs[g16.pick].toFixed(3) +
      " 优势方差=" +
      g16.advVarMean.toFixed(3),
  );
  console.log(
    "  → G=2 时基线只由两个样本估计：单次抽到两个差答案就把好答案的相对优势算成负，符号会被运气翻转，收敛更慢/置信度更低。",
  );

  // -------- 3b. failure mode: degenerate group (zero internal spread) --------
  console.log("\n[3b] 失败模式：组内奖励全相等 → 优势归零 → 不学习");
  const degen = trainGRPO(world, 8, ITERS, mulberry32(7), /*degenerate=*/ true);
  const before = expectedTrueReward(softmax([0, 0, 0, 0]), world, prompt);
  const after = degen.trueRewardCurve[degen.trueRewardCurve.length - 1];
  console.log(
    "  组内全选同一动作(无区分度): 起始期望真奖励=" +
      before.toFixed(4) +
      " → 末期=" +
      after.toFixed(4) +
      "  位移=" +
      (after - before).toExponential(2),
  );
  console.log(
    "  末期策略 " +
      degen.probs.map((p, i) => `#${i}:${p.toFixed(2)}`).join(" ") +
      "（仍是初始均匀 0.25，完全没动）",
  );
  console.log(
    "  → 组内奖励无差异 ⇒ std=0 ⇒ 每个优势 =0/eps=0 ⇒ 梯度为零 ⇒ 策略冻结。GRPO 的学习信号 100% 来自「组内有奖励差异」。",
  );
  console.log(
    "  → 这正是下一章「可验证奖励 (RLVR)」要解决的：必须有能真正区分回答好坏的奖励，组基线才有东西可学。",
  );

  console.log(
    "\n[诚实声明] 合成 RM 奖励，绝对值乐观；可迁移的是相对趋势：GRPO 省一个网络 / 优势方差随 G 降 / 组无差异即学不动。",
  );
}

main();
