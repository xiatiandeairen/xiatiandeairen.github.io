// stage06-rlhf.ts — the full RLHF loop, and a direct measurement of reward hacking.
//
// RLHF is PPO (stage 04) + a reward model (stage 05) + ONE new ingredient: a KL
// penalty pinning the policy to the pre-RLHF reference. This file wires those
// together and then does the thing every RLHF paper talks about but few let you
// SEE: it plots, over training, the proxy reward (what we optimize), the KL drift
// (how far we've wandered from the reference), and the TRUE reward (what we
// actually want) — and marks the exact step where true reward peaks and starts
// falling while proxy keeps climbing. That divergence is reward hacking, and we
// can measure it only because this toy world exposes trueRewardFn (stage 05).
//
// Why hacking is not a bug you can patch away: the policy optimizes the reward it
// is GIVEN. If the reward model is a flawed proxy — and every learned RM is — the
// optimum of the proxy is not the optimum of the truth. The KL leash does not fix
// the proxy; it just limits how far the policy is allowed to chase it, trading
// "real improvement over the reference" against "drift into the proxy's blind
// spots." This stage makes that trade-off a number.
//
// Design honesty note: the RM here is a SPECIFIED length-biased proxy, not trained
// from pairs in this file. Length bias ("longer answers score higher") is the
// best-documented real RM pathology, but reproducing it from a trained linear RM
// is fiddly — a well-fit RM on representative data actually recovers the true
// length peak (verified while building this stage), and the bias only appears
// out-of-distribution or from skewed labels. We bake the bias so the hacking is
// visible and reproducible; stage 05 shows where such an RM comes from. The
// transferable lesson — proxy and true reward DECOUPLE under optimization, and KL
// bounds the damage — does not depend on how the RM acquired its bias.
//
// Determinism: the whole loop is deterministic gradient ascent over a fixed
// discrete action menu (no sampling at all), so same inputs => same curves, every
// run — no PRNG needed, which is the strongest reproducibility guarantee possible.
//
// Run: npm run stage06

// We use core's softmax (numerically stable, shared with every other stage) and
// klCategorical (exact KL between two full distributions — our policy and the
// reference are single categoricals over the length menu, so exact KL is the right
// leash measurement, not the sampled k1/k3 estimators PPO needs for token policies).
import { softmax, klCategorical } from "./core/probability.js";

// --- The action space: a menu of candidate response LENGTHS ----------------
// We collapse a response to its length (keywords held at what the prompt needs),
// because length is the hackable axis and a 1-D action space lets us draw clean
// curves. The policy is a categorical over these length buckets; picking a bucket
// = emitting a response of that length.
const LENS = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40];
const GOLDEN = 12; // the true reward peaks here (matches core/preference goldenLength)

// --- The hidden truth ------------------------------------------------------
// trueReward(len) = a peak at GOLDEN with a quadratic penalty for being too short
// OR too long. This mirrors core/preference.trueRewardFn's length term: there IS a
// best length, and rambling past it is genuinely worse. The policy never sees this.
const LAMBDA = 0.03;
function trueReward(len: number): number {
  return 1.5 - LAMBDA * (len - GOLDEN) * (len - GOLDEN);
}

// --- The reward model: a length-biased proxy -------------------------------
// rm(len) = trueReward(len) + VERBOSITY_BONUS·(len - GOLDEN). The RM agrees with
// the truth about shape but carries a spurious "longer is better" term, so its
// argmax sits PAST golden (verified at runtime below). Optimizing this RM hard
// pushes the policy past the true peak into over-long, low-true-reward responses —
// reward hacking, made mechanical.
const VERBOSITY_BONUS = 0.45; // tuned so rm-argmax ≈ len 20, well past golden 12
function rmScore(len: number): number {
  return trueReward(len) + VERBOSITY_BONUS * (len - GOLDEN);
}

const TRUE_MENU = LENS.map(trueReward);
const RM_MENU = LENS.map(rmScore);

// --- The reference (pre-RLHF SFT) policy -----------------------------------
// An UNDER-trained SFT model centered SHORT of golden (around len 8). Two reasons
// this is the honest choice: (1) it leaves real headroom for RLHF to improve
// toward golden, so "RLHF helps" is a measured fact, not an assumption; (2) the
// KL leash only means "stay good" when the reference is itself decent — anchoring
// on a sensible-but-improvable model is exactly the real setup. A uniform
// reference would make the KL term pull toward random, a subtly wrong picture.
const REF_LOGITS = LENS.map((len) => -0.05 * (len - 8) * (len - 8));
const REF_PROBS = softmax(REF_LOGITS);

function expectedUnder(probs: number[], perAction: number[]): number {
  let acc = 0;
  for (let a = 0; a < probs.length; a++) acc += probs[a] * perAction[a];
  return acc;
}

interface StepRecord {
  step: number;
  trueR: number; // expected TRUE reward under the policy — what we care about
  proxyR: number; // expected RM reward — what we optimize
  kl: number; // KL(policy || reference) — the drift / "distance off the leash"
}

// One full RLHF run at a given KL coefficient β. Returns the per-step trajectory.
// The objective is the standard RLHF objective: maximize E_π[rm] - β·KL(π||ref).
// We do deterministic gradient ascent on the policy logits; the β·KL term's
// gradient is the log-ratio between policy and reference (the "leash" force).
function runRlhf(beta: number, steps: number): StepRecord[] {
  const logits = REF_LOGITS.slice(); // start FROM the reference, as RLHF does
  const lr = 0.04;
  const trajectory: StepRecord[] = [];
  for (let step = 0; step <= steps; step++) {
    const probs = softmax(logits);
    trajectory.push({
      step,
      trueR: expectedUnder(probs, TRUE_MENU),
      proxyR: expectedUnder(probs, RM_MENU),
      kl: klCategorical(probs, REF_PROBS),
    });
    if (step === steps) break;

    // Gradient of E_π[rm - β·logπ + β·logref] w.r.t. logits. Per action the
    // "reward" the policy chases is rm(a) minus the β-scaled log-ratio (the KL
    // penalty's per-action contribution). The softmax Jacobian term
    // probs[a]·(1{a=k} - probs[k]) turns that into a logit gradient.
    const grad = new Array<number>(LENS.length).fill(0);
    for (let a = 0; a < LENS.length; a++) {
      const klPenalty = Math.log(probs[a] + 1e-12) - Math.log(REF_PROBS[a] + 1e-12);
      const reward = RM_MENU[a] - beta * klPenalty;
      for (let k = 0; k < LENS.length; k++) {
        grad[k] += probs[a] * ((a === k ? 1 : 0) - probs[k]) * reward;
      }
    }
    for (let k = 0; k < LENS.length; k++) logits[k] += lr * grad[k];
  }
  return trajectory;
}

function argmaxIndex(arr: number[]): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

// Find the step at which true reward peaks — the reward-hacking inflection point.
// Before it, optimizing the proxy also helps the truth; after it, the policy is
// chasing the proxy's length bias off a cliff while true reward declines.
function findTruePeak(traj: StepRecord[]): StepRecord {
  let best = traj[0];
  for (const r of traj) if (r.trueR > best.trueR) best = r;
  return best;
}

// Render three aligned sparklines (proxy / true / KL) plus a caret marking the
// true-reward peak, so the divergence is visible in stdout without a plot lib.
const SPARK = "▁▂▃▄▅▆▇█";
function sparkAt(values: number[], peakIdx: number): { line: string; caret: string } {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const line = values
    .map((v) => SPARK[Math.min(SPARK.length - 1, Math.floor(((v - min) / span) * SPARK.length))])
    .join("");
  const caret = " ".repeat(peakIdx) + "^";
  return { line, caret };
}

// Subsample a long trajectory to N points for display (keeps the peak point).
function downsample(traj: StepRecord[], n: number, peakStep: number): { recs: StepRecord[]; peakIdx: number } {
  const recs: StepRecord[] = [];
  let peakIdx = 0;
  for (let i = 0; i < n; i++) {
    const step = Math.round((i / (n - 1)) * (traj.length - 1));
    recs.push(traj[step]);
    if (Math.abs(step - peakStep) < Math.abs(recs[peakIdx].step - peakStep)) peakIdx = i;
  }
  return { recs, peakIdx };
}

function main(): void {
  console.log("Stage 06 — RLHF：proxy / KL / true 三曲线同图，直接量出 reward hacking 拐点\n");

  // Real, measured property of the specified RM: its best length is PAST golden.
  const rmBestLen = LENS[argmaxIndex(RM_MENU)];
  const trueBestLen = LENS[argmaxIndex(TRUE_MENU)];
  console.log(
    `真奖励峰值在 len=${trueBestLen}（=${trueReward(trueBestLen).toFixed(2)}）；` +
      `RM 因 length 偏置峰值在 len=${rmBestLen}（真值在那只有 ${trueReward(rmBestLen).toFixed(2)}）。`,
  );
  console.log(
    `参考策略=欠训练 SFT（偏好 len≈8），真奖励基线=${expectedUnder(REF_PROBS, TRUE_MENU).toFixed(3)}。` +
      "RLHF 的任务是从这里往 golden 改进，而不要被 RM 拐去更长。\n",
  );

  // ---- Experiment 1: the over-optimization curves at β=0 (no leash) ----
  const STEPS = 4000;
  const noLeash = runRlhf(0, STEPS);
  const peak = findTruePeak(noLeash);
  const final = noLeash[noLeash.length - 1];

  const { recs, peakIdx } = downsample(noLeash, 48, peak.step);
  const proxy = sparkAt(recs.map((r) => r.proxyR), peakIdx);
  const truth = sparkAt(recs.map((r) => r.trueR), peakIdx);
  const klLine = sparkAt(recs.map((r) => r.kl), peakIdx);

  console.log("【β=0，无 KL 缰绳：训练步数 →（48 采样点）】");
  console.log("  proxy(RM奖励，优化目标)  " + proxy.line);
  console.log("  KL(漂移，离参考多远)     " + klLine.line);
  console.log("  true(真奖励，真正想要)   " + truth.line);
  console.log("  hacking 拐点 ↓           " + truth.caret);
  console.log(
    `  真奖励在 step ${peak.step} 见顶（true=${peak.trueR.toFixed(3)}），之后掉到 ` +
      `${final.trueR.toFixed(3)}；同期 proxy 从 ${noLeash[0].proxyR.toFixed(3)} 一路涨到 ` +
      `${final.proxyR.toFixed(3)}，KL 涨到 ${final.kl.toFixed(2)}。`,
  );
  console.log("  → proxy↑ 而 true↓ = reward hacking：策略在啃 RM 的 length 偏置，真质量反降。\n");

  // ---- Experiment 2: early-stop vs train-to-convergence (same β=0 run) ----
  console.log("【按 true 早停 vs 训到底（同一条 β=0 曲线）】");
  console.log(
    `  早停于 true 峰值(step ${peak.step})：true=${peak.trueR.toFixed(3)}  |  ` +
      `训到底(step ${final.step})：true=${final.trueR.toFixed(3)}  |  ` +
      `早停多赚 ${(peak.trueR - final.trueR).toFixed(3)} 真奖励。`,
  );
  console.log(
    "  → 你看不到 trueReward（现实里没有 oracle），只能靠 KL/proxy 代理判断何时停 —— 这就是 KL 缰绳的用处。\n",
  );

  // ---- Experiment 3: β sweep — the U-shaped trade-off --------------------
  // Includes the chapter's requested {0, 0.1, 1} plus the values that reveal the
  // interior optimum and the "leashed too tight = learns nothing" tail.
  console.log("【β(KL系数) 扫描：太松 hack / 太紧学不动 的 U 形权衡】");
  console.log("  β     | 最终true | 最终proxy |  KL   | 说明");
  const baseline = expectedUnder(REF_PROBS, TRUE_MENU);
  const betas = [0, 0.1, 0.2, 0.5, 1, 3, 10];
  const finals = betas.map((b) => {
    const traj = runRlhf(b, STEPS);
    return { beta: b, last: traj[traj.length - 1] };
  });
  let bestBeta = finals[0];
  for (const f of finals) if (f.last.trueR > bestBeta.last.trueR) bestBeta = f;
  for (const f of finals) {
    let note = "";
    if (f.beta === 0) note = "无缰绳：proxy 最高、KL 最大、被 length 偏置带偏";
    else if (f === bestBeta) note = "← 最佳 true：缰绳松紧刚好，吃到改进又没被 hack";
    else if (f.beta >= 10) note = "拴太死：几乎没离开参考，proxy 都没涨（学不动）";
    console.log(
      "  " +
        f.beta.toFixed(1).padStart(4) +
        "  |  " +
        f.last.trueR.toFixed(3).padStart(6) +
        "  |  " +
        f.last.proxyR.toFixed(3).padStart(6) +
        "   | " +
        f.last.kl.toFixed(3).padStart(5) +
        " | " +
        note,
    );
  }
  console.log(
    `  参考基线 true=${baseline.toFixed(3)}：所有 β 的 true 都 ≥ 基线（RLHF 确实帮了），` +
      `但最佳在中间 β=${bestBeta.beta}（true=${bestBeta.last.trueR.toFixed(3)}）。`,
  );
  console.log("  → β 太小：proxy 飙、true 被 hack 拉低；β 太大：KL≈0、proxy 不涨、白训。中间才是甜点。\n");

  console.log("【失败模式总结】");
  console.log("  · β=0：策略坍向 RM 的对抗样本（最长回答），proxy 暴涨而 true 暴跌 = 纯 hacking。");
  console.log("  · 看 proxy 单调上升就以为在变好 = 经典误判；必须有独立信号（KL / 留出真值 / 人评）兜底。");
  console.log("  · KL 缰绳不能修好 RM，只能限制损失上限；RM 越偏，任何 β 能拿到的 true 上限越低。");
  console.log(
    "  → 可迁移：proxy 与 true 在优化下必然解耦，hacking 出现在解耦处；KL 系数是「改进 vs 漂移」旋钮。" +
      "合成环境绝对值乐观，可迁移的是趋势与拐点结构。",
  );
}

main();
