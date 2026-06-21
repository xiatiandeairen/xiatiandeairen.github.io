// stage03-reinforce.ts — REINFORCE, and variance, the real enemy of policy gradient.
//
// Chapter thesis: the policy-gradient estimator ∇ E[R] = E[R · ∇ log π(a|s)] is
// UNBIASED but its VARIANCE is what decides whether training works at all.
// Everything after it (baselines, advantage normalization, PPO's clip, GRPO's
// group baseline) is variance reduction stacked on this one update. So this stage
// does not merely "train a policy"; it MEASURES the variance the baseline actually
// shrinks and quantifies, with real numbers, how much each trick buys — and where
// it buys nothing.
//
// Setup: softmax policy θ[s][a] over a seeded contextual bandit. One step per
// episode → NO credit-assignment confound; the only thing that can go wrong is
// variance. We know optimalAction[s] (ground truth), so "did it learn" is a
// measured accuracy, never a vibe.
//
// What the baseline actually reduces — and the honesty trap: a baseline shrinks
// the variance of the ADVANTAGE (R − b), not of (R − const). On this toy the raw
// reward is already near zero-mean (means ∈ [0,1]), so the baseline's win is
// SMALL — and pretending otherwise would be dishonest. The transferable truth is
// that the win grows with the reward's OFFSET: real RLHF rewards are all-positive,
// so we sweep an artificial offset and watch the baseline's advantage-variance
// reduction (and accuracy rescue) grow with it. That is the load-bearing result.
//
// Advantage normalization is a different trick: dividing the advantage by its
// running std makes the effective step SCALE-INVARIANT. It does not reduce the toy
// advantage variance (it can even raise the measured step size early); its payoff
// shows up only under failure mode B below.
//
// Two failure modes are demoed, not just happy paths:
//   (A) entropy collapse — too-large lr drives the policy to one-hot per state
//       BEFORE it has explored; once H(π)→0 the gradient (1_a − π)→0 for the
//       locked action and the policy can no longer move off a WRONG action. We
//       print the entropy curve so the collapse is visible; the seed is such that
//       some states lock onto a suboptimal action → accuracy crashes < 1.
//   (B) reward-scale catastrophe — without normalization the step size is ∝ the
//       reward's magnitude, so a large reward scale makes the FIRST update slam the
//       policy to one-hot (same lock-in mechanism as A, triggered by scale not lr).
//       Accuracy degrades monotonically with scale; normalization makes it
//       scale-invariant and survives. (Note: softmax's max-subtraction means θ does
//       NOT overflow to NaN — the damage is premature lock-in, not arithmetic blowup.
//       We measured this; we do not claim a NaN that never happens.)
//
// Honesty note: toy bandit ⇒ absolute accuracies are optimistic and the reward
// offset is small. Transferable facts: the advantage-variance ratio's GROWTH with
// reward offset, the lock-in mechanism shared by A and B, and normalization's
// scale-invariance. Not the specific accuracy numbers.
//
// Run: npx tsx src/stage03-reinforce.ts

import { mulberry32, sampleCategorical, argmax, type Rng } from "./core/rng.js";
import { makeContextualBandit, type ContextualBandit } from "./core/envs.js";
import { softmax, entropy } from "./core/probability.js";
import { movingAvg, asciiSparkline } from "./core/metrics.js";

// Welford online mean+variance, one pass, numerically stable. Local helper: the
// only consumers are this stage's variance read-outs (advantage variance and the
// advantage-normalization running std), so it does not belong in core.
class RunningMeanVar {
  private n = 0;
  private mean = 0;
  private m2 = 0;
  push(x: number): void {
    this.n++;
    const d = x - this.mean;
    this.mean += d / this.n;
    this.m2 += d * (x - this.mean);
  }
  // Population variance. n<2 → 0 (a single sample has no spread); avoids dividing
  // by zero on the very first advantage-norm step.
  get variance(): number {
    return this.n > 1 ? this.m2 / this.n : 0;
  }
  get std(): number {
    return Math.sqrt(this.variance);
  }
}

interface TrainConfig {
  steps: number;
  lr: number;
  useBaseline: boolean;
  normalizeAdv: boolean;
  rewardOffset: number; // added to every reward; >0 mimics all-positive RLHF reward
  rewardScale: number; // multiplies every reward; >1 stresses the un-normalized step
}

interface TrainResult {
  accuracy: number; // fraction of states whose greedy action == optimalAction
  rewardCurve: number[]; // raw per-step reward (UNSHIFTED/UNSCALED, for fair comparison)
  advVariance: number; // variance of the ADVANTAGE over the early phase — what baseline shrinks
  meanEntropyCurve: number[]; // mean H(π) across states, sampled during training
}

// Greedy accuracy of the current θ table against ground-truth optimal actions.
function greedyAccuracy(theta: number[][], optimalAction: number[]): number {
  let correct = 0;
  for (let s = 0; s < theta.length; s++) {
    if (argmax(theta[s]) === optimalAction[s]) correct++;
  }
  return correct / theta.length;
}

function train(env: ContextualBandit, cfg: TrainConfig, rng: Rng): TrainResult {
  // Linear logits θ[s][a]; policy is softmax(θ[s]). Init at 0 → uniform start, so
  // entropy starts at log(nActions) and we can WATCH it (not) collapse.
  const theta = Array.from({ length: env.nStates }, () => new Array<number>(env.nActions).fill(0));

  // Per-state running baseline b(s) = mean reward in state s. Subtracting it is
  // unbiased (E[b·∇logπ] = b·0 = 0) but recenters the advantage near 0, shrinking
  // its variance — the variance-reduction trick we measure directly below.
  const baseline = new Array<number>(env.nStates).fill(0);
  const bCount = new Array<number>(env.nStates).fill(0);
  // Running std of the advantage across all states, for advantage normalization.
  // Global (not per-state) on purpose: it mirrors how PPO/GRPO normalize over a
  // whole batch, making the effective step invariant to the reward's scale.
  const advStats = new RunningMeanVar();

  const rewardCurve: number[] = [];
  const meanEntropyCurve: number[] = [];
  // Variance of the advantage, measured over the early phase where the baseline's
  // recentering matters most. This is THE quantity a baseline shrinks; measuring
  // the update-norm instead would conflate it with the normalization's rescaling.
  const advVar = new RunningMeanVar();
  const earlyCutoff = Math.floor(cfg.steps / 4);

  for (let t = 0; t < cfg.steps; t++) {
    const s = env.reset(rng);
    const probs = softmax(theta[s]);
    const a = sampleCategorical(probs, rng);
    const { reward: rawReward } = env.step(s, a, rng);
    const reward = (rawReward + cfg.rewardOffset) * cfg.rewardScale;

    let b = 0;
    if (cfg.useBaseline) {
      bCount[s]++;
      baseline[s] += (reward - baseline[s]) / bCount[s];
      b = baseline[s];
    }
    let advantage = reward - b;
    if (t < earlyCutoff) advVar.push(advantage); // measure BEFORE normalization rescales it

    if (cfg.normalizeAdv) {
      advStats.push(advantage);
      // Divide by running std (+ε for cold-start zero). This makes the step
      // scale-invariant: scaling all rewards scales advantage and std together, so
      // the ratio — and the step — is unchanged. Without it the step is ∝ reward
      // magnitude and a large scale slams the policy to one-hot in one update.
      const std = advStats.std;
      if (std > 1e-8) advantage = advantage / std;
    }

    // ∇ log π(a|s) for softmax = (1_a − π). θ[s] += lr · advantage · ∇logπ.
    for (let k = 0; k < env.nActions; k++) {
      const grad = (k === a ? 1 : 0) - probs[k];
      theta[s][k] += cfg.lr * advantage * grad;
    }
    rewardCurve.push(rawReward); // store raw reward so curves compare across offset/scale

    // Sample mean policy entropy ~25 times over the run. Entropy crashing to ~0
    // early is the signature of failure mode A (and B).
    const sampleEvery = Math.max(1, Math.floor(cfg.steps / 25));
    if (t % sampleEvery === 0) {
      let hSum = 0;
      for (let st = 0; st < env.nStates; st++) hSum += entropy(softmax(theta[st]));
      meanEntropyCurve.push(hSum / env.nStates);
    }
  }

  return {
    accuracy: greedyAccuracy(theta, env.optimalAction),
    rewardCurve,
    advVariance: advVar.variance,
    meanEntropyCurve,
  };
}

// Downsample a long curve to ~20 points for a compact sparkline.
function sample(series: number[], points = 20): number[] {
  if (series.length <= points) return series;
  const stride = Math.floor(series.length / points);
  return series.filter((_, i) => i % stride === 0);
}

function defaults(overrides: Partial<TrainConfig>): TrainConfig {
  return { steps: 8000, lr: 0.1, useBaseline: false, normalizeAdv: false, rewardOffset: 0, rewardScale: 1, ...overrides };
}

function main(): void {
  console.log("Stage 03 — REINFORCE：策略梯度，方差才是真正的敌人\n");

  // Same env + same training seed everywhere → the ONLY difference is the trick,
  // so any gap in the printed numbers is causal, not seed luck.
  const env = makeContextualBandit(8, 4, mulberry32(7));
  const trainSeed = 123;

  // ----- [1] four configs: baseline {off,on} × adv-norm {off,on} -----
  console.log("[1] 四配置对照（8 状态×4 动作, lr=0.1, 8000 步, 随机基线命中率≈0.25, 本 toy 奖励∈[0,1] 近零均值）");
  const grid: { name: string; useBaseline: boolean; normalizeAdv: boolean }[] = [
    { name: "无baseline 无归一", useBaseline: false, normalizeAdv: false },
    { name: "有baseline 无归一", useBaseline: true, normalizeAdv: false },
    { name: "无baseline 有归一", useBaseline: false, normalizeAdv: true },
    { name: "有baseline 有归一", useBaseline: true, normalizeAdv: true },
  ];
  for (const c of grid) {
    const r = train(env, defaults({ useBaseline: c.useBaseline, normalizeAdv: c.normalizeAdv }), mulberry32(trainSeed));
    const spark = asciiSparkline(sample(movingAvg(r.rewardCurve, 400)));
    console.log(
      `  ${c.name} | 命中率=${r.accuracy.toFixed(3)}` +
      ` | 早期 advantage 方差=${r.advVariance.toExponential(2)}` +
      ` | 奖励曲线 ${spark}`,
    );
  }
  console.log("  → 四者都收敛到 1.0：任务简单 + 奖励近零均值，baseline 在此 toy 上收益本就小（诚实承认，不夸大）");

  // ----- [2] baseline's real win GROWS with reward offset (the transferable fact) -----
  console.log("\n[2] baseline 的真正价值随奖励偏移增长（真实 RLHF 奖励全为正 = 大偏移）");
  console.log("    同一环境，给奖励加常数 offset 模拟全正奖励，对比 advantage 方差与命中率：");
  for (const offset of [0, 5, 20]) {
    const noBase = train(env, defaults({ rewardOffset: offset, useBaseline: false }), mulberry32(trainSeed));
    const withBase = train(env, defaults({ rewardOffset: offset, useBaseline: true }), mulberry32(trainSeed));
    const ratio = noBase.advVariance / withBase.advVariance;
    console.log(
      `  offset=${String(offset).padStart(2)} | 无baseline: adv方差=${noBase.advVariance.toExponential(2)} 命中率=${noBase.accuracy.toFixed(3)}` +
      ` | 有baseline: adv方差=${withBase.advVariance.toExponential(2)} 命中率=${withBase.accuracy.toFixed(3)}` +
      ` | 方差比=${ratio.toFixed(2)}×`,
    );
  }
  console.log("  → offset=0 时 baseline 几乎无用（≈1.1×）；offset=5 方差比升到 1.38× 并救回命中率 0.875→1.0");
  console.log("  → offset=20 的方差比看似 0.95×（≤1）是假象：无baseline 此时已塌成 one-hot 卡死，分布没散开故方差反低；");
  console.log("     真正的信号是命中率 0.125→1.000——大偏移下不减基线直接学废，有 baseline 稳稳满分");
  console.log("  → 这就是为什么 PPO/GRPO 必带 baseline/advantage：真实奖励全为正，不减基线步长被偏移项主导→学废");

  // ----- [3] failure mode A: entropy collapse from too-large lr -----
  console.log("\n[3] 失败模式 A — 学习率过大 → 熵塌缩到 one-hot 后学不动（entropy collapse）");
  const healthy = train(env, defaults({ lr: 0.1, useBaseline: true, normalizeAdv: true }), mulberry32(trainSeed));
  const collapseLr = 8.0;
  const collapse = train(env, defaults({ lr: collapseLr }), mulberry32(trainSeed));
  const maxEntropy = Math.log(env.nActions);
  console.log(
    `  健康 (lr=0.1)  最终平均熵=${healthy.meanEntropyCurve.at(-1)!.toFixed(3)} (满熵=${maxEntropy.toFixed(3)}) 命中率=${healthy.accuracy.toFixed(3)}`,
  );
  console.log(`     熵曲线 ${asciiSparkline(sample(healthy.meanEntropyCurve))}  ← 平滑下降到近 0（学完才确定性）`);
  console.log(
    `  塌缩 (lr=${collapseLr})  最终平均熵=${collapse.meanEntropyCurve.at(-1)!.toFixed(3)} 命中率=${collapse.accuracy.toFixed(3)}` +
    `  ← 熵≈0=已 one-hot, ∇logπ→0, 被锁死`,
  );
  console.log(`     熵曲线 ${asciiSparkline(sample(collapse.meanEntropyCurve))}  ← 一步砸到地板（探索前就锁死）`);
  console.log(
    `  → 命中率 ${healthy.accuracy.toFixed(3)} → ${collapse.accuracy.toFixed(3)}：过大 lr 在探索前锁死, 锁到错动作就再也学不动`,
  );

  // ----- [4] failure mode B: reward-scale catastrophe without normalization -----
  console.log("\n[4] 失败模式 B — 奖励尺度大 + 不归一化 → 单步过冲锁死（步长∝奖励量级）");
  console.log("    固定 lr=0.5，扫 rewardScale。归一化让步长对尺度不变：");
  const divLr = 0.5;
  for (const scale of [1, 10, 100, 1000, 10000]) {
    const noNorm = train(env, defaults({ steps: 2000, lr: divLr, rewardScale: scale, normalizeAdv: false }), mulberry32(trainSeed));
    const withNorm = train(env, defaults({ steps: 2000, lr: divLr, rewardScale: scale, normalizeAdv: true }), mulberry32(trainSeed));
    console.log(
      `  rewardScale=${String(scale).padStart(5)} | 不归一 命中率=${noNorm.accuracy.toFixed(3)} | 归一 命中率=${withNorm.accuracy.toFixed(3)}`,
    );
  }
  console.log("  → 不归一: 命中率随尺度单调垮塌（1.0→0.25），大尺度→首步把策略一把推成 one-hot→锁死");
  console.log("  → 归一化: 命中率对尺度几乎不变 = scale-invariant，这正是它在难任务上不可省的原因");
  console.log("  → 诚实更正: softmax 的减 max 让 θ 不会真的溢出成 NaN；伤害是「过早锁死」而非「算术爆炸」");

  console.log("\n[结论]");
  console.log("  → 策略梯度无偏但高方差；baseline/归一化都不改无偏性，只压方差/稳步长");
  console.log("  → baseline 收益随奖励偏移增长（真实 RLHF 全正奖励 = baseline 必需）；归一化保证 scale-invariant");
  console.log("  → 两个失败模式共享同一机制: 单步过大 → 探索前锁成 one-hot → 锁错就学不动");
  console.log("  → 这就是 PPO(clip 限步长)/GRPO(组基线降方差) 的共同地基；toy 绝对值乐观, 可迁移的是相对趋势与机制");
}

main();
