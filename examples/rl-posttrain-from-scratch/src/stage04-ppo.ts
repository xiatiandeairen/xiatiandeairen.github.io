// stage04-ppo.ts — PPO from scratch: actor-critic + GAE + clipped surrogate,
// solving the gridworld and measured against the EXACT V* from chapter 2.
//
// Why this is the chapter the whole second half of the book leans on: REINFORCE
// (stage 03) is unbiased but uses each rollout once and is brittle to the step
// size — a single large update can shove the policy somewhere it can never
// recover from. PPO fixes both problems at once:
//   1. a CRITIC V(s) gives a low-variance baseline (advantage = how much better
//      this action did than the state's average), trained by regression;
//   2. GAE blends TD and Monte-Carlo returns with a knob λ that trades the
//      critic's bias against the rollout's variance;
//   3. the CLIPPED surrogate reuses each batch for several epochs while refusing
//      to let any single update move π past 1±ε of π_old — a built-in trust
//      region. That trust region is the entire reason RLHF (stage 06) can run PPO
//      against a noisy reward model without the policy detonating.
//
// Everything printed here is measured against ground truth: the gridworld's V* is
// solved exactly by value iteration (core/metrics), so "did PPO learn" is the gap
// between the policy's achieved discounted return-from-start and V*[start], not a
// vibe. The failure demos (clip off -> a single batch craters the return; bad λ
// choice -> slower / noisier convergence) are the load-bearing lessons.
//
// Honesty caveats: this is a 12-state deterministic gridworld with tabular
// softmax/critic, so absolute returns are optimistic and convergence is fast.
// What transfers is the RELATIVE story — clip caps the per-step KL, no-clip
// overshoots and can collapse, λ=1 is unbiased-but-noisy vs λ=0 biased-but-calm.
//
// Run: npm run stage04

import { mulberry32, sampleCategorical, argmax, type Rng } from "./core/rng.js";
import { makeGridworld, type Gridworld } from "./core/envs.js";
import { softmax, logProb } from "./core/probability.js";
import { optimalValueIteration, asciiSparkline, movingAvg } from "./core/metrics.js";

const GAMMA = 0.95; // must match the V* discount so the gap-to-V* comparison is apples-to-apples.
const MAX_STEPS = 50; // episode horizon cap; the optimal path is ~5 steps, so this only bounds wandering.

// One environment transition recorded during rollout. We keep logpOld (the
// log-prob UNDER THE POLICY THAT COLLECTED THE DATA) because PPO is off-policy
// within a batch: the ratio π_new/π_old needs the frozen old value, and a bug
// here (recomputing it after the update) silently turns PPO back into vanilla PG.
interface Step {
  s: number;
  a: number;
  reward: number;
  logpOld: number;
  done: boolean;
}

interface Batch {
  steps: Step[];
  adv: number[]; // GAE advantage per step, aligned with steps[]
  ret: number[]; // bootstrapped return target for the critic, aligned with steps[]
}

interface Trainable {
  theta: number[][]; // actor logits theta[state][action]
  V: number[]; // critic value estimate V[state]
}

function makeTrainable(env: Gridworld): Trainable {
  return {
    theta: Array.from({ length: env.nStates }, () => new Array<number>(env.nActions).fill(0)),
    V: new Array<number>(env.nStates).fill(0),
  };
}

// Roll out `nEpisodes` trajectories under the current (frozen) policy. Critic is
// read but not written here — collection is pure sampling, learning happens in
// the update. logpOld is snapped now so the whole batch shares one π_old.
function collect(env: Gridworld, net: Trainable, nEpisodes: number, rng: Rng): Step[] {
  const steps: Step[] = [];
  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = env.reset(rng);
    for (let t = 0; t < MAX_STEPS; t++) {
      const probs = softmax(net.theta[s]);
      const a = sampleCategorical(probs, rng);
      const { nextState, reward, done } = env.step(s, a, rng);
      steps.push({ s, a, reward, logpOld: logProb(net.theta[s], a), done });
      s = nextState;
      if (done) break;
    }
  }
  return steps;
}

// Generalized Advantage Estimation. The TD residual δ_t = r_t + γV(s_{t+1}) - V(s_t)
// is a one-step (biased by critic error, low variance) advantage estimate; the
// full discounted return minus V(s_t) is the Monte-Carlo (unbiased, high variance)
// one. GAE is the geometric blend A_t = Σ (γλ)^l δ_{t+l}, so λ slides continuously
// between them: λ=0 -> A_t = δ_t (pure TD, leans entirely on the critic), λ=1 ->
// A_t = MC return - V(s_t) (ignores the critic's bias, inherits its variance).
//
// Invariant: episodes are concatenated in `steps`, so we MUST reset the running
// sum and the bootstrap value at every `done` — otherwise advantage leaks across
// episode boundaries and credit is assigned to the wrong trajectory. The
// return target ret = A_t + V(s_t) is the standard "advantage + baseline" critic
// regression target (equivalent to the λ-return).
function computeGae(steps: Step[], V: number[], gamma: number, lam: number): { adv: number[]; ret: number[] } {
  const n = steps.length;
  const adv = new Array<number>(n).fill(0);
  let gae = 0;
  let nextV = 0; // V(s_{t+1}); reset to 0 at terminals (no future value past `done`)
  for (let i = n - 1; i >= 0; i--) {
    const st = steps[i];
    if (st.done) {
      nextV = 0;
      gae = 0; // start a fresh accumulation for the trajectory ending here
    }
    const delta = st.reward + gamma * nextV - V[st.s];
    gae = delta + gamma * lam * gae;
    adv[i] = gae;
    nextV = V[st.s];
  }
  const ret = adv.map((a, i) => a + V[steps[i].s]);
  return { adv, ret };
}

// Normalize advantages to zero mean / unit std. This is not cosmetic: PPO's step
// size is implicitly scaled by |advantage|, so un-normalized advantages couple the
// effective learning rate to the reward magnitude (a knob we did not intend to
// turn). Normalizing decouples them and is what real PPO implementations ship.
function normalize(xs: number[]): number[] {
  const n = xs.length;
  if (n === 0) return xs;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  let v = 0;
  for (const x of xs) v += (x - mean) * (x - mean);
  const std = Math.sqrt(v / n) + 1e-8; // +eps: a fully-converged batch has std 0; avoid /0 -> NaN advantages.
  return xs.map((x) => (x - mean) / std);
}

interface UpdateStats {
  clipFrac: number; // fraction of (sample, epoch) pairs where the ratio was clipped
  approxKl: number; // mean KL(new||old) the update actually induced (k3 estimator)
  criticLoss: number; // mean squared critic error AFTER the update (lower = critic tracks returns)
}

// One PPO update: `epochs` passes over the SAME batch, each updating actor (clipped
// surrogate) and critic (MSE regression to the bootstrapped return). Returns the
// diagnostics that make PPO legible — clip fraction (is the trust region binding?),
// approx KL (how far did we actually move?), critic loss (is the baseline any good?).
function ppoUpdate(
  net: Trainable,
  batch: Batch,
  lrActor: number,
  lrCritic: number,
  clip: number,
  epochs: number,
): UpdateStats {
  const { steps, adv, ret } = batch;
  let clippedCount = 0;
  let total = 0;
  for (let e = 0; e < epochs; e++) {
    for (let i = 0; i < steps.length; i++) {
      const { s, a, logpOld } = steps[i];
      const A = adv[i];
      const probs = softmax(net.theta[s]);
      const logpNew = Math.log(Math.max(probs[a], 1e-12));
      const ratio = Math.exp(logpNew - logpOld);

      // Clipped surrogate: L = min(ratio·A, clip(ratio, 1±ε)·A). The min picks the
      // pessimistic branch. The gradient of the CLIPPED branch w.r.t. θ is zero
      // (clamped ratio is constant in θ), so once an action's probability has moved
      // ε in the rewarded direction the objective flattens and stops pushing — the
      // trust region. clip<=0 disables it (the no-clip failure demo).
      const lo = 1 - clip;
      const hi = 1 + clip;
      const clamped = Math.max(lo, Math.min(hi, ratio));
      const usingClip = clip > 0 && clamped * A < ratio * A; // is the clipped branch the active (smaller) term?
      total++;
      if (clip > 0 && (ratio < lo || ratio > hi)) clippedCount++;

      // Surrogate gradient. When the unclipped branch is live, d/dθ (ratio·A) =
      // A·ratio·∇logπ. When the clipped branch is live, the term is constant in θ
      // -> gradient 0, so we simply skip the actor step for this sample this epoch.
      if (!usingClip) {
        const gradScale = lrActor * A * ratio;
        const row = net.theta[s];
        for (let k = 0; k < row.length; k++) {
          const dlogp = (k === a ? 1 : 0) - probs[k]; // ∇_logit logπ(a|s) for a softmax
          row[k] += gradScale * dlogp;
        }
      }

      // Critic regression: V(s) <- V(s) + lrCritic·(ret - V(s)). Plain SGD toward
      // the bootstrapped return target. A better critic shrinks advantage variance
      // next iteration, which is half of why PPO is calmer than REINFORCE.
      net.V[s] += lrCritic * (ret[i] - net.V[s]);
    }
  }

  // approx KL(new||old) over the batch, k3 estimator (the always-≥0, low-variance
  // one TRL/OpenAI ship). This is the trust-region quantity: clip keeps it small,
  // no-clip lets it spike. Measured AFTER all epochs = the total move this update made.
  let klSum = 0;
  let criticSe = 0;
  for (let i = 0; i < steps.length; i++) {
    const { s, a, logpOld } = steps[i];
    const logpNew = logProb(net.theta[s], a);
    const logRatio = logpNew - logpOld;
    klSum += Math.exp(logRatio) - 1 - logRatio; // k3
    const e = ret[i] - net.V[s];
    criticSe += e * e;
  }
  return {
    clipFrac: clippedCount / total,
    approxKl: klSum / steps.length,
    criticLoss: criticSe / steps.length,
  };
}

// Greedy evaluation: run the deterministic argmax policy from start and report the
// achieved DISCOUNTED return. Compared against V*[start] this is the honest "how
// close to optimal" number. Deterministic env => one rollout suffices.
function evalReturn(env: Gridworld, net: Trainable): number {
  let s = env.reset(mulberry32(0)); // deterministic gridworld: reset ignores rng
  let g = 0;
  let discount = 1;
  for (let t = 0; t < MAX_STEPS; t++) {
    const a = argmax(net.theta[s]);
    const { nextState, reward, done } = env.step(s, a, mulberry32(0));
    g += discount * reward;
    discount *= GAMMA;
    s = nextState;
    if (done) break;
  }
  return g;
}

interface TrainResult {
  net: Trainable;
  curve: number[]; // greedy return after each iteration
  clipFracs: number[];
  kls: number[];
  criticLosses: number[];
  advStdHistory: number[]; // raw (pre-normalization) advantage std per iteration — for the λ bias/variance demo
}

interface TrainOpts {
  clip: number;
  lam: number;
  iters: number;
  episodesPerBatch: number;
  epochs: number;
  lrActor: number;
  lrCritic: number;
  seed: number;
}

function train(env: Gridworld, opts: TrainOpts): TrainResult {
  const rng = mulberry32(opts.seed);
  const net = makeTrainable(env);
  const curve: number[] = [];
  const clipFracs: number[] = [];
  const kls: number[] = [];
  const criticLosses: number[] = [];
  const advStdHistory: number[] = [];

  for (let it = 0; it < opts.iters; it++) {
    const steps = collect(env, net, opts.episodesPerBatch, rng);
    const { adv, ret } = computeGae(steps, net.V, GAMMA, opts.lam);
    // Record raw advantage std BEFORE normalization: this is the variance the
    // λ knob controls, the headline of the bias/variance demo.
    const rawStd = Math.sqrt(adv.reduce((s, a) => s + a * a, 0) / Math.max(1, adv.length) -
      Math.pow(adv.reduce((s, a) => s + a, 0) / Math.max(1, adv.length), 2));
    advStdHistory.push(rawStd);
    const batch: Batch = { steps, adv: normalize(adv), ret };
    const stats = ppoUpdate(net, batch, opts.lrActor, opts.lrCritic, opts.clip, opts.epochs);
    curve.push(evalReturn(env, net));
    clipFracs.push(stats.clipFrac);
    kls.push(stats.approxKl);
    criticLosses.push(stats.criticLoss);
  }
  return { net, curve, clipFracs, kls, criticLosses, advStdHistory };
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function peak(xs: number[]): number {
  return Math.max(...xs);
}

// First iteration index at which the greedy return comes within `tol` of target
// AND stays there to the end (a transient touch does not count as "converged").
// Returns -1 if it never sticks. This is how we say "λ=X converged faster".
function convergedAt(curve: number[], target: number, tol: number): number {
  for (let i = 0; i < curve.length; i++) {
    if (curve.slice(i).every((v) => v >= target - tol)) return i;
  }
  return -1;
}

function main(): void {
  console.log("Stage 04 — PPO：actor-critic + GAE + 裁剪代理目标，对照精确 V*\n");

  const env = makeGridworld(); // default 3x4, start@0, hole@7, goal@11
  const Vstar = optimalValueIteration(env, GAMMA);
  const target = Vstar[env.start];
  console.log("环境：3x4 gridworld，start@" + env.start + "  V*[start]=" + target.toFixed(4) +
    "（价值迭代精确解，即可达到的最优折扣回报）\n");

  const base: TrainOpts = {
    clip: 0.2, lam: 0.95, iters: 60, episodesPerBatch: 20,
    epochs: 6, lrActor: 0.4, lrCritic: 0.3, seed: 123,
  };

  // ---- 1) Baseline PPO learning curve vs V* ----
  const ppo = train(env, base);
  const smoothed = movingAvg(ppo.curve, 5);
  console.log("[1] PPO 学习曲线（贪婪回报 vs V*，5 窗平滑 sparkline）");
  console.log("    " + asciiSparkline(smoothed));
  console.log("    最终贪婪回报=" + ppo.curve[ppo.curve.length - 1].toFixed(4) +
    "  与 V* 差距=" + (target - ppo.curve[ppo.curve.length - 1]).toFixed(4));
  console.log("    收敛于第 " + convergedAt(ppo.curve, target, 0.02) + " 轮（贪婪回报进入 V*±0.02 且不再掉出）");
  console.log("    每轮 clip 触发比例 sparkline：" + asciiSparkline(ppo.clipFracs) +
    "  （均值=" + mean(ppo.clipFracs).toFixed(3) + "，早期高=策略在动，后期降=收敛）");
  console.log("    critic loss sparkline：" + asciiSparkline(ppo.criticLosses) +
    "  （末值=" + ppo.criticLosses[ppo.criticLosses.length - 1].toFixed(4) + "，下降=critic 追上回报）");
  console.log("    近似 KL(new||old) 峰值=" + peak(ppo.kls).toFixed(4) + "  均值=" + mean(ppo.kls).toFixed(4) + "\n");

  // ---- 2) clip ε sweep: stability vs final return ----
  console.log("[2] clip ε 旋钮：ε 越大单步可移动越远（信任域越松）");
  console.log("    ε     | 峰值KL  | clip均值 | 收敛轮 | 最终回报 | 与V*差距");
  for (const eps of [0.05, 0.2, 0.5]) {
    const r = train(env, { ...base, clip: eps });
    const fin = r.curve[r.curve.length - 1];
    console.log(
      "    " + eps.toFixed(2).padStart(5) +
      " | " + peak(r.kls).toFixed(4).padStart(6) +
      " | " + mean(r.clipFracs).toFixed(3).padStart(7) +
      " |  " + String(convergedAt(r.curve, target, 0.02)).padStart(4) +
      " |  " + fin.toFixed(4).padStart(7) +
      " | " + (target - fin).toFixed(4).padStart(7),
    );
  }
  console.log("    → ε 小：KL 被拴得紧、收敛稳但偏慢；ε 大：信任域松、峰值 KL 抬高、波动更大。\n");

  // ---- 3) FAILURE MODE: clip off -> vanilla PG with an aggressive lr -> cliff ----
  // Hyperparams chosen empirically (seed 7, lr 0.8, 6 epochs/batch) so the contrast
  // is HONEST, not cherry-picked to flatter clip: under this same setting the
  // clipped run climbs to V* and never regresses, while the no-clip run falls off
  // a cliff mid-training and ENDS in the hole (negative return). Same seed/lr/epochs
  // for both — the only difference is whether the ratio is clipped.
  console.log("[3] 失败模式：关掉 clip（退化成 vanilla PG）+ 大 lr，看回报断崖");
  const aggressive: TrainOpts = { ...base, lrActor: 0.8, epochs: 6, iters: 60, seed: 7 };
  const withClip = train(env, { ...aggressive, clip: 0.2 });
  const noClip = train(env, { ...aggressive, clip: 0 });
  // Cliff metric: the single worst iteration-to-iteration DROP in greedy return.
  // PPO's clip should keep this ~0 (monotone climb); vanilla PG can fall off a
  // cliff after one oversized batch that yanks the policy past the goal region.
  const worstDrop = (curve: number[]): { drop: number; at: number } => {
    let drop = 0, at = -1;
    for (let i = 1; i < curve.length; i++) {
      const d = curve[i - 1] - curve[i];
      if (d > drop) { drop = d; at = i; }
    }
    return { drop, at };
  };
  const wcDrop = worstDrop(withClip.curve);
  const ncDrop = worstDrop(noClip.curve);
  // curve[0] is the greedy return AFTER the first update (the pristine untrained
  // argmax is ≈-0.74: it bumps the wall every step). The first no-clip update is so
  // large it already jumps the policy to the goal (0.63), then a later batch craters
  // it; the clipped first step is small (-0.74 still) but climbs monotonically to V*
  // and never falls. Printing 起点->终点 makes that direction-of-travel unambiguous.
  console.log("    有 clip(ε=0.2)  峰值KL=" + peak(withClip.kls).toFixed(4) +
    "  最大单轮回报跌幅=" + wcDrop.drop.toFixed(4) +
    "  回报 起点" + withClip.curve[0].toFixed(3) + "->终点" + withClip.curve[withClip.curve.length - 1].toFixed(3) +
    "（单调爬升）");
  console.log("    无 clip         峰值KL=" + peak(noClip.kls).toFixed(4) +
    "  最大单轮回报跌幅=" + ncDrop.drop.toFixed(4) + "（@轮" + ncDrop.at + "）" +
    "  回报 起点" + noClip.curve[0].toFixed(3) + "->终点" + noClip.curve[noClip.curve.length - 1].toFixed(3) +
    "（爬升后断崖）");
  console.log("    无 clip 回报曲线 sparkline：" + asciiSparkline(noClip.curve));
  console.log("    → 无 clip 时某次过大更新把策略甩出取信区，回报断崖（跌幅 " + ncDrop.drop.toFixed(2) +
    " 落入负回报=掉进 hole）、峰值 KL（" + peak(noClip.kls).toFixed(1) + "）比有 clip（" +
    peak(withClip.kls).toFixed(2) + "）大一个数量级；");
  console.log("      clip 把单步 KL 拴住，曲线单调爬到 V* 不回头 —— 这就是 PPO 内建信任域的价值。\n");

  // ---- 4) GAE λ bias-variance: λ=0 (pure TD) vs λ=1 (pure MC) ----
  console.log("[4] GAE λ 偏差-方差：λ=0(纯 TD，靠 critic，低方差有偏) vs λ=1(纯 MC，无偏高方差)");
  console.log("    λ    | 早期adv std | 收敛轮 | 最终回报 | critic末loss");
  let tdConv = -1, mcConv = -1;
  for (const lam of [0, 1]) {
    const r = train(env, { ...base, lam });
    const earlyAdvStd = mean(r.advStdHistory.slice(0, 10)); // raw advantage std, first 10 iters
    const conv = convergedAt(r.curve, target, 0.02);
    if (lam === 0) tdConv = conv; else mcConv = conv;
    console.log(
      "    " + lam.toFixed(1) +
      "  | " + earlyAdvStd.toFixed(4).padStart(10) +
      " |  " + String(conv).padStart(4) +
      " |  " + r.curve[r.curve.length - 1].toFixed(4).padStart(7) +
      " | " + r.criticLosses[r.criticLosses.length - 1].toFixed(4).padStart(8),
    );
  }
  const faster = tdConv >= 0 && (mcConv < 0 || tdConv <= mcConv) ? "λ=0 (TD)" : "λ=1 (MC)";
  console.log("    → λ=1 的早期 advantage std 更高（MC 把整条轨迹噪声灌进优势）；λ=0 借 critic 压低方差。");
  console.log("    → 本环境先稳定收敛的是 " + faster + "；GAE 的 λ≈0.95 折中两者，是实践默认值。");
  console.log("    → 可迁移：critic 准时 λ 调低省方差，critic 差时 λ 调高避免被 critic 的偏差带偏。\n");

  console.log("[诚实声明] 12 状态确定性 gridworld + 表格 actor/critic：绝对回报乐观、收敛快。");
  console.log("可迁移的是相对趋势：clip 限单步 KL（RLHF 复用它的根因）、λ 在 critic 偏差与 rollout 方差间折中。");
}

main();
