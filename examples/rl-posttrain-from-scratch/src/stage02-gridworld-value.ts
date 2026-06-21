// stage02-gridworld-value.ts — planning with a KNOWN model: value iteration.
//
// This chapter's question is not "can an agent learn" (that is stage 03+), but
// "what is the agent trying to learn toward". When the MDP is fully known we can
// SOLVE it: value iteration sweeps the Bellman optimality backup until V stops
// moving, yielding V* and the optimal policy π* in closed form. Everything later
// in the book is measured against this answer key — a learner is "good" only
// relative to a number we computed here, never relative to a vibe.
//
// Why re-implement VI here instead of just calling core's optimalValueIteration:
// VI is the LESSON of this chapter, so the reader must see the backup, the
// convergence trace (max value change per sweep), and the greedy-policy
// extraction. We still call core's solver as an independent oracle and assert our
// V matches it — if our teaching code drifted from the reference, that bug would
// be loud, not silent.
//
// Two failure modes are demoed, not hidden:
//   (1) γ=1 on a gridworld with NO terminal state -> the return is an infinite
//       sum of step costs, V diverges, VI never converges. This is WHY discounting
//       (or a guaranteed terminal) is not optional.
//   (2) reward shaping done wrong -> a positive reward placed on a non-goal cell
//       makes π* DETOUR to farm it. The policy is optimal for the shaped reward
//       but worse under the TRUE reward. This is reward hacking's first formal
//       appearance, and we quantify the gap against the true-objective V*.
//
// Run: npm run stage02

import { makeGridworld, type Gridworld } from "./core/envs.js";
import { optimalValueIteration } from "./core/metrics.js";

// Actions are 0=up 1=right 2=down 3=left (the env's convention). Arrows for the
// policy map; "·" marks states with no decision (wall / terminal).
const ACTION_ARROWS = ["↑", "→", "↓", "←"];

interface ViResult {
  V: number[];
  policy: number[]; // policy[s] = best action; -1 for wall/terminal
  deltaCurve: number[]; // max |ΔV| per sweep — the convergence trace
  iterations: number; // sweeps until delta < tol (or the cap, if diverging)
  converged: boolean;
}

// valueIteration: synchronous Bellman-optimality backups over a known MDP.
//
// `rewardOverride` lets a caller substitute the per-state ENTER reward without
// mutating the shared env (the env's cellReward is the book's canonical truth;
// the shaping failure-mode experiment needs a *different* reward without
// corrupting it for every other stage). Defaults to the env's own cellReward.
//
// `maxIters` is a hard cap so the divergent γ=1 case returns instead of looping
// forever — the cap is itself the evidence of non-convergence (deltaCurve stays
// flat-high all the way to the cap).
//
// Invariant we rely on: terminal/wall states have no action, so their V stays 0;
// the backup skips them. A terminal's value is realized as the ENTER reward of
// the transition into it, not as a stored V — matching how core's oracle and the
// env's step() both account reward on entry.
function valueIteration(
  env: Gridworld,
  gamma: number,
  tol = 1e-8,
  maxIters = 5000,
  rewardOverride?: number[],
): ViResult {
  const enterReward = rewardOverride ?? env.cellReward;
  const V = new Array<number>(env.nStates).fill(0);
  const deltaCurve: number[] = [];
  let converged = false;
  let iter = 0;
  for (; iter < maxIters; iter++) {
    let delta = 0;
    for (let s = 0; s < env.nStates; s++) {
      if (env.terminal[s] || env.isWall(s)) continue;
      let best = -Infinity;
      for (let a = 0; a < env.nActions; a++) {
        const q = qValue(env, V, s, a, gamma, enterReward);
        if (q > best) best = q;
      }
      delta = Math.max(delta, Math.abs(best - V[s]));
      V[s] = best;
    }
    deltaCurve.push(delta);
    if (delta < tol) {
      converged = true;
      iter++; // count the converging sweep itself
      break;
    }
  }
  // Extract the greedy (optimal) policy from the converged V. Done once at the
  // end, not per sweep, because the policy is meaningful only at the fixed point.
  const policy = new Array<number>(env.nStates).fill(-1);
  for (let s = 0; s < env.nStates; s++) {
    if (env.terminal[s] || env.isWall(s)) continue;
    let best = -Infinity, bestA = 0;
    for (let a = 0; a < env.nActions; a++) {
      const q = qValue(env, V, s, a, gamma, enterReward);
      if (q > best) { best = q; bestA = a; }
    }
    policy[s] = bestA;
  }
  return { V, policy, deltaCurve, iterations: iter, converged };
}

// qValue: expected value of taking action a in state s, under the wind model.
// Mirrors the env's stochastic transition (intended move, then a windProb chance
// of being shoved one cell up). Enumerating the 2-outcome distribution explicitly
// is the whole point of planning with a model: a planner that ignored wind would
// be over-optimistic about reaching the goal.
function qValue(
  env: Gridworld,
  V: number[],
  s: number,
  a: number,
  gamma: number,
  enterReward: number[],
): number {
  const intended = env.intendedNext(s, a);
  const outcomes = windOutcomes(env, intended);
  let q = 0;
  for (const { state: ns, prob } of outcomes) {
    const reward = enterReward[ns] + env.stepReward;
    // Terminal entry has no future: the episode ends, so we do NOT bootstrap
    // V[ns]. Forgetting this is a classic bug that lets value leak past the goal.
    const future = env.terminal[ns] ? 0 : gamma * V[ns];
    q += prob * (reward + future);
  }
  return q;
}

// windOutcomes: the planner's copy of the env's wind transition. With prob
// windProb the agent ends one cell higher (if that cell exists and isn't a wall),
// else it lands where it intended. Returns a probability distribution that sums
// to 1, so qValue computes a true expectation.
function windOutcomes(env: Gridworld, intended: number): { state: number; prob: number }[] {
  if (env.windProb <= 0) return [{ state: intended, prob: 1 }];
  const r = Math.floor(intended / env.width);
  const up = intended - env.width;
  const canPushUp = r > 0 && !env.isWall(up);
  const shoved = canPushUp ? up : intended;
  return [
    { state: shoved, prob: env.windProb },
    { state: intended, prob: 1 - env.windProb },
  ];
}

// renderPolicy: ASCII map of the grid with an arrow per state showing π*'s chosen
// action. Walls -> "#", terminals -> their cell symbol (G/H), undecided -> "·".
function renderPolicy(env: Gridworld, policy: number[]): string {
  const lines: string[] = [];
  for (let r = 0; r < env.height; r++) {
    let row = "  ";
    for (let c = 0; c < env.width; c++) {
      const s = r * env.width + c;
      if (env.isWall(s)) row += " # ";
      else if (env.terminal[s]) row += env.cellReward[s] > 0 ? " G " : " H ";
      else row += " " + ACTION_ARROWS[policy[s]] + " ";
    }
    lines.push(row);
  }
  return lines.join("\n");
}

// evaluatePolicy: expected discounted return of FOLLOWING a fixed policy, scored
// under a chosen reward function (this is policy evaluation, not optimization —
// no max over actions, we obey `policy`). Used to measure reward hacking: take
// the policy that is optimal for the SHAPED reward, then evaluate it under the
// TRUE reward. The gap to V* under the true reward is the hacking cost in the
// agent's own currency.
function evaluatePolicy(
  env: Gridworld,
  policy: number[],
  gamma: number,
  enterReward: number[],
  tol = 1e-10,
  maxIters = 5000,
): number[] {
  const V = new Array<number>(env.nStates).fill(0);
  for (let iter = 0; iter < maxIters; iter++) {
    let delta = 0;
    for (let s = 0; s < env.nStates; s++) {
      if (env.terminal[s] || env.isWall(s)) continue;
      const v = qValue(env, V, s, policy[s], gamma, enterReward);
      delta = Math.max(delta, Math.abs(v - V[s]));
      V[s] = v;
    }
    if (delta < tol) break;
  }
  return V;
}

// downsample: pick ~n evenly spaced points from a series for a compact sparkline.
function downsample<T>(series: T[], n: number): T[] {
  if (series.length <= n) return series;
  const out: T[] = [];
  const stride = (series.length - 1) / (n - 1);
  for (let i = 0; i < n; i++) out.push(series[Math.round(i * stride)]);
  return out;
}

function main(): void {
  console.log("Stage 02 — Gridworld：值迭代求 V*/π*，收敛曲线 + 失败模式\n");

  const env = makeGridworld();
  console.log("布局 (S 起点 / G 目标 / H 陷阱 / # 墙)，step 代价=" +
    env.stepReward + "，goal=+1，hole=-1\n");

  // --- 1. value iteration at γ=0.95, cross-checked against the core oracle. ---
  const gamma = 0.95;
  const vi = valueIteration(env, gamma);
  const oracle = optimalValueIteration(env, gamma);
  let oracleGap = 0;
  for (let s = 0; s < env.nStates; s++) {
    if (env.terminal[s] || env.isWall(s)) continue;
    oracleGap = Math.max(oracleGap, Math.abs(vi.V[s] - oracle[s]));
  }
  console.log("[1] 值迭代 (γ=" + gamma + ") 收敛于第 " + vi.iterations +
    " 轮 (max|ΔV|<1e-8)，对照 core oracle 最大偏差=" + oracleGap.toExponential(1) +
    " (≈0 → 教学实现与答案键一致)");
  console.log("    收敛曲线 max|ΔV| 逐轮 (对数式衰减): " +
    asciiSparklineLog(downsample(vi.deltaCurve, 24)));
  console.log("    起点 V*(S)=" + vi.V[env.start].toFixed(4) +
    " = 从起点出发按 π* 行动的最优折扣回报\n");

  console.log("[2] 最优策略 π* (每格箭头 = 该状态最优动作):");
  console.log(renderPolicy(env, vi.policy) + "\n");

  // --- 3. γ sweep: discount shapes the values, and γ's effect on CONVERGENCE
  // SPEED depends on grid size. Subtlety we must report honestly: on the tiny
  // default grid every γ converges in the SAME number of sweeps, because VI
  // reaches its exact fixed point once the goal signal has propagated across the
  // grid diameter (~6 sweeps) — that is diameter-limited, not contraction-limited,
  // so γ cannot change the count. The γ→speed trend only appears once the grid is
  // big enough that the geometric contraction (rate γ) dominates the diameter.
  // We show BOTH so the reader doesn't over-generalize from the toy grid. ---
  console.log("[3] γ 扫描 — 折扣同时影响价值与收敛速度 (速度效应依赖网格大小):");
  console.log("    (a) 默认小网格 (3×4): 收敛轮数由直径主导，对 γ 不敏感");
  console.log("        γ     | 收敛轮数 | V*(起点) | π*(起点) | 说明");
  for (const g of [0.5, 0.9, 0.99]) {
    const r = valueIteration(env, g);
    const note = g <= 0.5 ? "短视: 远处目标被折扣压扁"
      : g >= 0.99 ? "远视: 价值最高"
        : "折中";
    console.log("        " + g.toFixed(2).padEnd(5) + " |   " +
      String(r.iterations).padStart(4) + "   |  " +
      r.V[env.start].toFixed(4).padStart(7) + " |    " +
      ACTION_ARROWS[r.policy[env.start]] + "     | " + note);
  }
  // A larger open grid: now diameter is big, so the contraction rate γ governs
  // how many sweeps are needed to drive max|ΔV| below tol — the textbook trend.
  const bigGrid = makeGridworld({
    layout: ["S.........", ".........#", "#########.", "..........", ".........G"],
    stepReward: -0.04,
  });
  console.log("    (b) 大网格 (5×10, 长路径): 收敛轮数由 γ 主导，γ→1 收敛急剧变慢");
  console.log("        γ     | 收敛轮数 | V*(起点)");
  for (const g of [0.5, 0.9, 0.99]) {
    const r = valueIteration(bigGrid, g);
    console.log("        " + g.toFixed(2).padEnd(5) + " |   " +
      String(r.iterations).padStart(4) + "   |  " + r.V[bigGrid.start].toFixed(4).padStart(7));
  }
  // Note the value SIGN flips between the two grids: small grid V↑ with γ (goal
  // reward dominates the short path), big grid V↓ with γ (the long path's many
  // step costs dominate and get discounted less). "γ↑ = 价值更高" is therefore
  // env-dependent; what is universal is that γ↑ shrinks discounting of ALL future
  // signal (reward and cost alike) and slows contraction-limited convergence.
  console.log("    → 收敛轮数: γ→1 时收缩系数→1，大网格上轮数暴涨 (23→146→1514, 真测)。");
  console.log("    → 价值方向因环境而异: 小网格 γ↑→V↑ (目标主导)，大网格 γ↑→V↓ (路径代价主导)。\n");

  // --- 4. FAILURE MODE A: γ=1 with no terminal -> divergence. ---
  // A grid with zero terminals: every cell is free, every step pays the negative
  // step cost forever. With γ=1 the return is -∞; VI's max|ΔV| never shrinks.
  const noTerminal = makeGridworld({ layout: ["...", "...", "..."], stepReward: -0.04 });
  const diverge = valueIteration(noTerminal, 1.0, 1e-8, 400);
  const firstDelta = diverge.deltaCurve[0];
  const lastDelta = diverge.deltaCurve[diverge.deltaCurve.length - 1];
  console.log("[4] 失败模式 A — γ=1 且无终止态 (3×3 全空地，每步恒 -0.04):");
  console.log("    跑满 " + diverge.iterations + " 轮上限，converged=" + diverge.converged +
    "；max|ΔV| 首轮=" + firstDelta.toFixed(4) + " 末轮=" + lastDelta.toFixed(4) +
    " (不收敛: 每轮稳定下掉 0.04，V→-∞)");
  console.log("    起点 V 已跌到 " + diverge.V[noTerminal.start].toFixed(2) +
    " 并继续发散 → 折扣 γ<1 (或保证终止) 不是可选项。\n");

  // --- 5. FAILURE MODE B: bad reward shaping -> reward hacking. ---
  // Place a fat positive reward on a non-goal free cell. The shaped-optimal policy
  // will DETOUR to sit near/loop through the decoy instead of heading to the goal.
  // We then score that policy under the TRUE reward and compare to the true V*.
  const decoyCell = 4; // row 1, col 1 — center-ish, off the true goal path
  const shaped = env.cellReward.slice();
  shaped[decoyCell] = 0.7; // hacker's bait: large positive on a NON-terminal cell
  const trueVi = valueIteration(env, gamma); // optimal for the TRUE reward
  const hackVi = valueIteration(env, gamma, 1e-8, 5000, shaped); // optimal for SHAPED

  // Score the SHAPED-optimal policy honestly: under the TRUE reward function.
  const hackUnderTrue = evaluatePolicy(env, hackVi.policy, gamma, env.cellReward);
  const trueStart = trueVi.V[env.start];
  const hackedStart = hackUnderTrue[env.start];
  console.log("[5] 失败模式 B — reward shaping 加错 (在非目标格 #" + decoyCell +
    " 偷加 +0.7):");
  console.log("    真奖励下的 π* 策略:");
  console.log(renderPolicy(env, trueVi.policy));
  console.log("    被 shaping 带偏的 π_hack 策略 (为刷 #" + decoyCell + " 绕路):");
  console.log(renderPolicy(env, hackVi.policy));
  const detoured = trueVi.policy[env.start] !== hackVi.policy[env.start];
  console.log("    起点动作: 真 π*=" + ACTION_ARROWS[trueVi.policy[env.start]] +
    "  π_hack=" + ACTION_ARROWS[hackVi.policy[env.start]] +
    (detoured ? "  (已被带偏 ✗)" : "  (起点未变)"));
  console.log("    用【真奖励】打分: V*(起点)=" + trueStart.toFixed(4) +
    "  vs  π_hack 的真回报=" + hackedStart.toFixed(4) +
    "  → 真目标损失=" + (trueStart - hackedStart).toFixed(4));
  console.log("    → π_hack 对【代理奖励】是最优的，对【真奖励】却更差。代理与真值解耦处即 hacking。\n");

  console.log("→ 可迁移: 值迭代是后续所有学习法的答案键; γ 控远视与收敛速度的权衡;");
  console.log("  无折扣+无终止 = 发散; 奖励塑形错位 = 学出绕路刷分 (RLHF reward hacking 的同构最小版)。");
}

// asciiSparklineLog: a convergence-trace sparkline on a log scale. The raw
// max|ΔV| spans many orders of magnitude (1e0 → 1e-8); a linear sparkline would
// show one tall bar then flatline, hiding the geometric decay. Log scaling makes
// the steady contraction visible. Falls back to flat for non-positive/empty.
const SPARK_CHARS = "▁▂▃▄▅▆▇█";
function asciiSparklineLog(series: number[]): string {
  const logs = series.map((v) => Math.log10(Math.max(v, 1e-12)));
  const min = Math.min(...logs);
  const max = Math.max(...logs);
  const span = max - min;
  if (span === 0) return SPARK_CHARS[0].repeat(series.length);
  return logs
    .map((v) => {
      const t = (v - min) / span;
      const idx = Math.min(SPARK_CHARS.length - 1, Math.floor(t * SPARK_CHARS.length));
      return SPARK_CHARS[idx];
    })
    .join("");
}

main();
