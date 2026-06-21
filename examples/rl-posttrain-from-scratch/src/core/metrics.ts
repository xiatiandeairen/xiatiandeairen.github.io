// core/metrics.ts — the "honest numbers" toolkit.
//
// Rule for the whole book: every reward / win-rate / KL / regret / accuracy a
// stage prints must be COMPUTED here from real rollouts, never asserted or
// hard-coded. This file is where the measurement primitives live so that no
// stage is tempted to fudge a curve. Each metric is paired with the ground truth
// the relevant env/preference-world exposes (best arm, V*, trueRewardFn), which
// is the only reason these numbers mean anything in a toy setting.

import type { Bandit, Gridworld } from "./envs.js";
import { klCategorical } from "./probability.js";

// runningMeanReward: online mean with O(1) memory (Welford-style increment).
// Why online and not "sum then divide": RL training loops are long; we report a
// running average per step so the reader watches the curve form. Numerically the
// incremental form (mean += (x-mean)/n) avoids the precision loss of summing
// millions of rewards into one large accumulator.
export class RunningMean {
  private n = 0;
  private mean = 0;
  push(x: number): number {
    this.n++;
    this.mean += (x - this.mean) / this.n;
    return this.mean;
  }
  get value(): number {
    return this.mean;
  }
  get count(): number {
    return this.n;
  }
}

// regret: cumulative gap between the best arm's mean and the arm actually pulled.
// This is THE bandit metric: total reward is uninformative (depends on arm
// magnitudes), but regret measures the *cost of not knowing*. A perfect agent
// has regret growing sub-linearly (it stops paying once it's confident); a
// random agent's regret grows linearly. Plotting regret is how stage 01 proves
// ε-greedy/UCB actually explore. Uses TRUE means (the agent never sees these).
export function regret(bandit: Bandit, pulledArms: number[]): number[] {
  const best = bandit.trueMeans[bandit.optimalArm];
  const cumulative: number[] = [];
  let acc = 0;
  for (const arm of pulledArms) {
    acc += best - bandit.trueMeans[arm]; // per-pull regret is the mean gap, not the noisy sample
    cumulative.push(acc);
  }
  return cumulative;
}

// winRateVsRef: fraction of prompts where policyA's response beats the reference
// policy's response, AS SCORED BY the (hidden) true judge. This is the headline
// post-training metric — "is the tuned model actually better than where we
// started" — and the analogue of an LLM eval win-rate. Ties count as 0.5 (a draw
// is not a win), which keeps the metric honest when both policies output the same
// thing on easy prompts.
export function winRateVsRef<P>(
  prompts: P[],
  policyA: (p: P) => number, // returns A's score on the prompt under the true judge
  ref: (p: P) => number, // returns the reference's score
): number {
  let wins = 0;
  for (const p of prompts) {
    const a = policyA(p);
    const r = ref(p);
    if (a > r) wins += 1;
    else if (a === r) wins += 0.5;
  }
  return wins / prompts.length;
}

// klToRef: exact KL(current || reference) averaged over a set of states, using
// full categorical policies. This is the quantity RLHF's penalty term is trying
// to bound — the "leash" keeping the tuned policy from drifting into gibberish.
// Measuring it directly (we can, in the toy setting) lets the RLHF stage show the
// core trade-off as a NUMBER: stronger KL coefficient -> smaller drift here but
// slower reward gain. Returns mean KL across states; Infinity if any state's
// policy assigns mass where the reference assigned zero (catastrophic drift).
export function klToRef(
  current: number[][], // current[s] = policy distribution over actions in state s
  reference: number[][],
): number {
  let total = 0;
  for (let s = 0; s < current.length; s++) {
    total += klCategorical(current[s], reference[s]);
  }
  return total / current.length;
}

// rewardModelAccuracy: pairwise accuracy of a reward model on held-out preference
// pairs — fraction where the RM scores `chosen` above `rejected`. This is how we
// know an RM learned anything BEFORE using it to train a policy (a 50% RM is a
// coin flip and will teach the policy noise). The held-out split is essential:
// train-set accuracy can be inflated by memorization, and an RM that hacks the
// train set is the upstream cause of a policy that hacks the reward.
export function rewardModelAccuracy<Pair>(
  pairs: Pair[],
  score: (response: Pair extends { chosen: infer R; rejected: infer R } ? R : never) => number,
  getChosen: (p: Pair) => Pair extends { chosen: infer R } ? R : never,
  getRejected: (p: Pair) => Pair extends { rejected: infer R } ? R : never,
): number {
  let correct = 0;
  for (const p of pairs) {
    const sc = score(getChosen(p) as never);
    const sr = score(getRejected(p) as never);
    if (sc > sr) correct += 1;
    else if (sc === sr) correct += 0.5; // a tie is half-credit, not a free pass
  }
  return correct / pairs.length;
}

// pearson: linear correlation between learned reward and true reward. Answers
// "does the RM's scale track the truth," sensitive to outliers and nonlinearity.
// Used with spearman to diagnose RM quality: high Spearman + low Pearson means
// the RM got the ORDER right but the SHAPE wrong (often fine for ranking-based
// methods like DPO/GRPO, fatal for value-based RLHF that uses raw magnitudes).
export function pearson(x: number[], y: number[]): number {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let cov = 0, vx = 0, vy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    cov += dx * dy;
    vx += dx * dx;
    vy += dy * dy;
  }
  // Zero variance => correlation undefined. Return 0 (no linear relationship
  // detectable) rather than NaN, which would silently break a printed table.
  if (vx === 0 || vy === 0) return 0;
  return cov / Math.sqrt(vx * vy);
}

// spearman: rank correlation = Pearson on the ranks. This is the metric that
// matters most for reward models, because RLHF/DPO only ever use the RM to
// COMPARE responses, never its absolute scale. A high Spearman is the real
// "the RM understands the preference" signal.
export function spearman(x: number[], y: number[]): number {
  return pearson(toRanks(x), toRanks(y));
}

// Average-rank assignment so ties don't bias Spearman. (Fractional ranks are the
// standard correction; integer ranks on ties would distort correlation.)
function toRanks(arr: number[]): number[] {
  const idx = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(arr.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1].v === idx[i].v) j++;
    const avgRank = (i + j) / 2; // 0-based average rank over the tie block
    for (let k = i; k <= j; k++) ranks[idx[k].i] = avgRank;
    i = j + 1;
  }
  return ranks;
}

// movingAvg: smooth a noisy training curve for ASCII display. RL curves are so
// noisy that the raw per-step value hides the trend; a trailing window makes the
// learning visible. Window is trailing (not centered) so it can be used online
// and never peeks at the future — the same constraint the agent lives under.
export function movingAvg(series: number[], window: number): number[] {
  const out: number[] = [];
  let acc = 0;
  for (let i = 0; i < series.length; i++) {
    acc += series[i];
    if (i >= window) acc -= series[i - window];
    out.push(acc / Math.min(i + 1, window));
  }
  return out;
}

// timeIt: wall-clock a thunk and return its result plus elapsed ms. The book's
// honesty rule: any speedup/throughput we PRINT must be really measured, and
// estimates must be labeled "(est.)". This is the real-measurement path. Uses
// performance.now() (monotonic, sub-ms) not Date.now().
export function timeIt<T>(fn: () => T): { result: T; ms: number } {
  const t0 = performance.now();
  const result = fn();
  const ms = performance.now() - t0;
  return { result, ms };
}

// asciiSparkline: render a numeric series as one line of block characters, for
// curves in stdout (no plotting lib in a zero-dep repo). Linear-scaled to the
// series' own min/max, so it shows SHAPE, not absolute scale — pair it with a
// printed min/max. Empty/flat series degrade gracefully to a flat line.
const SPARK_CHARS = "▁▂▃▄▅▆▇█";
export function asciiSparkline(series: number[]): string {
  if (series.length === 0) return "";
  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = max - min;
  if (span === 0) return SPARK_CHARS[0].repeat(series.length); // flat: all lowest block
  return series
    .map((v) => {
      const t = (v - min) / span; // 0..1
      const idx = Math.min(SPARK_CHARS.length - 1, Math.floor(t * SPARK_CHARS.length));
      return SPARK_CHARS[idx];
    })
    .join("");
}

// optimalValueIteration: solve a gridworld's V* exactly by value iteration.
// This is the ground truth stage 02 (and any value-based stage) measures its
// learner against — the whole reason gridworld exists. Synchronous sweeps over
// the full known MDP (we have the transition model), iterating Bellman optimality
// until the max change < tol. NOT an RL method (it cheats by reading the model);
// it is the answer key. Returns V* per state.
export function optimalValueIteration(
  env: Gridworld,
  gamma = 0.95,
  tol = 1e-8,
): number[] {
  const V = new Array<number>(env.nStates).fill(0);
  for (let iter = 0; iter < 10000; iter++) {
    let delta = 0;
    for (let s = 0; s < env.nStates; s++) {
      if (env.terminal[s] || env.isWall(s)) continue; // terminals/walls have no decisions
      let best = -Infinity;
      for (let a = 0; a < env.nActions; a++) {
        // Expected value of action a under the wind model. With prob windProb the
        // intended next cell is shoved up one (if possible); otherwise it stays.
        // We enumerate this 2-outcome distribution explicitly so V* accounts for
        // stochasticity — a planner that ignored wind would be over-optimistic.
        const intended = env.intendedNext(s, a);
        const r = env.terminal[intended];
        const outcomes = windOutcomes(env, intended);
        let q = 0;
        for (const { state: ns, prob } of outcomes) {
          const reward = env.cellReward[ns] + env.stepReward;
          const future = env.terminal[ns] ? 0 : gamma * V[ns];
          q += prob * (reward + future);
        }
        void r;
        if (q > best) best = q;
      }
      const nv = best;
      delta = Math.max(delta, Math.abs(nv - V[s]));
      V[s] = nv;
    }
    if (delta < tol) break;
  }
  return V;
}

// Helper mirroring envs.step's wind logic for the planner. Kept here (not in
// envs) because it is part of the SOLVER's model of the env, used only to build
// the expectation; the env itself samples a single outcome at runtime.
function windOutcomes(
  env: Gridworld,
  intended: number,
): { state: number; prob: number }[] {
  if (env.windProb <= 0) return [{ state: intended, prob: 1 }];
  const width = env.width;
  const r = Math.floor(intended / width);
  const up = intended - width;
  const canPushUp = r > 0 && !env.isWall(up);
  const shoved = canPushUp ? up : intended;
  return [
    { state: shoved, prob: env.windProb },
    { state: intended, prob: 1 - env.windProb },
  ];
}
