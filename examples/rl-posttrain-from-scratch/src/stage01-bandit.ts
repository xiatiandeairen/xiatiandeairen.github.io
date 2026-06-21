// stage01-bandit.ts — exploration vs exploitation, and regret as the ruler.
//
// Why bandits open an RL book: a k-armed bandit strips RL down to a SINGLE
// decision with no future state — so credit assignment, bootstrapping, and
// value propagation all vanish, and the only thing left to get wrong is the
// explore/exploit trade-off. Whatever intuition you build here ("never knowing
// is a cost you pay forever unless you explore") reappears unchanged in a 70B
// RLHF run. Because the arm means are fixed and known to US (never to the agent),
// REGRET — the cumulative gap to the best arm — is exactly computable, which is
// the only reason any number below is honest rather than a vibe.
//
// This stage runs four classic strategies on ONE shared 10-arm Gaussian bandit
// and measures, across ≥20 seeds:
//   - mean cumulative regret (lower = explored efficiently),
//   - optimal-arm selection rate over the last window (did it actually find it),
//   - run-to-run std of final regret (the whole reason we average over seeds).
//
// Two failure modes are demoed, not hidden:
//   (F1) ε=0 pure greedy locks onto an early-lucky suboptimal arm; regret grows
//        LINEARLY forever.
//   (F2) optimistic-init set too LOW degenerates into that same greedy lock-in —
//        the "exploration" was an illusion created entirely by the initial value.
//
// Run: npx tsx src/stage01-bandit.ts   (or: npm run stage01)

import { mulberry32, gaussian, argmax, type Rng } from "./core/rng.js";
import { makeBandit, type Bandit } from "./core/envs.js";
import { regret, asciiSparkline } from "./core/metrics.js";

// One run's result. We keep the per-pull arm list (regret is derived from it
// against the hidden true means) and the per-step "was the optimal arm chosen"
// flags, so optimal-selection-rate is a measured frequency, not an estimate.
interface RunResult {
  regretCurve: number[]; // cumulative regret per step, length = steps
  optimalFlags: boolean[]; // optimalFlags[t] = (arm chosen at t === bandit.optimalArm)
}

// A bandit agent: given the env, a budget, and a private rng, produce one run.
// The rng is the ONLY randomness source — same seed in, same run out. Agents
// never read bandit.trueMeans / bandit.optimalArm; those are the answer key the
// metrics use, and peeking would make the lesson a lie.
type Agent = (bandit: Bandit, steps: number, rng: Rng) => RunResult;

// Shared bookkeeping: incremental sample-average Q and pull counts. Q[a] is the
// agent's running estimate of arm a's mean; counts[a] is how many times it was
// pulled. The incremental update Q += (r - Q)/n is the O(1)-memory form (same as
// metrics.RunningMean) — important because real RL runs are too long to store
// every reward.
interface Estimates {
  Q: number[];
  counts: number[];
}

function makeEstimates(nArms: number, initialValue: number): Estimates {
  return { Q: new Array<number>(nArms).fill(initialValue), counts: new Array<number>(nArms).fill(0) };
}

function recordPull(est: Estimates, arm: number, reward: number): void {
  est.counts[arm]++;
  est.Q[arm] += (reward - est.Q[arm]) / est.counts[arm];
}

// Drives one episode given a per-step arm-selection rule. Centralizes the
// step/record/regret bookkeeping so each strategy below is JUST its selection
// logic — the part the chapter is actually about.
function runWith(
  bandit: Bandit,
  steps: number,
  rng: Rng,
  est: Estimates,
  selectArm: (t: number) => number,
): RunResult {
  const pulls: number[] = [];
  const optimalFlags: boolean[] = [];
  for (let t = 0; t < steps; t++) {
    const arm = selectArm(t);
    const { reward } = bandit.step(0, arm, rng);
    recordPull(est, arm, reward);
    pulls.push(arm);
    optimalFlags.push(arm === bandit.optimalArm);
  }
  return { regretCurve: regret(bandit, pulls), optimalFlags };
}

// ---------------------------------------------------------------------------
// Strategy 1: ε-greedy. Exploit the current-best arm, but with probability ε
// pick a uniformly random arm. The constant ε means it NEVER stops exploring,
// so its regret grows linearly at rate ~ε·(mean gap) even after it has found the
// best arm — the canonical "fixed exploration is a permanent tax" result.
// ε=0 is the F1 failure mode (see main): pure greedy, lock-in risk.
// ---------------------------------------------------------------------------
function epsilonGreedy(epsilon: number): Agent {
  return (bandit, steps, rng) => {
    const est = makeEstimates(bandit.nActions, 0);
    return runWith(bandit, steps, rng, est, () => {
      // Two rng draws on the explore branch (coin + arm), one on exploit. The
      // count is fixed per branch, which is what keeps seeded runs reproducible.
      if (rng() < epsilon) return Math.floor(rng() * bandit.nActions);
      return argmax(est.Q); // first-max tie-break: fine here, ties are ~never exact
    });
  };
}

// ---------------------------------------------------------------------------
// Strategy 2: optimistic initial values. Pure greedy (ε=0), but every Q starts
// at an OPTIMISTICALLY high value. Because greedy always picks the highest Q, and
// real rewards pull each pulled arm's Q DOWN toward its true mean, every arm
// looks "best" until it's been tried — so the agent sweeps all arms early, then
// settles. Exploration emerges with zero randomness in the policy itself.
//
// The catch (F2, demoed in main): the whole effect lives in the initial value.
// Set it BELOW the true means and the first arm whose noisy sample lands high
// stays on top forever — it silently becomes plain greedy with lock-in.
// ---------------------------------------------------------------------------
function optimisticGreedy(initialValue: number): Agent {
  return (bandit, steps, rng) => {
    const est = makeEstimates(bandit.nActions, initialValue);
    return runWith(bandit, steps, rng, est, () => argmax(est.Q));
  };
}

// ---------------------------------------------------------------------------
// Strategy 3: UCB1 (upper confidence bound). Pick the arm maximizing
//   Q[a] + c·sqrt(ln t / N[a]).
// The bonus term is large for under-pulled arms (small N) and shrinks as an arm
// is sampled, so UCB explores by OPTIMISM-UNDER-UNCERTAINTY rather than random
// coin flips: it deterministically tries whatever it is least sure about. This
// is why UCB achieves logarithmic (sub-linear) regret — it stops exploring an
// arm once it's confident, unlike fixed-ε.
//
// NOTE: UCB1's regret bound assumes bounded rewards in [0,1]; our payoffs are
// unbounded Gaussian, so this is UCB used as a heuristic (c tunes the bonus
// scale to the reward magnitude). The qualitative lesson — sub-linear regret via
// uncertainty-directed exploration — holds; the constant is not the textbook one.
// ---------------------------------------------------------------------------
function ucb1(c: number): Agent {
  return (bandit, steps, rng) => {
    const est = makeEstimates(bandit.nActions, 0);
    return runWith(bandit, steps, rng, est, (t) => {
      // Seed phase: pull each arm once so ln(t)/N is defined (N=0 -> +Infinity
      // bonus would already force this, but doing it explicitly is clearer and
      // matches the standard statement of UCB1).
      if (t < bandit.nActions) return t;
      const tt = t + 1; // 1-indexed time for ln; ln(1)=0 would zero the first bonus
      let best = 0;
      let bestScore = -Infinity;
      for (let a = 0; a < bandit.nActions; a++) {
        const bonus = c * Math.sqrt(Math.log(tt) / est.counts[a]);
        const score = est.Q[a] + bonus;
        if (score > bestScore) {
          bestScore = score;
          best = a;
        }
      }
      return best;
    });
  };
}

// ---------------------------------------------------------------------------
// Strategy 4: Thompson sampling. Maintain a posterior over each arm's mean,
// SAMPLE one mean from each arm's posterior, and pull the arm with the highest
// sample. Arms we're unsure about have wide posteriors, so they occasionally
// sample high and get tried — exploration as a natural consequence of Bayesian
// uncertainty, no bonus term to tune.
//
// Conjugacy: rewards are Gaussian with KNOWN noise variance σ²=1 (that's the
// bandit's contract). With a Normal prior on the mean, the posterior after n
// pulls with sample mean Q is Normal too:
//   posterior mean     = (n·Q) / (n + 1/priorVar)      [precision-weighted]
//   posterior variance = 1 / (n + 1/priorVar)
// We use a broad prior (priorVar large) so the data dominates quickly. Unpulled
// arms (n=0) keep the wide prior, guaranteeing they get sampled.
// ---------------------------------------------------------------------------
function thompson(priorVar: number): Agent {
  return (bandit, steps, rng) => {
    const est = makeEstimates(bandit.nActions, 0);
    const priorPrecision = 1 / priorVar;
    return runWith(bandit, steps, rng, est, () => {
      let best = 0;
      let bestSample = -Infinity;
      for (let a = 0; a < bandit.nActions; a++) {
        const n = est.counts[a];
        const postPrecision = n + priorPrecision; // known σ²=1 => each obs adds precision 1
        const postMean = (n * est.Q[a]) / postPrecision; // prior mean 0 contributes nothing
        const postStd = Math.sqrt(1 / postPrecision);
        const sample = gaussian(rng, postMean, postStd);
        if (sample > bestSample) {
          bestSample = sample;
          best = a;
        }
      }
      return best;
    });
  };
}

// Aggregate one agent over many seeds. Each seed is a fresh, independent rng, so
// the agent faces independent reward-noise streams; averaging the regret curves
// is what turns a single lucky/unlucky run into a TREND. We also keep every
// seed's final regret to report run-to-run std — the spread that a single-seed
// plot would have hidden entirely (the headline methodological point of the
// chapter).
interface AgentSummary {
  meanRegretCurve: number[]; // averaged over seeds
  finalRegrets: number[]; // one per seed
  meanFinalRegret: number;
  stdFinalRegret: number;
  optimalRateLastWindow: number; // mean over seeds of (optimal pulls in last window / window)
}

function evaluate(agent: Agent, bandit: Bandit, steps: number, seeds: number[], lastWindow: number): AgentSummary {
  // Shared bandit: every seed differs only in reward noise, so variance below is
  // PURE run-to-run noise — the cleanest signal for "why average over seeds".
  return aggregate(agent, () => bandit, steps, seeds, lastWindow);
}

// Same aggregation but with a per-seed bandit factory. Used for the lock-in
// failure modes: greedy's pathology is "first lucky arm wins", which only shows
// up when the optimal arm is NOT always the index-0 arm greedy pulls first. A
// FRESH bandit per seed scatters the optimal arm across indices (it lands on
// index 0 only ~1/k of the time), so genuine lock-in onto a suboptimal arm
// appears — invisible on any single fixed bandit whose optimum happens to be
// arm 0. Regret is always scored against THAT seed's own bandit.
function evaluatePerSeedBandit(
  agent: Agent,
  makeBanditForSeed: (seed: number) => Bandit,
  steps: number,
  seeds: number[],
  lastWindow: number,
): AgentSummary {
  return aggregate(agent, makeBanditForSeed, steps, seeds, lastWindow);
}

function aggregate(
  agent: Agent,
  banditForSeed: (seed: number) => Bandit,
  steps: number,
  seeds: number[],
  lastWindow: number,
): AgentSummary {
  const sumCurve = new Array<number>(steps).fill(0);
  const finalRegrets: number[] = [];
  let optimalRateSum = 0;
  for (const seed of seeds) {
    const bandit = banditForSeed(seed);
    const run = agent(bandit, steps, mulberry32(seed));
    for (let t = 0; t < steps; t++) sumCurve[t] += run.regretCurve[t];
    finalRegrets.push(run.regretCurve[steps - 1]);
    // Optimal-arm rate over the FINAL window only: early steps are exploration
    // for everyone, so the converged behavior is what distinguishes strategies.
    let hits = 0;
    for (let t = steps - lastWindow; t < steps; t++) if (run.optimalFlags[t]) hits++;
    optimalRateSum += hits / lastWindow;
  }
  const n = seeds.length;
  const meanRegretCurve = sumCurve.map((s) => s / n);
  const meanFinalRegret = finalRegrets.reduce((a, b) => a + b, 0) / n;
  // Population std (divide by n): we're describing THIS seed set's spread, not
  // inferring a wider population, so Bessel's correction is not the point here.
  const variance = finalRegrets.reduce((acc, r) => acc + (r - meanFinalRegret) ** 2, 0) / n;
  return {
    meanRegretCurve,
    finalRegrets,
    meanFinalRegret,
    stdFinalRegret: Math.sqrt(variance),
    optimalRateLastWindow: optimalRateSum / n,
  };
}

// Downsample a long curve to a short sparkline so the ASCII line shows SHAPE
// (linear vs flattening) without 1000 glyphs. Every k-th point.
function sparkOf(curve: number[], points: number): string {
  const stride = Math.max(1, Math.floor(curve.length / points));
  return asciiSparkline(curve.filter((_, i) => i % stride === 0));
}

function pad(s: string, n: number): string {
  return s.length >= n ? s : s + " ".repeat(n - s.length);
}

function main(): void {
  console.log("Stage 01 — Bandit：探索 vs 利用，四策略 × 多 seed regret 真测\n");

  const nArms = 10;
  const steps = 1000;
  const lastWindow = 200; // converged-behavior window for optimal-arm rate
  const seeds = Array.from({ length: 24 }, (_, i) => 1000 + i); // 24 ≥ 20 seeds

  // Fixed shared bandit (its own construction seed). All strategies face the
  // SAME arm means, so the comparison isolates the policy, not the problem.
  // Seed 3 deliberately: its optimal arm is NOT index 0, so greedy gets no free
  // lunch from argmax's index-0 tie-break, and the gap (~0.45) is small enough
  // that strategies actually differentiate (a huge gap makes everyone look good).
  const bandit = makeBandit(nArms, mulberry32(3));
  const gap = bandit.trueMeans[bandit.optimalArm] - Math.max(
    ...bandit.trueMeans.filter((_, i) => i !== bandit.optimalArm),
  );
  console.log(
    `${nArms} 臂真均值 [${bandit.trueMeans.map((m) => m.toFixed(2)).join(", ")}]`,
  );
  console.log(
    `最优臂 #${bandit.optimalArm}(${bandit.trueMeans[bandit.optimalArm].toFixed(3)})，` +
      `与次优差距 ${gap.toFixed(3)}（差距越小越难分辨）`,
  );
  console.log(`预算 ${steps} 步 / arm；跨 ${seeds.length} 个独立 seed 平均\n`);

  // --- The four well-tuned strategies ---
  const strategies: { tag: string; agent: Agent }[] = [
    { tag: "ε-greedy ε=0.1", agent: epsilonGreedy(0.1) },
    { tag: "optimistic Q0=5", agent: optimisticGreedy(5) },
    { tag: "UCB1 c=2", agent: ucb1(2) },
    { tag: "Thompson", agent: thompson(100) },
  ];

  console.log(pad("策略", 18) + pad("均regret", 12) + pad("±std", 10) + pad("最优臂率", 10) + "regret曲线(均)");
  for (const { tag, agent } of strategies) {
    const s = evaluate(agent, bandit, steps, seeds, lastWindow);
    console.log(
      pad(tag, 18) +
        pad(s.meanFinalRegret.toFixed(1), 12) +
        pad("±" + s.stdFinalRegret.toFixed(1), 10) +
        pad((s.optimalRateLastWindow * 100).toFixed(0) + "%", 10) +
        sparkOf(s.meanRegretCurve, 40),
    );
  }
  console.log(
    "→ ε-greedy 曲线全程近线性（固定 ε 永远在交探索税）；UCB1/Thompson 找到最优臂后曲线变缓(次线性)。",
  );
  console.log(
    "→ optimistic 纯贪心靠高初值逼出早期扫描，收敛后几乎不再 regret —— 但这完全依赖初值设对(见 F2)。",
  );

  // --- Why we MUST average over seeds: show the single-seed spread ---
  console.log("\n[为什么必须多 seed 平均] 同一策略不同 seed 的最终 regret 差异：");
  const probe = epsilonGreedy(0.1);
  const probeSummary = evaluate(probe, bandit, steps, seeds, lastWindow);
  const minR = Math.min(...probeSummary.finalRegrets);
  const maxR = Math.max(...probeSummary.finalRegrets);
  console.log(
    `  ε-greedy(0.1) 跨 ${seeds.length} seed：最小 ${minR.toFixed(1)} / 最大 ${maxR.toFixed(1)} / ` +
      `均值 ${probeSummary.meanFinalRegret.toFixed(1)} ± ${probeSummary.stdFinalRegret.toFixed(1)}`,
  );
  console.log(
    `  单 seed 极差 ${(maxR - minR).toFixed(1)}（≈均值的 ${((maxR - minR) / probeSummary.meanFinalRegret * 100).toFixed(0)}%）` +
      ` → 只跑 1 个 seed 可能挑到最好/最差的，结论全错；均±std 才是诚实数字。`,
  );

  // --- Failure mode F1: ε=0 pure greedy locks onto a suboptimal arm ---
  // Per-seed bandits here (see evaluatePerSeedBandit): on the single shared
  // bandit above, greedy's lock-in is invisible whenever the optimum happens to
  // be the arm greedy tries first. Scattering the optimum across seeds surfaces
  // the real pathology. Construction seed is offset from the run seed so the
  // bandit's structure and the reward noise stream are independently varied.
  const banditForSeed = (seed: number): Bandit => makeBandit(nArms, mulberry32(seed + 7777));
  console.log("\n[失败模式 F1] ε=0 纯贪心：早期幸运样本锁死次优臂（每 seed 独立 bandit 才看得见）");
  const greedy = evaluatePerSeedBandit(epsilonGreedy(0), banditForSeed, steps, seeds, lastWindow);
  console.log(
    `  ε=0 均regret ${greedy.meanFinalRegret.toFixed(1)} ± ${greedy.stdFinalRegret.toFixed(1)}，` +
      `最优臂率 ${(greedy.optimalRateLastWindow * 100).toFixed(0)}%`,
  );
  // Count seeds genuinely locked onto a suboptimal arm: never picks the optimal
  // arm in the whole final window. This is the lock-in fingerprint, measured per
  // seed against that seed's own ground-truth optimal arm.
  let lockedSeeds = 0;
  for (const seed of seeds) {
    const run = epsilonGreedy(0)(banditForSeed(seed), steps, mulberry32(seed));
    let hits = 0;
    for (let t = steps - lastWindow; t < steps; t++) if (run.optimalFlags[t]) hits++;
    if (hits === 0) lockedSeeds++; // never on the optimal arm at convergence = locked
  }
  console.log(
    `  ${seeds.length} seed 中 ${lockedSeeds} 个完全锁死在次优臂（末段 ${lastWindow} 步 0 次碰最优）` +
      ` → 巨大 std 正是锁死指纹：有 seed 幸运命中最优(regret≈0)，有 seed 灾难性锁死(regret 线性涨)。`,
  );

  // --- Failure mode F2: optimistic init too LOW degenerates to greedy ---
  // Also per-seed bandits, same reason: with init too low, optimistic IS greedy,
  // and greedy's lock-in only shows when the optimum isn't always arm 0.
  console.log("\n[失败模式 F2] optimistic 初值设太低 → 退化成纯贪心（探索假象消失）");
  for (const q0 of [5, 1, 0, -2]) {
    const s = evaluatePerSeedBandit(optimisticGreedy(q0), banditForSeed, steps, seeds, lastWindow);
    // True arm means here ~ N(0,1), so a Q0 well above ~1 sits above essentially
    // every arm and forces a full sweep; Q0 at/below the mean range degenerates.
    const verdict =
      q0 >= 5 ? "← 远高于真均值：逼出全臂扫描" : q0 <= 0 ? "← 不高于真均值：退化贪心，锁死" : "← 临界：仅部分臂被逼探索";
    console.log(
      `  Q0=${pad(String(q0), 3)} 均regret ${pad(s.meanFinalRegret.toFixed(1), 8)} ` +
        `最优臂率 ${pad((s.optimalRateLastWindow * 100).toFixed(0) + "%", 5)} ${verdict}`,
    );
  }
  console.log(
    "  → optimistic 的探索完全寄生于「初值 > 真均值」这个前提；前提一破，它就是 F1 的纯贪心。",
  );

  console.log("\n[诚实声明] 绝对 regret 依赖这套 seed 与 toy bandit 的真均值，乐观；");
  console.log("可迁移的是相对趋势：UCB/Thompson 次线性 < ε-greedy 线性 < 纯贪心锁死风险，且必须多 seed 才看得清。");
}

main();
