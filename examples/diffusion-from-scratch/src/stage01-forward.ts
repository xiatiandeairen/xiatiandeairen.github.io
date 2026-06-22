// stage01-forward.ts — The forward (noising) process q(x_t | x_0): a Markov chain that
// drowns clean data in Gaussian noise step by step. This is the HALF of diffusion you can
// write with zero learning — pure closed-form algebra — and it defines the target the
// reverse network must later undo.
//
// WHY this stage exists before any network: the forward process is not learned. It is a
//   fixed corruption schedule. If you don't trust THIS, nothing downstream means anything,
//   because the reverse model is trained to predict exactly the ε this process injected.
//   So stage01 does three things and PROVES each with a real number, not a claim:
//     1. precompute ᾱ_t and show it decays monotonically to ~0 (x_T must be ~pure noise)
//     2. apply the one-shot formula x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε and SHOW the moons dissolve
//     3. Monte-Carlo verify the formula's first two moments against theory (<2% error)
//   Then it breaks the schedule on purpose to show the dominant real-world failure:
//   noising too fast collapses ᾱ early, so the middle timesteps carry no signal and the
//   reverse network has nothing to learn there.
//
// KEY IDENTITY (the whole stage rests on this): the per-step chain
//   q(x_t | x_{t-1}) = N(√(1-β_t)·x_{t-1}, β_t·I)  composes in closed form to
//   q(x_t | x_0)     = N(√ᾱ_t·x_0, (1-ᾱ_t)·I),   ᾱ_t = ∏ α_s,  α_s = 1-β_s.
//   So we can jump to ANY t in one step instead of simulating t Markov steps — that is what
//   makes training tractable (sample a random t per batch, no chain rollout).
//
// DETERMINISM: every random byte comes from RNG(1337) (core/rng.ts). Re-running prints
//   identical numbers. The Monte-Carlo section deliberately draws fresh ε each trial but
//   from the same seeded stream, so the estimated moments are reproducible too.

import { RNG } from "./core/rng.js";
import { Tensor, randnLike } from "./core/tensor.js";
import { linearSchedule, type NoiseSchedule } from "./core/schedule.js";
import { twoMoons } from "./core/data.js";
import { scatterASCII } from "./core/plot.js";

/**
 * Apply the closed-form forward marginal q(x_t | x_0) = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε.
 *
 * Pure given (x0, schedule, t, eps): no IO, no hidden RNG — the caller owns the noise so the
 * result is reproducible and the function is trivially testable. eps MUST be the SAME shape
 * as x0 (one independent N(0,1) draw per scalar); using a single shared scalar would correlate
 * the coordinates and break the isotropic-noise assumption the reverse step later inverts.
 *
 * Returns a fresh Tensor; x0 and eps are not mutated.
 */
function forwardSample(x0: Tensor, schedule: NoiseSchedule, t: number, eps: Tensor): Tensor {
  const sa = schedule.sqrtAlphaBar[t]; // √ᾱ_t  — how much clean signal survives
  const so = schedule.sqrtOneMinusAlphaBar[t]; // √(1-ᾱ_t) — how much noise replaces it
  // signal·x0 + noise·eps, both scalar broadcasts. As t→T, sa→0 and so→1, so x_t → pure ε.
  return x0.mulScalar(sa).add(eps.mulScalar(so));
}

/** Print √ᾱ progression at probe timesteps and assert the schedule's load-bearing invariants. */
function reportScheduleDecay(schedule: NoiseSchedule): void {
  const T = schedule.T;
  const probes = [0, 250, 500, 750, T - 1];
  console.log(`\n=== 1. 噪声调度 ᾱ_t 衰减 (linearSchedule, T=${T}) ===`);
  console.log("  ᾱ_t = 累积信号保留系数; ᾱ_0≈1 (几乎全是原信号), ᾱ_{T-1}→0 (几乎全是噪声)");
  for (const t of probes) {
    const ab = schedule.alphaBars[t];
    const signalPct = (100 * schedule.sqrtAlphaBar[t]).toFixed(1);
    console.log(`  t=${String(t).padStart(3)}  ᾱ_t=${ab.toExponential(3)}  √ᾱ_t=${schedule.sqrtAlphaBar[t].toFixed(4)} (信号占比≈${signalPct}%)`);
  }

  // INVARIANT 1: strict monotonic decrease. The schedule constructor already asserts this,
  // but we re-check here because it is the premise the reverse sampler depends on (each step
  // must remove a well-defined amount of noise). A silent violation = garbage samples.
  for (let t = 1; t < T; t++) {
    if (schedule.alphaBars[t] >= schedule.alphaBars[t - 1]) {
      throw new Error(`ᾱ not strictly decreasing at t=${t}: ${schedule.alphaBars[t]} >= ${schedule.alphaBars[t - 1]}`);
    }
  }
  // INVARIANT 2: ᾱ_{T-1} < 0.01. If x_T still retains >1% signal, sampling from N(0,I) at
  // inference starts off-distribution (the model never saw that residual) and never recovers.
  const abLast = schedule.alphaBars[T - 1];
  if (abLast >= 0.01) {
    throw new Error(`ᾱ_{T-1}=${abLast} >= 0.01: schedule does not drive x_T to ~pure noise`);
  }
  console.log(`  [PASS] ᾱ 严格单调递减 且 ᾱ_{${T - 1}}=${abLast.toExponential(3)} < 0.01 (x_T ≈ 纯噪声)`);
}

/** Noise a fixed two-moons batch to several t and print the point clouds dissolving. */
function reportDissolution(x0: Tensor, schedule: NoiseSchedule, rng: RNG): void {
  const probes = [0, 50, 200, 500, schedule.T - 1];
  console.log(`\n=== 2. 双月 → 高斯球: 一步加噪后的点云 (n=${x0.shape[0]}) ===`);
  console.log("  肉眼可见: t 越大, 两条月牙越被噪声抹平, 最终塌成各向同性的圆形噪声团");
  for (const t of probes) {
    // Fresh independent noise per probe (different t = different corruption draw). One eps
    // per scalar coordinate, same shape as x0.
    const eps = randnLike(x0, () => rng.gaussian());
    const xt = forwardSample(x0, schedule, t, eps);
    console.log(`\n  --- t=${t}  (√ᾱ=${schedule.sqrtAlphaBar[t].toFixed(3)}, √(1-ᾱ)=${schedule.sqrtOneMinusAlphaBar[t].toFixed(3)}) ---`);
    console.log(scatterASCII(xt, 56, 18));
  }
}

/**
 * Monte-Carlo proof that forwardSample really has the claimed marginal. Take ONE fixed clean
 * point, noise it `trials` times to a fixed t, and estimate E[x_t] and Var[x_t]. Theory says
 * E[x_t] = √ᾱ_t·x_0 (the noise is zero-mean) and Var[x_t] = (1-ᾱ_t) (per coordinate, since ε
 * is N(0,1) scaled by √(1-ᾱ_t)).
 *
 * HONEST TOLERANCE — why the tests aren't all "relative error < 2%": a Monte-Carlo mean has
 *   standard error σ/√N = √(1-ᾱ)/√N. The expected mean √ᾱ·x_0 here is ~0.195, so RELATIVE
 *   mean error is SE/0.195 ≈ √(0.92/N)/0.195 — at N=5000 that is ~7% at 1σ, statistically
 *   IMPOSSIBLE to push under 2% without ~60× more samples. That is sampling noise, not a code
 *   bug. The meaningful checks are: (a) mean error measured against the NOISE SCALE √(1-ᾱ)
 *   (the natural unit — "is the recovered signal within a small fraction of the noise we
 *   added?"), which falls well under 2%; (b) variance relative error < 2% (variance's SE is
 *   √(2/N)·σ², small at large N). We crank N to 200k (sub-second on CPU) so both are robustly
 *   under 2% AND print each estimate's standard error so the reader can see the noise floor.
 */
function reportMonteCarlo(schedule: NoiseSchedule, rng: RNG): void {
  const t = 500;
  const trials = 200_000;
  // A single fixed clean point (not from the moons batch — we want a known x0 we can read off).
  const x0 = new Tensor([0.7, -0.4], [1, 2]);
  const sa = schedule.sqrtAlphaBar[t];
  const noiseScale = schedule.sqrtOneMinusAlphaBar[t]; // √(1-ᾱ_t): the std of injected noise
  const expectedMean = [x0.data[0] * sa, x0.data[1] * sa];
  const expectedVar = 1 - schedule.alphaBars[t]; // same for both coords (isotropic)

  // Streaming mean/variance over trials. We accumulate sums rather than storing all samples;
  // population variance E[x²]-E[x]² is fine here (large N, no small-sample bias concern).
  const sum = [0, 0];
  const sumSq = [0, 0];
  for (let i = 0; i < trials; i++) {
    const eps = randnLike(x0, () => rng.gaussian());
    const xt = forwardSample(x0, schedule, t, eps);
    for (let c = 0; c < 2; c++) {
      sum[c] += xt.data[c];
      sumSq[c] += xt.data[c] * xt.data[c];
    }
  }
  const estMean = [sum[0] / trials, sum[1] / trials];
  const estVar = [sumSq[0] / trials - estMean[0] * estMean[0], sumSq[1] / trials - estMean[1] * estMean[1]];
  const seMean = noiseScale / Math.sqrt(trials); // theoretical 1σ standard error of the mean

  console.log(`\n=== 3. 蒙特卡洛验证闭式公式 (固定 x_0=[${x0.data[0]}, ${x0.data[1]}], t=${t}, ${trials} 次加噪) ===`);
  console.log(`  理论: E[x_t]=√ᾱ_t·x_0=[${expectedMean[0].toFixed(4)}, ${expectedMean[1].toFixed(4)}], Var[x_t]=(1-ᾱ_t)=${expectedVar.toFixed(4)}`);
  console.log(`  实测: E[x_t]=[${estMean[0].toFixed(4)}, ${estMean[1].toFixed(4)}], Var[x_t]=[${estVar[0].toFixed(4)}, ${estVar[1].toFixed(4)}]`);
  console.log(`  均值标准误 (噪声地板) σ/√N=±${seMean.toFixed(4)} — 实测均值偏差应落在它几倍之内`);

  // Mean error judged against the NOISE SCALE (see header): the natural denominator. Variance
  // judged relatively. All four must be < 2% — these are the statistically achievable checks.
  const checks: Array<[string, number]> = [
    ["mean[0]/noise", Math.abs(estMean[0] - expectedMean[0]) / noiseScale],
    ["mean[1]/noise", Math.abs(estMean[1] - expectedMean[1]) / noiseScale],
    ["var[0] rel", Math.abs(estVar[0] - expectedVar) / expectedVar],
    ["var[1] rel", Math.abs(estVar[1] - expectedVar) / expectedVar],
  ];
  const TOL = 0.02;
  let worst = 0;
  for (const [name, err] of checks) {
    console.log(`  ${name.padEnd(14)} 误差 ${(100 * err).toFixed(2)}%`);
    worst = Math.max(worst, err);
  }
  // Informational: the raw relative mean error, to make the noise-floor point concrete.
  const relMean0 = (100 * Math.abs(estMean[0] - expectedMean[0]) / Math.abs(expectedMean[0])).toFixed(2);
  console.log(`  (参考: mean[0] 直接相对误差 ${relMean0}% — 受 σ/√N 采样噪声主导, 非代码错误)`);
  if (worst >= TOL) {
    throw new Error(`Monte-Carlo moments deviate ${(100 * worst).toFixed(2)}% (>= ${100 * TOL}%): closed-form formula or sampling is wrong`);
  }
  console.log(`  [PASS] 最大误差 ${(100 * worst).toFixed(2)}% < ${100 * TOL}% — 闭式公式 √ᾱ·x_0 + √(1-ᾱ)·ε 的前两阶矩与理论一致`);
}

/**
 * FAILURE MODE — noising too fast. Build a degenerate linear schedule with a huge terminal β
 * (b1=0.5 instead of 0.02). β that large makes α=1-β tiny, so the cumulative product ᾱ_t
 * collapses to ~0 within the first ~100 steps. Consequence: q(x_t|x_0) for t≈100 and t≈999 are
 * the SAME distribution (both ≈ N(0,I)) — the chain reached pure noise almost immediately.
 *
 * WHY this kills training (the real lesson): the reverse network is trained to predict ε given
 * x_t and t. If x_200 and x_999 are statistically indistinguishable, there is no ε-dependent
 * signal left in the middle/late timesteps for the network to key off — its loss is flat there,
 * gradients vanish, and those steps learn nothing. The good linear/cosine schedules deliberately
 * keep ᾱ falling SLOWLY so every t carries a distinct, learnable amount of signal.
 */
function reportFastNoiseFailure(x0: Tensor, rng: RNG): void {
  const T = 1000;
  const bad = linearSchedule(T, 1e-4, 0.5); // b1=0.5: noise injected per step is enormous
  console.log(`\n=== 4. 失败模式: 加噪太快 (linearSchedule b1=0.5, T=${T}) ===`);
  const probe = [10, 50, 100, 200, 999];
  console.log("  ᾱ_t 在前 ~100 步就塌到 ~0 — 之后所有 t 的分布几乎相同 (都是纯噪声):");
  for (const t of probe) {
    console.log(`    t=${String(t).padStart(3)}  ᾱ_t=${bad.alphaBars[t].toExponential(3)}  √ᾱ_t=${bad.sqrtAlphaBar[t].toExponential(3)}`);
  }

  // Quantify "the middle and the end look the same": compare the empirical variance of x_t at
  // t=200 vs t=999 over the batch. With a sane schedule var(t=200) < var(t=999); here both ≈ 1.
  const varAt = (t: number): number => {
    const eps = randnLike(x0, () => rng.gaussian());
    const xt = forwardSample(x0, bad, t, eps);
    const n = xt.size;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += xt.data[i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (xt.data[i] - mean) ** 2;
    return v / n;
  };
  const v200 = varAt(200);
  const v999 = varAt(999);
  console.log(`\n  批方差 var(x_200)=${v200.toFixed(4)}  vs  var(x_999)=${v999.toFixed(4)}  (差异 ${(100 * Math.abs(v200 - v999) / v999).toFixed(1)}%)`);

  const eps200 = randnLike(x0, () => rng.gaussian());
  const eps999 = randnLike(x0, () => rng.gaussian());
  console.log("\n  --- t=200 (本应仍带可辨认结构) ---");
  console.log(scatterASCII(forwardSample(x0, bad, 200, eps200), 56, 14));
  console.log("  --- t=999 ---");
  console.log(scatterASCII(forwardSample(x0, bad, 999, eps999), 56, 14));
  console.log("\n  结论: t=200 与 t=999 的散点几乎无差别 → 中间步不含 ε 相关信号 →");
  console.log("        反向网络在这些 t 上无梯度可学 (loss 平坦)。这就是为何真实 schedule");
  console.log("        要让 ᾱ 平缓下降: 每个 t 都携带一份不同的、可学习的信号量。");
}

function main(): void {
  const rng = new RNG(1337);
  const schedule = linearSchedule(1000); // Ho et al. 2020 defaults (b0=1e-4, b1=0.02)
  const x0 = twoMoons(400, 0.08, rng); // fixed clean data; seed makes it byte-reproducible

  console.log("# Stage 01 — 前向加噪过程 q(x_t | x_0)");
  console.log("# 离线确定性 (seed=1337), 所有数字均为真实计算/采样, 非估算。");

  reportScheduleDecay(schedule);
  reportDissolution(x0, schedule, rng);
  reportMonteCarlo(schedule, rng);
  reportFastNoiseFailure(x0, rng);

  console.log("\n# 完成。前向过程是无参数的固定算子; 下一章用网络学习反向去噪。");
}

main();
