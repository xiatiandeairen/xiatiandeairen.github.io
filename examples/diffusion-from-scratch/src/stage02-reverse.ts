// stage02-reverse.ts — The reverse process: walking from pure noise back to data, and the
// one quantity that forces us to learn a network (chapter 03+).
//
// THE CENTRAL CLAIM OF THIS CHAPTER: undoing ONE forward step has an exact closed form — the
//   posterior q(x_{t-1}|x_t, x_0), a Gaussian whose mean/variance are pure schedule arithmetic
//   (computePosterior). The catch: its mean depends on x_0, the clean sample. At generation
//   time you do NOT have x_0 — it is what you are trying to produce. So the reverse step needs
//   an ESTIMATE of x_0 (equivalently of the noise ε) from x_t and t. THAT estimate is the only
//   thing chapters 03+ learn with a network. Everything else in the sampler is fixed arithmetic.
//
// THE HONEST "CHEAT": a naive demo would plug each point's OWN true x_0 back in every step.
//   That is a tautology — the last step (coefX0=1) just returns x_0, so "recovery" is literally
//   the input and noise vs no-noise look identical. It proves nothing. Instead we use the
//   OPTIMAL analytic denoiser for a FINITE dataset: E[x_0 | x_t] = Σ_i w_i · x_0^(i), with
//   w_i ∝ exp(-||x_t - √ᾱ_t·x_0^(i)||² / (2(1-ᾱ_t))) — the exact Bayes posterior over WHICH
//   data point generated x_t (estimateX0). This is what a perfect network would output. No
//   per-point cheating, no training. It is the ground-truth score, computable here because the
//   dataset is finite and tiny.
//
// WHY THIS SETUP EXPOSES THE REAL MECHANISM:
//   §2 (works): chain x_T -> x_0 with the optimal denoiser + injected noise z. Early steps have
//     ᾱ_t≈0 so all weights are ~equal -> the x_0 estimate is the data MEAN (one blurry point);
//     the injected z is what nudges each trajectory toward a different mode. As t->0 the weights
//     sharpen onto real data points. Recovered cloud matches two-moons (Chamfer < threshold).
//   §3 (FAILURE): SAME optimal denoiser, but posterior variance forced to 0 (drop z). Now every
//     x_T flows deterministically toward the conditional MEAN at each step. With nothing to
//     break the symmetry early on, trajectories funnel together -> the cloud collapses to a few
//     clusters in the gap between the two moons. The mean was right at every step; only the
//     stochasticity was removed. This is the miniature of mode collapse and is exactly why z is
//     load-bearing in DDPM.
//
// WHAT IS REAL (honesty): no network, no training. Seeded Gaussian noise + exact posterior +
//   exact finite-data score. Every printed number — posterior coefficients, Chamfer distance,
//   cluster counts — is computed from the actual buffers. Absolute Chamfer values are optimistic
//   (2-D toy, n=200); the transferable fact is the RELATIVE gap between §2 and §3.
//
// PREVIEW (chapter 05): DDIM ALSO drops z (deterministic sampling) yet does NOT collapse — it
//   escapes by making each step's x_0 estimate sharper / more informative so trajectories stay
//   separated without random kicks. Not implemented here; this stage only plants the question.

import { RNG } from "./core/rng.js";
import { Tensor } from "./core/tensor.js";
import { twoMoons } from "./core/data.js";
import { cosineSchedule, type NoiseSchedule } from "./core/schedule.js";
import { scatterASCII } from "./core/plot.js";

// All randomness in this stage flows from ONE seeded stream so the printed numbers are
// byte-reproducible (see core/rng.ts). Reordering draws would change every figure below.
const rng = new RNG(1337);

/**
 * Optimal x_0 estimate for a FINITE dataset: E[x_0 | x_t] = Σ_i softmax_i(logits) · x_0^(i).
 *
 * The forward model says x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε. For data point i the likelihood of the
 * observed x_t is Gaussian centered at √ᾱ_t·x_0^(i) with variance (1-ᾱ_t). With a uniform prior
 * over the n data points, the posterior weight is the softmax of -||x_t - √ᾱ_t·x_0^(i)||² /
 * (2(1-ᾱ_t)). The conditional mean is the weighted average of the data points. This IS the
 * Bayes-optimal denoiser — the thing a trained network approximates. We can compute it exactly
 * only because the "distribution" is a finite point set; that is the whole point of the toy.
 *
 * INVARIANT: returns a [batch,2] estimate. As ᾱ_t -> 0 (large t) the logits flatten and the
 *   estimate -> data mean (one point); as ᾱ_t -> 1 (small t) it concentrates on the nearest
 *   data point. FAILURE MODE guarded: (1-ᾱ_t) can be ~0 at t=0 -> division blows up; we floor it.
 * COST: O(batch · n · 2). Fine for batch=n=200; a real model replaces this whole function.
 */
function estimateX0(xt: Tensor, dataset: Tensor, t: number, sched: NoiseSchedule): Tensor {
  const batch = xt.shape[0];
  const n = dataset.shape[0];
  const sqrtAB = sched.sqrtAlphaBar[t];
  // Floor the noise variance: at t=0 (1-ᾱ_t)≈0 makes the softmax temperature 0 -> Inf logits.
  const twoVar = Math.max(2 * (1 - sched.alphaBars[t]), 1e-12);
  const xtD = xt.data;
  const dsD = dataset.data;
  const out = new Float64Array(batch * 2);
  const logits = new Float64Array(n);
  for (let b = 0; b < batch; b++) {
    const px = xtD[b * 2];
    const py = xtD[b * 2 + 1];
    // logit_i = -||x_t - √ᾱ_t·x_0^(i)||² / (2(1-ᾱ_t)); track max for a stable softmax.
    let maxLogit = -Infinity;
    for (let i = 0; i < n; i++) {
      const dx = px - sqrtAB * dsD[i * 2];
      const dy = py - sqrtAB * dsD[i * 2 + 1];
      const l = -(dx * dx + dy * dy) / twoVar;
      logits[i] = l;
      if (l > maxLogit) maxLogit = l;
    }
    let sumW = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.exp(logits[i] - maxLogit); // shift by max -> no overflow, ratios unchanged
      logits[i] = w;
      sumW += w;
    }
    let ex = 0;
    let ey = 0;
    for (let i = 0; i < n; i++) {
      const w = logits[i] / sumW;
      ex += w * dsD[i * 2];
      ey += w * dsD[i * 2 + 1];
    }
    out[b * 2] = ex;
    out[b * 2 + 1] = ey;
  }
  return new Tensor(out, [batch, 2]);
}

/**
 * Closed-form posterior q(x_{t-1} | x_t, x_0) coefficients (Ho et al. 2020 eq. 6-7):
 *   μ̃_t = coefX0 · x_0 + coefXt · x_t
 *   coefX0 = √ᾱ_{t-1}·β_t / (1-ᾱ_t),  coefXt = √α_t·(1-ᾱ_{t-1}) / (1-ᾱ_t)
 *   β̃_t   = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t        (isotropic posterior variance)
 * Here x_0 is the ESTIMATE from estimateX0, not the true clean sample. INVARIANT: t in [0,T-1].
 *   t=0 uses the convention ᾱ_{-1}=1 -> coefX0=1, coefXt=0, var=0: the last step returns the
 *   x_0 estimate, which is correct (nothing left to denoise).
 */
function computePosterior(
  xt: Tensor,
  x0Estimate: Tensor,
  t: number,
  sched: NoiseSchedule,
): { mean: Tensor; variance: number } {
  const n = xt.shape[0];
  const alphaBarT = sched.alphaBars[t];
  const alphaBarPrev = t > 0 ? sched.alphaBars[t - 1] : 1; // ᾱ_{-1}=1 boundary convention
  const betaT = sched.betas[t];
  const alphaT = sched.alphas[t];
  const oneMinusAlphaBarT = 1 - alphaBarT;
  const coefX0 = (Math.sqrt(alphaBarPrev) * betaT) / oneMinusAlphaBarT;
  const coefXt = (Math.sqrt(alphaT) * (1 - alphaBarPrev)) / oneMinusAlphaBarT;
  const variance = oneMinusAlphaBarT === 0 ? 0 : ((1 - alphaBarPrev) / oneMinusAlphaBarT) * betaT;
  const mean = new Float64Array(n * 2);
  const xtD = xt.data;
  const x0D = x0Estimate.data;
  for (let i = 0; i < n * 2; i++) mean[i] = coefX0 * x0D[i] + coefXt * xtD[i];
  return { mean: new Tensor(mean, [n, 2]), variance };
}

/**
 * One reverse step: estimate x_0 from x_t, form the posterior, sample x_{t-1} ~ N(mean, var·I).
 * `injectNoise=false` (the §3 failure mode) forces std=0: x_{t-1} becomes the posterior MEAN
 * deterministically. The x_0 estimate (hence the mean) is identical to the §2 path — only the
 * stochasticity differs, which is what isolates the role of z.
 */
function reverseStep(
  xt: Tensor,
  dataset: Tensor,
  t: number,
  sched: NoiseSchedule,
  injectNoise: boolean,
): Tensor {
  const x0Estimate = estimateX0(xt, dataset, t, sched);
  const { mean, variance } = computePosterior(xt, x0Estimate, t, sched);
  // Standard DDPM convention: no noise on the final step (t=0) regardless.
  const std = injectNoise && t > 0 ? Math.sqrt(variance) : 0;
  const out = mean.data.slice();
  if (std > 0) for (let i = 0; i < out.length; i++) out[i] += std * rng.gaussian();
  return new Tensor(out, xt.shape);
}

/**
 * Run the full reverse chain x_T -> x_0 with the optimal analytic denoiser. Starts from
 * x_T ~ N(0, I): cosine schedule gives ᾱ_T ≈ 0 so x_T is essentially pure noise — the same
 * prior a real sampler starts from. Returns the final cloud plus snapshots at the requested
 * timesteps for the "noise blob -> two moons" progression.
 */
function runReverseChain(
  dataset: Tensor,
  sched: NoiseSchedule,
  injectNoise: boolean,
  snapshotAt: number[],
  nSamples: number,
): { final: Tensor; snapshots: Map<number, Tensor> } {
  const T = sched.T;
  const xT = new Float64Array(nSamples * 2);
  for (let i = 0; i < nSamples * 2; i++) xT[i] = rng.gaussian();
  let x: Tensor = new Tensor(xT, [nSamples, 2]);
  const snapshots = new Map<number, Tensor>();
  const snapSet = new Set(snapshotAt);
  for (let t = T - 1; t >= 0; t--) {
    if (snapSet.has(t)) snapshots.set(t, new Tensor(x.data.slice(), [nSamples, 2]));
    x = reverseStep(x, dataset, t, sched, injectNoise);
  }
  return { final: x, snapshots };
}

/**
 * Symmetric Chamfer distance between two [·,2] clouds: mean over A of squared distance to the
 * nearest point in B, plus the same swapped. WHY this metric: the generated points are NOT the
 * same identities as the data (x_T is fresh noise), so a per-point error is meaningless. We
 * only care that the supports overlap — every generated point near some data point AND every
 * data point near some generated point. Low in both directions => no missing modes, no spurious
 * mass. COST: O(nA·nB) brute force; fine for n≈200.
 */
function chamferDistance(a: Tensor, b: Tensor): number {
  const ad = a.data;
  const bd = b.data;
  const nearestMean = (srcD: Float64Array, srcN: number, dstD: Float64Array, dstN: number): number => {
    let total = 0;
    for (let i = 0; i < srcN; i++) {
      let best = Infinity;
      const sx = srcD[i * 2];
      const sy = srcD[i * 2 + 1];
      for (let j = 0; j < dstN; j++) {
        const dx = sx - dstD[j * 2];
        const dy = sy - dstD[j * 2 + 1];
        const d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
      }
      total += best;
    }
    return total / srcN;
  };
  return nearestMean(ad, a.shape[0], bd, b.shape[0]) + nearestMean(bd, b.shape[0], ad, a.shape[0]);
}

/**
 * Count distinct clusters by greedy radius merging: a point joins an existing cluster if within
 * `radius` of its seed, else seeds a new one. Used to QUANTIFY the §3 collapse — a healthy cloud
 * spreads into many clusters (continuous support), a collapsed one funnels into a few. Greedy &
 * order-dependent, but deterministic given the fixed buffer order: turns "looks collapsed" into
 * a number.
 */
function countClusters(cloud: Tensor, radius: number): number {
  const n = cloud.shape[0];
  const d = cloud.data;
  const seedX: number[] = [];
  const seedY: number[] = [];
  const r2 = radius * radius;
  for (let i = 0; i < n; i++) {
    const x = d[i * 2];
    const y = d[i * 2 + 1];
    let joined = false;
    for (let k = 0; k < seedX.length; k++) {
      const dx = x - seedX[k];
      const dy = y - seedY[k];
      if (dx * dx + dy * dy <= r2) {
        joined = true;
        break;
      }
    }
    if (!joined) {
      seedX.push(x);
      seedY.push(y);
    }
  }
  return seedX.length;
}

function main(): void {
  const N = 200; // dataset size = denoiser "memory"; also the per-step softmax cost driver.
  const T = 200; // small T keeps it fast; cosine drives ᾱ_T ~ 0 even here (see stage01).
  const data = twoMoons(N, 0.08, rng);
  const sched = cosineSchedule(T);

  console.log("=== 第 02 章:反向去噪 — 已知最优 x_0 估计时, 反向链完美工作 ===\n");
  console.log(`数据: two-moons n=${N}, schedule: cosine T=${T}`);
  console.log(`ᾱ_T = ${sched.alphaBars[T - 1].toExponential(3)} (≈0 -> x_T 几乎纯噪声, 反向从 N(0,I) 出发合理)`);
  console.log("去噪器: 有限数据集的解析最优 E[x_0|x_t] (Bayes 后验, 无网络无训练 — 网络要学的就是它)\n");

  // --- §1: posterior coefficients are pure schedule arithmetic ---
  console.log("--- §1 解析后验 q(x_{t-1}|x_t,x_0) 的系数 (纯 schedule 算术) ---");
  console.log("  t  |   coefX0   coefXt   posterior_var   (μ = coefX0·x̂_0 + coefXt·x_t)");
  for (const t of [T - 1, 150, 100, 50, 10, 1, 0]) {
    const abT = sched.alphaBars[t];
    const abPrev = t > 0 ? sched.alphaBars[t - 1] : 1;
    const om = 1 - abT;
    const coefX0 = (Math.sqrt(abPrev) * sched.betas[t]) / om;
    const coefXt = (Math.sqrt(sched.alphas[t]) * (1 - abPrev)) / om;
    const v = om === 0 ? 0 : ((1 - abPrev) / om) * sched.betas[t];
    console.log(`  ${String(t).padStart(3)} | ${coefX0.toFixed(6)} ${coefXt.toFixed(6)}   ${v.toExponential(3)}`);
  }
  console.log("  注: t 越小 coefX0 越大 -> 越靠终点越信任 x̂_0; t=0 时 var=0 直接返回 x̂_0\n");

  // --- §2: optimal-denoiser sampler with full posterior noise recovers the distribution ---
  const snapshotAt = [T - 1, 100, 50, 0];
  const { final, snapshots } = runReverseChain(data, sched, /*injectNoise=*/ true, snapshotAt, N);

  console.log("--- §2 采样: x_T 纯噪声出发, 每步用最优 E[x_0|x_t] 算后验并加噪采样 x_{t-1} ---");
  console.log("中间散点 (噪声团逐步收拢成双月):\n");
  for (const t of snapshotAt) {
    const snap = snapshots.get(t)!;
    const tag = t === T - 1 ? `t=${t} (起点, 纯噪声)` : t === 0 ? `t=${t} (x_0 前最后一步)` : `t=${t}`;
    console.log(`  [ ${tag} ]`);
    console.log(scatterASCII(snap, 56, 13));
    console.log("");
  }
  console.log("  [ 还原结果 (反向链终点) ]");
  console.log(scatterASCII(final, 56, 13));
  console.log("");
  console.log("  [ 原始 two-moons (对照) ]");
  console.log(scatterASCII(data, 56, 13));
  console.log("");

  const chamferRecovered = chamferDistance(final, data);
  // Baseline: data vs fresh N(0,I) of the same size — the "did nothing" reference. Recovery
  // must be dramatically closer or the word "recovery" is empty. Computed real, not asserted.
  const noiseBuf = new Float64Array(N * 2);
  for (let i = 0; i < N * 2; i++) noiseBuf[i] = rng.gaussian();
  const chamferNoise = chamferDistance(new Tensor(noiseBuf, [N, 2]), data);

  const THRESHOLD = 0.05; // toy threshold; recovered cloud should sit well under it.
  const pass = chamferRecovered < THRESHOLD;
  console.log("--- §2 量化: Chamfer 距离 (越小越接近原分布) ---");
  console.log(`  还原 vs 原数据    : ${chamferRecovered.toFixed(6)}`);
  console.log(`  纯噪声 vs 原数据  : ${chamferNoise.toFixed(6)}  (基线: 啥也不做)`);
  console.log(`  改善倍数          : ${(chamferNoise / chamferRecovered).toFixed(1)}x`);
  console.log(`  断言 ${chamferRecovered.toFixed(6)} < ${THRESHOLD} -> ${pass ? "PASS" : "FAIL"}`);
  if (!pass) throw new Error(`stage02 §2: recovered Chamfer ${chamferRecovered} >= ${THRESHOLD}`);
  console.log("  结论: 给定最优 x_0 估计 + 反向随机性, 链把 N(0,I) 噪声搬回双月流形.\n");

  // --- §3: FAILURE MODE — same optimal denoiser, posterior variance forced to 0 (no z) ---
  const { final: collapsed } = runReverseChain(data, sched, /*injectNoise=*/ false, [], N);
  console.log("--- §3 失败模式: 同一最优去噪器 (均值完全正确), 但把后验方差置 0 (去掉随机项 z) ---");
  console.log("  [ 塌缩后的还原点云 ]");
  console.log(scatterASCII(collapsed, 56, 13));
  console.log("");

  const CLUSTER_RADIUS = 0.12;
  const clustersHealthy = countClusters(final, CLUSTER_RADIUS);
  const clustersCollapsed = countClusters(collapsed, CLUSTER_RADIUS);
  const chamferCollapsed = chamferDistance(collapsed, data);
  console.log(`--- §3 量化 (radius=${CLUSTER_RADIUS} 贪心聚类, n=${N}) ---`);
  console.log(`  健康还原 (有 z)   : ${clustersHealthy} 个簇, Chamfer ${chamferRecovered.toFixed(6)}`);
  console.log(`  塌缩 (var=0)      : ${clustersCollapsed} 个簇, Chamfer ${chamferCollapsed.toFixed(6)}`);
  console.log(`  簇数收缩          : ${clustersHealthy} -> ${clustersCollapsed} (${(clustersHealthy / Math.max(clustersCollapsed, 1)).toFixed(1)}x 减少)`);
  console.log(`  Chamfer 恶化      : ${chamferRecovered.toFixed(6)} -> ${chamferCollapsed.toFixed(6)} (${(chamferCollapsed / Math.max(chamferRecovered, 1e-9)).toFixed(1)}x 变差)`);
  console.log("");
  console.log("  机制: 均值正确但去掉每步注入的 z, 反向映射变成确定性收缩 ->");
  console.log("        早期 ᾱ_t≈0 时 x̂_0 退化成数据均值, 没有 z 打破对称, 轨迹一起涌向均值");
  console.log("        = mode collapse 的微缩版. 反向随机性不是噪声, 它维持样本多样性 / 覆盖流形.");
  console.log("  预告 (第 05 章): DDIM 也去掉 z (确定性采样), 但靠每步更准的 x_0 估计绕过塌缩");
  console.log("        — 确定性却不丢多样性. 本章不实现, 仅埋点.");
}

main();
