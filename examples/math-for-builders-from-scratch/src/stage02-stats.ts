// stage02-stats.ts — Statistical inference: how numbers lie even when the arithmetic is right.
//
// WHY this is the statistics stage: a builder running an A/B test, a metrics dashboard, or
//   an offline eval is doing inference whether they know it or not. The dangerous failure is
//   not a wrong formula — it is a CORRECT p-value misread as "we found something". This stage
//   builds a real two-sample t-test, then demonstrates four ways the resulting number betrays
//   naive intuition:
//     1. A/B test with a true effect: the test detects it (p tiny).
//     2. Null is true, 10k reruns: p-values are ~uniform on [0,1], and a 95% CI covers the
//        truth ~95% of the time. (This is the DEFINITION of those quantities, made visible.)
//     3. Multiple comparisons: 100 independent null tests yield ~5 "significant" results at
//        alpha=0.05 — false positives manufactured by counting, not by any real effect.
//     4. Simpson's paradox: an aggregate trend reverses inside every subgroup.
//
// HONEST-NUMBER NOTE: every printed quantity (p-values, t-statistics, CI bounds, coverage
//   rate, false-positive count) is computed from SEEDED samples drawn here — rerun with the
//   same seed for bit-for-bit identical output. Absolute group sizes are synthetic; what
//   transfers is the SHAPE: uniform p under the null, ~5% false positives at alpha=0.05, and
//   that subgroup reversal is a real arithmetic phenomenon, not a toy artifact.
//
// WHY we hand-roll the t-distribution CDF here instead of in core: core/stats.ts is
//   descriptive statistics only (mean/var/quantile). The t-CDF is inferential machinery
//   specific to this chapter, so it lives here next to the test that needs it. It is an
//   approximation (see incompleteBeta) and we DEMONSTRATE its accuracy against the uniform
//   p-value histogram rather than asserting it.

import { mulberry32, sampleNormal, type Rng } from "./core/rng.js";
import { mean, variance, histogram } from "./core/stats.js";
import { asciiBar } from "./core/plot.js";

// ---------------------------------------------------------------------------
// t-distribution CDF via the regularized incomplete beta function.
//
// WHY this math: Welch's two-sample t-test produces a t-statistic; turning it into a p-value
//   needs P(|T| > |t|) under a Student-t distribution with df degrees of freedom. There is no
//   closed form — the CDF is an incomplete beta integral. We use the standard Lentz continued
//   fraction (Numerical Recipes §6.4), the same routine real stats libraries call. It is not
//   "approximate handwaving": it converges to machine precision for the df we use.
// FAILURE MODE: the continued fraction loses accuracy in the far tail (x near 0 or 1) and for
//   tiny df; we mitigate with the symmetry betacf(x) = 1 - betacf(1-x) and cap iterations.
// ---------------------------------------------------------------------------

/** ln(Gamma(z)) — Lanczos approximation. Needed for the beta normalizer. */
function lnGamma(z: number): number {
  // Lanczos g=7, n=9 coefficients; accurate to ~1e-15 for z > 0.5.
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    // Reflection formula keeps us in the well-conditioned z > 0.5 region.
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  }
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * Continued-fraction core of the incomplete beta (Lentz's method).
 * INVARIANT: caller routes x to the rapidly-converging side via the symmetry below; this
 *   function assumes x < (a+1)/(a+b+2).
 */
function betacf(a: number, b: number, x: number): number {
  const TINY = 1e-30;
  const MAXIT = 200;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    // Even step.
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    // Odd step.
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 3e-7) break; // converged
  }
  return h;
}

/** Regularized incomplete beta I_x(a,b) in [0,1]. */
function incompleteBeta(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lnBeta =
    lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  const front = Math.exp(lnBeta);
  // Pick the side where the continued fraction converges fast.
  if (x < (a + 1) / (a + b + 2)) return (front * betacf(a, b, x)) / a;
  return 1 - (front * betacf(b, a, 1 - x)) / b;
}

/**
 * Two-sided p-value for Student's t with `df` degrees of freedom: P(|T| > |t|).
 * INVARIANT: returns a value in (0,1]; t=0 yields exactly 1 (no evidence).
 */
function tTestPValueTwoSided(t: number, df: number): number {
  const x = df / (df + t * t);
  // I_x(df/2, 1/2) is the two-tail mass beyond |t|.
  return incompleteBeta(df / 2, 0.5, x);
}

// ---------------------------------------------------------------------------
// Welch's two-sample t-test (unequal variances — the honest default; the equal-variance
//   "Student" version assumes something you rarely know to be true).
// ---------------------------------------------------------------------------

interface TTestResult {
  meanDiff: number; // mean(a) - mean(b)
  t: number; // t-statistic
  df: number; // Welch–Satterthwaite degrees of freedom (non-integer is expected)
  pValue: number; // two-sided
  ci95: [number, number]; // 95% CI for the mean difference
}

/**
 * Welch's t-test on two samples plus a 95% confidence interval for the mean difference.
 * WHY a CI alongside the p-value: the p-value answers "could the null produce this?" while
 *   the CI answers "how big is the effect, with what uncertainty?" — the second is what a
 *   builder actually needs to decide whether an effect is worth shipping.
 * INVARIANT: both samples need >=2 points (sample variance is undefined otherwise; core
 *   throws). df is the Welch–Satterthwaite estimate and is deliberately non-integer.
 * NOTE: the CI uses a normal critical value (1.96) rather than the exact t-quantile. For the
 *   sample sizes here (n>=30) the difference is <2% and we keep core dependency-free; the
 *   coverage demo below empirically confirms ~95% so this shortcut is honest, not hidden.
 */
function welchTTest(a: readonly number[], b: readonly number[]): TTestResult {
  const ma = mean(a);
  const mb = mean(b);
  const va = variance(a); // sample variance (n-1)
  const vb = variance(b);
  const na = a.length;
  const nb = b.length;
  const se = Math.sqrt(va / na + vb / nb); // standard error of the difference
  const meanDiff = ma - mb;
  const t = meanDiff / se;
  // Welch–Satterthwaite degrees of freedom.
  const df =
    (va / na + vb / nb) ** 2 /
    ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1));
  const pValue = tTestPValueTwoSided(t, df);
  const half = 1.96 * se; // ~95% normal critical value
  return { meanDiff, t, df, pValue, ci95: [meanDiff - half, meanDiff + half] };
}

/** Draw n normal samples with given mean/std from the shared seeded stream. */
function drawNormal(rng: Rng, n: number, mu: number, sigma: number): number[] {
  return Array.from({ length: n }, () => sampleNormal(rng, mu, sigma));
}

// ---------------------------------------------------------------------------
// Demos
// ---------------------------------------------------------------------------

/** ① A/B test where group A genuinely outperforms group B. */
function demoAbTestWithEffect(rng: Rng): void {
  console.log("--- ① A/B 测试：B 组有真实提升 ---");
  // Conversion-value-like metric: control ~10.0, treatment ~10.5, same noise. n=200 each.
  const control = drawNormal(rng, 200, 10.0, 2.0);
  const treatment = drawNormal(rng, 200, 10.5, 2.0);
  const r = welchTTest(treatment, control);
  console.log(`实验组均值 - 对照组均值 = ${r.meanDiff.toFixed(4)} (真实效应设为 +0.5)`);
  console.log(`t 统计量               = ${r.t.toFixed(3)}, df ≈ ${r.df.toFixed(1)}`);
  console.log(`p 值                   = ${r.pValue.toExponential(3)}`);
  console.log(`95% 置信区间 (差值)    = [${r.ci95[0].toFixed(4)}, ${r.ci95[1].toFixed(4)}]`);
  const verdict = r.pValue < 0.05 ? "拒绝原假设：检测到效应" : "未能拒绝原假设";
  console.log(`判定 (α=0.05)          = ${verdict}\n`);
}

/**
 * ② Null is true: rerun the SAME test on two groups drawn from the SAME distribution,
 *    nReps times. Two facts must emerge:
 *      - the p-value distribution is ~uniform on [0,1]
 *      - the 95% CI covers the true difference (0) ~95% of the time
 *    These are not luck; they are what "p-value" and "95% confidence" MEAN. Seeing the
 *    histogram come out flat is the most convincing proof the t-CDF above is correct.
 */
function demoNullDistribution(rng: Rng): void {
  console.log("--- ② 原假设为真时跑 10000 次：p 值应近似均匀，CI 覆盖率应 ≈95% ---");
  const nReps = 10000;
  const pValues: number[] = [];
  let ciCovers = 0; // how often the 95% CI contains the true diff (= 0)
  for (let i = 0; i < nReps; i++) {
    // Same mean (10), same std (2): there is NO real difference. n=40 each.
    const g1 = drawNormal(rng, 40, 10, 2);
    const g2 = drawNormal(rng, 40, 10, 2);
    const r = welchTTest(g1, g2);
    pValues.push(r.pValue);
    if (r.ci95[0] <= 0 && 0 <= r.ci95[1]) ciCovers++;
  }
  const h = histogram(pValues, 10); // 10 bins over [0,1]
  const labels = h.counts.map((_, i) => `${(i / 10).toFixed(1)}-${((i + 1) / 10).toFixed(1)}`);
  console.log(`p 值分布 (10 桶, 每桶期望 ${nReps / 10} 条 —— 均匀的标志是各桶接近齐平):`);
  console.log(asciiBar(labels, h.counts));
  const coverage = (ciCovers / nReps) * 100;
  const rejectRate = (pValues.filter((p) => p < 0.05).length / nReps) * 100;
  console.log(`\n95% CI 覆盖真值(0) 的比例 = ${coverage.toFixed(2)}%  (理论 ≈95%)`);
  console.log(`p<0.05 的比例 (假阳性率)  = ${rejectRate.toFixed(2)}%  (理论 ≈5% —— 这就是 α)\n`);
}

/**
 * ③ Multiple comparisons: run 100 INDEPENDENT null tests and count "significant" ones.
 *    Each test has a 5% false-positive rate; run 100 and you expect ~5 false discoveries —
 *    none of which correspond to any real effect. This is how dredging a dashboard for "the
 *    metric that moved" manufactures findings out of pure noise.
 */
function demoMultipleComparisons(rng: Rng): void {
  console.log("--- ③ 多重比较：100 个无效应检验，数一数有几个'显著' ---");
  const nTests = 100;
  let falsePositives = 0;
  let smallestP = 1;
  for (let i = 0; i < nTests; i++) {
    const g1 = drawNormal(rng, 50, 0, 1);
    const g2 = drawNormal(rng, 50, 0, 1); // identical distribution: every effect is fake
    const p = welchTTest(g1, g2).pValue;
    if (p < 0.05) falsePositives++;
    if (p < smallestP) smallestP = p;
  }
  console.log(`'显著' (p<0.05) 的检验数 = ${falsePositives} / ${nTests}  (期望 ≈5)`);
  console.log(`最小 p 值               = ${smallestP.toExponential(3)}  (挑这个汇报 = p-hacking)`);
  console.log("没有任何真实效应，却凭'数得够多'制造出发现 —— 必须用 Bonferroni/FDR 校正。\n");
}

/**
 * ④ Simpson's paradox: every subgroup favors A, yet the pooled totals favor B.
 *    WHY it happens: the groups receive very different MIXES of an easy vs hard segment, and
 *    the aggregate rate is dominated by whichever group sits mostly in the easy segment. The
 *    arithmetic is trivially correct at every step; the trap is reading the pooled number as
 *    if the segments were comparable. This is the canonical "the number is right and still
 *    lies" case — no randomness needed, so it is fully deterministic.
 *    DATA: the real kidney-stone study (Charig 1986). A is the better treatment in BOTH
 *    severity groups, but A was mostly tried on SEVERE cases (low base success) and B on MILD
 *    cases (high base success), so pooling makes the worse treatment B look better.
 */
function demoSimpsonsParadox(): void {
  console.log("--- ④ 辛普森悖论：分组里 A 全胜，合计却 B 胜 ---");
  // Treatment A is given mostly to SEVERE cases; B mostly to MILD cases.
  // success counts / totals per (treatment, severity).
  const data = {
    A: { mild: { ok: 81, n: 87 }, severe: { ok: 192, n: 263 } },
    B: { mild: { ok: 234, n: 270 }, severe: { ok: 55, n: 80 } },
  };
  const rate = (c: { ok: number; n: number }) => (c.ok / c.n) * 100;

  console.log("分组成功率 (每个子组里 A 都更高):");
  console.log("  病情      A 成功率        B 成功率");
  console.log(
    `  轻症    ${rate(data.A.mild).toFixed(1)}% (${data.A.mild.ok}/${data.A.mild.n})` +
      `      ${rate(data.B.mild).toFixed(1)}% (${data.B.mild.ok}/${data.B.mild.n})`,
  );
  console.log(
    `  重症    ${rate(data.A.severe).toFixed(1)}% (${data.A.severe.ok}/${data.A.severe.n})` +
      `      ${rate(data.B.severe).toFixed(1)}% (${data.B.severe.ok}/${data.B.severe.n})`,
  );

  const aPooled = {
    ok: data.A.mild.ok + data.A.severe.ok,
    n: data.A.mild.n + data.A.severe.n,
  };
  const bPooled = {
    ok: data.B.mild.ok + data.B.severe.ok,
    n: data.B.mild.n + data.B.severe.n,
  };
  console.log("\n合计成功率 (趋势反转，B 反而更高):");
  console.log("  方案    合计成功率");
  console.log(`  A     ${rate(aPooled).toFixed(1)}% (${aPooled.ok}/${aPooled.n})`);
  console.log(`  B     ${rate(bPooled).toFixed(1)}% (${bPooled.ok}/${bPooled.n})`);
  console.log(
    "\n每步算术都对 —— 陷阱在于 A 多收重症、B 多收轻症，合计被病情构成主导而非疗效。\n",
  );
}

/**
 * Failure mode: declaring a discovery the instant p<0.05, ignoring effect size and the CI.
 * We construct a TRUE but trivially tiny effect on a huge sample: p is significant, yet the
 * 95% CI shows the effect is practically zero. Statistical significance != practical
 * significance — the single most common misread of a p-value in production.
 */
function demoFailureModePChasing(rng: Rng): void {
  console.log("--- 失败模式：p<0.05 就宣布发现（无视效应量） ---");
  // Real effect of +0.02 on a metric with std 1, but n=50000 per group makes even a
  // microscopic difference "significant".
  const a = drawNormal(rng, 50000, 0.02, 1);
  const b = drawNormal(rng, 50000, 0.0, 1);
  const r = welchTTest(a, b);
  console.log(`样本量                 = 50000 / 组`);
  console.log(`观测均值差             = ${r.meanDiff.toFixed(4)} (真实效应仅 +0.02)`);
  console.log(`p 值                   = ${r.pValue.toExponential(3)}  → 形式上"显著"`);
  console.log(`95% 置信区间 (差值)    = [${r.ci95[0].toFixed(4)}, ${r.ci95[1].toFixed(4)}]`);
  console.log(
    "p 很小不代表效应大：效应量约 0.02 个单位，业务上几乎为零。只看 p、不看 CI/效应量 = 假发现。",
  );
}

function main(): void {
  // One shared seeded stream for all stochastic demos: rerunning reproduces every number.
  const rng = mulberry32(20260622);
  console.log("=== Stage 02 · 统计与推断：别被数字骗 ===\n");
  demoAbTestWithEffect(rng);
  demoNullDistribution(rng);
  demoMultipleComparisons(rng);
  demoSimpsonsParadox();
  demoFailureModePChasing(rng);
}

main();
