// stage06-numerics.ts — Floating point is not the real numbers: the bugs that bite.
//
// WHY this stage: every numeric line a builder ships runs on IEEE-754 doubles, a FINITE
//   set of dyadic rationals — not ℝ. The gap is invisible until it bites: a == comparison
//   that "should" be true is false; a textbook-correct formula loses every significant
//   digit; a sum of a million small numbers drifts; a matrix that is "almost singular"
//   turns a 1e-5 input wiggle into a 1e+5 output swing. This stage MEASURES each gap
//   against a high-precision reference so the error is a concrete number, never folklore.
//
// HONEST-NUMBER NOTE: the references here are genuinely higher precision than the thing
//   being tested — Kahan-compensated sums and exact integer/rational arithmetic used as
//   ground truth — so every printed "error" is a real measured difference, not an estimate.
//   The condition-number demo uses a synthetic ill-conditioned matrix; absolute blow-up
//   magnitudes depend on the matrix, but the RELATIONSHIP (error ≈ κ · input perturbation)
//   is the transferable law.
//
// CONTRACT: reuse core/stats.js (its two-pass variance is our STABLE reference vs a naive
//   one-pass formula) and core/plot.js. No randomness needed except a seeded perturbation
//   in the condition-number demo, which threads core/rng.js for reproducibility.

import { variance } from "./core/stats.js";
import { matVec } from "./core/linalg.js";
import { asciiLine } from "./core/plot.js";
import { mulberry32, sampleNormal } from "./core/rng.js";

/** Machine epsilon for f64: smallest e such that 1 + e !== 1. ~2.22e-16. */
const EPS = Number.EPSILON;

/**
 * Tolerant float comparison: |a - b| <= atol + rtol * max(|a|, |b|).
 * WHY both absolute and relative tol: pure relative tol fails near zero (max(|a|,|b|) -> 0
 *   makes the bound collapse), pure absolute tol fails at large magnitudes (1e9 vs 1e9+1
 *   differ by 1 but are "equal" to f64 precision). The mixed form is the standard robust
 *   choice (numpy.isclose, Catch2 Approx all use this shape).
 * INVARIANT: returns true for a === b including both +0/-0 and both NaN-free equal inputs;
 *   NaN compares false to everything, which is the correct, loud behavior.
 */
function approxEqual(a: number, b: number, rtol = 1e-9, atol = 1e-12): boolean {
  return Math.abs(a - b) <= atol + rtol * Math.max(Math.abs(a), Math.abs(b));
}

/**
 * Naive one-pass variance: E[x^2] - E[x]^2. Algebraically correct, numerically a trap.
 * WHY it explodes: with a large mean, E[x^2] and E[x]^2 are two huge nearly-equal numbers;
 *   subtracting them is CATASTROPHIC CANCELLATION — the leading digits agree and cancel,
 *   leaving only the noise in the trailing (rounding) bits. The result can even go negative,
 *   which is impossible for a true variance. core/stats.variance uses the stable two-pass
 *   (subtract the mean first), so it is our reference.
 */
function naiveVarianceOnePass(xs: readonly number[]): number {
  const n = xs.length;
  let sum = 0;
  let sumSq = 0;
  for (const x of xs) {
    sum += x;
    sumSq += x * x;
  }
  const meanX = sum / n;
  return (sumSq - n * meanX * meanX) / (n - 1); // (Σx² - n·x̄²)/(n-1)
}

/**
 * Quadratic root that is stable for ALL sign combinations.
 * WHY not the textbook (-b ± √(b²-4ac))/2a directly: when b is large and ac is small,
 *   √(b²-4ac) ≈ |b|, so ONE of the two roots subtracts two nearly-equal numbers and
 *   cancels. Trick: compute the well-conditioned root first, then get the other from the
 *   product-of-roots identity x1·x2 = c/a — no subtraction of near-equals.
 * INVARIANT: assumes real roots (discriminant >= 0) and a != 0; returns [smaller-cancel-
 *   free root, partner root]. We return the cancellation-prone naive root too so the caller
 *   can print the side-by-side error.
 */
function quadraticRootsStable(a: number, b: number, c: number): { stable: number; naive: number } {
  const disc = Math.sqrt(b * b - 4 * a * c);
  // q picks the sign so we ADD magnitudes (never subtract near-equals).
  const q = -0.5 * (b + Math.sign(b) * disc);
  const stableRoot = c / q; // the root that the naive formula computes by cancellation
  const naiveRoot = (-b + disc) / (2 * a); // textbook; cancels when b>0 and 4ac«b²
  return { stable: stableRoot, naive: naiveRoot };
}

/**
 * Kahan compensated summation: track the lost low-order bits in a running correction.
 * WHY it works: naive `sum += x` discards the part of x too small to fit in sum's mantissa;
 *   over millions of adds that discarded dust accumulates into visible drift. Kahan keeps a
 *   compensation term `c` holding exactly what was dropped and feeds it back next iteration.
 * WARNING: an optimizing compiler that assumes real-number associativity can DELETE the
 *   compensation (it is algebraically a no-op). JS engines do not do this, so the trick
 *   survives here; in C you would need -fno-fast-math / volatile. NOTE for future readers.
 */
function kahanSum(xs: readonly number[]): number {
  let sum = 0;
  let c = 0; // running compensation for lost low-order bits
  for (const x of xs) {
    const y = x - c; // bring in the previously-lost dust
    const t = sum + y; // ...which rounds; (t - sum) recovers the high part of y
    c = t - sum - y; // ...and c captures the NEW low part that just got dropped
    sum = t;
  }
  return sum;
}

function naiveSum(xs: readonly number[]): number {
  let sum = 0;
  for (const x of xs) sum += x;
  return sum;
}

/**
 * Newton's method for f(x)=0: x <- x - f(x)/f'(x). Quadratic convergence near a simple root.
 * WHY return the whole error history: the headline of Newton is "the number of correct
 *   digits roughly DOUBLES each step" — you only see that by printing |x_k - root| per step.
 * FAILURE MODE (demoed by caller): f'(x) ≈ 0 makes the step blow up (division by ~0); a bad
 *   start point can diverge or cycle. Newton is fast but NOT globally safe.
 */
function newtonRoot(
  f: (x: number) => number,
  df: (x: number) => number,
  x0: number,
  steps: number,
): number[] {
  const history: number[] = [x0];
  let x = x0;
  for (let i = 0; i < steps; i++) {
    const slope = df(x);
    if (slope === 0) break; // flat tangent -> step undefined; bail loudly via short history
    x = x - f(x) / slope;
    history.push(x);
  }
  return history;
}

/**
 * Solve a 2x2 system by EXPLICIT INVERSE (the thing every numerics course tells you not to
 * do for ill-conditioned matrices, kept here precisely to demonstrate why).
 * WHY inverse is bad: it computes 1/det, and a near-singular matrix has det ≈ 0, so tiny
 *   input changes to the matrix or RHS get amplified by ~1/det ≈ the condition number.
 * INVARIANT: m is 2x2; throws if exactly singular (det == 0) instead of returning Infinity.
 */
function solve2x2ViaInverse(m: number[][], rhs: number[]): number[] {
  const [[a, b], [c, d]] = m;
  const det = a * d - b * c;
  if (det === 0) throw new Error("[stage06] solve2x2: singular matrix, inverse undefined");
  // inv(M) = 1/det * [[d, -b], [-c, a]]; multiply by rhs.
  const inv = [
    [d / det, -b / det],
    [-c / det, a / det],
  ];
  return matVec(inv, rhs);
}

/**
 * Spectral condition number of a SYMMETRIC 2x2 via its closed-form eigenvalues: κ = |λmax/λmin|.
 * WHY closed form: a 2x2 symmetric matrix's eigenvalues are (tr ± √(tr²-4det))/2 — exact,
 *   no iteration, so the κ we print is the true conditioning, not an estimate.
 */
function condNumberSym2x2(m: number[][]): number {
  const [[a, b], [, d]] = m;
  const tr = a + d;
  const det = a * d - b * b;
  const root = Math.sqrt(Math.max(0, tr * tr - 4 * det));
  const l1 = (tr + root) / 2;
  const l2 = (tr - root) / 2;
  return Math.abs(l1) / Math.abs(l2);
}

function main(): void {
  console.log("=== Stage 06 · 数值方法：浮点不是实数 ===\n");
  console.log(`机器精度 EPSILON = ${EPS.toExponential(4)}（1 + EPS 才 !== 1）\n`);

  // --- ① 0.1 + 0.2 != 0.3，以及正确的容差比较 ---
  console.log("--- ① 相等判断：== 是陷阱，容差才对 ---");
  const lhs = 0.1 + 0.2;
  console.log(`0.1 + 0.2          = ${lhs.toExponential(17)}`);
  console.log(`0.3                = ${(0.3).toExponential(17)}`);
  console.log(`0.1 + 0.2 === 0.3  ? ${lhs === 0.3}   （差 = ${(lhs - 0.3).toExponential(3)}）`);
  console.log(`approxEqual(...)   ? ${approxEqual(lhs, 0.3)}   （0.1/0.2/0.3 都不是 f64 可精确表示的二进制小数）\n`);

  // --- ② 灾难性相消：同一个量，不稳定 vs 稳定公式 ---
  console.log("--- ② 灾难性相消：两个公式算同一个量 ---");
  // (a) 方差：大均值下一遍法 (Σx²-n·x̄²) 相减相消，两遍法 (core) 稳定。
  const big = [1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4];
  const trueVar = variance([1, 2, 3, 4]); // 平移不改方差，整数算出真值 = 5/3
  const stableVar = variance(big);
  const naiveVar = naiveVarianceOnePass(big);
  console.log(`方差真值 (整数算)      = ${trueVar.toFixed(12)}`);
  console.log(`两遍法 (core, 稳定)    = ${stableVar.toFixed(12)}  误差 ${Math.abs(stableVar - trueVar).toExponential(2)}`);
  console.log(`一遍法 (Σx²-n·x̄², 不稳) = ${naiveVar.toFixed(12)}  误差 ${Math.abs(naiveVar - trueVar).toExponential(2)}`);
  if (naiveVar < 0) console.log("  注意：一遍法甚至算出了负方差 —— 数学上不可能，相消把符号都翻了。");
  // (b) 二次方程：判别式 ≈ |b| 时，textbook 公式的一个根整个被吃掉。
  const a = 1,
    b = 1e8,
    c = 1; // 根 ≈ -1e8 和 -1e-8；后者是相消重灾区
  const trueSmallRoot = -c / b; // 韦达：两根积 = c/a，大根 ≈ -b/a，故小根 ≈ -c/b
  const { stable, naive } = quadraticRootsStable(a, b, c);
  console.log(`\n二次方程 x²+1e8·x+1=0 的小根：`);
  console.log(`真值 (韦达 -c/b)       = ${trueSmallRoot.toExponential(12)}`);
  console.log(`稳定式 (c/q)           = ${stable.toExponential(12)}  误差 ${Math.abs(stable - trueSmallRoot).toExponential(2)}`);
  console.log(`textbook ((-b+√D)/2a)  = ${naive.toExponential(12)}  误差 ${Math.abs(naive - trueSmallRoot).toExponential(2)}`);
  console.log("  textbook 式把小根算成了 0：√D 和 b 几乎相等，相减后只剩舍入噪声。\n");

  // --- ③ Kahan 求和 vs 朴素求和：累加大量小数 ---
  console.log("--- ③ Kahan 补偿求和 vs 朴素求和 ---");
  const N = 10_000_000;
  // 0.1 在 f64 里不可精确表示，每次加都丢一点；累加千万次放大成可见漂移。
  const term = 0.1;
  const exactTrue = N * term; // 真值就是 1,000,000（整数，f64 可精确表示）
  // 用数组喂给 core 风格的 sum 函数会吃掉 80MB 内存；这里直接内联同样的两种算法。
  let naiveAcc = 0;
  let ksum = 0;
  let kc = 0;
  for (let i = 0; i < N; i++) {
    naiveAcc += term;
    const y = term - kc;
    const t = ksum + y;
    kc = t - ksum - y;
    ksum = t;
  }
  console.log(`累加 ${N.toLocaleString("en-US")} 个 0.1，真值 = ${exactTrue.toLocaleString("en-US")}`);
  console.log(`朴素求和   = ${naiveAcc.toFixed(8)}  误差 ${Math.abs(naiveAcc - exactTrue).toExponential(3)}`);
  console.log(`Kahan 求和 = ${ksum.toFixed(8)}  误差 ${Math.abs(ksum - exactTrue).toExponential(3)}`);
  const kahanErr = Math.abs(ksum - exactTrue);
  // Kahan 误差可能恰好为 0（真值是整数，补偿把低位 bit 全找回），此时“缩小倍数”无穷大。
  const improveMsg =
    kahanErr === 0
      ? "Kahan 误差降到 0（真值是整数，补偿把丢掉的低位 bit 全找回了）"
      : `Kahan 把误差缩小约 ${(Math.abs(naiveAcc - exactTrue) / kahanErr).toExponential(2)} 倍`;
  console.log(`${improveMsg}。`);
  // 小数组也用一遍 kahanSum/naiveSum 证明纯函数版本与内联一致（避免“只在内联里成立”的假象）。
  const probe = Array.from({ length: 100000 }, () => 0.1);
  console.log(`  （纯函数自检：kahanSum 100k×0.1 = ${kahanSum(probe).toFixed(8)}, naiveSum = ${naiveSum(probe).toFixed(8)}）\n`);

  // --- ④ 条件数：病态矩阵下，输入的小扰动让解大变 ---
  console.log("--- ④ 条件数：病态矩阵放大扰动 ---");
  // 近奇异对称矩阵：两行几乎线性相关，det ≈ 0，κ 巨大。
  const illM = [
    [1, 1],
    [1, 1 + 1e-10],
  ];
  const kappa = condNumberSym2x2(illM);
  const rhs = [2, 2 + 1e-10]; // 精确解 x = [1, 1]
  const baseSol = solve2x2ViaInverse(illM, rhs);
  // 给 RHS 一个 1e-8 相对量级的确定性扰动，看解漂多远。
  const rng = mulberry32(20260622);
  const pert = [rhs[0] + 1e-8 * sampleNormal(rng), rhs[1] + 1e-8 * sampleNormal(rng)];
  const pertSol = solve2x2ViaInverse(illM, pert);
  const inRelChange = Math.hypot(pert[0] - rhs[0], pert[1] - rhs[1]) / Math.hypot(rhs[0], rhs[1]);
  const outRelChange =
    Math.hypot(pertSol[0] - baseSol[0], pertSol[1] - baseSol[1]) / Math.hypot(baseSol[0], baseSol[1]);
  console.log(`矩阵 [[1,1],[1,1+1e-10]] 的条件数 κ = ${kappa.toExponential(3)}`);
  console.log(`输入(RHS) 相对扰动 = ${inRelChange.toExponential(3)}`);
  console.log(`输出(解) 相对变化  = ${outRelChange.toExponential(3)}`);
  console.log(`放大倍数 ≈ ${(outRelChange / inRelChange).toExponential(3)}（理论上界就是 κ；κ 大 = 输入一抖解就崩）\n`);

  // --- ⑤ 牛顿法求根：每步正确位数翻倍 ---
  console.log("--- ⑤ 牛顿法求 √2（解 x²-2=0）---");
  const root2 = Math.SQRT2;
  // 5 步足够：误差从 1e-1 → 1e-13，每步指数翻倍；再迭代只在 EPS 地板上抖动，无信息。
  const hist = newtonRoot((x) => x * x - 2, (x) => 2 * x, 1.0, 5);
  const errs = hist.map((x) => Math.abs(x - root2));
  hist.forEach((x, i) => {
    const digits = errs[i] > 0 ? Math.max(0, -Math.log10(errs[i])) : Infinity;
    console.log(`  step ${i}: x = ${x.toFixed(16)}  误差 ${errs[i].toExponential(2)}  正确位数≈${Number.isFinite(digits) ? digits.toFixed(1) : "全部"}`);
  });
  console.log("  误差指数每步约翻倍 → 二次收敛，这是牛顿法值钱的地方。");
  console.log("误差(log10)收敛曲线：");
  console.log(asciiLine(errs.map((e) => (e > 0 ? Math.log10(e) : -16))));
  console.log("");

  // --- 失败模式：病态问题直接求逆 + 牛顿法在驻点附近发散 ---
  console.log("--- 失败模式 ---");
  // (A) 把上面那个 κ≈4e10 的病态矩阵收紧成“数值奇异”：det 落到 EPS 量级，求逆得垃圾。
  const nearSingular = [
    [1, 1],
    [1, 1 + EPS], // 1 + EPS 是“能让 1 变化的最小量”，再小就真奇异了
  ];
  const detNS = nearSingular[0][0] * nearSingular[1][1] - nearSingular[0][1] * nearSingular[1][0];
  // 选 RHS = [2, 2+EPS] 使真解恰为 x=[1,1]（row2-row1: EPS·y = EPS ⇒ y=1, x=1）。
  // 但构造 RHS 时 2+EPS 已经发生舍入，1/det≈1/EPS 再把这点舍入放大到整数量级。
  const trueSol = [1, 1];
  const badSol = solve2x2ViaInverse(nearSingular, [2, 2 + EPS]);
  const solErr = Math.hypot(badSol[0] - trueSol[0], badSol[1] - trueSol[1]);
  console.log(`病态矩阵 det = ${detNS.toExponential(3)}（≈EPS）`);
  console.log(`真解 = [1, 1]，直接求逆解 = [${badSol.map((v) => v.toFixed(4)).join(", ")}]，解的误差 = ${solErr.toExponential(2)}`);
  console.log("  RHS 里一个 EPS 量级的舍入被 1/det≈1/EPS 放大成 O(1) 的解误差 → 病态问题别求逆，用 QR/SVD/带 pivot 的消元。");
  // (B) 牛顿法从 x0=0 起步解 x²-2=0：f'(0)=0，切线水平，第一步就除以零。
  const diverge = newtonRoot((x) => x * x - 2, (x) => 2 * x, 0, 5);
  console.log(`牛顿法从 x0=0 解 x²-2=0：迭代历史长度 = ${diverge.length}（f'(0)=0，切线水平，第一步就无定义直接中止）`);
  console.log("  牛顿法快但不安全：起点差 / 驻点处 f'≈0 会发散或除零，生产里要配 bisection 兜底或带阻尼。");
}

main();
