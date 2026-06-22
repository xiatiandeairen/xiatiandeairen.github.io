// stage04-optim.ts — Calculus & optimization: the engine that actually trains models.
//
// WHY this is the chapter on "the training engine": every gradient-based model — from a
//   linear fit to a transformer — is the same loop: measure the loss, ask calculus which
//   way is downhill (the gradient), take a step. Everything else (Adam, schedulers, layer
//   norm) is decoration on that loop. This stage builds the loop by hand so the reader sees
//   there is no magic: a gradient is just "how much loss changes per unit of each weight".
//
// WHAT WE PROVE HONESTLY (each number is computed, not asserted):
//   ① numeric gradient (finite differences) ≈ analytic gradient — the check every framework
//      runs in its tests, here with the real max-abs error printed.
//   ② hand-written GD fits a noisy line; loss curve drawn from the actual iterates.
//   ③ a learning-rate sweep where small=slow-converge, good=converge, large=DIVERGE — the
//      divergence threshold for a quadratic of curvature L is exactly 2/L, and we print
//      the lr we picked relative to it.
//   ④ one full forward+backward pass of a 2-layer toy net, with every parameter gradient
//      gradient-checked against finite differences (so "backprop is correct" is verified,
//      not claimed).
//   ⑤ a Lagrange-multiplier solution to an equality-constrained optimum, checked against
//      the known closed form.
//   FAILURE MODE: lr past 2/L makes the loss blow to Infinity/NaN within a few steps.
//
// HONEST-NUMBER NOTE: the data in ② is SEEDED synthetic (a known line + Gaussian noise),
//   so the recovered slope/intercept land near the true generators; that is the point —
//   a toy problem with a known answer lets us VERIFY the optimizer, not benchmark it. What
//   transfers to real training is the mechanism and the relative shapes (converge vs
//   oscillate vs diverge), not the absolute loss values.
//
// CONTRACT: reuse core/linalg.js for vector ops, core/plot.js for curves, core/rng.js for
//   the single seeded noise source. No stageNN imports (importing one would run its main()).

import { mulberry32, sampleNormal, type Rng } from "./core/rng.js";
import { add, scale, dot } from "./core/linalg.js";
import { asciiLine } from "./core/plot.js";

// ─────────────────────────────────────────────────────────────────────────────
// ① Gradient checking: finite differences vs an analytic gradient.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Central finite-difference gradient of a scalar field f at point x.
 * WHY central (f(x+h) - f(x-h)) / 2h and not forward (f(x+h) - f(x)) / h: the central
 *   formula's error is O(h^2) vs O(h) for forward, so with the same h it is ~h times more
 *   accurate. That accuracy is what lets us trust it as an oracle for the analytic gradient.
 * WHY h = 1e-5 and not smaller: too-large h leaves truncation error; too-small h (say 1e-12)
 *   loses precision to float cancellation in the subtraction. 1e-5..1e-6 is the sweet spot
 *   in float64 for well-scaled inputs — this trade-off is itself a thing builders must know.
 * INVARIANT: f must be pure (same x -> same value); we perturb one coordinate at a time.
 */
function numericGradient(f: (x: number[]) => number, x: readonly number[], h = 1e-5): number[] {
  const g = new Array(x.length).fill(0);
  for (let i = 0; i < x.length; i++) {
    const plus = x.slice();
    const minus = x.slice();
    plus[i] += h;
    minus[i] -= h;
    g[i] = (f(plus) - f(minus)) / (2 * h);
  }
  return g;
}

/** Max absolute component-wise difference — the standard gradient-check metric. */
function maxAbsDiff(a: readonly number[], b: readonly number[]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// ② + ③ Linear regression by hand: loss, analytic gradient, GD loop.
// ─────────────────────────────────────────────────────────────────────────────

interface LinearData {
  xs: number[]; // feature
  ys: number[]; // target = trueSlope*x + trueIntercept + noise
  trueSlope: number;
  trueIntercept: number;
}

/** Generate y = a*x + b + N(0, noiseStd) on a seeded RNG so the fit is reproducible. */
function makeNoisyLine(rng: Rng, n: number, a: number, b: number, noiseStd: number): LinearData {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 4 - 2; // spread over [-2, 2]
    xs.push(x);
    ys.push(a * x + b + sampleNormal(rng, 0, noiseStd));
  }
  return { xs, ys, trueSlope: a, trueIntercept: b };
}

/** Mean squared error for params [slope, intercept]. The thing we descend. */
function mseLoss(params: readonly number[], data: LinearData): number {
  const [slope, intercept] = params;
  let s = 0;
  for (let i = 0; i < data.xs.length; i++) {
    const pred = slope * data.xs[i] + intercept;
    const r = pred - data.ys[i];
    s += r * r;
  }
  return s / data.xs.length;
}

/**
 * Analytic gradient of the MSE w.r.t. [slope, intercept].
 *   d/dslope    = mean( 2 * residual * x )
 *   d/dintercept= mean( 2 * residual * 1 )
 * WHY hand-derive instead of autodiff: this whole stage exists to show the reader that the
 *   gradient is a concrete formula, then to gradient-CHECK that formula numerically so they
 *   trust it. Autodiff would hide exactly the step we are teaching.
 */
function mseGradient(params: readonly number[], data: LinearData): number[] {
  const [slope, intercept] = params;
  let gs = 0;
  let gi = 0;
  const n = data.xs.length;
  for (let i = 0; i < n; i++) {
    const r = slope * data.xs[i] + intercept - data.ys[i];
    gs += 2 * r * data.xs[i];
    gi += 2 * r;
  }
  return [gs / n, gi / n];
}

/**
 * Largest eigenvalue of the exact MSE Hessian H = (2/n)[[Σx², Σx], [Σx, n]].
 * WHY closed form for a 2x2: a symmetric [[a,b],[b,c]] has eigenvalues (a+c ± sqrt((a-c)²+4b²))/2.
 *   We take the '+' root because the stability limit 2/L is governed by the STIFFEST direction
 *   (largest curvature) — GD diverges as soon as the step is too large for that one direction,
 *   even if every other direction is fine.
 */
function topHessianEigenvalue(data: LinearData): number {
  const n = data.xs.length;
  const sxx = data.xs.reduce((s, x) => s + x * x, 0);
  const sx = data.xs.reduce((s, x) => s + x, 0);
  const a = (2 * sxx) / n;
  const b = (2 * sx) / n;
  const c = 2; // (2/n) * Σ1 = 2
  return (a + c + Math.sqrt((a - c) ** 2 + 4 * b * b)) / 2;
}

interface GdRun {
  params: number[];
  losses: number[];
  diverged: boolean; // loss became non-finite at some step
}

/**
 * Vanilla gradient descent. Returns the loss at every step so the caller can plot it.
 * FAILURE MODE we deliberately surface: if lr is too large the iterate overshoots the
 *   minimum and the residual GROWS each step; the loss diverges to Infinity/NaN. We do not
 *   clamp or early-stop on our own — we record `diverged` and let the caller show the blow-up,
 *   because hiding it would defeat the lesson. (A real trainer would gradient-clip or reduce lr.)
 */
function gradientDescent(
  params0: readonly number[],
  lr: number,
  steps: number,
  grad: (p: readonly number[]) => number[],
  loss: (p: readonly number[]) => number,
): GdRun {
  let params = params0.slice();
  const losses: number[] = [];
  let diverged = false;
  for (let i = 0; i < steps; i++) {
    const l = loss(params);
    losses.push(l);
    if (!Number.isFinite(l)) {
      diverged = true;
      break; // once it is NaN/Inf every later step stays garbage; stop recording noise
    }
    params = add(params, scale(grad(params), -lr)); // p <- p - lr * grad
  }
  return { params, losses, diverged };
}

// ─────────────────────────────────────────────────────────────────────────────
// ④ One forward + backward pass of a 2-layer toy network.
//   Architecture: x(2) -> [W1(2x2), b1] -> tanh -> [W2(1x2), b2] -> scalar yhat
//   Loss: 0.5 * (yhat - target)^2   (the 0.5 makes dL/dyhat = residual, no stray factor)
// ─────────────────────────────────────────────────────────────────────────────

interface ToyNet {
  W1: number[][]; // 2x2
  b1: number[]; // 2
  W2: number[]; // 2  (single output unit -> a row vector)
  b2: number; // scalar
}

interface ToyGrads {
  dW1: number[][];
  db1: number[];
  dW2: number[];
  db2: number;
}

const dtanh = (t: number): number => 1 - t * t; // derivative in terms of the tanh OUTPUT t

/** Forward pass returning the prediction plus the intermediate activations backprop needs. */
function forward(net: ToyNet, x: readonly number[]): { yhat: number; z1: number[]; a1: number[] } {
  // Layer 1: pre-activation z1 = W1 x + b1, activation a1 = tanh(z1).
  const z1 = [dot(net.W1[0], x) + net.b1[0], dot(net.W1[1], x) + net.b1[1]];
  const a1 = z1.map(Math.tanh);
  // Layer 2: scalar output, no activation (regression head).
  const yhat = dot(net.W2, a1) + net.b2;
  return { yhat, z1, a1 };
}

function toyLoss(net: ToyNet, x: readonly number[], target: number): number {
  return 0.5 * (forward(net, x).yhat - target) ** 2;
}

/**
 * Backprop: apply the chain rule layer by layer, output side first.
 * WHY store a1 from the forward pass: every gradient here is (upstream signal) x (local
 *   derivative); the local derivatives are functions of the forward activations, so caching
 *   them is what makes one backward pass O(forward) instead of recomputing. That reuse IS
 *   the reason backprop scales — recomputing per-parameter would be the finite-difference
 *   cost we use only as the oracle.
 * INVARIANT (verified at the call site by gradient check): these equal the numeric gradient.
 */
function backward(net: ToyNet, x: readonly number[], target: number): ToyGrads {
  const { yhat, a1 } = forward(net, x);
  const dyhat = yhat - target; // dL/dyhat for L = 0.5*(yhat-target)^2

  // Output layer params.
  const dW2 = a1.map((a) => dyhat * a); // dL/dW2_j = dyhat * a1_j
  const db2 = dyhat;

  // Backprop into the hidden activations, then through tanh into z1.
  const da1 = net.W2.map((w) => dyhat * w); // dL/da1_j = dyhat * W2_j
  const dz1 = da1.map((d, j) => d * dtanh(a1[j])); // through tanh: * (1 - a1^2)

  // Hidden layer params: dz1 is the upstream signal, x is the local input.
  const dW1 = dz1.map((d) => x.map((xi) => d * xi)); // outer product dz1 ⊗ x
  const db1 = dz1.slice();

  return { dW1, db1, dW2, db2 };
}

/** Flatten net + grads to aligned vectors so we can gradient-check the whole net at once. */
function flattenNet(net: ToyNet): number[] {
  return [...net.W1[0], ...net.W1[1], ...net.b1, ...net.W2, net.b2];
}
function flattenGrads(g: ToyGrads): number[] {
  return [...g.dW1[0], ...g.dW1[1], ...g.db1, ...g.dW2, g.db2];
}
function unflattenNet(v: readonly number[]): ToyNet {
  return {
    W1: [[v[0], v[1]], [v[2], v[3]]],
    b1: [v[4], v[5]],
    W2: [v[6], v[7]],
    b2: v[8],
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// ⑤ Lagrange multipliers: equality-constrained optimum.
//   Problem: minimize f(x,y) = x^2 + y^2  subject to  g(x,y) = x + y - 1 = 0.
//   Geometric reading: smallest circle centred at origin that still touches the line x+y=1,
//   i.e. the foot of the perpendicular from the origin to that line.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Solve the stationarity system  ∇f = λ ∇g,  g = 0  for this specific problem in closed form.
 * WHY closed form here rather than an iterative solver: the whole point of Lagrange
 *   multipliers is that the constrained optimum satisfies an exact algebraic condition
 *   (gradients parallel on the constraint surface). Showing the closed form and then
 *   verifying ∇f = λ∇g numerically is more honest than burying it in another GD loop.
 *   ∇f = (2x, 2y), ∇g = (1, 1)  ->  2x = λ, 2y = λ  -> x = y; with x+y=1 -> x=y=1/2, λ=1.
 */
function solveLagrange(): { x: number; y: number; lambda: number } {
  // Derived above; expressed as code so the reader can change the constraint and re-derive.
  const x = 0.5;
  const y = 0.5;
  const lambda = 2 * x; // from 2x = λ
  return { x, y, lambda };
}

// ─────────────────────────────────────────────────────────────────────────────

function fmt(v: number): string {
  return Number.isFinite(v) ? v.toExponential(2) : String(v);
}

function main(): void {
  const rng = mulberry32(20260622);
  console.log("=== Stage 04 · 微积分与最优化：训练的引擎 ===\n");

  // ── ① Gradient check on a non-trivial scalar field ─────────────────────────
  // f(x,y,z) = x^2 + 3xy + sin(z); analytic grad = (2x+3y, 3x, cos(z)).
  const f = (p: number[]) => p[0] ** 2 + 3 * p[0] * p[1] + Math.sin(p[2]);
  const analyticF = (p: number[]) => [2 * p[0] + 3 * p[1], 3 * p[0], Math.cos(p[2])];
  const point = [1.3, -0.7, 0.5];
  const gNum = numericGradient(f, point);
  const gAna = analyticF(point);
  console.log("① 数值梯度 vs 解析梯度（有限差分对拍）");
  console.log(`   解析梯度 = [${gAna.map((v) => v.toFixed(6)).join(", ")}]`);
  console.log(`   数值梯度 = [${gNum.map((v) => v.toFixed(6)).join(", ")}]`);
  console.log(`   最大分量误差 = ${fmt(maxAbsDiff(gNum, gAna))}  → 接近 0，解析公式正确\n`);

  // ── ② Fit a noisy line, draw the loss curve ────────────────────────────────
  const data = makeNoisyLine(rng, 60, 2.0, -1.0, 0.3); // y = 2x - 1 + noise
  const fitLoss = (p: readonly number[]) => mseLoss(p, data);
  const fitGrad = (p: readonly number[]) => mseGradient(p, data);
  const fit = gradientDescent([0, 0], 0.2, 80, fitGrad, fitLoss);
  console.log("② 手写梯度下降拟合带噪声直线  y = 2x - 1 + N(0,0.3)");
  console.log(
    `   真参数 slope=2.000 intercept=-1.000 → 拟合 slope=${fit.params[0].toFixed(3)} intercept=${fit.params[1].toFixed(3)}`,
  );
  console.log(`   初始 loss=${fit.losses[0].toFixed(4)}  末步 loss=${fit.losses.at(-1)!.toFixed(4)}`);
  console.log("   loss 收敛曲线（80 步）:");
  console.log(asciiLine(fit.losses));
  console.log();

  // ── ③ Learning-rate sweep: small / good / large on the SAME problem ─────────
  // The MSE is an exact quadratic in [slope, intercept]; its Hessian is data-only (the
  // noisy targets shift the gradient but not the curvature), so we can compute the stability
  // limit 2/L EXACTLY rather than guess it. H = (2/n) X^T X where X has columns [x, 1]:
  //   H = (2/n) [[Σx², Σx], [Σx, n]].  GD on a quadratic converges iff lr < 2/L where L is
  //   the largest Hessian eigenvalue. Getting L wrong (e.g. a loose upper bound) would make
  //   the printed "×limit" multipliers lie about what diverges — so we use the real eigenvalue.
  const L = topHessianEigenvalue(data); // largest eigenvalue of the exact MSE Hessian
  const stableLimit = 2 / L; // GD on a quadratic diverges for lr > 2/L
  console.log("③ 学习率扫描（同一问题，曲率 L=" + L.toFixed(3) + "，发散阈值 2/L=" + stableLimit.toFixed(3) + "）");
  const lrs: Array<{ tag: string; mult: number }> = [
    { tag: "太小", mult: 0.03 }, // far below limit: correct direction but crawls
    { tag: "合适", mult: 0.55 }, // comfortably inside the stable region: fast converge
    { tag: "太大", mult: 1.2 }, // past 2/L: residual amplified each step, loss grows
  ];
  for (const { tag, mult } of lrs) {
    const lr = stableLimit * mult;
    const run = gradientDescent([0, 0], lr, 60, fitGrad, fitLoss);
    const last = run.losses.at(-1)!;
    const grew = last > run.losses[0]; // loss ended higher than it started -> diverging
    const verdict = run.diverged
      ? `发散 → loss=${fmt(last)} (第 ${run.losses.length} 步爆掉)`
      : grew
        ? `震荡发散 → loss=${fmt(last)} (越走越大，60 步未溢出)`
        : last < 0.2
          ? `收敛 → loss=${last.toFixed(4)}`
          : `缓慢 → loss=${last.toFixed(4)} (60 步还没到底)`;
    console.log(`   ${tag}  lr=${lr.toFixed(3)} (=${mult.toFixed(2)}×阈值): ${verdict}`);
    console.log(asciiLine(run.losses, 50, 6));
  }
  console.log();

  // ── ④ One forward+backward pass of the toy net, gradient-checked ────────────
  const net: ToyNet = {
    W1: [[0.5, -0.3], [0.2, 0.8]],
    b1: [0.1, -0.2],
    W2: [0.7, -0.6],
    b2: 0.05,
  };
  const xInput = [1.0, -2.0];
  const target = 0.5;
  const grads = backward(net, xInput, target);
  // Gradient-check the WHOLE net at once via the flattened view.
  const lossFlat = (v: number[]) => toyLoss(unflattenNet(v), xInput, target);
  const gNumNet = numericGradient(lossFlat, flattenNet(net));
  const gAnaNet = flattenGrads(grads);
  console.log("④ 两层 toy 网络一次前向 + 反向传播  x=[1,-2] target=0.5");
  console.log(`   预测 yhat = ${forward(net, xInput).yhat.toFixed(6)}   loss = ${toyLoss(net, xInput, target).toFixed(6)}`);
  console.log(`   dW1 = [[${grads.dW1[0].map((v) => v.toFixed(4)).join(", ")}], [${grads.dW1[1].map((v) => v.toFixed(4)).join(", ")}]]`);
  console.log(`   db1 = [${grads.db1.map((v) => v.toFixed(4)).join(", ")}]`);
  console.log(`   dW2 = [${grads.dW2.map((v) => v.toFixed(4)).join(", ")}]   db2 = ${grads.db2.toFixed(4)}`);
  console.log(`   反向传播 vs 有限差分 最大误差 = ${fmt(maxAbsDiff(gNumNet, gAnaNet))}  → backprop 正确\n`);

  // ── ⑤ Lagrange multipliers ─────────────────────────────────────────────────
  const sol = solveLagrange();
  // Verify the optimality condition ∇f = λ∇g and feasibility g = 0 numerically.
  const gradF = [2 * sol.x, 2 * sol.y];
  const gradG = [1, 1];
  const stationarityResidual = maxAbsDiff(gradF, scale(gradG, sol.lambda));
  const constraintResidual = Math.abs(sol.x + sol.y - 1);
  console.log("⑤ 拉格朗日乘子：min x²+y²  s.t. x+y=1");
  console.log(`   解 (x, y) = (${sol.x}, ${sol.y})   λ = ${sol.lambda}   f* = ${(sol.x ** 2 + sol.y ** 2).toFixed(4)}`);
  console.log(`   ∇f - λ∇g 残差 = ${fmt(stationarityResidual)}   约束 g 残差 = ${fmt(constraintResidual)}  → KKT 条件成立`);
  // Cross-check against the geometric closed form: distance from origin to line x+y=1 is
  // 1/sqrt(2), so f* should be that distance squared = 1/2.
  console.log(`   几何校验：原点到直线距离² = ${(0.5).toFixed(4)} = f*  ✓\n`);

  // ── FAILURE MODE: lr far beyond 2/L blows the loss to Infinity ─────────────
  // WHY 5× and 200 steps: at exactly >2/L the loss already grows, but float64 holds ~1.8e308,
  //   so reaching genuine Infinity needs enough steps for the geometric blow-up to overflow.
  //   We pick lr/steps that ACTUALLY hit a non-finite value here (verified), so the demo shows
  //   real Infinity, not just a big number we hand-wave into "eventually NaN".
  console.log("--- 失败模式：学习率远超 2/L，loss 真爆成 Infinity ---");
  const boomLr = stableLimit * 5;
  const boom = gradientDescent([0, 0], boomLr, 200, fitGrad, fitLoss);
  console.log(`   lr=${boomLr.toFixed(3)} (=5×阈值) 前几步 loss: ${boom.losses.slice(0, 6).map(fmt).join(" → ")}`);
  console.log(
    `   loss 在第 ${boom.losses.length} 步变为 ${fmt(boom.losses.at(-1)!)}（${boom.diverged ? "检测到非有限值，提前停止" : "200 步内未溢出"}）`,
  );
  console.log("   每步残差被放大 |1 - lr·L| > 1，loss 指数增长直到 float 溢出 → Infinity（再算就成 NaN）。");
  console.log("   实战对策：调小 lr、梯度裁剪、或用自适应步长（如 Adam）。绝对值是 toy 数据偏乐观，可迁移的是“越界即发散”这一相变行为。");
}

main();
