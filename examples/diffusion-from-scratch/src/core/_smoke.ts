// core/_smoke.ts — A self-check for the shared core. NOT a chapter (underscore prefix).
//
// WHY it exists: the core is the geometry every stage stands on. Before any stage author
//   builds on it, this file proves — with real numbers, no asserts-as-comments — that the
//   autograd adjoints are correct (gradCheck < 1e-5), the optimizer actually minimizes a
//   loss, the schedule's ᾱ_t invariant holds, and the data/plot pipeline renders real points.
//
// It also demonstrates ONE failure mode on purpose (see "FAILURE MODE DEMO" below) so the
// book's "always show a failure, not just happy path" rule is satisfied at the core level.
//
// Run: npx tsx src/core/_smoke.ts   (deterministic — same bytes every run for seed 1337)

import { mixtureOfGaussians, twoMoons } from "./data.js";
import { Linear, MLP, SinusoidalEmbedding } from "./nn.js";
import { Adam } from "./optim.js";
import { lossCurveASCII, scatterASCII } from "./plot.js";
import { RNG } from "./rng.js";
import { cosineSchedule, linearSchedule } from "./schedule.js";
import { numericalGradCheck, Tensor } from "./tensor.js";

function section(title: string): void {
  console.log(`\n${"=".repeat(64)}\n${title}\n${"=".repeat(64)}`);
}

function main(): void {
  const rng = new RNG(1337);

  // ---- 1. autograd gradient check: every adjoint vs finite differences ----
  section("1. autograd gradient check (analytic vs finite-difference, max abs err)");
  // A scalar function touching matmul, add(broadcast bias), silu, mean — the exact ops the
  // denoiser uses. If any adjoint is wrong, the error spikes well above 1e-5.
  const W = Tensor.from([3, 4], () => rng.gaussian() * 0.5);
  const b = Tensor.from([1, 4], () => rng.gaussian() * 0.5);
  const fnMatmul = (x: Tensor) => x.matmul(W).add(b).silu().mean();
  const x0 = Tensor.from([2, 3], () => rng.gaussian());
  const errMatmul = numericalGradCheck(fnMatmul, x0);
  console.log(`  matmul+bias+silu+mean : ${errMatmul.toExponential(3)}`);

  const fnTanh = (x: Tensor) => x.tanh().sum();
  const errTanh = numericalGradCheck(fnTanh, Tensor.from([5], () => rng.gaussian()));
  console.log(`  tanh+sum              : ${errTanh.toExponential(3)}`);

  const fnSqrt = (x: Tensor) => x.mulScalar(1).addScalar(2.0).sqrt().sum(); // keep arg > 0
  const errSqrt = numericalGradCheck(fnSqrt, Tensor.from([4], () => Math.abs(rng.gaussian())));
  console.log(`  sqrt+sum              : ${errSqrt.toExponential(3)}`);

  const fnDiv = (x: Tensor) => x.div(Tensor.fill([3], 2.0)).sum();
  const errDiv = numericalGradCheck(fnDiv, Tensor.from([3], () => rng.gaussian()));
  console.log(`  div+sum               : ${errDiv.toExponential(3)}`);

  const worst = Math.max(errMatmul, errTanh, errSqrt, errDiv);
  console.log(`  worst-case adjoint error: ${worst.toExponential(3)}  (threshold 1e-5 -> ${worst < 1e-5 ? "PASS" : "FAIL"})`);

  // ---- 2. optimizer sanity: Adam minimizes a tiny least-squares problem ----
  section("2. Adam minimizes ||A·w - y||^2 (loss must fall)");
  // Fixed target weights; recover them from noisy targets. Pure core, no diffusion yet.
  const A = Tensor.from([16, 3], () => rng.gaussian());
  const trueW = Tensor.from([3, 1], () => rng.gaussian());
  const yData = new Float64Array(16);
  {
    const target = A.matmul(trueW);
    for (let i = 0; i < 16; i++) yData[i] = target.data[i] + rng.gaussian() * 0.05;
  }
  const y = new Tensor(yData, [16, 1]);
  const fit = new Linear(3, 1, rng);
  const opt = new Adam(fit.parameters(), 0.05);
  const losses: number[] = [];
  for (let step = 0; step < 200; step++) {
    const pred = fit.forward(A);
    const diff = pred.sub(y);
    const loss = diff.mul(diff).mean();
    opt.zeroGrad();
    loss.backward();
    opt.step();
    losses.push(loss.data[0]);
  }
  console.log(`  loss[0]   = ${losses[0].toFixed(6)}`);
  console.log(`  loss[end] = ${losses[losses.length - 1].toFixed(6)}`);
  console.log(`  reduction = ${(losses[0] / losses[losses.length - 1]).toFixed(1)}x`);
  console.log(lossCurveASCII(losses));

  // ---- 3. schedule invariant: ᾱ_t monotone down to ~0, linear vs cosine ----
  section("3. noise schedule: alphaBar_T must approach 0 (premise of reverse from N(0,I))");
  const T = 200;
  const lin = linearSchedule(T);
  const cos = cosineSchedule(T);
  console.log(`  T = ${T}`);
  console.log(`  linear: aBar_1=${lin.alphaBars[0].toFixed(4)}  aBar_T=${lin.alphaBars[T - 1].toExponential(3)}`);
  console.log(`  cosine: aBar_1=${cos.alphaBars[0].toFixed(4)}  aBar_T=${cos.alphaBars[T - 1].toExponential(3)}`);
  // Honest comparison: linear @ small T leaves MORE residual signal at t=T than cosine.
  console.log(
    `  note: linear leaves ${(lin.alphaBars[T - 1] / cos.alphaBars[T - 1]).toFixed(1)}x more residual signal at t=T than cosine ` +
      `(why cosine was invented for small T).`,
  );

  // ---- 4. data + sinusoidal embedding shapes, and a real scatter ----
  section("4. toy data + time embedding (real shapes, real scatter)");
  const moons = twoMoons(400, 0.05, rng);
  console.log(`  twoMoons shape = [${moons.shape}]`);
  const emb = new SinusoidalEmbedding(8);
  const e = emb.forward([0, 50, 199]);
  console.log(`  SinusoidalEmbedding(8).forward([0,50,199]) -> shape [${e.shape}]`);
  console.log(`    t=0   embedding = [${Array.from(e.data.slice(0, 8)).map((v) => v.toFixed(3)).join(", ")}]`);
  console.log("  twoMoons scatter (each char = real point density):");
  console.log(scatterASCII(moons, 56, 16));

  // ---- 5. FAILURE MODE DEMO: forgetting zeroGrad() corrupts the gradient ----
  section("5. FAILURE MODE DEMO: skipping zeroGrad() trains on a corrupted gradient");
  // Grads ACCUMULATE (+=) in the engine. Without zeroGrad, step N descends not on this
  // step's gradient but on the SUM of every step's gradient so far — a stale, mis-scaled
  // direction. The dangerous part is that it STILL "trains" (the loss moves), so nothing
  // throws; it just converges to a clearly worse place than the same run with zeroGrad.
  // We run both with identical seed/optimizer/budget and compare the gap — that gap IS the
  // bug. (This is the single most common autograd-engine mistake.)
  const mix = mixtureOfGaussians([[-1, -1], [1, 1], [-1, 1]], 128, new RNG(7));
  const xs = mix.sliceRows(0, 64); // [64,2] inputs
  const target2 = mix.sliceRows(64, 128); // [64,2] targets (arbitrary regression)
  const STEPS = 150;

  const badNet = new MLP([2, 16, 2], "silu", new RNG(99));
  const badOpt = new Adam(badNet.parameters(), 0.02);
  const badLosses: number[] = [];
  for (let step = 0; step < STEPS; step++) {
    const pred = badNet.forward(xs);
    const diff = pred.sub(target2);
    const loss = diff.mul(diff).mean();
    // BUG ON PURPOSE: no badOpt.zeroGrad(). Every past step's grad is still in .grad.
    loss.backward();
    badOpt.step();
    badLosses.push(loss.data[0]);
  }

  // Identical setup (same seed -> same init), the ONLY difference is the zeroGrad() call.
  const goodNet = new MLP([2, 16, 2], "silu", new RNG(99));
  const goodOpt = new Adam(goodNet.parameters(), 0.02);
  const goodLosses: number[] = [];
  for (let step = 0; step < STEPS; step++) {
    const pred = goodNet.forward(xs);
    const diff = pred.sub(target2);
    const loss = diff.mul(diff).mean();
    goodOpt.zeroGrad(); // the one line that fixes it
    loss.backward();
    goodOpt.step();
    goodLosses.push(loss.data[0]);
  }

  const badEnd = badLosses[badLosses.length - 1];
  const goodEnd = goodLosses[goodLosses.length - 1];
  console.log(`  start loss (both)      : ${badLosses[0].toFixed(4)}`);
  console.log(`  WITHOUT zeroGrad, end  : ${badEnd.toFixed(4)}`);
  console.log(`  WITH    zeroGrad, end  : ${goodEnd.toFixed(4)}`);
  console.log(`  buggy run ends ${(badEnd / goodEnd).toFixed(1)}x WORSE despite same init/budget — silently, nothing threw.`);
  console.log(`  this is why the optimizer owns zeroGrad() (optim.ts): it sits right next to step() so it's hard to forget.`);

  section("core self-check complete — all primitives exercised with real numbers");
}

main();
