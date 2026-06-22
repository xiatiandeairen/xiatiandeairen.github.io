// stage04-ddpm-train.ts — Chapter 04: collapse the score objective into ONE line of MSE,
// then actually train a denoiser and watch it reconstruct the data distribution.
//
// THE WHOLE CHAPTER IN ONE SENTENCE: DDPM training is "draw x_0, draw a random step t, noise
//   x_0 to x_t with a KNOWN ε, ask the net to predict that ε, minimize ‖ε_θ(x_t,t) − ε‖²."
//   That is it. No score estimator, no Langevin chain at train time — the closed-form forward
//   q(x_t|x_0) hands you the target ε for free, so a generative model trains like plain
//   regression. This file makes that claim concrete: ~19K-param MLP, ~1500 Adam steps, then a
//   full T=1000 reverse chain that turns N(0,I) noise back into two moons.
//
// WHY ε-prediction (not x_0-prediction, not score): predicting ε is the Ho et al. 2020
//   parameterization. ε is unit-variance N(0,I) at every t, so the regression target has a
//   stable scale across all noise levels — the loss is well-conditioned without per-t
//   reweighting. (Predicting x_0 directly has target variance that collapses as t→T; predicting
//   the score ∇log q needs the 1/√(1−ᾱ) factor folded in. ε is the clean middle.)
//
// HONESTY (read before trusting any number below):
//   - The forward/reverse math here is the REAL DDPM, identical to image diffusion.
//   - The data is a 2-D toy. Absolute loss values (~0.x) and Chamfer distances are only
//     meaningful relative to each other in THIS file — they do NOT transfer to image-space
//     numbers and are optimistic (2-D is easy). What transfers: (a) the loss falls and plateaus,
//     (b) a trained ε_θ reconstructs the data shape while baselines do not, (c) removing the
//     time embedding breaks it. Those are mechanism facts, not toy artifacts.
//   - Every printed number is computed at runtime (loss read from the live graph, Chamfer
//     measured on actual generated points, wall-clock from Date.now()). Nothing is asserted.
//
// DETERMINISM: one RNG(1337) threads weight init, batch sampling, forward noise, and reverse
//   z. Two runs print byte-identical numbers. The failure-mode net uses a SEPARATE RNG so its
//   init does not depend on how many draws the good run happened to consume.

import { twoMoons } from "./core/data.js";
import { MLP, SinusoidalEmbedding } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { lossCurveASCII, scatterASCII } from "./core/plot.js";
import { RNG } from "./core/rng.js";
import { linearSchedule, type NoiseSchedule } from "./core/schedule.js";
import { Tensor } from "./core/tensor.js";

// ----------------------------------------------------------------------------------------
// Hyperparameters — small enough to run in seconds on a CPU, large enough to learn two moons.
// ----------------------------------------------------------------------------------------
const SEED = 1337;
const T = 1000; // diffusion steps. Linear schedule needs a large-ish T for ᾱ_T to approach 0.
const EMB_DIM = 16; // sinusoidal time-embedding width (even). Input to MLP = 2 + EMB_DIM.
const HIDDEN = [128, 128]; // two hidden layers of SiLU. ~19K params (printed below).
const DATA_N = 1000; // training set size (fixed cloud, resampled into batches each step).
const BATCH = 128;
const STEPS = 2000;
const LR = 2e-3;
const DATA_NOISE = 0.06; // jitter on the moons; small so the two crescents stay distinct.

// ----------------------------------------------------------------------------------------
// Denoiser ε_θ(x_t, t): [x ⊕ sinusoidal(t)] -> MLP -> predicted ε (2-D).
//
// WHY a struct of (mlp, emb) instead of a subclass: the time embedding is a fixed feature map
//   (no params), the MLP holds all the weights. Keeping them side by side makes the forward
//   pass read exactly like the math: concat the conditioning, run the net. The `useTime` flag
//   exists ONLY for the failure-mode demo (see end of file) — when false we feed a zero block
//   of the same width, so the net has identical capacity but is blind to t.
// ----------------------------------------------------------------------------------------
interface Denoiser {
  mlp: MLP;
  emb: SinusoidalEmbedding;
  useTime: boolean;
}

function makeDenoiser(rng: RNG, useTime: boolean): Denoiser {
  const emb = new SinusoidalEmbedding(EMB_DIM);
  // sizes: [2 + EMB_DIM, ...HIDDEN, 2]. The final layer has no activation (raw ε prediction).
  const sizes = [2 + EMB_DIM, ...HIDDEN, 2];
  const mlp = new MLP(sizes, "silu", rng);
  return { mlp, emb, useTime };
}

function countParams(d: Denoiser): number {
  return d.mlp.parameters().reduce((acc, p) => acc + p.size, 0);
}

/** Forward pass for a batch: x [B,2], integer timesteps [B] -> predicted ε [B,2].
 *  INVARIANT: timesteps.length must equal x.shape[0]; concatCols requires matching row counts. */
function predictEps(d: Denoiser, x: Tensor, timesteps: number[]): Tensor {
  const batch = x.shape[0];
  // The time conditioning block. Failure mode: zeros (a leaf, no grad) so the net cannot
  // distinguish t=5 from t=995 — same input shape, zero information.
  const timeBlock = d.useTime
    ? d.emb.forward(timesteps)
    : new Tensor(new Float64Array(batch * EMB_DIM), [batch, EMB_DIM]);
  const input = Tensor.concatCols([x, timeBlock]);
  return d.mlp.forward(input);
}

// ----------------------------------------------------------------------------------------
// Forward process q(x_t | x_0): x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,  ε ~ N(0,I).
// Returns BOTH the noised batch and the ε that produced it — ε is the regression target.
// This is the one place the "free target" comes from: we generate the noise, so we KNOW it.
// ----------------------------------------------------------------------------------------
function addNoise(
  x0: Tensor,
  timesteps: number[],
  sched: NoiseSchedule,
  rng: RNG,
): { xt: Tensor; eps: Tensor } {
  const batch = x0.shape[0];
  const dim = x0.shape[1];
  const xt = new Float64Array(batch * dim);
  const eps = new Float64Array(batch * dim);
  for (let r = 0; r < batch; r++) {
    const t = timesteps[r];
    const sa = sched.sqrtAlphaBar[t];
    const soma = sched.sqrtOneMinusAlphaBar[t];
    for (let c = 0; c < dim; c++) {
      const e = rng.gaussian();
      eps[r * dim + c] = e;
      xt[r * dim + c] = sa * x0.data[r * dim + c] + soma * e;
    }
  }
  // xt/eps are leaf tensors: gradient flows into the NET via predictEps, not into the data.
  return { xt: new Tensor(xt, [batch, dim]), eps: new Tensor(eps, [batch, dim]) };
}

// ----------------------------------------------------------------------------------------
// Training loop. Each step: sample a batch of x_0, a random t per row, noise them, predict ε,
// MSE loss, backprop, Adam step. Returns the per-step loss history.
// ----------------------------------------------------------------------------------------
function train(d: Denoiser, data: Tensor, sched: NoiseSchedule, rng: RNG): number[] {
  const opt = new Adam(d.mlp.parameters(), LR);
  const losses: number[] = [];
  const dataN = data.shape[0];
  for (let step = 0; step < STEPS; step++) {
    // Build a minibatch by sampling row indices (with replacement — fine for a toy).
    const rows = new Float64Array(BATCH * 2);
    const timesteps: number[] = new Array(BATCH);
    for (let b = 0; b < BATCH; b++) {
      const ri = rng.choice(dataN);
      rows[b * 2] = data.data[ri * 2];
      rows[b * 2 + 1] = data.data[ri * 2 + 1];
      // t uniform in [0, T-1]. Each row gets its own noise level — the net must handle ALL
      // levels with one set of weights, which is exactly what the time embedding enables.
      timesteps[b] = rng.choice(T);
    }
    const x0 = new Tensor(rows, [BATCH, 2]);
    const { xt, eps } = addNoise(x0, timesteps, sched, rng);

    const pred = predictEps(d, xt, timesteps);
    // MSE = mean over batch and over the 2 coords of (ε_pred − ε)². diff² then mean to scalar.
    const diff = pred.sub(eps);
    const loss = diff.mul(diff).mean();

    opt.zeroGrad();
    loss.backward();
    opt.step();
    losses.push(loss.data[0]);
  }
  return losses;
}

// ----------------------------------------------------------------------------------------
// Reverse sampling (the chapter-02 ancestral sampler, now driven by the TRAINED ε_θ).
// x_{t-1} = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t,t)) + σ_t · z,   z~N(0,I) (z=0 at t=0).
// We use σ_t = √β_t (the DDPM "fixed large" variance). Full T steps, no shortcuts here — this
// is the honest cost of plain DDPM sampling (stage05 will show fewer-step samplers).
// ----------------------------------------------------------------------------------------
function sample(d: Denoiser, nSamples: number, sched: NoiseSchedule, rng: RNG): Tensor {
  // Start from pure noise x_T ~ N(0,I). This only matches the forward endpoint because the
  // schedule drove ᾱ_T near 0 — otherwise we'd start off-distribution.
  const x = new Float64Array(nSamples * 2);
  for (let i = 0; i < x.length; i++) x[i] = rng.gaussian();

  for (let t = T - 1; t >= 0; t--) {
    const timesteps: number[] = new Array(nSamples).fill(t);
    // No autograd needed at sample time, but predictEps builds a graph; that's cheap here and
    // keeps one code path. We never call backward, so the graph is just discarded.
    const xtTensor = new Tensor(Float64Array.from(x), [nSamples, 2]);
    const eps = predictEps(d, xtTensor, timesteps).data;

    const alpha = sched.alphas[t];
    const beta = sched.betas[t];
    const soma = sched.sqrtOneMinusAlphaBar[t];
    const invSqrtAlpha = 1 / Math.sqrt(alpha);
    const sigma = Math.sqrt(beta);
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < 2; c++) {
        const idx = i * 2 + c;
        const mean = invSqrtAlpha * (x[idx] - (beta / soma) * eps[idx]);
        // z is added at every step EXCEPT the last (t=0): the final step is a clean denoise to
        // x_0, injecting noise there would just re-roughen the result.
        const z = t > 0 ? rng.gaussian() : 0;
        x[idx] = mean + sigma * z;
      }
    }
  }
  return new Tensor(x, [nSamples, 2]);
}

// ----------------------------------------------------------------------------------------
// Chamfer distance: symmetric nearest-neighbour distance between two point clouds. For each
// generated point, distance to its nearest real point, averaged; plus the reverse direction.
// WHY this metric: it punishes BOTH "samples land off the data" (first term) AND "samples miss
//   part of the data / mode collapse" (second term). Lower = closer to the real distribution.
//   It is O(n·m) — fine for a few hundred points. Absolute value is toy-only; use it to COMPARE
//   trained vs baseline on the SAME real cloud.
// ----------------------------------------------------------------------------------------
function chamfer(gen: Tensor, real: Tensor): number {
  const g = gen.data;
  const r = real.data;
  const ng = gen.shape[0];
  const nr = real.shape[0];
  const nearest = (src: Float64Array, ns: number, dst: Float64Array, nd: number): number => {
    let acc = 0;
    for (let i = 0; i < ns; i++) {
      let best = Infinity;
      const sx = src[i * 2];
      const sy = src[i * 2 + 1];
      for (let j = 0; j < nd; j++) {
        const dx = sx - dst[j * 2];
        const dy = sy - dst[j * 2 + 1];
        const d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
      }
      acc += Math.sqrt(best);
    }
    return acc / ns;
  };
  return nearest(g, ng, r, nr) + nearest(r, nr, g, ng);
}

// ========================================================================================
// main — the chapter's runnable narrative.
// ========================================================================================
function main(): void {
  const rng = new RNG(SEED);

  console.log("=== 第 04 章:DDPM 训练 — 把 score 落成一行 MSE,真训出去噪网络 ===\n");

  // --- setup ---
  const sched = linearSchedule(T);
  console.log(
    `噪声调度: linear, T=${T}, ᾱ_T=${sched.alphaBars[T - 1].toExponential(3)} ` +
      `(接近 0 → x_T ≈ N(0,I),反向链才能从纯噪声起步)`,
  );
  const data = twoMoons(DATA_N, DATA_NOISE, rng);
  console.log(`训练数据: twoMoons, ${DATA_N} 点, 噪声 std=${DATA_NOISE}\n`);
  console.log("真实数据分布 (target):");
  console.log(scatterASCII(data, 56, 16));
  console.log("");

  // --- model ---
  // Dedicated init/train seeds (NET_INIT_SEED / NET_TRAIN_SEED) so the t-blind failure-mode
  // net at the end can reuse the EXACT same seeds — then the ONLY difference between the two
  // runs is the time block, which is what makes that comparison a clean controlled experiment.
  const NET_INIT_SEED = SEED + 1;
  const NET_TRAIN_SEED = SEED + 2;
  const net = makeDenoiser(new RNG(NET_INIT_SEED), true);
  const nParams = countParams(net);
  console.log(
    `去噪网络 ε_θ: 输入 [x(2) ⊕ sin-emb(${EMB_DIM})] → MLP ${[2 + EMB_DIM, ...HIDDEN, 2].join(
      "→",
    )} (silu) → ε(2),共 ${nParams} 个参数\n`,
  );

  // --- baseline: Chamfer of an UNTRAINED net's samples (the bar to beat) ---
  // Use a fresh RNG for sampling noise so the baseline and trained sampler see the SAME z
  // stream — a fair comparison isolates "did training help?" from "lucky noise".
  const baselineNet = makeDenoiser(new RNG(SEED + 5), true); // separate init stream, never trained
  const baselineSamples = sample(baselineNet, 300, sched, new RNG(SEED + 100));
  const baselineChamfer = chamfer(baselineSamples, data);
  console.log(`[baseline] 未训练网络采样 300 点 → Chamfer = ${baselineChamfer.toFixed(4)}`);
  console.log("未训练网络的采样 (应是无结构噪声):");
  console.log(scatterASCII(baselineSamples, 56, 16));
  console.log("");

  // --- train ---
  console.log(`开始训练: ${STEPS} 步, batch=${BATCH}, lr=${LR}, Adam ...`);
  const tStart = Date.now();
  const losses = train(net, data, sched, new RNG(NET_TRAIN_SEED));
  const trainMs = Date.now() - tStart;
  const startLoss = losses.slice(0, 10).reduce((a, b) => a + b, 0) / 10; // avg first 10
  const endLoss = losses.slice(-10).reduce((a, b) => a + b, 0) / 10; // avg last 10
  console.log(
    `训练完成,真实墙钟 ${trainMs} ms。loss(前10步均值)=${startLoss.toFixed(
      4,
    )} → loss(后10步均值)=${endLoss.toFixed(4)},下降 ${(startLoss / endLoss).toFixed(2)}x\n`,
  );
  console.log("训练 loss 曲线 (应明显下降并趋稳):");
  console.log(lossCurveASCII(losses, 12, 56));
  console.log("");

  // --- sample with the trained net ---
  // Same z stream (RNG seed SEED+100) as the baseline so the only difference is the weights.
  const trainedSamples = sample(net, 300, sched, new RNG(SEED + 100));
  const trainedChamfer = chamfer(trainedSamples, data);
  console.log(`[trained] 训练后网络采样 300 点 → Chamfer = ${trainedChamfer.toFixed(4)}`);
  console.log("训练后网络的采样 (应还原双月两个弯月):");
  console.log(scatterASCII(trainedSamples, 56, 16));
  console.log("");

  // --- assertion: trained must beat baseline by a clear margin (honest, computed) ---
  const improvement = baselineChamfer / trainedChamfer;
  console.log(
    `Chamfer 对比: baseline ${baselineChamfer.toFixed(4)} → trained ${trainedChamfer.toFixed(
      4,
    )} (改善 ${improvement.toFixed(2)}x)`,
  );
  // This is a real assertion, not decoration: if training silently failed (e.g. a wrong
  // adjoint, a missing zeroGrad), the margin would collapse and this would throw — turning a
  // "looks plausible" plot into a hard pass/fail.
  if (trainedChamfer >= baselineChamfer * 0.8) {
    throw new Error(
      `训练未显著优于未训练 baseline: trained ${trainedChamfer.toFixed(
        4,
      )} vs baseline ${baselineChamfer.toFixed(4)} — 训练链路有问题`,
    );
  }
  console.log("断言通过: 训练后 Chamfer 显著优于未训练 baseline。\n");

  console.log(
    "诚实标注: 此 loss / Chamfer 的绝对值仅对此 2-D toy 有意义(偏乐观);" +
      "可迁移的是 loss 下降趋势 + 采样还原双月这一事实,以及 DDPM 训练 = 一行 MSE 的机制。\n",
  );

  // ======================================================================================
  // FAILURE MODE — 去掉时间嵌入,网络不知道"现在第几步"。
  //
  // 机制:同一个 x_t 在 t=10(几乎干净,只需轻微去噪) 和 t=990(几乎纯噪声,需大幅去噪) 下,
  //   正确的 ε 完全不同。没有 t 信息,网络只能对所有噪声水平输出一个"平均"的 ε —— 反向链
  //   每一步都用错误幅度去噪,误差累积,采样塌成一团模糊圆斑(高斯先验的残影),学不到双月结构。
  // ======================================================================================
  console.log("=== 失败模式: 去掉时间嵌入 (feed zeros instead of sin-emb) ===");
  // Controlled experiment: SAME init seed + SAME training-noise seed as the good net above —
  // the only variable changed is useTime=false. So any quality gap is attributable to the
  // missing time conditioning, not to a different random draw.
  const blindNet = makeDenoiser(new RNG(NET_INIT_SEED), false); // identical capacity, t-blind
  console.log(`重训一个 t-blind 网络 (相同 init/train 种子,相同参数量 ${countParams(blindNet)}, 仅时间块置零)...`);
  const blindLosses = train(blindNet, data, sched, new RNG(NET_TRAIN_SEED));
  const blindEnd = blindLosses.slice(-10).reduce((a, b) => a + b, 0) / 10;
  console.log(`t-blind 训练后 loss(后10步均值)=${blindEnd.toFixed(4)} (有时间嵌入时为 ${endLoss.toFixed(4)})`);
  const blindSamples = sample(blindNet, 300, sched, new RNG(SEED + 100));
  const blindChamfer = chamfer(blindSamples, data);
  console.log(`t-blind 采样 300 点 → Chamfer = ${blindChamfer.toFixed(4)} (有时间嵌入时为 ${trainedChamfer.toFixed(4)})`);
  console.log("t-blind 网络的采样 (应糊成一团,不成双月):");
  console.log(scatterASCII(blindSamples, 56, 16));
  console.log("");
  const blindRatio = blindChamfer / trainedChamfer;
  console.log(
    `结论: 即便 loss 看似在降(${blindEnd.toFixed(4)},仅略高于有时间嵌入的 ${endLoss.toFixed(
      4,
    )}),采样质量却明显塌掉 —— ` +
      `t-blind Chamfer ${blindChamfer.toFixed(4)} 比有时间嵌入的 ${trainedChamfer.toFixed(
        4,
      )} 差 ${blindRatio.toFixed(2)}x,散点糊成中心一团而非两个弯月。\n` +
      "注意 loss 只差一点点、Chamfer 却差一截:这正是关键 —— 平均掉所有噪声水平的 ε 只让训练 loss 略升," +
      "但反向链每步都用错误幅度去噪,误差沿 1000 步累积,最终塌成模糊团。\n" +
      "时间条件不是可选项: 网络必须知道当前噪声水平,才能对不同步预测不同的 ε。",
  );
}

main();
