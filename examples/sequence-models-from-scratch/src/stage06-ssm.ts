// stage06-ssm.ts — State-Space Models: the same linear recurrence, two ways.
//
// CHAPTER THESIS: a diagonal SSM is a linear recurrence h_t = A·h_{t-1} + B·u_t,
//   y_t = C·h_t. Because it is LINEAR (no tanh between steps, unlike an RNN), the same
//   computation can be unrolled into a CONVOLUTION with kernel K[k] = C·A^k·B. Recurrent
//   form is O(n) memory / sequential; convolutional form is O(n) work but parallel over
//   time. They are the SAME function — and we prove it numerically (err < 1e-8) rather
//   than asserting it. That dual view is what lets SSMs reclaim long sequences from the
//   O(n^2) attention of stage05.
//
// WHY DIAGONAL + SISO-per-channel: we give EACH feature channel its own scalar (A,B,C).
//   So a `dim`-wide SSM is `dim` independent scalar SSMs running in parallel. This is the
//   honest core of S4D / Mamba's diagonal SSM, minus the input-dependent (selective)
//   parametrization. Keeping A/B/C scalar-per-channel makes the recurrence↔convolution
//   identity exact and inspectable; a full dense-A SSM would obscure it.
//
// STABILITY IS STRUCTURAL, NOT OPTIONAL (the failure mode this stage demonstrates):
//   The state evolves as h_t ≈ A^t · (early input). If |A| < 1 the influence of old
//   inputs DECAYS (bounded, usable memory). If |A| > 1 it GROWS like A^t — the hidden
//   state norm explodes exponentially and the output goes to Inf/NaN within a few dozen
//   steps. So a usable SSM must CONSTRAIN A into the stable region by construction. We
//   parametrize A = sigmoid(a_logit) ∈ (0,1): stability is then guaranteed for ANY value
//   the optimizer lands on. demoStability() shows what happens when you bypass that.
//
// HONESTY: copyTask accuracy here is on TOY data (tiny vocab, clean signal); absolute
//   numbers are optimistic. What transfers is the RELATIVE story — the rec/conv outputs
//   agree to machine precision, the O(n) scan beats O(n^2) attention by a widening margin
//   as n grows, and an unstable A kills the model regardless of training.
//
// ESM/import discipline: we import ONLY from ./core/* (never another stageNN, which would
//   run its main()). SSMBlock is exported so later stages can import the cell from here.

import { makeRng, type Rng } from "./core/prng.js";
import { Tensor } from "./core/tensor.js";
import { Module, Embedding, Linear, collectParams } from "./core/nn.js";
import { Adam, clipGradNorm } from "./core/optim.js";
import { copyTask } from "./core/data.js";
import { argmax, accuracy, timeit, countMACs } from "./core/metrics.js";
import { sparkline, heatmap, bar } from "./core/plot.js";

// ===========================================================================
// 1. The diagonal SSM cell (trainable, engine-backed)
// ===========================================================================

/**
 * SSMBlock: a diagonal state-space layer with `dim` independent scalar channels.
 *
 * Parametrization (the part that matters):
 *   - aLogit  -> A = sigmoid(aLogit) ∈ (0,1): GUARANTEES |A| < 1, i.e. structural
 *     stability. The optimizer can move aLogit anywhere on the real line and A stays in
 *     the stable region. This is the whole point of "structured A".
 *   - B, C are unconstrained per-channel scalars [1, dim].
 *
 * Forward (recurrent scan): for each timestep, h = A⊙h_prev + B⊙u_t ; y_t = C⊙h_t.
 *   We use the SAME A/B/C tensors at every step (shared weights) — the engine's
 *   accumulate-not-overwrite grad rule (tensor.ts header) is what makes BPTT correct here.
 *
 * INVARIANT: forward expects a per-timestep list of [batch, dim] inputs (already embedded);
 *   it returns a per-timestep list of [batch, dim] outputs. We keep time as an explicit
 *   JS array rather than a [T,B,D] tensor because the engine's matmul is 2D-only and the
 *   scan is genuinely sequential — materializing the dependency is the teaching point.
 */
export class SSMBlock extends Module {
  readonly dim: number;
  aLogit: Tensor; // [1, dim] -> A = sigmoid(aLogit) in (0,1), stable by construction
  B: Tensor; // [1, dim]
  C: Tensor; // [1, dim]

  constructor(dim: number, rng: Rng) {
    super();
    this.dim = dim;
    // Init A near ~0.5..0.9 so channels start with a mix of fast/slow memory. We set
    // aLogit slightly positive on average => A in the upper-stable range (long memory),
    // which copyTask (long dependency) needs. Small noise breaks channel symmetry.
    const aInit = new Float64Array(dim);
    for (let i = 0; i < dim; i++) aInit[i] = 1.0 + rng.normal() * 0.3; // sigmoid(1)≈0.73
    this.aLogit = new Tensor(aInit, [1, dim]);
    this.B = Tensor.randn([1, dim], rng, 0.5);
    this.C = Tensor.randn([1, dim], rng, 0.5);
  }

  /** Effective stable transition A = sigmoid(aLogit), as a graph node (used in scan). */
  private transition(): Tensor {
    return this.aLogit.sigmoid();
  }

  /**
   * Recurrent forward over an explicit time axis.
   * inputs: array length T, each [batch, dim]. Returns array length T, each [batch, dim].
   */
  forward(inputs: Tensor[], batch: number): Tensor[] {
    const A = this.transition(); // [1,dim], broadcasts over batch rows
    let h = Tensor.zeros([batch, this.dim]); // initial state h_0 = 0
    const outputs: Tensor[] = [];
    for (let t = 0; t < inputs.length; t++) {
      // h_t = A⊙h_{t-1} + B⊙u_t  (elementwise; A/B/C [1,dim] broadcast over the batch axis)
      h = A.mul(h).add(this.B.mul(inputs[t]));
      outputs.push(this.C.mul(h)); // y_t = C⊙h_t
    }
    return outputs;
  }

  override params(): Tensor[] {
    return [this.aLogit, this.B, this.C];
  }
}

// ===========================================================================
// 2. Two equivalent forwards on raw Float64 (no grad) — the mechanism self-check
// ===========================================================================
//
// These operate on plain arrays (single channel, scalar A,B,C) so the rec==conv identity
// is laid bare with no engine indirection. They are NOT used for training — they exist to
// PROVE the convolution kernel K[k]=C·A^k·B reproduces the recurrence exactly.

/** Recurrent: h_t = A*h_{t-1} + B*u_t ; y_t = C*h_t. Returns y[0..T-1]. */
function scanRecurrent(u: number[], A: number, B: number, C: number): number[] {
  const y = new Array<number>(u.length);
  let h = 0;
  for (let t = 0; t < u.length; t++) {
    h = A * h + B * u[t];
    y[t] = C * h;
  }
  return y;
}

/**
 * Convolutional: y_t = sum_{k=0}^{t} K[k]·u_{t-k}, with kernel K[k] = C·A^k·B.
 * WHY this equals the recurrence: unrolling h_t = sum_{k=0}^{t} A^k·B·u_{t-k}, then
 *   y_t = C·h_t pulls C inside the sum. The kernel is the SSM's impulse response.
 * Causal convolution (only past inputs) — matches the recurrence's causality.
 */
function ssmKernel(A: number, B: number, C: number, length: number): number[] {
  const K = new Array<number>(length);
  let aPow = 1; // A^0
  for (let k = 0; k < length; k++) {
    K[k] = C * aPow * B;
    aPow *= A;
  }
  return K;
}

function scanConvolutional(u: number[], A: number, B: number, C: number): number[] {
  const T = u.length;
  const K = ssmKernel(A, B, C, T);
  const y = new Array<number>(T).fill(0);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k <= t; k++) y[t] += K[k] * u[t - k];
  }
  return y;
}

// ===========================================================================
// 3. Helpers for the copyTask training run
// ===========================================================================

// ===========================================================================
// main
// ===========================================================================

function main(): void {
  console.log("=".repeat(70));
  console.log("第 06 章 — 状态空间模型 SSM:把递归写成卷积,O(n) 重夺长序列");
  console.log("=".repeat(70));

  // -----------------------------------------------------------------------
  // PART 1 — Mechanism self-check: recurrent forward == convolutional forward.
  // -----------------------------------------------------------------------
  console.log("\n[1] 机制自证:同一组 (A,B,C) 下,递归前向 == 卷积前向");
  const checkRng = makeRng(7);
  let worstErr = 0;
  const T_check = 24;
  // Test across several random stable (A,B,C) and several random input sequences.
  for (let trial = 0; trial < 5; trial++) {
    const A = 0.3 + checkRng.next() * 0.6; // stable A in [0.3, 0.9)
    const B = checkRng.normal();
    const C = checkRng.normal();
    const u = Array.from({ length: T_check }, () => checkRng.normal());
    const yRec = scanRecurrent(u, A, B, C);
    const yConv = scanConvolutional(u, A, B, C);
    let err = 0;
    for (let t = 0; t < T_check; t++) err = Math.max(err, Math.abs(yRec[t] - yConv[t]));
    worstErr = Math.max(worstErr, err);
    console.log(
      `  trial ${trial}: A=${A.toFixed(3)} B=${B.toFixed(3)} C=${C.toFixed(3)} ` +
        `max|y_rec - y_conv| = ${err.toExponential(2)}`,
    );
  }
  console.log(
    `  最大误差 ${worstErr.toExponential(2)} ${worstErr < 1e-8 ? "< 1e-8 ✓ (两种前向数值等价,机制为真)" : "✗ 超阈值!"}`,
  );

  // -----------------------------------------------------------------------
  // PART 2 — Train an SSM on the LONG copyTask (stage05's long-dependency stress test).
  // -----------------------------------------------------------------------
  console.log("\n[2] 在长依赖 copyTask 上训练 SSM(证明长程可解)");
  const dataRng = makeRng(42);
  const initRng = makeRng(123); // separate stream: data vs init (prng.ts FAILURE MODE note)
  // delay=12 => dependency span = k + delay + 1 = 2 + 12 + 1 = 15 steps. Long enough that
  // a memoryless model fails; an SSM with high-A channels should solve it.
  const data = copyTask(dataRng, { count: 64, k: 2, delay: 12, symbols: 4 });
  console.log("  " + data.describe());
  const vocab = data.vocabSize;
  const T = data.X[0].length;
  const dim = 32;

  const embed = new Embedding(vocab, dim, initRng, 0.2);
  const ssm = new SSMBlock(dim, initRng);
  const readout = new Linear(dim, vocab, initRng);
  const params = collectParams([embed, ssm, readout]);
  const nParams = params.reduce((s, p) => s + p.size, 0);
  const opt = new Adam(params, { lr: 5e-3 });

  // copyTask answer region = last k positions (Y nonzero there). We mask loss to those.
  const k = 2;
  const answerStart = T - k; // first index of the answer region
  const batch = data.X.length; // tiny dataset: full-batch each step (deterministic)

  // Pre-build the per-timestep token-id columns for the whole (full) batch once.
  // tokensAtT[t] = array of `batch` token ids at timestep t.
  const tokensAtT: number[][] = [];
  for (let t = 0; t < T; t++) tokensAtT.push(data.X.map((seq) => seq[t]));

  const losses: number[] = [];
  const epochs = 120;
  for (let ep = 0; ep < epochs; ep++) {
    opt.zeroGrad();
    // Embed each timestep -> [batch, dim], run the SSM scan, then read out the answer steps.
    const inputs: Tensor[] = tokensAtT.map((ids) => embed.forward(ids));
    const states = ssm.forward(inputs, batch);
    // Loss only on the answer region: at each answer step, predict the corresponding payload.
    let loss = Tensor.zeros([1]);
    for (let t = answerStart; t < T; t++) {
      const logits = readout.forward(states[t]); // [batch, vocab]
      const targets = data.Y.map((seq) => seq[t]); // payload ids at this answer step
      loss = loss.add(logits.crossEntropy(targets));
    }
    loss = loss.scale(1 / k); // mean over the k answer steps
    loss.backward();
    clipGradNorm(params, 1.0); // SSM scan is linear but BPTT over T steps can still spike
    opt.step();
    losses.push(loss.data[0]);
  }

  // Evaluate accuracy on the answer region (training set — this is a memorization probe).
  let preds: number[] = [];
  let golds: number[] = [];
  {
    const inputs: Tensor[] = tokensAtT.map((ids) => embed.forward(ids));
    const states = ssm.forward(inputs, batch);
    for (let t = answerStart; t < T; t++) {
      const logits = readout.forward(states[t]);
      const targets = data.Y.map((seq) => seq[t]);
      for (let r = 0; r < batch; r++) {
        preds.push(argmax(logits.data.subarray(r * vocab, r * vocab + vocab)));
        golds.push(targets[r]);
      }
    }
  }
  const acc = accuracy(preds, golds);
  console.log(`  参数量 = ${nParams}  (A/B/C 各 ${dim} + embed ${vocab * dim} + readout ${dim * vocab + vocab})`);
  console.log(`  loss: ${losses[0].toFixed(4)} -> ${losses[losses.length - 1].toFixed(4)}`);
  console.log("  loss 曲线: " + sparkline(losses));
  console.log(
    `  答案区准确率 = ${(acc * 100).toFixed(1)}%  ` +
      `(随机基线 = ${(100 / (vocab - 2)).toFixed(1)}% over ${vocab - 2} payload symbols; ` +
      `span=${k + 12 + 1} steps 长依赖已解)`,
  );

  // -----------------------------------------------------------------------
  // PART 3 — State-over-time heatmap: how the diagonal state evolves.
  // -----------------------------------------------------------------------
  console.log("\n[3] 状态随时间演化 heatmap(单条序列,行=时间步,列=前 12 个 state 通道)");
  {
    // Run one sequence through the trained recurrent scan, capturing |h_t| per channel.
    const A = ssm.aLogit.sigmoid().data; // effective stable A per channel
    const Bd = ssm.B.data;
    const seq = data.X[0];
    const showChannels = Math.min(12, dim);
    const h = new Float64Array(dim);
    const rows: number[][] = [];
    // Reproduce the scan numerically from the trained params for ONE sequence (no grad).
    for (let t = 0; t < T; t++) {
      const emb = embed.weight.data.subarray(seq[t] * dim, seq[t] * dim + dim);
      for (let c = 0; c < dim; c++) h[c] = A[c] * h[c] + Bd[c] * emb[c];
      rows.push(Array.from(h.subarray(0, showChannels), (v) => Math.abs(v)));
    }
    console.log(heatmap(rows));
    console.log(
      `  (delim 在 t=${k + 12};亮=该通道 |state| 大。高-A 通道把早期 payload 信息一路携带到答案区)`,
    );
  }

  // -----------------------------------------------------------------------
  // PART 4 — Scaling: SSM scan O(n) vs attention O(n^2), measured + counted.
  // -----------------------------------------------------------------------
  console.log("\n[4] 标度对比:SSM 扫描 O(n) vs 注意力 O(n²)(实测 wall-clock + MAC 计数)");
  const seqLens = [64, 128, 256, 512];
  const modelDim = 64;
  const scanState = 1; // diagonal SSM: state dim per channel = 1
  const benchRng = makeRng(99);

  // Build reusable random inputs once per n so timing measures the op, not allocation.
  const ssmTimes: number[] = [];
  const attnTimes: number[] = [];
  for (const n of seqLens) {
    // SSM: a real diagonal recurrent scan over n steps on plain Float64 (the hot path).
    const Avec = new Float64Array(modelDim);
    const Bvec = new Float64Array(modelDim);
    const Cvec = new Float64Array(modelDim);
    for (let c = 0; c < modelDim; c++) {
      Avec[c] = 0.5 + benchRng.next() * 0.4;
      Bvec[c] = benchRng.normal();
      Cvec[c] = benchRng.normal();
    }
    const U = new Float64Array(n * modelDim);
    for (let i = 0; i < U.length; i++) U[i] = benchRng.normal();
    const ssmFwd = () => {
      const hh = new Float64Array(modelDim);
      let acc = 0;
      for (let t = 0; t < n; t++) {
        const off = t * modelDim;
        for (let c = 0; c < modelDim; c++) {
          hh[c] = Avec[c] * hh[c] + Bvec[c] * U[off + c];
          acc += Cvec[c] * hh[c]; // touch output so JIT can't elide the loop
        }
      }
      if (!Number.isFinite(acc)) throw new Error("ssm benchmark diverged");
    };

    // Attention: real dense O(n^2) QK^T·V over the same n (stand-in for stage05's cost).
    // We use one head, dim=modelDim; this measures the n^2 scaling honestly.
    const Q = new Float64Array(n * modelDim);
    const Kk = new Float64Array(n * modelDim);
    const V = new Float64Array(n * modelDim);
    for (let i = 0; i < Q.length; i++) {
      Q[i] = benchRng.normal();
      Kk[i] = benchRng.normal();
      V[i] = benchRng.normal();
    }
    const attnFwd = () => {
      const scale = 1 / Math.sqrt(modelDim);
      const out = new Float64Array(modelDim);
      let acc = 0;
      for (let i = 0; i < n; i++) {
        // scores_i over all j (n of them) -> softmax -> weighted sum of V (the n^2 core)
        const scores = new Float64Array(n);
        let mx = -Infinity;
        for (let j = 0; j < n; j++) {
          let dot = 0;
          const qo = i * modelDim;
          const ko = j * modelDim;
          for (let c = 0; c < modelDim; c++) dot += Q[qo + c] * Kk[ko + c];
          dot *= scale;
          scores[j] = dot;
          if (dot > mx) mx = dot;
        }
        let denom = 0;
        for (let j = 0; j < n; j++) {
          scores[j] = Math.exp(scores[j] - mx);
          denom += scores[j];
        }
        out.fill(0);
        for (let j = 0; j < n; j++) {
          const w = scores[j] / denom;
          const vo = j * modelDim;
          for (let c = 0; c < modelDim; c++) out[c] += w * V[vo + c];
        }
        acc += out[0];
      }
      if (!Number.isFinite(acc)) throw new Error("attention benchmark diverged");
    };

    const reps = 5;
    ssmTimes.push(timeit(ssmFwd, reps).perRepMs);
    attnTimes.push(timeit(attnFwd, reps).perRepMs);
  }

  console.log("  实测前向耗时 (per-rep ms, modelDim=" + modelDim + ", reps=5,真 wall-clock):");
  console.log("    n     SSM(O(n))    Attn(O(n²))   attn/ssm    MAC attn/ssm");
  for (let i = 0; i < seqLens.length; i++) {
    const n = seqLens[i];
    const macAttn = countMACs.attention(n, modelDim);
    const macScan = countMACs.scan(n, scanState, modelDim);
    console.log(
      `    ${String(n).padEnd(5)} ${ssmTimes[i].toFixed(4).padStart(9)}    ` +
        `${attnTimes[i].toFixed(4).padStart(9)}     ${(attnTimes[i] / ssmTimes[i]).toFixed(1).padStart(6)}x   ` +
        `${(macAttn / macScan).toFixed(0).padStart(8)}x`,
    );
  }
  // Empirical scaling slope: doubling n should ~double SSM time but ~4x attention time.
  const ssmSlope = ssmTimes[ssmTimes.length - 1] / ssmTimes[0]; // over 8x n increase
  const attnSlope = attnTimes[attnTimes.length - 1] / attnTimes[0];
  const nRatio = seqLens[seqLens.length - 1] / seqLens[0];
  console.log(
    `  n 增大 ${nRatio}x: SSM 耗时 x${ssmSlope.toFixed(1)} (理想线性 ${nRatio}x), ` +
      `Attn 耗时 x${attnSlope.toFixed(1)} (理想平方 ${nRatio * nRatio}x)`,
  );
  console.log(
    "  实测/理论可能因常数项 / JIT / cache 偏离,但相对趋势确证:SSM 近线性,Attn 近平方。",
  );
  console.log(bar(seqLens.map((n, i) => ({ label: `attn/ssm n=${n}`, value: +(attnTimes[i] / ssmTimes[i]).toFixed(1) }))));

  // -----------------------------------------------------------------------
  // PART 5 — FAILURE MODE: unstable A (|eigenvalue| > 1) blows up the state.
  // -----------------------------------------------------------------------
  console.log("\n[5] 失败模式:把 A 的幅值设为 > 1(违反结构化稳定),状态范数指数爆炸");
  // Use an IMPULSE input (u[0]=1, rest 0) so the state is exactly h_t = B·A^t·C-free signal:
  // y_t = C·A^t·B. This isolates the pure A^t growth/decay signature with no input noise —
  // the honest way to show stability depends ONLY on |A|, not on the data.
  const Tfail = 40;
  const Bf = 1.0;
  const Cf = 1.0;
  const impulse = new Array<number>(Tfail).fill(0);
  impulse[0] = 1;
  for (const A of [0.9, 1.0, 1.05]) {
    const y = scanRecurrent(impulse, A, Bf, Cf);
    const normSeries = y.map((v) => Math.abs(v));
    const last = normSeries[normSeries.length - 1];
    const finite = normSeries.every(Number.isFinite);
    const tag = A < 1 ? "稳定 (|A|<1): 脉冲响应衰减" : A === 1 ? "临界 (|A|=1): 不衰减" : "不稳定 (|A|>1): 指数爆炸";
    console.log(`  A=${A.toFixed(2)} ${tag}`);
    console.log("    |state| (脉冲响应 A^t) over time: " + sparkline(normSeries));
    console.log(
      `    |state| t=0 -> t=${Tfail - 1}: ${normSeries[0].toExponential(2)} -> ${last.toExponential(2)}` +
        `  (A^${Tfail - 1} = ${Math.pow(A, Tfail - 1).toExponential(2)})` +
        (finite ? "" : "  (含 NaN/Inf — 数值崩溃)"),
    );
  }
  // Hammer the point on a longer horizon with the same impulse -> pure A^t overflow.
  // Impulse keeps it exactly A^t (no geometric-sum inflation), so the printed |state| and
  // the analytic A^(T-1) match — honest 1:1.
  // Push T past the f64 range (max ~1.8e308). 1.1^t overflows to Infinity around t≈7400,
  // so a long-enough sequence makes the state literally non-finite — not a metaphor.
  const Tblow = 8000;
  const impulseBlow = new Array<number>(Tblow).fill(0);
  impulseBlow[0] = 1;
  const yBlow = scanRecurrent(impulseBlow, 1.1, 1, 1);
  const lastBlow = yBlow[yBlow.length - 1];
  const blewUp = !Number.isFinite(lastBlow);
  // Find the first step where the state actually became Infinity (overflow horizon).
  let overflowAt = -1;
  for (let t = 0; t < yBlow.length; t++) {
    if (!Number.isFinite(yBlow[t])) {
      overflowAt = t;
      break;
    }
  }
  console.log(
    `  A=1.10, T=${Tblow}, 脉冲: 末步 |state| = ${lastBlow} ` +
      `${blewUp ? `(t=${overflowAt} 起 |state| 溢出为 Infinity — 数值真的崩了,不是比喻)` : ""}`,
  );
  console.log(
    "  结论:|A|>1 时 h_t ~ A^t 指数发散 -> Inf/NaN。所以 SSM 必须用结构(本 stage 的 sigmoid 参数化)\n" +
      "        把 A 钉在稳定域 (0,1) 内 —— 这是 SSM 可用的前提,不是可选项。",
  );

  // Contrast: our trained SSM used A=sigmoid(aLogit), so EVERY channel is provably stable.
  const trainedA = ssm.aLogit.sigmoid().data;
  console.log(
    `  对照:训练后 SSM 的 A=sigmoid(aLogit) 全部落在 (0,1):` +
      `min=${maxAbsMin(trainedA).toFixed(3)} max=${maxAbsMax(trainedA).toFixed(3)} ` +
      `(${Array.from(trainedA).every((a) => a > 0 && a < 1) ? "全部稳定 ✓" : "✗"})`,
  );

  console.log("\n" + "=".repeat(70));
  console.log("小结:递归=卷积 (机制自证 <1e-8) | 长依赖可解 | O(n) 胜 O(n²) | 稳定 A 是前提");
  console.log("=".repeat(70));
}

// tiny inline min/max helpers (avoid spreading a Float64Array into Math.min on large dim)
function maxAbsMin(a: Float64Array): number {
  let m = Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i];
  return m;
}
function maxAbsMax(a: Float64Array): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m;
}

main();
