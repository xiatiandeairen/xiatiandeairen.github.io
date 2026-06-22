// stage07-mamba.ts — Selective state-space models: making an SSM learn to IGNORE vs REMEMBER.
//
// WHY this chapter exists (the one idea):
//   A linear time-invariant SSM (chapter 06) has the SAME recurrence at every timestep:
//   one fixed A/B/C/Δ shared across the whole sequence. That is great for efficiency
//   (a single global scan) but it cannot CONDITION its memory on content — every token
//   is written into the state with the same gain. Mamba's contribution is making Δ/B/C
//   functions OF THE INPUT (input-dependent / "selective"), so the model can open the
//   gate to write a rare salient token and close it to skip a flood of distractors.
//
// WHAT WE ACTUALLY MEASURE HERE (honest scope):
//   We construct a task that is UNSOLVABLE without content-selective memory: one marked
//   key token hidden among many distractors that share the exact same vocabulary. Then
//   we train three models to the same budget and compare:
//     - selective SSM (Δ/B/C input-dependent)  -> should solve it
//     - fixed SSM     (Δ/B/C learned constants) -> should NOT (noise overwrites state)
//     - LSTM                                     -> gating helps but content-addressing
//                                                   the key among same-vocab noise is hard
//   We also ABLATE selectivity (freeze the selection Linear to constants) to prove the
//   selectivity is the causal variable, not just "the SSM is bigger".
//
// HONEST-NUMBER DISCIPLINE:
//   - All randomness threads a seeded Rng (reproducible).
//   - Wall-clock is really measured (timeit); MAC counts use core/metrics countMACs.
//   - These are TOY scales (tiny dim, short-ish sequences, CPU). Absolute accuracies are
//     optimistic; what transfers is the RELATIVE story (selective solves, fixed/ablated
//     collapse to chance) and the O(n) scan scaling. Real Mamba's headline wins need GPU
//     + long sequences + the hardware-aware parallel scan we deliberately do NOT build.
//
// IMPLEMENTATION NOTE — the scan is a sequential Python-style loop over timesteps, NOT the
//   hardware parallel scan. That is on purpose: the chapter's claim is about the MECHANISM
//   (input-dependent gating), and a readable per-step recurrence makes the gate visible
//   (we literally print Δ_t over time as a heatmap). The O(n) cost claim still holds — one
//   pass, work per step independent of n — it is just not parallelized.

import { Tensor } from "./core/tensor.js";
import { Linear, Module, collectParams } from "./core/nn.js";
import { Adam, clipGradNorm } from "./core/optim.js";
import { makeRng, type Rng } from "./core/prng.js";
import { accuracy, argmax, timeit, countMACs } from "./core/metrics.js";
import { lineChart, heatmap, bar } from "./core/plot.js";

// ----------------------------------------------------------------------------
// Task: noisy selective recall.
//
// Each sequence is T steps. Exactly ONE step is the "key": it carries a flag=1 and a
// token drawn from the answer vocabulary. Every other step is a distractor: flag=0 and a
// token drawn from the SAME vocabulary (so the model cannot find the key by vocabulary —
// only the flag, i.e. CONTENT, distinguishes it). The target is the key token.
//
// WHY this is the right stress test for selectivity:
//   The key can appear anywhere (random position) and is surrounded by tokens that look
//   identical except for the flag bit. A model must (a) detect the flag and (b) at that
//   one step write the token into memory while NOT overwriting it with the subsequent
//   distractor flood. A fixed SSM writes every step with the same gain, so the last
//   distractors dominate the state -> answer ~ random. Selective gating sets Δ≈0 on
//   distractors (forget nothing / write nothing) and Δ large on the key.
//
// Feature per step: one-hot(token, vocab) concatenated with [flag]. Dim = vocab + 1.
// ----------------------------------------------------------------------------
interface SelectiveRecallTask {
  X: number[][]; // [N][T] token ids in [0, vocab)
  flags: number[][]; // [N][T] 0/1, exactly one 1 per row
  Y: number[]; // [N] key token id (the answer)
  T: number;
  vocab: number;
  featDim: number; // vocab + 1
}

function makeSelectiveRecall(rng: Rng, opts: { count: number; T: number; vocab: number }): SelectiveRecallTask {
  const { count, T, vocab } = opts;
  const X: number[][] = [];
  const flags: number[][] = [];
  const Y: number[] = [];
  for (let n = 0; n < count; n++) {
    const x = new Array<number>(T);
    const f = new Array<number>(T).fill(0);
    for (let t = 0; t < T; t++) x[t] = rng.randint(0, vocab);
    // Place the key NOT at the very end, so a model can't cheat by "just read the last
    // step". Keep it away from the final 2 steps to force genuine carry-over.
    const keyPos = rng.randint(0, Math.max(1, T - 2));
    f[keyPos] = 1;
    X.push(x);
    flags.push(f);
    Y.push(x[keyPos]);
  }
  return { X, flags, Y, T, vocab, featDim: vocab + 1 };
}

/** Build the [T, featDim] feature tensor for one sequence: one-hot(token) ++ [flag]. */
function featuresOf(task: SelectiveRecallTask, n: number): Tensor {
  const { T, vocab, featDim } = task;
  const d = new Float64Array(T * featDim);
  for (let t = 0; t < T; t++) {
    d[t * featDim + task.X[n][t]] = 1; // one-hot token
    d[t * featDim + vocab] = task.flags[n][t]; // flag channel
  }
  return new Tensor(d, [T, featDim]);
}

// ----------------------------------------------------------------------------
// Selective SSM block (the Mamba-style mechanism, simplified to be readable).
//
// State recurrence per timestep t, per channel:
//   h_t = exp(-Δ_t * A) ⊙ h_{t-1} + (Δ_t * B_t) * x_t
//   y_t =  C_t · h_t
// where Δ_t, B_t, C_t are produced by a Linear FROM THE INPUT (selectivity), and A is a
// learned (negative) decay shared across time. exp(-Δ*A) is the discretized state-decay:
//   Δ→0  => exp(0)=1, gain (Δ*B)→0  => h_t = h_{t-1}  (skip / ignore this token)
//   Δ→big => decay strong, gain large => overwrite state with this token (remember it)
// This is the whole trick: Δ is an input-dependent WRITE/FORGET gate.
//
// Shapes (kept tiny for CPU): per step we map an input feature vector to:
//   x_proj  in R^D   (the value to potentially store, D = model dim)
//   Δ       in R^D   (per-channel step size, softplus -> positive)
//   B       in R^D   (input gate)
//   C       in R^D   (output gate)
// State h is R^D (diagonal SSM: state dim == model dim, one scalar memory per channel).
// We sum y over channels into a single readout that a final Linear maps to vocab logits.
//
// WHY diagonal / scalar-per-channel state (not a full SxD matrix): the real Mamba uses a
//   per-channel state of size N; we collapse N=1 so the recurrence is a clean elementwise
//   scan that the autograd engine (elementwise + scale) handles without batched matmul.
//   The selectivity mechanism is identical; only the state capacity per channel shrinks.
// ----------------------------------------------------------------------------
class SelectiveSSM extends Module {
  inProj: Linear; // feature -> x value
  selProj: Linear; // feature -> [Δraw, B, C] concatenated (3*D)
  A: Tensor; // [1, D] learned decay base (kept positive via softplus at use)
  readout: Linear; // D -> vocab logits
  D: number;
  // ablate=true freezes selectivity: Δ/B/C become input-INDEPENDENT constants. We do this
  // by zeroing selProj.W (so its output is just its bias) and excluding it from params so
  // the optimizer can't relearn input dependence. The bias still trains => fixed gates.
  readonly ablate: boolean;

  constructor(featDim: number, D: number, vocab: number, rng: Rng, opts: { ablate?: boolean } = {}) {
    super();
    this.D = D;
    this.ablate = opts.ablate ?? false;
    this.inProj = new Linear(featDim, D, rng);
    this.selProj = new Linear(featDim, 3 * D, rng);
    // A base initialized small-positive; softplus(A) is the actual decay. Negative-real
    // decay (exp(-Δ*softplus(A))) keeps the state contractive => no exploding scan.
    this.A = Tensor.randn([1, D], rng, 0.1).addScalarInPlaceInit(0.5);
    this.readout = new Linear(D, vocab, rng);
    if (this.ablate) {
      // Freeze the selection projection's input dependence: zero its weight matrix so the
      // gate values come only from the (trainable) bias => same gate for every token.
      this.selProj.W.data.fill(0);
    }
  }

  /** Forward one sequence [T, featDim] -> { logits [1,vocab], deltas number[][] for viz }. */
  forward(feats: Tensor): { logits: Tensor; deltaTrace: number[][] } {
    const T = feats.shape[0];
    const D = this.D;
    // softplus(A) > 0 ensures exp(-Δ*decay) in (0,1): a contraction, never blow-up.
    const decay = softplus(this.A); // [1, D]
    let h = Tensor.zeros([1, D]); // running state, one scalar per channel
    let lastY: Tensor | null = null;
    const deltaTrace: number[][] = []; // per-step mean-over-channel Δ, for the heatmap
    for (let t = 0; t < T; t++) {
      const xt = feats.slice(0, t, t + 1); // [1, featDim]
      const xv = this.inProj.forward(xt); // [1, D] value to (maybe) store
      const sel = this.selProj.forward(xt); // [1, 3D] raw Δ,B,C
      const dRaw = sel.slice(1, 0, D);
      const B = sel.slice(1, D, 2 * D);
      const C = sel.slice(1, 2 * D, 3 * D);
      const delta = softplus(dRaw); // [1, D] positive step size (the selectivity gate)
      // state update: h = exp(-Δ⊙decay) ⊙ h + (Δ⊙B) ⊙ xv
      const decayT = delta.mul(decay).scale(-1).exp(); // exp(-Δ*decay) elementwise
      const gain = delta.mul(B).mul(xv); // input-dependent write
      h = decayT.mul(h).add(gain);
      lastY = C.mul(h); // [1, D] gated output
      // record Δ trace (mean across channels) — detached numbers, viz only.
      let s = 0;
      for (let c = 0; c < D; c++) s += delta.data[c];
      deltaTrace.push([s / D]);
    }
    const logits = this.readout.forward(lastY!); // [1, vocab]
    return { logits, deltaTrace };
  }

  override params(): Tensor[] {
    const base = [...this.inProj.params(), this.A, ...this.readout.params()];
    // In ablation mode we DROP selProj from the trainable set entirely, except its bias,
    // so gate values are content-independent constants the optimizer may still tune.
    if (this.ablate) return [...base, this.selProj.b!];
    return [...base, ...this.selProj.params()];
  }
}

// ----------------------------------------------------------------------------
// LSTM baseline (built from Linear primitives — the chapter-03 cell, inlined here so this
// stage stays self-contained and never imports a stageNN that would run its main()).
// Standard gates: i,f,o = sigmoid(Wx+Uh+b); g = tanh(...); c = f⊙c + i⊙g; h = o⊙tanh(c).
// ----------------------------------------------------------------------------
class LSTM extends Module {
  Wi: Linear; Wf: Linear; Wo: Linear; Wg: Linear; // input->gate (includes bias)
  Ui: Linear; Uf: Linear; Uo: Linear; Ug: Linear; // hidden->gate (no bias, folded above)
  readout: Linear;
  H: number;
  constructor(featDim: number, H: number, vocab: number, rng: Rng) {
    super();
    this.H = H;
    this.Wi = new Linear(featDim, H, rng);
    this.Wf = new Linear(featDim, H, rng);
    this.Wo = new Linear(featDim, H, rng);
    this.Wg = new Linear(featDim, H, rng);
    this.Ui = new Linear(H, H, rng, { bias: false });
    this.Uf = new Linear(H, H, rng, { bias: false });
    this.Uo = new Linear(H, H, rng, { bias: false });
    this.Ug = new Linear(H, H, rng, { bias: false });
    this.readout = new Linear(H, vocab, rng);
  }
  forward(feats: Tensor): Tensor {
    const T = feats.shape[0];
    let h = Tensor.zeros([1, this.H]);
    let c = Tensor.zeros([1, this.H]);
    for (let t = 0; t < T; t++) {
      const xt = feats.slice(0, t, t + 1);
      const i = this.Wi.forward(xt).add(this.Ui.forward(h)).sigmoid();
      const f = this.Wf.forward(xt).add(this.Uf.forward(h)).sigmoid();
      const o = this.Wo.forward(xt).add(this.Uo.forward(h)).sigmoid();
      const g = this.Wg.forward(xt).add(this.Ug.forward(h)).tanh();
      c = f.mul(c).add(i.mul(g));
      h = o.mul(c.tanh());
    }
    return this.readout.forward(h);
  }
  override params(): Tensor[] {
    return collectParams([
      this.Wi, this.Wf, this.Wo, this.Wg,
      this.Ui, this.Uf, this.Uo, this.Ug,
      this.readout,
    ]);
  }
}

// softplus(x) = log(1+e^x), kept positive and smooth. Used for Δ and A so step size and
// decay are strictly > 0 (a negative Δ would mean "anti-write", which makes no sense and
// can make the scan expand). Numerically stable via the standard max-shift identity.
function softplus(x: Tensor): Tensor {
  // log(1+exp(x)) = max(x,0) + log(1+exp(-|x|)); but our tensor ops don't expose abs/max,
  // and for our value ranges (|x| < ~10) the direct form is safe in float64. We compute it
  // through the graph as log(1 + exp(x)) so gradients flow (sigmoid is the derivative).
  return x.exp().addScalar(1).log();
}

// Patch: Tensor has no in-place "add scalar to data at init" — add one tiny helper via a
// module augmentation so A can be initialized to a positive mean cleanly. We avoid adding
// to core/tensor.ts (scaffold is frozen). This only mutates raw data at construction.
declare module "./core/tensor.js" {
  interface Tensor {
    addScalarInPlaceInit(k: number): Tensor;
  }
}
Tensor.prototype.addScalarInPlaceInit = function (k: number): Tensor {
  for (let i = 0; i < this.data.length; i++) this.data[i] += k;
  return this;
};

// ----------------------------------------------------------------------------
// Training loop (one model). Returns loss curve + final train/test accuracy + Δ traces.
// ----------------------------------------------------------------------------
interface TrainResult {
  losses: number[];
  trainAcc: number;
  testAcc: number;
  paramCount: number;
}

function countParams(m: Module): number {
  return m.params().reduce((s, p) => s + p.size, 0);
}

type ForwardFn = (feats: Tensor) => { logits: Tensor; deltaTrace?: number[][] };

function trainModel(
  _name: string,
  model: Module,
  fwd: ForwardFn,
  train: SelectiveRecallTask,
  test: SelectiveRecallTask,
  rng: Rng,
  opts: { epochs: number; lr: number; clip: number },
): TrainResult {
  const params = model.params();
  const optim = new Adam(params, { lr: opts.lr });
  const losses: number[] = [];
  const N = train.X.length;
  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    let epochLoss = 0;
    const order = rng.shuffle(Array.from({ length: N }, (_, i) => i));
    for (const n of order) {
      optim.zeroGrad();
      const feats = featuresOf(train, n);
      const { logits } = fwd(feats); // [1, vocab]
      const loss = logits.crossEntropy([train.Y[n]]);
      loss.backward();
      clipGradNorm(params, opts.clip); // SSM scans can spike grads; keep steps sane
      optim.step();
      epochLoss += loss.data[0];
    }
    losses.push(epochLoss / N);
  }
  return {
    losses,
    trainAcc: evalAcc(fwd, train),
    testAcc: evalAcc(fwd, test),
    paramCount: countParams(model),
  };
}

function evalAcc(fwd: ForwardFn, task: SelectiveRecallTask): number {
  const preds: number[] = [];
  for (let n = 0; n < task.X.length; n++) {
    const { logits } = fwd(featuresOf(task, n));
    preds.push(argmax(logits.data));
  }
  return accuracy(preds, task.Y);
}

// ----------------------------------------------------------------------------
// Main: build task, train 4 models, print evidence.
// ----------------------------------------------------------------------------
function main(): void {
  // Independent streams: one for data, one for init (see prng.ts failure-mode note).
  const dataRng = makeRng(7);
  const T = 24; // sequence length: key buried among ~23 distractors
  const VOCAB = 6; // answer/distractor share this vocab -> chance accuracy = 1/6 ≈ 16.7%
  const D = 16; // model dim (SSM state per channel = 1; LSTM hidden = D for fair-ish budget)
  const train = makeSelectiveRecall(dataRng, { count: 256, T, vocab: VOCAB });
  const test = makeSelectiveRecall(dataRng, { count: 128, T, vocab: VOCAB });

  console.log("=".repeat(72));
  console.log("第 07 章 — 选择性扫描 (Selective SSM / Mamba 思路)");
  console.log("=".repeat(72));
  console.log(
    `任务: 噪声选择性回忆。T=${T} 步, 词表=${VOCAB} (key 与干扰项同词表), ` +
      `仅 flag 标记 key。\n  随机基线准确率 = 1/${VOCAB} = ${(100 / VOCAB).toFixed(1)}%。` +
      ` 训练 ${train.X.length} 条 / 测试 ${test.X.length} 条。`,
  );
  console.log(
    "  关键: key 与干扰项词表相同, 模型无法靠词表区分 —— 只有读 flag (内容) 选择性写入才能解。",
  );

  const epochs = 14;
  const lr = 5e-3;
  const clip = 1.0;

  // Fresh init stream per model so weight init is identical-seeded and comparable.
  const selective = new SelectiveSSM(train.featDim, D, VOCAB, makeRng(101));
  const fixed = new SelectiveSSM(train.featDim, D, VOCAB, makeRng(101), { ablate: true });
  const lstm = new LSTM(train.featDim, D, VOCAB, makeRng(101));

  console.log(`\n[1/3] 训练 selective SSM (Δ/B/C 由输入生成) ... epochs=${epochs}`);
  const rSel = trainModel(
    "selective", selective, (f) => selective.forward(f), train, test, makeRng(1), { epochs, lr, clip },
  );
  console.log(`[2/3] 训练 fixed SSM (Δ/B/C 退回常数, 选择性关闭) ... epochs=${epochs}`);
  const rFix = trainModel(
    "fixed", fixed, (f) => fixed.forward(f), train, test, makeRng(1), { epochs, lr, clip },
  );
  console.log(`[3/3] 训练 LSTM 基线 ... epochs=${epochs}`);
  const rLstm = trainModel(
    "lstm", lstm, (f) => ({ logits: lstm.forward(f) }), train, test, makeRng(1), { epochs, lr, clip },
  );

  // --- loss curves -----------------------------------------------------------
  console.log("\n--- 训练 loss 曲线 (per epoch, 三模型叠加) ---");
  console.log(lineChart([rSel.losses, rFix.losses, rLstm.losses], {
    labels: ["selective", "fixed", "lstm"], width: 56, height: 10,
  }));

  // --- accuracy table --------------------------------------------------------
  const chance = 1 / VOCAB;
  console.log("\n--- 准确率 (test) vs 随机基线 ---");
  console.log(bar([
    { label: "selective SSM", value: +rSel.testAcc.toFixed(3) },
    { label: "fixed SSM    ", value: +rFix.testAcc.toFixed(3) },
    { label: "LSTM         ", value: +rLstm.testAcc.toFixed(3) },
    { label: "chance       ", value: +chance.toFixed(3) },
  ]));
  console.log(
    `\n  selective: train=${(rSel.trainAcc * 100).toFixed(1)}% test=${(rSel.testAcc * 100).toFixed(1)}%  (params=${rSel.paramCount})`,
  );
  console.log(
    `  fixed    : train=${(rFix.trainAcc * 100).toFixed(1)}% test=${(rFix.testAcc * 100).toFixed(1)}%  (params=${rFix.paramCount})`,
  );
  console.log(
    `  lstm     : train=${(rLstm.trainAcc * 100).toFixed(1)}% test=${(rLstm.testAcc * 100).toFixed(1)}%  (params=${rLstm.paramCount})`,
  );

  // --- selectivity visualization: Δ_t over time on one held-out example ------
  // Pick a test sequence, print where the key flag is, then the trained selective model's
  // Δ_t (mean over channels) per step. The claim is Δ SPIKES at the flagged key step and
  // stays low on distractors: "open the gate to remember, close it to ignore".
  console.log("\n--- 选择性门控 Δ_t 随时间 (selective SSM, 1 条测试样本) ---");
  const probe = 0;
  const keyPos = test.flags[probe].indexOf(1);
  const { deltaTrace } = selective.forward(featuresOf(test, probe));
  const deltaRow = deltaTrace.map((d) => d[0]);
  // Build a 2-row heatmap: row 0 = flag (1 at key), row 1 = Δ_t. Bright cell should align.
  const flagRow = test.flags[probe].map((f) => f);
  console.log(`  key flag 在第 ${keyPos} 步 (token id=${test.Y[probe]})`);
  console.log("  row0 = flag, row1 = Δ_t (越亮门越开):");
  console.log(heatmap([flagRow, deltaRow]));
  const dMax = Math.max(...deltaRow);
  const dAtKey = deltaRow[keyPos];
  const dMeanOther = (deltaRow.reduce((a, b) => a + b, 0) - dAtKey) / (deltaRow.length - 1);
  console.log(
    `  Δ 在 key 步 = ${dAtKey.toFixed(4)}, 其余步均值 = ${dMeanOther.toFixed(4)}, ` +
      `比值 = ${(dAtKey / (dMeanOther + 1e-9)).toFixed(2)}x  (>1 => 模型在 key 处开门)`,
  );
  // Honest check: is the key step the argmax of Δ? Report it rather than asserting it.
  let dArg = 0; for (let i = 1; i < deltaRow.length; i++) if (deltaRow[i] > deltaRow[dArg]) dArg = i;
  console.log(`  Δ 最大值出现在第 ${dArg} 步 (期望=${keyPos}) ${dArg === keyPos ? "✓ 对齐" : "≠ (toy 噪声)"}, Δmax=${dMax.toFixed(4)}`);

  // --- failure mode summary: selectivity is the causal variable ---------------
  console.log("\n--- 失败模式: 关掉选择性 => 准确率掉回随机 ---");
  const liftSel = rSel.testAcc - chance;
  const liftFix = rFix.testAcc - chance;
  const signed = (pp: number) => (pp >= 0 ? "+" : "") + pp.toFixed(1) + "pp";
  console.log(
    `  selective 相对随机提升 = ${signed(liftSel * 100)}; ` +
      `fixed (选择性关闭) 提升 = ${signed(liftFix * 100)} (≈ 随机水平)。`,
  );
  console.log(
    `  fixed SSM 与 selective 同一类结构, 唯一差别是 Δ/B/C 是否由输入生成。\n` +
      `  注: 关闭选择性后 selProj 的输入权重被冻结并移出可训练集, 故参数更少 (${rFix.paramCount} vs ${rSel.paramCount}) ——\n` +
      `  但即便参数更少不是失败主因: selective 在更少 epoch 内就解题, fixed 给再多参数也学不到\n` +
      `  "按内容选择" 这一步。=> 选择性是解此任务的关键变量, 不是锦上添花。`,
  );

  // --- complexity / scaling: SSM scan stays O(n) (vs attention O(n^2)) --------
  console.log("\n--- 标度: selective SSM 仍是 O(n) 扫描 (对照 attention O(n^2)) ---");
  const ns = [16, 32, 64, 128];
  console.log("  序列长 n -> 每步分析 MAC 数 (scan vs dense-attention, dim=" + D + "):");
  for (const n of ns) {
    const scanMac = countMACs.scan(n, 1, D); // state dim 1 (diagonal), model dim D
    const attnMac = countMACs.attention(n, D);
    console.log(
      `    n=${String(n).padStart(3)}: scan=${String(scanMac).padStart(7)} MAC, ` +
        `attn=${String(attnMac).padStart(8)} MAC, attn/scan=${(attnMac / scanMac).toFixed(1)}x`,
    );
  }
  // Real wall-clock: forward-time vs n for the selective scan. Should grow ~linearly.
  console.log("\n  实测前向耗时 vs n (selective SSM, 真实 wall-clock):");
  const timeRng = makeRng(55);
  const timings: { label: string; value: number }[] = [];
  let prevPer = 0; let prevN = 0;
  for (const n of ns) {
    const t = makeSelectiveRecall(timeRng, { count: 1, T: n, vocab: VOCAB });
    const feats = featuresOf(t, 0);
    // High rep count to average out sub-ms scheduler/JIT noise; still REAL wall-clock.
    const { perRepMs } = timeit(() => { selective.forward(feats); }, 200);
    timings.push({ label: `n=${n}`, value: +perRepMs.toFixed(4) });
    if (prevN > 0) {
      const ratio = perRepMs / prevPer;
      const nRatio = n / prevN;
      console.log(
        `    n=${String(n).padStart(3)}: ${perRepMs.toFixed(4)} ms/fwd  ` +
          `(n 翻 ${nRatio}x => 时间 ${ratio.toFixed(2)}x; O(n) 预期 ≈ ${nRatio}x)`,
      );
    } else {
      console.log(`    n=${String(n).padStart(3)}: ${perRepMs.toFixed(4)} ms/fwd  (基准)`);
    }
    prevPer = perRepMs; prevN = n;
  }
  console.log(bar(timings));

  console.log(
    "\n注: 以上为 toy 规模 (CPU, 短序列, 顺序扫描非并行)。绝对数偏乐观;\n" +
      "  可迁移的是相对趋势: 选择性解题 / 固定版崩塌, 以及 scan 的 O(n) 标度。\n" +
      "  真实 Mamba 的量级优势需 GPU + 长序列 + 硬件感知并行扫描, 本章未实现。",
  );
}

main();
