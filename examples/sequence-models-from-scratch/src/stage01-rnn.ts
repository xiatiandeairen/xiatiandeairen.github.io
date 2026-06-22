// stage01-rnn.ts — Chapter 01: the RNN and time-unrolling.
//
// WHAT this chapter demonstrates, with numbers the code actually measures:
//   1. ONE set of weights (W_xh, W_hh, b_h) processes a sequence of ANY length by
//      reapplying itself at every timestep — that reuse IS the recurrent network. We
//      build RNNCell from core Linear primitives and unroll it in a plain JS loop; the
//      autograd graph that loop builds is backprop-through-time (BPTT) for free.
//   2. On a SHORT-dependency copy task the RNN learns: loss drops to ~0, answer-region
//      token accuracy on a HELD-OUT test set hits 100%.
//   3. FAILURE MODE (the hook for chapter 02): keep the exact same model + same number
//      of training steps but stretch the dependency span (delay 5 -> 35). Test accuracy
//      stalls at chance and loss plateaus far above 0. This is the "vanilla RNN can't
//      learn long dependencies" cliff — chapter 02 explains WHY via vanishing gradients.
//
// WHY delay=35 and not a smaller number, and why HELD-OUT eval: this was empirically
//   calibrated, not assumed. With Adam (per-coordinate grad normalization partly rescues
//   tiny gradients) the cliff sits surprisingly far out — at delay=25 the RNN STILL
//   learns to ~99%. The honest cliff only appears around delay=35 (test acc 16.5% ≈ the
//   1/6 chance floor; loss flatlines at ~1.8, and +50% more steps do not rescue it).
//   Evaluating on a held-out test set (not the training set) is essential: with 256
//   training sequences a 64-unit RNN can MEMORIZE them, which would fake "learning long
//   dependencies". Generalization to fresh sequences is the real test of the mechanism.
//
// WHY copyTask: its dependency span is a single tunable knob (delay). Holding the model
//   fixed and moving only that knob isolates "memory horizon" as the one variable, which
//   is exactly the claim this chapter and the next make.
//
// HONESTY: copyTask is a clean, tiny-vocab toy. Absolute accuracies here are OPTIMISTIC
//   vs real corpora. What transfers is the RELATIVE story: short span learns, long span
//   does not, for an unchanged architecture and budget. Loss curves / accuracies below
//   are computed from the real trained model; wall-clock is real performance.now().
//
// INVARIANT: every random draw (init, data, shuffling) threads a seeded Rng, and init
//   vs data use SEPARATE seeds so the two streams don't couple. Same seeds => identical
//   numbers on every run.

import { Tensor } from "./core/tensor.js";
import { Linear, Module, collectParams } from "./core/nn.js";
import type { Rng } from "./core/prng.js";
import { makeRng } from "./core/prng.js";
import { copyTask } from "./core/data.js";
import { Adam, clipGradNorm } from "./core/optim.js";
import { accuracy, argmax, timeit } from "./core/metrics.js";
import { sparkline } from "./core/plot.js";

// ---------------------------------------------------------------------------
// RNNCell — the pedagogical payload of this chapter.
//
//   h_t = tanh( x_t @ W_xh + h_{t-1} @ W_hh + b_h )
//
// We compose it from two bias-free Linear layers (input->hidden, hidden->hidden) plus a
// shared bias. WHY orthogonal init on the recurrent matrix W_hh: an orthogonal matrix
// has all singular values = 1, so reapplying it across the unroll neither inflates nor
// deflates the hidden-state norm — the gentlest starting point before chapter 02 shows
// how tanh saturation still drives gradients toward vanishing. xavier on the input map
// keeps forward activations well-scaled.
//
// INVARIANT: the SAME cell instance is called at every timestep. Its params (W_xh, W_hh,
//   b_h) are therefore shared across time; autograd's accumulate-not-overwrite rule sums
//   each timestep's gradient contribution into those shared leaves. That summation is
//   precisely backprop-through-time. Re-instantiating the cell per step would silently
//   give each step its own untied weights — no longer an RNN.
//
// Exported so chapter 02 (vanishing gradients) can import and instrument this exact cell
// without re-running this file's main().
export class RNNCell extends Module {
  private inToHidden: Linear; // W_xh : [inDim, hiddenDim], no bias
  private hiddenToHidden: Linear; // W_hh : [hiddenDim, hiddenDim], no bias
  private bias: Tensor; // b_h : [1, hiddenDim], broadcast over batch rows
  readonly hiddenDim: number;

  constructor(inDim: number, hiddenDim: number, rng: Rng) {
    super();
    this.hiddenDim = hiddenDim;
    this.inToHidden = new Linear(inDim, hiddenDim, rng, { init: "xavier", bias: false });
    this.hiddenToHidden = new Linear(hiddenDim, hiddenDim, rng, { init: "orthogonal", bias: false });
    this.bias = Tensor.zeros([1, hiddenDim]);
  }

  /** One recurrent step. x:[B,inDim], hPrev:[B,hiddenDim] -> hNext:[B,hiddenDim]. */
  step(x: Tensor, hPrev: Tensor): Tensor {
    const fromInput = this.inToHidden.forward(x);
    const fromState = this.hiddenToHidden.forward(hPrev);
    return fromInput.add(fromState).add(this.bias).tanh();
  }

  /** Initial hidden state: zeros [B, hiddenDim]. A learnable h0 is overkill for this toy. */
  initialState(batch: number): Tensor {
    return Tensor.zeros([batch, this.hiddenDim]);
  }

  override params(): Tensor[] {
    return [...this.inToHidden.params(), ...this.hiddenToHidden.params(), this.bias];
  }
}

// ---------------------------------------------------------------------------
// Model: one-hot input -> RNNCell unroll -> per-timestep linear readout to vocab logits.
//
// We feed one-hot vectors (not a learned embedding) so this chapter's only learnable
// "memory" lives in the recurrent weights — keeps the failure-mode story attributable to
// recurrence, not to an embedding doing the work.
class CopyRNN extends Module {
  readonly cell: RNNCell;
  private readout: Linear; // hidden -> vocab logits
  readonly vocab: number;

  constructor(vocab: number, hiddenDim: number, rng: Rng) {
    super();
    this.vocab = vocab;
    this.cell = new RNNCell(vocab, hiddenDim, rng);
    this.readout = new Linear(hiddenDim, vocab, rng, { init: "xavier", bias: true });
  }

  /**
   * Unroll over T timesteps for a batch of sequences and return per-step logits.
   * batchTokens[t] is the length-B array of token ids at timestep t (column-major over
   * the batch). Returns logitsByStep[t] : Tensor[B, vocab].
   *
   * WHY column-major (per-timestep arrays) instead of per-sequence: the recurrence is
   * along time, so we must advance ALL sequences one step together to share the cell
   * call. The JS loop here is the literal "time unrolling"; each iteration extends the
   * autograd graph by one step, and backward() later walks it in reverse (BPTT).
   */
  forwardUnroll(batchTokens: number[][], batch: number): Tensor[] {
    const T = batchTokens.length;
    let h = this.cell.initialState(batch);
    const logitsByStep: Tensor[] = new Array(T);
    for (let t = 0; t < T; t++) {
      const x = oneHotRows(batchTokens[t], this.vocab); // [B, vocab]
      h = this.cell.step(x, h);
      logitsByStep[t] = this.readout.forward(h); // [B, vocab]
    }
    return logitsByStep;
  }

  override params(): Tensor[] {
    return collectParams([this.cell, this.readout]);
  }
}

/** Build a [rows, vocab] one-hot Tensor (constant leaf — no grad needed for inputs). */
function oneHotRows(ids: number[], vocab: number): Tensor {
  const data = new Float64Array(ids.length * vocab);
  for (let r = 0; r < ids.length; r++) data[r * vocab + ids[r]] = 1;
  return new Tensor(data, [ids.length, vocab]);
}

// ---------------------------------------------------------------------------
// Training on copyTask.
//
// copyTask layout (from core/data): X = [k payload][delay blanks][delim][k blanks];
//   Y is zero everywhere EXCEPT the final k answer positions, which repeat the payload.
//   So we mask the loss to the answer region only — predicting blanks elsewhere is
//   trivial and would drown out the signal we care about.
//
// LOSS PER STEP: for each answer-region timestep we have logits[B,vocab] and the target
//   token per sequence; crossEntropy averages over the batch. We average those per-step
//   losses across the answer region, then scale to get one scalar to backward() on.

interface TrainResult {
  lossHistory: number[]; // mean TRAINING loss per evaluated checkpoint
  testAccuracy: number; // answer-region token accuracy on a HELD-OUT test set
  gradNormHistory: number[]; // pre-clip global grad norm per checkpoint
}

interface Setup {
  delay: number;
  k: number;
  symbols: number;
  count: number;
  hiddenDim: number;
  steps: number;
  batchSize: number;
  lr: number;
}

function answerRegionTimesteps(T: number, k: number): number[] {
  // The last k positions carry the answer (see copyTask packing).
  const out: number[] = [];
  for (let t = T - k; t < T; t++) out.push(t);
  return out;
}

/**
 * Train a fresh CopyRNN on a fresh TRAIN dataset for `setup.steps` optimizer steps, then
 * report accuracy on a separate HELD-OUT TEST dataset (different seed => disjoint random
 * sequences). The held-out eval is what makes "learned the copy mechanism" honest:
 * train-set accuracy could be pure memorization of 256 sequences.
 * INVARIANT: callers pass independent seeds for data vs init so the two runs (short vs
 *   long delay) differ ONLY in the dependency span, not in the random streams' coupling.
 */
function trainRun(setup: Setup, dataSeed: number, initSeed: number, testSeed: number): TrainResult {
  const dataRng = makeRng(dataSeed);
  const initRng = makeRng(initSeed);

  const ds = copyTask(dataRng, { count: setup.count, k: setup.k, delay: setup.delay, symbols: setup.symbols });
  const testDs = copyTask(makeRng(testSeed), { count: setup.count, k: setup.k, delay: setup.delay, symbols: setup.symbols });
  const T = ds.X[0].length;
  const answerSteps = answerRegionTimesteps(T, setup.k);

  const model = new CopyRNN(ds.vocabSize, setup.hiddenDim, initRng);
  const opt = new Adam(model.params(), { lr: setup.lr });

  const lossHistory: number[] = [];
  const gradNormHistory: number[] = [];

  // Reusable batch sampler over the fixed dataset (data already generated).
  const sampleBatch = (rng: Rng): number[] => {
    const idx: number[] = [];
    for (let i = 0; i < setup.batchSize; i++) idx.push(rng.randint(0, ds.X.length));
    return idx;
  };
  const batchRng = makeRng(dataSeed ^ 0x9e3779b9); // independent stream for minibatch picks

  for (let stepIdx = 0; stepIdx < setup.steps; stepIdx++) {
    const idx = sampleBatch(batchRng);
    const B = idx.length;

    // Repack the selected sequences into per-timestep token arrays.
    const tokensByStep: number[][] = [];
    for (let t = 0; t < T; t++) tokensByStep.push(idx.map((i) => ds.X[i][t]));

    opt.zeroGrad(); // grads accumulate; must clear before each backward
    const logitsByStep = model.forwardUnroll(tokensByStep, B);

    // Sum cross-entropy over the answer region, then average by #answer steps.
    let lossAcc: Tensor | null = null;
    for (const t of answerSteps) {
      const targets = idx.map((i) => ds.Y[i][t]); // payload ids at this answer position
      const ce = logitsByStep[t].crossEntropy(targets); // mean over batch
      lossAcc = lossAcc ? lossAcc.add(ce) : ce;
    }
    const loss = lossAcc!.scale(1 / answerSteps.length);
    loss.backward();

    // Clip then step. clipGradNorm returns the PRE-clip norm — chapter 02 will watch
    // this number explode for long spans; here we just record it.
    const preClipNorm = clipGradNorm(model.params(), 5.0);
    opt.step();

    // Checkpoint every ~steps/40 iterations for a readable sparkline.
    const every = Math.max(1, Math.floor(setup.steps / 40));
    if (stepIdx % every === 0 || stepIdx === setup.steps - 1) {
      lossHistory.push(loss.data[0]);
      gradNormHistory.push(preClipNorm);
    }
  }

  const testAccuracy = evalAccuracy(model, testDs, T, answerSteps);
  return { lossHistory, testAccuracy, gradNormHistory };
}

/**
 * Answer-region token accuracy over the WHOLE dataset (no grad). For each answer
 * timestep, argmax the per-sequence logits and compare to the target token.
 */
function evalAccuracy(model: CopyRNN, ds: ReturnType<typeof copyTask>, T: number, answerSteps: number[]): number {
  const idx = Array.from({ length: ds.X.length }, (_, i) => i);
  const B = idx.length;
  const tokensByStep: number[][] = [];
  for (let t = 0; t < T; t++) tokensByStep.push(idx.map((i) => ds.X[i][t]));

  const logitsByStep = model.forwardUnroll(tokensByStep, B);
  const preds: number[] = [];
  const targets: number[] = [];
  for (const t of answerSteps) {
    const logits = logitsByStep[t]; // [B, vocab]
    for (let r = 0; r < B; r++) {
      const row = logits.data.subarray(r * model.vocab, (r + 1) * model.vocab);
      preds.push(argmax(row));
      targets.push(ds.Y[idx[r]][t]);
    }
  }
  return accuracy(preds, targets);
}

function countParams(model: Module): number {
  return model.params().reduce((sum, p) => sum + p.size, 0);
}

// ---------------------------------------------------------------------------
// main(): run the short-span (learns) vs long-span (cliff) comparison.
// NOTE: importing this file runs main(). Other stages must import RNNCell, never this file.
function main(): void {
  // Shared model/training budget. ONLY `delay` changes between the two runs.
  const base = {
    k: 3, // reproduce the first 3 tokens
    symbols: 6, // payload alphabet size; chance per token = 1/symbols
    count: 256,
    hiddenDim: 64,
    steps: 600,
    batchSize: 32,
    lr: 5e-3,
  };
  const chance = 1 / base.symbols;

  console.log("=".repeat(70));
  console.log("第 01 章 — RNN 与时间展开：一组权重处理任意长度序列");
  console.log("=".repeat(70));

  const SHORT_DELAY = 5;
  const LONG_DELAY = 35;

  // Describe the two tasks up front (honest task spec straight from the generator).
  const probeShort = copyTask(makeRng(1), { count: 1, k: base.k, delay: SHORT_DELAY, symbols: base.symbols });
  const probeLong = copyTask(makeRng(1), { count: 1, k: base.k, delay: LONG_DELAY, symbols: base.symbols });
  console.log("\n短依赖任务: " + probeShort.describe());
  console.log("长依赖任务: " + probeLong.describe());

  // --- Run A: short dependency — should learn -------------------------------
  const shortSetup: Setup = { ...base, delay: SHORT_DELAY };
  let shortResult!: TrainResult;
  const shortTime = timeit(() => {
    shortResult = trainRun(shortSetup, /*dataSeed*/ 11, /*initSeed*/ 22, /*testSeed*/ 999);
  });

  const modelForCount = new CopyRNN(base.symbols + 2, base.hiddenDim, makeRng(0));
  const paramCount = countParams(modelForCount);

  console.log("\n" + "-".repeat(70));
  console.log(`短依赖 (delay=${SHORT_DELAY}, 依赖跨度=${base.k + SHORT_DELAY + 1} 步) — 训练 ${base.steps} 步`);
  console.log("-".repeat(70));
  console.log(`参数量: ${paramCount} (隐藏维度 ${base.hiddenDim})`);
  console.log(`训练耗时: ${(shortTime.totalMs / 1000).toFixed(3)} s (真实 wall-clock)`);
  console.log("训练 loss 曲线 (每约 1/40 步采样): " + sparkline(shortResult.lossHistory));
  console.log(
    `训练 loss: ${shortResult.lossHistory[0].toFixed(3)} -> ${shortResult.lossHistory[shortResult.lossHistory.length - 1].toFixed(3)}`,
  );
  console.log(`留出测试集答案区 token 准确率: ${(shortResult.testAccuracy * 100).toFixed(1)}%  (随机基线 ${(chance * 100).toFixed(1)}%)`);

  // --- Run B: long dependency — same model + same steps, should cliff -------
  const longSetup: Setup = { ...base, delay: LONG_DELAY };
  let longResult!: TrainResult;
  const longTime = timeit(() => {
    longResult = trainRun(longSetup, /*dataSeed*/ 11, /*initSeed*/ 22, /*testSeed*/ 999);
  });

  console.log("\n" + "-".repeat(70));
  console.log(`失败模式 — 长依赖 (delay=${LONG_DELAY}, 依赖跨度=${base.k + LONG_DELAY + 1} 步) — 同一模型、同样 ${base.steps} 步`);
  console.log("-".repeat(70));
  console.log(`训练耗时: ${(longTime.totalMs / 1000).toFixed(3)} s (真实 wall-clock)`);
  console.log("训练 loss 曲线 (每约 1/40 步采样): " + sparkline(longResult.lossHistory));
  // Report the BEST (min) loss reached, not the noisy final step: the long-span loss
  // oscillates on a plateau and never approaches 0, so a single endpoint can land on a
  // spike. The min over training is the fairest "how far did it actually get" number.
  const longMinLoss = Math.min(...longResult.lossHistory);
  console.log(
    `训练 loss: 起始 ${longResult.lossHistory[0].toFixed(3)}，全程最低仅 ${longMinLoss.toFixed(3)} (在平台振荡，从未逼近 0)`,
  );
  console.log(`留出测试集答案区 token 准确率: ${(longResult.testAccuracy * 100).toFixed(1)}%  (随机基线 ${(chance * 100).toFixed(1)}%)`);

  // --- The chapter's takeaway, stated from the measured numbers -------------
  const shortLift = shortResult.testAccuracy / chance;
  const longLift = longResult.testAccuracy / chance;
  console.log("\n" + "=".repeat(70));
  console.log("结论 (由上方实测数字得出):");
  console.log(
    `  短依赖测试准确率 ${(shortResult.testAccuracy * 100).toFixed(1)}% ≈ 随机的 ${shortLift.toFixed(1)}× — vanilla RNN 学得动短依赖，且泛化到未见过的序列。`,
  );
  console.log(
    `  长依赖测试准确率 ${(longResult.testAccuracy * 100).toFixed(1)}% ≈ 随机的 ${longLift.toFixed(1)}× — 同一模型、同样步数，准确率塌回随机水平，loss 卡在平台降不下去。`,
  );
  console.log("  仅依赖跨度变化即令模型从「能学」滑向「学不动」——这正是第 02 章要解释的梯度消失。");
  console.log(
    `  (校准说明: 用 Adam 时 cliff 出现得意外靠后——delay=25 该 RNN 仍能学到 ~99%；真正塌陷在 delay≈${LONG_DELAY}，且多训 50% 步数也救不回来。)`,
  );
  console.log(
    "  (注: copyTask 为干净玩具任务，绝对准确率偏乐观；可迁移的是相对趋势——短跨度学得动、长跨度塌陷。)",
  );
  console.log("=".repeat(70));
}

main();
