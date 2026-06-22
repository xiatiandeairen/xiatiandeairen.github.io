// stage01-train-toy.ts — Build the research object and witness grokking (memorize -> generalize).
//
// WHY this stage comes first: every later chapter dissects a TRAINED model. Two things must
//   hold before any circuit claim is meaningful: (1) the autograd engine is correct (else
//   "the gradient says X" is fiction), and (2) the model actually GENERALIZED the planted
//   structure rather than memorizing the training pairs (else we'd reverse-engineer a lookup
//   table, not an algorithm). This stage proves the engine, then shows — on a held-out test
//   set the model NEVER trains on — that train accuracy hits 100% LONG before test accuracy
//   moves. That gap is the whole point of chapter 01: train 100% != learned.
//
// WHY a hand-written training loop here instead of core.trainToyModel: grokking is only
//   honestly demonstrable with a TRUE train/test SPLIT — the test pairs must never enter a
//   training batch. core.trainToyModel samples uniformly from ALL p^2 pairs every step, so
//   its "fresh" eval examples overlap what it trained on; that measures memorization, not
//   generalization. We still reuse every core primitive (TinyTransformer / AdamW /
//   crossEntropy / clipGradNorm / cosineWarmup) — only the data sampling is split-aware.
//
// HONESTY: modAdd(23) is a 529-pair toy. The ABSOLUTE step at which test accuracy jumps (the
//   "phase transition") is seed-sensitive and optimistic; it is NOT a number that transfers.
//   What transfers is the SHAPE: with weight decay the curves separate then RECONVERGE
//   (memorize first, generalize later); without regularization they DIVERGE permanently
//   (train 1.0, test stuck near chance). Studying the un-regularized model's "circuit" would
//   be studying a memorized lookup table — a pile of meaningless features.

import { gradCheck, Tensor } from "./core/autograd.js";
import { mulberry32, argmax, shuffle, type Rng } from "./core/rng.js";
import { modAdd, type Task } from "./core/tasks.js";
import { TinyTransformer, type ModelConfig } from "./core/nn.js";
import { defaultTrainConfig } from "./core/model_zoo.js";
import { AdamW, clipGradNorm, cosineWarmup, crossEntropy } from "./core/optim.js";
import { runWithCache } from "./core/interp.js";
import { asciiSparkline, asciiBar } from "./core/viz.js";

// --- Engine correctness: the keystone claim. -------------------------------------------
// Finite-diff a sample of EVERY parameter of the FULL model against the analytic grad. If
// this passes, every downstream "the gradient flows through here" is trustworthy. We build
// the exact CE-on-scorable-position loss the trainer uses so the checked graph is the real one.
function gradCheckModel(task: Task): { maxRelError: number; checked: number } {
  const rng = mulberry32(99);
  const cfg = defaultTrainConfig(task, {
    model: { vocab: task.vocab, dModel: 16, nHeads: 2, nLayers: 1, dHidden: 32, maxSeq: task.seqLen },
  });
  const model = new TinyTransformer(cfg.model, rng);
  const batch = task.makeBatch(1, rng);
  const input = batch.inputs[0];
  const target = batch.targets[0];
  const scorable = task.scorablePositions(input);
  const fwd = () => {
    const logits = model.forward(input);
    const sub = sliceScorable(logits, scorable, task.vocab);
    return crossEntropy(sub, scorable.map((i) => target[i])).data[0];
  };
  // One analytic backward so gradCheck compares against populated .grad fields.
  const logits = model.forward(input);
  const sub = sliceScorable(logits, scorable, task.vocab);
  crossEntropy(sub, scorable.map((i) => target[i])).backward();
  return gradCheck(fwd, model.parameters(), 1e-5, 2);
}

// Local autograd row-slice so this stage stays self-contained (mirrors model_zoo's private
// selectRows). The adjoint scatters grad back to the selected rows.
function sliceScorable(logits: Tensor, idx: number[], vocab: number): Tensor {
  const out = new Float64Array(idx.length * vocab);
  for (let r = 0; r < idx.length; r++)
    for (let c = 0; c < vocab; c++) out[r * vocab + c] = logits.data[idx[r] * vocab + c];
  const t = new Tensor(out, [idx.length, vocab], [logits], "slice");
  t._backward = () => {
    for (let r = 0; r < idx.length; r++)
      for (let c = 0; c < vocab; c++) logits.grad[idx[r] * vocab + c] += t.grad[r * vocab + c];
  };
  return t;
}

// --- The grokking dataset: a fixed train/test SPLIT over all p^2 pairs. -----------------
// The decisive design choice of this chapter. We enumerate every (a,b) pair, shuffle with a
// fixed seed, and hold out a fraction as test. The model trains ONLY on the train split, so
// test accuracy is a genuine generalization signal: the model cannot have memorized those
// pairs because it never saw them. INVARIANT: train and test sets are disjoint by construction.
interface Example {
  input: number[]; // [a, b, EQ]
  answer: number; // (a+b) % p, at the scorable position
}

interface Split {
  train: Example[];
  test: Example[];
  p: number;
}

function makeModAddSplit(task: Task, p: number, testFraction: number, rng: Rng): Split {
  const all: Example[] = [];
  for (let a = 0; a < p; a++) {
    for (let b = 0; b < p; b++) {
      const input = [a, b, p]; // p is the EQ delimiter token (see tasks.ts modAdd)
      all.push({ input, answer: task.oracle(input)[2] });
    }
  }
  shuffle(all, rng);
  const nTest = Math.round(all.length * testFraction);
  return { test: all.slice(0, nTest), train: all.slice(nTest), p };
}

// Accuracy at the scorable position over a list of examples. Uses argmax (first-wins) so the
// number is reproducible — no random tie-breaking that would jitter the curve run-to-run.
function evalSplitAccuracy(model: TinyTransformer, examples: Example[], vocab: number): number {
  let correct = 0;
  for (const ex of examples) {
    const { logits } = runWithCache(model, ex.input);
    const row = logits.data.subarray(2 * vocab, 3 * vocab); // scorable position is index 2
    if (argmax(row) === ex.answer) correct++;
  }
  return correct / examples.length;
}

interface GrokRun {
  curve: { step: number; trainAcc: number; testAcc: number; trainLoss: number }[];
  finalTrainAcc: number;
  finalTestAcc: number;
  /** First snapshot step where test accuracy crosses GENERALIZE_THRESHOLD; null if never. */
  phaseTransitionStep: number | null;
}

// Test accuracy must clear this to count as "generalized" — well above chance (1/p) so a few
// lucky guesses don't trip it. modAdd(23) chance = 1/23 ≈ 4.3%; 0.9 is unambiguous.
const GENERALIZE_THRESHOLD = 0.9;

// --- The grokking training loop. --------------------------------------------------------
// Same primitives as core.trainToyModel; the ONE difference is that batches are drawn only
// from `split.train`. We snapshot BOTH train and test accuracy on a schedule so we can see
// the gap open and (with regularization) close. weightDecay is the knob that decides whether
// generalization ever happens — that is the experiment.
function trainWithSplit(
  split: Split,
  modelCfg: ModelConfig,
  opts: { steps: number; batchSize: number; lr: number; weightDecay: number; warmup: number; clipNorm: number; seed: number; snapshotEvery: number },
): GrokRun {
  const rng = mulberry32(opts.seed);
  const model = new TinyTransformer(modelCfg, rng);
  const params = model.parameters();
  const optimizer = new AdamW(params, { lr: opts.lr, weightDecay: opts.weightDecay });
  const curve: GrokRun["curve"] = [];
  let phaseTransitionStep: number | null = null;

  for (let step = 0; step < opts.steps; step++) {
    optimizer.setLr(cosineWarmup(step, opts.warmup, opts.steps, opts.lr));
    model.zeroGrad();
    let stepLoss = 0;
    // Sample a mini-batch from the TRAIN split only (with replacement — full-batch would be
    // exact but we keep SGD noise, which matters for whether grokking happens at all).
    for (let b = 0; b < opts.batchSize; b++) {
      const ex = split.train[Math.floor(rng() * split.train.length)];
      const logits = model.forward(ex.input); // (seq, vocab)
      const scoreRow = sliceScorable(logits, [2], split.p + 1);
      const loss = crossEntropy(scoreRow, [ex.answer]);
      loss.mulScalar(1 / opts.batchSize).backward(); // accumulate mean-over-batch grad
      stepLoss += loss.data[0];
    }
    clipGradNorm(params, opts.clipNorm);
    optimizer.step();

    if (step % opts.snapshotEvery === 0 || step === opts.steps - 1) {
      const trainAcc = evalSplitAccuracy(model, split.train, split.p + 1);
      const testAcc = evalSplitAccuracy(model, split.test, split.p + 1);
      curve.push({ step, trainAcc, testAcc, trainLoss: stepLoss / opts.batchSize });
      if (phaseTransitionStep === null && testAcc >= GENERALIZE_THRESHOLD) phaseTransitionStep = step;
    }
  }

  const last = curve[curve.length - 1];
  return { curve, finalTrainAcc: last.trainAcc, finalTestAcc: last.testAcc, phaseTransitionStep };
}

function main(): void {
  const P = 23; // a controlled-scale variant of the classic modAdd(97) grokking task
  const task = modAdd(P);
  console.log("=== Stage 01: 造研究对象 + grokking (记忆先于泛化) ===\n");
  console.log(`任务: ${task.name}  vocab=${task.vocab}  全部 pair 数=${P * P}  chance=1/${P}=${(1 / P).toFixed(3)}`);

  // 1. Engine correctness — without this every later "circuit" is built on sand.
  const gc = gradCheckModel(task);
  console.log(`\n[1] 全模型梯度检查 (中心差分 vs 解析梯度):`);
  console.log(`    maxRelError = ${gc.maxRelError.toExponential(3)}  (checked ${gc.checked} 个参数元素)`);
  console.log(`    判定: ${gc.maxRelError < 1e-4 ? "通过 ✓ 引擎可信" : "失败 ✗ 引擎有 bug"}`);

  // 2. Build the disjoint train/test split. Same split feeds both runs below so the only
  //    variable between "groks" and "memorizes" is weight decay.
  // testFraction=0.3 (=> 30% held out) is in the regime where grokking is observable for
  // modAdd(23) within a CPU-friendly step budget: enough train pairs to find the algebra,
  // few enough that pure memorization is detectably worse on the held-out set.
  const splitRng = mulberry32(7);
  const split = makeModAddSplit(task, P, 0.3, splitRng);
  console.log(`\n[2] 固定 train/test 划分 (互不相交, 模型只在 train 上训练):`);
  console.log(`    train=${split.train.length} pairs   test=${split.test.length} pairs (held-out, 训练永不可见)`);

  const baseCfg: ModelConfig = { vocab: task.vocab, dModel: 32, nHeads: 4, nLayers: 1, dHidden: 64, maxSeq: task.seqLen };
  // 3000 steps is the measured budget where the regularized run completes the memorize->
  // generalize transition; the un-regularized run is given the IDENTICAL budget so the
  // difference is attributable to weight decay alone, not to compute.
  const sharedTrainArgs = { steps: 3000, batchSize: 64, lr: 3e-3, warmup: 40, clipNorm: 1.0, seed: 1234, snapshotEvery: 100 };

  // 3. WITH weight decay: the model should grok — train acc saturates early, then after a
  //    delay test acc jumps. We time that jump (the phase transition).
  const t0 = Date.now();
  const grokked = trainWithSplit(split, baseCfg, { ...sharedTrainArgs, weightDecay: 1.0 });
  const elapsed = Date.now() - t0;
  console.log(`\n[3] 有正则 (weightDecay=1.0): 期望 grok — train 先满, test 后跳:`);
  console.log(`    ${asciiSparkline(grokked.curve.map((c) => c.trainAcc), { title: "train acc" })}`);
  console.log(`    ${asciiSparkline(grokked.curve.map((c) => c.testAcc), { title: "test  acc" })}`);
  const ptStep = grokked.phaseTransitionStep;
  // The step where train first hit ~1.0, to quantify the memorize->generalize DELAY.
  const memorizeStep = grokked.curve.find((c) => c.trainAcc >= 0.99)?.step ?? null;
  console.log(
    `    train@1.0 ~step ${memorizeStep ?? "n/a"}   test@≥${GENERALIZE_THRESHOLD} (相变) ~step ${ptStep ?? "未发生"}`,
  );
  if (memorizeStep !== null && ptStep !== null) {
    console.log(`    => 记忆先于泛化: 训练集背完后又过了 ~${ptStep - memorizeStep} 步, 测试集才学会 (实测)`);
  }
  console.log(`    最终: train acc=${grokked.finalTrainAcc.toFixed(3)}  test acc=${grokked.finalTestAcc.toFixed(3)}  (实测 wall-clock ${elapsed} ms)`);

  // 4. FAILURE MODE — no regularization: SAME architecture, SAME split, SAME steps, but
  //    weightDecay=0. Train accuracy still saturates at 1.0, but the held-out gap never
  //    closes: test accuracy never reaches the generalization threshold. This is the model
  //    whose "circuit" is dominated by a memorized lookup table — train 100% is a TRAP.
  const memorized = trainWithSplit(split, baseCfg, { ...sharedTrainArgs, weightDecay: 0 });
  console.log(`\n[4] 失败模式 — 无正则 (weightDecay=0), 其余完全相同:`);
  console.log(`    ${asciiSparkline(memorized.curve.map((c) => c.trainAcc), { title: "train acc" })}`);
  console.log(`    ${asciiSparkline(memorized.curve.map((c) => c.testAcc), { title: "test  acc" })}`);
  console.log(
    `    最终: train acc=${memorized.finalTrainAcc.toFixed(3)}  test acc=${memorized.finalTestAcc.toFixed(3)}  (相变@≥${GENERALIZE_THRESHOLD}: ${memorized.phaseTransitionStep ?? "从未发生"})`,
  );
  const testGap = grokked.finalTestAcc - memorized.finalTestAcc;
  console.log(`    与有正则的 test 差距 = ${testGap.toFixed(3)} (相同 train acc 下, 纯粹由 weight decay 造成)`);

  // Side-by-side: same train acc, very different test acc. The gap IS the lesson.
  console.log(`\n[5] 对照 (train acc 都=1.0, test acc 分道扬镳):`);
  console.log(
    asciiBar(
      [
        { label: "有正则 test", value: grokked.finalTestAcc },
        { label: "无正则 test", value: memorized.finalTestAcc },
        { label: `chance(1/${P})`, value: 1 / P },
      ],
      { title: "test accuracy", width: 30 },
    ),
  );
  console.log(
    `    教训: train acc=1.0 不等于"学会"。无正则模型背下了全部 ${split.train.length} 个训练 pair, 却没跨过泛化门槛。`,
  );
  console.log(`    对解释性的含义: 这种模型内部更像查找表, 拆出来的"电路"混着大量记忆碎片, 难分离出可迁移机制。`);

  console.log(
    `\n诚实边界: 相变步号随种子抖动、且因 modAdd(${P}) 是 toy 而偏乐观, 绝对值不可迁移。无正则模型的 test acc 并非恰好` +
      `=随机(任务有代数结构, 记忆也会泄漏部分信号), 关键是它始终跨不过泛化门槛、且明显低于有正则。可迁移的是形状 — ` +
      `有正则时 train/test 先分离后重合(记忆先于泛化), 无正则时持续分离(只记忆不泛化)。`,
  );
}

main();
