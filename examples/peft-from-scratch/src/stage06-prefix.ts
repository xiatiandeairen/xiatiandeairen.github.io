// stage06-prefix.ts — Chapter 6: Prompt tuning (soft prompts as trainable virtual tokens).
//
// THE IDEA: instead of touching ANY weight of a pretrained model, prepend p continuous
//   "soft tokens" — learnable d_model vectors that live in the same space as real token
//   embeddings — to the input sequence, and train ONLY those vectors. The frozen base
//   reads the soft prefix as extra context and conditions its computation on it. Trainable
//   params = p × d_model, fully DECOUPLED from base size (this is why it is the cheapest
//   PEFT method per the param-bar this stage prints: no rank-r matrices, no per-layer
//   adapters, just p vectors).
//
// WHAT THIS STAGE MEASURES (all honest, computed at runtime):
//   1. Task-B accuracy with only the soft prompt trained (base + head frozen).
//   2. Trainable param count = p·d_model and its ratio vs the frozen base (lowest of the book).
//   3. A sweep over prefix length p to see accuracy(p): does more virtual tokens help?
//   4. FAILURE MODE: a soft prompt initialized from a high-variance Gaussian converges
//      slowly / stalls, because random vectors land far outside the manifold the frozen
//      base's first layer (LayerNorm + attention) expects. The stable recipe — init each
//      soft token near the MEAN of real token embeddings — sits in-distribution and trains
//      fast. We print both loss curves so the reader SEES the init sensitivity.
//
// WHY a classification head (not seq->seq generation): prompt tuning's canonical win is
//   "adapt a frozen encoder to a new label space with a sliver of params". A pooled
//   representation -> linear classifier gives a single crisp accuracy number per p, which
//   is exactly what the sweep and the init-failure contrast need to be legible.
//
// DETERMINISM: seed(1234) once at top; all init/shuffle/jitter draw from the global PRNG.
//   The base is pretrained ONCE and snapshotted; every prompt-tuning run starts from the
//   identical frozen base so differences come only from the soft prompt, not base drift.
//
// ⚠ TOY-SCALE: d_model=32, one TransformerBlock, ~hundreds of examples. Absolute accuracy
//   and step counts are optimistic; what transfers is the RELATIVE story — prompt tuning's
//   param ratio is tiny, and soft-prompt training is highly init-sensitive.

import { seed, normal, randint } from "./core/prng.js";
import { Tensor } from "./core/tensor.js";
import { Embedding, LayerNorm, MultiHeadAttention, Linear } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { sparkline, lossCurve, bar } from "./core/viz.js";

// ---------------------------------------------------------------------------
// Config — kept tiny so the whole stage runs on CPU in a couple seconds.
// ---------------------------------------------------------------------------
const D_MODEL = 32;
const HEADS = 2;
const VOCAB = 12;
const SEQ_LEN = 8;
const N_CLASSES = 3;
const N_TRAIN = 180;
const N_EVAL = 90;

// ---------------------------------------------------------------------------
// Frozen base encoder: Embedding -> [optional soft prefix] -> 1 attention block
//   (pre-LN) -> mean-pool over positions. The head is a separate Linear.
//
// WHY a hand-assembled forward instead of TransformerBlock: prompt tuning must INJECT the
//   soft prefix between the embedding lookup and the attention, i.e. at the sequence axis.
//   TransformerBlock.forward(x) takes an already-formed (seq,d) tensor, so we reuse its
//   sub-modules (LayerNorm/MHA/Linear) but control the seq-axis concat ourselves. This is
//   the realistic injection point: the prefix becomes extra "tokens" the attention attends to.
// ---------------------------------------------------------------------------
class FrozenEncoder {
  embed: Embedding;
  ln1: LayerNorm;
  attn: MultiHeadAttention;
  ln2: LayerNorm;
  ff1: Linear;
  ff2: Linear;
  head: Linear;

  constructor() {
    this.embed = new Embedding(VOCAB, D_MODEL);
    this.ln1 = new LayerNorm(D_MODEL);
    this.attn = new MultiHeadAttention(D_MODEL, HEADS);
    this.ln2 = new LayerNorm(D_MODEL);
    this.ff1 = new Linear(D_MODEL, D_MODEL * 2);
    this.ff2 = new Linear(D_MODEL * 2, D_MODEL);
    this.head = new Linear(D_MODEL, N_CLASSES);
  }

  /** Every leaf tensor of the base + head (for freezing and honest param counting). */
  allParams(): Tensor[] {
    return [
      this.embed.table,
      this.ln1.gamma, this.ln1.beta,
      this.attn.q.W, this.attn.q.b, this.attn.k.W, this.attn.k.b,
      this.attn.v.W, this.attn.v.b, this.attn.o.W, this.attn.o.b,
      this.ln2.gamma, this.ln2.beta,
      this.ff1.W, this.ff1.b, this.ff2.W, this.ff2.b,
      this.head.W, this.head.b,
    ];
  }

  trainableBaseParams(): Tensor[] {
    return this.allParams().filter((p) => p.requires_grad);
  }

  freeze(): void {
    for (const p of this.allParams()) p.requires_grad = false;
  }

  numBaseParams(): number {
    return this.allParams().reduce((acc, p) => acc + p.size, 0);
  }

  /**
   * Forward over one example. `prefix` (p,d) is concatenated on the SEQUENCE axis before
   *   the embedded tokens; pass null for the no-prompt base. Returns class logits (1,C).
   * INVARIANT: pooling is mean over ALL positions including the prefix, so soft tokens
   *   directly steer the pooled representation — the mechanism prompt tuning relies on.
   */
  forward(ids: number[], prefix: Tensor | null): Tensor {
    const tokEmb = this.embed.lookup(ids); // (seq, d)
    const x = prefix ? concatRows(prefix, tokEmb) : tokEmb; // (p+seq, d)
    // pre-LN attention block with two residuals (mirrors TransformerBlock semantics)
    const a = this.attn.forward(this.ln1.forward(x)).add(x);
    const f = this.ff2.forward(this.ff1.forward(this.ln2.forward(a)).gelu()).add(a);
    const pooled = meanRows(f); // (1, d)
    return this.head.forward(pooled); // (1, C)
  }
}

// ---------------------------------------------------------------------------
// Seq-axis concat + mean-pool with grad. Both must route grad into the soft prefix so
//   the optimizer can move it; that is the entire training signal in prompt tuning.
// ---------------------------------------------------------------------------

/** Stack two (·,d) tensors along rows -> (r1+r2, d). Grad routed back to each part. */
function concatRows(top: Tensor, bottom: Tensor): Tensor {
  const d = top.shape[1];
  const r1 = top.shape[0];
  const r2 = bottom.shape[0];
  const out = new Tensor(new Float64Array((r1 + r2) * d), [r1 + r2, d], top.requires_grad || bottom.requires_grad, [top, bottom], "concatRows");
  out.data.set(top.data, 0);
  out.data.set(bottom.data, r1 * d);
  out._backward = () => {
    if (top.requires_grad) for (let i = 0; i < r1 * d; i++) top.grad[i] += out.grad[i];
    if (bottom.requires_grad) for (let i = 0; i < r2 * d; i++) bottom.grad[i] += out.grad[r1 * d + i];
  };
  return out;
}

/** Mean over the row (sequence) axis -> (1, d). Grad spreads 1/rows to every input row. */
function meanRows(x: Tensor): Tensor {
  const [rows, d] = x.shape;
  const out = new Tensor(new Float64Array(d), [1, d], x.requires_grad, [x], "meanRows");
  for (let i = 0; i < rows; i++) for (let j = 0; j < d; j++) out.data[j] += x.data[i * d + j];
  for (let j = 0; j < d; j++) out.data[j] /= rows;
  out._backward = () => {
    if (!x.requires_grad) return;
    for (let i = 0; i < rows; i++) for (let j = 0; j < d; j++) x.grad[i * d + j] += out.grad[j] / rows;
  };
  return out;
}

// ---------------------------------------------------------------------------
// Loss: cross-entropy via softmax + (-log p[target]), built so backward starts from a
//   scalar. softmax() is row-wise on (1,C); we then index the target column.
// ---------------------------------------------------------------------------
function crossEntropy(logits: Tensor, target: number): Tensor {
  const probs = logits.softmax(); // (1, C)
  // Select target prob with a one-hot dot, then -log. mul is elementwise on (1,C).
  const oneHot = new Tensor(new Float64Array(N_CLASSES), [1, N_CLASSES], false);
  oneHot.data[target] = 1;
  const picked = probs.mul(oneHot).sum(); // scalar prob of the true class
  return negLog(picked);
}

/** -log(x) as a custom scalar node (no log op in core). Guards against log(0) blowups. */
function negLog(x: Tensor): Tensor {
  const v = Math.max(x.data[0], 1e-12); // clamp: a stalled prompt can drive prob -> ~0
  const out = new Tensor(new Float64Array([-Math.log(v)]), [1], x.requires_grad, [x], "negLog");
  out._backward = () => {
    if (x.requires_grad) x.grad[0] += (-1 / v) * out.grad[0];
  };
  return out;
}

function argmax(logits: Tensor): number {
  let best = 0;
  for (let c = 1; c < N_CLASSES; c++) if (logits.data[c] > logits.data[best]) best = c;
  return best;
}

// ---------------------------------------------------------------------------
// Toy labeling rules. Both tasks read the SAME sequences but assign DIFFERENT labels,
//   so a base pretrained on A is a useful-but-imperfect init for B — the gap a soft
//   prompt must close. (Mirrors core/data.ts pretrain->finetune philosophy, specialized
//   to classification because this stage needs a label per sequence.)
// ---------------------------------------------------------------------------
interface Labeled { ids: number[]; y: number; }

function genSeqs(n: number): number[][] {
  return Array.from({ length: n }, () => Array.from({ length: SEQ_LEN }, () => randint(0, VOCAB)));
}

/**
 * Task A: bucket the MEAN token value into C bins.
 * WHY a mean-bucket and not (sum mod C): the encoder pools token embeddings by mean, so a
 *   "bag-of-tokens" statistic like the average is something the frozen representation can
 *   actually GENERALIZE (verified: ~91% eval). A modular hash like (sum mod C) has no
 *   smooth structure — the model memorizes train and scores at chance on unseen sequences,
 *   which would make every downstream number meaningless. We need A to truly generalize so
 *   that "the frozen base is useful" is a real claim, not an artifact of memorization.
 */
function labelTaskA(ids: number[]): number {
  let s = 0;
  for (const t of ids) s += t;
  const mean = s / ids.length;
  return Math.min(N_CLASSES - 1, Math.floor(mean / (VOCAB / N_CLASSES)));
}

/**
 * Task B: task A's labels CYCLICALLY PERMUTED (+1 mod C) on the SAME mean-bucket feature.
 * WHY this exact relation: the discriminative feature B needs (the mean bucket) is ALREADY
 *   extracted-and-generalized by the frozen base; B only RELABELS it. So the A-trained head
 *   is now systematically wrong (baseline accuracy drops BELOW chance, ~2%), yet a soft
 *   prompt that shifts the pooled vector can re-route the frozen head onto the permuted
 *   labels — and because the underlying feature generalizes, the prompt's fix generalizes.
 *   This isolates the lesson: prompt tuning STEERS a capability the base already has; it
 *   cannot inject a capability the base lacks (an unrelated, position-sensitive rule would
 *   overfit train loss yet stay at chance — mean-pooling discards position).
 */
function labelTaskB(ids: number[]): number {
  return (labelTaskA(ids) + 1) % N_CLASSES;
}

function label(seqs: number[][], fn: (ids: number[]) => number): Labeled[] {
  return seqs.map((ids) => ({ ids, y: fn(ids) }));
}

function accuracy(enc: FrozenEncoder, data: Labeled[], prefix: Tensor | null): number {
  let correct = 0;
  for (const ex of data) if (argmax(enc.forward(ex.ids, prefix)) === ex.y) correct++;
  return correct / data.length;
}

// ---------------------------------------------------------------------------
// Soft prompt initializers — the heart of the failure-mode demo.
// ---------------------------------------------------------------------------

/** STABLE init: each soft token = embedding-table mean + tiny jitter. In-distribution,
 *   so the frozen LayerNorm/attention see vectors of the magnitude they were trained on. */
function initPromptFromEmbedMean(enc: FrozenEncoder, p: number): Tensor {
  const tbl = enc.embed.table;
  const mean = new Float64Array(D_MODEL);
  for (let r = 0; r < VOCAB; r++) for (let j = 0; j < D_MODEL; j++) mean[j] += tbl.data[r * D_MODEL + j];
  for (let j = 0; j < D_MODEL; j++) mean[j] /= VOCAB;
  const buf = new Float64Array(p * D_MODEL);
  for (let i = 0; i < p; i++) for (let j = 0; j < D_MODEL; j++) buf[i * D_MODEL + j] = mean[j] + normal(0, 0.02);
  return new Tensor(buf, [p, D_MODEL], true, [], "softPrompt.stable");
}

/** UNSTABLE init: large-variance Gaussian, far from the embedding manifold. Demonstrates
 *   prompt tuning's init sensitivity — these vectors push the first LayerNorm into a
 *   regime the frozen base never saw, so gradients are tiny and loss barely moves. */
function initPromptLargeRandom(p: number): Tensor {
  const buf = new Float64Array(p * D_MODEL);
  for (let i = 0; i < buf.length; i++) buf[i] = normal(0, 5.0); // std=5 vs embed std ~0.18
  return new Tensor(buf, [p, D_MODEL], true, [], "softPrompt.unstable");
}

// ---------------------------------------------------------------------------
// Training loops
// ---------------------------------------------------------------------------

/** Full-base training (used ONCE to pretrain on task A). Steps all base params. */
function trainBase(enc: FrozenEncoder, data: Labeled[], steps: number, lr: number): number[] {
  const opt = new Adam(enc.trainableBaseParams(), lr);
  const curve: number[] = [];
  for (let s = 0; s < steps; s++) {
    let lossSum = 0;
    opt.zeroGrad();
    for (const ex of data) {
      const loss = crossEntropy(enc.forward(ex.ids, null), ex.y);
      loss.backward(); // grads accumulate across the batch (full-batch GD)
      lossSum += loss.data[0];
    }
    // scale grads to mean-over-batch so lr is batch-size independent
    for (const p of enc.trainableBaseParams()) for (let i = 0; i < p.size; i++) p.grad[i] /= data.length;
    opt.step();
    curve.push(lossSum / data.length);
  }
  return curve;
}

/** Prompt tuning: base frozen, ONLY the soft prefix is in the optimizer. Returns the
 *   per-step mean loss curve so the caller can show convergence (or the lack of it). */
function trainPrompt(enc: FrozenEncoder, prompt: Tensor, data: Labeled[], steps: number, lr: number): number[] {
  const opt = new Adam([prompt], lr);
  const curve: number[] = [];
  for (let s = 0; s < steps; s++) {
    let lossSum = 0;
    opt.zeroGrad();
    for (const ex of data) {
      const loss = crossEntropy(enc.forward(ex.ids, prompt), ex.y);
      loss.backward();
      lossSum += loss.data[0];
    }
    for (let i = 0; i < prompt.size; i++) prompt.grad[i] /= data.length;
    opt.step();
    curve.push(lossSum / data.length);
  }
  return curve;
}

function gradNorm(t: Tensor): number {
  let s = 0;
  for (let i = 0; i < t.size; i++) s += t.grad[i] * t.grad[i];
  return Math.sqrt(s);
}

/** L2 norm of the mean-over-batch gradient on the prompt at step 0 (no optimizer step).
 *   Used to expose, before any training, how much learning signal the init lets through. */
function firstStepGradNorm(enc: FrozenEncoder, prompt: Tensor, data: Labeled[]): number {
  prompt.zeroGrad();
  for (const ex of data) crossEntropy(enc.forward(ex.ids, prompt), ex.y).backward();
  for (let i = 0; i < prompt.size; i++) prompt.grad[i] /= data.length;
  return gradNorm(prompt);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
function main(): void {
  seed(1234); // single re-seed; everything below is reproducible to the last decimal

  console.log("=== 第 06 章：Prompt Tuning（连续向量当软提示）===\n");

  // --- data: same sequences, two label rules ---
  const trainSeqs = genSeqs(N_TRAIN);
  const evalSeqs = genSeqs(N_EVAL);
  const trainA = label(trainSeqs, labelTaskA);
  const evalA = label(evalSeqs, labelTaskA);
  const trainB = label(trainSeqs, labelTaskB);
  const evalB = label(evalSeqs, labelTaskB);

  // --- pretrain the base on task A, then FREEZE it (the PEFT base) ---
  const enc = new FrozenEncoder();
  console.log("[1] 预训练基座于任务 A（label = mean(tokens) 分桶到 3 类），随后整体冻结");
  const baseCurve = trainBase(enc, trainA, 120, 0.02);
  const baseAccA = accuracy(enc, trainA, null);
  const baseEvalAccA = accuracy(enc, evalA, null);
  console.log(`    base loss: ${baseCurve[0].toFixed(4)} -> ${baseCurve[baseCurve.length - 1].toFixed(4)}  curve ${sparkline(baseCurve)}`);
  console.log(`    base 任务 A 准确率：训练集 ${(baseAccA * 100).toFixed(1)}%  验证集 ${(baseEvalAccA * 100).toFixed(1)}%（真泛化，非死记）`);
  enc.freeze();
  console.log(`    冻结后 base 可训练参数 = ${enc.trainableBaseParams().length} 个张量（期望 0）\n`);

  // --- baseline: frozen base, NO prompt, evaluated on task B (the gap to close) ---
  const baseAccB = accuracy(enc, evalB, null);
  console.log(`[2] 冻结 base 直接评测任务 B（label = A 的标签循环 +1，特征相同但重新映射）`);
  console.log(`    无软提示时任务 B 准确率 = ${(baseAccB * 100).toFixed(1)}%  ← 低于随机（A 的 head 系统性答错），这是 prompt tuning 要填的差距\n`);

  // --- prompt tuning with a fixed p, stable init ---
  const P = 8;
  console.log(`[3] Prompt tuning：拼 p=${P} 个可训练软 token，base 全冻结`);
  const promptStable = initPromptFromEmbedMean(enc, P);
  const stableCurve = trainPrompt(enc, promptStable, trainB, 200, 0.05);
  const accStable = accuracy(enc, evalB, promptStable);
  console.log(lossCurve(stableCurve, { label: "    稳定初始化（embedding 均值附近）loss", height: 6, width: 50 }));
  console.log(`    任务 B 准确率：无提示 ${(baseAccB * 100).toFixed(1)}% -> 软提示 ${(accStable * 100).toFixed(1)}%`);

  // --- honest param accounting: trainable = p * d_model, decoupled from base size ---
  const promptParams = P * D_MODEL;
  const baseParams = enc.numBaseParams();
  const ratioPct = (promptParams / baseParams) * 100;
  console.log(`\n[4] 可训练参数（诚实计数，非论文引用）`);
  console.log(`    软提示参数 = p×d_model = ${P}×${D_MODEL} = ${promptParams}`);
  console.log(`    冻结基座参数 = ${baseParams}`);
  console.log(`    可训练占比 = ${ratioPct.toFixed(3)}%  ← 与 base 大小解耦，是各章里最低的`);
  console.log(
    bar(
      [
        { label: "frozen base", value: baseParams, note: "(冻结，不计入可训练)" },
        { label: `soft prompt p=${P}`, value: promptParams, note: `${ratioPct.toFixed(2)}% of base` },
      ],
      { width: 44 },
    ),
  );

  // --- sweep prefix length p: accuracy(p) ---
  console.log(`\n[5] 扫 prefix 长度 p（每个 p 用稳定初始化重训软提示）`);
  const ps = [1, 2, 4, 8, 16];
  const sweepAcc: number[] = [];
  for (const p of ps) {
    const prompt = initPromptFromEmbedMean(enc, p);
    trainPrompt(enc, prompt, trainB, 200, 0.05);
    const acc = accuracy(enc, evalB, prompt);
    sweepAcc.push(acc);
    const trainable = p * D_MODEL;
    console.log(`    p=${String(p).padStart(2)}  可训练=${String(trainable).padStart(3)}  任务B准确率=${(acc * 100).toFixed(1)}%`);
  }
  console.log(`    accuracy(p) 形状 ${sparkline(sweepAcc)}  (左=p小 右=p大)`);

  // --- FAILURE MODE: large-variance random init costs a huge convergence tax ---
  // WHY it stalls: a high-variance prefix (std=5, ~28× the embedding manifold's std≈0.18)
  //   pushes the FROZEN LayerNorm into a regime it was never trained on, which flattens the
  //   gradient before it reaches the soft prompt — measured below as a ~50× weaker first-step
  //   gradient. The optimizer then needs far more steps to claw back.
  // HONEST FRAMING (toy-scale): we compare both inits under a MATCHED, SHORT budget
  //   (FAIL_STEPS) so the gap is apples-to-apples. Under that budget the bad init sits far
  //   behind. We do NOT claim it is permanently broken: this prompt is only p·d=256-dim and
  //   Adam can eventually escape given enough steps. The transferable lesson is the
  //   CONVERGENCE TAX (weaker gradient, much slower, worse at any fixed budget), not a
  //   permanent stall — at real scale that tax is what makes init choice matter.
  const FAIL_STEPS = 60;
  console.log(`\n[6] 失败模式：大方差随机初始化（std=5，约为 embedding 流形 std≈0.18 的 28 倍）`);
  const promptBad = initPromptLargeRandom(P);
  const badGradNorm = firstStepGradNorm(enc, promptBad, trainB);
  const stableMatched = initPromptFromEmbedMean(enc, P);
  const stableGradNorm = firstStepGradNorm(enc, stableMatched, trainB);
  console.log(`    首步梯度范数：大随机=${badGradNorm.toExponential(2)}  稳定=${stableGradNorm.toExponential(2)}  (相差约 ${Math.round(stableGradNorm / badGradNorm)}×，越小越学不动)`);
  // matched-budget comparison: train BOTH from scratch for FAIL_STEPS steps
  const badCurve = trainPrompt(enc, promptBad, trainB, FAIL_STEPS, 0.05);
  const stableMatchedCurve = trainPrompt(enc, stableMatched, trainB, FAIL_STEPS, 0.05);
  const accBad = accuracy(enc, evalB, promptBad);
  const accStableMatched = accuracy(enc, evalB, stableMatched);
  console.log(lossCurve(badCurve, { label: `    大方差随机初始化 loss（同样 ${FAIL_STEPS} 步，降得慢、停在高位）`, height: 6, width: 50 }));
  console.log(`    同 ${FAIL_STEPS} 步预算下任务 B 准确率：大随机 ${(accBad * 100).toFixed(1)}%  vs 稳定 ${(accStableMatched * 100).toFixed(1)}%`);
  console.log(`    同 ${FAIL_STEPS} 步 loss 末值：大随机 ${badCurve[badCurve.length - 1].toFixed(4)}  vs 稳定 ${stableMatchedCurve[stableMatchedCurve.length - 1].toFixed(4)}`);

  console.log(`\n[结论]`);
  console.log(`  • prompt tuning 只训 ${promptParams} 个参数（${ratioPct.toFixed(3)}% of base）即把任务 B 从 ${(baseAccB * 100).toFixed(1)}% 提到 ${(accStable * 100).toFixed(1)}%（200 步）`);
  console.log(`  • 软提示对初始化高度敏感：均值附近初始化首步梯度强、收敛快；大方差随机首步梯度被冻结 LayerNorm 压扁，同预算下远落后`);
  console.log(`  • 诚实补充：本 toy 上大随机并非永久卡死（256 维软提示，Adam 给足步数能爬回），可迁移的是“收敛代价”而非“一定不收敛”`);
  console.log(`\n⚠ toy-scale：d_model=${D_MODEL}、单层 block、~百级样本，绝对准确率/步数偏乐观；可迁移的是“参数占比极低 + 软提示初始化敏感（收敛代价）”这两个相对趋势。`);
}

main();
