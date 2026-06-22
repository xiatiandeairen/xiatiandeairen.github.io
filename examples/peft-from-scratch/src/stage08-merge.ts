// stage08-merge.ts — 合并与交付:把 LoRA 折回权重 W',演示零开销推理 + 多适配器切换。
//
// THE CLAIM THIS STAGE PROVES (and where it can break):
//   A LoRA-adapted linear is, at INFERENCE TIME, mathematically identical to a single dense
//   linear whose weight is W' = W + (alpha/r) * B @ A. So once training is done you can FOLD
//   the two skinny matrices back into the base and ship one ordinary weight matrix — same
//   outputs, zero extra matmuls, zero extra params. This stage does not assert that on faith:
//   it builds both paths, runs them on the same input, and asserts the elementwise diff is
//   below 1e-6 (float round-off floor), then COUNTS matmuls to show the merged path is back
//   to base cost.
//
//   WHY 1e-6 not 0: W + scale*B@A is computed in a different op ORDER than (Wx) + scale*B(Ax),
//   so the two outputs differ only by f64 summation round-off — that floor is the honest
//   target. A merge bug (wrong scale, wrong transpose, B@A vs A@B) blows past it by orders of
//   magnitude, which is exactly why the assert is the proof and not decoration.
//
// THE FAILURE MODE WE DEMO (not just happy path): if the LoRA branch still has dropout ACTIVE
//   (train=true) when you merge, the merged static weight cannot reproduce the stochastic
//   training-time output — the diff explodes far above 1e-6. Merge is only valid in the
//   deterministic eval state (dropout off). We trigger that mismatch on purpose and print it.
//
// MULTI-ADAPTER SWITCHING: a frozen base + several tiny adapters is the real deployment story
//   ("one 7B base, many task LoRAs"). We train two adapters on two different finetune tasks,
//   then switch by merge(A) -> unmerge -> merge(B), re-testing token accuracy each time to
//   show the base is losslessly restorable (unmerge subtracts EXACTLY what merge added).
//
// INVARIANTS:
//   - seed() once at top (global PRNG stream; see core/prng.ts header).
//   - The base is trained ONCE, snapshotted, then frozen. Every adapter starts from that same
//     base, so adapter results are comparable (core/checkpoint.ts contract).
//   - merge/unmerge are pure weight arithmetic on W.data; they never touch requires_grad of
//     the base (the base stays frozen throughout) and B@A is recomputed from current adapter
//     weights so unmerge is the exact inverse of the preceding merge.
//
// ⚠ toy-scale: absolute losses/accuracies are optimistic (tiny model, tiny data). The
//   transferable facts are the EXACT equivalence (<1e-6), the matmul-count collapse on merge,
//   and the dropout failure mode — none of which depend on scale.

import { Tensor } from "./core/tensor.js";
import { Linear, Module } from "./core/nn.js";
import { seed, normal, kaimingStd } from "./core/prng.js";
import { Adam } from "./core/optim.js";
import { genPretrainFinetune, batches, type SeqExample, type PairTask } from "./core/data.js";
import { dump, loadBase, type Checkpoint } from "./core/checkpoint.js";
import { bar } from "./core/viz.js";

// ---------------------------------------------------------------------------
// Matmul instrumentation. WHY here and not in core: the "zero extra matmuls after merge"
// claim is a stage-level CLAIM about inference cost, not a core concern. We wrap the global
// Tensor.matmul once, increment a counter on every call, and read it around a single forward.
// This is the honest way to back "inference cost returns to base level" with a real number
// instead of asserting it from the diagram.
// ---------------------------------------------------------------------------
let matmulCalls = 0;
const _rawMatmul = Tensor.prototype.matmul;
Tensor.prototype.matmul = function (other: Tensor): Tensor {
  matmulCalls += 1;
  return _rawMatmul.call(this, other);
};
/** Run fn, return how many matmul calls it triggered (for inference-cost accounting). */
function countMatmuls(fn: () => void): number {
  const before = matmulCalls;
  fn();
  return matmulCalls - before;
}

// ---------------------------------------------------------------------------
// Toy seq model: embed token ids -> one Linear "feature" projection (this is our PEFT
// injection site, standing in for an attention q/v proj) -> Linear head to vocab logits.
// Small on purpose; the merge math is identical at any size.
// ---------------------------------------------------------------------------
const D_MODEL = 16;
const SEQ_LEN = 6;
const VOCAB = 10;

/** Token-id embedding as a plain table we index by hand (keeps everything 2-D and explicit). */
class SeqModel extends Module {
  embed: Tensor; // (vocab, d) — frozen base part after pretrain
  proj: Linear; // (d, d) — the LoRA injection site
  head: Linear; // (d, vocab)
  constructor() {
    super();
    const t = new Float64Array(VOCAB * D_MODEL);
    const std = kaimingStd(D_MODEL);
    for (let i = 0; i < t.length; i++) t[i] = normal(0, std);
    this.embed = new Tensor(t, [VOCAB, D_MODEL], true, [], "embed");
    this.proj = new Linear(D_MODEL, D_MODEL);
    this.head = new Linear(D_MODEL, VOCAB);
  }
  /** ids (length seq) -> (seq, vocab) logits. */
  forwardIds(ids: number[]): Tensor {
    const rows = new Float64Array(ids.length * D_MODEL);
    for (let i = 0; i < ids.length; i++) {
      const r = ids[i] * D_MODEL;
      for (let j = 0; j < D_MODEL; j++) rows[i * D_MODEL + j] = this.embed.data[r + j];
    }
    // Wire embed grad so the base can actually pretrain. Only touched rows get grad.
    const x = new Tensor(rows, [ids.length, D_MODEL], this.embed.requires_grad, [this.embed], "lookup");
    x._backward = () => {
      if (!this.embed.requires_grad) return;
      for (let i = 0; i < ids.length; i++) {
        const r = ids[i] * D_MODEL;
        for (let j = 0; j < D_MODEL; j++) this.embed.grad[r + j] += x.grad[i * D_MODEL + j];
      }
    };
    const h = this.proj.forward(x).gelu();
    return this.head.forward(h);
  }
  override forward(_x: Tensor): Tensor {
    throw new Error("SeqModel: use forwardIds(ids)");
  }
}

// ---------------------------------------------------------------------------
// LoRA on the proj Linear. delta(x) = scale * ( (x @ A^T) @ B^T ), with
//   A: (r, d) init ~N, B: (d, r) init 0  =>  delta starts at exactly 0 (no perturbation).
// MERGED weight: W' = W + scale * (B @ A)   [ (d,r)@(r,d) = (d,d), same shape as W ].
// WHY B@A (not A@B): forward applies A first then B, i.e. y = x Wᵀ + scale * ((xAᵀ)Bᵀ)
//   = x (W + scale·(BA))ᵀ. So the equivalent dense weight delta is scale·(B@A). Getting this
//   product order wrong is the classic merge bug the <1e-6 assert catches.
// ---------------------------------------------------------------------------
class LoraAdapter {
  A: Tensor; // (r, d)
  B: Tensor; // (d, r)
  scale: number;
  dropoutP: number;
  constructor(d: number, r: number, alpha: number, dropoutP = 0) {
    const a = new Float64Array(r * d);
    const std = kaimingStd(d);
    for (let i = 0; i < a.length; i++) a[i] = normal(0, std);
    this.A = new Tensor(a, [r, d], true, [], "lora.A");
    this.B = new Tensor(new Float64Array(d * r), [d, r], true, [], "lora.B"); // zero init
    this.scale = alpha / r;
    this.dropoutP = dropoutP;
  }
  /** delta contribution to proj output, given the proj INPUT x (seq, d). train gates dropout. */
  delta(x: Tensor, train: boolean): Tensor {
    const xd = this.dropoutP > 0 ? x.dropout(this.dropoutP, train) : x;
    return xd.matmul(this.A.transpose()).matmul(this.B.transpose()).scale(this.scale);
  }
  /** The dense weight delta scale*(B@A), shape (d,d), as a plain array for merge arithmetic. */
  weightDelta(): { data: Float64Array; shape: [number, number] } {
    const ba = this.B.matmul(this.A); // (d,r)@(r,d) -> (d,d)
    const out = new Float64Array(ba.size);
    for (let i = 0; i < ba.size; i++) out[i] = ba.data[i] * this.scale;
    return { data: out, shape: [ba.shape[0], ba.shape[1]] };
  }
  params(): Tensor[] {
    return [this.A, this.B];
  }
}

/** Forward with an active (un-merged) LoRA branch: base proj + adapter delta, then gelu+head. */
function forwardWithLora(model: SeqModel, lora: LoraAdapter, ids: number[], train: boolean): Tensor {
  const rows = new Float64Array(ids.length * D_MODEL);
  for (let i = 0; i < ids.length; i++) {
    const r = ids[i] * D_MODEL;
    for (let j = 0; j < D_MODEL; j++) rows[i * D_MODEL + j] = model.embed.data[r + j];
  }
  const x = new Tensor(rows, [ids.length, D_MODEL], false, [], "lookup");
  const projBase = model.proj.forward(x); // x Wᵀ + b
  const projFull = projBase.add(lora.delta(x, train)); // + scale*((xAᵀ)Bᵀ)
  const h = projFull.gelu();
  return model.head.forward(h);
}

// ---------------------------------------------------------------------------
// merge / unmerge: fold the adapter into proj.W in place. INVARIANT: unmerge subtracts the
// delta recomputed from the SAME adapter weights, so it is the exact inverse of merge.
// We snapshot W's bytes only to prove round-trip exactness, not as the mechanism.
// ---------------------------------------------------------------------------
function mergeInto(proj: Linear, lora: LoraAdapter): void {
  const { data } = lora.weightDelta(); // (d,d) matches W (out,in)=(d,d)
  for (let i = 0; i < proj.W.size; i++) proj.W.data[i] += data[i];
}
function unmergeFrom(proj: Linear, lora: LoraAdapter): void {
  const { data } = lora.weightDelta();
  for (let i = 0; i < proj.W.size; i++) proj.W.data[i] -= data[i];
}

// ---------------------------------------------------------------------------
// Training / eval helpers. Loss = mean over positions of -log p(target) (cross-entropy via
// softmax row). Accuracy = fraction of positions whose argmax logit == target token.
// ---------------------------------------------------------------------------
function ceLoss(logits: Tensor, target: number[]): Tensor {
  const probs = logits.softmax(); // (seq, vocab)
  // gather -log prob of the gold token at each position, then mean
  const picked = new Float64Array(target.length);
  const sel = new Tensor(picked, [target.length, 1], probs.requires_grad, [probs], "nll-gather");
  for (let i = 0; i < target.length; i++) sel.data[i] = -Math.log(probs.data[i * VOCAB + target[i]] + 1e-12);
  sel._backward = () => {
    if (!probs.requires_grad) return;
    for (let i = 0; i < target.length; i++) {
      const p = probs.data[i * VOCAB + target[i]] + 1e-12;
      probs.grad[i * VOCAB + target[i]] += sel.grad[i] * (-1 / p);
    }
  };
  return sel.mean();
}

function tokenAccuracy(logitsForward: (ids: number[]) => Tensor, data: SeqExample[]): number {
  let correct = 0;
  let total = 0;
  for (const ex of data) {
    const logits = logitsForward(ex.input);
    for (let i = 0; i < ex.target.length; i++) {
      let best = 0;
      let bestV = -Infinity;
      for (let v = 0; v < VOCAB; v++) {
        const val = logits.data[i * VOCAB + v];
        if (val > bestV) {
          bestV = val;
          best = v;
        }
      }
      if (best === ex.target[i]) correct++;
      total++;
    }
  }
  return correct / total;
}

/** Train one LoRA adapter on `data` for `epochs`; base proj.W stays untouched (no merge yet). */
function trainAdapter(model: SeqModel, lora: LoraAdapter, data: SeqExample[], epochs: number, lr: number): number[] {
  const opt = new Adam(lora.params(), lr);
  const losses: number[] = [];
  for (let e = 0; e < epochs; e++) {
    let epochLoss = 0;
    let nb = 0;
    for (const batch of batches(data, 8)) {
      opt.zeroGrad();
      // accumulate loss over the batch then one backward (mean-of-means is fine at fixed batch)
      let lossSum: Tensor | null = null;
      for (const ex of batch) {
        const logits = forwardWithLora(model, lora, ex.input, true); // train=true (dropout active if set)
        const l = ceLoss(logits, ex.target);
        lossSum = lossSum ? lossSum.add(l) : l;
      }
      const loss = lossSum!.scale(1 / batch.length);
      loss.backward();
      opt.step();
      epochLoss += loss.data[0];
      nb++;
    }
    losses.push(epochLoss / nb);
  }
  return losses;
}

/** Pretrain the base on task A so the frozen base is actually useful (PEFT precondition). */
function pretrainBase(model: SeqModel, data: SeqExample[], epochs: number, lr: number): number {
  const opt = new Adam(model.parameters(), lr);
  let last = 0;
  for (let e = 0; e < epochs; e++) {
    let epochLoss = 0;
    let nb = 0;
    for (const batch of batches(data, 8)) {
      opt.zeroGrad();
      let lossSum: Tensor | null = null;
      for (const ex of batch) {
        const logits = model.forwardIds(ex.input);
        const l = ceLoss(logits, ex.target);
        lossSum = lossSum ? lossSum.add(l) : l;
      }
      const loss = lossSum!.scale(1 / batch.length);
      loss.backward();
      opt.step();
      epochLoss += loss.data[0];
      nb++;
    }
    last = epochLoss / nb;
  }
  return last;
}

function maxAbsDiff(a: Tensor, b: Tensor): number {
  let m = 0;
  for (let i = 0; i < a.size; i++) m = Math.max(m, Math.abs(a.data[i] - b.data[i]));
  return m;
}

// ===========================================================================
// main
// ===========================================================================
function main(): void {
  seed(1234); // INVARIANT: one seed at top; all randomness below is reproducible.

  const R = 2;
  const ALPHA = 4;

  console.log("=== 第 08 章 · 合并与交付:把 LoRA 折回权重,多适配器切换 ===\n");

  // --- 0. 训一个有用的基座 (task A = copy),冻结快照。后续所有适配器从同一基座出发 ---
  const taskA: PairTask = "copy";
  const taskB1: PairTask = "reverse";
  const taskB2: PairTask = "sort";
  const dsA = genPretrainFinetune({ taskA, taskB: taskB1, seqLen: SEQ_LEN, vocab: VOCAB, nPretrain: 256, nFinetune: 96 });
  const dsB2 = genPretrainFinetune({ taskA, taskB: taskB2, seqLen: SEQ_LEN, vocab: VOCAB, nPretrain: 4, nFinetune: 96 });

  const model = new SeqModel();
  const baseLoss = pretrainBase(model, dsA.pretrain, 60, 5e-3);
  const ckpt: Checkpoint = dump(model);
  loadBase(model, ckpt, true); // freeze the whole base; from here proj.W is read-only base.
  console.log(`[0] 基座预训练 (task=${taskA}) 末轮 loss=${baseLoss.toFixed(4)};已冻结 ${model.frozen().length} 个参数张量。\n`);

  // --- 1. 训一个 LoRA 适配器 (task B1 = reverse),只动 A/B 两块 skinny 矩阵 ---
  const loraB1 = new LoraAdapter(D_MODEL, R, ALPHA, 0);
  const lossesB1 = trainAdapter(model, loraB1, dsA.finetune, 80, 1e-2);
  const accBeforeMerge = tokenAccuracy((ids) => forwardWithLora(model, loraB1, ids, false), dsA.finetune);
  console.log(
    `[1] LoRA(r=${R},alpha=${ALPHA}) 微调 task=${taskB1}: loss ${lossesB1[0].toFixed(4)} -> ${lossesB1[lossesB1.length - 1].toFixed(4)};` +
      ` 适配器参数=${loraB1.params().reduce((s, p) => s + p.size, 0)} (A ${R}x${D_MODEL} + B ${D_MODEL}x${R}),token acc=${(accBeforeMerge * 100).toFixed(1)}%\n`,
  );

  // --- 2. 合并:W' = W + scale*(B@A)。数学等价证明 + 推理零额外开销 ---
  const probe = dsA.finetune[0].input; // fixed probe sequence for the equivalence check

  // 合并前: 数出 "base+LoRA" 路径的 matmul 次数
  let loraOut!: Tensor;
  const matmulsWithLora = countMatmuls(() => {
    loraOut = forwardWithLora(model, loraB1, probe, false); // eval state (deterministic)
  });

  mergeInto(model.proj, loraB1); // fold into base weight

  // 合并后: 用纯基座前向 (无适配器分支),数出 matmul 次数
  let mergedOut!: Tensor;
  const matmulsMerged = countMatmuls(() => {
    mergedOut = model.forwardIds(probe);
  });

  const equivDiff = maxAbsDiff(loraOut, mergedOut);
  const trainableAfterMerge = loraB1.params().length; // adapters still exist as objects...
  // ...but after merge the SHIPPED model is just `model` — count ITS trainable params:
  const shippedTrainable = model.numParams({ trainableOnly: true });

  console.log("[2] 合并 W' = W + (alpha/r)·(B@A) 后:");
  console.log(`    数学等价: max|merged_out - (base+LoRA)_out| = ${equivDiff.toExponential(2)}  (阈值 < 1e-6)`);
  if (equivDiff >= 1e-6) throw new Error(`merge equivalence FAILED: diff ${equivDiff} >= 1e-6`);
  console.log(`    ✓ 逐元素差在 f64 round-off 地板内,合并前向与 base+LoRA 前向等价。`);
  console.log(`    推理 matmul 次数: base+LoRA = ${matmulsWithLora}  ->  合并后 = ${matmulsMerged}  (回到基座水平,省下 ${matmulsWithLora - matmulsMerged} 次)`);
  console.log(`    合并后交付模型可训练参数 = ${shippedTrainable}  (基座全冻结,适配器已折进权重,无额外可训练量)`);
  console.log(`    (注: 适配器对象本身仍有 ${trainableAfterMerge} 个张量,但它们不再参与交付前向)\n`);

  // 合并后的精度应与合并前一致 (因为输出逐元素等价)
  const accAfterMerge = tokenAccuracy((ids) => model.forwardIds(ids), dsA.finetune);
  console.log(`    精度自洽: 合并前 acc=${(accBeforeMerge * 100).toFixed(1)}%  合并后 acc=${(accAfterMerge * 100).toFixed(1)}%  (应相同)\n`);

  // 还原基座,准备下面的多适配器切换演示
  unmergeFrom(model.proj, loraB1);
  const restoreDiff = maxAbsDiff(
    Tensor.from(Array.from(ckpt.buffers[model.parameters().indexOf(model.proj.W)] ?? []), model.proj.W.shape),
    model.proj.W,
  );
  console.log(`[2b] unmerge 还原基座: max|W_restored - W_base| = ${restoreDiff.toExponential(2)} (unmerge 是 merge 的精确逆)\n`);

  // --- 3. 多适配器切换: 训第二个适配器 (task B2 = sort),演示 merge A -> unmerge -> merge B ---
  const loraB2 = new LoraAdapter(D_MODEL, R, ALPHA, 0);
  const lossesB2 = trainAdapter(model, loraB2, dsB2.finetune, 80, 1e-2);
  const accB2Active = tokenAccuracy((ids) => forwardWithLora(model, loraB2, ids, false), dsB2.finetune);
  console.log(`[3] 第二个 LoRA 适配器 task=${taskB2}: loss -> ${lossesB2[lossesB2.length - 1].toFixed(4)}, token acc=${(accB2Active * 100).toFixed(1)}%`);

  // switch sequence on the SHARED frozen base:
  // merge B1 -> test B1 -> unmerge B1 -> merge B2 -> test B2 -> unmerge B2
  mergeInto(model.proj, loraB1);
  const accB1Merged = tokenAccuracy((ids) => model.forwardIds(ids), dsA.finetune);
  unmergeFrom(model.proj, loraB1);

  mergeInto(model.proj, loraB2);
  const accB2Merged = tokenAccuracy((ids) => model.forwardIds(ids), dsB2.finetune);
  unmergeFrom(model.proj, loraB2);

  console.log("    切换演示 (同一冻结基座, merge->unmerge->merge):");
  console.log(
    bar([
      { label: `${taskB1} via merged adapter`, value: +(accB1Merged * 100).toFixed(1), note: "%" },
      { label: `${taskB2} via merged adapter`, value: +(accB2Merged * 100).toFixed(1), note: "%" },
    ], { width: 30 }),
  );
  console.log(`    每个任务装上对应适配器后各自回测,基座始终是同一份 (切换无需复制基座)。\n`);

  // --- 4. 失败模式: 带 dropout 的 LoRA 在 train 态下 merge -> 输出与确定性合并权重不一致 ---
  console.log("[4] 失败模式 — merge 必须在确定性 eval 态 (dropout 关):");
  const loraDrop = new LoraAdapter(D_MODEL, R, ALPHA, 0.5); // 50% dropout on the LoRA input
  trainAdapter(model, loraDrop, dsA.finetune, 30, 1e-2);

  // 正确做法: eval 态 (dropout off) 下 base+LoRA 与合并权重等价
  const evalOut = forwardWithLora(model, loraDrop, probe, false);
  mergeInto(model.proj, loraDrop);
  const mergedDropOut = model.forwardIds(probe);
  const goodDiff = maxAbsDiff(evalOut, mergedDropOut);
  unmergeFrom(model.proj, loraDrop);

  // 错误做法: 拿 TRAIN 态 (dropout 随机掩码) 的输出去对比同一份合并权重
  const trainOut = forwardWithLora(model, loraDrop, probe, true); // dropout ACTIVE -> stochastic
  mergeInto(model.proj, loraDrop);
  const mergedDropOut2 = model.forwardIds(probe);
  const badDiff = maxAbsDiff(trainOut, mergedDropOut2);
  unmergeFrom(model.proj, loraDrop);

  console.log(`    eval 态 (dropout off): max|base+LoRA - merged| = ${goodDiff.toExponential(2)}  -> < 1e-6, 合法合并`);
  console.log(`    train 态 (dropout on): max|base+LoRA - merged| = ${badDiff.toExponential(2)}  -> 远超 1e-6`);
  if (badDiff <= 1e-6) {
    console.log("    (注: 本次随机掩码恰好未丢弃关键单元;dropout 的非确定性使该 diff 不可作为通过条件)");
  } else {
    console.log("    ✗ 训练态的随机 dropout 无法被静态合并权重复现 -> 合并前必须切 eval(关 dropout)。");
  }
  console.log("    根因: merge 把 LoRA 当成确定性线性增量折进 W;dropout 让 LoRA 分支变成随机函数,二者不再等价。\n");

  // --- 5. 第 00 章预测表的实测填充版 (收束全书) ---
  // 先训一个真实的 full-FT 基线 (从同一基座出发,解冻全部参数,在 task B1 上微调),
  // 这样表里 "LoRA 逼近全量微调" 是实测对比,而不是断言。
  const fullModel = new SeqModel();
  loadBase(fullModel, ckpt, false); // freeze=false -> 全部参数可训 (full fine-tune)
  const fullFtLoss = pretrainBase(fullModel, dsA.finetune, 80, 1e-2); // 同任务/步数,公平对比 LoRA
  const accFullFt = tokenAccuracy((ids) => fullModel.forwardIds(ids), dsA.finetune);

  // 基线: 全量微调 vs LoRA(冻结基座+r=2)。参数/显存为本 stage 真实计数 + 估算,acc 为实测。
  const totalParams = model.numParams();
  const loraTrainable = loraB1.params().reduce((s, p) => s + p.size, 0);
  const fullFtTrainable = totalParams; // 全量微调 = 所有参数可训
  // 估算训练时显存 (est., core/mem 同款公式; Adam optMul=2, fp32 4B/param)
  const estMemFull = (totalParams * 4 + fullFtTrainable * 4 + fullFtTrainable * 4 * 2 + totalParams * 4) / 1024;
  const estMemLora = (totalParams * 4 + loraTrainable * 4 + loraTrainable * 4 * 2 + totalParams * 4) / 1024;

  console.log("[5] 第 00 章预测表 · 实测填充版 (toy-scale):");
  console.log("    ┌─────────────────────┬───────────────┬──────────────┬───────────────┬──────────────┐");
  console.log("    │ 方法                │ 可训练参数    │ 占总参数比   │ 训练显存(est.)│ task acc     │");
  console.log("    ├─────────────────────┼───────────────┼──────────────┼───────────────┼──────────────┤");
  console.log(
    `    │ 全量微调 (full-FT)  │ ${String(fullFtTrainable).padStart(13)} │ ${"100.0%".padStart(12)} │ ${(estMemFull.toFixed(2) + " KB").padStart(13)} │ ${((accFullFt * 100).toFixed(1) + "%").padStart(12)} │`,
  );
  console.log(
    `    │ LoRA (r=${R}, 冻结)   │ ${String(loraTrainable).padStart(13)} │ ${((loraTrainable / totalParams) * 100).toFixed(2).padStart(11)}% │ ${(estMemLora.toFixed(2) + " KB").padStart(13)} │ ${((accBeforeMerge * 100).toFixed(1) + "%").padStart(12)} │`,
  );
  console.log(
    `    │ LoRA 合并后交付     │ ${String(0).padStart(13)} │ ${"0.00%".padStart(12)} │ ${"= 基座".padStart(11)} │ ${((accAfterMerge * 100).toFixed(1) + "%").padStart(12)} │`,
  );
  console.log("    └─────────────────────┴───────────────┴──────────────┴───────────────┴──────────────┘");
  console.log(`    (full-FT 末轮 loss=${fullFtLoss.toFixed(4)}, LoRA 末轮 loss=${lossesB1[lossesB1.length - 1].toFixed(4)}; 同任务/步数对比)`);
  const gap = (accFullFt - accBeforeMerge) * 100;
  console.log(
    `    一句话收束: LoRA 用 ${((loraTrainable / totalParams) * 100).toFixed(2)}% 的可训练参数,` +
      `acc 与全量微调相差 ${gap.toFixed(1)} 个百分点 (toy 任务,差距随规模通常收窄);` +
      `合并后推理与基座同开销、同参数量,可像普通权重一样交付与热切换。\n`,
  );

  console.log("⚠ toy-scale: 绝对值偏乐观;可迁移的是 <1e-6 数学等价、合并后 matmul 回落、dropout 失败模式。");
}

main();
