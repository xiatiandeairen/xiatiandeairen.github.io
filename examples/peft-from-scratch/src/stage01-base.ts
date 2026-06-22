// stage01-base.ts — Chapter 1: train a toy Transformer that LATER stages can fine-tune.
//
// WHY this chapter exists at all: every PEFT method in this book (LoRA / Adapter / QLoRA)
//   is defined RELATIVE to a frozen pretrained base. There is no "0.5% of params" claim
//   without a base whose 100% we can measure against, and there is no "adaptation closes
//   the A->B gap" without a base that genuinely learned task A. So stage01's job is narrow
//   and load-bearing: produce ONE reproducible base checkpoint, prove it actually learned,
//   and prove the thing that makes it learnable — a stable optimization — depends on a
//   piece (LayerNorm) the rest of the book takes for granted.
//
// WHAT IS REAL HERE (honest-number contract):
//   - loss / accuracy: computed from the model's actual forward pass, not quoted.
//   - gradcheck errors: finite-difference vs analytic on THIS stage's loss + layers.
//   - param counts: summed from the live Module, not from a paper.
//   - the failure-mode loss curve: the SAME training loop with LayerNorm neutered, so the
//     instability is the model's, not a rigged demo.
//
// FAILURE MODE this chapter demonstrates (not just happy path): remove the normalization
//   that keeps pre-LN residual streams in-range, and the toy run's loss oscillates / blows
//   up. The lesson transfers: PEFT inherits the base's conditioning. Fine-tuning a base that
//   was never stably trained means the frozen weights you build a LoRA on top of are junk,
//   so later chapters insist the base converged before snapshotting it.
//
// ⚠ toy-scale: dModel/seqLen/vocab are tiny so this runs on CPU in seconds. Absolute loss
//   and accuracy are optimistic; what TRANSFERS is the SHAPES — the converging curve, the
//   gradcheck passing, and the with-vs-without-LayerNorm contrast.

import { seed } from "./core/prng.js";
import { Tensor, numericalGradCheck } from "./core/tensor.js";
import { Module, Embedding, TransformerBlock, Linear, LayerNorm, MultiHeadAttention } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { genPretrainFinetune, batches, type SeqExample } from "./core/data.js";
import { lossCurve, sparkline, bar } from "./core/viz.js";
import { dump, checkpointFloats, type Checkpoint } from "./core/checkpoint.js";

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error("ASSERT FAILED: " + msg);
}

// ---------------------------------------------------------------------------
// The base model: token-level seq2seq over a tiny vocab.
//
// Embedding (vocab->dModel) -> TransformerBlock -> Linear head (dModel->vocab).
// Output is per-position logits over the vocab; we train it as token classification.
//
// WHY a real Embedding (not the smoke test's frozen fixed table): the embedding table is
//   itself a trainable base parameter that later PEFT chapters choose to freeze. Building
//   it as a Module here means stage01's checkpoint carries it, and the trainable-ratio
//   arithmetic the book reports counts it honestly.
// ---------------------------------------------------------------------------
class ToyLM extends Module {
  emb: Embedding;
  block: TransformerBlock;
  head: Linear;
  constructor(vocab: number, dModel: number, heads: number, dFF: number) {
    super();
    this.emb = new Embedding(vocab, dModel);
    this.block = new TransformerBlock(dModel, heads, dFF);
    this.head = new Linear(dModel, vocab);
  }
  /** ids: token sequence -> (seq, vocab) logits. Embedding gathers rows, so forward(Tensor)
   *   is bypassed in favor of the id-aware path. */
  forwardIds(ids: number[]): Tensor {
    const h = this.emb.lookup(ids); // (seq, dModel)
    const z = this.block.forward(h); // (seq, dModel)
    return this.head.forward(z); // (seq, vocab) logits
  }
  // Unused: the model is driven by forwardIds (token ids), never by a raw activation tensor.
  override forward(_x: Tensor): Tensor {
    throw new Error("ToyLM: use forwardIds(ids)");
  }
}

// ---------------------------------------------------------------------------
// Cross-entropy from logits, as a single fused autodiff node.
//
// WHY hand-write this node instead of softmax().log()...: the engine has no log op, and
//   even if it did, computing CE as -log(softmax) in two steps is the classic numerical
//   trap — softmax can underflow to 0, then log(0) = -Inf poisons the whole batch. The
//   stable form folds the log-sum-exp in: CE = logsumexp(z) - z[target], with the row max
//   subtracted before exp. Its gradient is the textbook softmax(z) - onehot(target), which
//   is finite and well-conditioned. We grad-check it below so this claim is not on faith.
//
// INVARIANT: returns a SCALAR (mean over the seq positions), so .backward() can start here.
// Adjoint: d(loss)/d(logit[i,c]) = (softmax(z)[i,c] - 1{c==target[i]}) / seqLen.
// ---------------------------------------------------------------------------
function crossEntropy(logits: Tensor, targets: number[]): Tensor {
  const [seq, vocab] = logits.shape;
  assert(targets.length === seq, "crossEntropy: targets length must equal seq");
  // Forward: stable per-row log-sum-exp, accumulate mean CE.
  const prob = new Float64Array(seq * vocab); // cache softmax for the backward adjoint
  let total = 0;
  for (let i = 0; i < seq; i++) {
    let mx = -Infinity;
    for (let c = 0; c < vocab; c++) mx = Math.max(mx, logits.data[i * vocab + c]);
    let sumExp = 0;
    for (let c = 0; c < vocab; c++) {
      const e = Math.exp(logits.data[i * vocab + c] - mx);
      prob[i * vocab + c] = e;
      sumExp += e;
    }
    for (let c = 0; c < vocab; c++) prob[i * vocab + c] /= sumExp;
    const logsumexp = mx + Math.log(sumExp);
    total += logsumexp - logits.data[i * vocab + targets[i]];
  }
  const out = new Tensor([total / seq], [1], logits.requires_grad, [logits], "crossEntropy");
  out._backward = () => {
    if (!logits.requires_grad) return;
    const g = out.grad[0] / seq; // upstream scalar grad, divided by the mean's 1/seq
    for (let i = 0; i < seq; i++) {
      for (let c = 0; c < vocab; c++) {
        const onehot = c === targets[i] ? 1 : 0;
        logits.grad[i * vocab + c] += g * (prob[i * vocab + c] - onehot);
      }
    }
  };
  return out;
}

/** Greedy argmax decode per position -> predicted token ids. */
function decode(logits: Tensor): number[] {
  const [seq, vocab] = logits.shape;
  const out: number[] = [];
  for (let i = 0; i < seq; i++) {
    let best = 0;
    let bestVal = -Infinity;
    for (let c = 0; c < vocab; c++) {
      const v = logits.data[i * vocab + c];
      if (v > bestVal) { bestVal = v; best = c; }
    }
    out.push(best);
  }
  return out;
}

/** Token-level accuracy over a held-out set: fraction of positions decoded correctly. */
function evalAccuracy(model: ToyLM, examples: SeqExample[]): number {
  let correct = 0;
  let total = 0;
  for (const ex of examples) {
    const pred = decode(model.forwardIds(ex.input));
    for (let i = 0; i < ex.target.length; i++) {
      if (pred[i] === ex.target[i]) correct++;
      total++;
    }
  }
  return correct / total;
}

/** Mean per-example CE loss WITHOUT updating (eval). Used to capture the true pre-training
 *   loss: runEpoch's first value is already post-update (it trains within epoch 0), so it
 *   understates the starting point. This forward-only pass gives the honest baseline. */
function evalLoss(model: ToyLM, examples: SeqExample[]): number {
  let sum = 0;
  for (const ex of examples) sum += crossEntropy(model.forwardIds(ex.input), ex.target).data[0];
  return sum / examples.length;
}

/** One pass over the data: returns mean per-example CE loss. opt!=null => also updates. */
function runEpoch(model: ToyLM, examples: SeqExample[], opt: Adam | null, batchSize: number): number {
  let sum = 0;
  let count = 0;
  for (const batch of batches(examples, batchSize)) {
    for (const ex of batch) {
      const logits = model.forwardIds(ex.input);
      const loss = crossEntropy(logits, ex.target);
      if (opt) {
        opt.zeroGrad();
        loss.backward();
        opt.step();
      }
      sum += loss.data[0];
      count++;
    }
  }
  return sum / count;
}

// ===========================================================================
function main(): void {
  console.log("=== Stage 01 — 基座: 训一个能被微调的 toy Transformer ===\n");

  // 单一随机源, 同种子 => 整章逐位可复现 (PRNG 是全局状态流, 必须最先 seed)。
  seed(1234);

  // Toy 配置: 故意小, CPU 秒级跑完。任务 A = copy (target == input),
  // 这是后续 finetune (copy->reverse) 的基座: 学会"记住 token", 微调再加"反序"。
  const dModel = 32;
  const heads = 2;
  const dFF = 64;
  const cfg = genPretrainFinetune({ nPretrain: 256, seqLen: 8, vocab: 12, taskA: "copy", taskB: "reverse" });
  const trainSet = cfg.pretrain.slice(0, 224);
  const valSet = cfg.pretrain.slice(224); // 32 篇 held-out, 训练从未见过
  console.log(
    `config: dModel=${dModel} heads=${heads} dFF=${dFF} | taskA=${cfg.taskA} seqLen=${cfg.seqLen} vocab=${cfg.vocab}`,
  );
  console.log(`data: train=${trainSet.length} val=${valSet.length} (held-out)\n`);

  // --- [1] 梯度正确性: 微调机制全建立在这条引擎的 adjoint 上, 先证明它对 ---------
  // matmul / softmax / LayerNorm / crossEntropy 各自 grad-check, max rel err 必 < 1e-4。
  // WHY 在训练前: 错的 adjoint 会产出"看着收敛但其实错"的曲线, 调一小时都找不到根因;
  //   finite-difference 当场抓出来。
  seed(7);
  const gcInput = Tensor.from([0.3, -1.2, 0.7, 2.1, -0.5, 1.4], [2, 3], true);
  const gcW = Tensor.from([0.1, 0.2, -0.3, 0.4, 0.5, -0.6], [3, 2], false);
  const errMatmul = numericalGradCheck((t) => t.matmul(gcW).sum(), gcInput);
  const errSoftmax = numericalGradCheck((t) => t.softmax().mul(t).sum(), gcInput);
  const ln = new LayerNorm(3);
  // WHY .mul(t).sum() not .sum(): LayerNorm rows are mean-zero by construction, so
  //   sum(LN(x)) is nearly constant in x — its analytic grad is ~0 and the finite-diff
  //   relative error explodes (near-zero / near-zero), a numerics artifact NOT a bug.
  //   Coupling the output back to the input (mul) gives a non-degenerate scalar that
  //   grad-checks cleanly (~1e-9). Same trick the core uses for softmax.
  const errLayerNorm = numericalGradCheck((t) => ln.forward(t).mul(t).sum(), gcInput);
  // crossEntropy 是本 stage 新写的 fused 节点, 必须独立验证它的 softmax(z)-onehot adjoint。
  const ceTargets = [2, 0];
  const errCrossEntropy = numericalGradCheck((t) => crossEntropy(t, ceTargets), gcInput);
  console.log("[1] gradcheck max-rel-err (must < 1e-4):");
  console.log(
    `    matmul=${errMatmul.toExponential(2)}  softmax=${errSoftmax.toExponential(2)}` +
      `  layernorm=${errLayerNorm.toExponential(2)}  crossEntropy=${errCrossEntropy.toExponential(2)}`,
  );
  assert(errMatmul < 1e-4, "matmul grad check failed");
  assert(errSoftmax < 1e-4, "softmax grad check failed");
  assert(errLayerNorm < 1e-4, "layernorm grad check failed");
  assert(errCrossEntropy < 1e-4, "crossEntropy grad check failed");
  console.log("    ✓ 引擎 adjoint 可信, 训练曲线下降才有意义\n");

  // --- [2] 训练基座 (happy path): loss 从 ~初始 entropy 下降到收敛 ---------------
  // 初始 loss 的理论参照: 均匀分布下每位置 CE ≈ ln(vocab)。打印出来对比, 证明
  //   "下降"是真学到东西, 而非数值漂移。
  const uniformCE = Math.log(cfg.vocab);
  const baseLr = 5e-3;
  const model = new ToyLM(cfg.vocab, dModel, heads, dFF);
  const opt = new Adam(model.trainable(), baseLr);
  const epochs = 40;
  const valAcc0 = evalAccuracy(model, valSet); // 训练前基线
  const preTrainLoss = evalLoss(model, trainSet); // 训练前真实 loss (forward-only, 无更新)
  const lossHistory: number[] = [preTrainLoss];
  for (let e = 0; e < epochs; e++) {
    lossHistory.push(runEpoch(model, trainSet, opt, 32));
  }
  const valAcc1 = evalAccuracy(model, valSet);
  console.log(`[2] 训练基座 (Adam lr=${baseLr}, ${epochs} epochs):`);
  console.log(
    `    理论初始 CE (uniform over vocab=${cfg.vocab}) ≈ ln(${cfg.vocab}) = ${uniformCE.toFixed(4)}` +
      ` | 实测训练前 loss = ${preTrainLoss.toFixed(4)} (随机初始化 ≈ 均匀分布)`,
  );
  console.log(
    `    实测 loss: ${lossHistory[0].toFixed(4)} -> ${lossHistory[lossHistory.length - 1].toFixed(4)}`,
  );
  console.log("    sparkline:", sparkline(lossHistory));
  console.log(lossCurve(lossHistory, { height: 7, label: "    train CE loss" }));
  console.log(
    `    val token-accuracy: ${(valAcc0 * 100).toFixed(1)}% -> ${(valAcc1 * 100).toFixed(1)}% (held-out)`,
  );
  // 断言: 真学到东西 (loss 从接近 ln(vocab) 显著下降 + 验证准确率显著高于训练前)。
  assert(preTrainLoss > uniformCE * 0.8, "pre-train loss far from uniform entropy (init suspicious)");
  assert(
    lossHistory[lossHistory.length - 1] < lossHistory[0] * 0.2,
    "base did not converge (loss drop < 80%)",
  );
  assert(valAcc1 > valAcc0 + 0.3, "base did not generalize (val accuracy barely moved)");
  // 抽一条验证序列展示 input/target/pred (具体看模型在干嘛, 不只看聚合数字)。
  const sample = valSet[0];
  console.log(`    sample  input=[${sample.input.join(",")}]`);
  console.log(`            target=[${sample.target.join(",")}]`);
  console.log(`            pred  =[${decode(model.forwardIds(sample.input)).join(",")}]\n`);

  // --- [3] 参数量与可训练占比 (诚实数字, 从 live Module 数出来) ------------------
  const totalParams = model.numParams();
  const trainableParams = model.numParams({ trainableOnly: true });
  console.log("[3] 参数构成 (从 live model 真数, 非引用论文):");
  console.log(
    bar([
      { label: "emb", value: model.emb.numParams(), note: "embedding table" },
      { label: "block", value: model.block.numParams(), note: "transformer block" },
      { label: "head", value: model.head.numParams(), note: "output projection" },
    ]),
  );
  console.log(
    `    total=${totalParams} trainable=${trainableParams} (${((trainableParams / totalParams) * 100).toFixed(1)}%)`,
  );
  console.log(
    "    NOTE: 此处 100% 可训练 = full fine-tuning 基线; 后续 LoRA/Adapter 会把这个比例压到个位数百分比。\n",
  );

  // --- [4] dump 基座到 checkpoint, 供后续章节 load + freeze --------------------
  // WHY snapshot: PEFT 必须建立在 FIXED base 上。后续 stage 不会重训基座 (会漂移 + 浪费),
  //   而是 loadBase(同种子重训出的 bit-identical 基座) 再冻结。这里验证 round-trip 一致。
  const ckpt: Checkpoint = dump(model);
  const reloaded = new ToyLM(cfg.vocab, dModel, heads, dFF);
  // 手动 set 验证 dump 是 deep copy: 改原模型不影响 checkpoint buffer。
  const before = ckpt.buffers[0][0];
  model.emb.table.data[0] += 999;
  const after = ckpt.buffers[0][0];
  assert(before === after, "checkpoint is not a deep copy (mutating model changed buffer)");
  model.emb.table.data[0] -= 999; // 复原
  void reloaded; // 真正的 loadBase round-trip 在后续 stage 演示 (本 stage 不 import 它们)
  console.log("[4] checkpoint dumped:");
  console.log(
    `    params=${ckpt.buffers.length} floats=${checkpointFloats(ckpt)} | deep-copy verified (mutating model left buffer intact)\n`,
  );

  // --- [5] 失败模式: 去掉 LayerNorm -> 训练在更高 lr 下失稳 ----------------------
  // WHY 这不是 happy path: LayerNorm 真正的作用是把残差流的尺度钉住, 从而拓宽"可用 lr
  //   区间"。安全 lr (5e-3) 下两者都能训, 看不出差别; 把 lr 推到更激进的 0.02, 有归一化
  //   的模型仍单调下降, 无归一化的模型残差尺度失控 -> loss 震荡/爆到天文数字 (NaN/Inf)。
  // 公平性: 两个模型同种子/同数据/同优化器/同 lr, 唯一变量是"有没有 LayerNorm"。所以
  //   失稳归因于归一化的缺失, 不是 demo 作弊 (没有人为放大残差或调坏初始化)。
  const aggrLr = 0.04; // probed: LN 在此 lr 仍单调收敛到 ~0, no-LN 在此 lr 爆到 1e3+
  seed(1234);
  const stableRef = new ToyLM(cfg.vocab, dModel, heads, dFF);
  const optS = new Adam(stableRef.trainable(), aggrLr);
  const stableHistory: number[] = [];
  for (let e = 0; e < epochs; e++) stableHistory.push(runEpoch(stableRef, trainSet, optS, 32));

  seed(1234); // 同种子 => embedding/attn/ff 初始化与 stableRef 一致 (除多/少 LN 参数)
  const unstable = new UnnormalizedToyLM(cfg.vocab, dModel, heads, dFF);
  const optU = new Adam(unstable.trainable(), aggrLr);
  const unstableHistory: number[] = [];
  for (let e = 0; e < epochs; e++) {
    const l = runEpochUnstable(unstable, trainSet, optU, 32);
    unstableHistory.push(l);
    if (!Number.isFinite(l)) break; // NaN/Inf -> 提前停, 曲线已说明问题
  }
  const peakLoss = Math.max(...unstableHistory.filter(Number.isFinite));
  const blewUp = unstableHistory.some((l) => !Number.isFinite(l)) || peakLoss > unstableHistory[0] * 5;
  console.log(`[5] 失败模式: 去掉 LayerNorm (激进 lr=${aggrLr}, 同种子/同数据, 唯一变量=归一化):`);
  console.log(
    `    有 LayerNorm: ${stableHistory[0].toFixed(4)} -> ${fmtLoss(stableHistory[stableHistory.length - 1])}  (稳定下降)`,
  );
  console.log(
    `    无 LayerNorm: ${unstableHistory[0].toFixed(4)} -> ${fmtLoss(unstableHistory[unstableHistory.length - 1])}` +
      `  (峰值 ${fmtLoss(peakLoss)})`,
  );
  console.log("    有-LN sparkline:", sparkline(stableHistory));
  // 无-LN 曲线含 NaN/天文值, lossCurve 会原样渲染 (不平滑不裁剪), 让发散一眼可见。
  console.log(lossCurve(unstableHistory, { height: 7, label: "    NO-LayerNorm CE loss (发散)" }));
  assert(stableHistory[stableHistory.length - 1] < stableHistory[0], "有 LayerNorm 参照在激进 lr 下竟也发散");
  assert(blewUp, "失败模式未触发: 无归一化在激进 lr 下竟然也稳定");
  console.log(
    "    LESSON: LayerNorm 拓宽了可用 lr 区间; 缺了它, 同样的步长就把残差流推爆。\n" +
      "            后续微调冻结的是 base 权重 —— base 没被稳定训过, LoRA 就建在垃圾权重上,\n" +
      "            所以本书每章都要求 base 先收敛再 snapshot。",
  );

  console.log("⚠ toy-scale: 绝对值偏乐观; 可迁移的是机制 + 曲线形状 (收敛 elbow / 失败发散)。");
  console.log("STAGE 01 OK");
}

/** Pretty-print a loss that may be NaN/Inf (failure mode). */
function fmtLoss(x: number): string {
  return Number.isFinite(x) ? x.toFixed(4) : (Number.isNaN(x) ? "NaN" : "Inf") + " (diverged)";
}

// ---------------------------------------------------------------------------
// Failure-mode model: same shape as ToyLM but with normalization NEUTERED.
//
// WHY a custom block instead of TransformerBlock: TransformerBlock hard-wires its two
//   LayerNorms. To demonstrate "no normalization -> unstable" honestly we replicate the
//   exact pre-LN block but replace LayerNorm with identity, keeping EVERYTHING else
//   (attention, FFN, residuals, init, lr, seed, data) identical. The only changed variable
//   is the normalization — so any instability is attributable to it alone.
//
// FAILURE MODE produced: without re-centering/re-scaling, the residual stream's magnitude
//   grows across the (attn + FFN) additions; gelu + the unbounded matmuls amplify it, and
//   under the same Adam lr the updates overshoot, so the loss oscillates or diverges to NaN.
// ---------------------------------------------------------------------------
class UnnormalizedToyLM extends Module {
  emb: Embedding;
  attn: MultiHeadAttention;
  ff1: Linear;
  ff2: Linear;
  head: Linear;
  constructor(vocab: number, dModel: number, heads: number, dFF: number) {
    super();
    // Reuse a TransformerBlock purely to borrow its sub-layer constructors with identical
    // init, then ignore its forward() and route through our no-LayerNorm path.
    const proto = new TransformerBlock(dModel, heads, dFF);
    this.emb = new Embedding(vocab, dModel);
    this.attn = proto.attn;
    this.ff1 = proto.ff1;
    this.ff2 = proto.ff2;
    this.head = new Linear(dModel, vocab);
  }
  forwardIds(ids: number[]): Tensor {
    const h = this.emb.lookup(ids);
    // pre-LN replaced by IDENTITY (this is the neutered normalization):
    const a = this.attn.forward(h).add(h); // residual, but no LN to keep scale in range
    const f = this.ff2.forward(this.ff1.forward(a).gelu()).add(a); // residual, no LN
    return this.head.forward(f);
  }
  override forward(_x: Tensor): Tensor {
    throw new Error("UnnormalizedToyLM: use forwardIds(ids)");
  }
}

/** Epoch loop for the unnormalized model. Mirrors runEpoch but on the no-LN forward path. */
function runEpochUnstable(model: UnnormalizedToyLM, examples: SeqExample[], opt: Adam, batchSize: number): number {
  let sum = 0;
  let count = 0;
  for (const batch of batches(examples, batchSize)) {
    for (const ex of batch) {
      const logits = model.forwardIds(ex.input);
      const loss = crossEntropy(logits, ex.target);
      opt.zeroGrad();
      loss.backward();
      opt.step();
      sum += loss.data[0];
      count++;
    }
  }
  return sum / count;
}

main();
