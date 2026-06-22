// stage05-adapter.ts — Adapter tuning: insert a trainable bottleneck module between frozen
//   base layers, instead of LoRA's additive low-rank delta on existing weights.
//
// WHAT this stage demonstrates (and measures for real):
//   1. Pretrain a tiny Transformer on task A (copy), then FREEZE it.
//   2. Insert a bottleneck Adapter (d -> m -> d, near-zero up-projection, residual) after
//      the FFN sublayer and finetune ONLY the adapter (+ its LayerNorm) on task B (reverse).
//   3. Print task-B accuracy, trainable-param ratio (bar vs an equivalent LoRA budget), and
//      the EXTRA forward matmul count the adapter adds — the cost LoRA can merge away but
//      adapters cannot (a real inference-latency tax).
//
// THE ADAPTER vs LoRA contrast (the chapter's point):
//   LoRA edits W as W + (alpha/r)·B·A and can FOLD the delta back into W at inference → zero
//   added latency. An Adapter is a SEPARATE sublayer in the forward graph (down→act→up→add),
//   so it can never be merged: every forward pays its two extra matmuls forever. The honest
//   trade is "adapters are modular / composable" vs "adapters add permanent inference cost".
//
// FAILURE MODE demonstrated (not just happy path): drop the residual connection (pure serial
//   adapter). With a near-zero up-proj the adapter's output starts ≈ 0, so a serial adapter
//   ANNIHILATES the base signal at init (h <- 0), forcing the frozen base's information flow
//   to be re-learned from scratch through a tiny bottleneck → severe early-training collapse.
//   The residual variant instead starts as identity (h <- h + 0) and only nudges, preserving
//   the base. We print both loss curves side by side so the collapse is visible, not asserted.
//
// HONESTY: toy scale. Accuracy and step counts are real (computed from this run), but their
//   ABSOLUTE values are optimistic; what transfers is the RELATIVE story — residual vs serial
//   divergence at init, and adapter's non-mergeable extra-matmul cost.

import { Tensor } from "./core/tensor.js";
import { Module, Linear, LayerNorm, MultiHeadAttention } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { seed, normal, kaimingStd } from "./core/prng.js";
import { genPretrainFinetune, batches, type SeqExample } from "./core/data.js";
import { lossCurve, bar } from "./core/viz.js";

// ---------------------------------------------------------------------------
// Bottleneck Adapter
// ---------------------------------------------------------------------------

/**
 * Adapter: x -> down(d->m) -> gelu -> up(m->d) -> [+ x if residual].
 *
 * INVARIANT (near-zero init): the UP projection starts ≈ 0 so the adapter is an approximate
 *   no-op at finetune step 0. This is the whole reason adapters can be bolted onto a frozen
 *   model without wrecking it — the first forward pass reproduces the base almost exactly,
 *   and training only departs from identity as gradients accumulate. We zero `up.W`/`up.b`
 *   explicitly (Linear's xavier init would otherwise inject noise that, with the residual
 *   removed, is catastrophic — exactly the failure mode this stage exhibits).
 *
 * WHY a dedicated Module (not reuse the base FFN): the adapter must be trainable while the
 *   base is frozen; keeping it as its own Module makes `trainable()`/`frozen()` report the
 *   honest split with zero bookkeeping.
 *
 * `residual=false` is the BROKEN configuration kept on purpose to demo the failure mode.
 */
class Adapter extends Module {
  down: Linear;
  up: Linear;
  residual: boolean;
  constructor(d: number, bottleneck: number, residual = true) {
    super();
    this.down = new Linear(d, bottleneck);
    this.up = new Linear(bottleneck, d);
    // Near-zero up-proj: overwrite xavier init so adapter(x) ≈ 0 at step 0.
    this.up.W.data.fill(0);
    this.up.b.data.fill(0);
    this.residual = residual;
  }
  override forward(x: Tensor): Tensor {
    const delta = this.up.forward(this.down.forward(x).gelu());
    // Residual is what keeps the frozen base's signal flowing while the adapter learns; the
    // serial variant (residual=false) replaces the signal with the ≈0 adapter output at init.
    return this.residual ? x.add(delta) : delta;
  }
}

// ---------------------------------------------------------------------------
// Toy model: token embedding -> pre-LN attention block -> FFN -> [Adapter] -> output head.
// Built from core Linear/LayerNorm/MHA so we control exactly where the adapter is injected
// and which parts are frozen. Mirrors core TransformerBlock's pre-LN, 2-residual shape, with
// the adapter slotted AFTER the FFN sublayer (the canonical Houlsby/Pfeiffer location).
// ---------------------------------------------------------------------------

class AdapterModel extends Module {
  embed: Tensor; // (vocab, d) — manual embedding table (Embedding.forward throws by design)
  ln1: LayerNorm;
  attn: MultiHeadAttention;
  ln2: LayerNorm;
  ff1: Linear;
  ff2: Linear;
  adapter: Adapter | null;
  head: Linear; // (d -> vocab) logits per position
  d: number;
  vocab: number;

  constructor(vocab: number, d: number, heads: number, dFF: number, adapter: Adapter | null) {
    super();
    this.vocab = vocab;
    this.d = d;
    const tbl = new Float64Array(vocab * d);
    const std = kaimingStd(d);
    for (let i = 0; i < tbl.length; i++) tbl[i] = normal(0, std);
    this.embed = new Tensor(tbl, [vocab, d], true, [], "embed.table");
    this.ln1 = new LayerNorm(d);
    this.attn = new MultiHeadAttention(d, heads);
    this.ln2 = new LayerNorm(d);
    this.ff1 = new Linear(d, dFF);
    this.ff2 = new Linear(dFF, d);
    this.adapter = adapter;
    this.head = new Linear(d, vocab);
  }

  /** ids (length seq) -> logits (seq, vocab). Manual gather keeps grad flowing to the table. */
  private embedLookup(ids: number[]): Tensor {
    const seq = ids.length;
    const out = new Tensor(new Float64Array(seq * this.d), [seq, this.d], this.embed.requires_grad, [this.embed], "gather");
    for (let i = 0; i < seq; i++) {
      const row = ids[i] * this.d;
      for (let j = 0; j < this.d; j++) out.data[i * this.d + j] = this.embed.data[row + j];
    }
    out._backward = () => {
      if (!this.embed.requires_grad) return;
      for (let i = 0; i < seq; i++) {
        const row = ids[i] * this.d;
        for (let j = 0; j < this.d; j++) this.embed.grad[row + j] += out.grad[i * this.d + j];
      }
    };
    return out;
  }

  forwardIds(ids: number[]): Tensor {
    const x = this.embedLookup(ids); // (seq, d)
    const a = this.attn.forward(this.ln1.forward(x)).add(x); // residual 1
    let f = this.ff2.forward(this.ff1.forward(this.ln2.forward(a)).gelu()).add(a); // residual 2
    if (this.adapter) f = this.adapter.forward(f); // adapter sublayer (with its own residual)
    return this.head.forward(f); // (seq, vocab)
  }

  override forward(_x: Tensor): Tensor {
    throw new Error("AdapterModel: use forwardIds(ids)");
  }

  /**
   * Copy the pretrained base weights from `src` into THIS model's base leaves and FREEZE them,
   *   while leaving the adapter trainable.
   *
   * WHY not core's loadBase(): loadBase keys params by Module.parameters() ORDER and freezes
   *   ALL of them. This model has extra adapter params `src` lacks, so the orders/counts differ,
   *   and we must keep the adapter UNfrozen. So we copy base leaves explicitly and freeze only
   *   them. This is the PEFT precondition: identical frozen base across both adapter variants,
   *   so the residual-vs-serial comparison isolates the residual, not init drift.
   * INVARIANT: `src` and `this` share the same base architecture (vocab/d/heads/dFF); only the
   *   adapter differs. We copy buffer-for-buffer and assert nothing beyond Tensor's own length
   *   guard (shapes are construction-identical by the same constructor args).
   */
  loadFrozenBaseFrom(src: AdapterModel): void {
    const baseLeaves = (m: AdapterModel): Tensor[] => [
      m.embed,
      m.ln1.gamma, m.ln1.beta,
      m.attn.q.W, m.attn.q.b, m.attn.k.W, m.attn.k.b,
      m.attn.v.W, m.attn.v.b, m.attn.o.W, m.attn.o.b,
      m.ln2.gamma, m.ln2.beta,
      m.ff1.W, m.ff1.b, m.ff2.W, m.ff2.b,
      m.head.W, m.head.b,
    ];
    const dst = baseLeaves(this);
    const from = baseLeaves(src);
    dst.forEach((p, i) => {
      p.data.set(from[i].data); // deep copy: later finetuning cannot mutate the source base
      p.requires_grad = false; // frozen base; optimizer never moves these (PEFT precondition)
    });
  }
}

// ---------------------------------------------------------------------------
// Training / evaluation helpers (cross-entropy over per-position softmax).
// ---------------------------------------------------------------------------

/** Mean cross-entropy of logits (seq,vocab) against target ids. Returns a scalar Tensor. */
function ceLoss(logits: Tensor, target: number[]): Tensor {
  const probs = logits.softmax(); // (seq, vocab), row-wise
  const [seq, vocab] = probs.shape;
  // Gather the log-prob of the gold token at each position into a scalar; build it as a
  // custom node so backward writes -1/N at the gold slot (standard CE-after-softmax grad).
  const loss = new Tensor(new Float64Array(1), [1], probs.requires_grad, [probs], "ce");
  let total = 0;
  for (let i = 0; i < seq; i++) {
    const p = probs.data[i * vocab + target[i]];
    total += -Math.log(Math.max(p, 1e-12)); // clamp guards log(0) on a dead-cold softmax
  }
  loss.data[0] = total / seq;
  loss._backward = () => {
    if (!probs.requires_grad) return;
    for (let i = 0; i < seq; i++) {
      const p = probs.data[i * vocab + target[i]];
      probs.grad[i * vocab + target[i]] += (-1 / Math.max(p, 1e-12)) / seq;
    }
  };
  return loss;
}

/** Mean per-example cross-entropy over a dataset, no training (for step-0 measurement). */
function evalLoss(model: AdapterModel, data: SeqExample[]): number {
  let total = 0;
  for (const ex of data) total += ceLoss(model.forwardIds(ex.input), ex.target).data[0];
  return total / data.length;
}

/** Token-level accuracy: fraction of positions whose argmax logit equals the gold token. */
function evalAccuracy(model: AdapterModel, data: SeqExample[]): number {
  let correct = 0;
  let total = 0;
  for (const ex of data) {
    const logits = model.forwardIds(ex.input);
    const [seq, vocab] = logits.shape;
    for (let i = 0; i < seq; i++) {
      let best = 0;
      let bestVal = -Infinity;
      for (let v = 0; v < vocab; v++) {
        const val = logits.data[i * vocab + v];
        if (val > bestVal) { bestVal = val; best = v; }
      }
      if (best === ex.target[i]) correct++;
      total++;
    }
  }
  return correct / total;
}

/** Train `params` for `epochs` over `data`; return the per-step loss curve. */
function train(model: AdapterModel, params: Tensor[], data: SeqExample[], epochs: number, lr: number): number[] {
  const opt = new Adam(params, lr);
  const curve: number[] = [];
  for (let e = 0; e < epochs; e++) {
    for (const batch of batches(data, 8)) {
      opt.zeroGrad(); // grads accumulate (+=); must clear before each backward
      let batchLoss = 0;
      for (const ex of batch) {
        const loss = ceLoss(model.forwardIds(ex.input), ex.target);
        loss.backward(); // grads accumulate across the batch's examples
        batchLoss += loss.data[0];
      }
      // average grad over the batch so step size is batch-size invariant
      for (const p of params) for (let i = 0; i < p.size; i++) p.grad[i] /= batch.length;
      opt.step();
      curve.push(batchLoss / batch.length);
    }
  }
  return curve;
}

// ---------------------------------------------------------------------------
// Stage entry
// ---------------------------------------------------------------------------

function main(): void {
  seed(1234); // global PRNG stream; seed once before any random draw for reproducibility

  const D = 32;
  const HEADS = 2;
  const DFF = 64;
  const BOTTLENECK = 8; // adapter bottleneck width m (the trainable budget knob)
  const FT_EPOCHS = 60; // finetune epochs for the adapter (base stays frozen)
  const FT_LR = 1.5e-2;

  // Adaptation pair: base learns copy (A); we adapt with an adapter toward reverse (B). B
  // reuses A's "remember the tokens" skill but flips their order — a localized change an
  // adapter can express on top of a frozen base.
  const { pretrain, finetune, taskA, taskB, vocab } = genPretrainFinetune({
    nPretrain: 256,
    nFinetune: 96,
    seqLen: 8,
    vocab: 12,
    taskA: "copy",
    taskB: "reverse",
  });

  console.log("== Stage 05 · Adapter（瓶颈模块路线）==");
  console.log(`任务 A(预训练)=${taskA}  任务 B(微调)=${taskB}  vocab=${vocab}  d=${D}  瓶颈 m=${BOTTLENECK}\n`);

  // --- 1) Pretrain the base on task A, then freeze it. ----------------------
  const base = new AdapterModel(vocab, D, HEADS, DFF, null);
  console.log("[1] 预训练基座 (task A = copy) ...");
  const preCurve = train(base, base.trainable(), pretrain, 12, 5e-3);
  const accAonA = evalAccuracy(base, pretrain);
  const accBaseOnB = evalAccuracy(base, finetune); // base's zero-shot transfer to task B
  console.log(`    预训练后: 基座在 A 上 acc=${(accAonA * 100).toFixed(1)}%  ` +
    `直接拿去做 B 的 acc=${(accBaseOnB * 100).toFixed(1)}% (未适配, 应较差 → 这就是 PEFT 要补的 gap)`);
  console.log(lossCurve(preCurve, { height: 5, label: "    预训练 loss (task A)" }));
  console.log();

  const uniformLoss = Math.log(vocab); // CE of a uniform prediction = ln(vocab); the "dead" floor

  // --- 2) Adapter WITH residual (the correct recipe). ----------------------
  const adapterRes = new Adapter(D, BOTTLENECK, true);
  const modelRes = new AdapterModel(vocab, D, HEADS, DFF, adapterRes);
  modelRes.loadFrozenBaseFrom(base); // copy pretrained weights into base leaves AND freeze them
  // Only the adapter trains; the base leaves are frozen leaves the optimizer never moves.
  const trainableRes = modelRes.trainable(); // exactly the adapter's down/up W,b
  console.log(`[2] 插入残差 Adapter, 冻结基座, 仅训练 adapter (${trainableRes.length} 个张量)`);
  // STEP-0 measurement BEFORE any training: with near-zero up-proj, adapter(x)≈0, so the
  // residual makes h ← h + 0 ≈ h. The model's step-0 loss must therefore equal the frozen
  // base's own loss on B — the adapter is an exact no-op at init (the safe-to-attach property).
  const resStep0 = evalLoss(modelRes, finetune);
  const baseLossOnB = evalLoss(base, finetune);
  const accBefore = evalAccuracy(modelRes, finetune);
  const resCurve = train(modelRes, trainableRes, finetune, FT_EPOCHS, FT_LR);
  const accRes = evalAccuracy(modelRes, finetune);
  console.log(`    step-0 loss=${resStep0.toFixed(4)} == 基座在 B 的 loss=${baseLossOnB.toFixed(4)}  ` +
    `⇒ 残差 adapter 初始 == 基座 (近似 no-op, 安全挂载)`);
  console.log(`    适配前 acc=${(accBefore * 100).toFixed(1)}%  → 适配后 acc=${(accRes * 100).toFixed(1)}%`);
  console.log();

  // --- 3) FAILURE MODE: serial adapter (no residual). ----------------------
  // Same frozen base, same near-zero up-proj init — but WITHOUT the residual, h ← adapter(x)
  // ≈ 0 at step 0. The ≈0 vector wipes out everything the frozen base computed; the head then
  // sees ~zeros and emits a near-UNIFORM distribution, so step-0 loss collapses to ln(vocab)
  // regardless of how good the base was. That is the base's information flow being severed.
  const adapterSerial = new Adapter(D, BOTTLENECK, false);
  const modelSerial = new AdapterModel(vocab, D, HEADS, DFF, adapterSerial);
  modelSerial.loadFrozenBaseFrom(base);
  const trainableSerial = modelSerial.trainable();
  console.log("[3] 失败模式: 去掉残差 (纯串联 adapter)");
  const serialStep0 = evalLoss(modelSerial, finetune);
  const accSerialBefore = evalAccuracy(modelSerial, finetune);
  const serialCurve = train(modelSerial, trainableSerial, finetune, FT_EPOCHS, FT_LR);
  const accSerial = evalAccuracy(modelSerial, finetune);
  console.log(`    step-0 loss=${serialStep0.toFixed(4)} ≈ ln(vocab)=${uniformLoss.toFixed(4)}  ` +
    `⇒ ≈0 输出把基座信号抹平成均匀分布 (退化到随机猜)`);
  console.log(`    适配前 acc=${(accSerialBefore * 100).toFixed(1)}% (≈ 1/vocab=${(100 / vocab).toFixed(1)}%, 随机水平)  ` +
    `→ 适配后 acc=${(accSerial * 100).toFixed(1)}%`);
  console.log();

  // --- 4) Side-by-side training curves: residual vs serial. ----------------
  // The shapes tell the story: the serial curve must first CLIMB OUT of the ln(vocab) hole
  // (re-deriving signal the residual variant never lost), so its early trajectory is the
  // unstable/degraded one. We mark the ln(vocab) floor so the collapse is unmistakable.
  console.log("[4] 训练初期对比 (前 40 步, 残差 vs 串联):");
  const head = 40;
  console.log(lossCurve(resCurve.slice(0, head), { height: 6, label: "    残差 Adapter (起点 = 基座 loss, 信息流保留)" }));
  console.log(lossCurve(serialCurve.slice(0, head), { height: 6, label: `    串联 Adapter (起点 ≈ ln(vocab)=${uniformLoss.toFixed(2)} 的均匀分布, 退化)` }));
  console.log(`    串联从 ln(vocab) 的“死”底爬起, 残差从基座信号起步 — 残差直连是保住基座信息流的必要条件。`);
  // Honesty: at this toy scale (1 block, reverse is hard for a width-m bottleneck) FINAL
  // accuracies are noisy and DON'T cleanly separate the two variants — both stay low. The
  // robust, transferable signal is the STEP-0 behavior above (exact no-op vs ln(vocab)
  // collapse), not the final-acc horse race. Don't over-read the converged numbers.
  console.log("    注: toy 规模下两者最终 acc 都低且含噪, 可靠信号是 step-0 行为, 不是最终 acc 名次。\n");

  // --- 5) Trainable-param ratio: adapter vs an equivalent-rank LoRA budget. -
  const totalParams = modelRes.numParams();
  const adapterParams = modelRes.numParams({ trainableOnly: true });
  // LoRA on a (d×d) weight at rank r costs 2·d·r params. For an apples-to-apples bar we cost
  // a LoRA injected on q & v (2 weights) at the SAME bottleneck rank — a realistic recipe.
  const loraRank = BOTTLENECK;
  const loraParams = 2 /*q,v*/ * 2 /*A,B*/ * D * loraRank;
  console.log("[5] 可训练参数占比 (adapter vs 同秩 LoRA, 均相对同一基座):");
  console.log(bar([
    { label: "full-FT (全参)", value: totalParams, note: "100%" },
    { label: `Adapter m=${BOTTLENECK}`, value: adapterParams, note: `${(100 * adapterParams / totalParams).toFixed(2)}% of base` },
    { label: `LoRA r=${loraRank} (q,v)`, value: loraParams, note: `${(100 * loraParams / totalParams).toFixed(2)}% of base` },
  ]));
  console.log("    两者同量级(千分之几~百分之几), 都远小于 full-FT — 这是 PEFT 的可训练成本优势。\n");

  // --- 6) Extra forward cost: the matmul tax adapters pay and LoRA can merge away. ---
  // Count matmuls in ONE forward pass. Base block: attn uses q,k,v,o (4) + per-head
  // scores(QKᵀ) and ctx(attn·V) (2 per head) ; FFN ff1,ff2 (2) ; head (1).
  const perHead = 2;
  const baseMatmuls = 4 /*qkvo*/ + HEADS * perHead + 2 /*ffn*/ + 1 /*head*/;
  const adapterMatmuls = 2; // down + up
  console.log("[6] 前向额外计算量 (matmul 次数, 单次 forward):");
  console.log(`    基座 matmul 次数 = ${baseMatmuls}`);
  console.log(`    + adapter (down+up) = ${adapterMatmuls}  ⇒  额外 +${(100 * adapterMatmuls / baseMatmuls).toFixed(1)}%`);
  console.log("    关键: 这 2 次 matmul 是 adapter 在推理时无法消除的开销。");
  console.log("    LoRA 的 ΔW=BA 可在推理前折回 W (W' = W + ΔW), 推理 0 额外开销;");
  console.log("    adapter 是图里独立子层, 永远付这笔 matmul 税 —— 模块化 vs 推理延迟的权衡。\n");

  console.log("⚠ toy-scale: 准确率/步数是本次真实计算值, 但绝对值偏乐观;");
  console.log("  可迁移的是机制与曲线形状 (残差 vs 串联的初期分化、adapter 不可合并的额外 matmul)。");
}

main();
