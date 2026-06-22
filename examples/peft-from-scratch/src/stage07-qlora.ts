// stage07-qlora.ts — QLoRA: quantize the FROZEN base, then train LoRA on top of it.
//
// THE ONE IDEA this stage demonstrates with real numbers: a frozen base can tolerate LOW
//   precision (we store every base weight via per-channel int-k quant + dequant on read),
//   while the tiny TRAINABLE increment (LoRA's B@A) stays full fp32. "Frozen tolerates low
//   precision, increment needs high precision" is the transferable structure of QLoRA — not
//   the toy absolute error magnitudes or accuracies below.
//
// WHY this is more than LoRA: chapter 03's LoRA already shrinks gradients + optimizer state
//   (only the LoRA params get them). QLoRA additionally shrinks the WEIGHTS term by storing
//   the resident frozen base at < 4 bytes/param. So on the memory bar QLoRA is the only
//   variant whose *base weights* column drops, on top of LoRA's already-tiny grad/optim.
//
// WHAT IS REALLY MEASURED HERE (honest-number contract):
//   - dequant reconstruction error: real max/mean |W - dequant(W)| over a base matrix.
//   - base task-A retention: real argmax accuracy of the quantized-but-frozen base on its
//     pretrain task, swept over bit-width × {per-channel, global}. THIS is the clean, robust
//     signal (monotone, reproducible) — it directly measures "did quantization break the
//     base's learned skill".
//   - QLoRA vs FP-LoRA on task B: both trained from the SAME pretrained base, same seed,
//     same steps; only base storage precision differs. The gap is computed, not asserted.
//   - memory: estBytes (est., toy formula), labeled (est.).
//
// FAILURE MODE demonstrated (not just happy path): drop to 2-bit AND use a single global
//   scale for the whole matrix. Reconstruction error explodes (a few large rows crush every
//   other row's resolution to ~3 levels), the frozen base's task-A skill collapses, and LoRA
//   — which can only ADD a low-rank delta on q/v — cannot rebuild a base whose internal
//   representation was flattened. We print the collapsed retention next to the healthy run.
//
// ⚠ toy-scale: bit-widths, error magnitudes and absolute accuracies are toy quantization on a
//   ~few-thousand-param model over synthetic copy→reverse data, so absolutes are optimistic.
//   Transferable: the STRUCTURE (frozen=low-precision OK down to a bit-width floor, increment=
//   high-precision needed) and the relative ordering (per-channel > global; 8/4/3-bit fine,
//   2-bit breaks; QLoRA ≈ FP-LoRA ≫ over-quantized base).

import { Tensor } from "./core/tensor.js";
import { Module, Embedding, Linear, TransformerBlock } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { seed, normal } from "./core/prng.js";
import { genPretrainFinetune, batches, type SeqExample } from "./core/data.js";
import { dump, loadBase, type Checkpoint } from "./core/checkpoint.js";
import { estBytes, toMB } from "./core/mem.js";
import { histogram, bar, lossCurve } from "./core/viz.js";

// ---------------------------------------------------------------------------
// Toy quantization. Symmetric int-k: q = round(w / scale), w_hat = q * scale, with scale
//   chosen so the largest-magnitude weight maps to the int range edge.
// PER-CHANNEL means one scale per output row of W (shape (out,in)); GLOBAL means one scale for
//   the whole matrix. The contrast is the pedagogical core of the failure mode: a single
//   global scale lets one large-magnitude row crush the resolution of every other row, so most
//   weights round to a handful of coarse levels.
// ---------------------------------------------------------------------------

/** Largest representable signed magnitude for k-bit symmetric quant, e.g. 4-bit -> 7. */
function intMax(bits: number): number {
  // signed range [-(2^(k-1)), 2^(k-1)-1]; symmetric scheme uses ±(2^(k-1)-1) so 0 is exact.
  return Math.pow(2, bits - 1) - 1;
}

/** Count of distinct quant levels for k-bit symmetric quant, for honest "台阶数" reporting. */
function numLevels(bits: number): number {
  return 2 * intMax(bits) + 1;
}

/**
 * Quantize then immediately dequantize a (out,in) weight buffer. Returns the reconstructed
 *   fp64 buffer (what a QLoRA forward pass actually reads after on-the-fly dequant).
 * INVARIANT: a zero/near-zero row must NOT divide by zero — scale is floored at a tiny epsilon
 *   so a dead channel reconstructs to zero instead of NaN.
 * WHY round-trip in one function: the book never needs the packed ints, only the
 *   reconstruction the forward pass sees plus its error against the original.
 */
function quantDequant(
  W: Float64Array,
  shape: [number, number],
  bits: number,
  mode: "per-channel" | "global",
): Float64Array {
  const [out, inDim] = shape;
  const qmax = intMax(bits);
  const recon = new Float64Array(W.length);
  if (mode === "global") {
    let maxAbs = 0;
    for (let i = 0; i < W.length; i++) maxAbs = Math.max(maxAbs, Math.abs(W[i]));
    const scale = Math.max(maxAbs / qmax, 1e-12);
    for (let i = 0; i < W.length; i++) {
      const q = Math.max(-qmax, Math.min(qmax, Math.round(W[i] / scale)));
      recon[i] = q * scale;
    }
    return recon;
  }
  for (let r = 0; r < out; r++) {
    let maxAbs = 0;
    for (let c = 0; c < inDim; c++) maxAbs = Math.max(maxAbs, Math.abs(W[r * inDim + c]));
    const scale = Math.max(maxAbs / qmax, 1e-12);
    for (let c = 0; c < inDim; c++) {
      const idx = r * inDim + c;
      const q = Math.max(-qmax, Math.min(qmax, Math.round(W[idx] / scale)));
      recon[idx] = q * scale;
    }
  }
  return recon;
}

/** Max and mean absolute reconstruction error between an original and its dequantized form. */
function reconError(orig: Float64Array, recon: Float64Array): { max: number; mean: number } {
  let max = 0;
  let sum = 0;
  for (let i = 0; i < orig.length; i++) {
    const e = Math.abs(orig[i] - recon[i]);
    if (e > max) max = e;
    sum += e;
  }
  return { max, mean: sum / orig.length };
}

// ---------------------------------------------------------------------------
// Model: Embedding -> TransformerBlock -> output projection to vocab logits.
// Thin wrapper so we can (a) checkpoint a clean base, (b) re-instantiate identical copies for
//   the FP-LoRA vs QLoRA comparison, and (c) reach into attn.q / attn.v to attach LoRA — the
//   realistic PEFT injection site.
// ---------------------------------------------------------------------------

const D_MODEL = 24;
const HEADS = 2;
const D_FF = 48;

class SeqModel extends Module {
  embed: Embedding;
  block: TransformerBlock;
  outProj: Linear; // (vocab, d_model) -> logits over vocab per position
  constructor(vocab: number) {
    super();
    this.embed = new Embedding(vocab, D_MODEL);
    this.block = new TransformerBlock(D_MODEL, HEADS, D_FF);
    this.outProj = new Linear(D_MODEL, vocab);
  }
  override forward(_x: Tensor): Tensor {
    throw new Error("SeqModel: use runIds(ids)");
  }
  runIds(ids: number[]): Tensor {
    const h = this.embed.lookup(ids); // (seq, d_model)
    const z = this.block.forward(h); // (seq, d_model)
    return this.outProj.forward(z); // (seq, vocab)
  }
  /** The Linear weight matrices that make up the base "knowledge" we quantize. */
  baseLinears(): Linear[] {
    const a = this.block.attn;
    return [a.q, a.k, a.v, a.o, this.block.ff1, this.block.ff2, this.outProj];
  }
}

/**
 * Quantize EVERY base Linear weight in place (per-channel or global, k-bit), simulating a
 *   QLoRA base stored low-precision and dequantized on read. We overwrite W.data with the
 *   reconstruction so the forward pass naturally reads dequantized weights.
 * INVARIANT: only weight matrices are quantized (biases / LayerNorm gains stay fp32) — matches
 *   real QLoRA which leaves the tiny non-matmul params high-precision.
 */
function quantizeBaseInPlace(model: SeqModel, bits: number, mode: "per-channel" | "global"): void {
  for (const lin of model.baseLinears()) {
    const recon = quantDequant(lin.W.data, lin.W.shape as [number, number], bits, mode);
    lin.W.data.set(recon);
  }
}

// ---------------------------------------------------------------------------
// LoRA adapter wrapping a single base Linear. delta = (alpha/r) * (x @ Aᵀ @ Bᵀ) added to the
//   FROZEN base output. A is normal-init, B is ZERO-init so training starts as a no-op — the
//   standard LoRA trick that begins adaptation from the base's exact behavior.
// We read the base Linear's OWN W/b tensors (already quantized in place for QLoRA) so there is
//   one source of truth for the base weights.
// ---------------------------------------------------------------------------

class LoraLinear extends Module {
  base: Linear; // frozen; W already dequantized for QLoRA
  loraA: Tensor; // (r, in) trainable, normal-init
  loraB: Tensor; // (out, r) trainable, ZERO-init -> delta starts at 0
  scale: number; // alpha / r
  constructor(base: Linear, rank: number, alpha: number) {
    super();
    const [out, inDim] = base.W.shape;
    this.base = base;
    base.W.requires_grad = false;
    base.b.requires_grad = false;
    const a = new Float64Array(rank * inDim);
    for (let i = 0; i < a.length; i++) a[i] = normal(0, 1 / Math.sqrt(inDim));
    this.loraA = new Tensor(a, [rank, inDim], true, [], "lora.A");
    this.loraB = new Tensor(new Float64Array(out * rank), [out, rank], true, [], "lora.B");
    this.scale = alpha / rank;
  }
  override forward(x: Tensor): Tensor {
    const baseOut = x.matmul(this.base.W.transpose()).add(this.base.b);
    const lowRank = x.matmul(this.loraA.transpose()).matmul(this.loraB.transpose()).scale(this.scale);
    return baseOut.add(lowRank);
  }
}

/**
 * A SeqModel whose attention q and v projections route through LoRA wrappers over the (possibly
 *   quantized) frozen base. Everything else is frozen.
 * INVARIANT: only the 4 LoRA tensors (A,B for q and v) have requires_grad=true; trainable()
 *   must return exactly those 4.
 */
class QLoraSeqModel extends Module {
  base: SeqModel;
  loraQ: LoraLinear;
  loraV: LoraLinear;
  constructor(base: SeqModel, rank: number, alpha: number) {
    super();
    base.freeze(); // every base leaf read-only first
    this.base = base;
    this.loraQ = new LoraLinear(base.block.attn.q, rank, alpha);
    this.loraV = new LoraLinear(base.block.attn.v, rank, alpha);
    // Route attention q/v through the LoRA wrappers. WHY monkey-patch forward rather than
    //   subclass MHA: keeps core/nn untouched and the injection site local + explicit.
    base.block.attn.q.forward = (x: Tensor) => this.loraQ.forward(x);
    base.block.attn.v.forward = (x: Tensor) => this.loraV.forward(x);
  }
  override forward(_x: Tensor): Tensor {
    throw new Error("QLoraSeqModel: use runIds(ids)");
  }
  runIds(ids: number[]): Tensor {
    return this.base.runIds(ids);
  }
  /** Base leaves + the 4 LoRA tensors. trainable() filters to just the LoRA tensors. */
  override parameters(): Tensor[] {
    return [...this.base.parameters(), this.loraQ.loraA, this.loraQ.loraB, this.loraV.loraA, this.loraV.loraB];
  }
}

// ---------------------------------------------------------------------------
// Loss + accuracy. Loss = MSE between softmax(logits) and one-hot target (deterministic and
//   differentiable with core's ops). Accuracy = fraction of positions whose argmax logit
//   equals the target token — the metric a reader actually cares about.
// ---------------------------------------------------------------------------

function oneHotTarget(target: number[], vocab: number): Tensor {
  const data = new Float64Array(target.length * vocab);
  for (let i = 0; i < target.length; i++) data[i * vocab + target[i]] = 1;
  return Tensor.from(Array.from(data), [target.length, vocab], false);
}

function seqLoss(logits: Tensor, target: number[], vocab: number): Tensor {
  const diff = logits.softmax().sub(oneHotTarget(target, vocab));
  return diff.mul(diff).mean(); // scalar MSE
}

function argmaxRow(data: Float64Array, row: number, vocab: number): number {
  let best = 0;
  let bestV = -Infinity;
  for (let j = 0; j < vocab; j++) {
    const v = data[row * vocab + j];
    if (v > bestV) {
      bestV = v;
      best = j;
    }
  }
  return best;
}

/** Per-token argmax accuracy over a dataset. */
function evalAccuracy(run: (ids: number[]) => Tensor, examples: SeqExample[], vocab: number): number {
  let correct = 0;
  let total = 0;
  for (const ex of examples) {
    const logits = run(ex.input);
    for (let i = 0; i < ex.target.length; i++) {
      if (argmaxRow(logits.data, i, vocab) === ex.target[i]) correct++;
      total++;
    }
  }
  return correct / total;
}

// ---------------------------------------------------------------------------
// Training loops (small, deterministic). Return the loss curve for rendering.
// ---------------------------------------------------------------------------

function trainAll(model: SeqModel, data: SeqExample[], vocab: number, epochs: number, lr: number): number[] {
  const opt = new Adam(model.parameters(), lr);
  return runEpochs(opt, (ex) => seqLoss(model.runIds(ex.input), ex.target, vocab), data, epochs);
}

function trainLora(model: QLoraSeqModel, data: SeqExample[], vocab: number, epochs: number, lr: number): number[] {
  const opt = new Adam(model.trainable(), lr); // optimizer sees only the 4 LoRA tensors
  return runEpochs(opt, (ex) => seqLoss(model.runIds(ex.input), ex.target, vocab), data, epochs);
}

function runEpochs(opt: Adam, lossOf: (ex: SeqExample) => Tensor, data: SeqExample[], epochs: number): number[] {
  const curve: number[] = [];
  for (let e = 0; e < epochs; e++) {
    let epochLoss = 0;
    let n = 0;
    for (const batch of batches(data, 8)) {
      for (const ex of batch) {
        opt.zeroGrad(); // grads accumulate; must zero before each backward
        const loss = lossOf(ex);
        loss.backward();
        opt.step();
        epochLoss += loss.data[0];
        n++;
      }
    }
    curve.push(epochLoss / n);
  }
  return curve;
}

/** Fresh base from a checkpoint so every run starts from the SAME pretrained weights, frozen. */
function freshBase(vocab: number, ckpt: Checkpoint): SeqModel {
  const m = new SeqModel(vocab);
  loadBase(m, ckpt, true); // copy pretrained weights AND freeze (PEFT precondition)
  return m;
}

const RANK = 4;
const ALPHA = 8;
const PRETRAIN_EPOCHS = 14;
const FINETUNE_EPOCHS = 40;

function main(): void {
  seed(1234); // single global PRNG stream; seed once before any draw

  const RULE = "─".repeat(72);
  console.log(RULE);
  console.log("Stage 07 — QLoRA: 量化冻结基座 + 在其上训练 LoRA");
  console.log(RULE);

  // --- 1. Data + pretrain a base on task A, checkpoint it ---
  const ds = genPretrainFinetune({ nPretrain: 256, nFinetune: 96, seqLen: 8, vocab: 12 });
  const vocab = ds.vocab;
  const holdout = genPretrainFinetune({ nPretrain: 1, nFinetune: 48, seqLen: 8, vocab }).finetune;
  const taskAEval = ds.pretrain.slice(0, 64);

  console.log(`\n[1] 任务: A=${ds.taskA} (pretrain) → B=${ds.taskB} (finetune). vocab=${vocab} seqLen=${ds.seqLen}`);
  const base = new SeqModel(vocab);
  const baseCurve = trainAll(base, ds.pretrain, vocab, PRETRAIN_EPOCHS, 5e-3);
  const ckpt = dump(base);
  const accClean = evalAccuracy((ids) => base.runIds(ids), taskAEval, vocab);
  console.log(`    pretrain loss ${baseCurve[0].toFixed(4)} → ${baseCurve[baseCurve.length - 1].toFixed(4)} (${baseCurve.length} epochs)`);
  console.log(`    base task-A accuracy = ${(accClean * 100).toFixed(1)}%  (the skill quantization must NOT destroy)`);

  // --- 2. Quantize a representative base matrix: histograms + reconstruction error ---
  const wq = base.block.attn.q.W;
  const qShape = wq.shape as [number, number];
  const wqOrig = wq.data.slice();
  console.log(`\n[2] 量化基座权重 (示例 Wq, shape ${qShape}) — 4-bit per-channel 对称量化`);
  const wq4pc = quantDequant(wqOrig, qShape, 4, "per-channel");
  const err4pc = reconError(wqOrig, wq4pc);
  console.log(histogram(Array.from(wqOrig), { bins: 12, width: 32, label: "  Wq 原始权重分布 (fp32):" }));
  console.log(histogram(Array.from(wq4pc), { bins: 12, width: 32, label: "  Wq 4-bit per-channel 反量化分布:" }));
  console.log(`  反量化误差 |W - dequant(W)|:  max=${err4pc.max.toExponential(3)}  mean=${err4pc.mean.toExponential(3)}`);
  console.log(`  → 离散化把连续权重钉到 ${numLevels(4)} 个台阶, per-channel scale 让每行误差受控.`);

  // --- 3. Base task-A RETENTION sweep: bit-width × {per-channel, global} ---
  //     This is the clean, robust signal: does the FROZEN quantized base still solve task A?
  console.log(`\n[3] 冻结基座的 task-A 保持率扫描 (量化全部 ${base.baseLinears().length} 个线性层, 不训练):`);
  console.log(`    bit  per-channel   global      (clean fp32 = ${(accClean * 100).toFixed(1)}%)`);
  for (const bits of [8, 4, 3, 2]) {
    const accs: Record<string, number> = {};
    for (const mode of ["per-channel", "global"] as const) {
      const m = freshBase(vocab, ckpt);
      quantizeBaseInPlace(m, bits, mode);
      accs[mode] = evalAccuracy((ids) => m.runIds(ids), taskAEval, vocab);
    }
    console.log(`    ${bits}-bit    ${(accs["per-channel"] * 100).toFixed(1).padStart(6)}%     ${(accs["global"] * 100).toFixed(1).padStart(6)}%`);
  }
  console.log(`  → 8/4/3-bit 几乎无损 (冻结基座可低精度); 2-bit 开始崩, 且 per-channel 明显优于 global.`);

  // --- 4. Full-precision LoRA (chapter 03 recipe) as the reference ---
  console.log(`\n[4] 基准: 全精度 LoRA (第 03 章配方) — 冻结 fp32 基座 + LoRA(r=${RANK}, α=${ALPHA}) on q,v`);
  const baseFP = freshBase(vocab, ckpt);
  const fpLora = new QLoraSeqModel(baseFP, RANK, ALPHA);
  const accFPbefore = evalAccuracy((ids) => fpLora.runIds(ids), holdout, vocab);
  trainLora(fpLora, ds.finetune, vocab, FINETUNE_EPOCHS, 1e-2);
  const accFP = evalAccuracy((ids) => fpLora.runIds(ids), holdout, vocab);
  const trainable = fpLora.numParams({ trainableOnly: true });
  const totalParams = fpLora.numParams();
  console.log(`    trainable=${trainable} / total=${totalParams} (${((trainable / totalParams) * 100).toFixed(2)}% 可训练)`);
  console.log(`    task-B accuracy: ${(accFPbefore * 100).toFixed(1)}% (LoRA=0 起点) → ${(accFP * 100).toFixed(1)}% (训练后)`);

  // --- 5. QLoRA: SAME base, every weight stored 4-bit per-channel (dequant on read) ---
  console.log(`\n[5] QLoRA — 同一基座, 全部权重存为 4-bit per-channel (读时反量化) + 同样的 LoRA`);
  const baseQ = freshBase(vocab, ckpt);
  quantizeBaseInPlace(baseQ, 4, "per-channel"); // quantize BEFORE freezing into LoRA
  const accQretain = evalAccuracy((ids) => baseQ.runIds(ids), taskAEval, vocab);
  const qLora = new QLoraSeqModel(baseQ, RANK, ALPHA);
  const accQbefore = evalAccuracy((ids) => qLora.runIds(ids), holdout, vocab);
  const qCurve = trainLora(qLora, ds.finetune, vocab, FINETUNE_EPOCHS, 1e-2);
  const accQ = evalAccuracy((ids) => qLora.runIds(ids), holdout, vocab);
  console.log(`    4-bit 量化后基座 task-A 保持 = ${(accQretain * 100).toFixed(1)}% (vs fp32 ${(accClean * 100).toFixed(1)}%)`);
  console.log(`    task-B accuracy: ${(accQbefore * 100).toFixed(1)}% (LoRA=0 起点) → ${(accQ * 100).toFixed(1)}% (训练后)`);
  const gap = (accFP - accQ) * 100;
  console.log(`    QLoRA vs 全精度 LoRA: ${(accQ * 100).toFixed(1)}% vs ${(accFP * 100).toFixed(1)}% → 差 ${gap.toFixed(1)}pp ` +
    `(预期小幅波动; reverse 对 q/v-only LoRA 本就难, 两者都偏低).`);
  console.log(lossCurve(qCurve, { height: 6, width: 40, label: "    QLoRA task-B 训练 loss:" }));

  // --- 6. FAILURE MODE: 2-bit + single global scale → base destroyed, LoRA can't rebuild it ---
  console.log(`\n[6] 失败模式 — 2-bit + 全局单一 scale (而非 per-channel)`);
  const wq2g = quantDequant(wqOrig, qShape, 2, "global");
  const err2g = reconError(wqOrig, wq2g);
  console.log(`  Wq 反量化误差暴涨:  max=${err2g.max.toExponential(3)}  mean=${err2g.mean.toExponential(3)} ` +
    `(对比 4-bit pc mean ${err4pc.mean.toExponential(3)}, 放大 ${(err2g.mean / err4pc.mean).toFixed(1)}×)`);
  console.log(histogram(Array.from(wq2g), { bins: 12, width: 32, label: "  Wq 2-bit global 反量化分布 (台阶塌缩):" }));

  const baseBad = freshBase(vocab, ckpt);
  quantizeBaseInPlace(baseBad, 2, "global");
  const accBadRetain = evalAccuracy((ids) => baseBad.runIds(ids), taskAEval, vocab);
  const badLora = new QLoraSeqModel(baseBad, RANK, ALPHA);
  const badCurve = trainLora(badLora, ds.finetune, vocab, FINETUNE_EPOCHS, 1e-2);
  // After training LoRA for task B, did the base's ORIGINAL task-A skill come back? (It can't:
  //   LoRA only adds a low-rank delta on q/v, and it was optimized for B, not to repair A.)
  const accBadAonA = evalAccuracy((ids) => badLora.runIds(ids), taskAEval, vocab);
  console.log(`  基座 task-A 保持率: 4-bit pc ${(accQretain * 100).toFixed(1)}%  →  2-bit global ${(accBadRetain * 100).toFixed(1)}% (基座能力被抹平)`);
  console.log(`  训练 LoRA(任务 B) 后再测 task-A = ${(accBadAonA * 100).toFixed(1)}% — LoRA 救不回被量化打崩的基座能力`);
  console.log(`    (LoRA 只能在 q/v 上 ADD 一个低秩增量, 且是为 B 优化的, 无法重建 A 的内部表征)`);
  console.log(lossCurve(badCurve, { height: 6, width: 40, label: "    2-bit global task-B 训练 loss:" }));
  console.log(`  教训: per-channel scale 隔离大幅度行; 位宽有下限 — 冻结基座可低精度, 但不是任意低.`);

  // --- 7. Memory comparison (est.): full-FT vs LoRA vs QLoRA ---
  console.log(`\n[7] 显存估算 (est., toy 公式) — full-FT vs LoRA vs QLoRA`);
  const memFull = estBytes({ totalParams, trainableParams: totalParams, optMultiplier: 2 });
  const memLora = estBytes({ totalParams, trainableParams: trainable, optMultiplier: 2 });
  const memQlora = estBytes({ totalParams, trainableParams: trainable, bytesPerBaseParam: 0.5, optMultiplier: 2 }); // 4-bit = 0.5 byte
  const mbMilli = (b: number): number => Math.round(toMB(b) * 1000); // bars in 1/1000 MB so toy values are visible
  console.log(bar([
    { label: "full-FT total", value: mbMilli(memFull.totalBytes), note: `${toMB(memFull.totalBytes).toFixed(3)} MB (est.)` },
    { label: "LoRA    total", value: mbMilli(memLora.totalBytes), note: `${toMB(memLora.totalBytes).toFixed(3)} MB (est.)` },
    { label: "QLoRA   total", value: mbMilli(memQlora.totalBytes), note: `${toMB(memQlora.totalBytes).toFixed(3)} MB (est.)` },
  ], { width: 40 }));
  console.log(`  权重项细分 (条长=相对大小, note=真实 MB):`);
  console.log(bar([
    { label: "full-FT weights", value: mbMilli(memFull.weightsBytes), note: `${toMB(memFull.weightsBytes).toFixed(3)} MB` },
    { label: "LoRA    weights", value: mbMilli(memLora.weightsBytes), note: `${toMB(memLora.weightsBytes).toFixed(3)} MB (fp32 base)` },
    { label: "QLoRA   weights", value: mbMilli(memQlora.weightsBytes), note: `${toMB(memQlora.weightsBytes).toFixed(3)} MB (4-bit base)` },
  ], { width: 40 }));
  console.log(`  grad+optim 项: full-FT=${toMB(memFull.gradBytes + memFull.optimStateBytes).toFixed(3)} MB, ` +
    `LoRA=QLoRA=${toMB(memLora.gradBytes + memLora.optimStateBytes).toFixed(3)} MB (只随可训练参数走, QLoRA 与 LoRA 相同).`);
  console.log(`  → LoRA 砍掉 grad+optim; QLoRA 在此之上再把常驻 base weights 压低 ` +
    `(${toMB(memLora.weightsBytes).toFixed(3)} → ${toMB(memQlora.weightsBytes).toFixed(3)} MB).`);

  console.log(`\n${RULE}`);
  console.log("⚠ toy-scale: 位宽/误差/准确率/显存绝对值是 toy 量化, 偏乐观.");
  console.log("  可迁移的是结构: 冻结基座可低精度 (per-channel + 足够位宽), 增量 (LoRA) 需高精度;");
  console.log("  以及相对趋势 — per-channel > global; 8/4/3-bit 几乎无损, 2-bit 打崩基座, LoRA 救不回.");
  console.log(RULE);
}

main();
