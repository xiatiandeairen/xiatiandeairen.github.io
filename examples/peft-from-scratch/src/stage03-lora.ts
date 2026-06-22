// stage03-lora.ts — Chapter 03: LoRA = the low-rank delta ΔW = BA.
//
// THE ONE IDEA: fine-tuning learns a weight UPDATE ΔW for some base matrix W. Empirically
//   that update has low "intrinsic rank" — it lives in a tiny subspace. So instead of
//   training the full (out×in) ΔW, LoRA parameterizes it as ΔW = (alpha/r) · B·A with
//   B:(out,r) and A:(r,in), r ≪ min(out,in). The forward becomes y = x·Wᵀ + (alpha/r)·x·Aᵀ·Bᵀ.
//   W stays FROZEN; only A and B train. Trainable params drop from out·in to r·(out+in).
//
// WHY B IS ZERO-INITIALIZED (the load-bearing detail this chapter proves):
//   At t=0 we need ΔW = 0 so the adapted model is EXACTLY the pretrained base — fine-tuning
//   must START from the base, not from a randomly perturbed one. BA = 0 is guaranteed iff B=0
//   (A is random so the subspace is seeded, but B=0 zeroes the product). If you also random-
//   init B, then at t=0 ΔW≠0: you have silently corrupted the base before a single gradient
//   step, and training must first CLIMB BACK toward the base before it can improve — the
//   "cold-start spike". We demonstrate both paths and measure the spike.
//
// WHY RETENTION HOLDS: the base weights are never touched (requires_grad=false). The ONLY
//   thing adaptation changes is the additive delta. So task-A performance is recoverable
//   exactly by turning the adapter OFF (scaling→0). We verify "base task barely drops" in
//   that adapter-off sense, which is the honest claim — with the adapter ON you are running
//   task B, so of course the A-shaped behavior is replaced.
//
// HONESTY: toy scale. Absolute accuracies/losses are optimistic and d_model is tiny, so the
//   measured trainable-param RATIO is far less dramatic than at real scale. We therefore
//   print BOTH the real measured toy ratio AND the same formula projected to a realistic
//   d_model, clearly labeled. Transferable signals: (a) t=0 equivalence, (b) ratio shrinks
//   as d grows, (c) cold-start spike shape, (d) adapter-off retention.

import { seed, normal } from "./core/prng.js";
import { Tensor } from "./core/tensor.js";
import { Module, Linear, MultiHeadAttention } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { genPretrainFinetune, batches, type SeqExample } from "./core/data.js";
import { sparkline, lossCurve, bar, heatmap } from "./core/viz.js";

// ---------------------------------------------------------------------------
// LoRALinear — wrap a (frozen) base Linear with a trainable low-rank delta.
// ---------------------------------------------------------------------------

/**
 * y = base(x) + scaling · (x · Aᵀ) · Bᵀ,  scaling = alpha/r.
 *
 * Shapes: base.W is (out,in). A is (r,in), B is (out,r). Then x·Aᵀ is (m,r) and (x·Aᵀ)·Bᵀ is
 *   (m,out), matching base output. The order x→A→B (down-project then up-project) is the
 *   bottleneck that makes the delta cost r·(out+in) instead of out·in.
 *
 * WHY extend Linear (not just Module): the injection site is MultiHeadAttention.q/v, which are
 *   typed `Linear` and called as `this.q.forward(x)` inside MHA. By being a Linear subclass we
 *   can swap a LoRALinear in place with no change to MHA. We ADOPT the base's W,b tensors BY
 *   REFERENCE (not copy) so: (1) Module.parameters() enumerates the SAME frozen W,b plus the
 *   trainable A,B — giving an honest trainable count; (2) there is no second `base` Module
 *   field to double-count. The base must already be frozen by the caller (requires_grad=false).
 *
 * `enabled=false` zeroes the delta contribution at forward time WITHOUT changing the learned
 *   B,A — the adapter-off switch used to demonstrate base-task retention.
 *
 * `zeroInitB` toggles the failure-mode demo: true = correct LoRA (ΔW=0 at t=0);
 *   false = the buggy variant that perturbs the base at t=0.
 */
class LoRALinear extends Linear {
  A: Tensor; // (r, in) — down-projection; random so the delta's row-space is seeded
  B: Tensor; // (out, r) — up-projection; ZERO at init so BA=0 ⇒ ΔW=0 ⇒ output == base
  scaling: number; // alpha / r
  enabled = true; // adapter on/off at forward time (does not alter learned weights)
  constructor(base: Linear, r: number, alpha: number, zeroInitB = true) {
    const [outDim, inDim] = base.W.shape;
    super(inDim, outDim, true); // allocates throwaway W,b; immediately replaced by base's refs
    // Adopt the frozen base weights BY REFERENCE. The throwaway W,b from super() are discarded.
    this.W = base.W;
    this.b = base.b;
    // A ~ N(0, σ): seeds the r-dim subspace the delta can express. 1/√in keeps the pre-B
    //   activations O(1); the exact constant is not load-bearing at toy scale.
    const aStd = 1 / Math.sqrt(inDim);
    const aData = new Float64Array(r * inDim);
    for (let i = 0; i < aData.length; i++) aData[i] = normal(0, aStd);
    this.A = new Tensor(aData, [r, inDim], true, [], "LoRA.A");
    // B zero (correct) vs random (failure demo). This single flag is the chapter's experiment.
    const bData = new Float64Array(outDim * r);
    if (!zeroInitB) {
      const bStd = 1 / Math.sqrt(r);
      for (let i = 0; i < bData.length; i++) bData[i] = normal(0, bStd);
    }
    this.B = new Tensor(bData, [outDim, r], true, [], "LoRA.B");
    this.scaling = alpha / r;
  }
  override forward(x: Tensor): Tensor {
    const baseOut = super.forward(x); // (m, out), through the frozen W,b
    if (!this.enabled) return baseOut; // adapter off ⇒ exactly the base
    const down = x.matmul(this.A.transpose()); // (m, r)
    const up = down.matmul(this.B.transpose()); // (m, out)
    return baseOut.add(up.scale(this.scaling));
  }
  /** Materialize the effective delta matrix ΔW = scaling · B·A (out,in), for the heatmap. */
  deltaW(): Tensor {
    return this.B.matmul(this.A).scale(this.scaling);
  }
}

// ---------------------------------------------------------------------------
// Toy model: fixed embeddings -> attention -> FFN, in embedding space.
// We treat the task as regression in embedding space (differentiable end-to-end) and define
//   "accuracy" by decoding each output row to its nearest vocab embedding — a real metric.
// The FFN gives the model enough capacity that the copy→reverse adaptation is actually
//   learnable, so the gap LoRA closes is non-trivial.
// ---------------------------------------------------------------------------

const D_MODEL = 32;
const HEADS = 4;
const D_FF = 64;
const SEQ_LEN = 6;
const VOCAB = 8;

// Frozen random embedding table (shared by both tasks; not trained here so the LoRA story is
//   purely about the attention projections). Built once under the active seed.
let EMB: number[][] = [];
function buildEmbeddings(): void {
  EMB = Array.from({ length: VOCAB }, () => Array.from({ length: D_MODEL }, () => normal(0, 0.5)));
}
function embed(ids: number[]): Tensor {
  const d: number[] = [];
  for (const id of ids) d.push(...EMB[id]);
  return Tensor.from(d, [ids.length, D_MODEL], false); // frozen input
}

/** Decode an (seq,d) output tensor to token ids by nearest embedding (L2). Real metric. */
function decodeNearest(out: Tensor): number[] {
  const ids: number[] = [];
  for (let i = 0; i < SEQ_LEN; i++) {
    let best = 0;
    let bestDist = Infinity;
    for (let t = 0; t < VOCAB; t++) {
      let dist = 0;
      for (let j = 0; j < D_MODEL; j++) {
        const diff = out.data[i * D_MODEL + j] - EMB[t][j];
        dist += diff * diff;
      }
      if (dist < bestDist) { bestDist = dist; best = t; }
    }
    ids.push(best);
  }
  return ids;
}

/** Per-token accuracy over a dataset: fraction of positions decoded to the target token. */
function tokenAccuracy(model: Module, data: SeqExample[]): number {
  let correct = 0;
  let total = 0;
  for (const ex of data) {
    const pred = decodeNearest(model.forward(embed(ex.input)));
    for (let i = 0; i < SEQ_LEN; i++) {
      if (pred[i] === ex.target[i]) correct++;
      total++;
    }
  }
  return correct / total;
}

/** Single attention layer + FFN, all in embedding space. q/v are the LoRA injection sites. */
class TinyModel extends Module {
  attn: MultiHeadAttention;
  ff1: Linear;
  ff2: Linear;
  constructor() {
    super();
    this.attn = new MultiHeadAttention(D_MODEL, HEADS);
    this.ff1 = new Linear(D_MODEL, D_FF);
    this.ff2 = new Linear(D_FF, D_MODEL);
  }
  override forward(x: Tensor): Tensor {
    const a = this.attn.forward(x).add(x); // residual keeps gradients healthy
    return this.ff2.forward(this.ff1.forward(a).gelu());
  }
}

/** Train `model` on `data` for `epochs`; return per-epoch mean MSE loss. */
function train(model: Module, data: SeqExample[], epochs: number, lr: number): number[] {
  const opt = new Adam(model.trainable(), lr);
  const losses: number[] = [];
  for (let e = 0; e < epochs; e++) {
    let totalLoss = 0;
    let count = 0;
    for (const batch of batches(data, 16)) {
      for (const ex of batch) {
        const out = model.forward(embed(ex.input));
        const tgt = embed(ex.target);
        const diff = out.sub(tgt);
        const loss = diff.mul(diff).mean();
        opt.zeroGrad();
        loss.backward();
        opt.step();
        totalLoss += loss.data[0];
        count++;
      }
    }
    losses.push(totalLoss / count);
  }
  return losses;
}

// ===========================================================================
// MAIN
// ===========================================================================

function main(): void {
  seed(1234); // single global stream: embeddings, weights, A/B, batch order all derive from here
  buildEmbeddings();

  // copy (A) -> reverse (B): B reuses "remember the tokens" but adds "emit in reverse order"
  //   — a localized change a low-rank delta should be able to express.
  const ds = genPretrainFinetune({ nPretrain: 256, nFinetune: 128, seqLen: SEQ_LEN, vocab: VOCAB, taskA: "copy", taskB: "reverse" });

  console.log("=== 第 03 章 LoRA：ΔW = BA 低秩分解 ===");
  console.log(`模型：MHA(d=${D_MODEL}, heads=${HEADS}) + FFN(${D_FF})；任务 A=${ds.taskA} → 任务 B=${ds.taskB}；vocab=${VOCAB}, seqLen=${SEQ_LEN}`);

  // --- Phase 1: pretrain the base on task A, then freeze it -----------------
  const base = new TinyModel();
  const preLoss = train(base, ds.pretrain, 120, 4e-3);
  const accAbefore = tokenAccuracy(base, ds.pretrain);
  base.freeze(); // PEFT base setup: every leaf requires_grad=false
  console.log(`\n[预训练] 任务 A loss ${preLoss[0].toFixed(4)} -> ${preLoss[preLoss.length - 1].toFixed(4)}  ${sparkline(preLoss)}`);
  console.log(`[预训练] 任务 A token 准确率 = ${pct(accAbefore)}（冻结后回测的基准）`);
  console.log(`[冻结后] 基座可训练参数 = ${base.numParams({ trainableOnly: true })} / ${base.numParams()}（应为 0）`);

  const accBraw = tokenAccuracy(base, ds.finetune);
  console.log(`[未适配] 冻结基座直接做任务 B 的准确率 = ${pct(accBraw)}（待弥合的 gap）`);

  // --- Phase 2: attach correct (zero-init B) LoRA to q and v ----------------
  const R = 4;
  const ALPHA = 8;
  const loraQ = new LoRALinear(base.attn.q, R, ALPHA, true);
  const loraV = new LoRALinear(base.attn.v, R, ALPHA, true);
  base.attn.q = loraQ; // inject: q/v now route through frozen base + low-rank delta
  base.attn.v = loraV;

  // INVARIANT CHECK: zero-init B ⇒ ΔW=0 ⇒ adapted output == frozen base output at t=0.
  const deltaMaxQ = maxAbs(loraQ.deltaW().data);
  const deltaMaxV = maxAbs(loraV.deltaW().data);
  const t0Delta = Math.max(deltaMaxQ, deltaMaxV);
  console.log(`\n[t=0 等价性] 零初始化 B ⇒ ΔW 逐元素最大绝对值 = ${deltaMaxQ.toExponential(2)}（q）/ ${deltaMaxV.toExponential(2)}（v）`);
  assert(t0Delta < 1e-6, `zero-init LoRA perturbed base at t=0 (ΔW max=${t0Delta})`);
  console.log(`[t=0 等价性] 断言 ΔW < 1e-6 通过：注入 LoRA 后、训练前的模型 === 冻结基座，未被扰动`);
  // Direct output-level confirmation: adapter-off output equals adapter-on output at t=0.
  const probe = embed(ds.finetune[0].input);
  loraQ.enabled = false; loraV.enabled = false;
  const offOut = base.forward(probe).data.slice();
  loraQ.enabled = true; loraV.enabled = true;
  const onOut = base.forward(probe).data;
  let outDiff = 0;
  for (let i = 0; i < onOut.length; i++) outDiff = Math.max(outDiff, Math.abs(onOut[i] - offOut[i]));
  assert(outDiff < 1e-9, `adapter on/off output differs at t=0 (${outDiff})`);
  console.log(`[t=0 等价性] 输出级确认：adapter 开/关 输出最大差 = ${outDiff.toExponential(2)}（即 t=0 时 adapter 不改变任何输出）`);

  // Param-ratio story: LoRA-trainable vs FULL fine-tune of the SAME q/v projection matrices.
  const trainableLoRA = base.numParams({ trainableOnly: true });
  const fullFTtrainable = baseProjParams(D_MODEL);
  const ratioToy = (trainableLoRA / fullFTtrainable) * 100;
  console.log(`\n[参数] LoRA: r=${R}, alpha=${ALPHA}, scaling=${(ALPHA / R).toFixed(2)}`);
  console.log(`[参数·实测] LoRA 可训练 = ${trainableLoRA}；全量微调(同一组 q/v) = ${fullFTtrainable}；占比 = ${ratioToy.toFixed(2)}%`);
  // Why this isn't <1% here: at d=${D_MODEL}, r·(out+in)=r·2d 相对 d² 还不够小。把同一公式投到真实
  //   规模，比例才掉到论文级 <1%。这是诚实的 toy 局限，可迁移的是“比例随 d 增大而下降”。
  printRatioProjection(R);

  // --- Phase 3: fine-tune the LoRA delta on task B --------------------------
  const loraLoss = train(base, ds.finetune, 200, 8e-3);
  const accBlora = tokenAccuracy(base, ds.finetune); // adapter ON
  // Retention: turn adapter OFF (delta→0, base weights untouched) and re-measure task A.
  loraQ.enabled = false; loraV.enabled = false;
  const accAretain = tokenAccuracy(base, ds.pretrain);
  // For contrast, also report task A WITH the adapter still on (expected to drop — that's task B's delta).
  loraQ.enabled = true; loraV.enabled = true;
  const accAwithAdapter = tokenAccuracy(base, ds.pretrain);

  console.log(`\n[LoRA 微调] 任务 B loss ${loraLoss[0].toFixed(4)} -> ${loraLoss[loraLoss.length - 1].toFixed(4)}  ${sparkline(loraLoss)}`);
  console.log(lossCurve(loraLoss, { height: 6, label: "    LoRA 任务 B MSE loss" }));
  console.log(`[结果] 任务 B 准确率：未适配 ${pct(accBraw)}  →  LoRA 后 ${pct(accBlora)}`);
  console.log(`[回测·关掉 adapter] 任务 A 准确率：冻结前 ${pct(accAbefore)}  →  关掉 adapter 后 ${pct(accAretain)}（几乎不降——基座本体从未被改）`);
  console.log(`[回测·开着 adapter] 任务 A 准确率 = ${pct(accAwithAdapter)}（开着 B 的 delta 跑 A，自然会降——这正说明 delta 才是被学的东西）`);
  assert(accAretain >= accAbefore - 0.05, `retention broke: adapter-off task A dropped too far (${accAretain})`);

  // The PEFT payoff visual: the learned ΔW is by construction at most rank-r.
  const dW = loraQ.deltaW();
  console.log(`\n[低秩可视化] 学到的 ΔW (q 投影, 形状 ${dW.shape[0]}×${dW.shape[1]}, 秩 ≤ r=${R})：`);
  console.log(heatmap(dW.data, [dW.shape[0], dW.shape[1]], { label: "    ΔW = (α/r)·B·A" }));

  // --- Phase 4: FAILURE MODE — random-init B ⇒ cold-start spike -------------
  console.log(`\n=== 失败模式：把 B 也随机初始化（非零）===`);
  runColdStartDemo(ds);

  console.log(`\n[结论] LoRA 用一组低秩矩阵把任务 B 准确率从 ${pct(accBraw)} 拉到 ${pct(accBlora)}，`);
  console.log(`        关掉 adapter 即恢复任务 A 到 ${pct(accAretain)}；零初始化 B 保证微调从基座出发，不是从扰动点出发。`);
  console.log("\n⚠ toy-scale: 绝对值与参数占比偏乐观（d 太小），可迁移的是机制：ΔW=BA、t=0 等价性、占比随 d 下降、冷启动尖峰、adapter-off 回退。");
}

// ---------------------------------------------------------------------------
// Failure-mode demo: random-init B perturbs the base at t=0. The honest axis of comparison
//   is zero-B vs random-B fine-tuning STARTED FROM THE SAME FROZEN BASE: random-B's first-
//   epoch loss starts ABOVE zero-B's because its nonzero ΔW must first be unlearned. We
//   rebuild the base bit-identically (re-seed) so the ONLY difference is B's init.
// ---------------------------------------------------------------------------
function runColdStartDemo(ds: ReturnType<typeof genPretrainFinetune>): void {
  const buildBase = (): TinyModel => {
    seed(1234); // bit-identical rebuild of phase-1 base + embeddings
    buildEmbeddings();
    const m = new TinyModel();
    train(m, ds.pretrain, 120, 4e-3);
    m.freeze();
    return m;
  };

  // Reference: frozen base loss on task B with NO delta — where fine-tuning starts from.
  const baseLossB = meanLoss(buildBase(), ds.finetune);

  // Correct LoRA (zero B): first-epoch loss sits at ≈ baseLossB (ΔW=0 ⇒ starts AT the base).
  const good = buildBase();
  good.attn.q = new LoRALinear(good.attn.q, 4, 8, true);
  good.attn.v = new LoRALinear(good.attn.v, 4, 8, true);
  const goodCurve = train(good, ds.finetune, 60, 8e-3);

  // Buggy LoRA (random B): ΔW≠0 at t=0 ⇒ base already perturbed ⇒ first-epoch loss starts
  //   ABOVE baseLossB AND above zero-B; training must climb back before improving.
  const bad = buildBase();
  bad.attn.q = new LoRALinear(bad.attn.q, 4, 8, false);
  bad.attn.v = new LoRALinear(bad.attn.v, 4, 8, false);
  const badCurve = train(bad, ds.finetune, 60, 8e-3);

  // The honest comparison axis is zero-B vs random-B (identical base, identical training,
  //   only B's init differs). The frozen-base loss is a secondary anchor: zero-B's first
  //   epoch sits at/just below it (started AT the base); random-B sits ABOVE zero-B because
  //   its nonzero ΔW(t=0) must first be unlearned. We do NOT claim random-B exceeds the
  //   frozen-base loss — one epoch of training can already pull it under that anchor.
  const startGap = badCurve[0] - goodCurve[0];
  console.log(`[基准] 冻结基座在任务 B 上的 loss（无任何 delta） = ${baseLossB.toFixed(4)}（微调的出发点）`);
  console.log(`[零初始化 B] 首个 epoch loss = ${goodCurve[0].toFixed(4)}（≈ 基准：t=0 时 ΔW=0，从基座出发）`);
  console.log(`[随机初始化 B] 首个 epoch loss = ${badCurve[0].toFixed(4)}（比零初始化高 +${startGap.toFixed(4)}，t=0 即扰动基座）`);
  console.log(`[对比曲线] 零初始化: ${sparkline(goodCurve)}  最终 ${goodCurve[goodCurve.length - 1].toFixed(4)}`);
  console.log(`[对比曲线] 随机初始化: ${sparkline(badCurve)}  最终 ${badCurve[badCurve.length - 1].toFixed(4)}`);
  console.log(bar([
    { label: "frozen-base (起点参照)", value: round4(baseLossB), note: "loss" },
    { label: "zero-B  init (正确)", value: round4(goodCurve[0]), note: "首 epoch loss ≈基准" },
    { label: "rand-B  init (错误)", value: round4(badCurve[0]), note: "首 epoch loss 冷启动尖峰" },
  ], { width: 28 }));
  if (startGap > 0) {
    console.log(`[解读] 随机初始化首 epoch loss 比零初始化高 ${startGap.toFixed(4)}：这是 LoRA 自己制造、需要先“撤销”的扰动。`);
    console.log(`        零初始化 B 在 t=0 把它消灭——这就是“B 必须零初始化”的必要性。`);
  } else {
    // Honesty guard: if toy randomness makes the spike disappear this run, say so plainly
    //   and fall back to the t=0 ΔW≠0 evidence rather than overclaiming.
    console.log(`[解读] 本次 toy 随机种子下首 epoch 差异不显著（${startGap.toFixed(4)}）；`);
    console.log(`        但随机 B 的 ΔW(t=0)≠0 已扰动基座，更大规模/更难任务下尖峰更稳定可见。`);
  }
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error("ASSERT FAILED: " + msg);
}

function pct(x: number): string {
  return (x * 100).toFixed(1) + "%";
}

function maxAbs(arr: Float64Array): number {
  let m = 0;
  for (let i = 0; i < arr.length; i++) m = Math.max(m, Math.abs(arr[i]));
  return m;
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

/** Trainable params of a FULL fine-tune over the same q/v projections LoRA targets:
 *   q and v, each W(d,d)+b(d). Honest denominator for the LoRA-vs-full ratio. */
function baseProjParams(d: number): number {
  return 2 * (d * d + d);
}

/** LoRA trainable params for adapting q and v at rank r: 2 matrices × (B:d×r + A:r×d). */
function loraProjParams(d: number, r: number): number {
  return 2 * (d * r + r * d);
}

/** Project the SAME ratio formula to realistic d_model values so the reader sees the toy
 *   number is small ONLY because d is small. This is arithmetic on the formula, labeled est. */
function printRatioProjection(r: number): void {
  console.log(`[参数·公式外推 (est.)] 同一 LoRA 公式 2·2dr / 2·(d²+d)，固定 r=${r}，随 d_model 变化的占比：`);
  for (const d of [32, 256, 1024, 4096]) {
    const ratio = (loraProjParams(d, r) / baseProjParams(d)) * 100;
    const tag = d === D_MODEL ? "  ← 本次 toy 实测点" : "";
    console.log(`    d_model=${String(d).padStart(4)}  →  ${ratio.toFixed(3)}% trainable${tag}`);
  }
  console.log(`    （d 越大占比越低；真实 LLM 的 d_model 常在数千量级，故论文报 <1%。est. 因只算 q/v 两矩阵。）`);
}

/** Mean MSE loss on a dataset (no training). */
function meanLoss(model: Module, data: SeqExample[]): number {
  let total = 0;
  let count = 0;
  for (const ex of data) {
    const out = model.forward(embed(ex.input));
    const tgt = embed(ex.target);
    total += out.sub(tgt).mul(out.sub(tgt)).mean().data[0];
    count++;
  }
  return total / count;
}

main();
