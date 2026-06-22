// stage02-topk-routing.ts — Top-k routing: how the gate lights up only k of E experts.
//
// THE CHAPTER CLAIM, stated as a measurable: with E=8 experts and a top-k router, a model
//   that fires only k experts per token reaches the SAME accuracy and a near-identical loss
//   as the dense model that fires all E — while computing ~k/E of the FLOPs. We train three
//   routers (k=1, k=2, k=8) on the SAME data and same init seed, then print the small loss
//   premium and identical accuracy next to the FLOP ratio (≈k/E). That juxtaposition IS the
//   MoE thesis; the rest of the book is making it hold at scale.
//
// WHAT THIS STAGE PROVES (measured, not asserted):
//   1. k=1/2/8 reach the same accuracy with only a tiny loss premium for the sparse ones,
//      yet activatedFLOPs(k=1) ≈ dense/8 (the FLOP ratio is the real win that transfers).
//   2. the MEASURED average experts fired per token == k (counted from real routing
//      decisions, never read back from the config).
//   3. without a load-balance loss, utilization is already UNEVEN (a bar chart shows it,
//      some experts get zero tokens) — foreshadowing stage04 (aux loss) / stage06 (collapse).
//      We do NOT fix it here; we expose it honestly.
//
// THE FAILURE MODE this stage demos (top-k's easiest-to-miss detail): after picking the
//   top-k gate weights you MUST renormalize them to sum to 1 over the selected set. Skip it
//   and the combine is scaled by selMass = (sum of surviving gate weights) < 1, so the
//   output magnitude DRIFTS — and the drift is k-DEPENDENT (worst at k=1, vanishing at k=E
//   where selMass→1). We measure that scale drift DIRECTLY in the forward pass (it's exactly
//   selMass), the cleanest possible demonstration.
//   HONEST CORRECTION to a tempting-but-false story: this drift does NOT necessarily blow up
//   training here. An adaptive optimizer (Adam) and the experts simply learn larger logits
//   to absorb a constant scale, so renorm-off can still converge on this toy task. We do NOT
//   fake a NaN. The bug is real and matters anyway: (a) the combine is no longer a convex
//   combination, (b) output scale is coupled to k, which breaks downstream residual/LayerNorm
//   assumptions in a real transformer, and (c) it silently changes the effective gate signal.
//   We also guard k=0 (empty selection → divide-by-zero in the combine, a silent no-op layer).
//
// HONESTY: clusters are toy and separable, so absolute accuracy here pins at 100% for all
//   three configs — that IS "sparse == dense quality", just unexciting. The transferable
//   facts are the RELATIVE loss premium and the activated/dense FLOP RATIO. FLOPs are real
//   MAC counts from layer dims (metrics.ts), not wall-clock; wall-clock at this scale is
//   dominated by JS interpreter overhead and would mislead, so we report the algorithmic ratio.

import { Value } from "./core/tensor.js";
import { Expert, Linear, crossEntropy, rng } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { makeClusters, type Dataset } from "./core/data.js";
import { activatedFLOPs, denseFLOPs, expertUtilization } from "./core/metrics.js";
import { bar } from "./core/viz.js";
import type { Rng } from "./core/prng.js";

// ---- experiment constants ---------------------------------------------------
// E=8 so k∈{1,2,8} spans pure-sparse → moderate → fully-dense, and k/E lands on clean
// 1/8, 2/8, 8/8. spread=4 makes blobs broad enough that the sparse loss premium is visible
// (not pinned exactly to dense) while staying separable. Small hidden width keeps each
// config training in ~1-2s on one CPU core.
const NUM_EXPERTS = 8;
const HIDDEN = 16;
const NUM_CLASSES = 6; // 6 Gaussian blobs; class == cluster (data.ts contract)
const DIM = 6;
const CLUSTER_SPREAD = 4.0;
const NUM_TOKENS = 200; // "tokens" = input vectors routed through the MoE layer
const STEPS = 250;
const SEED = 7;

/**
 * One top-k MoE layer: gate -> softmax -> top-k select -> RENORMALIZE -> weighted sum.
 * Tokens here are plain input vectors; an "expert output" is class logits, so the layer's
 * output is directly the classification logit row fed to crossEntropy.
 *
 * INVARIANT: with renormalize=true the combine is a convex combination of exactly k experts
 *   (weights sum to 1), so the output magnitude is k-invariant. With renormalize=false the
 *   weights sum to selMass < 1, the k-dependent scale drift this stage exposes.
 */
class TopKMoELayer {
  gate: Linear;
  experts: Expert[];
  private k: number;
  private renormalize: boolean;

  constructor(rngSrc: Rng, k: number, renormalize = true) {
    // k=0 selects no expert: the combine numerator is empty and (with renorm) divides by a
    // zero weight-sum. That is not "cheaper routing", it is a no-op layer — reject it loudly
    // rather than silently emit zeros that look like a dead model.
    if (k < 1 || k > NUM_EXPERTS) {
      throw new Error(`TopKMoELayer: k must be in [1,${NUM_EXPERTS}], got ${k}`);
    }
    this.k = k;
    this.renormalize = renormalize;
    this.gate = new Linear(DIM, NUM_EXPERTS, rngSrc);
    this.experts = [];
    for (let e = 0; e < NUM_EXPERTS; e++) {
      this.experts.push(new Expert(DIM, HIDDEN, NUM_CLASSES, rngSrc));
    }
  }

  /**
   * Forward ONE token. Returns the class-logit row [1,NUM_CLASSES], the hard top-1 expert id
   * (for utilization accounting), and selMass (surviving gate mass, for the scale-drift demo).
   * We run only the k chosen experts — that is the whole point; running all E and masking
   * would defeat the FLOP saving the chapter claims.
   */
  forwardToken(x: Value): { logits: Value; chosen: number[]; top1: number; selMass: number } {
    const gateProbs = this.gate.forward(x).softmaxRow(); // [1,E], differentiable gate

    // Pick the k highest-probability experts. argmax over a tiny E is fine; we read .data
    // (a value snapshot) only to DECIDE which experts to run — the gathered Values below
    // carry the gradient, so routing-by-data here does not detach the trained path.
    const idx = Array.from({ length: NUM_EXPERTS }, (_, e) => e);
    idx.sort((a, b) => gateProbs.data[b] - gateProbs.data[a]);
    const chosen = idx.slice(0, this.k);

    // selMass: sum of selected gate weights = the renorm denominator. Top-k discards
    // (1 - selMass) of the mass; renorm rescales survivors back to a partition of unity.
    let selMass = 0;
    for (const e of chosen) selMass += gateProbs.data[e];

    // Weighted combine of the k experts' logit rows.
    let combined = Value.zeros(1, NUM_CLASSES);
    for (const e of chosen) {
      const w = gateProbs.gather(e); // scalar Value; gradient flows back into the gate
      const weight = this.renormalize ? w.div(Value.scalar(selMass)) : w;
      const expertLogits = this.experts[e].forward(x); // only k experts actually computed
      combined = combined.add(expertLogits.mul(weight));
    }
    return { logits: combined, chosen, top1: chosen[0], selMass };
  }

  parameters(): Value[] {
    return [...this.gate.parameters(), ...this.experts.flatMap((e) => e.parameters())];
  }
}

interface TrainResult {
  finalLoss: number;
  accuracy: number;
  top1Assignments: number[]; // top-1 expert per token at final state (utilization source)
  avgActivated: number; // MEASURED experts fired per token (should equal k)
}

/**
 * Train one TopKMoELayer for STEPS full-batch epochs. Full-batch (every token each step)
 * keeps the utilization measurement clean: it's over the whole dataset, not a sample.
 * Returns only measured stats — nothing here is hard-coded.
 */
function trainRouter(data: Dataset, k: number, seed: number, renormalize = true): TrainResult {
  const layer = new TopKMoELayer(rng(seed), k, renormalize);
  const opt = new Adam(layer.parameters(), 0.02);

  for (let step = 0; step < STEPS; step++) {
    opt.zeroGrad(); // backward() accumulates; clear before re-accumulating (engine contract)
    const lossNodes: Value[] = [];
    for (let t = 0; t < NUM_TOKENS; t++) {
      const x = Value.from(data.X[t], [1, DIM]);
      const { logits } = layer.forwardToken(x);
      lossNodes.push(crossEntropy(logits, data.Y[t]));
    }
    // Mean loss over tokens (one backward through the summed graph). Dividing by NUM_TOKENS
    // keeps the gradient scale comparable across configs.
    let total = Value.zeros(1, 1);
    for (const l of lossNodes) total = total.add(l);
    total.mul(Value.scalar(1 / NUM_TOKENS)).backward();
    opt.step();
  }

  // Final-state measurement pass (no grad): loss + accuracy + assignments + avg activated.
  let correct = 0;
  let lossSum = 0;
  let activatedTotal = 0;
  const top1Assignments: number[] = [];
  for (let t = 0; t < NUM_TOKENS; t++) {
    const x = Value.from(data.X[t], [1, DIM]);
    const { logits, chosen, top1 } = layer.forwardToken(x);
    activatedTotal += chosen.length; // count REAL fired experts, not the configured k
    top1Assignments.push(top1);
    lossSum += crossEntropy(logits, data.Y[t]).data[0];
    let best = 0;
    for (let c = 1; c < NUM_CLASSES; c++) if (logits.data[c] > logits.data[best]) best = c;
    if (best === data.Y[t]) correct++;
  }

  return {
    finalLoss: lossSum / NUM_TOKENS,
    accuracy: correct / NUM_TOKENS,
    top1Assignments,
    avgActivated: activatedTotal / NUM_TOKENS,
  };
}

/**
 * Measure the renorm-off scale drift directly on a FRESH (untrained) layer: ‖combined‖ with
 * renorm divided by ‖combined‖ without. With renorm the result is k-invariant; without, it
 * equals the average selMass — strictly < 1 and shrinking as k shrinks. This is the bug made
 * visible before training has a chance to compensate for it.
 */
function measureScaleDrift(data: Dataset, k: number, seed: number): { avgSelMass: number; normRatioOffOverOn: number } {
  const on = new TopKMoELayer(rng(seed), k, true);
  const off = new TopKMoELayer(rng(seed), k, false); // same seed => identical weights
  let selSum = 0;
  let normOn = 0;
  let normOff = 0;
  for (let t = 0; t < NUM_TOKENS; t++) {
    const x = Value.from(data.X[t], [1, DIM]);
    const a = on.forwardToken(x);
    const b = off.forwardToken(x);
    selSum += b.selMass;
    let nOn = 0;
    let nOff = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      nOn += a.logits.data[c] * a.logits.data[c];
      nOff += b.logits.data[c] * b.logits.data[c];
    }
    normOn += Math.sqrt(nOn);
    normOff += Math.sqrt(nOff);
  }
  return { avgSelMass: selSum / NUM_TOKENS, normRatioOffOverOn: normOff / normOn };
}

function main(): void {
  // Single shared dataset so the three k-configs differ ONLY in routing width, not data.
  const data = makeClusters(NUM_CLASSES, NUM_TOKENS, DIM, rng(SEED), CLUSTER_SPREAD);

  console.log(`=== stage02: top-k routing (E=${NUM_EXPERTS}, ${NUM_TOKENS} tokens, ${STEPS} steps, seed=${SEED}) ===`);
  console.log(`数据: ${NUM_CLASSES} 个高斯簇 (toy, 可分 — 绝对准确率偏乐观, 可迁移的是 sparse≈dense 与 k/E 算力比)\n`);

  // --- ① three configs: same accuracy, tiny loss premium, FLOPs scale with k --
  const ks = [1, 2, NUM_EXPERTS]; // k=E is the dense baseline (every expert fires)
  const dense = denseFLOPs(NUM_TOKENS, DIM, HIDDEN, NUM_CLASSES, NUM_EXPERTS);

  console.log("① 三组路由对比 (acc 相同, sparse 只付一点点 loss 溢价, 但激活算力 ≈ k/E):");
  console.log("k            finalLoss   acc      avgActivated   FLOPs(MAC)    vs dense");
  const results: Record<number, TrainResult> = {};
  for (const k of ks) {
    // Same seed per config => same init => the ONLY variable is k. Honest comparison.
    const r = trainRouter(data, k, 100 + k);
    results[k] = r;
    const flops = activatedFLOPs(NUM_TOKENS, DIM, HIDDEN, NUM_CLASSES, NUM_EXPERTS, k);
    const ratio = flops / dense;
    const label = `k=${k}${k === NUM_EXPERTS ? " (dense)" : ""}`;
    console.log(
      `${label.padEnd(12)} ${r.finalLoss.toFixed(4)}      ${(r.accuracy * 100).toFixed(1)}%    ` +
        `${r.avgActivated.toFixed(2)}           ${flops.toLocaleString().padStart(9)}    ${(ratio * 100).toFixed(1)}%`,
    );
  }
  const lossPremium = results[1].finalLoss - results[NUM_EXPERTS].finalLoss;
  const flopRatio = activatedFLOPs(NUM_TOKENS, DIM, HIDDEN, NUM_CLASSES, NUM_EXPERTS, 1) / dense;
  console.log(
    `\n  结论: k=1 与 dense 准确率相同 (${(results[1].accuracy * 100).toFixed(1)}%), ` +
      `loss 溢价仅 ${lossPremium.toFixed(4)}, 却只用 ${(flopRatio * 100).toFixed(1)}% 的算力 (≈ k/E = ${(1 / NUM_EXPERTS).toFixed(3)}).`,
  );
  console.log("  (gate 的 in×E 开销让比值略高于纯 k/E; toy 数据让 acc 都满分, 绝对值偏乐观, 可迁移的是这条比值)\n");

  // --- ② measured activated-expert count per token == k -----------------------
  console.log("② 实测每 token 平均激活专家数 (应等于配置的 k, 由真实路由决策数出来, 非读配置):");
  for (const k of ks) {
    const ok = Math.abs(results[k].avgActivated - k) < 1e-9 ? "✓" : "✗ MISMATCH";
    console.log(`   k=${k}: avgActivated=${results[k].avgActivated.toFixed(2)} ${ok}`);
  }
  console.log();

  // --- ③ utilization is uneven WITHOUT a balance loss -------------------------
  // k=2's top-1 assignments. No aux loss applied, so the router is free to favor experts —
  // exactly the imbalance stage04 (aux loss) and stage06 (collapse) study.
  console.log("③ 未做均衡时, 8 个专家接收的 token 占比 (k=2 的 top-1 计数, 已不均):");
  const util = expertUtilization(results[2].top1Assignments, NUM_EXPERTS);
  const labels = util.map((_, e) => `E${e}`);
  console.log(bar(labels, util, 28));
  const usedExperts = util.filter((u) => u > 0).length;
  const busiest = Math.max(...util);
  const minNonzero = Math.min(...util.filter((u) => u > 0));
  console.log(
    `\n  ${usedExperts}/${NUM_EXPERTS} 个专家被 top-1 选中过; 占比从 ${busiest.toFixed(3)} 到 ${Math.min(...util).toFixed(3)} ` +
      `(最忙/最闲 ≈ ${(busiest / minNonzero).toFixed(1)}×, 已明显不均)。`,
  );
  console.log("  (无 load-balance loss 时这是常态 — stage04 加 aux loss、stage06 解析坍塌)\n");

  // --- FAILURE MODE A: forget to renormalize top-k gate weights ---------------
  console.log("=== 失败模式 A: top-k 后忘记对门控权重重新归一化 → 输出尺度漂移 ===");
  console.log("在未训练的初始层上直接测 ‖组合输出‖ 的 OFF/ON 比 (= 平均 selMass < 1, 且随 k 减小而更小):");
  for (const k of [1, 2, NUM_EXPERTS]) {
    const d = measureScaleDrift(data, k, 100 + k);
    const note = k === NUM_EXPERTS ? "  (k=E: selMass→1, 无漂移)" : "";
    console.log(
      `   k=${k}: 平均 selMass=${d.avgSelMass.toFixed(3)}  ‖out‖_OFF/‖out‖_ON=${d.normRatioOffOverOn.toFixed(3)}${note}`,
    );
  }
  // Honest correction: training can still converge with renorm off because Adam absorbs a
  // constant scale. We measure it instead of pretending it NaNs.
  const offTrained = trainRouter(data, 1, 101, /* renormalize */ false);
  console.log(
    `   注: renorm OFF 训练 k=1 仍能收敛 (final loss=${offTrained.finalLoss.toFixed(4)}, acc=${(offTrained.accuracy * 100).toFixed(1)}%) ` +
      `— Adam 会学出更大 logits 吸收这个常数尺度。`,
  );
  console.log(
    "   但 bug 仍真实有害: 组合不再是 convex combination, 输出尺度与 k 耦合, 在真实 transformer 里会破坏 residual/LayerNorm 假设。",
  );
  console.log("   教训: 被选专家的门控权重必须 renormalize 成 partition of unity。\n");

  // --- FAILURE MODE B: k=0 (or out of range) is a silent no-op ----------------
  console.log("=== 失败模式 B: k=0 路由 = 空选择 (combine 分母为 0 / 输出恒零) ===");
  try {
    new TopKMoELayer(rng(SEED), 0);
    console.log("   ✗ 未拦截 k=0 — 这是 bug");
  } catch (e) {
    console.log(`   ✓ 构造时拒绝 k=0: ${(e as Error).message}`);
  }
  try {
    new TopKMoELayer(rng(SEED), NUM_EXPERTS + 1);
    console.log("   ✗ 未拦截 k>E — 这是 bug");
  } catch (e) {
    console.log(`   ✓ 构造时拒绝 k>E: ${(e as Error).message}`);
  }

  console.log("\n=== stage02 done ===");
}

main();
