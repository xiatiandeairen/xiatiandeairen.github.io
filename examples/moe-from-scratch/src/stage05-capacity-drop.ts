// stage05-capacity-drop.ts — Chapter 5: capacity factor & token drop.
//
// THE CLAIM this chapter makes runnable: an MoE expert is not an infinite bucket. Each
//   expert gets a fixed CAPACITY per batch = capacity_factor * (tokens * k / E). When more
//   tokens route to an expert than it can hold, the overflow is DROPPED — those tokens skip
//   the expert and fall back to the residual path. Capacity is therefore a knob that trades
//   compute for fidelity, AND it interacts with load balance: an imbalanced router piles
//   tokens onto a few experts, blowing their capacity and dropping far more than a balanced
//   one would at the same factor.
//
// WHY this is the honest, non-survey version: every number below is computed from a real
//   forward pass over a real (seeded) dataset. Drop rate is a literal count of overflow
//   tokens. Accuracy is argmax over real combined logits. Per-class accuracy collapse is
//   measured by bucketing predictions by ground-truth class. FLOPs are real MAC counts that
//   RISE with capacity_factor — the counter-intuitive bit the chapter exists to show.
//
// HONESTY on absolutes: makeClusters is toy and near-separable, so absolute accuracy is
//   optimistic. What transfers to real MoE is the SHAPE: drop rate falls monotonically as
//   capacity rises and accuracy saturates; aux loss cuts drop at fixed capacity; a tight
//   capacity + an imbalanced router systematically destroys whole classes.
//
// FAILURE MODE demoed (not just happy path): capacity_factor=0.5 with NO load-balance loss.
//   The router collapses onto a few experts, their capacity overflows hard, the dropped
//   tokens of the starved classes only get the (weak) residual path, and the accuracy of
//   those classes craters while the favored classes look fine — information loss is
//   SYSTEMATIC, not uniform noise.
//
// Run: npx tsx src/stage05-capacity-drop.ts

import { rng } from "./core/prng.js";
import { Value } from "./core/tensor.js";
import { Linear, Expert, crossEntropy } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { makeClusters, type Dataset } from "./core/data.js";
import { bar, heatmap } from "./core/viz.js";
import {
  expertUtilization,
  coefficientOfVariation,
} from "./core/metrics.js";

// ---- toy problem geometry (kept small so a CPU converges in seconds) ----------
const SEED = 2025;
const DIM = 8;
const NUM_CLASSES = 6; // == clusters == experts; the "one expert per class" ideal
const E = NUM_CLASSES;
const HIDDEN = 24;
const K = 2; // top-2 routing
const N_TRAIN = 360;
const N_EVAL = 240;
const EPOCHS = 140;
const AUX_WEIGHT = 0.02; // load-balance loss coefficient when enabled

/** A trainable top-k MoE layer: a gate + E experts + a FROZEN residual class projection.
 *  The residual is what a DROPPED token falls back to — it mirrors the skip connection that
 *  carries an un-dispatched token forward UNCHANGED. CRITICAL design choice: the residual is
 *  NOT trained (excluded from parameters()), so it is a weak, fixed fallback — it cannot
 *  learn to be a competent standalone classifier. If we let it train, on near-separable toy
 *  data it would silently become a full classifier and a dropped token would lose almost
 *  nothing, hiding the very failure mode this chapter must show. A frozen residual makes
 *  "the experts did the real work; dropping them costs accuracy" literally true. */
class MoELayer {
  gate: Linear; // inDim -> E routing logits
  experts: Expert[]; // each inDim -> NUM_CLASSES
  residual: Linear; // inDim -> NUM_CLASSES, the FROZEN drop fallback path
  constructor(r: ReturnType<typeof rng>) {
    this.gate = new Linear(DIM, E, r);
    this.experts = [];
    for (let e = 0; e < E; e++) this.experts.push(new Expert(DIM, HIDDEN, NUM_CLASSES, r));
    this.residual = new Linear(DIM, NUM_CLASSES, r);
  }
  /** Trainable params EXCLUDE the residual — see class doc: it must stay a weak fallback. */
  parameters(): Value[] {
    return [...this.gate.parameters(), ...this.experts.flatMap((x) => x.parameters())];
  }
}

/** top-k indices of a [1,E] gate-prob row, highest first. Plain numbers — this is the hard
 *  routing decision (argmax-like), deliberately NON-differentiable, exactly like real MoE. */
function topKIndices(probs: Float64Array, k: number): number[] {
  const idx = Array.from({ length: probs.length }, (_, i) => i);
  idx.sort((a, b) => probs[b] - probs[a]);
  return idx.slice(0, k);
}

interface RouteRecord {
  tokenIdx: number;
  experts: number[]; // top-k expert ids for this token
  gateProbs: Float64Array; // [E] softmax row, captured for weighting + reuse
}

/** Phase 1: route every token (top-k) and capture the gate softmax. No capacity yet — this
 *  is the raw demand each expert sees, the input to the capacity decision. Returns per-token
 *  route records plus the top-1 assignment array used for utilization metrics. */
function routeAll(moe: MoELayer, ds: Dataset): { routes: RouteRecord[]; top1: number[] } {
  const routes: RouteRecord[] = [];
  const top1: number[] = [];
  for (let i = 0; i < ds.X.length; i++) {
    const x = Value.from(ds.X[i], [1, DIM]);
    const probs = moe.gate.forward(x).softmaxRow();
    const chosen = topKIndices(probs.data, K);
    routes.push({ tokenIdx: i, experts: chosen, gateProbs: probs.data.slice() });
    top1.push(chosen[0]);
  }
  return { routes, top1 };
}

interface CapacityResult {
  factor: number;
  capacityPerExpert: number;
  dropRate: number; // fraction of token-expert dispatches dropped to residual-only
  fullyDroppedRate: number; // fraction of tokens that lost ALL their experts
  accuracy: number;
  perClassAcc: number[];
  activatedMac: number; // real activated FLOPs given how many dispatches actually fired
}

/** Phase 2: apply a capacity_factor, drop overflow, run the surviving experts + residual,
 *  and MEASURE. This is the heart of the chapter — capacity is enforced here as a hard
 *  per-expert slot count, filled in gate-priority order (Switch-Transformer style: the most
 *  confident tokens win the slot; the rest overflow). */
function evalWithCapacity(
  moe: MoELayer,
  ds: Dataset,
  routes: RouteRecord[],
  factor: number,
): CapacityResult {
  const T = ds.X.length;
  // Capacity per expert. ceil so factor=1.0 on perfectly balanced load drops nothing.
  const capacity = Math.max(1, Math.ceil((factor * T * K) / E));

  // Fill slots in gate-confidence priority. For each expert, sort its applicants by the
  // gate prob they assigned to it; keep the top `capacity`, drop the rest. WHY priority by
  // confidence: dropping the LEAST-confident dispatches is the standard, least-harmful
  // policy — a uniformly random drop would understate how well capacity can be managed.
  const applicants: { token: number; prob: number }[][] = Array.from({ length: E }, () => []);
  for (const r of routes) {
    for (const e of r.experts) applicants[e].push({ token: r.tokenIdx, prob: r.gateProbs[e] });
  }
  const accepted: Set<number>[] = applicants.map((list) => {
    list.sort((a, b) => b.prob - a.prob);
    return new Set(list.slice(0, capacity).map((a) => a.token));
  });

  // Count drops and run the forward for accuracy.
  let totalDispatches = 0;
  let droppedDispatches = 0;
  let firedExpertRuns = 0; // for honest activated-FLOP accounting
  let fullyDropped = 0;
  let correct = 0;
  const perClassCorrect = new Array(NUM_CLASSES).fill(0);
  const perClassTotal = new Array(NUM_CLASSES).fill(0);

  for (const r of routes) {
    const x = Value.from(ds.X[r.tokenIdx], [1, DIM]);
    // Combine surviving experts, gate-weighted; renormalize weights over survivors so a
    // partially-dropped token isn't unfairly down-scaled. A fully-dropped token gets ONLY
    // the residual — the literal "skip connection carried me, the experts didn't" path.
    let combined: Value | null = null;
    let weightSum = 0;
    let survived = 0;
    for (const e of r.experts) {
      totalDispatches++;
      if (!accepted[e].has(r.tokenIdx)) {
        droppedDispatches++;
        continue;
      }
      survived++;
      firedExpertRuns++;
      const w = r.gateProbs[e];
      weightSum += w;
      const contrib = moe.experts[e].forward(x).mul(Value.scalar(w));
      combined = combined === null ? contrib : combined.add(contrib);
    }
    if (survived === 0) fullyDropped++;
    const res = moe.residual.forward(x);
    let logits: Value;
    if (combined === null) {
      logits = res; // residual-only fallback
    } else {
      // renormalize expert mixture by surviving weight, then add residual.
      logits = combined.mul(Value.scalar(1 / weightSum)).add(res);
    }
    // argmax prediction.
    let best = 0;
    for (let c = 1; c < NUM_CLASSES; c++) if (logits.data[c] > logits.data[best]) best = c;
    const y = ds.Y[r.tokenIdx];
    perClassTotal[y]++;
    if (best === y) {
      correct++;
      perClassCorrect[y]++;
    }
  }

  const perClassAcc = perClassCorrect.map((c, i) => (perClassTotal[i] ? c / perClassTotal[i] : 0));
  // Real activated FLOPs: gate for every token, plus exactly the expert runs that FIRED
  // (dropped dispatches cost nothing), plus the residual projection per token. This is why
  // higher capacity => MORE compute: fewer drops means more expert runs actually execute.
  const oneExpertMac = DIM * HIDDEN + HIDDEN * NUM_CLASSES;
  const gateMac = T * (DIM * E);
  const residualMac = T * (DIM * NUM_CLASSES);
  const activatedMac = gateMac + firedExpertRuns * oneExpertMac + residualMac;

  return {
    factor,
    capacityPerExpert: capacity,
    dropRate: droppedDispatches / totalDispatches,
    fullyDroppedRate: fullyDropped / T,
    accuracy: correct / T,
    perClassAcc,
    activatedMac,
  };
}

/** Train a MoE end-to-end. The forward here does NO capacity drop (training sees every
 *  token through its top-k experts) — capacity is an INFERENCE-time stressor we sweep after.
 *  WHY train without capacity: it isolates the variable. The aux-loss-on vs -off contrast
 *  then shows up purely as a difference in how BALANCED the learned router is, which is the
 *  thing that determines drop rate under capacity. */
function train(ds: Dataset, useAux: boolean): MoELayer {
  const r = rng(SEED + (useAux ? 1 : 0)); // distinct seeds so the two runs aren't identical
  const moe = new MoELayer(r);
  const opt = new Adam(moe.parameters(), 0.03);

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    opt.zeroGrad();
    let total: Value = Value.scalar(0);
    const gateRows: Value[] = [];
    const top1: number[] = [];
    for (let i = 0; i < ds.X.length; i++) {
      const x = Value.from(ds.X[i], [1, DIM]);
      const probs = moe.gate.forward(x).softmaxRow();
      gateRows.push(probs);
      const chosen = topKIndices(probs.data, K);
      top1.push(chosen[0]);
      // gate-weighted top-k expert mixture + residual (same combine rule as eval, minus drop).
      let combined: Value | null = null;
      let wSum = 0;
      for (const e of chosen) {
        const w = probs.data[e];
        wSum += w;
        const contrib = moe.experts[e].forward(x).mul(probs.gather(e));
        combined = combined === null ? contrib : combined.add(contrib);
      }
      const res = moe.residual.forward(x);
      const logits = combined!.mul(Value.scalar(1 / wSum)).add(res);
      total = total.add(crossEntropy(logits, ds.Y[i]));
    }
    let loss = total.mul(Value.scalar(1 / ds.X.length));
    if (useAux) {
      // Switch/GShard aux: pushes the router to spread probability mass so no expert is
      // over-picked. Built inline (not core.loadBalanceLoss) because we already hold the
      // gate rows; the formula is identical: E * sum_e f_e * P_e, grad only through P_e.
      const f = expertUtilization(top1, E); // hard counts, constant
      let acc: Value = Value.zeros(1, E);
      for (const g of gateRows) acc = acc.add(g);
      const meanP = acc.mul(Value.scalar(1 / gateRows.length));
      const aux = meanP.mul(Value.from(f, [1, E])).sum().mul(Value.scalar(E));
      loss = loss.add(aux.mul(Value.scalar(AUX_WEIGHT)));
    }
    loss.backward();
    opt.step();
  }
  return moe;
}

/** Expert×class co-occurrence (top-1) for the heatmap: did experts specialize per class? */
function cooccurrence(routes: RouteRecord[], ds: Dataset): number[][] {
  const m: number[][] = Array.from({ length: E }, () => new Array(NUM_CLASSES).fill(0));
  for (const r of routes) m[r.experts[0]][ds.Y[r.tokenIdx]]++;
  return m;
}

function pct(x: number): string {
  return (x * 100).toFixed(1) + "%";
}

// ============================ run the chapter ================================

console.log("=== Stage 05: 容量因子与 token drop（top-2, E=" + E + "）===\n");
console.log("玩具数据：" + NUM_CLASSES + " 类高斯簇，cluster==class==理想专家。绝对准确率偏乐观，");
console.log("可迁移的是相对趋势：drop 率随容量单调下降、aux loss 显著降 drop、紧容量+不均衡=系统性丢信息。\n");

// spread=0.6 keeps blobs separable (so a healthy router/expert pair reaches high accuracy
// and the failure-mode contrast is unambiguous) while still overlapping enough that routing
// quality actually matters — not a trivially-solved toy.
// INVARIANT (the bug this avoids): makeClusters draws FRESH centers each call from the rng,
//   so calling it twice yields two UNRELATED cluster geometries — a model trained on one
//   cannot generalize to the other. We therefore generate ONE dataset and slice train/eval
//   from it, guaranteeing identical centers across the split.
const rdata = rng(SEED);
const fullDs = makeClusters(NUM_CLASSES, N_TRAIN + N_EVAL, DIM, rdata, 0.6);
function sliceDs(ds: Dataset, from: number, to: number): Dataset {
  return {
    X: ds.X.slice(from, to),
    Y: ds.Y.slice(from, to),
    clusterId: ds.clusterId.slice(from, to),
    dim: ds.dim,
    numClasses: ds.numClasses,
  };
}
const trainDs = sliceDs(fullDs, 0, N_TRAIN);
const evalDs = sliceDs(fullDs, N_TRAIN, N_TRAIN + N_EVAL);

const FACTORS = [0.5, 1.0, 1.5, 2.0];

// ---- Part 1: sweep capacity_factor on a BALANCED (aux-trained) router --------
console.log("─".repeat(70));
console.log("① 扫 capacity_factor（均衡路由器，带 aux loss）");
console.log("─".repeat(70));

const moeBalanced = train(trainDs, true);
const { routes: balRoutes, top1: balTop1 } = routeAll(moeBalanced, evalDs);
const balUtil = expertUtilization(balTop1, E);
console.log(
  "路由器负载 CV = " +
    coefficientOfVariation(balUtil).toFixed(3) +
    "（越接近 0 越均衡），top-1 利用率：[" +
    balUtil.map((u) => u.toFixed(2)).join(", ") +
    "]\n",
);

const balResults = FACTORS.map((f) => evalWithCapacity(moeBalanced, evalDs, balRoutes, f));
console.log("cap_factor | 每专家容量 | drop率(派发) | 全丢token | accuracy | 激活MAC");
for (const r of balResults) {
  console.log(
    "   " +
      r.factor.toFixed(1) +
      "     |    " +
      String(r.capacityPerExpert).padStart(3) +
      "     |   " +
      pct(r.dropRate).padStart(6) +
      "    |  " +
      pct(r.fullyDroppedRate).padStart(5) +
      "   |  " +
      pct(r.accuracy).padStart(6) +
      "  | " +
      r.activatedMac.toLocaleString(),
  );
}
console.log("\n观察：drop 率随 cap_factor 单调↓，accuracy ↑后饱和，激活 MAC 随容量↑（容量是省算力的反向旋钮）。");

// drop-rate bar + verify monotonic.
console.log("\ndrop 率（派发口径）随容量：");
console.log(bar(balResults.map((r) => "cf" + r.factor), balResults.map((r) => r.dropRate)));
console.log("\n激活 FLOPs（MAC）随容量上升：");
console.log(bar(balResults.map((r) => "cf" + r.factor), balResults.map((r) => r.activatedMac)));

let monotonicDrop = true;
for (let i = 1; i < balResults.length; i++) if (balResults[i].dropRate > balResults[i - 1].dropRate + 1e-9) monotonicDrop = false;
let monotonicFlop = true;
for (let i = 1; i < balResults.length; i++) if (balResults[i].activatedMac < balResults[i - 1].activatedMac - 1) monotonicFlop = false;
console.log(
  "\n断言：drop 率单调下降 = " +
    (monotonicDrop ? "✓" : "✗") +
    "；激活 MAC 随容量单调上升 = " +
    (monotonicFlop ? "✓" : "✗"),
);

// ---- Part 2: aux ON vs OFF at a FIXED tight capacity -------------------------
console.log("\n" + "─".repeat(70));
console.log("② 固定容量下 aux loss 开/关 对比 drop 率（连回第 04 章：均衡降 drop）");
console.log("─".repeat(70));

const FIXED_FACTOR = 1.0;
const moeNoAux = train(trainDs, false);
const { routes: noAuxRoutes, top1: noAuxTop1 } = routeAll(moeNoAux, evalDs);

const balAt1 = balResults.find((r) => r.factor === FIXED_FACTOR)!;
const noAuxAt1 = evalWithCapacity(moeNoAux, evalDs, noAuxRoutes, FIXED_FACTOR);

const cvNoAux = coefficientOfVariation(expertUtilization(noAuxTop1, E));
const cvBal = coefficientOfVariation(balUtil);

console.log("（cap_factor=" + FIXED_FACTOR.toFixed(1) + " 固定）");
console.log("              负载CV   drop率    accuracy");
console.log("  无 aux loss  " + cvNoAux.toFixed(3) + "   " + pct(noAuxAt1.dropRate).padStart(6) + "    " + pct(noAuxAt1.accuracy));
console.log("  有 aux loss  " + cvBal.toFixed(3) + "   " + pct(balAt1.dropRate).padStart(6) + "    " + pct(balAt1.accuracy));
const dropDelta = noAuxAt1.dropRate - balAt1.dropRate;
console.log(
  "\n观察：aux loss 把负载 CV 从 " +
    cvNoAux.toFixed(3) +
    " 压到 " +
    cvBal.toFixed(3) +
    "，drop 率相应降低 " +
    (dropDelta >= 0 ? pct(dropDelta) : "(本次未降, 见下)") +
    "。均衡 → 没有专家被挤爆 → 少 drop。",
);

// ---- Part 3: FAILURE MODE — tight capacity + imbalance = class collapse ------
console.log("\n" + "─".repeat(70));
console.log("③ 失败模式：cap_factor=0.5 且无均衡 → drop 飙升 + 对应类别 accuracy 崩塌");
console.log("─".repeat(70));

const failResult = evalWithCapacity(moeNoAux, evalDs, noAuxRoutes, 0.5);
const safeResult = balResults.find((r) => r.factor === 2.0)!;

console.log(
  "\n无 aux + cap_factor=0.5：drop 率 = " +
    pct(failResult.dropRate) +
    "，全丢 token = " +
    pct(failResult.fullyDroppedRate) +
    "（这些 token 只剩残差路径），整体 accuracy = " +
    pct(failResult.accuracy),
);
console.log("对照（有 aux + cap_factor=2.0）：drop = " + pct(safeResult.dropRate) + "，accuracy = " + pct(safeResult.accuracy));

console.log("\n按类别拆 accuracy（失败配置 vs 安全配置）——看丢信息是否「系统性」而非均匀噪声：");
console.log("  类别 |  失败(无aux,cf0.5) | 安全(aux,cf2.0)");
for (let c = 0; c < NUM_CLASSES; c++) {
  console.log(
    "   c" +
      c +
      "  |       " +
      pct(failResult.perClassAcc[c]).padStart(6) +
      "       |     " +
      pct(safeResult.perClassAcc[c]).padStart(6),
  );
}

const failVar = coefficientOfVariation(failResult.perClassAcc.map((a) => a + 1e-6));
const safeVar = coefficientOfVariation(safeResult.perClassAcc.map((a) => a + 1e-6));
const worstFail = Math.min(...failResult.perClassAcc);
console.log(
  "\n失败配置类别准确率 CV = " +
    failVar.toFixed(3) +
    "（高=不均匀，某些类被牺牲），最差类别 acc = " +
    pct(worstFail) +
    "；安全配置 CV = " +
    safeVar.toFixed(3) +
    "。",
);
console.log("结论：容量太紧 + 路由不均衡，被挤爆专家服务的类别 token 被成批 drop，只能走弱残差 →");
console.log("      这些类别准确率崩塌，而被偏爱的类别看起来完好。信息丢失是系统性的，不是均匀噪声。");

// expert×class heatmap on the imbalanced router: hot column(s) = collapse.
console.log("\n失败路由器 expert×class 共现（top-1；某行/列过热 = 专家抢光、其他饿死）：");
console.log(
  heatmap(
    cooccurrence(noAuxRoutes, evalDs),
    Array.from({ length: E }, (_, e) => "E" + e),
    Array.from({ length: NUM_CLASSES }, (_, c) => "c" + c),
  ),
);

// ---- hard assertions so a regression is caught loudly -----------------------
function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error("STAGE05 ASSERT FAIL: " + msg);
}
assert(monotonicDrop, "drop rate must fall monotonically with capacity");
assert(monotonicFlop, "activated FLOPs must rise with capacity");
assert(failResult.dropRate > safeResult.dropRate, "tight+imbalanced must drop more than loose+balanced");
assert(worstFail < safeResult.accuracy, "failure mode must crater at least one class below safe accuracy");

console.log("\n=== 全部断言通过 ===");
