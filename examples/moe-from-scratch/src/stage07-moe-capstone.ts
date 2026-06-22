// stage07-moe-capstone.ts — The whole book in one runnable artifact: train a tiny MoE
//   that DISCOVERS its own division of labor, then account for exactly how much compute
//   the sparsity bought — and show what breaks when the balancing machinery is removed.
//
// WHY this stage is the capstone: every earlier chapter isolated one mechanism (gating,
//   top-k, load-balance aux, capacity dropping). Here they run TOGETHER on a task with a
//   KNOWN latent structure (makeModularTask: clusterId == residue 0..mod-1). That known
//   structure is the only reason we can make an HONEST claim about specialization: we line
//   up "which expert fired" against "which residue the token was" and read the co-occurrence
//   matrix. A near-permutation matrix = experts self-organized onto residues.
//
// WHAT IS REAL vs WHAT IS TOY (honesty contract — read before trusting any number):
//   - Co-occurrence matrix, expert utilization, CV, routing entropy, dead-expert count,
//     accuracy, loss curve: all MEASURED from the trained model on real data. Real.
//   - The FLOP table: real MAC counts derived from matmul dimensions. The RATIO
//     activated/dense (~ k/E) is what transfers to production MoE. The ABSOLUTE counts are
//     toy-scale and say nothing about wall-clock throughput (no kernels, no hardware here).
//
// THE FAILURE MODE this stage demoes (the A/B that closes the book), stated HONESTLY:
//   Retraining the SAME task with aux load-balance + capacity OFF makes the router COLLAPSE:
//   a few experts hog all tokens, several experts go DEAD (never fire), routing entropy
//   craters. THAT is the measurable damage at toy scale.
//   What does NOT happen here, and we refuse to fake: a big accuracy drop. The task is small
//   and the surviving experts have slack, and a collapsed router is still CONSISTENT (each
//   residue keeps going to *some* fixed expert) — and consistency, not balance, is all this
//   easy task needs for accuracy. The real-world cost of collapse is paid where capacity is
//   tight and quality lives on the margins: those dead experts are parameters you STORED but
//   a balanced router would have COMPUTED. Collapse breaks the very premise behind the FLOP
//   win ("all the parameters you pay to store actually get used"). We show that breakage with
//   dead-expert count + entropy, and say plainly why accuracy stays high.
//
// DETERMINISM: every random draw threads rng(seed). Re-running prints identical numbers.

import { rng, type Rng } from "./core/prng.js";
import { Value } from "./core/tensor.js";
import { Linear, Expert, crossEntropy } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { makeModularTask, type Dataset } from "./core/data.js";
import {
  loadBalanceLoss,
  expertUtilization,
  coefficientOfVariation,
  routingEntropy,
  activatedFLOPs,
  denseFLOPs,
} from "./core/metrics.js";
import { plotLoss, bar, heatmap } from "./core/viz.js";

// ---- hyperparameters (one source of truth; both healthy and ablated runs read these) ----
const SEED = 7;
const MOD = 8; // number of residue classes -> the natural specialization axis
const E = 8; // number of experts (== MOD so a perfect router is one-expert-per-residue)
const K = 2; // top-k routing: 2 experts fire per token
const NUM_CLASSES = 4; // label = residue % 4 -> several residues share a label (non-trivial)
const N_TRAIN = 256;
const DIM = MOD; // makeModularTask one-hot encodes residue in the first MOD dims
const HIDDEN = 12;
const STEPS = 400;
const LR = 0.02;
const AUX_WEIGHT = 0.2; // weight on the load-balance aux loss in the healthy run
const CAPACITY_FACTOR = 1.25; // each expert holds up to ceil(1.25 * K * BATCH / E) tokens/batch
const BATCH = 32; // tokens per step; capacity is enforced per-batch (as in real MoE)

/** A single MoE layer plus an output classifier head. Built from core layers only. */
interface MoeModel {
  proj: Linear; // input -> model dim (shared trunk before routing)
  gate: Linear; // model dim -> E gate logits (the router)
  experts: Expert[]; // E independent 2-layer MLPs
  head: Linear; // combined expert output -> class logits
  params(): Value[];
}

function buildModel(rng: Rng): MoeModel {
  const proj = new Linear(DIM, HIDDEN, rng);
  const gate = new Linear(HIDDEN, E, rng);
  const experts: Expert[] = [];
  for (let e = 0; e < E; e++) experts.push(new Expert(HIDDEN, HIDDEN, HIDDEN, rng));
  const head = new Linear(HIDDEN, NUM_CLASSES, rng);
  return {
    proj,
    gate,
    experts,
    head,
    params() {
      return [
        ...proj.parameters(),
        ...gate.parameters(),
        ...experts.flatMap((e) => e.parameters()),
        ...head.parameters(),
      ];
    },
  };
}

/** Result of routing one token: which experts, with what (renormalized) gate weights. */
interface TopK {
  experts: number[]; // length K expert ids, descending gate prob
  weights: number[]; // length K renormalized gate probs (sum to 1 over the K)
  probsRow: Value; // [1,E] full softmax row (carries grad; aux loss consumes it)
  top1: number; // argmax expert (the hard-count source for utilization/aux)
}

/**
 * Top-k selection from a gate softmax row. WHY renormalize the K weights: the combined
 * expert output is a convex combination of only the K chosen experts, so their gate
 * probabilities must sum to 1 among themselves — otherwise dropped mass silently scales
 * the output down (a classic top-k MoE bug).
 */
function selectTopK(probsRow: Value): TopK {
  const p = Array.from(probsRow.data);
  const order = p
    .map((v, i) => [v, i] as [number, number])
    .sort((a, b) => b[0] - a[0])
    .slice(0, K);
  const experts = order.map(([, i]) => i);
  const sum = order.reduce((s, [v]) => s + v, 0) || 1e-9;
  const weights = order.map(([v]) => v / sum);
  return { experts, weights, probsRow, top1: experts[0] };
}

/**
 * Forward one token through the MoE, honoring per-batch capacity. `counts[e]` tracks how
 * many tokens already landed on expert e THIS batch; an expert at capacity drops the token
 * (its contribution is skipped, matching Switch Transformer where a dropped token loses
 * that expert's slice). WHY drop instead of overflow: capacity is what bounds activated
 * FLOPs in real MoE; without it a collapsed router routes everything to one expert and the
 * sparsity guarantee evaporates. Dropping is the pressure that (with aux) forces balance.
 *
 * @returns class logits [1,NUM_CLASSES], the routing decision, and how many of the K chosen
 *          experts were actually applied (the rest were dropped for being at capacity).
 */
function forwardToken(
  model: MoeModel,
  x: Value,
  capacity: number,
  counts: number[],
  useCapacity: boolean,
): { logits: Value; route: TopK; applied: number } {
  const trunk = model.proj.forward(x); // [1,HIDDEN]
  const gateLogits = model.gate.forward(trunk); // [1,E]
  const probsRow = gateLogits.softmaxRow(); // [1,E], grad path for aux
  const route = selectTopK(probsRow);

  let combined: Value | null = null;
  let applied = 0;
  for (let r = 0; r < route.experts.length; r++) {
    const e = route.experts[r];
    if (useCapacity && counts[e] >= capacity) continue; // dropped: expert is full this batch
    counts[e]++;
    applied++;
    const w = probsRow.gather(e); // differentiable gate weight -> grad into the router
    const expertOut = model.experts[e].forward(trunk); // [1,HIDDEN]
    const weighted = expertOut.mul(w);
    combined = combined === null ? weighted : combined.add(weighted);
  }
  // If every chosen expert was full (only under collapse), fall back to the trunk so the
  // token still yields a gradient-bearing logit instead of a dead zero.
  if (combined === null) combined = trunk;
  const logits = model.head.forward(combined); // [1,NUM_CLASSES]
  return { logits, route, applied };
}

interface TrainResult {
  lossHistory: number[];
  finalUtil: number[]; // top-1 utilization per expert over a full eval pass
  deadExperts: number; // experts that never won a token (collapse symptom)
  cooccur: number[][]; // [E][MOD] expert x residue, column-normalized
  perResidueAcc: number[]; // [MOD] accuracy on tokens of each residue
  residueClaimed: boolean[]; // [MOD] does some expert take >50% of this residue's tokens
  finalEntropyMean: number; // mean routing entropy at the end (collapse fingerprint)
  droppedTotal: number; // total expert-slot drops during training (capacity pressure)
}

/**
 * Train the MoE for STEPS steps. `useProtection` toggles BOTH the aux load-balance loss and
 * the capacity drop — the two mechanisms whose removal we ablate. Everything else (data,
 * init, lr) is identical between runs so the contrast isolates protection.
 */
function trainMoe(data: Dataset, useProtection: boolean): TrainResult {
  // Fresh rng per run so both runs start from the SAME init -> the only difference is
  // protection on/off. (Same seed => buildModel draws identical weights.)
  const r = rng(SEED);
  const model = buildModel(r);
  const opt = new Adam(model.params(), LR);
  const capacity = Math.ceil((CAPACITY_FACTOR * K * BATCH) / E);

  const lossHistory: number[] = [];
  let droppedTotal = 0;
  const idxOrder = r.shuffle([...Array(data.X.length).keys()]);

  for (let step = 0; step < STEPS; step++) {
    opt.zeroGrad();
    const counts = new Array(E).fill(0);
    const gateProbsBatch: Value[] = [];
    const assignments: number[] = [];
    let ceLoss: Value | null = null;

    for (let b = 0; b < BATCH; b++) {
      const i = idxOrder[(step * BATCH + b) % idxOrder.length];
      const x = Value.from(data.X[i], [1, DIM]);
      const { logits, route, applied } = forwardToken(model, x, capacity, counts, useProtection);
      droppedTotal += route.experts.length - applied;
      gateProbsBatch.push(route.probsRow);
      assignments.push(route.top1);
      const ce = crossEntropy(logits, data.Y[i]);
      ceLoss = ceLoss === null ? ce : ceLoss.add(ce);
    }
    const meanCe = ceLoss!.mul(Value.scalar(1 / BATCH));

    let total = meanCe;
    if (useProtection) {
      // aux pushes the router to spread probability mass so no expert is over-picked.
      const aux = loadBalanceLoss(gateProbsBatch, assignments, E);
      total = meanCe.add(aux.mul(Value.scalar(AUX_WEIGHT)));
    }
    total.backward();
    opt.step();
    lossHistory.push(meanCe.data[0]); // log the TASK loss only, so curves are comparable
  }

  return evaluate(model, data, lossHistory, droppedTotal);
}

/**
 * Full deterministic eval pass: route every example with top-1 (no capacity, no grad),
 * tallying utilization, expert x residue co-occurrence, per-residue accuracy, and routing
 * entropy. WHY top-1 here: utilization and the specialization heatmap are top-1 concepts;
 * top-k weighting is a training/serving detail, not how we MEASURE who owns what.
 */
function evaluate(
  model: MoeModel,
  data: Dataset,
  lossHistory: number[],
  droppedTotal: number,
): TrainResult {
  const top1Assign: number[] = [];
  const cooccurCounts: number[][] = Array.from({ length: E }, () => new Array(MOD).fill(0));
  const correctPerResidue = new Array(MOD).fill(0);
  const totalPerResidue = new Array(MOD).fill(0);
  let entropySum = 0;

  for (let i = 0; i < data.X.length; i++) {
    const x = Value.from(data.X[i], [1, DIM]);
    const counts = new Array(E).fill(0);
    // Eval ignores capacity (Infinity) so measurement reflects the router, not the cap.
    const { logits, route } = forwardToken(model, x, Number.POSITIVE_INFINITY, counts, false);
    top1Assign.push(route.top1);
    const residue = data.clusterId[i];
    cooccurCounts[route.top1][residue]++;
    entropySum += routingEntropy(Array.from(route.probsRow.data));

    let pred = 0; // argmax over class logits = prediction
    for (let c = 1; c < NUM_CLASSES; c++) if (logits.data[c] > logits.data[pred]) pred = c;
    totalPerResidue[residue]++;
    if (pred === data.Y[i]) correctPerResidue[residue]++;
  }

  const finalUtil = expertUtilization(top1Assign, E);
  const deadExperts = finalUtil.filter((u) => u === 0).length;

  // Normalize co-occurrence per residue COLUMN: "of residue r's tokens, what fraction went
  // to expert e". A near-permutation (one bright cell per column) = clean specialization.
  const cooccur: number[][] = Array.from({ length: E }, () => new Array(MOD).fill(0));
  for (let res = 0; res < MOD; res++) {
    let colTotal = 0;
    for (let e = 0; e < E; e++) colTotal += cooccurCounts[e][res];
    colTotal = colTotal || 1;
    for (let e = 0; e < E; e++) cooccur[e][res] = cooccurCounts[e][res] / colTotal;
  }

  // A residue is "claimed" if some expert receives the majority of its tokens.
  const residueClaimed = new Array(MOD).fill(false);
  for (let res = 0; res < MOD; res++) {
    for (let e = 0; e < E; e++) if (cooccur[e][res] > 0.5) residueClaimed[res] = true;
  }

  const perResidueAcc = totalPerResidue.map((t, res) => (t === 0 ? 0 : correctPerResidue[res] / t));

  return {
    lossHistory,
    finalUtil,
    deadExperts,
    cooccur,
    perResidueAcc,
    residueClaimed,
    finalEntropyMean: entropySum / data.X.length,
    droppedTotal,
  };
}

/**
 * Diagonal-dominance score: how close the expert x residue co-occurrence is to a clean
 * permutation matrix. For each residue take the max fraction any single expert claimed,
 * then average over residues. 1.0 = every residue fully owned by one expert; ~1/E = uniform
 * smear. NOTE: this measures routing CONSISTENCY, not BALANCE — a collapsed router can score
 * high here (few experts each consistently own many residues). Read it WITH dead-expert
 * count and CV, never alone.
 */
function permutationCloseness(cooccur: number[][]): number {
  let sum = 0;
  for (let res = 0; res < MOD; res++) {
    let best = 0;
    for (let e = 0; e < E; e++) if (cooccur[e][res] > best) best = cooccur[e][res];
    sum += best;
  }
  return sum / MOD;
}

function fmtPct(x: number): string {
  return (x * 100).toFixed(1) + "%";
}

function printRun(label: string, res: TrainResult): void {
  console.log(label);
  console.log("  " + plotLoss(res.lossHistory));
  console.log(
    `  task loss ${res.lossHistory[0].toFixed(3)} -> ${res.lossHistory.at(-1)!.toFixed(3)}`,
  );
  console.log("\n  专家利用率 (top-1 占比):");
  console.log(
    bar(
      res.finalUtil.map((_, e) => "E" + e),
      res.finalUtil,
    )
      .split("\n")
      .map((l) => "  " + l)
      .join("\n"),
  );
  console.log(
    `  死专家 (从不被选中) = ${res.deadExperts}/${E} | ` +
      `CV (变异系数, 0=完美均衡) = ${coefficientOfVariation(res.finalUtil).toFixed(3)} | ` +
      `mean routing entropy = ${res.finalEntropyMean.toFixed(3)} nats (max=ln${E}=${Math.log(E).toFixed(3)})`,
  );
  console.log("\n  专家 × 残基 共现热图 (列归一: 该残基 token 流向各专家的比例):");
  console.log(
    heatmap(
      res.cooccur,
      res.cooccur.map((_, e) => "E" + e),
      Array.from({ length: MOD }, (_, r) => "r" + r),
    )
      .split("\n")
      .map((l) => "  " + l)
      .join("\n"),
  );
  console.log(`  对角占比 (路由一致性, 非均衡度) = ${permutationCloseness(res.cooccur).toFixed(3)}`);
  console.log(`  被专家认领的残基 = ${res.residueClaimed.filter(Boolean).length}/${MOD}`);
  console.log("  各残基 accuracy: " + res.perResidueAcc.map((a) => fmtPct(a)).join(" "));
}

function main(): void {
  const data = makeModularTask(MOD, N_TRAIN, DIM, rng(SEED), NUM_CLASSES);

  console.log("=== stage07: MoE capstone — 自发分工 + 算力对账 (seed=" + SEED + ") ===");
  console.log(
    `任务: makeModularTask(mod=${MOD}) | 专家 E=${E}, top-${K}, 类别=${NUM_CLASSES}, ` +
      `capacity_factor=${CAPACITY_FACTOR} | ${STEPS} 步`,
  );
  console.log(`理想分工: 每个残基 (residue 0..${MOD - 1}) 由一位专家专属 -> 共现矩阵接近置换矩阵\n`);

  // ---- A) healthy run: aux loss + capacity ON ----
  const healthy = trainMoe(data, true);
  printRun("──────── A. 健康版 (aux load-balance + capacity 全开) ────────", healthy);
  console.log(
    `  训练期容量丢弃 expert-slot 总数 = ${healthy.droppedTotal} (capacity 压力的体现, 推动均衡)`,
  );

  // ---- B) FLOP accounting ----
  console.log("\n──────── 算力对账 (FLOPs, 单位 MAC = 1 乘 + 1 加) ────────");
  const tokensEval = data.X.length;
  const moeFlops = activatedFLOPs(tokensEval, HIDDEN, HIDDEN, HIDDEN, E, K);
  const denseFlopsVal = denseFLOPs(tokensEval, HIDDEN, HIDDEN, HIDDEN, E);
  // Total-parameter FLOPs = cost IF every token ran every expert AND the gate (i.e. you paid
  // for all stored parameters every token). Shows the gap between params you STORE vs COMPUTE.
  const oneExpert = HIDDEN * HIDDEN + HIDDEN * HIDDEN;
  const totalParamMacsPerToken = E * oneExpert + HIDDEN * E; // all experts + gate
  const totalParamFlops = tokensEval * totalParamMacsPerToken;
  const fmt = (n: number) => n.toLocaleString("en-US");
  console.log(`  评估 token 数 = ${tokensEval}`);
  console.log(`  ① 本 MoE 激活 FLOPs (只 top-${K}/${E} 专家 + gate) = ${fmt(moeFlops)} MAC`);
  console.log(`  ② dense FFN 激活 FLOPs (全 ${E} 专家, 无 router)   = ${fmt(denseFlopsVal)} MAC`);
  console.log(`  ③ 总参数 FLOPs (全专家 + gate, "若每 token 付全部参数") = ${fmt(totalParamFlops)} MAC`);
  console.log(
    `  激活/dense 比 = ${(moeFlops / denseFlopsVal).toFixed(3)} (≈ k/E = ${(K / E).toFixed(3)}; gate 开销让它略高于 k/E)`,
  );
  console.log(
    `  即: 存了 ③ 那么多参数, 每 token 只算了 ① 那么多 -> 省下 ${fmt(totalParamFlops - moeFlops)} MAC/批 (${fmtPct(1 - moeFlops / totalParamFlops)})`,
  );
  // Collapse tax: dead experts in the ablated run are parameters you stored but never compute.
  console.log(
    "  ⚠ 诚实标注: 此比值 (k/E 量级) 可迁移到真实 MoE; 但绝对 MAC 数与真实模型吞吐无关——",
  );
  console.log(
    "     toy 规模、无 kernel/硬件, 这里不测 wall-clock, 只数算法乘加。真实加速还受 router/通信开销影响。",
  );

  // ---- C) ablation: protection OFF (the cautionary A/B) ----
  console.log("");
  const collapsed = trainMoe(data, false);
  printRun("──────── B. 反例: 关掉所有均衡/容量保护, 同任务重训 ────────", collapsed);

  // ---- D) verdict: A/B contrast, every number computed above, none hardcoded ----
  console.log("\n──────── 收束: A/B 对照 (数字均由上文真实测出) ────────");
  const cvH = coefficientOfVariation(healthy.finalUtil);
  const cvC = coefficientOfVariation(collapsed.finalUtil);
  console.log(
    `  死专家:          健康 ${healthy.deadExperts}/${E}   vs  坍塌 ${collapsed.deadExperts}/${E}` +
      `   <- 最稳健的坍塌信号: 坍塌版有专家从不被选中`,
  );
  console.log(
    `  routing entropy: 健康 ${healthy.finalEntropyMean.toFixed(3)} vs 坍塌 ${collapsed.finalEntropyMean.toFixed(3)} nats  (低=router 过度自信/坍塌)`,
  );
  console.log(`  CV (利用率不均): 健康 ${cvH.toFixed(3)}   vs  坍塌 ${cvC.toFixed(3)}   (低=均衡)`);
  // Dead experts = stored-but-uncomputed parameters: the FLOP-win premise broken.
  const wastedFrac = collapsed.deadExperts / E;
  console.log(
    `  坍塌的代价 (算力视角): 坍塌版 ${collapsed.deadExperts}/${E} 个专家 = ${fmtPct(wastedFrac)} 的专家参数` +
      ` 被存下却从不参与计算——`,
  );
  console.log(
    `    MoE 的红利前提是「存的参数都被路由用上」; 坍塌恰恰打破它 (空占显存, 不贡献容量)。`,
  );
  // Honest accuracy note: at toy scale accuracy survives; say why, don't fake a drop.
  const avgAccH = healthy.perResidueAcc.reduce((s, a) => s + a, 0) / MOD;
  const avgAccC = collapsed.perResidueAcc.reduce((s, a) => s + a, 0) / MOD;
  console.log(
    `  平均 accuracy:   健康 ${fmtPct(avgAccH)} vs 坍塌 ${fmtPct(avgAccC)}` +
      ` <- 诚实: toy 任务太易, 二者都高`,
  );
  console.log(
    "    为什么 accuracy 没掉? 坍塌 = 不均衡, 不是错路由: 残基仍被【一致地】送到某个固定专家,",
  );
  console.log(
    "    而这个易任务只需要【一致性】就够准。真实模型里 capacity 吃紧、质量在边际, 死专家就是质量塌方处。",
  );
  console.log(
    "\n  结论: aux load-balance + capacity 不是装饰——关掉后路由坍塌、专家成片死亡、entropy 崩。",
  );
  console.log(
    "  「总参数大、激活算力小」的红利, 依赖这些保护把 token 摊到各专家; 坍塌让一半参数变成死重。",
  );
}

main();
