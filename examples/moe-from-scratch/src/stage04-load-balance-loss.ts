// stage04-load-balance-loss.ts — Chapter 4: the load-balance loss, the thing that stops a
//   gate from dumping every token into a handful of experts.
//
// WHY this chapter exists: a top-k router left alone tends to COLLAPSE. The first expert
//   that gets slightly better gets picked slightly more, trains slightly faster, gets
//   picked even more — a rich-get-richer loop. The starved experts never see gradient and
//   stay dead, so capacity you paid for in parameters is wasted. The load-balance auxiliary
//   loss (Switch/GShard) and the bias trick (auxiliary-loss-free / DeepSeek-style) are the
//   two production answers. This stage trains the SAME router four ways and measures the
//   difference with the book's instruments (expertUtilization / CV / routingEntropy).
//
// WHAT we actually prove here (every number is computed below, none hard-coded):
//   1. no-balance      -> high utilization CV (a few experts hog tokens, others starve)
//   2. aux loss (0.01) -> CV drops sharply, main-task accuracy unchanged (balance ~free)
//   3. bias-based      -> CV drops too, WITHOUT adding a term to the loss (the gradient
//                         never sees balancing; only the routing DECISION is nudged)
//   4. entropy-penalty -> the TRUE over-balancing knob: a term that maximizes per-token gate
//                         entropy, driving routing toward uniform/random. Watch entropy climb
//                         toward ln(E).
//
// HONEST RESULT, and the real lesson of this chapter — read carefully:
//   The Switch/GShard aux loss constrains the MEAN load per expert, NOT the per-token
//   decision. A router can satisfy "uniform average load" while still sending each token to
//   the right expert (different tokens to different experts, averaging to uniform). So even
//   at weight 1.0 the aux loss does NOT randomize routing and does NOT tank accuracy on this
//   task. Only an explicit ENTROPY penalty forces per-token routing toward random.
//
//   AND on this clean, separable toy, EVEN forced-uniform routing keeps accuracy high —
//   because each expert here emits full class logits, so when every expert sees every
//   cluster (uniform routing) each becomes a competent generalist and the combine is fine.
//   The textbook "over-balancing wrecks accuracy" needs a task where uniform load and good
//   routing are in TENSION (heterogeneous, specialization-dependent tokens — real LLM data).
//   Toy Gaussian blobs have no such tension, so we DEMONSTRATE the routing change (entropy
//   up, specialization purity down) and state plainly that the accuracy cost is the part
//   that does not transfer down to this toy. Faking an accuracy drop here would be a lie.
//
// HONESTY: makeClusters is toy, separable, tiny. Absolute accuracy is optimistic. What
//   TRANSFERS is the relative story: CV ordering across conditions, that aux-loss balancing
//   is near-free, that the aux loss equalizes mean-load not per-token routing, and that the
//   entropy penalty is what actually randomizes routing (entropy -> ln E, purity collapses).
//
// INVARIANT: all four conditions share the SAME data seed and the SAME init seed, so any
//   difference between them is attributable ONLY to the balancing mechanism, not to luck.
//   We rebuild the model from the same INIT_SEED before each condition so condition order
//   cannot leak through PRNG state.

import { rng, type Rng } from "./core/prng.js";
import { Value } from "./core/tensor.js";
import { Linear, Expert, crossEntropy } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { makeClusters, type Dataset } from "./core/data.js";
import {
  expertUtilization,
  coefficientOfVariation,
  routingEntropy,
  loadBalanceLoss,
} from "./core/metrics.js";
import { bar, plotLoss } from "./core/viz.js";

// ---- experiment constants -------------------------------------------------
// E=8 experts, top-2 routing, k=8 clusters: the smallest setup where collapse is visible AND
// a perfectly specialized router could map one cluster -> one expert. Whether it does is
// exactly the question the balancing mechanisms move.
const E = 8;
const K = 2; // top-k
const DIM = 6;
const HIDDEN = 3; // small but enough for separable blobs; keeps the model fast and honest
const NUM_CLASSES = 8; // == k clusters; expert.outDim == NUM_CLASSES so experts emit logits
const N_TRAIN = 240;
const STEPS = 400;
const LR = 0.02;
const SPREAD = 0.6; // separable blobs (small spread vs center separation ~10)
const DATA_SEED = 7;
const INIT_SEED = 101; // re-applied before every condition so init is identical across them

/**
 * One MoE classifier: a linear gate over DIM -> E logits, E experts each mapping DIM ->
 * NUM_CLASSES logits, and top-K sparse combination weighted by the (renormalized) gate
 * probabilities of the chosen experts. There is deliberately NO separate classifier head:
 * experts emit class logits directly so that "which expert handled this token" and "what the
 * model predicted" run through the same path — that is what makes utilization a load-bearing
 * number rather than decoration.
 *
 * `gateBias` is the bias-trick knob: a per-expert additive bias applied to gate LOGITS at
 * ROUTING (top-k selection) time only. It is NOT a learned parameter and never appears in
 * the loss. For every other condition it stays all-zero (no effect).
 */
class MoEClassifier {
  gate: Linear;
  experts: Expert[];
  // Per-expert routing bias (the auxiliary-loss-free trick). NOT a Value: never receives
  // gradient. We adjust it by hand toward balance. All-zero => no effect.
  gateBias: Float64Array;

  constructor(r: Rng) {
    this.gate = new Linear(DIM, E, r, true);
    this.experts = [];
    for (let e = 0; e < E; e++) this.experts.push(new Expert(DIM, HIDDEN, NUM_CLASSES, r, "gelu"));
    this.gateBias = new Float64Array(E); // zeros
  }

  /**
   * Forward one example. Returns the differentiable class logits, the full gate prob row
   * (for the aux/entropy terms), and the top-1 expert id (the hard count for utilization).
   *
   * INVARIANT: selection uses gate logits PLUS gateBias (so the bias trick can steer load),
   *   but the COMBINING weights use the UNBIASED gate softmax restricted to the chosen
   *   experts. WHY: the bias is a load-steering hint, not a value judgement — letting it into
   *   the combine weights would corrupt the model output, defeating the whole point of the
   *   trick (fix balance WITHOUT touching what the gate "believes").
   */
  forward(x: Value): { logits: Value; gateProbs: Value; top1: number } {
    const gateLogits = this.gate.forward(x); // [1,E]
    const gateProbs = gateLogits.softmaxRow(); // [1,E], differentiable

    // Top-K selection on biased logits. We read raw numbers for the argsort (selection is
    // non-differentiable — the hard routing decision), then rebuild the differentiable
    // combine from gateProbs.gather on the chosen indices.
    const scores: { e: number; s: number }[] = [];
    for (let e = 0; e < E; e++) scores.push({ e, s: gateLogits.data[e] + this.gateBias[e] });
    scores.sort((a, b) => b.s - a.s);
    const chosen = scores.slice(0, K).map((c) => c.e);

    // Renormalize the chosen experts' UNBIASED gate probs so combine weights sum to 1.
    let denom = 0;
    for (const e of chosen) denom += gateProbs.data[e];
    if (denom < 1e-12) denom = 1e-12; // guard: degenerate all-near-zero row

    let combined: Value | null = null;
    for (const e of chosen) {
      const w = gateProbs.gather(e).mul(Value.scalar(1 / denom)); // scalar weight, grad flows
      const out = this.experts[e].forward(x).mul(w); // [1,NUM_CLASSES]
      combined = combined === null ? out : combined.add(out);
    }
    const top1 = chosen[0]; // the expert that "owns" this token for utilization
    return { logits: combined as Value, gateProbs, top1 };
  }

  parameters(): Value[] {
    return [...this.gate.parameters(), ...this.experts.flatMap((e) => e.parameters())];
  }
}

type Balance =
  | { kind: "none" }
  | { kind: "aux"; weight: number } // Switch/GShard load-balance aux loss (equalizes MEAN load)
  | { kind: "bias"; rate: number } // auxiliary-loss-free per-expert routing bias
  | { kind: "entropy"; weight: number }; // explicit per-token entropy MAX -> forces uniform routing

interface TrainResult {
  label: string;
  util: number[]; // final top-1 utilization per expert (sums to 1)
  cv: number; // coefficient of variation of util (0 = perfectly balanced)
  entropy: number; // mean routing entropy over the final pass (nats; max = ln E)
  accuracy: number; // main-task accuracy after training
  purity: number; // mean dominant-cluster share per expert (1 = each owns one cluster)
  taskLossCurve: number[]; // per-step main cross-entropy loss
  auxLossCurve: number[]; // per-step load-balance aux value (1.0 = ideal; only set for "aux")
}

/**
 * Differentiable Shannon entropy of a [1,E] probability row: -sum p*log p. Used as the
 * over-balancing penalty (we ADD -weight*entropy so minimizing the loss MAXIMIZES entropy,
 * pushing the per-token route toward uniform). WHY a separate term and not the aux loss:
 * the aux loss only equalizes the MEAN load and cannot, by construction, randomize a single
 * token's route — this term can, which is exactly the failure we want to exhibit.
 */
function rowEntropy(probs: Value): Value {
  return probs.mul(probs.log()).sum().mul(Value.scalar(-1));
}

/**
 * Train one condition. Returns measured (not estimated) metrics. Data and init seeds are
 * fixed by the caller so the ONLY independent variable is `balance`.
 */
function trainCondition(label: string, data: Dataset, balance: Balance): TrainResult {
  const model = new MoEClassifier(rng(INIT_SEED));
  const opt = new Adam(model.parameters(), LR);

  const taskLossCurve: number[] = [];
  const auxLossCurve: number[] = [];
  const order = data.X.map((_, i) => i);
  // Dedicated shuffle stream: batch order is reproducible but independent of init draws.
  const shuffleRng = rng(INIT_SEED ^ 0x9e3779b9);

  for (let step = 0; step < STEPS; step++) {
    shuffleRng.shuffle(order);
    opt.zeroGrad();

    // Full-batch per step (small data -> cheap, and a stable per-step utilization count,
    // which the bias rule needs to read).
    const gateProbsPerToken: Value[] = [];
    const assignments: number[] = [];
    let taskLoss: Value | null = null;
    let entropyAcc: Value | null = null;

    for (const i of order) {
      const x = Value.from(data.X[i], [1, DIM]);
      const { logits, gateProbs, top1 } = model.forward(x);
      const ce = crossEntropy(logits, data.Y[i]);
      taskLoss = taskLoss === null ? ce : taskLoss.add(ce);
      if (balance.kind === "entropy") {
        const h = rowEntropy(gateProbs);
        entropyAcc = entropyAcc === null ? h : entropyAcc.add(h);
      }
      gateProbsPerToken.push(gateProbs);
      assignments.push(top1);
    }
    const meanTaskLoss = (taskLoss as Value).mul(Value.scalar(1 / order.length));

    // Total loss. Only "aux" and "entropy" add a term; "none"/"bias" leave the loss alone.
    let total = meanTaskLoss;
    let auxValue = 1.0; // perfectly-balanced reference; overwritten if we actually compute it
    if (balance.kind === "aux") {
      const aux = loadBalanceLoss(gateProbsPerToken, assignments, E); // scalar Value, grad into probs
      auxValue = aux.data[0];
      total = total.add(aux.mul(Value.scalar(balance.weight)));
    } else if (balance.kind === "entropy") {
      // ADD -weight * mean entropy: gradient descent then MAXIMIZES entropy (uniform routing).
      const meanEnt = (entropyAcc as Value).mul(Value.scalar(1 / order.length));
      total = total.add(meanEnt.mul(Value.scalar(-balance.weight)));
    }

    total.backward();
    opt.step();

    taskLossCurve.push(meanTaskLoss.data[0]);
    auxLossCurve.push(auxValue);

    // Bias-trick update (only for "bias"): nudge each expert's routing bias DOWN when it is
    // overloaded and UP when underloaded, proportional to its distance from uniform load.
    // The optimizer's gradient never sees this — gateBias is not a parameter. Sign: load >
    // 1/E (overloaded) -> lower its bias so it wins fewer top-k slots next step.
    if (balance.kind === "bias") {
      const util = expertUtilization(assignments, E);
      const target = 1 / E;
      for (let e = 0; e < E; e++) model.gateBias[e] += balance.rate * (target - util[e]);
    }
  }

  // Final measurement pass (no training): END-state utilization, entropy, accuracy, purity.
  const finalAssign: number[] = [];
  let entropySum = 0;
  let correct = 0;
  for (let i = 0; i < data.X.length; i++) {
    const x = Value.from(data.X[i], [1, DIM]);
    const { logits, gateProbs, top1 } = model.forward(x);
    finalAssign.push(top1);
    entropySum += routingEntropy(Array.from(gateProbs.data));
    let best = 0;
    for (let c = 1; c < NUM_CLASSES; c++) if (logits.data[c] > logits.data[best]) best = c;
    if (best === data.Y[i]) correct++;
  }
  const util = expertUtilization(finalAssign, E);
  return {
    label,
    util,
    cv: coefficientOfVariation(util),
    entropy: entropySum / data.X.length,
    accuracy: correct / data.X.length,
    purity: specializationPurity(finalAssign, data.clusterId),
    taskLossCurve,
    auxLossCurve,
  };
}

/**
 * Specialization purity: for each expert, the share of its assigned tokens that belong to
 * its single dominant cluster, averaged over experts that received any tokens. 1.0 means
 * every expert owns exactly one cluster (perfect specialization); 1/k means an expert's
 * tokens are spread evenly across clusters (no specialization — what over-balancing trends
 * toward). This is the direct "did experts specialize by cluster?" measurement.
 */
function specializationPurity(assignments: number[], clusterId: number[]): number {
  const perExpert: Map<number, number>[] = Array.from({ length: E }, () => new Map());
  const counts = new Array(E).fill(0);
  for (let i = 0; i < assignments.length; i++) {
    const e = assignments[i];
    const cl = clusterId[i];
    perExpert[e].set(cl, (perExpert[e].get(cl) ?? 0) + 1);
    counts[e]++;
  }
  let sum = 0;
  let n = 0;
  for (let e = 0; e < E; e++) {
    if (counts[e] === 0) continue;
    const dominant = Math.max(...perExpert[e].values());
    sum += dominant / counts[e];
    n++;
  }
  return n ? sum / n : 0;
}

function expertLabels(): string[] {
  return Array.from({ length: E }, (_, e) => `E${e}`);
}

function main(): void {
  const data = makeClusters(NUM_CLASSES, N_TRAIN, DIM, rng(DATA_SEED), SPREAD);
  const maxEntropy = Math.log(E);

  console.log("=== stage04: load-balance loss (top-2, E=8, k=8 clusters) ===");
  console.log(
    `data: ${N_TRAIN} points, ${NUM_CLASSES} separable clusters in ${DIM}-d (spread=${SPREAD}) | ` +
      `model: ${E} experts, top-${K}, hidden=${HIDDEN}, ${STEPS} steps`,
  );
  console.log(`max routing entropy = ln(${E}) = ${maxEntropy.toFixed(4)} nats (fully uniform routing)`);
  console.log("注意: 这是 toy 数据, accuracy / 绝对值偏乐观; 可迁移的是各组之间的相对差距与趋势。\n");

  const conditions: { label: string; balance: Balance }[] = [
    { label: "no-balance     ", balance: { kind: "none" } },
    { label: "aux loss (0.01)", balance: { kind: "aux", weight: 0.01 } },
    { label: "bias-based     ", balance: { kind: "bias", rate: 0.1 } },
    { label: "aux loss (1.00)", balance: { kind: "aux", weight: 1.0 } }, // strong load-balance
    { label: "entropy-pen 0.3", balance: { kind: "entropy", weight: 0.3 } }, // forces random routing
  ];

  const results = conditions.map((c) => trainCondition(c.label.trim(), data, c.balance));
  const by = (name: string) => results.find((r) => r.label === name)!;

  // ---- (1) utilization bars + CV per condition ----
  console.log("--- (1) expert utilization (top-1 share, sums to 1) + CV ---");
  for (const r of results) {
    console.log(`\n[${r.label}]  CV=${r.cv.toFixed(3)}  (0=perfectly balanced)`);
    console.log(bar(expertLabels(), r.util));
  }

  // ---- (2) main-task accuracy (healthy balancing must not hurt it) ----
  console.log("\n--- (2) main-task accuracy (健康的均衡几乎不伤性能) ---");
  for (const r of results) {
    console.log(
      `  ${r.label}  acc=${(r.accuracy * 100).toFixed(1)}%   ` +
        `final task loss=${r.taskLossCurve.at(-1)!.toFixed(4)}`,
    );
  }

  // ---- (3) load-balance aux value over steps (only the aux conditions move it) ----
  console.log("\n--- (3) load-balance aux value over steps (1.0 = ideal balance; >1 = imbalanced) ---");
  for (const r of results) {
    if (!r.label.startsWith("aux")) continue;
    console.log(`\n[${r.label}]  aux ${r.auxLossCurve[0].toFixed(3)} -> ${r.auxLossCurve.at(-1)!.toFixed(3)}`);
    console.log("  " + plotLoss(r.auxLossCurve));
  }

  // ---- (4) routing entropy + specialization purity: what balancing does to the ROUTER ----
  console.log("\n--- (4) routing entropy (decisiveness) + specialization purity (cluster->expert) ---");
  for (const r of results) {
    const entPct = ((r.entropy / maxEntropy) * 100).toFixed(0);
    console.log(
      `  ${r.label}  entropy=${r.entropy.toFixed(3)} (${entPct}% of max)   ` +
        `purity=${(r.purity * 100).toFixed(1)}%`,
    );
  }

  // ---- (5) the honest verdict on over-balancing, derived from the measured numbers ----
  console.log("\n--- (5) 解读: 均衡是怎么影响路由的 (诚实结论) ---");
  const none = by("no-balance");
  const aux01 = by("aux loss (0.01)");
  const aux1 = by("aux loss (1.00)");
  const bias = by("bias-based");
  const entp = by("entropy-pen 0.3");

  // Derived booleans — each is a real comparison of measured quantities.
  const cvDropAux = aux01.cv < none.cv;
  const accKeptAux = Math.abs(aux01.accuracy - none.accuracy) <= 0.02;
  const cvDropBias = bias.cv < none.cv;
  const auxOnlyMeanLoad = aux1.entropy - aux01.entropy < 0.6; // strong aux did NOT randomize routing
  const entropyForced = entp.entropy > aux1.entropy + 0.5; // entropy penalty DID randomize routing
  const purityFell = entp.purity < none.purity; // and specialization degraded under it

  console.log(
    `  [1] aux(0.01): CV ${none.cv.toFixed(3)} -> ${aux01.cv.toFixed(3)} ` +
      `(${cvDropAux ? "下降, 死专家被激活" : "未下降"}), accuracy ` +
      `${(none.accuracy * 100).toFixed(1)}% -> ${(aux01.accuracy * 100).toFixed(1)}% ` +
      `(${accKeptAux ? "几乎不变 => 均衡近乎免费" : "明显变化"})。`,
  );
  console.log(
    `  [2] bias-based: CV ${none.cv.toFixed(3)} -> ${bias.cv.toFixed(3)} ` +
      `(${cvDropBias ? "也下降" : "未下降"}), 且没往 loss 里加任何项 (梯度看不到均衡)。`,
  );
  console.log(
    `  [3] 关键事实: aux 权重拉到 1.0, 熵只从 ${aux01.entropy.toFixed(3)} 到 ${aux1.entropy.toFixed(3)} ` +
      `(${auxOnlyMeanLoad ? "几乎没升" : "明显升"}), accuracy=${(aux1.accuracy * 100).toFixed(1)}%。`,
  );
  console.log(
    "      => Switch/GShard aux loss 只约束[平均负载], 不约束[单 token 路由]; " +
      "它能让每个 token 仍去对的专家、平均下来却均匀。所以它不会把路由变随机, 也不掉 accuracy。",
  );
  console.log(
    `  [4] 真正的"过度均衡": 显式最大化逐 token 熵 -> 熵冲到 ${entp.entropy.toFixed(3)} ` +
      `(${((entp.entropy / maxEntropy) * 100).toFixed(0)}% of ln E, ${entropyForced ? "确实逼近随机路由" : "未逼近随机"}), ` +
      `purity ${(none.purity * 100).toFixed(1)}% -> ${(entp.purity * 100).toFixed(1)}% ` +
      `(${purityFell ? "下降 => 专家不再按簇分工" : "未下降"})。`,
  );
  console.log(
    `      但在这个可分的 toy 上, 即便路由近乎随机 accuracy 仍 ${(entp.accuracy * 100).toFixed(1)}% — ` +
      "因为每个专家都看遍所有簇、各自学成了通才, 拼起来照样对。",
  );
  console.log(
    "  结论 (会迁移的部分): 均衡机制能压 CV、激活死专家, 基本不伤性能; 过度均衡的代价是路由变随机、" +
      "专家丧失分工。accuracy 真正回落需要[负载均匀]与[正确路由]冲突的任务 (真实 LLM token 异质、依赖分工), " +
      "干净高斯簇没有这种张力, 所以 accuracy 这一刀在 toy 上砍不出来 — 这点本 demo 不假装。",
  );
}

main();
