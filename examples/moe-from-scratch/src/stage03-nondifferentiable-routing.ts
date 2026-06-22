// stage03-nondifferentiable-routing.ts — Routing is NON-differentiable: how does gradient
//   get through the "take the max" wall?
//
// THE WALL: a top-k router picks experts by argmax over gate logits. argmax is a step
//   function: nudge a logit a hair and the chosen expert is unchanged (gradient 0) until
//   it crosses a neighbor and the choice SNAPS (gradient undefined). You cannot
//   backpropagate through "which expert was picked". Yet the gate net is just a Linear
//   whose weights we want to train. So where does its gradient come from?
//
// THE ANSWER this stage demonstrates: gradient does NOT flow through the discrete CHOICE.
//   It flows through the continuous gate PROBABILITY of the experts that were chosen — the
//   softmax weight each selected expert's output is multiplied by. The argmax only routes
//   WHICH path the graph takes; the magnitude on that path is a differentiable softmax
//   value, and that is the only thing the gate Linear learns from. Unselected experts get
//   exactly zero gradient (their path was never built into the graph this step).
//
// WHAT WE MEASURE (all numbers are really computed, not asserted):
//   A. gate-Linear weight grad-norm per step: hard routing vs noisy-top-k gating. Noise
//      smooths the decision boundary so grad norm is steadier (lower std, fewer snap spikes).
//      This effect HOLDS at this scale and is the chapter's transferable A/B result.
//   B. across-expert assignment entropy per step: the textbook claim is "noise => more
//      exploration => higher entropy". On these cleanly-separable toy clusters that claim
//      does NOT hold (printed honestly): the deterministic gate already finds the right
//      experts fast, so there's no local optimum for noise to escape. Noise's exploration
//      payoff appears on landscapes WITH traps (expert collapse, stage06), not here. We
//      report the negative result rather than tuning the task to manufacture the story.
//   C. micro-experiment on a SINGLE fixed input: print the grad of every expert's params
//      and prove selected != 0, unselected == 0 exactly.
//   D. FAILURE MODE: backprop through the argmax INDEX (a detached number) instead of the
//      gate probability. The gate Linear receives ~0 gradient, its weights freeze, and
//      utilization stays pinned to the random init preference. This is the concrete reason
//      "you must let the continuous weights carry the gradient".
//
// HONESTY: toy clusters, tiny nets => absolute losses/entropies are optimistic. What
//   transfers is the RELATIVE story (noisy vs hard grad-norm/entropy shape) and the EXACT
//   structural facts in B/D (zero vs nonzero grad), which are properties of the graph, not
//   of scale, so they hold verbatim at any size.

import { Value } from "./core/tensor.js";
import { Linear, Expert } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { rng, type Rng } from "./core/prng.js";
import { makeClusters, type Dataset } from "./core/data.js";
import { routingEntropy, expertUtilization } from "./core/metrics.js";
import { plotLoss, bar } from "./core/viz.js";

// ---- experiment configuration (small enough to converge in <1s/CPU, big enough to show
//      collapse-vs-balance and a meaningful grad-norm curve) ---------------------------
const DIM = 6;
const NUM_EXPERTS = 4;
const TOP_K = 2;
const HIDDEN = 8;
const NUM_TRAIN = 80;
const STEPS = 120;
const LR = 0.05;
// Noise std for the noisy-top-k variant. Sized RELATIVE to early gate-logit spread: too
// small and it never flips a top-k selection (the demo would show no difference); large
// enough here to actually perturb which experts win early, so exploration is visible.
const NOISE_STD = 1.5;

/** A top-k MoE layer reduced to exactly what this chapter needs: a gate Linear + E experts
 *  mapping DIM->DIM. Output is the gate-prob-weighted sum of the TOP_K selected experts.
 *  We keep the gate and experts as separate fields so the micro-experiments below can read
 *  each expert's .grad independently. */
class TopKMoE {
  gate: Linear;
  experts: Expert[];
  constructor(dim: number, numExperts: number, hidden: number, r: Rng) {
    this.gate = new Linear(dim, numExperts, r);
    this.experts = Array.from({ length: numExperts }, () => new Expert(dim, hidden, dim, r));
  }
  parameters(): Value[] {
    return [...this.gate.parameters(), ...this.experts.flatMap((e) => e.parameters())];
  }
}

/** Pick the indices of the k largest entries of a [1,E] row. Pure JS over .data — this is
 *  the NON-DIFFERENTIABLE step: the returned indices are plain numbers, detached from the
 *  graph. The gradient story is entirely about what we do with these indices afterward. */
function topKIndices(row: Value, k: number): number[] {
  const pairs = Array.from(row.data, (v, j) => ({ v, j }));
  pairs.sort((a, b) => b.v - a.v);
  return pairs.slice(0, k).map((p) => p.j);
}

/**
 * Forward one token through the MoE.
 *
 * routeMode === "prob"   : output = sum over selected experts of gateProb_e * expert_e(x).
 *   This is the CORRECT path. The argmax (topKIndices) only selects WHICH experts run; the
 *   multiplier on each is the differentiable softmax gate prob, so the gate Linear gets a
 *   real gradient through gather() of the chosen columns.
 *
 * routeMode === "index"  : output = sum over selected experts of (argmaxRank as a DETACHED
 *   scalar) * expert_e(x). The weight is built from the discrete index, NOT from the gate
 *   graph, so NO gradient reaches the gate Linear. This is the failure mode (D).
 *
 * `noiseStd > 0` adds Gaussian noise to the gate logits BEFORE top-k — the noisy-top-k
 *   trick. Noise is detached (a constant added via leaf), so it perturbs the SELECTION and
 *   the prob values but does not itself carry gradient.
 *
 * Returns the output row plus the softmax gate-prob row (for entropy/utilization stats).
 */
function forwardToken(
  moe: TopKMoE,
  x: Value,
  r: Rng,
  opts: { noiseStd: number; routeMode: "prob" | "index" },
): { out: Value; gateProbs: Value; selected: number[] } {
  const logits = moe.gate.forward(x); // [1,E]
  // Noise is added as a constant leaf: it shifts the logits seen by softmax+argmax but has
  // no _backward, so gradient flows only through the underlying logits (the real impl too).
  let noisy = logits;
  if (opts.noiseStd > 0) {
    const noise = new Float64Array(logits.cols);
    for (let j = 0; j < logits.cols; j++) noise[j] = r.normal() * opts.noiseStd;
    noisy = logits.add(Value.from(Array.from(noise), [1, logits.cols]));
  }
  const gateProbs = noisy.softmaxRow(); // [1,E], differentiable
  const selected = topKIndices(noisy, TOP_K); // detached indices — the non-diff step

  let out = Value.zeros(1, x.cols); // [1,DIM] accumulator leaf
  for (let rank = 0; rank < selected.length; rank++) {
    const e = selected[rank];
    const expertOut = moe.experts[e].forward(x); // [1,DIM]
    let weight: Value;
    if (opts.routeMode === "prob") {
      // Differentiable: the gate prob of the chosen expert carries gradient into the gate.
      weight = gateProbs.gather(e); // scalar, part of the graph
    } else {
      // FAILURE MODE: weight from the discrete rank, detached from the gate graph.
      // (1/(rank+1) is an arbitrary monotone-by-rank scalar; the point is it's a plain
      //  number with no path back to gate.W, so gate.W.grad stays ~0.)
      weight = Value.scalar(1 / (rank + 1));
    }
    out = out.add(expertOut.mul(weight));
  }
  return { out, gateProbs, selected };
}

/** L2 norm of a parameter's grad buffer — our proxy for "how strong a learning signal did
 *  this tensor receive this step". Reads .grad AFTER backward(), BEFORE zeroGrad(). */
function gradL2(p: Value): number {
  let s = 0;
  for (let k = 0; k < p.grad.length; k++) s += p.grad[k] * p.grad[k];
  return Math.sqrt(s);
}

/** Mean-squared reconstruction loss of a [1,DIM] output against a [1,DIM] target row. A
 *  trivial objective; this chapter cares about WHERE gradient goes, not WHAT it optimizes. */
function reconLoss(out: Value, target: Value): Value {
  return out.sub(target).pow(2).mean();
}

/** One full training run, returning per-step diagnostics. Each step processes the whole
 *  toy set token-by-token (no batching op in the engine), accumulating loss and the
 *  load-balance bookkeeping, then does one Adam step. */
function train(
  data: Dataset,
  seed: number,
  opts: { noiseStd: number; routeMode: "prob" | "index" },
): { lossCurve: number[]; gateGradNormCurve: number[]; entropyCurve: number[]; finalUtil: number[] } {
  const r = rng(seed);
  const moe = new TopKMoE(DIM, NUM_EXPERTS, HIDDEN, r);
  const opt = new Adam(moe.parameters(), LR);
  // Targets: map each cluster to a fixed point so experts have something to specialize on.
  const targets = data.X.map((_, i) => {
    const t = new Array(DIM).fill(0);
    t[data.clusterId[i] % DIM] = 1; // one-hot-ish target per cluster
    return Value.from(t, [1, DIM]);
  });

  const lossCurve: number[] = [];
  const gateGradNormCurve: number[] = [];
  const entropyCurve: number[] = [];
  let finalUtil: number[] = [];

  for (let step = 0; step < STEPS; step++) {
    opt.zeroGrad();
    let totalLoss = Value.scalar(0);
    const assignments: number[] = [];

    for (let i = 0; i < data.X.length; i++) {
      const x = Value.from(data.X[i], [1, DIM]);
      const { out, selected } = forwardToken(moe, x, r, opts);
      totalLoss = totalLoss.add(reconLoss(out, targets[i]));
      assignments.push(selected[0]); // top-1 for utilization
    }
    const meanLoss = totalLoss.mul(Value.scalar(1 / data.X.length));
    meanLoss.backward(); // accumulates grads across all tokens (engine sums, no double-count
    //                       because we zeroGrad'd at the top of THIS step)

    // Snapshot gate-Linear weight grad norm BEFORE the optimizer consumes it.
    gateGradNormCurve.push(gradL2(moe.gate.W));
    lossCurve.push(meanLoss.data[0]);
    // Exploration metric: entropy of the ACROSS-EXPERT top-1 assignment distribution this
    // step (nats, max=ln E). High = tokens spread over many experts (router exploring);
    // collapsing toward 0 = the router commits to a few experts. This is a more honest
    // "exploration" signal than per-token softmax entropy, which a confident gate drives to
    // ~0 regardless of how many distinct experts are actually used.
    entropyCurve.push(routingEntropy(expertUtilization(assignments, NUM_EXPERTS)));

    opt.step();
    if (step === STEPS - 1) finalUtil = expertUtilization(assignments, NUM_EXPERTS);
  }
  return { lossCurve, gateGradNormCurve, entropyCurve, finalUtil };
}

/** Mean of a numeric array (small helper to keep printouts honest about "average over the
 *  whole curve" rather than cherry-picking a step). */
function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}

/** Standard deviation — used as a "spikiness" proxy for the grad-norm curves. A jagged,
 *  snap-driven curve has higher std than a smooth one. */
function std(xs: number[]): number {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) * (x - m))));
}

function main(): void {
  console.log("=== stage03: 路由不可微 — 梯度如何穿过『取最大』这道墙 (seed=7) ===\n");
  const data = makeClusters(NUM_EXPERTS, NUM_TRAIN, DIM, rng(7), 0.6);

  // ---- A/B: hard routing vs noisy top-k gating ------------------------------------------
  console.log("[A] 硬路由(无噪声) vs noisy top-k gating — 门控 Linear 权重梯度范数 & 路由熵\n");
  const hard = train(data, 7, { noiseStd: 0, routeMode: "prob" });
  const noisy = train(data, 7, { noiseStd: NOISE_STD, routeMode: "prob" });

  console.log("门控权重梯度范数曲线 (gate.W grad L2 per step):");
  console.log("  hard :", plotLoss(hard.gateGradNormCurve, 50));
  console.log("  noisy:", plotLoss(noisy.gateGradNormCurve, 50));
  console.log(
    `  抖动(std of grad-norm): hard=${std(hard.gateGradNormCurve).toFixed(4)}  ` +
      `noisy=${std(noisy.gateGradNormCurve).toFixed(4)}  ` +
      `→ noisy ${std(noisy.gateGradNormCurve) < std(hard.gateGradNormCurve) ? "更平滑(更小)" : "未更平滑"}`,
  );
  console.log();

  console.log("路由熵曲线 (跨专家 top-1 分配分布的熵 per step, nats; max=ln4=1.3863):");
  console.log("  hard :", plotLoss(hard.entropyCurve, 50));
  console.log("  noisy:", plotLoss(noisy.entropyCurve, 50));
  const peakHard = Math.max(...hard.entropyCurve);
  const peakNoisy = Math.max(...noisy.entropyCurve);
  console.log(
    `  探索峰值熵(max over steps): hard=${peakHard.toFixed(4)}  noisy=${peakNoisy.toFixed(4)}  ` +
      `→ noisy ${peakNoisy > peakHard ? "峰值更高(探索更广)" : "峰值未更高(见下方诚实标注)"}`,
  );
  console.log(`  收敛 loss: hard=${hard.lossCurve.at(-1)!.toFixed(4)}  noisy=${noisy.lossCurve.at(-1)!.toFixed(4)}`);
  console.log("  最终专家利用率(top-1):");
  console.log(bar(["hard E0", "E1", "E2", "E3"], hard.finalUtil));
  console.log(bar(["noisyE0", "E1", "E2", "E3"], noisy.finalUtil));
  const sameFinal = hard.finalUtil.every((u, j) => Math.abs(u - noisy.finalUtil[j]) < 1e-9);
  const gradSmoother = std(noisy.gateGradNormCurve) < std(hard.gateGradNormCurve);
  // Report exactly what THIS run shows, including the negative result. Forcing the textbook
  // "noise => more exploration" story onto cleanly-separable toy clusters would be dishonest:
  // the deterministic gate already nails the right experts fast, so there's no bad local
  // optimum for noise to escape. Noise's exploration payoff is real, but it surfaces on
  // landscapes WITH traps (e.g. expert collapse, stage06), not on this easy task.
  console.log("  诚实标注(toy, 单 seed=7):");
  console.log(`   • 梯度更平滑: ${gradSmoother ? "成立 ✓ (noisy std 更小)" : "不成立"} — 这是噪声此处唯一稳健的可迁移收益。`);
  console.log(`   • 探索更广: 本任务 ${peakNoisy > peakHard ? "成立" : "不成立 ✗"} — 干净可分的 clusters 没有局部最优陷阱,`);
  console.log("     确定性门控已快速找对专家,噪声无处可探。探索收益要在『有坍塌陷阱』的任务上才显现(见 stage06)。");
  console.log(`   • 终态利用率 hard 与 noisy ${sameFinal ? "完全一致" : "不同"}: 收敛后门控足够自信,噪声不再翻转 top-2。`);
  console.log("   • 绝对数(loss/熵)偏乐观,可迁移的是相对趋势,不是绝对值。");
  console.log();

  // ---- C: micro-experiment — gradient ONLY travels the selected path --------------------
  console.log("[C] 数值小实验: 固定输入,验证『梯度只走被选专家』\n");
  microGradientPathExperiment();

  // ---- D: failure mode — backprop through the argmax index, gate freezes -----------------
  console.log("[D] 失败模式: 对离散 argmax 索引求梯度 → 门控网络几乎不更新\n");
  failureModeArgmaxBackprop(data);
}

/** C: build ONE forward graph on a single fixed input, backprop, and read every expert's
 *  fc1.W grad. The selected experts (top-2 by gate logit) must have nonzero grad; the rest
 *  must be EXACTLY zero — because their forward was never added into the graph this step,
 *  so backward() never visits their _backward. This is the load-bearing claim of the
 *  chapter, proven by reading the buffers rather than asserting it. */
function microGradientPathExperiment(): void {
  const r = rng(123);
  const moe = new TopKMoE(DIM, NUM_EXPERTS, HIDDEN, r);
  const x = Value.from(
    Array.from({ length: DIM }, () => r.normal()),
    [1, DIM],
  );
  const target = Value.from(new Array(DIM).fill(0.3), [1, DIM]);

  // Zero all grads first so we read THIS step's contribution only.
  for (const p of moe.parameters()) p.grad.fill(0);
  const { out, selected } = forwardToken(moe, x, rng(0) /*no noise path*/, {
    noiseStd: 0,
    routeMode: "prob",
  });
  const loss = reconLoss(out, target);
  loss.backward();

  console.log(`  固定输入下被选中的 top-${TOP_K} 专家: [${selected.join(", ")}]`);
  const selectedSet = new Set(selected);
  for (let e = 0; e < NUM_EXPERTS; e++) {
    const g = gradL2(moe.experts[e].fc1.W);
    const picked = selectedSet.has(e);
    const verdict = picked
      ? g > 0
        ? "✓ 被选 → grad ≠ 0"
        : "✗ 被选却 grad=0 (BUG)"
      : g === 0
        ? "✓ 未选 → grad 恒为 0"
        : "✗ 未选却 grad≠0 (BUG)";
    console.log(`  expert ${e} fc1.W grad L2 = ${g.toExponential(3)}  [${verdict}]`);
  }
  // The gate itself must receive gradient (through the gather of the chosen probs).
  console.log(`  gate.W grad L2 = ${gradL2(moe.gate.W).toExponential(3)}  [门控自身有梯度: 通过被选概率的 gather]`);
  console.log();
}

/** D: train with routeMode="index" (weight comes from the detached argmax rank, not the
 *  gate prob). The gate Linear is structurally disconnected from the loss, so its grad
 *  norm stays ~0 and its weights never move — utilization stays frozen at whatever the
 *  random init happened to prefer. We print the gate grad norm (≈0) and show utilization
 *  before/after training is identical. This is WHY you must route the multiplier through
 *  the continuous gate prob (routeMode="prob"), not the discrete index. */
function failureModeArgmaxBackprop(data: Dataset): void {
  // Snapshot the gate weights before training to prove they don't move.
  const r = rng(7);
  const moe = new TopKMoE(DIM, NUM_EXPERTS, HIDDEN, r);
  const gateBefore = Float64Array.from(moe.gate.W.data);
  const opt = new Adam(moe.parameters(), LR);
  const targets = data.X.map((_, i) => {
    const t = new Array(DIM).fill(0);
    t[data.clusterId[i] % DIM] = 1;
    return Value.from(t, [1, DIM]);
  });

  // Utilization at init (step 0, before any update).
  let utilInit: number[] = [];
  let gateGradNorms: number[] = [];

  for (let step = 0; step < STEPS; step++) {
    opt.zeroGrad();
    let totalLoss = Value.scalar(0);
    const assignments: number[] = [];
    for (let i = 0; i < data.X.length; i++) {
      const x = Value.from(data.X[i], [1, DIM]);
      const { out, selected } = forwardToken(moe, x, r, { noiseStd: 0, routeMode: "index" });
      totalLoss = totalLoss.add(reconLoss(out, targets[i]));
      assignments.push(selected[0]);
    }
    const meanLoss = totalLoss.mul(Value.scalar(1 / data.X.length));
    meanLoss.backward();
    gateGradNorms.push(gradL2(moe.gate.W));
    if (step === 0) utilInit = expertUtilization(assignments, NUM_EXPERTS);
    opt.step();
  }

  // Utilization after training (run one forward pass over the data).
  const rEval = rng(7);
  const assignAfter: number[] = [];
  for (let i = 0; i < data.X.length; i++) {
    const x = Value.from(data.X[i], [1, DIM]);
    const { selected } = forwardToken(moe, x, rEval, { noiseStd: 0, routeMode: "index" });
    assignAfter.push(selected[0]);
  }
  const utilAfter = expertUtilization(assignAfter, NUM_EXPERTS);

  // How much did the gate weights actually move?
  let maxDelta = 0;
  for (let k = 0; k < gateBefore.length; k++) {
    maxDelta = Math.max(maxDelta, Math.abs(moe.gate.W.data[k] - gateBefore[k]));
  }

  console.log(`  门控权重梯度范数: 平均=${mean(gateGradNorms).toExponential(3)}  最大=${Math.max(...gateGradNorms).toExponential(3)}`);
  console.log(`  → 梯度全程 ≈0: argmax 索引是 detached 标量,损失到 gate.W 之间没有可导路径`);
  console.log(`  训练前后门控权重最大变化量 = ${maxDelta.toExponential(3)}  (≈0 = 门控被冻结)`);
  console.log("  专家利用率冻结在初始化偏好上:");
  console.log(bar(["init E0", "E1", "E2", "E3"], utilInit));
  console.log(bar(["aftr E0", "E1", "E2", "E3"], utilAfter));
  const frozen = utilInit.every((u, j) => Math.abs(u - utilAfter[j]) < 1e-9);
  console.log(`  利用率是否原地不动: ${frozen ? "✓ 完全冻结 (利用率向量逐元素相等)" : "× 有变化"}`);
  console.log(
    "\n  教训: argmax 只决定『走哪条路』,不可导;必须让连续的门控概率(softmax 后 gather 被选列)\n" +
      "        充当乘子来承梯度。把乘子建在离散索引上 = 门控网络收不到任何学习信号。",
  );
}

main();
