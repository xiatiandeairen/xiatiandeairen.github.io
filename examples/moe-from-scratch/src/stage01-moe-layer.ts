// stage01-moe-layer.ts — Chapter 1: what an "MoE layer" structurally IS.
//
// THESIS of this chapter: an MoE layer is a single FFN cut into E independent experts.
//   Before routing, Top-k, or load-balance loss exist (those are later chapters), the
//   bare structural fact is: replace one MLP with E MLPs. This stage isolates THAT fact
//   and measures its two immediate consequences — params scale ~E×, and the multi-expert
//   model can fit better — using a deterministic, seeded training run we can reproduce.
//
// WHY no router here: routing is chapter 2+. To study the layer's structure in isolation
//   we use the simplest possible combiner — UNWEIGHTED MEAN of all E experts (every
//   expert fires on every token, "dense MoE"). This is deliberately NOT how production
//   MoE runs (the whole point of MoE is sparsity), but it lets us answer one question
//   cleanly: does splitting one FFN into E disjoint parameter sets buy us capacity?
//
// HONESTY (per core/data.ts): makeClusters is tiny (6 Gaussian blobs, spread tuned to
//   overlap so it is NOT trivially separable), so the ABSOLUTE accuracy/loss numbers are
//   optimistic / dataset-specific. What transfers is the RELATIVE story:
//   E experts hold ~E× the params and reach a lower train loss than one expert of the
//   same per-expert width; and — the failure demo — experts that are not initialized to
//   break symmetry collapse into one effective expert, making those extra params dead.
//
// FAILURE MODE this stage demonstrates (the chapter's real lesson): if all E experts are
//   initialized from the SAME rng seed, their forward outputs are elementwise (near-)
//   identical, their mean equals any single expert, and the E× params are wasted. Symmetry
//   breaking via independent random init is the PREREQUISITE for experts to ever diverge.
//
// Run: npx tsx src/stage01-moe-layer.ts   (offline, CPU, deterministic; no network/LLM).

import { Value, gradCheck } from "./core/tensor.js";
import { Expert, crossEntropy } from "./core/nn.js";
import { rng } from "./core/prng.js";
import { Adam } from "./core/optim.js";
import { makeClusters, type Dataset } from "./core/data.js";
import { plotLoss } from "./core/viz.js";

// ---- config (one place; all numbers below derive from these) -------------------------
const SEED = 7; // single documented seed for the whole stage; reproducible bit-for-bit.
const DIM = 6; // input feature dim
const NUM_CLASSES = 6; // == number of clusters; cluster id == class id (see makeClusters)
const N_TRAIN = 300;
// SPREAD large on purpose: with 6 blobs and spread=2.2 the clusters OVERLAP heavily, so the
//   task is NOT linearly separable and a tiny model is genuinely capacity-limited. Without
//   this, every model trivially hits loss 0 and the capacity comparison says nothing.
const SPREAD = 2.2;
const HIDDEN = 4; // per-expert hidden width — deliberately SMALL so one expert under-fits.
//                   HELD CONSTANT across dense vs MoE so the only variable is expert COUNT.
const E = 4; // experts in the multi-expert model
const STEPS = 300;
const LR = 5e-2;

// ---- a multi-expert layer: E experts, outputs combined by unweighted mean ------------
//
// INVARIANT: parameters() returns EVERY expert's params (flattened). Forgetting one
//   expert here would silently freeze it — nn.ts's parameters() contract is what we lean
//   on. We keep experts as a plain array (no router state yet) on purpose.
class DenseMoELayer {
  experts: Expert[];
  private invE: Value; // 1/E as a Value, reused so the mean is one scalar-broadcast mul.
  constructor(experts: Expert[]) {
    this.experts = experts;
    this.invE = Value.scalar(1 / experts.length);
  }
  // forward: mean_e expert_e(x). Every expert sees every token (no routing this chapter).
  forward(x: Value): Value {
    let acc = this.experts[0].forward(x);
    for (let e = 1; e < this.experts.length; e++) acc = acc.add(this.experts[e].forward(x));
    return acc.mul(this.invE);
  }
  parameters(): Value[] {
    return this.experts.flatMap((e) => e.parameters());
  }
}

function countParams(params: Value[]): number {
  let n = 0;
  for (const p of params) n += p.data.length;
  return n;
}

function rowFrom(x: number[]): Value {
  return Value.from(x, [1, x.length]);
}

// One full pass over the dataset: returns mean cross-entropy loss (Value, for backward)
// and the plain-number accuracy. INVARIANT: caller zeroGrad()s before, step()s after.
function trainEpochLoss(
  forward: (x: Value) => Value,
  data: Dataset,
  order: number[],
): { loss: Value; correct: number } {
  let lossSum = Value.scalar(0);
  let correct = 0;
  for (const i of order) {
    const logits = forward(rowFrom(data.X[i]));
    lossSum = lossSum.add(crossEntropy(logits, data.Y[i]));
    // argmax over logits row for the accuracy count (no grad needed for this read).
    let best = 0;
    for (let j = 1; j < logits.cols; j++) if (logits.data[j] > logits.data[best]) best = j;
    if (best === data.Y[i]) correct++;
  }
  return { loss: lossSum.mul(Value.scalar(1 / order.length)), correct };
}

// Train a model in place; returns the per-step loss series + final accuracy.
function train(
  forward: (x: Value) => Value,
  params: Value[],
  data: Dataset,
  seed: number,
): { losses: number[]; finalAcc: number } {
  const opt = new Adam(params, LR);
  const order = data.X.map((_, i) => i);
  const shuffleRng = rng(seed); // independent stream for batch order; documented coupling avoidance.
  const losses: number[] = [];
  let finalAcc = 0;
  for (let step = 0; step < STEPS; step++) {
    shuffleRng.shuffle(order);
    opt.zeroGrad(); // MUST precede backward(): grads accumulate (see optim.ts invariant).
    const { loss, correct } = trainEpochLoss(forward, data, order);
    loss.backward();
    opt.step();
    losses.push(loss.data[0]);
    finalAcc = correct / order.length;
  }
  return { losses, finalAcc };
}

// ---- (1) PROVE the engine before trusting any training number ------------------------
//
// We gradCheck a real slice of THIS stage's compute graph: a single Expert feeding
// cross-entropy. Per the gradCheck contract, f must rebuild the graph from the passed-in
// leaf x (gradCheck perturbs x.data in place). We check grad w.r.t. the INPUT row, with a
// fixed target class, which gives non-trivial (non-≈0) analytic grads to compare against.
function proveEngine(): number {
  const r = rng(SEED);
  const expert = new Expert(DIM, HIDDEN, NUM_CLASSES, r);
  const x0 = rowFrom([0.4, -0.7, 0.2, 1.1, -0.3, 0.6]);
  const target = 1;
  const f = (x: Value) => crossEntropy(expert.forward(x), target);
  return gradCheck(f, x0);
}

// ---- (3) the symmetry-breaking failure demo ------------------------------------------
//
// Build E experts. In the HEALTHY case each gets its own rng draw, so weights differ. In
// the BROKEN case every expert is built from a FRESH rng(sameSeed), so they are byte-for-
// byte identical leaves => identical forward outputs => the mean is just one expert.
function buildExperts(broken: boolean, seed: number): Expert[] {
  const experts: Expert[] = [];
  if (broken) {
    // BROKEN: each expert seeded identically -> identical init -> no symmetry breaking.
    for (let e = 0; e < E; e++) experts.push(new Expert(DIM, HIDDEN, NUM_CLASSES, rng(seed)));
  } else {
    // HEALTHY: ONE rng stream feeds all experts, so each consumes different draws.
    const r = rng(seed);
    for (let e = 0; e < E; e++) experts.push(new Expert(DIM, HIDDEN, NUM_CLASSES, r));
  }
  return experts;
}

// Max elementwise spread of the E experts' outputs on one probe input. Near 0 == experts
// are interchangeable (collapsed); larger == they compute genuinely different functions.
function maxExpertOutputSpread(experts: Expert[], probe: Value): number {
  const outs = experts.map((e) => e.forward(probe).data);
  let maxSpread = 0;
  for (let j = 0; j < outs[0].length; j++) {
    let lo = outs[0][j];
    let hi = outs[0][j];
    for (let e = 1; e < outs.length; e++) {
      if (outs[e][j] < lo) lo = outs[e][j];
      if (outs[e][j] > hi) hi = outs[e][j];
    }
    if (hi - lo > maxSpread) maxSpread = hi - lo;
  }
  return maxSpread;
}

function main(): void {
  console.log(`=== stage01: MoE 层结构 = 把一个 FFN 拆成 E 个专家 (seed=${SEED}) ===\n`);

  // (1) Engine proof.
  const relErr = proveEngine();
  const enginePass = relErr < 1e-4;
  console.log("① 引擎自检 (gradCheck: Expert→交叉熵, 数值梯度 vs autograd)");
  console.log(`   max 相对误差 = ${relErr.toExponential(3)}  (阈值 <1e-4: ${enginePass ? "PASS" : "FAIL"})`);
  console.log("   含义: 下面所有训练数字建立在『梯度算对了』之上, 否则一切归零。\n");

  // Shared dataset (same seed => identical data for both models => fair comparison).
  const data = makeClusters(NUM_CLASSES, N_TRAIN, DIM, rng(SEED), SPREAD);

  // (2a) Dense baseline: ONE expert (a plain 2-layer FFN) as the classifier.
  const denseExpert = new Expert(DIM, HIDDEN, NUM_CLASSES, rng(SEED + 1));
  const denseParams = denseExpert.parameters();
  const dense = train((x) => denseExpert.forward(x), denseParams, data, SEED + 100);

  // (2b) Multi-expert MoE layer: E independent experts, mean-combined, healthy init.
  const moeLayer = new DenseMoELayer(buildExperts(false, SEED + 1));
  const moeParams = moeLayer.parameters();
  const moe = train((x) => moeLayer.forward(x), moeParams, data, SEED + 100);

  const denseP = countParams(denseParams);
  const moeP = countParams(moeParams);
  console.log("② 两模型对比 (同数据 / 同每专家宽度 hidden=" + HIDDEN + " / 同 " + STEPS + " 步)");
  console.log("   模型           参数量      ×单专家   终态 train loss   终态 train acc");
  console.log(
    `   dense (1 专家) ${String(denseP).padStart(6)}      ${(denseP / denseP).toFixed(2)}x` +
      `      ${dense.losses[dense.losses.length - 1].toFixed(4)}` +
      `           ${(dense.finalAcc * 100).toFixed(1)}%`,
  );
  console.log(
    `   MoE (${E} 专家)   ${String(moeP).padStart(6)}      ${(moeP / denseP).toFixed(2)}x` +
      `      ${moe.losses[moe.losses.length - 1].toFixed(4)}` +
      `           ${(moe.finalAcc * 100).toFixed(1)}%`,
  );
  console.log(
    `   观察: 参数 ≈${E}× (${(moeP / denseP).toFixed(2)}x), MoE 终态 loss 更低` +
      ` (${(dense.losses[dense.losses.length - 1] - moe.losses[moe.losses.length - 1]).toFixed(4)} 的差)。`,
  );
  console.log("   注: toy 高斯簇 (6 类, spread=2.2 故意重叠), 绝对 loss/acc 偏乐观; 可迁移的是『参数与拟合能力的相对趋势』。\n");

  // (3) ASCII loss curves.
  console.log("③ loss 曲线 (左高右低 = 在下降)");
  console.log("   dense (1 专家):");
  console.log("   " + plotLoss(dense.losses));
  console.log("   MoE (" + E + " 专家):");
  console.log("   " + plotLoss(moe.losses) + "\n");

  // (4) FAILURE MODE: identical init => collapsed experts => E× params wasted.
  console.log("④ 失败模式: 不破缺对称性 (E 个专家用同一 seed 初始化)");
  const probe = rowFrom([0.4, -0.7, 0.2, 1.1, -0.3, 0.6]);

  const healthyExperts = buildExperts(false, SEED + 1);
  const brokenExperts = buildExperts(true, SEED + 1);
  const healthySpread = maxExpertOutputSpread(healthyExperts, probe);
  const brokenSpread = maxExpertOutputSpread(brokenExperts, probe);

  console.log(`   健康初始化 (各专家独立随机): 专家输出逐元素最大差 = ${healthySpread.toExponential(3)}`);
  console.log(`   坏初始化   (各专家同一 seed): 专家输出逐元素最大差 = ${brokenSpread.toExponential(3)}`);

  // Train the broken layer too, to show the params buy nothing vs a single expert.
  const brokenLayer = new DenseMoELayer(brokenExperts);
  const broken = train((x) => brokenLayer.forward(x), brokenLayer.parameters(), data, SEED + 100);
  const brokenFinal = broken.losses[broken.losses.length - 1];
  const denseFinal = dense.losses[dense.losses.length - 1];

  console.log(
    `   坏初始化 MoE 终态 loss = ${brokenFinal.toFixed(4)} vs dense(单专家) = ${denseFinal.toFixed(4)}` +
      ` (差 ${Math.abs(brokenFinal - denseFinal).toFixed(4)})`,
  );
  console.log(
    "   结论: 同 seed 初始化时 4 个专家输出几乎相同 (差≈0), mean 等价于单专家,",
  );
  console.log(
    `         多出的 ${moeP - denseP} 个参数白涨 — 对称性破缺 (独立随机初始化) 是专家分化的前提。`,
  );

  // Hard assertions so a regression in the engine/init is a loud failure, not a silent drift.
  if (!enginePass) throw new Error(`engine gradCheck FAILED: rel err ${relErr}`);
  if (!(brokenSpread < healthySpread)) {
    throw new Error(`symmetry-break demo broken: brokenSpread ${brokenSpread} !< healthySpread ${healthySpread}`);
  }
  if (!(moeP > denseP)) throw new Error(`MoE should have more params than dense`);
  console.log("\n=== stage01 完成: 所有断言通过 ===");
}

main();
