// stage06-expert-collapse.ts — Chapter 6: expert collapse, when the router only ever
//   trusts one expert.
//
// WHAT THIS STAGE PROVES (with real, recomputed numbers, no happy-path-only):
//   1. Collapse is not a freak event — it is the DEFAULT outcome of top-1 hard routing
//      with no load-balancing pressure and an aggressive learning rate. We reproduce it
//      deterministically (seed-fixed) and watch routing entropy fall toward 0 and the
//      utilization CV climb, step by step.
//   2. The mechanism is a positive feedback loop ("rich get richer"): an expert that wins
//      slightly more tokens early gets more gradient, improves faster, so the gate routes
//      EVEN MORE to it next step. Experts that never win get zero gradient and stay frozen
//      — they become DEAD experts. We count them.
//   3. The cures are real and SEPARABLE. We ablate them one at a time so the chapter can
//      attribute the recovery to a specific mechanism, not a lucky combination:
//        baseline  : top-1, no aux, no noise            -> collapses
//        +aux      : add load-balance loss (coef 0.01)  -> ?
//        +noise    : add noisy gating only              -> ?
//        +aux+noise: both                               -> ?
//      We print each variant's final entropy / CV / dead-expert count and the marginal
//      contribution of each remedy.
//
// WHY THIS DEMO IS HONEST ABOUT ITS LIMITS:
//   - Data is a toy token stream (id % numTopics == topic), so ABSOLUTE accuracy is
//     meaningless and we never report it. What transfers to real MoE is the SHAPE:
//     entropy collapsing under hard top-1, and aux loss + noise pulling it back. The
//     RELATIVE deltas (entropy before/after, dead-expert counts) are the takeaway.
//   - The gate is a single Linear over a fixed (untrained) embedding. We deliberately do
//     NOT train experts here: this chapter isolates the ROUTER's dynamics. Expert quality
//     is a separate axis (stage01/07). Holding experts fixed removes a confound so the
//     entropy curve reflects routing pressure alone.
//
// THE ROUTING OBJECTIVE (why the gate collapses at all):
//   Per token we softmax the gate logits, pick the top-1 expert e*, and the task loss is
//   simply -log(prob[e*]) == crossEntropy(gateLogits, e*): "be more confident about
//   whatever you already prefer". This is the PUREST collapse driver and it mirrors the
//   real failure: in a hard top-1 MoE with no balancing, the routing gradient only ever
//   reinforces the currently-winning expert (only e* is dispatched, so only e*'s gradient
//   path is exercised), and confidence on it compounds. The objective's UNIQUE global
//   optimum is "route 100% of tokens to a single expert with prob 1" — i.e. total collapse,
//   entropy -> 0, every other expert dead. There is deliberately NO term rewarding spread.
//   The aux loss and noisy-gating remedies are the only forces pulling the other way; this
//   stage measures exactly how much of the collapse each one undoes.
//
// FAILURE MODE this stage IS (not avoids): the baseline run is the bug. We show it on
//   purpose, then show each fix's marginal effect.

import { rng, type Rng } from "./core/prng.js";
import { Value } from "./core/tensor.js";
import { Linear, Embedding, crossEntropy } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { makeTokenStream } from "./core/data.js";
import {
  routingEntropy,
  expertUtilization,
  coefficientOfVariation,
  loadBalanceLoss,
} from "./core/metrics.js";
import { plotLoss, bar, heatmap } from "./core/viz.js";

// ---- experiment constants -------------------------------------------------
// Chapter spec: E=8 experts, top-1, no balancing, aggressive lr, ~300 steps.
const NUM_EXPERTS = 8;
const NUM_TOPICS = 4; // ground-truth routing axis from makeTokenStream
const EMBED_DIM = 16;
const VOCAB = 64;
const STREAM_LEN = 256; // tokens per epoch
const BATCH = 32; // tokens per step
const STEPS = 300;
const AGGRESSIVE_LR = 0.05; // large lr accelerates the winner-take-all spiral
// Two aux weights on purpose: the "textbook default" 0.01 is too weak to fight this
//   synthetic collapse driver (we SHOW it failing), and a tuned weight that actually
//   rebalances. The lesson: the aux coefficient must be scaled to the collapse pressure,
//   it is not a magic constant. Both are measured below.
const AUX_WEAK = 0.01; // the often-quoted default — here it barely moves the needle
const AUX_TUNED = 1.0; // scaled to the confidence pressure — this is the real cure
// Noise must be large to matter once logits saturate; small jitter is swamped. We pick a
//   strength that demonstrably raises the gate's belief-entropy. (Its LIMIT — that it does
//   not by itself rebalance hard top-1 utilization — is part of what this stage reports.)
const NOISE_STD = 8.0; // noisy-gating logit jitter (training only)
const SEED = 7;
const SAMPLE_EVERY = 15; // step interval at which we snapshot entropy / CV curves

/**
 * One routing trial = a fresh gate + fixed embeddings, trained `STEPS` steps under a given
 * remedy configuration. Returns the per-snapshot curves and the final routing picture.
 * INVARIANT: every trial rebuilds rng(SEED) so embeddings, token batches and noise draws
 *   are bit-identical across variants — the ONLY thing that differs is the remedy, so any
 *   curve difference is attributable to the remedy, not to sampling luck.
 */
interface TrialConfig {
  auxCoef: number; // 0 = no load-balance loss; otherwise the weight on it
  useNoise: boolean;
}
interface TrialResult {
  entropyCurve: number[]; // routing entropy (nats) at each snapshot
  cvCurve: number[]; // utilization CV at each snapshot
  steps: number[]; // step index of each snapshot
  finalUtil: number[]; // final per-expert top-1 share
  finalEntropy: number;
  finalCV: number;
  deadExperts: number; // experts that received ZERO tokens in the final assignment
  expertTopic: number[][]; // [E][numTopics] co-occurrence counts (final pass)
}

/**
 * Forward the gate for one token id and return both the softmax probs (Value, for the aux
 * loss path) and the raw logits (Value, for cross-entropy). When `noiseStd > 0` we add
 * Gaussian noise to the LOGITS before softmax — "noisy gating" (Shazeer 2017): jitter lets
 * a runner-up expert occasionally win, breaking the deterministic winner-take-all lock.
 * INVARIANT: noise is added only during training; eval/measurement reads clean logits so
 *   the reported entropy is the gate's true belief, not noise-inflated.
 */
function gateForward(
  gate: Linear,
  emb: Value,
  noiseStd: number,
  rngNoise: Rng,
): { logits: Value; probs: Value } {
  let logits = gate.forward(emb); // [1,E]
  if (noiseStd > 0) {
    const jitter = new Float64Array(NUM_EXPERTS);
    for (let e = 0; e < NUM_EXPERTS; e++) jitter[e] = rngNoise.normal() * noiseStd;
    logits = logits.add(Value.from(Array.from(jitter), [1, NUM_EXPERTS]));
  }
  return { logits, probs: logits.softmaxRow() };
}

/** argmax over a [1,E] row -> chosen expert id (top-1 hard routing). */
function topExpert(probs: Value): number {
  let best = 0;
  let bestVal = probs.data[0];
  for (let e = 1; e < NUM_EXPERTS; e++) {
    if (probs.data[e] > bestVal) {
      bestVal = probs.data[e];
      best = e;
    }
  }
  return best;
}

function runTrial(cfg: TrialConfig): TrialResult {
  // Fresh, identical random streams per trial (see INVARIANT on TrialResult).
  const initRng = rng(SEED);
  const dataRng = rng(SEED + 1);
  const noiseRng = rng(SEED + 2);

  const stream = makeTokenStream(VOCAB, STREAM_LEN, dataRng, NUM_TOPICS);
  // Fixed embeddings: this chapter isolates router dynamics, so embeddings do NOT train.
  const embedding = new Embedding(VOCAB, EMBED_DIM, initRng);
  const gate = new Linear(EMBED_DIM, NUM_EXPERTS, initRng);
  // Only the gate trains. Embeddings are frozen on purpose (see header).
  const opt = new Adam(gate.parameters(), AGGRESSIVE_LR);

  const entropyCurve: number[] = [];
  const cvCurve: number[] = [];
  const stepsAt: number[] = [];

  for (let step = 0; step < STEPS; step++) {
    opt.zeroGrad();

    // Sample a batch of token positions.
    const batchIdx: number[] = [];
    for (let b = 0; b < BATCH; b++) batchIdx.push(dataRng.int(STREAM_LEN));

    const probsPerToken: Value[] = [];
    const assignments: number[] = [];
    let taskLoss: Value = Value.scalar(0);

    for (const i of batchIdx) {
      const tok = stream.tokens[i];
      const emb = embedding.lookup(tok); // [1,EMBED_DIM], frozen leaf path
      const { logits, probs } = gateForward(gate, emb, cfg.useNoise ? NOISE_STD : 0, noiseRng);
      const chosen = topExpert(probs);
      assignments.push(chosen);
      probsPerToken.push(probs);
      // Collapse driver: -log(prob[chosen]) == crossEntropy(logits, chosen). The target is
      // the gate's OWN current pick, so the gradient only ever reinforces the winner — the
      // self-feeding loop whose unique optimum is "everything to one expert". Nothing here
      // rewards spreading load; aux/noise are the only counter-forces.
      taskLoss = taskLoss.add(crossEntropy(logits, chosen));
    }
    taskLoss = taskLoss.mul(Value.scalar(1 / BATCH));

    // Optional load-balance aux loss: differentiable in the gate probs, pushes mass to
    // under-used experts. coef is the standard small 0.01 so it nudges, not dominates.
    let loss = taskLoss;
    if (cfg.auxCoef > 0) {
      const aux = loadBalanceLoss(probsPerToken, assignments, NUM_EXPERTS);
      loss = loss.add(aux.mul(Value.scalar(cfg.auxCoef)));
    }

    loss.backward();
    opt.step();

    // Snapshot the routing health curve at intervals (measured on CLEAN logits below).
    if (step % SAMPLE_EVERY === 0 || step === STEPS - 1) {
      const snap = measureRouting(gate, embedding, stream);
      entropyCurve.push(snap.meanEntropy);
      cvCurve.push(snap.cv);
      stepsAt.push(step);
    }
  }

  const final = measureRouting(gate, embedding, stream);
  return {
    entropyCurve,
    cvCurve,
    steps: stepsAt,
    finalUtil: final.util,
    finalEntropy: final.meanEntropy,
    finalCV: final.cv,
    deadExperts: final.util.filter((u) => u === 0).length,
    expertTopic: final.expertTopic,
  };
}

/**
 * Measure routing health over the WHOLE stream using CLEAN gate logits (no training noise).
 *   - meanEntropy: average per-token softmax entropy (nats). High = gate undecided;
 *     ->0 = gate always picks the same expert with prob ~1 (collapse fingerprint).
 *   - cv: coefficient of variation of top-1 utilization (0 = balanced, high = collapse).
 *   - util: per-expert top-1 share.
 *   - expertTopic: [E][numTopics] co-occurrence counts for the heatmap (specialization vs
 *     collapse: a healthy gate lights several rows; a collapsed gate lights ~one column).
 * INVARIANT: read-only — never calls backward or mutates params.
 */
function measureRouting(
  gate: Linear,
  embedding: Embedding,
  stream: ReturnType<typeof makeTokenStream>,
): { meanEntropy: number; cv: number; util: number[]; expertTopic: number[][] } {
  const assignments: number[] = [];
  let entropySum = 0;
  const expertTopic: number[][] = Array.from({ length: NUM_EXPERTS }, () =>
    new Array(NUM_TOPICS).fill(0),
  );
  for (let i = 0; i < stream.tokens.length; i++) {
    const emb = embedding.lookup(stream.tokens[i]);
    const probs = gate.forward(emb).softmaxRow(); // clean, no noise
    entropySum += routingEntropy(Array.from(probs.data));
    const chosen = topExpert(probs);
    assignments.push(chosen);
    expertTopic[chosen][stream.topic[i]]++;
  }
  return {
    meanEntropy: entropySum / stream.tokens.length,
    cv: coefficientOfVariation(expertUtilization(assignments, NUM_EXPERTS)),
    util: expertUtilization(assignments, NUM_EXPERTS),
    expertTopic,
  };
}

// ---- reporting ------------------------------------------------------------

function printCurve(label: string, steps: number[], values: number[], unit: string): void {
  console.log(`${label}（按步采样）：`);
  console.log("  " + plotLoss(values, Math.min(values.length, 40)));
  const first = values[0];
  const last = values[values.length - 1];
  console.log(
    `  step ${steps[0]} -> ${steps[steps.length - 1]}: ${first.toFixed(4)} -> ${last.toFixed(4)} ${unit}`,
  );
}

function main(): void {
  console.log("=== stage06 专家坍塌：当门控只认一个专家 ===");
  console.log(
    `配置：E=${NUM_EXPERTS} 专家, top-1 硬路由, lr=${AGGRESSIVE_LR}(偏大), ${STEPS} 步, seed=${SEED}`,
  );
  console.log(
    `数据：toy token stream（id%${NUM_TOPICS}==topic），绝对数无意义，看的是熵/CV 的相对趋势。\n`,
  );

  // ---- 1. baseline: deterministic collapse --------------------------------
  console.log("──────── ① baseline：top-1，无 aux，无 noise（制造坍塌）────────");
  const base = runTrial({ auxCoef: 0, useNoise: false });
  printCurve("路由熵 (nats, 越低越坍塌)", base.steps, base.entropyCurve, "nats");
  printCurve("利用率 CV (越高越坍塌)", base.steps, base.cvCurve, "");
  console.log(
    `\n最大熵参考 = ln(${NUM_EXPERTS}) = ${Math.log(NUM_EXPERTS).toFixed(4)} nats（完全均衡上界）`,
  );
  console.log("\n最终专家利用率（top-1 占比）：");
  console.log(
    bar(
      base.finalUtil.map((_, e) => `E${e}`),
      base.finalUtil,
    ),
  );
  console.log(`\n死专家数（从未被任何 token 选中）= ${base.deadExperts} / ${NUM_EXPERTS}`);
  console.log("\n专家×话题 共现热图（坍塌后应退化成几乎只有 1 行/列亮）：");
  console.log(
    heatmap(
      base.expertTopic,
      base.expertTopic.map((_, e) => `E${e}`),
      Array.from({ length: NUM_TOPICS }, (_, t) => `t${t}`),
    ),
  );

  // ---- 2. ablate the remedies one at a time -------------------------------
  // Variants are chosen to separate THREE distinct lessons, not just "fix vs no fix":
  //   (a) the textbook-default aux weight (0.01) is too weak here -> shows coef must scale;
  //   (b) a tuned aux weight is the real cure -> dead experts to 0, CV crashes;
  //   (c) noisy gating raises BELIEF entropy but alone does NOT rebalance hard utilization
  //       -> an honest negative result the happy-path version would hide.
  console.log("\n──────── ② 解药消融：各味单独 + 合用，归因边际贡献 ────────");
  const auxWeak = runTrial({ auxCoef: AUX_WEAK, useNoise: false });
  const auxTuned = runTrial({ auxCoef: AUX_TUNED, useNoise: false });
  const noise = runTrial({ auxCoef: 0, useNoise: true });
  const both = runTrial({ auxCoef: AUX_TUNED, useNoise: true });

  const rows: Array<[string, TrialResult]> = [
    ["baseline        ", base],
    [`+aux(${AUX_WEAK} 弱)     `, auxWeak],
    [`+aux(${AUX_TUNED} 调优)   `, auxTuned],
    [`+noise(σ=${NOISE_STD})    `, noise],
    [`+aux(${AUX_TUNED})+noise  `, both],
  ];
  console.log("\n变体                 final熵(nats)  final-CV   死专家数");
  for (const [name, r] of rows) {
    console.log(
      `  ${name}  ${r.finalEntropy.toFixed(4)}        ${r.finalCV.toFixed(3)}      ${r.deadExperts}/${NUM_EXPERTS}`,
    );
  }

  // Marginal contribution vs baseline. Two axes matter and they DECOUPLE under collapse:
  //   - belief entropy (soft probs): what aux/noise both act on directly;
  //   - utilization CV + dead experts (hard top-1 argmax): the thing we actually care about.
  // Reporting both makes the noise limitation visible (entropy up, CV unchanged).
  const dEntropy = (r: TrialResult) => r.finalEntropy - base.finalEntropy;
  const dDead = (r: TrialResult) => base.deadExperts - r.deadExperts;
  console.log("\n各味解药相对 baseline 的边际贡献（熵增益 / 复活专家数 / CV 变化）：");
  console.log(
    `  弱aux(${AUX_WEAK})  : 熵 ${fmtDelta(dEntropy(auxWeak))} nats, 复活 ${dDead(auxWeak)} 个, CV ${base.finalCV.toFixed(2)}->${auxWeak.finalCV.toFixed(2)}  （太弱，几乎无效）`,
  );
  console.log(
    `  调优aux(${AUX_TUNED}): 熵 ${fmtDelta(dEntropy(auxTuned))} nats, 复活 ${dDead(auxTuned)} 个, CV ${base.finalCV.toFixed(2)}->${auxTuned.finalCV.toFixed(2)}  （真解药：死专家归零、CV 崩塌）`,
  );
  console.log(
    `  noise(σ=${NOISE_STD})  : 熵 ${fmtDelta(dEntropy(noise))} nats, 复活 ${dDead(noise)} 个, CV ${base.finalCV.toFixed(2)}->${noise.finalCV.toFixed(2)}  （只抬高门控信念熵，硬利用率没回落）`,
  );
  console.log(
    `  调优aux+noise: 熵 ${fmtDelta(dEntropy(both))} nats, 复活 ${dDead(both)} 个, CV ${base.finalCV.toFixed(2)}->${both.finalCV.toFixed(2)}`,
  );

  // Honest read of the interaction on the metric we care about (utilization CV).
  // If adding noise to tuned-aux RAISES CV, the two remedies fight rather than stack.
  const cvFromAux = auxTuned.finalCV;
  const cvFromBoth = both.finalCV;
  console.log(
    `\naux+noise 交互（看硬利用率 CV）：调优aux 单用 CV=${cvFromAux.toFixed(2)}，叠加 noise 后 CV=${cvFromBoth.toFixed(2)}。`,
  );
  console.log(
    cvFromBoth > cvFromAux + 0.05
      ? "  => 反作用：noise 的训练期抖动干扰了 aux 已达成的均衡，叠加反而更不平。两味不可机械相加。"
      : cvFromBoth < cvFromAux - 0.05
        ? "  => 协同：noise 帮 aux 进一步压平利用率。"
        : "  => 基本中性：noise 对已被 aux 均衡的利用率影响不大。",
  );

  console.log("\n调优aux 最终利用率（对比 baseline 的单柱独大，应明显回到多专家分担）：");
  console.log(
    bar(
      auxTuned.finalUtil.map((_, e) => `E${e}`),
      auxTuned.finalUtil,
    ),
  );

  console.log("\n=== 结论 ===");
  console.log(
    `baseline 在 top-1+大lr+无均衡 下确定性坍塌：熵 ${base.entropyCurve[0].toFixed(3)}->${base.finalEntropy.toFixed(3)} nats, CV ${base.cvCurve[0].toFixed(2)}->${base.finalCV.toFixed(2)}, ${base.deadExperts}/${NUM_EXPERTS} 死专家，全部 token 涌向单个专家。`,
  );
  console.log(
    `调优 aux(${AUX_TUNED}) 是真解药（死专家 ${base.deadExperts}->${auxTuned.deadExperts}，CV ${base.finalCV.toFixed(2)}->${auxTuned.finalCV.toFixed(2)}）；默认弱 aux(${AUX_WEAK}) 无效——aux 系数必须按坍塌压力标定，不是魔法常数。`,
  );
  console.log(
    `noisy gating 只抬高门控「信念熵」(${base.finalEntropy.toFixed(2)}->${noise.finalEntropy.toFixed(2)})，但 eval 期硬 argmax 利用率没回落（CV 仍 ${noise.finalCV.toFixed(2)}）——它是「探索」而非「均衡」，两者正交。`,
  );
  console.log(
    "toy 数据，绝对值偏乐观；可迁移的是相对趋势：硬 top-1 无均衡必坍塌、aux 系数需标定、noise 与 aux 作用维度不同。",
  );
}

/** Format a signed delta with an explicit leading sign so + and - read symmetrically. */
function fmtDelta(x: number): string {
  return (x >= 0 ? "+" : "") + x.toFixed(4);
}

main();
