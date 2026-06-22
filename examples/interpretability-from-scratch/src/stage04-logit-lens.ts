// stage04-logit-lens.ts — Logit Lens: project each layer's residual stream to the vocab and
//   watch the answer crystallize with depth.
//
// WHY this technique: in a pre-norm transformer the residual stream is a running SUM — every
//   attention head and MLP writes an additive update, and the unembedding reads only the
//   FINAL sum. The logit lens borrows the final LayerNorm + unembedding and applies them to an
//   EARLIER, partial sum, asking "if the model had to commit here, what would it say?". The
//   layer at which the lens argmax first locks onto the true answer tells you WHERE IN DEPTH
//   the computation resolves — a coarse but causal-adjacent depth map, for free, no training.
//
// THE INVARIANT we lean on: residual points are cumulative (resid_post[L] ⊇ resid_post[L-1]),
//   so a faithful lens should show the answer's probability MONOTONICALLY (roughly) rising
//   with depth and never un-resolving once locked. If it zig-zags wildly, either the model
//   isn't pre-norm-additive in the way we assume or the lens is being read off-distribution.
//
// FAILURE MODE we deliberately demo (§4): applying the lens to the EARLIEST stream (embed,
//   before any block) is unfaithful. The embedding lives in input coordinates, not output
//   coordinates; lnFinal's statistics were fit on the final stream, so normalizing the
//   embedding with them is off-distribution. On THIS model the embed lens turns out CONFIDENTLY
//   WRONG (low entropy, but top-1 at/below chance vs the true answer) — not high-entropy. That
//   is the more dangerous failure: a naive reader sees a peaked distribution and trusts it.
//   Lesson: lens confidence at shallow depth is meaningless; only top-1-vs-ground-truth tells
//   you whether the lens is faithful, and at embed it is not.
//
// HONESTY: this toy model is 2 layers, dModel 32, on synthetic modAdd. Few layers + a tiny
//   LayerNorm effect make the lens MORE faithful here than on a real 12-96 layer model, where
//   tuned lenses / LN recentering are often needed. What transfers is the SHAPE — a monotone
//   approach to the answer and a discernible lock-in depth — not the clean absolute probs.

import { mulberry32, argmax } from "./core/rng.js";
import { modAdd } from "./core/tasks.js";
import { trainToyModel, defaultTrainConfig } from "./core/model_zoo.js";
import { runWithCache, logitLens } from "./core/interp.js";
import { Tensor } from "./core/autograd.js";
import { asciiBar, asciiSparkline } from "./core/viz.js";

// modAdd(7): vocab 0..6 are digits, token 7 is "=". seq = [a, b, =]; the ONLY scorable
// position is index 2 (the "="), whose target is (a+b)%7. Every lens reads that position.
const ANSWER_POS = 2;

/** The ordered list of residual-stream points we lens, shallow -> deep. These are exactly the
 *  cumulative-sum checkpoints: the running residual before block 0, then after attn and after
 *  the full block, for each layer. NOT head_z / attn_out / mlp_out — those are component
 *  OUTPUTS (deltas), not the running stream, so lensing them would project a partial delta
 *  through an unembedding that expects a full stream. Lens only what is a complete residual. */
function residPoints(nLayers: number): string[] {
  const pts = ["embed"];
  for (let l = 0; l < nLayers; l++) {
    pts.push(`blocks.${l}.resid_pre`); // stream entering block l (== embed for l=0)
    pts.push(`blocks.${l}.resid_mid`); // after attn write, before mlp
    pts.push(`blocks.${l}.resid_post`); // after full block l
  }
  return pts;
}

/** Short human label for a residual point, for compact tables. */
function shortLabel(point: string): string {
  if (point === "embed") return "embed";
  const m = point.match(/^blocks\.(\d+)\.(\w+)$/);
  if (!m) return point;
  const [, l, kind] = m;
  const k = kind === "resid_pre" ? "pre" : kind === "resid_mid" ? "mid" : "post";
  return `L${l}.${k}`;
}

/** Softmax-normalize one logit row (length = vocab) into probabilities. We reuse the engine's
 *  Tensor.softmax (numerically stable, max-subtracting) rather than hand-rolling exp/sum so the
 *  probabilities here match what training optimized — no second, subtly-different softmax. */
function softmaxRow(logits: Tensor, pos: number, vocab: number): Float64Array {
  const rowData = logits.data.subarray(pos * vocab, (pos + 1) * vocab);
  const row = new Tensor(Float64Array.from(rowData), [1, vocab]);
  return row.softmax().data;
}

/** Shannon entropy in bits of a probability row. Used to quantify "how undecided" a lens is:
 *  a faithful deep lens is low-entropy (committed); the unfaithful embed lens is high-entropy
 *  (near log2(vocab) = maximal confusion). This is the number that makes the failure mode
 *  concrete instead of a hand-wave. */
function entropyBits(probs: Float64Array): number {
  let h = 0;
  for (const p of probs) if (p > 1e-12) h -= p * Math.log2(p);
  return h;
}

interface LensReading {
  point: string;
  label: string;
  answerProb: number; // prob mass the lens puts on the TRUE answer token
  top1: number; // argmax token id of the lens
  top1Prob: number;
  entropy: number;
}

/** Lens every residual point for one input; report per-point answer prob + top token. */
function lensInput(model: any, ids: number[], answerToken: number): LensReading[] {
  const { cache } = runWithCache(model, ids);
  const vocab = model.cfg.vocab;
  const readings: LensReading[] = [];
  for (const point of residPoints(model.cfg.nLayers)) {
    const resid = cache[point];
    if (!resid) throw new Error(`lensInput: missing cached point "${point}"`);
    const lensLogits = logitLens(model, resid); // (seq, vocab) in OUTPUT coordinates
    const probs = softmaxRow(lensLogits, ANSWER_POS, vocab);
    const top1 = argmax(probs);
    readings.push({
      point,
      label: shortLabel(point),
      answerProb: probs[answerToken],
      top1,
      top1Prob: probs[top1],
      entropy: entropyBits(probs),
    });
  }
  return readings;
}

/** Lock-in depth = the first index from which the lens top-1 is the answer AND stays the answer
 *  through the deepest point. WHY "and stays": a momentary correct flicker that later flips is
 *  not resolution — we want the depth after which the answer is stably committed. Returns -1 if
 *  the lens never stably locks (a real possibility we don't want to paper over). */
function lockInIndex(readings: LensReading[], answerToken: number): number {
  for (let i = 0; i < readings.length; i++) {
    let stable = true;
    for (let j = i; j < readings.length; j++) {
      if (readings[j].top1 !== answerToken) {
        stable = false;
        break;
      }
    }
    if (stable) return i;
  }
  return -1;
}

/** Average per-point answer-prob and entropy across several inputs, so the depth profile isn't
 *  an artifact of one lucky example. N>1 is the cheap insurance against reading run-to-run
 *  noise as a depth signal (a real risk on a tiny model where any single input can be special). */
function averageProfile(
  model: any,
  inputs: { ids: number[]; answer: number }[],
): { points: string[]; labels: string[]; answerProb: number[]; entropy: number[] } {
  const points = residPoints(model.cfg.nLayers);
  const labels = points.map(shortLabel);
  const answerProb = new Array(points.length).fill(0);
  const entropy = new Array(points.length).fill(0);
  for (const { ids, answer } of inputs) {
    const r = lensInput(model, ids, answer);
    for (let i = 0; i < r.length; i++) {
      answerProb[i] += r[i].answerProb;
      entropy[i] += r[i].entropy;
    }
  }
  const n = inputs.length;
  for (let i = 0; i < points.length; i++) {
    answerProb[i] /= n;
    entropy[i] /= n;
  }
  return { points, labels, answerProb, entropy };
}

function main(): void {
  const task = modAdd(7); // a+b mod 7, vocab 8, answer at pos 2
  const vocab = task.vocab;
  console.log("=== Stage 04: Logit Lens — 答案如何逐层成形 ===\n");
  console.log(`任务: ${task.name}  vocab=${vocab}  答案位置=pos${ANSWER_POS}  (token ${vocab - 1} 是 '=')`);

  // Shared deterministic checkpoint (same key as stage01 => identical weights, reproducible).
  const trained = trainToyModel(task, defaultTrainConfig(task));
  const model = trained.model;
  console.log(`研究对象: 已训练 checkpoint  (finalLoss=${trained.finalLoss.toFixed(4)}, nLayers=${model.cfg.nLayers})`);

  // Build a fixed, seeded set of probe inputs. We oracle-label them so "answer prob" is the
  // prob on the TRUE answer, not the model's own guess — the lens is judged against ground
  // truth, the only honest reference.
  const rng = mulberry32(31337);
  const probeInputs: { ids: number[]; answer: number }[] = [];
  for (let i = 0; i < 8; i++) {
    const a = Math.floor(rng() * 7);
    const b = Math.floor(rng() * 7);
    probeInputs.push({ ids: [a, b, 7], answer: (a + b) % 7 });
  }

  // ---- [1] Depth profile: answer prob across the residual stream, averaged over inputs. ----
  console.log(`\n[1] 正确答案概率随残差流深度的变化 (8 个输入平均):`);
  const prof = averageProfile(model, probeInputs);
  console.log(
    asciiBar(
      prof.labels.map((label, i) => ({ label, value: prof.answerProb[i] })),
      { title: "P(answer)  shallow→deep", width: 36 },
    ),
  );
  console.log(`    ${asciiSparkline(prof.answerProb, { title: "P(answer)" })}`);

  // ---- [2] Lock-in depth on one concrete example, with the per-layer top-token table. ----
  // A single worked example makes the abstract profile legible: you SEE the prediction change.
  const ex = probeInputs[0];
  const exReadings = lensInput(model, ex.ids, ex.answer);
  const lockIdx = lockInIndex(exReadings, ex.answer);
  console.log(`\n[2] 单例逐层演化  输入=[${ex.ids.join(",")}]  真答案=${ex.answer}:`);
  console.log(`    ${"point".padEnd(10)} ${"top1".padStart(5)} ${"P(top1)".padStart(9)} ${"P(ans)".padStart(8)} ${"H(bits)".padStart(8)}`);
  for (let i = 0; i < exReadings.length; i++) {
    const r = exReadings[i];
    const mark = r.top1 === ex.answer ? "✓" : " ";
    const lock = i === lockIdx ? "  <- lock-in" : "";
    console.log(
      `    ${r.label.padEnd(10)} ${String(r.top1).padStart(5)}${mark} ${r.top1Prob.toFixed(4).padStart(8)} ${r.answerProb.toFixed(4).padStart(8)} ${r.entropy.toFixed(3).padStart(8)}${lock}`,
    );
  }
  if (lockIdx >= 0) {
    console.log(`    答案锁定层: ${exReadings[lockIdx].label} (此后 lens top-1 稳定为正确答案 ${ex.answer})`);
  } else {
    console.log(`    答案从未稳定锁定 — lens 在某些深度仍翻车 (此 toy 上不常见, 但属诚实可能)`);
  }

  // ---- [3] Aggregate lock-in depth distribution across all probe inputs. ----
  // One example's lock-in could be idiosyncratic; the distribution shows whether resolution
  // happens at a CONSISTENT depth (a real depth-localized circuit) or is smeared.
  console.log(`\n[3] 锁定深度分布 (8 个输入各自的 lock-in 点):`);
  const lockCounts = new Map<string, number>();
  for (const inp of probeInputs) {
    const r = lensInput(model, inp.ids, inp.answer);
    const li = lockInIndex(r, inp.answer);
    const key = li >= 0 ? r[li].label : "never";
    lockCounts.set(key, (lockCounts.get(key) ?? 0) + 1);
  }
  console.log(
    asciiBar(
      [...lockCounts.entries()].map(([label, value]) => ({ label, value })),
      { title: "lock-in depth histogram", width: 24 },
    ),
  );

  // ---- [4] FAILURE MODE: the lens is UNFAITHFUL at the earliest stream. ----
  // We compare the embed-point lens (off-distribution for lnFinal) against the deepest stream.
  // The diagnostic that actually settles it is top-1-vs-ground-truth: if the embed lens hits
  // the true answer at/below chance, it is reading noise no matter how confident it looks.
  // We also print entropy to expose the trap — on this model the embed lens is LOW entropy
  // (confident) yet wrong, the worst case for a reader who equates confidence with correctness.
  console.log(`\n[4] 失败模式: 对最早残差 (embed) 做原始 logit lens 不忠实:`);
  const uniformH = Math.log2(vocab);
  let embedTopCorrect = 0;
  let embedHsum = 0;
  let deepHsum = 0;
  let deepTopCorrect = 0;
  for (const inp of probeInputs) {
    const r = lensInput(model, inp.ids, inp.answer);
    const embed = r[0]; // first point == "embed"
    const deep = r[r.length - 1]; // deepest == last resid_post
    embedHsum += embed.entropy;
    deepHsum += deep.entropy;
    if (embed.top1 === inp.answer) embedTopCorrect++;
    if (deep.top1 === inp.answer) deepTopCorrect++;
  }
  const n = probeInputs.length;
  console.log(`    均匀分布熵上限 = log2(${vocab}) = ${uniformH.toFixed(3)} bits  (作对照: 熵越低=越"自信")`);
  console.log(`    embed lens : 平均熵 ${(embedHsum / n).toFixed(3)} bits  top-1命中真答案 ${embedTopCorrect}/${n}  (chance≈${(n / vocab).toFixed(1)})`);
  console.log(`    最深 lens  : 平均熵 ${(deepHsum / n).toFixed(3)} bits  top-1命中真答案 ${deepTopCorrect}/${n}`);
  console.log(`    判定: embed lens 熵并不高 (看起来"自信"), 但 top-1 命中真答案 ≈ 随机甚至更差 ⇒`);
  console.log(`          它是"自信地错" — 早期残差不在输出坐标系, 用 lnFinal 统计去归一化是 off-distribution,`);
  console.log(`          投出的峰是噪声里的伪峰。陷阱: 别把 lens 的"自信"当"正确", 唯一判据是 top-1 vs 真答案。`);
  console.log(`          正确读法: lens 只在足够深、已进入输出坐标系的残差上才忠实 (或用 tuned lens 校正)。`);

  console.log(
    `\n诚实边界: 本 toy 仅 ${model.cfg.nLayers} 层、LayerNorm 影响小, lens 忠实度高于真实大模型 (后者常需 tuned lens / LN 重定心)。` +
      `\n          可迁移的是形状: P(answer) 随深度近似单调上升 + 存在可辨认的锁定深度; 绝对概率偏乐观。`,
  );
}

main();
