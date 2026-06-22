// stage05-activation-patching.ts — From correlation to causation: which component is
//   actually responsible? Activation patching is the causal arbiter.
//
// WHY this chapter exists: chapter 02 showed attention PATTERNS — and a pattern that looks
//   "sharp" (a head attending strongly to one position) is seductive: it FEELS like the head
//   is doing the work. But a sharp pattern is correlational; it says where the head LOOKS,
//   not whether the model's answer DEPENDS on it. Patching settles the question by surgery:
//   run a CLEAN input (correct answer) and a CORRUPT input (wrong answer), then re-run clean
//   while splicing in ONE activation from the corrupt run. If that single splice destroys the
//   clean answer, that component is CAUSALLY responsible. No splice, no causation — full stop.
//
// THE CORRUPTION DESIGN (the part people get wrong): clean and corrupt must be STRUCTURALLY
//   identical and differ in exactly ONE semantic factor, so the recovered logit gap isolates
//   THAT factor's circuit. Here clean=[a,b,=] and corrupt=[a',b,=] differ only in the first
//   operand. A component that recovers the gap is one that carries the value of operand `a`
//   to the "=" position. Change two things at once and patching localizes nothing.
//
// HONESTY (toy caveat, stated up front and revisited at the end): on this tiny algebraic task
//   the responsible circuit is concentrated, so the causal heatmap is unusually clean and
//   single-point recovery often nears 1.0. Real models have DISTRIBUTED computation and
//   BACKUP circuits (ablate the "main" head and a dormant one takes over), so single-point
//   patching UNDER-estimates responsibility there. What transfers is the METHOD and the
//   SHAPE: recovery concentrating on a few (layer, position, component) cells = a localized
//   circuit; a sharp attention pattern with ~0 recovery = the pattern lied.

import { argmax } from "./core/rng.js";
import { modAdd } from "./core/tasks.js";
import { TinyTransformer } from "./core/nn.js";
import { trainToyModel, defaultTrainConfig } from "./core/model_zoo.js";
import { patch, runWithCache } from "./core/interp.js";
import { asciiHeatmap, asciiBar } from "./core/viz.js";

const P = 7; // modular base; vocab = P+1 (0..6 plus the "=" delimiter token P)
const EQ_POS = 2; // the "=" position — the only scorable position; all patches target it

// ----------------------------------------------------------------------------
// A clean/corrupt pair that differs in EXACTLY ONE operand.
// ----------------------------------------------------------------------------
//
// Invariant the rest of the stage relies on: cleanAnswer != corruptAnswer (otherwise the
//   clean->corrupt logit gap is ~0 and recovery is undefined — division by a near-zero
//   denominator, which interp.patch guards by returning 0 but which would silently make the
//   whole heatmap meaningless). We also keep operand `b` and the "=" token fixed so the only
//   thing a responsible component can be carrying is the value of operand `a`.
interface PatchPair {
  cleanIds: number[];
  corruptIds: number[];
  cleanAnswer: number; // (a + b) % P
  corruptAnswer: number; // (a' + b) % P
}

function makePair(a: number, aCorrupt: number, b: number): PatchPair {
  if (a === aCorrupt) throw new Error("makePair: clean and corrupt operand must differ");
  const cleanAnswer = (a + b) % P;
  const corruptAnswer = (aCorrupt + b) % P;
  if (cleanAnswer === corruptAnswer) throw new Error("makePair: answers must differ for a usable gap");
  return {
    cleanIds: [a, b, P], // P is the "=" delimiter token id
    corruptIds: [aCorrupt, b, P],
    cleanAnswer,
    corruptAnswer,
  };
}

// ----------------------------------------------------------------------------
// Sanity gate: the model must actually get the clean pair right, else patching noise.
// ----------------------------------------------------------------------------
//
// Patching a model that doesn't solve the clean input measures nothing causal — you'd be
//   moving logits between two wrong answers. We assert the trained model predicts cleanAnswer
//   on cleanIds and corruptAnswer on corruptIds before trusting any recovery number. This is
//   the chapter-05 analogue of stage01's "did it learn?" gate.
function assertSolves(model: TinyTransformer, pair: PatchPair): { cleanOk: boolean; corruptOk: boolean } {
  const clean = runWithCache(model, pair.cleanIds);
  const corrupt = runWithCache(model, pair.corruptIds);
  const cleanPred = argmax(clean.logits.data.subarray(EQ_POS * (P + 1), (EQ_POS + 1) * (P + 1)));
  const corruptPred = argmax(corrupt.logits.data.subarray(EQ_POS * (P + 1), (EQ_POS + 1) * (P + 1)));
  return { cleanOk: cleanPred === pair.cleanAnswer, corruptOk: corruptPred === pair.corruptAnswer };
}

// ----------------------------------------------------------------------------
// The causal scan: patch every (layer, component) at the "=" position.
// ----------------------------------------------------------------------------
//
// We enumerate the residual-stream landmarks (resid_pre / attn_out / resid_mid / mlp_out /
//   resid_post) per layer plus every individual head_z. Each gets one patch and we record the
//   recovery fraction. interp.patch(clean, corrupt, point) is DENOISING by construction: it
//   runs clean and overwrites `point` with the corrupt value, asking "does injecting corrupt
//   info here break the right answer?" A high recovery means this point is where operand `a`'s
//   contribution lives.
function scanComponents(model: TinyTransformer, pair: PatchPair): { point: string; recovery: number }[] {
  const out: { point: string; recovery: number }[] = [];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    const landmarks = [
      `blocks.${l}.resid_pre`,
      `blocks.${l}.attn_out`,
      `blocks.${l}.resid_mid`,
      `blocks.${l}.mlp_out`,
      `blocks.${l}.resid_post`,
    ];
    for (const point of landmarks) {
      const r = patch(model, pair.cleanIds, pair.corruptIds, point, EQ_POS, pair.cleanAnswer);
      out.push({ point, recovery: r.recovery });
    }
  }
  return out;
}

function scanHeads(model: TinyTransformer, pair: PatchPair): { point: string; recovery: number }[] {
  const out: { point: string; recovery: number }[] = [];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    for (let h = 0; h < model.cfg.nHeads; h++) {
      const point = `blocks.${l}.attn.head_z.${h}`;
      const r = patch(model, pair.cleanIds, pair.corruptIds, point, EQ_POS, pair.cleanAnswer);
      out.push({ point, recovery: r.recovery });
    }
  }
  return out;
}

// ----------------------------------------------------------------------------
// Denoising vs noising: two directions of the same causal question.
// ----------------------------------------------------------------------------
//
// DENOISING: start CORRUPT (wrong answer), splice in the CLEAN activation. Recovery here = how
//   much the clean value of this point RESTORES the right answer = "is this component
//   SUFFICIENT to carry the signal?"
// NOISING: start CLEAN (right answer), splice in the CORRUPT activation (this is exactly
//   interp.patch). Recovery here = how much corrupting this point DESTROYS the right answer =
//   "is this component NECESSARY here?"
// They usually agree on a clean circuit but can diverge: a component can be necessary-in-place
//   yet not sufficient-alone (it needs upstream context that the corrupt run lacks). Showing
//   both keeps us honest about which causal claim we're actually making.
function patchDirections(model: TinyTransformer, pair: PatchPair, point: string): { noising: number; denoising: number } {
  // noising: clean run, overwrite point with corrupt value, measure recovery of clean answer
  const noising = patch(model, pair.cleanIds, pair.corruptIds, point, EQ_POS, pair.cleanAnswer).recovery;
  // denoising: swap the roles — corrupt run, overwrite point with clean value, measure how
  // much it recovers the corrupt run toward the CLEAN answer (the answer we are trying to
  // re-instate). Same formula, roles flipped: clean<->corrupt and target answer = cleanAnswer.
  const denoising = patch(model, pair.corruptIds, pair.cleanIds, point, EQ_POS, pair.cleanAnswer).recovery;
  return { noising, denoising };
}

// ----------------------------------------------------------------------------
// Attention sharpness — the correlational signal we are about to DEBUNK.
// ----------------------------------------------------------------------------
//
// "Sharpness" = how concentrated a head's attention at the query "=" position is (low entropy
//   = one key dominates = looks decisive). A high-sharpness head is exactly the kind chapter
//   02 would flag as "interesting". We compute it so we can put it next to patch recovery and
//   show the two disagree: sharp != causal.
function headSharpness(model: TinyTransformer, ids: number[]): { layer: number; head: number; sharpness: number }[] {
  // One forward populates each attention module's lastPatterns (per-head flattened seq*seq).
  runWithCache(model, ids);
  const seq = ids.length;
  const result: { layer: number; head: number; sharpness: number }[] = [];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    const patterns = model.blocks[l].attn.lastPatterns; // Float64Array[] per head, length seq*seq
    for (let h = 0; h < model.cfg.nHeads; h++) {
      const flat = patterns[h];
      // Row EQ_POS of the (seq, seq) attention matrix = how the "=" query distributes over keys.
      const row = Array.from({ length: seq }, (_, k) => flat[EQ_POS * seq + k]);
      // Sharpness = 1 - normalized entropy. 1.0 => all mass on one key (maximally sharp);
      // 0.0 => uniform attention (maximally diffuse). Normalized so seq length doesn't bias it.
      let entropy = 0;
      for (const p of row) if (p > 1e-12) entropy -= p * Math.log(p);
      const maxEntropy = Math.log(seq);
      const sharpness = maxEntropy > 0 ? 1 - entropy / maxEntropy : 0;
      result.push({ layer: l, head: h, sharpness });
    }
  }
  return result;
}

function shortName(point: string): string {
  // "blocks.0.resid_pre" -> "L0.resid_pre"; "blocks.1.attn.head_z.2" -> "L1.h2"
  return point
    .replace(/^blocks\.(\d+)\./, "L$1.")
    .replace(/attn\.head_z\.(\d+)/, "h$1");
}

function main(): void {
  const task = modAdd(P);
  console.log("=== Stage 05: Activation Patching — 从相关到因果 ===\n");
  console.log(`研究对象: 与 stage01 同一 checkpoint (${task.name}), 这样本章因果结论可与前几章互相印证。`);

  // Reuse the EXACT shared checkpoint (cache hit -> identical weights as stage01).
  const trained = trainToyModel(task, defaultTrainConfig(task));
  const model = trained.model;

  // --- 1. Build and validate a clean/corrupt pair. ----------------------------------------
  const pair = makePair(/*a=*/ 2, /*aCorrupt=*/ 5, /*b=*/ 1);
  const solves = assertSolves(model, pair);
  console.log(`\n[1] 构造 clean/corrupt 对 (只改第一个操作数):`);
  console.log(`    clean   = [${pair.cleanIds.join(",")}]  期望答案 ${pair.cleanAnswer}  (= (2+1)%7)`);
  console.log(`    corrupt = [${pair.corruptIds.join(",")}]  期望答案 ${pair.corruptAnswer}  (= (5+1)%7)`);
  console.log(`    模型解对 clean? ${solves.cleanOk ? "是" : "否"}   解对 corrupt? ${solves.corruptOk ? "是" : "否"}`);
  if (!solves.cleanOk || !solves.corruptOk) {
    console.log("    ⚠ 模型未解对基线对 — 后续 recovery 是在两个错误答案间搬运, 无因果意义。");
  }

  // --- 2. Causal heatmap over residual-stream landmarks. ----------------------------------
  const comp = scanComponents(model, pair);
  console.log(`\n[2] 因果热图: 对每个 (layer, 组件) 在 "=" 位做 patch 的恢复率 recovery`);
  console.log(`    recovery ≈ 1 => 这一点单独就足以决定答案; ≈ 0 => patch 它不影响答案 (无因果责任)`);
  // Lay out as a heatmap: rows = layers, cols = the 5 landmark kinds.
  const kinds = ["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"];
  const grid: number[][] = [];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    grid.push(kinds.map((k) => comp.find((c) => c.point === `blocks.${l}.${k}`)!.recovery));
  }
  console.log(
    asciiHeatmap(grid, {
      title: "    recovery heatmap (行=layer, 列=组件)",
      rowLabels: grid.map((_, l) => `L${l}`),
      colLabels: kinds,
      vmin: 0,
      vmax: 1,
    }),
  );
  // Name the responsible cells: those whose recovery clears a high bar.
  const responsible = comp.filter((c) => c.recovery >= 0.5).sort((a, b) => b.recovery - a.recovery);
  console.log(`\n    负责电路 (recovery ≥ 0.5):`);
  for (const c of responsible) console.log(`      ${shortName(c.point).padEnd(16)} recovery=${c.recovery.toFixed(3)}`);
  if (responsible.length === 0) console.log("      (无单点越过阈值 — 信号可能分布在多点, 见结尾诚实边界)");

  // --- 3. Per-head causal scan. -----------------------------------------------------------
  const heads = scanHeads(model, pair);
  console.log(`\n[3] 逐头 patch (head_z) 的恢复率:`);
  console.log(
    asciiBar(
      heads.map((h) => ({ label: shortName(h.point), value: h.recovery })),
      { title: "    per-head recovery", width: 36 },
    ),
  );

  // --- 4. Denoising vs noising on the top responsible point. ------------------------------
  // Pick the single most-recovering component overall as the focal point for both directions.
  const focal = [...comp, ...heads].sort((a, b) => b.recovery - a.recovery)[0].point;
  const dir = patchDirections(model, pair, focal);
  console.log(`\n[4] 同一点 (${shortName(focal)}) 的两个方向:`);
  console.log(`    noising  (clean 注入 corrupt, 测"必要性"): recovery=${dir.noising.toFixed(3)}`);
  console.log(`    denoising (corrupt 注入 clean, 测"充分性"): recovery=${dir.denoising.toFixed(3)}`);
  console.log(`    两方向${Math.abs(dir.noising - dir.denoising) < 0.15 ? "一致 => 干净的局部电路" : "分歧 => 必要≠充分, 该点依赖上游 context"}`);

  // --- 5. FAILURE MODE: a sharp attention pattern that patching shows is NOT causal. -------
  // This is the chapter's punchline. We rank heads by attention sharpness (the correlational
  // signal chapter 02 trusts) and by patch recovery (the causal verdict), then exhibit the
  // head where they disagree most: sharp attention, near-zero recovery. The pattern lied.
  console.log(`\n[5] 失败模式: 注意力很"尖锐"但 patch 证明它无因果作用`);
  const sharp = headSharpness(model, pair.cleanIds);
  const recByHead = new Map(heads.map((h) => [h.point, h.recovery]));
  const joined = sharp.map((s) => ({
    label: `L${s.layer}.h${s.head}`,
    sharpness: s.sharpness,
    recovery: recByHead.get(`blocks.${s.layer}.attn.head_z.${s.head}`) ?? 0,
  }));
  console.log("    head        sharpness   recovery   判定");
  for (const j of joined.sort((a, b) => b.sharpness - a.sharpness)) {
    const verdict =
      j.sharpness >= 0.4 && Math.abs(j.recovery) < 0.1
        ? "尖锐但无因果 ← 注意力图骗了你"
        : Math.abs(j.recovery) >= 0.1
          ? "有因果责任"
          : "既不尖锐也无责任";
    console.log(`    ${j.label.padEnd(8)}  ${j.sharpness.toFixed(3)}      ${j.recovery.toFixed(3)}      ${verdict}`);
  }
  // Find the most damning disagreement to call out explicitly.
  const liar = joined
    .filter((j) => j.sharpness >= 0.4 && Math.abs(j.recovery) < 0.1)
    .sort((a, b) => b.sharpness - a.sharpness)[0];
  if (liar) {
    console.log(
      `\n    ⇒ ${liar.label}: 注意力 sharpness=${liar.sharpness.toFixed(3)} (看起来很决定性), 但 patch recovery=${liar.recovery.toFixed(3)} ≈ 0。`,
    );
    console.log(`      把它的输出替换成 corrupt 值, 答案纹丝不动 => 它根本没参与计算答案。`);
    console.log(`      教训: 注意力图是"它在看哪"的相关证据, patch 才是"答案是否依赖它"的因果裁判。`);
  } else {
    console.log(`\n    (本对未出现"尖锐但无责任"的头 — 换一个 clean/corrupt 对常能复现; 这正是 toy 任务电路太干净的副作用。)`);
  }

  console.log(
    `\n诚实边界: toy 任务因果高度集中, 热图干净、单点 recovery 常接近 1.0。真模型存在分布式计算与 backup 电路` +
      ` (敲掉主组件, 备用组件顶上), 单点 patch 会 LOW-estimate 责任, 需配合 path patching / 多点联合 ablation。` +
      ` 可迁移的是方法与形状: recovery 集中在少数 (layer,pos,组件) = 局部电路; 尖锐注意力 + ~0 recovery = 图在撒谎。`,
  );
}

main();
