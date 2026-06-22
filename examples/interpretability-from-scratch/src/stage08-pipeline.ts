// stage08-pipeline.ts — Confluence and honesty: chaining the tools into a falsifiable
//                       explanation PIPELINE, then scoring how faithful the explanation is.
//
// WHY this stage is the capstone: earlier chapters each wielded ONE tool (attention viz,
//   probe, logit lens, patching, SAE, ablation). In isolation any single tool can mislead —
//   a probe shows information is PRESENT, attention shows where a head LOOKS, but neither
//   shows the component is USED. This stage runs all of them on ONE behavior in a fixed
//   order and assembles an EVIDENCE MATRIX: each row a technique, whether it supports the
//   hypothesis, and how strong the signal is. Then it computes a single end-to-end FAITHFULNESS
//   number from the CAUSAL tools only (patch recovery x ablation necessity), because faithful
//   means "the explanation predicts what happens when you intervene", not "the story is
//   internally consistent".
//
// THE FAILURE MODE THIS STAGE EXISTS TO SHOW: we deliberately surface a component where the
//   CORRELATIONAL evidence is all green (attention high AND probe high) but the CAUSAL evidence
//   is red (patching/ablation move the answer ~0). That component is a decoy: it correlates
//   with the answer but does not compute it. An explanation built without the intervention
//   step would confidently and WRONGLY name it. This is why correlation-agreement across
//   several correlational tools does NOT establish causation — they can all be fooled by the
//   same confound, and only intervention breaks the tie.
//
// HONESTY (load-bearing, do not delete): on this toy modAdd model the pipeline converges
//   cleanly — one position, one component often dominates, recovery near 1.0. On real models
//   the techniques routinely CONTRADICT each other (a probe lights up where patching finds
//   nothing; ablation drop != patch recovery), circuits are distributed, and "is this
//   explanation faithful?" is an OPEN research problem with no settled metric. The transferable
//   content here is the SHAPE of the discipline: (1) always close the loop with intervention,
//   (2) report disagreement instead of cherry-picking the agreeing tool, (3) treat a single
//   faithfulness scalar as a summary, never a proof.

import { Tensor } from "./core/autograd.js";
import { TinyTransformer } from "./core/nn.js";
import { modAdd, type Task } from "./core/tasks.js";
import { mulberry32 } from "./core/rng.js";
import { trainToyModel, defaultTrainConfig } from "./core/model_zoo.js";
import {
  runWithCache,
  logitLens,
  linearProbe,
  patch,
  ablate,
  trainSAE,
  saeEncode,
  activationNames,
} from "./core/interp.js";
import { asciiBar, asciiSparkline } from "./core/viz.js";

// ----------------------------------------------------------------------------
// The behavior under explanation.
// ----------------------------------------------------------------------------
//
// modAdd(7): sequence [a, b, =] -> predict (a+b) mod 7 at the "=" position (index 2). The
//   ANSWER position is the only scorable one; everything we measure is the answer token's
//   logit at that position. We freeze ONE clean example and ONE corrupt counterpart that
//   differ ONLY in b, so the corrupt run has a DIFFERENT correct answer. Patching a clean
//   activation with its corrupt value then asks "does this component carry the addends'
//   information into the answer?".
const P = 7;
const EQ = P; // delimiter token id (see tasks.ts modAdd)
const ANSWER_POS = 2; // the "=" position; the only determined output

interface Behavior {
  cleanIds: number[];
  corruptIds: number[];
  cleanAnswer: number; // (a+b) % P for the clean run
  corruptAnswer: number; // (a+b') % P for the corrupt run
}

/** Pick a clean [a,b,=] and a corrupt [a,b',=] whose answers differ, so the clean->corrupt
 *  logit gap on the clean answer token is real (a patch has something to move). We require
 *  cleanAnswer != corruptAnswer; otherwise the "gap" is zero and recovery is undefined. */
function makeBehavior(rng: () => number): Behavior {
  for (let tries = 0; tries < 100; tries++) {
    const a = Math.floor(rng() * P);
    const b = Math.floor(rng() * P);
    let bPrime = Math.floor(rng() * P);
    if (bPrime === b) bPrime = (bPrime + 1) % P;
    const cleanAnswer = (a + b) % P;
    const corruptAnswer = (a + bPrime) % P;
    if (cleanAnswer !== corruptAnswer) {
      return { cleanIds: [a, b, EQ], corruptIds: [a, bPrime, EQ], cleanAnswer, corruptAnswer };
    }
  }
  throw new Error("makeBehavior: could not find distinct-answer pair (impossible for P>1)");
}

// ----------------------------------------------------------------------------
// Evidence matrix scaffolding.
// ----------------------------------------------------------------------------
//
// Each tool emits one EvidenceRow. `kind` separates CORRELATIONAL evidence (information is
//   present / where a head looks) from CAUSAL evidence (intervention changes the output). The
//   faithfulness score reads ONLY causal rows — that separation is the whole point of the
//   stage, encoded in the data type so the failure-mode demo can't accidentally count a
//   correlational row as proof.
type EvidenceKind = "correlational" | "causal";
interface EvidenceRow {
  technique: string;
  kind: EvidenceKind;
  metric: number; // the raw measured number (recovery, accuracy, drop, ...)
  supports: boolean; // does this row support "component X computes the answer"?
  strength: number; // [0,1] normalized confidence the row contributes
  note: string;
}

/** The hypothesis a real interpreter would write down before testing it. We name the most
 *  causally responsible attention head as the candidate "answer-writing" component; the
 *  pipeline's job is to corroborate or refute it across tools. */
interface Hypothesis {
  point: string; // hook name of the candidate component
  label: string; // human-readable name
}

// ----------------------------------------------------------------------------
// Tool 1 (CORRELATIONAL): attention — where does the "=" query look?
// ----------------------------------------------------------------------------
//
// At the answer position the model must combine a and b. A head that ATTENDS to the addend
//   positions (0 and 1) is a candidate "reader". We read the cached pattern matrix for each
//   head: row ANSWER_POS is the query's distribution over keys. High mass on positions {0,1}
//   is correlational support — necessary-looking but, as the failure mode shows, not
//   sufficient. Returns per-head attention-to-addends so we can both pick a candidate and
//   later expose a decoy.
function attentionToAddends(model: TinyTransformer, ids: number[]): { layer: number; head: number; mass: number }[] {
  // One forward populates lastPatterns on every attention module (see nn.ts).
  runWithCache(model, ids);
  const out: { layer: number; head: number; mass: number }[] = [];
  const seq = ids.length;
  for (let l = 0; l < model.cfg.nLayers; l++) {
    const patterns = model.blocks[l].attn.lastPatterns; // per head, flat (seq*seq)
    for (let h = 0; h < patterns.length; h++) {
      const flat = patterns[h];
      // row ANSWER_POS of the (seq,seq) matrix = the "=" query's attention over keys.
      const base = ANSWER_POS * seq;
      const mass = flat[base + 0] + flat[base + 1]; // mass landing on the two addend tokens
      out.push({ layer: l, head: h, mass });
    }
  }
  return out;
}

// ----------------------------------------------------------------------------
// Tool 2 (CORRELATIONAL): linear probe — is the answer linearly decodable here?
// ----------------------------------------------------------------------------
//
// We collect the residual stream AT the answer position across many sampled examples, labeled
//   by their true (a+b)%P, and train a linear probe. Accuracy >> baseline means the answer is
//   linearly present in that stream — strong correlational evidence the computation has
//   resolved by this point. The mandatory shuffled-label baseline (built into linearProbe)
//   guards against the classic "probe noise to above chance" trap.
function probeAnswerAt(model: TinyTransformer, task: Task, point: string, nSamples: number, rng: () => number) {
  const dim = model.cfg.dModel;
  const X = new Float64Array(nSamples * dim);
  const labels: number[] = [];
  for (let s = 0; s < nSamples; s++) {
    const batch = task.makeBatch(1, rng);
    const ids = batch.inputs[0];
    const { cache } = runWithCache(model, ids);
    const act = cache[point]; // (seq, dim)
    // copy the answer-position row into the probe design matrix
    for (let d = 0; d < dim; d++) X[s * dim + d] = act.data[ANSWER_POS * dim + d];
    labels.push((ids[0] + ids[1]) % P);
  }
  const acts = new Tensor(X, [nSamples, dim]);
  return linearProbe(acts, labels, P, rng, { epochs: 150, lr: 0.05 });
}

// ----------------------------------------------------------------------------
// Tool 3 (CORRELATIONAL): logit lens — at what depth does the answer emerge?
// ----------------------------------------------------------------------------
//
// We project the residual stream at the answer position through the final LN + unembed at each
//   stage of depth and read the clean answer token's lens-logit. A monotone climb across depth
//   localizes WHERE the computation resolves. This is correlational: the lens reads the stream,
//   it doesn't perturb it. (Caveat from interp.ts: early-layer lens logits use final-LN stats
//   off-distribution; trust the SHAPE, not the absolute early values.)
function logitLensTrace(model: TinyTransformer, ids: number[], answerToken: number): { point: string; logit: number }[] {
  const { cache } = runWithCache(model, ids);
  const points = ["embed"];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    points.push(`blocks.${l}.resid_mid`);
    points.push(`blocks.${l}.resid_post`);
  }
  const trace: { point: string; logit: number }[] = [];
  for (const p of points) {
    const resid = cache[p];
    if (!resid) continue;
    const lens = logitLens(model, resid); // (seq, vocab)
    trace.push({ point: p, logit: lens.data[ANSWER_POS * model.cfg.vocab + answerToken] });
  }
  return trace;
}

// ----------------------------------------------------------------------------
// Tool 4 (CAUSAL): activation patching — which component CARRIES the answer?
// ----------------------------------------------------------------------------
//
// The decisive experiment. For each candidate component we splice its corrupt-run value into
//   the clean run and measure recovery = fraction of the clean->corrupt logit gap reproduced.
//   Recovery ~1 means this component alone routes the addend information; ~0 means it is causally
//   irrelevant to THIS behavior regardless of how it correlates. We scan all attention heads'
//   outputs plus the per-layer attn_out / mlp_out to find the causal locus.
function patchScan(model: TinyTransformer, beh: Behavior): { point: string; recovery: number }[] {
  const points: string[] = [];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    for (let h = 0; h < model.cfg.nHeads; h++) points.push(`blocks.${l}.attn.head_z.${h}`);
    points.push(`blocks.${l}.attn_out`);
    points.push(`blocks.${l}.mlp_out`);
  }
  return points.map((point) => {
    const r = patch(model, beh.cleanIds, beh.corruptIds, point, ANSWER_POS, beh.cleanAnswer);
    return { point, recovery: r.recovery };
  });
}

// ----------------------------------------------------------------------------
// Tool 5 (CORRELATIONAL): SAE — does a sparse feature track the answer?
// ----------------------------------------------------------------------------
//
// We train a small overcomplete SAE on answer-position residuals, encode the held example, and
//   ask whether SOME feature fires selectively for the clean answer class. A feature whose top
//   activations cluster on one answer value is a candidate "answer feature". This is
//   correlational (a learned readout of existing structure), and on a tiny model it finds clean
//   atoms easily — the honesty note in interp.ts applies: the METHOD transfers, the pristine
//   result does not.
function saeAnswerFeature(
  model: TinyTransformer,
  task: Task,
  point: string,
  nSamples: number,
  rng: () => number,
): { featureSelectivity: number; nFeatures: number; reconLossFinal: number } {
  const dim = model.cfg.dModel;
  const X = new Float64Array(nSamples * dim);
  const labels: number[] = [];
  for (let s = 0; s < nSamples; s++) {
    const ids = task.makeBatch(1, rng).inputs[0];
    const { cache } = runWithCache(model, ids);
    const act = cache[point];
    for (let d = 0; d < dim; d++) X[s * dim + d] = act.data[ANSWER_POS * dim + d];
    labels.push((ids[0] + ids[1]) % P);
  }
  const acts = new Tensor(X, [nSamples, dim]);
  const nFeatures = dim * 2; // overcomplete dictionary
  const { sae, curve } = trainSAE(acts, nFeatures, rng, { steps: 400, lr: 0.01, l1: 0.03 });
  const codes = saeEncode(sae, acts); // (nSamples, nFeatures)
  // Selectivity: for the best feature, how concentrated is its activation on a single answer
  // class? We compute, per feature, the fraction of its TOTAL activation mass that falls on the
  // single most-activated class, then take the max over features. 1.0 = a perfectly
  // class-selective ("monosemantic for the answer") feature; ~1/P = uniform / uninformative.
  let best = 0;
  for (let f = 0; f < nFeatures; f++) {
    const perClass = new Float64Array(P);
    let total = 0;
    for (let s = 0; s < nSamples; s++) {
      const v = codes.data[s * nFeatures + f];
      perClass[labels[s]] += v;
      total += v;
    }
    if (total < 1e-9) continue; // dead feature: contributes no evidence
    let topClass = 0;
    for (let c = 1; c < P; c++) if (perClass[c] > perClass[topClass]) topClass = c;
    const selectivity = perClass[topClass] / total;
    if (selectivity > best) best = selectivity;
  }
  return { featureSelectivity: best, nFeatures, reconLossFinal: curve[curve.length - 1].reconLoss };
}

// ----------------------------------------------------------------------------
// Tool 6 (CAUSAL): ablation — is the component NECESSARY?
// ----------------------------------------------------------------------------
//
// Patching tests sufficiency-of-routing; ablation tests necessity: remove the component and
//   measure how far the clean answer logit drops. We use MEAN ablation (less off-distribution
//   than zero; zeroing a normalized stream can look catastrophic for reasons unrelated to the
//   circuit — see interp.ts). A large drop = necessary; ~0 drop = the model routes around it.
function ablationDrop(model: TinyTransformer, beh: Behavior, point: string): number {
  const r = ablate(model, beh.cleanIds, point, ANSWER_POS, beh.cleanAnswer, "mean");
  return r.drop;
}

/** The clean->corrupt logit gap on the clean answer token: the total "swing" the behavior is
 *  worth. Both patch recovery and ablation necessity are measured AGAINST this same scale, so
 *  the two causal numbers are comparable rather than living on different axes. */
function behaviorGap(model: TinyTransformer, beh: Behavior): number {
  const V = model.cfg.vocab;
  const clean = runWithCache(model, beh.cleanIds).logits.data[ANSWER_POS * V + beh.cleanAnswer];
  const corrupt = runWithCache(model, beh.corruptIds).logits.data[ANSWER_POS * V + beh.cleanAnswer];
  return clean - corrupt;
}

// ----------------------------------------------------------------------------
// Faithfulness aggregation.
// ----------------------------------------------------------------------------
//
// Faithfulness reads ONLY the causal rows, and reads them as TWO DISTINCT questions, not one
//   blended score:
//     - sufficiency = patch recovery  (does splicing this component in reproduce the behavior?)
//     - necessity   = ablation drop   (does removing it destroy the behavior?)
//   We deliberately DO NOT collapse these into a single product. On a redundant circuit a
//   component can be sufficient (recovery~1) yet only PARTIALLY necessary (ablation drop small,
//   because a parallel head re-derives the answer). A naive product would then label the
//   genuine circuit "unfaithful" — which is wrong; the honest reading is "sufficient + the
//   circuit is redundant". So the verdict is a 2-of-2 / 1-of-2 / 0-of-2 gate on the causal
//   rows, with redundancy reported, not punished. The decoy (both causal signals ~0) fails the
//   gate decisively — THAT is the contrast the stage exists to draw. Even this gate is a
//   SUMMARY, never a proof (see header honesty note).
interface FaithfulnessVerdict {
  sufficiency: number; // patch-recovery strength of the hypothesis
  necessity: number; // ablation-drop strength of the hypothesis
  causalPassed: number; // how many of the 2 causal rows clear their support bar
  redundant: boolean; // sufficient-but-not-necessary => parallel component re-derives answer
}
function faithfulness(rows: EvidenceRow[]): FaithfulnessVerdict {
  const suf = rows.find((r) => r.kind === "causal" && r.technique === "patching");
  const nec = rows.find((r) => r.kind === "causal" && r.technique === "ablation");
  const causal = rows.filter((r) => r.kind === "causal");
  const causalPassed = causal.filter((r) => r.supports).length;
  return {
    sufficiency: suf ? suf.strength : 0,
    necessity: nec ? nec.strength : 0,
    causalPassed,
    redundant: !!(suf?.supports && nec && !nec.supports),
  };
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function formatMatrix(rows: EvidenceRow[]): string {
  const lines: string[] = [];
  lines.push("  technique          kind           metric    support  strength");
  lines.push("  -----------------  -------------  --------  -------  --------");
  for (const r of rows) {
    lines.push(
      "  " +
        r.technique.padEnd(17) +
        "  " +
        r.kind.padEnd(13) +
        "  " +
        r.metric.toFixed(3).padStart(8) +
        "  " +
        (r.supports ? "  yes  " : "  NO   ") +
        "  " +
        r.strength.toFixed(3).padStart(8) +
        "   " +
        r.note,
    );
  }
  return lines.join("\n");
}

// ----------------------------------------------------------------------------
// Pipeline run on the genuine circuit.
// ----------------------------------------------------------------------------
function runHonestPipeline(model: TinyTransformer, task: Task, beh: Behavior): Hypothesis {
  const rng = mulberry32(20240608);
  console.log(
    `行为: clean=[${beh.cleanIds.join(",")}] 答案=${beh.cleanAnswer}  |  corrupt=[${beh.corruptIds.join(
      ",",
    )}] 答案=${beh.corruptAnswer}  (position ${ANSWER_POS})`,
  );

  // --- causal scan first: it decides the hypothesis (most causally responsible component) ---
  const patches = patchScan(model, beh);
  const topPatch = patches.reduce((a, b) => (Math.abs(b.recovery) > Math.abs(a.recovery) ? b : a));
  const hyp: Hypothesis = { point: topPatch.point, label: topPatch.point };
  console.log(`\n[假设] 因果最强组件 = ${hyp.label}  (patch recovery=${topPatch.recovery.toFixed(3)})`);
  console.log("       全组件 patch recovery 扫描 (因果证据):");
  console.log(
    asciiBar(
      patches.map((p) => ({ label: p.point.replace("blocks.", "b").replace(".attn.head_z.", ".h").replace(".attn_out", ".ao").replace(".mlp_out", ".mlp"), value: p.recovery })),
      { width: 34 },
    ),
  );

  const rows: EvidenceRow[] = [];

  // --- Tool 1: attention to addends for the hypothesized head (if it is a head) ---------
  const attn = attentionToAddends(model, beh.cleanIds);
  // find the attention mass of the hypothesized component if it is a head; else best head.
  const headMatch = hyp.point.match(/blocks\.(\d+)\.attn\.head_z\.(\d+)/);
  let attnForHyp: { layer: number; head: number; mass: number };
  if (headMatch) {
    const l = Number(headMatch[1]);
    const h = Number(headMatch[2]);
    attnForHyp = attn.find((a) => a.layer === l && a.head === h)!;
  } else {
    attnForHyp = attn.reduce((a, b) => (b.mass > a.mass ? b : a));
  }
  rows.push({
    technique: "attention",
    kind: "correlational",
    metric: attnForHyp.mass,
    supports: attnForHyp.mass > 0.3,
    strength: clamp01(attnForHyp.mass),
    note: `"="查询落在加数{0,1}上的注意力质量`,
  });

  // --- Tool 2: probe at the hypothesized component's residual neighborhood -------------
  // Probe the post-attention residual (resid_post) of the hypothesis layer; that stream is
  // where a head's contribution has been written into the residual.
  const probePoint = headMatch ? `blocks.${headMatch[1]}.resid_post` : "blocks.1.resid_post";
  const probe = probeAnswerAt(model, task, probePoint, 240, rng);
  rows.push({
    technique: "probe",
    kind: "correlational",
    metric: probe.accuracy,
    supports: probe.accuracy > probe.baselineAccuracy + 0.2,
    strength: clamp01((probe.accuracy - probe.baselineAccuracy) / (1 - probe.baselineAccuracy)),
    note: `答案线性可解码 (baseline=${probe.baselineAccuracy.toFixed(3)})`,
  });

  // --- Tool 3: logit lens depth trace --------------------------------------------------
  const lens = logitLensTrace(model, beh.cleanIds, beh.cleanAnswer);
  const lensClimb = lens[lens.length - 1].logit - lens[0].logit;
  console.log(`\n[logit lens] 答案 token logit 随深度的轨迹 (相关证据):`);
  console.log("  " + asciiSparkline(lens.map((p) => p.logit), { title: "answer-logit" }));
  rows.push({
    technique: "logit-lens",
    kind: "correlational",
    metric: lensClimb,
    supports: lensClimb > 0,
    strength: clamp01(lensClimb / 8), // crude normalization: ~8 logits = confident
    note: `答案 logit 从 embed 到末层的净增长`,
  });

  // --- Tool 4: patching (causal, already scanned) -------------------------------------
  rows.push({
    technique: "patching",
    kind: "causal",
    metric: topPatch.recovery,
    supports: topPatch.recovery > 0.5,
    strength: clamp01(topPatch.recovery),
    note: `clean<-corrupt 单组件恢复率 (充分性)`,
  });

  // --- Tool 5: SAE answer feature ------------------------------------------------------
  const sae = saeAnswerFeature(model, task, probePoint, 240, rng);
  rows.push({
    technique: "sae",
    kind: "correlational",
    metric: sae.featureSelectivity,
    supports: sae.featureSelectivity > 0.4,
    strength: clamp01((sae.featureSelectivity - 1 / P) / (1 - 1 / P)),
    note: `最优特征对单一答案类的选择性 (chance≈${(1 / P).toFixed(2)})`,
  });

  // --- Tool 6: ablation (causal) ------------------------------------------------------
  const drop = ablationDrop(model, beh, hyp.point);
  // Normalize necessity by the clean->corrupt logit GAP — the same behavior scale patching
  // uses — not by the raw clean logit. necessity = fraction of the behavior's logit gap that
  // mean-ablating this single component removes. On a redundant circuit this is honestly small
  // even for the genuine head, because a parallel head re-derives the answer; we surface that
  // as "redundant", not as failure (see faithfulness()).
  const gap = behaviorGap(model, beh);
  const necessity = clamp01(drop / Math.max(1e-6, gap));
  rows.push({
    technique: "ablation",
    kind: "causal",
    metric: drop,
    supports: necessity > 0.5,
    strength: necessity,
    note: `mean-ablation 删该头后答案 logit 的下降 (必要性, /gap=${gap.toFixed(2)})`,
  });

  console.log(`\n[证据矩阵] 假设: "${hyp.label} 计算了答案"`);
  console.log(formatMatrix(rows));

  const fa = faithfulness(rows);
  const corrSupport = rows.filter((r) => r.kind === "correlational" && r.supports).length;
  const corrTotal = rows.filter((r) => r.kind === "correlational").length;
  console.log(
    `\n[端到端忠实度] 因果两问分开看 (不合并成单一乘积):` +
      `\n  充分性 (patch recovery) = ${fa.sufficiency.toFixed(3)}` +
      `\n  必要性 (ablation drop)  = ${fa.necessity.toFixed(3)}` +
      `\n  因果支持 = ${fa.causalPassed}/2   相关支持 = ${corrSupport}/${corrTotal}`,
  );
  const verdict = fa.causalPassed === 2
    ? "✓ 充分且必要, 解释可信"
    : fa.redundant
      ? "◐ 充分但非唯一必要 ⇒ 电路冗余 (有并联组件重算答案); 解释方向对, 但「该头独占答案」过强"
      : "✗ 因果证据不足, 解释不可信";
  console.log(`  判定: ${verdict}`);
  return hyp;
}

// ----------------------------------------------------------------------------
// FAILURE MODE: a decoy that is correlationally green but causally red.
// ----------------------------------------------------------------------------
//
// We search the model for a component whose CORRELATIONAL evidence agrees with the answer
//   (attention high to addends AND probe-decodable) but whose CAUSAL evidence is ~0 (patch
//   recovery and ablation drop both near zero). Such a component is a CONFOUND: it co-varies
//   with the answer (because everything downstream of the real circuit does) yet does not
//   compute it. An explanation that stops at the correlational tools would name it and feel
//   well-supported — the matrix would be green — and be WRONG. Printing this proves that
//   correlation-agreement across multiple correlational tools is not causation; only the
//   intervention rows break the tie.
function runDecoyDemo(model: TinyTransformer, task: Task, beh: Behavior, genuine: Hypothesis): void {
  const rng = mulberry32(13371337);

  // Enumerate every head; score each on attention (correlational) and patch recovery (causal).
  // We use the SAME behavior pipeline A used, so the decoy and the genuine circuit are judged on
  // identical inputs — the contrast is then purely "which tool agrees", not "which example".
  const seq = beh.cleanIds.length;
  runWithCache(model, beh.cleanIds); // populate lastPatterns
  type Candidate = { point: string; attnMass: number; recovery: number };
  const cands: Candidate[] = [];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    const patterns = model.blocks[l].attn.lastPatterns;
    for (let h = 0; h < patterns.length; h++) {
      const base = ANSWER_POS * seq;
      const attnMass = patterns[h][base + 0] + patterns[h][base + 1];
      const point = `blocks.${l}.attn.head_z.${h}`;
      if (point === genuine.point) continue; // never pick the real causal head as the "decoy"
      const r = patch(model, beh.cleanIds, beh.corruptIds, point, ANSWER_POS, beh.cleanAnswer);
      cands.push({ point, attnMass, recovery: Math.abs(r.recovery) });
    }
  }

  // The decoy: among heads that are CAUSALLY irrelevant (recovery ~0), pick the one with the
  // HIGHEST attention to the addends. That is the trap — maximal correlational signal, zero
  // causal signal. Sorting by (attnMass - recovery) selects "looks most responsible, does
  // least". This head is causally downstream/parallel: its attention co-varies with the answer
  // because the real circuit already resolved it, not because this head computes it.
  const decoy = cands.reduce((a, b) => (b.attnMass - b.recovery > a.attnMass - a.recovery ? b : a));

  // Confirm with a probe that the decoy's residual neighborhood is ALSO correlationally green,
  // so this is a genuine multi-tool correlational agreement (not one weak tool).
  const lMatch = decoy.point.match(/blocks\.(\d+)\./)!;
  const probePoint = `blocks.${lMatch[1]}.resid_post`;
  const probe = probeAnswerAt(model, task, probePoint, 200, rng);
  const drop = ablationDrop(model, beh, decoy.point);
  const gap = behaviorGap(model, beh);

  const rows: EvidenceRow[] = [
    {
      technique: "attention",
      kind: "correlational",
      metric: decoy.attnMass,
      supports: decoy.attnMass > 0.3,
      strength: clamp01(decoy.attnMass),
      note: `注意力高 → 看起来"在读加数"`,
    },
    {
      technique: "probe",
      kind: "correlational",
      metric: probe.accuracy,
      supports: probe.accuracy > probe.baselineAccuracy + 0.2,
      strength: clamp01((probe.accuracy - probe.baselineAccuracy) / (1 - probe.baselineAccuracy)),
      note: `答案可线性解码 → 看起来"含答案"`,
    },
    {
      technique: "patching",
      kind: "causal",
      metric: decoy.recovery,
      supports: decoy.recovery > 0.5,
      strength: clamp01(decoy.recovery),
      note: `splice 进去几乎不动答案 → 不充分`,
    },
    {
      technique: "ablation",
      kind: "causal",
      metric: drop,
      supports: clamp01(drop / Math.max(1e-6, gap)) > 0.5,
      strength: clamp01(drop / Math.max(1e-6, gap)),
      note: `删掉它答案几乎不掉 → 不必要 (/gap=${gap.toFixed(2)})`,
    },
  ];

  console.log(`\n=== 失败模式: 相关全绿 / 因果全红的 decoy ===`);
  console.log(`decoy 组件 = ${decoy.point}  (区别于真因果头 ${genuine.point})`);
  console.log(`\n[证据矩阵] 假设: "${decoy.point} 计算了答案" (只看相关证据会接受)`);
  console.log(formatMatrix(rows));

  const corrSupport = rows.filter((r) => r.kind === "correlational" && r.supports).length;
  const corrTotal = rows.filter((r) => r.kind === "correlational").length;
  const fa = faithfulness(rows);
  console.log(
    `\n相关证据: ${corrSupport}/${corrTotal} 条支持 (若在此停手 → 错误地接受该解释)`,
  );
  console.log(
    `因果证据: 充分性=${fa.sufficiency.toFixed(3)} 必要性=${fa.necessity.toFixed(3)} (${fa.causalPassed}/2) → 否决`,
  );
  console.log(
    `教训: 多个相关工具一致 ≠ 因果成立。它们可能被同一个 confound 同时骗过 ` +
      `(decoy 与真电路并联/在其下游, 自然与答案共变); 只有干预 (patch/ablation) 能打破并列。` +
      `缺了干预环节的解释会"自洽地"骗过你。`,
  );
}

function main(): void {
  console.log("=== Stage 08: 合流与诚实 — 可证伪的解释流水线 ===\n");
  const task = modAdd(P);
  console.log(`任务: ${task.name}  vocab=${task.vocab}  研究对象 = 全书共享 checkpoint\n`);

  // Reuse the SAME pinned checkpoint every chapter dissects (model_zoo caches by config).
  const trained = trainToyModel(task, defaultTrainConfig(task));
  console.log(`checkpoint finalLoss=${trained.finalLoss.toFixed(4)}  (训练在 stage01 已实测计时)`);
  console.log(`激活点总数 (hook 名): ${activationNames(trained.model).length}`);

  // ONE behavior, shared by both pipelines — so the genuine circuit and the decoy are judged on
  // identical inputs and the contrast is purely about which tools agree.
  const beh = makeBehavior(mulberry32(20240608));

  console.log(`\n--- 流水线 A: 在真实电路上串联六工具 ---`);
  const genuine = runHonestPipeline(trained.model, task, beh);

  console.log(`\n--- 流水线 B: 故意制造"相关骗局" ---`);
  runDecoyDemo(trained.model, task, beh, genuine);

  console.log(
    `\n诚实边界: 此 toy 流水线干净可收敛 (单 position, 常单组件主导, recovery 接近 1)。` +
      `真模型上各工具经常互相矛盾 (probe 亮的地方 patch 找不到; ablation drop ≠ patch recovery), ` +
      `电路是分布式的, "解释是否忠实"本身仍是开放问题、无公认指标。` +
      `可迁移的是方法论形状: 永远用干预闭环、报告分歧而非挑一致的工具、把单一忠实度数字当摘要而非证明。`,
  );
}

main();
