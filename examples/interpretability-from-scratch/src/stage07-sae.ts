// stage07-sae.ts — Sparse autoencoders: prying monosemantic features out of superposition.
//
// WHY this stage exists: a single MLP neuron in a trained transformer rarely means ONE thing
//   — it fires for several unrelated inputs (superposition packs more features than there are
//   dimensions). So "read the neuron" is a dead end. An overcomplete sparse autoencoder (SAE)
//   re-expresses the same activation as a sparse sum of MANY dictionary atoms; with enough
//   atoms and an L1 penalty, individual atoms tend to be monosemantic, i.e. NAMEABLE. This
//   stage trains such an SAE on a fixed checkpoint's activations and then does the three
//   things that separate "I trained an SAE" from "I understand a feature":
//     (1) measure the recon-vs-sparsity trade-off, dead-feature fraction, and L0 (avg active
//         features per sample) — the honest health metrics of a dictionary;
//     (2) NAME a feature by its top-activating inputs (asciiHeatmap of the inputs that fire it);
//     (3) prove a named feature CAUSALLY drives a prediction by intervening in feature space
//         (ablate / boost that one atom, reconstruct, splice back) and reading the logit move.
//   Then a FAILURE MODE: crank L1 to the extreme and watch the dictionary COLLAPSE — dead
//   features spike, reconstruction blows up, no features left to name.
//
// HONESTY (true for the whole chapter): the toy model has FEW real features and LOW
//   superposition, so the SAE finds clean, nameable atoms easily and a feature intervention
//   moves the logit a lot. Real models have astronomically many features, most SAE atoms stay
//   uninterpretable, and dead-feature/shrinkage pathologies dominate. The ABSOLUTE numbers
//   here (low dead fraction, large causal logit swing) are optimistic. What transfers is the
//   METHOD and the SHAPES: overcomplete+L1 -> sparser, more-interpretable basis; pushing L1
//   too hard -> collapse; a real feature -> a causal, not merely correlational, logit effect.

import { Tensor, noGrad } from "./core/autograd.js";
import { mulberry32 } from "./core/rng.js";
import { modAdd } from "./core/tasks.js";
import { type Hooks } from "./core/nn.js";
import { trainToyModel, defaultTrainConfig } from "./core/model_zoo.js";
import { runWithCache, trainSAE, saeEncode, topSAEfeatures, type SAE } from "./core/interp.js";
import { asciiHeatmap, asciiBar, asciiSparkline } from "./core/viz.js";

// The activation point we dissect. resid_post of the last block is where the MLP has already
// written its computation into the stream and the unembedding reads it next — the most
// "feature-bearing" point to dictionary-learn on for a predict-at-"=" task.
const LAYER = 1;
const POINT = `blocks.${LAYER}.resid_post`;
const SEED = 1234; // SAME seed family as the shared checkpoint, so findings are comparable.

// ----------------------------------------------------------------------------
// Build the activation dataset: for every (a,b) pair, the resid_post vector at the "=" row.
// ----------------------------------------------------------------------------
//
// WHY enumerate ALL pairs instead of sampling: modAdd(7) has only 49 pairs. Exhaustive
//   coverage makes the SAE dataset deterministic and lets us label each sample by its exact
//   answer (a+b)%p — which is what turns "top-activating inputs" into a NAMEABLE pattern and
//   lets us measure whether a feature tracks a specific answer class.
interface ActDataset {
  acts: Tensor; // (nSamples, dModel) resid_post at the "=" position
  labels: number[]; // (a+b) % p per sample — the ground-truth answer class
  pairs: [number, number][]; // (a,b) per sample, for naming features by their inputs
  p: number;
  eqPos: number; // the scorable "=" position index
  vocab: number;
}

function buildActDataset(model: ReturnType<typeof trainToyModel>["model"], p: number): ActDataset {
  const eqPos = 2; // modAdd seq is [a, b, =]; "=" is index 2 (the only scorable position)
  const rows: number[][] = [];
  const labels: number[] = [];
  const pairs: [number, number][] = [];
  const dModel = model.cfg.dModel;
  for (let a = 0; a < p; a++) {
    for (let b = 0; b < p; b++) {
      const ids = [a, b, p]; // p is the "=" delimiter token id (vocab = p+1)
      const { cache } = runWithCache(model, ids);
      const resid = cache[POINT]; // (seq, dModel)
      const row = Array.from(resid.data.subarray(eqPos * dModel, (eqPos + 1) * dModel));
      rows.push(row);
      labels.push((a + b) % p);
      pairs.push([a, b]);
    }
  }
  const flat = new Float64Array(rows.length * dModel);
  for (let r = 0; r < rows.length; r++) flat.set(rows[r], r * dModel);
  return { acts: new Tensor(flat, [rows.length, dModel]), labels, pairs, p, eqPos, vocab: model.cfg.vocab };
}

// ----------------------------------------------------------------------------
// Dictionary health metrics: dead fraction, L0, and reconstruction quality.
// ----------------------------------------------------------------------------
//
// These three numbers are how you judge an SAE before believing any feature story:
//   - deadFrac: atoms that NEVER fire on the whole dataset are wasted capacity (and a warning
//     that L1 is too high or training too short). High dead frac => the dictionary collapsed.
//   - L0: average number of atoms active per sample. The whole point is sparsity, so L0 should
//     be small relative to nFeatures; L0 ~ nFeatures means "not sparse" (L1 too low).
//   - reconMSE / normalized FVU: does the sparse code still reconstruct the activation? A
//     pretty-but-useless dictionary reconstructs poorly. We report Fraction of Variance
//     Unexplained so the scale is interpretable (0 = perfect, 1 = no better than the mean).
interface SaeHealth {
  deadFrac: number;
  l0: number; // avg active features per sample
  reconMse: number;
  fvu: number; // fraction of variance unexplained in [~0, ~1]
  activeFeatures: number;
}

function measureSaeHealth(sae: SAE, ds: ActDataset): SaeHealth {
  const codes = saeEncode(sae, ds.acts); // (nSamples, nFeatures)
  const [nSamples, nFeatures] = codes.shape;

  // L0 + which features ever fire.
  const everActive = new Array<boolean>(nFeatures).fill(false);
  let activeCount = 0;
  for (let s = 0; s < nSamples; s++) {
    for (let f = 0; f < nFeatures; f++) {
      if (codes.data[s * nFeatures + f] > 1e-6) {
        activeCount++;
        everActive[f] = true;
      }
    }
  }
  const l0 = activeCount / nSamples;
  const activeFeatures = everActive.filter(Boolean).length;
  const deadFrac = 1 - activeFeatures / nFeatures;

  // Reconstruction: recon = codes @ Wdec + bDec, then MSE vs the original activations.
  const recon = noGrad(() => codes.matmul(sae.Wdec).add(sae.bDec.broadcastRow(nSamples)));
  const dModel = ds.acts.shape[1];
  let sse = 0; // sum of squared errors
  let sst = 0; // total sum of squares around the per-dim mean (denominator for FVU)
  const mean = new Float64Array(dModel);
  for (let s = 0; s < nSamples; s++) for (let c = 0; c < dModel; c++) mean[c] += ds.acts.data[s * dModel + c];
  for (let c = 0; c < dModel; c++) mean[c] /= nSamples;
  for (let s = 0; s < nSamples; s++) {
    for (let c = 0; c < dModel; c++) {
      const i = s * dModel + c;
      const e = recon.data[i] - ds.acts.data[i];
      sse += e * e;
      const d = ds.acts.data[i] - mean[c];
      sst += d * d;
    }
  }
  const reconMse = sse / (nSamples * dModel);
  const fvu = sst < 1e-12 ? 0 : sse / sst;
  return { deadFrac, l0, reconMse, fvu, activeFeatures };
}

// ----------------------------------------------------------------------------
// Name a feature: which answer class do its top-activating inputs share?
// ----------------------------------------------------------------------------
//
// A feature is only "named" if its strongest inputs share an obvious property. For modAdd that
//   property is the answer (a+b)%p. We pick the feature with the sharpest answer-purity among
//   its top-k inputs (so we name a feature that actually IS something, not a diffuse one). The
//   purity number is the honesty check: a high purity means the atom genuinely tracks one
//   answer; a low purity would mean "I drew a heatmap but the feature isn't monosemantic."
interface NamedFeature {
  feature: number;
  answerClass: number; // the modal answer among its top-activating inputs
  purity: number; // fraction of top-k inputs whose answer == answerClass
  topSamples: number[]; // dataset rows that most activate it
}

function nameSharpestFeature(sae: SAE, ds: ActDataset, codes: Tensor, k: number): NamedFeature {
  const nFeatures = sae.nFeatures;
  let best: NamedFeature | null = null;
  for (let f = 0; f < nFeatures; f++) {
    const top = topSAEfeatures(codes, f, k);
    if (top[0].value < 1e-6) continue; // dead feature: nothing to name
    // modal answer class among the top-k inputs + its purity
    const counts = new Map<number, number>();
    for (const t of top) {
      const ans = ds.labels[t.sample];
      counts.set(ans, (counts.get(ans) ?? 0) + 1);
    }
    let modal = -1;
    let modalCount = 0;
    counts.forEach((c, ans) => {
      if (c > modalCount) {
        modalCount = c;
        modal = ans;
      }
    });
    const purity = modalCount / top.length;
    if (!best || purity > best.purity) {
      best = { feature: f, answerClass: modal, purity, topSamples: top.map((t) => t.sample) };
    }
  }
  if (!best) throw new Error("no live feature to name — dictionary fully collapsed");
  return best;
}

// ----------------------------------------------------------------------------
// Causal feature intervention: ablate or boost ONE atom, reconstruct, splice back.
// ----------------------------------------------------------------------------
//
// THE CAUSAL EXPERIMENT (the payoff). A probe shows information is PRESENT; this shows it is
//   USED. For a chosen input we: encode resid_post -> codes; set ONE feature's code to 0
//   (ablate) or scale it up (boost); decode back to a resid_post vector; and run the model
//   with that reconstructed vector spliced in at POINT. If killing the feature drops the logit
//   of the answer it's named for, that feature CAUSALLY drives that prediction. We compare
//   against the SAE's own reconstruction baseline (recon WITHOUT editing) so the measured
//   swing is attributable to the EDIT, not to SAE reconstruction error — the crucial control.
interface CausalResult {
  cleanLogit: number; // model's true answer logit (no SAE in the loop)
  reconLogit: number; // answer logit when we splice the UNEDITED SAE reconstruction (baseline)
  ablatedLogit: number; // answer logit when we zero the named feature before decoding
  boostedLogit: number; // answer logit when we amplify the named feature before decoding
  ablationDrop: number; // reconLogit - ablatedLogit (positive => feature was driving it up)
  boostGain: number; // boostedLogit - reconLogit
}

function interveneOnFeature(
  model: ReturnType<typeof trainToyModel>["model"],
  sae: SAE,
  ids: number[],
  eqPos: number,
  answerToken: number,
  feature: number,
): CausalResult {
  const vocab = model.cfg.vocab;
  const dModel = model.cfg.dModel;
  const readAnswerLogit = (logits: Tensor) => logits.data[eqPos * vocab + answerToken];

  // 1. Clean run: the model's real prediction, SAE not involved.
  const clean = runWithCache(model, ids);
  const cleanLogit = readAnswerLogit(clean.logits);
  const resid = clean.cache[POINT]; // (seq, dModel)

  // Encode -> (optionally edit) -> decode the "=" row, then build a full (seq, dModel)
  // replacement that is identical to the clean stream EXCEPT the "=" row is the SAE output.
  // We only touch the scorable row: editing other rows would conflate effects.
  const decodeRow = (edit: (code: Float64Array) => void): Tensor => {
    const eqRow = new Tensor(resid.data.slice(eqPos * dModel, (eqPos + 1) * dModel), [1, dModel]);
    const code = saeEncode(sae, eqRow); // (1, nFeatures)
    const edited = code.data.slice();
    edit(edited);
    const editedCode = new Tensor(edited, [1, nF(sae)]);
    const reconRow = noGrad(() => editedCode.matmul(sae.Wdec).add(sae.bDec)); // (1, dModel)
    // Splice the reconstructed "=" row into a copy of the clean stream.
    const full = resid.data.slice();
    full.set(reconRow.data.subarray(0, dModel), eqPos * dModel);
    return new Tensor(full, resid.shape);
  };

  const runWith = (replacement: Tensor): number => {
    const hooks: Hooks = { [POINT]: () => replacement };
    const logits = noGrad(() => model.forward(ids, hooks));
    return readAnswerLogit(logits);
  };

  const reconLogit = runWith(decodeRow(() => {})); // unedited reconstruction = baseline
  const ablatedLogit = runWith(decodeRow((c) => (c[feature] = 0))); // kill the feature
  const boostedLogit = runWith(decodeRow((c) => (c[feature] = c[feature] * 4 + 1))); // amplify it

  return {
    cleanLogit,
    reconLogit,
    ablatedLogit,
    boostedLogit,
    ablationDrop: reconLogit - ablatedLogit,
    boostGain: boostedLogit - reconLogit,
  };
}

function nF(sae: SAE): number {
  return sae.nFeatures;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
function main(): void {
  const task = modAdd(7);
  console.log("=== Stage 07: 稀疏自编码器 (SAE) — 从叠加里拆出单义特征 ===\n");

  // Shared checkpoint (cached by model_zoo; same object stage01 trained).
  const trained = trainToyModel(task, defaultTrainConfig(task));
  const model = trained.model;
  const ds = buildActDataset(model, task.vocab - 1); // p = vocab-1
  console.log(`研究对象: ${task.name}  采样点: ${POINT}  数据集: 全部 ${ds.acts.shape[0]} 个 (a,b) 对`);
  console.log(`激活维度 dModel=${model.cfg.dModel}  字典将过完备到 nFeatures > dModel\n`);

  // --- 1. Train SAEs across an L1 sweep -> the recon/sparsity trade-off curve. ----------
  // The central SAE trade-off: stronger L1 -> sparser codes (lower L0) but worse recon (higher
  // FVU) and more dead features. We sweep L1 and print all three so the reader SEES the knee,
  // not just one chosen point. nFeatures = 4*dModel makes the dictionary overcomplete.
  const nFeatures = model.cfg.dModel * 4;
  const l1Sweep = [0.001, 0.01, 0.05, 0.2];
  console.log(`[1] 重构-稀疏 权衡扫描 (nFeatures=${nFeatures}, 4x 过完备; 各 800 步训练):`);
  console.log(`    L1 越大 → 越稀疏 (L0↓) 但重构越差 (FVU↑) 且死特征越多`);
  const healths: { l1: number; h: SaeHealth }[] = [];
  for (const l1 of l1Sweep) {
    const rng = mulberry32(SEED); // SAME seed per run: only L1 differs -> a clean controlled sweep
    const { sae } = trainSAE(ds.acts, nFeatures, rng, { steps: 800, lr: 0.01, l1 });
    const h = measureSaeHealth(sae, ds);
    healths.push({ l1, h });
    console.log(
      `    L1=${l1.toString().padEnd(6)} | FVU=${h.fvu.toFixed(4)}  L0=${h.l0.toFixed(2).padStart(6)}  ` +
        `死特征=${(h.deadFrac * 100).toFixed(1).padStart(5)}%  存活=${h.activeFeatures}/${nFeatures}`,
    );
  }
  console.log(
    asciiBar(
      healths.map((x) => ({ label: `L1=${x.l1}`, value: x.h.l0 })),
      { title: "  L0 (平均每样本激活特征数) 随 L1 变化", width: 32 },
    ),
  );

  // --- 2. Pick a healthy SAE and NAME its sharpest feature. -----------------------------
  // Choose a moderate L1 from the sweep: sparse enough to be interpretable, not so sparse it
  // collapsed. We retrain it (deterministic) and name the feature whose top inputs are purest.
  const chosenL1 = 0.01;
  const rng = mulberry32(SEED);
  const { sae, curve } = trainSAE(ds.acts, nFeatures, rng, { steps: 800, lr: 0.01, l1: chosenL1 });
  const codes = saeEncode(sae, ds.acts);
  console.log(`\n[2] 选定 L1=${chosenL1} 的字典, 命名最尖锐的特征:`);
  console.log(`    ${asciiSparkline(curve.map((c) => c.reconLoss), { title: "  SAE 重构 loss" })}`);
  const named = nameSharpestFeature(sae, ds, codes, 7);
  console.log(
    `    特征 #${named.feature}: top-7 激活输入里 ${(named.purity * 100).toFixed(0)}% 的答案都是 ` +
      `(a+b)%${task.vocab - 1} = ${named.answerClass}  ⇒ 命名为「答案=${named.answerClass} 特征」`,
  );
  // Show the top-activating inputs as a heatmap: rows = top inputs, cols = [a, b, answer].
  // A reader can eyeball that the answer column is constant -> the feature IS monosemantic here.
  const heat: number[][] = named.topSamples.map((s) => {
    const [a, b] = ds.pairs[s];
    return [a, b, ds.labels[s]];
  });
  console.log(
    asciiHeatmap(heat, {
      title: `    特征 #${named.feature} 的 top-7 激活输入 (每行一个样本):`,
      rowLabels: named.topSamples.map((s) => `s${s}`),
      colLabels: ["a", "b", "ans"],
    }),
  );
  console.log(`    (上图 'ans' 列若整列同色 = 该特征只对一个答案类激活, 即单义)`);

  // --- 3. CAUSAL test: ablate / boost the named feature, read the logit move. -----------
  // Find an input whose true answer == the feature's named answer class, then intervene on
  // that ONE feature in SAE space. Baseline is the unedited reconstruction (controls for SAE
  // recon error). If ablation drops the answer logit and boosting raises it, the feature is
  // causally responsible — correlation-proof, unlike the naming step alone.
  let chosenIds: number[] | null = null;
  let answerToken = -1;
  for (let a = 0; a < ds.p && !chosenIds; a++) {
    for (let b = 0; b < ds.p; b++) {
      if ((a + b) % ds.p === named.answerClass) {
        chosenIds = [a, b, ds.p];
        answerToken = named.answerClass;
        break;
      }
    }
  }
  const cr = interveneOnFeature(model, sae, chosenIds!, ds.eqPos, answerToken, named.feature);
  console.log(`\n[3] 因果干预 (输入 [${chosenIds}], 答案 token=${answerToken}):`);
  console.log(
    asciiBar(
      [
        { label: "clean(真模型)", value: cr.cleanLogit },
        { label: "SAE重构(基线)", value: cr.reconLogit },
        { label: "消融该特征", value: cr.ablatedLogit },
        { label: "放大该特征", value: cr.boostedLogit },
      ],
      { title: `  答案 token=${answerToken} 的 logit`, width: 30 },
    ),
  );
  console.log(
    `    消融落差 = ${cr.ablationDrop.toFixed(3)} (>0 ⇒ 该特征在把答案 logit 撑高)  ` +
      `放大增益 = ${cr.boostGain.toFixed(3)}`,
  );
  // Verdict: only claim causality if BOTH directions move as predicted.
  const causal = cr.ablationDrop > 0.05 && cr.boostGain > 0;
  console.log(
    `    判定: ${causal ? "因果成立 ✓ 消融压低、放大抬高同一答案 logit" : "证据不足 — 双向未一致, 不能断言因果"}`,
  );

  // --- 4. FAILURE MODE: extreme L1 collapses the dictionary. ----------------------------
  // Push L1 absurdly high. The sparsity penalty overwhelms the reconstruction term: nearly
  // every feature is driven dead, codes go all-zero, and reconstruction degenerates to the
  // bias (FVU -> ~1). There is nothing left to name. This is the most common real SAE failure:
  // over-regularize and you "wash out" the dictionary instead of sharpening it.
  console.log(`\n[4] 失败模式: L1 惩罚拉到极端 (l1=5.0) → 字典塌缩:`);
  const rngBad = mulberry32(SEED);
  const { sae: badSae } = trainSAE(ds.acts, nFeatures, rngBad, { steps: 800, lr: 0.01, l1: 5.0 });
  const badHealth = measureSaeHealth(badSae, ds);
  console.log(
    `    死特征=${(badHealth.deadFrac * 100).toFixed(1)}%  存活=${badHealth.activeFeatures}/${nFeatures}  ` +
      `L0=${badHealth.l0.toFixed(2)}  FVU=${badHealth.fvu.toFixed(3)}`,
  );
  // Try to name a feature anyway — quantify how unnamed it is.
  let collapseNote: string;
  try {
    const badCodes = saeEncode(badSae, ds.acts);
    const badNamed = nameSharpestFeature(badSae, ds, badCodes, 7);
    collapseNote =
      badHealth.activeFeatures <= 1
        ? `仅剩 ${badHealth.activeFeatures} 个存活特征, 无法分解出多义结构`
        : `勉强存活但纯度/重构已崩 (FVU=${badHealth.fvu.toFixed(2)}), 命名的特征 #${badNamed.feature} 不再可信`;
  } catch {
    collapseNote = "全部特征死亡, nameSharpestFeature 抛错 — 没有特征可命名";
  }
  console.log(`    后果: ${collapseNote}`);
  console.log(`    教训: L1 不是越大越好; 过稀疏让字典塌缩, 拆不出特征 (欠拟合而非过拟合)。`);

  // --- Honest boundary. -----------------------------------------------------------------
  console.log(
    `\n诚实边界: toy 激活叠加程度低、真实特征少且与答案一一对应, 所以字典干净、命名容易、` +
      `因果干预 logit 摆动大且死特征少。真模型特征海量、大量 SAE 原子仍不可解释, 死特征/收缩问题更重。` +
      `可迁移的是方法与形状: 过完备+L1 → 更稀疏更可解释的基; L1 过大 → 塌缩; 真特征 → 因果而非相关的 logit 效应。`,
  );
}

main();
