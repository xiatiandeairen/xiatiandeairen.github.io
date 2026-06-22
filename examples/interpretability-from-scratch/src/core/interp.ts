// core/interp.ts — The interpretability toolbox: cache, logit lens, probes, patching,
//                   ablation, SAE feature reading. The verbs of the whole book.
//
// DESIGN PRINCIPLE: every tool here is built on the ONE hook mechanism in nn.ts. Caching is
//   "hook that observes and stores". Patching is "hook that returns a stored activation from
//   another run". Ablation is "hook that returns a zeroed/mean-replaced activation". Because
//   they share the hook taxonomy, a head named in one tool means the same thing in another —
//   that consistency is what lets findings corroborate across techniques.
//
// HONESTY (restated): absolute numbers from these tools on the toy model are optimistic. A
//   probe hitting 100% or a patch recovering 100% of the logit gap is normal on a tiny
//   synthetic circuit and does NOT imply real models are that clean. What transfers: the
//   SHAPE — logit lens monotonically approaching the answer across layers, patch recovery
//   concentrating on a few components, a probe beating its random baseline by a wide margin,
//   an SAE decomposing superposed activations into nameable features.

import { Tensor, noGrad } from "./autograd.js";
import { TinyTransformer, type Hooks } from "./nn.js";
import { AdamW, crossEntropy } from "./optim.js";
import { argmax, mulberry32, shuffle, type Rng } from "./rng.js";

// ----------------------------------------------------------------------------
// runWithCache — one forward, capture every named activation.
// ----------------------------------------------------------------------------
//
// Returns logits AND a cache mapping hook-name -> a DETACHED copy of the activation. We
//   detach so later analysis (probing, building a patch graph) cannot accidentally write
//   grads back into this forward's graph. Runs under noGrad: pure observation, no training
//   pollution. The cache is the raw material for every other tool.
export interface Cache {
  [name: string]: Tensor; // detached activations, addressed by hook name
}

export function runWithCache(model: TinyTransformer, ids: number[]): { logits: Tensor; cache: Cache } {
  const cache: Cache = {};
  const hooks: Hooks = {};
  // Register a passive cache hook on every activation point we know about for this model.
  for (const name of activationNames(model)) {
    hooks[name] = (n, v) => {
      cache[n] = v.detach(); // observe-only: store a detached snapshot, pass v onward
      return v;
    };
  }
  const logits = noGrad(() => model.forward(ids, hooks));
  return { logits, cache };
}

/** Enumerate all hook names this model exposes, in forward order. The single source of
 *  truth for "what activation points exist" — tools iterate this rather than hard-coding
 *  strings, so adding a layer automatically extends every tool's coverage. */
export function activationNames(model: TinyTransformer): string[] {
  const names = ["embed"];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    names.push(`blocks.${l}.resid_pre`);
    names.push(`blocks.${l}.attn_out`);
    names.push(`blocks.${l}.resid_mid`);
    for (let h = 0; h < model.cfg.nHeads; h++) names.push(`blocks.${l}.attn.head_z.${h}`);
    names.push(`blocks.${l}.mlp_out`);
    names.push(`blocks.${l}.resid_post`);
  }
  return names;
}

// ----------------------------------------------------------------------------
// logitLens — project an intermediate residual stream through the unembedding.
// ----------------------------------------------------------------------------
//
// IDEA: in a pre-norm transformer the residual stream is the running sum of every
//   component's output, and the unembedding reads the FINAL stream. The logit lens applies
//   the final LayerNorm + unembedding to an EARLIER stream, asking "if we stopped here,
//   what would the model predict?" Watching the predicted token approach the true answer
//   layer-by-layer reveals WHERE in depth the computation resolves.
// CAVEAT (honest): the lens uses the final LN's statistics on an off-distribution earlier
//   stream, so early-layer lens logits are approximate; the SHAPE (monotone approach) is the
//   robust signal, not the exact early probabilities.
export function logitLens(model: TinyTransformer, residStream: Tensor): Tensor {
  return noGrad(() => {
    const normed = model.lnFinal.forward(residStream);
    return model.unembed.forward(normed);
  });
}

// ----------------------------------------------------------------------------
// linearProbe — does a linear readout of an activation recover a label?
// ----------------------------------------------------------------------------
//
// A probe is a single Linear trained to predict labels from frozen activations. High probe
//   accuracy means the information is LINEARLY present at that point. The mandatory control:
//   train the SAME probe on SHUFFLED labels (or random features) to get the chance baseline;
//   a probe is only evidence if it beats that baseline by a wide margin. Without the control
//   you can "probe" pure noise to above-chance by overfitting — the classic probing failure.
export interface ProbeResult {
  accuracy: number;
  baselineAccuracy: number; // shuffled-label control trained identically
  nClasses: number;
  nSamples: number;
}

export function linearProbe(
  acts: Tensor, // (samples, dim) frozen activations
  labels: number[], // length samples, integer class ids
  nClasses: number,
  rng: Rng,
  opts: { epochs?: number; lr?: number } = {},
): ProbeResult {
  const { epochs = 200, lr = 0.05 } = opts;
  const [samples, dim] = acts.shape;
  if (labels.length !== samples) throw new Error(`linearProbe: labels ${labels.length} != samples ${samples}`);

  const train = (ys: number[]): number => {
    const W = Tensor.from([dim, nClasses], () => (rng() - 0.5) * 0.01);
    const b = Tensor.zeros([1, nClasses]);
    const opt = new AdamW([W, b], { lr });
    const X = acts.detach(); // freeze activations: probe must not change the model's signal
    for (let e = 0; e < epochs; e++) {
      W.zeroGrad();
      b.zeroGrad();
      const logits = X.matmul(W).add(b.broadcastRow(samples));
      const loss = crossEntropy(logits, ys);
      loss.backward();
      opt.step();
    }
    // final accuracy
    const finalLogits = noGrad(() => X.matmul(W).add(b.broadcastRow(samples)));
    let correct = 0;
    for (let r = 0; r < samples; r++) {
      const row = finalLogits.data.subarray(r * nClasses, (r + 1) * nClasses);
      if (argmax(row) === ys[r]) correct++;
    }
    return correct / samples;
  };

  const acc = train(labels);
  const shuffled = labels.slice();
  shuffle(shuffled, rng); // destroy the activation-label correspondence -> chance baseline
  const baseline = train(shuffled);
  return { accuracy: acc, baselineAccuracy: baseline, nClasses, nSamples: samples };
}

// ----------------------------------------------------------------------------
// patch — activation patching (a.k.a. causal tracing).
// ----------------------------------------------------------------------------
//
// THE CAUSAL EXPERIMENT: run the model on a CLEAN input (gives the right answer) and a
//   CORRUPT input (breaks it). Then re-run CLEAN but OVERWRITE one activation with the value
//   it had in the CORRUPT run. If that single substitution destroys the clean answer, that
//   activation is CAUSALLY responsible. We measure recovery as how much of the clean-vs-
//   corrupt logit gap the patch moves. Concentrated recovery on few components = a localized
//   circuit; diffuse recovery = distributed computation. This is correlation-proof evidence,
//   unlike a probe (which only shows information is present, not that it's USED).
export interface PatchResult {
  point: string;
  cleanLogit: number; // logit of `answerToken` at `position` in clean run
  corruptLogit: number; // same, corrupt run
  patchedLogit: number; // clean run with this point overwritten by corrupt's value
  recovery: number; // (patchedLogit - cleanLogit)/(corruptLogit - cleanLogit) in [~0..~1]
}

export function patch(
  model: TinyTransformer,
  cleanIds: number[],
  corruptIds: number[],
  point: string,
  position: number,
  answerToken: number,
): PatchResult {
  // Cache the corrupt activation we will splice in, and the baseline logits of both runs.
  const corrupt = runWithCache(model, corruptIds);
  const clean = runWithCache(model, cleanIds);
  const corruptAct = corrupt.cache[point];
  if (!corruptAct) throw new Error(`patch: unknown activation point "${point}"`);

  const cleanLogit = clean.logits.data[position * model.cfg.vocab + answerToken];
  const corruptLogit = corrupt.logits.data[position * model.cfg.vocab + answerToken];

  // Re-run clean, but replace `point` with the corrupt activation. The hook ignores the
  // observed value and returns the spliced one — that IS the intervention.
  const hooks: Hooks = {
    [point]: () => corruptAct,
  };
  const patchedLogits = noGrad(() => model.forward(cleanIds, hooks));
  const patchedLogit = patchedLogits.data[position * model.cfg.vocab + answerToken];

  // Recovery: fraction of the clean->corrupt drop reproduced by this single patch. ~1 means
  // this component alone explains the behavior; ~0 means it's irrelevant here.
  const denom = corruptLogit - cleanLogit;
  const recovery = Math.abs(denom) < 1e-9 ? 0 : (patchedLogit - cleanLogit) / denom;
  return { point, cleanLogit, corruptLogit, patchedLogit, recovery };
}

// ----------------------------------------------------------------------------
// ablate — zero or mean ablation of a component.
// ----------------------------------------------------------------------------
//
// Ablation asks the necessity question: remove a component and see how much the answer
//   degrades. ZERO ablation sets the activation to 0; MEAN ablation replaces it with its
//   average over a reference distribution (less off-distribution than zero — zeroing can
//   push the model into a regime it never sees, overstating importance). We return the drop
//   in the answer logit. FAILURE MODE this exposes: zero-ablation of a LayerNorm'd stream
//   can look catastrophic for reasons unrelated to the circuit (it breaks normalization),
//   which is why mean ablation is the more honest default.
export function ablate(
  model: TinyTransformer,
  ids: number[],
  point: string,
  position: number,
  answerToken: number,
  mode: "zero" | "mean" = "zero",
  meanValue?: Tensor,
): { point: string; baseLogit: number; ablatedLogit: number; drop: number } {
  const base = runWithCache(model, ids);
  const baseLogit = base.logits.data[position * model.cfg.vocab + answerToken];
  const ref = base.cache[point];
  if (!ref) throw new Error(`ablate: unknown point "${point}"`);

  const replacement =
    mode === "zero"
      ? Tensor.zeros(ref.shape)
      : meanValue ?? meanRows(ref); // mean over positions if no reference mean supplied

  const hooks: Hooks = { [point]: () => replacement };
  const ablatedLogits = noGrad(() => model.forward(ids, hooks));
  const ablatedLogit = ablatedLogits.data[position * model.cfg.vocab + answerToken];
  return { point, baseLogit, ablatedLogit, drop: baseLogit - ablatedLogit };
}

/** Mean over rows of a (rows, cols) tensor, broadcast back to (rows, cols). Used as the
 *  mean-ablation replacement: every position gets the across-position average. */
function meanRows(x: Tensor): Tensor {
  const [rows, cols] = x.shape;
  const mean = new Float64Array(cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) mean[c] += x.data[r * cols + c];
  for (let c = 0; c < cols; c++) mean[c] /= rows;
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) out.set(mean, r * cols);
  return new Tensor(out, [rows, cols]);
}

// ----------------------------------------------------------------------------
// SAE — sparse autoencoder for dictionary learning over activations.
// ----------------------------------------------------------------------------
//
// WHY: a neuron in superposition encodes several features at once, so reading neurons
//   directly is misleading. An SAE learns an OVERCOMPLETE dictionary (more features than
//   dims) with an L1 sparsity penalty, so each activation is reconstructed as a sparse sum
//   of dictionary atoms — and those atoms are often monosemantic ("nameable"). This is the
//   only tool here that builds NEW structure rather than reading existing components.
// HONESTY: on a tiny model with few real features the SAE easily finds clean atoms; at scale
//   features are messier and dead-feature / shrinkage problems dominate. The transferable
//   lesson is the METHOD (overcomplete + L1 -> sparse, more-interpretable basis), not the
//   pristine toy result.
export interface SAE {
  Wenc: Tensor; // (dim, nFeatures)
  bEnc: Tensor; // (1, nFeatures)
  Wdec: Tensor; // (nFeatures, dim)
  bDec: Tensor; // (1, dim)
  dim: number;
  nFeatures: number;
}

export interface SAETrainResult {
  sae: SAE;
  curve: { step: number; reconLoss: number; l1: number; activeFrac: number }[];
}

export function trainSAE(
  acts: Tensor, // (samples, dim)
  nFeatures: number,
  rng: Rng,
  opts: { steps?: number; lr?: number; l1?: number } = {},
): SAETrainResult {
  const { steps = 600, lr = 0.01, l1 = 0.05 } = opts;
  const [samples, dim] = acts.shape;
  const Wenc = Tensor.from([dim, nFeatures], () => (rng() - 0.5) * 0.1);
  const bEnc = Tensor.zeros([1, nFeatures]);
  const Wdec = Tensor.from([nFeatures, dim], () => (rng() - 0.5) * 0.1);
  const bDec = Tensor.zeros([1, dim]);
  const opt = new AdamW([Wenc, bEnc, Wdec, bDec], { lr });
  const X = acts.detach();
  const curve: SAETrainResult["curve"] = [];

  for (let step = 0; step < steps; step++) {
    Wenc.zeroGrad();
    bEnc.zeroGrad();
    Wdec.zeroGrad();
    bDec.zeroGrad();
    // encode: features = relu(X @ Wenc + bEnc) ; relu enforces non-negative codes (atoms add)
    const pre = X.matmul(Wenc).add(bEnc.broadcastRow(samples));
    const feats = pre.relu();
    // decode: recon = feats @ Wdec + bDec
    const recon = feats.matmul(Wdec).add(bDec.broadcastRow(samples));
    // recon loss = MSE(recon, X) implemented via (recon - X)^2 mean
    const diff = recon.sub(X);
    const reconLoss = diff.mul(diff).mean();
    // L1 sparsity on features (mean abs == mean since relu >=0). This is the term that forces
    // each input to use FEW atoms, which is what makes atoms monosemantic.
    const l1Loss = feats.sum().mulScalar(l1 / samples);
    const loss = reconLoss.add(l1Loss);
    loss.backward();
    opt.step();

    if (step % Math.max(1, Math.floor(steps / 20)) === 0 || step === steps - 1) {
      // measure active fraction = mean over features of "is this feature ever active?"
      let active = 0;
      for (let i = 0; i < feats.size; i++) if (feats.data[i] > 1e-6) active++;
      curve.push({
        step,
        reconLoss: reconLoss.data[0],
        l1: l1Loss.data[0],
        activeFrac: active / feats.size,
      });
    }
  }
  const sae: SAE = { Wenc, bEnc, Wdec, bDec, dim, nFeatures };
  return { sae, curve };
}

/** Encode activations through a trained SAE -> (samples, nFeatures) sparse codes. */
export function saeEncode(sae: SAE, acts: Tensor): Tensor {
  const samples = acts.shape[0];
  return noGrad(() => acts.matmul(sae.Wenc).add(sae.bEnc.broadcastRow(samples)).relu());
}

/** For a given SAE feature, return the indices of the top-k samples that most activate it.
 *  This is how you NAME a feature: look at what inputs fire it. Returns {sample, value}. */
export function topSAEfeatures(
  codes: Tensor, // (samples, nFeatures)
  feature: number,
  k: number,
): { sample: number; value: number }[] {
  const [samples, nFeatures] = codes.shape;
  if (feature < 0 || feature >= nFeatures) throw new Error(`topSAEfeatures: feature ${feature} OOB`);
  const scored: { sample: number; value: number }[] = [];
  for (let s = 0; s < samples; s++) scored.push({ sample: s, value: codes.data[s * nFeatures + feature] });
  scored.sort((a, b) => b.value - a.value);
  return scored.slice(0, k);
}

// Convenience re-export so stages don't import mulberry32 from two places.
export { mulberry32 };
