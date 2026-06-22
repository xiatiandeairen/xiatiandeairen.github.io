// core/model_zoo.ts — Train (and in-process cache) the shared research objects.
//
// WHY a zoo with a fixed checkpoint: the whole book dissects the SAME one or two trained
//   models across chapters. If stage04 (logit lens) and stage05 (patching) trained their
//   own models with different seeds, their findings could not corroborate each other — the
//   reader could never check "does the patch land where the lens pointed?" A single pinned
//   checkpoint per task makes cross-chapter results comparable, which is the entire payoff
//   of doing many techniques on one object.
//
// DETERMINISM: training threads ONE seeded rng for init AND data, so trainToyModel(task,
//   cfg, seed) is a pure function of its arguments — bit-for-bit reproducible. The cache is
//   keyed on (task.name, cfg, seed, steps); a second call with the same key returns the
//   exact same trained model object (same weights in memory), not a retrain.
//
// HONESTY: we train to "good enough to study", not SOTA. The loss curve is real and the
//   convergence is real; absolute final loss is optimistic for the synthetic task. We
//   print the actual measured curve so the reader sees the real trajectory, not a claim.

import { TinyTransformer, type ModelConfig } from "./nn.js";
import { AdamW, clipGradNorm, cosineWarmup, crossEntropy } from "./optim.js";
import { mulberry32 } from "./rng.js";
import type { Task } from "./tasks.js";

export interface TrainConfig {
  model: ModelConfig;
  steps: number;
  batchSize: number;
  lr: number;
  weightDecay: number;
  warmup: number;
  clipNorm: number;
  seed: number;
  /** Log loss every `logEvery` steps into the returned curve. */
  logEvery: number;
}

export interface TrainedModel {
  model: TinyTransformer;
  lossCurve: { step: number; loss: number }[];
  finalLoss: number;
  config: TrainConfig;
}

// In-process cache. Key includes everything that affects the weights so a stale key can't
// silently return a model trained under different settings (a real foot-gun if you tweak
// cfg but forget to bump the key).
const _cache = new Map<string, TrainedModel>();

function cacheKey(task: Task, cfg: TrainConfig): string {
  return JSON.stringify({ task: task.name, ...cfg });
}

/**
 * Train one TinyTransformer on `task` deterministically. Loss is the mean cross-entropy
 * over SCORABLE positions only (see tasks.ts): scoring undetermined positions would add
 * irreducible noise to the curve and make "did it learn?" unanswerable.
 *
 * The training loop processes one sequence at a time (the model does 2-D per-sequence math)
 * and accumulates grads over the batch before stepping — equivalent to batched training but
 * matching the engine's 2-D matmul contract. SLOW vs a batched kernel; fine at toy scale.
 */
export function trainToyModel(task: Task, cfg: TrainConfig): TrainedModel {
  const key = cacheKey(task, cfg);
  const hit = _cache.get(key);
  if (hit) return hit;

  const rng = mulberry32(cfg.seed);
  const model = new TinyTransformer(cfg.model, rng);
  const params = model.parameters();
  const opt = new AdamW(params, { lr: cfg.lr, weightDecay: cfg.weightDecay });
  const lossCurve: { step: number; loss: number }[] = [];

  for (let step = 0; step < cfg.steps; step++) {
    opt.setLr(cosineWarmup(step, cfg.warmup, cfg.steps, cfg.lr));
    model.zeroGrad();
    const batch = task.makeBatch(cfg.batchSize, rng);
    let stepLoss = 0;
    let counted = 0;
    for (let b = 0; b < cfg.batchSize; b++) {
      const input = batch.inputs[b];
      const target = batch.targets[b];
      const scorable = task.scorablePositions(input);
      const logits = model.forward(input); // (seq, vocab)
      // Select only scorable rows of logits + their targets. We build a (k, vocab) sub-tensor
      // view by slicing rows so the loss is over learnable positions only.
      const subLogits = selectRows(logits, scorable);
      const subTargets = scorable.map((i) => target[i]);
      const loss = crossEntropy(subLogits, subTargets);
      // Scale by 1/batch so accumulated grads = mean-over-batch grad (matches batched train).
      const scaled = loss.mulScalar(1 / cfg.batchSize);
      scaled.backward();
      stepLoss += loss.data[0];
      counted++;
    }
    clipGradNorm(params, cfg.clipNorm);
    opt.step();
    const meanLoss = stepLoss / counted;
    if (step % cfg.logEvery === 0 || step === cfg.steps - 1) lossCurve.push({ step, loss: meanLoss });
  }

  const finalLoss = lossCurve[lossCurve.length - 1].loss;
  const trained: TrainedModel = { model, lossCurve, finalLoss, config: cfg };
  _cache.set(key, trained);
  return trained;
}

// Row selection with autograd: pick rows `idx` from a (rows, cols) tensor -> (k, cols).
// Adjoint scatters grad back to the selected rows. Used to train on scorable positions only.
import { Tensor } from "./autograd.js";
function selectRows(x: Tensor, idx: number[]): Tensor {
  const [, cols] = x.shape;
  const k = idx.length;
  const out = new Float64Array(k * cols);
  for (let r = 0; r < k; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = x.data[idx[r] * cols + c];
  const t = new Tensor(out, [k, cols], [x], "selectRows");
  t._backward = () => {
    for (let r = 0; r < k; r++) for (let c = 0; c < cols; c++) x.grad[idx[r] * cols + c] += t.grad[r * cols + c];
  };
  return t;
}

/** A sensible default training config for the toy transformer. Small enough for seconds on
 *  CPU; large enough to actually learn the planted structure. Stages can override fields. */
export function defaultTrainConfig(task: Task, overrides: Partial<TrainConfig> = {}): TrainConfig {
  return {
    model: {
      vocab: task.vocab,
      dModel: 32,
      nHeads: 4,
      nLayers: 2,
      dHidden: 64,
      maxSeq: task.seqLen,
    },
    steps: 400,
    batchSize: 32,
    lr: 3e-3,
    weightDecay: 1e-2,
    warmup: 40,
    clipNorm: 1.0,
    seed: 1234,
    logEvery: 25,
    ...overrides,
  };
}
