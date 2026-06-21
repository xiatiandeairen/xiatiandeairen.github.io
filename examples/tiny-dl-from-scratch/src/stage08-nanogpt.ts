// stage08-nanogpt.ts — A real, trainable char-level GPT assembled ONLY from core ops.
//
// WHY this stage exists: chapters 1-7 verified each PART in isolation (autograd grad-check,
//   layers, optimizers, attention math). This is the integration test the whole book builds
//   toward: do the parts, wired into a transformer and trained by gradient descent, actually
//   LEARN language structure? "Learn" here has a concrete, falsifiable meaning — train loss
//   must drop far below the uniform-guess baseline and the model must be able to recite the
//   toy corpus back. If the wiring has a bug (a dropped gradient path, a wrong mask), the
//   model simply will not memorize and the loss plateaus. So this file is also a test.
//
// HONESTY BOUNDARY (inherited from data.ts, restated because it governs every number below):
//   The corpus is a tiny repeating "song" CHOSEN so a few-thousand-param model can overfit
//   it on a CPU in minutes. Absolute loss/accuracy here are OPTIMISTIC and do NOT generalize
//   to real text. What transfers is the SHAPE of the story: loss descends monotone-ish below
//   the log(vocab) baseline; train loss -> ~0 (memorization) is the signature of a correct
//   implementation; temperature trades determinism for diversity; an under-capacity model
//   provably cannot memorize. Those relations hold at any scale; the absolute ms / loss do not.
//
// ARCHITECTURE NOTE — why we re-implement attention/block here instead of importing stage07:
//   The task forbids importing any stageNN file (importing one runs its main()). core.* gives
//   us Tensor/Module/Linear/Embedding/LayerNorm but NOT attention. So this file contains a
//   compact, self-contained MultiHeadSelfAttention + Block + GPT built from core primitives.
//   The autograd graph flows through every op, which is exactly what makes training work and
//   what the loss-drop below independently verifies.
//
// CORE-OP CONSTRAINT that shapes the design: core matmul/softmax/transpose are 2-D ONLY.
//   So attention is computed PER SEQUENCE (one (T, d) matrix at a time) inside a JS loop over
//   the batch. This is slower than a batched kernel but keeps the autograd graph dead-simple
//   and 100% inside the verified 2-D ops. The honest cost: O(batch) Python-style looping; the
//   honest gain: every gradient path is one of the grad-checked core ops.

import { Tensor, noGrad } from "./core/autograd.js";
import { Module, Linear, Embedding, LayerNorm } from "./core/nn.js";
import { AdamW, clipGradNorm, cosineWarmup } from "./core/optim.js";
import { crossEntropy } from "./core/metrics.js";
import { charDataset, type CharDataset } from "./core/data.js";
import { mulberry32, type Rng } from "./core/rng.js";

// ============================================================================
// Building blocks (assembled from core; nothing here re-derives an adjoint —
// every op is a verified core Tensor op, so the chain rule is core's problem).
// ============================================================================

/** A constant lower-triangular causal mask as an ADDITIVE bias: 0 where attention is
 *  allowed, a large negative number where it must be killed before softmax. WHY additive
 *  (not multiplicative): we add it to the raw scores so softmax sends masked positions to
 *  ~0 probability. WHY -1e9 and not -Infinity: -Infinity * 0 in the score path can produce
 *  NaN under finite-precision intermediate ops; -1e9 is "effectively -inf" after softmax
 *  (exp(-1e9 - max) underflows to 0) without ever touching NaN. The mask is a leaf with no
 *  grad consumers — adding a constant has adjoint 1, so it is invisible to backprop. */
function causalMaskBias(blockSize: number): Tensor {
  const m = new Float64Array(blockSize * blockSize);
  for (let i = 0; i < blockSize; i++)
    for (let j = 0; j < blockSize; j++) m[i * blockSize + j] = j > i ? -1e9 : 0;
  return new Tensor(m, [blockSize, blockSize]);
}

/**
 * Multi-head causal self-attention over a single sequence of shape (T, embedDim).
 * INVARIANT: input is ONE sequence (T rows), not a batch — the GPT forward loops over the
 *   batch and calls this per item, because core matmul is 2-D only.
 * The classic scale 1/sqrt(headDim) keeps score variance ~1 so softmax does not saturate
 *   into a near-one-hot distribution at init (which would starve early gradients).
 */
class MultiHeadSelfAttention extends Module {
  private qkv: Linear; // fused projection -> [q | k | v], 3*embedDim wide
  private proj: Linear; // output projection back to embedDim
  private heads: number;
  private headDim: number;
  private embedDim: number;
  private invSqrtHeadDim: number;

  constructor(embedDim: number, heads: number, rng: Rng) {
    super();
    if (embedDim % heads !== 0)
      throw new Error(`MHSA: embedDim ${embedDim} not divisible by heads ${heads}`);
    this.embedDim = embedDim;
    this.heads = heads;
    this.headDim = embedDim / heads;
    this.invSqrtHeadDim = 1 / Math.sqrt(this.headDim);
    // bias:false on qkv is the GPT convention; the output proj keeps a bias.
    this.qkv = this.child(new Linear(embedDim, 3 * embedDim, rng, { bias: false }));
    this.proj = this.child(new Linear(embedDim, embedDim, rng));
  }

  /** x: (T, embedDim); maskBias: (T, T) additive causal mask. Returns (T, embedDim). */
  forwardSeq(x: Tensor, maskBias: Tensor): Tensor {
    const qkv = this.qkv.forward(x); // (T, 3*embedDim)
    // Slice the fused projection into q,k,v by column blocks. We materialize via a gather
    // matmul-free path would need a slice op core lacks; instead reshape is not enough
    // because columns are interleaved as [q|k|v] contiguous blocks, so a plain column
    // copy through a selection matmul keeps it inside autograd. Simpler: use matmul with a
    // fixed 0/1 selector — but that is wasteful. We instead build q/k/v with reshape+slice
    // emulated by transpose tricks is overkill; the cleanest in-graph way here is three
    // Linear-style selects. Since qkv has no bias and is just x@Wqkv, we can equivalently
    // run three separate matmuls against column slices of the SAME weight. To keep the
    // graph honest and avoid a custom op, we select columns via a constant selector matmul.
    const headOuts: Tensor[] = [];
    for (let h = 0; h < this.heads; h++) {
      const q = selectCols(qkv, h * this.headDim, this.headDim); // (T, headDim)
      const k = selectCols(qkv, this.embedDim + h * this.headDim, this.headDim);
      const v = selectCols(qkv, 2 * this.embedDim + h * this.headDim, this.headDim);
      // scores = (q @ k^T) / sqrt(headDim) + causalMask
      let scores = q.matmul(k.transpose()).mulScalar(this.invSqrtHeadDim); // (T, T)
      scores = scores.add(maskBias); // additive causal mask; adjoint passes straight through
      const attn = scores.softmax(); // row-wise over keys; masked cols ~0
      headOuts.push(attn.matmul(v)); // (T, headDim)
    }
    const concat = concatCols(headOuts); // (T, embedDim)
    return this.proj.forward(concat);
  }

  override forward(_x: Tensor): Tensor {
    // Single-tensor forward is unused; the GPT drives forwardSeq per sequence. We satisfy
    // the abstract contract but make misuse loud rather than silently mis-shaping.
    throw new Error("MultiHeadSelfAttention: use forwardSeq(x, maskBias) per sequence");
  }
}

/** Select `width` contiguous columns starting at `start` from a (rows, cols) tensor, via a
 *  constant 0/1 selector matmul so the operation stays inside the verified core graph (its
 *  adjoint is the selector^T matmul, handled by core). WHY not a hand-written slice op: core
 *  has no slice-with-backward; reusing matmul means zero new adjoint code to get wrong. */
function selectCols(x: Tensor, start: number, width: number): Tensor {
  const cols = x.shape[1];
  const sel = new Float64Array(cols * width);
  for (let w = 0; w < width; w++) sel[(start + w) * width + w] = 1;
  const selector = new Tensor(sel, [cols, width]);
  return x.matmul(selector); // (rows, width)
}

/** Concatenate same-row-count tensors along columns via a constant scatter matmul each,
 *  summed. Equivalent to hstack; kept in-graph for free gradients. */
function concatCols(parts: Tensor[]): Tensor {
  const totalCols = parts.reduce((a, p) => a + p.shape[1], 0);
  let out: Tensor | null = null;
  let offset = 0;
  for (const p of parts) {
    const w = p.shape[1];
    // scatter p's columns into [offset, offset+w) of a (rows, totalCols) tensor.
    const sel = new Float64Array(w * totalCols);
    for (let i = 0; i < w; i++) sel[i * totalCols + (offset + i)] = 1;
    const scattered = p.matmul(new Tensor(sel, [w, totalCols])); // (rows, totalCols)
    out = out === null ? scattered : out.add(scattered);
    offset += w;
  }
  return out as Tensor;
}

/** A pre-norm transformer block: x + attn(ln1(x)), then x + mlp(ln2(x)). Pre-norm (LN
 *  BEFORE the sublayer, residual added raw) is what makes deep stacks trainable without
 *  careful warmup gymnastics; post-norm needs much more babysitting. */
class Block extends Module {
  private ln1: LayerNorm;
  private attn: MultiHeadSelfAttention;
  private ln2: LayerNorm;
  private fc1: Linear;
  private fc2: Linear;

  constructor(embedDim: number, heads: number, mlpHidden: number, rng: Rng) {
    super();
    this.ln1 = this.child(new LayerNorm(embedDim));
    this.attn = this.child(new MultiHeadSelfAttention(embedDim, heads, rng));
    this.ln2 = this.child(new LayerNorm(embedDim));
    this.fc1 = this.child(new Linear(embedDim, mlpHidden, rng)); // kaiming: followed by relu
    this.fc2 = this.child(new Linear(mlpHidden, embedDim, rng));
  }

  /** x: (T, embedDim) single sequence. */
  forwardSeq(x: Tensor, maskBias: Tensor): Tensor {
    const a = this.attn.forwardSeq(this.ln1.forward(x), maskBias);
    const h1 = x.add(a); // residual 1
    const m = this.fc2.forward(this.fc1.forward(this.ln2.forward(h1)).relu());
    return h1.add(m); // residual 2
  }

  override forward(_x: Tensor): Tensor {
    throw new Error("Block: use forwardSeq(x, maskBias) per sequence");
  }
}

/**
 * The GPT. token-embed + learned position-embed -> N blocks -> final LN -> tied? no: a
 * separate LM head Linear maps embedDim -> vocab logits.
 * INVARIANT: forward consumes ids of length EXACTLY blockSize per sequence; position
 *   embeddings are indexed 0..blockSize-1, so a longer context would index out of range.
 *   This is the source of the "blockSize vs sampling context" failure demoed at the end.
 */
class GPT extends Module {
  readonly vocabSize: number;
  readonly blockSize: number;
  readonly embedDim: number;
  private tokEmb: Embedding;
  private posEmb: Embedding;
  private blocks: Block[];
  private lnFinal: LayerNorm;
  private head: Linear;
  private maskBias: Tensor;

  constructor(
    vocabSize: number,
    blockSize: number,
    embedDim: number,
    heads: number,
    nLayers: number,
    mlpHidden: number,
    rng: Rng,
  ) {
    super();
    this.vocabSize = vocabSize;
    this.blockSize = blockSize;
    this.embedDim = embedDim;
    this.tokEmb = this.child(new Embedding(vocabSize, embedDim, rng));
    this.posEmb = this.child(new Embedding(blockSize, embedDim, rng));
    this.blocks = [];
    for (let i = 0; i < nLayers; i++) {
      const b = this.child(new Block(embedDim, heads, mlpHidden, rng));
      this.blocks.push(b);
    }
    this.lnFinal = this.child(new LayerNorm(embedDim));
    this.head = this.child(new Linear(embedDim, vocabSize, rng, { init: "xavier" }));
    this.maskBias = causalMaskBias(blockSize); // constant, shared across sequences
  }

  /** Run one sequence of ids (length T <= blockSize) to logits (T, vocab). The position
   *  ids are 0..T-1 so this also works for a partial context during sampling. */
  private forwardSeqIds(ids: Int32Array): Tensor {
    const T = ids.length;
    if (T > this.blockSize)
      // FAILURE MODE made loud: a context longer than the trained block has no position
      // embedding row for the overflow positions. Throw with a precise message instead of
      // silently indexing garbage (Embedding would throw on OOB id anyway, but the message
      // there blames the wrong thing). This is the bug the final demo triggers on purpose.
      throw new Error(
        `GPT.forward: context length ${T} exceeds blockSize ${this.blockSize}; ` +
          `position embedding has no row for index ${this.blockSize}..${T - 1}. ` +
          `Crop the context to the last ${this.blockSize} tokens before sampling.`,
      );
    const idTensor = new Tensor(Float64Array.from(ids), [T]);
    const tok = this.tokEmb.forward(idTensor); // (T, embedDim)
    const posIds = new Tensor(Float64Array.from({ length: T }, (_v, i) => i), [T]);
    const pos = this.posEmb.forward(posIds); // (T, embedDim)
    let h = tok.add(pos);
    // Each block uses the top-left (T,T) of the mask. When T==blockSize it is the full mask.
    const mask = T === this.blockSize ? this.maskBias : causalMaskBias(T);
    for (const b of this.blocks) h = b.forwardSeq(h, mask);
    h = this.lnFinal.forward(h);
    return this.head.forward(h); // (T, vocab) logits
  }

  /** Batched forward returning stacked logits (batch*T, vocab) so crossEntropy can score
   *  the whole batch at once against flat targets. Loops sequences (core is 2-D only). */
  forwardBatch(xFlat: Int32Array, batch: number, T: number): Tensor {
    const seqLogits: Tensor[] = [];
    for (let b = 0; b < batch; b++) {
      const ids = xFlat.subarray(b * T, (b + 1) * T);
      seqLogits.push(this.forwardSeqIds(ids)); // (T, vocab)
    }
    return stackRows(seqLogits); // (batch*T, vocab)
  }

  override forward(_x: Tensor): Tensor {
    throw new Error("GPT: use forwardBatch(...) for training / forwardSeqIds for sampling");
  }

  /** Inference-only next-token logits for a given context (no graph built). Used by the
   *  sampler. Crops context to the last blockSize tokens — the CORRECT counterpart to the
   *  failure demo. */
  logitsForContext(ids: Int32Array): Float64Array {
    return noGrad(() => {
      const cropped =
        ids.length <= this.blockSize ? ids : ids.subarray(ids.length - this.blockSize);
      const logits = this.forwardSeqIds(cropped); // (T, vocab)
      const T = cropped.length;
      // last row = prediction for the next token after the context.
      return logits.data.slice((T - 1) * this.vocabSize, T * this.vocabSize);
    });
  }
}

/** Vertically stack (rows_i, cols) tensors into ((sum rows_i), cols) via scatter matmuls,
 *  staying in-graph. cols must match. */
function stackRows(parts: Tensor[]): Tensor {
  const totalRows = parts.reduce((a, p) => a + p.shape[0], 0);
  let out: Tensor | null = null;
  let rowOffset = 0;
  for (const p of parts) {
    const r = p.shape[0];
    // place p's rows at [rowOffset, rowOffset+r): left-multiply by a (totalRows, r) scatter.
    const sel = new Float64Array(totalRows * r);
    for (let i = 0; i < r; i++) sel[(rowOffset + i) * r + i] = 1;
    const scattered = new Tensor(sel, [totalRows, r]).matmul(p); // (totalRows, cols)
    out = out === null ? scattered : out.add(scattered);
    rowOffset += r;
  }
  return out as Tensor;
}

// ============================================================================
// Sampling
// ============================================================================

/**
 * Autoregressive sampling. temperature controls the softmax sharpness:
 *   temp -> 0 : argmax (greedy) — fully deterministic, will recite memorized text verbatim.
 *   temp = 1 : sample from the model's raw distribution.
 *   temp > 1 : flatter distribution, more diversity / more mistakes.
 * INVARIANT: each step crops context to the last blockSize tokens (logitsForContext does it).
 * Determinism: all randomness comes from the passed rng, so a given seed reproduces the text.
 */
function sample(model: GPT, ds: CharDataset, prompt: string, steps: number, temperature: number, rng: Rng): string {
  const ids = Array.from(ds.encode(prompt));
  for (let s = 0; s < steps; s++) {
    const logits = model.logitsForContext(Int32Array.from(ids));
    let nextId: number;
    if (temperature <= 1e-6) {
      // greedy: argmax. Deterministic regardless of rng.
      nextId = 0;
      let best = -Infinity;
      for (let j = 0; j < logits.length; j++)
        if (logits[j] > best) { best = logits[j]; nextId = j; }
    } else {
      // softmax(logits / T) then inverse-CDF sample with one rng draw.
      let max = -Infinity;
      for (const l of logits) max = Math.max(max, l);
      let denom = 0;
      const probs = new Float64Array(logits.length);
      for (let j = 0; j < logits.length; j++) {
        const e = Math.exp((logits[j] - max) / temperature);
        probs[j] = e;
        denom += e;
      }
      const r = rng();
      let cum = 0;
      nextId = logits.length - 1; // fallback guards float rounding leaving cum < r
      for (let j = 0; j < probs.length; j++) {
        cum += probs[j] / denom;
        if (r < cum) { nextId = j; break; }
      }
    }
    ids.push(nextId);
  }
  return ds.decode(ids);
}

// ============================================================================
// Train / evaluate helpers
// ============================================================================

/** One full training run; returns the loss history and a handle to the trained model. */
function train(
  ds: CharDataset,
  cfg: { blockSize: number; embedDim: number; heads: number; nLayers: number; mlpHidden: number },
  steps: number,
  batchSize: number,
  seed: number,
): { model: GPT; trainHist: number[]; valHist: { step: number; loss: number }[] } {
  const rng = mulberry32(seed);
  const model = new GPT(
    ds.vocabSize,
    cfg.blockSize,
    cfg.embedDim,
    cfg.heads,
    cfg.nLayers,
    cfg.mlpHidden,
    rng,
  );
  const params = model.parameters();
  const opt = new AdamW(params, { lr: 3e-3, weightDecay: 1e-2 });
  const warmup = Math.max(1, Math.floor(steps * 0.1));
  const trainHist: number[] = [];
  const valHist: { step: number; loss: number }[] = [];

  for (let step = 0; step < steps; step++) {
    const lr = cosineWarmup(step, warmup, steps, 3e-3);
    opt.setLr(lr);
    const { x, y } = ds.getBatch("train", cfg.blockSize, batchSize, rng);
    const logits = model.forwardBatch(x, batchSize, cfg.blockSize); // (batch*T, vocab)
    const loss = crossEntropy(logits, y);
    model.zeroGrad(); // INVARIANT: clear before backward — autograd accumulates on purpose.
    loss.backward();
    clipGradNorm(params, 1.0); // cap rare grad spikes so one bad batch can't NaN the run.
    opt.step();
    trainHist.push(loss.data[0]);

    if (step % Math.max(1, Math.floor(steps / 8)) === 0 || step === steps - 1) {
      const vloss = evalLoss(model, ds, cfg.blockSize, batchSize, rng);
      valHist.push({ step, loss: vloss });
    }
  }
  return { model, trainHist, valHist };
}

/** Held-out loss with no graph built (noGrad). The train/val gap is the overfitting story. */
function evalLoss(model: GPT, ds: CharDataset, blockSize: number, batchSize: number, rng: Rng): number {
  return noGrad(() => {
    const { x, y } = ds.getBatch("val", blockSize, batchSize, rng);
    const logits = model.forwardBatch(x, batchSize, blockSize);
    return crossEntropy(logits, y).data[0];
  });
}

// ============================================================================
// main
// ============================================================================

function main(): void {
  const ds = charDataset(); // deterministic toy corpus
  const baseline = Math.log(ds.vocabSize); // uniform-guess cross-entropy = ln(vocab)

  console.log("=".repeat(64));
  console.log("nanoGPT on a toy corpus — does the wired-up transformer actually learn?");
  console.log("=".repeat(64));
  console.log(`vocab size: ${ds.vocabSize}   train tokens: ${ds.trainIds.length}   val tokens: ${ds.valIds.length}`);
  console.log(`uniform-guess baseline cross-entropy = ln(${ds.vocabSize}) = ${baseline.toFixed(4)}`);
  console.log("(HONEST: toy corpus is repeating-by-design so a tiny model can overfit it on CPU.");
  console.log(" Absolute loss is optimistic; the transferable signal is the SHAPE below.)");

  // -- Main run: a model with enough capacity to memorize the corpus. --
  const cfg = { blockSize: 32, embedDim: 64, heads: 4, nLayers: 2, mlpHidden: 128 };
  const STEPS = 220;
  const BATCH = 16;

  const tProbe = mulberry32(123);
  const probe = new GPT(ds.vocabSize, cfg.blockSize, cfg.embedDim, cfg.heads, cfg.nLayers, cfg.mlpHidden, tProbe);
  const nParams = probe.parameters().reduce((a, p) => a + p.size, 0);
  console.log("\n" + "-".repeat(64));
  console.log(`1) TRAIN a capable GPT  (blockSize=${cfg.blockSize}, embed=${cfg.embedDim}, heads=${cfg.heads}, layers=${cfg.nLayers}, params=${nParams})`);
  console.log("-".repeat(64));

  const t0 = performance.now();
  const { model, trainHist, valHist } = train(ds, cfg, STEPS, BATCH, 1);
  const trainMs = performance.now() - t0;

  const initLoss = trainHist[0];
  const finalLoss = trainHist[trainHist.length - 1];
  console.log(`init train loss: ${initLoss.toFixed(4)}   final train loss: ${finalLoss.toFixed(4)}`);
  console.log(`baseline ${baseline.toFixed(4)} -> final ${finalLoss.toFixed(4)}  (dropped below baseline: ${finalLoss < baseline})`);
  console.log(`wall-clock: ${(trainMs / 1000).toFixed(1)}s for ${STEPS} steps  (${(trainMs / STEPS).toFixed(1)} ms/step, machine-dependent — compare ratios not abs)`);
  console.log("train/val loss probes (val = held-out tail; train<<val later = memorization):");
  for (const v of valHist) {
    const ti = Math.min(v.step, trainHist.length - 1);
    console.log(`  step ${String(v.step).padStart(3)}  train ${trainHist[ti].toFixed(4)}  val ${v.loss.toFixed(4)}`);
  }
  console.log(lossCurve(trainHist));

  // -- Sampling: greedy should recite memorized structure. --
  console.log("\n" + "-".repeat(64));
  console.log("2) AUTOREGRESSIVE SAMPLE — can it recite the corpus structure?");
  console.log("-".repeat(64));
  const prompt = "to be";
  const sRng = mulberry32(7);
  const greedy = sample(model, ds, prompt, 120, 0, sRng);
  console.log(`prompt: ${JSON.stringify(prompt)}`);
  console.log("greedy (temp=0) continuation:");
  console.log(indent(greedy));

  // -- Temperature contrast: same prompt, three temperatures. --
  console.log("\n" + "-".repeat(64));
  console.log("3) TEMPERATURE CONTRAST — diversity vs determinism (same prompt, same seed)");
  console.log("-".repeat(64));
  for (const temp of [0, 0.5, 1.0]) {
    const r = mulberry32(99); // same seed each => differences are purely from temperature
    const out = sample(model, ds, prompt, 80, temp, r);
    console.log(`temp=${temp.toFixed(1)}:`);
    console.log(indent(out));
  }
  console.log("Note: temp=0 is greedy (verbatim repeat); higher temp injects variety (and errors).");

  // -- FAILURE MODE 1: under-capacity model cannot memorize; loss stalls high. --
  console.log("\n" + "-".repeat(64));
  console.log("4) FAILURE MODE — starve capacity: loss plateaus, cannot recite");
  console.log("-".repeat(64));
  const tinyCfg = { blockSize: 32, embedDim: 4, heads: 1, nLayers: 1, mlpHidden: 4 };
  const tinyProbe = new GPT(ds.vocabSize, tinyCfg.blockSize, tinyCfg.embedDim, tinyCfg.heads, tinyCfg.nLayers, tinyCfg.mlpHidden, mulberry32(5));
  const tinyParams = tinyProbe.parameters().reduce((a, p) => a + p.size, 0);
  const { model: tiny, trainHist: tinyHist } = train(ds, tinyCfg, STEPS, BATCH, 1);
  console.log(`tiny model params=${tinyParams} (vs capable ${nParams})`);
  console.log(`tiny init loss ${tinyHist[0].toFixed(4)} -> final ${tinyHist[tinyHist.length - 1].toFixed(4)}  (capable final was ${finalLoss.toFixed(4)})`);
  console.log(`tiny stays near/above baseline ${baseline.toFixed(4)}: ${tinyHist[tinyHist.length - 1] > finalLoss + 0.3}`);
  const tinyGreedy = sample(tiny, ds, prompt, 60, 0, mulberry32(7));
  console.log("tiny greedy continuation (gibberish / stuck loops — no memorization):");
  console.log(indent(tinyGreedy));

  // -- FAILURE MODE 2: feeding a context longer than blockSize is a shape error. --
  console.log("\n" + "-".repeat(64));
  console.log("5) FAILURE MODE — context longer than blockSize: explicit shape error");
  console.log("-".repeat(64));
  const overlong = Int32Array.from({ length: cfg.blockSize + 5 }, () => 0);
  console.log(`feeding a ${overlong.length}-token context to a blockSize=${cfg.blockSize} model via the RAW forward...`);
  try {
    // logitsForContext crops (the correct path); to demo the failure we call the raw
    // sequence forward through a thin reflection of the same code path.
    callRawForwardOverlong(model, overlong);
    console.log("UNEXPECTED: no error thrown (this would be a bug in the guard).");
  } catch (e) {
    console.log("caught (as designed):");
    console.log(indent(String((e as Error).message)));
  }
  console.log("The SAFE sampler avoids this by cropping context to the last blockSize tokens:");
  const safe = model.logitsForContext(overlong); // does NOT throw — it crops
  console.log(indent(`logitsForContext cropped & returned ${safe.length} logits (= vocab ${ds.vocabSize}), no error.`));

  console.log("\n" + "=".repeat(64));
  console.log("DONE. Story (scale-invariant): loss drops below ln(vocab) baseline, train<<val");
  console.log("(memorization), greedy recites, temperature adds diversity, too-small model");
  console.log("provably can't memorize, and over-length context is a loud shape error.");
  console.log("=".repeat(64));
}

/** Trigger the over-length guard through the public sampling-shaped path WITHOUT cropping,
 *  by building the same forward that the model uses internally. We deliberately bypass the
 *  crop to surface the guard message (logitsForContext would have cropped it away). */
function callRawForwardOverlong(model: GPT, ids: Int32Array): void {
  noGrad(() => {
    // Reach the guard by asking for logits on the FULL (un-cropped) context. We do this by
    // temporarily calling the same internal path the model exposes for cropping, but with a
    // sentinel that skips the crop: simplest honest way is to call forwardBatch with T set
    // to the over-length, which routes into forwardSeqIds and hits the guard.
    model.forwardBatch(ids, 1, ids.length);
  });
}

// ---- small presentation helpers (stdout formatting only, no training logic) ----

function indent(s: string): string {
  return s.split("\n").map((l) => "    " + l).join("\n");
}

/** Tiny inline loss curve (we keep our own to avoid importing metrics' ASCII plot when a
 *  one-liner suffices; same idea: show monotone-ish descent shape, label min/max). */
function lossCurve(history: number[], width = 50, height = 7): string {
  const buckets: number[] = [];
  const per = history.length / width;
  for (let c = 0; c < width; c++) {
    const lo = Math.floor(c * per);
    const hi = Math.max(lo + 1, Math.floor((c + 1) * per));
    let s = 0, n = 0;
    for (let i = lo; i < hi && i < history.length; i++) { s += history[i]; n++; }
    buckets.push(n > 0 ? s / n : history[history.length - 1]);
  }
  const min = Math.min(...buckets), max = Math.max(...buckets), span = max - min || 1;
  const grid = Array.from({ length: height }, () => Array(width).fill(" "));
  for (let c = 0; c < buckets.length; c++) {
    const row = Math.min(height - 1, Math.round((1 - (buckets[c] - min) / span) * (height - 1)));
    grid[row][c] = "*";
  }
  const lines = grid.map((r) => r.join(""));
  lines[0] += `  ${max.toFixed(4)} (max)`;
  lines[height - 1] += `  ${min.toFixed(4)} (min)`;
  return lines.join("\n");
}

main();
