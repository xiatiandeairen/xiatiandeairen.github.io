// core/model.ts — a toy decoder-only Transformer: synthetic weights, two forward
// paths, and the KV-cache type the optimization stages mutate.
//
// What this is NOT: a trained model. Every weight is generated from a seeded PRNG.
// The book says this out loud and repeatedly: the output is "structurally correct
// gibberish" — the shapes, the causal masking, the RoPE rotations, the cache
// bookkeeping are all real and correct, but the model has learned nothing, so the
// tokens it emits are noise. THAT IS THE POINT. This book is about the inference
// *engine*, and an engine's correctness (does the cache match the no-cache path?
// does batching change outputs? does int8 drift the logits?) is completely
// independent of whether the model is any good. A trained model would only add a
// multi-GB download and obscure the mechanism.
//
// Architecture: a small LLaMA-style decoder — RMSNorm (pre-norm), RoPE attention
// with GQA support, SiLU-gated FFN, weight-tied output head. Chosen because it is
// the lineage every modern open inference engine targets.
//
// Determinism invariant: buildModel(cfg, seed) is a pure function of (cfg, seed).
// Same inputs -> bit-for-bit identical weights -> bit-for-bit identical logits, on
// any machine. This is what makes every cross-stage comparison (cache vs no-cache,
// quantized vs not) a controlled experiment rather than an anecdote.

import {
  type Tensor,
  tensor,
  zeros,
  matmul,
  rmsNorm,
  softmax,
  silu,
  rope,
} from "./tensor.js";

export type ModelConfig = {
  dModel: number;
  nLayers: number;
  nHeads: number;
  dHead: number;
  dFF: number;
  vocabSize: number;
  maxSeq: number;
  // GQA: number of KEY/VALUE heads. When nKVHeads < nHeads, each KV head is shared
  // by (nHeads / nKVHeads) query heads — this is what shrinks the KV cache. Set
  // nKVHeads === nHeads for vanilla multi-head attention (MHA).
  nKVHeads: number;
};

// The book's baseline config. Tiny on purpose: small enough that the unoptimized
// float64 kernels run in milliseconds (so a stage finishes interactively) yet large
// enough that prefill-vs-decode and cache-vs-no-cache differences are clearly
// measurable. nKVHeads(2) < nHeads(4) so GQA is exercised by default.
export const DEFAULT_CONFIG: ModelConfig = {
  dModel: 64,
  nLayers: 4,
  nHeads: 4,
  dHead: 16,
  dFF: 256,
  vocabSize: 256, // byte-level, matches core/tokenizer
  maxSeq: 256,
  nKVHeads: 2,
};

// mulberry32 — a tiny, fast, seedable PRNG. NOT cryptographic; we want exactly the
// opposite: full reproducibility from a 32-bit seed. Returns floats in [0, 1).
// This single function is the root of the book's determinism guarantee.
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// One Transformer block's weights. Names mirror the math so stage code reads like
// the architecture diagram. wQ/wK/wV/wO are attention projections; w1/w2/w3 are the
// SiLU-gated FFN (w1=gate, w3=up, w2=down). normAtt/normFFN are the pre-norm RMSNorm
// gains. Biases are omitted (LLaMA-style) — fewer params, and bias-free is the norm.
export type LayerWeights = {
  wQ: Tensor; // [dModel, nHeads * dHead]
  wK: Tensor; // [dModel, nKVHeads * dHead]   <- GQA: narrower than wQ
  wV: Tensor; // [dModel, nKVHeads * dHead]
  wO: Tensor; // [nHeads * dHead, dModel]
  w1: Tensor; // [dModel, dFF]   gate
  w3: Tensor; // [dModel, dFF]   up
  w2: Tensor; // [dFF, dModel]   down
  normAtt: Tensor; // [dModel]
  normFFN: Tensor; // [dModel]
};

export type Model = {
  cfg: ModelConfig;
  embed: Tensor; // [vocabSize, dModel] — also reused as the output head (weight tying)
  normOut: Tensor; // [dModel] final RMSNorm gain
  layers: LayerWeights[];
};

// Initialize a weight matrix with values in [-scale, scale). scale ~ 1/sqrt(fanIn)
// (a Xavier-ish heuristic) keeps activations from exploding through nLayers of
// matmuls — without it, the synthetic logits overflow and softmax produces NaN,
// which is itself a nice cautionary tale but not the default we want.
function randMat(rng: () => number, rows: number, cols: number, fanIn: number): Tensor {
  const scale = 1 / Math.sqrt(fanIn);
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < data.length; i++) data[i] = (rng() * 2 - 1) * scale;
  return tensor(data, [rows, cols]);
}

// RMSNorm gains init to ~1 (the identity for a normalizer) with a touch of jitter
// so the layers are not all identical.
function randGain(rng: () => number, n: number): Tensor {
  const data = new Float64Array(n);
  for (let i = 0; i < n; i++) data[i] = 1 + (rng() - 0.5) * 0.02;
  return tensor(data, [n]);
}

export function buildModel(cfg: ModelConfig, seed: number): Model {
  const rng = mulberry32(seed);
  const qDim = cfg.nHeads * cfg.dHead;
  const kvDim = cfg.nKVHeads * cfg.dHead;
  if (cfg.nHeads % cfg.nKVHeads !== 0) {
    // GQA invariant: query heads must split evenly across KV heads, else the
    // head-grouping in attention has a remainder and is ill-defined.
    throw new Error(`buildModel: nHeads(${cfg.nHeads}) must be divisible by nKVHeads(${cfg.nKVHeads})`);
  }
  const layers: LayerWeights[] = [];
  for (let l = 0; l < cfg.nLayers; l++) {
    layers.push({
      wQ: randMat(rng, cfg.dModel, qDim, cfg.dModel),
      wK: randMat(rng, cfg.dModel, kvDim, cfg.dModel),
      wV: randMat(rng, cfg.dModel, kvDim, cfg.dModel),
      wO: randMat(rng, qDim, cfg.dModel, qDim),
      w1: randMat(rng, cfg.dModel, cfg.dFF, cfg.dModel),
      w3: randMat(rng, cfg.dModel, cfg.dFF, cfg.dModel),
      w2: randMat(rng, cfg.dFF, cfg.dModel, cfg.dFF),
      normAtt: randGain(rng, cfg.dModel),
      normFFN: randGain(rng, cfg.dModel),
    });
  }
  return {
    cfg,
    embed: randMat(rng, cfg.vocabSize, cfg.dModel, cfg.dModel),
    normOut: randGain(rng, cfg.dModel),
    layers,
  };
}

// Per-layer KV cache: the keys and values already computed for past tokens, ready
// to be reused. K and V each hold [nKVHeads * dHead] per cached position, stored as
// a flat row-major [len, kvDim] block. `len` is how many positions are filled.
//
// This type is the protagonist of the whole book. stage02 builds it; stage05 swaps
// the flat array for paged blocks; the math is in core/metrics.estimateKVBytes.
export type KVCache = {
  k: Float64Array[]; // per layer: flat [len * kvDim] (we grow by pushing rows)
  v: Float64Array[];
  len: number; // number of positions currently cached (same for all layers)
  cfg: ModelConfig;
};

export function newCache(cfg: ModelConfig): KVCache {
  return {
    k: Array.from({ length: cfg.nLayers }, () => new Float64Array(0)),
    v: Array.from({ length: cfg.nLayers }, () => new Float64Array(0)),
    len: 0,
    cfg,
  };
}

// --- shared attention/FFN math, parameterized by how K/V are sourced ---------
//
// Both forward paths (cached and not) share this block-evaluation core so the two
// can NEVER silently diverge — the only difference between them is which keys/values
// the attention sees, which is exactly the variable the book is studying. Keeping
// the arithmetic identical is what makes "cache == no-cache" a meaningful assertion.

// Compute Q,K,V projections for a single token's hidden vector h (length dModel).
function projectQKV(
  h: Float64Array,
  w: LayerWeights,
  cfg: ModelConfig
): { q: Float64Array; k: Float64Array; v: Float64Array } {
  const hT = tensor(h, [1, cfg.dModel]);
  const q = matmul(hT, w.wQ).data;
  const k = matmul(hT, w.wK).data;
  const v = matmul(hT, w.wV).data;
  return { q, k, v };
}

// Apply RoPE per-head to a token's q (nHeads heads) and k (nKVHeads heads) at `pos`.
function applyRope(q: Float64Array, k: Float64Array, pos: number, cfg: ModelConfig): void {
  // q has nHeads heads, k has nKVHeads heads, each dHead wide. RoPE rotates within
  // a head. A subtle invariant: query head g and the kv head it maps to (g div
  // groupSize) MUST be rotated by the same pos, or the relative-position property
  // breaks and attention scores go wrong. Same `pos` for all heads guarantees it.
  for (let hh = 0; hh < cfg.nHeads; hh++) {
    const qSlice = q.subarray(hh * cfg.dHead, (hh + 1) * cfg.dHead);
    // pair each query head with ANY key head for the rope() signature; we only need
    // q rotated here, so reuse a throwaway view of the corresponding kv slot.
    const kvHead = Math.floor(hh / (cfg.nHeads / cfg.nKVHeads));
    const kSlice = k.subarray(kvHead * cfg.dHead, (kvHead + 1) * cfg.dHead);
    // rope mutates both; but a kv head shared by G query heads would be rotated G
    // times. To avoid that, rotate k separately below and pass a scratch copy here.
    const kScratch = Float64Array.from(kSlice);
    rope(qSlice, kScratch, pos, 10000);
  }
  for (let kh = 0; kh < cfg.nKVHeads; kh++) {
    const kSlice = k.subarray(kh * cfg.dHead, (kh + 1) * cfg.dHead);
    const qScratch = Float64Array.from(kSlice); // scratch, rotation symmetric
    rope(qScratch, kSlice, pos, 10000);
  }
}

// Attention for one query token against `len` cached (K,V) rows + the layer output.
// kAll/vAll are flat [len * kvDim]. Returns the attention output [qDim] for this token.
function attendToken(
  q: Float64Array,
  kAll: Float64Array,
  vAll: Float64Array,
  len: number,
  cfg: ModelConfig
): Float64Array {
  const groupSize = cfg.nHeads / cfg.nKVHeads;
  const out = new Float64Array(cfg.nHeads * cfg.dHead);
  const invSqrtD = 1 / Math.sqrt(cfg.dHead);
  for (let hh = 0; hh < cfg.nHeads; hh++) {
    const kvHead = Math.floor(hh / groupSize); // GQA: which KV head this query reads
    const qOff = hh * cfg.dHead;
    // scores against every cached position (causal: caller only ever caches the past
    // plus the current token, so no explicit mask needed here — the cache *is* the mask).
    const scores = new Float64Array(len);
    for (let p = 0; p < len; p++) {
      const kOff = p * (cfg.nKVHeads * cfg.dHead) + kvHead * cfg.dHead;
      let dotv = 0;
      for (let d = 0; d < cfg.dHead; d++) dotv += q[qOff + d] * kAll[kOff + d];
      scores[p] = dotv * invSqrtD;
    }
    const probs = softmax(tensor(scores, [1, len])).data;
    for (let p = 0; p < len; p++) {
      const vOff = p * (cfg.nKVHeads * cfg.dHead) + kvHead * cfg.dHead;
      const w = probs[p];
      for (let d = 0; d < cfg.dHead; d++) out[qOff + d] += w * vAll[vOff + d];
    }
  }
  return out;
}

// FFN: SiLU-gated MLP. down( silu(gate(x)) * up(x) ).
function ffn(h: Float64Array, w: LayerWeights, cfg: ModelConfig): Float64Array {
  const hT = tensor(h, [1, cfg.dModel]);
  const gate = silu(matmul(hT, w.w1));
  const up = matmul(hT, w.w3);
  const prod = new Float64Array(cfg.dFF);
  for (let i = 0; i < cfg.dFF; i++) prod[i] = gate.data[i] * up.data[i];
  return matmul(tensor(prod, [1, cfg.dFF]), w.w2).data;
}

// Run one full block for one token, given the running K/V history. Mutates kHist/
// vHist by appending this token's (rope'd) K and V, then attends over the whole
// history. Returns the new hidden vector. This is the per-token unit both paths use.
function blockStep(
  h: Float64Array,
  w: LayerWeights,
  pos: number,
  kHist: Float64Array,
  vHist: Float64Array,
  histLen: number,
  cfg: ModelConfig
): { h: Float64Array; k: Float64Array; v: Float64Array } {
  // pre-norm
  const normed = rmsNorm(tensor(h, [1, cfg.dModel]), w.normAtt).data;
  const { q, k, v } = projectQKV(normed, w, cfg);
  applyRope(q, k, pos, cfg);
  // the attention sees history[0..histLen) plus THIS token at index histLen.
  const kvDim = cfg.nKVHeads * cfg.dHead;
  const kAll = new Float64Array((histLen + 1) * kvDim);
  const vAll = new Float64Array((histLen + 1) * kvDim);
  kAll.set(kHist.subarray(0, histLen * kvDim));
  vAll.set(vHist.subarray(0, histLen * kvDim));
  kAll.set(k, histLen * kvDim);
  vAll.set(v, histLen * kvDim);
  const attn = attendToken(q, kAll, vAll, histLen + 1, cfg);
  const attnOut = matmul(tensor(attn, [1, cfg.nHeads * cfg.dHead]), w.wO).data;
  // residual
  const h1 = new Float64Array(cfg.dModel);
  for (let i = 0; i < cfg.dModel; i++) h1[i] = h[i] + attnOut[i];
  // FFN sub-block, pre-norm + residual
  const normed2 = rmsNorm(tensor(h1, [1, cfg.dModel]), w.normFFN).data;
  const ff = ffn(normed2, w, cfg);
  const h2 = new Float64Array(cfg.dModel);
  for (let i = 0; i < cfg.dModel; i++) h2[i] = h1[i] + ff[i];
  return { h: h2, k, v };
}

// Project a final hidden vector to vocab logits via the tied embedding matrix.
// embed is [vocab, dModel]; logits = h @ embed^T -> [vocab]. Weight tying (reusing
// the input embedding as the output head) is standard and halves the parameter
// count; we implement the transpose by hand to keep the row-major story honest.
function logitsFromHidden(h: Float64Array, model: Model): Float64Array {
  const { vocabSize, dModel } = model.cfg;
  const normed = rmsNorm(tensor(h, [1, dModel]), model.normOut).data;
  const out = new Float64Array(vocabSize);
  const E = model.embed.data;
  for (let t = 0; t < vocabSize; t++) {
    let s = 0;
    const row = t * dModel;
    for (let d = 0; d < dModel; d++) s += normed[d] * E[row + d];
    out[t] = s;
  }
  return out;
}

// forwardNoCache — the REFERENCE path. Recompute everything from scratch for the
// whole token sequence and return the logits for the LAST position. O(seq^2) per
// layer because every token re-attends over all prior tokens with no reuse.
//
// This is the "obviously correct, obviously slow" baseline. Stage02's cached path
// must match its last-token logits to within float64 noise — that equality is the
// proof the cache is correct, and the speedup over THIS is the headline of stage02.
export function forwardNoCache(model: Model, tokenIds: number[]): Float64Array {
  const cfg = model.cfg;
  const seq = tokenIds.length;
  if (seq === 0) throw new Error("forwardNoCache: empty token sequence");
  if (seq > cfg.maxSeq) throw new Error(`forwardNoCache: seq ${seq} > maxSeq ${cfg.maxSeq}`);
  const kvDim = cfg.nKVHeads * cfg.dHead;
  // hidden states for all positions
  let h: Float64Array[] = tokenIds.map((id) => {
    const e = new Float64Array(cfg.dModel);
    e.set(model.embed.data.subarray(id * cfg.dModel, (id + 1) * cfg.dModel));
    return e;
  });
  for (let l = 0; l < cfg.nLayers; l++) {
    const w = model.layers[l];
    const newH: Float64Array[] = [];
    const kHist = new Float64Array(seq * kvDim);
    const vHist = new Float64Array(seq * kvDim);
    for (let pos = 0; pos < seq; pos++) {
      const r = blockStep(h[pos], w, pos, kHist, vHist, pos, cfg);
      kHist.set(r.k, pos * kvDim);
      vHist.set(r.v, pos * kvDim);
      newH.push(r.h);
    }
    h = newH;
  }
  return logitsFromHidden(h[seq - 1], model);
}

// forwardStep — the CACHED path. Process ONE token at absolute position cache.len,
// reusing all previously-cached K/V, append this token's K/V to the cache, and
// return its logits. O(seq) per step instead of O(seq^2): this is the entire reason
// inference engines exist. The cache is mutated in place; caller advances len.
//
// Precondition (invariant the cache type enforces): cache.len is the absolute
// position of the incoming token. Prefill is just calling this once per prompt
// token; decode is calling it once per generated token. Same function, two phases —
// which is the prefill/decode dichotomy the metrics module talks about.
export function forwardStep(model: Model, tokenId: number, cache: KVCache): Float64Array {
  const cfg = model.cfg;
  const pos = cache.len;
  if (pos >= cfg.maxSeq) throw new Error(`forwardStep: position ${pos} >= maxSeq ${cfg.maxSeq}`);
  const kvDim = cfg.nKVHeads * cfg.dHead;
  // annotate as plain Float64Array (not the ArrayBuffer-narrowed `new` type) so the
  // reassignment from blockStep's return below type-checks under @types/node 22.
  let h: Float64Array = new Float64Array(cfg.dModel);
  h.set(model.embed.data.subarray(tokenId * cfg.dModel, (tokenId + 1) * cfg.dModel));
  for (let l = 0; l < cfg.nLayers; l++) {
    const w = model.layers[l];
    const r = blockStep(h, w, pos, cache.k[l], cache.v[l], pos, cfg);
    // grow this layer's cache by one row (the just-computed K/V). Reallocating per
    // step is O(n) copying and is the naive approach stage05 replaces with paging.
    const grownK = new Float64Array((pos + 1) * kvDim);
    grownK.set(cache.k[l].subarray(0, pos * kvDim));
    grownK.set(r.k, pos * kvDim);
    cache.k[l] = grownK;
    const grownV = new Float64Array((pos + 1) * kvDim);
    grownV.set(cache.v[l].subarray(0, pos * kvDim));
    grownV.set(r.v, pos * kvDim);
    cache.v[l] = grownV;
    h = r.h;
  }
  cache.len = pos + 1;
  return logitsFromHidden(h, model);
}

// expose a few internals so optimization stages (paged-KV, quant) can reuse the
// exact same arithmetic instead of re-deriving it and drifting from the reference.
export const _internal = { logitsFromHidden, blockStep, projectQKV, applyRope, attendToken, ffn, zeros };
