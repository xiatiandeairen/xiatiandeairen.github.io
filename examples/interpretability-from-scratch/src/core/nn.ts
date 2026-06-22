// core/nn.ts — Layers + the HOOK system that makes this a book about interpretability.
//
// WHAT MAKES THIS FILE DIFFERENT from an ordinary nn library: every layer's forward takes
//   an optional `hooks` object and, at named points in the computation (resid_pre,
//   attn_out, mlp_out, resid_post, head_z, attn_pattern), it calls a hook that can both
//   OBSERVE the activation (caching, for logit lens / probing / SAE) and REPLACE it
//   (returning a new Tensor, for activation patching / ablation). One mechanism serves the
//   whole book: caching is "observe and keep", patching is "observe and substitute".
//
// WHY a unified hook point instead of ad-hoc instrumentation per experiment: if logit lens
//   read the residual stream one way and patching grabbed it another way, the two could
//   disagree about WHAT "the residual stream at layer 1" even is, and cross-experiment
//   corroboration (the whole point — does the patch land where the lens says?) would be
//   meaningless. One hook taxonomy = one ground truth for "where" in the network.
//
// INVARIANT: hook names are STABLE and layer-indexed (e.g. "blocks.0.resid_post"). Stages
//   address activations by these strings; renaming a hook silently breaks every saved
//   reference. The naming scheme is the public API of the model's internals.
//
// FAILURE MODE this guards against: a hook that mutates the activation in place instead of
//   returning a replacement — that would corrupt the cached "clean" copy patching relies
//   on. Hooks MUST be pure: take a Tensor, return a Tensor (the same one to observe-only).

import { Tensor, layerNorm, embeddingLookup, noGradActive } from "./autograd.js";
import { gaussian, type Rng } from "./rng.js";

// A hook receives the activation at a named point and returns what should flow onward.
// Return the same tensor to observe-only; return a new tensor to intervene.
export type Hook = (name: string, value: Tensor) => Tensor;
export interface Hooks {
  [pattern: string]: Hook;
}

/** Apply the matching hook for `name`, if any. Centralized so every call site behaves
 *  identically (exact-name match; stages can register one hook per exact name). */
function applyHook(hooks: Hooks | undefined, name: string, value: Tensor): Tensor {
  if (!hooks) return value;
  const h = hooks[name];
  return h ? h(name, value) : value;
}

export abstract class Module {
  // Explicit registries (not field reflection): stable order => reproducible optimizer
  // state, and zero ambiguity about what is a learnable vs a recomputed intermediate.
  protected _params: Tensor[] = [];
  protected _modules: Module[] = [];

  protected param(t: Tensor): Tensor {
    this._params.push(t);
    return t;
  }
  protected child<M extends Module>(m: M): M {
    this._modules.push(m);
    return m;
  }
  parameters(): Tensor[] {
    const out = [...this._params];
    for (const m of this._modules) out.push(...m.parameters());
    return out;
  }
  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }
}

// ----------------------------------------------------------------------------
// Linear: y = x @ W + b. x:(seq, inF) W:(inF, outF) b:(1, outF).
// ----------------------------------------------------------------------------
export class Linear extends Module {
  W: Tensor;
  b: Tensor | null;
  constructor(inF: number, outF: number, rng: Rng, bias = true) {
    super();
    // Xavier-ish init scaled by 1/sqrt(inF): keeps initial logits modest so softmax starts
    // near-uniform; an over-large init makes the model confidently wrong at step 0 and
    // distorts the early loss curve interp stages reason about.
    const std = 1 / Math.sqrt(inF);
    this.W = this.param(Tensor.from([inF, outF], () => gaussian(rng, 0, std)));
    this.b = bias ? this.param(Tensor.zeros([1, outF])) : null;
  }
  forward(x: Tensor): Tensor {
    const out = x.matmul(this.W);
    if (!this.b) return out;
    return out.add(this.b.broadcastRow(out.shape[0]));
  }
}

// ----------------------------------------------------------------------------
// Embedding table (token or positional). Wraps embeddingLookup so the table is a
// registered, trainable param.
// ----------------------------------------------------------------------------
export class Embedding extends Module {
  weight: Tensor;
  constructor(num: number, dim: number, rng: Rng, std = 0.02) {
    super();
    // GPT-convention small init: embeddings aren't behind a fan-in nonlinearity, so a
    // plain small normal keeps initial residual norms sane.
    this.weight = this.param(Tensor.from([num, dim], () => gaussian(rng, 0, std)));
  }
  forward(ids: number[]): Tensor {
    return embeddingLookup(this.weight, ids);
  }
}

// ----------------------------------------------------------------------------
// LayerNorm with learnable affine. Pure normalization is the autograd op; here we add
// gamma*xhat + beta and register them as params.
// ----------------------------------------------------------------------------
export class LayerNorm extends Module {
  gamma: Tensor;
  beta: Tensor;
  constructor(dim: number, eps = 1e-5) {
    super();
    this.gamma = this.param(Tensor.fill([1, dim], 1));
    this.beta = this.param(Tensor.zeros([1, dim]));
    this.eps = eps;
  }
  eps: number;
  forward(x: Tensor): Tensor {
    const normed = layerNorm(x, this.eps);
    const rows = x.shape[0];
    return normed.mul(this.gamma.broadcastRow(rows)).add(this.beta.broadcastRow(rows));
  }
}

// ----------------------------------------------------------------------------
// MultiHeadSelfAttention — computed per-head over a single sequence (2-D math).
// ----------------------------------------------------------------------------
//
// WHY per-head explicit loops rather than one big batched matmul: interpretability needs
//   PER-HEAD attention matrices and per-head value outputs as first-class observable
//   objects (induction heads, QK circuits, head ablation all act on a single head). A
//   fused kernel hides exactly the structure we want to expose. The honest cost is speed;
//   on toy sizes (seq<=16, heads<=4) it's microseconds, and clarity wins.
//
// CAUSAL MASK: position i may attend only to j<=i. We add -1e9 to masked logits BEFORE
//   softmax (not after) so masked weights are ~0 with a correct gradient. Masking after
//   softmax would leave a normalization bug.
//
// Hook points exposed per head/layer:
//   blocks.L.attn.head_z.H   — this head's value-weighted output (seq, d_head)
//   blocks.L.attn.pattern.H  — this head's attention matrix (seq, seq), observe-only*
//   blocks.L.attn_out        — concatenated+projected attention output (seq, d_model)
//   (*pattern is exposed for caching/viz; replacing it is out of scope for this toy book.)
export class MultiHeadSelfAttention extends Module {
  Wq: Linear;
  Wk: Linear;
  Wv: Linear;
  Wo: Linear;
  nHeads: number;
  dModel: number;
  dHead: number;
  layerIdx: number;
  // Filled each forward so viz/interp can read the last attention pattern without re-running.
  lastPatterns: Float64Array[] = []; // per head, (seq*seq)

  constructor(dModel: number, nHeads: number, layerIdx: number, rng: Rng) {
    super();
    if (dModel % nHeads !== 0) throw new Error(`dModel ${dModel} not divisible by nHeads ${nHeads}`);
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.dHead = dModel / nHeads;
    this.layerIdx = layerIdx;
    this.Wq = this.child(new Linear(dModel, dModel, rng, false));
    this.Wk = this.child(new Linear(dModel, dModel, rng, false));
    this.Wv = this.child(new Linear(dModel, dModel, rng, false));
    this.Wo = this.child(new Linear(dModel, dModel, rng, true));
  }

  /** x: (seq, dModel). Returns (seq, dModel). hooks may observe/replace head_z and the
   *  concatenated attn_out. */
  forward(x: Tensor, hooks?: Hooks): Tensor {
    const q = this.Wq.forward(x); // (seq, dModel)
    const k = this.Wk.forward(x);
    const v = this.Wv.forward(x);
    const scale = 1 / Math.sqrt(this.dHead);
    this.lastPatterns = [];

    // Build concatenated per-head outputs into one (seq, dModel) tensor by adding each
    // head's contribution into the right column block. We do head math by slicing columns.
    const headOuts: Tensor[] = [];
    for (let h = 0; h < this.nHeads; h++) {
      const qh = sliceCols(q, h * this.dHead, this.dHead); // (seq, dHead)
      const kh = sliceCols(k, h * this.dHead, this.dHead);
      const vh = sliceCols(v, h * this.dHead, this.dHead);
      // scores = qh @ kh^T * scale, then causal mask, then softmax -> (seq, seq)
      let scores = qh.matmul(kh.transpose()).mulScalar(scale);
      scores = causalMask(scores);
      const pattern = scores.softmax(); // (seq, seq)
      // cache raw pattern for viz/interp (detached numbers; grad still flows via `pattern`)
      this.lastPatterns.push(pattern.data.slice());
      let z = pattern.matmul(vh); // (seq, dHead)
      z = applyHook(hooks, `blocks.${this.layerIdx}.attn.head_z.${h}`, z);
      headOuts.push(z);
    }
    let concat = concatCols(headOuts); // (seq, dModel)
    let out = this.Wo.forward(concat);
    out = applyHook(hooks, `blocks.${this.layerIdx}.attn_out`, out);
    return out;
  }
}

// ----------------------------------------------------------------------------
// MLP: Linear -> GELU -> Linear, the per-position feature transform.
// ----------------------------------------------------------------------------
export class MLP extends Module {
  fc: Linear;
  proj: Linear;
  layerIdx: number;
  constructor(dModel: number, dHidden: number, layerIdx: number, rng: Rng) {
    super();
    this.layerIdx = layerIdx;
    this.fc = this.child(new Linear(dModel, dHidden, rng, true));
    this.proj = this.child(new Linear(dHidden, dModel, rng, true));
  }
  forward(x: Tensor, hooks?: Hooks): Tensor {
    const hidden = this.fc.forward(x).gelu(); // (seq, dHidden) — SAE/neuron analysis target
    let out = this.proj.forward(hidden);
    out = applyHook(hooks, `blocks.${this.layerIdx}.mlp_out`, out);
    return out;
  }
}

// ----------------------------------------------------------------------------
// TransformerBlock: pre-norm residual block. The residual stream is the spine interp reads.
// ----------------------------------------------------------------------------
//
// PRE-NORM (norm before the sublayer, add the raw sublayer output to the residual): this
//   keeps the residual stream a clean linear accumulator — each component WRITES to it
//   additively. That additivity is exactly what logit lens and patching exploit: the
//   stream at any point is a sum of contributions you can read or swap. Post-norm would
//   entangle them and blur attribution.
export class TransformerBlock extends Module {
  attn: MultiHeadSelfAttention;
  mlp: MLP;
  ln1: LayerNorm;
  ln2: LayerNorm;
  layerIdx: number;
  constructor(dModel: number, nHeads: number, dHidden: number, layerIdx: number, rng: Rng) {
    super();
    this.layerIdx = layerIdx;
    this.ln1 = this.child(new LayerNorm(dModel));
    this.attn = this.child(new MultiHeadSelfAttention(dModel, nHeads, layerIdx, rng));
    this.ln2 = this.child(new LayerNorm(dModel));
    this.mlp = this.child(new MLP(dModel, dHidden, layerIdx, rng));
  }
  forward(x: Tensor, hooks?: Hooks): Tensor {
    let resid = applyHook(hooks, `blocks.${this.layerIdx}.resid_pre`, x);
    const attnOut = this.attn.forward(this.ln1.forward(resid), hooks);
    resid = resid.add(attnOut);
    resid = applyHook(hooks, `blocks.${this.layerIdx}.resid_mid`, resid);
    const mlpOut = this.mlp.forward(this.ln2.forward(resid), hooks);
    resid = resid.add(mlpOut);
    resid = applyHook(hooks, `blocks.${this.layerIdx}.resid_post`, resid);
    return resid;
  }
}

// ----------------------------------------------------------------------------
// TinyTransformer: embeddings -> blocks -> final LN -> unembedding.
// ----------------------------------------------------------------------------
//
// SIZE BY DESIGN: 1-2 layers, 1-4 heads, dModel 16-32, a few thousand params. Small enough
//   to train on CPU in seconds AND small enough that a human can hold the whole circuit in
//   their head — which is the only regime where "we fully reverse-engineered it" is honest.
// HONESTY (repeated in every stage): absolute numbers from this model (loss, probe acc,
//   patch recovery, SAE feature count) are OPTIMISTIC because the task is synthetic and the
//   model tiny. What transfers to real models is the MECHANISM and the SHAPE of curves, not
//   the magnitudes.
export interface ModelConfig {
  vocab: number;
  dModel: number;
  nHeads: number;
  nLayers: number;
  dHidden: number;
  maxSeq: number;
}

export class TinyTransformer extends Module {
  tokEmb: Embedding;
  posEmb: Embedding;
  blocks: TransformerBlock[];
  lnFinal: LayerNorm;
  unembed: Linear;
  cfg: ModelConfig;

  constructor(cfg: ModelConfig, rng: Rng) {
    super();
    this.cfg = cfg;
    this.tokEmb = this.child(new Embedding(cfg.vocab, cfg.dModel, rng));
    this.posEmb = this.child(new Embedding(cfg.maxSeq, cfg.dModel, rng));
    this.blocks = [];
    for (let l = 0; l < cfg.nLayers; l++) {
      const b = this.child(new TransformerBlock(cfg.dModel, cfg.nHeads, cfg.dHidden, l, rng));
      this.blocks.push(b);
    }
    this.lnFinal = this.child(new LayerNorm(cfg.dModel));
    this.unembed = this.child(new Linear(cfg.dModel, cfg.vocab, rng, false));
  }

  /** Forward over a single sequence of token ids. Returns logits (seq, vocab). hooks may
   *  observe/replace any named activation. The embedding sum (tok+pos) is the resid stream
   *  at the input; we expose it as "embed" so logit lens has a layer-0 anchor. */
  forward(ids: number[], hooks?: Hooks): Tensor {
    const seq = ids.length;
    if (seq > this.cfg.maxSeq) throw new Error(`seq ${seq} > maxSeq ${this.cfg.maxSeq}`);
    const positions = Array.from({ length: seq }, (_, i) => i);
    let resid = this.tokEmb.forward(ids).add(this.posEmb.forward(positions));
    resid = applyHook(hooks, "embed", resid);
    for (const block of this.blocks) resid = block.forward(resid, hooks);
    const normed = this.lnFinal.forward(resid);
    const logits = this.unembed.forward(normed);
    return logits;
  }
}

// ---- column-slice helpers with autograd (used by attention head splitting) ----

/** Slice columns [start, start+width) from a (rows, cols) tensor -> (rows, width).
 *  Adjoint scatters grad back into the source columns. WHY needed: head splitting reads a
 *  contiguous column block per head; doing it as an autograd op keeps grads correct without
 *  a separate "head dim" in the tensor. */
function sliceCols(x: Tensor, start: number, width: number): Tensor {
  const [rows, cols] = x.shape;
  const out = new Float64Array(rows * width);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < width; c++) out[r * width + c] = x.data[r * cols + start + c];
  const t = new Tensor(out, [rows, width], [x], "sliceCols");
  t._backward = () => {
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < width; c++) x.grad[r * cols + start + c] += t.grad[r * width + c];
  };
  return t;
}

/** Concatenate (rows, w_i) tensors along columns -> (rows, sum w_i). Inverse adjoint of
 *  sliceCols. Used to reassemble per-head outputs before the output projection. */
function concatCols(parts: Tensor[]): Tensor {
  const rows = parts[0].shape[0];
  let totalCols = 0;
  for (const p of parts) totalCols += p.shape[1];
  const out = new Float64Array(rows * totalCols);
  let colOff = 0;
  const offsets: number[] = [];
  for (const p of parts) {
    offsets.push(colOff);
    const w = p.shape[1];
    for (let r = 0; r < rows; r++) for (let c = 0; c < w; c++) out[r * totalCols + colOff + c] = p.data[r * w + c];
    colOff += w;
  }
  const t = new Tensor(out, [rows, totalCols], parts, "concatCols");
  t._backward = () => {
    for (let pi = 0; pi < parts.length; pi++) {
      const p = parts[pi];
      const w = p.shape[1];
      const off = offsets[pi];
      for (let r = 0; r < rows; r++) for (let c = 0; c < w; c++) p.grad[r * w + c] += t.grad[r * totalCols + off + c];
    }
  };
  return t;
}

/** Add -1e9 to upper-triangular (future) positions of a (seq, seq) score matrix before
 *  softmax => causal masking. Adjoint passes grad through unchanged for kept positions and
 *  (effectively) zero for masked ones since the additive constant has zero derivative.
 *  WHY -1e9 not -Inf: -Inf*0 in the softmax grad would produce NaN; a large finite number
 *  makes exp() underflow to ~0 while keeping grads finite. A real failure mode if you use
 *  -Infinity here. */
function causalMask(scores: Tensor): Tensor {
  const [seq, seq2] = scores.shape;
  const out = new Float64Array(seq * seq2);
  for (let i = 0; i < seq; i++)
    for (let j = 0; j < seq2; j++) out[i * seq2 + j] = j > i ? scores.data[i * seq2 + j] - 1e9 : scores.data[i * seq2 + j];
  const t = new Tensor(out, scores.shape, [scores], "causalMask");
  t._backward = () => {
    for (let i = 0; i < scores.size; i++) scores.grad[i] += t.grad[i];
  };
  return t;
}

// Re-export noGradActive for layers that may want eval-time shortcuts later.
export { noGradActive };
