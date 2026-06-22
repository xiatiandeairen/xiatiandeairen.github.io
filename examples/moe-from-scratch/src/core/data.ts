// core/data.ts — Toy, seeded, deterministic data generators for MoE intuition.
//
// WHY toy data with KNOWN structure: the whole MoE thesis is "different experts SHOULD
//   specialize on different sub-distributions". To check whether routing actually learns
//   that, we need data whose ground-truth structure we already know — clusters with a
//   known cluster id, a modular task with a known sub-rule per residue. Then a stage can
//   ask: did expert e end up handling cluster c? (see metrics.expertUtilization + viz.heatmap).
//
// HONESTY: these datasets are linearly-ish separable and tiny, so absolute accuracy is
//   optimistic. What transfers to real MoE is the SHAPE of the story: specialization
//   emerges, load-balance loss flattens utilization, collapse tanks entropy. Stages must
//   say so when they print absolute numbers.
//
// INVARIANT: every generator threads a Rng. Same seed => identical {X, Y, clusterId}.

import type { Rng } from "./prng.js";

export interface Dataset {
  X: number[][]; // [n][dim] inputs
  Y: number[]; // [n] integer labels (class id)
  clusterId: number[]; // [n] ground-truth latent group, for routing-vs-semantics checks
  dim: number;
  numClasses: number;
}

/**
 * k Gaussian blobs in `dim`-space, one class per blob. The canonical "experts should
 * each own a region of input space" intuition: cluster id == class id here, so a perfect
 * router would send all of cluster c to one expert.
 * Centers are spread on a scaled grid so blobs are separable but not trivially far.
 */
export function makeClusters(k: number, n: number, dim: number, rng: Rng, spread = 1.0): Dataset {
  const centers: number[][] = [];
  for (let c = 0; c < k; c++) {
    const center = new Array(dim);
    // Place centers on a random sphere-ish shell so no class is privileged near origin.
    for (let d = 0; d < dim; d++) center[d] = rng.normal() * 3;
    centers.push(center);
  }
  const X: number[][] = [];
  const Y: number[] = [];
  const clusterId: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = rng.int(k);
    const x = new Array(dim);
    for (let d = 0; d < dim; d++) x[d] = centers[c][d] + rng.normal() * spread;
    X.push(x);
    Y.push(c);
    clusterId.push(c);
  }
  return { X, Y, clusterId, dim, numClasses: k };
}

/**
 * Modular task: input is a `dim`-d random vector whose FIRST coordinate encodes an integer
 * 0..mod-1 (as integer + small noise); label = that integer mod some sub-rule. By design
 * the task decomposes into `mod` sub-rules, so an MoE can in principle assign one expert
 * per residue class. clusterId == residue, enabling the "did expert i learn residue i?" check.
 *
 * Concretely: we encode value v in coord 0, and label = v % numClasses. With numClasses<mod
 * several residues share a label, which makes specialization NON-trivial (a single expert
 * that memorizes one residue is not enough) — a more honest stress than pure clusters.
 */
export function makeModularTask(mod: number, n: number, dim: number, rng: Rng, numClasses = 2): Dataset {
  const X: number[][] = [];
  const Y: number[] = [];
  const clusterId: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = rng.int(mod);
    const x = new Array(dim).fill(0);
    // One-hot-ish encoding of v across the first `mod` dims (if room), else scalar coord.
    if (dim >= mod) {
      x[v] = 1 + rng.normal() * 0.05;
      for (let d = mod; d < dim; d++) x[d] = rng.normal() * 0.1; // distractor dims
    } else {
      x[0] = v / mod + rng.normal() * 0.01;
      for (let d = 1; d < dim; d++) x[d] = rng.normal() * 0.1;
    }
    X.push(x);
    Y.push(v % numClasses);
    clusterId.push(v); // ground-truth residue = the natural specialization axis
  }
  return { X, Y, clusterId, dim, numClasses };
}

export interface TokenStream {
  tokens: number[]; // [len] integer token ids in [0, vocab)
  vocab: number;
  /** Ground-truth "topic" of each token = id % numTopics, the natural routing axis. */
  topic: number[];
  numTopics: number;
}

/**
 * A toy token sequence for per-token routing demos. Tokens are drawn so that ids cluster
 * into `numTopics` bands (id % numTopics gives a stable topic), letting a stage check
 * whether the router sends same-topic tokens to the same expert.
 * INVARIANT: returns plain ints; embedding lookup happens in the stage via nn.Embedding.
 */
export function makeTokenStream(vocab: number, len: number, rng: Rng, numTopics = 4): TokenStream {
  const tokens: number[] = [];
  const topic: number[] = [];
  for (let i = 0; i < len; i++) {
    // Pick a topic, then a token within that topic's residue band — gives learnable structure.
    const t = rng.int(numTopics);
    // tokens with id % numTopics == t belong to topic t.
    let id = rng.int(vocab);
    id = id - (id % numTopics) + t;
    if (id >= vocab) id -= numTopics;
    tokens.push(id);
    topic.push(t);
  }
  return { tokens, vocab, topic, numTopics };
}
