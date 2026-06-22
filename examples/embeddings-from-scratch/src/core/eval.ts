// core/eval.ts — how we judge an embedding space honestly.
//
// An embedding is only as good as the geometry it induces: similar things close,
// analogies parallel, the space neither collapsed nor degenerate. This module is
// the measuring tape. All functions here work on PLAIN number[] (detached from
// the autograd graph) because evaluation must never extend or perturb the graph —
// we are reading the trained weights, not training.
//
// Why each metric earns its place:
//  - cosine/euclidean: the two notions of "close". The book shows they disagree
//    when vectors aren't normalized — a teaching point, not a footnote.
//  - nearestNeighbors: the qualitative smell test ("what's near 'king'?").
//  - analogySolve/Accuracy: the famous king-man+woman test; a single QUANTITATIVE
//    number that either moves or doesn't.
//  - alignment/uniformity: the contrastive-learning diagnostic pair (Wang & Isola)
//    that explains *how* a space can score okay on similarity yet be quietly
//    collapsed. Used heavily in the InfoNCE / temperature stages.

// Cosine similarity in [-1, 1]. Direction-only: magnitude-invariant. This is the
// "right" similarity for embeddings because training pushes on dot products and
// we usually care about angle, not length. Zero-vector input returns 0 (not NaN)
// — a degenerate but legal state during early training we must not crash on.
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

// Euclidean distance. Magnitude-sensitive: two vectors pointing the same way but
// different lengths are "far". Contrasting this with cosine is exactly how the
// book motivates normalization.
export function euclidean(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

// A neighbor result: which row, and how similar.
export interface Neighbor {
  index: number;
  score: number;
}

// Top-k nearest neighbors of `vec` among rows of `matrix` by cosine similarity.
// EXCLUDES any row that is the query vector itself by identity check on the array
// reference — so calling nearestNeighbors(emb[i], emb) won't return i as its own
// top neighbor (the trivial self-match that hides real structure). Pass the same
// array instance for that to work.
export function nearestNeighbors(vec: number[], matrix: number[][], k: number): Neighbor[] {
  const scored: Neighbor[] = [];
  for (let i = 0; i < matrix.length; i++) {
    if (matrix[i] === vec) continue; // skip self (reference identity)
    scored.push({ index: i, score: cosineSimilarity(vec, matrix[i]) });
  }
  // desc by score; stable enough at toy sizes. Ties broken by lower index.
  scored.sort((a, b) => b.score - a.score || a.index - b.index);
  return scored.slice(0, k);
}

// Solve the analogy a:b :: c:? by the vector-arithmetic rule: target ≈ b - a + c,
// return the index whose embedding is closest (cosine) to that target, excluding
// the three input words (the standard protocol — otherwise the answer is often
// one of the inputs, inflating accuracy).
//
// Returns -1 only if every candidate was excluded (degenerate tiny vocab).
export function analogySolve(
  matrix: number[][],
  a: number[],
  b: number[],
  c: number[],
  excludeIdx: number[] = [],
): number {
  const target = b.map((_, i) => b[i] - a[i] + c[i]);
  const exclude = new Set(excludeIdx);
  let best = -1;
  let bestScore = -Infinity;
  for (let i = 0; i < matrix.length; i++) {
    if (exclude.has(i)) continue;
    const s = cosineSimilarity(target, matrix[i]);
    if (s > bestScore) {
      bestScore = s;
      best = i;
    }
  }
  return best;
}

// One analogy question over token IDs.
export interface AnalogyQuestion {
  a: number;
  b: number;
  c: number;
  expected: number;
}

// Accuracy over a set of analogy questions. The headline quantitative metric for
// the eval stage. Each question excludes its own a/b/c from the candidate pool,
// per analogySolve. Reported as a fraction in [0,1]; the book pairs it with the
// raw correct/total count so a "0.50 on 4 questions" can't masquerade as robust.
export function analogyAccuracy(
  matrix: number[][],
  questions: AnalogyQuestion[],
): { accuracy: number; correct: number; total: number } {
  let correct = 0;
  for (const q of questions) {
    const got = analogySolve(matrix, matrix[q.a], matrix[q.b], matrix[q.c], [q.a, q.b, q.c]);
    if (got === q.expected) correct++;
  }
  const total = questions.length;
  return { accuracy: total === 0 ? 0 : correct / total, total, correct };
}

// alignment: mean squared distance between L2-normalized POSITIVE pairs. Lower is
// better — it measures whether things that SHOULD be close actually are. Defined
// on normalized vectors so it's purely angular, matching contrastive training.
//
// (Wang & Isola 2020, "alignment".) The book uses it to show that a contrastive
// loss with the wrong temperature can keep positives close (good alignment) while
// the overall space collapses (bad uniformity) — the two must be read together.
export function alignment(pairs: Array<[number[], number[]]>): number {
  if (pairs.length === 0) return 0;
  let s = 0;
  for (const [x, y] of pairs) {
    const nx = l2normalize(x);
    const ny = l2normalize(y);
    let d = 0;
    for (let i = 0; i < nx.length; i++) {
      const diff = nx[i] - ny[i];
      d += diff * diff;
    }
    s += d;
  }
  return s / pairs.length;
}

// uniformity: log of the mean Gaussian-potential between all pairs of normalized
// vectors. More negative = points spread more evenly on the hypersphere (good).
// Near 0 = points collapsed to a cluster (bad — the "representation collapse"
// failure contrastive learning is designed to avoid). t=2 is the paper's default.
//
// This is the metric that EXPOSES collapse, which similarity alone hides: a
// collapsed space has great pairwise similarity yet useless geometry.
export function uniformity(vectors: number[][], t = 2): number {
  const n = vectors.length;
  if (n < 2) return 0;
  const normed = vectors.map(l2normalize);
  let sum = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let d = 0;
      for (let d2 = 0; d2 < normed[i].length; d2++) {
        const diff = normed[i][d2] - normed[j][d2];
        d += diff * diff;
      }
      sum += Math.exp(-t * d);
      count++;
    }
  }
  return Math.log(sum / count);
}

// L2-normalize to unit length. Zero vector maps to itself (can't normalize a
// zero) rather than producing NaN — same defensive choice as cosineSimilarity.
function l2normalize(v: number[]): number[] {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n);
  if (n === 0) return v.slice();
  return v.map((x) => x / n);
}
