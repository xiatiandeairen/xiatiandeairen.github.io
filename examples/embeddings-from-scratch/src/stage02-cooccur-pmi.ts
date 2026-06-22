// stage02-cooccur-pmi.ts — Chapter 02: vectors WITHOUT training, from co-occurrence + PMI.
//
// The thesis of this chapter, stated as a falsifiable claim the code must back up:
// you do NOT need gradient descent to get word vectors with real semantic
// structure. Plain counting + an information-theoretic reweighting (PMI) +
// classical linear algebra (truncated SVD) already separates "king/queen" from
// "the/a". word2vec (next chapters) is then re-framed not as magic but as an
// implicit, scalable factorization of essentially this same PMI matrix
// (Levy & Goldberg 2014). Seeing the count-based version first makes the neural
// version demystifiable instead of mysterious.
//
// What this file proves, in order, each with REAL printed numbers:
//   (1) raw co-occurrence counts exist and look like a (banded) matrix → heatmap.
//   (2) raw counts measure ASSOCIATION STRENGTH (un-normalized dot product), and
//       that measure is dominated by frequency: the top "neighbors" of EVERY query
//       are the same handful of frequent words ("is","rules","the","a"), because a
//       frequent word contributes a big term to everyone's dot product regardless
//       of meaning. Show the SAME polluted list recurring across unrelated queries
//       — the failure mode, demonstrated, not asserted.
//   (3) PPMI (positive pointwise mutual information) divides out each word's
//       frequency baseline before we compare. Under PPMI rows, each query's
//       neighbors become its OWN distinctive associates. Same data, frequency
//       discounted, qualitatively different geometry — the whole point of PMI.
//   (4) truncated SVD on the PPMI matrix compresses V-dim sparse rows to d dense
//       dims and the semantic clusters (royalty / animals / food / directions)
//       survive in the low-rank space — that compressed dense vector IS the
//       count-based word embedding.
//
// HONESTY NOTE (a trap this file deliberately avoids): a common textbook claim is
// "raw-count nearest neighbors are all stopwords". On THIS toy corpus that does
// NOT reproduce under COSINE similarity, because cosine already normalizes away
// each word's magnitude, so it does half of PMI's job for free — measured here,
// cosine-raw NN of content words are already mostly clean. So we do NOT print that
// false relative trend. The real, reproducible frequency pollution lives in the
// UN-NORMALIZED association measure (raw dot product), which is exactly the
// quantity PMI is designed to correct. The corpus is ~40 templated words, so
// absolute neighbor quality is optimistic; what transfers to real corpora is the
// mechanism (un-normalized association is frequency-biased; PMI fixes it).
//
// No autograd / no optimizer is imported on purpose: this chapter's claim is
// precisely that no training is involved. The only "learning" is counting and an
// eigen-style decomposition. Determinism: the one stochastic input is the corpus
// (seeded Rng) and SVD power-iteration init (seeded Rng); same seed → same output.

import { Rng } from "./core/rng.js";
import { generateCorpus, buildVocab, cooccurrenceMatrix, type Vocab } from "./core/text.js";
import { asciiHeatmap, asciiBar } from "./core/plot.js";
import { nearestNeighbors, cosineSimilarity } from "./core/eval.js";

// ---------------------------------------------------------------------------
// PPMI: the reweighting that turns counts into "is this co-occurrence surprising?"
// ---------------------------------------------------------------------------

// Pointwise mutual information of words i,j:
//   PMI(i,j) = log( P(i,j) / (P(i) P(j)) )
// Intuition: how much MORE often do i and j co-occur than if they were
// independent. "the"+anything ≈ baseline (PMI≈0) because "the" is everywhere;
// "king"+"queen" >> baseline (PMI large) because they share template slots.
//
// We clamp negatives to 0 → PPMI (positive PMI). Why: negative PMI ("these two
// co-occur LESS than chance") is statistically unreliable in sparse counts (most
// pairs simply never co-occur, giving log(0) = -inf), and empirically PPMI is the
// variant that produces usable vectors (Bullinaria & Levy 2007). The clamp is the
// single most important line for why the resulting geometry is clean.
//
// Failure modes guarded here:
//  - log(0): a pair that never co-occurs has P(i,j)=0 → PMI=-inf. The PPMI clamp
//    maps it to 0, so a never-seen pair contributes nothing rather than NaN/-inf.
//  - empty matrix / zero total: caller passes a corpus that produced no windows.
//    We throw rather than emit a matrix of NaN (0/0), which would silently poison
//    every downstream cosine.
function computePpmi(cooc: number[][]): number[][] {
  const V = cooc.length;
  if (V === 0) throw new Error("computePpmi: empty co-occurrence matrix");

  // Marginals. total = sum of all cells; rowSum[i] = how often i appeared as a
  // center (== how often anything appeared in i's window, by symmetry of the toy
  // window). We use these as the empirical P(i), P(j), P(i,j) estimates.
  let total = 0;
  const rowSum = new Array(V).fill(0);
  const colSum = new Array(V).fill(0);
  for (let i = 0; i < V; i++) {
    for (let j = 0; j < V; j++) {
      const c = cooc[i][j];
      total += c;
      rowSum[i] += c;
      colSum[j] += c;
    }
  }
  if (total === 0) throw new Error("computePpmi: co-occurrence matrix is all zeros");

  const ppmi: number[][] = Array.from({ length: V }, () => new Array(V).fill(0));
  for (let i = 0; i < V; i++) {
    for (let j = 0; j < V; j++) {
      const cij = cooc[i][j];
      if (cij === 0) continue; // never co-occurred → PPMI stays 0 (avoids log(0))
      // P(i,j) = cij/total ; P(i) = rowSum[i]/total ; P(j) = colSum[j]/total
      // pmi = log( (cij*total) / (rowSum[i]*colSum[j]) ) — algebraically identical
      // but keeps the division stable (no tiny-probability underflow).
      const pmi = Math.log((cij * total) / (rowSum[i] * colSum[j]));
      ppmi[i][j] = pmi > 0 ? pmi : 0; // the PPMI clamp
    }
  }
  return ppmi;
}

// ---------------------------------------------------------------------------
// Truncated SVD via power iteration — classical dimensionality reduction, by hand.
// ---------------------------------------------------------------------------

// We want the top-d singular directions of the (V x V) PPMI matrix M so that each
// word's V-dim sparse row collapses to a d-dim dense vector. Full SVD is O(V^3);
// we don't need it. Because M is symmetric here (symmetric window → symmetric
// co-occurrence → symmetric PPMI), M = U Σ Uᵀ and the SVD reduces to the
// eigendecomposition of M. Top eigenvectors of a symmetric matrix are exactly
// what power iteration + deflation gives, in ~50 lines, fully auditable.
//
// Word embedding convention: row i of the result = sqrt(λ_k) * eigvec_k[i] across
// the top-d k. Scaling each axis by sqrt(eigenvalue) is the SVD reconstruction
// U Σ^{1/2}; it weights stronger directions more, which empirically gives better
// similarity geometry than raw eigenvectors.
//
// WARNING: this is a TOY eigensolver. Power iteration converges slowly for close
// eigenvalues and deflation accumulates floating-point error past ~the first
// handful of components — fine for d≤8 on a ~40x40 matrix, NOT a substitute for
// LAPACK on real matrices. The book states this explicitly; the lesson is the
// mechanism, not the numerics.
function truncatedSvdSymmetric(
  matrix: number[][],
  d: number,
  rng: Rng,
  iters = 200,
): number[][] {
  const V = matrix.length;
  // Work on a mutable copy: deflation subtracts each found component from it.
  const M = matrix.map((row) => row.slice());
  const eigvecs: number[][] = [];
  const eigvals: number[] = [];

  for (let comp = 0; comp < d; comp++) {
    // Random unit start vector. Seeded so the whole SVD is reproducible — a fixed
    // start can be orthogonal-by-bad-luck to the top eigenvector and stall, so we
    // use a random one, which is the standard power-iteration recommendation.
    let v = normalize(Array.from({ length: V }, () => rng.nextFloat() - 0.5));

    let eigval = 0;
    for (let it = 0; it < iters; it++) {
      const Mv = matVec(M, v);
      const norm = vecNorm(Mv);
      if (norm < 1e-12) break; // matrix exhausted (rank < comp) → remaining dims are 0
      const next = Mv.map((x) => x / norm);
      // Rayleigh quotient vᵀMv estimates the eigenvalue at convergence; sign of
      // norm-vs-rayleigh also recovers negative eigenvalues that pure norm misses.
      eigval = dotPlain(next, matVec(M, next));
      v = next;
    }

    eigvecs.push(v);
    eigvals.push(eigval);
    // Deflate: M ← M − λ v vᵀ removes this eigenpair so the next power iteration
    // finds the next-largest. This is where FP error accumulates; acceptable here.
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) {
        M[i][j] -= eigval * v[i] * v[j];
      }
    }
  }

  // Build embeddings: emb[i][k] = sqrt(max(λ_k,0)) * eigvec_k[i].
  // max(λ,0): a toy deflated matrix can yield small negative eigenvalues from FP
  // noise; sqrt of a negative would be NaN and poison every cosine. Clamp them —
  // a near-zero eigenvalue carries no real signal anyway.
  const emb: number[][] = Array.from({ length: V }, () => new Array(d).fill(0));
  for (let k = 0; k < d; k++) {
    const scale = Math.sqrt(Math.max(eigvals[k], 0));
    for (let i = 0; i < V; i++) {
      emb[i][k] = eigvecs[k][i] * scale;
    }
  }
  return emb;
}

// --- small dense-vector helpers (plain numbers; NOT autograd Values) -----------
// These are intentionally local and untyped-fancy: this chapter is pure numeric
// linear algebra, deliberately NOT routed through core/autograd (no gradients
// here, that's the whole "no training" point).

function matVec(M: number[][], v: number[]): number[] {
  const out = new Array(M.length).fill(0);
  for (let i = 0; i < M.length; i++) {
    let s = 0;
    const row = M[i];
    for (let j = 0; j < v.length; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

function dotPlain(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function vecNorm(v: number[]): number {
  return Math.sqrt(dotPlain(v, v));
}

function normalize(v: number[]): number[] {
  const n = vecNorm(v);
  return n < 1e-12 ? v.slice() : v.map((x) => x / n);
}

// --- reporting helpers ---------------------------------------------------------

// Pretty-print the top-k neighbors of a query word given a matrix whose rows are
// that word's vectors (raw counts, PPMI rows, or SVD embeddings). Returns the
// neighbor tokens so the caller can assert/compare programmatically.
function reportNeighbors(
  label: string,
  queryWord: string,
  vocab: Vocab,
  rows: number[][],
  k: number,
): string[] {
  const qid = vocab.stoi.get(queryWord);
  if (qid === undefined) throw new Error(`reportNeighbors: '${queryWord}' not in vocab`);
  const nbrs = nearestNeighbors(rows[qid], rows, k);
  const tokens = nbrs.map((n) => vocab.itos[n.index]);
  const pretty = nbrs.map((n) => `${vocab.itos[n.index]}(${n.score.toFixed(2)})`).join("  ");
  console.log(`  ${label} nearest('${queryWord}'): ${pretty}`);
  return tokens;
}

// Raw ASSOCIATION-strength neighbors: rank other words by the UN-NORMALIZED dot
// product of co-occurrence rows. This is the failure-mode measure — unlike cosine
// it does NOT divide out magnitude, so a frequent word j contributes cooc[q][j] *
// cooc[i][j] (two big factors) to many pairs and floods every query's top list.
// We compute it by hand rather than via core's cosine NN precisely because the
// whole demonstration is "what happens when you DON'T normalize".
function rawDotNeighbors(cooc: number[][], qid: number, k: number): number[] {
  const scored: Array<{ i: number; s: number }> = [];
  for (let i = 0; i < cooc.length; i++) {
    if (i === qid) continue;
    let s = 0;
    for (let j = 0; j < cooc.length; j++) s += cooc[qid][j] * cooc[i][j];
    scored.push({ i, s });
  }
  scored.sort((a, b) => b.s - a.s || a.i - b.i);
  return scored.slice(0, k).map((x) => x.i);
}

// Pollution metric, frequency-grounded (no hand-labeled stopword list, which would
// beg the question): what fraction of neighbor slots are taken by HIGH-FREQUENCY
// words? Vocab ids are assigned in descending frequency, so "id < cutoff" == "in
// the top frequency band". A high share means frequent words flood the neighbor
// lists irrespective of the query's meaning — exactly the failure PMI corrects.
function highFreqShare(neighborLists: number[][], highFreqCutoffId: number): number {
  let total = 0;
  let highFreq = 0;
  for (const list of neighborLists) {
    for (const id of list) {
      total++;
      if (id < highFreqCutoffId) highFreq++;
    }
  }
  return total === 0 ? 0 : highFreq / total;
}

// ---------------------------------------------------------------------------
// main — runs the four demonstrations end to end.
// ---------------------------------------------------------------------------

function main(): void {
  // Two independent Rngs (per the core's "one Rng per concern" rule): one drives
  // the corpus, one drives SVD's random start. Decoupling them means changing the
  // SVD seed never perturbs the corpus and vice versa.
  const corpusRng = new Rng(42);
  const svdRng = new Rng(7);

  const corpus = generateCorpus(corpusRng, 400);
  const vocab = buildVocab(corpus.tokens);
  console.log(`[corpus] sentences=${corpus.sentences.length} tokens=${corpus.tokens.length} vocab=${vocab.size}`);
  console.log(`[corpus] most frequent (id 0..4): ${vocab.itos.slice(0, 5).join(", ")}`);

  const windowSize = 2;
  const cooc = cooccurrenceMatrix(corpus.sentences, vocab, windowSize);

  // (1) Heatmap of raw co-occurrence — the raw material. We show a readable
  // sub-block (first 14 words by frequency) because the full V x V is wide; the
  // structure (frequent words = dark rows/cols everywhere) is already visible.
  const previewN = Math.min(14, vocab.size);
  const previewLabels = vocab.itos.slice(0, previewN);
  const previewMatrix = cooc.slice(0, previewN).map((row) => row.slice(0, previewN));
  console.log(`\n[1] raw co-occurrence heatmap (top ${previewN} words by frequency, window=${windowSize}):`);
  console.log(asciiHeatmap(previewMatrix, previewLabels));

  // (2) + (3): the headline comparison. Take several UNRELATED content queries and
  // show that RAW association (un-normalized dot product) returns nearly the same
  // frequent words for all of them (pollution), while PPMI cosine returns each
  // query's own distinctive associates.
  const ppmi = computePpmi(cooc);
  const probes = ["king", "dog", "north"]; // royalty / animal / direction — disjoint meanings
  const k = 5;

  console.log(`\n[2] FAILURE MODE — RAW association (un-normalized dot of count rows):`);
  console.log(`    Watch the SAME frequent words recur across unrelated queries.`);
  const rawLists: number[][] = [];
  for (const w of probes) {
    const ids = rawDotNeighbors(cooc, vocab.stoi.get(w)!, k);
    rawLists.push(ids);
    console.log(`    raw-dot('${w}')  → ${ids.map((i) => vocab.itos[i]).join(", ")}`);
  }

  console.log(`\n[3] FIX — same queries on PPMI rows, cosine (each frequency baseline divided out):`);
  const ppmiLists: number[][] = [];
  for (const w of probes) {
    const nbrs = nearestNeighbors(ppmi[vocab.stoi.get(w)!], ppmi, k);
    const ids = nbrs.map((n) => n.index);
    ppmiLists.push(ids);
    console.log(`    ppmi('${w}')     → ${nbrs.map((n) => `${vocab.itos[n.index]}(${n.score.toFixed(2)})`).join("  ")}`);
  }

  // Quantify the pollution: fraction of neighbor slots occupied by high-frequency
  // words (top frequency quartile of the vocab). Raw association should be heavily
  // frequency-loaded; PPMI should not. Averaged over a WIDER probe set than the 3
  // printed above so the number isn't a 3-sample fluke.
  const widerProbes = ["king", "dog", "north", "bread", "queen", "cat", "fish", "man"];
  const highFreqCutoffId = Math.ceil(vocab.size / 4); // top quartile by frequency
  const rawWide = widerProbes.map((w) => rawDotNeighbors(cooc, vocab.stoi.get(w)!, k));
  const ppmiWide = widerProbes.map((w) =>
    nearestNeighbors(ppmi[vocab.stoi.get(w)!], ppmi, k).map((n) => n.index),
  );
  const rawShare = highFreqShare(rawWide, highFreqCutoffId);
  const ppmiShare = highFreqShare(ppmiWide, highFreqCutoffId);
  console.log(`\n[3b] pollution metric over ${widerProbes.length} content queries — share of top-${k} neighbor slots held by high-frequency words (lower = cleaner):`);
  console.log(asciiBar(["raw dot", "PPMI cos"], [rawShare, ppmiShare]));
  console.log(`      raw=${(rawShare * 100).toFixed(0)}%  PPMI=${(ppmiShare * 100).toFixed(0)}%  → PPMI ${ppmiShare < rawShare ? "cuts frequency pollution" : "does NOT cut pollution"} (high-freq band = top ${highFreqCutoffId} of ${vocab.size} words)`);

  // (4) Truncated SVD on the PPMI matrix → dense d-dim embeddings. Verify the
  // semantic clusters survive the compression by checking neighbors of cluster
  // exemplars. These are the count-based "word vectors" — no gradient ever ran.
  const d = 8;
  const emb = truncatedSvdSymmetric(ppmi, d, svdRng);
  console.log(`\n[4] truncated SVD(PPMI) → ${d}-dim dense embeddings. Cluster neighbors:`);
  const exemplars = ["king", "dog", "bread", "north"];
  for (const w of exemplars) reportNeighbors(`svd  `, w, vocab, emb, 3);

  // Contrast: run the SAME SVD on the RAW count matrix to show the failure mode
  // survives compression — low-rank-ifying garbage gives low-rank garbage. The
  // 'king' neighbor under raw-count SVD should still skew toward frequent words.
  const embRaw = truncatedSvdSymmetric(cooc, d, new Rng(7));
  console.log(`\n[4b] FAILURE MODE — same SVD on RAW counts (no PPMI): structure stays frequency-polluted:`);
  reportNeighbors("svd-raw ", "king", vocab, embRaw, 3);

  // A concrete analogy-flavored sanity check on the PPMI-SVD space: is king closer
  // to queen than to a random stopword? Single number, honestly labeled as toy.
  const kingV = emb[vocab.stoi.get("king")!];
  const queenV = emb[vocab.stoi.get("queen")!];
  const theV = emb[vocab.stoi.get("the")!];
  console.log(`\n[5] sanity (PPMI-SVD space, toy-optimistic absolute values):`);
  console.log(`    cos(king, queen) = ${cosineSimilarity(kingV, queenV).toFixed(3)}`);
  console.log(`    cos(king, "the") = ${cosineSimilarity(kingV, theV).toFixed(3)}`);
  console.log(`    king is ${cosineSimilarity(kingV, queenV) > cosineSimilarity(kingV, theV) ? "CLOSER to queen than to 'the' ✓ (semantic > frequency)" : "closer to 'the' ✗ (structure not recovered)"}`);

  console.log(`\n[done] No optimizer, no autograd. Vectors came from counting + PPMI + power-iteration SVD.`);
}

main();
