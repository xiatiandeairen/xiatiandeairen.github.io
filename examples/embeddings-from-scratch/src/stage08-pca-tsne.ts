// stage08-pca-tsne.ts — making a high-dimensional embedding space VISIBLE.
//
// Chapter 8 thesis: a trained embedding lives in D dimensions (here D=16); humans
// see 2. Every 2D picture of it is a LOSSY projection, and the projection method
// decides which structure survives and which is destroyed. This stage builds two
// projectors from scratch on the SAME trained vectors so the reader can compare:
//
//   1. PCA — a *linear* projection onto the two directions of maximum variance.
//      Hand-rolled: center -> covariance -> power iteration (+ deflation) for the
//      top-2 eigenvectors -> project. Honest because PCA is closed-form linear
//      algebra; we do not need gradients, and pretending we do would obscure that
//      PCA has a unique global answer (unlike t-SNE).
//
//   2. t-SNE — a *nonlinear* embedding that preserves local neighborhoods by
//      matching a high-D Gaussian neighbor distribution P against a low-D
//      Student-t distribution Q, minimizing KL(P||Q) by gradient descent. We drive
//      the gradient through core/autograd (the book's own engine) so the reader
//      SEES it is the same backprop used to train the embeddings, not a special
//      closed form. Local clusters come out tighter than PCA.
//
// Then a perplexity sweep (5 / 15 / 40) produces three visibly different t-SNE
// pictures of ONE dataset — the chapter's hard lesson that a t-SNE plot is a
// hyperparameter-dependent cartoon, not ground truth.
//
// HONESTY about numbers (toy scale): vectors come from ~150 epochs of skip-gram on
// a ~40-word synthetic vocab. Absolute cluster crispness is optimistic vs a real
// corpus. What transfers is the RELATIVE story: PCA explained-variance ratio,
// PCA-vs-t-SNE local tightness, and how perplexity reshapes the map. All printed
// quantities (explained variance, KL loss, neighbor-preservation %) are computed
// from the actual run, not asserted.
//
// Determinism: every stochastic step (weight init, training-pair shuffle, t-SNE Y
// init, power-iteration seed vector) pulls from a seeded Rng. Same seed -> same
// pictures, the book's reproducibility contract.

import { Rng, gaussian, shuffle } from "./core/rng.js";
import {
  Value,
  Vec,
  Mat,
  dot,
  makeMat,
  makeVec,
  vecData,
  sumValues,
} from "./core/autograd.js";
import { Adam, collectParams } from "./core/optim.js";
import { asciiLine, asciiScatter } from "./core/plot.js";
import {
  generateCorpus,
  buildVocab,
  windowPairs,
  Vocab,
  Pair,
} from "./core/text.js";
import { cosineSimilarity, nearestNeighbors } from "./core/eval.js";

// ---------------------------------------------------------------------------
// Step 0 — obtain real trained embeddings.
//
// We can't visualize a space we didn't learn, so this stage trains its own tiny
// skip-gram model rather than importing another stage (importing a stageNN file
// would execute its main()). The training here is intentionally minimal but REAL:
// the loss must actually drop and the geometry must actually emerge, otherwise the
// projections below would be drawing pictures of noise.
// ---------------------------------------------------------------------------

const EMBED_DIM = 16; // high-D source space we will project to 2D
const TRAIN_EPOCHS = 150;
const BATCH_PER_EPOCH = 64; // sampled pairs per epoch; full sweep is unnecessary at toy scale

interface TrainedSpace {
  vocab: Vocab;
  // One detached number[] per token: the learned center embedding. Detached
  // because everything downstream (PCA, neighbor-preservation eval) reads weights,
  // it must never extend the training graph.
  vectors: number[][];
}

function trainSkipGram(): TrainedSpace {
  // Separate Rng per concern (corpus / init / training order) so changing one
  // does not silently shift another's stream — the multi-generator discipline
  // core/rng.ts mandates.
  const corpusRng = new Rng(7);
  const initRng = new Rng(42);
  const orderRng = new Rng(1234);

  const corpus = generateCorpus(corpusRng, 500);
  const vocab = buildVocab(corpus.tokens);
  const allPairs = windowPairs(corpus.sentences, vocab, 2);

  // Two matrices (center "in" and context "out") is the standard skip-gram
  // parameterization; the center matrix is what we visualize, by convention.
  const inEmb: Mat = makeMat(vocab.size, EMBED_DIM, () => gaussian(initRng, 0, 0.3));
  const outEmb: Mat = makeMat(vocab.size, EMBED_DIM, () => gaussian(initRng, 0, 0.3));
  const params = collectParams(inEmb, outEmb);
  const opt = new Adam(params, 0.05);

  const lossCurve: number[] = [];
  for (let epoch = 0; epoch < TRAIN_EPOCHS; epoch++) {
    // Reshuffle each epoch so SGD does not see a fixed pair order (order biases
    // the path). shuffle is in-place + seeded -> reproducible visit sequence.
    shuffle(orderRng, allPairs);
    const batch = allPairs.slice(0, BATCH_PER_EPOCH);

    opt.zeroGrad();
    // Full-softmax skip-gram loss over the batch: for each (center, context),
    // -log softmax(score(center, context)) across the whole vocab. Toy vocab makes
    // the full softmax affordable; this keeps the loss honest (true normalized
    // likelihood, no negative-sampling approximation to explain here).
    const losses = batch.map((p: Pair) => negLogSoftmax(inEmb[p.center], outEmb, p.context));
    const loss = sumValues(losses).div(batch.length);
    loss.backward();
    opt.step();
    lossCurve.push(loss.data);
  }

  console.log("【训练】skip-gram 全 softmax 损失曲线（确认几何来自真实学习，不是噪声）:");
  console.log(asciiLine(lossCurve));
  console.log(
    `  loss ${lossCurve[0].toFixed(3)} -> ${lossCurve[lossCurve.length - 1].toFixed(3)}  ` +
      `(下降 ${(lossCurve[0] - lossCurve[lossCurve.length - 1]).toFixed(3)})\n`,
  );

  const vectors = inEmb.map((row) => vecData(row));
  return { vocab, vectors };
}

// -log softmax score of the true context over the full vocab, built as a Value
// graph so backward() trains the embeddings. Numerically this is fine at toy vocab
// (scores stay small); core/autograd.log clamps the rare underflow.
function negLogSoftmax(center: Vec, outEmb: Mat, contextId: number): Value {
  const scores = outEmb.map((o) => dot(center, o)); // V scores
  // softmax denominator = sum_j exp(score_j); -log p(context) = log Z - score_ctx.
  const expSum = sumValues(scores.map((s) => s.exp()));
  return expSum.log().sub(scores[contextId]);
}

// ---------------------------------------------------------------------------
// Step 1 — PCA from scratch (linear, closed-form).
//
// PCA finds the orthogonal directions along which the data varies most. The top-2
// directions are the eigenvectors of the covariance matrix with the two largest
// eigenvalues. We get them by power iteration (repeatedly multiplying a random
// vector by the covariance matrix converges to the top eigenvector) plus one
// deflation step to peel off the first component before finding the second.
//
// CRITICAL INVARIANT: the data MUST be centered (mean-subtracted) first. Otherwise
// the "direction of maximum variance" is dominated by the offset of the cloud from
// the origin — i.e. you recover the MEAN direction, not the structure. The
// failure-mode demo at the end runs PCA *without* centering to make this visible.
// ---------------------------------------------------------------------------

interface PcaResult {
  coords2d: Array<[number, number]>;
  // Fraction of total variance captured by each of the 2 components. Tells the
  // reader how much of the 16-D structure the 2-D picture actually represents.
  explainedRatio: [number, number];
  // The first principal axis itself (unit vector in the original D-space). The
  // centering failure-mode demo measures how aligned this is with the data's mean
  // direction — that alignment is the robust, always-visible symptom of skipping
  // centering, even when downstream neighbor metrics move little.
  pc1: number[];
}

function computePca(vectors: number[][], centered: boolean, rng: Rng): PcaResult {
  const n = vectors.length;
  const d = vectors[0].length;

  // Center (or deliberately skip centering for the failure demo).
  const mean = new Array(d).fill(0);
  if (centered) {
    for (const v of vectors) for (let j = 0; j < d; j++) mean[j] += v[j] / n;
  }
  const X = vectors.map((v) => v.map((x, j) => x - mean[j]));

  // Covariance matrix C = (1/n) X^T X, a d x d symmetric matrix.
  const cov = symmetricCovariance(X, n, d);

  const totalVar = trace(cov); // sum of eigenvalues = total variance
  const [pc1, lambda1] = topEigenvector(cov, rng);
  const cov2 = deflate(cov, pc1, lambda1); // remove pc1's contribution
  const [pc2, lambda2] = topEigenvector(cov2, rng);

  const coords2d = X.map((row): [number, number] => [dotNum(row, pc1), dotNum(row, pc2)]);
  // Guard totalVar=0 (all vectors identical) -> ratio 0 rather than NaN.
  const denom = totalVar || 1;
  return { coords2d, explainedRatio: [lambda1 / denom, lambda2 / denom], pc1 };
}

function symmetricCovariance(X: number[][], n: number, d: number): number[][] {
  const cov: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const row of X) {
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) {
        const val = (row[a] * row[b]) / n;
        cov[a][b] += val;
        if (a !== b) cov[b][a] += val; // exploit symmetry: compute upper, mirror
      }
    }
  }
  return cov;
}

// Power iteration: v <- normalize(C v) converges to the dominant eigenvector
// because C^k v amplifies the component along the largest-|eigenvalue| direction
// fastest. Returns [eigenvector, eigenvalue]. Fixed iteration count is enough at
// toy d with a clear spectral gap; the seeded start vector keeps it deterministic.
function topEigenvector(C: number[][], rng: Rng): [number[], number] {
  const d = C.length;
  let v = normalize(Array.from({ length: d }, () => gaussian(rng, 0, 1)));
  for (let iter = 0; iter < 200; iter++) {
    const Cv = matVec(C, v);
    v = normalize(Cv);
  }
  // Rayleigh quotient v^T C v gives the eigenvalue for the converged eigenvector.
  const lambda = dotNum(v, matVec(C, v));
  return [v, lambda];
}

// Deflation: C' = C - lambda * (v v^T) removes the eigenpair (v, lambda) so the
// next power iteration finds the SECOND eigenvector. Without this, power iteration
// would just re-find pc1.
function deflate(C: number[][], v: number[], lambda: number): number[][] {
  return C.map((row, a) => row.map((c, b) => c - lambda * v[a] * v[b]));
}

// ---------------------------------------------------------------------------
// Step 2 — t-SNE from scratch, gradient via core/autograd.
//
// Pipeline:
//   (a) P: high-D affinities. For each point i, p_{j|i} ~ exp(-||xi-xj||^2 / 2σ_i^2),
//       with σ_i chosen so the perplexity (effective neighbor count) matches a
//       target. Symmetrize: p_{ij} = (p_{j|i}+p_{i|j}) / 2N. Computed in plain
//       numbers — P is a fixed target, not a parameter, so no graph needed.
//   (b) Q: low-D affinities under a Student-t (1 dof) kernel q_{ij} ~ (1+||yi-yj||^2)^-1.
//       Built as a Value graph over the trainable 2-D coordinates Y.
//   (c) Loss = KL(P||Q) = sum_{i≠j} p_{ij} log(p_{ij}/q_{ij}). Minimize by Adam on Y.
//
// The Student-t heavy tail in Q is the "t" in t-SNE: it lets distant points sit far
// apart without exploding the gradient, fixing the "crowding problem" plain SNE has.
// ---------------------------------------------------------------------------

interface TsneResult {
  coords2d: Array<[number, number]>;
  klCurve: number[];
}

function computeTsne(
  vectors: number[][],
  perplexity: number,
  epochs: number,
  rng: Rng,
): TsneResult {
  const n = vectors.length;
  const P = highDimAffinities(vectors, perplexity); // fixed target, plain numbers

  // Y: trainable 2-D coordinates, one Vec[2] per point. Small init so early
  // gradients are well-scaled (standard t-SNE init is ~N(0, 1e-4); we use 0.01).
  const Y: Vec[] = [];
  for (let i = 0; i < n; i++) Y.push(makeVec(2, () => gaussian(rng, 0, 0.01)));
  const params = collectParams(Y as unknown as Value[][]);
  const opt = new Adam(params, 0.5); // larger lr: t-SNE landscapes are shallow early

  const klCurve: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    opt.zeroGrad();
    const loss = klLoss(P, Y, n);
    loss.backward();
    opt.step();
    klCurve.push(loss.data);
  }

  const coords2d = Y.map((y): [number, number] => [y[0].data, y[1].data]);
  return { coords2d, klCurve };
}

// KL(P||Q) as a Value graph. Q is normalized INSIDE the graph (its denominator is
// the sum of all unnormalized Student-t affinities) so gradients flow through the
// normalization — dropping that is a classic t-SNE bug that collapses the map.
function klLoss(P: number[][], Y: Vec[], n: number): Value {
  // Unnormalized Student-t affinities w_{ij} = 1 / (1 + ||yi - yj||^2), i != j.
  const w: Value[][] = Array.from({ length: n }, () => new Array<Value>(n));
  const wSumTerms: Value[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = Y[i][0].sub(Y[j][0]);
      const dy = Y[i][1].sub(Y[j][1]);
      const sq = dx.mul(dx).add(dy.mul(dy));
      const wij = sq.add(1).pow(-1); // (1 + dist^2)^-1
      w[i][j] = wij;
      wSumTerms.push(wij);
    }
  }
  const Z = sumValues(wSumTerms); // global normalizer for Q

  // KL = sum_{i!=j} p_ij * (log p_ij - log q_ij), q_ij = w_ij / Z.
  // The constant p_ij*log p_ij term does not affect gradients but we include it so
  // the printed KL is the true divergence, not an offset proxy (honesty of the
  // number). We skip p_ij = 0 entries (log 0 undefined; their KL contribution is 0).
  const logZ = Z.log();
  const terms: Value[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const pij = P[i][j];
      if (pij <= 0) continue;
      // p*(log p - log q) = p*(log p - log w + log Z)
      const logQ = w[i][j].log().sub(logZ);
      terms.push(logQ.mul(-pij).add(pij * Math.log(pij)));
    }
  }
  return sumValues(terms);
}

// High-D affinities P with per-point bandwidth σ_i tuned by binary search so each
// point's conditional distribution has the target perplexity. Perplexity ≈ the
// effective number of neighbors; it is THE knob the sweep below varies.
function highDimAffinities(vectors: number[][], perplexity: number): number[][] {
  const n = vectors.length;
  const dist2: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < vectors[i].length; k++) {
        const dd = vectors[i][k] - vectors[j][k];
        s += dd * dd;
      }
      dist2[i][j] = s;
      dist2[j][i] = s;
    }
  }

  const targetEntropy = Math.log(perplexity); // perplexity = 2^entropy(bits) -> use nats consistently
  const condP: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    condP[i] = conditionalRowForPerplexity(dist2[i], i, targetEntropy);
  }

  // Symmetrize into a joint distribution: p_ij = (p_{j|i} + p_{i|j}) / 2N.
  const P: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      P[i][j] = (condP[i][j] + condP[j][i]) / (2 * n);
    }
  }
  return P;
}

// Binary-search the precision beta = 1/(2σ^2) so row i's Gaussian neighbor
// distribution hits the target Shannon entropy (== log perplexity). This is the
// standard t-SNE per-point bandwidth fit; without it, dense and sparse regions
// would get the same σ and the map would distort.
function conditionalRowForPerplexity(
  d2: number[],
  selfIdx: number,
  targetEntropy: number,
): number[] {
  let betaLo = 1e-8;
  let betaHi = 1e8;
  let beta = 1.0;
  let row = new Array(d2.length).fill(0);

  for (let iter = 0; iter < 50; iter++) {
    let sum = 0;
    for (let j = 0; j < d2.length; j++) {
      if (j === selfIdx) {
        row[j] = 0;
        continue;
      }
      row[j] = Math.exp(-beta * d2[j]);
      sum += row[j];
    }
    if (sum <= 0) sum = 1e-12; // degenerate: all neighbors infinitely far
    let entropy = 0;
    for (let j = 0; j < d2.length; j++) {
      const p = row[j] / sum;
      row[j] = p;
      if (p > 1e-12) entropy -= p * Math.log(p);
    }
    const diff = entropy - targetEntropy;
    if (Math.abs(diff) < 1e-5) break;
    // Higher beta (smaller σ) -> lower entropy (fewer effective neighbors).
    if (diff > 0) {
      betaLo = beta;
      beta = betaHi === 1e8 ? beta * 2 : (beta + betaHi) / 2;
    } else {
      betaHi = beta;
      beta = (beta + betaLo) / 2;
    }
  }
  return row;
}

// ---------------------------------------------------------------------------
// Step 3 — neighbor-preservation metric (honest comparison of the two projections).
//
// "t-SNE clusters look tighter" is a vibe. We turn it into a number: for each word,
// what fraction of its k nearest neighbors in the ORIGINAL 16-D space are still
// among its k nearest neighbors in the 2-D projection. Higher = the projection kept
// local structure. This is the quantitative claim behind "t-SNE preserves locality
// better than PCA".
// ---------------------------------------------------------------------------

function neighborPreservation(
  highD: number[][],
  low2d: Array<[number, number]>,
  k: number,
): number {
  const n = highD.length;
  const lowVectors = low2d.map(([a, b]) => [a, b]);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const hiN = new Set(nearestNeighbors(highD[i], highD, k).map((x) => x.index));
    const loN = nearestNeighbors(lowVectors[i], lowVectors, k).map((x) => x.index);
    let overlap = 0;
    for (const idx of loN) if (hiN.has(idx)) overlap++;
    total += overlap / k;
  }
  return total / n;
}

// ---------------------------------------------------------------------------
// Plain-number linear-algebra helpers (PCA path runs outside the autograd graph).
// ---------------------------------------------------------------------------

function dotNum(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function matVec(M: number[][], v: number[]): number[] {
  return M.map((row) => dotNum(row, v));
}

function normalize(v: number[]): number[] {
  const n = Math.sqrt(dotNum(v, v)) || 1;
  return v.map((x) => x / n);
}

function trace(M: number[][]): number {
  let s = 0;
  for (let i = 0; i < M.length; i++) s += M[i][i];
  return s;
}

// Single-character labels for asciiScatter, picked so clusters are readable on the
// grid. We map each token to its first letter; collisions are expected and the plot
// marks them '*', which is itself informative (cluster collapsed to one cell).
function scatterLabels(vocab: Vocab): string[] {
  return vocab.itos.map((t) => t[0]);
}

// Print which tokens map to which scatter glyph + their cluster, so the reader can
// decode the picture. Limited to the semantically loaded words; filler is noise.
function printLegend(vocab: Vocab): void {
  const interesting = [
    "king",
    "queen",
    "prince",
    "princess",
    "man",
    "woman",
    "dog",
    "cat",
    "mouse",
    "bird",
    "bread",
    "cheese",
    "rice",
    "fish",
    "north",
    "south",
    "east",
    "west",
  ];
  const present = interesting.filter((t) => vocab.stoi.has(t));
  console.log("  图例（glyph = 词首字母, 同字母会 '*' 碰撞）:");
  console.log("    " + present.map((t) => `${t[0]}=${t}`).join("  "));
}

// ---------------------------------------------------------------------------
// main — orchestrate the chapter's narrative end to end.
// ---------------------------------------------------------------------------

function main(): void {
  console.log("=".repeat(70));
  console.log("Stage 08 — 降维可视化: PCA 与 t-SNE 把 16 维词向量画到 2D");
  console.log("=".repeat(70) + "\n");

  const space = trainSkipGram();
  const { vocab, vectors } = space;
  console.log(`【数据】vocab=${vocab.size} 词, 每词 ${EMBED_DIM} 维嵌入\n`);

  const labels = scatterLabels(vocab);
  const pcaRng = new Rng(99);

  // ---- PCA (correct: centered) ----
  console.log("-".repeat(70));
  console.log("① PCA（线性投影，已中心化）");
  console.log("-".repeat(70));
  const pca = computePca(vectors, true, pcaRng);
  console.log(asciiScatter(pca.coords2d, labels));
  console.log(
    `  解释方差比: PC1=${(pca.explainedRatio[0] * 100).toFixed(1)}%  ` +
      `PC2=${(pca.explainedRatio[1] * 100).toFixed(1)}%  ` +
      `(两轴合计 ${((pca.explainedRatio[0] + pca.explainedRatio[1]) * 100).toFixed(1)}% — ` +
      `剩余维度的结构在 2D 图里看不到)`,
  );
  printLegend(vocab);
  const pcaPreserve = neighborPreservation(vectors, pca.coords2d, 3);
  console.log(`  近邻保留率(k=3): ${(pcaPreserve * 100).toFixed(1)}%\n`);

  // ---- t-SNE (perplexity 15, the "default" view) ----
  console.log("-".repeat(70));
  console.log("② t-SNE（非线性，KL 梯度下降，perplexity=15）");
  console.log("-".repeat(70));
  const tsne = computeTsne(vectors, 15, 300, new Rng(2024));
  console.log("  KL 损失收敛:");
  console.log(asciiLine(tsne.klCurve));
  console.log(
    `  KL ${tsne.klCurve[0].toFixed(4)} -> ${tsne.klCurve[tsne.klCurve.length - 1].toFixed(4)}\n`,
  );
  console.log(asciiScatter(tsne.coords2d, labels));
  printLegend(vocab);
  const tsnePreserve = neighborPreservation(vectors, tsne.coords2d, 3);
  console.log(`  近邻保留率(k=3): ${(tsnePreserve * 100).toFixed(1)}%`);
  console.log(
    `  PCA ${(pcaPreserve * 100).toFixed(1)}% vs t-SNE ${(tsnePreserve * 100).toFixed(1)}% — ` +
      `t-SNE 通常更高: 它专门优化局部近邻, 而 PCA 只保全局方差\n`,
  );

  // ---- Perplexity sweep: same data, three different maps ----
  console.log("-".repeat(70));
  console.log("③ perplexity 扫描 5 / 15 / 40 — 同一份数据, 三张差异很大的图");
  console.log("-".repeat(70));
  console.log("   教训: t-SNE 图的形状/簇间距强烈依赖 perplexity, 不是数据的客观属性。\n");
  for (const perp of [5, 15, 40]) {
    // Fresh Rng per perplexity so each map's init is independent yet reproducible.
    const sweep = computeTsne(vectors, perp, 300, new Rng(2024));
    console.log(`   perplexity=${perp}  (最终 KL=${sweep.klCurve[sweep.klCurve.length - 1].toFixed(4)})`);
    console.log(asciiScatter(sweep.coords2d, labels, 50, 14));
    const preserve = neighborPreservation(vectors, sweep.coords2d, 3);
    console.log(`     近邻保留率(k=3): ${(preserve * 100).toFixed(1)}%\n`);
  }

  // ---- Failure mode 1: PCA without centering ----
  console.log("-".repeat(70));
  console.log("④ 失败模式 A: PCA 前不中心化 → 把『均值方向』当成主成分");
  console.log("-".repeat(70));
  const meanVec = meanVector(vectors);
  const meanNorm = vectorNorm(meanVec);
  const meanDir = normalize(meanVec);
  const pcaNoCenter = computePca(vectors, false, new Rng(99));

  // Robust, always-visible symptom: how aligned is the uncentered PC1 with the
  // MEAN direction? |cos| near 1 means PC1 spent itself describing where the cloud
  // sits, not how it varies. The centered PC1 is (by construction) free of the
  // mean, so its alignment is near 0 — the contrast is the whole point.
  const uncenteredAlign = Math.abs(cosineSimilarity(pcaNoCenter.pc1, meanDir));
  const centeredAlign = Math.abs(cosineSimilarity(pca.pc1, meanDir));
  const uncenteredPreserve = neighborPreservation(vectors, pcaNoCenter.coords2d, 3);

  console.log(
    `   嵌入均值向量的模长 = ${meanNorm.toFixed(3)} (非零 → 数据云偏离原点)`,
  );
  console.log(asciiScatter(pcaNoCenter.coords2d, labels));
  console.log(
    `   PC1 与『均值方向』的 |cosine|: ` +
      `不中心化 ${uncenteredAlign.toFixed(3)} vs 中心化 ${centeredAlign.toFixed(3)}`,
  );
  console.log(
    `     → 不中心化时 PC1 几乎指向均值方向(=数据云的偏移), 把第一个、` +
      `也是方差最大的坐标轴浪费在『所有点共有的常量』上, 而非区分词的结构。`,
  );
  console.log(
    `   近邻保留率(k=3): 正确中心化 ${(pcaPreserve * 100).toFixed(1)}% vs ` +
      `不中心化 ${(uncenteredPreserve * 100).toFixed(1)}% ` +
      `(此 toy 数据均值模长不算大, 近邻指标差距小; ` +
      `偏移越大、维度越高, 浪费一根主轴的代价越严重)\n`,
  );

  // ---- Failure mode 2: over-reading t-SNE inter-cluster distances ----
  console.log("-".repeat(70));
  console.log("⑤ 失败模式 B: 按 t-SNE 簇间距离下语义结论 (不可做)");
  console.log("-".repeat(70));
  demoInterClusterDistanceTrap(vectors, vocab);

  console.log("\n" + "=".repeat(70));
  console.log("小结: PCA = 全局线性、可解释方差、唯一解; t-SNE = 局部非线性、");
  console.log("依赖 perplexity、多解。两者都是 lossy 投影, 结论要回到原始 16 维验证。");
  console.log("=".repeat(70));
}

// Demonstrate why t-SNE inter-cluster GAPS are not meaningful: we measure the true
// high-D cosine distance between two cluster centroids, then measure the 2-D
// distance between the same centroids in two different perplexity maps. The high-D
// relationship is fixed; the 2-D gap swings with perplexity — so reading "these
// clusters are far apart, therefore semantically unrelated" off ONE t-SNE plot is
// unsound.
function demoInterClusterDistanceTrap(vectors: number[][], vocab: Vocab): void {
  const royalty = ["king", "queen", "prince", "princess"].filter((t) => vocab.stoi.has(t));
  const animals = ["dog", "cat", "mouse", "bird"].filter((t) => vocab.stoi.has(t));
  if (royalty.length === 0 || animals.length === 0) {
    console.log("   (toy vocab 缺簇, 跳过)");
    return;
  }

  const cReg = centroid(royalty.map((t) => vectors[vocab.stoi.get(t)!]));
  const cAni = centroid(animals.map((t) => vectors[vocab.stoi.get(t)!]));
  const highDcos = cosineSimilarity(cReg, cAni);
  console.log(
    `   原始 16 维: royalty 簇质心 与 animals 簇质心 cosine = ${highDcos.toFixed(3)} (固定不变)`,
  );

  console.log("   同一对簇在不同 perplexity 的 2D 图里, 质心欧氏距离:");
  for (const perp of [5, 15, 40]) {
    const map = computeTsne(vectors, perp, 300, new Rng(2024));
    const coordOf = (t: string) => map.coords2d[vocab.stoi.get(t)!];
    const cReg2d = centroid2d(royalty.map(coordOf));
    const cAni2d = centroid2d(animals.map(coordOf));
    const gap = Math.hypot(cReg2d[0] - cAni2d[0], cReg2d[1] - cAni2d[1]);
    console.log(`     perplexity=${perp}: 2D 簇间距 = ${gap.toFixed(3)}`);
  }
  console.log(
    "   → 高维关系恒定, 但 2D 簇间距随 perplexity 大幅变化。" +
      "结论必须回到高维 cosine, 不能照 t-SNE 图的间距读。",
  );
}

function meanVector(vectors: number[][]): number[] {
  const d = vectors[0].length;
  const m = new Array(d).fill(0);
  for (const v of vectors) for (let j = 0; j < d; j++) m[j] += v[j] / vectors.length;
  return m;
}

function vectorNorm(v: number[]): number {
  return Math.sqrt(dotNum(v, v));
}

function centroid(vs: number[][]): number[] {
  const d = vs[0].length;
  const c = new Array(d).fill(0);
  for (const v of vs) for (let j = 0; j < d; j++) c[j] += v[j] / vs.length;
  return c;
}

function centroid2d(pts: Array<[number, number]>): [number, number] {
  let x = 0;
  let y = 0;
  for (const [a, b] of pts) {
    x += a / pts.length;
    y += b / pts.length;
  }
  return [x, y];
}

main();
