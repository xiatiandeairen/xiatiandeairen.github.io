// stage03-linalg.ts — Linear algebra as the language of data: PCA, SVD low-rank
// compression, and cosine retrieval, all built from core/linalg with NO BLAS.
//
// WHY this is the linear-algebra stage: every ML pipeline a builder ships is, under the
//   hood, three operations from this file — PCA (find the directions data varies in),
//   SVD (compress a matrix to its essential rank), and cosine search (rank vectors by
//   angle). Seeing them assembled from dot/matmul/normalize alone is the whole point:
//   there is no magic, just eigenvectors of a covariance matrix and a few matmuls.
//
// HONEST-NUMBER NOTE: the point cloud, the "image", and the embedding corpus are all
//   SEEDED synthetic data. Absolute numbers (variance, Frobenius error, similarity) are
//   toy and optimistic — real images are not rank-3, real embeddings are 768-dim. What
//   TRANSFERS is the SHAPE of the trade-offs: explained-variance concentrates in few PCs,
//   reconstruction error falls monotonically as rank rises, and cosine ranks by direction
//   not magnitude. Those relationships are mathematically exact here, not approximated.
//
// METHOD NOTE on eigendecomposition: we have no LAPACK, so eigenpairs come from POWER
//   ITERATION + DEFLATION (subtract λ·v·vᵀ to expose the next eigenvector). This is the
//   honest from-scratch route and exposes the real failure mode below: power iteration
//   needs a spectral GAP. When the top two eigenvalues are near-equal the dominant
//   direction is ill-defined and the "principal" axis flips run-to-run.

import { mulberry32, sampleNormal, type Rng } from "./core/rng.js";
import {
  dot,
  matVec,
  matmul,
  transpose,
  normalize,
  scale,
  norm,
} from "./core/linalg.js";
import { asciiBar } from "./core/plot.js";

// ---------------------------------------------------------------------------
// Eigen primitives (symmetric matrices only — covariance and AᵀA are symmetric).
// ---------------------------------------------------------------------------

interface EigenPair {
  value: number;
  vector: number[];
}

/**
 * Dominant eigenpair of a SYMMETRIC matrix via power iteration.
 * INVARIANT: caller passes a symmetric S; we do not check (cheap correctness relies on
 *   it — covariance and AᵀA are symmetric by construction).
 * WHY a fixed iteration count instead of a residual tolerance: it makes the cost (and the
 *   failure mode) explicit. If the spectral gap is tiny, MORE iterations do not help — the
 *   iterate just drifts in the degenerate eigenspace; we surface that rather than hide it
 *   behind a "converged" flag.
 * The Rayleigh quotient vᵀSv is the eigenvalue estimate for the returned (unit) vector.
 */
function powerIteration(S: readonly number[][], rng: Rng, iters = 200): EigenPair {
  const n = S.length;
  // Random start so we do not accidentally seed exactly orthogonal to the top eigenvector
  // (a zero starting component never grows — that is the classic "it returned the 2nd
  // eigenvector" bug).
  let v = normalize(Array.from({ length: n }, () => sampleNormal(rng)));
  for (let i = 0; i < iters; i++) {
    const Sv = matVec(S, v);
    const next = normalize(Sv);
    if (norm(next) === 0) break; // S annihilated v (zero matrix / null space) — stop.
    v = next;
  }
  const value = dot(v, matVec(S, v)); // Rayleigh quotient
  return { value, vector: v };
}

/**
 * Top-k eigenpairs of a symmetric matrix by power iteration + DEFLATION.
 * Deflation: after finding (λ, v), subtract λ·vvᵀ so the next power iteration sees the
 *   residual spectrum and converges to the next eigenvector. Errors accumulate across
 *   deflations, so this is fine for k ≤ a handful (PCA top-2, SVD rank ≤ 8), not k = 500.
 */
function topEigenpairs(S: readonly number[][], k: number, rng: Rng): EigenPair[] {
  let residual = S.map((row) => row.slice()); // mutable copy; we deflate in place
  const out: EigenPair[] = [];
  for (let p = 0; p < k; p++) {
    const { value, vector } = powerIteration(residual, rng);
    out.push({ value, vector });
    // residual := residual - value * (vector ⊗ vector)
    residual = residual.map((row, i) =>
      row.map((x, j) => x - value * vector[i] * vector[j]),
    );
  }
  return out;
}

// ---------------------------------------------------------------------------
// Part 1 — PCA on a 2D point cloud.
// ---------------------------------------------------------------------------

/** Column means of an (n × d) data matrix (one row per sample). */
function columnMeans(data: readonly number[][]): number[] {
  const n = data.length;
  const d = data[0].length;
  const sums = new Array(d).fill(0);
  for (const row of data) for (let j = 0; j < d; j++) sums[j] += row[j];
  return sums.map((s) => s / n);
}

/** Subtract a per-column mean vector from every row -> centered copy. */
function centerRows(data: readonly number[][], mean: readonly number[]): number[][] {
  return data.map((row) => row.map((x, j) => x - mean[j]));
}

/**
 * Covariance matrix of an (n × d) data matrix as Xᶜᵀ Xᶜ / (n-1).
 * INVARIANT: pass CENTERED data (Xᶜ). If you pass raw data the "covariance" is actually a
 *   second-moment matrix about the origin, and its top eigenvector points at the data's
 *   MEAN, not its spread — that is exactly the failure mode demonstrated below.
 */
function covariance(centered: readonly number[][]): number[][] {
  const n = centered.length;
  const cov = matmul(transpose(centered), centered);
  return cov.map((row) => scale(row, 1 / (n - 1)));
}

/** Sum of the diagonal — total variance, the denominator for explained-variance ratios. */
function trace(m: readonly number[][]): number {
  let t = 0;
  for (let i = 0; i < m.length; i++) t += m[i][i];
  return t;
}

function demoPca(rng: Rng): void {
  console.log("\n--- ① PCA：从协方差矩阵的特征向量找主方向 ---");
  // Generate a cloud that lies mostly along the direction (1, 0.5): wide spread on one
  // axis, narrow on the other, then offset far from the origin so "not centering" hurts.
  const n = 400;
  const dirSpread = 4.0; // std along the principal direction
  const offSpread = 0.6; // std perpendicular to it
  const center = [10, 6]; // far from origin -> uncentered PCA will chase this
  const axis = normalize([1, 0.5]);
  const perp = [-axis[1], axis[0]]; // unit vector ⟂ axis
  const data: number[][] = Array.from({ length: n }, () => {
    const a = sampleNormal(rng, 0, dirSpread);
    const b = sampleNormal(rng, 0, offSpread);
    return [
      center[0] + a * axis[0] + b * perp[0],
      center[1] + a * axis[1] + b * perp[1],
    ];
  });

  const mean = columnMeans(data);
  const centered = centerRows(data, mean);
  const cov = covariance(centered);
  const eig = topEigenpairs(cov, 2, rng);
  const total = trace(cov);

  console.log(`样本数 n=${n}，真实主轴方向 ≈ [${axis[0].toFixed(3)}, ${axis[1].toFixed(3)}]`);
  console.log(`数据中心 (列均值)        = [${mean.map((x) => x.toFixed(3)).join(", ")}]`);
  for (let i = 0; i < eig.length; i++) {
    const v = eig[i].vector;
    // Sign of an eigenvector is arbitrary; flip to the +x half-plane for stable printing.
    const s = v[0] < 0 ? -1 : 1;
    const ratio = eig[i].value / total;
    console.log(
      `PC${i + 1}: 方向 [${(s * v[0]).toFixed(4)}, ${(s * v[1]).toFixed(4)}]  ` +
        `特征值 ${eig[i].value.toFixed(4)}  解释方差 ${(ratio * 100).toFixed(2)}%`,
    );
  }
  // Project the cloud onto PC1 and report how much of the variance the 1-D code keeps.
  const pc1 = eig[0].vector;
  const coords1d = centered.map((row) => dot(row, pc1));
  const keptVar = (coords1d.reduce((s, x) => s + x * x, 0) / (n - 1)) / total;
  console.log(
    `降到 1 维（投影到 PC1）后保留方差 = ${(keptVar * 100).toFixed(2)}% ` +
      `→ 用 1 个数代替 2 个数，几乎不丢信息`,
  );
  const angleErrDeg =
    (Math.acos(Math.min(1, Math.abs(dot(normalize(pc1), axis)))) * 180) / Math.PI;
  console.log(`PC1 与真实主轴夹角 = ${angleErrDeg.toFixed(2)}°（≈0 说明 PCA 找对了）`);

  // FAILURE MODE: skip centering. The covariance() helper computes XᵀX/(n-1) on RAW data,
  // whose top eigenvector is dragged toward the data MEAN (here ≈[10,6]), not the spread.
  console.log("\n  失败模式：忘了中心化（直接对原始数据做 PCA）");
  const covRaw = covariance(data); // raw, uncentered — the bug
  const eigRaw = topEigenpairs(covRaw, 1, rng);
  const vr = eigRaw[0].vector;
  const sr = vr[0] < 0 ? -1 : 1;
  const meanDir = normalize(mean);
  const wrongAngle =
    (Math.acos(Math.min(1, Math.abs(dot(normalize(vr), axis)))) * 180) / Math.PI;
  const meanAngle =
    (Math.acos(Math.min(1, Math.abs(dot(normalize(vr), meanDir)))) * 180) / Math.PI;
  console.log(
    `  未中心化 PC1 方向 [${(sr * vr[0]).toFixed(4)}, ${(sr * vr[1]).toFixed(4)}]`,
  );
  console.log(
    `  它与真实主轴差 ${wrongAngle.toFixed(2)}°，却与「指向均值的方向」只差 ` +
      `${meanAngle.toFixed(2)}° → 没中心化时主成分追的是位置不是形状`,
  );
}

// ---------------------------------------------------------------------------
// Part 2 — SVD low-rank image compression.
// ---------------------------------------------------------------------------

interface Svd {
  U: number[][]; // m × r  (left singular vectors as columns)
  s: number[]; //  r     (singular values, descending)
  Vt: number[][]; // r × n (right singular vectors as rows)
}

/**
 * Thin SVD via the symmetric eigendecomposition of AᵀA.
 *   AᵀA = V Σ² Vᵀ  → right vectors V and singular values σ = sqrt(eigenvalue).
 *   uᵢ = A vᵢ / σᵢ → left vectors.
 * WHY AᵀA and not AAᵀ: we choose the smaller Gram matrix (n × n here) so power iteration
 *   runs on fewer dimensions. INVARIANT: σ²>0 to form uᵢ; near-zero singular values are
 *   clamped (their left vector is irrelevant to a low-rank reconstruction anyway).
 * LIMIT: deflation error grows with rank, so this is a teaching SVD for r ≤ ~8, not a
 *   production decomposition. We only ever ask for the top few ranks, which is the point.
 */
function svd(A: readonly number[][], rank: number, rng: Rng): Svd {
  const At = transpose(A);
  const gram = matmul(At, A); // n × n, symmetric PSD
  const eig = topEigenpairs(gram, rank, rng);
  const s: number[] = [];
  const V: number[][] = []; // columns are right singular vectors
  const U: number[][] = []; // columns are left singular vectors
  for (const { value, vector } of eig) {
    const sigma = Math.sqrt(Math.max(0, value));
    s.push(sigma);
    V.push(vector);
    // uᵢ = A vᵢ / σᵢ ; if σ≈0 the column is arbitrary, use a zero vector (it contributes 0).
    const Av = matVec(A, vector);
    U.push(sigma > 1e-9 ? scale(Av, 1 / sigma) : Av.map(() => 0));
  }
  // U/V columns -> store U as m×r (columns = left vectors) and Vt as r×n (rows = right).
  return { U: transpose(U), s, Vt: V };
}

/** Reconstruct A from its rank-k factors: Aₖ = Σ σᵢ uᵢ vᵢᵀ (sum of the top-k outer products). */
function reconstruct(svd: Svd, k: number): number[][] {
  const m = svd.U.length;
  const n = svd.Vt[0].length;
  const out: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let r = 0; r < k; r++) {
    const sigma = svd.s[r];
    for (let i = 0; i < m; i++) {
      const ui = svd.U[i][r];
      const vrow = svd.Vt[r];
      for (let j = 0; j < n; j++) out[i][j] += sigma * ui * vrow[j];
    }
  }
  return out;
}

/** Frobenius norm of (A - B): sqrt of the sum of squared element differences. */
function frobeniusDiff(A: readonly number[][], B: readonly number[][]): number {
  let s = 0;
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[0].length; j++) {
      const d = A[i][j] - B[i][j];
      s += d * d;
    }
  return Math.sqrt(s);
}

function demoSvd(rng: Rng): void {
  console.log("\n--- ② SVD 图像压缩：低秩近似换存储 ---");
  // A synthetic 32×32 "grayscale image" with intrinsic low rank: a few smooth gradients
  // plus a little noise. Real photos are not this clean, but the rank/error curve shape is.
  const m = 32;
  const n = 32;
  const img: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const g1 = Math.sin((Math.PI * i) / m) * Math.cos((Math.PI * j) / n); // rank-1 pattern
      const g2 = 0.4 * Math.sin((2 * Math.PI * j) / n); // a second column pattern
      const g3 = 0.2 * ((i + j) / (m + n)); // a ramp
      return g1 + g2 + g3 + sampleNormal(rng, 0, 0.02); // small noise lifts the tail of σ
    }),
  );

  const maxRank = 8;
  const dec = svd(img, maxRank, rng);
  const fullNorm = frobeniusDiff(
    img,
    img.map((row) => row.map(() => 0)),
  ); // ‖A‖_F, used to normalize the error
  console.log(`原始矩阵 ${m}×${n} = ${m * n} 个数，‖A‖_F = ${fullNorm.toFixed(4)}`);
  console.log("秩 k | 存储数 | 压缩比 | 重构误差(Frobenius) | 相对误差");
  const errSeries: number[] = [];
  for (let k = 1; k <= maxRank; k++) {
    const approx = reconstruct(dec, k);
    const err = frobeniusDiff(img, approx);
    errSeries.push(err);
    // Rank-k factors store k·(m + n + 1) numbers: k left vectors, k right vectors, k σ.
    const stored = k * (m + n + 1);
    const ratio = (m * n) / stored;
    const rel = err / fullNorm;
    console.log(
      `  ${String(k).padStart(2)} | ${String(stored).padStart(6)} | ` +
        `${ratio.toFixed(2)}× | ${err.toFixed(5).padStart(18)} | ${(rel * 100).toFixed(3)}%`,
    );
  }
  console.log("误差随秩单调下降（每条 bar 越短越好）：");
  console.log(
    asciiBar(
      errSeries.map((_, i) => `k=${i + 1}`),
      errSeries,
    ),
  );
  console.log(
    "注：toy 图本身近似低秩，所以 k=3 误差已极小；真实照片需要大得多的 k，" +
      "但「误差随 k 单调降、前几个奇异值占大头」这个趋势是真的。",
  );
}

// ---------------------------------------------------------------------------
// Part 3 — Cosine-similarity top-k retrieval.
// ---------------------------------------------------------------------------

interface Hit {
  id: number;
  score: number;
}

/**
 * Cosine similarity = dot(a,b) / (‖a‖·‖b‖) = the cosine of the angle between vectors.
 * WHY cosine and not raw dot or Euclidean distance: retrieval cares about DIRECTION
 *   (topic), not magnitude (document length / term frequency scale). Two docs about the
 *   same thing should rank close even if one is 10× longer. Cosine divides out magnitude.
 * FAILURE MODE: a zero vector has no direction; normalize() returns zero, so the score is
 *   0 (not NaN). That is a deliberate "no information" answer, not a crash.
 */
function cosine(a: readonly number[], b: readonly number[]): number {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0; // undefined angle -> treat as no similarity
  return dot(a, b) / (na * nb);
}

/** Top-k corpus rows by cosine similarity to the query, highest first. */
function topKBySimilarity(
  query: readonly number[],
  corpus: readonly number[][],
  k: number,
): Hit[] {
  const scored = corpus.map((v, id) => ({ id, score: cosine(query, v) }));
  // Sort desc by score; ties keep lower id first (stable enough for a deterministic demo).
  scored.sort((x, y) => y.score - x.score || x.id - y.id);
  return scored.slice(0, k);
}

function demoRetrieval(rng: Rng): void {
  console.log("\n--- ③ 余弦相似度检索 top-k ---");
  // A tiny 5-dim "embedding" corpus. Build 3 latent topics, then make docs as noisy
  // mixtures of them so similarity is meaningful (not random).
  // Topics are NOT orthogonal: A and B share a dimension. That overlap is what lets a
  // huge off-topic vector leak dot-product score onto the query — the realistic case.
  const topicA = normalize([1, 1, 1, 0, 0]); // shares dims 0,1 with B below
  const topicB = normalize([1, 1, 0, 0, 0]);
  const topicC = normalize([0, 0, 0, 1, 1]);
  const topics = [topicA, topicB, topicC];
  const labels = ["A", "B", "C"];

  // Hand-built corpus so the dot-vs-cosine divergence is a designed teaching case, not an
  // accident of the seed: one B doc is a perfect topic match but SHORT; one A doc is
  // off-topic but HUGE. Cosine ranks the B doc first (right answer); raw dot ranks the
  // giant A doc first (wrong answer) purely because of its magnitude.
  const corpus: number[][] = [];
  const docTopic: string[] = [];
  const addDoc = (topicIdx: number, mag: number) => {
    docTopic.push(labels[topicIdx]);
    corpus.push(topics[topicIdx].map((x) => x * mag + sampleNormal(rng, 0, 0.03)));
  };
  addDoc(1, 1.0); // doc#0  B, short  — the true best match by direction
  addDoc(0, 9.0); // doc#1  A, HUGE   — off-topic but long; dot product loves it
  addDoc(1, 2.5); // doc#2  B, medium
  addDoc(2, 4.0); // doc#3  C, large but orthogonal-ish to query
  addDoc(1, 0.8); // doc#4  B, short
  addDoc(0, 1.2); // doc#5  A, small

  const query = topicB; // looking for topic B
  const hits = topKBySimilarity(query, corpus, 3);
  console.log("查询属于主题 B。余弦 top-3（应全是 B，无视模长）：");
  for (const h of hits) {
    console.log(
      `  doc#${h.id}  主题=${docTopic[h.id]}  ‖doc‖=${norm(corpus[h.id]).toFixed(2)}  ` +
        `余弦=${h.score.toFixed(4)}`,
    );
  }
  const cosTopWrong = docTopic[hits[0].id] !== "B";
  console.log(
    `  余弦第一名是 ${docTopic[hits[0].id]} 类（模长仅 ${norm(corpus[hits[0].id]).toFixed(2)}）` +
      `${cosTopWrong ? "—— 意外，检查数据" : "—— 正确，方向对就排前"}`,
  );

  // Contrast: same corpus, ranked by RAW dot product (magnitude-sensitive). The giant
  // off-topic A doc#1 hijacks the top slot because A overlaps B and A is 9× longer.
  const dotHits = corpus
    .map((v, id) => ({ id, score: dot(query, v) }))
    .sort((a, b) => b.score - a.score || a.id - b.id)
    .slice(0, 3);
  console.log("对照：改用原始点积排序（对模长敏感）：");
  for (const h of dotHits) {
    console.log(
      `  doc#${h.id}  主题=${docTopic[h.id]}  ‖doc‖=${norm(corpus[h.id]).toFixed(2)}  ` +
        `点积=${h.score.toFixed(4)}`,
    );
  }
  const dotTopWrong = docTopic[dotHits[0].id] !== "B";
  console.log(
    `  点积第一名是 ${docTopic[dotHits[0].id]} 类（模长 ${norm(corpus[dotHits[0].id]).toFixed(2)}）` +
      `${dotTopWrong ? " —— 被超长的跑题文档劫持了！" : ""}`,
  );
  console.log(
    `结论：余弦把短而精准的 B 文档排第一，点积却让 9× 长的跑题 A 文档冒头 —— ` +
      `检索默认用余弦，正是为了不被文档长度带偏`,
  );
}

function main(): void {
  console.log("=== Stage 03 · 线性代数：数据的语言 ===");
  // Single seeded stream for the whole stage -> bit-for-bit reproducible across runs.
  const rng = mulberry32(3);
  demoPca(rng);
  demoSvd(rng);
  demoRetrieval(rng);
}

main();
