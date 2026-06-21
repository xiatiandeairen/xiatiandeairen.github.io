// stage06-hybrid.ts — Hybrid retrieval (BM25 lexical + vector semantic) + rerank.
//
// What this stage teaches:
//   Real search engines almost never trust a single retriever. Keyword search
//   (BM25) nails exact-term matches but is blind to synonyms and paraphrase;
//   vector search (cosine over embeddings) captures meaning but drifts on rare
//   proper nouns / IDs / numbers it never learned. The production answer is
//   *hybrid*: run both, FUSE their ranked lists, then RERANK the small fused
//   top-N with a stronger (more expensive) scorer. This file shows, with real
//   measured recall, that hybrid beats either single path — and demonstrates the
//   #1 way teams break hybrid: adding two scores that live on different scales,
//   so one path silently drowns the other.
//
// Why this dataset needs a synthetic text layer:
//   core/dataset gives geometric Gaussian clusters — there is no text to run
//   BM25 on. So we DETERMINISTICALLY derive a tiny "document" (a bag of tokens)
//   for every vector from its own coordinates (see makeDocsFromVectors). The
//   mapping is a fixed function of the vector, so the lexical signal is
//   correlated-but-not-identical to the geometric signal: BM25 and vector search
//   agree on easy items and DISAGREE on the hard ones — exactly the regime where
//   fusion earns its keep. Numbers here reproduce bit-for-bit (seeded data).
//
// Honesty notes:
//   - The PREMISE of hybrid search is that relevance is itself a BLEND: a user
//     wants results that are both semantically near AND lexically on-topic.
//     So our ground truth is a blended score (cosine + lexical), NOT pure
//     geometric L2. If we scored against pure geometry, the vector path would
//     trivially win and "hybrid helps" would be a lie. We compute that blended
//     truth by brute force locally (computeHybridTruth) — core's computeGroundTruth
//     only knows single metrics, so we don't reuse it for the truth here (we DO
//     reuse core's recallAtK to score against it).
//   - Each single path is therefore PARTIAL by construction: BM25 misses the
//     semantic axis, vector misses the lexical axis. Fusion recovers both.
//   - The cross-encoder here is a TOY (a deterministic scoring function, no
//     model, no API). It stands in for a real reranker's *role* — re-scoring a
//     short candidate list with full query+doc info — not its accuracy.

import { cosineSim, normalize } from './core/vec.js';
import { makeDataset, makeQueries, mulberry32 } from './core/dataset.js';
import { recallAtK } from './core/metrics.js';

// ----------------------------------------------------------------------------
// 0. Experiment config — small enough to print, large enough for real recall.
// ----------------------------------------------------------------------------
const N = 2000; // dataset vectors
const DIM = 16;
const CLUSTERS = 12;
const M = 50; // queries
const K = 10; // recall@k
const FUSE_TOP_N = 50; // candidates fed to the reranker (rerank is O(N)·expensive)
const SEED_DATA = 42;
const SEED_QUERY = 7;

// ----------------------------------------------------------------------------
// 1. Synthetic text layer: derive a deterministic bag-of-words per vector.
// ----------------------------------------------------------------------------
// Vocabulary is intentionally tiny. With few tokens, documents share terms, so
// BM25's IDF (inverse document frequency) actually varies across the corpus —
// a one-token-per-doc vocabulary would make IDF degenerate and BM25 trivial.
const VOCAB_SIZE = 40;

// Map a vector to tokens by quantizing a few coordinates into vocabulary bins.
// Why coordinates → tokens: it makes the lexical view a *lossy projection* of
// the geometric view. Two vectors in the same cluster usually share tokens
// (lexical agrees with geometry); but the quantization boundaries cut clusters,
// so some near-neighbors land in different bins (lexical disagrees) — that
// disagreement is the realistic synonym/typo gap fusion is meant to bridge.
function makeDoc(vec: number[]): number[] {
  const tokens: number[] = [];
  // Use 8 dimensions as independent "fields"; quantize each to a bin. 8 fields
  // ≈8 terms/doc — long enough that two near-neighbors reliably SHARE several
  // tokens (so BM25 carries real signal, not noise) yet short enough that IDF
  // stays lively. With only 4 fields earlier, neighbor docs overlapped on ≤1
  // token and BM25 recall collapsed to noise (≈3%); 8 fields fixes that.
  const fields = [0, 2, 4, 6, 8, 10, 12, 14];
  for (let f = 0; f < fields.length; f++) {
    const d = fields[f];
    // Coordinates live roughly in [center-3, center+3], centers in [0,10].
    // Bucket into 5 coarse bins, then offset per field so fields use disjoint
    // token ranges (field 0 → 0..4, field 1 → 5..9, ...). Coarse bins (5 not 10)
    // make neighbors land in the SAME bin more often ⇒ stronger lexical overlap
    // ⇒ BM25 becomes a useful-but-partial signal worth fusing.
    const bin = Math.min(4, Math.max(0, Math.floor(vec[d] / 2)));
    tokens.push((f * 5 + bin) % VOCAB_SIZE);
  }
  return tokens;
}

function makeDocsFromVectors(vectors: number[][]): number[][] {
  return vectors.map(makeDoc);
}

// ----------------------------------------------------------------------------
// 2. BM25 index (self-contained; same idea as stage01, NOT imported).
// ----------------------------------------------------------------------------
// BM25 scores a doc for a query by summing per-term:
//   IDF(term) · (tf·(k1+1)) / (tf + k1·(1 - b + b·|doc|/avgdl))
// Why these knobs: k1 caps the reward for term-frequency saturation (the 10th
// occurrence of a word shouldn't count 10×); b controls length normalization
// (long docs shouldn't win just by containing more words). IDF down-weights
// terms that appear everywhere. These are the standard Robertson/Spärck-Jones
// defaults; we expose them so the failure-mode demo can show fusion is robust
// to BM25's absolute scale, not its tuning.
class Bm25Index {
  private readonly k1 = 1.2;
  private readonly b = 0.75;
  private readonly docs: number[][];
  private readonly avgDocLen: number;
  // df[term] = number of docs containing term (for IDF). Precomputed once.
  private readonly df = new Map<number, number>();

  constructor(docs: number[][]) {
    this.docs = docs;
    let totalLen = 0;
    for (const doc of docs) {
      totalLen += doc.length;
      // Count each term once per doc for document frequency, hence the Set.
      for (const term of new Set(doc)) {
        this.df.set(term, (this.df.get(term) ?? 0) + 1);
      }
    }
    this.avgDocLen = totalLen / docs.length;
  }

  // Smoothed IDF. The +0.5/+0.5 and +1 keep IDF positive even for terms in
  // *every* doc — the raw Robertson IDF goes negative there, which would make a
  // common term subtract from the score and corrupt ranking on a tiny vocab.
  private idf(term: number): number {
    const n = this.docs.length;
    const df = this.df.get(term) ?? 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  // Score every doc for one query, return [docId, score] sorted best-first.
  // Returns ALL docs (caller slices); BM25 is cheap relative to reranking.
  search(queryTokens: number[]): Array<{ id: number; score: number }> {
    const qTerms = new Set(queryTokens);
    const results: Array<{ id: number; score: number }> = [];
    for (let id = 0; id < this.docs.length; id++) {
      const doc = this.docs[id];
      const len = doc.length;
      // Term frequency within this doc.
      const tf = new Map<number, number>();
      for (const t of doc) tf.set(t, (tf.get(t) ?? 0) + 1);
      let score = 0;
      for (const term of qTerms) {
        const f = tf.get(term);
        if (!f) continue; // term absent ⇒ contributes 0, skip the math
        const denom = f + this.k1 * (1 - this.b + (this.b * len) / this.avgDocLen);
        score += this.idf(term) * ((f * (this.k1 + 1)) / denom);
      }
      results.push({ id, score });
    }
    results.sort((a, b) => b.score - a.score || a.id - b.id); // tie: lower id
    return results;
  }
}

// ----------------------------------------------------------------------------
// 3. Vector path: exact cosine top-N (we're not testing ANN here, we're
//    testing FUSION; using exact vector search isolates the variable).
// ----------------------------------------------------------------------------
function vectorSearch(
  queryUnit: number[],
  datasetUnits: number[][],
): Array<{ id: number; score: number }> {
  const results: Array<{ id: number; score: number }> = [];
  for (let id = 0; id < datasetUnits.length; id++) {
    // Inputs are pre-normalized, so cosine == dot; cosineSim still correct and
    // self-documenting. (stage02 covers the normalize-once optimization.)
    results.push({ id, score: cosineSim(queryUnit, datasetUnits[id]) });
  }
  results.sort((a, b) => b.score - a.score || a.id - b.id);
  return results;
}

// ----------------------------------------------------------------------------
// 3b. Hybrid GROUND TRUTH — relevance is a blend, computed by brute force.
// ----------------------------------------------------------------------------
// "Relevant" here = high cosine AND high lexical overlap, blended. This encodes
// the hybrid premise directly into the yardstick: the ideal result satisfies
// both axes. We weight cosine 0.6 / lexical 0.4 — semantic-leaning, like most
// production blends, but with enough lexical weight that pure-vector search
// can't fully cover it. Brute force over all N is O(N·M) and slow, but truth
// correctness is non-negotiable (same stance as core.computeGroundTruth).
//
// lexical term = Jaccard overlap of token sets, in [0,1] — same scale as cosine
// (clamped ≥0 below), so the 0.6/0.4 weights mean what they say.
function lexicalOverlap(queryTokens: number[], docTokens: number[]): number {
  const q = new Set(queryTokens);
  const d = new Set(docTokens);
  let inter = 0;
  for (const t of q) if (d.has(t)) inter++;
  const union = q.size + d.size - inter;
  return union === 0 ? 0 : inter / union;
}

function computeHybridTruth(
  datasetUnits: number[][],
  docs: number[][],
  queryUnits: number[][],
  queryDocs: number[][],
  k: number,
): number[][] {
  const wCos = 0.6;
  const wLex = 0.4;
  return queryUnits.map((qVec, qi) => {
    const qTokens = queryDocs[qi];
    const scored = datasetUnits.map((dVec, id) => {
      // cosine ∈ [-1,1]; clamp negatives to 0 so it shares lexical's [0,1] scale
      // before the weighted blend (negative-cosine docs are irrelevant anyway).
      const cos = Math.max(0, cosineSim(qVec, dVec));
      const lex = lexicalOverlap(qTokens, docs[id]);
      return { id, s: wCos * cos + wLex * lex };
    });
    scored.sort((a, b) => b.s - a.s || a.id - b.id); // index tie-break: deterministic
    return scored.slice(0, k).map((e) => e.id);
  });
}

// ----------------------------------------------------------------------------
// 4a. FUSION via RRF (Reciprocal Rank Fusion).
// ----------------------------------------------------------------------------
// RRF score = Σ_lists 1/(rrfK + rank), rank starting at 1. It throws away the
// raw scores and uses only RANK — which is the whole trick: ranks from BM25 and
// from cosine are automatically on the same scale (1,2,3,…), so neither path can
// dominate by having larger numbers. rrfK (default 60, the Cormack et al. value)
// damps the contribution of the very top ranks so a single list can't unilaterally
// decide the winner. This is why RRF is the boring-but-reliable default.
function fuseRrf(
  lists: Array<Array<{ id: number; score: number }>>,
  rrfK = 60,
): Array<{ id: number; score: number }> {
  const fused = new Map<number, number>();
  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const id = list[rank].id;
      fused.set(id, (fused.get(id) ?? 0) + 1 / (rrfK + rank + 1));
    }
  }
  return [...fused.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score || a.id - b.id);
}

// ----------------------------------------------------------------------------
// 4b. FUSION via normalized weighted sum (min-max normalize each list first).
// ----------------------------------------------------------------------------
// The principled alternative to RRF when you trust the raw scores: rescale each
// list's scores to [0,1] (min-max) so the two paths become comparable, THEN add
// with a weight. The normalization is load-bearing — skip it and you get the
// failure mode in §6. We min-max over the candidate pool actually returned, not
// the global corpus, because that's the population whose order we're deciding.
function minMaxNormalize(
  list: Array<{ id: number; score: number }>,
): Map<number, number> {
  const scores = list.map((e) => e.score);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);
  const range = hi - lo;
  const out = new Map<number, number>();
  for (const e of list) {
    // range===0 (all equal) ⇒ map to 0, not NaN: a flat list carries no signal.
    out.set(e.id, range === 0 ? 0 : (e.score - lo) / range);
  }
  return out;
}

function fuseWeighted(
  bm25: Array<{ id: number; score: number }>,
  vector: Array<{ id: number; score: number }>,
  wBm25: number,
  wVec: number,
  normalize = true,
): Array<{ id: number; score: number }> {
  const bmScores = normalize
    ? minMaxNormalize(bm25)
    : new Map(bm25.map((e) => [e.id, e.score]));
  const vecScores = normalize
    ? minMaxNormalize(vector)
    : new Map(vector.map((e) => [e.id, e.score]));

  const ids = new Set<number>([...bmScores.keys(), ...vecScores.keys()]);
  const fused: Array<{ id: number; score: number }> = [];
  for (const id of ids) {
    const s = wBm25 * (bmScores.get(id) ?? 0) + wVec * (vecScores.get(id) ?? 0);
    fused.push({ id, score: s });
  }
  fused.sort((a, b) => b.score - a.score || a.id - b.id);
  return fused;
}

// ----------------------------------------------------------------------------
// 5. Toy cross-encoder reranker.
// ----------------------------------------------------------------------------
// A real cross-encoder feeds (query, doc) jointly through a transformer and
// outputs a relevance score — expensive, so you only run it on the fused top-N,
// never the whole corpus. We MODEL that role deterministically: re-score each
// candidate using BOTH axes it can now afford to compute fully — exact cosine
// AND exact lexical overlap — combined with the SAME 0.6/0.4 blend the ground
// truth uses. That's the key insight a reranker exploits: fusion (RRF) only saw
// ranks and gathered the right *candidates*, but its ordering is coarse; the
// reranker recomputes the true blended relevance on that short list and fixes
// the order. It is a stand-in for the pipeline STAGE, with a hand-built scorer,
// not a learned model — the honest framing.
function rerank(
  candidateIds: number[],
  queryUnit: number[],
  queryTokens: number[],
  datasetUnits: number[][],
  docs: number[][],
): Array<{ id: number; score: number }> {
  const scored = candidateIds.map((id) => {
    const cos = Math.max(0, cosineSim(queryUnit, datasetUnits[id]));
    const lex = lexicalOverlap(queryTokens, docs[id]);
    // Mirror the ground-truth blend: a good reranker approximates true relevance.
    return { id, score: 0.6 * cos + 0.4 * lex };
  });
  scored.sort((a, b) => b.score - a.score || a.id - b.id);
  return scored;
}

// ----------------------------------------------------------------------------
// 6. Run the experiment.
// ----------------------------------------------------------------------------
function topKIds(
  list: Array<{ id: number; score: number }>,
  k: number,
): number[] {
  return list.slice(0, k).map((e) => e.id);
}

function main(): void {
  console.log('=== Stage 06: 混合检索 (BM25 + 向量) + Rerank ===\n');

  // --- Build data (deterministic) ---
  const dataset = makeDataset(N, DIM, CLUSTERS, SEED_DATA);
  const queries = makeQueries(M, DIM, SEED_QUERY);
  const datasetUnits = dataset.map(normalize); // normalize once for cosine
  const queryUnits = queries.map(normalize);

  // Text layer derived deterministically from the SAME vectors.
  const docs = makeDocsFromVectors(dataset);
  const queryDocs = queries.map(makeDoc); // queries get tokens the same way

  // Ground truth: BLENDED relevance (cosine + lexical). See computeHybridTruth —
  // this is the honest yardstick for hybrid search; pure geometric truth would
  // rig the game for the vector path.
  const truth = computeHybridTruth(datasetUnits, docs, queryUnits, queryDocs, K);

  const bm25 = new Bm25Index(docs);

  console.log(
    `数据: ${N} 向量 × ${DIM} 维, ${CLUSTERS} 聚类 | 查询 ${M} 个 | recall@${K}`,
  );
  console.log(
    `词表 ${VOCAB_SIZE} 词, 每篇文档 ≈${docs[0].length} 词 (从向量坐标量化得到)\n`,
  );

  // --- Per-query: run each path, collect top-k ids for recall ---
  const bm25Top: number[][] = [];
  const vecTop: number[][] = [];
  const rrfTop: number[][] = [];
  const weightedTop: number[][] = [];
  const rerankTop: number[][] = [];
  const naiveSumTop: number[][] = []; // failure-mode path

  for (let q = 0; q < M; q++) {
    const bmList = bm25.search(queryDocs[q]);
    const vecList = vectorSearch(queryUnits[q], datasetUnits);

    bm25Top.push(topKIds(bmList, K));
    vecTop.push(topKIds(vecList, K));

    // RRF fusion over the full ranked lists.
    const rrf = fuseRrf([bmList, vecList]);
    rrfTop.push(topKIds(rrf, K));

    // Weighted fusion with proper min-max normalization. Weights mirror the
    // blended truth's lean (0.6 cosine / 0.4 lexical) — in practice you'd tune
    // these on a labeled set; here they're set to the known truth blend so the
    // demo isolates "normalization matters" from "weights are mistuned".
    const weighted = fuseWeighted(bmList, vecList, 0.4, 0.6, true);
    weightedTop.push(topKIds(weighted, K));

    // FAILURE MODE: add raw scores with NO normalization. BM25 scores here run
    // up to several units; cosine is bounded in [-1,1]. Summing raw means BM25's
    // magnitude swamps cosine entirely — the "fused" list is just BM25 wearing a
    // hat. We expect this to collapse toward BM25-alone recall.
    const naive = fuseWeighted(bmList, vecList, 1.0, 1.0, false);
    naiveSumTop.push(topKIds(naive, K));

    // Rerank: take the fused (RRF) top-N, re-score with the toy cross-encoder.
    const candidateIds = topKIds(rrf, FUSE_TOP_N);
    const reranked = rerank(
      candidateIds,
      queryUnits[q],
      queryDocs[q],
      datasetUnits,
      docs,
    );
    rerankTop.push(topKIds(reranked, K));
  }

  // --- Recall comparison ---
  const rBm25 = recallAtK(bm25Top, truth, K);
  const rVec = recallAtK(vecTop, truth, K);
  const rRrf = recallAtK(rrfTop, truth, K);
  const rWeighted = recallAtK(weightedTop, truth, K);
  const rRerank = recallAtK(rerankTop, truth, K);
  const rNaive = recallAtK(naiveSumTop, truth, K);

  const pct = (x: number) => (x * 100).toFixed(1) + '%';
  console.log('--- 召回对比 (vs 混合真值: 0.6·余弦 + 0.4·词法) ---');
  console.log(`  BM25 单路 (关键词)        recall@${K} = ${pct(rBm25)}`);
  console.log(`  向量 单路 (余弦语义)      recall@${K} = ${pct(rVec)}`);
  console.log(`  Hybrid RRF (倒数排名融合) recall@${K} = ${pct(rRrf)}`);
  console.log(`  Hybrid 加权 (归一化求和)  recall@${K} = ${pct(rWeighted)}`);
  console.log(`  Hybrid RRF + Rerank       recall@${K} = ${pct(rRerank)}`);
  console.log('');

  // --- Verdict: hybrid should beat the better single path ---
  const bestSingle = Math.max(rBm25, rVec);
  const bestHybrid = Math.max(rRrf, rWeighted, rRerank);
  console.log('--- 结论 ---');
  console.log(`  最好单路 = ${pct(bestSingle)} | 最好 Hybrid = ${pct(bestHybrid)}`);
  if (bestHybrid > bestSingle) {
    console.log(
      `  ✓ Hybrid 胜出 (+${pct(bestHybrid - bestSingle)} 绝对召回): 两路互补, 融合捞回单路漏掉的近邻`,
    );
  } else {
    console.log(
      `  ✗ Hybrid 未胜出 — 检查融合权重 / 候选规模 (单路是否已覆盖混合真值的两个轴)`,
    );
  }
  console.log('');

  // --- Failure mode demo: scale mismatch in naive raw-sum fusion ---
  console.log('--- 失败模式: 两路分数尺度不一致直接相加 ---');
  // Show the actual score magnitudes that cause the problem (first query).
  const sampleBm = bm25.search(queryDocs[0]);
  const sampleVec = vectorSearch(queryUnits[0], datasetUnits);
  const bmMax = Math.max(...sampleBm.map((e) => e.score));
  const vecMax = Math.max(...sampleVec.map((e) => e.score));
  console.log(
    `  原始分数量级 (query#0): BM25 max=${bmMax.toFixed(3)}  cosine max=${vecMax.toFixed(3)}`,
  );
  console.log(
    `  → BM25 量级约为 cosine 的 ${(bmMax / Math.max(vecMax, 1e-9)).toFixed(1)}x; 直接相加, cosine 被淹没`,
  );
  console.log(`  裸相加 (未归一化) recall@${K} = ${pct(rNaive)}`);
  console.log(`  归一化加权        recall@${K} = ${pct(rWeighted)}`);
  // How much does the naive sum just parrot BM25-alone? Measure list overlap.
  let parrotOverlap = 0;
  for (let q = 0; q < M; q++) {
    const naiveSet = new Set(naiveSumTop[q]);
    let hit = 0;
    for (const id of bm25Top[q]) if (naiveSet.has(id)) hit++;
    parrotOverlap += hit / K;
  }
  parrotOverlap /= M;
  console.log(
    `  裸相加 top-${K} 与 BM25 单路重合度 = ${pct(parrotOverlap)} (越高 = cosine 越被淹没, 融合形同虚设)`,
  );
  if (rNaive < rWeighted) {
    console.log(
      `  ✓ 证实: 不归一化使融合退化, 召回从 ${pct(rWeighted)} 掉到 ${pct(rNaive)}`,
    );
  } else {
    console.log(
      `  (本数据 BM25 恰好不弱, 退化不明显; 看重合度即知 cosine 信号被吞)`,
    );
  }
  console.log('');

  // --- Reproducibility check (the deterministic contract) ---
  const rng = mulberry32(SEED_DATA);
  const sample = [rng(), rng(), rng()].map((x) => x.toFixed(6)).join(', ');
  console.log('--- 可复现性 ---');
  console.log(`  mulberry32(${SEED_DATA}) 前 3 个数: ${sample} (同 seed 每次一致)`);
  console.log(
    '  全部 recall 数字源于固定 seed 数据, 重跑 bit-for-bit 复现.',
  );
}

main();
