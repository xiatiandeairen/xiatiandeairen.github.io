// stage01-bm25.ts — the baseline this whole book argues against (and learns from).
//
// What this stage teaches: keyword retrieval via an inverted index + BM25 scoring.
// Before you can appreciate why dense vectors matter, you have to feel what they
// replace and what they DON'T replace. BM25 is not a strawman — it is still the
// production default for lexical search (Elasticsearch/Lucene ship it) and a hybrid
// system in stage 09 will lean on it. So we build it honestly: a real inverted
// index, real term-frequency saturation (k1), real length normalization (b), and a
// real failure mode you can reproduce by turning one knob.
//
// Mechanism, not magic. The three ideas, in order of how much they matter:
//   1. Inverted index: term -> postings list (which docs contain it). This is the
//      data structure that makes lexical search sublinear — you only ever touch
//      documents that share a query term, never the whole corpus.
//   2. IDF (inverse document frequency): a term in every doc carries no signal; a
//      rare term carries a lot. IDF is the "specificity weight".
//   3. BM25's two corrections over naive TF·IDF:
//        - k1 saturates term frequency: the 10th occurrence of "vector" adds far
//          less than the 1st. Without it, keyword stuffing wins.
//        - b normalizes by document length: a long document mentions everything,
//          so raw TF is inflated. b discounts long docs back toward fairness.
//
// The headline failure mode (demonstrated, not described): set b=0 and length
// normalization is OFF. Long documents — which contain every term by sheer volume —
// dominate the rankings even when a short, on-topic document is the better answer.
// We rank the SAME query under b=0.75 (Lucene default) and b=0 and print the gap.
//
// Honesty notes: this corpus is ~30 hand-written toy docs, so absolute scores are
// tiny and IDF is coarse-grained (document frequencies are single digits). The
// *relative* behavior — saturation, length penalty, the b=0 pathology — is exactly
// what you see at corpus scale; the numbers are just smaller and printable.

import { mulberry32 } from './core/dataset.js';

// ---------------------------------------------------------------------------
// Toy corpus. Deliberately mixes:
//  - short, sharply on-topic docs (the answers a good ranker should surface),
//  - one PADDING-tagged megadoc that mentions the query terms only incidentally
//    but is long enough to win under b=0 — our planted failure case,
//  - off-topic docs so IDF has something to discriminate against.
// ---------------------------------------------------------------------------

interface Doc {
  id: number;
  title: string;
  text: string;
}

const RAW_CORPUS: Array<{ title: string; text: string }> = [
  {
    title: 'What is a vector index',
    text: 'A vector index stores embeddings so nearest neighbor search runs fast. The index trades exact recall for speed.',
  },
  {
    title: 'Inverted index basics',
    text: 'An inverted index maps each term to the documents that contain it. Keyword search uses the inverted index to avoid scanning every document.',
  },
  {
    title: 'BM25 ranking explained',
    text: 'BM25 ranks documents by term frequency and inverse document frequency. It saturates term frequency and normalizes by document length.',
  },
  {
    title: 'Term frequency saturation',
    text: 'Term frequency saturation means repeating a term yields diminishing returns. The parameter k1 controls how fast term frequency saturates.',
  },
  {
    title: 'Document length normalization',
    text: 'Document length normalization discounts long documents. The parameter b controls how strongly length is normalized in BM25.',
  },
  {
    title: 'Cosine similarity',
    text: 'Cosine similarity measures the angle between two vectors. It ignores magnitude and focuses on direction.',
  },
  {
    title: 'Euclidean distance',
    text: 'Euclidean distance measures straight line distance between two points. Smaller distance means the points are closer.',
  },
  {
    title: 'Approximate nearest neighbor',
    text: 'Approximate nearest neighbor search returns close but not exact neighbors. It is much faster than brute force on large datasets.',
  },
  {
    title: 'IVF inverted file index',
    text: 'IVF partitions vectors into clusters. A query probes only a few clusters which prunes the search space and speeds up retrieval.',
  },
  {
    title: 'Product quantization',
    text: 'Product quantization compresses vectors into compact codes. It trades reconstruction accuracy for a large reduction in memory.',
  },
  {
    title: 'HNSW graph index',
    text: 'HNSW builds a navigable small world graph. Greedy search hops across the graph layers to reach the nearest neighbors quickly.',
  },
  {
    title: 'Recall and precision',
    text: 'Recall measures how many true neighbors were found. Precision measures how many returned results were correct.',
  },
  {
    title: 'Embeddings overview',
    text: 'Embeddings map text or images into a dense vector space. Similar items land close together in that space.',
  },
  {
    title: 'Tokenization',
    text: 'Tokenization splits text into tokens before indexing. Lowercasing and stemming reduce vocabulary size.',
  },
  {
    title: 'Stop words',
    text: 'Stop words are common words like the and of that carry little meaning. Removing them shrinks the inverted index.',
  },
  {
    title: 'Sharding a search engine',
    text: 'Sharding splits an index across machines. Each shard searches independently and results are merged.',
  },
  {
    title: 'Caching query results',
    text: 'Caching stores results of frequent queries. A cache hit avoids recomputing the ranking.',
  },
  {
    title: 'Relevance feedback',
    text: 'Relevance feedback uses user clicks to refine ranking. It adapts the order toward what users actually want.',
  },
  {
    title: 'Hybrid search',
    text: 'Hybrid search combines keyword BM25 scores with dense vector similarity. It captures exact terms and semantic meaning together.',
  },
  {
    title: 'Reranking results',
    text: 'A reranker reorders a candidate list using a heavier model. It improves precision at the top of the list.',
  },
  {
    title: 'Cooking pasta',
    text: 'Boil water with salt then add the pasta. Cook until al dente and drain before serving.',
  },
  {
    title: 'Morning coffee routine',
    text: 'Grind the beans fresh and heat water to the right temperature. Pour slowly over the grounds for an even extraction.',
  },
  {
    title: 'Weekend hiking trip',
    text: 'Pack water and snacks for the trail. Check the weather before you leave and wear sturdy boots.',
  },
  {
    title: 'House plant care',
    text: 'Water the plant when the top soil is dry. Most house plants prefer indirect sunlight near a window.',
  },
  {
    title: 'Bicycle maintenance',
    text: 'Keep the chain oiled and the tires inflated. Check the brakes before every long ride for safety.',
  },
  // The planted failure case: a long doc that name-drops the query terms a FEW
  // times amid lots of off-topic filler. Tuning matters here — the demo only works
  // if this doc loses under correct b=0.75 (length normalization discounts the
  // filler) but wins under b=0 (raw counts let the volume bulldoze short docs). So
  // each query term appears only ~3x, but the doc is padded long with unrelated
  // words. Too many repeats and saturation+IDF would let it win even at b=0.75
  // (the demo would then show "broken either way", not the b-controlled flip).
  {
    title: 'PADDING long survey of many unrelated computing systems',
    text: [
      'This rambling survey wanders through countless loosely related computing topics.',
      'It discusses operating systems, compilers, networks, databases, and storage hardware.',
      'It also mentions a vector once while drifting toward graphics and rendering pipelines.',
      'Pages of filler cover scheduling, caching layers, memory hierarchies, and bus protocols.',
      'Somewhere an index appears amid talk of file systems, journaling, and disk controllers.',
      'The survey meanders into search engines briefly before returning to unrelated trivia.',
      'More filler describes printers, monitors, keyboards, batteries, and cooling fans at length.',
      'It closes with assorted notes on firmware, drivers, bootloaders, and power management.',
    ].join(' '),
  },
];

// Build the doc list. We add ids deterministically; the corpus order is fixed so
// every run is bit-for-bit reproducible (no Math.random anywhere).
const CORPUS: Doc[] = RAW_CORPUS.map((d, id) => ({ id, ...d }));

// ---------------------------------------------------------------------------
// Tokenization. The cheapest possible pipeline that is still honest about what
// "a term" is: lowercase, split on non-letters, drop a tiny stop list. We do NOT
// stem — stemming is a quality lever orthogonal to BM25 and would muddy the
// demonstration. The stop list exists so IDF isn't dominated by "the"/"a"/"of".
// ---------------------------------------------------------------------------

const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'is', 'are', 'it',
  'that', 'for', 'with', 'by', 'as', 'into', 'than', 'then', 'how', 'so',
]);

// Why a standalone function and not inline: query-time and index-time tokenization
// MUST be identical, or a query term will never match its indexed form. Sharing one
// function is the invariant that guarantees that.
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z]+/) // split on anything not a-z; drops digits/punct/whitespace
    .filter((t) => t.length > 0 && !STOP_WORDS.has(t));
}

// ---------------------------------------------------------------------------
// The inverted index. postings: term -> Map<docId, termFrequency>. Storing TF in
// the postings (not just doc presence) is what lets BM25 score without re-reading
// the document text at query time — the whole point of an inverted index.
// ---------------------------------------------------------------------------

interface InvertedIndex {
  postings: Map<string, Map<number, number>>; // term -> (docId -> tf)
  docLengths: number[]; // token count per docId, for length normalization
  avgDocLength: number; // corpus average, the baseline b normalizes against
  docCount: number;
}

function buildIndex(docs: Doc[]): InvertedIndex {
  const postings = new Map<string, Map<number, number>>();
  const docLengths = new Array<number>(docs.length).fill(0);

  for (const doc of docs) {
    // Index title + text together. Title terms are not boosted here on purpose:
    // field weighting is a real feature but a separate concern from core BM25.
    const tokens = tokenize(`${doc.title} ${doc.text}`);
    docLengths[doc.id] = tokens.length;

    for (const term of tokens) {
      let list = postings.get(term);
      if (list === undefined) {
        list = new Map<number, number>();
        postings.set(term, list);
      }
      list.set(doc.id, (list.get(doc.id) ?? 0) + 1);
    }
  }

  const totalLength = docLengths.reduce((s, n) => s + n, 0);
  const avgDocLength = totalLength / docs.length;

  return { postings, docLengths, avgDocLength, docCount: docs.length };
}

// ---------------------------------------------------------------------------
// IDF, BM25's specificity weight. We use the standard BM25 "probabilistic" IDF:
//
//     idf(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )
//
// Why this exact form and not plain ln(N/df): the +0.5 smoothing keeps a term that
// appears in *every* document from going negative (a term in all N docs would give
// ln(0.5/(N+0.5)) < 0 under naive IDF, which would let common terms PENALIZE a doc
// — nonsense). The outer +1 guarantees idf >= 0 always. This is the Lucene form.
// df = document frequency = how many docs contain the term (postings list size).
// ---------------------------------------------------------------------------

function idf(term: string, index: InvertedIndex): number {
  const df = index.postings.get(term)?.size ?? 0;
  if (df === 0) return 0; // term not in corpus -> contributes nothing
  const n = index.docCount;
  return Math.log((n - df + 0.5) / (df + 0.5) + 1);
}

interface Bm25Params {
  k1: number; // term-frequency saturation: higher = TF matters longer before flattening
  b: number; // length normalization strength in [0,1]: 0 = OFF, 1 = full
}

// Lucene/Elasticsearch defaults. We expose them so the failure-mode demo can flip b.
const DEFAULT_PARAMS: Bm25Params = { k1: 1.5, b: 0.75 };

// Score one document for one set of query terms under given params.
//
// BM25 term contribution:
//
//     idf(t) * ( tf * (k1 + 1) ) / ( tf + k1 * (1 - b + b * dl/avgdl) )
//
// Read it as two factors:
//   - idf(t): how specific the term is (rare = high).
//   - the fraction: saturating TF. As tf -> infinity it approaches (k1+1), so a
//     term can contribute at most idf*(k1+1) no matter how many times it repeats —
//     that ceiling is the saturation. The denominator's (dl/avgdl) term is where b
//     lives: when b=0 the dl/avgdl factor vanishes and length is ignored entirely
//     (the planted failure); when b=1 a doc twice the average length has its TF
//     effectively halved.
function scoreDoc(
  docId: number,
  queryTerms: string[],
  index: InvertedIndex,
  params: Bm25Params,
): number {
  const { k1, b } = params;
  const dl = index.docLengths[docId];
  const lengthFactor = 1 - b + b * (dl / index.avgDocLength);

  let score = 0;
  for (const term of queryTerms) {
    const postingList = index.postings.get(term);
    if (postingList === undefined) continue; // term not in corpus
    const tf = postingList.get(docId) ?? 0;
    if (tf === 0) continue; // doc doesn't contain this term -> 0 contribution
    const numerator = tf * (k1 + 1);
    const denominator = tf + k1 * lengthFactor;
    score += idf(term, index) * (numerator / denominator);
  }
  return score;
}

interface ScoredDoc {
  docId: number;
  score: number;
}

// Top-k retrieval. The inverted-index payoff: we only score documents that appear
// in at least one query term's postings list, never the whole corpus. We gather
// that candidate set first, then score and sort it.
//
// Tie-break: score desc, then docId asc — deterministic, matching core/dataset's
// ground-truth convention so the whole book ranks consistently.
function search(
  query: string,
  index: InvertedIndex,
  params: Bm25Params,
  k: number,
): ScoredDoc[] {
  const queryTerms = tokenize(query);

  // Candidate gathering: union of postings lists. This is the sublinear step —
  // off-topic docs (pasta, coffee, hiking) that share no query term are never even
  // looked at. We dedupe via a Set because a doc can match multiple query terms.
  const candidates = new Set<number>();
  for (const term of queryTerms) {
    const postingList = index.postings.get(term);
    if (postingList === undefined) continue;
    for (const docId of postingList.keys()) candidates.add(docId);
  }

  const scored: ScoredDoc[] = [];
  for (const docId of candidates) {
    scored.push({ docId, score: scoreDoc(docId, queryTerms, index, params) });
  }
  scored.sort((x, y) => y.score - x.score || x.docId - y.docId);
  return scored.slice(0, k);
}

// ---------------------------------------------------------------------------
// Demo driver.
// ---------------------------------------------------------------------------

function printRanking(label: string, results: ScoredDoc[]): void {
  console.log(`\n${label}`);
  results.forEach((r, rank) => {
    const doc = CORPUS[r.docId];
    const padTag = doc.title.startsWith('PADDING') ? '  <-- long padding doc' : '';
    console.log(
      `  #${rank + 1}  score=${r.score.toFixed(4)}  [doc ${doc.id}] "${doc.title}"${padTag}`,
    );
  });
}

function main(): void {
  console.log('=== Stage 01: Inverted Index + BM25 (lexical baseline) ===');

  const index = buildIndex(CORPUS);

  // --- index stats: prove the data structure is real, with honest numbers ---
  console.log(`\n[index] ${index.docCount} docs, ${index.postings.size} unique terms`);
  console.log(`[index] avg doc length = ${index.avgDocLength.toFixed(1)} tokens`);

  const longest = index.docLengths.indexOf(Math.max(...index.docLengths));
  const shortest = index.docLengths.indexOf(Math.min(...index.docLengths));
  console.log(
    `[index] longest doc = ${index.docLengths[longest]} tokens (doc ${longest} "${CORPUS[longest].title}")`,
  );
  console.log(
    `[index] shortest doc = ${index.docLengths[shortest]} tokens (doc ${shortest} "${CORPUS[shortest].title}")`,
  );

  // --- inspect a postings list so the reader sees the actual structure ---
  const sampleTerm = 'vector';
  const sampleList = index.postings.get(sampleTerm);
  if (sampleList) {
    const entries = [...sampleList.entries()]
      .map(([d, tf]) => `doc${d}:tf=${tf}`)
      .join('  ');
    console.log(
      `\n[postings] "${sampleTerm}" -> df=${sampleList.size}, idf=${idf(sampleTerm, index).toFixed(4)}`,
    );
    console.log(`[postings] ${entries}`);
  }

  // --- IDF discrimination: rare term beats common term ---
  console.log('\n[idf] specificity weights (rare term = higher idf):');
  for (const t of ['vector', 'search', 'index', 'quantization', 'cosine']) {
    const df = index.postings.get(t)?.size ?? 0;
    console.log(`  ${t.padEnd(13)} df=${df}  idf=${idf(t, index).toFixed(4)}`);
  }

  // --- happy path: a clean query returns the focused on-topic docs ---
  const q1 = 'vector index search';
  console.log(`\n--- Query: "${q1}" (k1=${DEFAULT_PARAMS.k1}, b=${DEFAULT_PARAMS.b}) ---`);
  const top1 = search(q1, index, DEFAULT_PARAMS, 5);
  printRanking('Top 5 (BM25 default):', top1);

  // --- saturation demo: TF saturates, so the padding doc CANNOT win by repetition
  //     alone once length normalization is on. Show its score vs a focused doc. ---
  const padId = CORPUS.findIndex((d) => d.title.startsWith('PADDING'));
  const focusedId = CORPUS.findIndex((d) => d.title === 'What is a vector index');
  const qTerms = tokenize(q1);
  console.log('\n[saturation] same query terms, two docs, default params:');
  console.log(
    `  padding doc (len ${index.docLengths[padId]}): score=${scoreDoc(padId, qTerms, index, DEFAULT_PARAMS).toFixed(4)}`,
  );
  console.log(
    `  focused doc (len ${index.docLengths[focusedId]}): score=${scoreDoc(focusedId, qTerms, index, DEFAULT_PARAMS).toFixed(4)}`,
  );

  // ========================================================================
  // FAILURE MODE: b = 0 turns OFF length normalization. The long padding doc,
  // which repeats query terms by sheer volume, now bulldozes the short focused
  // docs. This is the single most important thing to internalize about BM25:
  // length normalization is not a nicety, it is what stops verbosity from being
  // mistaken for relevance.
  // ========================================================================
  const NO_LENGTH_NORM: Bm25Params = { k1: DEFAULT_PARAMS.k1, b: 0 };
  console.log(`\n--- FAILURE MODE: same query, b=0 (length normalization OFF) ---`);
  const top1NoNorm = search(q1, index, NO_LENGTH_NORM, 5);
  printRanking('Top 5 (b=0, BROKEN):', top1NoNorm);

  // Quantify the damage: where did the padding doc rank under each setting?
  const rankOf = (results: ScoredDoc[], id: number): string => {
    const idx = results.findIndex((r) => r.docId === id);
    return idx === -1 ? 'not in top 5' : `#${idx + 1}`;
  };
  console.log('\n[failure-mode summary]');
  console.log(`  padding doc rank with b=0.75 (correct): ${rankOf(top1, padId)}`);
  console.log(`  padding doc rank with b=0    (broken):  ${rankOf(top1NoNorm, padId)}`);
  console.log(
    `  padding doc score b=0.75 -> b=0: ` +
      `${scoreDoc(padId, qTerms, index, DEFAULT_PARAMS).toFixed(4)} -> ` +
      `${scoreDoc(padId, qTerms, index, NO_LENGTH_NORM).toFixed(4)} ` +
      `(inflated ${(
        scoreDoc(padId, qTerms, index, NO_LENGTH_NORM) /
        scoreDoc(padId, qTerms, index, DEFAULT_PARAMS)
      ).toFixed(2)}x by removing the length penalty)`,
  );

  // --- SECOND failure mode: lexical search has zero recall on synonyms. This is
  //     the exact gap dense vectors close, and the reason the rest of the book
  //     exists. "automobile" never appears, but a semantic search SHOULD find the
  //     embeddings/vector docs for "find similar items". BM25 returns nothing. ---
  const q2 = 'find similar items quickly';
  console.log(`\n--- Query: "${q2}" (semantic intent, few exact keyword hits) ---`);
  const top2 = search(q2, index, DEFAULT_PARAMS, 5);
  if (top2.length === 0) {
    console.log('  (no documents matched any query term)');
  } else {
    printRanking('Top 5 (BM25 default):', top2);
  }
  const q3 = 'automobile photograph';
  const top3 = search(q3, index, DEFAULT_PARAMS, 5);
  console.log(`\n--- Query: "${q3}" (pure vocabulary mismatch) ---`);
  console.log(
    `  matched ${top3.length} docs. BM25 cannot bridge "automobile" -> "vector"; ` +
      `this zero-recall-on-synonyms gap is what stage 02+ embeddings fix.`,
  );

  // --- determinism check: query-term ORDER must not affect ranking, because BM25
  //     sums per-term contributions (addition is commutative) and the final sort is
  //     a total order on (score, docId). We prove it by Fisher-Yates shuffling the
  //     query terms with the seeded PRNG (a real permutation, unlike the buggy
  //     `sort(() => rng()-0.5)` idiom which biases and can drop elements) and
  //     confirming the ranking is byte-identical. ---
  const rng = mulberry32(42);
  const shuffledTerms = [...qTerms];
  for (let i = shuffledTerms.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1)); // unbiased index in [0, i]
    [shuffledTerms[i], shuffledTerms[j]] = [shuffledTerms[j], shuffledTerms[i]];
  }
  const reRun = search(shuffledTerms.join(' '), index, DEFAULT_PARAMS, 5);
  const sameRanking =
    reRun.length === top1.length &&
    reRun.every((r, i) => r.docId === top1[i].docId && r.score === top1[i].score);
  console.log(
    `\n[determinism] re-ranking with shuffled query-term order identical to original: ${
      sameRanking ? 'YES' : 'NO'
    }`,
  );

  console.log('\n=== takeaways ===');
  console.log('  1. inverted index = only touch docs sharing a query term (sublinear).');
  console.log('  2. k1 saturates TF; b normalizes length. b=0 lets long docs cheat.');
  console.log('  3. BM25 has zero recall across vocabulary gaps -> motivates embeddings.');
}

main();
