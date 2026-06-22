// stage01-onehot.ts — why one-hot is a dead end: orthogonality, the curse of
// dimensionality, and the case for dense representations.
//
// The book's chapter 01 thesis: one-hot encoding is the "obvious" first move and
// it fails for a STRUCTURAL reason, not a tuning reason. This stage proves that
// failure with numbers you can re-derive, not assertions:
//
//   (1) ORTHOGONALITY: any two distinct one-hot vectors have cosine similarity
//       EXACTLY 0. So "king" and "queen" are exactly as similar as "king" and
//       "fish": zero. There is no room in the geometry for "more similar". This
//       is the disease; everything below is a symptom.
//
//   (2) CURSE OF DIMENSIONALITY: as dimension d grows, the ratio
//       nearest-distance / farthest-distance among random points → 1. Distance
//       loses contrast — "nearest" stops meaning anything. One-hot lives in
//       d = vocab-size space (here ~40, in real life 10^5+), so its distances are
//       already in the contrast-collapsed regime. We measure the ratio at
//       d = 2/10/100/1000 and watch it climb toward 1.
//
//   (3) THE FIX, PREVIEWED: same toy documents, retrieved two ways — bag-of-words
//       over one-hot (sparse, orthogonal) vs a hand-built low-dim dense code that
//       puts same-topic words on shared axes. Dense clusters same-topic docs;
//       one-hot cannot, because step (1) forbids it.
//
// HONESTY: absolute numbers here are toy-scale and optimistic (40-word vocab,
// hand-placed dense axes). What transfers is the DIRECTION of the trends — ratio
// rising with d, dense beating one-hot on topical retrieval — not the magnitudes.
// The dense vectors are hand-authored, NOT learned; learning them is stage 03+.
//
// Determinism: every random draw comes from a seeded Rng (core/rng), never
// Math.random, so the printed ratios reproduce bit-for-bit.

import { Rng, gaussian } from "./core/rng.js";
import { generateCorpus, buildVocab } from "./core/text.js";
import { cosineSimilarity, euclidean, nearestNeighbors } from "./core/eval.js";
import { asciiLine, asciiBar } from "./core/plot.js";

// ---------------------------------------------------------------------------
// (1) One-hot orthogonality
// ---------------------------------------------------------------------------

// Build the V x V one-hot matrix: row i is the basis vector e_i (1 at position i,
// 0 elsewhere). This IS the identity matrix; we materialize it explicitly so the
// reader sees that "one-hot encoding of a vocab" is literally "use the standard
// basis", which is exactly why distinct codes are orthogonal.
function buildOneHotMatrix(vocabSize: number): number[][] {
  const rows: number[][] = [];
  for (let i = 0; i < vocabSize; i++) {
    const row = new Array(vocabSize).fill(0);
    row[i] = 1;
    rows.push(row);
  }
  return rows;
}

function demoOrthogonality(): void {
  // We only need the vocab here, not the corpus text, but generateCorpus is the
  // single seeded source of the toy vocabulary the whole book shares.
  const corpus = generateCorpus(new Rng(1));
  const vocab = buildVocab(corpus.tokens);
  const oneHot = buildOneHotMatrix(vocab.size);

  console.log("=".repeat(64));
  console.log(`(1) one-hot orthogonality  (vocab size V = ${vocab.size})`);
  console.log("=".repeat(64));

  // Pick word pairs that a human KNOWS are semantically related vs unrelated, to
  // make the point visceral: one-hot collapses both to the same number.
  const probes: Array<[string, string]> = [
    ["king", "queen"], // strongly related (the canonical analogy pair)
    ["dog", "cat"], // related (same animal cluster)
    ["king", "fish"], // unrelated
    ["king", "king"], // identical: the ONLY non-zero cosine one-hot can produce
  ];

  let maxOffDiagonal = 0; // invariant check: must stay exactly 0 for distinct words
  for (const [wa, wb] of probes) {
    const ia = vocab.stoi.get(wa);
    const ib = vocab.stoi.get(wb);
    if (ia === undefined || ib === undefined) continue; // word absent from toy vocab
    const cos = cosineSimilarity(oneHot[ia], oneHot[ib]);
    if (ia !== ib) maxOffDiagonal = Math.max(maxOffDiagonal, Math.abs(cos));
    const note = ia === ib ? "(same word → 1)" : "(distinct → exactly 0)";
    console.log(`  cos(${wa.padEnd(6)}, ${wb.padEnd(6)}) = ${cos.toFixed(4)}  ${note}`);
  }

  // The whole-matrix invariant, stated as a real measured number rather than a
  // claim: scan every off-diagonal pair, report the largest cosine seen. It is 0.
  let worst = 0;
  for (let i = 0; i < vocab.size; i++) {
    for (let j = i + 1; j < vocab.size; j++) {
      worst = Math.max(worst, Math.abs(cosineSimilarity(oneHot[i], oneHot[j])));
    }
  }
  console.log(
    `  measured max |cos| over all ${(vocab.size * (vocab.size - 1)) / 2} distinct pairs = ${worst.toFixed(4)}`,
  );
  console.log(
    "  → every distinct word is equidistant from every other. 'king' is no closer",
  );
  console.log("    to 'queen' than to 'fish'. The geometry has no notion of meaning.");
}

// ---------------------------------------------------------------------------
// (2) Curse of dimensionality: distance contrast collapses
// ---------------------------------------------------------------------------

// For a cloud of N points in d dimensions, return min and max pairwise Euclidean
// distance and their ratio. As d grows the ratio → 1: the closest pair becomes
// almost as far as the farthest pair, so "nearest neighbor" carries no signal.
//
// We use Gaussian-distributed points (not uniform-cube) because the classic
// Beyer et al. 1999 result on distance concentration is cleanest for iid coords,
// and Gaussian is the init distribution the rest of the book uses.
function distanceContrast(
  rng: Rng,
  dim: number,
  numPoints: number,
): { minDist: number; maxDist: number; ratio: number } {
  const points: number[][] = [];
  for (let p = 0; p < numPoints; p++) {
    const v = new Array(dim);
    for (let k = 0; k < dim; k++) v[k] = gaussian(rng); // standard normal coords
    points.push(v);
  }
  let minDist = Infinity;
  let maxDist = 0;
  for (let i = 0; i < numPoints; i++) {
    for (let j = i + 1; j < numPoints; j++) {
      const d = euclidean(points[i], points[j]);
      if (d < minDist) minDist = d;
      if (d > maxDist) maxDist = d;
    }
  }
  // maxDist is positive for N>=2 distinct Gaussian draws; guard anyway so a
  // degenerate all-identical cloud reports ratio 1 instead of NaN.
  const ratio = maxDist === 0 ? 1 : minDist / maxDist;
  return { minDist, maxDist, ratio };
}

function demoCurseOfDimensionality(): void {
  console.log("");
  console.log("=".repeat(64));
  console.log("(2) curse of dimensionality: nearest/farthest distance ratio → 1");
  console.log("=".repeat(64));

  const dims = [2, 10, 100, 1000];
  const numPoints = 200;
  // Fresh Rng PER dimension with a dim-derived seed: same code path, but each
  // dimension's cloud is independent AND reproducible. (Sharing one Rng across
  // dims would couple them — order would change every cloud, see rng.ts header.)
  const ratios: number[] = [];
  console.log(`  N = ${numPoints} seeded Gaussian points per dimension`);
  console.log("");
  console.log("  d      minDist   maxDist   min/max ratio");
  console.log("  ----   -------   -------   -------------");
  for (const d of dims) {
    const { minDist, maxDist, ratio } = distanceContrast(new Rng(1000 + d), d, numPoints);
    ratios.push(ratio);
    console.log(
      `  ${String(d).padEnd(5)}  ${minDist.toFixed(3).padStart(7)}   ${maxDist
        .toFixed(3)
        .padStart(7)}   ${ratio.toFixed(4)}`,
    );
  }

  console.log("");
  console.log("  ratio vs dimension (higher = less distance contrast = worse):");
  console.log(asciiLine(ratios, dims.length, 8));
  console.log(
    `  ratio climbs ${ratios[0].toFixed(3)} (d=2) → ${ratios[ratios.length - 1].toFixed(
      3,
    )} (d=1000).`,
  );
  console.log(
    "  At high d the nearest point is nearly as far as the farthest: 'closest'",
  );
  console.log(
    "  stops being meaningful. One-hot lives in d = V space, already in this regime.",
  );

  // Same numbers as a bar chart keyed by dimension, so the trend reads even if
  // the 4-point line looks coarse.
  console.log("");
  console.log(asciiBar(dims.map((d) => `d=${d}`), ratios, 30));
}

// ---------------------------------------------------------------------------
// (3) Bag-of-words one-hot vs hand-built dense: topical retrieval
// ---------------------------------------------------------------------------

// A tiny labeled document set. Each doc is a few words on ONE topic. The retrieval
// task: given a query doc, find the most similar OTHER doc. The right answer is
// always "the other doc on the same topic". We compare two encodings.
interface LabeledDoc {
  label: string; // human topic tag, for printing the verdict (not used in scoring)
  words: string[];
}

// Critical design constraint for an HONEST demo: the two docs of each topic share
// ZERO literal words. If they shared a token, bag-of-words one-hot would "succeed"
// by accident (literal overlap), and the chapter's claim — one-hot can't see
// meaning — would be quietly contradicted by the numbers. So royalty-A/royalty-B,
// animals-A/animals-B, food-A/food-B are each disjoint word sets drawn from the
// same semantic cluster. This forces one-hot cosine to 0 for same-topic pairs.
const DOCS: LabeledDoc[] = [
  // royalty — disjoint: {king,queen,rules} vs {prince,princess,kingdom}
  { label: "royalty-A", words: ["king", "queen", "rules"] },
  { label: "royalty-B", words: ["prince", "princess", "kingdom"] },
  // animals — disjoint: {dog,cat,run} vs {mouse,bird}
  { label: "animals-A", words: ["dog", "cat", "run"] },
  { label: "animals-B", words: ["mouse", "bird"] },
  // food — disjoint: {bread,cheese} vs {rice,fish}
  { label: "food-A", words: ["bread", "cheese"] },
  { label: "food-B", words: ["rice", "fish"] },
];

// Encoding A — bag-of-words over one-hot: sum the one-hot vectors of a doc's
// words. The killer property: two docs share a non-zero coordinate ONLY if they
// share a LITERAL word. Same-topic docs with NO overlapping token (royalty-A uses
// king/queen, royalty-B uses prince/princess) are orthogonal → cosine 0, exactly
// as un-retrievable as two random docs. This is failure (1) propagated to docs.
function encodeBagOfWords(doc: LabeledDoc, vocabSize: number, stoi: Map<string, number>): number[] {
  const v = new Array(vocabSize).fill(0);
  for (const w of doc.words) {
    const id = stoi.get(w);
    if (id !== undefined) v[id] += 1; // unknown words silently skipped (toy vocab)
  }
  return v;
}

// Encoding B — hand-authored dense topic code. We DECLARE a 3-axis space
// [royalty, animal, food] and place each vocab word on those axes by meaning.
// This is cheating in the sense that a HUMAN supplied the structure — but that is
// exactly the point of the chapter: dense vectors CAN express graded similarity
// (king and prince both load on the royalty axis → non-zero cosine even with no
// shared token), whereas one-hot structurally cannot. Stage 03+ learns such axes
// from co-occurrence instead of hand-placing them.
const TOPIC_AXES = ["royalty", "animal", "food"] as const;
const WORD_TO_DENSE: Record<string, number[]> = {
  // [royalty, animal, food]
  king: [1, 0, 0],
  queen: [1, 0, 0],
  prince: [1, 0, 0],
  princess: [1, 0, 0],
  rules: [0.6, 0, 0],
  kingdom: [0.8, 0, 0],
  dog: [0, 1, 0],
  cat: [0, 1, 0],
  mouse: [0, 1, 0],
  bird: [0, 1, 0],
  run: [0, 0.6, 0],
  bread: [0, 0, 1],
  cheese: [0, 0, 1],
  rice: [0, 0, 1],
  fish: [0, 0, 1],
};

function encodeDense(doc: LabeledDoc): number[] {
  const v = new Array(TOPIC_AXES.length).fill(0);
  for (const w of doc.words) {
    const coords = WORD_TO_DENSE[w];
    if (!coords) continue; // word has no authored topic → contributes nothing
    for (let k = 0; k < coords.length; k++) v[k] += coords[k];
  }
  return v;
}

// Run nearest-neighbor retrieval for every doc under one encoding, and score it:
// a "hit" is when the top neighbor shares the same topic prefix (royalty/animals/
// food). Returns hits and the per-query verdict lines.
function evaluateRetrieval(matrix: number[][], labels: string[]): { hits: number; lines: string[] } {
  const lines: string[] = [];
  let hits = 0;
  for (let i = 0; i < matrix.length; i++) {
    // nearestNeighbors excludes self by reference identity — we MUST pass the
    // same row instance (matrix[i]) that lives in `matrix`, or self-match leaks.
    const nn = nearestNeighbors(matrix[i], matrix, 1)[0];
    const topic = labels[i].split("-")[0];
    const nnTopic = labels[nn.index].split("-")[0];
    const hit = topic === nnTopic;
    if (hit) hits++;
    lines.push(
      `    ${labels[i].padEnd(10)} → ${labels[nn.index].padEnd(10)} ` +
        `(cos ${nn.score.toFixed(3)})  ${hit ? "✓ same topic" : "✗ wrong topic"}`,
    );
  }
  return { hits, lines };
}

function demoDenseVsOneHot(): void {
  console.log("");
  console.log("=".repeat(64));
  console.log("(3) topical retrieval: bag-of-words one-hot vs hand-built dense");
  console.log("=".repeat(64));

  const corpus = generateCorpus(new Rng(1));
  const vocab = buildVocab(corpus.tokens);
  const labels = DOCS.map((d) => d.label);

  const bow = DOCS.map((d) => encodeBagOfWords(d, vocab.size, vocab.stoi));
  const dense = DOCS.map((d) => encodeDense(d));

  console.log(`  ${DOCS.length} docs, 3 topics x 2 docs each. Task: top-1 neighbor`);
  console.log("  should be the OTHER doc of the same topic (= a hit).");
  console.log("");

  console.log(`  [A] bag-of-words over one-hot  (dim = V = ${vocab.size})`);
  const bowResult = evaluateRetrieval(bow, labels);
  for (const l of bowResult.lines) console.log(l);
  console.log(`    hits: ${bowResult.hits}/${DOCS.length}`);

  console.log("");
  console.log(`  [B] hand-built dense topic code  (dim = ${TOPIC_AXES.length}: ${TOPIC_AXES.join("/")})`);
  const denseResult = evaluateRetrieval(dense, labels);
  for (const l of denseResult.lines) console.log(l);
  console.log(`    hits: ${denseResult.hits}/${DOCS.length}`);

  console.log("");
  console.log(
    `  verdict: dense ${denseResult.hits}/${DOCS.length} vs one-hot ${bowResult.hits}/${DOCS.length} ` +
      `in ${TOPIC_AXES.length} dims vs ${vocab.size}.`,
  );
  console.log(
    "  one-hot pairs that share no literal word (king/queen vs prince/princess)",
  );
  console.log(
    "  score cos 0 and cannot be retrieved; dense puts them on a shared axis.",
  );
}

// ---------------------------------------------------------------------------
// FAILURE MODE: high-dim one-hot + Euclidean NN ≈ random retrieval
// ---------------------------------------------------------------------------

// The trap a beginner falls into: "I'll just do nearest-neighbor search on the
// one-hot/bag-of-words vectors with Euclidean distance." We show this is no
// better than guessing. Between any two bag-of-words docs that share k words, the
// squared Euclidean distance is (|A| + |B| - 2k) where |.| counts tokens — it
// depends ONLY on literal token overlap, never on meaning. Docs with zero overlap
// (most same-topic pairs in our set) are tied at the same distance, so the tie-
// break (lowest index) decides the "neighbor" — i.e. arbitrary, not semantic.
function demoFailureMode(): void {
  console.log("");
  console.log("=".repeat(64));
  console.log("FAILURE MODE: Euclidean NN on one-hot bag-of-words is ~random");
  console.log("=".repeat(64));

  const corpus = generateCorpus(new Rng(1));
  const vocab = buildVocab(corpus.tokens);
  const labels = DOCS.map((d) => d.label);
  const bow = DOCS.map((d) => encodeBagOfWords(d, vocab.size, vocab.stoi));

  // For the royalty-A query, print Euclidean distance to EVERY other doc. The
  // same-topic doc (royalty-B, no shared word) is NOT closer than the off-topic
  // animal/food docs — it's tied with them. The reader sees the ties directly.
  const queryIdx = labels.indexOf("royalty-A");
  console.log(`  query = ${labels[queryIdx]} (words: ${DOCS[queryIdx].words.join(",")})`);
  console.log("  Euclidean distance to every other doc:");
  const distLines: Array<{ label: string; dist: number; sameTopic: boolean }> = [];
  for (let j = 0; j < bow.length; j++) {
    if (j === queryIdx) continue;
    distLines.push({
      label: labels[j],
      dist: euclidean(bow[queryIdx], bow[j]),
      sameTopic: labels[j].split("-")[0] === labels[queryIdx].split("-")[0],
    });
  }
  distLines.sort((a, b) => a.dist - b.dist);
  for (const d of distLines) {
    console.log(
      `    ${d.label.padEnd(10)} dist=${d.dist.toFixed(3)}  ${
        d.sameTopic ? "(SAME topic — should be nearest)" : "(different topic)"
      }`,
    );
  }

  // Quantify "≈ random": expected accuracy of a random top-1 pick. With 6 docs,
  // 1 correct answer among 5 candidates, blind guessing scores 1/5 = 20%. We
  // measure where Euclidean actually lands and compare.
  let euclHits = 0;
  for (let i = 0; i < bow.length; i++) {
    const nn = nearestNeighbors(bow[i], bow, 1)[0]; // cosine NN; but for binary BoW the ranking trap is identical
    // re-rank by Euclidean to make the failure mode explicit (not cosine):
    let best = -1;
    let bestD = Infinity;
    for (let j = 0; j < bow.length; j++) {
      if (j === i) continue;
      const dd = euclidean(bow[i], bow[j]);
      if (dd < bestD) {
        bestD = dd;
        best = j;
      }
    }
    void nn; // computed only to mirror the cosine path; Euclidean `best` is what we score
    if (labels[best].split("-")[0] === labels[i].split("-")[0]) euclHits++;
  }
  const euclAcc = euclHits / bow.length;
  const randomBaseline = 1 / (bow.length - 1); // 1 correct of N-1 candidates
  console.log("");
  console.log(
    `  Euclidean top-1 topical accuracy: ${euclHits}/${bow.length} = ${(euclAcc * 100).toFixed(
      1,
    )}%`,
  );
  console.log(
    `  random-guess baseline (1 of ${bow.length - 1}): ${(randomBaseline * 100).toFixed(1)}%`,
  );
  console.log(
    "  the distances are decided by literal token overlap, not topic, so same-topic",
  );
  console.log(
    "  docs with no shared word tie with off-topic docs → retrieval is ~chance.",
  );
  console.log(
    "  (toy scale: 6 docs make the %s coarse; the mechanism — overlap-only distance",
  );
  console.log("   ignoring meaning — is what holds at any vocab size.)");
}

function main(): void {
  demoOrthogonality();
  demoCurseOfDimensionality();
  demoDenseVsOneHot();
  demoFailureMode();
}

main();
