// core/text.ts — a toy corpus with semantic structure deliberately baked in.
//
// The whole book needs a corpus where the right answer is KNOWN, so we can check
// "did the model actually learn structure?" against ground truth instead of vibes.
// Real corpora don't give you that. So we generate text from templates over a
// small vocabulary clustered into semantic groups (royalty, gender, animals,
// food, directions...). The templates force co-occurrence patterns that encode
// analogies (king:queen :: man:woman) and category neighborhoods.
//
// Why this is honest: we are NOT cheating by handing the model the answer. The
// model only ever sees flat token streams. The structure is *latent* in the
// co-occurrence statistics, exactly as in real language — it just happens to be
// learnable in seconds because the vocab is ~40 words and the patterns are dense.
// The book states clearly: absolute neighbor quality here is optimistic; what
// transfers to real corpora is the mechanism and the relative trends.
//
// Failure modes this design surfaces (used in later stages):
//  - words that NEVER co-occur with anything informative (we include a few
//    "filler" tokens) → their embeddings stay near init: shows you can't learn
//    structure that isn't in the data.
//  - frequency skew (some templates repeat more) → motivates negative-sampling's
//    unigram^0.75 and Adam's per-param lr.

import { Rng, sampleCategorical } from "./rng.js";

// Semantic clusters. Each inner pattern is a template; {X} slots get filled from
// a paired list so the analogy axes (royalty<->commoner, male<->female) recur.
// The design goal: pairs that should end up parallel in vector space (king-queen,
// man-woman) appear in the SAME template slots, so their context distributions
// differ only along the intended axis.
const TEMPLATES: string[][] = [
  // royalty / gender axis — the canonical analogy testbed
  ["the", "king", "rules", "the", "kingdom"],
  ["the", "queen", "rules", "the", "kingdom"],
  ["the", "king", "is", "a", "man"],
  ["the", "queen", "is", "a", "woman"],
  ["a", "man", "and", "a", "woman", "walk"],
  ["the", "prince", "is", "a", "young", "man"],
  ["the", "princess", "is", "a", "young", "woman"],
  // animals cluster
  ["the", "dog", "and", "the", "cat", "run"],
  ["the", "cat", "chased", "the", "mouse"],
  ["the", "dog", "ate", "the", "bone"],
  ["a", "bird", "and", "a", "cat", "sit"],
  // food cluster
  ["she", "ate", "bread", "and", "cheese"],
  ["he", "ate", "rice", "and", "fish"],
  ["the", "cat", "ate", "fish"],
  ["they", "ate", "bread", "and", "rice"],
  // direction cluster
  ["go", "north", "then", "go", "south"],
  ["go", "east", "then", "go", "west"],
  ["north", "is", "not", "south"],
  ["east", "is", "not", "west"],
  // filler / low-signal sentences (intentional dead weight, see header)
  ["um", "the", "thing", "is", "here"],
  ["well", "that", "thing", "is", "there"],
];

// Tokenized result of corpus generation.
export interface Corpus {
  tokens: string[]; // flat token stream
  sentences: string[][]; // sentence boundaries preserved (windows shouldn't cross them)
}

// Generate a seeded corpus by sampling templates `numSentences` times. We use a
// skewed template distribution (some templates 2-3x likelier) ON PURPOSE to
// create frequency imbalance, which later stages need to demonstrate sampling
// tricks. Determinism: identical seed → identical token stream.
export function generateCorpus(rng: Rng, numSentences = 400): Corpus {
  // Skewed weights: royalty/gender templates (indices 0..6) get higher weight so
  // those words are frequent; filler (last two) get low weight.
  const weights = TEMPLATES.map((_, i) => {
    if (i <= 6) return 3; // analogy templates: frequent
    if (i >= TEMPLATES.length - 2) return 1; // filler: rare
    return 2;
  });
  const sentences: string[][] = [];
  const tokens: string[] = [];
  for (let s = 0; s < numSentences; s++) {
    const t = TEMPLATES[sampleCategorical(rng, weights)];
    const sent = t.slice(); // copy; never mutate the template
    sentences.push(sent);
    for (const w of sent) tokens.push(w);
  }
  return { tokens, sentences };
}

// Tokenize a raw string (used when a stage wants to feed custom text). Lowercase,
// split on non-letters. Minimal on purpose — real tokenization is a rabbit hole
// orthogonal to representation learning, which is the book's actual subject.
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter((t) => t.length > 0);
}

// Vocabulary: bidirectional token<->id mapping plus frequency counts. IDs are
// assigned in DESCENDING frequency order (id 0 = most frequent), a convention
// that makes frequency-based negative sampling tables and any "drop rare words"
// logic trivially sliceable. Deterministic given a fixed token stream.
export interface Vocab {
  itos: string[]; // id -> token
  stoi: Map<string, number>; // token -> id
  counts: number[]; // id -> raw frequency
  size: number;
}

export function buildVocab(tokens: string[]): Vocab {
  const freq = new Map<string, number>();
  for (const t of tokens) freq.set(t, (freq.get(t) ?? 0) + 1);
  // Sort by count desc, then token asc for a stable, seed-independent order.
  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  const itos = sorted.map((e) => e[0]);
  const counts = sorted.map((e) => e[1]);
  const stoi = new Map<string, number>();
  itos.forEach((t, i) => stoi.set(t, i));
  return { itos, stoi, counts, size: itos.length };
}

// A single (center, context) training pair as token IDs, the atomic unit skip-gram
// trains on.
export interface Pair {
  center: number;
  context: number;
}

// Enumerate skip-gram (center, context) pairs within a symmetric window.
// Windows DO NOT cross sentence boundaries — crossing them would invent
// co-occurrences that the "language" never produced, polluting the statistics
// the model is supposed to recover. This per-sentence loop is that guard.
//
// Returns every ordered (center, context) pair; later stages may subsample.
export function windowPairs(sentences: string[][], vocab: Vocab, windowSize = 2): Pair[] {
  const pairs: Pair[] = [];
  for (const sent of sentences) {
    const ids = sent.map((w) => vocab.stoi.get(w)!);
    for (let i = 0; i < ids.length; i++) {
      const lo = Math.max(0, i - windowSize);
      const hi = Math.min(ids.length - 1, i + windowSize);
      for (let j = lo; j <= hi; j++) {
        if (j === i) continue; // a word is not its own context
        pairs.push({ center: ids[i], context: ids[j] });
      }
    }
  }
  return pairs;
}

// Dense co-occurrence matrix M[i][j] = how often word j appears in word i's
// window. Symmetric by construction here (window is symmetric). This is the raw
// material for the PMI / count-based stage and a useful sanity heatmap. O(V^2)
// memory — fine at toy V, flagged in the book as the reason count-based methods
// don't scale to real vocab.
export function cooccurrenceMatrix(
  sentences: string[][],
  vocab: Vocab,
  windowSize = 2,
): number[][] {
  const V = vocab.size;
  const M: number[][] = Array.from({ length: V }, () => new Array(V).fill(0));
  for (const { center, context } of windowPairs(sentences, vocab, windowSize)) {
    M[center][context] += 1;
  }
  return M;
}
