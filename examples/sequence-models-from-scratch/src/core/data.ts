// core/data.ts — Deterministic toy task generators for sequence models.
//
// WHY toy tasks (copy / adding / parity / delayed-recall) instead of real corpora:
//   Every task here has a TUNABLE dependency span. The whole point of the book is to
//   show that vanilla RNNs fail when the span exceeds their effective memory, and that
//   gating (LSTM/GRU) / attention / SSM extend that span. A real corpus mixes many
//   span lengths and hides the mechanism. These tasks isolate the one variable.
//
// HONESTY: absolute accuracies on these toy tasks are OPTIMISTIC — tiny vocab, clean
//   signal, no distribution shift. What transfers to real settings is the RELATIVE
//   story (RNN cliff vs LSTM plateau, grad-norm magnitude vs span), not the numbers.
//
// INVARIANT: every generator threads an Rng (core/prng.ts). Same seed => same dataset,
//   bit for bit. No Math.random anywhere.
//
// REPRESENTATION: sequences are number[][] (examples x timesteps); we keep raw integer
//   ids / floats and let each stage embed/encode as it sees fit. describe() prints the
//   exact shape + dependency span so a chapter's "honest number" header is grounded.

import type { Rng } from "./prng.js";

export interface Dataset {
  X: number[][]; // examples x timesteps (token ids, or packed features per stage)
  Y: number[][]; // targets; shape depends on task (see each generator)
  vocabSize: number;
  describe(): string;
}

/**
 * copyTask: emit k random tokens, then `delay` blanks, then a delimiter; the model must
 * reproduce the first k tokens at the end. The classic long-dependency stress test:
 * the answer depends on input from `delay+k` steps earlier. Increase `delay` to push
 * any architecture past its memory horizon.
 *
 * Vocab layout: 0 = blank, 1 = delimiter, 2..vocab-1 = payload symbols.
 * X length = k + delay + 1 + k (payload, blanks, delim, answer-region placeholders).
 * Y length = same; only the last k positions carry the target ids, rest are 0 (blank)
 *   so a stage can mask the loss to the answer region.
 */
export function copyTask(rng: Rng, opts: { count: number; k: number; delay: number; symbols: number }): Dataset {
  const { count, k, delay, symbols } = opts;
  const BLANK = 0;
  const DELIM = 1;
  const base = 2; // first payload symbol id
  const vocabSize = base + symbols;
  const T = k + delay + 1 + k;
  const X: number[][] = [];
  const Y: number[][] = [];
  for (let n = 0; n < count; n++) {
    const payload: number[] = [];
    for (let i = 0; i < k; i++) payload.push(base + rng.randint(0, symbols));
    const x = new Array<number>(T).fill(BLANK);
    const y = new Array<number>(T).fill(BLANK);
    for (let i = 0; i < k; i++) x[i] = payload[i]; // payload at the front
    x[k + delay] = DELIM; // delimiter signals "start reproducing now"
    for (let i = 0; i < k; i++) y[k + delay + 1 + i] = payload[i]; // answer at the tail
    X.push(x);
    Y.push(y);
  }
  return {
    X,
    Y,
    vocabSize,
    describe: () =>
      `copyTask: ${count} seqs, T=${T} (k=${k} payload + delay=${delay} + delim + k answer), ` +
      `vocab=${vocabSize}, dependency span=${k + delay + 1} steps`,
  };
}

/**
 * addingProblem (LSTM paper's original benchmark): two input channels per timestep —
 * a random value in [0,1) and a 0/1 marker. Exactly two timesteps are marked 1; the
 * target is the SUM of the two marked values. Long dependency because the two marks
 * can be far apart.
 *
 * Packing: X[n] is length 2*T, interleaved [val_0, mark_0, val_1, mark_1, ...] so it
 * fits the number[][] shape; a stage unpacks to a [T,2] feature sequence.
 * Y[n] = [sum] (a single regression target). vocabSize is meaningless (regression);
 * set to 0 as a signal "not a classification task".
 */
export function addingProblem(rng: Rng, opts: { count: number; T: number }): Dataset {
  const { count, T } = opts;
  const X: number[][] = [];
  const Y: number[][] = [];
  for (let n = 0; n < count; n++) {
    const vals = new Array<number>(T);
    for (let t = 0; t < T; t++) vals[t] = rng.next();
    // mark two distinct positions: one in the first half, one in the second half
    // (matches the paper; guarantees a genuinely long-range pair).
    const i = rng.randint(0, Math.max(1, Math.floor(T / 2)));
    let j = rng.randint(Math.floor(T / 2), T);
    if (j === i) j = (j + 1) % T;
    const packed: number[] = [];
    for (let t = 0; t < T; t++) {
      packed.push(vals[t]);
      packed.push(t === i || t === j ? 1 : 0);
    }
    X.push(packed);
    Y.push([vals[i] + vals[j]]);
  }
  return {
    X,
    Y,
    vocabSize: 0,
    describe: () =>
      `addingProblem: ${count} seqs, T=${T}, regression target=sum of 2 marked values, ` +
      `marks split across halves => span up to ~${T} steps`,
  };
}

/**
 * parityTask: input is a random bit string; target at the final step is the XOR
 * (parity) of all bits. Minimal task that REQUIRES carrying one bit of state across
 * the whole sequence — a memoryless model cannot do better than chance.
 * Vocab: {0,1}. Y[n] = [parity] (single target at sequence end).
 */
export function parityTask(rng: Rng, opts: { count: number; T: number }): Dataset {
  const { count, T } = opts;
  const X: number[][] = [];
  const Y: number[][] = [];
  for (let n = 0; n < count; n++) {
    const x = new Array<number>(T);
    let parity = 0;
    for (let t = 0; t < T; t++) {
      const b = rng.randint(0, 2);
      x[t] = b;
      parity ^= b;
    }
    X.push(x);
    Y.push([parity]);
  }
  return {
    X,
    Y,
    vocabSize: 2,
    describe: () => `parityTask: ${count} seqs, T=${T}, target=XOR of all bits (1 bit of state across full span)`,
  };
}

/**
 * delayedRecall: show ONE cue token at position 0, then `delay` random distractor
 * tokens; the model must output the cue token at the final step. The simplest knob to
 * sweep dependency span: span == delay+1 exactly. Use it to find an architecture's
 * memory horizon by increasing delay until accuracy collapses.
 * Vocab: 0 = distractor-only filler is avoided; tokens 1..symbols are cues/distractors.
 * Y[n] = [cue] (single target at the end).
 */
export function delayedRecall(rng: Rng, opts: { count: number; delay: number; symbols: number }): Dataset {
  const { count, delay, symbols } = opts;
  const vocabSize = symbols + 1; // +1 reserves 0 as a neutral filler id
  const T = 1 + delay;
  const X: number[][] = [];
  const Y: number[][] = [];
  for (let n = 0; n < count; n++) {
    const cue = 1 + rng.randint(0, symbols);
    const x = new Array<number>(T);
    x[0] = cue;
    for (let t = 1; t < T; t++) x[t] = 1 + rng.randint(0, symbols); // distractors share the vocab
    X.push(x);
    Y.push([cue]);
  }
  return {
    X,
    Y,
    vocabSize,
    describe: () => `delayedRecall: ${count} seqs, T=${T}, recall cue from step 0 (exact span=${delay + 1})`,
  };
}

// A fixed, license-free snippet (public-domain Aesop) for char-level LM. Kept short so
// a toy model overfits it in seconds — the chapter's point is the training DYNAMICS
// (loss/perplexity curve, sampling coherence), not generalization.
const CHAR_CORPUS =
  "the north wind and the sun disputed which was the stronger. " +
  "a traveler came along wrapped in a warm cloak. " +
  "they agreed that the one who first made the traveler take his cloak off " +
  "should be considered stronger than the other. " +
  "then the north wind blew as hard as he could, " +
  "but the more he blew the more closely did the traveler fold his cloak around him; " +
  "and at last the north wind gave up the attempt. " +
  "then the sun shone out warmly, and immediately the traveler took off his cloak. " +
  "and so the north wind was obliged to confess that the sun was the stronger of the two.";

/**
 * charSeq: character-level language modeling over a built-in fixed corpus (or override).
 * Returns next-char prediction pairs: X[n] = chars[0..L-1], Y[n] = chars[1..L] (shifted
 * by one). vocabSize = number of distinct chars. Exposes the id<->char maps via a
 * sidecar so stages can decode samples.
 */
export interface CharDataset extends Dataset {
  stoi: Map<string, number>;
  itos: string[];
  encode(s: string): number[];
  decode(ids: number[]): string;
}

export function charSeq(opts: { seqLen: number; stride?: number; text?: string } = { seqLen: 32 }): CharDataset {
  const text = opts.text ?? CHAR_CORPUS;
  const stride = opts.stride ?? opts.seqLen;
  const chars = Array.from(new Set(text.split(""))).sort(); // sort => deterministic id order
  const stoi = new Map<string, number>();
  chars.forEach((c, i) => stoi.set(c, i));
  const itos = chars;
  const encode = (s: string): number[] => Array.from(s).map((c) => stoi.get(c)!);
  const decode = (ids: number[]): string => ids.map((i) => itos[i]).join("");
  const ids = encode(text);
  const X: number[][] = [];
  const Y: number[][] = [];
  const L = opts.seqLen;
  for (let start = 0; start + L < ids.length; start += stride) {
    X.push(ids.slice(start, start + L));
    Y.push(ids.slice(start + 1, start + L + 1)); // shifted-by-one next-char targets
  }
  return {
    X,
    Y,
    vocabSize: chars.length,
    stoi,
    itos,
    encode,
    decode,
    describe: () =>
      `charSeq: ${X.length} windows of len ${L} (stride ${stride}), vocab=${chars.length} chars, ` +
      `corpus=${text.length} chars`,
  };
}

/**
 * Batch iterator: yields arrays of row indices of size `batchSize`. Threads the Rng so
 * shuffling is reproducible. INVARIANT: the last partial batch is INCLUDED (not dropped)
 * so tiny toy datasets don't lose examples — stages must handle variable batch length.
 */
export function* batches(n: number, batchSize: number, rng: Rng, shuffle = true): Generator<number[]> {
  const idx = Array.from({ length: n }, (_, i) => i);
  if (shuffle) rng.shuffle(idx);
  for (let i = 0; i < n; i += batchSize) yield idx.slice(i, Math.min(i + batchSize, n));
}
