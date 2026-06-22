// core/tasks.ts — Toy research objects whose ground truth we can compute EXACTLY.
//
// WHY synthetic tasks with oracles: interpretability is only credible when you can check
//   your claim against a known answer. "We think this head copies the previous token" is
//   verifiable only if the task HAS a previous-token-copy structure we defined. Each task
//   ships an oracle(x) that returns the exact correct next-token target, so every stage can
//   ask "did the model actually learn the structure, or just memorize?" with a real number.
//
// FORMAT: each task produces SEQUENCES of token ids and a NEXT-TOKEN target per position
//   (autoregressive). makeBatch returns {inputs, targets, meta} where inputs[b] is one
//   sequence and targets[b][i] is the correct token to predict AT position i (given tokens
//   0..i). meta carries task-specific ground truth (e.g. the repeat boundary) for interp.
//
// HONESTY: these absolute difficulties are tiny. The point is the STRUCTURE — modular
//   arithmetic has a known Fourier/algebraic solution; induction needs a 2-head copy
//   circuit; skip-trigram is a 1-layer attention pattern. Stages locate THOSE structures;
//   the magnitudes (loss, accuracy) don't transfer, the mechanisms do.

import { type Rng, shuffle } from "./rng.js";

export interface TaskBatch {
  inputs: number[][]; // [batch][seq] token ids
  targets: number[][]; // [batch][seq] next-token target per position
  meta: Record<string, unknown>; // task-specific ground truth for interp
}

export interface Task {
  name: string;
  vocab: number;
  seqLen: number;
  makeBatch(batchSize: number, rng: Rng): TaskBatch;
  /** Exact correct next-token targets for ONE input sequence. The ground truth oracle. */
  oracle(input: number[]): number[];
  /** Which positions are "scorable" — positions where the task is actually determined.
   *  Early positions in copy/induction have no determined answer; scoring them would
   *  dilute accuracy with un-learnable noise and HIDE whether the circuit works. */
  scorablePositions(input: number[]): number[];
}

// ----------------------------------------------------------------------------
// modAdd(p): sequence [a, b, =] -> predict (a+b) mod p at the "=" position.
// ----------------------------------------------------------------------------
//
// VOCAB: 0..p-1 are numbers, p is the "=" delimiter token. seq = 3: [a, b, =].
// GROUND TRUTH: only the "=" position (index 2) is scorable; its target is (a+b) mod p.
// WHY this task is a famous interp object: the optimal solution has a clean ALGEBRAIC
//   structure (the model learns to embed numbers on a circle and add angles). Stages can
//   check whether the learned embeddings show that circular structure — a mechanism that
//   is identical in kind at any scale, even though tiny p makes it trivially small.
export function modAdd(p: number): Task {
  const EQ = p; // delimiter token id
  const vocab = p + 1;
  return {
    name: `modAdd(${p})`,
    vocab,
    seqLen: 3,
    makeBatch(batchSize, rng) {
      const inputs: number[][] = [];
      const targets: number[][] = [];
      for (let b = 0; b < batchSize; b++) {
        const a = Math.floor(rng() * p);
        const bb = Math.floor(rng() * p);
        const seq = [a, bb, EQ];
        inputs.push(seq);
        targets.push(this.oracle(seq));
      }
      return { inputs, targets, meta: { p, EQ } };
    },
    oracle(input) {
      const [a, b] = input;
      // Targets at positions 0,1 are "don't care" (no determined answer); we fill them with
      // a benign placeholder (the EQ token) and rely on scorablePositions to ignore them.
      return [EQ, EQ, (a + b) % p];
    },
    scorablePositions() {
      return [2];
    },
  };
}

// ----------------------------------------------------------------------------
// copyTask: [t0..t_{n-1}, SEP, t0..t_{n-1}] — predict the second copy from the first.
// ----------------------------------------------------------------------------
//
// VOCAB: 0..k-1 content tokens, k = SEP. seq = 2n+1. After SEP, position i must reproduce
//   token i of the first half. SOLUTION requires an induction-like copy: attend back n+1
//   positions and copy. Only the second-half positions are scorable.
export function copyTask(k: number, n: number): Task {
  const SEP = k;
  const vocab = k + 1;
  const seqLen = 2 * n + 1;
  return {
    name: `copy(k=${k},n=${n})`,
    vocab,
    seqLen,
    makeBatch(batchSize, rng) {
      const inputs: number[][] = [];
      const targets: number[][] = [];
      for (let b = 0; b < batchSize; b++) {
        const first: number[] = [];
        for (let i = 0; i < n; i++) first.push(Math.floor(rng() * k));
        const seq = [...first, SEP, ...first];
        inputs.push(seq);
        targets.push(this.oracle(seq));
      }
      return { inputs, targets, meta: { SEP, n } };
    },
    oracle(input) {
      // Next-token target at position i is input[i+1] (standard AR); for the copy region
      // that equals the corresponding first-half token, which is the learnable structure.
      const t: number[] = [];
      for (let i = 0; i < input.length; i++) t.push(i + 1 < input.length ? input[i + 1] : input[i]);
      return t;
    },
    scorablePositions(input) {
      // Scorable: positions whose NEXT token lies in the second copy (after SEP). Those are
      // positions SEP_index .. seqLen-2.
      const sepIdx = input.indexOf(SEP);
      const pos: number[] = [];
      for (let i = sepIdx; i < input.length - 1; i++) pos.push(i);
      return pos;
    },
  };
}

// ----------------------------------------------------------------------------
// inductionTask: random sequence containing a repeated bigram [A][B]...[A] -> predict [B].
// ----------------------------------------------------------------------------
//
// This is the CANONICAL induction-head probe. We plant a bigram (A,B) early, then place A
//   again near the end; the position right after the second A must predict B. A model with
//   an induction head solves it (attend to the token AFTER the previous occurrence of the
//   current token). meta.queryPos and meta.answer give the exact ground truth so stage06
//   can verify the head's attention lands on the right source position.
export function inductionTask(vocab: number, seqLen: number): Task {
  return {
    name: `induction(vocab=${vocab},seq=${seqLen})`,
    vocab,
    seqLen,
    makeBatch(batchSize, rng) {
      const inputs: number[][] = [];
      const targets: number[][] = [];
      const metas: { aPos1: number; aPos2: number; A: number; B: number }[] = [];
      for (let b = 0; b < batchSize; b++) {
        // random filler
        const seq: number[] = [];
        for (let i = 0; i < seqLen; i++) seq.push(Math.floor(rng() * vocab));
        // plant the bigram: choose A, B; place at an early pos and re-place A near the end
        const A = Math.floor(rng() * vocab);
        let B = Math.floor(rng() * vocab);
        if (B === A) B = (B + 1) % vocab; // keep bigram non-trivial
        const aPos1 = 1 + Math.floor(rng() * Math.max(1, Math.floor(seqLen / 2) - 1));
        seq[aPos1] = A;
        seq[aPos1 + 1] = B;
        const aPos2 = seqLen - 2; // second A so that next-token at aPos2 should be B
        seq[aPos2] = A;
        inputs.push(seq);
        targets.push(this.oracle(seq));
        metas.push({ aPos1, aPos2, A, B });
      }
      return { inputs, targets, meta: { plants: metas } };
    },
    oracle(input) {
      const t: number[] = [];
      for (let i = 0; i < input.length; i++) t.push(i + 1 < input.length ? input[i + 1] : input[i]);
      return t;
    },
    scorablePositions(input) {
      // The decisive position is the second A (last-but-one); its next token should be B.
      return [input.length - 2];
    },
  };
}

// ----------------------------------------------------------------------------
// skipTrigram: pattern "A ... B C" where seeing A earlier biases predicting C after B.
// ----------------------------------------------------------------------------
//
// A 1-layer attention head can implement skip-trigrams (Anthropic's "A... B -> C"). We
//   plant: token A somewhere early; near the end token B; the position after B should
//   predict C, but ONLY because A appeared. This isolates the simplest non-trivial
//   attention computation. meta records the planted (A,B,C) and positions.
export function skipTrigram(vocab: number, seqLen: number): Task {
  return {
    name: `skipTrigram(vocab=${vocab},seq=${seqLen})`,
    vocab,
    seqLen,
    makeBatch(batchSize, rng) {
      const inputs: number[][] = [];
      const targets: number[][] = [];
      const plants: { A: number; B: number; C: number; aPos: number; bPos: number }[] = [];
      for (let b = 0; b < batchSize; b++) {
        const seq: number[] = [];
        for (let i = 0; i < seqLen; i++) seq.push(Math.floor(rng() * vocab));
        const A = Math.floor(rng() * vocab);
        const B = (A + 1) % vocab;
        const C = (A + 2) % vocab; // deterministic A->C mapping the head must learn
        const aPos = 1 + Math.floor(rng() * Math.max(1, Math.floor(seqLen / 2)));
        const bPos = seqLen - 2;
        seq[aPos] = A;
        seq[bPos] = B;
        // target after B is C
        inputs.push(seq);
        const tgt = this.oracle(seq);
        tgt[bPos] = C; // override AR target at the decisive position with the skip-trigram answer
        targets.push(tgt);
        plants.push({ A, B, C, aPos, bPos });
      }
      return { inputs, targets, meta: { plants } };
    },
    oracle(input) {
      const t: number[] = [];
      for (let i = 0; i < input.length; i++) t.push(i + 1 < input.length ? input[i + 1] : input[i]);
      return t;
    },
    scorablePositions(input) {
      return [input.length - 2];
    },
  };
}

/** Build a flat (rows, vocab) training tensor view: flatten all sequences' positions into
 *  rows. Returned as parallel arrays the training loop turns into per-sequence forwards.
 *  Shuffling here keeps batch order reproducible yet decorrelated across epochs. */
export function flattenBatch(batch: TaskBatch, rng: Rng): { inputs: number[][]; order: number[] } {
  const order = batch.inputs.map((_, i) => i);
  shuffle(order, rng);
  return { inputs: batch.inputs, order };
}
