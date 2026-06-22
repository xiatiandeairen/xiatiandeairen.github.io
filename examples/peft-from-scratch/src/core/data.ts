// core/data.ts — Seeded toy datasets that make "fine-tuning" actually observable.
//
// WHY these specific tasks: PEFT is about ADAPTING a pretrained model to a NEW but RELATED
//   distribution with few params. To show that, we need a base task A and a finetune task B
//   that (a) share structure (so the frozen base is useful) yet (b) differ (so adaptation
//   is measurable). The pretrain→finetune pair below is built exactly for that: task A and
//   task B draw from the same token alphabet and sequence shape but apply a DIFFERENT target
//   transform. A model pretrained on A is a good init for B but not perfect — leaving room a
//   tiny LoRA/adapter can close. That measurable gap is the book's recurring evidence.
//
// INVARIANT: everything is seeded via core/prng. Same seed => identical splits, so a
//   stage's reported "converged in N steps" is reproducible.
//
// FAILURE MODE guarded: tasks A and B too similar (base already solves B -> nothing to
//   learn, PEFT looks falsely free) OR too different (frozen base useless -> PEFT looks
//   falsely weak). The transforms below are picked to be related-but-distinct on purpose.

import { randint, shuffle } from "./prng.js";

/** A single supervised example: integer input sequence -> integer target sequence. */
export interface SeqExample {
  input: number[];
  target: number[];
}

/** A classification example: feature vector -> class label. */
export interface ClsExample {
  x: number[];
  y: number; // class index
}

// ---------------------------------------------------------------------------
// Sequence transforms. Vocab is small so a toy model can model it; the target is a
// deterministic function of the input so loss can in principle reach ~0.
// ---------------------------------------------------------------------------

/** copy: target = input. The trivial identity task — useful as a sanity baseline. */
export function genCopy(n: number, seqLen: number, vocab: number): SeqExample[] {
  const out: SeqExample[] = [];
  for (let i = 0; i < n; i++) {
    const input = Array.from({ length: seqLen }, () => randint(0, vocab));
    out.push({ input, target: input.slice() });
  }
  return out;
}

/** reverse: target = input reversed. Needs the model to use positional information. */
export function genReverse(n: number, seqLen: number, vocab: number): SeqExample[] {
  const out: SeqExample[] = [];
  for (let i = 0; i < n; i++) {
    const input = Array.from({ length: seqLen }, () => randint(0, vocab));
    out.push({ input, target: input.slice().reverse() });
  }
  return out;
}

/** sort: target = input sorted ascending. The hardest of the three (global reordering). */
export function genSort(n: number, seqLen: number, vocab: number): SeqExample[] {
  const out: SeqExample[] = [];
  for (let i = 0; i < n; i++) {
    const input = Array.from({ length: seqLen }, () => randint(0, vocab));
    out.push({ input, target: input.slice().sort((a, b) => a - b) });
  }
  return out;
}

/**
 * Toy linearly-separable-ish classification: k Gaussian blobs in dim-d space.
 * Used by stages that want a classification head rather than a seq task.
 */
export function genClassification(n: number, dim: number, classes: number): ClsExample[] {
  // fixed per-class centers derived from class index (deterministic, no extra draws)
  const centers: number[][] = [];
  for (let c = 0; c < classes; c++) {
    centers.push(Array.from({ length: dim }, (_, j) => Math.cos(c * 1.7 + j * 0.9) * 2));
  }
  const out: ClsExample[] = [];
  for (let i = 0; i < n; i++) {
    const y = randint(0, classes);
    const x = centers[y].map((m) => m + (randint(0, 1000) / 1000 - 0.5)); // small uniform jitter
    out.push({ x, y });
  }
  return out;
}

// ---------------------------------------------------------------------------
// The pretrain A -> finetune B pair — the spine of the whole book.
// ---------------------------------------------------------------------------

export type PairTask = "copy" | "reverse" | "sort";

export interface PretrainFinetune {
  pretrain: SeqExample[]; // task A
  finetune: SeqExample[]; // task B (related distribution, different transform)
  taskA: PairTask;
  taskB: PairTask;
  seqLen: number;
  vocab: number;
}

/**
 * Build a related (A, B) pair over a SHARED vocab and sequence length.
 * DEFAULT pairing copy->reverse: B reuses A's "remember the tokens" skill but adds
 *   "emit them in opposite order" — a localized change a low-rank delta can express.
 *   That is the canonical setup behind "LoRA closes the A->B gap with 0.5% of params".
 */
export function genPretrainFinetune(opts?: {
  nPretrain?: number;
  nFinetune?: number;
  seqLen?: number;
  vocab?: number;
  taskA?: PairTask;
  taskB?: PairTask;
}): PretrainFinetune {
  const nPretrain = opts?.nPretrain ?? 256;
  const nFinetune = opts?.nFinetune ?? 64;
  const seqLen = opts?.seqLen ?? 8;
  const vocab = opts?.vocab ?? 12;
  const taskA = opts?.taskA ?? "copy";
  const taskB = opts?.taskB ?? "reverse";
  const gen = (t: PairTask, n: number): SeqExample[] =>
    t === "copy" ? genCopy(n, seqLen, vocab) : t === "reverse" ? genReverse(n, seqLen, vocab) : genSort(n, seqLen, vocab);
  return {
    pretrain: gen(taskA, nPretrain),
    finetune: gen(taskB, nFinetune),
    taskA,
    taskB,
    seqLen,
    vocab,
  };
}

// ---------------------------------------------------------------------------
// Batch iteration
// ---------------------------------------------------------------------------

/**
 * Yield shuffled minibatches over examples. Shuffling uses the global PRNG, so the batch
 *   order is reproducible. INVARIANT: the last (possibly short) batch is included, not
 *   dropped — at toy data sizes dropping it would discard a meaningful fraction of data.
 */
export function* batches<T>(items: T[], batchSize: number, shuffleEach = true): Generator<T[]> {
  const idx = items.map((_, i) => i);
  if (shuffleEach) shuffle(idx);
  for (let i = 0; i < idx.length; i += batchSize) {
    yield idx.slice(i, i + batchSize).map((j) => items[j]);
  }
}
