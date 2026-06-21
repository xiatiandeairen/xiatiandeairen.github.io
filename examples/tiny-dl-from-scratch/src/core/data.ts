// core/data.ts — Toy, fully-reproducible datasets. Offline, no downloads, no network.
//
// WHY toy data: this book teaches the MACHINE, not SOTA accuracy. Toy sets let every
//   example converge on a CPU in seconds-to-minutes and stay bit-for-bit reproducible.
//
// HONESTY BOUNDARY (stated once, applies to every stage that trains):
//   Absolute metrics here are OPTIMISTIC — spiral/moons are low-dimensional and
//   linearly-almost-separable after one hidden layer; the GPT corpus is a tiny repeating
//   structure picked SO THAT a few-thousand-param model can overfit it in CPU-minutes.
//   What transfers is the TREND: train loss decreasing monotonically, grad-check error
//   < 1e-6, and the overfitting signature (train loss -> ~0 while val loss stalls/rises).
//   Do not quote these accuracies as if they generalize.

import { randn, shuffle, type Rng } from "./rng.js";

export interface Dataset2D {
  X: number[][]; // (n, 2) points
  y: number[]; // (n,) integer class labels
  classes: number;
}

/**
 * Two interleaved spiral arms (classes). Classic non-linear toy problem: NOT linearly
 * separable, so a linear model is stuck near chance — a clean way to show that hidden
 * layers + nonlinearity actually buy something (chapters 3/5).
 * Determinism: all noise comes from the passed rng.
 */
export function makeSpiral(pointsPerClass: number, classes: number, rng: Rng, noise = 0.15): Dataset2D {
  const X: number[][] = [];
  const y: number[] = [];
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < pointsPerClass; i++) {
      const r = i / pointsPerClass; // radius 0..1
      // angle: each class offset by a full turn fraction; +r*4 makes the arm spiral.
      const theta = c * ((2 * Math.PI) / classes) + r * 4 + randn(rng) * noise;
      X.push([r * Math.sin(theta), r * Math.cos(theta)]);
      y.push(c);
    }
  }
  return { X, y, classes };
}

/**
 * Two interleaving half-moons (binary). Slightly easier than spiral but with a curved
 * boundary; good for visual sanity checks of a 2-class decision surface.
 */
export function makeMoons(n: number, rng: Rng, noise = 0.1): Dataset2D {
  const X: number[][] = [];
  const y: number[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const isUpper = i < half;
    const t = (isUpper ? i : i - half) / Math.max(1, half - 1); // 0..1
    const angle = Math.PI * t;
    if (isUpper) {
      X.push([Math.cos(angle) + randn(rng) * noise, Math.sin(angle) + randn(rng) * noise]);
      y.push(0);
    } else {
      // shifted & flipped lower moon
      X.push([1 - Math.cos(angle) + randn(rng) * noise, 0.5 - Math.sin(angle) + randn(rng) * noise]);
      y.push(1);
    }
  }
  return { X, y, classes: 2 };
}

/** Split a 2-D dataset into train/val by ratio, shuffling first (deterministic via rng). */
export function trainValSplit(ds: Dataset2D, valRatio: number, rng: Rng): { train: Dataset2D; val: Dataset2D } {
  const idx = shuffle(
    Array.from({ length: ds.y.length }, (_, i) => i),
    rng,
  );
  const nVal = Math.floor(ds.y.length * valRatio);
  const valIdx = idx.slice(0, nVal);
  const trainIdx = idx.slice(nVal);
  const pick = (ids: number[]): Dataset2D => ({
    X: ids.map((i) => ds.X[i]),
    y: ids.map((i) => ds.y[i]),
    classes: ds.classes,
  });
  return { train: pick(trainIdx), val: pick(valIdx) };
}

// ----------------------------------------------------------------------------
// Character-level language dataset.
// ----------------------------------------------------------------------------

/**
 * A deterministic, structured toy corpus. WHY repeating structure: a tiny model (a few
 * thousand params) can memorize/overfit it in CPU-minutes, letting us SHOW real loss
 * descent and the overfitting signature without GPUs or downloads. The pattern is
 * regular enough that a working transformer's loss visibly drops well below the bigram
 * baseline — a meaningful, checkable signal that attention is doing something.
 */
export const TOY_CORPUS: string =
  // A small, looping "song" with predictable structure: easy to overfit, hard to do at
  // chance. Newlines included so the model must learn line structure too.
  [
    "to be or not to be that is the question",
    "whether tis nobler in the mind to suffer",
    "the slings and arrows of outrageous fortune",
    "or to take arms against a sea of troubles",
    "and by opposing end them to die to sleep",
    "no more and by a sleep to say we end",
    "the heartache and the thousand natural shocks",
    "that flesh is heir to tis a consummation",
  ]
    .join("\n")
    .repeat(6) + "\n"; // repeat to give the model enough tokens to chew on

export interface CharDataset {
  vocab: string[]; // index -> char
  stoi: Map<string, number>; // char -> index
  vocabSize: number;
  encode: (s: string) => number[];
  decode: (ids: number[]) => string;
  trainIds: Int32Array;
  valIds: Int32Array;
  /** Sample a batch of (x, y) where y is x shifted by one (next-char prediction).
   *  Returns flat Int32Arrays of length batchSize*blockSize plus the dims, so callers
   *  can feed them straight into Embedding without reshaping bookkeeping here. */
  getBatch: (
    split: "train" | "val",
    blockSize: number,
    batchSize: number,
    rng: Rng,
  ) => { x: Int32Array; y: Int32Array; batchSize: number; blockSize: number };
}

/**
 * Build a char-level dataset from text. Vocab is the sorted set of unique chars so the
 * id mapping is DETERMINISTIC (independent of insertion order) — critical for
 * reproducibility across runs. 90/10 train/val split by position (LM val must be a
 * held-out contiguous tail, not random tokens).
 */
export function charDataset(text: string = TOY_CORPUS): CharDataset {
  const vocab = Array.from(new Set(text.split(""))).sort();
  const stoi = new Map<string, number>();
  vocab.forEach((ch, i) => stoi.set(ch, i));
  const encode = (s: string) => Array.from(s).map((ch) => {
    const id = stoi.get(ch);
    if (id === undefined) throw new Error(`charDataset.encode: char ${JSON.stringify(ch)} not in vocab`);
    return id;
  });
  const decode = (ids: number[]) => ids.map((i) => vocab[i]).join("");
  const all = Int32Array.from(encode(text));
  const nTrain = Math.floor(all.length * 0.9);
  const trainIds = all.slice(0, nTrain);
  const valIds = all.slice(nTrain);

  const getBatch = (split: "train" | "val", blockSize: number, batchSize: number, rng: Rng) => {
    const src = split === "train" ? trainIds : valIds;
    if (src.length <= blockSize + 1)
      throw new Error(`getBatch: split '${split}' too short (${src.length}) for blockSize ${blockSize}`);
    const x = new Int32Array(batchSize * blockSize);
    const y = new Int32Array(batchSize * blockSize);
    for (let b = 0; b < batchSize; b++) {
      // random window start; y is x shifted by one => next-char targets.
      const start = Math.floor(rng() * (src.length - blockSize - 1));
      for (let t = 0; t < blockSize; t++) {
        x[b * blockSize + t] = src[start + t];
        y[b * blockSize + t] = src[start + t + 1];
      }
    }
    return { x, y, batchSize, blockSize };
  };

  return { vocab, stoi, vocabSize: vocab.length, encode, decode, trainIds, valIds, getBatch };
}
