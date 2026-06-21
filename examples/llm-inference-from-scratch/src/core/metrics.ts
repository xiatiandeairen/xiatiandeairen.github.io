// core/metrics.ts — the honest scorecard the whole book is judged against.
//
// The promise of this book is "real tok/s". That promise lives or dies here. The
// rules, inherited from the vector-search book's honesty convention:
//
//   1. Anything that is a *duration* is wall-clocked with performance.now(). Never
//      estimated, never hard-coded. tok/s, TTFT, inter-token latency, speedup.
//   2. Anything that is *memory* is ESTIMATED from the float64 payload size and
//      labeled `(est.)` — we do NOT read process.memoryUsage().rss, because RSS on
//      a GC'd VM is noisy, allocator-dependent, and would lie about the *algorithm's*
//      footprint. The KV-cache byte math, by contrast, is exact arithmetic on the
//      model config and is the number a real engine's capacity planning uses.
//   3. Equivalence claims ("the optimization didn't change the output") are proven
//      with perplexity / logit-drift, never asserted.
//
// Toy-data caveat that every stage must echo: the synthetic model is small and the
// kernels are unoptimized float64 loops, so ABSOLUTE tok/s is pessimistic. What
// transfers to a real engine is the RELATIVE story — the speedup ratios and the
// shape of the prefill-vs-decode curve.

import type { ModelConfig } from "./model.js";

// timeIt: wall-clock a thunk, in milliseconds.
//
// Warmup convention (this is the easy thing to get wrong): the FIRST call to any
// JS hot path pays JIT compilation + cold-cache costs that have nothing to do with
// steady-state throughput. So callers MUST warm up before the measured run:
//
//     timeIt(() => run());          // throwaway: triggers JIT, warms caches
//     const ms = timeIt(() => run()); // this one counts
//
// We deliberately do NOT bake warmup into timeIt: some measurements (TTFT, the
// very first token) are explicitly about the cold path and warming them up would
// be a lie. Forcing the caller to decide keeps the honesty visible at the call site.
export function timeIt(fn: () => void): number {
  const t0 = performance.now();
  fn();
  return performance.now() - t0;
}

// tok/s. The headline number. nTok is tokens *produced* (decode steps), ms is the
// wall time to produce them. Guard against a zero/negative interval (can happen if
// a run is so fast it underflows the timer's resolution) by returning Infinity,
// which is a louder signal than a divide-by-zero NaN that "your workload is too
// small to measure — make it bigger".
export function tokensPerSecond(nTok: number, ms: number): number {
  if (ms <= 0) return Infinity;
  return (nTok / ms) * 1000;
}

// TTFT — time to first token. Just a labeled pass-through of a measured duration,
// but naming it makes the two-phase structure (prefill -> first token, then decode)
// explicit in stage code. TTFT is dominated by PREFILL: the whole prompt is run
// through the model before any token comes out. It grows with prompt length.
export function ttft(ms: number): number {
  return ms;
}

// Inter-token latency stats over the per-step decode times. The DECODE phase is
// the other half of the inference dichotomy: after prefill, each new token is one
// forward step over a single position (reusing the KV cache). Its latency should be
// roughly flat (and much smaller than TTFT) — if mean rises steadily with step
// index, your "cache" is secretly re-reading history (the exact bug stage02 hunts).
export function interTokenLatency(msPerStep: number[]): {
  mean: number;
  p50: number;
  p95: number;
  min: number;
  max: number;
} {
  if (msPerStep.length === 0) return { mean: 0, p50: 0, p95: 0, min: 0, max: 0 };
  const sorted = [...msPerStep].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const pct = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
  return {
    mean: sum / sorted.length,
    p50: pct(50),
    p95: pct(95),
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
}

// estimateKVBytes — the capacity-planning number, labeled (est.) when printed.
//
// This is EXACT arithmetic, not a measurement: a KV cache stores, per layer, the
// key and value for every past token of every sequence. Shape per layer:
//   2 (K and V) * nKVHeads * dHead * seqLen, times nLayers, times nSeqs.
// We assume float64 (8 bytes) to match this toy engine's storage; the formula and
// the 8 is the only place real-vs-toy diverges (a real fp16 cache is 4x smaller).
// nKVHeads (not nHeads) is what makes GQA's memory win show up: with nKVHeads <
// nHeads the cache shrinks by exactly nHeads/nKVHeads. This number is why paged-KV
// (stage05) and GQA exist, so the formula is in core where every stage can cite it.
export function estimateKVBytes(cfg: ModelConfig, seqLen: number, nSeqs: number): number {
  const bytesPerFloat = 8; // float64 payload — see module header rule #2
  const elemsPerLayer = 2 * cfg.nKVHeads * cfg.dHead * seqLen;
  return elemsPerLayer * cfg.nLayers * nSeqs * bytesPerFloat;
}

// Human-readable bytes. Callers append the literal "(est.)" suffix per the honesty
// rule so a reader can never confuse an estimate for a measurement.
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let b = bytes / 1024;
  let i = 0;
  while (b >= 1024 && i < units.length - 1) {
    b /= 1024;
    i++;
  }
  return `${b.toFixed(2)} ${units[i]}`;
}

// speedup: baseline / optimized. >1 means faster. Reported as the primary result
// of every optimization stage because it is robust to the toy-data caveat: even if
// both absolute numbers are pessimistic, their RATIO transfers to a real engine.
export function speedup(baselineMs: number, optimizedMs: number): number {
  if (optimizedMs <= 0) return Infinity;
  return baselineMs / optimizedMs;
}

// perplexity of a sequence under a stream of next-token logit rows.
//
// Equivalence judge for the "fast but not worse" claim (speculative decoding,
// quantization). PPL = exp(mean negative log-likelihood of the actual next tokens).
// logits[t] is the unnormalized distribution predicting targetIds[t]. Lower is
// "more confident / better"; for our purposes the ABSOLUTE value is meaningless
// (the model is untrained — expect ~vocabSize), but a near-identical PPL between
// an optimized run and the baseline run is hard proof the optimization preserved
// the distribution. Computed via log-sum-exp for numerical stability.
//
// Failure mode it catches: a quantization scheme that "runs" but quietly skews the
// output distribution shows up here as a PPL gap even when the argmax token happens
// to match.
export function perplexity(logits: number[][], targetIds: number[]): number {
  if (logits.length !== targetIds.length) {
    throw new Error(`perplexity: ${logits.length} logit rows != ${targetIds.length} targets`);
  }
  let nll = 0;
  for (let t = 0; t < logits.length; t++) {
    const row = logits[t];
    let max = -Infinity;
    for (const v of row) if (v > max) max = v;
    let sumExp = 0;
    for (const v of row) sumExp += Math.exp(v - max);
    const logSumExp = max + Math.log(sumExp);
    const logProb = row[targetIds[t]] - logSumExp; // log softmax at the target
    nll += -logProb;
  }
  return Math.exp(nll / logits.length);
}

// maxLogitDrift: L∞ (max absolute) difference between two logit vectors.
//
// The on-the-wire correctness check for any kernel rewrite or quantization. Two
// logit vectors that should be identical (same model, same input, different code
// path) must drift by ~0 in float64; an int8 path will drift by some bounded
// amount that this number quantifies. Reporting L∞ (not mean) is deliberate: a
// single catastrophically-wrong logit (e.g. one overflowed dot product) is exactly
// the failure that a mean would hide and that flips an argmax.
export function maxLogitDrift(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error(`maxLogitDrift: length ${a.length} != ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

// argmax over a logit row — greedy next-token. Tiny but shared so every stage's
// "did the token change?" comparison uses identical tie-breaking (first-max wins,
// deterministic).
export function argmax(row: number[] | Float64Array): number {
  let best = 0;
  let bestV = -Infinity;
  for (let i = 0; i < row.length; i++) {
    if (row[i] > bestV) {
      bestV = row[i];
      best = i;
    }
  }
  return best;
}
