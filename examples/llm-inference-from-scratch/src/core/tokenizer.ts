// core/tokenizer.ts — a byte-level, zero-file, fully deterministic tokenizer.
//
// Why byte-level and not BPE: a real BPE tokenizer needs a trained merge table
// (a downloaded file, a vocab of 50k+). That would drag a network/asset dependency
// into a book that promises "offline, reproducible, from scratch". A UTF-8 byte
// tokenizer needs nothing: vocab is exactly the 256 byte values, encode is
// "give me the bytes", decode is "give me the string back". Every prompt on Earth
// is encodable, identically, on any machine, forever.
//
// The cost — sequences are longer than BPE would produce (one token per byte, so
// multi-byte UTF-8 chars span several tokens) — is irrelevant to this book: we are
// measuring the *engine* (cache, batching, paging), and a longer sequence just
// means more decode steps to time, which is if anything more honest.
//
// Invariant the whole book leans on: decode(encode(s)) === s for any string s.
// This round-trip is what lets a stage print human-readable (if gibberish, since
// the model is untrained) output and trust the token bookkeeping underneath it.

export const VOCAB_SIZE = 256;

const ENC = new TextEncoder();
const DEC = new TextDecoder();

// encode: string -> token ids (one per UTF-8 byte). TextEncoder is deterministic
// and built in. ids are in [0, 255], matching the toy model's vocabSize=256, so a
// logit row indexes directly by token id with no embedding-table lookup table file.
export function encode(s: string): number[] {
  return Array.from(ENC.encode(s));
}

// decode: token ids -> string. We clamp/mask to a byte because the sampler can,
// by construction of a 256-wide vocab, only ever emit a valid byte — but an
// untrained model will emit *arbitrary* bytes, so the decoded string is often
// invalid UTF-8. TextDecoder's default (fatal:false) replaces bad sequences with
// U+FFFD rather than throwing, which is the right call: a garbled byte is a
// model-quality artifact, not an engine bug, and must not crash the pipeline.
export function decode(ids: number[]): string {
  const bytes = new Uint8Array(ids.length);
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    // Loud-ish guard: an out-of-range id means the sampler/model is misconfigured
    // (e.g. vocabSize mismatch), which we would rather surface than silently &0xff.
    if (id < 0 || id > 255 || !Number.isInteger(id)) {
      throw new Error(`decode: token id ${id} out of byte range — vocab mismatch?`);
    }
    bytes[i] = id;
  }
  return DEC.decode(bytes);
}

// A fixed prompt set used across stages. ≥3 prompts of *different lengths* on
// purpose: a single prompt risks N=1 luck (e.g. a length that happens to align
// with a page boundary in stage05, or a TTFT that happens to look flat). Reporting
// metrics across this spread is the difference between "it worked once" and "it
// works". Kept ASCII so byte-count == char-count, making the prefill-length math
// in the stages easy to eyeball.
export const PROMPTS: readonly string[] = [
  "The",
  "Inference engines must be measured, not guessed.",
  "A long prompt stresses the prefill stage: every token here is processed in " +
    "one batched forward pass before a single output token is produced, which is " +
    "why time-to-first-token grows with prompt length while inter-token latency " +
    "does not.",
] as const;
