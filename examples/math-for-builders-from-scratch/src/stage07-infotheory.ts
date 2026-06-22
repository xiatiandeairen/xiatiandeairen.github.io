// stage07-infotheory.ts — Information theory: entropy, Huffman coding, cross-entropy / KL,
// and the pigeonhole impossibility of universal compression.
//
// WHY this is stage 07: every byte a builder ships is bounded by Shannon's theorem.
//   "How small can this log/JSON/model-weight get?" is the entropy of its source, full stop.
//   "Why is my cross-entropy loss the right objective?" is because minimizing it minimizes
//   the extra bits your model wastes vs the true distribution (= KL divergence). And
//   "why can't I just compress everything once more?" is the pigeonhole principle. This
//   stage computes all four from counts so the numbers are mechanical, not asserted.
//
// HONEST-NUMBER NOTE: entropy and the Huffman ratio below are measured on a REAL fixed
//   string embedded here (deterministic, no RNG needed for the text path — char counts are
//   exact). The Huffman average code length is the true weighted mean over that string, and
//   the compression ratio is bits-out / bits-in counted bit-by-bit, not estimated. The KL /
//   cross-entropy demo uses a seeded synthetic sample only to show the empirical→true gap;
//   absolute KL magnitude is toy, the SIGN facts (KL>=0, KL asymmetric) are exact and general.
//
// CONTRACT: reuse core/stats.js::histogram for the seeded-sample frequencies; core/plot.js
//   for the readable bars. The text-frequency path is a plain count map (a histogram over a
//   continuous range is the wrong tool for discrete symbols — see countChars below).

import { mulberry32, type Rng } from "./core/rng.js";
import { histogram } from "./core/stats.js";
import { asciiBar } from "./core/plot.js";

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

/**
 * Shannon entropy in BITS of a discrete distribution given as a probability vector.
 * H = -Σ p·log2(p), measured in bits because log base 2 answers "how many yes/no
 *   questions / binary digits per symbol".
 * INVARIANT: probs should sum to ~1; we do NOT renormalize (caller's job) so a buggy
 *   distribution surfaces as a wrong number instead of being silently "fixed".
 * WHY skip p==0 terms: 0·log0 is the limit 0, but log2(0) = -Infinity in IEEE float, so
 *   evaluating it would poison the sum with NaN. Skipping is the mathematically correct
 *   convention, not a hack.
 */
function entropyBits(probs: readonly number[]): number {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log2(p);
  return h;
}

/**
 * Same entropy but in NATS (natural log). Exists ONLY to demo the bits-vs-nats failure
 * mode: H_nats = H_bits · ln(2) ≈ 0.693·H_bits. Mixing the two is the single most common
 * unit bug in info-theory code — a "loss of 0.69" is 1 bit, not 0.69 bits.
 */
function entropyNats(probs: readonly number[]): number {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log(p);
  return h;
}

/**
 * Exact symbol counts of a string. WHY a Map, not core/stats histogram: histogram bins a
 * CONTINUOUS range into equal-width buckets; characters are discrete labels with no order
 * or distance, so binning them is meaningless. Reaching for histogram here would be the
 * "use the only tool I imported" trap. (We DO use histogram later where it fits: binning a
 * continuous seeded sample.)
 */
function countChars(text: string): Map<string, number> {
  const counts = new Map<string, number>();
  for (const ch of text) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  return counts;
}

// ---------------------------------------------------------------------------
// Huffman coding (built from scratch)
// ---------------------------------------------------------------------------

// A Huffman tree node. Leaves carry a symbol; internal nodes only carry weight.
// WHY a discriminated shape via optional `symbol`: a leaf has a symbol and no children,
//   an internal node has two children and no symbol. We keep it as one type for the simple
//   priority-queue merge loop; the `symbol === undefined` test is the leaf/internal switch.
interface HuffNode {
  weight: number;
  symbol?: string;
  left?: HuffNode;
  right?: HuffNode;
}

/**
 * Build a Huffman tree from symbol→count. Classic greedy algorithm: repeatedly merge the
 * two LOWEST-weight nodes; rare symbols sink deep (long codes), frequent symbols stay
 * shallow (short codes) — that is exactly what makes the average length approach entropy.
 *
 * INVARIANT: needs >=1 symbol. With exactly ONE distinct symbol there is no pair to merge,
 *   so we special-case it to a single leaf and assign it the 1-bit code "0" in buildCodes
 *   (you cannot encode a symbol in 0 bits in a prefix code — see FAILURE MODE there).
 *
 * NOTE: this uses an O(n^2) linear scan for the two minima instead of a real binary heap.
 *   For an alphabet of a few hundred symbols this is irrelevant and the code stays readable;
 *   a heap matters only for huge alphabets, which a teaching stage will never hit.
 */
function buildHuffmanTree(counts: Map<string, number>): HuffNode {
  if (counts.size === 0) throw new Error("[stage07] huffman: empty alphabet");
  const nodes: HuffNode[] = [...counts].map(([symbol, weight]) => ({ symbol, weight }));
  // Tie-break by weight only; the merge order among equal weights is deterministic given
  // the insertion order above, so the resulting codes are reproducible run-to-run.
  while (nodes.length > 1) {
    nodes.sort((a, b) => a.weight - b.weight);
    const left = nodes.shift()!;
    const right = nodes.shift()!;
    nodes.push({ weight: left.weight + right.weight, left, right });
  }
  return nodes[0];
}

/**
 * Walk the tree to produce symbol→bitstring. Left edge = "0", right edge = "1".
 * INVARIANT (prefix property): codes live only at leaves, so no code is a prefix of
 *   another — that is what lets a decoder split a bitstream with no separators.
 * FAILURE MODE handled: a single-leaf tree (one distinct symbol) would otherwise produce
 *   the empty code "", which is unencodable; we floor every leaf to at least "0".
 */
function buildCodes(root: HuffNode): Map<string, string> {
  const codes = new Map<string, string>();
  const walk = (node: HuffNode, prefix: string): void => {
    if (node.symbol !== undefined) {
      codes.set(node.symbol, prefix === "" ? "0" : prefix);
      return;
    }
    walk(node.left!, prefix + "0");
    walk(node.right!, prefix + "1");
  };
  walk(root, "");
  return codes;
}

/** Total encoded length in BITS = Σ count(sym)·len(code(sym)). The honest "bits out". */
function encodedBits(counts: Map<string, number>, codes: Map<string, string>): number {
  let bits = 0;
  for (const [sym, n] of counts) bits += n * codes.get(sym)!.length;
  return bits;
}

// ---------------------------------------------------------------------------
// Cross-entropy and KL divergence
// ---------------------------------------------------------------------------

/**
 * Cross-entropy H(p, q) = -Σ p·log2(q): the average bits per symbol you pay if the data
 * really follows p but you encode with a code optimal for q. This is the quantity a model
 * minimizes during training (p = data, q = model).
 * FAILURE MODE (the lesson): if q(x)=0 for some x with p(x)>0, log2(0) = -Infinity and the
 *   cross-entropy is +Infinity — you assigned zero probability to something that happened,
 *   so no finite code length can describe it. We return Infinity honestly rather than
 *   guarding it away; callers must add smoothing (never assign zero probability).
 */
function crossEntropyBits(p: readonly number[], q: readonly number[]): number {
  let ce = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      if (q[i] <= 0) return Infinity; // unobservable-under-q event actually occurred
      ce -= p[i] * Math.log2(q[i]);
    }
  }
  return ce;
}

/**
 * KL divergence D(p‖q) = H(p,q) − H(p) = Σ p·log2(p/q): the EXTRA bits wasted by coding
 * p's data with q's code, over the optimal H(p).
 * Two facts this function demonstrates numerically in main():
 *   - D(p‖q) >= 0 always (Gibbs' inequality), = 0 iff p == q.
 *   - D(p‖q) != D(q‖p) in general — KL is NOT a distance, it is asymmetric.
 */
function klBits(p: readonly number[], q: readonly number[]): number {
  return crossEntropyBits(p, q) - entropyBits(p);
}

// ---------------------------------------------------------------------------
// Pigeonhole: no lossless compressor shrinks every input
// ---------------------------------------------------------------------------

/**
 * Count, by exhaustive enumeration over all 2^n bitstrings of length n, how many a lossless
 * codec could possibly map to a SHORTER string. There are only (2^n − 1) strings shorter
 * than n bits (lengths 0..n-1), but 2^n inputs of length n; an injective (decodable) map
 * cannot fit 2^n pigeons into 2^n − 1 holes. So at least one length-n input must map to
 * length >= n. "Compress everything" is therefore impossible, not merely unsolved.
 * Returns { inputs, shorterSlots }: shorterSlots < inputs is the whole proof.
 */
function pigeonholeCompression(n: number): { inputs: number; shorterSlots: number } {
  const inputs = 2 ** n; // all length-n strings
  let shorterSlots = 0;
  for (let len = 0; len < n; len++) shorterSlots += 2 ** len; // strings of length < n
  return { inputs, shorterSlots };
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// A real, fixed English-ish string. Its char distribution is skewed (lots of spaces / 'e'),
// so Huffman should beat a flat fixed-width code AND land just above the entropy floor.
const SAMPLE_TEXT =
  "the quick brown fox jumps over the lazy dog. " +
  "information is the resolution of uncertainty, measured in bits. " +
  "entropy bounds compression: you cannot beat the source, only approach it.";

/** Sample an integer symbol in [0, k) from an explicit probability vector via the CDF. */
function sampleSymbol(rng: Rng, probs: readonly number[]): number {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u < acc) return i;
  }
  return probs.length - 1; // float guard: u just under 1 falls into the last bucket
}

function main(): void {
  console.log("=== Stage 07 · 信息论：熵、霍夫曼编码与不可压缩性 ===\n");

  // --- 1. Entropy of a real string -----------------------------------------
  const counts = countChars(SAMPLE_TEXT);
  const total = SAMPLE_TEXT.length;
  const probs = [...counts.values()].map((c) => c / total);
  const hBits = entropyBits(probs);
  const alphabet = counts.size;
  const fixedWidthBits = Math.ceil(Math.log2(alphabet)); // naive fixed-length code per char

  console.log("① 文本熵（按字符频率，真实计数）");
  console.log(`  文本长度        = ${total} chars，去重符号数 = ${alphabet}`);
  console.log(`  经验熵 H        = ${hBits.toFixed(4)} bits/char`);
  console.log(`  定宽编码需要    = ${fixedWidthBits} bits/char (ceil(log2 ${alphabet}))`);
  console.log(`  熵给出的下界比定宽省 ${(fixedWidthBits - hBits).toFixed(2)} bits/char\n`);

  // --- 2. Huffman code: measured ratio vs entropy bound --------------------
  const tree = buildHuffmanTree(counts);
  const codes = buildCodes(tree);
  const outBits = encodedBits(counts, codes);
  const avgCodeLen = outBits / total; // true weighted average, bits/char
  const inBitsFixed = total * fixedWidthBits; // baseline: fixed-width encoding of same text
  const inBitsAscii = total * 8; // baseline: raw 8-bit-per-char
  const ratioVsFixed = inBitsFixed / outBits;
  const ratioVsAscii = inBitsAscii / outBits;

  console.log("② 霍夫曼编码（从零构建，逐 bit 计数）");
  console.log(`  Huffman 平均码长 = ${avgCodeLen.toFixed(4)} bits/char`);
  console.log(`  Shannon 下界 H   = ${hBits.toFixed(4)} bits/char`);
  // Shannon source-coding theorem: H <= avgCodeLen < H + 1. We assert the lower half here.
  const beatsBound = avgCodeLen < hBits - 1e-9;
  console.log(
    `  下界检查         = ${beatsBound ? "✗ 违反！(码长 < 熵, 不可能)" : "✓ 码长 ≥ 熵 (符合香农定理)"}`,
  );
  console.log(`  冗余 (码长 − H)  = ${(avgCodeLen - hBits).toFixed(4)} bits/char (定理保证 < 1)`);
  console.log(`  压缩比 vs 定宽${fixedWidthBits}bit = ${ratioVsFixed.toFixed(3)}×  (${inBitsFixed} → ${outBits} bits)`);
  console.log(`  压缩比 vs ASCII8 = ${ratioVsAscii.toFixed(3)}×  (${inBitsAscii} → ${outBits} bits)`);

  // Show the codes for the most frequent symbols — frequent => short, the whole point.
  const topSyms = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  console.log("  高频符号码长（频率越高码越短）:");
  console.log(
    asciiBar(
      topSyms.map(([s]) => JSON.stringify(s)),
      topSyms.map(([s]) => codes.get(s)!.length),
      24,
    ),
  );
  console.log("");

  // --- 3. Cross-entropy / KL: asymmetry and non-negativity -----------------
  // True source p over 4 symbols, and a wrong model q. Both are valid distributions.
  const p = [0.5, 0.25, 0.15, 0.1];
  const q = [0.25, 0.25, 0.25, 0.25]; // a model that wrongly assumes uniform
  const dPQ = klBits(p, q);
  const dQP = klBits(q, p);
  const dPP = klBits(p, p);

  console.log("③ 交叉熵 / KL 散度");
  console.log(`  H(p)        = ${entropyBits(p).toFixed(4)} bits  (真实分布的最优码长)`);
  console.log(`  H(p, q)     = ${crossEntropyBits(p, q).toFixed(4)} bits  (用 q 的码编码 p 的数据)`);
  console.log(`  D(p‖q)      = ${dPQ.toFixed(4)} bits  (= H(p,q) − H(p), 浪费的额外 bit)`);
  console.log(`  D(q‖p)      = ${dQP.toFixed(4)} bits`);
  console.log(`  非对称性    = D(p‖q) ${Math.abs(dPQ - dQP) > 1e-9 ? "≠" : "="} D(q‖p)  → KL 不是距离`);
  console.log(`  自身散度    = D(p‖p) = ${dPP.toFixed(6)} bits  (= 0, 当且仅当分布相同)`);
  const nonNeg = dPQ >= -1e-12 && dQP >= -1e-12 && dPP >= -1e-12;
  console.log(`  非负性检查  = ${nonNeg ? "✓ 所有 KL ≥ 0 (Gibbs 不等式)" : "✗ 出现负 KL!"}\n`);

  // --- 3b. Empirical KL: a seeded sample drawn from p, scored against p and q ----
  // WHY a seeded sample here: to show the EMPIRICAL cross-entropy of real draws converges
  //   toward the true H(p,·); this is the training-loss picture. histogram fits here because
  //   we map symbols to a continuous index range [0, k) and bin them.
  const rng = mulberry32(7);
  const N = 20000;
  const draws = Array.from({ length: N }, () => sampleSymbol(rng, p));
  const { counts: binCounts } = histogram(draws, p.length);
  const empP = binCounts.map((c) => c / N);
  console.log("   实证（种子样本 N=20000，从 p 抽样）");
  console.log(`   经验分布 p̂   = [${empP.map((x) => x.toFixed(3)).join(", ")}]`);
  console.log(`   真实 p       = [${p.map((x) => x.toFixed(3)).join(", ")}]`);
  console.log(`   经验交叉熵 H(p̂, q) = ${crossEntropyBits(empP, q).toFixed(4)} bits (≈ 理论 ${crossEntropyBits(p, q).toFixed(4)})`);
  console.log("   合成绝对值偏乐观；可迁移的是 p̂→p 的收敛与 KL 的符号性质。\n");

  // --- 4. Pigeonhole impossibility -----------------------------------------
  const n = 8;
  const { inputs, shorterSlots } = pigeonholeCompression(n);
  console.log("④ 鸽巢论证：没有无损压缩器能压缩所有输入");
  console.log(`  长度 ${n} 的输入共 = ${inputs} 个 (2^${n})`);
  console.log(`  更短的串总共只有 = ${shorterSlots} 个 (所有长度 0..${n - 1} 之和 = 2^${n} − 1)`);
  console.log(
    `  鸽巢: ${inputs} 个输入塞进 ${shorterSlots} 个"更短"格子 → 至少 ${inputs - shorterSlots} 个必须不变短或变长。`,
  );
  console.log(`  推论: 任何"对某些输入变短"的无损编码，必对另一些输入变长。免费午餐不存在。\n`);

  // --- 5. Failure mode: bits vs nats ---------------------------------------
  console.log("⑤ 失败模式：log 底搞错 (bits vs nats)");
  const hb = entropyBits(p);
  const hn = entropyNats(p);
  console.log(`  H(p) 用 log2  = ${hb.toFixed(4)} bits`);
  console.log(`  H(p) 用 ln    = ${hn.toFixed(4)} nats`);
  console.log(`  比值 bits/nats = ${(hb / hn).toFixed(4)} = 1/ln2 = ${(1 / Math.LN2).toFixed(4)}`);
  console.log(
    `  陷阱: 把 ${hn.toFixed(2)} nats 当成 bits 读 → 低估熵 ${((1 - hn / hb) * 100).toFixed(0)}%。` +
      ` 训练 loss 写 ${hn.toFixed(2)} 看着比 ${hb.toFixed(2)} "好"，其实同一个量、不同单位。`,
  );
}

main();
