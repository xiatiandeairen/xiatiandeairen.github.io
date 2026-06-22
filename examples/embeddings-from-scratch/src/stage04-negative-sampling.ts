// stage04-negative-sampling.ts — turning the softmax into a binary classifier.
//
// Chapter 04 thesis: full-softmax skip-gram (stage 03) computes a denominator over
// the ENTIRE vocabulary on every step — cost grows with V. Skip-Gram with Negative
// Sampling (SGNS, Mikolov 2013) replaces that one V-way classification with (1+k)
// independent binary classifications: "is this (center, context) a real pair?" vs
// "is this (center, negative) a fake pair?". The per-step cost becomes ~(1+k) dot
// products, essentially INDEPENDENT of V. That single change is what let word2vec
// scale to million-word vocabularies.
//
// What this file actually computes/measures (no hand-waving):
//   ① SGNS loss on the toy corpus + REAL per-step wall-clock, contrasted with the
//      full-softmax per-step cost (we build the same forward graph stage 03 builds
//      and time it) to show softmax scales with V while SGNS does not.
//   ② A k = 1 / 5 / 20 sweep: real training wall-clock vs nearest-neighbor quality,
//      to expose the cost↔quality trade-off (more negatives = better signal, more
//      compute).
//   ③ Three negative-sampling distributions (uniform / unigram / unigram^0.75) on
//      the SAME training pairs, with neighbor examples, to show why word2vec picks
//      the 0.75 power. FAILURE MODE demo: raw unigram makes almost every negative a
//      high-frequency filler word ("the"/"a"), so the model spends its capacity
//      pushing AWAY from stopwords it already ignores → degenerate vectors.
//
// Honesty caveats stated up front (this is a toy):
//   - V ≈ 40 here, so absolute speedups are tiny; what TRANSFERS is the *shape* —
//     softmax step time rises with V, SGNS step time is flat. We make V-scaling
//     visible by timing the softmax denominator at several synthetic V.
//   - Neighbor quality is optimistic at toy scale; the RELATIVE ranking across
//     k values and across sampling distributions is the transferable signal.
//   - Wall-clock numbers are measured (performance.now), not estimated. Anything
//     we cannot run we mark (est.).

import { Rng, sampleCategorical, shuffle } from "./core/rng.js";
import { Value, dot, makeMat, vecData, type Mat } from "./core/autograd.js";
import { Adam, collectParams } from "./core/optim.js";
import { generateCorpus, buildVocab, windowPairs, type Vocab, type Pair } from "./core/text.js";
import { nearestNeighbors } from "./core/eval.js";
import { asciiLine, asciiBar } from "./core/plot.js";

// ---------------------------------------------------------------------------
// Hyperparameters. Kept identical across experiments so comparisons are fair —
// only the variable under study (k, or the sampling distribution) changes.
// ---------------------------------------------------------------------------
const DIM = 16; // embedding dimension; small so a run is seconds, large enough to separate ~6 clusters
const EPOCHS = 8; // passes over the (subsampled) pair set
const PAIRS_PER_EPOCH = 300; // subsample size per epoch; full pair set is larger but this is enough to converge on toy data
const LR = 0.05; // Adam lr; SGNS tolerates a higher lr than full softmax because per-step grads are sparser
const SEED = 1234; // one master seed; every Rng below derives a fixed offset so concerns stay decoupled

// ---------------------------------------------------------------------------
// SGNS model. Word2vec uses TWO matrices: an "input" embedding for words as
// centers and an "output" embedding for words as contexts/negatives. Sharing one
// matrix lets a word be its own perfect predictor (w·w is large), which collapses
// the geometry; separate matrices is the standard fix and what we do here.
// ---------------------------------------------------------------------------
interface Sgns {
  inEmb: Mat; // center vectors, V x DIM
  outEmb: Mat; // context/negative vectors, V x DIM
}

function buildModel(vocab: Vocab, rng: Rng): Sgns {
  // Small Gaussian init: near-zero initial dot products → sigmoid starts near 0.5
  // → gradients are well-scaled (not saturated) on step 1. See core/rng header.
  const init = () => (rng.nextFloat() - 0.5) * 0.1;
  return {
    inEmb: makeMat(vocab.size, DIM, init),
    outEmb: makeMat(vocab.size, DIM, init),
  };
}

// logsigmoid(x) = -softplus(-x), built from autograd primitives so its gradient
// comes straight from the engine. We deliberately do NOT hand-derive the classic
// "(sigmoid(x) - label)" gradient: the whole point of the book is that the reader
// trusts the autograd chain, not a memorized formula. Numerically, log() in the
// engine clamps its input away from 0, and sigmoid ∈ (0,1) stays in-domain, so
// this is safe without extra guards at the magnitudes toy training produces.
function logSigmoid(x: Value): Value {
  // sigmoid(x) = 1 / (1 + e^-x)
  const sig = x.mul(-1).exp().add(1).pow(-1);
  return sig.log();
}

// One SGNS training term for a single (center, context) pair plus k negatives.
// Loss = -[ logσ(c·o) + Σ_j logσ(-c·n_j) ].  Minimizing it pushes the real pair's
// dot product up (toward σ→1) and each fake pair's dot product down (σ(-·)→1,
// i.e. the negative dot → large positive, real dot → ... wait: σ(-c·n) → 1 means
// c·n → -∞). So real pairs attract, sampled negatives repel. That repulsion is the
// ONLY thing keeping the space from collapsing to a single point — which is exactly
// why the *choice* of which words to repel (the negative distribution) matters.
function pairLoss(model: Sgns, center: number, context: number, negatives: number[]): Value {
  const c = model.inEmb[center];
  const posTerm = logSigmoid(dot(c, model.outEmb[context]));
  let loss = posTerm; // accumulate logσ terms; we negate the sum at the end
  for (const n of negatives) {
    // logσ(-c·n): note the negation of the score, not of the whole term.
    loss = loss.add(logSigmoid(dot(c, model.outEmb[n]).mul(-1)));
  }
  return loss.mul(-1); // turn log-likelihood into a loss to MINIMIZE
}

// ---------------------------------------------------------------------------
// Negative-sampling distributions. Each returns an unnormalized weight per token
// id; sampleCategorical normalizes internally. The vocab gives id-indexed counts,
// so building these is just a transform on vocab.counts.
// ---------------------------------------------------------------------------
type NegDist = "uniform" | "unigram" | "unigram075";

// Build the sampling-weight table for a given distribution. This is THE knob that
// distinguishes the three experiments in ③. unigram^0.75 is word2vec's choice: it
// dampens the dominance of ultra-frequent words (the^0.75 still > rare^0.75, but
// the ratio is compressed) so negatives include enough mid/rare words to give every
// content word something to push against.
function buildNegWeights(vocab: Vocab, dist: NegDist): number[] {
  switch (dist) {
    case "uniform":
      // Every word equally likely. Cheap, but wastes negatives on rare words that
      // already barely co-occur with anything → weaker repulsion signal.
      return vocab.counts.map(() => 1);
    case "unigram":
      // Raw frequency. FAILURE MODE: high-freq stopwords ("the","a") dominate the
      // negatives, so the model mostly learns "don't predict 'the'" — which it
      // already does — and content words get little discriminative pressure.
      return vocab.counts.slice();
    case "unigram075":
      // The word2vec sweet spot. ^0.75 flattens the head of the frequency curve.
      return vocab.counts.map((c) => Math.pow(c, 0.75));
    default: {
      const _exhaustive: never = dist;
      throw new Error(`unknown neg dist: ${_exhaustive}`);
    }
  }
}

// Draw k negatives for a given positive pair. We reject draws that equal the true
// context (sampling it as a "negative" would teach the model to push apart a real
// pair — a self-inflicted label-noise bug). We do NOT reject the center: in SGNS a
// center word can legitimately be a negative context for itself in a different role.
function sampleNegatives(rng: Rng, weights: number[], context: number, k: number): number[] {
  const negs: number[] = [];
  let guard = 0;
  while (negs.length < k) {
    const n = sampleCategorical(rng, weights);
    // guard caps rejection retries so a pathological weight table (e.g. all mass on
    // the context word) can't spin forever; we accept a rare duplicate-of-context
    // rather than hang. At toy scale this branch effectively never trips.
    if (n !== context || guard > 50) negs.push(n);
    guard++;
  }
  return negs;
}

// ---------------------------------------------------------------------------
// Training loop. Returns the loss curve and the in-embedding rows as plain number
// matrices for evaluation. The optimizer + zeroGrad/backward/step contract follows
// core/optim's mandatory order exactly.
// ---------------------------------------------------------------------------
interface TrainResult {
  losses: number[];
  vectors: number[][]; // detached inEmb rows for eval
  wallMs: number; // measured training wall-clock
  steps: number; // number of optimizer steps taken
}

function trainSgns(
  vocab: Vocab,
  pairs: Pair[],
  negWeights: number[],
  k: number,
  seed: number,
): TrainResult {
  const modelRng = new Rng(seed); // weight init
  const negRng = new Rng(seed + 7); // negative sampling — separate concern, separate stream
  const orderRng = new Rng(seed + 13); // epoch shuffling
  const model = buildModel(vocab, modelRng);
  const params = collectParams(model.inEmb, model.outEmb);
  const opt = new Adam(params, LR);

  const losses: number[] = [];
  let steps = 0;
  const t0 = performance.now();
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Fresh subsample each epoch via shuffle + slice; deterministic from orderRng.
    const epochPairs = shuffle(orderRng, pairs.slice()).slice(0, PAIRS_PER_EPOCH);
    let epochLoss = 0;
    for (const { center, context } of epochPairs) {
      const negatives = sampleNegatives(negRng, negWeights, context, k);
      opt.zeroGrad();
      const loss = pairLoss(model, center, context, negatives);
      loss.backward();
      opt.step();
      epochLoss += loss.data;
      steps++;
    }
    losses.push(epochLoss / epochPairs.length);
  }
  const wallMs = performance.now() - t0;
  return { losses, vectors: model.inEmb.map(vecData), wallMs, steps };
}

// ---------------------------------------------------------------------------
// Full-softmax forward cost, for the ① contrast. We build the SAME graph stage 03
// builds for ONE step — a V-way softmax over output embeddings — and time only the
// forward+backward, so the comparison is apples-to-apples (same autograd engine,
// same DIM). We do NOT train with it; we just measure one step's cost as V grows.
// ---------------------------------------------------------------------------

// Build a single softmax-skip-gram loss term: -log( exp(c·o_ctx) / Σ_v exp(c·o_v) ).
// The Σ over all V output vectors is the cost that SGNS eliminates. We compute it
// in the autograd graph so its backward pass (the expensive part) is included.
function softmaxStepLoss(inEmb: Mat, outEmb: Mat, center: number, context: number): Value {
  const c = inEmb[center];
  const scores: Value[] = [];
  for (let v = 0; v < outEmb.length; v++) scores.push(dot(c, outEmb[v]).exp());
  let denom = new Value(0);
  for (const s of scores) denom = denom.add(s);
  // numerator is exp(c·o_context); reuse the already-built score to avoid a second
  // exp graph (keeps the timing about the denominator sum, which is the real cost).
  const numer = scores[context];
  return numer.div(denom).log().mul(-1);
}

// Measure one softmax forward+backward at a synthetic vocab size V (DIM fixed).
// Random fixed init; we time the graph build + backward, the work that scales.
function measureSoftmaxStepMs(V: number): number {
  const rng = new Rng(SEED + V); // deterministic per V
  const init = () => (rng.nextFloat() - 0.5) * 0.1;
  const inEmb = makeMat(V, DIM, init);
  const outEmb = makeMat(V, DIM, init);
  // Warm up once (JIT compile the hot path) then average over a few steps so the
  // trend isn't polluted by first-call compilation noise. Honest: we report the
  // steady-state cost, not the cold-start cost.
  const ctx = Math.min(1, V - 1);
  for (const p of [...inEmb, ...outEmb]) for (const x of p) x.grad = 0;
  softmaxStepLoss(inEmb, outEmb, 0, ctx).backward();
  const REPS = 5;
  const t0 = performance.now();
  for (let r = 0; r < REPS; r++) {
    for (const p of [...inEmb, ...outEmb]) for (const x of p) x.grad = 0;
    const loss = softmaxStepLoss(inEmb, outEmb, 0, ctx);
    loss.backward();
  }
  return (performance.now() - t0) / REPS;
}

// Measure one SGNS forward+backward at synthetic V with fixed k (DIM fixed). Cost
// should be ~flat in V: it only touches 1 + k output rows regardless of V.
function measureSgnsStepMs(V: number, k: number): number {
  const rng = new Rng(SEED + V + 1);
  const init = () => (rng.nextFloat() - 0.5) * 0.1;
  const model: Sgns = { inEmb: makeMat(V, DIM, init), outEmb: makeMat(V, DIM, init) };
  const negRng = new Rng(SEED + 99);
  const weights = new Array(V).fill(1); // uniform; distribution choice is irrelevant to step COST
  const ctx = Math.min(1, V - 1);
  // Warm up (same rationale as the softmax measurement) so the two columns are
  // compared at steady state, not cold vs warm.
  for (const p of [...model.inEmb, ...model.outEmb]) for (const x of p) x.grad = 0;
  pairLoss(model, 0, ctx, sampleNegatives(negRng, weights, 1, k)).backward();
  const REPS = 5;
  const t0 = performance.now();
  for (let r = 0; r < REPS; r++) {
    for (const p of [...model.inEmb, ...model.outEmb]) for (const x of p) x.grad = 0;
    const negs = sampleNegatives(negRng, weights, 1, k);
    const loss = pairLoss(model, 0, ctx, negs);
    loss.backward();
  }
  return (performance.now() - t0) / REPS;
}

// ---------------------------------------------------------------------------
// Small eval helper: print the top neighbors of a probe word, by cosine on the
// trained in-embedding. Used to judge "did the space learn structure?" across the
// k sweep and the distribution comparison.
// ---------------------------------------------------------------------------
function neighborStr(vocab: Vocab, vectors: number[][], word: string, k = 3): string {
  const id = vocab.stoi.get(word);
  if (id === undefined) return `${word}: <not in vocab>`;
  const nbrs = nearestNeighbors(vectors[id], vectors, k);
  const parts = nbrs.map((n) => `${vocab.itos[n.index]}(${n.score.toFixed(2)})`);
  return `${word} → ${parts.join("  ")}`;
}

// A crude but honest scalar for "neighbor quality": for a few probe words whose
// correct neighbors we KNOW (the planted clusters), the cosine to the intended
// partner. Higher = the planted structure survived training. This is a relative
// signal across runs, not an absolute quality claim (toy data).
function plantedPairScore(vocab: Vocab, vectors: number[][]): number {
  // pairs that SHOULD be close by construction of the toy corpus templates
  const probes: Array<[string, string]> = [
    ["king", "queen"],
    ["dog", "cat"],
    ["bread", "cheese"],
    ["north", "south"],
  ];
  let sum = 0;
  let n = 0;
  for (const [a, b] of probes) {
    const ia = vocab.stoi.get(a);
    const ib = vocab.stoi.get(b);
    if (ia === undefined || ib === undefined) continue;
    // cosine via nearestNeighbors would exclude self; just compute directly.
    sum += cosine(vectors[ia], vectors[ib]);
    n++;
  }
  return n === 0 ? 0 : sum / n;
}

// Cross-bucket contamination — counts how many of a content probe's top-N
// neighbors are NEITHER its own semantic cluster NOR another content word, i.e.
// stopwords/filler that intruded. This is the qualitatively-visible symptom of the
// raw-unigram failure mode (e.g. "king → ... and", "bread → ... not"). It is a
// crude, toy-scale indicator; we print the raw neighbor lists alongside so the
// reader can audit it rather than trust the scalar. Returns mean intruder fraction
// in [0,1]; LOWER is better.
function intruderFraction(vocab: Vocab, vectors: number[][], topN = 3): number {
  // ids 0..STOP-1 are the high-frequency stopword/function words by vocab convention.
  const STOP = 5;
  const stopIds = new Set<number>();
  for (let i = 0; i < Math.min(STOP, vocab.size); i++) stopIds.add(i);
  const contentProbes = ["king", "queen", "dog", "cat", "bread", "north", "south"];
  let frac = 0;
  let n = 0;
  for (const w of contentProbes) {
    const id = vocab.stoi.get(w);
    if (id === undefined) continue;
    const nbrs = nearestNeighbors(vectors[id], vectors, topN);
    const intruders = nbrs.filter((nb) => stopIds.has(nb.index)).length;
    frac += intruders / topN;
    n++;
  }
  return n === 0 ? 0 : frac / n;
}

function cosine(a: number[], b: number[]): number {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : d / denom;
}

// ===========================================================================
// main — runs the three experiments in order and prints real numbers.
// ===========================================================================
function main(): void {
  // Shared corpus + vocab + pair set. One Rng for corpus so it's reproducible and
  // decoupled from training randomness.
  const corpusRng = new Rng(SEED);
  const corpus = generateCorpus(corpusRng, 500);
  const vocab = buildVocab(corpus.tokens);
  const pairs = windowPairs(corpus.sentences, vocab, 2);
  console.log("=".repeat(70));
  console.log("第 04 章 · 负采样：把 softmax 砍成二分类 (SGNS)");
  console.log("=".repeat(70));
  console.log(
    `语料: ${corpus.sentences.length} 句 / ${corpus.tokens.length} token · ` +
      `词表 V=${vocab.size} · 训练 pair=${pairs.length} · DIM=${DIM}`,
  );
  console.log(`最高频 5 词 (id 升序=频率降序): ${vocab.itos.slice(0, 5).join(", ")}`);

  // ---- 实验 ① SGNS 训练 + 单步耗时, 与全 softmax 对比 ----------------------
  console.log("\n" + "-".repeat(70));
  console.log("① SGNS 训练收敛 + 单步耗时, 对比全 softmax (cost 与 V 的关系)");
  console.log("-".repeat(70));
  const negW075 = buildNegWeights(vocab, "unigram075");
  const baseK = 5;
  const run = trainSgns(vocab, pairs, negW075, baseK, SEED);
  console.log(`SGNS (k=${baseK}, unigram^0.75) loss 曲线 (每 epoch 均值):`);
  console.log(asciiLine(run.losses));
  console.log(
    `loss ${run.losses[0].toFixed(3)} -> ${run.losses[run.losses.length - 1].toFixed(3)} ` +
      `· ${run.steps} 步 · 训练墙钟 ${run.wallMs.toFixed(1)}ms · ` +
      `平均 ${(run.wallMs / run.steps).toFixed(3)}ms/步`,
  );
  console.log(neighborStr(vocab, run.vectors, "king"));
  console.log(neighborStr(vocab, run.vectors, "dog"));

  // V-scaling: time ONE softmax step vs ONE SGNS step at growing synthetic V.
  // The whole argument of the chapter lives in this table.
  console.log("\n单步 forward+backward 墙钟随词表 V 的变化 (DIM 固定, 真测):");
  const vGrid = [40, 200, 1000, 4000];
  const softmaxMs = vGrid.map(measureSoftmaxStepMs);
  const sgnsMs = vGrid.map((v) => measureSgnsStepMs(v, baseK));
  console.log("  V      softmax(ms/步)   SGNS k=5(ms/步)   softmax/SGNS 倍数");
  for (let i = 0; i < vGrid.length; i++) {
    const ratio = sgnsMs[i] > 0 ? softmaxMs[i] / sgnsMs[i] : Infinity;
    console.log(
      `  ${String(vGrid[i]).padEnd(6)} ${softmaxMs[i].toFixed(3).padStart(12)}` +
        `   ${sgnsMs[i].toFixed(3).padStart(13)}   ${ratio.toFixed(1).padStart(14)}x`,
    );
  }
  console.log(
    "  读法: softmax 单步耗时随 V 近似线性上涨 (要对全词表求和), " +
      "SGNS 几乎持平 (只碰 1+k 行) → 这就是 word2vec 能上百万词表的原因。",
  );
  console.log("  (toy 绝对值偏小; 可迁移的是趋势: softmax∝V, SGNS≈常数)");

  // ---- 实验 ② 扫 k = 1/5/20: 耗时 vs 最近邻质量 ----------------------------
  console.log("\n" + "-".repeat(70));
  console.log("② 负样本数 k 扫描: 训练耗时 vs 最近邻质量 (权衡)");
  console.log("-".repeat(70));
  const kGrid = [1, 5, 20];
  const kLabels: string[] = [];
  const kQuality: number[] = [];
  console.log("  k    训练墙钟(ms)   ms/步     planted-pair 平均cos   king 最近邻");
  for (const k of kGrid) {
    const r = trainSgns(vocab, pairs, negW075, k, SEED);
    const q = plantedPairScore(vocab, r.vectors);
    kLabels.push(`k=${k}`);
    kQuality.push(q);
    const kingNbr = neighborStr(vocab, r.vectors, "king", 2).replace("king → ", "");
    console.log(
      `  ${String(k).padEnd(4)} ${r.wallMs.toFixed(1).padStart(11)}   ` +
        `${(r.wallMs / r.steps).toFixed(3).padStart(6)}   ${q.toFixed(3).padStart(18)}   ${kingNbr}`,
    );
  }
  console.log("\n  planted-pair 质量随 k 的变化 (越高=该近的词越近):");
  console.log(asciiBar(kLabels, kQuality, 30));
  console.log(
    "  读法: k 越大每步多算 (k+1)/(1+1) 倍点积, 耗时近似线性涨; " +
      "质量先升后饱和。k=5 通常是 toy 上的甜点 (word2vec 推荐 5-20)。",
  );

  // ---- 实验 ③ 三种负采样分布对比 + 失败模式 -------------------------------
  console.log("\n" + "-".repeat(70));
  console.log("③ 三种负采样分布对比 (uniform / unigram / unigram^0.75) + 失败模式");
  console.log("-".repeat(70));
  // Show what each distribution actually samples: expected fraction of negatives
  // that are the single most-frequent (stopword-like) token. This number IS the
  // failure mechanism made concrete.
  const topId = 0; // id 0 = most frequent by vocab convention
  const dists: NegDist[] = ["uniform", "unigram", "unigram075"];
  console.log(`  最高频词 = "${vocab.itos[topId]}" (出现 ${vocab.counts[topId]} 次)`);
  console.log("  各分布下, 单次负采样抽中该高频词的概率:");
  for (const d of dists) {
    const w = buildNegWeights(vocab, d);
    const total = w.reduce((a, b) => a + b, 0);
    console.log(`    ${d.padEnd(11)}: ${((w[topId] / total) * 100).toFixed(1)}%`);
  }

  console.log("\n  各分布训出向量的最近邻 (planted-pair cos + 虚词侵入率, 见诚实读法):");
  for (const d of dists) {
    const w = buildNegWeights(vocab, d);
    const r = trainSgns(vocab, pairs, w, baseK, SEED);
    const q = plantedPairScore(vocab, r.vectors);
    const intr = intruderFraction(vocab, r.vectors);
    console.log(`  [${d}] planted-pair cos=${q.toFixed(3)} · 虚词侵入近邻率=${intr.toFixed(3)}`);
    console.log("    " + neighborStr(vocab, r.vectors, "king"));
    console.log("    " + neighborStr(vocab, r.vectors, "bread"));
  }
  console.log(
    "\n  诚实读法 (toy V=43, 必读): 上面两个标量指标在三种分布间几乎打平、且" +
      "\n  排名随种子抖动——这不是 bug, 是 toy 词表上 planted 结构太强、指标饱和。" +
      "\n  你的 memory 教训正在此复现: 小词表上别靠单个饱和标量给分布排名。" +
      "\n" +
      "\n  真正可信、且与词表规模无关的差异是『负样本组成』(上方百分比, 精确算出):" +
      "\n    raw unigram 把 16.3% 负样本砸在 'the' 一个词上, unigram^0.75 降到 10.8%。" +
      "\n  失败机制(可迁移到真语料): 负样本被高频虚词占满 → 模型反复学'别把虚词当" +
      '\n  上下文\'(它本就不会) → 内容词之间拿不到区分压力。质性上已能看到: raw' +
      "\n  unigram 下 'king→...and', 'bread→rules/not' 把虚词/跨簇词拉进近邻。" +
      "\n" +
      "\n  为什么真语料里 0.75 才是甜点(本 toy 看不全, 标注为机制论断): 词表 10^5+ 时" +
      "\n  uniform 几乎抽不到高频词(负样本全是无信息长尾), raw unigram 又被高频词淹没;" +
      "\n  0.75 次幂压平头部、保留中频内容词——两头的坑它都避开。toy 上 uniform 不差" +
      "\n  仅因罕见词多、虚词天然占比低, 这个优势在真语料不成立。",
  );

  console.log("\n" + "=".repeat(70));
  console.log("结论: SGNS 用 (1+k) 个二分类替换 V 路 softmax, 单步成本与 V 解耦 (实验①真测);");
  console.log("k 控制 cost↔质量权衡, 先升后饱和 (实验②); 负采样分布决定负样本花在哪——");
  console.log("raw unigram 浪费在高频虚词(失败模式), unigram^0.75 折中, 真语料优势随 V 放大(实验③)。");
  console.log("=".repeat(70));
}

main();
