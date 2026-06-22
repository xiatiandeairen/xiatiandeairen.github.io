// stage03-skipgram.ts — the first time we ACTUALLY train word vectors.
//
// Chapters 01/02 produced "embeddings" by reading co-occurrence counts directly.
// That works but it is not learning: nothing is optimized, there are no parameters
// that move. This stage is the pivot of the whole book — an embedding stops being a
// statistic you compute and becomes a PARAMETER you train by gradient descent on a
// predictive loss. Concretely: word2vec skip-gram with a full softmax.
//
// The model (deliberately the textbook form, no tricks yet):
//   - Two embedding tables. W_in[center] is the vector used WHEN a word is the
//     center; W_out[context] is a separate vector used WHEN a word is predicted as
//     context. Two tables, not one, because a word's role as "what I am" differs
//     from its role as "what predicts me" — tying them hurts at this scale and
//     obscures the mechanism. The trained embedding we keep for downstream use is
//     W_in (convention; W_out is discarded, as in the original paper).
//   - score(center, w) = dot(W_in[center], W_out[w]) for every w in the vocab.
//   - p(context | center) = softmax(scores).
//   - loss for one (center, context) pair = -log p(context | center)  (NLL).
//
// Why this is the honest, mechanism-revealing version and also why it does NOT
// scale: the softmax denominator sums over the ENTIRE vocabulary, so every single
// training pair costs O(V) dot products and O(V) graph nodes. That is the whole
// point of demo ③ below — we MEASURE the per-step cost growing ~linearly with V,
// which is exactly the wall that motivates negative sampling / hierarchical softmax
// in the next chapter. We pay the full-softmax cost here on purpose so the reader
// feels the bottleneck before seeing the fix.
//
// Honesty caveats stated up front (see also core/text.ts header):
//   - Toy vocab (~43 words), tiny dim, minibatch subsampling: absolute neighbor
//     quality and loss numbers are OPTIMISTIC. What transfers to real corpora is
//     the MECHANISM (loss really descends via real gradients) and the RELATIVE
//     trend (per-step cost ~linear in V). Absolute throughput is meaningless:
//     scalar autograd is thousands of times slower than a tensor lib.
//   - The vocab-scaling sweep (③) uses SYNTHETIC random vocab rows so we can dial
//     V freely; it measures forward-pass cost only, which is the term that the
//     softmax dominates. It is a wall-clock measurement, not an estimate.

import { Rng, gaussian, shuffle } from "./core/rng.js";
import { Value, dot, makeMat, vecData, type Vec, type Mat } from "./core/autograd.js";
import { Adam, collectParams } from "./core/optim.js";
import { asciiLine, asciiBar } from "./core/plot.js";
import { nearestNeighbors, cosineSimilarity } from "./core/eval.js";
import {
  generateCorpus,
  buildVocab,
  windowPairs,
  type Vocab,
  type Pair,
} from "./core/text.js";

// ---------------------------------------------------------------------------
// Model: a pair of embedding tables + a full-softmax NLL forward pass.
// ---------------------------------------------------------------------------

interface SkipGram {
  win: Mat; // V x D, center-word vectors (the embedding we keep)
  wout: Mat; // V x D, context-word vectors (auxiliary, discarded after training)
  vocabSize: number;
  dim: number;
}

// Build the model with small-Gaussian init. The init SCALE is load-bearing and is
// the subject of failure demo ②: vectors must start small so initial dot products
// are near zero, the first softmax is near-uniform, and gradients are well scaled.
// Too-large init saturates softmax (one prob ~1, rest ~0) → vanishing gradient →
// the model barely learns. We expose initStd so the failure demo can crank it.
function buildSkipGram(rng: Rng, vocabSize: number, dim: number, initStd: number): SkipGram {
  const init = () => gaussian(rng, 0, initStd);
  return {
    win: makeMat(vocabSize, dim, init),
    wout: makeMat(vocabSize, dim, init),
    vocabSize,
    dim,
  };
}

// Forward + loss for ONE (center, context) pair, returned as a scalar Value so the
// caller can sum a minibatch and call .backward() once.
//
// `stable` controls the log-sum-exp trick: when true (always, in real training) we
// subtract the (constant) max score before exp so a large positive score can't
// overflow exp() to Infinity. The max is detached data, so it shifts every term
// equally and does NOT change the gradient — it only changes the numerics.
//
// We expose `stable=false` ONLY for the NaN failure demo: it reproduces the naive
// softmax that overflows the moment scores grow large, which is a DISTINCT failure
// from learning-rate divergence and worth seeing separately (it is exactly why the
// shift exists).
function pairLoss(model: SkipGram, pair: Pair, stable = true): Value {
  const center = model.win[pair.center];
  // score_w = <W_in[center], W_out[w]> for every w. This O(V) loop is the cost.
  const scores: Value[] = [];
  for (let w = 0; w < model.vocabSize; w++) {
    scores.push(dot(center, model.wout[w]));
  }
  let maxScore = 0;
  if (stable) {
    maxScore = -Infinity;
    for (const s of scores) if (s.data > maxScore) maxScore = s.data;
  }
  const expScores = scores.map((s) => s.sub(maxScore).exp());
  let denom = new Value(0);
  for (const e of expScores) denom = denom.add(e);
  // NLL = -log( exp(score_context) / sum_w exp(score_w) ) = -(score_context_shifted - log denom).
  const pContext = expScores[pair.context].div(denom);
  return pContext.log().mul(-1);
}

// Mean loss over a minibatch of pairs. Mean (not sum) keeps the loss scale
// independent of batch size, so the lr in the failure demos means the same thing
// regardless of how many pairs we drew.
function batchLoss(model: SkipGram, pairs: Pair[], stable = true): Value {
  let acc = new Value(0);
  for (const p of pairs) acc = acc.add(pairLoss(model, p, stable));
  return acc.div(pairs.length);
}

// ---------------------------------------------------------------------------
// Training loop. Returns the per-epoch loss series for plotting.
// ---------------------------------------------------------------------------

interface TrainResult {
  losses: number[];
  hitNaN: boolean; // true if loss hit NaN/Inf — the explicit numerical-overflow signal
}

// One full training run. The minibatch is re-sampled (shuffled prefix) each epoch
// so SGD/Adam sees a fresh slice; determinism comes from the passed-in Rng.
//
// `stable` is forwarded to the loss; the NaN demo flips it off.
//
// Invariant (the autograd loop contract): zeroGrad → forward → backward → step,
// in that order, every step. Skipping zeroGrad sums grads across steps and is the
// canonical divergence bug; the optimizer owns zeroGrad so we cannot forget which
// params to clear.
function trainSkipGram(
  model: SkipGram,
  pairs: Pair[],
  rng: Rng,
  opts: { epochs: number; batchSize: number; lr: number; stable?: boolean },
): TrainResult {
  const stable = opts.stable ?? true;
  const params = collectParams(model.win, model.wout);
  const opt = new Adam(params, opts.lr);
  const losses: number[] = [];
  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    shuffle(rng, pairs);
    const batch = pairs.slice(0, Math.min(opts.batchSize, pairs.length));
    opt.zeroGrad();
    const loss = batchLoss(model, batch, stable);
    loss.backward();
    opt.step();
    losses.push(loss.data);
    // Fail-fast on NaN/Inf: once the loss is non-finite, every later number is
    // garbage, so we stop and report rather than printing a meaningless curve.
    if (!Number.isFinite(loss.data)) {
      return { losses, hitNaN: true };
    }
  }
  return { losses, hitNaN: false };
}

// ---------------------------------------------------------------------------
// Helpers for evaluation / reporting.
// ---------------------------------------------------------------------------

// Detach the trained input embeddings to a plain number[][] for eval. Eval must
// never extend the live autograd graph (see core/eval.ts header), hence vecData.
function embeddingMatrix(model: SkipGram): number[][] {
  return model.win.map((row: Vec) => vecData(row));
}

// Pretty-print the nearest neighbors of a probe word using the trained space.
function reportNeighbors(emb: number[][], vocab: Vocab, word: string, k: number): void {
  const id = vocab.stoi.get(word);
  if (id === undefined) {
    console.log(`  "${word}": not in vocab`);
    return;
  }
  const nbrs = nearestNeighbors(emb[id], emb, k);
  const shown = nbrs.map((n) => `${vocab.itos[n.index]}(${n.score.toFixed(2)})`).join("  ");
  console.log(`  ${word.padEnd(8)} → ${shown}`);
}

// ---------------------------------------------------------------------------
// Demo ③: vocab-size scaling. Measure the wall-clock cost of ONE forward+backward
// step as V grows, to expose the full-softmax O(V) bottleneck empirically.
// ---------------------------------------------------------------------------

// We build a fresh model at each V on a SYNTHETIC vocab (random rows) so we can dial
// V independently of the toy corpus. A single (center=0, context=1) pair drives one
// full forward+backward. We time the median of several reps to damp JIT/GC jitter.
// This is a real measurement (wall-clock), not an estimate.
function measureStepMs(rng: Rng, vocabSize: number, dim: number, reps: number): number {
  const samples: number[] = [];
  for (let r = 0; r < reps; r++) {
    const model = buildSkipGram(rng, vocabSize, dim, 0.1);
    const params = collectParams(model.win, model.wout);
    const opt = new Adam(params, 0.01);
    const pair: Pair = { center: 0, context: 1 };
    const t0 = performance.now();
    opt.zeroGrad();
    const loss = pairLoss(model, pair);
    loss.backward();
    opt.step();
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)]; // median
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function main(): void {
  // Fixed seeds: every number below is reproducible bit-for-bit on a cold machine.
  const corpusRng = new Rng(7);
  const corpus = generateCorpus(corpusRng, 400);
  const vocab = buildVocab(corpus.tokens);
  const pairs = windowPairs(corpus.sentences, vocab, 2);
  const DIM = 16;

  console.log("=".repeat(70));
  console.log("Stage 03 — word2vec skip-gram: 第一次真的训练词向量");
  console.log("=".repeat(70));
  console.log(
    `语料: ${corpus.sentences.length} 句, ${corpus.tokens.length} token; ` +
      `词表 V=${vocab.size}, 维度 D=${DIM}; 训练对 ${pairs.length} 个`,
  );
  console.log(
    "注意: toy 语料，绝对数字偏乐观；可迁移的是机制(loss 真降)与相对趋势(单步成本~随 V 线性)。\n",
  );

  // ---- ① 正常训练: loss 真的下降 + 学到语义聚类 ----
  console.log("① 正常训练 (Adam, 小随机初始化, 全 softmax NLL)");
  const modelRng = new Rng(42);
  const model = buildSkipGram(modelRng, vocab.size, DIM, 0.1);
  const trainRng = new Rng(123);
  const res = trainSkipGram(model, pairs, trainRng, { epochs: 120, batchSize: 256, lr: 0.05 });
  console.log(asciiLine(res.losses, 60, 10));
  const first = res.losses[0];
  const last = res.losses[res.losses.length - 1];
  console.log(
    `   loss ${first.toFixed(3)} → ${last.toFixed(3)} ` +
      `(降幅 ${(((first - last) / first) * 100).toFixed(1)}%, ${res.losses.length} epochs)`,
  );
  // Sanity baseline: a uniform softmax over V classes has loss ln(V). Beating it
  // proves the model learned a non-trivial conditional distribution, not noise.
  console.log(`   均匀分布基线 loss = ln(V) = ${Math.log(vocab.size).toFixed(3)} (训练后应低于此)\n`);

  // ---- ② 最近邻: 语义结构是学出来的, 不是抄 co-occurrence ----
  console.log("② 训练后最近邻 (用 W_in, cosine):");
  const emb = embeddingMatrix(model);
  for (const probe of ["king", "queen", "cat", "north", "bread"]) {
    reportNeighbors(emb, vocab, probe, 4);
  }
  // One quantitative receipt: a planted analogy pair should be among each other's
  // top neighbors. We print the raw cosine so the reader can see the magnitude.
  const kId = vocab.stoi.get("king")!;
  const qId = vocab.stoi.get("queen")!;
  console.log(`   cos(king, queen) = ${cosineSimilarity(emb[kId], emb[qId]).toFixed(3)}`);
  // Honest filler check: 'um'/'thing' never co-occur with informative context, so
  // their vectors stay near init — you can't learn structure that isn't in the data.
  if (vocab.stoi.has("um") && vocab.stoi.has("king")) {
    const uId = vocab.stoi.get("um")!;
    console.log(
      `   cos(um, king)    = ${cosineSimilarity(emb[uId], emb[kId]).toFixed(3)} ` +
        `(filler 词无信号, 接近噪声)\n`,
    );
  } else {
    console.log("");
  }

  // ---- ③ 词表规模扫描: 单步成本随 V 近线性增长 (全 softmax 瓶颈) ----
  console.log("③ 词表规模扫描: 单步 forward+backward 耗时随 V 增长 (全 softmax = O(V))");
  const sweepRng = new Rng(2024);
  const vocabSizes = [50, 200, 800];
  const stepMs = vocabSizes.map((v) => measureStepMs(sweepRng, v, DIM, 7));
  console.log(asciiBar(vocabSizes.map((v) => `V=${v}`), stepMs, 30) + "  (ms, median of 7)");
  // Print the cost-per-V ratio. For a true O(V) cost the ratio ms/V is ~constant
  // and the 4x V jumps should show ~4x time. We show both so the reader can judge.
  for (let i = 1; i < vocabSizes.length; i++) {
    const vRatio = vocabSizes[i] / vocabSizes[i - 1];
    const tRatio = stepMs[i] / stepMs[i - 1];
    console.log(
      `   V ${vocabSizes[i - 1]}→${vocabSizes[i]} (×${vRatio.toFixed(1)}): ` +
        `时间 ×${tRatio.toFixed(2)} ` +
        `(理想线性应≈×${vRatio.toFixed(1)})`,
    );
  }
  console.log(
    "   结论: 每个训练对都要对全词表做 V 次点积, 成本随 V 线性涨 → 这就是下一章\n" +
      "         negative sampling / hierarchical softmax 要砍掉的瓶颈。\n",
  );

  // ---- 失败模式 ① 学习率过大 → loss 发散 (越训越差, 高于均匀基线) ----
  // Honest framing: with a stable softmax, a too-large lr does NOT NaN — it
  // DIVERGES. The loss climbs and parks far above the ln(V) uniform baseline, i.e.
  // the model is worse than guessing. NaN specifically needs the naive softmax,
  // demoed separately just below.
  console.log("失败模式 ① 学习率过大 (lr=5.0) → loss 发散, 越训越差 (高于均匀基线):");
  const badLrModel = buildSkipGram(new Rng(42), vocab.size, DIM, 0.1);
  const badLrRes = trainSkipGram(badLrModel, pairs, new Rng(123), {
    epochs: 60,
    batchSize: 256,
    lr: 5.0,
  });
  const badLrFirst = badLrRes.losses[0];
  const badLrLast = badLrRes.losses[badLrRes.losses.length - 1];
  console.log(
    `   loss ${badLrFirst.toFixed(3)} → ${badLrLast.toFixed(3)} ` +
      `(不降反升; 均匀基线 ln(V)=${Math.log(vocab.size).toFixed(3)}, 正常 lr=0.05 收敛到 ${last.toFixed(3)})`,
  );
  console.log(
    "   why: Adam 每步 effective step ≈ lr; lr 过大时每步跨过极小点来回弹, 参数发散,\n" +
      "        loss 停在远高于基线处 = 学崩了 (但 stable softmax 兜住没 NaN)。\n",
  );

  // ---- 失败模式 ②(数值) 朴素 softmax (无 max 平移) → score 增大时 exp 溢出 → NaN ----
  // This is the failure the stable shift exists to prevent. Same lr=5.0 that merely
  // diverged above now produces a true NaN once scores grow, because exp() of a
  // large score overflows to Infinity → Inf/Inf = NaN → poisons every parameter.
  console.log("失败模式 ② 朴素 softmax (去掉 max 平移) + 大 lr → exp 溢出 → 真正爆 NaN:");
  const naiveModel = buildSkipGram(new Rng(42), vocab.size, DIM, 0.1);
  const naiveRes = trainSkipGram(naiveModel, pairs, new Rng(123), {
    epochs: 60,
    batchSize: 256,
    lr: 5.0,
    stable: false,
  });
  console.log(
    `   ${naiveRes.hitNaN ? "已爆 NaN" : "未爆 NaN"}: ` +
      `第 ${naiveRes.losses.length} epoch loss=${naiveRes.losses[naiveRes.losses.length - 1]} ` +
      `(对比 ① 同 lr 但 stable softmax 不爆, 只发散)`,
  );
  console.log(
    "   why: score 变大 → exp(score)=Inf → Inf/Inf=NaN → NaN 顺着 backward 污染全部参数。\n" +
      "        这正是 pairLoss 默认减去 max 平移 (log-sum-exp trick) 要防的事。\n",
  );

  // ---- 失败模式 ③ 初始化过大 → softmax 饱和不学 ----
  console.log("失败模式 ③ 初始化方差过大 (initStd=5.0) → softmax 饱和, 梯度消失, 几乎不学:");
  const bigInitModel = buildSkipGram(new Rng(42), vocab.size, DIM, 5.0);
  const bigInitRes = trainSkipGram(bigInitModel, pairs, new Rng(123), {
    epochs: 120,
    batchSize: 256,
    lr: 0.05,
  });
  const bigFirst = bigInitRes.losses[0];
  const bigLast = bigInitRes.losses[bigInitRes.losses.length - 1];
  const bigDrop = ((bigFirst - bigLast) / bigFirst) * 100;
  console.log(
    `   loss ${bigFirst.toFixed(3)} → ${bigLast.toFixed(3)} ` +
      `(降幅仅 ${bigDrop.toFixed(1)}%, 对比小初始化降幅 ${(((first - last) / first) * 100).toFixed(1)}%)`,
  );
  // Quantify the saturation directly: with huge init, dot products are huge, so the
  // softmax puts ~all mass on one class → the top prob is near 1 → near-zero grad.
  const satCenter = bigInitModel.win[kId];
  let satMax = -Infinity;
  for (let w = 0; w < bigInitModel.vocabSize; w++) {
    const s = dot(satCenter, bigInitModel.wout[w]).data;
    if (s > satMax) satMax = s;
  }
  // Recompute the post-softmax top prob for the 'king' center as the saturation receipt.
  let denomSat = 0;
  for (let w = 0; w < bigInitModel.vocabSize; w++) {
    denomSat += Math.exp(dot(satCenter, bigInitModel.wout[w]).data - satMax);
  }
  const topProb = 1 / denomSat; // exp(max - max)=1 over denom
  console.log(
    `   "king" 中心词 softmax 最大概率 ≈ ${topProb.toFixed(3)} ` +
      `(接近 1 = 饱和, 几乎无梯度可学)`,
  );
  console.log(
    "   why: 大初始化 → 点积量级大 → softmax 一峰独大 → ∂loss/∂score≈0 → 参数推不动。\n",
  );

  console.log("=".repeat(70));
  console.log("小结: embedding = 被梯度下降训练的参数 (本章) ≠ 直接读统计 (01/02)。");
  console.log("      全 softmax 让机制透明但成本 O(V); 下一章用 negative sampling 砍掉它。");
  console.log("=".repeat(70));
}

main();
