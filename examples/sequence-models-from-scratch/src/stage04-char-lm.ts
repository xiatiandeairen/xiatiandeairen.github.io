// stage04-char-lm.ts — Training a char-level LSTM language model: teacher forcing,
//   temperature sampling, and the exposure-bias / fixed-state memory-decay failure mode.
//
// WHY this chapter exists (and what the "honest number" is):
//   Stages 01-03 proved a gated cell CAN carry information across a span. This chapter
//   asks the practical question: once you actually TRAIN one with teacher forcing on a
//   tiny corpus, (a) does validation perplexity really fall, (b) does sampled text go
//   from noise to locally-structured English, and crucially (c) WHERE does it break.
//   The break is the point: a model trained to predict-next-given-ground-truth is never
//   shown its own mistakes (exposure bias), and an LSTM's memory is a FIXED-WIDTH state
//   vector — push generation far past the training window and errors compound + the
//   bounded state forgets, so text drifts into repetition / loss of structure. We
//   MEASURE that drift (repetition ratio, char entropy) rather than just asserting it.
//
// HONESTY: the corpus is ~567 chars of public-domain Aesop and the model is tiny. Absolute
//   perplexity here is OPTIMISTIC (the model can essentially memorize). What transfers is
//   the RELATIVE story: val-ppl drops far below the uniform floor (=vocab) during fit, and
//   long-context autoregression degrades measurably vs in-distribution length. Wall-clock
//   is REAL (timeit); MAC/param counts are computed from shapes, not guessed.
//
// DESIGN: matmul in core is 2D-only (no batched matmul), so we run ONE sequence at a time
//   (batch=1): each timestep is a [1,D] row, gates come from a single Linear over
//   concat([x_t, h_{t-1}]) then sliced into 4 chunks. Per-sequence logits are concat'd to
//   [T,vocab] and fed to crossEntropy once. Slow but transparent — the chapter is about
//   dynamics, not throughput.

import { makeRng, type Rng } from "./core/prng.js";
import { Tensor } from "./core/tensor.js";
import { Embedding, Linear, Module, collectParams } from "./core/nn.js";
import { Adam, clipGradNorm } from "./core/optim.js";
import { charSeq, type CharDataset } from "./core/data.js";
import { perplexity, timeit } from "./core/metrics.js";
import { lineChart, sparkline } from "./core/plot.js";

// ---------------------------------------------------------------------------
// LSTMCell — built here from core primitives (stage03's payload; self-contained
// so this file does not import a stageNN that would run its own main()).
//
// Gate math (Hochreiter & Schmidhuber, with the standard forget gate):
//   z      = [x_t ; h_{t-1}] @ W + b          (one Linear, 4*hidden outputs)
//   i,f,o  = sigmoid(z chunks)                (input / forget / output gates)
//   g      = tanh(z chunk)                    (candidate cell update)
//   c_t    = f * c_{t-1} + i * g              (the additive cell path = long memory)
//   h_t    = o * tanh(c_t)
//
// WHY the additive c_t path matters: gradients flow through c_t via f (a multiply by a
//   number in (0,1)) rather than through repeated tanh saturation — that is the whole
//   reason an LSTM's effective memory beats a vanilla RNN's. We expose c_t so the failure
//   demo can inspect state behavior.
// ---------------------------------------------------------------------------
class LSTMCell extends Module {
  readonly gates: Linear; // [in+hidden, 4*hidden]
  readonly hidden: number;
  constructor(inDim: number, hidden: number, rng: Rng) {
    super();
    this.hidden = hidden;
    // Single fused projection for all 4 gates: fewer matmuls, identical math to 4 Linears.
    this.gates = new Linear(inDim + hidden, 4 * hidden, rng, { init: "xavier" });
    // Forget-gate bias = 1: a classic init trick so the cell DEFAULTS to remembering at
    // step 0 (f≈sigmoid(1)≈0.73). Without it early training erases memory before it learns
    // to keep it. The forget chunk is columns [hidden, 2*hidden).
    const b = this.gates.b!.data;
    for (let j = this.hidden; j < 2 * this.hidden; j++) b[j] = 1;
  }
  /** One timestep. x:[1,in], h/c:[1,hidden] -> {h,c} each [1,hidden]. */
  step(x: Tensor, h: Tensor, c: Tensor): { h: Tensor; c: Tensor } {
    const z = this.gates.forward(Tensor.concat([x, h], 1)); // [1, 4*hidden]
    const H = this.hidden;
    const i = z.slice(1, 0, H).sigmoid();
    const f = z.slice(1, H, 2 * H).sigmoid();
    const o = z.slice(1, 2 * H, 3 * H).sigmoid();
    const g = z.slice(1, 3 * H, 4 * H).tanh();
    const cNext = f.mul(c).add(i.mul(g)); // additive long-memory path
    const hNext = o.mul(cNext.tanh());
    return { h: hNext, c: cNext };
  }
  override params(): Tensor[] {
    return this.gates.params();
  }
}

// ---------------------------------------------------------------------------
// CharLM — Embedding -> LSTM (unrolled) -> Linear readout to vocab logits.
// ---------------------------------------------------------------------------
class CharLM extends Module {
  readonly embed: Embedding;
  readonly cell: LSTMCell;
  readonly readout: Linear;
  readonly hidden: number;
  constructor(vocab: number, embDim: number, hidden: number, rng: Rng) {
    super();
    this.hidden = hidden;
    this.embed = new Embedding(vocab, embDim, rng, 0.1);
    this.cell = new LSTMCell(embDim, hidden, rng);
    this.readout = new Linear(hidden, vocab, rng, { init: "xavier" });
  }

  /**
   * Teacher-forcing forward over a full id sequence: feed the GROUND-TRUTH char at every
   * step (never the model's own prediction). Returns logits [T, vocab] and the final
   * (h,c) so callers can continue the same state for autoregression.
   * INVARIANT: state starts at zeros each call (no cross-sequence leakage).
   */
  forwardTeacherForced(ids: number[]): { logits: Tensor; h: Tensor; c: Tensor } {
    let h = Tensor.zeros([1, this.hidden]);
    let c = Tensor.zeros([1, this.hidden]);
    const rows: Tensor[] = [];
    for (let t = 0; t < ids.length; t++) {
      const x = this.embed.forward([ids[t]]); // [1, embDim]
      const out = this.cell.step(x, h, c);
      h = out.h;
      c = out.c;
      rows.push(this.readout.forward(h)); // [1, vocab]
    }
    return { logits: Tensor.concat(rows, 0), h, c };
  }

  override params(): Tensor[] {
    return collectParams([this.embed, this.cell, this.readout]);
  }
}

// ---------------------------------------------------------------------------
// Sampling: advance the model one char at a time using its OWN prediction.
// temperature scales logits before softmax: T<1 sharpens (greedier), T>1 flattens
// (more random). T->0 == argmax. We draw from the categorical via the seeded Rng so a
// fixed seed yields bit-identical text (honest-number requirement).
// ---------------------------------------------------------------------------
function sampleFromLogits(logitsRow: Float64Array, temperature: number, rng: Rng): number {
  const T = Math.max(1e-6, temperature); // guard div-by-zero; tiny T ~ argmax
  // Stable softmax with temperature: subtract max, exp, normalize.
  let max = -Infinity;
  for (const v of logitsRow) if (v > max) max = v;
  const probs = new Float64Array(logitsRow.length);
  let denom = 0;
  for (let i = 0; i < logitsRow.length; i++) {
    const e = Math.exp((logitsRow[i] - max) / T);
    probs[i] = e;
    denom += e;
  }
  // Inverse-CDF sample. One Rng draw -> deterministic given seed.
  const r = rng.next() * denom;
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1; // numerical fallback (r == denom edge)
}

/**
 * Autoregressive generation of `n` chars starting from `seed` chars. Reuses the model's
 * own predictions as next input (NO teacher forcing) — this is where exposure bias and
 * fixed-state memory limits show up. Returns generated char ids (excluding the seed).
 */
function generate(
  model: CharLM,
  data: CharDataset,
  seedText: string,
  n: number,
  temperature: number,
  rng: Rng,
): number[] {
  let h = Tensor.zeros([1, model.hidden]);
  let c = Tensor.zeros([1, model.hidden]);
  // Warm the state on the seed (teacher-forced on the prompt itself).
  const seedIds = data.encode(seedText);
  let lastLogits: Tensor | null = null;
  for (const id of seedIds) {
    const x = model.embed.forward([id]);
    const out = model.cell.step(x, h, c);
    h = out.h;
    c = out.c;
    lastLogits = model.readout.forward(h);
  }
  const generated: number[] = [];
  let cur = lastLogits!;
  for (let t = 0; t < n; t++) {
    const next = sampleFromLogits(cur.data, temperature, rng);
    generated.push(next);
    const x = model.embed.forward([next]); // feed OWN prediction back in
    const out = model.cell.step(x, h, c);
    h = out.h;
    c = out.c;
    cur = model.readout.forward(h);
  }
  return generated;
}

// ---------------------------------------------------------------------------
// Evaluation: mean cross-entropy (nats) over a set of sequences -> perplexity.
// No grad needed but we reuse the forward; cheap on this toy size.
// ---------------------------------------------------------------------------
function evalPerplexity(model: CharLM, seqs: { x: number[]; y: number[] }[]): number {
  let ceSum = 0;
  let rows = 0;
  for (const s of seqs) {
    const { logits } = model.forwardTeacherForced(s.x);
    const ce = logits.crossEntropy(s.y); // mean CE over this seq's T rows
    ceSum += ce.data[0] * s.y.length; // un-mean to sum, so global mean is exact
    rows += s.y.length;
  }
  return perplexity(ceSum / rows);
}

/** Repetition ratio: 1 - unique/total over a sliding measure of consecutive-char loops. */
function repetitionMetrics(ids: number[]): { uniqueRatio: number; entropyBits: number; topRun: number } {
  if (ids.length === 0) return { uniqueRatio: 0, entropyBits: 0, topRun: 0 };
  const counts = new Map<number, number>();
  let topRun = 1;
  let run = 1;
  for (let i = 0; i < ids.length; i++) {
    counts.set(ids[i], (counts.get(ids[i]) ?? 0) + 1);
    if (i > 0 && ids[i] === ids[i - 1]) {
      run++;
      if (run > topRun) topRun = run;
    } else run = 1;
  }
  const uniqueRatio = counts.size / ids.length;
  // Shannon entropy (bits) of the char distribution: high = varied, collapses as text
  // degenerates into a few repeated chars.
  let entropy = 0;
  for (const cnt of counts.values()) {
    const p = cnt / ids.length;
    entropy -= p * Math.log2(p);
  }
  return { uniqueRatio, entropyBits: entropy, topRun };
}

// ===========================================================================
// main
// ===========================================================================
function main(): void {
  // Init/training uses its own stream; sampling makes a fresh per-temperature stream below
  // (see prng.ts) so changing one does not perturb the other's reproducibility.
  const initRng = makeRng(1234);

  // --- data: char windows over the built-in corpus -------------------------
  const SEQ_LEN = 24;
  const data = charSeq({ seqLen: SEQ_LEN, stride: SEQ_LEN });
  console.log("=== 数据 ===");
  console.log(data.describe());

  const seqs = data.X.map((x, i) => ({ x, y: data.Y[i] }));
  // Deterministic split: last 20% windows held out for validation. No shuffle so the
  // split is stable run-to-run and val truly is unseen tail text.
  const nVal = Math.max(1, Math.floor(seqs.length * 0.2));
  const trainSeqs = seqs.slice(0, seqs.length - nVal);
  const valSeqs = seqs.slice(seqs.length - nVal);
  console.log(`train windows=${trainSeqs.length}  val windows=${valSeqs.length}  vocab=${data.vocabSize}`);

  // --- model ---------------------------------------------------------------
  const EMB = 24;
  const HIDDEN = 64;
  const model = new CharLM(data.vocabSize, EMB, HIDDEN, initRng);
  const params = model.params();
  const paramCount = params.reduce((s, p) => s + p.size, 0);
  console.log("\n=== 模型 ===");
  console.log(`CharLM: embed[${data.vocabSize}x${EMB}] -> LSTM(hidden=${HIDDEN}) -> readout[${HIDDEN}x${data.vocabSize}]`);
  console.log(`参数量 (真实计数): ${paramCount}`);
  const uniformPpl = data.vocabSize; // perplexity of uniform guessing == vocab size
  console.log(`均匀猜测困惑度下界 (=vocab): ${uniformPpl}`);

  // --- training: teacher forcing + Adam + grad clipping --------------------
  const opt = new Adam(params, { lr: 5e-3 });
  const EPOCHS = 40;
  const trainPplCurve: number[] = [];
  const valPplCurve: number[] = [];
  let lastGradNorm = 0;

  // Honest timing: measure one real epoch of teacher-forced fwd+bwd+step.
  const oneEpoch = (): void => {
    for (const s of trainSeqs) {
      opt.zeroGrad();
      const { logits } = model.forwardTeacherForced(s.x);
      const loss = logits.crossEntropy(s.y);
      loss.backward();
      lastGradNorm = clipGradNorm(params, 5);
      opt.step();
    }
  };

  const startCe = evalPerplexity(model, trainSeqs);
  console.log("\n=== 训练 (teacher forcing) ===");
  console.log(`训练前 train 困惑度: ${startCe.toFixed(2)} (接近均匀下界 ${uniformPpl} = 尚未学习)`);

  // Record train+val perplexity EVERY epoch so the curve captures the val MINIMUM (on a
  // tiny corpus val ppl dips early then rises as the model overfits — we must not miss the
  // dip by sampling too coarsely). timeit's warmup rep is itself a real epoch (params
  // persist), so epoch 0's measurement happens inside the timed call; we record from there.
  let bestVal = Infinity;
  let bestValEpoch = 0;
  const recordCurve = (epoch: number): void => {
    const tp = evalPerplexity(model, trainSeqs);
    const vp = evalPerplexity(model, valSeqs);
    trainPplCurve.push(tp);
    valPplCurve.push(vp);
    if (vp < bestVal) {
      bestVal = vp;
      bestValEpoch = epoch;
    }
  };

  let epoch = 0;
  const perEpochWall = timeit(() => {
    oneEpoch();
    recordCurve(epoch);
    epoch++;
  }, 1).perRepMs; // REAL ms for one epoch (warmup epoch also recorded; warmup time discarded)
  for (; epoch < EPOCHS; epoch++) {
    oneEpoch();
    recordCurve(epoch);
  }

  console.log(`单 epoch 真实耗时: ${perEpochWall.toFixed(1)} ms  (wall-clock, warmup 已丢弃)`);
  console.log(`末次梯度范数 (clip 前): ${lastGradNorm.toFixed(3)} (clip 阈值=5)`);
  console.log("\ntrain 困惑度随 epoch:");
  console.log("  " + sparkline(trainPplCurve) + `   ${trainPplCurve[0].toFixed(1)} -> ${trainPplCurve.at(-1)!.toFixed(2)}`);
  console.log("val 困惑度随 epoch (注意先降后升 = 过拟合):");
  console.log("  " + sparkline(valPplCurve) + `   ${valPplCurve[0].toFixed(1)} -> 最低 ${bestVal.toFixed(2)} (epoch ${bestValEpoch}) -> ${valPplCurve.at(-1)!.toFixed(2)}`);
  console.log("\n困惑度曲线 (train=* val=o, 越低越好):");
  console.log(lineChart([trainPplCurve, valPplCurve], { labels: ["train-ppl", "val-ppl"], height: 10, width: 50 }));

  const finalTrain = trainPplCurve.at(-1)!;
  const finalVal = valPplCurve.at(-1)!;
  console.log(`\n最终 train 困惑度=${finalTrain.toFixed(2)}  val 困惑度=${finalVal.toFixed(2)}  均匀下界=${uniformPpl}`);
  console.log(`最佳 val 困惑度=${bestVal.toFixed(2)} @ epoch ${bestValEpoch} (early-stopping 点)`);
  // Honest verdict: success = val ppl ever fell well below the uniform floor (model learned
  // real char structure on unseen tail), NOT that final val is low. On a 567-char corpus a
  // 25k-param LSTM WILL overfit, so val rises again after the dip — that rise is itself the
  // chapter's lesson, not a bug to hide.
  console.log(
    bestVal < uniformPpl * 0.5
      ? `=> 最佳 val (${bestVal.toFixed(2)}) 远低于均匀下界 (${uniformPpl}): 模型确实学到了字符分布 (非 noise).`
      : `=> 最佳 val (${bestVal.toFixed(2)}) 未明显低于均匀下界: 容量/数据需调整.`,
  );
  console.log(
    `   train 持续降到 ${finalTrain.toFixed(2)} 而 val 反弹到 ${finalVal.toFixed(2)} = 在 567 字符小语料上记忆训练集 (过拟合). ` +
      "真实大语料下二者都会持续下降; 这里的相对趋势 (train<<val 后期张开) 才是可迁移结论.",
  );

  // --- sampling at two temperatures (deterministic via fixed seed) ---------
  const PROMPT = "the ";
  const GEN_LEN = 80;
  console.log("\n=== 采样 (确定性: 每个 temperature 用固定 seed 782/787) ===");
  for (const temp of [0.5, 1.0]) {
    // Fixed per-temperature seed (782, 787): bit-identical text run-to-run, but the two
    // temperatures draw from independent streams so their randomness is not coupled.
    const tempRng = makeRng(777 + Math.round(temp * 10));
    const ids = generate(model, data, PROMPT, GEN_LEN, temp, tempRng);
    const text = data.decode(ids);
    const m = repetitionMetrics(ids);
    console.log(`temperature=${temp.toFixed(1)}  unique字符比=${m.uniqueRatio.toFixed(2)}  熵=${m.entropyBits.toFixed(2)}bit`);
    console.log(`  "${PROMPT}${text}"`);
  }
  console.log("(低温 0.5 更确定/重复, 高温 1.0 更随机. 二者都应从乱码变为有局部英文结构.)");

  // --- FAILURE MODE: exposure bias + fixed-state memory decay over long gen -
  // We generate FAR beyond the 24-char training window and measure how text quality
  // decays as a function of position. Two compounding failures:
  //   (1) Exposure bias: trained only on ground-truth inputs, the model never saw its own
  //       errors; one off char shifts it off the training manifold and errors snowball.
  //   (2) Fixed-width state: the LSTM's memory is HIDDEN floats; arbitrarily long context
  //       cannot be retained, so it falls back on short-range habits (repeat / loop).
  console.log("\n=== 失败模式: 远超训练长度的自回归生成 ===");
  console.log(`训练窗口=${SEQ_LEN} 字符. 现在用 greedy (temp->0) 连续生成 400 字符, 分段看退化:`);
  const LONG = 400;
  const longRng = makeRng(42);
  const longIds = generate(model, data, PROMPT, LONG, 0.01, longRng); // near-greedy: isolates exposure bias from sampling noise
  const CHUNK = 80;
  const uniqueByChunk: number[] = [];
  const entropyByChunk: number[] = [];
  const chunkLabels: { label: string; value: number }[] = [];
  for (let start = 0; start < LONG; start += CHUNK) {
    const chunk = longIds.slice(start, start + CHUNK);
    const m = repetitionMetrics(chunk);
    uniqueByChunk.push(m.uniqueRatio);
    entropyByChunk.push(m.entropyBits);
    chunkLabels.push({ label: `pos ${start}-${start + chunk.length}`, value: m.uniqueRatio });
    const preview = data.decode(chunk).replace(/\n/g, "\\n");
    console.log(`  [${String(start).padStart(3)}-${start + chunk.length}] unique=${m.uniqueRatio.toFixed(2)} 熵=${m.entropyBits.toFixed(2)}bit 最长重复串=${m.topRun}  "${preview}"`);
  }
  console.log("\nunique字符比 随生成位置 (越低=越退化/重复):");
  console.log("  " + sparkline(uniqueByChunk) + `   ${uniqueByChunk[0].toFixed(2)} -> ${uniqueByChunk.at(-1)!.toFixed(2)}`);
  console.log("字符熵(bit) 随生成位置 (越低=分布越塌缩到少数字符):");
  console.log("  " + sparkline(entropyByChunk) + `   ${entropyByChunk[0].toFixed(2)} -> ${entropyByChunk.at(-1)!.toFixed(2)}`);

  const firstU = uniqueByChunk[0];
  const lastU = uniqueByChunk.at(-1)!;
  const drift = ((firstU - lastU) / Math.max(1e-9, firstU)) * 100;
  console.log(
    `\n量化 exposure bias / 记忆衰减: 首段 unique=${firstU.toFixed(2)} -> 末段 unique=${lastU.toFixed(2)} ` +
      `(下降 ${drift.toFixed(0)}%).`,
  );
  console.log(
    "解读: teacher forcing 训练时模型从未见过自己的输出; 自回归时一旦走偏, 误差滚雪球, " +
      "且固定宽度的 hidden state 无法记住任意长上下文 -> 文本逐渐塌缩为重复/失序.",
  );
  console.log("(toy 语料绝对值偏乐观; 可迁移的是: 生成长度 >> 训练窗口时质量单调下降这一相对趋势.)");

  // Sanity: compare in-distribution-length generation (should look better) vs the tail.
  const shortQuality = repetitionMetrics(longIds.slice(0, SEQ_LEN));
  console.log(
    `\n对照: 前 ${SEQ_LEN} 字符 (=训练窗口长度) unique=${shortQuality.uniqueRatio.toFixed(2)}, ` +
      `远段 unique=${lastU.toFixed(2)} => 长度内 vs 长度外的质量差就是 exposure bias 的代价.`,
  );
}

main();
