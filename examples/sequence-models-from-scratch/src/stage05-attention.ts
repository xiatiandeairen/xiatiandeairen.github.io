// stage05-attention.ts — Single-head scaled dot-product attention vs. an LSTM baseline.
//
// CHAPTER THESIS: attention replaces the RNN's sequential memory bottleneck with a
//   DIRECT, content-addressed lookup. An RNN/LSTM must squeeze everything it might later
//   need through a fixed-width hidden state that is overwritten every timestep; on a copy
//   task the payload token at step 0 has to survive `delay` overwrites to reach the answer
//   region. Attention skips that: the answer position computes a similarity score against
//   EVERY earlier position and reads the matching value in ONE hop. No information squeeze,
//   no vanishing path. That is why one O(n^2) operation displaced the recurrent stack.
//
// WHAT THIS FILE PROVES, with honest numbers:
//   1. On a copy task long enough to strain a small LSTM, attention reaches higher
//      answer-region accuracy in fewer steps (loss curves + accuracy printed side by side).
//   2. The attention weight matrix, drawn as a heatmap, shows the answer positions wiring
//      DIRECTLY back to the payload positions at the front — the "direct connection".
//   3. THE COST WALL (the failure mode that motivates ch.06): doubling sequence length
//      ~quadruples attention's forward MACs and wall-clock. We fit a log-log slope to the
//      MEASURED times and to the MAC counts; both land near 2.0 (O(n^2)). A recurrent scan
//      would land near 1.0. This is the concrete pain ch.06 (SSM) sets out to remove.
//
// HONESTY NOTES:
//   - This is a TOY: tiny vocab, clean signal, one batch reused. Absolute accuracies are
//     optimistic. What transfers is the RELATIVE story (attention > small LSTM on long
//     copy) and the SCALING SHAPE (slope ~2 for attention), not the specific percentages.
//   - Wall-clock is really measured (core timeit, warmup discarded). The fitted slope is
//     labeled (measured) vs the MAC slope (analytic). Single-machine ms is only meaningful
//     relative to itself, which is exactly what a slope captures.
//
// DETERMINISM: every random draw threads a seeded Rng; separate seeds for data vs init so
//   the two streams don't couple. Same seed => identical output.
//
// REUSE: builds only on core primitives (Tensor, Linear, Embedding, Adam, copyTask, plot,
//   metrics). Does NOT import any stageNN file (those run main() on import). The LSTM cell
//   here is a compact local baseline, not the pedagogical LSTM of ch.03.

import { Tensor } from "./core/tensor.js";
import { Linear, Embedding, collectParams } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { makeRng, type Rng } from "./core/prng.js";
import { copyTask } from "./core/data.js";
import { argmax, accuracy, countMACs, timeit, perplexity } from "./core/metrics.js";
import { lineChart, heatmap, bar } from "./core/plot.js";

// ---------------------------------------------------------------------------
// Single-head causal self-attention.
//
// One sequence at a time: input x is [T, dim] (T timesteps, model dim). We project to
// Q,K,V each [T, dh], score = Q Kᵀ / sqrt(dh) giving [T, T], apply a CAUSAL mask (a query
// at step t may only attend to keys at steps <= t — required for next-token prediction so
// position t never peeks at its own answer), softmax over the key axis, then weights @ V.
//
// WHY 2D-only matmul is fine here: there is no batch axis inside one sequence, so every
// matmul is genuinely 2D. We loop over sequences in the batch instead of batching the
// matmul (the core engine deliberately has no batched matmul — see tensor.ts).
//
// WHY we ADD a mask of -1e9 to scores instead of zeroing post-softmax: zeroing after
// softmax breaks the normalization (rows no longer sum to 1) and corrupts gradients.
// Adding a large negative BEFORE softmax sends those entries to ~0 probability while
// keeping the row a valid distribution and the gradient correct.
// ---------------------------------------------------------------------------
class SelfAttention {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  oProj: Linear;
  private dh: number;
  private maskCache = new Map<number, Tensor>(); // T -> additive causal mask [T,T]

  constructor(dim: number, headDim: number, rng: Rng) {
    this.dh = headDim;
    // bias:false on Q/K/V is conventional — a constant bias adds nothing a softmax can't
    // absorb via the value path, and keeps the param count comparable to the baseline.
    this.qProj = new Linear(dim, headDim, rng, { bias: false });
    this.kProj = new Linear(dim, headDim, rng, { bias: false });
    this.vProj = new Linear(dim, headDim, rng, { bias: false });
    this.oProj = new Linear(headDim, dim, rng);
  }

  // Additive causal mask: 0 on/below the diagonal, -1e9 above. Cached per T because the
  // mask is a constant (no params, no grad) and rebuilding it every forward is wasteful.
  private causalMask(T: number): Tensor {
    let m = this.maskCache.get(T);
    if (!m) {
      const data = new Float64Array(T * T);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) data[i * T + j] = j > i ? -1e9 : 0;
      m = new Tensor(data, [T, T]);
      this.maskCache.set(T, m);
    }
    return m;
  }

  // x: [T, dim] -> context: [T, dim]. Also returns the [T,T] weight matrix for plotting.
  forward(x: Tensor): { out: Tensor; weights: Tensor } {
    const T = x.shape[0];
    const q = this.qProj.forward(x); // [T, dh]
    const k = this.kProj.forward(x); // [T, dh]
    const v = this.vProj.forward(x); // [T, dh]
    // scaled scores: 1/sqrt(dh) keeps the dot-product variance ~1 so softmax doesn't
    // saturate as dh grows (the "scaled" in scaled dot-product attention).
    const scores = q.matmul(k.transpose()).scale(1 / Math.sqrt(this.dh)); // [T, T]
    const masked = scores.add(this.causalMask(T));
    const weights = masked.softmax(); // row t = distribution over keys 0..t
    const ctx = weights.matmul(v); // [T, dh]
    return { out: this.oProj.forward(ctx), weights };
  }

  params(): Tensor[] {
    return collectParams([this.qProj, this.kProj, this.vProj, this.oProj]);
  }
}

// Attention classifier for the copy task: embed -> 1 attention block (residual) -> readout.
// A single head + single layer is enough to solve copy because the task is pure routing:
// "at an answer position, find the payload position with the matching slot and copy it".
class AttentionModel {
  embed: Embedding;
  attn: SelfAttention;
  readout: Linear;
  private dim: number;
  private dh: number;
  private vocab: number;

  constructor(vocab: number, dim: number, headDim: number, rng: Rng) {
    this.vocab = vocab;
    this.dim = dim;
    this.dh = headDim;
    this.embed = new Embedding(vocab, dim, rng);
    this.attn = new SelfAttention(dim, headDim, rng);
    this.readout = new Linear(dim, vocab, rng);
  }

  // ids: one sequence (length T). Returns logits [T, vocab] + attention weights [T,T].
  forward(ids: number[]): { logits: Tensor; weights: Tensor } {
    const e = this.embed.forward(ids); // [T, dim]
    const { out, weights } = this.attn.forward(e);
    const h = e.add(out); // residual: keep token identity + attended context
    return { logits: this.readout.forward(h), weights };
  }

  params(): Tensor[] {
    return [...this.embed.params(), ...this.attn.params(), ...this.readout.params()];
  }

  // Forward MAC estimate for one sequence of length T (dominant terms only). Used to argue
  // the O(n^2) shape against measured wall-clock. countMACs.attention(T, dh) is the QKᵀ +
  // (·)V cost; the projections are O(T) and shown separately so the n^2 term dominates.
  forwardMACs(T: number): number {
    return this.projMACs(T) + this.attentionCoreMACs(T) + countMACs.linear(T, this.dim, this.vocab);
  }
  // The O(n) part: Q/K/V/O projections + readout are all per-token Linear, linear in T.
  projMACs(T: number): number {
    return countMACs.linear(T, this.dim, this.dh) * 3 + countMACs.linear(T, this.dh, this.dim);
  }
  // The O(n^2) part: QKᵀ then (softmax)·V, both ~T^2*dh. THIS is the term that walls.
  attentionCoreMACs(T: number): number {
    return countMACs.attention(T, this.dh);
  }
}

// ---------------------------------------------------------------------------
// Compact LSTM baseline (the thing attention is competing against).
//
// This is a local, minimal LSTM cell — NOT the pedagogical ch.03 implementation (we must
// not import a stage file). Standard gates: input/forget/output + candidate, all from a
// single concat([h, x]) @ W. We deliberately keep the hidden width SMALL so the copy task
// strains its fixed-capacity memory — that strain is the comparison's whole point.
//
// The recurrence is the bottleneck attention removes: the payload from step 0 must be
// re-encoded into h at every one of the `delay` intermediate steps to still be present at
// the answer region. A small h cannot hold k payload slots through a long delay.
// ---------------------------------------------------------------------------
class LSTMModel {
  embed: Embedding;
  Wf: Linear;
  Wi: Linear;
  Wg: Linear;
  Wo: Linear;
  readout: Linear;
  private hdim: number;

  constructor(vocab: number, dim: number, hdim: number, rng: Rng) {
    this.hdim = hdim;
    this.embed = new Embedding(vocab, dim, rng);
    const gateIn = dim + hdim; // gates read concat([h_prev, x_t])
    this.Wf = new Linear(gateIn, hdim, rng);
    this.Wi = new Linear(gateIn, hdim, rng);
    this.Wg = new Linear(gateIn, hdim, rng);
    this.Wo = new Linear(gateIn, hdim, rng);
    this.readout = new Linear(hdim, vocab, rng);
  }

  forward(ids: number[]): { logits: Tensor } {
    const T = ids.length;
    const e = this.embed.forward(ids); // [T, dim]
    let h = Tensor.zeros([1, this.hdim]);
    let c = Tensor.zeros([1, this.hdim]);
    const hs: Tensor[] = [];
    for (let t = 0; t < T; t++) {
      const x = e.slice(0, t, t + 1); // [1, dim]
      const z = Tensor.concat([h, x], 1); // [1, dim+hdim]
      const f = this.Wf.forward(z).sigmoid();
      const i = this.Wi.forward(z).sigmoid();
      const g = this.Wg.forward(z).tanh();
      const o = this.Wo.forward(z).sigmoid();
      c = f.mul(c).add(i.mul(g)); // cell update: keep + write
      h = o.mul(c.tanh()); // hidden = gated cell readout
      hs.push(h);
    }
    const H = Tensor.concat(hs, 0); // [T, hdim]
    return { logits: this.readout.forward(H) };
  }

  params(): Tensor[] {
    return [
      ...this.embed.params(),
      ...this.Wf.params(),
      ...this.Wi.params(),
      ...this.Wg.params(),
      ...this.Wo.params(),
      ...this.readout.params(),
    ];
  }
}

// ---------------------------------------------------------------------------
// Shared training / evaluation helpers (functional core: pure given the model + data).
// ---------------------------------------------------------------------------

function paramCount(params: Tensor[]): number {
  return params.reduce((s, p) => s + p.size, 0);
}

// The copy task's loss must be MASKED to the answer region: positions outside it have
// target 0 (blank) and predicting blank everywhere would otherwise dominate the loss and
// let the model "win" without ever copying. We compute CE only over answer positions.
// answerStart = k + delay + 1 (first answer index); answer region is the last k positions.
function answerRange(k: number, delay: number): { start: number; len: number } {
  return { start: k + delay + 1, len: k };
}

// One full-batch training step over all sequences. Returns mean answer-region CE (nats).
// We accumulate per-sequence losses, average, backward once. WHY full-batch: the toy set
// is tiny and a single deterministic batch keeps the loss curve smooth and reproducible.
function trainStep(
  model: { forward(ids: number[]): { logits: Tensor }; params(): Tensor[] },
  X: number[][],
  Y: number[][],
  ans: { start: number; len: number },
  opt: Adam,
): number {
  opt.zeroGrad();
  const losses: Tensor[] = [];
  for (let n = 0; n < X.length; n++) {
    const { logits } = model.forward(X[n]); // [T, vocab]
    // slice the answer rows, CE against the answer targets only.
    const ansLogits = logits.slice(0, ans.start, ans.start + ans.len); // [k, vocab]
    const tgt = Y[n].slice(ans.start, ans.start + ans.len);
    losses.push(ansLogits.crossEntropy(tgt)); // mean CE over the k answer rows
  }
  const loss = Tensor.concat(losses.map((l) => l.reshape([1, 1])), 0).mean(); // mean over batch
  loss.backward();
  opt.step();
  return loss.data[0];
}

// Answer-region accuracy over the dataset (fraction of answer tokens predicted exactly).
function evalAccuracy(
  model: { forward(ids: number[]): { logits: Tensor } },
  X: number[][],
  Y: number[][],
  ans: { start: number; len: number },
  vocab: number,
): number {
  const preds: number[] = [];
  const tgts: number[] = [];
  for (let n = 0; n < X.length; n++) {
    const { logits } = model.forward(X[n]);
    for (let i = 0; i < ans.len; i++) {
      const row = logits.data.subarray((ans.start + i) * vocab, (ans.start + i) * vocab + vocab);
      preds.push(argmax(row));
      tgts.push(Y[n][ans.start + i]);
    }
  }
  return accuracy(preds, tgts);
}

// Least-squares slope of log(y) vs log(x): the scaling exponent. For y ~ x^p, this returns
// p. We use it to turn "doubling n quadruples cost" into a single honest number (~2.0 for
// O(n^2)). Standard closed-form: slope = cov(logx,logy)/var(logx).
function logLogSlope(xs: number[], ys: number[]): number {
  const lx = xs.map(Math.log);
  const ly = ys.map(Math.log);
  const n = lx.length;
  const mx = lx.reduce((a, b) => a + b, 0) / n;
  const my = ly.reduce((a, b) => a + b, 0) / n;
  let cov = 0;
  let varx = 0;
  for (let i = 0; i < n; i++) {
    cov += (lx[i] - mx) * (ly[i] - my);
    varx += (lx[i] - mx) ** 2;
  }
  return cov / varx;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
function main(): void {
  const SEED_DATA = 1234;
  const SEED_ATTN = 7;
  const SEED_LSTM = 7; // same init seed so the comparison isn't a lucky-init artifact

  // Task: copy k=3 payload tokens across a delay of 12 blanks. Dependency span = k+delay+1
  // = 16 steps — long enough that a small LSTM's fixed memory struggles to hold 3 slots
  // through 12 overwrites, while attention reads them back directly.
  const k = 3;
  const delay = 12;
  const symbols = 6;
  const count = 24;
  const data = copyTask(makeRng(SEED_DATA), { count, k, delay, symbols });
  const ans = answerRange(k, delay);
  const T = k + delay + 1 + k;
  const vocab = data.vocabSize;

  console.log("=".repeat(70));
  console.log("第 05 章 — 注意力 vs LSTM:为什么一个 O(n²) 操作取代了 RNN");
  console.log("=".repeat(70));
  console.log(data.describe());
  console.log(`answer region: positions [${ans.start}, ${ans.start + ans.len}) (last ${k} steps)`);
  console.log(`uniform-guess accuracy floor ≈ 1/payload_symbols = ${(1 / symbols).toFixed(3)}`);
  console.log();

  // --- build models (matched dims for a fair-ish comparison) ---
  const dim = 32;
  const headDim = 32;
  const hdim = 24; // small on purpose: this is the memory bottleneck under test
  const attn = new AttentionModel(vocab, dim, headDim, makeRng(SEED_ATTN));
  const lstm = new LSTMModel(vocab, dim, hdim, makeRng(SEED_LSTM));
  const attnParams = attn.params();
  const lstmParams = lstm.params();

  console.log("--- model sizes ---");
  console.log(`attention (dim=${dim}, headDim=${headDim}): ${paramCount(attnParams)} params`);
  console.log(`LSTM      (dim=${dim}, hidden=${hdim}):    ${paramCount(lstmParams)} params`);
  console.log();

  // --- train both ---
  const EPOCHS = 120;
  const attnOpt = new Adam(attnParams, { lr: 5e-3 });
  const lstmOpt = new Adam(lstmParams, { lr: 5e-3 });
  const attnLoss: number[] = [];
  const lstmLoss: number[] = [];
  for (let ep = 0; ep < EPOCHS; ep++) {
    attnLoss.push(trainStep(attn, data.X, data.Y, ans, attnOpt));
    lstmLoss.push(trainStep(lstm, data.X, data.Y, ans, lstmOpt));
  }

  const attnAcc = evalAccuracy(attn, data.X, data.Y, ans, vocab);
  const lstmAcc = evalAccuracy(lstm, data.X, data.Y, ans, vocab);

  console.log(`--- training loss over ${EPOCHS} epochs (answer-region CE, nats) ---`);
  console.log(lineChart([attnLoss, lstmLoss], { labels: ["attention", "LSTM"], height: 12, width: 60 }));
  console.log();
  console.log("--- final answer-region accuracy (copy span = 16 steps) ---");
  console.log(
    bar([
      { label: "attention", value: +attnAcc.toFixed(4) },
      { label: "LSTM", value: +lstmAcc.toFixed(4) },
    ]),
  );
  console.log(`attention final loss=${attnLoss[EPOCHS - 1].toFixed(4)} (ppl=${perplexity(attnLoss[EPOCHS - 1]).toFixed(2)})`);
  console.log(`LSTM      final loss=${lstmLoss[EPOCHS - 1].toFixed(4)} (ppl=${perplexity(lstmLoss[EPOCHS - 1]).toFixed(2)})`);
  console.log(
    `=> attention is ${(attnAcc - lstmAcc >= 0 ? "+" : "")}${((attnAcc - lstmAcc) * 100).toFixed(1)} pts vs LSTM on this span (toy numbers; relative gap is the signal)`,
  );
  console.log();

  // --- the "direct connection" heatmap ---
  // Pick the trained attention weights for one sequence. We expect the ANSWER rows (last k)
  // to put their mass on the PAYLOAD columns (first k) — a bright block linking tail->head,
  // i.e. the answer position reading the payload directly, no recurrence in between.
  const sample = 0;
  const { weights } = attn.forward(data.X[sample]);
  const W: number[][] = [];
  for (let i = 0; i < T; i++) {
    const r: number[] = [];
    for (let j = 0; j < T; j++) r.push(weights.data[i * T + j]);
    W.push(r);
  }
  console.log("--- attention weight matrix, sequence #0 (row=query step, col=key step) ---");
  console.log(`payload at cols 0..${k - 1}, delim at col ${k + delay}, answer rows ${ans.start}..${ans.start + ans.len - 1}`);
  console.log(`input ids : [${data.X[sample].join(",")}]`);
  console.log(heatmap(W));
  // Quantify the "direct connection": for each answer row, how much probability mass lands
  // on the payload columns (0..k-1)? High mass = the model literally points back to the
  // source tokens. This is the heatmap claim turned into a number.
  let massOnPayload = 0;
  for (let i = 0; i < ans.len; i++) {
    const row = ans.start + i;
    for (let j = 0; j < k; j++) massOnPayload += weights.data[row * T + j];
  }
  massOnPayload /= ans.len;
  console.log(
    `mean attention mass an answer row places on the k payload columns: ${(massOnPayload * 100).toFixed(1)}% ` +
      `(of 1.0; high => direct tail->head routing)`,
  );
  console.log();

  // --- THE COST WALL: forward cost vs sequence length (motivates ch.06) ---
  // For each n we MEASURE forward wall-clock (core timeit, warmup discarded) and COMPUTE
  // the analytic MAC count split into its O(n) part (projections+readout) and O(n^2) part
  // (the QKᵀ/AV attention core). The per-doubling ratios are the most direct evidence:
  // a pure O(n^2) term multiplies by 4 each doubling, a pure O(n) term by 2.
  //
  // HONESTY about the slope: the FULL forward is core(n^2) + proj(n) + readout(n). At small
  // n the linear terms dilute the slope below 2, so a single fit over [16..256] understates
  // it. We therefore report (a) the isolated attention-core slope (must be ~2.0 by
  // construction — sanity that countMACs is the n^2 it claims), and (b) the full-forward
  // slope over the LARGE-n TAIL where the n^2 term has taken over. Reporting a misleading
  // single small-n slope and calling it "~2.0" would be dishonest, so we don't.
  console.log("--- failure mode: attention forward cost scales toward O(n^2) ---");
  const lengths = [32, 64, 128, 256, 512, 1024];
  const measuredMs: number[] = [];
  const coreMACs: number[] = [];
  const projMACs: number[] = [];
  // Reuse one model; cost depends on T only, not on the weights' values.
  const costModel = new AttentionModel(vocab, dim, headDim, makeRng(SEED_ATTN));
  for (const n of lengths) {
    const seq: number[] = []; // content is irrelevant to cost; just valid ids
    const rng = makeRng(99);
    for (let t = 0; t < n; t++) seq.push(rng.randint(0, vocab));
    const { perRepMs } = timeit(() => void costModel.forward(seq), 20);
    measuredMs.push(perRepMs);
    coreMACs.push(costModel.attentionCoreMACs(n));
    projMACs.push(costModel.projMACs(n));
  }
  console.log("   n     fwd ms (measured)   core MACs (n^2)   proj MACs (n)   ms x/double   coreMAC x/double");
  for (let i = 0; i < lengths.length; i++) {
    const msR = i === 0 ? "—" : (measuredMs[i] / measuredMs[i - 1]).toFixed(2) + "x";
    const coreR = i === 0 ? "—" : (coreMACs[i] / coreMACs[i - 1]).toFixed(2) + "x";
    console.log(
      `  ${String(lengths[i]).padStart(4)}   ${measuredMs[i].toFixed(4).padStart(10)}      ` +
        `${String(coreMACs[i]).padStart(11)}   ${String(projMACs[i]).padStart(11)}   ` +
        `${msR.padStart(9)}     ${coreR.padStart(9)}`,
    );
  }
  // Tail = last 3 points (n >= 256), where the n^2 core dominates the linear terms.
  const tail = 3;
  const tailMsSlope = logLogSlope(lengths.slice(-tail), measuredMs.slice(-tail));
  const coreSlope = logLogSlope(lengths, coreMACs);
  const fullForward = lengths.map((n) => costModel.forwardMACs(n));
  const fullSlopeAll = logLogSlope(lengths, fullForward);
  const fullSlopeTail = logLogSlope(lengths.slice(-tail), fullForward.slice(-tail));
  console.log();
  console.log(`attention-core MAC slope (all n):          ${coreSlope.toFixed(2)}  (must be ~2.0 by construction)`);
  console.log(`full-forward MAC slope (all n):            ${fullSlopeAll.toFixed(2)}  (< 2: O(n) terms dilute small n)`);
  console.log(`full-forward MAC slope (tail n>=256):      ${fullSlopeTail.toFixed(2)}  (-> 2 as n^2 takes over)`);
  console.log(`measured wall-clock slope (tail n>=256):   ${tailMsSlope.toFixed(2)}  (real ms; JIT/GC noise ~ this)`);
  console.log();

  // Make the O(n) vs O(n^2) contrast a single number: at the longest n, how many MACs would
  // an equivalent recurrent scan (ch.06) need vs dense attention? countMACs.scan(n,s,d) is
  // ~n*s*d (linear in n). We use stateDim ~= headDim so the comparison is apples-to-apples.
  const big = lengths[lengths.length - 1];
  const scanMACsBig = countMACs.scan(big, headDim, dim);
  const attnMACsBig = costModel.attentionCoreMACs(big);
  console.log("--- O(n) scan vs O(n^2) attention at the longest n (the ch.06 motivation) ---");
  console.log(
    bar([
      { label: `attention core n=${big}`, value: attnMACsBig },
      { label: `linear scan  n=${big}`, value: scanMACsBig },
    ]),
  );
  console.log(
    `attention needs ${(attnMACsBig / scanMACsBig).toFixed(1)}x the MACs of a linear scan at n=${big}, ` +
      `and the ratio itself grows ~n. That widening gap is the wall ch.06 (SSM) removes.`,
  );
}

main();
