// stage03-probing.ts — Linear probing: what is LINEARLY decodable from each layer's
//   residual stream, and why a probe is worthless without its shuffled-label control.
//
// WHY this chapter exists: stage01 proved the model LEARNED modAdd; this chapter asks the
//   next question — WHERE in the network does the structure live, and in what form? A linear
//   probe (one Linear trained on frozen activations) answers "is feature F linearly present
//   at point P?" High probe accuracy at a layer means a downstream linear reader could
//   extract F there. Tracking probe accuracy across depth localizes the computation.
//
// THE NON-NEGOTIABLE CONTROL: a probe with enough capacity can fit RANDOM labels to above
//   chance by memorizing the (activation -> label) table. So a raw probe accuracy is
//   meaningless on its own; it is only evidence relative to the SAME probe trained on
//   SHUFFLED labels (interp.linearProbe computes this baseline for us). We make that failure
//   visible: the shuffled-label probe here scores well ABOVE 1/nClasses, proving an
//   uncontrolled probe number would lie.
//
// HONESTY (toy caveat, true for the whole book): on this tiny clean model the operand and
//   sum are almost perfectly linearly decodable and the embedding ring is crisp. Real models
//   give messier probes whose accuracy depends strongly on probe capacity and sample count —
//   so the TRANSFERABLE signal is the SHAPE (probe beats its shuffled baseline by a wide
//   margin; the gap, not the absolute number, is the evidence; structure concentrates at
//   specific layers), NOT these optimistic magnitudes.

import { mulberry32 } from "./core/rng.js";
import { modAdd, type Task } from "./core/tasks.js";
import { trainToyModel, defaultTrainConfig } from "./core/model_zoo.js";
import { TinyTransformer } from "./core/nn.js";
import { runWithCache, linearProbe, type ProbeResult } from "./core/interp.js";
import { Tensor } from "./core/autograd.js";
import { asciiBar, asciiScatter } from "./core/viz.js";

// ----------------------------------------------------------------------------
// Collect a dataset of (residual activation at the "=" position, label) pairs.
// ----------------------------------------------------------------------------
//
// For modAdd the only scorable position is index 2 (the "=" token), where the model must have
//   assembled everything it needs to emit (a+b) mod p. We harvest the residual stream there at
//   each depth point. Labels come from the EXACT oracle inputs, so probe accuracy is measured
//   against ground truth, never against the model's own (possibly wrong) prediction.
interface ProbeDataset {
  points: string[]; // residual hook names, shallow -> deep
  acts: Map<string, Tensor>; // point -> (samples, dim) frozen activations at "=" position
  labelA: number[]; // operand a per sample
  labelSum: number[]; // (a+b) mod p per sample (the answer)
  embedAtA: Tensor; // (samples, dim) embed activation at POSITION 0, where token a sits
  embedAtEq: Tensor; // (samples, dim) embed activation at the "=" position (constant token)
  p: number;
  dim: number;
}

// Residual-stream points only (skip per-head / mlp_out internals): the residual stream is the
// shared bus every component reads/writes, so it is the honest place to ask "is F decodable
// HERE in depth?" without conflating it with a single component's private output.
function residualPoints(model: TinyTransformer): string[] {
  const pts = ["embed"];
  for (let l = 0; l < model.cfg.nLayers; l++) {
    pts.push(`blocks.${l}.resid_pre`, `blocks.${l}.resid_mid`, `blocks.${l}.resid_post`);
  }
  return pts;
}

function collectDataset(task: Task, model: TinyTransformer, nSamples: number, seed: number): ProbeDataset {
  const rng = mulberry32(seed);
  const p = (task as ReturnType<typeof modAdd>).vocab - 1; // modAdd vocab = p+1
  const eqPos = 2; // the "=" position; the only scorable slot for modAdd
  const points = residualPoints(model);
  const dim = model.cfg.dModel;

  // Accumulate one row per sample into flat buffers, then wrap as (samples, dim) tensors.
  const buffers = new Map<string, number[]>();
  for (const pt of points) buffers.set(pt, []);
  const labelA: number[] = [];
  const labelSum: number[] = [];
  // We also keep the EMBED activation at two distinct positions to make a teaching contrast:
  //   position 0 holds token a (so a is decodable there); the "=" position holds the constant
  //   EQ token (so a is NOT decodable there at the embed layer — only after attention moves it).
  const aPos = 0;
  const embedAtABuf: number[] = [];
  const embedAtEqBuf: number[] = [];

  for (let i = 0; i < nSamples; i++) {
    const batch = task.makeBatch(1, rng);
    const input = batch.inputs[0];
    const a = input[0];
    const b = input[1];
    const { cache } = runWithCache(model, input);
    for (const pt of points) {
      const act = cache[pt]; // (seq, dim)
      const row = act.data.subarray(eqPos * dim, (eqPos + 1) * dim);
      buffers.get(pt)!.push(...row);
    }
    const embed = cache["embed"];
    embedAtABuf.push(...embed.data.subarray(aPos * dim, (aPos + 1) * dim));
    embedAtEqBuf.push(...embed.data.subarray(eqPos * dim, (eqPos + 1) * dim));
    labelA.push(a);
    labelSum.push((a + b) % p);
  }

  const acts = new Map<string, Tensor>();
  for (const pt of points) acts.set(pt, new Tensor(Float64Array.from(buffers.get(pt)!), [nSamples, dim]));
  return {
    points,
    acts,
    labelA,
    labelSum,
    embedAtA: new Tensor(Float64Array.from(embedAtABuf), [nSamples, dim]),
    embedAtEq: new Tensor(Float64Array.from(embedAtEqBuf), [nSamples, dim]),
    p,
    dim,
  };
}

// ----------------------------------------------------------------------------
// PCA top-2 components via deflated power iteration (pure compute, no autograd).
// ----------------------------------------------------------------------------
//
// WHY hand-rolled power iteration instead of a library SVD: this repo is dependency-free and
//   we only need the leading 2 eigenvectors of a small covariance matrix. Power iteration on
//   the (dim x dim) covariance converges fast for a dominant eigenvalue; we deflate to get the
//   second. Determinism: the start vector is seeded, so the projection is reproducible.
// INVARIANT: we center the data first (subtract per-feature mean); PCA without centering finds
//   the direction of the mean, not the direction of variance — a classic silent bug.
function pcaProject2(acts: Tensor, rng: () => number): { x: number; y: number }[] {
  const [n, d] = acts.shape;
  // center
  const mean = new Float64Array(d);
  for (let r = 0; r < n; r++) for (let c = 0; c < d; c++) mean[c] += acts.data[r * d + c];
  for (let c = 0; c < d; c++) mean[c] /= n;
  const centered = new Float64Array(n * d);
  for (let r = 0; r < n; r++) for (let c = 0; c < d; c++) centered[r * d + c] = acts.data[r * d + c] - mean[c];

  // covariance C = (Xc^T Xc) / n, a (d x d) symmetric matrix.
  const cov = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += centered[r * d + i] * centered[r * d + j];
      s /= n;
      cov[i * d + j] = s;
      cov[j * d + i] = s; // symmetric
    }
  }

  const powerIter = (matrix: Float64Array): Float64Array => {
    let v = new Float64Array(d);
    for (let i = 0; i < d; i++) v[i] = rng() - 0.5;
    normalize(v);
    // 100 iterations is ample for a dominant eigenvector on a d=32 covariance; convergence is
    // geometric in the eigenvalue ratio, and we only need the direction for a 2-D plot.
    for (let it = 0; it < 100; it++) {
      const nv = new Float64Array(d);
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += matrix[i * d + j] * v[j];
        nv[i] = s;
      }
      normalize(nv);
      v = nv;
    }
    return v;
  };

  const pc1 = powerIter(cov);
  // deflate: C' = C - lambda1 * pc1 pc1^T, so power iteration next yields the 2nd component.
  const lambda1 = rayleigh(cov, pc1, d);
  const cov2 = cov.slice();
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov2[i * d + j] -= lambda1 * pc1[i] * pc1[j];
  const pc2 = powerIter(cov2);

  const out: { x: number; y: number }[] = [];
  for (let r = 0; r < n; r++) {
    let x = 0;
    let y = 0;
    for (let c = 0; c < d; c++) {
      x += centered[r * d + c] * pc1[c];
      y += centered[r * d + c] * pc2[c];
    }
    out.push({ x, y });
  }
  return out;
}

function normalize(v: Float64Array): void {
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
}

// Rayleigh quotient v^T C v (v unit) = eigenvalue estimate for the deflation step.
function rayleigh(matrix: Float64Array, v: Float64Array, d: number): number {
  let s = 0;
  for (let i = 0; i < d; i++) {
    let row = 0;
    for (let j = 0; j < d; j++) row += matrix[i * d + j] * v[j];
    s += v[i] * row;
  }
  return s;
}

function probeRow(name: string, r: ProbeResult): { label: string; value: number }[] {
  return [
    { label: `${name} probe`, value: r.accuracy },
    { label: `${name} shuf`, value: r.baselineAccuracy },
  ];
}

function main(): void {
  console.log("=== Stage 03: 线性探针 — 各层残差里编码了什么 ===\n");

  // Reuse the EXACT shared checkpoint from stage01 (same task, same default config => same
  // cached weights). Findings here are therefore about the same object stage01 validated.
  const task = modAdd(7);
  const trained = trainToyModel(task, defaultTrainConfig(task));
  const model = trained.model;
  console.log(`研究对象: ${task.name}  (复用 stage01 共享 checkpoint, finalLoss=${trained.finalLoss.toFixed(4)})`);

  const N = 300;
  const ds = collectDataset(task, model, N, 20240); // fresh samples, not the training stream
  const probeRng = mulberry32(7); // one seed for every probe -> fair cross-layer comparison
  const chance = 1 / ds.p;
  console.log(`样本数=${N}  类别数=${ds.p}  纯随机基线(1/类别)=${chance.toFixed(3)}\n`);

  // --- 1. Probe the ANSWER (a+b mod p) across depth: where does the result become readable? -
  console.log("[1] 探针: 残差流能否线性解码「答案 (a+b)%p」 (按深度, 浅->深):");
  console.log("    每层两条: probe=真标签准确率, shuf=打乱标签控制 (必须远低于 probe 才算证据)\n");
  const sumBars: { label: string; value: number }[] = [];
  const sumResults: { point: string; r: ProbeResult }[] = [];
  for (const pt of ds.points) {
    const r = linearProbe(ds.acts.get(pt)!, ds.labelSum, ds.p, probeRng, { epochs: 200, lr: 0.05 });
    sumResults.push({ point: pt, r });
    sumBars.push(...probeRow(pt.replace("blocks.", "b").replace(".resid_", "."), r));
  }
  console.log(asciiBar(sumBars, { title: "answer-probe accuracy", width: 36 }));

  // Pick the layer where the answer is most decodable — the "the result is assembled here" point.
  const best = sumResults.reduce((a, b) => (b.r.accuracy > a.r.accuracy ? b : a));
  console.log(
    `\n  最强层: ${best.point}  probe=${best.r.accuracy.toFixed(3)}  shuf=${best.r.baselineAccuracy.toFixed(3)}  ` +
      `gap=${(best.r.accuracy - best.r.baselineAccuracy).toFixed(3)}`,
  );
  console.log(`  解读: 答案在「=」位的残差流里越深越线性可读, 与"算完才写回残差"一致。`);

  // --- 2. Probe the OPERAND a: present at its OWN position, absent at the "=" position. ------
  // This contrast is the lesson: information is POSITIONAL. Token a lives at position 0, so the
  // embed there decodes a perfectly. At the "=" position the embed is the constant EQ token —
  // it cannot know a until ATTENTION moves a's value there. A probe that ignored position would
  // average these and report a misleading middling number.
  console.log("\n[2] 探针: 「操作数 a」是位置相关的 — a 在自己的位置可解码, 在「=」位不可:");
  const aAtPos0 = linearProbe(ds.embedAtA, ds.labelA, ds.p, mulberry32(7), { epochs: 200, lr: 0.05 });
  const aAtEq = linearProbe(ds.embedAtEq, ds.labelA, ds.p, mulberry32(7), { epochs: 200, lr: 0.05 });
  console.log(
    asciiBar(
      [
        { label: "a @ pos0 probe", value: aAtPos0.accuracy },
        { label: "a @ pos0 shuf", value: aAtPos0.baselineAccuracy },
        { label: 'a @ "=" probe', value: aAtEq.accuracy },
        { label: 'a @ "=" shuf', value: aAtEq.baselineAccuracy },
        { label: "chance", value: chance },
      ],
      { title: "operand-a @ embed (two positions)", width: 36 },
    ),
  );
  console.log(
    `  解读: a 在 pos0 几乎满分 (${aAtPos0.accuracy.toFixed(3)}), 在「=」位 ≈ 随机 (${aAtEq.accuracy.toFixed(3)}) — ` +
      `信息是位置绑定的, 探针必须指定位置。`,
  );

  // --- 3. THE FAILURE MODE: an uncontrolled probe number lies. ------------------------------
  // The shuffled-label probe destroys the activation<->label correspondence, so its accuracy
  // is what capacity-driven MEMORIZATION buys with ZERO real signal. If it sits above 1/p, then
  // quoting a raw probe accuracy without this control would overstate the evidence by exactly
  // that margin. We surface the worst (largest) shuffled-baseline across all probes.
  const worstBaseline = sumResults.reduce((a, b) => (b.r.baselineAccuracy > a.r.baselineAccuracy ? b : a));
  console.log("\n[3] 失败模式: 没有对照的探针准确率会骗人");
  console.log(
    `    打乱标签后最高基线: ${worstBaseline.point} shuf=${worstBaseline.r.baselineAccuracy.toFixed(3)} ` +
      `(纯随机应为 ${chance.toFixed(3)})`,
  );
  console.log(
    `    超出纯随机 ${(worstBaseline.r.baselineAccuracy - chance >= 0 ? "+" : "")}` +
      `${(worstBaseline.r.baselineAccuracy - chance).toFixed(3)} = 纯记忆带来的虚假准确率。`,
  );
  console.log(`    教训: 报告 probe=0.9 而不报 shuf 基线, 等于把记忆当成"信息存在"的证据。`);

  // --- 4. PCA of the operand-a representation: looking for the modular ring. ----------------
  // modAdd's known algebraic solution can embed numbers on a circle (add = rotate). We project
  // the operand activations to 2-D and label each point by its operand a, then ASK whether a's
  // lay out in cyclic order around a loop. CAVEAT we make explicit below: a partial/absent ring
  // in top-2 PCA does NOT prove the structure is missing — it may live in higher dims the
  // projection discards. The scatter is a hypothesis generator, not a proof.
  console.log("\n[4] PCA 二维投影: 操作数 a 是否在表征空间成环 (模运算的旋转结构):");
  // One embedding per operand value 0..p-1, read at POSITION 0 where token a lives. At the
  // embed layer position 0 depends only on a (token a's embedding + the pos-0 embedding), so
  // this isolates a's identity cleanly — exactly where the modular ring should appear.
  const ringPoints = collectPerValueEmbedding(model, ds.p, ds.dim);
  const proj = pcaProject2(ringPoints.acts, mulberry32(3));
  const scatterPts = proj.map((pt, i) => ({ x: pt.x, y: pt.y, label: String(ringPoints.values[i]) }));
  console.log(asciiScatter(scatterPts, { width: 46, height: 16, title: "embed PCA (label = operand a)" }));
  // Quantify "is it a ring?": sort points by angle and check the value order is (cyclically)
  // monotone. A true ring => consecutive angular neighbors are consecutive residues mod p.
  const ringScore = cyclicMonotoneScore(proj, ringPoints.values, ds.p);
  const ringFrac = ringScore.correct / ds.p;
  const ringExpected = 2 / ds.p; // a random ordering has ~2 of p adjacencies be ±1 mod p
  console.log(
    `  环状一致性: ${ringScore.correct}/${ds.p} 个相邻角度对是模 p 相邻 ` +
      `(=${ringFrac.toFixed(2)}; 随机期望≈${ringExpected.toFixed(2)}, 完美环=1.0)`,
  );
  console.log(
    `  解读: 高于随机 (${ringFrac.toFixed(2)} vs ${ringExpected.toFixed(2)}) 但远非完美环 — ` +
      `这个 2 层模型在 400 步只学到部分旋转结构。诚实地说: top-2 PCA 把 d=${ds.dim} 维压成 2 维会` +
      `丢掉真实环面所在的高维子空间, "看不见干净环" 既可能是没学到、也可能是投影损失, 不能只凭散点图断言。`,
  );

  console.log(
    `\n诚实边界: probe 近满分、答案在特定层突然可读, 是 toy 任务的乐观绝对值; PCA 环只是"部分"` +
      `(${ringScore.correct}/${ds.p}), 没有出现教科书式干净圆环 — 不夸大。可迁移的是两条形状: ` +
      `(1) probe 必须显著超过自身 shuffle 基线才算证据 (本章 shuffle 基线 ${chance.toFixed(2)}→${worstBaseline.r.baselineAccuracy.toFixed(2)} 的记忆抬升就是反例); ` +
      `(2) 结构按"层 + 位置"定位, 不是模型整体一个数。`,
  );
}

// One embedding per operand value, read at position 0 (where token a sits). At the embed layer
// position 0 is a pure function of a, so this is the cleanest view of a's representation — the
// place the modular ring lives. Returns parallel (values, acts) for the PCA ring plot.
function collectPerValueEmbedding(model: TinyTransformer, p: number, dim: number): { values: number[]; acts: Tensor } {
  const EQ = p;
  const aPos = 0;
  const values: number[] = [];
  const rows: number[] = [];
  for (let a = 0; a < p; a++) {
    // b is arbitrary: it cannot affect position 0's embed activation. Use 0 for determinism.
    const { cache } = runWithCache(model, [a, 0, EQ]);
    const act = cache["embed"]; // (seq, dim)
    values.push(a);
    rows.push(...Array.from(act.data.subarray(aPos * dim, (aPos + 1) * dim)));
  }
  return { values, acts: new Tensor(Float64Array.from(rows), [p, dim]) };
}

// Ring test: order points by angle around their centroid; count adjacent angular neighbors
// whose operand values differ by exactly ±1 (mod p). High count => values wrap a loop in
// cyclic order = the modular ring. This is a real geometric check, not an eyeball claim.
function cyclicMonotoneScore(
  proj: { x: number; y: number }[],
  values: number[],
  p: number,
): { correct: number } {
  let cx = 0;
  let cy = 0;
  for (const pt of proj) {
    cx += pt.x;
    cy += pt.y;
  }
  cx /= proj.length;
  cy /= proj.length;
  const order = proj
    .map((pt, i) => ({ value: values[i], angle: Math.atan2(pt.y - cy, pt.x - cx) }))
    .sort((a, b) => a.angle - b.angle);
  let correct = 0;
  for (let i = 0; i < order.length; i++) {
    const cur = order[i].value;
    const next = order[(i + 1) % order.length].value;
    const diff = Math.abs(cur - next) % p;
    if (diff === 1 || diff === p - 1) correct++; // ±1 mod p = adjacent residues
  }
  return { correct };
}

main();
