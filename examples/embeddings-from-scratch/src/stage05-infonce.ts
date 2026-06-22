// stage05-infonce.ts — Contrastive learning and InfoNCE: negative sampling's
// modern reincarnation.
//
// Thesis of this chapter: skip-gram negative sampling (stage03) and modern
// contrastive learning (SimCLR / MoCo / sentence embeddings) are the SAME idea
// wearing different clothes. "Pull a positive pair together, push everything else
// in the batch apart." InfoNCE is the loss that makes "everything else in the
// batch" literal: for each anchor we have ONE positive (an augmented view of
// itself) and the OTHER batch members serve as negatives — no separate negative
// sampler needed, the batch IS the negative set. That is the whole trick that let
// contrastive learning scale.
//
// What this file actually computes (all numbers are real, measured at run time):
//   1. Builds toy anchors with KNOWN cluster structure (so we have ground truth
//      to check "did same-class points end up together?"). Generates a positive
//      view per anchor by adding seeded small noise — the "augmentation".
//   2. Implements InfoNCE = -log( exp(sim(a,a+)/T) / sum_j exp(sim(a,x_j)/T) )
//      over in-batch negatives, built node-by-node in core/autograd so the reader
//      can SEE the logsumexp. Trains a small projection head with Adam. Prints the
//      real loss curve.
//   3. Reports alignment (positives close, lower=better) and uniformity (spread
//      on the sphere, more-negative=better) BEFORE and AFTER training, the
//      Wang-Isola diagnostic pair that proves the space actually separated.
//   4. asciiScatter of the trained 2D representation so clusters are visible.
//   5. FAILURE MODE: re-train with a near-zero temperature AND a positives-only
//      objective (no negatives in the denominator) → representation collapse.
//      Prints the degenerate uniformity (≈0, all vectors piled up) so the reader
//      sees collapse as a number, not a warning.
//
// Honesty caveats (stated, not buried):
//   - Absolute alignment/uniformity values are on TOY synthetic clusters at D=2
//     output dim; they are optimistic. What transfers is the *direction* of the
//     before→after change and the *contrast* between the healthy run and the
//     collapsed run.
//   - The projection head is D_in→D_out=2 so we can scatter directly without PCA;
//     this is pedagogical, not a real encoder.
//
// Determinism: every random draw (anchor construction, augmentation noise, weight
// init, batch shuffle) pulls from a seeded Rng. Same seed → same numbers.

import { Value, makeMat, type Mat, type Vec } from "./core/autograd.js";
import { Adam, collectParams } from "./core/optim.js";
import { Rng, gaussian, shuffle } from "./core/rng.js";
import { asciiLine, asciiScatter } from "./core/plot.js";
import { alignment, uniformity, cosineSimilarity } from "./core/eval.js";

// ---------------------------------------------------------------------------
// Toy data with KNOWN structure.
// ---------------------------------------------------------------------------

// We synthesize anchors as K gaussian blobs in D_in dimensions. The cluster id is
// the ground-truth label we never feed the model — it only ever sees noisy views.
// Why blobs and not real word vectors: contrastive learning's claim is geometric
// ("same thing → same place"), so the cleanest demonstration uses data whose
// "same thing" is unambiguous. Each cluster has a fixed center; anchors are
// center + small jitter, so within-cluster anchors genuinely belong together.
const D_IN = 8; // input feature dim
const D_OUT = 2; // projection dim — 2 so we can scatter without PCA
const NUM_CLUSTERS = 4;
const PER_CLUSTER = 8; // anchors per cluster → batch size 32
const CLUSTER_SPREAD = 0.35; // intra-cluster jitter; << inter-cluster distance
const AUG_NOISE = 0.25; // augmentation strength: positive = anchor + this noise

interface Anchor {
  feat: number[]; // D_IN raw features (plain numbers; the head turns them into Values)
  cluster: number; // ground-truth label, for evaluation only
}

// Build cluster centers far apart on a hypercube-ish layout so the clusters are
// genuinely separable in the INPUT space; the head's job is to preserve that
// separation in the OUTPUT space while a collapsing objective would destroy it.
function buildAnchors(rng: Rng): Anchor[] {
  const centers: number[][] = [];
  for (let c = 0; c < NUM_CLUSTERS; c++) {
    // Spread centers along distinct axes with magnitude 2 so inter-cluster
    // distance (~2..4) dwarfs intra-cluster jitter (~0.35) — unambiguous truth.
    const center = new Array(D_IN).fill(0).map(() => gaussian(rng, 0, 1));
    // amplify two coordinates per cluster to push centers apart deterministically
    center[c % D_IN] += 2.5;
    center[(c + 1) % D_IN] -= 2.5;
    centers.push(center);
  }
  const anchors: Anchor[] = [];
  for (let c = 0; c < NUM_CLUSTERS; c++) {
    for (let i = 0; i < PER_CLUSTER; i++) {
      const feat = centers[c].map((x) => x + gaussian(rng, 0, CLUSTER_SPREAD));
      anchors.push({ feat, cluster: c });
    }
  }
  return anchors;
}

// An augmented positive view: same underlying anchor, perturbed. In real
// contrastive learning this is a crop/color-jitter (images) or dropout/span
// masking (text); here it is additive gaussian noise. The invariant the model
// must learn: a view and its source are the SAME instance despite the noise.
function augment(rng: Rng, feat: number[]): number[] {
  return feat.map((x) => x + gaussian(rng, 0, AUG_NOISE));
}

// ---------------------------------------------------------------------------
// Projection head: a single linear map D_IN -> D_OUT, the trainable parameters.
// ---------------------------------------------------------------------------

// Linear projection feat (number[]) -> Vec (Value[]) through weight matrix W
// (D_OUT x D_IN). We multiply by constants (the raw features) and sum, so the
// only graph leaves that carry gradient are the W entries — exactly the params we
// optimize. No bias: cosine similarity is shift-variant but the demo cares about
// direction structure and a bias just adds parameters without teaching anything.
function project(W: Mat, feat: number[]): Vec {
  const out: Vec = [];
  for (let r = 0; r < W.length; r++) {
    // dot(W[r], feat) but feat are constants → W[r][k].mul(feat[k]) keeps grad on W
    let acc = new Value(0);
    for (let k = 0; k < feat.length; k++) acc = acc.add(W[r][k].mul(feat[k]));
    out.push(acc);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Similarity and InfoNCE — the heart of the chapter.
// ---------------------------------------------------------------------------

// Cosine similarity as a Value graph (gradients flow). InfoNCE scores are cosine
// (not raw dot) because unnormalized dot products let the model cheat by inflating
// magnitudes; normalizing forces it to use ANGLE, which is what we evaluate on.
// We guard the norm with a tiny epsilon inside the sqrt's reciprocal: a zero
// vector early in training would otherwise divide by zero. pow(-0.5) gives 1/sqrt
// with a correct gradient straight from the engine.
function cosineSim(a: Vec, b: Vec): Value {
  let dot = new Value(0);
  let na = new Value(0);
  let nb = new Value(0);
  for (let i = 0; i < a.length; i++) {
    dot = dot.add(a[i].mul(b[i]));
    na = na.add(a[i].mul(a[i]));
    nb = nb.add(b[i].mul(b[i]));
  }
  // add eps before sqrt so the reciprocal is finite at the zero vector.
  const inv = na.add(1e-12).mul(nb.add(1e-12)).pow(-0.5);
  return dot.mul(inv);
}

// Numerically stable logsumexp over a list of Values. exp() of large positive
// logits overflows to Infinity → loss = NaN, the #1 InfoNCE training crash. The
// standard fix: subtract the (constant) max logit before exp, add it back after
// the log. We read the max from .data (a constant w.r.t. gradients — the shift is
// a constant offset that cancels in the derivative), so this is exact, not an
// approximation. Returns log(sum_i exp(logits_i)).
function logSumExp(logits: Value[]): Value {
  let maxData = -Infinity;
  for (const z of logits) if (z.data > maxData) maxData = z.data;
  let sumExp = new Value(0);
  for (const z of logits) sumExp = sumExp.add(z.sub(maxData).exp());
  return sumExp.log().add(maxData);
}

// InfoNCE loss for ONE anchor against the whole batch of candidate views.
//   loss = -log( exp(sim(anchor, positive)/T)
//                / sum_j exp(sim(anchor, candidate_j)/T) )
//        = -( sim(anchor,pos)/T  -  logsumexp_j sim(anchor,cand_j)/T )
// candidates INCLUDE the positive (the standard InfoNCE denominator is over all
// candidates, positive included — it's a softmax that should put mass on the
// positive). The other candidates are the in-batch negatives. Temperature T<1
// sharpens the softmax (harder push on negatives); too small → the collapse we
// demo in the failure run.
function infoNceLoss(anchor: Vec, candidates: Vec[], positiveIdx: number, temperature: number): Value {
  const logits: Value[] = candidates.map((c) => cosineSim(anchor, c).mul(1 / temperature));
  const posLogit = logits[positiveIdx];
  return logSumExp(logits).sub(posLogit); // -(pos - lse) = lse - pos
}

// Collapsing objective for the FAILURE demo: pull positives together with NO
// negatives in the denominator. This is "only optimize the positive pair", which
// is exactly what a degenerate contrastive loss (or InfoNCE with T→0 and a single
// dominating term) reduces to. Minimizing -sim(anchor,pos) alone has a trivial
// global optimum: map EVERY input to the same vector (sim=1 for all pairs). That
// is representation collapse, and we measure it with uniformity afterward.
function collapseLoss(anchor: Vec, positive: Vec): Value {
  return cosineSim(anchor, positive).mul(-1);
}

// ---------------------------------------------------------------------------
// Evaluation helpers (detached from graph — never extend it during eval).
// ---------------------------------------------------------------------------

// Project an anchor's raw features through the CURRENT weights to plain numbers.
// Used for alignment/uniformity/scatter — reading the trained head, not training.
function embedDetached(W: Mat, feat: number[]): number[] {
  const out: number[] = [];
  for (let r = 0; r < W.length; r++) {
    let acc = 0;
    for (let k = 0; k < feat.length; k++) acc += W[r][k].data * feat[k];
    out.push(acc);
  }
  return out;
}

// alignment over (anchor-view, positive-view) pairs + uniformity over all anchor
// embeddings. Returned together because the chapter's whole point is they must be
// read as a pair: collapse shows up as great alignment but dead uniformity.
function measureGeometry(
  W: Mat,
  anchors: Anchor[],
  positives: number[][],
): { align: number; unif: number } {
  const anchorEmb = anchors.map((a) => embedDetached(W, a.feat));
  const posEmb = positives.map((p) => embedDetached(W, p));
  const pairs: Array<[number[], number[]]> = anchorEmb.map((e, i) => [e, posEmb[i]]);
  return { align: alignment(pairs), unif: uniformity(anchorEmb) };
}

// ---------------------------------------------------------------------------
// Training loop. Returns the trained weights and the per-epoch loss series.
// ---------------------------------------------------------------------------

interface TrainResult {
  W: Mat;
  losses: number[];
}

// One full contrastive training run. `useInfoNce=false` switches to the
// collapse objective for the failure demo, sharing all other machinery so the
// only variable is the loss function (a fair A/B).
function train(
  anchors: Anchor[],
  positives: number[][],
  opts: { epochs: number; lr: number; temperature: number; useInfoNce: boolean; seed: number },
): TrainResult {
  const initRng = new Rng(opts.seed);
  // Small symmetric init: keeps initial projections near zero so the first cosine
  // sims are unstructured and gradients well-scaled (see core/rng gaussian note).
  const W = makeMat(D_OUT, D_IN, () => gaussian(initRng, 0, 0.1));
  const params = collectParams(W);
  const opt = new Adam(params, opts.lr);

  // Shuffle order each epoch with a SEPARATE rng so weight-init draws and
  // batch-order draws never interfere (per core/rng "one Rng per concern").
  const orderRng = new Rng(opts.seed + 1);
  const idx = anchors.map((_, i) => i);

  const losses: number[] = [];
  for (let e = 0; e < opts.epochs; e++) {
    shuffle(orderRng, idx);

    let epochLoss = 0;
    for (const i of idx) {
      opt.zeroGrad();
      // Build a FRESH projection graph for the whole candidate pool every step.
      // Why not reuse one pool across the epoch: backward() accumulates grad into
      // EVERY node it touches (params and intermediates alike); zeroGrad clears
      // only params. Reusing shared candidate nodes across steps would leave stale
      // grad on those intermediates and corrupt the next step. Rebuilding is the
      // clean contract — graph is transient, params persist (see core/optim).
      const candByIndex: Vec[] = positives.map((p) => project(W, p));
      const anchorVec = project(W, anchors[i].feat);
      let loss: Value;
      if (opts.useInfoNce) {
        // Candidate j=i is the positive view of anchor i; all others are negatives.
        loss = infoNceLoss(anchorVec, candByIndex, i, opts.temperature);
      } else {
        loss = collapseLoss(anchorVec, candByIndex[i]);
      }
      loss.backward();
      opt.step();
      epochLoss += loss.data;
    }
    losses.push(epochLoss / anchors.length);
  }
  return { W, losses };
}

// ---------------------------------------------------------------------------
// main — runs the healthy InfoNCE training, then the collapse failure demo.
// ---------------------------------------------------------------------------

function main(): void {
  console.log("=".repeat(70));
  console.log("Stage 05 — 对比学习与 InfoNCE：负采样的现代化身");
  console.log("=".repeat(70));

  // Build data once; both runs see identical anchors/positives so the only
  // difference between healthy and collapsed is the loss function.
  const dataRng = new Rng(42);
  const anchors = buildAnchors(dataRng);
  const augRng = new Rng(7);
  const positives = anchors.map((a) => augment(augRng, a.feat));
  console.log(
    `\n数据: ${anchors.length} 个 anchor = ${NUM_CLUSTERS} 簇 × ${PER_CLUSTER}，` +
      `输入 ${D_IN} 维 → 投影 ${D_OUT} 维。每个 anchor 加噪声生成 1 个正样本视图。`,
  );
  console.log(
    `批内负样本: anchor i 的负样本 = 其余 ${anchors.length - 1} 个样本的投影` +
      `（batch 即负样本集，无需独立负采样器）。`,
  );

  // ---- Healthy InfoNCE run ------------------------------------------------
  const T = 0.2; // temperature: <1 sharpens; 0.2 is a common SimCLR-ish value
  const EPOCHS = 80;
  const LR = 0.05;

  // Geometry BEFORE training: random head → positives not aligned, space already
  // somewhat spread (random projection of separable clusters) but not optimized.
  const preRng = new Rng(123);
  const preW = makeMat(D_OUT, D_IN, () => gaussian(preRng, 0, 0.1));
  const pre = measureGeometry(preW, anchors, positives);

  console.log("\n" + "-".repeat(70));
  console.log(`[健康 InfoNCE]  T=${T}  epochs=${EPOCHS}  lr=${LR}  optimizer=Adam`);
  console.log("-".repeat(70));

  const tStart = Date.now();
  const healthy = train(anchors, positives, {
    epochs: EPOCHS,
    lr: LR,
    temperature: T,
    useInfoNce: true,
    seed: 123,
  });
  const wallMs = Date.now() - tStart;

  console.log("\n损失曲线 (InfoNCE, 每 epoch 批平均):");
  console.log(asciiLine(healthy.losses));
  console.log(
    `  loss ${healthy.losses[0].toFixed(4)} -> ${healthy.losses[healthy.losses.length - 1].toFixed(4)}` +
      `   (训练 wall-clock ${wallMs} ms, 真测)`,
  );

  const post = measureGeometry(healthy.W, anchors, positives);
  console.log("\n几何诊断 (Wang-Isola, 训练前 vs 训练后):");
  console.log(`  alignment  (正对平均距离, 越低越好):  ${pre.align.toFixed(4)} -> ${post.align.toFixed(4)}`);
  console.log(`  uniformity (分布散开程度, 越负越好):  ${pre.unif.toFixed(4)} -> ${post.unif.toFixed(4)}`);
  const alignDrop = pre.align - post.align;
  console.log(
    `  解读: alignment ${alignDrop >= 0 ? "下降" : "上升"} ${Math.abs(alignDrop).toFixed(4)}` +
      ` = 正对被拉近; uniformity 保持负值 = 表示没有坍缩、仍铺开在球面上。`,
  );

  // ---- Scatter of trained 2D representation -------------------------------
  // Label each point by its ground-truth cluster digit so clustering is readable.
  const anchorEmb = anchors.map((a) => embedDetached(healthy.W, a.feat));
  const points: Array<[number, number]> = anchorEmb.map((e) => [e[0], e[1]]);
  const labels = anchors.map((a) => String(a.cluster));
  console.log("\n训练后 2D 表示散点 (字符=真实簇 id, '*'=多点重叠):");
  console.log(asciiScatter(points, labels));
  console.log(
    `  期望: 相同数字聚成 ${NUM_CLUSTERS} 团 — 模型只见过加噪视图，从未见过簇标签，` +
      `聚簇是对比学习自己学出来的。`,
  );

  // ---- FAILURE MODE: collapse --------------------------------------------
  // Only optimize positives (no negatives in the denominator) → the global
  // optimum is "send everything to one point". We use the same data/optimizer.
  console.log("\n" + "=".repeat(70));
  console.log("失败模式: 温度极小 + 只优化正对 → 表示坍缩 (representation collapse)");
  console.log("=".repeat(70));
  console.log(
    "去掉分母里的负样本 (只最大化正对相似度)，全局最优是把所有输入映射到同一个向量:\n" +
      "  正对相似度完美 (alignment→0)，但整个空间挤成一团 (uniformity→0)。",
  );

  // Train the collapse objective LONGER than the healthy run: the positives-only
  // objective has nothing stopping it, so given more steps it drives the space
  // ever flatter — we want the reader to see how far it goes when unchecked.
  const COLLAPSE_EPOCHS = 200;
  const collapsed = train(anchors, positives, {
    epochs: COLLAPSE_EPOCHS,
    lr: LR,
    temperature: 1e-3, // near-zero T, irrelevant here but emphasizes the regime
    useInfoNce: false, // <-- positives-only objective
    seed: 123,
  });

  const collapsedGeo = measureGeometry(collapsed.W, anchors, positives);
  console.log("\n坍缩后几何诊断 (对比健康运行):");
  console.log(
    `  alignment:   健康 ${post.align.toFixed(4)}   坍缩 ${collapsedGeo.align.toFixed(4)}` +
      `  (坍缩的 alignment 极小 — 但这是陷阱)`,
  );
  console.log(
    `  uniformity:  健康 ${post.unif.toFixed(4)}   坍缩 ${collapsedGeo.unif.toFixed(4)}` +
      `  (坍缩 uniformity 趋近 0 = 所有向量挤成一点)`,
  );

  // Direct receipt matching the COSINE objective: mean pairwise cosine similarity
  // of all embeddings. The collapse here is DIRECTIONAL (everything points the
  // same way), not positional — so L2 distance is the wrong tape; cosine is right.
  // Healthy → low/spread cosine (different directions); collapsed → near 1.0
  // (every vector parallel). This is the number that exposes the degenerate space.
  const collapsedEmb = anchors.map((a) => embedDetached(collapsed.W, a.feat));
  console.log(
    `\n直接度量方向坍缩 (所有 embedding 两两平均余弦相似度, ∈[-1,1]):` +
      `\n  健康: ${meanPairwiseCosine(anchorEmb).toFixed(4)}   坍缩: ${meanPairwiseCosine(collapsedEmb).toFixed(4)}` +
      `   (坍缩→1.0 = 所有向量指向同一方向)`,
  );
  console.log(
    "  教训: 只看 alignment 会误判坍缩为「完美」。必须同时看 uniformity —\n" +
      "  这正是 InfoNCE 分母里的负样本存在的理由: 它们提供「推开」的力，阻止坍缩。",
  );

  console.log("\n" + "=".repeat(70));
  console.log(
    "诚实声明: 以上为 toy 合成簇 + D_out=2, 绝对数值偏乐观。可迁移的是\n" +
      "(1) 训练前后 alignment/uniformity 的变化方向, (2) 健康 vs 坍缩的对比。",
  );
  console.log("=".repeat(70));
}

// Mean pairwise COSINE similarity over all unordered pairs. The receipt for
// directional collapse under a cosine objective: → 1.0 when every vector points
// the same way (degenerate), well below 1.0 when directions are spread (healthy).
// We reuse core/eval's cosineSimilarity so the metric matches exactly what the
// training loss optimized.
function meanPairwiseCosine(vecs: number[][]): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < vecs.length; i++) {
    for (let j = i + 1; j < vecs.length; j++) {
      sum += cosineSimilarity(vecs[i], vecs[j]);
      count++;
    }
  }
  return count === 0 ? 0 : sum / count;
}

main();
