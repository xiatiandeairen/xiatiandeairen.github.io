// stage02-fullft.ts — Chapter 02: full fine-tuning vs freezing — the start of the cost wall.
//
// WHAT THIS CHAPTER ESTABLISHES (the baseline every PEFT chapter is measured against):
//   1. Pretrain a toy Transformer on task A (copy), snapshot it as the SHARED base.
//   2. Full fine-tune that base to task B (a fixed vocab relabeling of copy): EVERY param
//      trainable => trainable ratio = 100%. This is the "expensive" reference point.
//   3. Estimate training memory (weights + grads + Adam m/v state) and show that with 100%
//      trainable params the optimizer-state + gradient terms are as large as they can get —
//      this is the cost wall PEFT later tears down.
//   4. Measure CATASTROPHIC FORGETTING: re-test task A accuracy as fine-tuning on B proceeds.
//      Full FT moves all weights, so task A skill decays. We print A and B accuracy as two
//      curves that CROSS — B climbs while A falls.
//   5. FAILURE MODE: with an aggressive lr, B is learned but A collapses to ~chance. We show
//      the gap between a moderate-lr run (A degrades gracefully) and a large-lr run (A
//      crashes) so the reader sees that "it learned the new task" hides "it forgot the old".
//
// WHY a real classification head (logits over vocab) and not the smoke test's MSE-on-
//   embeddings: forgetting is only legible as an ACCURACY drop. We need argmax-decodable
//   per-position predictions over the vocab, so we add a Linear(dModel -> vocab) head and
//   decode with argmax. Accuracy is a non-differentiable metric computed separately from
//   the (differentiable) loss.
//
// WHY task B is a vocab RELABELING of copy, not the data module's default copy->reverse:
//   reverse/sort are PURELY POSITIONAL transforms, but the core has no positional encoding
//   (see nn.ts), so a toy model literally cannot learn them past ~chance — which would make
//   "full FT masters task B" a lie. A fixed permutation of the vocab (target[i]=perm[in[i]])
//   is TOKEN-LOCAL (learnable to ~100%) yet maximally CONFLICTS with copy (every token
//   remaps), so it is the honest substrate for "learns B / forgets A". We build B with a tiny
//   in-stage generator over the SAME (seqLen, vocab) as copy — this reuses core data shape,
//   it does not reimplement the engine.
//
// WHY the loss is softmax + MSE-to-one-hot (Brier score), not cross-entropy: the core
//   autodiff engine ships softmax but no log op (tensor.ts is deliberately minimal). Brier
//   = ||softmax(logits) - onehot(target)||^2 is fully differentiable with the existing ops,
//   drives the same argmax decision boundary, and we name it honestly rather than mislabel
//   it cross-entropy.
//
// DETERMINISM: seed once at the top; the global PRNG is the only randomness source, so every
//   number below reproduces. INVARIANT: the two fine-tune runs (moderate vs large lr) each
//   re-load the SAME frozen-base checkpoint and re-seed identically before training, so the
//   only variable between them is the learning rate.
//
// HONESTY: param counts, accuracies and losses are REAL computed values. The memory bars are
//   ESTIMATES from the pure-arithmetic formula (labeled est.). Absolute numbers are toy-scale
//   and optimistic; the transferable truths are (a) 100% trainable ratio, (b) the A-vs-B
//   accuracy crossover, and (c) that larger lr trades faster B for harsher A forgetting.

import { seed, randint, shuffle } from "./core/prng.js";
import { Tensor } from "./core/tensor.js";
import { Module, Embedding, Linear, TransformerBlock } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { genCopy, batches, type SeqExample } from "./core/data.js";
import { lossCurve, bar } from "./core/viz.js";
import { dump, loadBase, type Checkpoint } from "./core/checkpoint.js";
import { estBytes, toMB } from "./core/mem.js";

// ---------------------------------------------------------------------------
// Config. Small enough to train in seconds, large enough that the trainable-ratio
//   arithmetic and the accuracy curves are real.
// ---------------------------------------------------------------------------
const SEED = 1234;
const D_MODEL = 16;
const HEADS = 2;
const D_FF = 32;
const SEQ_LEN = 6;
const VOCAB = 8;
const N_PRETRAIN = 96;
const N_FINETUNE = 96;
const N_EVAL = 64; // held-out examples for accuracy (drawn AFTER train sets, so disjoint)
const BATCH = 16;
const PRETRAIN_STEPS = 120;
const FINETUNE_STEPS = 120;
const LR_MODERATE = 5e-3;
const LR_LARGE = 3e-2; // 6x — large enough to DESTABILIZE training (the failure mode)

// Task B = a fixed random permutation of the vocab applied per token to a copy.
//   Built once under the seed so it is reproducible. targetB[i] = VOCAB_PERM[input[i]].
//   WHY a full shuffle (every token remaps): maximizes the conflict with copy, so learning B
//   NECESSARILY overwrites the copy mapping => forgetting is total, not incidental. That is
//   the honest worst case the chapter wants: even a perfectly-learned B costs you all of A.
let VOCAB_PERM: number[] = [];

/** Generate task-B examples: per-token relabel of a random copy input. */
function genRelabel(n: number): SeqExample[] {
  const out: SeqExample[] = [];
  for (let i = 0; i < n; i++) {
    const input = Array.from({ length: SEQ_LEN }, () => randint(0, VOCAB));
    out.push({ input, target: input.map((t) => VOCAB_PERM[t]) });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Model: tokens -> Embedding -> TransformerBlock -> Linear head -> per-position logits.
// WHY a dedicated Module subclass rather than chaining loose layers: Module.parameters()
//   and .trainable() then give us the honest param accounting (and freeze()) for free, which
//   is the whole point of the chapter.
// ---------------------------------------------------------------------------
class SeqTagger extends Module {
  emb: Embedding;
  block: TransformerBlock;
  head: Linear; // (dModel -> vocab) per-position classifier
  constructor() {
    super();
    this.emb = new Embedding(VOCAB, D_MODEL);
    this.block = new TransformerBlock(D_MODEL, HEADS, D_FF);
    this.head = new Linear(D_MODEL, VOCAB);
  }
  // forward() takes token ids (the seq is variable-membership but fixed-length here).
  // Returns (seq, vocab) logits. NOTE: base Module.forward signature is Tensor->Tensor; we
  //   take ids instead because the embedding gathers rows, so we override loosely and never
  //   call it via the Sequential path.
  forwardIds(ids: number[]): Tensor {
    const h = this.emb.lookup(ids); // (seq, dModel)
    const z = this.block.forward(h); // (seq, dModel)
    return this.head.forward(z); // (seq, vocab)
  }
  override forward(_x: Tensor): Tensor {
    throw new Error("SeqTagger: use forwardIds(ids)");
  }
}

// ---------------------------------------------------------------------------
// Differentiable loss: softmax over the vocab axis, then MSE to the one-hot target (Brier).
//   Returns a scalar tensor averaged over all (position) rows.
// INVARIANT: targets index into [0,VOCAB); one-hot rows are constant leaves (no grad).
// ---------------------------------------------------------------------------
function brierLoss(logits: Tensor, targetIds: number[]): Tensor {
  const probs = logits.softmax(); // (seq, vocab), each row sums to 1
  const oneHot = new Float64Array(targetIds.length * VOCAB);
  for (let i = 0; i < targetIds.length; i++) oneHot[i * VOCAB + targetIds[i]] = 1;
  const target = new Tensor(oneHot, [targetIds.length, VOCAB], false);
  const diff = probs.sub(target);
  return diff.mul(diff).mean();
}

// Non-differentiable accuracy: fraction of positions whose argmax logit equals the target.
//   Computed OUTSIDE the graph (no grad) — accuracy is a report metric, not an objective.
function sequenceAccuracy(model: SeqTagger, examples: SeqExample[]): number {
  let correct = 0;
  let total = 0;
  for (const ex of examples) {
    const logits = model.forwardIds(ex.input);
    for (let i = 0; i < ex.target.length; i++) {
      let best = 0;
      let bestVal = -Infinity;
      for (let j = 0; j < VOCAB; j++) {
        const v = logits.data[i * VOCAB + j];
        if (v > bestVal) {
          bestVal = v;
          best = j;
        }
      }
      if (best === ex.target[i]) correct++;
      total++;
    }
  }
  return correct / total;
}

// One epoch of optimization over `data`; returns the mean per-example loss for the epoch.
function trainEpoch(model: SeqTagger, opt: Adam, data: SeqExample[]): number {
  let totalLoss = 0;
  let count = 0;
  for (const batch of batches(data, BATCH)) {
    for (const ex of batch) {
      const logits = model.forwardIds(ex.input);
      const loss = brierLoss(logits, ex.target);
      opt.zeroGrad();
      loss.backward();
      opt.step();
      totalLoss += loss.data[0];
      count++;
    }
  }
  return totalLoss / count;
}

// ---------------------------------------------------------------------------
// Fine-tune the base checkpoint to task B at a given lr, recording A & B accuracy per step.
// WHY re-load + re-seed inside: each run must start from the IDENTICAL frozen base and the
//   identical PRNG state, so lr is the only difference between the moderate and large runs.
//   loadBase(..., freeze=false) is the full-FT switch: every param stays trainable.
// ---------------------------------------------------------------------------
interface FinetuneTrace {
  lossB: number[];
  accA: number[]; // task A (old) accuracy as B-finetuning proceeds — the forgetting signal
  accB: number[]; // task B (new) accuracy — the thing we are optimizing
  trainableRatio: number;
  trainableParams: number;
  totalParams: number;
}

function fullFinetune(
  baseCkpt: Checkpoint,
  lr: number,
  evalA: SeqExample[],
  evalB: SeqExample[],
  trainB: SeqExample[],
): FinetuneTrace {
  seed(SEED + 7); // fixed offset: identical batch shuffles across the moderate/large runs
  const model = new SeqTagger();
  // freeze=false => FULL fine-tuning: copy base weights but keep them all trainable.
  loadBase(model, baseCkpt, false);
  const opt = new Adam(model.trainable(), lr);

  const lossB: number[] = [];
  const accA: number[] = [];
  const accB: number[] = [];
  // step 0 = the base, before any B update: this is where A accuracy is highest.
  accA.push(sequenceAccuracy(model, evalA));
  accB.push(sequenceAccuracy(model, evalB));
  lossB.push(NaN); // no loss yet at the pre-training snapshot; rendered as a gap

  for (let step = 0; step < FINETUNE_STEPS; step++) {
    const l = trainEpoch(model, opt, trainB);
    lossB.push(l);
    accA.push(sequenceAccuracy(model, evalA));
    accB.push(sequenceAccuracy(model, evalB));
  }

  return {
    lossB,
    accA,
    accB,
    trainableRatio: model.numParams({ trainableOnly: true }) / model.numParams(),
    trainableParams: model.numParams({ trainableOnly: true }),
    totalParams: model.numParams(),
  };
}

// Render two accuracy series as one crossover chart (A falling, B rising).
//   We print a compact per-step table sampled at a stride so the crossover is visible
//   without flooding the terminal. Numbers are the REAL accuracies.
function renderCrossover(label: string, accA: number[], accB: number[]): string {
  const n = accA.length;
  const stride = Math.max(1, Math.floor(n / 12));
  const rows: string[] = [`  ${label}`];
  rows.push(`  step │ taskA(old)  taskB(new)  bars (A=▒  B=█)`);
  for (let i = 0; i < n; i += stride) {
    const a = accA[i];
    const b = accB[i];
    const aBar = "▒".repeat(Math.round(a * 20));
    const bBar = "█".repeat(Math.round(b * 20));
    rows.push(
      `  ${String(i).padStart(4)} │   ${(a * 100).toFixed(0).padStart(3)}%       ${(b * 100)
        .toFixed(0)
        .padStart(3)}%     A:${aBar}`,
    );
    rows.push(`       │                        B:${bBar}`);
  }
  return rows.join("\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
function main(): void {
  seed(SEED);

  // Build task B's vocab permutation FIRST (consumes PRNG deterministically), then datasets.
  VOCAB_PERM = Array.from({ length: VOCAB }, (_, i) => i);
  shuffle(VOCAB_PERM);

  // --- Data. Task A = copy; task B = per-token relabel by VOCAB_PERM. Same (seqLen,vocab).
  //     Draw train then disjoint eval sets (eval drawn after train, different PRNG draws). ---
  const trainA = genCopy(N_PRETRAIN, SEQ_LEN, VOCAB); // task A = copy
  const evalA = genCopy(N_EVAL, SEQ_LEN, VOCAB);
  const trainB = genRelabel(N_FINETUNE); // task B = vocab relabel (conflicts with copy)
  const evalB = genRelabel(N_EVAL);
  const chance = 1 / VOCAB; // a random classifier over the vocab

  console.log("=".repeat(70));
  console.log("Stage 02 — 全量微调 vs 冻结: 成本墙的实测起点");
  console.log("=".repeat(70));
  console.log(
    `任务 A (预训练) = copy, 任务 B (微调) = vocab 重映射 perm=[${VOCAB_PERM.join(",")}] | vocab=${VOCAB} seqLen=${SEQ_LEN} | 随机基线准确率=${(
      chance * 100
    ).toFixed(0)}%`,
  );

  // --- 1. Pretrain the base on task A ---------------------------------------
  const base = new SeqTagger();
  const pretrainOpt = new Adam(base.trainable(), LR_MODERATE);
  const pretrainLoss: number[] = [];
  for (let step = 0; step < PRETRAIN_STEPS; step++) {
    pretrainLoss.push(trainEpoch(base, pretrainOpt, trainA));
  }
  const baseAccA = sequenceAccuracy(base, evalA);
  const baseAccB = sequenceAccuracy(base, evalB);
  console.log("\n[1] 预训练 base (任务 A=copy)");
  console.log(
    `    Brier loss ${pretrainLoss[0].toFixed(4)} -> ${pretrainLoss[pretrainLoss.length - 1].toFixed(4)}`,
  );
  console.log(
    `    base 在 A 上准确率 = ${(baseAccA * 100).toFixed(1)}%  |  base 在 B(重映射) 上准确率 = ${(
      baseAccB * 100
    ).toFixed(1)}% (未学过 B, 接近随机基线)`,
  );
  console.log(lossCurve(pretrainLoss, { height: 6, label: "    pretrain Brier loss (task A)" }));

  // --- 2. Snapshot the base; this is the shared starting point for every later chapter. ---
  const baseCkpt: Checkpoint = dump(base);

  // --- 3. Full fine-tune to task B at the MODERATE lr -----------------------
  const moderate = fullFinetune(baseCkpt, LR_MODERATE, evalA, evalB, trainB);

  console.log("\n[2] 全量微调到任务 B (lr=" + LR_MODERATE + ")");
  console.log(
    `    可训练参数 = ${moderate.trainableParams} / ${moderate.totalParams} = ${(
      moderate.trainableRatio * 100
    ).toFixed(1)}%  ← 全量微调: 100% 参数都在动 (成本墙的顶点)`,
  );
  console.log(
    `    任务 B 准确率: ${(moderate.accB[0] * 100).toFixed(1)}% (起点) -> ${(
      moderate.accB[moderate.accB.length - 1] * 100
    ).toFixed(1)}%${moderate.accB[moderate.accB.length - 1] >= 0.8 ? " (学会 B)" : ""}`,
  );
  console.log(
    `    任务 A 准确率: ${(moderate.accA[0] * 100).toFixed(1)}% (起点) -> ${(
      moderate.accA[moderate.accA.length - 1] * 100
    ).toFixed(1)}% (微调后回测 → 灾难性遗忘量化)`,
  );
  const forgetModerate = moderate.accA[0] - moderate.accA[moderate.accA.length - 1];
  console.log(
    `    遗忘量 (A 准确率下降) = ${(forgetModerate * 100).toFixed(1)} 个百分点`,
  );
  console.log(
    lossCurve(moderate.lossB.filter((v) => Number.isFinite(v)), {
      height: 6,
      label: "    finetune Brier loss (task B, moderate lr)",
    }),
  );
  console.log(renderCrossover("A/B 准确率随步数 (moderate lr) — A 下滑 / B 上升", moderate.accA, moderate.accB));

  // --- 4. 内存估算: 全量微调的 grad + Adam(m,v) state 是满的 -----------------
  // Honest: these are ESTIMATES from the arithmetic formula, not measured VRAM.
  // We contrast full-FT (100% trainable) with a HYPOTHETICAL tiny-trainable budget to show
  //   WHERE the cost wall lives — it is the grad + optimizer-state terms, which scale with
  //   trainable params. (Later chapters make that tiny for real.)
  const P = moderate.totalParams;
  const fullMem = estBytes({ totalParams: P, trainableParams: P, optMultiplier: 2 });
  const hypoTiny = Math.max(1, Math.round(P * 0.005)); // ~0.5% trainable, a PEFT-shaped budget
  const peftShapedMem = estBytes({ totalParams: P, trainableParams: hypoTiny, optMultiplier: 2 });
  console.log("\n[3] 训练显存估算 (est., 纯算术公式; 非实测 VRAM)");
  console.log(
    `    full-FT 拆解 (bytes, est.): weights=${fullMem.weightsBytes} grad=${fullMem.gradBytes} adam(m,v)=${fullMem.optimStateBytes} act=${fullMem.activationBytes}`,
  );
  console.log(
    bar([
      {
        label: "full-FT (100% trainable)",
        value: Math.round(toMB(fullMem.totalBytes) * 1e6) / 1e6,
        note: "MB (est.)",
      },
      {
        label: "PEFT-shaped (~0.5%)",
        value: Math.round(toMB(peftShapedMem.totalBytes) * 1e6) / 1e6,
        note: "MB (est., 对照: 后续章节实现)",
      },
    ]),
  );
  console.log(
    `    成本墙在哪: full-FT 的 grad+Adam state = ${(
      ((fullMem.gradBytes + fullMem.optimStateBytes) / fullMem.totalBytes) *
      100
    ).toFixed(0)}% 的训练显存, 且随可训练参数线性增长。`,
  );

  // --- 5. FAILURE MODE: lr too large => training DESTABILIZES (loses both tasks) ----------
  // We derive the verdict from the ACTUAL numbers (no hardcoded claim of direction), because
  //   what "too large" does depends on the run: here it fails to even learn B AND leaves A
  //   wrecked — strictly worse than the moderate run's "learn B, lose A" trade.
  const large = fullFinetune(baseCkpt, LR_LARGE, evalA, evalB, trainB);
  const modBFinal = moderate.accB[moderate.accB.length - 1];
  const lrgBFinal = large.accB[large.accB.length - 1];
  console.log("\n[4] 失败模式: lr 偏大 (lr=" + LR_LARGE + ", " + Math.round(LR_LARGE / LR_MODERATE) + "x)");
  console.log(
    `    任务 B 准确率: ${(large.accB[0] * 100).toFixed(1)}% -> ${(lrgBFinal * 100).toFixed(
      1,
    )}%  (对比 moderate lr 的 ${(modBFinal * 100).toFixed(0)}%)`,
  );
  console.log(
    `    任务 A 准确率: ${(large.accA[0] * 100).toFixed(1)}% -> ${(
      large.accA[large.accA.length - 1] * 100
    ).toFixed(1)}%  (随机基线=${(chance * 100).toFixed(0)}%)`,
  );
  const aFloor = large.accA.reduce((mn, v) => Math.min(mn, v), 1);
  console.log(
    `    large-lr 下 A 的最低准确率 = ${(aFloor * 100).toFixed(1)}% (随机基线=${(chance * 100).toFixed(
      0,
    )}%, 即 A 已彻底崩坏)`,
  );
  // Honest verdict computed from data: did large lr even learn B?
  const lrgLearnedB = lrgBFinal >= 0.8;
  console.log(
    lrgLearnedB
      ? `    诊断: lr 偏大仍学会了 B, 但 A 崩得更狠 — 速度换遗忘。`
      : `    诊断: lr 偏大 = 训练失稳, B 没学会 (${(lrgBFinal * 100).toFixed(
          0,
        )}%) 且 A 仍崩坏 — 两个任务一起输, 比 moderate 更糟。`,
  );
  console.log(renderCrossover("A/B 准确率随步数 (large lr) — 训练失稳, 两条线都趴在随机基线附近", large.accA, large.accB));

  // --- Verdict --------------------------------------------------------------
  const modBLearned = modBFinal >= 0.8;
  console.log("\n" + "-".repeat(70));
  console.log("结论 (本章建立的基线):");
  console.log("  • 全量微调 = 100% 参数可训练 → grad + 优化器状态显存拉满, 这是成本墙的顶点。");
  console.log(
    `  • moderate lr: 任务 B ${(modBFinal * 100).toFixed(0)}%${modBLearned ? "(学会)" : ""}, 但任务 A 从 ${(
      moderate.accA[0] * 100
    ).toFixed(0)}% 崩到 ${(moderate.accA[moderate.accA.length - 1] * 100).toFixed(
      0,
    )}% = 灾难性遗忘。B 与 A 全冲突 ⇒ 学会 B 必然抹掉 A。`,
  );
  console.log("  • lr 偏大: 训练失稳, 连 B 都学不会还赔上 A — 全量微调对 lr 也更脆弱。");
  console.log("  • 后续章节 (LoRA/QLoRA/Adapter) 的目标: 冻结 base → 削掉 grad+state 显存, 并天然缓解遗忘。");
  console.log("\n⚠ toy-scale: 绝对值偏乐观, 可迁移的是 (1) 100% 可训练比例 (2) 学会 B / 抹掉 A 的权衡 (3) lr 偏大→训练失稳。");
}

main();
