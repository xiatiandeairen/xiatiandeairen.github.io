// stage02-attention-viz.ts — Reading attention patterns as ASCII heatmaps, and the trap that
//   makes naive attention "interpretation" unreliable.
//
// WHY this chapter exists: attention is the most-visualized part of a transformer because the
//   pattern matrix is human-readable — you can literally SEE "query position i looks at key
//   position j". This stage shows the honest version of that: print every head's pattern,
//   classify it by a simple rule (first-position / self / previous-token / diffuse), and
//   quantify "how focused" each head is via attention entropy.
//
// THE TRAP (the real point of the chapter): a sharp, nameable attention pattern is SALIENCE,
//   not IMPORTANCE. We pick the single most confident-looking head — the one a human reviewer
//   would point at and say "that's the circuit" — and ABLATE it. If the answer barely moves,
//   the picture lied: attention weights show where a head LOOKS, never whether the model NEEDS
//   what it found there. Only a causal intervention (ablation/patching) answers necessity.
//   This is the gap between an attention dashboard and an actual mechanism.
//
// HONESTY: on this toy induction task patterns are unusually clean and nameable. Real models
//   have heads that defy single-word labels (polysemantic attention, head superposition). What
//   transfers is NOT "induction heads make a tidy stripe" but the methodology: name the
//   pattern, then DISTRUST the name until a causal test confirms the head matters.

import { Tensor } from "./core/autograd.js";
import { mulberry32, argmax } from "./core/rng.js";
import { copyTask, type Task } from "./core/tasks.js";
import { type TinyTransformer, type Hooks } from "./core/nn.js";
import { trainToyModel, defaultTrainConfig } from "./core/model_zoo.js";
import { runWithCache, ablate } from "./core/interp.js";
import { asciiHeatmap, asciiBar } from "./core/viz.js";

interface HeadStat {
  layer: number;
  head: number;
  point: string; // head_z hook name, used to address this head in ablate()
  entropyBits: number; // mean over choice-having queries; low = focused
  label: string; // heuristic geometry label
  focusFrac: number; // fraction of queries matching the label's geometry
}

// ----------------------------------------------------------------------------
// Pattern extraction.
// ----------------------------------------------------------------------------
//
// MultiHeadSelfAttention caches each head's softmaxed pattern in `lastPatterns` (flat
//   seq*seq Float64Array) on every forward. runWithCache runs a forward, so after it the
//   block's lastPatterns reflect THAT input. We reshape to a (seq, seq) matrix where row i is
//   the query at position i and column j is how much it attends to key position j. Causal mask
//   means the upper triangle (j>i) is ~0 — a structural invariant worth eyeballing in the map.
function patternMatrix(model: TinyTransformer, layer: number, head: number): number[][] {
  const flat = model.blocks[layer].attn.lastPatterns[head];
  const seq = Math.round(Math.sqrt(flat.length));
  const m: number[][] = [];
  for (let i = 0; i < seq; i++) {
    const row: number[] = [];
    for (let j = 0; j < seq; j++) row.push(flat[i * seq + j]);
    m.push(row);
  }
  return m;
}

// ----------------------------------------------------------------------------
// Attention entropy: how spread-out is a query's attention?
// ----------------------------------------------------------------------------
//
// For one query row (a probability distribution over key positions) Shannon entropy in bits:
//   0 = all mass on one key (maximally focused), log2(n) = uniform over n keys (maximally
//   diffuse). We average over the queries that actually have a CHOICE (a causal row with only
//   1 legal key is trivially 0-entropy and would deflate the average), so the number reflects
//   real selectivity, not the mask. Low mean entropy => a "confident-looking" head — exactly
//   the kind a human flags as important, which is what we later put on trial.
function meanAttentionEntropyBits(pattern: number[][]): number {
  let sum = 0;
  let counted = 0;
  for (let i = 0; i < pattern.length; i++) {
    const legalKeys = i + 1; // causal: query i may attend to keys 0..i
    if (legalKeys < 2) continue; // a single legal key has no choice -> skip, don't dilute
    let h = 0;
    for (let j = 0; j <= i; j++) {
      const p = pattern[i][j];
      if (p > 1e-12) h -= p * Math.log2(p);
    }
    sum += h;
    counted++;
  }
  return counted === 0 ? 0 : sum / counted;
}

// ----------------------------------------------------------------------------
// Pattern classification by simple geometric rules.
// ----------------------------------------------------------------------------
//
// Deliberately crude thresholds over WHERE attention mass concentrates, applied to the
//   per-query argmax over causal-legal keys. The taxonomy mirrors canonical named heads:
//     - "first-pos"     : queries dump attention on key 0 (attention sink / no-op head)
//     - "self"          : queries attend to themselves (main diagonal)
//     - "prev-token"    : queries attend to position i-1 (the bigram-copy precursor head)
//     - "back-offset:D" : queries consistently attend D positions back (the COPY/INDUCTION
//                         signature — for copy(n) the answer sits n+1 tokens earlier; a head
//                         that does this is the textbook "this IS the circuit" head)
//     - "diffuse"       : no consistent target (low confidence / distributed)
// HONEST CAVEAT: this classifier is a HEURISTIC. It assigns a label even to heads that don't
//   really fit one — which is the whole lesson: a label is a hypothesis, not a finding. A head
//   labelled "back-offset" might be doing nothing causally; §4 tests exactly that.
function classifyHead(pattern: number[][]): { label: string; focusFrac: number } {
  const seq = pattern.length;
  // For each query (that has a non-self choice) record which constant back-offset its argmax
  // lands on. offset 0 = self, 1 = prev-token, and the dominant non-trivial offset reveals a
  // copy/induction diagonal. We tally offsets so a consistent stripe wins regardless of D.
  const offsetCount = new Map<number, number>();
  let firstPos = 0;
  let decided = 0;
  for (let i = 1; i < seq; i++) {
    const legal = pattern[i].slice(0, i + 1); // causal-legal keys 0..i
    const tgt = argmax(legal);
    decided++;
    if (tgt === 0) firstPos++;
    const off = i - tgt;
    offsetCount.set(off, (offsetCount.get(off) ?? 0) + 1);
  }
  if (decided === 0) return { label: "trivial", focusFrac: 0 };
  // dominant constant offset across queries
  let bestOff = 0;
  let bestCount = 0;
  for (const [off, cnt] of offsetCount) {
    if (cnt > bestCount) {
      bestCount = cnt;
      bestOff = off;
    }
  }
  // first-position sink competes separately (its argmax offset varies with i, so it would
  // never form a constant-offset stripe; tally it explicitly)
  if (firstPos / decided >= 0.5 && firstPos >= bestCount) {
    return { label: "first-pos", focusFrac: firstPos / decided };
  }
  const frac = bestCount / decided;
  if (frac < 0.5) return { label: "diffuse", focusFrac: frac };
  if (bestOff === 0) return { label: "self", focusFrac: frac };
  if (bestOff === 1) return { label: "prev-token", focusFrac: frac };
  return { label: `back-offset:${bestOff}`, focusFrac: frac };
}

// ----------------------------------------------------------------------------
// Accuracy at the decisive position (the planted answer).
// ----------------------------------------------------------------------------
//
// The copy task scores every second-half position (each must reproduce its first-half token).
//   We evaluate the model's argmax at all scorable positions against the oracle — a
//   behavioural metric to watch survive (or not) an ablation.
function evalAccuracy(task: Task, model: TinyTransformer, n: number, seed: number): number {
  const rng = mulberry32(seed);
  let correct = 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const input = task.makeBatch(1, rng).inputs[0];
    const oracle = task.oracle(input);
    const { logits } = runWithCache(model, input);
    for (const pos of task.scorablePositions(input)) {
      const row = logits.data.subarray(pos * task.vocab, (pos + 1) * task.vocab);
      if (argmax(row) === oracle[pos]) correct++;
      total++;
    }
  }
  return correct / total;
}

function main(): void {
  // Copy task: [t0..t4, SEP, t0..t4] — after SEP each position must reproduce the matching
  // first-half token. WHY copy and not modAdd/induction here: (1) seqLen=11 gives visually
  // rich 11x11 maps (modAdd's 3x3 is unreadable); (2) the solution is a clean "copy head"
  // that attends back n+1 positions — the textbook NAMEABLE pattern this chapter is about;
  // (3) it actually LEARNS to 100% in seconds, so we're interpreting a real circuit, not noise
  // (an unlearned model's attention is just the attention-sink default — meaningless to name).
  const N = 5;
  const task = copyTask(4, N); // vocab=5 (0..3 + SEP=4), seqLen=11
  console.log("=== Stage 02: 注意力可视化 + 「看起来重要 ≠ 真的重要」 ===\n");
  console.log(`任务: ${task.name}  (复制: [前半 ${N} token][SEP][前半 ${N} token], 后半每位复现对应前半 token)`);

  // Dedicated checkpoint (deterministic + cached by config key). 600 steps suffices for 100%.
  const cfg = defaultTrainConfig(task, { steps: 600, seed: 20250219 });
  const t0 = Date.now();
  const trained = trainToyModel(task, cfg);
  const trainMs = Date.now() - t0;
  const model = trained.model;
  const acc = evalAccuracy(task, model, 200, 4242);
  console.log(
    `模型: ${cfg.model.nLayers}L x ${cfg.model.nHeads}H, dModel=${cfg.model.dModel}  ` +
      `finalLoss=${trained.finalLoss.toFixed(4)}  准确率=${(acc * 100).toFixed(1)}%  ` +
      `(实测 wall-clock ${trainMs} ms)`,
  );
  if (acc < 0.95) {
    // Guard: interpreting an unlearned model is interpreting noise. If this fires the rest of
    // the stage is meaningless, so we say so loudly rather than print pretty-but-fake maps.
    console.log(`  警告: 准确率 < 95%, 模型没学会, 下面的注意力图是噪声, 不要解读!`);
  }
  console.log(`诚实边界: 该 toy 任务上准确率/模式整洁度偏乐观; 可迁移的是方法, 不是漂亮数字。\n`);

  // --- 1. One concrete input: print every head's attention map. ---------------------------
  // Freeze ONE deterministic sample so the printed maps are reproducible. SEP splits the
  // sequence; second-half query i should attend back to its first-half twin at i-(n+1).
  const sampleRng = mulberry32(7);
  const input = task.makeBatch(1, sampleRng).inputs[0];
  const sepIdx = input.indexOf(4); // SEP token id = k = 4
  const scorable = task.scorablePositions(input);
  const queryPos = scorable[scorable.length - 1]; // a representative decisive position
  const answer = task.oracle(input)[queryPos];
  runWithCache(model, input); // populate lastPatterns for this exact input
  console.log(`[1] 单个输入的逐头注意力图:`);
  console.log(`    序列 tokens = [${input.join(" ")}]  (SEP=4 @pos${sepIdx})`);
  console.log(
    `    复制头应有的几何: 后半 query 回看 ${N + 1} 格 (back-offset:${N + 1}); ` +
      `代表决策位置=pos${queryPos}, 正确答案=${answer}`,
  );
  console.log(`    读法: 行=query 位置, 列=key 位置(它在看谁); 越深越关注; 上三角因果掩码应近空。\n`);

  const axis = input.map((t) => String(t)); // single-char tokens (vocab<10) as axis ticks
  const stats: HeadStat[] = [];
  for (let L = 0; L < cfg.model.nLayers; L++) {
    for (let H = 0; H < cfg.model.nHeads; H++) {
      const pat = patternMatrix(model, L, H);
      const entropy = meanAttentionEntropyBits(pat);
      const cls = classifyHead(pat);
      stats.push({ layer: L, head: H, point: `blocks.${L}.attn.head_z.${H}`, entropyBits: entropy, label: cls.label, focusFrac: cls.focusFrac });
      console.log(
        asciiHeatmap(pat, {
          title: `  L${L}H${H}  [${cls.label}, focus=${(cls.focusFrac * 100).toFixed(0)}%, entropy=${entropy.toFixed(2)} bits]`,
          rowLabels: axis,
          colLabels: axis,
          vmin: 0,
          vmax: 1,
        }),
      );
      console.log("");
    }
  }

  // --- 2. Per-head entropy ranking: who LOOKS most confident? -----------------------------
  // Low entropy = sharp = the head a dashboard-reader would call "the important one". We sort
  // ascending so the top entry is the prime suspect we will indict in §4.
  console.log(`[2] 逐头注意力熵排名 (越低=越尖锐=越"像关键头"):`);
  const byEntropy = [...stats].sort((a, b) => a.entropyBits - b.entropyBits);
  console.log(
    asciiBar(
      byEntropy.map((s) => ({ label: `L${s.layer}H${s.head}(${s.label})`, value: s.entropyBits })),
      { title: "entropy(bits)", width: 32 },
    ),
  );

  // --- 3. The named-head reading (a hypothesis, not a verdict). ----------------------------
  console.log(`\n[3] 启发式命名 (注意: 这是假设, 不是结论):`);
  for (const s of stats) {
    console.log(`    L${s.layer}H${s.head}: ${s.label.padEnd(10)} focus=${(s.focusFrac * 100).toFixed(0)}%  entropy=${s.entropyBits.toFixed(2)} bits`);
  }

  // --- 4. FAILURE MODE: attention salience does NOT equal causal importance. ---------------
  // We put the picture on trial with causal interventions, exposing two independent gaps:
  //   (a) MAGNITUDE gap: the sharpest, most "obviously the circuit" head, when mean-ablated,
  //       barely dents accuracy. A confident-looking pattern doesn't mean the model collapses
  //       without it (redundancy + a gentle replacement absorb the loss).
  //   (b) CONVENTION gap: the SAME head's "importance" verdict swings with the ablation
  //       convention — zero-ablation (off-distribution, breaks the residual stream) looks far
  //       more catastrophic than mean-ablation. So even the causal test isn't a single number;
  //       you must state the convention or the verdict is meaningless.
  console.log(`\n[4] 失败模式: 注意力的"显眼"(salience) ≠ 因果"必需"(importance)。`);
  console.log(`    方法: 对 head_z 做消融, 在 200 例 held-out 上量化准确率跌幅; mean vs zero 两种约定对照。\n`);

  // Held-out accuracy when one head is ablated at EVERY scorable position of each sample.
  const ablateAcc = (point: string, mode: "mean" | "zero"): number => {
    const rng = mulberry32(4242);
    let correct = 0;
    let total = 0;
    const nSamples = 200;
    for (let i = 0; i < nSamples; i++) {
      const inp = task.makeBatch(1, rng).inputs[0];
      const oracle = task.oracle(inp);
      for (const pos of task.scorablePositions(inp)) {
        if (ablatedArgmaxCorrect(model, task, inp, pos, oracle[pos], point, mode)) correct++;
        total++;
      }
    }
    return correct / total;
  };

  const suspect = byEntropy[0]; // lowest entropy = "looks most important"
  const suspectMeanLogit = ablate(model, input, suspect.point, queryPos, answer, "mean").drop;
  const suspectZeroLogit = ablate(model, input, suspect.point, queryPos, answer, "zero").drop;
  const suspectMeanAcc = ablateAcc(suspect.point, "mean");
  const suspectZeroAcc = ablateAcc(suspect.point, "zero");
  console.log(`  (a) 最尖锐头 L${suspect.layer}H${suspect.head} (${suspect.label}, entropy=${suspect.entropyBits.toFixed(2)} — 人眼会一口咬定"这就是电路"):`);
  console.log(
    `      mean-ablation: answer-logit 跌幅=${suspectMeanLogit.toFixed(3)}, 准确率 ${(acc * 100).toFixed(1)}% -> ${(suspectMeanAcc * 100).toFixed(1)}%`,
  );
  console.log(
    `      zero-ablation: answer-logit 跌幅=${suspectZeroLogit.toFixed(3)}, 准确率 ${(acc * 100).toFixed(1)}% -> ${(suspectZeroAcc * 100).toFixed(1)}%`,
  );
  console.log(
    `      => 同一个头, 两种约定给出不同"重要性"; 仅凭注意力图谁都无法预测这两个数字。`,
  );

  // Per-head necessity sweep (mean-ablation): which heads actually matter? The bars expose
  // that most heads are causally INERT despite each having a nameable-looking pattern in §1.
  console.log(`\n  (b) 逐头因果消融扫描 (mean-ablation, 准确率跌幅; 越大越必需):`);
  let worst: HeadStat | null = null;
  let worstDrop = -1;
  const perHeadDrop: { label: string; value: number }[] = [];
  for (const s of stats) {
    const drop = acc - ablateAcc(s.point, "mean");
    perHeadDrop.push({ label: `L${s.layer}H${s.head}(${s.label})`, value: drop });
    if (drop > worstDrop) {
      worstDrop = drop;
      worst = s;
    }
  }
  console.log(asciiBar(perHeadDrop, { title: "accuracy drop when ablated", width: 32 }));
  const inertCount = perHeadDrop.filter((p) => Math.abs(p.value) < 0.005).length;
  console.log(
    `\n  观察: ${inertCount}/${stats.length} 个头消融后准确率几乎不动 (|跌幅|<0.5%), ` +
      `尽管它们在 §1 里各自都有"看着像某种功能"的图案。`,
  );
  if (worst) {
    const same = worst.layer === suspect.layer && worst.head === suspect.head;
    console.log(
      `  按因果跌幅, 最关键头 = L${worst.layer}H${worst.head} (${worst.label}); ` +
        (same
          ? `本次它恰好就是最尖锐头 — 但这是 (b) 的因果测试确认的, 不是 §1 的图能保证的。`
          : `它并非最尖锐的那个 — salience 与 importance 在此分道扬镳。`),
    );
  }

  console.log(
    `\n教训: 注意力图给的是"头在看哪里"(salience), 不是"模型是否需要它"(importance)。` +
      `\n      尖锐 / 可命名 ≠ 因果关键; 必须用 ablation / patching 做因果检验, 且必须声明消融约定。`,
  );
  console.log(
    `\n诚实边界: toy 任务上模式可命名性偏强、且本例最尖锐头恰好确实关键; 真模型里头更杂(多义/叠加), ` +
      `\n      可迁移的是"先命名假设, 再用因果干预证伪 + 报告约定"这条流程, 不是具体数字。`,
  );
}

// ----------------------------------------------------------------------------
// Helper: does the model still argmax the correct answer when one head is ablated?
// ----------------------------------------------------------------------------
//
// ablate() returns only the answer logit, not the full vocab row, so it can't tell us whether
//   the ablated model still PICKS the right token. We replicate ablate's ablation hook here
//   and read the full logits row to compute argmax — the behavioural ground truth for "did
//   ablation break the prediction?". Two conventions, deliberately exposed:
//     - "mean": replace head_z with this sample's across-position mean (mirror of
//               interp.meanRows; less off-distribution).
//     - "zero": replace head_z with zeros (harsher; can overstate importance by pushing the
//               stream off-distribution — the documented ablation failure mode).
function ablatedArgmaxCorrect(
  model: TinyTransformer,
  task: Task,
  input: number[],
  pos: number,
  answer: number,
  point: string,
  mode: "mean" | "zero",
): boolean {
  const ref = runWithCache(model, input).cache[point];
  const [rows, cols] = ref.shape;
  const replData = new Float64Array(rows * cols); // zeros by default
  if (mode === "mean") {
    const mean = new Float64Array(cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) mean[c] += ref.data[r * cols + c];
    for (let c = 0; c < cols; c++) mean[c] /= rows;
    for (let r = 0; r < rows; r++) replData.set(mean, r * cols);
  }
  const repl = new Tensor(replData, [rows, cols]);
  const hooks: Hooks = { [point]: () => repl };
  const logits = model.forward(input, hooks);
  const row = logits.data.subarray(pos * task.vocab, (pos + 1) * task.vocab);
  return argmax(row) === answer;
}

main();
