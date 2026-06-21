// stage00-landscape.ts — the map of the territory + the three headline metrics.
//
// Why this stage exists: chapter 00 must answer "why bother with RL after
// pre-training?" before any algorithm. The honest answer is operational, not
// rhetorical — the rest of the book is judged by THREE numbers, and a reader
// should see all three actually computed (not asserted) on the shared bench
// before trusting any later curve:
//
//   (a) win-rate vs a reference policy under the hidden true judge
//       — "is the tuned policy actually better than where we started?" The
//         headline post-training eval. A policy that ignores reward must land
//         at ~50% (a coin flip vs ref); anything above 50% is real signal.
//   (b) label-noise sensitivity of preference separability
//       — every reward-based method (RM/RLHF/DPO/GRPO) is bounded by how
//         cleanly the preference labels separate good from bad. We measure that
//         ceiling at 0% vs 30% annotator noise; the gap is the failure mode the
//         whole post-training half fights.
//   (c) KL-to-reference between two synthetic policies
//       — the "leash" RLHF/DPO use to stop the policy drifting into gibberish.
//         We define it in one line and print real numbers, including the
//         catastrophic Infinity case so the reader sees why direction matters.
//
// This file also doubles as a core smoke test: it exercises rng / probability /
// envs / preference / metrics, and proves bit-for-bit reproducibility by running
// the same seeded pass twice. If the two digests differ, some path touched
// Math.random and every downstream curve is untrustworthy.
//
// Honesty caveat (applies to every absolute number here): the preference world
// is synthetic and far simpler than real RLHF, so absolute values are optimistic.
// What transfers is the RELATIVE story — noise drops separability, KL bounds
// drift, win-rate measures real improvement.
//
// Run: npx tsx src/stage00-landscape.ts   (or: npm run stage00)

import { mulberry32, gaussian, sampleCategorical } from "./core/rng.js";
import { softmax, logProb, entropy, klCategorical, klFromLogprobs } from "./core/probability.js";
import { makeBandit } from "./core/envs.js";
import { makePreferenceWorld, type Prompt, type Response } from "./core/preference.js";
import { RunningMean, regret, winRateVsRef, asciiSparkline } from "./core/metrics.js";

function section(title: string): void {
  console.log("\n" + "─".repeat(64) + "\n" + title + "\n" + "─".repeat(64));
}

// One deterministic pass over the core primitives, returning a digest string of
// computed numbers. Run twice with the same seed to PROVE reproducibility: if
// the two digests differ, some code path touched Math.random or cached state and
// the whole book's "same seed => same curve" guarantee is void.
function corePass(seed: number): string {
  const rng = mulberry32(seed);
  const out: string[] = [];

  const g = [gaussian(rng), gaussian(rng), gaussian(rng)].map((x) => x.toFixed(4));
  out.push("gauss=" + g.join(","));
  out.push("cat=" + sampleCategorical([0.1, 0.2, 0.7], rng));

  const logits = [2.0, 0.5, -1.0];
  const p = softmax(logits);
  out.push("softmax0=" + p[0].toFixed(4));
  out.push("logp(a=0)=" + logProb(logits, 0).toFixed(4));
  out.push("H=" + entropy(p).toFixed(4));
  const q = softmax([1.0, 1.0, 1.0]);
  out.push("KLexact=" + klCategorical(p, q).toFixed(4));
  // Both KL estimators on a single sample — only their EXPECTATION equals KL, so
  // these per-sample values differ from KLexact; later stages average them.
  const est = klFromLogprobs(logProb(logits, 0), logProb([1.0, 1.0, 1.0], 0));
  out.push("KLk1=" + est.k1.toFixed(4) + " KLk3=" + est.k3.toFixed(4));

  return out.join(" | ");
}

// (a) Win-rate vs reference. The "policy" here is just a CHOOSER over a fixed
// candidate set per prompt; we are not training, only showing the metric. The
// reference always picks candidate index 0; the random policy picks uniformly.
// Both are scored by the hidden trueRewardFn, so a no-learning random chooser
// must land near 50% — that is the floor every later stage has to beat.
function evalWinRate(seed: number, nCandidates: number, nPrompts: number): {
  randomVsRef: number;
  goldenVsRef: number;
  sampleScores: number[];
} {
  const rng = mulberry32(seed);
  const world = makePreferenceWorld();

  // Build a fixed candidate set per prompt up front so reference / random /
  // golden choosers all see the SAME options — otherwise the comparison would
  // confound "better chooser" with "luckier candidate draw."
  const prompts: { prompt: Prompt; candidates: Response[]; trueScores: number[] }[] = [];
  for (let i = 0; i < nPrompts; i++) {
    const prompt = world.samplePrompt(rng);
    const candidates = Array.from({ length: nCandidates }, () => world.sampleResponse(prompt, rng));
    const trueScores = candidates.map((c) => world.trueRewardFn(prompt, c));
    prompts.push({ prompt, candidates, trueScores });
  }

  // Reference policy: always candidate 0 (an arbitrary fixed "starting point").
  const refScore = (e: (typeof prompts)[number]) => e.trueScores[0];
  // Random policy: pick a uniformly random candidate, score it under the truth.
  const randomScore = (e: (typeof prompts)[number]) =>
    e.trueScores[sampleCategorical(new Array(nCandidates).fill(1 / nCandidates), rng)];
  // Golden policy: an oracle that picks the truly best candidate. This is the
  // ceiling — it should win nearly always, proving the metric can see real skill
  // (a metric stuck at 50% for everyone would be broken).
  const goldenScore = (e: (typeof prompts)[number]) => Math.max(...e.trueScores);

  const randomVsRef = winRateVsRef(prompts, randomScore, refScore);
  const goldenVsRef = winRateVsRef(prompts, goldenScore, refScore);

  // A few raw (golden − ref) gaps to show win-rate is a real comparison, not a
  // hard-coded constant.
  const sampleScores = prompts.slice(0, 5).map((e) => goldenScore(e) - refScore(e));
  return { randomVsRef, goldenVsRef, sampleScores };
}

// (b) Label-noise sensitivity. "Separability" = how strongly the preference
// labels themselves order good above bad, measured WITHOUT any reward model:
// just check what fraction of pairs have trueReward(chosen) > trueReward(rejected).
// At 0% explicit noise this is already < 1.0 (Bradley–Terry sampling flips
// near-ties); at 30% it drops further. This single number is the ceiling on any
// RM trained from these labels — no optimizer recovers signal the labels lost.
function evalNoiseSeparability(seed: number, nPairs: number, flipProb: number): {
  separability: number;
  flipFraction: number;
} {
  const rng = mulberry32(seed);
  const world = makePreferenceWorld();
  const pairs = world.generatePairs(nPairs, rng, flipProb);

  let labelAgreesWithTruth = 0;
  let explicitlyFlipped = 0;
  for (const pair of pairs) {
    const rc = world.trueRewardFn(pair.prompt, pair.chosen);
    const rr = world.trueRewardFn(pair.prompt, pair.rejected);
    if (rc > rr) labelAgreesWithTruth++;
    if (pair.flipped) explicitlyFlipped++;
  }
  return {
    separability: labelAgreesWithTruth / pairs.length,
    flipFraction: explicitlyFlipped / pairs.length,
  };
}

function main(): void {
  console.log("RL & 后训练 从零 — Stage 00: 地形图 + 三个核心度量自检");
  console.log("零依赖 / 离线 / 种子 PRNG 驱动 / run-to-run 逐位可复现\n");

  // ---- 0. The unifying frame --------------------------------------------
  section("0. 统一框架：本书所有方法都是「用奖励改进策略」");
  console.log("  bandit      : 奖励=环境真值      策略=动作分布   无信用分配");
  console.log("  REINFORCE   : 奖励=环境真值      策略梯度        高方差");
  console.log("  PPO         : 奖励=环境真值      裁剪比率 + 价值基线");
  console.log("  RLHF        : 奖励=学到的 RM     PPO + KL 拴住参考策略");
  console.log("  DPO         : 奖励=隐式(偏好对)  无 RM、无采样、闭式损失");
  console.log("  GRPO        : 奖励=RM/可验证     组内相对优势、去掉价值网络");
  console.log("  RLVR        : 奖励=可验证(对/错) 推理任务、奖励无噪声");
  console.log("  → 区别只在两处：奖励从哪来、策略被什么约束。");
  console.log("  → 而判定它们好坏，全书反复用三个量：胜率 / 噪声敏感度 / KL。");

  // ---- 1. Reproducibility proof -----------------------------------------
  section("1. 复现性：同种子两次 core pass 必须逐位相同（否则有 Math.random 泄漏）");
  const passA = corePass(42);
  const passB = corePass(42);
  const passC = corePass(43);
  console.log("  seed 42 (1st): " + passA);
  console.log("  seed 42 (2nd): " + passB);
  console.log("  逐位相同      : " + (passA === passB ? "✅ 是" : "❌ 否(随机源泄漏!)"));
  console.log("  seed 43       : " + passC);
  console.log("  换种子即不同  : " + (passA !== passC ? "✅ 是" : "⚠ 否(随机源可能失效)"));

  // ---- 2. Bench is real: bandit regret computed vs known best arm --------
  section("2. 实验台真实：随机策略的 regret 真测（对照已知最优臂）");
  {
    const rng = mulberry32(7);
    const bandit = makeBandit(5, rng);
    const pulls: number[] = [];
    const mean = new RunningMean();
    for (let t = 0; t < 200; t++) {
      const arm = sampleCategorical([0.2, 0.2, 0.2, 0.2, 0.2], rng); // no learning
      const { reward } = bandit.step(0, arm, rng);
      pulls.push(arm);
      mean.push(reward);
    }
    const reg = regret(bandit, pulls);
    console.log("  真臂均值     : [" + bandit.trueMeans.map((m) => m.toFixed(2)).join(", ") + "]");
    console.log("  最优臂       : #" + bandit.optimalArm + " (均值 " + bandit.trueMeans[bandit.optimalArm].toFixed(2) + ")");
    console.log("  随机策略均奖 : " + mean.value.toFixed(4));
    console.log("  累计 regret  : " + reg[reg.length - 1].toFixed(2) + "  (随机策略≈线性增长)");
    console.log("  regret 曲线  : " + asciiSparkline(reg));
  }

  // ---- 3. Headline metric (a): win-rate vs reference ---------------------
  section("3. 度量(a) 胜率 vs 参考策略：在隐藏真值判官下，谁更好");
  {
    const { randomVsRef, goldenVsRef, sampleScores } = evalWinRate(99, 4, 1000);
    console.log("  定义：随机抽 N=4 个候选回答，参考策略固定取#0，真值判官(trueRewardFn)打分。");
    console.log("  随机策略 vs 参考 胜率 : " + randomVsRef.toFixed(3) + "  (不学习 ⇒ 必须≈0.500，平局算0.5)");
    console.log("  黄金策略 vs 参考 胜率 : " + goldenVsRef.toFixed(3) + "  (oracle 选真最优 ⇒ 远高于0.5)");
    console.log("  前5个 (黄金−参考) 真值差 : [" + sampleScores.map((s) => s.toFixed(2)).join(", ") + "]");
    console.log("  → 胜率能区分「会选」和「乱选」⇒ 是有效的后训练 eval；后续每个 stage 都要打这条线。");
  }

  // ---- 4. Headline metric (b): label-noise sensitivity ------------------
  section("4. 度量(b) 噪声敏感度：标注噪声如何砸掉偏好可分性（RM 的上限）");
  {
    const clean = evalNoiseSeparability(123, 2000, 0.0);
    const noisy = evalNoiseSeparability(123, 2000, 0.3);
    console.log("  可分性 = 标签里 trueReward(chosen)>trueReward(rejected) 的比例（不训练 RM，直接量标签质量）。");
    console.log("  0%  注入翻转 : 可分性=" + clean.separability.toFixed(3) +
      "  实际翻转占比=" + clean.flipFraction.toFixed(3) + "  (<1.0：BT 抽样在近似对上会标反)");
    console.log("  30% 注入翻转 : 可分性=" + noisy.separability.toFixed(3) +
      "  实际翻转占比=" + noisy.flipFraction.toFixed(3) + "  (噪声单调砸可分性)");
    console.log("  可分性下降幅度 : " + (clean.separability - noisy.separability).toFixed(3) +
      "  ← 这就是 RM 准确率的天花板，optimizer 再强也救不回标签丢掉的信号。");
    console.log("  → 失败模式：30% 噪声下约 1/3 的监督是反的，下游 RLHF/DPO 会照着错标签学。");
  }

  // ---- 5. Headline metric (c): KL-to-reference ---------------------------
  section("5. 度量(c) KL-to-ref：策略离参考策略多远（RLHF/DPO 的「拴绳」）");
  {
    // One-line definition for the reader, then real numbers on hand-built
    // policies so the concept is concrete before any training stage uses it.
    console.log("  一句话定义：KL(π_new ‖ π_ref) = Σ π_new(a)·log( π_new(a) / π_ref(a) )，");
    console.log("             衡量新策略相对参考策略的「信息距离」，越大=漂移越远；非对称。");

    // Reference policy: a near-uniform softmax over 3 actions.
    const ref = softmax([0.2, 0.0, -0.2]);
    // Policy A: a mild shift — same support, slightly sharper. Small finite KL.
    const polA = softmax([1.0, 0.0, -0.5]);
    // Policy B: a large shift — most mass on action 0 but still finite support
    // on every action, so KL stays finite (and visibly larger than A). We keep B
    // off the degenerate corner so its printed distribution differs from C below;
    // a too-sharp softmax rounds to [1,0,0] and hides the finite-vs-∞ contrast.
    const polB = softmax([3.0, -0.5, -1.5]);
    // Policy C: a DEGENERATE policy putting hard zero on an action the reference
    // still supports. KL is Infinity — this is exactly the "drift into a region
    // the reference never visits" catastrophe the KL leash exists to prevent.
    const polC = [1.0, 0.0, 0.0];

    console.log("  参考策略 π_ref          : [" + ref.map((x) => x.toFixed(3)).join(", ") + "]");
    console.log("  小漂移 π_A  KL(A‖ref)   = " + klCategorical(polA, ref).toFixed(4) +
      "   策略=[" + polA.map((x) => x.toFixed(3)).join(", ") + "]");
    console.log("  大漂移 π_B  KL(B‖ref)   = " + klCategorical(polB, ref).toFixed(4) +
      "   策略=[" + polB.map((x) => x.toFixed(3)).join(", ") + "]");
    console.log("  退化   π_C  KL(C‖ref)   = " + fmtKl(klCategorical(polC, ref)) +
      "       策略=[" + polC.map((x) => x.toFixed(3)).join(", ") + "] ← 在 ref 有概率处给了硬0");
    // Direction matters: KL is asymmetric. Show KL(ref‖C) is finite while
    // KL(C‖ref) blew up, so the reader sees why RLHF picks the forward direction.
    console.log("  方向不对称：KL(ref‖C)   = " + fmtKl(klCategorical(ref, polC)) +
      "    (反方向有限) ⇒ RLHF 用 KL(new‖ref) 正是为了惩罚「新策略往参考的零概率区漂」。");
  }

  // ---- 6. Honesty note ---------------------------------------------------
  section("诚实声明（全书统一）");
  console.log("  合成偏好世界远比真实 RLHF 简单，绝对数值偏乐观。");
  console.log("  可迁移的是相对趋势：");
  console.log("    · 胜率 > 0.5    ⇒ 后训练真的带来改进（=0.5 是不学习的下界）");
  console.log("    · 标注噪声↑     ⇒ 偏好可分性↓ ⇒ RM 上限↓ ⇒ 下游学到噪声");
  console.log("    · KL(new‖ref)↑  ⇒ 策略漂移越远；硬0 处→∞ 是要拴住的灾难");
  console.log("  以上三个量都由本文件实时计算，无硬编码 assert。");
}

// Render a KL value, surfacing Infinity explicitly rather than printing "Infinity"
// raw — the ∞ is a teaching point (catastrophic drift), not a bug.
function fmtKl(kl: number): string {
  return Number.isFinite(kl) ? kl.toFixed(4) + "  " : "∞ (灾难漂移)";
}

main();
