// stage01-bayes.ts — Probability & Bayes, from intuition to running code.
//
// WHY this is chapter 01: every "the model is X% confident" number a builder ships is a
//   posterior P(hypothesis | evidence). The single most expensive bug in applied
//   probability is ignoring the base rate — a 99%-accurate test for a 1-in-1000 condition
//   is mostly WRONG when it fires. This chapter makes that concrete in three runnable parts:
//     (1) a disease-test calculator + a sweep of the base rate to draw the PPV curve;
//     (2) a from-scratch naive-Bayes spam classifier (Bayes on word counts, log-domain);
//     (3) a seeded Monte-Carlo estimate of pi, showing convergence ~ 1/sqrt(n).
//   Each part prints numbers this code actually computed, plus a deliberate failure mode.
//
// HONEST-NUMBER NOTE: the spam corpus is a TINY hand-written toy. Its absolute accuracy is
//   meaningless as a benchmark — what transfers is the MECHANISM (priors x likelihoods,
//   Laplace smoothing, log-domain sum) and the qualitative failure (unseen words / no
//   smoothing -> probability collapses to 0). The Monte-Carlo numbers ARE honest estimates
//   from a seeded stream; rerun with the same seed for bit-for-bit identical output.

import { mulberry32, type Rng } from "./core/rng.js";
import { mean } from "./core/stats.js";
import { asciiBar, asciiLine } from "./core/plot.js";

// ---------------------------------------------------------------------------
// Part 1 — Disease test: the base-rate trap, analytic + Monte-Carlo cross-check.
// ---------------------------------------------------------------------------

/**
 * Posterior P(disease | positive) via Bayes' theorem.
 *   prior = P(disease) (the base rate), sens = P(positive|disease),
 *   spec = P(negative|healthy). This is the "positive predictive value" (PPV).
 * INVARIANT: all three in [0,1]. The denominator is total P(positive); it is 0 only if the
 *   test literally never fires (sens=0 and 1-spec=0), which we guard against.
 */
function posteriorPositive(prior: number, sens: number, spec: number): number {
  const fpr = 1 - spec; // false-positive rate P(positive | healthy)
  const pPos = sens * prior + fpr * (1 - prior); // law of total probability
  if (pPos === 0) throw new Error("test never returns positive; posterior undefined");
  return (sens * prior) / pPos;
}

/**
 * Monte-Carlo cross-check: simulate a population, keep only those who tested positive,
 * and measure what fraction actually have the disease.
 * WHY simulate when we have a closed form: it proves the closed form is not a derivation
 *   slip. If analytic and simulated disagree beyond sampling noise, ONE of them is wrong —
 *   and a reader can see which. Returns NaN if no positive was sampled (the rare-event trap).
 */
function simulatePosterior(prior: number, sens: number, spec: number, n: number, rng: Rng): number {
  const diseaseGivenPositive: number[] = [];
  for (let i = 0; i < n; i++) {
    const hasDisease = rng() < prior;
    const testsPositive = hasDisease ? rng() < sens : rng() < 1 - spec;
    if (testsPositive) diseaseGivenPositive.push(hasDisease ? 1 : 0);
  }
  if (diseaseGivenPositive.length === 0) return NaN; // no positives sampled at all
  return mean(diseaseGivenPositive);
}

function runDiseaseTest(): void {
  const rng = mulberry32(20260619);
  const prior = 0.001; // 1 in 1000 — the base rate intuition forgets
  const sens = 0.99;
  const spec = 0.99;

  const analytic = posteriorPositive(prior, sens, spec);
  const mc = simulatePosterior(prior, sens, spec, 2_000_000, rng);

  console.log("=== Part 1 · 疾病检测与基率陷阱 ===\n");
  console.log(`先验 (基率) P(病)      = ${prior}`);
  console.log(`灵敏度 P(阳|病)         = ${sens}`);
  console.log(`特异度 P(阴|健康)       = ${spec}\n`);
  console.log(`解析 PPV  P(病|阳)      = ${(analytic * 100).toFixed(2)}%`);
  console.log(`蒙特卡洛 PPV (2e6 抽样) = ${(mc * 100).toFixed(2)}%`);
  console.log(`两者之差               = ${(Math.abs(analytic - mc) * 100).toFixed(3)} 个百分点`);
  console.log("\n直觉答案约 99%，真实 PPV 约 9% —— 这 90 个百分点的差就是基率陷阱。\n");
  console.log(asciiBar(["直觉(误)", "真实PPV", "灵敏度"], [0.99, analytic, sens]));

  // --- The PPV curve: how the posterior rises as the base rate rises. ---
  // WHY sweep the base rate: the whole trap is that PPV depends MOSTLY on the base rate, not
  //   on test accuracy. Fixing sens/spec and sweeping prior shows PPV climbing from ~9% (rare
  //   disease) toward ~99% (common disease) — the same 99% test, wildly different trust.
  console.log("\n--- PPV vs 患病率（灵敏度/特异度固定在 99%）---");
  const priors: number[] = [];
  const ppvs: number[] = [];
  for (let i = 0; i <= 50; i++) {
    const p = i / 50; // 0 .. 1 inclusive
    priors.push(p);
    ppvs.push(posteriorPositive(p, sens, spec));
  }
  console.log(asciiLine(ppvs, 51, 12));
  // Print a few anchor points so the curve's absolute values are readable, not just its shape.
  for (const p of [0.001, 0.01, 0.05, 0.5]) {
    console.log(`  患病率 ${(p * 100).toFixed(1).padStart(4)}% → PPV ${(posteriorPositive(p, sens, spec) * 100).toFixed(1)}%`);
  }

  // --- Failure mode: a tiny simulation under-samples the rare positives. ---
  console.log("\n--- 失败模式 A：样本太小时蒙特卡洛不可信 ---");
  const small = simulatePosterior(prior, sens, spec, 500, mulberry32(7));
  console.log(`n=500 时的 PPV 估计     = ${Number.isNaN(small) ? "NaN (没采到阳性)" : (small * 100).toFixed(2) + "%"}`);
  console.log("罕见事件下小样本要么估计剧烈抖动、要么一个阳性都没采到 → 必须用解析式或大样本。");
}

// ---------------------------------------------------------------------------
// Part 2 — Naive-Bayes spam classifier from scratch (Bayes on bag-of-words).
// ---------------------------------------------------------------------------

interface NaiveBayesModel {
  logPrior: Record<"spam" | "ham", number>; // log P(class)
  logLikelihood: Record<"spam" | "ham", Map<string, number>>; // log P(word | class)
  logUnseen: Record<"spam" | "ham", number>; // log-prob mass for a word never seen in class
  vocab: Set<string>;
}

/** Lowercase + split on non-letters. Toy tokenizer; real ones handle unicode/stemming. */
function tokenize(text: string): string[] {
  return text.toLowerCase().split(/[^a-z]+/).filter((w) => w.length > 0);
}

/**
 * Train a multinomial naive-Bayes classifier with Laplace (add-one) smoothing.
 * WHY "naive": we assume words are conditionally independent given the class. That is FALSE
 *   for real language, yet the classifier works well because we only need the argmax to land
 *   on the right side, not calibrated probabilities.
 * WHY log-domain: multiplying hundreds of small probabilities underflows float64 to 0; summing
 *   their logs does not. Every score below is a sum of logs, compared, never exponentiated.
 * WHY Laplace smoothing (the +1): without it, a single word never seen in a class gives that
 *   class likelihood 0, which annihilates the whole product regardless of all other evidence.
 *   See failure mode B for what happens when we turn it off.
 * INVARIANT: each class's word-likelihoods + the unseen-word fallback form a proper
 *   distribution over (vocab + 1) outcomes: counts get +1, denominator gets +(|vocab|+1).
 */
function trainNaiveBayes(docs: { text: string; label: "spam" | "ham" }[], smoothing: number): NaiveBayesModel {
  const vocab = new Set<string>();
  const counts: Record<"spam" | "ham", Map<string, number>> = { spam: new Map(), ham: new Map() };
  const totalWords: Record<"spam" | "ham", number> = { spam: 0, ham: 0 };
  const docCount: Record<"spam" | "ham", number> = { spam: 0, ham: 0 };

  for (const { text, label } of docs) {
    docCount[label]++;
    for (const w of tokenize(text)) {
      vocab.add(w);
      counts[label].set(w, (counts[label].get(w) ?? 0) + 1);
      totalWords[label]++;
    }
  }

  const v = vocab.size;
  const n = docs.length;
  const logPrior = { spam: Math.log(docCount.spam / n), ham: Math.log(docCount.ham / n) };
  const logLikelihood = { spam: new Map<string, number>(), ham: new Map<string, number>() };
  const logUnseen = { spam: 0, ham: 0 };

  for (const cls of ["spam", "ham"] as const) {
    // Denominator: total words in class + smoothing * (vocab + 1 unseen slot).
    const denom = totalWords[cls] + smoothing * (v + 1);
    for (const w of vocab) {
      const c = counts[cls].get(w) ?? 0;
      logLikelihood[cls].set(w, Math.log((c + smoothing) / denom));
    }
    // A word outside the training vocab still needs a probability; with smoothing it is
    // (0 + smoothing)/denom > 0, with smoothing=0 it is log(0) = -Infinity (the failure mode).
    logUnseen[cls] = Math.log(smoothing / denom);
  }
  return { logPrior, logLikelihood, logUnseen, vocab };
}

/** Return the argmax class and the two log-scores (for inspecting confidence / failures). */
function classify(model: NaiveBayesModel, text: string): { label: "spam" | "ham"; scores: Record<"spam" | "ham", number> } {
  const scores: Record<"spam" | "ham", number> = { spam: model.logPrior.spam, ham: model.logPrior.ham };
  for (const w of tokenize(text)) {
    for (const cls of ["spam", "ham"] as const) {
      scores[cls] += model.logLikelihood[cls].get(w) ?? model.logUnseen[cls];
    }
  }
  return { label: scores.spam > scores.ham ? "spam" : "ham", scores };
}

function runSpamClassifier(): void {
  console.log("\n\n=== Part 2 · 从零朴素贝叶斯垃圾邮件分类器 ===\n");

  // Toy corpus — deliberately small so the mechanism is inspectable, not a benchmark.
  const train: { text: string; label: "spam" | "ham" }[] = [
    { text: "win a free prize now click here", label: "spam" },
    { text: "free money claim your cash prize", label: "spam" },
    { text: "winner you win cash click to claim", label: "spam" },
    { text: "limited offer buy now free shipping", label: "spam" },
    { text: "cheap pills order now discount offer", label: "spam" },
    { text: "lunch meeting tomorrow at noon", label: "ham" },
    { text: "can you review the project report", label: "ham" },
    { text: "the meeting is moved to friday", label: "ham" },
    { text: "thanks for sending the invoice", label: "ham" },
    { text: "see you at the office tomorrow", label: "ham" },
  ];
  const test: { text: string; label: "spam" | "ham" }[] = [
    { text: "claim your free prize now", label: "spam" },
    { text: "win cash click the offer", label: "spam" },
    { text: "are we still meeting tomorrow", label: "ham" },
    { text: "please review the report", label: "ham" },
    { text: "free shipping on your order", label: "spam" },
    { text: "lunch at noon friday", label: "ham" },
  ];

  const model = trainNaiveBayes(train, 1); // Laplace add-one smoothing
  let correct = 0;
  console.log("预测明细：");
  for (const { text, label } of test) {
    const { label: pred, scores } = classify(model, text);
    const hit = pred === label;
    if (hit) correct++;
    const margin = (scores.spam - scores.ham).toFixed(2);
    console.log(`  [${hit ? "✓" : "✗"}] 真实=${label.padEnd(4)} 预测=${pred.padEnd(4)} logΔ(spam-ham)=${margin.padStart(7)}  "${text}"`);
  }
  const acc = correct / test.length;
  console.log(`\n训练集 ${train.length} 封 / 测试集 ${test.length} 封，分类准确率 = ${(acc * 100).toFixed(1)}% (${correct}/${test.length})`);
  console.log("注：toy 语料，绝对准确率无意义；可迁移的是「先验×似然、log 域求和、Laplace 平滑」这套机制。");

  // --- Failure mode B: drop smoothing -> a single unseen word collapses the score to -Inf. ---
  console.log("\n--- 失败模式 B：关掉 Laplace 平滑，未见词把概率打成 0 ---");
  const noSmooth = trainNaiveBayes(train, 0);
  const tricky = "lottery jackpot tomorrow"; // "lottery"/"jackpot" never appear in training
  const { scores } = classify(noSmooth, tricky);
  console.log(`  文本 "${tricky}"`);
  console.log(`  log P(spam) = ${scores.spam}   log P(ham) = ${scores.ham}`);
  console.log("  两个类都得到 -Infinity（log 0）：未见词 P(word|class)=0 湮灭了整个乘积 → argmax 退化、无法判别。");
  console.log("  这就是为什么必须平滑：给从未见过的事件留一点非零概率质量。");
}

// ---------------------------------------------------------------------------
// Part 3 — Monte-Carlo estimate of pi (seeded), and its convergence rate.
// ---------------------------------------------------------------------------

/**
 * Estimate pi by throwing darts into the unit square and counting the fraction inside the
 * quarter unit circle. P(inside) = area_quarter_circle / area_square = (pi/4)/1, so pi ~= 4*fraction.
 * WHY this is honest Monte-Carlo: the estimator is unbiased; its standard error shrinks like
 *   1/sqrt(n). That sqrt is the whole lesson — to gain one more correct digit you need ~100x
 *   the samples. Returns the running estimate at each requested checkpoint.
 */
function estimatePi(rng: Rng, checkpoints: number[]): { n: number; estimate: number; absError: number }[] {
  const maxN = checkpoints[checkpoints.length - 1];
  const sorted = new Set(checkpoints);
  const out: { n: number; estimate: number; absError: number }[] = [];
  let inside = 0;
  for (let i = 1; i <= maxN; i++) {
    const x = rng();
    const y = rng();
    if (x * x + y * y <= 1) inside++;
    if (sorted.has(i)) {
      const est = (4 * inside) / i;
      out.push({ n: i, estimate: est, absError: Math.abs(est - Math.PI) });
    }
  }
  return out;
}

function runMonteCarloPi(): void {
  console.log("\n\n=== Part 3 · 蒙特卡洛估计 π（seeded，可复现）===\n");
  const rng = mulberry32(424242);
  const checkpoints = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];
  const rows = estimatePi(rng, checkpoints);

  console.log("样本数 n        π 估计        绝对误差      理论误差≈1/√n");
  for (const { n, estimate, absError } of rows) {
    const theo = 1 / Math.sqrt(n); // crude scale of the standard error, for comparison
    console.log(`  ${String(n).padStart(10)}   ${estimate.toFixed(6)}    ${absError.toFixed(6)}    ${theo.toFixed(6)}`);
  }
  console.log(`\n真值 π = ${Math.PI.toFixed(6)}`);
  console.log("收敛曲线（误差随样本数下降，注意纵轴是绝对误差）：");
  console.log(asciiLine(rows.map((r) => r.absError), rows.length, 10));
  console.log("误差大致按 1/√n 收缩：每多一位准确数字，约需 100 倍样本 —— 这是蒙特卡洛的根本代价。");

  // --- Failure mode C: reusing one rng draw for both x and y correlates the coordinates. ---
  // WHY this is a real bug: if x and y are the SAME number, every dart lands on the diagonal
  //   y=x, and "inside the circle" reduces to 2x^2<=1 i.e. x<=1/sqrt(2)~=0.707, giving pi~=2.83,
  //   not 3.14. The estimator silently converges to the WRONG constant — no crash, just lies.
  console.log("\n--- 失败模式 C：x 与 y 用同一个随机数（坐标相关）→ 收敛到错误常数 ---");
  const badRng = mulberry32(424242);
  let badInside = 0;
  const badN = 1_000_000;
  for (let i = 0; i < badN; i++) {
    const x = badRng();
    const y = x; // BUG on purpose: y must be an INDEPENDENT draw, not a copy of x
    if (x * x + y * y <= 1) badInside++;
  }
  console.log(`  错误估计 (x=y, n=${badN}) = ${((4 * badInside) / badN).toFixed(6)}  (应为 ${Math.PI.toFixed(6)})`);
  console.log("  独立性被破坏后估计器不报错，只是稳稳地收敛到错的数 → Monte-Carlo 必须保证维度间独立。");
}

function main(): void {
  console.log("=== Stage 01 · 概率与贝叶斯：从直觉到代码 ===\n");
  runDiseaseTest();
  runSpamClassifier();
  runMonteCarloPi();
}

main();
