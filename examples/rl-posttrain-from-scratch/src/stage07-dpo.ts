// stage07-dpo.ts — Direct Preference Optimization, head-to-head against RLHF.
//
// The chapter's thesis: DPO reaches the same place RLHF does (a policy leashed to
// a reference by an implicit KL) but throws away two moving parts — the separate
// reward model AND the on-policy sampling loop. This file proves that empirically
// on the SAME synthetic preference world stage 06 used, by running both methods
// to convergence and printing the four numbers that matter side by side:
//   - true reward   : quality under the hidden judge (what we actually want)
//   - win-rate vs ref: fraction of prompts the tuned policy beats the SFT ref on
//   - KL(π||ref)     : how far the policy drifted (the leash quantity)
//   - forward passes : a hardware-free cost proxy (DPO's whole selling point)
//
// Why the comparison is fair-but-not-rigged: both methods optimize over the SAME
// discrete menu and pin to the SAME good-SFT reference, so KL and win-rate are
// directly comparable. The ONE deliberate asymmetry is the lesson — RLHF
// optimizes a flawed *reward model* (length-overweighting, hackable), while DPO
// optimizes *preference pairs* labeled by the true judge. That asymmetry is why
// DPO's true reward comes out higher here: it never has to launder its signal
// through a mis-calibrated RM. The book is explicit that a *good* RM closes this
// gap; the point is that DPO removes a thing that can go wrong, not that it is
// magic.
//
// Experiments, each demoing a MEASURED claim (not happy-path only); the failure
// modes below are written to match what the code actually prints, not folklore:
//   [A] head-to-head at clean labels — DPO matches/beats RLHF at lower cost.
//   [B] the β sweep overlaid — DPO's β and RLHF's KL coefficient act as the SAME
//       leash: KL(π||ref) shrinks monotonically as β grows in BOTH. (Caveat below
//       in [C1]: the leash DIRECTION is inverted between the two formulations.)
//   [C1] DPO's real β failure: LARGE β saturates the logistic gradient (g->0)
//        early, so the policy UNDER-trains and stays stuck near a mediocre
//        reference => true reward drops. This is the opposite of RLHF, where a
//        large KL coefficient is the SAFE/conservative end. Surfacing that
//        inversion is the point — students routinely set DPO β like an RLHF KL
//        coefficient and get under-trained models.
//   [C2] DPO-flavored reward hacking, shown HONESTLY: it needs a WEAK (uniform)
//        reference, not a small β. With a weak anchor + moderate β, DPO overfits
//        noisy pairs and true reward collapses. With the good SFT reference of
//        [A]/[B] the small-β end is actually the most noise-ROBUST, so we do not
//        pretend otherwise.
//   [C3] robustness at 30% label noise: how much DPO's true reward drops, and why
//        a hard-baked proxy RM does not (and what that omission hides).
//
// Determinism: every sampled prompt/pair flows from a seeded PRNG; same seed in =>
// identical tables out (the scaffold note for the book verifies bit-identical
// reruns). RLHF here is pure gradient ascent over the menu and takes no rng.
//
// Run: npm run stage07

import { mulberry32, type Rng } from "./core/rng.js";
import {
  makePreferenceWorld,
  type Response,
  type Prompt,
  type PreferenceWorld,
} from "./core/preference.js";
import { winRateVsRef, asciiSparkline } from "./core/metrics.js";

// Shared discrete action space for BOTH methods, identical to stage 06 so the two
// chapters are directly comparable. #1 (len12,kw3) sits near the golden length =>
// it is the genuinely-good answer; #3 (len40) is the catastrophic over-long hack
// target that a length-overweighting RM loves but the true judge punishes hard.
const MENU: Response[] = [
  { length: 6, keywordHits: 1 },
  { length: 12, keywordHits: 3 }, // near-golden: the genuinely good answer
  { length: 24, keywordHits: 3 }, // padded: high proxy reward, mediocre truth
  { length: 40, keywordHits: 2 }, // very long: the hack target
];

// Shared reference policy = the pre-tuning SFT model. NOT uniform: a real SFT
// model already leans toward good answers, so we concentrate its mass on #1. This
// matters for BOTH methods — the KL leash only means "stay good" when the thing
// you are leashed to is itself good. (Uniform ref would make win-rate saturate at
// 1.0 because the ref averages in the len40 disaster, hiding all signal; the old
// scaffold hit exactly that trap.) logits, not probs:
const REF_LOGITS = [0, 2.0, 0, 0];

function softmaxLocal(logits: number[]): number[] {
  const m = Math.max(...logits);
  const e = logits.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function klExact(p: number[], q: number[]): number {
  // KL(p||q) over the menu. p[i]=0 contributes 0 (limit x·log x -> 0); q[i]>0
  // always holds here (softmax of finite logits), so no Infinity guard needed.
  let kl = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) kl += p[i] * Math.log(p[i] / q[i]);
  return kl;
}

function menuIndex(r: Response): number {
  // Snap a continuously-sampled response to the nearest menu slot (L1 on the two
  // features) so generated preference pairs land in the discrete space the toy
  // policy optimizes over. Without this, pairs would reference responses the
  // policy cannot represent and the gradient would have nothing to push on.
  let best = 0,
    bd = Infinity;
  for (let i = 0; i < MENU.length; i++) {
    const d =
      Math.abs(MENU[i].length - r.length) + Math.abs(MENU[i].keywordHits - r.keywordHits);
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  return best;
}

// A crude reward model that overweights length — identical bias to stage 06. This
// is the hackable proxy RLHF must optimize. Monotone increasing in length => a
// policy chasing it walks toward #3 (len40), away from the true peak at len12.
// DPO never touches this; it is here only so RLHF has the same handicap a real
// learned RM would impose.
function rmScore(r: Response): number {
  return 0.5 * r.keywordHits + 0.08 * r.length;
}

// trueReward of a policy distribution, averaged over a prompt set. This is the
// EXPECTED reward under the full distribution, not the argmax — a policy that is
// merely less confident (more mass on bad menu items, which is what label noise
// causes) genuinely scores lower. Argmax would hide that rot behind a stable #1.
function meanTrueReward(world: PreferenceWorld, dist: number[], prompts: Prompt[]): number {
  let total = 0;
  for (const p of prompts) {
    let e = 0;
    for (let a = 0; a < MENU.length; a++) e += dist[a] * world.trueRewardFn(p, MENU[a]);
    total += e;
  }
  return total / prompts.length;
}

interface MethodResult {
  trueReward: number; // mean expected true reward over eval prompts
  winRate: number; // vs the SFT reference, judged by trueRewardFn
  kl: number; // KL(policy||ref) over the menu
  forwardPasses: number; // cost proxy: # of per-action reward/logprob evals
  probs: number[];
}

// ---------------------------------------------------------------------------
// RLHF: PPO-style gradient ascent on RM reward minus β·KL(policy||ref).
// Deterministic (no sampling here — the menu is small enough to optimize the
// expected objective in closed form), so it takes no rng.
// ---------------------------------------------------------------------------
function runRLHF(
  world: PreferenceWorld,
  beta: number,
  prompts: Prompt[],
): MethodResult {
  const refProbs = softmaxLocal(REF_LOGITS);
  let logits = REF_LOGITS.slice();
  const lr = 0.05;
  const STEPS = 2000;
  let forwardPasses = 0;
  for (let step = 0; step < STEPS; step++) {
    const probs = softmaxLocal(logits);
    const grad = new Array<number>(MENU.length).fill(0);
    for (let a = 0; a < MENU.length; a++) {
      // Per-action objective: RM(a) - β·(logπ(a) - logπ_ref(a)). The β term is the
      // entire leash; β->0 chases the RM (and hacks toward len40), β->∞ freezes at
      // the reference.
      const reward =
        rmScore(MENU[a]) -
        beta * (Math.log(probs[a] + 1e-12) - Math.log(refProbs[a] + 1e-12));
      for (let k = 0; k < MENU.length; k++)
        grad[k] += probs[a] * ((a === k ? 1 : 0) - probs[k]) * reward;
      forwardPasses++; // one RM evaluation per action per step (the dominant cost)
    }
    for (let k = 0; k < MENU.length; k++) logits[k] += lr * grad[k];
  }
  const probs = softmaxLocal(logits);
  // RLHF in practice also pays for TRAINING the RM first (stage 05). We add that
  // one-time cost so the proxy is honest: a real RM is trained on the same pairs
  // DPO trains on, i.e. RM_EPOCHS * nPairs forward passes, before PPO even starts.
  const RM_EPOCHS = 30;
  const RM_PAIRS = 1500;
  forwardPasses += RM_EPOCHS * RM_PAIRS * 2; // 2 = score(chosen)+score(rejected)
  return {
    trueReward: meanTrueReward(world, probs, prompts),
    winRate: winRateVsRef(
      prompts,
      (p) => expectedReward(world, probs, p),
      (p) => expectedReward(world, refProbs, p),
    ),
    kl: klExact(probs, refProbs),
    forwardPasses,
    probs,
  };
}

function expectedReward(world: PreferenceWorld, dist: number[], p: Prompt): number {
  let e = 0;
  for (let a = 0; a < MENU.length; a++) e += dist[a] * world.trueRewardFn(p, MENU[a]);
  return e;
}

// ---------------------------------------------------------------------------
// DPO: optimize the policy DIRECTLY on (chosen, rejected) pairs with the closed-
// form loss. No reward model, no rollouts. The loss is
//   -log σ( β·[ (logπ(yw)-logπ_ref(yw)) - (logπ(yl)-logπ_ref(yl)) ] )
// The reference logprobs are the "implicit reward" baseline that makes β a KL
// leash — the exact same β that RLHF spends on its explicit KL penalty.
// ---------------------------------------------------------------------------
function runDPO(
  world: PreferenceWorld,
  beta: number,
  flipProb: number,
  evalPrompts: Prompt[],
  rng: Rng,
): MethodResult {
  // Default DPO uses the good SFT reference; runDPOFrom lets [C2] swap in a weak
  // (uniform) anchor to expose the noise-overfitting failure.
  return runDPOFrom(world, REF_LOGITS, beta, flipProb, evalPrompts, rng);
}

function runDPOFrom(
  world: PreferenceWorld,
  refLogits: number[],
  beta: number,
  flipProb: number,
  evalPrompts: Prompt[],
  rng: Rng,
): MethodResult {
  const refProbs = softmaxLocal(refLogits);
  const refLogp = (a: number) => Math.log(refProbs[a] + 1e-12);
  let logits = refLogits.slice();
  const lr = 0.05;
  const EPOCHS = 30;
  const N_PAIRS = 1500;
  const pairs = world.generatePairs(N_PAIRS, rng, flipProb).map((p) => ({
    yw: menuIndex(p.chosen),
    yl: menuIndex(p.rejected),
  }));
  let forwardPasses = 0;
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for (const { yw, yl } of pairs) {
      if (yw === yl) continue; // both snapped to same menu slot: no preference signal
      const probs = softmaxLocal(logits);
      const lp = probs.map((x) => Math.log(x + 1e-12));
      const margin = beta * ((lp[yw] - refLogp(yw)) - (lp[yl] - refLogp(yl)));
      // g = 1 - σ(margin): the logistic loss gradient scale. Big when the policy
      // has NOT yet separated chosen from rejected; -> 0 once it has, which is the
      // self-regularizing part. With small β the margin saturates slowly, so the
      // policy keeps pushing on every pair INCLUDING the mislabeled ones — that is
      // the mechanism behind the [C] overfitting failure mode.
      const g = 1 / (1 + Math.exp(margin));
      for (let k = 0; k < logits.length; k++) {
        const dYw = (k === yw ? 1 : 0) - probs[k];
        const dYl = (k === yl ? 1 : 0) - probs[k];
        logits[k] += lr * beta * g * (dYw - dYl);
      }
      forwardPasses += 2; // one logprob eval for yw and yl (the dominant cost)
    }
  }
  const probs = softmaxLocal(logits);
  return {
    trueReward: meanTrueReward(world, probs, evalPrompts),
    winRate: winRateVsRef(
      evalPrompts,
      (p) => expectedReward(world, probs, p),
      (p) => expectedReward(world, refProbs, p),
    ),
    kl: klExact(probs, refProbs),
    forwardPasses,
    probs,
  };
}

function pickIdx(probs: number[]): number {
  let p = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[p]) p = i;
  return p;
}

function fmtRow(label: string, r: MethodResult): string {
  return (
    label.padEnd(10) +
    "| " +
    r.trueReward.toFixed(3).padStart(7) +
    " | " +
    r.winRate.toFixed(3) +
    " | " +
    r.kl.toFixed(3).padStart(6) +
    " | " +
    String(r.forwardPasses).padStart(8) +
    " | #" +
    pickIdx(r.probs) +
    "(" +
    r.probs[pickIdx(r.probs)].toFixed(3) +
    ")"
  );
}

function main(): void {
  console.log("Stage 07 — DPO：扔掉奖励模型，直接从偏好优化（对照 RLHF）\n");
  const world = makePreferenceWorld();
  const rng = mulberry32(7);
  // One fixed eval prompt set shared by every method/config so all true-reward
  // and win-rate numbers are comparable across the whole stage.
  const evalPrompts = Array.from({ length: 200 }, () => world.samplePrompt(rng));

  console.log("menu: " + MENU.map((m, i) => `#${i}[len${m.length},kw${m.keywordHits}]`).join(" "));
  console.log(
    "参考策略=SFT(质量集中在 #1，接近 golden len=" +
      world.goldenLength +
      ")；真奖励峰值也在 #1。\n",
  );

  // ---- [A] head-to-head at clean labels -----------------------------------
  console.log("【A】干净标签下 RLHF vs DPO（同参考、同 menu、同 eval 集）");
  console.log("方法      | 真奖励  | 胜率  |   KL   | 前向次数 | 策略选中项");
  const betaA = 0.5;
  const rlhfA = runRLHF(world, betaA, evalPrompts);
  const dpoA = runDPO(world, betaA, 0.0, evalPrompts, mulberry32(7));
  console.log(fmtRow("RLHF β" + betaA, rlhfA));
  console.log(fmtRow("DPO  β" + betaA, dpoA));
  const speedup = rlhfA.forwardPasses / dpoA.forwardPasses;
  console.log(
    "→ 前向次数：RLHF 含「训 RM(30×1500×2) + PPO rollout」两笔，DPO 只有一笔 pair 训练。",
  );
  console.log(
    "→ 成本代理：DPO 约为 RLHF 的 1/" +
      speedup.toFixed(2) +
      "（少一个 RM 训练 + 少一圈采样）。",
  );
  console.log(
    "→ 真奖励 DPO≥RLHF：RLHF 的信号要过「会偏好长答案」的代理 RM，DPO 直接吃真判官标的偏好。\n",
  );

  // ---- [B] the β / KL-coefficient sweep, overlaid -------------------------
  console.log("【B】β 扫描：DPO 的 β 与 RLHF 的 KL 系数是同一根缰绳（KL 随 β 增大而收紧）");
  const betas = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0];
  const rlhfKl: number[] = [];
  const dpoKl: number[] = [];
  console.log("  β   | RLHF KL | DPO KL | RLHF真奖 | DPO真奖");
  for (const b of betas) {
    const rr = runRLHF(world, b, evalPrompts);
    const dd = runDPO(world, b, 0.0, evalPrompts, mulberry32(7));
    rlhfKl.push(rr.kl);
    dpoKl.push(dd.kl);
    console.log(
      "  " +
        b.toFixed(1) +
        " |  " +
        rr.kl.toFixed(3) +
        "  | " +
        dd.kl.toFixed(3) +
        " |  " +
        rr.trueReward.toFixed(3).padStart(6) +
        "  | " +
        dd.trueReward.toFixed(3).padStart(6),
    );
  }
  console.log("  RLHF KL(β↑→) " + asciiSparkline(rlhfKl) + "  (左高右低=β 越大漂移越小)");
  console.log("  DPO  KL(β↑→) " + asciiSparkline(dpoKl) + "  (同向：β 是同一根 KL 缰绳)");
  console.log(
    "→ 两条 KL 曲线同向单调下降——RLHF 的「KL 系数」与 DPO 的 β 控制同一个量：离参考多远。\n",
  );

  // ---- [C1] DPO's real β failure: LARGE β under-trains (gradient saturates) -
  // NOTE: this contradicts the common "small β overfits" intuition borrowed from
  // RLHF, and the table below is exactly why. In the DPO loss β multiplies the
  // margin, so large β makes g=1-σ(margin) hit 0 fast => updates die before the
  // policy finishes sharpening onto the good mode #1. Small β keeps g near 0.5 for
  // many steps => the policy fully converges to pure #1. So in DPO, SMALL β is the
  // aggressive/expressive end and LARGE β is the under-trained/conservative end —
  // inverted vs RLHF's KL coefficient. We print clean true reward + the policy's
  // mass on #1 to make the under-training visible.
  console.log("【C1】失败模式（真实）：DPO 的 β 太大 → 对数 sigmoid 梯度提前饱和 → 欠训练，卡在平庸参考附近");
  console.log("  说明：参考策略真奖励=" + meanTrueReward(world, softmaxLocal(REF_LOGITS), evalPrompts).toFixed(3) +
    "（含 9.6% 灾难性 #3），纯 #1 真奖励=" + meanTrueReward(world, [0, 1, 0, 0], evalPrompts).toFixed(3) + "（最优）");
  console.log("  β   | 干净真奖励 |  #1 概率  | 诊断");
  const c1Betas = [0.1, 0.5, 1.0, 3.0];
  for (const b of c1Betas) {
    const r = runDPO(world, b, 0.0, evalPrompts, mulberry32(7));
    const diag = b <= 0.1 ? "充分收敛到纯 #1" : b >= 3.0 ? "梯度早死，几乎没离开参考" : "";
    console.log(
      "  " + b.toFixed(1) + " |   " + r.trueReward.toFixed(3).padStart(7) +
        "  |  " + r.probs[1].toFixed(3) + "  | " + diag,
    );
  }
  console.log("→ β↑ → margin 一上来就大 → g=1-σ(margin)→0 → 更新停得太早 → #1 概率上不去 → 真奖励掉。");
  console.log("→ 关键反直觉：DPO 里 β 大 = 欠训练（保守），与 RLHF「KL 系数大=保守」方向相反，别照搬调参直觉。\n");

  // ---- [C2] DPO-flavored reward hacking: weak reference + noisy pairs --------
  // The honest version of "DPO overfits preference noise." It does NOT happen at
  // small β with a good reference (that end is the most robust — see C3). It needs
  // a WEAK anchor: a uniform reference gives the policy nothing to fall back on, so
  // mislabeled pairs pull it onto genuinely-bad menu items. We rerun DPO from a
  // uniform reference (not the good SFT one) to expose it; everything else is held.
  console.log("【C2】失败模式（DPO 版 reward hacking）：参考策略弱(均匀) + 脏标签 → 过拟合噪声，真奖励崩");
  console.log("  (对照：参考=均匀，没有好答案可回退；β 适中让策略敢动)");
  console.log("  β   | 翻转0% | 翻转30% | 噪声损失");
  const weakRef = [0, 0, 0, 0]; // uniform: a weak anchor with nothing good to hold onto
  for (const b of [0.3, 0.5, 1.0]) {
    const clean = runDPOFrom(world, weakRef, b, 0.0, evalPrompts, mulberry32(7)).trueReward;
    const noisy = runDPOFrom(world, weakRef, b, 0.3, evalPrompts, mulberry32(7)).trueReward;
    console.log(
      "  " + b.toFixed(1) + " | " + clean.toFixed(3).padStart(7) + " | " +
        noisy.toFixed(3).padStart(7) + " |  " + (clean - noisy).toFixed(3),
    );
  }
  console.log("→ 弱参考下没有「好答案」当缰绳锚点，被翻转标错的对直接把策略拽到坏菜单项上 → 真奖励崩。");
  console.log("→ 这就是「DPO 没有 RM 也会 hacking」：可被污染的信号从 RM 换成了偏好标签本身。");
  console.log("→ 推论：DPO 的隐式 KL 缰绳只在「锚点本身够好」时才保命——好 SFT 参考是 DPO 的前提条件。\n");

  // ---- [C3] robustness at 30% label noise: DPO vs RLHF --------------------
  console.log("【C3】30% 标签噪声下 RLHF vs DPO（同 β=0.5，量化对比）");
  const betaC = 0.5;
  const rlhfClean = runRLHF(world, betaC, evalPrompts).trueReward;
  const dpoClean = runDPO(world, betaC, 0.0, evalPrompts, mulberry32(7)).trueReward;
  const dpoNoisy = runDPO(world, betaC, 0.3, evalPrompts, mulberry32(7)).trueReward;
  console.log("  RLHF (写死的代理RM，与标签翻转解耦) 真奖励 " + rlhfClean.toFixed(3));
  console.log("  DPO  翻转0%  真奖励 " + dpoClean.toFixed(3));
  console.log("  DPO  翻转30% 真奖励 " + dpoNoisy.toFixed(3));
  const dpoDrop = ((dpoClean - dpoNoisy) / Math.abs(dpoClean)) * 100;
  console.log(
    "→ DPO 真奖励因 30% 噪声下降 " + dpoDrop.toFixed(1) +
      "%：DPO 直连标签，标注质量是它的硬上限（garbage in, garbage out）。",
  );
  console.log("→ 诚实说明：本表 RLHF 用「写死的代理 RM」，其偏置与标签翻转解耦，故数字不随翻转率变——");
  console.log("  这不代表 RLHF 真的抗噪。真实里 RM 要在脏对上训练而退化（stage05：翻转40%→准确率0.70），");
  console.log("  两法都吃标注质量；DPO 少一层 RM 失真，但也少一层把噪声平均掉的缓冲。");

  console.log("\n→ 总览：DPO 在约 1/" + speedup.toFixed(1) +
    " 成本下匹配/超过 RLHF；β 是隐式 KL 缰绳但方向与 RLHF 相反；二者都不能凭空补偿脏标签。");
  console.log("→ 合成环境绝对值偏乐观；可迁移的是相对趋势：成本↓、KL 随 β 单调、弱参考+噪声→DPO 崩。");
}

main();
