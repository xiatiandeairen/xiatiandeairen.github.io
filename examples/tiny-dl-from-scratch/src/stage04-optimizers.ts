// stage04-optimizers.ts — Chapter 4: optimizers, who actually moves the parameters.
//
// WHAT THIS STAGE PROVES (all numbers below are computed at runtime, not quoted):
//   1. Same net, same seed, same data, four optimizers (SGD / SGD+momentum / Adam / AdamW):
//      plot each loss curve and count steps-to-target. The fair-comparison invariant is that
//      every run starts from BIT-IDENTICAL weights — otherwise you are comparing init luck,
//      not optimizers. We enforce it by rebuilding the net from the same seed each run.
//   2. Adam bias-correction on vs off: print the EFFECTIVE step length of the first 10 steps.
//      Without correction, m/v start at 0 and the sqrt(v) denominator is tiny, so early
//      updates are BLOWN UP — a real, observable instability, not folklore.
//   3. FAILURE MODE: push lr past the divergence threshold and report the exact step where
//      the loss becomes NaN. Happy-path-only demos hide this; here we make it happen.
//   4. Decoupled (AdamW) vs coupled (Adam) weight decay: same wd value, print the final L2
//      weight norm. Coupled decay is scaled by the adaptive denominator, decoupled is not,
//      so the resulting norms differ — that difference is the whole reason AdamW exists.
//
// HONESTY NOTE: this is toy 2-D spiral data on a tiny MLP, CPU, deterministic seeds. Absolute
//   step counts and absolute losses are OPTIMISTIC and machine/data specific. What transfers
//   is the RELATIVE story: ordering of optimizers, the shape of the bias-correction blowup,
//   the existence of a divergence cliff, and the SIGN of the AdamW-vs-Adam norm gap.

import { mulberry32, type Rng } from "./core/rng.js";
import { Tensor } from "./core/autograd.js";
import { Linear, Module } from "./core/nn.js";
import { SGD, Adam, AdamW, clipGradNorm, type Optimizer } from "./core/optim.js";
import { makeSpiral } from "./core/data.js";
import { crossEntropy, mseLoss, accuracy } from "./core/metrics.js";

function section(title: string): void {
  console.log("\n" + "=".repeat(64) + "\n" + title + "\n" + "=".repeat(64));
}

// Two-hidden-layer MLP. Kept identical across every optimizer run by always constructing it
// from the SAME seed (see freshNet) — this is the load-bearing fairness invariant.
class SpiralMLP extends Module {
  private h1: Linear;
  private h2: Linear;
  private out: Linear;
  constructor(rng: Rng) {
    super();
    this.h1 = this.child(new Linear(2, 32, rng));
    this.h2 = this.child(new Linear(32, 32, rng));
    this.out = this.child(new Linear(32, 3, rng, { init: "xavier" }));
  }
  override forward(x: Tensor): Tensor {
    return this.out.forward(this.h2.forward(this.h1.forward(x).relu()).relu());
  }
}

const NET_SEED = 4040; // fixed so every optimizer sees identical initial weights
const DATA_SEED = 2024;

/** Rebuild the net from a frozen seed. Calling this before each run guarantees bit-identical
 *  initial parameters, so any difference in the loss curve is attributable to the optimizer,
 *  not to a different random init. */
function freshNet(): SpiralMLP {
  return new SpiralMLP(mulberry32(NET_SEED));
}

// Spiral: 3 interleaved arms, NOT linearly separable, so a plain linear model can't cheat.
const ds = makeSpiral(60, 3, mulberry32(DATA_SEED)); // 180 points
const X = (() => {
  const flat: number[] = [];
  for (const row of ds.X) flat.push(...row);
  let k = 0;
  return Tensor.from([ds.y.length, 2], () => flat[k++]);
})();
const Y = Int32Array.from(ds.y);
// One-hot regression targets, used ONLY by the NaN demo (section 3). We need an UNBOUNDED
// loss there: crossEntropy's stable softmax caps the loss (~18.4 max) so it never overflows
// even at absurd lr — masking divergence. mseLoss on raw logits is unbounded, so an lr past
// the cliff makes weights -> logits -> squared error blow up to Inf -> NaN, the real failure.
const T = (() => {
  const arr: number[] = [];
  for (const y of ds.y) arr.push(y === 0 ? 1 : 0, y === 1 ? 1 : 0, y === 2 ? 1 : 0);
  let k = 0;
  return Tensor.from([ds.y.length, 3], () => arr[k++]);
})();

/** Full-batch train for `steps`, recording loss each step. clipGradNorm keeps the SGD runs
 *  comparable to Adam by capping pathological grad spikes (returns pre-clip norm, unused here).
 *  Returns the loss history; NaN entries are propagated (the divergence demo relies on this). */
function train(opt: Optimizer, net: SpiralMLP, steps: number): number[] {
  const history: number[] = [];
  for (let step = 0; step < steps; step++) {
    const logits = net.forward(X);
    const loss = crossEntropy(logits, Y);
    net.zeroGrad(); // INVARIANT: clear grads every step; autograd accumulates on purpose.
    loss.backward();
    clipGradNorm(net.parameters(), 5.0);
    opt.step();
    history.push(loss.data[0]);
  }
  return history;
}

/** First step index whose loss <= target, or -1 if never reached (incl. NaN runs). */
function stepsToTarget(history: number[], target: number): number {
  for (let i = 0; i < history.length; i++) if (history[i] <= target) return i + 1;
  return -1;
}

/** Single-line sparkline so four curves fit on screen for eyeball comparison. NaN -> '!'. */
function sparkline(history: number[], cols = 50): string {
  const finite = history.filter((v) => Number.isFinite(v));
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const ramp = "▁▂▃▄▅▆▇█";
  const span = hi - lo || 1;
  const out: string[] = [];
  for (let c = 0; c < cols; c++) {
    const v = history[Math.min(history.length - 1, Math.floor((c / cols) * history.length))];
    if (!Number.isFinite(v)) {
      out.push("!");
      continue;
    }
    const t = (v - lo) / span;
    out.push(ramp[Math.min(ramp.length - 1, Math.floor(t * (ramp.length - 1)))]);
  }
  return out.join("");
}

function l2Norm(net: Module): number {
  let sq = 0;
  for (const p of net.parameters()) for (let i = 0; i < p.size; i++) sq += p.data[i] * p.data[i];
  return Math.sqrt(sq);
}

// ===========================================================================
section("1) 同网络 / 同种子 / 同数据，四个优化器横向对比");
const STEPS = 300;
const TARGET = 0.35; // a loss every optimizer can plausibly reach, so steps-to-target differs

interface RunResult {
  name: string;
  history: number[];
  finalLoss: number;
  acc: number;
  steps: number;
}

// lr per optimizer is intentionally tuned to each one's sane regime (SGD wants a bigger lr
// than Adam). Comparing optimizers each at a BAD lr would be a strawman; we give each a fair lr.
function makeRuns(): RunResult[] {
  const specs: Array<{ name: string; build: (net: SpiralMLP) => Optimizer }> = [
    { name: "SGD (no momentum)", build: (n) => new SGD(n.parameters(), { lr: 0.5 }) },
    { name: "SGD + momentum0.9", build: (n) => new SGD(n.parameters(), { lr: 0.5, momentum: 0.9 }) },
    { name: "Adam            ", build: (n) => new Adam(n.parameters(), { lr: 0.02 }) },
    { name: "AdamW (wd=0.01) ", build: (n) => new AdamW(n.parameters(), { lr: 0.02, weightDecay: 0.01 }) },
  ];
  return specs.map(({ name, build }) => {
    const net = freshNet(); // bit-identical init for every optimizer
    const history = train(build(net), net, STEPS);
    const logits = net.forward(X);
    return {
      name,
      history,
      finalLoss: crossEntropy(logits, Y).data[0],
      acc: accuracy(logits, Y),
      steps: stepsToTarget(history, TARGET),
    };
  });
}

const runs = makeRuns();
// sanity: all runs really did start from the same init loss (proves the fairness invariant).
const initLosses = runs.map((r) => r.history[0].toFixed(4));
console.log("init loss per run (must be identical):", initLosses.join("  "));
console.log("identical init confirmed:", new Set(initLosses).size === 1);
console.log("");
console.log(`loss curves (left=step0  right=step${STEPS - 1}, lower bar = lower loss):`);
for (const r of runs) console.log(`  ${r.name}  ${sparkline(r.history)}`);
console.log("");
console.log("optimizer          | init   | final  | train acc | steps to loss<=" + TARGET);
console.log("-------------------+--------+--------+-----------+--------------------");
for (const r of runs) {
  const reached = r.steps === -1 ? `>${STEPS} (not reached)` : `${r.steps}`;
  console.log(
    `${r.name} | ${r.history[0].toFixed(4)} | ${r.finalLoss.toFixed(4)} | ${(r.acc * 100)
      .toFixed(1)
      .padStart(8)}% | ${reached}`,
  );
}
console.log("\n读法: 绝对步数是 toy 数据下的乐观值; 可迁移的是相对趋势(谁更快收敛、动量/Adam 是否更稳).");

// ===========================================================================
section("2) Adam bias-correction 开 / 关：前 10 步的有效步长");
// We measure the EFFECTIVE step length = || param_after - param_before ||_2 of the FIRST param
// tensor, step by step, on identical grads. The core Adam always corrects, so to show "off"
// we run a minimal hand-rolled Adam update on the SAME grads with correction toggled.
// Both numbers are really computed; nothing is quoted.
function biasCorrectionProbe(corrected: boolean): number[] {
  const net = freshNet();
  const params = net.parameters();
  const m = params.map((p) => new Float64Array(p.size));
  const v = params.map((p) => new Float64Array(p.size));
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const lr = 0.02;
  const stepLens: number[] = [];
  for (let t = 1; t <= 10; t++) {
    const logits = net.forward(X);
    const loss = crossEntropy(logits, Y);
    net.zeroGrad();
    loss.backward();
    // bias-correction denominators; when off, we keep them at 1 (i.e. use raw biased moments).
    const bc1 = corrected ? 1 - Math.pow(beta1, t) : 1;
    const bc2 = corrected ? 1 - Math.pow(beta2, t) : 1;
    let moved = 0;
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      for (let j = 0; j < p.size; j++) {
        const g = p.grad[j];
        m[i][j] = beta1 * m[i][j] + (1 - beta1) * g;
        v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g;
        const mhat = m[i][j] / bc1;
        const vhat = v[i][j] / bc2;
        const upd = (lr * mhat) / (Math.sqrt(vhat) + eps);
        if (i === 0) moved += upd * upd; // measure step length on the first param tensor only
        p.data[j] -= upd;
      }
    }
    stepLens.push(Math.sqrt(moved));
  }
  return stepLens;
}
const withBC = biasCorrectionProbe(true);
const noBC = biasCorrectionProbe(false);
console.log("step |  with bias-corr | WITHOUT bias-corr | ratio (no/with)");
console.log("-----+-----------------+-------------------+----------------");
for (let t = 0; t < 10; t++) {
  const ratio = withBC[t] === 0 ? Infinity : noBC[t] / withBC[t];
  console.log(
    `${String(t + 1).padStart(4)} | ${withBC[t].toExponential(3).padStart(15)} | ${noBC[t]
      .toExponential(3)
      .padStart(17)} | ${ratio.toFixed(2)}x`,
  );
}
console.log(
  "\n结论: 不校正时, 第1步 sqrt(vhat) 用的是被 0 拉低的 v, 分母极小 => 有效步长被放大 ~" +
    (noBC[0] / withBC[0]).toFixed(0) +
    "x; 校正项 (1-beta^t) 把早期 moment 还原回真实尺度, 几步后两者趋同.",
);

// ===========================================================================
section("3) 失败模式: 学习率越过发散阈值, loss 在第几步变 NaN");
// Sweep lr upward until a run produces NaN; report the exact step where it first appears.
function firstNaNStep(history: number[]): number {
  for (let i = 0; i < history.length; i++) if (!Number.isFinite(history[i])) return i + 1;
  return -1;
}
const lrSweep = [0.5, 2.0, 5.0, 20.0, 50.0];
const NAN_STEPS = 200;
console.log("loss = mseLoss (UNBOUNDED) on raw logits, plain SGD, no momentum, grad clip OFF.");
console.log("(crossEntropy here would NOT show this: its stable softmax caps loss ~18.4, so it");
console.log(" saturates instead of overflowing even at lr=1e12 — the loss function masks the cliff.)");
console.log("lr     | first NaN step | final loss      | verdict");
console.log("-------+----------------+-----------------+---------------------------");
for (const lr of lrSweep) {
  const net = freshNet();
  const opt = new SGD(net.parameters(), { lr });
  // NOTE: clip intentionally disabled here so the raw divergence is visible; in real training
  // clipGradNorm is exactly the guard that pushes this cliff to a higher lr.
  const history: number[] = [];
  for (let step = 0; step < NAN_STEPS; step++) {
    const pred = net.forward(X);
    const loss = mseLoss(pred, T);
    net.zeroGrad();
    loss.backward();
    opt.step(); // no clip
    history.push(loss.data[0]);
    if (!Number.isFinite(loss.data[0])) break; // once Inf/NaN, every later step stays bad
  }
  const nanStep = firstNaNStep(history);
  const last = history[history.length - 1];
  // A finite-but-astronomically-large final loss (>> init ~0.7) is already diverging; it just
  // hasn't overflowed within the step budget. Don't mislabel it "stable".
  const verdict =
    nanStep !== -1
      ? "DIVERGED -> Inf/NaN"
      : last > 1e3
        ? "diverging (not yet Inf)"
        : "converged/stable";
  const lastStr = Number.isFinite(last) ? last.toExponential(3) : String(last);
  console.log(
    `${String(lr).padStart(6)} | ${(nanStep === -1 ? "—" : String(nanStep)).padStart(14)} | ${lastStr.padStart(
      15,
    )} | ${verdict}`,
  );
}
console.log(
  "\n读法: 存在一个发散悬崖. lr 越过它, 每步 overshoot 让权重指数增长 -> logits -> 平方误差溢出到 Inf -> 再一步算梯度变 NaN, 不可恢复. lr 越大触发越早(50 比 20 更早炸). 这正是 grad clipping / warmup / lr 调度要防的失败.",
);

// ===========================================================================
section("4) 解耦 vs 耦合 weight decay: 最终权重 L2 范数差异");
// Same wd, same lr, same seed/data, same steps. Only difference: coupled (Adam, wd folded into
// grad before the adaptive denom) vs decoupled (AdamW, wd applied straight to params).
const WD = 0.05;
const WD_STEPS = 300;

const netNoWd = freshNet();
train(new Adam(netNoWd.parameters(), { lr: 0.02, weightDecay: 0 }), netNoWd, WD_STEPS);

const netCoupled = freshNet();
train(new Adam(netCoupled.parameters(), { lr: 0.02, weightDecay: WD }), netCoupled, WD_STEPS);

const netDecoupled = freshNet();
train(new AdamW(netDecoupled.parameters(), { lr: 0.02, weightDecay: WD }), netDecoupled, WD_STEPS);

const initNorm = l2Norm(freshNet());
const loss = (n: SpiralMLP) => crossEntropy(n.forward(X), Y).data[0];
console.log(`init weight L2 norm (before training): ${initNorm.toFixed(4)}`);
console.log("");
console.log("config                    | final ‖W‖₂ | final loss | note");
console.log("--------------------------+-----------+-----------+-----------------------------");
console.log(
  `Adam,  wd=0    (no decay)  | ${l2Norm(netNoWd).toFixed(4).padStart(9)} | ${loss(netNoWd)
    .toFixed(4)
    .padStart(9)} | baseline, no shrink`,
);
console.log(
  `Adam,  wd=${WD} (coupled)   | ${l2Norm(netCoupled).toFixed(4).padStart(9)} | ${loss(netCoupled)
    .toFixed(4)
    .padStart(9)} | decay scaled by sqrt(vhat)`,
);
console.log(
  `AdamW, wd=${WD} (decoupled) | ${l2Norm(netDecoupled).toFixed(4).padStart(9)} | ${loss(netDecoupled)
    .toFixed(4)
    .padStart(9)} | decay applied directly`,
);
const gap = l2Norm(netCoupled) - l2Norm(netDecoupled);
console.log(
  `\n耦合与解耦的范数差: ${gap >= 0 ? "+" : ""}${gap.toFixed(4)} ` +
    "(符号/量级是真实算出的). 关键: 耦合把 wd 折进梯度, 再被自适应分母 sqrt(vhat) 缩放, 大梯度参数被衰减得更少; " +
    "AdamW 直接对参数衰减, 与 vhat 无关 -> 这就是 transformer 训练偏好 AdamW 的原因.",
);

console.log("\n" + "=".repeat(64));
console.log("stage04 done. 所有数字均为本次运行实算/实测; toy 数据绝对值偏乐观, 可迁移的是相对趋势.");
console.log("=".repeat(64));
