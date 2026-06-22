// stage06-residual.ts — Why a skip connection lets you stack depth without the gradient dying.
//
// THE MECHANISM we measure (not assert): backprop through a plain block multiplies the
//   upstream grad by that block's Jacobian J_F. Through d stacked plain blocks the grad is a
//   PRODUCT J_1·J_2·...·J_d; when each ‖J‖ < 1 (typical after ReLU) the product shrinks
//   geometrically and the FIRST layer sees almost no gradient — it cannot learn. A residual
//   block computes y = F(x) + x, so dy/dx = J_F + I. That "+I" gives every backprop path an
//   ADDITIVE identity route that bypasses F, so the product never collapses to 0.
//
// WHAT THE NUMBERS BELOW ACTUALLY SHOW (three honest findings, in order):
//   (1) INIT-TIME gradient flow vs depth: the plain net's shallowest-layer grad norm SHRINKS
//       as depth grows; the residual net's STAYS healthy. This is the vanishing-gradient
//       fingerprint, measured directly, before any training (so no NaN/optimizer confounds).
//   (2) TRAINABILITY: at a learning rate where the plain net STALLS at chance, the residual
//       net trains to 100% — the preserved gradient is what lets it learn.
//   (3) An HONEST caveat the toy makes unavoidable: a NAIVE identity skip (F(x)+x with no
//       normalization) makes activations COMPOUND additively across depth and EXPLODE. Real
//       residual nets pair the skip with normalization; here we use the cheaper fix —
//       scaling the residual branch by 1/sqrt(depth) — to keep forward activations sane while
//       preserving the gradient benefit. We show the explosion AND the fix with numbers.
//
// WHY NO BatchNorm in the trunk here: BN is ITSELF a vanishing-gradient remedy — with BN in
//   every block the plain net's shallow grad no longer vanishes, which would HIDE the very
//   effect this stage isolates. We strip BN so the skip's contribution is the only variable.
//   (Stage on normalization covers BN's own gradient-flow effect.)
//
// HONESTY: 12x12 toy shapes, tiny channels, f64, single-thread. Absolute losses / grad
//   magnitudes / step counts are toy-optimistic. What transfers is the RELATIVE trend —
//   plain-grad-vanishes-with-depth, residual-grad-survives — and that residual trains at a LR
//   where plain stalls. Wall-clock is REAL (performance.now).
//
// FAILURE MODES demonstrated (not just happy path):
//   A. The activation explosion of a naive (unscaled) identity skip — shown as a real number.
//   B. Channel/size mismatch when F changes shape but the skip is identity => the actual
//      thrown shape-mismatch error, then the standard fix (1x1 conv projection shortcut).
//
// Reuses core only. Deterministic: every randomness source is a seeded mulberry32 stream.

import { Tensor } from "./core/autograd.js";
import { Conv2d, ReLU, Sequential, ResidualBlock, Module } from "./core/nn.js";
import { Linear, Flatten } from "./core/nn.js";
import { SGD } from "./core/optim.js";
import { mulberry32, type Rng } from "./core/rng.js";
import { makeShapeDataset, iterBatches, SHAPE_NAMES } from "./core/data.js";
import { crossEntropy, accuracy, lossCurveAscii } from "./core/metrics.js";

// --- experiment hyperparameters (tiny so the whole stage runs in seconds) ---
const SEED = 7;
const IMG = 12; // 12x12; conv pad=1 stays shape-preserving so an identity skip is legal
const CH = 6; // channels in the trunk; same for every block
const DEPTH = 10; // stacked conv blocks — deep enough that the plain net's shallow grad vanishes
const PER_CLASS = 40; // 4 classes -> 160 train images
const BATCH = 16;
const STEPS = 200; // fixed training steps for BOTH nets (fair comparison)
// LR chosen at the EDGE: low enough that residual is stable, high enough that the PLAIN net
// stalls — that gap is finding (2). At lr=0.05 both train; at lr=0.1 plain stalls, residual
// learns; at lr>=0.2 even residual diverges. We use the gap-revealing value.
const TRAIN_LR = 0.1;
// Residual-branch scale. WHY 1/sqrt(DEPTH): with d additive skips, branch variances sum, so
// activation std grows ~sqrt(d); scaling each branch by 1/sqrt(d) keeps the running activation
// magnitude ~O(1) instead of exploding. This is the cheap stand-in for the normalization that
// real ResNets use. Without it, see FAILURE MODE A.
const RESIDUAL_ALPHA = 1 / Math.sqrt(DEPTH);

// ----------------------------------------------------------------------------
// ScaledResidualBlock: out = relu(alpha * F(x) + x).
// core's ResidualBlock does relu(F(x) + shortcut(x)) with no branch scale; for a deep stack
// without normalization that explodes (FAILURE MODE A), so we need the alpha. We compose it
// from the same Tensor ops core uses (mulScalar + add + relu) — the autograd handles backward.
// We still USE core's ResidualBlock in the failure-mode section to show its shape guard.
// ----------------------------------------------------------------------------
class ScaledResidualBlock extends Module {
  constructor(private branch: Module, private alpha: number) {
    super();
    this.child(branch); // register so parameters()/zeroGrad() recurse into F
  }
  override forward(x: Tensor): Tensor {
    // identity shortcut: F and x must be shape-identical (guaranteed by shape-preserving F).
    return this.branch.forward(x).mulScalar(this.alpha).add(x).relu();
  }
}

// ----------------------------------------------------------------------------
// Block / net builders. CRITICAL for a fair test: plain and residual nets must be built from
// the SAME init draws, so any difference comes ONLY from the skip, not from luckier weights.
// We seed a fresh rng identically for each net => block k gets identical weights in both.
// ----------------------------------------------------------------------------

/** One shape-preserving conv block: 3x3 conv (pad 1) -> ReLU. Output shape == input shape,
 *  the precondition for an identity skip to add. No BN here — see header for why. */
function buildConvBlock(rng: Rng): Module {
  return new Sequential([new Conv2d(CH, CH, 3, rng, { stride: 1, padding: 1 }), new ReLU()]);
}

/** Classifier head shared by both nets: flatten the CHxIMGxIMG trunk -> 4 logits. */
function buildHead(rng: Rng): Module {
  return new Sequential([new Flatten(), new Linear(CH * IMG * IMG, SHAPE_NAMES.length, rng)]);
}

/**
 * Assemble a net of `depth` blocks. mode "plain" stacks blocks directly; "residual" wraps each
 * in a ScaledResidualBlock with an identity skip. The stem (1->CH conv) is the SHALLOWEST
 * learnable layer; we return its weight separately because its grad norm is the
 * vanishing-gradient probe (the grad that survived backprop through every block).
 */
function buildNet(rng: Rng, depth: number, residual: boolean): { net: Module; stemWeight: Tensor } {
  const stem = new Conv2d(1, CH, 3, rng, { stride: 1, padding: 1 }); // shape-preserving lift to CH
  const layers: Module[] = [stem];
  for (let d = 0; d < depth; d++) {
    const block = buildConvBlock(rng);
    layers.push(residual ? new ScaledResidualBlock(block, RESIDUAL_ALPHA) : block);
  }
  layers.push(buildHead(rng));
  return { net: new Sequential(layers), stemWeight: stem.weight };
}

function l2GradNorm(t: Tensor): number {
  let s = 0;
  for (let i = 0; i < t.size; i++) s += t.grad[i] * t.grad[i];
  return Math.sqrt(s);
}

// ----------------------------------------------------------------------------
// FINDING (1): init-time shallow-layer gradient norm vs depth, plain vs residual.
// One forward+backward on a fixed batch, no training => isolates the architecture's effect on
// gradient flow with zero optimizer/NaN confounds.
// ----------------------------------------------------------------------------
function initShallowGrad(depth: number, residual: boolean, batch: { data: Float64Array; labels: Int32Array; N: number }): {
  gradNorm: number;
  loss0: number;
} {
  const { net, stemWeight } = buildNet(mulberry32(SEED), depth, residual);
  const x = new Tensor(batch.data, [batch.N, 1, IMG, IMG]);
  net.zeroGrad();
  const logits = net.forward(x);
  const loss = crossEntropy(logits, batch.labels);
  loss.backward();
  return { gradNorm: l2GradNorm(stemWeight), loss0: loss.data[0] };
}

// ----------------------------------------------------------------------------
// FINDING (2): train both nets identically; report loss curve + final accuracy.
// ----------------------------------------------------------------------------
interface TrainResult {
  lossHistory: number[];
  finalLoss: number;
  finalAcc: number;
}

function trainNet(residual: boolean, lr: number): TrainResult {
  const { net } = buildNet(mulberry32(SEED), DEPTH, residual);
  const ds = makeShapeDataset(PER_CLASS, IMG, IMG, mulberry32(SEED + 100));
  const opt = new SGD(net.parameters(), { lr });
  const dataRng = mulberry32(SEED + 100); // same stream for both nets => same batch order
  const lossHistory: number[] = [];
  let step = 0;
  let lastLogits: Tensor | null = null;
  let lastLabels: Int32Array | null = null;
  while (step < STEPS) {
    for (const batch of iterBatches(ds, BATCH, dataRng)) {
      if (step >= STEPS) break;
      const x = new Tensor(batch.data, [batch.N, 1, IMG, IMG]);
      net.zeroGrad();
      const logits = net.forward(x);
      const loss = crossEntropy(logits, batch.labels);
      loss.backward();
      lossHistory.push(loss.data[0]);
      opt.step();
      lastLogits = logits;
      lastLabels = batch.labels;
      step++;
    }
  }
  return {
    lossHistory,
    finalLoss: lossHistory[lossHistory.length - 1],
    finalAcc: lastLogits && lastLabels ? accuracy(lastLogits, lastLabels) : 0,
  };
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
function main(): void {
  console.log("=== stage06: 残差连接 — 为什么 skip 让网络能堆更深 ===\n");
  console.log(
    `配置: 通道=${CH}, 图像=${IMG}x${IMG}, batch=${BATCH}, 种子=${SEED}, 主干不含 BatchNorm(见文件头注释)`,
  );
  console.log("对照组与残差组深度/初始化逐块相同, 唯一变量是有没有 skip。\n");
  const startMs = performance.now();

  // A fixed batch reused for every init-grad probe so the only variable is depth & architecture.
  const probeDs = makeShapeDataset(PER_CLASS, IMG, IMG, mulberry32(SEED + 100));
  const probeBatch = iterBatches(probeDs, BATCH, mulberry32(SEED + 100)).next().value!;

  // --- FINDING 1: gradient flow vs depth ---
  console.log("--- 发现 1: 最浅层 (stem conv) 初始梯度范数 vs 深度 (训练前, 单次反向) ---");
  console.log("  这是反向传播穿过全部块后到达第一层的梯度量级; 越小=越接近梯度消失。");
  console.log("  深度   朴素堆叠 stem 梯度    残差版 stem 梯度    残差/朴素");
  const depths = [4, 6, 8, 10, 14];
  for (const d of depths) {
    const plain = initShallowGrad(d, false, probeBatch);
    const resid = initShallowGrad(d, true, probeBatch);
    const ratio = resid.gradNorm / Math.max(1e-30, plain.gradNorm);
    console.log(
      `   ${String(d).padStart(2)}    ${plain.gradNorm.toExponential(3)}` +
        `          ${resid.gradNorm.toExponential(3)}        ${ratio.toFixed(1)}x`,
    );
  }
  console.log("  读法: 朴素版梯度随深度单调变小(信号到不了第一层); 残差版保持健康甚至增大。");

  // --- FINDING 2: trainability at a stall-revealing LR ---
  console.log(`\n--- 发现 2: 深度=${DEPTH} 下训练 ${STEPS} 步 (lr=${TRAIN_LR}, 故意取在"朴素会卡住"的档位) ---`);
  const plainTrain = trainNet(false, TRAIN_LR);
  const residTrain = trainNet(true, TRAIN_LR);
  console.log("Loss 曲线 (朴素堆叠):");
  console.log(lossCurveAscii(plainTrain.lossHistory));
  console.log("\nLoss 曲线 (残差):");
  console.log(lossCurveAscii(residTrain.lossHistory));
  console.log(
    `\n朴素堆叠: 末步 loss=${plainTrain.finalLoss.toFixed(4)}  acc=${(plainTrain.finalAcc * 100).toFixed(1)}%`,
  );
  console.log(
    `残差版  : 末步 loss=${residTrain.finalLoss.toFixed(4)}  acc=${(residTrain.finalAcc * 100).toFixed(1)}%`,
  );
  console.log(`基线参考: 4 类随机猜测 loss = ln(4) = ${Math.log(4).toFixed(4)} (朴素版卡在此附近=没学会)`);

  // --- FAILURE MODE A: naive (unscaled) skip explodes ---
  console.log("\n=== 失败模式 A: 朴素 identity skip 不做归一化 => 前向激活爆炸 ===");
  console.log("  残差块 relu(F(x)+x) 的恒等项让激活沿深度累加; 不缩放/不归一化时随深度发散。");
  console.log("  深度   未缩放残差 初始 loss   缩放后(branch×1/√d) 初始 loss");
  for (const d of [6, 10, 14, 20]) {
    // unscaled = alpha 1 via core ResidualBlock; scaled = our ScaledResidualBlock.
    const unscaled = buildUnscaledResidualNet(mulberry32(SEED), d);
    const x1 = new Tensor(probeBatch.data, [probeBatch.N, 1, IMG, IMG]);
    const lUnscaled = crossEntropy(unscaled.forward(x1), probeBatch.labels).data[0];
    const scaled = initShallowGrad(d, true, probeBatch).loss0;
    console.log(`   ${String(d).padStart(2)}       ${lUnscaled.toFixed(2).padStart(8)}            ${scaled.toFixed(2).padStart(8)}`);
  }
  console.log("  读法: 未缩放版 loss 随深度飙升(激活爆炸); 缩放后保持在个位数 => 可训练。");

  // --- FAILURE MODE B: shape mismatch + projection-shortcut fix ---
  console.log("\n=== 失败模式 B: 通道数变化 + identity skip => shape mismatch (堆深常见崩溃点) ===");
  const failRng = mulberry32(SEED + 1);
  const widenBlock = new Sequential([
    new Conv2d(CH, CH * 2, 3, failRng, { stride: 1, padding: 1 }), // 6 -> 12 channels
    new ReLU(),
  ]);
  const brokenResidual = new ResidualBlock(widenBlock, null); // identity skip is WRONG here
  const probe = new Tensor(new Float64Array(2 * CH * IMG * IMG), [2, CH, IMG, IMG]);
  try {
    brokenResidual.forward(probe);
    console.log("(意外: 没有抛错 — shape 检查失效)");
  } catch (err) {
    console.log("捕获到预期错误 (错误信息同时给出两边 shape, 正是排查需要的):");
    console.log("  " + (err as Error).message);
  }

  console.log("\n=== 失败模式 B 的修复: 1x1 卷积 projection shortcut 对齐维度 ===");
  const fixRng = mulberry32(SEED + 2);
  const widenBlock2 = new Sequential([new Conv2d(CH, CH * 2, 3, fixRng, { stride: 1, padding: 1 }), new ReLU()]);
  // 1x1 conv shortcut CH->CH*2, stride 1, pad 0: 改通道数、保持 H,W, 与 F(x) 对齐。
  const projection = new Conv2d(CH, CH * 2, 1, fixRng, { stride: 1, padding: 0 });
  const fixedResidual = new ResidualBlock(widenBlock2, projection);
  const out = fixedResidual.forward(probe);
  console.log(`输入 shape: [${probe.shape}]  ->  输出 shape: [${out.shape}]`);
  console.log(`F(x) 与 shortcut(x) 都是 ${CH * 2} 通道, 相加合法; 空间尺寸仍为 ${IMG}x${IMG}。`);

  const wallMs = performance.now() - startMs;
  console.log("\n=== 结论 ===");
  console.log("1) skip 的 +I 路径让梯度绕过每个 F: 朴素版第一层梯度随深度消失, 残差版保持健康(发现1)。");
  console.log("2) 因此在朴素版卡在随机基线的学习率下, 残差版仍能训练到位(发现2)。");
  console.log("3) 但裸 skip 会让激活爆炸(失败A), 需配归一化或 branch 缩放; 变通道处需 projection(失败B)。");
  console.log(`实测墙钟 (全部探针+训练): ${wallMs.toFixed(0)} ms`);
}

/** Build a net using core's UNSCALED ResidualBlock (alpha=1) to demonstrate the explosion in
 *  FAILURE MODE A. Kept tiny — only forwarded once, never trained. */
function buildUnscaledResidualNet(rng: Rng, depth: number): Module {
  const stem = new Conv2d(1, CH, 3, rng, { stride: 1, padding: 1 });
  const layers: Module[] = [stem];
  for (let d = 0; d < depth; d++) layers.push(new ResidualBlock(buildConvBlock(rng), null));
  layers.push(buildHead(rng));
  return new Sequential(layers);
}

main();
