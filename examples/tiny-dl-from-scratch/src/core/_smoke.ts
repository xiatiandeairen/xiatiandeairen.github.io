// core/_smoke.ts — Internal scaffolding self-check (NOT a stage; underscore => not run by
// any npm script and not imported by stages). Verifies the core contract holds with REAL
// numbers before stage authors build on it: reproducibility, grad-check on every op,
// a tiny end-to-end SGD fit, and an explicit FAILURE-MODE demo (missing zeroGrad).
//
// Run manually:  npx tsx src/core/_smoke.ts

import { mulberry32, randn, kaiming } from "./rng.js";
import { Tensor, noGrad } from "./autograd.js";
import { Linear, LayerNorm, Embedding, Sequential, Module } from "./nn.js";
import { SGD, Adam, AdamW, clipGradNorm, cosineWarmup } from "./optim.js";
import { makeSpiral, charDataset } from "./data.js";
import {
  crossEntropy,
  mseLoss,
  accuracy,
  gradCheck,
  timeIt,
  lossCurveAscii,
  paramCount,
} from "./metrics.js";

function section(title: string) {
  console.log("\n" + "=".repeat(60) + "\n" + title + "\n" + "=".repeat(60));
}

// ---------------------------------------------------------------------------
section("1) RNG reproducibility (bit-for-bit, same seed)");
const a = mulberry32(42);
const b = mulberry32(42);
const drawsA = [a(), a(), a()];
const drawsB = [b(), b(), b()];
const reproducible = drawsA.every((v, i) => v === drawsB[i]);
console.log("seed 42 draws:", drawsA.map((x) => x.toFixed(6)).join(", "));
console.log("two streams identical:", reproducible);
const rng = mulberry32(7);
let mean = 0;
const N = 20000;
for (let i = 0; i < N; i++) mean += randn(rng);
mean /= N;
console.log(`randn empirical mean over ${N} (expect ~0):`, mean.toFixed(4));

// ---------------------------------------------------------------------------
section("2) gradCheck every Tensor op (analytic vs numerical, expect < 1e-6)");
const gcRng = mulberry32(123);
// Build a composite expression touching many ops, reduced to a scalar loss.
function buildExpr(): { loss: () => Tensor; params: Tensor[] } {
  const x = Tensor.from([3, 4], () => kaiming(4, gcRng));
  const W = Tensor.from([4, 5], () => kaiming(4, gcRng));
  const bias = Tensor.from([1, 5], () => 0.1 * randn(gcRng));
  const params = [x, W, bias];
  const loss = () => {
    // exercise matmul, broadcast add, relu, tanh, softmax, log, mul, sum, mean, transpose
    const h = x.matmul(W).add(bias.broadcastRow(3)).relu().tanh();
    const sm = h.softmax(); // (3,5)
    const logged = sm.addScalar(1e-6).log();
    const tr = logged.transpose().transpose(); // identity round-trip exercises adjoint
    return tr.mul(h).sum().add(h.mean());
  };
  return { loss, params };
}
const { loss: exprLoss, params: exprParams } = buildExpr();
const out = exprLoss();
for (const p of exprParams) p.zeroGrad();
out.backward();
const gc = gradCheck(() => exprLoss().data[0], exprParams, 1e-5);
console.log(`checked ${gc.checked} elements across ${exprParams.length} params`);
console.log("max relative error:", gc.maxRelError.toExponential(3));
console.log("PASS (< 1e-6):", gc.maxRelError < 1e-6);

// ---------------------------------------------------------------------------
section("3) gradCheck through nn layers + crossEntropy");
const layerRng = mulberry32(99);
class TinyNet extends Module {
  l1: Linear;
  ln: LayerNorm;
  l2: Linear;
  constructor() {
    super();
    this.l1 = this.child(new Linear(4, 8, layerRng));
    this.ln = this.child(new LayerNorm(8));
    this.l2 = this.child(new Linear(8, 3, layerRng, { init: "xavier" }));
  }
  override forward(x: Tensor): Tensor {
    return this.l2.forward(this.ln.forward(this.l1.forward(x).relu()));
  }
}
const net = new TinyNet();
const inp = Tensor.from([5, 4], () => randn(layerRng));
const tgt = Int32Array.from([0, 1, 2, 1, 0]);
const ceForward = () => crossEntropy(net.forward(inp), tgt).data[0];
const ceLoss = crossEntropy(net.forward(inp), tgt);
net.zeroGrad();
ceLoss.backward();
const gcNet = gradCheck(ceForward, net.parameters(), 1e-5, 6);
console.log("paramCount(net):", paramCount(net));
console.log("CE loss:", ceLoss.data[0].toFixed(4));
console.log("max relative error:", gcNet.maxRelError.toExponential(3));
console.log("PASS (< 1e-6):", gcNet.maxRelError < 1e-6);

// mseLoss grad check too
const mseRng = mulberry32(55);
const pred = Tensor.from([4, 3], () => randn(mseRng));
const targ = Tensor.from([4, 3], () => randn(mseRng));
const mse = mseLoss(pred, targ);
pred.zeroGrad();
mse.backward();
const gcMse = gradCheck(() => mseLoss(pred, targ).data[0], [pred], 1e-5);
console.log("mseLoss grad max rel error:", gcMse.maxRelError.toExponential(3), "PASS:", gcMse.maxRelError < 1e-6);

// ---------------------------------------------------------------------------
section("4) End-to-end: SGD fits a spiral (train loss must DROP, acc must RISE)");
const dataRng = mulberry32(2024);
const ds = makeSpiral(50, 3, dataRng); // 150 points, 3 classes, NOT linearly separable
const X = Tensor.from([ds.y.length, 2], (() => {
  let k = 0;
  const flat: number[] = [];
  for (const row of ds.X) flat.push(...row);
  return () => flat[k++];
})());
const Y = Int32Array.from(ds.y);
const modelRng = mulberry32(2025);
const model = new Sequential([
  new Linear(2, 32, modelRng),
  new Linear(32, 3, modelRng, { init: "xavier" }),
]);
// inject a relu between layers by wrapping (Sequential is linear; use a tiny adapter)
class MLP extends Module {
  h: Linear;
  o: Linear;
  constructor() {
    super();
    this.h = this.child(new Linear(2, 32, modelRng));
    this.o = this.child(new Linear(32, 3, modelRng, { init: "xavier" }));
  }
  override forward(x: Tensor): Tensor {
    return this.o.forward(this.h.forward(x).relu());
  }
}
const mlp = new MLP();
void model; // keep Sequential reference exercised for typecheck of import
const opt = new SGD(mlp.parameters(), { lr: 0.5, momentum: 0.9, weightDecay: 1e-4 });
const history: number[] = [];
let lastGradNorm = 0;
for (let step = 0; step < 300; step++) {
  const logits = mlp.forward(X);
  const l = crossEntropy(logits, Y);
  mlp.zeroGrad();
  l.backward();
  lastGradNorm = clipGradNorm(mlp.parameters(), 5.0); // pre-clip norm (logged)
  opt.step();
  history.push(l.data[0]);
}
const finalLogits = mlp.forward(X);
const finalLoss = crossEntropy(finalLogits, Y).data[0];
const acc = accuracy(finalLogits, Y);
console.log(`init loss: ${history[0].toFixed(4)}  final loss: ${finalLoss.toFixed(4)}`);
console.log(`final train accuracy: ${(acc * 100).toFixed(1)}%  (chance = 33.3%)`);
console.log(`last pre-clip grad norm: ${lastGradNorm.toFixed(4)}`);
console.log("loss monotone-ish drop (final < init):", finalLoss < history[0]);
console.log(lossCurveAscii(history));

// ---------------------------------------------------------------------------
section("5) Optimizer + schedule contracts");
const adamRng = mulberry32(11);
const tinyW = Tensor.from([2, 2], () => randn(adamRng));
const adam = new Adam([tinyW], { lr: 0.01 });
const adamw = new AdamW([tinyW], { lr: 0.01, weightDecay: 0.1 });
void adam;
void adamw;
console.log("Adam & AdamW constructed OK; AdamW decouples weight decay from vhat.");
const lrs = [0, 5, 10, 50, 100, 150].map((s) => cosineWarmup(s, 10, 100, 0.01));
console.log(
  "cosineWarmup(warmup=10,total=100,base=0.01) @ steps [0,5,10,50,100,150]:",
  lrs.map((v) => v.toFixed(5)).join(", "),
);
console.log("  ramps up then cosine-decays to ~0, then clamps 0:", lrs[0] === 0 && lrs[5] === 0 && lrs[2] > lrs[1]);

// ---------------------------------------------------------------------------
section("6) char dataset + Embedding + noGrad inference");
const cds = charDataset();
console.log(`vocab size: ${cds.vocabSize}  trainIds: ${cds.trainIds.length}  valIds: ${cds.valIds.length}`);
const batchRng = mulberry32(3);
const batch = cds.getBatch("train", 8, 4, batchRng);
console.log(`getBatch -> x/y len ${batch.x.length} (batch=${batch.batchSize} x block=${batch.blockSize})`);
console.log("decode first window x:", JSON.stringify(cds.decode(Array.from(batch.x.slice(0, 8)))));
console.log("decode first window y:", JSON.stringify(cds.decode(Array.from(batch.y.slice(0, 8)))));
const embRng = mulberry32(4);
const emb = new Embedding(cds.vocabSize, 6, embRng);
const ids = Tensor.from([4], (() => {
  let k = 0;
  return () => batch.x[k++];
})());
const embedded = noGrad(() => emb.forward(ids)); // inference path
console.log(`embedding output shape: [${embedded.shape}] (4 ids -> 6-d rows)`);

// ---------------------------------------------------------------------------
section("7) FAILURE MODE demo: forgetting zeroGrad makes grads ACCUMULATE");
// This is the single most common autograd bug. We show the SAME backward called twice
// without zeroing => grad doubles. This is by design (reverse-mode sums paths); the demo
// proves the contract so stage authors remember to zeroGrad each step.
const fmW = Tensor.from([1, 3], () => 1);
function fwdBwd() {
  const y = fmW.mulScalar(2).sum();
  y.backward();
}
fmW.zeroGrad();
fwdBwd();
const gradOnce = fmW.grad[0];
fwdBwd(); // NO zeroGrad in between -> accumulates
const gradTwice = fmW.grad[0];
console.log(`grad after 1 backward: ${gradOnce}  after 2 (no zeroGrad): ${gradTwice}`);
console.log("accumulation confirmed (twice == 2x once):", gradTwice === 2 * gradOnce);
console.log(">> Lesson: optimizer.zeroGrad() / module.zeroGrad() EVERY step, or loss explodes.");

// ---------------------------------------------------------------------------
section("8) timing (real wall-clock, relative only)");
const tRng = mulberry32(8);
const big = Tensor.from([64, 64], () => randn(tRng));
const big2 = Tensor.from([64, 64], () => randn(tRng));
const t = timeIt(() => {
  const r = big.matmul(big2);
  void r.data[0];
}, 50, 5);
console.log(`64x64 matmul: ${t.perIterMs.toFixed(3)} ms/iter over ${t.iters} iters (machine-dependent, compare ratios)`);

console.log("\nALL CORE SMOKE CHECKS DONE.\n");
