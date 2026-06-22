// core/_smoke.ts — Scaffold validation. NOT a stage; never imported by stages.
//
// Verifies the core contract holds before any chapter is written:
//   1. PRNG determinism (re-seed reproduces the stream)
//   2. Autodiff correctness via numericalGradCheck on several ops
//   3. Freeze semantics: frozen leaf participates in forward but gets zero grad / never moves
//   4. A real (tiny) training run on a TransformerBlock drives loss down
//   5. viz + mem + checkpoint round-trip
//
// Run: npx tsx src/core/_smoke.ts

import { seed, uniform, normal, randint } from "./prng.js";
import { Tensor, numericalGradCheck } from "./tensor.js";
import { Linear, TransformerBlock } from "./nn.js";
import { Adam, SGD } from "./optim.js";
import { genPretrainFinetune, batches } from "./data.js";
import { lossCurve, sparkline, bar, heatmap, histogram } from "./viz.js";
import { estBytes, toMB } from "./mem.js";
import { dump, loadBase } from "./checkpoint.js";

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error("ASSERT FAILED: " + msg);
}

// 1. PRNG determinism --------------------------------------------------------
seed(1234);
const a = [uniform(), normal(), randint(0, 100)];
seed(1234);
const b = [uniform(), normal(), randint(0, 100)];
assert(a.every((v, i) => v === b[i]), "PRNG not reproducible after re-seed");
console.log("[1] PRNG determinism OK; first uniform =", a[0].toFixed(8));

// 2. Grad check on several ops ----------------------------------------------
seed(1);
const x = Tensor.from([0.3, -1.2, 0.7, 2.1, -0.5, 1.4], [2, 3], true);
const errGelu = numericalGradCheck((t) => t.gelu().sum(), x);
const errRelu = numericalGradCheck((t) => t.relu().sum(), x);
const errSoftmax = numericalGradCheck((t) => t.softmax().mul(t).sum(), x);
const W = Tensor.from([0.1, 0.2, -0.3, 0.4, 0.5, -0.6], [3, 2], false);
const errMatmul = numericalGradCheck((t) => t.matmul(W).sum(), x);
console.log("[2] gradcheck max-rel-err: gelu=%s relu=%s softmax=%s matmul=%s",
  errGelu.toExponential(2), errRelu.toExponential(2), errSoftmax.toExponential(2), errMatmul.toExponential(2));
assert(errGelu < 1e-4 && errRelu < 1e-4 && errSoftmax < 1e-4 && errMatmul < 1e-4, "grad check exceeded 1e-4");

// 3. Freeze semantics --------------------------------------------------------
seed(2);
const lin = new Linear(4, 3);
lin.W.requires_grad = false; // freeze the weight; keep bias trainable
const inp = Tensor.from([1, 2, 3, 4], [1, 4], true);
const y = lin.forward(inp).sum();
lin.zeroGrad();
y.backward();
const wGradNorm = lin.W.grad.reduce((s, v) => s + Math.abs(v), 0);
assert(wGradNorm === 0, "frozen weight received nonzero grad");
assert(lin.b.grad.some((v) => v !== 0), "trainable bias got no grad");
console.log("[3] freeze OK: frozen W grad-norm=%s (expected 0), trainable=%d/%d params",
  wGradNorm, lin.numParams({ trainableOnly: true }), lin.numParams());

// 4. Real training run on a TransformerBlock ---------------------------------
seed(1234);
const dModel = 16;
const block = new TransformerBlock(dModel, 2, 32);
// task: map an input (seq,dModel) to its row-reversed version (uses positional structure)
const data = genPretrainFinetune({ nPretrain: 64, seqLen: 6, vocab: 8 });
// build fixed random embeddings for the small vocab (frozen lookup table substitute)
const embTable: number[][] = Array.from({ length: data.vocab }, () =>
  Array.from({ length: dModel }, () => normal(0, 0.5)),
);
function embed(ids: number[]): Tensor {
  const d: number[] = [];
  for (const id of ids) d.push(...embTable[id]);
  return Tensor.from(d, [ids.length, dModel], false);
}
const opt = new Adam(block.trainable(), 5e-3);
const losses: number[] = [];
for (let step = 0; step < 60; step++) {
  let totalLoss = 0;
  let count = 0;
  for (const batch of batches(data.pretrain, 16)) {
    for (const ex of batch) {
      const inT = embed(ex.input);
      const tgtT = embed(ex.target);
      const out = block.forward(inT);
      const diff = out.sub(tgtT);
      const loss = diff.mul(diff).mean(); // MSE on embeddings (regression proxy)
      opt.zeroGrad();
      loss.backward();
      opt.step();
      totalLoss += loss.data[0];
      count++;
    }
  }
  losses.push(totalLoss / count);
}
console.log("[4] training: loss %s -> %s", losses[0].toFixed(4), losses[losses.length - 1].toFixed(4));
console.log("    sparkline:", sparkline(losses));
assert(losses[losses.length - 1] < losses[0] * 0.7, "loss did not drop >=30%");
console.log(lossCurve(losses, { height: 6, label: "    MSE loss" }));

// SGD also works (sanity)
seed(5);
const lin2 = new Linear(3, 1);
const sgd = new SGD(lin2.trainable(), 0.05, 0.9);
const tx = Tensor.from([1, 2, 3], [1, 3], false);
let l0 = 0, l1 = 0;
for (let i = 0; i < 50; i++) {
  const pred = lin2.forward(tx);
  const target = Tensor.from([5], [1, 1], false);
  const loss = pred.sub(target).mul(pred.sub(target)).mean();
  sgd.zeroGrad();
  loss.backward();
  sgd.step();
  if (i === 0) l0 = loss.data[0];
  l1 = loss.data[0];
}
console.log("[5] SGD regression: %s -> %s", l0.toFixed(4), l1.toFixed(4));
assert(l1 < l0, "SGD did not reduce loss");

// 6. checkpoint round-trip ---------------------------------------------------
const ckpt = dump(block);
const block2 = new TransformerBlock(dModel, 2, 32);
loadBase(block2, ckpt, true); // freeze
const out1 = block.forward(embed(data.pretrain[0].input));
const out2 = block2.forward(embed(data.pretrain[0].input));
let maxDiff = 0;
for (let i = 0; i < out1.size; i++) maxDiff = Math.max(maxDiff, Math.abs(out1.data[i] - out2.data[i]));
assert(maxDiff < 1e-12, "checkpoint reload not bit-identical");
assert(block2.numParams({ trainableOnly: true }) === 0, "loaded base not frozen");
console.log("[6] checkpoint reload max-diff=%s; reloaded trainable params=%d (frozen base OK)",
  maxDiff.toExponential(2), block2.numParams({ trainableOnly: true }));

// 7. mem estimator + bars ----------------------------------------------------
const P = 1_000_000;
const fullFT = estBytes({ totalParams: P, trainableParams: P, optMultiplier: 2 });
const lora = estBytes({ totalParams: P, trainableParams: 5000, optMultiplier: 2 });
const qlora = estBytes({ totalParams: P, trainableParams: 5000, optMultiplier: 2, bytesPerBaseParam: 0.5 });
console.log("[7] mem estimate (est., toy formula):");
console.log(bar([
  { label: "full-FT", value: Math.round(toMB(fullFT.totalBytes) * 100) / 100, note: "MB (est.)" },
  { label: "LoRA", value: Math.round(toMB(lora.totalBytes) * 100) / 100, note: "MB (est.)" },
  { label: "QLoRA", value: Math.round(toMB(qlora.totalBytes) * 100) / 100, note: "MB (est.)" },
]));

// 8. heatmap of a rank-1 matrix (the LoRA payoff visual) ---------------------
const u = [1, 0.6, -0.4, 0.2];
const v = [0.9, -0.5, 0.3, 0.7, -0.2];
const rank1 = new Float64Array(u.length * v.length);
for (let i = 0; i < u.length; i++) for (let j = 0; j < v.length; j++) rank1[i * v.length + j] = u[i] * v[j];
console.log("[8] rank-1 matrix heatmap (LoRA BA structure):");
console.log(heatmap(rank1, [u.length, v.length], { label: "    ΔW≈BA" }));
console.log("    weight histogram:");
console.log(histogram(Array.from(rank1), { bins: 8, width: 24, label: "    " }));

console.log("\n⚠ toy-scale: 绝对值偏乐观，可迁移的是机制与曲线形状。");
console.log("ALL CORE SMOKE CHECKS PASSED");
