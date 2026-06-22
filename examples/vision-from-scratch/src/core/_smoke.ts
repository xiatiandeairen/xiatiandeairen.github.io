// core/_smoke.ts — Self-check that the shared core is correct before any stage relies on it.
//
// WHY this exists: stages make "honest number" claims that assume conv/pool/BN gradients
//   are right. This file independently verifies them with gradCheck (analytic vs numerical)
//   and sanity-checks shapes/data. It is NOT a stage (not wired into package.json scripts),
//   so running stages never triggers it. Run manually: `npx tsx src/core/_smoke.ts`.
//
// HONESTY: gradCheck tolerances are real numerical results, not asserted constants. We
//   print the measured max relative error per op; anything > 1e-4 is a failing gradient.

import { Tensor, conv2d, maxpool2d, avgpool2d, type Conv2dParams } from "./autograd.js";
import { Conv2d, BatchNorm2d, ResidualBlock, Sequential, ReLU } from "./nn.js";
import { mulberry32, randn, type Rng } from "./rng.js";
import { crossEntropy, gradCheck, accuracy } from "./metrics.js";
import { makeShapeDataset, stackBatch, SHAPE_NAMES } from "./data.js";

function randTensor(shape: number[], rng: Rng): Tensor {
  return Tensor.from(shape, () => randn(rng));
}

let failures = 0;
function report(name: string, relError: number, tol = 1e-4): void {
  const ok = relError <= tol && Number.isFinite(relError);
  if (!ok) failures++;
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name.padEnd(28)} max rel error = ${relError.toExponential(2)}`);
}

function main(): void {
  const rng = mulberry32(7);
  console.log("=== core/_smoke: gradient checks (analytic vs numerical) ===");

  // conv2d: small input, 2 out channels, 3x3 kernel, stride 1, pad 1.
  {
    const x = randTensor([1, 2, 5, 5], rng);
    const w = randTensor([2, 2, 3, 3], rng);
    const b = randTensor([2], rng);
    const p: Conv2dParams = { stride: 1, padding: 1 };
    const tgt = new Int32Array([1]);
    const f = () => {
      const out = conv2d(x, w, b, p); // (1,2,5,5)
      return crossEntropy(out.flatten(), tgt).data[0];
    };
    const out = conv2d(x, w, b, p);
    const loss = crossEntropy(out.flatten(), tgt);
    loss.backward();
    report("conv2d (x,w,b)", gradCheck(f, [x, w, b]).maxRelError);
  }

  // conv2d stride 2, no padding — exercises the strided im2col index map.
  {
    const x = randTensor([2, 1, 6, 6], rng);
    const w = randTensor([3, 1, 2, 2], rng);
    const p: Conv2dParams = { stride: 2, padding: 0 };
    const tgt = new Int32Array([0, 2]);
    const f = () => crossEntropy(conv2d(x, w, null, p).flatten().reshape([2, 27]), tgt).data[0];
    const out = conv2d(x, w, null, p);
    crossEntropy(out.flatten().reshape([2, 27]), tgt).backward();
    report("conv2d stride2", gradCheck(f, [x, w]).maxRelError);
  }

  // maxpool2d: gradient routes only to argmax positions.
  {
    const x = randTensor([1, 2, 4, 4], rng);
    const tgt = new Int32Array([3]);
    const f = () => crossEntropy(maxpool2d(x, { kernel: 2, stride: 2 }).flatten(), tgt).data[0];
    crossEntropy(maxpool2d(x, { kernel: 2, stride: 2 }).flatten(), tgt).backward();
    report("maxpool2d", gradCheck(f, [x]).maxRelError);
  }

  // avgpool2d: gradient spreads equally over the window.
  {
    const x = randTensor([1, 2, 4, 4], rng);
    const tgt = new Int32Array([5]);
    const f = () => crossEntropy(avgpool2d(x, { kernel: 2, stride: 2 }).flatten(), tgt).data[0];
    crossEntropy(avgpool2d(x, { kernel: 2, stride: 2 }).flatten(), tgt).backward();
    report("avgpool2d", gradCheck(f, [x]).maxRelError);
  }

  // BatchNorm2d (training path): per-channel normalization backward.
  {
    const bn = new BatchNorm2d(2);
    const x = randTensor([4, 2, 3, 3], rng);
    const tgt = new Int32Array([0, 1, 0, 1]);
    const f = () => crossEntropy(bn.forward(x).flatten().reshape([4, 18]), tgt).data[0];
    crossEntropy(bn.forward(x).flatten().reshape([4, 18]), tgt).backward();
    report("batchnorm2d", gradCheck(f, [x, bn.gamma, bn.beta]).maxRelError);
  }

  // ResidualBlock end-to-end: skip connection grad path.
  {
    const block = new ResidualBlock(
      new Sequential([new Conv2d(2, 2, 3, rng, { padding: 1 }), new ReLU()]),
    );
    const x = randTensor([1, 2, 5, 5], rng);
    const tgt = new Int32Array([7]);
    const f = () => crossEntropy(block.forward(x).flatten(), tgt).data[0];
    block.zeroGrad();
    crossEntropy(block.forward(x).flatten(), tgt).backward();
    report("residual block", gradCheck(f, [x, ...block.parameters()]).maxRelError);
  }

  console.log("\n=== core/_smoke: data + metrics sanity ===");
  const ds = makeShapeDataset(3, 16, 16, rng, 0.05);
  const batch = stackBatch(ds.samples.slice(0, 6), 16, 16);
  console.log(`  dataset: ${ds.samples.length} imgs, ${ds.classes} classes (${SHAPE_NAMES.join("/")})`);
  console.log(`  batch tensor shape: [${batch.N}, ${batch.C}, ${batch.H}, ${batch.W}]`);
  // accuracy on random logits should be near chance — a sanity floor.
  const logits = randTensor([batch.N, ds.classes], rng);
  console.log(`  random-logit accuracy: ${(accuracy(logits, batch.labels) * 100).toFixed(1)}% (chance ~${(100 / ds.classes).toFixed(0)}%)`);

  console.log(`\n${failures === 0 ? "ALL CORE CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
  if (failures > 0) process.exitCode = 1;
}

main();
