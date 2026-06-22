// core/_smoke.ts — Internal sanity check for the shared core. NOT a book stage.
//
// WHY this exists: the core is frozen and every stage trusts it. Before any stage author
//   builds on top, this proves the engine's adjoints (via gradCheck), the optimizer
//   actually minimizes, the metrics compute the expected values on hand-checkable inputs,
//   and the data generators are deterministic. Run: `npx tsx src/core/_smoke.ts`.
//
// It is intentionally verbose and asserts hard (throws) so CI / a reader catches
// regressions immediately rather than discovering a wrong gradient three chapters later.

import { rng } from "./prng.js";
import { Value, gradCheck } from "./tensor.js";
import { Linear, Expert, LayerNorm, crossEntropy } from "./nn.js";
import { Adam, SGD } from "./optim.js";
import { makeClusters, makeModularTask, makeTokenStream } from "./data.js";
import { plotLoss, bar, heatmap, hist } from "./viz.js";
import {
  routingEntropy,
  expertUtilization,
  loadBalanceLoss,
  activatedFLOPs,
  denseFLOPs,
  coefficientOfVariation,
} from "./metrics.js";

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error("SMOKE FAIL: " + msg);
}

console.log("=== core smoke test (seed=42) ===\n");

// ---- 1. gradCheck the engine on a composite scalar loss ---------------------
// f(x) = sum( softmax(x*W)·... ) — exercise matmul, softmax, gather, log, sum.
{
  const r = rng(42);
  const W = new Value(
    Float64Array.from({ length: 4 * 3 }, () => r.normal()),
    [4, 3],
  );
  const target = 1;
  // f rebuilds graph from x each call (gradCheck perturbs x.data in place).
  const f = (x: Value): Value => {
    const logits = x.matmul(W); // [1,3]
    return crossEntropy(logits, target);
  };
  const x = new Value(Float64Array.from({ length: 4 }, () => r.normal()), [1, 4]);
  const err = gradCheck(f, x);
  console.log(`gradCheck (matmul+crossEntropy) max rel err = ${err.toExponential(3)}`);
  assert(err < 1e-4, `gradCheck error too high: ${err}`);
}

// ---- 2. gradCheck an Expert MLP forward -------------------------------------
{
  const r = rng(7);
  const exp = new Expert(3, 5, 2, r, "gelu");
  const f = (x: Value): Value => crossEntropy(exp.forward(x), 0);
  const x = new Value(Float64Array.from({ length: 3 }, () => r.normal()), [1, 3]);
  const err = gradCheck(f, x);
  console.log(`gradCheck (Expert MLP + GELU)   max rel err = ${err.toExponential(3)}`);
  assert(err < 1e-3, `expert gradCheck too high: ${err}`);
}

// ---- 3. LayerNorm gradCheck -------------------------------------------------
// NOTE: we DON'T test ln(x).sum() — its analytic grad is ~0 (LayerNorm output is mean-
//   centered so the sum barely moves), which inflates RELATIVE finite-diff error into
//   false alarms. We instead dot the normed output with fixed weights so the target has
//   a non-degenerate gradient; then eps=1e-4 central diff is trustworthy.
{
  const r = rng(11);
  const ln = new LayerNorm(4);
  const w = new Value([0.3, -0.7, 1.1, 0.5], [4, 1]); // fixed projection -> scalar
  const f = (x: Value): Value => ln.forward(x).matmul(w);
  const x = new Value(Float64Array.from({ length: 4 }, () => r.normal() * 2 + 1), [1, 4]);
  const err = gradCheck(f, x, 1e-4);
  console.log(`gradCheck (LayerNorm·w)         max rel err = ${err.toExponential(3)}`);
  assert(err < 1e-4, `layernorm gradCheck too high: ${err}`);
}

// ---- 4. Optimizer actually minimizes a tiny classification ------------------
{
  const r = rng(123);
  const ds = makeClusters(3, 120, 4, r);
  const clf = new Linear(4, 3, r);
  const opt = new Adam(clf.parameters(), 0.05);
  const losses: number[] = [];
  for (let epoch = 0; epoch < 40; epoch++) {
    opt.zeroGrad();
    let total: Value = Value.scalar(0);
    for (let i = 0; i < ds.X.length; i++) {
      const x = Value.from(ds.X[i], [1, 4]);
      total = total.add(crossEntropy(clf.forward(x), ds.Y[i]));
    }
    const loss = total.mul(Value.scalar(1 / ds.X.length));
    loss.backward();
    opt.step();
    losses.push(loss.data[0]);
  }
  console.log(`\nAdam on 3-cluster classify: loss ${losses[0].toFixed(3)} -> ${losses[losses.length - 1].toFixed(3)}`);
  console.log("  " + plotLoss(losses));
  assert(losses[losses.length - 1] < losses[0] * 0.5, "Adam failed to halve loss");
}

// ---- 5. SGD+momentum also descends -----------------------------------------
{
  const r = rng(123);
  const ds = makeClusters(3, 80, 4, r);
  const clf = new Linear(4, 3, r);
  const opt = new SGD(clf.parameters(), 0.1, 0.9);
  let first = 0,
    last = 0;
  for (let epoch = 0; epoch < 60; epoch++) {
    opt.zeroGrad();
    let total: Value = Value.scalar(0);
    for (let i = 0; i < ds.X.length; i++) total = total.add(crossEntropy(clf.forward(Value.from(ds.X[i], [1, 4])), ds.Y[i]));
    const loss = total.mul(Value.scalar(1 / ds.X.length));
    loss.backward();
    opt.step();
    if (epoch === 0) first = loss.data[0];
    last = loss.data[0];
  }
  console.log(`SGD+momentum: loss ${first.toFixed(3)} -> ${last.toFixed(3)}`);
  assert(last < first, "SGD failed to descend");
}

// ---- 6. Metrics on hand-checkable inputs -----------------------------------
{
  // uniform vs collapsed assignments over 4 experts.
  const uniform = expertUtilization([0, 1, 2, 3, 0, 1, 2, 3], 4);
  const collapsed = expertUtilization([0, 0, 0, 0, 0, 0, 0, 1], 4);
  console.log(`\nutil uniform   = [${uniform.map((u) => u.toFixed(2)).join(", ")}]  CV=${coefficientOfVariation(uniform).toFixed(3)}`);
  console.log(`util collapsed = [${collapsed.map((u) => u.toFixed(2)).join(", ")}]  CV=${coefficientOfVariation(collapsed).toFixed(3)}`);
  assert(Math.abs(coefficientOfVariation(uniform)) < 1e-9, "uniform CV should be 0");
  assert(coefficientOfVariation(collapsed) > 1.0, "collapsed CV should be large");

  const hUniform = routingEntropy([0.25, 0.25, 0.25, 0.25]);
  const hPeaked = routingEntropy([0.97, 0.01, 0.01, 0.01]);
  console.log(`entropy uniform = ${hUniform.toFixed(4)} nats (max=ln4=${Math.log(4).toFixed(4)}), peaked = ${hPeaked.toFixed(4)}`);
  assert(Math.abs(hUniform - Math.log(4)) < 1e-9, "uniform entropy should equal ln(4)");
  assert(hPeaked < hUniform, "peaked entropy should be lower");

  // loadBalanceLoss: build E=4 gate rows. Balanced rows -> aux near 1.0.
  const balanced: Value[] = [];
  const assigns: number[] = [];
  for (let e = 0; e < 4; e++) {
    const row = Value.from([0.25, 0.25, 0.25, 0.25], [1, 4]);
    balanced.push(row);
    assigns.push(e);
  }
  const auxBal = loadBalanceLoss(balanced, assigns, 4);
  console.log(`loadBalanceLoss (perfectly balanced) = ${auxBal.data[0].toFixed(4)} (ideal=1.0)`);
  assert(Math.abs(auxBal.data[0] - 1.0) < 1e-9, "balanced aux should be 1.0");

  // imbalanced: everything routed to expert 0, prob mass also on 0.
  const imbal: Value[] = [];
  const imbalAssigns: number[] = [];
  for (let t = 0; t < 4; t++) {
    imbal.push(Value.from([0.85, 0.05, 0.05, 0.05], [1, 4]));
    imbalAssigns.push(0);
  }
  const auxImbal = loadBalanceLoss(imbal, imbalAssigns, 4);
  console.log(`loadBalanceLoss (all -> expert 0)     = ${auxImbal.data[0].toFixed(4)} (>1.0 = imbalanced)`);
  assert(auxImbal.data[0] > 1.5, "imbalanced aux should exceed balanced");

  // aux loss must be differentiable wrt gate probs (grad flows into meanP).
  auxImbal.backward();
  let gradMag = 0;
  for (const row of imbal) for (let j = 0; j < row.grad.length; j++) gradMag += Math.abs(row.grad[j]);
  console.log(`aux loss grad magnitude into gate probs = ${gradMag.toFixed(4)} (must be > 0)`);
  assert(gradMag > 0, "aux loss must be differentiable into gate probs");
}

// ---- 7. FLOP accounting: sparse beats dense by ~k/E ------------------------
{
  const T = 256,
    inDim = 64,
    hidden = 128,
    outDim = 64,
    E = 8,
    k = 2;
  const act = activatedFLOPs(T, inDim, hidden, outDim, E, k);
  const dense = denseFLOPs(T, inDim, hidden, outDim, E);
  const ratio = act / dense;
  console.log(`\nFLOPs (toy): MoE top-${k}/${E} = ${act.toLocaleString()} MAC, dense = ${dense.toLocaleString()} MAC`);
  console.log(`  activated/dense ratio = ${ratio.toFixed(3)} (≈ k/E = ${(k / E).toFixed(3)}; gate overhead makes it slightly above)`);
  assert(ratio < k / E + 0.1 && ratio > k / E - 0.01, "sparse ratio off");
}

// ---- 8. Data generators are deterministic ----------------------------------
{
  const a = makeModularTask(6, 30, 8, rng(99), 2);
  const b = makeModularTask(6, 30, 8, rng(99), 2);
  let same = true;
  for (let i = 0; i < a.X.length; i++) {
    if (a.Y[i] !== b.Y[i] || a.clusterId[i] !== b.clusterId[i]) same = false;
    for (let d = 0; d < a.dim; d++) if (a.X[i][d] !== b.X[i][d]) same = false;
  }
  console.log(`\nmakeModularTask determinism (same seed twice): ${same ? "IDENTICAL ✓" : "DIVERGED ✗"}`);
  assert(same, "data generator not deterministic");

  const ts = makeTokenStream(40, 20, rng(5), 4);
  const topicsOk = ts.tokens.every((tok, i) => tok % 4 === ts.topic[i]);
  console.log(`makeTokenStream topic invariant (id % numTopics == topic): ${topicsOk ? "OK ✓" : "BROKEN ✗"}`);
  assert(topicsOk, "token stream topic invariant broken");
}

// ---- 9. viz renders (visual confirmation only) ------------------------------
{
  console.log("\n--- viz smoke ---");
  console.log("bar (expert util):");
  console.log(bar(["E0", "E1", "E2", "E3"], [0.4, 0.3, 0.2, 0.1]));
  console.log("\nheatmap (expert×cluster co-occurrence, near-diagonal = specialized):");
  console.log(
    heatmap(
      [
        [8, 1, 0, 1],
        [0, 9, 1, 0],
        [1, 0, 7, 2],
        [0, 1, 1, 8],
      ],
      ["E0", "E1", "E2", "E3"],
      ["c0", "c1", "c2", "c3"],
    ),
  );
  console.log("\nhist (gate prob distribution, mass near 1.0 = confident routing):");
  console.log(hist([0.9, 0.95, 0.88, 0.3, 0.5, 0.99, 0.92, 0.85, 0.4, 0.97], 6));
}

console.log("\n=== ALL SMOKE ASSERTIONS PASSED ===");
