// core/_smoke.ts — Standalone validation of the shared core (NOT a stage; not imported
// by any stage). Run with `npx tsx src/core/_smoke.ts`. Exists so the scaffold's
// "honest number" claim is itself verified: the autograd engine is numerically checked
// against finite differences before any chapter trusts it.
//
// WHY finite-difference grad-check is the load-bearing test: a from-scratch autograd's
//   only real failure mode is a wrong adjoint (broadcast sum, matmul transpose, softmax
//   Jacobian). Comparing analytic grads to central finite differences catches all of
//   them. If max relative error here is ~1e-6, the engine is trustworthy at Float64.

import { makeRng } from "./prng.js";
import { Tensor } from "./tensor.js";
import { Linear, Embedding, LayerNorm } from "./nn.js";
import { Adam, clipGradNorm } from "./optim.js";
import { copyTask, addingProblem, parityTask, delayedRecall, charSeq, batches } from "./data.js";
import { sparkline, lineChart, histogram, heatmap, bar } from "./plot.js";
import { accuracy, perplexity, timeit, countMACs, argmax, mse } from "./metrics.js";

let failures = 0;
function check(name: string, ok: boolean, detail = ""): void {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
}

// --- 1. PRNG determinism --------------------------------------------------
{
  const a = makeRng(42);
  const b = makeRng(42);
  const sa = [a.next(), a.next(), a.next()];
  const sb = [b.next(), b.next(), b.next()];
  check("prng determinism (same seed)", sa.every((v, i) => v === sb[i]), `first=${sa[0].toFixed(6)}`);
  const c = makeRng(43);
  check("prng seed sensitivity", c.next() !== sa[0]);
  // normal() mean ~0 over many draws
  const r = makeRng(7);
  let s = 0;
  const N = 20000;
  for (let i = 0; i < N; i++) s += r.normal();
  const mean = s / N;
  check("prng normal mean ~0", Math.abs(mean) < 0.03, `mean=${mean.toFixed(4)}`);
}

// --- 2. Autograd numerical grad-check -------------------------------------
// Build a small scalar-output function of two leaf tensors using a representative mix of
// ops (matmul, add-broadcast, tanh, softmax via cross-entropy, slice, concat) and verify
// analytic grad == central finite-difference grad.
{
  const rng = makeRng(123);
  const x = Tensor.randn([3, 4], rng); // input rows
  const W = Tensor.randn([4, 5], rng, 0.5); // weights
  const b = Tensor.randn([1, 5], rng, 0.5); // bias (broadcast over rows)
  const targets = [0, 2, 4];

  const forward = (): Tensor => {
    const h = x.matmul(W).add(b).tanh(); // [3,5]
    // exercise slice+concat: split columns then re-concat (identity-ish, but tests adjoints)
    const left = h.slice(1, 0, 2); // [3,2]
    const right = h.slice(1, 2, 5); // [3,3]
    const recombined = Tensor.concat([left, right], 1); // [3,5]
    return recombined.crossEntropy(targets); // scalar
  };

  const loss = forward();
  loss.backward();
  const analytic = Float64Array.from(W.grad); // check grad wrt W

  const eps = 1e-6;
  let maxRel = 0;
  for (let i = 0; i < W.size; i++) {
    const orig = W.data[i];
    W.data[i] = orig + eps;
    const lp = forward().data[0];
    W.data[i] = orig - eps;
    const lm = forward().data[0];
    W.data[i] = orig;
    const num = (lp - lm) / (2 * eps);
    const denom = Math.max(1e-8, Math.abs(num) + Math.abs(analytic[i]));
    maxRel = Math.max(maxRel, Math.abs(num - analytic[i]) / denom);
  }
  check("autograd grad-check (W via matmul+tanh+slice+concat+CE)", maxRel < 1e-5, `maxRelErr=${maxRel.toExponential(2)}`);
}

// --- 3. nn primitives grad-check (Embedding scatter-add, LayerNorm) --------
{
  const rng = makeRng(321);
  const emb = new Embedding(6, 4, rng, 0.5);
  const ln = new LayerNorm(4);
  const lin = new Linear(4, 3, rng);
  const ids = [1, 3, 1, 5]; // note repeated id 1 -> tests scatter-ADD
  const tgts = [0, 1, 2, 0];
  const fwd = (): Tensor => lin.forward(ln.forward(emb.forward(ids))).crossEntropy(tgts);
  const loss = fwd();
  emb.zeroGrad();
  ln.zeroGrad();
  lin.zeroGrad();
  loss.backward();
  // check embedding grad for the repeated row (id=1) numerically
  const w = emb.weight;
  const analytic = Float64Array.from(w.grad);
  const eps = 1e-6;
  let maxRel = 0;
  for (let i = 0; i < w.size; i++) {
    const o = w.data[i];
    w.data[i] = o + eps;
    const lp = fwd().data[0];
    w.data[i] = o - eps;
    const lm = fwd().data[0];
    w.data[i] = o;
    const num = (lp - lm) / (2 * eps);
    const denom = Math.max(1e-8, Math.abs(num) + Math.abs(analytic[i]));
    maxRel = Math.max(maxRel, Math.abs(num - analytic[i]) / denom);
  }
  check("nn grad-check (Embedding scatter-add + LayerNorm + Linear)", maxRel < 1e-5, `maxRelErr=${maxRel.toExponential(2)}`);
}

// --- 4. Optimizer reduces a convex loss -----------------------------------
// Fit Linear to a fixed linear target; Adam should drive MSE-ish CE loss down.
{
  const rng = makeRng(99);
  const lin = new Linear(3, 2, rng);
  const opt = new Adam(lin.params(), { lr: 0.05 });
  const X = Tensor.randn([16, 3], rng);
  const tgts = Array.from({ length: 16 }, (_, i) => i % 2);
  const losses: number[] = [];
  let lastNorm = 0;
  for (let step = 0; step < 80; step++) {
    opt.zeroGrad();
    const loss = lin.forward(X).crossEntropy(tgts);
    loss.backward();
    lastNorm = clipGradNorm(lin.params(), 5.0);
    opt.step();
    losses.push(loss.data[0]);
  }
  check("optim Adam reduces loss", losses[losses.length - 1] < losses[0] * 0.9, `loss ${losses[0].toFixed(3)} -> ${losses[losses.length - 1].toFixed(3)}`);
  check("clipGradNorm returns finite pre-clip norm", Number.isFinite(lastNorm), `norm=${lastNorm.toFixed(4)}`);
  console.log("  loss sparkline:", sparkline(losses));
}

// --- 5. Data generators are well-formed -----------------------------------
{
  const rng = makeRng(11);
  const copy = copyTask(rng, { count: 4, k: 2, delay: 3, symbols: 4 });
  check("copyTask shape", copy.X.length === 4 && copy.X[0].length === copy.Y[0].length, copy.describe());
  const add = addingProblem(rng, { count: 4, T: 6 });
  check("addingProblem packs 2*T features + 1 target", add.X[0].length === 12 && add.Y[0].length === 1, add.describe());
  const par = parityTask(rng, { count: 4, T: 8 });
  // verify parity is actually XOR of bits
  const okParity = par.X.every((x, n) => (x.reduce((a, b) => a ^ b, 0)) === par.Y[n][0]);
  check("parityTask target == XOR of bits", okParity, par.describe());
  const rec = delayedRecall(rng, { count: 4, delay: 5, symbols: 3 });
  const okRecall = rec.X.every((x, n) => x[0] === rec.Y[n][0]);
  check("delayedRecall target == cue at step 0", okRecall, rec.describe());
  const ch = charSeq({ seqLen: 16, stride: 16 });
  const roundtrip = ch.decode(ch.encode("the sun")) === "the sun";
  check("charSeq encode/decode roundtrip", roundtrip, ch.describe());
  // batch iterator covers all indices exactly once
  const seen = new Set<number>();
  for (const bt of batches(10, 3, rng)) for (const i of bt) seen.add(i);
  check("batches cover all indices (incl partial last)", seen.size === 10);
}

// --- 6. Metrics ------------------------------------------------------------
{
  check("accuracy", accuracy([1, 2, 3, 4], [1, 0, 3, 0]) === 0.5);
  check("perplexity(0)==1", Math.abs(perplexity(0) - 1) < 1e-9);
  check("perplexity(log V)==V", Math.abs(perplexity(Math.log(7)) - 7) < 1e-9);
  check("argmax deterministic on ties", argmax([0.5, 0.5, 0.1]) === 0);
  check("mse", Math.abs(mse([1, 2], [1, 4]) - 2) < 1e-9);
  // countMACs: attention ~n^2, scan ~n -> ratio grows with n
  const m1 = countMACs.attention(32, 16) / countMACs.scan(32, 8, 16);
  const m2 = countMACs.attention(64, 16) / countMACs.scan(64, 8, 16);
  check("countMACs attention/scan ratio grows with n", m2 > m1 * 1.5, `ratio 32->${m1.toFixed(1)} 64->${m2.toFixed(1)}`);
  // timeit returns real positive ms
  const t = timeit(() => {
    let s = 0;
    for (let i = 0; i < 1e5; i++) s += Math.sqrt(i);
    if (s < 0) throw new Error("unreachable");
  }, 3);
  check("timeit returns positive wall-clock", t.perRepMs > 0, `perRep=${t.perRepMs.toFixed(3)}ms`);
}

// --- 7. Plot renderers don't throw on edge cases --------------------------
{
  check("sparkline handles NaN", sparkline([1, NaN, 3]).includes("?"));
  const lc = lineChart([[3, 2, 1], [1, 2, 3]], { width: 20, height: 5, labels: ["down", "up"] });
  check("lineChart renders axis + legend", lc.includes("|") && lc.includes("down"));
  const hg = histogram([0.1, 0.2, 0.2, 5.0], 5);
  check("histogram renders bars", hg.includes("#"));
  const hm = heatmap([[0, 1], [1, 0]]);
  check("heatmap renders 2 rows", hm.split("\n").length === 2);
  const br = bar([{ label: "rnn", value: 10 }, { label: "lstm", value: 4 }]);
  check("bar renders blocks", br.includes("█"));
  console.log("\n  --- sample heatmap (identity-ish 4x4) ---");
  console.log(heatmap([[1, 0.2, 0.1, 0], [0.2, 1, 0.2, 0.1], [0.1, 0.2, 1, 0.2], [0, 0.1, 0.2, 1]]));
}

console.log(`\n${failures === 0 ? "ALL CORE SMOKE CHECKS PASSED" : failures + " CHECK(S) FAILED"}`);
if (failures > 0) process.exit(1);
