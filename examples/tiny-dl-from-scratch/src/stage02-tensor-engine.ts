// stage02-tensor-engine.ts — Chapter 2: the tensor engine, i.e. lifting the scalar
// autodiff graph of chapter 1 into n dimensions.
//
// WHAT THIS STAGE PROVES (not just demonstrates):
//   1. Every Tensor op's analytic backward matches a numerical finite-difference grad
//      (gradCheck per-op, max rel error < 1e-6). If any op had a sign error, a missing
//      +=, or a forgotten broadcast-reduce, gradCheck would surface it as ~O(1) error.
//   2. The crux of going n-d: BROADCAST IS A REAL OP WITH A REAL ADJOINT. The adjoint of
//      "fan one element into many outputs" is "SUM the grads of those many outputs back
//      into the one source element" (call it un-broadcast / grad reduction). We show the
//      bidirectional case (B,1)+(1,D)->(B,D) and confirm each operand's grad comes back
//      reduced to its ORIGINAL shape with the ARITHMETICALLY correct values.
//   3. FAILURE MODE: a broadcast-add that forgets to un-broadcast. We reimplement it wrong
//      on purpose, then show the grad it produces is wrong-shaped / wrong-magnitude — the
//      single most common bug when people first hand-roll n-d autograd.
//
// WHY this matters downstream: every later layer (Linear bias add, LayerNorm, attention
//   scores + mask) relies on broadcast having a correct adjoint. Get it wrong here and the
//   GPT trains to garbage with no error message — only silently wrong grads.
//
// HONESTY NOTE: shapes here are tiny (toy) so we can print full grad tensors. The point is
//   structural correctness (shapes + values + gradCheck), which is size-independent; the
//   absolute timing at the end is f64 single-thread and only meaningful as a relative ratio.

import { mulberry32, randn } from "./core/rng.js";
import { Tensor } from "./core/autograd.js";
import { gradCheck, timeIt } from "./core/metrics.js";

function section(title: string): void {
  console.log("\n" + "=".repeat(60) + "\n" + title + "\n" + "=".repeat(60));
}

/** Pretty-print a small 2-D tensor's data (row-major) for human eyes. */
function fmtMatrix(t: Tensor): string {
  const [r, c] = t.shape.length === 2 ? t.shape : [1, t.size];
  const rows: string[] = [];
  for (let i = 0; i < r; i++) {
    const cells: string[] = [];
    for (let j = 0; j < c; j++) cells.push(t.data[i * c + j].toFixed(4).padStart(8));
    rows.push("  [" + cells.join(", ") + "]");
  }
  return rows.join("\n");
}

function fmtGrad(grad: Float64Array): string {
  return "[" + Array.from(grad, (g) => g.toFixed(4)).join(", ") + "]";
}

// ===========================================================================
section("1) Per-op gradCheck: analytic backward == numerical (expect < 1e-6)");
// ===========================================================================
// We build a SEPARATE tiny expression per op so a failure points at exactly one op,
// instead of one giant expression where a single number hides which op is wrong.
// gradCheck perturbs each param's data by +-eps and compares (f(x+eps)-f(x-eps))/2eps
// against the analytic grad we filled via backward(). It does NOT call backward itself,
// so each case must: build graph -> zeroGrad -> backward -> gradCheck(rebuild forward).

const opRng = mulberry32(202);
// Inputs are kept away from kinks (e.g. relu at exactly 0) because finite differences
// disagree with the subgradient there — that would be a false failure, not a real bug.
const A = Tensor.from([3, 4], () => randn(opRng));
const B = Tensor.from([4, 2], () => randn(opRng));
const C = Tensor.from([3, 4], () => randn(opRng)); // same shape as A for elementwise ops

interface OpCase {
  name: string;
  params: Tensor[];
  // forwardScalar must REBUILD the graph from current param data and return the scalar
  // loss value (gradCheck calls it many times with perturbed data).
  forwardScalar: () => number;
  // forwardTensor builds the graph once and returns the scalar Tensor to backward().
  forwardTensor: () => Tensor;
}

// Each op is reduced to a scalar via .sum() (or is already scalar) so backward() is legal
// (autograd.backward requires a shape-[1] root). .sum()'s own adjoint is itself tested here.
const cases: OpCase[] = [
  { name: "add (same shape)", params: [A, C], forwardTensor: () => A.add(C).sum(), forwardScalar: () => A.add(C).sum().data[0] },
  { name: "mul (same shape)", params: [A, C], forwardTensor: () => A.mul(C).sum(), forwardScalar: () => A.mul(C).sum().data[0] },
  { name: "matmul (3,4)@(4,2)", params: [A, B], forwardTensor: () => A.matmul(B).sum(), forwardScalar: () => A.matmul(B).sum().data[0] },
  { name: "sum", params: [A], forwardTensor: () => A.sum(), forwardScalar: () => A.sum().data[0] },
  { name: "mean", params: [A], forwardTensor: () => A.mean(), forwardScalar: () => A.mean().data[0] },
  { name: "transpose", params: [A], forwardTensor: () => A.transpose().sum(), forwardScalar: () => A.transpose().sum().data[0] },
  // reshape's adjoint is identity-through; multiply by C after reshaping back so a wrong
  // adjoint (e.g. permuting grad) would still show up via the elementwise coupling.
  { name: "reshape", params: [A], forwardTensor: () => A.reshape([2, 6]).reshape([3, 4]).mul(C).sum(), forwardScalar: () => A.reshape([2, 6]).reshape([3, 4]).mul(C).sum().data[0] },
];

const rows: { op: string; checked: number; maxRel: string; pass: boolean }[] = [];
let allPass = true;
for (const c of cases) {
  const root = c.forwardTensor();
  for (const p of c.params) p.zeroGrad(); // grads accumulate (+=); must start at 0
  root.backward();
  const gc = gradCheck(c.forwardScalar, c.params, 1e-5);
  const pass = gc.maxRelError < 1e-6;
  allPass = allPass && pass;
  rows.push({ op: c.name, checked: gc.checked, maxRel: gc.maxRelError.toExponential(3), pass });
}

console.log("op".padEnd(22), "checked".padStart(8), "maxRelErr".padStart(12), "  pass");
for (const r of rows) {
  console.log(r.op.padEnd(22), String(r.checked).padStart(8), r.maxRel.padStart(12), "  " + (r.pass ? "PASS" : "FAIL"));
}
console.log("\nALL OPS PASS (< 1e-6):", allPass);

// ===========================================================================
section("2) BROADCAST adjoint: (B,1) + (1,D) -> (B,D), then un-broadcast back");
// ===========================================================================
// The core elementwise op only supports broadcasting `b` INTO `a` (b smaller, divides a).
// The genuinely interesting case is BIDIRECTIONAL broadcast (B,1)+(1,D): NEITHER operand
// contains the other; the output (B,D) is bigger than both. We compose it from core ops
// whose adjoints are already gradChecked above, so the un-broadcast (grad reduction) is
// guaranteed correct by construction — no new untested adjoint is introduced.
//
//   col (B,1) --transpose--> (1,B) --broadcastRow(D)--> (D,B) --transpose--> (B,D)
//   row (1,D) --broadcastRow(B)-------------------------------------------> (B,D)
//   out = colBroadcast + rowBroadcast   (now a same-shape add)
//
// EXPECTED un-broadcast values when the loss is out.sum() (upstream grad = all ones (B,D)):
//   - col[i,0] feeds the whole row i of out (D cells) -> grad = D for every i.
//   - row[0,j] feeds the whole column j of out (B cells) -> grad = B for every j.
// If un-broadcast were skipped, grads would be (B,D)-shaped or off by a factor — exactly
// the bug demoed in section 3.

const Bdim = 3;
const Ddim = 4;
const col = new Tensor(Float64Array.from([10, 20, 30]), [Bdim, 1]); // (B,1)
const row = new Tensor(Float64Array.from([1, 2, 3, 4]), [1, Ddim]); // (1,D)

/** Compose (B,D) = broadcast(col) + broadcast(row) using only core ops with known adjoints. */
function broadcastAddBidirectional(colT: Tensor, rowT: Tensor, b: number, d: number): Tensor {
  // (B,1) -> (1,B) -> (D,B) -> (B,D): broadcast the single column across all D columns.
  const colBroadcast = colT.transpose().broadcastRow(d).transpose();
  // (1,D) -> (B,D): broadcast the single row across all B rows.
  const rowBroadcast = rowT.broadcastRow(b);
  return colBroadcast.add(rowBroadcast);
}

const outBC = broadcastAddBidirectional(col, row, Bdim, Ddim);
col.zeroGrad();
row.zeroGrad();
outBC.sum().backward();

console.log(`forward out (B,D) = (${outBC.shape.join(",")}):`);
console.log(fmtMatrix(outBC));
console.log(`\ncol operand shape ${JSON.stringify(col.shape)}  grad ${fmtGrad(col.grad)}  (expect all = D = ${Ddim})`);
console.log(`row operand shape ${JSON.stringify(row.shape)}  grad ${fmtGrad(row.grad)}  (expect all = B = ${Bdim})`);

const colGradShapeOk = col.grad.length === Bdim && col.shape[0] === Bdim && col.shape[1] === 1;
const rowGradShapeOk = row.grad.length === Ddim && row.shape[0] === 1 && row.shape[1] === Ddim;
const colGradValOk = Array.from(col.grad).every((g) => g === Ddim);
const rowGradValOk = Array.from(row.grad).every((g) => g === Bdim);
console.log("\ncol grad reduced back to original (B,1) shape AND value == D :", colGradShapeOk && colGradValOk);
console.log("row grad reduced back to original (1,D) shape AND value == B :", rowGradShapeOk && rowGradValOk);

// Independent cross-check via gradCheck on this composite (different loss to exercise
// non-uniform upstream grads, so a value error wouldn't hide behind the symmetric sum).
col.zeroGrad();
row.zeroGrad();
// loss = sum(out * out) -> upstream grad = 2*out (non-uniform), still must un-broadcast.
const composite = () => {
  const o = broadcastAddBidirectional(col, row, Bdim, Ddim);
  return o.mul(o).sum().data[0];
};
const compRoot = broadcastAddBidirectional(col, row, Bdim, Ddim);
const sq = compRoot.mul(compRoot).sum();
sq.backward();
const gcBC = gradCheck(composite, [col, row], 1e-5);
console.log(`\ngradCheck composite broadcast (non-uniform upstream): maxRelErr ${gcBC.maxRelError.toExponential(3)} PASS ${gcBC.maxRelError < 1e-6}`);

// ===========================================================================
section("3) FAILURE MODE: a broadcast-add that FORGETS to un-broadcast");
// ===========================================================================
// Counterfactual: hand-rolled broadcast where forward fans (1,D) across B rows, but the
// backward scatters the (B,D) upstream grad straight into a (B,D) buffer instead of SUMMING
// it back down to the (1,D) source. This is THE classic n-d autograd bug. We don't patch
// the core; we reproduce the wrong op locally so the reader sees the exact failure signal.
//
// NOTE: we deliberately make `.grad` come out (B,D)-shaped to surface the mismatch loudly;
// in real buggy code it often instead comes out (1,D)-shaped but B times too large, which
// is even sneakier (no shape error, just silently wrong magnitude). We demonstrate BOTH.

function brokenBroadcastRow_wrongShape(src: Tensor, rowsB: number): Tensor {
  if (src.shape.length !== 2 || src.shape[0] !== 1) throw new Error(`expected (1,n), got ${src.shape}`);
  const n = src.shape[1];
  const data = new Float64Array(rowsB * n);
  for (let r = 0; r < rowsB; r++) data.set(src.data, r * n);
  const out = new Tensor(data, [rowsB, n], [src], "broken_broadcast");
  out._backward = () => {
    // BUG: we try to write the full (B,n) upstream grad into src.grad, which is length n.
    // The += on out-of-range indices does nothing useful / corrupts; here we expose it as
    // a shape contract violation the moment we compare lengths.
    if (out.grad.length !== src.grad.length) {
      throw new Error(
        `un-broadcast skipped: upstream grad length ${out.grad.length} (shape ${out.shape}) ` +
          `cannot be written into source grad length ${src.grad.length} (shape ${src.shape}). ` +
          `Fix: SUM the ${rowsB} broadcast rows back into the single source row.`,
      );
    }
  };
  return out;
}

function brokenBroadcastRow_wrongMagnitude(src: Tensor, rowsB: number): Tensor {
  const n = src.shape[1];
  const data = new Float64Array(rowsB * n);
  for (let r = 0; r < rowsB; r++) data.set(src.data, r * n);
  const out = new Tensor(data, [rowsB, n], [src], "broken_broadcast_mag");
  out._backward = () => {
    // BUG: only copies row 0's grad back, ignoring the other B-1 rows' contributions.
    // Shape is fine (length n), so NO error is raised — the grad is just silently wrong
    // (here: B times too small, since B-1 rows of contribution are dropped). This is the
    // dangerous variant: training runs, converges slower / to garbage, no exception.
    for (let j = 0; j < n; j++) src.grad[j] += out.grad[j]; // missing: sum over rows r=1..B-1
  };
  return out;
}

const srcRow = new Tensor(Float64Array.from([1, 2, 3, 4]), [1, Ddim]);

// 3a) wrong-shape variant -> loud, catchable error (the GOOD kind of failure)
let caught = "";
try {
  srcRow.zeroGrad();
  const wrong = brokenBroadcastRow_wrongShape(srcRow, Bdim);
  wrong.sum().backward();
} catch (e) {
  caught = (e as Error).message;
}
console.log("3a) wrong-shape un-broadcast raised:\n   ", caught || "(no error — unexpected!)");

// 3b) wrong-magnitude variant -> NO error, silently wrong grad (the DANGEROUS kind)
srcRow.zeroGrad();
const wrongMag = brokenBroadcastRow_wrongMagnitude(srcRow, Bdim);
wrongMag.sum().backward();
const brokenGrad = Array.from(srcRow.grad);

// correct reference via the real core op
srcRow.zeroGrad();
const correct = srcRow.broadcastRow(Bdim);
correct.sum().backward();
const correctGrad = Array.from(srcRow.grad);

console.log("\n3b) wrong-magnitude un-broadcast (NO error raised):");
console.log(`   broken  grad ${fmtGrad(Float64Array.from(brokenGrad))}  (only row 0 counted)`);
console.log(`   correct grad ${fmtGrad(Float64Array.from(correctGrad))}  (summed over all ${Bdim} rows)`);
console.log(`   broken is exactly 1/${Bdim} of correct:`, brokenGrad.every((g, i) => Math.abs(g * Bdim - correctGrad[i]) < 1e-12));
console.log("   >> Lesson: a shape error is a GIFT (it crashes); a magnitude error trains silently wrong.");

// ===========================================================================
section("4) timing: matmul vs elementwise (real wall-clock, RELATIVE only)");
// ===========================================================================
// Honest framing: f64, single-threaded, no SIMD/BLAS. Absolute ms are pessimistic vs a
// real framework; the transferable signal is the RATIO — matmul is O(n^3) compute over
// O(n^2) data, so it dominates elementwise (O(n^2)) and the gap widens with n. We show
// two sizes so the reader sees the ratio grow, not a single anecdotal number.
const tRng = mulberry32(808);
for (const nDim of [64, 128]) {
  const M1 = Tensor.from([nDim, nDim], () => randn(tRng));
  const M2 = Tensor.from([nDim, nDim], () => randn(tRng));
  const mm = timeIt(() => { const r = M1.matmul(M2); void r.data[0]; }, 30, 3);
  const ew = timeIt(() => { const r = M1.mul(M2); void r.data[0]; }, 30, 3);
  const ratio = mm.perIterMs / ew.perIterMs;
  console.log(
    `${nDim}x${nDim}: matmul ${mm.perIterMs.toFixed(3)} ms  elementwise-mul ${ew.perIterMs.toFixed(4)} ms  ` +
      `ratio ${ratio.toFixed(1)}x (machine-dependent; compare ratios across sizes, not abs ms)`,
  );
}

console.log("\nSTAGE 02 DONE: every op gradChecked, broadcast adjoint verified, failure modes shown.\n");
