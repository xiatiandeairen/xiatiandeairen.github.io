// stage01-scalar-autograd.ts — Chapter 1 runnable: "a number that knows how to backprop".
//
// WHY this stage exists: before tensors, optimizers, attention, the GPT — there is ONE
//   idea, and it fits in a scalar. A Value is a number plus a closure that says how a
//   change in it ripples to a change in the final output (its gradient). Reverse-mode
//   autodiff is just: build the expression graph, then walk it backwards applying the
//   chain rule. Everything later (Tensor in chapter 2+) is the same idea at n-d scale.
//
// WHAT THIS FILE PROVES (with real, reproducible numbers, all offline, no LLM/network):
//   1. Analytic gradients from backward() match numerical central-difference gradients
//      to < 1e-6 on a real 2-input neuron y = tanh(w1*x1 + w2*x2 + b).
//   2. The CORE INVARIANT of reverse-mode (grads ACCUMULATE, +=, never overwrite, =)
//      is not a stylistic choice: we re-run the SAME graph with an overwriting backward
//      and print the exact wrong number it produces on a node used twice (x*x). This is
//      the single most common autograd bug, demonstrated quantitatively, not asserted.
//
// We re-implement a tiny Value here (NOT importing core/autograd's Value) ONLY so the
//   overwrite-vs-accumulate failure demo can swap the backward rule. The core Value is
//   frozen-correct and gives no hook to inject the bug; teaching the bug requires a
//   class we control. Identifiers/numbers are validated against core's design in §1.
//
// DETERMINISM: fixed seed via core's mulberry32 -> randn for the neuron's inputs/params,
//   so every run prints byte-identical numbers (same machine).

import { mulberry32, randn } from "./core/rng.js";

// ============================================================================
// A scalar reverse-mode autodiff node. Mirrors core/autograd.ts Value, but with
// one knob: `accumulate` controls whether _backward does += (correct) or = (the bug).
// ============================================================================

class Value {
  data: number;
  grad: number;
  // _backward: scatter THIS node's grad into its parents. No-op for leaves.
  _backward: () => void;
  // _prev: parents, for topo discovery. Set dedupes x-used-twice (x*x).
  _prev: Set<Value>;
  op: string;

  // WHY a static flag instead of per-node: the failure demo must flip the rule for an
  // ENTIRE graph at once (every node's adjoint), then flip back. A per-instance flag
  // would force threading it through every op. Static keeps the two ops below honest:
  // they read one switch. Restored in finally by withAccumulate() so it can't leak.
  static accumulate = true;

  constructor(data: number, prev: Value[] = [], op = "") {
    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = new Set(prev);
    this.op = op;
  }

  // bump: the ONLY place the +=/= decision lives. += sums contributions from every
  // path to this node (required when a node fans out to multiple consumers); = keeps
  // only the last path written and silently drops the rest — that is the bug.
  private static bump(node: Value, delta: number): void {
    if (Value.accumulate) node.grad += delta;
    else node.grad = delta;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o], "+");
    // d(out)/d(self) = d(out)/d(o) = 1, scaled by upstream out.grad.
    out._backward = () => {
      Value.bump(this, out.grad);
      Value.bump(o, out.grad);
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, [this, o], "*");
    out._backward = () => {
      Value.bump(this, o.data * out.grad);
      Value.bump(o, this.data * out.grad);
    };
    return out;
  }

  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.mul(-1));
  }

  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  pow(exp: number): Value {
    // Constant exponent only: variable exponent needs d/dexp = out*ln(self), a different op.
    const out = new Value(Math.pow(this.data, exp), [this], `**${exp}`);
    out._backward = () => {
      Value.bump(this, exp * Math.pow(this.data, exp - 1) * out.grad);
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], "tanh");
    out._backward = () => {
      Value.bump(this, (1 - t * t) * out.grad); // d/dx tanh = 1 - tanh^2
    };
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], "exp");
    out._backward = () => {
      Value.bump(this, e * out.grad);
    };
    return out;
  }

  // Reverse-topological backward. Build topo (parents before children), seed self.grad=1
  // (d(self)/d(self)=1), walk reversed so a node's grad is final before it scatters.
  // INVARIANT: caller zeroes grads first if reusing leaves (we build fresh graphs here,
  // so leaves start at grad=0 from the constructor).
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

// Run fn with the overwrite-bug backward rule, restoring accumulate even on throw so a
// failure in the demo can't poison later experiments.
function withOverwrite<T>(fn: () => T): T {
  const prev = Value.accumulate;
  Value.accumulate = false;
  try {
    return fn();
  } finally {
    Value.accumulate = prev;
  }
}

// ============================================================================
// Experiment 1: a real 2-input neuron, analytic grad vs numerical central difference.
// ============================================================================

// The neuron: y = tanh(w1*x1 + w2*x2 + b). Returns the output Value AND the leaves so
// the caller can read .grad after backward(). Leaves are passed in (not created here) so
// numerical differencing can perturb the SAME parameters the analytic pass sees.
function neuronForward(
  x1: Value,
  x2: Value,
  w1: Value,
  w2: Value,
  b: Value,
): Value {
  return x1.mul(w1).add(x2.mul(w2)).add(b).tanh();
}

// Plain-number version of the same neuron, for numerical differencing. Keeping a pure
// f(params)->number lets us perturb one input by +-eps without touching the graph.
function neuronEval(
  x1: number,
  x2: number,
  w1: number,
  w2: number,
  b: number,
): number {
  return Math.tanh(x1 * w1 + x2 * w2 + b);
}

// Central difference: (f(p+eps) - f(p-eps)) / 2eps. WHY central not forward-difference:
// forward diff error is O(eps); central is O(eps^2), so at eps=1e-5 central lands near
// f64's noise floor and lets the < 1e-6 claim hold. Forward diff would falsely "fail".
function numericalGrad(
  evalAt: (...p: number[]) => number,
  params: number[],
  index: number,
  eps = 1e-5,
): number {
  const plus = params.slice();
  const minus = params.slice();
  plus[index] += eps;
  minus[index] -= eps;
  return (evalAt(...plus) - evalAt(...minus)) / (2 * eps);
}

function relErr(a: number, b: number): number {
  // Relative error with an absolute-floor denominator so a true-zero grad doesn't divide
  // by ~0 and report a spurious huge error. 1e-12 is well below f64 grad magnitudes here.
  return Math.abs(a - b) / Math.max(Math.abs(a) + Math.abs(b), 1e-12);
}

function runGradCheck(): void {
  console.log("=".repeat(60));
  console.log("1) 2-input neuron: analytic grad vs numerical central diff");
  console.log("=".repeat(60));

  // Deterministic inputs/params from a seeded stream (no Math.random anywhere).
  const rng = mulberry32(1);
  const x1n = randn(rng);
  const x2n = randn(rng);
  const w1n = randn(rng);
  const w2n = randn(rng);
  const bn = randn(rng);

  // Build the graph and backprop once.
  const x1 = new Value(x1n);
  const x2 = new Value(x2n);
  const w1 = new Value(w1n);
  const w2 = new Value(w2n);
  const b = new Value(bn);
  const y = neuronForward(x1, x2, w1, w2, b);
  y.backward();

  console.log(
    `inputs:  x1=${x1n.toFixed(4)}  x2=${x2n.toFixed(4)}` +
      `   params: w1=${w1n.toFixed(4)}  w2=${w2n.toFixed(4)}  b=${bn.toFixed(4)}`,
  );
  console.log(`forward: y = ${y.data.toFixed(6)}`);

  // Numerical grad wrt each of the 5 leaves; param order matches neuronEval signature.
  const params = [x1n, x2n, w1n, w2n, bn];
  const names = ["x1", "x2", "w1", "w2", "b"];
  const analytic = [x1.grad, x2.grad, w1.grad, w2.grad, b.grad];

  let maxRel = 0;
  console.log("  param   analytic        numerical       rel.error");
  for (let i = 0; i < params.length; i++) {
    const num = numericalGrad(neuronEval, params, i);
    const rel = relErr(analytic[i], num);
    maxRel = Math.max(maxRel, rel);
    console.log(
      `  ${names[i].padEnd(6)}` +
        `${analytic[i].toExponential(6).padEnd(16)}` +
        `${num.toExponential(6).padEnd(16)}` +
        `${rel.toExponential(3)}`,
    );
  }
  console.log(`max relative error: ${maxRel.toExponential(3)}`);
  console.log(`PASS (< 1e-6): ${maxRel < 1e-6}`);
}

// ============================================================================
// Experiment 2: FAILURE MODE — overwrite (=) vs accumulate (+=) on a shared node.
// ============================================================================
//
// Setup: f = x*x with a SINGLE Value x feeding BOTH factors. df/dx = 2x. Reverse-mode
// reaches x along TWO paths (left factor and right factor); each path contributes x to
// the grad, and they must SUM to 2x. With overwrite, the second path clobbers the first,
// so x.grad ends at x (one path) instead of 2x. We print both numbers and the gap.
function runAccumulateVsOverwrite(): void {
  console.log("=".repeat(60));
  console.log("2) FAILURE MODE: overwrite (=) vs accumulate (+=) on shared node x in f=x*x");
  console.log("=".repeat(60));

  const rng = mulberry32(7);
  const xVal = randn(rng);
  const expected = 2 * xVal; // analytic df/dx for f = x^2

  // Correct: accumulate.
  const xa = new Value(xVal);
  xa.mul(xa).backward(); // x*x, x shared
  const accumulatedGrad = xa.grad;

  // Buggy: overwrite. SAME graph shape, only the backward rule changes.
  const xb = new Value(xVal);
  withOverwrite(() => {
    xb.mul(xb).backward();
  });
  const overwrittenGrad = xb.grad;

  console.log(`x = ${xVal.toFixed(6)}   (f = x*x, true df/dx = 2x = ${expected.toFixed(6)})`);
  console.log(`accumulate (+=): x.grad = ${accumulatedGrad.toFixed(6)}   -> matches 2x: ${
    Math.abs(accumulatedGrad - expected) < 1e-12
  }`);
  console.log(`overwrite  (=) : x.grad = ${overwrittenGrad.toFixed(6)}   -> wrong by ${
    Math.abs(overwrittenGrad - expected).toFixed(6)
  } (only 1 of 2 paths counted)`);
  console.log(
    `ratio accumulate/overwrite = ${(accumulatedGrad / overwrittenGrad).toFixed(4)}` +
      ` (exactly 2x: the dropped path)`,
  );
  console.log(
    ">> Lesson: a value reused N times fans grad into N paths; reverse-mode MUST sum them.",
  );
  console.log(
    ">> Overwrite silently halves (here) the grad — training would crawl or diverge, no crash.",
  );
}

// ============================================================================
// main — runnable per the no-import-stage rule. Pure, offline, deterministic.
// ============================================================================
function main(): void {
  runGradCheck();
  runAccumulateVsOverwrite();
}

main();
