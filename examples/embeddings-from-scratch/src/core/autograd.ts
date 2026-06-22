// core/autograd.ts — a scalar reverse-mode autodiff engine, ~Karpathy-micrograd
// in spirit, rewritten with explicit invariants for this book.
//
// Why build our own instead of importing a tensor lib: the thesis of the book is
// that an embedding is just "a parameter trained by gradient descent on a loss".
// A reader must be able to SEE the gradient flow — not trust a black-box .backward().
// So every operation here is a scalar Value node that records (a) its data, (b)
// its parents, and (c) a closure that knows how to push its grad to those parents.
// backward() then runs the chain rule by topological order. Nothing magic.
//
// Performance note (honest): scalar autograd is O(edges) with one closure call
// per edge and heap-allocated nodes everywhere. It is *thousands* of times slower
// than a real tensor lib. That is fine and intentional at toy vocab/dim scale
// (a few seconds per stage). The book states explicitly that absolute throughput
// here is pedagogical, not production; what transfers is the *mechanism* and the
// *shape* of the loss curve, not the wall-clock.
//
// Key invariants:
//  - grad accumulates (+=), it does not overwrite. A node used by two parents
//    must sum both incoming gradients (the multivariate chain rule). Forgetting
//    this is the #1 autodiff bug; we accumulate everywhere.
//  - backward() seeds the root grad to 1.0 and must be called on a SCALAR loss.
//    Calling it on a non-loss node is meaningless; the seed-1 convention assumes
//    d(self)/d(self) = 1.
//  - You MUST zero grads between optimizer steps (see optim.zeroGrad), otherwise
//    accumulation silently sums gradients across steps and training diverges.

export class Value {
  data: number;
  grad: number;
  // Parents in the compute graph. Empty for leaves (parameters / constants).
  private prev: Value[];
  // Local backward: given this node's grad already set, push contributions to
  // each parent's grad. No-op for leaves.
  private backwardFn: () => void;

  constructor(data: number, prev: Value[] = [], backwardFn: () => void = () => {}) {
    this.data = data;
    this.grad = 0;
    this.prev = prev;
    this.backwardFn = backwardFn;
  }

  add(other: Value | number): Value {
    const o = toValue(other);
    const out = new Value(this.data + o.data, [this, o]);
    // d(a+b)/da = 1, d/db = 1 → just route grad straight through to both.
    out.backwardFn = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = toValue(other);
    const out = new Value(this.data * o.data, [this, o]);
    // d(a*b)/da = b, d/db = a. The classic product rule; note we read the *other*
    // operand's data, captured by closure at forward time.
    out.backwardFn = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  sub(other: Value | number): Value {
    return this.add(toValue(other).mul(-1));
  }

  // True division via b^-1, so we get correct grad for free from pow + mul.
  div(other: Value | number): Value {
    return this.mul(toValue(other).pow(-1));
  }

  // Power by a CONSTANT exponent only. Variable exponents (a^b) need log of a,
  // which is undefined for a<=0 and rarely needed in embedding training, so we
  // deliberately restrict the API to the case that is always safe to differentiate.
  pow(exp: number): Value {
    const out = new Value(Math.pow(this.data, exp), [this]);
    // d(a^n)/da = n * a^(n-1).
    out.backwardFn = () => {
      this.grad += exp * Math.pow(this.data, exp - 1) * out.grad;
    };
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this]);
    // d(e^x)/dx = e^x = out.data. Reusing the forward value avoids recompute and
    // keeps fwd/bwd numerically consistent.
    out.backwardFn = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  // Natural log. Domain guard: log(<=0) is the most common NaN source in
  // contrastive losses (log of a softmax prob that underflowed to 0). We clamp
  // the *input used for the value and the derivative* to a tiny epsilon and rely
  // on the upstream design (stable softmax) to keep us in-domain; the clamp is a
  // last-resort guard, not a license to feed it zeros.
  log(): Value {
    const x = this.data < 1e-12 ? 1e-12 : this.data;
    const out = new Value(Math.log(x), [this]);
    // d(ln x)/dx = 1/x.
    out.backwardFn = () => {
      this.grad += (1 / x) * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this]);
    // d(tanh)/dx = 1 - tanh^2. Bounded grad → why tanh is a gentle nonlinearity.
    out.backwardFn = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data > 0 ? this.data : 0, [this]);
    // Subgradient at 0 chosen as 0. ReLU's dead-unit failure mode (grad 0 for all
    // negative pre-activations → unit never recovers) is a teaching point in the
    // book; this is exactly where it originates.
    out.backwardFn = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  // Reverse-mode backward over the whole graph rooted at `this`.
  //
  // Step 1: build a topological order so every node is processed only after all
  // nodes that depend on it. Step 2: seed this node's grad = 1. Step 3: walk in
  // reverse topo order calling each node's local backwardFn, which += into parents.
  //
  // Why topo order is non-negotiable: if we called backwardFn before a node's
  // grad was fully accumulated from ALL its children, we'd push an incomplete
  // gradient to parents — the silent-wrong-answer failure. Topo guarantees a
  // node's grad is final before it propagates.
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v.prev) build(child);
      topo.push(v);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i].backwardFn();
  }
}

// Coerce a raw number into a constant Value (leaf, no grad flows past it because
// nobody holds a reference to optimize it). Lets ops accept `x.mul(2)`.
function toValue(x: Value | number): Value {
  return x instanceof Value ? x : new Value(x);
}

// Sum a list of Values into one. Used constantly (dot products, loss reductions).
// We fold with .add so the graph is a left-leaning chain — fine for toy sizes.
export function sumValues(xs: Value[]): Value {
  let acc = new Value(0);
  for (const x of xs) acc = acc.add(x);
  return acc;
}

// ---------------------------------------------------------------------------
// Thin Vec / Mat shells over Value, so word-vector code reads like linear algebra
// instead of scalar spaghetti. These do NOT introduce a new tensor type — they
// are just typed aliases plus a handful of helpers that build Value graphs.
// ---------------------------------------------------------------------------

// A vector is a row of trainable scalars. Each entry is its own Value leaf, which
// is why a D=16 embedding for a V=40 vocab is 640 live nodes — cheap at toy scale.
export type Vec = Value[];
export type Mat = Value[][];

// dot(a,b) as a Value graph. This is THE operation embeddings live in: similarity
// is a dot product, and skip-gram / InfoNCE scores are dots between vectors. We
// build it from sumValues(mul) so its gradient (∂/∂a_i = b_i, ∂/∂b_i = a_i) comes
// straight from the engine — no hand-derived special case to get wrong.
export function dot(a: Vec, b: Vec): Value {
  if (a.length !== b.length) throw new Error(`dot: dim mismatch ${a.length} vs ${b.length}`);
  const terms: Value[] = [];
  for (let i = 0; i < a.length; i++) terms.push(a[i].mul(b[i]));
  return sumValues(terms);
}

// Allocate a fresh parameter vector of `dim` Value leaves from an init function.
// Separating allocation from init (the caller passes the init fn, usually a
// gaussian closure over an Rng) keeps determinism explicit at the call site.
export function makeVec(dim: number, init: () => number): Vec {
  const v: Vec = [];
  for (let i = 0; i < dim; i++) v.push(new Value(init()));
  return v;
}

// Allocate rows x cols of parameters — e.g. an embedding matrix V x D.
export function makeMat(rows: number, cols: number, init: () => number): Mat {
  const m: Mat = [];
  for (let r = 0; r < rows; r++) m.push(makeVec(cols, init));
  return m;
}

// Read raw numbers out of a Vec (for similarity computed *outside* the graph,
// e.g. evaluation/nearest-neighbor where we don't need gradients). Detaching to
// number[] here is intentional: eval must not accidentally extend the live graph.
export function vecData(v: Vec): number[] {
  return v.map((x) => x.data);
}
