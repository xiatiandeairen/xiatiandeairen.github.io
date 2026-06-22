// core/nn.ts — Minimal layers + the book's denoiser shape.
//
// WHY this exists: every diffusion stage trains the SAME network shape —
//   [ x(2D) ⊕ time-embed ⊕ optional cond ] -> MLP -> predict 2D (the noise ε, or the score).
//   Putting that shape here once means a stage is "make the net, train it, sample" with no
//   re-derivation. The layers are thin wrappers over core/tensor's autograd ops; the
//   gradient that updates these weights is the real chain rule from tensor.ts, nothing hidden.
//
// INVARIANT: Module.parameters() returns EVERY trainable Tensor exactly once. The optimizer
//   and zeroGrad walk this list — a parameter missing from it silently never learns, and a
//   parameter listed twice gets its grad double-counted. Both are classic, invisible bugs.

import { RNG } from "./rng.js";
import { Tensor } from "./tensor.js";

export interface Module {
  /** All trainable tensors, each exactly once. */
  parameters(): Tensor[];
}

export type Activation = "relu" | "tanh" | "silu";

function applyActivation(t: Tensor, act: Activation): Tensor {
  switch (act) {
    case "relu":
      return t.relu();
    case "tanh":
      return t.tanh();
    case "silu":
      return t.silu();
    default: {
      const _exhaustive: never = act; // adding an Activation without a case fails to compile
      throw new Error(`unknown activation: ${_exhaustive}`);
    }
  }
}

/**
 * Fully-connected layer: y = x @ W + b, x is [batch, in], W is [in, out], b is [1, out].
 * Init: Xavier/Glorot N(0, 2/(in+out)) on W, zeros on b.
 * WHY Xavier (not Kaiming): the book's nets use tanh/silu, symmetric-ish activations where
 *   Xavier keeps both forward-activation and backward-grad variance ~constant. Kaiming's 2×
 *   factor is tuned for ReLU's half-zeroing; using it here over-scales tanh nets.
 * INVARIANT: W is drawn from the RNG BEFORE b — stages depend on this draw order for
 *   byte-identical init across runs.
 */
export class Linear implements Module {
  W: Tensor;
  b: Tensor;

  constructor(inFeatures: number, outFeatures: number, rng: RNG) {
    const std = Math.sqrt(2 / (inFeatures + outFeatures));
    this.W = Tensor.from([inFeatures, outFeatures], () => rng.gaussian() * std);
    this.b = Tensor.zeros([1, outFeatures]); // bias starts at 0 (no symmetry to break)
  }

  forward(x: Tensor): Tensor {
    // b is [1, out], broadcast across the batch rows via elementwise tiling in tensor.add.
    return x.matmul(this.W).add(this.b);
  }

  parameters(): Tensor[] {
    return [this.W, this.b];
  }
}

/**
 * MLP: a stack of Linear layers with `act` between them (no activation after the last —
 * the output is a raw 2-D prediction, not a probability). sizes = [in, h1, h2, ..., out].
 */
export class MLP implements Module {
  layers: Linear[];
  private act: Activation;

  constructor(sizes: number[], act: Activation, rng: RNG) {
    if (sizes.length < 2) throw new Error(`MLP: need >= 2 sizes (in,out), got ${sizes}`);
    this.act = act;
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      this.layers.push(new Linear(sizes[i], sizes[i + 1], rng));
    }
  }

  forward(x: Tensor): Tensor {
    let h = x;
    for (let i = 0; i < this.layers.length; i++) {
      h = this.layers[i].forward(h);
      if (i < this.layers.length - 1) h = applyActivation(h, this.act); // no act on output
    }
    return h;
  }

  parameters(): Tensor[] {
    return this.layers.flatMap((l) => l.parameters());
  }
}

/**
 * Sinusoidal embedding of the diffusion timestep t (the Transformer positional-encoding
 * trick, reused by DDPM). WHY: the denoiser must behave very differently at t=1 (almost
 * clean) vs t=T (almost pure noise). Feeding the raw integer t as one feature gives the
 * net almost no resolution to distinguish nearby steps; sinusoids of many frequencies make
 * every t a distinct, smooth, high-dimensional vector the MLP can condition on.
 *
 * For dim = 2k, output[2i] = sin(t / 10000^(2i/dim)), output[2i+1] = cos(...). No trainable
 * params — it is a fixed, deterministic feature map (so it is NOT a Module).
 * INVARIANT: dim must be even (sin/cos come in pairs).
 */
export class SinusoidalEmbedding {
  readonly dim: number;
  private invFreq: Float64Array;

  constructor(dim: number) {
    if (dim % 2 !== 0) throw new Error(`SinusoidalEmbedding: dim must be even, got ${dim}`);
    this.dim = dim;
    const half = dim / 2;
    this.invFreq = new Float64Array(half);
    for (let i = 0; i < half; i++) this.invFreq[i] = 1 / Math.pow(10000, (2 * i) / dim);
  }

  /** Embed a batch of timesteps (one t per row) into a [batch, dim] Tensor. This is a leaf
   *  tensor (no parents) — gradient stops here, which is correct: t is an input, not a param. */
  forward(timesteps: number[]): Tensor {
    const batch = timesteps.length;
    const half = this.dim / 2;
    const out = new Float64Array(batch * this.dim);
    for (let r = 0; r < batch; r++) {
      for (let i = 0; i < half; i++) {
        const arg = timesteps[r] * this.invFreq[i];
        out[r * this.dim + 2 * i] = Math.sin(arg);
        out[r * this.dim + 2 * i + 1] = Math.cos(arg);
      }
    }
    return new Tensor(out, [batch, this.dim]);
  }
}
