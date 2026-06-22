// core/nn.ts — Module base + the conv-net layers the whole book is assembled from.
//
// WHY a Module base at all: training needs three cross-cutting operations on a tree of
//   layers — collect all trainable Tensors (for the optimizer), zero their grads, and run
//   forward. Encoding these as a base contract means the tiny-CNN is "just"
//   Sequential(Conv2d, ReLU, MaxPool2d, ..., Flatten, Linear) and the training loop never
//   special-cases any layer.
//
// INVARIANT: parameters() returns LEAF tensors only (the actual learnables), recursively.
//   A tensor produced by an op (it has _prev) is NOT a parameter — it's recomputed each
//   forward. Returning intermediates would make the optimizer "update" throwaway nodes.
//
// FAILURE MODE this guards: a sub-module stored in a plain field that parameters() forgets
//   to recurse into => its weights never get grads applied => that layer silently never
//   trains. We make registration explicit via _modules / _params to avoid relying on
//   reflection over arbitrary fields.

import { Tensor, conv2d, maxpool2d, avgpool2d, type Conv2dParams } from "./autograd.js";
import { kaiming, xavier, type Rng } from "./rng.js";
import { noGradActive } from "./autograd.js";

export abstract class Module {
  protected _params: Tensor[] = [];
  protected _modules: Module[] = [];
  training = true;

  /** Register a leaf parameter and return it for convenient assignment. */
  protected param(t: Tensor): Tensor {
    this._params.push(t);
    return t;
  }

  /** Register a child module and return it. */
  protected child<M extends Module>(m: M): M {
    this._modules.push(m);
    return m;
  }

  /** Recursively collect all leaf parameters. Stable order (this layer's params, then
   *  children in registration order) is what lets Adam's per-param state stay aligned. */
  parameters(): Tensor[] {
    const out = [...this._params];
    for (const m of this._modules) out.push(...m.parameters());
    return out;
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }

  /** Propagate train/eval mode down the tree. BatchNorm reads this to pick its stat path. */
  setTraining(mode: boolean): void {
    this.training = mode;
    for (const m of this._modules) m.setTraining(mode);
  }

  abstract forward(x: Tensor): Tensor;
}

// ----------------------------------------------------------------------------
// Conv2d: NCHW input -> NCHW output. weight OIHW, optional per-output-channel bias.
// ----------------------------------------------------------------------------
export class Conv2d extends Module {
  weight: Tensor; // (outC, inC, kH, kW)
  bias: Tensor | null;
  params: Conv2dParams;

  constructor(
    inC: number,
    outC: number,
    kernel: number,
    rng: Rng,
    opts: { stride?: number; padding?: number; bias?: boolean } = {},
  ) {
    super();
    const { stride = 1, padding = 0, bias = true } = opts;
    this.params = { stride, padding };
    // Kaiming fan-in for conv = inC*kH*kW (the count of weights feeding one output unit).
    // Using just inC here is a classic init bug: it ignores the kernel area and over-scales
    // the weights, making early activations explode for large kernels.
    const fanIn = inC * kernel * kernel;
    this.weight = this.param(Tensor.from([outC, inC, kernel, kernel], () => kaiming(fanIn, rng)));
    this.bias = bias ? this.param(Tensor.zeros([outC])) : null;
  }

  override forward(x: Tensor): Tensor {
    return conv2d(x, this.weight, this.bias, this.params);
  }
}

// ----------------------------------------------------------------------------
// Pooling wrappers — thin Modules over the autograd ops (so they fit Sequential).
// ----------------------------------------------------------------------------
export class MaxPool2d extends Module {
  constructor(private kernel: number, private stride = kernel) {
    super();
  }
  override forward(x: Tensor): Tensor {
    return maxpool2d(x, { kernel: this.kernel, stride: this.stride });
  }
}

export class AvgPool2d extends Module {
  constructor(private kernel: number, private stride = kernel) {
    super();
  }
  override forward(x: Tensor): Tensor {
    return avgpool2d(x, { kernel: this.kernel, stride: this.stride });
  }
}

// ----------------------------------------------------------------------------
// BatchNorm2d: per-channel normalization over the (N, H, W) population.
// ----------------------------------------------------------------------------
export class BatchNorm2d extends Module {
  gamma: Tensor; // (C,) learnable scale, init 1
  beta: Tensor; // (C,) learnable shift, init 0
  runningMean: Float64Array; // (C,) inference statistics, NOT a learnable param
  runningVar: Float64Array; // (C,)
  channels: number;
  eps: number;
  momentum: number;

  constructor(channels: number, opts: { eps?: number; momentum?: number } = {}) {
    super();
    this.channels = channels;
    this.eps = opts.eps ?? 1e-5;
    this.momentum = opts.momentum ?? 0.1;
    this.gamma = this.param(Tensor.fill([channels], 1));
    this.beta = this.param(Tensor.zeros([channels]));
    // running stats are buffers, NOT params: they are updated by an EMA of batch stats, not
    // by the optimizer. Registering them as params would let SGD corrupt them. Init to the
    // identity transform (mean 0, var 1) so an untrained-but-eval'd net is at least sane.
    this.runningMean = new Float64Array(channels);
    this.runningVar = new Float64Array(channels).fill(1);
  }

  /**
   * WHY a hand-written fused backward: BatchNorm's adjoint couples every element sharing a
   *   channel (the per-channel mean/var depend on all of them). Composing sub/mean/div
   *   ops would be correct but obscures the gradient the book wants to teach.
   * TRAIN vs EVAL (the canonical BN bug): in training we normalize by the CURRENT BATCH's
   *   mean/var and update the running EMA; in eval we use the FROZEN running stats and skip
   *   the EMA update. Using batch stats at eval makes single-image inference depend on
   *   whatever else is in the batch — non-deterministic predictions. We branch on
   *   training && !noGradActive() so noGrad() inference always takes the eval path.
   */
  override forward(x: Tensor): Tensor {
    if (x.shape.length !== 4 || x.shape[1] !== this.channels)
      throw new Error(`BatchNorm2d: expected (N,${this.channels},H,W), got ${x.shape}`);
    const [N, C, H, W] = x.shape;
    const perChan = N * H * W; // population size per channel
    const useBatch = this.training && !noGradActive();

    const mean = new Float64Array(C);
    const invStd = new Float64Array(C);

    if (useBatch) {
      for (let c = 0; c < C; c++) {
        let m = 0;
        for (let n = 0; n < N; n++)
          for (let i = 0; i < H * W; i++) m += x.data[((n * C + c) * H * W) + i];
        m /= perChan;
        let v = 0;
        for (let n = 0; n < N; n++)
          for (let i = 0; i < H * W; i++) {
            const d = x.data[((n * C + c) * H * W) + i] - m;
            v += d * d;
          }
        v /= perChan;
        mean[c] = m;
        invStd[c] = 1 / Math.sqrt(v + this.eps);
        // EMA update of running stats for inference. Mutating buffers here (a side effect in
        // forward) is the standard BN design; it's why BN forward is NOT a pure function.
        this.runningMean[c] = (1 - this.momentum) * this.runningMean[c] + this.momentum * m;
        this.runningVar[c] = (1 - this.momentum) * this.runningVar[c] + this.momentum * v;
      }
    } else {
      for (let c = 0; c < C; c++) {
        mean[c] = this.runningMean[c];
        invStd[c] = 1 / Math.sqrt(this.runningVar[c] + this.eps);
      }
    }

    const out = new Float64Array(x.size);
    const xhat = new Float64Array(x.size);
    for (let n = 0; n < N; n++)
      for (let c = 0; c < C; c++) {
        const g = this.gamma.data[c];
        const b = this.beta.data[c];
        for (let i = 0; i < H * W; i++) {
          const idx = ((n * C + c) * H * W) + i;
          const xh = (x.data[idx] - mean[c]) * invStd[c];
          xhat[idx] = xh;
          out[idx] = xh * g + b;
        }
      }

    const t = new Tensor(out, x.shape, [x, this.gamma, this.beta], "batchnorm2d");
    t._backward = () => {
      // Per-channel BN backward. For each channel, the standard reduction:
      //   dx = (gamma * invStd / M) * (M*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
      // where M = perChan and dxhat = dout. At eval (frozen stats) mean/var are constants,
      // so the correct adjoint is just dx = dout*gamma*invStd; we keep the train form here
      // because backward is only ever driven in training (eval runs under noGrad).
      const M = perChan;
      for (let c = 0; c < C; c++) {
        const inv = invStd[c];
        const gscale = this.gamma.data[c];
        let sumDxhat = 0;
        let sumDxhatXhat = 0;
        for (let n = 0; n < N; n++)
          for (let i = 0; i < H * W; i++) {
            const idx = ((n * C + c) * H * W) + i;
            const dxhat = t.grad[idx] * gscale;
            sumDxhat += dxhat;
            sumDxhatXhat += dxhat * xhat[idx];
            // gamma/beta grads are summed over the whole channel population.
            this.gamma.grad[c] += t.grad[idx] * xhat[idx];
            this.beta.grad[c] += t.grad[idx];
          }
        for (let n = 0; n < N; n++)
          for (let i = 0; i < H * W; i++) {
            const idx = ((n * C + c) * H * W) + i;
            const dxhat = t.grad[idx] * gscale;
            x.grad[idx] += (inv / M) * (M * dxhat - sumDxhat - xhat[idx] * sumDxhatXhat);
          }
      }
    };
    return t;
  }
}

// ----------------------------------------------------------------------------
// Linear: y = x @ W + b , x:(batch, inF) W:(inF, outF) b:(1, outF). The classifier head.
// ----------------------------------------------------------------------------
export class Linear extends Module {
  W: Tensor;
  b: Tensor | null;

  constructor(inF: number, outF: number, rng: Rng, opts: { bias?: boolean; init?: "kaiming" | "xavier" } = {}) {
    super();
    const { bias = true, init = "kaiming" } = opts;
    const gen = init === "kaiming" ? () => kaiming(inF, rng) : () => xavier(inF, outF, rng);
    this.W = this.param(Tensor.from([inF, outF], gen));
    this.b = bias ? this.param(Tensor.zeros([1, outF])) : null; // bias starts at 0
  }

  override forward(x: Tensor): Tensor {
    const out = x.matmul(this.W);
    if (!this.b) return out;
    return out.add(this.b.broadcastRow(out.shape[0]));
  }
}

// ----------------------------------------------------------------------------
// ReLU / Flatten — stateless Modules so they slot into Sequential.
// ----------------------------------------------------------------------------
export class ReLU extends Module {
  override forward(x: Tensor): Tensor {
    return x.relu();
  }
}

export class Flatten extends Module {
  override forward(x: Tensor): Tensor {
    return x.flatten();
  }
}

// ----------------------------------------------------------------------------
// Sequential: run modules in order. The composition primitive for everything.
// ----------------------------------------------------------------------------
export class Sequential extends Module {
  constructor(mods: Module[]) {
    super();
    for (const m of mods) this.child(m);
  }
  override forward(x: Tensor): Tensor {
    let h = x;
    for (const m of this._modules) h = m.forward(h);
    return h;
  }
}

// ----------------------------------------------------------------------------
// ResidualBlock: out = relu(F(x) + shortcut(x)).
// ----------------------------------------------------------------------------
//
// WHY this is its own class and not just Sequential: the defining feature is the SKIP — the
//   block's input is added back to its branch output. That addition is what gives gradients
//   a direct path around F (the identity term has gradient 1), which is the whole reason
//   residual nets train deep without vanishing grads. Stage06 demonstrates the convergence
//   gap vs the same net without the skip.
//
// SHAPE INVARIANT: F(x) and shortcut(x) MUST have identical shape to add. When the branch
//   changes channels/spatial size, the caller passes a `shortcut` module (typically a 1x1
//   Conv2d) to match; otherwise shortcut defaults to identity and F must be shape-preserving.
export class ResidualBlock extends Module {
  constructor(private branch: Module, private shortcut: Module | null = null) {
    super();
    this.child(branch);
    if (shortcut) this.child(shortcut);
  }

  override forward(x: Tensor): Tensor {
    const f = this.branch.forward(x);
    const s = this.shortcut ? this.shortcut.forward(x) : x;
    if (f.size !== s.size)
      throw new Error(
        `ResidualBlock: branch output ${f.shape} != shortcut ${s.shape}; pass a projection shortcut.`,
      );
    return f.add(s).relu();
  }
}
