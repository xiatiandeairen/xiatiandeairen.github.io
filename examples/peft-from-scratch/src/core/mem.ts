// core/mem.ts — Memory estimator. PURE ARITHMETIC, allocates nothing.
//
// WHY estimate instead of measure: the dramatic PEFT memory story is about GPU VRAM during
//   training of billion-param models — which we cannot and should not allocate on a laptop.
//   So we compute the SAME accounting formula the real case obeys, applied to whatever param
//   counts a stage has. The shape of the bars (full-FT towers over PEFT) is the transferable
//   truth; the absolute MB at toy scale is meaningless and we say so.
//
// THE FORMULA (training-time, mixed-precision-agnostic, in element counts):
//   weights          = P                      (all params, frozen + trainable)
//   gradients        = P_trainable            (only trainable params get grads)
//   optimizer state  = optMultiplier * P_trainable   (Adam=2: m and v; SGD-momentum=1)
//   activations      ≈ actFactor * P          (rough; real value depends on batch/seq)
//   => the PEFT win lives in gradients + optimizer state shrinking with P_trainable, NOT
//      in weights (the frozen base is still resident). QLoRA additionally shrinks the
//      WEIGHTS term by quantizing the base — modeled via bytesPerBaseParam below.
//
// HONESTY: every number returned is an ESTIMATE. Callers must label outputs (est.) and the
//   book prints the toy-scale disclaimer. We never claim wall-clock VRAM here.

export interface MemBreakdown {
  weightsBytes: number;
  gradBytes: number;
  optimStateBytes: number;
  activationBytes: number;
  totalBytes: number;
}

export interface MemConfig {
  totalParams: number; // P
  trainableParams: number; // P_trainable
  bytesPerParam?: number; // grads/optim/trainable storage; default fp32 = 4
  bytesPerBaseParam?: number; // frozen base weight storage; QLoRA sets this < 4 (e.g. 0.5 for 4-bit)
  optMultiplier?: number; // Adam=2, SGD-momentum=1, plain SGD=0
  actFactor?: number; // activations ≈ actFactor * P (very rough)
}

/**
 * Estimate the training-time memory breakdown in BYTES. Nothing is allocated.
 * INVARIANT: frozen params contribute to weightsBytes only — never to grad/optim. That
 *   single asymmetry is the entire PEFT memory thesis, expressed in arithmetic.
 */
export function estBytes(cfg: MemConfig): MemBreakdown {
  const bpp = cfg.bytesPerParam ?? 4; // fp32
  const baseBpp = cfg.bytesPerBaseParam ?? bpp;
  const optMul = cfg.optMultiplier ?? 2; // Adam default
  const actFactor = cfg.actFactor ?? 1;
  const frozen = cfg.totalParams - cfg.trainableParams;

  const weightsBytes = frozen * baseBpp + cfg.trainableParams * bpp;
  const gradBytes = cfg.trainableParams * bpp; // only trainable params have grads
  const optimStateBytes = cfg.trainableParams * bpp * optMul;
  const activationBytes = cfg.totalParams * actFactor * bpp; // forward needs full base activations
  const totalBytes = weightsBytes + gradBytes + optimStateBytes + activationBytes;
  return { weightsBytes, gradBytes, optimStateBytes, activationBytes, totalBytes };
}

/** Human-readable MB with 2 decimals. Convenience for printing bars. */
export function toMB(bytes: number): number {
  return bytes / (1024 * 1024);
}
