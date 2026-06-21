// core/probability.ts — the probability math under every policy in the book.
//
// Everything downstream (REINFORCE, PPO, RLHF, DPO, GRPO) is, mechanically, an
// operation on log π(a|s). Get these primitives numerically right once and the
// whole stack is stable; get softmax wrong (overflow) and stage 03 "diverges"
// for reasons that have nothing to do with RL. So this file is deliberately
// paranoid about numerical stability and explicit about WHICH KL estimator is
// being used, because RLHF papers conflate three of them.

// softmax: logits -> probability distribution. The max-subtraction is not an
// optimization, it is correctness: exp(1000) is +Infinity, and Infinity/Infinity
// is NaN. Subtracting max(logits) leaves the distribution identical (softmax is
// shift-invariant) but keeps every exp argument <= 0, hence in (0, 1].
//
// temperature: divides logits before exp. T -> 0 makes it argmax (greedy,
// exploitation); T large flattens toward uniform (exploration). This is the
// single knob that controls the explore/exploit trade-off for a softmax policy,
// so it threads through every sampling-based stage.
export function softmax(logits: number[], temperature = 1): number[] {
  if (temperature <= 0) {
    // T=0 is the limit "all mass on the argmax". We special-case it instead of
    // dividing by zero; a NaN here would silently break greedy evaluation.
    const m = Math.max(...logits);
    return logits.map((x) => (x === m ? 1 : 0));
  }
  const scaled = logits.map((x) => x / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// logSoftmax: log of softmax, computed WITHOUT going through softmax then log.
// Why separate: log(softmax) re-introduces the overflow we just dodged and adds
// log(0) = -Infinity for any starved action. The log-sum-exp form below is the
// numerically stable path and is what policy gradients actually consume — they
// need log π, never π directly, so this is the hotter primitive.
export function logSoftmax(logits: number[], temperature = 1): number[] {
  const scaled = temperature > 0 ? logits.map((x) => x / temperature) : logits.slice();
  const max = Math.max(...scaled);
  let sumExp = 0;
  for (const x of scaled) sumExp += Math.exp(x - max);
  const logSumExp = max + Math.log(sumExp);
  return scaled.map((x) => x - logSumExp);
}

// logProb: log π(action | state) for a single chosen action, from raw logits.
// This three-character function is the foundation the entire book stands on:
// the policy-gradient estimator is ∇ E[R] = E[R · ∇ log π(a|s)], the PPO ratio
// is exp(logProbNew - logProbOld), the DPO loss is a difference of logProbs.
// Implemented via logSoftmax (not log(softmax(...)[a])) for the stability above.
export function logProb(logits: number[], action: number, temperature = 1): number {
  return logSoftmax(logits, temperature)[action];
}

// entropy of a categorical distribution, in nats. Used as an exploration bonus
// (REINFORCE/PPO add +β·H to keep the policy from collapsing to a single action
// too early) and as a diagnostic: entropy crashing to ~0 early is the signature
// of premature convergence / mode collapse, a failure mode we demo explicitly.
export function entropy(probs: number[]): number {
  let h = 0;
  for (const p of probs) {
    // 0·log0 := 0 (continuous extension). Skipping zero-prob actions avoids
    // NaN from log(0) while leaving the sum correct.
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

// klCategorical: EXACT KL(p || q) = Σ p·log(p/q) when both full distributions
// are known. This is the ground-truth divergence. RLHF wants to keep the trained
// policy close to the reference policy; when we have both full softmaxes (toy
// setting) we can measure that drift exactly. Direction matters: KL is not
// symmetric. KL(p||q) penalizes p putting mass where q is near zero — the
// forward direction used as the RLHF reference penalty.
export function klCategorical(p: number[], q: number[]): number {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      // q[i] === 0 where p[i] > 0 means infinite KL: p explores where the
      // reference assigned zero probability. We surface it as Infinity rather
      // than clamp, because clamping would hide a genuinely catastrophic drift.
      kl += p[i] * Math.log(p[i] / q[i]);
    }
  }
  return kl;
}

// klFromLogprobs: the ESTIMATED KL used in real PPO/RLHF, where we do NOT have
// the full distribution — only the log-prob of the tokens we actually sampled.
// Two estimators, returned together so the reader can SEE the bias trade-off
// that papers gloss over:
//
//   k1 = logpOld - logpNew                  (mean ≈ KL, but high variance,
//                                            and can go NEGATIVE on a sample —
//                                            nonsensical for a "distance")
//   k3 = (logpNew - logpOld).exp() - 1
//        - (logpNew - logpOld)              (the unbiased, always-≥0, low-variance
//                                            estimator from Schulman's note,
//                                            what TRL/OpenAI actually ship)
//
// Invariant: both are estimators of KL(new || old) averaged over the sampled
// batch; on a single sample only their *expectation* equals KL. We expose the
// per-sample pair and let the stage average them — averaging is where k3's
// variance win shows up numerically.
export function klFromLogprobs(
  logpNew: number,
  logpOld: number,
): { k1: number; k3: number } {
  const logRatio = logpNew - logpOld; // log(new/old)
  const k1 = -logRatio; // = logpOld - logpNew, the naive estimator
  const k3 = Math.exp(logRatio) - 1 - logRatio; // always >= 0 by convexity of exp
  return { k1, k3 };
}
