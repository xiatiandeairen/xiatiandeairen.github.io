// core/metrics.ts — MoE-specific measurements. These ARE the book's instruments: every
//   honest claim ("the aux loss flattened utilization", "the router collapsed") is one of
//   these numbers moving. They take plain arrays (decoupled from the autograd engine) so
//   stages can measure post-hoc without building a graph.
//
// HONESTY: FLOP counts here are real MAC counts derived from matmul dimensions — not
//   wall-clock. They quantify ALGORITHMIC work (the sparse-vs-dense ratio that transfers),
//   not throughput (which depends on hardware/kernels and is optimistic at toy scale).
//   Stages that print FLOPs must say "ratio transfers, absolute count is toy".

import { Value } from "./tensor.js";

/**
 * Shannon entropy (nats) of a probability row. High entropy = router is undecided /
 * spreading mass; entropy collapsing toward 0 across training is the fingerprint of
 * expert collapse (router became a constant). Guards log(0) by skipping zero-mass entries.
 */
export function routingEntropy(probs: number[]): number {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log(p);
  return h;
}

/**
 * Fraction of tokens routed to each expert. `assignments[i]` = the expert id chosen for
 * token i (top-1). Returns length-E array summing to 1. The basic load-balance picture:
 * uniform (1/E each) is healthy; one expert near 1.0 is collapse.
 * INVARIANT: out-of-range expert ids throw — a silent drop would hide a routing bug.
 */
export function expertUtilization(assignments: number[], E: number): number[] {
  const counts = new Array(E).fill(0);
  for (const a of assignments) {
    if (a < 0 || a >= E) throw new Error(`expertUtilization: assignment ${a} outside [0,${E})`);
    counts[a]++;
  }
  const n = assignments.length || 1;
  return counts.map((c) => c / n);
}

/**
 * Switch/GShard-style auxiliary load-balance loss, DIFFERENTIABLE in the gate probs.
 *   aux = E * sum_e ( f_e * P_e )
 * where f_e = fraction of tokens dispatched to expert e (top-1 count fraction, a CONSTANT
 * w.r.t. params — it's a hard count), and P_e = mean gate probability mass on expert e
 * (this carries gradient). Minimizing it pushes the router to spread probability mass so
 * no expert is both over-picked AND over-weighted. The E factor makes the minimum 1.0 at
 * perfect balance, so the printed value reads as "how many× off uniform".
 *
 * WHY f_e is detached (a plain number, not a Value): the count is non-differentiable
 *   (argmax). The GShard trick is to multiply the differentiable P_e by the constant f_e,
 *   so gradient flows ONLY through P_e — exactly mirroring the real implementation. Getting
 *   this wrong (making f_e a Value) is a classic bug that this signature prevents.
 *
 * @param gateProbsPerToken array of [1,E] Value rows, one per token (softmax outputs).
 * @param assignments top-1 expert id per token (the hard count source).
 * @param E number of experts.
 * @returns a scalar Value carrying gradient into the gate probs.
 */
export function loadBalanceLoss(gateProbsPerToken: Value[], assignments: number[], E: number): Value {
  const T = gateProbsPerToken.length;
  if (T === 0) throw new Error("loadBalanceLoss: no tokens");
  // f_e: hard fraction dispatched to each expert — a CONSTANT (no grad).
  const f = expertUtilization(assignments, E); // numbers
  // P_e: mean gate probability on expert e — a Value (carries grad).
  // Build sum_t gateProbs[t] then mean, then dot with f and scale by E.
  let acc: Value = Value.zeros(1, E); // [1,E] zeros leaf, accumulate via add
  for (const g of gateProbsPerToken) acc = acc.add(g);
  const meanP = acc.mul(Value.scalar(1 / T)); // [1,E], P_e
  // dot(f, meanP) = sum_e f_e * P_e. Multiply elementwise by f (constant row) then sum.
  const fRow = Value.from(f, [1, E]); // constant leaf; we never backprop into it (it's f_e)
  const weighted = meanP.mul(fRow); // [1,E], grad flows only into meanP path
  return weighted.sum().mul(Value.scalar(E));
}

/**
 * activated FLOPs (MACs) for a top-k MoE layer: each token runs through k experts, each
 * expert being two matmuls (in×hidden and hidden×out), plus the gate (in×E).
 * Counts MULTIPLY-ACCUMULATEs (1 MAC = 1 mul + 1 add).
 * INVARIANT: this is what a sparse MoE ACTUALLY computes — only k of E experts fire.
 */
export function activatedFLOPs(
  tokens: number,
  inDim: number,
  hidden: number,
  outDim: number,
  E: number,
  k: number,
): number {
  const gate = inDim * E; // gate logits per token
  const oneExpert = inDim * hidden + hidden * outDim;
  return tokens * (gate + k * oneExpert);
}

/**
 * dense FLOPs (MACs): the SAME total parameters as the MoE but every token runs through
 * ALL E experts (equivalently one big FFN of width E*hidden). This is the baseline the
 * MoE claims to beat: the ratio activated/dense ~ k/E is the sparsity win that transfers.
 */
export function denseFLOPs(tokens: number, inDim: number, hidden: number, outDim: number, E: number): number {
  const gate = inDim * E; // a dense model has no gate, but include for apples-to-apples? No:
  void gate; // dense baseline has no router; we deliberately exclude it.
  const allExperts = E * (inDim * hidden + hidden * outDim);
  return tokens * allExperts;
}

/**
 * Coefficient of variation of expert utilization = std/mean. THE scalar collapse metric:
 *   0   => perfectly balanced (all experts equal)
 *   high => some experts hog tokens (collapse). Used as the headline number in stage06.
 * WHY CV not raw std: it's scale-free, so it's comparable across different E and token counts.
 */
export function coefficientOfVariation(util: number[]): number {
  const n = util.length;
  if (n === 0) return 0;
  let mean = 0;
  for (const u of util) mean += u;
  mean /= n;
  if (mean === 0) return 0;
  let varSum = 0;
  for (const u of util) varSum += (u - mean) * (u - mean);
  const std = Math.sqrt(varSum / n);
  return std / mean;
}
