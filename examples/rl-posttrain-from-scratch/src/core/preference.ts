// core/preference.ts — the synthetic "preference world" the post-training half
// of the book (RM / RLHF / DPO / GRPO) trains against.
//
// The honesty problem this file solves: real RLHF is judged by human raters we
// cannot put in a CPU-only repo. So we fabricate a *known* preference function
// `trueRewardFn` and treat it as the (hidden) ground-truth human. The reward
// model, the DPO policy, the GRPO group ranker — none of them are allowed to see
// trueRewardFn; they only see noisy preference *labels* derived from it. That
// gap is the entire lesson: "the model optimizes the proxy it was given, not the
// truth," and reward hacking is what happens when the proxy and the truth
// decouple. Because trueRewardFn is computable, we can MEASURE that decoupling
// (Pearson/Spearman of learned vs true reward, win-rate vs reference under the
// true judge) instead of hand-waving it.
//
// Determinism: prompts, responses, pairs, and label flips all derive from the
// passed rng. Same seed => same dataset => same RM/DPO curves.

import { gaussian, sampleCategorical, type Rng } from "./rng.js";

// A "response" is a feature vector, not text — this is a math repo, not an NLP
// one. Two interpretable features keep trueRewardFn explainable:
//   length      — how long the answer is (there is a GOLDEN length; too short or
//                 too long is penalized). This is the classic "models learn to
//                 ramble because longer == higher reward" hacking vector.
//   keywordHits — count of required "key tokens" present (more is better, with
//                 diminishing returns). Stand-in for "did it actually answer".
// The reward trades these off. The hackable axis is length: an RM that overfits
// length can be gamed by padding, and the RLHF/GRPO stages demonstrate exactly
// that when the KL leash is loose.
export interface Prompt {
  readonly id: number;
  readonly requiredKeywords: number; // how many key tokens THIS prompt needs
}

export interface Response {
  readonly length: number;
  readonly keywordHits: number;
}

export interface PreferencePair {
  readonly prompt: Prompt;
  readonly chosen: Response;
  readonly rejected: Response;
  readonly flipped: boolean; // true if label noise inverted this pair (for audits)
}

export interface PreferenceWorld {
  readonly goldenLength: number;
  trueRewardFn(prompt: Prompt, response: Response): number;
  // Sample a plausible response for a prompt (the "policy rollouts" RM/DPO learn
  // from). Spread around the prompt's needs with noise so pairs are non-trivial.
  sampleResponse(prompt: Prompt, rng: Rng): Response;
  samplePrompt(rng: Rng): Prompt;
  generatePairs(n: number, rng: Rng, flipProb?: number): PreferencePair[];
  // Gold ranking of a fixed response set per prompt, the evaluation ground truth
  // for "did the RM order things the way the true reward does."
  goldRanking(prompt: Prompt, responses: Response[]): number[];
}

export interface PreferenceWorldOpts {
  goldenLength?: number;
  maxKeywords?: number;
  lengthPenaltyWeight?: number; // how hard wrong length is punished
}

export function makePreferenceWorld(opts: PreferenceWorldOpts = {}): PreferenceWorld {
  const goldenLength = opts.goldenLength ?? 12;
  const maxKeywords = opts.maxKeywords ?? 5;
  const lambda = opts.lengthPenaltyWeight ?? 0.05;

  // The hidden truth. Quadratic length penalty around goldenLength + concave
  // keyword reward (sqrt => diminishing returns, so spamming one keyword class
  // does not dominate). The shape matters pedagogically: the length term has a
  // PEAK, so a reward model that learns "monotonically longer = better" (the
  // easy thing to overfit) is provably wrong past the peak — that is the gap the
  // hacking stages exploit.
  const trueRewardFn = (prompt: Prompt, response: Response): number => {
    const lengthErr = response.length - goldenLength;
    const lengthTerm = -lambda * lengthErr * lengthErr;
    const usefulHits = Math.min(response.keywordHits, prompt.requiredKeywords);
    const keywordTerm = Math.sqrt(usefulHits);
    return keywordTerm + lengthTerm;
  };

  const samplePrompt = (rng: Rng): Prompt => ({
    id: Math.floor(rng() * 1e9),
    // requiredKeywords in [1, maxKeywords].
    requiredKeywords: 1 + sampleCategorical(new Array(maxKeywords).fill(1 / maxKeywords), rng),
  });

  const sampleResponse = (prompt: Prompt, rng: Rng): Response => {
    // Length around golden but wide: many responses miss the peak, creating
    // clear winners/losers in a pair. Round + clamp to a sane positive integer.
    const length = Math.max(1, Math.round(gaussian(rng, goldenLength, 5)));
    // keywordHits around what the prompt needs, also noisy. Clamp to [0, maxKw].
    const hits = Math.max(
      0,
      Math.min(maxKeywords, Math.round(gaussian(rng, prompt.requiredKeywords, 1.2))),
    );
    return { length, keywordHits: hits };
  };

  const generatePairs = (n: number, rng: Rng, flipProb = 0): PreferencePair[] => {
    const pairs: PreferencePair[] = [];
    let attempts = 0;
    // We need each pair to have a STRICT preference (different true rewards),
    // otherwise the label is meaningless and pollutes RM accuracy. Resample ties.
    while (pairs.length < n && attempts < n * 50) {
      attempts++;
      const prompt = samplePrompt(rng);
      const a = sampleResponse(prompt, rng);
      const b = sampleResponse(prompt, rng);
      const ra = trueRewardFn(prompt, a);
      const rb = trueRewardFn(prompt, b);
      if (ra === rb) continue;

      // Bradley–Terry labeling: the probability that a is preferred is
      // σ(rewardA - rewardB). We do NOT always pick the higher-reward one — that
      // would give NOISELESS labels and make RM accuracy a meaningless 100%.
      // Sampling from BT injects the natural ambiguity of close pairs (raters
      // disagree on near-ties), which is what makes RM accuracy < 100% honest.
      const pAOverB = 1 / (1 + Math.exp(-(ra - rb)));
      let aIsChosen = rng() < pAOverB;

      // Additional EXPLICIT label noise: with flipProb, invert the label. This is
      // the "lazy/adversarial annotator" knob. The lesson (demoed in RM/DPO
      // stages): RM/DPO accuracy degrades smoothly as flipProb rises, and a high
      // flipProb is indistinguishable from a weak reward signal => garbage in,
      // garbage out, no matter how good the optimizer.
      let flipped = false;
      if (rng() < flipProb) {
        aIsChosen = !aIsChosen;
        flipped = true;
      }

      pairs.push({
        prompt,
        chosen: aIsChosen ? a : b,
        rejected: aIsChosen ? b : a,
        flipped,
      });
    }
    return pairs;
  };

  const goldRanking = (prompt: Prompt, responses: Response[]): number[] => {
    // Indices sorted by true reward, best first. Stable on ties via index. This
    // is the yardstick for Spearman against an RM's predicted ranking.
    return responses
      .map((r, i) => ({ i, r: trueRewardFn(prompt, r) }))
      .sort((x, y) => y.r - x.r || x.i - y.i)
      .map((o) => o.i);
  };

  return {
    goldenLength,
    trueRewardFn,
    sampleResponse,
    samplePrompt,
    generatePairs,
    goldRanking,
  };
}
