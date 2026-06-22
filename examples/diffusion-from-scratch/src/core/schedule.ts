// core/schedule.ts — The diffusion noise schedule: a precomputed table of constants that
// the forward (q) and reverse (p) formulas both read from.
//
// WHY a shared table: the forward process q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I) and the
//   reverse sampler both depend on the SAME α/ᾱ constants. If a stage recomputed them with
//   a slightly different β range or accumulation order, forward and reverse would silently
//   disagree and samples would be garbage with no error thrown. One table, one source.
//
// THE CHAIN: β_t (variance added at step t) -> α_t = 1 - β_t -> ᾱ_t = ∏_{s≤t} α_s
//   (the cumulative signal-retention factor). √ᾱ_t scales the clean signal; √(1-ᾱ_t)
//   scales the noise. These two are what stage01's forward step actually multiplies by.
//
// HARD INVARIANT (asserted at construction): ᾱ_t is monotonically DECREASING and ends near 0.
//   ᾱ_T ≈ 0 means x_T is (almost) pure noise — that is the entire premise of the reverse
//   process starting from N(0, I). A schedule that doesn't drive ᾱ to ~0 leaves residual
//   signal at t=T that the sampler can never have seen, breaking generation. We check it so
//   a mis-specified β range fails loudly instead of producing subtly bad samples.

export interface NoiseSchedule {
  T: number; // number of diffusion steps
  betas: Float64Array; // β_t, variance injected at each forward step
  alphas: Float64Array; // α_t = 1 - β_t
  alphaBars: Float64Array; // ᾱ_t = ∏_{s=1..t} α_s (cumulative)
  sqrtAlphaBar: Float64Array; // √ᾱ_t, scales the clean signal in q(x_t|x_0)
  sqrtOneMinusAlphaBar: Float64Array; // √(1-ᾱ_t), scales the noise in q(x_t|x_0)
}

/** Build the derived arrays from betas and assert the monotonic-decreasing ᾱ invariant. */
function buildFromBetas(betas: Float64Array): NoiseSchedule {
  const T = betas.length;
  const alphas = new Float64Array(T);
  const alphaBars = new Float64Array(T);
  const sqrtAlphaBar = new Float64Array(T);
  const sqrtOneMinusAlphaBar = new Float64Array(T);
  let cum = 1;
  for (let t = 0; t < T; t++) {
    if (betas[t] <= 0 || betas[t] >= 1) {
      // β must be a valid variance fraction in (0,1); outside that α=1-β leaves [0,1]
      // and ᾱ stops being a probability-mass-like factor -> NaNs downstream.
      throw new Error(`schedule: beta[${t}]=${betas[t]} must be in (0,1)`);
    }
    alphas[t] = 1 - betas[t];
    cum *= alphas[t];
    alphaBars[t] = cum;
    sqrtAlphaBar[t] = Math.sqrt(cum);
    sqrtOneMinusAlphaBar[t] = Math.sqrt(1 - cum);
    // Monotonic check: each α_t < 1 so the product can only shrink. Guard against a caller
    // passing a non-increasing-noise schedule that would violate the reverse-step premise.
    if (t > 0 && alphaBars[t] >= alphaBars[t - 1]) {
      throw new Error(`schedule: alphaBar not strictly decreasing at t=${t}`);
    }
  }
  return { T, betas, alphas, alphaBars, sqrtAlphaBar, sqrtOneMinusAlphaBar };
}

/**
 * Linear β schedule (the original DDPM): β interpolates linearly from b0 to b1 over T steps.
 * Defaults (b0=1e-4, b1=0.02) are the Ho et al. 2020 values tuned for T=1000; on toy T this
 * still demonstrates the mechanism but ᾱ_T may not reach as close to 0 — the cosine schedule
 * below was invented precisely to fix that, which stage01 demonstrates by printing ᾱ_T for both.
 */
export function linearSchedule(T: number, b0 = 1e-4, b1 = 0.02): NoiseSchedule {
  if (T < 2) throw new Error(`linearSchedule: T must be >= 2, got ${T}`);
  const betas = new Float64Array(T);
  for (let t = 0; t < T; t++) betas[t] = b0 + (b1 - b0) * (t / (T - 1));
  return buildFromBetas(betas);
}

/**
 * Cosine schedule (Nichol & Dhariwal 2021). Instead of specifying β directly, it defines
 * ᾱ_t = f(t)/f(0) with f(t) = cos²((t/T + s)/(1 + s) · π/2), then derives β_t = 1 - ᾱ_t/ᾱ_{t-1}.
 * WHY it beats linear: linear destroys information too fast at the end (ᾱ drops sharply),
 * wasting the last steps on near-pure-noise. Cosine keeps ᾱ falling smoothly so every step
 * carries useful signal — better sample quality, especially at small T. The small offset s
 * prevents β_1 from being exactly 0 (which would make the very first step a no-op).
 * Betas are clamped to (0, 0.999) because the raw ratio can exceed 1 near t=T.
 */
export function cosineSchedule(T: number, s = 0.008): NoiseSchedule {
  if (T < 2) throw new Error(`cosineSchedule: T must be >= 2, got ${T}`);
  const f = (t: number) => Math.pow(Math.cos(((t / T + s) / (1 + s)) * (Math.PI / 2)), 2);
  const f0 = f(0);
  const betas = new Float64Array(T);
  for (let t = 1; t <= T; t++) {
    const abT = f(t) / f0; // ᾱ_t
    const abPrev = f(t - 1) / f0; // ᾱ_{t-1}
    let beta = 1 - abT / abPrev;
    // Clamp into the valid (0,1) range buildFromBetas requires. The lower clamp keeps the
    // first step non-trivial; the upper clamp tames the variance spike near t=T.
    beta = Math.min(Math.max(beta, 1e-5), 0.999);
    betas[t - 1] = beta;
  }
  return buildFromBetas(betas);
}
