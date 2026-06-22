// stage07-latent.ts — Latent diffusion: why we don't diffuse in pixel space, and where the
// compute actually gets saved. This is the book's closing chapter, run fully offline & seeded.
//
// THE ARGUMENT (Rombach et al. 2022, "High-Resolution Image Synthesis with Latent Diffusion"):
//   real images are mostly perceptually-irrelevant high-frequency detail. Training a diffusion
//   model on raw pixels burns almost all its capacity (and almost all the sampler's NFE × dim
//   cost) modeling noise no human cares about. The fix is a TWO-STAGE split:
//     1. Perceptual compression: an autoencoder learns enc: data -> latent, dec: latent -> data,
//        throwing away the imperceptible detail ONCE, up front.
//     2. Generative modeling: a diffusion model runs entirely in the small latent space; at
//        sample time you decode the generated latent back with dec().
//   The diffusion model never sees a pixel. Its per-step cost scales with LATENT dim, not data
//   dim — that is the whole saving.
//
// WHAT THIS STAGE PROVES ON A TOY (2-D swiss roll, latent = 1-D):
//   (1) a tiny autoencoder really does fold the 2-D curled manifold onto a usable 1-D latent
//       (reconstruction MSE falls, decoded cloud keeps the swiss-roll shape);
//   (2) a SMALLER diffusion model trained in that 1-D latent + dec() generates a cloud whose
//       Chamfer distance to the data is in the same ballpark as a direct 2-D diffusion model;
//   (3) the latent route uses a smaller denoiser and ~half the sampler compute (NFE × dim);
//   (4) FAILURE MODE: if the autoencoder is under-trained, the latent loses the curl, and NO
//       amount of diffusion quality can recover it — "latent quality is the HARD CEILING on
//       generative quality; the autoencoder is the foundation, and if the foundation collapses
//       the diffusion model on top cannot save it."
//
// HONESTY (this is a toy, the numbers are deliberately small):
//   - Our compression is 2D -> 1D (2× on dims). A real LDM uses an 8×-downsampled VAE, i.e.
//     ~48× fewer latent elements; the compute win there is enormous, not the modest factor here.
//   - Absolute Chamfer / loss values are optimistic vs real image diffusion (the manifold is
//     trivial). What TRANSFERS is the *architecture decision* — decouple perceptual compression
//     from generation — and the *shape* of the trade-off: cost drops, quality drops a little,
//     and a bad autoencoder caps everything. Treat ratios, not absolutes, as the signal.
//
// Run: npx tsx src/stage07-latent.ts   (deterministic for seed 1337)

import { swissRoll2D } from "./core/data.js";
import { MLP, SinusoidalEmbedding } from "./core/nn.js";
import { Adam } from "./core/optim.js";
import { scatterASCII } from "./core/plot.js";
import { RNG } from "./core/rng.js";
import { cosineSchedule, type NoiseSchedule } from "./core/schedule.js";
import { Tensor } from "./core/tensor.js";

const SEED = 1337;
const N_DATA = 600; // swiss-roll points; enough to read structure in a 56×18 scatter

function section(title: string): void {
  console.log(`\n${"=".repeat(72)}\n${title}\n${"=".repeat(72)}`);
}

// ============================ shared diffusion helpers ============================
// These are dimension-agnostic (work on [n,1] latents and [n,2] data alike), which is the
// whole point: the SAME DDPM math runs in latent space or data space — only `dim` changes.

/** q(x_t | x_0) = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε, vectorized over a batch that all share one t.
 *  Returns a plain Float64Array (we only need values for building training targets — the
 *  graph for the loss is rebuilt from these as a leaf, so no autograd is needed here). */
function forwardNoise(x0: Float64Array, eps: Float64Array, sched: NoiseSchedule, t: number): Float64Array {
  const sa = sched.sqrtAlphaBar[t];
  const so = sched.sqrtOneMinusAlphaBar[t];
  const out = new Float64Array(x0.length);
  for (let i = 0; i < x0.length; i++) out[i] = sa * x0[i] + so * eps[i];
  return out;
}

/** A noise-prediction (ε-prediction) denoiser: input row = [x_t ⊕ time-embed], output = ε̂.
 *  This is exactly the book's denoiser shape from nn.ts, just parameterized by `dim` so one
 *  class serves both the 1-D latent model and the 2-D data model. */
class Denoiser {
  readonly dim: number;
  private emb: SinusoidalEmbedding;
  private net: MLP;

  constructor(dim: number, embDim: number, hidden: number[], rng: RNG) {
    this.dim = dim;
    this.emb = new SinusoidalEmbedding(embDim);
    this.net = new MLP([dim + embDim, ...hidden, dim], "silu", rng);
  }

  /** Predict ε for a batch x_t (shape [n,dim]) all at the same timestep t. */
  forward(xt: Tensor, t: number): Tensor {
    const te = this.emb.forward(new Array(xt.shape[0]).fill(t)); // [n, embDim] leaf
    return this.net.forward(Tensor.concatCols([xt, te]));
  }

  parameters(): Tensor[] {
    return this.net.parameters();
  }

  /** Trainable scalar count — the apples-to-apples "model size" number we compare. */
  paramCount(): number {
    return this.net.parameters().reduce((s, p) => s + p.size, 0);
  }
}

/** Train an ε-prediction DDPM on a fixed dataset (any dim). Full-batch per step (toy size),
 *  one random t per step shared across the batch — standard for this scale. Returns the
 *  trained denoiser and its loss history. */
function trainDdpm(
  data: Float64Array,
  n: number,
  dim: number,
  sched: NoiseSchedule,
  denoiser: Denoiser,
  steps: number,
  lr: number,
  rng: RNG,
): number[] {
  const opt = new Adam(denoiser.parameters(), lr);
  const losses: number[] = [];
  for (let step = 0; step < steps; step++) {
    const t = rng.choice(sched.T); // sample a timestep uniformly
    // Fresh noise ε and the corresponding x_t for this step.
    const eps = new Float64Array(n * dim);
    for (let i = 0; i < eps.length; i++) eps[i] = rng.gaussian();
    const xt = new Tensor(forwardNoise(data, eps, sched, t), [n, dim]); // leaf input
    const epsTarget = new Tensor(eps, [n, dim]); // leaf target

    const pred = denoiser.forward(xt, t);
    const diff = pred.sub(epsTarget);
    const loss = diff.mul(diff).mean(); // MSE(ε̂, ε)
    opt.zeroGrad();
    loss.backward();
    opt.step();
    losses.push(loss.data[0]);
  }
  return losses;
}

/** Ancestral DDPM sampler (Ho et al. 2020 eq. 11): start from N(0,I), walk t = T-1 .. 0,
 *  each step removing the predicted noise and re-injecting a little fresh noise (except t=0).
 *  NFE = T (one denoiser eval per step). Returns generated samples [n, dim]. */
function sampleDdpm(denoiser: Denoiser, sched: NoiseSchedule, n: number, dim: number, rng: RNG): Tensor {
  let x = new Float64Array(n * dim);
  for (let i = 0; i < x.length; i++) x[i] = rng.gaussian(); // x_T ~ N(0,I)

  for (let t = sched.T - 1; t >= 0; t--) {
    const xt = new Tensor(x, [n, dim]);
    const epsHat = denoiser.forward(xt, t).data; // ε̂(x_t, t)
    const alpha = sched.alphas[t];
    const alphaBar = sched.alphaBars[t];
    const beta = sched.betas[t];
    const coef = beta / sched.sqrtOneMinusAlphaBar[t]; // = (1-α_t)/√(1-ᾱ_t)
    const invSqrtAlpha = 1 / Math.sqrt(alpha);
    // Posterior variance: for t>0 inject √β·z; at t=0 the mean IS the sample (no noise).
    const sigma = t > 0 ? Math.sqrt(beta) : 0;
    const next = new Float64Array(n * dim);
    for (let i = 0; i < next.length; i++) {
      const mean = invSqrtAlpha * (x[i] - coef * epsHat[i]);
      next[i] = mean + sigma * (t > 0 ? rng.gaussian() : 0);
    }
    void alphaBar; // (kept for readers cross-checking the closed form; not needed in this arr)
    x = next;
  }
  return new Tensor(x, [n, dim]);
}

// ============================ autoencoder (the perceptual-compression stage) ============================
// enc: 2D -> latentDim,  dec: latentDim -> 2D. Reconstruction objective = MSE(dec(enc(x)), x).
// WHY two separate MLPs (not one bottlenecked net): we must be able to (a) encode the WHOLE
//   dataset into latents once, freeze, then (b) diffuse purely in latent space, then (c) decode
//   GENERATED latents that the encoder never produced. Keeping enc/dec separate makes that
//   freeze-and-reuse explicit, and mirrors the real LDM where the VAE is frozen before the
//   diffusion model is ever trained.

class Autoencoder {
  private enc: MLP;
  private dec: MLP;
  readonly latentDim: number;

  constructor(latentDim: number, hidden: number, rng: RNG) {
    this.latentDim = latentDim;
    // Small, tanh/silu MLPs. enc squeezes 2 -> latentDim; dec expands back to 2.
    this.enc = new MLP([2, hidden, latentDim], "silu", rng);
    this.dec = new MLP([latentDim, hidden, 2], "silu", rng);
  }

  encode(x: Tensor): Tensor {
    return this.enc.forward(x);
  }

  decode(z: Tensor): Tensor {
    return this.dec.forward(z);
  }

  parameters(): Tensor[] {
    return [...this.enc.parameters(), ...this.dec.parameters()];
  }
}

/** Train the autoencoder to reconstruct `data` ([n,2]). Returns per-step recon-MSE history.
 *  Full-batch (toy). The data Tensor is a fixed leaf reused every step (no grad into it). */
function trainAutoencoder(ae: Autoencoder, data: Tensor, steps: number, lr: number): number[] {
  const opt = new Adam(ae.parameters(), lr);
  const losses: number[] = [];
  for (let step = 0; step < steps; step++) {
    const recon = ae.decode(ae.encode(data));
    const diff = recon.sub(data);
    const loss = diff.mul(diff).mean(); // reconstruction MSE
    opt.zeroGrad();
    loss.backward();
    opt.step();
    losses.push(loss.data[0]);
  }
  return losses;
}

// ============================ metrics ============================

/** Symmetric Chamfer distance between two [n,2] / [m,2] clouds: mean over A of nearest-in-B
 *  squared distance, plus the same B->A, summed. WHY this metric: with no labels and no
 *  density estimate, "are the generated points near the data points (and vice-versa)?" is the
 *  cheap, honest scalar for 2-D sample quality. Lower = better coverage + fidelity. O(n·m),
 *  fine at toy sizes. (Toy caveat: insensitive to local density / mode weights — it rewards
 *  coverage, so read it together with the scatter, not alone.) */
/** Standardize a flat latent buffer to zero-mean/unit-var (the latent-space twin of data.ts's
 *  normalize). Returns the buffer plus the mean/std needed to INVERT it. INVARIANT: the same
 *  mean/std must un-standardize generated latents before decode, or the decoder sees
 *  off-distribution inputs and the swiss roll never reappears. */
function standardize(buf: Float64Array): { standardized: Float64Array; mean: number; std: number } {
  let mean = 0;
  for (let i = 0; i < buf.length; i++) mean += buf[i];
  mean /= buf.length;
  let varc = 0;
  for (let i = 0; i < buf.length; i++) varc += (buf[i] - mean) * (buf[i] - mean);
  varc /= buf.length;
  const std = Math.sqrt(varc) + 1e-8; // +eps guards a degenerate near-constant latent
  const standardized = new Float64Array(buf.length);
  for (let i = 0; i < buf.length; i++) standardized[i] = (buf[i] - mean) / std;
  return { standardized, mean, std };
}

/** Undo standardize(): generated-standardized latent -> raw latent the decoder expects. */
function unstandardize(buf: Float64Array, mean: number, std: number): Float64Array {
  const out = new Float64Array(buf.length);
  for (let i = 0; i < buf.length; i++) out[i] = buf[i] * std + mean;
  return out;
}

function chamfer2D(a: Tensor, b: Tensor): number {
  const na = a.shape[0];
  const nb = b.shape[0];
  const ad = a.data;
  const bd = b.data;
  const dir = (src: Float64Array, ns: number, dst: Float64Array, nd: number): number => {
    let acc = 0;
    for (let i = 0; i < ns; i++) {
      let best = Infinity;
      const sx = src[i * 2];
      const sy = src[i * 2 + 1];
      for (let j = 0; j < nd; j++) {
        const dx = sx - dst[j * 2];
        const dy = sy - dst[j * 2 + 1];
        const d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
      }
      acc += best;
    }
    return acc / ns;
  };
  return dir(ad, na, bd, nb) + dir(bd, nb, ad, na);
}

// ============================ main ============================

function main(): void {
  const rng = new RNG(SEED);

  // The dataset: a 2-D swiss roll — a thin curled 1-D manifold sitting in 2-D. Perfect for
  // a 1-D latent test: in principle ONE coordinate (position along the curve) determines a
  // point, so a good autoencoder should reach a low recon error; a bad one smears the curl.
  const data = swissRoll2D(N_DATA, rng);
  const dataBuf = data.data;

  section("0. The data: 2-D swiss roll (a 1-D curled manifold). Latent diffusion's test bed.");
  console.log(`  swissRoll2D shape = [${data.shape}]  (n=${N_DATA})`);
  console.log(scatterASCII(data, 56, 16));

  // ---- 1. Perceptual compression: train enc 2D->1D / dec 1D->2D ----
  section("1. Autoencoder: fold the 2-D curve onto a 1-D latent (reconstruction MSE)");
  const ae = new Autoencoder(1, 24, new RNG(SEED + 1));
  const aeLosses = trainAutoencoder(ae, data, 1500, 0.01);
  const reconStart = aeLosses[0];
  const reconEnd = aeLosses[aeLosses.length - 1];
  console.log(`  recon MSE: ${reconStart.toFixed(5)} -> ${reconEnd.toFixed(5)}  (${(reconStart / reconEnd).toFixed(1)}x lower)`);
  const reconCloud = ae.decode(ae.encode(data));
  console.log("  decoded reconstruction (should still read as a swiss roll):");
  console.log(scatterASCII(reconCloud, 56, 16));

  // Freeze: encode the whole dataset to 1-D latents ONCE. From here the diffusion model only
  // ever sees these latents — the encoder/decoder weights never change again.
  const latents = ae.encode(data).data; // [n,1] flat
  let lMin = Infinity;
  let lMax = -Infinity;
  for (let i = 0; i < latents.length; i++) {
    if (latents[i] < lMin) lMin = latents[i];
    if (latents[i] > lMax) lMax = latents[i];
  }
  console.log(`  frozen 1-D latent range: [${lMin.toFixed(3)}, ${lMax.toFixed(3)}]  (the space the latent-DDPM will model)`);

  // The latent-DDPM's forward process assumes ~unit-scale data (so x_T ≈ N(0,I)). Latents are
  // not unit-scale, so standardize for diffusion and invert at decode time (see standardize()).
  const { standardized: latentsStd, mean: lMean, std: lStd } = standardize(latents);

  // ---- 2. Latent diffusion: train a SMALL 1-D DDPM, sample, decode ----
  section("2. Latent-DDPM: diffuse in 1-D, then decode generated latents back to 2-D");
  const T = 100;
  const sched = cosineSchedule(T); // cosine: keeps signal across all T even at small T (stage01)
  const latentDenoiser = new Denoiser(1, 8, [32, 32], new RNG(SEED + 2));
  const latLosses = trainDdpm(latentsStd, N_DATA, 1, sched, latentDenoiser, 2500, 0.005, new RNG(SEED + 3));
  console.log(`  latent ε-MSE: ${latLosses[0].toFixed(5)} -> ${latLosses[latLosses.length - 1].toFixed(5)}`);
  const genLatentStd = sampleDdpm(latentDenoiser, sched, N_DATA, 1, new RNG(SEED + 4)).data;
  // Un-standardize, then DECODE back to 2-D with the frozen decoder.
  const genLatent = unstandardize(genLatentStd, lMean, lStd);
  const genFromLatent = ae.decode(new Tensor(genLatent, [N_DATA, 1]));
  const chamferLatent = chamfer2D(genFromLatent, data);
  console.log(`  latent-DDPM generated cloud (decoded from 1-D), Chamfer to data = ${chamferLatent.toFixed(4)}:`);
  console.log(scatterASCII(genFromLatent, 56, 16));

  // ---- 3. Baseline: a direct 2-D DDPM (no autoencoder), then a head-to-head cost/quality table ----
  // NOTE: this is a freshly-trained inline 2-D DDPM, NOT an import of stage04 (importing a
  //   stage file would run its main()). It plays the role of "stage04's direct pixel-space
  //   model" for the comparison. Roughly matched training budget so the comparison is fair.
  section("3. Baseline direct 2-D DDPM (the 'pixel-space' route) — train + sample");
  const dataDenoiser = new Denoiser(2, 16, [64, 64], new RNG(SEED + 5));
  const dataLosses = trainDdpm(dataBuf, N_DATA, 2, sched, dataDenoiser, 2500, 0.005, new RNG(SEED + 6));
  console.log(`  2-D ε-MSE: ${dataLosses[0].toFixed(5)} -> ${dataLosses[dataLosses.length - 1].toFixed(5)}`);
  const gen2D = sampleDdpm(dataDenoiser, sched, N_DATA, 2, new RNG(SEED + 7));
  const chamfer2DBaseline = chamfer2D(gen2D, data);
  console.log(`  direct 2-D DDPM generated cloud, Chamfer to data = ${chamfer2DBaseline.toFixed(4)}:`);
  console.log(scatterASCII(gen2D, 56, 16));

  // ---- The comparison table: param count, sampler compute (NFE × dim), quality ----
  section("4. Head-to-head: direct 2-D DDPM vs latent-DDPM (cost vs quality)");
  const p2d = dataDenoiser.paramCount();
  const pLat = latentDenoiser.paramCount();
  const pAe = ae.parameters().reduce((s, q) => s + q.size, 0);
  // Sampler compute proxy = NFE × denoiser-input-dim. The dominant per-step work is the
  // denoiser forward; its first matmul width scales with the input dim, so NFE × dim is a
  // fair (toy) proxy for "where the sampling FLOPs go". Decoder is ONE forward, amortized.
  const nfe = T;
  const cost2d = nfe * 2;
  const costLat = nfe * 1;
  const pad = (s: string, w: number) => s.padEnd(w);
  console.log(`  ${pad("metric", 34)}${pad("direct 2-D DDPM", 18)}latent-DDPM`);
  console.log(`  ${"-".repeat(34 + 18 + 12)}`);
  console.log(`  ${pad("denoiser params", 34)}${pad(String(p2d), 18)}${pLat}  (${(p2d / pLat).toFixed(2)}x smaller)`);
  console.log(`  ${pad("+ one-time autoencoder params", 34)}${pad("0", 18)}${pAe}  (paid ONCE, not per sample)`);
  console.log(`  ${pad("sampler NFE", 34)}${pad(String(nfe), 18)}${nfe}`);
  console.log(`  ${pad("sampler compute (NFE × dim)", 34)}${pad(String(cost2d), 18)}${costLat}  (${(cost2d / costLat).toFixed(2)}x less)`);
  console.log(`  ${pad("Chamfer to data (lower=better)", 34)}${pad(chamfer2DBaseline.toFixed(4), 18)}${chamferLatent.toFixed(4)}`);
  console.log("");
  console.log(`  Reading it: the latent route runs a ${(p2d / pLat).toFixed(2)}x smaller denoiser at ${(cost2d / costLat).toFixed(2)}x less per-sample compute.`);
  if (chamferLatent <= chamfer2DBaseline * 1.5) {
    console.log(`  Quality stayed close (Chamfer ${chamferLatent.toFixed(4)} vs ${chamfer2DBaseline.toFixed(4)}) — the cost win came nearly for free here.`);
  } else {
    console.log(`  Quality dropped some (Chamfer ${chamferLatent.toFixed(4)} vs ${chamfer2DBaseline.toFixed(4)}) — the toy's 2→1 squeeze is lossy; on real 8× VAEs the cost win dwarfs this gap.`);
  }
  console.log(`  HONEST: this is a 2→1 (2×) squeeze; a real LDM uses an 8×-downsampled VAE (~48× fewer latent elements),`);
  console.log(`  where the same architecture buys an order-of-magnitude more, not the modest factors above.`);

  // ---- 5. FAILURE MODE: under-trained autoencoder caps everything downstream ----
  section("5. FAILURE MODE: a weak autoencoder is a HARD CEILING on latent-diffusion quality");
  // SAME everything as section 1-2, the ONLY change: train the autoencoder for 100 steps
  // instead of 1500. The encoder never learns to lay the curl out cleanly on the 1-D line, so
  // the latents it produces are a tangled, near-constant mush. We then run the FULL latent
  // diffusion pipeline (same budget as the good run) on top — and watch it fail to recover,
  // because the structure was destroyed BEFORE diffusion ever started.
  const aeBad = new Autoencoder(1, 24, new RNG(SEED + 1)); // identical init
  const aeBadLosses = trainAutoencoder(aeBad, data, 100, 0.01); // <-- under-trained on purpose
  const reconBadEnd = aeBadLosses[aeBadLosses.length - 1];
  console.log(`  weak autoencoder recon MSE after 100 steps: ${reconBadEnd.toFixed(5)}  (vs ${reconEnd.toFixed(5)} for the 1500-step one)`);
  const reconBadCloud = aeBad.decode(aeBad.encode(data));
  console.log("  weak reconstruction (the curl is already gone — the ceiling is set HERE):");
  console.log(scatterASCII(reconBadCloud, 56, 12));

  // Encode -> standardize -> diffuse -> sample -> decode, exactly as the good run.
  const { standardized: latentsBadStd, mean: mB, std: sB } = standardize(aeBad.encode(data).data);
  const badDenoiser = new Denoiser(1, 8, [32, 32], new RNG(SEED + 2)); // same model as good run
  trainDdpm(latentsBadStd, N_DATA, 1, sched, badDenoiser, 2500, 0.005, new RNG(SEED + 3));
  const genBadStd = sampleDdpm(badDenoiser, sched, N_DATA, 1, new RNG(SEED + 4)).data;
  const genBad = unstandardize(genBadStd, mB, sB);
  const genBadCloud = aeBad.decode(new Tensor(genBad, [N_DATA, 1]));
  const chamferBad = chamfer2D(genBadCloud, data);
  console.log(`  latent-DDPM ON TOP of the weak autoencoder, Chamfer to data = ${chamferBad.toFixed(4)}`);
  console.log(`  (good autoencoder + same diffusion gave ${chamferLatent.toFixed(4)} — ${(chamferBad / chamferLatent).toFixed(1)}x worse):`);
  console.log(scatterASCII(genBadCloud, 56, 12));
  console.log("");
  console.log(`  THE LESSON: the diffusion model on top was IDENTICAL (same seed, arch, budget). The only`);
  console.log(`  difference was 100 vs 1500 autoencoder steps. Latent quality is the ceiling: dec(latent) can`);
  console.log(`  never show structure the latent threw away. The autoencoder is the foundation — a great`);
  console.log(`  diffusion model on a collapsed latent still produces a collapsed cloud. This is the toy face`);
  console.log(`  of LDM's real tension: push the compression ratio too hard and reconstruction (hence sample)`);
  console.log(`  quality caps out, no matter how good the generator is.`);

  section("stage07 complete — latent diffusion: decouple perceptual compression from generation");
}

main();
