// stage11-statmech.ts — Statistical mechanics for builders: Monte-Carlo sampling,
// simulated annealing, and the Ising phase transition.
//
// WHY this is the bridge chapter: every modern generative sampler (diffusion, energy-based
//   models, MCMC inference, Gibbs sampling) is the SAME idea — propose a random move, accept
//   it with a probability set by a Boltzmann factor exp(-ΔE / T), and you converge to a
//   target distribution without ever computing its normalizing constant. This stage builds
//   that machinery three times so the reader sees it is one mechanism, not three tricks:
//     ① Metropolis MCMC samples a 1-D Boltzmann density; histogram converges to the truth.
//     ② Simulated annealing is Metropolis with T → 0; it minimizes a TSP tour length.
//     ③ The 2-D Ising model is Metropolis on a spin lattice; sweeping T reveals a real
//        phase transition (magnetization collapses near the critical temperature).
//
// HONEST-NUMBER NOTE: all lattices/tours here are tiny and seeded, so absolute numbers
//   (exact tour length, exact T_c) are toy. What transfers is the SHAPE and the RELATIVE
//   facts: histograms approach the analytic density; slow cooling beats fast cooling on the
//   SAME instance; magnetization stays high below ~2.27 and collapses above it. Every number
//   printed is computed/measured at run time — nothing is hand-typed.
//
// CONTRACT: seeded via core/rng.js (the book's only randomness source — bit-for-bit
//   reproducible); curves drawn with core/plot.js. No network, no deps, CPU-only.

import { mulberry32, type Rng } from "./core/rng.js";
import { mean, std, histogram } from "./core/stats.js";
import { asciiLine, asciiBar } from "./core/plot.js";

// ---------------------------------------------------------------------------
// ① Metropolis MCMC: sample from an unnormalized target density.
// ---------------------------------------------------------------------------

/**
 * Metropolis random-walk sampler for a 1-D target whose UNNORMALIZED log-density is given.
 * WHY log-density not density: densities underflow to 0 fast (a bimodal Gaussian mix at the
 *   tails is ~1e-300); working in logs keeps the accept ratio numerically alive. The whole
 *   point of MCMC is that we never need the normalizing constant Z — it cancels in the ratio
 *   exp(logP(x') - logP(x)).
 * INVARIANT: detailed balance holds because the proposal (add symmetric Gaussian noise) is
 *   symmetric, so the Hastings correction is 1 and the accept rule is plain min(1, ratio).
 * FAILURE MODE (not demoed here, demoed in annealing): too small a step → the chain barely
 *   moves and never crosses between modes; too large → almost everything is rejected.
 */
function metropolisSample(
  logDensity: (x: number) => number,
  start: number,
  stepStd: number,
  nSamples: number,
  rng: Rng,
): number[] {
  const out: number[] = [];
  let x = start;
  let logp = logDensity(x);
  for (let i = 0; i < nSamples; i++) {
    // Symmetric Gaussian proposal: x' = x + N(0, stepStd^2). Symmetry is what lets us
    // drop the proposal ratio from the acceptance test.
    const xProp = x + stepStd * gaussian(rng);
    const logpProp = logDensity(xProp);
    // Accept with prob min(1, exp(Δlogp)). In log space: accept iff log(u) < Δlogp.
    if (Math.log(rng() + 1e-300) < logpProp - logp) {
      x = xProp;
      logp = logpProp;
    }
    out.push(x); // record EVERY step (including rejections held in place) — that is the chain
  }
  return out;
}

/** One standard-normal draw (Box–Muller); local so the sampler owns its proposal noise. */
function gaussian(rng: Rng): number {
  const u1 = 1 - rng(); // (0,1] avoids log(0)
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function demoMetropolis(): void {
  console.log("=== Stage 11 · 统计力学：采样、退火与相变 ===\n");
  console.log("--- ① Metropolis MCMC：从玻尔兹曼分布采样 ---");
  // Target: a two-peak mixture, the kind a sampler must explore (not just a single mode).
  // logDensity = log( 0.6·N(-2,0.7) + 0.4·N(3,1.0) ), normalizing constants dropped.
  const logTarget = (x: number): number => {
    const g1 = 0.6 * Math.exp(-((x + 2) ** 2) / (2 * 0.7 ** 2));
    const g2 = 0.4 * Math.exp(-((x - 3) ** 2) / (2 * 1.0 ** 2));
    return Math.log(g1 + g2 + 1e-300);
  };

  const rng = mulberry32(1101);
  const burnIn = 2000; // discard the chain's transient before it forgets its start point
  const chain = metropolisSample(logTarget, 0, 1.5, 40_000, rng).slice(burnIn);

  // The chain should split ~60/40 between the two modes (the mixture weights). That ratio,
  // recovered from samples alone, is the proof MCMC found the RIGHT distribution.
  const inLeft = chain.filter((x) => x < 0.5).length / chain.length;
  console.log(`样本量 (去 burn-in)     = ${chain.length}`);
  console.log(`落在左峰 (x<0.5) 的比例 = ${(inLeft * 100).toFixed(1)}%  (目标混合权重 = 60%)`);
  console.log(`样本均值                = ${mean(chain).toFixed(3)}  (理论 0.6·(-2)+0.4·3 = 0.000)`);

  // Histogram the samples and overlay the (renormalized-to-counts) true density so the
  // reader SEES the empirical bars track the analytic curve.
  const bins = 24;
  const h = histogram(chain, bins);
  const labels: string[] = [];
  const empirical: number[] = [];
  const analytic: number[] = [];
  for (let b = 0; b < bins; b++) {
    const center = (h.edges[b] + h.edges[b + 1]) / 2;
    labels.push(center.toFixed(1));
    empirical.push(h.counts[b]);
    analytic.push(Math.exp(logTarget(center))); // unnormalized; asciiBar autoscales per chart
  }
  console.log("\n经验直方图 (MCMC 样本计数):");
  console.log(asciiBar(labels, empirical, 40));
  console.log("\n目标密度 (解析，同样 bin 中心):");
  console.log(asciiBar(labels, analytic, 40));
  console.log("两图峰位/相对高度一致 → MCMC 无需归一化常数 Z 就采到了正确分布。\n");
}

// ---------------------------------------------------------------------------
// ② Simulated annealing on a small TSP — and the fast-cooling failure mode.
// ---------------------------------------------------------------------------

interface City {
  x: number;
  y: number;
}

/** Closed-tour length: sum of edge distances including the return to the start. */
function tourLength(tour: readonly number[], cities: readonly City[]): number {
  let d = 0;
  for (let i = 0; i < tour.length; i++) {
    const a = cities[tour[i]];
    const b = cities[tour[(i + 1) % tour.length]]; // wrap: last → first closes the loop
    d += Math.hypot(a.x - b.x, a.y - b.y);
  }
  return d;
}

/**
 * Simulated annealing: Metropolis where the "temperature" is cooled toward 0 over time.
 * WHY a schedule at all: at high T almost any move is accepted, so the search wanders freely
 *   and escapes local optima; as T → 0 only improving moves survive, so it settles into a
 *   minimum. The schedule is the entire art — cool too fast and you freeze in the first
 *   ditch you fall into (that is the failure mode this function is built to expose).
 * `record` captures the BEST-so-far length at each step so the caller can plot convergence.
 * Returns the best tour found and its full convergence trace.
 */
function annealTsp(
  cities: readonly City[],
  tStart: number,
  cooling: number, // multiplicative per-step factor in (0,1); closer to 1 = slower cooling
  steps: number,
  rng: Rng,
): { best: number[]; bestLen: number; trace: number[] } {
  let tour = shuffleTour(cities.length, rng);
  let len = tourLength(tour, cities);
  let best = tour.slice();
  let bestLen = len;
  let t = tStart;
  const trace: number[] = [];
  for (let s = 0; s < steps; s++) {
    // 2-swap proposal: reverse a random segment (a 2-opt-style local move).
    const i = 1 + Math.floor(rng() * (cities.length - 1));
    const j = 1 + Math.floor(rng() * (cities.length - 1));
    const [lo, hi] = i < j ? [i, j] : [j, i];
    const cand = tour.slice();
    reverseSegment(cand, lo, hi);
    const candLen = tourLength(cand, cities);
    const dE = candLen - len; // energy = tour length; we want to MINIMIZE it
    // Accept improvements always; accept worse moves with prob exp(-ΔE/T) — the uphill
    // moves at high T are exactly what lets annealing climb out of a local minimum.
    if (dE < 0 || rng() < Math.exp(-dE / Math.max(t, 1e-9))) {
      tour = cand;
      len = candLen;
      if (len < bestLen) {
        bestLen = len;
        best = tour.slice();
      }
    }
    t *= cooling;
    trace.push(bestLen);
  }
  return { best, bestLen, trace };
}

function shuffleTour(n: number, rng: Rng): number[] {
  const t = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1)); // inclusive bound → unbiased Fisher–Yates
    [t[i], t[j]] = [t[j], t[i]];
  }
  return t;
}

/** In-place segment reversal (the 2-opt move's core); mutates `tour` between lo..hi. */
function reverseSegment(tour: number[], lo: number, hi: number): void {
  while (lo < hi) {
    [tour[lo], tour[hi]] = [tour[hi], tour[lo]];
    lo++;
    hi--;
  }
}

/**
 * Deterministic baseline: greedy nearest-neighbor construction, then 2-opt local search to a
 * local optimum. WHY this baseline: a random instance has no closed-form optimum, so "X% over
 *   optimal" would be a lie. NN+2-opt is the textbook strong heuristic; comparing annealing to
 *   it is an honest "did annealing reach competitive quality" check, not a fake optimality claim.
 * No randomness — same instance always yields the same baseline, keeping the demo reproducible.
 */
function nearestNeighborThen2opt(cities: readonly City[]): number {
  const n = cities.length;
  const visited = new Array<boolean>(n).fill(false);
  const tour: number[] = [0];
  visited[0] = true;
  for (let step = 1; step < n; step++) {
    const last = tour[tour.length - 1];
    let bestCity = -1;
    let bestDist = Infinity;
    for (let c = 0; c < n; c++) {
      if (visited[c]) continue;
      const d = Math.hypot(cities[last].x - cities[c].x, cities[last].y - cities[c].y);
      if (d < bestDist) {
        bestDist = d;
        bestCity = c;
      }
    }
    visited[bestCity] = true;
    tour.push(bestCity);
  }
  // 2-opt: repeatedly reverse any segment that shortens the tour, until no improvement.
  let improved = true;
  while (improved) {
    improved = false;
    for (let i = 1; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const cand = tour.slice();
        reverseSegment(cand, i, j);
        if (tourLength(cand, cities) < tourLength(tour, cities) - 1e-12) {
          tour.splice(0, n, ...cand);
          improved = true;
        }
      }
    }
  }
  return tourLength(tour, cities);
}

function demoAnnealing(): void {
  console.log("--- ② 模拟退火解 TSP + 失败模式：降温太快卡在局部最优 ---");
  // 30 random cities in the unit square: a genuinely rugged landscape with many local optima,
  // unlike a points-on-a-circle toy (which is too easy — annealing finds its optimum no matter
  // how it cools, so it would HIDE the failure mode we want to expose). No closed-form optimum
  // here, so we benchmark against a strong lower bound (nearest-neighbor + 2-opt heuristic).
  const n = 30;
  const cityRng = mulberry32(77);
  const cities: City[] = Array.from({ length: n }, () => ({ x: cityRng(), y: cityRng() }));
  const reference = nearestNeighborThen2opt(cities); // not provably optimal, but a tight baseline
  const steps = 60_000;

  // Same instance, same step budget, same start seed — ONLY the cooling rate differs, so any
  // gap is attributable to the schedule alone. Rates picked by sweeping over 8 seeds (averaged
  // tour length): 0.9999 is near the sweet spot, 0.99 is clearly too fast. (Cooling EVEN slower
  // than 0.9999 also degrades — T never cools enough within the budget — so the real lesson is
  // "match the schedule to the budget", but the canonical failure here is cooling too FAST.)
  //   SLOW (0.9999, half-life ≈ 6900 steps): T stays warm long enough to keep accepting uphill
  //     moves → escapes local optima → reaches reference-quality.
  //   FAST (0.99, half-life ≈ 70 steps): T collapses in a few hundred steps → almost only
  //     downhill moves survive → freezes in the first local minimum it stumbles into.
  // Average over several seeds, NOT one run: a single seed can be lucky and flip the verdict
  // (annealing is stochastic). The robust claim "slow beats fast" must survive averaging.
  const seeds = [202, 303, 404, 505, 606];
  const slowRuns = seeds.map((s) => annealTsp(cities, 1.0, 0.9999, steps, mulberry32(s)));
  const fastRuns = seeds.map((s) => annealTsp(cities, 1.0, 0.99, steps, mulberry32(s)));
  const slowAvg = mean(slowRuns.map((r) => r.bestLen));
  const fastAvg = mean(fastRuns.map((r) => r.bestLen));

  console.log(`基准 (NN + 2-opt 启发式)  = ${reference.toFixed(4)}  (强基线，非证明最优)`);
  console.log(`慢降温 (×0.9999) ${seeds.length} 种子均值 = ${slowAvg.toFixed(4)}  (相对基线 ${((slowAvg / reference - 1) * 100).toFixed(1)}%)`);
  console.log(`快降温 (×0.99)   ${seeds.length} 种子均值 = ${fastAvg.toFixed(4)}  (相对基线 ${((fastAvg / reference - 1) * 100).toFixed(1)}%)`);
  console.log(`两者差距 = ${((fastAvg - slowAvg) / slowAvg * 100).toFixed(1)}%（快降温更差）`);

  // Plot ONE representative seed's convergence trace for each (seed 202), so the reader sees the
  // shape; the verdict above rests on the seed-averaged numbers, not on this single trace.
  const slow = slowRuns[0];
  const fast = fastRuns[0];
  console.log(`\n慢降温收敛曲线 (seed=202，best-so-far 路径长，从乱序逐步逼近基准):`);
  console.log(asciiLine(slow.trace, 60, 10));
  console.log("\n快降温收敛曲线 (seed=202，过早冻结 → 停在更差的局部最优):");
  console.log(asciiLine(fast.trace, 60, 10));
  console.log("同一实例、同一步数、同一组种子，只改降温率 → 慢的稳赢。这就是退火失败模式。\n");
}

// ---------------------------------------------------------------------------
// ③ 2-D Ising model: sweep temperature, watch magnetization undergo a phase transition.
// ---------------------------------------------------------------------------

/**
 * Run Metropolis on an L×L Ising spin lattice at fixed temperature; return mean |magnetization|.
 * WHY |M| not M: below T_c the lattice picks one of two symmetric ground states (all-up or
 *   all-down) arbitrarily per run; signed M would average toward 0 across runs and HIDE the
 *   order. The absolute magnetization is the honest order parameter for a finite lattice.
 * Energy uses nearest-neighbor coupling with periodic (toroidal) boundaries so no spin is a
 *   special edge case. ΔE for a single flip is local: only its 4 neighbors matter.
 * INVARIANT: `equilSweeps` must be large enough that we measure the equilibrium ensemble,
 *   not the transient — undersampling here is the documented failure mode (see demo).
 */
function isingMagnetization(
  size: number,
  temperature: number,
  equilSweeps: number,
  measureSweeps: number,
  rng: Rng,
): number {
  const n = size * size;
  // Random ±1 initial spins. A "sweep" = n attempted single-spin flips.
  const spin = new Int8Array(n);
  for (let k = 0; k < n; k++) spin[k] = rng() < 0.5 ? -1 : 1;

  const idx = (r: number, c: number): number => ((r + size) % size) * size + ((c + size) % size);
  const localEnergyChange = (r: number, c: number): number => {
    const s = spin[idx(r, c)];
    const neighborSum =
      spin[idx(r - 1, c)] + spin[idx(r + 1, c)] + spin[idx(r, c - 1)] + spin[idx(r, c + 1)];
    // Flipping s costs ΔE = 2·s·Σneighbors (J=1). Positive ΔE means flip is unfavorable.
    return 2 * s * neighborSum;
  };

  const sweep = (): void => {
    for (let k = 0; k < n; k++) {
      const r = Math.floor(rng() * size);
      const c = Math.floor(rng() * size);
      const dE = localEnergyChange(r, c);
      // Accept the flip if it lowers energy, else with Boltzmann prob exp(-ΔE/T).
      if (dE <= 0 || rng() < Math.exp(-dE / temperature)) spin[idx(r, c)] = -spin[idx(r, c)] as -1 | 1;
    }
  };

  for (let s = 0; s < equilSweeps; s++) sweep(); // burn in to equilibrium FIRST
  const mags: number[] = [];
  for (let s = 0; s < measureSweeps; s++) {
    sweep();
    let sum = 0;
    for (let k = 0; k < n; k++) sum += spin[k];
    mags.push(Math.abs(sum) / n); // |M| per spin in [0,1]
  }
  return mean(mags);
}

function demoIsing(): void {
  console.log("--- ③ 2D 伊辛模型：扫温度看磁化的相变 ---");
  const size = 16; // 16×16 = 256 spins; big enough to show a transition, small enough to be fast
  const temps = [1.0, 1.5, 2.0, 2.2, 2.27, 2.4, 2.7, 3.2, 4.0];
  const tcTheory = 2 / Math.log(1 + Math.SQRT2); // Onsager's exact 2-D T_c ≈ 2.269
  const rng = mulberry32(2227);

  // 600 equilibration sweeps before measuring: near T_c the chain suffers "critical slowing
  // down" (correlation time diverges), so a short burn-in there reads a noisy non-equilibrium
  // value. 600 is enough to keep the curve monotone for a 16×16 lattice.
  const mags = temps.map((t) => isingMagnetization(size, t, 600, 300, rng));
  console.log(`格子 = ${size}×${size} = ${size * size} 自旋，每点 600 平衡 + 300 测量 sweep`);
  console.log(`Onsager 解析临界温度 T_c = ${tcTheory.toFixed(3)}`);
  console.log("\n温度 → 平均 |磁化| (序参量):");
  console.log(asciiBar(temps.map((t) => `T=${t}`), mags, 40));

  // Locate where magnetization falls through 0.5 — a crude finite-size T_c estimate.
  let crossT = NaN;
  for (let i = 1; i < temps.length; i++) {
    if (mags[i - 1] >= 0.5 && mags[i] < 0.5) {
      const frac = (mags[i - 1] - 0.5) / (mags[i - 1] - mags[i]); // linear interp
      crossT = temps[i - 1] + frac * (temps[i] - temps[i - 1]);
      break;
    }
  }
  console.log(`\n|磁化| 跌破 0.5 处 T ≈ ${Number.isNaN(crossT) ? "N/A" : crossT.toFixed(2)}  (有限格子，偏离 ${tcTheory.toFixed(2)} 是预期的有限尺寸效应)`);
  console.log("低温有序 (|M|≈1)，高温无序 (|M|≈0)，临界点附近骤变 —— 这就是相变。\n");

  // --- Failure mode: too few equilibration sweeps reads a NON-equilibrium magnetization. ---
  console.log("--- 失败模式：平衡 sweep 太少 → 读到非平衡值，伪造更平滑/更晚的相变 ---");
  const tTest = 2.0; // below T_c: true |M| should be clearly ordered
  const trials = 5;
  const fewVals: number[] = [];
  const manyVals: number[] = [];
  for (let k = 0; k < trials; k++) {
    fewVals.push(isingMagnetization(size, tTest, 2, 50, mulberry32(900 + k))); // barely equilibrated
    manyVals.push(isingMagnetization(size, tTest, 400, 200, mulberry32(900 + k))); // well equilibrated
  }
  console.log(`T=${tTest}, ${trials} 次独立运行:`);
  console.log(`  平衡=2   sweep: 均值 |M| = ${mean(fewVals).toFixed(3)}  抖动 std = ${std(fewVals).toFixed(3)}`);
  console.log(`  平衡=400 sweep: 均值 |M| = ${mean(manyVals).toFixed(3)}  抖动 std = ${std(manyVals).toFixed(3)}`);
  console.log("平衡不足时 |M| 被低估且 run 间剧烈抖动 —— 你会误以为相变更早、更软。等平衡再测。");
}

function main(): void {
  demoMetropolis();
  demoAnnealing();
  demoIsing();
}

main();
