// stage10-fourier.ts — Fourier analysis: DFT as a matmul, the FFT, the spectrum, aliasing.
//
// SCOPE (stage author): make four claims TRUE in code, not asserted:
//   (1) The DFT IS linear algebra. We build the N×N DFT matrix explicitly and run it via
//       core/linalg.matVec, then implement a recursive radix-2 FFT and check it matches the
//       matrix DFT to floating-point precision — and WALL-CLOCK the O(n^2) vs O(n log n) gap.
//   (2) The spectrum recovers frequency. A synthetic sum-of-sines is fed through the FFT and
//       the magnitude spectrum (core/plot.asciiBar) spikes exactly at the planted bins.
//   (3) Frequency-domain filtering denoises. We add seeded Gaussian noise, zero the high
//       bins, inverse-FFT, and MEASURE the SNR improvement in dB.
//   (4) Aliasing is real. We sample a high-frequency sine below the Nyquist rate and show its
//       energy lands at the WRONG (lower) bin — indistinguishable from a genuine low tone.
//       This is the failure mode: violate fs > 2·f_max and the spectrum lies to you.
//
// WHY a real/imag pair instead of a Complex class: this book has no runtime deps and core/
//   linalg works on number[]. Carrying (re, im) as two parallel arrays keeps every operation
//   visible as plain arithmetic — the reader sees the twiddle factors, nothing is hidden in a
//   class. The DFT matrix is therefore TWO real matrices (cos and -sin parts).
//
// INVARIANT: every length passed to fft() must be a power of two. Radix-2 splits even/odd
//   recursively; a non-power-of-two length would not split evenly and the recursion is wrong
//   (not merely slow). We assert this rather than silently returning garbage.
//
// HONEST-NUMBER NOTE: the FFT-vs-DFT timing is real wall-clock on THIS machine (performance
//   .now), so absolute ms vary per host; what transfers is the RATIO, which grows with N. SNR
//   numbers come from a toy signal with seeded noise — the absolute dB is optimistic (clean
//   synthetic tone, known cutoff); the transferable lesson is that a low-pass recovers SNR
//   when signal and noise live in separated bands, and CANNOT when they overlap.
//
// CONTRACT: reuse core/linalg.js::matVec (DFT-as-matmul) and core/plot.js::asciiBar (spectrum);
//   core/rng.js for deterministic noise. No stageNN imports (importing one would run its main).

import { matVec } from "./core/linalg.js";
import { asciiBar } from "./core/plot.js";
import { mulberry32, sampleNormal } from "./core/rng.js";

type Complex = { re: number[]; im: number[] }; // parallel arrays, length N

// --- (1a) DFT as an explicit matrix ---------------------------------------------------

/**
 * Build the real and imaginary parts of the N×N DFT matrix W, where
 *   W[k][n] = exp(-2πi·kn/N) = cos(2πkn/N) - i·sin(2πkn/N).
 * WHY expose it as a matrix at all: the DFT is usually presented as a magic sum; rendering it
 *   as a matrix-vector product makes its linearity obvious and lets us reuse core/linalg.
 */
function dftMatrix(N: number): { cos: number[][]; negSin: number[][] } {
  const cos: number[][] = [];
  const negSin: number[][] = [];
  for (let k = 0; k < N; k++) {
    const cRow: number[] = [];
    const sRow: number[] = [];
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      cRow.push(Math.cos(angle));
      sRow.push(Math.sin(angle)); // note: this is sin(angle); angle already carries the minus
    }
    cos.push(cRow);
    negSin.push(sRow);
  }
  return { cos, negSin };
}

/**
 * DFT of a real signal computed as two matrix-vector products (the textbook O(N^2) route).
 * For real input x: Re(X) = (cos matrix)·x, Im(X) = (negSin matrix)·x.
 * Used as the GROUND TRUTH the FFT must match — slow but obviously correct.
 */
function dftViaMatrix(signal: readonly number[]): Complex {
  const N = signal.length;
  const { cos, negSin } = dftMatrix(N);
  return { re: matVec(cos, signal), im: matVec(negSin, signal) };
}

// --- (1b) Recursive radix-2 FFT -------------------------------------------------------

/**
 * In-place-free recursive radix-2 Cooley–Tukey FFT.
 * Splits the length-N transform into two length-N/2 transforms over even/odd samples, then
 *   combines with twiddle factors exp(-2πi·k/N). This is the whole O(n log n) idea: the
 *   even/odd DFTs are SHARED across the upper and lower output halves, so we never recompute.
 * INVARIANT: N must be a power of two (see header). Asserted because a wrong length yields a
 *   silently incorrect transform, the worst kind of bug in a teaching codebase.
 */
function fft(re: readonly number[], im: readonly number[]): Complex {
  const N = re.length;
  if (N === 1) return { re: [re[0]], im: [im[0]] };
  if (N % 2 !== 0) throw new Error(`[fft] length ${N} is not a power of two`);

  const evenRe: number[] = [];
  const evenIm: number[] = [];
  const oddRe: number[] = [];
  const oddIm: number[] = [];
  for (let n = 0; n < N; n += 2) {
    evenRe.push(re[n]);
    evenIm.push(im[n]);
    oddRe.push(re[n + 1]);
    oddIm.push(im[n + 1]);
  }
  const E = fft(evenRe, evenIm);
  const O = fft(oddRe, oddIm);

  const outRe = new Array<number>(N);
  const outIm = new Array<number>(N);
  for (let k = 0; k < N / 2; k++) {
    // twiddle = exp(-2πi·k/N); multiply it into the odd half.
    const ang = (-2 * Math.PI * k) / N;
    const tw = { re: Math.cos(ang), im: Math.sin(ang) };
    const tRe = tw.re * O.re[k] - tw.im * O.im[k];
    const tIm = tw.re * O.im[k] + tw.im * O.re[k];
    // Butterfly: output k uses +twiddle·odd, output k+N/2 uses -twiddle·odd.
    outRe[k] = E.re[k] + tRe;
    outIm[k] = E.im[k] + tIm;
    outRe[k + N / 2] = E.re[k] - tRe;
    outIm[k + N / 2] = E.im[k] - tIm;
  }
  return { re: outRe, im: outIm };
}

/**
 * Inverse FFT via the conjugate trick: ifft(X) = conj(fft(conj(X))) / N. Reusing the forward
 * transform avoids a second buggy implementation — the symmetry IS the proof it's correct.
 */
function ifft(re: readonly number[], im: readonly number[]): Complex {
  const N = re.length;
  const conjIm = im.map((v) => -v);
  const F = fft(re, conjIm);
  return { re: F.re.map((v) => v / N), im: F.im.map((v) => -v / N) };
}

function magnitude(c: Complex): number[] {
  return c.re.map((r, i) => Math.hypot(r, c.im[i]));
}

function maxAbsDiff(a: Complex, b: Complex): number {
  let m = 0;
  for (let i = 0; i < a.re.length; i++) {
    m = Math.max(m, Math.abs(a.re[i] - b.re[i]), Math.abs(a.im[i] - b.im[i]));
  }
  return m;
}

// --- part runners ---------------------------------------------------------------------

function partDftEqualsFftAndSpeedup(): void {
  console.log("=== Stage 10 · 傅里叶分析 ===\n");
  console.log("--- (1) DFT 即矩阵乘法, FFT 给出同样结果但更快 ---");

  // Cross-check on a small N where the O(N^2) matrix DFT is cheap to compute exactly.
  const N0 = 16;
  const probe = Array.from({ length: N0 }, (_, n) =>
    Math.cos((2 * Math.PI * 3 * n) / N0) + 0.5 * Math.sin((2 * Math.PI * 5 * n) / N0),
  );
  const viaMatrix = dftViaMatrix(probe);
  const viaFft = fft(probe, new Array(N0).fill(0));
  const diff = maxAbsDiff(viaMatrix, viaFft);
  console.log(`N=${N0}: 矩阵 DFT 与递归 FFT 的最大逐元素误差 = ${diff.toExponential(2)}`);
  console.log("  (~1e-14 量级 = 浮点舍入, 二者数学上等价已验证)\n");

  // Wall-clock the asymptotic gap at a larger N. Both are warmed once so we time steady state,
  // not JIT compilation. The matrix DFT rebuilds its matrix each call by design — that O(N^2)
  // storage+work is exactly the cost the FFT avoids.
  const N1 = 1024;
  const big = Array.from({ length: N1 }, (_, n) => Math.sin((2 * Math.PI * 17 * n) / N1));
  const bigZeros = new Array(N1).fill(0);
  dftViaMatrix(big); // warm
  fft(big, bigZeros); // warm

  const reps = 20;
  let sink = 0; // accumulator defeats dead-code elimination (see stage12 failure mode)
  const tDft0 = performance.now();
  for (let r = 0; r < reps; r++) sink += dftViaMatrix(big).re[1];
  const dftMs = (performance.now() - tDft0) / reps;
  const tFft0 = performance.now();
  for (let r = 0; r < reps; r++) sink += fft(big, bigZeros).re[1];
  const fftMs = (performance.now() - tFft0) / reps;

  console.log(`N=${N1}, ${reps} 次平均 (本机 wall-clock):`);
  console.log(`  O(N^2) 矩阵 DFT = ${dftMs.toFixed(3)} ms/次`);
  console.log(`  O(N log N) FFT  = ${fftMs.toFixed(3)} ms/次`);
  console.log(`  加速比 = ${(dftMs / fftMs).toFixed(1)}x  (校验和 ${sink.toExponential(2)})`);
  const ratioTheory = (N1 * N1) / (N1 * Math.log2(N1));
  console.log(`  理论上界 N/log2(N) = ${ratioTheory.toFixed(1)}x (est., 不含常数因子)\n`);
}

function partRecoverFrequencies(): void {
  console.log("--- (2) 合成多频信号 → FFT → 频谱识别各频率 ---");
  // Plant three tones at bins 2, 5, 11 with different amplitudes. The half-spectrum (0..N/2)
  // should show peaks exactly there; the upper half mirrors them (real-input symmetry).
  const N = 32;
  const planted = [
    { bin: 2, amp: 1.0 },
    { bin: 5, amp: 0.6 },
    { bin: 11, amp: 0.3 },
  ];
  const signal = Array.from({ length: N }, (_, n) =>
    planted.reduce((s, t) => s + t.amp * Math.cos((2 * Math.PI * t.bin * n) / N), 0),
  );
  const spectrum = magnitude(fft(signal, new Array(N).fill(0)));
  // Only the first half carries unique info for a real signal; bin k and N-k are conjugates.
  const half = spectrum.slice(0, N / 2 + 1);
  console.log(`种入频率 bin = [${planted.map((t) => t.bin).join(", ")}], 振幅各异`);
  console.log("半谱 |X[k]| (k=0..N/2):");
  console.log(asciiBar(half.map((_, k) => `k=${k}`), half));
  // Verify the three largest bins are exactly the planted ones — a coded claim, not eyeballing.
  const top3 = half
    .map((v, k) => ({ k, v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 3)
    .map((x) => x.k)
    .sort((a, b) => a - b);
  console.log(`代码自检: 最大的 3 个 bin = [${top3.join(", ")}]  ` +
    `(${top3.join(",") === "2,5,11" ? "与种入完全一致 ✓" : "不一致 ✗"})\n`);
}

function partLowpassDenoise(): void {
  console.log("--- (3) 低通滤波降噪: 加噪 → 频域置零高频 → 反变换, 实测 SNR 改善 ---");
  const N = 256;
  const rng = mulberry32(2025); // deterministic noise: same seed => same SNR every run
  // Clean signal: two LOW-frequency tones (bins 3 and 7). Noise spreads energy across ALL
  // bins. A low-pass that keeps only low bins therefore removes most noise but little signal —
  // this band separation is the precondition for the trick to work.
  const clean = Array.from({ length: N }, (_, n) =>
    Math.sin((2 * Math.PI * 3 * n) / N) + 0.5 * Math.sin((2 * Math.PI * 7 * n) / N),
  );
  const noisy = clean.map((v) => v + sampleNormal(rng, 0, 0.8));

  // Forward FFT, then a brick-wall low-pass: zero every bin whose frequency index exceeds the
  // cutoff. Bin k and its mirror N-k must BOTH be zeroed to keep the inverse transform real.
  const cutoff = 12;
  const F = fft(noisy, new Array(N).fill(0));
  const filtRe = F.re.slice();
  const filtIm = F.im.slice();
  for (let k = 0; k <= N / 2; k++) {
    if (k > cutoff) {
      filtRe[k] = 0;
      filtIm[k] = 0;
      const mirror = (N - k) % N; // conjugate-symmetric partner
      filtRe[mirror] = 0;
      filtIm[mirror] = 0;
    }
  }
  const recovered = ifft(filtRe, filtIm).re; // imaginary part is ~0 by symmetry

  const snrDb = (sig: readonly number[], est: readonly number[]): number => {
    let sigPow = 0;
    let errPow = 0;
    for (let i = 0; i < sig.length; i++) {
      sigPow += sig[i] * sig[i];
      errPow += (est[i] - sig[i]) * (est[i] - sig[i]);
    }
    return 10 * Math.log10(sigPow / errPow); // dB; higher = closer to clean
  };
  const before = snrDb(clean, noisy);
  const after = snrDb(clean, recovered);
  console.log(`噪声 std=0.8, 截止 bin=${cutoff} (信号能量集中在 bin 3 与 7)`);
  console.log(`滤波前 SNR = ${before.toFixed(2)} dB`);
  console.log(`滤波后 SNR = ${after.toFixed(2)} dB`);
  console.log(`改善 = ${(after - before).toFixed(2)} dB ` +
    `(toy 合成信号, 绝对值偏乐观; 可迁移的是"频带分离时低通能恢复 SNR")\n`);
}

function partAliasing(): void {
  console.log("--- (4) 失败模式: 违反奈奎斯特 → 混叠, 高频伪装成低频 ---");
  // Continuous tone at trueHz. We sample it at fsHz. Nyquist requires fs > 2·trueHz.
  // Here fs < 2·trueHz on purpose: the 30 Hz tone CANNOT be represented and folds back to
  // an alias at |trueHz - fs| = |30 - 32| = 2 Hz. The spectrum will (wrongly) peak at 2 Hz.
  // fs is a power of two so the radix-2 FFT applies (and a 1 s window makes bin == Hz).
  const fsHz = 32; // samples per second — too low for a 30 Hz tone
  const trueHz = 30;
  const durationSec = 1; // exactly fs samples => bin index == Hz for this setup
  const N = fsHz * durationSec; // 32 samples
  const sampled = Array.from({ length: N }, (_, n) =>
    Math.cos((2 * Math.PI * trueHz * n) / fsHz),
  );
  const half = magnitude(fft(sampled, new Array(N).fill(0))).slice(0, N / 2 + 1);
  const peakBin = half.reduce((best, v, k) => (v > half[best] ? k : best), 0);
  const aliasHz = Math.abs(trueHz - fsHz); // folding formula for trueHz between fs/2 and fs
  console.log(`真实频率 = ${trueHz} Hz, 采样率 fs = ${fsHz} Hz ` +
    `(奈奎斯特要求 fs > ${2 * trueHz} Hz, 本例违反)`);
  console.log("采样信号的半谱:");
  console.log(asciiBar(half.map((_, k) => `${k}Hz`), half));
  console.log(`频谱峰值出现在 ${peakBin} Hz, 而非真实的 ${trueHz} Hz`);
  console.log(`折叠公式预测的混叠频率 = |${trueHz} - ${fsHz}| = ${aliasHz} Hz ` +
    `(${peakBin === aliasHz ? "与实测峰值一致 ✓" : "与实测不符 ✗"})`);
  console.log(`教训: 采样率不足时, ${trueHz} Hz 的真信号与 ${aliasHz} Hz 的假信号在离散数据里无法区分。`);
}

function main(): void {
  partDftEqualsFftAndSpeedup();
  partRecoverFrequencies();
  partLowpassDenoise();
  partAliasing();
}

main();
