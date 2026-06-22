// stage03-conv-backward.ts — Backprop through convolution, and WHY col2im must SUM.
//
// THE CHAPTER'S ONE IDEA: the conv-input gradient (dInput) is col2im — the transpose of
//   the im2col unfold. Because overlapping sliding windows READ the same input pixel
//   multiple times in the forward pass, the chain rule says their gradient contributions
//   must be SUMMED back onto that pixel. Writing the scatter as `grad[pixel] = ...`
//   (assignment) instead of `grad[pixel] += ...` (accumulation) keeps only the LAST
//   window's contribution and silently drops the rest.
//
// WHY THIS BUG IS DANGEROUS (the teaching payload): it is stride-dependent.
//   - stride < kernel  -> windows OVERLAP -> a pixel is hit by >1 window -> `=` is WRONG.
//   - stride >= kernel -> windows TILE without overlap -> every pixel hit exactly once ->
//                          `=` and `+=` are INDISTINGUISHABLE -> gradCheck PASSES.
//   So a test suite that only exercises stride==kernel (a very common default for the
//   "downsampling conv" pattern) would give this bug a clean bill of health. We reproduce
//   exactly that false-PASS, then expose the bug by switching to stride 1.
//
// WHAT WE VERIFY (honest numbers, all computed at runtime, none asserted):
//   - dInput and dWeight from a hand-written conv backward vs central finite differences,
//     reported as the MAX RELATIVE error per tensor (PASS if < 1e-5).
//   - the per-pixel overlap-count map (how many windows touch each input pixel).
//   - the broken (`=`) backward's error: large under stride 1, ~machine-zero under stride 3.
//
// HONESTY / SCOPE: inputs are tiny seeded random tensors (1x2x5x5, kernel 3x2x3x3) in f64.
//   Absolute error magnitudes are not "a real framework's" — f64 + central differences make
//   a correct op land near 1e-9 here. The transferable signal is the CONTRAST: correct
//   backward ~1e-9, broken backward O(1) on overlap, ~1e-12 without overlap.
//
// We deliberately DO NOT import core's conv2d backward (it is already correct, via +=).
//   To show both the right and the wrong scatter we hand-write the backward here so the one
//   line that matters (`+=` vs `=`) is visible and swappable. core.im2col is reused for the
//   FORWARD only; the manual dWeight reuses core matmul-style logic inline.

import { Tensor, im2col, convOutSize, type Conv2dParams } from "./core/autograd.js";
import { mulberry32, randn, type Rng } from "./core/rng.js";

// ----------------------------------------------------------------------------
// A self-contained conv forward + backward, written so the col2im scatter mode
// (sum vs overwrite) is a single explicit knob. This mirrors core's composition
// (im2col -> matmul) but keeps the gradient code local and inspectable.
// ----------------------------------------------------------------------------

type ScatterMode = "sum" | "overwrite";

interface ConvBackward {
  dInput: Float64Array; // shape N*C*H*W, gradient w.r.t. input
  dWeight: Float64Array; // shape outC*inC*kH*kW, gradient w.r.t. weight
}

/** Forward conv via im2col + manual matmul. Returns flat output (N*outC*outH*outW) plus the
 *  cols matrix so backward can reuse it (cols are the dWeight Jacobian factor). */
function convForward(
  input: Tensor,
  weight: Tensor,
  p: Conv2dParams,
): { out: Float64Array; outH: number; outW: number; cols: Tensor } {
  const [outC, inC, kH, kW] = weight.shape;
  const [, , H, W] = input.shape;
  const N = input.shape[0];
  const outH = convOutSize(H, kH, p.stride, p.padding);
  const outW = convOutSize(W, kW, p.stride, p.padding);
  const cols = im2col(input, kH, kW, p); // (N*outH*outW, inC*kH*kW)
  const patch = inC * kH * kW;
  const rows = N * outH * outW;
  // out (pixel-major): row r, channel oc = sum_p cols[r,p] * weight[oc,p]
  const out = new Float64Array(rows * outC);
  for (let r = 0; r < rows; r++) {
    for (let oc = 0; oc < outC; oc++) {
      let acc = 0;
      for (let pp = 0; pp < patch; pp++) acc += cols.data[r * patch + pp] * weight.data[oc * patch + pp];
      out[r * outC + oc] = acc;
    }
  }
  return { out, outH, outW, cols };
}

/**
 * Backward for loss = sum(output). With L = sum over all outputs, dL/d(out[r,oc]) = 1 for
 * every output element, so dCols = ones @ weight^T and dWeight = ones^T @ cols. The only
 * conv-specific step is col2im: scatter dCols back to the input pixels.
 *
 * `mode` selects the scatter operator:
 *   "sum"       -> input.grad[pixel] += contribution   (CORRECT)
 *   "overwrite" -> input.grad[pixel]  = contribution   (THE BUG: keeps last window only)
 *
 * INVARIANT this function exists to test: under "sum", dInput equals numerical FD to f64
 *   precision for ANY stride; under "overwrite" it equals FD ONLY when stride >= kernel.
 */
function convBackwardSumLoss(
  input: Tensor,
  weight: Tensor,
  p: Conv2dParams,
  cols: Tensor,
  outH: number,
  outW: number,
  mode: ScatterMode,
): ConvBackward {
  const [outC, inC, kH, kW] = weight.shape;
  const [, , H, W] = input.shape;
  const N = input.shape[0];
  const { stride, padding } = p;
  const patch = inC * kH * kW;
  const rows = N * outH * outW;

  // dWeight[oc, pp] = sum_r dOut[r,oc] * cols[r,pp]; dOut is all-ones (loss = sum).
  const dWeight = new Float64Array(outC * patch);
  for (let oc = 0; oc < outC; oc++) {
    for (let pp = 0; pp < patch; pp++) {
      let acc = 0;
      for (let r = 0; r < rows; r++) acc += cols.data[r * patch + pp]; // dOut[r,oc] == 1
      dWeight[oc * patch + pp] = acc;
    }
  }

  // dCols[r, pp] = sum_oc dOut[r,oc] * weight[oc,pp] = sum_oc weight[oc,pp] (dOut == 1).
  // Precompute the per-column weight-sum once: it is identical for every row r.
  const colGradPerPatch = new Float64Array(patch);
  for (let pp = 0; pp < patch; pp++) {
    let acc = 0;
    for (let oc = 0; oc < outC; oc++) acc += weight.data[oc * patch + pp];
    colGradPerPatch[pp] = acc;
  }

  // col2im: walk the SAME index map im2col used in forward, scatter dCols to source pixels.
  const dInput = new Float64Array(N * inC * H * W);
  for (let n = 0; n < N; n++) {
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let col = 0;
        for (let c = 0; c < inC; c++) {
          for (let ky = 0; ky < kH; ky++) {
            const iy = oy * stride + ky - padding;
            for (let kx = 0; kx < kW; kx++) {
              const ix = ox * stride + kx - padding;
              if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                const dst = ((n * inC + c) * H + iy) * W + ix;
                const g = colGradPerPatch[col];
                // THE ONE LINE THAT MATTERS. += accumulates contributions from every window
                // that reads this pixel; = throws away all but the last writer.
                if (mode === "sum") dInput[dst] += g;
                else dInput[dst] = g;
              }
              col++;
            }
          }
        }
      }
    }
  }
  return { dInput, dWeight };
}

/** Recompute the scalar loss = sum(conv(input, weight)) from current tensor .data — the
 *  fresh-forward closure central finite differences need. */
function sumLoss(input: Tensor, weight: Tensor, p: Conv2dParams): number {
  const { out } = convForward(input, weight, p);
  let s = 0;
  for (let i = 0; i < out.length; i++) s += out[i];
  return s;
}

/**
 * Central finite-difference gradient for every element of `target.data`, holding everything
 * else fixed. Returns a flat array parallel to target.data. Central differences are O(eps^2),
 * so with f64 a correct analytic grad matches to ~1e-9; a wrong one stands out by orders.
 */
function numericalGrad(
  target: Tensor,
  loss: () => number,
  eps = 1e-5,
): Float64Array {
  const g = new Float64Array(target.size);
  for (let i = 0; i < target.size; i++) {
    const orig = target.data[i];
    target.data[i] = orig + eps;
    const lp = loss();
    target.data[i] = orig - eps;
    const lm = loss();
    target.data[i] = orig; // restore; a leaked perturbation corrupts later elements
    g[i] = (lp - lm) / (2 * eps);
  }
  return g;
}

/** Max relative error between analytic and numerical grads, element by element. Same denom
 *  convention as core gradCheck so the numbers are comparable to _smoke's. */
function maxRelError(analytic: ArrayLike<number>, numeric: ArrayLike<number>): number {
  let mx = 0;
  for (let i = 0; i < analytic.length; i++) {
    const denom = Math.max(1e-8, Math.abs(analytic[i]) + Math.abs(numeric[i]));
    const rel = Math.abs(analytic[i] - numeric[i]) / denom;
    if (rel > mx) mx = rel;
  }
  return mx;
}

function randTensor(shape: number[], rng: Rng): Tensor {
  return Tensor.from(shape, () => randn(rng));
}

/**
 * Count, per input pixel, how many sliding windows read it (= how many gradient
 * contributions col2im must sum onto it). This is exactly the multiplicity that the
 * overwrite bug discards down to 1. Returns a flat H*W map for channel 0 of image 0
 * (all channels/images share the same spatial pattern).
 */
function overlapCountMap(H: number, W: number, kH: number, kW: number, p: Conv2dParams): Int32Array {
  const outH = convOutSize(H, kH, p.stride, p.padding);
  const outW = convOutSize(W, kW, p.stride, p.padding);
  const counts = new Int32Array(H * W);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      for (let ky = 0; ky < kH; ky++) {
        const iy = oy * p.stride + ky - p.padding;
        for (let kx = 0; kx < kW; kx++) {
          const ix = ox * p.stride + kx - p.padding;
          if (iy >= 0 && iy < H && ix >= 0 && ix < W) counts[iy * W + ix]++;
        }
      }
    }
  }
  return counts;
}

function printCountMap(counts: Int32Array, H: number, W: number): void {
  for (let y = 0; y < H; y++) {
    let line = "    ";
    for (let x = 0; x < W; x++) line += String(counts[y * W + x]).padStart(3);
    console.log(line);
  }
}

function main(): void {
  const rng = mulberry32(20240607);

  // Fixed problem: 1 image, 2 in-channels, 5x5; kernel 3 out, 2 in, 3x3.
  const N = 1;
  const inC = 2;
  const H = 5;
  const W = 5;
  const outC = 3;
  const kH = 3;
  const kW = 3;
  const x = randTensor([N, inC, H, W], rng);
  const w = randTensor([outC, inC, kH, kW], rng);

  console.log("=== stage03: conv backward — col2im must SUM overlaps ===");
  console.log(`  input ${N}x${inC}x${H}x${W}, kernel ${outC}x${inC}x${kH}x${kW}, loss = sum(conv(x,w))`);

  // ---------------------------------------------------------------------------
  // PART 1 — correct backward (sum), padded stride-1 conv (windows DO overlap).
  // ---------------------------------------------------------------------------
  const pSame: Conv2dParams = { stride: 1, padding: 1 }; // outH=outW=5, heavy overlap
  {
    const fwd = convForward(x, w, pSame);
    const bw = convBackwardSumLoss(x, w, pSame, fwd.cols, fwd.outH, fwd.outW, "sum");
    const dInputNum = numericalGrad(x, () => sumLoss(x, w, pSame));
    const dWeightNum = numericalGrad(w, () => sumLoss(x, w, pSame));
    const eInput = maxRelError(bw.dInput, dInputNum);
    const eWeight = maxRelError(bw.dWeight, dWeightNum);
    const TOL = 1e-5;
    console.log("\n[PART 1] correct col2im (+=), stride 1 pad 1 (overlapping windows)");
    console.log(`  dInput  max rel error = ${eInput.toExponential(2)}  ${eInput < TOL ? "PASS" : "FAIL"}`);
    console.log(`  dWeight max rel error = ${eWeight.toExponential(2)}  ${eWeight < TOL ? "PASS" : "FAIL"}`);
  }

  // ---------------------------------------------------------------------------
  // PART 2 — the overlap-count map: WHY summation is mandatory here.
  // With 3x3 kernel, stride 1, pad 1 over 5x5, interior pixels are read by up to
  // 9 windows; a corner by 4. Overwrite would collapse every one of these to 1.
  // ---------------------------------------------------------------------------
  {
    const counts = overlapCountMap(H, W, kH, kW, pSame);
    let mn = Infinity;
    let mx = -Infinity;
    let sum = 0;
    for (let i = 0; i < counts.length; i++) {
      mn = Math.min(mn, counts[i]);
      mx = Math.max(mx, counts[i]);
      sum += counts[i];
    }
    console.log("\n[PART 2] overlap-count map (windows reading each input pixel, stride 1 pad 1):");
    printCountMap(counts, H, W);
    console.log(`  per-pixel reads: min=${mn} max=${mx} mean=${(sum / counts.length).toFixed(2)}`);
    console.log(`  overwrite (=) would force every one of these counts down to 1 (last writer wins).`);
  }

  // ---------------------------------------------------------------------------
  // PART 3 — THE FAILURE MODE. Same broken backward (overwrite) under two strides.
  //   stride 1 (overlap) -> dInput grossly wrong -> FAIL.
  //   stride 3 == kernel (no overlap) -> dInput exact -> PASS.
  // dWeight is identical in both modes (it doesn't go through col2im), so the bug is
  // input-gradient-only and stride-gated — the kind that hides for months.
  // ---------------------------------------------------------------------------
  console.log("\n[PART 3] FAILURE MODE — overwrite (=) col2im, stride-gated bug");

  const broken = (p: Conv2dParams, label: string, overlapping: boolean) => {
    const fwd = convForward(x, w, p);
    const bw = convBackwardSumLoss(x, w, p, fwd.cols, fwd.outH, fwd.outW, "overwrite");
    const dInputNum = numericalGrad(x, () => sumLoss(x, w, p));
    const eInput = maxRelError(bw.dInput, dInputNum);
    const TOL = 1e-5;
    const verdict = eInput < TOL ? "PASS" : "FAIL";
    const expectFail = overlapping ? "(expected FAIL: windows overlap)" : "(expected PASS: windows tile, no overlap)";
    console.log(
      `  ${label.padEnd(22)} dInput max rel error = ${eInput.toExponential(2)}  ${verdict}  ${expectFail}`,
    );
    return eInput;
  };

  // stride 1, pad 1 over 5x5: outH=outW=5, windows overlap heavily.
  const eBrokenOverlap = broken({ stride: 1, padding: 1 }, "stride 1 (overlap)", true);
  // stride 3 == kernel, no pad, over 5x5: outH=outW=convOutSize(5,3,3,0)=1, single window,
  // so NO pixel is read twice -> overwrite happens to equal sum. We use a 6x6 input below to
  // make the "tiling, multiple windows, still no overlap" case unmistakable.
  const xTile = randTensor([N, inC, 6, 6], rng);
  const wTile = w; // reuse 3x3 kernel
  const pTile: Conv2dParams = { stride: 3, padding: 0 }; // 6x6 -> 2x2 windows, perfectly tiled
  {
    const fwd = convForward(xTile, wTile, pTile);
    const bwOK = convBackwardSumLoss(xTile, wTile, pTile, fwd.cols, fwd.outH, fwd.outW, "sum");
    const bwBad = convBackwardSumLoss(xTile, wTile, pTile, fwd.cols, fwd.outH, fwd.outW, "overwrite");
    const num = numericalGrad(xTile, () => sumLoss(xTile, wTile, pTile));
    const eOK = maxRelError(bwOK.dInput, num);
    const eBad = maxRelError(bwBad.dInput, num);
    const counts = overlapCountMap(6, 6, kH, kW, pTile);
    let maxCount = 0;
    for (let i = 0; i < counts.length; i++) maxCount = Math.max(maxCount, counts[i]);
    console.log(
      `  stride 3 == kernel     dInput max rel error = ${eBad.toExponential(2)}  ${eBad < 1e-5 ? "PASS" : "FAIL"}  (expected PASS: max per-pixel reads = ${maxCount})`,
    );
    console.log(`    (correct += backward here: ${eOK.toExponential(2)} — identical, because no pixel is read twice)`);

    console.log("\n  VERDICT:");
    console.log(`    overwrite under overlap (stride 1): rel err ${eBrokenOverlap.toExponential(2)}  -> bug EXPOSED`);
    console.log(`    overwrite under no-overlap (stride 3): rel err ${eBad.toExponential(2)}  -> bug HIDDEN`);
    console.log("    A test suite using only stride==kernel would ship this bug silently.");
  }

  console.log("\n=== stage03 done ===");
}

main();
