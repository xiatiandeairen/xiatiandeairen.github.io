// stage04-receptive-field.ts — How big a patch of the ORIGINAL image does one deep
//   neuron actually look at? ("Receptive field", RF.)
//
// WHY this chapter matters: people reason about CNNs as if a unit in layer L "sees the
//   whole image". It does not — early on it sees a tiny window, and that window grows with
//   depth and (multiplicatively) with stride. Misjudging the RF is how you build a detector
//   that physically cannot see an object larger than its field, or a classifier whose top
//   neuron only ever looked at a 7x7 corner. So we make the RF a MEASURED quantity, not a
//   recited formula.
//
// THREE THINGS THIS STAGE DOES, each cross-checking the next:
//   1. THEORY: print the textbook RF recurrence layer by layer (1 -> 3 -> 5 -> 7).
//   2. EMPIRICAL: measure the RF two independent ways and confirm they equal the theory:
//        (a) forward all-ones impulse — a single bright input pixel; the non-zero footprint
//            it produces in a feature map IS the RF (every output pixel whose field covers
//            the impulse lights up). Rendered as an ASCII heatmap.
//        (b) gradient impulse — backprop a 1 into ONE chosen deep unit; the non-zero input
//            gradient region is EXACTLY the set of input pixels that unit can see. This is
//            the rigorous definition and (unlike forward) survives stride/downsampling,
//            because it lands back in input coordinates directly.
//   3. SHAPE: swap the all-ones kernel for random Gaussian weights and measure the
//        EFFECTIVE RF — the gradient magnitude is not a flat box but decays toward the
//        edges (center pixels feed more paths), the well-known "Gaussian-ish ERF".
//
// FAILURE MODE (the headline of the chapter): insert a stride-2 layer but forget that
//   stride MULTIPLIES the jump in the RF recurrence. The naive recurrence (jump kept at 1)
//   under-reports the deep RF badly; the gradient impulse reveals the true, larger field.
//   We print naive-vs-correct-vs-measured side by side so the gap is undeniable.
//
// DETERMINISM: all randomness threads a single seeded Rng (mulberry32). Re-running prints
//   identical numbers — the honesty contract for every figure below.
//
// HONESTY: the RF integers and the gradient footprints are computed by the real conv2d /
//   autograd in core (the same ops _smoke.ts grad-checks at <1e-8). Nothing here is an
//   asserted constant; the "[MATCH]" tags compare the formula against a fresh measurement.

import { Tensor, conv2d, type Conv2dParams } from "./core/autograd.js";
import { mulberry32, randn, type Rng } from "./core/rng.js";
import { heatmapAscii, receptiveFieldAscii } from "./core/metrics.js";

// A layer spec for the toy stacks: square kernel + stride, NO padding (padding shifts the
// RF center but not its size, so we drop it to keep the size arithmetic clean).
interface LayerSpec {
  kernel: number;
  stride: number;
}

// ---------------------------------------------------------------------------
// Theory: the receptive-field recurrence.
// ---------------------------------------------------------------------------
//
// Two coupled quantities accumulate down the stack:
//   jump   j_l = j_{l-1} * stride_l        (input pixels between adjacent output units)
//   field  r_l = r_{l-1} + (kernel_l - 1) * j_{l-1}
// with j_0 = 1, r_0 = 1 (one input pixel sees one input pixel).
//
// THE WHOLE POINT: the (kernel-1) term is scaled by the *previous* jump, so once a stride
//   has inflated the jump, every later layer's kernel reaches that many times further. Drop
//   the jump (the `useJump=false` branch) and you compute the classic underestimate.

interface RfStep {
  layer: number;
  jumpBefore: number;
  field: number;
}

function computeRfTrace(layers: LayerSpec[], useJump: boolean): RfStep[] {
  const trace: RfStep[] = [{ layer: 0, jumpBefore: 1, field: 1 }];
  let jump = 1;
  let field = 1;
  for (let i = 0; i < layers.length; i++) {
    const { kernel, stride } = layers[i];
    const jumpBefore = jump;
    // The bug we demo: when useJump=false we pretend every step is jump=1, i.e. we forget
    // that a previous stride spreads this kernel's reach. Correct math uses jumpBefore.
    field = field + (kernel - 1) * (useJump ? jumpBefore : 1);
    jump = jump * stride; // jump only ever GROWS through stride; =1 forever if all strides 1
    trace.push({ layer: i + 1, jumpBefore: jump, field });
  }
  return trace;
}

function finalField(layers: LayerSpec[], useJump: boolean): number {
  const t = computeRfTrace(layers, useJump);
  return t[t.length - 1].field;
}

// ---------------------------------------------------------------------------
// Building a conv stack to forward/backward through (no nn.Module needed — we drive the
// raw conv2d op directly so the weights are whatever we choose: all-ones or Gaussian).
// ---------------------------------------------------------------------------

// All-ones single-channel kernel (1,1,k,k). WHY all-ones: a positive kernel can't cancel,
// so "did this output pixel get ANY contribution from the impulse?" reduces to "is it
// non-zero?" — a clean footprint with no accidental zeros from sign cancellation.
function onesKernel(kernel: number): Tensor {
  return Tensor.fill([1, 1, kernel, kernel], 1);
}

function gaussianKernel(kernel: number, rng: Rng): Tensor {
  return Tensor.from([1, 1, kernel, kernel], () => randn(rng));
}

// Forward a single-channel NCHW input through a list of (weight, params) layers.
// Returns every intermediate activation (including the input at index 0) so callers can
// inspect any layer's footprint. Pure w.r.t. the passed tensors aside from building graph.
function forwardStack(
  input: Tensor,
  weights: Tensor[],
  params: Conv2dParams[],
): Tensor[] {
  const acts: Tensor[] = [input];
  let cur = input;
  for (let i = 0; i < weights.length; i++) {
    cur = conv2d(cur, weights[i], null, params[i]);
    acts.push(cur);
  }
  return acts;
}

// Bounding box of strictly-non-zero entries in a single-image single-channel map (1,1,H,W),
// using |v| > tol to ignore floating-point dust. Returns null if the map is all (near) zero.
function nonzeroBox(
  map: Float64Array,
  H: number,
  W: number,
  tol = 1e-12,
): { y0: number; y1: number; x0: number; x1: number; count: number } | null {
  let y0 = H,
    y1 = -1,
    x0 = W,
    x1 = -1,
    count = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (Math.abs(map[y * W + x]) > tol) {
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        count++;
      }
    }
  }
  if (count === 0) return null;
  // y1/x1 are inclusive maxima; return half-open [y0, y1+1) to match receptiveFieldAscii.
  return { y0, y1: y1 + 1, x0, x1: x1 + 1, count };
}

// Measure the RF of ONE chosen output unit by the gradient-impulse method: seed a 1 into
// that single unit's grad and run backward; every input pixel with non-zero grad is, by the
// definition of the derivative, a pixel that can change that unit — i.e. its receptive
// field. WHY this is the gold standard: it is exact, lives in input coordinates (immune to
// stride downsampling), and reuses the grad-checked autograd rather than re-deriving index
// math by hand. INVARIANT: caller must zeroGrad the whole graph first (we do).
function measureRfByGradient(
  input: Tensor,
  weights: Tensor[],
  params: Conv2dParams[],
  pick: (outH: number, outW: number) => { oy: number; ox: number },
): { box: ReturnType<typeof nonzeroBox>; outH: number; outW: number } {
  input.zeroGrad();
  for (const w of weights) w.zeroGrad();
  const acts = forwardStack(input, weights, params);
  const out = acts[acts.length - 1];
  const [, , outH, outW] = out.shape;
  // Non-scalar output: seed grad manually then drive the topo via a scalar proxy. Simplest
  // correct route: build a scalar = out * mask summed, but we just hand-seed and call the
  // op chain's backward by summing the single unit. We multiply the picked unit into a sum.
  const { oy, ox } = pick(outH, outW);
  // Select the single unit as a scalar loss so Tensor.backward() (which requires a [1] loss)
  // applies: loss = out[oy,ox]. We realize it by elementwise-mul with a one-hot then sum.
  const oneHot = Tensor.zeros(out.shape);
  oneHot.data[oy * outW + ox] = 1; // (1,1,outH,outW) -> flat index oy*outW+ox
  const loss = out.mul(oneHot).sum();
  loss.backward();
  return { box: nonzeroBox(input.grad, input.shape[2], input.shape[3]), outH, outW };
}

// ---------------------------------------------------------------------------
// Pretty-printers
// ---------------------------------------------------------------------------

function printRfTable(title: string, layers: LayerSpec[]): void {
  console.log(title);
  const trace = computeRfTrace(layers, true);
  console.log("  layer |   spec    | jump | receptive field");
  console.log("  ------+-----------+------+-----------------");
  console.log(`  ${"in".padStart(5)} |    --     |    1 |  1x1 (one pixel)`);
  for (let i = 0; i < layers.length; i++) {
    const l = layers[i];
    const step = trace[i + 1];
    const spec = `${l.kernel}x${l.kernel} s${l.stride}`;
    console.log(
      `  ${String(i + 1).padStart(5)} | ${spec.padStart(9)} | ${String(step.jumpBefore).padStart(4)} |  ${step.field}x${step.field}`,
    );
  }
}

function main(): void {
  const rng = mulberry32(404);

  console.log("================================================================");
  console.log(" stage04 — Receptive Field: what does a deep neuron actually see?");
  console.log("================================================================\n");

  // ===================================================================
  // PART 1 — Theory then forward-impulse, all stride-1 (the clean case).
  // ===================================================================
  console.log("### PART 1 — three 3x3 conv layers, stride 1, no padding\n");
  const stack1: LayerSpec[] = [
    { kernel: 3, stride: 1 },
    { kernel: 3, stride: 1 },
    { kernel: 3, stride: 1 },
  ];
  printRfTable("Theoretical receptive field (recurrence r += (k-1)*jump):", stack1);
  const theory1 = finalField(stack1, true);
  console.log(`\n  => theory says the layer-3 unit sees a ${theory1}x${theory1} input window.\n`);

  // Forward all-ones impulse: a single bright pixel at the input center. The footprint it
  // leaves in each feature map grows 1 -> 3 -> 5 -> 7, matching the table.
  // SIZE INVARIANT: input must be big enough that the deepest feature map is itself >= the
  //   7x7 field, else the footprint gets CLIPPED by the map boundary and under-reports. With
  //   H=15 and three stride-1 3x3 layers the output is 9x9 — room for the full 7x7 spread.
  const H = 15,
    W = 15; // odd so there's an exact center; large enough that 7x7 isn't clipped
  const cy = (H - 1) / 2,
    cx = (W - 1) / 2;
  const impulse = Tensor.zeros([1, 1, H, W]);
  impulse.data[cy * W + cx] = 1;

  const ones1 = stack1.map((l) => onesKernel(l.kernel));
  const params1 = stack1.map((l) => ({ stride: l.stride, padding: 0 }));
  const acts1 = forwardStack(impulse, ones1, params1);

  console.log("Forward all-ones impulse — non-zero footprint per layer (should match theory):");
  for (let i = 0; i < acts1.length; i++) {
    const a = acts1[i];
    const [, , h, w] = a.shape;
    const box = nonzeroBox(a.data, h, w);
    const sz = box ? `${box.y1 - box.y0}x${box.x1 - box.x0}` : "0";
    const label = i === 0 ? "input" : `conv${i}`;
    console.log(`  ${label.padEnd(6)} feature map ${h}x${w}, non-zero footprint = ${sz}`);
  }

  // Heatmap of the deepest feature map: the impulse has fanned out to a 7x7 blob.
  const deep = acts1[acts1.length - 1];
  const [, , dh, dw] = deep.shape;
  console.log("\nLayer-3 feature map (all-ones kernels) — the spread of one input pixel:");
  console.log(heatmapAscii(deep.data, dh, dw));

  // Gradient impulse on the center unit: the input pixels it can see.
  const grad1 = measureRfByGradient(impulse, ones1, params1, (oh, ow) => ({
    oy: (oh - 1) / 2,
    ox: (ow - 1) / 2,
  }));
  const gb1 = grad1.box!;
  const measured1 = gb1.y1 - gb1.y0;
  console.log(
    `\nGradient impulse (center unit of ${grad1.outH}x${grad1.outW} output): RF = ${measured1}x${gb1.x1 - gb1.x0} input pixels`,
  );
  console.log(`  theory ${theory1}x${theory1} vs measured ${measured1}x${measured1}  ` +
    `[${measured1 === theory1 ? "MATCH" : "MISMATCH"}]`);
  console.log(`Receptive field of that unit, drawn on the ${H}x${W} input grid:`);
  console.log(receptiveFieldAscii(H, W, gb1.y0, gb1.y1, gb1.x0, gb1.x1));

  // ===================================================================
  // PART 2 — Effective receptive field: random kernels reveal Gaussian decay.
  // ===================================================================
  console.log("\n\n### PART 2 — effective RF shape with random (Gaussian) kernels\n");
  console.log(
    "Same 3-layer stack, but kernels are N(0,1) random. The RF *outline* is unchanged\n" +
      "(still 7x7), yet the gradient MAGNITUDE is not a flat box: center pixels feed many\n" +
      "more paths to the output, so |grad| decays toward the edges — the effective RF.\n",
  );
  const randK = stack1.map((l) => gaussianKernel(l.kernel, rng));
  const gradR = measureRfByGradient(impulse, randK, params1, (oh, ow) => ({
    oy: (oh - 1) / 2,
    ox: (ow - 1) / 2,
  }));
  const rb = gradR.box!;
  console.log(`Random-kernel gradient impulse: RF outline = ${rb.y1 - rb.y0}x${rb.x1 - rb.x0} ` +
    `(same support as all-ones), but look at the magnitudes:`);
  // Crop the input-grad to the RF box and heatmap |grad| so the decay is visible.
  const rh = rb.y1 - rb.y0,
    rw = rb.x1 - rb.x0;
  const crop = new Float64Array(rh * rw);
  for (let y = 0; y < rh; y++)
    for (let x = 0; x < rw; x++)
      crop[y * rw + x] = Math.abs(impulse.grad[(rb.y0 + y) * W + (rb.x0 + x)]);
  console.log(heatmapAscii(crop, rh, rw));
  // Quantify center-vs-edge decay so it's a number, not just a picture.
  const centerMag = crop[((rh - 1) / 2) * rw + (rw - 1) / 2];
  const cornerMag = crop[0];
  console.log(
    `  center |grad| = ${centerMag.toFixed(4)},  corner |grad| = ${cornerMag.toFixed(4)},  ` +
      `center/corner = ${(centerMag / Math.max(cornerMag, 1e-12)).toFixed(1)}x`,
  );
  console.log("  (center weighted far more heavily — this is why the 'effective' RF is");
  console.log("   smaller than the theoretical one: the box edges barely contribute.)");

  // ===================================================================
  // PART 3 — FAILURE MODE: a stride-2 layer, and forgetting to count it.
  // ===================================================================
  console.log("\n\n### PART 3 — FAILURE MODE: stride multiplies the RF; forgetting it lies\n");
  // Stack: 3x3 s1, then 3x3 s2 (downsample), then 3x3 s1.
  const stack2: LayerSpec[] = [
    { kernel: 3, stride: 1 },
    { kernel: 3, stride: 2 },
    { kernel: 3, stride: 1 },
  ];
  printRfTable("Correct recurrence (jump grows at the stride-2 layer):", stack2);
  const correct = finalField(stack2, true);
  const naive = finalField(stack2, false); // the bug: jump pinned at 1
  console.log(`\n  correct final RF = ${correct}x${correct}`);
  console.log(`  naive  final RF = ${naive}x${naive}   <- forgot to multiply jump by stride`);
  console.log(
    `  the naive value UNDER-reports by ${correct - naive} pixels per side ` +
      `(${(((correct - naive) / correct) * 100).toFixed(0)}% smaller field).`,
  );

  // Ground truth by measurement. Use a larger input so the stride-2 field fits with margin.
  const H2 = 17,
    W2 = 17;
  const impulse2 = Tensor.zeros([1, 1, H2, W2]);
  impulse2.data[((H2 - 1) / 2) * W2 + (W2 - 1) / 2] = 1;
  const ones2 = stack2.map((l) => onesKernel(l.kernel));
  const params2 = stack2.map((l) => ({ stride: l.stride, padding: 0 }));
  // Pick the output unit whose field is centered on the impulse. With stride the output is
  // downsampled, so the center output index maps back near the input center.
  const grad2 = measureRfByGradient(impulse2, ones2, params2, (oh, ow) => ({
    oy: (oh - 1) / 2,
    ox: (ow - 1) / 2,
  }));
  const g2 = grad2.box!;
  const measured2 = g2.y1 - g2.y0;
  console.log(
    `\nGradient impulse (center unit of ${grad2.outH}x${grad2.outW} output): MEASURED RF = ` +
      `${measured2}x${g2.x1 - g2.x0} input pixels`,
  );
  console.log("  comparison:");
  console.log(`    naive formula   : ${naive}x${naive}     [${measured2 === naive ? "match" : "WRONG — underestimate"}]`);
  console.log(`    correct formula : ${correct}x${correct}     [${measured2 === correct ? "MATCH" : "mismatch"}]`);
  console.log(`Measured receptive field on the ${H2}x${W2} input (note it is wider than naive's ${naive}x${naive}):`);
  console.log(receptiveFieldAscii(H2, W2, g2.y0, g2.y1, g2.x0, g2.x1));

  console.log("\n----------------------------------------------------------------");
  console.log("Takeaway: stride has a MULTIPLICATIVE effect on receptive field via");
  console.log("the jump term. Counting only kernel sizes (ignoring stride) makes you");
  console.log("think a deep unit sees a small window when it really sees a large one —");
  console.log("a silent design error that the gradient impulse exposes immediately.");
  console.log("----------------------------------------------------------------");
}

main();
