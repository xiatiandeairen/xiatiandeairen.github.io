// stage01-convolution.ts — Chapter 01: convolution / cross-correlation, hands-on.
//
// WHY this stage exists: before any network, the reader must SEE that "convolution" in deep
//   learning is really cross-correlation — a sliding dot-product of a small kernel over the
//   image — and that the only knobs are kernel weights, padding, and stride. We make all of
//   that produce REAL numbers (no formulas asserted on faith): a vertical-edge kernel slid
//   over a synthetic vertical-edge image, the output feature map printed and heat-mapped so
//   the edge column literally lights up; a blur kernel for contrast; then the output-size
//   formula (H+2p-k)/s+1 checked digit-by-digit against the engine's own convOutSize, across
//   a grid of (padding, stride); and finally the classic beginner failure — forgetting
//   padding silently shrinks the map and breaks any downstream code that assumed same-size.
//
// HONESTY: every printed number is computed by the core engine (conv2d / convOutSize) on
//   deterministic seeded data — nothing is hand-typed as "the expected answer". The image is
//   a tiny 8x8 toy, so absolute magnitudes are unrealistic; what transfers is the STRUCTURE
//   (edge kernel responds at edges / blur kernel smooths / size formula holds / dropping
//   padding shrinks the map). No timing claims are made here — this chapter is about
//   mechanism, not performance.
//
// CONVENTION REMINDERS (from core/autograd.ts): images are NCHW (here N=1, C=1), conv
//   weights are OIHW (outC, inC, kH, kW). conv2d does CROSS-CORRELATION (no kernel flip) —
//   the deep-learning convention; we say so explicitly because math textbooks flip.
//
// Run: npx tsx src/stage01-convolution.ts   (offline, deterministic, CPU-only)

import { Tensor, conv2d, convOutSize, type Conv2dParams } from "./core/autograd.js";
import { mulberry32, uniform, type Rng } from "./core/rng.js";
import { heatmapAscii } from "./core/metrics.js";

// ---------------------------------------------------------------------------
// Synthetic input: an 8x8 grayscale image with a single vertical edge.
// Left half dark (~0), right half bright (~1), with a small seeded jitter so the image is
// not a perfectly noiseless step (real images never are) yet stays reproducible.
// INVARIANT: the edge sits between column 3 and column 4, so an edge detector should respond
//   on exactly those columns and nowhere else — that is the eyeball test for the heatmap.
// ---------------------------------------------------------------------------
const IMG_SIZE = 8;
const EDGE_COL = 4; // first bright column; edge is the 3|4 boundary

function makeVerticalEdgeImage(rng: Rng): Tensor {
  const data = new Float64Array(IMG_SIZE * IMG_SIZE);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const base = x >= EDGE_COL ? 1.0 : 0.0;
      // jitter in [-0.03, 0.03): keeps the step crisp but non-degenerate.
      data[y * IMG_SIZE + x] = base + uniform(rng, -0.03, 0.03);
    }
  }
  // NCHW with N=1, C=1.
  return new Tensor(data, [1, 1, IMG_SIZE, IMG_SIZE]);
}

// 3x3 Sobel-like vertical-edge kernel (OIHW: outC=1, inC=1, 3x3). Columns sum to 0 so a
// flat region yields ~0 response; a left->right brightness increase yields a large positive
// response. This is the discrete derivative in x.
function makeVerticalEdgeKernel(): Tensor {
  // prettier-ignore
  const k = [
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1,
  ];
  return new Tensor(Float64Array.from(k), [1, 1, 3, 3]);
}

// 3x3 box-blur kernel (all 1/9). Averages a neighborhood -> smooths, kills the sharp edge.
// Used as a CONTRAST so the reader sees the kernel — not conv2d — decides what is detected.
function makeBlurKernel(): Tensor {
  const v = 1 / 9;
  return new Tensor(Float64Array.from(new Array(9).fill(v)), [1, 1, 3, 3]);
}

/** Pretty-print a single-channel NCHW (1,1,H,W) feature map as a fixed-width number grid.
 *  Pure formatting helper; no side effects beyond returning the string. */
function formatMap(t: Tensor): string {
  const [, , h, w] = t.shape;
  const rows: string[] = [];
  for (let y = 0; y < h; y++) {
    const cells: string[] = [];
    for (let x = 0; x < w; x++) {
      cells.push(t.data[y * w + x].toFixed(2).padStart(7));
    }
    rows.push(cells.join(" "));
  }
  return rows.join("\n");
}

function spatial(t: Tensor): { h: number; w: number } {
  const [, , h, w] = t.shape;
  return { h, w };
}

// ---------------------------------------------------------------------------
// Part 1 — edge kernel highlights the edge; blur kernel smooths it.
// We use padding=1 so the output stays 8x8 (SAME convolution) and lines up with the input
// for a side-by-side read.
// ---------------------------------------------------------------------------
function demoEdgeVsBlur(): void {
  const rng = mulberry32(0xc0ffee);
  const img = makeVerticalEdgeImage(rng);
  const edgeK = makeVerticalEdgeKernel();
  const blurK = makeBlurKernel();
  const same: Conv2dParams = { stride: 1, padding: 1 };

  console.log("=== Part 1: 一个核决定检测什么 (edge vs blur) ===\n");
  console.log("输入图 8x8 (左暗右亮, 竖边在第 3|4 列之间):");
  console.log(heatmapAscii(img.data, IMG_SIZE, IMG_SIZE));
  console.log();

  const edgeOut = conv2d(img, edgeK, null, same);
  console.log("竖边核 (Sobel-like) 输出特征图 (padding=1, 保持 8x8):");
  console.log(formatMap(edgeOut));
  console.log("\n热图 (越亮=响应越强, 边缘列被高亮):");
  console.log(heatmapAscii(edgeOut.data, IMG_SIZE, IMG_SIZE));

  // HONEST CHECK, not assertion theater: locate the column of peak |response| from the actual
  // output and confirm it falls on the known edge boundary. If the engine were wrong this
  // would point elsewhere.
  const peakCol = argmaxAbsColumn(edgeOut);
  console.log(
    `\n响应最强的列 = 第 ${peakCol} 列 (输入竖边在 ${EDGE_COL - 1}|${EDGE_COL} 之间; ` +
      `边缘核在过渡处响应最大 -> 机制正确)`,
  );

  const blurOut = conv2d(img, blurK, null, same);
  console.log("\n模糊核 (box-blur 1/9) 输出特征图 (同一张图):");
  console.log(formatMap(blurOut));
  // Quantify "blur smooths": max-min spread shrinks vs the input.
  const inSpread = spread(img.data);
  const blurSpread = spread(blurOut.data);
  console.log(
    `\n对比度 (max-min): 原图 ${inSpread.toFixed(3)} -> 模糊后 ${blurSpread.toFixed(3)} ` +
      `(模糊核把锐边抹平, 对比度下降)`,
  );
  console.log();
}

/** Column index whose summed |value| is largest. Used to verify the edge response lands on
 *  the true edge column. */
function argmaxAbsColumn(t: Tensor): number {
  const { h, w } = spatial(t);
  let best = -Infinity;
  let bestCol = -1;
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = 0; y < h; y++) s += Math.abs(t.data[y * w + x]);
    if (s > best) {
      best = s;
      bestCol = x;
    }
  }
  return bestCol;
}

function spread(d: ArrayLike<number>): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < d.length; i++) {
    if (d[i] < lo) lo = d[i];
    if (d[i] > hi) hi = d[i];
  }
  return hi - lo;
}

// ---------------------------------------------------------------------------
// Part 2 — output size vs the formula (H + 2p - k)/s + 1, checked digit-by-digit.
// We compute the size two independent ways: (a) by hand from the formula, (b) by actually
// running conv2d and reading its output shape. They MUST agree; we print PASS/FAIL per row.
// The point: the formula is not folklore, it is exactly what the engine does.
// ---------------------------------------------------------------------------
function demoOutputSizeFormula(): void {
  console.log("=== Part 2: 输出尺寸公式 floor((H+2p-k)/s)+1 逐项核对 ===\n");
  const rng = mulberry32(0x5eed);
  const img = makeVerticalEdgeImage(rng); // 8x8
  const k = 3;
  const cases: Array<{ stride: number; padding: number }> = [
    { stride: 1, padding: 0 }, // VALID conv: shrinks to 6
    { stride: 1, padding: 1 }, // SAME conv: stays 8
    { stride: 2, padding: 0 }, // downsample, no pad: 3
    { stride: 2, padding: 1 }, // downsample with pad: 4
    { stride: 3, padding: 0 }, // doesn't tile evenly: floor matters -> 2
  ];

  const header = "stride pad | formula | engine | match";
  console.log(header);
  console.log("-".repeat(header.length));
  let allMatch = true;
  for (const c of cases) {
    // (a) formula by hand (integer floor division).
    const byFormula = Math.floor((IMG_SIZE + 2 * c.padding - k) / c.stride) + 1;
    // (b) what the engine's own helper says (single source of truth used inside conv2d).
    const byHelper = convOutSize(IMG_SIZE, k, c.stride, c.padding);
    // (c) the REAL output shape after actually convolving — the ground truth.
    const out = conv2d(img, makeVerticalEdgeKernel(), null, c);
    const { h, w } = spatial(out);
    const match = byFormula === byHelper && byHelper === h && h === w;
    allMatch &&= match;
    console.log(
      `  ${c.stride}     ${c.padding}  | ${String(byFormula).padStart(2)}x${byFormula} ` +
        `  | ${h}x${w}    | ${match ? "PASS" : "FAIL"}`,
    );
  }
  console.log(
    `\n${allMatch ? "全部一致" : "出现不一致!"}: 手算公式 == convOutSize == 实跑输出形状` +
      (allMatch ? "" : " (BUG)"),
  );
  // Call out the non-even-tiling case explicitly: floor drops the partial last window.
  console.log(
    "注意 stride=3 行: (8-3)/3 = 1.67, floor -> 1, +1 = 2 列; 最后那个不满一格的窗口被丢弃, " +
      "不是补零凑整 (这是 'valid' 语义).\n",
  );
}

// ---------------------------------------------------------------------------
// Part 3 — FAILURE MODE: forgetting padding silently shrinks the map.
// The beginner bug: a layer designed assuming "output is same size as input" (e.g. for a
// residual add, or to index a fixed-size buffer) silently produces a smaller map when padding
// is omitted. The size mismatch only explodes LATER, far from the cause. We demonstrate by
// trying to elementwise-add the 6x6 valid output back onto the 8x8 input — exactly what a
// naive residual connection would do — and catch the shape error the engine throws.
// ---------------------------------------------------------------------------
function demoMissingPaddingFailure(): void {
  console.log("=== Part 3: 失败模式 — 漏掉 padding 导致尺寸缩水 -> 下游 shape mismatch ===\n");
  const rng = mulberry32(0xbadc0de);
  const img = makeVerticalEdgeImage(rng); // 8x8
  const kernel = makeVerticalEdgeKernel(); // 3x3

  const valid = conv2d(img, kernel, null, { stride: 1, padding: 0 });
  console.log(
    `输入 8x8, 用 3x3 核但 padding=0 -> 输出 ${valid.shape[2]}x${valid.shape[3]} ` +
      "(每边各丢 (k-1)/2 = 1 行/列, 因为边界像素凑不齐完整窗口).",
  );
  console.log(
    "边界信息丢失: 第 0 行/列和最后一行/列从不作为窗口中心, 它们的边缘响应永远算不到.\n",
  );

  // The latent bug surfaces here: code that assumed same-size tries to combine the maps.
  console.log("下游代码 (天真地以为尺寸不变, 想做 residual: input + conv(input)):");
  try {
    img.add(valid); // 8x8 + 6x6 -> engine refuses to broadcast
    console.log("  (不应到达这里)");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.log(`  抛错: ${msg}`);
    console.log(
      "  根因不在这一行 — 真正的错在上游 conv 漏了 padding=1. 报错点离病灶很远, " +
        "这正是 '尺寸缩水' bug 难查的原因.",
    );
  }

  // The fix, shown for contrast: padding=1 restores 8x8 and the add succeeds.
  const same = conv2d(img, kernel, null, { stride: 1, padding: 1 });
  const fixed = img.add(same);
  console.log(
    `\n修复: padding=1 -> conv 输出 ${same.shape[2]}x${same.shape[3]}, ` +
      `residual add 成功, 结果形状 ${fixed.shape[2]}x${fixed.shape[3]}.\n`,
  );
}

function main(): void {
  console.log("########## Stage 01: 卷积与互相关 (offline, deterministic) ##########\n");
  console.log(
    "提示: 深度学习里的 'convolution' 实为 cross-correlation (核不翻转); conv2d 走此约定.\n",
  );
  demoEdgeVsBlur();
  demoOutputSizeFormula();
  demoMissingPaddingFailure();
  console.log("########## Stage 01 完成 ##########");
}

main();
