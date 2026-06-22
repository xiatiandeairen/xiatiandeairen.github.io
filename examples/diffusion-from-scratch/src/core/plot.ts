// core/plot.ts — ASCII visualization. The book has no GUI, so every "figure" is text.
//
// WHY this matters for honesty: these are NOT illustrative diagrams. Every character is
//   placed from real numbers the stage just computed — a scatter cell is dark because that
//   many sampled/real points actually landed there, a loss-curve row is high because the
//   loss really was that value. If a plot looks wrong, the model IS wrong.
//
// All functions return a string (the caller console.logs it) so they stay pure and testable.

import { Tensor } from "./tensor.js";

const DENSITY_RAMP = " .:-=+*#@"; // 9 levels, light -> dark, by point count per cell

/**
 * Scatter a [n,2] point cloud into a w×h character grid. Points are mapped into the grid by
 * their min/max bounding box (so any normalized cloud fills the frame); each cell's char is
 * chosen by how many points fell in it (DENSITY_RAMP). This is the primary way to eyeball
 * "did the samples land where the data is?".
 * INVARIANT: input must be shape [n,2]. Degenerate (all points identical) collapses to one
 *   cell — guarded by a span floor so we don't divide by zero.
 */
export function scatterASCII(points: Tensor, w = 60, h = 24): string {
  if (points.shape.length !== 2 || points.shape[1] !== 2) {
    throw new Error(`scatterASCII: expected [n,2], got ${points.shape}`);
  }
  const n = points.shape[0];
  const d = points.data;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    minX = Math.min(minX, d[i * 2]);
    maxX = Math.max(maxX, d[i * 2]);
    minY = Math.min(minY, d[i * 2 + 1]);
    maxY = Math.max(maxY, d[i * 2 + 1]);
  }
  const spanX = maxX - minX || 1e-9; // floor avoids /0 on a degenerate cloud
  const spanY = maxY - minY || 1e-9;
  const counts = new Int32Array(w * h);
  let maxCount = 0;
  for (let i = 0; i < n; i++) {
    const cx = Math.min(w - 1, Math.floor(((d[i * 2] - minX) / spanX) * (w - 1)));
    // Row 0 is the TOP, so invert y: large y -> small row index.
    const cy = Math.min(h - 1, Math.floor((1 - (d[i * 2 + 1] - minY) / spanY) * (h - 1)));
    const idx = cy * w + cx;
    counts[idx]++;
    maxCount = Math.max(maxCount, counts[idx]);
  }
  const lines: string[] = [];
  for (let r = 0; r < h; r++) {
    let line = "";
    for (let c = 0; c < w; c++) {
      const ct = counts[r * w + c];
      if (ct === 0) {
        line += " ";
      } else {
        // Map count -> ramp index 1..8 (reserve 0/space for empty).
        const level = 1 + Math.floor(((ct - 1) / Math.max(maxCount - 1, 1)) * (DENSITY_RAMP.length - 2));
        line += DENSITY_RAMP[level];
      }
    }
    lines.push(line);
  }
  return lines.join("\n");
}

/**
 * Render a precomputed h×w numeric grid (row-major number[][]) as a density/intensity
 * heatmap — used for learned score-field magnitude or density. Values are min/max scaled
 * across the whole grid onto DENSITY_RAMP. Row 0 is printed at the TOP as given.
 */
export function heatmapASCII(grid: number[][]): string {
  const h = grid.length;
  if (h === 0) return "";
  const w = grid[0].length;
  let min = Infinity;
  let max = -Infinity;
  for (const row of grid) for (const v of row) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  const span = max - min || 1e-9;
  const lines: string[] = [];
  for (let r = 0; r < h; r++) {
    let line = "";
    for (let c = 0; c < w; c++) {
      const level = Math.floor(((grid[r][c] - min) / span) * (DENSITY_RAMP.length - 1));
      line += DENSITY_RAMP[level];
    }
    lines.push(line);
  }
  return lines.join("\n");
}

/**
 * Plot a 1-D loss curve as an ASCII line chart with `rows` height and a fixed width = the
 * number of losses (down-sampled if longer than `width`). The y-axis is auto-scaled to the
 * observed [min, max] and labeled, so the reader can confirm the loss actually FELL (the
 * single most important training signal) and roughly by how much.
 */
export function lossCurveASCII(losses: number[], rows = 12, width = 60): string {
  if (losses.length === 0) return "(no losses)";
  // Down-sample by averaging into `width` buckets so a long run still fits one screen.
  const cols = Math.min(width, losses.length);
  const bucket: number[] = new Array(cols).fill(0);
  const bucketCount: number[] = new Array(cols).fill(0);
  for (let i = 0; i < losses.length; i++) {
    const b = Math.min(cols - 1, Math.floor((i / losses.length) * cols));
    bucket[b] += losses[i];
    bucketCount[b]++;
  }
  for (let b = 0; b < cols; b++) bucket[b] /= Math.max(bucketCount[b], 1);
  let min = Math.min(...bucket);
  let max = Math.max(...bucket);
  if (max - min < 1e-12) max = min + 1e-12; // flat curve: avoid /0, still render a line
  const grid: string[][] = Array.from({ length: rows }, () => new Array(cols).fill(" "));
  for (let c = 0; c < cols; c++) {
    // Row 0 is top (high loss). Map value -> row.
    const norm = (bucket[c] - min) / (max - min);
    const row = Math.min(rows - 1, Math.floor((1 - norm) * (rows - 1)));
    grid[row][c] = "*";
  }
  const lines = grid.map((r, i) => {
    const yLabel = i === 0 ? max.toFixed(4) : i === rows - 1 ? min.toFixed(4) : "";
    return `${yLabel.padStart(8)} |${r.join("")}`;
  });
  lines.push(`${" ".repeat(8)} +${"-".repeat(cols)}`);
  lines.push(`${" ".repeat(9)} step 0${" ".repeat(Math.max(cols - 12, 1))}step ${losses.length}`);
  return lines.join("\n");
}

/**
 * Horizontal ASCII histogram of a value array into `bins` buckets. Used to show, e.g., the
 * distribution of per-sample errors or a marginal coordinate — confirming a claimed mean/
 * spread is real, not asserted. Bar length is scaled to the most-populated bin.
 */
export function histogramASCII(values: number[], bins = 20, barWidth = 40): string {
  if (values.length === 0) return "(no values)";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1e-9;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const b = Math.min(bins - 1, Math.floor(((v - min) / span) * bins));
    counts[b]++;
  }
  const maxCount = Math.max(...counts);
  const lines: string[] = [];
  for (let b = 0; b < bins; b++) {
    const lo = min + (span * b) / bins;
    const len = Math.round((counts[b] / Math.max(maxCount, 1)) * barWidth);
    lines.push(`${lo.toFixed(3).padStart(8)} | ${"#".repeat(len)} ${counts[b]}`);
  }
  return lines.join("\n");
}
