// core/plot.ts — ASCII charts so numeric results are READABLE in a terminal.
//
// WHY ASCII plots in a math book:
//   Convergence curves, frequency spectra, phase-transition curves — these only mean
//   something when you SEE the shape. A reader running a stage offline has no plotting
//   window; a 60-char ASCII bar/line conveys "it converged", "the spectrum peaks at k=3",
//   "there is a sharp transition near T=2.3" instantly and honestly (the chart is drawn
//   from the same numbers we print). No dependency, no GUI, works over SSH.
//
// DESIGN: pure functions returning a multi-line string; callers console.log the result.
//   We never console.log inside — keeps these testable and lets callers add headers.
//
// FAILURE MODE: NaN/Infinity in the data would otherwise blow up the min/max scaling.
//   We coerce non-finite values to 0 before scaling and note it is the caller's job to
//   feed clean numbers; silently plotting garbage is worse than a flat line at 0.

function finite(v: number): number {
  return Number.isFinite(v) ? v : 0;
}

/**
 * Horizontal bar chart: one row per (label, value).
 * WHY scale to max, not to a fixed range: histograms and frequency counts have wildly
 *   different magnitudes per stage; auto-scaling to the largest bar keeps every chart
 *   readable without the caller hand-tuning a range.
 * Negative values are clamped to 0-length bars — asciiBar is for magnitudes (counts,
 *   energies, |coefficients|). Use asciiLine for signed series.
 */
export function asciiBar(
  labels: readonly string[],
  values: readonly number[],
  width = 48,
): string {
  if (labels.length !== values.length) {
    throw new Error(`[plot] asciiBar: ${labels.length} labels vs ${values.length} values`);
  }
  const vs = values.map(finite);
  const max = Math.max(1e-12, ...vs.map((v) => Math.max(0, v))); // avoid /0
  const labelW = Math.max(...labels.map((l) => l.length));
  const lines = labels.map((label, i) => {
    const v = Math.max(0, vs[i]);
    const filled = Math.round((v / max) * width);
    const bar = "█".repeat(filled) + "·".repeat(width - filled);
    return `${label.padStart(labelW)} │${bar}│ ${vs[i].toPrecision(4)}`;
  });
  return lines.join("\n");
}

/**
 * Single-series line chart drawn on a character grid (rows = height, cols = series length
 * or sampled down to `width`).
 * WHY sample down: a 1000-point convergence curve cannot fit in 60 columns; we pick evenly
 *   spaced indices so the overall SHAPE survives even if fine wiggles are lost. The y-axis
 *   labels print the true min/max so the reader can read absolute scale, not just shape.
 * INVARIANT: handles signed data (unlike asciiBar) — the baseline floats wherever min/max
 *   put it, so a curve crossing zero renders correctly.
 */
export function asciiLine(series: readonly number[], width = 60, height = 12): string {
  if (series.length === 0) return "(empty series)";
  // Sample the series down to at most `width` evenly spaced points.
  const cols = Math.min(width, series.length);
  const sampled: number[] = [];
  for (let c = 0; c < cols; c++) {
    const idx = Math.round((c / Math.max(1, cols - 1)) * (series.length - 1));
    sampled.push(finite(series[idx]));
  }
  const lo = Math.min(...sampled);
  const hi = Math.max(...sampled);
  const range = hi - lo || 1; // flat series -> single row, avoid /0
  // grid[row][col]; row 0 is the TOP (highest value).
  const grid: string[][] = Array.from({ length: height }, () =>
    new Array(cols).fill(" "),
  );
  for (let c = 0; c < cols; c++) {
    const norm = (sampled[c] - lo) / range; // 0..1
    let row = height - 1 - Math.round(norm * (height - 1));
    if (row < 0) row = 0;
    if (row >= height) row = height - 1;
    grid[row][c] = "●";
  }
  const axisW = Math.max(hi.toPrecision(3).length, lo.toPrecision(3).length);
  return grid
    .map((rowChars, r) => {
      // Label only the top and bottom rows with the true value bounds.
      let label = "".padStart(axisW);
      if (r === 0) label = hi.toPrecision(3).padStart(axisW);
      else if (r === height - 1) label = lo.toPrecision(3).padStart(axisW);
      return `${label} │${rowChars.join("")}`;
    })
    .join("\n");
}
