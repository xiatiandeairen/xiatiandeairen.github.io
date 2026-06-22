// core/viz.ts — ASCII visualization. No plotting library, no browser; the terminal IS
//   the display. Every chart is pure string-building so stage output is diffable in CI
//   and readable in a book listing.
//
// WHY ASCII: this book ships as runnable listings. A loss curve you can SEE in stdout
//   (and that a reader reproduces bit-for-bit) beats a PNG they can't regenerate. The
//   honesty contract — "numbers are real" — extends to charts: these render the actual
//   arrays the stage computed, never decorative placeholders.
//
// INVARIANT: rendering is side-effect free (returns a string). Stages decide when to print.

/** Unicode shading ramp from empty to full, used by heatmap/sparkline. */
const RAMP = " ░▒▓█";

/**
 * Sparkline of a loss series on one line, plus min/max annotation. Good for "did it go
 * down?" at a glance. FAILURE MODE handled: a flat series (max==min) would divide by 0;
 * we render a mid-level line instead.
 */
export function plotLoss(series: number[], width = 50): string {
  if (series.length === 0) return "(empty series)";
  const blocks = "▁▂▃▄▅▆▇█";
  // Resample to `width` columns by simple bucketed averaging so long runs still fit.
  const cols: number[] = [];
  const per = series.length / width;
  for (let c = 0; c < Math.min(width, series.length); c++) {
    const start = Math.floor(c * per);
    const end = Math.max(start + 1, Math.floor((c + 1) * per));
    let s = 0,
      cnt = 0;
    for (let i = start; i < end && i < series.length; i++) {
      s += series[i];
      cnt++;
    }
    cols.push(s / cnt);
  }
  let min = Infinity,
    max = -Infinity;
  for (const v of cols) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min;
  const line = cols
    .map((v) => {
      if (span < 1e-12) return blocks[3];
      const t = (v - min) / span;
      return blocks[Math.min(blocks.length - 1, Math.floor(t * (blocks.length - 1)))];
    })
    .join("");
  return `${line}  [${min.toFixed(4)} .. ${max.toFixed(4)}], n=${series.length}`;
}

/**
 * Horizontal bar chart. Used for expert utilization / load. Bars are normalized to the
 * max value so the longest bar fills `width`. Prints label, bar, and the raw value.
 */
export function bar(labels: string[], values: number[], width = 30): string {
  if (labels.length !== values.length) throw new Error("bar: labels/values length mismatch");
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  if (max <= 0) max = 1;
  const labW = Math.max(...labels.map((l) => l.length));
  return labels
    .map((lab, i) => {
      const filled = Math.round((values[i] / max) * width);
      const barStr = "█".repeat(filled) + "·".repeat(width - filled);
      return `${lab.padStart(labW)} │${barStr}│ ${values[i].toFixed(4)}`;
    })
    .join("\n");
}

/**
 * Heatmap of a matrix using the shading ramp. Each cell scaled by the global max.
 * Used for routing distributions and expert×cluster co-occurrence — the visual proof
 * that experts specialized (a near-diagonal heatmap) or collapsed (one hot column).
 */
export function heatmap(matrix: number[][], rowLabels?: string[], colLabels?: string[]): string {
  const rows = matrix.length;
  if (rows === 0) return "(empty matrix)";
  const cols = matrix[0].length;
  let max = -Infinity;
  for (const r of matrix) for (const v of r) if (v > max) max = v;
  if (max <= 0) max = 1;
  const out: string[] = [];
  if (colLabels) out.push("    " + colLabels.map((c) => c.padStart(2)).join(""));
  for (let i = 0; i < rows; i++) {
    const lab = rowLabels ? rowLabels[i].padStart(3) + " " : "";
    let line = lab;
    for (let j = 0; j < cols; j++) {
      const t = matrix[i][j] / max;
      const ch = RAMP[Math.min(RAMP.length - 1, Math.floor(t * (RAMP.length - 1)))];
      line += ch + ch; // double-width so cells read as squares
    }
    out.push(line);
  }
  return out.join("\n");
}

/**
 * Histogram of a value array into `bins` buckets, rendered as horizontal bars.
 * Used for gating-probability distributions (are gates confident or uniform?).
 */
export function hist(values: number[], bins = 10): string {
  if (values.length === 0) return "(empty)";
  let min = Infinity,
    max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min || 1;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let b = Math.floor(((v - min) / span) * bins);
    if (b >= bins) b = bins - 1; // max value lands in last bin, not out of range
    if (b < 0) b = 0;
    counts[b]++;
  }
  const labels = counts.map((_, i) => {
    const lo = min + (i / bins) * span;
    return lo.toFixed(2);
  });
  return bar(labels, counts, 24);
}
