// core/plot.ts — ASCII visualization, terminal-only (no plotting library, no browser).
//
// WHY ASCII: this book runs offline on a CPU and its outputs must be inspectable in a
//   plain stdout log (and pasteable into the book text). A loss curve that drops, a
//   gradient-norm histogram that piles up at the explode threshold, an attention heatmap
//   with a bright diagonal — these are the chapter's EVIDENCE, and they have to survive
//   as monospace text. Every renderer returns a string; the caller prints it.
//
// INVARIANT: all renderers tolerate empty / constant / NaN-containing input without
//   throwing — a viz helper crashing the run is worse than an ugly chart. NaNs are
//   rendered as a distinct glyph so "the run diverged" is visible, not hidden.

const SPARK = "▁▂▃▄▅▆▇█"; // 8 levels, low -> high

/** One-line sparkline of a numeric series. NaN/Inf -> '?'. Constant series -> mid level. */
export function sparkline(arr: number[]): string {
  if (arr.length === 0) return "";
  const finite = arr.filter((x) => Number.isFinite(x));
  if (finite.length === 0) return "?".repeat(arr.length);
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (max === min) {
    const mid = SPARK[Math.floor((SPARK.length - 1) / 2)];
    return arr.map((x) => (Number.isFinite(x) ? mid : "?")).join("");
  }
  return arr
    .map((x) => {
      if (!Number.isFinite(x)) return "?";
      const t = (x - min) / (max - min);
      return SPARK[Math.min(SPARK.length - 1, Math.floor(t * SPARK.length))];
    })
    .join("");
}

export interface LineChartOpts {
  width?: number; // plot columns (default 60)
  height?: number; // plot rows (default 12)
  labels?: string[]; // one per series, for the legend
}

/**
 * Multi-series line chart with a y-axis scale. Series are resampled to `width` columns
 * (nearest sample) and drawn with distinct glyphs. Used for "RNN vs LSTM loss" overlays.
 * WHY nearest-sample (not interpolation): keeps each plotted point a REAL measured value,
 *   never a synthetic average — honest-number discipline extends to the chart.
 */
export function lineChart(series: number[][], opts: LineChartOpts = {}): string {
  const width = opts.width ?? 60;
  const height = opts.height ?? 12;
  const glyphs = ["*", "o", "+", "x", "#", "@"];
  const all = series.flat().filter((x) => Number.isFinite(x));
  if (all.length === 0) return "(no finite data)";
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  // grid[row][col]; row 0 = top (max). Each cell holds a glyph or ' '.
  const grid: string[][] = Array.from({ length: height }, () => new Array<string>(width).fill(" "));
  series.forEach((s, si) => {
    if (s.length === 0) return;
    const g = glyphs[si % glyphs.length];
    for (let col = 0; col < width; col++) {
      const srcIdx = Math.round((col / Math.max(1, width - 1)) * (s.length - 1));
      const v = s[srcIdx];
      if (!Number.isFinite(v)) continue;
      const t = (v - min) / span;
      const row = Math.min(height - 1, Math.max(0, Math.round((1 - t) * (height - 1))));
      grid[row][col] = g;
    }
  });
  const lines: string[] = [];
  for (let r = 0; r < height; r++) {
    const yVal = max - (r / Math.max(1, height - 1)) * span;
    lines.push(yVal.toFixed(3).padStart(9) + " |" + grid[r].join(""));
  }
  lines.push(" ".repeat(9) + " +" + "-".repeat(width));
  if (opts.labels) {
    lines.push(
      " ".repeat(11) + opts.labels.map((l, i) => `${glyphs[i % glyphs.length]}=${l}`).join("  "),
    );
  }
  return lines.join("\n");
}

/**
 * Histogram of a value distribution into `bins` buckets. Used for gradient-magnitude /
 * activation distributions (e.g. showing grads pile up at the clip threshold).
 */
export function histogram(arr: number[], bins = 20): string {
  const finite = arr.filter((x) => Number.isFinite(x));
  const nanCount = arr.length - finite.length;
  if (finite.length === 0) return `(all ${arr.length} values non-finite)`;
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (max === min) max = min + 1; // avoid zero-width bins for constant data
  const counts = new Array<number>(bins).fill(0);
  for (const x of finite) {
    const b = Math.min(bins - 1, Math.floor(((x - min) / (max - min)) * bins));
    counts[b]++;
  }
  const peak = Math.max(...counts) || 1;
  const barW = 40;
  const lines = counts.map((c, i) => {
    const lo = min + (i / bins) * (max - min);
    const bar = "#".repeat(Math.round((c / peak) * barW));
    return `${lo.toExponential(2).padStart(11)} | ${bar} ${c}`;
  });
  if (nanCount > 0) lines.push(`  (+${nanCount} non-finite values omitted)`);
  return lines.join("\n");
}

const SHADES = " .:-=+*#%@"; // light -> dark, 10 levels

/**
 * Grayscale heatmap of a 2D matrix (rows of equal length). Used for attention weight
 * matrices and SSM state-over-time. Bright diagonal in attention, decaying trail in an
 * SSM state — both read clearly as shaded text.
 * INVARIANT: each row must have the same length; ragged input throws (a malformed
 *   attention matrix is a real bug worth surfacing).
 */
export function heatmap(matrix: number[][]): string {
  if (matrix.length === 0) return "(empty matrix)";
  const cols = matrix[0].length;
  for (const r of matrix) if (r.length !== cols) throw new Error("heatmap: ragged matrix");
  const flat = matrix.flat().filter((x) => Number.isFinite(x));
  if (flat.length === 0) return "(no finite cells)";
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const span = max - min || 1;
  return matrix
    .map((row) =>
      row
        .map((v) => {
          if (!Number.isFinite(v)) return "?";
          const t = (v - min) / span;
          return SHADES[Math.min(SHADES.length - 1, Math.floor(t * SHADES.length))];
        })
        .join(""),
    )
    .join("\n");
}

/** Horizontal bar chart of labeled values. Used for complexity / wall-clock comparisons. */
export function bar(labeledValues: { label: string; value: number }[]): string {
  if (labeledValues.length === 0) return "(no bars)";
  const max = Math.max(...labeledValues.map((d) => Math.abs(d.value))) || 1;
  const labelW = Math.max(...labeledValues.map((d) => d.label.length));
  const barW = 40;
  return labeledValues
    .map((d) => {
      const len = Math.round((Math.abs(d.value) / max) * barW);
      return `${d.label.padEnd(labelW)} | ${"█".repeat(len)} ${d.value}`;
    })
    .join("\n");
}
