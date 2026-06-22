// core/viz.ts — ASCII visualization. The book runs in a terminal, so "show, don't tell"
//   means drawing in characters.
//
// WHY ASCII and not a plotting lib: zero dependencies, works over SSH/CI logs, and forces
//   the numbers into a shape the reader can eyeball. The TRANSFERABLE lessons of this book
//   are largely about SHAPES — a loss curve's elbow, ΔW's near-rank-1 heatmap, the param
//   bar where PEFT is a sliver of full-FT. ASCII renders those shapes honestly.
//
// HONESTY NOTE: these render exactly the numbers passed in. No smoothing, no autoscaling
//   tricks that hide divergence. If a loss curve is flat, the sparkline is flat.

const BLOCKS = " ▁▂▃▄▅▆▇█"; // 9 levels for sparklines
const GRAY = " .:-=+*#%@"; // 10 levels, light->dark, for heatmaps/histograms

function finiteRange(xs: number[]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const v of xs) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) hi = lo + 1; // avoid div-by-zero on constant series
  return [lo, hi];
}

/** One-line sparkline of a series (e.g. a loss curve compressed to a glance). */
export function sparkline(xs: number[]): string {
  const [lo, hi] = finiteRange(xs);
  return xs
    .map((v) => {
      if (!Number.isFinite(v)) return "?";
      const t = (v - lo) / (hi - lo);
      return BLOCKS[Math.min(BLOCKS.length - 1, Math.max(0, Math.round(t * (BLOCKS.length - 1))))];
    })
    .join("");
}

/**
 * Multi-line loss curve with axis labels. height rows tall.
 * INVARIANT: y axis is min..max of the data (printed), so the reader can read absolute
 *   values, not just relative shape. Lower loss = lower on screen (intuitive).
 */
export function lossCurve(xs: number[], opts: { height?: number; width?: number; label?: string } = {}): string {
  const height = opts.height ?? 8;
  const width = opts.width ?? Math.min(xs.length, 60);
  // resample/clip to width by simple striding
  const step = Math.max(1, Math.ceil(xs.length / width));
  const sampled: number[] = [];
  for (let i = 0; i < xs.length; i += step) sampled.push(xs[i]);
  const [lo, hi] = finiteRange(sampled);
  const grid: string[][] = Array.from({ length: height }, () => Array.from({ length: sampled.length }, () => " "));
  sampled.forEach((v, x) => {
    if (!Number.isFinite(v)) return;
    const t = (v - lo) / (hi - lo);
    const row = Math.min(height - 1, Math.max(0, Math.round((1 - t) * (height - 1))));
    grid[row][x] = "•";
  });
  const lines = grid.map((r, i) => {
    const yval = hi - ((hi - lo) * i) / (height - 1);
    return `${yval.toFixed(3).padStart(8)} │${r.join("")}`;
  });
  const header = opts.label ? `${opts.label}\n` : "";
  const footer = `${" ".repeat(8)} └${"─".repeat(sampled.length)}  (n=${xs.length})`;
  return header + lines.join("\n") + "\n" + footer;
}

/**
 * Histogram of a value distribution (e.g. weight magnitudes, activations).
 * Buckets into `bins` equal-width bins over [min,max]; bar length = count, scaled to width.
 */
export function histogram(values: number[], opts: { bins?: number; width?: number; label?: string } = {}): string {
  const bins = opts.bins ?? 12;
  const width = opts.width ?? 40;
  const finite = values.filter(Number.isFinite);
  const [lo, hi] = finiteRange(finite);
  const counts = new Array(bins).fill(0);
  for (const v of finite) {
    let b = Math.floor(((v - lo) / (hi - lo)) * bins);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    counts[b]++;
  }
  const maxC = Math.max(1, ...counts);
  const lines = counts.map((c, i) => {
    const binLo = lo + ((hi - lo) * i) / bins;
    const barLen = Math.round((c / maxC) * width);
    return `${binLo.toFixed(2).padStart(7)} │${"█".repeat(barLen)} ${c}`;
  });
  const header = opts.label ? `${opts.label}\n` : "";
  return header + lines.join("\n");
}

/**
 * Heatmap of a 2-D matrix (row-major flat + shape). Magnitude -> gray ramp char.
 * THE PEFT PAYOFF VISUAL: render ΔW and its low-rank reconstruction BA side by side and
 *   the reader SEES that a near-rank-1 matrix is mostly one dominant stripe — the whole
 *   reason LoRA works. We scale by max |value| so sign-agnostic magnitude structure shows.
 */
export function heatmap(data: Float64Array | number[], shape: [number, number], opts: { label?: string } = {}): string {
  const [m, n] = shape;
  let maxAbs = 0;
  for (let i = 0; i < m * n; i++) maxAbs = Math.max(maxAbs, Math.abs(data[i]));
  if (maxAbs === 0) maxAbs = 1;
  const lines: string[] = [];
  for (let i = 0; i < m; i++) {
    let row = "";
    for (let j = 0; j < n; j++) {
      const t = Math.abs(data[i * n + j]) / maxAbs;
      row += GRAY[Math.min(GRAY.length - 1, Math.round(t * (GRAY.length - 1)))];
    }
    lines.push(row);
  }
  const header = opts.label ? `${opts.label} (max|·|=${maxAbs.toFixed(4)})\n` : "";
  return header + lines.join("\n");
}

/**
 * Horizontal bar chart for labeled magnitudes (trainable-param ratio, memory MB, etc).
 * Bars are scaled to the largest value so relative size is the message. Each row also
 *   prints the raw value so absolute numbers stay honest.
 */
export function bar(entries: { label: string; value: number; note?: string }[], opts: { width?: number } = {}): string {
  const width = opts.width ?? 40;
  const maxV = Math.max(1e-12, ...entries.map((e) => e.value));
  const labelW = Math.max(...entries.map((e) => e.label.length));
  return entries
    .map((e) => {
      const len = Math.round((e.value / maxV) * width);
      const note = e.note ? `  ${e.note}` : "";
      return `${e.label.padEnd(labelW)} │${"█".repeat(len)}${"·".repeat(width - len)}│ ${e.value.toLocaleString()}${note}`;
    })
    .join("\n");
}
