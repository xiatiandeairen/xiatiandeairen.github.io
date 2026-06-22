// core/viz.ts — ASCII visualization. The PRIMARY output form of this book.
//
// WHY ASCII and not a plotting library: this is an offline, zero-runtime-dependency book
//   meant to run in any terminal and have its output pasted verbatim into prose. A heatmap
//   you can read in stdout is more honest than a PNG nobody regenerates — the picture in the
//   text IS the measured data. Every function here is pure: numbers in, string out.
//
// INVARIANT: these render whatever numbers they're given without smoothing or rescaling
//   beyond an explicit min/max normalization. If an attention matrix looks wrong, the bug is
//   upstream — viz never lies about its input. (The one transform is min-max -> shade index,
//   which we label so the reader knows the absolute scale.)

// Shade ramp from empty to full. Index 0 = smallest value, last = largest.
const SHADES = " .:-=+*#%@";

/** Map a normalized value in [0,1] to a shade char. Clamps out-of-range defensively so a
 *  stray NaN/Inf renders as the lowest shade instead of crashing the whole plot. */
function shade(norm: number): string {
  if (!Number.isFinite(norm)) return SHADES[0];
  const i = Math.max(0, Math.min(SHADES.length - 1, Math.round(norm * (SHADES.length - 1))));
  return SHADES[i];
}

export interface HeatmapOpts {
  rowLabels?: string[];
  colLabels?: string[];
  title?: string;
  /** Fixed value range; if omitted, min/max of data is used (and printed). */
  vmin?: number;
  vmax?: number;
}

/**
 * Render a (rows x cols) matrix as a shaded ASCII heatmap. Used for attention patterns
 * (each cell = how much query row attends to key col) and SAE feature activations.
 * The min/max actually used is printed so the reader knows the absolute scale behind the
 * relative shading — without it a uniform matrix and a peaky one could look identical.
 */
export function asciiHeatmap(data: number[][], opts: HeatmapOpts = {}): string {
  const rows = data.length;
  const cols = rows > 0 ? data[0].length : 0;
  let vmin = opts.vmin ?? Infinity;
  let vmax = opts.vmax ?? -Infinity;
  if (opts.vmin === undefined || opts.vmax === undefined) {
    for (const row of data) for (const v of row) {
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
  }
  const range = vmax - vmin || 1; // avoid div-by-0 on a constant matrix
  const lines: string[] = [];
  if (opts.title) lines.push(opts.title);
  const rowLabelW = opts.rowLabels ? Math.max(...opts.rowLabels.map((s) => s.length)) : 0;
  if (opts.colLabels) {
    const pad = " ".repeat(rowLabelW + 1);
    lines.push(pad + opts.colLabels.map((c) => c[0] ?? " ").join(""));
  }
  for (let r = 0; r < rows; r++) {
    let line = "";
    if (opts.rowLabels) line += opts.rowLabels[r].padStart(rowLabelW) + " ";
    for (let c = 0; c < cols; c++) line += shade((data[r][c] - vmin) / range);
    lines.push(line);
  }
  lines.push(`  scale: '${SHADES}'  vmin=${vmin.toFixed(3)} vmax=${vmax.toFixed(3)}`);
  return lines.join("\n");
}

/**
 * Horizontal bar chart. Each entry -> a bar proportional to value/max. Used for per-head
 * patch recovery, per-layer logit-lens confidence, probe-vs-baseline accuracy. Negative
 * values render leftward-marked so sign is visible (patch recovery can go negative).
 */
export function asciiBar(
  items: { label: string; value: number }[],
  opts: { width?: number; title?: string } = {},
): string {
  const width = opts.width ?? 40;
  const lines: string[] = [];
  if (opts.title) lines.push(opts.title);
  const labelW = Math.max(...items.map((i) => i.label.length));
  const maxAbs = Math.max(1e-9, ...items.map((i) => Math.abs(i.value)));
  for (const it of items) {
    const n = Math.round((Math.abs(it.value) / maxAbs) * width);
    const bar = (it.value < 0 ? "-" : "#").repeat(Math.max(0, n));
    lines.push(`${it.label.padStart(labelW)} | ${bar} ${it.value.toFixed(3)}`);
  }
  return lines.join("\n");
}

/**
 * Sparkline of a 1-D series on one line, using the shade ramp as a mini bar chart. Used for
 * loss curves — a compact "did it go down?" glance. Prints first/last so the reader sees the
 * actual endpoints, not just the silhouette.
 */
export function asciiSparkline(values: number[], opts: { title?: string } = {}): string {
  if (values.length === 0) return opts.title ?? "";
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const spark = values.map((v) => shade((v - min) / range)).join("");
  const head = opts.title ? opts.title + " " : "";
  return `${head}${spark}  [first=${values[0].toFixed(4)} last=${values[values.length - 1].toFixed(4)} min=${min.toFixed(4)} max=${max.toFixed(4)}]`;
}

/**
 * Scatter of 2-D points onto an ASCII grid. Used for PCA / embedding projections (e.g. the
 * circular structure of modAdd embeddings). Points are binned to a grid; '*' marks occupied
 * cells. Optional per-point single-char labels mark structure. The grid is min-max scaled
 * per axis and the ranges are printed (absolute geometry, not just the shape).
 */
export function asciiScatter(
  points: { x: number; y: number; label?: string }[],
  opts: { width?: number; height?: number; title?: string } = {},
): string {
  const width = opts.width ?? 50;
  const height = opts.height ?? 18;
  const lines: string[] = [];
  if (opts.title) lines.push(opts.title);
  if (points.length === 0) return lines.join("\n");
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const rx = maxX - minX || 1;
  const ry = maxY - minY || 1;
  const grid: string[][] = Array.from({ length: height }, () => Array(width).fill(" "));
  for (const p of points) {
    const cx = Math.min(width - 1, Math.max(0, Math.round(((p.x - minX) / rx) * (width - 1))));
    // invert y so larger y is higher on screen (natural plot orientation)
    const cy = Math.min(height - 1, Math.max(0, Math.round((1 - (p.y - minY) / ry) * (height - 1))));
    grid[cy][cx] = p.label ? p.label[0] : "*";
  }
  for (const row of grid) lines.push("|" + row.join("") + "|");
  lines.push(`  x:[${minX.toFixed(2)},${maxX.toFixed(2)}] y:[${minY.toFixed(2)},${maxY.toFixed(2)}]`);
  return lines.join("\n");
}

/**
 * Diff heatmap: render (after - before) with a SIGNED ramp so increases and decreases are
 * visually distinct. Used to show what activation patching changed. Positive cells use the
 * '#'..'@' end of a signed scheme, negatives use '.'-side; zero is blank. The symmetric
 * scale is centered on 0 so "no change" is unambiguous.
 */
export function heatmapDiff(before: number[][], after: number[][], opts: { title?: string } = {}): string {
  const rows = before.length;
  const cols = rows > 0 ? before[0].length : 0;
  let maxAbs = 0;
  const diff: number[][] = [];
  for (let r = 0; r < rows; r++) {
    diff.push([]);
    for (let c = 0; c < cols; c++) {
      const d = after[r][c] - before[r][c];
      diff[r].push(d);
      if (Math.abs(d) > maxAbs) maxAbs = Math.abs(d);
    }
  }
  maxAbs = maxAbs || 1;
  const POS = " .+*#@"; // increasing positive change
  const NEG = " ,;ox&"; // increasing negative change (distinct glyphs so sign is readable)
  const lines: string[] = [];
  if (opts.title) lines.push(opts.title);
  for (let r = 0; r < rows; r++) {
    let line = "";
    for (let c = 0; c < cols; c++) {
      const d = diff[r][c];
      const norm = Math.abs(d) / maxAbs;
      const ramp = d >= 0 ? POS : NEG;
      const i = Math.max(0, Math.min(ramp.length - 1, Math.round(norm * (ramp.length - 1))));
      line += ramp[i];
    }
    lines.push(line);
  }
  lines.push(`  +ramp:'${POS}' -ramp:'${NEG}'  maxAbsDiff=${maxAbs.toFixed(4)}`);
  return lines.join("\n");
}
