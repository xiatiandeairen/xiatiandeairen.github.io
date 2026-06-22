// core/plot.ts — turn numbers into console-readable pictures.
//
// Why ASCII plots at all: the book is read in a terminal / static page, not a
// notebook. A loss curve that "goes down" is a claim the reader should SEE, not
// take on faith. These renderers are deliberately tiny and dependency-free; they
// trade fidelity for the property that the exact same chart appears for every
// reader on every machine (determinism extends to the visuals).
//
// Honesty constraint baked in: every renderer prints the real min/max it scaled
// against, so a flat-looking line can't hide a tiny-but-real change and a
// dramatic-looking line can't exaggerate a tiny one. The axis labels ARE the
// receipts.

// Render a series as a vertical-ish line chart using block heights per column.
// Used for loss-over-steps. We sample/compress to `width` columns so a 2000-step
// run still fits 60 chars; compression is mean-pooling, stated in the caption so
// nobody reads a smoothed curve as raw.
export function asciiLine(series: number[], width = 60, height = 12): string {
  if (series.length === 0) return "(empty series)";
  // Compress to `width` buckets by mean-pooling. If series shorter than width,
  // we just use it as-is (no upsampling — inventing points would be dishonest).
  const cols = compress(series, width);
  const min = Math.min(...cols);
  const max = Math.max(...cols);
  const span = max - min || 1; // avoid /0 when the series is constant
  const grid: string[][] = Array.from({ length: height }, () =>
    new Array(cols.length).fill(" "),
  );
  for (let x = 0; x < cols.length; x++) {
    // Map value to a row; higher value = higher up (row 0 is top).
    const norm = (cols[x] - min) / span;
    const row = Math.round((1 - norm) * (height - 1));
    grid[row][x] = "●";
  }
  const body = grid.map((r) => "│" + r.join("")).join("\n");
  const axis = "└" + "─".repeat(cols.length);
  return (
    `${body}\n${axis}\n` +
    `  top=${fmt(max)}  bottom=${fmt(min)}  n=${series.length}` +
    (cols.length < series.length ? ` (mean-pooled to ${cols.length} cols)` : "")
  );
}

// Horizontal bar chart for distributions / per-item similarity. Labels left,
// bars right, real value printed at the end of each bar so the bar length is
// decorative, not the data of record.
export function asciiBar(labels: string[], values: number[], barWidth = 30): string {
  if (labels.length !== values.length) {
    throw new Error("asciiBar: labels and values length mismatch");
  }
  if (labels.length === 0) return "(no bars)";
  const max = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  const labelW = Math.max(...labels.map((l) => l.length));
  return labels
    .map((l, i) => {
      const v = values[i];
      const filled = Math.round((Math.abs(v) / max) * barWidth);
      const bar = "█".repeat(filled) + "░".repeat(barWidth - filled);
      return `${l.padEnd(labelW)} │${bar}│ ${fmt(v)}`;
    })
    .join("\n");
}

// Heatmap for similarity / co-occurrence matrices. Maps each cell to one of a few
// shade characters by its position in [min,max]. Square-ish matrices only; we
// print the scale legend so a dark cell's actual number is recoverable.
export function asciiHeatmap(matrix: number[][], labels?: string[]): string {
  if (matrix.length === 0) return "(empty matrix)";
  const shades = " .:-=+*#%@"; // 10 levels, light → dark
  let min = Infinity;
  let max = -Infinity;
  for (const row of matrix) {
    for (const c of row) {
      if (c < min) min = c;
      if (c > max) max = c;
    }
  }
  const span = max - min || 1;
  const lab = labels ?? matrix.map((_, i) => String(i));
  const labelW = Math.max(...lab.map((l) => l.length), 1);
  const lines = matrix.map((row, r) => {
    const cells = row
      .map((c) => {
        const idx = Math.min(shades.length - 1, Math.floor(((c - min) / span) * shades.length));
        // double the glyph so cells read roughly square in a monospace terminal
        return shades[idx] + shades[idx];
      })
      .join("");
    return `${(lab[r] ?? String(r)).padEnd(labelW)} ${cells}`;
  });
  return `${lines.join("\n")}\n  scale "${shades}" → [${fmt(min)} .. ${fmt(max)}]`;
}

// Scatter of 2D points (after PCA/t-SNE) onto a character grid, with single-char
// labels. Collisions (two points same cell) are marked '*' so the reader knows a
// cluster collapsed rather than seeing one silently overwrite another.
export function asciiScatter(
  points: Array<[number, number]>,
  labels: string[],
  width = 50,
  height = 20,
): string {
  if (points.length === 0) return "(no points)";
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const grid: string[][] = Array.from({ length: height }, () => new Array(width).fill(" "));
  points.forEach((p, i) => {
    const cx = Math.min(width - 1, Math.round(((p[0] - minX) / spanX) * (width - 1)));
    // invert y so larger y is higher on screen
    const cy = Math.min(height - 1, Math.round((1 - (p[1] - minY) / spanY) * (height - 1)));
    const ch = (labels[i] ?? "?")[0];
    grid[cy][cx] = grid[cy][cx] === " " ? ch : "*";
  });
  const body = grid.map((r) => "│" + r.join("") + "│").join("\n");
  const border = "+" + "─".repeat(width) + "+";
  return `${border}\n${body}\n${border}\n  x∈[${fmt(minX)},${fmt(maxX)}] y∈[${fmt(minY)},${fmt(maxY)}]`;
}

// Mean-pool `series` down to at most `width` buckets. Pure helper; never invents
// data (returns input untouched when already short enough).
function compress(series: number[], width: number): number[] {
  if (series.length <= width) return series.slice();
  const out: number[] = [];
  const bucket = series.length / width;
  for (let i = 0; i < width; i++) {
    const start = Math.floor(i * bucket);
    const end = Math.floor((i + 1) * bucket);
    let sum = 0;
    for (let j = start; j < end; j++) sum += series[j];
    out.push(sum / Math.max(1, end - start));
  }
  return out;
}

// Compact number formatter shared by all renderers so axes line up.
function fmt(x: number): string {
  if (!isFinite(x)) return String(x);
  if (Math.abs(x) >= 1000 || (Math.abs(x) < 0.001 && x !== 0)) return x.toExponential(2);
  return x.toFixed(3);
}
