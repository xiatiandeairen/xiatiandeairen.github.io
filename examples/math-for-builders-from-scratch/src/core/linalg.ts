// core/linalg.ts — Minimal vector/matrix ops on plain number[] / number[][].
//
// WHY plain arrays and not a typed-array BLAS wrapper:
//   This book teaches the MECHANISM of linear algebra (PCA, gradient descent, the DFT
//   as a matmul). A reader must be able to read every line and see exactly which
//   multiply-add happens. Readability beats throughput here; the honest-number stages
//   that care about speed (stage12-hardware) measure naive vs blocked variants and say
//   so explicitly rather than hiding behind a library.
//
// CONVENTION: a matrix is row-major number[][]; m[i] is row i, m[i][j] is (row i, col j).
//   A vector is number[]. We do NOT introduce a Matrix class — fewer abstractions to
//   learn, and the shapes stay visible at every call site.
//
// FAILURE MODE (shared by all ops): JS arrays silently allow ragged/mismatched shapes.
//   A 3-long vector dotted with a 4-long vector will NOT throw on its own; it would just
//   read undefined and produce NaN. We add cheap dimension checks so the failure is loud
//   and points at the real bug instead of surfacing as NaN three functions later.

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`[linalg] ${msg}`);
}

/** Inner product. INVARIANT: a.length === b.length. */
export function dot(a: readonly number[], b: readonly number[]): number {
  assert(a.length === b.length, `dot: length ${a.length} != ${b.length}`);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Element-wise a + b -> new vector. */
export function add(a: readonly number[], b: readonly number[]): number[] {
  assert(a.length === b.length, `add: length ${a.length} != ${b.length}`);
  return a.map((v, i) => v + b[i]);
}

/** Scalar multiple k*a -> new vector. */
export function scale(a: readonly number[], k: number): number[] {
  return a.map((v) => v * k);
}

/** Euclidean (L2) norm. */
export function norm(a: readonly number[]): number {
  return Math.sqrt(dot(a, a));
}

/**
 * Unit vector a / |a|.
 * FAILURE MODE: the zero vector has no direction; we return a fresh zero copy rather
 *   than dividing by 0 (which would yield NaNs). Callers that must distinguish "was
 *   zero" should check norm() first — silent zero-return is a deliberate, documented
 *   choice to keep downstream math finite.
 */
export function normalize(a: readonly number[]): number[] {
  const n = norm(a);
  if (n === 0) return a.map(() => 0);
  return scale(a, 1 / n);
}

/** Transpose of an m x n matrix -> n x m. INVARIANT: input is rectangular. */
export function transpose(m: readonly number[][]): number[][] {
  const rows = m.length;
  if (rows === 0) return [];
  const cols = m[0].length;
  const out: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    assert(m[i].length === cols, `transpose: ragged row ${i}`);
    for (let j = 0; j < cols; j++) out[j][i] = m[i][j];
  }
  return out;
}

/**
 * Matrix product a (m x k) * b (k x n) -> (m x n), schoolbook triple loop.
 * INVARIANT: inner dims agree (a's cols === b's rows).
 * WHY ikj loop order: i-k-j touches b row-major in the inner loop, which is the
 *   cache-friendly order — stage12 measures exactly this against ijk to show the
 *   memory hierarchy is real, not folklore.
 */
export function matmul(a: readonly number[][], b: readonly number[][]): number[][] {
  const m = a.length;
  const k = a[0]?.length ?? 0;
  const k2 = b.length;
  const n = b[0]?.length ?? 0;
  assert(k === k2, `matmul: inner dim ${k} != ${k2}`);
  const out: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i][p];
      const brow = b[p];
      const orow = out[i];
      for (let j = 0; j < n; j++) orow[j] += aip * brow[j];
    }
  }
  return out;
}

/** Matrix-vector product m (r x c) * v (c) -> (r). INVARIANT: c === v.length. */
export function matVec(m: readonly number[][], v: readonly number[]): number[] {
  return m.map((row) => dot(row, v));
}

/** n x n identity matrix. */
export function identity(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
}
