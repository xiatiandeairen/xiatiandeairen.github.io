// core/assert.ts — invariant checks and the one table printer the whole book uses.
//
// Why a tiny assert module instead of node:assert: this book's claims are
// invariants ("the tree stays balanced", "keys stay sorted", "the recovered DB
// equals the pre-crash DB"). Making those invariants executable — and failing
// LOUDLY with a domain message when violated — is how the book proves it isn't
// hand-waving. node:assert works but its messages are generic; these wrappers
// force a why-message at every call site.
//
// printTable is here, not in some ui module, on purpose: every stage ends by
// printing MEASURED numbers, and if each stage formatted its own table the
// numbers would be visually incomparable across chapters. One printer ==
// one column discipline == numbers a reader can scan down the book.
//
// Failure-mode philosophy: an assert that fires is not a handled error, it is a
// bug in the engine the chapter just built. So these throw plain Error (crash
// the demo) rather than returning a status — a silently-skipped invariant is
// worse than a stack trace.

import { compareKeys } from "./codec.js";

/** Assert structural equality of two scalars; the message states the invariant,
 *  not just the values, so a failure reads like a sentence ("rows survived crash"). */
export function assertEq<T>(actual: T, expected: T, msg: string): void {
  if (actual !== expected) {
    throw new Error(`assertEq failed: ${msg}\n  expected: ${String(expected)}\n  actual:   ${String(actual)}`);
  }
}

/** General invariant guard. Use for conditions that should be impossible if the
 *  engine is correct (e.g. "free_end >= free_start"). */
export function invariant(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(`invariant violated: ${msg}`);
}

/** Assert an array of encoded keys is non-decreasing under the book's byte order.
 *  This is the index layer's master invariant; B+tree and LSM call it after
 *  builds/merges. Catches a mis-sorted merge immediately instead of as a wrong
 *  query result a thousand lines later. */
export function assertSorted(keys: Uint8Array[], msg: string): void {
  for (let i = 1; i < keys.length; i++) {
    if (compareKeys(keys[i - 1], keys[i]) > 0) {
      throw new Error(`assertSorted failed at index ${i}: ${msg}`);
    }
  }
}

export type TableRow = Record<string, string | number>;

/** Print rows as a left/right-aligned ASCII table. Strings left-align, numbers
 *  right-align (so digits line up under each other — the readability point of a
 *  metrics table). Columns are the union of all row keys, in first-seen order so
 *  the caller controls layout by key insertion order. */
export function printTable(rows: TableRow[]): void {
  if (rows.length === 0) {
    console.log("(no rows)");
    return;
  }
  // First-seen column order across all rows; later rows may add columns.
  const cols: string[] = [];
  for (const row of rows) for (const k of Object.keys(row)) if (!cols.includes(k)) cols.push(k);

  const isNum = (c: string) => rows.every((r) => r[c] === undefined || typeof r[c] === "number");
  const cell = (r: TableRow, c: string) => (r[c] === undefined ? "" : String(r[c]));

  const width: Record<string, number> = {};
  for (const c of cols) {
    width[c] = c.length;
    for (const r of rows) width[c] = Math.max(width[c], cell(r, c).length);
  }

  const pad = (s: string, w: number, right: boolean) =>
    right ? s.padStart(w) : s.padEnd(w);

  const header = cols.map((c) => pad(c, width[c], isNum(c))).join("  ");
  const sep = cols.map((c) => "-".repeat(width[c])).join("  ");
  console.log(header);
  console.log(sep);
  for (const r of rows) {
    console.log(cols.map((c) => pad(cell(r, c), width[c], isNum(c))).join("  "));
  }
}
