// core/assert.ts — executable invariants + the one table printer the book uses.
//
// Why a tiny assert module instead of node:assert: this book's claims ARE
// invariants — "at most one leader per term", "committed entries are never
// lost", "all replicas converge to the same state". Making those executable and
// failing LOUDLY with a domain message (plus a state snapshot) is how the book
// proves it isn't hand-waving. node:assert works but its messages are generic;
// these wrappers force a why-sentence at every call site.
//
// invariant() is special: it is meant to be re-checked after EVERY simulation
// tick (the scenario runner calls a set of named invariant fns per event). The
// classic safety property "≤1 leader/term" is meaningless if only checked at the
// end — split-brain can occur transiently mid-run and "heal" before you look.
// So invariants are continuous guards, not final assertions.
//
// Failure-mode philosophy: a fired invariant is NOT a handled error — it is a
// bug in the protocol the chapter just built. So these throw and crash the demo
// (with a snapshot) rather than returning a status. A silently-skipped invariant
// is worse than a stack trace, because it lets a broken consensus look correct.

/** Assert a boolean condition. The message must state the INVARIANT, not just
 *  the values, so a failure reads like a sentence ("two leaders in term 4"). */
export function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(`assert failed: ${msg}`);
}

/** Assert scalar equality with expected/actual echo. For end-of-stage checks
 *  like "recovered value == pre-crash value". */
export function assertEq<T>(actual: T, expected: T, msg: string): void {
  if (actual !== expected) {
    throw new Error(
      `assertEq failed: ${msg}\n  expected: ${String(expected)}\n  actual:   ${String(actual)}`,
    );
  }
}

/** A named, re-runnable global invariant. `check` returns true when the property
 *  HOLDS. `snapshot` (optional) is called only on violation to dump cluster
 *  state into the error — so the failure message shows WHY, not just THAT.
 *  Stages register these once; the scenario runner evaluates them every tick. */
export interface Invariant {
  name: string;
  check: () => boolean;
  snapshot?: () => unknown;
}

/** Build an Invariant. Sugar so call sites read declaratively:
 *  `invariant("at-most-one-leader", () => leaders <= 1, () => cluster.dump())`. */
export function invariant(
  name: string,
  check: () => boolean,
  snapshot?: () => unknown,
): Invariant {
  return { name, check, snapshot };
}

/** Evaluate all invariants; throw with the name + snapshot of the FIRST that
 *  fails. Called by the scenario runner after each event. Throwing on first
 *  failure (not collecting all) is deliberate: the first violated safety
 *  property is the root cause; later ones are usually fallout. */
export function checkInvariants(invs: readonly Invariant[]): void {
  for (const inv of invs) {
    if (!inv.check()) {
      const snap = inv.snapshot ? `\n  state: ${JSON.stringify(inv.snapshot(), null, 2)}` : "";
      throw new Error(`invariant violated: "${inv.name}"${snap}`);
    }
  }
}

export type TableRow = Record<string, string | number>;

/** Print rows as an aligned ASCII table. Strings left-align, numbers right-align
 *  (digits line up under each other — the readability point of a metrics table).
 *  Columns are the union of all row keys in first-seen order, so the caller
 *  controls layout via key insertion order. One printer across all stages keeps
 *  numbers visually comparable down the book. */
export function printTable(rows: TableRow[]): void {
  if (rows.length === 0) {
    console.log("(no rows)");
    return;
  }
  const cols: string[] = [];
  for (const row of rows) for (const k of Object.keys(row)) if (!cols.includes(k)) cols.push(k);

  const isNum = (c: string) => rows.every((r) => r[c] === undefined || typeof r[c] === "number");
  const cell = (r: TableRow, c: string) => (r[c] === undefined ? "" : String(r[c]));

  const width: Record<string, number> = {};
  for (const c of cols) {
    width[c] = c.length;
    for (const r of rows) width[c] = Math.max(width[c], cell(r, c).length);
  }

  const pad = (s: string, w: number, right: boolean) => (right ? s.padStart(w) : s.padEnd(w));

  console.log(cols.map((c) => pad(c, width[c], isNum(c))).join("  "));
  console.log(cols.map((c) => "-".repeat(width[c])).join("  "));
  for (const r of rows) {
    console.log(cols.map((c) => pad(cell(r, c), width[c], isNum(c))).join("  "));
  }
}
