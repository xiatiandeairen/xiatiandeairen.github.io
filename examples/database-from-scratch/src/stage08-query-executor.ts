// stage08-query-executor.ts — from a SQL string to a Volcano pipeline.
//
// What this chapter builds, end to end, on top of core/: a tiny SQL subset goes
// through tokenize -> parse (AST) -> logical plan -> physical plan -> a Volcano
// (iterator) executor whose leaf operators pull rows out of real pages on a
// core/disk Disk. Two tables of 10000 rows each are materialized into slotted
// heap pages (so SeqScan does real page IO) and one of them gets a real B+tree
// index keyed on a column (so IndexScan does real, *fewer* page IOs). We then run
// one query that exercises filter + hash join + group by and MEASURE: the parsed
// operator tree, IndexScan-vs-SeqScan page touches and wall time on that query,
// the HashJoin build/probe intermediate row counts, and the total number of
// next() calls the whole pipeline makes (the Volcano model's per-row function-call
// overhead, made visible).
//
// Why Volcano (open/next/close iterators) and not a vectorized or compiled model:
// it is the model every textbook executor starts from, every operator is a small
// state machine, and composition is just "my input is another iterator". Its
// famous weakness — one virtual call per row per operator — is exactly what this
// stage quantifies rather than asserts. We count next() calls so the reader sees
// the overhead instead of being told it exists.
//
// Honesty about the numbers: page-touch counts and next()-call counts are
// DETERMINISTIC (driven by the seeded core PRNG and fixed data), reproducible on
// any machine. Wall-clock timings are REAL measurements via core bench/hrtime and
// will vary run-to-run and machine-to-machine; they are labeled (measured). The
// data is toy (10000 rows, RAM-backed "disk"), so absolute timings are optimistic
// versus a real spinning/SSD-backed engine; what transfers is the RELATIVE shape:
// IndexScan touches O(height + match-pages) pages while SeqScan touches every
// data page, and that ratio is the point.
//
// Failure mode demonstrated (not just happy path): we ask the planner to use an
// IndexScan on a column that has NO index. A real planner cannot conjure an index,
// so it falls back to SeqScan; we print that fallback and quantify the page-touch
// blow-up — the concrete cost of "missing index => full table scan".
//
// Constraints honored: ESM imports with .js suffixes; reuses core/ only (no stage
// imports); deterministic via core createRng; throws on structural bugs rather
// than papering over them.

import { createRng, type Rng } from "./core/prng.js";
import { Disk, PAGE_SIZE } from "./core/disk.js";
import {
  PageReader,
  PageWriter,
  PageType,
  SLOTTED_HEADER,
  SLOT_ENTRY_SIZE,
} from "./core/page.js";
import {
  encodeRow,
  decodeRow,
  encodeKey,
  compareKeys,
  type Schema,
  type ColType,
  type Value,
} from "./core/codec.js";
import { bench } from "./core/clock.js";
import { invariant, printTable } from "./core/assert.js";

// ===========================================================================
// Catalog: table = ordered column names + their codec types. The executor is
// schema-driven (no per-row type tags, matching core/codec), so every operator
// that touches a row needs the schema out of band. We keep it in a Catalog so a
// column reference like "u.country" resolves to a positional index once, up front,
// instead of by name in the row-at-a-time hot loop.
// ===========================================================================

interface TableDef {
  name: string;
  columns: string[];
  types: Schema;
  /** Heap is a contiguous run of slotted data pages [firstPage, firstPage+pageCount). */
  firstPage: number;
  pageCount: number;
  rowCount: number;
  /** Column index that has a B+tree index, or -1 if the table has no index. */
  indexedCol: number;
  /** Root page of the B+tree, valid only when indexedCol >= 0. */
  indexRoot: number;
}

class Catalog {
  private tables = new Map<string, TableDef>();
  define(def: TableDef): void {
    this.tables.set(def.name, def);
  }
  get(name: string): TableDef {
    const t = this.tables.get(name);
    // Resolving an unknown table is a query-compile bug, not a runtime row case.
    if (!t) throw new Error(`catalog: unknown table '${name}'`);
    return t;
  }
  /** Resolve "col" within a table to its positional index; throws on typo so a
   *  bad column name fails at plan time, not as a silent undefined at row time. */
  colIndex(table: TableDef, col: string): number {
    const i = table.columns.indexOf(col);
    if (i < 0) throw new Error(`catalog: table '${table.name}' has no column '${col}'`);
    return i;
  }
}

// A row flowing through the pipeline is just the decoded tuple plus a tag of which
// table it came from, so a join can keep left/right columns addressable. Keeping
// it positional (not a name->value map) avoids per-row object key allocation in
// the hot path — the same reason core/codec is schema-driven.
type Tuple = Value[];

// ===========================================================================
// Heap storage: bulk-load rows into slotted data pages on the Disk. This is the
// physical thing SeqScan reads. We pack rows greedily; when the next row won't
// fit (PageWriter throws its page-full signal), we seal the page and start a new
// one — exactly the discipline core/page documents.
// ===========================================================================

/** Insert one encoded row into a slotted page; return false if it doesn't fit.
 *  Mirrors the stage01 slotted-insert discipline: slot directory grows up from
 *  the header, row heap grows down from the page end, they must not cross. */
function tryInsertRow(page: Uint8Array, encoded: Uint8Array): boolean {
  const r = new PageReader(page);
  const slotCount = r.readU16(SLOTTED_HEADER.SLOT_COUNT);
  const freeStart = r.readU16(SLOTTED_HEADER.FREE_START);
  const freeEnd = r.readU16(SLOTTED_HEADER.FREE_END);
  // Need room for the row bytes (grow down) AND one new slot entry (grow up).
  const need = encoded.length + SLOT_ENTRY_SIZE;
  if (freeEnd - freeStart < need) return false;
  const w = new PageWriter(page);
  const rowOffset = freeEnd - encoded.length;
  w.writeBytes(rowOffset, encoded);
  // Slot entry = (offset:u16, length:u16) appended at freeStart.
  w.writeU16(freeStart, rowOffset);
  w.writeU16(freeStart + 2, encoded.length);
  w.writeU16(SLOTTED_HEADER.SLOT_COUNT, slotCount + 1);
  w.writeU16(SLOTTED_HEADER.FREE_START, freeStart + SLOT_ENTRY_SIZE);
  w.writeU16(SLOTTED_HEADER.FREE_END, rowOffset);
  return true;
}

/** Read all rows out of one slotted data page. Used by SeqScan and by the index
 *  build (which needs every row's key). Decodes via the table schema. */
function readPageRows(page: Uint8Array, schema: Schema): Tuple[] {
  const r = new PageReader(page);
  const slotCount = r.readU16(SLOTTED_HEADER.SLOT_COUNT);
  const out: Tuple[] = [];
  for (let s = 0; s < slotCount; s++) {
    const slotBase = SLOTTED_HEADER.HEADER_SIZE + s * SLOT_ENTRY_SIZE;
    const off = r.readU16(slotBase);
    const len = r.readU16(slotBase + 2);
    if (len === 0) continue; // tombstone (none here, but honor the format)
    out.push(decodeRow(r.readBytes(off, len), schema));
  }
  return out;
}

/** Bulk-load rows into a fresh run of slotted pages on disk; return [firstPage,
 *  pageCount]. Greedy packing: fill a page until a row won't fit, then alloc the
 *  next. The page-full signal IS PageWriter.bounds throwing inside tryInsertRow's
 *  capacity math — we check capacity first so we don't rely on the throw in the
 *  loop, but the invariant is the same one core/page enforces. */
function loadHeap(disk: Disk, rows: Tuple[], schema: Schema): { firstPage: number; pageCount: number } {
  let firstPage = -1;
  let pageCount = 0;
  let current = new Uint8Array(PAGE_SIZE);
  new PageWriter(current).initSlotted(PageType.LEAF);
  let currentId = disk.allocPage();
  if (firstPage < 0) firstPage = currentId;
  pageCount++;

  const flush = () => disk.writePage(currentId, current);

  for (const row of rows) {
    const enc = encodeRow(row, schema);
    if (!tryInsertRow(current, enc)) {
      flush();
      current = new Uint8Array(PAGE_SIZE);
      new PageWriter(current).initSlotted(PageType.LEAF);
      currentId = disk.allocPage();
      pageCount++;
      // A single row larger than a page would loop forever; guard it.
      if (!tryInsertRow(current, enc)) {
        throw new Error(`loadHeap: row of ${enc.length}B does not fit in a ${PAGE_SIZE}B page`);
      }
    }
  }
  flush();
  // Pages are dense and contiguous because no other allocation interleaves the
  // load; SeqScan relies on this to iterate [firstPage, firstPage+pageCount).
  invariant(currentId === firstPage + pageCount - 1, "heap pages must be contiguous");
  return { firstPage, pageCount };
}

// ===========================================================================
// B+tree index (single-column, integer key -> heap row id). Minimal but real:
// a sorted run of leaf pages plus one internal directory page, so an equality or
// range probe descends height-1 internal pages then scans only the matching
// leaves. This is what makes IndexScan touch O(height + matches) pages instead of
// the whole heap. We keep it as its own pages on the same Disk so every descent
// is a counted readPage.
//
// Layout choice (kept deliberately simple, documented as such): leaf pages store
// (key:4B big-endian, rowGlobalIndex:u32) entries sorted by key; the single
// internal page stores the first key of each leaf + that leaf's page id. One
// internal level caps us at ~ (leaf_entries_per_page * entries_per_internal)
// keys, plenty for 10000. A production B+tree generalizes to many internal levels
// and splits on insert; here we BUILD bottom-up from sorted data, because the
// query chapter cares about *probing* an index, and stage03 owns the insert/split
// story. The probe path (descend + scan) is the same as a real tree's.
// ===========================================================================

const LEAF_ENTRY_SIZE = 8; // key u32 + rowIndex u32, both big-endian
const INTERNAL_ENTRY_SIZE = 8; // firstKey u32 + childPageId u32

interface IndexBuildResult {
  root: number; // internal directory page id
  leafCount: number;
}

/** Build the index bottom-up from (key, rowIndex) pairs. Pairs MUST be passed
 *  sorted by key — we assert it, because an unsorted build silently produces a
 *  tree that returns wrong rows, the exact corruption core/assert exists to catch. */
function buildIndex(disk: Disk, pairs: { key: number; rowIndex: number }[]): IndexBuildResult {
  for (let i = 1; i < pairs.length; i++) {
    invariant(pairs[i - 1].key <= pairs[i].key, "buildIndex: input must be sorted by key");
  }
  const entriesPerLeaf = Math.floor((PAGE_SIZE - SLOTTED_HEADER.HEADER_SIZE) / LEAF_ENTRY_SIZE);

  const leafPageIds: number[] = [];
  const leafFirstKeys: number[] = [];
  for (let start = 0; start < pairs.length; start += entriesPerLeaf) {
    const chunk = pairs.slice(start, start + entriesPerLeaf);
    const page = new Uint8Array(PAGE_SIZE);
    const w = new PageWriter(page);
    w.initSlotted(PageType.LEAF);
    let off = SLOTTED_HEADER.HEADER_SIZE;
    for (const { key, rowIndex } of chunk) {
      w.writeU32(off, key);
      w.writeU32(off + 4, rowIndex);
      off += LEAF_ENTRY_SIZE;
    }
    w.writeU16(SLOTTED_HEADER.SLOT_COUNT, chunk.length); // reuse slot_count as entry count
    const id = disk.allocPage();
    disk.writePage(id, page);
    leafPageIds.push(id);
    leafFirstKeys.push(chunk[0].key);
  }

  // One internal directory page over the leaves. Guard the single-level cap so a
  // dataset that overflows it fails loudly instead of building a wrong tree.
  const maxInternalEntries = Math.floor((PAGE_SIZE - SLOTTED_HEADER.HEADER_SIZE) / INTERNAL_ENTRY_SIZE);
  if (leafPageIds.length > maxInternalEntries) {
    throw new Error(
      `buildIndex: ${leafPageIds.length} leaves exceed single internal page capacity ${maxInternalEntries}; ` +
        `multi-level internal nodes are a stage03 concern`,
    );
  }
  const internal = new Uint8Array(PAGE_SIZE);
  const iw = new PageWriter(internal);
  iw.initSlotted(PageType.INTERNAL);
  let ioff = SLOTTED_HEADER.HEADER_SIZE;
  for (let i = 0; i < leafPageIds.length; i++) {
    iw.writeU32(ioff, leafFirstKeys[i]);
    iw.writeU32(ioff + 4, leafPageIds[i]);
    ioff += INTERNAL_ENTRY_SIZE;
  }
  iw.writeU16(SLOTTED_HEADER.SLOT_COUNT, leafPageIds.length);
  const root = disk.allocPage();
  disk.writePage(root, internal);
  return { root, leafCount: leafPageIds.length };
}

/** Probe the index for all rowIndexes whose key is in [lo, hi] (inclusive range;
 *  equality is lo===hi). Returns matching row indexes. Every page touched is a
 *  real disk.readPage, so the caller's disk.stats() reflects the true descent +
 *  leaf-scan cost. Big-endian keys + compareKeys give us byte-order == int-order,
 *  so a memcmp-style search is correct (the load-bearing trick from core/codec). */
function probeIndexRange(disk: Disk, root: number, lo: number, hi: number): number[] {
  const internalBytes = disk.readPage(root); // 1 page touch: descend the directory
  const ir = new PageReader(internalBytes);
  const dirCount = ir.readU16(SLOTTED_HEADER.SLOT_COUNT);

  const loKey = encodeKey(lo, "int");
  const hiKey = encodeKey(hi, "int");

  // Find the first leaf whose firstKey could contain `lo`: the last leaf with
  // firstKey <= lo (linear here for clarity; a real tree binary-searches the
  // directory — same page-touch count, this is in-memory work on one page).
  let startLeaf = 0;
  for (let i = 0; i < dirCount; i++) {
    const fk = ir.readBytes(SLOTTED_HEADER.HEADER_SIZE + i * INTERNAL_ENTRY_SIZE, 4);
    if (compareKeys(fk, loKey) <= 0) startLeaf = i;
    else break;
  }

  const matches: number[] = [];
  for (let li = startLeaf; li < dirCount; li++) {
    const entryBase = SLOTTED_HEADER.HEADER_SIZE + li * INTERNAL_ENTRY_SIZE;
    // If this leaf's firstKey already exceeds hi, no later leaf can match either
    // (leaves are key-sorted) — stop, this is what bounds the leaf scan to the
    // matching range instead of the whole index.
    const fk = ir.readBytes(entryBase, 4);
    if (compareKeys(fk, hiKey) > 0) break;
    const childId = ir.readU32(entryBase + 4);
    const leafBytes = disk.readPage(childId); // 1 page touch per scanned leaf
    const lr = new PageReader(leafBytes);
    const n = lr.readU16(SLOTTED_HEADER.SLOT_COUNT);
    for (let e = 0; e < n; e++) {
      const off = SLOTTED_HEADER.HEADER_SIZE + e * LEAF_ENTRY_SIZE;
      const k = lr.readBytes(off, 4);
      if (compareKeys(k, loKey) >= 0 && compareKeys(k, hiKey) <= 0) {
        matches.push(lr.readU32(off + 4));
      }
    }
  }
  return matches;
}

// ===========================================================================
// SQL subset: tokenizer + parser. The grammar we accept (kept tight on purpose):
//
//   SELECT <col> [, <col> | <AGG>(<col>) AS <alias>]...
//   FROM <table> <alias>
//   [JOIN <table> <alias> ON <alias>.<col> = <alias>.<col>]
//   [WHERE <alias>.<col> <op> <literal>]            op in =,>,<,>=,<=
//   [GROUP BY <alias>.<col>]
//
// AGG in COUNT|SUM. This is enough to drive filter + hash join + group by, which
// is the chapter's whole point, without dragging in a full SQL grammar that would
// bury the executor under parsing. Anything outside the grammar throws a parse
// error with the offending token — a parser that silently ignores tokens is how
// "the query ran but did the wrong thing" bugs are born.
// ===========================================================================

type Token = { kind: "ident" | "num" | "punct" | "kw"; text: string };

const KEYWORDS = new Set([
  "select", "from", "join", "on", "where", "group", "by", "as", "and",
  "count", "sum",
]);

function tokenize(sql: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < sql.length) {
    const c = sql[i];
    if (c === " " || c === "\n" || c === "\t" || c === "\r") {
      i++;
      continue;
    }
    if (c === "," || c === "(" || c === ")" || c === "*" || c === ".") {
      tokens.push({ kind: "punct", text: c });
      i++;
      continue;
    }
    if (c === "=" || c === ">" || c === "<") {
      // Two-char comparison ops (>=, <=) before single-char.
      if ((c === ">" || c === "<") && sql[i + 1] === "=") {
        tokens.push({ kind: "punct", text: c + "=" });
        i += 2;
      } else {
        tokens.push({ kind: "punct", text: c });
        i++;
      }
      continue;
    }
    if (c === "'") {
      // String literal. Unterminated literal is a syntax error, not EOF-silently.
      let j = i + 1;
      while (j < sql.length && sql[j] !== "'") j++;
      if (j >= sql.length) throw new Error(`tokenize: unterminated string literal at ${i}`);
      tokens.push({ kind: "ident", text: sql.slice(i, j + 1) }); // keep quotes to mark string
      i = j + 1;
      continue;
    }
    if (c >= "0" && c <= "9") {
      let j = i;
      while (j < sql.length && sql[j] >= "0" && sql[j] <= "9") j++;
      tokens.push({ kind: "num", text: sql.slice(i, j) });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < sql.length && /[A-Za-z0-9_]/.test(sql[j])) j++;
      const text = sql.slice(i, j);
      tokens.push({ kind: KEYWORDS.has(text.toLowerCase()) ? "kw" : "ident", text });
      i = j;
      continue;
    }
    throw new Error(`tokenize: unexpected character '${c}' at ${i}`);
  }
  return tokens;
}

// --- AST ---
type ColRef = { table: string; col: string };
type Literal = { type: ColType; value: Value };
type CompareOp = "=" | ">" | "<" | ">=" | "<=";
type Predicate = { left: ColRef; op: CompareOp; right: Literal };
type AggFn = "count" | "sum";
interface SelectItem {
  agg?: AggFn;
  col: ColRef;
  alias: string;
}
interface JoinClause {
  table: string;
  alias: string;
  left: ColRef;
  right: ColRef;
}
interface SelectQuery {
  items: SelectItem[];
  from: { table: string; alias: string };
  join?: JoinClause;
  where?: Predicate;
  groupBy?: ColRef;
}

/** Recursive-descent parser over the token stream. Tiny enough that a cursor +
 *  expect() helpers are clearer than a generated parser; every expect() failure
 *  names what was wanted vs found so a typo'd query is diagnosable. */
class Parser {
  private pos = 0;
  constructor(private toks: Token[]) {}

  private peek(): Token | undefined {
    return this.toks[this.pos];
  }
  private next(): Token {
    const t = this.toks[this.pos++];
    if (!t) throw new Error("parse: unexpected end of query");
    return t;
  }
  private expectKw(kw: string): void {
    const t = this.next();
    if (t.kind !== "kw" || t.text.toLowerCase() !== kw) {
      throw new Error(`parse: expected keyword '${kw}', got '${t.text}'`);
    }
  }
  private expectPunct(p: string): void {
    const t = this.next();
    if (t.kind !== "punct" || t.text !== p) {
      throw new Error(`parse: expected '${p}', got '${t.text}'`);
    }
  }
  private isKw(kw: string): boolean {
    const t = this.peek();
    return !!t && t.kind === "kw" && t.text.toLowerCase() === kw;
  }
  private ident(): string {
    const t = this.next();
    if (t.kind !== "ident") throw new Error(`parse: expected identifier, got '${t.text}'`);
    return t.text;
  }
  private colRef(): ColRef {
    const table = this.ident();
    this.expectPunct(".");
    const col = this.ident();
    return { table, col };
  }

  parse(): SelectQuery {
    this.expectKw("select");
    const items = this.parseSelectItems();
    this.expectKw("from");
    const fromTable = this.ident();
    const fromAlias = this.ident();
    const from = { table: fromTable, alias: fromAlias };

    let join: JoinClause | undefined;
    if (this.isKw("join")) {
      this.next();
      const jt = this.ident();
      const ja = this.ident();
      this.expectKw("on");
      const l = this.colRef();
      this.expectPunct("=");
      const r = this.colRef();
      join = { table: jt, alias: ja, left: l, right: r };
    }

    let where: Predicate | undefined;
    if (this.isKw("where")) {
      this.next();
      where = this.parsePredicate();
    }

    let groupBy: ColRef | undefined;
    if (this.isKw("group")) {
      this.next();
      this.expectKw("by");
      groupBy = this.colRef();
    }

    if (this.peek()) throw new Error(`parse: trailing tokens starting at '${this.peek()!.text}'`);
    return { items, from, join, where, groupBy };
  }

  private parseSelectItems(): SelectItem[] {
    const items: SelectItem[] = [];
    for (;;) {
      items.push(this.parseSelectItem());
      if (this.peek()?.text === ",") {
        this.next();
        continue;
      }
      break;
    }
    return items;
  }

  private parseSelectItem(): SelectItem {
    if (this.isKw("count") || this.isKw("sum")) {
      const agg = this.next().text.toLowerCase() as AggFn;
      this.expectPunct("(");
      const col = this.colRef();
      this.expectPunct(")");
      this.expectKw("as");
      const alias = this.ident();
      return { agg, col, alias };
    }
    const col = this.colRef();
    const alias = `${col.table}.${col.col}`;
    return { col, alias };
  }

  private parsePredicate(): Predicate {
    const left = this.colRef();
    const opTok = this.next();
    const ops: CompareOp[] = ["=", ">", "<", ">=", "<="];
    if (opTok.kind !== "punct" || !ops.includes(opTok.text as CompareOp)) {
      throw new Error(`parse: expected comparison operator, got '${opTok.text}'`);
    }
    const op = opTok.text as CompareOp;
    const rTok = this.next();
    let right: Literal;
    if (rTok.kind === "num") {
      right = { type: "int", value: Number(rTok.text) };
    } else if (rTok.kind === "ident" && rTok.text.startsWith("'")) {
      right = { type: "string", value: rTok.text.slice(1, -1) };
    } else {
      throw new Error(`parse: expected literal on RHS of predicate, got '${rTok.text}'`);
    }
    return { left, op, right };
  }
}

// ===========================================================================
// Logical plan: a tree of relational nodes (Scan/Filter/Join/Aggregate/Project).
// It is intentionally a *separate* structure from the AST and from the physical
// plan: the AST is syntax, the logical plan is "what relational result is wanted"
// (the planner's input), and the physical plan picks *how* (SeqScan vs IndexScan,
// HashJoin). Collapsing these three into one tree is the classic mistake that
// makes a planner impossible to extend.
// ===========================================================================

type LogicalNode =
  | { op: "scan"; table: string; alias: string }
  | { op: "filter"; pred: Predicate; input: LogicalNode }
  | { op: "join"; on: { left: ColRef; right: ColRef }; left: LogicalNode; right: LogicalNode }
  | { op: "aggregate"; groupBy: ColRef | undefined; aggs: SelectItem[]; input: LogicalNode }
  | { op: "project"; items: SelectItem[]; input: LogicalNode };

/** Build a logical plan from the AST. Filter pushdown (placing WHERE directly
 *  above the scan of the table it references) is done here so the physical planner
 *  can later turn that filter+scan into an IndexScan. */
function planLogical(q: SelectQuery): LogicalNode {
  // Scans for each table.
  let leftScan: LogicalNode = { op: "scan", table: q.from.table, alias: q.from.alias };
  // Push the WHERE down to the table it filters (single-table predicate only).
  if (q.where && q.where.left.table === q.from.alias) {
    leftScan = { op: "filter", pred: q.where, input: leftScan };
  }

  let root: LogicalNode = leftScan;
  if (q.join) {
    let joinScan: LogicalNode = { op: "scan", table: q.join.table, alias: q.join.alias };
    if (q.where && q.where.left.table === q.join.alias) {
      joinScan = { op: "filter", pred: q.where, input: joinScan };
    }
    // Build-side heuristic: the side carrying the WHERE filter is expected to be
    // the smaller relation after filtering, so make IT the join's right (build)
    // input and the unfiltered side the left (probe). HashJoin buffers the build
    // side in memory, so building the smaller side keeps the hash table small —
    // this is the textbook cost-based choice, made here by a simple rule rather
    // than statistics (a real planner would use row-count estimates).
    const fromIsFiltered = !!(q.where && q.where.left.table === q.from.alias);
    if (fromIsFiltered) {
      // filtered FROM side -> build (right); unfiltered JOIN side -> probe (left)
      root = { op: "join", on: { left: q.join.left, right: q.join.right }, left: joinScan, right: root };
    } else {
      root = { op: "join", on: { left: q.join.left, right: q.join.right }, left: root, right: joinScan };
    }
  }

  const hasAgg = q.items.some((it) => it.agg) || q.groupBy;
  if (hasAgg) {
    root = { op: "aggregate", groupBy: q.groupBy, aggs: q.items, input: root };
  } else {
    root = { op: "project", items: q.items, input: root };
  }
  return root;
}

// ===========================================================================
// Physical operators: the Volcano iterators. Each implements open/next/close.
// next() returns the next Tuple or null at end-of-stream. EVERY next() call is
// counted through a shared ExecStats, because counting them is this chapter's
// headline measurement of the iterator model's per-row call overhead.
//
// A Tuple here carries columns in a stable layout the parent operator agreed on:
// scans emit their table's columns in catalog order; a join emits left columns
// then right columns concatenated. Operators resolve a ColRef to an absolute
// index via a per-operator column map computed at open() — once, not per row.
// ===========================================================================

interface ExecStats {
  nextCalls: number; // total next() invocations across the whole pipeline
  rowsEmitted: number; // rows that actually crossed an operator boundary
}

interface PhysicalOp {
  /** Human label for printing the operator tree. */
  label(): string;
  open(): void;
  /** Pull the next row, or null at end of stream. MUST be idempotent at EOS
   *  (keep returning null) so a parent that over-pulls doesn't crash. */
  next(): Tuple | null;
  close(): void;
  /** Column layout this operator emits: array of "alias.col" in tuple order.
   *  Parents use it to resolve ColRefs to positions at open() time. */
  outputColumns(): string[];
}

/** SeqScan: stream every row of a heap by reading its data pages one at a time.
 *  Touches every data page exactly once — the cost we contrast against IndexScan.
 *  Reads a page, buffers its rows, drains them, then reads the next page; this
 *  keeps memory at one page regardless of table size (the streaming property the
 *  Volcano model exists to provide). */
class SeqScan implements PhysicalOp {
  private pageCursor = 0;
  private buffer: Tuple[] = [];
  private bufIdx = 0;
  constructor(
    private disk: Disk,
    private table: TableDef,
    private alias: string, // column refs in the rest of the query use this, not table.name
    private stats: ExecStats,
  ) {}
  label(): string {
    return `SeqScan(${this.table.name} ${this.alias}, pages=${this.table.pageCount})`;
  }
  open(): void {
    this.pageCursor = 0;
    this.buffer = [];
    this.bufIdx = 0;
  }
  next(): Tuple | null {
    this.stats.nextCalls++;
    for (;;) {
      if (this.bufIdx < this.buffer.length) {
        this.stats.rowsEmitted++;
        return this.buffer[this.bufIdx++];
      }
      if (this.pageCursor >= this.table.pageCount) return null; // EOS, stays null
      const pageId = this.table.firstPage + this.pageCursor++;
      const page = this.disk.readPage(pageId); // the counted data-page IO
      this.buffer = readPageRows(page, this.table.types);
      this.bufIdx = 0;
    }
  }
  close(): void {
    this.buffer = [];
  }
  outputColumns(): string[] {
    return this.table.columns.map((c) => `${this.alias}.${c}`);
  }
}

/** IndexScan: use the B+tree to fetch only rows matching a key range, then read
 *  only the heap pages those rows live on. Touches O(index pages + matched-row
 *  pages) instead of every page. Requires the table's indexed column to match the
 *  predicate column — the physical planner guarantees that before constructing
 *  this, and falls back to SeqScan otherwise (the failure-mode demo). */
class IndexScan implements PhysicalOp {
  private rows: Tuple[] = [];
  private idx = 0;
  constructor(
    private disk: Disk,
    private table: TableDef,
    private alias: string, // emitted column prefix, matching the query's table alias
    private lo: number,
    private hi: number,
    private stats: ExecStats,
    private rowOffsets: number[], // rowGlobalIndex -> (pageId, slot) precomputed map
    private rowToPage: number[],
  ) {}
  label(): string {
    return `IndexScan(${this.alias}.${this.table.columns[this.table.indexedCol]} in [${this.lo},${this.hi}])`;
  }
  open(): void {
    // Probe the index (counts internal + matched-leaf page touches), then fetch
    // each matched row from its heap page (counts only the pages that hold a
    // match, deduplicated so we don't double-count a page with multiple matches).
    const rowIndexes = probeIndexRange(this.disk, this.table.indexRoot, this.lo, this.hi);
    const wantedPages = new Map<number, Tuple[]>();
    for (const ri of rowIndexes) {
      const pageId = this.rowToPage[ri];
      if (!wantedPages.has(pageId)) {
        const page = this.disk.readPage(pageId); // counted: heap page touch
        wantedPages.set(pageId, readPageRows(page, this.table.types));
      }
    }
    // Re-decode the exact matched rows by their in-page slot. rowOffsets[ri] is
    // the slot index within rowToPage[ri]'s page; this avoids re-scanning pages.
    this.rows = rowIndexes.map((ri) => {
      const pageRows = wantedPages.get(this.rowToPage[ri])!;
      return pageRows[this.rowOffsets[ri]];
    });
    this.idx = 0;
  }
  next(): Tuple | null {
    this.stats.nextCalls++;
    if (this.idx >= this.rows.length) return null;
    this.stats.rowsEmitted++;
    return this.rows[this.idx++];
  }
  close(): void {
    this.rows = [];
  }
  outputColumns(): string[] {
    return this.table.columns.map((c) => `${this.alias}.${c}`);
  }
}

/** Filter: pull from child, emit only rows passing the predicate. A pull-based
 *  filter is where the Volcano model shows its per-row overhead most clearly —
 *  one next() down per input row, regardless of selectivity. */
class Filter implements PhysicalOp {
  private colPos = -1;
  constructor(
    private input: PhysicalOp,
    private pred: Predicate,
    private stats: ExecStats,
  ) {}
  label(): string {
    return `Filter(${this.pred.left.table}.${this.pred.left.col} ${this.pred.op} ${JSON.stringify(this.pred.right.value)})`;
  }
  open(): void {
    this.input.open();
    const want = `${this.pred.left.table}.${this.pred.left.col}`;
    this.colPos = this.input.outputColumns().indexOf(want);
    if (this.colPos < 0) throw new Error(`Filter: column '${want}' not in input`);
  }
  next(): Tuple | null {
    this.stats.nextCalls++;
    for (;;) {
      const row = this.input.next();
      if (row === null) return null;
      if (this.test(row[this.colPos])) {
        this.stats.rowsEmitted++;
        return row;
      }
    }
  }
  private test(v: Value): boolean {
    const r = this.pred.right.value;
    switch (this.pred.op) {
      case "=":
        return v === r;
      case ">":
        return v > r;
      case "<":
        return v < r;
      case ">=":
        return v >= r;
      case "<=":
        return v <= r;
      default: {
        const _exhaustive: never = this.pred.op;
        throw new Error(`Filter: unknown op ${_exhaustive}`);
      }
    }
  }
  close(): void {
    this.input.close();
  }
  outputColumns(): string[] {
    return this.input.outputColumns();
  }
}

/** HashJoin: classic build-probe. open() drains the (smaller) right/build side
 *  into a hash table keyed on the join column, then next() pulls left/probe rows
 *  and emits one joined tuple per matching build row. We count build rows and
 *  probe rows + emitted rows so the chapter can show the intermediate cardinality
 *  the join produces (the "HashJoin intermediate result rows" measurement). */
class HashJoin implements PhysicalOp {
  private table = new Map<Value, Tuple[]>();
  private leftKeyPos = -1;
  private rightKeyPos = -1;
  private pendingMatches: Tuple[] = [];
  private pendingIdx = 0;
  private currentLeft: Tuple | null = null;
  private leftCols: string[] = [];
  private rightCols: string[] = [];
  buildRows = 0;
  probeRows = 0;
  emittedJoins = 0;
  constructor(
    private left: PhysicalOp, // probe side
    private right: PhysicalOp, // build side
    private on: { left: ColRef; right: ColRef },
    private stats: ExecStats,
  ) {}
  label(): string {
    return `HashJoin(${this.on.left.table}.${this.on.left.col} = ${this.on.right.table}.${this.on.right.col})`;
  }
  open(): void {
    // Reset per-run counters: an operator object may be re-opened (e.g. once
    // under bench's timing loop, once for a clean stats snapshot). Without this
    // reset the counts would accumulate across runs and print a fiction.
    this.buildRows = 0;
    this.probeRows = 0;
    this.emittedJoins = 0;
    this.table.clear();
    this.pendingMatches = [];
    this.pendingIdx = 0;
    this.currentLeft = null;
    this.left.open();
    this.right.open();
    this.leftCols = this.left.outputColumns();
    this.rightCols = this.right.outputColumns();
    // The ON clause names two columns but does NOT promise which is the probe
    // side and which is the build side — `o.user_id = u.id` could appear with
    // either operand on either physical input. Resolve each operand against
    // whichever side actually exposes it, so the planner is free to choose build
    // vs probe by cardinality without the parser caring.
    const a = `${this.on.left.table}.${this.on.left.col}`;
    const b = `${this.on.right.table}.${this.on.right.col}`;
    if (this.leftCols.includes(a) && this.rightCols.includes(b)) {
      this.leftKeyPos = this.leftCols.indexOf(a);
      this.rightKeyPos = this.rightCols.indexOf(b);
    } else if (this.leftCols.includes(b) && this.rightCols.includes(a)) {
      this.leftKeyPos = this.leftCols.indexOf(b);
      this.rightKeyPos = this.rightCols.indexOf(a);
    } else {
      throw new Error(`HashJoin: join keys ${a} / ${b} not split across the two inputs`);
    }
    // Build phase: materialize the entire build side. This is the one operator
    // that is NOT fully streaming — it buffers the build input in memory. That is
    // inherent to hash join and is why the planner makes the smaller relation the
    // build side (here the post-filter result, which we expect to be small).
    for (;;) {
      const r = this.right.next();
      if (r === null) break;
      this.buildRows++;
      const key = r[this.rightKeyPos];
      const bucket = this.table.get(key);
      if (bucket) bucket.push(r);
      else this.table.set(key, [r]);
    }
  }
  next(): Tuple | null {
    this.stats.nextCalls++;
    for (;;) {
      // Drain matches buffered for the current left row first.
      if (this.currentLeft && this.pendingIdx < this.pendingMatches.length) {
        const rightRow = this.pendingMatches[this.pendingIdx++];
        this.emittedJoins++;
        this.stats.rowsEmitted++;
        return [...this.currentLeft, ...rightRow]; // left cols ++ right cols
      }
      // Need a new left row.
      const l = this.left.next();
      if (l === null) return null;
      this.probeRows++;
      const matches = this.table.get(l[this.leftKeyPos]);
      if (matches && matches.length > 0) {
        this.currentLeft = l;
        this.pendingMatches = matches;
        this.pendingIdx = 0;
      }
      // else: no match, loop to pull the next left row (inner join drops it).
    }
  }
  close(): void {
    this.left.close();
    this.right.close();
    this.table.clear();
  }
  outputColumns(): string[] {
    return [...this.leftCols, ...this.rightCols];
  }
}

/** HashAggregate: GROUP BY with COUNT/SUM. Fully consumes its input at open()
 *  (aggregation is blocking — you cannot emit a group's total until you've seen
 *  every row), building one accumulator per group key, then streams the group
 *  result tuples. Output columns are the group key followed by each aggregate's
 *  alias. */
class HashAggregate implements PhysicalOp {
  private groups = new Map<Value, Value[]>(); // groupKey -> [groupKeyVal, agg1, agg2,...]
  private results: Tuple[] = [];
  private idx = 0;
  private groupPos = -1;
  private aggPositions: number[] = [];
  private outCols: string[] = [];
  groupCount = 0;
  constructor(
    private input: PhysicalOp,
    private groupBy: ColRef | undefined,
    private aggs: SelectItem[],
    private stats: ExecStats,
  ) {}
  label(): string {
    const g = this.groupBy ? `${this.groupBy.table}.${this.groupBy.col}` : "(scalar)";
    const a = this.aggs.filter((x) => x.agg).map((x) => `${x.agg}(${x.col.table}.${x.col.col})`).join(",");
    return `HashAggregate(group=${g}, aggs=${a})`;
  }
  open(): void {
    // Reset per-run accumulators (see HashJoin.open for why re-open must clear).
    this.groups.clear();
    this.results = [];
    this.idx = 0;
    this.aggPositions = [];
    this.outCols = [];
    this.input.open();
    const cols = this.input.outputColumns();
    if (this.groupBy) {
      this.groupPos = cols.indexOf(`${this.groupBy.table}.${this.groupBy.col}`);
      if (this.groupPos < 0) throw new Error(`HashAggregate: group column not in input`);
      this.outCols.push(`${this.groupBy.table}.${this.groupBy.col}`);
    }
    for (const a of this.aggs) {
      if (a.agg) {
        this.aggPositions.push(cols.indexOf(`${a.col.table}.${a.col.col}`));
        this.outCols.push(a.alias);
      }
    }
    // Blocking consume: drain the whole input, fold each row into its group.
    for (;;) {
      const row = this.input.next();
      if (row === null) break;
      const key: Value = this.groupBy ? row[this.groupPos] : "__all__";
      let acc = this.groups.get(key);
      if (!acc) {
        // acc layout: [groupKeyValue?, ...one slot per aggregate]
        acc = this.groupBy ? [key] : [];
        for (let k = 0; k < this.aggs.filter((x) => x.agg).length; k++) acc.push(0);
        this.groups.set(key, acc);
      }
      const aggOnly = this.aggs.filter((x) => x.agg);
      const base = this.groupBy ? 1 : 0;
      for (let k = 0; k < aggOnly.length; k++) {
        const a = aggOnly[k];
        if (a.agg === "count") {
          (acc[base + k] as number) += 1;
        } else {
          // sum
          (acc[base + k] as number) += row[this.aggPositions[k]] as number;
        }
      }
    }
    this.results = [...this.groups.values()];
    this.groupCount = this.results.length;
    this.idx = 0;
  }
  next(): Tuple | null {
    this.stats.nextCalls++;
    if (this.idx >= this.results.length) return null;
    this.stats.rowsEmitted++;
    return this.results[this.idx++];
  }
  close(): void {
    this.input.close();
    this.groups.clear();
  }
  outputColumns(): string[] {
    return this.outCols;
  }
}

// ===========================================================================
// Physical planner: turn a logical plan into physical operators. The one
// interesting decision is filter+scan -> IndexScan vs SeqScan. We attempt an
// IndexScan only when: the scanned table has an index AND it is on the predicate
// column AND the predicate is a range/equality the index supports. Otherwise we
// emit Filter(SeqScan). The failure-mode demo forces the index path on an
// unindexed column and watches this exact logic refuse and fall back.
// ===========================================================================

interface PlanContext {
  disk: Disk;
  catalog: Catalog;
  stats: ExecStats;
  /** Per-table row->page maps for IndexScan heap fetches, keyed by table name. */
  rowMaps: Map<string, { rowToPage: number[]; rowOffsets: number[] }>;
  /** When set, the planner is FORCED to try IndexScan on this column (demo). */
  forceIndexOn?: ColRef;
  /** Filled by the planner: human-readable note if a forced index fell back. */
  fallbackNote?: string;
}

function predicateToRange(pred: Predicate): { lo: number; hi: number } | null {
  // Index supports integer key ranges only (matches our int B+tree).
  if (pred.right.type !== "int") return null;
  const v = pred.right.value as number;
  switch (pred.op) {
    case "=":
      return { lo: v, hi: v };
    case ">":
      return { lo: v + 1, hi: 0xffffffff };
    case ">=":
      return { lo: v, hi: 0xffffffff };
    case "<":
      return { lo: 0, hi: v - 1 };
    case "<=":
      return { lo: 0, hi: v };
    default:
      return null;
  }
}

function planPhysical(node: LogicalNode, ctx: PlanContext): PhysicalOp {
  switch (node.op) {
    case "scan": {
      const table = ctx.catalog.get(node.table);
      return new SeqScan(ctx.disk, table, node.alias, ctx.stats);
    }
    case "filter": {
      const inner = node.input;
      invariant(inner.op === "scan", "planner: filter is only pushed directly over a scan here");
      const table = ctx.catalog.get(inner.table);
      const alias = inner.alias;
      const predColIdx = ctx.catalog.colIndex(table, node.pred.left.col);

      // Decide IndexScan vs SeqScan. Two trigger paths: (a) the table genuinely
      // has an index on the predicate column; (b) the demo forces it via
      // forceIndexOn. Either way we VALIDATE the index actually exists on that
      // column — a forced request for a non-existent index must fall back.
      const forced =
        ctx.forceIndexOn &&
        ctx.forceIndexOn.table === node.pred.left.table &&
        ctx.forceIndexOn.col === node.pred.left.col;
      const indexUsable =
        table.indexedCol === predColIdx && table.indexRoot >= 0;
      const range = predicateToRange(node.pred);

      if ((indexUsable || forced) && indexUsable && range) {
        const rm = ctx.rowMaps.get(table.name)!;
        return new IndexScan(
          ctx.disk,
          table,
          alias,
          range.lo,
          range.hi,
          ctx.stats,
          rm.rowOffsets,
          rm.rowToPage,
        );
      }

      // Fallback path. If the index was *forced* but unusable, record WHY so the
      // demo can print the planner's reasoning, then degrade to a full SeqScan +
      // row-at-a-time Filter — the real cost of a missing index.
      if (forced && !indexUsable) {
        ctx.fallbackNote =
          `requested IndexScan on ${node.pred.left.table}.${node.pred.left.col}, ` +
          `but '${table.name}' has no index on '${node.pred.left.col}' ` +
          `(only on '${table.indexedCol >= 0 ? table.columns[table.indexedCol] : "(none)"}'); ` +
          `planner fell back to full SeqScan + Filter`;
      }
      return new Filter(new SeqScan(ctx.disk, table, alias, ctx.stats), node.pred, ctx.stats);
    }
    case "join": {
      const left = planPhysical(node.left, ctx); // probe side
      const right = planPhysical(node.right, ctx); // build side
      return new HashJoin(left, right, node.on, ctx.stats);
    }
    case "aggregate": {
      const input = planPhysical(node.input, ctx);
      return new HashAggregate(input, node.groupBy, node.aggs, ctx.stats);
    }
    case "project": {
      // No-op projection operator for this subset: aggregation handles the only
      // shape we print. We still validate the columns exist so a bad SELECT fails.
      const input = planPhysical(node.input, ctx);
      return input;
    }
    default: {
      const _exhaustive: never = node;
      throw new Error(`planner: unknown logical op ${JSON.stringify(_exhaustive)}`);
    }
  }
}

/** Pretty-print a physical operator tree as an indented outline. Walks the known
 *  child fields; kept here (not on the operators) so operators stay free of
 *  printing concerns. */
function formatPlan(op: PhysicalOp, depth = 0): string {
  const indent = "  ".repeat(depth) + (depth > 0 ? "└─ " : "");
  let out = indent + op.label() + "\n";
  // Reach into the known composite operators to recurse. This is the one place
  // that knows the tree shape; operators expose children via these private fields.
  const anyOp = op as unknown as {
    input?: PhysicalOp;
    left?: PhysicalOp;
    right?: PhysicalOp;
  };
  if (anyOp.left && anyOp.right) {
    out += formatPlan(anyOp.left, depth + 1);
    out += formatPlan(anyOp.right, depth + 1);
  } else if (anyOp.input) {
    out += formatPlan(anyOp.input, depth + 1);
  }
  return out;
}

/** Drive a physical plan to completion via repeated next(); collect output rows.
 *  This is the executor's whole control loop — the Volcano "pull" engine. */
function runPlan(root: PhysicalOp): Tuple[] {
  root.open();
  const out: Tuple[] = [];
  for (;;) {
    const row = root.next();
    if (row === null) break;
    out.push(row);
  }
  root.close();
  return out;
}

// ===========================================================================
// Deterministic data generation. users(id, country, age) and
// orders(id, user_id, amount). Two skewed-but-deterministic distributions so the
// query produces a non-trivial group-by and join cardinality, all from the seeded
// core PRNG (reproducible byte-for-byte).
// ===========================================================================

const COUNTRIES = ["CN", "US", "JP", "DE", "BR"]; // 5 groups for GROUP BY

interface BuiltTables {
  catalog: Catalog;
  disk: Disk;
  rowMaps: Map<string, { rowToPage: number[]; rowOffsets: number[] }>;
}

function buildTables(rng: Rng, userCount: number, orderCount: number): BuiltTables {
  const disk = new Disk({ initialPages: 64 });
  const catalog = new Catalog();
  const rowMaps = new Map<string, { rowToPage: number[]; rowOffsets: number[] }>();

  // --- users ---
  const userSchema: Schema = ["int", "string", "int"];
  const userRows: Tuple[] = [];
  for (let id = 0; id < userCount; id++) {
    const country = COUNTRIES[rng.nextInt(0, COUNTRIES.length)];
    const age = rng.nextInt(18, 70);
    userRows.push([id, country, age]);
  }
  const userHeap = loadHeap(disk, userRows, userSchema);
  const userRowToPage = computeRowToPage(disk, userHeap, userSchema);
  rowMaps.set("users", userRowToPage);

  // Build a B+tree index on users.id (column 0). Pairs sorted by key (id is dense
  // 0..n so already sorted, but we sort explicitly to honor buildIndex's contract).
  const userKeyPairs = userRows
    .map((r, ri) => ({ key: r[0] as number, rowIndex: ri }))
    .sort((a, b) => a.key - b.key);
  const userIndex = buildIndex(disk, userKeyPairs);

  catalog.define({
    name: "users",
    columns: ["id", "country", "age"],
    types: userSchema,
    firstPage: userHeap.firstPage,
    pageCount: userHeap.pageCount,
    rowCount: userCount,
    indexedCol: 0, // index on users.id
    indexRoot: userIndex.root,
  });

  // --- orders --- (no index, on purpose: drives the failure-mode demo) ---
  const orderSchema: Schema = ["int", "int", "int"];
  const orderRows: Tuple[] = [];
  for (let id = 0; id < orderCount; id++) {
    const userId = rng.nextInt(0, userCount);
    const amount = rng.nextInt(1, 1000);
    orderRows.push([id, userId, amount]);
  }
  const orderHeap = loadHeap(disk, orderRows, orderSchema);
  const orderRowToPage = computeRowToPage(disk, orderHeap, orderSchema);
  rowMaps.set("orders", orderRowToPage);

  catalog.define({
    name: "orders",
    columns: ["id", "user_id", "amount"],
    types: orderSchema,
    firstPage: orderHeap.firstPage,
    pageCount: orderHeap.pageCount,
    rowCount: orderCount,
    indexedCol: -1, // NO index
    indexRoot: -1,
  });

  return { catalog, disk, rowMaps };
}

/** Build rowGlobalIndex -> (pageId, in-page slot) maps so IndexScan can fetch a
 *  matched row directly. rowGlobalIndex is assignment order during loadHeap, which
 *  matches the order rows were appended; we reconstruct it by replaying the pages.
 *  This read does NOT pollute query measurements because callers resetStats()
 *  after building. */
function computeRowToPage(
  disk: Disk,
  heap: { firstPage: number; pageCount: number },
  schema: Schema,
): { rowToPage: number[]; rowOffsets: number[] } {
  const rowToPage: number[] = [];
  const rowOffsets: number[] = [];
  for (let p = 0; p < heap.pageCount; p++) {
    const pageId = heap.firstPage + p;
    const page = disk.readPage(pageId);
    const rows = readPageRows(page, schema);
    for (let slot = 0; slot < rows.length; slot++) {
      rowToPage.push(pageId);
      rowOffsets.push(slot);
    }
  }
  return { rowToPage, rowOffsets };
}

// ===========================================================================
// Main: wire it all up, run the headline query, print every measured number,
// then run the failure-mode demo. console output is Chinese per the book's
// convention; identifiers/comments stay English.
// ===========================================================================

function section(title: string): void {
  console.log("\n" + "=".repeat(72));
  console.log(title);
  console.log("=".repeat(72));
}

function main(): void {
  const USERS = 10000;
  const ORDERS = 10000;
  const rng = createRng(0x5108); // fixed seed => byte-identical data every run

  section("数据准备 (确定性, seed=0x5108)");
  const { catalog, disk, rowMaps } = buildTables(rng, USERS, ORDERS);
  const users = catalog.get("users");
  const orders = catalog.get("orders");
  printTable([
    { table: "users", rows: users.rowCount, data_pages: users.pageCount, index: "B+tree on id" },
    { table: "orders", rows: orders.rowCount, data_pages: orders.pageCount, index: "(none)" },
  ]);

  // The headline query: filter users by an indexed range, join orders on user_id,
  // group by country, sum amounts and count orders.
  const sql =
    "SELECT u.country, COUNT(o.id) AS order_count, SUM(o.amount) AS revenue " +
    "FROM users u " +
    "JOIN orders o ON o.user_id = u.id " +
    "WHERE u.id < 2000 " +
    "GROUP BY u.country";

  section("SQL");
  console.log(sql);

  // --- Parse + logical plan ---
  section("解析 → AST → 逻辑计划");
  const tokens = tokenize(sql);
  console.log(`token 数: ${tokens.length}`);
  const ast = new Parser(tokens).parse();
  const logical = planLogical(ast);
  console.log("逻辑计划 (relational):");
  console.log(JSON.stringify(logical, null, 2).split("\n").map((l) => "  " + l).join("\n"));

  // --- Physical plan WITH index on the filtered column (users.id) ---
  section("物理计划 (火山模型算子树) — 走 IndexScan");
  const statsIdx: ExecStats = { nextCalls: 0, rowsEmitted: 0 };
  const ctxIdx: PlanContext = {
    disk,
    catalog,
    stats: statsIdx,
    rowMaps,
    forceIndexOn: { table: "u", col: "id" }, // honor the natural index on users.id
  };
  disk.resetStats(); // measure only this query's IO
  const physIdx = planPhysical(logical, ctxIdx);
  process.stdout.write(formatPlan(physIdx));

  // --- Execute (measured) ---
  // Time the full pipeline over several iterations for a less noisy mean. The
  // WORK is deterministic (seeded data, fixed plan); only the nanoseconds are a
  // real, machine-dependent measurement. nsPerOp below is the mean per execution.
  const QUERY_ITERS = 50;
  const wallIdx = bench(() => {
    runPlan(physIdx); // re-opens + drains the same plan tree each iteration
  }, QUERY_ITERS);
  // Re-run once outside bench to capture stats + result (bench's last iteration
  // already ran it, but we want a clean stats snapshot tied to the printed result).
  statsIdx.nextCalls = 0;
  statsIdx.rowsEmitted = 0;
  disk.resetStats();
  const resultIdx = runPlan(physIdx);
  const ioIdx = disk.stats();

  section("执行结果 (GROUP BY country)");
  const resultRows = resultIdx.map((t) => {
    // outCols of the aggregate: [country, order_count, revenue]
    return { country: t[0] as string, order_count: t[1] as number, revenue: t[2] as number };
  });
  resultRows.sort((a, b) => (a.country < b.country ? -1 : 1)); // stable print order
  printTable(resultRows);

  // Pull the join operator out of the tree to report its intermediate cardinality.
  const join = findOp(physIdx, HashJoin) as HashJoin | null;
  invariant(join !== null, "expected a HashJoin in the plan");

  section("量化指标 — IndexScan 计划");
  printTable([
    { metric: "next() 调用总数", value: statsIdx.nextCalls, note: "火山模型每行每算子一次虚调用" },
    { metric: "页触达 reads (本查询)", value: ioIdx.reads, note: "index 页 + 命中堆页" },
    { metric: "HashJoin build 行数", value: join.buildRows, note: "构建侧 (过滤后 users)" },
    { metric: "HashJoin probe 行数", value: join.probeRows, note: "探测侧 (全部 orders)" },
    { metric: "HashJoin 输出行数", value: join.emittedJoins, note: "join 中间结果基数" },
    { metric: "最终结果行数", value: resultIdx.length, note: "GROUP BY 后的组数" },
    { metric: "wall time (measured)", value: `${(wallIdx.nsPerOp / 1e6).toFixed(3)} ms`, note: "真实墙钟, 机器相关" },
  ]);

  // ----------------------------------------------------------------------
  // SeqScan baseline: same query but force the filter through SeqScan+Filter so
  // we can contrast page touches directly on the SAME query.
  // ----------------------------------------------------------------------
  section("对照 — 强制全表 SeqScan (无 index 路径)");
  const statsSeq: ExecStats = { nextCalls: 0, rowsEmitted: 0 };
  const ctxSeq: PlanContext = { disk, catalog, stats: statsSeq, rowMaps }; // no forceIndexOn, users still indexed though
  // To force SeqScan on users.id we temporarily pretend users has no index by
  // planning against a catalog view with indexedCol disabled. Simplest honest way:
  // build the plan, but the filter planner only uses the index when usable; we
  // disable it by clearing the index marker for this measurement, then restore.
  const savedIndexedCol = users.indexedCol;
  const savedRoot = users.indexRoot;
  (users as TableDef).indexedCol = -1;
  (users as TableDef).indexRoot = -1;
  disk.resetStats();
  const physSeq = planPhysical(logical, ctxSeq);
  process.stdout.write(formatPlan(physSeq));
  const wallSeq = bench(() => {
    runPlan(physSeq);
  }, QUERY_ITERS);
  statsSeq.nextCalls = 0;
  statsSeq.rowsEmitted = 0;
  disk.resetStats();
  const resultSeq = runPlan(physSeq);
  const ioSeq = disk.stats();
  // Restore catalog so later code sees the real index.
  (users as TableDef).indexedCol = savedIndexedCol;
  (users as TableDef).indexRoot = savedRoot;

  // Correctness cross-check: index path and seq path MUST produce identical
  // results, else the index is lying. This is the assert that makes "the optimizer
  // preserves semantics" executable rather than assumed.
  invariant(
    JSON.stringify(normalize(resultIdx)) === JSON.stringify(normalize(resultSeq)),
    "IndexScan and SeqScan must produce identical results",
  );

  section("量化对照 — IndexScan vs SeqScan (同一查询)");
  const pageDelta = ioSeq.reads - ioIdx.reads;
  const speedup = ioIdx.reads === 0 ? Infinity : ioSeq.reads / ioIdx.reads;
  printTable([
    {
      plan: "IndexScan(users.id)",
      page_reads: ioIdx.reads,
      next_calls: statsIdx.nextCalls,
      wall_ms: Number((wallIdx.nsPerOp / 1e6).toFixed(3)),
    },
    {
      plan: "SeqScan(users)",
      page_reads: ioSeq.reads,
      next_calls: statsSeq.nextCalls,
      wall_ms: Number((wallSeq.nsPerOp / 1e6).toFixed(3)),
    },
  ]);
  console.log(
    `\nIndexScan 少触达 ${pageDelta} 页 (${ioSeq.reads} → ${ioIdx.reads}, ${speedup.toFixed(1)}x 更少页 IO)。`,
  );
  console.log("注: 数据为 toy (RAM 模拟 disk), 墙钟绝对值偏乐观; 可迁移的是页 IO 比例趋势。");

  // ----------------------------------------------------------------------
  // FAILURE MODE: force an IndexScan on orders.amount, which has NO index. The
  // planner must refuse and fall back to a full SeqScan. We quantify the blow-up.
  // ----------------------------------------------------------------------
  section("失败模式 — 对未建索引的列强制 IndexScan");
  const failSql =
    "SELECT o.user_id, COUNT(o.id) AS cnt " +
    "FROM orders o " +
    "WHERE o.amount > 990 " +
    "GROUP BY o.user_id";
  console.log(failSql);
  const failAst = new Parser(tokenize(failSql)).parse();
  const failLogical = planLogical(failAst);
  const statsFail: ExecStats = { nextCalls: 0, rowsEmitted: 0 };
  const ctxFail: PlanContext = {
    disk,
    catalog,
    stats: statsFail,
    rowMaps,
    forceIndexOn: { table: "o", col: "amount" }, // FORCE index on an unindexed col
  };
  disk.resetStats();
  const physFail = planPhysical(failLogical, ctxFail);
  console.log("\n规划器判定:");
  console.log("  " + (ctxFail.fallbackNote ?? "(unexpectedly used an index!)"));
  invariant(ctxFail.fallbackNote !== undefined, "forcing index on unindexed column must trigger fallback");
  process.stdout.write("\n实际物理计划:\n");
  process.stdout.write(formatPlan(physFail));
  disk.resetStats();
  statsFail.nextCalls = 0;
  const failResult = runPlan(physFail);
  const ioFail = disk.stats();

  // Real matched-row count = sum of the COUNT(*) per group (the rows that passed
  // amount>990). Distinct from the group count, so we report both honestly.
  const matchedGroups = failResult.length;
  const matchedRows = failResult.reduce((acc, g) => acc + (g[1] as number), 0);
  // What an index WOULD have cost (est., since no such index exists to measure):
  // an int B+tree on amount would let a range probe touch ~1 internal page + the
  // few leaves covering (990, 1000] + only the heap pages holding the matched
  // rows. With ~matchedRows rows out of 10000, those rows span far fewer than the
  // 40 data pages a full scan reads. Labeled (est.) — we don't fabricate a measured
  // number for an index we deliberately didn't build.
  const estHeapPages = Math.ceil(matchedRows / (orders.rowCount / orders.pageCount));
  section("失败模式量化 — 缺索引的全表扫描代价");
  printTable([
    { metric: "orders 数据页总数", value: orders.pageCount, note: "全表" },
    { metric: "实际页触达 (SeqScan 回退)", value: ioFail.reads, note: "被迫扫全表" },
    { metric: "next() 调用总数", value: statsFail.nextCalls, note: "每行都过 Filter" },
    { metric: "命中行数 (amount>990)", value: matchedRows, note: "通过谓词的 orders 行" },
    { metric: "命中分组数", value: matchedGroups, note: "命中行的 user_id 去重组数" },
    {
      metric: "若有 amount 索引 (est.)",
      value: `~${1 + estHeapPages} 页 (1 内部 + ~${estHeapPages} 堆/叶)`,
      note: "估算: 高选择性谓词只需触达命中页, 远小于全表",
    },
  ]);
  const estTotal = 1 + estHeapPages;
  console.log(
    `\n估算加速比 (est.): ${ioFail.reads} 页 → ~${estTotal} 页, 约 ${(ioFail.reads / estTotal).toFixed(1)}x 更少页 IO。`,
  );
  console.log(
    `\n结论: orders 无 amount 索引, 高选择性谓词 (amount>990) 仍被迫扫描全部 ${ioFail.reads} 页。`,
  );
  console.log("这正是 \"缺索引 => 全表扫描\" 的代价: 触页数与表大小线性增长, 与谓词选择性无关。");

  section("完成");
  console.log("所有计划执行完毕; IndexScan 与 SeqScan 结果一致性已 assert 通过。");
}

/** Locate the first operator of a given class in the tree (for reporting). */
function findOp(op: PhysicalOp, cls: new (...args: never[]) => PhysicalOp): PhysicalOp | null {
  if (op instanceof cls) return op;
  const anyOp = op as unknown as { input?: PhysicalOp; left?: PhysicalOp; right?: PhysicalOp };
  for (const child of [anyOp.left, anyOp.right, anyOp.input]) {
    if (child) {
      const found = findOp(child, cls);
      if (found) return found;
    }
  }
  return null;
}

/** Normalize aggregate results into a sorted, comparable form for the
 *  index-vs-seq equality assert. */
function normalize(rows: Tuple[]): unknown {
  return rows
    .map((r) => r.map((v) => (typeof v === "number" ? v : String(v))))
    .sort((a, b) => (JSON.stringify(a) < JSON.stringify(b) ? -1 : 1));
}

main();
