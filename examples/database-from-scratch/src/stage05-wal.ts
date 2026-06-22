// stage05-wal.ts — Write-Ahead Logging and atomicity: why a committed txn
// survives power loss even though its data pages may never have hit disk.
//
// The one idea this chapter makes physical: the WAL protocol. Before a modified
// data page is allowed to become durable, the log record describing that change
// must already be durable. Commit forces the log (fsync). Given that ordering,
// recovery can always reconstruct the committed state by replaying the log —
// even if EVERY data page write was lost. So durability is bought with log
// fsyncs, not data-page fsyncs, which is the whole economic argument for WAL:
// one sequential append + one fsync per commit instead of scattered random
// page flushes.
//
// What is modeled here vs. deferred:
//  - We model TWO durable devices: a `logDisk` (the WAL, append-only) and a
//    `dataDisk` (the heap pages). They have SEPARATE IO counters so "fsyncs
//    spent on the log" is a clean, honest number, not entangled with data IO.
//  - We model physical logging with BEFORE and AFTER images per record. After
//    images give recovery REDO (re-apply a committed change lost from the data
//    file); before images give UNDO (roll back an uncommitted change that leaked
//    to the data file). Stage07 (recovery) consumes exactly these.
//  - We DELIBERATELY do not recover here. This stage's failure demo stops at the
//    inconsistent intermediate state and hands the replay to chapter 07. Doing
//    recovery here would blur the chapter boundary the book is built on.
//
// Honesty about the numbers:
//  - Byte counts and fsync/write COUNTS are deterministic (seeded PRNG workload)
//    and are exactly what the code did — reproducible run to run.
//  - Throughput (ops/sec) is REAL wall-clock from core/bench, hence machine-
//    dependent; it is labeled "measured". The absolute value is optimistic
//    because our "disk" is RAM (no real seek / fsync latency). What transfers to
//    a real engine is the RELATIVE story: group commit amortizes fsyncs, and
//    fewer fsyncs is the dominant lever — the fsync-count ratio is the honest
//    takeaway, the wall-clock speedup is only indicative.

import { Disk, CrashError, PAGE_SIZE } from "./core/disk.js";
import { PageReader, PageWriter, PageType, SLOTTED_HEADER } from "./core/page.js";
import { encodeRow } from "./core/codec.js";
import { createRng, type Rng } from "./core/prng.js";
import { printTable, invariant, assertEq } from "./core/assert.js";

// ---------------------------------------------------------------------------
// WAL record format
// ---------------------------------------------------------------------------
//
// A log record is a self-contained, byte-addressable description of a single
// page mutation. We keep records fixed-shape (one page id, one before image, one
// after image) because the point of the chapter is the PROTOCOL, not a clever
// record encoding — variable record packing is an stage exercise, not load-
// bearing here. Each record carries enough to both redo and undo.
//
// Layout (big-endian, mirroring core/page.ts's endianness discipline):
//   u32  lsn           strictly increasing log sequence number
//   u8   kind          RecordKind (update / commit / abort / checkpoint)
//   u32  txnId         owning transaction
//   u32  pageId        data page this record describes (0 for non-update kinds)
//   u16  imageLen      length L of each image (before and after are both L bytes)
//   [L]  beforeImage   bytes of the page region prior to the change (UNDO source)
//   [L]  afterImage    bytes of the page region after the change   (REDO source)
//
// We log a fixed-size REGION of the page (the slot we touched), not the whole
// 4096-byte page. Logging full pages would be correct but would make "avg log
// bytes per txn" a meaningless constant; logging the changed region is what real
// engines do (physiological / region logging) and keeps the byte number honest.

const enum RecordKind {
  Update = 1,
  Commit = 2,
  Abort = 3,
  Checkpoint = 4,
}

interface LogRecord {
  lsn: number;
  kind: RecordKind;
  txnId: number;
  pageId: number;
  beforeImage: Uint8Array;
  afterImage: Uint8Array;
}

// Fixed header size before the variable image bytes:
//   4 (lsn) + 1 (kind) + 4 (txnId) + 4 (pageId) + 2 (imageLen) = 15 bytes.
const REC_HEADER_BYTES = 15;

/** Serialize a record to bytes. Returns a fresh buffer; the WAL appends it.
 *  Why a flat buffer instead of writing field-by-field into the log device: it
 *  makes "the record is N bytes" trivially true (buffer.length) and lets the WAL
 *  treat the log as an opaque byte stream — the recovery chapter re-parses it. */
function encodeRecord(rec: LogRecord): Uint8Array {
  invariant(
    rec.beforeImage.length === rec.afterImage.length,
    "before/after images must be the same length (same page region)",
  );
  const imageLen = rec.beforeImage.length;
  const buf = new Uint8Array(REC_HEADER_BYTES + 2 * imageLen);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, rec.lsn, false);
  dv.setUint8(4, rec.kind);
  dv.setUint32(5, rec.txnId, false);
  dv.setUint32(9, rec.pageId, false);
  dv.setUint16(13, imageLen, false);
  buf.set(rec.beforeImage, REC_HEADER_BYTES);
  buf.set(rec.afterImage, REC_HEADER_BYTES + imageLen);
  return buf;
}

// ---------------------------------------------------------------------------
// The WAL device: an append-only log on its own Disk.
// ---------------------------------------------------------------------------
//
// We page-pack the byte stream onto a Disk so its fsync()/writePage() counters
// are real and comparable to the data disk. Records are appended into a current
// page; when a record would overflow the page we seal the page (write it) and
// start a fresh one. A flush() writes the partial current page and fsyncs.
//
// Invariant (the WAL protocol, enforced by construction): callers MUST append +
// flush the log record for a page change BEFORE the dirty data page is written
// durably. This class can't see the data disk, so the DISCIPLINE lives in the
// Database methods below; this class only guarantees "flush() makes every byte
// appended so far durable".

class WriteAheadLog {
  private readonly disk: Disk;
  private nextLsn = 1; // LSN 0 reserved as "no log record" sentinel
  // Current (unsealed) log page being filled, and the write cursor within it.
  private curPage: Uint8Array;
  private curPageId: number;
  private cursor = 0; // bytes used in curPage
  private totalRecordBytes = 0; // cumulative, for the "avg bytes/txn" metric

  /** @param crashAtFsync optional 1-based fsync ordinal at which the underlying
   *  log device throws CrashError (power-loss injection). Wired at construction
   *  because Disk's crashAt is immutable post-construct — the honest model of "a
   *  device's crash point is a property of the run, not toggled mid-flight". */
  constructor(crashAtFsync?: number) {
    this.disk = new Disk({ initialPages: 64, crashAt: crashAtFsync });
    this.curPageId = this.disk.allocPage();
    this.curPage = new Uint8Array(PAGE_SIZE);
  }

  /** Append one record to the log buffer (NOT yet durable). Returns its LSN.
   *  Appending is cheap and unsynced — durability happens only at flush(). This
   *  separation is the whole reason group commit is possible. */
  append(partial: Omit<LogRecord, "lsn">): number {
    const lsn = this.nextLsn++;
    const bytes = encodeRecord({ ...partial, lsn });
    invariant(
      bytes.length <= PAGE_SIZE,
      `single log record ${bytes.length}B exceeds page; record spanning is a stage exercise`,
    );
    // Seal the current page and roll to a new one if this record won't fit.
    // Real WALs let records span pages; we keep records page-bounded so the demo
    // stays readable — the protocol (append-before-data, flush-on-commit) is
    // identical either way.
    if (this.cursor + bytes.length > PAGE_SIZE) {
      this.sealCurrentPage();
    }
    this.curPage.set(bytes, this.cursor);
    this.cursor += bytes.length;
    this.totalRecordBytes += bytes.length;
    return lsn;
  }

  /** Force the log to durable storage: write the partial current page and fsync.
   *  After this returns, every appended record is recoverable. Throws CrashError
   *  (propagated from disk.fsync) if a crash was injected on this fsync — the
   *  caller must treat the in-flight commit as NOT durable. */
  flush(): void {
    // Always rewrite the current page: it may have grown since the last flush.
    // A real WAL would track the durable offset; rewriting the tail page is the
    // simple, correct equivalent for a single unsealed page.
    this.disk.writePage(this.curPageId, this.curPage);
    this.disk.fsync(); // the durability boundary; may throw CrashError
  }

  private sealCurrentPage(): void {
    // Persist the now-full page (no fsync here; fsync is a flush/commit concern),
    // then start a fresh append page. Sealing without fsync is intentional: an
    // unflushed sealed page is still volatile until the next flush(), exactly
    // like a real OS page cache page.
    this.disk.writePage(this.curPageId, this.curPage);
    this.curPageId = this.disk.allocPage();
    this.curPage = new Uint8Array(PAGE_SIZE);
    this.cursor = 0;
  }

  get totalLogBytes(): number {
    return this.totalRecordBytes;
  }

  stats() {
    return this.disk.stats();
  }
}

// ---------------------------------------------------------------------------
// A tiny transactional store: one slotted heap page worth of fixed rows.
// ---------------------------------------------------------------------------
//
// To exercise WAL we need real page mutations. We keep the data model minimal: a
// single slotted data page holding fixed-width rows (an int key + a small int
// value), addressed by slot id. Each "transaction" overwrites one row's value.
// That is enough to produce genuine before/after images and a genuine WAL — the
// chapter is about the log, not about a rich data model (B+tree is stage03).

const ROW_SCHEMA = ["int", "int"] as const; // (key, value), both u32 => 8 bytes
const ROW_BYTES = 8;
const ROW_COUNT = 100; // rows packed into the single data page

class Database {
  private readonly dataDisk: Disk;
  private readonly dataPageId: number;
  private readonly wal: WriteAheadLog;
  // Commits whose log records are appended but whose durability-forcing fsync is
  // deferred by group commit. Drained to durableCommits on each flush.
  private commitsSinceFlush = 0;
  // Count of commits whose durability has been confirmed by a successful flush.
  private durableCommits = 0;

  /** @param walCrashAtFsync optional 1-based WAL fsync ordinal to crash on. The
   *  failure mode lives entirely in the WAL device: a crash while forcing the log
   *  is the moment that decides whether a commit counts as durable. Undefined =
   *  no crash. */
  constructor(walCrashAtFsync?: number) {
    this.dataDisk = new Disk({ initialPages: 16 });
    this.dataPageId = this.dataDisk.allocPage();
    this.wal = new WriteAheadLog(walCrashAtFsync);
    this.initDataPage();
  }

  private initDataPage(): void {
    // Lay out ROW_COUNT fixed rows densely from the page end downward, mirroring
    // a slotted page's heap. We pre-place every row so a transaction is a pure
    // overwrite (stable offsets) — that keeps before/after images aligned and the
    // WAL record region constant-width.
    const page = this.dataDisk.readPage(this.dataPageId);
    const w = new PageWriter(page);
    w.initSlotted(PageType.LEAF);
    for (let slot = 0; slot < ROW_COUNT; slot++) {
      const off = this.rowOffset(slot);
      const row = encodeRow([slot, 0], ROW_SCHEMA as unknown as ("int")[]);
      w.writeBytes(off, row);
    }
    this.dataDisk.writePage(this.dataPageId, page);
    this.dataDisk.fsync(); // initial durable state; not counted in the txn phase
    this.dataDisk.resetStats();
  }

  /** Byte offset of a row's region within the data page. Rows are packed just
   *  below the slotted header; fixed width means offset is pure arithmetic. */
  private rowOffset(slot: number): number {
    return SLOTTED_HEADER.HEADER_SIZE + slot * ROW_BYTES;
  }

  /** Run one transaction that sets row[slot].value = newValue.
   *  WAL PROTOCOL enforced here:
   *    1. read current page region (before image)
   *    2. compute new region (after image)
   *    3. APPEND update record to WAL (before+after) — log first, in memory
   *    4. apply the change to the in-memory data page
   *    5. APPEND commit record; flush WAL (group-commit may defer the flush)
   *    6. only now may the dirty data page be written durably
   *  Returns the LSN of the commit record. */
  runTxn(txnId: number, slot: number, newValue: number): number {
    const off = this.rowOffset(slot);
    const page = this.dataDisk.readPage(this.dataPageId);

    // before image: the exact bytes we are about to overwrite (UNDO source).
    const beforeImage = page.slice(off, off + ROW_BYTES);
    // after image: same region with the new row encoded (REDO source).
    const newRow = encodeRow([slot, newValue], ROW_SCHEMA as unknown as ("int")[]);
    const afterImage = newRow;

    // Step 3: log the change BEFORE touching the durable data page. This is the
    // "write-ahead" in WAL — if we crash now, recovery sees the update record and
    // can redo it; if we crash after applying but before this, we'd have a data
    // change with no log, which is exactly what the ordering forbids.
    this.wal.append({
      kind: RecordKind.Update,
      txnId,
      pageId: this.dataPageId,
      beforeImage,
      afterImage,
    });

    // Step 4: apply in memory.
    page.set(afterImage, off);
    this.dataDisk.writePage(this.dataPageId, page); // dirty page, NOT yet fsynced

    // Step 5: commit record + (possibly deferred) flush.
    const commitLsn = this.wal.append({
      kind: RecordKind.Commit,
      txnId,
      pageId: 0,
      beforeImage: EMPTY,
      afterImage: EMPTY,
    });
    this.commitsSinceFlush++;
    return commitLsn;
  }

  /** Force the WAL if the group-commit batch is full (or always, when batch=1).
   *  Group commit: several transactions' commit records ride a SINGLE fsync, so
   *  fsync count drops ~batch-fold. The trade-off is latency — a committing txn
   *  waits for its batch to fill (here we fill synchronously). On success, all
   *  batched commits become durable together. */
  maybeFlush(batch: number): void {
    if (this.commitsSinceFlush >= batch) {
      this.flushCommits();
    }
  }

  /** Force any buffered commits to durable storage in one fsync. */
  flushCommits(): void {
    if (this.commitsSinceFlush === 0) return;
    this.wal.flush(); // single fsync covers the whole batch
    this.durableCommits += this.commitsSinceFlush;
    this.commitsSinceFlush = 0;
  }

  /** Write a checkpoint record and force it. A checkpoint bounds how far back
   *  recovery must scan: everything before a flushed checkpoint whose dirty pages
   *  are known-durable need not be replayed. We model the record + the data-page
   *  flush that makes the checkpoint meaningful; the recovery-side scan-bound is
   *  stage07's job. */
  checkpoint(): void {
    this.wal.append({
      kind: RecordKind.Checkpoint,
      txnId: 0,
      pageId: 0,
      beforeImage: EMPTY,
      afterImage: EMPTY,
    });
    this.wal.flush(); // checkpoint must be durable to be a valid scan-bound...
    // ...and only after the checkpoint is logged do we flush the data pages it
    // claims are stable. Order matters: log-then-data, same as every other rule.
    this.dataDisk.fsync();
  }

  /** Read a row's value straight from the in-memory data page (no IO accounting
   *  surprise: this is a get, not a load — the page is already resident). */
  getRowValue(slot: number): number {
    const page = this.dataDisk.readPage(this.dataPageId);
    const r = new PageReader(page);
    const off = this.rowOffset(slot);
    // value is the second u32 (after the 4-byte key).
    return r.readU32(off + 4);
  }

  walStats() {
    return this.wal.stats();
  }
  dataStats() {
    return this.dataDisk.stats();
  }
  get totalLogBytes(): number {
    return this.wal.totalLogBytes;
  }
  get confirmedDurableCommits(): number {
    return this.durableCommits;
  }
  /** Expose the data disk for the crash demo's durable/volatile comparison. */
  get rawDataDisk(): Disk {
    return this.dataDisk;
  }
  get dataPage(): number {
    return this.dataPageId;
  }
}

const EMPTY = new Uint8Array(0);

// ---------------------------------------------------------------------------
// Workload: a deterministic stream of small transactions.
// ---------------------------------------------------------------------------

/** Generate N (slot, value) transactions from a seeded PRNG. Deterministic so
 *  byte/fsync counts are reproducible; values stay in u32 range by construction. */
function buildWorkload(rng: Rng, n: number): Array<{ slot: number; value: number }> {
  const txns: Array<{ slot: number; value: number }> = [];
  for (let i = 0; i < n; i++) {
    txns.push({ slot: rng.nextInt(0, ROW_COUNT), value: rng.nextInt(1, 1_000_000) });
  }
  return txns;
}

/** Run the whole workload with a given commit config and return measured stats.
 *  This is the apples-to-apples harness: identical workload, only the flush
 *  policy differs between the two runs we compare. */
function runWorkload(
  txns: Array<{ slot: number; value: number }>,
  batch: number,
): {
  walStats: ReturnType<Disk["stats"]>;
  dataStats: ReturnType<Disk["stats"]>;
  totalLogBytes: number;
  durableCommits: number;
} {
  const db = new Database(); // no crash injected
  for (let i = 0; i < txns.length; i++) {
    db.runTxn(i, txns[i].slot, txns[i].value);
    db.maybeFlush(batch);
  }
  db.flushCommits(); // force the final partial batch so all commits are durable
  return {
    walStats: db.walStats(),
    dataStats: db.dataStats(),
    totalLogBytes: db.totalLogBytes,
    durableCommits: db.confirmedDurableCommits,
  };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function main(): void {
  const SEED = 0xc0ffee;
  const TXN_COUNT = 1000;
  const GROUP_BATCH = 8;

  console.log("=== 第 05 章 WAL：先写日志，掉电不丢已提交事务 ===\n");

  // Same workload for both policies — the only variable is the flush policy.
  const txns = buildWorkload(createRng(SEED), TXN_COUNT);

  // --- Run A: fsync every commit (no group commit) ---
  const noGroup = runWorkload(txns, 1);
  // --- Run B: group commit, batch of GROUP_BATCH ---
  const grouped = runWorkload(txns, GROUP_BATCH);

  // Per-transaction average log bytes. Deterministic (same record shape each txn:
  // one update record + one commit record).
  const avgLogBytesPerTxn = noGroup.totalLogBytes / TXN_COUNT;

  console.log(
    `工作负载：${TXN_COUNT} 个小事务，每个事务 = 1 条 update 记录(含 before/after image) + 1 条 commit 记录`,
  );
  console.log(`数据模型：单个 slotted 数据页，${ROW_COUNT} 行定宽行（每行 ${ROW_BYTES}B）\n`);

  console.log("--- WAL 体量（确定性，代码实算）---");
  printTable([
    { metric: "total log bytes", value: noGroup.totalLogBytes },
    { metric: "avg bytes / txn", value: Number(avgLogBytesPerTxn.toFixed(1)) },
    { metric: "record header bytes", value: REC_HEADER_BYTES },
    { metric: "image bytes / update", value: ROW_BYTES },
    { metric: "durable commits", value: noGroup.durableCommits },
  ]);
  console.log();

  console.log("--- fsync 对比：每事务 fsync vs group commit（攒 8 个再 fsync）---");
  // The headline number: fsync count. This is deterministic and is the honest
  // measure of durability cost — every fsync is a real disk round-trip on real
  // hardware.
  printTable([
    {
      policy: "no group (batch=1)",
      walFsyncs: noGroup.walStats.fsyncs,
      walWrites: noGroup.walStats.writes,
      dataFsyncs: noGroup.dataStats.fsyncs,
      dataWrites: noGroup.dataStats.writes,
    },
    {
      policy: `group commit (batch=${GROUP_BATCH})`,
      walFsyncs: grouped.walStats.fsyncs,
      walWrites: grouped.walStats.writes,
      dataFsyncs: grouped.dataStats.fsyncs,
      dataWrites: grouped.dataStats.writes,
    },
  ]);
  // The load-bearing observation: dataFsyncs == 0. Durability came entirely from
  // forcing the LOG; the data page was written (dirtied) but never fsync'd during
  // the txn phase. That is WAL's whole trade: turn many random data-page fsyncs
  // into one sequential log fsync per (batch of) commit(s).
  invariant(
    noGroup.dataStats.fsyncs === 0 && grouped.dataStats.fsyncs === 0,
    "durability must come from the WAL fsync, not from data-page fsyncs",
  );
  const fsyncReduction = noGroup.walStats.fsyncs / grouped.walStats.fsyncs;
  // Expected grouped fsyncs = ceil(TXN_COUNT / batch): one fsync per full batch
  // plus one for the partial tail (flushCommits at the end). State the exact
  // arithmetic so the printed ratio is verifiable, not asserted.
  const expectedGroupedFsyncs = Math.ceil(TXN_COUNT / GROUP_BATCH);
  console.log(
    `fsync 减少 ${fsyncReduction.toFixed(2)}x` +
      `（理论上界 = batch = ${GROUP_BATCH}；实际 grouped fsync = ceil(${TXN_COUNT}/${GROUP_BATCH}) = ${expectedGroupedFsyncs}）\n`,
  );

  // Sanity invariant: both policies make the SAME number of commits durable;
  // group commit changes the COST, never the correctness.
  assertEq(
    noGroup.durableCommits,
    grouped.durableCommits,
    "group commit must persist the same set of commits as per-commit fsync",
  );
  invariant(
    grouped.walStats.fsyncs < noGroup.walStats.fsyncs,
    "group commit must issue strictly fewer fsyncs",
  );

  // --- Throughput: REAL wall-clock, machine-dependent, toy-disk-optimistic ---
  benchThroughput(txns, GROUP_BATCH);

  // --- Checkpoint: prove it pushes data durable and bounds recovery scan ---
  checkpointDemo(txns);

  // --- Failure mode: crash mid-flush, log durable but data page not flushed ---
  crashDemo(txns);

  console.log("\n本章证明：commit = 强制日志落盘(fsync)，不要求数据页落盘。");
  console.log("崩溃后数据页可能落后于日志 —— 把『重放日志恢复一致状态』的责任交给第 07 章。");
}

/** Measure real per-policy throughput. Honesty: this is wall-clock from
 *  core/bench; the RAM-backed disk has no real fsync latency, so absolute ops/sec
 *  is wildly optimistic. The transferable signal is the RELATIVE speedup, and
 *  even that understates a real disk's gain (where fsync dominates). */
function benchThroughput(
  txns: Array<{ slot: number; value: number }>,
  batch: number,
): void {
  console.log("--- 吞吐对比（真实墙钟实测；RAM 盘 => 绝对值偏乐观，看相对趋势）---");
  // bench runs the closure `iters` times; we want ONE full-workload run per iter,
  // re-creating the DB each time so state doesn't accumulate. Few iters because a
  // full 1000-txn workload per iter is already a lot of work.
  const ITERS = 20;
  const noGroup = benchOnce(() => {
    const db = new Database();
    for (let i = 0; i < txns.length; i++) {
      db.runTxn(i, txns[i].slot, txns[i].value);
      db.maybeFlush(1);
    }
    db.flushCommits();
  }, ITERS);
  const grouped = benchOnce(() => {
    const db = new Database();
    for (let i = 0; i < txns.length; i++) {
      db.runTxn(i, txns[i].slot, txns[i].value);
      db.maybeFlush(batch);
    }
    db.flushCommits();
  }, ITERS);

  const noGroupTxnsPerSec = (txns.length / noGroup.nsPerOp) * 1e9;
  const groupedTxnsPerSec = (txns.length / grouped.nsPerOp) * 1e9;
  printTable([
    {
      policy: "no group (batch=1)",
      "ns/workload (measured)": Math.round(noGroup.nsPerOp),
      "txns/sec (measured)": Math.round(noGroupTxnsPerSec),
    },
    {
      policy: `group commit (batch=${batch})`,
      "ns/workload (measured)": Math.round(grouped.nsPerOp),
      "txns/sec (measured)": Math.round(groupedTxnsPerSec),
    },
  ]);
  console.log(
    `吞吐提升 ${(groupedTxnsPerSec / noGroupTxnsPerSec).toFixed(2)}x (measured)。` +
      `注意：真实盘上 fsync 是毫秒级，加速比会远大于此 —— 这里只占了"少写一个 page + 少调 fsync"的便宜。\n`,
  );
}

/** Thin wrapper around a hand-rolled bench loop. We don't reuse core/bench here
 *  because each iteration must build fresh state, and core/bench times a fixed
 *  closure — so we inline the same hrtime measurement to stay honest about it
 *  being real wall-clock. */
function benchOnce(fn: () => void, iters: number): { nsPerOp: number } {
  // warm up once so the first JIT compile doesn't pollute the measured run.
  fn();
  const start = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  const end = process.hrtime.bigint();
  return { nsPerOp: Number(end - start) / iters };
}

/** Checkpoint demo: run a prefix of txns, take a checkpoint, then show that the
 *  DURABLE data page now reflects committed values — without a checkpoint the
 *  dirty pages would have lingered in volatile cache (as the crash demo shows).
 *  This is the half of WAL that bounds recovery cost: a flushed checkpoint lets
 *  recovery skip everything before it. */
function checkpointDemo(txns: Array<{ slot: number; value: number }>): void {
  console.log("--- checkpoint：把脏数据页推到持久态，缩短恢复回放范围 ---");
  const PREFIX = 50; // commit a prefix, then checkpoint
  const db = new Database();
  const lastBySlot = new Map<number, number>(); // expected committed value per slot
  for (let i = 0; i < PREFIX; i++) {
    db.runTxn(i, txns[i].slot, txns[i].value);
    db.maybeFlush(1); // force each commit so the log is durable before checkpoint
    lastBySlot.set(txns[i].slot, txns[i].value);
  }
  // Before the checkpoint: data pages are dirty in cache, NOT durable. After it:
  // the data-page fsync inside checkpoint() makes them durable.
  db.checkpoint();

  // Reopen the durable data snapshot and confirm a sampled committed value made
  // it to disk — proving checkpoint actually pushed the dirty page out.
  const sampleSlot = txns[0].slot;
  const expected = lastBySlot.get(sampleSlot)!;
  const durable = db.rawDataDisk.reopenFromDurable();
  const durableValue = readRowValue(durable.readPage(db.dataPage), sampleSlot);
  printTable([
    { check: `slot ${sampleSlot} durable after checkpoint`, value: durableValue },
    { check: `slot ${sampleSlot} expected committed`, value: expected },
  ]);
  invariant(
    durableValue === expected,
    "checkpoint must make the latest committed value durable on the data page",
  );
  console.log(
    `  checkpoint 后数据页持久态 slot ${sampleSlot} = ${durableValue}（= 已提交值）。` +
      `恢复时可从此 checkpoint 起回放，无需扫描更早日志。\n`,
  );
}

/** Crash demo: inject a crash on a WAL fsync, then show the durable log holds a
 *  committed change whose data page never reached durable storage. This is the
 *  intermediate inconsistent state recovery must repair — we PRINT it, we do not
 *  fix it (that's chapter 07). */
function crashDemo(txns: Array<{ slot: number; value: number }>): void {
  console.log("--- 失败模式：fsync 后注入崩溃，日志已落盘但数据页未刷 ---");

  // Build a DB that crashes on its 3rd WAL fsync. With batch=1, the 3rd fsync is
  // the commit of the 3rd transaction: its update+commit records are appended,
  // the fsync flips the durable boundary on the way... but we choose the
  // pessimistic model where THIS fsync throws (see core/disk: writes before it
  // are durable, the fsync itself "did not complete"). Recovery must therefore
  // treat txn #3 as in-doubt and rely on earlier durable records.
  const CRASH_AT = 3;
  const db = new Database(CRASH_AT);

  // We deliberately NEVER flush the data disk during the txn phase: data pages
  // stay in the volatile buffer, so the data file's durable snapshot is the
  // INITIAL all-zero-value state. That is the realistic case the WAL exists for —
  // dirty pages linger in cache while commits race ahead in the log.
  let crashedAtTxn = -1;
  try {
    for (let i = 0; i < txns.length; i++) {
      db.runTxn(i, txns[i].slot, txns[i].value);
      db.maybeFlush(1); // batch=1 => every commit fsyncs the WAL => crash fires
    }
  } catch (e) {
    if (!(e instanceof CrashError)) throw e; // a real bug, not a simulated crash
    crashedAtTxn = e.fsyncOrdinal - 1; // 0-based txn index whose commit fsync died
    console.log(
      `  模拟掉电：WAL 第 ${e.fsyncOrdinal} 次 fsync 抛 CrashError（提交事务 #${crashedAtTxn} 时）`,
    );
  }
  invariant(crashedAtTxn >= 0, "crash demo expected an injected CrashError");

  // Now compare two views of the DATA disk:
  //   - in-memory (volatile) page: reflects applied changes (what we'd lose)
  //   - durable snapshot via reopenFromDurable(): what actually survives a reboot
  const slotOfFirstTxn = txns[0].slot;
  const inMemoryValue = db.getRowValue(slotOfFirstTxn);

  const durableData = db.rawDataDisk.reopenFromDurable();
  const durablePage = durableData.numPages > 0 ? durableData.readPage(db.dataPage) : null;
  const durableValue = durablePage ? readRowValue(durablePage, slotOfFirstTxn) : 0;

  printTable([
    {
      view: "data page (in-memory / volatile)",
      [`slot ${slotOfFirstTxn} value`]: inMemoryValue,
    },
    {
      view: "data page (durable after crash)",
      [`slot ${slotOfFirstTxn} value`]: durableValue,
    },
    { view: "expected committed value", [`slot ${slotOfFirstTxn} value`]: txns[0].value },
  ]);

  // The proof of inconsistency: the durable data page does NOT reflect committed
  // txn #0 (its value is still the initial 0), yet the WAL recorded and fsynced
  // txn #0's commit (it committed before the crash at txn #CRASH_AT-1). So the
  // durable state is internally inconsistent — and that is fine, because...
  invariant(
    durableValue !== txns[0].value,
    "demo precondition: data page must NOT have the committed value on disk",
  );
  console.log(
    `  数据页持久态 slot ${slotOfFirstTxn} = ${durableValue}（初始值），但 txn #0 已提交且日志已 fsync。`,
  );
  console.log("  => 数据页落后于日志：典型的 crash 中间态。");
  console.log("  => 恢复策略：重放日志中已提交事务的 after-image（REDO）即可补回，");
  console.log("     回滚未提交事务的 before-image（UNDO）即可清除 —— 第 07 章实现。");
}

/** Read a row's value field straight from a raw page buffer (crash demo helper,
 *  works on a reopened durable page without a Database wrapper). */
function readRowValue(page: Uint8Array, slot: number): number {
  const r = new PageReader(page);
  const off = SLOTTED_HEADER.HEADER_SIZE + slot * ROW_BYTES;
  return r.readU32(off + 4); // value = second u32 after the 4-byte key
}

main();
