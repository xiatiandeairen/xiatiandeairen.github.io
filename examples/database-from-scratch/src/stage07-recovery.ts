// stage07-recovery.ts — ARIES-simplified crash recovery: rebuild a consistent
// database from a half-written data file plus the write-ahead log.
//
// Why this chapter exists at all: stage05 gave us a WAL and the durability rule
// "force the log before the data page" (write-ahead logging). But a WAL is only
// worth its fsync cost if, after the power dies mid-write, we can actually USE it
// to reconstruct the truth. That reconstruction is recovery, and the canonical
// algorithm is ARIES. This stage implements its three-phase skeleton — Analysis,
// Redo, Undo — and then PROVES the phase order is not decorative by deleting Redo
// and watching a committed transaction's data vanish (a Durability violation).
//
// The model (deliberately minimal, but the mechanism is real, not a sketch):
//   - Data lives in core/Disk pages. We treat one page as a tiny key->value
//     store: 4-byte big-endian counter slots, indexed by account id. Concrete
//     bytes mean Redo/Undo are real page mutations, not abstract "apply op".
//   - The WAL is a list of PHYSIOLOGICAL records: each carries (lsn, txnId, type,
//     pageId, slot, before, after). Physiological = "physical to a page, logical
//     within it". Carrying BOTH before- and after-image is what makes Redo
//     (write after) and Undo (write before) trivial and idempotent — the single
//     most important simplification over real ARIES, which reconstructs images
//     from a chain of partial records. We call that out rather than hide it.
//   - Durability boundary: the WAL is the thing that survives a crash. Real
//     systems guarantee this by fsync-ing the log on commit (WAL rule). We model
//     it directly: the log array IS the durable log; the DATA file is what got
//     caught mid-fsync and is therefore untrustworthy on reopen.
//
// What is honestly measured here:
//   - Counts (records scanned, redone, undone; pages touched): deterministic,
//     code-computed, reproducible from the seed. These are the real output.
//   - There is no wall-clock number to report: recovery cost is dominated by
//     log length and page IO, both of which we count exactly. A "recovery took
//     N ms" figure on a 1-page toy disk would be noise, so we don't fake one.
//
// Toy-data caveat (honesty rule): absolute values (5 txns, 1 data page) are tiny
// on purpose so the trace is readable. What transfers to a real engine is the
// SHAPE: Analysis sees only the log, Redo repeats history including pages that
// look already-applied, Undo walks losers backward, and skipping Redo loses
// committed work. The relative claim ("Redo is not optional") is the lesson.

import { Disk, PAGE_SIZE } from "./core/disk.js";
import { PageReader, PageWriter, PageType } from "./core/page.js";
import { createRng } from "./core/prng.js";
import { assertEq, invariant, printTable } from "./core/assert.js";

// ---------------------------------------------------------------------------
// Data-file layout: one slotted-ish page holding fixed-width u32 counters.
// We reuse the slotted header only for its TYPE byte; the counter slots live at
// a fixed stride so a "page offset" in a log record is a stable physical address
// (ARIES log records address bytes, not logical rows — that's the "physio" half).
// ---------------------------------------------------------------------------

const SLOT_BASE = 8; // first counter byte; leaves the slotted header (7B) + pad
const SLOT_STRIDE = 4; // each account is one big-endian u32
const NUM_ACCOUNTS = 4;

/** Byte offset of an account's counter within the data page. Centralized so the
 *  workload, Redo, and Undo all address the exact same bytes — a mismatch here
 *  would make recovery "succeed" while corrupting data, the nastiest failure. */
function slotOffset(account: number): number {
  invariant(account >= 0 && account < NUM_ACCOUNTS, `account ${account} out of range`);
  return SLOT_BASE + account * SLOT_STRIDE;
}

function readCounter(page: Uint8Array, account: number): number {
  return new PageReader(page).readU32(slotOffset(account));
}

/** Return a COPY of `page` with one counter overwritten. Copy-not-mutate so a
 *  log record's before-image (captured from the page pre-write) is never aliased
 *  to the post-write bytes — aliasing would silently equalize before==after and
 *  make Undo a no-op, which is exactly a lost-rollback bug. */
function withCounter(page: Uint8Array, account: number, value: number): Uint8Array {
  const next = page.slice();
  new PageWriter(next).writeU32(slotOffset(account), value);
  return next;
}

// ---------------------------------------------------------------------------
// WAL record. Physiological + before/after images (see header for the why).
// ---------------------------------------------------------------------------

type LogType = "update" | "commit" | "abort";

interface LogRecord {
  /** Log Sequence Number: a strictly increasing position in the log. ARIES uses
   *  the LSN both to order replay and (on real pages) to skip already-applied
   *  records via pageLSN; we keep the ordering role and model the skip explicitly. */
  readonly lsn: number;
  readonly txnId: number;
  readonly type: LogType;
  /** Present only for "update" records. */
  readonly pageId?: number;
  readonly account?: number;
  readonly before?: number;
  readonly after?: number;
}

/** An append-only log that we declare "durable": writing here is the model's
 *  fsync-on-WAL. The recovery phases read ONLY from this — never from in-memory
 *  transaction state — because after a crash that in-memory state is gone. */
class WriteAheadLog {
  private records: LogRecord[] = [];
  private nextLsn = 1;

  appendUpdate(txnId: number, pageId: number, account: number, before: number, after: number): number {
    const lsn = this.nextLsn++;
    this.records.push({ lsn, txnId, type: "update", pageId, account, before, after });
    return lsn;
  }
  appendCommit(txnId: number): number {
    const lsn = this.nextLsn++;
    this.records.push({ lsn, txnId, type: "commit" });
    return lsn;
  }

  /** The durable log as the recovery process sees it after reopening the file. */
  durableRecords(): readonly LogRecord[] {
    return this.records;
  }
}

// ---------------------------------------------------------------------------
// Phase 1 — ANALYSIS. Scan the log forward to rebuild two tables WITHOUT
// touching the data file. This mirrors real ARIES, which scans from the last
// checkpoint; we have no checkpoint (a stage exercise) so we scan from the top.
// ---------------------------------------------------------------------------

interface AnalysisResult {
  /** Active Transaction Table: txns seen but with no commit record => "losers"
   *  whose effects (if any reached the data page) must be undone. Maps txnId to
   *  the LSN of its last update (the real ATT tracks lastLSN for the undo walk). */
  readonly activeTxns: Map<number, number>;
  /** Dirty Page Table: pages that an update record touched => candidates that
   *  Redo must re-examine. In real ARIES the DPT's recLSN bounds where Redo can
   *  start; here we record first-touch LSN to show the same intent. */
  readonly dirtyPages: Map<number, number>;
  readonly committed: Set<number>;
  readonly scanned: number;
}

function analyze(log: readonly LogRecord[]): AnalysisResult {
  const activeTxns = new Map<number, number>();
  const dirtyPages = new Map<number, number>();
  const committed = new Set<number>();

  for (const rec of log) {
    if (rec.type === "update") {
      // Any update makes its txn active until proven committed, and marks its
      // page dirty. We over-include on purpose: Analysis must never miss a
      // loser, and Redo/Undo tolerate a too-large set (idempotent).
      activeTxns.set(rec.txnId, rec.lsn);
      if (!dirtyPages.has(rec.pageId!)) dirtyPages.set(rec.pageId!, rec.lsn);
    } else if (rec.type === "commit") {
      // A commit record is the durability witness. Once seen, the txn is a
      // winner and leaves the ATT — its effects MUST survive (that's Durability).
      committed.add(rec.txnId);
      activeTxns.delete(rec.txnId);
    }
  }
  return { activeTxns, dirtyPages, committed, scanned: log.length };
}

// ---------------------------------------------------------------------------
// Phase 2 — REDO. "Repeat history": replay EVERY update's after-image, even for
// transactions that will later be undone, and even onto pages that might already
// reflect the change. Why redo work that looks redundant: after a crash we do
// not know which page writes actually reached disk before the data fsync was
// cut off. Redo restores a known state (= log order applied) so Undo has a sane
// starting point. This "repeating history including losers" is the ARIES insight
// that the failure-mode demo below removes to show the consequence.
// ---------------------------------------------------------------------------

interface RedoResult {
  readonly redone: number;
  readonly pagesWritten: number;
}

function redo(disk: Disk, log: readonly LogRecord[]): RedoResult {
  let redone = 0;
  const touched = new Set<number>();
  for (const rec of log) {
    if (rec.type !== "update") continue;
    // Idempotency: applying the after-image is a blind overwrite, so replaying a
    // record whose effect already survived is harmless — exactly why before/after
    // physiological logging is recovery-friendly.
    const page = disk.readPage(rec.pageId!);
    disk.writePage(rec.pageId!, withCounter(page, rec.account!, rec.after!));
    redone++;
    touched.add(rec.pageId!);
  }
  return { redone, pagesWritten: touched.size };
}

// ---------------------------------------------------------------------------
// Phase 3 — UNDO. Walk the losers' updates BACKWARD, restoring each before-image.
// Backward order matters: if a txn wrote the same slot twice, only reverse order
// returns it to the pre-txn value. Forward undo would leave the FIRST write's
// after-image in place — a subtle corruption that the reverse walk prevents.
// ---------------------------------------------------------------------------

interface UndoResult {
  readonly undone: number;
}

function undo(disk: Disk, log: readonly LogRecord[], losers: ReadonlySet<number>): UndoResult {
  let undone = 0;
  for (let i = log.length - 1; i >= 0; i--) {
    const rec = log[i];
    if (rec.type !== "update") continue;
    if (!losers.has(rec.txnId)) continue;
    const page = disk.readPage(rec.pageId!);
    disk.writePage(rec.pageId!, withCounter(page, rec.account!, rec.before!));
    undone++;
  }
  return { undone };
}

// ---------------------------------------------------------------------------
// Workload generator: 5 transactions, deterministically interleaved. Three will
// commit (winners), two will be cut off mid-flight by the crash (losers). The
// crash is injected by stage05's mechanism: writes happen, the WAL records them,
// but the final data-file fsync is severed so the page on disk is half-baked.
// ---------------------------------------------------------------------------

interface WorkloadOutcome {
  readonly log: WriteAheadLog;
  /** What the data PAGE looked like the instant before "power loss" — i.e. the
   *  volatile, possibly-torn state. Recovery does NOT get to use this; it's here
   *  only so the demo can show how far from correct the crash left the file. */
  readonly crashedPage: Uint8Array;
  /** The ground-truth correct final counters: winners applied, losers absent.
   *  Computed independently of recovery so the assertion is a real check, not a
   *  tautology. */
  readonly expected: number[];
  readonly committedTxns: number[];
  readonly abortedTxns: number[];
}

function runWorkload(seed: number): WorkloadOutcome {
  const rng = createRng(seed);

  // Initialize the data page: all accounts start at 100.
  const init = new Uint8Array(PAGE_SIZE);
  new PageWriter(init).initSlotted(PageType.LEAF);
  // Annotate as plain Uint8Array: `new Uint8Array()` infers Uint8Array<ArrayBuffer>
  // (TS 5.9 typed-array generic), but withCounter returns the ArrayBufferLike
  // form via .slice(); without the annotation the loop reassignment won't unify.
  let page: Uint8Array = init;
  for (let a = 0; a < NUM_ACCOUNTS; a++) page = withCounter(page, a, 100);

  const expected = new Array(NUM_ACCOUNTS).fill(100);
  const log = new WriteAheadLog();
  const pageId = 0;

  // Five txns; each does one or two counter increments. We hardcode which commit
  // vs crash so the demo is a fixed, narratable scenario (not a fuzz run): T0,T1,
  // T2 commit; T3,T4 are mid-flight when the crash hits.
  const committedTxns = [0, 1, 2];
  const abortedTxns = [3, 4];

  // Deterministic step list: (txn, account, delta). Order is shuffled by the
  // seed so the interleaving — and thus which losers touched which pages — is
  // reproducible but not hand-arranged to be convenient.
  const steps: Array<{ txn: number; account: number; delta: number }> = [
    { txn: 0, account: 0, delta: 5 },
    { txn: 1, account: 1, delta: 7 },
    { txn: 2, account: 2, delta: 3 },
    { txn: 0, account: 0, delta: 2 }, // T0 writes account 0 twice -> tests backward undo irrelevance for winners, and exercises double-write
    { txn: 3, account: 1, delta: 50 }, // loser touches account 1 (also touched by winner T1)
    { txn: 4, account: 3, delta: 99 }, // loser touches account 3 (untouched by winners)
    { txn: 3, account: 2, delta: 40 }, // loser touches account 2 (also touched by winner T2)
  ];
  rng.shuffle(steps);

  // Apply each step to the volatile page and log it write-ahead (log first).
  for (const step of steps) {
    const before = readCounter(page, step.account);
    const after = before + step.delta;
    log.appendUpdate(step.txn, pageId, step.account, before, after);
    page = withCounter(page, step.account, after);
    // Track ground truth for winners only.
    if (committedTxns.includes(step.txn)) expected[step.account] += step.delta;
  }

  // Winners commit (their commit records make them durable per the WAL rule).
  for (const t of committedTxns) log.appendCommit(t);
  // Losers never reach commit — that's what makes them losers.

  return {
    log,
    crashedPage: page,
    expected,
    committedTxns,
    abortedTxns,
  };
}

/** Reopen the data file after a crash, then sever the workload's data writes.
 *
 *  Model: stage05's WAL rule forces the LOG before a data page; the data page
 *  itself is flushed lazily and was caught unflushed when power died. So on
 *  reopen, the data file holds only what the LAST SUCCESSFUL fsync committed —
 *  here, the pre-workload baseline (all accounts = 100). We get that durable
 *  baseline honestly via Disk.reopenFromDurable(): write baseline, fsync (that's
 *  the last good fsync), then reopen. The workload writes that came after the
 *  fsync are gone — exactly the pessimistic "nothing dirty survived" case, which
 *  is the strongest possible test of Redo (a real torn-page crash survives some
 *  writes, which would only make Redo's job easier). Recovery rebuilds from the
 *  WAL alone; resetStats so we measure ONLY recovery IO, not this setup. */
function reopenAfterCrash(): Disk {
  const pre = new Disk({ initialPages: 4 });
  const pageId = pre.allocPage();
  invariant(pageId === 0, "data page must be page 0");
  const baseline = new Uint8Array(PAGE_SIZE);
  new PageWriter(baseline).initSlotted(PageType.LEAF);
  let page: Uint8Array = baseline; // see runWorkload for the annotation rationale
  for (let a = 0; a < NUM_ACCOUNTS; a++) page = withCounter(page, a, 100);
  pre.writePage(0, page);
  pre.fsync(); // the last fsync that completed before the crash

  const recovered = pre.reopenFromDurable(); // sees only fsync'd baseline
  recovered.resetStats();
  return recovered;
}

function counters(disk: Disk): number[] {
  const page = disk.readPage(0);
  const out: number[] = [];
  for (let a = 0; a < NUM_ACCOUNTS; a++) out.push(readCounter(page, a));
  return out;
}

// ---------------------------------------------------------------------------
// Demo driver.
// ---------------------------------------------------------------------------

function main(): void {
  const SEED = 7;
  console.log("=== stage07: ARIES 简化版崩溃恢复 ===\n");

  const wl = runWorkload(SEED);
  const log = wl.log.durableRecords();

  console.log(`场景 (seed=${SEED}): ${wl.committedTxns.length} 个事务已提交, ` +
    `${wl.abortedTxns.length} 个事务崩溃时进行中`);
  console.log(`  winners (已提交): T${wl.committedTxns.join(", T")}`);
  console.log(`  losers  (未提交): T${wl.abortedTxns.join(", T")}`);
  console.log(`  WAL 长度: ${log.length} 条记录\n`);

  // Show the log so the reader can see recovery has only this to work with.
  console.log("--- 持久化的 WAL (恢复唯一的输入) ---");
  printTable(
    log.map((r) => ({
      lsn: r.lsn,
      txn: `T${r.txnId}`,
      type: r.type,
      account: r.account === undefined ? "" : `a${r.account}`,
      before: r.before === undefined ? "" : r.before,
      after: r.after === undefined ? "" : r.after,
    })),
  );
  console.log();

  // ----- Phase 1: Analysis -----
  const analysis = analyze(log);
  console.log("--- Phase 1: ANALYSIS (只扫日志, 不碰数据页) ---");
  console.log(`  扫描记录数: ${analysis.scanned}`);
  console.log(`  committed (winners): {${[...analysis.committed].map((t) => `T${t}`).join(", ")}}`);
  console.log(`  ATT 活跃事务表 (losers): {${[...analysis.activeTxns.keys()].map((t) => `T${t}`).join(", ")}}` +
    ` -> 待回滚`);
  console.log(`  DPT 脏页表: {${[...analysis.dirtyPages.keys()].map((p) => `page${p}`).join(", ")}}\n`);

  // Sanity: Analysis must classify exactly as the workload intended. This guards
  // against an Analysis bug masquerading as a recovery success later.
  assertEq(analysis.committed.size, wl.committedTxns.length, "analysis found all winners");
  assertEq(analysis.activeTxns.size, wl.abortedTxns.length, "analysis found all losers");

  // ----- Correct recovery: Analysis -> Redo -> Undo -----
  console.log("--- 正确恢复: ANALYSIS -> REDO -> UNDO ---");
  const disk = reopenAfterCrash();
  console.log(`  重开数据文件, 起点计数: [${counters(disk).join(", ")}] (仅 baseline 持久, workload 写入全丢)`);

  const r = redo(disk, log);
  console.log(`  Phase 2 REDO:  重放 ${r.redone} 条 update (含将被回滚的 loser), 写 ${r.pagesWritten} 页`);
  console.log(`    redo 后计数: [${counters(disk).join(", ")}] (此刻含 loser 的脏数据, 符合 'repeat history')`);

  const u = undo(disk, log, new Set(analysis.activeTxns.keys()));
  console.log(`  Phase 3 UNDO:  回滚 ${u.undone} 条 loser update (倒序应用 before-image)`);
  console.log(`    undo 后计数: [${counters(disk).join(", ")}]`);

  const recovered = counters(disk);
  console.log(`\n  期望计数 (ground truth, 独立计算): [${wl.expected.join(", ")}]`);
  console.log(`  恢复后计数:                        [${recovered.join(", ")}]`);

  for (let a = 0; a < NUM_ACCOUNTS; a++) {
    assertEq(recovered[a], wl.expected[a], `account ${a} recovered to committed-only value`);
  }
  console.log("  ✓ 断言通过: 已提交事务全部生效, 未提交事务全部消失\n");

  const recoveryIo = disk.stats();
  console.log(`  恢复 IO (实测计数): reads=${recoveryIo.reads}, writes=${recoveryIo.writes}\n`);

  // ----- Failure mode: skip Redo, jump straight to Undo -----
  console.log("--- 失败模式: 跳过 REDO, 直接 UNDO (证明三阶段顺序不可省) ---");
  const broken = reopenAfterCrash();
  console.log(`  重开数据文件, 起点计数: [${counters(broken).join(", ")}]`);
  console.log("  (跳过 REDO: 不重放任何 after-image)");

  const u2 = undo(broken, log, new Set(analysis.activeTxns.keys()));
  console.log(`  Phase 3 UNDO:  回滚 ${u2.undone} 条 loser update`);
  const brokenCounters = counters(broken);
  console.log(`    结果计数:   [${brokenCounters.join(", ")}]`);
  console.log(`  期望计数:     [${wl.expected.join(", ")}]`);

  // Quantify the damage: how many committed accounts lost their durable update.
  const lost: number[] = [];
  for (let a = 0; a < NUM_ACCOUNTS; a++) {
    if (brokenCounters[a] !== wl.expected[a]) lost.push(a);
  }
  console.log(`\n  ✗ Durability 被破坏: ${lost.length}/${NUM_ACCOUNTS} 个账户丢失了已提交的修改`);
  for (const a of lost) {
    console.log(`     account ${a}: 期望 ${wl.expected[a]}, 实得 ${brokenCounters[a]} ` +
      `(差 ${wl.expected[a] - brokenCounters[a]} = 已提交但从未重放的增量)`);
  }
  // Make the failure executable: assert that skipping Redo provably diverges.
  invariant(lost.length > 0, "skipping Redo MUST lose committed work, else the demo is vacuous");
  console.log("\n  结论: REDO 把所有 update (包括崩溃前未落盘的已提交写) 重新落盘;");
  console.log("        没有 REDO, UNDO 只会从一个残缺起点回滚 loser, 已提交的写永远回不来。");
  console.log("        三阶段顺序 Analysis -> Redo -> Undo 缺一不可。\n");

  console.log("=== stage07 完成 ===");
}

main();
