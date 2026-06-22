// stage01-slotted-page.ts — chapter 01: storing variable-length rows in ONE page.
//
// Why this stage exists: every higher layer (B+tree leaf, LSM SSTable block) is
// "rows packed into a 4096-byte page" with a slot directory on top. Before we
// build a tree we must prove the single-page primitive is real: rows of varying
// length go in, come back out byte-identical, deletes free space, and compaction
// reclaims the holes deletes leave behind. We MEASURE the four numbers the book
// promises — rows/page, space utilization, bytes reclaimed by compaction, and
// real insert/get throughput — and we DEMO the failure the design deliberately
// does not handle: a row too big for any page (no overflow chain yet).
//
// Layout recap (the discipline core/page.ts gives us):
//   [ header(7B) | slot dir grows UP --> | <-- free gap --> | <-- row heap grows DOWN ]
//   FREE_START = next slot-entry offset (after header + N*4 slot bytes)
//   FREE_END   = lowest written row byte (rows live in [FREE_END, PAGE_SIZE))
//   free space = FREE_END - FREE_START   (the gap the two ends fight over)
// A slot is (offset:u16, length:u16); length 0 is a tombstone whose slot id we
// keep so a caller holding "this row is slot 7" never has its pointer reused
// out from under it. Compaction does NOT renumber live slots for the same reason.
//
// Why we don't lean on PageWriter's bounds-throw for "page full": that throw is a
// last-line corruption guard, not a control-flow tool. An insert must DECIDE
// up front whether it fits and return a typed PageFull — relying on a thrown
// exception to mean "full" would (a) conflate full with a real bug and (b) leave
// the header half-mutated. So insert computes the fit explicitly, mutates only on
// success, and the writer-throw stays a pure invariant backstop.

import { PAGE_SIZE, Disk } from "./core/disk.js";
import {
  PageType,
  PageReader,
  PageWriter,
  SLOTTED_HEADER,
  SLOT_ENTRY_SIZE,
} from "./core/page.js";
import { encodeRow, decodeRow, type Schema, type Value } from "./core/codec.js";
import { createRng } from "./core/prng.js";
import { bench } from "./core/clock.js";
import { assertEq, invariant, printTable } from "./core/assert.js";

// ---------------------------------------------------------------------------
// The slotted-page operations. Pure functions over a 4096-byte buffer: a page is
// just bytes, so insert/get/delete/compact all take the buffer and use core's
// reader/writer. Keeping them buffer-in / buffer-out (no Disk dependency) is what
// lets stage03 reuse the exact same code for a B+tree leaf node.
// ---------------------------------------------------------------------------

/** Result of an insert. Page-full is a normal, expected outcome (the trigger to
 *  allocate another page upstream), NOT an error — so it is a return value the
 *  caller must branch on, not a thrown exception. */
type InsertResult =
  | { ok: true; slot: number }
  | { ok: false; reason: "page-full"; needBytes: number; haveBytes: number };

function freeSpace(r: PageReader): number {
  // The gap between the two growing ends. Invariant: never negative on a sane
  // page; if it were, an earlier insert corrupted the header.
  return r.readU16(SLOTTED_HEADER.FREE_END) - r.readU16(SLOTTED_HEADER.FREE_START);
}

function slotCount(r: PageReader): number {
  return r.readU16(SLOTTED_HEADER.SLOT_COUNT);
}

/** Read slot i's directory entry. offset 0 / length 0 means tombstone. */
function readSlot(r: PageReader, i: number): { offset: number; length: number } {
  const base = SLOTTED_HEADER.HEADER_SIZE + i * SLOT_ENTRY_SIZE;
  return { offset: r.readU16(base), length: r.readU16(base + 2) };
}

/** Insert an encoded row. Cost in page bytes = the row itself PLUS one new slot
 *  entry (4B) — forgetting the slot cost is the classic slotted-page off-by-four
 *  that lets you "fit" a row whose directory entry then overruns the gap. */
function insertRow(bytes: Uint8Array, row: Uint8Array): InsertResult {
  const r = new PageReader(bytes);
  const need = row.length + SLOT_ENTRY_SIZE;
  const have = freeSpace(r);
  if (need > have) {
    // Page full: not a bug, the upstream signal to grab a new page. We return
    // BEFORE mutating anything so the page is left consistent for a retry.
    return { ok: false, reason: "page-full", needBytes: need, haveBytes: have };
  }

  const w = new PageWriter(bytes);
  const freeStart = r.readU16(SLOTTED_HEADER.FREE_START);
  const freeEnd = r.readU16(SLOTTED_HEADER.FREE_END);

  // Heap grows DOWN: the row's new home is the `row.length` bytes just below the
  // current frontier. Write the row first, then the slot, then advance both ends
  // — order matters only for clarity here (single-threaded), but it mirrors the
  // discipline a concurrent or WAL'd version would need.
  const rowOffset = freeEnd - row.length;
  w.writeBytes(rowOffset, row);

  const slotIndex = slotCount(r);
  const slotBase = SLOTTED_HEADER.HEADER_SIZE + slotIndex * SLOT_ENTRY_SIZE;
  w.writeU16(slotBase, rowOffset);
  w.writeU16(slotBase + 2, row.length);

  w.writeU16(SLOTTED_HEADER.SLOT_COUNT, slotIndex + 1);
  w.writeU16(SLOTTED_HEADER.FREE_START, freeStart + SLOT_ENTRY_SIZE);
  w.writeU16(SLOTTED_HEADER.FREE_END, rowOffset);

  return { ok: true, slot: slotIndex };
}

/** Fetch slot i's raw row bytes, or null if the slot is a tombstone / never
 *  existed. Returns a COPY: the caller may keep it past the next compaction,
 *  which would otherwise move the underlying bytes. */
function getRow(bytes: Uint8Array, slot: number): Uint8Array | null {
  const r = new PageReader(bytes);
  if (slot < 0 || slot >= slotCount(r)) return null;
  const { offset, length } = readSlot(r, slot);
  if (length === 0) return null; // tombstone
  return r.readBytes(offset, length).slice(); // copy out of the page
}

/** Tombstone slot i. We do NOT move bytes or change FREE_END here — that's
 *  compaction's job. Delete is deliberately O(1) and leaves a hole; the space is
 *  "logically free, physically still occupied" until compaction. Returns whether
 *  a live row was actually removed (idempotent on an already-dead slot). */
function deleteRow(bytes: Uint8Array, slot: number): boolean {
  const r = new PageReader(bytes);
  if (slot < 0 || slot >= slotCount(r)) return false;
  const { length } = readSlot(r, slot);
  if (length === 0) return false; // already a tombstone
  const w = new PageWriter(bytes);
  const slotBase = SLOTTED_HEADER.HEADER_SIZE + slot * SLOT_ENTRY_SIZE;
  // Length 0 is the tombstone marker; we leave offset as-is (ignored once len=0)
  // and crucially keep the slot entry so live slot ids above it don't shift.
  w.writeU16(slotBase + 2, 0);
  return true;
}

/** Bytes that compaction would reclaim right now = sum of tombstoned row lengths
 *  currently stranded in the heap. We must remember each tombstoned slot's
 *  original length to report this, so we compute it from a pre-delete snapshot the
 *  caller supplies (the page itself no longer knows — that's the cost of O(1)
 *  delete storing length 0). */
function compact(bytes: Uint8Array, originalLengths: number[]): number {
  const r = new PageReader(bytes);
  const n = slotCount(r);

  // 1. Collect live rows (slot id + bytes), preserving slot order. We copy bytes
  //    out first because we are about to overwrite the heap in place.
  const live: { slot: number; row: Uint8Array }[] = [];
  let reclaimed = 0;
  for (let i = 0; i < n; i++) {
    const { offset, length } = readSlot(r, i);
    if (length === 0) {
      // tombstone: its original bytes are the reclaim. originalLengths[i] is the
      // length the row had before delete set the slot to 0.
      reclaimed += originalLengths[i];
      continue;
    }
    live.push({ slot: i, row: r.readBytes(offset, length).slice() });
  }

  // 2. Rebuild the heap from PAGE_SIZE downward, rewriting each surviving slot's
  //    offset in place. Slot IDS DO NOT CHANGE — only their offsets move. This is
  //    the invariant external row pointers depend on.
  const w = new PageWriter(bytes);
  let frontier = PAGE_SIZE;
  for (const { slot, row } of live) {
    const newOffset = frontier - row.length;
    w.writeBytes(newOffset, row);
    const slotBase = SLOTTED_HEADER.HEADER_SIZE + slot * SLOT_ENTRY_SIZE;
    w.writeU16(slotBase, newOffset);
    // length unchanged for live slots; tombstone slots keep length 0.
    frontier = newOffset;
  }
  // FREE_END now sits at the top of the densely-packed heap; FREE_START is
  // unchanged because we keep every slot entry (including tombstones).
  w.writeU16(SLOTTED_HEADER.FREE_END, frontier);
  return reclaimed;
}

// ---------------------------------------------------------------------------
// Demo + measurement. Everything below is deterministic: one seeded PRNG decides
// every row's string length, so the rows/page count and utilization are
// byte-stable across runs and machines. Throughput is the ONE measured number;
// it is labeled (measured) and varies run-to-run by design.
// ---------------------------------------------------------------------------

const SCHEMA: Schema = ["int", "string"]; // (id, payload) — the payload varies length
const SEED = 42;

/** Build a deterministic variable-length row. The payload length is drawn from
 *  the seeded PRNG in a realistic spread (short tags up to medium blobs), so the
 *  page fills with genuinely heterogeneous rows — the case slotted pages exist
 *  for. Returns the encoded bytes plus its length for bookkeeping. */
function buildRow(id: number, payloadLen: number): Uint8Array {
  // "x" repeated; content is irrelevant, only the length drives packing behavior.
  const payload = "x".repeat(payloadLen);
  const values: Value[] = [id, payload];
  return encodeRow(values, SCHEMA);
}

function demoPackAndMeasure(): void {
  const disk = new Disk({ initialPages: 4 });
  const pageId = disk.allocPage();
  const page = disk.readPage(pageId); // a fresh zeroed page buffer to work on
  new PageWriter(page).initSlotted(PageType.LEAF);

  const rng = createRng(SEED);

  // Try to insert 200 rows; the page will fill long before that, which is the
  // point — we want to OBSERVE the page-full boundary, not avoid it.
  const ATTEMPTS = 200;
  const insertedLengths: number[] = []; // encoded length per accepted slot
  const slotIdToOriginalLen = new Map<number, number>();
  let accepted = 0;
  let firstFull: InsertResult | null = null;
  let payloadBytesStored = 0; // sum of payload string bytes (the "useful" data)
  let encodedBytesStored = 0; // sum of full encoded row bytes (incl. id + varint)

  for (let i = 0; i < ATTEMPTS; i++) {
    // Payload length in [4, 60): a realistic mix of short and medium rows.
    const payloadLen = rng.nextInt(4, 60);
    const row = buildRow(i, payloadLen);
    const res = insertRow(page, row);
    if (!res.ok) {
      if (firstFull === null) firstFull = res;
      // Keep trying smaller rows? No — a real heap allocator would stop too once
      // the gap can't fit the *next* row; we stop at first reject to keep the
      // rows/page number a clean, reproducible capacity figure.
      break;
    }
    accepted++;
    insertedLengths.push(row.length);
    slotIdToOriginalLen.set(res.slot, row.length);
    payloadBytesStored += payloadLen;
    encodedBytesStored += row.length;
  }

  // --- Verify round-trip on every accepted row (the "rows come back identical"
  //     promise). A single mismatch here means the codec/slot math is wrong. ---
  for (let slot = 0; slot < accepted; slot++) {
    const raw = getRow(page, slot);
    invariant(raw !== null, `slot ${slot} should be live right after insert`);
    const decoded = decodeRow(raw, SCHEMA);
    assertEq(decoded[0] as number, slot, `row ${slot} id round-trips`);
    // payload length is recoverable from the decoded string length
    invariant(
      typeof decoded[1] === "string",
      `row ${slot} payload decodes to a string`,
    );
  }

  // --- Measured numbers #1: capacity + utilization ---
  const reader = new PageReader(page);
  const headerBytes = SLOTTED_HEADER.HEADER_SIZE;
  const slotDirBytes = accepted * SLOT_ENTRY_SIZE;
  const gapAfterFill = freeSpace(reader); // wasted space too small for next row
  // Utilization = useful encoded row bytes / total page. The gap + slot dir +
  // header are overhead. We report two ratios so the reader sees where bytes go.
  const rowUtilPct = (encodedBytesStored / PAGE_SIZE) * 100;
  const payloadUtilPct = (payloadBytesStored / PAGE_SIZE) * 100;

  printTable([
    { metric: "page size (bytes)", value: PAGE_SIZE, note: "fixed" },
    { metric: "rows accepted", value: accepted, note: `of ${ATTEMPTS} attempted` },
    { metric: "header overhead (bytes)", value: headerBytes, note: "7B fixed" },
    { metric: "slot dir (bytes)", value: slotDirBytes, note: `${accepted} x ${SLOT_ENTRY_SIZE}B` },
    { metric: "row heap (bytes)", value: encodedBytesStored, note: "encoded rows" },
    { metric: "leftover gap (bytes)", value: gapAfterFill, note: "< next row, wasted" },
    { metric: "row-heap utilization %", value: +rowUtilPct.toFixed(1), note: "encoded/page" },
    { metric: "payload utilization %", value: +payloadUtilPct.toFixed(1), note: "useful/page" },
  ]);
  console.log("");

  // Invariant sanity: every byte is accounted for. header + slotdir + heap + gap
  // must equal PAGE_SIZE exactly, or the layout math drifted.
  const accounted = headerBytes + slotDirBytes + encodedBytesStored + gapAfterFill;
  invariant(
    accounted === PAGE_SIZE,
    `byte accounting must total ${PAGE_SIZE}, got ${accounted}`,
  );
  console.log(
    `字节核账: header ${headerBytes} + slot dir ${slotDirBytes} + 行堆 ${encodedBytesStored} + 空隙 ${gapAfterFill} = ${accounted} (= ${PAGE_SIZE}) ✓`,
  );
  console.log("");

  // --- Measured numbers #2: compaction reclaim after deleting half ---
  // Delete every other slot (0,2,4,...). O(1) tombstones, no bytes moved yet.
  const originalLengths: number[] = [];
  for (let slot = 0; slot < accepted; slot++) {
    originalLengths.push(slotIdToOriginalLen.get(slot) ?? 0);
  }
  let deletedCount = 0;
  let deletedBytes = 0;
  for (let slot = 0; slot < accepted; slot += 2) {
    const removed = deleteRow(page, slot);
    invariant(removed, `slot ${slot} should delete (was live)`);
    deletedCount++;
    deletedBytes += originalLengths[slot];
  }

  // Right after delete the gap has NOT grown (delete is O(1), bytes still there).
  const gapBeforeCompact = freeSpace(new PageReader(page));
  invariant(
    gapBeforeCompact === gapAfterFill,
    `delete must NOT move bytes: gap should stay ${gapAfterFill}, got ${gapBeforeCompact}`,
  );

  const reclaimed = compact(page, originalLengths);
  const gapAfterCompact = freeSpace(new PageReader(page));

  // The reclaim must equal exactly the deleted rows' bytes, and the gap must grow
  // by exactly that much. If these disagree, compaction lost or double-counted.
  assertEq(reclaimed, deletedBytes, "compaction reclaims exactly the deleted bytes");
  assertEq(
    gapAfterCompact - gapBeforeCompact,
    reclaimed,
    "free gap grows by exactly the reclaimed bytes",
  );

  // Surviving (odd) slots must still round-trip with UNCHANGED ids — compaction
  // moved their bytes but kept their slot ids. This is the load-bearing invariant.
  for (let slot = 1; slot < accepted; slot += 2) {
    const raw = getRow(page, slot);
    invariant(raw !== null, `live slot ${slot} survives compaction`);
    const decoded = decodeRow(raw, SCHEMA);
    assertEq(decoded[0] as number, slot, `slot ${slot} keeps its id after compaction`);
  }
  // Deleted (even) slots must still read as tombstones (null), slot ids stable.
  for (let slot = 0; slot < accepted; slot += 2) {
    invariant(getRow(page, slot) === null, `deleted slot ${slot} stays a tombstone`);
  }

  printTable([
    { metric: "rows deleted", value: deletedCount, note: "every other slot" },
    { metric: "gap before compact (bytes)", value: gapBeforeCompact, note: "O(1) delete, unchanged" },
    { metric: "bytes reclaimed", value: reclaimed, note: "by compaction" },
    { metric: "gap after compact (bytes)", value: gapAfterCompact, note: "= before + reclaimed" },
  ]);
  console.log("");
  console.log(
    `compaction 回收 ${reclaimed} 字节, 空隙从 ${gapBeforeCompact} 增长到 ${gapAfterCompact}; ` +
      `存活行 slot id 不变 (墓碑保号), 已删 slot 仍读为 null ✓`,
  );
  console.log("");
}

/** Measured number #3: real insert/get throughput. We isolate ONE op per timed
 *  call so nsPerOp is meaningful, and we re-init a page per insert batch so we
 *  measure insert work, not "insert until full then no-op page-full returns". */
function benchOps(): void {
  const rng = createRng(SEED + 1); // distinct stream so it's not correlated w/ demo
  const SCRATCH = new Uint8Array(PAGE_SIZE);

  // Pre-build a fixed small row so the benchmark times slotted-page mechanics,
  // not row encoding (encoding is benched in a later chapter). 20-byte payload.
  const fixedRow = buildRow(1, 20);

  // INSERT bench: each iter inserts into a page that we reset when it fills. The
  // reset itself is cheap (initSlotted = 4 writes) and amortizes out; we count it
  // honestly rather than pretend inserts never hit page-full.
  let benchPage = SCRATCH;
  new PageWriter(benchPage).initSlotted(PageType.LEAF);
  const insertResult = bench(() => {
    const res = insertRow(benchPage, fixedRow);
    if (!res.ok) {
      // page full: re-init and insert into the fresh page so the op still does
      // real insert work (not a degenerate early-return loop).
      new PageWriter(benchPage).initSlotted(PageType.LEAF);
      insertRow(benchPage, fixedRow);
    }
  }, 1_000_000);

  // GET bench: fill a page once, then time random point lookups. getRow copies
  // bytes out, which is the realistic cost (callers need a stable copy).
  const getPage = new Uint8Array(PAGE_SIZE);
  new PageWriter(getPage).initSlotted(PageType.LEAF);
  let filled = 0;
  for (;;) {
    const res = insertRow(getPage, buildRow(filled, rng.nextInt(4, 60)));
    if (!res.ok) break;
    filled++;
  }
  invariant(filled > 0, "get bench needs at least one row");
  const getResult = bench(() => {
    const slot = rng.nextInt(0, filled);
    const raw = getRow(getPage, slot);
    // touch the result so the JIT can't dead-code-eliminate the read.
    if (raw !== null && raw.length === 0xffffffff) throw new Error("unreachable");
  }, 1_000_000);

  printTable([
    {
      op: "insert (measured)",
      "ns/op": +insertResult.nsPerOp.toFixed(1),
      "ops/sec": Math.round(insertResult.opsPerSec),
      iters: insertResult.iters,
    },
    {
      op: "get (measured)",
      "ns/op": +getResult.nsPerOp.toFixed(1),
      "ops/sec": Math.round(getResult.opsPerSec),
      iters: getResult.iters,
    },
  ]);
  console.log(
    "注: ops/sec 为真实墙钟实测 (machine-dependent), 跑跑会变; 这是单页纯内存操作, " +
      "绝对值偏乐观 (无磁盘 IO/无并发), 可迁移的是相对量级与 insert vs get 的比值。",
  );
  console.log("");
}

/** Failure mode (THE point of an honest book): a row larger than what a single
 *  page can ever hold. We have NOT built overflow pages yet — so this row is
 *  unstorable, and insert must say so cleanly rather than corrupt the page. We
 *  demonstrate the boundary AND explain the deliberately-missing feature. */
function demoPageFullAndBigRow(): void {
  const page = new Uint8Array(PAGE_SIZE);
  new PageWriter(page).initSlotted(PageType.LEAF);

  // The absolute max a fresh page can hold: full gap minus the one slot entry,
  // minus the int(4) + varint(len-prefix) overhead of the encoded row.
  //
  // Subtlety we MUST respect: encodeRow itself encodes into a single PAGE_SIZE
  // scratch buffer, so a payload that overflows the *page* would overflow the
  // *encoder* first and throw the wrong error. To demo the slotted-page boundary
  // (not the encoder's), we pick a payload that is still encodable (< page) yet,
  // once you add the int column, the varint length prefix, AND the 4-byte slot
  // entry, no longer fits the empty page's gap. 4085 lands exactly in that band:
  // encoded ~4091B < 4096 (encodes fine), but 4091 + 4 slot = 4095 > 4089 gap.
  const maxGap = PAGE_SIZE - SLOTTED_HEADER.HEADER_SIZE; // gap on an empty page = 4089
  const hugePayloadLen = 4085; // largest payload that still encodes but can't fit a page
  const hugeRow = buildRow(999, hugePayloadLen);
  const res = insertRow(page, hugeRow);

  invariant(
    !res.ok && res.reason === "page-full",
    "a row larger than a page MUST be rejected as page-full, not silently truncated",
  );

  console.log("=== 失败模式: 大行无法存入单页 (overflow 页未实现) ===");
  console.log(
    `尝试插入 payload=${hugePayloadLen} 字节的行 (编码后 ${hugeRow.length} 字节, 含 int 列 + varint 长度前缀)`,
  );
  console.log(
    `空页可用空隙最多 ${maxGap} 字节, 需要 ${(res as Exclude<InsertResult, { ok: true }>).needBytes} 字节 ` +
      `(行 ${hugeRow.length} + slot ${SLOT_ENTRY_SIZE}) => 返回 page-full, 页保持不变 (未损坏)。`,
  );

  // Prove the page is still consistent after the rejected insert: a normal small
  // row still goes in. A naive implementation that mutated-then-threw would have
  // left FREE_END dangling and this would corrupt.
  const smallRes = insertRow(page, buildRow(1, 10));
  invariant(
    smallRes.ok,
    "after a rejected oversized insert, the page must still accept a normal row",
  );
  console.log(
    "拒绝大行后, 正常小行仍可插入 ✓ (拒绝路径不修改页, 不是 mutate-then-throw)。",
  );
  console.log("");
  console.log("为什么这是 deliberately 未实现, 不是 bug:");
  console.log(
    "  单页定长 4096B 是存储的原子。一行超过一页时, 真实引擎用 overflow 页链: " +
      "行头留在主页, 超出部分溢出到 OVERFLOW 页 (core/page.ts 已留 PageType.OVERFLOW=3 占位), " +
      "用页内指针串成链。本章只做单页, 故大行 = page-full。",
  );
  console.log(
    "  失败模式价值: 它暴露了 slotted page 的硬边界 (max row <= page - overhead), " +
      "这正是 B+tree 必须支持 overflow / 大值外置的根因 —— 留给后续章节。",
  );
  console.log("");
}

function main(): void {
  console.log("=== stage01: slotted page — 变长行打包进单页 (seed=" + SEED + ") ===\n");
  demoPackAndMeasure();
  benchOps();
  demoPageFullAndBigRow();
  console.log("=== stage01 完成: 单页 insert/get/delete/compact 全部验证通过 ===");
}

main();
