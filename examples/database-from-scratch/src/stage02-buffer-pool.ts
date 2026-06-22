// stage02-buffer-pool.ts — a fixed-size buffer pool over the page disk, with two
// replacement policies (LRU and CLOCK) measured under two opposing workloads.
//
// Why this chapter exists: the disk forces whole-page IO, but RAM is finite, so a
// real engine keeps only N pages cached and must decide *who gets evicted* when a
// new page is needed and all N frames are full. That single decision is the
// difference between a database that does one IO per query and one that thrashes.
// This stage makes the cost visible: it counts EVERY Disk.readPage and reports the
// hit rate, so the policy's quality is a measured number, not an opinion.
//
// What the numbers mean / honesty caveats:
//  - hitRate and the disk read count are EXACT counts of real calls, fully
//    deterministic given the seed (the workload is seeded-PRNG generated).
//  - The workloads are TOYS: a 64-page table and a 64-request trace. Absolute
//    hit rates here are optimistic versus a real OLTP buffer pool over millions of
//    pages. What transfers is the RELATIVE story — LRU's collapse under a
//    full-table scan, and CLOCK approximating LRU at lower bookkeeping cost.
//
// Invariants this file establishes and checks:
//  - Frame count is fixed at construction; the pool never holds more than N pages.
//  - A pinned frame is never evicted (a pin means "someone is reading these bytes
//    right now; stealing the frame would hand them a different page mid-read").
//  - If every frame is pinned and a new page is requested, the pool throws
//    BufferFull rather than silently over-allocating — see the deliberate failure
//    demo at the end. This is the deadlock precursor: a real pool would block the
//    requester forever, so the lesson is that callers MUST unpin promptly.
//
// We do NOT import any stageNN file (importing one runs its main()). We reuse
// core/disk (the counted IO) and core/prng (the only randomness source).

import { Disk, PAGE_SIZE } from "./core/disk.js";
import { createRng, type Rng } from "./core/prng.js";
import { printTable, invariant, assertEq } from "./core/assert.js";

/** Thrown when a new page is requested but every frame is pinned, so there is no
 *  victim to evict. Distinct type so callers can tell "pool is full of in-use
 *  pages" (a backpressure/lifecycle problem) from a generic bug. */
class BufferFull extends Error {
  constructor(public readonly frameCount: number) {
    super(`buffer pool full: all ${frameCount} frames are pinned, no victim to evict`);
    this.name = "BufferFull";
  }
}

type Policy = "LRU" | "CLOCK";

/** One cached page. We keep the bytes plus the bookkeeping each policy needs.
 *  `pinCount` gates eviction; `dirty` gates whether eviction must write back. */
interface Frame {
  pageId: number;
  bytes: Uint8Array; // a private copy; mutating it does NOT touch the disk until flush
  pinCount: number;
  dirty: boolean;
  refBit: boolean; // CLOCK only: set on access, cleared by the sweeping hand
}

interface PoolStats {
  hits: number;
  misses: number;
  evictions: number;
  writebacks: number; // dirty victims flushed to disk on eviction
  evictFailures: number; // times the policy scanned but found every frame pinned
}

/** A buffer pool of `frameCount` frames over a Disk. getPage pins-and-returns a
 *  frame; the caller MUST unpin when done (the failure demo shows why). The pool
 *  is the only thing that calls disk.readPage/writePage, so disk.stats().reads is
 *  the ground truth for "IO the cache could not avoid". */
class BufferPool {
  private frames: Frame[] = []; // dense; index is the frame slot
  private byPage = new Map<number, number>(); // pageId -> frame index, O(1) lookup
  private lruOrder: number[] = []; // LRU only: frame indices, front = least recently used
  private clockHand = 0; // CLOCK only: next frame the sweeping hand inspects
  private stats: PoolStats = { hits: 0, misses: 0, evictions: 0, writebacks: 0, evictFailures: 0 };

  constructor(
    private readonly disk: Disk,
    private readonly frameCount: number,
    private readonly policy: Policy,
  ) {
    invariant(frameCount > 0, "buffer pool needs at least one frame");
  }

  /** Get a page, pinning its frame. On a miss this triggers exactly one
   *  disk.readPage (and possibly one writeback of a dirty victim). The returned
   *  bytes are the frame's live buffer — pinning guarantees they won't be stolen
   *  until unpin. Throws BufferFull if a miss needs a victim but all frames are
   *  pinned. */
  getPage(pageId: number): Uint8Array {
    const existing = this.byPage.get(pageId);
    if (existing !== undefined) {
      this.stats.hits++;
      const frame = this.frames[existing];
      frame.pinCount++;
      this.touch(existing);
      return frame.bytes;
    }

    this.stats.misses++;
    const slot = this.acquireSlot(); // may evict; may throw BufferFull
    // The ONE unavoidable IO of a miss. Everything in this file exists to make
    // this call rarer.
    const bytes = this.disk.readPage(pageId);
    const frame: Frame = { pageId, bytes, pinCount: 1, dirty: false, refBit: true };
    this.frames[slot] = frame;
    this.byPage.set(pageId, slot);
    this.touch(slot);
    return bytes;
  }

  /** Release one pin. Marking dirty here (not on getPage) reflects the real
   *  contract: a reader pins read-only, a writer pins then declares the write. */
  unpin(pageId: number, markDirty: boolean): void {
    const slot = this.byPage.get(pageId);
    invariant(slot !== undefined, `unpin of page ${pageId} not in pool — double unpin or stale id`);
    const frame = this.frames[slot];
    invariant(frame.pinCount > 0, `unpin of page ${pageId} with pinCount already 0`);
    frame.pinCount--;
    if (markDirty) frame.dirty = true;
  }

  poolStats(): Readonly<PoolStats> {
    return this.stats;
  }

  /** Find or make a free frame slot. Order matters: fill empty slots first (cold
   *  start has no victims), only then run the policy to evict. */
  private acquireSlot(): number {
    if (this.frames.length < this.frameCount) {
      const slot = this.frames.length;
      // Placeholder frame so the array stays dense; getPage overwrites it.
      this.frames.push({ pageId: -1, bytes: new Uint8Array(0), pinCount: 0, dirty: false, refBit: false });
      return slot;
    }
    const victim = this.policy === "LRU" ? this.chooseLruVictim() : this.chooseClockVictim();
    this.evict(victim);
    return victim;
  }

  /** LRU victim = the least-recently-used UNPINNED frame. We walk lruOrder from
   *  the front (oldest) and skip pinned frames. If every frame is pinned the loop
   *  exhausts and we throw — that is the BufferFull deadlock precursor. */
  private chooseLruVictim(): number {
    for (const slot of this.lruOrder) {
      if (this.frames[slot].pinCount === 0) return slot;
    }
    this.stats.evictFailures++;
    throw new BufferFull(this.frameCount);
  }

  /** CLOCK victim: the second-chance approximation of LRU. The hand sweeps frames;
   *  a frame with refBit=1 gets it cleared and a reprieve, a frame with refBit=0
   *  and no pin is evicted. Pinned frames are skipped entirely. We bound the sweep
   *  to 2*N steps: one full lap to clear ref bits, a second to find a now-zero
   *  frame. If 2 laps find only pinned frames, every frame is pinned -> BufferFull. */
  private chooseClockVictim(): number {
    const maxSteps = this.frameCount * 2;
    for (let step = 0; step < maxSteps; step++) {
      const slot = this.clockHand;
      this.clockHand = (this.clockHand + 1) % this.frameCount;
      const frame = this.frames[slot];
      if (frame.pinCount > 0) continue; // never evict a pinned frame
      if (frame.refBit) {
        frame.refBit = false; // second chance: survive this lap, die next if untouched
        continue;
      }
      return slot;
    }
    this.stats.evictFailures++;
    throw new BufferFull(this.frameCount);
  }

  /** Evict a frame: flush if dirty (one real writePage), then drop the mapping.
   *  Writeback-on-evict (not write-through) is why a hot dirty page costs zero IO
   *  until it finally leaves the cache. */
  private evict(slot: number): void {
    const frame = this.frames[slot];
    invariant(frame.pinCount === 0, `tried to evict pinned page ${frame.pageId}`);
    if (frame.dirty) {
      this.disk.writePage(frame.pageId, frame.bytes);
      this.stats.writebacks++;
    }
    this.byPage.delete(frame.pageId);
    this.lruOrder = this.lruOrder.filter((s) => s !== slot);
    this.stats.evictions++;
  }

  /** Record an access for the active policy. LRU moves the slot to the back
   *  (most-recently-used); CLOCK just raises the ref bit. Keeping both behind one
   *  method means getPage/hit paths don't branch on policy. */
  private touch(slot: number): void {
    if (this.policy === "LRU") {
      this.lruOrder = this.lruOrder.filter((s) => s !== slot);
      this.lruOrder.push(slot);
    } else {
      this.frames[slot].refBit = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Workload generation. Deterministic given the seed; these are the access traces
// the policies are scored on.
// ---------------------------------------------------------------------------

/** Zipf-skewed page picks: a few "hot" pages get most of the accesses, the long
 *  tail is rarely touched — the classic OLTP shape where caching pays off. We
 *  build a cumulative weight table with weight(rank) = 1/(rank+1) (Zipf s=1) and
 *  draw via the seeded float. Why precompute the CDF: it makes each draw O(log n)
 *  and, more importantly, keeps the draw a pure function of one rng.nextFloat() so
 *  the trace is reproducible. */
function makeZipfTrace(rng: Rng, pageCount: number, requests: number): number[] {
  const weights: number[] = [];
  let total = 0;
  for (let rank = 0; rank < pageCount; rank++) {
    total += 1 / (rank + 1);
    weights.push(total);
  }
  const trace: number[] = [];
  for (let i = 0; i < requests; i++) {
    const target = rng.nextFloat() * total;
    // Linear scan is fine at this toy scale and keeps the demo readable; the
    // determinism (not the speed) is what matters for the trace.
    let pageId = 0;
    while (pageId < pageCount - 1 && weights[pageId] < target) pageId++;
    trace.push(pageId);
  }
  return trace;
}

/** A full-table sequential scan repeated until `requests` accesses: 0,1,2,...,N-1,
 *  0,1,... This is LRU's adversary — by the time the scan wraps to page 0, LRU has
 *  evicted it to make room for the tail, so every single access misses if the
 *  table is larger than the pool. The seed is unused (a scan is not random) but we
 *  take rng to keep every workload's signature uniform. */
function makeScanTrace(_rng: Rng, pageCount: number, requests: number): number[] {
  const trace: number[] = [];
  for (let i = 0; i < requests; i++) trace.push(i % pageCount);
  return trace;
}

/** Run a trace through a fresh pool, counting disk reads attributable ONLY to
 *  this run (resetStats first). Each access is a pin-immediately-unpin: a
 *  read-only point lookup that holds the page just long enough to read it. */
function runTrace(disk: Disk, frameCount: number, policy: Policy, trace: number[]): {
  pool: Readonly<PoolStats>;
  diskReads: number;
} {
  disk.resetStats();
  const pool = new BufferPool(disk, frameCount, policy);
  for (const pageId of trace) {
    pool.getPage(pageId);
    pool.unpin(pageId, false); // read-only access; release immediately
  }
  return { pool: pool.poolStats(), diskReads: disk.stats().reads };
}

// ---------------------------------------------------------------------------
// Demo driver.
// ---------------------------------------------------------------------------

const SEED = 42;
const TABLE_PAGES = 64; // logical table size
const POOL_FRAMES = 8; // cache holds 1/8 of the table — eviction is forced
const REQUESTS = 2000; // accesses per workload

function main(): void {
  const disk = new Disk({ initialPages: TABLE_PAGES });
  // Materialize the table: allocate every page so the trace can read real pages.
  // These allocs are pre-warm IO; runTrace resets the disk counter before measuring.
  const blank = new Uint8Array(PAGE_SIZE);
  for (let i = 0; i < TABLE_PAGES; i++) {
    const id = disk.allocPage();
    disk.writePage(id, blank);
  }

  console.log("=== stage02: buffer pool — LRU vs CLOCK under Zipf vs scan ===");
  console.log(
    `table=${TABLE_PAGES} pages, pool=${POOL_FRAMES} frames (${((POOL_FRAMES / TABLE_PAGES) * 100).toFixed(0)}% of table), ` +
      `${REQUESTS} accesses/workload, seed=${SEED}`,
  );
  console.log("(hit rate + disk reads are EXACT deterministic counts; absolute values are toy-scale, the LRU-vs-CLOCK relative story is what transfers)\n");

  const rng = createRng(SEED);
  const zipf = makeZipfTrace(rng, TABLE_PAGES, REQUESTS);
  const scan = makeScanTrace(rng, TABLE_PAGES, REQUESTS);

  const matrix: Array<{ workload: string; policy: Policy; trace: number[] }> = [
    { workload: "zipf-hot", policy: "LRU", trace: zipf },
    { workload: "zipf-hot", policy: "CLOCK", trace: zipf },
    { workload: "full-scan", policy: "LRU", trace: scan },
    { workload: "full-scan", policy: "CLOCK", trace: scan },
  ];

  const rows = matrix.map(({ workload, policy, trace }) => {
    const { pool, diskReads } = runTrace(disk, POOL_FRAMES, policy, trace);
    const total = pool.hits + pool.misses;
    // Sanity: the cache can avoid IO only on hits, so disk reads must equal misses.
    // If this assert ever fires, the pool leaked an uncounted read — the metric
    // would be a lie, exactly what this book refuses to print.
    assertEq(diskReads, pool.misses, `disk reads must equal misses for ${workload}/${policy}`);
    return {
      workload,
      policy,
      accesses: total,
      hits: pool.hits,
      misses: pool.misses,
      "hitRate%": Number(((pool.hits / total) * 100).toFixed(1)),
      diskReads,
      evictions: pool.evictions,
    };
  });
  printTable(rows);

  console.log("\nRead the table:");
  console.log("  - Zipf: both policies cache the hot pages; CLOCK tracks LRU closely at lower bookkeeping cost.");
  console.log("  - Full-scan: the scan is larger than the pool, so a page is always evicted before the next lap reuses it.");
  console.log("    This is the textbook 'sequential flooding' — LRU's worst case; hit rate floors out near 0.");

  demoBufferFull();
}

/** Failure mode: pin every frame, then ask for one more page. With no unpinned
 *  victim the policy cannot make room and throws BufferFull. In a real engine the
 *  requester would BLOCK here — and if it is itself holding a pin another waiter
 *  needs, that is a deadlock. The takeaway the exception makes concrete: the
 *  buffer pool's liveness depends entirely on callers unpinning promptly. */
function demoBufferFull(): void {
  console.log("\n=== failure mode: all frames pinned -> BufferFull (deadlock precursor) ===");
  const frames = 4;
  const disk = new Disk({ initialPages: frames + 1 });
  const blank = new Uint8Array(PAGE_SIZE);
  for (let i = 0; i < frames + 1; i++) {
    const id = disk.allocPage();
    disk.writePage(id, blank);
  }

  const pool = new BufferPool(disk, frames, "LRU");
  // Pin all `frames` distinct pages and HOLD them (no unpin). Models a caller that
  // grabbed pages and hasn't released them — e.g. a long-running cursor or a bug.
  for (let pageId = 0; pageId < frames; pageId++) pool.getPage(pageId);
  console.log(`pinned all ${frames} frames (pages 0..${frames - 1}), none unpinned`);

  try {
    pool.getPage(frames); // page 4 has no frame and no evictable victim
    console.log("BUG: expected BufferFull but getPage succeeded");
  } catch (err) {
    invariant(err instanceof BufferFull, "expected BufferFull, got something else");
    console.log(`caught as expected: ${err.message}`);
    console.log(`pool recorded evictFailures=${pool.poolStats().evictFailures} (each = a scan that found every frame pinned)`);
  }

  // Recovery: unpin one frame, then the same request succeeds — proving the pool
  // is not broken, it was correctly refusing to violate the pin invariant.
  pool.unpin(0, false);
  pool.getPage(frames);
  console.log("after unpinning page 0, the same getPage(4) succeeds — liveness restored by prompt unpin");
}

main();
