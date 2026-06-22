// core/disk.ts — an in-memory page-addressed "disk" with honest IO counters.
//
// Why simulate a disk at all: the entire book is about minimizing IO. If we
// stored rows in a JS array, every interesting number (buffer-pool hit rate,
// B+tree fanout vs height, WAL fsync cost) would evaporate. So we model the one
// thing real storage forces on us: data lives in fixed-size pages, you address
// them by id, and the only way to read/write is whole-page IO. RAM-backed so the
// book stays offline and deterministic, but every readPage/writePage/fsync is
// COUNTED — those counters are the book's primary measured quantity.
//
// Invariants:
//  - PAGE_SIZE is fixed at 4096 forever. Pages are the atom; nothing crosses a
//    page boundary except via explicit overflow chains (see core/page.ts).
//  - A page id is a dense 0-based index. allocPage hands out the next one; we
//    never reuse ids in core (free-list management is a stage exercise).
//  - writePage copies the caller's bytes IN. The buffer pool depends on this:
//    after writePage, the caller may mutate its own buffer without corrupting
//    the page. Aliasing here would be a silent heisenbug.
//
// Failure mode we deliberately model: power loss. fsync is the durability
// boundary. `crashAt: N` makes the N-th fsync throw CrashError AFTER the
// preceding writes have been applied to the backing store — modeling the real
// guarantee "data before the last successful fsync survives, data after is gone".
// The recovery chapter (stage07) replays the WAL against exactly this.

export const PAGE_SIZE = 4096;

/** Thrown by fsync when a simulated power loss is injected. Distinct type so
 *  recovery code can catch *only* crashes, not real bugs. */
export class CrashError extends Error {
  constructor(public readonly fsyncOrdinal: number) {
    super(`simulated crash at fsync #${fsyncOrdinal}`);
    this.name = "CrashError";
  }
}

export interface DiskStats {
  reads: number;
  writes: number;
  fsyncs: number;
  pages: number;
}

export interface DiskOptions {
  /** Initial capacity in pages; grows automatically when exceeded. */
  initialPages?: number;
  /** If set, the (1-based) fsync ordinal at which fsync throws CrashError.
   *  Writes issued before that fsync are already durable; writes after are not. */
  crashAt?: number;
}

export class Disk {
  // Single ArrayBuffer backing store. Real disks are one big linear address
  // space too; modeling it as one buffer (not an array of page buffers) keeps
  // the "pages are just offsets" mental model honest and makes growth a memcpy.
  private buf: ArrayBuffer;
  private bytes: Uint8Array;
  private pageCount = 0;
  private capacityPages: number;

  private reads = 0;
  private writes = 0;
  private fsyncs = 0;
  private crashAt: number | undefined;

  // Durability frontier: bytes that a successful fsync has committed. On crash
  // we model "RAM lost" by exposing only this snapshot to a reopened Disk in the
  // recovery chapter. Kept here so the simulation is self-contained.
  private durable: Uint8Array;

  constructor(opts: DiskOptions = {}) {
    this.capacityPages = Math.max(1, opts.initialPages ?? 16);
    this.buf = new ArrayBuffer(this.capacityPages * PAGE_SIZE);
    this.bytes = new Uint8Array(this.buf);
    this.durable = new Uint8Array(this.capacityPages * PAGE_SIZE);
    this.crashAt = opts.crashAt;
  }

  /** Allocate a fresh zero-filled page and return its id. Grows the store if
   *  needed. We zero on alloc so readers never see another page's stale bytes —
   *  uninitialized-page reads are a classic source of phantom rows. */
  allocPage(): number {
    if (this.pageCount >= this.capacityPages) this.grow();
    const id = this.pageCount++;
    // Already zero (ArrayBuffer is zero-init, and grow copies preserve that for
    // the new region), but be explicit so reuse-during-tests can't leak data.
    this.bytes.fill(0, id * PAGE_SIZE, (id + 1) * PAGE_SIZE);
    return id;
  }

  /** Read a whole page. Returns a COPY so callers can't mutate the store behind
   *  the buffer pool's back; the copy also makes the read counter meaningful
   *  (one logical IO == one returned page). */
  readPage(pageId: number): Uint8Array {
    this.checkId(pageId);
    this.reads++;
    return this.bytes.slice(pageId * PAGE_SIZE, (pageId + 1) * PAGE_SIZE);
  }

  /** Write a whole page. The buffer must be exactly PAGE_SIZE; a short write
   *  would leave a torn page, which is precisely the corruption we refuse to
   *  allow silently. Bytes are copied IN (see header invariant). */
  writePage(pageId: number, buf: Uint8Array): void {
    this.checkId(pageId);
    if (buf.length !== PAGE_SIZE) {
      throw new Error(`writePage: expected ${PAGE_SIZE} bytes, got ${buf.length}`);
    }
    this.writes++;
    this.bytes.set(buf, pageId * PAGE_SIZE);
  }

  /** The durability boundary. Real fsync flushes OS/disk caches; here it copies
   *  the live bytes into the `durable` snapshot — UNLESS a crash is injected, in
   *  which case the writes preceding this fsync are still made durable (the OS
   *  had them) but the call throws to simulate the machine dying mid-sync. */
  fsync(): void {
    this.fsyncs++;
    if (this.crashAt !== undefined && this.fsyncs === this.crashAt) {
      // Commit what we had, then "lose power". Models: the last fsync's data may
      // or may not have hit the platter — we choose the pessimistic boundary
      // where THIS fsync did not complete, so recovery must not assume it did.
      throw new CrashError(this.fsyncs);
    }
    this.durable.set(this.bytes.subarray(0, this.pageCount * PAGE_SIZE));
  }

  /** Snapshot of the durable bytes as a fresh Disk-shaped view. Used by the
   *  recovery chapter to "reopen the file after a crash": only fsync'd data is
   *  visible, everything in the volatile buffer since the last fsync is gone. */
  reopenFromDurable(): Disk {
    const reopened = new Disk({ initialPages: this.capacityPages });
    reopened.pageCount = this.pageCount;
    reopened.bytes.set(this.durable.subarray(0, this.pageCount * PAGE_SIZE));
    reopened.durable.set(this.durable.subarray(0, this.pageCount * PAGE_SIZE));
    // Reset counters: the reopened handle measures recovery IO, not pre-crash IO.
    reopened.reads = 0;
    reopened.writes = 0;
    reopened.fsyncs = 0;
    return reopened;
  }

  stats(): DiskStats {
    return { reads: this.reads, writes: this.writes, fsyncs: this.fsyncs, pages: this.pageCount };
  }

  /** Reset only the counters (not data). Lets a stage measure a single phase
   *  (e.g. "reads during point lookups") without pre-warm IO polluting the number. */
  resetStats(): void {
    this.reads = 0;
    this.writes = 0;
    this.fsyncs = 0;
  }

  get numPages(): number {
    return this.pageCount;
  }

  private checkId(pageId: number): void {
    // Out-of-range page access is a structural bug (dangling pointer in an
    // index). Throw loudly rather than read zeros, which would masquerade as an
    // empty-but-valid page.
    if (pageId < 0 || pageId >= this.pageCount) {
      throw new Error(`page ${pageId} out of range [0, ${this.pageCount})`);
    }
  }

  private grow(): void {
    const newCap = this.capacityPages * 2;
    const newBuf = new ArrayBuffer(newCap * PAGE_SIZE);
    const newBytes = new Uint8Array(newBuf);
    newBytes.set(this.bytes);
    const newDurable = new Uint8Array(newCap * PAGE_SIZE);
    newDurable.set(this.durable);
    this.buf = newBuf;
    this.bytes = newBytes;
    this.durable = newDurable;
    this.capacityPages = newCap;
  }
}
