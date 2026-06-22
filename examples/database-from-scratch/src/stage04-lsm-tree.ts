// stage04-lsm-tree.ts — turning random writes into sequential writes, and paying
// for it on the read side.
//
// The thesis of the LSM (Log-Structured Merge) tree: a B+tree updates rows IN
// PLACE, so a workload of random-key writes scatters random-page writes across
// the disk. An LSM instead BUFFERS writes in a sorted in-memory table (memtable),
// then FLUSHES it as one immutable, sorted, append-only run (an SSTable). Every
// physical write is now sequential. The bill arrives at read time: a key may live
// in any of N SSTables across several levels, so a single get may have to probe
// many runs. This stage builds the real machinery and MEASURES that trade:
//
//   write amplification = bytes physically written to disk / logical bytes put
//   read amplification  = SSTables probed per get (averaged over a read workload)
//   bloom filter        = a per-SSTable bit-set that answers "key DEFINITELY not
//                         here" cheaply, so most gets skip most runs without IO
//   compaction          = the background merge that bounds read amplification by
//                         collapsing many small runs into fewer bigger ones — and
//                         is itself the dominant source of write amplification
//
// Why these four numbers together: they are the LSM's central tension. Bloom
// filters and compaction both attack read amplification; compaction trades MORE
// write amplification to BUY less read amplification; bloom filters buy read wins
// for almost free. The failure-mode demo at the end disables the bloom filter and
// re-measures, quantifying the "no bloom filter" read-amplification disaster.
//
// Honesty boundaries for the printed numbers:
//  - Disk read/write COUNTS and BYTES are real: every SSTable byte goes through
//    the core Disk, which counts physical page IO. Write amplification is computed
//    from those counted bytes, not estimated.
//  - SSTables-probed-per-get and bloom-filter-saved-probes are real counters
//    incremented on the actual get path.
//  - Wall-clock get latency (the bloom on/off comparison) is genuinely measured
//    with the core bench()/hrtime; it is machine-dependent and labeled measured.
//  - Absolute throughput is optimistic: this is a RAM-backed disk with zero seek
//    latency, so the real-world penalty of probing many runs (each a random seek
//    on a spinning disk or an SSD page fault) is UNDER-represented here. What
//    transfers is the RELATIVE trend and the probe-count ratio, not the ns.
//
// Determinism: the entire workload (keys, value sizes, deletes, the read sample)
// is driven by one seeded PRNG, so every count below is byte-identical run to run.

import { createRng, type Rng } from "./core/prng.js";
import { Disk, PAGE_SIZE } from "./core/disk.js";
import { PageWriter, PageReader, PageType, SLOTTED_HEADER } from "./core/page.js";
import { encodeKey, compareKeys, type Value } from "./core/codec.js";
import { bench } from "./core/clock.js";
import { assertSorted, invariant, printTable } from "./core/assert.js";

// ---------------------------------------------------------------------------
// Workload parameters. Chosen so that with 50_000 puts we get several memtable
// flushes (=> multiple L0 runs) and at least one round of compaction into L1 —
// otherwise "compaction triggered N times" would print 0 and prove nothing.
// ---------------------------------------------------------------------------
const TOTAL_PUTS = 50_000;
const KEY_SPACE = 12_000; // < TOTAL_PUTS so ~75% of puts overwrite an existing key
const MEMTABLE_FLUSH_THRESHOLD = 1_000; // entries; small => many runs => visible compaction
const L0_COMPACTION_TRIGGER = 8; // L0 runs allowed before merging down (more runs => higher read amp, the cost we want to show)
const DELETE_RATIO = 0.08; // fraction of puts that are deletes (tombstones)
const SEED = 0x15a3_2bce;

// Single-column integer key; value is an opaque blob whose length we vary so that
// "logical bytes" is a meaningful denominator for write amplification.
const KEY_TYPE = "int" as const;

// A tombstone is a real LSM concept: deletes can't reach into older immutable
// SSTables to erase a key, so a delete is recorded as a marker that SHADOWS any
// older value for that key. It only truly disappears when compaction merges past
// the run that holds it. We model the value payload as a length-prefixed blob and
// reserve length 0xffff_ffff (sentinel) for "this entry is a tombstone".
const TOMBSTONE = Symbol("tombstone");
type Payload = Uint8Array | typeof TOMBSTONE;

interface Entry {
  key: Uint8Array; // encoded key bytes (sorted by compareKeys)
  payload: Payload;
}

// ---------------------------------------------------------------------------
// Bloom filter — a probabilistic set membership test that NEVER says "absent"
// for a present key (no false negatives) but MAY say "present" for an absent key
// (false positives). That asymmetry is exactly what an LSM read path needs: a
// "definitely not here" lets us skip an SSTable's IO with zero risk of missing a
// live key. We build one per SSTable from its keys at flush time.
//
// Sizing: m bits, k hashes. We pick m from the entry count at a target ~1% false
// positive rate (m/n ≈ 9.6 bits, k ≈ 7 by the standard formulas). We derive k
// independent hashes from TWO base hashes via the Kirsch-Mitzenmacher technique
// h_i = h1 + i*h2 — cheaper than k real hashes and statistically adequate here.
// ---------------------------------------------------------------------------
class BloomFilter {
  private readonly bits: Uint8Array;
  private readonly numBits: number;
  private readonly numHashes: number;

  constructor(expectedEntries: number, bitsPerEntry = 10, numHashes = 7) {
    // At least 1 byte so an empty SSTable's filter is still well-formed.
    this.numBits = Math.max(8, expectedEntries * bitsPerEntry);
    this.numHashes = numHashes;
    this.bits = new Uint8Array(Math.ceil(this.numBits / 8));
  }

  /** FNV-1a over the key bytes, returned as two 32-bit base hashes. We seed the
   *  second hash differently so h1 and h2 are independent enough for the
   *  double-hashing scheme; using the same hash twice would collapse k to 1. */
  private hashPair(key: Uint8Array): [number, number] {
    let h1 = 0x811c9dc5; // FNV offset basis
    let h2 = 0x01000193; // FNV prime as an alternate seed
    for (let i = 0; i < key.length; i++) {
      h1 = (Math.imul(h1 ^ key[i], 0x01000193)) >>> 0;
      h2 = (Math.imul(h2 ^ key[i], 0x85ebca6b)) >>> 0; // murmur-ish mix constant
    }
    return [h1 >>> 0, h2 >>> 0];
  }

  add(key: Uint8Array): void {
    const [h1, h2] = this.hashPair(key);
    for (let i = 0; i < this.numHashes; i++) {
      // >>> 0 keeps the combined hash unsigned before the modulo; a signed value
      // would index negative and silently miss bits, inflating false positives.
      const bit = ((h1 + Math.imul(i, h2)) >>> 0) % this.numBits;
      this.bits[bit >>> 3] |= 1 << (bit & 7);
    }
  }

  /** True => key MIGHT be present (probe the SSTable). False => key is GUARANTEED
   *  absent (skip the SSTable). The whole read-amplification win rides on how
   *  often this returns false for keys that aren't in this particular run. */
  mightContain(key: Uint8Array): boolean {
    const [h1, h2] = this.hashPair(key);
    for (let i = 0; i < this.numHashes; i++) {
      const bit = ((h1 + Math.imul(i, h2)) >>> 0) % this.numBits;
      if ((this.bits[bit >>> 3] & (1 << (bit & 7))) === 0) return false;
    }
    return true;
  }
}

// ---------------------------------------------------------------------------
// SSTable — an immutable, key-sorted run persisted as a chain of disk pages.
//
// On-disk layout per page: a slotted page (reusing the core slotted-page header)
// whose slots hold encoded entries laid out in ascending key order. Each entry is
// [keyLen:u16][key bytes][valLen:u32][val bytes], with valLen == TOMBSTONE_LEN
// meaning "tombstone, no value bytes follow". Entries never cross a page boundary;
// when a page fills we start a new one. Because the run is sorted and immutable,
// a get does a linear scan of the loaded pages (a real engine would binary-search
// a per-page index; we keep the scan to keep the IO model legible — the page
// COUNT, which drives read amplification, is identical either way).
//
// Why persist at all instead of keeping a JS array: write amplification is
// "bytes physically written", and that is only real if the bytes actually pass
// through the counted Disk. An in-memory SSTable would make the headline number
// fictional, which this book forbids.
// ---------------------------------------------------------------------------
const TOMBSTONE_LEN = 0xffff_ffff; // u32 sentinel in valLen meaning "tombstone"

interface SSTableMeta {
  level: number;
  pageIds: number[]; // pages holding this run, in key order
  minKey: Uint8Array;
  maxKey: Uint8Array;
  entryCount: number;
  byteLength: number; // physical bytes occupied (pages * PAGE_SIZE)
  bloom: BloomFilter;
}

/** Serialize a sorted entry list into fresh disk pages and return run metadata.
 *  Precondition: `entries` is sorted ascending by key and de-duplicated (one
 *  entry per key) — we assert the sort so a broken merge is caught here, not as a
 *  wrong query a thousand lines later. */
function flushEntriesToSSTable(disk: Disk, level: number, entries: Entry[]): SSTableMeta {
  assertSorted(entries.map((e) => e.key), `SSTable build at L${level}`);

  const bloom = new BloomFilter(entries.length);
  for (const e of entries) bloom.add(e.key);

  const pageIds: number[] = [];
  let page = new Uint8Array(PAGE_SIZE);
  let writer = new PageWriter(page);
  writer.initSlotted(PageType.LEAF);
  let pageId = disk.allocPage();
  let off: number = SLOTTED_HEADER.HEADER_SIZE; // append point within the current page

  const flushPage = () => {
    disk.writePage(pageId, page);
    pageIds.push(pageId);
  };

  for (const e of entries) {
    const valBytes = e.payload === TOMBSTONE ? new Uint8Array(0) : e.payload;
    const valLen = e.payload === TOMBSTONE ? TOMBSTONE_LEN : valBytes.length;
    // 2 (keyLen) + key + 4 (valLen) + value. We size-check against the page so a
    // record that wouldn't fit triggers a fresh page rather than a bounds throw.
    const recordSize = 2 + e.key.length + 4 + valBytes.length;
    invariant(recordSize <= PAGE_SIZE - SLOTTED_HEADER.HEADER_SIZE, "single record exceeds a page");

    if (off + recordSize > PAGE_SIZE) {
      // Current page is full: persist it and start a new one. This is the
      // sequential-write pattern — we only ever append, never seek back.
      flushPage();
      page = new Uint8Array(PAGE_SIZE);
      writer = new PageWriter(page);
      writer.initSlotted(PageType.LEAF);
      pageId = disk.allocPage();
      off = SLOTTED_HEADER.HEADER_SIZE;
    }

    writer.writeU16(off, e.key.length);
    off += 2;
    off = writer.writeBytes(off, e.key);
    writer.writeU32(off, valLen);
    off += 4;
    if (valBytes.length > 0) off = writer.writeBytes(off, valBytes);
  }
  flushPage(); // the final (possibly partial) page

  return {
    level,
    pageIds,
    minKey: entries[0].key,
    maxKey: entries[entries.length - 1].key,
    entryCount: entries.length,
    byteLength: pageIds.length * PAGE_SIZE,
    bloom,
  };
}

/** Linear-scan one persisted SSTable for an exact key. Returns the payload if
 *  found (possibly TOMBSTONE), or undefined if this run simply doesn't hold it.
 *  Reads whole pages through the counted Disk so the read counter stays honest.
 *  Returns the number of pages actually read so callers can attribute IO. */
function scanSSTable(disk: Disk, meta: SSTableMeta, key: Uint8Array): { payload?: Payload; pagesRead: number } {
  let pagesRead = 0;
  for (const pid of meta.pageIds) {
    const page = disk.readPage(pid);
    pagesRead++;
    const reader = new PageReader(page);
    let off: number = SLOTTED_HEADER.HEADER_SIZE;
    // Scan to the page's free frontier. We persisted entries contiguously from
    // HEADER_SIZE; an all-zero keyLen at the frontier means "no more records".
    // Guard against a sub-2-byte tail: a record header needs 2 bytes for keyLen,
    // so an offset that close to the page end is unwritten padding, not a record.
    for (;;) {
      if (off + 2 > PAGE_SIZE) break;
      const keyLen = reader.readU16(off);
      if (keyLen === 0) break; // reached unwritten tail of the page
      const recKey = reader.readBytes(off + 2, keyLen);
      const valLen = reader.readU32(off + 2 + keyLen);
      const cmp = compareKeys(recKey, key);
      if (cmp === 0) {
        if (valLen === TOMBSTONE_LEN) return { payload: TOMBSTONE, pagesRead };
        const val = reader.readBytes(off + 2 + keyLen + 4, valLen);
        // Copy out: readBytes is a view into the page buffer, which the caller may
        // outlive; retaining the view would alias a recycled page.
        return { payload: val.slice(), pagesRead };
      }
      // Run is sorted ascending: once we pass the target key, it isn't here, and
      // since pages are also in key order, no later page can hold it either.
      if (cmp > 0) return { pagesRead };
      const valBytesLen = valLen === TOMBSTONE_LEN ? 0 : valLen;
      off += 2 + keyLen + 4 + valBytesLen;
      if (off >= PAGE_SIZE) break;
    }
  }
  return { pagesRead };
}

/** Read every live entry of an SSTable back into memory, in key order. Used by
 *  compaction, which must re-read the runs it merges (that re-read IS the read
 *  half of compaction's cost). */
function readAllEntries(disk: Disk, meta: SSTableMeta): Entry[] {
  const out: Entry[] = [];
  for (const pid of meta.pageIds) {
    const page = disk.readPage(pid);
    const reader = new PageReader(page);
    let off: number = SLOTTED_HEADER.HEADER_SIZE;
    for (;;) {
      if (off + 2 > PAGE_SIZE) break; // sub-header tail padding, no record here
      const keyLen = reader.readU16(off);
      if (keyLen === 0) break;
      const key = reader.readBytes(off + 2, keyLen).slice();
      const valLen = reader.readU32(off + 2 + keyLen);
      if (valLen === TOMBSTONE_LEN) {
        out.push({ key, payload: TOMBSTONE });
        off += 2 + keyLen + 4;
      } else {
        const val = reader.readBytes(off + 2 + keyLen + 4, valLen).slice();
        out.push({ key, payload: val });
        off += 2 + keyLen + 4 + valLen;
      }
      if (off >= PAGE_SIZE) break;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// k-way merge of several sorted runs into one. Newer runs WIN on a key tie (their
// value/tombstone shadows older copies), which is why merge order matters: we
// pass runs newest-first within a level boundary so the first-seen value for a
// key is the freshest. This is the heart of how an LSM reconciles many versions
// of a key scattered across runs into a single current value.
// ---------------------------------------------------------------------------
function mergeRuns(runs: Entry[][], dropTombstones: boolean): Entry[] {
  // Cursor per run. A simple linear-min merge: with a handful of runs this is
  // clearer than a heap and the constant factor is irrelevant to the measured
  // disk numbers (the heap would only matter for hundreds of runs).
  const cursors = new Array(runs.length).fill(0);
  const out: Entry[] = [];

  for (;;) {
    // Find the smallest key across all run heads; among equal keys, the run with
    // the LOWEST index wins (caller passes newest-first), so we keep that one and
    // advance ALL runs sharing that key — collapsing duplicate versions.
    let minKey: Uint8Array | undefined;
    let minRun = -1;
    for (let r = 0; r < runs.length; r++) {
      if (cursors[r] >= runs[r].length) continue;
      const head = runs[r][cursors[r]].key;
      if (minKey === undefined || compareKeys(head, minKey) < 0) {
        minKey = head;
        minRun = r;
      }
    }
    if (minRun === -1) break; // all runs exhausted

    const winner = runs[minRun][cursors[minRun]];
    // Advance every run whose head equals the winning key; the winner (lowest
    // run index, i.e. newest) is the version we keep.
    for (let r = 0; r < runs.length; r++) {
      while (cursors[r] < runs[r].length && compareKeys(runs[r][cursors[r]].key, minKey!) === 0) {
        cursors[r]++;
      }
    }

    // Tombstone garbage collection: only safe to drop a tombstone when merging
    // into the BOTTOM-most level, where no older run below can still hold a live
    // value the tombstone is meant to shadow. dropTombstones encodes that.
    if (winner.payload === TOMBSTONE && dropTombstones) continue;
    out.push(winner);
  }
  return out;
}

// ---------------------------------------------------------------------------
// The LSM engine: a memtable, a stack of levels (each a list of SSTables), and
// the put/get/flush/compact machinery — plus the instrumentation counters that
// produce this stage's headline numbers.
// ---------------------------------------------------------------------------
interface LsmStats {
  logicalBytesPut: number; // sum of key+value bytes the user handed us (overwrites counted each time)
  flushes: number;
  compactions: number;
  getCount: number;
  ssTablesProbed: number; // sum over gets of SSTables actually scanned (post-bloom)
  bloomSkips: number; // SSTables skipped because bloom said "definitely absent"
}

class LsmTree {
  // memtable: key bytes (hex) -> entry. We key by hex string because Uint8Array
  // can't be a Map key by value; the hex round-trips losslessly and keeps the
  // memtable a true last-writer-wins map (a later put on the same key overwrites,
  // exactly as an in-memory write buffer should).
  private memtable = new Map<string, Entry>();
  // levels[0] is L0 (flushed runs, may have overlapping key ranges); deeper
  // levels hold progressively larger, non-overlapping (within level) runs.
  private levels: SSTableMeta[][] = [[]];
  readonly stats: LsmStats = {
    logicalBytesPut: 0,
    flushes: 0,
    compactions: 0,
    getCount: 0,
    ssTablesProbed: 0,
    bloomSkips: 0,
  };

  constructor(private readonly disk: Disk, private readonly bloomEnabled: boolean) {}

  put(keyValue: Value, value: Uint8Array): void {
    const key = encodeKey(keyValue, KEY_TYPE);
    this.stats.logicalBytesPut += key.length + value.length;
    this.memtable.set(hex(key), { key, payload: value });
    if (this.memtable.size >= MEMTABLE_FLUSH_THRESHOLD) this.flush();
  }

  delete(keyValue: Value): void {
    const key = encodeKey(keyValue, KEY_TYPE);
    // A delete is a logical write: it still costs key bytes (no value), and it
    // still occupies space in runs until compaction collapses it. We count the
    // key bytes so write amplification reflects that deletes aren't free.
    this.stats.logicalBytesPut += key.length;
    this.memtable.set(hex(key), { key, payload: TOMBSTONE });
    if (this.memtable.size >= MEMTABLE_FLUSH_THRESHOLD) this.flush();
  }

  /** get: probe newest-to-oldest. The FIRST run that holds the key wins, because
   *  newer runs shadow older ones. A tombstone hit means "deleted" => return
   *  undefined. The bloom filter lets us skip runs that definitely lack the key. */
  get(keyValue: Value): Uint8Array | undefined {
    const key = encodeKey(keyValue, KEY_TYPE);
    this.stats.getCount++;

    // 1) memtable is the newest data and is free to probe (no IO).
    const memHit = this.memtable.get(hex(key));
    if (memHit) return memHit.payload === TOMBSTONE ? undefined : memHit.payload;

    // 2) Then each level, newest level (L0) first, and within L0 newest run
    //    first (runs were pushed in flush order, so iterate in reverse).
    for (let lvl = 0; lvl < this.levels.length; lvl++) {
      const runs = this.levels[lvl];
      for (let i = runs.length - 1; i >= 0; i--) {
        const meta = runs[i];
        // Key-range prune is free and exact: if the key is outside [min,max] this
        // run cannot hold it. Real engines do this before even touching bloom.
        if (compareKeys(key, meta.minKey) < 0 || compareKeys(key, meta.maxKey) > 0) continue;
        // Bloom prune: probabilistic, but a "no" is certain. This is the line the
        // failure-mode demo disables to expose the read-amplification disaster.
        if (this.bloomEnabled && !meta.bloom.mightContain(key)) {
          this.stats.bloomSkips++;
          continue;
        }
        this.stats.ssTablesProbed++;
        const { payload } = scanSSTable(this.disk, meta, key);
        if (payload !== undefined) {
          return payload === TOMBSTONE ? undefined : payload;
        }
        // Bloom false positive (or a key-range-overlapping run that just lacks
        // this key): we paid for the scan and found nothing — keep going.
      }
    }
    return undefined;
  }

  /** Flush the memtable to a new L0 SSTable (sorted, immutable), then maybe
   *  compact. This is where random in-memory writes become one sequential run. */
  private flush(): void {
    if (this.memtable.size === 0) return;
    const entries = [...this.memtable.values()].sort((a, b) => compareKeys(a.key, b.key));
    const meta = flushEntriesToSSTable(this.disk, 0, entries);
    this.levels[0].push(meta);
    this.memtable.clear();
    this.stats.flushes++;
    this.maybeCompact();
  }

  /** Leveled compaction: when a level holds too many runs, merge the whole level
   *  (plus the next level's runs, since their ranges may overlap) into a single
   *  run one level down. This bounds read amplification (fewer runs to probe) at
   *  the cost of re-reading and re-writing data — the write-amplification engine.
   *
   *  Trigger: L0 uses a run-COUNT threshold (L0 runs overlap, so count is what
   *  hurts reads); deeper levels use a size threshold that grows by a fanout so
   *  the tree stays shallow. */
  private maybeCompact(): void {
    let lvl = 0;
    for (;;) {
      const runs = this.levels[lvl];
      const overLimit = lvl === 0
        ? runs.length > L0_COMPACTION_TRIGGER
        : runs.length > levelRunBudget(lvl);
      if (!overLimit) break;

      // Ensure the destination level exists.
      const nextLvl = lvl + 1;
      if (this.levels.length <= nextLvl) this.levels.push([]);

      // Read all runs of this level + the next level (their key ranges overlap),
      // newest-first so the merge keeps the freshest version of each key.
      const sourceMetas = [...this.levels[lvl], ...this.levels[nextLvl]];
      // newest-first ordering for the merge: within a level later-flushed runs are
      // newer; the current level is newer than the next level. Reverse each level
      // then concatenate current-before-next.
      const runsNewestFirst: Entry[][] = [
        ...[...this.levels[lvl]].reverse().map((m) => readAllEntries(this.disk, m)),
        ...[...this.levels[nextLvl]].reverse().map((m) => readAllEntries(this.disk, m)),
      ];

      // Drop tombstones only when merging into the deepest level (nothing below
      // can still need shadowing). Otherwise keep them to shadow older runs.
      const isBottom = nextLvl === this.levels.length - 1;
      const merged = mergeRuns(runsNewestFirst, isBottom);

      // Replace both levels' runs with the single merged run at nextLvl. (We do
      // not free the old pages — the core Disk has no free-list; reclaiming space
      // is a deliberate stage exercise, see core/disk.ts. The orphaned pages
      // still count as physical writes, which is honest: real LSMs also write the
      // new run before the old pages are reclaimable.)
      this.levels[lvl] = [];
      const mergedMeta = merged.length > 0
        ? flushEntriesToSSTable(this.disk, nextLvl, merged)
        : undefined;
      this.levels[nextLvl] = mergedMeta ? [mergedMeta] : [];
      this.stats.compactions++;
      void sourceMetas; // (kept for clarity that both levels were the input)

      // Cascade: the merge may have pushed nextLvl over its own budget.
      lvl = nextLvl;
    }
  }

  /** Average SSTables actually scanned per get (post-bloom, post-range-prune).
   *  THE read-amplification headline. */
  readAmplification(): number {
    return this.stats.getCount === 0 ? 0 : this.stats.ssTablesProbed / this.stats.getCount;
  }

  levelShape(): number[] {
    return this.levels.map((l) => l.length);
  }
}

/** Per-level run budget: level L tolerates ~10^L runs before compacting down.
 *  The fanout keeps the tree O(log n) deep, which bounds worst-case read
 *  amplification. (L0 is special-cased to a small count in maybeCompact.) */
function levelRunBudget(level: number): number {
  return Math.pow(10, level);
}

function hex(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += bytes[i].toString(16).padStart(2, "0");
  return s;
}

// ---------------------------------------------------------------------------
// Deterministic workload generation. Returns a fixed sequence of operations so
// both the bloom-on and bloom-off runs replay the IDENTICAL writes — only the
// read path differs, which is what makes the comparison fair.
// ---------------------------------------------------------------------------
type Op =
  | { kind: "put"; key: number; value: Uint8Array }
  | { kind: "delete"; key: number };

function buildWorkload(rng: Rng): Op[] {
  const ops: Op[] = [];
  for (let i = 0; i < TOTAL_PUTS; i++) {
    const key = rng.nextInt(0, KEY_SPACE);
    if (rng.nextFloat() < DELETE_RATIO) {
      ops.push({ kind: "delete", key });
    } else {
      // Variable value length 8..72 bytes; deterministic content so identical
      // across runs. Real rows vary; a fixed length would make write
      // amplification look artificially clean.
      const len = 8 + rng.nextInt(0, 64);
      const value = new Uint8Array(len);
      for (let b = 0; b < len; b++) value[b] = (key + b) & 0xff;
      ops.push({ kind: "put", key, value });
    }
  }
  return ops;
}

/** Replay a fixed workload into a fresh LSM and return the engine + a snapshot of
 *  the post-load disk write counters (so write amplification excludes read IO). */
function loadLsm(ops: Op[], bloomEnabled: boolean): { lsm: LsmTree; bytesWritten: number; pagesAllocated: number } {
  const disk = new Disk({ initialPages: 64 });
  const lsm = new LsmTree(disk, bloomEnabled);
  for (const op of ops) {
    if (op.kind === "put") lsm.put(op.key, op.value);
    else lsm.delete(op.key);
  }
  const s = disk.stats();
  // Physical bytes written = writePage calls * PAGE_SIZE. Every SSTable byte
  // (initial flushes + compaction rewrites) went through writePage, so this is
  // the real, counted physical write volume.
  return { lsm, bytesWritten: s.writes * PAGE_SIZE, pagesAllocated: s.pages };
}

// ---------------------------------------------------------------------------
// Main demo.
// ---------------------------------------------------------------------------
function main(): void {
  console.log("=== stage04: LSM 树 — 随机写变顺序写,以及读放大的代价 ===\n");

  const rng = createRng(SEED);
  const workload = buildWorkload(rng);
  const putCount = workload.filter((o) => o.kind === "put").length;
  const delCount = workload.length - putCount;
  console.log(
    `工作负载: ${workload.length} 次操作 (${putCount} put / ${delCount} delete), ` +
      `键空间 ${KEY_SPACE} (约 ${Math.round((1 - KEY_SPACE / putCount) * 100)}% 覆盖更新)\n`,
  );

  // ---- Build the LSM WITH bloom filters (the normal configuration). ----
  const withBloom = loadLsm(workload, true);
  const lsm = withBloom.lsm;
  console.log(`memtable 阈值=${MEMTABLE_FLUSH_THRESHOLD} 条, L0 触发合并=${L0_COMPACTION_TRIGGER} 个 run`);
  console.log(`flush 次数=${lsm.stats.flushes}, compaction 次数=${lsm.stats.compactions}`);
  console.log(`各层 run 数 [L0..]: [${lsm.levelShape().join(", ")}]\n`);

  // ---- Write amplification (computed from COUNTED physical bytes). ----
  const writeAmp = withBloom.bytesWritten / lsm.stats.logicalBytesPut;
  printTable([
    {
      metric: "写放大 (write amplification)",
      logical_bytes: lsm.stats.logicalBytesPut,
      physical_bytes: withBloom.bytesWritten,
      ratio_x: round(writeAmp, 2),
    },
  ]);
  console.log(
    "  解读: 物理写盘字节 / 逻辑 put 字节. >1 倍的部分是 flush 后的 compaction 重写代价.\n" +
      "  注: 4KB 页内有未填满空间, 这部分也计入物理字节 (真实 LSM 同样如此).\n",
  );

  // ---- Read workload: a deterministic sample of point lookups. We mix keys we
  //      know were written with keys outside the key space (guaranteed misses),
  //      because bloom filters earn their keep precisely on the MISS path. ----
  const readRng = createRng(SEED ^ 0x9e37_79b9);
  const READ_SAMPLE = 20_000;
  const lookups: number[] = [];
  for (let i = 0; i < READ_SAMPLE; i++) {
    // 70% hits (in key space), 30% guaranteed misses (above key space).
    if (readRng.nextFloat() < 0.7) lookups.push(readRng.nextInt(0, KEY_SPACE));
    else lookups.push(KEY_SPACE + readRng.nextInt(0, KEY_SPACE));
  }

  for (const k of lookups) lsm.get(k);

  printTable([
    {
      metric: "读放大 (bloom 开启)",
      gets: lsm.stats.getCount,
      sstables_probed: lsm.stats.ssTablesProbed,
      bloom_skips: lsm.stats.bloomSkips,
      probes_per_get: round(lsm.readAmplification(), 3),
    },
  ]);
  console.log(
    `  bloom 过滤掉的无效查表 = ${lsm.stats.bloomSkips} 次 ` +
      "(布隆说\"一定不在\", 直接跳过, 省掉这么多次 SSTable 扫描的 IO).\n",
  );

  // ---- Failure mode: rebuild the SAME data with bloom filters DISABLED, run
  //      the SAME read workload, and watch probes + wall-clock explode. ----
  console.log("--- 失败模式: 关闭布隆过滤器, 同样数据 + 同样查询 ---\n");
  const noBloom = loadLsm(workload, false).lsm;
  for (const k of lookups) noBloom.get(k);

  printTable([
    {
      config: "bloom 开启",
      sstables_probed: lsm.stats.ssTablesProbed,
      bloom_skips: lsm.stats.bloomSkips,
      probes_per_get: round(lsm.readAmplification(), 3),
    },
    {
      config: "bloom 关闭",
      sstables_probed: noBloom.stats.ssTablesProbed,
      bloom_skips: noBloom.stats.bloomSkips,
      probes_per_get: round(noBloom.readAmplification(), 3),
    },
  ]);

  const probeBlowup = noBloom.stats.ssTablesProbed / Math.max(1, lsm.stats.ssTablesProbed);
  console.log(`\n  查表次数膨胀 = ${round(probeBlowup, 1)}x (关闭布隆后多扫这么多倍 SSTable).`);

  // ---- Measured wall-clock get latency, bloom on vs off. Genuinely timed; the
  //      RAM-disk makes the absolute ns optimistic (no seek), so the RELATIVE
  //      slowdown is the transferable signal, not the nanoseconds themselves. ----
  // Use a fixed sub-sample replayed identically by both configs.
  const timedLookups = lookups.slice(0, 5_000);
  let idxOn = 0;
  const benchOn = bench(() => {
    lsm.get(timedLookups[idxOn % timedLookups.length]);
    idxOn++;
  }, timedLookups.length);
  let idxOff = 0;
  const benchOff = bench(() => {
    noBloom.get(timedLookups[idxOff % timedLookups.length]);
    idxOff++;
  }, timedLookups.length);

  printTable([
    { config: "bloom 开启", ns_per_get_measured: round(benchOn.nsPerOp, 1), gets_per_sec_measured: Math.round(benchOn.opsPerSec) },
    { config: "bloom 关闭", ns_per_get_measured: round(benchOff.nsPerOp, 1), gets_per_sec_measured: Math.round(benchOff.opsPerSec) },
  ]);
  console.log(
    `\n  延迟膨胀 = ${round(benchOff.nsPerOp / benchOn.nsPerOp, 2)}x (measured, 机器相关).\n` +
      "  注: 这是 RAM 盘, 无寻道延迟; 真实磁盘/SSD 上每次多扫一个 SSTable = 一次随机 IO,\n" +
      "  代价比这里大得多. 可迁移的是\"关掉布隆 → 读放大灾难\"这个相对趋势, 不是绝对 ns.\n",
  );

  // ---- Correctness spot-check: the LSM must answer gets consistently with a
  //      ground-truth replay of the workload (last-writer-wins per key). A wrong
  //      merge / tombstone bug would surface here as a mismatch, not as a silently
  //      plausible-but-wrong number above. ----
  verifyAgainstGroundTruth(workload, lsm);
  console.log("正确性校验: LSM get 结果与逐操作 ground-truth 全部一致. ✓");
}

/** Replay the workload into a plain Map (the authoritative last-writer-wins
 *  state) and assert the LSM returns the same value for a deterministic key
 *  sample. This is the executable proof that flush + compaction + tombstones did
 *  not corrupt or lose any value. */
function verifyAgainstGroundTruth(ops: Op[], lsm: LsmTree): void {
  const truth = new Map<number, Uint8Array | "deleted">();
  for (const op of ops) {
    if (op.kind === "put") truth.set(op.key, op.value);
    else truth.set(op.key, "deleted");
  }
  const checkRng = createRng(SEED ^ 0x1234_5678);
  for (let i = 0; i < 3_000; i++) {
    const key = checkRng.nextInt(0, KEY_SPACE);
    const expected = truth.get(key);
    const got = lsm.get(key);
    if (expected === undefined || expected === "deleted") {
      invariant(got === undefined, `key ${key}: expected absent/deleted, got a value`);
    } else {
      invariant(got !== undefined, `key ${key}: expected a value, got absent`);
      invariant(bytesEqual(got, expected), `key ${key}: value mismatch (merge/version bug)`);
    }
  }
}

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function round(n: number, digits: number): number {
  const f = Math.pow(10, digits);
  return Math.round(n * f) / f;
}

main();
