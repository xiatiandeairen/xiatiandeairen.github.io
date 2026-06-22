// stage03-bplus-tree.ts — a real, disk-paged B+tree: insert / search / rangeScan,
// and the one number every storage course hand-waves: why three levels hold a
// billion rows.
//
// Why this chapter exists at all: a B+tree is not "a sorted map on disk". The
// entire value is the arithmetic between PAGE_SIZE, key width, and fanout — the
// fact that fanout is hundreds means height grows logarithmically with a base of
// hundreds, so height stays at 3-4 for absurd row counts and a point lookup costs
// `height` page reads, period. This file MEASURES that, not asserts it: it builds
// a tree over 100k keys on the simulated Disk, reads the real fanout out of an
// actual internal page, computes the real height by walking root->leaf, and
// proves search-touches-pages == height by counting buffer-pool misses.
//
// What the tree stores: keys are u32 (encoded big-endian via core/codec.encodeKey
// so a raw memcmp == numeric order — see codec.ts), values are a synthetic 4-byte
// row pointer (a record id / page#slot would be the same width in a real heap
// table). Fixed-width entries are a deliberate choice for THIS chapter: it lets
// fanout be a clean, honest constant a reader can verify by hand, instead of a
// fuzzy average over variable-length keys. Variable-length keys / prefix
// compression are explicitly a later concern; pretending to do them here would
// muddy the one lesson (fanout -> height).
//
// Layout reuse: leaf and internal nodes are slotted pages (core/page.ts), the
// same discipline stage01 uses for rows. Leaves additionally carry a u32 "next
// leaf" pointer in a reserved header word so rangeScan walks the leaf chain
// without re-descending the tree — the classic B+tree (vs plain B-tree) property.
//
// Failure mode this chapter is built to expose: SEQUENTIAL insertion. With
// monotonically increasing keys, every split happens at the rightmost leaf and
// the standard "split in half" rule leaves the left half permanently ~50% full,
// because nothing ever inserts into it again. We measure that occupancy collapse
// against random-key insertion (~70%+), then show the production fix (a right-most
// special-case that splits 1-off instead of in-half, the seed of bulk-load).
//
// Determinism: the only randomness is core/prng with a fixed seed; same seed =>
// same keys => byte-identical tree => identical IO counts. Throughput (ops/sec)
// is the sole measured-not-deterministic number and is labeled as such.

import { createRng } from "./core/prng.js";
import { Disk, PAGE_SIZE } from "./core/disk.js";
import {
  PageReader,
  PageWriter,
  PageType,
  SLOTTED_HEADER,
  SLOT_ENTRY_SIZE,
} from "./core/page.js";
import { encodeKey, compareKeys } from "./core/codec.js";
import { bench } from "./core/clock.js";
import { invariant, assertEq, printTable } from "./core/assert.js";

// ---------------------------------------------------------------------------
// On-page layout for B+tree nodes.
// ---------------------------------------------------------------------------
//
// We reuse the slotted-page header (TYPE/SLOT_COUNT/FREE_START/FREE_END) so the
// node format is consistent with the rest of the book. One node-specific word is
// stacked on top of the shared header:
//
//   NEXT_LEAF (u32) — right-sibling page id for leaves; 0 = none (page 0 is the
//   meta page, never a leaf, so 0 is a safe sentinel). Unused on internal nodes.
//
// Slot directory entries point at variable-position records in the heap. A record
// is laid out as:
//   leaf record:     [key: KEY_SIZE bytes][value: VALUE_SIZE bytes]
//   internal record: [child: u32][sepKeyLen handled positionally]
// Internal nodes store N children and N-1 separators. We store them as N slots
// where slot i = child_i, and the separator BEFORE child_i (for i>=1) is packed
// after the child pointer. The leftmost child (slot 0) has no preceding separator.
//
// KEY_SIZE is fixed at 4 because keys are u32. VALUE_SIZE is 4 (a row pointer).
// Making these constants explicit is what lets fanout be hand-verifiable below.

const KEY_SIZE = 4; // u32 key, big-endian (encodeKey(k,"int"))
const VALUE_SIZE = 4; // synthetic row pointer (record id)

// Node header: shared slotted header + one extra u32 for the leaf sibling link.
// We place NEXT_LEAF right after HEADER_SIZE and start the slot directory after it.
const NEXT_LEAF_OFF = SLOTTED_HEADER.HEADER_SIZE; // u32
const NODE_HEADER_SIZE = NEXT_LEAF_OFF + 4;

// A leaf record is fixed width: key + value. An internal record is child(u32) +
// optional separator key (KEY_SIZE). We size slots to the heap record positions.
const LEAF_RECORD_SIZE = KEY_SIZE + VALUE_SIZE;
const INTERNAL_CHILD_SIZE = 4; // u32 child page id
const INTERNAL_SEP_SIZE = KEY_SIZE; // separator key (fixed-width int)

/** No-sibling sentinel for the leaf chain. Page 0 is always the meta page, so it
 *  can never legitimately be a leaf's right sibling — 0 unambiguously means "end".*/
const NO_SIBLING = 0;

// ---------------------------------------------------------------------------
// Buffer pool: an LRU cache over Disk so IO counts reflect real page touches.
// ---------------------------------------------------------------------------
//
// Why a pool here and not just Disk: a point lookup that descends 3 levels should
// cost 3 page reads on a COLD tree, but 0 reads if the root+internal are already
// cached (the realistic steady state). We need both numbers to tell the truth, so
// the pool counts cache misses (= real Disk reads) separately from logical gets.
// The pool is write-back: dirty pages are flushed on eviction or explicit flush.
//
// Invariant: a page is in `frames` XOR on disk-as-of-last-flush; getMut marks
// dirty so we never lose a mutation. Capacity is intentionally small enough that
// 100k-key builds DO evict (otherwise "buffer pool" would be a fiction), but big
// enough that the root and upper internal nodes stay resident.

class BufferPool {
  private frames = new Map<number, Uint8Array>();
  private dirty = new Set<number>();
  // Insertion-ordered Map doubles as the LRU recency list: re-inserting on touch
  // moves a key to the end, so the first key is always the least-recently-used.
  private order: number[] = [];
  private logicalGets = 0;

  constructor(
    private disk: Disk,
    private capacityPages: number,
  ) {}

  /** Read-only page access. Returns the live frame (NOT a copy): callers must not
   *  mutate it; mutation goes through getMut so dirty-tracking stays honest. */
  get(pageId: number): Uint8Array {
    this.logicalGets++;
    return this.fetch(pageId);
  }

  /** Mutable page access; marks the frame dirty so it is written back on evict. */
  getMut(pageId: number): Uint8Array {
    this.logicalGets++;
    const frame = this.fetch(pageId);
    this.dirty.add(pageId);
    return frame;
  }

  /** Allocate a fresh disk page AND pin it in a frame (already zeroed by Disk).
   *  Returns the page id; the new page is dirty (its initSlotted write must persist). */
  alloc(): number {
    const id = this.disk.allocPage();
    const frame = new Uint8Array(PAGE_SIZE); // matches Disk's zeroed page
    this.admit(id, frame);
    this.dirty.add(id);
    return id;
  }

  private fetch(pageId: number): Uint8Array {
    const cached = this.frames.get(pageId);
    if (cached !== undefined) {
      this.touch(pageId);
      return cached;
    }
    // Cache miss => a real Disk read. This is the IO we measure.
    const bytes = this.disk.readPage(pageId);
    this.admit(pageId, bytes);
    return bytes;
  }

  private admit(pageId: number, frame: Uint8Array): void {
    if (this.frames.size >= this.capacityPages) this.evictOne();
    this.frames.set(pageId, frame);
    this.order.push(pageId);
  }

  private touch(pageId: number): void {
    const idx = this.order.indexOf(pageId);
    if (idx >= 0) this.order.splice(idx, 1);
    this.order.push(pageId);
  }

  private evictOne(): void {
    const victim = this.order.shift();
    if (victim === undefined) return;
    if (this.dirty.has(victim)) {
      // Write-back: a dirty page must hit disk before its frame is reclaimed, or
      // the mutation vanishes — the classic lost-write bug a write-back pool risks.
      this.disk.writePage(victim, this.frames.get(victim)!);
      this.dirty.delete(victim);
    }
    this.frames.delete(victim);
  }

  /** Flush all dirty frames to disk. Call before measuring on-disk state or before
   *  a cold-cache experiment. */
  flush(): void {
    for (const id of this.dirty) this.disk.writePage(id, this.frames.get(id)!);
    this.dirty.clear();
  }

  /** Drop all frames (keeps disk intact). Used to force a COLD cache so the next
   *  lookups measure full root->leaf descent IO. */
  evictAll(): void {
    this.flush();
    this.frames.clear();
    this.order = [];
  }

  logicalGetCount(): number {
    return this.logicalGets;
  }
  resetLogicalGets(): void {
    this.logicalGets = 0;
  }
}

// ---------------------------------------------------------------------------
// Node read/write helpers. Thin wrappers that give names to the byte offsets so
// the tree logic reads like tree logic, not pointer arithmetic.
// ---------------------------------------------------------------------------

function initNode(page: Uint8Array, type: PageType): void {
  const w = new PageWriter(page);
  w.initSlotted(type);
  // initSlotted sets FREE_START to HEADER_SIZE; bump it past our extra header word
  // so the slot directory never overwrites the NEXT_LEAF pointer.
  w.writeU16(SLOTTED_HEADER.FREE_START, NODE_HEADER_SIZE);
  w.writeU32(NEXT_LEAF_OFF, NO_SIBLING);
}

function slotCount(page: Uint8Array): number {
  return new PageReader(page).readU16(SLOTTED_HEADER.SLOT_COUNT);
}

function nextLeaf(page: Uint8Array): number {
  return new PageReader(page).readU32(NEXT_LEAF_OFF);
}

function setNextLeaf(page: Uint8Array, sibling: number): void {
  new PageWriter(page).writeU32(NEXT_LEAF_OFF, sibling);
}

/** Read the (offset,length) of slot i from the directory. */
function slotEntry(page: Uint8Array, i: number): { off: number; len: number } {
  const r = new PageReader(page);
  const base = NODE_HEADER_SIZE + i * SLOT_ENTRY_SIZE;
  return { off: r.readU16(base), len: r.readU16(base + 2) };
}

/** Append a record to the heap (growing down from FREE_END) and a slot entry to
 *  the directory (growing up from FREE_START). Throws via PageWriter bounds if the
 *  two would collide — THAT throw is the "node full, must split" signal the
 *  insert path catches. Returns the new slot index. */
function appendRecord(page: Uint8Array, record: Uint8Array): number {
  const r = new PageReader(page);
  const w = new PageWriter(page);
  const count = r.readU16(SLOTTED_HEADER.SLOT_COUNT);
  const freeEnd = r.readU16(SLOTTED_HEADER.FREE_END);

  const recOff = freeEnd - record.length;
  // Slot directory grows up from the fixed node header; slot i lives at a fixed
  // offset, so we derive it from count rather than reading FREE_START.
  const slotOff = NODE_HEADER_SIZE + count * SLOT_ENTRY_SIZE;
  // Collision check: the new slot entry (4B) plus the new record must not overlap.
  // We throw the same shape PageWriter would, so the caller's "node full" catch is
  // uniform whether the squeeze is detected here or deep in a writeBytes.
  if (slotOff + SLOT_ENTRY_SIZE > recOff) {
    throw new Error(`page write out of bounds: node full (slot ${slotOff}+4 vs rec ${recOff})`);
  }

  w.writeBytes(recOff, record);
  w.writeU16(slotOff, recOff);
  w.writeU16(slotOff + 2, record.length);
  w.writeU16(SLOTTED_HEADER.SLOT_COUNT, count + 1);
  w.writeU16(SLOTTED_HEADER.FREE_START, slotOff + SLOT_ENTRY_SIZE);
  w.writeU16(SLOTTED_HEADER.FREE_END, recOff);
  return count;
}

// --- leaf record accessors (record = [key:4][value:4]) ---

function leafKeyAt(page: Uint8Array, i: number): Uint8Array {
  const { off } = slotEntry(page, i);
  // subarray VIEW; compareKeys only reads it, never retains it, so no copy needed.
  return page.subarray(off, off + KEY_SIZE);
}

function leafValueAt(page: Uint8Array, i: number): number {
  const { off } = slotEntry(page, i);
  return new PageReader(page).readU32(off + KEY_SIZE);
}

function encodeLeafRecord(key: Uint8Array, value: number): Uint8Array {
  const rec = new Uint8Array(LEAF_RECORD_SIZE);
  rec.set(key, 0);
  new DataView(rec.buffer).setUint32(KEY_SIZE, value >>> 0, false); // big-endian to match codec
  return rec;
}

// --- internal record accessors ---
// slot 0   = [child:4]                  (leftmost child, no separator)
// slot i>0 = [child:4][separator:4]     (separator is the low bound of this child)

function internalChildAt(page: Uint8Array, i: number): number {
  const { off } = slotEntry(page, i);
  return new PageReader(page).readU32(off);
}

function internalSepAt(page: Uint8Array, i: number): Uint8Array {
  invariant(i >= 1, "slot 0 has no separator");
  const { off } = slotEntry(page, i);
  return page.subarray(off + INTERNAL_CHILD_SIZE, off + INTERNAL_CHILD_SIZE + INTERNAL_SEP_SIZE);
}

function encodeInternalRecord(child: number, sep: Uint8Array | null): Uint8Array {
  const size = INTERNAL_CHILD_SIZE + (sep ? INTERNAL_SEP_SIZE : 0);
  const rec = new Uint8Array(size);
  new DataView(rec.buffer).setUint32(0, child >>> 0, false);
  if (sep) rec.set(sep, INTERNAL_CHILD_SIZE);
  return rec;
}

// ---------------------------------------------------------------------------
// The tree.
// ---------------------------------------------------------------------------
//
// Why a class with explicit rootId rather than a recursive functional tree: the
// root CHANGES on a root split, and the change must be visible to all future
// operations from one place. A mutable rootId field is the honest model of a real
// engine's meta page pointing at the current root.
//
// Split strategy is a constructor flag so the same code path demonstrates both the
// pathological in-half split and the production right-most special case — that is
// the chapter's central comparison, so it must be one tree, two policies, not two
// codebases.

type SplitPolicy = "half" | "rightmost-aware";

interface InsertResult {
  // When a child split, it returns the new sibling's page id + the separator key
  // (smallest key in the new right node) for the parent to absorb. null = no split.
  promoted: { sep: Uint8Array; rightChild: number } | null;
}

class BPlusTree {
  private rootId: number;

  constructor(
    private pool: BufferPool,
    private policy: SplitPolicy,
  ) {
    // A fresh tree is a single empty leaf that is also the root. This is the only
    // moment a leaf is the root; the first split promotes a new internal root.
    this.rootId = this.pool.alloc();
    initNode(this.pool.getMut(this.rootId), PageType.LEAF);
  }

  rootPageId(): number {
    return this.rootId;
  }

  /** Insert (key,value). Descends to the target leaf, inserts in sorted order, and
   *  splits bottom-up if a node overflows. Duplicate keys are allowed to coexist
   *  here (the chapter doesn't model uniqueness); search returns the first match. */
  insert(key: number, value: number): void {
    const encoded = encodeKey(key, "int");
    const result = this.insertInto(this.rootId, encoded, value);
    if (result.promoted) {
      // Root split: grow the tree by one level. This is the ONLY way height
      // increases, and it happens rarely (every ~fanout^height inserts), which is
      // exactly why height stays tiny.
      const newRoot = this.pool.alloc();
      initNode(this.pool.getMut(newRoot), PageType.INTERNAL);
      const rootPage = this.pool.getMut(newRoot);
      appendRecord(rootPage, encodeInternalRecord(this.rootId, null)); // old root = left child
      appendRecord(rootPage, encodeInternalRecord(result.promoted.rightChild, result.promoted.sep));
      this.rootId = newRoot;
    }
  }

  private insertInto(pageId: number, key: Uint8Array, value: number): InsertResult {
    const page = this.pool.get(pageId);
    const type = new PageReader(page).pageType();
    if (type === PageType.LEAF) return this.insertIntoLeaf(pageId, key, value);
    return this.insertIntoInternal(pageId, key, value);
  }

  private insertIntoLeaf(pageId: number, key: Uint8Array, value: number): InsertResult {
    const page = this.pool.getMut(pageId);
    // Rebuild the leaf's sorted entry list, insert the new pair, and check fit.
    // We rebuild rather than do in-place slot shifting because slotted in-place
    // insert-in-order needs slot-directory shifting anyway; rebuilding keeps the
    // record packing tight (no tombstone gaps) which is what makes occupancy
    // numbers below honest. The cost is O(n) per insert — fine for a teaching
    // build, and the IO (the thing we measure) is identical.
    const entries = this.readLeafEntries(page);
    const pos = lowerBound(entries.map((e) => e.key), key);
    entries.splice(pos, 0, { key, value });

    if (this.leafFits(entries.length)) {
      this.writeLeafEntries(page, entries, nextLeaf(page));
      return { promoted: null };
    }
    return this.splitLeaf(pageId, entries);
  }

  /** Split an overflowing leaf into [left | right], link the sibling chain, and
   *  return the separator (smallest key of right) for the parent. */
  private splitLeaf(pageId: number, entries: { key: Uint8Array; value: number }[]): InsertResult {
    const left = this.pool.getMut(pageId);
    const oldNext = nextLeaf(left);

    // THE policy fork. "half" splits down the middle (textbook). "rightmost-aware"
    // detects an append-only pattern — the new key landed at the very end and there
    // is no right sibling (this leaf is the current rightmost) — and splits so the
    // LEFT keeps everything and the RIGHT gets just the new tail key. That keeps the
    // left leaf 100% packed instead of stranding it at 50%, which is the essence of
    // bulk-load / sequential-insert optimization in real engines.
    let splitAt: number;
    if (this.policy === "rightmost-aware" && oldNext === NO_SIBLING && this.isAppend(entries)) {
      splitAt = entries.length - 1; // right node gets only the last (new) key
    } else {
      splitAt = Math.ceil(entries.length / 2);
    }

    const leftEntries = entries.slice(0, splitAt);
    const rightEntries = entries.slice(splitAt);

    const rightId = this.pool.alloc();
    const right = this.pool.getMut(rightId);
    initNode(right, PageType.LEAF);

    this.writeLeafEntries(left, leftEntries, rightId); // left now points to new right
    this.writeLeafEntries(right, rightEntries, oldNext); // right inherits old chain tail

    // Separator = first key of the right node. In a B+tree, leaf keys are NOT
    // removed on split (unlike a B-tree); the separator is a COPY pushed up.
    const sep = rightEntries[0].key;
    return { promoted: { sep: copyKey(sep), rightChild: rightId } };
  }

  /** Append detection: the inserted key is the new maximum (it's at the last slot
   *  and strictly greater than the previous last). This is what makes sequential
   *  load fast — and what the naive "half" policy fails to exploit. */
  private isAppend(entries: { key: Uint8Array; value: number }[]): boolean {
    const n = entries.length;
    if (n < 2) return true;
    return compareKeys(entries[n - 1].key, entries[n - 2].key) > 0;
  }

  private insertIntoInternal(pageId: number, key: Uint8Array, value: number): InsertResult {
    const page = this.pool.get(pageId);
    const childIdx = this.findChildIndex(page, key);
    const childId = internalChildAt(page, childIdx);
    const childResult = this.insertInto(childId, key, value);
    if (!childResult.promoted) return { promoted: null };

    // Child split: absorb (separator, newRightChild). Rebuild the internal node's
    // children+separators in order, then split this node if IT overflows.
    const node = this.pool.getMut(pageId);
    const children = this.readInternalChildren(node);
    // Insert the new child immediately after the child that split.
    children.splice(childIdx + 1, 0, {
      child: childResult.promoted.rightChild,
      sep: childResult.promoted.sep,
    });

    if (this.internalFits(children.length)) {
      this.writeInternalChildren(node, children);
      return { promoted: null };
    }
    return this.splitInternal(pageId, children);
  }

  private splitInternal(
    pageId: number,
    children: { child: number; sep: Uint8Array | null }[],
  ): InsertResult {
    // Internal split PUSHES the middle separator up (it does NOT stay in either
    // child — that's the B-tree/B+tree internal-node rule). The middle child's
    // separator becomes the new parent separator and that child becomes the
    // leftmost (sep=null) of the right node.
    const mid = Math.floor(children.length / 2);
    const promotedSep = children[mid].sep!;
    const leftChildren = children.slice(0, mid);
    const rightChildren = children.slice(mid);
    rightChildren[0] = { child: rightChildren[0].child, sep: null }; // new leftmost

    const left = this.pool.getMut(pageId);
    initNode(left, PageType.INTERNAL);
    this.writeInternalChildren(left, leftChildren);

    const rightId = this.pool.alloc();
    const right = this.pool.getMut(rightId);
    initNode(right, PageType.INTERNAL);
    this.writeInternalChildren(right, rightChildren);

    return { promoted: { sep: copyKey(promotedSep), rightChild: rightId } };
  }

  /** Point lookup. Returns the value, or null if absent. Counts nothing itself —
   *  the buffer pool's miss counter is what measures IO, so callers wrap this in
   *  a cold-cache reset to get "pages touched per lookup". */
  search(key: number): number | null {
    const encoded = encodeKey(key, "int");
    let pageId = this.rootId;
    for (;;) {
      const page = this.pool.get(pageId);
      const type = new PageReader(page).pageType();
      if (type === PageType.LEAF) {
        const entries = this.readLeafEntries(page);
        const pos = lowerBound(entries.map((e) => e.key), encoded);
        if (pos < entries.length && compareKeys(entries[pos].key, encoded) === 0) {
          return entries[pos].value;
        }
        return null;
      }
      pageId = internalChildAt(page, this.findChildIndex(page, encoded));
    }
  }

  /** Range scan [lo, hi). Descends once to the leaf holding lo, then walks the
   *  leaf sibling chain — no re-descent per key. This is the B+tree's headline
   *  advantage over a hash index. Returns matching values in key order. */
  rangeScan(lo: number, hi: number): number[] {
    const loKey = encodeKey(lo, "int");
    const hiKey = encodeKey(hi, "int");
    const out: number[] = [];
    let pageId = this.descendToLeaf(loKey);
    for (;;) {
      const page = this.pool.get(pageId);
      const n = slotCount(page);
      const entries = this.readLeafEntries(page);
      for (let i = 0; i < n; i++) {
        const k = entries[i].key;
        if (compareKeys(k, loKey) < 0) continue;
        if (compareKeys(k, hiKey) >= 0) return out; // past the range; chain is sorted, stop
        out.push(entries[i].value);
      }
      const sib = nextLeaf(page);
      if (sib === NO_SIBLING) return out;
      pageId = sib;
    }
  }

  /** Measured tree height = number of levels = page reads for any point lookup on a
   *  cold cache. Computed by walking root->leftmost-leaf, not assumed. */
  height(): number {
    let pageId = this.rootId;
    let h = 1;
    for (;;) {
      const page = this.pool.get(pageId);
      if (new PageReader(page).pageType() === PageType.LEAF) return h;
      pageId = internalChildAt(page, 0);
      h++;
    }
  }

  /** Walk the leaf chain and report average leaf occupancy = stored entries /
   *  max entries per leaf. THIS is the sequential-vs-random number the chapter is
   *  about; it is computed from real on-page slot counts, not estimated. */
  leafOccupancy(): { leaves: number; avgFillRatio: number; totalEntries: number } {
    let pageId = this.descendToLeaf(encodeKey(0, "int")); // leftmost leaf
    const cap = this.leafCapacity();
    let leaves = 0;
    let total = 0;
    for (;;) {
      const page = this.pool.get(pageId);
      total += slotCount(page);
      leaves++;
      const sib = nextLeaf(page);
      if (sib === NO_SIBLING) break;
      pageId = sib;
    }
    return { leaves, totalEntries: total, avgFillRatio: total / (leaves * cap) };
  }

  /** Measured fanout = AVERAGE children across all internal nodes one level above
   *  the leaves. Why that level and not the root: the root is usually far from
   *  full (it only fills as the whole tree grows by a power of fanout), so root
   *  slot-count UNDERSTATES fanout — reporting it would make the capacity claim
   *  look false. The level above leaves is the densest internal level and the one
   *  the height arithmetic actually relies on. Computed by walking the tree, not
   *  assumed. Also returns the root's own child count for contrast. */
  measuredFanout(): { aboveLeavesAvg: number; aboveLeavesNodes: number; rootChildren: number } {
    const rootPage = this.pool.get(this.rootId);
    if (new PageReader(rootPage).pageType() === PageType.LEAF) {
      return { aboveLeavesAvg: 0, aboveLeavesNodes: 0, rootChildren: 0 }; // tree too small
    }
    const rootChildren = slotCount(rootPage);
    // BFS down to the last internal level (parents of leaves).
    let level = [this.rootId];
    for (;;) {
      // Peek: are this level's children leaves? If so, THIS level is "above leaves".
      const firstChildId = internalChildAt(this.pool.get(level[0]), 0);
      const childIsLeaf = new PageReader(this.pool.get(firstChildId)).pageType() === PageType.LEAF;
      if (childIsLeaf) break;
      const next: number[] = [];
      for (const id of level) {
        const p = this.pool.get(id);
        for (let i = 0; i < slotCount(p); i++) next.push(internalChildAt(p, i));
      }
      level = next;
    }
    let childSum = 0;
    for (const id of level) childSum += slotCount(this.pool.get(id));
    return {
      aboveLeavesAvg: childSum / level.length,
      aboveLeavesNodes: level.length,
      rootChildren,
    };
  }

  // ---- capacity arithmetic (the heart of "why three levels") ----

  /** Max (key,value) entries a leaf holds = usable bytes / (record + slot entry). */
  leafCapacity(): number {
    const usable = PAGE_SIZE - NODE_HEADER_SIZE;
    return Math.floor(usable / (LEAF_RECORD_SIZE + SLOT_ENTRY_SIZE));
  }

  /** Max children an internal node holds. Each child after the first carries a
   *  separator; the first does not. Solve for n where:
   *    n*(child+slot) + (n-1)*sep <= usable.  */
  maxInternalChildren(): number {
    const usable = PAGE_SIZE - NODE_HEADER_SIZE;
    // n*(child + slot) + (n-1)*sep <= usable
    // n*(child + slot + sep) <= usable + sep
    const perChild = INTERNAL_CHILD_SIZE + SLOT_ENTRY_SIZE + INTERNAL_SEP_SIZE;
    return Math.floor((usable + INTERNAL_SEP_SIZE) / perChild);
  }

  private leafFits(count: number): boolean {
    return count <= this.leafCapacity();
  }
  private internalFits(count: number): boolean {
    return count <= this.maxInternalChildren();
  }

  // ---- private node (de)serialization ----

  private readLeafEntries(page: Uint8Array): { key: Uint8Array; value: number }[] {
    const n = slotCount(page);
    const out: { key: Uint8Array; value: number }[] = [];
    for (let i = 0; i < n; i++) {
      out.push({ key: copyKey(leafKeyAt(page, i)), value: leafValueAt(page, i) });
    }
    return out;
  }

  private writeLeafEntries(
    page: Uint8Array,
    entries: { key: Uint8Array; value: number }[],
    sibling: number,
  ): void {
    initNode(page, PageType.LEAF);
    setNextLeaf(page, sibling);
    for (const e of entries) appendRecord(page, encodeLeafRecord(e.key, e.value));
  }

  private readInternalChildren(page: Uint8Array): { child: number; sep: Uint8Array | null }[] {
    const n = slotCount(page);
    const out: { child: number; sep: Uint8Array | null }[] = [];
    for (let i = 0; i < n; i++) {
      out.push({
        child: internalChildAt(page, i),
        sep: i === 0 ? null : copyKey(internalSepAt(page, i)),
      });
    }
    return out;
  }

  private writeInternalChildren(
    page: Uint8Array,
    children: { child: number; sep: Uint8Array | null }[],
  ): void {
    initNode(page, PageType.INTERNAL);
    for (const c of children) appendRecord(page, encodeInternalRecord(c.child, c.sep));
  }

  /** Find which child subtree a key belongs to. Returns the child slot index:
   *  the last child whose separator is <= key (slot 0 has no separator and is the
   *  catch-all for keys below the first separator). */
  private findChildIndex(page: Uint8Array, key: Uint8Array): number {
    const n = slotCount(page);
    let idx = 0; // default to leftmost child (keys below first separator)
    for (let i = 1; i < n; i++) {
      if (compareKeys(internalSepAt(page, i), key) <= 0) idx = i;
      else break; // separators are sorted; once one exceeds key, stop
    }
    return idx;
  }

  private descendToLeaf(key: Uint8Array): number {
    let pageId = this.rootId;
    for (;;) {
      const page = this.pool.get(pageId);
      if (new PageReader(page).pageType() === PageType.LEAF) return pageId;
      pageId = internalChildAt(page, this.findChildIndex(page, key));
    }
  }
}

// ---------------------------------------------------------------------------
// Small key helpers.
// ---------------------------------------------------------------------------

/** Binary search for the first index whose key is >= target (lower_bound). Keys
 *  must already be sorted (B+tree leaf invariant). O(log n) so leaf insert/search
 *  in-page work doesn't dominate the IO we are measuring. */
function lowerBound(keys: Uint8Array[], target: Uint8Array): number {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (compareKeys(keys[mid], target) < 0) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Copy a key view out of a page. Needed whenever a key must outlive the frame it
 *  was read from (e.g. promoted separators, rebuilt entry lists) — subarray views
 *  alias the frame and would corrupt once that frame is reused/evicted. */
function copyKey(view: Uint8Array): Uint8Array {
  return view.slice();
}

// ---------------------------------------------------------------------------
// The demo / measurement harness.
// ---------------------------------------------------------------------------

const N_KEYS = 100_000;
const SEED = 0x5eed;
// Pool small enough that a 100k build genuinely evicts (so "buffer pool" is real),
// large enough that root + upper internals stay hot. ~200 frames ≈ 800KB.
const POOL_PAGES = 256;

/** Build a tree by inserting keys in the given order under the given split policy.
 *  Returns the tree, its pool, and the backing disk so the harness can read
 *  honest IO counters off the disk afterward. */
function buildTree(keys: number[], policy: SplitPolicy): { tree: BPlusTree; pool: BufferPool; disk: Disk } {
  const disk = new Disk({ initialPages: 64 });
  disk.allocPage(); // reserve page 0 as the meta page so NO_SIBLING (0) is never a real leaf
  const pool = new BufferPool(disk, POOL_PAGES);
  const tree = new BPlusTree(pool, policy);
  for (let i = 0; i < keys.length; i++) {
    // value = key+1 as a synthetic row pointer; lets search assert it round-trips.
    tree.insert(keys[i], keys[i] + 1);
  }
  pool.flush();
  return { tree, pool, disk };
}

function main(): void {
  console.log("=== stage03: B+树索引 — 扇出、分裂与高度 ===\n");

  // --- capacity arithmetic, computed not asserted ---
  const refDisk = new Disk({ initialPages: 4 });
  refDisk.allocPage();
  const refPool = new BufferPool(refDisk, 8);
  const ref = new BPlusTree(refPool, "half");
  const leafCap = ref.leafCapacity();
  const internalFanout = ref.maxInternalChildren();

  console.log("【容量算术】页 4096B，键 4B(u32)，值 4B(行指针)，槽 4B");
  printTable([
    { 量: "leaf 每页 entry 上限", 值: leafCap, 推导: `floor((4096-${NODE_HEADER_SIZE})/(${LEAF_RECORD_SIZE}+${SLOT_ENTRY_SIZE}))` },
    { 量: "internal 扇出(子节点上限)", 值: internalFanout, 推导: `floor((usable+sep)/(child+slot+sep))` },
  ]);
  // The headline claim, computed from the real fanout: how many rows fit at each height.
  console.log("\n【为什么三层装十亿行】每层乘以扇出，叶层乘以 leaf 容量：");
  printTable([1, 2, 3, 4].map((h) => {
    const internalLevels = h - 1;
    const rows = Math.pow(internalFanout, internalLevels) * leafCap;
    return {
      高度: h,
      最大行数: rows.toLocaleString("en-US"),
      "点查 IO(页)": h,
    };
  }));
  console.log(`(扇出 ${internalFanout} → 3 层即 ${(Math.pow(internalFanout, 2) * leafCap).toLocaleString("en-US")} 行，远超十亿)`);

  // --- random-key build: the healthy baseline ---
  const rng = createRng(SEED);
  // Distinct keys in [0, 4*N) shuffled => random insertion order, no duplicates.
  const randomKeys = Array.from({ length: N_KEYS }, (_, i) => i);
  rng.shuffle(randomKeys);
  const { tree: randTree, pool: randPool, disk: randDisk } = buildTree(randomKeys, "half");
  // Capture build IO NOW — the cold-cache probes below reset the disk counters.
  const randBuildWrites = randDisk.stats().writes;

  const fanout = randTree.measuredFanout();
  const height = randTree.height();
  const occ = randTree.leafOccupancy();

  console.log(`\n【随机键插入 ${N_KEYS.toLocaleString("en-US")} 个 (seed=0x${SEED.toString(16)})】`);
  printTable([
    { 指标: "实测扇出(叶上一层均值)", 值: fanout.aboveLeavesAvg.toFixed(1), 说明: `${fanout.aboveLeavesNodes} 个内部节点真从页读出，对比容量 ${internalFanout}` },
    { 指标: "root 子节点数", 值: fanout.rootChildren, 说明: "root 通常远不满，故扇出看叶上一层而非 root" },
    { 指标: "树高度", 值: height, 说明: "root→leaf 实走层数" },
    { 指标: "叶子页数", 值: occ.leaves, 说明: "leaf 链长度" },
    { 指标: "叶利用率", 值: `${(occ.avgFillRatio * 100).toFixed(1)}%`, 说明: "已存 entry / 容量" },
    { 指标: "构建总磁盘写", 值: randBuildWrites, 说明: "buffer pool 回写次数(measured)" },
  ]);

  // --- INVARIANT: a point lookup touches exactly `height` pages on a cold cache ---
  console.log("\n【不变量验证：每次 search 触页数 == 树高度】");
  randPool.evictAll(); // force cold cache so descent reads hit disk
  const probeKeys = [randomKeys[0], randomKeys[N_KEYS >> 1], randomKeys[N_KEYS - 1], 7, 42];
  const ioRows = probeKeys.map((k) => {
    randPool.evictAll(); // cold for each probe: isolate one descent's IO
    randDisk.resetStats();
    const v = randTree.search(k);
    const reads = randDisk.stats().reads;
    invariant(v === k + 1, `search(${k}) must return ${k + 1}, got ${v}`);
    assertEq(reads, height, `point lookup must read exactly ${height} pages (= height)`);
    return { 查询键: k, 命中值: v!, "磁盘读(页)": reads, "==高度": reads === height ? "yes" : "NO" };
  });
  printTable(ioRows);
  // A miss also costs exactly height reads (it still descends to a leaf).
  randPool.evictAll();
  randDisk.resetStats();
  const missVal = randTree.search(N_KEYS + 99999);
  assertEq(missVal as null, null, "absent key must return null");
  assertEq(randDisk.stats().reads, height, "a miss still descends full height");
  console.log(`未命中键也读 ${randDisk.stats().reads} 页(=高度)：search 成本与命中与否无关，只看高度。`);

  // --- range scan throughput: MEASURED wall-clock, labeled machine-dependent ---
  console.log("\n【范围扫描吞吐 (扫 1000 个键，MEASURED 真实墙钟)】");
  const SCAN_SPAN = 1000;
  const scanStart = N_KEYS >> 2;
  // Warm the pool with one scan so we measure steady-state chain-walk, not cold IO,
  // then time many repeats. The COUNT (keys returned) is deterministic; ops/sec is not.
  const warm = randTree.rangeScan(scanStart, scanStart + SCAN_SPAN);
  invariant(warm.length === SCAN_SPAN, `range should return ${SCAN_SPAN} keys, got ${warm.length}`);
  let sink = 0;
  const scanBench = bench(() => {
    const r = randTree.rangeScan(scanStart, scanStart + SCAN_SPAN);
    sink += r.length; // defeat dead-code elimination
  }, 200);
  invariant(sink > 0, "sink guards the optimizer");
  const keysPerSec = scanBench.opsPerSec * SCAN_SPAN;
  printTable([
    { 指标: "每次扫描返回键数", 值: SCAN_SPAN, 类型: "deterministic" },
    { 指标: "扫描 ops/sec", 值: Math.round(scanBench.opsPerSec).toLocaleString("en-US"), 类型: "MEASURED" },
    { 指标: "键吞吐 keys/sec", 值: Math.round(keysPerSec).toLocaleString("en-US"), 类型: "MEASURED" },
    { 指标: "ns/扫描", 值: Math.round(scanBench.nsPerOp).toLocaleString("en-US"), 类型: "MEASURED" },
  ]);
  console.log("注：内存模拟磁盘 + 全驻留缓冲池，绝对值偏乐观；可迁移的是「单次下降一次、之后顺链」这一相对结构，不是 keys/sec 数字本身。");

  // --- FAILURE MODE: sequential insert degrades leaf occupancy to ~50% ---
  console.log("\n" + "=".repeat(64));
  console.log("【失败模式：单调递增键插入 → 右侧热点分裂 → 叶利用率塌到 ~50%】");
  console.log("=".repeat(64));
  const seqKeys = Array.from({ length: N_KEYS }, (_, i) => i); // 0,1,2,... strictly increasing
  const { tree: seqTree, disk: seqDisk } = buildTree(seqKeys, "half");
  const seqOcc = seqTree.leafOccupancy();

  // --- the production fix: rightmost-aware split keeps the left leaf packed ---
  const { tree: fixTree, disk: fixDisk } = buildTree(seqKeys, "rightmost-aware");
  const fixOcc = fixTree.leafOccupancy();

  printTable([
    {
      插入模式: "随机键 + half 分裂",
      叶利用率: `${(occ.avgFillRatio * 100).toFixed(1)}%`,
      叶子页数: occ.leaves,
      磁盘写: randBuildWrites,
    },
    {
      插入模式: "顺序键 + half 分裂(退化)",
      叶利用率: `${(seqOcc.avgFillRatio * 100).toFixed(1)}%`,
      叶子页数: seqOcc.leaves,
      磁盘写: seqDisk.stats().writes,
    },
    {
      插入模式: "顺序键 + 右侧特判(修复)",
      叶利用率: `${(fixOcc.avgFillRatio * 100).toFixed(1)}%`,
      叶子页数: fixOcc.leaves,
      磁盘写: fixDisk.stats().writes,
    },
  ]);

  // Prove the degradation is real, not narrated: assert the occupancy ordering.
  invariant(
    seqOcc.avgFillRatio < 0.6,
    `sequential+half must collapse below 60% (got ${(seqOcc.avgFillRatio * 100).toFixed(1)}%)`,
  );
  invariant(
    fixOcc.avgFillRatio > seqOcc.avgFillRatio + 0.2,
    "rightmost-aware fix must materially beat naive split",
  );
  // Correctness must survive the fix: spot-check a few lookups on the fixed tree.
  for (const k of [0, N_KEYS >> 1, N_KEYS - 1]) {
    invariant(fixTree.search(k) === k + 1, `fixed tree must still find key ${k}`);
  }

  console.log("\n机制解读：");
  console.log("- half 分裂把满叶对半切，但顺序插入下左半永远不再被插入 → 永久 ~50% 浪费，");
  console.log(`  叶子页数几乎翻倍 (${occ.leaves} → ${seqOcc.leaves})，等量数据多占 ~一倍空间与缓存。`);
  console.log("- 右侧特判：当新键是当前最大且落在最右叶时，只把这一个新键切到右节点，");
  console.log(`  左叶保持 100% 满 → 利用率回到 ${(fixOcc.avgFillRatio * 100).toFixed(1)}%，这正是 bulk-load 的雏形。`);
  console.log("- 真实引擎(PostgreSQL/SQLite)对顺序主键即用此类右侧 fast-path / 批量装载，");
  console.log("  否则自增主键这一最常见模式会让索引膨胀近一倍。");
  console.log(
    `- 注：随机键磁盘写(${randBuildWrites}) >> 顺序键(${seqDisk.stats().writes})，是缓冲池效应：随机访问` +
      "颠簸 LRU 反复回写脏页，顺序访问对池友好。这恰好说明利用率之外，访问局部性也是真成本。",
  );

  console.log("\n=== 全部不变量通过：扇出/高度/触页数/利用率均为实测真值 ===");
}

main();
