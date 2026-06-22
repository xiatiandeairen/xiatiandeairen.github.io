// core/page.ts — the binary layout primitives every page format is built on.
//
// Why this exists: a page is just 4096 raw bytes. Everything above (slotted rows,
// B+tree nodes, WAL records) is a discipline imposed on those bytes. If each
// chapter invented its own read/write helpers, off-by-one byte bugs would be
// unattributable and the book's "the format is real" promise would be hollow.
// So all binary access goes through PageReader/PageWriter, which are thin, honest
// wrappers over DataView — no hidden allocation, no endianness surprises.
//
// Endianness: BIG-ENDIAN everywhere. Why big-endian specifically: core/codec's
// key comparison is a raw byte memcmp, and big-endian integers sort correctly
// under byte comparison while little-endian do not. Mixing endianness between
// page.ts and codec.ts would silently break index ordering — so the choice is
// load-bearing, not aesthetic. (Real engines like SQLite also store big-endian.)
//
// Invariant: writers never auto-grow. A page is exactly PAGE_SIZE; an overflow
// is a structural decision (overflow page chain), not something a setter should
// hide. Writing past the end throws — that's the "page full" signal slotted-page
// and B+tree split logic depend on.

import { PAGE_SIZE } from "./disk.js";

export enum PageType {
  // Stored as one byte at the page header. Numeric values are part of the
  // on-disk format: renumbering them would make old pages unreadable.
  META = 0,
  LEAF = 1,
  INTERNAL = 2,
  OVERFLOW = 3,
}

// Slotted-page header layout (offsets in bytes from page start). A slotted page
// keeps a slot directory growing down from the header and the row heap growing
// up from the page end; they meet in the middle, and "free space" is the gap.
// This is the canonical layout for variable-length rows and is reused by stage01
// (the page) and stage03 (B+tree leaf nodes hold rows the same way).
export const SLOTTED_HEADER = {
  TYPE: 0, // u8  PageType
  SLOT_COUNT: 1, // u16 number of slots in the directory
  FREE_START: 3, // u16 offset where the next slot entry would go (grows up)
  FREE_END: 5, // u16 offset of the lowest row byte written (grows down)
  HEADER_SIZE: 7, // first byte available for the slot directory
} as const;

// Each slot directory entry is (offset:u16, length:u16). A length of 0 marks a
// tombstoned slot — we keep the entry so existing slot ids stay stable after a
// delete (callers may hold slot ids as record pointers).
export const SLOT_ENTRY_SIZE = 4;

export class PageReader {
  private view: DataView;
  constructor(public readonly bytes: Uint8Array) {
    // Bind a DataView once; per-read DataView allocation would dominate the IO
    // we're trying to measure. Length is asserted so a truncated buffer fails
    // here, not as a confusing OOB three calls later.
    if (bytes.length !== PAGE_SIZE) {
      throw new Error(`PageReader: expected ${PAGE_SIZE}-byte page, got ${bytes.length}`);
    }
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  readU8(off: number): number {
    return this.view.getUint8(off);
  }
  readU16(off: number): number {
    return this.view.getUint16(off, false); // false = big-endian (see header)
  }
  readU32(off: number): number {
    return this.view.getUint32(off, false);
  }

  /** Read a LEB128-style unsigned varint; returns value and bytes consumed.
   *  Varints keep small lengths (the common case for row field sizes) to 1 byte
   *  instead of 2-4, which directly raises rows-per-page. The caller MUST use
   *  the returned `next` to advance — guessing the width reintroduces the very
   *  fragility varints exist to remove. */
  readVarint(off: number): { value: number; next: number } {
    let value = 0;
    let shift = 0;
    let pos = off;
    for (;;) {
      const b = this.view.getUint8(pos++);
      value |= (b & 0x7f) << shift;
      if ((b & 0x80) === 0) break;
      shift += 7;
      // 5 bytes covers 32 bits; beyond that the value can't fit and the stream
      // is corrupt. Better to throw than to silently wrap.
      if (shift > 35) throw new Error("readVarint: overlong / corrupt varint");
    }
    return { value: value >>> 0, next: pos };
  }

  /** Read `len` raw bytes as a subarray VIEW (no copy). Callers that need to
   *  retain the bytes past the page's lifetime must copy; we don't copy here
   *  because most reads are immediately decoded. */
  readBytes(off: number, len: number): Uint8Array {
    return this.bytes.subarray(off, off + len);
  }

  pageType(): PageType {
    return this.readU8(SLOTTED_HEADER.TYPE) as PageType;
  }
}

export class PageWriter {
  private view: DataView;
  constructor(public readonly bytes: Uint8Array) {
    if (bytes.length !== PAGE_SIZE) {
      throw new Error(`PageWriter: expected ${PAGE_SIZE}-byte page, got ${bytes.length}`);
    }
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  writeU8(off: number, v: number): void {
    this.bounds(off, 1);
    this.view.setUint8(off, v & 0xff);
  }
  writeU16(off: number, v: number): void {
    this.bounds(off, 2);
    this.view.setUint16(off, v & 0xffff, false);
  }
  writeU32(off: number, v: number): void {
    this.bounds(off, 4);
    this.view.setUint32(off, v >>> 0, false);
  }

  /** Write an unsigned varint; returns the next free offset. Mirrors readVarint. */
  writeVarint(off: number, value: number): number {
    let v = value >>> 0;
    let pos = off;
    for (;;) {
      let b = v & 0x7f;
      v >>>= 7;
      if (v !== 0) b |= 0x80;
      this.bounds(pos, 1);
      this.view.setUint8(pos++, b);
      if (v === 0) break;
    }
    return pos;
  }

  /** Copy raw bytes into the page at off; returns next offset. */
  writeBytes(off: number, src: Uint8Array): number {
    this.bounds(off, src.length);
    this.bytes.set(src, off);
    return off + src.length;
  }

  // ---- slotted-page header helpers (the discipline, not just the bytes) ----

  /** Initialize an empty slotted page of the given type. Sets the free-space
   *  window to the whole page below the header. Must be called on a fresh page
   *  before any insert, otherwise FREE_END reads as 0 and the first row write
   *  collides with the header. */
  initSlotted(type: PageType): void {
    this.writeU8(SLOTTED_HEADER.TYPE, type);
    this.writeU16(SLOTTED_HEADER.SLOT_COUNT, 0);
    this.writeU16(SLOTTED_HEADER.FREE_START, SLOTTED_HEADER.HEADER_SIZE);
    this.writeU16(SLOTTED_HEADER.FREE_END, PAGE_SIZE);
  }

  private bounds(off: number, len: number): void {
    // The page-full / page-overrun guard. Slotted insert and B+tree split BOTH
    // rely on this throwing rather than corrupting an adjacent page — that throw
    // is the trigger to allocate a new page or split a node.
    if (off < 0 || off + len > PAGE_SIZE) {
      throw new Error(`page write out of bounds: [${off}, ${off + len}) vs ${PAGE_SIZE}`);
    }
  }
}
