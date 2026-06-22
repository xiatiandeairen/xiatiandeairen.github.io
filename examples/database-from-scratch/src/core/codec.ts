// core/codec.ts — row/key encoding, and the byte-order comparison the whole
// index layer sorts by.
//
// Why one codec for the whole book: a B+tree, an LSM tree, and a query executor
// must agree, to the bit, on how a row turns into bytes and how two keys order.
// If they disagreed, a row written by the storage layer would be unreadable by
// the index, and "the engine is real" would collapse. So encoding and
// comparison live here, once.
//
// The load-bearing trick: keys are encoded so that BYTE-WISE comparison equals
// LOGICAL comparison. That is why ints are big-endian (high byte first sorts
// correctly) and strings are raw UTF-8 (lexicographic bytes == lexicographic
// codepoints for the ASCII/BMP range we use). This lets every index sort with a
// single memcmp (`compareKeys`) instead of type-aware comparators — a real
// engine technique (it's how key-ordered storage stays type-agnostic).
//
// Invariant for compareKeys: it must be a total order consistent with the typed
// values. The encoding upholds that; if you ever change encodeRow's layout you
// MUST re-verify compareKeys, or sorted indexes silently corrupt.
//
// Failure mode modeled: a fixed-width int that overflows 32 bits, and a string
// longer than its varint length prefix can address — both throw rather than
// truncate, because a truncated key would sort wrong and the corruption would
// surface much later as "missing rows".

import { PageReader, PageWriter } from "./page.js";
import { PAGE_SIZE } from "./disk.js";

export type ColType = "int" | "string";
export type Schema = ColType[];
export type Value = number | string;

// Scratch page used purely as an encoding buffer. Reusing one avoids allocating
// a DataView per encode (encoding happens in hot loops during bulk load), and
// the bytes are always sliced out before the next call, so reuse is safe.
const scratch = new Uint8Array(PAGE_SIZE);

/** Encode a row to a self-describing-enough byte string given an out-of-band
 *  schema. Layout per column: int -> 4 bytes big-endian; string -> varint(len) +
 *  UTF-8 bytes. No per-row type tags: the schema is known to both writer and
 *  reader, so spending bytes on tags would just lower rows-per-page. */
export function encodeRow(values: Value[], schema: Schema): Uint8Array {
  if (values.length !== schema.length) {
    throw new Error(`encodeRow: ${values.length} values for ${schema.length} columns`);
  }
  const w = new PageWriter(scratch);
  let off = 0;
  for (let i = 0; i < schema.length; i++) {
    if (schema[i] === "int") {
      const v = values[i] as number;
      if (!Number.isInteger(v)) throw new Error(`encodeRow: column ${i} not an int: ${v}`);
      // Encode as unsigned 32-bit. The book's datasets use non-negative ids;
      // signed keys would need a sign-flip bias trick to keep byte-order sane,
      // which we call out as a stage exercise rather than hide here.
      if (v < 0 || v > 0xffffffff) throw new Error(`encodeRow: int ${v} out of u32 range`);
      w.writeU32(off, v);
      off += 4;
    } else {
      const s = values[i] as string;
      const enc = new TextEncoder().encode(s);
      off = w.writeVarint(off, enc.length);
      off = w.writeBytes(off, enc);
    }
  }
  return scratch.slice(0, off); // copy out; scratch will be overwritten next call
}

/** Decode a row previously produced by encodeRow with the same schema. */
export function decodeRow(buf: Uint8Array, schema: Schema): Value[] {
  // Pad into a page-sized buffer because PageReader is page-shaped; the extra
  // zero bytes are never read since we advance strictly by the schema.
  const padded = buf.length === PAGE_SIZE ? buf : padToPage(buf);
  const r = new PageReader(padded);
  const out: Value[] = [];
  let off = 0;
  for (const t of schema) {
    if (t === "int") {
      out.push(r.readU32(off));
      off += 4;
    } else {
      const { value: len, next } = r.readVarint(off);
      const bytes = r.readBytes(next, len);
      out.push(new TextDecoder().decode(bytes));
      off = next + len;
    }
  }
  return out;
}

/** Encode a single key value to bytes whose memcmp matches typed order.
 *  Used by indexes that key on one column; for composite keys, concatenate
 *  encoded components (each int is fixed-width, each string is length-prefixed,
 *  so the concatenation is unambiguous and still byte-ordered for the prefix). */
export function encodeKey(value: Value, type: ColType): Uint8Array {
  if (type === "int") {
    const v = value as number;
    if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) {
      throw new Error(`encodeKey: int ${v} out of u32 range`);
    }
    // Big-endian so byte 0 is the most significant — the reason compareKeys can
    // be a plain memcmp.
    return new Uint8Array([(v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff]);
  }
  return new TextEncoder().encode(value as string);
}

/** Total byte-order comparison: -1 | 0 | 1. THE comparison the entire index
 *  layer sorts by. Compares the shared prefix byte-by-byte, then by length, so
 *  it is a correct lexicographic order (the standard memcmp+length tie-break).
 *  A shorter key that is a prefix of a longer one sorts first — matching how
 *  string ordering and big-endian-int prefixes both behave. */
export function compareKeys(a: Uint8Array, b: Uint8Array): -1 | 0 | 1 {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  }
  if (a.length === b.length) return 0;
  return a.length < b.length ? -1 : 1;
}

function padToPage(buf: Uint8Array): Uint8Array {
  if (buf.length > PAGE_SIZE) {
    throw new Error(`decodeRow: row ${buf.length}B exceeds page ${PAGE_SIZE}B`);
  }
  const p = new Uint8Array(PAGE_SIZE);
  p.set(buf);
  return p;
}
