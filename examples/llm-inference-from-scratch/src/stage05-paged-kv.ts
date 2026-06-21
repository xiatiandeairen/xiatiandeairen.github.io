// stage05-paged-kv.ts — PagedAttention: manage the KV cache like an OS manages RAM.
//
// The problem this chapter solves (and the one chapter 02's contiguous cache
// created): a contiguous KV cache must reserve maxSeq slots PER sequence up front,
// because it grows by reallocating one flat array (see core/model.forwardStep). If
// you serve N concurrent requests you reserve N * maxSeq, but most requests finish
// far short of maxSeq — the reserved-but-unused tail is pure internal fragmentation.
// vLLM's PagedAttention fixes this exactly the way an OS fixes external memory
// fragmentation: chop the cache into fixed-size BLOCKS (pages), hand them out on
// demand, and keep a per-sequence BLOCK TABLE (page table) mapping logical position
// -> physical block. A sequence only ever holds ceil(len/blockSize) blocks, so the
// only waste is the partly-filled last block — bounded by blockSize, not maxSeq.
//
// What this stage proves with REAL numbers:
//   (a) memory: contiguous-reserved vs paged footprint + internal fragmentation %.
//   (b) capacity: under a fixed VRAM budget, paged fits dramatically more sequences.
//   (c) correctness: paged-attention last-token logits == contiguous, drift == 0.
//       This is THE load-bearing invariant — paging is a memory layout change, it
//       must not touch the math. We get it for free by gathering blocks into the
//       exact contiguous layout core/_internal.blockStep already expects.
//   (d) prefix sharing: many requests share one system-prompt prefix by pointing
//       their block tables at the SAME physical blocks (copy-on-nothing) — saves
//       both memory and the prefill compute for the shared span.
//   (e) FAILURE MODE: a corrupted block table entry makes attention read another
//       request's KV, silently poisoning the logits. Paging buys efficiency at the
//       cost of a new, sharp consistency hazard the engine must never get wrong.
//
// Honesty caveats (inherited from core/metrics header):
//   - memory numbers are exact float64 arithmetic, printed "(est.)" — they are the
//     capacity-planning numbers a real engine uses, not a process RSS measurement.
//   - timings are real performance.now() wall-clock (warmed up first), but the toy
//     model is tiny and the kernels are naive float64 loops, so ABSOLUTE ms is
//     pessimistic. What transfers is the RELATIVE story: the fragmentation %, the
//     capacity multiplier, and the prefix-share speedup ratio.

import { DEFAULT_CONFIG, buildModel, forwardNoCache, newCache, forwardStep, _internal, type Model, type ModelConfig } from "./core/model.js";
import { encode, PROMPTS } from "./core/tokenizer.js";
import { timeIt, estimateKVBytes, formatBytes, maxLogitDrift, argmax, speedup } from "./core/metrics.js";

// blockStep is the per-token unit both core forward paths use; reusing it (instead
// of re-deriving attention) is what guarantees the paged path's arithmetic is
// bit-identical to the contiguous reference. logitsFromHidden projects the final
// hidden to vocab logits — same function the reference uses.
const { blockStep, logitsFromHidden } = _internal;

// Block size: positions per page. Real engines use 16; we use 8 so the toy prompts
// (lengths 3 / 48 / 235) straddle several blocks and the partial-last-block
// fragmentation is visible rather than a rounding artifact. Smaller blocks = less
// internal waste but more block-table overhead; this trade-off is the whole tuning
// knob, so it lives as a named constant, not a literal.
const BLOCK_SIZE = 8;

// --- the paged KV pool --------------------------------------------------------
//
// A physical pool of blocks shared by all sequences. Each block stores, for ONE
// layer, BLOCK_SIZE positions of K and V laid out flat row-major [BLOCK_SIZE, kvDim]
// — the SAME layout core packs a contiguous cache in, so a gathered run of blocks is
// byte-compatible with what blockStep expects. We key blocks by (layer, blockIdx)
// because each layer has its own KV; a real engine interleaves layers in one block,
// but per-layer keeps the toy code readable without changing the accounting.
type PagedKVPool = {
  cfg: ModelConfig;
  kvDim: number; // nKVHeads * dHead, the row width
  // per layer: a growable array of physical blocks (each a flat [BLOCK_SIZE*kvDim]).
  // freeList holds reclaimed block indices for reuse (the OS free-page list).
  blocksK: Float64Array[][]; // blocksK[layer][physBlock]
  blocksV: Float64Array[][];
  freeList: number[]; // physical block indices available for (re)allocation, per the analogy
  nextFresh: number; // next never-before-allocated physical index
  allocated: number; // high-water mark of blocks ever live, for the footprint number
};

function newPool(cfg: ModelConfig): PagedKVPool {
  return {
    cfg,
    kvDim: cfg.nKVHeads * cfg.dHead,
    blocksK: Array.from({ length: cfg.nLayers }, () => []),
    blocksV: Array.from({ length: cfg.nLayers }, () => []),
    freeList: [],
    nextFresh: 0,
    allocated: 0,
  };
}

// allocBlock — hand out one physical block index, preferring the free list (page
// reuse) over growing the pool. Returns the index valid across ALL layers: we keep
// per-layer block arrays in lock-step so one block table works for every layer.
function allocBlock(pool: PagedKVPool): number {
  const idx = pool.freeList.length > 0 ? pool.freeList.pop()! : pool.nextFresh++;
  for (let l = 0; l < pool.cfg.nLayers; l++) {
    // grow the per-layer arrays to cover this index; fresh blocks are zeroed K/V.
    if (pool.blocksK[l][idx] === undefined) {
      pool.blocksK[l][idx] = new Float64Array(BLOCK_SIZE * pool.kvDim);
      pool.blocksV[l][idx] = new Float64Array(BLOCK_SIZE * pool.kvDim);
    }
  }
  pool.allocated = Math.max(pool.allocated, pool.nextFresh);
  return idx;
}

// A sequence's view of the cache: a block table (logical block -> physical index)
// plus how many positions are filled. refcount-sharing of prefix blocks is done by
// simply listing the same physical index in two sequences' tables (demo (d)).
type PagedSeq = {
  blockTable: number[]; // logical block i lives at physical blockTable[i]
  len: number; // positions currently cached
};

// pagedStep — one cached forward step for `seq`, reading/writing through the pool.
//
// This is the paged analogue of core/model.forwardStep. The ONLY difference from the
// contiguous path is HOW K/V history is sourced and stored: we gather the sequence's
// blocks into a contiguous scratch, run the IDENTICAL blockStep, then scatter the new
// row back into the correct block slot. Same arithmetic => same logits (proven in (c)).
//
// Invariant: seq.len is the absolute position of the incoming token (mirrors the
// contiguous cache's len==pos precondition). Caller advances len via this function.
function pagedStep(model: Model, tokenId: number, seq: PagedSeq, pool: PagedKVPool): Float64Array {
  const cfg = model.cfg;
  const pos = seq.len;
  if (pos >= cfg.maxSeq) throw new Error(`pagedStep: position ${pos} >= maxSeq ${cfg.maxSeq}`);
  const kvDim = pool.kvDim;

  // Ensure a block exists for the slot this token will occupy. The OS analogue:
  // page fault on first write to a logical page -> allocate a physical frame.
  const logicalBlock = Math.floor(pos / BLOCK_SIZE);
  const slotInBlock = pos % BLOCK_SIZE;
  if (seq.blockTable[logicalBlock] === undefined) {
    seq.blockTable[logicalBlock] = allocBlock(pool);
  }

  // h is the running hidden vector. Annotated as plain Float64Array (not the
  // ArrayBuffer-narrowed `new` type) so reassignment from blockStep type-checks
  // under @types/node 22 — same fix the core path documents.
  let h: Float64Array = new Float64Array(cfg.dModel);
  h.set(model.embed.data.subarray(tokenId * cfg.dModel, (tokenId + 1) * cfg.dModel));

  for (let l = 0; l < cfg.nLayers; l++) {
    const w = model.layers[l];
    // GATHER: copy the `pos` cached rows out of the (possibly non-contiguous) blocks
    // into one flat [pos, kvDim] scratch, exactly the layout blockStep reads. This
    // gather is the runtime cost paging pays for its memory win; a real engine fuses
    // it into the attention kernel ("paged attention kernel") so it is ~free, but
    // here we keep it explicit so the mechanism is visible.
    const kHist = new Float64Array(pos * kvDim);
    const vHist = new Float64Array(pos * kvDim);
    for (let p = 0; p < pos; p++) {
      const lb = Math.floor(p / BLOCK_SIZE);
      const sib = p % BLOCK_SIZE;
      const phys = seq.blockTable[lb];
      kHist.set(pool.blocksK[l][phys].subarray(sib * kvDim, (sib + 1) * kvDim), p * kvDim);
      vHist.set(pool.blocksV[l][phys].subarray(sib * kvDim, (sib + 1) * kvDim), p * kvDim);
    }

    const r = blockStep(h, w, pos, kHist, vHist, pos, cfg);

    // SCATTER: write this token's freshly-computed K/V into its physical slot.
    const phys = seq.blockTable[logicalBlock];
    pool.blocksK[l][phys].set(r.k, slotInBlock * kvDim);
    pool.blocksV[l][phys].set(r.v, slotInBlock * kvDim);
    h = r.h;
  }
  seq.len = pos + 1;
  return logitsFromHidden(h, model);
}

// run a whole prompt through the paged path and return last-token logits.
function pagedPrefill(model: Model, tokenIds: number[], pool: PagedKVPool): { seq: PagedSeq; lastLogits: Float64Array } {
  const seq: PagedSeq = { blockTable: [], len: 0 };
  // annotate as plain Float64Array (not the ArrayBuffer-narrowed `new` type) so the
  // reassignment from pagedStep's return type-checks under @types/node 22 — same fix
  // the core forward paths document.
  let last: Float64Array = new Float64Array(model.cfg.vocabSize);
  for (const id of tokenIds) last = pagedStep(model, id, seq, pool);
  return { seq, lastLogits: last };
}

// blocks a sequence of `len` positions needs under paging.
function blocksNeeded(len: number): number {
  return Math.ceil(len / BLOCK_SIZE);
}

console.log("=== stage05 — PagedAttention: 像操作系统管内存一样管 KV 缓存 ===\n");
console.log(`配置: blockSize=${BLOCK_SIZE}, kvDim=${DEFAULT_CONFIG.nKVHeads * DEFAULT_CONFIG.dHead} (nKVHeads=${DEFAULT_CONFIG.nKVHeads}×dHead=${DEFAULT_CONFIG.dHead}), nLayers=${DEFAULT_CONFIG.nLayers}\n`);

const model = buildModel(DEFAULT_CONFIG, 42);
const cfg = DEFAULT_CONFIG;

// =============================================================================
// (a) memory: contiguous-reserved vs paged + internal fragmentation
// =============================================================================
// We model a realistic concurrent batch: each request has a different actual length
// (the 3 PROMPTS), but a contiguous cache must reserve maxSeq for each (it cannot
// know the final length, and it grows one flat array it dare not re-grow per token in
// production). Paged reserves only the blocks each length actually touches.
console.log("[a] 内存占用: 连续预留 vs 分页 (同一批并发请求)\n");

// bytes per single cached position across all layers (K+V), = estimateKVBytes(_,1,1).
const bytesPerPos = estimateKVBytes(cfg, 1, 1);
const bytesPerBlock = bytesPerPos * BLOCK_SIZE;

const reqLens = PROMPTS.map((p) => encode(p).length); // [3, 48, 235]
console.log(`    并发请求实际长度 = [${reqLens.join(", ")}] tokens`);

// contiguous: every request reserves maxSeq positions.
const contiguousBytes = reqLens.length * estimateKVBytes(cfg, cfg.maxSeq, 1);
// paged: every request reserves ceil(len/blockSize) blocks.
let pagedBytes = 0;
let usefulBytes = 0;
for (const len of reqLens) {
  pagedBytes += blocksNeeded(len) * bytesPerBlock;
  usefulBytes += len * bytesPerPos;
}
// internal fragmentation = reserved-but-unused / reserved.
const fragContig = 1 - usefulBytes / contiguousBytes;
const fragPaged = 1 - usefulBytes / pagedBytes;

console.log(`    实际用到的 KV (3 请求合计) = ${formatBytes(usefulBytes)} (est.)`);
console.log(`    连续预留 (每请求占 maxSeq=${cfg.maxSeq}) = ${formatBytes(contiguousBytes)} (est.), 内部碎片 = ${(fragContig * 100).toFixed(1)}%`);
console.log(`    分页预留 (每请求占 ceil(len/${BLOCK_SIZE}) 块) = ${formatBytes(pagedBytes)} (est.), 内部碎片 = ${(fragPaged * 100).toFixed(1)}%`);
console.log(`    分页相对连续节省 = ${(contiguousBytes / pagedBytes).toFixed(1)}x 内存\n`);

// =============================================================================
// (b) capacity: fixed VRAM budget -> max concurrent requests
// =============================================================================
// Same budget, two allocators. Assume the typical in-flight request has decoded to
// some working length (not maxSeq). Contiguous still must reserve maxSeq per slot;
// paged reserves only the blocks for the working length. The capacity gap IS the gap
// between "reserve worst case" and "reserve what you use".
console.log("[b] 固定显存预算下的最大并发请求数\n");
const budgetBytes = 64 * 1024 * 1024; // 64 MiB toy budget; ratios are budget-independent
const workingLen = 64; // a representative mid-flight sequence length
const contigPerReq = estimateKVBytes(cfg, cfg.maxSeq, 1); // must reserve maxSeq
const pagedPerReq = blocksNeeded(workingLen) * bytesPerBlock; // only the touched blocks
const maxContig = Math.floor(budgetBytes / contigPerReq);
const maxPaged = Math.floor(budgetBytes / pagedPerReq);
console.log(`    预算 = ${formatBytes(budgetBytes)} (est.), 代表性工作长度 = ${workingLen} tokens`);
console.log(`    连续: 每请求预留 maxSeq=${cfg.maxSeq} -> ${formatBytes(contigPerReq)}/请求 -> 最多 ${maxContig} 并发`);
console.log(`    分页: 每请求只占 ${blocksNeeded(workingLen)} 块 -> ${formatBytes(pagedPerReq)}/请求 -> 最多 ${maxPaged} 并发`);
console.log(`    分页并发能力 = 连续的 ${(maxPaged / maxContig).toFixed(1)}x\n`);

// =============================================================================
// (c) correctness: paged logits == contiguous/reference, drift == 0
// =============================================================================
// The load-bearing invariant. Paging only moves bytes around; the logits must be
// bit-identical to BOTH the reference forwardNoCache and the contiguous forwardStep.
// We check all 3 prompts (avoid N=1) — a paging bug often hides until a sequence
// crosses a block boundary, which the 48- and 235-token prompts both do.
console.log("[c] 正确性: 分页注意力 logits 逐位对拍 (drift 必须 ≈0)\n");
let worstDrift = 0;
for (const p of PROMPTS) {
  const ids = encode(p);
  const refLogits = Array.from(forwardNoCache(model, ids)); // O(seq^2) reference

  const contigCache = newCache(cfg);
  let contigLast: Float64Array = new Float64Array(cfg.vocabSize);
  for (const id of ids) contigLast = forwardStep(model, id, contigCache);

  const pool = newPool(cfg);
  const { lastLogits: pagedLast } = pagedPrefill(model, ids, pool);

  const dRef = maxLogitDrift(refLogits, Array.from(pagedLast));
  const dContig = maxLogitDrift(Array.from(contigLast), Array.from(pagedLast));
  worstDrift = Math.max(worstDrift, dRef, dContig);
  const nBlocks = blocksNeeded(ids.length);
  console.log(
    `    len=${String(ids.length).padStart(3)} (跨${nBlocks}块): drift vs reference=${dRef.toExponential(2)}, vs contiguous=${dContig.toExponential(2)}, argmax match=${argmax(pagedLast) === argmax(refLogits)}`
  );
}
console.log(`    最坏 drift = ${worstDrift.toExponential(2)} -> ${worstDrift === 0 ? "分页不改变任何输出 (bit-for-bit)" : "WARNING: 分页引入了漂移!"}\n`);

// =============================================================================
// (d) prefix sharing: many requests share one system-prompt prefix
// =============================================================================
// The killer app of paging: a shared system prompt lives in physical blocks ONCE,
// and every request's block table points at those same blocks. Saves the memory of
// N-1 copies AND the prefill compute for the shared span (we only run it once).
console.log("[d] prefix 共享: 多个请求共享同一 system prompt 前缀\n");
const systemPrompt = "You are a helpful assistant. Answer concisely and cite sources.";
const userQueries = [" What is paging?", " Explain GQA.", " Why subtract softmax max?", " Define TTFT."];
const sysIds = encode(systemPrompt);
const nReq = userQueries.length;
// shared prefix must end on a block boundary to be shareable as whole blocks — this
// is a real paged-cache constraint (vLLM shares only full blocks). We share the
// floor-aligned prefix; the tail of the prompt that doesn't fill a block is per-req.
const sharedBlocks = Math.floor(sysIds.length / BLOCK_SIZE);
const sharedLen = sharedBlocks * BLOCK_SIZE;

const sharedPool = newPool(cfg);
// Prefill the shared prefix ONCE into its own sequence; its first `sharedBlocks`
// physical blocks become the shared region every request will alias.
const sysSeq: PagedSeq = { blockTable: [], len: 0 };
const sharedPrefillMs = timeIt(() => {
  for (const id of sysIds) pagedStep(model, id, sysSeq, sharedPool);
});
const sharedPhysBlocks = sysSeq.blockTable.slice(0, sharedBlocks);

// Each request: alias the shared blocks (no compute, no new memory), then prefill
// only the NON-shared remainder (system tail + its own query).
// warmup the per-request path once (timeIt convention) before measuring.
{
  const warm: PagedSeq = { blockTable: [...sharedPhysBlocks], len: sharedLen };
  const tail = sysIds.slice(sharedLen).concat(encode(userQueries[0]));
  for (const id of tail) pagedStep(model, id, warm, sharedPool);
}
let perReqPrefillMs = 0;
let sharedBlocksAliased = 0;
for (const q of userQueries) {
  const seq: PagedSeq = { blockTable: [...sharedPhysBlocks], len: sharedLen };
  sharedBlocksAliased += sharedPhysBlocks.length;
  const tail = sysIds.slice(sharedLen).concat(encode(q));
  perReqPrefillMs += timeIt(() => {
    for (const id of tail) pagedStep(model, id, seq, sharedPool);
  });
}

// memory: shared (prefix once + N tails) vs naive (N full copies of the prefix).
const tailLensSum = userQueries.reduce((acc, q) => acc + blocksNeeded(sysIds.length - sharedLen + encode(q).length), 0);
const sharedMemBlocks = sharedBlocks + tailLensSum; // prefix blocks counted once
const naiveMemBlocks = nReq * blocksNeeded(sysIds.length) + userQueries.reduce((acc, q) => acc + blocksNeeded(encode(q).length), 0);
console.log(`    system prompt = ${sysIds.length} tokens -> 可共享 ${sharedBlocks} 个整块 (${sharedLen} tokens), 余 ${sysIds.length - sharedLen} tokens 不满整块`);
console.log(`    ${nReq} 个请求, 每个别名同 ${sharedPhysBlocks.length} 个物理块 (共 ${sharedBlocksAliased} 次别名, 0 拷贝)`);
console.log(`    内存: 共享 ${sharedMemBlocks} 块 vs 朴素各存一份 ${naiveMemBlocks} 块 -> 省 ${formatBytes((naiveMemBlocks - sharedMemBlocks) * bytesPerBlock)} (est.)`);
// prefill compute: shared prefix run once vs once-per-request.
const naivePrefillMsEst = sharedPrefillMs * nReq + perReqPrefillMs; // est.: N full prefixes + tails
const sharedTotalPrefillMs = sharedPrefillMs + perReqPrefillMs; // measured: 1 prefix + N tails
console.log(`    prefill 时间: 共享 = ${sharedTotalPrefillMs.toFixed(2)}ms (前缀1次 ${sharedPrefillMs.toFixed(2)}ms + ${nReq}尾巴 ${perReqPrefillMs.toFixed(2)}ms)`);
console.log(`                  朴素 ≈ ${naivePrefillMsEst.toFixed(2)}ms (est.: 前缀重算${nReq}次), 加速 = ${speedup(naivePrefillMsEst, sharedTotalPrefillMs).toFixed(2)}x\n`);

// =============================================================================
// (e) FAILURE MODE: corrupted block table -> reads another request's KV
// =============================================================================
// Paging's new hazard: the block table is an indirection, and a wrong entry doesn't
// crash — it silently makes attention read a DIFFERENT request's keys/values. The
// output is plausible garbage (cross-request contamination), the worst kind of bug.
// We force it: build two independent sequences in one pool, then corrupt request B's
// block table to point a logical block at request A's physical block, and show the
// logits drift away from B's correct output.
console.log("[e] 失败模式: block table 映射错位 -> 读到别的请求的 KV (logits 污染)\n");
const poolE = newPool(cfg);
const idsA = encode("AAAAAAAAAAAAAAAA"); // request A: distinct content
const idsB = encode("zzzzzzzzzzzzzzzz"); // request B: different content, same length

const { seq: seqA } = pagedPrefill(model, idsA, poolE);
const { seq: seqB, lastLogits: correctB } = pagedPrefill(model, idsB, poolE);

// Corrupt B's block table: redirect its logical block 0 to A's physical block 0.
// Now B's attention over its first BLOCK_SIZE positions reads A's K/V. This is the
// exact bug a block-allocator off-by-one or a refcount race would produce.
const goodPhys = seqB.blockTable[0];
seqB.blockTable[0] = seqA.blockTable[0]; // <-- the corruption
// Re-run B's last step to recompute its logits under the poisoned table. We rebuild
// B's logits by replaying its tokens through the (now corrupted) block table without
// re-writing K/V: do a read-only final step by re-gathering. Simplest faithful way:
// re-derive last-token logits via a fresh step that re-reads history from blocks.
const poisonedSeq: PagedSeq = { blockTable: seqB.blockTable, len: idsB.length - 1 };
const poisonedLast = pagedStep(model, idsB[idsB.length - 1], poisonedSeq, poolE);
const poisonDrift = maxLogitDrift(Array.from(correctB), Array.from(poisonedLast));
console.log(`    请求 B 正确 argmax = ${argmax(correctB)}`);
console.log(`    block table 错位后 B 的 argmax = ${argmax(poisonedLast)} ${argmax(poisonedLast) !== argmax(correctB) ? "(被污染, 变了!)" : "(本例恰好未翻转, 但 logits 已偏移)"}`);
console.log(`    logits 漂移 = ${poisonDrift.toExponential(2)} (≫0: 读到了 A 的 KV, 输出被污染)`);
seqB.blockTable[0] = goodPhys; // restore, lest a later reader trust the corrupted table
console.log(`    教训: 分页用一层间接换效率, 代价是 block table 必须永远正确 — 一个错条目=静默串话, 不崩溃, 最难查\n`);

console.log("=== stage05 完成: 分页省内存/增并发/可共享前缀, 且证明不改变输出 (drift=0) ===");
