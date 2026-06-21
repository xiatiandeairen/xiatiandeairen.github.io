// stage04-continuous-batch.ts — 连续批处理：让推理引擎不空转，把吞吐榨干。
//
// 问题: 一批请求的生成长度方差极大（一个回 "yes"，一个写一篇长文）。怎么调度
// 它们共享同一份模型权重，决定了 GPU 是满载还是大半时间在等最慢那个请求。
//
// 两种策略，本 stage 用 REAL forwardStep 实测对照:
//
//   STATIC batching（朴素）: 一批请求一起开跑，整批必须等到 *最长* 的那个生成完才
//     释放。短请求早早算完，但它的 slot 在原地空转直到全批结束 —— 这就是 head-of-
//     line blocking（队头阻塞）: 一个长请求把同批所有短请求的完成延迟一起拖到它的
//     长度。利用率 = 每步实际还在算的 slot 数 / batch 容量，随长请求拖尾断崖式下跌。
//
//   CONTINUOUS batching（vLLM/Orca 的做法）: 调度单位是 *一个 decode step*，不是
//     "一整批"。每步结束: 算完的序列立刻离场释放 slot，等待队列里的新请求立刻补位。
//     于是每一步 batch 都尽量填满，短请求一算完就走、不被长请求绑架。
//
// 不变量（本 stage 的 load-bearing 断言）: 两种策略对 *同一个请求* 产出 bit-for-bit
//   相同的 token 序列。调度只改"谁和谁同一步算"，绝不改单个序列的数学。我们用
//   maxLogitDrift==0 / token 序列逐位相等来证明，否则任何吞吐数字都是耍流氓。
//
// 诚实数字: 模型是 toy（synthetic 权重、float64 标量 kernel），所以绝对 tok/s 偏
//   悲观。可迁移的是 *相对* 量: 连续批 vs 静态批的吞吐加速比、利用率差、短请求 p99
//   的改善 —— 这些比值在真实引擎上同样成立。所有时间都是 performance.now wall-clock。
//
// 这里没有真 GPU，"利用率"指 *算子有效占用率*: 每个 decode step 我们真正执行了多少
//   个 active slot 的 forwardStep，除以 batch 容量。真实引擎里这正比于 GPU 利用率
//   （padding 出来的空 slot 在真硬件上要么浪费算力、要么靠 ragged kernel 省掉）。

import { DEFAULT_CONFIG, buildModel, forwardStep, newCache, type KVCache, type Model } from "./core/model.js";
import { encode, PROMPTS } from "./core/tokenizer.js";
import {
  timeIt,
  tokensPerSecond,
  estimateKVBytes,
  formatBytes,
  speedup,
  maxLogitDrift,
  argmax,
} from "./core/metrics.js";

const cfg = DEFAULT_CONFIG;
const model: Model = buildModel(cfg, 42);

// A single inference request. genLen is how many tokens it wants to decode — the
// quantity whose VARIANCE is the whole point of this chapter. promptIds drives a
// real prefill so timings and KV growth are honest, not synthetic counters.
type Request = {
  id: number;
  promptIds: number[];
  genLen: number; // target number of decode steps for this request
};

// Live decode state for one request occupying a batch slot. cache holds its KV
// history; produced is the tokens generated so far; nextToken is the input for the
// upcoming step (argmax-greedy so the run is deterministic).
type Slot = {
  req: Request;
  cache: KVCache;
  produced: number[];
  nextToken: number;
};

// --- workload: high-variance generation lengths -------------------------------
//
// The classic bimodal traffic an inference server sees: a few long generations
// mixed with many short ones. Static batching suffers exactly here; uniform lengths
// would hide the problem (and be an N=1-style lie about real traffic).
function makeWorkload(): Request[] {
  const genLens = [4, 40, 8, 32, 6, 36, 5, 30]; // mean 20, but spread 4..40 (10x)
  return genLens.map((genLen, i) => ({
    id: i,
    // alternate the short(3) and medium(48) prompts so prefill cost varies too. We
    // avoid the 235-token prompt here: 235 + genLen(≤40) would exceed maxSeq(256),
    // and KV growth is studied in stage05 — this chapter is about scheduling.
    promptIds: encode(PROMPTS[i % 2]),
    genLen,
  }));
}

// Prefill a fresh slot: run the prompt through the cached path so its KV is warm,
// and seed nextToken from the prompt's last-token logits. This is real work (one
// forwardStep per prompt token), shared identically by both schedulers — so any
// throughput difference comes from SCHEDULING, not from a kernel change.
function admit(req: Request): Slot {
  const cache = newCache(cfg);
  // annotate as plain Float64Array (not the ArrayBuffer-narrowed `new` type) so the
  // reassignment from forwardStep type-checks under @types/node 22 — same fix the
  // core applies in forwardStep itself.
  let last: Float64Array = new Float64Array(cfg.vocabSize);
  for (const id of req.promptIds) last = forwardStep(model, id, cache);
  return { req, cache, produced: [], nextToken: argmax(last) };
}

// Execute ONE useful decode step for ONE slot: run the model, append the token.
// Mutates the slot's cache (KV grows one row). This is the atomic unit of useful
// work; both schedulers are built from it.
function decodeOneStep(slot: Slot): void {
  const logits = forwardStep(model, slot.nextToken, slot.cache);
  slot.nextToken = argmax(logits);
  slot.produced.push(slot.nextToken);
}

// padStep — the WASTE of a static batch, modeled honestly.
//
// Why this exists: a GPU runs one batched matmul over the FULL batch width. A slot
// whose request already finished is not free — unless you use ragged/variable-length
// kernels (exactly the complexity continuous batching buys), the dead slot is PADDED
// and its rows still flow through every matmul. So a static step costs `capacity`
// forward passes regardless of how many slots are still alive. We reproduce that
// real cost by burning one genuine forward pass on a throwaway scratch slot. Without
// this, our scalar toy would skip dead slots and falsely show static == continuous
// in wall-clock; padStep is what makes the wall-clock speedup honest rather than a
// step-count artifact.
const scratchCache: KVCache = (() => {
  const c = newCache(cfg);
  forwardStep(model, 0, c); // 1 cached row so subsequent steps are O(1)-ish, like a live slot
  return c;
})();
function padStep(): void {
  // reset scratch when it nears maxSeq so the padded compute stays representative of
  // a mid-decode step and never throws on the position bound.
  if (scratchCache.len >= cfg.maxSeq - 1) {
    const fresh = newCache(cfg);
    forwardStep(model, 0, fresh);
    scratchCache.k = fresh.k;
    scratchCache.v = fresh.v;
    scratchCache.len = fresh.len;
  }
  forwardStep(model, 0, scratchCache); // real compute, discarded — the padding cost
}

type RunResult = {
  label: string;
  wallMs: number;
  totalTokens: number; // useful tokens produced (excludes padded waste)
  totalSteps: number; // number of global scheduler steps taken
  busySlotSteps: number; // sum over steps of active-slot count (utilization numerator)
  capacitySteps: number; // sum over steps of capacity (utilization denominator)
  trajectory: string[]; // per-step textual snapshot of slot occupancy
  completionStep: Map<number, number>; // reqId -> global step at which it finished (arrival t=0)
  tokensByReq: Map<number, number[]>; // reqId -> produced tokens (for equivalence check)
};

// --- STATIC batching ----------------------------------------------------------
//
// Fill the batch once, run until EVERY slot's request is done, then (and only then)
// admit the next batch. A slot whose request finished early goes idle but is NOT
// refilled until the whole batch drains — and it still pays a padded forward pass
// every step (padStep). That padded waste is the defining cost of static batching.
//
// Latency is recorded as the GLOBAL step at finish. All requests arrive at t=0, so
// a request stuck behind an earlier batch carries that whole queue wait in its
// completion step — this is the head-of-line blocking the chapter is about.
function runStatic(requests: Request[], capacity: number): RunResult {
  const trajectory: string[] = [];
  const completionStep = new Map<number, number>();
  const tokensByReq = new Map<number, number[]>();
  let totalTokens = 0;
  let busySlotSteps = 0;
  let capacitySteps = 0;
  let globalStep = 0;
  const queue = [...requests];

  const wallMs = timeIt(() => {
    while (queue.length > 0) {
      // form one static batch
      const batch: (Slot | null)[] = [];
      while (batch.length < capacity && queue.length > 0) {
        batch.push(admit(queue.shift()!));
      }
      // run until ALL slots in this batch are finished
      let anyActive = true;
      while (anyActive) {
        anyActive = false;
        let activeThisStep = 0;
        const marks: string[] = [];
        for (let s = 0; s < capacity; s++) {
          const slot = batch[s] ?? null;
          if (slot && slot.produced.length < slot.req.genLen) {
            decodeOneStep(slot);
            totalTokens++;
            activeThisStep++;
            anyActive = true;
            marks.push(`R${slot.req.id}`);
            if (slot.produced.length === slot.req.genLen) {
              completionStep.set(slot.req.id, globalStep + 1); // 1-based: arrival t=0
              tokensByReq.set(slot.req.id, slot.produced);
            }
          } else if (slot) {
            // finished-but-not-evicted slot: pays a PADDED forward pass (real GPU
            // cost) while producing nothing. This is the waste continuous removes.
            padStep();
            marks.push("··");
          } else {
            marks.push("--");
          }
        }
        if (anyActive) {
          busySlotSteps += activeThisStep;
          capacitySteps += capacity;
          if (trajectory.length < 24) {
            trajectory.push(`step ${String(globalStep).padStart(3)} [${marks.join(" ")}] active=${activeThisStep}/${capacity}`);
          }
          globalStep++;
        }
      }
    }
  });

  return {
    label: `static (cap=${capacity})`,
    wallMs,
    totalTokens,
    totalSteps: globalStep,
    busySlotSteps,
    capacitySteps,
    trajectory,
    completionStep,
    tokensByReq,
  };
}

// --- CONTINUOUS batching ------------------------------------------------------
//
// The scheduler's unit is a step, not a batch. Each step: run every active slot
// once; then evict finished slots and immediately backfill empty slots from the
// queue. The batch composition churns step-to-step — that churn is exactly what
// keeps utilization near 100% and lets a short request leave the instant it's done.
function runContinuous(requests: Request[], capacity: number): RunResult {
  const trajectory: string[] = [];
  const completionStep = new Map<number, number>();
  const tokensByReq = new Map<number, number[]>();
  let totalTokens = 0;
  let busySlotSteps = 0;
  let capacitySteps = 0;
  let globalStep = 0;
  const queue = [...requests];
  const slots: (Slot | null)[] = new Array(capacity).fill(null);

  const wallMs = timeIt(() => {
    // seed: fill all slots up front
    for (let s = 0; s < capacity && queue.length > 0; s++) {
      slots[s] = admit(queue.shift()!);
    }
    while (slots.some((x) => x !== null)) {
      let activeThisStep = 0;
      const marks: string[] = [];
      for (let s = 0; s < capacity; s++) {
        const slot = slots[s];
        if (slot) {
          decodeOneStep(slot);
          totalTokens++;
          activeThisStep++;
          marks.push(`R${slot.req.id}`);
        } else {
          // an empty slot is NOT padded here: continuous batching only ever leaves a
          // slot empty when the queue is fully drained (nothing left to backfill), so
          // there is no waste to model — that absence of padding IS the optimization.
          marks.push("--");
        }
      }
      busySlotSteps += activeThisStep;
      capacitySteps += capacity;
      if (trajectory.length < 24) {
        trajectory.push(`step ${String(globalStep).padStart(3)} [${marks.join(" ")}] active=${activeThisStep}/${capacity}`);
      }
      // evict finished, backfill from queue — the continuous part.
      for (let s = 0; s < capacity; s++) {
        const slot = slots[s];
        if (slot && slot.produced.length >= slot.req.genLen) {
          completionStep.set(slot.req.id, globalStep + 1); // 1-based: arrival t=0
          tokensByReq.set(slot.req.id, slot.produced);
          // admit the next queued request into this freed slot THIS same step.
          slots[s] = queue.length > 0 ? admit(queue.shift()!) : null;
        }
      }
      globalStep++;
    }
  });

  return {
    label: `continuous (cap=${capacity})`,
    wallMs,
    totalTokens,
    totalSteps: globalStep,
    busySlotSteps,
    capacitySteps,
    trajectory,
    completionStep,
    tokensByReq,
  };
}

// percentile over an array of latencies (steps), nearest-rank, sorted ascending.
function pct(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
}

function utilization(r: RunResult): number {
  return r.capacitySteps === 0 ? 0 : r.busySlotSteps / r.capacitySteps;
}

// ============================================================================
console.log("=== stage04 连续批处理 (continuous batching) ===\n");

const CAPACITY = 4;
const workload = makeWorkload();
console.log(`workload: ${workload.length} 个请求, 生成长度 = [${workload.map((r) => r.genLen).join(", ")}]`);
console.log(`          (mean=${(workload.reduce((a, r) => a + r.genLen, 0) / workload.length).toFixed(1)}, min=${Math.min(...workload.map((r) => r.genLen))}, max=${Math.max(...workload.map((r) => r.genLen))} — 10x 方差)`);
console.log(`batch 容量 = ${CAPACITY} slots\n`);

// warmup (metrics.timeIt convention: first run pays JIT/cold-cache cost)
runStatic(makeWorkload(), CAPACITY);
runContinuous(makeWorkload(), CAPACITY);

const stat = runStatic(makeWorkload(), CAPACITY);
const cont = runContinuous(makeWorkload(), CAPACITY);

// --- (0) equivalence: scheduling must not change any request's output ---------
console.log("[0] 不变量 — 调度不改单序列数学 (per-request token equivalence):");
let maxDrift = 0;
let allMatch = true;
for (const req of workload) {
  const a = stat.tokensByReq.get(req.id)!;
  const b = cont.tokensByReq.get(req.id)!;
  const same = a.length === b.length && a.every((t, i) => t === b[i]);
  if (!same) allMatch = false;
  maxDrift = Math.max(maxDrift, maxLogitDrift(a, b));
}
console.log(`    所有 ${workload.length} 个请求: static 与 continuous 产出 token 序列逐位相等 = ${allMatch}`);
console.log(`    token-id maxDrift = ${maxDrift.toExponential(2)} (0 => 调度纯粹是编排, 数学不变)\n`);

// --- (a) throughput + utilization ---------------------------------------------
console.log("[a] 整体吞吐 + 算子有效利用率 (real wall-clock):");
const statTps = tokensPerSecond(stat.totalTokens, stat.wallMs);
const contTps = tokensPerSecond(cont.totalTokens, cont.wallMs);
console.log(`    static     : ${stat.totalTokens} tok in ${stat.wallMs.toFixed(1)} ms over ${stat.totalSteps} steps`);
console.log(`                 throughput = ${statTps.toFixed(1)} tok/s   利用率 = ${(utilization(stat) * 100).toFixed(1)}% (busy ${stat.busySlotSteps}/${stat.capacitySteps} slot-steps)`);
console.log(`    continuous : ${cont.totalTokens} tok in ${cont.wallMs.toFixed(1)} ms over ${cont.totalSteps} steps`);
console.log(`                 throughput = ${contTps.toFixed(1)} tok/s   利用率 = ${(utilization(cont) * 100).toFixed(1)}% (busy ${cont.busySlotSteps}/${cont.capacitySteps} slot-steps)`);
console.log(`    speedup (continuous / static) = ${speedup(stat.wallMs, cont.wallMs).toFixed(2)}x wall-clock, ${(contTps / statTps).toFixed(2)}x tok/s`);
console.log(`    (绝对 tok/s 偏悲观 — toy 模型 + 标量 kernel; 可迁移的是利用率差与加速比)\n`);

// --- (b) per-request completion latency, p50/p99 ------------------------------
//
// Measured in SCHEDULER STEPS from ARRIVAL (all requests arrive at t=0), not wall-
// clock, because the head-of-line story is about *position in the schedule*, not
// kernel speed. The completion step includes any queue wait: a request in a later
// static batch eats the full drain time of every batch ahead of it. Under continuous
// it is admitted the instant a slot frees, so it waits far less.
console.log("[b] 每请求完成延迟 (steps from arrival t=0 to done) + 队头阻塞:");
const statLat = workload.map((r) => stat.completionStep.get(r.id)!);
const contLat = workload.map((r) => cont.completionStep.get(r.id)!);
console.log(`    request ids       : [${workload.map((r) => ("R" + r.id).padStart(3)).join(", ")}]`);
console.log(`    request genLens   : [${workload.map((r) => String(r.genLen).padStart(3)).join(", ")}]`);
console.log(`    static latency    : [${statLat.map((x) => String(x).padStart(3)).join(", ")}]  <- 第二批请求(R4..R7)含整批排队等待`);
console.log(`    continuous latency: [${contLat.map((x) => String(x).padStart(3)).join(", ")}]  <- 一有空 slot 立即补位, 等待少`);
console.log(`    static     p50=${pct(statLat, 50)} p99=${pct(statLat, 99)} max=${Math.max(...statLat)} steps`);
console.log(`    continuous p50=${pct(contLat, 50)} p99=${pct(contLat, 99)} max=${Math.max(...contLat)} steps`);
// R4 (genLen=6) lands in the 2nd static batch -> waits for batch-1's longest (R1, 40).
console.log(`    短请求 R4 (genLen=6): static 完成于第 ${stat.completionStep.get(4)} step vs continuous 第 ${cont.completionStep.get(4)} step`);
console.log(`    => 静态批的队头阻塞: 一个 6-token 请求因排在长批之后, p99 被拖到批内最长请求量级\n`);

// --- (c) continuous-batch composition trajectory ------------------------------
//
// Read the columns: each step a slot shows its request id (R<n>) or -- (empty).
// Watch slots churn — a short request leaves and a queued one takes its place mid-
// flight, which a static batch can never do.
console.log("[c] 连续批 — batch 组成随 step 变化轨迹 (R<n>=该 slot 在算请求 n, --=空):");
for (const line of cont.trajectory) console.log(`    ${line}`);
console.log(`    (slot 列在 step 间换人 = backfill 生效; 静态批同一批的列固定不变)\n`);

// Contrast: static trajectory shows the idle '··' waste once short reqs finish.
console.log("    对照 static 轨迹 (··=slot 已算完但被占住空转, 队头阻塞的浪费):");
for (const line of stat.trajectory) console.log(`    ${line}`);
console.log("");

// --- (d) FAILURE MODES --------------------------------------------------------
console.log("[d] 失败模式 — 调度器必须算两笔账: 单步延迟 与 显存:\n");

// (d.1) capacity set too large => single-step latency spike.
//
// Throughput tempts you to crank capacity. But one decode step now runs `capacity`
// forwardSteps back-to-back before ANY token is returned, so per-step latency (the
// thing every concurrent user feels as a stall) grows ~linearly with capacity. A
// real engine bounds capacity precisely to cap this tail latency.
console.log("(d.1) batch 容量设太大 => 单步延迟尖峰 (吞吐与延迟的对立):");
for (const capTest of [1, 4, 16]) {
  const reqs: Request[] = Array.from({ length: capTest }, (_, i) => ({
    id: i,
    promptIds: encode(PROMPTS[1]),
    genLen: 8,
  }));
  const slots = reqs.map((r) => admit(r));
  // warmup one step
  for (const sl of slots) decodeOneStep(sl);
  // measure ONE full scheduler step (all slots advance once) — the latency a user waits.
  const stepMs = timeIt(() => {
    for (const sl of slots) decodeOneStep(sl);
  });
  console.log(`    cap=${String(capTest).padStart(2)}: 单步全 batch = ${stepMs.toFixed(3)} ms  => 每 token 平均 ${(stepMs / capTest).toFixed(3)} ms, 但首字延迟随容量线性上升`);
}
console.log("    => 容量越大吞吐越高, 但单步(及尾延迟)随之上升; 调度器要在两者间设上限\n");

// (d.2) KV cache memory overflow => scheduler must REJECT, not OOM.
//
// Each admitted sequence costs estimateKVBytes. A scheduler that admits blindly
// will run the box out of memory mid-decode (the worst time to fail). A correct
// scheduler does capacity-planning arithmetic on admission and rejects/queues when
// the next sequence won't fit. We demonstrate the rejection, not a crash.
console.log("(d.2) KV cache 显存被并发请求撑爆 => 调度器必须拒绝, 不能 OOM:");
const KV_BUDGET_BYTES = 6 * 1024 * 1024; // 6 MiB toy budget for the KV pool
const ADMIT_SEQ_LEN = 256; // assume each admitted seq reserves maxSeq worth of KV
const perSeqBytes = estimateKVBytes(cfg, ADMIT_SEQ_LEN, 1);
console.log(`    KV 池预算 = ${formatBytes(KV_BUDGET_BYTES)} (est.), 每序列保留 seq=${ADMIT_SEQ_LEN} => ${formatBytes(perSeqBytes)} (est.)`);
let admittedBytes = 0;
let admittedCount = 0;
let rejected = 0;
for (let i = 0; i < 64; i++) {
  // admission control: only admit if the running KV total still fits the budget.
  if (admittedBytes + perSeqBytes <= KV_BUDGET_BYTES) {
    admittedBytes += perSeqBytes;
    admittedCount++;
  } else {
    rejected++;
    if (rejected === 1) {
      console.log(`    在第 ${i + 1} 个请求处拒绝: ${formatBytes(admittedBytes)} + ${formatBytes(perSeqBytes)} > ${formatBytes(KV_BUDGET_BYTES)} (est.)`);
    }
  }
}
console.log(`    => 接纳 ${admittedCount} 个请求 (占 ${formatBytes(admittedBytes)} est.), 拒绝 ${rejected} 个`);
console.log(`    理论上限 = floor(${formatBytes(KV_BUDGET_BYTES)} / ${formatBytes(perSeqBytes)}) = ${Math.floor(KV_BUDGET_BYTES / perSeqBytes)} 序列`);
console.log(`    => 调度器算显存账后主动 backpressure (排队/拒绝), 而不是跑到一半 OOM 崩掉\n`);

console.log("=== stage04 done — 调度只改编排, 不改单序列数学; 连续批用利用率换吞吐, 但容量与显存是硬上限 ===");
