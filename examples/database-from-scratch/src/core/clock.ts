// core/clock.ts — logical time (for transactions) and real-time benchmarking.
//
// Two unrelated-looking things live here because both answer "what time is it?"
// for the book, and both must be deterministic in *what they affect* while honest
// in *what they report*.
//
// LamportClock: the source of transaction timestamps and MVCC version numbers.
// Why a logical clock instead of Date.now(): MVCC visibility ("can txn T see
// version V?") is decided purely by ordering, and wall-clock time is neither
// monotonic nor reproducible. A monotonically increasing counter gives a total
// order that is identical on every run — so a concurrency anomaly the scheduler
// produces is reproducible, which is the whole point of the transaction chapters.
//
// Invariant: tick() is strictly increasing and never reused. A duplicated
// timestamp would make two versions indistinguishable to the visibility rule —
// the failure mode is a lost update that looks like correct behavior.
//
// bench(): the honesty valve. The book promises measured numbers, not vibes.
// bench runs fn `iters` times and reports REAL nanoseconds via hrtime.bigint().
// Determinism note we are careful about: the *work* is deterministic (driven by
// seeded PRNG elsewhere), but the *timing* is genuinely measured and will vary
// run-to-run — that's correct. A stage that needs a stable number reports op
// COUNTS (deterministic); a stage reporting ops/sec must label it as a real,
// machine-dependent measurement. We never fabricate a "nice" throughput.

export class LamportClock {
  // Starts at 0; first tick yields 1 so that 0 can mean "no version yet" /
  // "before all transactions", a useful sentinel for MVCC initial state.
  private counter = 0;

  /** Advance and return the new logical time. Monotonic, gap-free, unique. */
  tick(): number {
    return ++this.counter;
  }

  /** Observe a timestamp from "another node"/event and move our clock past it.
   *  This is the actual Lamport rule (max(local, received) + 1). The single-node
   *  book rarely needs it, but MVCC's "merge a committed txn's ts" uses it, and
   *  including it keeps the abstraction honest rather than a glorified counter. */
  witness(received: number): number {
    this.counter = Math.max(this.counter, received) + 1;
    return this.counter;
  }

  /** Current time without advancing — for reads that need a snapshot ts. */
  now(): number {
    return this.counter;
  }
}

export interface BenchResult {
  /** Total real wall time across all iterations, in nanoseconds. Measured. */
  totalNs: number;
  /** Iterations executed. Echoed so callers can sanity-check the divisor. */
  iters: number;
  /** Real throughput. NOT deterministic; depends on the machine. Label it. */
  opsPerSec: number;
  /** Mean nanoseconds per op. Often the more honest figure for tiny ops. */
  nsPerOp: number;
}

/** Time `fn` over `iters` calls using the highest-resolution clock Node offers.
 *  We deliberately do NOT do warmup/percentiles here — that statistical rigor is
 *  a stage-07 (benchmark) concern; core keeps the primitive minimal and truthful.
 *  The return is real measured time; treat opsPerSec as machine-dependent. */
export function bench(fn: () => void, iters: number): BenchResult {
  if (iters <= 0) throw new Error(`bench: iters must be positive, got ${iters}`);
  const start = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  const end = process.hrtime.bigint();
  const totalNs = Number(end - start);
  return {
    totalNs,
    iters,
    opsPerSec: totalNs === 0 ? Infinity : (iters / totalNs) * 1e9,
    nsPerOp: totalNs / iters,
  };
}
