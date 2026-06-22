// core/clock.ts — discrete-event virtual time (SimClock) + honest wall-clock bench.
//
// Why a virtual clock instead of Date.now()/setTimeout: a distributed system is
// defined by its TIMING — election timeouts, heartbeat intervals, message
// latency, partition-then-heal windows. If those rode real time, two problems
// follow: (1) the demo would take real seconds and (2) the interleaving would
// vary run-to-run, so a split-brain caused by a specific timeout race could
// never be reproduced. SimClock makes time a deterministic, fast-forwardable
// integer driven entirely by the event queue. "Wait 150ms" advances a counter,
// not the wall.
//
// Model: a min-heap of (fireAt, seq, fn). now() is logical milliseconds. Every
// delay/timeout in the book (Network delivery, Node.setTimer) schedules onto
// this one heap. run() drains it in time order — that single ordering IS the
// global execution order of the whole distributed system, which is why it's
// reproducible.
//
// Invariant: time never goes backward; events at the same fireAt fire in
// insertion order (FIFO via a monotonic seq tiebreak). Without the seq tiebreak,
// two events scheduled for the same ms would fire in heap-internal (effectively
// random) order and quietly break determinism — the nastiest possible failure
// mode because it only shows up as "sometimes the test fails".
//
// bench() is the honesty valve: the book promises measured numbers. bench runs
// real CPU work `iters` times and reports REAL nanoseconds via hrtime.bigint().
// The WORK is deterministic (seeded), but the TIMING is genuinely measured and
// machine-dependent — we label opsPerSec as such and never fabricate a nice one.

interface ScheduledEvent {
  fireAt: number;
  seq: number; // FIFO tiebreak for equal fireAt — guarantees deterministic order
  fn: () => void;
}

export class SimClock {
  private heap: ScheduledEvent[] = [];
  private clock = 0;
  private seqCounter = 0;

  /** Current logical time in ms. Pure read — never advances the clock. Code that
   *  needs "when did this happen" (e.g. failure detector heartbeat arrival)
   *  reads now(); it must match what the event that's running was scheduled for. */
  now(): number {
    return this.clock;
  }

  /** Schedule fn to run `delayMs` from now. delayMs=0 is legal and means "later
   *  this tick, after currently-queued work" — used for async-but-immediate
   *  message delivery so a node never observes its own send synchronously. */
  schedule(delayMs: number, fn: () => void): void {
    if (delayMs < 0) throw new Error(`schedule: negative delay ${delayMs}`);
    this.push({ fireAt: this.clock + delayMs, seq: this.seqCounter++, fn });
  }

  /** Pop and execute the single earliest event, advancing now() to its fireAt.
   *  Returns false when the queue is empty (system quiesced). Stages that want
   *  to step the simulation one event at a time (to snapshot state between
   *  events) call this in a loop instead of run(). */
  advance(): boolean {
    const ev = this.pop();
    if (!ev) return false;
    // Clamp forward only. Two events at the same fireAt keep `clock` flat; a
    // later event moves it forward. It can NEVER move back — see class invariant.
    this.clock = ev.fireAt;
    ev.fn();
    return true;
  }

  /** Drain the entire queue (run to quiescence) or until `untilMs` logical time.
   *  Returns the number of events fired — a deterministic figure stages assert
   *  on. `untilMs` is how we model "let the cluster run for 5 seconds then cut
   *  the network": run to a deadline, inject a fault, run again. */
  run(untilMs = Infinity): number {
    let fired = 0;
    while (this.heap.length > 0 && this.peekFireAt() <= untilMs) {
      this.advance();
      fired++;
    }
    // If we stopped at a deadline, snap logical time to the deadline so a
    // subsequent fault is timestamped at `untilMs`, not at the last event.
    if (untilMs !== Infinity && this.clock < untilMs) this.clock = untilMs;
    return fired;
  }

  /** Pending event count — for stages asserting the system has gone quiet
   *  (e.g. "after heal, no retransmits remain in flight"). */
  pending(): number {
    return this.heap.length;
  }

  // --- binary min-heap keyed by (fireAt, seq) ---------------------------------
  // Hand-rolled rather than sorting an array each insert: schedule/advance are
  // the hottest path in the whole simulation; O(log n) keeps long runs cheap.

  private peekFireAt(): number {
    return this.heap[0].fireAt;
  }

  private less(a: ScheduledEvent, b: ScheduledEvent): boolean {
    // Primary key fireAt, secondary key seq. The seq comparison is the
    // determinism guarantee, not a micro-optimization.
    return a.fireAt < b.fireAt || (a.fireAt === b.fireAt && a.seq < b.seq);
  }

  private push(ev: ScheduledEvent): void {
    const h = this.heap;
    h.push(ev);
    let i = h.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.less(h[i], h[parent])) break;
      [h[i], h[parent]] = [h[parent], h[i]];
      i = parent;
    }
  }

  private pop(): ScheduledEvent | undefined {
    const h = this.heap;
    if (h.length === 0) return undefined;
    const top = h[0];
    const last = h.pop()!;
    if (h.length > 0) {
      h[0] = last;
      this.siftDown(0);
    }
    return top;
  }

  private siftDown(i: number): void {
    const h = this.heap;
    const n = h.length;
    for (;;) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let smallest = i;
      if (l < n && this.less(h[l], h[smallest])) smallest = l;
      if (r < n && this.less(h[r], h[smallest])) smallest = r;
      if (smallest === i) break;
      [h[i], h[smallest]] = [h[smallest], h[i]];
      i = smallest;
    }
  }
}

export interface BenchResult {
  /** Total real wall time across all iterations, in nanoseconds. Measured. */
  totalNs: number;
  /** Iterations executed. Echoed so callers can sanity-check the divisor. */
  iters: number;
  /** Real throughput. NOT deterministic; depends on the machine. Label it. */
  opsPerSec: number;
  /** Mean nanoseconds per op — often the more honest figure for tiny ops. */
  nsPerOp: number;
}

/** Time `fn` over `iters` calls using the highest-resolution clock Node offers.
 *  No warmup/percentiles here — that statistical rigor is a stage concern; core
 *  keeps the primitive minimal and truthful. Returns REAL measured time; treat
 *  opsPerSec as machine-dependent and label it (est. relative) in any table. */
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
