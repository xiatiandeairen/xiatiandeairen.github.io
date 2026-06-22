// core/scheduler.ts — a deterministic, single-thread, cooperative scheduler that
// interleaves "transactions" so concurrency anomalies become reproducible.
//
// The problem it solves: isolation-level bugs (dirty read, lost update,
// write skew) only appear under a SPECIFIC interleaving of operations from
// concurrent transactions. With real threads that interleaving is
// non-deterministic — you can't put a write-skew in a book and have the reader
// reproduce it. So we don't use threads. Each transaction is a GENERATOR that
// `yield`s at every point where a real txn could be preempted (each read, each
// write). The scheduler holds all the suspended generators and, at each step,
// picks one to advance. The pick is driven by the book's seeded PRNG, so a given
// seed reproduces a given interleaving exactly — including the seed that triggers
// the anomaly.
//
// Mental model: a transaction body is `function* (ctx) { ... yield 'r:x'; ... }`.
// Between two yields the code runs atomically (that step is the indivisible unit).
// The yielded string is just a human-readable label for the trace, so the book
// can PRINT the exact schedule that produced an anomaly — proof, not assertion.
//
// Invariants:
//  - Determinism: same seed + same txn set => same schedule => same trace. The
//    only randomness is `rng`; introducing Math.random or Date here would break
//    reproducibility and is forbidden.
//  - Fairness is NOT guaranteed and that's intentional: the scheduler may starve
//    a txn within a run; over seeds it explores many schedules. The MVCC/txn
//    chapters sweep seeds to FIND the anomaly-producing schedule.
//
// Failure mode it surfaces (not hides): if a step throws (e.g. a txn aborts on
// conflict), the scheduler records the abort in the trace and drops that txn,
// so the chapter can show "txn B aborted under serializable, committed under
// read-committed" as concrete, replayable output.

import type { Rng } from "./prng.js";

/** A transaction is a generator. Each `yield` is a preemption point; the yielded
 *  string labels that step for the trace. The return value (if any) is captured
 *  as the txn's result. */
export type TxnGen<TCtx> = (ctx: TCtx) => Generator<string, void, void>;

export interface TraceEntry {
  /** Which transaction took this step (its index in the input array). */
  txn: number;
  /** Step counter, global across the schedule — the x-axis of the interleaving. */
  step: number;
  /** The label the txn yielded, or a lifecycle marker (start/commit/abort). */
  label: string;
}

export interface ScheduleResult {
  /** The exact interleaving that ran. Print it to SHOW the schedule. */
  trace: TraceEntry[];
  /** Per-txn outcome: 'commit' if the generator finished, 'abort' if it threw. */
  outcome: ("commit" | "abort")[];
  /** Abort reasons keyed by txn index, for the ones that aborted. */
  aborts: Record<number, string>;
}

/** Run the given transactions to completion under a PRNG-chosen interleaving.
 *
 *  @param txns  the transaction bodies (generators)
 *  @param ctx   shared context passed to every txn (e.g. the MVCC store). All
 *               txns see the SAME ctx object — that shared mutable state is
 *               exactly where anomalies live.
 *  @param rng   the book's seeded PRNG; the sole source of scheduling choice.
 */
export function runSchedule<TCtx>(
  txns: TxnGen<TCtx>[],
  ctx: TCtx,
  rng: Rng,
): ScheduleResult {
  // Materialize each generator and remember which are still runnable. We keep a
  // parallel `active` list of indices so PRNG selection is a clean nextInt over
  // the live set (selecting then skipping dead txns would consume the PRNG
  // stream unevenly and quietly change which schedule a seed produces).
  const gens = txns.map((t) => t(ctx));
  const active: number[] = txns.map((_, i) => i);
  const outcome: ("commit" | "abort")[] = txns.map(() => "commit");
  const aborts: Record<number, string> = {};
  const trace: TraceEntry[] = [];

  let step = 0;
  for (const i of active) trace.push({ txn: i, step: step++, label: "start" });

  while (active.length > 0) {
    // Pick a live transaction uniformly at random from the seeded stream.
    const pos = rng.nextInt(0, active.length);
    const txnIdx = active[pos];
    const gen = gens[txnIdx];

    try {
      const res = gen.next();
      if (res.done) {
        // Generator returned normally => the transaction committed.
        trace.push({ txn: txnIdx, step: step++, label: "commit" });
        active.splice(pos, 1);
      } else {
        // It yielded a step label; record it and leave it runnable.
        trace.push({ txn: txnIdx, step: step++, label: res.value });
      }
    } catch (err) {
      // A throw out of a txn body is an abort (e.g. write-write conflict under
      // a stricter isolation level). We record the reason and remove it — the
      // book prints this to contrast isolation levels. We do NOT rethrow: an
      // abort is an expected outcome of concurrency, not a crash.
      outcome[txnIdx] = "abort";
      aborts[txnIdx] = err instanceof Error ? err.message : String(err);
      trace.push({ txn: txnIdx, step: step++, label: `abort(${aborts[txnIdx]})` });
      active.splice(pos, 1);
    }
  }

  return { trace, outcome, aborts };
}

/** Render a schedule as a compact one-line-per-step string. The txn chapters use
 *  this to print the offending interleaving alongside the anomaly, so the reader
 *  can replay it mentally. Kept in core so trace formatting is uniform. */
export function formatTrace(result: ScheduleResult): string {
  return result.trace
    .map((e) => `  step ${String(e.step).padStart(2)}  T${e.txn}  ${e.label}`)
    .join("\n");
}
