// Stage 12 — deterministic orchestration (a tiny workflow engine).
//
// Stages 01–06 each ran ONE agent loop. Real systems run MANY units of work and
// need a layer ABOVE the loop that decides: what runs in parallel, how much
// parallelism is safe, where everything must converge before the next step, and
// what happens when one unit fails. That layer is the workflow engine, and the
// single most important word in this chapter is DETERMINISTIC.
//
// "Deterministic" here does NOT mean "an LLM produces the same text twice" (it
// won't). It means the ORCHESTRATION is plain code with no model in the control
// path: given the same task graph, the same things run, fan out, barrier, and
// fail in the same shape every time. The non-determinism is quarantined inside
// the leaf tasks (which may call a model); the scaffolding around them is
// boring, inspectable code. That is the whole point — you can reason about,
// test, and replay the engine even when the work it schedules is fuzzy.
//
// Four mechanisms, built from scratch on Node primitives (no LLM, no deps):
//   1. STAGE         — an ordered step in the pipeline; stage N+1 starts only
//                      after stage N's barrier resolves.
//   2. FAN-OUT       — run the N items of one stage concurrently...
//   3. CONCURRENCY CAP — ...but never more than `limit` at once (the difference
//                      between "fast" and "OOM / rate-limit-banned").
//   4. FAILURE ISOLATION — one item throwing does not abort its siblings; the
//                      engine collects per-item Results so a partial failure is
//                      a reported outcome, not a crashed pipeline.
//
// Plus the two failure modes this chapter exists to make visceral:
//   - UNBOUNDED FAN-OUT: `Promise.all(items.map(run))` with no cap = peak
//     concurrency == N. At N=200 that is 200 simultaneous model calls; the
//     resource ceiling (sockets / rate limit / memory) is hit and the run dies.
//   - BARRIER TAIL LATENCY: a stage cannot finish until its SLOWEST item does.
//     One straggler taxes the entire pipeline; the demo measures the exact cost.
//
// Run it: `npx tsx src/stage12-workflow.ts`. Pure mechanism — fake async tasks
// (timed sleeps + arithmetic) stand in for "an agent loop"; nothing here needs a
// network or a key.

// We use Result-style outcomes instead of letting tasks throw past the engine.
// Rationale (arch-runtime: business failure is a value, not an exception): the
// engine's contract is "I always return one outcome per item." A thrown error
// that escaped would violate that and take down siblings — exactly the failure
// isolation we are trying to demonstrate the cure for.
type TaskOutcome<T> =
  | { ok: true; value: T; durationMs: number }
  | { ok: false; error: string; durationMs: number };

// A unit of work. `id` is for human-readable reporting only; the engine never
// branches on it. The task is an async thunk so the engine controls *when* it
// starts (lazy) — passing already-started promises would defeat the cap, because
// a Promise begins executing the instant it is created, before the limiter can
// say "not yet".
interface Task<T> {
  id: string;
  run: () => Promise<T>;
}

// ----------------------------------------------------------------------------
// Mechanism 3 (the heart): a bounded-concurrency map.
//
// This is a hand-rolled `pLimit`/worker-pool. The naive version is
// `Promise.all(tasks.map(t => t.run()))`, which starts ALL tasks at once. Here
// we instead spin up exactly `limit` workers that pull from a shared cursor, so
// at most `limit` tasks are ever in flight. The cursor (`next`) is the shared
// mutable state; it is safe without a lock ONLY because JS is single-threaded
// and `next++` cannot be interleaved — there is no preemption between reading
// and incrementing. (In a real multithreaded runtime this would need an atomic.)
//
// Invariants:
//   - results[i] corresponds to tasks[i] (we index by captured `i`, not by
//     completion order — completion order is nondeterministic, slot order is not).
//   - exactly tasks.length outcomes are returned, one per task, always.
//   - a task that throws becomes an { ok: false } outcome; siblings keep running.
// ----------------------------------------------------------------------------
async function runStage<T>(
  tasks: Task<T>[],
  limit: number,
  // Observer for live concurrency, so the demo can PROVE the cap holds rather
  // than assert it. Called on every start/finish with the current in-flight count.
  onInflightChange?: (inflight: number) => void
): Promise<TaskOutcome<T>[]> {
  const results: TaskOutcome<T>[] = new Array(tasks.length);
  let next = 0; // shared cursor: index of the next task to claim
  let inflight = 0;

  // One worker = one "lane". It loops claiming tasks until the cursor is drained.
  // We create min(limit, tasks.length) of them — spawning more lanes than tasks
  // would just create idle workers that immediately exit, harmless but wasteful.
  async function worker(): Promise<void> {
    while (true) {
      const i = next;
      if (i >= tasks.length) return; // cursor drained: this lane retires
      next += 1; // claim slot i; the next worker sees i+1 (no double-run)

      inflight += 1;
      onInflightChange?.(inflight);
      const startedAtMs = Date.now();
      try {
        const value = await tasks[i].run();
        results[i] = { ok: true, value, durationMs: Date.now() - startedAtMs };
      } catch (err) {
        // FAILURE ISOLATION lives here: the throw is caught INSIDE the lane, so
        // it never rejects the Promise.all below. The lane records the failure
        // and immediately loops to claim the next task — a poison item costs one
        // slot, not the stage.
        results[i] = { ok: false, error: (err as Error).message, durationMs: Date.now() - startedAtMs };
      } finally {
        inflight -= 1;
        onInflightChange?.(inflight);
      }
    }
  }

  const laneCount = Math.min(limit, tasks.length);
  // Promise.all over the LANES (a fixed, bounded set), not over the tasks. This
  // is the structural trick: bounded parallelism = bounded number of promises
  // we await at the top level, each of which internally sequences many tasks.
  await Promise.all(Array.from({ length: laneCount }, () => worker()));
  return results;
}

// ----------------------------------------------------------------------------
// Mechanism 1 + 2: the pipeline. A list of stages run in order; within a stage,
// items fan out under the cap; between stages, a barrier (the `await`) forces
// full convergence before the next stage sees any input.
//
// The barrier is implicit but absolute: `runStage` does not resolve until every
// lane has retired, i.e. every item is done. That is what lets stage N+1 safely
// assume stage N is fully complete — and also what creates tail latency (one
// straggler holds the barrier shut). The demo measures both faces of this coin.
// ----------------------------------------------------------------------------
interface StageSpec<T> {
  name: string;
  tasks: Task<T>[];
  limit: number;
}

interface StageReport<T> {
  name: string;
  outcomes: TaskOutcome<T>[];
  wallClockMs: number; // stage duration = its slowest lane = barrier cost
  peakInflight: number; // observed max concurrency; must be <= limit (the proof)
  succeeded: number;
  failed: number;
}

async function runPipeline<T>(stages: StageSpec<T>[]): Promise<StageReport<T>[]> {
  const reports: StageReport<T>[] = [];
  for (const stage of stages) {
    let peakInflight = 0;
    const startedAtMs = Date.now();
    const outcomes = await runStage(stage.tasks, stage.limit, (inflight) => {
      if (inflight > peakInflight) peakInflight = inflight;
    });
    // Barrier reached here: the line above does not return until ALL items of
    // this stage have settled. Only now do we move to the next stage.
    reports.push({
      name: stage.name,
      outcomes,
      wallClockMs: Date.now() - startedAtMs,
      peakInflight,
      succeeded: outcomes.filter((o) => o.ok).length,
      failed: outcomes.filter((o) => !o.ok).length,
    });
  }
  return reports;
}

// --- Fake tasks: deterministic stand-ins for "an agent loop". ----------------
//
// We use sleeps (not real model calls) so durations are controlled and the
// numbers below are reproducible. The durations are REAL wall-clock (Date.now
// deltas), so the tail-latency and cap proofs are measured, not asserted.

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

// A task that "works" for `ms` then returns a label. Stands in for a fast item.
function makeWorkTask(id: string, ms: number): Task<string> {
  return { id, run: async () => { await sleep(ms); return `${id} done`; } };
}

// A task that throws after `ms`. Stands in for a model/tool failure mid-stage.
function makeFailingTask(id: string, ms: number): Task<string> {
  return {
    id,
    run: async () => {
      await sleep(ms);
      throw new Error(`${id} hit a simulated tool error`);
    },
  };
}

// --- The four demos. ---------------------------------------------------------

// Demo A: fan out N items under a cap, and PROVE the cap held by reporting the
// observed peak concurrency. With limit=4 and 12 items, peak must be exactly 4.
async function demoCappedFanout(): Promise<void> {
  console.log('\n--- A. Fan-out with concurrency cap (12 tasks, limit=4) ---');
  const tasks = Array.from({ length: 12 }, (_, i) => makeWorkTask(`t${i}`, 50));
  const [report] = await runPipeline([{ name: 'fanout', tasks, limit: 4 }]);
  // Serial would be 12*50=600ms; unbounded ~50ms; capped sits in between.
  console.log(`  items        : ${report.outcomes.length}`);
  console.log(`  peak inflight: ${report.peakInflight} (cap was 4 — never exceeded)`);
  console.log(`  wall clock   : ${report.wallClockMs}ms  (serial would be ~600ms; cap=4 ⇒ ~3 waves of 50ms)`);
  console.log(`  succeeded    : ${report.succeeded}/${report.outcomes.length}`);
}

// Demo B: one item throws. The other items in the SAME stage still complete.
// Without isolation, the single throw would reject Promise.all and the surviving
// items' results would be lost (their work wasted, their outcomes unobservable).
async function demoFailureIsolation(): Promise<void> {
  console.log('\n--- B. Failure isolation (1 poison item among 5) ---');
  const tasks: Task<string>[] = [
    makeWorkTask('ok-0', 30),
    makeWorkTask('ok-1', 30),
    makeFailingTask('boom-2', 30), // throws — must NOT take down siblings
    makeWorkTask('ok-3', 30),
    makeWorkTask('ok-4', 30),
  ];
  const [report] = await runPipeline([{ name: 'isolated', tasks, limit: 3 }]);
  for (const o of report.outcomes) {
    console.log(o.ok ? `  ✓ ${o.value} (${o.durationMs}ms)` : `  ✗ ${o.error} (${o.durationMs}ms)`);
  }
  console.log(`  result       : ${report.succeeded} ok, ${report.failed} failed — pipeline kept going`);
}

// Demo C: a multi-stage pipeline with a barrier between stages. Stage 2 must not
// start until stage 1 fully drains. We prove ordering by timestamping the first
// start of each stage relative to t0 and showing stage 2 begins AFTER stage 1's
// barrier (its wall clock).
async function demoBarrierPipeline(): Promise<void> {
  console.log('\n--- C. Multi-stage pipeline (barrier between stages) ---');
  const stages: StageSpec<string>[] = [
    { name: 'fetch', tasks: Array.from({ length: 6 }, (_, i) => makeWorkTask(`fetch-${i}`, 40)), limit: 3 },
    { name: 'summarize', tasks: Array.from({ length: 4 }, (_, i) => makeWorkTask(`sum-${i}`, 40)), limit: 2 },
  ];
  const t0 = Date.now();
  const reports = await runPipeline(stages);
  let cumulative = 0;
  for (const r of reports) {
    console.log(`  stage "${r.name}": ${r.succeeded} ok, wall=${r.wallClockMs}ms, started ~${cumulative}ms after t0`);
    cumulative += r.wallClockMs;
  }
  console.log(`  total pipeline: ${Date.now() - t0}ms (stages are SEQUENTIAL — sum of stage barriers, not overlapped)`);
}

// Demo D: the two failure modes, measured side by side.
//
// D1 — barrier tail latency: one straggler (200ms) among fast items (20ms). The
//      stage's wall clock is dragged to the straggler's duration: the barrier
//      can only open when the SLOWEST lane finishes. This is the hidden tax of
//      "wait for everything" — your p50 is irrelevant, your p100 is the bill.
//
// D2 — unbounded fan-out: we contrast the SAME workload at limit=N (unbounded)
//      vs a small cap. We do NOT actually open 200 sockets (that would harm the
//      machine running the book's CI). Instead the fake task records peak
//      concurrency, so we can SHOW that limit=N drives peak to N — the exact
//      mechanism by which real unbounded fan-out exhausts sockets / rate limits
//      / memory. The danger is demonstrated structurally, not by causing harm.
async function demoFailureModes(): Promise<void> {
  console.log('\n--- D1. Barrier tail latency (one 200ms straggler among 20ms items) ---');
  const tailTasks: Task<string>[] = [
    makeWorkTask('fast-0', 20),
    makeWorkTask('fast-1', 20),
    makeWorkTask('fast-2', 20),
    makeWorkTask('straggler-3', 200), // the tail that holds the barrier shut
  ];
  const [tailReport] = await runPipeline([{ name: 'with-straggler', tasks: tailTasks, limit: 4 }]);
  const fastest = Math.min(...tailReport.outcomes.map((o) => o.durationMs));
  const slowest = Math.max(...tailReport.outcomes.map((o) => o.durationMs));
  console.log(`  fastest item : ${fastest}ms`);
  console.log(`  slowest item : ${slowest}ms (the straggler)`);
  console.log(`  stage wall   : ${tailReport.wallClockMs}ms — barrier waited for the slowest, not the average`);
  console.log(`  tax          : ${tailReport.wallClockMs - fastest}ms wasted by 3 fast lanes idling at the barrier`);

  console.log('\n--- D2. Unbounded vs capped fan-out (peak concurrency = the resource bill) ---');
  const N = 50;
  // SAME N tasks, two policies. We keep each task near-instant so the only thing
  // we are measuring is PEAK CONCURRENCY, which is what exhausts resources.
  const mkTasks = () => Array.from({ length: N }, (_, i) => makeWorkTask(`task-${i}`, 5));
  const [unbounded] = await runPipeline([{ name: 'unbounded', tasks: mkTasks(), limit: N }]);
  const [capped] = await runPipeline([{ name: 'capped', tasks: mkTasks(), limit: 8 }]);
  console.log(`  unbounded (limit=N=${N}): peak inflight = ${unbounded.peakInflight}  ← N concurrent model calls = OOM / rate-limit ban`);
  console.log(`  capped    (limit=8)    : peak inflight = ${capped.peakInflight}  ← bounded blast radius, predictable resource use`);
  console.log(`  same ${N} tasks both runs; the cap changed peak from ${unbounded.peakInflight} to ${capped.peakInflight} without losing work`);
}

async function main(): Promise<void> {
  console.log('=== Stage 12: deterministic workflow engine (no LLM — pure orchestration) ===');
  await demoCappedFanout();
  await demoFailureIsolation();
  await demoBarrierPipeline();
  await demoFailureModes();
  console.log('\nTakeaway: the engine is boring, replayable code. Cap bounds the resource bill,');
  console.log('isolation turns a crash into a reported outcome, the barrier buys ordering at the');
  console.log('price of tail latency. The non-determinism stays inside the leaf tasks.\n');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
