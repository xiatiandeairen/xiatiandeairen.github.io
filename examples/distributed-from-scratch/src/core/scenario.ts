// core/scenario.ts — a reproducible experiment = seed + fault script + invariants.
//
// Why this is the book's spine: every chapter ultimately makes a claim of the
// form "under THESE faults, THIS protocol preserves THIS invariant and pays THIS
// cost". A Scenario packages exactly that: the seed (so randomness is fixed), a
// timeline of fault-injection steps (partition at t, heal at t', crash a node),
// the set of safety invariants to check continuously, and the place to collect
// cost metrics. runScenario() then runs it and prints a structured result the
// reader can reproduce by re-running with the same seed.
//
// The teaching payoff: a Scenario makes failure REPEATABLE. "It split-brained
// once" becomes "seed 7 split-brains at event 142, here's the snapshot". That is
// the difference between this book and a survey.
//
// Invariant-checking cadence: invariants are evaluated after EVERY event (see
// stepwise run below), not just at the end. Safety properties ("≤1 leader/term")
// can be violated transiently and self-heal; checking only at the end would miss
// the very bug the chapter is about.

import { SimClock } from "./clock.js";
import { seededRng, type Rng } from "./prng.js";
import { Network, type NetworkOptions } from "./network.js";
import { Stats } from "./metrics.js";
import { checkInvariants, type Invariant, printTable } from "./assert.js";

/** Everything a scenario builder receives: the deterministic primitives plus a
 *  way to register faults and invariants. A stage's setup function gets this,
 *  wires up nodes, and returns the things runScenario needs to drive + judge. */
export interface ScenarioContext {
  readonly clock: SimClock;
  readonly rng: Rng;
  readonly net: Network;
  readonly stats: Stats;
  /** Schedule a fault at a virtual time: e.g. `at(5000, () => net.partition(...))`.
   *  Faults are just SimClock events, so they interleave with protocol messages
   *  deterministically. */
  at(timeMs: number, fault: () => void): void;
  /** Register a safety invariant checked after every event. */
  watch(inv: Invariant): void;
}

/** What a stage's setup returns to runScenario. `result` is called once the
 *  simulation quiesces (or hits the deadline) to produce the printed summary
 *  rows — the stage decides what's worth reporting (message cost, rounds, final
 *  agreed value). */
export interface ScenarioSpec {
  /** Optional time bound. If set, run stops at this virtual ms even if events
   *  remain (models "observe the system for N seconds"). */
  untilMs?: number;
  /** Network tuning (latency floors/jitter) for this experiment. */
  network?: NetworkOptions;
  /** Wire nodes + faults + invariants here. Return a function that computes the
   *  result rows once the run finishes. */
  setup(ctx: ScenarioContext): () => Record<string, string | number>[];
}

export interface ScenarioResult {
  name: string;
  seed: number;
  /** Virtual time at quiescence/deadline — deterministic for a seed. */
  finalTimeMs: number;
  /** Total simulation events fired — a deterministic proxy for "how much
   *  happened", useful for regression detection. */
  events: number;
  /** Whether all invariants held throughout. False ⇒ violation threw (the run
   *  aborted); this field is true on any run that completes. */
  safe: boolean;
  /** Message ledger snapshot — sent/delivered/dropped, so cost claims are
   *  auditable. */
  stats: Record<string, number>;
}

/** Run a scenario end to end with a fixed seed and print a structured report.
 *  This is the uniform entry point every stage calls. Determinism contract:
 *  same (name, seed, spec) ⇒ identical output, byte for byte. */
export function runScenario(name: string, seed: number, spec: ScenarioSpec): ScenarioResult {
  const clock = new SimClock();
  const rng = seededRng(seed);
  const stats = new Stats();
  const net = new Network(clock, rng, stats, spec.network);

  const invariants: Invariant[] = [];
  const ctx: ScenarioContext = {
    clock,
    rng,
    net,
    stats,
    at: (timeMs, fault) => clock.schedule(timeMs, fault),
    watch: (inv) => invariants.push(inv),
  };

  const computeResult = spec.setup(ctx);

  // Stepwise drive: one event at a time so invariants are checked at EVERY point
  // in the execution, not just the end. This is the difference between catching a
  // transient split-brain and missing it. We honor untilMs as a virtual deadline.
  let events = 0;
  const deadline = spec.untilMs ?? Infinity;
  while (clock.pending() > 0 && clock.now() <= deadline) {
    if (!clock.advance()) break;
    events++;
    // Throws on first violation, with the offending invariant's name + snapshot.
    // We deliberately let it propagate: a violated safety property is a failed
    // experiment the reader must see, not a caught-and-logged warning.
    checkInvariants(invariants);
    if (clock.now() > deadline) break;
  }

  const result: ScenarioResult = {
    name,
    seed,
    finalTimeMs: clock.now(),
    events,
    safe: true, // reached here ⇒ no invariant threw
    stats: stats.snapshot(),
  };

  // Uniform report: header line + the stage's own metric rows + the message
  // ledger. One layout across all chapters so numbers are comparable.
  console.log(`\n=== Scenario: ${name} (seed=${seed}) ===`);
  const rows = computeResult();
  if (rows.length > 0) printTable(rows);
  console.log(
    `\n[sim] finalTime=${result.finalTimeMs}ms events=${result.events} ` +
      `safe=${result.safe}`,
  );
  console.log(
    `[msgs] sent=${stats.get("msg.sent")} delivered=${stats.get("msg.delivered")} ` +
      `dropped=${stats.get("msg.dropped")} ` +
      `(loss=${stats.get("msg.dropped.loss")} partition=${stats.get("msg.dropped.partition")})`,
  );

  return result;
}
