// stage02-vector-clock.ts — vector clocks capture causality EXACTLY; scalar
// (Lamport) clocks cannot.
//
// The whole chapter rests on one fact a survey will tell you but never shows:
// a Lamport scalar imposes a TOTAL order, but the happened-before relation is a
// PARTIAL order. So Lamport must lie about concurrent events — it always reports
// "a before b" (or b before a) for two events that are genuinely independent.
// A vector clock keeps one counter per node, so it can represent the partial
// order faithfully and answer "concurrent?" with a definite yes.
//
// What this stage MEASURES, on the same in-memory network and SAME SEED as the
// Lamport chapter so the comparison is apples-to-apples:
//   1. Run N nodes exchanging messages. Stamp EVERY event (send + receive) with
//      both a vector clock and the equivalent Lamport scalar.
//   2. Compare all event pairs two ways and tabulate {a->b, b->a, concurrent,
//      equal}. The vector-clock classification is ground truth.
//   3. Count how many TRULY-concurrent pairs Lamport silently ordered. That is
//      Lamport's "missed concurrency" — it must be > 0 (the bug the chapter is
//      about) while the vector clock's missed count is 0 by construction.
//   4. FAILURE MODE: a tempting "optimization" that compares only the FIRST
//      vector component. Show it misclassifies real concurrent pairs as causal,
//      then switch back to full-component compare and watch the number return to
//      0. This is the "don't shortcut the partial order" lesson made numeric.
//
// Honesty notes on the numbers:
//   - Counts are REAL: produced by the deterministic SimClock + seeded Network,
//     reproducible byte-for-byte. No estimates here.
//   - The ABSOLUTE concurrency count depends on the toy topology/seed (a denser
//     or sparser message schedule shifts it). What transfers is the RELATIVE
//     claim: Lamport's missed-concurrency > 0, vector clock's = 0, and the
//     first-component shortcut produces a nonzero misclassification that the
//     full compare eliminates.
//
// We do NOT use runScenario here: that runner is built for fault-injection +
// safety invariants over a protocol, but this chapter's "result" is a pairwise
// analysis over a recorded event log, not a safety property checked per tick. We
// drive the SimClock directly and own the reporting, which keeps the causality
// bookkeeping front-and-center instead of buried in a protocol handler.

import { SimClock } from "./core/clock.js";
import { seededRng, type Rng } from "./core/prng.js";
import { Network, type Message } from "./core/network.js";
import { Node } from "./core/node.js";
import { Stats } from "./core/metrics.js";
import { printTable } from "./core/assert.js";

// Same seed the Lamport chapter (stage01) uses, so "missed concurrency" is a
// like-for-like comparison rather than two unrelated runs.
const SEED = 7;
const NODE_IDS = ["n0", "n1", "n2"] as const;

/** A vector clock: one logical counter per node, indexed by NODE_IDS order.
 *  Plain number[] (not a Map) because the node set is fixed and small — array
 *  indexing keeps the merge a tight loop and the printed form compact. */
type VectorClock = number[];

/** One recorded event in the global history. We stamp BOTH clocks at the instant
 *  the event happens so the later pairwise analysis compares what each clock
 *  actually knew — not a clock reconstructed after the fact. `lamport` is the
 *  scalar a Lamport process would have carried; it exists only to reproduce the
 *  scalar comparison and expose its blind spot. */
interface EventRecord {
  /** Index of the owning node in NODE_IDS — also the component this event bumps. */
  nodeIdx: number;
  /** "send" or "recv": both are events; internal-only events are omitted because
   *  they never cross nodes and so never create the inter-node concurrency the
   *  chapter is about. */
  kind: "send" | "recv";
  /** Snapshot of the node's vector clock AT this event (copied, not aliased —
   *  aliasing would let later mutations rewrite history and corrupt the analysis). */
  vc: VectorClock;
  /** The Lamport scalar at this event. Single number == the whole problem. */
  lamport: number;
  /** Human-readable label for the violation table, e.g. "n0#1 send". */
  label: string;
}

/** Pairwise causal relation between two events under some comparator. */
type Relation = "before" | "after" | "concurrent" | "equal";

/** Vector-clock comparison — the GROUND TRUTH for happened-before.
 *  a "before" b  iff  a.vc <= b.vc componentwise AND a.vc != b.vc.
 *  Neither <= the other  ==>  concurrent (the case a scalar cannot represent).
 *  Why this is correct: vc[k] counts events at node k that causally precede this
 *  event, so domination in every component IS the transitive happened-before. */
function compareVectorClocks(a: VectorClock, b: VectorClock): Relation {
  let aLessEq = true; // a <= b in every component so far
  let bLessEq = true; // b <= a in every component so far
  for (let k = 0; k < a.length; k++) {
    if (a[k] > b[k]) aLessEq = false;
    if (b[k] > a[k]) bLessEq = false;
  }
  if (aLessEq && bLessEq) return "equal"; // identical vectors
  if (aLessEq) return "before";
  if (bLessEq) return "after";
  return "concurrent"; // each has a component the other lacks — independent
}

/** BROKEN comparator (the chapter's failure mode): compare ONLY component 0.
 *  This is the seductive "I'll just look at the first counter" shortcut. It
 *  treats one node's local progress as a global order — exactly the Lamport
 *  mistake in vector clothing. It will call genuinely concurrent events causal
 *  whenever they happen to differ on n0's counter. Kept to demonstrate, then
 *  discard: NOTE: never ship this; it is here to be falsified by the numbers. */
function compareFirstComponentOnly(a: VectorClock, b: VectorClock): Relation {
  if (a[0] === b[0]) return "equal"; // collapses all of n1/n2 disagreement
  return a[0] < b[0] ? "before" : "after"; // a total order — concurrency erased
}

/** Lamport scalar comparison. A scalar can only ever say before/after/equal —
 *  there is no representable "concurrent". That structural inability is the
 *  point: equal scalars are the ONLY hint of possible concurrency, and even that
 *  is unreliable (two causally-ordered events can tie if timestamps collide). */
function compareLamport(a: number, b: number): Relation {
  if (a === b) return "equal";
  return a < b ? "before" : "after";
}

/** A node that maintains a vector clock + a Lamport scalar and appends an
 *  EventRecord to the shared history on every send and receive. We pigg-back on
 *  core/Node only for the network plumbing (inbox registration, sendTo); the
 *  clock-update rules are the actual subject and live here in plain sight. */
class VectorClockNode extends Node {
  private readonly idx: number;
  private vc: VectorClock;
  private lamport = 0;
  // Per-node event counter purely for readable labels ("n0#3"); not part of the
  // protocol — labels make the violation dump diagnosable.
  private localSeq = 0;

  constructor(
    id: string,
    clock: SimClock,
    net: Network,
    private readonly history: EventRecord[],
  ) {
    super(id, clock, net);
    this.idx = NODE_IDS.indexOf(id as (typeof NODE_IDS)[number]);
    if (this.idx < 0) throw new Error(`unknown node id "${id}"`);
    this.vc = new Array(NODE_IDS.length).fill(0);
  }

  /** Send rule: a send is a local event, so bump our own component (and the
   *  scalar) FIRST, then ship a COPY of the resulting clock with the message.
   *  Copy, not reference: the receiver must see the value at send time, immune to
   *  our later increments. */
  sendEvent(to: string): void {
    this.vc[this.idx]++;
    this.lamport++;
    this.record("send");
    this.sendTo(to, "tick", { vc: [...this.vc], lamport: this.lamport });
  }

  /** Receive rule: merge (componentwise max) the sender's knowledge into ours,
   *  THEN bump our own component for the receive event itself. Merge-before-bump
   *  is what makes happened-before transitive: we inherit everything the sender
   *  causally knew. Lamport's scalar analogue: max(local, msg)+1. */
  override onMessage(_from: string, msg: Message): void {
    const payload = msg.payload as { vc: VectorClock; lamport: number };
    for (let k = 0; k < this.vc.length; k++) {
      this.vc[k] = Math.max(this.vc[k], payload.vc[k]);
    }
    this.vc[this.idx]++;
    this.lamport = Math.max(this.lamport, payload.lamport) + 1;
    this.record("recv");
  }

  private record(kind: "send" | "recv"): void {
    this.localSeq++;
    this.history.push({
      nodeIdx: this.idx,
      kind,
      vc: [...this.vc], // snapshot: see EventRecord.vc rationale
      lamport: this.lamport,
      label: `${this.id}#${this.localSeq} ${kind}`,
    });
  }
}

/** Tally the four relations across all unordered event pairs under a comparator.
 *  We iterate i<j so each pair is counted once; "before"/"after" are reported as
 *  a single directed-causal bucket since direction is symmetric for the count. */
interface RelationCounts {
  causal: number; // before + after — one event happened-before the other
  concurrent: number; // genuinely independent
  equal: number; // identical timestamp
}

function tallyVectorClock(history: EventRecord[]): RelationCounts {
  const counts: RelationCounts = { causal: 0, concurrent: 0, equal: 0 };
  for (let i = 0; i < history.length; i++) {
    for (let j = i + 1; j < history.length; j++) {
      const rel = compareVectorClocks(history[i].vc, history[j].vc);
      if (rel === "concurrent") counts.concurrent++;
      else if (rel === "equal") counts.equal++;
      else counts.causal++;
    }
  }
  return counts;
}

/** Count pairs that vector clocks call CONCURRENT but `scalarCompare` claims are
 *  causally ordered (before/after). That gap is the comparator's "missed
 *  concurrency": every such pair is a real independence the scalar invented an
 *  order for. Ground truth is always the vector clock. */
function countMissedConcurrency(
  history: EventRecord[],
  scalarRelOf: (i: number, j: number) => Relation,
): { missed: number; firstExample?: [EventRecord, EventRecord] } {
  let missed = 0;
  let firstExample: [EventRecord, EventRecord] | undefined;
  for (let i = 0; i < history.length; i++) {
    for (let j = i + 1; j < history.length; j++) {
      const truth = compareVectorClocks(history[i].vc, history[j].vc);
      if (truth !== "concurrent") continue; // only concurrent pairs can be "missed"
      const claimed = scalarRelOf(i, j);
      if (claimed === "before" || claimed === "after") {
        missed++;
        if (!firstExample) firstExample = [history[i], history[j]];
      }
    }
  }
  return { missed, firstExample };
}

/** Drive a deterministic message workload over the SimClock and return the
 *  recorded global event history. The schedule is fixed (no randomness in WHO
 *  sends WHEN) so the causal structure is reproducible; the seeded Network still
 *  supplies realistic per-message latency/jitter, which is what creates the
 *  interleavings that make some pairs concurrent and others causal. */
function runWorkload(): EventRecord[] {
  const clock = new SimClock();
  const rng: Rng = seededRng(SEED);
  const stats = new Stats();
  const net = new Network(clock, rng, stats);
  const history: EventRecord[] = [];

  const nodes = NODE_IDS.map((id) => new VectorClockNode(id, clock, net, history));
  const [n0, n1, n2] = nodes;

  // A hand-built schedule that deliberately mixes causal chains with independent
  // activity, so the history contains BOTH happened-before pairs and concurrent
  // pairs. n0->n1 forms a chain; n2 acts independently early (concurrent with the
  // n0/n1 chain) and only later receives, joining the causal order. Times are in
  // virtual ms; the seeded Network adds ~10ms baseline + jitter so receives land
  // after their sends but interleave nondeterministically across links.
  clock.schedule(0, () => n0.sendEvent("n1")); // n0#1 -> n1
  clock.schedule(1, () => n2.sendEvent("n1")); // n2#1 -> n1  (concurrent w/ n0#1)
  clock.schedule(40, () => n1.sendEvent("n0")); // n1 replies to n0
  clock.schedule(41, () => n1.sendEvent("n2")); // n1 fans out to n2
  clock.schedule(80, () => n0.sendEvent("n2")); // n0 -> n2 (carries n0's history)
  clock.schedule(81, () => n2.sendEvent("n0")); // n2 -> n0

  // Drain to quiescence: every scheduled send fires, every delivery callback
  // (onMessage) runs, the history fills. Deterministic given SEED.
  clock.run();
  return history;
}

function main(): void {
  const history = runWorkload();

  console.log(`\n=== Stage 02: Vector Clock — capturing concurrency (seed=${SEED}) ===`);
  console.log(
    `nodes=${NODE_IDS.length} events=${history.length} ` +
      `pairs=${(history.length * (history.length - 1)) / 2}\n`,
  );

  // (1) Ground-truth classification by vector clock.
  const vc = tallyVectorClock(history);
  printTable([
    { relation: "a -> b / b -> a (causal)", "pair count": vc.causal },
    { relation: "a || b (concurrent)", "pair count": vc.concurrent },
    { relation: "a == b (equal)", "pair count": vc.equal },
  ]);

  // (2) Missed-concurrency: vector clock vs Lamport scalar, same history.
  const vcMissed = countMissedConcurrency(history, (i, j) =>
    compareVectorClocks(history[i].vc, history[j].vc),
  );
  const lamportMissed = countMissedConcurrency(history, (i, j) =>
    compareLamport(history[i].lamport, history[j].lamport),
  );

  console.log("\n--- Missed concurrency (concurrent pairs reported as causal) ---");
  printTable([
    { clock: "vector clock (ground truth)", "missed concurrency": vcMissed.missed },
    { clock: "lamport scalar (ch.1)", "missed concurrency": lamportMissed.missed },
  ]);
  if (lamportMissed.firstExample) {
    const [a, b] = lamportMissed.firstExample;
    console.log(
      `  e.g. ${a.label} [${a.vc.join(",")}] (L=${a.lamport}) ` +
        `vs ${b.label} [${b.vc.join(",")}] (L=${b.lamport}): ` +
        `vector clock says CONCURRENT, lamport says ${
          a.lamport < b.lamport ? "before" : "after"
        } (L ${a.lamport} vs ${b.lamport}) — a fabricated order`,
    );
  }
  console.log(
    `  => lamport invented an order for ${lamportMissed.missed} independent pair(s); ` +
      `vector clock missed ${vcMissed.missed}.`,
  );

  // (3) FAILURE MODE: the "compare first component only" shortcut, then the fix.
  console.log("\n--- Failure mode: 'compare first vector component only' ---");
  const brokenMissed = countMissedConcurrency(history, (i, j) =>
    compareFirstComponentOnly(history[i].vc, history[j].vc),
  );
  // Also count pairs the shortcut OUTRIGHT mislabels vs ground truth (any
  // disagreement, not only concurrent->causal) — a broader honesty check that
  // the shortcut is wrong, not merely incomplete.
  let brokenDisagreements = 0;
  for (let i = 0; i < history.length; i++) {
    for (let j = i + 1; j < history.length; j++) {
      const truth = compareVectorClocks(history[i].vc, history[j].vc);
      const broken = compareFirstComponentOnly(history[i].vc, history[j].vc);
      if (truth !== broken) brokenDisagreements++;
    }
  }
  printTable([
    {
      comparator: "first-component-only (BROKEN)",
      "missed concurrency": brokenMissed.missed,
      "total mislabels": brokenDisagreements,
    },
    {
      comparator: "full-component (FIXED)",
      "missed concurrency": vcMissed.missed,
      "total mislabels": 0,
    },
  ]);
  if (brokenMissed.firstExample) {
    const [a, b] = brokenMissed.firstExample;
    console.log(
      `  broken e.g. ${a.label} [${a.vc.join(",")}] vs ${b.label} [${b.vc.join(",")}]: ` +
        `truly CONCURRENT, but first-component compare (${a.vc[0]} vs ${b.vc[0]}) ` +
        `calls it ${a.vc[0] < b.vc[0] ? "before" : "after"} — concurrency erased`,
    );
  }
  console.log(
    `  => switching to full-component compare drops missed concurrency to ` +
      `${vcMissed.missed} and mislabels to 0.`,
  );

  // Determinism reminder: re-running prints identical numbers (SEED fixed).
  console.log(
    `\n[determinism] seed=${SEED} fixed schedule => byte-identical output across runs.`,
  );
}

main();
