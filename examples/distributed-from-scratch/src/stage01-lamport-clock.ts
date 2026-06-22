// stage01-lamport-clock.ts — Lamport logical clocks, and the one thing they
// CANNOT do: detect concurrency.
//
// Chapter claim being made executable: a Lamport clock gives every event an
// integer stamp such that a -> b (a happens-before b) IMPLIES stamp(a) <
// stamp(b). That implication is one-directional. The converse is FALSE:
// stamp(a) < stamp(b) does NOT imply a -> b. Two events with different stamps
// may be genuinely concurrent. This stage proves both halves on real, replayable
// data:
//   (1) over EVERY causal pair in a real 3-node run, stamp ordering is correct
//       (the guarantee holds — 0 violations, computed not asserted);
//   (2) we exhibit concrete concurrent pairs (neither happens-before the other,
//       proven from the message DAG) that nonetheless carry DIFFERENT stamps —
//       so a reader-of-stamps would wrongly infer an ordering. That is the
//       failure mode the next chapter (vector clocks) exists to fix.
//
// Why this design and not a toy two-event example: concurrency is only
// interesting when it ARISES from real message interleaving on a faulty network
// (jitter + reorder). We let three nodes exchange messages over the deterministic
// SimClock/Network, record every local/send/recv event with its Lamport stamp,
// then reconstruct the TRUE happens-before relation independently — from program
// order + send->recv edges — and compare it against what the stamps say.
//
// The honesty boundary: the "true" happens-before here is GROUND TRUTH by
// construction (we own the event log and the message DAG), so the comparison is
// exact, not statistical. The numbers are real counts over a fixed seed, not
// estimates. Absolute pair counts are specific to this toy topology; what
// transfers to real systems is the QUALITATIVE result: a non-trivial fraction of
// concurrent pairs get distinct stamps, i.e. Lamport stamps overstate ordering.

import { runScenario } from "./core/scenario.js";
import { Node } from "./core/node.js";
import { assert, printTable, type TableRow } from "./core/assert.js";
import type { Message } from "./core/network.js";

const SEED = 7;
const NODE_IDS = ["n0", "n1", "n2"] as const;

// A single recorded event in the global trace. `gid` is a globally-unique
// sequence number assigned at record time — it is the IDENTITY of the event,
// used to build the happens-before DAG. It is NOT the Lamport stamp and is never
// shown to the protocol; it only exists so the analysis can refer to events
// unambiguously. `lamport` is the stamp the node computed under Lamport's rules.
interface TraceEvent {
  gid: number;
  node: string;
  // "local" = an internal step (no message); "send"/"recv" = the two endpoints
  // of a message. The kind drives both the Lamport update rule and the DAG edges.
  kind: "local" | "send" | "recv";
  lamport: number;
  // For send: the gid this send will be received as is unknown at send time, so
  // we thread a message id (mid) and resolve the send->recv edge afterward.
  mid?: number;
  detail: string;
}

// The shared experiment trace. A module-level array is acceptable here because
// the whole stage is one self-contained simulation; nodes append in clock order,
// so insertion order == global causal-consistent order of *recording*.
const trace: TraceEvent[] = [];
let gidCounter = 0;
let midCounter = 0;

function recordEvent(
  node: string,
  kind: TraceEvent["kind"],
  lamport: number,
  detail: string,
  mid?: number,
): TraceEvent {
  const ev: TraceEvent = { gid: gidCounter++, node, kind, lamport, mid, detail };
  trace.push(ev);
  return ev;
}

// A process running a Lamport logical clock. The clock is the WHOLE state — the
// point of the chapter is how few moving parts buy you a consistent timestamp of
// causality. Every event type bumps the counter by the Lamport rules:
//   - local / send: counter += 1, then stamp the event with the new counter
//   - recv:         counter = max(counter, msgStamp) + 1
// The max-then-increment on recv is the load-bearing line: it forces the
// receiver's clock strictly past the sender's send stamp, which is exactly what
// guarantees send -> recv => stamp(send) < stamp(recv).
class LamportProcess extends Node {
  private counter = 0;

  // The driver hands each process a fixed script of sends so the run is fully
  // determined by SEED + script (no Math.random, no wall-clock). Each entry is a
  // (delayMs, to) pair: at delayMs virtual time, do a local step then send to
  // `to`. Genuine concurrency comes from independent processes acting at
  // overlapping virtual times with no message between them.
  scheduleScript(script: ReadonlyArray<{ atMs: number; to: string }>): void {
    for (const step of script) {
      this.setTimer(step.atMs, () => this.doLocalThenSend(step.to));
    }
  }

  private doLocalThenSend(to: string): void {
    // A purely internal step first: models "the process did some work" between
    // messages. This manufactures events that are concurrent with other nodes'
    // internal work — the raw material for the concurrency demo.
    this.counter += 1;
    recordEvent(this.id, "local", this.counter, "internal work");

    // Send event: bump, stamp, attach the stamp to the wire so the receiver can
    // apply max(). We thread a `mid` so the analysis can pair this send with its
    // delivery and draw the send->recv DAG edge.
    this.counter += 1;
    const mid = midCounter++;
    recordEvent(this.id, "send", this.counter, `send -> ${to} (mid=${mid})`, mid);
    this.sendTo(to, "ping", { lamport: this.counter, mid });
  }

  onMessage(_from: string, msg: Message): void {
    const payload = msg.payload as { lamport: number; mid: number };
    // THE Lamport receive rule. Without max(), a receiver whose clock lagged the
    // sender could stamp the recv BELOW the send — breaking the entire guarantee.
    // The +1 makes it strictly greater, so send -> recv is always stamp-ordered.
    this.counter = Math.max(this.counter, payload.lamport) + 1;
    recordEvent(this.id, "recv", this.counter, `recv from ${msg.from} (mid=${payload.mid})`, payload.mid);
  }
}

// --- happens-before reconstruction (GROUND TRUTH, independent of stamps) ------
//
// Build the partial order -> from two and only two edge sources (Lamport's own
// definition of happens-before):
//   (PO) program order: within one process, an earlier-recorded event precedes a
//        later one. We use recording order per node, which equals real execution
//        order because a node's events fire on the single SimClock thread.
//   (MSG) message order: a send precedes its matching recv.
// Transitive closure of (PO ∪ MSG) is the happens-before relation. Two distinct
// events are CONCURRENT iff neither reaches the other in this DAG.
//
// We compute reachability with a BFS per source event. n is tiny (tens of
// events) so O(n^2) closure is fine and keeps the code legible; a real system
// would not enumerate all pairs, but the chapter's claim is about the relation,
// not performance.
function computeHappensBefore(events: readonly TraceEvent[]): {
  reaches: boolean[][];
  sendOf: Map<number, number>;
  recvOf: Map<number, number>;
} {
  const n = events.length;
  const byGid = new Map<number, number>(); // gid -> index in events[]
  events.forEach((e, i) => byGid.set(e.gid, i));

  // Adjacency list of direct -> edges.
  const adj: number[][] = events.map(() => []);

  // (PO) program-order edges: consecutive events on the same node.
  const lastIndexByNode = new Map<string, number>();
  events.forEach((e, i) => {
    const prev = lastIndexByNode.get(e.node);
    if (prev !== undefined) adj[prev].push(i);
    lastIndexByNode.set(e.node, i);
  });

  // (MSG) send->recv edges: pair each send with the recv carrying the same mid.
  const sendIdxByMid = new Map<number, number>();
  const recvIdxByMid = new Map<number, number>();
  events.forEach((e, i) => {
    if (e.kind === "send" && e.mid !== undefined) sendIdxByMid.set(e.mid, i);
    if (e.kind === "recv" && e.mid !== undefined) recvIdxByMid.set(e.mid, i);
  });
  const sendOf = new Map<number, number>(); // mid -> gid of send
  const recvOf = new Map<number, number>(); // mid -> gid of recv
  for (const [mid, si] of sendIdxByMid) {
    const ri = recvIdxByMid.get(mid);
    sendOf.set(mid, events[si].gid);
    // A send whose recv was dropped (partition/loss) has no edge — correct: a
    // lost message creates NO causal relation, which is itself a teaching point.
    if (ri !== undefined) {
      adj[si].push(ri);
      recvOf.set(mid, events[ri].gid);
    }
  }

  // Transitive closure via BFS from each node. reaches[a][b] === true means a -> b.
  const reaches: boolean[][] = events.map(() => new Array<boolean>(n).fill(false));
  for (let s = 0; s < n; s++) {
    const queue = [s];
    while (queue.length > 0) {
      const u = queue.shift()!;
      for (const v of adj[u]) {
        if (!reaches[s][v]) {
          reaches[s][v] = true;
          queue.push(v);
        }
      }
    }
  }
  return { reaches, sendOf, recvOf };
}

interface PairStats {
  causalPairs: number;
  causalStampViolations: number;
  concurrentPairs: number;
  concurrentSameStamp: number;
  concurrentDiffStamp: number;
  // A concrete witness: two concurrent events with different stamps.
  witness?: { a: TraceEvent; b: TraceEvent };
}

// Classify every unordered pair {a,b} of distinct events. The classification is
// exhaustive: a pair is either causal (exactly one direction reaches) or
// concurrent (neither reaches). For causal pairs we VERIFY Lamport's guarantee;
// for concurrent pairs we MEASURE how often stamps falsely differ.
function analyzePairs(events: readonly TraceEvent[], reaches: boolean[][]): PairStats {
  const stats: PairStats = {
    causalPairs: 0,
    causalStampViolations: 0,
    concurrentPairs: 0,
    concurrentSameStamp: 0,
    concurrentDiffStamp: 0,
  };
  const n = events.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const aFirst = reaches[i][j];
      const bFirst = reaches[j][i];
      // A DAG cannot have both directions; if it did, our edge construction is
      // buggy. Guard it — a cycle would silently corrupt every count below.
      assert(!(aFirst && bFirst), `happens-before cycle between ${events[i].gid} and ${events[j].gid}`);

      if (aFirst || bFirst) {
        stats.causalPairs += 1;
        const earlier = aFirst ? events[i] : events[j];
        const later = aFirst ? events[j] : events[i];
        // The guarantee: a -> b => stamp(a) < stamp(b). Count violations; expect 0.
        if (!(earlier.lamport < later.lamport)) stats.causalStampViolations += 1;
      } else {
        stats.concurrentPairs += 1;
        if (events[i].lamport === events[j].lamport) {
          stats.concurrentSameStamp += 1;
        } else {
          stats.concurrentDiffStamp += 1;
          // Keep the FIRST witness deterministically (smallest gid pair) so the
          // printed failure example is stable across runs.
          if (!stats.witness) stats.witness = { a: events[i], b: events[j] };
        }
      }
    }
  }
  return stats;
}

function fmtEvent(e: TraceEvent): string {
  return `[gid ${e.gid}] ${e.node} ${e.kind} L=${e.lamport} (${e.detail})`;
}

runScenario("lamport-clock-3-nodes", SEED, {
  untilMs: 5000,
  // Jitter + reorder so message interleaving is non-trivial; this is what makes
  // independent nodes' events genuinely overlap and produce concurrency.
  network: { defaultBaselineMs: 10, defaultJitterMs: 8 },
  setup(ctx) {
    const procs = NODE_IDS.map((id) => new LamportProcess(id, ctx.clock, ctx.net));
    const [n0, n1, n2] = procs;
    ctx.net.reorder(true);

    // Hand-crafted send scripts. The overlapping `atMs` across DIFFERENT nodes is
    // deliberate: at ~t=20 all three are doing independent local work + sends
    // with no causal link between them -> that's where concurrency is born. The
    // later messages weave causal chains (n0->n1->n2) so we also get a healthy
    // population of CAUSAL pairs to verify the guarantee against.
    n0.scheduleScript([
      { atMs: 20, to: "n1" }, // chain seed: n0 -> n1
      { atMs: 200, to: "n2" },
    ]);
    n1.scheduleScript([
      { atMs: 20, to: "n2" }, // concurrent with n0's t=20 send (no link between them)
      { atMs: 120, to: "n0" },
    ]);
    n2.scheduleScript([
      { atMs: 20, to: "n0" }, // concurrent with both other t=20 sends
      { atMs: 300, to: "n1" },
    ]);

    return () => {
      // --- analysis runs AFTER the simulation has quiesced ---------------------
      const events = [...trace].sort((a, b) => a.gid - b.gid);
      const { reaches } = computeHappensBefore(events);
      const stats = analyzePairs(events, reaches);

      // (a) Global total order by Lamport stamp. Ties broken by node id then gid
      // so the printed order is deterministic. NOTE: this total order is a valid
      // LINEARIZATION of happens-before, but it INVENTS an order between concurrent
      // events — the very over-claim this stage is about.
      const ordered = [...events].sort(
        (x, y) => x.lamport - y.lamport || x.node.localeCompare(y.node) || x.gid - y.gid,
      );
      console.log("\n--- (a) Global event order sorted by Lamport stamp ---");
      const orderRows: TableRow[] = ordered.map((e) => ({
        L: e.lamport,
        node: e.node,
        kind: e.kind,
        gid: e.gid,
        detail: e.detail,
      }));
      printTable(orderRows);

      // (b) Causal vs concurrent accounting.
      console.log("\n--- (b) Happens-before vs Lamport-stamp discrimination ---");
      const total = stats.causalPairs + stats.concurrentPairs;
      console.log(`total distinct event pairs: ${total}`);
      console.log(
        `causal pairs (a->b): ${stats.causalPairs}  | ` +
          `stamp-order violations: ${stats.causalStampViolations} ` +
          `(guarantee: must be 0)`,
      );
      console.log(
        `concurrent pairs (a||b): ${stats.concurrentPairs}  | ` +
          `same stamp: ${stats.concurrentSameStamp}  ` +
          `different stamp: ${stats.concurrentDiffStamp}`,
      );
      const pctMisleading =
        stats.concurrentPairs === 0
          ? 0
          : (stats.concurrentDiffStamp / stats.concurrentPairs) * 100;
      console.log(
        `=> of concurrent pairs, ${pctMisleading.toFixed(1)}% carry DIFFERENT stamps, ` +
          `i.e. Lamport falsely implies an order for them`,
      );

      // --- FAILURE MODE: a concrete concurrent pair with different stamps ------
      console.log("\n--- FAILURE MODE: 'different stamp' != 'has causal order' ---");
      assert(
        stats.witness !== undefined,
        "expected at least one concurrent pair with different Lamport stamps " +
          "(if this fires, the topology produced no concurrency to demo)",
      );
      const { a, b } = stats.witness;
      console.log("Two events with DIFFERENT Lamport stamps:");
      console.log(`  A: ${fmtEvent(a)}`);
      console.log(`  B: ${fmtEvent(b)}`);
      console.log(
        `Stamps differ (${a.lamport} vs ${b.lamport}), so a stamp-reader infers an order.`,
      );
      // Prove from the DAG that NEITHER happens-before the other.
      const ai = events.findIndex((e) => e.gid === a.gid);
      const bi = events.findIndex((e) => e.gid === b.gid);
      assert(
        !reaches[ai][bi] && !reaches[bi][ai],
        "witness must be genuinely concurrent in the happens-before DAG",
      );
      console.log(
        "But in the happens-before DAG, A does NOT reach B and B does NOT reach A:",
      );
      console.log("  => they are CONCURRENT; the stamp difference is an artifact, not causality.");
      console.log("  => Lamport clocks cannot detect concurrency. (Vector clocks, next chapter, can.)");

      // The stage's headline metrics, returned to runScenario for the uniform table.
      return [
        { metric: "events recorded", value: events.length },
        { metric: "causal pairs", value: stats.causalPairs },
        { metric: "causal stamp violations", value: stats.causalStampViolations },
        { metric: "concurrent pairs", value: stats.concurrentPairs },
        { metric: "concurrent w/ different stamp", value: stats.concurrentDiffStamp },
      ];
    };
  },
});
