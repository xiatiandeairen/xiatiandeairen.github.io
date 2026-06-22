// stage03-rpc-failure-detector.ts — chapter 3: telling "slow" from "dead".
//
// The chapter's thesis: in an asynchronous network you CANNOT directly observe
// whether a remote node crashed or is merely slow — both look identical (no
// reply arrives). Two mechanisms cope with that ambiguity, and each has a price
// this stage measures with real simulated numbers:
//
//   1. RPC under at-least-once retry. Because a missing reply is ambiguous, a
//      client that wants its request to eventually land MUST retry. Retries +
//      lossy links + network duplication mean the SERVER sees a request more
//      than once. The fix is idempotency: tag each logical request with a stable
//      requestId and dedup on the server so the SIDE EFFECT runs exactly once.
//      We print: server arrivals (inflated by retries/dups) vs side effects
//      applied (== logical request count). The gap IS the cost of ambiguity.
//
//   2. φ-accrual failure detection (Hayashibara et al. 2004). Instead of a
//      binary "up/down" from a fixed timeout, the detector models heartbeat
//      inter-arrival times as a normal distribution and outputs φ = -log10(P(a
//      heartbeat is later than the time since the last one)). φ rises smoothly
//      the longer a heartbeat is overdue; a threshold turns it into a suspicion.
//      We inject a REAL crash, watch φ climb, and report detection latency =
//      (time φ first crosses threshold) − (crash time). That latency is the
//      unavoidable tax for distinguishing dead from slow: you must wait long
//      enough that "slow" becomes statistically implausible.
//
// The failure mode the chapter is really about — a tight timeout MISCLASSIFIES a
// healthy-but-slow node as dead (a false positive, a "murder"). We reproduce it
// by setting the φ threshold below what the link's p99 latency warrants: the
// detector kills a node that is alive and just jittery. Raising the threshold
// past the tail drives false positives to zero — at the cost of slower real-
// crash detection. That tradeoff (fast detection ⇄ few false positives) is the
// whole design space of failure detectors and it falls straight out of the φ
// math; we print both ends of it.
//
// Determinism: every drop coin, jitter sample, duplicate coin, and crash time
// flows through the seeded Rng / SimClock in core. Same seed ⇒ identical φ
// trajectory and identical false-positive counts, so the murder is reproducible,
// not a flake.
//
// Honesty caveat on absolute numbers: latencies here are simulated (baseline +
// uniform jitter from core/network), not measured on a real NIC, so absolute ms
// are optimistic and clean. What transfers is the RELATIVE structure — that φ
// crosses threshold a bounded multiple of the heartbeat interval after a crash,
// and that lowering the threshold below the latency tail trades correctness for
// speed. Those relationships hold on real networks; the absolute ms do not.

import { runScenario } from "./core/scenario.js";
import { Node } from "./core/node.js";
import type { Message } from "./core/network.js";
import { Histogram } from "./core/metrics.js";
import { invariant } from "./core/assert.js";

// --- protocol constants ------------------------------------------------------
// Named so the why is explicit and a reader can perturb one knob at a time.

/** Heartbeat cadence the server emits at while alive. The φ detector's entire
 *  notion of "on time" is relative to this; detection latency scales with it. */
const HEARTBEAT_INTERVAL_MS = 100;

/** Client RPC retry interval. Must exceed the round-trip (baseline*2 + jitter)
 *  or every request retries before its own reply can arrive — which would inflate
 *  server arrivals for a reason unrelated to loss. We keep it comfortably above
 *  the RTT so retries fire only on genuinely missing replies. */
const RPC_RETRY_INTERVAL_MS = 80;

/** How many distinct logical requests the client issues. Each is retried until
 *  acked, so server arrivals ≥ this and side effects applied == this. */
const LOGICAL_REQUEST_COUNT = 20;

/** φ-accrual: window of recent inter-arrival samples the detector fits a normal
 *  distribution to. Too small ⇒ variance estimate is noisy; too large ⇒ slow to
 *  adapt to a latency regime change. 1000 is the paper's default order. */
const PHI_WINDOW = 1000;

/** Minimum stdev (ms) the detector assumes even if observed samples are nearly
 *  identical. Without a floor, a run of equal inter-arrivals gives stdev≈0 and φ
 *  explodes to Infinity the instant a heartbeat is one ms late — a degenerate
 *  detector that murders everything. The floor is the detector's humility. */
const PHI_MIN_STDEV_MS = 5;

// --- φ-accrual estimator -----------------------------------------------------
// Pure computation over a sample window: NO IO, NO clock reads inside. The
// caller feeds it arrival timestamps; it reports φ for a given "now". Keeping it
// pure is what lets us unit-reason about it and replay it deterministically.

/**
 * Hayashibara φ-accrual failure detector.
 *
 * Models heartbeat inter-arrival gaps as a normal distribution N(mean, stdev)
 * and, given the elapsed time since the last heartbeat, returns
 *   φ = -log10( P(next gap > elapsed) )
 * i.e. how surprising it is that no heartbeat has arrived yet, on a log scale.
 * φ=1 ≈ 10% chance we're wrong to suspect; φ=2 ≈ 1%; φ=3 ≈ 0.1%. A threshold on
 * φ is therefore a direct knob on the false-positive rate — that's the whole
 * appeal over a raw timeout.
 *
 * Invariant: needs ≥2 samples before φ is meaningful (can't estimate variance
 * from one gap); phi() returns 0 (= "no suspicion") until then, so a node is
 * never suspected during warmup.
 */
class PhiAccrualDetector {
  private readonly intervals: number[] = [];
  private lastHeartbeatAtMs = -1;

  /** Record a heartbeat arrival at virtual time `nowMs`. The gap since the
   *  previous arrival becomes a sample of the inter-arrival distribution. */
  recordHeartbeat(nowMs: number): void {
    if (this.lastHeartbeatAtMs >= 0) {
      const gap = nowMs - this.lastHeartbeatAtMs;
      this.intervals.push(gap);
      // Bounded window: drop oldest so the estimate tracks the current regime
      // rather than averaging over ancient history.
      if (this.intervals.length > PHI_WINDOW) this.intervals.shift();
    }
    this.lastHeartbeatAtMs = nowMs;
  }

  /** Suspicion level at virtual time `nowMs`. Higher = more likely the peer is
   *  dead. Returns 0 during warmup (no opinion yet). */
  phi(nowMs: number): number {
    if (this.intervals.length < 2 || this.lastHeartbeatAtMs < 0) return 0;
    const mean = this.computeMean();
    const stdev = Math.max(PHI_MIN_STDEV_MS, this.computeStdev(mean));
    const elapsed = nowMs - this.lastHeartbeatAtMs;
    // P(gap > elapsed) under N(mean, stdev). We use the logistic approximation
    // to the normal tail from the original paper (Eq. 5) — closed form, no erf,
    // and deterministic. φ = -log10 of that probability.
    const y = (elapsed - mean) / stdev;
    const e = Math.exp(-y * (1.5976 + 0.070566 * y * y));
    const pLater = elapsed > mean ? e / (1 + e) : 1 - 1 / (1 + e);
    // Clamp the probability away from 0 so -log10 stays finite; an "infinite" φ
    // is just a very high finite suspicion for printing purposes.
    const pSafe = Math.min(Math.max(pLater, 1e-12), 1);
    return -Math.log10(pSafe);
  }

  private computeMean(): number {
    let sum = 0;
    for (const v of this.intervals) sum += v;
    return sum / this.intervals.length;
  }

  private computeStdev(mean: number): number {
    let sumSq = 0;
    for (const v of this.intervals) sumSq += (v - mean) * (v - mean);
    return Math.sqrt(sumSq / this.intervals.length);
  }
}

// --- RPC server --------------------------------------------------------------

/** Payload of an RPC request. `requestId` is the idempotency key: the SAME
 *  logical request keeps the SAME id across retries, so the server can dedup. */
interface RpcRequest {
  requestId: number;
  /** The amount this request intends to add to the server's balance — a stand-in
   *  for any non-idempotent side effect (charge a card, append a row). */
  amount: number;
}

/**
 * RPC server with idempotent request handling + liveness heartbeats.
 *
 * Two responsibilities the chapter pairs deliberately:
 *  - Apply each logical request's side effect EXACTLY ONCE despite at-least-once
 *    delivery, by caching results per requestId (the dedup table). A naive
 *    server that applied on every arrival would over-charge by exactly the
 *    retry+duplicate count — which we measure against this one to make the point.
 *  - Emit periodic heartbeats so a client-side φ detector has a signal. When the
 *    node crash()es, heartbeats stop (the base class makes a down node inert),
 *    which is precisely the "is it slow or dead?" stimulus.
 */
class RpcServer extends Node {
  /** Idempotency / dedup table: requestId → the ack we already computed. A
   *  second arrival of the same id replays this instead of re-applying. This is
   *  the exactly-once-effect mechanism over at-least-once delivery. */
  private readonly processed = new Map<number, number>();

  /** The protected side effect. Equals the sum of DISTINCT request amounts iff
   *  dedup works. If it ever exceeds that, a duplicate leaked through. */
  balance = 0;

  /** Instrumentation, not protocol: every onMessage call for a request, incl.
   *  duplicates — the inflated number we contrast with effects applied. */
  requestArrivals = 0;
  /** Times a request arrived whose id we'd already applied — the dedup saves. */
  duplicateArrivals = 0;

  // No explicit constructor: the base Node(id, clock, net) ctor (which also
  // self-registers the inbox) is exactly what we need, inherited verbatim.

  /** Begin emitting heartbeats. Self-rescheduling timer; stops automatically
   *  when the node crashes (setTimer fires are no-ops on a down node). */
  startHeartbeats(toClient: string): void {
    const beat = () => {
      this.sendTo(toClient, "Heartbeat", { sentAtMs: this.clock.now() });
      this.setTimer(HEARTBEAT_INTERVAL_MS, beat);
    };
    // First beat after one interval so inter-arrival samples are well-formed.
    this.setTimer(HEARTBEAT_INTERVAL_MS, beat);
  }

  override onMessage(from: string, msg: Message): void {
    if (msg.kind !== "RpcRequest") return; // server ignores anything else
    const req = msg.payload as RpcRequest;
    this.requestArrivals++;

    const cached = this.processed.get(req.requestId);
    if (cached !== undefined) {
      // Already applied: this is a retry or a network duplicate. Re-ack with the
      // SAME result, do NOT re-apply the side effect. This is the line that turns
      // at-least-once delivery into exactly-once effect.
      this.duplicateArrivals++;
      this.sendTo(from, "RpcReply", { requestId: req.requestId, balance: cached });
      return;
    }

    // First time we've seen this id: apply the effect, memoize, ack.
    this.balance += req.amount;
    this.processed.set(req.requestId, this.balance);
    this.sendTo(from, "RpcReply", { requestId: req.requestId, balance: this.balance });
  }
}

// --- RPC client + failure detector ------------------------------------------

/**
 * RPC client: issues idempotent requests with retry, and runs a φ-accrual
 * detector over the server's heartbeats.
 *
 * Retry policy is at-least-once: re-send the same requestId on a fixed interval
 * until an ack for it arrives. This is the only safe policy when a missing reply
 * is ambiguous (could be lost request, lost reply, or dead server) — and it is
 * exactly why the server must be idempotent.
 */
class RpcClient extends Node {
  private serverId = "";
  private nextRequestId = 0;
  /** requestId → amount, for requests we've sent but not yet had acked. The
   *  retry loop walks this; an ack deletes from it. Empty ⇒ all logical requests
   *  durably landed. */
  private readonly pending = new Map<number, number>();

  private readonly detector = new PhiAccrualDetector();

  /** φ threshold above which we declare the server SUSPECTED (dead). The single
   *  knob that trades detection speed against false-positive rate. */
  suspicionThreshold = 0;

  /** Set once when the detector first crosses threshold and we have NOT yet
   *  flipped to suspected — used to compute detection latency against crash time. */
  firstSuspectedAtMs = -1;
  /** Whether we currently believe the server is dead. */
  suspected = false;

  /** Instrumentation. acked = distinct logical requests confirmed landed. */
  ackedRequests = 0;
  /** Total request transmissions incl. retries — the client-side view of the
   *  at-least-once cost (matches server arrivals minus network duplicates). */
  requestSends = 0;

  /** φ samples taken at the detector's poll cadence, for plotting the trajectory
   *  and proving φ climbs after a crash. (timeMs, phi). */
  readonly phiTrace: Array<{ timeMs: number; phi: number }> = [];

  /** Observed heartbeat inter-arrival gaps (ms). Its p99 is the honest reason a
   *  tight timeout murders: the tail is what a "dead-or-slow?" timeout sits on
   *  top of. We print p50/p99 so the threshold choice is grounded in the actual
   *  distribution, not a guess. Last gap before crash is the detector's baseline. */
  readonly interArrival = new Histogram();
  private lastHeartbeatAtMs = -1;

  /** Count of DISTINCT suspicion episodes raised against a server we know is
   *  actually alive — the false positives ("murders") this stage exists to
   *  surface. Incremented on each rising edge into `suspected` while alive. */
  falsePositives = 0;

  /** Set by the scenario when the server is genuinely crashed, so the detector
   *  can label a suspicion as true-positive vs false-positive. The detector
   *  itself NEVER reads this (it can't, in reality) — it's test-only ground
   *  truth for scoring, not an input to the algorithm. */
  serverActuallyDown = false;

  // No explicit constructor: inherits Node(id, clock, net) verbatim.

  /** Kick off the workload: enqueue all logical requests, start the retry loop
   *  and the detector poll loop. */
  start(serverId: string): void {
    this.serverId = serverId;
    for (let i = 0; i < LOGICAL_REQUEST_COUNT; i++) {
      const id = this.nextRequestId++;
      // Deterministic amount so the expected balance is checkable: 1..N.
      this.pending.set(id, i + 1);
    }
    this.retryLoop();
    this.detectorLoop();
  }

  /** Re-send every still-pending request, then re-arm. Stops naturally once
   *  `pending` drains (we re-arm only while work remains) — no busy spin. */
  private retryLoop(): void {
    if (this.pending.size === 0) return; // all acked: nothing left to retry
    for (const [requestId, amount] of this.pending) {
      this.requestSends++;
      this.sendTo(this.serverId, "RpcRequest", { requestId, amount });
    }
    this.setTimer(RPC_RETRY_INTERVAL_MS, () => this.retryLoop());
  }

  /** Poll φ on a fixed cadence (faster than heartbeats so we observe φ climbing
   *  BETWEEN heartbeats, which is the whole point of accrual vs binary timeout). */
  private detectorLoop(): void {
    const nowMs = this.clock.now();
    const phi = this.detector.phi(nowMs);
    this.phiTrace.push({ timeMs: nowMs, phi });

    const overThreshold = phi >= this.suspicionThreshold && this.suspicionThreshold > 0;
    if (overThreshold && !this.suspected) {
      // Rising edge into suspicion. Score it against ground truth: if the server
      // is actually alive, we just murdered a healthy node.
      this.suspected = true;
      if (this.firstSuspectedAtMs < 0) this.firstSuspectedAtMs = nowMs;
      if (!this.serverActuallyDown) this.falsePositives++;
    } else if (!overThreshold && this.suspected) {
      // φ fell back below threshold (a late heartbeat finally arrived): recant.
      // This is how accrual self-corrects a transient slow patch — the binary
      // timeout cannot. Reset firstSuspectedAtMs only matters for live recants.
      this.suspected = false;
    }

    // Poll ~5x per heartbeat interval. Cheap (pure math) and gives a smooth φ
    // curve for the trace without flooding the event queue.
    this.setTimer(Math.max(1, Math.floor(HEARTBEAT_INTERVAL_MS / 5)), () => this.detectorLoop());
  }

  // `from` is part of the Node.onMessage contract but the client only ever talks
  // to one server, so the sender id carries no decision here — underscore-prefixed
  // to satisfy noUnusedParameters without dropping the contractual signature.
  override onMessage(_from: string, msg: Message): void {
    if (msg.kind === "Heartbeat") {
      // Feed the detector the ARRIVAL time (now), not the sentAt — the detector
      // models what it can observe (arrivals), and arrival jitter is exactly the
      // signal it must tolerate without false-positiving.
      const nowMs = this.clock.now();
      if (this.lastHeartbeatAtMs >= 0) this.interArrival.record(nowMs - this.lastHeartbeatAtMs);
      this.lastHeartbeatAtMs = nowMs;
      this.detector.recordHeartbeat(nowMs);
      return;
    }
    if (msg.kind === "RpcReply") {
      const { requestId } = msg.payload as { requestId: number; balance: number };
      if (this.pending.delete(requestId)) {
        // First ack for this id (idempotent on the client side too: a duplicate
        // ack finds nothing to delete and is harmless).
        this.ackedRequests++;
      }
      return;
    }
  }
}

// --- scenarios ---------------------------------------------------------------

const SEED = 7;

/** Sum 1..n — the balance the server MUST end at if every logical request's
 *  effect applied exactly once. */
function expectedBalance(n: number): number {
  return (n * (n + 1)) / 2;
}

/**
 * Scenario B — the false-positive demo. NO crash: the server stays healthy the
 * whole run. We deliberately add a fat one-way latency tail on the heartbeat
 * link and set the φ threshold BELOW what that tail warrants. The detector then
 * "murders" a perfectly alive node every time a heartbeat rides the tail.
 *
 * Then we re-run identically with the threshold raised above the tail and show
 * false positives drop to zero — the speed/correctness tradeoff, quantified.
 */
function runFalsePositiveSweep() {
  const results: Array<{ threshold: number; falsePositives: number; phiPeak: number }> = [];

  // Two thresholds: one too tight (murders the slow-but-healthy node) and one
  // sized past the tail (zero murders). Same seed, same faults — only the knob
  // changes, so the difference is attributable solely to the threshold.
  // Heartbeat link with a fat tail: jitter window = 300ms over a 100ms heartbeat
  // means an alive node's beats can arrive up to ~3 intervals late. φ≥1 ("~10%
  // chance I'm wrong") is too tight for that tail and murders; φ≥8 (the standard
  // Cassandra/Akka default, ~1-in-10^8 wrong) sits above the tail and never does.
  const HEARTBEAT_TAIL_MS = 300;
  for (const threshold of [1, 8]) {
    let client!: RpcClient;
    runScenario(`false-positive-sweep-threshold-${threshold}`, SEED, {
      untilMs: 4000,
      network: { defaultBaselineMs: 10, defaultJitterMs: 8 },
      setup(ctx) {
        const server = new RpcServer("server", ctx.clock, ctx.net);
        client = new RpcClient("client", ctx.clock, ctx.net);

        // Asymmetric tail on the server→client (heartbeat) direction only. Most
        // beats are ~10ms but some ride the tail and arrive hundreds of ms late.
        // A healthy node that merely hit the tail looks dead to a tight threshold
        // — the canonical "slow != dead" trap.
        ctx.net.setLatency("server", "client", 10, HEARTBEAT_TAIL_MS);

        // No drop, no duplicate, NO crash — the server is alive and well the
        // entire run. Any suspicion is therefore by definition a false positive.
        client.serverActuallyDown = false;
        client.suspicionThreshold = threshold;

        client.start("server");
        server.startHeartbeats("client");

        return () => {
          const phiPeak = client.phiTrace.reduce((m, p) => Math.max(m, p.phi), 0);
          results.push({ threshold, falsePositives: client.falsePositives, phiPeak: round2(phiPeak) });
          const ia = client.interArrival;
          return [
            { metric: "server crashed?", value: "no (healthy all run)" },
            { metric: "heartbeat link tail (jitter ms)", value: HEARTBEAT_TAIL_MS },
            { metric: "heartbeat inter-arrival p50 (ms)", value: round2(ia.percentile(0.5)) },
            { metric: "heartbeat inter-arrival p99 (ms)", value: round2(ia.percentile(0.99)) },
            { metric: "heartbeat inter-arrival max (ms)", value: round2(ia.max()) },
            { metric: "suspicion threshold", value: threshold },
            { metric: "phi peak (healthy jittery node)", value: round2(phiPeak) },
            { metric: "FALSE POSITIVES (healthy murders)", value: client.falsePositives },
          ];
        };
      },
    });
  }

  return results;
}

// --- small pure helpers ------------------------------------------------------

/** φ recorded nearest a target time, for labeling "φ at crash". Pure scan. */
function phiNear(trace: ReadonlyArray<{ timeMs: number; phi: number }>, targetMs: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (const p of trace) {
    const d = Math.abs(p.timeMs - targetMs);
    if (d < bestDist) {
      bestDist = d;
      best = p.phi;
    }
  }
  return best;
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/** Print a coarse ASCII φ trajectory so the reader SEES φ flat-then-spike at the
 *  crash, not just a latency number. Buckets the trace into ~30 columns and marks
 *  the threshold crossing. Pure rendering over the captured trace. */
function printPhiSparkline(
  trace: ReadonlyArray<{ timeMs: number; phi: number }>,
  crashAtMs: number,
  threshold: number,
): void {
  if (trace.length === 0) return;
  const cols = 50;
  const tMin = trace[0].timeMs;
  const tMax = trace[trace.length - 1].timeMs;
  const span = Math.max(1, tMax - tMin);
  // Bucket: max φ per column (a spike must survive downsampling, so we take max
  // not mean — a transient murder-worthy φ should never be averaged away).
  const buckets = new Array<number>(cols).fill(0);
  for (const p of trace) {
    const c = Math.min(cols - 1, Math.floor(((p.timeMs - tMin) / span) * cols));
    buckets[c] = Math.max(buckets[c], p.phi);
  }
  const ramp = " .:-=+*#%@";
  const phiMax = Math.max(threshold, ...buckets);
  const line = buckets
    .map((v) => ramp[Math.min(ramp.length - 1, Math.floor((v / phiMax) * (ramp.length - 1)))])
    .join("");
  const crashCol = Math.min(cols - 1, Math.floor(((crashAtMs - tMin) / span) * cols));
  const marker = Array.from({ length: cols }, (_, i) => (i === crashCol ? "^" : " ")).join("");
  console.log(`\nphi trajectory (max per column, ${ramp[ramp.length - 1]}=high):`);
  console.log(`  ${line}`);
  console.log(`  ${marker}  (^ = crash @ ${crashAtMs}ms, threshold φ≥${threshold})`);
}

// --- scenario A: idempotent RPC + φ crash detection --------------------------
// runScenario returns only a result struct, but the sparkline needs the client's
// full φ trace. We surface it through these module-level captures, set inside the
// setup closure during the single run (no second run — that would burn the seed
// stream differently and could desync the printed table from the sparkline).

let capturedTrace: ReadonlyArray<{ timeMs: number; phi: number }> = [];
let capturedCrashAtMs = 3000;
let capturedThreshold = 8;

/**
 * Scenario A — idempotent RPC under loss + duplication, plus a real crash the φ
 * detector catches. Threshold is sized past the latency tail so a healthy server
 * is never falsely suspected; the only suspicion raised is the true crash. We
 * report the dedup gap (arrivals vs effects) and the crash detection latency.
 */
function runScenarioAWithCapture() {
  let client!: RpcClient;
  const crashAtMs = 3000;
  const res = runScenario("idempotent-rpc-and-phi-crash-detection", SEED, {
    untilMs: 6000,
    network: { defaultBaselineMs: 10, defaultJitterMs: 8 },
    setup(ctx) {
      const server = new RpcServer("server", ctx.clock, ctx.net);
      client = new RpcClient("client", ctx.clock, ctx.net);
      ctx.net.dropRate(0.25);
      ctx.net.duplicate(true);
      client.suspicionThreshold = 8;
      ctx.at(crashAtMs, () => {
        server.crash();
        client.serverActuallyDown = true;
      });
      ctx.watch(
        invariant(
          "side-effect-applied-at-most-once-per-request",
          () => server.balance <= expectedBalance(LOGICAL_REQUEST_COUNT),
          () => ({ balance: server.balance, max: expectedBalance(LOGICAL_REQUEST_COUNT) }),
        ),
      );
      client.start("server");
      server.startHeartbeats("client");
      return () => {
        const detectionLatencyMs =
          client.firstSuspectedAtMs >= 0 ? client.firstSuspectedAtMs - crashAtMs : -1;
        const phiAtCrash = phiNear(client.phiTrace, crashAtMs);
        const phiPeak = client.phiTrace.reduce((m, p) => Math.max(m, p.phi), 0);
        return [
          { metric: "logical requests", value: LOGICAL_REQUEST_COUNT },
          { metric: "client request sends (incl. retries)", value: client.requestSends },
          { metric: "server request arrivals (incl. dups)", value: server.requestArrivals },
          { metric: "  of which duplicate (dedup saves)", value: server.duplicateArrivals },
          { metric: "side effects applied (server balance)", value: server.balance },
          { metric: "expected balance if exactly-once", value: expectedBalance(LOGICAL_REQUEST_COUNT) },
          { metric: "logical requests acked", value: client.ackedRequests },
          { metric: "crash injected at (ms)", value: crashAtMs },
          { metric: "phi at crash time (healthy)", value: round2(phiAtCrash) },
          { metric: "phi peak after crash", value: round2(phiPeak) },
          { metric: "suspicion threshold", value: client.suspicionThreshold },
          { metric: "detection latency (ms)", value: detectionLatencyMs },
          { metric: "false positives (healthy node murders)", value: client.falsePositives },
        ];
      };
    },
  });
  capturedTrace = client.phiTrace;
  capturedCrashAtMs = crashAtMs;
  capturedThreshold = client.suspicionThreshold;
  return res;
}

function main(): void {
  console.log("================================================================");
  console.log("Stage 03 — RPC & failure detection: telling \"slow\" from \"dead\"");
  console.log("================================================================");

  console.log(
    "\n[Part 1] Idempotent RPC under loss+duplication, with a REAL crash the\n" +
      "φ-accrual detector catches. Retries inflate server arrivals; dedup keeps\n" +
      "the side effect exactly-once. Threshold sized past the latency tail ⇒ the\n" +
      "ONLY suspicion is the true crash.",
  );
  runScenarioAWithCapture();
  printPhiSparkline(capturedTrace, capturedCrashAtMs, capturedThreshold);

  console.log(
    "\n----------------------------------------------------------------\n" +
      "[Part 2] FAILURE MODE — false positives. Server stays HEALTHY all run,\n" +
      "but the heartbeat link has a fat latency tail. A φ threshold set BELOW the\n" +
      "tail murders the live node; raising it past the tail drives murders to 0.\n" +
      "Same seed, same faults — only the threshold changes.",
  );
  const sweep = runFalsePositiveSweep();

  console.log("\n=== False-positive tradeoff summary (healthy server, seed=7) ===");
  for (const r of sweep) {
    const verdict = r.falsePositives > 0 ? "MURDERS healthy node" : "no murders";
    console.log(
      `  threshold φ≥${r.threshold}: peak φ=${r.phiPeak}, false positives=${r.falsePositives}  (${verdict})`,
    );
  }

  console.log(
    "\nReading: a tight threshold detects a real crash faster but cannot tell a\n" +
      "tail-latency heartbeat from a death, so it murders healthy nodes. A loose\n" +
      "threshold never murders but detects real crashes slower. φ-accrual exposes\n" +
      "that dial directly — the number you pick IS your false-positive budget.",
  );
  console.log(
    "\nHonesty: latencies are simulated (seeded jitter, not a real NIC), so\n" +
      "absolute ms are optimistic. What transfers: φ stays low under healthy\n" +
      "jitter and spikes after a true crash, and threshold-below-tail ⇒ murders.",
  );
}

main();
