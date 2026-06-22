// core/network.ts — in-memory message bus with deterministic fault injection.
//
// Why a fake network instead of real sockets: the entire book is about what
// happens when the network MISBEHAVES — drops, delays, reorders, duplicates,
// partitions. Real sockets give you none of that on demand and reproducibly; a
// fake bus lets us say "drop 30% of messages on link A↔B, then partition {A,B}
// from {C,D,E} for 5 simulated seconds, then heal" and replay it byte-for-byte.
// This is where the book earns "the failures are real": they are the actual
// FLP/CAP-relevant faults, just injected deterministically instead of waited for.
//
// Model: send(from,to,msg) computes a delivery delay from a per-link latency
// distribution (baseline + jitter, both from the seeded Rng) and schedules the
// delivery on the shared SimClock. Delivery calls the recipient Node's
// onMessage. Faults are checked AT SEND TIME against current config — so toggling
// partition() between events changes the fate of subsequent messages, exactly
// like a real cut.
//
// Invariant we protect: a message is accounted exactly once — it is counted as
// sent, then exactly one of delivered/dropped. If that ledger ever disagrees
// (sent != delivered + dropped + in-flight), a stage's message-cost claim is
// wrong, so stages assert on it. The nastiest failure mode a message bus can
// have is silently losing a message it didn't mean to (counted sent, never
// delivered, never dropped) — the ledger exists to make that impossible to hide.
//
// Determinism: ALL randomness (delay, drop coin, reorder permutation, duplicate
// coin) flows through the injected Rng. Same seed ⇒ same fate for every message.

import type { SimClock } from "./clock.js";
import type { Rng } from "./prng.js";
import type { Stats } from "./metrics.js";

/** A message in flight. `kind` is the protocol verb (RequestVote, AppendEntries,
 *  Heartbeat, Gossip…); `payload` is opaque to the bus — the bus routes and
 *  delays, it never inspects protocol semantics. That separation is why the same
 *  Network serves every chapter unchanged. */
export interface Message {
  from: string;
  to: string;
  kind: string;
  payload: unknown;
}

/** Receiver side: a Node registers a delivery callback. Network never imports
 *  Node (avoids a cycle) — it only needs "given an id, who do I call". */
export type DeliverFn = (msg: Message) => void;

interface LinkConfig {
  baselineMs: number; // fixed propagation floor for this ordered pair
  jitterMs: number; // uniform extra delay in [0, jitterMs)
}

export interface NetworkOptions {
  /** Default one-way latency floor for any link not explicitly configured. */
  defaultBaselineMs?: number;
  /** Default jitter window. Real networks have variance; zero jitter would make
   *  every race resolve identically and hide timing-dependent bugs. */
  defaultJitterMs?: number;
}

export class Network {
  private deliver = new Map<string, DeliverFn>();
  private links = new Map<string, LinkConfig>(); // key: `${from}->${to}`
  private globalDropRate = 0;
  private duplicateOn = false;
  private reorderOn = false;
  // Partition state: a node id → group id. Two nodes can talk iff same group.
  // Map being empty means "one big group" (fully connected).
  private partitionGroup = new Map<string, number>();

  private readonly defaultBaselineMs: number;
  private readonly defaultJitterMs: number;

  constructor(
    private readonly clock: SimClock,
    private readonly rng: Rng,
    private readonly stats: Stats,
    opts: NetworkOptions = {},
  ) {
    this.defaultBaselineMs = opts.defaultBaselineMs ?? 10;
    this.defaultJitterMs = opts.defaultJitterMs ?? 5;
  }

  /** Register a node's inbox. Called once per node at cluster setup. */
  register(id: string, fn: DeliverFn): void {
    if (this.deliver.has(id)) throw new Error(`register: duplicate node id "${id}"`);
    this.deliver.set(id, fn);
  }

  // --- fault injection knobs --------------------------------------------------

  /** Pin a specific one-way link's latency (e.g. a slow cross-region hop). The
   *  asymmetry is intentional: real links are not symmetric, and asymmetric
   *  latency is a classic source of false failure-detector positives. */
  setLatency(from: string, to: string, baselineMs: number, jitterMs = 0): void {
    this.links.set(`${from}->${to}`, { baselineMs, jitterMs });
  }

  /** Probability in [0,1] that any message is silently dropped. Models lossy
   *  links; combined with retries it's how stages exercise at-least-once vs
   *  exactly-once. dropRate(0) is a perfectly reliable network. */
  dropRate(p: number): void {
    if (p < 0 || p > 1) throw new Error(`dropRate: p in [0,1], got ${p}`);
    this.globalDropRate = p;
  }

  /** When on, each delivered message MAY be delivered a second time (coin per
   *  message). Forces protocols to be idempotent — non-idempotent handlers break
   *  visibly here, which is the teaching point. */
  duplicate(on: boolean): void {
    this.duplicateOn = on;
  }

  /** When on, messages scheduled in the same tick get a shuffled extra jitter so
   *  delivery order need not match send order. Surfaces ordering assumptions
   *  (e.g. a protocol that assumes FIFO channels). */
  reorder(on: boolean): void {
    this.reorderOn = on;
  }

  /** Split the cluster into two groups that cannot exchange messages. This is THE
   *  CAP fault. Messages across the cut are counted as dropped (the sender can't
   *  tell drop from partition — that indistinguishability is the heart of FLP).
   *  Nodes not listed stay in their previous group (or the default group 0). */
  partition(groupA: string[], groupB: string[]): void {
    for (const id of groupA) this.partitionGroup.set(id, 0);
    for (const id of groupB) this.partitionGroup.set(id, 1);
  }

  /** Remove all partitions — every node back in one group. The "heal" event.
   *  Note: messages dropped DURING the partition are gone; only NEW messages
   *  flow. Reconciling state after heal is the protocol's job, not the bus's. */
  heal(): void {
    this.partitionGroup.clear();
  }

  // --- the send path ----------------------------------------------------------

  /** Send a message. Computes fate (drop? partitioned? delayed how long?) NOW
   *  against current fault config, then schedules delivery on the SimClock.
   *  Every outcome is recorded in Stats so the message ledger stays balanced. */
  send(from: string, to: string, kind: string, payload: unknown): void {
    this.stats.inc("msg.sent");

    // Partitioned pair? Indistinguishable from a drop, by design (and counted as
    // such) — a node on one side genuinely cannot tell the other crashed vs the
    // link cut. This is the assumption every consensus algorithm must survive.
    if (this.isPartitioned(from, to)) {
      this.stats.inc("msg.dropped");
      this.stats.inc("msg.dropped.partition");
      return;
    }

    // Lossy link coin-flip. Same seed ⇒ same message drops, so a "lost heartbeat
    // → spurious election" bug reproduces exactly.
    if (this.rng.bool(this.globalDropRate)) {
      this.stats.inc("msg.dropped");
      this.stats.inc("msg.dropped.loss");
      return;
    }

    this.scheduleDelivery({ from, to, kind, payload });

    // Duplicate AFTER scheduling the original: a duplicate is a SECOND delivery,
    // independently delayed, so it can arrive before OR after the original —
    // which is exactly what makes duplicates hard. Counted as its own send so the
    // ledger reflects the extra packet on the wire.
    if (this.duplicateOn && this.rng.bool(0.5)) {
      this.stats.inc("msg.sent");
      this.stats.inc("msg.duplicated");
      this.scheduleDelivery({ from, to, kind, payload });
    }
  }

  private scheduleDelivery(msg: Message): void {
    const link = this.links.get(`${msg.from}->${msg.to}`);
    const baseline = link ? link.baselineMs : this.defaultBaselineMs;
    const jitterWindow = link ? link.jitterMs : this.defaultJitterMs;
    // Base delay + uniform jitter. Reorder adds an EXTRA random slice so two
    // messages sent back-to-back can swap delivery order; without it, equal-delay
    // messages keep FIFO (SimClock's seq tiebreak) and reorder would be a no-op.
    let delay = baseline + (jitterWindow > 0 ? this.rng.int(jitterWindow) : 0);
    if (this.reorderOn) delay += this.rng.int(jitterWindow + 1);

    this.clock.schedule(delay, () => {
      const inbox = this.deliver.get(msg.to);
      if (!inbox) {
        // Recipient never registered: a topology bug in the stage, not a network
        // event. Surface it loudly rather than counting a phantom delivery.
        throw new Error(`Network: no registered node "${msg.to}" for message ${msg.kind}`);
      }
      this.stats.inc("msg.delivered");
      inbox(msg);
    });
  }

  private isPartitioned(from: string, to: string): boolean {
    if (this.partitionGroup.size === 0) return false; // fully connected
    const ga = this.partitionGroup.get(from) ?? 0;
    const gb = this.partitionGroup.get(to) ?? 0;
    return ga !== gb;
  }
}
