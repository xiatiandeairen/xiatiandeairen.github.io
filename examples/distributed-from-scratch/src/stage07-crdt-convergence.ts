// stage07-crdt-convergence.ts — eventual consistency via CRDTs: let concurrent
// writes converge automatically, with no coordination, no leader, no locks.
//
// The chapter's claim, made executable: if every replica's merge function is a
// join over a semilattice — commutative, associative, idempotent — then replicas
// that exchange state in ANY order, ANY number of times, with ANY messages
// dropped, all converge to the SAME state once they have all seen each other's
// updates. No consensus needed. That is the entire CRDT bargain.
//
// Why CRDT state lives as PURE functions here (GCounter / ORSet below), separate
// from the gossiping Node: the convergence properties are properties of the MERGE
// FUNCTION, not of the network. So we prove them where they live — by feeding the
// same set of operations into thousands of random merge orderings and asserting
// every ordering reaches byte-identical state. The network layer (GossipReplica +
// the runScenario demo) then shows the SAME state surviving a real partition +
// reorder + duplicate + heal, which is the realistic packaging of the same math.
//
// What "real failure" looks like in this chapter: the WRONG abstraction silently
// passes happy-path tests. An OR-Set whose remove just deletes the element
// (instead of recording which add-tags it observed) is NOT a CRDT — its merge is
// not commutative. We ship that broken version (BadOpReplica, an op-based set
// whose remove is a destructive delete) and show two replicas converging to
// DIFFERENT sets under "A removes x, B concurrently adds x" with reordered
// delivery. Then the correct add-wins / observed-remove OR-Set (GoodOpReplica /
// OrSet) converges to one set. Same seed reproduces both.
//
// Honesty note on numbers: latencies/rounds are SIMULATED (SimClock virtual ms),
// so absolute timings are not wall-clock — they are deterministic event counts.
// The load-bearing, transferable facts are the RELATIONAL ones: convergence
// rounds needed, "all orderings agree" (a boolean over real random trials), and
// "LWW drops a write that CRDT keeps" (a real diff between two final states).

import { runScenario } from "./core/scenario.js";
import { Node } from "./core/node.js";
import { invariant } from "./core/assert.js";
import { seededRng, type Rng } from "./core/prng.js";
import type { Message } from "./core/network.js";

const SEED = 7;

// ===========================================================================
// Section 1 — CRDT state types as pure join-semilattices.
//
// Each type exposes: a value() projection (what the application sees), local
// mutation ops, and merge(other) → the least upper bound. The merge MUST be
// commutative + associative + idempotent; that is the contract Section 2 tests.
// ===========================================================================

/** Grow-only counter (G-Counter). State = per-replica contribution vector; the
 *  observed count is the sum. Merge takes the element-wise MAX of the vectors.
 *
 *  Why per-replica entries instead of one shared integer: a single `count++`
 *  applied on two partitioned replicas and naively merged would either double-
 *  count or lose an increment — there is no way to tell "we both saw 5 then both
 *  +1 → should be 7" from the scalar 6 alone. Splitting the count by AUTHOR makes
 *  each replica's contribution independently mergeable: max is correct because a
 *  replica's own entry only ever grows and only it writes that entry. */
class GCounter {
  // replicaId → that replica's monotonically increasing local count. Invariant:
  // entry[r] is written ONLY by replica r (so it never needs reconciliation —
  // the writer is always strictly ahead of any copy others hold).
  private readonly contributions: Map<string, number>;

  constructor(entries?: Iterable<[string, number]>) {
    this.contributions = new Map(entries);
  }

  /** Local increment by replica `who`. Only `who` may grow its own entry. */
  increment(who: string, by = 1): void {
    if (by < 0) throw new Error("GCounter is grow-only; got negative increment");
    this.contributions.set(who, (this.contributions.get(who) ?? 0) + by);
  }

  /** Observed count = sum of all replica contributions. */
  value(): number {
    let sum = 0;
    for (const v of this.contributions.values()) sum += v;
    return sum;
  }

  /** Least upper bound: element-wise max. Commutative/associative because max is;
   *  idempotent because max(a,a)=a. This is the whole CRDT in one line. */
  merge(other: GCounter): void {
    for (const [who, count] of other.contributions) {
      this.contributions.set(who, Math.max(this.contributions.get(who) ?? 0, count));
    }
  }

  clone(): GCounter {
    return new GCounter(this.contributions);
  }

  /** Canonical, order-independent fingerprint so two states can be compared for
   *  EQUALITY regardless of internal map iteration order. Equal fingerprint ⇔
   *  converged. */
  fingerprint(): string {
    return [...this.contributions.entries()]
      .filter(([, c]) => c > 0)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([k, c]) => `${k}:${c}`)
      .join(",");
  }
}

/** A unique add-tag ("dot"): which replica added an element, and its local
 *  sequence number at that moment. Two adds of the same element produce two
 *  DISTINCT dots — that is what lets the set distinguish "added once, removed"
 *  from "added twice, removed once". */
interface Dot {
  replica: string;
  seq: number;
}

function dotKey(d: Dot): string {
  return `${d.replica}#${d.seq}`;
}

/** OR-Set (Observed-Remove Set), add-wins. The element x is "in the set" iff it
 *  has at least one live add-dot. remove(x) does NOT delete x; it records the set
 *  of x's currently-observed dots as tombstones. A concurrent add (a NEW dot that
 *  the remover never saw) therefore survives the merge — add wins over a
 *  concurrent remove.
 *
 *  Why this convoluted dot bookkeeping is the POINT: it is exactly what makes
 *  merge commutative. "remove just deletes the key" (see BadOpReplica in the
 *  failure-mode section) loses the information needed to reconcile a concurrent
 *  add, so its merge result depends on delivery order — not a CRDT. */
class OrSet {
  // element → set of live add-dots (dotKey). Element present iff this set non-empty.
  private readonly adds = new Map<string, Set<string>>();
  // dotKeys that have been removed (tombstones). A dot in here is dead even if it
  // still appears in `adds` from a stale merge — removedDots is the veto.
  private readonly removedDots = new Set<string>();
  // per-replica local dot counter, so each add() mints a globally-unique dot.
  private readonly seqByReplica = new Map<string, number>();

  /** Local add by replica `who`: mint a fresh unique dot for this element. */
  add(who: string, element: string): void {
    const seq = (this.seqByReplica.get(who) ?? 0) + 1;
    this.seqByReplica.set(who, seq);
    const dk = dotKey({ replica: who, seq });
    const dots = this.adds.get(element) ?? new Set<string>();
    dots.add(dk);
    this.adds.set(element, dots);
  }

  /** Local remove: tombstone exactly the dots this replica currently observes for
   *  `element`. Dots it has NOT yet seen (a concurrent remote add) are untouched
   *  → that add wins on merge. Observed-remove in one method. */
  remove(element: string): void {
    const dots = this.adds.get(element);
    if (!dots) return; // removing an element we never observed is a no-op, not an error
    for (const dk of dots) this.removedDots.add(dk);
  }

  /** Element is present iff it has at least one add-dot not yet tombstoned. */
  has(element: string): boolean {
    const dots = this.adds.get(element);
    if (!dots) return false;
    for (const dk of dots) if (!this.removedDots.has(dk)) return true;
    return false;
  }

  value(): string[] {
    const out: string[] = [];
    for (const el of this.adds.keys()) if (this.has(el)) out.push(el);
    return out.sort();
  }

  /** Merge = union of add-dots ∪ union of tombstones. Both are grow-only sets, so
   *  this is a join: commutative, associative, idempotent. Presence is then
   *  re-derived (a dot in both adds and removedDots is dead) — which is why a
   *  concurrent add (its dot only ever in adds, never tombstoned) survives. */
  merge(other: OrSet): void {
    for (const [el, dots] of other.adds) {
      const mine = this.adds.get(el) ?? new Set<string>();
      for (const dk of dots) mine.add(dk);
      this.adds.set(el, mine);
    }
    for (const dk of other.removedDots) this.removedDots.add(dk);
    // seqByReplica need NOT merge for correctness (dots already carry identity);
    // we keep replicas' own counters monotone so future local adds stay unique.
    for (const [r, s] of other.seqByReplica) {
      this.seqByReplica.set(r, Math.max(this.seqByReplica.get(r) ?? 0, s));
    }
  }

  clone(): OrSet {
    const c = new OrSet();
    for (const [el, dots] of this.adds) c.adds.set(el, new Set(dots));
    for (const dk of this.removedDots) c.removedDots.add(dk);
    for (const [r, s] of this.seqByReplica) c.seqByReplica.set(r, s);
    return c;
  }

  /** Order-independent fingerprint of the OBSERVABLE value (the set members).
   *  Two replicas converge ⇔ equal fingerprint. We fingerprint the value, not
   *  the internal dots, because users care about set membership — though here
   *  the correct OrSet also converges on internal state. */
  fingerprint(): string {
    return this.value().join(",");
  }

  /** Serialize to a JSON-able snapshot for the message bus (which treats payload
   *  as opaque). The receiver rebuilds a real OrSet via load() and merges it — we
   *  round-trip through the wire so the demo models "state crosses the network",
   *  not "two replicas alias one object". */
  dump(): OrSetWire {
    return {
      adds: [...this.adds.entries()].map(([el, dots]) => [el, [...dots]]),
      removedDots: [...this.removedDots],
      seq: [...this.seqByReplica.entries()],
    };
  }

  static load(w: OrSetWire): OrSet {
    const s = new OrSet();
    for (const [el, dots] of w.adds) s.adds.set(el, new Set(dots));
    for (const dk of w.removedDots) s.removedDots.add(dk);
    for (const [r, n] of w.seq) s.seqByReplica.set(r, n);
    return s;
  }
}

/** Wire-format for an OrSet snapshot: plain JSON-able structure so it survives the
 *  opaque message bus. */
interface OrSetWire {
  adds: [string, string[]][];
  removedDots: string[];
  seq: [string, number][];
}

// The pseudo-CRDT (a remove that keeps NO tombstone/dot) is modeled in Section 3
// as an OPERATION-BASED replica (BadOpReplica), not a state-based class: a
// state-based union of present elements is trivially commutative and only ever
// converges to the WRONG value, whereas the spec's target failure is two replicas
// converging to DIFFERENT sets — an op-ordering phenomenon. See "Failure mode:
// OPERATION-BASED replication" below for why.

/** Last-Writer-Wins register/set element — the chapter-6 baseline. Each element
 *  carries a (timestamp, replica) version; on conflict the higher (ts, replica)
 *  wins and the loser's write is DISCARDED. This is a valid CRDT (LWW is a
 *  semilattice on the version order) but it has a cost CRDTs avoid: concurrent
 *  writes are resolved by THROWING ONE AWAY. We use it only to measure the write
 *  that LWW loses where the OR-Set's add-wins keeps it. */
interface LwwEntry {
  present: boolean; // true=add, false=remove
  tsMs: number;
  replica: string;
}

class LwwSet {
  private readonly entries = new Map<string, LwwEntry>();

  private dominates(a: LwwEntry, b: LwwEntry): boolean {
    // Total order on (tsMs, replica) breaks ties deterministically — without the
    // replica tiebreak, two same-timestamp writes would be incomparable and LWW
    // would not be a function. The tiebreak is arbitrary but DETERMINISTIC, which
    // is exactly how it silently drops one of two concurrent writes.
    if (a.tsMs !== b.tsMs) return a.tsMs > b.tsMs;
    return a.replica > b.replica;
  }

  private write(element: string, e: LwwEntry): void {
    const cur = this.entries.get(element);
    if (!cur || this.dominates(e, cur)) this.entries.set(element, e);
  }

  add(who: string, element: string, tsMs: number): void {
    this.write(element, { present: true, tsMs, replica: who });
  }

  remove(who: string, element: string, tsMs: number): void {
    this.write(element, { present: false, tsMs, replica: who });
  }

  has(element: string): boolean {
    return this.entries.get(element)?.present ?? false;
  }

  value(): string[] {
    const out: string[] = [];
    for (const [el, e] of this.entries) if (e.present) out.push(el);
    return out.sort();
  }

  merge(other: LwwSet): void {
    for (const [el, e] of other.entries) this.write(el, e);
  }

  clone(): LwwSet {
    const c = new LwwSet();
    for (const [el, e] of this.entries) c.entries.set(el, { ...e });
    return c;
  }

  fingerprint(): string {
    return this.value().join(",");
  }
}

// ===========================================================================
// Section 2 — Pure algebraic tests of the merge law.
//
// These run WITHOUT the network. The claim "converges under any order, any
// duplication" reduces to: feeding a fixed multiset of replica states into the
// merge in many random orders (and re-merging duplicates) yields one final state.
// We test that directly over hundreds of random orderings — N>=1 lucky pass is
// not enough (see know: feasibility needs >=3 independent samples), so we run a
// large trial count and assert EVERY trial agrees.
// ===========================================================================

interface Mergeable<T> {
  merge(other: T): void;
  clone(): T;
  fingerprint(): string;
}

/** Fold a list of replica states into one, in the given index order, optionally
 *  re-merging each state twice (to exercise idempotency on the path, not just at
 *  the endpoints). Starts from a fresh empty accumulator so the base state never
 *  biases order. */
function foldMerge<T extends Mergeable<T>>(
  empty: () => T,
  states: T[],
  order: number[],
  duplicateEach: boolean,
): T {
  const acc = empty();
  for (const i of order) {
    acc.merge(states[i].clone());
    if (duplicateEach) acc.merge(states[i].clone()); // idempotency: second merge is a no-op
  }
  return acc;
}

interface ConvergenceReport {
  type: string;
  trials: number;
  distinctOutcomes: number;
  converged: boolean;
  finalFingerprint: string;
}

/** Run `trials` random merge orderings (with random duplication) over the SAME
 *  set of divergent replica states; assert they all collapse to one fingerprint.
 *  distinctOutcomes > 1 means the merge law is broken (the failure-mode signal). */
function checkMergeLaw<T extends Mergeable<T>>(
  typeName: string,
  empty: () => T,
  states: T[],
  rng: Rng,
  trials: number,
): ConvergenceReport {
  const outcomes = new Set<string>();
  for (let t = 0; t < trials; t++) {
    const order = rng.shuffle([...states.keys()]);
    const dup = rng.bool(0.5);
    outcomes.add(foldMerge(empty, states, order, dup).fingerprint());
  }
  return {
    type: typeName,
    trials,
    distinctOutcomes: outcomes.size,
    converged: outcomes.size === 1,
    finalFingerprint: [...outcomes][0] ?? "(none)",
  };
}

// ===========================================================================
// Section 3 — The realistic packaging: gossiping replicas under a real partition.
//
// Each replica holds a CRDT, applies local ops on a schedule, and periodically
// gossips its FULL state to peers (anti-entropy). On receiving a peer's state it
// merges. The network drops, reorders, and duplicates gossip messages, and we
// cut the cluster in half for a window. The claim: after heal, a few more gossip
// rounds and every replica's fingerprint is identical — measured, not asserted by
// fiat.
// ===========================================================================

const GOSSIP_INTERVAL_MS = 50;

/** A replica that gossips a generic mergeable CRDT. Generic over the CRDT so the
 *  same node drives both the good OrSet demo and (in a separate run) any other
 *  type — but here we instantiate it concretely for the OrSet convergence demo. */
class OrSetReplica extends Node {
  readonly state = new OrSet();

  constructor(
    id: string,
    clock: ConstructorParameters<typeof Node>[1],
    net: ConstructorParameters<typeof Node>[2],
    private readonly peers: string[],
  ) {
    super(id, clock, net);
  }

  /** Begin periodic anti-entropy. Idempotent gossip: re-sending full state is
   *  always safe because merge is idempotent — the reason we can be this cavalier
   *  about drops/dups is the merge law, not retransmission bookkeeping. */
  startGossip(): void {
    // We never cancel the timer (gossip runs for the whole experiment, bounded by
    // the scenario deadline), so the token is intentionally discarded.
    this.setTimer(GOSSIP_INTERVAL_MS, () => this.onGossipTick());
  }

  private onGossipTick(): void {
    for (const peer of this.peers) {
      if (peer === this.id) continue;
      // Send a serialized snapshot of our state (dots + tombstones). The receiver
      // rebuilds a fresh OrSet and merges a copy, never a shared reference.
      this.sendTo(peer, "gossip", this.state.dump());
    }
    this.setTimer(GOSSIP_INTERVAL_MS, () => this.onGossipTick());
  }

  override onMessage(_from: string, msg: Message): void {
    if (msg.kind !== "gossip") return;
    // Duplicate delivery (network fault) is harmless: merging the same snapshot
    // twice is a no-op by idempotency. THIS is the property under test.
    this.state.merge(OrSet.load(msg.payload as OrSetWire));
  }

  applyAdd(element: string): void {
    this.state.add(this.id, element);
  }

  applyRemove(element: string): void {
    this.state.remove(element);
  }
}

// ===========================================================================
// Main — run all four demonstrations in order, printing real numbers.
// ===========================================================================

const REPLICA_IDS = ["r0", "r1", "r2", "r3"];

/** Demo A: G-Counter convergence — concurrent increments, merge in random orders,
 *  all agree on the sum, and the sum equals the total increments applied (no lost
 *  or double counts). */
function demoGCounter(rng: Rng): ConvergenceReport & { sum: number; expected: number } {
  // Each replica independently increments its own entry while "partitioned".
  const states: GCounter[] = [];
  let expected = 0;
  for (let r = 0; r < REPLICA_IDS.length; r++) {
    const g = new GCounter();
    const bumps = 1 + rng.int(5); // 1..5 local increments
    for (let i = 0; i < bumps; i++) g.increment(REPLICA_IDS[r]);
    expected += bumps;
    states.push(g);
  }
  const report = checkMergeLaw("GCounter", () => new GCounter(), states, rng, 500);
  // Recompute the converged sum from one ordering.
  const merged = foldMerge(() => new GCounter(), states, [...states.keys()], true);
  return { ...report, sum: merged.value(), expected };
}

/** Demo B: OR-Set convergence under concurrent add/remove across replicas, then
 *  the same algebra check over random orders. */
function demoOrSet(rng: Rng): ConvergenceReport & { value: string } {
  const states: OrSet[] = [];
  // Build divergent replica states: each replica adds/removes different items,
  // and crucially r1 removes "x" while r2 concurrently re-adds "x" — the canonical
  // add-wins case.
  const r0 = new OrSet();
  r0.add("r0", "x");
  r0.add("r0", "y");
  const r1 = r0.clone(); // r1 starts having observed x,y then removes x
  r1.remove("x");
  const r2 = r0.clone(); // r2 concurrently adds x AGAIN (a new dot r1 never saw)
  r2.add("r2", "x");
  r2.add("r2", "z");
  const r3 = new OrSet();
  r3.add("r3", "w");
  states.push(r0, r1, r2, r3);

  const report = checkMergeLaw("OrSet", () => new OrSet(), states, rng, 500);
  const merged = foldMerge(() => new OrSet(), states, [...states.keys()], true);
  return { ...report, value: `{${merged.value().join(",")}}` };
}

// --- Failure mode: OPERATION-BASED replication exposes the broken merge --------
//
// Why op-based here (not state-based) for the failure demo: a state-based union of
// "present elements" is trivially commutative — it CANNOT diverge, it just
// silently converges to the WRONG value (a removed element resurrected). The
// spec's target failure is sharper: two replicas converge to DIFFERENT sets (one
// has x, one doesn't) under reordered delivery. That is an OP-BASED phenomenon: a
// replica applies remote OPERATIONS (add/remove) as they arrive, and a
// destructive remove that has no tombstone produces an order-dependent result.
//
// The contract an op-based CRDT needs: operations must COMMUTE. OrSet ops commute
// because each carries its dot — "remove" tombstones a specific dot, so applying
// it before or after the matching "add" yields the same live-dot set. BadOrSet's
// remove is a plain delete keyed only by element, so "delete x" then "insert x"
// ≠ "insert x" then "delete x". Reorder the two and the two replicas split.

/** An operation broadcast to a replica. The broken replica (BadOpReplica) acts on
 *  the element name alone (that's the bug — no dot/causal context), so add/remove
 *  of the same element do not commute. The correct replica (GoodOpReplica) acts on
 *  the specific dots, so its ops commute. */
type SetOp =
  | { kind: "add"; element: string; dot: string }
  | { kind: "remove"; element: string; dots: string[] };

/** Op-based replica interface: apply a remote op, read the value. */
interface OpReplica {
  apply(op: SetOp): void;
  value(): string[];
}

/** Op-based view over BadOrSet: remove is a destructive delete that ignores dots
 *  entirely (the bug). add/remove of the same element therefore do NOT commute. */
class BadOpReplica implements OpReplica {
  private readonly present = new Set<string>();
  apply(op: SetOp): void {
    if (op.kind === "add") this.present.add(op.element);
    else this.present.delete(op.element); // dots ignored — destructive, non-commutative
  }
  value(): string[] {
    return [...this.present].sort();
  }
}

/** Op-based view over OrSet's dot model: add installs its dot; remove tombstones
 *  exactly the dots it names. Because each op references specific dots, applying
 *  them in any order yields the same {live dots} → ops commute → convergence. */
class GoodOpReplica implements OpReplica {
  private readonly dots = new Map<string, Set<string>>(); // element → live dots
  private readonly tombstoned = new Set<string>();
  apply(op: SetOp): void {
    if (op.kind === "add") {
      const s = this.dots.get(op.element) ?? new Set<string>();
      s.add(op.dot);
      this.dots.set(op.element, s);
    } else {
      for (const d of op.dots) this.tombstoned.add(d);
    }
  }
  value(): string[] {
    const out: string[] = [];
    for (const [el, s] of this.dots) {
      for (const d of s) {
        if (!this.tombstoned.has(d)) {
          out.push(el);
          break;
        }
      }
    }
    return out.sort();
  }
}

/** The canonical concurrent history as a stream of ops both replicas must apply
 *  (eventual delivery = both see all ops), but in a per-replica delivery ORDER.
 *  Causal ancestor: origin added x (dot ox), seen by both. Then concurrently:
 *    - ra removes the x it observed → op removes dot {ox}
 *    - rb adds x again → op adds a fresh dot rx
 *  A correct add-wins OR-Set keeps x (rx never tombstoned) under either order;
 *  BadOrSet's remove deletes "x" wholesale, so order decides who ends with x. */
const ORIGIN_DOT = "ox";
const RB_DOT = "rx";
const CONCURRENT_OPS: SetOp[] = [
  { kind: "remove", element: "x", dots: [ORIGIN_DOT] }, // ra's observed-remove
  { kind: "add", element: "x", dot: RB_DOT }, // rb's concurrent fresh add
];

/** Apply the origin add then the two concurrent ops in `order` to a fresh replica;
 *  return its observable value. The origin add is causally-before both, so it is
 *  always applied first; only the two concurrent ops are reordered. */
function replayOps(order: number[], make: () => OpReplica): string {
  const r = make();
  r.apply({ kind: "add", element: "x", dot: ORIGIN_DOT }); // causal ancestor, same for all
  for (const i of order) r.apply(CONCURRENT_OPS[i]);
  return `{${r.value().join(",")}}`;
}

const OP_ORDERS: number[][] = [
  [0, 1], // remove then add
  [1, 0], // add then remove
];

/** Demo C: feed the two reorderable concurrent ops to two replicas in OPPOSITE
 *  delivery orders and compare their final sets. Two distinct sets ⇒ the type is
 *  not a CRDT (ops don't commute). We enumerate both orders exhaustively. */
function demoBadOrSet(): {
  badRaSet: string;
  badRbSet: string;
  badDiverged: boolean;
  goodRaSet: string;
  goodRbSet: string;
  goodConverged: boolean;
} {
  // ra applies in order OP_ORDERS[0], rb in OP_ORDERS[1] — the "A删 / B加 乱序" cut.
  const badRa = replayOps(OP_ORDERS[0], () => new BadOpReplica());
  const badRb = replayOps(OP_ORDERS[1], () => new BadOpReplica());
  const goodRa = replayOps(OP_ORDERS[0], () => new GoodOpReplica());
  const goodRb = replayOps(OP_ORDERS[1], () => new GoodOpReplica());
  return {
    badRaSet: badRa,
    badRbSet: badRb,
    badDiverged: badRa !== badRb,
    goodRaSet: goodRa,
    goodRbSet: goodRb,
    goodConverged: goodRa === goodRb,
  };
}

/** Demo D: LWW (chapter 6) vs OR-Set on the same concurrent add/remove — show LWW
 *  drops one of two concurrent writes while OR-Set keeps both. Returns the two
 *  final sets and which write LWW lost. */
function demoLwwVsCrdt(): {
  lwwFinal: string;
  crdtFinal: string;
  lwwLost: string;
} {
  // Scenario: at the "same" logical time, r1 removes "doc" and r2 adds "note".
  // Then a third concurrent pair: r1 adds "doc-v2" and r2 removes "doc" — the
  // classic concurrent-write-to-same-key conflict on "doc".
  // We give LWW timestamps that TIE on "doc" so the replica-id tiebreak silently
  // drops one write; the OR-Set keeps both adds (add-wins).
  const lwwA = new LwwSet();
  const lwwB = new LwwSet();
  // r1 and r2 both write "doc" at the SAME timestamp (concurrent, no causal order).
  lwwA.add("r1", "doc", 100); // r1: doc present @100
  lwwB.remove("r2", "doc", 100); // r2: doc removed @100 (concurrent)
  // independent non-conflicting writes survive on both
  lwwA.add("r1", "report", 100);
  lwwB.add("r2", "memo", 100);
  lwwA.merge(lwwB);
  lwwB.merge(lwwA);

  const crdtA = new OrSet();
  const crdtB = new OrSet();
  crdtA.add("r1", "doc"); // r1 adds doc
  crdtB.add("r2", "doc"); // r2 concurrently adds doc (different dot)
  crdtA.add("r1", "report");
  crdtB.add("r2", "memo");
  // r2 then removes the doc dot it observed (its OWN add) — but r1's add-dot lives
  crdtB.remove("doc");
  crdtA.merge(crdtB);
  crdtB.merge(crdtA);

  // LWW resolves "doc" by (ts=100, replica) tiebreak: "r2">"r1" so remove wins →
  // doc is DROPPED. r1's add @100 is the lost write.
  const lwwHasDoc = lwwA.has("doc");
  const crdtHasDoc = crdtA.has("doc");
  return {
    lwwFinal: `{${lwwA.value().join(",")}}`,
    crdtFinal: `{${crdtA.value().join(",")}}`,
    lwwLost: lwwHasDoc
      ? "(none — tiebreak happened to keep the add)"
      : `LWW 用 (ts=100, replica) tiebreak 丢掉了 r1 对 "doc" 的 add（保留 r2 的 remove）；OR-Set 保留 doc=${crdtHasDoc}`,
  };
}

/** Demo E: the network-level convergence run. 4 replicas gossip an OR-Set under
 *  drop+reorder+duplicate; cut {r0,r1} | {r2,r3} during [200,600]ms; apply
 *  divergent ops during the partition; heal at 600ms; measure how many gossip
 *  rounds after heal until all four fingerprints match. */
function runGossipScenario(): void {
  runScenario("crdt-gossip-partition-heal", SEED, {
    untilMs: 2000,
    network: { defaultBaselineMs: 8, defaultJitterMs: 6 },
    setup(ctx) {
      ctx.net.dropRate(0.3); // heavily lossy gossip — fine, merge is idempotent
      ctx.net.reorder(true); // gossip order must not matter
      ctx.net.duplicate(true); // duplicate snapshots must be no-ops

      const replicas = REPLICA_IDS.map((id) => new OrSetReplica(id, ctx.clock, ctx.net, REPLICA_IDS));
      const byId = new Map(replicas.map((r) => [r.id, r]));

      // Divergent local operations, scheduled to fall partly during the partition.
      ctx.at(50, () => byId.get("r0")!.applyAdd("alpha"));
      ctx.at(60, () => byId.get("r1")!.applyAdd("beta"));
      // During partition window [200,600]: each side writes independently.
      ctx.at(250, () => byId.get("r0")!.applyAdd("gamma"));
      ctx.at(300, () => byId.get("r2")!.applyAdd("gamma")); // SAME element, other side, different dot → add-wins, no conflict
      ctx.at(350, () => byId.get("r1")!.applyRemove("alpha")); // r1 removes alpha (observed)
      ctx.at(400, () => byId.get("r3")!.applyAdd("delta"));
      ctx.at(450, () => byId.get("r2")!.applyAdd("alpha")); // r2 concurrently re-adds alpha → must SURVIVE r1's remove
      ctx.at(580, () => byId.get("r3")!.applyAdd("epsilon")); // written just before heal → must propagate across the (just-healed) cut

      // The partition window [200,600]. Capture each side's DIVERGED state at the
      // instant of heal — this is the "replicas disagree" snapshot the chapter
      // promises (the two sides genuinely hold different sets here).
      const HEAL_MS = 600;
      const divergedAtHeal: Record<string, string> = {};
      ctx.at(200, () => ctx.net.partition(["r0", "r1"], ["r2", "r3"]));
      ctx.at(HEAL_MS, () => {
        for (const r of replicas) divergedAtHeal[r.id] = `{${r.state.value().join(",")}}`;
        ctx.net.heal();
      });

      // Convergence-round measurement. After heal, sample once per gossip interval;
      // record the FIRST sample where all four fingerprints match. Rounds =
      // ceil((convergeTime - heal) / interval) — i.e. how many anti-entropy rounds
      // the cluster needed to reconcile. This is a real measured count, not an
      // assumption; if it never converges within untilMs it stays -1 and the
      // printed "converged=NO" makes the failure visible.
      let convergeTimeMs = -1;
      const probe = (): void => {
        if (convergeTimeMs >= 0) return;
        const fps = replicas.map((r) => r.state.fingerprint());
        if (fps.every((f) => f === fps[0])) {
          convergeTimeMs = ctx.clock.now();
          return;
        }
        ctx.clock.schedule(GOSSIP_INTERVAL_MS, probe);
      };
      ctx.at(HEAL_MS, () => ctx.clock.schedule(GOSSIP_INTERVAL_MS, probe));

      // Start anti-entropy gossip on all replicas.
      for (const r of replicas) r.startGossip();

      // Safety invariant: NO replica may ever observe an element that nobody added.
      // (Monotonic read safety — gossip must never fabricate membership.) We don't
      // assert convergence as an invariant (it is only guaranteed eventually, not
      // at every tick — asserting it per-tick would be FALSE during the partition,
      // which is the whole point).
      const everAdded = new Set<string>(["alpha", "beta", "gamma", "delta", "epsilon"]);
      ctx.watch(
        invariant(
          "no-fabricated-elements",
          () => replicas.every((r) => r.state.value().every((el) => everAdded.has(el))),
          () => replicas.map((r) => ({ id: r.id, value: r.state.value() })),
        ),
      );

      return () => {
        // Measure convergence: collect fingerprints after the run quiesces.
        const fps = replicas.map((r) => r.state.fingerprint());
        const allEqual = fps.every((f) => f === fps[0]);
        const roundsToConverge =
          convergeTimeMs >= 0 ? Math.ceil((convergeTimeMs - HEAL_MS) / GOSSIP_INTERVAL_MS) : -1;
        return [
          // Snapshot at heal: the two partition sides genuinely disagree here.
          ...replicas.map((r) => ({
            metric: `at-heal(t=${HEAL_MS}) ${r.id}`,
            value: divergedAtHeal[r.id] ?? "(n/a)",
          })),
          ...replicas.map((r) => ({
            metric: `final ${r.id}`,
            value: `{${r.state.value().join(",")}}`,
          })),
          { metric: "all replicas converged", value: allEqual ? "yes" : "NO" },
          {
            metric: "convergence after heal (merge rounds)",
            value: roundsToConverge >= 0 ? roundsToConverge : "did-not-converge",
          },
          {
            metric: "convergence time after heal (sim ms)",
            value: convergeTimeMs >= 0 ? convergeTimeMs - HEAL_MS : -1,
          },
          {
            metric: "alpha survived (add-wins vs remove)",
            value: replicas[0].state.has("alpha") ? "yes" : "no",
          },
        ];
      };
    },
  });
}

function main(): void {
  const rng = seededRng(SEED);

  console.log("\n############ stage07 — CRDT 最终一致与收敛 ############");
  console.log(`# seed=${SEED}（固定，可复现）；时延为 SimClock 虚拟毫秒，相对趋势可迁移，绝对值非 wall-clock`);

  // -- Demo A: G-Counter -----------------------------------------------------
  console.log("\n## A. G-Counter：并发自增，任意顺序合并均收敛到同一和（且不丢不重）");
  const gc = demoGCounter(rng);
  console.log(
    `  trials=${gc.trials}  distinctOutcomes=${gc.distinctOutcomes}  converged=${gc.converged}`,
  );
  console.log(
    `  收敛后 sum=${gc.sum}  应有总增量=${gc.expected}  ${gc.sum === gc.expected ? "✓ 无丢失/重复计数" : "✗ 计数错误"}`,
  );

  // -- Demo B: OR-Set --------------------------------------------------------
  console.log("\n## B. OR-Set（add-wins，带 dot/tombstone）：并发增删，任意顺序+重复合并均收敛");
  const os = demoOrSet(rng);
  console.log(
    `  trials=${os.trials}  distinctOutcomes=${os.distinctOutcomes}  converged=${os.converged}`,
  );
  console.log(`  收敛终态 value=${os.value}  （x 被 r1 删、被 r2 并发重加 → add-wins，x 保留）`);

  // -- Demo C: failure mode --------------------------------------------------
  console.log("\n## C. 失败模式：伪 CRDT（remove 只删元素、不记 tombstone）op 不可交换 → 发散");
  console.log("    op-based 复制：共同祖先 x（dot=ox）；ra 收到顺序[删ox, 加rx]，rb 收到顺序[加rx, 删ox]");
  const bad = demoBadOrSet();
  console.log(`  BadOrSet   ra 终态=${bad.badRaSet}  rb 终态=${bad.badRbSet}  diverged=${bad.badDiverged}`);
  console.log(`    └─ 「A 删 / B 加」乱序下 ra 有 x、rb 没 x → 永久发散（remove 把 x 整个删掉，与 add 不可交换）`);
  console.log(`  正确 OR-Set ra 终态=${bad.goodRaSet}  rb 终态=${bad.goodRbSet}  converged=${bad.goodConverged}`);
  console.log(`    └─ remove 只 tombstone 它观察到的 dot(ox)；rx 未被删 → 两副本均收敛到 {x}（add-wins）`);

  // -- Demo D: LWW vs CRDT ---------------------------------------------------
  console.log("\n## D. 对比第 6 章 LWW：并发写「doc」时 LWW 丢一个写，CRDT 不丢");
  const cmp = demoLwwVsCrdt();
  console.log(`  LWW-Set  终态=${cmp.lwwFinal}`);
  console.log(`  OR-Set   终态=${cmp.crdtFinal}`);
  console.log(`  LWW 丢失的写: ${cmp.lwwLost}`);

  // -- Demo E: network gossip ------------------------------------------------
  console.log("\n## E. 网络层实测：4 副本 gossip，注入丢包30%+乱序+重复，分区[200,600]ms 后 heal");
  runGossipScenario();

  console.log("\n############ stage07 完成 ############\n");
}

main();
