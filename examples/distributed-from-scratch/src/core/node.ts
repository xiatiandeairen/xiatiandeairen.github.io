// core/node.ts — base class for a protocol state machine (one cluster member).
//
// Why a base class at all: every chapter's participant (a Lamport process, a
// Raft server, a CRDT replica) does the same three things — receive a message,
// set/clear timers, send messages — and differs only in the state machine on
// top. Factoring that lets each stage subclass and write ONLY the protocol
// logic, so the reader sees the algorithm, not the plumbing.
//
// Crucial design choice: timers go through the SimClock, NOT setTimeout. An
// election timeout, a heartbeat interval, a retry backoff — all are virtual time.
// This is what makes "the leader's heartbeat is late so a follower times out and
// starts an election" a reproducible event rather than a flaky one.
//
// Persistence model (used by Raft/CAP chapters): two state regions. `volatile`
// is plain instance fields a subclass declares — wiped on crash(). `persistent`
// is the Map below — survives crash(), models fsync'd state (currentTerm,
// votedFor, log). The whole correctness argument of crash-recovery rests on
// what's persistent vs volatile, so the base class makes the boundary explicit
// instead of leaving it to each subclass to get subtly wrong.
//
// Invariant: a crashed node delivers NO messages and fires NO timers until
// restart(). The failure mode we guard against is a "crashed" node that keeps
// participating — that would make crash tests pass for the wrong reason.

import type { SimClock } from "./clock.js";
import type { Network, Message } from "./network.js";

export abstract class Node {
  /** Persistent state: survives crash() (models fsync'd disk). Subclasses store
   *  currentTerm / votedFor / log entries here. Keep it to JSON-able values so a
   *  snapshot prints cleanly in an invariant violation dump. */
  protected readonly persistent = new Map<string, unknown>();

  /** True between crash() and restart(). Gates message delivery and timers — a
   *  down node is INERT, not slow. */
  private down = false;

  /** Active timer tokens → true, so clearTimer can cancel by ignoring stale
   *  fires. We can't remove an event from SimClock's heap, so timers are
   *  cancelled by token invalidation: the scheduled closure checks liveness. */
  private liveTimers = new Set<number>();
  private timerSeq = 0;

  constructor(
    readonly id: string,
    protected readonly clock: SimClock,
    protected readonly net: Network,
  ) {
    // Self-register the inbox so the Network can route to us. Subclass ctors run
    // after this, which is fine: messages can't arrive until the clock advances.
    this.net.register(this.id, (msg) => this.receive(msg));
  }

  /** Protocol entry point — subclass implements its state transition here.
   *  Called by the Network on delivery. `from` is msg.from (echoed for ergonomics
   *  since handlers branch on sender). MUST be pure w.r.t. wall-clock/Math.random:
   *  all time via this.clock, all randomness via the stage's Rng. */
  abstract onMessage(from: string, msg: Message): void;

  /** Internal delivery gate: a down node silently absorbs messages (models a
   *  crashed server — the packet arrived but nothing is listening). NOT counted
   *  as a drop here; the Network already counted delivery. The "lost" effect is
   *  that the handler never runs. */
  private receive(msg: Message): void {
    if (this.down) return;
    this.onMessage(msg.from, msg);
  }

  /** Send to a peer. Thin pass-through to Network so subclasses never touch the
   *  bus directly — keeps the from-id honest (can't spoof another node's id). */
  protected sendTo(to: string, kind: string, payload: unknown): void {
    if (this.down) return; // a crashed node sends nothing
    this.net.send(this.id, to, kind, payload);
  }

  /** Arm a one-shot timer for `ms` virtual milliseconds. Returns a token for
   *  clearTimer. Re-arming (typical for heartbeat/election) means: clear old,
   *  set new. The token/liveTimers dance is how we "cancel" without heap removal:
   *  a cleared timer's fire is a no-op. The failure mode this prevents is a STALE
   *  election timeout firing after a heartbeat already reset it — the #1 cause of
   *  spurious Raft elections if timers aren't cancellable. */
  protected setTimer(ms: number, fn: () => void): number {
    const token = ++this.timerSeq;
    this.liveTimers.add(token);
    this.clock.schedule(ms, () => {
      // Drop the fire if the node crashed or the timer was cleared/superseded.
      if (this.down || !this.liveTimers.has(token)) return;
      this.liveTimers.delete(token);
      fn();
    });
    return token;
  }

  /** Cancel a timer by token. Safe to call on an already-fired/unknown token. */
  protected clearTimer(token: number): void {
    this.liveTimers.delete(token);
  }

  // --- crash / recovery model -------------------------------------------------

  /** Crash this node: it stops delivering messages and firing timers, and all
   *  VOLATILE state (subclass fields) becomes meaningless. We don't (and can't)
   *  reach into subclass fields here — the contract is: on restart() the subclass
   *  REBUILDS volatile state from `persistent`. crash() clears live timers so no
   *  pre-crash timer fires post-restart. */
  crash(): void {
    this.down = true;
    this.liveTimers.clear();
    this.onCrash();
  }

  /** Restart a crashed node. Subclass overrides onRestart() to rebuild volatile
   *  state from persistent (e.g. reset commitIndex to 0, re-derive from log).
   *  Persistent state is intact — that's the whole point of the volatile/
   *  persistent split. */
  restart(): void {
    if (!this.down) return;
    this.down = false;
    this.onRestart();
  }

  /** True if currently crashed — for invariants like "leader count excludes
   *  down nodes". */
  isDown(): boolean {
    return this.down;
  }

  /** Hooks subclasses may override. Defaults are no-ops so a stage that doesn't
   *  model crash recovery (Lamport/vector-clock chapters) ignores them. */
  protected onCrash(): void {}
  protected onRestart(): void {}
}
