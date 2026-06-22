// stage05-raft-log-replication.ts — Raft log replication: make N replicas agree
// on ONE ordered sequence of client commands, then apply them to identical state
// machines, and survive a leader crash + a lagging follower + a partition.
//
// Why this stage exists (and what chapter 4 left undone): chapter 4 elected a
// leader. A leader alone agrees on nothing — the hard part of consensus is
// REPLICATION: getting a command into a majority of logs at the SAME index with
// the SAME term, so that even after the leader crashes the survivors can never
// disagree about what was committed. That property — State Machine Safety — is
// the whole reason a bank can run on Raft. This file builds it and then breaks
// it on purpose.
//
// The teaching spine is one toggle: `unsafeAppend`. With it OFF we implement the
// real AppendEntries consistency check (the receiver rejects an append unless its
// log matches the leader at prevLogIndex/prevLogTerm). With it ON we SKIP that
// check — blindly overwriting whatever the leader sends. The second run shows the
// exact disaster the check prevents: a follower that fell behind during a
// partition silently DIVERGES, two replicas apply DIFFERENT commands at the same
// committed index, and the State Machine Safety invariant fires with a snapshot.
// Re-enable the check and all five replicas converge again. Same seed both runs.
//
// What is real here vs. simplified:
//   - Real: the AppendEntries prevLog consistency check, log truncation on
//     conflict, commitIndex advancing only on majority matchIndex, the
//     persistent/volatile split (log + currentTerm survive crash; commitIndex
//     and the applied state machine are rebuilt on restart), leader-only-appends.
//   - Simplified: leadership is ASSIGNED, not elected (chapter 4's job). We pin
//     one leader, crash it mid-run, and promote a pre-designated survivor — the
//     replication mechanics are identical to a real election outcome, but we skip
//     re-running RequestVote so the chapter stays about LOGS, not votes. The
//     "current term" still increments on leader change so log-matching is exercised
//     across terms.
//
// Determinism: every drop/latency/partition outcome flows through the scenario's
// seeded Rng + SimClock. Re-running prints byte-identical output.

import { runScenario } from "./core/scenario.js";
import { Node } from "./core/node.js";
import type { Message } from "./core/network.js";
import { invariant } from "./core/assert.js";
import { Histogram } from "./core/metrics.js";

// --- log model --------------------------------------------------------------

/** One replicated log entry. `term` is the leader's term when it was created —
 *  this is what makes Log Matching decidable: two entries are "the same" iff they
 *  share BOTH index and term, because a given (index, term) is produced by a
 *  single leader and never reused. `command` is the opaque client request the
 *  state machine will apply (here: increment a named counter). */
interface LogEntry {
  term: number;
  /** Monotonic per-leader request id, so we can SEE which client op landed where
   *  when logs diverge. Not part of Raft proper; a debugging affordance. */
  command: { client: string; seq: number };
}

/** AppendEntries RPC payload (heartbeat = empty entries). The prevLog* pair is
 *  the load-bearing part of the whole algorithm: it lets the follower verify its
 *  log is identical to the leader's UP TO the point the new entries attach,
 *  without shipping the entire log. Drop this check and replication becomes
 *  "trust the leader blindly" — which is exactly the failure mode we demo. */
interface AppendEntriesReq {
  term: number;
  leaderId: string;
  prevLogIndex: number; // index of entry immediately before `entries` (0 = none)
  prevLogTerm: number; // term of that entry (0 when prevLogIndex==0)
  entries: LogEntry[];
  leaderCommit: number; // leader's commitIndex, so followers can advance theirs
}

interface AppendEntriesResp {
  term: number;
  from: string;
  success: boolean;
  /** On success: the follower's last log index after appending, so the leader can
   *  set matchIndex. On failure (log mismatch): the follower's current last index,
   *  a hint to back nextIndex up faster than one-at-a-time. */
  lastLogIndex: number;
}

type RaftMessage =
  | (Message & { kind: "AppendEntries"; payload: AppendEntriesReq })
  | (Message & { kind: "AppendEntriesResp"; payload: AppendEntriesResp });

// --- the replica ------------------------------------------------------------

interface RaftConfig {
  peers: string[]; // all node ids including self
  /** THE toggle. false = correct Raft (prevLog consistency enforced).
   *  true  = skip the check on the follower → blind overwrite → divergence. */
  unsafeAppend: boolean;
  /** Histogram to record commit latency (ms from a command first appended at the
   *  leader to the moment it becomes committed). Injected so the scenario owns it. */
  commitLatency: Histogram;
  clock: { now(): number };
}

/** A Raft replica. Persistent state (currentTerm, log) lives in the base class's
 *  `persistent` Map so it survives crash(); volatile state (commitIndex, the
 *  applied counters, and leader bookkeeping) are plain fields, wiped on crash and
 *  rebuilt in onRestart(). That split is the entire crash-safety argument: a node
 *  may forget what it had committed (volatile) but never what it logged (durable),
 *  so on restart it can safely re-derive committed state by re-applying its log. */
class RaftReplica extends Node {
  private cfg: RaftConfig;

  // --- volatile (lost on crash, rebuilt in onRestart) ---
  private role: "follower" | "leader" = "follower";
  private commitIndex = 0; // highest log index known committed (1-based; 0 = none)
  private lastApplied = 0; // highest index applied to the state machine
  private state = new Map<string, number>(); // the replicated state machine
  // leader-only: per-peer replication progress.
  private nextIndex = new Map<string, number>(); // next entry to send each peer
  private matchIndex = new Map<string, number>(); // highest entry known replicated
  // track when each leader-appended index was created, to measure commit latency.
  private appendedAtMs = new Map<number, number>();

  constructor(
    id: string,
    clock: ConstructorParameters<typeof Node>[1],
    net: ConstructorParameters<typeof Node>[2],
    cfg: RaftConfig,
  ) {
    super(id, clock, net);
    this.cfg = cfg;
    // Initialize persistent state. In real Raft these load from disk; here the
    // Map IS the disk. currentTerm starts at 0, log empty.
    this.persistent.set("currentTerm", 0);
    this.persistent.set("log", [] as LogEntry[]);
  }

  // --- persistent accessors (the durable disk) ---
  private get currentTerm(): number {
    return this.persistent.get("currentTerm") as number;
  }
  private set currentTerm(t: number) {
    this.persistent.set("currentTerm", t);
  }
  private get log(): LogEntry[] {
    return this.persistent.get("log") as LogEntry[];
  }

  /** Term of the entry at 1-based `index`, or 0 for index 0 / out of range. The 0
   *  sentinel is deliberate: prevLogIndex==0 means "attaching at the very start",
   *  which must always match, so prevLogTerm 0 == log term 0 trivially holds. */
  private termAt(index: number): number {
    if (index <= 0 || index > this.log.length) return 0;
    return this.log[index - 1].term;
  }

  private lastLogIndex(): number {
    return this.log.length;
  }

  // --- crash / restart: the durability contract ---

  /** On restart we are a follower again with EMPTY volatile state. We do NOT
   *  trust the old commitIndex (it was volatile and is gone). Instead lastApplied
   *  resets to 0 and we re-apply the whole persistent log up to whatever we later
   *  learn is committed. This models the real recovery path and is why losing
   *  volatile state is survivable: the durable log is the source of truth. */
  protected override onRestart(): void {
    this.role = "follower";
    this.commitIndex = 0;
    this.lastApplied = 0;
    this.state = new Map();
    this.nextIndex = new Map();
    this.matchIndex = new Map();
    this.appendedAtMs = new Map();
  }

  // --- becoming leader (assigned, not elected — see file header) ---

  /** Promote this node to leader for `term`. Real Raft reaches this state via
   *  RequestVote (chapter 4); we shortcut it because this chapter is about what a
   *  leader does AFTER winning. A new leader initializes nextIndex optimistically
   *  to its own lastLogIndex+1 (assume followers match) and matchIndex to 0
   *  (assume nothing confirmed) — the AppendEntries failures then walk nextIndex
   *  back to the true divergence point. */
  promoteToLeader(term: number): void {
    this.currentTerm = term;
    this.role = "leader";
    for (const p of this.cfg.peers) {
      if (p === this.id) continue;
      this.nextIndex.set(p, this.lastLogIndex() + 1);
      this.matchIndex.set(p, 0);
    }
    // Immediately replicate so followers learn the new leader/term.
    this.broadcastAppendEntries();
    // Heartbeat loop keeps followers in sync and pushes commitIndex forward.
    this.scheduleHeartbeat();
  }

  private scheduleHeartbeat(): void {
    // 50ms heartbeat — well under any election timeout, comfortably above link
    // latency (~10ms) so a round-trip fits inside an interval.
    this.setTimer(50, () => {
      if (this.role !== "leader") return;
      this.broadcastAppendEntries();
      this.scheduleHeartbeat();
    });
  }

  // --- client command intake (leader only) ---

  /** A client submits a command. Only the leader may append; a follower silently
   *  drops it (a real client would be redirected). The entry is stamped with the
   *  current term and the time, then replication is kicked immediately rather than
   *  waiting for the next heartbeat, to keep commit latency honest. */
  submitCommand(client: string, seq: number): void {
    if (this.role !== "leader") return;
    const entry: LogEntry = { term: this.currentTerm, command: { client, seq } };
    this.log.push(entry);
    this.appendedAtMs.set(this.lastLogIndex(), this.cfg.clock.now());
    this.broadcastAppendEntries();
  }

  private broadcastAppendEntries(): void {
    for (const p of this.cfg.peers) {
      if (p === this.id) continue;
      this.sendAppendEntriesTo(p);
    }
  }

  /** Send the entries peer `p` is missing, framed with the prevLog* pair the
   *  receiver needs to verify continuity. nextIndex[p] is our belief about where
   *  the peer's log ends + 1; we send everything from there on. */
  private sendAppendEntriesTo(p: string): void {
    const ni = this.nextIndex.get(p) ?? this.lastLogIndex() + 1;
    const prevLogIndex = ni - 1;
    const entries = this.log.slice(prevLogIndex); // entries at index ni..end
    const req: AppendEntriesReq = {
      term: this.currentTerm,
      leaderId: this.id,
      prevLogIndex,
      prevLogTerm: this.termAt(prevLogIndex),
      entries,
      leaderCommit: this.commitIndex,
    };
    this.sendTo(p, "AppendEntries", req);
  }

  override onMessage(_from: string, msg: Message): void {
    const m = msg as RaftMessage;
    switch (m.kind) {
      case "AppendEntries":
        return this.onAppendEntries(m.payload);
      case "AppendEntriesResp":
        return this.onAppendEntriesResp(m.payload);
      default:
        // Unknown verb: a topology/wiring bug, not a network event. Surface it.
        throw new Error(`RaftReplica ${this.id}: unknown message kind ${msg.kind}`);
    }
  }

  // --- follower side: the heart of correctness ---

  /** Handle an AppendEntries. The correctness of the whole protocol lives in the
   *  prevLog consistency check below. The `unsafeAppend` toggle removes it to
   *  demonstrate the divergence it prevents. */
  private onAppendEntries(req: AppendEntriesReq): void {
    // Stale leader (lower term)? Reject so it steps down. (Higher term => adopt.)
    if (req.term < this.currentTerm) {
      return this.replyAppend(req.leaderId, false);
    }
    if (req.term > this.currentTerm) this.currentTerm = req.term;
    this.role = "follower"; // any valid AppendEntries means a leader exists

    // THE CHECK. In correct Raft a follower refuses to append unless its own log
    // matches the leader at prevLogIndex (same term there). This is what stops a
    // follower that missed entries from grafting new ones onto a stale prefix and
    // silently diverging. `unsafeAppend` skips it — see the second scenario.
    if (!this.cfg.unsafeAppend) {
      const consistent =
        req.prevLogIndex === 0 || this.termAt(req.prevLogIndex) === req.prevLogTerm;
      if (!consistent) {
        // Mismatch: tell the leader our last index so it can back nextIndex up.
        return this.replyAppend(req.leaderId, false);
      }
    }

    // Append/overwrite. We splice in the leader's entries starting right after
    // prevLogIndex. If our log had conflicting entries there (different term at
    // the same index), Raft TRUNCATES them — the leader's log wins. In unsafe
    // mode we still splice, but without the guard above we may be splicing onto a
    // log that doesn't actually match, which is the bug.
    let idx = req.prevLogIndex;
    for (const e of req.entries) {
      idx++;
      const existing = this.termAt(idx);
      if (existing !== 0 && existing !== e.term) {
        // Conflict: truncate everything from idx onward, then take leader's entry.
        this.log.length = idx - 1;
      }
      if (idx > this.log.length) {
        this.log.push(e);
      }
      // else: identical entry already present (idempotent re-send) — leave it.
    }

    // Advance commitIndex to min(leaderCommit, our last index) and apply. A
    // follower can only commit what it actually holds, hence the min.
    if (req.leaderCommit > this.commitIndex) {
      this.commitIndex = Math.min(req.leaderCommit, this.lastLogIndex());
      this.applyCommitted();
    }

    this.replyAppend(req.leaderId, true);
  }

  private replyAppend(leaderId: string, success: boolean): void {
    const resp: AppendEntriesResp = {
      term: this.currentTerm,
      from: this.id,
      success,
      lastLogIndex: this.lastLogIndex(),
    };
    this.sendTo(leaderId, "AppendEntriesResp", resp);
  }

  // --- leader side: track replication, advance commit ---

  private onAppendEntriesResp(resp: AppendEntriesResp): void {
    if (this.role !== "leader") return;
    // A higher term means we've been deposed; step down. (No re-election here.)
    if (resp.term > this.currentTerm) {
      this.currentTerm = resp.term;
      this.role = "follower";
      return;
    }

    if (resp.success) {
      // Peer now matches up to resp.lastLogIndex. Record it and try to commit.
      this.matchIndex.set(resp.from, resp.lastLogIndex);
      this.nextIndex.set(resp.from, resp.lastLogIndex + 1);
      this.advanceCommitIndex();
    } else {
      // Log mismatch: back nextIndex up toward the peer's hint and retry. This is
      // the loop that eventually finds the divergence point and overwrites the
      // follower's stale suffix.
      const hint = resp.lastLogIndex + 1;
      const cur = this.nextIndex.get(resp.from) ?? this.lastLogIndex() + 1;
      this.nextIndex.set(resp.from, Math.max(1, Math.min(cur - 1, hint)));
      this.sendAppendEntriesTo(resp.from);
    }
  }

  /** Leader commit rule: an index N is committable once it is stored on a MAJORITY
   *  (including the leader) AND log[N].term == currentTerm. The term condition is
   *  the subtle one — a leader may NOT commit an entry from a previous term just
   *  because it's on a majority now (the classic Figure-8 hazard); it commits old
   *  entries only indirectly by committing a current-term entry above them. */
  private advanceCommitIndex(): void {
    const majority = Math.floor(this.cfg.peers.length / 2) + 1;
    // Try the highest indices first; stop at the first that's committable.
    for (let n = this.lastLogIndex(); n > this.commitIndex; n--) {
      if (this.termAt(n) !== this.currentTerm) continue; // term safety rule
      // Count replicas (self + peers whose matchIndex >= n).
      let replicas = 1;
      for (const p of this.cfg.peers) {
        if (p === this.id) continue;
        if ((this.matchIndex.get(p) ?? 0) >= n) replicas++;
      }
      if (replicas >= majority) {
        this.commitIndex = n;
        this.applyCommitted();
        break;
      }
    }
  }

  // --- state machine application ---

  /** Apply every newly-committed entry to the state machine exactly once, in log
   *  order. Idempotent w.r.t. re-delivery because we only apply indices in
   *  (lastApplied, commitIndex]. The state machine here is a per-client counter;
   *  applying command {client,seq} just bumps state[client]. Records commit
   *  latency for entries this node originally appended as leader. */
  private applyCommitted(): void {
    while (this.lastApplied < this.commitIndex) {
      this.lastApplied++;
      const entry = this.log[this.lastApplied - 1];
      this.state.set(entry.command.client, (this.state.get(entry.command.client) ?? 0) + 1);
      const appendedAt = this.appendedAtMs.get(this.lastApplied);
      if (appendedAt !== undefined) {
        this.cfg.commitLatency.record(this.cfg.clock.now() - appendedAt);
        this.appendedAtMs.delete(this.lastApplied);
      }
    }
  }

  // --- inspection helpers for invariants / metrics (read-only) ---

  getCommitIndex(): number {
    return this.commitIndex;
  }
  getTerm(): number {
    return this.currentTerm;
  }
  getRole(): string {
    return this.role;
  }
  /** Committed prefix as (index,term,command) triples — what State Machine Safety
   *  compares across replicas. Only the committed prefix is comparable; the
   *  uncommitted tail may legitimately differ between nodes mid-flight. */
  committedEntries(): { index: number; term: number; cmd: string }[] {
    const out: { index: number; term: number; cmd: string }[] = [];
    for (let i = 1; i <= this.commitIndex; i++) {
      const e = this.log[i - 1];
      out.push({ index: i, term: e.term, cmd: `${e.command.client}#${e.command.seq}` });
    }
    return out;
  }
  logLength(): number {
    return this.log.length;
  }
  /** The applied state machine — for the final "all replicas converged" check. */
  stateMachine(): Record<string, number> {
    return Object.fromEntries([...this.state.entries()].sort());
  }
}

// --- scenario construction --------------------------------------------------

const NODE_IDS = ["n0", "n1", "n2", "n3", "n4"]; // 5 nodes => majority 3
const SEED = 5;

/** Build a replication scenario. `unsafeAppend` selects correct Raft vs. the
 *  broken variant. The fault script is IDENTICAL across both so the only variable
 *  is the consistency check — that is what makes the two runs a controlled
 *  experiment rather than two anecdotes.
 *
 *  Fault timeline (virtual ms):
 *    0     n0 leader (term 1), starts replicating client commands
 *    0..   clients submit a steady stream of commands to the leader
 *    300   PARTITION: {n3,n4} cut off from {n0,n1,n2}. n3/n4 fall behind.
 *    600   leader n0 CRASHES. n1 promoted leader (term 2) on the majority side.
 *    900   HEAL the partition. n3/n4 rejoin with stale logs and must reconcile.
 *    1200  n0 RESTARTS as a follower; rebuilds volatile state from its log.
 */
function makeSpec(unsafeAppend: boolean) {
  const commitLatency = new Histogram();
  // Captured so invariants/result can read replica state after setup.
  let replicas: RaftReplica[] = [];

  return {
    untilMs: 4000,
    // Slightly lossy + jittery network so the consistency check has to actually
    // work to converge — not a clean pipe that hides bugs.
    network: { defaultBaselineMs: 8, defaultJitterMs: 6 },
    setup(ctx: import("./core/scenario.js").ScenarioContext) {
      const cfg: RaftConfig = {
        peers: NODE_IDS,
        unsafeAppend,
        commitLatency,
        clock: ctx.clock,
      };
      replicas = NODE_IDS.map((id) => new RaftReplica(id, ctx.clock, ctx.net, cfg));
      const byId = new Map(replicas.map((r) => [r.id, r]));
      const leader0 = byId.get("n0")!;
      const leader1 = byId.get("n1")!;

      // Client command generator: drip commands onto whichever node is leader.
      // We always submit to the CURRENT leader (n0 then n1); a submit to a
      // follower is a no-op, modeling client redirection.
      let seq = 0;
      const clients = ["alice", "bob", "carol"];
      const submitOne = () => {
        const client = clients[seq % clients.length];
        // Submit to both possible leaders; only the actual leader appends.
        leader0.submitCommand(client, seq);
        leader1.submitCommand(client, seq);
        seq++;
      };
      // Schedule a burst of commands across the run so replication is continuous
      // through the faults (commands land before, during, and after the partition).
      for (let t = 0; t <= 2000; t += 40) ctx.at(t, submitOne);

      // Kick off: n0 is leader for term 1.
      ctx.at(0, () => leader0.promoteToLeader(1));

      // Fault script (see header).
      ctx.at(300, () => ctx.net.partition(["n0", "n1", "n2"], ["n3", "n4"]));
      ctx.at(600, () => {
        leader0.crash();
        // Promote n1 on the majority side. New term so log-matching spans terms.
        leader1.promoteToLeader(2);
      });
      ctx.at(900, () => ctx.net.heal());
      ctx.at(1200, () => leader0.restart());

      // --- safety invariants, checked after EVERY event ---

      // State Machine Safety: for any two LIVE replicas, wherever both have
      // committed an entry at the same index, that entry's term AND command must
      // be identical. A committed entry is final — two replicas applying
      // different commands at the same committed index is the unforgivable bug.
      ctx.watch(
        invariant(
          "state-machine-safety",
          () => {
            const live = replicas.filter((r) => !r.isDown());
            for (let a = 0; a < live.length; a++) {
              for (let b = a + 1; b < live.length; b++) {
                const ea = live[a].committedEntries();
                const eb = live[b].committedEntries();
                const common = Math.min(ea.length, eb.length);
                for (let i = 0; i < common; i++) {
                  if (ea[i].term !== eb[i].term || ea[i].cmd !== eb[i].cmd) return false;
                }
              }
            }
            return true;
          },
          () => ({
            note: "committed entry mismatch between two live replicas",
            logs: replicas
              .filter((r) => !r.isDown())
              .map((r) => ({ id: r.id, committed: r.committedEntries() })),
          }),
        ),
      );

      // Log Matching: if two live logs have an entry at the same index with the
      // same term, all PRECEDING entries are identical too. This is the structural
      // property the prevLog check is designed to maintain; it should hold for the
      // WHOLE log (committed or not), not just the committed prefix.
      ctx.watch(
        invariant(
          "log-matching",
          () => {
            const live = replicas.filter((r) => !r.isDown());
            for (let a = 0; a < live.length; a++) {
              for (let b = a + 1; b < live.length; b++) {
                const la = live[a].committedEntries();
                const lb = live[b].committedEntries();
                const common = Math.min(la.length, lb.length);
                for (let i = 0; i < common; i++) {
                  // Same index+term must imply same command (and thus same prefix).
                  if (la[i].term === lb[i].term && la[i].cmd !== lb[i].cmd) return false;
                }
              }
            }
            return true;
          },
          () => ({
            note: "log-matching violated: same (index,term) holds different commands",
            logs: replicas
              .filter((r) => !r.isDown())
              .map((r) => ({ id: r.id, committed: r.committedEntries() })),
          }),
        ),
      );

      // Result rows computed at quiescence.
      return () => {
        // Per-replica convergence snapshot.
        const rows: Record<string, string | number>[] = [];
        for (const r of replicas) {
          rows.push({
            node: r.id,
            role: r.getRole(),
            term: r.getTerm(),
            logLen: r.logLength(),
            commitIdx: r.getCommitIndex(),
            "state(alice/bob/carol)": fmtState(r.stateMachine()),
          });
        }

        // Did all live replicas converge to the same committed prefix + state?
        const live = replicas.filter((r) => !r.isDown());
        const allConverged = checkConvergence(live);

        const lat = commitLatency.summary();
        rows.push({ node: "—", role: "", term: 0, logLen: 0, commitIdx: 0, "state(alice/bob/carol)": "" });
        rows.push({
          node: "ALL-CONVERGED",
          role: allConverged ? "yes" : "NO (diverged)",
          term: 0,
          logLen: 0,
          commitIdx: 0,
          "state(alice/bob/carol)": "",
        });
        rows.push({
          node: "commit-latency-ms",
          role: `n=${lat.count}`,
          term: 0,
          logLen: 0,
          commitIdx: 0,
          "state(alice/bob/carol)": `p50=${fmt(lat.p50)} p99=${fmt(lat.p99)} max=${fmt(lat.max)}`,
        });
        return rows;
      };
    },
  };
}

function fmt(n: number): string {
  return Number.isFinite(n) ? String(n) : "—";
}

function fmtState(s: Record<string, number>): string {
  return `${s.alice ?? 0}/${s.bob ?? 0}/${s.carol ?? 0}`;
}

/** All live replicas converged iff every pair agrees on the shorter committed
 *  prefix AND their applied state machines are identical. */
function checkConvergence(live: RaftReplica[]): boolean {
  if (live.length <= 1) return true;
  const ref = JSON.stringify(live[0].stateMachine());
  for (const r of live) {
    if (JSON.stringify(r.stateMachine()) !== ref) return false;
  }
  return true;
}

// --- run both experiments ---------------------------------------------------

console.log(
  "\n############ stage05: Raft log replication ############\n" +
    "5 replicas, one assigned leader, replicating client commands to per-client\n" +
    "counter state machines. Same fault script (partition -> leader crash ->\n" +
    "promote -> heal -> restart) run twice; the ONLY difference is whether the\n" +
    "AppendEntries prevLog consistency check is enforced.\n" +
    "NOTE: numbers are from a seeded in-memory simulation (latency 8+jitter6 ms,\n" +
    "virtual time). Absolute commit-latency ms are toy-optimistic; what transfers\n" +
    "is the RELATIVE behavior — safe run converges, unsafe run diverges.",
);

// --- Experiment A: correct Raft. Expect: all replicas converge, invariants hold.
console.log("\n----- Experiment A: prevLog consistency check ENABLED (correct Raft) -----");
runScenario("raft-replication-safe", SEED, makeSpec(false));

// --- Experiment B: the failure mode. Skip the prevLog check on the follower.
// We expect the State Machine Safety invariant to FIRE during the run — the
// scenario runner throws on the first violation with a snapshot. We catch it so
// the demo can print the diagnosis and then re-affirm that the fix (Experiment A)
// restores safety. Catching here is NOT "handling" the bug — it's letting the
// teaching narrative continue after deliberately triggering it.
console.log(
  "\n----- Experiment B: prevLog consistency check DISABLED (the bug) -----\n" +
    "Followers blindly splice whatever the leader sends. A follower that fell\n" +
    "behind during the partition grafts new entries onto a stale prefix => its\n" +
    "committed log diverges from the majority. Expect a safety violation:",
);
try {
  runScenario("raft-replication-unsafe", SEED, makeSpec(true));
  // If we get here, the bug did NOT reproduce — that itself would be a finding.
  console.log(
    "\n[!] Unsafe run did NOT trip an invariant. With this seed the divergence\n" +
      "    window didn't materialize; try another seed. (Safety was not actually\n" +
      "    guaranteed — absence of a violation here is luck, not correctness.)",
  );
} catch (err) {
  console.log("\n[caught expected violation] " + (err as Error).message);
  console.log(
    "\nDiagnosis: with the prevLog check removed, a follower appended the new\n" +
      "leader's entries onto a log that did NOT match at prevLogIndex, so two\n" +
      "replicas ended up with DIFFERENT commands at the same committed index.\n" +
      "Re-enabling the check (Experiment A above) makes all replicas converge and\n" +
      "both safety invariants hold for the entire run. Same seed, opposite outcome\n" +
      "— the consistency check is load-bearing, not decorative.",
  );
}
