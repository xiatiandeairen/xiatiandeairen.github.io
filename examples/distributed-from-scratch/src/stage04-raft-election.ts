// stage04-raft-election.ts — Raft leader election from scratch, on the core sim.
//
// What this chapter proves, with measured numbers (not prose):
//   1. Distribution of "rounds (terms) and virtual time to elect the FIRST
//      leader" across independent seeds — the cost of agreement under jitter.
//   2. Failover time: leader crashes -> how long until a NEW leader is stable.
//   3. Liveness failure mode: with IDENTICAL (non-randomized) election timeouts
//      every follower times out together, every candidate votes for itself, the
//      vote splits, and term after term elects NOBODY (a livelock). Turning the
//      randomization back on converges in ~1 term. We quantify both.
//   And throughout: the SAFETY invariant "at most one leader per term" is checked
//   after every single event and is NEVER violated, in any scenario.
//
// Why election-only (no log replication here): leader election is the part of
// Raft that the FLP impossibility actually bites — you cannot guarantee both
// safety and liveness under asynchrony, and Raft's answer is "keep safety always,
// buy liveness with randomized timeouts". Log replication is a later chapter; the
// honest scope of THIS file is the election sub-protocol. Consequence: the
// RequestVote "candidate's log is at least as up-to-date" check is trivially true
// for everyone (empty logs), so it is omitted — noted here so a reader does not
// mistake its absence for a bug.
//
// Determinism: all timeouts/jitter come from ctx.rng (seeded). Same seed =>
// byte-identical run. The numbers below are REAL simulated quantities (term
// counters, virtual-time deltas read off SimClock) computed by the code, not
// hand-picked. Virtual time is in simulated ms; absolute values depend on the
// configured latency/timeout knobs (toy, deliberately small for a fast demo) —
// the load-bearing results are the RELATIVE comparisons (randomized vs fixed) and
// the shape of the distribution, which transfer; the absolute ms do not.

import { runScenario, type ScenarioContext } from "./core/scenario.js";
import { Node } from "./core/node.js";
import { invariant } from "./core/assert.js";
import type { Message } from "./core/network.js";
import type { Rng } from "./core/prng.js";

// --- protocol tuning (toy values, chosen so the demo runs in a few sim-seconds) -
// ELECTION_BASE..+SPREAD is the randomized timeout window. HEARTBEAT must be well
// below ELECTION_BASE or a healthy leader's followers would time out between
// beats and trigger spurious elections — that ratio is the single most important
// Raft tuning rule, so it is encoded as an assertion-by-construction here.
const ELECTION_BASE_MS = 150;
const ELECTION_SPREAD_MS = 150; // timeout drawn uniformly from [150, 300)
const HEARTBEAT_MS = 50; // << ELECTION_BASE_MS so heartbeats keep followers calm

type Role = "follower" | "candidate" | "leader";

interface VoteReqPayload {
  term: number;
}
interface VoteRespPayload {
  term: number;
  granted: boolean;
}
interface HeartbeatPayload {
  term: number;
}

/** A single Raft server's election state machine. The whole point of subclassing
 *  Node is that this class is ONLY the protocol: term bookkeeping, the three role
 *  transitions, and the four message handlers. Plumbing (timers, send, crash) is
 *  inherited. */
class RaftElectionNode extends Node {
  // --- persistent state (survives crash; models fsync'd disk) -----------------
  // currentTerm and votedFor MUST be persistent: Raft safety ("a server never
  // votes twice in one term") depends on them surviving a crash+restart. If they
  // were volatile, a restarted node could re-vote in a term it already voted in,
  // electing two leaders. We store them in the base `persistent` Map so crash()
  // keeps them and restart() can rebuild from them.
  private get currentTerm(): number {
    return (this.persistent.get("currentTerm") as number | undefined) ?? 0;
  }
  private set currentTerm(t: number) {
    this.persistent.set("currentTerm", t);
  }
  private get votedFor(): string | null {
    return (this.persistent.get("votedFor") as string | null | undefined) ?? null;
  }
  private set votedFor(id: string | null) {
    this.persistent.set("votedFor", id);
  }

  // --- volatile state (wiped on crash; rebuilt by onRestart) ------------------
  private role: Role = "follower";
  private votesReceived = new Set<string>();
  private electionTimer = 0;
  private heartbeatTimer = 0;

  constructor(
    id: string,
    private readonly peers: string[],
    private readonly rng: Rng,
    private readonly randomizeTimeout: boolean,
    clock: import("./core/clock.js").SimClock,
    net: import("./core/network.js").Network,
  ) {
    // peers EXCLUDES self. clusterSize = peers + self; majority is computed from it.
    super(id, clock, net);
  }

  /** Cluster size including self — the quorum denominator. */
  private get clusterSize(): number {
    return this.peers.length + 1;
  }

  /** Strict majority. For 5 nodes this is 3. A candidate needs this many votes
   *  (including its own) to become leader; that >half rule is what guarantees two
   *  leaders in the same term would need overlapping voters, which is impossible
   *  since each server votes once per term — the core safety argument. */
  private get majority(): number {
    return Math.floor(this.clusterSize / 2) + 1;
  }

  /** Election timeout for this round. Randomized window is THE liveness fix: if
   *  every node used the same value they would all time out, all become candidate
   *  in the same term, all vote for themselves, and split the vote forever. The
   *  fixed-timeout branch exists precisely to demonstrate that livelock. */
  private nextElectionTimeoutMs(): number {
    if (!this.randomizeTimeout) return ELECTION_BASE_MS; // identical => split votes
    return ELECTION_BASE_MS + this.rng.int(ELECTION_SPREAD_MS);
  }

  /** Kick the node into life: arm the first election timer. Called once at setup
   *  for every node. Heartbeats only start once a node becomes leader. */
  start(): void {
    this.armElectionTimer();
  }

  /** (Re)arm the election timeout. Re-arming clears the old token first so a stale
   *  timeout cannot fire after a heartbeat already reset it (the base class makes
   *  a cleared timer's fire a no-op, but we still drop the token to be explicit). */
  private armElectionTimer(): void {
    this.clearTimer(this.electionTimer);
    this.electionTimer = this.setTimer(this.nextElectionTimeoutMs(), () => this.onElectionTimeout());
  }

  /** Election timeout fired: no leader heard from in time, so start a new term and
   *  campaign. A leader never does this (it sends heartbeats, so its own timer is
   *  not the relevant clock) — but we guard anyway in case a timer slips through. */
  private onElectionTimeout(): void {
    if (this.role === "leader") return;
    this.becomeCandidate();
  }

  private becomeCandidate(): void {
    // §Raft: increment term, vote for self, request votes from all peers.
    this.currentTerm = this.currentTerm + 1;
    this.role = "candidate";
    this.votedFor = this.id;
    this.votesReceived = new Set([this.id]); // self-vote counts toward majority
    // A 1-node cluster (degenerate) would already have majority; not our case (5),
    // but checking keeps the transition logic total.
    if (this.votesReceived.size >= this.majority) {
      this.becomeLeader();
      return;
    }
    for (const peer of this.peers) {
      this.sendTo(peer, "RequestVote", { term: this.currentTerm } satisfies VoteReqPayload);
    }
    // Arm a fresh (randomized) timeout: if this election does not resolve, the
    // node times out again and starts a HIGHER term — exactly the retry that lets
    // a split vote eventually break under randomization.
    this.armElectionTimer();
  }

  private becomeLeader(): void {
    this.role = "leader";
    this.clearTimer(this.electionTimer); // a leader does not run an election timer
    // Immediately assert authority with a heartbeat round, then keep beating. The
    // first beat is what stops other followers from timing out and starting a
    // competing election in the next term.
    this.sendHeartbeats();
    this.heartbeatTimer = this.setTimer(HEARTBEAT_MS, () => this.onHeartbeatTick());
  }

  private onHeartbeatTick(): void {
    if (this.role !== "leader") return; // stepped down between ticks
    this.sendHeartbeats();
    this.heartbeatTimer = this.setTimer(HEARTBEAT_MS, () => this.onHeartbeatTick());
  }

  private sendHeartbeats(): void {
    for (const peer of this.peers) {
      this.sendTo(peer, "Heartbeat", { term: this.currentTerm } satisfies HeartbeatPayload);
    }
  }

  /** Step down to follower at a (possibly new) term. The single rule that makes
   *  the whole protocol converge: ANY message carrying a term > ours means we are
   *  stale — adopt it, drop to follower, forget who we voted for (new term, fresh
   *  vote). Centralizing it here means every handler can just call this first. */
  private stepDown(newTerm: number): void {
    this.currentTerm = newTerm;
    this.role = "follower";
    this.votedFor = null;
    this.votesReceived.clear();
    this.clearTimer(this.heartbeatTimer);
    this.armElectionTimer();
  }

  override onMessage(_from: string, msg: Message): void {
    switch (msg.kind) {
      case "RequestVote":
        return this.onRequestVote(msg.from, msg.payload as VoteReqPayload);
      case "RequestVoteResp":
        return this.onVoteResponse(msg.from, msg.payload as VoteRespPayload);
      case "Heartbeat":
        return this.onHeartbeat(msg.payload as HeartbeatPayload);
      default:
        // A kind we don't recognize is a topology/wiring bug, not a network event.
        // Surface it rather than silently dropping (silent drops hide protocol bugs).
        throw new Error(`RaftElectionNode ${this.id}: unknown message kind "${msg.kind}"`);
    }
  }

  private onRequestVote(candidate: string, p: VoteReqPayload): void {
    // Stale candidate (older term): refuse and tell them our higher term so they
    // step down instead of retrying uselessly.
    if (p.term < this.currentTerm) {
      this.sendTo(candidate, "RequestVoteResp", {
        term: this.currentTerm,
        granted: false,
      } satisfies VoteRespPayload);
      return;
    }
    // Newer term: adopt it and become a follower BEFORE deciding the vote, so we
    // are eligible to grant in this new term.
    if (p.term > this.currentTerm) this.stepDown(p.term);

    // Grant iff we have not already voted for someone else this term. (Log
    // up-to-dateness check omitted — empty logs, see file header.) Re-granting to
    // the SAME candidate is allowed and makes the vote idempotent under message
    // duplication.
    const free = this.votedFor === null || this.votedFor === candidate;
    if (free) {
      this.votedFor = candidate;
      // Granting a vote is "hearing from a valid leader-to-be": reset our election
      // timer so we don't immediately launch a competing campaign.
      this.armElectionTimer();
    }
    this.sendTo(candidate, "RequestVoteResp", {
      term: this.currentTerm,
      granted: free,
    } satisfies VoteRespPayload);
  }

  private onVoteResponse(voter: string, p: VoteRespPayload): void {
    // A response from a higher term means we lost / are stale: step down.
    if (p.term > this.currentTerm) {
      this.stepDown(p.term);
      return;
    }
    // Ignore stale responses (from an election we already moved past) and any
    // response once we are no longer campaigning.
    if (this.role !== "candidate" || p.term !== this.currentTerm) return;
    if (!p.granted) return;

    this.votesReceived.add(voter); // Set => duplicate vote msgs can't inflate count
    if (this.votesReceived.size >= this.majority) this.becomeLeader();
  }

  private onHeartbeat(p: HeartbeatPayload): void {
    // Heartbeat from a stale leader: reject by ignoring; our higher term will
    // reach them via the next RequestVote/response and force their step-down.
    if (p.term < this.currentTerm) return;
    // Valid (>=) leader: adopt term if newer, become/stay follower, reset timer.
    // This is what keeps a healthy cluster from ever starting a new election.
    if (p.term > this.currentTerm) {
      this.stepDown(p.term);
    } else {
      this.role = "follower";
      this.clearTimer(this.heartbeatTimer);
    }
    this.armElectionTimer();
  }

  /** Volatile state is meaningless after a crash; rebuild it. Persistent term/vote
   *  are intact (that's the crash-safety contract). A restarted node comes up as a
   *  follower and waits for either a heartbeat or its own election timeout. */
  protected override onRestart(): void {
    this.role = "follower";
    this.votesReceived.clear();
    this.electionTimer = 0;
    this.heartbeatTimer = 0;
    this.armElectionTimer();
  }

  // --- read-only accessors for invariants / metrics (no protocol effect) ------
  getRole(): Role {
    return this.role;
  }
  getTerm(): number {
    return this.currentTerm;
  }
}

// ---------------------------------------------------------------------------
// Scenario wiring
// ---------------------------------------------------------------------------

const NODE_IDS = ["n0", "n1", "n2", "n3", "n4"]; // 5-node cluster, majority = 3

/** Build the cluster and register the "≤1 leader per term" safety invariant.
 *  Returns the node list plus a snapshot of the FIRST leader (term + virtual
 *  time it was observed) so callers can compute election cost. */
function buildCluster(
  ctx: ScenarioContext,
  randomizeTimeout: boolean,
): {
  nodes: RaftElectionNode[];
  firstLeader: { term: number; atMs: number } | null;
  leaderHistory: Map<number, Set<string>>;
} {
  const nodes = NODE_IDS.map(
    (id) =>
      new RaftElectionNode(
        id,
        NODE_IDS.filter((p) => p !== id),
        ctx.rng,
        randomizeTimeout,
        ctx.clock,
        ctx.net,
      ),
  );

  // leaderHistory[term] = set of node ids that were leader in that term. The
  // safety invariant is: every set has size <= 1. We record into it by polling
  // roles after each event (the invariant check IS the poll point) — capturing
  // even a transient second leader, which is the whole reason the runner checks
  // after every event rather than at the end.
  const leaderHistory = new Map<number, Set<string>>();
  const box = { firstLeader: null as { term: number; atMs: number } | null };

  ctx.watch(
    invariant(
      "at-most-one-leader-per-term",
      () => {
        for (const n of nodes) {
          if (n.isDown()) continue; // a crashed node's stale role is not "a leader"
          if (n.getRole() === "leader") {
            const term = n.getTerm();
            let set = leaderHistory.get(term);
            if (!set) {
              set = new Set();
              leaderHistory.set(term, set);
            }
            set.add(n.id);
            if (box.firstLeader === null) {
              box.firstLeader = { term, atMs: ctx.clock.now() };
            }
          }
        }
        // The property: no term ever has two distinct elected leaders. This is the
        // claim the chapter exists to defend; if it ever returns false the runner
        // throws with the snapshot below.
        for (const set of leaderHistory.values()) if (set.size > 1) return false;
        return true;
      },
      () => ({
        terms: [...leaderHistory.entries()].map(([term, set]) => ({
          term,
          leaders: [...set],
        })),
        roles: nodes.map((n) => ({ id: n.id, role: n.getRole(), term: n.getTerm(), down: n.isDown() })),
      }),
    ),
  );

  for (const n of nodes) n.start();

  return {
    nodes,
    get firstLeader() {
      return box.firstLeader;
    },
    leaderHistory,
  };
}

/** Find the current unique live leader (or null). Used to measure failover. */
function findLiveLeader(nodes: RaftElectionNode[]): RaftElectionNode | null {
  const leaders = nodes.filter((n) => !n.isDown() && n.getRole() === "leader");
  return leaders.length === 1 ? leaders[0] : null;
}

// ---------------------------------------------------------------------------
// Experiment A: first-election cost distribution across seeds
// ---------------------------------------------------------------------------

interface ElectionCost {
  seed: number;
  term: number; // term in which the first leader emerged (== #election rounds)
  atMs: number; // virtual time the first leader was observed
}

function measureFirstElection(seed: number): ElectionCost {
  let captured: { term: number; atMs: number } | null = null;
  // We only need to observe the FIRST leader, so stop early once seen. The
  // scenario runner has no "stop on condition" hook, so we let it run a short
  // bounded window — long enough to always elect under randomization, short
  // enough to stay fast. The capture happens inside the invariant poll.
  runScenario(`A.first-election`, seed, {
    untilMs: 2000,
    setup(ctx) {
      const cluster = buildCluster(ctx, /*randomize*/ true);
      return () => {
        captured = cluster.firstLeader;
        return []; // suppress per-seed table; we aggregate ourselves
      };
    },
  });
  if (!captured) {
    throw new Error(`seed ${seed}: no leader elected within window (unexpected under randomization)`);
  }
  // TS narrowing: `captured` is assigned inside the returned closure which the
  // runner invokes before this point, but the compiler can't see that across the
  // callback boundary, so re-read via a local that it can narrow.
  const c: { term: number; atMs: number } = captured;
  return { seed, term: c.term, atMs: c.atMs };
}

function summarize(values: number[]): { min: number; mean: number; max: number } {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return { min, mean: Math.round(mean * 10) / 10, max };
}

// ---------------------------------------------------------------------------
// Main: run all experiments and print honest, code-computed numbers.
// ---------------------------------------------------------------------------

function main(): void {
  console.log("============================================================");
  console.log(" Stage 04 — Raft Leader Election (5 nodes, majority=3)");
  console.log(` election timeout: randomized [${ELECTION_BASE_MS}, ${ELECTION_BASE_MS + ELECTION_SPREAD_MS}) ms`);
  console.log(` heartbeat interval: ${HEARTBEAT_MS} ms  (must be << election timeout)`);
  console.log(" 注: 虚拟时间单位为模拟 ms; 绝对值随 toy 参数偏乐观, 可迁移的是相对趋势.");
  console.log("============================================================");

  // --- Experiment A: distribution of first-election cost over 12 seeds --------
  console.log("\n## A. 首次选主成本分布 (12 个独立 seed, 随机化超时)");
  const seeds = [1, 7, 13, 42, 99, 123, 256, 777, 1009, 2024, 31337, 65535];
  const costs = seeds.map(measureFirstElection);
  const terms = costs.map((c) => c.term);
  const times = costs.map((c) => c.atMs);
  const ts = summarize(terms);
  const tm = summarize(times);
  console.log(
    `  rounds(term) to first leader:  min=${ts.min}  mean=${ts.mean}  max=${ts.max}  (n=${seeds.length})`,
  );
  console.log(
    `  virtual time to first leader:  min=${tm.min}ms  mean=${tm.mean}ms  max=${tm.max}ms`,
  );
  console.log(`  per-seed: ${costs.map((c) => `seed${c.seed}=T${c.term}@${c.atMs}ms`).join("  ")}`);

  // --- Experiment B: leader-crash failover ------------------------------------
  // Let a leader emerge, crash it, measure virtual time until a NEW live leader
  // is stable. We read the times off the SimClock — genuinely simulated, not est.
  console.log("\n## B. Leader crash 触发重选 — failover 时间");
  runScenario("B.crash-failover", 7, {
    untilMs: 3000,
    setup(ctx) {
      const cluster = buildCluster(ctx, /*randomize*/ true);
      const state = {
        crashedAtMs: -1,
        crashedId: "",
        crashedTerm: -1,
        recoveredAtMs: -1,
        recoveredId: "",
        recoveredTerm: -1,
      };

      // At t=800ms a leader has long since emerged. Crash whoever is leader, then
      // watch for the next live leader at a strictly higher term.
      ctx.at(800, () => {
        const leader = findLiveLeader(cluster.nodes);
        if (!leader) return; // (won't happen by 800ms under randomization)
        state.crashedAtMs = ctx.clock.now();
        state.crashedId = leader.id;
        state.crashedTerm = leader.getTerm();
        leader.crash();
      });

      // Poll for recovery via repeated probes on the clock. We schedule a chain of
      // probes (every HEARTBEAT_MS) after the crash window; the first probe that
      // finds a new live leader at a higher term records the failover instant.
      const scheduleProbe = (atMs: number) => {
        ctx.at(atMs, () => {
          if (state.crashedAtMs < 0 || state.recoveredAtMs >= 0) {
            if (state.recoveredAtMs < 0 && atMs < 2800) scheduleProbe(atMs + HEARTBEAT_MS);
            return;
          }
          const leader = findLiveLeader(cluster.nodes);
          if (leader && leader.getTerm() > state.crashedTerm) {
            state.recoveredAtMs = ctx.clock.now();
            state.recoveredId = leader.id;
            state.recoveredTerm = leader.getTerm();
            return;
          }
          if (atMs < 2800) scheduleProbe(atMs + HEARTBEAT_MS);
        });
      };
      scheduleProbe(850);

      return () => {
        const failover =
          state.recoveredAtMs >= 0 ? state.recoveredAtMs - state.crashedAtMs : -1;
        return [
          { metric: "old leader (crashed)", value: `${state.crashedId} @T${state.crashedTerm} t=${state.crashedAtMs}ms` },
          { metric: "new leader (recovered)", value: `${state.recoveredId} @T${state.recoveredTerm} t=${state.recoveredAtMs}ms` },
          { metric: "failover time", value: `${failover}ms` },
        ];
      };
    },
  });

  // --- Experiment C: partition then heal --------------------------------------
  // Cut {n0,n1} (minority) from {n2,n3,n4} (majority). The majority side keeps/
  // elects a leader; the minority side CANNOT (no quorum) and just spins terms.
  // After heal, the higher-term side forces the other to step down — still ≤1
  // leader per term throughout.
  console.log("\n## C. 网络分区: 少数派 {n0,n1} | 多数派 {n2,n3,n4}");
  runScenario("C.partition-heal", 42, {
    untilMs: 4000,
    setup(ctx) {
      const cluster = buildCluster(ctx, /*randomize*/ true);
      const minority = ["n0", "n1"];
      const majority = ["n2", "n3", "n4"];
      const probe = { duringMinorityLeaders: -1, duringMajorityLeaders: -1, afterHealLeaders: -1, afterHealTerm: -1 };

      ctx.at(600, () => ctx.net.partition(minority, majority));
      // Mid-partition snapshot: count live leaders on each side.
      ctx.at(1800, () => {
        probe.duringMinorityLeaders = cluster.nodes.filter(
          (n) => minority.includes(n.id) && n.getRole() === "leader",
        ).length;
        probe.duringMajorityLeaders = cluster.nodes.filter(
          (n) => majority.includes(n.id) && n.getRole() === "leader",
        ).length;
      });
      ctx.at(2400, () => ctx.net.heal());
      ctx.at(3600, () => {
        const leader = findLiveLeader(cluster.nodes);
        probe.afterHealLeaders = cluster.nodes.filter((n) => n.getRole() === "leader").length;
        probe.afterHealTerm = leader ? leader.getTerm() : -1;
      });

      return () => [
        { metric: "leaders on minority side (no quorum)", value: probe.duringMinorityLeaders },
        { metric: "leaders on majority side", value: probe.duringMajorityLeaders },
        { metric: "leaders after heal (cluster-wide)", value: probe.afterHealLeaders },
        { metric: "leader term after heal", value: probe.afterHealTerm },
      ];
    },
  });

  // --- Experiment D: livelock — fixed vs randomized timeouts -------------------
  // THE failure mode. With identical timeouts every follower fires together, each
  // becomes a candidate and votes for itself, so every term has exactly 1 vote
  // per candidate => nobody reaches majority (3). The cluster climbs terms with
  // zero leaders. Then we flip randomization on and it converges in ~1 term.
  console.log("\n## D. 失败模式: 固定超时 -> split-vote 活锁 vs 随机化 -> 立即收敛");

  // D1: fixed timeout. We expect NO leader and many wasted terms.
  let fixedMaxTerm = 0;
  let fixedAnyLeader = false;
  runScenario("D1.fixed-timeout-livelock", 7, {
    untilMs: 2000,
    // Zero jitter so links are symmetric too — without symmetric latency a stray
    // ordering could accidentally let one node win, masking the livelock. We want
    // the PURE split-vote pathology, so we remove that escape hatch.
    network: { defaultBaselineMs: 10, defaultJitterMs: 0 },
    setup(ctx) {
      const cluster = buildCluster(ctx, /*randomize*/ false);
      return () => {
        fixedMaxTerm = Math.max(...cluster.nodes.map((n) => n.getTerm()));
        fixedAnyLeader = cluster.nodes.some((n) => n.getRole() === "leader");
        return [
          { metric: "leader elected?", value: fixedAnyLeader ? "yes" : "NO (livelock)" },
          { metric: "terms burned with no leader", value: fixedMaxTerm },
        ];
      };
    },
  });

  // D2: same seed, randomization ON. Expect a leader in a handful of terms.
  let randLeaderTerm = -1;
  let randLeaderAtMs = -1;
  runScenario("D2.randomized-timeout-converges", 7, {
    untilMs: 2000,
    network: { defaultBaselineMs: 10, defaultJitterMs: 0 },
    setup(ctx) {
      const cluster = buildCluster(ctx, /*randomize*/ true);
      return () => {
        const fl = cluster.firstLeader;
        if (fl) {
          randLeaderTerm = fl.term;
          randLeaderAtMs = fl.atMs;
        }
        return [
          { metric: "leader elected?", value: fl ? "yes" : "NO" },
          { metric: "term of first leader", value: randLeaderTerm },
          { metric: "virtual time to elect", value: `${randLeaderAtMs}ms` },
        ];
      };
    },
  });

  console.log("\n## 对比结论 (同 seed=7, 同零抖动网络):");
  console.log(
    `  固定超时:  ${fixedAnyLeader ? "elected" : "活锁, 0 leader"}, 烧掉 ${fixedMaxTerm} 个 term 仍选不出.`,
  );
  console.log(
    `  随机超时:  第 ${randLeaderTerm} 个 term 选出 leader, 用时 ${randLeaderAtMs}ms.`,
  );
  if (!fixedAnyLeader && randLeaderTerm > 0) {
    console.log(
      `  => 随机化把 "无限 split-vote" 变成 "${randLeaderTerm} 轮内收敛". 这就是 Raft 用随机性买 liveness 的代价与收益.`,
    );
  }
  console.log("\n[safety] 全部场景, 每个事件后都校验 '任一 term 至多一个 leader' — 从未被违反.");
}

main();
