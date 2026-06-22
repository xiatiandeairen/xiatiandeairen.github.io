// stage06-cap-tradeoff.ts — CAP is not a slogan; it is a measurable bill.
//
// The chapter's claim, made executable: under a network PARTITION you cannot
// have both (C) every replica agreeing and (A) every request succeeding. You
// pick. This stage runs the SAME partition script against the SAME workload
// twice — once with a CP policy (quorum writes; the minority side refuses) and
// once with an AP policy (both sides accept) — and prints the two-dimensional
// price each pays: (availability = write success rate) vs (consistency = number
// of divergent keys observed after the partition heals).
//
// Why a quorum at all: with N=5 replicas, a WRITE quorum W and READ quorum R
// give single-key linearizable-ish reads iff W + R > N AND W > N/2. We use
// W=3, R=3 (W+R=6>5, W=3>2.5). The ">N/2" part is the CAP lever: only ONE side
// of any partition can ever hold a majority, so at most one side can complete a
// quorum write. That is mechanically why CP loses availability on the minority
// side — not a policy choice, a counting fact.
//
// The honest failure mode (the hole chapter 7 fills): the AP policy here pretends
// "merging divergent replicas is easy" by using last-write-wins keyed on each
// node's LOCAL PHYSICAL clock. We then give the minority coordinator a skewed
// clock (it runs behind real/virtual time). Result: a write that genuinely
// happened LATER in real time carries a SMALLER timestamp and loses the merge —
// a silent lost update. LWW-by-wall-clock is the canonical wrong answer; CRDTs
// (ch.7) replace the physical timestamp with causality so this can't happen.
//
// Honesty about the numbers: these are simulated counts on a 5-node toy with a
// scripted partition and a fixed workload — absolute rates are an artifact of
// the script (how many writes land on which side, when the cut happens). What
// transfers is the RELATIVE shape: CP trades availability for zero divergence;
// AP keeps availability but pays divergence; naive LWW merge silently DROPS a
// real write. Same seed ⇒ byte-identical, re-runnable.

import { SimClock } from "./core/clock.js";
import { seededRng } from "./core/prng.js";
import { Network } from "./core/network.js";
import { Stats } from "./core/metrics.js";
import { printTable } from "./core/assert.js";

const SEED = 7;
const NODE_IDS = ["n0", "n1", "n2", "n3", "n4"] as const;
const MAJORITY = ["n0", "n1", "n2"]; // 3 nodes — holds quorum during the cut
const MINORITY = ["n3", "n4"]; // 2 nodes — cannot reach W=3 alone
const WRITE_QUORUM = 3; // W: acks required for a CP write to commit
const N = NODE_IDS.length;

// Partition window in virtual ms. Writes are scheduled across [0, WORKLOAD_END]
// so some land before the cut, some during, some after the heal.
const CUT_AT_MS = 200;
const HEAL_AT_MS = 800;
const WORKLOAD_END_MS = 1000;
const ACK_TIMEOUT_MS = 120; // a CP write that hasn't collected W acks by now fails

// Clock skew injected into the MINORITY coordinator's LWW timestamps under the
// AP policy. Negative = its physical clock lags behind true virtual time, so its
// writes look "older" than they really are. This is what makes a later write
// lose the last-write-wins merge. 250ms of skew is large but not absurd: real
// NTP-less or VM-paused hosts drift this much, and the bug exists at ANY skew >
// inter-write gap — we pick a value that guarantees a visible inversion.
const MINORITY_CLOCK_SKEW_MS = -250;

/** One client write request: which key, what value, when issued, and which
 *  coordinator (side) it hits. `realTimeMs` is the TRUE virtual time the write
 *  is issued — the ground truth against which we judge whether LWW lost a real,
 *  later write. */
interface WriteRequest {
  key: string;
  value: string;
  issuedAtMs: number;
  coordinator: string; // node that receives the client write
}

/** A versioned value stored at a replica under the AP policy: the value plus the
 *  PHYSICAL timestamp the coordinator stamped it with. LWW compares these stamps.
 *  The whole bug lives in `stampMs` being a wall clock, not a logical clock. */
interface VersionedValue {
  value: string;
  stampMs: number; // coordinator's local physical clock at write time
  realTimeMs: number; // ground-truth virtual time (NOT visible to LWW) for auditing
}

/** Outcome of running one strategy: the availability + consistency numbers the
 *  cost table is built from, plus enough detail to prove the lost update. */
interface StrategyOutcome {
  label: string;
  writesIssued: number;
  writesCommitted: number;
  writesRejected: number; // CP: minority writes that couldn't reach quorum
  divergentKeysAfterHeal: number; // keys where replicas disagree post-heal
  lostUpdate?: { key: string; lostValue: string; lostRealTimeMs: number; winnerValue: string };
}

/** Build the shared workload: a fixed set of writes, deterministically split
 *  across the two sides and across the partition timeline. Both strategies see
 *  the EXACT same requests — that is what makes the cost comparison fair. */
function buildWorkload(): WriteRequest[] {
  const rng = seededRng(SEED);
  const writes: WriteRequest[] = [];
  // 12 writes spread over the timeline. Keys deliberately COLLIDE across sides
  // (k0/k1 written on both) so the AP policy has something to diverge on.
  const keys = ["k0", "k1", "k2"];
  for (let i = 0; i < 12; i++) {
    const key = keys[rng.int(keys.length)];
    // Alternate which side gets the write, but jitter the exact coordinator so
    // it's not a trivial round-robin.
    const side = rng.bool(0.5) ? MAJORITY : MINORITY;
    const coordinator = side[rng.int(side.length)];
    const issuedAtMs = Math.floor((i + 1) * (WORKLOAD_END_MS / 13));
    writes.push({ key, value: `${key}=v${i}`, issuedAtMs, coordinator });
  }
  return writes;
}

/** CP strategy: a write commits ONLY if its coordinator can collect W acks. We
 *  model the ack collection with the real Network + SimClock so the timeout is a
 *  genuine simulated event, not an assumption. On the minority side during the
 *  cut, cross-partition acks are dropped by the bus → the coordinator times out →
 *  the write is rejected (unavailable but never divergent). */
function runCpStrategy(workload: WriteRequest[]): StrategyOutcome {
  const clock = new SimClock();
  const rng = seededRng(SEED + 1); // distinct stream so latency jitter differs from AP, still fixed
  const stats = new Stats();
  const net = new Network(clock, rng, stats);

  // Replica store: ONLY committed values land here. Uncommitted writes live in a
  // per-replica `pending` buffer and are applied solely on an explicit Commit.
  // This two-phase shape is what makes CP's divergence provably 0: a write that
  // failed quorum was never committed anywhere, so it leaves no trace on any
  // replica — there is nothing to reconcile after heal.
  const store = new Map<string, Map<string, string>>(); // nodeId -> key -> committed value
  const pending = new Map<string, Map<number, { key: string; value: string }>>(); // nodeId -> writeId -> buffered
  for (const id of NODE_IDS) {
    store.set(id, new Map());
    pending.set(id, new Map());
  }

  let committed = 0;
  let rejected = 0;

  // Phase 1: replica buffers the write and acks. Phase 2: on Commit it promotes
  // the buffered value to the store. The coordinator counts acks; the Network
  // silently drops cross-partition acks — exactly how a minority coordinator
  // experiences the cut: the acks it needs never come back.
  const ackCounts = new Map<number, number>(); // writeId -> acks seen by its coordinator
  const writeCoordinator = new Map<number, string>(); // writeId -> coordinator (for Commit fan-out)
  const writeDecided = new Set<number>(); // committed OR rejected (terminal)

  for (const id of NODE_IDS) {
    net.register(id, (msg) => {
      if (msg.kind === "Prepare") {
        const { writeId, key, value, coordinator } = msg.payload as {
          writeId: number;
          key: string;
          value: string;
          coordinator: string;
        };
        pending.get(id)!.set(writeId, { key, value }); // buffer, do NOT apply yet
        net.send(id, coordinator, "Ack", { writeId });
      } else if (msg.kind === "Commit") {
        const { writeId } = msg.payload as { writeId: number };
        const buf = pending.get(id)!.get(writeId);
        if (buf) {
          store.get(id)!.set(buf.key, buf.value); // promote buffered -> committed
          pending.get(id)!.delete(writeId);
        }
      } else if (msg.kind === "CommitState") {
        // Heal-time anti-entropy: a replica that missed a Commit during the cut
        // accepts the committer's authoritative committed value. CP has a single
        // committed history (no concurrent commits to the same key from two sides,
        // because only one side ever held quorum), so this never conflicts.
        const { key, value } = msg.payload as { key: string; value: string };
        store.get(id)!.set(key, value);
      } else if (msg.kind === "Ack") {
        const { writeId } = msg.payload as { writeId: number };
        if (writeDecided.has(writeId)) return; // already terminal; ignore late acks
        const c = (ackCounts.get(writeId) ?? 0) + 1;
        ackCounts.set(writeId, c);
        if (c >= WRITE_QUORUM) {
          writeDecided.add(writeId);
          committed++;
          // Broadcast Commit from the write's own coordinator so the quorum (and,
          // post-heal, everyone) applies it. Cross-partition Commits during the cut
          // are dropped, but the committing side IS the majority, so the value
          // reaches a majority now; the heal-time CommitState sweep catches the rest.
          const coord = writeCoordinator.get(writeId)!;
          for (const peer of NODE_IDS) net.send(coord, peer, "Commit", { writeId });
        }
      }
    });
  }

  scheduleFaults(clock, net);

  // Issue each write: the coordinator buffers its own copy (self-ack counts as 1),
  // Prepares to peers, and arms a timeout. No quorum by the deadline ⇒ rejected.
  // A rejected write's buffer is dropped on every replica at heal-time gossip; it
  // never reaches `store`, so it can't cause divergence.
  for (let i = 0; i < workload.length; i++) {
    const w = workload[i];
    const writeId = i;
    clock.schedule(w.issuedAtMs, () => {
      ackCounts.set(writeId, 1); // coordinator's own buffered copy is one ack
      writeCoordinator.set(writeId, w.coordinator);
      pending.get(w.coordinator)!.set(writeId, { key: w.key, value: w.value });
      for (const peer of NODE_IDS) {
        if (peer === w.coordinator) continue;
        net.send(w.coordinator, peer, "Prepare", {
          writeId,
          key: w.key,
          value: w.value,
          coordinator: w.coordinator,
        });
      }
      clock.schedule(ACK_TIMEOUT_MS, () => {
        if (writeDecided.has(writeId)) return; // committed in time
        writeDecided.add(writeId);
        rejected++;
        // Abort: drop the buffered (never-committed) write on every replica. This
        // is the CP guarantee made concrete — an unavailable write leaves ZERO
        // residue, so post-heal divergence is 0 by construction.
        for (const id of NODE_IDS) pending.get(id)!.delete(writeId);
      });
    });
  }

  // Anti-entropy at heal: re-broadcast committed state so any replica that missed
  // a Commit during the cut (e.g. it was on the far side when a near-side quorum
  // committed) catches up. Reads the committing replica's store and pushes it.
  clock.schedule(HEAL_AT_MS + 100, () => {
    // n0 is always on the majority side, so its store is the authoritative
    // committed history; push it to everyone to converge stragglers.
    for (const [key, value] of store.get(NODE_IDS[0])!) {
      for (const peer of NODE_IDS) net.send(NODE_IDS[0], peer, "CommitState", { key, value });
    }
  });

  clock.run(WORKLOAD_END_MS + ACK_TIMEOUT_MS + 400);

  return {
    label: "CP (quorum write, minority refuses)",
    writesIssued: workload.length,
    writesCommitted: committed,
    writesRejected: rejected,
    divergentKeysAfterHeal: countDivergentKeys(store),
  };
}

/** AP strategy: EVERY coordinator accepts writes locally and immediately, even
 *  during the partition (high availability). Replicas reconcile after heal via
 *  last-write-wins on the coordinator's physical timestamp. This is where the
 *  consistency bill comes due — and where the naive merge silently drops a real
 *  write because the minority coordinator's clock is skewed. */
function runApStrategy(workload: WriteRequest[]): StrategyOutcome {
  const clock = new SimClock();
  const rng = seededRng(SEED + 2);
  const stats = new Stats();
  const net = new Network(clock, rng, stats);

  // Per-replica versioned store. During the cut, the two sides build up
  // DIVERGENT versions for the same key; after heal we run the LWW merge.
  const store = new Map<string, Map<string, VersionedValue>>();
  for (const id of NODE_IDS) store.set(id, new Map());

  let committed = 0;
  // The consistency cost is measured AT HEAL, before anti-entropy papers over the
  // conflicts. After the LWW sweep runs the replicas converge to a single value
  // per key, so a post-sweep divergence count would read 0 and HIDE the very cost
  // the chapter is about. The real bill is "how many keys did the two sides
  // disagree on when the cut healed" — that's what a correct merge would have had
  // to reconcile, and what LWW silently (sometimes wrongly) collapses.
  let divergentAtHeal = 0;

  for (const id of NODE_IDS) {
    net.register(id, (msg) => {
      if (msg.kind === "Replicate") {
        const v = msg.payload as { key: string } & VersionedValue;
        // Apply with LWW even on replication: keep the higher physical stamp.
        // (The bug is baked in here too — replication uses the same broken
        // comparator as the post-heal merge.)
        applyLww(store.get(id)!, v.key, {
          value: v.value,
          stampMs: v.stampMs,
          realTimeMs: v.realTimeMs,
        });
      }
    });
  }

  scheduleFaults(clock, net);

  for (const w of workload) {
    clock.schedule(w.issuedAtMs, () => {
      // AP accepts unconditionally — that's the availability win. The timestamp
      // is the coordinator's LOCAL physical clock: true virtual time PLUS this
      // coordinator's skew. Majority nodes have zero skew; the minority side lags.
      const skew = MINORITY.includes(w.coordinator) ? MINORITY_CLOCK_SKEW_MS : 0;
      const stampMs = clock.now() + skew;
      const versioned: VersionedValue = { value: w.value, stampMs, realTimeMs: clock.now() };
      applyLww(store.get(w.coordinator)!, w.key, versioned);
      committed++;
      // Best-effort replicate to peers. Cross-partition sends are dropped by the
      // bus during the cut — that's how divergence accumulates. After heal they
      // flow, but the damage (skewed stamps) is already encoded in the values.
      for (const peer of NODE_IDS) {
        if (peer === w.coordinator) continue;
        net.send(w.coordinator, peer, "Replicate", { key: w.key, ...versioned });
      }
    });
  }

  // Measure consistency cost AT heal (before reconciliation): how many keys do
  // the replicas disagree on right now. These are the conflicts the partition
  // produced — the consistency bill AP ran up while staying available.
  clock.schedule(HEAL_AT_MS, () => {
    divergentAtHeal = countDivergentKeys(store);
  });

  // The post-heal reconciliation: gossip every node's current versions to every
  // other node and let LWW settle. Scheduled just after heal so in-flight
  // replications have flushed. This models an anti-entropy sweep. It CONVERGES
  // the replicas (post-sweep divergence is 0), but the value it converges to can
  // be wrong — see detectLostUpdate. Convergence is not correctness.
  clock.schedule(HEAL_AT_MS + 100, () => {
    const snapshot = NODE_IDS.map((id) => ({ id, kv: new Map(store.get(id)!) }));
    for (const src of snapshot) {
      for (const dst of NODE_IDS) {
        if (dst === src.id) continue;
        for (const [key, v] of src.kv) {
          net.send(src.id, dst, "Replicate", { key, ...v });
        }
      }
    }
  });

  clock.run(WORKLOAD_END_MS + 400);

  const lostUpdate = detectLostUpdate(workload, store);

  return {
    label: "AP (both sides accept, LWW merge)",
    writesIssued: workload.length,
    writesCommitted: committed,
    writesRejected: 0, // AP never rejects — that's the point
    divergentKeysAfterHeal: divergentAtHeal,
    lostUpdate,
  };
}

/** Last-write-wins by physical stamp. The comparator at the heart of the bug:
 *  it trusts `stampMs` as a total order on real time. It isn't — a skewed clock
 *  makes a later write compare as older. Ties (equal stamp) keep the existing
 *  value (deterministic, but the tie itself is already a correctness smell). */
function applyLww(kv: Map<string, VersionedValue>, key: string, incoming: VersionedValue): void {
  const cur = kv.get(key);
  if (!cur || incoming.stampMs > cur.stampMs) kv.set(key, incoming);
}

/** Schedule the partition cut and heal as SimClock events so faults interleave
 *  with protocol messages deterministically. Both strategies use the IDENTICAL
 *  fault timeline — that's what makes the cost comparison fair. We rely purely on
 *  the bus: during [CUT, HEAL] cross-group sends are dropped, which is the only
 *  mechanism either strategy needs (CP times out for lack of acks; AP diverges
 *  for lack of replication). No local partition flag is needed. */
function scheduleFaults(clock: SimClock, net: Network): void {
  clock.schedule(CUT_AT_MS, () => net.partition(MAJORITY, MINORITY));
  clock.schedule(HEAL_AT_MS, () => net.heal());
}

/** Count keys on which the replicas do NOT all agree on the final value. This is
 *  the consistency metric: 0 = every replica converged (what CP guarantees);
 *  >0 = replicas serve different values for the same key (the AP cost, or, after
 *  a correct merge, what's left unreconciled). Compares by value string only —
 *  a reader querying any replica would observe the disagreement. */
function countDivergentKeys(store: Map<string, Map<string, unknown>>): number {
  const allKeys = new Set<string>();
  for (const kv of store.values()) for (const k of kv.keys()) allKeys.add(k);
  let divergent = 0;
  for (const key of allKeys) {
    const values = new Set<string>();
    for (const id of NODE_IDS) {
      const v = store.get(id)!.get(key);
      // Normalize: CP stores raw strings, AP stores VersionedValue. Extract the
      // observable value either way so the metric means "what would a client read".
      const observed =
        v === undefined ? "<absent>" : typeof v === "string" ? v : (v as VersionedValue).value;
      values.add(observed);
    }
    if (values.size > 1) divergent++;
  }
  return divergent;
}

/** Prove the lost update: find a key where the value that WON the LWW merge is
 *  NOT the one that was actually written last in real (virtual) time. If the
 *  merge were correct, the latest real write would always win. When the skewed
 *  minority clock makes an earlier-stamped-but-later-real write win, the genuine
 *  last write is gone — that's the silent data loss. Returns the dropped write so
 *  the demo can print exactly which real write vanished. */
function detectLostUpdate(
  workload: WriteRequest[],
  store: Map<string, Map<string, VersionedValue>>,
): StrategyOutcome["lostUpdate"] {
  // Ground truth: for each key, the value written latest in REAL virtual time.
  const latestRealWrite = new Map<string, WriteRequest>();
  for (const w of workload) {
    const cur = latestRealWrite.get(w.key);
    if (!cur || w.issuedAtMs > cur.issuedAtMs) latestRealWrite.set(w.key, w);
  }
  // What the cluster actually converged to (read from any replica; post-heal
  // they agree on the LWW winner — that's the whole point of the merge).
  for (const [key, truth] of latestRealWrite) {
    const winner = store.get(NODE_IDS[0])!.get(key);
    if (winner && winner.value !== truth.value) {
      return {
        key,
        lostValue: truth.value,
        lostRealTimeMs: truth.issuedAtMs,
        winnerValue: winner.value,
      };
    }
  }
  return undefined;
}

function main(): void {
  console.log("=== Stage 06: CAP 取舍 — 同一分区脚本下 CP vs AP 的二维代价 ===\n");
  console.log(
    `集群 N=${N} (多数派 ${MAJORITY.join(",")} | 少数派 ${MINORITY.join(",")}), ` +
      `写 quorum W=${WRITE_QUORUM}`,
  );
  console.log(
    `分区窗口 [${CUT_AT_MS}, ${HEAL_AT_MS}]ms, 工作负载 ${buildWorkload().length} 次写, seed=${SEED}\n`,
  );

  const workload = buildWorkload();
  const cp = runCpStrategy(workload);
  const ap = runApStrategy(workload);

  // The headline: one table, two strategies, the two CAP axes side by side.
  console.log("二维代价表 (可用性 = 写成功率; 一致性 = 恢复后分歧键数):");
  const availPct = (o: StrategyOutcome) =>
    `${((o.writesCommitted / o.writesIssued) * 100).toFixed(1)}%`;
  printTable([
    {
      strategy: cp.label,
      "写成功 (avail)": `${cp.writesCommitted}/${cp.writesIssued} (${availPct(cp)})`,
      被拒: cp.writesRejected,
      "恢复后分歧键 (consistency)": cp.divergentKeysAfterHeal,
    },
    {
      strategy: ap.label,
      "写成功 (avail)": `${ap.writesCommitted}/${ap.writesIssued} (${availPct(ap)})`,
      被拒: ap.writesRejected,
      "恢复后分歧键 (consistency)": ap.divergentKeysAfterHeal,
    },
  ]);

  console.log(
    "\n解读: CP 在分区期间拒绝少数派写 (可用性下降), 换来恢复后 0 分歧 — " +
      "选择了一致性。AP 两侧全收 (可用性满格), 但恢复后存在分歧键 — 选择了可用性。",
  );

  // The failure-mode payoff: AP's "merge is easy" claim is a lie under clock
  // skew. Print the exact real write that LWW silently dropped.
  console.log("\n--- 失败模式: AP 的 last-write-wins 在时钟偏移下丢写 ---");
  if (ap.lostUpdate) {
    const lu = ap.lostUpdate;
    console.log(
      `少数派 coordinator 时钟偏移 ${MINORITY_CLOCK_SKEW_MS}ms (落后)。` +
        `键 "${lu.key}" 真实最后一次写是 "${lu.lostValue}" (real t=${lu.lostRealTimeMs}ms),`,
    );
    console.log(
      `但 LWW 按物理时间戳合并后, 集群收敛到 "${lu.winnerValue}" —— ` +
        `真实更晚的写 "${lu.lostValue}" 被静默丢弃 (lost update)。`,
    );
    console.log(
      "根因: 物理时钟不是真实时间的全序。这正是第 7 章 CRDT 用因果关系 " +
        "(而非 wall-clock) 替代 LWW 要补的洞。",
    );
  } else {
    console.log("(未检测到丢写 — 检查 seed/skew 配置)");
  }
}

main();
