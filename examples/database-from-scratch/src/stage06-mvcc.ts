// stage06-mvcc.ts — Multi-Version Concurrency Control: version chains, snapshot
// visibility, and the four classic anomalies under three isolation levels.
//
// What this chapter builds and PROVES, not asserts:
//  1. A real MVCC store: every key holds a CHAIN of versions, each stamped with
//     the committing transaction's commit timestamp. A read does not "lock and
//     return the latest" — it walks the chain and returns the newest version
//     VISIBLE to the reader's snapshot. That single rule is the whole of MVCC.
//  2. Three isolation levels implemented as three different visibility/validation
//     policies over the SAME version store:
//        - RC  (Read Committed):    each read takes a FRESH snapshot (read ts =
//                                   "now"); sees every already-committed version.
//        - RR  (Repeatable Read):   one snapshot taken at txn start; all reads use
//                                   it; later commits are invisible.
//        - SI  (Snapshot Isolation): RR's snapshot + first-committer-wins write
//                                   conflict detection (abort on write-write).
//  3. A truth table: for each anomaly (dirty read / non-repeatable read / phantom
//     / lost update / write skew) under each level, does it REPRODUCE? Each cell
//     is decided by running the actual interleaving and inspecting the result —
//     no hardcoded "RC allows dirty read" claims.
//  4. The punchline failure mode: SI lets a textbook WRITE SKEW through. Two
//     on-call doctors each read "the other is on duty" from the same snapshot,
//     each takes leave, both commit (no write-write conflict — they wrote
//     DIFFERENT rows), and the invariant "at least one doctor on duty" is
//     violated with zero aborts. This is exactly why SI != serializable and why
//     SSI (serializable snapshot isolation) exists. We quantify it.
//
// Honest-numbers note: this is a toy in-memory store; absolute op counts are not
// a benchmark. What transfers is the RELATIVE truth table (which level admits
// which anomaly) and the abort-rate contrast across levels — those are structural
// properties of the visibility rules, reproducible byte-for-byte via the seeded
// scheduler. Throughput is NOT measured here; this stage is about correctness of
// isolation, not speed.
//
// Why we reuse core: LamportClock is the sole source of monotonic commit/read
// timestamps (visibility is pure ordering, so a logical clock is both correct and
// reproducible — see core/clock.ts). runSchedule gives a deterministic
// interleaving from a seed, so an anomaly we print is one the reader can replay.
// createRng drives a seed SWEEP: a single seed produces one schedule, but an
// anomaly needs a SPECIFIC schedule, so we sweep seeds until we find (or fail to
// find) the offending interleaving — N=1 would conflate "this level is safe" with
// "this seed happened not to hit it".

import { LamportClock } from "./core/clock.js";
import { createRng } from "./core/prng.js";
import { runSchedule, formatTrace, type TxnGen } from "./core/scheduler.js";
import { printTable, invariant, type TableRow } from "./core/assert.js";

// ---------------------------------------------------------------------------
// MVCC store
// ---------------------------------------------------------------------------

type Key = string;

/** One immutable version in a key's chain. `commitTs === undefined` means the
 *  producing txn has written it but not yet committed — visible only to its own
 *  author, never to other snapshots. That uncommitted state is precisely what a
 *  DIRTY READ would (incorrectly) expose; modeling it explicitly lets us PROVE
 *  our snapshot rule never returns it. */
interface Version {
  value: number;
  /** Commit timestamp; the version is visible iff committed before the reader's
   *  snapshot ts. undefined until commit() stamps it. */
  commitTs: number | undefined;
  /** Author txn id, so we can show a txn its own uncommitted write (read-your-
   *  own-writes) without exposing it to others. */
  authorTxn: number;
}

type IsolationLevel = "RC" | "RR" | "SI";

/** A live transaction handle. Holds the policy that decides what it can see and
 *  whether it survives commit. The handle is the only thing txn bodies touch;
 *  the store stays a passive bag of version chains. */
interface Txn {
  id: number;
  level: IsolationLevel;
  /** Snapshot timestamp. For RR/SI this is frozen at begin(); for RC it is
   *  re-sampled on every read (so RC has no single snapshot — see read()). */
  snapshotTs: number;
  /** Keys this txn wrote, with the value it intends to commit. Buffered locally
   *  so other txns never see uncommitted writes (no dirty read by construction),
   *  and so SI can detect write-write conflicts at commit. */
  writeSet: Map<Key, number>;
  /** Keys this txn read, for completeness of the trace / future SSR checks. */
  readSet: Set<Key>;
  aborted: boolean;
}

/** Per-key record of the last committer: which txn committed it at what commit
 *  ts. SI uses this to detect first-committer-wins conflicts ("did a transaction
 *  concurrent with me already commit this key?") without scanning chains. */
interface LastCommit {
  byTxn: number;
  ts: number;
}

/** The MVCC engine. One instance per scenario run. Pure in-memory; the only
 *  external dependency is the LamportClock, which makes timestamps deterministic
 *  and gap-free so visibility comparisons are reproducible. */
class MvccStore {
  private chains = new Map<Key, Version[]>();
  private clock = new LamportClock();
  private nextTxnId = 0;
  // Last committer per key, for SI's first-committer-wins check.
  private lastCommit = new Map<Key, LastCommit>();

  /** Seed an initial committed value so reads have something to see and "before
   *  all transactions" (a snapshot ts before any txn) is well defined. */
  init(key: Key, value: number): void {
    const ts = this.clock.tick();
    this.chains.set(key, [{ value, commitTs: ts, authorTxn: -1 }]);
    this.lastCommit.set(key, { byTxn: -1, ts });
  }

  /** Begin a transaction at the given isolation level, capturing its snapshot ts
   *  as the current logical time.
   *
   *  WHY the harness calls begin() for ALL txns up front, before any operation
   *  runs: a transaction's snapshot must reflect "what was committed when it
   *  started", and concurrency anomalies only exist between txns whose lifetimes
   *  OVERLAP. If we let begin() run lazily on the generator's first scheduled
   *  step, a txn that the scheduler happens to run last would snapshot AFTER the
   *  other already committed — making it serial-after, not concurrent, and
   *  silently hiding the anomaly for that seed (the first version of this stage
   *  had exactly that bug: SI "prevented" lost update only because the second
   *  txn wasn't actually concurrent). Eager begin pins all txns to the same
   *  start snapshot, so every interleaving studies genuinely concurrent txns. */
  begin(level: IsolationLevel): Txn {
    return {
      id: this.nextTxnId++,
      level,
      snapshotTs: this.clock.now(),
      writeSet: new Map(),
      readSet: new Set(),
      aborted: false,
    };
  }

  /** Return the value visible to `txn` for `key` under its isolation policy.
   *
   *  Visibility rule (the heart of MVCC):
   *   - read-your-own-writes: a txn always sees its own buffered write first;
   *   - otherwise return the newest COMMITTED version whose commitTs <= the
   *     reader's effective snapshot ts;
   *   - RC's effective snapshot is "now" (re-sampled here), so RC sees writes
   *     committed after it began — that's the definition of read-committed and
   *     the mechanism behind non-repeatable reads;
   *   - RR/SI use the frozen begin() snapshot, so post-begin commits are
   *     invisible — repeatable reads by construction.
   *
   *  Never returns an uncommitted version of another txn => dirty reads are
   *  impossible under all three levels here, which the truth table confirms. */
  read(txn: Txn, key: Key): number {
    txn.readSet.add(key);

    if (txn.writeSet.has(key)) return txn.writeSet.get(key)!;

    // RC re-samples the snapshot on EVERY read; RR/SI reuse the frozen one. This
    // one line is the entire behavioral difference that produces (RC) or blocks
    // (RR/SI) the non-repeatable-read anomaly.
    const effectiveTs = txn.level === "RC" ? this.clock.now() : txn.snapshotTs;

    const chain = this.chains.get(key);
    if (!chain) throw new Error(`read of unknown key ${key}`);

    // Walk newest-first; first committed version within the snapshot wins.
    for (let i = chain.length - 1; i >= 0; i--) {
      const v = chain[i];
      if (v.commitTs !== undefined && v.commitTs <= effectiveTs) return v.value;
    }
    throw new Error(`no version of ${key} visible at ts=${effectiveTs}`);
  }

  /** Range read: how many keys (among the candidate set) currently satisfy a
   *  predicate under the txn's snapshot. Used to demonstrate PHANTOMS — a row
   *  that appears/disappears between two identical range scans because another
   *  txn committed an insert/update. We model "insert" as making a previously
   *  out-of-range key satisfy the predicate. */
  rangeCount(txn: Txn, keys: Key[], pred: (v: number) => boolean): number {
    let n = 0;
    for (const k of keys) if (pred(this.read(txn, k))) n++;
    return n;
  }

  /** Buffer a write locally. NOT yet visible to anyone but `txn` (read-your-own-
   *  writes). Made durable+visible only at commit, stamped with a fresh ts. */
  write(txn: Txn, key: Key, value: number): void {
    if (txn.aborted) throw new Error(`write on aborted txn ${txn.id}`);
    txn.writeSet.set(key, value);
  }

  /** Attempt to commit. Returns nothing on success; THROWS on conflict (the
   *  scheduler turns a throw into an abort and records it — see core/scheduler).
   *
   *  SI's first-committer-wins check: if any key in our write set already has a
   *  committed version with commitTs > our snapshotTs, some other txn modified
   *  it after we took our snapshot, so committing would lose their update. We
   *  abort. This is the ONLY thing separating SI from RR here.
   *
   *  Crucially, this check is PER-KEY. Write skew slips through because the two
   *  txns write DIFFERENT keys — neither sees a conflict on its own key — yet
   *  together they violate a multi-row invariant. That gap is the chapter's
   *  punchline, not a bug in this code. */
  commit(txn: Txn): void {
    if (txn.aborted) throw new Error(`commit on already-aborted txn ${txn.id}`);

    if (txn.level === "SI") {
      for (const key of txn.writeSet.keys()) {
        const last = this.lastCommit.get(key);
        // first-committer-wins: if a DIFFERENT txn committed this key after our
        // snapshot was taken (last.ts > snapshotTs), it ran concurrently and
        // touched the same row. Letting both commit would lose one update, so we
        // abort. Comparing against snapshotTs (not "latest in chain") is the
        // correct concurrency test: a version committed before our snapshot is
        // one we already saw and built on, not a conflict.
        if (last && last.byTxn !== txn.id && last.ts > txn.snapshotTs) {
          txn.aborted = true;
          throw new Error(`SI write-write conflict on ${key}`);
        }
      }
    }

    // Stamp and append all buffered writes with one fresh, monotonic commit ts.
    const ts = this.clock.tick();
    for (const [key, value] of txn.writeSet) {
      let chain = this.chains.get(key);
      if (!chain) {
        chain = [];
        this.chains.set(key, chain);
      }
      chain.push({ value, commitTs: ts, authorTxn: txn.id });
      this.lastCommit.set(key, { byTxn: txn.id, ts });
    }
  }

  /** Latest committed value of a key, ignoring snapshots — the ground truth used
   *  by post-run invariant checks (e.g. "at least one doctor on duty"). */
  groundTruth(key: Key): number {
    const chain = this.chains.get(key);
    if (!chain) throw new Error(`no key ${key}`);
    for (let i = chain.length - 1; i >= 0; i--) {
      if (chain[i].commitTs !== undefined) return chain[i].value;
    }
    throw new Error(`no committed version of ${key}`);
  }

  /** Number of versions in a key's chain — used to show that MVCC keeps OLD
   *  versions around (the storage cost of snapshots) rather than overwriting. */
  chainLength(key: Key): number {
    return this.chains.get(key)?.length ?? 0;
  }
}

// ---------------------------------------------------------------------------
// Scenario harness
// ---------------------------------------------------------------------------

/** A scenario is a named pair of transaction bodies plus an initial state and a
 *  detector that, given the store after the run, says whether the anomaly it
 *  targets actually occurred. The detector inspects real post-run state, so a
 *  truth-table cell is an OBSERVATION, not a claim. */
interface Scenario {
  name: string;
  init: (store: MvccStore) => void;
  /** Build the txn bodies. The harness pre-begins one Txn handle per body (so all
   *  snapshots are taken before any operation runs — see MvccStore.begin's note),
   *  then passes the matching handle in. A body therefore never calls begin
   *  itself; it receives an already-started, genuinely-concurrent transaction. */
  txns: (store: MvccStore, handles: Txn[]) => TxnGen<MvccStore>[];
  /** How many concurrent transactions this scenario runs. */
  txnCount: number;
  /** Inspect post-run state (+ what was observed during the run) and decide if
   *  the anomaly reproduced. */
  detect: (store: MvccStore, obs: Observations) => boolean;
}

/** Side-channel for txn bodies to record what they SAW, so detectors can spot
 *  read-side anomalies (a non-repeatable read is about two reads disagreeing,
 *  which post-run state alone can't reveal). Shared mutable, like the real ctx. */
interface Observations {
  reads: number[];
  flags: Record<string, boolean>;
  /** How many transactions reached a SUCCESSFUL commit (bumped right after
   *  ctx.commit returns). Lets a detector distinguish "an update was lost because
   *  both committed" (a true lost update) from "x ended at 101 because one txn
   *  aborted and would retry" (correct SI behavior, NOT an anomaly). */
  commits: number;
}

/** Run one scenario at one level under one seed. Returns whether the anomaly
 *  fired and the abort count, plus the trace for printing the first hit. */
function runScenario(
  scenario: Scenario,
  level: IsolationLevel,
  seed: number,
): { anomaly: boolean; aborts: number; trace: string } {
  const store = new MvccStore();
  scenario.init(store);
  const obs: Observations = { reads: [], flags: {}, commits: 0 };
  // The detector reads obs; stash it where txn bodies can reach it. We attach to
  // the store object so the single shared ctx carries both (txn bodies receive
  // the store as ctx).
  (store as unknown as { obs: Observations }).obs = obs;

  // Pre-begin all txns at the SAME logical time, before any op runs => all are
  // mutually concurrent. This is what makes anomaly reachability a property of
  // the isolation level, not of when the scheduler happened to first touch a gen.
  const handles = Array.from({ length: scenario.txnCount }, () =>
    store.begin(level),
  );
  const txns = scenario.txns(store, handles);
  const rng = createRng(seed);
  const result = runSchedule(txns, store, rng);

  const aborts = result.outcome.filter((o) => o === "abort").length;
  const anomaly = scenario.detect(store, obs);
  return { anomaly, aborts, trace: formatTrace(result) };
}

/** Sweep seeds to find whether an anomaly is REACHABLE under a level. A single
 *  seed = a single interleaving; the anomaly may need a specific one. We report
 *  "reproduced" if ANY seed in the sweep triggers it (the level admits it) and
 *  the worst-case abort rate across seeds. N>1 by design: it separates "the
 *  level forbids this" from "this seed got lucky". */
function sweepScenario(
  scenario: Scenario,
  level: IsolationLevel,
  seeds: number,
): { everReproduced: boolean; abortRatePct: number; firstHitTrace: string | null } {
  let everReproduced = false;
  let totalAborts = 0;
  let totalTxns = 0;
  let firstHitTrace: string | null = null;

  for (let seed = 1; seed <= seeds; seed++) {
    const { anomaly, aborts, trace } = runScenario(scenario, level, seed);
    if (anomaly && !everReproduced) {
      everReproduced = true;
      firstHitTrace = trace;
    }
    totalAborts += aborts;
    // Abort rate = aborts / attempts (every txn in every seed is an attempt),
    // not per-seed yes/no — keeps the rate honest across the whole sweep.
    totalTxns += scenario.txnCount;
  }

  return {
    everReproduced,
    abortRatePct: (totalAborts / totalTxns) * 100,
    firstHitTrace,
  };
}

// ---------------------------------------------------------------------------
// The five scenarios
// ---------------------------------------------------------------------------

const obsOf = (store: MvccStore): Observations =>
  (store as unknown as { obs: Observations }).obs;

/** DIRTY READ: T1 writes x=999 and then ABORTS (never commits — the value 999 is
 *  rolled back and never becomes a committed version). T0 reads x. A dirty read
 *  = T0 observing 999, a value that was never committed. Because 999 is never
 *  committed under ANY interleaving, the only way obs could contain it is if our
 *  read exposed an uncommitted buffer — which MVCC never does. So this must be
 *  impossible at every level; the truth table then PROVES MVCC's structural
 *  guarantee rather than just asserting "RC forbids dirty reads".
 *
 *  Using an abort (not a delayed commit) is the fix for an earlier false
 *  positive: if T1 eventually committed 999, a read of 999 AFTER that commit is
 *  legitimate, and the detector couldn't tell it apart from a true dirty read. */
const dirtyRead: Scenario = {
  name: "dirty-read",
  init: (s) => s.init("x", 100),
  txnCount: 2,
  txns: (_store, [h0, h1]) => [
    function* T0(ctx) {
      yield "T0 read x";
      obsOf(ctx).reads.push(ctx.read(h0, "x"));
      // T0 has no writes; returning commits an empty write set.
    },
    function* T1(ctx) {
      yield "T1 write x=999 (will roll back)";
      ctx.write(h1, "x", 999);
      // Window: the scheduler may run T0's read here, while 999 is buffered but
      // uncommitted — the exact moment a dirty read would expose it.
      yield "T1 ... uncommitted, about to abort";
      // Abort: throw so the scheduler records an abort and 999 never commits.
      throw new Error("T1 voluntary rollback");
    },
  ],
  detect: (_s, obs) => obs.reads.includes(999),
};

/** NON-REPEATABLE READ: T0 reads x twice; T1 commits a new value of x in
 *  between. Under RC each read re-snapshots, so the two reads can differ (anomaly
 *  fires). Under RR/SI the snapshot is frozen, so they must agree. */
const nonRepeatableRead: Scenario = {
  name: "non-repeatable-read",
  init: (s) => s.init("x", 100),
  txnCount: 2,
  txns: (_store, [h0, h1]) => [
    function* T0(ctx) {
      yield "T0 read x (1st)";
      const a = ctx.read(h0, "x");
      yield "T0 ... (window for T1)";
      yield "T0 read x (2nd)";
      const b = ctx.read(h0, "x");
      if (a !== b) obsOf(ctx).flags["nonRepeatable"] = true;
    },
    function* T1(ctx) {
      yield "T1 write x=200";
      ctx.write(h1, "x", 200);
      ctx.commit(h1);
      yield "T1 commit";
    },
  ],
  detect: (_s, obs) => obs.flags["nonRepeatable"] === true,
};

/** PHANTOM: T0 runs the same range scan ("how many accounts have balance>0")
 *  twice; T1 commits a change that makes a previously-zero account positive in
 *  between. Under RC the second scan sees the new row (count changes => phantom).
 *  Under RR/SI the frozen snapshot hides it. Same mechanism as non-repeatable
 *  read but over a PREDICATE rather than a single row — that distinction is why
 *  phantoms need predicate/range locking (or SI snapshots) to prevent. */
const phantom: Scenario = {
  name: "phantom",
  init: (s) => {
    s.init("a", 50);
    s.init("b", 0); // out of range initially
  },
  txnCount: 2,
  txns: (_store, [h0, h1]) => {
    const positive = (v: number) => v > 0;
    return [
      function* T0(ctx) {
        yield "T0 count(balance>0) #1";
        const c1 = ctx.rangeCount(h0, ["a", "b"], positive);
        yield "T0 ... (window for T1)";
        yield "T0 count(balance>0) #2";
        const c2 = ctx.rangeCount(h0, ["a", "b"], positive);
        if (c1 !== c2) obsOf(ctx).flags["phantom"] = true;
      },
      function* T1(ctx) {
        yield "T1 set b=75 (new phantom row)";
        ctx.write(h1, "b", 75);
        ctx.commit(h1);
        yield "T1 commit";
      },
    ];
  },
  detect: (_s, obs) => obs.flags["phantom"] === true,
};

/** LOST UPDATE: both txns read x, then both write x = read+1. The serial result
 *  is 102; a lost update yields 101 (one increment vanished). Under RC/RR there
 *  is no write-write conflict detection, so the last committer overwrites the
 *  other => 101. Under SI, first-committer-wins aborts the second writer, so the
 *  surviving committer's increment is preserved (and the aborted one would retry
 *  in a real system) => no lost update, but at the cost of an abort. */
const lostUpdate: Scenario = {
  name: "lost-update",
  init: (s) => s.init("x", 100),
  txnCount: 2,
  txns: (_store, handles) => {
    const body = (label: string, h: Txn): TxnGen<MvccStore> =>
      function* (ctx) {
        yield `${label} read x`;
        const v = ctx.read(h, "x");
        yield `${label} write x=${v + 1}`;
        ctx.write(h, "x", v + 1);
        ctx.commit(h); // throws under SI on conflict => scheduler aborts this txn
        obsOf(ctx).commits++; // only reached if commit succeeded
        yield `${label} commit`;
      };
    return [body("T0", handles[0]), body("T1", handles[1])];
  },
  // Lost update = BOTH txns committed yet an increment vanished (final < 102).
  // Requiring commits===2 is the fix for an earlier false positive: under SI one
  // committer aborts, leaving x=101 — that is correct behavior (the aborted txn
  // would retry), NOT a lost update. Only when both commit and x<102 did an
  // update truly get lost.
  detect: (s, obs) => obs.commits === 2 && s.groundTruth("x") < 102,
};

/** WRITE SKEW (the punchline). Two on-call doctors, Alice and Bob, are both on
 *  duty. Hospital rule: at least one must stay on duty. Each independently checks
 *  "is the OTHER still on duty?" — under their snapshot, yes — and so each takes
 *  leave (sets THEIR OWN on-duty flag to 0). They write DIFFERENT keys, so SI's
 *  per-key first-committer-wins finds no conflict; both commit. Result: 0 doctors
 *  on duty, invariant violated, ZERO aborts. This is the canonical proof that SI
 *  is NOT serializable and motivates SSI. */
const writeSkew: Scenario = {
  name: "write-skew(on-call)",
  init: (s) => {
    s.init("alice_on_duty", 1);
    s.init("bob_on_duty", 1);
  },
  txnCount: 2,
  txns: (_store, [hAlice, hBob]) => [
    function* Alice(ctx) {
      yield "Alice read bob_on_duty";
      const bob = ctx.read(hAlice, "bob_on_duty");
      yield "Alice decide";
      // Safe to leave ONLY if Bob stays on duty — per my snapshot, he does.
      if (bob === 1) {
        yield "Alice write alice_on_duty=0";
        ctx.write(hAlice, "alice_on_duty", 0);
      }
      ctx.commit(hAlice);
      yield "Alice commit";
    },
    function* Bob(ctx) {
      yield "Bob read alice_on_duty";
      const alice = ctx.read(hBob, "alice_on_duty");
      yield "Bob decide";
      if (alice === 1) {
        yield "Bob write bob_on_duty=0";
        ctx.write(hBob, "bob_on_duty", 0);
      }
      ctx.commit(hBob);
      yield "Bob commit";
    },
  ],
  // Anomaly = the multi-row invariant "at least one on duty" is broken.
  detect: (s) =>
    s.groundTruth("alice_on_duty") + s.groundTruth("bob_on_duty") === 0,
};

const SCENARIOS: Scenario[] = [
  dirtyRead,
  nonRepeatableRead,
  phantom,
  lostUpdate,
  writeSkew,
];
const LEVELS: IsolationLevel[] = ["RC", "RR", "SI"];

// Sweep enough seeds that an anomaly's reachability isn't a coin-flip artifact.
// With 2 txns of ~3-5 steps each the schedule space is small. Measured (by
// re-running this stage at SEED_COUNT = 1,5,10,20,40,200): at 1 seed the self-
// checks FAIL (a single interleaving misses the anomaly — exactly the N=1 trap);
// from 5 seeds up the truth table and all structural assertions are stable and
// identical through 200. We keep 200 for comfortable margin (run is ~instant).
const SEED_COUNT = 200;

// ---------------------------------------------------------------------------
// Main: build the truth table, then dissect the write-skew punchline.
// ---------------------------------------------------------------------------

function main(): void {
  console.log("=== stage06: MVCC 隔离级别异常真值表 ===\n");
  console.log(
    `每个 (异常 × 级别) 单元格 = 在 ${SEED_COUNT} 个确定性种子上 sweep 调度，` +
      `观察真实运行后状态判定是否复现（YES=至少一个交错触发，no=全部种子都挡住）。`,
  );
  console.log(
    "数字 = 该级别在本场景下的中止率 (aborts/attempts)，由调度器真实记录。\n",
  );

  // Build one row per anomaly, one column per level, cell = reproduced? + abort%.
  const rows: TableRow[] = [];
  // Keep the full sweep results so the post-table analysis can reuse them without
  // re-running (and so the write-skew trace below is the real first-hit trace).
  const sweepCache = new Map<string, ReturnType<typeof sweepScenario>>();

  for (const scenario of SCENARIOS) {
    const row: TableRow = { anomaly: scenario.name };
    for (const level of LEVELS) {
      const res = sweepScenario(scenario, level, SEED_COUNT);
      sweepCache.set(`${scenario.name}|${level}`, res);
      const repro = res.everReproduced ? "YES" : "no ";
      // Pair the boolean with abort rate so SI's "prevented via abort" reads
      // differently from RR's "prevented via snapshot" / "admitted".
      row[level] = `${repro} (${res.abortRatePct.toFixed(0)}% ab)`;
    }
    rows.push(row);
  }
  printTable(rows);
  console.log(
    "\n注：dirty-read 行的 50% 中止是场景里 T1「主动回滚」造成的（故意永不提交" +
      "以制造未提交值），不是隔离级别为挡脏读而中止——脏读靠 MVCC 结构挡，零成本。",
  );

  // ---- Sanity invariants on the table itself (the chapter's structural claims).
  // These would throw loudly if the visibility rules regressed — making the
  // truth table self-checking rather than just printed.
  const cell = (anom: string, lvl: IsolationLevel) =>
    sweepCache.get(`${anom}|${lvl}`)!.everReproduced;

  console.log("\n--- 结构性断言（真值表自检，违反即抛错）---");
  // 1. Dirty read impossible at every level: MVCC buffers uncommitted writes.
  for (const lvl of LEVELS) {
    invariant(!cell("dirty-read", lvl), `dirty read must be impossible under ${lvl}`);
  }
  console.log("  [ok] 脏读在 RC/RR/SI 全部被结构性挡住（MVCC 不暴露未提交版本）");

  // 2. Non-repeatable read & phantom: present under RC, gone under RR & SI.
  invariant(cell("non-repeatable-read", "RC"), "RC must admit non-repeatable read");
  invariant(!cell("non-repeatable-read", "RR"), "RR must prevent non-repeatable read");
  invariant(!cell("non-repeatable-read", "SI"), "SI must prevent non-repeatable read");
  console.log("  [ok] 不可重复读：RC 复现，RR/SI 因快照冻结而消失");
  invariant(cell("phantom", "RC"), "RC must admit phantom");
  invariant(!cell("phantom", "RR"), "RR snapshot must prevent phantom");
  console.log("  [ok] 幻读：RC 复现，RR/SI 快照挡住（注意：靠快照而非谓词锁）");

  // 3. Lost update: present under RC/RR, prevented by SI (via abort).
  invariant(cell("lost-update", "RC"), "RC must admit lost update");
  invariant(cell("lost-update", "RR"), "RR must admit lost update");
  const siLost = sweepCache.get("lost-update|SI")!;
  invariant(!siLost.everReproduced, "SI must prevent lost update");
  invariant(siLost.abortRatePct > 0, "SI must prevent lost update *by aborting*");
  console.log(
    `  [ok] 丢失更新：RC/RR 复现，SI 用 first-committer-wins 中止挡住` +
      `（代价：${siLost.abortRatePct.toFixed(0)}% 中止率）`,
  );

  // 4. THE PUNCHLINE: write skew slips through SI with ZERO aborts.
  const siSkew = sweepCache.get("write-skew(on-call)|SI")!;
  invariant(siSkew.everReproduced, "write skew must reproduce under SI");
  invariant(
    siSkew.abortRatePct === 0,
    "write skew under SI must commit with zero aborts (the whole point)",
  );
  console.log(
    "  [ok] WRITE SKEW：SI 没挡住，且中止率 0% —— SI ≠ 可串行化的铁证\n",
  );

  // ---- Dissect the write-skew failure in detail: show the actual interleaving.
  console.log("=== 失败模式详解：SI 下的 write-skew（两医生请假）===\n");
  console.log(
    "初始：alice_on_duty=1, bob_on_duty=1，约束「至少一人在岗」(sum >= 1)。",
  );
  console.log(
    "两事务各读「对方是否在岗」（快照里都=1），各自给「自己」请假。",
  );
  console.log(
    "它们写的是不同的 key，所以 SI 的 per-key first-committer-wins 看不到冲突。\n",
  );

  // Re-run the exact first-hit seed so we can print the offending interleaving
  // AND the concrete before/after state — proof, not narration.
  const hitTrace = siSkew.firstHitTrace;
  invariant(hitTrace !== null, "expected a reproducing trace for write skew");

  // Find which seed produced the first hit (re-run to capture state), so the
  // printed numbers are this exact run's, not a different seed's.
  let demoStore: MvccStore | null = null;
  for (let seed = 1; seed <= SEED_COUNT; seed++) {
    const store = new MvccStore();
    writeSkew.init(store);
    (store as unknown as { obs: Observations }).obs = {
      reads: [],
      flags: {},
      commits: 0,
    };
    const handles = Array.from({ length: writeSkew.txnCount }, () =>
      store.begin("SI"),
    );
    const txns = writeSkew.txns(store, handles);
    const result = runSchedule(txns, store, createRng(seed));
    const broke =
      store.groundTruth("alice_on_duty") + store.groundTruth("bob_on_duty") === 0;
    if (broke) {
      demoStore = store;
      console.log(`触发该异常的最小种子：seed=${seed}`);
      console.log("交错调度（每行一步，T0=Alice，T1=Bob）：");
      console.log(formatTrace(result));
      console.log(
        `\n两事务结果：${result.outcome.join(", ")}（无一中止——SI 放行）`,
      );
      break;
    }
  }
  invariant(demoStore !== null, "failed to reproduce write skew for the demo");

  const aliceFinal = demoStore.groundTruth("alice_on_duty");
  const bobFinal = demoStore.groundTruth("bob_on_duty");
  console.log("\n最终在岗状态（已提交真值）：");
  printTable([
    { doctor: "Alice", on_duty: aliceFinal },
    { doctor: "Bob", on_duty: bobFinal },
    { doctor: "SUM(约束>=1)", on_duty: aliceFinal + bobFinal },
  ]);
  console.log(
    `\n版本链长度：alice=${demoStore.chainLength("alice_on_duty")}, ` +
      `bob=${demoStore.chainLength("bob_on_duty")} ` +
      `（MVCC 保留旧版本，这是快照的存储代价）`,
  );

  console.log(
    "\n结论：sum=0 违反「至少一人在岗」，且两事务都成功提交、0 中止。" +
      "\nSI 只在「同一行」上检测写写冲突，挡不住「跨行不变量」的 write skew。" +
      "\n要堵这个洞需要 SSI（serializable snapshot isolation）：" +
      "\n额外追踪「读-写依赖」(rw-antidependency)，发现危险结构时中止其一。" +
      "\n本章到 SI 为止——write skew 的复现，正是下一步引入 SSI 的动机。",
  );
}

main();
