// stage09-computability.ts — The limits of computation, made executable:
//   (1) a DFA / minimal regex engine that accepts or rejects strings,
//   (2) a tiny Turing-machine simulator running a real program (unary increment),
//   (3) the halting problem's diagonal argument as a CONCRETE code contradiction,
//   (4) an NP reduction SAT → Subset-Sum that PROVABLY preserves the answer,
//   (5) the failure mode: brute force is fine on n=20 and hopeless on n=60.
//
// WHY this is one stage and not five files: computability and complexity are one story —
//   what a machine CAN decide (DFA/TM), what NO machine can decide (halting), and what is
//   decidable-but-intractable (NP). A builder who feels these three boundaries stops asking
//   the LLM to "just brute force it" on inputs where 2^n is the wall.
//
// HONEST-NUMBER NOTE: every count below (TM steps, DFA transitions, SAT/Subset-Sum
//   assignments enumerated, wall-clock of the brute-force search) is measured by the code
//   on this run, not narrated. The brute-force timing is real wall-clock; the n=60
//   projection is explicitly marked (est.) because we refuse to actually run 2^60 steps.
//
// CONTRACT: reuse core/plot.js for the P-vs-exponential growth chart. No other stage files
//   are imported (importing a stageNN runs its main()).

import { asciiBar } from "./core/plot.js";

// ─────────────────────────────────────────────────────────────────────────────
// (1) DFA — a deterministic finite automaton. The minimal "regex engine": a regex
//     without backreferences IS a DFA. We hand-build one for the language
//     "binary strings with an EVEN number of 1s" (a classic regular language).
//
// WHY a DFA and not a regex literal: the point is that recognition needs only a fixed
//   amount of memory (here: 2 states). The machine never looks back; it cannot count
//   past its state set. That bounded memory is exactly why regular languages cannot match
//   "n a's followed by n b's" — there is no state to remember n. Knowing this stops you
//   from reaching for a regex to parse nested brackets.
// ─────────────────────────────────────────────────────────────────────────────

interface Dfa {
  readonly states: readonly string[];
  readonly start: string;
  readonly accept: ReadonlySet<string>;
  // transition[state][symbol] -> next state. Total function over the alphabet.
  readonly delta: Readonly<Record<string, Readonly<Record<string, string>>>>;
}

/** DFA for "even number of 1s". State = parity of 1s seen so far. */
const EVEN_ONES_DFA: Dfa = {
  states: ["even", "odd"],
  start: "even", // zero 1s is even
  accept: new Set(["even"]),
  delta: {
    even: { "0": "even", "1": "odd" }, // a 1 flips parity
    odd: { "0": "odd", "1": "even" },
  },
};

/**
 * Run a DFA on an input string. Returns whether it accepts plus the number of
 * transitions taken (= input length for a total DFA — that linear, single-pass cost
 * is the whole selling point of regular recognition).
 * FAILURE MODE: a symbol with no transition entry throws — a DFA's delta must be TOTAL.
 *   We surface that rather than silently treating "unknown symbol" as reject.
 */
function runDfa(dfa: Dfa, input: string): { accepted: boolean; steps: number } {
  let state = dfa.start;
  let steps = 0;
  for (const symbol of input) {
    const next = dfa.delta[state]?.[symbol];
    if (next === undefined) {
      throw new Error(`[dfa] no transition for state=${state} symbol=${symbol}`);
    }
    state = next;
    steps++;
  }
  return { accepted: dfa.accept.has(state), steps };
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) Turing machine — finite control + an UNBOUNDED tape. The one thing the DFA
//     lacked (rewritable memory) is exactly what makes the TM Turing-complete.
//
// Program below: unary increment. Input "111" (= 3) on the tape; the machine walks to
//   the blank past the number and writes one more 1, leaving "1111" (= 4). Trivial as
//   arithmetic, but it exercises: head movement, tape writes, halting via a final state.
// ─────────────────────────────────────────────────────────────────────────────

type Move = "L" | "R" | "N";
interface TmRule {
  write: string;
  move: Move;
  next: string;
}
interface TuringMachine {
  readonly blank: string;
  readonly start: string;
  readonly halt: string;
  // rules[state][readSymbol] -> action. Missing entry = halt-by-stuck (see runTm).
  readonly rules: Readonly<Record<string, Readonly<Record<string, TmRule>>>>;
}

/** Unary increment: scan right over 1s to the first blank, write 1, halt. */
const INCREMENT_TM: TuringMachine = {
  blank: "_",
  start: "scan",
  halt: "done",
  rules: {
    scan: {
      "1": { write: "1", move: "R", next: "scan" }, // keep walking over the number
      _: { write: "1", move: "N", next: "done" }, // hit the end -> append a 1, halt
    },
  },
};

/**
 * Simulate a TM with a step budget. Returns the final tape, whether it halted, and the
 * step count. The budget is NOT a halting decider — it is a resource bound (see part 3:
 * "budget exhausted" must never be reported as "this machine does not halt").
 * Tape grows on demand: reading past either end materializes a blank.
 */
function runTm(
  tm: TuringMachine,
  input: string,
  maxSteps: number,
): { tape: string; halted: boolean; steps: number } {
  const tape = new Map<number, string>();
  [...input].forEach((symbol, i) => tape.set(i, symbol));
  const read = (pos: number): string => tape.get(pos) ?? tm.blank;

  let head = 0;
  let state = tm.start;
  let steps = 0;
  while (state !== tm.halt && steps < maxSteps) {
    const rule = tm.rules[state]?.[read(head)];
    if (rule === undefined) break; // stuck = halted with no applicable rule
    tape.set(head, rule.write);
    if (rule.move === "L") head--;
    else if (rule.move === "R") head++;
    state = rule.next;
    steps++;
  }
  const positions = [...tape.keys()];
  const lo = Math.min(0, ...positions);
  const hi = Math.max(0, ...positions);
  let rendered = "";
  for (let p = lo; p <= hi; p++) rendered += read(p);
  return { tape: rendered.replace(/_+$/, "") || tm.blank, halted: state === tm.halt, steps };
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) The halting problem, as a contradiction you can EXECUTE.
//
// The classic proof: suppose a total decider halts(fn, input) exists. Build a "spite"
//   program that does the opposite of what halts predicts about it running on itself.
//   Feed spite to halts(spite, spite) and you get a paradox. We can't write a real
//   halts() (it provably can't exist), so we MOCK one with any guess and show the spite
//   program defeats it — for EVERY possible guess. That is the diagonalization made literal.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A pretend halting oracle. `guess` stands in for whatever the (impossible) decider
 * would output. We will show that no value of `guess` survives the diagonal program.
 */
type HaltingOracle = (programSrc: string, input: string) => boolean;

/**
 * The diagonal "spite" program: it asks the oracle what IT will do on `selfSrc`, then
 * does the opposite. If the oracle says "halts", spite loops forever; if "loops", spite
 * halts. So the oracle's verdict about spite is wrong by construction.
 * Returns the spite program's ACTUAL behavior ("halts" | "loops") so we can compare it
 * against the oracle's prediction and witness the mismatch.
 */
function runSpiteAgainst(oracle: HaltingOracle, selfSrc: string): "halts" | "loops" {
  const predictionHalts = oracle(selfSrc, selfSrc); // oracle predicts spite on itself
  // Do the OPPOSITE of the prediction. (We return a label instead of actually looping
  // forever — an infinite loop would just hang the demo and prove nothing extra.)
  return predictionHalts ? "loops" : "halts";
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) NP reduction: SAT → Subset-Sum, with the answer preserved.
//
// A reduction maps instance A into instance B so that A is a YES-instance iff B is. It is
//   how NP-completeness propagates: solve Subset-Sum and you've solved SAT. We build the
//   textbook positional encoding and VERIFY (by brute-forcing BOTH problems) that the
//   reduction never flips the answer — across many random formulas, SAT-sat ⇔ subset exists.
//
// Encoding (one decimal digit column per variable, plus one per clause):
//   - per variable x_i: number for x_i=true and a number for x_i=false, each with a 1 in
//     the variable column (forces exactly one polarity chosen) and a 1 in every clause the
//     literal satisfies.
//   - per clause: up to 2 "slack" numbers so a satisfied clause's column can reach target.
//   - target: 1 in every variable column, and `clauseCap` in every clause column.
// A subset summing to target ⇔ a consistent assignment that satisfies every clause.
// ─────────────────────────────────────────────────────────────────────────────

// A literal is a signed variable index: +i means x_i, -i means ¬x_i (1-based).
type Clause = readonly number[];
type Cnf = readonly Clause[];

interface SubsetSum {
  readonly numbers: readonly number[];
  readonly target: number;
  // Provenance labels, parallel to `numbers`, so the demo can show WHICH choice each
  // selected number represents. Not used by the solver; purely for honest reporting.
  readonly labels: readonly string[];
}

// clauseCap: max literals per clause we encode slack for. With 3-SAT, a clause column gets
// contributions from 1..3 satisfying literals; we pad with up to (cap-1) slack units so any
// nonzero count can be topped up to `cap`. Using cap = max clause width keeps columns clean.
function reduceSatToSubsetSum(cnf: Cnf, numVars: number): SubsetSum {
  const numClauses = cnf.length;
  const clauseCap = Math.max(1, ...cnf.map((c) => c.length));
  const numbers: number[] = [];
  const labels: string[] = [];

  // Digit layout (base 10, big enough since each column's max sum < 10 for our toy sizes):
  //   columns 0..numVars-1         -> variable-consistency columns
  //   columns numVars..numVars+numClauses-1 -> clause-satisfaction columns
  const totalCols = numVars + numClauses;
  const colValue = (col: number): number => 10 ** (totalCols - 1 - col);

  for (let v = 1; v <= numVars; v++) {
    // x_v = true and x_v = false: each owns the variable column (forces one polarity).
    for (const polarity of [true, false] as const) {
      let value = colValue(v - 1); // variable column
      cnf.forEach((clause, ci) => {
        const satisfies = clause.some((lit) =>
          polarity ? lit === v : lit === -v,
        );
        if (satisfies) value += colValue(numVars + ci);
      });
      numbers.push(value);
      labels.push(`x${v}=${polarity ? "T" : "F"}`);
    }
  }
  // Slack numbers per clause: let a satisfied clause's column reach exactly clauseCap.
  for (let ci = 0; ci < numClauses; ci++) {
    for (let s = 0; s < clauseCap - 1; s++) {
      numbers.push(colValue(numVars + ci));
      labels.push(`slack[c${ci + 1}]`);
    }
  }

  // Target: 1 in each variable column (exactly one polarity), clauseCap in each clause
  // column (at least one real literal + slack fills the rest).
  let target = 0;
  for (let v = 0; v < numVars; v++) target += colValue(v);
  for (let ci = 0; ci < numClauses; ci++) target += clauseCap * colValue(numVars + ci);

  return { numbers, target, labels };
}

/**
 * Brute-force SAT: try all 2^numVars assignments, return whether any satisfies the CNF
 * plus the count of assignments examined. The count is the honest measure of work.
 * INVARIANT: this is the ground truth we check the reduction against — slow but correct.
 */
function bruteForceSat(cnf: Cnf, numVars: number): { sat: boolean; tried: number } {
  let tried = 0;
  for (let bits = 0; bits < 1 << numVars; bits++) {
    tried++;
    const assign = (v: number): boolean => (bits & (1 << (v - 1))) !== 0;
    const allClausesTrue = cnf.every((clause) =>
      clause.some((lit) => (lit > 0 ? assign(lit) : !assign(-lit))),
    );
    if (allClausesTrue) return { sat: true, tried };
  }
  return { sat: false, tried };
}

/**
 * Brute-force Subset-Sum: try all 2^n subsets, return whether any hits the target plus the
 * count examined. Same exponential shape as SAT — which is the point of the reduction.
 */
function bruteForceSubsetSum(p: SubsetSum): { hit: boolean; tried: number } {
  const n = p.numbers.length;
  // GUARD: `1 << n` silently overflows the 32-bit signed int for n >= 31, which would make
  //   the loop search the WRONG (truncated) subset space and report bogus "no subset". The
  //   reduction inflates a SAT instance into many numbers, so this is a live hazard — fail
  //   loud rather than verify against a broken search. Callers keep instances small enough.
  if (n > 24) throw new Error(`[subset-sum] ${n} numbers too large to brute-force (2^n blowup)`);
  let tried = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    tried++;
    let sum = 0;
    for (let i = 0; i < n; i++) if (mask & (1 << i)) sum += p.numbers[i];
    if (sum === p.target) return { hit: true, tried };
  }
  return { hit: false, tried };
}

// A tiny seeded LCG, local to this part so part (4) stays self-contained without coupling
// to the book-wide Rng stream (we only need a handful of bits for random formulas, and the
// reduction's correctness must not depend on which global draws happened before).
function makeLcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Generate a random k-SAT-ish CNF over `numVars` with `numClauses` clauses. */
function randomCnf(
  rng: () => number,
  numVars: number,
  numClauses: number,
  litsPerClause: number,
): Cnf {
  const clauses: Clause[] = [];
  for (let c = 0; c < numClauses; c++) {
    const clause: number[] = [];
    while (clause.length < litsPerClause) {
      const v = 1 + Math.floor(rng() * numVars);
      const lit = rng() < 0.5 ? v : -v;
      if (!clause.includes(lit) && !clause.includes(-lit)) clause.push(lit);
    }
    clauses.push(clause);
  }
  return clauses;
}

function main(): void {
  console.log("=== Stage 09 · 可计算性与复杂度 ===\n");

  // ── (1) DFA ────────────────────────────────────────────────────────────────
  console.log("【1】DFA：识别『1 的个数为偶数』的二进制串（有限内存即可判定）");
  const dfaCases = ["", "11", "101", "1010", "111"];
  for (const s of dfaCases) {
    const { accepted, steps } = runDfa(EVEN_ONES_DFA, s);
    const ones = [...s].filter((c) => c === "1").length;
    console.log(
      `  "${s.padEnd(4)}" (${ones} 个 1) → ${accepted ? "接受" : "拒绝"}  (${steps} 次转移)`,
    );
  }
  try {
    runDfa(EVEN_ONES_DFA, "12"); // '2' is outside the alphabet
  } catch (e) {
    console.log(`  失败模式：喂入字母表外符号 '2' → ${(e as Error).message}`);
    console.log("  → DFA 的转移函数必须是全函数；缺转移不能当作『拒绝』静默吞掉。");
  }

  // ── (2) Turing machine ───────────────────────────────────────────────────────
  console.log("\n【2】图灵机：一元加一（111 → 1111），有限控制 + 无界纸带");
  const before = "111";
  const inc = runTm(INCREMENT_TM, before, 100);
  console.log(
    `  输入 "${before}" (=${before.length}) → 输出 "${inc.tape}" (=${inc.tape.length})`,
  );
  console.log(`  停机=${inc.halted}，走了 ${inc.steps} 步（扫到末尾的空格再写一个 1）`);
  // Failure mode demo for the budget: same machine, an artificially tiny budget.
  const starved = runTm(INCREMENT_TM, "11111111", 3);
  console.log(
    `  失败模式：预算只给 3 步 → 停机=${starved.halted}，走了 ${starved.steps} 步、纸带 "${starved.tape}"`,
  );
  console.log("  → 预算耗尽是『资源不够』，不是『不停机』；二者不可混为一谈（见【3】）。");

  // ── (3) Halting problem — diagonal contradiction ─────────────────────────────
  console.log("\n【3】停机问题：对角线论证的代码化（假设的判定器必被反例击穿）");
  // Try several different "oracles" (always-halts, always-loops, a content-based guess).
  // For EACH, the spite program does the opposite of the prediction -> prediction wrong.
  const oracles: { name: string; oracle: HaltingOracle }[] = [
    { name: "总是猜『停机』", oracle: () => true },
    { name: "总是猜『不停机』", oracle: () => false },
    { name: "按源码长度猜", oracle: (src) => src.length % 2 === 0 },
  ];
  for (const { name, oracle } of oracles) {
    const src = `spite<${name}>`; // stands in for the spite program's own source
    const predicted = oracle(src, src) ? "停机" : "不停机";
    const actual = runSpiteAgainst(oracle, src) === "halts" ? "停机" : "不停机";
    const wrong = predicted !== actual;
    console.log(
      `  判定器=${name}：预测 spite→${predicted}，实际 spite→${actual}  [${wrong ? "判错 ✓矛盾" : "判对?!"}]`,
    );
  }
  console.log("  → 对任意判定器都能造出『故意唱反调』的程序使其判错 ⇒ 通用停机判定器不存在。");

  // ── (4) NP reduction: SAT → Subset-Sum, answer-preserving ────────────────────
  console.log("\n【4】NP 归约：SAT → 子集和（暴力解两边，验证答案被保持）");
  const rng = makeLcg(20260622);
  let agree = 0;
  let total = 0;
  let satCount = 0;
  let mismatchExample: string | null = null;
  for (let trial = 0; trial < 40; trial++) {
    // Fix 3 vars: the reduction inflates instances to (2*vars + 2*clauses) numbers, and the
    // subset-sum side is itself brute-forced (2^numbers). We MUST keep numbers small enough
    // that the verifier's own search is valid (see bruteForceSubsetSum's overflow guard).
    // 3 vars + up to 9 clauses -> at most 6 + 18 = 24 numbers, right at the safe ceiling.
    const numVars = 3;
    // Over-constrained formulas (many clauses, few vars) make some instances UNSAT, so the
    // reduction is tested in BOTH directions: YES⇒subset exists AND NO⇒no subset exists.
    const numClauses = 4 + (trial % 6); // 4..9 clauses; over-constrained ones turn UNSAT
    const cnf = randomCnf(rng, numVars, numClauses, 3);
    const satResult = bruteForceSat(cnf, numVars);
    const reduced = reduceSatToSubsetSum(cnf, numVars);
    const ssResult = bruteForceSubsetSum(reduced);
    total++;
    if (satResult.sat) satCount++;
    if (satResult.sat === ssResult.hit) agree++;
    else if (mismatchExample === null) {
      mismatchExample = `SAT=${satResult.sat} 但 SubsetSum=${ssResult.hit} (vars=${numVars}, clauses=${numClauses})`;
    }
  }
  console.log(`  随机跑 ${total} 个公式：可满足 ${satCount} 个 / 不可满足 ${total - satCount} 个`);
  console.log(
    `  归约答案一致 (SAT ⇔ 子集和) = ${agree}/${total}  ${agree === total ? "✓ 归约保持答案" : "✗ 反例: " + mismatchExample}`,
  );
  // Show one concrete instance end-to-end so the encoding is not a black box.
  const demoCnf: Cnf = [
    [1, 2, -3],
    [-1, 2, 3],
    [1, -2, 3],
  ];
  const demoVars = 3;
  const demoSat = bruteForceSat(demoCnf, demoVars);
  const demoSs = reduceSatToSubsetSum(demoCnf, demoVars);
  const demoSsHit = bruteForceSubsetSum(demoSs);
  console.log(
    `  示例公式 (x1∨x2∨¬x3)∧(¬x1∨x2∨x3)∧(x1∨¬x2∨x3): SAT=${demoSat.sat}`,
  );
  console.log(
    `    → 子集和实例：${demoSs.numbers.length} 个数，目标 ${demoSs.target}，存在子集=${demoSsHit.hit}`,
  );

  // ── (5) Failure mode: "brute force is fine" breaks at the exponential wall ────
  console.log("\n【5】失败模式：以为暴力搜索对大实例可行（2^n 墙）");
  // Measure REAL wall-clock of brute-force SAT at growing n, then project the wall.
  const measured: { n: number; tried: number; ms: number }[] = [];
  for (const n of [16, 20, 22]) {
    // An UNSAT formula forces brute force to examine all 2^n assignments — worst case.
    // (x_i ∨ x_i) ∧ (¬x_i ∨ ¬x_i) for one variable is contradictory; we make a formula
    // with no satisfying assignment by demanding a var be both T and F.
    const cnf: Cnf = [[1], [-1]]; // contradiction on x1 -> UNSAT regardless of other vars
    const t0 = performance.now();
    const r = bruteForceSat(cnf, n); // n vars, but contradiction => scans all 2^n
    const ms = performance.now() - t0;
    measured.push({ n, tried: r.tried, ms });
    console.log(`  n=${n}: 暴力枚举 ${r.tried.toLocaleString()} 个赋值，实测 ${ms.toFixed(1)} ms`);
  }
  // Project n=40 and n=60 from the measured per-assignment cost. Marked (est.) — we will
  // NOT actually run 2^60 (it would outlive the universe at this rate).
  const last = measured[measured.length - 1];
  const nsPerAssign = (last.ms * 1e6) / last.tried; // nanoseconds per assignment, measured
  const project = (n: number): string => {
    const secs = (nsPerAssign * 2 ** n) / 1e9;
    if (secs < 60) return `${secs.toFixed(1)} 秒`;
    if (secs < 3600) return `${(secs / 60).toFixed(1)} 分钟`;
    if (secs < 86400 * 365) return `${(secs / 86400).toFixed(1)} 天`;
    return `${(secs / (86400 * 365)).toExponential(2)} 年`;
  };
  console.log(`  按实测 ${nsPerAssign.toFixed(1)} ns/赋值 外推 (est.)：`);
  console.log(`    n=40 → ${project(40)}   n=50 → ${project(50)}   n=60 → ${project(60)}`);
  // Plot log10(seconds) per n, not raw 2^n: on a linear bar the largest value dwarfs the
  // rest into empty bars. The log makes the STRAIGHT-LINE growth of the exponent visible —
  // that ruler-straight rise IS the signature of exponential cost.
  console.log("  log10(预计秒数) 对 n（直线上升 = 指数代价的指纹）：");
  console.log(
    asciiBar(
      ["n=20", "n=30", "n=40", "n=50", "n=60"],
      [20, 30, 40, 50, 60].map((n) =>
        Math.max(0, Math.log10((nsPerAssign * 2 ** n) / 1e9)),
      ),
    ),
  );
  console.log("  → 每 +10 个变量，暴力代价 ×1024。n=20 一眨眼，n=60 等到宇宙凉。");
  console.log(
    "  注：以上为玩具规模的最坏情形外推；绝对秒数偏乐观（真求解器有剪枝/启发式），",
  );
  console.log("     可迁移的是『指数斜率』本身——这正是第 5 章组合爆炸的延续。");
}

main();
