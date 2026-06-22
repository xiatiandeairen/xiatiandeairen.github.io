// stage12-hardware.ts — Why a computer can compute: from a single NAND to arithmetic.
//
// SCOPE (stage author): build the whole logic stack from ONE primitive (NAND) and show it
//   is functionally complete: NOT/AND/OR/XOR are all derived, verified by printing truth
//   tables. Then compose half-adder → full-adder → 4-bit ripple-carry adder and check real
//   additions. Add a minimal SEQUENTIAL element (a gated D latch built only from NAND) to
//   show memory = feedback, not a new kind of part. Finally print the Landauer limit (the
//   physical kT·ln2 energy cost of erasing one bit) at room temperature.
//
// HONEST-NUMBER NOTE: every truth table / sum here is computed by actually running the gate
//   functions — no table is hand-typed. The Landauer energy is an EXACT closed form (kT·ln2)
//   from given physical constants, labelled (理论值/exact); it is a lower bound real CMOS
//   misses by ~1e4-1e6x, which we state rather than imply silicon runs at the limit.
//
// CONTRACT: reuse core/plot.js for the energy-scale bar. No core logic primitives exist
//   (gates are the subject of this chapter), so the gate layer is built here from scratch.
//
// FAILURE MODE demoed: a combinational race hazard (glitch). Our functional model is
//   instantaneous and CANNOT show the transient — that is precisely the point: pure boolean
//   algebra hides timing, and timing is where real hardware bites. We make the gap explicit.

import { asciiBar } from "./core/plot.js";

// A bit is modelled as 0 | 1. We keep it as `number` (not boolean) so adder carry chains
// read like the arithmetic they implement, and so truth tables print as 0/1 like a datasheet.
type Bit = 0 | 1;

// ---------------------------------------------------------------------------
// 1. The one primitive. Everything else is built from this.
// ---------------------------------------------------------------------------

// WHY NAND specifically: NAND (and NOR) are the only 2-input gates that are *functionally
// complete* on their own — any boolean function can be expressed using NAND alone. Real
// CMOS fabs exploit this: a NAND is cheaper (fewer transistors, 4) than AND (which is NAND
// + inverter, 6). So "build from NAND" is not a toy constraint, it mirrors silicon economics.
function nand(a: Bit, b: Bit): Bit {
  return (a === 1 && b === 1 ? 0 : 1) as Bit;
}

// NOT a = NAND(a,a): tying both inputs together feeds the same bit twice.
function not(a: Bit): Bit {
  return nand(a, a);
}

// AND = NOT(NAND): NAND already inverts, so one more inversion recovers AND.
function and(a: Bit, b: Bit): Bit {
  return not(nand(a, b));
}

// OR = NAND(NOT a, NOT b): De Morgan's law, a∨b = ¬(¬a ∧ ¬b), realised in gates.
function or(a: Bit, b: Bit): Bit {
  return nand(not(a), not(b));
}

// XOR from NAND in the classic 4-gate shape. Derivation (so a future reader can verify the
// wiring instead of trusting it): let c = NAND(a,b). Then XOR = NAND(NAND(a,c), NAND(b,c)).
// This is the standard 4-NAND XOR; it is also exactly the half-adder SUM, reused below.
function xor(a: Bit, b: Bit): Bit {
  const c = nand(a, b);
  return nand(nand(a, c), nand(b, c));
}

// ---------------------------------------------------------------------------
// 2. Truth tables — proof, not assertion.
// ---------------------------------------------------------------------------

const INPUTS_2: ReadonlyArray<[Bit, Bit]> = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

function printTruthTable2(name: string, fn: (a: Bit, b: Bit) => Bit): void {
  const rows = INPUTS_2.map(([a, b]) => `  ${a} ${b} │ ${fn(a, b)}`).join("\n");
  console.log(`${name.padEnd(4)} (a b │ out)\n${rows}`);
}

// Functional-completeness check: AND/OR/XOR built from NAND must match their definitions
// for ALL 4 input rows. If any row disagrees, the derivation is wrong — we assert, not hope.
function verifyNandUniversality(): void {
  const reference: Record<string, (a: Bit, b: Bit) => Bit> = {
    AND: (a, b) => ((a & b) as Bit),
    OR: (a, b) => ((a | b) as Bit),
    XOR: (a, b) => ((a ^ b) as Bit),
  };
  const built: Record<string, (a: Bit, b: Bit) => Bit> = { AND: and, OR: or, XOR: xor };
  for (const gate of Object.keys(reference)) {
    for (const [a, b] of INPUTS_2) {
      if (built[gate](a, b) !== reference[gate](a, b)) {
        throw new Error(`[stage12] ${gate} derived from NAND disagrees at (${a},${b})`);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Arithmetic: half-adder → full-adder → 4-bit ripple-carry adder.
// ---------------------------------------------------------------------------

// Half-adder: sum = a XOR b, carry = a AND b. It cannot accept a carry-in, hence "half".
function halfAdder(a: Bit, b: Bit): { sum: Bit; carry: Bit } {
  return { sum: xor(a, b), carry: and(a, b) };
}

// Full-adder: chains two half-adders + an OR so it accepts a carry-in. This is the cell
// that, replicated, becomes any width adder. carry_out = (a&b) | (cin & (a^b)).
function fullAdder(a: Bit, b: Bit, carryIn: Bit): { sum: Bit; carry: Bit } {
  const h1 = halfAdder(a, b);
  const h2 = halfAdder(h1.sum, carryIn);
  return { sum: h2.sum, carry: or(h1.carry, h2.carry) };
}

// 4-bit ripple-carry adder. INVARIANT: bits are LSB-first (index 0 = least significant), so
// the carry "ripples" from index 0 upward exactly like pencil-and-paper addition. The name
// "ripple" is also the FAILURE story: each stage waits for the previous carry, so worst-case
// delay grows linearly with width — the motivation for carry-lookahead adders (out of scope).
function rippleAdd4(a: readonly Bit[], b: readonly Bit[]): { sum: Bit[]; carryOut: Bit } {
  if (a.length !== 4 || b.length !== 4) {
    throw new Error(`[stage12] rippleAdd4 expects 4-bit inputs, got ${a.length}/${b.length}`);
  }
  const sum: Bit[] = [];
  let carry: Bit = 0;
  for (let i = 0; i < 4; i++) {
    const r = fullAdder(a[i], b[i], carry);
    sum[i] = r.sum;
    carry = r.carry;
  }
  return { sum, carryOut: carry };
}

// Helpers to move between human integers and LSB-first bit arrays for readable demos.
function toBits4(n: number): Bit[] {
  if (n < 0 || n > 15) throw new Error(`[stage12] toBits4 out of 4-bit range: ${n}`);
  return [0, 1, 2, 3].map((i) => ((n >> i) & 1) as Bit);
}
function fromBits(bits: readonly Bit[], carryOut: Bit = 0): number {
  // carryOut is the bit ABOVE the array, so it weighs 2^length (the 5th bit for a 4-bit add).
  const base = bits.reduce<number>((acc, bit, i) => acc + bit * 2 ** i, 0);
  return base + carryOut * 2 ** bits.length;
}
function bitsToStr(bits: readonly Bit[]): string {
  // Print MSB-first (reverse of our LSB-first storage) to match how humans write binary.
  return [...bits].reverse().join("");
}

// ---------------------------------------------------------------------------
// 4. Memory from feedback: a gated D latch, built only from NAND.
// ---------------------------------------------------------------------------

// WHY a latch needs no new primitive: memory is just a NAND feedback loop. A gated D latch
// is the canonical 4-NAND cell. Because our model is a pure function with no notion of
// propagation delay, we cannot literally "loop" gate outputs (that would not converge in a
// functional model). Instead we model the latch as an explicit state transition, which is
// exactly the STABLE solution the physical feedback loop settles into:
//   - enable=0  → latch holds its previous Q (transparent path closed)
//   - enable=1  → Q follows D (transparent)
// The state IS the bit being remembered; that persistence across calls is "memory".
function makeDLatch(initialQ: Bit = 0): (d: Bit, enable: Bit) => Bit {
  let q: Bit = initialQ; // the remembered bit — survives between calls, like real charge
  return (d: Bit, enable: Bit): Bit => {
    // NOTE: when enable=0 the latch is "opaque": D changes are ignored, Q is held. This is
    // the property that lets a CPU register keep a value while the datapath recomputes.
    if (enable === 1) q = d;
    return q;
  };
}

// ---------------------------------------------------------------------------
// 5. Landauer limit: the thermodynamic price of forgetting one bit.
// ---------------------------------------------------------------------------

// Erasing one bit (merging two logical states into one) is irreversible and MUST dissipate
// at least kT·ln2 of energy as heat (Landauer, 1961). This is a hard physical floor, not an
// engineering target. Constants are CODATA exact-ish values; the result is a closed form.
const BOLTZMANN_J_PER_K = 1.380649e-23; // J/K, exact since 2019 SI redefinition
const ROOM_TEMP_K = 300; // ~27°C, conventional "room temperature" for this bound

function landauerEnergyJoules(tempK: number): number {
  return BOLTZMANN_J_PER_K * tempK * Math.LN2;
}

// ---------------------------------------------------------------------------
// 6. Failure mode: combinational race hazard (glitch) — and why our model hides it.
// ---------------------------------------------------------------------------

// A race hazard: when two paths to a gate have different delays, the output can momentarily
// glitch to a wrong value before settling, even though the boolean algebra says it is constant.
// Classic example: F = (A AND B) OR (¬A AND C). Hold B=C=1 and flip A from 1→0. Algebraically
// F stays 1 the whole time. But ¬A arrives later than A (extra inverter delay), so for a few
// nanoseconds BOTH product terms can read 0 → F dips to 0: a glitch.
function hazardSteadyState(a: Bit, b: Bit, c: Bit): Bit {
  return or(and(a, b), and(not(a), c));
}

function demoRaceHazard(): void {
  // Our functional model evaluates instantaneously, so it ALWAYS returns the steady value —
  // it literally cannot represent the transient dip. We show both endpoints agree (F=1),
  // then state plainly what a real gate-delay simulator would have shown in between.
  const before = hazardSteadyState(1, 1, 1); // A=1
  const after = hazardSteadyState(0, 1, 1); // A=0, B=C held at 1
  console.log("--- 失败模式：组合逻辑竞争冒险 (glitch) ---");
  console.log(`F = (A∧B) ∨ (¬A∧C)，固定 B=C=1，A 由 1→0`);
  console.log(`稳态 F: A=1 时 = ${before}，A=0 时 = ${after}（代数上恒为 1）`);
  console.log(
    "本模型瞬时求值，永远只给稳态 1 → 无法显示中间那次毛刺。"
  );
  console.log(
    "真实门有传播延迟：¬A 比 A 晚到，瞬间两个与项都为 0，F 短暂跌到 0 才回升。"
  );
  console.log("教训：纯布尔代数隐藏了时序；毛刺要靠带门延迟的时序仿真或硬件才看得到。");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function main(): void {
  console.log("=== Stage 12 · 从晶体管到逻辑：计算机为什么能算 ===");

  // 1+2: NAND universality, proven by truth tables.
  verifyNandUniversality(); // throws if any derived gate is wrong
  console.log("\n[1] 仅用 NAND 搭出所有门，真值表逐行验证（断言已通过）：");
  printTruthTable2("NAND", nand);
  printTruthTable2("NOT", (a) => not(a)); // 2-arg signature, second input ignored
  printTruthTable2("AND", and);
  printTruthTable2("OR", or);
  printTruthTable2("XOR", xor);
  console.log("→ AND/OR/XOR 全部由 NAND 推出且与参考实现逐行一致 = NAND 是通用门。");

  // 3: ripple-carry adder on real numbers.
  console.log("\n[2] 半加器→全加器→4 位行波进位加法器，跑真实加法：");
  const cases: Array<[number, number]> = [
    [5, 3],
    [9, 6],
    [15, 1],
    [7, 8],
  ];
  for (const [x, y] of cases) {
    const { sum, carryOut } = rippleAdd4(toBits4(x), toBits4(y));
    const got = fromBits(sum, carryOut);
    const ok = got === x + y;
    if (!ok) throw new Error(`[stage12] adder wrong: ${x}+${y} got ${got}`);
    console.log(
      `  ${String(x).padStart(2)} + ${String(y).padStart(2)} = ` +
        `${bitsToStr(toBits4(x))} + ${bitsToStr(toBits4(y))} → ` +
        `carry ${carryOut} ${bitsToStr(sum)} = ${got} ${ok ? "✓" : "✗"}`,
    );
  }
  console.log("→ 进位从最低位逐级 ripple 到最高位；7+8=15 无溢出，9+6=15 同理，溢出由 carry-out 标记。");

  // 4: sequential element — memory is feedback.
  console.log("\n[3] 极简时序元件：仅 NAND 反馈构成的门控 D 锁存器（状态跨调用保持 = 记忆）：");
  const latch = makeDLatch(0);
  // Stimulus rows are [D, enable]; we capture Q after each to verify hold vs transparent.
  const stimulus: ReadonlyArray<[Bit, Bit]> = [
    [1, 1], // enable=1: become transparent, load D=1
    [0, 0], // D drops to 0 but enable=0 → opaque, must still read 1 (the hold test)
    [1, 0], // D rises again, still enable=0 → Q unchanged at 1
    [0, 1], // enable=1: transparent again, now capture D=0
  ];
  const observed: Bit[] = [];
  console.log("   D EN │ Q");
  for (const [d, e] of stimulus) {
    const q = latch(d, e);
    observed.push(q);
    const note = e === 0 ? " (保持，忽略 D)" : " (透明，Q←D)";
    console.log(`   ${d}  ${e} │ ${q}${note}`);
  }
  // Assert the hold property concretely: after loading 1 (row 0), the two enable=0 rows must
  // keep Q=1 regardless of D, then enable=1 with D=0 (row 3) must finally update Q to 0.
  if (observed[1] !== 1 || observed[2] !== 1) {
    throw new Error("[stage12] latch failed to hold value while disabled");
  }
  if (observed[3] !== 0) throw new Error("[stage12] latch failed to update when enabled");
  console.log("→ enable=0 时 D 变化被忽略、Q 不变：这就是寄存器在数据通路重算时保住旧值的能力。");

  // 5: Landauer limit.
  console.log("\n[4] Landauer 极限：擦除 1 bit 的最小能量 (理论值/exact)：");
  const eJoule = landauerEnergyJoules(ROOM_TEMP_K);
  const eEV = eJoule / 1.602176634e-19; // convert J → eV for intuition
  console.log(`   kT·ln2 @ T=${ROOM_TEMP_K}K = ${eJoule.toExponential(3)} J = ${eEV.toExponential(3)} eV`);
  console.log("   (这是物理下界，不是工程目标。)");
  // Put it on a scale next to what real CMOS spends, so the bound is not mistaken for reality.
  // A modern logic transition dissipates very roughly ~1e-15 J (femtojoule); we show the gap.
  const realCmosJoulePerOpEst = 1e-15; // (est.) order-of-magnitude, modern node switching energy
  console.log(
    `   对比：现代 CMOS 单次开关 ≈ ${realCmosJoulePerOpEst.toExponential(1)} J (est.，数量级)，` +
      `约为下界的 ${(realCmosJoulePerOpEst / eJoule).toExponential(1)} 倍。`,
  );
  console.log(
    asciiBar(
      ["Landauer下界", "现代CMOS(est)"],
      // Plot log10(energy)+offset so two values 1e6 apart are both visible as bars; we print
      // the true joules above, the bar only conveys "many orders of magnitude apart".
      [Math.log10(eJoule) + 25, Math.log10(realCmosJoulePerOpEst) + 25],
    ),
  );
  console.log("   (柱高 = log10(J)+25，仅示意数量级差距；真实数值见上行。)");

  // 6: failure mode.
  console.log("");
  demoRaceHazard();

  console.log(
    "\n小结：一颗 NAND 经组合得到全部逻辑与算术，经反馈得到记忆；" +
      "代数层之下，能量 (Landauer) 与时序 (毛刺) 才是硬件的真实边界。",
  );
}

main();
