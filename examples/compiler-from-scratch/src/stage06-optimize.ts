// stage06-optimize.ts — 第06章「基本优化」: 常量折叠 + 死代码消除,用指令数说话。
//
// WHY 这个文件自带一套线性 IR(而非直接优化 core 的树形 AST):
//   优化的真正战场是「线性中间表示」(IR, intermediate representation) —— 三地址码
//   式的指令流 + 基本块。树形 AST 上做常量折叠尚可,但「死代码消除」「不可达块删除」
//   「活跃性分析」这些都以指令为单位、以控制流图为骨架,在树上根本没法自然表达。所以
//   本章把树降低(lower)成一个最小的指令数组 + 基本块,然后所有 pass 都在它上面跑。
//   IR 的指令计数是确定性的(同输入恒等),正好用来「用指令数说话」—— 这是优化效果的
//   诚实度量,不靠拍脑袋说「快多了」。
//
// 这个 toy IR 的形状(刻意极简,只够撑起本章四个机制):
//   - 每条指令产出一个虚拟寄存器(SSA-ish: 每个 dst 只被写一次,简化活跃性分析)。
//   - 指令分两类: 纯指令(Const/Bin/Un —— 无副作用,可安全删) 与 副作用指令
//     (Print/Call —— 删了会改变程序可观察行为,DCE 必须保守放过)。
//   - 程序 = 基本块数组;块之间用 Jump/Branch 连接,构成控制流图(CFG)。
//
// 不变量(INVARIANTS):
//   - 每个虚拟寄存器恰好被一条指令定义(单赋值)。活跃性据此 O(1) 判断「某 dst 是否被用」。
//   - 优化 pass 是「函数式」的: 输入一个 IR,返回新 IR + 本轮变更计数,不原地改输入
//     (便于打印折叠前/后对照、便于不动点迭代时比较「还有没有变化」)。
//   - 副作用指令(Print/Call)永远不被 DCE 删除,即使其 dst 无人使用 —— 见失败模式。
//   - 所有打印的「指令数 / 删除条数 / 变更计数 / 执行条数」都是代码真数出来的,确定性;
//     只有 wall-clock(ms)是本机实测的展示量,不进断言(断言只用相对加速比)。
//
// 失败模式(FAILURE MODES,本文件用 demoUnsafeDce 真演示):
//   - 「天真 DCE」: 凡 dst 没被后续指令引用就删 —— 会把 Print / 有副作用的 Call 一起删掉,
//     程序的输出 / 外部效果就此消失。这是优化器最经典的正确性事故: 优化必须保持
//     **可观察行为**不变,而不仅仅是「值」不变。本文件先跑安全 DCE(放过副作用),再跑
//     一次天真 DCE,打印两者输出差异,坐实「必须保守对待副作用」。

import { assertEq, fmtNum, speedup, timeIt } from "./core/index.js";

// ───────────────────────── IR 定义 ─────────────────────────

/** 虚拟寄存器名: r0, r1, ...。用 string 而非 number 让 IR dump 可读。 */
type Reg = string;

/**
 * 一条 IR 指令。判别联合,`op` 是判别字段。
 * 设计要点: 把「是否有副作用」编码进种类本身(Print/Call vs 其余),
 * 让 DCE 只需看 op 就能判断能不能删,无需额外的 effect 分析表。
 */
type Instr =
  | { readonly op: "Const"; readonly dst: Reg; readonly value: number | boolean } // 字面量
  | { readonly op: "Bin"; readonly dst: Reg; readonly l: Reg; readonly r: Reg; readonly fn: string } // 二元运算
  | { readonly op: "Un"; readonly dst: Reg; readonly a: Reg; readonly fn: string } // 一元运算(如 ! -)
  | { readonly op: "Print"; readonly a: Reg } // 副作用: 写 stdout。无 dst。
  | { readonly op: "Call"; readonly dst: Reg; readonly name: string; readonly args: readonly Reg[] }; // 副作用: 可能触发外部效果

/** 一个基本块: 一串无分支的指令 + 一个结尾跳转(terminator)。 */
interface Block {
  readonly label: string;
  readonly body: readonly Instr[];
  readonly term: Terminator;
}

/** 块的出口。Jump 无条件,Branch 按某寄存器真假二选一,Halt 结束程序。 */
type Terminator =
  | { readonly kind: "Jump"; readonly to: string }
  | { readonly kind: "Branch"; readonly cond: Reg; readonly then: string; readonly else: string }
  | { readonly kind: "Halt" };

/** 整个程序 = 基本块数组,第 0 块为入口。 */
type IrProgram = readonly Block[];

// ───────────────────────── 指令计数(诚实度量) ─────────────────────────

/** 数一个程序里的指令总数(不含 terminator —— terminator 是块结构,不是可优化的「工作」)。 */
function countInstrs(prog: IrProgram): number {
  return prog.reduce((sum, b) => sum + b.body.length, 0);
}

/** 把一条指令渲染成可读文本(对拍 + 打印用)。确定性。 */
function fmtInstr(i: Instr): string {
  switch (i.op) {
    case "Const":
      return `${i.dst} = const ${i.value}`;
    case "Bin":
      return `${i.dst} = ${i.l} ${i.fn} ${i.r}`;
    case "Un":
      return `${i.dst} = ${i.fn}${i.a}`;
    case "Print":
      return `print ${i.a}`;
    case "Call":
      return `${i.dst} = call ${i.name}(${i.args.join(", ")})`;
  }
}

/** 把整个程序 dump 成一行确定性字符串,供 assertEq 端到端对拍。 */

// ───────────────────────── Pass 1: 常量折叠 ─────────────────────────

/** 一个值环境: 已知为常量的寄存器 → 其常量值。只装真常量,不装运行期未知量。 */
type ConstEnv = Map<Reg, number | boolean>;

/** 求一个二元运算的常量结果。返回 undefined 表示「这个运算符不折叠 / 操作数非常量」。 */
function foldBin(fn: string, l: number | boolean, r: number | boolean): number | boolean | undefined {
  // 算术只在双数时折叠;逻辑只在双布尔时折叠。混类型不折叠(留给后续类型检查报错,优化器不僭越)。
  if (typeof l === "number" && typeof r === "number") {
    switch (fn) {
      case "+":
        return l + r;
      case "-":
        return l - r;
      case "*":
        return l * r;
      case "/":
        // WHY 不折叠除以 0: 折成 Infinity 会把「运行期错误」提前固化成一个错误的常量,
        // 改变了程序在该路径上的行为(本应在运行到这里时才出错)。保守放过。
        return r === 0 ? undefined : l / r;
      case "<":
        return l < r;
      case "<=":
        return l <= r;
      case ">":
        return l > r;
      case ">=":
        return l >= r;
      case "==":
        return l === r;
    }
  }
  if (typeof l === "boolean" && typeof r === "boolean") {
    switch (fn) {
      case "&&":
        return l && r;
      case "||":
        return l || r;
      case "==":
        return l === r;
    }
  }
  return undefined;
}

function foldUn(fn: string, a: number | boolean): number | boolean | undefined {
  if (fn === "-" && typeof a === "number") return -a;
  if (fn === "!" && typeof a === "boolean") return !a;
  return undefined;
}

/**
 * 常量折叠 pass。把操作数全为常量的 Bin/Un 替换成 Const,并把结果记进环境,
 * 让下游运算能链式折叠(这就是 `1+2+3` 能一路折到 6、`2*60*60` 折到 7200 的原因)。
 *
 * 返回新程序 + 本轮把多少条指令变成了 Const(变更计数,驱动不动点)。
 * Why 不原地改: 见文件头不变量 —— 函数式 pass 才能打印前/后对照、才能比较「是否还在变」。
 */
function passConstFold(prog: IrProgram): { prog: IrProgram; changed: number } {
  let changed = 0;
  const env: ConstEnv = new Map();

  const out = prog.map((block) => {
    const body = block.body.map((i): Instr => {
      switch (i.op) {
        case "Const":
          env.set(i.dst, i.value);
          return i;
        case "Bin": {
          const lv = env.get(i.l);
          const rv = env.get(i.r);
          if (lv !== undefined && rv !== undefined) {
            const folded = foldBin(i.fn, lv, rv);
            if (folded !== undefined) {
              changed++;
              env.set(i.dst, folded);
              return { op: "Const", dst: i.dst, value: folded };
            }
          }
          return i;
        }
        case "Un": {
          const av = env.get(i.a);
          if (av !== undefined) {
            const folded = foldUn(i.fn, av);
            if (folded !== undefined) {
              changed++;
              env.set(i.dst, folded);
              return { op: "Const", dst: i.dst, value: folded };
            }
          }
          return i;
        }
        case "Print":
        case "Call":
          // 副作用指令的 dst(若有)不是常量(可能依赖运行期),不进环境。
          return i;
      }
    });
    return { ...block, body };
  });

  return { prog: out, changed };
}

// ───────────────────── Pass 2: 死代码消除(基于活跃性) ─────────────────────

/** 收集一条指令「读了哪些寄存器」(它的 uses)。Const 不读任何寄存器。 */
function usesOf(i: Instr): readonly Reg[] {
  switch (i.op) {
    case "Const":
      return [];
    case "Bin":
      return [i.l, i.r];
    case "Un":
      return [i.a];
    case "Print":
      return [i.a];
    case "Call":
      return i.args;
  }
}

/** terminator 读的寄存器(Branch 读 cond)。这些寄存器无论如何都活,不能删其定义。 */
function usesOfTerm(t: Terminator): readonly Reg[] {
  return t.kind === "Branch" ? [t.cond] : [];
}

/**
 * 死代码消除 pass。删掉「结果从未被任何指令 / terminator 使用」的纯指令。
 *
 * 关键不变量(也是失败模式的根源): 只删「无副作用」指令。Print/Call 即使 dst 没人用,
 * 也必须保留 —— 它们的价值在副作用(输出 / 外部调用),不在返回值。把这条放进种类判断,
 * 而不是依赖调用方记得检查,正是为了让「漏判副作用」无法发生。
 *
 * 算法: 单遍计算活跃集 —— 先把所有「被读到的寄存器」收进 live,再保留满足以下之一的指令:
 *   (a) 有副作用(Print/Call),或 (b) 其 dst 在 live 集里。
 * 注: 真实编译器活跃性是逐块倒序的数据流不动点;这里 IR 单赋值 + 用途跨块少,用一遍
 *     全局 live 收集近似,足够演示机制且确定性。
 */
function passDce(prog: IrProgram, safe = true): { prog: IrProgram; removed: number } {
  // 1. 全局活跃集: 任何指令 / terminator 读过的寄存器都活。
  const live = new Set<Reg>();
  for (const block of prog) {
    for (const i of block.body) for (const r of usesOf(i)) live.add(r);
    for (const r of usesOfTerm(block.term)) live.add(r);
  }

  // 2. 保留: 副作用指令(safe 模式下) 或 dst 活的指令;其余删除。
  let removed = 0;
  const out = prog.map((block) => {
    const body = block.body.filter((i) => {
      const hasEffect = i.op === "Print" || i.op === "Call";
      if (safe && hasEffect) return true; // ← 保守对待副作用。unsafe 模式故意跳过这条 → 失败模式。
      const dst = "dst" in i ? i.dst : undefined;
      const keep = dst !== undefined && live.has(dst);
      if (!keep) removed++;
      return keep;
    });
    return { ...block, body };
  });

  return { prog: out, removed };
}

// ───────────────── Pass 3: 不可达基本块消除 ─────────────────

/**
 * 从入口块(第 0 块)做可达性遍历,删掉走不到的块。
 * 常量折叠常制造不可达块: 一旦 Branch 的 cond 被折成常量,有一条分支永远走不到,
 * 整个目标块(及其专属后继)就成了死块。这里只删块本身;块内指令的清理交给 DCE。
 *
 * 返回新程序 + 删除的块数。注意第 0 块永远可达,不会被删。
 */
function passUnreachableBlocks(prog: IrProgram): { prog: IrProgram; removed: number } {
  if (prog.length === 0) return { prog, removed: 0 };
  const byLabel = new Map(prog.map((b) => [b.label, b]));
  const reachable = new Set<string>();
  const stack = [prog[0].label]; // 入口

  while (stack.length > 0) {
    const label = stack.pop()!;
    if (reachable.has(label)) continue;
    reachable.add(label);
    const block = byLabel.get(label);
    if (!block) continue;
    const t = block.term;
    if (t.kind === "Jump") stack.push(t.to);
    else if (t.kind === "Branch") {
      stack.push(t.then);
      stack.push(t.else);
    }
  }

  const out = prog.filter((b) => reachable.has(b.label));
  return { prog: out, removed: prog.length - out.length };
}

/**
 * 折叠后的「分支简化」: 如果 Branch 的 cond 是一个已折成常量布尔的寄存器,
 * 把它改写成无条件 Jump 到对应分支。这是「让不可达块暴露出来」的前置步骤 ——
 * 不简化分支,unreachable pass 就以为两条分支都可达,永远删不掉死块。
 *
 * 返回新程序 + 简化了几个 Branch。
 */
function passSimplifyBranch(prog: IrProgram): { prog: IrProgram; changed: number } {
  let changed = 0;
  // 收集每个寄存器是否为已知常量布尔(只看 Const 指令)。
  const constBool = new Map<Reg, boolean>();
  for (const block of prog) {
    for (const i of block.body) {
      if (i.op === "Const" && typeof i.value === "boolean") constBool.set(i.dst, i.value);
    }
  }

  const out = prog.map((block) => {
    if (block.term.kind !== "Branch") return block;
    const known = constBool.get(block.term.cond);
    if (known === undefined) return block;
    changed++;
    const to = known ? block.term.then : block.term.else;
    return { ...block, term: { kind: "Jump", to } as Terminator };
  });

  return { prog: out, changed };
}

// ───────────────── 不动点驱动: 反复跑直到没有变化 ─────────────────

/**
 * 把所有 pass 跑到不动点(fixed point): 一轮里任何 pass 报告了变更,就再来一轮,
 * 直到一整轮下来变更计数归零。
 *
 * Why 需要不动点(本章核心论点): pass 之间互相喂材料 —— 常量折叠产出新常量,使下游
 * 运算这一轮还看不到、下一轮才折得动;折叠又把 Branch 变成可简化的;分支简化又让某些
 * 块不可达;删块又让某些寄存器不再被引用,使 DCE 这一轮删不掉、下一轮才删得掉。
 * 「只跑一遍」必然漏掉这些二次(乃至三次)优化机会。本函数打印每轮变更计数,让这条
 * 「机会瀑布」肉眼可见: 计数通常逐轮下降,最后一轮为 0 才停。
 *
 * 返回最终程序 + 每轮的变更明细(供打印与对比「只跑一遍」)。
 */
interface RoundLog {
  readonly round: number;
  readonly fold: number;
  readonly branch: number;
  readonly unreachableBlocks: number;
  readonly dce: number;
  readonly total: number;
}

function optimizeToFixpoint(input: IrProgram): { prog: IrProgram; rounds: RoundLog[] } {
  let prog = input;
  const rounds: RoundLog[] = [];
  let round = 0;
  // 上限防御: 若某 pass 有 bug 反复横跳,不至于死循环。正常程序远远到不了。
  const MAX_ROUNDS = 100;

  for (;;) {
    round++;
    const fold = passConstFold(prog);
    prog = fold.prog;
    const branch = passSimplifyBranch(prog);
    prog = branch.prog;
    const ub = passUnreachableBlocks(prog);
    prog = ub.prog;
    const dce = passDce(prog, /*safe*/ true);
    prog = dce.prog;

    const total = fold.changed + branch.changed + ub.removed + dce.removed;
    rounds.push({
      round,
      fold: fold.changed,
      branch: branch.changed,
      unreachableBlocks: ub.removed,
      dce: dce.removed,
      total,
    });

    if (total === 0 || round >= MAX_ROUNDS) break;
  }
  return { prog, rounds };
}

// ───────────────────── 极简 VM: 执行 IR 并计执行指令条数 ─────────────────────

/**
 * 解释执行一个 IR 程序,返回 (打印的输出, 执行了多少条指令)。
 * 「执行条数」是端到端优化效果的诚实度量: 优化删掉的不只是静态指令,更是运行期真正
 * 执行的工作量。VM 故意极简(只够跑本章的样例),不是第07章那个完整 VM —— 但口径一致:
 * 一条 body 指令执行一次记 1。
 *
 * 失败模式可见性: 如果 DCE 误删了 Print,这里的 output 就会缺行 —— 直接坐实「输出丢了」。
 */
function runIr(prog: IrProgram): { output: string[]; executed: number } {
  const byLabel = new Map(prog.map((b) => [b.label, b]));
  const regs = new Map<Reg, number | boolean>();
  const output: string[] = [];
  let executed = 0;
  let cur: string | undefined = prog.length > 0 ? prog[0].label : undefined;
  let steps = 0;
  const MAX_STEPS = 100000; // 防御无限循环(本章样例无环,远到不了)。

  while (cur !== undefined) {
    if (steps++ > MAX_STEPS) throw new Error("runIr: step budget exceeded (infinite loop?)");
    const block = byLabel.get(cur);
    if (!block) throw new Error(`runIr: jump to missing block ${cur}`);
    for (const i of block.body) {
      executed++;
      switch (i.op) {
        case "Const":
          regs.set(i.dst, i.value);
          break;
        case "Bin": {
          const l = regs.get(i.l)!;
          const r = regs.get(i.r)!;
          regs.set(i.dst, evalBin(i.fn, l, r));
          break;
        }
        case "Un":
          regs.set(i.dst, evalUn(i.fn, regs.get(i.a)!));
          break;
        case "Print":
          output.push(String(regs.get(i.a)));
          break;
        case "Call":
          // toy: 唯一支持的 call 是 "side_effect",记录一次外部效果(用于失败模式演示)。
          output.push(`<effect:${i.name}>`);
          regs.set(i.dst, 0);
          break;
      }
    }
    const t = block.term;
    if (t.kind === "Jump") cur = t.to;
    else if (t.kind === "Branch") cur = regs.get(t.cond) ? t.then : t.else;
    else cur = undefined; // Halt
  }
  return { output, executed };
}

function evalBin(fn: string, l: number | boolean, r: number | boolean): number | boolean {
  const folded = foldBin(fn, l, r);
  if (folded === undefined) throw new Error(`evalBin: unsupported ${l} ${fn} ${r}`);
  return folded;
}
function evalUn(fn: string, a: number | boolean): number | boolean {
  const folded = foldUn(fn, a);
  if (folded === undefined) throw new Error(`evalUn: unsupported ${fn}${a}`);
  return folded;
}

// ───────────────────────── 样例 IR 构造 ─────────────────────────
//
// 这些样例对应本章正文里的源码片段。手写 IR(而非走完整 lexer→parser→lower)是为了
// 让本章聚焦优化本身、每条指令都肉眼可数。注释标出对应的「源码语义」。

/** let secs = 2*60*60; print secs;  → 含可链式折叠的算术 (2*60*60 → 7200)。 */
function sampleTimeConst(): IrProgram {
  return [
    {
      label: "entry",
      body: [
        { op: "Const", dst: "r0", value: 2 },
        { op: "Const", dst: "r1", value: 60 },
        { op: "Const", dst: "r2", value: 60 },
        { op: "Bin", dst: "r3", l: "r0", r: "r1", fn: "*" }, // 2*60
        { op: "Bin", dst: "r4", l: "r3", r: "r2", fn: "*" }, // (2*60)*60 —— 第二步要等第一步折完
        { op: "Print", a: "r4" },
      ],
      term: { kind: "Halt" },
    },
  ];
}

/**
 * if (true && false) { print 1; } else { print 2; }
 * → 条件折成常量,then 分支不可达,整块该被删。
 * 同时含 1+2+3 链式折叠 + 一条「算了没用」的死指令(其 dst 无人引用)。
 */
function sampleBranchDead(): IrProgram {
  return [
    {
      label: "entry",
      body: [
        { op: "Const", dst: "t", value: true },
        { op: "Const", dst: "f", value: false },
        { op: "Bin", dst: "cond", l: "t", r: "f", fn: "&&" }, // true && false → false
        // 一条纯死代码: x = 1+2+3,算出来从没被任何指令用到 → DCE 应删。
        { op: "Const", dst: "a", value: 1 },
        { op: "Const", dst: "b", value: 2 },
        { op: "Const", dst: "c", value: 3 },
        { op: "Bin", dst: "ab", l: "a", r: "b", fn: "+" }, // 1+2
        { op: "Bin", dst: "x", l: "ab", r: "c", fn: "+" }, // (1+2)+3 → 6, 死值
      ],
      term: { kind: "Branch", cond: "cond", then: "then", else: "els" },
    },
    {
      label: "then", // 不可达: cond 恒为 false
      body: [
        { op: "Const", dst: "p1", value: 1 },
        { op: "Print", a: "p1" },
      ],
      term: { kind: "Halt" },
    },
    {
      label: "els",
      body: [
        { op: "Const", dst: "p2", value: 2 },
        { op: "Print", a: "p2" },
      ],
      term: { kind: "Halt" },
    },
  ];
}

/**
 * 「二次机会」专用样例: 一条 *非常量* 的死代码使用链,折叠帮不上忙,只能靠 DCE 多轮剥。
 *   v  = call read_input();   // 运行期未知, 折不掉
 *   d1 = v + v;               // 只被 d2 用
 *   d2 = d1 + d1;             // 只被 d3 用
 *   d3 = d2 + d2;             // 没人用 → 死
 *   print 99;
 * 单遍全局活跃性: 第一轮算 live 时 d1 被 d2 引用、d2 被 d3 引用,都「活」;只有链尾 d3 死,删 1 条。
 * 删了 d3 → d2 这轮才暴露成死的;再删 d2 → d1 暴露…… 每轮只剥一层。这正是「只跑一遍」漏掉的二次机会。
 * (注: call read_input 有副作用必须保留,但它的返回值 v 链下去的纯计算是可删的。)
 */
function sampleDceCascade(): IrProgram {
  return [
    {
      label: "entry",
      body: [
        { op: "Call", dst: "v", name: "read_input", args: [] },
        { op: "Bin", dst: "d1", l: "v", r: "v", fn: "+" },
        { op: "Bin", dst: "d2", l: "d1", r: "d1", fn: "+" },
        { op: "Bin", dst: "d3", l: "d2", r: "d2", fn: "+" }, // 死值链的尾
        { op: "Const", dst: "k", value: 99 },
        { op: "Print", a: "k" },
      ],
      term: { kind: "Halt" },
    },
  ];
}

/**
 * 失败模式专用样例: 一条副作用 Call 的 dst 从没被使用。
 *   tmp = call side_effect();   // 返回值没人要, 但调用本身有外部效果
 *   print 42;
 * 安全 DCE 必须放过这条 Call;天真 DCE 会因为「dst 没人用」把它删掉 → 副作用消失。
 */
function sampleEffect(): IrProgram {
  return [
    {
      label: "entry",
      body: [
        { op: "Call", dst: "tmp", name: "side_effect", args: [] }, // dst 无人用,但有副作用
        { op: "Const", dst: "v", value: 42 },
        { op: "Print", a: "v" },
      ],
      term: { kind: "Halt" },
    },
  ];
}

// ───────────────────────── Demo ─────────────────────────

function hr(title: string): void {
  console.log(`\n=== ${title} ===`);
}

function printIr(prog: IrProgram, indent = "  "): void {
  for (const b of prog) {
    const term =
      b.term.kind === "Jump"
        ? `jump ${b.term.to}`
        : b.term.kind === "Branch"
          ? `branch ${b.term.cond} ? ${b.term.then} : ${b.term.else}`
          : "halt";
    console.log(`${indent}${b.label}:`);
    for (const i of b.body) console.log(`${indent}  ${fmtInstr(i)}`);
    console.log(`${indent}  ${term}`);
  }
}

/** ① 常量折叠: 打印折叠前/后 IR + 减少的「待求值运算」条数,并 assertEq。 */
function demoConstFold(): void {
  hr("① 常量折叠 (2*60*60 与 1+2+3)");
  const before = sampleTimeConst();
  console.log("折叠前 IR:");
  printIr(before);

  const { prog: after, changed } = passConstFold(before);
  console.log("折叠后 IR:");
  printIr(after);

  // 「省下的运算」= 被折成 Const 的 Bin/Un 条数(运行期不再需要做这些乘法/加法)。
  console.log(`\n  折叠掉的运算指令 = ${changed} 条 (2*60 与 *60 两次乘法 → 一个常量 7200)`);

  // 对拍: 折叠后入口块最后产出的应是常量 7200。
  const entry = after[0];
  const last = entry.body[entry.body.length - 2]; // 倒数第二条(最后一条是 Print)
  assertEq(last.op === "Const" ? last.value : "NOT_CONST", 7200, "2*60*60 应折成 7200");
  assertEq(changed, 2, "应折叠 2 条乘法");
  console.log("  assertEq 通过: 折叠结果 = 7200, 折叠条数 = 2");

  // 链式折叠样例对拍: 1+2+3 → 6
  const chain = passConstFold(sampleBranchDead());
  const xInstr = chain.prog[0].body.find((i) => "dst" in i && i.dst === "x");
  assertEq(xInstr?.op === "Const" ? xInstr.value : "NOT_CONST", 6, "1+2+3 应折成 6");
  console.log("  assertEq 通过: 1+2+3 链式折叠 = 6");
}

/** ② 死代码消除 + 不可达块: 打印删除条数。 */
function demoDce(): void {
  hr("② 死代码消除 + 不可达块消除");
  // 先折叠 + 简化分支,制造可删的死值和不可达块,再看 DCE / unreachable 删多少。
  const folded = passConstFold(sampleBranchDead());
  const branched = passSimplifyBranch(folded.prog);
  console.log("折叠 + 分支简化后 (cond=true&&false=false → 直接 jump els, then 块变孤立):");
  printIr(branched.prog);

  const ub = passUnreachableBlocks(branched.prog);
  console.log(`\n  不可达块删除 = ${ub.removed} 个 (then 块: cond 恒 false 永远走不到)`);

  const dce = passDce(ub.prog, /*safe*/ true);
  console.log(`  死指令删除 = ${dce.removed} 条 (含 x=1+2+3 等无人使用的纯计算)`);
  console.log("DCE 后:");
  printIr(dce.prog);

  assertEq(ub.removed, 1, "应删 1 个不可达块 (then)");
  console.log("  assertEq 通过: 删除不可达块 = 1");
}

/** ③ 不动点迭代: 打印每轮变更计数,并对比「只跑一遍」漏掉的二次机会。 */
function demoFixpoint(): void {
  hr("③ 不动点迭代 vs 只跑一遍");
  // 用 DCE 级联样例: 一条非常量死代码链 (d3←d2←d1←v), 折叠帮不上, 单遍全局活跃性每轮只剥一层。
  const input = sampleDceCascade();
  const before = countInstrs(input);

  // 只跑一遍(每个 pass 各一次,不回头)。
  const onePass = (() => {
    let p = input;
    p = passConstFold(p).prog;
    p = passSimplifyBranch(p).prog;
    p = passUnreachableBlocks(p).prog;
    p = passDce(p, true).prog;
    return p;
  })();

  // 跑到不动点。
  const { prog: fixpoint, rounds } = optimizeToFixpoint(input);

  console.log("每轮变更计数 (fold / branch / unreachable-blocks / dce / 合计):");
  for (const r of rounds) {
    console.log(
      `  第 ${r.round} 轮: fold=${r.fold} branch=${r.branch} ub=${r.unreachableBlocks} dce=${r.dce} → 合计 ${r.total}`,
    );
  }
  console.log(`  共 ${rounds.length} 轮, 末轮合计 0 → 到达不动点`);

  const oneCount = countInstrs(onePass);
  const fixCount = countInstrs(fixpoint);
  console.log(`\n  指令数: 原始 ${before} → 只跑一遍 ${oneCount} → 不动点 ${fixCount}`);
  console.log(
    `  只跑一遍比不动点多留 ${oneCount - fixCount} 条死指令` +
      (oneCount > fixCount
        ? " (死代码链 d3←d2←d1: 删一层才暴露下一层, 需要逐轮 DCE 才剥得干净)"
        : ""),
  );

  // 行为不变对拍: 两种优化与原始程序的可观察输出必须一致。
  const o0 = runIr(input).output;
  const o1 = runIr(onePass).output;
  const o2 = runIr(fixpoint).output;
  assertEq(o1, o0, "只跑一遍后输出应与原始一致");
  assertEq(o2, o0, "不动点优化后输出应与原始一致");
  console.log(`  行为对拍通过: 三者输出均为 [${o0.join(", ")}] (优化保持可观察行为)`);
}

/** ④ 端到端: 优化前/后 静态指令数 + VM 执行条数 + wall-clock 加速比。 */
function demoEndToEnd(): void {
  hr("④ 端到端: 指令数 + 执行条数 + 运行时加速比");

  // 用一个「重复 N 次同一段可全折叠计算」的程序放大效果,让 wall-clock 测得出来。
  // 源码语义: 把 2*60*60 这类常量算 N 遍再 print —— 朴素 VM 每遍都真做乘法,
  // 优化后每遍只剩一个 Const。N 大到能被 timeIt 稳定测量。
  const N = 4000;
  const body: Instr[] = [];
  for (let k = 0; k < N; k++) {
    body.push({ op: "Const", dst: `a${k}`, value: 2 });
    body.push({ op: "Const", dst: `b${k}`, value: 60 });
    body.push({ op: "Const", dst: `c${k}`, value: 60 });
    body.push({ op: "Bin", dst: `m${k}`, l: `a${k}`, r: `b${k}`, fn: "*" });
    body.push({ op: "Bin", dst: `s${k}`, l: `m${k}`, r: `c${k}`, fn: "*" });
  }
  body.push({ op: "Print", a: `s${N - 1}` }); // 只 print 最后一个,其余 s 是死值
  const heavy: IrProgram = [{ label: "entry", body, term: { kind: "Halt" } }];

  const { prog: optimized, rounds } = optimizeToFixpoint(heavy);

  const beforeInstrs = countInstrs(heavy);
  const afterInstrs = countInstrs(optimized);

  const runBefore = runIr(heavy);
  const runAfter = runIr(optimized);

  console.log(`  静态指令数: ${fmtNum(beforeInstrs)} → ${fmtNum(afterInstrs)} (优化 ${rounds.length} 轮)`);
  console.log(`  VM 执行条数: ${fmtNum(runBefore.executed)} → ${fmtNum(runAfter.executed)}`);
  console.log(
    `  指令削减率 = ${(((beforeInstrs - afterInstrs) / beforeInstrs) * 100).toFixed(1)}% (确定性, 可断言)`,
  );

  // 行为不变: 输出必须一致。
  assertEq(runAfter.output, runBefore.output, "优化前后输出应一致");
  console.log(`  输出一致对拍通过: [${runBefore.output.join(", ")}]`);

  // wall-clock: 真测两个 VM 跑同一程序的耗时。绝对 ms 仅展示, 断言只用相对加速比。
  const tSlow = timeIt(() => void runIr(heavy), { runs: 9, inner: 1 });
  const tFast = timeIt(() => void runIr(optimized), { runs: 9, inner: 50 });
  const sp = speedup(tSlow, tFast);
  console.log(
    `  VM 执行耗时(本机实测): 优化前 ${tSlow.medianMs.toFixed(4)} ms, 优化后 ${tFast.medianMs.toFixed(4)} ms`,
  );
  console.log(`  运行时加速比 = ${sp.toFixed(1)}x (相对量, 可复现)`);
  console.log(
    "  NOTE: 这是 toy IR + toy VM 上的合成数据, 绝对加速比偏乐观(真实程序不会全是可折叠常量);",
  );
  console.log("        可迁移的是趋势: 常量折叠 + DCE 把「运行期重复计算」削成「编译期一次算好」。");
}

/** 失败模式: 天真 DCE 误删有副作用的指令,程序输出/外部效果消失。 */
function demoUnsafeDce(): void {
  hr("⑤ 失败模式: 天真 DCE 误删副作用指令");
  const prog = sampleEffect();
  console.log("原始 IR (tmp = call side_effect() 的返回值无人使用, 但调用有副作用):");
  printIr(prog);

  const safe = passDce(prog, /*safe*/ true);
  const unsafe = passDce(prog, /*safe*/ false); // 故意不放过副作用 —— 错误的 DCE

  const outOriginal = runIr(prog).output;
  const outSafe = runIr(safe.prog).output;
  const outUnsafe = runIr(unsafe.prog).output;

  console.log(`\n  原始程序输出:        [${outOriginal.join(", ")}]`);
  console.log(`  安全 DCE 后输出:     [${outSafe.join(", ")}] (删 ${safe.removed} 条, 放过 Call)`);
  console.log(`  天真 DCE 后输出:     [${outUnsafe.join(", ")}] (删 ${unsafe.removed} 条, 含那条 Call)`);

  const lostEffects = outOriginal.filter((line) => !outUnsafe.includes(line));
  console.log(`\n  天真 DCE 丢失的外部效果: [${lostEffects.join(", ")}]`);
  console.log(
    "  ROOT CAUSE: 天真 DCE 只看「dst 是否被使用」, 把「值没人要」误判为「指令可删」。",
  );
  console.log(
    "  但 Print/Call 的价值在副作用而非返回值 —— 优化必须保持「可观察行为」, 不只是「值」。",
  );

  // 把「事故」固化成断言: 安全 DCE 输出 = 原始;天真 DCE 输出 ≠ 原始(确实出了事故)。
  assertEq(outSafe, outOriginal, "安全 DCE 必须保持输出不变");
  if (JSON.stringify(outUnsafe) === JSON.stringify(outOriginal)) {
    throw new Error("失败模式演示失效: 天真 DCE 本应丢失副作用却没有");
  }
  console.log("  assertEq 通过: 安全 DCE 输出不变; 天真 DCE 确实丢了副作用(失败模式坐实)。");
}

function main(): void {
  console.log("第06章 基本优化: 常量折叠 + 死代码消除 — 用指令数说话");
  demoConstFold();
  demoDce();
  demoFixpoint();
  demoEndToEnd();
  demoUnsafeDce();
  console.log("\n=== stage06 全部 demo 通过 ===");
}

main();
