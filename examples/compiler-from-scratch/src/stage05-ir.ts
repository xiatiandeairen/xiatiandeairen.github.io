// stage05-ir.ts — 第 5 章「中间表示」: 把树状 AST 降成线性三地址 IR(intermediate
// representation), 后端从此面对的是指令序列而非语法树。
//
// WHY 这一章存在:
//   语法树是「嵌套」的 —— 一个 `if` 把 then/else 整块抱在子节点里, 一个 `while` 把
//   循环体抱在里面。后端(寄存器分配、指令选择、数据流分析)想要的是「线性」的: 一条
//   挨一条的指令, 控制流靠显式 label + 跳转表达。把 AST 降成 IR 这一步, 就是把「结构
//   的嵌套」翻译成「跳转的拓扑」。降低(lowering)之后, 后端再也不必认识 if/while/&&
//   这些语法概念 —— 它只认识 jump / cond_jump / binary。
//
// 本文件实测什么(对应章节 4 个要点):
//   ① fib + loopsum 降为三地址 IR, 打印线性指令(含临时 %t0、label、cond_jump),
//      用 astToSExpr 之外的「IR 文本」做 assertEq 对拍 —— 同一棵树永远降出同一序列。
//   ② 控制流降低: if/while → label + cond_jump; 打印 while 的回边(back-edge)。
//   ③ 基本块 + CFG: 切块、算每块的前驱/后继集合。
//   ④ AST 节点数 vs IR 指令数, 算降低展开比。
//   ⑤ 失败模式: 短路 && 当成普通二元算符降低 → 右侧本不该求值却被求值。用一个带
//      副作用计数的 IR 解释器实跑两种降低, 用「副作用次数」证明短路语义被破坏。
//
// 为什么自己手搓 AST 而不 import parser:
//   stageNN 文件加载即跑 main(), 严禁互相 import(见 core/index.ts 注释)。parser 在
//   stage02。所以本章直接用 core/ast.ts 的节点类型「手工建树」—— 这恰好也让「降低」
//   的输入完全确定, 对拍不受 parser 演进影响。建的就是 fib.lox / loopsum.lox 那两棵树。
//
// 不变量(INVARIANTS):
//   - 降低是确定性的: temp / label 计数器从 0 单调递增, 同一 AST 永远产同一 IR 文本。
//   - 每个基本块: 唯一入口(块首是 leader), 唯一出口(块尾是 terminator: jump/cond_jump/
//     return, 或 fall-through 到下一块)。块中间不含 label、不含跳转。
//   - CFG 的 succ/pred 互为反向: b ∈ succ(a) ⟺ a ∈ pred(b)。selftest 末尾会校验。
//
// 失败模式(FAILURE MODES):
//   - 短路运算符若按普通 binary 降低: 先把左右两侧都算成 temp 再做 &&, 右侧的副作用
//     无条件发生。正确降低必须用 cond_jump 跳过右侧。本文件 demo 这个差异(§5)。

import {
  span,
  type Span,
  type Expr,
  type Stmt,
  type Program,
  astToSExpr,
  assertEq,
  fmtNum,
} from "./core/index.js";

// 手搓 AST 用的零宽 span: 这些树不指回真实源码偏移(我们没跑 lexer), 降低逻辑也不读
// span, 所以统一用 span(0)。span 在这里只是为了满足 AST 节点的必填字段。
const S: Span = span(0);

// ---------- AST 建造小工具(只为让手搓树可读, 非 IR 的一部分) ----------

const lit = (value: number | string | boolean): Expr => ({ kind: "Literal", value, span: S });
const v = (name: string): Expr => ({ kind: "Var", name, span: S });
const bin = (op: string, left: Expr, right: Expr): Expr => ({ kind: "Binary", op, left, right, span: S });
const call = (callee: Expr, args: Expr[]): Expr => ({ kind: "Call", callee, args, span: S });
const assign = (name: string, value: Expr): Expr => ({ kind: "Assign", name, value, span: S });

const letS = (name: string, init: Expr): Stmt => ({ kind: "Let", name, init, span: S });
const printS = (value: Expr): Stmt => ({ kind: "Print", value, span: S });
const ret = (value?: Expr): Stmt => ({ kind: "Return", value, span: S });
const ifS = (cond: Expr, then: Stmt[], els?: Stmt[]): Stmt => ({ kind: "If", cond, then, else: els, span: S });
const whileS = (cond: Expr, body: Stmt[]): Stmt => ({ kind: "While", cond, body, span: S });
const exprS = (expr: Expr): Stmt => ({ kind: "ExprStmt", expr, span: S });

// fib 体(对应 fib.lox 的函数体, 不含顶层 print, 因为 fib 是被调用的函数):
//   if (n < 2) { return n; }
//   return fib(n-1) + fib(n-2);
function buildFibBody(): Program {
  return [
    ifS(bin("<", v("n"), lit(2)), [ret(v("n"))]),
    ret(bin("+", call(v("fib"), [bin("-", v("n"), lit(1))]), call(v("fib"), [bin("-", v("n"), lit(2))]))),
  ];
}

// loopsum 体(对应 loopsum.lox 顶层):
//   let sum = 0; let i = 1;
//   while (i <= 100) { sum = sum + i; i = i + 1; }
//   print sum;
function buildLoopSum(): Program {
  return [
    letS("sum", lit(0)),
    letS("i", lit(1)),
    whileS(bin("<=", v("i"), lit(100)), [
      exprS(assign("sum", bin("+", v("sum"), v("i")))),
      exprS(assign("i", bin("+", v("i"), lit(1)))),
    ]),
    printS(v("sum")),
  ];
}

// ---------- IR 定义: 三地址指令(判别联合) ----------
//
// 「三地址」= 每条指令最多引用三个地址(两个操作数 + 一个结果), 形如 `t = a op b`。
// 复合表达式靠引入临时变量(temp)拆平: `a + b * c` → `t0 = b*c; t1 = a+t0`。

// IR 操作数: 要么是临时/具名变量(用名字引用), 要么是字面量常量。
type Operand =
  | { readonly kind: "temp"; readonly name: string } // %t0 等; 也用于具名变量 sum/i/n
  | { readonly kind: "const"; readonly value: number | string | boolean };

type Instr =
  | { readonly kind: "const"; readonly dst: string; readonly value: number | string | boolean }
  | { readonly kind: "copy"; readonly dst: string; readonly src: Operand } // 赋值 / move
  | { readonly kind: "binary"; readonly dst: string; readonly op: string; readonly left: Operand; readonly right: Operand }
  | { readonly kind: "unary"; readonly dst: string; readonly op: string; readonly operand: Operand }
  | { readonly kind: "call"; readonly dst: string; readonly callee: string; readonly args: readonly Operand[] }
  | { readonly kind: "label"; readonly name: string } // 跳转目标; 是基本块的 leader 标志之一
  | { readonly kind: "jump"; readonly target: string } // 无条件跳转(terminator)
  | { readonly kind: "cond_jump"; readonly cond: Operand; readonly ifTrue: string; readonly ifFalse: string } // 条件跳转(terminator)
  | { readonly kind: "print"; readonly src: Operand }
  | { readonly kind: "return"; readonly src?: Operand };

// ---------- Lowering: AST → 线性 IR ----------
//
// 一个 Lowerer 实例 = 一次降低会话。temp/label 计数器封在实例里, 保证「同一棵树降两次
// 得到同样的编号」—— 这是对拍可复现的前提(不能用全局计数器, 否则跑顺序会污染编号)。

interface LowerOptions {
  // shortCircuit=false 时, && / || 按普通 binary 降低(故意制造 §5 的失败模式)。
  readonly shortCircuit: boolean;
}

class Lowerer {
  private readonly out: Instr[] = [];
  private tempCounter = 0;
  private labelCounter = 0;
  private readonly opts: LowerOptions;

  constructor(opts: LowerOptions = { shortCircuit: true }) {
    this.opts = opts;
  }

  lower(program: Program): readonly Instr[] {
    for (const s of program) this.lowerStmt(s);
    return this.out;
  }

  private freshTemp(): string {
    return `%t${this.tempCounter++}`;
  }
  private freshLabel(hint: string): string {
    return `L${this.labelCounter++}_${hint}`;
  }
  private emit(i: Instr): void {
    this.out.push(i);
  }

  private lowerStmt(s: Stmt): void {
    switch (s.kind) {
      case "Let":
        // let x = e  →  算出 e 到某操作数, copy 进具名变量 x。
        this.emit({ kind: "copy", dst: s.name, src: this.lowerExpr(s.init) });
        return;
      case "Print":
        this.emit({ kind: "print", src: this.lowerExpr(s.value) });
        return;
      case "Return":
        this.emit({ kind: "return", src: s.value === undefined ? undefined : this.lowerExpr(s.value) });
        return;
      case "ExprStmt":
        // 表达式语句: 求值取副作用, 丢弃结果。仍需 lower(里面可能有 call / assign)。
        this.lowerExpr(s.expr);
        return;
      case "If":
        this.lowerIf(s.cond, s.then, s.else);
        return;
      case "While":
        this.lowerWhile(s.cond, s.body);
        return;
      case "Block":
        for (const inner of s.body) this.lowerStmt(inner);
        return;
      default: {
        const _never: never = s;
        throw new Error(`lowerStmt: unhandled ${JSON.stringify(_never)}`);
      }
    }
  }

  // if c { then } [else { else }]  →
  //   cond_jump c -> Lthen, Lelse
  //   Lthen: <then>; jump Lend
  //   Lelse: <else>
  //   Lend:
  private lowerIf(cond: Expr, then: readonly Stmt[], els?: readonly Stmt[]): void {
    const c = this.lowerExpr(cond);
    const lThen = this.freshLabel("then");
    const lElse = this.freshLabel("else");
    const lEnd = this.freshLabel("endif");
    this.emit({ kind: "cond_jump", cond: c, ifTrue: lThen, ifFalse: els ? lElse : lEnd });
    this.emit({ kind: "label", name: lThen });
    for (const s of then) this.lowerStmt(s);
    this.emit({ kind: "jump", target: lEnd });
    if (els) {
      this.emit({ kind: "label", name: lElse });
      for (const s of els) this.lowerStmt(s);
      this.emit({ kind: "jump", target: lEnd });
    }
    this.emit({ kind: "label", name: lEnd });
  }

  // while c { body }  →
  //   Lcond: cond_jump c -> Lbody, Lend
  //   Lbody: <body>; jump Lcond     ← 这条 jump Lcond 就是「回边(back-edge)」
  //   Lend:
  // 回边 = 目标在自己之前的跳转, 它是「循环」在 CFG 上的唯一标志(后端据此识别循环、
  // 做循环不变量外提等优化)。
  private lowerWhile(cond: Expr, body: readonly Stmt[]): void {
    const lCond = this.freshLabel("cond");
    const lBody = this.freshLabel("body");
    const lEnd = this.freshLabel("endwhile");
    this.emit({ kind: "label", name: lCond });
    const c = this.lowerExpr(cond);
    this.emit({ kind: "cond_jump", cond: c, ifTrue: lBody, ifFalse: lEnd });
    this.emit({ kind: "label", name: lBody });
    for (const s of body) this.lowerStmt(s);
    this.emit({ kind: "jump", target: lCond }); // back-edge
    this.emit({ kind: "label", name: lEnd });
  }

  // 表达式降低: 返回「持有结果的操作数」。叶子(字面量/变量)直接返回, 不浪费 temp;
  // 复合表达式 emit 指令并返回新 temp。
  private lowerExpr(e: Expr): Operand {
    switch (e.kind) {
      case "Literal":
        return { kind: "const", value: e.value };
      case "Var":
        return { kind: "temp", name: e.name }; // 具名变量在 IR 里和 temp 同构(都按名引用)
      case "Assign": {
        const val = this.lowerExpr(e.value);
        this.emit({ kind: "copy", dst: e.name, src: val });
        return { kind: "temp", name: e.name }; // 赋值表达式的值 = 被赋的变量
      }
      case "Unary": {
        const o = this.lowerExpr(e.operand);
        const dst = this.freshTemp();
        this.emit({ kind: "unary", dst, op: e.op, operand: o });
        return { kind: "temp", name: dst };
      }
      case "Binary":
        return this.lowerBinary(e);
      case "Call": {
        const args = e.args.map((a) => this.lowerExpr(a));
        const callee = e.callee.kind === "Var" ? e.callee.name : "<expr-callee>";
        const dst = this.freshTemp();
        this.emit({ kind: "call", dst, callee, args });
        return { kind: "temp", name: dst };
      }
      case "Fn":
        // 函数表达式的 IR 降低(闭包/独立 CFG)超出本章范围。本章树里不含 Fn 节点。
        throw new Error("lowerExpr: Fn lowering is out of scope for stage05");
      default: {
        const _never: never = e;
        throw new Error(`lowerExpr: unhandled ${JSON.stringify(_never)}`);
      }
    }
  }

  private lowerBinary(e: Extract<Expr, { kind: "Binary" }>): Operand {
    const isShortCircuit = e.op === "&&" || e.op === "||";
    if (isShortCircuit && this.opts.shortCircuit) {
      return this.lowerShortCircuit(e.op, e.left, e.right);
    }
    // 普通二元(含「失败模式」下的 && / ||): 左右都先算成操作数, 再一条 binary。
    // 注意: 这里对 && / || 是错的 —— 右侧无条件被求值, 副作用必然发生(见 §5)。
    const left = this.lowerExpr(e.left);
    const right = this.lowerExpr(e.right);
    const dst = this.freshTemp();
    this.emit({ kind: "binary", dst, op: e.op, left, right });
    return { kind: "temp", name: dst };
  }

  // 正确的短路降低: 用 cond_jump 跳过右侧。
  //   a && b  →  t = a; cond_jump t -> Lrhs, Lend(t 已是 false, 不算 b)
  //             Lrhs: t = b
  //             Lend: (结果在 t)
  //   a || b  →  t = a; cond_jump t -> Lend, Lrhs(t 已是 true, 不算 b)
  //             Lrhs: t = b
  //             Lend:
  // 关键: 右侧 b 的指令落在 Lrhs 块里, 只有跳进来才执行 —— 副作用条件化了。
  private lowerShortCircuit(op: string, lhs: Expr, rhs: Expr): Operand {
    const result = this.freshTemp();
    const lRhs = this.freshLabel("rhs");
    const lEnd = this.freshLabel("scend");
    const l = this.lowerExpr(lhs);
    this.emit({ kind: "copy", dst: result, src: l });
    const cond: Operand = { kind: "temp", name: result };
    if (op === "&&") {
      this.emit({ kind: "cond_jump", cond, ifTrue: lRhs, ifFalse: lEnd });
    } else {
      this.emit({ kind: "cond_jump", cond, ifTrue: lEnd, ifFalse: lRhs });
    }
    this.emit({ kind: "label", name: lRhs });
    const r = this.lowerExpr(rhs);
    this.emit({ kind: "copy", dst: result, src: r });
    this.emit({ kind: "jump", target: lEnd });
    this.emit({ kind: "label", name: lEnd });
    return cond;
  }
}

// ---------- IR 文本化(打印 + 对拍) ----------

function fmtOperand(o: Operand): string {
  if (o.kind === "const") {
    if (typeof o.value === "string") return JSON.stringify(o.value);
    if (typeof o.value === "boolean") return o.value ? "#t" : "#f";
    return String(o.value);
  }
  return o.name;
}

function fmtInstr(i: Instr): string {
  switch (i.kind) {
    case "const":
      return `  ${i.dst} = const ${typeof i.value === "string" ? JSON.stringify(i.value) : i.value}`;
    case "copy":
      return `  ${i.dst} = ${fmtOperand(i.src)}`;
    case "binary":
      return `  ${i.dst} = ${fmtOperand(i.left)} ${i.op} ${fmtOperand(i.right)}`;
    case "unary":
      return `  ${i.dst} = ${i.op}${fmtOperand(i.operand)}`;
    case "call":
      return `  ${i.dst} = call ${i.callee}(${i.args.map(fmtOperand).join(", ")})`;
    case "label":
      return `${i.name}:`;
    case "jump":
      return `  jump ${i.target}`;
    case "cond_jump":
      return `  cond_jump ${fmtOperand(i.cond)} ? ${i.ifTrue} : ${i.ifFalse}`;
    case "print":
      return `  print ${fmtOperand(i.src)}`;
    case "return":
      return `  return${i.src ? " " + fmtOperand(i.src) : ""}`;
    default: {
      const _never: never = i;
      throw new Error(`fmtInstr: unhandled ${JSON.stringify(_never)}`);
    }
  }
}

// 单行紧凑形态, 给 assertEq 对拍用(整段 IR 压成一行可 diff 文本)。
function irToLine(instrs: readonly Instr[]): string {
  return instrs.map((i) => fmtInstr(i).trim()).join(" ; ");
}

function printIr(title: string, instrs: readonly Instr[]): void {
  console.log(`\n--- ${title} (${instrs.length} 条指令) ---`);
  for (const i of instrs) console.log(fmtInstr(i));
}

// ---------- 基本块 + CFG ----------
//
// 基本块(basic block) = 一段「要么全执行、要么全不执行」的指令: 单入口(块首)、单出口
// (块尾)。切块规则(教科书 leader 法):
//   leader(块首)是: ① 第一条指令; ② 任何 label; ③ 任何跳转的「下一条」指令。
// 一个 leader 到下一个 leader 之前就是一个块。

interface BasicBlock {
  readonly id: number;
  readonly label?: string; // 若块首是 label
  readonly start: number; // 在原指令数组中的 [start, end) 区间
  readonly end: number;
  readonly instrs: readonly Instr[];
  readonly succ: number[]; // 后继块 id
  readonly pred: number[]; // 前驱块 id
}

function buildCfg(instrs: readonly Instr[]): BasicBlock[] {
  // 1) 标记所有 leader 的下标。
  const leaders = new Set<number>();
  if (instrs.length > 0) leaders.add(0);
  for (let i = 0; i < instrs.length; i++) {
    const ins = instrs[i];
    if (ins.kind === "label") leaders.add(i);
    if (ins.kind === "jump" || ins.kind === "cond_jump" || ins.kind === "return") {
      if (i + 1 < instrs.length) leaders.add(i + 1); // terminator 的下一条是新块起点
    }
  }
  const sortedLeaders = [...leaders].sort((a, b) => a - b);

  // 2) 按 leader 切块。label → blockId 索引, 给跳转解析后继用。
  const blocks: BasicBlock[] = [];
  const labelToBlock = new Map<string, number>();
  for (let b = 0; b < sortedLeaders.length; b++) {
    const start = sortedLeaders[b];
    const end = b + 1 < sortedLeaders.length ? sortedLeaders[b + 1] : instrs.length;
    const slice = instrs.slice(start, end);
    const first = slice[0];
    const label = first && first.kind === "label" ? first.name : undefined;
    const block: BasicBlock = { id: b, label, start, end, instrs: slice, succ: [], pred: [] };
    blocks.push(block);
    if (label) labelToBlock.set(label, b);
  }

  // 3) 连边。块尾决定后继:
  //    - jump L        → 唯一后继 = block(L)
  //    - cond_jump ?L:M → 两个后继 = block(L), block(M)
  //    - return        → 无后继(函数出口)
  //    - 其它(fall-through) → 后继 = 下一个块(顺序流入)
  const link = (from: number, to: number): void => {
    if (!blocks[from].succ.includes(to)) blocks[from].succ.push(to);
    if (!blocks[to].pred.includes(from)) blocks[to].pred.push(from);
  };
  const resolve = (label: string): number => {
    const id = labelToBlock.get(label);
    if (id === undefined) throw new Error(`CFG: jump to unknown label ${label}`); // 失败模式: 悬空 label
    return id;
  };
  for (const block of blocks) {
    const last = block.instrs[block.instrs.length - 1];
    if (!last) continue;
    if (last.kind === "jump") {
      link(block.id, resolve(last.target));
    } else if (last.kind === "cond_jump") {
      link(block.id, resolve(last.ifTrue));
      link(block.id, resolve(last.ifFalse));
    } else if (last.kind === "return") {
      // 无后继。
    } else if (block.id + 1 < blocks.length) {
      link(block.id, block.id + 1); // fall-through
    }
  }
  return blocks;
}

function printCfg(title: string, blocks: readonly BasicBlock[]): void {
  console.log(`\n--- CFG: ${title} (${blocks.length} 个基本块) ---`);
  for (const b of blocks) {
    const name = b.label ?? `(no-label)`;
    const succ = b.succ.length ? b.succ.map((s) => `B${s}`).join(",") : "—";
    const pred = b.pred.length ? b.pred.map((p) => `B${p}`).join(",") : "—";
    console.log(`  B${b.id} ${name}  指令[${b.start},${b.end})  succ={${succ}}  pred={${pred}}`);
  }
}

// 检测回边: succ 指向「id ≤ 自己」的块 = 回边(循环存在的标志)。
function findBackEdges(blocks: readonly BasicBlock[]): Array<{ from: number; to: number }> {
  const edges: Array<{ from: number; to: number }> = [];
  for (const b of blocks) {
    for (const s of b.succ) {
      if (s <= b.id) edges.push({ from: b.id, to: s });
    }
  }
  return edges;
}

// ---------- AST 节点计数(给「展开比」用) ----------

function countAstNodes(node: Expr | Stmt | Program): number {
  if (Array.isArray(node)) return (node as readonly Stmt[]).reduce((acc, s) => acc + countAstNodes(s), 0);
  const n = node as Expr | Stmt;
  switch (n.kind) {
    case "Literal":
    case "Var":
      return 1;
    case "Unary":
      return 1 + countAstNodes(n.operand);
    case "Binary":
      return 1 + countAstNodes(n.left) + countAstNodes(n.right);
    case "Assign":
      return 1 + countAstNodes(n.value);
    case "Call":
      return 1 + countAstNodes(n.callee) + n.args.reduce((a, x) => a + countAstNodes(x), 0);
    case "Fn":
      return 1 + countAstNodes(n.body as readonly Stmt[]);
    case "Let":
      return 1 + countAstNodes(n.init);
    case "Print":
      return 1 + countAstNodes(n.value);
    case "Return":
      return 1 + (n.value ? countAstNodes(n.value) : 0);
    case "ExprStmt":
      return 1 + countAstNodes(n.expr);
    case "If":
      return 1 + countAstNodes(n.cond) + countAstNodes(n.then as readonly Stmt[]) + (n.else ? countAstNodes(n.else as readonly Stmt[]) : 0);
    case "While":
      return 1 + countAstNodes(n.cond) + countAstNodes(n.body as readonly Stmt[]);
    case "Block":
      return 1 + countAstNodes(n.body as readonly Stmt[]);
    default: {
      const _never: never = n;
      throw new Error(`countAstNodes: unhandled ${JSON.stringify(_never)}`);
    }
  }
}

// ---------- 极简 IR 解释器(只为 §5 用「副作用次数」证明短路被破坏) ----------
//
// 不是完整 VM —— 只支持本节那棵小树需要的指令子集。它的唯一用途: 真的跑一遍 IR, 数
// 「有副作用的调用执行了几次」, 把「短路 vs 非短路」的语义差变成一个可打印的真实数字。
// 失败模式不能靠嘴说「右边会被算」, 要靠跑出来的计数证明。

interface InterpResult {
  readonly value: boolean | number | string;
  readonly sideEffectCount: number; // sideEffect() 被调用的次数
}

function interpret(instrs: readonly Instr[]): InterpResult {
  const env = new Map<string, boolean | number | string>();
  let sideEffectCount = 0;
  // label → 指令下标, 供跳转。
  const labelIndex = new Map<string, number>();
  instrs.forEach((ins, idx) => {
    if (ins.kind === "label") labelIndex.set(ins.name, idx);
  });

  const read = (o: Operand): boolean | number | string => {
    if (o.kind === "const") return o.value;
    const val = env.get(o.name);
    if (val === undefined) throw new Error(`interpret: read undefined ${o.name}`);
    return val;
  };

  let pc = 0;
  let lastValue: boolean | number | string = false;
  let guard = 0; // 防失控循环: 本节小树指令数极少, 超界即 bug。
  while (pc < instrs.length) {
    if (guard++ > 10000) throw new Error("interpret: runaway (infinite loop in IR?)");
    const ins = instrs[pc];
    switch (ins.kind) {
      case "const":
        env.set(ins.dst, ins.value);
        lastValue = ins.value;
        pc++;
        break;
      case "copy": {
        const val = read(ins.src);
        env.set(ins.dst, val);
        lastValue = val;
        pc++;
        break;
      }
      case "binary": {
        // 只支持本节失败模式那棵树需要的 && / || —— 注意这条 binary 是「错误降低」的
        // 产物: 执行到这里时右侧 call 早已在上一条 emit 时被求值过了, 副作用已发生。
        // 这条 binary 只是事后把两个布尔合并, 救不回已经发生的副作用。
        const l = read(ins.left);
        const r = read(ins.right);
        const val = ins.op === "&&" ? Boolean(l) && Boolean(r) : ins.op === "||" ? Boolean(l) || Boolean(r) : (() => { throw new Error(`interpret: unsupported binary op ${ins.op}`); })();
        env.set(ins.dst, val);
        lastValue = val;
        pc++;
        break;
      }
      case "call": {
        // 本节唯一的 callee 是 sideEffect(), 它返回 true 并把计数 +1。
        if (ins.callee === "sideEffect") {
          sideEffectCount++;
          env.set(ins.dst, true);
          lastValue = true;
        } else {
          throw new Error(`interpret: unknown callee ${ins.callee}`);
        }
        pc++;
        break;
      }
      case "cond_jump": {
        const c = read(ins.cond);
        pc = labelIndex.get(c ? ins.ifTrue : ins.ifFalse)!;
        break;
      }
      case "jump":
        pc = labelIndex.get(ins.target)!;
        break;
      case "label":
        pc++;
        break;
      default:
        throw new Error(`interpret: unsupported instr ${ins.kind} (interpreter is intentionally minimal)`);
    }
  }
  return { value: lastValue, sideEffectCount };
}

// ====================================================================
// main: 顺序跑 §1-§5。每段都打印「代码真算出来的」数字。
// ====================================================================

function main(): void {
  console.log("================ 第 5 章 stage05-ir: AST → 线性三地址 IR ================");

  // ---- §1 降低 fib / loopsum + 对拍 ----
  console.log("\n【§1】把 fib 与 loopsum 降为三地址 IR(临时 %t / label / cond_jump)");
  const fibBody = buildFibBody();
  const loopSum = buildLoopSum();

  const fibIr = new Lowerer().lower(fibBody);
  const loopIr = new Lowerer().lower(loopSum);

  console.log("\nfib 体 AST(S 表达式):");
  console.log("  " + astToSExpr(fibBody));
  printIr("fib 体 IR", fibIr);

  console.log("\nloopsum AST(S 表达式):");
  console.log("  " + astToSExpr(loopSum));
  printIr("loopsum IR", loopIr);

  // 对拍: 降低是确定性的 —— 重跑一次必须逐字节相同。这就是「同一棵树永远降出同一序列」。
  const fibIrAgain = new Lowerer().lower(buildFibBody());
  assertEq(irToLine(fibIrAgain), irToLine(fibIr), "fib IR 降低确定性");
  const loopIrAgain = new Lowerer().lower(buildLoopSum());
  assertEq(irToLine(loopIrAgain), irToLine(loopIr), "loopsum IR 降低确定性");
  console.log("\n  [对拍] 同一棵树降低两次, IR 文本逐字相同 (确定性 OK)");

  // ---- §2 控制流降低 + while 回边 ----
  console.log("\n【§2】控制流降低: if/while → label + cond_jump; while 的回边");
  const loopBlocks = buildCfg(loopIr);
  const backEdges = findBackEdges(loopBlocks);
  console.log("  loopsum 的 while 在 IR 里的形态: Lcond 求条件 → cond_jump → 块尾 jump 回 Lcond");
  for (const e of backEdges) {
    const fromBlock = loopBlocks[e.from];
    const toBlock = loopBlocks[e.to];
    console.log(`  回边(back-edge): B${e.from}(${fromBlock.label ?? "?"}) → B${e.to}(${toBlock.label ?? "?"})  ← 这条边证明这是循环`);
  }
  assertEq(backEdges.length, 1, "loopsum 恰好 1 条回边(单个 while)");

  // ---- §3 基本块 + CFG ----
  console.log("\n【§3】基本块切分 + CFG 前驱/后继");
  const fibBlocks = buildCfg(fibIr);
  printCfg("fib 体", fibBlocks);
  printCfg("loopsum", loopBlocks);

  // 校验不变量: succ/pred 互为反向。b∈succ(a) ⟺ a∈pred(b)。
  const checkCfgConsistency = (blocks: readonly BasicBlock[], name: string): void => {
    for (const a of blocks) {
      for (const s of a.succ) {
        if (!blocks[s].pred.includes(a.id)) {
          throw new Error(`CFG ${name} 不一致: B${a.id}→B${s} 缺反向 pred`);
        }
      }
    }
  };
  checkCfgConsistency(fibBlocks, "fib");
  checkCfgConsistency(loopBlocks, "loopsum");
  console.log("\n  [不变量] succ/pred 互为反向, 两个 CFG 均一致 (OK)");

  // ---- §4 AST 节点数 vs IR 指令数 ----
  console.log("\n【§4】降低展开比: AST 节点数 vs IR 指令数");
  const report = (name: string, ast: Program, ir: readonly Instr[]): void => {
    const astN = countAstNodes(ast);
    const irN = ir.length;
    const ratio = irN / astN;
    console.log(`  ${name.padEnd(9)} AST ${fmtNum(astN)} 节点 → IR ${fmtNum(irN)} 条指令  (展开比 ${ratio.toFixed(2)}x)`);
  };
  report("fib", fibBody, fibIr);
  report("loopsum", loopSum, loopIr);
  console.log("  诚实解读: 这里展开比 < 1(IR 指令比 AST 节点少), 不是『膨胀』。原因是三地址");
  console.log("        IR 把字面量/变量这些叶子节点收进『操作数』而非独立指令 —— AST 每个");
  console.log("        leaf(每个 n / 1 / 2)都算 1 个节点, 但 IR 里它们只是 binary 的操作数。");
  console.log("        反方向的展开来自控制流: 一个 if/while 节点会摊成 cond_jump + 多个 label");
  console.log("        + jump。叶子密集的小程序前者占优(比 < 1); 控制流密集的程序后者占优(比 > 1)。");
  console.log("        绝对值依赖『AST 节点怎么数 / IR 含不含 label』两种口径, 偏 toy; 可迁移的是");
  console.log("        相对趋势: 叶子越多越接近 1, 分支/循环越多越大于 1。");

  // ---- §5 失败模式: 短路 && 被当普通二元算符 ----
  console.log("\n【§5】失败模式: 短路 && 不拆跳转, 当普通 binary 降低");
  console.log("  测试表达式: false && sideEffect()");
  console.log("  正确语义: 左侧 false ⇒ 右侧 sideEffect() 不该求值 ⇒ 副作用计数应为 0。");

  // 树: (false && sideEffect())  —— 一条 ExprStmt 包一个 && 二元表达式。
  const buildAndTree = (): Program => [exprS(bin("&&", lit(false), call(v("sideEffect"), [])))];

  const correctIr = new Lowerer({ shortCircuit: true }).lower(buildAndTree());
  const brokenIr = new Lowerer({ shortCircuit: false }).lower(buildAndTree());

  printIr("正确降低(短路 → cond_jump 跳过右侧)", correctIr);
  printIr("错误降低(&& 当普通 binary, 左右都先算)", brokenIr);

  const correctRun = interpret(correctIr);
  const brokenRun = interpret(brokenIr);
  console.log(`\n  正确降低: sideEffect() 实际执行 ${correctRun.sideEffectCount} 次  (期望 0)`);
  console.log(`  错误降低: sideEffect() 实际执行 ${brokenRun.sideEffectCount} 次  (短路语义被破坏!)`);

  assertEq(correctRun.sideEffectCount, 0, "短路降低: 右侧副作用应为 0");
  assertEq(brokenRun.sideEffectCount, 1, "非短路降低: 右侧副作用被错误触发(失败模式实锤)");
  console.log("\n  结论: 数字证明 —— 把 && 当普通算符降低, 右侧副作用从『不该发生』变成『发生 1 次』。");
  console.log("        短路必须在 IR 层用 cond_jump 表达, 不能留到运算符层。");

  console.log("\n================ stage05-ir 全部通过 ================");
}

main();
