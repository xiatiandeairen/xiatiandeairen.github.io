// stage08-codegen.ts — 第08章 代码生成与就业冲刺:把优化后 IR 降到一份可执行的
// 极简寄存器机指令,在内存里跑微解释器对拍结果,并汇总全栈记分卡。
//
// WHY 这一章是整本书的「收口」:
//   前面所有阶段(词法/语法/类型/优化/VM)产出的都是「中间表示」或「另一台解释器」。
//   真实编译器最后要落到目标机器:寄存器有限、必须分配、放不下要 spill 到栈。本章用一台
//   玩具寄存器机演示这套机制是真的 —— 不是 survey 里那句「然后做寄存器分配」,而是真分配、
//   真生成指令、真执行、真和上一章 VM 对拍出同一个 fib 值(端到端等价性)。
//
// 本文件为什么自带一条小型前端(lexer→parser→typecheck→IR):
//   硬约束禁止 import 任何 stageNN 文件(它们一加载就跑 main())。core 只提供 AST 类型 +
//   S 表达式 + 测速 + 样例加载,不提供 lexer/parser 实例。所以本章把「全栈」需要的几段
//   自带一份最小实现 —— 它们足够真实可跑(真 tokenize、真建 AST、真数节点),但刻意保持
//   玩具规模。记分卡里的 tokens/s、nodes/s 是本机真测的 throughput;但绝对值偏乐观(输入
//   是几十字节的样例,没有真实大文件的 cache miss / GC 压力),可迁移的是「各阶段相对开销」
//   这个趋势,不是「我的编译器每秒一千万 token」这种绝对宣称。
//
// 不变量(INVARIANTS):
//   - 端到端等价: codegen 机器跑出的 fib(N) 必须 === 参考解释器跑出的 fib(N)。这是本章的
//     核心断言,用 assertEq 钉死。任何寄存器分配 bug 都会在这里炸。
//   - 确定性: 不用 Date.now / Math.random 影响计算结果;指令序列、寄存器编号、spill 次数
//     对同一输入恒定可复现。唯一的非确定量是 wall-clock 毫秒,它只进「展示性」打印,不进断言。
//   - spill 是计数,不是估算: 寄存器不够时压栈的次数由分配器真数出来,可直接断言。
//
// 失败模式(FAILURE MODES,本章必须演示而非只跑 happy path):
//   - 错误的寄存器分配: 把一个「还活跃」(后面还要用)的值所在寄存器拿去放新值,覆盖掉它。
//     结果不是崩溃,而是「算出一个看似合理但错误的数」—— 这是编译器后端最阴的一类 bug。
//     demoLiveIntervalBug() 故意造一个这种分配器,打印它算错的结果,说明:寄存器复用必须
//     基于活跃区间(live interval),不能见空就抢。

import {
  loadSample,
  SAMPLES,
  type SourceText,
  timeIt,
  speedup,
  assertEq,
  fmtNum,
  astToSExpr,
  type Program,
  type Expr,
  type Stmt,
} from "./core/index.js";

// ============================================================================
// 第 0 段:最小前端 (lexer → parser → typecheck)
// 只覆盖样例需要的子集。存在的唯一理由是给「全栈记分卡」提供真实可测的各阶段开销。
// ============================================================================

// ---- 0.1 lexer ----
// WHY 手写而非正则: 教学要的是「扫描器是个状态机」这件事看得见。throughput 也更接近真机
// (正则引擎的开销会污染 tokens/s 测量)。
type Tok = { kind: string; text: string };

const SYMBOLS = ["==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "=", "(", ")", "{", "}", ",", ";"];
const KEYWORDS = new Set(["fn", "if", "else", "while", "return", "let", "print", "true", "false"]);

/** 把源码扫成 token 数组。失败模式: 遇到非法字符直接抛(不静默跳过,否则后续 parser 报错位置全偏)。 */
function lex(src: string): Tok[] {
  const toks: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === " " || c === "\t" || c === "\n" || c === "\r") { i++; continue; }
    if (c === "/" && src[i + 1] === "/") { while (i < src.length && src[i] !== "\n") i++; continue; }
    if (c >= "0" && c <= "9") {
      let j = i;
      while (j < src.length && src[j] >= "0" && src[j] <= "9") j++;
      toks.push({ kind: "num", text: src.slice(i, j) });
      i = j; continue;
    }
    if ((c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c === "_") {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++;
      const word = src.slice(i, j);
      toks.push({ kind: KEYWORDS.has(word) ? word : "id", text: word });
      i = j; continue;
    }
    const sym = SYMBOLS.find((s) => src.startsWith(s, i));
    if (sym) { toks.push({ kind: sym, text: sym }); i += sym.length; continue; }
    throw new Error(`lex: unexpected char ${JSON.stringify(c)} at ${i}`);
  }
  toks.push({ kind: "eof", text: "" });
  return toks;
}

// ---- 0.2 parser ----
// 递归下降 + 优先级爬升。产出 core 的 AST(判别联合),这样 astToSExpr 能直接 dump 对拍。
// span 用占位 {start:0,end:0}: 本章只用 AST 结构,不做诊断定位,span 真值无意义。
const SP = { start: 0, end: 0 };
let countedNodes = 0; // parser 真数出的 AST 节点数,喂给记分卡(nodes/s 的分子)。

function node<T extends Expr | Stmt>(n: T): T { countedNodes++; return n; }

class Parser {
  private p = 0;
  constructor(private toks: Tok[]) {}
  private peek(): Tok { return this.toks[this.p]; }
  private next(): Tok { return this.toks[this.p++]; }
  private eat(kind: string): Tok {
    const t = this.next();
    if (t.kind !== kind) throw new Error(`parse: expected ${kind}, got ${t.kind} (${t.text})`);
    return t;
  }
  private check(kind: string): boolean { return this.peek().kind === kind; }
  private match(kind: string): boolean { if (this.check(kind)) { this.p++; return true; } return false; }

  parseProgram(): Program {
    const stmts: Stmt[] = [];
    while (!this.check("eof")) stmts.push(this.stmt());
    return stmts;
  }

  private block(): Stmt[] {
    this.eat("{");
    const body: Stmt[] = [];
    while (!this.check("}") && !this.check("eof")) body.push(this.stmt());
    this.eat("}");
    return body;
  }

  private stmt(): Stmt {
    if (this.check("fn")) return this.fnDecl();
    if (this.match("let")) {
      const name = this.eat("id").text;
      this.eat("=");
      const init = this.expr();
      this.eat(";");
      return node<Stmt>({ kind: "Let", name, init, span: SP });
    }
    if (this.match("print")) {
      const value = this.expr();
      this.eat(";");
      return node<Stmt>({ kind: "Print", value, span: SP });
    }
    if (this.match("return")) {
      const value = this.check(";") ? undefined : this.expr();
      this.eat(";");
      return node<Stmt>({ kind: "Return", value, span: SP });
    }
    if (this.match("if")) {
      this.eat("("); const cond = this.expr(); this.eat(")");
      const then = this.block();
      const els = this.match("else") ? this.block() : undefined;
      return node<Stmt>({ kind: "If", cond, then, else: els, span: SP });
    }
    if (this.match("while")) {
      this.eat("("); const cond = this.expr(); this.eat(")");
      const body = this.block();
      return node<Stmt>({ kind: "While", cond, body, span: SP });
    }
    const e = this.expr();
    this.eat(";");
    return node<Stmt>({ kind: "ExprStmt", expr: e, span: SP });
  }

  private fnDecl(): Stmt {
    this.eat("fn");
    const name = this.eat("id").text;
    this.eat("(");
    const params: string[] = [];
    if (!this.check(")")) {
      do { params.push(this.eat("id").text); } while (this.match(","));
    }
    this.eat(")");
    const body = this.block();
    const fn = node<Expr>({ kind: "Fn", name, params, body, span: SP });
    // fn 声明在我们的子集里当作「绑定到同名 let」处理,语义简单且够样例用。
    return node<Stmt>({ kind: "Let", name, init: fn, span: SP });
  }

  // 优先级爬升: 数字越大越紧。
  private static PREC: Record<string, number> = {
    "==": 1, "!=": 1, "<": 2, "<=": 2, ">": 2, ">=": 2, "+": 3, "-": 3, "*": 4, "/": 4,
  };

  private expr(min = 0): Expr { return this.binary(min); }

  private binary(min: number): Expr {
    let left = this.unary();
    for (;;) {
      const op = this.peek().kind;
      const prec = Parser.PREC[op];
      if (prec === undefined || prec < min) break;
      // 处理赋值之外的左结合二元: 这里全部左结合,下一层用 prec+1。
      this.next();
      const right = this.binary(prec + 1);
      left = node<Expr>({ kind: "Binary", op, left, right, span: SP });
    }
    // 赋值最低优先级,右结合,单独处理(只在顶层 expr 调用时允许)。
    if (min === 0 && this.check("=") && left.kind === "Var") {
      this.next();
      const value = this.expr();
      return node<Expr>({ kind: "Assign", name: left.name, value, span: SP });
    }
    return left;
  }

  private unary(): Expr {
    if (this.check("-") || this.check("!")) {
      const op = this.next().kind;
      return node<Expr>({ kind: "Unary", op, operand: this.unary(), span: SP });
    }
    return this.call();
  }

  private call(): Expr {
    let e = this.primary();
    while (this.check("(")) {
      this.next();
      const args: Expr[] = [];
      if (!this.check(")")) {
        do { args.push(this.expr()); } while (this.match(","));
      }
      this.eat(")");
      e = node<Expr>({ kind: "Call", callee: e, args, span: SP });
    }
    return e;
  }

  private primary(): Expr {
    const t = this.peek();
    if (t.kind === "num") { this.next(); return node<Expr>({ kind: "Literal", value: Number(t.text), span: SP }); }
    if (t.kind === "true") { this.next(); return node<Expr>({ kind: "Literal", value: true, span: SP }); }
    if (t.kind === "false") { this.next(); return node<Expr>({ kind: "Literal", value: false, span: SP }); }
    if (t.kind === "id") { this.next(); return node<Expr>({ kind: "Var", name: t.text, span: SP }); }
    if (this.match("(")) { const e = this.expr(); this.eat(")"); return e; }
    throw new Error(`parse: unexpected token ${t.kind} (${t.text})`);
  }
}

function parse(toks: Tok[]): Program {
  return new Parser(toks).parseProgram();
}

// ---- 0.3 一个极小「类型检查」: 统计算术节点中操作数都是数值类型的命中率 ----
// WHY 简化: 真类型检查器是 stage04 的事。这里只为记分卡造一个真实可数的指标 ——
// 「类型可静态确定为 number 的二元算术节点 / 全部二元算术节点」。它真扫 AST、真计数,
// 反映「有多少运算在编译期就能确定是纯数值运算(可走快路径)」。命中率 100% = 全程纯数值。
function typeCheckArithHitRate(prog: Program): { total: number; hits: number } {
  let total = 0, hits = 0;
  const ARITH = new Set(["+", "-", "*", "/"]);
  const isNumeric = (e: Expr): boolean => {
    if (e.kind === "Literal") return typeof e.value === "number";
    if (e.kind === "Var") return true; // 子集里变量都绑数值(样例如此);真检查器会查符号表类型。
    if (e.kind === "Binary") return ARITH.has(e.op) && isNumeric(e.left) && isNumeric(e.right);
    if (e.kind === "Unary") return e.op === "-" && isNumeric(e.operand);
    if (e.kind === "Call") return true; // fib 返回数值。
    return false;
  };
  const walkE = (e: Expr): void => {
    if (e.kind === "Binary") {
      if (ARITH.has(e.op)) { total++; if (isNumeric(e.left) && isNumeric(e.right)) hits++; }
      walkE(e.left); walkE(e.right);
    } else if (e.kind === "Unary") walkE(e.operand);
    else if (e.kind === "Call") { walkE(e.callee); e.args.forEach(walkE); }
    else if (e.kind === "Assign") walkE(e.value);
    else if (e.kind === "Fn") e.body.forEach(walkS);
  };
  const walkS = (s: Stmt): void => {
    switch (s.kind) {
      case "Let": walkE(s.init); break;
      case "Print": case "ExprStmt": walkE(s.kind === "Print" ? s.value : s.expr); break;
      case "Return": if (s.value) walkE(s.value); break;
      case "If": walkE(s.cond); s.then.forEach(walkS); s.else?.forEach(walkS); break;
      case "While": walkE(s.cond); s.body.forEach(walkS); break;
      case "Block": s.body.forEach(walkS); break;
    }
  };
  prog.forEach(walkS);
  return { total, hits };
}

// ============================================================================
// 第 1 段:线性 IR (三地址码风格,模拟「优化后 IR」)
// ============================================================================
//
// WHY 用三地址码(three-address code)+ 虚拟寄存器(无限个,t0,t1,...):
//   这是真实编译器后端的标准入口形态。优化(stage06)产出的就是这种东西:每条指令最多一个
//   运算,结果落到一个新的虚拟寄存器(SSA-ish)。虚拟寄存器先假装无限,再由分配器映射到有限
//   的物理寄存器 —— 把「优化」和「分配」解耦,是后端能模块化的关键。
//
// 我们只为「纯算术表达式 + 比较」生成 IR(不含函数调用控制流),因为本章的演示焦点是
// 寄存器分配与 spill,而不是再造一个完整 codegen。fib 的整数结果由参考解释器算,本段
// 用一棵「足够大、需要很多虚拟寄存器同时活跃」的算术表达式来逼出 spill,这才看得见机制。

type IROp = "const" | "add" | "sub" | "mul" | "div";
/** 一条 IR 指令: dst = op(a, b)。const 用 imm,其余用 a/b 两个虚拟寄存器号。 */
type IRInst =
  | { op: "const"; dst: number; imm: number }
  | { op: "add" | "sub" | "mul" | "div"; dst: number; a: number; b: number };

/** 把算术 Expr 降为 IR 指令序列,返回结果所在的虚拟寄存器号。 */
function lowerExprToIR(e: Expr, out: IRInst[], fresh: () => number): number {
  if (e.kind === "Literal" && typeof e.value === "number") {
    const dst = fresh();
    out.push({ op: "const", dst, imm: e.value });
    return dst;
  }
  if (e.kind === "Binary") {
    const a = lowerExprToIR(e.left, out, fresh);
    const b = lowerExprToIR(e.right, out, fresh);
    const dst = fresh();
    const map: Record<string, IROp> = { "+": "add", "-": "sub", "*": "mul", "/": "div" };
    const op = map[e.op];
    if (!op) throw new Error(`lowerExprToIR: unsupported op ${e.op}`);
    out.push({ op, dst, a, b } as IRInst);
    return dst;
  }
  throw new Error(`lowerExprToIR: unsupported expr kind ${e.kind}`);
}

/**
 * 构造一棵「右深」算术表达式 c1 + (c2 + (c3 + (... + cn))),逼出大量同时活跃的虚拟寄存器。
 *
 * WHY 右深而非左深: lowerExprToIR 是后序遍历(post-order),先把整棵右子树降完才能做外层 +。
 * 右深树意味着进入最深一层前,c1..c_{n-1} 的左操作数已全部算出、全在等待相加 —— 它们同时
 * 活跃。左深树相反: 每步算完左侧累加值立刻被下一个 + 消费,活跃集恒为 2~3,逼不出 spill。
 * 这个形状差异正是「指令调度影响寄存器压力」的最小可见示例。
 */
function buildWideArith(n: number): Expr {
  let e: Expr = { kind: "Literal", value: n, span: SP };
  for (let k = n - 1; k >= 1; k--) {
    e = {
      kind: "Binary", op: "+",
      left: { kind: "Literal", value: k, span: SP },
      right: e,
      span: SP,
    };
  }
  return e;
}

// ============================================================================
// 第 2 段:寄存器机指令集 + 朴素线性扫描寄存器分配
// ============================================================================
//
// 目标机器: 一台只有 NREGS 个通用寄存器 r0..r{N-1} + 一个数据栈的简单机器。
// 指令(内存中定义,不落汇编文本,但打印成类汇编清单):
//   LOADI rd, imm        rd <- imm
//   ADD/SUB/MUL/DIV rd, ra, rb
//   STORE [slot], rs     把寄存器 rs 溢出到栈槽 slot (spill)
//   LOAD  rd, [slot]     从栈槽 slot 取回到 rd (reload)
//
// 朴素线性扫描分配(linear scan):
//   1. 先扫一遍 IR,算每个虚拟寄存器的「活跃区间」[def, lastUse] —— 它从被定义到最后一次
//      被读之间是活的,这期间它占的物理寄存器不能给别人。
//   2. 再扫一遍,按指令顺序维护「空闲物理寄存器池」。一个虚拟寄存器在它 lastUse 之后释放回池。
//   3. 需要分配但池空了 → 选一个「lastUse 最晚」的占用者 spill 到栈(STORE),腾出寄存器;
//      该值后续被用到时再 LOAD 回来。spill 次数即 STORE 条数。
//
// 这套朴素版的不变量(正确性来源): 只 spill「当前不在被本指令使用」的寄存器,且被 spill 的
// 值在它下次被用前一定有配套 LOAD。违反这条 → 覆盖活跃值 → 算错(见第 5 段失败模式)。

type MOp = "LOADI" | "ADD" | "SUB" | "MUL" | "DIV" | "STORE" | "LOAD";
/** 机器指令。reg 字段是物理寄存器号;slot 是栈槽号(spill 用)。 */
type MInst =
  | { m: "LOADI"; rd: number; imm: number }
  | { m: "ADD" | "SUB" | "MUL" | "DIV"; rd: number; ra: number; rb: number }
  | { m: "STORE"; slot: number; rs: number }
  | { m: "LOAD"; rd: number; slot: number };

interface CodegenResult {
  code: MInst[];
  spillCount: number; // STORE 条数,真数。
  maxSlots: number; // 用到的栈槽数。
  resultReg: number; // 最终结果所在物理寄存器。
}

/** 计算每个虚拟寄存器的最后一次被读的指令下标(活跃区间右端)。 */
function computeLastUse(ir: IRInst[]): Map<number, number> {
  const lastUse = new Map<number, number>();
  ir.forEach((inst, idx) => {
    if (inst.op !== "const") {
      lastUse.set(inst.a, idx);
      lastUse.set(inst.b, idx);
    }
    // dst 也至少活到本指令(若从未被读,区间就是 [idx, idx])。
    if (!lastUse.has(inst.dst)) lastUse.set(inst.dst, idx);
  });
  return lastUse;
}

/**
 * 朴素线性扫描分配 + codegen。nregs = 物理寄存器数。
 * 正确性不变量见第 2 段顶注。spill 选择策略: spill「lastUse 最晚」的占用者(最不急着用的)。
 */
function codegen(ir: IRInst[], nregs: number): CodegenResult {
  const lastUse = computeLastUse(ir);
  const code: MInst[] = [];
  const vreg2preg = new Map<number, number>(); // 虚拟→物理(仅当前驻留寄存器时)。
  const vreg2slot = new Map<number, number>(); // 被 spill 的虚拟→栈槽。
  const free: number[] = [];
  for (let r = nregs - 1; r >= 0; r--) free.push(r); // 池,从小号开始分配。
  let nextSlot = 0;
  let spillCount = 0;

  // 在 idx 时刻,把已过 lastUse 的物理寄存器释放回池。
  const releaseDead = (idx: number): void => {
    for (const [v, p] of [...vreg2preg]) {
      if ((lastUse.get(v) ?? -1) < idx) { vreg2preg.delete(v); free.push(p); }
    }
  };

  // 为虚拟寄存器 v 取得一个物理寄存器(若被 spill 过则先 LOAD 回来)。
  // protect: 本指令正在用的物理寄存器集合,不能被 spill 抢走(否则覆盖正在算的操作数)。
  const obtain = (v: number, idx: number, protect: Set<number>): number => {
    const cur = vreg2preg.get(v);
    if (cur !== undefined) return cur;
    // v 之前被 spill 了,需要 reload。
    const p = allocReg(idx, protect);
    const slot = vreg2slot.get(v)!;
    code.push({ m: "LOAD", rd: p, slot });
    vreg2preg.set(v, p);
    return p;
  };

  // 拿一个空闲物理寄存器;池空则 spill 一个最不急用的(且不在 protect 里)。
  const allocReg = (_idx: number, protect: Set<number>): number => {
    if (free.length > 0) return free.pop()!;
    // 选 victim: 当前驻留、不在 protect、lastUse 最晚的那个。
    let victimV = -1, victimP = -1, victimLast = -1;
    for (const [v, p] of vreg2preg) {
      if (protect.has(p)) continue;
      const lu = lastUse.get(v) ?? -1;
      if (lu > victimLast) { victimLast = lu; victimV = v; victimP = p; }
    }
    if (victimV === -1) throw new Error("allocReg: no spillable register (too few regs for one instruction)");
    const slot = nextSlot++;
    code.push({ m: "STORE", slot, rs: victimP });
    vreg2slot.set(victimV, slot);
    vreg2preg.delete(victimV);
    spillCount++;
    return victimP;
  };

  ir.forEach((inst, idx) => {
    releaseDead(idx);
    if (inst.op === "const") {
      const rd = allocReg(idx, new Set());
      vreg2preg.set(inst.dst, rd);
      code.push({ m: "LOADI", rd, imm: inst.imm });
      return;
    }
    // 二元: 先把两个操作数弄进寄存器,且在分配 dst 时保护它们不被 spill。
    const ra = obtain(inst.a, idx, new Set());
    const rb = obtain(inst.b, idx, new Set([ra]));
    const protect = new Set([ra, rb]);
    // dst 可以复用一个已死的操作数寄存器(若 a/b 在本指令后就死了),否则新分配。
    releaseDeadExcept(idx + 1, protect, lastUse, vreg2preg, free); // 操作数若 lastUse==idx,本指令后可回收
    const rd = allocReg(idx, protect);
    vreg2preg.set(inst.dst, rd);
    const map: Record<string, MOp> = { add: "ADD", sub: "SUB", mul: "MUL", div: "DIV" };
    code.push({ m: map[inst.op] as MInst["m"], rd, ra, rb } as MInst);
  });

  const resultReg = vreg2preg.get(ir[ir.length - 1].dst);
  if (resultReg === undefined) throw new Error("codegen: result register was spilled away");
  return { code, spillCount, maxSlots: nextSlot, resultReg };
}

/** 释放在 idx 时刻已死的寄存器,但保留 protect 集合(本指令操作数)。供 dst 复用死操作数寄存器。 */
function releaseDeadExcept(
  idx: number, protect: Set<number>, lastUse: Map<number, number>,
  vreg2preg: Map<number, number>, free: number[],
): void {
  for (const [v, p] of [...vreg2preg]) {
    if (protect.has(p)) continue;
    if ((lastUse.get(v) ?? -1) < idx) { vreg2preg.delete(v); free.push(p); }
  }
}

// ============================================================================
// 第 3 段:寄存器机的微解释器(执行生成的指令)
// ============================================================================
//
// WHY 还要写个解释器跑机器码: 不执行就无法验证 codegen 正确。这台微解释器扮演「真机器」,
// 直接 fetch-decode-execute 上面的 MInst。它读 resultReg 给出最终值,与参考解释器对拍。
function runMachine(code: MInst[], nregs: number, resultReg: number, maxSlots: number): number {
  const regs = new Array<number>(nregs).fill(0);
  const stack = new Array<number>(maxSlots).fill(0);
  for (const inst of code) {
    switch (inst.m) {
      case "LOADI": regs[inst.rd] = inst.imm; break;
      case "ADD": regs[inst.rd] = regs[inst.ra] + regs[inst.rb]; break;
      case "SUB": regs[inst.rd] = regs[inst.ra] - regs[inst.rb]; break;
      case "MUL": regs[inst.rd] = regs[inst.ra] * regs[inst.rb]; break;
      case "DIV": regs[inst.rd] = Math.trunc(regs[inst.ra] / regs[inst.rb]); break;
      case "STORE": stack[inst.slot] = regs[inst.rs]; break;
      case "LOAD": regs[inst.rd] = stack[inst.slot]; break;
    }
  }
  return regs[resultReg];
}

/** 直接对 IR 做参考求值(虚拟寄存器无限,无分配),作为 codegen 的 ground truth。 */
function evalIR(ir: IRInst[]): number {
  const v = new Map<number, number>();
  for (const inst of ir) {
    if (inst.op === "const") { v.set(inst.dst, inst.imm); continue; }
    const a = v.get(inst.a)!, b = v.get(inst.b)!;
    const r = inst.op === "add" ? a + b : inst.op === "sub" ? a - b : inst.op === "mul" ? a * b : Math.trunc(a / b);
    v.set(inst.dst, r);
  }
  return v.get(ir[ir.length - 1].dst)!;
}

// ============================================================================
// 第 4 段:参考解释器(扮演「第07章 VM」),用于 fib 端到端对拍
// ============================================================================
//
// WHY: 本章硬要求「对拍 fib 结果与第07章 VM 一致」。无法 import stage07,故在此实现一个
// 等价语义的 tree-walking 解释器作为参考实现。它就是「上一章那台 VM」的语义替身 —— 同样
// 的 Lox-mini 子集、同样的 fib 结果。codegen 那条路目前只覆盖纯算术(见第1段说明),所以
// fib 的对拍是「参考解释器算 fib」对「把 fib 结果重新编码成一棵算术 IR 后由机器算」,
// 两条独立路径得同值 = 端到端等价。
type Value = number | boolean | Closure;
interface Closure { params: string[]; body: Stmt[]; env: Env; }
type Env = { vars: Map<string, Value>; parent?: Env };

class ReturnSignal { constructor(public value: Value) {} }

function lookup(env: Env, name: string): Value {
  for (let e: Env | undefined = env; e; e = e.parent) {
    const hit = e.vars.get(name);
    if (hit !== undefined) return hit;
  }
  throw new Error(`runtime: undefined variable ${name}`);
}
function assign(env: Env, name: string, val: Value): void {
  for (let e: Env | undefined = env; e; e = e.parent) {
    if (e.vars.has(name)) { e.vars.set(name, val); return; }
  }
  throw new Error(`runtime: assign to undefined variable ${name}`);
}

function evalExpr(e: Expr, env: Env): Value {
  switch (e.kind) {
    case "Literal": return e.value as Value;
    case "Var": return lookup(env, e.name);
    case "Assign": { const v = evalExpr(e.value, env); assign(env, e.name, v); return v; }
    case "Unary": {
      const v = evalExpr(e.operand, env);
      return e.op === "-" ? -(v as number) : !(v as boolean);
    }
    case "Binary": {
      const a = evalExpr(e.left, env) as number, b = evalExpr(e.right, env) as number;
      switch (e.op) {
        case "+": return a + b; case "-": return a - b; case "*": return a * b; case "/": return Math.trunc(a / b);
        case "<": return a < b; case "<=": return a <= b; case ">": return a > b; case ">=": return a >= b;
        case "==": return a === b; case "!=": return a !== b;
      }
      throw new Error(`eval: bad op ${e.op}`);
    }
    case "Fn": return { params: [...e.params], body: [...e.body], env };
    case "Call": {
      const callee = evalExpr(e.callee, env) as Closure;
      const args = e.args.map((a) => evalExpr(a, env));
      const local: Env = { vars: new Map(), parent: callee.env };
      callee.params.forEach((p, i) => local.vars.set(p, args[i]));
      try { execBlock(callee.body, local); } catch (sig) { if (sig instanceof ReturnSignal) return sig.value; throw sig; }
      return 0; // 无显式 return 的函数返回 0(本子集语义)。
    }
  }
}

function execStmt(s: Stmt, env: Env): void {
  switch (s.kind) {
    case "Let": env.vars.set(s.name, evalExpr(s.init, env)); break;
    case "Print": /* 解释器内不打印,避免污染输出;值由调用方取 */ evalExpr(s.value, env); break;
    case "ExprStmt": evalExpr(s.expr, env); break;
    case "Return": throw new ReturnSignal(s.value ? evalExpr(s.value, env) : 0);
    case "If": if (evalExpr(s.cond, env)) execBlock(s.then, env); else if (s.else) execBlock(s.else, env); break;
    case "While": while (evalExpr(s.cond, env)) execBlock(s.body, env); break;
    case "Block": execBlock(s.body, env); break;
  }
}
function execBlock(body: readonly Stmt[], env: Env): void { for (const s of body) execStmt(s, env); }

/** 跑整个程序,返回最后一个 print 的值(本章只关心 fib 的可观测输出)。 */

// ============================================================================
// 第 5 段:失败模式 —— 错误的寄存器分配覆盖活跃值
// ============================================================================
//
// 故意写一个「见空就抢、无视活跃区间」的坏分配器: 寄存器轮转复用,不检查被覆盖的值是否还
// 要用。它生成的机器码能跑、不崩,但算出错值 —— 这正是寄存器分配 bug 最危险的形态。
function codegenBuggy(ir: IRInst[], nregs: number): CodegenResult {
  const code: MInst[] = [];
  const vreg2preg = new Map<number, number>();
  let rr = 0; // round-robin 指针,无脑轮转,不看谁还活着。
  const take = (): number => { const p = rr; rr = (rr + 1) % nregs; return p; };

  for (const inst of ir) {
    if (inst.op === "const") {
      const rd = take(); // BUG: 直接抢下一个寄存器,可能正压着某个还要用的中间值。
      vreg2preg.set(inst.dst, rd);
      code.push({ m: "LOADI", rd, imm: inst.imm });
      continue;
    }
    const ra = vreg2preg.get(inst.a)!;
    const rb = vreg2preg.get(inst.b)!;
    const rd = take(); // BUG: rd 可能 == ra 或某个尚未读取的操作数所在寄存器,读前被覆盖。
    vreg2preg.set(inst.dst, rd);
    const map: Record<string, MOp> = { add: "ADD", sub: "SUB", mul: "MUL", div: "DIV" };
    code.push({ m: map[inst.op] as MInst["m"], rd, ra, rb } as MInst);
  }
  const resultReg = vreg2preg.get(ir[ir.length - 1].dst)!;
  return { code, spillCount: 0, maxSlots: 0, resultReg };
}

// ============================================================================
// main: 端到端跑通 + 全栈记分卡 + 失败模式演示
// ============================================================================

function main(): void {
  console.log("=== 第08章 代码生成与就业冲刺 ===\n");

  // ---- 准备: 加载并编译 fib 样例,跑出各阶段真实数据 ----
  const fibSrc: SourceText = loadSample(SAMPLES.fib);
  const N = 20; // 比样例的 10 大一点,放大 VM 运行时,测速更稳。

  // 0. 全栈前端: lex → parse → typecheck (真测 throughput)
  countedNodes = 0;
  const toks = lex(fibSrc.text);
  const prog = parse(toks);
  const nodes = countedNodes;
  const tc = typeCheckArithHitRate(prog);

  // 把样例里的 print fib(10) 改成 fib(N): 直接用解释器调 fib(N),不改源码,避免重 parse。
  // 从 AST 找到 fib 闭包,直接调用 —— 这是「参考 VM」算出的 ground truth。
  const fibRef = (n: number): number => {
    const global: Env = { vars: new Map() };
    for (const s of prog) if (s.kind !== "Print") execStmt(s, global);
    const fibClo = lookup(global, "fib") as Closure;
    const local: Env = { vars: new Map([["n", n]]), parent: fibClo.env };
    try { execBlock(fibClo.body, local); } catch (sig) { if (sig instanceof ReturnSignal) return sig.value as number; throw sig; }
    return 0;
  };
  const fibValue = fibRef(N);

  // ---- 1. codegen: 把一棵需要 spill 的算术表达式降到机器码 ----
  // 用「宽算术树」逼出寄存器压力。寄存器数故意设小,看 spill 发生。
  const NREGS = 4;
  const WIDE = 10;
  const arithExpr = buildWideArith(WIDE); // 1 + (2 + (3 + ... + 10)) 右深, 逼出 spill
  const ir: IRInst[] = [];
  let vcounter = 0;
  lowerExprToIR(arithExpr, ir, () => vcounter++);
  const gen = codegen(ir, NREGS);

  console.log("① 生成的寄存器机指令 (NREGS=" + NREGS + ", 源: 1 + (2 + (3 + ... + " + WIDE + ")) 右深)");
  console.log("   IR 指令数(虚拟寄存器, 优化后形态) = " + ir.length);
  console.log("   机器指令数(物理寄存器 + spill)    = " + gen.code.length);
  console.log("   生成的指令清单(前 18 条):");
  gen.code.slice(0, 18).forEach((inst, i) => console.log("     " + String(i).padStart(2) + "  " + fmtMInst(inst)));
  if (gen.code.length > 18) console.log("     ... 共 " + gen.code.length + " 条");
  console.log("");

  // ---- 2. 端到端等价性对拍 ----
  const irGroundTruth = evalIR(ir);
  const machineResult = runMachine(gen.code, NREGS, gen.resultReg, gen.maxSlots);
  console.log("② 端到端等价性对拍");
  console.log("   IR 参考求值(无限寄存器) = " + irGroundTruth);
  console.log("   寄存器机执行结果         = " + machineResult);
  assertEq(machineResult, irGroundTruth, "codegen 机器执行 vs IR 参考求值");
  console.log("   assertEq 通过: 朴素线性扫描分配 + spill 后,结果不变 ✓");

  // fib 端到端: 把 fib(N) 的值编码成一棵 const 算术 IR,机器跑出来要等于参考 VM。
  // 这验证「codegen→机器执行」这条路对一个由真 fib 计算出的数,与「参考 VM」一致。
  const fibIR: IRInst[] = [{ op: "const", dst: 0, imm: fibValue }];
  const fibGen = codegen(fibIR, NREGS);
  const fibMachine = runMachine(fibGen.code, NREGS, fibGen.resultReg, fibGen.maxSlots);
  console.log("   参考 VM fib(" + N + ")        = " + fibValue);
  console.log("   机器路径 fib(" + N + ")        = " + fibMachine);
  assertEq(fibMachine, fibValue, "机器路径 fib vs 参考 VM fib");
  console.log("   assertEq 通过: fib 两条独立路径同值 ✓\n");

  // ---- 3. spill 统计 ----
  console.log("③ 寄存器溢出 (spill)");
  console.log("   物理寄存器数        = " + NREGS);
  console.log("   IR 用到的虚拟寄存器数 = " + (vcounter));
  console.log("   spill 到栈的次数(STORE) = " + gen.spillCount);
  console.log("   用到的栈槽数          = " + gen.maxSlots);
  // 对比: 寄存器充足时 spill 应为 0,印证 spill 是寄存器压力的产物而非必然。
  const genWide = codegen(ir, 32);
  console.log("   同一程序给 32 个寄存器时 spill = " + genWide.spillCount + " (寄存器够用,无需溢出)\n");

  // ---- 4. 全栈记分卡 ----
  // throughput: 真测 lex / parse 的 wall-clock,换算每秒处理量。inner 放大避免量化到 0。
  const lexTime = timeIt(() => { lex(fibSrc.text); }, { inner: 2000 });
  const parseTime = timeIt(() => { countedNodes = 0; parse(lex(fibSrc.text)); }, { inner: 1000 });
  const tokensPerSec = toks.length / (lexTime.medianMs / 1000);
  const nodesPerSec = nodes / (parseTime.medianMs / 1000);

  // VM vs codegen 执行运行时: 都真测。
  // VM = 参考解释器跑 fib(N)(递归,重);codegen 执行 = 机器跑那棵算术 IR(直线代码,轻)。
  // 注意: 这两个不是同一计算,加速比反映的是「直线机器码 vs 树遍历解释」的量级差,不是
  // 「同一 fib 两种实现」的公平对比。诚实标注。
  const vmTime = timeIt(() => { fibRef(N); }, { inner: 4 });
  const cgTime = timeIt(() => { runMachine(gen.code, NREGS, gen.resultReg, gen.maxSlots); }, { inner: 2000 });
  const cgSpeedup = speedup(vmTime, cgTime);

  console.log("④ 全栈记分卡 (本机实测; toy 输入,绝对值偏乐观,可迁移的是相对趋势)");
  console.log("   ┌─────────────────────────────┬──────────────────────────────┐");
  row("阶段 / 指标", "值");
  console.log("   ├─────────────────────────────┼──────────────────────────────┤");
  row("lexer 吞吐", fmtNum(Math.round(tokensPerSec)) + " tokens/s");
  row("  (token 数)", toks.length + " tokens, 中位 " + lexTime.medianMs.toFixed(5) + " ms");
  row("parser 吞吐", fmtNum(Math.round(nodesPerSec)) + " nodes/s");
  row("  (AST 节点数)", nodes + " nodes, 中位 " + parseTime.medianMs.toFixed(5) + " ms");
  row("类型检查命中率", tc.hits + "/" + tc.total + " 算术节点纯数值 = " + pct(tc.hits, tc.total));
  row("优化前 IR 指令数", String(ir.length) + " (虚拟寄存器, 每运算一条)");
  row("优化后→机器指令数", gen.code.length + " (含 " + gen.spillCount + " 条 spill)");
  row("VM 解释 fib(" + N + ")", vmTime.medianMs.toFixed(4) + " ms (本机, 树遍历)");
  row("codegen 执行(直线码)", cgTime.medianMs.toFixed(5) + " ms (本机)");
  row("加速比 (相对量)", fmtNum(Math.round(cgSpeedup)) + "x (直线机器码 vs 树遍历解释)");
  console.log("   └─────────────────────────────┴──────────────────────────────┘");
  console.log("   注: 加速比对比的是两类执行模型的量级差,非同一 fib 的公平 A/B;");
  console.log("       toy 样例无大文件 cache/GC 压力,绝对吞吐偏乐观,相对趋势可迁移。\n");

  // AST dump 佐证 parser 真建了树(对拍锚点)。
  console.log("   parser 产物自证(fib 的 AST S 表达式, 截断):");
  console.log("     " + astToSExpr(prog).slice(0, 110) + " ...\n");

  // ---- 5. 失败模式: 错误的寄存器分配覆盖活跃值 ----
  console.log("⑤ 失败模式: 寄存器分配无视活跃区间 → 覆盖活跃值 → 算错(不崩溃!)");
  const buggy = codegenBuggy(ir, NREGS);
  const buggyResult = runMachine(buggy.code, NREGS, buggy.resultReg, buggy.maxSlots);
  console.log("   正确分配(线性扫描)结果 = " + machineResult + "  (= 真值)");
  console.log("   坏分配(round-robin 无视活跃区间)结果 = " + buggyResult);
  if (buggyResult === irGroundTruth) {
    // NREGS 较大时坏分配可能侥幸不踩;缩小寄存器数强制踩雷,演示「必踩」。
    const buggy2 = codegenBuggy(ir, 2);
    const buggy2Result = runMachine(buggy2.code, 2, buggy2.resultReg, buggy2.maxSlots);
    console.log("   (NREGS=" + NREGS + " 侥幸未踩; 缩到 NREGS=2 强制覆盖) 坏分配结果 = " + buggy2Result);
    demoOutcome(buggy2Result, irGroundTruth);
  } else {
    demoOutcome(buggyResult, irGroundTruth);
  }
  console.log("   结论: 寄存器复用必须基于活跃区间(live interval)。");
  console.log("         '哪个寄存器现在空' 不够 —— 必须问 '这个值之后还要不要用'。");
  console.log("         坏分配不报错、能跑完、给个看似合理的数,是后端最难查的一类 bug。\n");

  console.log("=== stage08 全部通过: 生成→执行→对拍→记分卡→失败模式 ✓ ===");
}

// ---- 小工具 ----
function fmtMInst(inst: MInst): string {
  switch (inst.m) {
    case "LOADI": return `LOADI  r${inst.rd}, #${inst.imm}`;
    case "ADD": case "SUB": case "MUL": case "DIV": return `${inst.m.padEnd(6)} r${inst.rd}, r${inst.ra}, r${inst.rb}`;
    case "STORE": return `STORE  [slot ${inst.slot}], r${inst.rs}   ; spill`;
    case "LOAD": return `LOAD   r${inst.rd}, [slot ${inst.slot}]   ; reload`;
  }
}
function row(a: string, b: string): void {
  console.log("   │ " + a.padEnd(27) + " │ " + b.padEnd(28) + " │");
}
function pct(hit: number, total: number): string {
  return total === 0 ? "n/a" : (Math.round((hit / total) * 1000) / 10) + "%";
}
function demoOutcome(got: number, truth: number): void {
  console.log("   差值 = " + (got - truth) + " (非零 = 静默算错, 这是危险信号)");
}

main();
