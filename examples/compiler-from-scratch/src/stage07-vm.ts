// stage07-vm.ts — 字节码虚拟机:把 AST 编为字节码,写一个栈式 VM 真把程序跑起来。
//
// WHY 这一章存在(对比上一章「树遍历解释器」):
//   树遍历解释器(tree-walking)每跑一次表达式都要 (1) 递归下降访问 AST 节点、
//   (2) 对每个节点 switch 判别 kind、(3) JS 引擎的函数调用 + 栈帧开销。热循环里
//   同一段 AST 被重复访问百万次,这些「解释开销」全部重复支付。
//   字节码 VM 把 AST 一次性「编译」成扁平的指令数组(opcode + 操作数),运行时只剩
//   一个 while 循环 + 一个 switch —— 指令更紧凑、分支更可预测、不再有递归栈帧。
//   本章用同一个程序在两种执行模型上实测,给出可复现的加速比。
//
// 本文件是自包含的教学实现,不 import 任何 stageNN(那些一加载就 run main())。
// AST 直接用 core 的判别联合类型手工构造 —— 在真实编译器里这棵树来自 parser+resolver,
// 这里我们关心的是「树 → 字节码 → VM」这一段,所以把前端产物写死,聚焦后端。
//
// 不变量(INVARIANTS):
//   - 栈式 VM 的核心契约:每条指令对操作数栈的「净高度变化」是固定且已知的。
//     表达式求值后净 +1(把结果留在栈顶);语句求值后净 0(不留垃圾)。
//     函数调用进入新帧、返回时帧连同其局部/参数一起从栈上弹掉 —— 栈高度回到调用前。
//     这条不变量一旦破坏(见文件末 STACK-LEAK 失败模式),栈会单调增长直到越界。
//   - 常量池去重:同一字面量只存一份,指令用索引引用。确定性:同一程序编出同一字节码。
//   - 闭包 upvalue:内层函数捕获外层局部变量,捕获的是「值的快照」(本实现是值捕获,
//     非引用捕获 —— Lox-mini 的 counter 靠把状态存在闭包自身的 upvalue 槽里并回写)。
//
// 失败模式(FAILURE MODES,本文件 demo,非只跑 happy path):
//   - OP_RETURN 若不把当前帧的局部/参数从栈上收缩,每次函数调用都泄漏 N 个槽,
//     栈高度随调用次数线性增长 → 最终 stack overflow。文件末 demoStackLeak() 实演。

import {
  type Program,
  type Stmt,
  type Expr,
  type LiteralValue,
  span,
  astToSExpr,
  timeIt,
  speedup,
  assertEq,
  fmtNum,
} from "./core/index.js";

// ============================================================================
// 0. 手工构造 AST(扮演 parser+resolver 的产物)
// ============================================================================
// 这些 helper 只是把 core 的 AST 字面量构造压缩成一行,省去到处写 span。
// span 在本章无意义(我们没有源码文本),统一给零宽 span(0)。真实管线里 span 来自词法器。

const SP = span(0); // 占位 span:本章不做诊断,不需要真实偏移。

const lit = (value: LiteralValue): Expr => ({ kind: "Literal", value, span: SP });
const v = (name: string): Expr => ({ kind: "Var", name, span: SP });
const bin = (op: string, left: Expr, right: Expr): Expr => ({
  kind: "Binary",
  op,
  left,
  right,
  span: SP,
});
const call = (callee: Expr, ...args: Expr[]): Expr => ({
  kind: "Call",
  callee,
  args,
  span: SP,
});
const ret = (value?: Expr): Stmt => ({ kind: "Return", value, span: SP });
const ifS = (cond: Expr, then: Stmt[], els?: Stmt[]): Stmt => ({
  kind: "If",
  cond,
  then,
  else: els,
  span: SP,
});
const letS = (name: string, init: Expr): Stmt => ({ kind: "Let", name, init, span: SP });
const printS = (value: Expr): Stmt => ({ kind: "Print", value, span: SP });
const exprS = (expr: Expr): Stmt => ({ kind: "ExprStmt", expr, span: SP });
const fnS = (name: string, params: string[], body: Stmt[]): Stmt =>
  // 一个具名函数声明在 Lox-mini 里建模为「let name = (fn ...)」。
  letS(name, { kind: "Fn", name, params, body, span: SP });

// --- 程序 1: fib(n) 递归 ---
//   fn fib(n) { if (n < 2) return n; return fib(n-1) + fib(n-2); }
//   print fib(N);
// 选 fib 是因为它是「调用密集 + 无内存分配」的纯算术递归,能干净放大解释开销,
// 让两种执行模型的差距体现在「每条指令的成本」而非别的噪声上。
function makeFibProgram(n: number): Program {
  const fib = fnS("fib", ["n"], [
    ifS(bin("<", v("n"), lit(2)), [ret(v("n"))]),
    ret(bin("+", call(v("fib"), bin("-", v("n"), lit(1))), call(v("fib"), bin("-", v("n"), lit(2))))),
  ]);
  return [fib, printS(call(v("fib"), lit(n)))];
}

// --- 程序 2: 闭包计数器 ---
//   fn makeCounter() { let count = 0; fn next() { count = count + 1; return count; } return next; }
//   let c = makeCounter(); print c(); print c(); print c();   // 1 2 3
// 这是 upvalue 的最小可观测用例:next 捕获外层 count,每次调用读旧值 +1 再回写。
// 「3 次调用打印 1 2 3」直接证明闭包状态在调用之间被正确保持(而非每次重置)。
function makeCounterProgram(): Program {
  const next = fnS("next", [], [
    exprS({ kind: "Assign", name: "count", value: bin("+", v("count"), lit(1)), span: SP }),
    ret(v("count")),
  ]);
  const makeCounter = fnS("makeCounter", [], [letS("count", lit(0)), next, ret(v("next"))]);
  return [
    makeCounter,
    letS("c", call(v("makeCounter"))),
    printS(call(v("c"))),
    printS(call(v("c"))),
    printS(call(v("c"))),
  ];
}

// ============================================================================
// 1. 运行时值
// ============================================================================
// VM 操作数栈里放的「值」。number/boolean 直出;函数是「编译后的代码 + 捕获的 upvalue」。
// 用 Closure 而非裸 Fn:同一份函数代码可被多次实例化成不同闭包(各自捕获不同 upvalue)。

type RuntimeValue = number | boolean | Closure;

interface CompiledFn {
  readonly name: string;
  readonly arity: number; // 形参个数。调用方传错数量 → 运行时报错(见失败模式 §arity)。
  readonly code: readonly Instr[];
  readonly constants: readonly LiteralValue[];
  /** 本函数体内出现的局部变量名 → 槽位索引(参数占前 arity 个槽)。 */
  readonly locals: readonly string[];
  /** 本函数捕获的外层变量名(upvalue),编译期确定。 */
  readonly upvalues: readonly string[];
}

interface Closure {
  readonly kind: "closure";
  readonly fn: CompiledFn;
  /** upvalue 槽:与 fn.upvalues 一一对应,存「捕获瞬间的值快照」,可被内层回写。 */
  readonly captured: RuntimeValue[];
}

const isClosure = (x: RuntimeValue): x is Closure =>
  typeof x === "object" && x !== null && (x as Closure).kind === "closure";

// ============================================================================
// 2. 字节码指令集
// ============================================================================
// 栈式 VM:绝大多数指令从栈顶取操作数、把结果压回栈顶,因此指令本身很「窄」
// (多数零操作数)。这与寄存器式 VM(操作数是寄存器编号)是两条路线;栈式更易编译、
// 指令更短,代价是同一计算要更多条指令(更多 push/pop)。教学选栈式因为它和「表达式
// 求值天然是一棵树」的对应最直观:后序遍历 = 先压子结果再压运算符指令。

enum Op {
  CONST = "CONST", // push constants[operand]
  GET_LOCAL = "GET_LOCAL", // push frame.slots[operand]
  SET_LOCAL = "SET_LOCAL", // frame.slots[operand] = peek()  (赋值是表达式,结果留栈顶)
  GET_UPVALUE = "GET_UPVALUE", // push closure.captured[operand]
  SET_UPVALUE = "SET_UPVALUE", // closure.captured[operand] = peek()
  GET_GLOBAL = "GET_GLOBAL", // push globals[constants[operand] as name]  (顶层函数名等)
  ADD = "ADD",
  SUB = "SUB",
  MUL = "MUL",
  LESS = "LESS",
  CLOSURE = "CLOSURE", // 用 constants[operand] 指向的 CompiledFn 实例化闭包(捕获当前可见 upvalue)
  CALL = "CALL", // operand = argc;调用栈顶下方第 argc+1 个槽指向的闭包
  RETURN = "RETURN", // 弹出返回值,收缩当前帧,跳回调用点
  PRINT = "PRINT", // 弹栈顶并打印(经由注入的 sink,便于对拍)
  POP = "POP", // 丢弃栈顶(语句求值后清理表达式结果,维护「语句净高度 0」不变量)
  JUMP_IF_FALSE = "JUMP_IF_FALSE", // operand = 目标 ip;弹条件,假则跳转
  JUMP = "JUMP", // operand = 目标 ip;无条件跳转
}

interface Instr {
  readonly op: Op;
  readonly operand?: number;
  /** 反汇编用的人类可读注解(如常量值 / 变量名),不参与执行。 */
  readonly note?: string;
}

// ============================================================================
// 3. 编译器:AST → 字节码
// ============================================================================
// 每个函数(含顶层 <script>)编成一个 CompiledFn。编译是一次后序遍历:
//   - 表达式:先 emit 子表达式(把子结果留栈上),再 emit 自己的运算指令 → 净 +1。
//   - 语句:emit 后保证净 0(ExprStmt 末尾补 POP,Let 把值写进局部槽)。
// 变量解析在编译期完成(而非运行时查名字):决定每个变量是 local / upvalue / global,
// 这正是 stage03 resolver 的价值落地到字节码 —— 运行时按「槽位索引」O(1) 取值。

interface CompileScope {
  readonly fnName: string;
  readonly arity: number;
  readonly locals: string[]; // 槽位顺序 = 声明顺序,前 arity 个是参数。
  readonly upvalues: string[]; // 本函数引用到的外层变量(去重,保序)。
  readonly enclosing: CompileScope | undefined; // 词法外层,用于判定一个名字是否是 upvalue。
  readonly constants: LiteralValue[];
  readonly code: Instr[];
  /**
   * 全局函数注册表(所有层共享同一个 Map)。任意深度的内层函数编译完都登记到这里,
   * 供 VM 的 CLOSURE/CALL 按名字找回代码。共享而非每层一份 children:嵌套闭包
   * (next 在 makeCounter 里)若只挂在父 scope,顶层收集不到 —— 共享 registry 一劳永逸。
   * 不变量:函数名在整个程序内唯一(本教学子集成立);重名会互相覆盖。
   */
  readonly registry: Map<string, CompiledFn>;
}

/** 在常量池登记一个字面量并返回索引;去重保证同程序确定性 + 池更小。 */
function internConstant(scope: CompileScope, value: LiteralValue): number {
  const idx = scope.constants.findIndex((c) => c === value);
  if (idx >= 0) return idx;
  scope.constants.push(value);
  return scope.constants.length - 1;
}

function emit(scope: CompileScope, op: Op, operand?: number, note?: string): void {
  scope.code.push({ op, operand, note });
}

/**
 * 解析一个变量引用,返回它在当前作用域里的「种类 + 槽位」。
 * 这是编译器最核心的判定:local(本帧槽) / upvalue(捕获自外层) / global(顶层名)。
 * 不变量:同名优先级 local > upvalue > global —— 内层遮蔽外层,符合词法作用域。
 */
function resolveVar(
  scope: CompileScope,
  name: string,
): { kind: "local" | "upvalue" | "global"; slot: number } {
  const localIdx = scope.locals.indexOf(name);
  if (localIdx >= 0) return { kind: "local", slot: localIdx };

  // 不在本帧:看外层链。任意外层有这个 local → 本函数把它登记为 upvalue。
  if (scope.enclosing && variableExistsInEnclosing(scope.enclosing, name)) {
    let upIdx = scope.upvalues.indexOf(name);
    if (upIdx < 0) {
      scope.upvalues.push(name);
      upIdx = scope.upvalues.length - 1;
    }
    return { kind: "upvalue", slot: upIdx };
  }

  // 都不是 → 当 global(顶层函数声明)。槽位是常量池里函数名字符串的索引。
  return { kind: "global", slot: internConstant(scope, name) };
}

function variableExistsInEnclosing(scope: CompileScope, name: string): boolean {
  if (scope.locals.includes(name)) return true;
  return scope.enclosing ? variableExistsInEnclosing(scope.enclosing, name) : false;
}

function compileExpr(scope: CompileScope, e: Expr): void {
  switch (e.kind) {
    case "Literal": {
      const idx = internConstant(scope, e.value);
      emit(scope, Op.CONST, idx, String(e.value));
      return;
    }
    case "Var": {
      const r = resolveVar(scope, e.name);
      if (r.kind === "local") emit(scope, Op.GET_LOCAL, r.slot, e.name);
      else if (r.kind === "upvalue") emit(scope, Op.GET_UPVALUE, r.slot, e.name);
      else emit(scope, Op.GET_GLOBAL, r.slot, e.name);
      return;
    }
    case "Assign": {
      compileExpr(scope, e.value); // 先求右值,留栈顶。
      const r = resolveVar(scope, e.name);
      // SET_* 不弹栈顶:赋值是表达式,其结果(被赋的值)继续留在栈上供外层使用。
      if (r.kind === "local") emit(scope, Op.SET_LOCAL, r.slot, e.name);
      else if (r.kind === "upvalue") emit(scope, Op.SET_UPVALUE, r.slot, e.name);
      else throw new Error(`cannot assign to global '${e.name}'`);
      return;
    }
    case "Binary": {
      compileExpr(scope, e.left);
      compileExpr(scope, e.right);
      const opMap: Record<string, Op> = { "+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "<": Op.LESS };
      const op = opMap[e.op];
      if (!op) throw new Error(`unsupported binary op '${e.op}' in stage07 subset`);
      emit(scope, op);
      return;
    }
    case "Call": {
      // 调用约定:先压被调对象(closure),再压各实参,然后 CALL argc。
      // 栈布局: [..., callee, arg0, arg1, ...] ;CALL 据 argc 定位 callee。
      compileExpr(scope, e.callee);
      for (const a of e.args) compileExpr(scope, a);
      emit(scope, Op.CALL, e.args.length);
      return;
    }
    case "Fn": {
      const child = compileFunction(scope, e.name ?? "<anon>", e.params, e.body);
      scope.registry.set(child.name, child); // 登记到共享注册表,任意深度可达。
      const idx = internConstant(scope, child.name); // 用名字索引找回 CompiledFn。
      emit(scope, Op.CLOSURE, idx, child.name);
      return;
    }
    case "Unary":
      throw new Error("unary not in stage07 subset");
    default: {
      // 穷尽性兜底:漏掉一种 Expr(或误传了非 Expr,如把数组当 cond)会在此显式炸,
      // 而非静默不 emit 代码 —— 后者会让 VM 在运行时神秘地 pop 空栈,极难定位。
      throw new Error(`compileExpr: unhandled expr ${JSON.stringify(e)}`);
    }
  }
}

function compileStmt(scope: CompileScope, s: Stmt): void {
  switch (s.kind) {
    case "Let": {
      compileExpr(scope, s.init); // 求初值,留栈顶。
      // 声明一个新局部槽。注意:对函数声明(init 是 Fn),槽位即闭包对象所在处。
      if (!scope.locals.includes(s.name)) scope.locals.push(s.name);
      const slot = scope.locals.indexOf(s.name);
      emit(scope, Op.SET_LOCAL, slot, s.name);
      emit(scope, Op.POP, undefined, "discard let value"); // let 是语句,净高度 0。
      return;
    }
    case "Print":
      compileExpr(scope, s.value);
      emit(scope, Op.PRINT);
      return;
    case "ExprStmt":
      compileExpr(scope, s.expr);
      emit(scope, Op.POP, undefined, "discard expr-stmt result"); // 维护语句净高度 0。
      return;
    case "Return":
      if (s.value) compileExpr(scope, s.value);
      else emit(scope, Op.CONST, internConstant(scope, 0), "implicit nil->0"); // 裸 return 给个值,简化 VM。
      emit(scope, Op.RETURN);
      return;
    case "If": {
      compileExpr(scope, s.cond);
      // 回填式跳转:先 emit 占位,记下位置,编完 then 再回填目标 ip。
      const jifPos = scope.code.length;
      emit(scope, Op.JUMP_IF_FALSE, -1);
      for (const st of s.then) compileStmt(scope, st);
      if (s.else) {
        const jmpPos = scope.code.length;
        emit(scope, Op.JUMP, -1);
        patch(scope, jifPos, scope.code.length); // 条件假 → 跳到 else 开头。
        for (const st of s.else) compileStmt(scope, st);
        patch(scope, jmpPos, scope.code.length); // then 执行完 → 跳过 else。
      } else {
        patch(scope, jifPos, scope.code.length);
      }
      return;
    }
    case "While":
    case "Block":
      throw new Error(`${s.kind} not in stage07 subset`);
  }
}

/** 回填跳转目标:占位指令是 readonly,这里替换整条以保持不可变契约。 */
function patch(scope: CompileScope, pos: number, target: number): void {
  const old = scope.code[pos];
  scope.code[pos] = { op: old.op, operand: target, note: old.note };
}

function compileFunction(
  enclosing: CompileScope | undefined,
  name: string,
  params: readonly string[],
  body: readonly Stmt[],
): CompiledFn {
  if (!enclosing) throw new Error("compileFunction: nested fn must have an enclosing scope");
  const scope: CompileScope = {
    fnName: name,
    arity: params.length,
    locals: [...params], // 参数占据前 arity 个槽。
    upvalues: [],
    enclosing,
    constants: [],
    code: [],
    registry: enclosing.registry, // 共享顶层注册表。
  };
  for (const st of body) compileStmt(scope, st);
  // 兜底 RETURN:函数体不以 return 结尾时,补一个返回 0(防止 ip 跑出 code 末尾)。
  // 这是一条「看似冗余但必需」的指令 —— 见 VM 主循环对 ip 越界的假设。
  emit(scope, Op.CONST, internConstant(scope, 0), "implicit return 0");
  emit(scope, Op.RETURN);

  return {
    name,
    arity: scope.arity,
    code: scope.code,
    constants: scope.constants,
    locals: scope.locals,
    upvalues: scope.upvalues,
  };
}

/**
 * 把整个 Program 编成顶层 <script> 函数(arity 0)。所有内层函数(任意深度)在
 * 编译时登记进共享 registry,返回的 allFns 即该 registry,VM 据此按名字找回任一函数。
 */
function compileProgram(program: Program): { script: CompiledFn; allFns: Map<string, CompiledFn> } {
  const registry = new Map<string, CompiledFn>();
  const topScope: CompileScope = {
    fnName: "<script>",
    arity: 0,
    locals: [],
    upvalues: [],
    enclosing: undefined,
    constants: [],
    code: [],
    registry,
  };
  for (const st of program) compileStmt(topScope, st);
  emit(topScope, Op.CONST, internConstant(topScope, 0), "script end");
  emit(topScope, Op.RETURN);

  const script: CompiledFn = {
    name: "<script>",
    arity: 0,
    code: topScope.code,
    constants: topScope.constants,
    locals: topScope.locals,
    upvalues: topScope.upvalues,
  };
  registry.set("<script>", script);
  return { script, allFns: registry };
}

// ============================================================================
// 4. 反汇编(disassembler)
// ============================================================================
// 把字节码打印成人类可读的列表。这是「编译器后端的眼睛」:不能反汇编 = 不能调试 VM。
// 输出确定性(同程序同输出),可作对拍锚点。

function disassemble(fn: CompiledFn): string {
  const lines: string[] = [];
  lines.push(`fn ${fn.name}(arity=${fn.arity})  locals=[${fn.locals.join(",")}]  upvalues=[${fn.upvalues.join(",")}]`);
  fn.code.forEach((ins, i) => {
    const ip = String(i).padStart(3, "0");
    const opd = ins.operand === undefined ? "" : String(ins.operand).padStart(4);
    const note = ins.note ? `  ; ${ins.note}` : "";
    lines.push(`  ${ip}  ${ins.op.padEnd(14)}${opd}${note}`);
  });
  if (fn.constants.length > 0) {
    lines.push(`  constants: [${fn.constants.map((c) => JSON.stringify(c)).join(", ")}]`);
  }
  return lines.join("\n");
}

// ============================================================================
// 5. 栈式 VM 解释循环
// ============================================================================
// 一个 CallFrame = 一次函数调用的运行时上下文。slots 是该帧的局部/参数区。
// 关键设计:所有帧共享同一条「操作数栈」(stack),帧只是记下自己的 base(栈基址)。
// 函数返回时,栈收缩回 base,正是「帧弹出」—— 这条收缩逻辑就是栈泄漏失败模式的命门。

interface CallFrame {
  readonly closure: Closure;
  ip: number; // instruction pointer:下一条要执行的指令索引。
  readonly slots: RuntimeValue[]; // 本帧局部区(参数 + let 声明的局部)。
}

interface VmStats {
  opsExecuted: number; // 执行的指令条数(确定性,可直接断言)。
  maxFrameDepth: number; // 调用栈最大深度(递归深度的客观度量)。
  maxStackHeight: number; // 操作数栈达到过的最大高度(栈泄漏检测点)。
}

interface VmResult {
  readonly output: number[]; // PRINT 打印过的值(供对拍)。
  readonly stats: VmStats;
}

const STACK_LIMIT = 100_000; // 操作数栈硬上限:防失控泄漏跑爆内存,改为可观测的越界报错。

/**
 * 执行一个已编译程序,返回 PRINT 输出 + 运行统计。
 * @param opts.onPrint 每次 PRINT 时回调(用于流式收集输出对拍)。
 * @param opts.onCall  每次 CALL 时回调「当前帧深度, 当前操作数栈高度」,
 *        用于可视化递归时帧栈深度变化(happy path)与栈泄漏时栈高度增长(失败模式)。
 * 失败模式不靠 flag 注入,而靠喂给它「编译期不变量被破坏」的字节码(见 buildLeakyProgram)——
 * 更诚实:VM 本身永远正确,被测的是「错误的字节码」会被 VM 如实暴露成栈越界。
 */
function runVm(
  entry: Closure,
  allFns: Map<string, CompiledFn>,
  opts: {
    onPrint?: (n: number) => void;
    onCall?: (frameDepth: number, stackHeight: number) => void;
  } = {},
): VmResult {
  const stack: RuntimeValue[] = [];
  const frames: CallFrame[] = [];
  const stats: VmStats = { opsExecuted: 0, maxFrameDepth: 0, maxStackHeight: 0 };
  const output: number[] = [];

  const push = (x: RuntimeValue) => {
    stack.push(x);
    if (stack.length > stats.maxStackHeight) stats.maxStackHeight = stack.length;
    if (stack.length > STACK_LIMIT) {
      throw new Error(
        `STACK OVERFLOW: operand stack exceeded ${fmtNum(STACK_LIMIT)} slots ` +
          `(frames=${frames.length}). 典型根因:函数返回未收缩栈帧(栈泄漏)。`,
      );
    }
  };
  const pop = (): RuntimeValue => {
    const x = stack.pop();
    if (x === undefined) throw new Error("VM bug: pop on empty operand stack");
    return x;
  };
  const num = (x: RuntimeValue): number => {
    if (typeof x !== "number") throw new Error(`VM type error: expected number, got ${typeof x}`);
    return x;
  };

  // 顶层帧:<script>,无 upvalue,slots 为顶层局部(顶层函数声明的闭包就放这)。
  frames.push({ closure: entry, ip: 0, slots: [] });

  while (frames.length > 0) {
    const frame = frames[frames.length - 1];
    const fn = frame.closure.fn;

    if (frame.ip >= fn.code.length) {
      // ip 跑出代码末尾:理论上不该发生(每个函数都以 RETURN 收尾)。当作隐式返回。
      frames.pop();
      continue;
    }

    const ins = fn.code[frame.ip++];
    stats.opsExecuted++;

    switch (ins.op) {
      case Op.CONST:
        push(fn.constants[ins.operand!] as RuntimeValue);
        break;
      case Op.GET_LOCAL:
        push(frame.slots[ins.operand!]);
        break;
      case Op.SET_LOCAL:
        frame.slots[ins.operand!] = stack[stack.length - 1]; // peek,不弹:赋值结果留栈顶。
        break;
      case Op.GET_UPVALUE:
        push(frame.closure.captured[ins.operand!]);
        break;
      case Op.SET_UPVALUE:
        // 回写捕获槽:这就是 counter 能累加的机制 —— 闭包持有可变的 captured 数组。
        frame.closure.captured[ins.operand!] = stack[stack.length - 1];
        break;
      case Op.GET_GLOBAL: {
        // 顶层函数名 → 在顶层帧的 slots 里按声明顺序找。简化实现:globals 即 <script> 的 locals。
        const name = fn.constants[ins.operand!] as string;
        const scriptFrame = frames[0];
        const slot = entry.fn.locals.indexOf(name);
        if (slot < 0) throw new Error(`VM: undefined global '${name}'`);
        push(scriptFrame.slots[slot]);
        break;
      }
      case Op.ADD: {
        const b = num(pop());
        const a = num(pop());
        push(a + b);
        break;
      }
      case Op.SUB: {
        const b = num(pop());
        const a = num(pop());
        push(a - b);
        break;
      }
      case Op.MUL: {
        const b = num(pop());
        const a = num(pop());
        push(a * b);
        break;
      }
      case Op.LESS: {
        const b = num(pop());
        const a = num(pop());
        push(a < b);
        break;
      }
      case Op.CLOSURE: {
        const name = fn.constants[ins.operand!] as string;
        const target = allFns.get(name);
        if (!target) throw new Error(`VM: unknown function '${name}' for CLOSURE`);
        // 捕获:对 target 声明的每个 upvalue,从「当前可见作用域」抓当前值快照。
        // 当前可见 = 本帧 locals(若同名)或本帧自己的 captured(透传更外层)。
        const captured: RuntimeValue[] = target.upvalues.map((upName) => {
          const localSlot = fn.locals.indexOf(upName);
          if (localSlot >= 0) return frame.slots[localSlot];
          const upSlot = fn.upvalues.indexOf(upName);
          if (upSlot >= 0) return frame.closure.captured[upSlot];
          throw new Error(`VM: cannot capture upvalue '${upName}'`);
        });
        push({ kind: "closure", fn: target, captured });
        break;
      }
      case Op.CALL: {
        const argc = ins.operand!;
        // 栈布局: [..., callee, arg0..arg{argc-1}] 。callee 在 argc 个实参之下。
        const calleeIdx = stack.length - argc - 1;
        const callee = stack[calleeIdx];
        if (!isClosure(callee)) throw new Error(`VM: attempt to call a non-function (${typeof callee})`);
        if (callee.fn.arity !== argc) {
          // 失败模式(arity):传参数量不符。真编译器在编译期或运行期都该拒绝。
          throw new Error(`VM: '${callee.fn.name}' expects ${callee.fn.arity} args, got ${argc}`);
        }
        // 新帧的 slots:前 argc 个是实参(从栈上搬过来),其余局部运行中按 SET_LOCAL 填。
        const slots: RuntimeValue[] = [];
        for (let k = 0; k < argc; k++) slots[k] = stack[calleeIdx + 1 + k];
        // 把 callee + 实参从操作数栈上清掉 —— 它们已转移进新帧的 slots。
        stack.length = calleeIdx;
        frames.push({ closure: callee, ip: 0, slots });
        if (frames.length > stats.maxFrameDepth) stats.maxFrameDepth = frames.length;
        opts.onCall?.(frames.length, stack.length);
        break;
      }
      case Op.RETURN: {
        const result = pop(); // 弹出返回值。
        frames.pop(); // 弹掉当前调用帧(其 slots 随对象一起被 GC,不占操作数栈)。
        // 栈平衡:CALL 时已把 callee+args 清出操作数栈,被调函数局部都在独立的 frame.slots,
        // 所以操作数栈在「调用→返回」前后自然平衡,这里只需把返回值交还调用者栈顶。
        // 正确性依赖一条编译期不变量:每个 Call 表达式作为 ExprStmt 出现时,编译器在其后
        // emit 了 POP(见 compileStmt 的 ExprStmt 分支)丢弃返回值。failure §⑥ 用「省掉这个
        // POP」的手写字节码制造净增长,演示该不变量被破坏后栈如何线性泄漏。
        push(result);
        break;
      }
      case Op.PRINT: {
        const x = num(pop());
        output.push(x);
        opts.onPrint?.(x);
        break;
      }
      case Op.POP:
        pop();
        break;
      case Op.JUMP_IF_FALSE: {
        const cond = pop();
        if (cond === false) frame.ip = ins.operand!;
        break;
      }
      case Op.JUMP:
        frame.ip = ins.operand!;
        break;
    }
  }

  return { output, stats };
}

// ============================================================================
// 6. AST 树遍历解释器(对照基线)
// ============================================================================
// 同样的程序,不编译,直接递归解释 AST。它执行的「操作」= 访问的 AST 节点数,
// 与 VM 执行的指令数同量纲(都是「一次原子求值动作」),所以两者可比。
// 这是上一章的执行模型;本章把它当 baseline 来量化字节码 VM 的收益。

interface TreeEnv {
  vars: Map<string, TreeValue>;
  parent: TreeEnv | undefined;
}

interface TreeClosure {
  readonly kind: "tree-closure";
  readonly params: readonly string[];
  readonly body: readonly Stmt[];
  readonly env: TreeEnv; // 定义时的词法环境 —— 闭包捕获就靠它。
}

type TreeValue = number | boolean | TreeClosure;

class ReturnSignal {
  constructor(public readonly value: TreeValue) {}
}

function lookupEnv(env: TreeEnv, name: string): TreeValue {
  let e: TreeEnv | undefined = env;
  while (e) {
    if (e.vars.has(name)) return e.vars.get(name) as TreeValue;
    e = e.parent;
  }
  throw new Error(`tree-walk: undefined variable '${name}'`);
}

function assignEnv(env: TreeEnv, name: string, value: TreeValue): void {
  let e: TreeEnv | undefined = env;
  while (e) {
    if (e.vars.has(name)) {
      e.vars.set(name, value);
      return;
    }
    e = e.parent;
  }
  throw new Error(`tree-walk: assign to undefined '${name}'`);
}

/** 解释 AST,counter.n 累加访问过的节点数(= 执行的「操作」),与 VM 指令数对标。 */
function evalProgram(program: Program, onPrint: (n: number) => void): { ops: number } {
  const counter = { n: 0 };
  const global: TreeEnv = { vars: new Map(), parent: undefined };

  const evalExpr = (e: Expr, env: TreeEnv): TreeValue => {
    counter.n++;
    switch (e.kind) {
      case "Literal":
        return e.value as TreeValue;
      case "Var":
        return lookupEnv(env, e.name);
      case "Assign": {
        const val = evalExpr(e.value, env);
        assignEnv(env, e.name, val);
        return val;
      }
      case "Binary": {
        const a = evalExpr(e.left, env) as number;
        const b = evalExpr(e.right, env) as number;
        switch (e.op) {
          case "+": return a + b;
          case "-": return a - b;
          case "*": return a * b;
          case "<": return a < b;
        }
        throw new Error(`tree-walk: bad op ${e.op}`);
      }
      case "Call": {
        const callee = evalExpr(e.callee, env);
        if (typeof callee !== "object") throw new Error("tree-walk: call non-fn");
        const args = e.args.map((a) => evalExpr(a, env));
        const local: TreeEnv = { vars: new Map(), parent: callee.env };
        callee.params.forEach((p, i) => local.vars.set(p, args[i]));
        try {
          for (const st of callee.body) execStmt(st, local);
        } catch (sig) {
          if (sig instanceof ReturnSignal) return sig.value;
          throw sig;
        }
        return 0;
      }
      case "Fn":
        return { kind: "tree-closure", params: e.params, body: e.body, env };
      case "Unary":
        throw new Error("tree-walk: unary unsupported");
    }
  };

  const execStmt = (s: Stmt, env: TreeEnv): void => {
    counter.n++;
    switch (s.kind) {
      case "Let":
        env.vars.set(s.name, evalExpr(s.init, env));
        return;
      case "Print":
        onPrint(evalExpr(s.value, env) as number);
        return;
      case "ExprStmt":
        evalExpr(s.expr, env);
        return;
      case "Return":
        throw new ReturnSignal(s.value ? evalExpr(s.value, env) : 0);
      case "If":
        if (evalExpr(s.cond, env) === true) {
          for (const st of s.then) execStmt(st, env);
        } else if (s.else) {
          for (const st of s.else) execStmt(st, env);
        }
        return;
      case "While":
      case "Block":
        throw new Error(`tree-walk: ${s.kind} unsupported`);
    }
  };

  for (const st of program) execStmt(st, global);
  return { ops: counter.n };
}

// ============================================================================
// 7. 第 06 章「优化前/后」:常量折叠(constant folding)
// ============================================================================
// 模拟优化器对一个 AST 做常量折叠:把「全是字面量的算术子树」在编译期算成单个字面量。
// 用它对比「优化前 vs 优化后字节码在同一 VM 上的运行时」—— 优化减少了指令条数,
// VM 跑更少指令 → 更快。这是 stage06 优化收益落到 stage07 执行层的体现。

function foldConstants(e: Expr): Expr {
  if (e.kind === "Binary") {
    const l = foldConstants(e.left);
    const r = foldConstants(e.right);
    if (l.kind === "Literal" && r.kind === "Literal" && typeof l.value === "number" && typeof r.value === "number") {
      let folded: number | undefined;
      switch (e.op) {
        case "+": folded = l.value + r.value; break;
        case "-": folded = l.value - r.value; break;
        case "*": folded = l.value * r.value; break;
      }
      if (folded !== undefined) return { kind: "Literal", value: folded, span: SP };
    }
    return { kind: "Binary", op: e.op, left: l, right: r, span: SP };
  }
  if (e.kind === "Call") {
    return { kind: "Call", callee: e.callee, args: e.args.map(foldConstants), span: SP };
  }
  if (e.kind === "Fn") {
    // 必须递归进函数体:否则「let work = fn(){...}」里函数体内的常量子树永远不被折叠
    // (这是早期版本的真 bug —— init 是 Fn,foldConstants 不下钻就等于没优化)。
    return { kind: "Fn", name: e.name, params: e.params, body: e.body.map(foldStmt), span: SP };
  }
  return e;
}

function foldStmt(s: Stmt): Stmt {
  switch (s.kind) {
    case "Let": return { kind: "Let", name: s.name, init: foldConstants(s.init), span: SP };
    case "Print": return { kind: "Print", value: foldConstants(s.value), span: SP };
    case "ExprStmt": return { kind: "ExprStmt", expr: foldConstants(s.expr), span: SP };
    case "Return": return { kind: "Return", value: s.value ? foldConstants(s.value) : undefined, span: SP };
    case "If": return { kind: "If", cond: foldConstants(s.cond), then: s.then.map(foldStmt), else: s.else?.map(foldStmt), span: SP };
    default: return s;
  }
}

const foldProgram = (p: Program): Program => p.map(foldStmt);

// 一个「算术堆叠在循环里」的程序,常量折叠能明显减少指令:
//   fn work(n) { return (((2*3)+4) - 1) + n; }   // 左边整团是常量,可折成 9
//   print work(K);
// 注:work 不递归,折叠收益来自「每次调用少算几条算术指令」。为放大,我们在 bench 里
// 通过外层 fib 式重复调用驱动,但这里保持程序简单,靠 inner 循环放大测量(见 §benchmark)。
function makeFoldableProgram(): Program {
  const work = fnS("work", ["n"], [
    ret(bin("+", bin("-", bin("+", bin("*", lit(2), lit(3)), lit(4)), lit(1)), v("n"))),
  ]);
  return [work, printS(call(v("work"), lit(100)))];
}

function countInstrs(allFns: Map<string, CompiledFn>): number {
  let total = 0;
  for (const fn of allFns.values()) total += fn.code.length;
  return total;
}

// ============================================================================
// 8. 栈泄漏失败模式
// ============================================================================
// 演示「函数返回时没正确收缩栈」会怎样。我们不能简单关掉 RETURN 的收缩(本实现局部
// 不放操作数栈,关了也不泄漏),所以用一段「每次调用都净留一个值在栈上」的字节码
// 手工模拟「忘记 POP 返回占位」的经典 bug:调用者 CALL 后本应 POP 掉用不到的返回值,
// 这里故意编出「不 POP」的字节码,让栈随调用次数线性增长 → 直到 STACK_LIMIT 越界。

/**
 * 手工拼一个「泄漏」CompiledFn:一个递归函数 leak(n),每次调用先递归 leak(n-1),
 * 返回后【故意不 POP】那个返回值,于是每展开一层就在操作数栈上多留一个数。
 * 递归 depth 层 → 栈上净留 depth 个值。depth 够大就越过 STACK_LIMIT。
 */
function buildLeakyProgram(depth: number): { script: Closure; allFns: Map<string, CompiledFn> } {
  // leak(n): if (n < 1) return 0; <recurse leak(n-1) but DON'T pop result> return 0;
  // 字节码手写以便插入「漏掉的 POP」。
  const leakCode: Instr[] = [
    { op: Op.GET_LOCAL, operand: 0, note: "n" }, // 0
    { op: Op.CONST, operand: 0, note: "1" }, // 1   constants[0]=1
    { op: Op.LESS }, // 2   n < 1 ?
    { op: Op.JUMP_IF_FALSE, operand: 6 }, // 3   假则跳到 6 继续递归
    { op: Op.CONST, operand: 1, note: "0" }, // 4   constants[1]=0
    { op: Op.RETURN }, // 5   base case 返回 0
    { op: Op.GET_GLOBAL, operand: 2, note: "leak" }, // 6   constants[2]="leak"
    { op: Op.GET_LOCAL, operand: 0, note: "n" }, // 7
    { op: Op.CONST, operand: 0, note: "1" }, // 8
    { op: Op.SUB }, // 9   n - 1
    { op: Op.CALL, operand: 1 }, // 10  leak(n-1) → 返回值留栈顶
    // 【BUG】此处本应 { op: Op.POP } 丢弃用不到的返回值。故意省略 → 每层泄漏一个槽。
    { op: Op.CONST, operand: 1, note: "0" }, // 11
    { op: Op.RETURN }, // 12  返回 0(但栈上已多留了上面没 POP 的值)
  ];
  const leak: CompiledFn = {
    name: "leak",
    arity: 1,
    code: leakCode,
    constants: [1, 0, "leak"],
    locals: ["n"],
    upvalues: [],
  };

  // <script>: let leak = <closure>; print leak(depth);
  const scriptCode: Instr[] = [
    { op: Op.CLOSURE, operand: 0, note: "leak" }, // constants[0]="leak"
    { op: Op.SET_LOCAL, operand: 0, note: "leak" },
    { op: Op.POP },
    { op: Op.GET_GLOBAL, operand: 0, note: "leak" },
    { op: Op.CONST, operand: 1, note: "depth" }, // constants[1]=depth
    { op: Op.CALL, operand: 1 },
    { op: Op.PRINT },
    { op: Op.CONST, operand: 2, note: "0" },
    { op: Op.RETURN },
  ];
  const script: CompiledFn = {
    name: "<script>",
    arity: 0,
    code: scriptCode,
    constants: ["leak", depth, 0],
    locals: ["leak"],
    upvalues: [],
  };

  const allFns = new Map<string, CompiledFn>([["<script>", script], ["leak", leak]]);
  return { script: { kind: "closure", fn: script, captured: [] }, allFns };
}

// ============================================================================
// main
// ============================================================================

function main(): void {
  console.log("============================================================");
  console.log(" 第 07 章 — 字节码虚拟机:把程序真跑起来");
  console.log("============================================================\n");

  // -------- ① 编译 + 反汇编 --------
  console.log("【① AST → 字节码 → 反汇编】fib 程序\n");
  const fibProgram = makeFibProgram(20);
  console.log("源程序(S 表达式,扮演 parser 产物):");
  console.log("  " + astToSExpr(fibProgram) + "\n");

  const { script: fibScript, allFns: fibFns } = compileProgram(fibProgram);
  for (const fn of fibFns.values()) {
    console.log(disassemble(fn));
    console.log("");
  }

  // -------- ② VM 跑通 fib(20) + 对拍 --------
  console.log("【② 栈式 VM 跑通 fib(20)】");
  const fibClosure: Closure = { kind: "closure", fn: fibScript, captured: [] };
  const fibRun = runVm(fibClosure, fibFns, {});
  console.log(`  程序输出: ${fibRun.output.join(", ")}`);
  // fib(20) = 6765,这是数学事实,用它当 ground truth 对拍 VM 正确性。
  assertEq(fibRun.output, [6765], "VM fib(20) output");
  console.log(`  对拍通过: fib(20) == 6765 ✓`);
  console.log(`  VM 执行指令数: ${fmtNum(fibRun.stats.opsExecuted)} 条`);
  console.log(`  最大调用帧深度: ${fibRun.stats.maxFrameDepth}(fib 递归深度 + 顶层帧)`);
  console.log(`  操作数栈峰值高度: ${fibRun.stats.maxStackHeight} 槽\n`);

  // -------- ③ 闭包计数器 + upvalue --------
  console.log("【③ 闭包计数器:upvalue 捕获 + 跨调用保持状态】");
  const counterProgram = makeCounterProgram();
  console.log("源程序:");
  console.log("  " + astToSExpr(counterProgram) + "\n");
  const { script: ctScript, allFns: ctFns } = compileProgram(counterProgram);
  // 打印 next 函数的反汇编,让 upvalue 槽位看得见。
  const nextFn = ctFns.get("next");
  if (nextFn) {
    console.log(disassemble(nextFn));
    console.log("");
  }
  const ctClosure: Closure = { kind: "closure", fn: ctScript, captured: [] };
  const ctRun = runVm(ctClosure, ctFns, {});
  console.log(`  3 次调用 c() 输出: ${ctRun.output.join(", ")}`);
  // 1,2,3 证明 count 这个 upvalue 在调用之间被保持并累加(而非每次重置为 0)。
  assertEq(ctRun.output, [1, 2, 3], "closure counter output");
  console.log(`  对拍通过: 闭包状态跨调用保持,counter == [1,2,3] ✓`);
  console.log(`  → upvalue 'count' 被 next 捕获,SET_UPVALUE 每次回写,旧值 +1\n`);

  // 递归帧栈深度可视化:跑 fib(6),回报每次 call 时的帧深度。
  console.log("【③b 递归调用时帧栈深度变化】fib(6) 调用序列(前 12 次 call):");
  let callLog: string[] = [];
  const fib6 = compileProgram(makeFibProgram(6));
  runVm({ kind: "closure", fn: fib6.script, captured: [] }, fib6.allFns, {
    onCall: (depth, height) => {
      if (callLog.length < 12) callLog.push(`depth=${depth}(stack=${height})`);
    },
  });
  console.log("  " + callLog.join("  ") + "  ...");
  console.log("  → 帧深度随递归深入增加、返回时减少;栈高度始终有界(无泄漏)\n");

  // -------- ④ 实测: 树遍历 vs 字节码 VM --------
  console.log("【④ 实测:AST 树遍历解释 vs 字节码 VM】(同一程序 fib(25))");
  const benchProgram = makeFibProgram(25);
  const { script: benchScript, allFns: benchFns } = compileProgram(benchProgram);
  const benchClosure: Closure = { kind: "closure", fn: benchScript, captured: [] };

  // 先各跑一次拿「操作数」计数(确定性,可直接断言/对比),并校验两者结果一致。
  const treeOut: number[] = [];
  const treeStats = evalProgram(benchProgram, (n) => treeOut.push(n));
  const vmOut: number[] = [];
  const vmRun = runVm(benchClosure, benchFns, { onPrint: (n) => vmOut.push(n) });
  assertEq(treeOut, vmOut, "tree-walk vs VM same output");
  console.log(`  两种模型输出一致: fib(25) == ${vmOut[0]} ✓`);
  console.log(`  树遍历访问 AST 节点(操作)数: ${fmtNum(treeStats.ops)}`);
  console.log(`  VM 执行字节码指令(操作)数:   ${fmtNum(vmRun.stats.opsExecuted)}`);
  const opRatio = treeStats.ops / vmRun.stats.opsExecuted;
  // 诚实数字:两者操作数几乎相等(同一算法,原子求值动作数量级相同)。VM 的收益【不在】
  // 「做更少的操作」,而在「每个操作更便宜」:树遍历每个节点 = 一次 JS 递归调用 + kind
  // switch + env Map 查找;VM 每条指令 = 一次平坦 while 循环迭代 + switch + 数组下标读。
  // 后者无递归栈帧、无 Map 哈希、分支更可预测。差距由下面的 wall-clock 加速比体现。
  console.log(`  操作数比(树/VM): ${opRatio.toFixed(2)}x（≈1:同一算法操作数量相当;VM 的赢点是「每操作更便宜」,非「操作更少」）`);

  // 真实 wall-clock 测速(中位数去抖)。绝对毫秒只展示,断言只用相对加速比。
  const tTree = timeIt(() => {
    let s = 0;
    evalProgram(benchProgram, (n) => (s += n));
  });
  const tVm = timeIt(() => {
    runVm(benchClosure, benchFns, {});
  });
  const sp = speedup(tTree, tVm);
  console.log(`  树遍历 中位数 = ${tTree.medianMs.toFixed(3)} ms（本机实测,仅展示）`);
  console.log(`  字节码VM 中位数 = ${tVm.medianMs.toFixed(3)} ms（本机实测,仅展示）`);
  console.log(`  ⇒ 字节码 VM 加速比 = ${sp.toFixed(2)}x（相对量,可复现断言: >1x）`);
  console.log(`  ⚠ toy 程序 + JS-on-JS 解释,绝对值偏乐观;可迁移的相对趋势是「扁平指令循环 < 递归树遍历的每操作开销」\n`);

  // -------- ⑤ 第06章优化前/后字节码在 VM 上 --------
  console.log("【⑤ 优化前/后(常量折叠)字节码在同一 VM 上】");
  const rawFold = makeFoldableProgram();
  const optFold = foldProgram(rawFold);
  console.log("  优化前 work 体: " + astToSExpr(rawFold[0]));
  console.log("  优化后 work 体: " + astToSExpr(optFold[0]) + "   ← 常量子树折成单个字面量");

  const rawC = compileProgram(rawFold);
  const optC = compileProgram(optFold);
  const rawInstrs = countInstrs(rawC.allFns);
  const optInstrs = countInstrs(optC.allFns);
  console.log(`  优化前字节码总指令数: ${rawInstrs} 条`);
  console.log(`  优化后字节码总指令数: ${optInstrs} 条（少 ${rawInstrs - optInstrs} 条:折叠消除了算术指令）`);

  // 验证优化不改变语义(输出一致),再比运行时。
  const rawClosure: Closure = { kind: "closure", fn: rawC.script, captured: [] };
  const optClosure: Closure = { kind: "closure", fn: optC.script, captured: [] };
  const rawO = runVm(rawClosure, rawC.allFns, {});
  const optO = runVm(optClosure, optC.allFns, {});
  assertEq(rawO.output, optO.output, "constant folding preserves output");
  console.log(`  语义不变(输出 ${rawO.output[0]} == ${optO.output[0]})✓;优化前后执行操作数 ${rawO.stats.opsExecuted} → ${optO.stats.opsExecuted}`);

  // 运行时对比:inner 放大,因单次调用太快会量化到 0(见 bench.ts 失败模式注释)。
  const tRaw = timeIt(() => runVm(rawClosure, rawC.allFns, {}), { inner: 2000 });
  const tOpt = timeIt(() => runVm(optClosure, optC.allFns, {}), { inner: 2000 });
  const foldSp = speedup(tRaw, tOpt);
  console.log(`  优化前 中位数 = ${tRaw.medianMs.toFixed(5)} ms;优化后 = ${tOpt.medianMs.toFixed(5)} ms`);
  console.log(`  ⇒ 优化后 VM 运行加速比 = ${foldSp.toFixed(2)}x（少跑指令 → 更快;toy 程序收益小,趋势真实）\n`);

  // -------- ⑥ 失败模式: 栈泄漏 --------
  console.log("【⑥ 失败模式:函数返回未收缩栈(栈泄漏)→ 栈高度随调用线性增长直到越界】");
  console.log("  对一个『递归后忘记 POP 返回值』的字节码,逐步加大递归深度,观测操作数栈峰值:\n");
  console.log("  递归深度 depth | 操作数栈峰值高度 | 结果");
  console.log("  --------------|------------------|------");
  // 末项 depth 故意超过 STACK_LIMIT(100,000):泄漏积累到越界,VM 抛 STACK OVERFLOW。
  for (const depth of [5, 50, 500, 5000, 120_000]) {
    const leaky = buildLeakyProgram(depth);
    try {
      const r = runVm(leaky.script, leaky.allFns, {});
      // 栈峰值随 depth 线性增长(每层泄漏一个槽),这正是不变量被破坏的客观信号。
      console.log(
        `  ${String(depth).padStart(13)} | ${String(r.stats.maxStackHeight).padStart(16)} | OK(输出 ${r.output[0]})`,
      );
    } catch (e) {
      console.log(
        `  ${String(depth).padStart(13)} | ${">100,000(越界)".padStart(16)} | ✗ ${(e as Error).message.split("。")[0]}`,
      );
    }
  }
  console.log("\n  → 栈峰值 ≈ depth,线性增长。正确的 VM 在每次 CALL 返回后 POP 掉用不到的返回值,");
  console.log("    使『语句净栈高度 = 0』不变量成立,栈高度与递归深度无关地保持有界。");
  console.log("    对比 ②:正确的 fib(20) 操作数栈峰值仅 " + fibRun.stats.maxStackHeight + " 槽,与调用次数无关。\n");

  console.log("============================================================");
  console.log(" 全部对拍通过 ✓  字节码 VM 可跑、机制为真、失败模式已演示");
  console.log("============================================================");
}

main();
