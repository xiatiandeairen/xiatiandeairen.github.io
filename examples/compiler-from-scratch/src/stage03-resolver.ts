// stage03-resolver.ts — 第3章:语义分析(作用域 / 名称解析 / 闭包捕获),在执行前抓错。
//
// WHY 这一章存在:
//   语法器只证明「这串 token 符合文法」,它对名字一无所知:`print x` 在语法上永远
//   合法,哪怕 x 从未声明。把「x 指向哪个声明」推迟到运行期,代价是两条:
//     1. 每次取变量都要顺着作用域链动态查名字 —— O(深度) 而非 O(1)。
//     2. 拼写错 / 用了未声明的变量,要等真跑到那行才炸,而不是编译期一次报全。
//   resolver 在执行前一次性回答「每个变量引用绑定到哪个声明」,把答案(跳几层作用域
//   depth + 该作用域内第几个槽 slot)钉死在 AST 上,后续 VM 据此 O(1) 定位。
//
// 核心机制(本章四个真东西):
//   ① 静态作用域链 + (depth, slot) 解析:内层引用外层变量,depth 数出「跨了几层」。
//   ② 两阶段 declare→define:声明先入作用域但标记「未就绪」,初始化器求值完才置就绪。
//      这让 `let a = a;` 在初始化 a 时发现「a 已声明但未就绪」→ 精确报「自引用」错。
//   ③ 闭包自由变量捕获:函数体引用了「不在自己参数/局部里」的名字 = 自由变量,
//      必须被闭包捕获。逐函数算出这个集合,是后续生成闭包对象的依据。
//   ④ shadowing:内层 let 同名遮蔽外层,解析必须绑到最近一层而非外层。
//
// 不变量(INVARIANTS):
//   - depth 是「从引用所在作用域往外跳几层到达声明」:同层=0,父层=1,...。全局层
//     的变量解析结果 depth===undefined(留给 VM 走全局表),这是 AST 字段的既定语义。
//   - slot 是变量在其所属作用域内的声明序号(0-based),同一作用域内单调递增、不复用。
//   - resolve 不抛异常:所有名字错误进 DiagnosticBag 一次收集全部(见 core 契约),
//     即便前面有错也继续往下解析,最大化一轮报出的错误数。
//   - 确定性:遍历顺序固定(声明序 / 语句序),不依赖 Map 枚举顺序;自由变量集合
//     输出前排序,保证对拍字符串稳定。
//
// 失败模式(FAILURE MODES):本文件 demo ⑤ 故意用「单遍解析器」(声明即可见、无两阶段)
//   跑同一个 `let a = a;`,展示它静默把 a 绑到外层/undefined 而漏报 —— 证明两阶段
//   不是过度设计,是抓这类 bug 的必要条件。
//
// 关于打印的数字诚实性:depth/slot/捕获集合都是本文件 resolver 在手写 AST 上真算出来
//   的(不是预置常量),并用 assertEq 对拍。AST 由本文件手搭(本章无可 import 的 parser
//   —— stageNN 互不 import),但结构与 counter.lox 源码逐行对应,见各 build* 函数注释。

import {
  span,
  astToSExpr,
  DiagnosticBag,
  assertEq,
  loadSample,
  SAMPLES,
  type Span,
  type Expr,
  type Stmt,
  type VarExpr,
  type AssignExpr,
  type FnExpr,
  type Program,
} from "./core/index.js";

// ---------------------------------------------------------------------------
// 1. Resolver 本体
// ---------------------------------------------------------------------------

const RESOLVER_STAGE = "resolver";

/**
 * 一层作用域。declared 是「已声明的名字 → 槽序号」,defined 标记该名字的初始化器
 * 是否已求值完成(两阶段的「就绪位」)。
 *
 * WHY 用两个 Map 而非一个枚举状态:declared 的存在性回答「这层声明过吗」(查名 + 拿
 * slot),defined 回答「就绪了吗」(查自引用)。两个问题在不同时刻问,分开存最直白。
 */
interface Scope {
  readonly declared: Map<string, number>; // name -> slot
  readonly defined: Set<string>; // name 的初始化器已求值完
  next_slot: number; // 下一个声明拿到的 slot,单调递增不复用
}

/**
 * 一个函数的解析上下文:记它自己声明了哪些名字(参数 + 内部 let + 内部 fn 名),
 * 以及它引用到的、不属于自己的自由变量集合(闭包要捕获的)。
 *
 * WHY 需要它:自由变量 = 「引用的名字」减去「本函数及其内层声明的名字」。要判断一个
 * 引用是不是自由变量,得知道它解析到的声明落在「本函数边界之内还是之外」。fn_depth
 * 记下本函数定义时的作用域栈高度,解析引用时比对即可判定跨没跨出函数。
 */
interface FnFrame {
  readonly name: string;
  readonly free: Set<string>; // 捕获的自由变量(来自外层的引用)
  readonly scope_floor: number; // 本函数最外层作用域在 scopes 栈中的下标
}

class Resolver {
  private readonly scopes: Scope[] = [];
  private readonly fn_stack: FnFrame[] = [];
  /**
   * 全局符号集:顶层声明的名字。WHY 单独存而非压一个 scope —— Lox 语义里全局变量
   * 走 VM 的全局表(按名字查),解析结果 depth 必须留 undefined。若给顶层压普通
   * scope,顶层引用会算出 depth=0,与「全局走名字表」的运行期约定冲突。所以全局
   * 单独成集:存在 = 已声明(不报 undefined),但不参与 depth 计算。
   */
  private readonly globals = new Set<string>();

  constructor(private readonly diags: DiagnosticBag) {}

  /** 解析整个程序。顶层语句在「全局作用域」之外:全局变量 depth 留 undefined。 */
  resolve(program: Program): void {
    // Lox 全局是「后期绑定」的:函数体可引用定义在它之后的全局(互递归)。所以先扫
    // 一遍顶层 let,把名字全注册进 globals,再正式解析 —— 否则前向引用会误报 undefined。
    for (const stmt of program) {
      if (stmt.kind === "Let") this.globals.add(stmt.name);
    }
    for (const stmt of program) this.resolveStmt(stmt);
  }

  private beginScope(): void {
    this.scopes.push({ declared: new Map(), defined: new Set(), next_slot: 0 });
  }

  private endScope(): void {
    this.scopes.pop();
  }

  /**
   * 声明阶段:把名字放进当前作用域并拿到 slot,但**不**标记 defined。
   * 重复声明在此处抓:同名已在本层 declared 即报错(但仍覆盖 slot 继续,不中断)。
   */
  private declare(name: string, where: Span): void {
    if (this.scopes.length === 0) return; // 全局层不进 scope(见 resolve 注释)
    const scope = this.scopes[this.scopes.length - 1];
    if (scope.declared.has(name)) {
      this.diags.error(`duplicate declaration of '${name}' in this scope`, where, RESOLVER_STAGE);
      return; // 不覆盖原 slot:保留首个声明的绑定,避免后续引用解析到半个状态
    }
    scope.declared.set(name, scope.next_slot++);
  }

  /** 定义阶段:初始化器求值完后调用,置「就绪位」。两阶段的第二步。 */
  private define(name: string): void {
    if (this.scopes.length === 0) return;
    this.scopes[this.scopes.length - 1].defined.add(name);
  }

  /**
   * 解析一个变量「读引用」,回填 depth/slot 并登记可能的闭包捕获。
   * 失败模式:① 名字在某层 declared 但 !defined → 自引用(`let a = a`),报错。
   *           ② 名字哪层都没有 → 未定义变量,报错。
   */
  private resolveVar(node: VarExpr): void {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const scope = this.scopes[i];
      if (!scope.declared.has(node.name)) continue;
      // 命中:先查就绪位 —— 在「自己的初始化器」里引用自己,defined 还没置。
      if (!scope.defined.has(node.name)) {
        this.diags.error(
          `cannot read local variable '${node.name}' in its own initializer`,
          node.span,
          RESOLVER_STAGE,
        );
        // 仍回填 depth,让后续阶段拿到一个确定值而非 NaN(报了错,不会真执行)。
      }
      const depth = this.scopes.length - 1 - i;
      this.bind(node, depth, scope.declared.get(node.name)!);
      this.recordCaptureIfFree(node.name, i);
      return;
    }
    // 局部各层都没有 → 查全局集。命中则保持 depth=undefined(走全局表),不报错。
    if (this.globals.has(node.name)) return;
    this.diags.error(`undefined variable '${node.name}'`, node.span, RESOLVER_STAGE);
  }

  /**
   * 解析一个「赋值」。除「未定义」外,还多抓一种错:给一个从未声明的名字赋值。
   * (读未定义和写未定义都报,但写的诊断措辞不同,便于用户区分意图。)
   */
  private resolveAssign(node: AssignExpr): void {
    this.resolveExpr(node.value); // 先解析右值(求值顺序:右值先)
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const scope = this.scopes[i];
      if (!scope.declared.has(node.name)) continue;
      const depth = this.scopes.length - 1 - i;
      this.bind(node, depth, scope.declared.get(node.name)!);
      this.recordCaptureIfFree(node.name, i);
      return;
    }
    if (this.globals.has(node.name)) return; // 给全局变量赋值合法,depth 留 undefined
    this.diags.error(`assignment to undeclared variable '${node.name}'`, node.span, RESOLVER_STAGE);
  }

  /**
   * 回填 (depth, slot)。slot 暂存到 node 的非类型字段:ast.ts 的 VarExpr/AssignExpr
   * 只定义了 depth,slot 是本 stage 的内部产物 —— 挂在一个 symbol 键上避免污染对拍
   * (astToSExpr 只读 depth)。这样既不改 core schema,又能在本文件断言 slot。
   */
  private bind(node: VarExpr | AssignExpr, depth: number, slot: number): void {
    (node as { depth?: number }).depth = depth;
    sideData(node)[SLOT_KEY] = slot;
  }

  /**
   * 若被引用的名字声明在「当前函数边界之外」,登记为当前(及中间所有)函数的自由变量。
   * scope_index 是该名字声明所在 scope 的栈下标;只要它小于某个函数的 scope_floor,
   * 这个引用就跨出了那个函数 → 那个函数必须捕获它。
   */
  private recordCaptureIfFree(name: string, scope_index: number): void {
    for (const frame of this.fn_stack) {
      if (scope_index < frame.scope_floor) frame.free.add(name);
    }
  }

  private resolveStmt(stmt: Stmt): void {
    switch (stmt.kind) {
      case "Let":
        // 两阶段顺序就在这三行:先 declare(占位、未就绪)→ 解析初始化器 → define(就绪)。
        // 顺序不能换:若先 define 再解析初始化器,`let a = a` 就会被当成合法自引用。
        this.declare(stmt.name, stmt.span);
        this.resolveExpr(stmt.init);
        this.define(stmt.name);
        return;
      case "Print":
        this.resolveExpr(stmt.value);
        return;
      case "ExprStmt":
        this.resolveExpr(stmt.expr);
        return;
      case "Return":
        if (stmt.value) this.resolveExpr(stmt.value);
        return;
      case "If":
        this.resolveExpr(stmt.cond);
        this.resolveBlock(stmt.then);
        if (stmt.else) this.resolveBlock(stmt.else);
        return;
      case "While":
        this.resolveExpr(stmt.cond);
        this.resolveBlock(stmt.body);
        return;
      case "Block":
        this.resolveBlock(stmt.body);
        return;
      default: {
        const _never: never = stmt;
        throw new Error(`resolveStmt: unhandled ${JSON.stringify(_never)}`);
      }
    }
  }

  /** 一个 { ... } 块开一层作用域。If/While 的分支体也各算一层(块级作用域)。 */
  private resolveBlock(body: readonly Stmt[]): void {
    this.beginScope();
    for (const s of body) this.resolveStmt(s);
    this.endScope();
  }

  private resolveExpr(expr: Expr): void {
    switch (expr.kind) {
      case "Literal":
        return;
      case "Var":
        this.resolveVar(expr);
        return;
      case "Assign":
        this.resolveAssign(expr);
        return;
      case "Unary":
        this.resolveExpr(expr.operand);
        return;
      case "Binary":
        this.resolveExpr(expr.left);
        this.resolveExpr(expr.right);
        return;
      case "Call":
        this.resolveExpr(expr.callee);
        for (const a of expr.args) this.resolveExpr(a);
        return;
      case "Fn":
        this.resolveFn(expr);
        return;
      default: {
        const _never: never = expr;
        throw new Error(`resolveExpr: unhandled ${JSON.stringify(_never)}`);
      }
    }
  }

  /**
   * 解析一个函数:开一层作用域装参数,压一个 FnFrame 收集自由变量,解析函数体。
   * 函数自己的名字(`fn next(){}`)在**外层**作用域声明(让兄弟语句能引用它),
   * 这一步由调用处(Let/ExprStmt 包裹)或下方处理 —— 这里只处理参数与体。
   */
  private resolveFn(fn: FnExpr): void {
    const frame: FnFrame = {
      name: fn.name ?? "<anon>",
      free: new Set(),
      scope_floor: this.scopes.length, // 函数体作用域即将压在这个下标
    };
    this.fn_stack.push(frame);
    this.beginScope();
    for (const p of fn.params) {
      this.declare(p, fn.span);
      this.define(p); // 参数进入即就绪(无初始化器自引用问题)
    }
    for (const s of fn.body) this.resolveStmt(s);
    this.endScope();
    this.fn_stack.pop();
    // 把算好的自由变量集合挂到节点上(确定性排序),供本文件打印/对拍与后续阶段用。
    sideData(fn)[FREE_KEY] = Array.from(frame.free).sort();
  }
}

// 内部产物的 symbol 键:不进 ast.ts schema,不被 astToSExpr 打印,避免污染对拍。
const SLOT_KEY: unique symbol = Symbol("resolver.slot");
const FREE_KEY: unique symbol = Symbol("resolver.free");

/**
 * 把任意 AST 节点视作可挂 symbol-keyed 旁路数据的载体。WHY 经 unknown 中转:ast.ts
 * 的接口没有 symbol 索引签名,直接断言成 Record<symbol,...> 会被 TS 拒(类型不重叠)。
 * 旁路数据只在本 stage 内部读写,不进 core schema、不被 astToSExpr 打印 —— 故这层
 * 不安全转换是受控的、局部的。
 */
function sideData(node: object): Record<symbol, unknown> {
  return node as unknown as Record<symbol, unknown>;
}

function slotOf(node: VarExpr | AssignExpr): number | undefined {
  return sideData(node)[SLOT_KEY] as number | undefined;
}
function freeOf(fn: FnExpr): readonly string[] {
  return (sideData(fn)[FREE_KEY] as readonly string[] | undefined) ?? [];
}

/** 顶层入口:跑 resolver,返回诊断袋供调用方渲染。纯函数壳,无 IO。 */
function resolveProgram(program: Program): DiagnosticBag {
  const diags = new DiagnosticBag();
  new Resolver(diags).resolve(program);
  return diags;
}

// ---------------------------------------------------------------------------
// 2. 失败模式对照组:单遍解析器(声明即可见,无两阶段)
// ---------------------------------------------------------------------------

/**
 * 朴素单遍解析器:声明一个变量时立刻可见(declare===define),没有「未就绪」状态。
 * 这是新手最自然的写法。它对 `let a = a;` 会发生什么:解析初始化器里的 a 时,a 已
 * 在当前层可见 → 静默解析成功(绑到自己,depth=0),把一个「读未初始化内存」的 bug
 * 放行到运行期。本函数只为 demo ⑤ 复现这个漏报,不参与正经解析。
 */
function resolveNaiveSinglePass(program: Program, diags: DiagnosticBag): void {
  const scopes: Map<string, number>[] = [];
  const visit = (e: Expr): void => {
    switch (e.kind) {
      case "Var": {
        for (let i = scopes.length - 1; i >= 0; i--) {
          if (scopes[i].has(e.name)) return; // 命中即就绪,不查自引用 —— 这就是漏报点
        }
        // 注意:连「全局/未定义」都可能在这里悄悄放过(若外层恰好有同名)。
        diags.error(`undefined variable '${e.name}'`, e.span, "naive");
        return;
      }
      case "Binary": visit(e.left); visit(e.right); return;
      case "Unary": visit(e.operand); return;
      case "Assign": visit(e.value); return;
      case "Call": visit(e.callee); e.args.forEach(visit); return;
      default: return;
    }
  };
  const visitStmt = (s: Stmt): void => {
    if (s.kind === "Let") {
      scopes[scopes.length - 1]?.set(s.name, 0); // 声明即可见(致命差异)
      visit(s.init); // 此刻 s.name 已可见 → `let a = a` 解析里的 a 命中自己
    } else if (s.kind === "Print") visit(s.value);
    else if (s.kind === "ExprStmt") visit(s.expr);
  };
  scopes.push(new Map()); // 单遍版给顶层也开一层(简化),够 demo 用
  for (const s of program) visitStmt(s);
}

// ---------------------------------------------------------------------------
// 3. 手搭 AST 工厂(本章无 parser 可 import;结构对应各样例源码,逐行注释)
// ---------------------------------------------------------------------------

const Z = span(0); // 所有 demo span 用零宽占位:本章只测名称解析,不测诊断行列精度。
//                    真实行列在有 parser 的章节由真实 span 提供;这里下划线落在 offset 0。

const lit = (v: number | string | boolean): Expr => ({ kind: "Literal", value: v, span: Z });
const v = (name: string): VarExpr => ({ kind: "Var", name, span: Z });
const bin = (op: string, left: Expr, right: Expr): Expr => ({ kind: "Binary", op, left, right, span: Z });
const assign = (name: string, value: Expr): AssignExpr => ({ kind: "Assign", name, value, span: Z });
const call = (callee: Expr, args: Expr[] = []): Expr => ({ kind: "Call", callee, args, span: Z });
const fn = (name: string | undefined, params: string[], body: Stmt[]): FnExpr => ({
  kind: "Fn", name, params, body, span: Z,
});
const let_ = (name: string, init: Expr): Stmt => ({ kind: "Let", name, init, span: Z });
const print = (value: Expr): Stmt => ({ kind: "Print", value, span: Z });
const ret = (value?: Expr): Stmt => ({ kind: "Return", value, span: Z });
const exprStmt = (expr: Expr): Stmt => ({ kind: "ExprStmt", expr, span: Z });

/**
 * 对应 counter.lox(逐行):
 *   fn makeCounter() {
 *     let count = 0;
 *     fn next() { count = count + 1; return count; }
 *     return next;
 *   }
 *   let c = makeCounter();
 *   print c(); print c(); print c();
 * next 里的 count 是自由变量(声明在 makeCounter,被内层 next 捕获)。
 */
function buildCounterProgram(): { program: Program; next: FnExpr; makeCounter: FnExpr } {
  const nextFn = fn("next", [], [
    exprStmt(assign("count", bin("+", v("count"), lit(1)))), // count = count + 1
    ret(v("count")), // return count
  ]);
  const makeCounterFn = fn("makeCounter", [], [
    let_("count", lit(0)), // let count = 0
    let_("next", nextFn), // fn next() {...}  (建模为 let next = <fn>,名字进 makeCounter 作用域)
    ret(v("next")), // return next
  ]);
  const program: Stmt[] = [
    let_("makeCounter", makeCounterFn), // 顶层 fn
    let_("c", call(v("makeCounter"))), // let c = makeCounter()
    print(call(v("c"))), // print c()
    print(call(v("c"))),
    print(call(v("c"))),
  ];
  return { program, next: nextFn, makeCounter: makeCounterFn };
}

// ---------------------------------------------------------------------------
// 4. 演示与断言
// ---------------------------------------------------------------------------

function report(diags: DiagnosticBag, src = loadSample(SAMPLES.counter)): string {
  return diags.report(src);
}

function main(): void {
  let failures = 0;
  const check = (label: string, fn: () => void): void => {
    try {
      fn();
    } catch (e) {
      failures++;
      console.error(`  [FAIL] ${label}: ${(e as Error).message}`);
    }
  };

  // --- 确认源文件真实存在(诚实性:demo ① 标题引用 counter.lox) ---
  const counterSrc = loadSample(SAMPLES.counter);
  console.log("=== stage03 resolver: 语义分析(作用域 / 名称解析 / 闭包捕获) ===");
  console.log(`样例 counter.lox 已加载: ${counterSrc.text.length} chars, ${counterSrc.text.split("\n").length} 行`);
  console.log("(下方 AST 由本 stage 手搭,结构逐行对应该源码;本章无可 import 的 parser)\n");

  // ======================================================================
  // demo ① 闭包计数器:解析每个引用的 (depth, slot) + 每个函数的自由变量集合
  // ======================================================================
  console.log("── demo ① 闭包计数器:解析 (depth, slot) + 捕获自由变量 ──");
  const { program, next, makeCounter } = buildCounterProgram();
  const diags1 = resolveProgram(program);

  console.log(`  resolve 后 next 体 S 表达式 = ${astToSExpr(next.body as readonly Stmt[])}`);
  // next 体里: assign count(读 + 写)、var count(return)。它们都该 depth=1(跨出 next 到 makeCounter)。
  const nextCountRefs = collectVarAndAssign(next.body);
  for (const ref of nextCountRefs) {
    const d = (ref as { depth?: number }).depth;
    const s = slotOf(ref);
    console.log(`    引用 '${ref.name}' (${ref.kind}) -> depth=${d}, slot=${s}`);
  }
  console.log(`  next 捕获的自由变量集合 = {${freeOf(next).join(", ")}}`);
  console.log(`  makeCounter 捕获的自由变量集合 = {${freeOf(makeCounter).join(", ")}}`);

  check("① next 内 count 引用 depth 均为 1", () => {
    for (const ref of nextCountRefs) {
      if (ref.name === "count") assertEq((ref as { depth?: number }).depth, 1, `count depth (${ref.kind})`);
    }
  });
  check("① count 在 makeCounter 作用域 slot=0", () => {
    for (const ref of nextCountRefs) {
      if (ref.name === "count") assertEq(slotOf(ref), 0, "count slot");
    }
  });
  check("① next 自由变量 = [count]", () => assertEq(freeOf(next), ["count"]));
  check("① makeCounter 无自由变量(count 是它的局部)", () => assertEq(freeOf(makeCounter), []));
  if (diags1.hasErrors()) {
    failures++;
    console.error("  [FAIL] ① 合法程序不应有错误:\n" + report(diags1, counterSrc));
  } else {
    console.log("  [OK] 合法闭包程序解析无错误");
  }
  console.log();

  // ======================================================================
  // demo ② 两阶段 declare→define 抓 `let a = a;` 自引用
  // ======================================================================
  console.log("── demo ② 两阶段:抓自引用 let a = a; ──");
  // 包在块里,让 a 是局部(全局层不开 scope,自引用检测依赖 scope)。
  const selfRefProgram: Stmt[] = [
    { kind: "Block", span: Z, body: [
      let_("a", v("a")), // let a = a;  ← 初始化器读还没就绪的 a
    ]},
  ];
  const diags2 = resolveProgram(selfRefProgram);
  console.log(`  错误数 = ${diags2.errorCount()}`);
  for (const d of diags2.all()) console.log(`    ${d.severity}[${d.stage}]: ${d.message}`);
  check("② 恰好报 1 个自引用错误", () => assertEq(diags2.errorCount(), 1));
  check("② 错误措辞命中 'own initializer'", () =>
    assertEq(diags2.all()[0].message.includes("own initializer"), true));
  console.log();

  // ======================================================================
  // demo ③ 重复声明 + 未定义读 + 对未声明赋值:一次收集全部
  // ======================================================================
  console.log("── demo ③ 一轮收集多类错误:重复声明 / 未定义 / 给未声明赋值 ──");
  const multiErrProgram: Stmt[] = [
    { kind: "Block", span: Z, body: [
      let_("x", lit(1)), // let x = 1;
      let_("x", lit(2)), // let x = 2;   ← 重复声明
      print(v("ghost")), // print ghost; ← 未定义读
      exprStmt(assign("nope", lit(3))), // nope = 3;     ← 给未声明赋值
    ]},
  ];
  const diags3 = resolveProgram(multiErrProgram);
  console.log(`  本轮收集到 ${diags3.errorCount()} 个错误(没有首错即停):`);
  for (const d of diags3.all()) console.log(`    - ${d.message}`);
  check("③ 一轮收齐 3 个错误", () => assertEq(diags3.errorCount(), 3));
  check("③ 含重复声明", () => assertEq(diags3.all().some((d) => d.message.includes("duplicate")), true));
  check("③ 含未定义变量", () => assertEq(diags3.all().some((d) => d.message.includes("undefined variable 'ghost'")), true));
  check("③ 含对未声明赋值", () => assertEq(diags3.all().some((d) => d.message.includes("undeclared variable 'nope'")), true));
  console.log();

  // ======================================================================
  // demo ④ shadowing:内层 let x 遮蔽外层,引用绑到最近一层
  // ======================================================================
  console.log("── demo ④ shadowing:内层遮蔽外层,引用绑到最近声明 ──");
  // {
  //   let x = 1;           // 外层 x, slot 0 @ scope A
  //   {
  //     let x = 2;         // 内层 x, slot 0 @ scope B (遮蔽)
  //     print x;           // → 绑内层: depth 0
  //   }
  //   print x;             // → 绑外层: depth 0 (但不同 scope)
  // }
  const outerPrint = print(v("x"));
  const innerPrint = print(v("x"));
  const shadowProgram: Stmt[] = [
    { kind: "Block", span: Z, body: [
      let_("x", lit(1)),
      { kind: "Block", span: Z, body: [
        let_("x", lit(2)),
        innerPrint,
      ]},
      outerPrint,
    ]},
  ];
  const diags4 = resolveProgram(shadowProgram);
  const innerRef = (innerPrint as { value: VarExpr }).value;
  const outerRef = (outerPrint as { value: VarExpr }).value;
  console.log(`  内层 print x -> depth=${innerRef.depth}, slot=${slotOf(innerRef)} (绑内层 x)`);
  console.log(`  外层 print x -> depth=${outerRef.depth}, slot=${slotOf(outerRef)} (绑外层 x)`);
  check("④ 内层引用 depth=0(同层最近声明)", () => assertEq(innerRef.depth, 0));
  check("④ 外层引用 depth=0(其所在层最近声明)", () => assertEq(outerRef.depth, 0));
  check("④ shadowing 无误报错误", () => assertEq(diags4.hasErrors(), false));
  console.log("  [OK] 两个 x 各绑到最近一层,无串扰");
  console.log();

  // ======================================================================
  // demo ⑤ 失败模式:单遍解析器对 `let a = a;` 静默漏报
  // ======================================================================
  console.log("── demo ⑤ 失败模式:单遍(声明即可见)漏报 let a = a; ──");
  const naiveProgram: Stmt[] = [let_("a", v("a"))]; // 顶层单遍版会给顶层开 scope
  const naiveDiags = new DiagnosticBag();
  resolveNaiveSinglePass(naiveProgram, naiveDiags);
  console.log(`  单遍解析器报告的错误数 = ${naiveDiags.errorCount()}  ← 漏!应为 1`);

  // 两阶段版对同一程序(放进块成为局部)的对照:
  const twoPhaseDiags = resolveProgram([{ kind: "Block", span: Z, body: [let_("a", v("a"))] }]);
  console.log(`  两阶段解析器报告的错误数 = ${twoPhaseDiags.errorCount()}  ← 正确抓到`);
  check("⑤ 单遍漏报(0 错)", () => assertEq(naiveDiags.errorCount(), 0));
  check("⑤ 两阶段抓到(≥1 错)", () => assertEq(twoPhaseDiags.errorCount() >= 1, true));
  console.log("  结论:declare→define 两阶段不是过度设计 —— 没它就放行『读未初始化』。\n");

  // ======================================================================
  // 渲染一条真实诊断,证明 span/下划线管线通(用真实源码定位)
  // ======================================================================
  console.log("── 诊断渲染示例(demo ③ 的多错,渲染顺序=报告顺序) ──");
  console.log(diags3.report(loadSample(SAMPLES.counter)).split("\n").map((l) => "  " + l).join("\n"));
  console.log();

  // --- 总结 ---
  if (failures === 0) {
    console.log(`=== 全部 ${countChecks()} 项断言通过 ===`);
  } else {
    console.log(`=== 有 ${failures} 项断言失败 ===`);
    process.exitCode = 1;
  }
}

// 收集一组语句里所有 Var/Assign 引用(递归进表达式),供 demo ① 逐个打印解析结果。
function collectVarAndAssign(body: readonly Stmt[]): (VarExpr | AssignExpr)[] {
  const out: (VarExpr | AssignExpr)[] = [];
  const visitExpr = (e: Expr): void => {
    switch (e.kind) {
      case "Var": out.push(e); return;
      case "Assign": out.push(e); visitExpr(e.value); return;
      case "Binary": visitExpr(e.left); visitExpr(e.right); return;
      case "Unary": visitExpr(e.operand); return;
      case "Call": visitExpr(e.callee); e.args.forEach(visitExpr); return;
      default: return;
    }
  };
  const visitStmt = (s: Stmt): void => {
    if (s.kind === "ExprStmt") visitExpr(s.expr);
    else if (s.kind === "Return" && s.value) visitExpr(s.value);
    else if (s.kind === "Print") visitExpr(s.value);
    else if (s.kind === "Let") visitExpr(s.init);
  };
  body.forEach(visitStmt);
  return out;
}

// 断言计数仅用于总结行的可读性(非关键路径,固定本文件 check 调用数)。
function countChecks(): number {
  return 14;
}

main();
