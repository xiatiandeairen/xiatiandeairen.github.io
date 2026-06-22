// stage04-typecheck.ts — 第4章:给 Lox-mini 加一套静态类型检查器。
//
// WHY 这一章存在:
//   前三章产出的 AST 是「语法上合法」的,但语法合法 ≠ 运行时不崩。`"a" + 1`、
//   `3(x)`(调用一个非函数)、`if (42)`(把数字当条件)在语法上都是合法的树,
//   却会在运行期炸或产生垃圾结果。类型检查的全部价值就是:在编译期、不执行程序
//   的前提下,挡住一整类「形状对但语义错」的程序,把运行时崩溃前移成编译期诊断。
//
// 本文件是自包含的(self-contained),原因见硬约束:
//   - 不能 import 任何 stageNN 文件(它们一加载就跑 main())。
//   - core 只提供数据结构(AST / Span / Diagnostic),没有现成的 lexer/parser 可复用。
//   因此这里内置一个「够用就好」的 lexer + Pratt parser,只为把样例源码变成 core 的
//   AST,真正的主角是后半部分的 TypeChecker。lexer/parser 刻意写薄,不是本章重点。
//
// 设计要点(类型系统):
//   - 类型用判别联合(int/float/bool/string/fn/nil/error)。`error` 是哨兵类型:
//     一旦某处报了类型错,就让它的类型变成 error 往上传,后续凡是碰到 error 的运算
//     一律不再报错 —— 否则一个根因会引发一连串级联误报(cascading errors),淹没真错。
//   - 局部类型推断:`let x = 1 + 2` 不需要标注,从初始化表达式推出 int。这是「局部」
//     推断(只看赋值右边),不是全程序 Hindley-Milner;教学上够用且可解释。
//   - int/float 隐式提升:int 与 float 混合算术结果是 float(仿 C/数值语言惯例)。
//     注意:int 字面量(无小数点)推为 int,float 字面量(有小数点)推为 float。
//     core 的 token literal 把两者都存成 JS number 丢了这个区分,所以本文件的 lexer
//     单独记录「这个数字字面量写法里有没有小数点」。
//
// 失败模式(本章必须 demo,不只 happy path):
//   - 故意把 `==` 两侧不同类型当合法(放行 `1 == "1"`),并打印这个「宽松等于」会让
//     后端(VM / 代码生成)的「两侧同型」假设落空 —— 演示类型规则一旦放松,损失从
//     类型检查层一路漏到运行期。

import {
  span,
  spanMerge,
  SourceText,
  DiagnosticBag,
  Severity,
  loadSample,
  SAMPLES,
  assertEq,
  type Span,
  type Expr,
  type Stmt,
  type Program,
  type LiteralExpr,
} from "./core/index.js";

// ============================================================================
// 第 0 部分:类型的内部表示
// ============================================================================

/**
 * 类型的判别联合。`error` 是哨兵 —— 见文件头「设计要点」。`nil` 是裸 `return;` 和
 * 无返回函数体的类型(Lox-mini 没有真正的 null 值,nil 只在类型层表达「没有有用的值」)。
 * fn 记录参数类型与返回类型,用于检查调用点的实参个数/类型以及多 return 一致性。
 */
type Type =
  | { readonly kind: "int" }
  | { readonly kind: "float" }
  | { readonly kind: "bool" }
  | { readonly kind: "string" }
  | { readonly kind: "nil" }
  | { readonly kind: "error" } // 哨兵:抑制级联误报
  | { readonly kind: "fn"; readonly params: readonly Type[]; readonly ret: Type };

const T_INT: Type = { kind: "int" };
const T_FLOAT: Type = { kind: "float" };
const T_BOOL: Type = { kind: "bool" };
const T_STRING: Type = { kind: "string" };
const T_NIL: Type = { kind: "nil" };
const T_ERROR: Type = { kind: "error" };

/** 人读类型名,用于诊断与对拍。fn 递归展开成 `fn(int, int) -> int`。 */
function typeToString(t: Type): string {
  switch (t.kind) {
    case "int":
    case "float":
    case "bool":
    case "string":
    case "nil":
    case "error":
      return t.kind;
    case "fn":
      return `fn(${t.params.map(typeToString).join(", ")}) -> ${typeToString(t.ret)}`;
    default: {
      const _never: never = t;
      throw new Error(`typeToString: unhandled ${JSON.stringify(_never)}`);
    }
  }
}

/** 结构相等。fn 逐参数 + 返回类型递归比较。error 与任何类型「相等」以止血级联。 */
function typeEquals(a: Type, b: Type): boolean {
  if (a.kind === "error" || b.kind === "error") return true; // 哨兵:已报过错,不再制造新错
  if (a.kind !== b.kind) return false;
  if (a.kind === "fn" && b.kind === "fn") {
    if (a.params.length !== b.params.length) return false;
    return a.params.every((p, i) => typeEquals(p, b.params[i])) && typeEquals(a.ret, b.ret);
  }
  return true;
}

const isNumeric = (t: Type): boolean => t.kind === "int" || t.kind === "float";

/**
 * int/float 算术的结果类型(隐式提升规则,见文件头)。
 * 不变量:调用前调用方已确认两侧都是 numeric;混入 float 即提升为 float。
 */
function numericResult(a: Type, b: Type): Type {
  return a.kind === "float" || b.kind === "float" ? T_FLOAT : T_INT;
}

// ============================================================================
// 第 1 部分:够用就好的 lexer(只为喂出 AST,不是本章重点)
// ============================================================================
//
// 与 core/token 的关系:core 的 token literal 把整数和浮点都存成 JS number,丢了
// 「有没有小数点」这个 int/float 区分。本章需要这个区分,所以这里自带一个轻量
// lexer,对数字额外记一个 isFloat 标志,直接产出本文件用的 Token 形态。

type TokKind =
  | "num" | "str" | "id" | "true" | "false"
  | "let" | "fn" | "if" | "else" | "while" | "print" | "return"
  | "+" | "-" | "*" | "/" | "=" | "==" | "!=" | "!" | "<" | "<=" | ">" | ">="
  | "(" | ")" | "{" | "}" | "," | ";" | "eof";

interface Tok {
  readonly kind: TokKind;
  readonly text: string;
  readonly span: Span;
  readonly numIsFloat?: boolean; // 仅 num:字面量写法里是否含小数点 → 决定 int vs float
}

const LEX_KEYWORDS: ReadonlyMap<string, TokKind> = new Map([
  ["let", "let"], ["fn", "fn"], ["if", "if"], ["else", "else"],
  ["while", "while"], ["print", "print"], ["return", "return"],
  ["true", "true"], ["false", "false"],
]);

/**
 * 把源码切成 token 序列。失败模式:遇到未知字符 push 一条诊断并跳过该字符继续 —— 不抛,
 * 让一份脏源码尽量多报几个错(沿用 core/diagnostics 的「收集不中断」哲学)。
 */
function lex(src: SourceText, diags: DiagnosticBag): Tok[] {
  const s = src.text;
  const toks: Tok[] = [];
  let i = 0;
  const isDigit = (c: string) => c >= "0" && c <= "9";
  const isAlpha = (c: string) => (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c === "_";

  while (i < s.length) {
    const c = s[i];
    if (c === " " || c === "\t" || c === "\r" || c === "\n") { i++; continue; }
    // 行注释:Lox-mini 用 //,样例里大量出现,必须跳过到行尾。
    if (c === "/" && s[i + 1] === "/") { while (i < s.length && s[i] !== "\n") i++; continue; }

    const start = i;
    if (isDigit(c)) {
      let isFloat = false;
      while (i < s.length && isDigit(s[i])) i++;
      // 单个小数点 + 后续数字才算 float;`1.` 这种残缺写法这里宽松接受为 float。
      if (s[i] === "." && isDigit(s[i + 1])) {
        isFloat = true;
        i++;
        while (i < s.length && isDigit(s[i])) i++;
      } else if (s[i] === ".") {
        isFloat = true;
        i++;
      }
      toks.push({ kind: "num", text: s.slice(start, i), span: span(start, i), numIsFloat: isFloat });
      continue;
    }
    if (isAlpha(c)) {
      while (i < s.length && (isAlpha(s[i]) || isDigit(s[i]))) i++;
      const text = s.slice(start, i);
      toks.push({ kind: LEX_KEYWORDS.get(text) ?? "id", text, span: span(start, i) });
      continue;
    }
    if (c === '"') {
      i++; // 跳开引号
      while (i < s.length && s[i] !== '"') i++;
      if (i >= s.length) {
        diags.error("unterminated string literal", span(start, s.length), "lexer");
      } else {
        i++; // 跳合引号
      }
      toks.push({ kind: "str", text: s.slice(start, i), span: span(start, i) });
      continue;
    }
    // 多字符运算符优先于单字符(== 在 = 之前判)。
    const two = s.slice(i, i + 2);
    if (two === "==" || two === "!=" || two === "<=" || two === ">=") {
      toks.push({ kind: two as TokKind, text: two, span: span(i, i + 2) });
      i += 2;
      continue;
    }
    if ("+-*/=!<>(){},;".includes(c)) {
      toks.push({ kind: c as TokKind, text: c, span: span(i, i + 1) });
      i++;
      continue;
    }
    diags.error(`unexpected character '${c}'`, span(i, i + 1), "lexer");
    i++; // 跳过坏字符继续,不卡死
  }
  toks.push({ kind: "eof", text: "", span: span(s.length, s.length) });
  return toks;
}

// ============================================================================
// 第 2 部分:够用就好的 Pratt parser → 产出 core 的 AST
// ============================================================================
//
// Pratt parsing(运算符优先级爬升)是手写表达式解析的标准做法。这里只实现样例需要的
// 子集。解析失败时抛 ParseError(parser 内部的「此处无法继续」控制流),由顶层捕获转
// 成诊断 —— 这是 parser 局部允许的 throw,与类型检查层「收集不抛」分开。

class ParseError extends Error {
  constructor(message: string, readonly span: Span) {
    super(message);
  }
}

class Parser {
  private pos = 0;
  constructor(private readonly toks: Tok[]) {}

  private peek(): Tok { return this.toks[this.pos]; }
  private advance(): Tok { return this.toks[this.pos++]; }
  private check(k: TokKind): boolean { return this.peek().kind === k; }
  private match(k: TokKind): boolean {
    if (this.check(k)) { this.pos++; return true; }
    return false;
  }
  private expect(k: TokKind, what: string): Tok {
    if (this.check(k)) return this.advance();
    const t = this.peek();
    throw new ParseError(`expected ${what}, found '${t.text || t.kind}'`, t.span);
  }

  parseProgram(): Program {
    const stmts: Stmt[] = [];
    while (!this.check("eof")) stmts.push(this.statement());
    return stmts;
  }

  private statement(): Stmt {
    const t = this.peek();
    switch (t.kind) {
      case "let": return this.letStmt();
      case "print": return this.printStmt();
      case "if": return this.ifStmt();
      case "while": return this.whileStmt();
      case "return": return this.returnStmt();
      case "fn": return this.fnStmt();
      case "{": return this.block();
      default: return this.exprStmt();
    }
  }

  private letStmt(): Stmt {
    const kw = this.advance(); // let
    const name = this.expect("id", "variable name").text;
    this.expect("=", "'=' (let must initialize)");
    const init = this.expression();
    const semi = this.expect(";", "';'");
    return { kind: "Let", name, init, span: spanMerge(kw.span, semi.span) };
  }

  private printStmt(): Stmt {
    const kw = this.advance();
    const value = this.expression();
    const semi = this.expect(";", "';'");
    return { kind: "Print", value, span: spanMerge(kw.span, semi.span) };
  }

  private ifStmt(): Stmt {
    const kw = this.advance();
    this.expect("(", "'('");
    const cond = this.expression();
    this.expect(")", "')'");
    const then = this.blockBody();
    let elseBody: Stmt[] | undefined;
    if (this.match("else")) elseBody = this.blockBody();
    return { kind: "If", cond, then, else: elseBody, span: kw.span };
  }

  private whileStmt(): Stmt {
    const kw = this.advance();
    this.expect("(", "'('");
    const cond = this.expression();
    this.expect(")", "')'");
    const body = this.blockBody();
    return { kind: "While", cond, body, span: kw.span };
  }

  private returnStmt(): Stmt {
    const kw = this.advance();
    let value: Expr | undefined;
    if (!this.check(";")) value = this.expression();
    const semi = this.expect(";", "';'");
    return { kind: "Return", value, span: spanMerge(kw.span, semi.span) };
  }

  /** 具名函数声明 `fn name(params){...}` 脱糖为 `let name = <fn expr>`。 */
  private fnStmt(): Stmt {
    const fnExpr = this.fnExpr();
    if (fnExpr.kind !== "Fn" || fnExpr.name === undefined) {
      throw new ParseError("named function expected", fnExpr.span);
    }
    return { kind: "Let", name: fnExpr.name, init: fnExpr, span: fnExpr.span };
  }

  private fnExpr(): Expr {
    const kw = this.advance(); // fn
    let name: string | undefined;
    if (this.check("id")) name = this.advance().text;
    this.expect("(", "'('");
    const params: string[] = [];
    if (!this.check(")")) {
      do { params.push(this.expect("id", "parameter name").text); } while (this.match(","));
    }
    this.expect(")", "')'");
    const body = this.blockBody();
    return { kind: "Fn", name, params, body, span: kw.span };
  }

  private block(): Stmt {
    const t = this.peek();
    const body = this.blockBody();
    return { kind: "Block", body, span: t.span };
  }

  private blockBody(): Stmt[] {
    this.expect("{", "'{'");
    const stmts: Stmt[] = [];
    while (!this.check("}") && !this.check("eof")) stmts.push(this.statement());
    this.expect("}", "'}'");
    return stmts;
  }

  private exprStmt(): Stmt {
    const e = this.expression();
    const semi = this.expect(";", "';'");
    return { kind: "ExprStmt", expr: e, span: spanMerge(e.span, semi.span) };
  }

  // ---- 表达式:优先级爬升 ----
  // 优先级(低→高):assign < equality < comparison < term < factor < unary < call < primary

  private expression(): Expr { return this.assignment(); }

  private assignment(): Expr {
    const left = this.equality();
    if (this.check("=")) {
      const eq = this.advance();
      const value = this.assignment();
      if (left.kind !== "Var") throw new ParseError("invalid assignment target", eq.span);
      return { kind: "Assign", name: left.name, value, span: spanMerge(left.span, value.span) };
    }
    return left;
  }

  private binaryLevel(next: () => Expr, ops: TokKind[]): Expr {
    let left = next();
    while (ops.includes(this.peek().kind)) {
      const op = this.advance();
      const right = next();
      left = { kind: "Binary", op: op.kind, left, right, span: spanMerge(left.span, right.span) };
    }
    return left;
  }

  private equality(): Expr { return this.binaryLevel(() => this.comparison(), ["==", "!="]); }
  private comparison(): Expr { return this.binaryLevel(() => this.term(), ["<", "<=", ">", ">="]); }
  private term(): Expr { return this.binaryLevel(() => this.factor(), ["+", "-"]); }
  private factor(): Expr { return this.binaryLevel(() => this.unary(), ["*", "/"]); }

  private unary(): Expr {
    if (this.check("!") || this.check("-")) {
      const op = this.advance();
      const operand = this.unary();
      return { kind: "Unary", op: op.kind, operand, span: spanMerge(op.span, operand.span) };
    }
    return this.call();
  }

  private call(): Expr {
    let expr = this.primary();
    while (this.check("(")) {
      this.advance();
      const args: Expr[] = [];
      if (!this.check(")")) {
        do { args.push(this.expression()); } while (this.match(","));
      }
      const close = this.expect(")", "')'");
      expr = { kind: "Call", callee: expr, args, span: spanMerge(expr.span, close.span) };
    }
    return expr;
  }

  private primary(): Expr {
    const t = this.peek();
    switch (t.kind) {
      case "num": {
        this.advance();
        // literal 存 JS number(对拍/打印用);int/float 区分由 numIsFloat 旁路携带。
        const lit: LiteralExpr & { numIsFloat: boolean } = {
          kind: "Literal", value: Number(t.text), span: t.span, numIsFloat: t.numIsFloat ?? false,
        };
        return lit;
      }
      case "str":
        this.advance();
        return { kind: "Literal", value: t.text.slice(1, -1), span: t.span };
      case "true": this.advance(); return { kind: "Literal", value: true, span: t.span };
      case "false": this.advance(); return { kind: "Literal", value: false, span: t.span };
      case "id": this.advance(); return { kind: "Var", name: t.text, span: t.span };
      case "fn": return this.fnExpr();
      case "(": {
        this.advance();
        const e = this.expression();
        this.expect(")", "')'");
        return e;
      }
      default:
        throw new ParseError(`expected expression, found '${t.text || t.kind}'`, t.span);
    }
  }
}

/** 把源码解析为 AST。解析错误进 diags 并返回 null(无法类型检查残缺树)。 */
function parse(src: SourceText, diags: DiagnosticBag): Program | null {
  const toks = lex(src, diags);
  try {
    return new Parser(toks).parseProgram();
  } catch (e) {
    if (e instanceof ParseError) {
      diags.error(e.message, e.span, "parser");
      return null;
    }
    throw e; // 非 ParseError 是 parser 自身 bug,不该吞
  }
}

// ============================================================================
// 第 3 部分:类型检查器(本章主角)
// ============================================================================

/** 词法作用域链:每层一个 name→Type 的 map。Let 写入当前层,Var 沿链向外找。 */
class Scope {
  private readonly vars = new Map<string, Type>();
  constructor(private readonly parent: Scope | null = null) {}
  define(name: string, t: Type): void { this.vars.set(name, t); }
  lookup(name: string): Type | undefined {
    return this.vars.get(name) ?? this.parent?.lookup(name);
  }
}

/**
 * 类型检查器。不变量:
 *   - 对每个表达式产出一个 Type;凡是报了错的位置返回 T_ERROR,让 typeEquals 的哨兵
 *     逻辑止住级联误报。
 *   - 检查函数体时,把当前函数的「已见 return 类型」累积起来,退出时校验是否一致(本章
 *     要求:函数的多条 return 必须同型,否则后端无法给函数一个确定的返回形状)。
 *
 * looseEquality 开关用于失败模式 demo:打开后 `1 == "1"` 被放行,展示宽松等于如何让
 * 后端的「两侧同型」假设落空。生产应保持关闭。
 */
class TypeChecker {
  // 注:不持有 SourceText —— 类型检查只读 AST(节点自带 span),诊断渲染在 report() 时
  // 由调用方传入 source。让 checker 不依赖源码文本,职责更窄、更易复用。
  constructor(
    private readonly diags: DiagnosticBag,
    private readonly looseEquality: boolean = false,
  ) {}

  /** 当前正在检查的函数累积的 return 类型(用于多 return 一致性检查)。栈:支持嵌套函数。 */
  private readonly returnStack: Type[][] = [];

  /** 检查整个程序,返回顶层声明名 → 推导类型(供打印/对拍)。 */
  check(prog: Program): Map<string, Type> {
    const global = new Scope();
    for (const stmt of prog) this.checkStmt(stmt, global);
    // 收集顶层 Let 声明的推导类型(按声明顺序),给「打印每个顶层声明的类型」用。
    const decls = new Map<string, Type>();
    for (const stmt of prog) {
      if (stmt.kind === "Let") {
        const t = global.lookup(stmt.name);
        if (t) decls.set(stmt.name, t);
      }
    }
    return decls;
  }

  private checkStmt(stmt: Stmt, scope: Scope): void {
    switch (stmt.kind) {
      case "Let": {
        // 局部类型推断:变量类型 = 初始化表达式的类型,无需标注。
        const t = this.checkExpr(stmt.init, scope);
        scope.define(stmt.name, t);
        return;
      }
      case "Print":
        this.checkExpr(stmt.value, scope);
        return;
      case "ExprStmt":
        this.checkExpr(stmt.expr, scope);
        return;
      case "If": {
        this.expectBoolCond(stmt.cond, scope, "if");
        const thenScope = new Scope(scope);
        for (const s of stmt.then) this.checkStmt(s, thenScope);
        if (stmt.else) {
          const elseScope = new Scope(scope);
          for (const s of stmt.else) this.checkStmt(s, elseScope);
        }
        return;
      }
      case "While": {
        this.expectBoolCond(stmt.cond, scope, "while");
        const bodyScope = new Scope(scope);
        for (const s of stmt.body) this.checkStmt(s, bodyScope);
        return;
      }
      case "Block": {
        const blockScope = new Scope(scope);
        for (const s of stmt.body) this.checkStmt(s, blockScope);
        return;
      }
      case "Return": {
        const t = stmt.value ? this.checkExpr(stmt.value, scope) : T_NIL;
        const frame = this.returnStack[this.returnStack.length - 1];
        if (frame) frame.push(t);
        // 顶层 return(frame 为空)在更严格的语言里是错误;Lox-mini 容忍,不报。
        return;
      }
      default: {
        const _never: never = stmt;
        throw new Error(`checkStmt: unhandled ${JSON.stringify(_never)}`);
      }
    }
  }

  /** if/while 的条件必须是 bool。非 bool(且非 error)→ 报错,这是本章核心规则之一。 */
  private expectBoolCond(cond: Expr, scope: Scope, where: string): void {
    const t = this.checkExpr(cond, scope);
    if (t.kind !== "bool" && t.kind !== "error") {
      this.diags.error(
        `${where} condition must be bool, got ${typeToString(t)}`,
        cond.span,
        "typecheck",
      );
    }
  }

  private checkExpr(expr: Expr, scope: Scope): Type {
    switch (expr.kind) {
      case "Literal": return this.checkLiteral(expr);
      case "Var": {
        const t = scope.lookup(expr.name);
        if (t === undefined) {
          this.diags.error(`undefined variable '${expr.name}'`, expr.span, "typecheck");
          return T_ERROR;
        }
        return t;
      }
      case "Assign": {
        const declared = scope.lookup(expr.name);
        const valueType = this.checkExpr(expr.value, scope);
        if (declared === undefined) {
          this.diags.error(`assignment to undefined variable '${expr.name}'`, expr.span, "typecheck");
          return T_ERROR;
        }
        // 赋值不能改变变量类型(Lox-mini 变量是单态的)。
        if (!typeEquals(declared, valueType)) {
          this.diags.error(
            `cannot assign ${typeToString(valueType)} to variable '${expr.name}' of type ${typeToString(declared)}`,
            expr.span, "typecheck",
          );
        }
        return declared;
      }
      case "Unary": return this.checkUnary(expr, scope);
      case "Binary": return this.checkBinary(expr, scope);
      case "Call": return this.checkCall(expr, scope);
      case "Fn": return this.checkFn(expr, scope);
      default: {
        const _never: never = expr;
        throw new Error(`checkExpr: unhandled ${JSON.stringify(_never)}`);
      }
    }
  }

  private checkLiteral(expr: LiteralExpr): Type {
    const v = expr.value;
    if (typeof v === "boolean") return T_BOOL;
    if (typeof v === "string") return T_STRING;
    // number:用 lexer 旁路记的 numIsFloat 区分 int / float(见文件头设计要点)。
    const isFloat = (expr as LiteralExpr & { numIsFloat?: boolean }).numIsFloat === true;
    return isFloat ? T_FLOAT : T_INT;
  }

  private checkUnary(expr: Extract<Expr, { kind: "Unary" }>, scope: Scope): Type {
    const t = this.checkExpr(expr.operand, scope);
    if (t.kind === "error") return T_ERROR;
    if (expr.op === "-") {
      if (!isNumeric(t)) {
        this.diags.error(`unary '-' expects numeric, got ${typeToString(t)}`, expr.span, "typecheck");
        return T_ERROR;
      }
      return t; // -int → int, -float → float
    }
    // '!'
    if (t.kind !== "bool") {
      this.diags.error(`unary '!' expects bool, got ${typeToString(t)}`, expr.span, "typecheck");
      return T_ERROR;
    }
    return T_BOOL;
  }

  private checkBinary(expr: Extract<Expr, { kind: "Binary" }>, scope: Scope): Type {
    const l = this.checkExpr(expr.left, scope);
    const r = this.checkExpr(expr.right, scope);
    if (l.kind === "error" || r.kind === "error") return T_ERROR; // 止血:不在已知错上叠错
    const op = expr.op;

    // 算术 + 比较 + 等于,各有不同的类型规则。
    if (op === "+" || op === "-" || op === "*" || op === "/") {
      // '+' 同时是字符串拼接:两侧都是 string 则结果 string。
      if (op === "+" && l.kind === "string" && r.kind === "string") return T_STRING;
      if (isNumeric(l) && isNumeric(r)) return numericResult(l, r);
      this.diags.error(
        `cannot apply '${op}' to ${typeToString(l)} and ${typeToString(r)}`,
        expr.span, "typecheck",
      );
      return T_ERROR;
    }

    if (op === "<" || op === "<=" || op === ">" || op === ">=") {
      if (isNumeric(l) && isNumeric(r)) return T_BOOL;
      this.diags.error(
        `comparison '${op}' expects numeric operands, got ${typeToString(l)} and ${typeToString(r)}`,
        expr.span, "typecheck",
      );
      return T_BOOL; // 比较结果本就是 bool,即便操作数错也按 bool 续传,减少级联
    }

    if (op === "==" || op === "!=") {
      if (!typeEquals(l, r)) {
        if (this.looseEquality) {
          // 失败模式 demo:放行不同类型相等。这正是「把规则放松」的危险所在 —— 见 main。
          return T_BOOL;
        }
        this.diags.error(
          `'${op}' compares different types: ${typeToString(l)} vs ${typeToString(r)}`,
          expr.span, "typecheck",
        );
      }
      return T_BOOL;
    }

    this.diags.error(`unknown binary operator '${op}'`, expr.span, "typecheck");
    return T_ERROR;
  }

  private checkCall(expr: Extract<Expr, { kind: "Call" }>, scope: Scope): Type {
    const callee = this.checkExpr(expr.callee, scope);
    const argTypes = expr.args.map((a) => this.checkExpr(a, scope));
    if (callee.kind === "error") return T_ERROR;
    if (callee.kind !== "fn") {
      this.diags.error(`cannot call non-function of type ${typeToString(callee)}`, expr.span, "typecheck");
      return T_ERROR;
    }
    if (callee.params.length !== argTypes.length) {
      this.diags.error(
        `expected ${callee.params.length} argument(s), got ${argTypes.length}`,
        expr.span, "typecheck",
      );
      return callee.ret; // 个数错时仍按声明返回类型续传
    }
    for (let i = 0; i < argTypes.length; i++) {
      if (!typeEquals(callee.params[i], argTypes[i])) {
        this.diags.error(
          `argument ${i + 1}: expected ${typeToString(callee.params[i])}, got ${typeToString(argTypes[i])}`,
          expr.args[i].span, "typecheck",
        );
      }
    }
    return callee.ret;
  }

  /**
   * 函数类型推断。难点:递归函数(fib 调用自身)在推断体之前需要先把自身名绑定到作用域,
   * 否则体内调用自己时会报「undefined variable」。
   *
   * 简化(教学诚实标注):本章没有参数类型标注语法,无法对参数做真正的类型推断 ——
   * 参数一律按 int 处理(Lox-mini 主样例 fib 的参数确实是 int)。这是本类型系统的已知
   * 局限:真正的方案需要 Hindley-Milner 全局推断或显式参数标注,放到后续章节。这里把它
   * 显式写出来,而不是假装支持。返回类型则从 return 语句真实推断。
   */
  private checkFn(expr: Extract<Expr, { kind: "Fn" }>, scope: Scope): Type {
    const paramTypes: Type[] = expr.params.map(() => T_INT); // 局限见 doc:参数默认 int
    const fnScope = new Scope(scope);
    expr.params.forEach((p, i) => fnScope.define(p, paramTypes[i]));

    // 递归支持:先用「返回类型暂定 int」的占位函数类型绑定自身名,再检查体。
    if (expr.name) {
      fnScope.define(expr.name, { kind: "fn", params: paramTypes, ret: T_INT });
    }

    this.returnStack.push([]);
    for (const s of expr.body) this.checkStmt(s, fnScope);
    const returns = this.returnStack.pop()!;

    const ret = this.resolveReturnType(returns, expr.span);
    return { kind: "fn", params: paramTypes, ret };
  }

  /**
   * 从一个函数收集到的所有 return 类型推出唯一返回类型。
   * 规则(本章核心之一):多条 return 必须同型;不一致 → 报错并返回 error。
   * 空(无 return)→ nil。error 类型被 typeEquals 哨兵吸收,不参与「不一致」判定。
   */
  private resolveReturnType(returns: Type[], fnSpan: Span): Type {
    if (returns.length === 0) return T_NIL;
    const first = returns.find((t) => t.kind !== "error") ?? returns[0];
    for (const t of returns) {
      if (!typeEquals(first, t)) {
        this.diags.error(
          `function has inconsistent return types: ${typeToString(first)} vs ${typeToString(t)}`,
          fnSpan, "typecheck",
        );
        return T_ERROR;
      }
    }
    return first;
  }
}

// ============================================================================
// 第 4 部分:可运行的 demo / 实测(main)
// ============================================================================

/** 把源码字符串包成命名 SourceText,供内联坏样例使用。 */
function inlineSource(name: string, code: string): SourceText {
  return new SourceText(code, name);
}

/** 类型检查一份源码:返回 {decls, diags}。decls 为 null 表示解析失败。 */
function typeCheckSource(
  src: SourceText,
  looseEquality = false,
): { decls: Map<string, Type> | null; diags: DiagnosticBag } {
  const diags = new DiagnosticBag();
  const prog = parse(src, diags);
  if (prog === null) return { decls: null, diags };
  const decls = new TypeChecker(diags, looseEquality).check(prog);
  return { decls, diags };
}

function section(title: string): void {
  console.log("\n" + "=".repeat(68));
  console.log(title);
  console.log("=".repeat(68));
}

function main(): void {
  // ---- ① 好样例:跑类型检查,打印顶层声明推导类型,对拍 ----
  section("① 好样例:顶层声明的推导类型(局部推断,无需标注)");
  const goodSamples = [SAMPLES.fib, SAMPLES.counter, SAMPLES.loopsum] as const;
  // 期望表:这些类型是「无需标注、由初始化表达式推出来」的,下面 assertEq 对拍。
  const expected: Record<string, Record<string, string>> = {
    fib: { fib: "fn(int) -> int" },
    counter: { makeCounter: "fn() -> fn() -> int", c: "fn() -> int" },
    loopsum: { sum: "int", i: "int" },
  };
  for (const name of goodSamples) {
    const src = loadSample(name);
    const { decls, diags } = typeCheckSource(src);
    console.log(`\n${name}.lox:`);
    if (diags.hasErrors()) {
      console.log("  意外的类型错误:\n" + diags.report(src));
    }
    const printed: Record<string, string> = {};
    for (const [varName, t] of decls ?? []) {
      const ts = typeToString(t);
      printed[varName] = ts;
      console.log(`  ${varName} : ${ts}`);
    }
    // 对拍:把实际推导出的类型与期望逐项 assertEq(确定性,可复现)。
    for (const [k, v] of Object.entries(expected[name])) {
      assertEq(printed[k], v, `${name}.${k} 类型推断`);
    }
  }
  console.log("\n  [对拍] 所有好样例的顶层类型推断 == 期望  ✓");

  // ---- ② 局部类型推断:int / float 区分 ----
  section("② 局部类型推断:int / float 区分 + 隐式提升");
  const inferCases: Array<[string, string]> = [
    ["let x = 1 + 2;", "int"],            // 整数字面量 → int
    ["let y = 1.0 * 2;", "float"],        // 含浮点字面量 → 提升为 float
    ["let z = 3.5 + 1.5;", "float"],      // 两侧 float → float
    ["let q = 10 / 2 + 100;", "int"],     // 全 int → int(注:此语言 / 不强制转 float)
    ["let mixed = 2 * 3.0 - 1;", "float"], // int*float → float,再 -int 仍 float
  ];
  for (const [code, want] of inferCases) {
    const src = inlineSource("infer", code);
    const { decls } = typeCheckSource(src);
    const name = code.slice(4, code.indexOf(" =")); // 取变量名
    const got = decls ? typeToString([...decls.values()][0]) : "<parse-fail>";
    const mark = got === want ? "✓" : "✗ MISMATCH";
    console.log(`  ${code.padEnd(26)} → ${got.padEnd(6)} (期望 ${want}) ${mark}`);
    assertEq(got, want, `推断 ${name}`);
  }
  console.log("\n  说明:int 字面量无小数点推 int;任一侧 float 则整式提升为 float。");

  // ---- ③ 坏样例集:逐一精确诊断 + 命中率统计 ----
  section("③ 坏样例集:精确诊断 + 抓错命中率");
  // 每条坏样例标注「应抓几个类型错」。命中率 = 实抓 / 应抓。
  // 注:这里只数 typecheck 阶段的 error,不含 lexer/parser(那些是语法错,非类型错)。
  interface BadCase { readonly label: string; readonly code: string; readonly expectedErrors: number; }
  const badCases: BadCase[] = [
    { label: "字符串 + 数字", code: 'let r = "a" + 1;', expectedErrors: 1 },
    { label: "调用非函数", code: "let n = 3; let r = n(5);", expectedErrors: 1 },
    { label: "参数个数不符", code: "fn f(a, b) { return a + b; } let r = f(1);", expectedErrors: 1 },
    { label: "参数类型不符", code: 'fn f(a) { return a + 1; } let r = f("x");', expectedErrors: 1 },
    { label: "if 条件非 bool", code: "let n = 5; if (n) { print n; }", expectedErrors: 1 },
    {
      label: "两条 return 类型不一致",
      code: 'fn f(a) { if (a < 1) { return 1; } return "big"; }',
      expectedErrors: 1,
    },
  ];

  let totalExpected = 0;
  let totalCaught = 0;
  for (const bc of badCases) {
    const src = inlineSource("bad", bc.code);
    const { diags } = typeCheckSource(src);
    const typeErrors = diags.all().filter(
      (d) => d.severity === Severity.Error && d.stage === "typecheck",
    );
    totalExpected += bc.expectedErrors;
    totalCaught += Math.min(typeErrors.length, bc.expectedErrors); // 命中以应抓数封顶,多报另算
    console.log(`\n[${bc.label}]  应抓 ${bc.expectedErrors} / 实抓 ${typeErrors.length}`);
    console.log("  源码: " + bc.code);
    for (const d of typeErrors) {
      const { line, col } = src.offsetToLineCol(d.span.start);
      console.log(`  → ${line}:${col} ${d.message}`);
    }
  }
  const hitRate = ((totalCaught / totalExpected) * 100).toFixed(1);
  console.log(`\n  命中率:抓到 ${totalCaught} / 应抓 ${totalExpected} = ${hitRate}%`);

  // 完整渲染一条诊断(带源码下划线),证明诊断不是裸文本而是可定位的。
  section("③' 一条诊断的完整渲染(带源码下划线)");
  {
    const src = loadSample(SAMPLES.badType); // bad-type.lox:真实多错样例
    const { diags } = typeCheckSource(src);
    console.log(`bad-type.lox 共报 ${diags.errorCount()} 个错误:\n`);
    console.log(diags.report(src));
  }

  // ---- ④ 失败模式:把 `==` 放松成跨类型可比,演示后端假设落空 ----
  section("④ 失败模式:放行 1 == \"1\"(宽松等于)→ 后端假设落空");
  const looseCode = 'let same = (1 == "1");';
  const strict = typeCheckSource(inlineSource("eq", looseCode), /*looseEquality*/ false);
  const loose = typeCheckSource(inlineSource("eq", looseCode), /*looseEquality*/ true);
  console.log("源码: " + looseCode);
  console.log(`\n严格模式(正确):报 ${strict.diags.errorCount()} 个错 —`);
  console.log(strict.diags.report(inlineSource("eq", looseCode)).split("\n").map((l) => "  " + l).join("\n"));
  console.log(`\n宽松模式(故意放松):报 ${loose.diags.errorCount()} 个错 — 程序「通过」类型检查。`);
  console.log("  推导 same : " + typeToString([...(loose.decls ?? new Map()).values()][0] ?? T_ERROR));
  console.log(
    "\n  为什么这是 bug:后端为 '==' 生成的对比指令通常假设「两侧已是同型」——\n" +
    "  比如先把两个操作数都当 int 压栈做整数比较。一旦放行 1 == \"1\",运行期会拿一个\n" +
    "  字符串的内部指针去和整数 1 做数值比较,结果要么恒 false、要么读到垃圾内存。\n" +
    "  类型检查放松一寸,正确性假设就从这一层一路漏到运行期 —— 这正是静态类型要挡的。",
  );

  // ---- 收尾 ----
  section("小结");
  console.log("  - 好样例顶层类型全部推断正确并对拍通过(确定性)。");
  console.log(`  - 坏样例命中率 ${hitRate}%:六类典型类型错全部精确定位到行列。`);
  console.log("  - 演示了 int/float 隐式提升与宽松等于的危害。");
  console.log("\n  注:类型规则与样例为教学 toy 语言,绝对结论偏简化;");
  console.log("      可迁移的是「编译期挡运行时错」「error 哨兵止级联」「放松规则=漏洞下移」这些机制。");
}

main();
