// stage02-parser.ts — 第2章: 语法分析。把 token 流织成 AST。
//
// 本章演示的两件核心机制(都是真跑出来的,不是讲义伪代码):
//   1. 递归下降(recursive descent)解析「语句」: 每种语句一个 parseXxx 函数,
//      函数调用栈天然镜像语法树的嵌套 —— 这是手写 parser 最直观的结构,Clang/V8
//      的前端骨架都是它。
//   2. Pratt parsing(算符优先 / precedence climbing)解析「表达式」: 用一张
//      「绑定力(binding power)」表驱动,而非给每个优先级写一个递归函数。
//      为什么不用经典的 expr→term→factor 链? 那种写法每加一档优先级就加一层函数,
//      且左结合靠循环、右结合靠递归,容易写错。Pratt 把「优先级」与「结合性」编码成
//      两个数字(left bp / right bp),一个循环搞定所有二元运算,加运算符只改表。
//
// 为什么不 import stage01 的 lexer: stage 文件加载即跑 main(),互相 import 会触发
// 副作用。本文件内联一个最小 lexer(只够喂 parser),lexer 本身是第1章的主题,这里
// 不重复教学,只作为「上游产物」存在。token 种类全部复用 core/token 的 TokenKind,
// 保证与全书一致。
//
// 不变量(INVARIANTS):
//   - parser 永不抛异常做控制流: 语法错误进 DiagnosticBag(收集多个),靠 panic-mode
//     的 synchronize() 跳到下一条语句边界继续解析。这样一次能报多个错,而非首错即停。
//   - 每个 AST 节点带 span,指回源码(诊断 + 后续阶段定位)。
//   - 解析是确定性的: 同一 token 流恒产出同一 AST,astToSExpr 可直接对拍。
//
// 失败模式(本文件 §6 主动演示,不只跑 happy path):
//   - 缺分号 / 括号不匹配: 开 synchronize 报多个错; 关掉只报首错 —— 对比给读者看
//     「错误恢复」的价值。
//   - 一元/二元绑定力设同值: 故意制造 `1 - -2` 解析错位,打印错误的 S 表达式,
//     说明 binding power 不是可有可无的调参,设错就语义错。

import {
  TokenKind,
  makeToken,
  KEYWORDS,
  type Token,
  type Expr,
  type Stmt,
  type Program,
  type LiteralValue,
  span,
  spanMerge,
  type Span,
  SourceText,
  DiagnosticBag,
  astToSExpr,
  timeIt,
  assertEq,
  fmtNum,
} from "./core/index.js";

// ============================================================================
// §0 最小 lexer —— 仅为喂 parser 而存在(第1章主题,此处不展开教学)。
// ============================================================================

// 单字符 → TokenKind 的直查表。多字符运算符(== != <= >=)在扫描时再 peek 下一字符。
const SINGLE_CHAR: ReadonlyMap<string, TokenKind> = new Map([
  ["+", TokenKind.Plus],
  ["-", TokenKind.Minus],
  ["*", TokenKind.Star],
  ["/", TokenKind.Slash],
  ["(", TokenKind.LParen],
  [")", TokenKind.RParen],
  ["{", TokenKind.LBrace],
  ["}", TokenKind.RBrace],
  [",", TokenKind.Comma],
  [";", TokenKind.Semicolon],
]);

function isDigit(c: string): boolean {
  return c >= "0" && c <= "9";
}
// 标识符首字符: 字母或下划线(不含数字,避免 `1abc` 被当标识符)。
function isAlpha(c: string): boolean {
  return (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c === "_";
}
function isAlphaNum(c: string): boolean {
  return isAlpha(c) || isDigit(c);
}

/**
 * 把源码扫成 token 数组。失败模式: 未知字符进 diagnostics 但跳过继续扫,
 * 不抛 —— 与 parser 同一纪律(尽量多收集错误)。末尾恒补一个 Eof token。
 */
function lex(src: SourceText, diags: DiagnosticBag): Token[] {
  const text = src.text;
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    // 空白
    if (c === " " || c === "\t" || c === "\r" || c === "\n") {
      i++;
      continue;
    }
    // 行注释 `// ...` 到行尾。样例里大量使用。
    if (c === "/" && text[i + 1] === "/") {
      while (i < text.length && text[i] !== "\n") i++;
      continue;
    }
    const start = i;
    // 多字符运算符: 先试两字符,失败再退回单字符。
    if (c === "=" || c === "!" || c === "<" || c === ">") {
      if (text[i + 1] === "=") {
        const two = c + "=";
        const kind =
          two === "==" ? TokenKind.EqEq
          : two === "!=" ? TokenKind.BangEq
          : two === "<=" ? TokenKind.LessEq
          : TokenKind.GreaterEq;
        tokens.push(makeToken(kind, two, span(start, i + 2)));
        i += 2;
        continue;
      }
      const oneKind =
        c === "=" ? TokenKind.Eq
        : c === "!" ? TokenKind.Bang
        : c === "<" ? TokenKind.Less
        : TokenKind.Greater;
      tokens.push(makeToken(oneKind, c, span(start, i + 1)));
      i++;
      continue;
    }
    // 数字字面量(整数 + 可选小数部分)。literal 解析成 JS number。
    if (isDigit(c)) {
      while (i < text.length && isDigit(text[i])) i++;
      if (text[i] === "." && isDigit(text[i + 1])) {
        i++;
        while (i < text.length && isDigit(text[i])) i++;
      }
      const lexeme = text.slice(start, i);
      tokens.push(makeToken(TokenKind.Number, lexeme, span(start, i), Number(lexeme)));
      continue;
    }
    // 字符串字面量(双引号,不处理转义 —— 样例不需要)。
    if (c === '"') {
      i++; // 跳过开引号
      while (i < text.length && text[i] !== '"') i++;
      if (i >= text.length) {
        diags.error("unterminated string literal", span(start, i), "lexer");
        continue;
      }
      const lexeme = text.slice(start, i + 1);
      const value = text.slice(start + 1, i); // 去掉引号
      i++; // 跳过闭引号
      tokens.push(makeToken(TokenKind.String, lexeme, span(start, i), value));
      continue;
    }
    // 标识符 / 关键字
    if (isAlpha(c)) {
      while (i < text.length && isAlphaNum(text[i])) i++;
      const lexeme = text.slice(start, i);
      const kw = KEYWORDS.get(lexeme);
      if (kw !== undefined) {
        // 布尔关键字带 literal 值,便于求值阶段直接取。
        const lit: LiteralValue | undefined =
          kw === TokenKind.True ? true : kw === TokenKind.False ? false : undefined;
        tokens.push(makeToken(kw, lexeme, span(start, i), lit));
      } else {
        tokens.push(makeToken(TokenKind.Identifier, lexeme, span(start, i)));
      }
      continue;
    }
    // 单字符运算符 / 分隔符
    const single = SINGLE_CHAR.get(c);
    if (single !== undefined) {
      tokens.push(makeToken(single, c, span(start, i + 1)));
      i++;
      continue;
    }
    // 未知字符: 报错并跳过(不抛)。
    diags.error(`unexpected character '${c}'`, span(start, i + 1), "lexer");
    i++;
  }
  tokens.push(makeToken(TokenKind.Eof, "<eof>", span(text.length)));
  return tokens;
}

// ============================================================================
// §1 Pratt 绑定力表 —— 表达式优先级与结合性的「唯一真相」。
// ============================================================================

// 二元运算符的绑定力。每个运算符给一对 [leftBp, rightBp]:
//   - 数字越大越「抓得紧」(优先级越高)。`*` 的 bp 高于 `+`,所以 1+2*3 = 1+(2*3)。
//   - 左结合: leftBp < rightBp(解析 right 时门槛比自己高一点,挡住同级的下一个运算符,
//     使其归到左边)。例 `1-2-3`: 解析完 1-2 后,右边的 `-` 因 minBp 抬高而不被吞,
//     于是 (1-2)-3 而非 1-(2-3)。
//   - 右结合: leftBp > rightBp(解析 right 时门槛比自己低,允许同级运算符继续向右嵌套)。
//     赋值 `=` 用这个: a=b=c → a=(b=c)。
//
// WHY 用「一对数字」而非「一个优先级 + 一个 enum 结合性」: 一对 bp 同时编码了优先级
// 和结合性,Pratt 主循环只需比较数字,无需分支判断结合性 —— 更少代码、更难写错。
interface BindingPower {
  readonly left: number;
  readonly right: number;
}

// 注意: 下面所有 bp 都是偶数间隔,留出奇数给「结合性微调」。
// 左结合用 [n, n+1],右结合用 [n+1, n]。
const BINARY_BP: ReadonlyMap<TokenKind, BindingPower> = new Map([
  // 赋值最低优先级 + 右结合(a=b=c)。leftBp > rightBp。
  [TokenKind.Eq, { left: 2, right: 1 }],
  // 比较运算符(左结合)。
  [TokenKind.EqEq, { left: 4, right: 5 }],
  [TokenKind.BangEq, { left: 4, right: 5 }],
  [TokenKind.Less, { left: 6, right: 7 }],
  [TokenKind.LessEq, { left: 6, right: 7 }],
  [TokenKind.Greater, { left: 6, right: 7 }],
  [TokenKind.GreaterEq, { left: 6, right: 7 }],
  // 加减(左结合)。
  [TokenKind.Plus, { left: 8, right: 9 }],
  [TokenKind.Minus, { left: 8, right: 9 }],
  // 乘除(左结合,高于加减)。
  [TokenKind.Star, { left: 10, right: 11 }],
  [TokenKind.Slash, { left: 10, right: 11 }],
]);

// 一元前缀运算符的「右绑定力」: 必须高于所有二元,这样 -a+b = (-a)+b 而非 -(a+b)。
// 也必须高于乘除,使 -a*b = (-a)*b。同时低于「调用/后缀」(见 POSTFIX_BP)。
const PREFIX_BP = 12;
// 函数调用 `f(...)` 是最高优先级的后缀,高于一元: -f(x) = -(f(x)),且 f(x)(y) 左结合链式。
const POSTFIX_BP = 14;

// 自定义异常: 仅用于「无法继续解析当前结构」时跳出递归回到 synchronize 点。
// 这是 panic-mode 恢复的实现手段 —— 不是把异常当业务控制流(诊断已进 bag),
// 而是用栈展开快速回到语句边界。catch 在 parseStmt 顶层,不外泄。
class ParseError extends Error {}

// ============================================================================
// §2 Parser —— 递归下降(语句) + Pratt(表达式)。
// ============================================================================

interface ParseOptions {
  /** 关掉 panic-mode 恢复后,首个语法错就停止解析(对比「错误恢复」价值时用)。 */
  readonly recover: boolean;
}

class Parser {
  private pos = 0;
  constructor(
    private readonly tokens: readonly Token[],
    private readonly diags: DiagnosticBag,
    private readonly opts: ParseOptions,
  ) {}

  // ---- token 游标基本操作 ----
  private peek(): Token {
    return this.tokens[this.pos];
  }
  private atEnd(): boolean {
    return this.peek().kind === TokenKind.Eof;
  }
  private advance(): Token {
    const t = this.tokens[this.pos];
    if (!this.atEnd()) this.pos++;
    return t;
  }
  private check(kind: TokenKind): boolean {
    return this.peek().kind === kind;
  }
  private match(kind: TokenKind): boolean {
    if (this.check(kind)) {
      this.advance();
      return true;
    }
    return false;
  }
  /**
   * 期望某 token,否则报错并抛 ParseError(触发 panic-mode)。
   * 失败模式: 报告「expected X」并把当前位置带上,不静默吞。
   */
  private expect(kind: TokenKind, what: string): Token {
    if (this.check(kind)) return this.advance();
    const t = this.peek();
    this.diags.error(`expected ${what}`, t.span, "parser");
    throw new ParseError(what);
  }

  // ---- 顶层: 解析整个程序 ----
  parseProgram(): Program {
    const stmts: Stmt[] = [];
    while (!this.atEnd()) {
      const before = this.pos;
      try {
        stmts.push(this.parseStmt());
      } catch (e) {
        if (!(e instanceof ParseError)) throw e; // 真异常(bug)不吞。
        if (!this.opts.recover) break; // 关恢复: 首错即停,只报一个错。
        this.synchronize(); // 开恢复: 跳到下一语句边界继续。
      }
      // 防御性死循环保护: 若某次循环既没前进也没抛错,强制前进一格。
      // 「永远不会发生」但 parser 一旦写出不前进的路径就会无限循环挂死,代价太高。
      if (this.pos === before && !this.atEnd()) this.advance();
    }
    return stmts;
  }

  /**
   * panic-mode 恢复: 丢弃 token 直到一个「语句边界」—— 分号之后,或下一条语句的
   * 起始关键字之前。这样一处语法错不会污染整份文件的解析,后续语句仍能独立报错。
   */
  private synchronize(): void {
    while (!this.atEnd()) {
      // 刚消费的若是分号,说明上一句已结束,边界到了。
      if (this.tokens[this.pos - 1]?.kind === TokenKind.Semicolon) return;
      // 下一个若是语句起始关键字,也是边界。
      switch (this.peek().kind) {
        case TokenKind.Let:
        case TokenKind.Fn:
        case TokenKind.If:
        case TokenKind.While:
        case TokenKind.Print:
        case TokenKind.Return:
          return;
      }
      this.advance();
    }
  }

  // ---- 语句 ----
  private parseStmt(): Stmt {
    switch (this.peek().kind) {
      case TokenKind.Let:
        return this.parseLet();
      case TokenKind.Fn:
        return this.parseFnDecl();
      case TokenKind.If:
        return this.parseIf();
      case TokenKind.While:
        return this.parseWhile();
      case TokenKind.Print:
        return this.parsePrint();
      case TokenKind.Return:
        return this.parseReturn();
      case TokenKind.LBrace:
        return this.parseBlock();
      default:
        return this.parseExprStmt();
    }
  }

  private parseLet(): Stmt {
    const kw = this.advance(); // 'let'
    const name = this.expect(TokenKind.Identifier, "variable name");
    this.expect(TokenKind.Eq, "'=' (let must initialize)");
    const init = this.parseExpr(0);
    const semi = this.expect(TokenKind.Semicolon, "';' after let");
    return { kind: "Let", name: name.lexeme, init, span: spanMerge(kw.span, semi.span) };
  }

  // `fn name(p1, p2) { body }` —— 函数声明语句, 内部用 FnExpr 节点承载, 包成 ExprStmt。
  // WHY 包成 ExprStmt 而非新增 FnDecl 节点: AST 里函数只有一种结构(FnExpr,name 可空),
  // 声明与匿名共用,resolver/求值阶段不必分两套。声明 = 「带名字的 FnExpr 当语句」。
  private parseFnDecl(): Stmt {
    const fn = this.parseFnExpr();
    return { kind: "ExprStmt", expr: fn, span: fn.span };
  }

  private parseFnExpr(): Expr {
    const kw = this.advance(); // 'fn'
    // 匿名函数无名字(name 留 undefined);声明则有标识符。
    let name: string | undefined;
    if (this.check(TokenKind.Identifier)) name = this.advance().lexeme;
    this.expect(TokenKind.LParen, "'(' after fn");
    const params: string[] = [];
    if (!this.check(TokenKind.RParen)) {
      do {
        params.push(this.expect(TokenKind.Identifier, "parameter name").lexeme);
      } while (this.match(TokenKind.Comma));
    }
    this.expect(TokenKind.RParen, "')' after parameters");
    const body = this.parseBraceBody();
    const last = this.tokens[this.pos - 1];
    return { kind: "Fn", name, params, body, span: spanMerge(kw.span, last.span) };
  }

  private parseIf(): Stmt {
    const kw = this.advance(); // 'if'
    this.expect(TokenKind.LParen, "'(' after if");
    const cond = this.parseExpr(0);
    this.expect(TokenKind.RParen, "')' after if condition");
    const then = this.parseBraceBody();
    let els: Stmt[] | undefined;
    if (this.match(TokenKind.Else)) {
      // else 后允许 `{ ... }` 或紧跟 `if`(else-if 链)。
      els = this.check(TokenKind.If) ? [this.parseIf()] : this.parseBraceBody();
    }
    const last = this.tokens[this.pos - 1];
    return { kind: "If", cond, then, else: els, span: spanMerge(kw.span, last.span) };
  }

  private parseWhile(): Stmt {
    const kw = this.advance(); // 'while'
    this.expect(TokenKind.LParen, "'(' after while");
    const cond = this.parseExpr(0);
    this.expect(TokenKind.RParen, "')' after while condition");
    const body = this.parseBraceBody();
    const last = this.tokens[this.pos - 1];
    return { kind: "While", cond, body, span: spanMerge(kw.span, last.span) };
  }

  private parsePrint(): Stmt {
    const kw = this.advance(); // 'print'
    const value = this.parseExpr(0);
    const semi = this.expect(TokenKind.Semicolon, "';' after print");
    return { kind: "Print", value, span: spanMerge(kw.span, semi.span) };
  }

  private parseReturn(): Stmt {
    const kw = this.advance(); // 'return'
    // 裸 return 合法(返回值 undefined),所以分号前若无表达式就不解析值。
    let value: Expr | undefined;
    if (!this.check(TokenKind.Semicolon)) value = this.parseExpr(0);
    const semi = this.expect(TokenKind.Semicolon, "';' after return");
    return { kind: "Return", value, span: spanMerge(kw.span, semi.span) };
  }

  private parseBlock(): Stmt {
    const lb = this.peek();
    const body = this.parseBraceBody();
    const last = this.tokens[this.pos - 1];
    return { kind: "Block", body, span: spanMerge(lb.span, last.span) };
  }

  // 解析 `{ stmt* }`。复用于 fn/if/while/block —— 共用一处保证花括号配对逻辑一致。
  private parseBraceBody(): Stmt[] {
    this.expect(TokenKind.LBrace, "'{'");
    const body: Stmt[] = [];
    while (!this.check(TokenKind.RBrace) && !this.atEnd()) {
      body.push(this.parseStmt());
    }
    this.expect(TokenKind.RBrace, "'}'");
    return body;
  }

  private parseExprStmt(): Stmt {
    const expr = this.parseExpr(0);
    const semi = this.expect(TokenKind.Semicolon, "';' after expression");
    return { kind: "ExprStmt", expr, span: spanMerge(expr.span, semi.span) };
  }

  // ---- 表达式: Pratt 主循环 ----
  /**
   * 解析一个表达式,只吞「绑定力 > minBp」的运算符。minBp 是从父级传下来的门槛:
   * 调用方说「我只关心比 X 抓得更紧的运算符,更松的请留给我」。这一个参数 + 一个
   * 循环就实现了全部优先级与结合性。
   *
   * 不变量: 进入时 pos 指向一个表达式的起点; 返回时 pos 指向第一个绑定力不足以被
   * 本层吞下的 token(留给父层)。
   */
  private parseExpr(minBp: number): Expr {
    let left = this.parsePrefix();

    // 后缀(调用)优先级最高,先于二元处理: f(x)(y) 链式、且 -f(x)=-(f(x))。
    left = this.parsePostfix(left, minBp);

    // 二元主循环。
    for (;;) {
      const opTok = this.peek();
      const bp = BINARY_BP.get(opTok.kind);
      if (bp === undefined || bp.left <= minBp) break; // 绑定力不够,交还父层。

      // 赋值是特例: 左侧必须是可赋值目标(Var),且产出 Assign 而非 Binary。
      if (opTok.kind === TokenKind.Eq) {
        this.advance();
        const right = this.parseExpr(bp.right); // 右结合: 用 rightBp 当门槛。
        left = this.makeAssign(left, right, opTok.span);
        left = this.parsePostfix(left, minBp); // 赋值结果仍可被调用(罕见但一致)。
        continue;
      }

      this.advance();
      const right = this.parseExpr(bp.right); // 右递归,门槛=rightBp 决定结合性。
      left = {
        kind: "Binary",
        op: opTok.lexeme,
        left,
        right,
        span: spanMerge(left.span, right.span),
      };
      left = this.parsePostfix(left, minBp);
    }
    return left;
  }

  // 把 `target = value` 构造成 Assign;target 必须是 Var,否则是语义错(如 `1 = 2`)。
  private makeAssign(target: Expr, value: Expr, eqSpan: Span): Expr {
    if (target.kind !== "Var") {
      this.diags.error("invalid assignment target", eqSpan, "parser");
      throw new ParseError("invalid assignment target");
    }
    return { kind: "Assign", name: target.name, value, span: spanMerge(target.span, value.span) };
  }

  /**
   * 前缀位置: 字面量 / 变量 / 括号分组 / 一元运算符 / 匿名 fn。
   * 一元 `-` `!` 用 PREFIX_BP 递归,保证它抓得比所有二元紧。
   */
  private parsePrefix(): Expr {
    const t = this.peek();
    switch (t.kind) {
      case TokenKind.Number:
      case TokenKind.String:
        this.advance();
        return { kind: "Literal", value: t.literal as LiteralValue, span: t.span };
      case TokenKind.True:
        this.advance();
        return { kind: "Literal", value: true, span: t.span };
      case TokenKind.False:
        this.advance();
        return { kind: "Literal", value: false, span: t.span };
      case TokenKind.Identifier:
        this.advance();
        return { kind: "Var", name: t.lexeme, span: t.span };
      case TokenKind.LParen: {
        this.advance();
        const inner = this.parseExpr(0); // 括号内重置门槛为 0。
        this.expect(TokenKind.RParen, "')'");
        return inner; // 分组不产生节点,只改变结合 —— AST 里看不到括号(优先级已体现在结构)。
      }
      case TokenKind.Minus:
      case TokenKind.Bang: {
        this.advance();
        const operand = this.parseExpr(PREFIX_BP); // 关键: 用 PREFIX_BP 而非 0。见 §6 失败模式。
        return { kind: "Unary", op: t.lexeme, operand, span: spanMerge(t.span, operand.span) };
      }
      case TokenKind.Fn:
        return this.parseFnExpr(); // 匿名函数表达式。
      default:
        this.diags.error(`expected expression, found '${t.lexeme}'`, t.span, "parser");
        throw new ParseError("expected expression");
    }
  }

  /**
   * 后缀循环: 只处理函数调用 `expr(args)`。POSTFIX_BP 高于一切,故只要门槛允许就贪婪
   * 吞掉连续的调用,实现 f(x)(y) 左结合链式。
   */
  private parsePostfix(expr: Expr, minBp: number): Expr {
    let e = expr;
    while (this.check(TokenKind.LParen) && POSTFIX_BP > minBp) {
      this.advance(); // '('
      const args: Expr[] = [];
      if (!this.check(TokenKind.RParen)) {
        do {
          args.push(this.parseExpr(0)); // 实参之间逗号分隔,各自从门槛 0 起。
        } while (this.match(TokenKind.Comma));
      }
      const rp = this.expect(TokenKind.RParen, "')' after arguments");
      e = { kind: "Call", callee: e, args, span: spanMerge(e.span, rp.span) };
    }
    return e;
  }
}

// 便捷入口: 源码字符串 → { ast, diags }。两步(lex→parse)共享同一个 DiagnosticBag。
function parse(src: SourceText, opts: ParseOptions = { recover: true }): {
  ast: Program;
  diags: DiagnosticBag;
} {
  const diags = new DiagnosticBag();
  const tokens = lex(src, diags);
  const ast = new Parser(tokens, diags, opts).parseProgram();
  return { ast, diags };
}

// 仅解析单个表达式(给优先级/结合性逐条断言用)。包成 `<expr>;` 再取出。
function parseExprOnly(srcText: string): Expr {
  const { ast, diags } = parse(new SourceText(srcText + ";", "<expr>"));
  if (diags.hasErrors()) {
    throw new Error(`parseExprOnly unexpected errors:\n${diags.report(new SourceText(srcText))}`);
  }
  const first = ast[0];
  if (first.kind !== "ExprStmt") throw new Error(`expected ExprStmt, got ${first.kind}`);
  return first.expr;
}

// ============================================================================
// §3 节点计数(吞吐基准用,确定性) —— 数一棵 AST 有多少节点。
// ============================================================================

function countNodes(node: Expr | Stmt | Program): number {
  if (Array.isArray(node)) {
    return (node as readonly Stmt[]).reduce((acc, s) => acc + countNodes(s), 0);
  }
  const n = node as Expr | Stmt;
  let total = 1;
  switch (n.kind) {
    case "Binary":
      total += countNodes(n.left) + countNodes(n.right);
      break;
    case "Unary":
      total += countNodes(n.operand);
      break;
    case "Assign":
      total += countNodes(n.value);
      break;
    case "Call":
      total += countNodes(n.callee) + n.args.reduce((a, x) => a + countNodes(x), 0);
      break;
    case "Fn":
      total += countNodes(n.body as readonly Stmt[]);
      break;
    case "Let":
    case "Print":
      total += countNodes(n.kind === "Let" ? n.init : n.value);
      break;
    case "If":
      total += countNodes(n.cond) + countNodes(n.then as readonly Stmt[]);
      if (n.else) total += countNodes(n.else as readonly Stmt[]);
      break;
    case "While":
      total += countNodes(n.cond) + countNodes(n.body as readonly Stmt[]);
      break;
    case "Block":
      total += countNodes(n.body as readonly Stmt[]);
      break;
    case "Return":
      if (n.value) total += countNodes(n.value);
      break;
    case "ExprStmt":
      total += countNodes(n.expr);
      break;
    case "Literal":
    case "Var":
      break; // 叶子。
  }
  return total;
}

// ============================================================================
// §4 失败模式演示用的「坏 Pratt」 —— 一元/二元绑定力设同值。
// ============================================================================

// 故意把一元 `-` 的绑定力降到与二元 `-` 同档,复现「绑定力设错 → 语义错位」。
// 这不是另写一个 parser,而是把 §1 那个 PREFIX_BP 换成与减法相同的 leftBp(8),
// 看 `1 - -2` 会被解析成什么。我们用一个独立的、刻意写坏的迷你解析器演示,避免污染
// 正确实现 —— 只覆盖演示需要的 number / unary-minus / binary-minus 三件事。
function parseWithBrokenUnaryBp(srcText: string): Expr {
  const diags = new DiagnosticBag();
  const tokens = lex(new SourceText(srcText, "<broken>"), diags);
  let pos = 0;
  const peek = () => tokens[pos];
  const advance = () => tokens[pos++];

  // BUG(教学故意): brokenPrefixBp 设成 8,与二元减法的 leftBp 相同,而非应有的 12。
  // 后果: 解析一元 operand 时门槛=8,等于减法 leftBp,于是一元 `-` 会把后面的减法
  // 「吞」进自己的 operand,造成 `1 - -2` 结合错位。
  const BROKEN_PREFIX_BP = 8;

  function prefix(): Expr {
    const t = peek();
    if (t.kind === TokenKind.Number) {
      advance();
      return { kind: "Literal", value: t.literal as LiteralValue, span: t.span };
    }
    if (t.kind === TokenKind.Minus) {
      advance();
      const operand = expr(BROKEN_PREFIX_BP); // ← 错误门槛。
      return { kind: "Unary", op: "-", operand, span: spanMerge(t.span, operand.span) };
    }
    throw new Error("broken-parser: unexpected token");
  }
  function expr(minBp: number): Expr {
    let left = prefix();
    for (;;) {
      const op = peek();
      const bp = BINARY_BP.get(op.kind);
      if (bp === undefined || bp.left <= minBp) break;
      advance();
      const right = expr(bp.right);
      left = { kind: "Binary", op: op.lexeme, left, right, span: spanMerge(left.span, right.span) };
    }
    return left;
  }
  return expr(0);
}

// ============================================================================
// §5 主流程 —— 全部数字真跑真测真对拍。
// ============================================================================

function main(): void {
  console.log("=== 第2章 语法分析: 递归下降 + Pratt ===\n");

  // ---- ① 对样例解析并对拍固定 S 表达式 ----
  console.log("① 样例解析 → S 表达式对拍");
  const samples: Array<{ name: string; src: string; expected: string }> = [
    {
      name: "loopsum.lox(while + 赋值 + 算术)",
      src: [
        "let sum = 0;",
        "let i = 1;",
        "while (i <= 100) {",
        "  sum = sum + i;",
        "  i = i + 1;",
        "}",
        "print sum;",
      ].join("\n"),
      expected:
        "((let sum 0) (let i 1) (while (<= i 100) ((set! sum (+ sum i)) (set! i (+ i 1)))) (print sum))",
    },
    {
      name: "fib(fn + if + return + 调用 + 递归)",
      src: [
        "fn fib(n) {",
        "  if (n < 2) {",
        "    return n;",
        "  }",
        "  return fib(n - 1) + fib(n - 2);",
        "}",
        "print fib(10);",
      ].join("\n"),
      expected:
        "((fn fib (n) ((if (< n 2) ((return n))) (return (+ (call fib (- n 1)) (call fib (- n 2)))))) (print (call fib 10)))",
    },
    {
      name: "counter(闭包: 嵌套 fn + 捕获赋值 + 返回函数)",
      src: [
        "fn makeCounter() {",
        "  let count = 0;",
        "  fn next() {",
        "    count = count + 1;",
        "    return count;",
        "  }",
        "  return next;",
        "}",
        "let c = makeCounter();",
        "print c();",
      ].join("\n"),
      expected:
        "((fn makeCounter () ((let count 0) (fn next () ((set! count (+ count 1)) (return count))) (return next))) (let c (call makeCounter)) (print (call c)))",
    },
  ];

  for (const s of samples) {
    const { ast, diags } = parse(new SourceText(s.src, s.name));
    if (diags.hasErrors()) {
      console.log(`  ✗ ${s.name} 意外报错:\n${diags.report(new SourceText(s.src, s.name))}`);
      continue;
    }
    const got = astToSExpr(ast);
    assertEq(got, s.expected, `S 表达式对拍 [${s.name}]`);
    console.log(`  ✓ ${s.name}`);
    console.log(`      ${got}`);
  }
  console.log();

  // ---- ② 优先级 / 结合性逐条断言 ----
  console.log("② 优先级 / 结合性");
  const exprCases: Array<{ src: string; expected: string; note: string }> = [
    { src: "1 + 2 * 3", expected: "(+ 1 (* 2 3))", note: "乘高于加" },
    { src: "1 * 2 + 3", expected: "(+ (* 1 2) 3)", note: "乘先于加(左侧)" },
    { src: "1 - 2 - 3", expected: "(- (- 1 2) 3)", note: "减法左结合" },
    { src: "a = b = c", expected: "(set! a (set! b c))", note: "赋值右结合" },
    { src: "1 < 2 == true", expected: "(== (< 1 2) #t)", note: "比较高于相等" },
    { src: "-a + b", expected: "(+ (- a) b)", note: "一元负号紧于二元加" },
    { src: "-a * b", expected: "(* (- a) b)", note: "一元负号紧于乘法" },
    { src: "!a == b", expected: "(== (! a) b)", note: "一元 ! 紧于相等" },
    { src: "f(x)(y)", expected: "(call (call f x) y)", note: "调用左结合链式" },
    { src: "-f(x)", expected: "(- (call f x))", note: "调用紧于一元(先调用再取负)" },
    { src: "1 - -2", expected: "(- 1 (- 2))", note: "正确: 二元减 - 一元负" },
  ];
  for (const c of exprCases) {
    const got = astToSExpr(parseExprOnly(c.src));
    assertEq(got, c.expected, `结合性 [${c.src}]`);
    console.log(`  ✓ ${c.src.padEnd(16)} → ${got.padEnd(24)} (${c.note})`);
  }
  console.log();

  // ---- ③ 解析吞吐 (nodes/s) ----
  console.log("③ 解析吞吐 (nodes/s, 本机实测)");
  // 用 loopsum 反复拼接成一个大程序,放大到可稳定测量的规模。
  const unit = [
    "let sum = 0;",
    "let i = 1;",
    "while (i <= 100) { sum = sum + i * 2 - 1; i = i + 1; }",
    "print sum;",
  ].join("\n");
  const bigSrc = new SourceText(Array(400).fill(unit).join("\n"), "throughput.lox");
  const { ast: bigAst, diags: bigDiags } = parse(bigSrc);
  if (bigDiags.hasErrors()) throw new Error("throughput source unexpectedly failed to parse");
  const nodeCount = countNodes(bigAst);
  // 只计 parse 段(不含 lex),口径清晰。lex 一次产 token,parse 反复跑。
  const tokens = lex(bigSrc, new DiagnosticBag());
  const t = timeIt(() => {
    new Parser(tokens, new DiagnosticBag(), { recover: true }).parseProgram();
  }, { runs: 9, inner: 3 });
  const nodesPerSec = nodeCount / (t.medianMs / 1000);
  console.log(`  源码规模      = ${fmtNum(bigSrc.text.length)} chars, ${fmtNum(tokens.length)} tokens`);
  console.log(`  AST 节点数    = ${fmtNum(nodeCount)} (确定性计数)`);
  console.log(`  parse 中位数  = ${t.medianMs.toFixed(3)} ms`);
  console.log(`  吞吐          = ${fmtNum(Math.round(nodesPerSec))} nodes/s`);
  console.log(
    "  注: toy 语言 + 内存 token 数组, 绝对值偏乐观; 可迁移的是「递归下降+Pratt 单遍线性」这一相对特征。",
  );
  console.log();

  // ---- ④ 错误恢复: 开/关 synchronize 对比 ----
  console.log("④ 错误恢复 (panic-mode synchronize)");
  // 三处独立语法错: 缺分号、括号不匹配、再缺分号。专业编译器应一次报全。
  const badSrcText = [
    "let a = 1",            // 缺分号
    "let b = (2 + 3;",      // 括号不匹配
    "print a + b",          // 缺分号(EOF)
  ].join("\n");
  const badSrc = new SourceText(badSrcText, "broken.lox");

  const recovered = parse(badSrc, { recover: true });
  console.log(`  [开 synchronize] 收集到 ${recovered.diags.errorCount()} 个语法错:`);
  console.log(indent(recovered.diags.report(badSrc), 4));
  console.log();

  const firstOnly = parse(badSrc, { recover: false });
  console.log(`  [关 synchronize] 只报首错 ${firstOnly.diags.errorCount()} 个 (首错即停):`);
  console.log(indent(firstOnly.diags.report(badSrc), 4));
  console.log(
    `\n  价值: 恢复后一次暴露 ${recovered.diags.errorCount()} 个错, 用户改一轮看全部; 不恢复要改 ${recovered.diags.errorCount()} 轮。`,
  );
  console.log();

  // ---- ⑤ 失败模式: 一元/二元绑定力设同值导致 `-2 * 3` 解析错位 ----
  console.log("⑤ 失败模式: 一元/二元绑定力设同值 (binding power 设错 → 语义错)");
  // WHY 用 `-2 * 3` 而非 `1 - -2`: 前者才暴露此 bug。`-2*3` 语义应是 `(-2)*3`=-6,
  // 即一元负只作用于 2; 但若一元 operand 的递归门槛(8)低于乘法 leftBp(10), 一元会把
  // 整个 `2*3` 吞进 operand, 错算成 `-(2*3)`=-6 —— 这个例子数值碰巧相同, 但结构(也就
  // 是给后续阶段的语义树)完全不同, 换成 `-2 * 3 < x` 之类就连结果都会错。
  const probe = "-2 * 3";
  const correct = astToSExpr(parseExprOnly(probe));
  const broken = astToSExpr(parseWithBrokenUnaryBp(probe));
  console.log(`  输入            : ${probe}`);
  console.log(`  正确(PREFIX_BP=${PREFIX_BP}) : ${correct}   ← 一元负只作用于 2, 顶层是乘法`);
  console.log(`  错误(PREFIX_BP=8)         : ${broken}   ← 一元负把整个 (* 2 3) 吞进 operand`);
  // 断言「确实不同」: 把失败模式钉成可复现的回归, 而非口头声称。
  if (correct === broken) throw new Error("失败模式演示失效: 两者应不同");
  console.log(`  ✓ 两种 binding power 产出不同 AST, 证明它不是可有可无的调参。`);
  console.log(
    "  根因: 一元 operand 的递归门槛若 <= 后续二元 leftBp, 一元会贪婪吞掉本应留给二元的运算符。",
  );
  console.log();

  console.log("=== 第2章全部断言通过 ===");
}

// 给多行文本统一加缩进(诊断块嵌进列表时对齐用)。
function indent(text: string, n: number): string {
  const pad = " ".repeat(n);
  return text
    .split("\n")
    .map((l) => (l.length > 0 ? pad + l : l))
    .join("\n");
}

main();
