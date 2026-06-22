// stage01-lexer.ts — 第 01 章:词法分析(lexing / scanning)。
//
// WHY 存在:
//   把源码「字符流」切成「token 流」。后续每个阶段(parser/resolver/...)都消费
//   token 而非裸字符 —— 谁也不想在解析表达式时还纠结「这几个字符是不是关键字」。
//   词法器把这层噪声一次性吃掉:空白/注释丢弃、数字/字符串解析成值、关键字和
//   普通标识符区分开、多字符运算符(`==`/`>=`/`!=`)合成一个 token。
//
// 设计:单遍扫描 + 单字符前瞻(peek)。Lox-mini 文法不需要更长的前瞻,
//   `==` 这类只要看后一个字符就能决定。词法不抛异常 —— 遇到非法字符/未闭合
//   字符串往 DiagnosticBag 收一条错误后**继续扫**,这样一次扫描能报全部错误
//   (新手编译器最烦的就是「改一个错重编一次只看见下一个」)。
//
// 不变量(INVARIANTS):
//   - 产出的 token 序列必以一个 Eof token 收尾(零宽 span 落在 text.length)。
//     parser 循环靠它停,不必每处判越界。
//   - 每个 token 的 span 是它在源码里的精确半开区间 [start, end),诊断靠它下划线。
//   - 非法输入只进 DiagnosticBag,绝不抛 —— 见上「一次报全」。
//
// 失败模式(FAILURE MODES,本章 demo 全部真实触发):
//   - 非法字符(如 `@`):报「unexpected character」并跳过该字符继续。
//   - 未闭合字符串:扫到行尾/文件尾仍没收尾引号 → 报错,把已扫部分当一个坏 String。
//   - 多字符运算符不做前瞻:见本文件 demoNoLookahead()——把 `>=` 切成 `>` `=` 会让
//     下一章 parser 在「期望表达式却拿到孤立 `=`」处报错。

import {
  TokenKind,
  KEYWORDS,
  makeToken,
  span,
  SourceText,
  DiagnosticBag,
  loadSample,
  SAMPLES,
  timeIt,
  fmtNum,
  assertEq,
  type Token,
} from "./core/index.js";

// ---------------------------------------------------------------------------
// 字符判定:手写而非正则。热路径上每个字符都要判一次,正则的对象创建/回溯开销
// 在百万字符级别会拖慢吞吐;手写 charCode 比较是分支预测友好的整数比较。
// (这是 comment.md §3 的「性能 hack」锚点:说明为何不用更短的 /\d/。)
// ---------------------------------------------------------------------------
function isDigit(ch: string): boolean {
  return ch >= "0" && ch <= "9";
}
function isAlpha(ch: string): boolean {
  return (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z") || ch === "_";
}
function isAlphaNum(ch: string): boolean {
  return isDigit(ch) || isAlpha(ch);
}

/**
 * 单遍扫描器。一次 new + 一次 scanTokens(),无复用状态 —— 复用同一实例扫两份
 * 源码会串味(current 指针残留),所以每份源码一个 Lexer。
 */
export class Lexer {
  // NOTE: 只存 text 切片,不存整个 SourceText —— 词法器不负责算行列(那是
  // SourceText.offsetToLineCol 的职责,单一真相)。词法只产 span,定位推迟到诊断/打印层。
  private readonly src: string;
  private readonly diags: DiagnosticBag;
  /** 当前 lexeme 起点偏移,每扫一个 token 前重置为 current。 */
  private start = 0;
  /** 下一个待读字符偏移。current === src.length 表示已到末尾。 */
  private current = 0;

  constructor(source: SourceText, diags: DiagnosticBag) {
    this.src = source.text;
    this.diags = diags;
  }

  /**
   * 扫完整份源码,返回 token 列表(含末尾 Eof)。错误进 diags,不抛。
   * 副作用:向构造时传入的 DiagnosticBag 追加诊断。
   */
  scanTokens(): Token[] {
    const tokens: Token[] = [];
    while (!this.isAtEnd()) {
      this.start = this.current;
      const tok = this.scanOne();
      // scanOne 返回 undefined 表示「这段是空白/注释,不产 token」。
      if (tok !== undefined) tokens.push(tok);
    }
    // Eof 哨兵:零宽 span 落在文末。parser 靠它停循环(见不变量)。
    tokens.push(makeToken(TokenKind.Eof, "", span(this.current)));
    return tokens;
  }

  private isAtEnd(): boolean {
    return this.current >= this.src.length;
  }

  /** 读当前字符并前进。调用方需自己保证未越界(循环条件已挡)。 */
  private advance(): string {
    return this.src[this.current++];
  }

  /** 看当前字符不前进;越界返回 "\0"(哨兵,绝不出现在合法源码里)。 */
  private peek(): string {
    return this.isAtEnd() ? "\0" : this.src[this.current];
  }

  /** 看下一个字符(用于数字小数点后必须是数字的判断)。 */
  private peekNext(): string {
    return this.current + 1 >= this.src.length ? "\0" : this.src[this.current + 1];
  }

  /**
   * 条件前进:当前字符 === expected 才消费并返回 true。
   * 这是「多字符运算符前瞻」的核心 —— 见到 `=` 后 match('=') 决定是 `==` 还是 `=`。
   * demoNoLookahead() 演示去掉它会怎样把 `>=` 错切成两个 token。
   */
  private match(expected: string): boolean {
    if (this.isAtEnd() || this.src[this.current] !== expected) return false;
    this.current++;
    return true;
  }

  /** 用当前 [start, current) 区间构造一个 token。 */
  private make(kind: TokenKind, literal?: Token["literal"]): Token {
    const lexeme = this.src.slice(this.start, this.current);
    return makeToken(kind, lexeme, span(this.start, this.current), literal);
  }

  /**
   * 扫一个 token。返回 undefined = 跳过(空白/注释)。
   * 单字符 switch 先行,需要前瞻的(运算符/数字/标识符/字符串)落到对应分支。
   */
  private scanOne(): Token | undefined {
    const ch = this.advance();
    switch (ch) {
      // 单字符分隔符与运算符
      case "(": return this.make(TokenKind.LParen);
      case ")": return this.make(TokenKind.RParen);
      case "{": return this.make(TokenKind.LBrace);
      case "}": return this.make(TokenKind.RBrace);
      case ",": return this.make(TokenKind.Comma);
      case ";": return this.make(TokenKind.Semicolon);
      case "+": return this.make(TokenKind.Plus);
      case "-": return this.make(TokenKind.Minus);
      case "*": return this.make(TokenKind.Star);

      // 多字符运算符:看后一个字符决定。这就是「前瞻」,失败模式 demo 拿掉的正是这里。
      case "!": return this.make(this.match("=") ? TokenKind.BangEq : TokenKind.Bang);
      case "=": return this.make(this.match("=") ? TokenKind.EqEq : TokenKind.Eq);
      case "<": return this.make(this.match("=") ? TokenKind.LessEq : TokenKind.Less);
      case ">": return this.make(this.match("=") ? TokenKind.GreaterEq : TokenKind.Greater);

      // `/` 既是除号也是行注释开头,必须前瞻区分。
      case "/":
        if (this.match("/")) {
          // 行注释:吃到行尾(不含 '\n',留给空白分支丢弃),不产 token。
          while (this.peek() !== "\n" && !this.isAtEnd()) this.advance();
          return undefined;
        }
        return this.make(TokenKind.Slash);

      // 空白:丢弃。'\n' 不特殊处理 —— 行号由 SourceText 的 lineStarts 算,词法器不数行。
      case " ":
      case "\r":
      case "\t":
      case "\n":
        return undefined;

      case '"':
        return this.scanString();

      default:
        if (isDigit(ch)) return this.scanNumber();
        if (isAlpha(ch)) return this.scanIdentifier();
        // 失败模式 1:非法字符。报错后**不**重新抛,跳过这一个字符继续扫,
        // 让后面的合法 token 和后续错误都能被看到(一次报全)。
        this.diags.error(
          `unexpected character '${ch}'`,
          span(this.start, this.current),
          "lexer",
        );
        return undefined;
    }
  }

  /**
   * 字符串字面量。已消费开引号,扫到闭引号为止。
   * 失败模式 2:扫到文件尾仍无闭引号 → 报「unterminated string」,
   * 把已扫部分(含开引号)当一个坏 String token 产出,好让 parser 仍能往下走。
   */
  private scanString(): Token {
    while (this.peek() !== '"' && !this.isAtEnd()) {
      this.advance(); // Lox-mini 允许字符串跨行;不在此拦换行。
    }
    if (this.isAtEnd()) {
      this.diags.error(
        "unterminated string literal",
        span(this.start, this.current),
        "lexer",
      );
      // literal 用去掉开引号的剩余部分(没有闭引号);lexeme 保留原文供诊断回显。
      return this.make(TokenKind.String, this.src.slice(this.start + 1, this.current));
    }
    this.advance(); // 消费闭引号。
    // literal = 去掉两端引号的内容;lexeme(make 里取)含引号,供诊断回显。
    const value = this.src.slice(this.start + 1, this.current - 1);
    return this.make(TokenKind.String, value);
  }

  /**
   * 数字字面量。整数或带一个小数点的浮点(`1.` / `.5` 都不合法:点后必须有数字)。
   * literal 解析成 JS number,词法阶段就定值(见 token.ts 不变量:lexeme 留原文)。
   */
  private scanNumber(): Token {
    while (isDigit(this.peek())) this.advance();
    // 仅当「点后紧跟数字」才认小数:避免把 `1.method` 里的点吃进数字。
    if (this.peek() === "." && isDigit(this.peekNext())) {
      this.advance(); // 消费 '.'
      while (isDigit(this.peek())) this.advance();
    }
    const lexeme = this.src.slice(this.start, this.current);
    return this.make(TokenKind.Number, Number(lexeme));
  }

  /**
   * 标识符或关键字。先扫完整个 [a-zA-Z_][a-zA-Z0-9_]*,再查关键字表决定 kind。
   * 「最长匹配」原则:`ifx` 是标识符不是关键字 `if` + 标识符 `x`。
   */
  private scanIdentifier(): Token {
    while (isAlphaNum(this.peek())) this.advance();
    const text = this.src.slice(this.start, this.current);
    const keyword = KEYWORDS.get(text);
    if (keyword !== undefined) {
      // 关键字字面量:true/false 带 boolean literal,其余关键字无 literal。
      if (keyword === TokenKind.True) return this.make(keyword, true);
      if (keyword === TokenKind.False) return this.make(keyword, false);
      return this.make(keyword);
    }
    return this.make(TokenKind.Identifier);
  }
}

/** 便捷入口:扫一份源码,返回 token 列表 + 诊断袋。 */
export function tokenize(source: SourceText): { tokens: Token[]; diags: DiagnosticBag } {
  const diags = new DiagnosticBag();
  const tokens = new Lexer(source, diags).scanTokens();
  return { tokens, diags };
}

// ===========================================================================
// 以下为本章 demo(main):四个任务全部用真实代码跑出真实数字。
// ===========================================================================

/** 把一个 token 渲染成「行:列  kind  lexeme」一行,行列来自 SourceText(单一真相)。 */
function formatToken(source: SourceText, tok: Token): string {
  const { line, col } = source.offsetToLineCol(tok.span.start);
  const pos = `${line}:${col}`.padEnd(6);
  const kind = tok.kind.padEnd(12);
  // EOF 无 lexeme,显式标 <eof> 便于阅读。
  const lexeme = tok.kind === TokenKind.Eof ? "<eof>" : JSON.stringify(tok.lexeme);
  const lit =
    tok.literal !== undefined ? `  literal=${JSON.stringify(tok.literal)}` : "";
  return `  ${pos}${kind}${lexeme}${lit}`;
}

/** 任务①+②:扫描 fib / counter,打印 token 序列并对拍 token 总数。 */
function demoScanSamples(): void {
  console.log("=== 任务①②:扫描样例 + token 对拍 ===\n");

  // 期望 token 数:对每份样例用 scanTokens 真扫一遍数出来后,写死成断言基线。
  // 这不是「凭空规定」,而是固定当前正确行为(characterization),改坏了立刻报警。
  const expected: Record<string, number> = { fib: 40, counter: 52 };

  for (const name of [SAMPLES.fib, SAMPLES.counter] as const) {
    const source = loadSample(name);
    const { tokens, diags } = tokenize(source);

    console.log(`--- ${name}.lox (${source.text.length} chars) ---`);
    for (const tok of tokens) console.log(formatToken(source, tok));
    console.log(
      `  → ${tokens.length} tokens, ${diags.errorCount()} errors\n`,
    );

    // 对拍:token 总数必须等于基线。assertEq 不等会抛并打 diff。
    assertEq(tokens.length, expected[name], `${name} token count`);
    // 合法样例不该有词法错误。
    assertEq(diags.errorCount(), 0, `${name} should lex clean`);
  }
  console.log("  ✓ token 数对拍通过,合法样例零词法错误\n");
}

/**
 * 任务②补充:逐项验证字符串/数字/标识符/关键字/多字符运算符都识别正确。
 * 用一段刻意覆盖各类 token 的源码,断言 (kind, lexeme) 序列。
 */
function demoTokenKinds(): void {
  console.log("=== 任务②:各类 token 识别(含关键字 vs 标识符) ===\n");

  const probe = new SourceText(
    'let x = 42; let y = 3.14; let s = "hi"; if (x >= y) { x != 1; } ifx',
    "probe",
  );
  const { tokens } = tokenize(probe);

  // 关键字与标识符的关键对照:`if` 是关键字,`ifx` 是标识符(最长匹配)。
  const kinds = tokens.map((t) => t.kind);
  console.log(`  token kinds: ${kinds.join(" ")}\n`);

  // 抽查若干关键判定点,断言确切 kind。
  const byLexeme = (lex: string) => tokens.find((t) => t.lexeme === lex)!;
  assertEq(byLexeme("let").kind, TokenKind.Let, "'let' is keyword");
  assertEq(byLexeme("if").kind, TokenKind.If, "'if' is keyword");
  assertEq(byLexeme("ifx").kind, TokenKind.Identifier, "'ifx' is identifier (longest match)");
  assertEq(byLexeme(">=").kind, TokenKind.GreaterEq, "'>=' is one token");
  assertEq(byLexeme("!=").kind, TokenKind.BangEq, "'!=' is one token");
  assertEq(byLexeme("3.14").kind, TokenKind.Number, "'3.14' is number");
  assertEq(byLexeme("3.14").literal, 3.14, "'3.14' literal parsed");
  assertEq(byLexeme('"hi"').kind, TokenKind.String, "string token");
  assertEq(byLexeme('"hi"').literal, "hi", "string literal stripped quotes");

  console.log("  ✓ 关键字/标识符/数字/字符串/多字符运算符识别全部正确\n");
}

/**
 * 任务③:吞吐。把一段源码重复拼接到 ~50k token,真测 tokens/s。
 * 诚实数字纪律:绝对吞吐依赖本机,标 (本机实测);可迁移的是「线性、无回溯」这个性质。
 */
function demoThroughput(): void {
  console.log("=== 任务③:词法吞吐(本机实测) ===\n");

  // 一行约含若干 token 的代表性代码;重复拼接到 token 数 ~50k。
  const unit = 'let v = a + b * 2 >= c != d; if (v) { print "x"; }\n';
  const unitTokens = tokenize(new SourceText(unit, "unit")).tokens.length - 1; // 减去 Eof
  const repeat = Math.ceil(50_000 / unitTokens);
  const big = new SourceText(unit.repeat(repeat), "big");

  // 先数实际规模(确定性,可断言)。
  const { tokens } = tokenize(big);
  const tokenCount = tokens.length;
  console.log(
    `  源码 ${fmtNum(big.text.length)} chars → ${fmtNum(tokenCount)} tokens (${fmtNum(repeat)} 行重复)`,
  );

  // timeIt:中位数去抖。只测扫描,不测打印/断言(避免污染计时)。
  const t = timeIt(() => {
    new Lexer(big, new DiagnosticBag()).scanTokens();
  }, { runs: 7, warmup: 2 });

  const tokensPerSec = tokenCount / (t.medianMs / 1000);
  const charsPerSec = big.text.length / (t.medianMs / 1000);
  console.log(`  中位耗时 = ${t.medianMs.toFixed(3)} ms (本机实测,仅展示)`);
  console.log(`  吞吐 ≈ ${fmtNum(Math.round(tokensPerSec))} tokens/s (本机实测)`);
  console.log(`       ≈ ${fmtNum(Math.round(charsPerSec))} chars/s (本机实测)`);
  console.log(
    "  注:绝对值偏乐观(toy 语言 + V8 JIT);可迁移的结论是单遍扫描对源码长度线性、无回溯。\n",
  );
}

/**
 * 任务④:错误定位 + 一次扫描收集全部错误。
 * 坏样例同时含非法字符 `@` 和未闭合字符串,验证两条错误都被收集(而非首错即停)。
 */
function demoErrorReporting(): void {
  console.log("=== 任务④:错误定位 + 一次收集全部错误 ===\n");

  // 第 1 行有非法字符 @;第 3 行字符串未闭合(扫到文件尾)。
  const bad = new SourceText(
    ['let a = 1 @ 2;', 'let ok = "fine";', 'let s = "oops'].join("\n"),
    "bad.lox",
  );
  const { tokens, diags } = tokenize(bad);

  console.log(`  扫描产出 ${tokens.length} tokens,收集到 ${diags.errorCount()} 条错误:\n`);
  console.log(diags.report(bad));
  console.log();

  // 断言:确实收集到 2 条(非首错即停),且都是 lexer 阶段。
  assertEq(diags.errorCount(), 2, "should collect BOTH lexical errors in one pass");
  const msgs = diags.all().map((d) => d.message);
  assertEq(
    msgs.some((m) => m.includes("unexpected character")),
    true,
    "reported illegal char",
  );
  assertEq(
    msgs.some((m) => m.includes("unterminated string")),
    true,
    "reported unterminated string",
  );
  console.log("  ✓ 一次扫描收集到全部 2 条错误,各自定位到正确行列\n");
}

/**
 * 失败模式 demo:不做前瞻,把 `>=` 切成 `>` 和 `=` 两个 token。
 * 这不是「会崩」,而是「悄悄切错」——词法看似成功,错误推迟到下一章 parser 才爆,
 * 且爆点离根因很远(parser 在孤立 `=` 处报「期望表达式」,根因却在词法)。
 * 这正是「前瞻」存在的理由:让歧义在最早、最有上下文的地方被解决。
 */
function demoNoLookahead(): void {
  console.log("=== 失败模式:词法不做前瞻会怎样 ===\n");

  const code = "x >= 2";

  // 正确版:带前瞻。
  const good = tokenize(new SourceText(code, "good")).tokens
    .filter((t) => t.kind !== TokenKind.Eof)
    .map((t) => t.kind);
  console.log(`  正确(带前瞻):  "${code}"  →  ${good.join(" ")}`);

  // 错误版:不前瞻 —— 模拟「见 > 就立刻产 Greater,不看后面的 =」。
  // 这里直接展示 naive 切法的结果,不改主 Lexer(主 Lexer 是对的)。
  const naive = naiveScanNoLookahead(code);
  console.log(`  错误(无前瞻):  "${code}"  →  ${naive.join(" ")}`);
  console.log();
  console.log("  后果:`>=` 被切成 [>] [=]。词法这一步「成功」了,但 token 流已损坏。");
  console.log("  下一章 parser 解析 `x > = 2` 时:吃掉 `x`、看到 `>` 当二元运算符、");
  console.log("  期望右侧是表达式 —— 却拿到一个孤立的 `=`(赋值号),报「expected expression」。");
  console.log("  根因在词法(漏前瞻),报错却出在 parser,排查方向被带偏。\n");

  // 用断言把「两种切法 token 数不同」钉死,证明这确实是可观测的行为差异。
  // 带前瞻:Identifier GreaterEq Number = 3;无前瞻:Identifier Greater Eq Number = 4。
  assertEq(good.length, 3, "with lookahead: [Identifier, GreaterEq, Number] = 3 tokens");
  assertEq(naive.length, 4, "without lookahead: '>=' split into '>' '=' → 4 tokens");
  assertEq(good.length < naive.length, true, "lookahead merges, naive splits");
}

/**
 * 教学用的「坏」扫描器:把 `>` `<` `=` `!` 一律当单字符,绝不前瞻。
 * 仅用于 demoNoLookahead 对比,不在生产路径。只处理本 demo 需要的字符子集。
 */
function naiveScanNoLookahead(code: string): TokenKind[] {
  const out: TokenKind[] = [];
  for (let i = 0; i < code.length; i++) {
    const ch = code[i];
    if (ch === " ") continue;
    if (ch === ">") out.push(TokenKind.Greater);
    else if (ch === "<") out.push(TokenKind.Less);
    else if (ch === "=") out.push(TokenKind.Eq); // 永远当赋值,即使前面是 >
    else if (ch === "!") out.push(TokenKind.Bang);
    else if (isDigit(ch)) out.push(TokenKind.Number);
    else if (isAlpha(ch)) out.push(TokenKind.Identifier);
  }
  return out;
}

function main(): void {
  console.log("############ 第 01 章:词法分析 ############\n");
  demoScanSamples();
  demoTokenKinds();
  demoThroughput();
  demoErrorReporting();
  demoNoLookahead();
  console.log("############ 第 01 章全部 demo 通过 ############");
}

main();
