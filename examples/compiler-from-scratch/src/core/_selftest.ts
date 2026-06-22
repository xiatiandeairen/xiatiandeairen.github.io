// core/_selftest.ts — core 底座的自检脚本(脚手架交付物,非书章节)。
//
// WHY: 脚手架阶段还没有任何 stage,但 core 必须先证明自己能跑、数字真实、
// 失败模式真触发。本文件构造手写 AST + 诊断 + 计时,断言它们的行为,并打印
// 给人看的真实输出。后续每个 stage 各自证明自己,这个文件只验证 core 契约。

import {
  SourceText,
  span,
  spanMerge,
  TokenKind,
  KEYWORDS,
  makeToken,
  astToSExpr,
  type Expr,
  type Stmt,
  DiagnosticBag,
  Severity,
  timeIt,
  speedup,
  assertEq,
  fmtNum,
  loadSample,
  SAMPLES,
} from "./index.js";

console.log("=== core 自检: source.ts ===");
{
  const src = new SourceText("let x = 1;\nlet y = x + 2;\nprint y;", "demo.lox");
  // 不变量: lineStarts[0]===0, 第二行从换行后开始。
  const lc = src.offsetToLineCol(11); // 'let y' 的 'l',第2行第1列
  assertEq(lc, { line: 2, col: 1 }, "offsetToLineCol 第二行行首");
  // 越界 clamp(失败模式): 不抛,返回末尾位置。
  const oob = src.offsetToLineCol(9999);
  assertEq(oob.line, 3, "越界 offset clamp 到末行");
  console.log(`  offsetToLineCol(11) = 第 ${lc.line} 行第 ${lc.col} 列  (OK)`);
  console.log("  snippet 下划线演示:");
  console.log(
    src
      .snippet(span(15, 16)) // 第二行的 'y'(offset 11 行首 + "let " 4 字符)
      .split("\n")
      .map((l) => "    " + l)
      .join("\n"),
  );
  assertEq(spanMerge(span(2, 5), span(8, 10)), { start: 2, end: 10 }, "spanMerge 取并");
}

console.log("\n=== core 自检: token.ts ===");
{
  assertEq(KEYWORDS.get("while"), TokenKind.While, "关键字表命中 while");
  assertEq(KEYWORDS.get("If"), undefined, "大小写敏感: If 不是关键字(失败模式)");
  const t = makeToken(TokenKind.Number, "42", span(0, 2), 42);
  assertEq(t.literal, 42, "字面量 token 携带解析后的值");
  const id = makeToken(TokenKind.Identifier, "foo", span(0, 3));
  assertEq(id.literal, undefined, "非字面量 token 无 literal");
  console.log(`  KEYWORDS.size = ${KEYWORDS.size} 个关键字  (OK)`);
}

console.log("\n=== core 自检: ast.ts (astToSExpr 确定性) ===");
{
  // 手写 AST: (let n (+ 1 (* 2 3)))
  const e: Expr = {
    kind: "Binary",
    op: "+",
    left: { kind: "Literal", value: 1, span: span(0, 1) },
    right: {
      kind: "Binary",
      op: "*",
      left: { kind: "Literal", value: 2, span: span(0, 1) },
      right: { kind: "Literal", value: 3, span: span(0, 1) },
      span: span(0, 1),
    },
    span: span(0, 1),
  };
  const stmt: Stmt = { kind: "Let", name: "n", init: e, span: span(0, 1) };
  const dump = astToSExpr(stmt);
  assertEq(dump, "(let n (+ 1 (* 2 3)))", "S 表达式确定性");
  // resolve 前后差异: depth 体现在 dump。
  const v1 = astToSExpr({ kind: "Var", name: "count", span: span(0, 5) });
  const v2 = astToSExpr({ kind: "Var", name: "count", span: span(0, 5), depth: 1 });
  assertEq(v1, "count", "未 resolve var 无 @depth");
  assertEq(v2, "count@1", "已 resolve var 带 @depth");
  console.log(`  dump = ${dump}  (确定性 OK)`);
  console.log(`  resolve 前 = ${v1} ; resolve 后 = ${v2}`);
}

console.log("\n=== core 自检: diagnostics.ts (多错收集 + 渲染) ===");
{
  const src = new SourceText("let x = (1 + 2;\nprint badvar;", "diag.lox");
  const bag = new DiagnosticBag();
  bag.error("expected ')'", span(14, 15), "parser"); // ';' 处
  bag.error("undefined variable 'badvar'", span(22, 28), "resolver");
  bag.warning("unused variable 'x'", span(4, 5), "resolver");
  // 不变量: hasErrors 只看 Error,warning 不阻断。
  assertEq(bag.hasErrors(), true, "有 error");
  assertEq(bag.errorCount(), 2, "2 个 error");
  assertEq(bag.count(), 3, "共 3 条(含 warning)");
  console.log(bag.report(src).split("\n").map((l) => "  " + l).join("\n"));
}

console.log("\n=== core 自检: bench.ts (真实测速 + 加速比 + fmtNum) ===");
{
  // 构造一个真有性能差的对比: 朴素递归 fib vs 记忆化 fib。
  // 这是「相对加速比」的演示 —— 绝对毫秒只展示、不断言(诚实数字纪律)。
  function fibNaive(n: number): number {
    return n < 2 ? n : fibNaive(n - 1) + fibNaive(n - 2);
  }
  function fibMemo(n: number, m: number[] = []): number {
    if (n < 2) return n;
    if (m[n] !== undefined) return m[n];
    return (m[n] = fibMemo(n - 1, m) + fibMemo(n - 2, m));
  }
  const N = 28;
  // 先验证两者结果一致(对拍): 速度无意义除非结果相同。
  assertEq(fibMemo(N), fibNaive(N), "memo 与 naive 结果一致");

  const slow = timeIt(() => void fibNaive(N), { runs: 5, warmup: 1 });
  const fast = timeIt(() => void fibMemo(N, []), { runs: 5, warmup: 1, inner: 50 });
  const ratio = speedup(slow, fast);

  // 断言用相对量(可复现): memo 必定显著快于 naive。绝对毫秒只打印。
  if (!(ratio > 5)) {
    throw new Error(`期望 memo 至少快 5 倍,实测 ${ratio.toFixed(1)}x —— 测量可能被噪声污染`);
  }
  console.log(`  fib(${N}) naive 中位数 = ${slow.medianMs.toFixed(3)} ms (本机实测,仅展示)`);
  console.log(`  fib(${N}) memo  中位数 = ${fast.medianMs.toFixed(4)} ms (本机实测,仅展示)`);
  console.log(`  加速比 = ${ratio.toFixed(1)}x  (相对量,可复现断言: >5x)`);
  console.log(`  fmtNum(1234567) = ${fmtNum(1234567)} ; fmtNum(-1234.5) = ${fmtNum(-1234.5)} ; fmtNum(Infinity) = ${fmtNum(Infinity)}`);

  // assertEq 失败模式演示: 故意比一对不等的值,捕获它打印的 diff。
  let caught = "";
  try {
    assertEq([1, 2, 3], [1, 2, 4], "演示对拍失败");
  } catch (err) {
    caught = (err as Error).message;
  }
  if (!caught.includes("mismatch")) throw new Error("assertEq 未在不等时抛错");
  console.log("  assertEq 失败模式(演示,已捕获):");
  console.log(caught.split("\n").map((l) => "    " + l).join("\n"));
}

console.log("\n=== core 自检: programs.ts (样例加载,cwd 无关) ===");
{
  const fib = loadSample(SAMPLES.fib);
  const counter = loadSample(SAMPLES.counter);
  // 真读到文件且非空。
  if (fib.text.length === 0) throw new Error("fib.lox 为空");
  const fibLines = fib.text.split("\n").length;
  console.log(`  loadSample(fib)     -> ${fib.name}, ${fib.text.length} chars, ${fibLines} 行`);
  console.log(`  loadSample(counter) -> ${counter.name}, ${counter.text.length} chars`);
  console.log(`  样例清单: ${Object.values(SAMPLES).join(", ")}`);
}

console.log("\n=== core 自检全部通过 ===");
console.log("Severity 枚举值:", Object.values(Severity).join(", "));
