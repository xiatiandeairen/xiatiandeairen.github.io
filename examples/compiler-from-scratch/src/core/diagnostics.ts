// core/diagnostics.ts — 统一诊断收集器:所有阶段往这里报错,最后渲染成带下划线的文本。
//
// WHY 存在:
//   新手编译器最常见的两个毛病:(1) 到处 `throw new Error("bad token")`,错误信息
//   散乱、没位置、首错即停;(2) 用户改一个错重编一次只看见下一个错。专业编译器都
//   「收集多个错误一次报全」+「每条错误带源码下划线」。这个收集器把这两件事变成
//   全书统一基础设施,各阶段只管 .error(msg, span),渲染交给它。
//
// 不变量(INVARIANTS):
//   - 报错不抛异常(默认): error()/warning() 只是 push 进列表,编译继续往下跑,
//     尽量多收集错误。需要「此处无法继续」时阶段自行决定 break/return,但不靠抛异常
//     做控制流。(panic-mode 恢复是各 parser 的事,不在这层。)
//   - hasErrors() 只看 severity===Error;warning 不阻断后续阶段。
//   - 渲染顺序 = 报告顺序(报错的先后),不重排。便于「按发现顺序读」与对拍稳定。
//
// 失败模式(FAILURE MODES):
//   - span 落在源码外: 交给 SourceText.snippet/offsetToLineCol clamp,不在这里崩。
//   - 海量错误(如词法彻底乱): 调用方可设上限提前停;本收集器不强制上限(教学样例
//     规模小),但 report() 渲染 N 条就是 N 条,不截断 —— 截断会掩盖真实失败规模。

import type { Span } from "./source.js";
import type { SourceText } from "./source.js";

export enum Severity {
  Error = "error",
  Warning = "warning",
}

export interface Diagnostic {
  readonly severity: Severity;
  readonly message: string;
  readonly span: Span;
  /** 哪个阶段报的(lexer/parser/...),仅用于渲染前缀,帮读者定位「谁报的」。 */
  readonly stage?: string;
}

export class DiagnosticBag {
  private readonly items: Diagnostic[] = [];

  /** 收集一条 error。注意: 不抛异常 —— 见文件头不变量。 */
  error(message: string, span: Span, stage?: string): void {
    this.items.push({ severity: Severity.Error, message, span, stage });
  }

  warning(message: string, span: Span, stage?: string): void {
    this.items.push({ severity: Severity.Warning, message, span, stage });
  }

  /** 只有 Error 才算「编译失败」;Warning 不阻断。 */
  hasErrors(): boolean {
    return this.items.some((d) => d.severity === Severity.Error);
  }

  count(): number {
    return this.items.length;
  }

  errorCount(): number {
    return this.items.filter((d) => d.severity === Severity.Error).length;
  }

  /** 只读快照,供对拍 / 测试断言具体条目。 */
  all(): readonly Diagnostic[] {
    return this.items;
  }

  /**
   * 把所有诊断渲染成人读文本,每条形如:
   *
   *   foo.lox:3:5: error[parser]: expected ')'
   *     let x = (1 + 2;
   *                  ^
   *
   * 需要 SourceText 才能算行列、取源码行。渲染顺序 = 报告顺序(不变量)。
   * 返回纯字符串,由调用方决定 console.error 还是写文件 —— 这层不做 IO。
   */
  report(source: SourceText): string {
    if (this.items.length === 0) return "";
    const blocks: string[] = [];
    for (const d of this.items) {
      const { line, col } = source.offsetToLineCol(d.span.start);
      const tag = d.stage ? `${d.severity}[${d.stage}]` : d.severity;
      const head = `${source.name}:${line}:${col}: ${tag}: ${d.message}`;
      // snippet 返回「代码行\n下划线行」,统一缩进 2 空格成块。
      const snip = source
        .snippet(d.span)
        .split("\n")
        .map((l) => "  " + l)
        .join("\n");
      blocks.push(`${head}\n${snip}`);
    }
    return blocks.join("\n\n");
  }
}
