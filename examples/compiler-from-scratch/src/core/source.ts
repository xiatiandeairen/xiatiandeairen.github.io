// core/source.ts — SourceText: 把一份源码字符串包成「可定位、可下划线」的诊断底座。
//
// WHY 存在:
//   编译器每个阶段(词法/语法/语义/类型)都要把一个错误指回源码的某个位置。
//   如果每个阶段各自扫一遍换行算行号,既重复又容易算错(off-by-one、CRLF、
//   末行无换行)。这里集中算一次 lineStarts 前缀表,O(log n) 反查行列,
//   所有阶段共享同一份真相。
//
// 不变量(INVARIANTS):
//   - Span 是半开区间 [start, end),按字节偏移(这里即 UTF-16 code unit 偏移,
//     因为 JS string 按 code unit 索引)。end >= start。空 span(start===end)
//     合法,用于在某个点而非某段下划线(例如「此处缺少分号」)。
//   - lineStarts[i] = 第 i 行(0-based)首字符的偏移。lineStarts[0] === 0 恒成立。
//   - offset 允许等于 text.length(EOF token 落在末尾),offsetToLineCol 对其有定义。
//
// 失败模式(FAILURE MODES):
//   - 越界 offset(<0 或 >length): clamp 到合法区间而不是抛异常 —— 诊断渲染
//     绝不能因为一个坏 span 把整个编译器打挂,宁可下划线画歪也要把别的错误报出来。
//   - 跨行 span: snippet 只渲染 span 起点所在那一行,跨行部分下划线在行尾截断。
//     编译器的错误几乎都落在单行内,跨行下划线收益低、实现复杂,这里刻意不做。

/** 源码中的一段:半开区间 [start, end),按字符偏移。所有 AST 节点 / Token 都携带它。 */
export interface Span {
  readonly start: number;
  readonly end: number;
}

/** 一个 [start, end) span 的便捷构造。end 默认等于 start(零宽,用于「此处」类诊断)。 */
export function span(start: number, end: number = start): Span {
  return { start, end };
}

/** 把两个 span 合并成覆盖二者的最小 span(例如 Binary 节点 = 覆盖左右操作数)。 */
export function spanMerge(a: Span, b: Span): Span {
  return { start: Math.min(a.start, b.start), end: Math.max(a.end, b.end) };
}

/** 行列号,均 1-based —— 给人看的,编辑器/编译器报错都从 1 数起。 */
export interface LineCol {
  readonly line: number;
  readonly col: number;
}

export class SourceText {
  readonly text: string;
  /** name 仅用于诊断头("foo.lox:3:5"),不参与定位逻辑。 */
  readonly name: string;
  /** 每行首字符偏移的前缀表;长度 = 行数。见文件头不变量。 */
  private readonly lineStarts: number[];

  constructor(text: string, name = "<source>") {
    this.text = text;
    this.name = name;
    this.lineStarts = SourceText.computeLineStarts(text);
  }

  /**
   * 扫一遍源码记下每个 '\n' 之后的偏移。
   * 这是 O(n) 一次性成本,换来后续每次定位 O(log n) 二分。
   * CRLF 注记: 我们只认 '\n' 作为行边界,'\r' 当普通字符留在行内 —— Lox-mini
   * 不在意 \r 的列位轻微偏差,简单胜过精确。
   */
  private static computeLineStarts(text: string): number[] {
    const starts = [0];
    for (let i = 0; i < text.length; i++) {
      if (text.charCodeAt(i) === 10 /* '\n' */) starts.push(i + 1);
    }
    return starts;
  }

  /**
   * 偏移 → 行列(1-based)。二分找到 offset 落在哪一行。
   * 越界 offset 被 clamp(见失败模式),保证永远返回合法 LineCol。
   */
  offsetToLineCol(offset: number): LineCol {
    const off = Math.max(0, Math.min(offset, this.text.length));
    // 二分: 找最大的 lineStarts[i] <= off。
    let lo = 0;
    let hi = this.lineStarts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.lineStarts[mid] <= off) lo = mid;
      else hi = mid - 1;
    }
    return { line: lo + 1, col: off - this.lineStarts[lo] + 1 };
  }

  /** 返回某一行(0-based 行号在内部用,这里取 1-based)的完整文本,不含换行。 */
  private lineText(line1: number): string {
    const i = line1 - 1;
    const start = this.lineStarts[i];
    const end = i + 1 < this.lineStarts.length ? this.lineStarts[i + 1] - 1 : this.text.length;
    // 去掉行尾可能存在的 \r(配合只认 \n 的策略)。
    let e = end;
    if (e > start && this.text.charCodeAt(e - 1) === 13 /* '\r' */) e--;
    return this.text.slice(start, e);
  }

  /**
   * 把一个 span 渲染成两行可打印片段:
   *     源码那一行
   *     <空格>^^^^   <- 下划线对齐到 span
   * 用于诊断输出。跨行 span 只渲染起点行(见失败模式)。
   * 返回不带前导/尾随的 "代码行\n下划线行" 字符串,由调用方决定缩进。
   */
  snippet(s: Span): string {
    const { line, col } = this.offsetToLineCol(s.start);
    const code = this.lineText(line);
    // 下划线宽度: span 在本行内的可见长度,至少 1(零宽 span 也画一个 ^)。
    const lineEndOffset = this.lineStarts[line] !== undefined
      ? (line < this.lineStarts.length ? this.lineStarts[line] + code.length : this.text.length)
      : this.text.length;
    const visibleEnd = Math.min(s.end, lineEndOffset);
    const underlineLen = Math.max(1, visibleEnd - s.start);
    const pad = " ".repeat(col - 1);
    const caret = "^".repeat(underlineLen);
    return `${code}\n${pad}${caret}`;
  }
}
