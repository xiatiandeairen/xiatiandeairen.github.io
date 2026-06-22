// core/token.ts — 词法单元定义:TokenKind 枚举 + Token 结构 + 关键字表。
//
// WHY 存在:
//   词法器(stage01)产出 Token,语法器(stage02)消费 Token。两者必须对同一套
//   TokenKind 达成一致,否则一改枚举两处对不上。把它抽到 core,语法器只 import、
//   绝不重新定义 token 种类。关键字表同理 —— "if"/"while" 是关键字还是标识符,
//   是词法器的判断,但这张表是全书唯一真相。
//
// 不变量(INVARIANTS):
//   - 每个 Token 必带 span(指回 SourceText),诊断阶段靠它定位。EOF token 的 span
//     落在 text.length 处(零宽),用于「文件意外结束」类报错。
//   - literal 仅在字面量 token(Number/String/True/False)上有值;其余为 undefined。
//     数字字面量在词法阶段就解析成 JS number(整数/浮点统一),字符串去掉引号。
//   - lexeme 是该 token 在源码中的原始切片(含引号、含原始数字写法),用于诊断回显;
//     literal 是解析后的语义值。二者刻意分开:报错要显原文,求值要用值。
//
// 失败模式(FAILURE MODES):
//   - 未知关键字: keywords 表查不到的标识符一律 Identifier,不报错(可能是变量名)。
//   - 关键字大小写: Lox-mini 关键字大小写敏感,"If" 是标识符不是关键字。表里只放小写。

import type { Span } from "./source.js";

/**
 * 所有词法单元种类。分组:字面量 / 标识符与关键字 / 运算符 / 分隔符 / 文件末尾。
 * 注: 用字符串枚举(而非数字)是为了 astToSExpr / 诊断打印时直接可读、可对拍。
 */
export enum TokenKind {
  // 字面量
  Number = "Number",
  String = "String",
  // 标识符
  Identifier = "Identifier",
  // 关键字
  Let = "let",
  Fn = "fn",
  If = "if",
  Else = "else",
  While = "while",
  Print = "print",
  Return = "return",
  True = "true",
  False = "false",
  // 运算符
  Plus = "+",
  Minus = "-",
  Star = "*",
  Slash = "/",
  Eq = "=",       // 赋值
  EqEq = "==",
  BangEq = "!=",
  Bang = "!",
  Less = "<",
  LessEq = "<=",
  Greater = ">",
  GreaterEq = ">=",
  // 分隔符
  LParen = "(",
  RParen = ")",
  LBrace = "{",
  RBrace = "}",
  Comma = ",",
  Semicolon = ";",
  // 文件末尾哨兵 —— 语法器循环靠它停止,不必每处判越界。
  Eof = "<eof>",
}

/** 字面量的运行期值。Lox-mini 没有 null,字面量只有数与串(bool 用专门 token 表达)。 */
export type LiteralValue = number | string | boolean;

export interface Token {
  readonly kind: TokenKind;
  /** 源码中的原始切片(诊断回显用)。见文件头不变量。 */
  readonly lexeme: string;
  readonly span: Span;
  /** 仅字面量 token 有值;非字面量为 undefined。 */
  readonly literal?: LiteralValue;
}

/**
 * 关键字表: 源码文本 → TokenKind。词法器扫出一个标识符后查这张表,
 * 命中则是关键字 token,否则是 Identifier。全表小写(大小写敏感,见失败模式)。
 */
export const KEYWORDS: ReadonlyMap<string, TokenKind> = new Map([
  ["let", TokenKind.Let],
  ["fn", TokenKind.Fn],
  ["if", TokenKind.If],
  ["else", TokenKind.Else],
  ["while", TokenKind.While],
  ["print", TokenKind.Print],
  ["return", TokenKind.Return],
  ["true", TokenKind.True],
  ["false", TokenKind.False],
]);

/** 便捷构造,集中一处保证字段顺序/默认值一致。 */
export function makeToken(kind: TokenKind, lexeme: string, span: Span, literal?: LiteralValue): Token {
  return literal === undefined ? { kind, lexeme, span } : { kind, lexeme, span, literal };
}
