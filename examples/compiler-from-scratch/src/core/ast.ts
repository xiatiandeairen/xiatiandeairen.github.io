// core/ast.ts — Expr/Stmt 判别联合 AST + 确定性 S 表达式打印器(对拍用)。
//
// WHY 存在:
//   AST 是语法器的产出、也是语义/类型/IR 阶段的共同输入。它是全书的「中央数据
//   结构」,必须只有一处定义。用判别联合(discriminated union,靠 `kind` 字段区分)
//   而非 class 继承: TS 的穷尽性检查能在你新增一种节点却忘了处理时,在编译期就
//   把每个 switch 报红 —— 这正是教学想要的「编译器自己帮你查漏」。
//
//   astToSExpr 存在是为了「逐章对拍」: 第2章产出的 AST、第3章 resolve 后的 AST,
//   都能 dump 成一个确定性字符串,直接 assertEq 比对,而不必肉眼读对象树。
//
// 不变量(INVARIANTS):
//   - 每个节点带 span,指回源码(诊断 + 后续阶段定位)。
//   - 判别字段统一叫 `kind`,且其字符串值全局唯一(Expr 与 Stmt 不重名)。
//   - astToSExpr 输出确定性: 字段顺序固定(按下面 switch 的书写顺序),不依赖
//     对象 key 枚举顺序、不含地址/时间/随机量。同一棵树永远 dump 出同一字符串。
//   - astToSExpr 不打印 span —— span 含偏移量,会让对拍对源码空白敏感。对拍只比结构。
//
// 失败模式(FAILURE MODES):
//   - 新增节点种类却没在 astToSExpr 加分支: default 分支抛错(而非静默漏打),
//     让你立刻发现。TS 的 never 检查也会在编译期提示。

import type { Span } from "./source.js";
import type { LiteralValue } from "./token.js";

// ---------- 表达式 (Expr) ----------

export interface BinaryExpr {
  readonly kind: "Binary";
  readonly op: string; // 运算符 lexeme: "+" "==" "<" ...
  readonly left: Expr;
  readonly right: Expr;
  readonly span: Span;
}

export interface UnaryExpr {
  readonly kind: "Unary";
  readonly op: string; // "-" 或 "!"
  readonly operand: Expr;
  readonly span: Span;
}

export interface LiteralExpr {
  readonly kind: "Literal";
  readonly value: LiteralValue;
  readonly span: Span;
}

export interface VarExpr {
  readonly kind: "Var";
  readonly name: string;
  readonly span: Span;
  /**
   * resolver(stage03)回填: 变量到声明的「跳几层作用域」距离。
   * 语法阶段为 undefined;resolve 后填上,VM/解释器据此 O(1) 定位而非动态查链。
   * 放在 AST 上而非旁路 map: 让「未 resolve」与「已 resolve」是同一结构的两个状态,
   * 后续阶段只读一个可空字段,不必管理两份数据。
   */
  depth?: number;
}

export interface AssignExpr {
  readonly kind: "Assign";
  readonly name: string;
  readonly value: Expr;
  readonly span: Span;
  depth?: number; // 同 VarExpr.depth,resolver 回填。
}

export interface CallExpr {
  readonly kind: "Call";
  readonly callee: Expr;
  readonly args: readonly Expr[];
  readonly span: Span;
}

/** 函数表达式(一等函数;`fn name(...)` 与匿名 `fn(...)` 共用此结构,name 可空)。 */
export interface FnExpr {
  readonly kind: "Fn";
  readonly name?: string;
  readonly params: readonly string[];
  readonly body: readonly Stmt[];
  readonly span: Span;
}

export type Expr =
  | BinaryExpr
  | UnaryExpr
  | LiteralExpr
  | VarExpr
  | AssignExpr
  | CallExpr
  | FnExpr;

// ---------- 语句 (Stmt) ----------

export interface LetStmt {
  readonly kind: "Let";
  readonly name: string;
  readonly init: Expr; // Lox-mini 要求 let 必须初始化(无未定义变量),简化语义。
  readonly span: Span;
}

export interface PrintStmt {
  readonly kind: "Print";
  readonly value: Expr;
  readonly span: Span;
}

export interface IfStmt {
  readonly kind: "If";
  readonly cond: Expr;
  readonly then: readonly Stmt[];
  readonly else?: readonly Stmt[];
  readonly span: Span;
}

export interface WhileStmt {
  readonly kind: "While";
  readonly cond: Expr;
  readonly body: readonly Stmt[];
  readonly span: Span;
}

export interface BlockStmt {
  readonly kind: "Block";
  readonly body: readonly Stmt[];
  readonly span: Span;
}

export interface ReturnStmt {
  readonly kind: "Return";
  readonly value?: Expr; // 裸 return 合法(返回 nil 语义,值为 undefined)。
  readonly span: Span;
}

/** 表达式语句: 一个表达式 + 分号(求值后丢弃结果,通常用于调用副作用)。 */
export interface ExprStmt {
  readonly kind: "ExprStmt";
  readonly expr: Expr;
  readonly span: Span;
}

export type Stmt =
  | LetStmt
  | PrintStmt
  | IfStmt
  | WhileStmt
  | BlockStmt
  | ReturnStmt
  | ExprStmt;

/** 一个完整程序就是顶层语句序列。 */
export type Program = readonly Stmt[];

// ---------- 确定性 S 表达式打印(对拍) ----------

/**
 * 把 Expr/Stmt/Program dump 成 S 表达式字符串。确定性见文件头不变量。
 * 形如: (let x (+ 1 2)) / (if (< n 2) ((return n)) ...)
 * 用于 stage 之间端到端对拍 —— 把整棵树压成一行可 diff 的文本。
 */
export function astToSExpr(node: Expr | Stmt | Program): string {
  if (Array.isArray(node)) {
    // Program 或语句序列 → 用空格连接的括号组。
    return "(" + (node as readonly Stmt[]).map(astToSExpr).join(" ") + ")";
  }
  const n = node as Expr | Stmt;
  switch (n.kind) {
    // --- Expr ---
    case "Binary":
      return `(${n.op} ${astToSExpr(n.left)} ${astToSExpr(n.right)})`;
    case "Unary":
      return `(${n.op} ${astToSExpr(n.operand)})`;
    case "Literal":
      return sexprLiteral(n.value);
    case "Var":
      // depth 已 resolve 时附带 @深度,让「resolve 前/后」对拍能看出差异。
      return n.depth === undefined ? n.name : `${n.name}@${n.depth}`;
    case "Assign":
      return `(set! ${n.name}${n.depth === undefined ? "" : "@" + n.depth} ${astToSExpr(n.value)})`;
    case "Call":
      return `(call ${astToSExpr(n.callee)}${n.args.map((a) => " " + astToSExpr(a)).join("")})`;
    case "Fn":
      return `(fn ${n.name ?? "<anon>"} (${n.params.join(" ")}) ${astToSExpr(n.body as readonly Stmt[])})`;
    // --- Stmt ---
    case "Let":
      return `(let ${n.name} ${astToSExpr(n.init)})`;
    case "Print":
      return `(print ${astToSExpr(n.value)})`;
    case "If":
      return `(if ${astToSExpr(n.cond)} ${astToSExpr(n.then as readonly Stmt[])}${
        n.else ? " " + astToSExpr(n.else as readonly Stmt[]) : ""
      })`;
    case "While":
      return `(while ${astToSExpr(n.cond)} ${astToSExpr(n.body as readonly Stmt[])})`;
    case "Block":
      return `(block ${astToSExpr(n.body as readonly Stmt[])})`;
    case "Return":
      return n.value === undefined ? "(return)" : `(return ${astToSExpr(n.value)})`;
    case "ExprStmt":
      return astToSExpr(n.expr);
    default: {
      // 穷尽性兜底: 新增节点忘了加分支会走到这。TS 的 never 也会在编译期报。
      const _never: never = n;
      throw new Error(`astToSExpr: unhandled node ${JSON.stringify(_never)}`);
    }
  }
}

/** 字面量的确定性渲染: 字符串加引号、布尔用 #t/#f(Scheme 风,避免和标识符混)。 */
function sexprLiteral(v: LiteralValue): string {
  if (typeof v === "string") return JSON.stringify(v); // 转义引号,确定性。
  if (typeof v === "boolean") return v ? "#t" : "#f";
  return String(v); // number: JS 的 String(number) 是确定性的。
}
