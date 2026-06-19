---
title: "IR 与 SSA（编译器域）"
slug: "5-03"
collection: "tech-library"
group: "编译器"
order: 5003
summary: "TL;DR 编译器中端的整个世界，是被一个朴素愿望逼出来的：分析一段代码时，我想问\"这个变量现在是什么值\"，而不必把控制流走一遍。AST 答不了这个问题（树形结构里\"现在\"无定义），机器码也答不了（寄存器被反复覆写）。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-14T21:26:41.000Z"
updatedAt: "2026-06-14T21:26:41.000Z"
---
> **TL;DR**
> 编译器中端的整个世界，是被一个朴素愿望逼出来的：分析一段代码时，我想问"这个变量现在是什么值"，而不必把控制流走一遍。AST 答不了这个问题（树形结构里"现在"无定义），机器码也答不了（寄存器被反复覆写）。于是出现了两层抽象：**线性三地址 IR**（把表达式树拍平成 `t = a op b` 的指令流 + 基本块 + CFG），以及在它之上的 **SSA（Static Single Assignment）**——每个变量在文本上只被赋值一次，控制流汇合点用 **φ 函数** 选择来源。SSA 的回报是：每个变量的"定义"全程唯一，use-def 链退化成一根指针，常量传播 / DCE / GVN 这些优化从"需要 dataflow 不动点迭代"降级成"扫一遍即可"。代价是构造 SSA 本身需要支配树和支配边界（dominance frontier）这套非平凡机器，以及 φ 这个"伪指令"必须在最后被 lower 回真实拷贝（out-of-SSA）。
>
> 本章把这条路径走完：AST → 线性 IR → CFG → 支配关系 → 支配边界 → 插 φ → 重命名，每一步配真实源码（QBE 的 `ssa.c` / `cfg.c` 逐字、LLVM LangRef 逐字）和一个**实测通过**的 Python demo（菱形 merge + while 循环两个场景）。重点辨析两条工业路线：**LLVM 让前端只发 `alloca`/`load`/`store`，由 `mem2reg` 构造 SSA**；**QBE 让前端发非 SSA 的临时量，由 QBE 自己 fixup**——两者都刻意不让前端碰 φ，这个共识本身就是一个深刻的工程结论。
>
> **前置依赖**：你需要熟悉控制流图（CFG）、基本块、支配（dominator）的直觉；读过第 2 章（词法/语法/AST，本系列）会更顺。代码 demo 仅依赖 Python 3.8+ 标准库。术语保留英文。

---

## 一、设计考古：为什么会有"中间"这一层

### 1.1 三地址码（Three-Address Code, TAC）：把树拍平

最早的编译器（FORTRAN I, 1957）几乎是直接 AST → 机器码。问题很快暴露：表达式 `a = b * c + d` 在 AST 里是一棵树，要做优化（公共子表达式消除、常量折叠）必须在树上反复模式匹配，而树的"求值顺序"是隐式的、嵌套的，难以谈论"第几步算什么"。

三地址码的提出就是为了**显式化求值顺序与中间结果**。据 Wikipedia 对 TAC 的定义【真实源码 en.wikipedia.org/wiki/Three-address_code】：

> "three-address code (often abbreviated to TAC or 3AC) is an intermediate code used by optimizing compilers to aid in the implementation of code-improving transformations."

名字"三地址"来自指令形态——**至多三个操作数**，典型是"一次赋值 + 一个二元运算"：

> "Each TAC instruction has at most three operands and is typically a combination of assignment and a binary operator. For example, `t1 := t2 + t3`."（同上）

于是 `a = b * c + d` 被拍平成：

```
t1 = b * c
t2 = t1 + d
a  = t2
```

每个中间结果都有名字（`t1`/`t2`），求值顺序变成线性指令流。这套表示的经典出处是 Aho / Sethi / Ullman 的 *Compilers: Principles, Techniques, and Tools*（1986，俗称龙书 Dragon Book），Wikipedia 即以此为奠基引用【真实来源 en.wikipedia.org/wiki/Three-address_code，引 Aho-Sethi-Ullman 1986】。TAC 在实现上常落成 **quadruples**（四元组 `(op, arg1, arg2, result)`）或 **triples**（三元组，用指令下标代替显式 result）。Wikipedia 还指出 TAC 的一个近亲是 **A-normal form (ANF)**，可视为 TAC 的函数式精化。

> ⚠ 易混点：TAC 是"形态"约束（≤3 操作数、线性），不是"唯一赋值"约束。同一个变量 `a` 在 TAC 里可以被赋值很多次。SSA 才加上"每个名字静态只定义一次"。两者是正交的两层。

### 1.2 从 TAC 到 SSA：被"变量名复用"逼出来的

TAC 解决了"显式中间结果"，但留下一个分析上的硬伤：变量名被复用。看：

```
x = 1
x = 2          ; x 被重新定义
y = x + 1      ; 这里的 x 是哪个？要做 reaching-definition 分析才知道
```

要回答"`y = x + 1` 里的 `x` 来自哪条定义"，传统做法是 **reaching definitions** dataflow 分析——在 CFG 上做不动点迭代，对每个程序点维护"哪些定义能到达这里"。这很贵，而且每加一种优化都要重做类似分析。

SSA 的核心 insight（Cornell CS6120 课程原话）【真实来源 cs.cornell.edu/courses/cs6120/2020fa/lesson/5】：

> "many of the annoying problems in implementing analyses & optimizations stem from variable name conflicts." … SSA ensures that "every variable has exactly one static assignment location."

即：**把每次赋值改名成一个新版本**，`x` 拆成 `x1, x2, x3…`，那么"`y` 用的是哪个 `x`"在语法上就直接写死了——它写的是 `x2` 就是 `x2`，无需任何 dataflow。Wikipedia 的总结【真实来源 en.wikipedia.org/wiki/Static_single-assignment_form】：

> "when looking at a use of a variable there is only one place where that variable may have received a value."

但改名会撞到控制流汇合点：

```
        x = 1
   if c ──┬── x = 2     (B1)
          └── x = 3     (B2)
        y = x + 1       (B3：x 来自 B1 还是 B2？)
```

在 B3，`x` 可能是 `x2`（来自 B1）也可能是 `x3`（来自 B2），到底用哪个取决于运行时走了哪条边。SSA 的解法是引入 **φ 函数**：在 B3 入口写 `x4 = φ(x2, x3)`，语义是"如果控制流从 B1 来就取 x2，从 B2 来就取 x3"。

### 1.3 φ 函数与 SSA 的正式诞生（设计考古：论文与人物）

SSA 不是一个人一篇论文一次性发明的，而是 IBM 研究院在 1980 年代逐步成型。Wikipedia 给出清晰时间线【真实来源 en.wikipedia.org/wiki/Static_single-assignment_form】：

- **1986**：Cytron, Lowry, Zadeck 引入 "birthpoints, identity assignments, and variable renaming"——已有改名 + 汇合点占位的雏形。
- **1987**：Ferrante 与 Cytron 证明这种改名 "removes all false dependencies for scalars"（消除标量的假依赖）。
- **1988**：Rosen, Wegman, Zadeck **"replaced the identity assignments with Φ-functions, introduced the name 'static single-assignment form'"**——φ 函数和"SSA"这个名字正式登场。
- **1989**：Rosen, Wegman, Zadeck, Cytron, Ferrante 找到 "an efficient means of converting programs to SSA form"。

这条 1989 年的高效构造法，最终成文为编译器领域被引最高的论文之一：

> **Ron Cytron, Jeanne Ferrante, Barry K. Rosen, Mark N. Wegman, F. Kenneth Zadeck. "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph." ACM Transactions on Programming Languages and Systems (TOPLAS), Vol. 13, No. 4, pp. 451–490, October 1991.** DOI: 10.1145/115372.115320
> 【真实来源 dl.acm.org/doi/10.1145/115372.115320 与 scholar.google.com 元数据，2428+ citations】

这篇论文的核心贡献，是用 **dominance frontier（支配边界）** 这个新概念，给出在**任意 CFG** 上高效放置 φ 函数的算法（详见第三节）。

还有一个流传甚广的冷知识——**φ（Phi）这个符号的来历**。Wikipedia 原文【真实来源 en.wikipedia.org/wiki/Static_single-assignment_form】：

> The Φ symbol was chosen as "a more publishable version of 'phony function.'"

φ 最初被戏称为 "phony function"（假函数，因为它不是真实可执行的运算，只是个"按来路选值"的占位），Φ 是它"更适合发表"的版本。这个细节很能说明 φ 的本质：**它不是一条会被 CPU 执行的指令，而是一个分析期的语义标注**，最终必须被 lower 回真实拷贝。

---

## 二、模板六段之一：核心机制拆解

### 2.1 LLVM IR 的 φ：工业级 IR 的规范长什么样

先看真实工业 IR 怎么定义这一切。LLVM IR 的官方文档 LangRef 开篇即声明它有三种等价形态【真实来源 llvm.org/docs/LangRef.html】：

> "The LLVM code representation is designed to be used in three different forms: as an in-memory compiler IR, as an on-disk bitcode representation (suitable for fast loading by a Just-In-Time compiler), and as a human readable assembly language representation."

三态等价——内存中的对象图、磁盘上的 bitcode（给 JIT 快速加载）、人类可读的 `.ll` 汇编。这是工业 IR 的标配设计：同一份 IR，调试看文本、传输用二进制、优化在内存。

LLVM IR 是 **SSA-based representation**。它的 φ 指令规范（LangRef，**逐字**）【真实源码 releases.llvm.org/19.1.0/docs/LangRef.html】：

> **Syntax:**
> `<result> = phi <ty> [ <val0>, <label0>], ...`
>
> **Overview:** The 'phi' instruction is used to implement the φ node in the SSA graph representing the program.
>
> **Semantics:** At runtime, the phi instruction evaluates to whichever value corresponds to the basic block from which control flow most recently arrived.
>
> **Example:**
> ```llvm
> Loop:       ; Infinite loop that counts from 0 on up...
>   %Next = phi i32 [ 0, %Entry ], [ %Next2, %Loop ]
>   %Next2 = add i32 %Next, 1
>   br label %Loop
> ```

注意这个 example 本身就是一个**循环的 φ**：`%Next` 的来源是"从 %Entry 来取 0，从 %Loop 来取 %Next2"——而 `%Next2` 在文本上定义于 `%Next` 之后。这正是循环 SSA 的特征：φ 可以引用一个"程序顺序在它之后"的值。记住这个形态，我们的 demo 会精确复现它。

LangRef 还规定了配套的基本块结构与终结指令。分支指令（**逐字**）【真实源码 releases.llvm.org/19.1.0/docs/LangRef.html】：

> **Syntax:** `br i1 <cond>, label <iftrue>, label <iffalse>` or `br label <target>`
> **Example:**
> ```llvm
>   %cond = icmp eq i32 %a, %b
>   br i1 %cond, label %IfEqual, label %IfUnequal
> ```

以及【示意，基于 LangRef 表述归纳】：一个 **basic block** 是一串指令，以一条 **terminator**（`br`/`ret`/`switch` 等）结尾；φ 指令必须出现在基本块**最前面**（在任何非 φ 指令之前）。这条"φ 必须在块首"的硬约束，根源是 φ 的语义——它要在"块的任何真实计算发生之前"就根据来路定好值。

### 2.2 QBE IL 的 φ：极简 IR 的对照

QBE 是 Quentin Carbonneaux 写的极简 backend（MIT License，© 2015-2017 Quentin Carbonneaux）【真实来源 github.com/8l/qbe/LICENSE】，代码量比 LLVM 小两个数量级，非常适合精读。它的 φ 语法（IL 文档，**逐字**）【真实源码 c9x.me/compile/doc/il.html】：

```
PHI := %IDENT '=' BASETY 'phi' ( @IDENT VAL ),
```

例子（**逐字**）：

```
@retstmt
        %y =w phi @ift 1, @iff 2
        ret %y
```

（`%y` 是 word 类型，"从 @ift 块来取 1，从 @iff 块来取 2"。）块结构（**逐字**）：

```
BLOCK :=
    @IDENT NL     # Block label
    ( PHI NL )*   # Phi instructions
    ( INST NL )*  # Regular instructions
    JUMP NL       # Jump or return
```

φ 同样被强制放在块首（`PHI` 在 `INST` 之前）。QBE 文档明确：

> "Phi instructions are specific to SSA form" … "QBE assumes that if a variable is defined by a phi it respects all the SSA invariants."（同上）

### 2.3 关键工程共识：前端不该自己造 SSA

这是本章最重要的一个"机制由来"。**两大工业 IR 都刻意让前端不直接生成 φ**，但路线不同：

**LLVM 路线**——前端（如 Clang）把所有局部可变变量都放进栈：用 `alloca` 开一块栈空间，读写变量翻译成 `load`/`store`，**完全不发 φ**；然后由优化 pass `mem2reg`（PromoteMemoryToRegister）把这些 alloca 提升成 SSA 值。LLVM Kaleidoscope 教程的原话（**逐字**）【真实源码 releases.llvm.org/8.0.0/docs/tutorial/LangImpl07.html】：

> "SSA construction requires non-trivial algorithms and data structures, so it is inconvenient and wasteful for every front-end to have to reproduce this logic."

它给出 `mem2reg` 能提升的精确条件（**逐字**，同上）：

> - "mem2reg only looks for alloca instructions in the entry block of the function"
> - "mem2reg only promotes allocas whose uses are direct loads and stores"
> - "mem2reg only works on allocas of first class values … and only if the array size of the allocation is 1"

为什么只看 entry block 的 alloca？因为 entry block 保证只执行一次，alloca 只分配一次，分析最简单。为什么只提升"只被 load/store 使用"的 alloca？因为一旦变量地址被取走传给函数（escape），它就可能被别名修改，无法安全地放进 SSA 寄存器。

**QBE 路线**——前端发**非 SSA** 的临时量（同一个 `%x` 可以被定义多次），QBE 自己 fixup 成 SSA。QBE IL 文档（**逐字**）【真实源码 c9x.me/compile/doc/il.html】：

> "First and foremost, phi instructions are NOT necessary when writing a frontend to QBE."
> "Contrary to LLVM, QBE is able to fixup programs not in SSA form without requiring the boilerplate of loading and storing in memory."

**对照本质**：LLVM 让前端把可变状态"塞进内存"（alloca），用一个通用 pass 把内存提升回 SSA——好处是前端极简（连"哪些是变量"都不用想，全 alloca），调试信息（debug info）天然落在内存操作上；代价是要先生成一堆 load/store 再消掉。QBE 让前端直接发非 SSA 寄存器，省掉 load/store 往返——好处是 IR 更短、更接近最终形态；代价是前端要自己保证"非 SSA 但语义正确"。但**两者的共识是一致的**：φ 的放置（支配边界 + 重命名）是非平凡算法，不该让每个前端各写一遍。这正是 Cytron 1991 算法被沉淀进 backend 的工程理由。

---

## 三、模板六段之二：SSA 构造算法逐行精读

构造 SSA 标准流程分三步：① 算支配关系 → ② 算支配边界 → ③ 用支配边界放 φ + 沿支配树重命名。我们对照 **QBE 真实源码**（最完整、最短）+ **Cornell CS6120 标准伪码**（最清晰）双轨精读。

### 3.1 支配关系：定义与 QBE 的迭代实现

正式定义（Wikipedia，**逐字**）【真实来源 en.wikipedia.org/wiki/Dominator_(graph_theory)】：

> "A node d of a control-flow graph dominates a node n if every path from the entry node to n must go through d."
> "A node d strictly dominates a node n if d dominates n and d does not equal n."
> "The immediate dominator or idom of a node n is the unique node that strictly dominates n but does not strictly dominate any other node that strictly dominates n."

历史（同上）：支配概念由 **Reese T. Prosser 1959** 提出；首个计算算法 **Lowry & Medlock 1969**；近线性算法 **Lengauer & Tarjan 1979**（"A fast algorithm for finding dominators in a flowgraph"）；以及工程上最常用、最易实现的 **Cooper, Harvey, Kennedy 2001**（"A Simple, Fast Dominance Algorithm"）——它故意放弃 Lengauer-Tarjan 的渐进最优，换取实现简单且在真实 CFG 上实际更快。

QBE 用的就是 Cooper-Harvey-Kennedy 的迭代法。`cfg.c` 的 `filldom`（**真实源码 8l/qbe@cfg.c**，逐字）：

```c
filldom(Fn *fn)
{
	Blk *b, *d;
	int ch;
	uint n, p;

	for (b=fn->start; b; b=b->link) {
		b->idom = 0;
		b->dom = 0;
		b->dlink = 0;
	}
	do {
		ch = 0;
		for (n=1; n<fn->nblk; n++) {
			b = fn->rpo[n];
			d = 0;
			for (p=0; p<b->npred; p++)
				if (b->pred[p]->idom
				||  b->pred[p] == fn->start)
					d = inter(d, b->pred[p]);
			if (d != b->idom) {
				ch++;
				b->idom = d;
			}
		}
	} while (ch);
	for (b=fn->start; b; b=b->link)
		if ((d=b->idom)) {
			assert(d != b);
			b->dlink = d->dom;
			d->dom = b;
		}
}
```

**逐行注解**：
- `for (b=...) { b->idom = 0; ... }`：初始化，所有块的 idom 设为空。
- `do { ... } while (ch)`：不动点迭代，`ch`（changed）为本轮是否有 idom 变化；不再变化就收敛。
- `b = fn->rpo[n]`：**按 reverse postorder（RPO）遍历**——这是 Cooper-Harvey-Kennedy 的关键，RPO 保证处理一个块时它的（前向边）前驱多半已被处理，迭代收敛极快（通常 2 轮）。从 `n=1` 开始是跳过 entry（entry 的 idom 是它自己/空）。
- 内层 `for (p...) ... d = inter(d, b->pred[p])`：对当前块所有**已知 idom 的前驱**求交（`inter` 在支配树上"爬到公共祖先"）。`|| b->pred[p] == fn->start` 处理 entry 作为前驱的特例。
- `if (d != b->idom) { ch++; b->idom = d; }`：idom 变了就标记继续迭代。
- 最后一段 `b->dlink = d->dom; d->dom = b;`：把 idom 关系**反向串成支配树**——`d->dom` 是 d 的孩子链表头，`dlink` 是兄弟指针。后面重命名要 DFS 这棵树。

QBE 的 `dom`/`sdom` 查询（**真实源码 8l/qbe@cfg.c**，逐字），利用了 RPO 编号 `id` 的单调性：

```c
int
sdom(Blk *b1, Blk *b2)
{
	assert(b1 && b2);
	if (b1 == b2)
		return 0;
	while (b2->id > b1->id)
		b2 = b2->idom;
	return b1 == b2;
}

int
dom(Blk *b1, Blk *b2)
{
	return b1 == b2 || sdom(b1, b2);
}
```

`sdom(b1, b2)` 判断 b1 是否严格支配 b2：沿 b2 的 idom 链上爬（`b2->id > b1->id` 时），爬到 id 不大于 b1 时看是否撞上 b1。简洁到只有一个 while 循环——因为支配树 + RPO 编号已经把信息编码好了。

### 3.2 支配边界（Dominance Frontier）：SSA 的心脏

**为什么需要它**：直觉上，变量 `x` 在块 D 被定义后，"这个定义能独占控制到哪里"就是 D 支配的区域；一旦走出这个区域、和别的路径汇合，就需要 φ。**支配边界 DF(D) 恰好就是"D 的支配区域的边缘汇合点"**。

正式定义（Wikipedia，**逐字**）【真实来源 en.wikipedia.org/wiki/Dominator_(graph_theory)】：

> "The dominance frontier of a node d is the set of all nodes ni such that d dominates an immediate predecessor of ni, but d does not strictly dominate ni."

换句话说：B 在 DF(A) 里 ⟺ A 支配 B 的某个前驱，但 A 不严格支配 B 自己。Cornell CS6120 的口语版（**逐字**）【真实来源 cs.cornell.edu/courses/cs6120/2020fa/lesson/5】：

> "A dominance frontier is the set of nodes that are just 'one edge away' from being dominated by a given node." … "A's dominance frontier contains B iff A does not strictly dominate B, but A does dominate some predecessor of B."

QBE 的 `fillfron`（**真实源码 8l/qbe@cfg.c**，逐字）是 Cytron 1991 DF 算法的精炼实现：

```c
static void
addfron(Blk *a, Blk *b)
{
	uint n;

	for (n=0; n<a->nfron; n++)
		if (a->fron[n] == b)
			return;
	if (!a->nfron)
		a->fron = vnew(++a->nfron, sizeof a->fron[0], Pfn);
	else
		vgrow(&a->fron, ++a->nfron);
	a->fron[a->nfron-1] = b;
}

/* fill the dominance frontier */
void
fillfron(Fn *fn)
{
	Blk *a, *b;

	for (b=fn->start; b; b=b->link)
		b->nfron = 0;
	for (b=fn->start; b; b=b->link) {
		if (b->s1)
			for (a=b; !sdom(a, b->s1); a=a->idom)
				addfron(a, b->s1);
		if (b->s2)
			for (a=b; !sdom(a, b->s2); a=a->idom)
				addfron(a, b->s2);
	}
}
```

**逐行注解**（这段是全章最精妙的代码）：
- 外层遍历每个块 `b` 及其后继 `b->s1` / `b->s2`（QBE 每块至多两个后继）。
- 核心是 `for (a=b; !sdom(a, b->s1); a=a->idom) addfron(a, b->s1);`：对每条边 `b → s`，**从 b 出发沿 idom 链上爬**，把 `s` 加入沿途每个块的 DF，直到爬到某个**严格支配 s** 的块为止。
- 为什么这样就对？因为对边 `b → s`：b 显然支配 b 自己（s 的前驱），所以只要 b 不严格支配 s，s 就在 DF(b) 里——满足定义。然后 b 的 idom、idom 的 idom……只要它们也不严格支配 s，同样把 s 加进它们的 DF。一旦爬到严格支配 s 的祖先，再往上就都严格支配 s 了，停止。
- `addfron` 只是去重的动态数组 append。

这就是 Cytron 1991 那个被引 2400+ 次的 DF 算法，落成 20 行 C。注意它**完全不需要显式存"支配集合"**，只靠 idom 链 + `sdom` 查询，O(边数 × 支配树高) 就算完。

**标准伪码对照**（Cornell CS6120，给出更声明式的视角；这里是它的"算支配集合"基础步，**逐字**）【真实来源 cs.cornell.edu/courses/cs6120/2020fa/lesson/5】：

```
dom = {every block -> all blocks}
while dom is still changing:
    for vertex in CFG:
        dom[vertex] = {vertex} ∪ ⋂(dom[p] for p in vertex.preds}
```

（我们的 demo 用的就是这个声明式版本算 dom，再从 dom 推 idom 和 DF，便于教学；QBE 用 idom 链优化掉了显式集合。）

### 3.3 插 φ：迭代支配边界 + worklist

有了 DF，放 φ 的规则极简：**对每个变量 v，在"v 所有定义块的支配边界"处放 φ；而放了 φ 等于在那里又产生一个 v 的定义，于是要迭代**——这叫 **iterated dominance frontier (DF⁺)**。

Cornell CS6120 标准伪码（**逐字**）【真实来源 cs.cornell.edu/courses/cs6120/2020fa/lesson/5】：

```
for v in vars:
   for d in Defs[v]:  # Blocks where v is assigned.
     for block in DF[d]:  # Dominance frontier.
       Add a ϕ-node to block,
         unless we have done so already.
       Add block to Defs[v] (because it now writes to v!),
         unless it's already in there.
```

注意 "Add block to Defs[v]" 这一步——把新放 φ 的块也当作 v 的定义块，于是它的 DF 也要处理，这就是"迭代"。

QBE 的对应实现是 `ssa.c` 里的 `phiins`（**真实源码 8l/qbe@ssa.c**，逐字，核心 worklist 段）：

```c
		bscopy(defs, u);
		while (bp != be) {
			fn->tmp[t].visit = t;
			b = *bp++;
			bsclr(u, b->id);
			for (n=0; n<b->nfron; n++) {
				a = b->fron[n];
				if (a->visit++ == 0)
				if (bshas(a->in, t)) {
					p = alloc(sizeof *p);
					p->cls = k;
					p->to = TMP(t);
					p->link = a->phi;
					a->phi = p;
					if (!bshas(defs, a->id))
					if (!bshas(u, a->id)) {
						bsset(u, a->id);
						*--bp = a;
					}
				}
			}
		}
```

**逐行注解**：
- `while (bp != be)`：`bp..be` 是 worklist（一个指针窗口），初始装着变量 `t` 的所有定义块（前面代码用位集 `u` 收集）。
- `b = *bp++; bsclr(u, b->id);`：取出一个定义块 b。
- `for (n...) a = b->fron[n];`：遍历 b 的支配边界 `b->fron`（就是 3.2 算出来的）。
- `if (a->visit++ == 0)`：每个边界块只放一次 φ（`visit` 计数去重）。
- **`if (bshas(a->in, t))`**：关键——只在变量 t **live-in** 于 a 时才放 φ。这一步把"minimal SSA"提升为 **"pruned SSA"**（见 §五对照表）：t 在 a 入口都不活跃，放 φ 纯属浪费。QBE 因此在 `ssa()` 里先调 `filllive`。
- `p = alloc(...); p->to = TMP(t); a->phi = p;`：在块 a 头部挂一个 φ 节点（目标是临时量 t）。
- 末尾 `if (!bshas(defs, a->id)) ... *--bp = a;`：如果 a 还不是定义块，把它**加入 worklist**——这就是 iterated DF 的"迭代"落地。

### 3.4 重命名：支配树 DFS + 版本栈

放完 φ（此时所有 φ 的目标和实参都还写着同一个旧名 v），最后一步是**给每个定义取新版本号、把每个使用改写成"当前作用域里最新的版本"**。算法是：**沿支配树做 DFS，每个变量维护一个版本栈**。

Cornell CS6120 标准伪码（**逐字**）【真实来源 cs.cornell.edu/courses/cs6120/2020fa/lesson/5】：

```
stack[v] is a stack of variable names (for every variable v)

def rename(block):
  for instr in block:
    replace each argument to instr with stack[old name]
    replace instr's destination with a new name
    push that new name onto stack[old name]
  for s in block's successors:
    for p in s's ϕ-nodes:
      Assuming p is for a variable v, make it read from stack[v].
  for b in blocks immediately dominated by block:
    rename(b)
  pop all the names we just pushed onto the stacks

rename(entry)
```

四个要点：① 块内先改写 use（用栈顶版本），再给 def 压新版本；② 在**后继**块的 φ 里填"来自本块"的实参（用本块退出时的栈顶）；③ 递归**支配树直接孩子**（不是 CFG 后继！）；④ 回溯时弹出本块压的所有版本。

QBE 对应 `renblk` + `getstk` + `rendef`（**真实源码 8l/qbe@ssa.c**，逐字）：

```c
static void
renblk(Blk *b, Name **stk, Fn *fn)
{
	Phi *p;
	Ins *i;
	Blk *s, **ps, *succ[3];
	int t, m;

	for (p=b->phi; p; p=p->link)
		rendef(&p->to, b, stk, fn);
	for (i=b->ins; i-b->ins < b->nins; i++) {
		for (m=0; m<2; m++) {
			t = i->arg[m].val;
			if (rtype(i->arg[m]) == RTmp)
			if (fn->tmp[t].visit)
				i->arg[m] = getstk(t, b, stk);
		}
		rendef(&i->to, b, stk, fn);
	}
	t = b->jmp.arg.val;
	if (rtype(b->jmp.arg) == RTmp)
	if (fn->tmp[t].visit)
		b->jmp.arg = getstk(t, b, stk);
	succ[0] = b->s1;
	succ[1] = b->s2 == b->s1 ? 0 : b->s2;
	succ[2] = 0;
	for (ps=succ; (s=*ps); ps++)
		for (p=s->phi; p; p=p->link) {
			t = p->to.val;
			if ((t=fn->tmp[t].visit)) {
				m = p->narg++;
				if (m == NPred)
					die("renblk, too many phi args");
				p->arg[m] = getstk(t, b, stk);
				p->blk[m] = b;
			}
		}
	for (s=b->dom; s; s=s->dlink)
		renblk(s, stk, fn);
}
```

**逐行注解**：
- `for (p=b->phi...) rendef(&p->to, ...)`：先给本块所有 φ 的**目标**取新版本（φ 在块首，是这些变量在本块的最新定义）。
- 中间 `for (i...)`：遍历指令，对每个 arg 是临时量且需要改名的（`fn->tmp[t].visit`），调 `getstk` 拿当前版本改写 use；然后 `rendef(&i->to, ...)` 给指令目标取新版本。完全对应伪码"先 use 后 def"。
- `succ[...]` 那段：枚举后继（去重 s1==s2），对后继块的每个 φ，`p->arg[m] = getstk(t, b, stk); p->blk[m] = b;`——**填后继 φ 中来自本块 b 的实参**，正是伪码"make it read from stack[v]"。
- `for (s=b->dom; s; s=s->dlink) renblk(s, ...)`：**递归支配树孩子**（`b->dom` 是孩子链表头，§3.1 `filldom` 末尾建的），不是 CFG 后继。

QBE 处理"退栈"用了个比显式 pop 更优雅的技巧——`getstk`（**真实源码 8l/qbe@ssa.c**，逐字）：

```c
static Ref
getstk(int t, Blk *b, Name **stk)
{
	Name *n, *n1;

	n = stk[t];
	while (n && !dom(n->b, b)) {
		n1 = n;
		n = n->up;
		nfree(n1);
	}
	stk[t] = n;
	if (!n) {
		/* uh, oh, warn */
		return CON_Z;
	} else
		return n->r;
}
```

**注解**：QBE 不在回溯时显式 pop，而是在**取栈顶时惰性清理**：`while (n && !dom(n->b, b))`——只要栈顶版本的定义块 `n->b` **不支配**当前块 b，它对 b 就不可见（已离开作用域），直接弹掉。换言之，栈顶始终是"支配当前块的最近定义"。这把"退栈"和"取值"合并成一个 `dom()` 判断，省掉伪码里显式记录 pushed 列表。`if (!n)` 分支是"用了未定义变量"的兜底（返回 0 常量并本应告警）。

`rendef`（取新版本号 + 压栈，**真实源码 8l/qbe@ssa.c**，逐字）：

```c
rendef(Ref *r, Blk *b, Name **stk, Fn *fn)
{
	Ref r1;
	int t;

	t = r->val;
	if (req(*r, R) || !fn->tmp[t].visit)
		return;
	r1 = refindex(t, fn);
	fn->tmp[r1.val].visit = t;
	stk[t] = nnew(r1, b, stk[t]);
	*r = r1;
}
```

`refindex` 分配新版本临时量 `r1`，`stk[t] = nnew(r1, b, stk[t])` 把 `(新版本, 定义块 b)` 压栈，`*r = r1` 原地改写。

最后看 QBE 把整条流水线串起来的 `ssa` 驱动（**真实源码 8l/qbe@ssa.c**，逐字核心段）：

```c
/* require rpo and ndef */
void
ssa(Fn *fn)
{
	Name **stk, *n;
	int d, nt;
	Blk *b, *b1;

	nt = fn->ntmp;
	stk = emalloc(nt * sizeof stk[0]);
	d = debug['L'];
	debug['L'] = 0;
	filldom(fn);
	...
	fillfron(fn);
	filllive(fn);
	phiins(fn);
	renblk(fn->start, stk, fn);
	while (nt--)
		while ((n=stk[nt])) {
			stk[nt] = n->up;
			nfree(n);
		}
	...
}
```

**这五行就是 SSA 构造的全部骨架**，和本节顺序一一对应：
1. `filldom` — 支配树（§3.1）
2. `fillfron` — 支配边界（§3.2）
3. `filllive` — 活跃性分析（为 pruned SSA，§3.3 那个 `bshas(a->in, t)` 判断要用）
4. `phiins` — 插 φ（§3.3）
5. `renblk(fn->start, ...)` — 从 entry 开始重命名（§3.4）

记住这五步顺序，你就抓住了所有 SSA 构造器的主干（LLVM `mem2reg` 内部也是这套，只是和 alloca 提升耦合在一起）。

---

## 四、⭐ 可运行 demo（重中之重）

**目标**：用纯 Python 标准库实现上面五步流水线（dom → idom → DF → 插 φ → 重命名），跑两个场景——经典菱形 merge 和 while 循环——输出每一步的 IR，与 §三的 QBE 源码逐一呼应。

> **设计为可运行，请在你环境验证。** 依赖：仅 Python 3.8+ 标准库（用到 `dataclasses`、`typing`）。无第三方包。
> 运行：把下面代码存成 `ssa_demo.py`，执行 `python3 ssa_demo.py`。
> 本 demo 已在 Python 3.12.12 实测通过（`python3 -W error` 无任何告警），下方"预期输出"是真实运行结果原样粘贴。

### 4.1 完整代码

```python
#!/usr/bin/env python3
"""
mini-ssa: AST -> linear three-address IR -> CFG -> dominance -> dominance
frontier -> phi insertion -> SSA renaming.

设计为可运行，请在你环境验证。依赖：仅 Python 3.8+ 标准库。
运行： python3 ssa_demo.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 0. 我们直接用"已建好 CFG"的源程序，聚焦 IR/SSA 本身。
#    模拟的源程序（一个带 if 的菱形 + 之后使用 x）：
#
#        x = 1                 ; B0 (entry)
#        if cond goto B2 else B1
#    B1: x = 2
#        goto B3
#    B2: x = 3
#        goto B3
#    B3: y = x + 1            ; 这里 x 是 merge 点 -> 需要 phi
#        ret y
#
# 这是 SSA 教科书最经典的"菱形 merge"形状：x 在 B3 的入口有两个可能来源。
# ---------------------------------------------------------------------------

# 三地址指令：dst = a op b ；op=None 表示传送 dst = a ；特殊 'phi' 由后面插入。
@dataclass
class Instr:
    dst: Optional[str]
    op: Optional[str]
    a: Optional[str] = None
    b: Optional[str] = None
    # phi 节点专用：[(pred_block_id, value), ...]
    phi_args: Optional[list] = None

    def __str__(self) -> str:
        if self.op == "phi":
            args = ", ".join(f"{bid}:{v}" for bid, v in self.phi_args)
            return f"    {self.dst} = phi [{args}]"
        if self.op is None:
            return f"    {self.dst} = {self.a}"
        if self.dst is None:  # ret / 纯副作用
            return f"    {self.op} {self.a}"
        return f"    {self.dst} = {self.a} {self.op} {self.b}"


@dataclass
class Block:
    id: int
    instrs: list = field(default_factory=list)
    preds: list = field(default_factory=list)
    succs: list = field(default_factory=list)


def build_cfg() -> dict:
    blocks = {i: Block(i) for i in range(4)}
    blocks[0].instrs = [Instr("x", None, "1")]            # x = 1
    blocks[1].instrs = [Instr("x", None, "2")]            # x = 2
    blocks[2].instrs = [Instr("x", None, "3")]            # x = 3
    blocks[3].instrs = [Instr("y", "+", "x", "1"),        # y = x + 1
                        Instr(None, "ret", "y")]          # ret y
    # 边：0->1, 0->2, 1->3, 2->3
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for u, v in edges:
        blocks[u].succs.append(v)
        blocks[v].preds.append(u)
    return blocks


# ---------------------------------------------------------------------------
# 1. 支配关系：iterative dataflow（对照 QBE filldom 的迭代思路）。
#    dom[n] = {n} ∪ (∩ dom[p] for p in preds[n])，entry 的 dom = {entry}。
# ---------------------------------------------------------------------------
def compute_dom(blocks: dict, entry: int) -> dict:
    all_ids = set(blocks)
    dom = {n: set(all_ids) for n in all_ids}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for n in all_ids:
            if n == entry:
                continue
            new = set(all_ids)
            for p in blocks[n].preds:
                new &= dom[p]
            new |= {n}
            if new != dom[n]:
                dom[n] = new
                changed = True
    return dom


def strictly_dominates(dom: dict, a: int, b: int) -> bool:
    return a != b and a in dom[b]


def compute_idom(blocks: dict, dom: dict, entry: int) -> dict:
    """immediate dominator：n 的 strict dominator 中，离 n 最近的那个。"""
    idom = {}
    for n in blocks:
        if n == entry:
            continue
        sdoms = [d for d in dom[n] if d != n]
        # idom = sdom 中离 n 最近的那个：即被所有其他 sdom 严格支配的那个。
        for cand in sdoms:
            if all(strictly_dominates(dom, other, cand)
                   for other in sdoms if other != cand):
                idom[n] = cand
                break
    return idom


# ---------------------------------------------------------------------------
# 2. 支配边界 DF：A 的 DF 包含 B，当 A 支配 B 的某个前驱、但不严格支配 B。
#    这是 Cytron et al. 1991 的核心定义。
# ---------------------------------------------------------------------------
def compute_df(blocks: dict, dom: dict) -> dict:
    df = {n: set() for n in blocks}
    for b in blocks.values():
        if len(b.preds) < 2:
            continue  # 只有 join 点（>=2 前驱）才可能是别人的 DF
        for p in b.preds:
            runner = p
            # 从前驱 p 沿支配链上行，直到 runner 严格支配 b 为止
            while not strictly_dominates(dom, runner, b.id):
                df[runner].add(b.id)
                if runner == p and runner in dom[b.id]:
                    # runner 自身支配 b（非严格），停止
                    break
                # 上行一步：用 idom 链
                idoms = [d for d in dom[runner] if d != runner]
                if not idoms:
                    break
                runner = max(idoms, key=lambda d: len(dom[d]))
    return df


# ---------------------------------------------------------------------------
# 3. phi 插入：对每个变量 v，在 (其所有定义块的 DF 的迭代闭包) 处插 phi。
#    对照 Cornell CS6120 的 phi 插入伪码 / QBE phiins 的 worklist。
# ---------------------------------------------------------------------------
def defs_of(blocks: dict) -> dict:
    d = {}
    for b in blocks.values():
        for ins in b.instrs:
            if ins.dst is not None:
                d.setdefault(ins.dst, set()).add(b.id)
    return d


def insert_phis(blocks: dict, df: dict, var_defs: dict) -> None:
    for v, def_blocks in var_defs.items():
        worklist = list(def_blocks)
        has_phi = set()
        # iterated dominance frontier
        while worklist:
            d = worklist.pop()
            for frontier in df[d]:
                if frontier in has_phi:
                    continue
                n_pred = len(blocks[frontier].preds)
                phi = Instr(v, "phi", phi_args=[(p, v) for p in
                                                blocks[frontier].preds])
                blocks[frontier].instrs.insert(0, phi)
                has_phi.add(frontier)
                if frontier not in def_blocks:
                    worklist.append(frontier)


# ---------------------------------------------------------------------------
# 4. 重命名：dominator tree 上 DFS + 每个变量一个版本栈。
#    对照 Cornell CS6120 rename() / QBE renblk + getstk。
# ---------------------------------------------------------------------------
def dom_children(idom: dict, entry: int, blocks: dict) -> dict:
    children = {n: [] for n in blocks}
    for n, d in idom.items():
        children[d].append(n)
    return children


def rename(blocks: dict, idom: dict, entry: int) -> None:
    children = dom_children(idom, entry, blocks)
    counter: dict = {}
    stack: dict = {}

    def new_name(v: str) -> str:
        counter[v] = counter.get(v, 0) + 1
        name = f"{v}{counter[v]}"
        stack.setdefault(v, []).append(name)
        return name

    def top(v: str) -> str:
        s = stack.get(v)
        return s[-1] if s else f"{v}?"  # 未定义即用，理论上不应发生

    def base(name: str) -> str:
        # 去掉版本号后缀，得到原变量名
        i = len(name)
        while i > 0 and name[i - 1].isdigit():
            i -= 1
        return name[:i]

    def is_var(tok: Optional[str]) -> bool:
        return tok is not None and tok[:1].isalpha()

    def rename_block(bid: int):
        pushed = []
        for ins in blocks[bid].instrs:
            if ins.op == "phi":
                nm = new_name(ins.dst)
                pushed.append(base(nm))
                ins.dst = nm
                continue
            # 先 rewrite 使用 (a, b)，再给 dst 取新版本
            if is_var(ins.a):
                ins.a = top(ins.a)
            if is_var(ins.b):
                ins.b = top(ins.b)
            if ins.dst is not None:
                nm = new_name(ins.dst)
                pushed.append(base(nm))
                ins.dst = nm
        # 填后继 phi 节点里来自本块的实参
        for s in blocks[bid].succs:
            for ins in blocks[s].instrs:
                if ins.op == "phi":
                    v = base(ins.dst)
                    ins.phi_args = [(p, top(v) if p == bid else val)
                                    for p, val in ins.phi_args]
        # 递归 dominator tree 子节点
        for c in children[bid]:
            rename_block(c)
        # 退栈
        for v in pushed:
            stack[v].pop()

    rename_block(entry)


# ---------------------------------------------------------------------------
def dump(blocks: dict, title: str) -> None:
    print(f"===== {title} =====")
    for bid in sorted(blocks):
        b = blocks[bid]
        preds = ",".join(map(str, b.preds)) or "-"
        print(f"B{bid}: (preds: {preds})")
        for ins in b.instrs:
            print(ins)
    print()


def main() -> None:
    blocks = build_cfg()
    dump(blocks, "1. 线性三地址 IR (pre-SSA)")

    dom = compute_dom(blocks, 0)
    idom = compute_idom(blocks, dom, 0)
    print("===== 2. 支配关系 =====")
    for n in sorted(blocks):
        ds = ",".join(map(str, sorted(dom[n])))
        print(f"  dom(B{n}) = {{{ds}}}   idom = "
              f"{('B'+str(idom[n])) if n in idom else '— (entry)'}")
    print()

    df = compute_df(blocks, dom)
    print("===== 3. 支配边界 DF =====")
    for n in sorted(blocks):
        fs = ",".join(f"B{x}" for x in sorted(df[n])) or "{}"
        print(f"  DF(B{n}) = {fs}")
    print()

    var_defs = defs_of(blocks)
    insert_phis(blocks, df, var_defs)
    dump(blocks, "4. 插入 phi 之后 (未重命名)")

    rename(blocks, idom, 0)
    dump(blocks, "5. SSA 形式 (重命名 + phi 实参回填)")


if __name__ == "__main__":
    main()
```

### 4.2 预期输出（真实运行结果，原样粘贴）

```
===== 1. 线性三地址 IR (pre-SSA) =====
B0: (preds: -)
    x = 1
B1: (preds: 0)
    x = 2
B2: (preds: 0)
    x = 3
B3: (preds: 1,2)
    y = x + 1
    ret y

===== 2. 支配关系 =====
  dom(B0) = {0}   idom = — (entry)
  dom(B1) = {0,1}   idom = B0
  dom(B2) = {0,2}   idom = B0
  dom(B3) = {0,3}   idom = B0

===== 3. 支配边界 DF =====
  DF(B0) = {}
  DF(B1) = B3
  DF(B2) = B3
  DF(B3) = {}

===== 4. 插入 phi 之后 (未重命名) =====
B0: (preds: -)
    x = 1
B1: (preds: 0)
    x = 2
B2: (preds: 0)
    x = 3
B3: (preds: 1,2)
    x = phi [1:x, 2:x]
    y = x + 1
    ret y

===== 5. SSA 形式 (重命名 + phi 实参回填) =====
B0: (preds: -)
    x1 = 1
B1: (preds: 0)
    x2 = 2
B2: (preds: 0)
    x3 = 3
B3: (preds: 1,2)
    x4 = phi [1:x2, 2:x3]
    y1 = x4 + 1
    ret y1
```

### 4.3 与源码 + 论文的呼应

逐项对照，确认 demo 不是玩具而是真实算法的缩影：

| demo 输出 | 对应的真实机制 | 出处 |
|---|---|---|
| `dom(B3) = {0,3}`（B3 不被 B1/B2 支配） | 菱形 merge 点不被任一分支支配 | dominator 定义，Wikipedia |
| `DF(B1) = B3`, `DF(B2) = B3` | 两分支的支配边界都是汇合块 | Cytron 1991 / QBE `fillfron` §3.2 |
| φ 被放到 B3（x 的定义块的 DF⁺） | iterated dominance frontier 放 φ | Cornell CS6120 / QBE `phiins` §3.3 |
| `x4 = phi [1:x2, 2:x3]` | φ 按前驱块选版本 | LLVM LangRef phi 语义 §2.1 |
| `y1 = x4 + 1`（use 改写成 x4） | 重命名用支配树栈顶版本 | Cornell CS6120 / QBE `renblk` §3.4 |

输出与 §2.1 那条 LLVM LangRef 的 φ example（`%Next = phi i32 [ 0, %Entry ], [ %Next2, %Loop ]`）形态完全一致：φ 实参是"(前驱块, 该来路的值)"对。

### 4.4 验证更难的场景：while 循环（back-edge φ）

把 `build_cfg` 换成下面的循环 CFG（其余代码不动），可验证 demo 正确处理**回边**——这是 SSA 真正的难点，因为 φ 要引用一个"程序顺序在它之后才定义"的值：

```python
# while 循环：
#   i = 0                ; B0 entry
# B1: if i<n goto B2 else B3   ; loop header（回边目标）
# B2: i = i + 1 ; goto B1       ; loop body
# B3: ret i
def build_cfg() -> dict:
    B = {i: Block(i) for i in range(4)}
    B[0].instrs = [Instr("i", None, "0")]
    B[1].instrs = [Instr(None, "iflt", "i")]     # 用 i
    B[2].instrs = [Instr("i", "+", "i", "1")]    # i = i + 1
    B[3].instrs = [Instr(None, "ret", "i")]
    for u, v in [(0, 1), (1, 2), (1, 3), (2, 1)]:  # 2->1 是回边
        B[u].succs.append(v); B[v].preds.append(u)
    return B
```

实测输出（真实运行结果节选）：

```
  dom(B1) = {0,1}   idom = B0
  dom(B2) = {0,1,2}   idom = B1
  dom(B3) = {0,1,3}   idom = B1

  DF(B1) = B1            <-- loop header 在自己的支配边界里！
  DF(B2) = B1

B1: (preds: 0,2)
    i2 = phi [0:i1, 2:i3]   <-- 入口来 i1，回边来 i3
    iflt i2
B2: (preds: 1)
    i3 = i2 + 1
B3: (preds: 1)
    ret i2
```

两个关键正确性信号：① **`DF(B1) = B1`**——循环头在自己的支配边界里（回边使然），所以 φ 正确地放在 B1；② **`i2 = phi [0:i1, 2:i3]`** 引用了 `i3`，而 `i3 = i2 + 1` 在程序顺序上**晚于** φ——这正是 §2.1 LLVM LangRef 那个循环 example 的形态，证明 demo 的支配树重命名正确处理了回边。

> 注意 `compute_df` 里 `if len(b.preds) < 2: continue` 这行：只有 ≥2 前驱的 join 点才可能成为别人的 DF。循环头 B1 有两个前驱（entry 边 + 回边），所以被正确识别。

---

## 五、模板六段之三：方案对比

### 5.1 IR 表示形态对比

| 维度 | AST（树） | 线性 TAC（非 SSA） | SSA |
|---|---|---|---|
| 求值顺序 | 隐式（树形嵌套） | 显式（指令流） | 显式 |
| 变量定义 | 词法作用域 | 可多次重定义 | **静态唯一** |
| use→def 查询 | 需作用域解析 | 需 reaching-def 分析 | **一根指针** |
| 控制流 | 隐含在语句结构 | 显式 CFG | 显式 CFG + φ |
| 适合优化 | 差（要反复树匹配） | 中（要 dataflow） | **优（扫一遍）** |
| 适合 codegen | 差 | 好 | 需先 out-of-SSA |
| 典型使用者 | 前端 / 解释器 | 老式 backend | LLVM/QBE/GCC 中端 |

**具体场景定位**：
- **写树遍历解释器**（如 Crafting Interpreters 的 jlox）：用 AST 足矣，不需要 IR。Crafting Interpreters 在转向 bytecode VM（clox）时给出理由（**逐字**）【真实源码 craftinginterpreters.com/a-virtual-machine.html】："our `run()` function is not recursive—the nested expression tree is flattened out into a linear series of instructions"——把树拍平成线性指令是为了用显式栈替代语言级递归栈。这其实就是 TAC 思想的 bytecode 版。
- **写需要常量传播/DCE/GVN 的优化编译器**：上 SSA。
- **写 JIT 的 baseline 层**：常停在线性 bytecode/TAC，不上 SSA（构造成本 > 收益）；optimizing tier 才上 SSA。

### 5.2 SSA 变体对比

| 变体 | φ 放置策略 | φ 数量 | 计算成本 | 出处 |
|---|---|---|---|---|
| **Minimal SSA** | 在所有 DF⁺ 放 φ | 多（含死 φ） | 低 | Cytron 1991 |
| **Pruned SSA** | 仅变量 live-in 时放 φ | 最少 | 高（需 liveness） | — |
| **Semi-pruned SSA** | 跳过块局部变量 | 中 | 中 | Briggs et al. |

Wikipedia 的定义（**逐字**）【真实来源 en.wikipedia.org/wiki/Static_single-assignment_form】：

> "Pruned SSA" uses "live-variable information" to avoid inserting Φ functions for variables not "live" at merge points. "Semi-pruned SSA" omits Φ functions for "block-local variables," offering faster computation than pruned SSA with fewer optimizations.

**QBE 选 pruned**：§3.3 的 `if (bshas(a->in, t))` 就是 live-in 检查，所以 `ssa()` 里先 `filllive`。我们的 demo 是 **minimal SSA**（没做 liveness，所以可能在变量已死的汇合点也放 φ）——对教学足够，但你扩展时可加 liveness 升级为 pruned（见五件套代码题）。

### 5.3 前端造 SSA 路线对比（§2.3 的总结表）

| 维度 | LLVM（alloca + mem2reg） | QBE（非 SSA temp + fixup） |
|---|---|---|
| 前端发什么 | `alloca`/`load`/`store` | 可多次定义的 `%temp` |
| φ 谁来放 | `mem2reg` pass | QBE `ssa()` |
| 前端复杂度 | **最低**（全 alloca） | 低（要保证语义正确） |
| 初始 IR 体积 | 大（一堆 load/store） | 小 |
| debug info | 天然落在内存操作 | 需另行处理 |
| 不适用边界 | 地址逃逸的变量无法提升 | — |

LLVM 的取舍理由（**逐字**）【真实源码 releases.llvm.org/8.0.0/docs/tutorial/LangImpl07.html】："it is inconvenient and wasteful for every front-end to have to reproduce this logic." 两条路线殊途同归：**φ 放置算法属于 backend，不属于前端**。

---

## 六、模板六段之四：扎根——失败模式 / 真坑 / 根因

下面五个坑，前三个是本 demo 开发中**真实踩到并修复**的，后两个是工业 SSA 的经典陷阱。

### 坑 1：idom 计算的"最近 strict dominator"取反（demo 真实 bug）

**现象**：菱形场景输出正确，一换成循环就崩——`idom(B2)` 算成了 `B0` 而非 `B1`，导致重命名时 `i3 = i1 + 1` 用了错版本（应是 `i2`）。

**根因**：idom 是"strict dominator 中**离 n 最近**的那个"。最近 ⟺ 它被所有其他 strict dominator 严格支配。我最初写成：

```python
# 错：找"不严格支配任何其他 sdom"的——这是最远的那个（=entry 附近）
if all(not strictly_dominates(dom, other, cand) for other in sdoms ...):
```

这条件选出的是**最远**的 strict dominator（谁都不被它支配 → 它在最上面）。正确应反过来——选**被所有其他 sdom 支配**的那个（在最下面，离 n 最近）：

```python
# 对：cand 被所有其他 sdom 严格支配 -> cand 是最近的
if all(strictly_dominates(dom, other, cand) for other in sdoms ...):
```

**教训**："最近支配者"这个词在脑子里是清楚的，落成谓词时极易把方向写反。**菱形测试用例掩盖了 bug**（菱形里所有非 entry 块的 idom 都是 entry，取最远=取最近，碰巧对）——只有循环这种"支配链长度 >2"的结构才暴露。这印证了一条铁律：**SSA 代码必须用带循环的 CFG 测，菱形测不出支配链 bug**。

### 坑 2：重命名递归 CFG 后继而非支配树（经典致命错）

**现象**（若写错）：循环里 φ 实参填错、或在不该可见的地方用到某版本。

**根因**：重命名必须沿**支配树**递归（`for c in children[bid]`），不是沿 CFG 后继。原因：变量版本的可见作用域 = 定义块支配的区域。若沿 CFG 后继递归，会在"不被定义块支配"的旁路块里错误地看到该版本。QBE 用 `for (s=b->dom; s; s=s->dlink)`（支配树孩子），不是 `b->s1/b->s2`（CFG 后继）。**这是新手实现 SSA 最高频的致命 bug。** 我们 demo 的 `rename_block` 末尾 `for c in children[bid]` 严格遵守这一点。

### 坑 3：φ 实参用错"时机"的栈顶（demo 设计要点）

**根因**：填后继块 φ 的实参时，要用**当前块处理完所有指令、退栈之前**的栈顶（代表"沿本块这条边过去时该变量的最新版本"）。在 demo 里，填后继 φ 的代码（`for s in blocks[bid].succs:`）必须放在"块内指令重命名完成之后、递归支配树孩子之前、退栈之前"。放错位置（比如退栈后）会取到错误版本。QBE `renblk` 的顺序——先 rename 本块指令、再填后继 φ、再递归、最后（惰性）退栈——精确编码了这个时序。

### 坑 4：φ 是并行赋值，out-of-SSA 时的 lost-copy / swap 问题

**现象**：把 SSA lower 回普通代码（在前驱块尾部插拷贝替代 φ）时，如果一个块有多个 φ，它们语义上是**并行**发生的。`a1=φ(...); b1=φ(a1...)` 这种存在依赖时，朴素地按顺序插拷贝会用到已被覆写的值（lost copy），或两 φ 互换值时需要临时变量（swap problem）。

**根因**：φ 的语义是"在块入口同时确定所有 φ 的值"，是并行赋值；而机器指令是串行的。正确的 out-of-SSA 要做**并行拷贝串行化**（拓扑排序 + 必要时引入临时量打破环）。这是 SSA 工业实现里 bug 密度最高的环节之一，本 demo 未涉及 out-of-SSA（停在 SSA 形式），扩展见五件套。

### 坑 5：critical edge 导致 φ 无处安放拷贝

**根因**：out-of-SSA 要把"φ 的某个实参"变成"对应前驱块尾的一条拷贝"。但若该前驱有多个后继、且该后继有多个前驱（**critical edge**，关键边），拷贝放前驱块尾会污染它的其他后继，放后继块头会污染来自其他前驱的路径——无处可放。**解法**：先做 **critical edge splitting**（在关键边中间插一个空块），再 lower。LLVM 有专门的 `BreakCriticalEdges` pass。这解释了为什么真实 backend 在 out-of-SSA 前总要先 split critical edges。

---

## 七、章末五件套

### ① 一句话主旨

SSA 用"每个变量静态只定义一次 + 汇合点 φ 选值"把 use-def 链压成常数查询，代价是需要支配边界来放 φ、需要 out-of-SSA 把 φ lower 回拷贝；**φ 放置属于 backend，前端只管发非 SSA 代码（QBE）或 alloca（LLVM）**。

### ② 五个必答检查问题（自测）

1. 为什么 φ 要放在"定义块的支配边界"而不是"所有汇合点"？（答：只有定义能独占控制到达的区域的边缘汇合点才需要选择；非边界的汇合点要么被某定义支配只有一个来源，要么……展开论证。）
2. 为什么放 φ 要**迭代**支配边界（DF⁺）？（答：φ 本身是新定义，会产生新的支配边界。）
3. 重命名为什么沿**支配树**而非 CFG 后继 DFS？（答：版本可见域 = 定义块支配区，见坑 2。）
4. minimal / pruned / semi-pruned SSA 各靠什么信息区分？QBE 选哪个、靠哪行代码？（答：liveness；pruned；`bshas(a->in, t)`。）
5. out-of-SSA 时 lost-copy / swap / critical-edge 三个问题各自的根因和解法？（见坑 4、5。）

### ③ 一个可复现实验

跑 §4.1 demo 得菱形结果；按 §4.4 换成循环 CFG 重跑，观察 `DF(B1)=B1` 和 `i2 = phi [0:i1, 2:i3]`。再故意把 `compute_idom` 的条件改回 `not strictly_dominates(...)`（复现坑 1），看循环场景如何输出错误的 `i3 = i1 + 1`——亲手制造再修复一遍支配链 bug。

### ④ 与真实代码的三个锚点

- **QBE `ssa()` 五步**（filldom → fillfron → filllive → phiins → renblk），8l/qbe@ssa.c — 整个 SSA 构造的最短完整骨架。
- **QBE `fillfron`**，8l/qbe@cfg.c — Cytron 1991 支配边界算法落成 20 行 C。
- **LLVM LangRef phi 指令** + **Kaleidoscope Ch.7** — 工业 IR 的 φ 规范 + "前端别自己造 SSA"的官方论证。

### ⑤ 代码题（扩展 demo，由易到难）

1. **（易）minimal → pruned**：给 demo 加一个简单 liveness 分析（活跃变量 dataflow），在 `insert_phis` 里跳过"在汇合点不 live-in"的 φ。验证：构造一个"定义后在汇合点已死"的变量，确认它不再生成 φ。
2. **（中）out-of-SSA**：实现 `lower_phis`——把每个 φ 替换成"在每个前驱块尾插入 `dst = phi_arg` 拷贝"，删掉 φ。先处理无依赖的简单情况，跑通菱形。
3. **（中）并行拷贝串行化**：扩展第 2 题，处理同一块多个 φ 之间有依赖（拓扑排序）和互换（引入临时量）的情况，构造 `a1=φ(b0,b2); b1=φ(a0,a2)` 这种 swap 用例验证。
4. **（难）critical edge splitting**：在 out-of-SSA 前检测关键边（前驱多后继 ∧ 后继多前驱），自动插入分裂块，再 lower。构造一个带关键边的 CFG 证明不分裂会出错、分裂后正确。
5. **（难）常量传播 on SSA**：在 SSA 形式上实现一趟稀疏常量传播——因为 use-def 是一根指针，沿 def-use 链传播常量，遇到 `φ(c, c)`（所有实参同常量）可折叠。验证 demo 菱形若三分支都赋同一常量，最终 `x4` 被折叠成常量。

---

## 附录：本章引用源一览

**逐字真实源码**：
- LLVM LangRef（phi/br 指令、IR 三态）：llvm.org/docs/LangRef.html、releases.llvm.org/19.1.0/docs/LangRef.html
- LLVM Kaleidoscope Ch.7（mem2reg 哲学）：releases.llvm.org/8.0.0/docs/tutorial/LangImpl07.html
- QBE `ssa.c`（phiins/renblk/getstk/rendef/ssa）：github.com/8l/qbe → ssa.c（raw.githubusercontent.com/8l/qbe/master/ssa.c）
- QBE `cfg.c`（filldom/sdom/dom/fillfron/addfron）：github.com/8l/qbe → cfg.c
- QBE IL 文档（φ 语法、前端不需 φ）：c9x.me/compile/doc/il.html
- QBE LICENSE（MIT, © 2015-2017 Quentin Carbonneaux）：github.com/8l/qbe/LICENSE
- Crafting Interpreters（bytecode 作为线性 IR）：craftinginterpreters.com/a-virtual-machine.html
- Cornell CS6120 lesson 5（SSA 构造标准伪码）：cs.cornell.edu/courses/cs6120/2020fa/lesson/5

**设计考古 / 论文**：
- Cytron, Ferrante, Rosen, Wegman, Zadeck, "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph", ACM TOPLAS 13(4):451–490, 1991（DOI 10.1145/115372.115320）：dl.acm.org/doi/10.1145/115372.115320
- Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau, "Simple and Efficient Construction of Static Single Assignment Form", CC 2013（DOI 10.1007/978-3-642-37051-9_6）：link.springer.com/chapter/10.1007/978-3-642-37051-9_6
- Cooper, Harvey, Kennedy, "A Simple, Fast Dominance Algorithm", 2001（QBE filldom 所用）
- Lengauer & Tarjan, "A fast algorithm for finding dominators in a flowgraph", 1979
- SSA 史 / 三地址码 / 支配定义：en.wikipedia.org/wiki/Static_single-assignment_form、en.wikipedia.org/wiki/Three-address_code、en.wikipedia.org/wiki/Dominator_(graph_theory)

**核实说明**：所有标【真实源码】的 LLVM LangRef、QBE 源码、Cornell 伪码、QBE IL 文档均经 WebFetch 实际抓取核实；QBE 的 `ssa.c`/`cfg.c` 经 curl 直取原始字节后逐字精读（WebFetch 因引用长度限制无法整段返回，故改用直取）。Cytron 1991、Braun 2013、CHK 2001 论文 PDF 为二进制流，WebFetch/Read 均无法解析正文，故其标题/作者/年份/页码经 ACM、Google Scholar、Springer 元数据交叉核实，**算法正文以可解析的 Cornell CS6120 标准伪码 + QBE 真实源码为准**，未从无法读取的 PDF 编造细节。φ 等于 "phony function" 的词源、支配概念史（Prosser 1959 等）均引 Wikipedia 原文。
