# 编译器从零 · compiler-from-scratch

从零用 TypeScript 实现一个**完整、能跑、机制是真的**的编译器。一门固定的小语言
**Lox-mini**(整数/浮点/布尔/字符串、`let`/赋值、`if`/`while`、一等函数 + 闭包、`print`)
贯穿全书,八个阶段在同一组样例程序上逐章推进、可逐章对拍。

纯算法,**离线、零运行时依赖**(只用 Node 内置 `fs`/`path`/`url` 读样例)。不需要任何
LLM / API key / 网络。所有打印的量化数字都是代码真测/真数出来的。

## 运行

```bash
npm install          # 安装 tsx / typescript / @types/node(devDependencies)
npm run typecheck    # tsc --noEmit,全仓类型检查

npm run stage01      # 词法分析   Lexer
npm run stage02      # 语法分析   Parser    -> AST
npm run stage03      # 名称解析   Resolver  (作用域 / 闭包深度回填)
npm run stage04      # 类型检查   Type check
npm run stage05      # 中间表示   IR lowering
npm run stage06      # 优化       Optimize  (常量折叠 / 死代码等)
npm run stage07      # 虚拟机     VM        (字节码执行 + 指令计数)
npm run stage08      # 代码生成   Codegen
```

每个 `stageNN` 文件加载即运行其 `main()`,自成一个端到端 demo:读一份样例 `.lox`,
跑到本阶段,打印产物 + 真实数字 + 一个**失败模式**演示(不只 happy path)。

> ⚠ stage 文件**不可互相 import**(加载即跑 main),共享逻辑一律放 `src/core/`。

## 目录

```
src/
  core/              全书共享底座(纯 TS,零运行时依赖,确定性)
    source.ts        SourceText: 行列定位 + 「第N行 ^^^」下划线片段
    token.ts         TokenKind 枚举 + Token + 关键字表(词法/语法共用)
    ast.ts           Expr/Stmt 判别联合 + astToSExpr(确定性 S 表达式,对拍用)
    diagnostics.ts   DiagnosticBag: 多错收集 + report(source) 渲染带下划线文本
    bench.ts         timeIt(中位数去抖) / speedup / assertEq(diff) / fmtNum
    programs.ts      样例加载器(基于 import.meta.url,cwd 无关)
    index.ts         统一出口(纯 re-export,无副作用)
  stage01-lexer.ts ... stage08-codegen.ts   各章 demo(逐章交付)
examples/programs/   固定样例程序(全书共用输入)
    fib.lox          斐波那契递归  (函数 + 递归 + 条件)
    counter.lox      闭包计数器    (一等函数 + 词法闭包 + 可变捕获)
    loopsum.lox      循环求和      (while + 赋值 + 算术)
    bad-type.lox     坏样例        (故意类型错误,供类型检查 demo 失败模式)
```

## 设计纪律

- **诚实数字**:`bench.ts` 的正文断言只用**相对量**(加速比 / 指令计数,确定性可复现);
  绝对毫秒只作展示并标注「本机实测」。toy 数据的绝对值偏乐观,可迁移的是相对趋势。
- **统一诊断**:各阶段不散乱 `throw`,一律往 `DiagnosticBag` 报,支持「收集多个错误
  而非首错即停」,最后用 `source` 渲染成带行列下划线的文本。
- **确定性对拍**:`astToSExpr` 把任意 AST 压成固定顺序的字符串,逐章 `assertEq` 比对。
- **注释讲 why**:每个 core 文件头注释列出「为何存在 / 不变量 / 失败模式」。
