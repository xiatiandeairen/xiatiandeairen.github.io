// core/index.ts — 共享底座的统一出口。stage 作者从这里 import,不必记每个文件路径。
//
// 注意: 这是纯 re-export,无运行逻辑、无副作用。import 它不会跑任何东西
// (与 stageNN 文件相反 —— 那些一加载就跑 main(),严禁互相 import)。

export * from "./source.js";
export * from "./token.js";
export * from "./ast.js";
export * from "./diagnostics.js";
export * from "./bench.js";
export * from "./programs.js";
