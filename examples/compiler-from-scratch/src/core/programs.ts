// core/programs.ts — 固定样例程序的加载器:把 examples/programs/*.lox 读成 SourceText。
//
// WHY 存在:
//   全书每章在同一组 .lox 上推进、逐章对拍。各 stage 不该各自拼路径读文件 —— 路径
//   一改全崩。集中到这里,且基于本文件的 URL 解析路径,无论从哪个 cwd 运行 `tsx`
//   都能找到样例(不依赖进程工作目录)。
//
// 不变量(INVARIANTS):
//   - 路径基于 import.meta.url 解析,稳定于 cwd。core 在 src/core/,样例在
//     ../../examples/programs/(相对仓库布局)。
//   - 返回 SourceText 已带 name=文件名,诊断头能直接显示 "fib.lox:3:5"。
//
// 失败模式(FAILURE MODES):
//   - 文件不存在: readFileSync 抛 ENOENT,不吞 —— 样例缺失是配置错误,应当显式炸。

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { SourceText } from "./source.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROGRAMS_DIR = join(HERE, "..", "..", "examples", "programs");

/** 全书共用的样例名(不含扩展名)。stage 引用这些常量,避免裸字符串拼错。 */
export const SAMPLES = {
  fib: "fib",
  counter: "counter",
  loopsum: "loopsum",
  badType: "bad-type",
} as const;

export type SampleName = (typeof SAMPLES)[keyof typeof SAMPLES];

/** 读取一个样例为 SourceText。name 形如 "fib"(不带 .lox)。 */
export function loadSample(name: SampleName): SourceText {
  const file = `${name}.lox`;
  const text = readFileSync(join(PROGRAMS_DIR, file), "utf8");
  return new SourceText(text, file);
}
