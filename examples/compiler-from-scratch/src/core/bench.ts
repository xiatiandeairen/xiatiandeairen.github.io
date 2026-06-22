// core/bench.ts — 实测 / 对拍 / 数字格式化:全书量化输出的唯一出口。
//
// WHY 存在:
//   书里每个「快了 X 倍」「执行了 N 条指令」都必须是代码真测/真数出来的,不能拍脑袋。
//   把测量集中到一处,保证:(1) 测速口径一致(中位数去抖,而非单次 wall-clock,
//   单次受 GC/JIT/调度噪声影响极大);(2) 打印格式一致(千分位);(3) 对拍工具统一
//   (不等就打 diff,而非只说 "failed")。
//
// 诚实数字的纪律(写在代码里强制自己遵守):
//   - timeIt 返回中位数毫秒,但「正文断言」只应使用相对量(加速比 = a.median/b.median),
//     不把绝对毫秒写进断言 —— 绝对值依赖机器,换台机就变,不可复现。
//   - 绝对时间只用于「展示性」打印(标明是本机实测),不作为通过/失败判据。
//   - 指令数 / 节点数 / 分配次数这类「计数」是确定性的,可以直接断言(同输入恒等)。
//
// 不变量(INVARIANTS):
//   - timeIt 先 warmup 再测,丢弃首次(JIT 预热),取剩余的中位数 —— 中位数对偶发
//     长尾(GC)鲁棒,均值不鲁棒。
//   - assertEq 用结构化深比较;不等抛 Error(测试就该停),并打印期望/实际 diff。
//
// 失败模式(FAILURE MODES):
//   - 被测 fn 抛异常: timeIt 不吞,直接上抛 —— 测一个会崩的函数,结果无意义,该崩。
//   - 极快的 fn(< 计时分辨率): 中位数可能为 0;调用方应放大迭代规模而非信 0ms。
//     timeIt 提供 inner 重复参数来放大单次测量,避免量化到 0。

export interface TimeResult {
  /** 中位数毫秒(每次 inner 批的耗时中位数 / inner)。见不变量。 */
  readonly medianMs: number;
  /** 最快一次(仅展示用,衡量「下限」)。 */
  readonly minMs: number;
  readonly samples: number;
}

export interface TimeOptions {
  /** 计时采样次数(取中位数)。默认 7。 */
  runs?: number;
  /** 每次采样内部重复调用次数,放大极快函数的单次测量。默认 1。 */
  inner?: number;
  /** 预热次数(丢弃,触发 JIT)。默认 2。 */
  warmup?: number;
}

/**
 * 测一个无副作用(或副作用幂等)函数的执行时间,返回中位数。
 * 故意不返回 fn 的结果 —— 鼓励调用方把「测正确性」(assertEq)和「测速度」分开,
 * 否则容易把验证逻辑塞进计时循环污染测量。
 */
export function timeIt(fn: () => void, opts: TimeOptions = {}): TimeResult {
  const runs = opts.runs ?? 7;
  const inner = opts.inner ?? 1;
  const warmup = opts.warmup ?? 2;

  for (let i = 0; i < warmup; i++) fn();

  const samples: number[] = [];
  for (let r = 0; r < runs; r++) {
    const t0 = performance.now();
    for (let k = 0; k < inner; k++) fn();
    const t1 = performance.now();
    samples.push((t1 - t0) / inner);
  }
  samples.sort((a, b) => a - b);
  const mid = samples.length >> 1;
  const medianMs =
    samples.length % 2 === 1 ? samples[mid] : (samples[mid - 1] + samples[mid]) / 2;
  return { medianMs, minMs: samples[0], samples: runs };
}

/**
 * 两个 TimeResult 的加速比(b 相对 a)。这是「正文断言」该用的量:相对、可复现。
 * speedup > 1 表示 b 比 a 快。
 */
export function speedup(slow: TimeResult, fast: TimeResult): number {
  // 防 0 除: 若 fast 中位数为 0(过快未放大 inner),用 minMs 兜底,仍为 0 则返回 Infinity。
  const f = fast.medianMs || fast.minMs;
  const s = slow.medianMs || slow.minMs;
  if (f === 0) return Infinity;
  return s / f;
}

/**
 * 对拍断言: actual 与 expected 深比较,不等则打印 diff 并抛异常。
 * 用于 stage 间端到端比对(如 astToSExpr 字符串、VM 输出、指令计数)。
 */
export function assertEq<T>(actual: T, expected: T, msg = "assertEq"): void {
  if (!deepEqual(actual, expected)) {
    const a = stringify(actual);
    const e = stringify(expected);
    throw new Error(
      `${msg}: mismatch\n  expected: ${e}\n  actual:   ${a}`,
    );
  }
}

/** 结构化深比较。够用于 number/string/bool/array/plain object(AST dump、指令数组)。 */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a === null || b === null) return a === b;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((x, i) => deepEqual(x, b[i]));
  }
  if (typeof a === "object" && typeof b === "object") {
    const ak = Object.keys(a as object).sort();
    const bk = Object.keys(b as object).sort();
    if (ak.length !== bk.length || !ak.every((k, i) => k === bk[i])) return false;
    return ak.every((k) =>
      deepEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]),
    );
  }
  return false;
}

function stringify(v: unknown): string {
  if (typeof v === "string") return JSON.stringify(v);
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

/**
 * 千分位格式化整数/浮点。NaN/Infinity 原样返回(不假装成数字)—— 见诚实数字纪律,
 * 一个 Infinity 加速比就该显眼地写 "Infinity",不被格式化掩盖。
 */
export function fmtNum(n: number): string {
  if (!Number.isFinite(n)) return String(n);
  const neg = n < 0;
  const abs = Math.abs(n);
  const [intPart, fracPart] = (Number.isInteger(abs) ? abs.toFixed(0) : abs.toString()).split(".");
  const grouped = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  const body = fracPart !== undefined ? `${grouped}.${fracPart}` : grouped;
  return neg ? "-" + body : body;
}
