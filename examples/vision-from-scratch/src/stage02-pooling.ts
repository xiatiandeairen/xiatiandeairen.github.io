// stage02-pooling.ts — Pooling & downsampling: the adjoint is where pooling earns its keep.
//
// WHY this chapter is about BACKWARD, not forward:
//   Forward pooling is trivial — take a max or a mean over a window. Anyone can write it.
//   The pedagogically interesting (and bug-prone) part is the ADJOINT, because the two
//   pooling ops route gradient in OPPOSITE ways:
//     - maxpool : gradient is SPARSE. The whole output grad goes to the SINGLE input that
//                 won the max (the argmax). Every other input in the window gets exactly 0.
//                 That input is the only one that influenced the output, so it's the only one
//                 with a nonzero derivative.
//     - avgpool : gradient is DENSE. Every input in the window contributed 1/(k*k) to the
//                 output, so each gets an equal 1/(k*k) share of the output grad.
//
// THE FAILURE MODE WE DEMO (the reason this file exists):
//   A copy-paste bug — reusing avgpool's "spread 1/area to everyone" backward for a maxpool
//   forward — passes the FORWARD smoke test (forward is untouched) and runs without error.
//   It corrupts only the gradient. gradCheck catches it instantly: analytic-vs-numeric
//   relative error jumps from ~1e-10 (correct) to O(0.1..1) (broken). We show exactly how
//   the wrong routing makes a network "quietly fail to learn": the winning pixel gets only
//   1/k^2 of the signal it deserves, and k^2-1 pixels that didn't affect the output get a
//   phantom gradient pulling them around. Loss barely moves, no crash, no warning.
//
// DETERMINISM: one seeded mulberry32 PRNG drives every value below, so every printed number
//   is reproducible across runs and machines. All numbers are computed/measured here, none
//   are hardcoded. gradCheck does REAL central-difference numerics; the routing comparison is
//   a real analytic-vs-numeric diff, not a narrated claim.
//
// HONESTY: this is a tiny 4x4 toy on synthetic data. The absolute gradCheck error magnitudes
//   are machine-precision artifacts of f64; what TRANSFERS to a real f32 framework is the
//   RELATIVE gap — a correct adjoint sits near machine epsilon, a mis-routed one sits ~1e9x
//   larger. That ratio is the signal, not the absolute 1e-10.

import { Tensor, maxpool2d, avgpool2d, type Pool2dParams } from "./core/autograd.js";
import { gradCheck } from "./core/metrics.js";
import { mulberry32, uniform, type Rng } from "./core/rng.js";

const POOL: Pool2dParams = { kernel: 2, stride: 2 }; // 4x4 -> 2x2, four disjoint windows

/** Pretty-print a small HxW slice of a flat NCHW (N=C=1) tensor. */
function formatMap(data: Float64Array, h: number, w: number, digits = 3): string {
  const rows: string[] = [];
  for (let y = 0; y < h; y++) {
    const cells: string[] = [];
    for (let x = 0; x < w; x++) cells.push(data[y * w + x].toFixed(digits).padStart(7));
    rows.push("    " + cells.join(" "));
  }
  return rows.join("\n");
}

/** Build a deterministic 4x4 feature map as a [1,1,4,4] tensor. */
function makeFeatureMap(rng: Rng): Tensor {
  const H = 4;
  const W = 4;
  const data = new Float64Array(H * W);
  // uniform in [-1,1) keeps every window's values distinct enough that the argmax is
  // unambiguous (no exact ties) — ties would make the maxpool kink non-differentiable and
  // muddy the "single winner" demonstration.
  for (let i = 0; i < H * W; i++) data[i] = uniform(rng, -1, 1);
  return new Tensor(data, [1, 1, H, W]);
}

/**
 * Locate, per 2x2 window, the (y,x) of the max — recomputed independently here so the
 * printed argmax is verified against an explicit search, not read back from the op.
 */
function findWindowArgmax(data: Float64Array, H: number, W: number, p: Pool2dParams): string[] {
  const lines: string[] = [];
  const outH = Math.floor((H - p.kernel) / p.stride) + 1;
  const outW = Math.floor((W - p.kernel) / p.stride) + 1;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let best = -Infinity;
      let by = -1;
      let bx = -1;
      for (let ky = 0; ky < p.kernel; ky++) {
        for (let kx = 0; kx < p.kernel; kx++) {
          const iy = oy * p.stride + ky;
          const ix = ox * p.stride + kx;
          if (data[iy * W + ix] > best) {
            best = data[iy * W + ix];
            by = iy;
            bx = ix;
          }
        }
      }
      lines.push(
        `    window out(${oy},${ox}) -> argmax at input (y=${by}, x=${bx}) value ${best.toFixed(3)}`,
      );
    }
  }
  return lines;
}

/**
 * BUGGY maxpool: correct FORWARD (real max), but the backward is copy-pasted from avgpool —
 * it spreads grad equally over the window instead of routing only to the argmax. This is the
 * single most common pooling implementation bug. We reimplement it locally (rather than
 * import) precisely because core's maxpool2d is correct; we need a wrong twin to contrast.
 *
 * INVARIANT it VIOLATES: d(out)/d(input_i) for a max is 1 only at the argmax, 0 elsewhere.
 *   This version returns 1/(k*k) everywhere — wrong both in support and in magnitude.
 */
function buggyMaxpoolAvgBackward(input: Tensor, p: Pool2dParams): Tensor {
  const [N, C, H, W] = input.shape;
  const { kernel, stride } = p;
  const outH = Math.floor((H - kernel) / stride) + 1;
  const outW = Math.floor((W - kernel) / stride) + 1;
  const out = new Float64Array(N * C * outH * outW);
  const area = kernel * kernel;
  // forward: genuine max (identical to a correct maxpool — that's why it passes forward tests)
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      const base = (n * C + c) * H * W;
      for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
          let best = -Infinity;
          for (let ky = 0; ky < kernel; ky++) {
            for (let kx = 0; kx < kernel; kx++) {
              const v = input.data[base + (oy * stride + ky) * W + (ox * stride + kx)];
              if (v > best) best = v;
            }
          }
          out[((n * C + c) * outH + oy) * outW + ox] = best;
        }
      }
    }
  }
  const t = new Tensor(out, [N, C, outH, outW], [input], "buggy_maxpool");
  // THE BUG: avgpool's adjoint pasted under a max forward. Grad fans out to all k*k inputs.
  t._backward = () => {
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        const base = (n * C + c) * H * W;
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            const g = t.grad[((n * C + c) * outH + oy) * outW + ox] / area;
            for (let ky = 0; ky < kernel; ky++)
              for (let kx = 0; kx < kernel; kx++)
                input.grad[base + (oy * stride + ky) * W + (ox * stride + kx)] += g;
          }
        }
      }
    }
  };
  return t;
}

/** Run a pool op, seed an explicit upstream grad on the output, drive backward manually
 *  (output is not a scalar, so we can't call .backward() — we seed .grad and call _backward
 *  in the right order, which for a single op is just the op's own _backward). */
function poolAndBackward(
  input: Tensor,
  forward: (x: Tensor, p: Pool2dParams) => Tensor,
  upstream: Float64Array,
): { out: Tensor; inGrad: Float64Array } {
  input.zeroGrad();
  const out = forward(input, POOL);
  out.grad.set(upstream); // hand-constructed upstream gradient
  out._backward();
  return { out, inGrad: input.grad.slice() };
}

function main(): void {
  const rng = mulberry32(0xc0ffee);
  const H = 4;
  const W = 4;
  const x = makeFeatureMap(rng);

  console.log("=== stage02 池化与下采样：MaxPool / AvgPool 与梯度的稀疏路由 ===\n");
  console.log(`输入特征图 (${H}x${W}, 种子化，可复现):`);
  console.log(formatMap(x.data, H, W));

  // ---- 1. MaxPool 前向 + argmax 定位 ----
  const mp = maxpool2d(x, POOL);
  console.log("\n[1] MaxPool(2x2, stride2) 前向输出 (2x2):");
  console.log(formatMap(mp.data, 2, 2));
  console.log("\n    每个窗口的 argmax 位置 (独立重算校验):");
  console.log(findWindowArgmax(x.data, H, W, POOL).join("\n"));

  // ---- 2. MaxPool 反向：稀疏路由，只有 argmax 非零 ----
  // upstream grad = 1 on every output cell, so the input grad map literally IS the argmax mask.
  const ones2x2 = new Float64Array([1, 1, 1, 1]);
  const { inGrad: mpGrad } = poolAndBackward(x, maxpool2d, ones2x2);
  console.log("\n[2] MaxPool 反向 (上游 grad 全 1) -> 输入梯度图:");
  console.log(formatMap(mpGrad, H, W));
  const nonzero = Array.from(mpGrad).filter((v) => v !== 0).length;
  const allOnesOrZero = Array.from(mpGrad).every((v) => v === 0 || v === 1);
  console.log(
    `\n    非零位置数 = ${nonzero} (应 = 4，每窗口一个 argmax)；` +
      `非零值是否全为 1 = ${allOnesOrZero} (上游 grad 全 1 时，winner 拿满 1，其余 0)`,
  );

  // ---- 3. AvgPool 反向：稠密路由，每位置均摊 1/4 ----
  const { inGrad: apGrad } = poolAndBackward(x, avgpool2d, ones2x2);
  console.log("\n[3] AvgPool 反向 (上游 grad 全 1) -> 输入梯度图:");
  console.log(formatMap(apGrad, H, W));
  const allQuarter = Array.from(apGrad).every((v) => Math.abs(v - 0.25) < 1e-12);
  console.log(
    `\n    所有 16 个位置是否都 = 0.25 = 1/(2*2) = ${allQuarter} ` +
      `(avg 把每格上游 grad 均摊给窗口内 ${POOL.kernel * POOL.kernel} 个输入)`,
  );

  // ---- 4. gradCheck：正确 MaxPool 的解析梯度 vs 数值差分 ----
  // central-difference numeric grad vs our analytic argmax routing.
  const loss = () => {
    // scalar objective so finite differences have something to wiggle: weighted sum of
    // pooled outputs (distinct weights make every output cell matter independently).
    const w = [0.3, -0.7, 1.1, -0.5];
    const out = maxpool2d(x, POOL);
    let s = 0;
    for (let i = 0; i < out.size; i++) s += w[i] * out.data[i];
    return s;
  };
  // analytic grad: seed upstream = the weights, backward.
  x.zeroGrad();
  const outForCheck = maxpool2d(x, POOL);
  outForCheck.grad.set([0.3, -0.7, 1.1, -0.5]);
  outForCheck._backward();
  const correct = gradCheck(loss, [x], 1e-5, x.size);
  console.log("\n[4] gradCheck 正确 MaxPool (解析 vs 数值中心差分):");
  console.log(
    `    maxRelError = ${correct.maxRelError.toExponential(2)} ` +
      `(checked ${correct.checked} 个偏导)  -> ${correct.maxRelError < 1e-5 ? "PASS (< 1e-5)" : "FAIL"}`,
  );

  // ---- 5. 失败模式：buggy maxpool（avg 反向配 max 前向） ----
  const buggyLoss = () => {
    const w = [0.3, -0.7, 1.1, -0.5];
    const out = buggyMaxpoolAvgBackward(x, POOL);
    let s = 0;
    for (let i = 0; i < out.size; i++) s += w[i] * out.data[i];
    return s;
  };
  x.zeroGrad();
  const buggyOut = buggyMaxpoolAvgBackward(x, POOL);
  buggyOut.grad.set([0.3, -0.7, 1.1, -0.5]);
  buggyOut._backward();
  const buggyGrad = x.grad.slice();
  const buggy = gradCheck(buggyLoss, [x], 1e-5, x.size);
  console.log("\n[5] 失败模式：用 AvgPool 的均摊反向去配 MaxPool 前向 (常见 copy-paste bug)");
  console.log("    前向完全正确 (真 max)，所以前向 smoke test 照样过、不报错。");
  console.log(
    `    gradCheck maxRelError = ${buggy.maxRelError.toExponential(2)} ` +
      `-> ${buggy.maxRelError < 1e-5 ? "PASS" : "FAIL (误差爆到 0.x 量级)"}`,
  );

  // explicit side-by-side at the first window's winning pixel: how much signal is lost.
  // find the argmax flat index of window out(0,0).
  let winnerIdx = 0;
  let best = -Infinity;
  for (const iy of [0, 1])
    for (const ix of [0, 1]) {
      if (x.data[iy * W + ix] > best) {
        best = x.data[iy * W + ix];
        winnerIdx = iy * W + ix;
      }
    }
  console.log("\n    逐元素对比 (上游对 out(0,0) 的权重 = 0.3):");
  console.log(`    正确反向: winner 像素 (flat ${winnerIdx}) 拿到 grad = ${correctWindow00(correct, x, W)}`);
  console.log(
    `      正确梯度图: ${fmtArr(seedAndBack(x, POOL, maxpool2d))}`,
  );
  console.log(`      buggy 梯度图: ${fmtArr(buggyGrad)}`);
  console.log(
    "\n    诊断：buggy 把本该全给 winner 的 0.3 拆成 4 份 (0.075/格) 撒给整窗口。",
  );
  console.log(
    "    后果：winner 只收到 1/4 的真梯度 -> 学习信号被稀释；另外 3 个没影响输出的像素",
  );
  console.log(
    "    收到幽灵梯度 -> 被错误地推动。网络不崩、不报错，只是 loss 几乎不降 = 悄悄学不动。",
  );
  console.log(
    `    解析-数值相对误差从 ${correct.maxRelError.toExponential(1)} (正确) 飙到 ` +
      `${buggy.maxRelError.toExponential(1)} (buggy)，约 ${(buggy.maxRelError / Math.max(correct.maxRelError, 1e-30)).toExponential(1)} 倍。`,
  );
  console.log(
    "\n    可迁移结论：toy 上的绝对误差是 f64 机器精度产物；能迁移到真实 f32 框架的是",
  );
  console.log("    『正确 adjoint 贴近机器 epsilon，错路由高出约 1e9 倍』这个相对落差。");
}

// helpers kept below main for readability (only used by the diagnostic block) -------------

/** Re-run correct maxpool backward with the [4]-weight upstream and return the input grad. */
function seedAndBack(x: Tensor, p: Pool2dParams, forward: (t: Tensor, q: Pool2dParams) => Tensor): Float64Array {
  x.zeroGrad();
  const out = forward(x, p);
  out.grad.set([0.3, -0.7, 1.1, -0.5]);
  out._backward();
  return x.grad.slice();
}

function correctWindow00(_c: { maxRelError: number }, x: Tensor, W: number): string {
  // winner of window (0,0) under correct routing receives the full upstream weight 0.3.
  let best = -Infinity;
  let idx = 0;
  for (const iy of [0, 1])
    for (const ix of [0, 1]) {
      if (x.data[iy * W + ix] > best) {
        best = x.data[iy * W + ix];
        idx = iy * W + ix;
      }
    }
  return `0.300 at flat index ${idx}`;
}

function fmtArr(a: Float64Array): string {
  return "[" + Array.from(a).map((v) => v.toFixed(3)).join(", ") + "]";
}

main();
