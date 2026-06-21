---
title: "标量 autograd：一个会反向传播的数"
slug: "01"
collection: "tiny-dl"
order: 1
summary: "全书从最小单元起步：一个标量 Value 节点同时携带 data 和 grad，重载 +/*/和非线性，在做前向算术的同时把局部导数挂到一张动态计算图上。本章手写 _backward 闭包加拓扑排序 backward()，把链式法则变成可执行代码，讲透为什么 grad 必须累加、为什么需要拓扑序、忘记 zero_grad 会怎样。第 2 章把这同一个核扩到 n 维张量，后面的层、优化器、attention、nanoGPT 全是这个核在规模上的放大。"
topics:
  - "深度学习"
tags: []
createdAt: "2026-06-21T08:00:00.000Z"
updatedAt: "2026-06-21T08:00:00.000Z"
---

先把一句话钉死：反向传播不是一条公式，是「**对计算图做拓扑逆序遍历，沿途把局部雅可比 (local Jacobian，每个算子对自己输入的导数) 累加进每个节点的 grad**」。你要是把它当成「∂L/∂w 的链式法则展开」去记，迟早被 PyTorch 的 `retain_graph` 报错、二阶导、`x*x` 这种共享节点搞懵。把它当成图遍历去记，这些就全是同一件事的不同侧面。

这一章我们手写这个图遍历。整章只处理**标量**——一个数。不是因为标量有用，是因为标量能让你看清反向传播的全部机制，而不被张量 shape、broadcast、内存布局这些第 2 章才该操心的东西分心。本章对应的可跑代码是 `examples/tiny-dl-from-scratch/src/stage01-scalar-autograd.ts`,它编译后纯离线运行,不连网不调任何 LLM,固定种子,同一台机器每次跑出逐字节相同的数。下面引用的所有数字都来自它的真实输出。

## 一个数,外加「我怎么影响最终输出」

普通的 `number` 是死的:`3.0` 就是 `3.0`,它不知道自己被谁用了、用完会怎样。autograd 要的是一个**活的数**——它记得自己是怎么算出来的,于是能反过来回答「我变一点点,最终的 loss 变多少」。这个「变多少」就是梯度 (gradient,某个量对它的偏导)。

所以 `Value` 节点带三样东西:`data`(前向算出来的值)、`grad`(反向填进来的梯度)、以及一个 `_backward` 闭包(知道怎么把自己的 grad 分发给「生我的那些父节点」)。看 `stage01-scalar-autograd.ts` 的类定义:

```typescript
class Value {
  data: number;
  grad: number;
  // _backward: scatter THIS node's grad into its parents. No-op for leaves.
  _backward: () => void;
  // _prev: parents, for topo discovery. Set dedupes x-used-twice (x*x).
  _prev: Set<Value>;
  op: string;

  constructor(data: number, prev: Value[] = [], op = "") {
    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = new Set(prev);
    this.op = op;
  }
```

注意 `grad` 初始化成 `0`,`_backward` 初始化成空函数(叶子节点没有父节点要分发)。还有 `_prev` 是 `Set` 不是数组——这是个关键细节,等讲到 `x*x` 时你会明白为什么。

### 算术即建图:重载 + 和 *

前向计算和建图是**同一个动作**。你写 `a.mul(b)` 时,它一边算出 `a.data * b.data`,一边把「这个结果的两个父节点是 a 和 b」记下来,还把「这一步的局部导数怎么回传」打包成闭包挂上去:

```typescript
mul(other: Value | number): Value {
  const o = other instanceof Value ? other : new Value(other);
  const out = new Value(this.data * o.data, [this, o], "*");
  out._backward = () => {
    Value.bump(this, o.data * out.grad);
    Value.bump(o, this.data * out.grad);
  };
  return out;
}
```

乘法的局部导数是大白话级别的高中数学:`out = a*b`,那 `∂out/∂a = b`、`∂out/∂b = a`。`_backward` 干的事就是「把上游传下来的 `out.grad`,按局部导数缩放后,塞给每个父节点」——这正是链式法则的一步:`∂L/∂a = ∂L/∂out · ∂out/∂a = out.grad · b`。

`add` 更简单,`out = a+b` 时两个父节点的局部导数都是 1,所以原样把 `out.grad` 传下去。`tanh` / `exp` / `pow` 同理,各自只是换个局部导数公式。**整个 autograd 引擎的「智能」就这么多**:每个算子知道自己那一步的导数。剩下的全是遍历。

`Value.bump` 这个辅助函数先按下不表,它藏着本章最重要的那个坑——`+=` 还是 `=`。

## backward():拓扑序为什么不能省

光有每个节点的 `_backward` 闭包还不够,你得**按正确顺序**调它们。这就是 `backward()`:

```typescript
backward(): void {
  const topo: Value[] = [];
  const visited = new Set<Value>();
  const build = (v: Value) => {
    if (visited.has(v)) return;
    visited.add(v);
    for (const p of v._prev) build(p);
    topo.push(v);
  };
  build(this);
  this.grad = 1;
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
}
```

三步:(1) 深度优先建一个拓扑序 `topo`,父节点排在子节点前面;(2) 给起点(通常是 loss)种下 `grad = 1`,因为 `∂L/∂L = 1`;(3) **逆着拓扑序**走,逐个调 `_backward`。

为什么必须逆拓扑序?**因为一个节点要把 grad 分发给父节点之前,它自己的 grad 必须先收齐**。设想 `d = a + a*a` 这种结构,节点 `a` 的 grad 来自两条路径——一条经过加法,一条经过乘法。如果你在乘法那条路还没算完时就让 `a` 往下分发,`a` 拿到的是个半成品 grad,全错。逆拓扑序保证:轮到某个节点 `_backward` 时,**它所有的下游消费者都已经处理完、grad 都已经累加进它的 `grad` 字段了**。

✦ 这就是为什么 PyTorch 默认一次 `backward()` 之后图就被释放、再调一次会报 `Trying to backward through the graph a second time`。图不是公式,是一次性的遍历计划:建好 `topo`、走一遍、释放。你想走第二遍(比如算高阶导、或同一图喂两个 loss)就得 `retain_graph=True` 显式留着这个遍历结构。理解了「backward = 一次拓扑逆序遍历」,这个报错就不再是黑魔法,而是「你想重放一个已经播完释放了的录像」。我们这个手写版每次 `backward()` 都重新 `build` 一遍 `topo`,等于每次都重新录,所以不会撞上这个限制——代价是不能复用图,这正是 PyTorch 用 `retain_graph` 在管理的那个 trade-off。

### 先证明它真的对:解析梯度 vs 数值梯度

手写的导数公式怎么知道没写错符号、没漏个系数?标准做法是 **gradient check**:拿解析梯度(autograd 算的)和数值梯度(用有限差分硬怼出来的)对一下。数值梯度用中心差分 `(f(x+ε) - f(x-ε)) / 2ε`——选中心差分而不是前向差分,是因为前向差分误差是 O(ε)、中心差分是 O(ε²),在 ε=1e-5 时中心差分能压到 float64 的噪声地板附近。

`stage01` 拿一个真实的 2 输入神经元 `y = tanh(w1·x1 + w2·x2 + b)` 跑这个检查,种子固定,输出是:

```text
1) 2-input neuron: analytic grad vs numerical central diff
inputs:  x1=0.9660  x2=1.1231   params: w1=-0.0492  w2=-0.1809  b=1.3061
forward: y = 0.783875
  param   analytic        numerical       rel.error
  x1    -1.897934e-2    -1.897934e-2    2.674e-11
  x2    -6.974103e-2    -6.974103e-2    2.133e-11
  w1    3.724223e-1     3.724223e-1     7.831e-12
  w2    4.330024e-1     4.330024e-1     1.723e-11
  b     3.855407e-1     3.855407e-1     1.729e-11
max relative error: 2.674e-11
PASS (< 1e-6): true
```

最大相对误差 `2.674e-11`,比 `1e-6` 的及格线低五个数量级。这个数本身有信息量:它不是 0(浮点中心差分必然带 ε² 量级的截断误差 + 浮点舍入),但小到只剩数值噪声——说明解析公式和真实导数在数学上一致,差距全来自有限差分这个近似手段,不是来自 bug。**gradient check 是你写任何新算子后第一件该做的事**,后面第 3 章加新层、新激活函数,都靠它兜底。

## 失败模式:grad 必须累加,不能覆盖

现在揭开 `Value.bump` 的盖子。这是整个引擎里唯一决定「`+=` 还是 `=`」的地方:

```typescript
private static bump(node: Value, delta: number): void {
  if (Value.accumulate) node.grad += delta;
  else node.grad = delta;
}
```

`stage01` 故意留了个静态开关 `Value.accumulate`,好让我们在同一张图上把规则从「累加」翻成「覆盖」,亲眼看错误的数。(正式的 core 引擎 `examples/tiny-dl-from-scratch/src/core/autograd.ts` 没有这个开关——它的 `_backward` 直接写死 `this.grad += ...`,冻结成正确的;教 bug 需要一个我们能注入错误的类,所以本章另写了一份可控的 `Value`。)

为什么这事是死规定而非风格选择?**因为一个节点被用了 N 次,反向时 grad 就从 N 条路径汇回来,必须求和**。最干净的例子是 `f = x*x`:同一个 `x` 同时当左因子和右因子。`df/dx = 2x`,但反向模式是顺着两条路径分别到达 `x`,每条贡献 `x`,两条加起来才是 `2x`。用覆盖,第二条路径会把第一条写进去的值冲掉,`x.grad` 停在 `x`(一条路径)而不是 `2x`。

`stage01` 在同一张 `x*x` 图上分别用两套规则跑,输出:

```text
2) FAILURE MODE: overwrite (=) vs accumulate (+=) on shared node x in f=x*x
x = 2.759373   (f = x*x, true df/dx = 2x = 5.518746)
accumulate (+=): x.grad = 5.518746   -> matches 2x: true
overwrite  (=) : x.grad = 2.759373   -> wrong by 2.759373 (only 1 of 2 paths counted)
ratio accumulate/overwrite = 2.0000 (exactly 2x: the dropped path)
```

`5.518746 / 2.759373 = 2.0000`,差的正好是被覆盖掉的那一条路径。这里之所以恰好差 2 倍,是因为 `x*x` 刚好两条路径、两条相等;一般情况下被吞的是「除最后写入那条之外的所有路径之和」,差多少看图的扇出 (fan-out,一个节点被多少下游用到) 结构。

这个 bug 最阴险的地方在 `stage01` 输出的最后两行点破了:

```text
>> Lesson: a value reused N times fans grad into N paths; reverse-mode MUST sum them.
>> Overwrite silently halves (here) the grad — training would crawl or diverge, no crash.
```

**它不崩溃**。没有异常、没有 NaN、没有 shape mismatch。程序照跑,loss 照降,只是梯度系统性偏小——模型训得慢得离谱,或者在某些结构下因为梯度方向被扭曲而发散。你会以为是学习率没调好、是数据有问题、是模型容量不够,排查几天才发现是 autograd 的累加规则错了。这就是为什么所有正经框架(以及我们的 core 引擎)把 `+=` 焊死在引擎里,不给你犯错的接口。回头看那个 `_prev` 是 `Set` 而不是数组:`x*x` 里 `x` 出现两次,`Set` 把它去重成一个节点,所以 `topo` 里 `x` 只排一次、只被分发到一次——而那一次的 `_backward` 内部对 `x` 调了两次 `bump`(左因子一次、右因子一次),靠 `+=` 把两条路径汇总。去重 + 累加,缺一不可。

### 同一个坑的孪生兄弟:忘记 zero_grad

上面是「**一次** backward 内部」的累加。还有个跨 step 的版本:`+=` 是把 grad 累进去,从不清零。所以**单步内**它是对的(汇总多路径),**跨步之间**它就成了污染源——这一步的梯度会叠在上一步残留的梯度上。

core 引擎的注释把这个不变量写得很清楚(`core/autograd.ts`):

```text
//   Corollary: callers MUST zero grads between steps (see nn.zeroGrad / optim.zeroGrad).
```

也就是说:训练循环每个 step 必须先 `zeroGrad()` 清零所有参数的 grad,再 `backward()`,再 `step()` 更新。漏掉清零,第 5 步的梯度里就掺着第 1234 步的陈年梯度,等效于一个你没意识到的、不断膨胀的 momentum,训练行为彻底不可预测。这正是 PyTorch 里 `optimizer.zero_grad()` 那一行的由来——很多人当它是仪式性样板,其实它在对冲的就是「`+=` 不自己清零」这个底层设计的必然后果。**累加是特性,清零是你的责任**,这俩是同一枚硬币的两面。本书把清零的活儿留给第 4 章的优化器 `optim.zeroGrad`,这里你只要记住:看到 `+=`,就该条件反射地问「谁负责清零」。

## ⚡ 前沿:动态 tape vs 静态 trace,内核没变

你刚手写的这个东西,术语叫 **tape-based autodiff**(基于磁带的自动微分):前向时把每个算子按执行顺序「录」进一张图(我们的 `topo` 就是这张磁带),反向时倒带重放。这正是 PyTorch 的 eager 模式、JAX 的 `grad`、TensorFlow 的 `GradientTape` 共用的内核。**你写的不是玩具版,是真东西的最小核**——区别只在我们用标量、它们用张量,我们每次重建图、它们做了缓存和算子融合优化。

那现代框架在卷什么?卷的不是 autograd 算法本身,而是**什么时候录磁带**:

- **动态(tape-based / define-by-run)**:边跑边录,每次前向都重建图。灵活——控制流、循环、递归随便写,图跟着 Python 走。代价是每次都付建图开销,且优化器看不到完整图、难做跨算子优化。这就是本章的做法,也是 PyTorch 的默认模式。
- **静态(traced / define-and-run)**:先把计算「描」成一张固定的图,编译、优化(算子融合、内存复用、kernel 选择)、然后反复执行。快,但你得先有图——动态控制流要特殊处理。`torch.compile`、JAX `jit`、XLA 走的是这条路。

`torch.compile` 的整个存在意义,就是想「**写代码像动态、跑起来像静态**」:用 tracing 在运行时捕获一段动态图,编译成优化后的静态 kernel,下次同样的 shape 直接复用。

⚡ 但这两者的统一——**既要动态的完全灵活、又要静态的极致性能、还不让用户改一行代码**——目前没有通用解,仍是活跃研究方向。数据依赖的控制流(循环次数取决于运行时数据)、动态 shape(每个 batch shape 不同)这些场景,trace 出来的图要么频繁失效重编译、要么得 fallback 回动态模式,性能收益就吃掉了。`torch.compile` 的 graph break(图被动态控制流打断、退回 eager)、JAX 对动态 shape 的种种限制,都是这个未解张力的具体症状。但无论哪一派、未来怎么演进,反向那一步——**拓扑逆序遍历 + 局部雅可比累加**——和你这章手写的一模一样。内核四十年没变。

---

下一章把这个标量 `Value` 升级成 `Tensor`:同一套 `_backward` 闭包 + 拓扑 `backward()`,但每个节点从一个数变成一块 `Float64Array`,局部导数从标量乘法变成矩阵运算,还要处理 broadcast 这个新的梯度路径来源。累加规则、拓扑序、清零责任——这三条本章定下的铁律,到张量、到 attention、到 nanoGPT,一条都不会变。
