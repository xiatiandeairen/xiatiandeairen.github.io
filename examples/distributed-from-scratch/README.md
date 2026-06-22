# 分布式系统从零：用 TypeScript 手写故障、时间、复制与共识

这是《分布式系统从零》一书的配套代码仓。目标：**专家级、可跑、反 survey**——
代码真能跑、机制是真的、注释讲 why 不讲 what。

每个 stage 是一个独立可执行文件，跑在 `src/core/` 提供的**确定性内存仿真底座**上：
时间是离散事件虚拟时钟（不碰 `Date.now()`），网络是内存消息总线（可注入延迟 /
丢包 / 分区 / 乱序 / 重复），随机性来自单一种子 PRNG。因此每次运行**字节级可复现**，
一个共识 bug 或一次脑裂能被精确重放。

- 纯算法 / 纯计算，离线 CPU 可跑。
- **无需任何 LLM / API key / 网络**。
- 确定性：换种子换实验，同种子同结果。
- 每个 stage 都 demo **失败模式**，不只 happy path。

## 跑法

先安装（仅 dev 依赖：tsx / typescript / @types/node）：

```bash
npm install
```

然后跑任意 stage：

```bash
npm run stage01   # Lamport 逻辑时钟：因果序 vs 全序，及它分不出并发
npm run stage02   # 向量时钟：精确检测并发与因果，代价是 O(N) 元数据
npm run stage03   # RPC 与故障检测：超时重试、幂等、phi-accrual 误判 vs 漏判
npm run stage04   # Raft 领导选举：term / 投票 / 脑裂安全 / 分区下的选举
npm run stage05   # Raft 日志复制：matchIndex / commitIndex / 落后跟随者修复
npm run stage06   # CAP 取舍：分区下 CP 拒写 vs AP 可用但分叉
npm run stage07   # CRDT 收敛：无协调的最终一致，乱序+重复仍收敛
```

类型检查（不产物）：

```bash
npm run typecheck
```

## 仓库结构

```
src/
  core/            共享仿真底座（全书复用，零外部依赖）
    prng.ts        种子 PRNG —— 全书随机性唯一来源
    clock.ts       SimClock —— 离散事件虚拟时钟 + 真实 wall-clock bench
    network.ts     Network —— 内存消息总线 + 故障注入
    node.ts        Node —— 协议状态机基类（onMessage / setTimer）
    metrics.ts     Histogram / Stats —— 延迟 / 轮次 / 消息数统计
    assert.ts      assert / invariant —— 每 tick 校验全局不变量
    scenario.ts    Scenario —— seed + 故障脚本 + 断言打包成可复现实验
  stage01-..stage07-*.ts   各章节可执行示例（互不 import）
```

> 注意：stage 文件加载即跑 `main()`，所以它们**互不 import**；要复用逻辑请放进 `src/core/`。

## 诚实数字声明

- 消息数 / 轮次 / 不变量违反次数等都是仿真**真算出来**的计数，确定性可复现。
- 标 `(est.)` 的是估算（如折算成真实网络的耗时）。
- 仿真用 toy 拓扑（3~5 节点）和合成延迟分布：**绝对值偏乐观**，可迁移的是
  相对趋势（如向量时钟元数据随节点数线性增长、分区下 CP 与 AP 的可用性差异）。
- 个别 stage 会用 `core/clock.ts::bench` 真测 wall-clock 算法吞吐，结果**机器相关**，已标注。
