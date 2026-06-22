# 数据库引擎从零（database-from-scratch）

用 TypeScript 从零手写一个存储引擎 / 数据库内核。**纯算法、离线可跑、零运行时依赖**（不需要任何 LLM / API key / 网络）。每个 stage 都打印代码真算 / 真测出来的数字：真实 IO 次数、缓冲池命中率、B+树高度、LSM 写放大、WAL fsync 成本、MVCC 可见性、崩溃恢复正确性、查询算子吞吐。

配套《数据库引擎从零》书库合集。设计原则：代码真能跑、机制是真的、注释讲 why 不讲 what、所有量化诚实（能墙钟实测的真测，估算标 `(est.)`，toy 数据说明绝对值偏乐观、可迁移的是相对趋势）。

## 运行

需要 Node ≥ 18。先安装开发依赖（仅 tsx / typescript / @types/node）：

```bash
npm install
```

逐章运行（每条命令独立、确定性、可复现）：

```bash
npm run stage01   # slotted page：变长行的页内布局，实测每页能塞多少行 + 页满失败
npm run stage02   # buffer pool：LRU 缓存页，实测命中率如何决定真实磁盘 IO
npm run stage03   # B+ tree：有序索引，实测 fanout 与树高、点查/范围查的页读次数
npm run stage04   # LSM tree：写优化结构，实测写放大与读放大的取舍
npm run stage05   # WAL：预写日志，实测 fsync 是耐久性的真实代价
npm run stage06   # MVCC：多版本并发控制，复现快照隔离下的可见性与写偏序
npm run stage07   # recovery：注入掉电崩溃，replay WAL，断言恢复后 DB == 崩溃前
npm run stage08   # query executor：火山模型算子，实测扫描/过滤/连接的代价

npm run typecheck # tsc --noEmit
```

## 结构

```
src/core/      全书共享底座（确定性 + 可测量），所有 stage 复用，严禁第三方库
  prng.ts        种子 PRNG（mulberry32），全书唯一随机源，run-to-run 完全确定
  disk.ts        内存模拟磁盘：4096B 固定页，真实计 read/write/fsync，可注入掉电
  page.ts        页二进制布局原语：PageReader/Writer + slotted-page header + PageType
  codec.ts       行/键编码：定长 int + 变长 string；compareKeys 让索引用 memcmp 排序
  clock.ts       LamportClock（事务时间戳 / MVCC 版本）+ bench（hrtime 真实纳秒）
  assert.ts      不变量断言 + printTable（全书统一对齐的指标表）
  scheduler.ts   确定性并发调度器：generator 协程 + PRNG 交错，复现并发异常
src/stage0N-*.ts 每章一个可运行 demo，import 自 core/，结尾 printTable 打印实测数字
```

每个 stage 文件加载即运行 `main()`，所以 **stage 之间互不 import**；要复用逻辑请放进 `core/`。
