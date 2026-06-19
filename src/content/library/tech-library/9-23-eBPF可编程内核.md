---
title: "eBPF 可编程内核（深化版 / 源码级）"
slug: "9-23"
collection: "tech-library"
group: "linux系统"
order: 9023
summary: "一句话定位：eBPF 是一套\"在内核里安全地跑用户逻辑\"的字节码虚拟机。它的真正突破不在\"虚拟机\"三个字，而在 verifier——一个在加载时对字节码做*抽象解释（abstract interpretation）*、用\"双数值域 + 状态去重\"逼近形式化证明的安全引擎。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T09:16:08.000Z"
updatedAt: "2026-06-12T09:16:08.000Z"
---
> **一句话定位**：eBPF 是一套"在内核里安全地跑用户逻辑"的字节码虚拟机。它的真正突破不在"虚拟机"三个字，而在 **verifier**——一个在加载时对字节码做*抽象解释（abstract interpretation）*、用"双数值域 + 状态去重"逼近形式化证明的安全引擎。正因为安全被前移成了"加载时的静态证明"，内核才敢让一段非签名的外来代码直接在 Ring 0 跑原生机器码。这是 Linux 近十年最大的架构变革。

---

## TL;DR · 本章最硬结论

1. **安全是"抽象解释"，不是口号**：verifier 不是运行时沙箱，而是在 `do_check()` 里逐指令做抽象解释。它对每个寄存器同时维护两套数值域——**tnum（位级 tristate）**和**区间（`smin/smax/umin/umax` + 32 位子寄存器版本）**——两域互相收窄。绝大多数 verifier reject 和历史上最严重的 verifier CVE（CVE-2021-3490）都发生在这两套域的更新逻辑里。
2. **它逼近形式化证明，但故意"不上 SMT solver"**：verifier 跑在 `bpf()` syscall 上下文里、要在毫秒级返回、其正确性本身要可人工审计。所以它选了"可判定、保守、会误杀"的抽象域，而不是完备但慢且难审计的符号执行/SMT。这是一个深思熟虑的工程权衡，不是技术不足。
3. **map 是唯一合法持久化手段**：eBPF 程序每次触发都是独立无状态执行，状态全在 map。`bpf()` syscall 从 2014 年第一版（commit `99c55f7d47c0`）起就把 map 设计成"fd 生命周期 + `union bpf_attr` 向后兼容多路复用"，这套 ABI 演进哲学让 eBPF 十年加了几十个命令而没破坏用户态。
4. **附着点决定能力边界，且边界一直在被推**：从 kprobe/XDP 一路扩张到 LSM、struct_ops，直到 6.12 把**调度器**（sched_ext）都开放成可加载 BPF——这一步在社区引发了 Peter Zijlstra 的正面 NAK。选错附着点是初学者最大的坑；理解"为什么这个附着点存在、它替前一代解决了什么"才是专家视角。
5. **CO-RE 是 eBPF 从玩具到基础设施的最后一块拼图**：它解决的不是性能，而是"一份字节码跨内核版本可移植"。在它之前，BCC 在目标机上现编译（背 Clang/LLVM、要内核头、可能压垮生产负载）；CO-RE 用 BTF + 编译期重定位记录 + libbpf 加载期改偏移取代了这一切。
6. **eBPF 不是银弹，verifier 本身就是头号攻击面**：4096（`BPF_MAXINSNS`，非特权硬上限）和 1,000,000（`BPF_COMPLEXITY_LIMIT_INSNS`，特权复杂度墙）是两个完全不同的限制；分支爆炸会在指令数远没到上限时先撞复杂度墙。verifier 是内核最复杂代码之一，历史上多次 bypass 型本地提权——这正是主流发行版默认关闭非特权 eBPF 的根因。

**前置依赖章**：第 2 章（系统调用机制，理解 `bpf()` 多路复用 syscall）、第 1 章（内核架构总览）、第 20/21 章（网络协议栈与 XDP 位置）、第 18 章（中断与下半部，理解 kprobe/tracepoint 执行上下文）、第 5 章（调度器，理解 sched_ext）。

**本章相对原版的增量**：① 新增"设计考古"整章，把 1993→2014→至今的关键 commit、LWN 辩论、社区 NAK 全部带出处落实；② verifier / 解释器 / JIT 硬化 / Spectre 缓解全部换成 **v6.12 真实源码逐段注解**；③ 每个机制补"为什么不用更简单替代 + 失败模式 + 微架构/排队论根因"；④ 三处"具体场景跑一遍"的方案对比。所有代码块标注了来源等级：`【真实源码 v6.12 路径】`=逐字、`【示意代码，非逐字】`=教学简化、`「待核」`=未独立核实。

---

## 23.1 背景：内核可编程的根本张力

### 23.1.1 一个无法两全的三角

系统工程师在生产里永远被同一个矛盾困住：

- **可观测性**：想知道"这个 `tcp_sendmsg` 到底慢在哪"——但内核那条路径上未必有你要的 tracepoint，加一个要改内核源码、等发行版、滚升级，最快几个月。
- **运行时策略**：想在不重启服务的前提下给某个 PID 的流量限速——`iptables` 规则引擎是 O(N) 线性扫描，十万条规则直接崩。

两者本质是同一句话：**内核是静态编译产物，生产需求是动态的**。在传统架构里，"安全（不能崩内核）"和"灵活（任意用户逻辑）"是一对不可调和的矛盾。

### 23.1.2 历史解法与它们的底层缺陷

| 解法 | 核心机制 | 致命缺陷（底层根因） |
|------|---------|---------|
| 内核模块 `.ko` | 动态加载，直接跑 Ring 0 | **共享地址空间、零隔离**：一个野指针即 panic；需签名信任链；热更新难。根因是模块和内核共用一张页表、同一特权级，硬件层面没有任何边界。 |
| kprobe + 自写模块 | 运行时挂任意函数 | 同上，且探针逻辑无人审计。 |
| ptrace / strace | 用户态拦截 syscall | **每次 syscall 两次上下文切换 + 一次调度往返**：tracer 被唤醒、tracee 阻塞。排队论上等于在热路径串了一个高延迟服务台，吞吐塌方。生产不可用。 |
| SystemTap | 内核脚本→编译成 `.ko` | 要 debuginfo、要现场编译、产物仍是模块（回到模块的安全问题），落地链条脆。 |
| netfilter/iptables | 包过滤框架 | 规则链式线性扫描，O(N)；表达力受限于固定匹配器，无法跑任意逻辑。 |

一句话概括：**这些方案要么牺牲安全换灵活（模块），要么牺牲性能换安全（ptrace），要么牺牲可移植/落地性（SystemTap）。三角形没人能占满。**

### 23.1.3 eBPF 的突破口：把"运行时检查"前移成"加载时证明"

eBPF 的核心洞见只有一句：

> **如果能在加载时静态*证明*程序安全（有界、不越界、不泄漏），那它就可以直接在内核裸跑，不需要任何运行时沙箱开销。**

这把成本从"每次执行都付的运行时隔离税"挪到了"加载时付一次的证明税"。运行时零开销，安全由数学保证。整个 eBPF 的设计哲学、它的能力边界、它的全部历史 CVE，都是这一句话的推论。下一节我们就从历史看这句话是怎么一步步被兑现的。

---

## 23.2 设计考古：从 1993 的包过滤器到"吞噬内核"

> 这一节是本章的地基。不了解 eBPF 每一步"为谁、何时、为解决前一代什么缺陷、社区辩论过什么"，就只能停在 API 使用层。下面每个结论都尽量带了 commit / LWN / 论文出处（本章末"取材来源"列出实际抓取的 URL）。

### 23.2.1 cBPF 的诞生（1992 年底成文，USENIX Winter 1993）

经典 BPF 来自 Steven McCanne 与 Van Jacobson 在 Lawrence Berkeley Laboratory 的论文 **《The BSD Packet Filter: A New Architecture for User-level Packet Capture》**（USENIX Winter 1993，1993-01-25 宣讲）。

它要解决的前一代缺陷很具体：当时 Unix 的包过滤器（如 CSPF）用**栈式**求值器，在新兴 RISC CPU 上表现差。论文的核心贡献是把它换成**寄存器式**求值器——按论文数据，比原栈式设计快约 **20×**，并能在拷贝到用户态*之前*在内核里就把不关心的包丢掉，整体比 Sun NIT 快 10–150×。

技术形态上 cBPF 极小：2 个 32 位寄存器（累加器 A、索引 X）+ 一个隐藏栈帧，指令类别就 load/store/alu/jump/ret 几种。`tcpdump host 1.2.3.4` 背后就是 libpcap 编译出的一段 cBPF，可用 `tcpdump -d host 1.2.3.4` 打印反汇编。

**为什么寄存器式比栈式快（底层根因）**：栈式求值器每条指令都要读写内存里的栈顶，数据依赖链长、难以被 CPU 乱序引擎并行；寄存器式把中间值留在寄存器，去掉了大量 load/store 和它们之间的伪依赖。这个"register VM 优于 stack VM"的结论，30 年后在 eBPF 的 64 位寄存器设计里再次被放大。

> cBPF 的历史包袱至今还在：**seccomp 用的是 cBPF，不是 eBPF**（见 23.5.3）。Docker/containerd 默认的 seccomp profile 本质是一段经典 BPF，跑在 syscall 入口过滤系统调用号。别把它和 eBPF 混为一谈。

### 23.2.2 2014 的大重写：Alexei Starovoitov 把 BPF 变成"内核通用 VM"

LWN 主编 Jonathan Corbet 在 **《BPF: the universal in-kernel virtual machine》**（LWN, 2014-05-21）里梳理了演进时间线：3.0 内核 Eric Dumazet 给 BPF 解释器加了 JIT；3.4 把 BPF 用进 seccomp 做 syscall 过滤；**3.15 内核语言正式分裂成 "classic BPF" 与 "internal BPF"**，后者把寄存器从 2 个扩到 10 个并改成 64 位。

真正的引擎重写是 Alexei Starovoitov（时在 PlumGrid，邮箱 `ast@plumgrid.com`）的 commit **`bd4cf0ed331a`（"net: filter: rework/optimize internal BPF interpreter's instruction set"，committer 是网络子系统 maintainer David S. Miller）**。这个 commit 干了什么、为什么干，commit message 本身讲得很清楚：

- 寄存器 **2 → 10** 个、字宽 **32 → 64 位**；栈空间从 16 个 4 字节 slot 扩到 **512 字节**。
- 跳转语义从"jt/jf 双目标"改成 **"jt / fall-through"**（条件成立跳，否则顺延）。
- 新增有符号比较（`JSGT/JSGE`）、`mov`、算术右移、字节序转换、原子操作、**函数调用**。

**为什么这样改（commit message 给的两条微架构理由，逐字大意）**：
1. *Fall-through jumps*：旧 BPF 跳转强制走 true/false 两个目标之一，制造分支预测失败惩罚；新跳转只有一个分支 + fall-through，"fits the CPU branch predictor logic better"。
2. *Jump-threaded 解释器 vs switch*：不再在 `switch` 顶部做单一 table-jump，而是让 gcc 生成多个 table-jump 指令，"helps CPU branch predictor logic"。

commit message 给的实测数字（x86_64，cache-hit）：

```
Filter #1:  old BPF 90ns  -> new BPF 31ns
Filter #2:  old BPF 192ns -> new BPF 47ns
seccomp 大 filter:  8.6s   -> 5.7s
```

**把 ABI 直接对齐硬件 calling convention**：r0 放返回值、r1–r5 传参、r6–r9 callee-saved、r10 只读栈帧指针——这套约定刻意照搬 x86_64/ARM64 的调用约定，目的就是让 JIT 能把 eBPF 寄存器**一对一映射**到物理寄存器，几乎不用做寄存器分配。这是"加载时证明完就能高效跑原生码"里"高效"二字的物理基础。LWN 引用原始 patch 说，eBPF 在某些网络过滤微基准上比 cBPF 快达 **4×**。

> **专家视角**：2014 这一步的真正意义不是"快了几纳秒"，而是把 BPF 从"socket 上的包过滤器"重新定位成"内核里任何子系统都能挂的通用 VM"。Corbet 的标题用词 "universal in-kernel virtual machine" 是精准的——后面十年的附着点扩张全是这次重定位的兑现。

### 23.2.3 syscall + maps：一个多路复用 syscall 撑起十年演进

光有 VM 不够，还要有"用户态怎么装载程序、内核态怎么存状态"。这由 commit **`99c55f7d47c0`（"bpf: introduce BPF syscall and maps"，Alexei Starovoitov，committer David S. Miller，2014）**奠定。两个设计决策值得专家级品味：

1. **`bpf()` 是一个多路复用器（multiplexor）**：第一版只支持 `BPF_MAP_CREATE`，但 syscall 形态是 `bpf(int cmd, union bpf_attr *attr, unsigned int size)`。十年里加了 `PROG_LOAD`、`MAP_LOOKUP_ELEM`、`LINK_CREATE`、`BTF_LOAD`……几十个 cmd，**没有新增一个 syscall 号**。
2. **`union bpf_attr` 故意为向后兼容设计**：用 union + 显式 `size` 参数，新命令往 union 里加字段、老用户态传更短的 `size`，内核按 size 判断哪些字段有效。这套"size 协商 + union 扩展"是 Linux syscall ABI 演进的经典手法（对比 `clone3`/`openat2` 的 `struct ... + size` 模式）。

map 的生命周期也定得很干净：`bpf(BPF_MAP_CREATE, ...)` 返回一个**进程局部的 fd**，关掉 fd 即销毁 map。map 由四要素定义：type、max_entries、key_size、value_size。

> **为什么不用现成的 syscall（如 ioctl）？** ioctl 的命令空间是各驱动私有的、类型不安全、长期被诟病。BPF 选了独立 syscall + 强类型 `union bpf_attr`，换来的是稳定、可审计、可被 strace/seccomp 识别的接口。这是"宁可加一个 syscall 也不滥用 ioctl"的正面案例。

### 23.2.4 verifier 与"有界循环"之争：从"禁掉一切危险"到"证明它会终止"

verifier 是 eBPF 安全模型的全部。它最初的策略极其保守：**任何循环（后向边 backward edge）一律拒绝**。理由很硬——不终止的程序就是内核内的 DoS。Matt Fleming 在 **《A thorough introduction to eBPF》**（LWN, 2017-12-02）把 verifier 总结为两步：第一步保证程序终止、无循环、无不可达指令；第二步逐指令模拟执行。

但"禁掉循环"逼得开发者用 `#pragma unroll` 把循环手工展开，或者像 John Fastabend 说的"做各种疯狂的事（crazy things）"来绕开。Jonathan Corbet 的 **《Bounded loops in BPF programs》**（LWN, 2018-12-03）记录了这场设计辩论的转折：与其禁，不如**证明终止**——

- 用**支配树（dominator tree）**分析识别出循环和它的归纳变量（induction variable）；
- 若能证明归纳变量**单调递增且有上界**，verifier 就能断定循环必然终止。

有界循环最终在内核 **5.3** 落地。这是 eBPF 安全哲学的一次根本升级：**从"黑名单一切可能危险的结构"转向"对结构做终止性证明"**。它直接催生了后来的 open-coded iterators（`bpf_iter_*`，源码里 `is_state_visited()` 专门为它写了收敛判定，见 23.3）。

### 23.2.5 CO-RE / BTF：解开可移植性死结

eBPF 程序常要读内核结构体字段（如 `task_struct->pid`）。死结在于：**同一字段的偏移量在不同内核版本/不同 `CONFIG` 下不一样**——字段会被挪位、被塞进新内嵌结构体。Andrii Nakryiko 的 **《BPF Portability and CO-RE》**（nakryiko.com, 2020-02-19）把前 CO-RE 时代讲得很透。

当时主流方案是 BCC：把 eBPF 的 C 源码作为字符串嵌进用户态程序，**在目标机上用 Clang/LLVM 现场编译**。三宗罪：
- Clang/LLVM 是个大库，产物"big fat binaries"；
- 启动时现场编译耗资源，可能"tip over a carefully balanced production workload"（把精心配平的生产负载压垮）；
- 目标机必须装内核头文件；且很多错误要到运行时才暴露。

CO-RE 用三层协作取代这一切：
1. **BTF（BPF Type Format）**：内核把自身所有类型信息编译进 vmlinux（`CONFIG_DEBUG_INFO_BTF=y`），通过 `/sys/kernel/btf/vmlinux` 暴露，可 `bpftool btf dump ... format c` 生成 `vmlinux.h`。
2. **编译器内建**：Clang 被扩展出若干 builtin，编译时不写死偏移，而是**发出 CO-RE 重定位记录**，描述"程序想读哪个类型的哪个字段"。
3. **libbpf 加载器**：在目标机上拿程序的重定位记录去比对目标内核的 BTF，**加载期改写偏移**，无需现场编译。

附带能力：**struct flavors**（`struct thread_struct___v46` 这种带 `___` 后缀的"风味"定义会被 libbpf 忽略后缀去匹配真实类型，用来处理字段改名）、`extern u32 CONFIG_HZ __kconfig;`（把内核构建参数当编译期常量，配合 verifier 死代码消除做特性裁剪）、`BPF_CORE_READ()`（可移植地链式解引用指针）。

> **专家视角**：CO-RE 的价值是"工程化"而非"性能"。没有它，每个 eBPF 工具都要维护一张"发行版 × 内核版本"矩阵，工具化无从谈起。到 2021，25+ 个 BCC 工具被改写成 libbpf+CO-RE 版本——这是 eBPF 真正落地为基础设施的标志。

### 23.2.6 附着点扩张的终点之争：sched_ext（计划 6.11，落地 6.12）

附着点一路扩张：kprobe/tracepoint（trace）→ XDP/TC/cgroup（网络）→ LSM（安全）→ struct_ops（替换内核函数表，如 TCP 拥塞控制）。最激进的一步是 **sched_ext**：用一组 BPF 程序实现**完整的 CPU 调度策略**，作为与 EEVDF/RT 并列的新调度类。

这一步在社区引爆了 eBPF 历史上最尖锐的设计辩论。Jonathan Corbet 的 **《The extensible scheduler class》**（LWN, 2023-02-10）记录了双方：

- 提出方 Tejun Heo / David Vernet / Josh Don / Barret Rhoden（Meta）的论点是：BPF 写调度策略**大幅降低实验门槛**，"without even needing to reboot the test machine"，且易于推到大规模机群。
- 调度器 maintainer **Peter Zijlstra 的态度极为直接**：

  > *"I hate all of this. Linus NAK'ed loadable schedulers a number of times in the past and this is just that again."*

  反对方的核心担忧是**碎片化**：厂商各自塞 out-of-tree 调度器，承认"一个通用调度器服务所有负载"的理想破产。
- 提出方反驳：自由实验反而会**加速 CFS/EEVDF 本身的改进**，而非分裂内核。

时间线本身就是这场拉锯的注脚：sched_ext 2022 年底上邮件列表，**一度计划并入 6.11（LWN 978007 标题即"to be merged for 6.11"），最终推迟到 6.12 才进 mainline**。Meta 在此之前已用它在生产跑定制调度器（如面向延迟敏感负载的 `scx_layered`）。

> **这一步为什么是哲学拐点**：调度器是内核最核心、最不容出错、Linus 历史上反复 NAK 过"可加载"化的子系统。它都开放给了可加载 BPF 策略，意味着"加载时证明 + 运行时零信任"这套模型，已经被内核社区（在激烈争论后）接受为可以触及最敏感子系统的通用机制。这是 "eBPF is eating the kernel" 这句话最硬的证据。

---

## 23.3 现状：核心架构与机制（源码级精读）

### 23.3.1 全局架构图

```
用户态                          内核态
┌─────────────────────────────────────────────────────────────────┐
│  bpftrace    bcc     libbpf-based tool                          │
│     │          │           │                                    │
│     └──────────┴─────────── bpf() syscall（多路复用） ───────────┐ │
└───────────────────────────────────────────────────────────────┼─┘
                                                                 │
                              ┌────────────────────────────────┐ │
                              │         BPF Subsystem          │ │
                              │  ┌──────────────────────────┐  │ │
                              │  │   Verifier（抽象解释）   │  │◄┘  do_check() 逐指令
                              │  │  tnum + 区间双数值域     │  │    模拟 + 状态去重剪枝
                              │  └────────────┬─────────────┘  │
                              │               │ 通过            │
                              │  ┌────────────▼─────────────┐  │
                              │  │  JIT（含 constant blind）│  │   bytecode → native
                              │  └────────────┬─────────────┘  │
                              │  ┌────────────▼─────────────┐  │
                              │  │   BPF Program（运行中）  │  │
                              │  └──────────────────────────┘  │
                              │  ┌──────────────────────────┐  │
                              │  │       BPF Maps           │  │   hash/array/ringbuf
                              │  │  （唯一合法持久化手段）  │  │   percpu/lru/sockhash
                              │  └──────────────────────────┘  │
                              └────────────────────────────────┘
                    ┌───────────────────────┼───────────────────┐
              ┌─────▼────┐          ┌───────▼──────┐    ┌──────▼──────┐
              │ kprobe   │          │  tracepoint  │    │     XDP     │
              │ uprobe   │          │ raw_tp/perf  │    │ TC/LSM/...  │
              └──────────┘          └──────────────┘    └─────────────┘
```

下面按"verifier → 解释器/JIT → map → helper → 附着点"逐个下钻，每个都配 v6.12 真实源码。

### 23.3.2 Verifier 精读（一）：它是一台抽象解释机

verifier 的本质是**抽象解释**：它不真正执行程序，而是在抽象域上"符号化地"模拟所有可达路径，对每条指令更新每个寄存器的*抽象值*，并在任何一步发现"可能越界/可能解引用空指针/可能不终止"时拒绝。

主循环就在 `do_check()` 里。下面是 v6.12 的真实片段（已删去与主线无关的分支，保留骨架）：

【真实源码 v6.12 · kernel/bpf/verifier.c · `do_check()` 主循环节选】
```c
static int do_check(struct bpf_verifier_env *env)
{
	...
	for (;;) {
		struct bpf_insn *insn;
		u8 class;
		int err;
		...
		insn = &insns[env->insn_idx];
		class = BPF_CLASS(insn->code);

		if (++env->insn_processed > BPF_COMPLEXITY_LIMIT_INSNS) {
			verbose(env,
				"BPF program is too large. Processed %d insn\n",
				env->insn_processed);
			return -E2BIG;
		}
		...
		if (is_prune_point(env, env->insn_idx)) {
			err = is_state_visited(env, env->insn_idx);
			if (err < 0)
				return err;
			if (err == 1) {
				/* found equivalent state, can prune the search */
				...
				goto process_bpf_exit;
			}
		}
		...
	}
}
```

逐行读：
- `for (;;)` + `env->insn_idx`：这是一台沿控制流图前进的**符号执行引擎**，不是一次线性扫描。遇到条件跳转时，verifier 会把"另一条分支"压栈（`push_stack()`），等当前路径走到 `bpf_exit` 再回溯——本质是对所有路径做 **DFS**。
- `++env->insn_processed > BPF_COMPLEXITY_LIMIT_INSNS`：每"模拟执行"一条指令计一次数。注意计的是**被模拟的指令次数（累加所有未被剪枝的路径）**，不是程序长度。超限直接 `-E2BIG`，错误串就是你常见的 *"BPF program is too large. Processed N insn"*。
- `is_prune_point → is_state_visited → err==1 → "safe" → prune`：这就是 verifier 能在指数级路径空间里活下来的命根子——**状态去重剪枝**，下一小节展开。

**两个限制常量，初学者必混（现在用真实定义钉死）：**

【真实源码 v6.12】
```c
/* include/uapi/linux/bpf_common.h */
#define BPF_MAXINSNS 4096

/* include/linux/bpf.h */
#define BPF_COMPLEXITY_LIMIT_INSNS      1000000 /* yes. 1M insns */

/* kernel/bpf/verifier.c */
#define BPF_COMPLEXITY_LIMIT_JMP_SEQ	8192
#define BPF_COMPLEXITY_LIMIT_STATES	64
```

- `BPF_MAXINSNS = 4096`：**程序长度**上限。早期对所有程序生效；如今对**非特权程序仍是硬上限**——常被误传为"4096 已废除"，其实只对 root/`CAP_BPF` 程序放宽了。
- `BPF_COMPLEXITY_LIMIT_INSNS = 1,000,000`（注释直接写 "yes. 1M insns"）：**verifier 放弃前最多模拟多少条指令**。特权程序的"实际可加载规模"由它兜底，而非 4096。
- `JMP_SEQ=8192 / STATES=64`：DFS 栈深与单点保存状态数的上限，防状态爆炸。

```
💡坑点：程序"才几百行"却被拒，报 "too many instructions" / "too large"，
多半不是行多，而是分支组合爆炸——verifier 模拟的路径数 × 每路径指令数
撞上了 1M 复杂度墙。解法是减分支 / 帮 verifier 剪枝（缩小变量取值范围、
给 verifier 喂 if 收窄边界），而不是删代码行。
```

### 23.3.3 Verifier 精读（二）：状态去重为什么是命根子

路径数对分支数是**指数级**（N 个独立条件跳转 → 最坏 2^N 条路径）。不剪枝的话，稍复杂的程序就会瞬间顶满 1M 复杂度墙。`is_state_visited()` 的思路是：**如果当前到达某指令时的寄存器/栈状态，"等价于"之前到达同一指令时的某个已验证安全的状态，那这条路径就不用再往下走了**。

【真实源码 v6.12 · kernel/bpf/verifier.c · `is_state_visited()` 节选】
```c
static int is_state_visited(struct bpf_verifier_env *env, int insn_idx)
{
	...
	/* bpf progs typically have pruning point every 4 instructions
	 * http://vger.kernel.org/bpfconf2019.html#session-1
	 * Do not add new state for future pruning if the verifier hasn't seen
	 * at least 2 jumps and at least 8 instructions.
	 * This heuristics helps decrease 'total_states' and 'peak_states' metric.
	 * In tests that amounts to up to 50% reduction into total verifier
	 * memory consumption and 20% verifier time speedup.
	 */
	add_new_state = force_new_state;
	if (env->jmps_processed - env->prev_jmps_processed >= 2 &&
	    env->insn_processed - env->prev_insn_processed >= 8)
		add_new_state = true;
	...
	while (sl) {
		states_cnt++;
		if (sl->state.insn_idx != insn_idx)
			goto next;
		...
	}
}
```

逐行读：
- 注释里的工程数据极有信息量：BPF 程序大约**每 4 条指令**就有一个剪枝点；但内核做了启发式——**至少见过 2 次跳转且 8 条指令**才存一个新检查点。这一招把 `total_states` 降了多达 50%、verifier 时间快了 20%。这是典型的"不是所有点都值得记忆化，记忆化本身有成本"的权衡。
- `while (sl)` 遍历同一 `insn_idx` 上已保存的状态链表，用 `states_equal()` 判等价。判等价不是逐位相等：对 `SCALAR_VALUE`，只要"当前状态的取值范围 ⊆ 已存状态的范围"且活跃性（liveness）标记兼容，就算等价可剪。

> 🎯 **面试加分点**：verifier 是"模拟执行所有可能路径"的指数级问题，靠"状态去重 + 区间包含判等价"剪枝活下来。问到"verifier 为什么会 state explosion"，标准答案是：**指针跟踪让大量寄存器变成相互关联的非常量，等价判定退化、剪不掉，路径数逼近 2^分支数**。

### 23.3.4 Verifier 精读（三）：双数值域——tnum + 区间，为什么是两套

verifier 对每个寄存器维护的抽象值，核心是两套并行的数值域。先看抽象域本身：

【真实源码 v6.12 · include/linux/tnum.h】
```c
/* tnum: tracked (or tristate) numbers
 *
 * A tnum tracks knowledge about the bits of a value.  Each bit can be either
 * known (0 or 1), or unknown (x).  Arithmetic operations on tnums will
 * propagate the unknown bits such that the tnum result represents all the
 * possible results for possible values of the operands.
 */
struct tnum {
	u64 value;
	u64 mask;
};
```

`tnum` 是**位级 tristate**：`mask` 里为 1 的位表示"未知"，`value` 给出已知位的取值。例如 `value=0b1000, mask=0b0011` 表示 `1 0 x x`（高位已知 10、低两位未知）。位运算（AND/OR/XOR/移位）在 tnum 上能精确传播——这正是它存在的理由。

再看寄存器状态结构体里与"安全"直接相关的字段：

【真实源码 v6.12 · include/linux/bpf_verifier.h · `struct bpf_reg_state` 节选】
```c
struct bpf_reg_state {
	/* Ordering of fields matters.  See states_equal() */
	enum bpf_reg_type type;
	s32 off;
	union { ... };               /* map_ptr / btf_id / mem_size / ... 按 type 区分 */
	struct tnum var_off;         /* 位级已知/未知 */
	/* Used to determine if any memory access using this register will
	 * result in a bad access.
	 */
	s64 smin_value; /* minimum possible (s64)value */
	s64 smax_value; /* maximum possible (s64)value */
	u64 umin_value; /* minimum possible (u64)value */
	u64 umax_value; /* maximum possible (u64)value */
	s32 s32_min_value; /* minimum possible (s32)value */
	s32 s32_max_value; /* maximum possible (s32)value */
	u32 u32_min_value; /* minimum possible (u32)value */
	u32 u32_max_value; /* maximum possible (u32)value */
	...
#define BPF_ADD_CONST (1U << 31)
	u32 id;
	...
};
```

这里藏着 verifier 设计最精妙、也最容易出 CVE 的地方：**两套数值域并存**。
- **区间域**：`smin/smax`（有符号）、`umin/umax`（无符号），外加 **32 位子寄存器版本** `s32_min/.../u32_max`。
- **位级域**：`var_off`（tnum）。

**为什么要两套？** 因为它们各自擅长不同操作，互相补盲：
- **比较跳转**（`<`, `>`, `<=`）天然更新**区间**：`if (r1 < 64)` 走 true 分支后 `umax_value=63`。
- **位运算**（AND/OR/移位）天然更新 **tnum**：`r1 &= 0x3f` 后 tnum 立刻知道高位全 0。
- 两者**互相收窄**（`__reg_deduce_bounds()`）：tnum 算出"高位全 0"能反推 `umax ≤ 0x3f`；区间算出 `umax < 256` 能反推 tnum 高位为 0。**32 位与 64 位之间也要互相同步**——这步同步历史上出过致命 bug（见 23.7 的 CVE-2021-3490，根因正是 ALU32 位运算后没正确更新这几个 `s32/u32` 字段）。

**为什么不直接上 SMT solver / 完整符号执行？** 这是一个被反复质疑、但内核坚持的工程权衡：

| 维度 | verifier 的"区间+tnum"抽象域 | 完整符号执行 / SMT solver |
|------|------------------------------|---------------------------|
| 时间复杂度 | 近线性（每指令 O(1) 更新 + 剪枝），毫秒级 | 最坏指数，可能秒级甚至不可判定 |
| 运行环境 | 在 `bpf()` syscall 上下文、可能持锁、要快速返回 | 不可能塞进 syscall 热路径 |
| 正确性可审计性 | 抽象转移函数是几百行可人工 review 的 C | solver 是巨型第三方依赖，自身可能有 bug |
| 代价 | **会误杀**（保守，安全但有时拒绝合法程序） | 更精确但更慢、更难证明自身可靠 |

> **底层根因**：verifier 的正确性是整个 eBPF 安全模型的根基，所以它必须**自身足够简单到能被人审计**，并且**宁可保守误杀也不能漏判**（unsound 即提权）。这就排除了"完备但庞大不可审计"的 SMT 路线。这也解释了为什么 eBPF 开发体验里"明明逻辑没错却被拒"如此常见——那是抽象域保守性的必然代价，不是 bug。学术界（如 PREVAIL 抽象解释验证器、Agni 用 SMT 离线验证 verifier 自身的转移函数）走的是"用形式化方法验证 verifier 正确性"的路，而不是把 SMT 塞进运行时。

### 23.3.5 Verifier 精读（四）：类型格、空指针收窄、资源泄漏

verifier 给每个寄存器打**类型标签**，构成一个类型格（lattice）：

【真实源码 v6.12 · include/linux/bpf.h · `enum bpf_reg_type` 节选】
```c
enum bpf_reg_type {
	NOT_INIT = 0,		 /* nothing was written into register */
	SCALAR_VALUE,		 /* reg doesn't contain a valid pointer */
	PTR_TO_CTX,		 /* reg points to bpf_context */
	CONST_PTR_TO_MAP,	 /* reg points to struct bpf_map */
	PTR_TO_MAP_VALUE,	 /* reg points to map element value */
	PTR_TO_MAP_KEY,		 /* reg points to a map element key */
	PTR_TO_STACK,		 /* reg == frame_pointer + offset */
	PTR_TO_PACKET_META,	 /* skb->data - meta_len */
	PTR_TO_PACKET,		 /* reg points to skb->data */
	PTR_TO_PACKET_END,	 /* skb->data + headlen */
	...
	PTR_TO_BTF_ID,
	...
	__BPF_REG_TYPE_MAX,
	/* Extended reg_types. */
	PTR_TO_MAP_VALUE_OR_NULL	= PTR_MAYBE_NULL | PTR_TO_MAP_VALUE,
	PTR_TO_SOCKET_OR_NULL		= PTR_MAYBE_NULL | PTR_TO_SOCKET,
	...
};
```

两个设计点：
- `..._OR_NULL = PTR_MAYBE_NULL | PTR_TO_...`：**"可能为空"是用一个 flag 位叠加到基础类型上**，不是单独的枚举。`bpf_map_lookup_elem()` 返回的就是 `PTR_TO_MAP_VALUE_OR_NULL`。verifier 拒绝在它被 `if (ptr)` **收窄**成非空之前解引用——一旦走过 `if (!ptr) return;`，fall-through 分支上 verifier 把类型从 `..._OR_NULL` 清掉 `PTR_MAYBE_NULL` 位，变成 `PTR_TO_MAP_VALUE`，解引用即合法。这就是经典三行 `val = lookup(); if (!val) return 0; *val += 1;` 背后的类型收窄。
- `PTR_TO_PACKET` / `PTR_TO_PACKET_END`：XDP/TC 的包边界检查靠这两类指针的**关系**追踪。`if (data + off + 1 > data_end)` 不是普通比较，而是让 verifier 在 fall-through 分支上把 `data` 的可访问 `range` 收窄到合法区间。

**资源泄漏检测**是 verifier 的另一项硬保证，而且源码注释把它讲得比任何教程都清楚：

【真实源码 v6.12 · include/linux/bpf_verifier.h · `struct bpf_reg_state` 的 `ref_obj_id` 注释】
```c
	/* Consider the following where "sk" is a reference counted
	 * pointer returned from "sk = bpf_sk_lookup_tcp();":
	 *
	 * 1: sk = bpf_sk_lookup_tcp();
	 * 2: if (!sk) { return 0; }
	 * 3: fullsock = bpf_sk_fullsock(sk);
	 * 4: if (!fullsock) { bpf_sk_release(sk); return 0; }
	 * 5: tp = bpf_tcp_sock(fullsock);
	 * 6: if (!tp) { bpf_sk_release(sk); return 0; }
	 * 7: bpf_sk_release(sk);
	 * 8: snd_cwnd = tp->snd_cwnd;  // verifier will complain
	 */
```

逐行读：`bpf_sk_lookup_tcp()` 返回一个**引用计数指针**，verifier 给它分配一个 `ref_obj_id`。程序退出前必须恰好 `bpf_sk_release()` 一次，否则 reject（泄漏）。更妙的是第 7 行 release 之后，所有从 `sk` 派生出来的指针（`fullsock`、`tp`）都被 verifier 标记失效——第 8 行再用 `tp` 就报错（use-after-release）。**这是 verifier 在做轻量级的所有权/借用分析**，思想和 Rust 借用检查同源，只是发生在字节码层。

### 23.3.6 解释器与 JIT：为什么用 computed-goto，为什么要 constant blinding

通过 verifier 后，程序要么被解释执行（`CONFIG_BPF_JIT_ALWAYS_ON` 未开时的回退路径），要么被 JIT 成原生码。先看解释器的核心——一个 **computed-goto（jump-threaded）派发**：

【真实源码 v6.12 · kernel/bpf/core.c · `___bpf_prog_run()` 节选】
```c
static u64 ___bpf_prog_run(u64 *regs, const struct bpf_insn *insn)
{
#define BPF_INSN_2_LBL(x, y)    [BPF_##x | BPF_##y] = &&x##_##y
#define BPF_INSN_3_LBL(x, y, z) [BPF_##x | BPF_##y | BPF_##z] = &&x##_##y##_##z
	static const void * const jumptable[256] __annotate_jump_table = {
		[0 ... 255] = &&default_label,
		BPF_INSN_MAP(BPF_INSN_2_LBL, BPF_INSN_3_LBL),
		...
	};
#define CONT	 ({ insn++; goto select_insn; })
select_insn:
	goto *jumptable[insn->code];
	...
	ALU64_ADD_X:  DST = DST + SRC;  CONT;   /* 由 ALU(ADD, +) 宏展开 */
	...
```

逐行读：
- `jumptable[256]` 是 **GCC 的 "labels as values" 扩展**（`&&label` 取标签地址）。`goto *jumptable[insn->code]` 直接跳到对应 opcode 的处理块。
- 关键在 `CONT` 宏：每个 opcode 处理块**结尾都各自重新 `goto *jumptable[...]`**，而不是回到一个公共的 `switch` 顶部。
- **微架构根因（这是 2014 commit 选 computed-goto 的真正理由）**：`switch` 派发只有**一个**间接跳转点，CPU 的 BTB（Branch Target Buffer）在这一个点上要预测约 100 种目标，命中率极低。computed-goto 把派发点**复制到每个 opcode 块尾**，于是每个派发点只需预测"这条 opcode 的下一条通常是什么"——而真实字节码里 opcode 是有强相关性的（LD 后常跟 ALU，ALU 后常跟 JMP）。BTB 能学到这种 **opcode bigram**，预测命中率大幅提升。这就是 commit message 里 "helps CPU branch predictor logic" 的硬核含义，也是解释型 VM 的经典优化（Ertl & Gregg 的 threaded code 研究）。

还有一个容易被忽略的安全/正确性细节——移位量掩码：

【真实源码 v6.12 · kernel/bpf/core.c · 移位处理上方注释 + SHT 宏】
```c
	/* Explicitly mask the register-based shift amounts with 63 or 31
	 * to avoid undefined behavior. ...
	 * In case of JIT backends, the AND must /not/ be added to the emitted
	 * LSH/RSH/ARSH translation.
	 */
#define SHT(OPCODE, OP)					\
	ALU64_##OPCODE##_X:				\
		DST = DST OP (SRC & 63);		\
		CONT;					\
	ALU_##OPCODE##_X:				\
		DST = (u32) DST OP ((u32) SRC & 31);	\
		CONT;					\
	...
```

逐行读：C 里移位量 ≥ 位宽是 **undefined behavior**。解释器显式 `& 63`（64 位）/`& 31`（32 位）保证行为已定义。注释特别强调 **JIT 后端不要加这个 AND**——因为目标硬件指令本身对超量移位有自己的（实现定义的）行为，整体程序行为仍是 defined，再加 AND 反而多余。这是"同一语义在解释器和 JIT 两条路径上要分别保证"的典型细节。

**JIT 的安全成本：constant blinding（常量盲化）**。开 JIT 不等于没安全成本——JIT 把字节码变成可执行原生码，攻击者可控的**立即数**有可能被当作 ROP/JIT-spraying 的 gadget。缓解手段是常量盲化：

【真实源码 v6.12 · kernel/bpf/core.c · `bpf_jit_blind_insn()` 节选】
```c
static int bpf_jit_blind_insn(const struct bpf_insn *from, ...)
{
	...
	u32 imm_rnd = get_random_u32();
	...
	case BPF_ALU64 | BPF_ADD | BPF_K:
	...
		*to++ = BPF_ALU64_IMM(BPF_MOV, BPF_REG_AX, imm_rnd ^ from->imm);
		*to++ = BPF_ALU64_IMM(BPF_XOR, BPF_REG_AX, imm_rnd);
		*to++ = BPF_ALU64_REG_OFF(from->code, from->dst_reg, BPF_REG_AX, from->off);
		break;
	...
}
```

逐行读：原本一条"用立即数 K 的指令"，被改写成三条——先把 `K ^ imm_rnd`（一个随机化后的值）装进辅助寄存器 AX，运行时再 `XOR imm_rnd` 还原出 K，最后用 AX 完成原操作。**于是 JITed 机器码流里永远不出现攻击者选定的常量 K**，JIT spraying 无法靠喂特定常量来"种"gadget。这由 `bpf_jit_harden` 开关控制：`1`=仅非特权用户盲化，`2`=所有用户盲化。配套的还有 JIT 代码**起始地址随机化**（`get_random_u32_below(hole)`）和**代码页只读**。

> ⚠️ **纠正一个高频误解**：`bpf_jit_enable`（用 JIT 还是解释器）和 `unprivileged_bpf_disabled`（谁能加载 eBPF）是**两个正交维度**，别混。和 Spectre 直接相关的硬化开关是 `bpf_jit_harden`，不是 `bpf_jit_enable`。生产上若开 JIT，通常配 `bpf_jit_harden=2`。

### 23.3.7 BPF Map：状态、跨域通信，与 percpu 的缓存行根因

map 是 eBPF 唯一合法的持久化与跨域通信手段。v6.12 的 map 类型全集（节选）：

【真实源码 v6.12 · include/uapi/linux/bpf.h · `enum bpf_map_type` 节选】
```c
enum bpf_map_type {
	BPF_MAP_TYPE_UNSPEC,
	BPF_MAP_TYPE_HASH,
	BPF_MAP_TYPE_ARRAY,
	BPF_MAP_TYPE_PROG_ARRAY,        /* tail call 跳转表 */
	BPF_MAP_TYPE_PERF_EVENT_ARRAY,
	BPF_MAP_TYPE_PERCPU_HASH,
	BPF_MAP_TYPE_PERCPU_ARRAY,
	BPF_MAP_TYPE_LRU_HASH,          /* 连接跟踪，自动驱逐 */
	...
	BPF_MAP_TYPE_SOCKHASH,
	BPF_MAP_TYPE_RINGBUF,           /* 5.8+，替代 perf array */
	...
	BPF_MAP_TYPE_ARENA,             /* 较新，用户态/内核态共享线性地址空间 */
	__MAX_BPF_MAP_TYPE
};
```

| Map 类型 | 典型用途 | 关键特性 / 根因 |
|---------|---------|---------|
| `HASH` | 任意 KV | O(1) 均摊，有碰撞，更新走内部锁 |
| `ARRAY` | 索引访问 | 预分配，下标固定，不能删元素 |
| `PERCPU_HASH/ARRAY` | 高频计数器 | 每 CPU 独立副本，**无锁**，用户态聚合时 sum |
| `LRU_HASH` | 连接跟踪 | 满了自动驱逐 LRU，防 FD/状态泄漏 |
| `RINGBUF` | 高性能事件上报 | MPSC 无锁 ringbuffer，单 fd poll |
| `PROG_ARRAY` | tail call | 跳到其他 BPF 程序，绕过单程序指令上限 |
| `SOCKHASH/SOCKMAP` | socket 重定向 | sockmap 数据面，Cilium 用它做 L4 LB |

**percpu map 为什么快（缓存行根因）**：普通 map 上多核并发更新同一计数器，需要原子操作（`LOCK` 前缀），更糟的是多核反复写同一 cache line 会触发 **cache-line ping-pong**（MESI 协议下该行在各核 L1 间反复 invalidate/transfer），延迟从几个周期飙到几十上百周期。percpu map 给每个 CPU 一份独立副本、各写各的、**零原子、零跨核失效**，用户态读时再把 N 份 sum 起来。代价是内存 × CPU 数。

```
💡坑点：PERCPU map 用户态读取拿到的是"所有 CPU 副本的数组"，必须自己 sum。
很多人直接取 [0] 号，结果只看到 1/N 的数据。
而在 eBPF 程序内，bpf_map_lookup_elem() 默认就返回"本 CPU"那一份（不跨 CPU），
所以程序里无需指定 CPU；要读别的 CPU 那份才用 bpf_map_lookup_percpu_elem()（5.18+）。
```

### 23.3.8 Helper 与 kfunc：内核服务的安全导出

eBPF 程序不能直接调 `kmalloc` 等任意内核函数（verifier 无法审计副作用），只能调内核维护的 **helper 白名单**。每种程序类型有自己的 helper 子集。高频 helper（分类，示意）：

```
Map：   bpf_map_lookup_elem / update_elem / delete_elem
读内存： bpf_probe_read_kernel / bpf_probe_read_user（带容错的安全读）
身份：   bpf_get_current_pid_tgid / bpf_get_current_comm / bpf_ktime_get_ns
上报：   bpf_perf_event_output（旧）/ bpf_ringbuf_reserve + submit（新，5.8+）
包：     bpf_xdp_adjust_head/tail / bpf_redirect / bpf_redirect_map
栈：     bpf_get_stackid（采样调用栈）
任务：   bpf_get_current_task（裸指针，需 probe_read）/ bpf_get_current_task_btf（5.11+，带 BTF 可直接解引用）
```

> 🎯 **面试加分点**：问"eBPF 能直接调 `kmalloc` 吗"——不能。这不只是安全，更是 **verifier 可分析性**的前提：helper 的参数类型、是否可能睡眠、副作用都预先声明给了 verifier。**现代趋势是 kfunc**：相比"固定 helper 白名单 + 稳定 UAPI 号"，kfunc 允许内核子系统更灵活地把自己的函数（带 BTF 类型）暴露给特定程序类型，无需走 UAPI 稳定承诺。sched_ext、很多新特性都靠 kfunc 而非 helper。

### 23.3.9 附着点全景与 prog_type 的"稳定性"真相

程序类型决定可用 helper、可访问的 ctx、以及附着语义。v6.12 程序类型（节选）：

【真实源码 v6.12 · include/uapi/linux/bpf.h · `enum bpf_prog_type` 与稳定性注释】
```c
/* Note that tracing related programs such as
 * BPF_PROG_TYPE_{KPROBE,TRACEPOINT,PERF_EVENT,RAW_TRACEPOINT}
 * are not subject to a stable API since kernel internal data
 * structures can change from release to release and may
 * therefore break existing tracing BPF programs. Tracing BPF
 * programs correspond to /a/ specific kernel which is to be
 * analyzed, and not /a/ specific kernel /and/ all future ones.
 */
enum bpf_prog_type {
	BPF_PROG_TYPE_UNSPEC,
	BPF_PROG_TYPE_SOCKET_FILTER,
	BPF_PROG_TYPE_KPROBE,
	BPF_PROG_TYPE_SCHED_CLS,   /* TC */
	BPF_PROG_TYPE_XDP,
	...
	BPF_PROG_TYPE_TRACING,     /* fentry/fexit/iter */
	BPF_PROG_TYPE_STRUCT_OPS,  /* 替换内核函数表，如 TCP CC、sched_ext */
	BPF_PROG_TYPE_LSM,
	...
};
```

这段内核自带注释直接回答了"附着点稳定性"这个高频考点：**tracing 类程序（kprobe/tracepoint/...）显式不承诺稳定 API**——它们针对的是"某个具体内核"，不是"某内核 + 所有未来内核"。这就是为什么生产工具优先用 CO-RE + 稳定 tracepoint，而非裸 kprobe。

**kprobe vs tracepoint vs fentry/fexit 对比**：

| 维度 | kprobe | tracepoint | fentry/fexit（5.5+） |
|------|--------|-----------|-------------|
| 稳定性 | 低（函数可内联/改名/消失） | 高（`TRACE_EVENT` ABI 保证） | 中（需 BTF，函数存在即可） |
| 参数访问 | 手解寄存器（按调用约定） | 类型安全的参数结构体 | BTF 提供类型 |
| 开销 | 中（`int3` 陷入或 ftrace） | 中 | **低**（ftrace nop + trampoline，近函数直调） |
| 覆盖 | 几乎所有函数 | 仅预定义点 | 有 BTF 信息的函数 |

**为什么 fentry 比 kprobe 快（底层根因）**：传统 kprobe 在指令上打 `int3` 断点，触发时走**异常路径**（陷入、保存全部寄存器、查 kprobe 哈希表）。fentry 利用编译器在每个函数开头预留的 ftrace nop（`-pg`/`__fentry__`），把 nop 直接 patch 成跳到 **BPF trampoline** 的调用——没有异常、没有断点、只保存需要的寄存器，接近一次普通函数调用的开销。

#### 方案对比·跑一遍：给 `tcp_sendmsg` 加延迟探针，选 kprobe / tracepoint / fentry？

具体场景：你要测 `tcp_sendmsg` 的执行延迟，按进程聚合 p99。带着推理走一遍：

1. **先找 tracepoint**：`tcp_sendmsg` 上有稳定 tracepoint 吗？没有（内核只在收发关键点埋了少量 `tcp:` tracepoint，覆盖不到任意函数）。→ tracepoint 出局，**不适用边界**：tracepoint 只能用预定义点。
2. **fentry/fexit**：`tcp_sendmsg` 是非内联的导出函数、内核开了 `CONFIG_DEBUG_INFO_BTF`，所以 fentry 能挂、且能用 `fexit` 直接拿返回值与入参（类型安全）。开销最低。→ **首选 fentry（进入）/ fexit（返回计时）**。
3. **fentry 不可用时退回 kprobe**：如果目标内核较老（< 5.5）或没 BTF，只能 kprobe `tcp_sendmsg` 入口 + kretprobe 出口配对计时。风险：若 `tcp_sendmsg` 在某编译配置下被内联进调用者，kprobe 直接挂不上——这正是 kprobe"不稳定"的现实表现。
4. **结论**：稳定性 > 性能 > 覆盖。有 BTF 的现代内核优先 fentry/fexit；要跨老内核、追稳定语义的网络事件才考虑稀有的 tracepoint；kprobe 是"没别的办法时"的万能兜底，但要接受内联/改名导致探针失效的脆性。

---

## 23.4 工具链：bpftrace / bcc / libbpf + CO-RE

### 23.4.1 三层定位

```
易用性 ▲   bpftrace  ── 单行脚本，一次性排障首选
       │   bcc       ── Python 前端 + C eBPF，原型快；生产不推荐（背 LLVM、启动现编译）
       │   libbpf    ── C API + CO-RE，生产工具标准方案
性能/控制▼  bpf() syscall ── 直接操作，工具作者用
```

### 23.4.2 bpftrace 排障单行（可直接运行）

【示意代码，非逐字 · bpftrace 脚本】
```bash
# 每进程 read 次数
bpftrace -e 'tracepoint:syscalls:sys_enter_read { @[comm] = count(); }'

# read 延迟分布直方图（kprobe/kretprobe 配对计时）
bpftrace -e '
kprobe:vfs_read { @start[tid] = nsecs; }
kretprobe:vfs_read /@start[tid]/ {
  @us = hist((nsecs - @start[tid]) / 1000);
  delete(@start[tid]);
}'

# TCP 重传按进程聚合
bpftrace -e 'kprobe:tcp_retransmit_skb { @[comm, pid] = count(); }'

# CPU 采样火焰图原料（99Hz）
bpftrace -e 'profile:hz:99 { @[kstack, ustack, comm] = count(); }'
```

> 常用内置变量：`pid/comm/tid/nsecs/retval`（kretprobe 返回值）、`arg0..argN`（kprobe 按调用约定取参）、`args`（tracepoint 类型化参数）、`curtask`（当前 `task_struct`，配 BTF 可写 `curtask->...`）、`kstack/ustack`。

### 23.4.3 CO-RE 在加载期到底做了什么（落地视角）

把 23.2.5 的历史落到操作：

【示意代码，非逐字 · CO-RE 工作流】
```bash
# 1. 目标机/开发机：确认 BTF 存在
ls /sys/kernel/btf/vmlinux

# 2. 开发机：从运行内核生成全类型头
bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# 3. 编译：clang 不写死偏移，发出 CO-RE 重定位
clang -O2 -g -target bpf -c prog.bpf.c -o prog.bpf.o
# prog.bpf.o 里：eBPF 字节码 + BTF + CO-RE 重定位记录

# 4. 目标机加载：libbpf 读 prog.bpf.o 的重定位，
#    比对 /sys/kernel/btf/vmlinux，把 BPF_CORE_READ(task, pid) 里的
#    "pid 字段偏移"按目标内核实际布局改写，再 BPF_PROG_LOAD
```

`BPF_CORE_READ(task, pid)` 不硬编码偏移，而是生成"读 `task_struct` 的 `pid` 字段"这条**语义级**重定位，libbpf 加载时用目标 BTF 填正确偏移。这就是"一次编译、到处运行"的机制本体。

### 23.4.4 libbpf skeleton 程序骨架

【示意代码，非逐字 · libbpf skeleton】
```c
struct example_bpf *skel = example_bpf__open();   // 解析 .o，建对象
example_bpf__load(skel);                           // CO-RE 重定位 + verifier + JIT
example_bpf__attach(skel);                          // attach 到探针，返回 bpf_link
// ... poll ring buffer / 读 map ...
example_bpf__destroy(skel);                         // 清理
```

---

## 23.5 三大应用域

### 23.5.1 可观测性

eBPF 把可观测从"需要预埋日志"变成"事后任意插桩"。生产高频：CPU profiling（`profile:hz:99` 采 kstack/ustack 出火焰图）、延迟分析（kprobe+kretprobe 配对 + `hist()`）、调度延迟（`sched:sched_wakeup` + `sched_switch` 量 runqueue wait）、网络（TCP 建连延迟/重传/每连接 RTT）、应用层（uprobe 挂 OpenSSL 在加密前抓明文、Go runtime GC pause）。知名工具：bpftrace、BCC tools（`biolatency`/`tcpconnect`）、Cilium Tetragon、Pixie、Parca。

### 23.5.2 网络：XDP 为什么快，以及它的不适用边界

**XDP（eXpress Data Path）** 在网卡驱动收包、DMA 完成后、**分配 `sk_buff` 之前**执行：

```
包到达 → [网卡驱动] → XDP hook（eBPF 在此）
                        ├ XDP_DROP    : 直接丢，连 skb 都不建
                        ├ XDP_TX      : 原路返回（DDoS 反打）
                        ├ XDP_REDIRECT: 转发到别的网卡 / AF_XDP socket
                        └ XDP_PASS    : 继续正常协议栈
```

**为什么快（根因）**：常规协议栈每个包要先分配 `sk_buff`（几百字节的元数据 + 关联内存）、做大量簿记。`XDP_DROP` 在 skb 分配前就决断，**省掉了整条路径最贵的内存分配与簿记**。性能量级（强依赖网卡/CPU/内核，是数量级不是保证值）：XDP 原始论文（Høiland-Jørgensen 等, 2018）`XDP_DROP` 单核约 24 Mpps、比常规栈最快路径快约 5×；Cloudflare 工程博客报告单核 `XDP_DROP` 可达 10 Mpps 量级。作标尺：100GbE 满 64B 小包线速约 1.48 亿 pps——单核 24 Mpps 也只是零头，所以高 pps 防护要么多核 RSS 扩展、要么下沉智能网卡 offload。

#### 方案对比·跑一遍：DDoS 丢包该用 XDP / TC BPF / DPDK？

- **要不要读 L4+ 并改包**：纯按源 IP 黑名单丢 → XDP 足够且最快（skb 都不建）。要按连接状态/conntrack 做有状态决策、或要改 egress → 必须 **TC BPF**（在 tc 层，已有 `sk_buff`，能读写完整元数据、支持 ingress+egress），代价是慢一档。
- **要不要彻底绕开内核**：若追求绝对极限 pps 且愿意把整张网卡交给用户态 → **DPDK**（kernel bypass，轮询、巨页、用户态驱动）。但 DPDK 的代价是：该网卡对内核不可见、要独占 CPU 轮询、要重写整套协议栈、运维复杂。XDP 的杀手锏是**留在内核里**——网卡仍能正常被系统使用，`XDP_PASS` 的包照走协议栈，只对你关心的包做快速决断。
- **不适用边界**：XDP 动作有限（只 ingress、4 个动作），要做复杂 L7 逻辑它不合适；DPDK 性能最高但生态侵入最大；TC BPF 是"能力/性能"的中间档。结论：DDoS 纯丢包 → XDP；有状态/egress → TC BPF;  独占式极限转发 → DPDK。

**Cilium** 用 eBPF（sockmap + BPF map O(1) 查找）替代 kube-proxy 的 iptables 线性扫描，把 K8s Service 负载均衡从"十万 Service 时延迟显著"变成"规模线性稳定"。

### 23.5.3 安全

- **BPF LSM（5.7+）**：把 BPF 程序挂到 LSM 钩子（如 `lsm/bprm_check_security`），返回非 0 即拒绝操作——可编程的强制访问控制。
- **seccomp-BPF（3.5）**：注意它用的是 **cBPF 不是 eBPF**，在 syscall 入口按系统调用号过滤，是 Docker/containerd 沙箱基础。
- **Falco**：用 eBPF（或内核模块）做运行时安全规则引擎，检测"容器内起 shell""敏感文件访问"等。

---

## 23.6 落地：生产排障

### 23.6.1 bpftool 工具箱

【示意代码，非逐字】
```bash
bpftool prog list                       # 已加载程序
bpftool map list / map dump id <ID>     # map 内容
bpftool prog dump xlated id <ID>        # 看 verifier 处理后的 eBPF 指令
bpftool prog dump jited  id <ID>        # 看 JIT 后的原生机器码
bpftool btf list / btf dump id <ID>     # BTF 类型
bpftool link list                       # bpf_link 附着关系
```

### 23.6.2 读懂 verifier 拒绝日志（回报最高的技能）

verifier reject 时吐的状态日志看似天书，结构其实固定。一个"map value 未做 NULL 检查就解引用"的典型片段（示意，不同内核字段略有出入）：

```
; val = bpf_map_lookup_elem(&m, &key);
3: (85) call bpf_map_lookup_elem#1
; *val += 1;
4: (79) r1 = *(u64 *)(r0 +0)
R0 invalid mem access 'map_value_or_null'
```

读法（结合 23.3.5 的类型格）：
- 行首 `4:` 是**字节码偏移**（不是 C 源码行号；前面的 `;` 注释行会回贴源码，开 BTF/`-g` 后更准）。
- `R0 ... 'map_value_or_null'`：r0 此刻类型是 `PTR_TO_MAP_VALUE_OR_NULL`（带 `PTR_MAYBE_NULL` 位），verifier 拒绝在它被 `if (r0)` 收窄成非空前解引用。
- 修复就是经典三行：`val = lookup(...); if (!val) return 0; *val += 1;`。走过 `if (!val) return` 后，fall-through 分支上 r0 被清掉 `PTR_MAYBE_NULL`、收窄为 `PTR_TO_MAP_VALUE`，解引用合法。

**排障套路**：`BPF_PROG_LOAD` 把 `log_level` 调到 `2`（libbpf 里逐指令最详尽），然后**从报错那一行往上回溯**对应寄存器的类型/区间是怎么变成当前状态的。几乎所有 reject 都能这样定位——因为 verifier 日志本质是 `do_check()` 抽象解释过程的逐步快照。

### 23.6.3 P2 项目：用 bpftrace 追一个真实 p99 抖动

**场景**：某服务 p99 偶发升高，疑似 VFS read 路径上某函数慢。

【示意代码，非逐字 · bpftrace 排障四步】
```bash
# Step1 看 VFS 层延迟分布
bpftrace -e '
kprobe:vfs_read { @start[tid] = nsecs; }
kretprobe:vfs_read /@start[tid]/ { @us = hist((nsecs-@start[tid])/1000); delete(@start[tid]); }
interval:s:10 { print(@us); clear(@us); }'

# Step2 定位到具体进程（> 1ms 的慢读）
bpftrace -e '
kprobe:vfs_read { @start[tid,comm] = nsecs; }
kretprobe:vfs_read {
  $s=@start[tid,comm];
  if ($s>0) { $lat=nsecs-$s;
    if ($lat>1000000) printf("SLOW: %s pid=%d %dms\n", comm, pid, $lat/1000000);
    delete(@start[tid,comm]); } }'

# Step3 慢读时采集内核+用户栈
bpftrace -e '
kprobe:vfs_read { @start[tid]=nsecs; }
kretprobe:vfs_read /@start[tid]/ {
  if (nsecs-@start[tid] > 5000000) @stacks[kstack,ustack,comm]=count();
  delete(@start[tid]); }
END { print(@stacks); }'
```

**验收**：能识别慢读属于哪个进程/fd；直方图能区分正常 vs 异常；调用栈能指向具体内核路径（PageCache miss vs 锁竞争）；全程不重启服务、不改代码。

### 23.6.4 常见症状 → eBPF 方法

| 症状 | 方法 |
|------|------|
| TCP 连接延迟高 | `kprobe:tcp_v4_connect` 配对计时 / `tracepoint:tcp:*` |
| 磁盘 I/O 延迟 | `tracepoint:block:block_rq_issue` + `block_rq_complete` |
| 调度延迟 | `sched:sched_wakeup` + `sched_switch` 算 runqueue wait |
| 锁竞争 | `tracepoint:lock:contention_begin/end`（需相应 CONFIG）或持锁配对计时 |
| CPU 热点 | `profile:hz:99` + 火焰图 |
| syscall 开销 | `tracepoint:syscalls:sys_enter_*` + `sys_exit_*` |

---

## 23.7 边界与风险：verifier 自己就是头号攻击面

### 23.7.1 能力边界

| 能做 | 不能做 |
|-----|--------|
| 只读访问内核数据结构 | 任意写内核内存（verifier 阻止） |
| 通过 helper 改包数据 | 直接调任意内核函数 |
| 事件触发时执行短暂逻辑 | 长时间阻塞（5.10 后 sleepable BPF 有限支持） |
| map 持久化状态 | 无限大程序（4096 / 1M 双上限） |
| struct_ops 替换 TCP 拥塞控制 | 过去无法替换调度器——6.12 起 sched_ext 已开放 |

### 23.7.2 Spectre 侧信道：verifier 不只是受害者

eBPF 在 Spectre 故事里既是受害者也是放大器。两条线要分清：

- **Spectre v2（CVE-2017-5715）**：BPF 解释器曾被当作 gadget 来源。直接后果是内核引入 `BPF_JIT_ALWAYS_ON`——干脆去掉解释器、只留 JIT（解释器的 computed-goto 间接跳转正是攻击者想要的可控间接分支）。
- **Spectre v1（边界检查绕过）**：攻击者用"先训练分支预测器、再让 CPU 推测性地越过 verifier 插入的边界检查"来读越界数据。verifier 的对策是**在指针运算/数组访问上插入掩码或限界**。源码里能直接看到这套机制：

【真实源码 v6.12 · kernel/bpf/verifier.c · `sanitize_check_bounds()`】
```c
static int sanitize_check_bounds(struct bpf_verifier_env *env,
				 const struct bpf_insn *insn,
				 const struct bpf_reg_state *dst_reg)
{
	u32 dst = insn->dst_reg;

	/* For unprivileged we require that resulting offset must be in bounds
	 * in order to be able to sanitize access later on.
	 */
	if (env->bypass_spec_v1)
		return 0;

	switch (dst_reg->type) {
	case PTR_TO_STACK:
		if (check_stack_access_for_ptr_arithmetic(env, dst, dst_reg,
					dst_reg->off + dst_reg->var_off.value))
			return -EACCES;
		break;
	case PTR_TO_MAP_VALUE:
		if (check_map_access(env, dst, dst_reg->off, 1, false, ACCESS_HELPER)) {
			verbose(env, "R%d pointer arithmetic of map value goes out of range, "
				"prohibited for !root\n", dst);
			return -EACCES;
		}
		break;
	default:
		break;
	}
	return 0;
}
```

逐行读：`env->bypass_spec_v1` 为真（受信任/硬件不受影响）时直接放行；否则对栈指针、map value 指针的算术结果做**额外的在界检查**——因为只有"结果保证在界"，后续才能对它做掩码（让推测执行也越不出界）。报错串 `"...goes out of range, prohibited for !root"` 是非特权程序常见的拒绝原因。verifier 里还有 `BPF_ST | BPF_NOSPEC`（插入推测屏障）、`sanitize_ptr_alu()` 等成套缓解。

### 23.7.3 verifier bug：历史最严重的提权来源

verifier 是内核最复杂代码之一，它一旦算错某个寄存器的取值范围，就会把"实际越界的访问"判成安全——这是 eBPF 最主要的安全风险。把 23.3.4 的双数值域知识接上一个真实 CVE：

**CVE-2021-3490（eBPF ALU32 位运算边界跟踪缺陷）**：
- **根因**：32 位 ALU 的**位运算 AND/OR/XOR 后，没有正确更新 `bpf_reg_state` 里的 `s32_min_value/s32_max_value/u32_min_value/u32_max_value`**（就是 23.3.4 看到的那几个字段）。verifier 因此误判某寄存器的 32 位边界，可被构造成越界读写内核内存、进而任意代码执行。
- **引入/修复 commit（精确到 rc）**：AND/OR 由 `3f50f132d840`（"bpf: Verifier, do explicit ALU32 bounds tracking"，5.7-rc1）引入；XOR 变体由 `2921c90d4718`（5.10-rc1）引入；修复是 `049c4e13714e`（"bpf: Fix alu32 const subreg bound tracking on bitwise operations"，5.13-rc4），回合并到 5.12.4 / 5.11.21 / 5.10.37。
- **发现者**：Manfred Paul（@_manfp，RedRocket CTF），经 Trend Micro ZDI 报告（ZDI-CAN-13590）。被用于容器逃逸。

> **专家视角**：这个 CVE 完美印证了 23.3.4 的论点——eBPF 的安全性收敛于"双数值域更新逻辑的正确性"这个针尖。`3f50f132d840` 引入"显式 ALU32 边界跟踪"本是为了**更精确**（让更多合法程序通过），却恰恰在这块新代码里漏了一种位运算的边界更新。**精确性与安全性在 verifier 里是一对内在张力**：越想少误杀就越要写复杂的收窄逻辑，而复杂逻辑本身就是 bug 温床。这也是学术界要用形式化方法（Agni 离线用 SMT 验证 verifier 的 tnum/区间转移函数、PREVAIL 用抽象解释做独立验证器）去守护 verifier 的根本动因。

### 23.7.4 纵深防御与默认值演进

- **权限**：`CAP_BPF`（细粒度，替代过去一刀切的 `CAP_SYS_ADMIN`）/ `CAP_NET_ADMIN`。
- **`unprivileged_bpf_disabled` sysctl**：`0`=允许非特权加载，`1`/`2`=禁止（`2` 还锁死、不可运行时改回，直到重启）。内核 **5.16 起 `CONFIG_BPF_UNPRIV_DEFAULT_OFF`** 让该值默认 `2`（动机正是 Spectre：不让非特权用户构造侧信道所需条件）；Ubuntu（自 5.13）、Debian 等主流发行版也已默认关闭非特权 eBPF。

【示意代码，非逐字】
```bash
cat /proc/sys/kernel/unprivileged_bpf_disabled   # 生产应为 1 或 2
sysctl kernel.unprivileged_bpf_disabled=2        # 禁止且锁定，直到重启
```

> **新手误区**：以为"有 verifier 所以 eBPF 绝对安全"。错——verifier 自身是内核最复杂代码之一，其安全性依赖持续审查与 fuzzing，不是绝对保证。CVE-2021-3490 就是教科书反例。

---

## 23.8 未来演进

### 23.8.1 sched_ext：把调度器开放给 BPF（mainline 6.12）

见 23.2.6 的考古。技术现状（2026 视角）：6.12 落地的是"框架 + cgroup 基础支持"，CPU 频率调节、CPU hotplug 等在后续版本补齐；社区还在推进**子调度器（sub-schedulers，让多个 BPF 调度器共存于一台机器）**。生产采用前务必核对目标内核版本的能力矩阵。它是"加载时证明 + 运行时零信任"模型触及内核最敏感子系统的标志，但也带着 Peter Zijlstra 们关于碎片化的长期警惕。

### 23.8.2 Rust for BPF（aya）

`aya` 提供内核态 + 用户态全 Rust 的 eBPF 开发，利用所有权系统在编译期捕获更多错误，相比 libbpf 的 C 接口更安全。已有生产用例，但生态成熟度仍低于 libbpf。

### 23.8.3 能力持续扩展

- **kfunc**：逐步成为比固定 helper 白名单更主流的内核能力导出方式（无 UAPI 稳定承诺、更灵活）。
- **Sleepable BPF（5.10+）**：允许在可睡眠上下文执行可能阻塞的操作（如 `bpf_copy_from_user`）。
- **BPF arena**：内核态/用户态共享的稀疏线性地址空间（`BPF_MAP_TYPE_ARENA`），让复杂数据结构在两侧共享更自然。
- **struct_ops 扩张**：更多子系统开放函数表替换。
- **hid_bpf（6.3）**：用 eBPF 给不合规 HID 设备打 quirk/补丁，无需改驱动（Red Hat 主导，配 `udev-hid-bpf`）。

---

## 章末总结

> **心法**：eBPF 不是"内核脚本"，而是"**加载时证明 + 运行时零信任**"。verifier 是一台跑在 syscall 上下文里的抽象解释机——它用 tnum + 区间双数值域逼近形式化证明、用状态去重在指数级路径空间里活下来、用类型格做空指针收窄与所有权分析，**宁可保守误杀也绝不漏判**。map 是唯一合法的状态与通信协议，附着点是触发点，CO-RE 是可移植性的工程闭环。选对附着点、写对 bounds check、看懂 verifier 日志，剩下的让 verifier 替你把关。而它最深的风险恰恰是：替你把关的这台机器，自己就是内核最复杂、最值得警惕的攻击面。

---

## 章末五件套

### 一、高频面试题（Staff 级）

**Q1：verifier 如何保证安全？根本局限是什么？**
- 加载时**抽象解释**（非运行时沙箱）：沿 CFG 做 DFS 符号执行，对每寄存器维护 tnum + 区间双域 + 类型标签。
- 三大保证：有界执行（路径有限/循环可证终止）、内存安全（类型格 + bounds check + Spectre 掩码）、资源不泄漏（`ref_obj_id` 所有权分析）。
- 根本局限：① 只证内存安全，**不证逻辑正确**；② 抽象域保守 → **误杀合法程序**；③ state explosion 限制复杂度；④ **verifier 自身复杂 → 有 bypass CVE**（CVE-2021-3490）。
- 陷阱答案："有 verifier 所以没安全风险"——错。

**Q2：4096 和 1,000,000 是同一回事吗？**
- 不是。`BPF_MAXINSNS=4096` 是**程序长度**上限，对非特权程序仍是硬墙；`BPF_COMPLEXITY_LIMIT_INSNS=1,000,000` 是 verifier **放弃前模拟的指令次数**上限。分支爆炸常在程序还很短时就先撞 1M 墙，报 "too large. Processed N insn"。

**Q3：kprobe / tracepoint / fentry 怎么选？**（标准答案见 23.3.9 的"跑一遍"）稳定性 > 性能 > 覆盖：有 BTF 的现代内核优先 fentry/fexit（ftrace nop + trampoline，近函数直调）；要稳定语义用 tracepoint；kprobe 是兜底但脆（内联/改名即失效）。内核 UAPI 注释本身就声明 tracing 类程序**不承诺稳定 API**。

**Q4：tnum 和 smin/smax 区间为什么要两套？**
- 位运算（AND/OR/移位）在 tnum 上精确；比较跳转在区间上精确；两者互相收窄（`__reg_deduce_bounds`），32/64 位之间也要同步。CVE-2021-3490 的根因正是 ALU32 位运算后漏更新 `s32/u32` 区间字段。

**Q5：为什么 verifier 不用 SMT solver？**（见 23.3.4 对照表）要在 syscall 上下文毫秒级返回、自身要可人工审计、宁可保守误杀也不能 unsound——这排除了完备但庞大慢的符号执行/SMT。学术界用 SMT 是**离线验证 verifier 自身**，不是塞进运行时。

**Q6：CO-RE 解决了什么？没它之前怎么办？**
- 问题：结构体字段偏移因 `CONFIG`/版本而变，无法编译期写死。
- 旧法：①硬编码（绑特定内核）②运行时解析 debuginfo ③BCC 在目标机现编译（背 LLVM、要内核头、可能压垮生产）。
- CO-RE：BTF 内嵌类型 + 编译器发 CO-RE 重定位 + libbpf 加载期改偏移。必要条件 `CONFIG_DEBUG_INFO_BTF=y`。

**Q7：XDP vs TC BPF？**（见 23.5.2"跑一遍"）XDP 在驱动层、skb 分配前、只 ingress、动作少、最快（DDoS 丢包）；TC BPF 已有 skb、ingress+egress、能力强、略慢（有状态/改 egress）。

**Q8：ring buffer 比 perf event array 强在哪？**
- perf_event_array：每 CPU 独立 buffer，用户态 epoll 多个 fd，高频可能丢、内存映射复杂。
- ringbuf（5.8+）：MPSC 单一/共享 buffer，用户态单 fd poll，预留-提交（reserve/submit）语义、生产者不阻塞、消费者慢时丢弃带水位通知。新代码优先 ringbuf。

### 二、实战项目（P2 主线）

**目标**：在一台跑真实服务的机器上，用 bpftrace 定位一个 I/O 或 CPU 热点，产出可解读报告。
- P2.0 环境：`/sys/kernel/btf/vmlinux` 存在；bpftrace ≥ 0.14；核对 `unprivileged_bpf_disabled`。
- P2.1 基线：`profile:hz:99` 跑 30s 出火焰图原料；`vfs_read/write` 延迟直方图；`biolatency` 测块设备。
- P2.2 注入：`dd if=/dev/urandom ...` 注 I/O 压；`stress-ng` 注 CPU 压；压力期间重采集对比。
- P2.3 深追：慢路径用 `kstack+ustack` 拿完整调用链；追特定进程 syscall 延迟分布。
- P2.4 报告：火焰图 SVG（内核+用户）、正常 vs 压力的延迟直方图、文字分析（热路径 + 推断 + 建议）。
- 验收：识别 CPU 热点（精确到 kstack 最深 3 层）；VFS p50/p99 在压力下可见变化；火焰图可解释；脚本全部能跑。

### 三、设计题

**设计一个实时 TCP 连接跟踪系统**：记录所有连接建立/销毁/字节数；① 业务进程零侵入 ② 支持按 PID/进程名/目标 IP 过滤 ③ 每秒汇总上报用户态 ④ 内核性能影响 < 1%。
- **考察**：附着点（`tracepoint:tcp:*` 优先于 kprobe，稳定）；map 选型（`PERCPU_ARRAY` 做计数器免锁 + `LRU_HASH` 存连接状态防泄漏）；上报通道（ringbuf vs perf array）；过滤放 eBPF 侧（省上报带宽）还是用户态（灵活）的权衡；连接生命周期（LRU 自动驱逐防 FD 泄漏）。

### 四、系统题

**场景**：K8s 核心微服务 p99 在白天高峰周期性升高约 50ms，但 CPU/内存监控正常；只有 Prometheus 15s 粒度、无 APM。
- 初始假设（≥3）：调度延迟（noisy neighbor / cgroup CPU throttling）、锁竞争、下游依赖/网络 RTT 抖动、GC 或周期性批任务。
- 缩小范围：`profile:hz:99` 排除纯 CPU 热点 → `syscalls:sys_enter_*/exit_*` 找最慢 syscall → `sched:sched_wakeup`+`sched_switch` 量化 runqueue wait。
- 若是调度延迟：查 cgroup `cpu.stat` 的 `nr_throttled`/`throttled_time`（CFS quota 节流）、查同核 noisy neighbor、确认是否 CPU 亲和/cpuset 冲突。缓解：调 quota、加亲和隔离、或上 sched_ext 定制策略。

### 五、代码题

**题目**：写一个 eBPF 程序（libbpf 风格），统计每进程 `write` 的次数与累计字节数，用户态每 5s 读 top 10。

【示意代码，非逐字 · 内核态参考实现（含暗坑注解）】
```c
// write_counter.bpf.c
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

struct proc_stat { __u64 count; __u64 bytes; };

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, __u32);
	__type(value, struct proc_stat);
} write_stats SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_write")
int trace_write_enter(struct trace_event_raw_sys_enter *ctx)
{
	__u32 pid = bpf_get_current_pid_tgid() >> 32;
	__u64 count = (__u64)ctx->args[2];        // write(fd, buf, count) 第三参

	struct proc_stat *st = bpf_map_lookup_elem(&write_stats, &pid);
	if (st) {                                 // 必须 NULL 检查，否则 verifier reject
		st->count++;                          // 指针直指 map 内部存储，原地改即可
		st->bytes += count;
	} else {
		struct proc_stat ns = { .count = 1, .bytes = count };
		bpf_map_update_elem(&write_stats, &pid, &ns, BPF_ANY);
	}
	return 0;
}
char LICENSE[] SEC("license") = "GPL";
```

**暗坑（全部能用 23.3 的源码知识解释）**：
1. `ctx->args[2]` 是 `unsigned long`，强转注意 32/64 位。
2. `bpf_map_lookup_elem` 返回 `PTR_TO_MAP_VALUE_OR_NULL`，更新前必须 `if (st)` 收窄成非空（见 23.3.5 类型格），漏掉直接 reject。
3. HASH map 的 value 指针直指内部存储，可原地 `st->count++`，无需写回。若换 PERCPU map：程序内 `lookup` 默认返回**本 CPU** 那份（无需指定 CPU），但**用户态聚合要遍历全部 CPU 副本求和**（见 23.3.7）。
4. tracepoint 上下文结构体字段名依赖内核版本，用 `vmlinux.h`（BTF）解析才稳，手写 struct 易错——这正是 CO-RE 要解决的事。

**验证**：`yes > /tmp/test &`，运行工具应看到 `yes` 进程 write count 持续增长。

---

## 取材来源（本章实际抓取/核实的 URL）

**真实源码（raw.githubusercontent.com，v6.12 tag）**
- kernel/bpf/verifier.c、kernel/bpf/core.c
- include/uapi/linux/bpf.h、include/linux/bpf.h、include/linux/bpf_verifier.h、include/linux/tnum.h、include/uapi/linux/bpf_common.h

**设计考古 / 史料**
- LWN 599755《BPF: the universal in-kernel virtual machine》(Corbet, 2014-05-21)
- LWN 740157《A thorough introduction to eBPF》(Fleming, 2017-12-02)
- LWN 773605《Bounded loops in BPF programs》(Corbet, 2018-12-03)
- LWN 922405《The extensible scheduler class》(Corbet, 2023-02-10) — Zijlstra NAK 原话
- LWN 978007《Extensible scheduler class to be merged for 6.11》（落地最终为 6.12）
- github.com/torvalds/linux commit `bd4cf0ed331a`（内部 BPF 解释器重写，含性能数字）
- github.com/torvalds/linux commit `99c55f7d47c0`（bpf syscall + maps 引入）
- nakryiko.com《BPF Portability and CO-RE》(Nakryiko, 2020-02-19)
- CVE-2021-3490：openwall oss-security / Ubuntu / Red Hat 公告（引入 `3f50f132d840`、`2921c90d4718`；修复 `049c4e13714e`；Manfred Paul / ZDI-CAN-13590）
- `CONFIG_BPF_UNPRIV_DEFAULT_OFF`（cateee lkddb；Daniel Borkmann patch；默认值 2 与 Spectre 动机）
- McCanne & Jacobson《The BSD Packet Filter: A New Architecture for User-level Packet Capture》USENIX Winter 1993（cBPF 起源）
