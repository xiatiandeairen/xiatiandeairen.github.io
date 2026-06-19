---
title: "ART 运行时:dex · AOT · JIT · GC（Android 域)"
slug: "8-03"
collection: "tech-library"
group: "android系统"
order: 8003
summary: "TL;DR ART 不是\"一个编译器\",而是一台 多档位执行机:同一个方法在生命周期里会依次经过 解释器(nterp / switch-interpreter)→ JIT baseline → JIT optimized → AOT,任意时刻可以因为 deopt / GC 回退一档。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:01:16.000Z"
updatedAt: "2026-06-12T11:01:16.000Z"
---
> **TL;DR**
> ART 不是"一个编译器",而是一台 **多档位执行机**:同一个方法在生命周期里会依次经过 解释器(nterp / switch-interpreter)→ JIT baseline → JIT optimized → AOT,任意时刻可以因为 deopt / GC 回退一档。理解 ART 的关键是抓住三条主线:
> 1. **代码以 dex 为唯一真理来源**(register-based bytecode),`.oat`/JIT code 都只是 dex 的某种"缓存编译产物",可被丢弃后退回解释执行。
> 2. **档位切换由 hotness counter 驱动**(`ArtMethod::hotness_count_`,默认阈值 `0xffff`),profile 把"哪些方法/类热"持久化下来,喂给 `dex2oat` 做 **profile-guided AOT**——这就是 N7 之后"边用边变快"的本质。
> 3. **GC 三代演进**:CMS(标记-清除,会碎片化)→ Concurrent Copying(读屏障 + region,O8 默认)→ **Concurrent Mark-Compact / userfaultfd**(O13+ 当前默认,用内核缺页拦截做并发整理,省掉 copying 的 to-space 内存开销)。**注意:很多二手资料还停在"CC 是默认",这是过时的。**
>
> **前置知识**:C++ 基础、虚拟内存(页/缺页/mmap/mprotect)、GC 基本概念(根集/可达性/STW)、对 JVM 字节码或任意一种 VM 有概念。不需要 Android 应用开发经验。
> **可运行性**:本章 dexdump demo **在 macOS 主机上用 Android SDK build-tools 实测可跑通**(已附真实输出);AOT/JIT 的运行时观测部分**需 Android 模拟器/真机 + adb**,已逐条标注前置。
> **源码版本**:全部源码取自 `android.googlesource.com/platform/art` 的 `refs/heads/main`(AOSP ART 主线),抓取时间 2026-06。逐字引用标【真实源码 repo@path】,示意改写标【示意,非逐字】,无把握标「待核」。

---

## 1. 设计考古:从 Dalvik 到 ART,从纯 AOT 到混合,从拷贝到缺页整理

做系统的人读运行时,第一件事不是看代码,是看**它在反抗什么**。ART 的每一档设计都是在反抗上一代的某个具体痛点。把这条反抗链捋清楚,后面读源码就是"印证"而不是"记忆"。

### 1.1 三个时间轴坐标(有出处)

| Android 版本 | 代号 | 运行时大事 | 反抗的痛点 |
|---|---|---|---|
| 4.4 | KitKat | ART 作为**技术预览**与 Dalvik 并存 | Dalvik 纯 JIT,每次启动重新 JIT,冷启动慢、抖动 |
| 5.0 | Lollipop | ART **完全取代** Dalvik,默认**纯 AOT**(install-time 全量 `dex2oat`) | 去掉运行时 JIT 抖动;但带来安装慢、占用大 |
| 7.0 | Nougat | 退回**混合 JIT/AOT**(profile-guided);JRE 从 Apache Harmony 换 **OpenJDK** | 纯 AOT 安装时全量编译太贵(应用大 / OTA 后要重编全世界) |
| 8.0 | Oreo | GC 默认换成 **Concurrent Copying (CC)** | CMS 标记-清除会**堆碎片化**,且 compaction 时要 STW |
| 10 | Q | CC 扩展为 **Generational CC**(默认开) | full-heap CC 对短命对象太浪费,加 young gen |
| 13 | T | GC 默认换成 **userfaultfd Concurrent Mark-Compact (CMC)** | CC 是 copying,需要 ~2x 存活区的 to-space,**RSS 开销大** |

> 出处(实际 WebFetch 核实):
> - 版本史 / "KitKat 预览、Lollipop 取代 Dalvik、Nougat 混合 JIT/AOT 并换 OpenJDK":**Wikipedia "Android Runtime"**(WebFetch 取得逐句)。
> - "CC 自 O8 默认、O10 generational":**source.android.com/docs/core/runtime/gc-debug**(WebFetch 逐句)。
> - "O13 引入 userfaultfd GC":Wikipedia "ART was updated with a new garbage collector (GC) utilizing the Linux userfaultfd system call in Android 13"(WebFetch 逐句);并由 AOSP 源码 `runtime/gc/collector/mark_compact.h` 中 `kCollectorTypeCMC` / `SigbusHandler` / `uffd_` 印证(下文 §5)。
> - 注:**source.android.com 的公开 gc-debug 文档目前仍未收录 CMC**(WebFetch 实测只提到 CC/CMS),文档落后于源码。本章以源码为准,并明确标注这处文档滞后。

### 1.2 N5→N7 的"反转":为什么 Google 自己推翻了纯 AOT

这是 ART 设计史上最值得品的一次决策。Lollipop 信誓旦旦把 Dalvik 的 JIT 全砍了,换成 install-time AOT——理由很硬:运行时不再有 JIT 抖动,代码全是原生的。

然后两年后被自己推翻。痛点是工程现实,不是技术理论:

- **安装慢**:每装一个 app,`dex2oat` 要把**全部**方法编译成机器码。大型 app(几万方法)安装要几十秒到分钟级。
- **OTA 灾难**:系统升级后 boot classpath 变了,**所有 app 的 oat 全部失效,要重编全世界**。用户看到的就是"升级后转圈几十分钟"。
- **占用大**:全量 AOT 产物经常比 dex 本身大几倍,而**大多数方法一辈子只跑几次**——为冷代码付的编译/存储成本纯浪费。

Nougat 的答案是 **profile-guided 混合编译**,核心 insight 一句话:**"只 AOT 真正热的代码,热不热让真实运行说了算。"**

官方文档(WebFetch from source.android.com/docs/core/runtime/jit-compiler)把流程讲得很清楚:
> "When an app launches, ART loads the `.dex` file. If an `.oat` file (AOT binary) exists and contains compiled code, it's used directly. Otherwise, ART runs through JIT and the interpreter to execute the `.dex` file."
> "The JIT profile data is dumped to a file in a system directory that only the application can access. The AOT daemon (`dex2oat`) then parses that file to drive its compilation."

落地成了今天每台 Android 上都在跑的循环:**装的时候不编(或只编 verify)→ 用的时候解释 + JIT,顺手记 profile → 半夜充电息屏时,后台 `dex2oat` 拿 profile 把热方法 AOT 成 `speed-profile` 的 oat → 下次启动直接用 oat。** 这就是"新装的 app 用几天会越来越快"的全部真相。

> **设计 takeaway**:这是一个典型的"全局最优 vs 摊还最优"的反转。纯 AOT 在单次执行上最优(没有解释/JIT 开销),但在**用户感知的总成本**(安装 + OTA + 存储 + 实际只跑热代码)上是负优化。系统设计里,**把成本摊到"空闲资源"(息屏充电)上**几乎永远比"在关键路径上一次付清"赢。

### 1.3 GC 的反抗链:碎片 → copying 开销 → 内核缺页

三代 GC 不是"越来越快"这么简单,每一代换的是**不同的 trade-off 维度**:

- **CMS(Concurrent Mark-Sweep)**:并发标记 + 并发清除,但**不并发整理**。清除留下碎片,大对象分配失败 → 触发 compaction → compaction 要 **STW**。痛点:**碎片 + 偶发长暂停**。
- **CC(Concurrent Copying)**:用 **region space + 读屏障(read barrier)** 实现**并发拷贝整理**。对象一边被 mutator 访问一边被搬,几乎无碎片,暂停只有一个跟堆大小无关的常数小停顿(flip)。痛点:**copying 天生要一块 to-space**——堆里始终有一部分内存"留着搬过去用",**RSS 开销大**(对内存紧张的低端机很痛)。
- **CMC(userfaultfd Concurrent Mark-Compact)**:**mark-compact**(原地滑动压缩,不需要等大的 to-space),并发性靠 **Linux `userfaultfd`**:把 moving-space 注册到 uffd,mutator 访问还没整理完的页会触发**缺页 / SIGBUS**,GC 在缺页处理里**按需把那一页整理好再映射回去**。痛点解决:**省掉 CC 的 to-space 内存开销**,同时保持并发。代价:依赖较新内核的 uffd 特性(老内核 / 被 SELinux 限制的设备要 fallback)。

> 这条链非常"系统":CMS→CC 是"用读屏障换碎片",CC→CMC 是"用内核缺页机制换内存开销"。每一步都把 GC 的难点**外包给硬件/内核能力**(屏障靠 CPU 的 data dependency,缺页靠 MMU + uffd)。

---

## 2. dex:一切的真理来源

在讲 AOT/JIT/GC 之前必须先夯实 dex,因为**ART 里所有编译产物都是 dex 的派生物**,可以被删掉退回 dex 解释执行。不懂 dex 的结构,后面 oat / vdex / profile 都是空中楼阁。

### 2.1 register-based bytecode:dex 与 JVM .class 的根本区别

JVM 的 `.class` 是 **stack-based**(操作数压栈/弹栈),dex 是 **register-based**(虚拟寄存器 vN)。区别不是风格,是为**移动端解释器的执行效率**服务的:register-based 指令数更少、解释 dispatch 次数更少。这在我们后面 demo 的真实反汇编里会直接看到。

### 2.2 dex header_item:逐字段解剖【真实源码】

dex 文件开头是一个定长 header。AOSP 用 C++ 结构体精确镜像了它:

```cpp
// 【真实源码 platform/art@libdexfile/dex/dex_file.h, refs/heads/main, L116-L163】
  static constexpr size_t kDexMagicSize = 4;
  static constexpr size_t kDexVersionLen = 4;
  static constexpr uint32_t kDexContainerVersion = 41;
  // ...
  static constexpr uint32_t kDexEndianConstant = 0x12345678;
  // ...
  using Magic = std::array<uint8_t, 8>;
  // ...
  // Raw header_item.
  struct Header {
    Magic magic_ = {};
    uint32_t checksum_ = 0;  // See also location_checksum_
    Sha1 signature_ = {};
    uint32_t file_size_ = 0;  // size of entire file
    uint32_t header_size_ = 0;  // offset to start of next section
    uint32_t endian_tag_ = 0;
    uint32_t link_size_ = 0;  // unused
    uint32_t link_off_ = 0;  // unused
    uint32_t map_off_ = 0;  // map list offset from data_off_
    uint32_t string_ids_size_ = 0;  // number of StringIds
    uint32_t string_ids_off_ = 0;  // file offset of StringIds array
    uint32_t type_ids_size_ = 0;   // number of TypeIds, we don't support more than 65535
    uint32_t type_ids_off_ = 0;    // file offset of TypeIds array
    uint32_t proto_ids_size_ = 0;  // number of ProtoIds, we don't support more than 65535
    uint32_t proto_ids_off_ = 0;   // file offset of ProtoIds array
    uint32_t field_ids_size_ = 0;  // number of FieldIds
    uint32_t field_ids_off_ = 0;   // file offset of FieldIds array
    uint32_t method_ids_size_ = 0; // number of MethodIds
    uint32_t method_ids_off_ = 0;  // file offset of MethodIds array
    uint32_t class_defs_size_ = 0; // number of ClassDefs
    uint32_t class_defs_off_ = 0;  // file offset of ClassDef array
    uint32_t data_size_ = 0;  // size of data section
    uint32_t data_off_ = 0;   // file offset of data section
    // ...
  };
```

逐行要点(系统视角):
- `magic_`:8 字节,`"dex\n0XY\0"`。版本号 035/037/038/039/040/041 是**三个十进制 ASCII 数字**(不是二进制!),所以你 hexdump 看到的是 `30 33 38`= `'0''3''8'`。版本随 dex 特性演进(invoke-polymorphic、container 格式等)。
- `checksum_`:**Adler-32**,覆盖 magic 之后的全部字节。ART 加载时校验,被改一个字节就拒绝。注意它和 `location_checksum_`(oat/profile 里用来标识 dex 身份的那个 checksum)是两个东西。
- `signature_`:20 字节 **SHA-1**,作为 dex 的内容指纹。oat / profile 用它来确认"我编译/记录的就是这个 dex"。
- `*_size_ / *_off_`:dex 是一堆**并行数组 + 偏移指针**的格式(string_ids / type_ids / proto_ids / field_ids / method_ids / class_defs),典型的"定长索引表 + data 区"布局,便于 mmap 后随机访问而不必整体反序列化。
- `kDexEndianConstant = 0x12345678`:`endian_tag_` 用它判定字节序;反序就是 `0x78563412`(官方 DEX 文档:REVERSE_ENDIAN_CONSTANT)。实践中几乎全是小端。

`StandardDexFile`(APK 里打包、d8/dexpacker 产出的标准格式)直接继承这个 header,几乎不加字段:

```cpp
// 【真实源码 platform/art@libdexfile/dex/standard_dex_file.h, refs/heads/main, L31-L36】
// Standard dex file. This is the format that is packaged in APKs and produced by tools.
class StandardDexFile : public DexFile {
 public:
  class Header : public DexFile::Header {
    // Same for now.
  };
```

> **磁盘 vs 内存**:dex 在 APK 里、也可被 ART **mmap** 进内存直接执行(`data_off_`/各 `*_off_` 都是文件偏移,mmap 后即内存偏移)。这就是为什么 dex 既能当"磁盘格式"又能当"运行时只读代码区"——它本来就是为 mmap 设计的。

---

## 3. ⭐ Demo A:亲手造一个 dex,把 header 和 register-bytecode 看出来(主机实测可跑)

这是本章第一个、也是**唯一在普通开发机上就能完整跑通**的 demo。它把 §2 的 `Header` 结构体和 §4 解释器要 dispatch 的 register-bytecode 同时落到真实输出上,形成"源码 ↔ 实物"的闭环。

> **前置**:macOS/Linux + JDK(`javac`/`java`)+ Android SDK build-tools(含 `d8`、`dexdump`)。**不需要设备/模拟器。** 本 demo 在 macOS + build-tools `36.1.0`(d8 9.0.3,dexdump from AOSP)上**实测通过**,下方输出为真实结果。

### 3.1 步骤

```bash
# 0) 定位工具(按你的 SDK 路径改)
BT="$ANDROID_HOME/build-tools/36.1.0"   # 内含 d8、dexdump
# macOS 默认: ~/Library/Android/sdk/build-tools/<ver>

# 1) 写一个带"热循环"的最小 Java 类
mkdir -p /tmp/dexdemo && cd /tmp/dexdemo
cat > Hot.java <<'EOF'
public class Hot {
  static int sum(int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += i;   // 这个循环就是将来 JIT 眼里的 "hot"
    return s;
  }
  public static void main(String[] a) {
    System.out.println(sum(1000));
  }
}
EOF

# 2) 编成 .class,再用 d8 降成 dex(min-api 26 => dex 038)
javac --release 8 Hot.java
"$BT/d8" --min-api 26 --output . Hot.class      # 产出 classes.dex

# 3) 看 magic(前 16 字节)
xxd -l 16 classes.dex

# 4) dump header(对照 §2 的 Header 结构体)
"$BT/dexdump" -f classes.dex | sed -n '/DEX file header/,/data_off/p'

# 5) 反汇编 sum 方法,看 register-based bytecode
"$BT/dexdump" -d classes.dex | sed -n '/Hot.sum/,/return v1/p'
```

### 3.2 真实输出(本机实测,非杜撰)

第 3 步,magic:
```
00000000: 6465 780a 3033 3800 fd8b 6047 4586 47d1  dex.038...`GE.G.
```
解读:`64 65 78 0a` = `"dex\n"`;`30 33 38 00` = `"038\0"` → **dex 版本 038**;紧接着 `fd 8b 60 47`... 就是 `checksum_`(Adler-32),再后面 20 字节是 `signature_`(SHA-1)。**这正是 §2 `Header` 结构体的前几个字段在磁盘上的样子。**

第 4 步,header(节选,真实输出):
```
DEX file header:
magic               : 'dex\n038\0'
checksum            : 47608bfd
signature           : 4586...d836
file_size           : 960
header_size         : 112
link_size           : 0
string_ids_size     : 17
type_ids_size       : 7
proto_ids_size      : 4
field_ids_size      : 1
method_ids_size     : 5
class_defs_size     : 1
data_size           : 624
data_off            : 336 (0x000150)
```
**逐字段对得上**:`header_size : 112` = 0x70(标准 header 长度,官方文档说的 header_size 应为 0x70);`file_size : 960` = 我们这个 dex 的总字节数(和 `ls -l classes.dex` 一致);`method_ids_size : 5`(`<init>`、`sum`、`main`、`println`、`Object.<init>` 等)。`checksum`/`signature` 就是结构体里那两个指纹字段。

第 5 步,`sum` 的真实反汇编:
```
000150:                                        |[000150] Hot.sum:(I)I
000162: 1200                                   |0001: const/4 v0, #int 0 // #0
000164: 0101                                   |0002: move v1, v0
000166: 3520 0600                              |0003: if-ge v0, v2, 0009 // +0006
00016a: b001                                   |0005: add-int/2addr v1, v0
00016c: d800 0001                              |0006: add-int/lit8 v0, v0, #int 1 // #01
000170: 28fb                                   |0008: goto 0003 // -0005
000172: 0f01                                   |0009: return v1
```
这就是 **register-based bytecode** 的真容:`v0/v1/v2` 是虚拟寄存器,没有 JVM 那种 push/pop 操作数栈。循环体只有 `if-ge / add-int/2addr / add-int/lit8 / goto` 四条指令。**记住这段——下一节解释器 `ExecuteSwitch` 就是对这串 opcode 做 dispatch;当 `sum(1000)` 被调用足够多次,正是这个 `goto` 回边触发 hotness 计数,最终把 `sum` 推上 JIT。**

> **demo ↔ 源码呼应点**:你在这里 hexdump 出来的字节序、dexdump 打印的每个 `*_size`/`*_off`,一一对应 `dex_file.h` 的 `struct Header` 字段。这不是"教学示意",是同一份二进制的两种视图。

---

## 4. 执行档位:解释器 → JIT baseline → JIT optimized →(后台)AOT

ART 的执行引擎是**多档位**的。一个方法第一次被调用时几乎一定是解释执行,随着变热逐级升档。理解这套"升档"机制,核心就两个东西:**dispatch 决策点** 和 **hotness 计数器**。

### 4.1 dispatch 决策点:解释器入口处的"要不要跳去编译码"【真实源码】

每次进入一个方法的解释执行,ART 在 `Execute()` 里先问一句:这方法是不是已经有 JIT code 了?有就别解释了,直接跳过去:

```cpp
// 【真实源码 platform/art@runtime/interpreter/interpreter.cc, refs/heads/main, L270-L296】
    ArtMethod *method = shadow_frame.GetMethod();

    // If we can continue in JIT and have JITed code available execute JITed code.
    if (!stay_in_interpreter &&
        !self->IsForceInterpreter() &&
        !shadow_frame.GetForcePopFrame() &&
        !shadow_frame.GetNotifyDexPcMoveEvents()) {
      jit::Jit* jit = Runtime::Current()->GetJit();
      if (jit != nullptr) {
        jit->MethodEntered(self, shadow_frame.GetMethod());   // ① 计 hotness
        if (jit->CanInvokeCompiledCode(method)) {             // ② 有可用编译码?
          JValue result;
          // Pop the shadow frame before calling into compiled code.
          self->PopShadowFrame();
          uint16_t arg_offset = accessor.RegistersSize() - accessor.InsSize();
          ArtInterpreterToCompiledCodeBridge(self, nullptr, &shadow_frame, arg_offset, &result);
          self->PushShadowFrame(&shadow_frame);
          return result;                                       // ③ 走编译码,不再解释
        }
      }
    }
    // ... 没有编译码,落到最后:
```
最后落到 switch 解释器:
```cpp
// 【真实源码 同文件 L339-L341】
  VLOG(interpreter) << "Interpreting " << method->PrettyMethod();
  return ExecuteSwitch(self, accessor, shadow_frame, result_register);
```
三个关键点:
- **① `MethodEntered` → AddSamples**:每次进方法都给 hotness 计数(见 §4.3),这是"变热"的来源之一。
- **② `CanInvokeCompiledCode`**:JIT/AOT 产物就绪后,这里返回 true,执行**当场升档**到机器码——通过 `ArtInterpreterToCompiledCodeBridge` 这个**桥**完成 shadow frame → 原生栈帧的切换。
- **`shadow_frame`**:解释器的"虚拟栈帧",dex 的 `vN` 寄存器就活在这里面。升档去编译码前要 `PopShadowFrame`(切到原生栈),回来再 `PushShadowFrame`。

> **注意**:上面这段是 **switch-interpreter** 路径(C++ 大 switch,可移植但慢)。**真机热路径上跑的其实是 nterp**——一个汇编写的快解释器,见 §4.2。switch-interpreter 在 debuggable、interpret-only、不支持 nterp 的架构等情况下兜底。

### 4.2 nterp:被很多资料忽略的"现代快解释器"【真实源码】

老 ART 的快解释器叫 **mterp**(汇编 + computed-goto)。现在主线是 **nterp**("new interpreter"),整段用平台汇编实现,直接用原生栈和 quick 栈帧布局,所以解释执行的开销大幅下降——以至于"很多过去必须靠 JIT 才能接受的代码,nterp 解释跑就够快了"。它的可用条件被写死在源码里:

```cpp
// 【真实源码 platform/art@runtime/interpreter/mterp/nterp.cc, refs/heads/main, L37-L57】
bool IsNterpSupported() {
#ifdef ART_USE_RESTRICTED_MODE
  // TODO(Simulator): Support Nterp.
  return false;
#else
  switch (kRuntimeQuickCodeISA) {
    case InstructionSet::kArm:
    case InstructionSet::kThumb2:
    case InstructionSet::kArm64:
      return kReserveMarkingRegister && !kUseTableLookupReadBarrier;
    case InstructionSet::kRiscv64:
      return true;
    case InstructionSet::kX86:
    case InstructionSet::kX86_64:
      return !kUseTableLookupReadBarrier;
    default:
      return false;
  }
#endif  // #ifdef ART_USE_RESTRICTED_MODE
}
```
```cpp
// 【真实源码 同文件 L60-L73】
bool CanRuntimeUseNterp() REQUIRES_SHARED(Locks::mutator_lock_) {
  Runtime* runtime = Runtime::Current();
  instrumentation::Instrumentation* instr = runtime->GetInstrumentation();
  return IsNterpSupported() && !runtime->IsJavaDebuggable() && !instr->EntryExitStubsInstalled() &&
         !instr->InterpretOnly() && !runtime->IsAotCompiler() &&
         !instr->NeedsSlowInterpreterForListeners() &&
         !runtime->AreAsyncExceptionsThrown() &&
         (runtime->GetJit() == nullptr || !runtime->GetJit()->JitAtFirstUse());
}
```
逐行读出三个系统事实:
- **架构相关**:nterp 只在 arm/arm64/riscv64/x86/x86_64 上有(`default: return false`)。它是手写汇编,必须逐架构移植。
- **与读屏障耦合**:arm/arm64 上要求 `kReserveMarkingRegister && !kUseTableLookupReadBarrier`——nterp 依赖 **Baker 读屏障保留了一个 marking register**(见 §5)。这是 GC 设计反过来约束解释器的活证据:**GC 与执行引擎不是两个独立模块**。
- **debuggable 即降级**:`!IsJavaDebuggable()` —— 一旦 app 可调试 / 开了 instrumentation,nterp 全部退回 switch-interpreter(因为 nterp 不支持逐指令 instrumentation 回调)。**这解释了一个经典生产现象:同一台机器,`android:debuggable="true"` 的 build 明显比 release 慢——不只是少了 R8 优化,更因为整条执行链从 nterp 掉到 switch-interpreter。**

nterp 也参与"变热":在做字段/方法/类查找的慢路径时给方法加 hotness:
```cpp
// 【真实源码 platform/art@runtime/interpreter/mterp/nterp.cc, refs/heads/main, L126-L130】
inline void UpdateHotness(ArtMethod* method) REQUIRES_SHARED(Locks::mutator_lock_) {
  // The hotness we will add to a method when we perform a
  // field/method/class/string lookup.
  method->UpdateCounter(0xf);
}
```

### 4.3 hotness 计数器:升档的"扳机"【真实源码】

"变热"的核心就是给 `ArtMethod` 上的一个计数器累加,过阈值就排队编译。逻辑在 `AddSamples`:

```cpp
// 【真实源码 platform/art@runtime/jit/jit-inl.h, refs/heads/main, L31-L54】
inline void Jit::AddSamples(Thread* self, ArtMethod* method) {
  // `hotness_count_` should always be 0 for intrinsics (which are considered hot from the first
  // call), and for memory shared methods which use `shared_method_hotness`.
  DCHECK_IMPLIES(method->IsIntrinsic(), method->CounterIsHot());
  DCHECK_IMPLIES(method->IsMemorySharedMethod(), method->CounterIsHot());

  if (method->CounterIsHot()) {                 // 已到阈值
    if (method->IsMemorySharedMethod()) {
      if (!method->IsIntrinsic()) {
        if (self->DecrementSharedMethodHotness() == 0) {
          self->ResetSharedMethodHotness();
        } else {
          return;
        }
      }
    } else {
      method->ResetCounter(Runtime::Current()->GetJITOptions()->GetWarmupThreshold());
    }
    MaybeEnqueueCompilation(method, self);       // 过线 → 入队编译
  } else {
    method->UpdateCounter(1);                     // 没到阈值 → 计数 +1
  }
}
```
注意这是**倒计数 / 阈值比较**模型:没热就 `+1`,一旦 `CounterIsHot()` 就触发编译并重置。默认阈值是 `0xffff`:

```cpp
// 【真实源码 platform/art@runtime/jit/jit_options.cc, refs/heads/main, L27-L42】
static constexpr uint32_t kJitDefaultOptimizeThreshold = 0xffff;
// ...
static constexpr uint32_t kJitDefaultWarmupThreshold = 0xffff;
// ...
static constexpr size_t kDefaultInvokeTransitionWeightRatio = 500;
```
> **解读这几个数**:`optimize`/`warmup` 默认都是 `0xffff`(65535)——但**不要**理解成"必须跑满 65535 次才编译":调用点权重、循环回边、`UpdateCounter(0xf)`(nterp 慢查找一次加 15)、`InvokeTransitionWeightRatio=500`(解释↔编译切换的额外权重)等会让计数走得很快。**这正是 know 里那条教训的现场:spec/常量的"算法含义"≠ 用户/直觉以为的"跑 N 次"。** 真要解释清楚一个方法"为什么这么快/这么晚被 JIT",必须读 `UpdateCounter` 的全部调用点,而不是看一个阈值数字拍脑袋。

### 4.4 两档 JIT + 升到 optimized:hotness gate 全貌【真实源码】

过线后进 `MaybeEnqueueCompilation`,这是决定"编成哪一档"的总闸:

```cpp
// 【真实源码 platform/art@runtime/jit/jit.cc, refs/heads/main, L1654-L1714】
void Jit::MaybeEnqueueCompilation(ArtMethod* method, Thread* self) {
  if (thread_pool_ == nullptr) {
    return;
  }
  if (JitAtFirstUse()) {
    // Tests might request JIT on first use (compiled synchronously in the interpreter).
    return;
  }
  if (!UseJitCompilation()) {
    return;
  }
  if (IgnoreSamplesForMethod(method)) {
    return;
  }
  if (GetCodeCache()->ContainsPc(method->GetEntryPointFromQuickCompiledCode())) {
    if (!method->IsNative() && !code_cache_->IsOsrCompiled(method)) {
      // If we already have compiled code for it, nterp may be stuck in a loop.
      // Compile OSR.
      AddCompileTask(self, method, CompilationKind::kOsr);     // ← 见 §4.5 OSR
    }
    return;
  }
  // Check if we have precompiled this method.
  if (UNLIKELY(method->IsPreCompiled())) {
    // ...（命中 AOT 预编译,直接换 entrypoint,不再 JIT）
    return;
  }

  static constexpr size_t kIndividualSharedMethodHotnessThreshold = 0x3f;
  // Intrinsics are always in the boot image and considered hot.
  if (method->IsMemorySharedMethod() && !method->IsIntrinsic()) {
    // ...（boot image 里跨进程共享的方法用单独的更高门槛 0x3f,避免每个进程都去编）
  }

  if (!method->IsNative() && GetCodeCache()->CanAllocateProfilingInfo()) {
    AddCompileTask(self, method, CompilationKind::kBaseline);   // ← 默认先 baseline
  } else {
    AddCompileTask(self, method, CompilationKind::kOptimized);
  }
}
```
关键设计:**默认先编 `kBaseline`,不是 `kOptimized`。** baseline 编译快、不做激进优化,但**会插桩收集 profiling info**(类型反馈、调用频次)。等 baseline code 自己又跑到热阈值,再升一档到 optimized:

```cpp
// 【真实源码 platform/art@runtime/jit/jit.cc, refs/heads/main, L1397-L1419】
void Jit::EnqueueOptimizedCompilation(ArtMethod* method, Thread* self) {
  // ...
  // We arrive here after a baseline compiled code has reached its baseline
  // hotness threshold. If we're not only using the baseline compiler, enqueue a compilation
  // task that will compile optimize the method.
  if (!options_->UseBaselineCompiler()) {
    AddCompileTask(self, method, CompilationKind::kOptimized);
  }
}
```
**为什么要两档?** 这是经典 tiered compilation 思路:
- baseline 让方法**尽快**离开解释器(响应快、profile 立刻开始收集);
- optimized 拿 baseline 收集到的真实类型/频次做**激进内联、去虚化(devirtualization)**,质量高但编译贵;
- 只有"热到 baseline 都嫌不够"的少数方法才付 optimized 的编译成本。

### 4.5 OSR:循环里的方法怎么"换轮胎不停车"

上面 `MaybeEnqueueCompilation` 里出现的 `CompilationKind::kOsr` 是个容易踩坑的点。考虑 §3 那个 `sum(n)`:如果 `n` 很大,方法**一次都还没返回**,但循环已经转了几百万次——它在解释器里"卡在循环里"出不来。普通 JIT 升档发生在"下次进方法时",可这方法这次进去就不出来了。

**OSR(On-Stack Replacement)** 解决这个:在循环回边检测到热,JIT 一份**能从当前 dex_pc + 当前寄存器状态接管**的 OSR code,然后**在栈上**把正在解释执行的这一帧替换成编译帧,从循环中间无缝接管。`PrepareForOsr` 就是干这事的(`runtime/jit/jit.cc`,接收 `dex_pc` 和 `vregs` 数组)。

> **失败模式**:不可 OSR 的循环(比如解释器状态无法映射到编译帧)会一直解释跑完,表现为"某个长循环 CPU 占满但迟迟不见 JIT 生效"。

### 4.6 profile-guided AOT:把"今天学到的热"存下来给明天

JIT 的 profile 不只服务当前进程。**`profile_saver`** 把"哪些方法热、哪些类被加载、热调用点的类型"写进 app 私有目录的 profile 文件(`runtime/jit/profile_saver.cc`)。息屏充电时,后台 `dex2oat`(`dex2oat/dex2oat.cc`)以 **`--compiler-filter=speed-profile`** 读这份 profile,**只 AOT profile 里标热的方法**,产出 oat。这就把"运行时学到的 know-how"摊还成了"下次冷启动直接快"。

> 这也是 **Cloud Profiles / baseline profile** 的基础:Play Store 可以把"大多数用户的热路径"作为云端 profile 随 app 下发,于是你**第一次**打开 app 就已经有了"别人帮你跑热"的 AOT 代码。源码里 `StartProfileSaver(..., ref_profile_filename, ...)` 的 `ref_profile`(reference profile)就是为合并云端/历史 profile 留的口子(`runtime/jit/jit.h`)。

---

## 5. GC:三代演进的源码证据

GC 是本章最"系统"的部分。我们不堆概念,直接用源码把三代的**机制差异**钉死。

### 5.1 CC 的心脏:Baker 读屏障【真实源码】

Concurrent Copying 能"边搬边用"的全部魔法,在读屏障一个函数里。每次从堆里读一个对象引用,都(在开了 RB 的 build 上)走这段:

```cpp
// 【真实源码 platform/art@runtime/read_barrier-inl.h, refs/heads/main, L35-L71】
inline MirrorType* ReadBarrier::Barrier(
    mirror::Object* obj, MemberOffset offset, mirror::HeapReference<MirrorType>* ref_addr) {
  constexpr bool with_read_barrier = kReadBarrierOption == kWithReadBarrier;
  if (gUseReadBarrier && with_read_barrier) {
    // ...
    if (kUseBakerReadBarrier) {
      // fake_address_dependency (must be zero) is used to create artificial data dependency from
      // the is_gray load to the ref field (ptr) load to avoid needing a load-load barrier between
      // the two.
      uintptr_t fake_address_dependency;
      bool is_gray = IsGray(obj, &fake_address_dependency);          // ① 读对象的 RB 状态位
      // ...
      ref_addr = reinterpret_cast<mirror::HeapReference<MirrorType>*>(
          fake_address_dependency | reinterpret_cast<uintptr_t>(ref_addr));
      MirrorType* ref = ref_addr->template AsMirrorPtr<kIsVolatile>();
      MirrorType* old_ref = ref;
      if (is_gray) {                                                  // ② 灰 = 可能还在 from-space
        // Slow-path.
        ref = reinterpret_cast<MirrorType*>(Mark(ref));              // ③ 慢路径:确保搬到 to-space
        if (kAlwaysUpdateField && ref != old_ref) {
          obj->CasFieldObjectWithoutWriteBarrier<false, false>(offset,
                                                               old_ref, ref,
                                                               CASMode::kStrong,
                                                               std::memory_order_release);  // ④ 顺手修指针
        }
      }
      AssertToSpaceInvariant(obj, offset, ref);                       // ⑤ 不变式:返回的一定在 to-space
      return ref;
    }
    // ...
```
逐行讲清这套"无锁并发搬运":
- **① Baker 状态位(gray)**:每个对象有 RB 状态。GC 把可能要搬的对象标 gray。
- **② fast path / slow path**:绝大多数读命中 **fast path**(非 gray,直接返回引用,几乎零成本)。只有 gray 才进 slow path。这就是 CC"低暂停"的来源——**成本被摊到极少数读上**。
- **③ `Mark`**:slow path 确保被引用对象已经从 from-space 拷到 to-space(没搬就现搬),返回 to-space 的新地址。
- **④ self-healing**:顺手把字段里的旧指针 CAS 成新指针,下次读这个字段就走 fast path 了。GC 是"自愈"地推进的。
- **⑤ to-space invariant**:这是整套算法的正确性内核——**mutator 永远不会拿到 from-space 的引用**,所以可以一边搬一边让 app 跑。
- **`fake_address_dependency`(注释里那句)**:一个精妙的工程技巧——用一个恒为 0 的"假地址依赖"在 `is_gray` 的 load 和 ref 的 load 之间制造 **data dependency**,从而**省掉一条 load-load 内存屏障**(ARM 弱内存序下屏障很贵)。这是"用 CPU 体系结构特性换性能"的教科书级例子。

回到 §4.2 的伏笔:nterp 在 arm64 上要 `kReserveMarkingRegister`——就是为了让汇编解释器能廉价地参与这套读屏障(marking register 常驻一个寄存器标记"GC 正在 marking"),**这就是 GC 设计反过来吃掉一个通用寄存器、并约束解释器能否启用的根因**。

### 5.2 CC 的"一个停顿":RunPhases 编排【真实源码】

CC 整个周期里**只有一次**真正的全局停顿(flip,跟堆大小无关的常数),其余阶段并发。编排一目了然:

```cpp
// 【真实源码 platform/art@runtime/gc/collector/concurrent_copying.cc, refs/heads/main, L213-L265】
void ConcurrentCopying::RunPhases() {
  CHECK(kUseBakerReadBarrier || kUseTableLookupReadBarrier);   // 没有读屏障就不可能并发拷贝
  // ...
  {
    ReaderMutexLock mu(self, *Locks::mutator_lock_);            // 注意:Reader 锁 = 不阻塞 mutator
    InitializePhase();
    if (use_generational_cc_ && !young_gen_ && !force_evacuate_all_) {
      MarkingPhase();                                            // generational:full GC 才标记
    }
  }
  if (kUseBakerReadBarrier && kGrayDirtyImmuneObjects) {
    ActivateReadBarrierEntrypoints();                            // 切到 RB 入口,让 mutator 走慢路径
    ReaderMutexLock mu(self, *Locks::mutator_lock_);
    GrayAllDirtyImmuneObjects();
  }
  FlipThreadRoots();                                             // ★ 唯一的全局停顿:flip 根
  {
    ReaderMutexLock mu(self, *Locks::mutator_lock_);
    CopyingPhase();                                              // 并发拷贝(mutator 同时在跑)
  }
  // ...
  {
    ReaderMutexLock mu(self, *Locks::mutator_lock_);
    ReclaimPhase();                                              // 并发回收 from-space
  }
  FinishPhase();
  // ...
}
```
读这段最该注意的不是函数名,是 **`ReaderMutexLock`**:GC 拿的是 mutator_lock 的**读锁**,意味着 **mutator 线程同时还在跑**(它们也持读锁)。真正让所有线程停下的只有 `FlipThreadRoots()` 这一下——把各线程根从 from-space 翻到 to-space。这就是官方文档说的"only one small pause, constant in time with regards to the heap size"。

`young_gen_` 那个分支就是 **Generational CC**(O10+ 默认):young GC 跳过 full marking,只扫新生代,大幅降低频繁小回收的成本。

### 5.3 CMC:userfaultfd 把"并发整理"外包给内核【真实源码 + 设计动机】

**当前(O13/T+)默认 GC 不是 CC,是 CMC(Concurrent Mark-Compact)。** 源码层面它是另一个 collector:

```cpp
// 【真实源码 platform/art@runtime/gc/collector/mark_compact.h, refs/heads/main, L115-L149】
class MarkCompact final : public GarbageCollector {
 public:
  using SigbusCounterType = uint32_t;
  // ...
  // Fake file descriptor for fall back mode (when uffd isn't available)
  static constexpr int kFallbackMode = -3;
  // ...
  void RunPhases() override REQUIRES(!Locks::mutator_lock_, !lock_);
  // ...
  // Called by SIGBUS handler. NO_THREAD_SAFETY_ANALYSIS for mutator-lock, which
  // is asserted in the function.
  bool SigbusHandler(siginfo_t* info) REQUIRES(!lock_) NO_THREAD_SAFETY_ANALYSIS;   // ← 缺页驱动
  // ...
  CollectorType GetCollectorType() const override {
    return kCollectorTypeCMC;                                                        // ← 标识
  }
```
```cpp
// 【真实源码 同文件,字段节选 L930, L937-L939】
  int uffd_;                       // userfaultfd 的 fd
  // ...
  // Flag indicating whether one-time uffd initialization has been done.
  // Its purpose is to minimize the userfaultfd overhead to the minimal in ...
```
机制(结合源码注释 + 设计动机)讲清它和 CC 的本质区别:

| 维度 | Concurrent Copying (CC) | Concurrent Mark-Compact (CMC) |
|---|---|---|
| 整理方式 | **copying**:from-space → to-space | **compaction**:原地滑动压缩 |
| 额外内存 | 需要一块 to-space(~存活区大小) | **不需要 to-space**,省 RSS |
| 并发手段 | **读屏障**(每次读引用拦截) | **userfaultfd 缺页**(访问未整理页才拦截) |
| 拦截粒度 | 每个对象引用读 | 每个**页**(page) |
| 依赖 | CPU 弱内存序 + RB 寄存器 | 内核 uffd + SIGBUS / `UFFDIO_*` |

**CMC 的执行直觉**:mark 阶段照常(并发标记可达对象);compact 阶段把 moving-space 注册进 `userfaultfd`,然后 GC 线程开始把对象往低地址滑动压缩。**mutator 此时若访问一个还没整理完的页,会触发缺页 → SIGBUS → `SigbusHandler` 现场把那一页整理好、映射回去,mutator 再继续。** 于是"并发整理"不再靠每次引用读的屏障,而是靠"**只有真正踩到未整理页时才付一次页级别的代价**"。

**为什么要换**(设计动机):CC 是 copying,堆里必须长期留出 to-space,**RSS 开销在低端/内存紧张设备上很痛**。mark-compact 原地压缩没有 to-space;但传统 mark-compact 的 compaction 阶段要 STW——userfaultfd 正好把"并发"这件事用内核缺页机制补回来,鱼和熊掌兼得。

> **出处与诚实标注**:
> - "O13 用 userfaultfd 新 GC":Wikipedia(WebFetch 逐句)+ 源码 `kCollectorTypeCMC`/`uffd_`/`SigbusHandler` 印证。
> - "CMC 默认范围(T+,后回移到 S+)":来自社区/issue 检索(WebSearch),**具体回移版本范围「待核」**。
> - **量化的 RSS 节省数字「待核」**:Google 在 tdcommons 有一篇 "Utilizing the Linux Userfaultfd System Call in a Compaction Phase of a GC" 的技术披露,但本次 WebFetch 取该 PDF 返回 **403**,未能取得逐字数字,故不杜撰具体百分比。
> - **source.android.com 公开 gc-debug 文档当前未收录 CMC**(WebFetch 实测),属文档滞后于源码,读者排障时以 logcat 里实际 collector 名为准。

---

## 6. ⭐ Demo B:在设备/模拟器上观测 AOT/JIT/GC(需 Android 环境)

Demo A 已在主机闭环了 dex 静态结构。要观测**运行时**的 AOT/JIT/GC 行为,必须有 Android 环境。下面命令给全,并逐条标前置;**本机当前未连设备/模拟器,故标注为"需 Android 模拟器/真机验证",输出为基于机制的预期格式**(adb 命令本身均为真实可用命令)。

> **前置**:① 已装 Android SDK(本机已确认 `adb` = `Android Debug Bridge version 1.0.41 / 37.0.0`);② 一台 `adb devices` 能列出的设备或模拟器(API 31+/Android 12+ 才有 nterp/CMC);③ 部分 `cmd package compile` / `dexdump` 需要;④ 观测 `userfaultfd` GC 需 Android 13+。

### 6.1 看一个 app 当前的编译状态(AOT 有没有生效)

```bash
adb shell dumpsys package dexopt | grep -A3 <你的包名>
# 或针对单包:
adb shell dumpsys package <pkg> | grep -i -A2 "compiler\|dexopt\|status"
```
**预期(格式示例,需设备核实)**:每个 dex 一行,形如
```
[status=speed-profile] [reason=bg-dexopt]
```
`status` 就是 compiler-filter:`verify`(只校验、纯解释/nterp + JIT)、`speed-profile`(profile-guided AOT,最常见)、`speed`(全量 AOT)。**这直接验证 §1.2 / §4.6 的混合编译模型在真机上的落地。**

### 6.2 手动触发 profile-guided AOT,再退回

```bash
# 强制按 profile 做 AOT(等价于半夜 bg-dexopt 干的事)
adb shell cmd package compile -m speed-profile -f <pkg>
# 退回只校验(清掉 AOT 产物,逼它回到解释 + JIT)
adb shell cmd package compile -m verify   -f <pkg>
adb shell pm art clear-app-profiles <pkg>   # 清 profile(Android 13+ 的 `pm art` 子命令)
```
**预期效果**:`-m speed-profile` 后该 app 冷启动更快(热方法已 AOT);`-m verify` 后第一次启动明显变慢(退回解释 + 现场 JIT),正是 §4 的"逐级升档"重新走一遍。**这是把"档位切换"做成可手动复现实验的最直接方式。**

### 6.3 观测 GC 类型与暂停(确认 CC vs CMC)

```bash
# 让 ART 打印每次 GC(也可通过 -Xgc 配),抓 logcat:
adb logcat -c && adb logcat | grep -i "art\|GC"
# 触发一次:
adb shell am dumpheap <pkg> /data/local/tmp/h.hprof   # 制造内存活动
```
**预期 GC 日志行格式**(来自官方 gc-debug 文档 WebFetch 的真实示例):
```
... young concurrent copying paused: Sum: 229.314ms 99% C.I. 37us-2287.499us Avg: 334.764us Max: 6850us
```
**判定 collector**:看这行的 collector 名:
- `concurrent copying` / `young concurrent copying` → **CC / Generational CC**(O8–O12 典型)。
- `concurrent mark compact`(或 logcat 中 CMC 相关字样)→ **CMC**(O13+ 默认)。
> 注:`young concurrent copying paused: ... Max: 6850us` 这条**暂停统计**直接对应 §5.2 里 `FlipThreadRoots()` 那"唯一停顿"——`Max` 就是最坏 flip 停顿。**能在真机日志里指认出这一个数字对应源码哪一行,才算真读懂了 CC。**

### 6.4 (进阶,需 root/userdebug)看 oat 反汇编

```bash
# oatdump 是设备侧/userdebug 镜像里的工具(普通 build-tools 不含)
adb shell oatdump --oat-file=/data/app/.../base.odex | less
```
**预期**:能看到 oat header(magic `oat\n` + 版本)、每个 dex、每个被 AOT 的方法的 **arm64 机器码**;没被 profile 标热的方法则只有"走解释/JIT"的入口桩。**这是 §4.6 "只编热方法"在二进制层的铁证。** `普通 build-tools 不含 oatdump/dex2oat`(本机已确认),故此步必须设备侧。

> **demo 总结对照**:Demo A(主机,实测真实输出)给"dex 是什么";Demo B(设备,命令真实/输出待设备核)给"AOT/JIT/GC 怎么动"。两者合起来覆盖本章全部机制,且每条命令都能在对应环境复现。

---

## 7. 方案对比:什么时候用哪一档 / 哪种 GC

### 7.1 执行档位选择(compiler-filter)

| filter | 行为 | 适用场景 | 不适用边界 |
|---|---|---|---|
| `verify` | 只校验 dex,运行时 nterp/解释 + JIT | 默认装机态、很少用的 app、省空间 | 启动密集型 app 冷启动会慢;CPU 敏感型首跑抖动 |
| `speed-profile` | 按 profile 只 AOT 热方法 | **绝大多数 app 的稳态**(bg-dexopt 产物) | 没有 profile(全新装、清过 profile)时退化成接近 verify |
| `speed` | 全量 AOT | 系统核心 app、明确要极致启动速度 | 占用大、OTA 后全失效要重编、冷方法白编 |
| `everything` | 连不可验证的也编 | 调试/特殊场景 | 体积爆炸,生产不用 |

### 7.2 GC 选择

| GC | 选它的场景 | 不适用边界 |
|---|---|---|
| **CMS** | 极老设备 / 特殊兼容;`-Xgc:CMS` | 会碎片化,大对象分配易失败;compaction 要 STW;基本已淘汰 |
| **CC / Generational CC** | O8–O12 默认;读屏障开销可接受、内存不是最紧 | 需 to-space,**RSS 高**;弱内存序架构上读屏障有成本 |
| **CMC(userfaultfd)** | O13+ 默认;**内存敏感**设备首选 | 依赖内核 uffd 特性;部分被 SELinux/Knox 限制的设备(如某些三星机型 issue)`UFFDIO_MOVE` 不可用 → fallback 模式 |

> **具体场景**:做一个内存紧张的低端机出海 app,堆经常逼近上限——CC 的 to-space 会让你更早 OOM/被 LMK 杀,**CMC 的"无 to-space 原地压缩"能实打实降 RSS**;但如果目标设备是被改过内核/SELinux 收紧 uffd 的奇葩 ROM,要预案 CMC 退回 fallback 后的暂停回升。

---

## 8. 扎根:失败模式 / 生产真坑 / 根因

> 五件套(failure / 真坑 / 根因 + 排查 + 防御),全部锚定本章源码。

### 8.1 失败模式

1. **"我明明优化了算法,profiler 却看不到 JIT 生效"**
   - 现象:某长循环 CPU 占满,迟迟跑的是解释/baseline。
   - 根因:① 方法**一直没返回**(困在循环),普通升档发生在"下次进方法",得靠 **OSR**(§4.5);若该循环不可 OSR,就一路解释到底。② build 是 `debuggable` / 开了 instrumentation → 整条链从 **nterp 掉到 switch-interpreter**(§4.2 `CanRuntimeUseNterp` 返回 false),还可能压根不上 JIT。
   - 排查:`adb shell dumpsys package <pkg> | grep debuggable`;确认 release build;`adb logcat | grep -i jit`。

2. **"release 包性能正常,排查时换 debuggable 包性能断崖"**
   - 根因:不止是少了 R8/AOT——`IsJavaDebuggable()` 为真直接禁用 nterp,执行引擎掉档(§4.2)。
   - 防御:性能问题永远在 **release/profileable** build 上量,别用 debuggable build 下结论。`<profileable android:shell="true"/>` 是兼顾可观测与真实性能的正解。

### 8.2 生产真坑

3. **OTA / 系统升级后首启巨慢(纯 AOT 时代的幽灵)**
   - 根因:boot image / boot classpath 变更使**所有 app 的 oat 失效**,需重编(§1.2)。混合编译时代被 bg-dexopt 摊还缓解,但**首次升级后那次冷启动仍会退回 verify 态**(profile 在、oat 没了)。
   - 防御:出货前做 OTA 后冷启动验收;依赖 **Cloud/baseline profile**(§4.6)让首启就有 AOT 热路径。

4. **CMC 在特定设备上崩溃 / `UFFDIO_MOVE` unsupported**
   - 现象:部分三星等 ROM 因 Knox/SELinux 限制 uffd 的 `MOVE` ioctl,触发 native crash 或 GC 异常(社区 issue 实有记录,WebSearch 命中)。
   - 根因:CMC 依赖内核 uffd 特性(§5.3),被安全策略 ban 后行为退化。
   - 防御:`mark_compact.h` 里就有 `kFallbackMode`(uffd 不可用时回落)——若你在 native 层也用 uffd(如某些加固/反调试),要做能力探测,别假设 uffd 一定可用。

5. **GC 暂停尖刺(Max pause 异常大)**
   - 现象:logcat 里 `... paused: ... Max: <很大>us`。
   - 根因:CC 的"唯一停顿"`FlipThreadRoots()`(§5.2)本应是常数小停顿;若 `Max` 异常,常见是**线程数暴多**(flip 要遍历所有线程根)或**被 mutator 长时间持锁卡住 flip**。
   - 排查:`adb logcat | grep "concurrent copying paused"` 看 `Max`/`C.I.`;查线程数、是否有 JNI 长临界区。

### 8.3 共同根因总结

本章五件套指向**同一个系统性根因**:**ART 是"可降级的多档执行机",任何让它停在低档的因素(debuggable、不可 OSR、profile/oat 失效、GC 能力被 ban)都表现为性能/稳定性问题,而表象千差万别。** 排查 ART 性能问题的正确起手式不是"看哪段代码慢",而是**先确认它现在跑在哪一档**(compiler-filter? nterp 还是 switch? JIT 生效没? 哪种 GC?),再回到本章对应源码定位降档原因。

---

## 9. 章末五件套(本章可复用沉淀)

```yaml
- when: 排查 Android 上某段 Java/Kotlin 代码的性能(CPU 占用高 / 启动慢 / 抖动)
  avoid: 直接盯着"哪行代码慢"或在 debuggable build 上下结论
  how: 先确认它当前跑在哪一档——dumpsys package <pkg> 看 compiler-filter(verify/speed-profile/speed);
       确认是 release/profileable build(debuggable 会禁 nterp 掉到 switch-interpreter);
       logcat grep jit 看 JIT 是否生效;长循环慢优先怀疑 OSR 没生效
  until: 已用 dumpsys/logcat 定位到"卡在解释/baseline 的具体原因",再谈代码优化

- when: 解读 ART 的 JIT 阈值 / hotness 常量(如 0xffff、UpdateCounter(0xf)、InvokeTransitionWeightRatio=500)
  avoid: 把"阈值=必须调用 N 次"当事实拿去解释/调参
  how: 读全部 UpdateCounter / AddSamples 调用点(回边、慢查找、解释↔编译切换各有不同权重);
       区分"算法含义"(计数器怎么走)与"直觉含义"(以为跑 N 次);用 Demo B 实测验证而非读单个常量
  until: 在目标 Android 版本上用真机实验复现了该方法被 JIT 的实际触发条件

- when: 写"Android GC 默认是 X"这类结论(尤其引用二手资料)
  avoid: 停在"CC 是默认"——这只到 Android 12;O13+ 默认已是 userfaultfd 的 CMC,且公开文档滞后于源码
  how: 以源码 collector 名为准(kCollectorTypeCC / kCollectorTypeCMC),logcat 里看实际 collector 字符串;
       涉及内存 RSS / to-space 取舍时,明确区分 copying(CC,要 to-space)与 mark-compact(CMC,原地)
  until: 在目标设备 logcat 里亲眼确认了实际运行的 collector

- when: 为内存紧张 / 出海低端机做内存与 GC 取舍
  avoid: 默认 CC 一定最好;忽视 to-space 的 RSS 成本与 uffd 的设备兼容性
  how: 内存敏感优先 CMC(无 to-space,降 RSS);但对被 SELinux/Knox 限制 uffd 的设备(如部分三星机型)
       预案 CMC fallback 后的暂停回升;若 native 层也用 uffd 必须做能力探测
  until: 在目标设备矩阵上实测了 CMC 生效/fallback 两种路径的 RSS 与暂停

- when: 引用 AOSP/ART 源码或机制结论写进文档/PR
  avoid: 凭记忆写源码片段、版本史、阈值数字
  how: 实际抓 android.googlesource.com/platform/art 对应文件(?format=TEXT 取 base64 解码得逐字源码),
       标 repo@path@分支;取不到的(大文件/403)明确标「待核」,绝不用"大概是"补全
  until: 该结论有源码逐字引用或官方文档逐句支撑
```

---

### 附:本章源码与史料出处清单(均实际 WebFetch / curl 取得,非凭记忆)

**逐字源码(`android.googlesource.com/platform/art` @ `refs/heads/main`,经 `?format=TEXT` base64 解码):**
- `libdexfile/dex/dex_file.h`(DEX `struct Header`、magic/checksum/signature)
- `libdexfile/dex/standard_dex_file.h`(StandardDexFile)
- `runtime/interpreter/interpreter.cc`(`Execute` 的解释→JIT dispatch)
- `runtime/interpreter/mterp/nterp.cc`(nterp 可用条件、UpdateHotness)
- `runtime/jit/jit-inl.h`(`AddSamples` hotness 计数)
- `runtime/jit/jit.cc`(`MaybeEnqueueCompilation`、`EnqueueOptimizedCompilation` 两档 JIT)
- `runtime/jit/jit_options.cc`(默认阈值 `0xffff`、`InvokeTransitionWeightRatio=500`)
- `runtime/read_barrier-inl.h`(Baker 读屏障 `ReadBarrier::Barrier`)
- `runtime/gc/collector/concurrent_copying.cc` / `.h`(CC `RunPhases`、单停顿)
- `runtime/gc/collector/mark_compact.h`(CMC `kCollectorTypeCMC`、`SigbusHandler`、`uffd_`)
- (并抓取 `oat_file.h`、`region_space.h`、`profile_saver.cc`、`dex2oat.cc`、`quick_trampoline_entrypoints.cc` 作交叉印证)

**设计史料 / 官方文档(WebFetch):**
- Wikipedia "Android Runtime"(KitKat 预览 / Lollipop 取代 Dalvik / Nougat 混合 JIT/AOT + OpenJDK / O13 userfaultfd GC)
- source.android.com/docs/core/runtime/jit-compiler(JIT + profile + dex2oat 流程)
- source.android.com/docs/core/runtime/gc-debug(CC/CMS、默认版本、GC 日志行格式;**实测未收录 CMC**)
- source.android.com/docs/core/runtime/dex-format(DEX_FILE_MAGIC、ENDIAN_CONSTANT、header_item)
- WebSearch:O13 userfaultfd CMC、CMC 默认范围、三星 uffd `MOVE` 兼容 issue(部分为社区来源,具体数字「待核」)

**主机实测工具链(Demo A 真实输出来源):** Android SDK build-tools `36.1.0`(`d8` 9.0.3、`dexdump` from AOSP)、JDK `javac`/`java`;`adb` 1.0.41 / 37.0.0(Demo B 命令真实,输出待设备核实)。
