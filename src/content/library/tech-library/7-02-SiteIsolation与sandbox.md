---
title: "Site Isolation 与 sandbox（Chromium 域）"
slug: "7-02"
collection: "tech-library"
group: "chromium内核"
order: 7002
summary: "前置依赖：已读第 1 章（多进程架构与进程模型，掌握 SiteInstance → SiteInfo → ProcessLock 决策链、OOPIF 基本概念、Browser/Renderer/GPU 进程分工）。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T10:47:03.000Z"
updatedAt: "2026-06-12T10:47:03.000Z"
---
> **前置依赖**：已读第 1 章（多进程架构与进程模型，掌握 SiteInstance → SiteInfo → ProcessLock 决策链、OOPIF 基本概念、Browser/Renderer/GPU 进程分工）。熟悉操作系统进程隔离、虚拟地址空间、Linux capabilities/namespaces/seccomp 基础、Windows token/integrity level 基础、CPU cache 与乱序/推测执行（speculative execution）基本原理、Same-Origin Policy（SOP）。
>
> **TL;DR**：第 1 章讲的是「进程怎么分」，本章讲的是「为什么必须这么分，以及分完之后怎么真正挡住攻击」。Site Isolation 与 sandbox 是 Chromium 安全模型的两条正交防线：**Site Isolation 是横向隔离**（把跨站数据从同一地址空间里赶出去，对抗 Spectre + UXSS + compromised renderer），**sandbox 是纵向降权**（即使 renderer 被完全攻陷，也碰不到 OS）。Spectre 的致命假设是「active web content 能读到 renderer 进程地址空间内任意数据，SOP 在进程内形同虚设」——这一句话直接推翻了「靠进程内检查保护跨站数据」的整个旧模型，逼出了 site-per-process。enforcement 的核心不在 renderer 而在 browser：renderer 声称的 origin **不可信**，browser 用 `CanAccessDataForOrigin`（Jail + Citadel 两类检查）拿进程的 `ProcessLock` 去校验每一条 IPC，对说谎的 renderer 直接 kill。sandbox 在 Windows 是 token/job/integrity/desktop 四件套 + broker 代理，在 Linux 是 Layer-1 语义层（user namespace/setuid）+ Layer-2 攻击面削减层（seccomp-bpf）。cross-origin isolation（COOP+COEP）是 Web 平台层把高精度 timer / SharedArrayBuffer 与 Spectre 风险绑定的开关。Electron 默认（≥20）开启 renderer sandbox，并用 Fuses 在打包期做二进制级硬化。

---

## 1. 设计考古：从 SOP 失效到地址空间隔离

### 1.1 旧模型：进程内 Same-Origin Policy

在 Site Isolation 之前（也是大多数工程师脑中的默认模型），浏览器的安全边界是**进程内的逻辑检查**。一个 renderer 进程可以同时持有 `a.com` 和 `b.com` 的文档，靠 Blink 里的 SecurityOrigin 比较来阻止 `a.com` 的脚本读 `b.com` 的 DOM。跨站资源（HTML/JSON/XML）可以进同一个进程的内存，SOP 只是在「JS 试图访问」那一刻拦一道。

这个模型有两个隐含前提：

1. **renderer 进程是可信的执行环境** —— Blink 的 SOP 检查不会被绕过。
2. **进程内的内存读取受 SOP 约束** —— 即使数据在内存里，没有 SOP 许可就读不到。

两个前提在 2018 年同时被击穿。

### 1.2 Spectre：把「数据在内存里」直接等价于「数据被泄露」

2018 年 1 月披露的 Spectre（CVE-2017-5753 / 5715）与 Meltdown（CVE-2017-5754）是 CPU 微架构级漏洞，不是软件 bug——打补丁也补不干净。其对浏览器威胁模型的颠覆，Chromium 官方 side-channel 威胁模型文档写得极其直白：

> "We must assume that *active web content* (JavaScript, WebAssembly, Native Client, Flash, PDFium, …) will be able to read any and all data in the address space of the process that hosts it."
>
> 【真实源码 chromium/src@docs/security/side-channel-threat-model.md，已 WebFetch 核实】

注意这句话的分量：它不是说「有个 bug 能读内存」，而是说**必须假设 active web content 总能读到宿主进程地址空间的任意数据**。直接推论：

> "We must consider any data that gets into a renderer process to have no confidentiality from any origins running in that process, regardless of the same origin policy."
>
> 【真实源码 chromium/src@docs/security/side-channel-threat-model.md，已 WebFetch 核实】

> "In this new mental model, we have to assume that user code can reliably gain access to all data within a renderer process through speculation."
>
> 【真实源码 chromium/src@docs/security/side-channel-threat-model.md，已 WebFetch 核实】

**SOP 在进程内彻底失效**。前提 2 没了——只要 `b.com` 的字节进了 `a.com` 的进程内存，`a.com` 的 JS 用 Spectre gadget 就能读出来，SOP 那道逻辑检查管不到 CPU 的推测执行。

### 1.3 结论：把数据赶出地址空间，而不是在进程内拦

既然进程内检查保不住，唯一的解法是**让敏感跨站数据根本不进入运行不可信代码的地址空间**。Chromium 把这条原则写成了 Site Isolation 的设计纲领：

> "The plan going forward must be to keep sensitive cross-origin data out of address spaces that run untrustworthy code, rather than relying on in-process checks."
>
> 【真实源码 chromium/src@docs/security/side-channel-threat-model.md，已 WebFetch 核实】

> Site Isolation 让 "the web security model (the same-origin policy) align with the underlying platform's security model (separate address spaces and privilege reduction)."
>
> 【真实源码 chromium/src@docs/security/side-channel-threat-model.md，已 WebFetch 核实】

这句话点出了本章两条主线的关系：

- **separate address spaces** = Site Isolation（横向把站点摊到不同进程）。
- **privilege reduction** = sandbox（纵向把每个进程降权）。

### 1.4 不止 Spectre：三类威胁共同驱动

Site Isolation 在 Chrome 67（2018-05 桌面默认开启）落地，但它防的不只是 Spectre。官方 Site Isolation 页面列了三类威胁，前提 1（renderer 可信）也是被主动放弃的：

1. **Compromised renderer**：Chrome 团队坦承「determined attackers will be able to find a way to compromise a renderer process」，每个大版本都有 10–15 个潜在可利用的 renderer bug。【来源 chromium.org/Home/chromium-security/site-isolation/，已 WebFetch 核实】
2. **UXSS（Universal XSS）**：能在 renderer 内绕过 SOP 的 bug，「common」。
3. **Side-channel（Spectre 类）**：无需 Chrome bug 也能读内存。

到 Chrome 77（2019），桌面把保护扩展到「完全攻陷的 renderer」并隔离所有 extension，Android 也上了 partial Site Isolation（只隔离用户登录过的站点）。这就是为什么 enforcement 必须假设 renderer 会说谎（见 §3）——三类威胁里有两类都意味着 renderer 本身已经在攻击者手里。

### 1.5 历史脉络一图

```
2009  Reis et al. 提出 process-per-site 雏形（按 web origin 隔离页面）
      —— Site Isolation 学术起点
2013-05  Chrome 团队公布 OOPIF（out-of-process iframe）计划
         —— 跨站 iframe 必须能放到独立进程，是 site-per-process 的工程前提
2017-06  --isolate-extensions 上 Stable（M56/M57 阶段），OOPIF 首次实用
         —— 把 web iframe 赶出特权 extension 进程
2018-01  Spectre / Meltdown 披露
         —— 威胁模型被颠覆，进程内 SOP 失效
2018-05  Chrome 67 桌面默认开启 Site Isolation（site-per-process）
         —— OOPIF 对所有网页生效
2019     Chrome 77 桌面扩展到 compromised renderer + 隔离所有 extension；
         Android 上 partial Site Isolation
2021-03  Firefox 公布 Project Fission（自家 Site Isolation）
```

工程成本：Wikipedia 引述「more than 4000 commits from 320 contributors over a period of 5 years」，内存开销「10 to 13 percent」，CPU「one to two cores more」。【来源 en.wikipedia.org/wiki/Site_isolation，已 WebFetch 核实】这组数字解释了为什么 Android 至今只做 partial——内存预算扛不住全量。

> ⚠ 待核：Reis et al. 2009 论文的精确标题/会议，与「M56 vs M57 isolate-extensions 首发版本号」细节我未逐字核到原始 commit，正文按官方 OOPIF 文档与 Wikipedia 转述，精确版本号请以 chromestatus 为准。

---

## 2. Site Isolation 的内核抽象：principal / SiteInstance / ProcessLock

第 1 章已讲过分配链，本章只补**安全语义**：这三层抽象到底在「保护什么数据不被谁读」。

### 2.1 principal（SiteInfo）：进程能碰哪些数据

principal 回答「这个进程被授权读/写哪些数据」。粒度是 **scheme + eTLD+1**（即「site」），不是 origin。为什么不用更细的 origin？

> "Any two documents with the same principal in the same browsing context group must live in the same process, because they have synchronous access to each other's content."
>
> 【真实源码 chromium/src@docs/process_model_and_site_isolation.md，已 WebFetch 核实】

synchronous access 的根源是 `document.domain`：`a.example.com` 和 `b.example.com` 可以各自把 `document.domain` 设成 `example.com` 来同步互访。既然它们能同步访问彼此内存，就**必须**在同一进程，否则 OOPIF 的异步 proxy 模型根本无法支撑同步 DOM 访问。所以 site（eTLD+1）是「能同步互访的最大集合」的安全下界。

> 注意 origin 级隔离是**可选加强**，不是默认：`--isolate-origins=https://accounts.example.com`、`Origin-Agent-Cluster: ?1` HTTP header、企业策略、或 `ContentBrowserClient::GetOriginsRequiringDedicatedProcess` 内建配置。【来源 chromium/src@docs/process_model_and_site_isolation.md】

### 2.2 SiteInstance：同进程的最小文档组

`SiteInstance` 是分配单元。其头注释把安全语义写得很清楚：

```cpp
// content/public/browser/site_instance.h
// a SiteInstance represents a group of documents and workers that can
// share memory with each other, and thus must live in the same renderer
// process.
class CONTENT_EXPORT SiteInstance : public base::RefCounted<SiteInstance> {
```
【真实源码 chromium/src@content/public/browser/site_instance.h，已 WebFetch 核实接口与注释；下方逐条方法签名为真实摘录】

关键方法（安全相关）：

```cpp
  virtual bool IsSameSiteWithURL(const GURL& url) = 0;   // url 是否与本 SiteInstance 同 site
  virtual bool RequiresDedicatedProcess() = 0;           // 是否必须独占进程（即必须 locked）
  virtual bool IsRelatedSiteInstance(const SiteInstance* instance) = 0; // 是否同 BrowsingInstance
  virtual BrowsingInstanceId GetBrowsingInstanceId() = 0;
```
【真实源码 chromium/src@content/public/browser/site_instance.h，已 WebFetch 核实】

「a group of documents and workers that can share memory」直接对应 §1.2 的威胁模型——**能共享内存 = Spectre 下能互读 = 必须同 principal**。SiteInstance 就是把「Spectre 可达的内存域」与「进程」对齐的那个单位。

### 2.3 ProcessLock：进程的「数据访问执照」

`ProcessLock` 是 enforcement 的钥匙。它给 `RenderProcessHost` 贴一张「这个进程只准加载/访问哪个 site 的数据」的执照：

> ProcessLock is "a core part of Site Isolation, which is used to determine which documents are allowed to load in a process and which site data the process is allowed to access, based on the SiteInfo principal."
>
> 【真实源码 chromium/src@content/browser/process_lock.h，已 WebFetch 核实】

三种状态：

| 状态 | 含义 | 何时 |
|---|---|---|
| **invalid** | 没有任何 SiteInstance 关联，不授权访问任何东西 | 进程刚创建、尚未承载内容 |
| **locked-to-site** | 锁定单一 site，不能访问其它 site 数据 | 关联了 `RequiresDedicatedProcess()` 的 SiteInstance |
| **allow-any-site** | 不锁 site，任意 site 可 commit | SiteInstance 不要求独占（如 Android 上低风险站点共享进程） |

关键方法（真实摘录）：

```cpp
// content/browser/process_lock.h
static ProcessLock CreateAllowAnySite(...);  // 造一个不锁 site 的 lock
static ProcessLock Create(...);              // 为某 UrlInfo 造 lock
bool AllowsAnySite() const;                  // 是否 allow-any-site
bool IsLockedToSite() const;                 // 是否锁定单一 site
GURL GetProcessLockURL() const;              // 取关联 URL
```
【真实源码 chromium/src@content/browser/process_lock.h，已 WebFetch 核实】

lock 的赋值时机至关重要（决定了它能不能挡住「加载前就越权」）：

> Locks are assigned "before any content is loaded in a renderer process, either at the start of a navigation or at OnResponseStarted time, just before a navigation commits."
>
> 【来源 chromium/src@docs/process_model_and_site_isolation.md，已 WebFetch 核实】

**在内容加载前就锁定**——这样 browser 在 response 还没 commit 时，就已经知道「这个进程只配看 `a.com`」，后面任何声称是 `b.com` 的 IPC 都能当场识破。

---

## 3. enforcement：为什么不信 renderer，以及 browser 怎么验

这是本章最核心、也最容易被误解的一节。很多人以为 Site Isolation = 「把站点放到不同进程就完事」。**错。** 放进程只是把数据隔开；真正挡住攻击的是 browser 进程**对每一条来自 renderer 的 IPC 做独立校验**。因为威胁模型里 renderer 可能已经被攻陷（§1.4），它声称的一切都不可信。

### 3.1 第一原则：renderer 说的 origin 不算数

> 安全决策必须 "making security decisions based on trustworthy knowledge, calculated within the privileged browser process"。
>
> 【真实源码 chromium/src@docs/security/compromised-renderers.md，已 WebFetch 核实】

browser 不读 renderer 自报的 origin，而是用自己算出来的权威状态去对照：

- `RenderFrameHost::GetLastCommittedOrigin()` —— browser 侧记录的「这个 frame 实际 commit 的 origin」。
- `RenderFrameHostImpl::CanCommitOriginAndUrl(...)` —— navigation commit 前，校验 renderer 想 commit 的 (origin, url) 是否合法。

整体是 **defense in depth**：

> "our threat model must use a 'defense in depth' approach to limit the damage that occurs if an attacker finds a way around the Same Origin Policy"
>
> 【真实源码 chromium/src@docs/security/compromised-renderers.md，已 WebFetch 核实】

### 3.2 核心闸口：CanAccessDataForOrigin

每当 renderer 通过 IPC 请求某 origin 的数据（读 storage、blob URL、cookie、postMessage 目标等），browser 调 `ChildProcessSecurityPolicyImpl::CanAccessDataForOrigin`：

```cpp
// content/browser/child_process_security_policy_impl.h
// Handle::CanAccessDataForOrigin:
// Returns true if the process is permitted to read and modify the data
// for the given `origin`.
```
【真实源码 chromium/src@content/browser/child_process_security_policy_impl.h，已 WebFetch 核实该注释】

它内部的实现核心是把**进程实际的 ProcessLock** 与**从请求 URL 重新算出的期望 ProcessLock** 对照，这套逻辑在私有方法里叫 **Jail / Citadel 检查**：

> `PerformJailAndCitadelChecks`：performing security checks by "comparing the actual ProcessLock of the process... to an expected ProcessLock computed from `url`"。
>
> 【真实源码 chromium/src@content/browser/child_process_security_policy_impl.h，已 WebFetch 核实】

两类检查的直觉（命名很形象）：

- **Jail check（牢笼）**：一个 **locked-to-site** 的进程（被关在牢里），只准访问它被锁定的那个 site。`a.com` 进程声称要读 `b.com` 数据 → actual lock(`a.com`) ≠ expected lock(`b.com`) → 拒绝。防「越狱去读别人」。
- **Citadel check（城堡）**：一个**需要被隔离的敏感 site**（住在城堡里），不准被 **unlocked / allow-any-site** 的进程访问。防「随便一个进程闯进城堡偷数据」。

> ⚠ 待核：「Jail / Citadel」的精确语义注释我核到了它存在于 `PerformJailAndCitadelChecks` 且比较 actual vs expected ProcessLock，但「牢笼/城堡」这套通俗对应是我基于命名与官方 compromised-renderers 文档的合理推断（标为讲解），并非逐字注释。

线程约束是个实战坑：

> `CanAccessDataForOrigin` is restricted to the UI thread, since other threads do not have sufficient information to perform the full set of checks. ... disabling the `kSupportPartitionedBlobUrl` feature also disables these checks, since the old blob URL code makes some `CanAccessDataForOrigin` calls on the IO thread.
>
> 【来源 chromium/src 提交 c09c9e13 / source.chromium.org，已 WebSearch 核实】

含义：早期 blob URL 代码在 IO 线程调用该检查，而 IO 线程信息不全 → 校验不完整 → 存在被 unlocked renderer 绕过的窗口。这正是「enforcement 的正确性依赖于在正确的线程、用完整的权威信息做判断」的真实教训。

### 3.3 抓到说谎的 renderer：直接 kill

当 browser 发现 renderer 发来一条它的 ProcessLock 不允许的 IPC（典型 = 越权访问别站数据），这被视为「renderer 已被攻陷」的信号，处理方式不是返回错误，而是**调用 bad-message 机制终止整个 renderer 进程**（`bad_message::ReceivedBadMessage(...)`，最终走到 `RenderProcessHost::ShutdownForBadMessage`）。

设计哲学：locked 进程发出越权请求，在正常运行中**不可能发生**——能发出来就说明 renderer 已被控制，留着它只会扩大攻击面。kill 掉，止损。

> 第 1 章 §4.5 已展示 `ChildProcessSecurityPolicy` 作为安全边界执行点；本章补的是它**为什么**这么做（威胁模型）与**怎么判**（Jail/Citadel + ProcessLock 对照）。

### 3.4 enforcement 全链路（一图）

```
renderer（可能已被攻陷）
   │  IPC: "我是 b.com，给我 b.com 的 cookie/storage/blob"
   ▼
browser process (UI thread)
   │  1. 取该进程的 actual ProcessLock  ← 内容加载前就锁好的（§2.3）
   │  2. 从请求 origin/url 算 expected ProcessLock
   │  3. CanAccessDataForOrigin → PerformJailAndCitadelChecks
   ├── actual == expected ──► 放行，返回数据
   └── actual != expected ──► bad_message::ReceivedBadMessage
                               └─► kill renderer（视为已攻陷）
```

**核心 takeaway**：Site Isolation 的安全性 = 「数据分到不同进程」×「browser 用 ProcessLock 校验每条 IPC」×「对违规 renderer 零容忍 kill」。三者缺一不可。只分进程不校验，攻陷的 renderer 照样能通过 IPC 把别站数据骗出来。

---

## 4. CORB / ORB：把跨站数据挡在进程门外

§3 解决「进程内的数据别被越权 IPC 偷走」。但还有一个更前置的问题：**有些跨站数据压根不该进 renderer 进程的内存**。这就是 CORB（Cross-Origin Read Blocking）/ ORB（Opaque Response Blocking）。

### 4.1 动机：no-cors 子资源请求会把数据塞进内存

考虑 `<img src="https://bank.com/account.json">`。浏览器会发一个 no-cors 请求，正常情况下 `account.json` 不是图片，渲染会失败——但**响应体已经到了 renderer 进程内存**。在 Spectre 模型下，这等于 `account.json` 已泄露（§1.2）。SOP 拦的是「JS 读 response」，拦不住「字节进内存」。

### 4.2 CORB 的做法：基于 MIME + 嗅探判定「数据型」资源

> "Cross-Origin Read Blocking (CORB) is a best effort approach that tries to protect as much sensitive content as possible, but it is limited by the need to preserve compatibility with incorrectly labeled resources."
>
> 【来源 chromium.org/Home/chromium-security/site-isolation/，已 WebFetch 核实】

> "Chrome currently tries to identify URLs that contain HTML, XML, JSON, and PDF files, based on MIME type and other HTTP headers."
>
> 【来源 chromium.org/Home/chromium-security/site-isolation/，已 WebFetch 核实】

CORB 在 **network service 侧**（browser 进程域，renderer 拿不到的地方）判断：如果一个跨站响应被嗅探/标注为 HTML/XML/JSON/PDF 这类「数据型」内容，且不是合法的 CORS 许可读取，就**把响应体替换成空**再交给 renderer。数据根本不进 renderer 地址空间。

「best effort」的限制：依赖正确的 `Content-Type` 与 `X-Content-Type-Options: nosniff`。标注错误的资源（如把 JSON 标成 `text/plain`）可能漏保护——所以 Chromium 反复呼吁开发者正确标注 + 加 `nosniff`。

### 4.3 ORB：CORB 的继任者

ORB（Opaque Response Blocking，规范名 `opaque-response-safelist`）是 CORB 的演进，思路从「黑名单嗅探数据型」转为「白名单放行明确安全的（图片/媒体/JS 等）+ 对其余 opaque 响应更激进地阻断」。对工程师的实践影响一致：**no-cors 跨站子资源里只放真正的媒体/脚本/样式，敏感数据一律走带凭证的 CORS/fetch，并正确标 Content-Type + nosniff。**

> ⚠ 待核：ORB 当前在各 Chrome 版本的默认启用状态与「完全取代 CORB」的进度我未逐版本核实，正文按规范方向描述；精确 ship 状态以 chromestatus「ORB」条目为准。

### 4.4 CORB/ORB 与 §3 的分工

| | 拦在哪 | 拦什么 | 触发对象 |
|---|---|---|---|
| **CORB/ORB** | network service（数据进 renderer **之前**） | 跨站「数据型」子资源响应体 | no-cors 子资源请求（img/script/media…） |
| **CanAccessDataForOrigin** | browser UI thread（renderer 已运行，发 IPC 时） | renderer 越权访问别站 storage/cookie/blob | renderer 主动 IPC |

CORB 是「门口安检」，CanAccessDataForOrigin 是「金库验证」。Spectre 模型下两道都要：前者减少进内存的敏感数据，后者防止已进程内的数据被越权 IPC 导出。

---

## 5. sandbox（纵向降权）：即使 renderer 被攻陷也碰不到 OS

Site Isolation 解决「横向」——`a.com` 读不到 `b.com`。sandbox 解决「纵向」——一个被**完全攻陷**的 renderer（攻击者已能在其中执行任意原生代码）也不能碰 OS：不能读文件、不能起子进程、不能调危险 syscall。两者正交：Site Isolation 把攻击者能偷的数据范围缩到一个 site，sandbox 把攻击者能用的 OS 能力缩到接近零。

### 5.1 总目标：硬保证，与输入无关

> sandbox 提供 "hard guarantees about what ultimately a piece of code can or cannot do no matter what its inputs are." 让代码 "cannot make persistent changes to the computer or access information that is confidential."
>
> 【真实源码 chromium/src@docs/design/sandbox.md，已 WebFetch 核实】

「no matter what its inputs are」是关键——sandbox 不依赖「renderer 代码正确」，而是依赖 **OS 内核强制的边界**。renderer 里跑的代码再恶意，OS 也不让它越过 token/seccomp 限制。

### 5.2 Windows：broker/target + 四层降权

Chromium Windows sandbox 是 **broker（browser 进程）/ target（renderer 进程）** 模型：

> broker 负责：为每个 target 指定 policy、spawn target、托管 policy engine 与 **interception manager**、托管 sandbox IPC、代 target 执行 policy 允许的操作。interception manager "patch[es] the Windows API calls that should be forwarded via IPC to the broker"。
>
> 【真实源码 chromium/src@docs/design/sandbox.md，已 WebFetch 核实】

直觉：target 被降权到几乎什么 OS 调用都做不了；它确实需要的少数操作（如打开某个被许可的文件），通过 interception 把 Win32 调用「劫持」成 IPC 发回 broker，由 broker 代为执行并把结果传回。**target 自己永远不直接碰 OS 敏感资源。**

四层降权（target 进程身上叠的四把锁）：

| 机制 | 作用 | renderer 配置 |
|---|---|---|
| **Restricted Token** | 去掉所有 privilege、deny-only SID | untrusted integrity（`S-1-16-0x0`），无任何权限 |
| **Job Object** | 全局禁令：禁起子进程、禁剪贴板等 | 收紧 |
| **Desktop Object** | 隔到一个不可见/不可交互的 alternate desktop | 防 UI 层攻击 |
| **Integrity Level** | 强制访问控制 | renderer = untrusted，GPU = low，browser = medium |

【真实源码 chromium/src@docs/design/sandbox.md，已 WebFetch 核实】

policy 接口对应这四层：`SetTokenLevel()` / `SetJobLevel()` / `SetIntegrityLevel()` / `SetDesktop()`，外加 `AddRule()` 给文件/命名管道/注册表等开细粒度例外。

启动技巧（**LowerToken**）：

> target 先用接近普通用户的 initial token（impersonation token）跑起来，等关键初始化做完，调 `LowerToken()` 才激活完整 sandbox 限制。
>
> 【真实源码 chromium/src@docs/design/sandbox.md，已 WebFetch 核实】

原因：很多 OS 资源（DLL 加载、字体、初始化）在「完全降权前」拿好，降权后就拿不到了。`LowerToken()` 是「关门」那一刻。

### 5.3 Linux：两层（语义层 + 攻击面削减层）

Linux 没有 Windows 那套 token/job，sandbox 拆成正交两层：

> **Layer-1（"semantics" 层）** "prevents access to most resources from a process where it's engaged."（用 setuid sandbox 或 user namespaces）
> **Layer-2（"attack surface reduction" 层）** "restricts access from a process to the attack surface of the kernel."（用 seccomp-bpf）
>
> 【真实源码 chromium/src@docs/linux_sandboxing.md，已 WebFetch 核实】

- **Layer-1 语义层**：让进程「看不到/碰不到」大部分资源——chroot 到空目录、进独立 user/PID/network namespace、丢 capabilities。现代内核走 **user namespace**，老内核走 **setuid sandbox** helper。
- **Layer-2 攻击面削减层**：即使某些 syscall 在语义上「允许」，内核本身可能有 bug；seccomp-bpf 把 renderer 能调的 syscall 白名单收到最小，**削减内核攻击面**。

> "A BPF compiler will compile a process-specific program to filter system calls and send it to the kernel. The kernel will interpret this program for each system call and allow or disallow the call."
>
> 【来源 chromium/src@docs/linux_sandboxing.md，已 WebFetch 核实】

跨进程完整性保证（这是 seccomp-bpf 正确性的核心约束）：

> "if a process A runs under seccomp-bpf, we need to guarantee that it cannot affect the integrity of process B running under a different seccomp-bpf policy (which would be a sandbox escape)."
>
> 【真实源码 chromium/src@docs/linux_sandboxing.md，已 WebFetch 核实】

所以 seccomp 不只是禁 `ptrace()` / `process_vm_writev()` 这种显眼的，还要堵住「对 `/proc` 某些条目的 `open()`」这类间接路径——任何能影响另一个进程的 syscall 都是潜在逃逸。

**Zygote**：Linux 上 renderer 不从 browser fork+exec，而是从预热的 **zygote** 进程 fork（共享已加载的 Blink/V8 等，省启动开销）。Layer-1 在 zygote 衍生的进程上生效。

### 5.4 真实源码：seccomp-bpf 与 namespace 的入口类

`SandboxBPF`（应用 syscall 过滤策略）：

```cpp
// sandbox/linux/seccomp-bpf/sandbox_bpf.h
// This class can be used to apply a syscall sandboxing policy expressed in
// a bpf_dsl::Policy object to the current process.
explicit SandboxBPF(std::unique_ptr<bpf_dsl::Policy> policy);

// Setting a policy and starting the sandbox is a one-way operation. The
// kernel does not provide any option for unloading a loaded sandbox.
[[nodiscard]] bool StartSandbox(SeccompLevel level, bool enable_ibpb = true);
static bool SupportsSeccompSandbox(SeccompLevel level);
```
【真实源码 chromium/src@sandbox/linux/seccomp-bpf/sandbox_bpf.h，已 WebFetch 核实】

`SeccompLevel` 两档：`SINGLE_THREADED`（只 sandbox 调用线程，要求进程单线程）/ `MULTI_THREADED`（sandbox 当前进程所有线程）。「one-way operation，内核不提供卸载」——sandbox 一旦上，就不能摘，这逼着 Chromium 把「降权前需要的资源」全部前置拿好（与 Windows 的 `LowerToken()` 同构）。

`NamespaceSandbox`（用 namespace 起进程）：

```cpp
// sandbox/linux/services/namespace_sandbox.h
// Helper class for starting a process inside a new user, PID, and network
// namespace.
```
头注释还描述了被启动进程 B 的职责：B 要准备好扮演 `init(1)`（除 SIGKILL/SIGSTOP 外，没注册 handler 的信号都收不到）；B 用 `Credentials::MoveToNewUserNS()` + `Credentials::DropFileSystemAccess()` 做 chroot；最后 `Credentials::DropAllCapabilities()` 丢掉进 user namespace 时获得的 capability。
【真实源码 chromium/src@sandbox/linux/services/namespace_sandbox.h，已 WebFetch 核实】

这正是 Layer-1 语义层的代码化：**新 namespace → chroot 断文件系统 → 丢 capabilities**，三步把进程「关进盒子」。

---

## 6. cross-origin isolation：COOP + COEP 与 Spectre 的 Web 平台合约

§1–§5 是引擎内部的防御。但 Web 平台层还有一道**面向开发者的开关**：cross-origin isolation。它直接回答「为什么我的 `SharedArrayBuffer` 不见了、`performance.now()` 精度变粗了」。

### 6.1 动机：高精度 timer 是 Spectre 的弹药

Spectre 攻击的第二步是**用高精度时钟测 cache 命中差**（§7 会逐行看）。`SharedArrayBuffer` + worker 能构造纳秒级计时器，`performance.now()` 原本也很精。Spectre 后，浏览器**默认把这些武器收走**：`SharedArrayBuffer` 在普通页面被禁/受限，`performance.now()` 精度被故意调粗。

想要回这些能力，页面必须先证明自己处在一个**已经把跨站数据隔离干净**的环境里——即 cross-origin isolated。

### 6.2 两个 header 缺一不可

> "A document requires both headers to be cross-origin isolated"：
> `Cross-Origin-Opener-Policy: same-origin`（COOP）
> `Cross-Origin-Embedder-Policy: require-corp`（或 `credentialless`）（COEP）
>
> 【来源 developer.mozilla.org/Window/crossOriginIsolated，已 WebFetch 核实】

- **COOP: same-origin** —— 切断与跨站 opener/popup 的 `window` 引用关系。否则跨站 popup 与本页可能共享 BrowsingInstance（同进程，§2.1），隔离就破了。
- **COEP: require-corp** —— 要求页面加载的所有子资源都显式声明可被嵌入（CORP 头或 CORS）。否则任意 no-cors 跨站资源进了内存，又回到 §4 的风险。

满足后 `window.crossOriginIsolated === true`：

> "A cross-origin isolated document only shares its browsing context group with same-origin documents... The document may also be hosted in a separate OS process alongside other documents... This mitigates the risk of side-channel attacks and cross-origin attacks referred to as XS-Leaks."
>
> 【来源 developer.mozilla.org/Window/crossOriginIsolated，已 WebFetch 核实】

解锁的能力：

> "`SharedArrayBuffer` can be created and sent via postMessage; `Performance.now()` offers better precision; `Performance.measureUserAgentSpecificMemory()` can be called."
>
> 【来源 developer.mozilla.org/Window/crossOriginIsolated，已 WebFetch 核实】

### 6.3 合约的本质

cross-origin isolation 是一份**对等交换合约**：开发者承诺「我的进程里只有同源内容 + 显式许可的资源（没有未经同意的跨站数据）」，浏览器才把「高精度 timer + 共享内存」这些 Spectre 弹药还给你。逻辑闭环——既然进程里没有别人的敏感数据，你拿 timer 测 cache 也只能测到自己的，无害。

---

## 7. ⭐Demo：在真实引擎上观测与最小实验

Chromium 全量 build 不现实，本章 demo 以「真实引擎上观测 + 小实验」为主，全部与前述源码呼应。除 Demo 5（Electron patch 走读）外均可真跑。

### Demo 1：观测 site-per-process —— 跨站 iframe 真的在不同进程

**呼应**：§2（SiteInstance/ProcessLock）、§1（OOPIF）。验证「跨站 iframe 进独立进程」这一 Site Isolation 地基。

步骤：

1. 准备一个含跨站 iframe 的页面。本地起两个端口模拟两个 site 不够（同 IP 不同端口仍是同 site，端口不计入 site，§2.1）。用真实跨站最简单：直接访问任意嵌了第三方 iframe 的页面，或用下面的 data 跑法。
2. 打开 Chrome，地址栏输入 `chrome://process-internals/#web-contents`。
3. 切到目标 tab，展开它的 frame tree。

**预期输出**：你会看到主 frame 与跨站 iframe 分属不同 **SiteInstance**，且 **Process ID 不同**；每个 SiteInstance 行会显示其 **Lock**（如 `https://example.com`，对应 §2.3 的 locked-to-site）。同站 iframe 则与父 frame 共享 SiteInstance 与进程。

> 对照源码：`chrome://process-internals` 的 Lock 列就是 `ProcessLock::GetProcessLockURL()`（§2.3）；不同 PID 证明 §2.2 的「能共享内存的才同进程」。

命令行强制全量隔离（即使在内存受限平台）以便观测：

```bash
# macOS 示例；--site-per-process 强制对所有站点独占进程
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --site-per-process \
  --user-data-dir=/tmp/coi-demo
# 反向实验：--disable-site-isolation-trials 关掉，再看 process-internals，跨站 iframe 会塌回同进程
```

### Demo 2：CDP 驱动 —— 用 `Target.getTargets` 枚举进程边界

**呼应**：§2、§3（每个 OOPIF target 是 browser 侧权威记录的边界）。可真跑。

```bash
# 1. 带远程调试端口启动 Chrome
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --site-per-process \
  --user-data-dir=/tmp/cdp-demo &

# 2. 打开一个含跨站 iframe 的页面（手动或下一步脚本里 navigate）
# 3. 列出所有 target（含 OOPIF）
curl -s http://localhost:9222/json | python3 -m json.tool
```

**预期输出**：JSON 数组里每个 frame/iframe 是一个 target，含 `type`（`page` / `iframe`）、`url`、独立的 `webSocketDebuggerUrl`。跨站 iframe 作为独立 target 出现，对应它在独立进程里。

进一步用 Node 脚本读进程级信息（可真跑，需 `npm i chrome-remote-interface`）：

```javascript
// list-targets.js —— 呼应 §3：target 边界由 browser 侧权威维护
const CDP = require('chrome-remote-interface');
(async () => {
  const client = await CDP({ port: 9222 });
  const { targetInfos } = await client.Target.getTargets();
  for (const t of targetInfos) {
    console.log(`[${t.type}] ${t.url}  (targetId=${t.targetId.slice(0,8)})`);
  }
  await client.close();
})().catch(console.error);
```

预期打印每个 target 一行，跨站 iframe 单独成行——CDP 看到的 target 拓扑就是 §2 的 SiteInstance/进程拓扑在调试协议层的投影。

### Demo 3：观测 cross-origin isolation 开关 —— SharedArrayBuffer 的有/无

**呼应**：§6（COOP+COEP 解锁 SharedArrayBuffer / 高精度 timer）。可真跑，是把「Spectre 弹药被收走」做成可见实验的最直接方式。

**A. 不带 header（默认）**：任意普通页面，DevTools Console 执行：

```javascript
console.log('crossOriginIsolated =', window.crossOriginIsolated);  // 预期 false
try { new SharedArrayBuffer(16); console.log('SAB ok'); }
catch (e) { console.log('SAB blocked:', e.message); }              // 预期 blocked
// 观察 timer 精度（被故意调粗）
const t0 = performance.now(); for (let i=0;i<1e6;i++); 
console.log('now() sample:', performance.now() - t0);
```

**B. 带 header**：用一行 Python 起一个发 COOP/COEP 的本地服务器：

```python
# coi_server.py —— 给每个响应加 COOP+COEP（§6.2）
from http.server import HTTPServer, SimpleHTTPRequestHandler
class H(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
HTTPServer(('127.0.0.1', 8777), H).serve_forever()
```

```bash
# 放一个 index.html 在当前目录，然后：
python3 coi_server.py
# 浏览器开 http://127.0.0.1:8777/ ，Console 再跑上面 A 段脚本
```

**预期输出**：B 场景 `crossOriginIsolated === true`，`new SharedArrayBuffer(16)` 成功，`performance.now()` 精度更高。**这就是 §6.3 合约的可见证据**：你向浏览器证明了「进程里没有未授权的跨站数据」，浏览器把 Spectre 弹药还给你。

> 注意：若页面引用了任何不带 CORP/CORS 的跨站子资源，COEP: require-corp 会让其加载失败——这正是 §6.2 要求的「子资源必须显式可嵌入」。

### Demo 4：观测 sandbox —— `chrome://sandbox` 与进程降权

**呼应**：§5（Windows 四层 / Linux 两层）。可真跑（只读观测）。

```text
1. 地址栏：chrome://sandbox        （Linux/Windows 显示各 renderer 的 sandbox 状态）
   预期：每个 renderer 一行，Linux 显示 "Seccomp-BPF sandbox" / "Yes" 等，
        对应 §5.3 Layer-2 已激活；Windows 显示 token/integrity 等。
2. 地址栏：chrome://gpu  → 搜 "Sandboxed"
   预期：GPU 进程 Sandboxed: true（GPU integrity = low，§5.2 表）。
```

Linux 下命令行佐证 renderer 被关进 namespace / 降 capabilities（呼应 §5.4 `NamespaceSandbox`）：

```bash
# 找一个 renderer 进程
pgrep -af 'type=renderer' | head
# 观察其 namespace（与 browser 进程对比，user/pid/net 应不同）
RPID=$(pgrep -f 'type=renderer' | head -1)
ls -l /proc/$RPID/ns/        # 预期 user/pid/net 的 inode 与 browser 不同
cat /proc/$RPID/status | grep -E 'Seccomp|CapEff'
# 预期：Seccomp: 2（filter mode，= seccomp-bpf 已加载，§5.4 SandboxBPF::StartSandbox）
#       CapEff: 0000000000000000（capabilities 已全丢，§5.4 DropAllCapabilities）
```

**预期输出**：`Seccomp: 2` 直接证明 §5.4 `StartSandbox` 已把该进程切到 seccomp filter 模式；`CapEff: 0` 证明 §5.4 的 `DropAllCapabilities()` 生效。这是把头文件里的代码意图落到 `/proc` 上的实测。

### Demo 5（偏 build）：Electron sandbox 与 Fuses patch 体系走读 + 最小硬化示例

**呼应**：§8（Electron）。Electron 改造偏 build，这里给 patch 体系走读 + 最小可跑的 Fuses 硬化（Fuses 部分可真跑）。

**(a) sandbox 默认开启验证（可真跑）** —— 最小 Electron app：

```javascript
// main.js —— Electron ≥20 默认 sandbox:true
const { app, BrowserWindow } = require('electron');
app.whenReady().then(() => {
  const win = new BrowserWindow({
    webPreferences: { /* 不写 sandbox/nodeIntegration，用默认 */ }
  });
  win.loadFile('index.html');
});
```

```html
<!-- index.html -->
<script>
  // 沙箱 renderer 里没有 Node 环境（§8）：require/process.binding 不可用
  document.body.innerText =
    'typeof require = ' + typeof require + '\n' +
    'process.type = ' + (window.process ? window.process.type : 'no process');
</script>
```

```bash
npm i -D electron && npx electron .
```

**预期输出**：窗口显示 `typeof require = undefined`（沙箱 renderer 拿不到完整 Node require，§8.1 引文「sandboxed renderer won't have a Node.js environment initialized」）。若把 `webPreferences` 改成 `{ sandbox: false, nodeIntegration: true }`，`require` 变 function——印证「nodeIntegration: true disables the sandbox」（§8.1 引文）。

**(b) Fuses 二进制硬化（可真跑，需已安装 electron）** —— 呼应 §8.2：

```javascript
// flip-fuses.js —— 在打包好的 Electron 二进制上翻位（§8.2）
const { flipFuses, FuseVersion, FuseV1Options } = require('@electron/fuses');
flipFuses(require('electron'), {           // 真实用法，来自官方文档
  version: FuseVersion.V1,
  [FuseV1Options.RunAsNode]: false,                       // 禁 ELECTRON_RUN_AS_NODE，防 LotL 攻击
  [FuseV1Options.EnableNodeCliInspectArguments]: false,   // 禁 --inspect
  [FuseV1Options.OnlyLoadAppFromAsar]: true,              // 只从校验过的 asar 加载
  [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
});
```

```bash
npm i -D @electron/fuses && node flip-fuses.js
# 验证 RunAsNode 已被关：下面这条应不再以 Node 模式执行
ELECTRON_RUN_AS_NODE=1 npx electron -e "console.log('should NOT run as node')"
```

**预期输出**：翻位后，`ELECTRON_RUN_AS_NODE=1` 不再让 Electron 退化成纯 Node 解释器执行任意脚本——这正是 §8.2「runAsNode 防 living off the land」的效果。Fuses 的位写在二进制的 sentinel 段（§8.2 的 `dL7pKGdnNz796PbbjQWNKmHXBZaB9tsX`）后，OS 代码签名保证不被改回。

**(c) patch 体系走读（偏 build，走读不真跑）**：Electron 不 fork Chromium，而是以 `patches/chromium/*.patch` 形式在 `gclient sync` 后打到 Chromium 源码树上。sandbox 相关改造（如把 Node 安全地植入 main 进程、保持 renderer sandbox 行为与上游一致）就活在这套 patch 里。走读路径：`electron/patches/config.json` 列出 patch 目录 → `electron/patches/chromium/` 下按文件名对应到 Chromium 源文件 → 每个 `.patch` 顶部有 commit message 说明动机。**最小补丁形态**（示意，非逐字）：

```diff
# 示意，非逐字：Electron patch 的典型形态
--- a/content/browser/renderer_host/render_process_host_impl.cc
+++ b/content/browser/renderer_host/render_process_host_impl.cc
@@ ... @@
   // Electron: 在不破坏 sandbox 的前提下注入 Node 集成钩子
+  electron::MaybeSetupNodeIntegration(...);
```

> 走读要点：Electron 的安全难点正是「既要 main 进程有完整 Node，又要 renderer 保持 Chromium sandbox」。patch 体系让它能跟上 Chromium 每次版本升级（rebase patch），而不是分叉维护整个引擎。

---

## 8. Electron 域：sandbox 落地与平台级硬化

### 8.1 默认 sandbox 与 Node 的张力

> "Starting from Electron 20, the sandbox is enabled for renderer processes without any further configuration." 且 "When renderer processes in Electron are sandboxed, they behave in the same way as a regular Chrome renderer would."
>
> 【来源 electronjs.org/docs/latest/tutorial/sandbox，已 WebFetch 核实】

但 Electron 的卖点是 renderer 能用 Node。矛盾点：

> "Sandboxing is tied to Node.js integration. Enabling Node.js integration for a renderer process by setting `nodeIntegration: true` disables the sandbox for the process." 且 "A sandboxed renderer won't have a Node.js environment initialized."
>
> 【来源 electronjs.org/docs/latest/tutorial/sandbox，已 WebFetch 核实】

沙箱 renderer 里 preload 只拿到**子集**能力：

> "A `require` function similar to Node's `require` module is exposed, but can only import a subset of Electron and Node's built-in modules"（如 `electron`/`events`/`timers`/`url`），并 polyfill `Buffer`/`process`/`setImmediate` 等全局。
>
> 【来源 electronjs.org/docs/latest/tutorial/sandbox，已 WebFetch 核实】

**正确姿势**：保持 `sandbox: true`（默认）+ `contextIsolation: true`（默认），特权操作（读文件、起进程）一律经 preload 用 `contextBridge` 暴露白名单 API，再走 IPC 到 main 进程执行。绝不用 `nodeIntegration: true` 图省事——那等于亲手拆掉 Chromium 几千 commit 堆出来的 renderer sandbox（§5）。这与第 1 章 §8.5「contextIsolation 误配导致安全漏洞」是同一根因的两面。

### 8.2 Fuses：打包期二进制级硬化

sandbox 是运行期降权，Fuses 是**打包期**把危险开关焊死：

> Fuses 是 Electron 二进制里的 "magic bits ... that can be flipped when packaging your Electron app"，翻位后 "the OS becomes responsible for ensuring those bits aren't flipped back via OS-level code signing validation (e.g. Gatekeeper on macOS or AppLocker on Windows)."
>
> 【来源 electronjs.org/docs/latest/tutorial/fuses，已 WebFetch 核实】

四个安全相关 Fuse：

| Fuse | 关掉它防什么 |
|---|---|
| **runAsNode** | 防 `ELECTRON_RUN_AS_NODE` 把你的 app 当通用 Node 解释器跑任意脚本（living-off-the-land） |
| **nodeCliInspect** | 防 `--inspect` 调试口被滥用；关掉后 SIGUSR1 不初始化 main inspector |
| **onlyLoadAppFromAsar** | 配合完整性校验，使「加载未校验代码」不可能 |
| **embeddedAsarIntegrityValidation** | 加载时校验 `app.asar` 内容，性能影响极小 |

【来源 electronjs.org/docs/latest/tutorial/fuses，已 WebFetch 核实】

实现是二进制级的（§7 Demo 5 已给可跑示例）：sentinel `dL7pKGdnNz796PbbjQWNKmHXBZaB9tsX` 后跟 fuse wire，每位 ASCII `0`(禁)/`1`(启)/`r`(移除)。

### 8.3 Electron 与上游 Chromium 的安全模型差异

| 维度 | 上游 Chromium | Electron |
|---|---|---|
| renderer sandbox | 默认全开 | ≥20 默认开（同上游行为） |
| Node in renderer | 无 | 可选（开则拆 sandbox，**不推荐**） |
| Site Isolation | 桌面默认 site-per-process | 继承，但典型单 app 场景跨站内容少，价值不如浏览器突出 |
| 平台硬化 | OS 自带（签名等） | 额外 Fuses 二进制焊死 |
| 数据访问 enforcement | `CanAccessDataForOrigin`（§3） | 同上游 + IPC 边界自管（contextBridge 白名单） |

> 关键认知：Electron 把 Chromium 的「横向 Site Isolation + 纵向 sandbox」整套继承下来，但又开了一道「main 进程全 Node」的特权口子。安全责任从「浏览器厂商」转移到「app 开发者」——你不正确用 contextBridge/Fuses，引擎再硬也白搭。

---

## 9. 失败模式、生产真坑与根因

### 9.1 误以为「分了进程就安全」，省掉 IPC 校验

**现象**：自研基于 CEF/Electron 的产品，开了 site-per-process，却在自定义 IPC handler 里直接信任 renderer 传来的 origin/path 去读文件。
**根因**：§3 的第一原则被违反——renderer 可能已被攻陷，它声称的 origin 不可信。Site Isolation 只隔数据，不替你校验自定义 IPC。
**正解**：自定义 IPC 一律在 browser/main 侧用权威状态（`GetLastCommittedOrigin` 思路）校验，对越权请求拒绝甚至 kill。

### 9.2 CORB/ORB 误伤：合法 JSON 被当跨站数据拦掉

**现象**：某第三方 `<script>` 实为 JSONP，但服务端把 `Content-Type` 标成 `application/json` 且带 `nosniff` → CORB 判为数据型 → 响应体清空 → 脚本报错。
**根因**：§4.2，CORB 按 MIME + nosniff 判定数据型。标注与用途不一致。
**正解**：JSONP 这类「当脚本用」的资源必须标 `application/javascript`；真正的数据接口走 `fetch` + CORS，别用 `<script>`/`<img>` 拉。

### 9.3 升级到需要 cross-origin isolation 后白屏 / 资源加载失败

**现象**：为了用 `SharedArrayBuffer`（如 ffmpeg.wasm）加了 COOP/COEP，结果第三方图片、字体、广告 iframe 全挂。
**根因**：§6.2，`COEP: require-corp` 要求**所有**子资源显式声明可嵌入；不合规的跨站资源被拦。
**正解**：给自有跨站资源加 `Cross-Origin-Resource-Policy: cross-origin`；第三方资源改用 `COEP: credentialless`（无凭证加载）或迁到同源代理。上线前用 `Cross-Origin-Embedder-Policy-Report-Only` 先收集会被拦的资源清单。

### 9.4 Linux 上 sandbox 起不来 / 退化到 no-sandbox

**现象**：自打包 Chromium/Electron 在某些 Linux 容器里报 sandbox 相关错误，或被迫加 `--no-sandbox`。
**根因**：§5.3，Layer-1 依赖 user namespace（`CLONE_NEWUSER`）；很多容器/老内核默认禁用 unprivileged user namespace，或 setuid sandbox helper（`chrome-sandbox`）权限位不对（需 4755 + root owned）。
**正解**：宿主开启 `kernel.unprivileged_userns_clone=1`，或正确安装 setuid `chrome-sandbox`。**生产严禁 `--no-sandbox`**——那等于拆掉 §5 整层纵向防御，一个 renderer RCE 直接拿 OS 权限。

### 9.5 `CanAccessDataForOrigin` 线程错位导致校验缺口（历史真坑）

**现象/根因**：§3.2 引文——早期 blob URL 代码在 IO 线程调 `CanAccessDataForOrigin`，而该检查需在 UI 线程才有完整信息，导致 unlocked renderer 可能绕过对 isolated site 的保护。
**教训**：enforcement 的正确性不仅看「有没有调校验」，还看「在不在对的线程、用没用完整权威信息」。Chromium 用 `kRestrictCanAccessDataForOriginToUIThread` feature 收口（提交 c09c9e13）。自研 IPC 校验同理：别在信息不全的线程做安全判定。

### 9.6 Electron `nodeIntegration: true` 一键拆穿所有防御

**现象**：图方便在 renderer 直接 `require('fs')`，远程内容（或被 XSS 注入的脚本）随即拿到完整文件系统。
**根因**：§8.1，`nodeIntegration: true` 直接 disable sandbox。Site Isolation/sandbox 全被绕过。
**正解**：`sandbox: true` + `contextIsolation: true` + `contextBridge` 白名单 + 关键 Fuses（§8.2）。把 renderer 当成「随时可能被攻陷的不可信区」。

---

## 章末五件套

### 1. 核心概念一句话

- **Site Isolation（横向）**：假设 Spectre 让 renderer 能读自己进程内任意内存 → 把跨站敏感数据从该地址空间赶出去（site-per-process），browser 用 ProcessLock 校验每条 IPC、对违规 renderer 直接 kill。
- **sandbox（纵向）**：假设 renderer 被完全攻陷 → 用 OS 内核强制边界（Win 四层 token/job/integrity/desktop + broker；Linux 两层 namespace/seccomp-bpf）把它降权到碰不到 OS。
- **CORB/ORB**：在数据进 renderer 之前，把跨站「数据型」子资源响应体挡在网络层。
- **cross-origin isolation（COOP+COEP）**：开发者证明「进程内无未授权跨站数据」，浏览器才归还高精度 timer/SharedArrayBuffer 这些 Spectre 弹药。

### 2. 最易混淆的三对概念

| A | B | 区别 |
|---|---|---|
| Site Isolation | sandbox | 横向隔数据（站点间）vs 纵向降权（进程对 OS）；正交，都要 |
| site | origin | site = scheme+eTLD+1（默认隔离粒度，因 `document.domain` 同步互访）；origin 更细，需显式 opt-in（`--isolate-origins`/`Origin-Agent-Cluster`）|
| CORB/ORB | CanAccessDataForOrigin | 前者拦「子资源进内存」（网络层、加载前）；后者拦「renderer 越权 IPC 导出数据」（browser UI 线程、运行时） |

### 3. 下一步深入路径

- 读第 3 章（若涉及 Mojo IPC）：`CanAccessDataForOrigin` 校验的就是 Mojo IPC，安全边界落在 Mojo message 校验上。
- 精读 `content/browser/child_process_security_policy_impl.cc` 的 `PerformJailAndCitadelChecks` 实现，把 §3.2 的 Jail/Citadel 从注释看到逐行。
- 读 `docs/security/compromised-renderers.md` 全文，掌握 Site Isolation **不防**什么（同 site 内 timing、navigation 推断、`<iframe sandbox>` 不与 precursor origin 隔离、`file:` frame 共享进程、Android WebView 无 Site Isolation）。
- 读 `sandbox/policy/` 与各平台 `sandbox/win`、`sandbox/linux`，把 §5 的 policy 接口看到实现。

### 4. 速查 CLI / 观测入口

```bash
# 强制全量 Site Isolation（观测用）
--site-per-process
# 指定 origin 级隔离
--isolate-origins=https://accounts.example.com
# 关闭隔离做对照实验（仅本地）
--disable-site-isolation-trials
# 生产严禁：--no-sandbox（拆纵向防御）

# 观测入口（地址栏）
chrome://process-internals/#web-contents   # SiteInstance/ProcessLock 拓扑（§2）
chrome://sandbox                           # 各 renderer sandbox 状态（§5）
chrome://gpu                               # GPU 进程 Sandboxed/integrity（§5.2）

# Linux 实测进程降权（§5.4）
cat /proc/<renderer_pid>/status | grep -E 'Seccomp|CapEff'   # Seccomp:2 / CapEff:0
ls -l /proc/<renderer_pid>/ns/                               # user/pid/net 独立
```

### 5. 关键源码位置速查

| 主题 | 路径（chromium/src，除注明外） |
|---|---|
| Spectre 威胁模型纲领 | `docs/security/side-channel-threat-model.md` |
| compromised renderer 防御原则 | `docs/security/compromised-renderers.md` |
| 进程模型/site 定义 | `docs/process_model_and_site_isolation.md` |
| SiteInstance 接口 | `content/public/browser/site_instance.h` |
| ProcessLock | `content/browser/process_lock.h` |
| 数据访问 enforcement（Jail/Citadel） | `content/browser/child_process_security_policy_impl.{h,cc}` |
| Windows sandbox 设计 | `docs/design/sandbox.md` |
| Linux sandbox 两层 | `docs/linux_sandboxing.md` |
| seccomp-bpf 入口 | `sandbox/linux/seccomp-bpf/sandbox_bpf.h` |
| namespace sandbox 入口 | `sandbox/linux/services/namespace_sandbox.h` |
| OOPIF 设计 | chromium.org/developers/design-documents/oop-iframes/ |
| Electron sandbox | electronjs.org/docs/latest/tutorial/sandbox |
| Electron Fuses | electronjs.org/docs/latest/tutorial/fuses |
| Spectre PoC（演示动机） | github.com/google/security-research-pocs（spectre.js） |

---

> **本章与第 1 章的边界**：第 1 章讲「进程怎么分配」（SiteInstance→SiteInfo→ProcessLock 决策链、进程类型、OOPIF 机制、Electron 进程映射）。本章讲「为什么必须这么分（Spectre 威胁模型）+ 分完怎么真正挡住攻击（browser 侧 IPC enforcement、CORB/ORB、sandbox 纵向降权、cross-origin isolation 合约、Electron sandbox/Fuses 硬化）」。两章在 SiteInstance/ProcessLock 上有意重叠一小段以保持各自自洽，深度不同：第 1 章看分配，本章看 enforcement。
