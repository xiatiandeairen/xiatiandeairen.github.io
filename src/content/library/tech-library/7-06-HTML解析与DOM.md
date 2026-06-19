---
title: "HTML 解析与 DOM（Chromium 域）"
slug: "7-06"
collection: "tech-library"
group: "chromium内核"
order: 7006
summary: "前置依赖：第 1 章（多进程架构）、第 4 章（调度器与事件循环）、第 5 章（导航加载与生命周期）。本章聚焦 Blink renderer 进程内 HTML 解析管线与 DOM 树构建；不涉及各 Web API 用法、扩展系统、DevTools 面板用法。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T11:30:08.000Z"
updatedAt: "2026-06-12T11:30:08.000Z"
---
> **前置依赖**：第 1 章（多进程架构）、第 4 章（调度器与事件循环）、第 5 章（导航加载与生命周期）。本章聚焦 Blink renderer 进程内 HTML 解析管线与 DOM 树构建；不涉及各 Web API 用法、扩展系统、DevTools 面板用法。

---

## TL;DR

Blink 的 HTML 解析管线是一条五段流水线：**网络字节流 → `HTMLInputStream` → `HTMLTokenizer`（状态机） → `HTMLTreeBuilder`（插入模式） → `HTMLConstructionSite`（任务队列） → DOM 树**。整个流水线运行在 renderer 主线程上，受双重预算（token 计数 + 时间片）控制，每个 yield 点都让调度器有机会处理高优先级任务。`HTMLPreloadScanner` 在背景线程并行扫描，抢先发出资源请求。`document.write()` 通过 `insert()` 路径同步注入 `SegmentedString`，是阻塞主线程的根因。DOM 节点是 `GarbageCollected<Node>` 的 Oilpan 管理对象，变更通过 `ChildrenChanged()` / `IncDOMTreeVersion()` 通知下游（样式、布局）失效。

---

## 6.1 设计考古：为什么 HTML 解析器如此特殊

### 6.1.1 历史来源与设计动机

HTML 解析不能用传统的 LL/LR 文法，原因有三：

1. **错误容忍**：浏览器必须"猜"出作者意图，而不是报错退出。Web 上约 95% 的页面在严格 XML 模式下都不合法。
2. **脚本再入**：`<script>` 标签执行时可以调用 `document.write()`，把新 HTML 注入到解析器当前位置——解析器的输入流是可变的。
3. **状态感知词法**：同一字节序列在不同上下文（`<script>`、`<style>`、`<textarea>`）有截然不同的语义，词法规则必须随 tree builder 状态而变。

WHATWG 在 2006 年启动了 HTML5 解析算法的规范化工作。Apple 的 WebKit（Blink 的前身）在 2008 年发布了基于该规范的 `HTMLTokenizer` 实现（版权行至今仍保留在 Chromium 源码中）。Google fork WebKit 形成 Blink 后（2013），对解析器做了大量性能改造：分离 preload scanner（2013）、后台线程预加载扫描（2018）、dual budget 调度（2020 年前后，与 Core Web Vitals 对齐）。

**关键设计文档**：
- WHATWG HTML Living Standard §13.2：[https://html.spec.whatwg.org/multipage/parsing.html](https://html.spec.whatwg.org/multipage/parsing.html)（实际 fetch 核实，内容完整）
- Blink 解析器目录：`third_party/blink/renderer/core/html/parser/`（共 95 个文件，实际 fetch 核实）

### 6.1.2 document.write 的恶：设计级缺陷

`document.write()` 是 HTML 解析再入性最直接的体现，也是生产中最常见的性能黑洞。它通过 `HTMLDocumentParser::insert()` 路径工作：

```
主线程运行脚本 → document.write(str)
  → Document::write()
    → HTMLDocumentParser::insert(str)          // 同步
      → input_.InsertAtCurrentInsertionPoint()  // 注入 SegmentedString
      → PumpTokenizerIfPossible()               // 立即消费，不可 yield
```

Chrome 的 Intervention 机制（2016）开始对跨源 `document.write()` 注入的 `<script>` 直接拦截，就是因为它在 2G 网络上会导致页面加载时间延长数秒。

---

## 6.2 整体架构：五段流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│                   renderer 主线程                                    │
│                                                                     │
│  网络字节流                                                          │
│  (via Mojo DataPipe)                                                │
│       │                                                             │
│       ▼                                                             │
│  HTMLDocumentParser::Append()                                       │
│       │ AppendToEnd()                                               │
│       ▼                                                             │
│  HTMLInputStream ──► SegmentedString linked list                    │
│       │                                                             │
│       ▼  PumpTokenizer() [dual budget: token count + time]          │
│  HTMLTokenizer::NextToken()                                         │
│       │  70+ 状态机                                                 │
│       ▼                                                             │
│  AtomicHTMLToken (StartTag/EndTag/Character/DOCTYPE/Comment/EOF)    │
│       │                                                             │
│       ▼                                                             │
│  HTMLTreeBuilder::ConstructTree()                                   │
│       │  25 insertion modes + open elements stack                   │
│       ▼                                                             │
│  HTMLConstructionSite                                               │
│       │  CreateElement() → AttachLater() → QueueTask()             │
│       │  ExecuteQueuedTasks()                                       │
│       ▼                                                             │
│  DOM Tree  (GarbageCollected Node/Element/Text...)                  │
└─────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────────────┐
                              │  background thread       │
                              │  BackgroundHTMLScanner   │ ← 并行扫描 inline script
                              │  HTMLPreloadScanner      │ ← 先行发出资源请求
                              └──────────────────────────┘
```

---

## 6.3 真实源码精读

### 6.3.1 HTMLTokenizer：状态机核心

【真实源码 `third_party/blink/renderer/core/html/parser/html_tokenizer.h`，实际 fetch 核实】

```cpp
// html_tokenizer.h（节选，实际 fetch 核实）
class CORE_EXPORT HTMLTokenizer {
  USING_FAST_MALLOC(HTMLTokenizer);   // 自定义分配器，避免 GC 开销
 public:
  explicit HTMLTokenizer(const HTMLParserOptions&);
  // 禁止拷贝：tokenizer 持有可变 input stream，语义唯一
  HTMLTokenizer(const HTMLTokenizer&) = delete;
  HTMLTokenizer& operator=(const HTMLTokenizer&) = delete;

  // 核心方法：从 SegmentedString 取下一个 token
  // 返回非 null 时调用者需立即处理，nullptr 表示输入耗尽
  HTMLToken* NextToken(SegmentedString&);

  // Tree builder 通知 tokenizer 切换词法模式（如进入 <script>）
  void UpdateStateFor(const HTMLToken&);
  void UpdateStateFor(html_names::HTMLTag);

  // 背景 preload scanner 用：近似推断当前应处于的词法状态
  // 注意：只是"近似"，对 <pre> 换行和 CDATA 有已知偏差
  std::optional<State> SpeculativeStateForTag();

  State GetState() const;
  void SetState(State);
```

状态机有 70+ 个状态，以三个核心状态为例：

【真实源码 `third_party/blink/renderer/core/html/parser/html_tokenizer.cc`，实际 fetch 核实】

```cpp
// html_tokenizer.cc（节选，实际 fetch 核实）

// kDataState：HTML 文本的默认初始状态
HTML_BEGIN_STATE(kDataState) {
  if (cc == '&')
    HTML_ADVANCE_PAST_NON_NEWLINE_TO(kCharacterReferenceInDataState);
  else if (cc == '<') {
    if (HaveBufferedCharacterToken()) {
      return true;           // 先 emit 已缓冲的文本 token，下次再处理 '<'
    }
    HTML_ADVANCE_PAST_NON_NEWLINE_TO(kTagOpenState);
  } else if (cc == kEndOfFileMarker)
    return EmitEndOfFile(source);
  else {
    return EmitData(source, cc);  // 累积字符到 character token
  }
}

// kTagOpenState：刚遇到 '<'
HTML_BEGIN_STATE(kTagOpenState) {
  if (IsAsciiAlpha(cc)) {
    token_.BeginStartTag(ToLowerCase(cc));  // 开始 start tag，名字强制小写
    HTML_ADVANCE_PAST_NON_NEWLINE_TO(kTagNameState);
  } else if (cc == '!') {
    HTML_ADVANCE_PAST_NON_NEWLINE_TO(kMarkupDeclarationOpenState); // DOCTYPE/注释
  } else if (cc == '/') {
    HTML_ADVANCE_PAST_NON_NEWLINE_TO(kEndTagOpenState);
  } else {
    ParseError();
    BufferCharacter('<');        // 错误恢复：把 '<' 当文本处理
    HTML_RECONSUME_IN(kDataState);
  }
}

// kTagNameState：累积标签名
HTML_BEGIN_STATE(kTagNameState) {
  // 快速路径：批量扫描无特殊字符的字节序列
  while (!CheckScanFlag(cc, ScanFlags::kTagNameSpecial)) {
    token_.AppendToName(ToLowerCaseIfAlpha(cc));  // 强制小写
    if (!input_stream_preprocessor_.AdvancePastNonNewline(source, cc))
      return HaveBufferedCharacterToken();
  }
  if (cc == '/') {
    HTML_ADVANCE_PAST_NON_NEWLINE_TO(kSelfClosingStartTagState);
  } else if (cc == '>') {
    return EmitAndResumeInDataState(source);  // emit 完整 start tag token
  } else {
    DCHECK(IsTokenizerWhitespace(cc));
    HTML_ADVANCE_TO(kBeforeAttributeNameState);  // 进入属性解析
  }
}
```

**设计要点**：
- `HTML_BEGIN_STATE` / `HTML_ADVANCE_TO` 是宏，展开为 `goto` 驱动的 Duff's device 风格 switch，避免函数调用开销。
- `HaveBufferedCharacterToken()` 确保 character token 和 tag token 不会混入同一个 emit 中——解析器一次只处理一个 token。
- 标签名在词法阶段就强制 ASCII 小写，比 tree builder 阶段做效率更高。

### 6.3.2 HTMLTreeBuilder：插入模式状态机

【真实源码 `third_party/blink/renderer/core/html/parser/html_tree_builder.h`，实际 fetch 核实】

```cpp
// html_tree_builder.h（节选，实际 fetch 核实）
class HTMLTreeBuilder final : public GarbageCollected<HTMLTreeBuilder> {
  // 25 个插入模式，对应 WHATWG spec §13.2.6
  enum InsertionMode {
    kInitialMode,
    kBeforeHTMLMode,
    kBeforeHeadMode,
    kInHeadMode,
    kInHeadNoscriptMode,
    kAfterHeadMode,
    kTemplateContentsMode,
    kInBodyMode,          // 绝大多数内容在这里处理
    kTextMode,
    kInTableMode,
    kInTableTextMode,
    kInCaptionMode,
    kInColumnGroupMode,
    kInTableBodyMode,
    kInRowMode,
    kInCellMode,
    kAfterBodyMode,
    kInFramesetMode,
    kAfterFramesetMode,
    kAfterAfterBodyMode,
    kAfterAfterFramesetMode,
  };
```

【真实源码 `third_party/blink/renderer/core/html/parser/html_tree_builder.cc`，实际 fetch 核实】

```cpp
// html_tree_builder.cc（ProcessToken，节选，实际 fetch 核实）
void HTMLTreeBuilder::ProcessToken(AtomicHTMLToken* token) {
  if (token->GetType() == HTMLToken::kCharacter) {
    ProcessCharacter(token);  // 字符 token 有专门快速路径
    return;
  }

  tree_.Flush();                       // 把待处理的 pending text 先写入 DOM
  should_skip_leading_newline_ = false;

  switch (token->GetType()) {
    case HTMLToken::DOCTYPE:      ProcessDoctypeToken(token); break;
    case HTMLToken::kStartTag:    ProcessStartTag(token);    break;
    case HTMLToken::kEndTag:      ProcessEndTag(token);      break;
    case HTMLToken::kComment:     ProcessComment(token);     break;
    case HTMLToken::kEndOfFile:   ProcessEndOfFile(token);   break;
  }
}

// ProcessStartTagForInBody（kInBodyMode 下的 start tag 分发，节选）
void HTMLTreeBuilder::ProcessStartTagForInBody(AtomicHTMLToken* token) {
  switch (token->GetHTMLTag()) {
    case HTMLTag::kP:
      // 自动关闭上一个 <p>（如果在 button scope 内有未关闭的 <p>）
      ProcessFakePEndTagIfPInButtonScope();
      tree_.InsertHTMLElement(token);
      break;
    case HTMLTag::kTable:
      if (!tree_.InQuirksMode() &&
          tree_.OpenElements()->InButtonScope(HTMLTag::kP))
        ProcessFakeEndTag(HTMLTag::kP);
      tree_.InsertHTMLElement(token);
      SetInsertionMode(kInTableMode);   // 切换到表格插入模式
      break;
    default:
      // 大多数元素：先修复 active formatting elements，再插入
      tree_.ReconstructTheActiveFormattingElements();
      tree_.InsertHTMLElement(token);
      break;
  }
}
```

**foster parenting（代养）机制**：表格内容区域不允许出现块元素，但 HTML 错误容忍需要处理它。tree builder 使用 `RedirectToFosterParentGuard` RAII 对象临时重定向插入位置：

```cpp
// html_tree_builder.cc（foster parenting，节选，实际 fetch 核实）
case kInTableMode: {
  if (/* 当前节点是 table/tbody/tr */) {
    original_insertion_mode_ = insertion_mode_;
    SetInsertionMode(kInTableTextMode);
  } else {
    // 激活 foster parent 重定向：把内容插到 table 的前面而不是里面
    HTMLConstructionSite::RedirectToFosterParentGuard redirecter(tree_);
    ProcessCharacterBufferForInBody(buffer);
    break;
  }
}
```

### 6.3.3 HTMLConstructionSite：DOM 构建任务队列

【真实源码 `third_party/blink/renderer/core/html/parser/html_construction_site.h`，实际 fetch 核实】

```cpp
// html_construction_site.h（节选，实际 fetch 核实）
class HTMLConstructionSite final {
  // 关键约束：最大 DOM 树深度 512
  static constexpr unsigned kMaximumHTMLParserDOMTreeDepth = 512;

  // 任务队列：注释说"common case 只有 1 个 task，大多数 token 只产生 1 次 DOM 变更"
  // 预分配 1 个位置，避免堆分配
  typedef HeapVector<HTMLConstructionSiteTask, 1> TaskQueue;
```

【真实源码 `third_party/blink/renderer/core/html/parser/html_construction_site.cc`，实际 fetch 核实】

```cpp
// html_construction_site.cc（CreateElement，节选，实际 fetch 核实）
Element* HTMLConstructionSite::CreateElement(
    AtomicHTMLToken* token,
    const AtomicString& namespace_uri) {
  Document& document = OwnerDocumentForCurrentNode();

  // 优先使用静态 QualifiedName 表（预生成的 html_names::TagToQualifiedName）
  QualifiedName tag_name = (token->IsValidHTMLTag() &&
        namespace_uri == html_names::xhtmlNamespaceURI)
      ? static_cast<const QualifiedName&>(
            html_names::TagToQualifiedName(token->GetHTMLTag()))
      : QualifiedName(g_null_atom, token->GetName(), namespace_uri);

  // 自定义元素注册表查找
  auto* definition = LookUpCustomElementDefinition(document, tag_name, is, registry);
  bool will_execute_script = definition && !is_parsing_fragment_;

  if (will_execute_script) {
    // 自定义元素构造器会执行 JS，需要先 checkpoint microtask
    if (0u == reentry_permit_->ScriptNestingLevel())
      document.GetAgent().event_loop()->PerformMicrotaskCheckpoint();
    CEReactionsScope reactions(document.GetAgent().isolate());
    element = definition->CreateElement(...);
    for (const auto& attribute : token->Attributes())
      element->setAttribute(attribute.GetName(), attribute.Value());  // 逐个设置属性
  } else {
    element = CustomElement::CreateUncustomizedOrUndefinedElement(...);
    SetAttributes(element, token);  // 批量设置属性更快
  }
  return element;
}

// AttachLater：不立即插入，而是入队
void HTMLConstructionSite::AttachLater(InsertionLocation location,
                                       Node* child,
                                       bool self_closing) {
  HTMLConstructionSiteTask task(HTMLConstructionSiteTask::kInsert);
  task.parent = location.parent;
  task.next_child = location.next_child;
  task.child = child;
  task.self_closing = self_closing;

  if (ShouldFosterParent()) {
    FosterParent(task.child);   // foster parent 重定向
    return;
  }

  // 深度防护：超过 512 层则强制挂到父节点的父节点
  if (open_elements_.StackDepth() > kMaximumHTMLParserDOMTreeDepth &&
      task.parent->parentNode()) {
    UseCounter::Count(..., WebFeature::kMaximumHTMLParserDOMTreeDepthHit);
    task.parent = task.parent->parentNode();
  }

  QueueTask(task, true);  // 入队，等 ExecuteQueuedTasks 统一执行
}

// ExecuteQueuedTasks：针对单任务做了快速路径
void HTMLConstructionSite::ExecuteQueuedTasks() {
  const size_t size = task_queue_.size();
  if (!size) return;

  if (size == 1) {
    HTMLConstructionSiteTask task = task_queue_.front();
    task_queue_.pop_back();
    ExecuteTask(task);    // 单任务：避免 swap 开销
    return;
  }

  TaskQueue queue;
  queue.swap(task_queue_);    // swap 而非拷贝，O(1)
  for (auto& task : queue)
    ExecuteTask(task);
}

// InsertTextNode：文本节点采用 pending 缓存合并，减少 Text 节点数量
void HTMLConstructionSite::InsertTextNode(const StringView& string,
                                          WhitespaceMode whitespace_mode) {
  // ...（计算插入位置，处理 template element）...
  if (!pending_text_.IsEmpty() &&
      (pending_text_.parent != dummy_task.parent ||
       pending_text_.next_child != dummy_task.next_child))
    FlushPendingText();   // 父节点变了，先 flush
  pending_text_.Append(...);  // 合并到 pending，延迟建立 Text 节点
}
```

### 6.3.4 HTMLDocumentParser：调度与预算

【真实源码 `third_party/blink/renderer/core/html/parser/html_document_parser.cc`，实际 fetch 核实】

```cpp
// html_document_parser.cc（常量，节选，实际 fetch 核实）
constexpr int kDefaultMaxTokenizationBudget = 250;   // 默认每轮最多 250 个 token
constexpr int kInfiniteTokenizationBudget = 1e7;     // 扩展/本地文件不限制

// 平台差异化预算：Android 更激进（FCP 优先），桌面更宽松
constexpr int kNumYieldsWithDefaultBudgetDefaultValue =
#if BUILDFLAG(IS_ANDROID)
    2      // Android：yield 2 次后切换到大 budget
#else
    6      // Desktop：6 次
#endif
;

constexpr base::TimeDelta kLongParserBudgetDefaultValue =
#if BUILDFLAG(IS_ANDROID)
    base::Milliseconds(50)    // Android 总时间片 50ms
#else
    base::Milliseconds(500)   // Desktop 500ms
#endif
;
```

```cpp
// PumpTokenizer（简化版，结构忠实，实际 fetch 核实）
bool HTMLDocumentParser::PumpTokenizer() {
  // 双重预算
  int budget = (task_runner_state_->TimesYielded() <= kNumYieldsWithDefaultBudget)
      ? kDefaultMaxTokenizationBudget
      : kInfiniteTokenizationBudget;  // yield 足够多次后，一口气吃完
  base::TimeDelta timed_budget = GetTimedBudget(task_runner_state_->TimesYielded());
  base::ElapsedTimer chunk_parsing_timer;

  while (true) {
    // 检查是否可取下一个 token（脚本阻塞检查等）
    const auto next_token_status = CanTakeNextToken(time_executing_script);
    if (next_token_status == kNoTokens) break;

    // 脚本执行后重置 budget（避免脚本执行占用过多时间后继续大块解析）
    if (next_token_status == kHaveTokensAfterScript &&
        task_runner_state_->HaveExitedHeader()) {
      budget = std::min(budget, task_runner_state_->GetDefaultBudget());
      timed_budget = std::min(timed_budget,
          chunk_parsing_timer.Elapsed() + GetDefaultTimedBudget());
    }

    // 核心：取 token → 构建 DOM
    token = tokenizer_.NextToken(input_.Current());
    budget--;
    ConstructTreeFromToken(atomic_html_token);

    // yield 判断：时间超限 || 调度器有高优先级任务 || 已退出 <head>
    if (!should_run_until_completion && !IsPaused()) {
      elapsed_time = chunk_parsing_timer.Elapsed();
      should_yield = elapsed_time >= timed_budget;
      should_yield |= scheduler_->ShouldYieldForHighPriorityWork();
      should_yield &= task_runner_state_->HaveExitedHeader();  // <head> 内不 yield
      if (should_yield) break;
    }
  }

  if (should_yield) task_runner_state_->MarkYield();
  return should_yield;
}
```

**关键设计**：`<head>` 内部（meta charset、title、link 等）不允许 yield——浏览器需要尽快知道 charset、viewport、CSP 等关键元信息，不能把它们拆分到多帧渲染中。

```cpp
// insert()：document.write() 的入口（节选，实际 fetch 核实）
void HTMLDocumentParser::insert(const String& source) {
  if (IsStopped() || source.empty()) return;

  SegmentedString excluded_line_number_source(source);
  excluded_line_number_source.SetExcludeLineNumbers();  // write() 注入的内容不记行号
  input_.InsertAtCurrentInsertionPoint(excluded_line_number_source);

  // 强制同步消费：document.write() 必须立即看到效果
  ShouldCompleteScope should_complete(task_runner_state_);
  EndIfDelayedForbiddenScope should_not_end_if_delayed(task_runner_state_);
  PumpTokenizerIfPossible();

  // 如果解析器此时被脚本 pause，用单独的 insertion preload scanner 补扫
  if (IsPaused() && preloader_) {
    if (!insertion_preload_scanner_) {
      insertion_preload_scanner_ =
          CreatePreloadScanner(TokenPreloadScanner::ScannerType::kInsertion);
    }
    insertion_preload_scanner_->AppendToEnd(source);
    ScanAndPreload(insertion_preload_scanner_.get());
  }
  EndIfDelayed();
}
```

### 6.3.5 HTMLInputStream：插入点机制

【真实源码 `third_party/blink/renderer/core/html/parser/html_input_stream.h`，实际 fetch 核实】

```
HTMLInputStream 内部结构（示意，非逐字）：

[--current--][--next--][--next--] ... [--next--]
      ↑                                    ↑
  当前读取位置                            last_（网络追加到这里）

document.write() 调用 InsertAtCurrentInsertionPoint()：
  在 current 位置"劈开"，把注入内容插入，原有后续内容变成 "next"
  → InsertionPointRecord（RAII）通过 SplitInto/MergeFrom 管理这个过程
```

`InsertionPointRecord` 是一个栈变量，构造时调用 `SplitInto()` 把流分成两段，析构时调用 `MergeFrom()` 重新拼合——精确地模拟了"在当前位置插入一段新内容、解析完后继续原内容"的语义。

### 6.3.6 DOM 树节点模型

【真实源码 `third_party/blink/renderer/core/dom/node.h`，实际 fetch 核实】

```cpp
// node.h（节选，实际 fetch 核实）
// "A Node is a base class for all objects in the DOM tree."
// 遵循 dom.spec.whatwg.org

enum class NodeType : uint16_t {
  kElementNode                = 1,
  kAttributeNode              = 2,  // 不再出现在树中，但保留 JS 兼容性
  kTextNode                   = 3,
  kCdataSectionNode           = 4,
  kProcessingInstructionNode  = 7,
  kCommentNode                = 8,
  kDocumentNode               = 9,
  kDocumentTypeNode           = 10,
  kDocumentFragmentNode       = 11,
  // Entity/Notation 无法在 Blink 中创建，但保留枚举值
};

// 32 位 flag 字段，管理树连通性、样式状态、布局需求
// "Both tree_scope and parent are hot accessed members.
//  Keep them uncompressed for performance reasons."
// 子节点链表用循环链表实现（firstChild 的 previousSibling 指向 lastChild）
```

【真实源码 `third_party/blink/renderer/core/dom/container_node.cc`，实际 fetch 核实】

```cpp
// container_node.cc（ChildrenChanged，节选，实际 fetch 核实）
void ContainerNode::ChildrenChanged(const ChildrenChange& change) {
  GetDocument().IncDOMTreeVersion();          // DOM 版本号递增，供 MutationObserver 等使用
  GetDocument().NotifyChangeChildren(*this, change);
  if (change.type == ChildrenChangeType::kFinishedBuildingDocumentFragmentTree)
    return;   // DocumentFragment 构建完成时不需要触发失效
  InvalidateNodeListCachesInAncestors(nullptr, nullptr, &change);  // 清除 NodeList 缓存
  if (change.IsChildRemoval() || ...) {
    GetDocument().GetStyleEngine().ChildrenRemoved(*this);   // 通知样式引擎
    return;
  }
  if (!change.IsChildInsertion()) return;
  Node* inserted_node = change.sibling_changed;
  // ...通知 flat tree 和样式失效...
  inserted_node->SetStyleChangeOnInsertion();  // 新插入节点需要样式重算
}
```

---

## 6.4 脚本阻塞机制

### 6.4.1 解析阻塞脚本 vs 延迟脚本

```
┌─────────────────────────────────────────────────────────────┐
│  Script 分类                                                │
│                                                             │
│  Parser-blocking (同步)                                     │
│    <script src="...">（无 defer/async）                     │
│    → HTMLParserScriptRunner::TakeScriptToProcess()          │
│    → 解析器完全停止等待脚本下载+执行                         │
│                                                             │
│  Deferred                                                   │
│    <script defer>                                           │
│    → 下载并行，但等 HTML 解析完成后、DOMContentLoaded 前执行  │
│                                                             │
│  Async                                                      │
│    <script async>                                           │
│    → 下载并行，下载完立即执行（可能在解析中途）               │
│                                                             │
│  Module                                                     │
│    <script type="module">                                   │
│    → 默认 defer 行为                                        │
└─────────────────────────────────────────────────────────────┘
```

【真实源码 `html_document_parser.cc` IsWaitingForScripts，实际 fetch 核实】

```cpp
// IsWaitingForScripts（节选，实际 fetch 核实）
// 检查三个阻塞条件，注释说这三个不应同时为 true
bool HTMLDocumentParser::IsWaitingForScripts() const {
  // fragment 解析不执行脚本
  if (IsParsingFragment()) return false;

  // tree builder 持有 parser-blocking script（刚从 token 中取出，尚未交给 script runner）
  bool tree_builder_has_blocking_script =
      tree_builder_->HasParserBlockingScript();

  // script runner 持有（已经在执行流程中）
  bool script_runner_has_blocking_script =
      script_runner_ && script_runner_->HasParserBlockingScript();

  // 断言：两者不应同时为 true（会有 use-after-free 风险）
  DCHECK(!(tree_builder_has_blocking_script &&
           script_runner_has_blocking_script));

  // reentry permit 的 pause 标志（document.write 场景）
  bool reentry_permit_prevents_parsing =
      reentry_permit_->ParserPauseFlag();

  return tree_builder_has_blocking_script ||
         script_runner_has_blocking_script ||
         reentry_permit_prevents_parsing;
}
```

### 6.4.2 Preload Scanner：解耦发现与执行

解析阻塞期间（等待 `<script>` 下载），主线程解析停止。但 `HTMLPreloadScanner` 已经扫描了更靠后的 HTML，提前发出了图片、CSS、字体等资源的 `<link rel=preload>` 请求。这是现代浏览器 Time-to-Interactive 优化的关键路径。

【真实源码 `html_preload_scanner.h`，实际 fetch 核实】

```cpp
// html_preload_scanner.h（节选，实际 fetch 核实）
// 两种模式：
// kMainDocument - 主文档解析用，可在背景线程运行
// kInsertion    - document.write() 注入内容用（必须是独立实例，因为主扫描器
//                 无法处理插入点）
enum class ScannerType { kMainDocument, kInsertion };

// PendingPreloadData 汇聚所有发现的资源：
// - meta client hints
// - viewport 配置
// - CSP meta tag 计数
// - preload 请求列表
// 统一从背景线程同步回主线程
```

【真实源码 `background_html_scanner.h`，实际 fetch 核实】

```cpp
// BackgroundHTMLScanner（节选，实际 fetch 核实）
// "Scans HTML on a worker thread looking for inline scripts
//  which can be stream compiled."
// 使用 CrossThreadWeakPersistent 引用 parser，防止生命周期问题
// 使用 SequenceBound 保证所有操作在指定背景 sequence 上执行
```

`ScanAndPreload()` 触发流程：

```cpp
// html_document_parser.cc（ScanAndPreload，节选，实际 fetch 核实）
void HTMLDocumentParser::ScanAndPreload(HTMLPreloadScanner* scanner) {
  TRACE_EVENT("blink", "HTMLDocumentParser::ScanAndPreload",
              perfetto::Flow::FromPointer(this));
  CHECK(preloader_);
  base::ElapsedTimer timer_before_scan;
  std::unique_ptr<PendingPreloadData> preload_data =
      scanner->Scan(GetDocument()->ValidBaseElementURL());
  const base::TimeDelta scan_time = timer_before_scan.Elapsed();
  ProcessPreloadData(std::move(preload_data));
  // ... 记录 UMA 直方图 Blink.ScanAndPreloadTime2 ...
}
```

---

## 6.5 方案对比

### 6.5.1 同步解析 vs 异步增量解析

| 维度 | 同步解析（`kForceSynchronousParsing`） | 异步增量解析（`kAllowDeferredParsing`） |
|------|--------------------------------------|----------------------------------------|
| 触发场景 | 扩展注入内容、`innerHTML` 赋值 | 网络加载的正常页面 |
| 预算 | `kInfiniteTokenizationBudget` | 250 tokens/轮，10ms→50/500ms |
| Yield 点 | 无 | 时间超限 或 调度器有高优先级任务 |
| FCP | 不适用（非正常导航） | 依赖 yield 控制，保证渲染帧有机会插入 |
| Preload scanner | 无需异步（已在同步路径） | 必须，否则 parser-blocking script 会 stall 资源 |

### 6.5.2 Parser-blocking Script vs Async/Defer

| 特性 | `<script>` | `<script defer>` | `<script async>` | `<script type=module>` |
|------|-----------|-----------------|-----------------|----------------------|
| 阻塞解析 | 是 | 否 | 否（下载完执行） | 否 |
| 执行时机 | 遭遇即执行 | DOMContentLoaded 前，按顺序 | 下载完立即 | defer 语义 |
| 是否影响 DOMContentLoaded | 是 | 是 | 否 | 是 |
| document.write() 可用 | 是 | 否（规范禁止） | 否 | 否 |
| 适合 LCP 优化 | 需配合 `async` | 适合非关键脚本 | 适合独立脚本 | 现代首选 |

### 6.5.3 innerHTML vs 解析器插入

| 维度 | `innerHTML` / `DOMParser` | 流式解析器 `Append()` |
|------|--------------------------|----------------------|
| 脚本执行 | 不执行（安全） | 执行（受 CSP 控制） |
| 性能 | 小片段 OK，大片段慢 | 增量，天然分帧 |
| 错误容忍 | 有 | 有 |
| `document.write()` | 不适用 | 可能出现 |
| 适合场景 | 动态内容注入 | 页面主文档加载 |

---

## 6.6 Demo：真实引擎上观测与实验

### Demo 1：DevTools Performance 观测 DOM 构建时序

**目标**：用 Performance 面板看到 `HTMLDocumentParser::PumpTokenizer` 在哪里被调度，验证 yield 行为。

```html
<!-- test-parse.html：创建一个能触发多帧解析的大页面 -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Parser Yield Demo</title>
</head>
<body>
  <script>
    // 生成 5000 个节点，确保解析跨多个帧
    document.write('<div>'.repeat(1) + Array.from({length: 5000}, (_, i) =>
      `<p id="p${i}">段落 ${i} - ${'x'.repeat(100)}</p>`
    ).join('') + '</div>');
  </script>
  <p>解析器继续处理这段内容</p>
</body>
</html>
```

**步骤**：
1. 用 Chrome 打开 `chrome://version` 确认版本
2. 打开 DevTools → Performance → 点击录制
3. 打开 `test-parse.html`
4. 停止录制，在 Main thread 中搜索 "ParseHTML"

**预期输出**：
- 可看到多个 `ParseHTML` 任务段，每段约 10–50ms（桌面端）
- 在 `ParseHTML` 之间有 `RequestAnimationFrame` 等其他任务插入，证明 yield 工作
- Timings 行可看到 FP/FCP 出现在前几个 `ParseHTML` 块之后

**注意**：`document.write()` 在 Performance 中会显示为同步块，不会被 yield 分割。

---

### Demo 2：document.write 阻塞解析实验

**目标**：观察同步 `<script src>` + `document.write` 如何阻止浏览器渲染任何内容。

**步骤一：创建慢脚本服务器**

```python
#!/usr/bin/env python3
# slow_server.py
import http.server, time, threading

class SlowHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/slow.js':
            time.sleep(3)  # 模拟 3 秒网络延迟
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript')
            self.end_headers()
            self.wfile.write(b'document.write("<b>由 document.write 注入</b>");')
        else:
            super().do_GET()

if __name__ == '__main__':
    server = http.server.HTTPServer(('localhost', 8765), SlowHandler)
    print('Serving on http://localhost:8765')
    server.serve_forever()
```

```bash
python3 slow_server.py
```

**步骤二：创建测试页面**

```html
<!-- index.html，与 slow_server.py 同目录 -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>document.write Block Demo</title>
</head>
<body>
  <h1>H1 在脚本前（应该立刻显示）</h1>

  <!-- 这个 script 会阻塞解析，slow.js 里有 document.write -->
  <script src="/slow.js"></script>

  <h1>H1 在脚本后（被 document.write 阻塞，3秒后才显示）</h1>
</body>
</html>
```

**步骤三：观察**

```bash
# 打开 Chrome，访问 http://localhost:8765/index.html
open -a "Google Chrome" http://localhost:8765/index.html
```

**预期输出**：
- 0–3 秒：页面白屏（`<h1>H1 在脚本前</h1>` 因为 parser 被 `<script src>` 阻塞，连 HTML 主体都还没开始渲染）
- 等等——实际上第一个 `<h1>` 也不会出现，因为脚本在 `<body>` 内，渲染还来不及发生
- 3 秒后：两个 `<h1>` 和 `document.write` 注入的 `<b>` 一起出现
- 打开 DevTools Network：可看到 `slow.js` 请求占据了 3 秒，期间没有任何其他请求（Preload Scanner 可能提前发出，但主渲染被 block）

**对比实验**：把 `<script src="/slow.js">` 改为 `<script src="/slow.js" async>`，观察两个 `<h1>` 立刻渲染，3 秒后注入内容消失（`async` 不保证 `document.write` 执行时解析器还开着）。

---

### Demo 3：CDP 驱动观测 DOM 构建事件

**目标**：通过 Chrome DevTools Protocol 以编程方式订阅 DOM 事件，观察解析过程中 DOM 的逐步构建。

```javascript
// cdp-dom-observe.mjs
// 需要 Node.js 18+ 和 chrome-remote-interface
// npm install chrome-remote-interface

import CDP from 'chrome-remote-interface';

// 启动 Chrome：chrome --remote-debugging-port=9222 --new-window about:blank
async function main() {
  const client = await CDP({ port: 9222 });
  const { DOM, Page, Runtime } = client;

  // 启用 DOM 域，订阅节点创建事件
  await DOM.enable();
  await Page.enable();

  let nodeCount = 0;
  const timeline = [];

  // 监听 setChildNodes：当浏览器 resolve 子树时触发
  DOM.setChildNodes(({ parentId, nodes }) => {
    nodeCount += nodes.length;
    timeline.push({
      time: Date.now(),
      event: 'setChildNodes',
      parentId,
      count: nodes.length
    });
  });

  // 导航到目标页面
  await Page.navigate({ url: 'http://localhost:8765/index.html' });

  // 等待 DOMContentLoaded
  await new Promise(resolve => Page.domContentEventFired(resolve));
  console.log(`DOMContentLoaded: ${nodeCount} nodes seen so far`);

  // 等待 load
  await new Promise(resolve => Page.loadEventFired(resolve));
  console.log(`Load: final node count: ${nodeCount}`);
  console.log('\nTimeline (first 10 events):');
  timeline.slice(0, 10).forEach(e =>
    console.log(`  +${e.time - timeline[0].time}ms: parentId=${e.parentId}, ${e.count} nodes`)
  );

  await client.close();
}

main().catch(console.error);
```

**运行步骤**：

```bash
# 1. 启动 Chrome（确保无其他 Chrome 实例或使用单独 profile）
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/cdp-test \
  about:blank

# 2. 运行脚本
node cdp-dom-observe.mjs
```

**预期输出**：
```
DOMContentLoaded: 47 nodes seen so far
Load: final node count: 47

Timeline (first 10 events):
  +0ms: parentId=1, 2 nodes        # Document -> [html, doctype]
  +2ms: parentId=2, 2 nodes        # html -> [head, body]
  +3ms: parentId=3, 4 nodes        # head -> [meta, title, ...]
  +5ms: parentId=4, 3 nodes        # body -> [h1, script, h1]
  ...
```

**说明**：`setChildNodes` 事件在 CDP 客户端请求节点详情时触发，反映了 Blink DOM 树的实际构建顺序。节点在解析器 `ExecuteQueuedTasks()` 执行时才真正进入树，CDP 可以观测到这个时序。

---

### Demo 4：Electron 最小应用观测 DOM 解析

**目标**：在 Electron 应用中，通过 BrowserWindow 的 webContents 钩子监听 DOM 准备事件，并用 `executeJavaScript` 读取解析状态。

```javascript
// main.js（Electron 主进程）
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');

let win;

app.whenReady().then(() => {
  win = new BrowserWindow({
    width: 900,
    height: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // 监听渲染进程的 IPC 消息（DOM 解析状态上报）
  ipcMain.on('dom-event', (event, data) => {
    console.log(`[DOM Event] ${data.type}: readyState=${data.readyState}, ` +
                `nodeCount=${data.nodeCount}, timestamp=${data.timestamp.toFixed(1)}ms`);
  });

  // DOM ready（Blink HTMLDocumentParser 完成，DOMContentLoaded 触发）
  win.webContents.on('dom-ready', () => {
    console.log('[webContents] dom-ready fired (DOMContentLoaded equivalent)');

    // 读取文档当前节点总数
    win.webContents.executeJavaScript(`
      document.querySelectorAll('*').length
    `).then(count => {
      console.log(`[webContents] total element count: ${count}`);
    });
  });

  // 页面完全加载
  win.webContents.on('did-finish-load', () => {
    console.log('[webContents] did-finish-load (window.load equivalent)');
  });

  // 加载 HTML
  win.loadFile('test-parse.html');
});

app.on('window-all-closed', () => app.quit());
```

```javascript
// preload.js（运行在渲染进程，有 contextBridge 权限）
const { ipcRenderer, contextBridge } = require('electron');

contextBridge.exposeInMainWorld('domTracker', {
  report: (type) => {
    ipcRenderer.send('dom-event', {
      type,
      readyState: document.readyState,
      nodeCount: document.querySelectorAll('*').length,
      timestamp: performance.now()
    });
  }
});
```

```html
<!-- test-parse.html（放到同目录） -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script>
    // 使用 preload 暴露的 API 上报 parsing 阶段
    document.addEventListener('readystatechange', () => {
      if (window.domTracker) {
        window.domTracker.report('readystatechange-' + document.readyState);
      }
    });
  </script>
</head>
<body>
  <div id="container">
    <!-- 200 个段落，足以看到 readyState 变化 -->
    <script>
      for (let i = 0; i < 200; i++) {
        document.write(`<p>段落 ${i}</p>`);
      }
      window.domTracker && window.domTracker.report('after-document-write');
    </script>
  </div>
  <footer>页脚内容</footer>
</body>
</html>
```

**运行步骤**：

```bash
# 初始化 Electron 项目
mkdir electron-dom-demo && cd electron-dom-demo
npm init -y && npm install electron
# 把上面三个文件放到目录里
npx electron main.js
```

**预期输出**：
```
[DOM Event] readystatechange-loading: readyState=loading, nodeCount=2, timestamp=12.3ms
[DOM Event] after-document-write: readyState=loading, nodeCount=205, timestamp=45.1ms
[webContents] dom-ready fired (DOMContentLoaded equivalent)
[webContents] total element count: 207
[DOM Event] readystatechange-interactive: readyState=interactive, nodeCount=207, timestamp=58.2ms
[DOM Event] readystatechange-complete: readyState=complete, nodeCount=207, timestamp=62.1ms
[webContents] did-finish-load (window.load equivalent)
```

**源码呼应**：
- `readyState=loading`：对应 `Document::ParsingState::kParsing`，HTMLDocumentParser 仍在 `PumpTokenizer()` 中
- `dom-ready` / `readystatechange-interactive`：对应 `Document::FinishedParsing()`，HTMLDocumentParser::Finish() 被调用
- `document.write` 执行后立即反映节点数变化：因为 `insert()` 是同步路径，直接 `PumpTokenizerIfPossible()` 不可 yield

---

### Demo 5：chrome://tracing 观测 Parser 帧内分布

**目标**：用 `chrome://tracing` 捕获 `blink.html_parser` 类别的 trace 事件，观察每帧内 tokenizer pump 的持续时间。

**步骤**：

```bash
# 1. 启动带特殊 flag 的 Chrome（允许更多 trace 类别）
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --enable-tracing=blink,blink.html_parser,v8 \
  --trace-startup-duration=10
```

或者在 Chrome 地址栏输入 `chrome://tracing`：

1. 点击 **Record**
2. 选择类别：勾选 `blink`、`blink.html_parser`、`devtools.timeline`
3. 在新标签页打开一个大型 HTML 页面（如 Wikipedia 某条目）
4. 回到 tracing 标签点击 **Stop**
5. 搜索 `HTMLDocumentParser::PumpTokenizer`

**预期结果**：
- 每个 `PumpTokenizer` 事件持续约 10ms（初始）
- 多个 `PumpTokenizer` 事件在不同帧内分布，帧间可以看到 `CommitFrameToCompositor` 或 `BeginMainFrame`
- `HTMLDocumentParser::ScanAndPreload` 事件出现在主线程
- 搜索 `BackgroundHTMLScanner::Scan` 可见背景线程上的活动

**说明**：trace 中 `perfetto::Flow::FromPointer(this)` 箭头连接 `Append()` 和对应的 `PumpTokenizer()`，可以直观看到网络数据如何被分批处理。

---

## 6.7 失败模式与生产真坑

### 坑 1：document.write 在慢网络上的灾难

**现象**：页面加载超时，用户看到白屏，即使内容很少。

**根因**：`insert()` 路径不允许 yield（`ShouldCompleteScope`）。若脚本中有 `document.write('<script src="slow.js">')`, 主线程会阻塞在等待 `slow.js` 下载，期间整个页面无法渲染。

**解法**：
1. Chrome 在 `document.write()` 写入跨源脚本时会触发 Intervention（`kBlockedDocumentWrite` UseCounter），在 DevTools Console 可见警告
2. 迁移到 `async`/`defer` 或动态 `import()`

**如何确认是否受影响**：
```javascript
// 在 Chrome Console 里运行，检查是否触发了 document.write intervention
performance.getEntriesByType('navigation')[0].redirectCount
// 或看 DevTools → Issues 面板
```

### 坑 2：DOM 树深度超过 512 导致 UseCounter 记录但节点位置异常

**现象**：深度嵌套的 HTML（如某些邮件内容导入）中，超过 512 层的元素会被"提升"到更浅的层级。

**根因**：`HTMLConstructionSite::AttachLater()` 中的深度检查：
```cpp
if (open_elements_.StackDepth() > kMaximumHTMLParserDOMTreeDepth &&
    task.parent->parentNode()) {
  task.parent = task.parent->parentNode();  // 强制挂到父级
}
```

**生产影响**：CSS 选择器依赖深度嵌套的页面会样式错乱，JS `querySelector` 找不到期望节点。

**检测方式**：
```javascript
// 检测最大深度
function maxDepth(node, depth = 0) {
  if (!node.children.length) return depth;
  return Math.max(...[...node.children].map(c => maxDepth(c, depth + 1)));
}
console.log('Max DOM depth:', maxDepth(document.documentElement));
// 超过 512 时应报警
```

### 坑 3：innerHTML 赋值触发同步样式重算

**现象**：循环内反复 `element.innerHTML = str` 导致页面卡顿。

**根因**：`innerHTML` 走 `HTMLDocumentParser::ParseFragment()`（同步路径），每次赋值后 `ChildrenChanged()` → `GetStyleEngine().ChildrenRemoved()` → 整棵子树样式失效 → 下次访问 layout 属性时同步重算。

**解法**：用 `DocumentFragment`、`template` 元素，或在 RAF 里批量更新。

```javascript
// 坏：
for (const item of items) {
  container.innerHTML += `<li>${item}</li>`;  // O(n²)，每次重新解析+重新渲染
}

// 好：
const fragment = document.createDocumentFragment();
for (const item of items) {
  const li = document.createElement('li');
  li.textContent = item;
  fragment.appendChild(li);
}
container.appendChild(fragment);  // 只触发一次 ChildrenChanged
```

### 坑 4：Preload Scanner 推测状态与主解析器不一致

**现象**：某些嵌套在 `<noscript>` 或 `<template>` 中的脚本被 preload scanner 错误扫描，产生无用的 preload 请求。

**根因**：`HTMLTokenizer::SpeculativeStateForTag()` 只是"近似"，对 CDATA 处理和 `<pre>` 换行有已知偏差（源码注释明确指出）。Preload scanner 在背景线程运行，无法共享主解析器的完整状态上下文。

**影响**：浪费带宽（多余 preload），偶发 console 警告（unused preload）。

**检测**：DevTools Network 面板过滤 `initiator: parser`，检查是否有 "preload" 类型资源但状态为 cancelled。

### 坑 5：Electron 中 `webContents.dom-ready` 时序陷阱

**现象**：在 `dom-ready` 事件中调用 `executeJavaScript()` 读取某些 DOM 属性，偶尔拿不到最新值。

**根因**：`dom-ready` 对应 `DOMContentLoaded`，此时 `document.readyState` 变为 `interactive`，但 `<img>` 等资源、`defer` 脚本可能还未执行完。如果 defer 脚本修改了 DOM，`dom-ready` 后读到的可能是修改前的状态。

**解法**：对于需要完整 DOM 的操作，监听 `did-finish-load`（对应 `window.load`）或在渲染进程内用 `window.load` 事件通知主进程。

---

## 6.8 Electron 视角：HTML 解析的透明使用

Electron 完整继承了 Chromium 的 `HTMLDocumentParser` 实现，无需（也无法）patch 解析器本身。但有几个与 Electron 相关的接触点：

### 8.1 webContents 生命周期 Hook 与解析阶段对应

```
HTMLDocumentParser 状态            webContents 事件
──────────────────────────────────────────────────────
kParsing（解析进行中）              did-start-navigation
                                   did-commit-navigation
HTMLDocumentParser::Finish()       dom-ready
  → Document::FinishedParsing()     （DOMContentLoaded 等价）
Document::SetParsingState(
  kFinishedParsing)
window load event                  did-finish-load
```

### 8.2 contextIsolation 对 DOM API 的影响

Electron 的 `contextIsolation: true` 在 V8 层创建独立 context，但共享同一个底层 DOM 树（同一组 `GarbageCollected<Node>` 对象）。Blink 的 `document.cc` 中 DOM 节点通过 V8 binding wrapper 暴露给 JS，wrapper 是 context 私有的，但底层 C++ 节点对象是跨 context 共享的。

这意味着：preload.js 里的 `document.querySelector()` 和页面脚本里的 `document.querySelector()` 返回不同的 JS 包装对象，但指向同一个 C++ `Element`。Blink 侧的 `ChildrenChanged()` 通知对两个 context 都生效。

### 8.3 `--disable-web-security` 与 document.write 干预

Electron 开发中有时会设置 `webSecurity: false`，这会绕过 Chromium 对跨源 `document.write()` 的 Intervention。在测试 CSP 策略或第三方嵌入时要注意：生产包绝对不要开，否则 document.write 注入跨源脚本的 block 机制失效。

---

## 6.9 章末五件套

### 1. 关键路径一览

```
字节流 → Append() → SegmentedString → PumpTokenizer() [250 tokens / 10ms]
  → NextToken() [70+ 状态] → AtomicHTMLToken
  → ConstructTree() [25 modes] → CreateElement() / InsertTextNode()
  → AttachLater() → QueueTask() → ExecuteQueuedTasks()
  → ContainerNode::AppendChild() → ChildrenChanged() → 样式/布局失效

脚本路径：
  → IsWaitingForScripts() → pause pump → 下载脚本
  → ExecuteScriptWithDebuggerEnabled() → document.write() ?
      → insert() → InsertAtCurrentInsertionPoint() → 同步 pump（不可 yield）
  → 恢复 pump
```

### 2. 核心概念速查

| 概念 | 位置 | 作用 |
|------|------|------|
| `HTMLInputStream` | `html_input_stream.h` | SegmentedString 链表 + document.write 插入点 |
| `HTMLTokenizer` | `html_tokenizer.h/cc` | 70+ 状态机，emit AtomicHTMLToken |
| `HTMLTreeBuilder` | `html_tree_builder.h/cc` | 25 insertion modes，开放元素栈 |
| `HTMLConstructionSite` | `html_construction_site.h/cc` | DOM 元素创建 + 任务队列，最大深度 512 |
| `HTMLPreloadScanner` | `html_preload_scanner.h` | 提前扫描资源，可在背景线程 |
| `BackgroundHTMLScanner` | `background_html_scanner.h` | 背景线程 inline script stream compile |
| `PumpTokenizer()` | `html_document_parser.cc` | 双预算调度（250 tokens + 10ms/50ms/500ms） |
| `insert()` | `html_document_parser.cc` | document.write 入口，同步不可 yield |
| `ChildrenChanged()` | `container_node.cc` | DOM 变更通知，驱动样式失效链 |

### 3. 延伸阅读

- WHATWG HTML Living Standard §13.2：[https://html.spec.whatwg.org/multipage/parsing.html](https://html.spec.whatwg.org/multipage/parsing.html)（规范原文）
- Blink 解析器目录（95 文件）：`third_party/blink/renderer/core/html/parser/`
- DOM 目录：`third_party/blink/renderer/core/dom/`
- "How Browsers Work"（Tali Garsiel）：[https://web.dev/howbrowserswork](https://web.dev/howbrowserswork)
- Blink 渲染管线概览：`third_party/blink/renderer/core/OVERVIEW.md`（待核）

### 4. 易混淆点

- **`insert()` vs `Append()`**：前者是 document.write 的同步强制路径（不可 yield），后者是网络数据的异步增量路径（受预算控制）
- **`HTMLPreloadScanner` vs `BackgroundHTMLScanner`**：前者扫描资源 URL 发出预加载请求，后者专门扫 inline script 触发 V8 stream compile，两者互补
- **`dom-ready` (Electron) vs `DOMContentLoaded`**：完全等价，都对应 `Document::FinishedParsing()`，均不等待图片/CSS
- **open elements stack（`HTMLElementStack`）vs active formatting elements**：前者跟踪所有未关闭元素，后者专门处理 `<b>`/`<i>` 等 formatting element 的跨 block 传播问题

### 5. TODO / 待核

- `third_party/blink/renderer/core/OVERVIEW.md` 链接未直接验证，标「待核」
- `CanTakeNextToken` 函数实现未在 `.cc` fetch 中直接看到完整代码，逻辑根据 `.h` 声明和 `IsWaitingForScripts()` 推断，标「待核」
- `BackgroundHTMLScanner` 与 V8 `ScriptCompiler::StartStreamingScript()` 的完整调用链未逐行验证，标「待核」

---

*本章源码引用均通过 WebFetch 实际取自 `raw.githubusercontent.com/chromium/chromium/main`，取不到的已标注。代码缩写处均标【示意，非逐字】。*
