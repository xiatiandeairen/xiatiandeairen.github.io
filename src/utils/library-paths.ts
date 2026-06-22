import { getLibraryChapters, LIBRARY_COLLECTIONS, type LibraryChapter } from './library';

// Learning paths = curated, goal-oriented reading routes that cut ACROSS
// collections. The library's collections are organized by subject (推理引擎,
// 检索底座, …); a path re-sequences chapters from several collections into one
// employment-oriented track ("想做推理工程师,按这个顺序读这些章"). Paths add
// navigation only — they never duplicate chapter content. Each step references a
// chapter by (collection, slug); titles/summaries/minutes are resolved from the
// content files at build time, so the chapter file stays the single source of
// truth (see getLearningPaths()).

export interface PathStepRef {
  collection: string;
  slug: string;
  // One line: why THIS chapter, in THIS position. The value a path adds over the
  // raw collection index is exactly this ordering rationale.
  why: string;
}

export interface LearningPathMeta {
  id: string;
  title: string;
  // Who this is for + the job/role it targets.
  audience: string;
  // One sentence: what you can DO after finishing the path.
  goal: string;
  steps: PathStepRef[];
}

// v1 scope: employment-first, tech-priority (per product decision 2026-06-21:
// 就业 tech 优先 · 窄而透). Four high-demand AI roles, each backed end-to-end by
// existing chapters. Deliberately NOT covering general-systems roles
// (chromium/android/webrtc/linux/db/compiler) yet — depth over breadth.
export const LEARNING_PATHS: LearningPathMeta[] = [
  {
    id: 'math-for-builders',
    title: '开发者的数学与物理 · 实用路径',
    audience: '面向就业 / 生活 · 想把数学物理接到 CS 工作与日常决策',
    goal: '按计算机实用性重排数学物理，每章配一个可跑 demo——从贝叶斯、PCA、反向传播到 FFT、MCMC、从晶体管搭加法器，把知识做出来。',
    steps: [
      { collection: 'math-for-builders', slug: '00', why: '全景:最高 ROI 的四块先学,计算机权重的取舍地图' },
      { collection: 'math-for-builders', slug: '01', why: '概率与贝叶斯:决策、分类、不被基率谬误骗' },
      { collection: 'math-for-builders', slug: '02', why: '统计与推断:A/B、p 值真义、辛普森悖论' },
      { collection: 'math-for-builders', slug: '03', why: '线性代数:PCA/SVD/embedding,数据的语言' },
      { collection: 'math-for-builders', slug: '04', why: '微积分与最优化:手推一次反向传播' },
      { collection: 'math-for-builders', slug: '05', why: '离散数学与逻辑:算法骨架与复杂度直觉' },
      { collection: 'math-for-builders', slug: '06', why: '数值方法:浮点坑与稳定算法' },
      { collection: 'math-for-builders', slug: '07', why: '信息论:熵=压缩下界=ML 损失' },
      { collection: 'math-for-builders', slug: '08', why: '数论与密码学:从零 RSA 与单向函数' },
      { collection: 'math-for-builders', slug: '09', why: '计算理论:可判定性与 P/NP' },
      { collection: 'math-for-builders', slug: '10', why: '傅里叶与信号:时域↔频域、FFT、采样定理' },
      { collection: 'math-for-builders', slug: '11', why: '统计力学↔ML:MCMC、退火、伊辛相变' },
      { collection: 'math-for-builders', slug: '12', why: '计算机为什么能算:晶体管→逻辑门→加法器' },
    ],
  },
  {
    id: 'database-engineer-from-scratch',
    title: '数据库工程师 · 从零构建',
    audience: '面向就业 · 后端 / 基础设施 / 数据库内核',
    goal: '不用现成数据库，亲手实现存储页、缓冲池、B+树、LSM、WAL、MVCC、崩溃恢复与查询执行，看懂一条 SELECT 背后的全部零件。',
    steps: [
      { collection: 'database', slug: '00', why: '全景:一条 SELECT 背后要造哪 7 个子系统' },
      { collection: 'database', slug: '01', why: '存储与页:把 ArrayBuffer 当磁盘,slotted page 塞变长行' },
      { collection: 'database', slug: '02', why: '缓冲池:LRU/Clock、pin、脏页,内存不够谁先踢' },
      { collection: 'database', slug: '03', why: 'B+树:扇出/分裂/高度,三层装十亿行' },
      { collection: 'database', slug: '04', why: 'LSM 树:随机写变顺序写,写/读/空间放大三角' },
      { collection: 'database', slug: '05', why: 'WAL:先写日志,掉电不丢已提交事务' },
      { collection: 'database', slug: '06', why: '事务与 MVCC:快照、版本链、四种隔离级别异常' },
      { collection: 'database', slug: '07', why: '崩溃恢复:ARIES 简化版 Analysis/Redo/Undo' },
      { collection: 'database', slug: '08', why: '查询执行:SQL→解析→计划→火山模型算子' },
    ],
  },
  {
    id: 'vector-search-engineer-from-scratch',
    title: '向量检索工程师 · 从零构建',
    audience: '面向就业 · 做检索 / RAG 底座 / 向量库',
    goal: '不用现成向量库，亲手实现倒排/BM25、HNSW、IVF-PQ、Hybrid+Rerank，并用合成数据实测 recall / QPS / 内存三角权衡。',
    steps: [
      { collection: 'vector-search', slug: '00', why: '先建地图:维度灾难 + 精确vs近似的 recall/速度/内存三角' },
      { collection: 'vector-search', slug: '01', why: '关键词基线:倒排索引 + BM25,也是 Hybrid 的一半' },
      { collection: 'vector-search', slug: '02', why: '向量与相似度:cos/dot/l2 + 维度灾难为何逼出近似' },
      { collection: 'vector-search', slug: '03', why: '暴力 KNN:recall 上限/速度下限,后续近似法的对照基线' },
      { collection: 'vector-search', slug: '04', why: 'HNSW 图索引:ef/M 旋钮,实测 recall↔QPS 权衡' },
      { collection: 'vector-search', slug: '05', why: 'IVF + PQ 量化:用内存换召回,nprobe 权衡' },
      { collection: 'vector-search', slug: '06', why: 'Hybrid 融合 + Rerank:关键词与语义两路互补' },
      { collection: 'vector-search', slug: '07', why: '评测与生产化:recall@k/nDCG、分片、过滤、更新' },
    ],
  },
  {
    id: 'rl-posttrain-engineer-from-scratch',
    title: '后训练 / 对齐工程师 · 从零构建',
    audience: '面向就业 / 科研 · RL、对齐、后训练方向',
    goal: '不用 RL 库，亲手实现 bandit、策略梯度、PPO、奖励模型、RLHF、DPO、GRPO 与可验证奖励，在玩具环境实测奖励曲线与 reward hacking。',
    steps: [
      { collection: 'rl-posttrain', slug: '00', why: '全景:预训练完为什么还要 RL/后训练' },
      { collection: 'rl-posttrain', slug: '01', why: '多臂老虎机:探索 vs 利用与 regret 这把尺' },
      { collection: 'rl-posttrain', slug: '02', why: 'Gridworld 与价值:用已知最优解校准 agent' },
      { collection: 'rl-posttrain', slug: '03', why: '策略梯度 REINFORCE:方差才是真正的敌人' },
      { collection: 'rl-posttrain', slug: '04', why: 'PPO 从零:clip/KL,不把策略训崩' },
      { collection: 'rl-posttrain', slug: '05', why: '奖励模型:从成对偏好学出标量奖励' },
      { collection: 'rl-posttrain', slug: '06', why: 'RLHF 全流程与 reward hacking:KL 是缰绳' },
      { collection: 'rl-posttrain', slug: '07', why: 'DPO:扔掉奖励模型,直接从偏好优化' },
      { collection: 'rl-posttrain', slug: '08', why: 'GRPO:去 critic 化与组内相对优势' },
      { collection: 'rl-posttrain', slug: '09', why: 'RL for reasoning:可验证奖励与综合冲刺' },
    ],
  },
  {
    id: 'dl-framework-from-scratch',
    title: '深度学习 / 算法工程师 · 从零构建训练框架',
    audience: '面向就业 / 科研 · 算法、训练、后训练方向',
    goal: '不用 PyTorch，亲手实现 autograd、层、优化器、自注意力与 Transformer，最后在玩具语料上训出一个会收敛的 nanoGPT。',
    steps: [
      { collection: 'tiny-dl', slug: '01', why: '地基:标量 autograd——一个会反向传播的数' },
      { collection: 'tiny-dl', slug: '02', why: '把标量计算图升维到 n 维张量引擎' },
      { collection: 'tiny-dl', slug: '03', why: '层与激活:Module/Linear 与非线性的梯度形状' },
      { collection: 'tiny-dl', slug: '04', why: '优化器:从 SGD 到 Adam,谁在动参数' },
      { collection: 'tiny-dl', slug: '05', why: '训练循环与诊断:loss 不降时怎么查' },
      { collection: 'tiny-dl', slug: '06', why: '自注意力从零:手写一个能反向的 attention 头' },
      { collection: 'tiny-dl', slug: '07', why: 'Transformer block:残差/LayerNorm/前馈装配' },
      { collection: 'tiny-dl', slug: '08', why: 'nanoGPT 真训练:在玩具语料上看它收敛' },
    ],
  },
  {
    id: 'llm-inference-engineer-from-scratch',
    title: 'LLM 推理工程师 · 从零构建',
    audience: '面向就业 · 进推理引擎 / infra 团队',
    goal: '不用现成推理框架，亲手实现前向、KV 缓存、采样、连续批处理、PagedAttention、投机解码、量化，并用 toy 权重实测 tok/s、延迟与显存。',
    steps: [
      { collection: 'llm-inference', slug: '00', why: '先建地图:自回归 + prefill/decode 两阶段 + tok/s 的真相' },
      { collection: 'llm-inference', slug: '01', why: 'tiny transformer 前向:token→logits,注意力 O(seq²) 的根' },
      { collection: 'llm-inference', slug: '02', why: 'KV 缓存:把 decode 从 O(n²) 砍成 O(n),回报最大的一步' },
      { collection: 'llm-inference', slug: '03', why: '采样:温度/top-k/top-p,decode 热路径上的取舍' },
      { collection: 'llm-inference', slug: '04', why: '连续批处理:让 GPU 不空转,把吞吐榨干' },
      { collection: 'llm-inference', slug: '05', why: 'PagedAttention:像 OS 管内存一样管 KV 缓存碎片' },
      { collection: 'llm-inference', slug: '06', why: '投机解码:小模型抢跑、大模型一次验一串降延迟' },
      { collection: 'llm-inference', slug: '07', why: '量化 int8/int4:用精度换显存与带宽' },
      { collection: 'llm-inference', slug: '08', why: '全栈基准:所有优化同台,把推理账算到底 + 面试冲刺' },
    ],
  },
  {
    id: 'agent-engineer-from-scratch',
    title: 'Agent 工程师 · 从零构建',
    audience: '面向就业 · 做 AI 原生应用 / Agent 平台',
    goal: '不用框架，亲手把一个生产级 Agent 从循环、工具、沙箱、权限、上下文、记忆，到检索、规划、反思、多 Agent、编排、评测、可观测、可靠性、成本逐个写出来并跑通，最后接成一个统一 agent。',
    steps: [
      { collection: 'agent-engineering', slug: '00', why: '先建地图:Agent = 循环+工具+终止条件,以及各器官怎么咬合' },
      { collection: 'agent-engineering', slug: '01', why: '地基:40 行写出控制循环,看清停止条件与 O(T²) 成本' },
      { collection: 'agent-engineering', slug: '02', why: '给循环装手:function calling 报文 + 从零手写 MCP' },
      { collection: 'agent-engineering', slug: '03', why: '挡"能不能做":隔离不可信命令,沙箱实测拦截' },
      { collection: 'agent-engineering', slug: '04', why: '挡"准不准做":权限三态 + 复现 lethal trifecta 注入外泄' },
      { collection: 'agent-engineering', slug: '05', why: '还第 1 章的 O(T²) 债:压缩/卸载/KV 缓存把窗口当预算' },
      { collection: 'agent-engineering', slug: '06', why: 'context 之外的长期记忆:分层召回、遗忘与记忆投毒' },
      { collection: 'agent-engineering', slug: '07', why: '知识:agent 何时该检索,agentic retrieval 与检索投毒' },
      { collection: 'agent-engineering', slug: '08', why: '大脑:规划与推理,ReAct vs Plan-and-Execute' },
      { collection: 'agent-engineering', slug: '09', why: '大脑:反思与自我纠错,critic/verifier 与震荡' },
      { collection: 'agent-engineering', slug: '10', why: '群体:子 Agent 与上下文隔离(subagent 的真正理由)' },
      { collection: 'agent-engineering', slug: '11', why: '群体:多 Agent 模式与成本,诚实的 token 倍数账' },
      { collection: 'agent-engineering', slug: '12', why: '群体:确定性编排,何时把 agent 降级为 workflow' },
      { collection: 'agent-engineering', slug: '13', why: '生产:评测,以及怎么不造一个饱和 rubric' },
      { collection: 'agent-engineering', slug: '14', why: '生产:可观测性与追踪,没有 trace 就调不动 agent' },
      { collection: 'agent-engineering', slug: '15', why: '生产:可靠性与持久化,checkpoint 续跑 + 幂等' },
      { collection: 'agent-engineering', slug: '16', why: '生产:成本与延迟,model cascade 省钱' },
      { collection: 'agent-engineering', slug: '17', why: '收束:把所有器官接成统一 agent + 就业面试题库' },
    ],
  },
  {
    id: 'llm-inference-engineer',
    title: '大模型推理工程师',
    audience: '面向就业 · 进推理引擎 / infra 团队',
    goal: '看懂并优化一个 LLM 推理服务的吞吐、延迟与显存占用。',
    steps: [
      { collection: 'tech-library', slug: '1-01', why: '推理优化的对象是模型本身,先吃透 Transformer 结构' },
      { collection: 'tech-library', slug: '1-03', why: 'MoE / GQA / MLA 决定了现代模型的推理特性,是后续优化的前提' },
      { collection: 'tech-library', slug: '2-01', why: '推理两阶段 + KV 缓存是整个推理性能模型的地基' },
      { collection: 'tech-library', slug: '2-02', why: '显存是推理第一瓶颈,PagedAttention 是绕不开的核心机制' },
      { collection: 'tech-library', slug: '2-03', why: '连续批处理决定吞吐上限' },
      { collection: 'tech-library', slug: '2-04', why: 'FlashAttention 是算子级加速的代表,面试高频' },
      { collection: 'tech-library', slug: '2-05', why: '量化是降显存 / 提速的主力手段' },
      { collection: 'tech-library', slug: '2-06', why: '投机解码是降延迟的前沿手段' },
      { collection: 'tech-library', slug: '2-07', why: '大模型必然多卡,分布式推理 + CUDA 入门' },
      { collection: 'ai-app-engineering', slug: '19', why: '把内核知识落到真实的推理与部署' },
      { collection: 'ai-app-engineering', slug: '20', why: '生产级服务需要 LLM Gateway 与平台工程收口' },
    ],
  },
  {
    id: 'rag-retrieval-engineer',
    title: 'RAG / 检索工程师',
    audience: '面向就业 · 做知识库 / 检索增强应用',
    goal: '从零搭一个生产级 RAG,并能定位召回与回答质量问题。',
    steps: [
      { collection: 'ai-app-engineering', slug: '01', why: '先懂模型能力边界,才明白为什么需要 RAG' },
      { collection: 'tech-library', slug: '1-04', why: '检索的向量从 Embedding 来,先理解它的来源' },
      { collection: 'tech-library', slug: '3-01', why: '倒排索引与 BM25 是关键词检索的基线,也是 Hybrid 的一半' },
      { collection: 'tech-library', slug: '3-02', why: '语义检索原理,RAG 召回的核心' },
      { collection: 'tech-library', slug: '3-03', why: 'HNSW / IVF / PQ 是向量检索的算法心脏' },
      { collection: 'tech-library', slug: '3-04', why: '理解向量库内核,才知道参数怎么调' },
      { collection: 'tech-library', slug: '3-05', why: 'Hybrid Search + Rerank 是把召回质量做到生产级的关键' },
      { collection: 'tech-library', slug: '3-06', why: '数据量大时要分布式检索与分片' },
      { collection: 'ai-app-engineering', slug: '07', why: '应用层落地 Embedding 与向量检索' },
      { collection: 'ai-app-engineering', slug: '08', why: '端到端把 RAG 完整流水线搭起来' },
      { collection: 'ai-app-engineering', slug: '09', why: '高级 RAG 的进阶优化' },
      { collection: 'ai-app-engineering', slug: '10', why: '真实文档处理(文档智能)' },
      { collection: 'ai-app-engineering', slug: '11', why: '没有评测就不知道好不好——RAG 评测收尾' },
    ],
  },
  {
    id: 'ai-application-agent-engineer',
    title: 'AI 应用 / Agent 工程师',
    audience: '面向就业 · 做 AI 原生应用 / Agent 产品',
    goal: '独立设计并实现一个带工具、记忆、多步推理的 Agent 应用并上线。',
    steps: [
      { collection: 'ai-app-engineering', slug: '00', why: '先看全景:学习路径与 P0–P5 项目链' },
      { collection: 'ai-app-engineering', slug: '01', why: 'LLM 物理学——一切应用决策的底层约束' },
      { collection: 'ai-app-engineering', slug: '02', why: 'API 工程:流式、并发、成本、错误处理' },
      { collection: 'ai-app-engineering', slug: '03', why: 'Prompt 工程化,从玄学到可维护' },
      { collection: 'ai-app-engineering', slug: '05', why: 'Function Calling 是 Agent 行动能力的起点' },
      { collection: 'ai-app-engineering', slug: '06', why: 'MCP 协议——当下工具生态的统一接口' },
      { collection: 'ai-app-engineering', slug: '12', why: 'Agent 本质与单 Agent 架构' },
      { collection: 'ai-app-engineering', slug: '13', why: '上下文工程与记忆,决定 Agent 能走多远' },
      { collection: 'ai-app-engineering', slug: '14', why: 'Workflow 编排:可控的多步流程' },
      { collection: 'ai-app-engineering', slug: '15', why: 'Multi-Agent 系统,复杂任务的拆解' },
      { collection: 'ai-app-engineering', slug: '17', why: '评测体系——上线前的质量底线' },
      { collection: 'ai-app-engineering', slug: '18', why: '可观测性,生产排障的眼睛' },
      { collection: 'ai-app-engineering', slug: '21', why: '安全:注入、越权、数据泄漏的防线' },
      { collection: 'ai-app-engineering', slug: '23', why: '把零件组装成 AI 产品系统设计' },
    ],
  },
  {
    id: 'llm-algorithm-posttrain-engineer',
    title: '大模型算法 / 后训练工程师',
    audience: '面向就业 / 科研 · 做训练、对齐与后训练',
    goal: '理解大模型训练全链路,并跟上现代对齐与推理模型训练范式。',
    steps: [
      { collection: 'tech-library', slug: '1-01', why: 'Transformer 解剖——训练与推理共同的起点' },
      { collection: 'tech-library', slug: '1-02', why: '位置编码 RoPE,长上下文的关键' },
      { collection: 'tech-library', slug: '1-03', why: '现代架构演进 MoE / GQA / MLA' },
      { collection: 'tech-library', slug: '1-04', why: 'Tokenizer 与 Embedding,数据进入模型的第一关' },
      { collection: 'tech-library', slug: '1-05', why: '预训练目标与 Scaling Law——训练决策的依据' },
      { collection: 'tech-library', slug: '1-06', why: '分布式训练 DP / TP / PP / ZeRO,大模型训练的工程地基' },
      { collection: 'tech-library', slug: '1-07', why: '对齐概览:SFT、RLHF、DPO 的全貌' },
      { collection: 'tech-library', slug: '6-06', why: 'RLHF / DPO / GRPO 对齐的算法细节' },
      { collection: 'tech-library', slug: '6-07', why: 'RL for Reasoning:可验证奖励,推理模型时代的核心' },
      { collection: 'tech-library', slug: '1-08', why: '从零搭一个 GPT,把全链路串成手感' },
      { collection: 'ai-app-engineering', slug: '22', why: '微调与模型定制,把训练能力接到应用' },
    ],
  },
];

export interface ResolvedStep extends PathStepRef {
  chapter: LibraryChapter;
  collectionTitle: string;
  minutes: number;
}

export interface ResolvedPath extends Omit<LearningPathMeta, 'steps'> {
  steps: ResolvedStep[];
  totalMinutes: number;
  chapterCount: number;
}

// Local copy of the collection page's reading-time heuristic. Kept inline rather
// than shared to avoid touching the existing collection/chapter pages; if a third
// consumer appears, promote this into library.ts.
function readingMinutes(content: string): number {
  return Math.max(1, Math.round(content.replace(/\s+/g, '').length / 300));
}

let cachedPaths: ResolvedPath[] | null = null;

// Resolve every step against the real chapter inventory. Throws at build time if
// a path points at a chapter that does not exist (same fail-fast contract as the
// collection slug validation) — this guards against typos and renamed slugs.
export function getLearningPaths(): ResolvedPath[] {
  if (cachedPaths !== null) return cachedPaths;

  const chapters = getLibraryChapters();
  const byKey = new Map(chapters.map((ch) => [`${ch.collection}/${ch.slug}`, ch]));

  cachedPaths = LEARNING_PATHS.map((path) => {
    const steps = path.steps.map((step) => {
      const chapter = byKey.get(`${step.collection}/${step.slug}`);
      if (!chapter) {
        throw new Error(
          `library-paths: path "${path.id}" references missing chapter ${step.collection}/${step.slug}`
        );
      }
      if (!LIBRARY_COLLECTIONS[step.collection]) {
        throw new Error(`library-paths: path "${path.id}" references unknown collection "${step.collection}"`);
      }
      return {
        ...step,
        chapter,
        collectionTitle: LIBRARY_COLLECTIONS[step.collection].title,
        minutes: readingMinutes(chapter.content),
      };
    });
    return {
      ...path,
      steps,
      totalMinutes: steps.reduce((sum, s) => sum + s.minutes, 0),
      chapterCount: steps.length,
    };
  });
  return cachedPaths;
}
