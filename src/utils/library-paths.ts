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
