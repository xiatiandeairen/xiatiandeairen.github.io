import { readFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';
import { z } from 'zod';

// Library = imported study/course material. Separate from `notes` (the blog):
// leaner frontmatter (no quality/objectivity/analysis), and grouped into ordered
// collections rather than the blog's flat date stream. See src/content/library/.

const iso8601Regex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$/;

const LibraryFrontmatterSchema = z.object({
  title: z.string(),
  slug: z.string(),
  collection: z.string(),
  group: z.string().optional(),
  order: z.number(),
  summary: z.string(),
  topics: z.array(z.string()),
  tags: z.array(z.string()),
  createdAt: z.string().regex(iso8601Regex),
  updatedAt: z.string().regex(iso8601Regex),
});

export type LibraryFrontmatter = z.infer<typeof LibraryFrontmatterSchema>;

export interface LibraryChapter extends LibraryFrontmatter {
  content: string;
  file: string;
}

export interface LibraryCollectionMeta {
  title: string;
  description: string;
  order: number;
}

// Collection registry. Each subdirectory under src/content/library/ must have an
// entry here; the importer (scripts/import-library.mjs) writes chapters whose
// `collection` field matches one of these keys. Keep titles in sync with the
// importer's COLLECTION_TOPIC map.
export const LIBRARY_COLLECTIONS: Record<string, LibraryCollectionMeta> = {
  // "从零做透" 系列（独立可跑代码 + 实测数字），排在最前。orders 0–9 预留。
  'agent-engineering': {
    title: 'Agent 工程',
    description: '从零用 TypeScript 拆一个生产级 Agent 的 17 个器官——循环/工具MCP/沙箱/权限/上下文/记忆/检索/规划/反思/多Agent/Workflow/评测/追踪/可靠性/成本——每章配离线可跑代码、实测数字与失败模式。',
    order: 0,
  },
  'vector-search': {
    title: '向量检索引擎',
    description: '从零手写一个向量检索引擎：BM25、暴力KNN、HNSW、IVF-PQ、Hybrid、评测七章，每章配可独立跑的 stage，把 recall/QPS/内存三角和每种索引怎么坏钉在真实代码上。',
    order: 1,
  },
  'llm-inference': {
    title: 'LLM 推理引擎',
    description: '从零用 TypeScript 手写会算真实 tok/s 的推理栈：前向、KV 缓存、采样、连续批、分页 KV、投机解码、量化到全栈基准，每章配可跑 stage 与实测数字，逐项证明"快了没变质"（drift=0/困惑度），讲透为什么/何时坏/代价。',
    order: 2,
  },
  'tiny-dl': {
    title: '训练框架从零',
    description: '从零用 TypeScript 手写训练框架：标量/张量 autograd→层→Adam→注意力→Transformer→会收敛的 nanoGPT。每章数字都由确定性离线 stage 实测可复现，且讲透每个零件怎么坏、为什么这么设计。',
    order: 3,
  },
  'rl-posttrain': {
    title: 'RL 与后训练',
    description: '从 bandit 到 PPO/RLHF/DPO/GRPO/RLVR，离线确定性 toy 逐位可复现，每章对照已知真值量出 reward hacking 拐点与一个失败模式。',
    order: 4,
  },
  'math-for-builders': {
    title: '开发者的数学与物理',
    description: '面向开发者、按计算机实用性筛过的数学与物理:12 章每章一个可跑 TypeScript demo + 一个亲手触发的失败模式,数字全来自真实运行(标清算出/实测/合成)——把「做出来 + 坏得明白」当精通判据。',
    order: 5,
  },
  'database': {
    title: '数据库引擎',
    description: '从零用 TypeScript 手写存储引擎:slotted page、缓冲池、B+树、LSM、WAL、MVCC、ARIES 恢复、火山模型查询。内存模拟磁盘,每章实测真实 IO/写放大/隔离异常/恢复正确性,断言钉死,不跑代码=没学。',
    order: 6,
  },
  'compiler': {
    title: '编译器',
    description: '从零用 TypeScript 实现一门小语言 Lox-mini 的完整编译器：词法、Pratt 语法、名称解析、类型检查、IR、优化、字节码 VM、寄存器分配——真能跑斐波那契和闭包，每章配实测指令数与一个失败模式。',
    order: 7,
  },
  'distributed': {
    title: '分布式系统',
    description: '从零手写分布式核心机制(Lamport/向量时钟、Raft 选主与日志复制、CAP、CRDT),全部跑在确定性内存仿真上:每个数字代码真算、脑裂可用 seed 字节级重放。',
    order: 8,
  },
  'diffusion': {
    title: '扩散与生成模型',
    description: '在 2D toy 上手写完整 DDPM(不碰 U-Net/卷积/注意力):前向加噪→DDIM 采样→CFG→latent,每个数字都是可复现的 stage 真实输出,每章配一个代码触发的失败模式。',
    order: 9,
  },
  // order 9.5 保持 PEFT 归入"从零做透"系列（排在 diffusion 之后、survey 系列之前），无需重排既有 0–13。
  'peft': {
    title: '高效微调 PEFT 从零',
    description: '从零用 TypeScript 手写 LoRA/Adapter/Prefix/QLoRA 并真跑梯度，每章实测参数占比、ΔW 低秩谱、合并等价误差，还讲透每种方法怎么坏、为什么这么设计。',
    order: 9.5,
  },
  'interpretability': {
    title: '机制可解释性',
    description: '在一个几千参数、CPU 秒级可复现的玩具 transformer 上,把注意力图/探针/logit lens/patching/induction head/SAE 串成一条"相关→因果"可证伪流水线,每个数字都来自真实 stage 输出、每章必演一个失败模式。',
    order: 9.6,
  },
  'embeddings': {
    title: '表示学习与 Embedding 从零',
    description: '从零手写表示学习:one-hot 维度灾难→共现/PMI→skip-gram+负采样→对比学习/温度→评估与降维,每章一个可跑 stage 真训出语义,是向量检索的上游。',
    order: 9.7,
  },
  'vision': {
    title: '计算机视觉',
    description: '从零用 TypeScript 手写卷积网络（卷积/池化/反向/感受野/BN/残差/真训练/数据增强），每章靠 gradCheck 与失败模式钉死「诚实数字」，是 tiny-dl 的视觉续作。',
    order: 9.8,
  },
  'sequence-models': {
    title: '序列模型 RNN→Mamba',
    description: '纯 TS 从零手写 RNN→LSTM/GRU→注意力→SSM→Mamba，把梯度消失到 1e-17、注意力 64x 扫描代价、A>1 状态溢出成 Infinity、选择门对齐 key 步等失败模式实测成可复现数字。',
    order: 9.85,
  },
  'moe': {
    title: '混合专家 MoE',
    description: '从零用 TypeScript 手写混合专家 MoE:每章一个可跑 stage,实测 top-k 省算力、负载均衡、容量丢弃、专家坍塌——每个机制都配它怎么坏、为什么这么设计,并诚实划清 toy 数字的可迁移边界。',
    order: 9.9,
  },
  // survey/课程系列，orders 10+。
  'ai-app-engineering': {
    title: 'AI 应用工程',
    description: '从 LLM 的物理特性出发，走完 API 工程、Prompt、RAG、Agent、评测、部署到产品系统设计的完整应用工程链路。',
    order: 10,
  },
  'tech-library': {
    title: '技术深挖',
    description: '推理引擎、数据库、编译器、操作系统内核到大模型与检索底座的系统级源码深读，每个领域自成一门小课。',
    order: 11,
  },
  'ai-research-compass': {
    title: 'AI 研究指南',
    description: '从方向横向对比到 MLSys、强化学习、大模型算法、计算机视觉、自然语言处理各专家课程的系统学习路径。',
    order: 12,
  },
  'indie-ai-fullstack': {
    title: '独立开发全栈',
    description: '从心法、出题力、产品设计、UI/UX 到全栈开发、分发增长、一人公司经营的完整独立开发课程。',
    order: 13,
  },
};

const libraryDir = join(process.cwd(), 'src/content/library');

let cachedChapters: LibraryChapter[] | null = null;

export function getLibraryChapters(): LibraryChapter[] {
  if (cachedChapters !== null) return cachedChapters;

  const chapters: LibraryChapter[] = [];
  let collectionDirs: string[];
  try {
    collectionDirs = readdirSync(libraryDir).filter((name) =>
      statSync(join(libraryDir, name)).isDirectory()
    );
  } catch {
    return [];
  }

  for (const collection of collectionDirs) {
    const dir = join(libraryDir, collection);
    for (const file of readdirSync(dir)) {
      if (!file.endsWith('.md')) continue;
      const fileContents = readFileSync(join(dir, file), 'utf-8');
      const { data, content } = matter(fileContents);
      const frontmatter = LibraryFrontmatterSchema.parse(data);
      if (frontmatter.collection !== collection) {
        throw new Error(
          `library: ${collection}/${file} declares collection "${frontmatter.collection}" but lives under "${collection}"`
        );
      }
      if (!LIBRARY_COLLECTIONS[collection]) {
        throw new Error(`library: ${collection}/${file} has no registry entry in LIBRARY_COLLECTIONS`);
      }
      chapters.push({ ...frontmatter, content, file: `${collection}/${file}` });
    }
  }

  validateChapterSlugs(chapters);
  cachedChapters = chapters;
  return chapters;
}

// Slugs must be unique within a collection (the detail route is
// /library/<collection>/<slug>), and order must form a contiguous run so the
// chapter list has no gaps.
function validateChapterSlugs(chapters: LibraryChapter[]): void {
  const byCollection = new Map<string, LibraryChapter[]>();
  for (const ch of chapters) {
    const arr = byCollection.get(ch.collection) || [];
    arr.push(ch);
    byCollection.set(ch.collection, arr);
  }
  for (const [collection, arr] of byCollection) {
    const slugs = new Set<string>();
    for (const ch of arr) {
      if (slugs.has(ch.slug)) {
        throw new Error(`library: duplicate slug "${ch.slug}" in collection "${collection}"`);
      }
      slugs.add(ch.slug);
    }
  }
}

export function getCollectionChapters(collection: string): LibraryChapter[] {
  return getLibraryChapters()
    .filter((ch) => ch.collection === collection)
    .sort((a, b) => a.order - b.order);
}

export interface ChapterGroup {
  // null for flat collections (no subdirectories, e.g. ai-app-engineering).
  group: string | null;
  chapters: LibraryChapter[];
}

// Group a collection's chapters by their `group` field, preserving the global
// `order` sequence both across groups and within each group. A flat collection
// returns a single group with a null title (rendered without a group header).
export function getCollectionGroups(collection: string): ChapterGroup[] {
  const chapters = getCollectionChapters(collection);
  const groups: ChapterGroup[] = [];
  const indexByName = new Map<string, number>();
  for (const ch of chapters) {
    const key = ch.group ?? '';
    let idx = indexByName.get(key);
    if (idx === undefined) {
      idx = groups.length;
      indexByName.set(key, idx);
      groups.push({ group: ch.group ?? null, chapters: [] });
    }
    groups[idx].chapters.push(ch);
  }
  return groups;
}

export interface CollectionSummary extends LibraryCollectionMeta {
  slug: string;
  chapterCount: number;
}

export function getCollections(): CollectionSummary[] {
  const chapters = getLibraryChapters();
  return Object.entries(LIBRARY_COLLECTIONS)
    .map(([slug, meta]) => ({
      ...meta,
      slug,
      chapterCount: chapters.filter((ch) => ch.collection === slug).length,
    }))
    .sort((a, b) => a.order - b.order);
}
