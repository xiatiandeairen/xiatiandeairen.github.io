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
    description: '从零用 TypeScript 拆解一个生产级 Agent 的七个器官——控制循环、工具与 MCP、执行沙箱、权限授权、上下文工程、记忆系统——每章配可跑代码与失败模式分析。',
    order: 0,
  },
  'vector-search': {
    title: '向量检索引擎',
    description: '从零用 TypeScript 手写一个向量检索引擎——倒排/BM25、暴力 KNN、HNSW、IVF-PQ 量化、Hybrid + Rerank、分片与评测——每章用合成数据实测 recall / QPS / 内存。',
    order: 1,
  },
  'llm-inference': {
    title: 'LLM 推理引擎',
    description: '从零用 TypeScript 手写一个会算真实 tok/s 的推理栈——前向、KV 缓存、采样、连续批处理、分页 KV、投机解码、量化、基准测试——每章配可跑代码与实测数字，讲透为什么/何时坏/代价。',
    order: 2,
  },
  'tiny-dl': {
    title: '训练框架从零',
    description: '从零用 TypeScript 手写一个深度学习训练框架——标量/张量 autograd、层与激活、SGD/Adam、训练循环诊断、自注意力、Transformer block，最后训出一个会收敛的 nanoGPT——每章离线实测 loss 下降与梯度。',
    order: 3,
  },
  'rl-posttrain': {
    title: 'RL 与后训练',
    description: '从零用 TypeScript 手写强化学习与大模型后训练——多臂老虎机、Gridworld、策略梯度、PPO、奖励模型、RLHF、DPO、GRPO、RL for reasoning——每章在玩具环境离线实测奖励/胜率曲线与 reward hacking。',
    order: 4,
  },
  'math-for-builders': {
    title: '开发者的数学与物理',
    description: '面向开发者、按计算机实用性挑选的数学与物理——概率统计、线代、最优化、离散逻辑、数值、信息论、密码学、计算理论、傅里叶、统计力学↔ML、从晶体管到逻辑门——每章一个可跑 demo，把知识「做出来」当精通判据。',
    order: 5,
  },
  'database': {
    title: '数据库引擎',
    description: '从零用 TypeScript 手写一个数据库引擎——slotted page 存储、缓冲池、B+树、LSM 树、WAL、事务与 MVCC、崩溃恢复、查询执行——用内存模拟磁盘，离线实测吞吐、写放大、隔离级别下的并发异常与崩溃恢复。',
    order: 6,
  },
  'compiler': {
    title: '编译器',
    description: '从零用 TypeScript 手写一个编译器/解释器——词法、语法(递归下降+Pratt)、语义、类型检查、中间表示、基本优化、字节码虚拟机、代码生成——实现一门小语言并真能跑程序，实测解析速度与优化前后指令数。',
    order: 7,
  },
  'distributed': {
    title: '分布式系统',
    description: '从零用 TypeScript 手写分布式系统核心——逻辑/向量时钟、RPC 与故障检测、Raft 选主与日志复制、CAP 与一致性模型、最终一致与 CRDT——用确定性内存网络仿真(可注入延迟/丢包/分区)，离线实测选主、日志一致与分区下的取舍。',
    order: 8,
  },
  'diffusion': {
    title: '扩散与生成模型',
    description: '从零用 TypeScript 手写扩散生成模型——前向加噪、反向去噪、score matching、DDPM 训练、DDPM/DDIM 采样、classifier-free guidance、latent diffusion——在 2D toy 数据上真训练，从噪声采样还原目标分布。',
    order: 9,
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
