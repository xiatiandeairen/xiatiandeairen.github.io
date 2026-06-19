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
  'ai-app-engineering': {
    title: 'AI 应用工程',
    description: '从 LLM 的物理特性出发，走完 API 工程、Prompt、RAG、Agent、评测、部署到产品系统设计的完整应用工程链路。',
    order: 1,
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
