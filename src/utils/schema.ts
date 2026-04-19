import { z } from 'zod';

const iso8601Regex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$/;

const QuestionSchema = z.object({
  type: z.string(),
  subType: z.string().optional()
});

const QualitySchema = z.object({
  overall: z.number().min(0).max(10),
  coverage: z.number().min(0).max(10),
  depth: z.number().min(0).max(10),
  specificity: z.number().min(0).max(10),
  reviewer: z.enum(['ai', 'human', 'hybrid'])
});

const ObjectivitySchema = z.object({
  factRatio: z.number().min(0).max(1),
  inferenceRatio: z.number().min(0).max(1),
  opinionRatio: z.number().min(0).max(1)
}).refine(
  (data) => {
    const sum = data.factRatio + data.inferenceRatio + data.opinionRatio;
    return Math.abs(sum - 1.0) < 0.01;
  },
  {
    message: 'factRatio + inferenceRatio + opinionRatio must equal 1.0'
  }
);

const AnalysisSchema = z.object({
  objectivity: ObjectivitySchema,
  assumptions: z.array(z.string()),
  limitations: z.array(z.string())
});

const ReviewSchema = z.object({
  status: z.enum(['draft', 'reviewed', 'deprecated']),
  reviewedAt: z.string().regex(iso8601Regex).optional()
});

const VersionHistorySchema = z.object({
  version: z.number(),
  updatedAt: z.string().regex(iso8601Regex),
  updatedBy: z.enum(['ai', 'human', 'hybrid']),
  changes: z.array(z.string()).optional(),
  changesSummary: z.string().optional()
});

const VersionSchema = z.object({
  current: z.number(),
  history: z.array(VersionHistorySchema).optional()
}).optional();

const TagSchema = z.object({
  name: z.string(),
  parent: z.string().optional(),
  alias: z.array(z.string()).optional()
});

const TopicSchema = z.object({
  name: z.string(),
  alias: z.array(z.string()).optional()
});

export const NoteFrontmatterSchema = z.object({
  title: z.string(),
  slug: z.string(),
  createdAt: z.string().regex(iso8601Regex),
  updatedAt: z.string().regex(iso8601Regex),
  date: z.string().regex(iso8601Regex),
  version: VersionSchema,
  question: QuestionSchema,
  quality: QualitySchema,
  analysis: AnalysisSchema,
  review: ReviewSchema,
  tags: z.array(TagSchema),
  topics: z.array(TopicSchema),
  featured: z.boolean().optional(),
  series: z.object({
    name: z.string(),
    order: z.number().optional()
  }).optional()
}).passthrough();

export type NoteFrontmatter = z.infer<typeof NoteFrontmatterSchema>;

export function validateNoteFrontmatter(data: unknown): NoteFrontmatter {
  return NoteFrontmatterSchema.parse(data);
}

export function validateSlugUniqueness(notes: Array<{ slug: string; file?: string }>): void {
  const slugMap = new Map<string, string[]>();
  
  for (const note of notes) {
    const existing = slugMap.get(note.slug) || [];
    existing.push(note.file || note.slug);
    slugMap.set(note.slug, existing);
  }
  
  const duplicates: Array<{ slug: string; files: string[] }> = [];
  for (const [slug, files] of slugMap.entries()) {
    if (files.length > 1) {
      duplicates.push({ slug, files });
    }
  }
  
  if (duplicates.length > 0) {
    const errorMessages = duplicates.map(
      ({ slug, files }) => `Slug "${slug}" is duplicated in: ${files.join(', ')}`
    );
    throw new Error(`Duplicate slugs found:\n${errorMessages.join('\n')}`);
  }
}

interface SeriesNote {
  slug: string;
  file?: string;
  series?: { name: string; order?: number };
}

// Build-time guard: series.order within each series must form [1..N] with no
// duplicates and no gaps. Notes without order are ignored by this check —
// they still render but won't participate in prev/next ordering.
export function validateSeriesOrder(notes: SeriesNote[]): void {
  const groups = new Map<string, Array<{ order: number; file: string }>>();

  for (const n of notes) {
    if (!n.series?.name || n.series.order === undefined) continue;
    const key = n.series.name;
    const arr = groups.get(key) || [];
    arr.push({ order: n.series.order, file: n.file || n.slug });
    groups.set(key, arr);
  }

  const problems: string[] = [];
  for (const [name, entries] of groups.entries()) {
    const orders = entries.map(e => e.order).sort((a, b) => a - b);
    const seen = new Set<number>();
    const dupes: number[] = [];
    for (const o of orders) {
      if (seen.has(o)) dupes.push(o);
      seen.add(o);
    }
    if (dupes.length > 0) {
      const offenders = entries.filter(e => dupes.includes(e.order)).map(e => `${e.file}(order=${e.order})`);
      problems.push(`Series "${name}": duplicate order ${[...new Set(dupes)].join(', ')} in ${offenders.join(', ')}`);
      continue;
    }
    const expected = Array.from({ length: orders.length }, (_, i) => i + 1);
    if (orders[0] !== 1 || orders.some((o, i) => o !== expected[i])) {
      problems.push(`Series "${name}": expected orders [${expected.join(',')}] but got [${orders.join(',')}]`);
    }
  }

  if (problems.length > 0) {
    throw new Error(`Invalid series.order:\n${problems.join('\n')}`);
  }
}
