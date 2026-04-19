import type { Note } from '../src/utils/notes';

let slugCounter = 0;

export function makeNote(overrides: Partial<Note> = {}): Note {
  slugCounter += 1;
  const slug = overrides.slug ?? `note-${slugCounter}`;
  return {
    title: `Title ${slug}`,
    slug,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    date: '2026-01-01T00:00:00Z',
    question: { type: 'general' },
    quality: { overall: 7, coverage: 7, depth: 7, specificity: 7, reviewer: 'ai' },
    analysis: {
      objectivity: { factRatio: 0.5, inferenceRatio: 0.3, opinionRatio: 0.2 },
      assumptions: [],
      limitations: [],
    },
    review: { status: 'reviewed' },
    tags: [],
    topics: [],
    content: 'body',
    file: `${slug}.md`,
    ...overrides,
  } as Note;
}
