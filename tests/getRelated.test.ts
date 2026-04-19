import { describe, it, expect } from 'vitest';
import { getRelated } from '../src/utils/rank';
import { makeNote } from './factories';

describe('getRelated', () => {
  it('ranks by weighted overlap: topic > series > tag > type', () => {
    const current = makeNote({
      slug: 'current',
      tags: [{ name: 'rust' }],
      topics: [{ name: 'AI 工程化' }],
      series: { name: 'S1', order: 1 },
      question: { type: 'howto' },
    });
    const tagMatch = makeNote({ slug: 'tag-match', tags: [{ name: 'rust' }] });            // +3
    const topicMatch = makeNote({ slug: 'topic-match', topics: [{ name: 'AI 工程化' }] });  // +5
    const seriesMatch = makeNote({ slug: 'series-match', series: { name: 'S1', order: 2 } }); // +4
    const typeMatch = makeNote({ slug: 'type-match', question: { type: 'howto' } });          // +2

    const related = getRelated(current, [tagMatch, topicMatch, seriesMatch, typeMatch, current], { max: 4 });
    expect(related.map(n => n.slug)).toEqual(['topic-match', 'series-match', 'tag-match', 'type-match']);
  });

  it('excludes the current note', () => {
    const current = makeNote({ slug: 'self', tags: [{ name: 'a' }] });
    const other = makeNote({ slug: 'other', tags: [{ name: 'a' }] });
    const related = getRelated(current, [current, other]);
    expect(related.map(n => n.slug)).toEqual(['other']);
  });

  it('excludes notes with zero score', () => {
    const current = makeNote({ slug: 'current', tags: [{ name: 'rust' }] });
    const unrelated = makeNote({ slug: 'other', tags: [{ name: 'python' }], question: { type: 'different' } });
    expect(getRelated(current, [unrelated])).toEqual([]);
  });

  it('respects max option', () => {
    const current = makeNote({ slug: 'c', tags: [{ name: 'x' }] });
    const pool = Array.from({ length: 5 }, (_, i) => makeNote({ slug: `n${i}`, tags: [{ name: 'x' }] }));
    expect(getRelated(current, pool, { max: 2 })).toHaveLength(2);
  });

  it('tie-breaks ties by date desc', () => {
    const current = makeNote({ slug: 'current', tags: [{ name: 'x' }] });
    const older = makeNote({ slug: 'older', tags: [{ name: 'x' }], date: '2026-01-01T00:00:00Z' });
    const newer = makeNote({ slug: 'newer', tags: [{ name: 'x' }], date: '2026-06-01T00:00:00Z' });
    const result = getRelated(current, [older, newer]);
    expect(result.map(n => n.slug)).toEqual(['newer', 'older']);
  });
});
