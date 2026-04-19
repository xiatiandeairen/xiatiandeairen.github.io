import { describe, it, expect } from 'vitest';
import { generateSearchIndex, searchNotes } from '../src/utils/search';
import { makeNote } from './factories';

function indexOf(notes: Parameters<typeof generateSearchIndex>[0]) {
  return generateSearchIndex(notes);
}

describe('searchNotes', () => {
  it('returns empty for empty query', () => {
    const idx = indexOf([makeNote()]);
    expect(searchNotes('', idx)).toEqual([]);
  });

  it('title exact match ranks above content match', () => {
    const notes = [
      makeNote({ slug: 'content-hit', title: 'unrelated', content: 'contains rust keyword deep inside' }),
      makeNote({ slug: 'title-hit', title: 'rust', content: 'body' }),
    ];
    const idx = indexOf(notes);
    const hits = searchNotes('rust', idx);
    expect(hits[0].note.slug).toBe('title-hit');
    expect(hits[0].score).toBeGreaterThan(hits[1].score);
  });

  it('tag: prefix restricts to tag field', () => {
    const notes = [
      makeNote({ slug: 'a', title: 'rust', tags: [{ name: 'python' }] }),
      makeNote({ slug: 'b', title: 'other', tags: [{ name: 'rust' }] }),
    ];
    const idx = indexOf(notes);
    const hits = searchNotes('tag:rust', idx);
    expect(hits.map(h => h.note.slug)).toEqual(['b']);
  });

  it('topic: prefix restricts to topic field', () => {
    const notes = [
      makeNote({ slug: 'a', topics: [{ name: 'ai' }] }),
      makeNote({ slug: 'b', topics: [{ name: 'other' }] }),
    ];
    const idx = indexOf(notes);
    expect(searchNotes('topic:ai', idx).map(h => h.note.slug)).toEqual(['a']);
  });

  it('type: prefix filters by question.type', () => {
    const notes = [
      makeNote({ slug: 'a', question: { type: 'howto' } }),
      makeNote({ slug: 'b', question: { type: 'design' } }),
    ];
    const idx = indexOf(notes);
    expect(searchNotes('type:howto', idx).map(h => h.note.slug)).toEqual(['a']);
  });

  it('case-insensitive match', () => {
    const idx = indexOf([makeNote({ slug: 'rust-note', title: 'Rust Basics' })]);
    expect(searchNotes('RUST', idx)).toHaveLength(1);
  });

  it('sorts equal-score hits by date desc', () => {
    const notes = [
      makeNote({ slug: 'older', title: 'hit', date: '2026-01-01T00:00:00Z' }),
      makeNote({ slug: 'newer', title: 'hit', date: '2026-06-01T00:00:00Z' }),
    ];
    const idx = indexOf(notes);
    const hits = searchNotes('hit', idx);
    expect(hits[0].note.slug).toBe('newer');
  });
});
