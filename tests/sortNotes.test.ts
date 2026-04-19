import { describe, it, expect } from 'vitest';
import { sortNotes } from '../src/utils/notes';
import { makeNote } from './factories';

describe('sortNotes', () => {
  it('sorts by date desc', () => {
    const notes = [
      makeNote({ slug: 'a', date: '2026-01-01T00:00:00Z' }),
      makeNote({ slug: 'b', date: '2026-03-01T00:00:00Z' }),
      makeNote({ slug: 'c', date: '2026-02-01T00:00:00Z' }),
    ];
    expect(sortNotes(notes, 'date', 'desc').map(n => n.slug)).toEqual(['b', 'c', 'a']);
  });

  it('sorts by date asc', () => {
    const notes = [
      makeNote({ slug: 'a', date: '2026-01-01T00:00:00Z' }),
      makeNote({ slug: 'b', date: '2026-03-01T00:00:00Z' }),
    ];
    expect(sortNotes(notes, 'date', 'asc').map(n => n.slug)).toEqual(['a', 'b']);
  });

  it('tie-breaks by slug when dates are equal', () => {
    const notes = [
      makeNote({ slug: 'zebra', date: '2026-01-01T00:00:00Z' }),
      makeNote({ slug: 'alpha', date: '2026-01-01T00:00:00Z' }),
      makeNote({ slug: 'mango', date: '2026-01-01T00:00:00Z' }),
    ];
    expect(sortNotes(notes, 'date', 'desc').map(n => n.slug)).toEqual(['alpha', 'mango', 'zebra']);
  });

  it('does not mutate input', () => {
    const notes = [
      makeNote({ slug: 'a', date: '2026-02-01T00:00:00Z' }),
      makeNote({ slug: 'b', date: '2026-01-01T00:00:00Z' }),
    ];
    const before = notes.map(n => n.slug);
    sortNotes(notes, 'date', 'asc');
    expect(notes.map(n => n.slug)).toEqual(before);
  });
});
