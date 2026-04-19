import { describe, it, expect } from 'vitest';
import { validateSeriesOrder } from '../src/utils/schema';

describe('validateSeriesOrder', () => {
  it('passes contiguous orders starting from 1', () => {
    expect(() => validateSeriesOrder([
      { slug: 'a', file: 'a.md', series: { name: 'S', order: 1 } },
      { slug: 'b', file: 'b.md', series: { name: 'S', order: 2 } },
      { slug: 'c', file: 'c.md', series: { name: 'S', order: 3 } },
    ])).not.toThrow();
  });

  it('ignores notes outside any series', () => {
    expect(() => validateSeriesOrder([
      { slug: 'a', file: 'a.md' },
      { slug: 'b', file: 'b.md', series: { name: 'S', order: 1 } },
    ])).not.toThrow();
  });

  it('ignores notes in a series without order', () => {
    expect(() => validateSeriesOrder([
      { slug: 'a', file: 'a.md', series: { name: 'S', order: 1 } },
      { slug: 'b', file: 'b.md', series: { name: 'S' } },
    ])).not.toThrow();
  });

  it('throws on duplicate order within a series', () => {
    expect(() => validateSeriesOrder([
      { slug: 'a', file: 'a.md', series: { name: 'S', order: 1 } },
      { slug: 'b', file: 'b.md', series: { name: 'S', order: 1 } },
    ])).toThrow(/duplicate order/);
  });

  it('throws on gap in orders', () => {
    expect(() => validateSeriesOrder([
      { slug: 'a', file: 'a.md', series: { name: 'S', order: 1 } },
      { slug: 'b', file: 'b.md', series: { name: 'S', order: 3 } },
    ])).toThrow(/expected orders/);
  });

  it('throws when series does not start at 1', () => {
    expect(() => validateSeriesOrder([
      { slug: 'a', file: 'a.md', series: { name: 'S', order: 2 } },
      { slug: 'b', file: 'b.md', series: { name: 'S', order: 3 } },
    ])).toThrow(/expected orders/);
  });

  it('validates series independently', () => {
    expect(() => validateSeriesOrder([
      { slug: 'a', file: 'a.md', series: { name: 'S1', order: 1 } },
      { slug: 'b', file: 'b.md', series: { name: 'S2', order: 1 } },
      { slug: 'c', file: 'c.md', series: { name: 'S2', order: 2 } },
    ])).not.toThrow();
  });
});
