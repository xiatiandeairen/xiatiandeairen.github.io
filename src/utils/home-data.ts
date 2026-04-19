/**
 * Home page data helpers.
 * Pure functions over Note[]; shared by zh (/index.astro) and en (/en/index.astro).
 */
import type { Note } from './notes';
import { getRelated } from './rank';

// ─── Fixed layout parameters ────────────────────────────────────────────
// Changing these may break the first-fold invariant.
// See CLAUDE.md §首页首屏不变量.

export const SIDEBAR_COUNT = 6;
export const HOT_CHIP_COUNT = 3;
export const SERIES_SLOT_COUNT = 2;
export const SUGGESTIONS_PICKS = 4;

// ─── Active series ──────────────────────────────────────────────────────

export interface ActiveSeries {
  name: string;
  notes: Note[];
  latestDate: string;
  count: number;
}

/** Group notes by series.name, return top N most-recently-updated series.
 *  Each series' inner notes are sorted by order asc, then date desc. */
export function getActiveSeries(notes: Note[], max: number = SERIES_SLOT_COUNT): ActiveSeries[] {
  const byName = new Map<string, Note[]>();
  for (const n of notes) {
    if (!n.series?.name) continue;
    const list = byName.get(n.series.name) || [];
    list.push(n);
    byName.set(n.series.name, list);
  }
  return Array.from(byName.entries())
    .map(([name, items]) => ({
      name,
      notes: [...items].sort((a, b) => {
        const ao = a.series?.order ?? 9999;
        const bo = b.series?.order ?? 9999;
        if (ao !== bo) return ao - bo;
        return a.date < b.date ? 1 : -1;
      }),
      latestDate: items.reduce((max, n) => (n.date > max ? n.date : max), ''),
      count: items.length,
    }))
    .sort((a, b) => (a.latestDate < b.latestDate ? 1 : -1))
    .slice(0, max);
}

// ─── Hot tags / topics chips ────────────────────────────────────────────

/** Frequency-ranked [name, count] pairs; takes `kind` to pick tags vs topics. */
export function getHotChips(notes: Note[], kind: 'tags' | 'topics', max: number = HOT_CHIP_COUNT): [string, number][] {
  const counts = new Map<string, number>();
  for (const n of notes) {
    const items = kind === 'tags' ? n.tags : n.topics;
    for (const item of items) counts.set(item.name, (counts.get(item.name) || 0) + 1);
  }
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, max);
}

// ─── Suggestions data (client-hydrated) ─────────────────────────────────

export interface SuggestionsData {
  related: Record<string, string[]>;
  titles: Record<string, string>;
  dates: Record<string, string>;
  types: Record<string, string>;
}

/** Build the static JSON payload the client-side suggestions script needs. */
export function getSuggestionsData(notes: Note[]): SuggestionsData {
  const related: Record<string, string[]> = {};
  const titles: Record<string, string> = {};
  const dates: Record<string, string> = {};
  const types: Record<string, string> = {};
  for (const n of notes) {
    titles[n.slug] = n.title;
    dates[n.slug] = n.date;
    types[n.slug] = n.question.type;
    related[n.slug] = getRelated(n, notes, { max: 6 }).map(r => r.slug);
  }
  return { related, titles, dates, types };
}
