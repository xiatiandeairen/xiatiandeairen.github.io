// Central recommendation utility.
// Pure functions over Note[] — no DOM, no IO. Deterministic except getRandomNote.

import type { Note } from './notes';

export interface RelatedOptions {
  max?: number;
  weights?: { tag?: number; topic?: number; series?: number; type?: number };
}

const DEFAULT_WEIGHTS = { tag: 3, topic: 5, series: 4, type: 2 };

/**
 * Multi-signal related: weighted overlap of tags, topics, series, question.type.
 * Returns notes with score > 0, sorted by score desc, date desc, capped at max.
 */
export function getRelated(current: Note, all: Note[], opts: RelatedOptions = {}): Note[] {
  const max = opts.max ?? 3;
  const w = { ...DEFAULT_WEIGHTS, ...(opts.weights || {}) };
  const curTags = new Set(current.tags.map(t => t.name));
  const curTopics = new Set(current.topics.map(t => t.name));
  const curSeries = current.series?.name;
  const curType = current.question.type;

  const scored = all
    .filter(n => n.slug !== current.slug)
    .map(n => {
      let score = 0;
      n.tags.forEach(t => { if (curTags.has(t.name)) score += w.tag!; });
      n.topics.forEach(t => { if (curTopics.has(t.name)) score += w.topic!; });
      if (curSeries && n.series?.name === curSeries) score += w.series!;
      if (curType && n.question.type === curType) score += w.type!;
      return { note: n, score };
    })
    .filter(r => r.score > 0);

  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.note.date < b.note.date ? 1 : a.note.date > b.note.date ? -1 : 0;
  });
  return scored.slice(0, max).map(r => r.note);
}

/**
 * Featured: prioritize quality.overall, fall back to date.
 * Notes with featured:true float to the top.
 */
export function getFeatured(all: Note[], n: number = 3): Note[] {
  const sorted = [...all].sort((a, b) => {
    if (a.featured && !b.featured) return -1;
    if (!a.featured && b.featured) return 1;
    const qa = a.quality?.overall ?? 0;
    const qb = b.quality?.overall ?? 0;
    if (qb !== qa) return qb - qa;
    return a.date < b.date ? 1 : a.date > b.date ? -1 : 0;
  });
  return sorted.slice(0, n);
}

/**
 * Pick one random note, optionally excluding a slug.
 * Returns null on empty input.
 */
export function getRandomNote(all: Note[], excludeSlug?: string): Note | null {
  const pool = excludeSlug ? all.filter(n => n.slug !== excludeSlug) : all;
  if (!pool.length) return null;
  return pool[Math.floor(Math.random() * pool.length)];
}

/**
 * Tags that frequently co-occur with the given tag, ranked by co-occurrence count.
 */
export function getRelatedTags(currentTagName: string, all: Note[], n: number = 5): Array<{ name: string; count: number }> {
  const counts: Record<string, number> = {};
  for (const note of all) {
    const names = note.tags.map(t => t.name);
    if (!names.includes(currentTagName)) continue;
    for (const name of names) {
      if (name === currentTagName) continue;
      counts[name] = (counts[name] || 0) + 1;
    }
  }
  return Object.entries(counts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, n);
}

/**
 * Topics that the given tag belongs to (i.e. topics on notes that have this tag).
 */
export function getParentTopics(currentTagName: string, all: Note[]): string[] {
  const set = new Set<string>();
  for (const note of all) {
    if (!note.tags.some(t => t.name === currentTagName)) continue;
    note.topics.forEach(t => set.add(t.name));
  }
  return Array.from(set).sort();
}

/**
 * Topics that co-occur with the given topic on the same notes.
 */
export function getCoTopics(currentTopicName: string, all: Note[], n: number = 5): Array<{ name: string; count: number }> {
  const counts: Record<string, number> = {};
  for (const note of all) {
    const names = note.topics.map(t => t.name);
    if (!names.includes(currentTopicName)) continue;
    for (const name of names) {
      if (name === currentTopicName) continue;
      counts[name] = (counts[name] || 0) + 1;
    }
  }
  return Object.entries(counts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, n);
}

/**
 * Top tags within a topic, by note count.
 */
export function getTopTagsInTopic(topicName: string, all: Note[], n: number = 5): Array<{ name: string; count: number }> {
  const counts: Record<string, number> = {};
  for (const note of all) {
    if (!note.topics.some(t => t.name === topicName)) continue;
    note.tags.forEach(t => {
      counts[t.name] = (counts[t.name] || 0) + 1;
    });
  }
  return Object.entries(counts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, n);
}
