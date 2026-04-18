import { writeFileSync } from 'fs';
import { join } from 'path';
import type { Note } from './notes';

export interface SearchIndexNote {
  slug: string;
  title: string;
  content: string;
  tags: string[];
  tagAliases: string[];
  topics: string[];
  topicAliases: string[];
  questionType: string;
  series: string;
  date: string;
}

export interface SearchIndex {
  version: string;
  notes: SearchIndexNote[];
  tags: string[];
  topics: string[];
  tagCounts: Record<string, number>;
  topicCounts: Record<string, number>;
}

export interface SearchHit {
  note: SearchIndexNote;
  score: number;
  matchedField: 'title' | 'tag' | 'topic' | 'series' | 'type' | 'content' | 'alias';
  snippet: string;
}

export interface SearchOptions {
  limit?: number;
}

function extractTextFromMarkdown(content: string): string {
  return content
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`[^`]+`/g, '')
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
    .replace(/[#*_~`]/g, '')
    .replace(/\n+/g, ' ')
    .trim();
}

export function generateSearchIndex(notes: Note[]): SearchIndex {
  const tagSet = new Set<string>();
  const topicSet = new Set<string>();
  const tagCounts: Record<string, number> = {};
  const topicCounts: Record<string, number> = {};

  const indexNotes: SearchIndexNote[] = notes.map((note) => {
    const noteTagAliases: string[] = [];
    const noteTopicAliases: string[] = [];

    note.tags.forEach(tag => {
      tagSet.add(tag.name);
      tagCounts[tag.name] = (tagCounts[tag.name] || 0) + 1;
      tag.alias?.forEach(alias => {
        tagSet.add(alias);
        noteTagAliases.push(alias);
      });
    });

    note.topics.forEach(topic => {
      topicSet.add(topic.name);
      topicCounts[topic.name] = (topicCounts[topic.name] || 0) + 1;
      topic.alias?.forEach(alias => {
        topicSet.add(alias);
        noteTopicAliases.push(alias);
      });
    });

    return {
      slug: note.slug,
      title: note.title,
      content: extractTextFromMarkdown(note.content),
      tags: note.tags.map(t => t.name),
      tagAliases: noteTagAliases,
      topics: note.topics.map(t => t.name),
      topicAliases: noteTopicAliases,
      questionType: note.question.type,
      series: (note as any).series?.name || '',
      date: note.date,
    };
  });

  return {
    version: '1.1',
    notes: indexNotes,
    tags: Array.from(tagSet).sort(),
    topics: Array.from(topicSet).sort(),
    tagCounts,
    topicCounts,
  };
}

export function writeSearchIndex(notes: Note[], outputPath: string = 'public/search-index.json'): void {
  const index = generateSearchIndex(notes);
  const fullPath = join(process.cwd(), outputPath);
  writeFileSync(fullPath, JSON.stringify(index, null, 2), 'utf-8');

  const sizeInMB = (JSON.stringify(index).length / 1024 / 1024).toFixed(2);
  if (parseFloat(sizeInMB) > 1) {
    console.warn(`Warning: Search index size is ${sizeInMB}MB. Consider splitting or compressing.`);
  }
}

// ─────────── runtime search ───────────

interface ParsedQuery {
  free: string;           // free-text portion
  tag?: string;
  topic?: string;
  type?: string;
  series?: string;
}

const PREFIX_RE = /\b(tag|topic|type|series):([^\s]+)/gi;

function parseQuery(raw: string): ParsedQuery {
  const parsed: ParsedQuery = { free: '' };
  let free = raw;
  let m: RegExpExecArray | null;
  while ((m = PREFIX_RE.exec(raw)) !== null) {
    const field = m[1].toLowerCase() as 'tag' | 'topic' | 'type' | 'series';
    parsed[field] = m[2].toLowerCase();
    free = free.replace(m[0], '');
  }
  parsed.free = free.trim().toLowerCase();
  return parsed;
}

function snippetOf(content: string, term: string): string {
  if (!content) return '';
  if (!term) return content.slice(0, 160);
  const idx = content.toLowerCase().indexOf(term);
  if (idx < 0) return content.slice(0, 160);
  const start = Math.max(0, idx - 60);
  const end = Math.min(content.length, idx + term.length + 100);
  return (start > 0 ? '…' : '') + content.slice(start, end) + (end < content.length ? '…' : '');
}

export function searchNotes(rawQuery: string, index: SearchIndex, opts: SearchOptions = {}): SearchHit[] {
  const q = parseQuery(rawQuery || '');
  if (!q.free && !q.tag && !q.topic && !q.type && !q.series) return [];

  const hits: SearchHit[] = [];
  for (const n of index.notes) {
    // field-restricted prefilter: if any prefix given and doesn't match, skip
    if (q.tag && !n.tags.some(t => t.toLowerCase().includes(q.tag!))) {
      if (!n.tagAliases.some(a => a.toLowerCase().includes(q.tag!))) continue;
    }
    if (q.topic && !n.topics.some(t => t.toLowerCase().includes(q.topic!))) {
      if (!n.topicAliases.some(a => a.toLowerCase().includes(q.topic!))) continue;
    }
    if (q.type && !n.questionType.toLowerCase().includes(q.type)) continue;
    if (q.series && !n.series.toLowerCase().includes(q.series)) continue;

    let score = 0;
    let matchedField: SearchHit['matchedField'] = 'content';
    const free = q.free;

    // Prefix-only match (no free text): score based on field match presence
    if (!free) {
      score = 50;
      if (q.tag) matchedField = 'tag';
      else if (q.topic) matchedField = 'topic';
      else if (q.series) matchedField = 'series';
      else if (q.type) matchedField = 'type';
    } else {
      const title = n.title.toLowerCase();
      const content = n.content.toLowerCase();
      const tagsLower = n.tags.map(t => t.toLowerCase());
      const topicsLower = n.topics.map(t => t.toLowerCase());
      const aliasesLower = [...n.tagAliases, ...n.topicAliases].map(a => a.toLowerCase());
      const seriesLower = n.series.toLowerCase();

      if (title === free) { score = 100; matchedField = 'title'; }
      else if (title.includes(free)) { score = 60; matchedField = 'title'; }
      else if (tagsLower.includes(free)) { score = 55; matchedField = 'tag'; }
      else if (topicsLower.includes(free)) { score = 55; matchedField = 'topic'; }
      else if (seriesLower === free) { score = 50; matchedField = 'series'; }
      else if (tagsLower.some(t => t.includes(free))) { score = 35; matchedField = 'tag'; }
      else if (topicsLower.some(t => t.includes(free))) { score = 35; matchedField = 'topic'; }
      else if (seriesLower.includes(free)) { score = 30; matchedField = 'series'; }
      else if (aliasesLower.some(a => a.includes(free))) { score = 25; matchedField = 'alias'; }
      else if (content.includes(free)) { score = 10; matchedField = 'content'; }
      else if (q.tag || q.topic || q.type || q.series) {
        // free text doesn't match but prefix-filter passed
        score = 20;
      } else {
        continue;
      }
    }

    hits.push({
      note: n,
      score,
      matchedField,
      snippet: snippetOf(n.content, free || q.tag || q.topic || q.series || q.type || ''),
    });
  }

  hits.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.note.date < b.note.date ? 1 : a.note.date > b.note.date ? -1 : 0;
  });

  return opts.limit ? hits.slice(0, opts.limit) : hits;
}
