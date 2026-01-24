import { writeFileSync } from 'fs';
import { join } from 'path';
import type { Note } from './notes';

export interface SearchIndex {
  version: string;
  notes: Array<{
    slug: string;
    title: string;
    content: string;
    tags: string[];
    topics: string[];
    questionType: string;
    date: string;
  }>;
  tags: string[];
  topics: string[];
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

  const indexNotes = notes.map((note) => {
    note.tags.forEach(tag => {
      tagSet.add(tag.name);
      tag.alias?.forEach(alias => tagSet.add(alias));
    });

    note.topics.forEach(topic => {
      topicSet.add(topic.name);
      topic.alias?.forEach(alias => topicSet.add(alias));
    });

    return {
      slug: note.slug,
      title: note.title,
      content: extractTextFromMarkdown(note.content),
      tags: note.tags.map(t => t.name),
      topics: note.topics.map(t => t.name),
      questionType: note.question.type,
      date: note.date
    };
  });

  const index: SearchIndex = {
    version: '1.0',
    notes: indexNotes,
    tags: Array.from(tagSet).sort(),
    topics: Array.from(topicSet).sort()
  };

  return index;
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
