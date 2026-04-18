import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';
import { validateNoteFrontmatter, validateSlugUniqueness } from './schema';
import { PAGINATION } from './constants';

export interface Note extends NoteFrontmatter {
  content: string;
  file?: string;
}

export type SortField = 'date' | 'question.type' | 'quality.overall' | 'title';
export type SortOrder = 'asc' | 'desc';

const notesDir = join(process.cwd(), 'src/content/notes');

// Module-level cache: getAllNotes is called by many pages/components per build;
// the file system + frontmatter parsing is deterministic within a single build,
// so we memoize. cachedNotes is invalidated only across builds (process restart).
let cachedNotes: Note[] | null = null;

export function getAllNotes(): Note[] {
  if (cachedNotes !== null) return cachedNotes;

  let files: string[];
  try {
    files = readdirSync(notesDir);
  } catch (error) {
    return [];
  }

  const notes: Note[] = [];

  for (const file of files) {
    if (!file.endsWith('.md')) continue;

    try {
      const filePath = join(notesDir, file);
      const fileContents = readFileSync(filePath, 'utf-8');
      const { data, content } = matter(fileContents);

      const frontmatter = validateNoteFrontmatter(data);
      notes.push({
        ...frontmatter,
        content,
        file
      });
    } catch (error) {
      console.error(`Error processing file ${file}:`, error);
      throw error;
    }
  }

  validateSlugUniqueness(notes);
  cachedNotes = notes;
  return notes;
}

export function getNotesByTag(tagName: string, notes: Note[]): Note[] {
  return notes.filter(note => {
    return note.tags.some(tag => {
      if (tag.name === tagName) return true;
      if (tag.alias?.includes(tagName)) return true;
      
      let currentTag = tag;
      while (currentTag.parent) {
        if (currentTag.parent === tagName) return true;
        const parentTag = note.tags.find(t => t.name === currentTag.parent);
        if (!parentTag) break;
        currentTag = parentTag;
      }
      return false;
    });
  });
}

export function getNotesByTopic(topicName: string, notes: Note[]): Note[] {
  return notes.filter(note => {
    return note.topics.some(topic => {
      if (topic.name === topicName) return true;
      return topic.alias?.includes(topicName) || false;
    });
  });
}

export function sortNotes(notes: Note[], sortBy: SortField = 'date', order: SortOrder = 'desc'): Note[] {
  const sorted = [...notes].sort((a, b) => {
    let aValue: any;
    let bValue: any;

    switch (sortBy) {
      case 'date':
        aValue = new Date(a.date).getTime();
        bValue = new Date(b.date).getTime();
        break;
      case 'question.type':
        aValue = a.question.type;
        bValue = b.question.type;
        break;
      case 'quality.overall':
        aValue = a.quality.overall;
        bValue = b.quality.overall;
        break;
      case 'title':
        aValue = a.title.toLowerCase();
        bValue = b.title.toLowerCase();
        break;
      default:
        return 0;
    }

    if (aValue < bValue) return order === 'asc' ? -1 : 1;
    if (aValue > bValue) return order === 'asc' ? 1 : -1;
    return 0;
  });

  return sorted;
}

export function paginateNotes(notes: Note[], page: number, perPage: number = PAGINATION.perPage) {
  const totalPages = Math.ceil(notes.length / perPage);
  const startIndex = (page - 1) * perPage;
  const endIndex = startIndex + perPage;
  const paginatedNotes = notes.slice(startIndex, endIndex);

  return {
    notes: paginatedNotes,
    pagination: {
      currentPage: page,
      totalPages,
      perPage,
      total: notes.length,
      hasNext: page < totalPages,
      hasPrev: page > 1
    }
  };
}

export function formatDate(dateString: string, locale: string = 'zh-CN'): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat(locale, {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  }).format(date);
}
