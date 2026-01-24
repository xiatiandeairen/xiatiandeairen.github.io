import type { AstroIntegration } from 'astro';
import { getAllNotes } from '../utils/notes';
import { writeSearchIndex } from '../utils/search';

export default function generateSearchIndex(): AstroIntegration {
  return {
    name: 'generate-search-index',
    hooks: {
      'build:done': async () => {
        try {
          const notes = getAllNotes();
          writeSearchIndex(notes);
          console.log(`✓ Search index generated with ${notes.length} notes`);
        } catch (error) {
          console.error('Error generating search index:', error);
        }
      }
    }
  };
}
