import type { AstroIntegration } from 'astro';
import { readFileSync, readdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';
import { validateNoteFrontmatter, validateSlugUniqueness } from '../utils/schema';
import { generateSearchIndex } from '../utils/search';

export default function generateSearchIndex(): AstroIntegration {
  return {
    name: 'generate-search-index',
    hooks: {
      'build:done': async () => {
        try {
          const notesDir = join(process.cwd(), 'src/content/notes');
          let files: string[];
          try {
            files = readdirSync(notesDir);
          } catch (error) {
            console.warn('Notes directory not found, skipping search index generation');
            return;
          }

          const notes = [];
          for (const file of files) {
            if (!file.endsWith('.md')) continue;
            try {
              const filePath = join(notesDir, file);
              const fileContents = readFileSync(filePath, 'utf-8');
              const { data, content } = matter(fileContents);
              const frontmatter = validateNoteFrontmatter(data);
              notes.push({
                ...frontmatter,
                content
              });
            } catch (error) {
              console.error(`Error processing file ${file}:`, error);
              throw error;
            }
          }

          validateSlugUniqueness(notes);
          const index = generateSearchIndex(notes);
          const outputPath = join(process.cwd(), 'public/search-index.json');
          writeFileSync(outputPath, JSON.stringify(index, null, 2), 'utf-8');
          
          const sizeInMB = (JSON.stringify(index).length / 1024 / 1024).toFixed(2);
          console.log(`✓ Search index generated: public/search-index.json (${notes.length} notes, ${sizeInMB}MB)`);
          
          if (parseFloat(sizeInMB) > 1) {
            console.warn(`Warning: Search index size is ${sizeInMB}MB. Consider splitting or compressing.`);
          }
        } catch (error) {
          console.error('Error generating search index:', error);
        }
      }
    }
  };
}
