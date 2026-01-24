import type { AstroIntegration } from 'astro';
import { readFileSync, readdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import matter from 'gray-matter';
import { validateNoteFrontmatter, validateSlugUniqueness } from '../utils/schema';
import { generateSearchIndex } from '../utils/search';

export default function generateSearchIndexIntegration(): AstroIntegration {
  return {
    name: 'generate-search-index',
    hooks: {
      'astro:build:done': async ({ dir }) => {
        try {
          const projectRoot = process.cwd();
          const notesDir = join(projectRoot, 'src/content/notes');
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
          
          const distDir = fileURLToPath(dir);
          const outputPath = join(distDir, 'search-index.json');
          writeFileSync(outputPath, JSON.stringify(index, null, 2), 'utf-8');
          
          const sizeInMB = (JSON.stringify(index).length / 1024 / 1024).toFixed(2);
          console.log(`✓ Search index generated: search-index.json (${notes.length} notes, ${sizeInMB}MB)`);
          
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
