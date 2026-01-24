import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import generateSearchIndex from './src/integrations/generate-search-index.ts';

export default defineConfig({
  site: 'https://xiatiandeairen.github.io',
  base: '/',
  integrations: [tailwind(), generateSearchIndex()],
  markdown: {
    shikiConfig: {
      theme: 'github-light',
      wrap: true
    }
  },
  output: 'static'
});
