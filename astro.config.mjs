import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import generateSearchIndex from './src/integrations/generate-search-index.ts';
import fetchGiscusComments from './src/integrations/fetch-giscus-comments.ts';

export default defineConfig({
  site: 'https://xiatiandeairen.github.io',
  base: '/',
  integrations: [tailwind(), sitemap(), generateSearchIndex(), fetchGiscusComments()],
  markdown: {
    shikiConfig: {
      theme: 'github-light',
      wrap: true
    }
  },
  output: 'static'
});
