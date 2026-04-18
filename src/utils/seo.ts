import type { Note } from './notes';
import { SITE_CONFIG } from './constants';

export interface SEOData {
  title: string;
  description: string;
  url: string;
  ogTitle?: string;
  ogDescription?: string;
  ogType?: string;
  ogImage?: string;
  canonical?: string;
}

export function generateSEOMeta(note: Note, siteConfig = SITE_CONFIG): SEOData {
  const title = `${note.title} | ${siteConfig.title}`;
  const description = note.content.slice(0, 160).replace(/\n/g, ' ').trim() || siteConfig.description;
  const url = `${siteConfig.url}/notes/${note.slug}`;

  return {
    title,
    description,
    url,
    ogTitle: note.title,
    ogDescription: description,
    ogType: 'article',
    // og-default.svg is the static fallback; per-slug PNG generation is v2.
    ogImage: `${siteConfig.url}/og-default.svg`,
    canonical: url
  };
}

export function generatePageMeta(page: { title: string; description?: string; path?: string }, siteConfig = SITE_CONFIG): SEOData {
  const title = page.path === '/' ? siteConfig.title : `${page.title} | ${siteConfig.title}`;
  const description = page.description || siteConfig.description;
  const url = `${siteConfig.url}${page.path || ''}`;

  return {
    title,
    description,
    url,
    ogTitle: page.title,
    ogDescription: description,
    ogType: 'website',
    ogImage: `${siteConfig.url}/og-default.svg`,
    canonical: url
  };
}
