import type { APIRoute } from 'astro';
import { getAllNotes, sortNotes } from '../utils/notes';
import { SITE_CONFIG } from '../utils/constants';

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function getExcerpt(content: string, maxLen: number = 300): string {
  const plain = content
    .replace(/#{1,6}\s/g, '')
    .replace(/\*\*?|__?/g, '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`[^`]+`/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\n+/g, ' ')
    .trim();
  if (plain.length <= maxLen) return plain;
  return plain.slice(0, maxLen).trim() + '…';
}

export const GET: APIRoute = () => {
  const allNotes = getAllNotes();
  const sorted = sortNotes(allNotes, 'date', 'desc');
  const recent = sorted.slice(0, 20);

  const items = recent.map(note => `
    <item>
      <title>${escapeXml(note.title)}</title>
      <link>${SITE_CONFIG.url}/notes/${note.slug}</link>
      <guid isPermaLink="true">${SITE_CONFIG.url}/notes/${note.slug}</guid>
      <pubDate>${new Date(note.date).toUTCString()}</pubDate>
      <description>${escapeXml(getExcerpt(note.content))}</description>
      ${note.tags.map(t => `<category>${escapeXml(t.name)}</category>`).join('\n      ')}
    </item>`).join('');

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${escapeXml(SITE_CONFIG.title)}</title>
    <description>${escapeXml(SITE_CONFIG.description)}</description>
    <link>${SITE_CONFIG.url}</link>
    <atom:link href="${SITE_CONFIG.url}/rss.xml" rel="self" type="application/rss+xml" />
    <language>zh-CN</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    ${items}
  </channel>
</rss>`;

  return new Response(xml, {
    headers: {
      'Content-Type': 'application/xml; charset=utf-8',
    },
  });
};
