/**
 * Render-time reader for the Giscus comments cache populated by
 * `src/integrations/fetch-giscus-comments.ts`.
 *
 * Safe to call with a missing or empty cache — returns [].
 */
import { readFileSync } from 'fs';
import { join } from 'path';

export interface RealComment {
  body: string;
  author: string;
  url: string;
}

type Cache = Record<string, RealComment[]>;

let _cache: Cache | null = null;

function loadCache(): Cache {
  if (_cache) return _cache;
  try {
    const path = join(process.cwd(), 'public/giscus-comments.json');
    const raw = readFileSync(path, 'utf-8');
    _cache = JSON.parse(raw) as Cache;
  } catch {
    _cache = {};
  }
  return _cache;
}

/** Pick up to `max` comments for a given article slug.
 *  Trims long bodies to a preview length. */
export function getRealComments(slug: string, max: number = 3, previewLen: number = 120): RealComment[] {
  const cache = loadCache();
  const key = `/notes/${slug}`;
  const raw = cache[key] || [];
  return raw.slice(0, max).map(c => ({
    ...c,
    body: c.body.length > previewLen ? c.body.slice(0, previewLen).trim() + '…' : c.body,
  }));
}
