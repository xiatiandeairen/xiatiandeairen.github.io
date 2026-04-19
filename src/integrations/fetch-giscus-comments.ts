/**
 * Fetch Giscus (GitHub Discussions) comments at build time and write
 * `public/giscus-comments.json` — consumed at render time by ArticleCard to
 * show real reader excerpts in the hero card.
 *
 * Cache shape: { [pathname]: Array<{ body, author, url }> }
 *
 * Silent fallback: missing GITHUB_TOKEN or any fetch error → empty cache.
 */
import type { AstroIntegration } from 'astro';
import { writeFileSync } from 'fs';
import { join } from 'path';

const GH_API = 'https://api.github.com/graphql';
const OWNER = 'xiatiandeairen';
const REPO = 'xiatiandeairen.github.io';
const CATEGORY_ID = 'DIC_kwDONiKBwM4C7GHk'; // from giscus data-category-id
const MAX_COMMENTS_PER_DISCUSSION = 10;
const MAX_DISCUSSIONS = 100;

interface CommentNode {
  bodyText: string;
  url: string;
  author: { login: string } | null;
}

interface DiscussionNode {
  title: string;
  comments: { nodes: CommentNode[] };
}

interface GqlResponse {
  data?: { repository?: { discussions?: { nodes?: DiscussionNode[] } } };
  errors?: unknown;
}

export interface CachedComment {
  body: string;
  author: string;
  url: string;
}

async function fetchDiscussions(token: string): Promise<DiscussionNode[]> {
  const query = `
    query($owner: String!, $repo: String!, $categoryId: ID!, $first: Int!, $commentFirst: Int!) {
      repository(owner: $owner, name: $repo) {
        discussions(first: $first, categoryId: $categoryId, orderBy: { field: UPDATED_AT, direction: DESC }) {
          nodes {
            title
            comments(first: $commentFirst) {
              nodes {
                bodyText
                url
                author { login }
              }
            }
          }
        }
      }
    }`;
  const res = await fetch(GH_API, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      'User-Agent': 'xiatiandeairen-build',
    },
    body: JSON.stringify({
      query,
      variables: {
        owner: OWNER,
        repo: REPO,
        categoryId: CATEGORY_ID,
        first: MAX_DISCUSSIONS,
        commentFirst: MAX_COMMENTS_PER_DISCUSSION,
      },
    }),
  });
  if (!res.ok) throw new Error(`GitHub API ${res.status}`);
  const json = (await res.json()) as GqlResponse;
  if (json.errors) throw new Error(`GraphQL: ${JSON.stringify(json.errors)}`);
  return json.data?.repository?.discussions?.nodes ?? [];
}

/** Normalize a giscus discussion title (= article pathname) to a canonical key.
 *  Strip trailing slash; ensure leading slash. */
function normalizePath(p: string): string {
  if (!p) return '';
  let s = p.trim();
  if (!s.startsWith('/')) s = '/' + s;
  if (s.length > 1 && s.endsWith('/')) s = s.slice(0, -1);
  return s;
}

async function buildCache(token: string): Promise<Record<string, CachedComment[]>> {
  const discussions = await fetchDiscussions(token);
  const cache: Record<string, CachedComment[]> = {};
  for (const d of discussions) {
    const key = normalizePath(d.title);
    if (!key.startsWith('/notes/')) continue; // only article discussions
    const comments = (d.comments.nodes || [])
      .filter(c => c && c.bodyText && c.bodyText.trim().length > 0)
      .map<CachedComment>(c => ({
        body: c.bodyText.trim(),
        author: c.author?.login || 'anonymous',
        url: c.url,
      }));
    if (comments.length > 0) cache[key] = comments;
  }
  return cache;
}

async function writeCache(projectRoot: string): Promise<void> {
  const token = process.env.GITHUB_TOKEN;
  const outputPath = join(projectRoot, 'public/giscus-comments.json');
  if (!token) {
    // Write empty cache so downstream code doesn't need to handle missing file.
    writeFileSync(outputPath, '{}', 'utf-8');
    console.log('[giscus] GITHUB_TOKEN not set — wrote empty comments cache');
    return;
  }
  try {
    const cache = await buildCache(token);
    writeFileSync(outputPath, JSON.stringify(cache, null, 2), 'utf-8');
    const total = Object.values(cache).reduce((n, arr) => n + arr.length, 0);
    console.log(`✓ Giscus cache: ${Object.keys(cache).length} discussions, ${total} comments`);
  } catch (err) {
    writeFileSync(outputPath, '{}', 'utf-8');
    console.warn(`[giscus] fetch failed, wrote empty cache: ${(err as Error).message}`);
  }
}

export default function fetchGiscusCommentsIntegration(): AstroIntegration {
  return {
    name: 'fetch-giscus-comments',
    hooks: {
      'astro:config:setup': async () => {
        await writeCache(process.cwd());
      },
    },
  };
}
