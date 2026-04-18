/* storage.ts — Single source of truth for all client-side localStorage keys.
   Inline IIFEs in .astro files reference STORAGE_KEYS directly via duplication
   (string literals must match), but this file is the canonical registry.
   When adding a new key:
     1. Add it here with a clear comment.
     2. Update consumers and tests.
     3. Bump storage version + add migration if shape changed. */

export const STORAGE_KEYS = {
  // Theme: 'light' | 'sepia' | 'dark'
  theme: 'theme',
  // Article reading mode: 'default' | 'reading' | 'book'
  mode: 'mode',
  // Per-slug scroll position for resume-reading toast: number (scrollY)
  // Real key: `read-pos:${pathname}` — use readPosKey(pathname).
  readPosPrefix: 'read-pos:',
  // Per-slug book-mode current spread index: number
  // Real key: `book-pos:${pathname}` — use bookPosKey(pathname).
  bookPosPrefix: 'book-pos:',
  // Recently viewed articles: Array<{slug, title, ts}> max 5
  recentlyViewed: 'recentlyViewed',
  // Recent search queries: string[] max 5
  recentQueries: 'search.recentQueries',
  // Typography prefs (only effective when typography-controls=on)
  typeFontSize: 'type.font-size',
  typeFontFamily: 'type.font-family',
  typeWidth: 'type.width',
} as const;

export type StorageKey = typeof STORAGE_KEYS[keyof typeof STORAGE_KEYS];

export const readPosKey = (pathname: string) => `${STORAGE_KEYS.readPosPrefix}${pathname}`;
export const bookPosKey = (pathname: string) => `${STORAGE_KEYS.bookPosPrefix}${pathname}`;

/**
 * SSR-safe get with JSON parse + fallback.
 */
export function getJSON<T>(key: string, fallback: T): T {
  if (typeof localStorage === 'undefined') return fallback;
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

/**
 * SSR-safe set with JSON stringify; silently swallows quota / disabled errors.
 */
export function setJSON(key: string, value: unknown): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* localStorage disabled or quota exceeded — ignore */
  }
}

/**
 * SSR-safe plain string get.
 */
export function getString(key: string, fallback: string = ''): string {
  if (typeof localStorage === 'undefined') return fallback;
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

/**
 * SSR-safe plain string set.
 */
export function setString(key: string, value: string): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}
