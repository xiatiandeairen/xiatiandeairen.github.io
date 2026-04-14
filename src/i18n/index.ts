import zh from './zh';
import en from './en';

export type Lang = 'zh' | 'en';
export type Messages = typeof zh;

const messages: Record<Lang, Messages> = { zh, en };

export const defaultLang: Lang = 'zh';

/**
 * Get language from URL path.
 * /en/* → 'en', everything else → 'zh'
 */
export function getLang(pathname: string): Lang {
  return pathname.startsWith('/en') ? 'en' : 'zh';
}

/**
 * Get the translation function for a given language.
 * Usage: const t = useT('zh'); t('nav.home') → '首页'
 */
export function useT(lang: Lang) {
  const msg = messages[lang];

  return function t(key: string, params?: Record<string, string | number>): string {
    const keys = key.split('.');
    let value: any = msg;
    for (const k of keys) {
      value = value?.[k];
      if (value === undefined) return key;
    }
    if (typeof value !== 'string') return key;
    if (!params) return value;

    return value.replace(/\{(\w+)\}/g, (_, name) =>
      params[name] !== undefined ? String(params[name]) : `{${name}}`
    );
  };
}

/**
 * Build path for the other language.
 * switchLangPath('/archive', 'zh') → '/en/archive'
 * switchLangPath('/en/archive', 'en') → '/archive'
 */
export function switchLangPath(pathname: string, currentLang: Lang): string {
  if (currentLang === 'zh') {
    // Switch to English
    const path = pathname === '/' ? '/' : pathname;
    return '/en' + (path === '/' ? '' : path);
  } else {
    // Switch to Chinese — remove /en prefix
    const path = pathname.replace(/^\/en/, '') || '/';
    return path;
  }
}

/**
 * Build localized path.
 * localePath('/archive', 'en') → '/en/archive'
 * localePath('/archive', 'zh') → '/archive'
 */
export function localePath(path: string, lang: Lang): string {
  if (lang === 'zh') return path;
  return '/en' + (path === '/' ? '' : path);
}
