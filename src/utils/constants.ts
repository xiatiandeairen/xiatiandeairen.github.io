export const SITE_CONFIG = {
  title: '夏天的爱人',
  description: '一份关于代码、文字与生活的个人刊物',
  tagline: 'AI 工程化 · 技术写作 · 开发者手记',
  url: 'https://xiatiandeairen.github.io'
};

// Search-engine verification tokens — public (they live in the rendered HTML).
// Add new provider tokens here; BaseLayout renders a meta tag for any non-empty value.
export const SEO_VERIFICATION = {
  google: 'ylFOZx_5CCV8npi4p7s-JHzdIk_J1KVNGjHUrq-8Oic',
  bing: '',
};

export const PAGINATION = {
  perPage: 20
};

export const TAG_MAX_DEPTH = 2;

export const QUALITY_COLORS = {
  high: 'var(--color-quality-high)',
  medium: 'var(--color-quality-medium)',
  low: 'var(--color-quality-low)'
} as const;

export const QUALITY_THRESHOLDS = {
  high: 8,
  medium: 6
} as const;
