/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        'display': ['"Playfair Display"', 'Georgia', 'serif'],
        'serif': ['"Noto Serif SC"', 'Georgia', 'serif'],
        'sans': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        'mono': ['"JetBrains Mono"', 'Menlo', 'monospace'],
      },
      colors: {
        'paper': 'var(--paper)',
        'ink': 'var(--ink)',
        'ink-light': 'var(--ink-light)',
        'ink-muted': 'var(--ink-muted)',
        'ink-faint': 'var(--ink-faint)',
        'ink-ghost': 'var(--ink-ghost)',
        'rule': 'var(--rule)',
        'rule-light': 'var(--rule-light)',
        'rule-faint': 'var(--rule-faint)',
        'bg-subtle': 'var(--bg-subtle)',
        'accent': 'var(--accent)',
        'text-primary': 'var(--ink)',
        'text-secondary': 'var(--ink-light)',
        'text-tertiary': 'var(--ink-muted)',
        'text-muted': 'var(--ink-faint)',
        'border': 'var(--rule)',
        'border-light': 'var(--rule-light)',
        'bg-page': 'var(--paper)',
        'link': 'var(--ink)',
        'link-hover': 'var(--accent)',
        'quality-high': '#059669',
        'quality-medium': '#d97706',
        'quality-low': '#dc2626'
      },
      fontSize: {
        'masthead': ['2.25rem', { lineHeight: '1.1', fontWeight: '900', letterSpacing: '-0.025em' }],
        'h1': ['1.625rem', { lineHeight: '1.25', fontWeight: '800', letterSpacing: '-0.02em' }],
        'h2': ['1.25rem', { lineHeight: '1.3', fontWeight: '700', letterSpacing: '-0.01em' }],
        'h3': ['1rem', { lineHeight: '1.35', fontWeight: '700' }],
        'body': ['0.9375rem', { lineHeight: '1.75' }],
        'small': ['0.8125rem', { lineHeight: '1.5' }],
        'caption': ['0.6875rem', { lineHeight: '1.4' }],
        'label': ['0.625rem', { lineHeight: '1', fontWeight: '600', letterSpacing: '0.08em' }],
      },
      maxWidth: {
        'content': '900px',
        'reading': '65ch'
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      borderWidth: {
        '3': '3px',
        '4': '4px',
      },
    }
  },
  plugins: [
    require('@tailwindcss/typography')
  ]
};
