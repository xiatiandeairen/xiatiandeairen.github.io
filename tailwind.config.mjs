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
        'mono': ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
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
        'link': 'var(--ink)',
        'link-hover': 'var(--accent)',
      },
      fontSize: {
        'masthead': ['var(--font-masthead)', { lineHeight: '1.1', fontWeight: '900', letterSpacing: '-0.025em' }],
        'h1': ['var(--font-h1)', { lineHeight: '1.25', fontWeight: '800', letterSpacing: '-0.02em' }],
        'h2': ['var(--font-h2)', { lineHeight: '1.3', fontWeight: '700', letterSpacing: '-0.01em' }],
        'h3': ['var(--font-h3)', { lineHeight: '1.35', fontWeight: '700' }],
        'body': ['var(--font-body)', { lineHeight: '1.8' }],
        'small': ['var(--font-small)', { lineHeight: '1.6' }],
        'caption': ['var(--font-caption)', { lineHeight: '1.5' }],
        'label': ['var(--font-label)', { lineHeight: '1.2', fontWeight: '600', letterSpacing: '0.06em' }],
      },
      maxWidth: {
        'content': '900px',
        'reading': '65ch'
      },
    }
  },
  plugins: [
    require('@tailwindcss/typography')
  ]
};
