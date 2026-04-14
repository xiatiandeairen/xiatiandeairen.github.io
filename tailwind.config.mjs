/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      fontFamily: {
        'display': ['"Playfair Display"', 'Georgia', 'serif'],
        'serif': ['"Noto Serif SC"', 'Georgia', 'serif'],
        'sans': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        'mono': ['"JetBrains Mono"', 'Menlo', 'monospace'],
      },
      colors: {
        'paper': '#f4f0e8',
        'ink': '#1a1a1a',
        'ink-light': '#444444',
        'ink-muted': '#888888',
        'ink-faint': '#aaaaaa',
        'ink-ghost': '#bbbbbb',
        'rule': '#c8c0b4',
        'rule-light': '#ddd5ca',
        'rule-faint': '#ece7de',
        'rule-heavy': '#1a1a1a',
        'accent': '#8b0000',
        // Keep old tokens as aliases for compatibility
        'text-primary': '#1a1a1a',
        'text-secondary': '#444444',
        'text-tertiary': '#888888',
        'text-muted': '#aaaaaa',
        'border': '#c8c0b4',
        'border-light': '#ddd5ca',
        'bg-subtle': '#efebe3',
        'bg-page': '#f4f0e8',
        'link': '#1a1a1a',
        'link-hover': '#8b0000',
        'quality-high': '#059669',
        'quality-medium': '#d97706',
        'quality-low': '#dc2626'
      },
      fontSize: {
        'masthead': ['2.625rem', { lineHeight: '1', fontWeight: '900', letterSpacing: '-0.02em' }],
        'h1': ['1.875rem', { lineHeight: '1.2', fontWeight: '900', letterSpacing: '-0.02em' }],
        'h2': ['1.5rem', { lineHeight: '1.3', fontWeight: '700' }],
        'h3': ['1.125rem', { lineHeight: '1.35', fontWeight: '700' }],
        'body': ['1rem', { lineHeight: '1.85' }],
        'small': ['0.8125rem', { lineHeight: '1.6' }],
        'caption': ['0.6875rem', { lineHeight: '1.5' }],
        'label': ['0.5625rem', { lineHeight: '1', fontWeight: '500', letterSpacing: '0.12em' }],
      },
      maxWidth: {
        'content': '900px',
        'reading': '65ch'
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        'prose': '1.8em'
      },
      boxShadow: {
        'card': 'none',
        'card-hover': 'none'
      },
      transitionDuration: {
        '200': '200ms',
        '300': '300ms'
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
