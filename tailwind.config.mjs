/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        'text-primary': '#1a1a1a',
        'text-secondary': '#6b7280',
        'text-tertiary': '#9ca3af',
        'text-muted': '#9ca3af',
        'border': '#e5e7eb',
        'border-light': '#f3f4f6',
        'bg-subtle': '#fafafa',
        'bg-page': '#ffffff',
        'link': '#0066cc',
        'link-hover': '#0052a3',
        'quality-high': '#059669',
        'quality-medium': '#d97706',
        'quality-low': '#dc2626'
      },
      fontSize: {
        'h1': ['2.5rem', { lineHeight: '1.2', fontWeight: '700', letterSpacing: '-0.02em' }],
        'h2': ['1.875rem', { lineHeight: '1.3', fontWeight: '600', letterSpacing: '-0.01em' }],
        'h3': ['1.5rem', { lineHeight: '1.4', fontWeight: '600' }],
        'body': ['1.125rem', { lineHeight: '1.8' }],
        'small': ['0.9375rem', { lineHeight: '1.6' }]
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem'
      },
      maxWidth: {
        'content': '720px',
        'reading': '65ch'
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        'prose': '1.8em'
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        'card-hover': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
      },
      transitionDuration: {
        '200': '200ms',
        '300': '300ms'
      }
    }
  },
  plugins: [
    require('@tailwindcss/typography')
  ]
};
