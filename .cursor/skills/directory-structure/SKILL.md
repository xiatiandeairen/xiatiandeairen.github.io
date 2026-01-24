---
name: directory-structure
description: Maintains project directory structure following clear layering, scalability, and maintainability principles. Use when creating new files, organizing code, or when the user asks about project structure, file organization, or directory layout.
---

# Directory Structure

## Standard Structure

```
/
├── .cursor/
│   └── skills/              # Cursor skills
├── .github/
│   └── workflows/           # GitHub Actions
├── public/                  # Static assets
│   └── search-index.json    # Generated at build time
├── src/
│   ├── content/
│   │   └── notes/           # Markdown notes
│   ├── components/          # Reusable components
│   ├── integrations/        # Astro integrations
│   ├── layouts/             # Page layouts
│   ├── pages/               # Route pages
│   ├── styles/              # Global styles
│   ├── types/               # TypeScript types
│   └── utils/               # Utility functions
├── astro.config.mjs
├── tailwind.config.mjs
├── tsconfig.json
└── package.json
```

## Directory Purposes

### `/.cursor/skills/`

**Purpose**: Project-level Cursor skills (shared with repository)

**Structure**: Each skill is a directory containing:
- `SKILL.md` - Required main instructions with YAML frontmatter
- `reference.md` - Optional detailed documentation
- `examples.md` - Optional usage examples
- `scripts/` - Optional utility scripts

**Naming**: kebab-case (e.g., `coding-standards/`, `commit-convention/`)

**Note**: Skills stored here are project-specific and shared with all repository users.

### `/src/content/notes/`

**Purpose**: Store all Markdown note files

**Naming**: kebab-case (e.g., `react-hooks-guide.md`)
- Optional numeric prefix for sorting: `01-basic-note.md`
- Filename should match slug (recommended)

**Scalability**:
- Future: subdirectories by category or year
- Large scale: organize by year `notes/2026/`, `notes/2027/`

### `/src/components/`

**Purpose**: Reusable Astro components

**Organization**:
- Flat structure preferred, avoid over-nesting
- Group by function if needed: `components/notes/`, `components/navigation/`
- Component names: PascalCase

### `/src/layouts/`

**Purpose**: Page layout templates

**Principle**: Keep minimal, only essential layouts

### `/src/pages/`

**Purpose**: Astro route pages

**Organization**:
- Use Astro file system routing
- Dynamic routes: `[param]` format
- Keep route structure clear

### `/src/utils/`

**Purpose**: Pure function utilities

**Organization**:
- Group by functionality per file
- Keep functions pure (no side effects)
- Easy to unit test

### `/src/types/`

**Purpose**: TypeScript type definitions

**Principle**: Centralized type management

## File Naming

- Markdown: `kebab-case.md`
- TypeScript/JavaScript: `camelCase.ts`
- Components: `PascalCase.astro`
- Config files: `kebab-case.config.mjs`

## Forbidden Directories

Do not create:
- `themes/` - Hexo theme directory (removed)
- `scaffolds/` - Hexo template directory (removed)
- `source/` - Hexo source directory (removed)
- `node_modules/` - Should be in .gitignore
- `dist/` - Should be in .gitignore
- `.astro/` - Should be in .gitignore

## Design Principles

1. **Clear layering**: Organize by function module, clear responsibilities
2. **Scalability**: Support 3-5 years of content growth (thousands of notes)
3. **Maintainability**: New members can quickly understand structure
4. **Avoid redundancy**: Remove unused directories, keep structure lean

## Additional Resources

For evolution roadmap and migration guidelines, see [reference.md](reference.md).
