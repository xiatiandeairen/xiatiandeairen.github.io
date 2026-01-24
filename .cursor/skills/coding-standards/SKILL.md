---
name: coding-standards
description: Enforces TypeScript strict mode, code style conventions, and best practices for Astro projects. Use when writing code, reviewing code, or when the user asks about coding standards, TypeScript configuration, or code quality.
---

# Coding Standards

## TypeScript Strict Mode

Enable strict mode in `tsconfig.json`:

```json
{
  "extends": "astro/tsconfigs/strict",
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true
  }
}
```

### Type Safety

- Avoid `any` - use specific types or `unknown`
- Explicit type definitions for functions and parameters
- Handle `null` and `undefined` explicitly with `?.` and `??`

## Code Style

### Naming Conventions

- Variables/functions: `camelCase`
- Types/interfaces: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` or `camelCase` (by scope)
- Component files: `PascalCase` (e.g., `NoteCard.astro`)

### Functions

- Single responsibility
- Pure functions preferred
- Max 5 parameters (use objects for more)
- Explicit return types

```typescript
function getNotesByTag(tagName: string, notes: Note[]): Note[] {
  return notes.filter(note => /* ... */);
}
```

### Error Handling

- Use custom error classes
- Throw immediately, don't fail silently
- Include context in error messages

```typescript
if (duplicates.length > 0) {
  const errorMessages = duplicates.map(
    ({ slug, files }) => `Slug "${slug}" is duplicated in: ${files.join(', ')}`
  );
  throw new Error(`Duplicate slugs found:\n${errorMessages.join('\n')}`);
}
```

## Code Organization

### Import Order

1. External dependencies
2. Internal utilities
3. Type definitions
4. Components
5. Styles

```typescript
import { z } from 'zod';
import { validateNoteFrontmatter } from './schema';
import type { Note } from './notes';
import NoteCard from '../components/NoteCard.astro';
import '../styles/global.css';
```

### File Organization

- One main function per file
- Related functions grouped together
- Split files over 300 lines

### Comments

**Default: No comments unless explicitly requested**

Only add comments for:
- Complex algorithms or business logic
- Non-obvious implementation details
- TODO or FIXME markers

## Astro Components

### Component Structure

```astro
---
// 1. Imports
import Layout from '../layouts/Layout.astro';

// 2. Type definitions
interface Props {
  title: string;
  description?: string;
}

// 3. Props destructuring
const { title, description } = Astro.props;

// 4. Data fetching/computation
const data = await fetchData();
---

<!-- 5. Template -->
<Layout>
  <Component data={data} />
</Layout>
```

All components must define Props interface.

## Utility Functions

### Pure Functions Preferred

```typescript
function sortNotes(notes: Note[], sortBy: SortField): Note[] {
  return [...notes].sort(/* ... */);
}
```

### Explicit Error Handling

```typescript
export function getAllNotes(): Note[] {
  let files: string[];
  try {
    files = readdirSync(notesDir);
  } catch (error) {
    return [];
  }
  // ...
}
```

## Performance

- Memoize expensive operations
- Execute file system operations at build time, not runtime
- Use caching to avoid repeated reads
- Batch process to reduce I/O

## Code Review Checklist

- [ ] TypeScript strict mode passes
- [ ] No `any` types (unless justified)
- [ ] All functions have explicit type definitions
- [ ] Error handling is complete
- [ ] Code follows naming conventions
- [ ] No unused imports or variables
- [ ] Code formatted consistently
- [ ] No console.log or debug code

## Additional Resources

For detailed configuration examples and tool setup, see [reference.md](reference.md).
