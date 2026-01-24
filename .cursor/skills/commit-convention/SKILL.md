---
name: commit-convention
description: Generates commit messages following Conventional Commits specification with English descriptions. Use when creating commits, writing commit messages, or when the user asks about commit format or git conventions.
---

# Commit Convention

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Type (Required)

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no runtime impact)
- `refactor`: Code refactoring
- `perf`: Performance optimization
- `test`: Test-related changes
- `build`: Build system or dependencies
- `ci`: CI configuration and scripts
- `chore`: Other changes (deps, tool config)
- `revert`: Revert previous commit

## Scope (Optional)

Examples: `schema`, `pages`, `components`, `utils`, `styles`, `config`, `deps`

## Subject (Required)

- Imperative mood, present tense ("add" not "added" or "adds")
- Lowercase first letter
- No period at end
- Max 50 characters

## Body (Optional)

- Explain motivation
- Compare with previous behavior
- Max 72 characters per line

## Footer (Optional)

- Close issues: `Closes #123`
- Breaking changes: `BREAKING CHANGE: <description>`

## Examples

### New Feature

```
feat(notes): add version history support

Implement version management system with history tracking.
Each note can now maintain version history with change summaries.

Closes #45
```

### Bug Fix

```
fix(pages): correct import path in page/[page].astro

The IndexLayout import path was incorrect, causing build failures.
Changed from '../IndexLayout.astro' to '../../layouts/IndexLayout.astro'.
```

### Documentation

```
docs(readme): update installation instructions

Add pnpm installation steps and clarify dependency requirements.
```

### Refactoring

```
refactor(components): simplify NoteCard component

Remove unnecessary card wrapper and use semantic HTML elements.
Improve accessibility with proper article tags.
```

### Style Changes

```
style(global): update typography for better readability

Increase base font size to 18px and line height to 1.8.
Apply pure style design principles.
```

### Performance

```
perf(build): optimize static page generation

Implement incremental build strategy for large note collections.
Reduce build time by 40% for 1000+ notes.
```

## Principles

1. Follow Conventional Commits specification
2. **All descriptions in English**
3. Single logical change per commit
4. Clear and specific descriptions

## Pre-Commit Checklist

- [ ] Follows Conventional Commits format
- [ ] Uses English descriptions
- [ ] Contains single logical change
- [ ] Correct type selected
- [ ] Subject is concise (≤50 chars)
- [ ] Body included if needed
- [ ] Breaking changes marked with BREAKING CHANGE

## Additional Resources

For tool support and detailed examples, see [reference.md](reference.md).
