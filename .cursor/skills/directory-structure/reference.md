# Directory Structure - Reference

## Evolution Roadmap

### Stage 1: Current (0-1000 notes)
- Flat structure, all notes in `src/content/notes/`
- Simple component organization

### Stage 2: Medium Scale (1000-5000 notes)
- Organize by year: `notes/2026/`, `notes/2027/`
- Or by topic: `notes/frontend/`, `notes/backend/`
- Group components by function

### Stage 3: Large Scale (5000+ notes)
- Multi-level categories: `notes/2026/frontend/react/`
- Component library: `components/ui/`, `components/features/`
- Modular utilities: `utils/notes/`, `utils/tags/`, `utils/search/`

## Future Extensions

### `/docs/` (Future)

Project documentation:
- Incremental build docs
- Build cache strategy
- Performance testing guide
- API documentation

### Component Extensions

Future additions:
- `components/forms/`
- `components/charts/`
- `components/ui/` (UI library)

### Page Extensions

Future additions:
- `pages/admin/`
- `pages/api/`

### Utility Extensions

Future subdirectories:
- `utils/validation/`
- `utils/formatting/`
- `utils/notes/`
- `utils/tags/`
- `utils/search/`

## Directory Maintenance

### Regular Checks

- Quarterly review for unused directories
- Clean up temporary and test files
- Ensure .gitignore is correctly configured

### Refactoring Principles

- Assess impact scope before refactoring
- Maintain backward compatibility when possible
- Update related documentation

## Migration Guide

When adjusting directory structure:

1. Create migration plan document
2. Update all import paths
3. Update build configuration
4. Test and verify
5. Update documentation
