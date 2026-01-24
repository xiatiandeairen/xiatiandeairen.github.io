# Commit Convention - Reference

## Tool Support

### commitizen

Interactive commit message generation:

```bash
npm install -g commitizen
cz commit
```

### commitlint

Validate commit message format:

```bash
npm install --save-dev @commitlint/cli @commitlint/config-conventional
```

Create `commitlint.config.js`:

```javascript
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',
        'fix',
        'docs',
        'style',
        'refactor',
        'perf',
        'test',
        'build',
        'ci',
        'chore',
        'revert',
      ],
    ],
  },
};
```

### husky

Automate checks with Git hooks:

```bash
npm install --save-dev husky
npx husky install
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit ${1}'
```

## Additional Examples

### Build System

```
build(deps): add @tailwindcss/typography plugin

Add typography plugin to support prose classes for Markdown content.
```

### CI Configuration

```
ci(workflows): update GitHub Pages deployment

Configure Astro build output and add build cache for faster CI runs.
```

### Chore

```
chore: remove Hexo-related files

Delete _config.yml, themes/, and other Hexo-specific files.
Clean up project structure for Astro migration.
```

## Notes

1. **Don't commit**: `node_modules/`, `dist/`, `.astro/` build artifacts
2. **Don't commit**: Personal config, temp files, sensitive info
3. **Commit promptly**: After completing a feature point, don't accumulate large changes
4. **Clear descriptions**: Messages should help other developers (including future you) understand changes
