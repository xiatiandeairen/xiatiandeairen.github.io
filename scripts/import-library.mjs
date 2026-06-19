// One-shot importer: study course markdown -> src/content/library/<collection>/.
//
// Usage: node scripts/import-library.mjs <sourceDir> <collectionSlug>
// Example: node scripts/import-library.mjs /Users/taoxia/study/study ai-app-engineering
//
// Reads NN-*.md chapter files, derives a lean frontmatter (title / slug /
// collection / order / summary / topics), and writes them under
// src/content/library/<collectionSlug>/. The body is copied verbatim minus the
// leading H1 (the detail page renders the title from frontmatter).
//
// Honest-data contract: we do NOT fabricate quality scores, objectivity ratios,
// or fine-grained tags. Only fields we can mechanically derive from the source
// are written. topics is collection-level; tags is left empty for later manual
// enrichment.

import { readdirSync, readFileSync, writeFileSync, mkdirSync, statSync } from 'fs';
import { join, basename } from 'path';

const [, , sourceDir, collectionSlug] = process.argv;
if (!sourceDir || !collectionSlug) {
  console.error('usage: node scripts/import-library.mjs <sourceDir> <collectionSlug>');
  process.exit(1);
}

// Collection-level topic shown on every chapter; keep in sync with the registry
// in src/utils/library.ts.
const COLLECTION_TOPIC = {
  'ai-app-engineering': 'AI 应用工程',
};

const destDir = join('src', 'content', 'library', collectionSlug);
mkdirSync(destDir, { recursive: true });

// Chapter files start with a numeric prefix (00-, 01-, ...). Everything else
// (README.md, build_site.py, index.html) is excluded by this filter.
const chapterFiles = readdirSync(sourceDir)
  .filter((f) => /^\d+-.*\.md$/.test(f))
  .sort();

function deriveTitle(firstLine) {
  // First line is the H1. Strip "# ", then the chapter marker in any of the
  // observed shapes: "第 N 章 ·", "第 N 章", "NN ·", and a leading "·".
  return firstLine
    .replace(/^#\s+/, '')
    .replace(/^第\s*\d+\s*章\s*[·.、]?\s*/, '')
    .replace(/^\d+\s*[·.、]\s*/, '')
    .trim();
}

function deriveSummary(lines) {
  // Collect blockquote paragraphs (split on blank "> " lines) right after the
  // H1, then pick the first SUBSTANTIAL one. Some chapters open with a stage
  // label paragraph ("阶段〇 · 底座 · 第 2 章") before the real intro, so a
  // naive "first paragraph" grabs the label. Finally truncate to a sentence
  // boundary near 150 chars for card/SEO use.
  const paragraphs = [];
  let current = [];
  let inQuote = false;
  for (const line of lines) {
    if (line.startsWith('>')) {
      inQuote = true;
      const text = line.replace(/^>\s?/, '');
      if (text.trim() === '') {
        if (current.length) { paragraphs.push(current.join(' ')); current = []; }
      } else {
        current.push(text);
      }
    } else if (inQuote) {
      break; // blockquote ended
    }
  }
  if (current.length) paragraphs.push(current.join(' '));

  const isLabel = (p) => p.length < 30 || /^阶段|^第\s*\d+\s*章\s*$/.test(p.trim());
  const picked = paragraphs.find((p) => !isLabel(p)) || paragraphs[0] || '';
  const clean = picked.replace(/\*\*/g, '').replace(/\s+/g, ' ').trim();

  if (clean.length <= 160) return clean;
  // Cut at the last sentence terminator before 160 chars; fall back to a hard
  // cut with an ellipsis.
  const head = clean.slice(0, 160);
  const lastStop = Math.max(head.lastIndexOf('。'), head.lastIndexOf('！'), head.lastIndexOf('？'));
  return lastStop >= 60 ? head.slice(0, lastStop + 1) : head.slice(0, 150).trim() + '…';
}

function stripLeadingH1(content) {
  const idx = content.indexOf('\n');
  return idx === -1 ? '' : content.slice(idx + 1).replace(/^\s+/, '');
}

function yamlString(s) {
  // Double-quote and escape; titles/summaries contain ":" and quotes.
  return '"' + s.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
}

let count = 0;
for (const file of chapterFiles) {
  const order = parseInt(file.match(/^(\d+)/)[1], 10);
  const slug = String(order).padStart(2, '0');
  const raw = readFileSync(join(sourceDir, file), 'utf-8');
  const lines = raw.split('\n');
  const title = deriveTitle(lines[0]);
  const summary = deriveSummary(lines.slice(1));
  const body = stripLeadingH1(raw);
  const mtime = statSync(join(sourceDir, file)).mtime.toISOString().replace(/\.\d{3}Z$/, '.000Z');
  const topic = COLLECTION_TOPIC[collectionSlug];

  const frontmatter = [
    '---',
    `title: ${yamlString(title)}`,
    `slug: ${yamlString(slug)}`,
    `collection: ${yamlString(collectionSlug)}`,
    `order: ${order}`,
    `summary: ${yamlString(summary)}`,
    `topics:${topic ? `\n  - ${yamlString(topic)}` : ' []'}`,
    'tags: []',
    `createdAt: ${yamlString(mtime)}`,
    `updatedAt: ${yamlString(mtime)}`,
    '---',
    '',
  ].join('\n');

  const destName = `${slug}-${basename(file).replace(/^\d+-/, '')}`;
  writeFileSync(join(destDir, destName), frontmatter + body, 'utf-8');
  count += 1;
  console.log(`  ${destName}  <-  ${file}  [order=${order}] ${title}`);
}

console.log(`\nimported ${count} chapters into ${destDir}`);
