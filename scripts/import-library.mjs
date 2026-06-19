// One-shot importer: study/course markdown -> src/content/library/<collection>/.
//
// Usage: node scripts/import-library.mjs <sourceDir> <collectionSlug>
//
// Two source shapes are supported:
//   * Flat  — NN-*.md chapter files directly under sourceDir (e.g. study/study).
//   * Grouped — chapters live in subdirectories, each subdir = a group within the
//     collection (e.g. tech-library/推理引擎/01-*.md). Groups are ordered by a
//     numeric dir prefix ("2-MLSys专家课程" -> 2) or, for un-prefixed dirs, by the
//     groupOrder map in COLLECTION_CONFIG below.
//
// Honest-data contract: we do NOT fabricate quality scores, objectivity ratios,
// or fine-grained tags. Only mechanically-derivable fields are written; topics is
// collection-level and tags is left empty for later manual enrichment.

import { readFileSync, readdirSync, writeFileSync, mkdirSync, statSync } from 'fs';
import { join } from 'path';

const [, , sourceDir, collectionSlug] = process.argv;
if (!sourceDir || !collectionSlug) {
  console.error('usage: node scripts/import-library.mjs <sourceDir> <collectionSlug>');
  process.exit(1);
}

const COLLECTION_CONFIG = {
  'ai-app-engineering': { topic: 'AI 应用工程' },
  'tech-library': {
    topic: '技术内核',
    // Un-prefixed group dirs need an explicit order (systems/AI-core first,
    // platforms last). Any dir missing here errors rather than guessing.
    groupOrder: {
      '大模型': 1, '推理引擎': 2, '数据检索底座': 3, '数据库': 4, '编译器': 5,
      '强化学习': 6, 'chromium内核': 7, 'android系统': 8, 'linux系统': 9, 'webrtc': 10,
    },
  },
  'ai-research-compass': { topic: 'AI 研究' },
  'indie-ai-fullstack': { topic: '独立开发' },
};

const config = COLLECTION_CONFIG[collectionSlug];
if (!config) {
  console.error(`no COLLECTION_CONFIG entry for "${collectionSlug}"`);
  process.exit(1);
}

const SKIP_DIRS = new Set(['notebooks']);

const destDir = join('src', 'content', 'library', collectionSlug);
mkdirSync(destDir, { recursive: true });

function deriveTitle(firstLine) {
  // First line is the H1. Strip "# ", then a chapter marker in any observed
  // shape: "第 N 章 ·", "第 N 章", "NN ·", leading "·". Titles without a marker
  // (e.g. "GPU 体系结构...：物理基础") are kept verbatim.
  return firstLine
    .replace(/^#\s+/, '')
    .replace(/^第\s*\d+\s*章\s*[·.、]?\s*/, '')
    .replace(/^\d+\s*[·.、]\s*/, '')
    .trim();
}

function deriveSummary(lines) {
  // First SUBSTANTIAL blockquote paragraph after the H1, skipping stage labels
  // ("阶段〇 · 底座 · 第 2 章"); truncated to a sentence boundary near 150 chars.
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
      break;
    }
  }
  if (current.length) paragraphs.push(current.join(' '));

  const isLabel = (p) => p.length < 30 || /^阶段|^第\s*\d+\s*章\s*$/.test(p.trim());
  const picked = paragraphs.find((p) => !isLabel(p)) || paragraphs[0] || '';
  const clean = picked.replace(/\*\*/g, '').replace(/\s+/g, ' ').trim();
  if (clean.length <= 160) return clean;
  const head = clean.slice(0, 160);
  const lastStop = Math.max(head.lastIndexOf('。'), head.lastIndexOf('！'), head.lastIndexOf('？'));
  return lastStop >= 60 ? head.slice(0, lastStop + 1) : head.slice(0, 150).trim() + '…';
}

function stripLeadingH1(content) {
  const idx = content.indexOf('\n');
  return idx === -1 ? '' : content.slice(idx + 1).replace(/^\s+/, '');
}

function yamlString(s) {
  return '"' + s.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
}

// Emit one chapter file. order is the global sequence within the collection
// (groupOrder*1000 + chapterNum) so prev/next and grouping both derive from it.
function writeChapter({ srcPath, chapterNum, slug, order, group, destName }) {
  const raw = readFileSync(srcPath, 'utf-8');
  const lines = raw.split('\n');
  const title = deriveTitle(lines[0]);
  const summary = deriveSummary(lines.slice(1));
  const body = stripLeadingH1(raw);
  const mtime = statSync(srcPath).mtime.toISOString().replace(/\.\d{3}Z$/, '.000Z');

  const fm = [
    '---',
    `title: ${yamlString(title)}`,
    `slug: ${yamlString(slug)}`,
    `collection: ${yamlString(collectionSlug)}`,
    ...(group ? [`group: ${yamlString(group)}`] : []),
    `order: ${order}`,
    `summary: ${yamlString(summary)}`,
    `topics:\n  - ${yamlString(config.topic)}`,
    'tags: []',
    `createdAt: ${yamlString(mtime)}`,
    `updatedAt: ${yamlString(mtime)}`,
    '---',
    '',
  ].join('\n');

  writeFileSync(join(destDir, destName), fm + body, 'utf-8');
  console.log(`  ${destName}  [order=${order}] ${title}`);
}

const entries = readdirSync(sourceDir);
const chapterFiles = entries.filter((f) => /^\d+-.*\.md$/.test(f));
let count = 0;

if (chapterFiles.length > 0) {
  // Flat collection (study/study): order = chapter number, slug = padded number.
  for (const file of chapterFiles.sort()) {
    const chapterNum = parseInt(file.match(/^(\d+)/)[1], 10);
    const slug = String(chapterNum).padStart(2, '0');
    writeChapter({
      srcPath: join(sourceDir, file),
      chapterNum,
      slug,
      order: chapterNum,
      group: null,
      destName: `${slug}-${file.replace(/^\d+-/, '')}`,
    });
    count += 1;
  }
} else {
  // Grouped collection: each subdir is a group.
  const groupDirs = entries.filter((name) => {
    if (SKIP_DIRS.has(name)) return false;
    return statSync(join(sourceDir, name)).isDirectory();
  });

  for (const dir of groupDirs) {
    const numbered = dir.match(/^(\d+)-(.+)$/);
    let groupOrder, groupTitle;
    if (numbered) {
      groupOrder = parseInt(numbered[1], 10);
      groupTitle = numbered[2].trim();
    } else {
      groupOrder = config.groupOrder?.[dir];
      groupTitle = dir;
      if (groupOrder === undefined) {
        console.error(`group "${dir}" in ${collectionSlug} has no order (add to COLLECTION_CONFIG.groupOrder)`);
        process.exit(1);
      }
    }

    const files = readdirSync(join(sourceDir, dir)).filter((f) => /^\d+-.*\.md$/.test(f));
    for (const file of files.sort()) {
      const chapterNum = parseInt(file.match(/^(\d+)/)[1], 10);
      const slug = `${groupOrder}-${String(chapterNum).padStart(2, '0')}`;
      writeChapter({
        srcPath: join(sourceDir, dir, file),
        chapterNum,
        slug,
        order: groupOrder * 1000 + chapterNum,
        group: groupTitle,
        destName: `${slug}-${file.replace(/^\d+-/, '')}`,
      });
      count += 1;
    }
  }
}

console.log(`\nimported ${count} chapters into ${destDir}`);
