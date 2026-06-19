// Shared markdown post-processing for content detail pages. slugify must stay
// identical between addHeadingIds (injects ids into rendered h2/h3) and
// extractToc (builds the TOC anchors) or TOC links break. Extracted from the
// notes detail page so the library detail page reuses the same anchor algorithm.

export function slugify(text: string): string {
  return text.toLowerCase().replace(/[^\w一-鿿]+/g, '-').replace(/^-|-$/g, '');
}

export function addHeadingIds(html: string): string {
  return html.replace(/<h([23])>([\s\S]*?)<\/h\1>/g, (_, level, inner) => {
    const raw = inner.replace(/<[^>]+>/g, '').replace(/&[^;]+;/g, ' ').trim();
    const id = slugify(raw);
    return `<h${level} id="${id}">${inner}</h${level}>`;
  });
}

export interface TocItem {
  id: string;
  text: string;
  level: number;
}

export function extractToc(content: string): TocItem[] {
  const toc: TocItem[] = [];
  for (const line of content.split('\n')) {
    const match = line.match(/^(#{2,3})\s+(.+)$/);
    if (match) {
      const text = match[2].replace(/\*\*?|`[^`]+`/g, '').trim();
      toc.push({ id: slugify(text), text, level: match[1].length });
    }
  }
  return toc;
}
