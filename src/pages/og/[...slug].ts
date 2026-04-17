import { OGImageRoute } from 'astro-og-canvas';
import { getAllNotes } from '../../utils/notes';

const notes = getAllNotes();
const pages = Object.fromEntries(
  notes.map((note) => [
    note.slug,
    { title: note.title, description: note.content.slice(0, 120).replace(/\n/g, ' ').trim() },
  ])
);

export const { getStaticPaths, GET } = await OGImageRoute({
  param: 'slug',
  pages,
  getImageOptions: (_path, page) => ({
    title: page.title,
    description: page.description,
    bgGradient: [[244, 240, 232]],
    font: {
      title: { families: ['Noto Serif SC'], weight: 'Bold', size: 64, color: [40, 36, 28] },
      description: { families: ['Noto Serif SC'], size: 28, color: [120, 112, 96] },
    },
    border: { width: 12, color: [40, 36, 28], side: 'inline-start' },
    padding: 60,
  }),
});
