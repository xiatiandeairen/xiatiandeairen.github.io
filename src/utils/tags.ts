import type { Note } from './notes';
import { TAG_MAX_DEPTH } from './constants';

export interface TagNode {
  name: string;
  parent?: string;
  alias?: string[];
  depth: number;
  children: TagNode[];
}

export function buildTagHierarchy(notes: Note[]): Map<string, TagNode> {
  const tagMap = new Map<string, TagNode>();
  const allTags = new Set<string>();

  for (const note of notes) {
    for (const tag of note.tags) {
      allTags.add(tag.name);
      if (tag.alias) {
        tag.alias.forEach(alias => allTags.add(alias));
      }
    }
  }

  for (const note of notes) {
    for (const tag of note.tags) {
      if (!tagMap.has(tag.name)) {
        tagMap.set(tag.name, {
          name: tag.name,
          parent: tag.parent,
          alias: tag.alias,
          depth: 0,
          children: []
        });
      }
    }
  }

  function calculateDepth(tagName: string, visited: Set<string> = new Set()): number {
    if (visited.has(tagName)) return 0;
    visited.add(tagName);

    const tag = tagMap.get(tagName);
    if (!tag || !tag.parent) return 1;

    return 1 + calculateDepth(tag.parent, visited);
  }

  for (const [tagName, tag] of tagMap.entries()) {
    tag.depth = calculateDepth(tagName);
    if (tag.parent) {
      const parentTag = tagMap.get(tag.parent);
      if (parentTag) {
        parentTag.children.push(tag);
      }
    }
  }

  return tagMap;
}

export function getTagPath(tagName: string, notes: Note[]): string {
  const hierarchy = buildTagHierarchy(notes);
  const tag = hierarchy.get(tagName);
  if (!tag || !tag.parent) return tagName;

  const path: string[] = [tagName];
  let current = tag;
  while (current.parent) {
    path.unshift(current.parent);
    const parent = hierarchy.get(current.parent);
    if (!parent) break;
    current = parent;
  }

  return path.join('/');
}

export function getChildTags(parentTag: string, notes: Note[]): string[] {
  const hierarchy = buildTagHierarchy(notes);
  const parent = hierarchy.get(parentTag);
  if (!parent) return [];

  const children: string[] = [];
  function collectChildren(node: TagNode) {
    for (const child of node.children) {
      children.push(child.name);
      collectChildren(child);
    }
  }

  collectChildren(parent);
  return children;
}

export function getParentTag(tagName: string, notes: Note[]): string | null {
  const hierarchy = buildTagHierarchy(notes);
  const tag = hierarchy.get(tagName);
  return tag?.parent || null;
}

export function resolveTagAlias(tagName: string, notes: Note[]): string {
  const hierarchy = buildTagHierarchy(notes);
  
  for (const [name, tag] of hierarchy.entries()) {
    if (name === tagName) return name;
    if (tag.alias?.includes(tagName)) return name;
  }

  return tagName;
}

export function resolveTopicAlias(topicName: string, notes: Note[]): string {
  const allTopics = new Set<{ name: string; alias?: string[] }>();

  for (const note of notes) {
    for (const topic of note.topics) {
      allTopics.add(topic);
    }
  }

  for (const topic of allTopics) {
    if (topic.name === topicName) return topic.name;
    if (topic.alias?.includes(topicName)) return topic.name;
  }

  return topicName;
}

export function getTagDepth(tagName: string, notes: Note[]): number {
  const hierarchy = buildTagHierarchy(notes);
  const tag = hierarchy.get(tagName);
  return tag?.depth || 0;
}

export function shouldCollapseTag(tagName: string, notes: Note[], maxDepth: number = TAG_MAX_DEPTH): boolean {
  return getTagDepth(tagName, notes) > maxDepth;
}
