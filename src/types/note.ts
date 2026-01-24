export interface Question {
  type: string;
  subType?: string;
}

export interface Quality {
  overall: number;
  coverage: number;
  depth: number;
  specificity: number;
  reviewer: 'ai' | 'human' | 'hybrid';
}

export interface Objectivity {
  factRatio: number;
  inferenceRatio: number;
  opinionRatio: number;
}

export interface Analysis {
  objectivity: Objectivity;
  assumptions: string[];
  limitations: string[];
}

export interface Review {
  status: 'draft' | 'reviewed' | 'deprecated';
  reviewedAt?: string;
}

export interface VersionHistory {
  version: number;
  updatedAt: string;
  updatedBy: 'ai' | 'human' | 'hybrid';
  changes?: string[];
  changesSummary?: string;
}

export interface Version {
  current: number;
  history?: VersionHistory[];
}

export interface Tag {
  name: string;
  parent?: string;
  alias?: string[];
}

export interface Topic {
  name: string;
  alias?: string[];
}

export interface NoteFrontmatter {
  title: string;
  slug: string;
  createdAt: string;
  updatedAt: string;
  date: string;
  version?: Version;
  question: Question;
  quality: Quality;
  analysis: Analysis;
  review: Review;
  tags: Tag[];
  topics: Topic[];
}

export interface Note extends NoteFrontmatter {
  content: string;
}
