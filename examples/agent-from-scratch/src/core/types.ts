// The wire-level shapes an agent passes around. Deliberately a small subset of
// the Anthropic Messages API: enough to show the mechanics, not so much that the
// SDK's surface drowns out the ideas. Both the real model adapter and the
// offline mock speak exactly these types, so every stage runs with or without a
// key (see ./llm.ts).

// A tool the model is allowed to call. `inputSchema` is a JSON Schema object —
// it is the entire contract AND the entire prompt the model sees for the tool,
// which is why tool descriptions are engineering, not documentation (chapter 02).
export interface ToolSpec {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
  };
}

// What the assistant emits. Either prose, or a request to run a tool. A single
// turn can contain several blocks (e.g. a sentence of reasoning + two parallel
// tool_use blocks).
export type AssistantBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> };

// What the caller feeds back. tool_result blocks must echo the tool_use id so
// the model can correlate the result with the call it made.
export type UserBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_result'; toolUseId: string; content: string; isError?: boolean };

export type Message =
  | { role: 'user'; content: string | UserBlock[] }
  | { role: 'assistant'; content: AssistantBlock[] };

// Why the model stopped. `tool_use` is the signal that drives the agent loop:
// the model wants the harness to run something and come back (chapter 01).
export type StopReason = 'end_turn' | 'tool_use' | 'max_tokens';

export interface AssistantTurn {
  content: AssistantBlock[];
  stopReason: StopReason;
  // Approximate; the mock fills these from a char heuristic, the real adapter
  // from the API's usage field. Used for the cost curves in chapters 01 and 05.
  usage: { inputTokens: number; outputTokens: number };
}

export interface GenerateOptions {
  system?: string;
  messages: Message[];
  tools?: ToolSpec[];
  maxTokens?: number;
}

// The one capability every stage depends on: turn a conversation into the next
// assistant turn. Keeping this to a single method is the point — an agent is a
// loop around exactly this call.
export interface LLM {
  readonly name: string;
  generate(opts: GenerateOptions): Promise<AssistantTurn>;
}
