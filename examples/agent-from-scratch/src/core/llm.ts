import type {
  AssistantBlock,
  AssistantTurn,
  GenerateOptions,
  LLM,
  Message,
  StopReason,
  UserBlock,
} from './types.js';

// Rough token estimate: ~4 chars per token for mixed English/code. This is NOT
// a real tokenizer (Anthropic's BPE would differ, especially for CJK), but it is
// stable and dependency-free, which is all the cost curves in chapters 01/05
// need. When a real key is set, the adapter reports the API's exact usage and
// this is bypassed.
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function estimateMessagesTokens(opts: GenerateOptions): number {
  let chars = (opts.system ?? '').length;
  for (const m of opts.messages) {
    if (typeof m.content === 'string') {
      chars += m.content.length;
    } else {
      for (const b of m.content) chars += JSON.stringify(b).length;
    }
  }
  for (const t of opts.tools ?? []) chars += JSON.stringify(t).length;
  return Math.ceil(chars / 4);
}

// ----------------------------------------------------------------------------
// Offline mock. A deterministic policy turns the running conversation into the
// next assistant turn, so the agent loop (chapter 01), tool round-trips (02),
// permission gates (04) etc. all run with no network and no key. The policy is
// pure; the mock only tracks how many turns have happened so a stage can script
// "turn 0 calls a tool, turn 1 answers".
// ----------------------------------------------------------------------------

export type MockPolicy = (
  opts: GenerateOptions,
  turnIndex: number
) => { content: AssistantBlock[]; stopReason?: StopReason };

export class MockLLM implements LLM {
  readonly name = 'mock';
  private turnIndex = 0;

  constructor(private readonly policy: MockPolicy) {}

  async generate(opts: GenerateOptions): Promise<AssistantTurn> {
    const { content, stopReason } = this.policy(opts, this.turnIndex);
    this.turnIndex += 1;
    const inferredStop: StopReason =
      stopReason ?? (content.some((b) => b.type === 'tool_use') ? 'tool_use' : 'end_turn');
    const outputChars = content.reduce((n, b) => n + JSON.stringify(b).length, 0);
    return {
      content,
      stopReason: inferredStop,
      usage: { inputTokens: estimateMessagesTokens(opts), outputTokens: Math.ceil(outputChars / 4) },
    };
  }
}

// ----------------------------------------------------------------------------
// Real adapter. Lazily loads @anthropic-ai/sdk only when a key is present, via a
// non-literal import specifier so `tsc` does not require the package to be
// installed for the offline path to typecheck. Maps our small wire types to/from
// the Messages API.
// ----------------------------------------------------------------------------

export class AnthropicLLM implements LLM {
  readonly name: string;

  constructor(private readonly apiKey: string, private readonly model: string) {
    this.name = `anthropic:${model}`;
  }

  async generate(opts: GenerateOptions): Promise<AssistantTurn> {
    // Non-literal specifier keeps tsc from resolving the optional dep.
    const spec = '@anthropic-ai/sdk';
    let mod: any;
    try {
      mod = await import(spec);
    } catch {
      throw new Error(
        'ANTHROPIC_API_KEY is set but @anthropic-ai/sdk is not installed. Run `npm install` in examples/agent-from-scratch, or unset the key to use the offline mock.'
      );
    }
    const Anthropic = mod.default;
    const client = new Anthropic({ apiKey: this.apiKey });

    const resp = await client.messages.create({
      model: this.model,
      max_tokens: opts.maxTokens ?? 1024,
      system: opts.system,
      messages: opts.messages.map(toSdkMessage),
      tools: opts.tools?.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.inputSchema,
      })),
    });

    const content: AssistantBlock[] = resp.content
      .map((b: any): AssistantBlock | null => {
        if (b.type === 'text') return { type: 'text', text: b.text };
        if (b.type === 'tool_use') return { type: 'tool_use', id: b.id, name: b.name, input: b.input };
        return null;
      })
      .filter((b: AssistantBlock | null): b is AssistantBlock => b !== null);

    return {
      content,
      stopReason: mapStopReason(resp.stop_reason),
      usage: { inputTokens: resp.usage.input_tokens, outputTokens: resp.usage.output_tokens },
    };
  }
}

function toSdkMessage(m: Message): any {
  if (m.role === 'assistant') {
    return { role: 'assistant', content: m.content };
  }
  if (typeof m.content === 'string') return { role: 'user', content: m.content };
  return {
    role: 'user',
    content: m.content.map((b: UserBlock) =>
      b.type === 'tool_result'
        ? { type: 'tool_result', tool_use_id: b.toolUseId, content: b.content, is_error: b.isError }
        : { type: 'text', text: b.text }
    ),
  };
}

function mapStopReason(r: string | null): StopReason {
  if (r === 'tool_use') return 'tool_use';
  if (r === 'max_tokens') return 'max_tokens';
  return 'end_turn';
}

// ----------------------------------------------------------------------------

// Pick the real model if a key is present, otherwise the supplied offline mock.
// Stages call this so the same code path runs in both modes; the only difference
// is which LLM answers.
export function createLLM(mock: LLM): LLM {
  const key = process.env.ANTHROPIC_API_KEY;
  if (key && key.trim().length > 0) {
    return new AnthropicLLM(key, process.env.AGENT_MODEL ?? 'claude-sonnet-4-6');
  }
  console.error('[llm] ANTHROPIC_API_KEY not set — running offline mock. Mechanics are real; the model is scripted.');
  return mock;
}
