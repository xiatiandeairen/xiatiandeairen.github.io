// Stage 01 — the agent loop.
//
// An agent is a `while` loop around one LLM call, plus tools, plus a stop
// condition. That sentence is the whole chapter; everything below is making it
// precise. Run it: `npm run stage01` (offline mock by default).

import { createLLM, MockLLM } from './core/llm.js';
import type { AssistantBlock, LLM, Message, ToolSpec, UserBlock } from './core/types.js';

// --- Tools: a spec the model sees, and an impl the harness runs. ------------

const TOOLS: ToolSpec[] = [
  {
    name: 'word_count',
    description: 'Count the words in a piece of text. Returns the integer count.',
    inputSchema: {
      type: 'object',
      properties: { text: { type: 'string', description: 'the text to count' } },
      required: ['text'],
    },
  },
];

const TOOL_IMPLS: Record<string, (input: Record<string, unknown>) => Promise<string>> = {
  async word_count(input) {
    const text = String(input.text ?? '');
    const count = text.trim().split(/\s+/).filter(Boolean).length;
    return String(count);
  },
};

// --- The loop. -------------------------------------------------------------

export interface LoopResult {
  answer: string;
  turns: number;
  // inputTokens for each model call, in order. The whole transcript is resent
  // every turn, so this grows turn over turn — the cost of a T-turn agent is the
  // SUM of these, i.e. O(T^2) in transcript length, not O(T). See printout.
  inputTokensPerTurn: number[];
  stopReason: 'answered' | 'max_turns';
}

export interface LoopConfig {
  system: string;
  // Hard ceiling so a model that never stops calling tools cannot bill forever.
  // This is the single most important reliability knob in the loop (failure
  // mode: infinite tool calling). Default low on purpose.
  maxTurns: number;
}

export async function runAgent(
  llm: LLM,
  tools: ToolSpec[],
  toolImpls: Record<string, (input: Record<string, unknown>) => Promise<string>>,
  userInput: string,
  config: LoopConfig
): Promise<LoopResult> {
  const messages: Message[] = [{ role: 'user', content: userInput }];
  const inputTokensPerTurn: number[] = [];

  for (let turn = 0; turn < config.maxTurns; turn++) {
    const assistant = await llm.generate({ system: config.system, messages, tools });
    inputTokensPerTurn.push(assistant.usage.inputTokens);
    messages.push({ role: 'assistant', content: assistant.content });

    // Stop condition #1: the model is done. Anything other than a tool request
    // ends the loop — this is the normal exit, not an error.
    if (assistant.stopReason !== 'tool_use') {
      return {
        answer: textOf(assistant.content),
        turns: turn + 1,
        inputTokensPerTurn,
        stopReason: 'answered',
      };
    }

    // Otherwise: run every requested tool, feed all results back as ONE user
    // turn (parallel tool_use blocks must be answered together), and loop.
    const results: UserBlock[] = [];
    for (const block of assistant.content) {
      if (block.type !== 'tool_use') continue;
      const impl = toolImpls[block.name];
      if (!impl) {
        // Failure mode: model hallucinated a tool name. Tell it, don't crash —
        // recovery is the model's job once it sees the error (chapter 02).
        results.push({ type: 'tool_result', toolUseId: block.id, content: `unknown tool: ${block.name}`, isError: true });
        continue;
      }
      try {
        results.push({ type: 'tool_result', toolUseId: block.id, content: await impl(block.input) });
      } catch (err) {
        results.push({
          type: 'tool_result',
          toolUseId: block.id,
          content: `tool error: ${(err as Error).message}`,
          isError: true,
        });
      }
    }
    messages.push({ role: 'user', content: results });
  }

  // Stop condition #2: ran out of turns. Returning a marker (not throwing) lets
  // the caller decide degrade-vs-fail — the loop's job is to never run forever.
  return { answer: '[stopped: max turns reached]', turns: config.maxTurns, inputTokensPerTurn, stopReason: 'max_turns' };
}

function textOf(content: AssistantBlock[]): string {
  return content.filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text').map((b) => b.text).join('');
}

// --- Offline mock policy: turn 0 calls the tool, turn 1 answers. ------------

const mock = new MockLLM((opts, turnIndex) => {
  if (turnIndex === 0) {
    const userText = typeof opts.messages[0].content === 'string' ? opts.messages[0].content : '';
    const quoted = userText.match(/"([^"]+)"/)?.[1] ?? userText;
    return {
      content: [
        { type: 'text', text: '我先数一下词数。' },
        { type: 'tool_use', id: 'call_1', name: 'word_count', input: { text: quoted } },
      ],
    };
  }
  // turnIndex >= 1: the last user turn holds the tool_result; read it and answer.
  const last = opts.messages[opts.messages.length - 1];
  const resultBlock = typeof last.content === 'string' ? null : last.content.find((b) => b.type === 'tool_result');
  const count = resultBlock && resultBlock.type === 'tool_result' ? Number(resultBlock.content) : 0;
  return {
    content: [{ type: 'text', text: `这句话有 ${count} 个词，${count > 5 ? '超过' : '没有超过'} 5 个。` }],
    stopReason: 'end_turn',
  };
});

async function main(): Promise<void> {
  const llm = createLLM(mock);
  console.log(`\n=== Stage 01: agent loop (llm=${llm.name}) ===\n`);

  const result = await runAgent(
    llm,
    TOOLS,
    TOOL_IMPLS,
    'How many words are in "the quick brown fox jumps over the lazy dog"? Tell me if it is more than 5.',
    { system: 'You are a precise assistant. Use tools when you need an exact count.', maxTurns: 6 }
  );

  console.log('answer    :', result.answer);
  console.log('turns     :', result.turns, `(stop: ${result.stopReason})`);
  console.log('input tok :', result.inputTokensPerTurn.join(' → '), '(grows every turn — full transcript resent)');
  const total = result.inputTokensPerTurn.reduce((a, b) => a + b, 0);
  console.log('total in  :', total, 'input tokens billed across the run\n');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
