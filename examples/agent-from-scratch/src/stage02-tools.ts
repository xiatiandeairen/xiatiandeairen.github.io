// Stage 02 — tools / function calling, the mechanism up close.
//
// Stage 01 showed the loop. This stage zooms into the one thing that makes the
// loop useful: the tool round-trip. Four claims, each demonstrated with a real
// printout, not prose:
//
//   1. A tool's `inputSchema` IS the prompt. The model never sees your impl —
//      only the name, description, and JSON Schema. Vague schema => vague calls.
//   2. The wire envelope is concrete. We `console.log` an actual `tool_use`
//      block and the `tool_result` block we feed back, so you can see the
//      `id`/`toolUseId` correlation that the whole protocol hinges on.
//   3. Parallel tool_use is an invariant, not a convenience: if one assistant
//      turn emits N tool_use blocks, the next user turn MUST carry all N
//      tool_results. Drop one and the API rejects the turn (mismatched ids).
//   4. Errors are data, not crashes. A failing tool comes back as a
//      tool_result with `isError: true`; the model reads it and recovers on the
//      next turn. We force a failure and print the recovery.
//
// Run it: `npm run stage02:tools` (offline mock by default — mechanics are real,
// the model is a scripted policy in ./core/llm.ts).
//
// Self-contained on purpose: this file re-implements a tiny agent loop instead
// of importing stage01. Importing any stageNN module would execute its main()
// at import time (they call main() at top level), polluting this stage's output.

import { createLLM, MockLLM } from './core/llm.js';
import type {
  AssistantBlock,
  AssistantTurn,
  LLM,
  Message,
  ToolSpec,
  UserBlock,
} from './core/types.js';

// ---------------------------------------------------------------------------
// Tools: three deliberately different schemas. The variety is the lesson —
// each schema shape teaches the model a different contract.
// ---------------------------------------------------------------------------

const TOOLS: ToolSpec[] = [
  {
    // Single required string. The description carries the units contract
    // ("metric", "returns °C") because the model has no other way to learn it —
    // there is no type for "Celsius". Prose in `description` is the only channel.
    name: 'get_weather',
    description:
      'Get the current temperature for a city. Units are metric; returns degrees Celsius. The city must be a real city name, not a country or region.',
    inputSchema: {
      type: 'object',
      properties: {
        city: { type: 'string', description: 'City name, e.g. "Tokyo" or "Paris".' },
      },
      required: ['city'],
    },
  },
  {
    // Enum-constrained string. The `enum` is a hard contract: it narrows the
    // model's output space at decode time far more reliably than a description
    // ever could ("only these four values are legal").
    name: 'convert_currency',
    description:
      'Convert an amount between two supported currencies using a fixed demo rate table.',
    inputSchema: {
      type: 'object',
      properties: {
        amount: { type: 'number', description: 'The amount to convert. Must be > 0.' },
        from: { type: 'string', enum: ['USD', 'EUR', 'JPY', 'CNY'], description: 'Source currency.' },
        to: { type: 'string', enum: ['USD', 'EUR', 'JPY', 'CNY'], description: 'Target currency.' },
      },
      required: ['amount', 'from', 'to'],
    },
  },
  {
    // Nested object + array. Shows the model can be asked for structured input,
    // not just flat scalars — but every field still needs a description, because
    // structure alone does not convey intent (what is `weights` for?).
    name: 'weighted_average',
    description:
      'Compute a weighted average of numbers. Each item has a value and a weight; the result is sum(value*weight)/sum(weight).',
    inputSchema: {
      type: 'object',
      properties: {
        items: {
          type: 'array',
          description: 'List of {value, weight} pairs. Must be non-empty.',
          items: {
            type: 'object',
            properties: {
              value: { type: 'number', description: 'The data point.' },
              weight: { type: 'number', description: 'Its weight; must be > 0.' },
            },
            required: ['value', 'weight'],
          },
        },
      },
      required: ['items'],
    },
  },
];

// Impls. Note these throw on bad input rather than returning a sentinel — the
// loop below turns thrown errors into isError tool_results, which is exactly how
// the model gets a chance to recover (claim #4). A demo rate table keeps the
// stage offline and the numbers honest (no live FX).
const FX_RATES_TO_USD: Record<string, number> = { USD: 1, EUR: 1.08, JPY: 0.0067, CNY: 0.14 };

const TOOL_IMPLS: Record<string, (input: Record<string, unknown>) => Promise<string>> = {
  async get_weather(input) {
    const city = String(input.city ?? '');
    // Tiny fixed table. Unknown city throws — this is the failure we exploit in
    // the recovery demo (model passes a country, gets an error, retries a city).
    const table: Record<string, number> = { tokyo: 18, paris: 12, 'new york': 9 };
    const temp = table[city.trim().toLowerCase()];
    if (temp === undefined) {
      throw new Error(`unknown city "${city}". Known cities: Tokyo, Paris, New York. Pass a city, not a country.`);
    }
    return `${temp}°C`;
  },

  async convert_currency(input) {
    const amount = Number(input.amount);
    const from = String(input.from);
    const to = String(input.to);
    if (!(amount > 0)) throw new Error(`amount must be > 0, got ${input.amount}`);
    const fromRate = FX_RATES_TO_USD[from];
    const toRate = FX_RATES_TO_USD[to];
    if (fromRate === undefined || toRate === undefined) {
      throw new Error(`unsupported currency pair ${from}->${to}`);
    }
    const converted = (amount * fromRate) / toRate;
    return `${amount} ${from} = ${converted.toFixed(2)} ${to}`;
  },

  async weighted_average(input) {
    const items = (input.items ?? []) as Array<{ value: number; weight: number }>;
    if (!Array.isArray(items) || items.length === 0) throw new Error('items must be a non-empty array');
    let weightedSum = 0;
    let weightTotal = 0;
    for (const it of items) {
      if (!(Number(it.weight) > 0)) throw new Error(`weight must be > 0, got ${it.weight}`);
      weightedSum += Number(it.value) * Number(it.weight);
      weightTotal += Number(it.weight);
    }
    return (weightedSum / weightTotal).toFixed(4);
  },
};

// ---------------------------------------------------------------------------
// Self-contained agent loop. Same shape as stage01's runAgent, but instrumented:
// it invokes a `trace` callback at each round so the demos can print the wire
// envelope and the recovery without the loop knowing what's being demonstrated.
// ---------------------------------------------------------------------------

interface RunTrace {
  // Called once per assistant turn, before tools run. Lets a demo inspect the
  // raw assistant blocks (e.g. to print a tool_use envelope).
  onAssistant?(turnIndex: number, turn: AssistantTurn): void;
  // Called once per assistant turn after all tools ran, with the exact
  // tool_result blocks about to be sent back as one user turn.
  onToolResults?(turnIndex: number, results: UserBlock[]): void;
}

interface RunResult {
  answer: string;
  turns: number;
  stopReason: 'answered' | 'max_turns';
}

async function runAgent(
  llm: LLM,
  tools: ToolSpec[],
  toolImpls: Record<string, (input: Record<string, unknown>) => Promise<string>>,
  userInput: string,
  system: string,
  maxTurns: number,
  trace: RunTrace = {}
): Promise<RunResult> {
  const messages: Message[] = [{ role: 'user', content: userInput }];

  for (let turn = 0; turn < maxTurns; turn++) {
    const assistant = await llm.generate({ system, messages, tools });
    trace.onAssistant?.(turn, assistant);
    messages.push({ role: 'assistant', content: assistant.content });

    if (assistant.stopReason !== 'tool_use') {
      return { answer: textOf(assistant.content), turns: turn + 1, stopReason: 'answered' };
    }

    // The invariant (claim #3): collect a tool_result for EVERY tool_use block
    // in this turn, then push them as ONE user message. The API correlates by
    // id, so order does not matter, but completeness does — a missing
    // toolUseId makes the next turn malformed. Running them sequentially here is
    // just for readable output; production code would Promise.all with a
    // concurrency cap (see arch-runtime: resource limits).
    const results: UserBlock[] = [];
    for (const block of assistant.content) {
      if (block.type !== 'tool_use') continue;
      const impl = toolImpls[block.name];
      if (!impl) {
        // Hallucinated tool name: report, don't crash. Same recovery path as a
        // thrown error — the model decides what to do once it sees isError.
        results.push({ type: 'tool_result', toolUseId: block.id, content: `unknown tool: ${block.name}`, isError: true });
        continue;
      }
      try {
        results.push({ type: 'tool_result', toolUseId: block.id, content: await impl(block.input) });
      } catch (err) {
        // Errors are data. Putting the message in `content` with isError lets the
        // model read WHY it failed and adjust — the whole point of claim #4.
        results.push({ type: 'tool_result', toolUseId: block.id, content: (err as Error).message, isError: true });
      }
    }
    trace.onToolResults?.(turn, results);
    messages.push({ role: 'user', content: results });
  }

  return { answer: '[stopped: max turns reached]', turns: maxTurns, stopReason: 'max_turns' };
}

function textOf(content: AssistantBlock[]): string {
  return content
    .filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('');
}

// Small helper: pull the last user turn's tool_result blocks so a mock policy
// can read what the tools returned and decide its next move. This is exactly how
// a real model "sees" tool output — it arrives as the previous user turn.
function lastToolResults(messages: Message[]): Array<Extract<UserBlock, { type: 'tool_result' }>> {
  const last = messages[messages.length - 1];
  if (!last || typeof last.content === 'string') return [];
  return last.content.filter((b): b is Extract<UserBlock, { type: 'tool_result' }> => b.type === 'tool_result');
}

// ---------------------------------------------------------------------------
// Demo A — wire envelope + a single tool round-trip.
// One tool_use, one tool_result. We print both raw objects so the reader sees
// the id correlation that everything else depends on.
// ---------------------------------------------------------------------------

function makeSingleCallMock(): MockLLM {
  return new MockLLM((opts, turnIndex) => {
    if (turnIndex === 0) {
      return {
        content: [
          { type: 'text', text: 'Let me check the temperature.' },
          { type: 'tool_use', id: 'toolu_weather_1', name: 'get_weather', input: { city: 'Tokyo' } },
        ],
      };
    }
    const temp = lastToolResults(opts.messages)[0]?.content ?? 'unknown';
    return { content: [{ type: 'text', text: `It is currently ${temp} in Tokyo.` }], stopReason: 'end_turn' };
  });
}

async function demoSingleCall(): Promise<void> {
  console.log('\n--- Demo A: the wire envelope (one tool round-trip) ---\n');
  const llm = createLLM(makeSingleCallMock());

  await runAgent(
    llm,
    TOOLS,
    TOOL_IMPLS,
    'What is the temperature in Tokyo?',
    'You are a precise assistant. Use tools for facts you cannot know.',
    4,
    {
      onAssistant(_turnIndex, turn) {
        const toolUse = turn.content.find((b) => b.type === 'tool_use');
        if (toolUse) {
          console.log('assistant emitted this tool_use block (raw wire object):');
          console.log(JSON.stringify(toolUse, null, 2));
        }
      },
      onToolResults(_turnIndex, results) {
        console.log('\nharness feeds back this tool_result block (raw wire object):');
        console.log(JSON.stringify(results[0], null, 2));
        console.log(
          `\nnote: tool_result.toolUseId "${results[0].type === 'tool_result' ? results[0].toolUseId : ''}" ` +
            'echoes the tool_use.id above. That id IS the correlation — content carries no other link.'
        );
      },
    }
  );
}

// ---------------------------------------------------------------------------
// Demo B — parallel tool_use. The model fans out two independent calls in ONE
// turn; the harness must answer both in ONE user turn. We assert that invariant
// in code so the demo fails loudly if the loop ever regresses.
// ---------------------------------------------------------------------------

function makeParallelMock(): MockLLM {
  return new MockLLM((opts, turnIndex) => {
    if (turnIndex === 0) {
      // Two calls with NO data dependency between them => the model is free to
      // issue both at once. Serializing independent calls would just burn turns.
      return {
        content: [
          { type: 'text', text: 'I will check both at once.' },
          { type: 'tool_use', id: 'toolu_par_weather', name: 'get_weather', input: { city: 'Paris' } },
          { type: 'tool_use', id: 'toolu_par_fx', name: 'convert_currency', input: { amount: 100, from: 'USD', to: 'EUR' } },
        ],
      };
    }
    const results = lastToolResults(opts.messages);
    const weather = results.find((r) => r.toolUseId === 'toolu_par_weather')?.content ?? '?';
    const fx = results.find((r) => r.toolUseId === 'toolu_par_fx')?.content ?? '?';
    return {
      content: [{ type: 'text', text: `Paris is ${weather}; ${fx}.` }],
      stopReason: 'end_turn',
    };
  });
}

async function demoParallel(): Promise<void> {
  console.log('\n\n--- Demo B: parallel tool_use (one turn, two calls) ---\n');
  const llm = createLLM(makeParallelMock());

  const result = await runAgent(
    llm,
    TOOLS,
    TOOL_IMPLS,
    'How warm is Paris, and what is 100 USD in EUR?',
    'You are a precise assistant. Issue independent tool calls in parallel when you can.',
    4,
    {
      onAssistant(turnIndex, turn) {
        const calls = turn.content.filter((b) => b.type === 'tool_use');
        if (calls.length > 1) {
          console.log(`turn ${turnIndex}: assistant emitted ${calls.length} tool_use blocks in a single turn:`);
          for (const c of calls) if (c.type === 'tool_use') console.log(`  - ${c.name}(${JSON.stringify(c.input)})  id=${c.id}`);
        }
      },
      onToolResults(_turnIndex, results) {
        // The invariant, checked in code (claim #3). The previous assistant turn
        // had 2 tool_use blocks; this user turn MUST carry 2 tool_results with
        // matching ids, or a real API rejects the request.
        const ids = results.map((r) => (r.type === 'tool_result' ? r.toolUseId : '')).sort();
        const expected = ['toolu_par_fx', 'toolu_par_weather'];
        const ok = ids.length === expected.length && ids.every((id, i) => id === expected[i]);
        console.log(
          `\nharness replies with ${results.length} tool_result blocks in ONE user turn (ids: ${ids.join(', ')})`
        );
        console.log(`invariant "every tool_use answered in same user turn": ${ok ? 'HELD ✓' : 'VIOLATED ✗'}`);
        if (!ok) throw new Error('parallel tool_result invariant violated — would be rejected by the API');
      },
    }
  );
  console.log('final answer:', result.answer);
}

// ---------------------------------------------------------------------------
// Demo C — error feedback + recovery. We force a tool failure (model passes a
// country to get_weather, which only knows cities), feed it back as isError,
// and the model reads the message and retries with a valid city. This is the
// failure-mode demo the chapter requires: not happy path.
// ---------------------------------------------------------------------------

function makeRecoveryMock(): MockLLM {
  return new MockLLM((opts, turnIndex) => {
    if (turnIndex === 0) {
      // Deliberately wrong: "France" is a country, not a city. The tool throws,
      // the harness returns isError. A model that ignored errors would loop or
      // give up here; a good one reads the message.
      return {
        content: [
          { type: 'text', text: 'Checking the weather in France.' },
          { type: 'tool_use', id: 'toolu_rec_1', name: 'get_weather', input: { city: 'France' } },
        ],
      };
    }
    if (turnIndex === 1) {
      const failed = lastToolResults(opts.messages)[0];
      // The recovery decision is driven by the actual error content — exactly
      // what a real model does. The mock keys off isError to prove the signal
      // is what matters, not lucky string matching.
      if (failed?.isError) {
        return {
          content: [
            { type: 'text', text: 'That failed because France is a country. Retrying with its capital, Paris.' },
            { type: 'tool_use', id: 'toolu_rec_2', name: 'get_weather', input: { city: 'Paris' } },
          ],
        };
      }
    }
    const temp = lastToolResults(opts.messages)[0]?.content ?? 'unknown';
    return { content: [{ type: 'text', text: `Paris is ${temp}.` }], stopReason: 'end_turn' };
  });
}

async function demoRecovery(): Promise<void> {
  console.log('\n\n--- Demo C: error feedback and recovery (failure mode) ---\n');
  const llm = createLLM(makeRecoveryMock());

  let attempts = 0;
  let errorsSeen = 0;
  const result = await runAgent(
    llm,
    TOOLS,
    TOOL_IMPLS,
    'What is the weather in France?',
    'You are a precise assistant. If a tool errors, read the message and adjust your input.',
    5,
    {
      onAssistant(turnIndex, turn) {
        for (const b of turn.content) {
          if (b.type === 'tool_use') {
            attempts += 1;
            console.log(`turn ${turnIndex}: call #${attempts} -> ${b.name}(${JSON.stringify(b.input)})`);
          } else if (b.type === 'text') {
            console.log(`turn ${turnIndex}: model says: "${b.text}"`);
          }
        }
      },
      onToolResults(turnIndex, results) {
        for (const r of results) {
          if (r.type !== 'tool_result') continue;
          if (r.isError) {
            errorsSeen += 1;
            console.log(`turn ${turnIndex}: tool_result isError=true -> "${r.content}"`);
          } else {
            console.log(`turn ${turnIndex}: tool_result ok -> "${r.content}"`);
          }
        }
      },
    }
  );

  console.log('\nfinal answer:', result.answer);
  // Honest numbers: these are counted from the trace above, not asserted by hand.
  console.log(
    `recovery summary: ${attempts} tool calls, ${errorsSeen} returned isError, ` +
      `recovered in ${result.turns} turns (stop: ${result.stopReason}).`
  );
  if (errorsSeen === 0) {
    // Guard: if this ever prints, the demo is no longer demonstrating recovery.
    console.log('WARNING: no error was observed — this demo is meant to surface one.');
  }
}

// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const probe = createLLM(makeSingleCallMock());
  console.log(`=== Stage 02: tools / function calling (llm=${probe.name}) ===`);
  console.log(`tools registered: ${TOOLS.map((t) => t.name).join(', ')}`);

  await demoSingleCall();
  await demoParallel();
  await demoRecovery();

  console.log('\n\ntakeaway: the schema is the prompt; the id is the wire; parallel calls');
  console.log('must be answered together; and an error is just another tool_result the');
  console.log('model can recover from.\n');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
