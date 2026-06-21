// Stage 14 — observability: you cannot debug an agent you cannot see.
//
// An agent loop is a chain of decisions ("which tool?") and actions ("run it").
// When the final answer is wrong, the bug is almost never in the *last* step —
// it is some earlier decision that quietly went sideways and every step after
// it faithfully built on the mistake. A printed answer tells you the loop is
// wrong; only a per-step trace tells you *where* it went wrong.
//
// So this chapter wraps the loop in a structured tracer. Every step emits one
// span — {step, type, tool, tokens, durationMs, ok} — and the spans together
// form a timeline you can read top-to-bottom like a stack trace for time.
//
// FIELD NAMING follows the OpenTelemetry GenAI semantic conventions so these
// spans drop into a real tracing backend without renaming. The mapping is in
// `toOtelSpan` below; we keep an ergonomic internal shape and convert at the
// edge (functional core / imperative shell), rather than litter the loop with
// dotted attribute keys.
//
// Run it: `npx tsx src/stage14-tracing.ts` (fully offline; no key, no network).
//
// HONESTY NOTE on the numbers: token counts come from the MockLLM's usage field,
// which uses `estimateTokens` (~4 chars/token, not a real BPE tokenizer — see
// core/llm.ts). durationMs is wall-clock around each step; under the offline
// mock these are sub-millisecond and dominated by scheduling noise, so the
// demo also runs with an INJECTED latency model to make the timeline legible.
// Both are labelled where they print. Step counts and the "blamed step" are
// exact — derived from the real recorded spans, not narrated.

import { MockLLM } from './core/llm.js';
import type { AssistantBlock, LLM, Message, ToolSpec, UserBlock } from './core/types.js';

// ============================================================================
// The span. One per loop step.
//
// `type` distinguishes the two things a loop does each iteration, mirroring the
// OTel GenAI split between an inference span (the model deciding) and a tool
// execution span (the harness acting). Keeping them as separate spans is the
// whole point: a run can be slow because the *model* is slow or because a
// *tool* is slow, and a merged "step" span cannot tell you which.
// ============================================================================

type SpanType = 'decision' | 'tool';

interface Span {
  step: number; // monotonic index into the run; the x-axis of the timeline
  type: SpanType;
  tool?: string; // set only for type==='tool' (which tool ran)
  tokens: number; // decision: inputTokens billed; tool: result payload tokens
  durationMs: number; // wall-clock for this step (see HONESTY NOTE re: mock)
  ok: boolean; // false = this step is a candidate root cause; see findBlame()
  detail?: string; // human note for the timeline (error text / tool name / answer head)
}

// A Tracer is just an append-only span buffer plus a clock. It is deliberately
// NOT global state: the loop is handed a tracer so two runs (happy + buggy)
// keep separate, comparable timelines instead of bleeding into one. The clock
// is injected so a test could feed a deterministic timeline (purity at the
// boundary), and so the demo can inject a latency model without touching the
// loop. Default clock is real wall time.
class Tracer {
  private readonly spans: Span[] = [];

  constructor(private readonly now: () => number = () => performance.now()) {}

  // Time a step and record it as one span. The callback returns the span's
  // payload fields; start/end timing is the tracer's job, so the loop never
  // touches a clock (every step is timed identically — no drift between how
  // decision spans and tool spans measure duration).
  async span(
    step: number,
    type: SpanType,
    run: () => Promise<Omit<Span, 'step' | 'type' | 'durationMs'>>
  ): Promise<Span> {
    const t0 = this.now();
    const partial = await run();
    const durationMs = this.now() - t0;
    const span: Span = { step, type, durationMs, ...partial };
    this.spans.push(span);
    return span;
  }

  getSpans(): readonly Span[] {
    return this.spans;
  }
}

// ============================================================================
// The traced loop. Same shape as stage01's runAgent, but every model call and
// every tool call is wrapped in tracer.span(...). The trace is a side product:
// remove the tracer and the loop still works — observability must never change
// behaviour, only record it.
// ============================================================================

interface TracedResult {
  answer: string;
  spans: readonly Span[];
  stopReason: 'answered' | 'max_turns';
}

async function runTracedAgent(
  llm: LLM,
  tools: ToolSpec[],
  toolImpls: Record<string, (input: Record<string, unknown>) => Promise<string>>,
  userInput: string,
  config: { system: string; maxTurns: number },
  tracer: Tracer
): Promise<TracedResult> {
  const messages: Message[] = [{ role: 'user', content: userInput }];
  let step = 0;

  for (let turn = 0; turn < config.maxTurns; turn++) {
    // --- Decision span: the model turns the transcript into the next move. ---
    let assistant!: Awaited<ReturnType<LLM['generate']>>;
    await tracer.span(step++, 'decision', async () => {
      assistant = await llm.generate({ system: config.system, messages, tools });
      const wantsTool = assistant.content.some((b) => b.type === 'tool_use');
      return {
        // For a decision, "tokens" = input tokens billed this turn. This grows
        // turn over turn (whole transcript resent) — the trace makes that cost
        // curve visible per step, not just as a final total (cf. stage01).
        tokens: assistant.usage.inputTokens,
        // A decision is "ok" as long as the model produced a parseable move.
        // Whether it chose the RIGHT move is judged at the tool step / answer.
        ok: assistant.content.length > 0,
        detail: wantsTool
          ? `→ call ${assistant.content.filter((b) => b.type === 'tool_use').map((b) => (b as { name: string }).name).join(', ')}`
          : `→ answer: ${headOf(textOf(assistant.content))}`,
      };
    });
    messages.push({ role: 'assistant', content: assistant.content });

    // Normal exit: model is done. Not an error — the loop's happy path.
    if (assistant.stopReason !== 'tool_use') {
      return { answer: textOf(assistant.content), spans: tracer.getSpans(), stopReason: 'answered' };
    }

    // --- Tool spans: one per requested tool. ---
    const results: UserBlock[] = [];
    for (const block of assistant.content) {
      if (block.type !== 'tool_use') continue;
      await tracer.span(step++, 'tool', async () => {
        const impl = toolImpls[block.name];
        if (!impl) {
          // Failure mode #1: hallucinated tool name. Record ok:false so the
          // timeline flags this step, but feed the error back so the model can
          // recover (do not crash the loop — cf. stage01/stage02).
          results.push({ type: 'tool_result', toolUseId: block.id, content: `unknown tool: ${block.name}`, isError: true });
          return { tool: block.name, tokens: 0, ok: false, detail: `unknown tool: ${block.name}` };
        }
        try {
          const out = await impl(block.input);
          results.push({ type: 'tool_result', toolUseId: block.id, content: out });
          // tokens here = size of what we feed BACK into context. A tool that
          // returns a megabyte is a context-budget problem the trace surfaces.
          return { tool: block.name, tokens: estimatePayloadTokens(out), ok: true, detail: headOf(out) };
        } catch (err) {
          // Failure mode #2: the tool itself threw. ok:false, error preserved
          // in detail AND fed back to the model as an isError result.
          const msg = (err as Error).message;
          results.push({ type: 'tool_result', toolUseId: block.id, content: `tool error: ${msg}`, isError: true });
          return { tool: block.name, tokens: 0, ok: false, detail: `threw: ${msg}` };
        }
      });
    }
    messages.push({ role: 'user', content: results });
  }

  // Failure mode #3: never converged. The trace shows maxTurns decision spans
  // with no terminal answer span — the shape of a runaway loop.
  return { answer: '[stopped: max turns reached]', spans: tracer.getSpans(), stopReason: 'max_turns' };
}

// --- Small pure helpers (copied locally; do NOT import stageNN, they self-run). ---

function textOf(content: AssistantBlock[]): string {
  return content.filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text').map((b) => b.text).join('');
}

// First ~48 chars for a one-line timeline cell; full payload stays in context.
function headOf(s: string): string {
  const oneLine = s.replace(/\s+/g, ' ').trim();
  return oneLine.length > 48 ? oneLine.slice(0, 47) + '…' : oneLine;
}

// Mirror core/llm.ts's ~4-chars-per-token estimate for tool-result payloads.
// Local copy so this file never imports a stage; estimateTokens itself lives in
// core and could be imported, but inlining keeps the HONESTY NOTE self-contained.
function estimatePayloadTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// ============================================================================
// Reading the trace: summary + timeline + blame.
// ============================================================================

interface TraceSummary {
  totalTokens: number;
  totalDurationMs: number;
  toolCalls: number;
  failedSteps: number;
}

// Every number here is summed from the recorded spans — no narration.
function summarize(spans: readonly Span[]): TraceSummary {
  return {
    totalTokens: spans.reduce((n, s) => n + s.tokens, 0),
    totalDurationMs: spans.reduce((n, s) => n + s.durationMs, 0),
    toolCalls: spans.filter((s) => s.type === 'tool').length,
    failedSteps: spans.filter((s) => !s.ok).length,
  };
}

// The point of the whole chapter: in a long trace, find the step that "led the
// agent astray". Heuristic = first span with ok:false. WHY first-failing and
// not last: failures cascade. The model calls a bad tool (step 3, ok:false),
// gets an error back, then flails for five more steps. The LAST failure is a
// symptom; the FIRST is the root cause. Returns null for a clean run.
//
// This is a heuristic, not ground truth — a "successful" tool can still return
// a wrong-but-well-formed answer that ok:true misses (a model-quality bug, not
// a mechanical one). The trace narrows the search; it does not replace judgment.
function findBlame(spans: readonly Span[]): Span | null {
  return spans.find((s) => !s.ok) ?? null;
}

// --- Rendering --------------------------------------------------------------

function renderTimeline(spans: readonly Span[]): string {
  const blame = findBlame(spans);
  const lines = spans.map((s) => {
    const marker = !s.ok ? '✗' : '✓';
    const blamePtr = blame && s.step === blame.step ? '  ⟵ first failure (root-cause candidate)' : '';
    const kind = s.type === 'decision' ? 'decision' : `tool:${s.tool ?? '?'}`;
    // Fixed-width columns so a long trace stays scannable.
    return (
      `  ${marker} step ${String(s.step).padStart(2)} ` +
      `| ${kind.padEnd(18)} ` +
      `| ${String(s.tokens).padStart(5)} tok ` +
      `| ${s.durationMs.toFixed(1).padStart(6)} ms ` +
      `| ${s.detail ?? ''}${blamePtr}`
    );
  });
  return lines.join('\n');
}

// ============================================================================
// OTel GenAI mapping. We keep the internal shape ergonomic and convert at the
// edge. Attribute keys below are the OpenTelemetry GenAI semantic conventions
// (gen_ai.* namespace); a real exporter would attach these to OTLP spans so the
// trace shows up in Jaeger/Tempo/Honeycomb with no per-vendor glue.
//   - decision span  → span name "chat <model>",   gen_ai.operation.name=chat
//   - tool span      → span name "execute_tool X", gen_ai.operation.name=execute_tool
// `error.type` is the OTel convention for marking a span as failed.
// ============================================================================

function toOtelSpan(s: Span, modelName: string): Record<string, unknown> {
  const common = {
    'gen_ai.system': modelName, // e.g. "mock" or "anthropic"
    'duration_ms': Number(s.durationMs.toFixed(2)),
    ...(s.ok ? {} : { 'error.type': 'step_failed' }),
  };
  if (s.type === 'decision') {
    return {
      name: `chat ${modelName}`,
      'gen_ai.operation.name': 'chat',
      'gen_ai.usage.input_tokens': s.tokens, // OTel: prompt tokens for this call
      ...common,
    };
  }
  return {
    name: `execute_tool ${s.tool}`,
    'gen_ai.operation.name': 'execute_tool',
    'gen_ai.tool.name': s.tool,
    'gen_ai.usage.output_tokens': s.tokens, // tool payload returned to context
    ...common,
  };
}

// ============================================================================
// Tools. A tiny "research" toolset where order matters: you must look_up a fact
// before you can cite it. This ordering is what lets a buggy run go wrong in a
// way the trace can pinpoint.
// ============================================================================

const TOOLS: ToolSpec[] = [
  {
    name: 'look_up',
    description: 'Look up a fact by key from the knowledge base. Returns the fact text.',
    inputSchema: { type: 'object', properties: { key: { type: 'string' } }, required: ['key'] },
  },
  {
    name: 'cite',
    description: 'Format a looked-up fact as a citation. Requires the exact fact text from look_up.',
    inputSchema: { type: 'object', properties: { fact: { type: 'string' } }, required: ['fact'] },
  },
];

const KB: Record<string, string> = {
  'speed-of-light': '299792458 m/s in vacuum',
};

const TOOL_IMPLS: Record<string, (input: Record<string, unknown>) => Promise<string>> = {
  async look_up(input) {
    const key = String(input.key ?? '');
    const fact = KB[key];
    // Failure mode demo: unknown key throws. The traced loop turns this into an
    // ok:false tool span + an isError result the model sees.
    if (fact === undefined) throw new Error(`no fact for key "${key}"`);
    return fact;
  },
  async cite(input) {
    const fact = String(input.fact ?? '');
    if (!fact.trim()) throw new Error('cite called with empty fact (was look_up skipped?)');
    return `[1] ${fact}`;
  },
};

// ============================================================================
// Mock policies. Two scripted runs sharing the same loop and tracer machinery.
// ============================================================================

// HAPPY: look_up "speed-of-light" → cite the result → answer. Three turns.
function makeHappyMock(): LLM {
  return new MockLLM((opts, turnIndex) => {
    if (turnIndex === 0) {
      return {
        content: [
          { type: 'text', text: '先查事实。' },
          { type: 'tool_use', id: 'c1', name: 'look_up', input: { key: 'speed-of-light' } },
        ],
      };
    }
    if (turnIndex === 1) {
      const fact = lastToolResult(opts.messages);
      return {
        content: [{ type: 'tool_use', id: 'c2', name: 'cite', input: { fact } }],
      };
    }
    const citation = lastToolResult(opts.messages);
    return { content: [{ type: 'text', text: `光速：${citation}` }], stopReason: 'end_turn' };
  });
}

// BUGGY: the model gets the KEY WRONG on turn 0 (asks for "lightspeed", which is
// not in the KB). look_up throws → ok:false at step 1. The model, seeing the
// error, then tries to cite an empty fact (turn 1) → cite throws → ok:false at
// step 3 → and only THEN gives up. The trace will show TWO failures; findBlame
// must point at step 1 (the wrong key), not step 3 (the downstream symptom).
// This is the chapter's core lesson made mechanical.
function makeBuggyMock(): LLM {
  return new MockLLM((_opts, turnIndex) => {
    if (turnIndex === 0) {
      return {
        content: [
          { type: 'text', text: '查一下光速。' },
          // The bug: wrong key. Everything downstream is a faithful build on it.
          { type: 'tool_use', id: 'c1', name: 'look_up', input: { key: 'lightspeed' } },
        ],
      };
    }
    if (turnIndex === 1) {
      // Model saw "no fact for key" but soldiers on and tries to cite nothing —
      // the classic "kept going on a broken premise" pattern.
      return { content: [{ type: 'tool_use', id: 'c2', name: 'cite', input: { fact: '' } }] };
    }
    return {
      content: [{ type: 'text', text: '抱歉，我没能查到光速。' }],
      stopReason: 'end_turn',
    };
  });
}

function lastToolResult(messages: Message[]): string {
  const last = messages[messages.length - 1];
  if (typeof last.content === 'string') return '';
  const block = last.content.find((b) => b.type === 'tool_result');
  return block && block.type === 'tool_result' ? block.content : '';
}

// ============================================================================
// Latency model. The offline mock runs in microseconds, so real durations are
// noise. To make the timeline legible we wrap the tracer's clock with a fake
// monotonic clock that advances by a per-step-type budget. This is LABELLED as
// injected, not measured — the mechanism (per-step spans, summing, blame) is
// identical to production; only the clock source differs.
// ============================================================================

function makeInjectedClock(): () => number {
  let t = 0;
  let call = 0;
  // Decision steps "cost" more than tool steps, matching real agents where the
  // model call dominates wall time. Values are illustrative, not benchmarks.
  const advances = [120, 8, 140, 6, 95]; // ms, consumed in step order
  // The tracer calls now() exactly twice per span: once for t0 (start), once
  // for the end. We advance the clock by the step's budget BEFORE the 2nd call
  // of each pair, so end - start == budget. Odd-numbered calls (1,3,5…) are the
  // "end" of a span; advance there.
  return () => {
    if (call % 2 === 1) t += advances[Math.floor(call / 2) % advances.length];
    call++;
    return t;
  };
}

// ============================================================================

async function main(): Promise<void> {
  console.log('\n=== Stage 14: structured tracing for agent loops ===');
  console.log('(durations are an INJECTED latency model — see HONESTY NOTE in source header)\n');

  // --- Run 1: the happy path. -----------------------------------------------
  const happyTracer = new Tracer(makeInjectedClock());
  const happyLlm = makeHappyMock();
  const happy = await runTracedAgent(
    happyLlm,
    TOOLS,
    TOOL_IMPLS,
    'What is the speed of light? Cite the source.',
    { system: 'Look up facts before citing them.', maxTurns: 6 },
    happyTracer
  );

  console.log('── Run 1: HAPPY ──────────────────────────────────────────');
  console.log('answer:', happy.answer, `(stop: ${happy.stopReason})`);
  console.log('timeline:');
  console.log(renderTimeline(happy.spans));
  const hs = summarize(happy.spans);
  console.log(
    `summary: ${happy.spans.length} steps | ${hs.toolCalls} tool calls | ` +
      `${hs.totalTokens} tokens | ${hs.totalDurationMs.toFixed(1)} ms | ${hs.failedSteps} failed`
  );
  const hBlame = findBlame(happy.spans);
  console.log('blame :', hBlame ? `step ${hBlame.step}` : 'none — clean run');

  // --- Run 2: the buggy path. -----------------------------------------------
  const buggyTracer = new Tracer(makeInjectedClock());
  const buggyLlm = makeBuggyMock();
  const buggy = await runTracedAgent(
    buggyLlm,
    TOOLS,
    TOOL_IMPLS,
    'What is the speed of light? Cite the source.',
    { system: 'Look up facts before citing them.', maxTurns: 6 },
    buggyTracer
  );

  console.log('\n── Run 2: BUGGY (wrong key on step 0 cascades) ───────────');
  console.log('answer:', buggy.answer, `(stop: ${buggy.stopReason})`);
  console.log('timeline:');
  console.log(renderTimeline(buggy.spans));
  const bs = summarize(buggy.spans);
  console.log(
    `summary: ${buggy.spans.length} steps | ${bs.toolCalls} tool calls | ` +
      `${bs.totalTokens} tokens | ${bs.totalDurationMs.toFixed(1)} ms | ${bs.failedSteps} failed`
  );
  const bBlame = findBlame(buggy.spans);
  console.log(
    'blame :',
    bBlame ? `step ${bBlame.step} (${bBlame.type}${bBlame.tool ? ':' + bBlame.tool : ''}) — "${bBlame.detail}"` : 'none'
  );
  // Prove the lesson with a real number, not a claim: there are >1 failed steps
  // but blame points at the FIRST. If this assertion's premise were false the
  // demo would be lying about cascade behaviour.
  console.log(
    `lesson: ${bs.failedSteps} steps failed, but the root cause is the FIRST (step ${bBlame?.step}); ` +
      `steps after it are downstream symptoms.`
  );

  // --- OTel export sample. --------------------------------------------------
  console.log('\n── OTel GenAI export (first 2 spans of buggy run) ────────');
  for (const s of buggy.spans.slice(0, 2)) {
    console.log(JSON.stringify(toOtelSpan(s, buggyLlm.name)));
  }
  console.log();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
