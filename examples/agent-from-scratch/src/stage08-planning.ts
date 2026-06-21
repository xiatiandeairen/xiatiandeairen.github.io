// Stage 08 — planning & reasoning: ReAct vs Plan-and-Execute.
//
// Two ways to spend an agent's "thinking":
//
//   (1) ReAct (reason+act)        — decide ONLY the next action each turn, look
//                                    at the result, decide again. No upfront
//                                    plan. Cheap to start, adapts for free, but
//                                    has no global view (can wander).
//   (2) Plan-and-Execute          — first emit a full PLAN (a visible todo
//                                    list), then execute steps against it; when
//                                    a step fails, REPLAN the tail. The plan is
//                                    a real artifact the user can read/edit —
//                                    this is what tools like TodoWrite surface.
//
// The thesis of this chapter: the plan is *data*, not vibes. Making it an
// explicit data structure (Todo[]) is what lets us (a) show it to the user,
// (b) detect that a step failed, and (c) replan deterministically. Most of the
// file is that data structure and the two control loops around it; the "model"
// is a scripted MockLLM so the comparison is reproducible offline.
//
// Run it: `npx tsx src/stage08-planning.ts` (fully offline; no key, no network).
//
// HONESTY NOTE on the numbers: "steps" and "llmCalls" are counted by the loops
// below (real). "success" is a real check against the world state the tools
// mutate. The MockLLM is scripted — a real model would produce these plans and
// reactions itself, but the control flow and the failure modes are exactly the
// ones a real model triggers. Where a number is illustrative (the "overplanning
// tax") it is labelled at the print site.

import { MockLLM } from './core/llm.js';
import type { AssistantBlock, GenerateOptions, Message, ToolSpec, UserBlock } from './core/types.js';

// ============================================================================
// The world. Both modes act on the same tiny mutable world so "did it succeed"
// is a real check, not a vibe. The task: produce a release note that requires
// (a) reading the changelog, (b) bumping the version, (c) writing the note.
// One tool intermittently fails the first time to force a replan (see below).
// ============================================================================

interface World {
  changelogRead: boolean;
  version: string;
  releaseNote: string | null;
  // bump_version fails the FIRST time it is called and succeeds on retry. This
  // is the canonical "transient step failure" that distinguishes the two modes:
  // ReAct just reacts to the error; Plan-and-Execute must decide replan vs
  // retry. Modeled as a counter so the failure is deterministic, not random
  // (random would make the offline run non-reproducible — see HONESTY NOTE).
  bumpAttempts: number;
}

function freshWorld(): World {
  return { changelogRead: false, version: '1.0.0', releaseNote: null, bumpAttempts: 0 };
}

// The goal predicate. A run "succeeded" iff the world satisfies this — checked
// against state the tools actually mutated, so a mode cannot claim success by
// merely saying "done".
function isGoalReached(w: World): boolean {
  return w.changelogRead && w.version === '1.1.0' && w.releaseNote !== null;
}

// --- Tools: spec the model sees + impl that mutates the shared world. --------
// Impls are closures over one World instance so each run is isolated.

const TOOLS: ToolSpec[] = [
  {
    name: 'read_changelog',
    description: 'Read the changelog. Must be done before writing a release note.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'bump_version',
    description: 'Bump the project version to the given value, e.g. "1.1.0".',
    inputSchema: {
      type: 'object',
      properties: { to: { type: 'string', description: 'target version' } },
      required: ['to'],
    },
  },
  {
    name: 'write_release_note',
    description: 'Write the release note text. Requires the changelog to be read first.',
    inputSchema: {
      type: 'object',
      properties: { text: { type: 'string' } },
      required: ['text'],
    },
  },
];

function makeToolImpls(w: World): Record<string, (input: Record<string, unknown>) => string> {
  return {
    read_changelog() {
      w.changelogRead = true;
      return 'changelog: feat X; fix Y';
    },
    bump_version(input) {
      w.bumpAttempts += 1;
      const to = String(input.to ?? '');
      // PERMANENT failure: a malformed target never succeeds, no matter how many
      // times it is retried. This is what makes the oscillation demo genuinely
      // unbounded (a plan that keeps proposing `to:"BAD"` loops forever) and is
      // distinct from the transient lock below — replan/retry can fix transient,
      // never permanent. Must be checked BEFORE the transient counter so a doomed
      // value stays doomed on every attempt.
      if (!/^\d+\.\d+\.\d+$/.test(to)) {
        throw new Error(`invalid version "${to}", expected semver`);
      }
      // Transient failure on the FIRST valid attempt — the canonical retryable
      // error. The impl signals failure by throwing and NOT mutating the version;
      // the loops wrap it into an isError tool_result.
      if (w.bumpAttempts === 1) {
        throw new Error('version registry locked, try again');
      }
      w.version = to;
      return `version is now ${w.version}`;
    },
    write_release_note(input) {
      // Precondition enforced by the tool, not just the prompt: a tool that can
      // be called out of order MUST guard its own invariant, because the model
      // (or a buggy plan) will eventually call it out of order.
      if (!w.changelogRead) throw new Error('cannot write note before reading changelog');
      w.releaseNote = String(input.text ?? '');
      return 'release note written';
    },
  };
}

// Run one tool call, returning the tool_result block the loop feeds back. Shared
// by both modes so the world-mutation path is identical — the modes differ only
// in HOW they decide what to call, not in how calls execute.
function runTool(
  impls: Record<string, (input: Record<string, unknown>) => string>,
  block: Extract<AssistantBlock, { type: 'tool_use' }>
): UserBlock {
  const impl = impls[block.name];
  if (!impl) return { type: 'tool_result', toolUseId: block.id, content: `unknown tool: ${block.name}`, isError: true };
  try {
    return { type: 'tool_result', toolUseId: block.id, content: impl(block.input) };
  } catch (err) {
    return { type: 'tool_result', toolUseId: block.id, content: (err as Error).message, isError: true };
  }
}

// ============================================================================
// The PLAN as data. A Todo list — the same shape a TodoWrite-style tool exposes
// to the user. Making the plan an inspectable array (not hidden model state) is
// what enables: render it, mark a step failed, and replan the tail.
// ============================================================================

type TodoStatus = 'pending' | 'in_progress' | 'done' | 'failed';

interface Todo {
  id: number;
  what: string;
  // The concrete action this step intends. Keeping the action ON the todo (not
  // re-derived each turn) is why Plan-and-Execute can execute without re-asking
  // the model per step — that is its whole efficiency claim vs ReAct.
  action: { tool: string; input: Record<string, unknown> };
  status: TodoStatus;
}

function renderPlan(todos: Todo[]): string {
  const mark: Record<TodoStatus, string> = { pending: '○', in_progress: '◐', done: '●', failed: '✗' };
  return todos.map((t) => `    ${mark[t.status]} [${t.id}] ${t.what} (${t.status})`).join('\n');
}

// Shared result shape so the final comparison table is apples-to-apples.
interface RunResult {
  mode: string;
  steps: number; // tool executions performed (the agent's real "work")
  llmCalls: number; // model invocations (the agent's real "thinking" cost)
  success: boolean; // checked against world state, not self-reported
  replans: number; // Plan-and-Execute only; 0 for ReAct
  note: string;
}

// ============================================================================
// MODE 1 — ReAct. One model call per step. The model sees the running transcript
// (including the last tool_result) and decides the single next action, or stops.
// No plan exists; adaptation is implicit in "look at result, decide again".
// ============================================================================

async function runReActAgent(llm: MockLLM, world: World, maxSteps: number): Promise<RunResult> {
  const impls = makeToolImpls(world);
  const messages: Message[] = [
    { role: 'user', content: 'Cut a 1.1.0 release: read changelog, bump version, write the release note.' },
  ];
  let steps = 0;
  let llmCalls = 0;

  for (let i = 0; i < maxSteps; i++) {
    const turn = await llm.generate({ system: 'ReAct: decide ONLY the next action.', messages, tools: TOOLS });
    llmCalls += 1;
    messages.push({ role: 'assistant', content: turn.content });

    if (turn.stopReason !== 'tool_use') break; // model says done

    const results: UserBlock[] = [];
    for (const b of turn.content) {
      if (b.type !== 'tool_use') continue;
      results.push(runTool(impls, b));
      steps += 1;
    }
    messages.push({ role: 'user', content: results });
  }

  return {
    mode: 'ReAct',
    steps,
    llmCalls,
    success: isGoalReached(world),
    replans: 0,
    note: 'recovers from the transient failure by simply reacting to the error and retrying',
  };
}

// ============================================================================
// MODE 2 — Plan-and-Execute. ONE model call to produce the whole plan, then
// execute steps WITHOUT a model call each — until a step fails. On failure we
// call the model again to REPLAN the tail. So llmCalls = 1 + (number of
// replans), which is the efficiency story: thinking is amortized over steps.
// ============================================================================

async function runPlanAndExecuteAgent(llm: MockLLM, world: World, maxReplans: number): Promise<RunResult> {
  const impls = makeToolImpls(world);

  // (1) PLAN. The model returns a plan as a tool_use of the special "plan" tool
  // whose input is the todo list. We surface it as an artifact immediately.
  const planTurn = await llm.generate({ system: 'Plan-and-Execute: emit a full plan first.', messages: [
    { role: 'user', content: 'Cut a 1.1.0 release: read changelog, bump version, write the release note.' },
  ], tools: TOOLS });
  let llmCalls = 1;
  let todos = extractPlan(planTurn.content);
  console.log('  [plan emitted]');
  console.log(renderPlan(todos));

  let steps = 0;
  let replans = 0;

  // (2) EXECUTE the plan top-to-bottom. Crucially: NO model call per step. The
  // action is read straight off the todo. This is the cost win and the risk: a
  // stale plan executes blindly until something fails.
  for (let i = 0; i < todos.length; i++) {
    const t = todos[i];
    if (t.status !== 'pending') continue;
    t.status = 'in_progress';
    const result = runTool(impls, { type: 'tool_use', id: `s${t.id}`, name: t.action.tool, input: t.action.input });
    steps += 1;

    if (result.type === 'tool_result' && result.isError) {
      t.status = 'failed';
      console.log(`  [step ${t.id} FAILED] ${result.content}`);
      if (replans >= maxReplans) {
        // Guard against replan oscillation (see overplanning/oscillation demo):
        // a fixed replan budget is the single most important reliability knob
        // here, mirroring stage01's maxTurns. Without it a plan that keeps
        // failing→replanning→failing bills forever.
        return { mode: 'Plan-and-Execute', steps, llmCalls, success: isGoalReached(world), replans, note: 'hit replan budget — gave up (oscillation guard)' };
      }

      // (3) REPLAN the tail: ask the model for a new plan for the remaining
      // work, given the failure. We keep done steps, replace the rest.
      replans += 1;
      const replanTurn = await llm.generate({
        system: 'Replan the remaining steps given the failure.',
        messages: [{ role: 'user', content: `Step "${t.what}" failed with: ${result.content}. Replan the rest.` }],
        tools: TOOLS,
      });
      llmCalls += 1;
      const tail = extractPlan(replanTurn.content);
      // Renumber the new tail after the steps already done, splice it in.
      const doneCount = todos.filter((x) => x.status === 'done').length;
      tail.forEach((x, k) => (x.id = doneCount + k + 1));
      todos = [...todos.filter((x) => x.status === 'done'), ...tail];
      console.log(`  [replanned, attempt ${replans}]`);
      console.log(renderPlan(todos));
      i = -1; // restart execution over the new plan from the top of the tail
      continue;
    }

    t.status = 'done';
  }

  return {
    mode: 'Plan-and-Execute',
    steps,
    llmCalls,
    success: isGoalReached(world),
    replans,
    note: '1 plan call + 1 replan call; steps executed without a model call each',
  };
}

// Pull a Todo[] out of the model's "plan" tool_use. A real model would emit this
// as structured tool input; the mock does the same so the parsing path is real.
function extractPlan(content: AssistantBlock[]): Todo[] {
  const planBlock = content.find((b) => b.type === 'tool_use' && b.name === 'plan');
  if (planBlock && planBlock.type === 'tool_use' && Array.isArray(planBlock.input.steps)) {
    return (planBlock.input.steps as Array<{ what: string; tool: string; input: Record<string, unknown> }>).map(
      (s, i) => ({ id: i + 1, what: s.what, action: { tool: s.tool, input: s.input }, status: 'pending' as TodoStatus })
    );
  }
  return [];
}

// ============================================================================
// MockLLM policies. Each is a deterministic script that reproduces the behavior
// a real model would exhibit, so the comparison is stable across runs.
// ============================================================================

// ReAct: read the last tool_result and decide the next action. The key behavior
// is recovery: when it sees the "locked" error from bump_version, it just calls
// bump_version again — no plan, no replan, pure reaction.
function makeReActPolicy(): MockLLM {
  return new MockLLM((opts) => {
    const w = inferReActState(opts.messages);
    if (!w.changelogRead) return toolUse('read_changelog', {});
    if (w.version !== '1.1.0') return toolUse('bump_version', { to: '1.1.0' }); // covers both first-try and post-error retry
    if (w.releaseNote === null) return toolUse('write_release_note', { text: 'Release 1.1.0: X, Y' });
    return { content: [{ type: 'text', text: 'Release cut.' }], stopReason: 'end_turn' as const };
  });
}

// Reconstruct what the world looks like from the transcript alone (the model
// only sees messages, not the World object). This mirrors how a real model
// infers state from tool_results — it has no privileged access to our `World`.
function inferReActState(messages: Message[]): { changelogRead: boolean; version: string; releaseNote: string | null } {
  let changelogRead = false;
  let version = '1.0.0';
  let releaseNote: string | null = null;
  for (const m of messages) {
    if (m.role !== 'user' || typeof m.content === 'string') continue;
    for (const b of m.content) {
      if (b.type !== 'tool_result' || b.isError) continue; // failed calls do not advance state
      if (b.content.startsWith('changelog:')) changelogRead = true;
      if (b.content.startsWith('version is now')) version = b.content.replace('version is now ', '').trim();
      if (b.content === 'release note written') releaseNote = 'written';
    }
  }
  return { changelogRead, version, releaseNote };
}

// Plan-and-Execute: emit a 3-step plan up front; on replan, re-emit only the
// remaining steps (the model is told what failed). The plan is correct, so the
// only "failure" is the transient tool error — exactly what replan is for.
function makePlanPolicy(): MockLLM {
  return new MockLLM((opts) => {
    const isReplan = opts.messages.some(
      (m) => typeof m.content === 'string' && m.content.includes('Replan the rest')
    );
    if (isReplan) {
      // Tail after read_changelog already succeeded: retry bump, then write.
      return plan([
        { what: 'bump version to 1.1.0', tool: 'bump_version', input: { to: '1.1.0' } },
        { what: 'write the release note', tool: 'write_release_note', input: { text: 'Release 1.1.0: X, Y' } },
      ]);
    }
    return plan([
      { what: 'read the changelog', tool: 'read_changelog', input: {} },
      { what: 'bump version to 1.1.0', tool: 'bump_version', input: { to: '1.1.0' } },
      { what: 'write the release note', tool: 'write_release_note', input: { text: 'Release 1.1.0: X, Y' } },
    ]);
  });
}

// --- failure-mode policies (see demoFailureModes) ---------------------------

// Overplanning: a trivial single-step task ("just read the changelog") gets a
// bloated 5-step plan with ceremony. The plan is "correct" but the work is 1
// step — the extra steps are pure overhead the user must read past.
function makeOverplanPolicy(): MockLLM {
  return new MockLLM(() =>
    plan([
      { what: 'analyze the request', tool: 'read_changelog', input: {} },
      { what: 'gather context', tool: 'read_changelog', input: {} },
      { what: 'read the changelog', tool: 'read_changelog', input: {} },
      { what: 'verify the read', tool: 'read_changelog', input: {} },
      { what: 'summarize findings', tool: 'read_changelog', input: {} },
    ])
  );
}

// Replan oscillation: the model keeps emitting the SAME failing plan after each
// failure (it never learns), so without a replan budget it would loop forever.
// The budget in runPlanAndExecuteAgent is what stops it — demonstrated here.
function makeOscillatingPolicy(): MockLLM {
  return new MockLLM(() =>
    // Always plans to bump_version with a value the (broken) registry rejects;
    // the tool never succeeds, so every replan reproduces the same failure.
    plan([{ what: 'bump version (doomed)', tool: 'bump_version', input: { to: 'BAD' } }])
  );
}

// --- tiny builders for policy return values ---------------------------------

function toolUse(name: string, input: Record<string, unknown>) {
  return { content: [{ type: 'tool_use' as const, id: `c_${name}`, name, input }] };
}

function plan(steps: Array<{ what: string; tool: string; input: Record<string, unknown> }>) {
  return { content: [{ type: 'tool_use' as const, id: 'plan_1', name: 'plan', input: { steps } }] };
}

// ============================================================================

async function main(): Promise<void> {
  console.log('\n=== Stage 08: planning — ReAct vs Plan-and-Execute (offline mock) ===\n');

  // Same task, same world rules, two strategies. ----------------------------
  console.log('--- MODE 1: ReAct (decide next action each turn) ---');
  const reactWorld = freshWorld();
  const reactResult = await runReActAgent(makeReActPolicy(), reactWorld, 8);
  console.log(`  ${reactResult.note}\n`);

  console.log('--- MODE 2: Plan-and-Execute (plan first, replan on failure) ---');
  const planWorld = freshWorld();
  const planResult = await runPlanAndExecuteAgent(makePlanPolicy(), planWorld, 2);
  console.log(`  ${planResult.note}\n`);

  // Apples-to-apples comparison on the SAME task. All numbers below are counted
  // by the loops / checked against world state above — none are hardcoded.
  console.log('--- COMPARISON (same task, both succeed) ---');
  const rows = [reactResult, planResult];
  console.log('  mode                 steps  llmCalls  replans  success');
  for (const r of rows) {
    console.log(
      `  ${r.mode.padEnd(20)} ${String(r.steps).padEnd(6)} ${String(r.llmCalls).padEnd(9)} ${String(r.replans).padEnd(8)} ${r.success}`
    );
  }
  console.log('');
  console.log('  Read it: ReAct pays 1 model call PER step (thinking every turn) — here');
  console.log(`    ${reactResult.llmCalls} calls for ${reactResult.steps} steps. Plan-and-Execute amortizes thinking:`);
  console.log(`    ${planResult.llmCalls} calls (1 plan + ${planResult.replans} replan) for ${planResult.steps} steps.`);
  console.log('    Trade-off: ReAct adapts for free but has no global view; Plan-and-Execute');
  console.log('    is cheaper per step and shows a visible plan, but executes blindly until');
  console.log('    a step fails. Neither is "better" — they trade adaptivity for cost.\n');

  await demoFailureModes();
}

// ============================================================================
// FAILURE MODES. Planning is not free; both modes have characteristic ways to
// burn cost or loop. Demo > description.
// ============================================================================

async function demoFailureModes(): Promise<void> {
  console.log('--- FAILURE MODE 1: overplanning (trivial task gets a 5-step plan) ---');
  // The actual task only needs read_changelog. A real read-the-changelog goal
  // is ONE step; the plan inflates it to 5. We count the planned-vs-needed gap.
  const NEEDED_STEPS = 1; // ground truth: this task needs exactly one tool call
  const overWorld = freshWorld();
  const overTurn = await makeOverplanPolicy().generate({ messages: [{ role: 'user', content: 'just read the changelog' }], tools: TOOLS } as GenerateOptions);
  const overPlan = extractPlan(overTurn.content);
  console.log(renderPlan(overPlan));
  const tax = overPlan.length - NEEDED_STEPS;
  console.log(`  planned ${overPlan.length} steps for a ${NEEDED_STEPS}-step task → overplanning tax = ${tax} redundant steps`);
  console.log('    Why it hurts: every redundant step is tokens the user reads + (if executed)');
  console.log('    a tool call. Smarter agents gate planning on task complexity; a fixed');
  console.log('    "always plan" policy pays this tax on every trivial request.');
  void overWorld;
  console.log('');

  console.log('--- FAILURE MODE 2: replan oscillation (same failing plan re-emitted) ---');
  // The model never learns; it re-plans the identical doomed step after every
  // failure. The replan BUDGET is what bounds the damage — without it, infinite.
  const oscWorld = freshWorld();
  const oscResult = await runPlanAndExecuteAgent(makeOscillatingPolicy(), oscWorld, 2);
  console.log(
    `  result: success=${oscResult.success}, replans=${oscResult.replans} (capped by budget), steps=${oscResult.steps}`
  );
  console.log(`  note: ${oscResult.note}`);
  console.log('    Without the replan budget this loop would never terminate — the model');
  console.log('    keeps proposing the same broken step. The budget (here 2) is the agent');
  console.log("    equivalent of stage01's maxTurns: the one knob that guarantees the run ends.\n");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
