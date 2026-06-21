// Stage 13 — evaluating an agent.
//
// You cannot improve what you cannot measure, and an agent is uniquely hard to
// measure: the output is open-ended text, the model is nondeterministic, and the
// same final answer can be reached by a correct path or a lucky one. This chapter
// builds the smallest honest eval harness and then shows the failure mode that
// silently destroys most home-grown agent evals: a SATURATING RUBRIC — a judge so
// lenient that every run scores full marks, so the metric stops discriminating
// good from bad and you "improve" forever against a flat signal.
//
// Two kinds of eval, both implemented here for real (no model judges the score —
// the grading is mechanical and deterministic, which is the whole point):
//
//   (1) OUTCOME eval     — did the final answer match a machine-checkable expected
//                          result? This is the pass-rate everyone quotes.
//   (2) TRAJECTORY eval  — did the agent take a sane PATH (call the tool it had to
//                          call)? An agent can get the right answer by guessing and
//                          still be broken; outcome eval alone misses this.
//
// Run it: `npx tsx src/stage13-eval.ts` (fully offline; no key, no network).
//
// HONESTY NOTE: pass rates and trajectory counts below are computed from the real
// task runs — nothing is hard-coded. The MockLLM policies are deterministic
// scripts that stand in for a model, so the SAME harness would grade a real model;
// only the thing being graded is scripted (so the demo is reproducible offline).

import { MockLLM } from './core/llm.js';
import type {
  AssistantBlock,
  LLM,
  Message,
  ToolSpec,
  UserBlock,
} from './core/types.js';

// ============================================================================
// A self-contained agent loop.
//
// NOTE: this is a trimmed copy of stage01's `runAgent`. We do NOT import the
// stage01 file because every stageNN module runs its own `main()` on import,
// which would execute stage01's demo as a side effect. The loop is the subject
// of stage01; here it is just the system-under-test, so we also record the
// trajectory (which tools were called, in order) — trajectory eval (2) needs it.
// ============================================================================

interface AgentRun {
  answer: string;
  // The ordered list of tool names the agent actually invoked across the run.
  // This is the raw material for trajectory eval — outcome eval ignores it.
  toolCalls: string[];
  turns: number;
}

async function runAgent(
  llm: LLM,
  tools: ToolSpec[],
  toolImpls: Record<string, (input: Record<string, unknown>) => Promise<string>>,
  userInput: string,
  maxTurns: number
): Promise<AgentRun> {
  const messages: Message[] = [{ role: 'user', content: userInput }];
  const toolCalls: string[] = [];

  for (let turn = 0; turn < maxTurns; turn++) {
    const assistant = await llm.generate({ messages, tools });
    messages.push({ role: 'assistant', content: assistant.content });

    if (assistant.stopReason !== 'tool_use') {
      return { answer: textOf(assistant.content), toolCalls, turns: turn + 1 };
    }

    const results: UserBlock[] = [];
    for (const block of assistant.content) {
      if (block.type !== 'tool_use') continue;
      toolCalls.push(block.name); // record trajectory before running the tool
      const impl = toolImpls[block.name];
      if (!impl) {
        results.push({ type: 'tool_result', toolUseId: block.id, content: `unknown tool: ${block.name}`, isError: true });
        continue;
      }
      try {
        results.push({ type: 'tool_result', toolUseId: block.id, content: await impl(block.input) });
      } catch (err) {
        results.push({ type: 'tool_result', toolUseId: block.id, content: `tool error: ${(err as Error).message}`, isError: true });
      }
    }
    messages.push({ role: 'user', content: results });
  }
  // Ran out of turns: return a marker, not an exception — the eval should grade
  // this as a failure, not crash the whole suite (one bad task must not abort it).
  return { answer: '[stopped: max turns reached]', toolCalls, turns: maxTurns };
}

function textOf(content: AssistantBlock[]): string {
  return content
    .filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('');
}

// ============================================================================
// The system under test: a tiny calculator agent with one tool.
//
// We deliberately give it ONE real tool (`add`) so trajectory eval has something
// to assert: a correct agent for "what is X + Y" MUST call `add`. An agent that
// answers the sum from its own head is fragile (it will be wrong on big numbers)
// even when it happens to be right on small ones — exactly the gap outcome eval
// cannot see but trajectory eval can.
// ============================================================================

const TOOLS: ToolSpec[] = [
  {
    name: 'add',
    description: 'Add two integers and return their exact sum.',
    inputSchema: {
      type: 'object',
      properties: { a: { type: 'number' }, b: { type: 'number' } },
      required: ['a', 'b'],
    },
  },
];

const TOOL_IMPLS: Record<string, (input: Record<string, unknown>) => Promise<string>> = {
  async add(input) {
    return String(Number(input.a) + Number(input.b));
  },
};

// ============================================================================
// The fixed task set.
//
// An eval is only as good as its tasks. Each task carries a MACHINE-CHECKABLE
// expectation — a concrete number to find in the answer — so grading needs no
// human and no model. `requiresTool` encodes the trajectory expectation: the path
// that a non-fragile agent must take. Keeping tasks fixed (a frozen set) is what
// makes scores comparable across agent versions; a moving task set hides
// regressions.
// ============================================================================

interface EvalTask {
  id: string;
  prompt: string;
  expectedSum: number; // the exact answer a correct agent must produce
  requiresTool: string; // trajectory: the tool a correct path must invoke
}

const TASKS: EvalTask[] = [
  { id: 'small', prompt: 'What is 2 + 3?', expectedSum: 5, requiresTool: 'add' },
  { id: 'medium', prompt: 'What is 40 + 2?', expectedSum: 42, requiresTool: 'add' },
  // Large operands: a model answering from its head is most likely to slip here.
  { id: 'large', prompt: 'What is 123456 + 654321?', expectedSum: 777777, requiresTool: 'add' },
];

// ============================================================================
// Two agents to grade, so the eval has something to discriminate BETWEEN.
//
// goodAgent: always calls `add`, returns the tool's exact result. Correct path,
// correct answer on every task.
//
// guessAgent: NEVER calls a tool; it answers the sum from a scripted lookup that
// is right on the easy tasks and WRONG on the large one. This is the agent a
// good eval must catch — and the saturating rubric below will fail to.
// ============================================================================

function makeGoodAgent(): LLM {
  return new MockLLM((opts, turnIndex) => {
    if (turnIndex === 0) {
      const { a, b } = parseOperands(firstUserText(opts.messages));
      return { content: [{ type: 'tool_use', id: 'call_1', name: 'add', input: { a, b } }] };
    }
    const sum = lastToolResult(opts.messages);
    return { content: [{ type: 'text', text: `The sum is ${sum}.` }], stopReason: 'end_turn' };
  });
}

function makeGuessAgent(): LLM {
  return new MockLLM((opts) => {
    // No tool call ever. Mental arithmetic that breaks on large numbers — it
    // "carries wrong" and is off by 100000 on the large task. The point: a
    // plausible, confident, wrong answer with no tool use.
    const { a, b } = parseOperands(firstUserText(opts.messages));
    const guessed = a + b >= 700000 ? a + b - 100000 : a + b;
    return { content: [{ type: 'text', text: `The sum is ${guessed}.` }], stopReason: 'end_turn' };
  });
}

// --- Mock-policy helpers (pure parsing of the scripted conversation). --------

function firstUserText(messages: Message[]): string {
  const first = messages[0];
  return typeof first.content === 'string' ? first.content : '';
}

function parseOperands(text: string): { a: number; b: number } {
  const nums = text.match(/\d+/g)?.map(Number) ?? [];
  return { a: nums[0] ?? 0, b: nums[1] ?? 0 };
}

function lastToolResult(messages: Message[]): string {
  const last = messages[messages.length - 1];
  if (typeof last.content === 'string') return '';
  const block = last.content.find((b) => b.type === 'tool_result');
  return block && block.type === 'tool_result' ? block.content : '';
}

// ============================================================================
// (1) OUTCOME EVAL — pass rate against the machine-checkable expectation.
// ============================================================================

interface OutcomeResult {
  taskId: string;
  pass: boolean;
  answer: string;
}

// A task PASSES iff the exact expected integer appears as a standalone number in
// the answer. Substring matching would let "777" pass "expected 777777", so we
// match on word boundaries — a small but load-bearing strictness choice (a lax
// matcher is the first step toward a saturating rubric).
function gradeOutcome(task: EvalTask, run: AgentRun): OutcomeResult {
  const pattern = new RegExp(`\\b${task.expectedSum}\\b`);
  return { taskId: task.id, pass: pattern.test(run.answer), answer: run.answer };
}

// ============================================================================
// (2) TRAJECTORY EVAL — did the agent take the required path?
//
// Independent of outcome: an agent can pass outcome (right number) yet fail
// trajectory (never called the tool, got lucky). That divergence is the single
// most useful thing trajectory eval surfaces — it is your early warning that a
// passing pass-rate is built on luck and will regress on harder inputs.
// ============================================================================

interface TrajectoryResult {
  taskId: string;
  pass: boolean;
  requiredTool: string;
  called: boolean;
}

function gradeTrajectory(task: EvalTask, run: AgentRun): TrajectoryResult {
  const called = run.toolCalls.includes(task.requiresTool);
  return { taskId: task.id, pass: called, requiredTool: task.requiresTool, called };
}

// ============================================================================
// The harness: run every task once, grade both ways, aggregate. Real numbers.
// ============================================================================

interface SuiteReport {
  outcomes: OutcomeResult[];
  trajectories: TrajectoryResult[];
  outcomePassRate: number; // computed, not hard-coded
  trajectoryPassRate: number;
}

// Takes a FACTORY, not an LLM instance. MockLLM counts turns per instance
// (turnIndex), and each task is a fresh conversation that must restart at turn 0;
// reusing one instance across tasks leaks turn state and breaks every task after
// the first. This mirrors a real harness invariant: each eval task runs in an
// isolated agent session, never sharing state with a previous task.
async function runSuite(makeAgent: () => LLM): Promise<SuiteReport> {
  const outcomes: OutcomeResult[] = [];
  const trajectories: TrajectoryResult[] = [];
  for (const task of TASKS) {
    const run = await runAgent(makeAgent(), TOOLS, TOOL_IMPLS, task.prompt, 6);
    outcomes.push(gradeOutcome(task, run));
    trajectories.push(gradeTrajectory(task, run));
  }
  return {
    outcomes,
    trajectories,
    outcomePassRate: outcomes.filter((o) => o.pass).length / outcomes.length,
    trajectoryPassRate: trajectories.filter((t) => t.pass).length / trajectories.length,
  };
}

// ============================================================================
// THE FAILURE MODE: a saturating rubric.
//
// Below are two GRADERS for the very same agent runs. The "lenient" grader is the
// kind teams reach for first ("does the answer look like a helpful response?").
// It checks only that the agent produced a confident, on-topic sentence — which
// EVERY run does, including the wrong ones. Result: 100% on a broken agent. The
// metric is saturated: it cannot tell good from bad, so any further "tuning"
// chases noise. The strict grader (gradeOutcome, above) restores discrimination.
// ============================================================================

// Lenient rubric: passes if the answer is a fluent sentence mentioning a sum.
// This is intentionally the WRONG check — it never reads the actual number, so a
// confidently-wrong answer sails through.
function gradeLenient(_task: EvalTask, run: AgentRun): boolean {
  const a = run.answer;
  return /sum is/i.test(a) && /\d/.test(a) && a.trim().endsWith('.');
}

async function lenientPassRate(makeAgent: () => LLM): Promise<number> {
  let pass = 0;
  for (const task of TASKS) {
    const run = await runAgent(makeAgent(), TOOLS, TOOL_IMPLS, task.prompt, 6);
    if (gradeLenient(task, run)) pass += 1;
  }
  return pass / TASKS.length;
}

// --- Reporting. -------------------------------------------------------------

function pct(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

function printSuite(label: string, report: SuiteReport): void {
  console.log(`\n--- ${label} ---`);
  for (const o of report.outcomes) {
    const traj = report.trajectories.find((t) => t.taskId === o.taskId)!;
    console.log(
      `  [${o.pass ? 'PASS' : 'FAIL'}] ${o.taskId.padEnd(7)} ` +
        `outcome=${o.pass ? 'ok ' : 'bad'} ` +
        `trajectory=${traj.called ? 'called add' : 'NO tool!'} ` +
        `| answer: ${o.answer}`
    );
  }
  console.log(`  outcome    pass rate: ${pct(report.outcomePassRate)} (${report.outcomes.filter((o) => o.pass).length}/${report.outcomes.length})`);
  console.log(`  trajectory pass rate: ${pct(report.trajectoryPassRate)} (${report.trajectories.filter((t) => t.pass).length}/${report.trajectories.length})`);
}

async function main(): Promise<void> {
  console.log('\n=== Stage 13: evaluating an agent ===');
  console.log(`tasks: ${TASKS.length} fixed, machine-checkable | grading: deterministic (no model judge)`);

  // Grade both agents with the honest (strict outcome + trajectory) harness.
  const goodReport = await runSuite(makeGoodAgent);
  const guessReport = await runSuite(makeGuessAgent);
  printSuite('goodAgent  (always calls add)', goodReport);
  printSuite('guessAgent (never calls a tool, mental math)', guessReport);

  // The lesson: outcome alone vs outcome + trajectory.
  console.log('\n=== outcome eval is not enough ===');
  console.log(
    `guessAgent outcome pass rate ${pct(guessReport.outcomePassRate)} looks "mostly fine",`
  );
  console.log(
    `but its trajectory pass rate is ${pct(guessReport.trajectoryPassRate)} — it never used the tool.`
  );
  console.log(
    'The one outcome failure (large task) is the tip: the agent is guessing on ALL tasks,'
  );
  console.log('it just got lucky on the easy ones. Trajectory eval flags that on every task.');

  // The failure mode: a saturating rubric scores the broken agent perfect.
  const goodLenient = await lenientPassRate(makeGoodAgent);
  const guessLenient = await lenientPassRate(makeGuessAgent);
  console.log('\n=== the saturating-rubric trap ===');
  console.log(`lenient rubric ("answer looks like a fluent sum sentence"):`);
  console.log(`  goodAgent : ${pct(goodLenient)}`);
  console.log(`  guessAgent: ${pct(guessLenient)}   <-- broken agent, still PERFECT`);
  console.log(
    'Both score 100% — the rubric is SATURATED, it cannot separate a correct agent'
  );
  console.log('from a guessing one. A metric that gives everyone full marks measures nothing.');
  console.log(`\ndiscriminating rubric (strict outcome) on the SAME runs:`);
  console.log(`  goodAgent : ${pct(goodReport.outcomePassRate)}`);
  console.log(`  guessAgent: ${pct(guessReport.outcomePassRate)}   <-- now they differ; the eval has signal`);
  console.log(
    `\nrubric discrimination (gap between best and worst agent): ` +
      `saturated=${pct(Math.abs(goodLenient - guessLenient))}  ` +
      `strict=${pct(Math.abs(goodReport.outcomePassRate - guessReport.outcomePassRate))}`
  );
  console.log('A useful rubric MAXIMIZES that gap; a saturated one collapses it to 0%.\n');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
