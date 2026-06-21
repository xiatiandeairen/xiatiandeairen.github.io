// Stage 11 — multi-agent patterns and their cost.
//
// "Multi-agent" sounds like more horsepower. The bill says otherwise. The cost
// of an agent is the SUM of input tokens across its turns, and the whole
// transcript is re-sent every turn (stage01's O(T^2) curve). Splitting one job
// across N agents does NOT split that cost — it MULTIPLIES it, because every
// sub-agent re-pays for its own system prompt, tool specs, and a copy of the
// shared task. This chapter builds two real coordination patterns and then puts
// a price tag on them against a single agent doing the same job.
//
//   (A) handoff      — one agent transfers control to a specialist (Swarm idea).
//                      Cheap when the routing decision is genuinely needed.
//   (B) parallel fan-out — split work to N workers, then a synthesizer merges.
//                      Buys wall-clock latency; pays for it in tokens, and the
//                      synthesizer turn re-ingests every worker's output.
//
// Then three failure modes you will actually hit:
//   (1) coordination failure — handoff routes to a dead end, nobody answers.
//   (2) context desync       — worker B contradicts worker A because neither
//                              saw the other's output; the merge is incoherent.
//   (3) cost blow-up         — the token multiple vs a single agent, computed.
//
// Run it: `npx tsx src/stage11-multiagent.ts` (fully offline; no key, no network).
//
// HONESTY NOTE on the numbers: every token count comes from estimateTokens
// (core/llm.ts), a ~4-chars-per-token heuristic — NOT a real BPE tokenizer, and
// especially wrong for CJK. All the "Nx more expensive" multiples below are real
// arithmetic on that one estimator applied identically to both arms, so the
// RATIO is meaningful even though the absolute token counts are approximate.

import { MockLLM } from './core/llm.js';
import type { AssistantBlock, GenerateOptions, Message } from './core/types.js';

// ----------------------------------------------------------------------------
// Cost accounting. We do not reuse runAgent from stage01 (importing a stageNN
// file runs its main()); instead we price conversations directly. The unit that
// matters for cost is INPUT tokens: a model is billed for everything it reads,
// and a long-running / multi-agent system re-reads the same prefix over and over.
// ----------------------------------------------------------------------------

// Tokens the model reads on ONE generate() call: system + tool specs + every
// message so far. This mirrors estimateMessagesTokens in core/llm.ts (we can't
// import that private helper, so the formula is duplicated here and kept in sync
// with it — same ~4-chars/token rule, same JSON-serialization of blocks).
function inputTokensOf(opts: GenerateOptions): number {
  let chars = (opts.system ?? '').length;
  for (const m of opts.messages) {
    if (typeof m.content === 'string') chars += m.content.length;
    else for (const b of m.content) chars += JSON.stringify(b).length;
  }
  for (const t of opts.tools ?? []) chars += JSON.stringify(t).length;
  return Math.ceil(chars / 4);
}

// One agent's lifetime cost = sum of input tokens it pays on every turn, plus
// the output tokens it emits. We track input separately because that is where
// the multi-agent tax lands: each agent re-pays its own prefix.
interface AgentCost {
  label: string;
  inputTokens: number;
  outputTokens: number;
  turns: number;
}

function totalTokens(costs: AgentCost[]): number {
  return costs.reduce((n, c) => n + c.inputTokens + c.outputTokens, 0);
}

// ----------------------------------------------------------------------------
// The shared task. A support ticket that needs (a) a refund decision and (b) a
// shipping-policy answer. One agent can do both; the multi-agent versions split
// it. Keeping the task identical across arms is the whole point — otherwise the
// cost comparison is rigged.
// ----------------------------------------------------------------------------

const TASK =
  'Customer ticket: "My order #4471 arrived broken. I want a refund AND I need ' +
  'to know if you ship replacements to Canada." Decide the refund and answer the ' +
  'shipping question.';

// A non-trivial system prompt: in the real world this is hundreds of tokens of
// persona, policy, and tool docs. Every agent that runs pays for its own copy —
// that duplication is the hidden cost the printout below makes visible.
const SUPPORT_SYSTEM =
  'You are a support agent for a hardware store. Refund policy: damaged-on-arrival ' +
  'items are always refunded in full within 30 days, no manager approval needed. ' +
  'Shipping policy: we ship replacements to US and Canada; Canada adds 5-7 business ' +
  'days. Always cite the order number. Be concise and decisive.';

// ============================================================================
// ARM 1 — SINGLE AGENT. One model, one transcript, answers the whole ticket in
// one turn. This is the baseline every multi-agent design must beat to justify
// its complexity.
// ============================================================================

function makeSingleAgent(): MockLLM {
  return new MockLLM((_opts, _turn) => ({
    content: [
      {
        type: 'text',
        text:
          'Order #4471: damaged on arrival → full refund approved (within 30-day ' +
          'window, no approval needed). Replacements DO ship to Canada, +5-7 business ' +
          'days. I have issued the refund and can dispatch a replacement to Canada.',
      },
    ],
    stopReason: 'end_turn',
  }));
}

async function runSingleAgent(): Promise<{ answer: string; costs: AgentCost[] }> {
  const llm = makeSingleAgent();
  const opts: GenerateOptions = {
    system: SUPPORT_SYSTEM,
    messages: [{ role: 'user', content: TASK }],
  };
  const inputTokens = inputTokensOf(opts);
  const turn = await llm.generate(opts);
  return {
    answer: textOf(turn.content),
    costs: [
      {
        label: 'solo',
        inputTokens,
        outputTokens: turn.usage.outputTokens,
        turns: 1,
      },
    ],
  };
}

// ============================================================================
// ARM 2 — HANDOFF (OpenAI Swarm idea). A cheap router agent looks at the ticket
// and TRANSFERS control to a specialist by emitting a transfer "tool call". The
// specialist then runs with its OWN system prompt and re-reads the task.
//
// Why this is the right pattern: routing is a small, well-bounded decision; you
// do not want your refund specialist's prompt bloated with shipping policy and
// vice versa. Handoff keeps each agent's context narrow.
//
// Why it still costs more than solo: the task text is paid for TWICE (router
// reads it, specialist reads it again), and you now run two system prompts. The
// win is quality/separation, not tokens — the printout proves the multiple.
// ============================================================================

const ROUTER_SYSTEM =
  'You are a triage router. Read the ticket and transfer it to exactly one ' +
  'specialist: "refund_agent" or "shipping_agent". Do not answer the ticket ' +
  'yourself. Emit a transfer.';

const REFUND_SYSTEM =
  'You are the refund specialist. Apply the damaged-on-arrival policy and decide ' +
  'the refund for the cited order. Refunds for damaged items are automatic within ' +
  '30 days.';

// Swarm-style transfer: the router does not produce prose, it requests a handoff
// the way it would request any tool. The orchestrator reads the requested target
// and dispatches. Modeling transfer AS a tool_use is the core Swarm insight —
// control flow rides the same channel as tool calls, so no special API is needed.
function makeRouter(target: string): MockLLM {
  return new MockLLM((_opts, _turn) => ({
    content: [
      { type: 'text', text: `Routing ticket to ${target}.` },
      { type: 'tool_use', id: 'xfer_1', name: 'transfer', input: { to: target } },
    ],
    stopReason: 'tool_use',
  }));
}

function makeRefundSpecialist(): MockLLM {
  return new MockLLM((_opts, _turn) => ({
    content: [
      {
        type: 'text',
        text:
          'Order #4471: damaged on arrival, within 30 days → full refund issued ' +
          'automatically, no manager approval required.',
      },
    ],
    stopReason: 'end_turn',
  }));
}

// The orchestrator. It runs the router, reads the transfer target, and — IF the
// target is a real agent — runs that specialist on the SAME task. Returns the
// per-agent costs so the caller can sum the multi-agent tax.
//
// `availableSpecialists` is what lets us demo failure mode (1): a router that
// transfers to an agent that does not exist routes the ticket into a black hole.
async function runHandoff(
  availableSpecialists: Record<string, () => MockLLM>,
  routeTo: string
): Promise<{ answer: string; costs: AgentCost[]; routedTo: string }> {
  const costs: AgentCost[] = [];

  // Step 1: router turn.
  const routerLlm = makeRouter(routeTo);
  const routerOpts: GenerateOptions = {
    system: ROUTER_SYSTEM,
    messages: [{ role: 'user', content: TASK }],
  };
  const routerInput = inputTokensOf(routerOpts);
  const routerTurn = await routerLlm.generate(routerOpts);
  costs.push({ label: 'router', inputTokens: routerInput, outputTokens: routerTurn.usage.outputTokens, turns: 1 });

  // Read the requested handoff target from the transfer tool_use block.
  const transfer = routerTurn.content.find(
    (b): b is Extract<AssistantBlock, { type: 'tool_use' }> => b.type === 'tool_use' && b.name === 'transfer'
  );
  const target = (transfer?.input.to as string | undefined) ?? '(none)';

  // FAILURE MODE (1) — coordination failure. The router named a specialist the
  // orchestrator cannot dispatch to. We do NOT silently swallow this; the loop
  // would otherwise just end with no answer and look "successful". Surface it.
  const makeSpecialist = availableSpecialists[target];
  if (!makeSpecialist) {
    return {
      answer: `[coordination failure] router transferred to unknown agent "${target}"; ticket dropped`,
      costs,
      routedTo: target,
    };
  }

  // Step 2: specialist turn. Note it RE-READS the task with its own system
  // prompt — this is the duplicated cost handoff cannot avoid.
  const specialistLlm = makeSpecialist();
  const specialistOpts: GenerateOptions = {
    system: REFUND_SYSTEM,
    messages: [{ role: 'user', content: TASK }],
  };
  const specialistInput = inputTokensOf(specialistOpts);
  const specialistTurn = await specialistLlm.generate(specialistOpts);
  costs.push({ label: `specialist:${target}`, inputTokens: specialistInput, outputTokens: specialistTurn.usage.outputTokens, turns: 1 });

  return { answer: textOf(specialistTurn.content), costs, routedTo: target };
}

// ============================================================================
// ARM 3 — PARALLEL FAN-OUT + SYNTHESIS. The orchestrator splits the ticket into
// independent sub-questions, runs one worker per sub-question (conceptually in
// parallel — Promise.all), then a synthesizer agent merges the worker outputs
// into one answer.
//
// What you buy: wall-clock latency (workers run concurrently). What you pay:
//   - each worker re-pays its own system prompt + its slice of the task, and
//   - the synthesizer's input = its prompt + EVERY worker's output concatenated.
// The synthesizer turn is where fan-out cost quietly explodes: more workers →
// fatter synthesizer input. The printout computes this.
// ============================================================================

const WORKER_SYSTEM =
  'You are a focused worker. Answer ONLY the single sub-question you are given, ' +
  'using the store policies. Do not address anything else.';

const SYNTH_SYSTEM =
  'You are the synthesizer. Combine the worker answers below into one coherent ' +
  'reply to the customer. Resolve any contradictions; do not contradict yourself.';

interface SubTask {
  id: string;
  question: string;
  // Scripted worker reply. In production this is a real model call; scripting it
  // keeps the demo deterministic AND lets us inject a contradiction for failure
  // mode (2) without any randomness.
  reply: string;
}

// Run one worker on one sub-question. Each call is an independent agent with the
// worker system prompt and NO visibility into sibling workers — that isolation
// is exactly what causes context desync (failure mode 2).
async function runWorker(sub: SubTask): Promise<{ answer: string; cost: AgentCost }> {
  const llm = new MockLLM(() => ({ content: [{ type: 'text', text: sub.reply }], stopReason: 'end_turn' }));
  const opts: GenerateOptions = {
    system: WORKER_SYSTEM,
    messages: [{ role: 'user', content: sub.question }],
  };
  const inputTokens = inputTokensOf(opts);
  const turn = await llm.generate(opts);
  return { answer: textOf(turn.content), cost: { label: `worker:${sub.id}`, inputTokens, outputTokens: turn.usage.outputTokens, turns: 1 } };
}

async function runFanOut(
  subTasks: SubTask[]
): Promise<{ answer: string; costs: AgentCost[]; workerAnswers: string[] }> {
  // Workers run concurrently — this is the latency win the pattern is sold on.
  // (Cost is unaffected by concurrency: you pay the same tokens, just sooner.)
  const workerResults = await Promise.all(subTasks.map(runWorker));
  const costs = workerResults.map((r) => r.cost);
  const workerAnswers = workerResults.map((r) => r.answer);

  // Synthesizer re-ingests ALL worker outputs. Its input grows with worker count
  // and output length — the part of fan-out cost that does not parallelize away.
  const merged: Message[] = [
    { role: 'user', content: TASK },
    ...workerAnswers.map((a, i): Message => ({ role: 'user', content: `Worker ${subTasks[i].id} says: ${a}` })),
  ];
  const synthLlm = makeSynthesizer(workerAnswers);
  const synthOpts: GenerateOptions = { system: SYNTH_SYSTEM, messages: merged };
  const synthInput = inputTokensOf(synthOpts);
  const synthTurn = await synthLlm.generate(synthOpts);
  costs.push({ label: 'synthesizer', inputTokens: synthInput, outputTokens: synthTurn.usage.outputTokens, turns: 1 });

  return { answer: textOf(synthTurn.content), costs, workerAnswers };
}

// The synthesizer is scripted to do the ONE thing a synthesizer must do: detect
// when workers disagree. If two worker answers carry opposite verdicts on the
// same fact, a real synthesizer cannot reconcile them (it has no ground truth) —
// so the honest output is to flag the conflict, not to fabricate a resolution.
// This is what powers failure mode (2): garbage in (contradictory workers),
// garbage (or at best a flagged conflict) out.
function makeSynthesizer(workerAnswers: string[]): MockLLM {
  const conflict = detectContradiction(workerAnswers);
  return new MockLLM(() => {
    if (conflict) {
      return {
        content: [
          {
            type: 'text',
            text:
              `[context desync] workers disagree: "${conflict.a}" vs "${conflict.b}". ` +
              'Neither worker saw the other, so this merge cannot be trusted. ' +
              'Escalating instead of guessing.',
          },
        ],
        stopReason: 'end_turn',
      };
    }
    return {
      content: [{ type: 'text', text: `Order #4471: ${workerAnswers.join(' ')}` }],
      stopReason: 'end_turn',
    };
  });
}

// Cheap, mechanical contradiction check: does one worker say "ship to Canada"
// while another says "do NOT ship to Canada"? This is a toy detector (string
// match on a known axis), NOT a general contradiction engine — it exists only to
// make failure mode (2) observable and deterministic in the demo.
function detectContradiction(answers: string[]): { a: string; b: string } | null {
  const shipsYes = answers.find((a) => /\bship(s|ping)?\b[^.]*\bCanada\b/i.test(a) && !/\bnot\b|\bno\b|\bdo not\b|\bcannot\b/i.test(a));
  const shipsNo = answers.find((a) => /\bCanada\b/i.test(a) && /\bnot\b|\bno\b|\bdo not\b|\bcannot\b/i.test(a));
  if (shipsYes && shipsNo) return { a: shipsYes.trim(), b: shipsNo.trim() };
  return null;
}

// ----------------------------------------------------------------------------

function textOf(content: AssistantBlock[]): string {
  return content
    .filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('');
}

// Quality gate: did the arm actually answer BOTH parts of the ticket (refund +
// Canada shipping)? Token cost is meaningless without this — a cheap arm that
// fails to answer is not a bargain. Mechanical check on the two load-bearing
// facts the task demands.
function answeredBoth(answer: string): boolean {
  const hasRefund = /refund/i.test(answer);
  const hasShipping = /Canada/i.test(answer);
  const isFailure = /\[(coordination failure|context desync)\]/.test(answer);
  return hasRefund && hasShipping && !isFailure;
}

function printCostTable(costs: AgentCost[]): void {
  for (const c of costs) {
    console.log(
      `  ${c.label.padEnd(20)} in=${String(c.inputTokens).padStart(4)} ` +
        `out=${String(c.outputTokens).padStart(4)} tok`
    );
  }
  console.log(`  ${'TOTAL'.padEnd(20)} ${String(totalTokens(costs)).padStart(13)} tok`);
}

async function main(): Promise<void> {
  console.log('\n=== Stage 11: multi-agent patterns and cost (llm=mock) ===\n');

  // --- ARM 1: single agent (the baseline to beat). ------------------------
  const solo = await runSingleAgent();
  console.log('[ARM 1] SINGLE AGENT');
  console.log('  answer    :', solo.answer);
  console.log('  answered both parts:', answeredBoth(solo.answer));
  printCostTable(solo.costs);
  const soloTotal = totalTokens(solo.costs);

  // --- ARM 2: handoff, happy path (router → refund specialist). -----------
  console.log('\n[ARM 2] HANDOFF (router -> specialist), happy path');
  const handoff = await runHandoff({ refund_agent: makeRefundSpecialist }, 'refund_agent');
  console.log('  routed to :', handoff.routedTo);
  console.log('  answer    :', handoff.answer);
  // The specialist answered the refund but NOT the shipping question — a real
  // gap of single-track handoff: the narrow specialist drops the other half.
  console.log('  answered both parts:', answeredBoth(handoff.answer), '(note: refund only — shipping half lost)');
  printCostTable(handoff.costs);
  const handoffTotal = totalTokens(handoff.costs);

  // --- ARM 3: parallel fan-out + synthesis, happy path. -------------------
  console.log('\n[ARM 3] PARALLEL FAN-OUT + SYNTHESIS, happy path');
  const fanOut = await runFanOut([
    { id: 'refund', question: 'Should order #4471 (damaged on arrival) be refunded?', reply: 'Damaged on arrival within 30 days → full refund, automatic.' },
    { id: 'ship', question: 'Do we ship replacements to Canada?', reply: 'Yes, we ship replacements to Canada, +5-7 business days.' },
  ]);
  console.log('  answer    :', fanOut.answer);
  console.log('  answered both parts:', answeredBoth(fanOut.answer));
  printCostTable(fanOut.costs);
  const fanOutTotal = totalTokens(fanOut.costs);

  // --- COST COMPARISON: the honest multiple. ------------------------------
  console.log('\n[COST] total tokens vs single agent (lower is cheaper)');
  console.log(`  single agent      : ${String(soloTotal).padStart(4)} tok  (1.00x baseline)`);
  console.log(`  handoff           : ${String(handoffTotal).padStart(4)} tok  (${(handoffTotal / soloTotal).toFixed(2)}x)`);
  console.log(`  fan-out + synth   : ${String(fanOutTotal).padStart(4)} tok  (${(fanOutTotal / soloTotal).toFixed(2)}x)`);
  console.log('  → both multi-agent arms cost MORE for this task. The single agent');
  console.log('    answered both parts in one turn; the multi-agent tax (duplicated');
  console.log('    system prompts + re-read task + synthesizer re-ingest) bought nothing.');

  // ========================================================================
  // FAILURE MODES. Each is a real way multi-agent systems break that a single
  // agent simply cannot exhibit.
  // ========================================================================
  console.log('\n=== FAILURE MODES ===');

  // (1) COORDINATION FAILURE — router transfers to an agent that isn't wired up.
  console.log('\n[FAIL 1] coordination failure: router transfers to a missing specialist');
  const dropped = await runHandoff({ refund_agent: makeRefundSpecialist }, 'billing_agent');
  console.log('  routed to :', dropped.routedTo, '(not in dispatch table)');
  console.log('  answer    :', dropped.answer);
  console.log('  answered both parts:', answeredBoth(dropped.answer));
  console.log('  cost paid for ZERO useful output:', totalTokens(dropped.costs), 'tok (router ran, ticket dropped)');

  // (2) CONTEXT DESYNC — two workers given the SAME shipping question reach
  // opposite conclusions because neither saw the other. The synthesizer cannot
  // reconcile contradictory facts it has no ground truth for.
  console.log('\n[FAIL 2] context desync: isolated workers contradict each other');
  const desync = await runFanOut([
    { id: 'ship-a', question: 'Do we ship replacements to Canada?', reply: 'Yes, we ship replacements to Canada.' },
    { id: 'ship-b', question: 'Do we ship replacements to Canada?', reply: 'No, we do not ship to Canada.' },
  ]);
  console.log('  worker A  :', desync.workerAnswers[0]);
  console.log('  worker B  :', desync.workerAnswers[1]);
  console.log('  merged    :', desync.answer);
  console.log('  answered both parts:', answeredBoth(desync.answer), '(merge is untrustworthy)');

  // (3) COST BLOW-UP — scale fan-out to many workers and watch the synthesizer
  // input balloon. Token multiple grows super-linearly in worker count because
  // the synthesizer re-reads ALL of them. We compute the real multiple here.
  console.log('\n[FAIL 3] cost blow-up: token multiple vs worker count (synthesizer re-ingests all)');
  for (const n of [1, 2, 4, 8]) {
    const subs: SubTask[] = Array.from({ length: n }, (_, i) => ({
      id: `q${i}`,
      question: `Sub-question ${i}: clarify store policy detail #${i} for order #4471 in full.`,
      reply: `Policy detail #${i} resolved for order #4471: refer to the damaged-on-arrival and Canada-shipping clauses as written.`,
    }));
    const r = await runFanOut(subs);
    const synthInput = r.costs.find((c) => c.label === 'synthesizer')!.inputTokens;
    const total = totalTokens(r.costs);
    console.log(
      `  workers=${n}: total=${String(total).padStart(4)} tok  ` +
        `(${(total / soloTotal).toFixed(2)}x solo), synthesizer input=${String(synthInput).padStart(4)} tok`
    );
  }
  console.log('  → synthesizer input grows with every worker; the fan-out total');
  console.log('    outpaces the single agent fast. Add workers only when the latency');
  console.log('    win is worth a roughly linear token premium PLUS the merge risk above.\n');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
