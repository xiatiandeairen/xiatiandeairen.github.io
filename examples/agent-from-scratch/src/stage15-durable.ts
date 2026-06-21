// Stage 15 — reliability & durability: an agent that survives a crash.
//
// Every stage so far assumed the process lives long enough to finish. Real
// agents do not get that luxury: they run for minutes against flaky tools, the
// box gets OOM-killed, a deploy rolls the pod, someone hits Ctrl-C. If the only
// copy of "what the agent has done so far" lives in process memory, a crash at
// step 4 of 6 means starting over from step 0 — re-running every tool, re-paying
// every token, and (worse) re-firing every SIDE EFFECT.
//
// Two mechanisms make an agent durable, and they are NOT the same problem:
//
//   1. CHECKPOINTING — persist enough state to disk after each step that a fresh
//      process can pick up where the dead one stopped. The invariant is "the
//      checkpoint is a complete description of progress": whatever is NOT in it is
//      lost on resume.
//
//   2. IDEMPOTENCY — checkpointing alone is not enough. You will ALWAYS have a
//      window where a tool's side effect has fired but the checkpoint recording it
//      has not landed yet (you cannot make "do the effect" and "record the effect"
//      one atomic act across a process boundary). Crash in that window and resume
//      re-runs the step. For a read that is harmless; for "charge the card" it is a
//      double charge. The fix is an idempotency key: the effect records "key K
//      already done → here is the prior result" the instant it fires, so a re-run
//      returns the old result WITHOUT repeating the effect.
//
// The honest framing: checkpoint-after-step gives you AT-LEAST-ONCE execution of
// the in-flight step. You cannot get exactly-once for free — idempotency is how
// you make at-least-once SAFE. That trade-off is the whole lesson, and §DRIFT
// shows what a single missing piece of durable state costs: the ledger that makes
// the retry safe is itself state; lose it and at-least-once becomes double-fire.
//
// Run it: `npx tsx src/stage15-durable.ts`. Fully offline and deterministic.
// State is checkpointed to real temp files (paths printed). The "crash" is a
// thrown error injected at a scripted point; a real SIGKILL/OOM lands in the same
// place — the next process reads the checkpoint and continues.

import { mkdtempSync, readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { MockLLM } from './core/llm.js';
import type { AssistantBlock, LLM, Message, ToolSpec, UserBlock } from './core/types.js';

// ============================================================================
// §EFFECT — the observable, non-undoable side effect we are protecting.
//
// A process-global counter standing in for an external action that cannot be
// taken back (charge a card, send an email, insert a row). We count every ACTUAL
// fire — not every call — so a duplicated effect shows up as a wrong number, the
// only honest way to assert "exactly once".
// ============================================================================

let chargeFireCount = 0;

// ============================================================================
// §IDEMPOTENT-TOOL — a tool whose retry does not duplicate its side effect.
//
// The ledger maps idempotency key -> the result we returned the first time.
// Presence of a key is the ONLY signal "this effect already happened"; we never
// re-derive it. Critically the ledger is DURABLE AT FIRE-TIME (written to its own
// file the instant the effect runs), independent of the agent's step checkpoint —
// because the dangerous window is exactly "effect fired, step checkpoint not yet
// written". If the ledger only lived in that step checkpoint, a crash in the
// window would lose the dedupe record and resume would re-charge (that is §DRIFT).
// ============================================================================

interface IdempotencyLedger {
  [key: string]: string;
}

// `ledgerPath === null` is the DRIFT injection: it makes the ledger live only in
// process memory, so a restart starts with an empty ledger — reproducing a real
// bug where idempotency state was never made durable. Production code has no such
// switch; it always persists.
interface LedgerStore {
  ledger: IdempotencyLedger;
  ledgerPath: string | null;
}

function loadLedger(ledgerPath: string | null): LedgerStore {
  if (ledgerPath && existsSync(ledgerPath)) {
    return { ledger: JSON.parse(readFileSync(ledgerPath, 'utf8')), ledgerPath };
  }
  return { ledger: {}, ledgerPath };
}

// Runs the effect at most once per key, and persists the dedupe record BEFORE
// returning. The key — not the arguments — is the identity of the operation: two
// calls with the same key are "the same charge retried", even across process
// restarts, because the record is durable.
//
// Invariant: a given key fires the underlying effect exactly once for the life of
// the durable ledger. Failure mode it guards: a crash-and-resume that re-runs the
// in-flight step.
function chargeOnce(store: LedgerStore, key: string, amountUsd: number): string {
  const prior = store.ledger[key];
  if (prior !== undefined) {
    // Retry path: the effect already happened. Return the SAME result as before —
    // the caller cannot tell a retry from the original, which is the point. No
    // second charge.
    return `${prior} (idempotent replay, no new charge)`;
  }
  // First time for this key: fire the real effect, record it, persist the record.
  // Persisting the ledger HERE (not at end-of-step) is what closes the dangerous
  // window: even if we crash one line later, the record is already on disk, so the
  // re-run dedupes. When ledgerPath is null (DRIFT) this persistence is skipped and
  // the record dies with the process.
  chargeFireCount += 1;
  const result = `charged $${amountUsd} (charge #${chargeFireCount})`;
  store.ledger[key] = result;
  if (store.ledgerPath) {
    writeFileSync(store.ledgerPath, JSON.stringify(store.ledger, null, 2), 'utf8');
  }
  return result;
}

// ============================================================================
// §CHECKPOINT — the durable agent progress, and read/write helpers.
//
// AgentState answers "if this process dies right now, what does the next one need
// to continue?". Note what it does NOT contain: the idempotency ledger. The
// ledger is durable on its OWN schedule (fire-time, §IDEMPOTENT-TOOL), because the
// crash window we care about is between a side effect and the step checkpoint —
// if the ledger waited for the step checkpoint it would be lost exactly when
// needed. Splitting the two durability schedules is the non-obvious design call
// this stage is teaching.
// ============================================================================

interface AgentState {
  // The conversation so far. This is what makes resume cheap: we replay it to the
  // model, we do NOT re-run the turns that produced it.
  messages: Message[];
  // How many loop steps are durably complete. The resume cursor: a fresh process
  // trusts this to know where to continue. Deliberately written AFTER the step's
  // work, so it can lag the actual side effects by one crash window — which is
  // exactly why the ledger, not this counter, is what prevents a double charge.
  completedSteps: number;
}

// Write the whole blob each step. A production checkpoint would write-to-temp +
// rename (atomic on POSIX) so a crash mid-write cannot leave a half-file; we note
// the gap rather than hide it (a half-written JSON here would throw in JSON.parse
// on resume — fail loud beats resume-from-garbage).
function saveCheckpoint(path: string, state: AgentState): void {
  writeFileSync(path, JSON.stringify(state, null, 2), 'utf8');
}

// Returns null when no checkpoint exists — that null is the "cold start" signal
// the loop branches on.
function loadCheckpoint(path: string): AgentState | null {
  if (!existsSync(path)) return null;
  const raw = JSON.parse(readFileSync(path, 'utf8'));
  return { messages: raw.messages ?? [], completedSteps: raw.completedSteps ?? 0 };
}

// ============================================================================
// §TOOLS — the model-facing spec + the harness-side impls.
// ============================================================================

const TOOLS: ToolSpec[] = [
  {
    name: 'charge_card',
    description:
      'Charge the customer card for an order. Pass a stable idempotency_key so that retrying the ' +
      'same logical charge never double-charges. Returns a human-readable receipt.',
    inputSchema: {
      type: 'object',
      properties: {
        amount_usd: { type: 'number', description: 'amount to charge in USD' },
        idempotency_key: {
          type: 'string',
          description: 'stable id for THIS logical charge; reuse it on retry to dedupe',
        },
      },
      required: ['amount_usd', 'idempotency_key'],
    },
  },
];

// Impls close over the live ledger store so a charge persists through it. Returns
// a string to honor the tool_result contract from chapter 02.
function makeToolImpls(store: LedgerStore): Record<string, (input: Record<string, unknown>) => string> {
  return {
    charge_card(input) {
      const amount = Number(input.amount_usd ?? 0);
      const key = String(input.idempotency_key ?? '');
      if (!key) {
        // Failure mode: no key means we cannot dedupe, so a retry WILL double-fire.
        // Refuse loudly rather than charge unsafely.
        throw new Error('charge_card called without idempotency_key (unsafe to retry)');
      }
      return chargeOnce(store, key, amount);
    },
  };
}

// TOOLS documents the model-facing contract for charge_card. The mock here is
// fully scripted (see generateAtStep), so it never reaches llm.generate to be
// consumed — held for shape, the same way generateAtStep holds `llm`. A real
// key would pass it through.
void TOOLS;

// ============================================================================
// §LOOP — a resumable agent loop.
//
// Structurally the stage-01 loop (while around one LLM call) plus two reliability
// bolt-ons: it can START from a checkpoint instead of a fresh user message, and it
// checkpoints after every step. The crash is injected at the most dangerous point
// on purpose — see `crashAfterEffectAtStep`.
// ============================================================================

interface RunOptions {
  llm: LLM;
  checkpointPath: string;
  // null => idempotency ledger is NOT made durable (the §DRIFT injection).
  ledgerPath: string | null;
  userInput: string;
  maxSteps: number;
  // Crash AFTER this step's tool effect has fired but BEFORE its checkpoint is
  // written — the realistic gap. -1 = never crash. On resume the loop re-runs this
  // step (its checkpoint never landed); whether that re-run is safe depends
  // entirely on the ledger being durable.
  crashAfterEffectAtStep: number;
}

interface RunResult {
  answer: string;
  stepsRunThisProcess: number;
  resumedFromStep: number;
}

function runResumableAgent(opts: RunOptions): RunResult {
  const { llm, checkpointPath, ledgerPath, userInput, maxSteps, crashAfterEffectAtStep } = opts;

  // Cold start vs resume is decided entirely by whether a checkpoint exists. This
  // branch is what makes the agent durable: resume reuses the dead process's
  // messages and skips the steps it already finished.
  const restored = loadCheckpoint(checkpointPath);
  const state: AgentState = restored ?? { messages: [{ role: 'user', content: userInput }], completedSteps: 0 };
  const resumedFromStep = restored?.completedSteps ?? 0;
  const store = loadLedger(ledgerPath);

  if (restored) {
    console.log(
      `  [resume] checkpoint at step ${restored.completedSteps}; ledger has ${Object.keys(store.ledger).length} prior charge(s) ` +
        `${ledgerPath ? '(durable)' : '(NOT durable — drift)'}`
    );
  } else {
    console.log('  [cold]   no checkpoint; starting fresh');
  }

  const toolImpls = makeToolImpls(store);
  let stepsRunThisProcess = 0;

  // Loop counter is RESTORED from disk, so a resumed process picks up mid-stream.
  for (let step = state.completedSteps; step < maxSteps; step++) {
    // The mock is stateless across processes (a new MockLLM each run resets its
    // turn counter), so the policy is keyed on the RESTORED `step`, not an in-memory
    // index — mirroring a real model, which only sees what we resend.
    const turn = generateAtStep(llm, state.messages, step);
    state.messages.push({ role: 'assistant', content: turn });
    stepsRunThisProcess += 1;

    const wantsTool = turn.some((b) => b.type === 'tool_use');
    if (wantsTool) {
      const results: UserBlock[] = [];
      for (const block of turn) {
        if (block.type !== 'tool_use') continue;
        const impl = toolImpls[block.name];
        try {
          results.push({ type: 'tool_result', toolUseId: block.id, content: impl(block.input) });
        } catch (err) {
          results.push({ type: 'tool_result', toolUseId: block.id, content: `error: ${(err as Error).message}`, isError: true });
        }
      }
      state.messages.push({ role: 'user', content: results });

      // CRASH WINDOW: the side effect has fired (and, when durable, the ledger
      // record is already on disk) but the STEP checkpoint has not been written.
      // This is the gap that exists in every real system. Crash here.
      if (step === crashAfterEffectAtStep) {
        throw new SimulatedCrash(state.completedSteps);
      }
    }

    // Step checkpoint: only now is `completedSteps` advanced and persisted. A crash
    // before this line means resume re-runs `step` — safe iff the ledger deduped it.
    state.completedSteps = step + 1;
    saveCheckpoint(checkpointPath, state);
    console.log(`  [ckpt]   step ${state.completedSteps} durable: ${describeStep(turn)} | charges fired so far=${chargeFireCount}`);

    if (!wantsTool) {
      return { answer: textOf(turn), stepsRunThisProcess, resumedFromStep };
    }
  }

  return { answer: '[stopped: max steps reached]', stepsRunThisProcess, resumedFromStep };
}

// Distinct error type so the demo catches ONLY the simulated crash and lets any
// genuine bug propagate. Carries the last durable step for the log — a real crash
// handler would read this off disk instead.
class SimulatedCrash extends Error {
  constructor(public readonly lastDurableStep: number) {
    super(`simulated crash; last durable step = ${lastDurableStep}`);
    this.name = 'SimulatedCrash';
  }
}

function textOf(content: AssistantBlock[]): string {
  return content.filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text').map((b) => b.text).join('');
}

function describeStep(content: AssistantBlock[]): string {
  const tool = content.find((b) => b.type === 'tool_use');
  return tool && tool.type === 'tool_use' ? `tool_use ${tool.name}` : 'text answer';
}

// ============================================================================
// §MOCK — a deterministic, step-indexed policy.
//
// A fixed 4-step plan: charge three orders, then summarize. Each charge uses a
// STABLE idempotency key derived from the order, so a re-run of any step reuses
// the key and the ledger dedupes it. The policy is keyed on the loop `step`, not a
// MockLLM turn counter, because a resume spawns a NEW MockLLM whose counter is
// back at 0 — the durable cursor is `step`.
// ============================================================================

const ORDERS: Array<{ id: string; amountUsd: number }> = [
  { id: 'A-1001', amountUsd: 30 },
  { id: 'A-1002', amountUsd: 12 },
  { id: 'A-1003', amountUsd: 45 },
];

function generateAtStep(llm: LLM, messages: Message[], step: number): AssistantBlock[] {
  // We still hold an LLM handle so the shape matches every other stage (a real key
  // would plug in here), but the scripted policy is what makes the demo
  // reproducible. usage/tokens are irrelevant to this stage, so we ignore them.
  void llm;
  if (step < ORDERS.length) {
    const order = ORDERS[step];
    return [
      { type: 'text', text: `Charging order ${order.id}.` },
      {
        type: 'tool_use',
        id: `call_${step}`,
        name: 'charge_card',
        // Stable, deterministic key: the operation's identity is the order, so a
        // re-run reuses the key and dedupes.
        input: { amount_usd: order.amountUsd, idempotency_key: `charge:${order.id}` },
      },
    ];
  }
  // Final step: count receipts from the transcript and answer. Computed from real
  // state, not asserted.
  return [{ type: 'text', text: `Done. ${countChargeReceipts(messages)} orders charged.` }];
}

// Count tool_results that are a FRESH charge (not an idempotent replay). Honest
// accounting: a replay is not a new charge and must not inflate the count.
function countChargeReceipts(messages: Message[]): number {
  let n = 0;
  for (const m of messages) {
    if (m.role !== 'user' || typeof m.content === 'string') continue;
    for (const b of m.content) {
      if (b.type === 'tool_result' && b.content.startsWith('charged $')) n += 1;
    }
  }
  return n;
}

// Thin pass-through mock: generateAtStep owns the script, so this never decides
// anything. Kept to mark the seam where a real model plugs in unchanged.
function makeMock(): LLM {
  return new MockLLM(() => ({ content: [{ type: 'text', text: '' }] }));
}

// ============================================================================
// §DEMO
// ============================================================================

// Run process #1 until it crashes mid-plan, then a fresh process #2 that survives
// only by reading the checkpoint. ledgerPath=null injects the drift bug.
function runAcrossCrash(label: string, checkpointPath: string, ledgerPath: string | null): RunResult {
  console.log(`\n${label} — process #1 (crashes mid-run):`);
  try {
    runResumableAgent({
      llm: makeMock(),
      checkpointPath,
      ledgerPath,
      userInput: `Charge ${ORDERS.length} pending orders, then summarize.`,
      maxSteps: ORDERS.length + 1,
      crashAfterEffectAtStep: 1, // crash after order #2's charge fires, before its checkpoint
    });
    throw new Error('expected a simulated crash but the run finished');
  } catch (err) {
    if (!(err instanceof SimulatedCrash)) throw err;
    console.log(`  [crash]  process died; ${err.message} (step 1's effect fired but its checkpoint did NOT land)`);
  }

  console.log(`${label} — process #2 (resumes from checkpoint):`);
  return runResumableAgent({
    llm: makeMock(),
    checkpointPath,
    ledgerPath,
    userInput: 'IGNORED ON RESUME — checkpoint supplies the real history',
    maxSteps: ORDERS.length + 1,
    crashAfterEffectAtStep: -1, // run to completion this time
  });
}

function main(): void {
  console.log('=== Stage 15: durable / resumable agent (offline, deterministic) ===');
  const dir = mkdtempSync(join(tmpdir(), 'agent-durable-'));
  console.log('temp dir:', dir);

  // ---- Scenario A: correct durability -----------------------------------
  // Crash after step 1's charge fires. Its checkpoint never landed, so resume
  // re-runs step 1 — but the ledger (durable at fire-time) dedupes the re-charge.
  // Net: 3 distinct charges, each fired exactly once, despite step 1 running twice.
  chargeFireCount = 0;
  const a = runAcrossCrash('SCENARIO A (ledger durable)', join(dir, 'a-ckpt.json'), join(dir, 'a-ledger.json'));
  console.log('\n  RESULT A:');
  console.log('    answer            :', a.answer);
  console.log('    resumed from step :', a.resumedFromStep, '(step 1 re-ran; earlier steps did not)');
  console.log('    steps in proc #2  :', a.stepsRunThisProcess, `(of ${ORDERS.length + 1} total)`);
  console.log('    real charges fired:', chargeFireCount, `(expected ${ORDERS.length}: re-run of step 1 was deduped)`);

  // ---- Idempotency in isolation -----------------------------------------
  console.log('\n  IDEMPOTENCY CHECK (same key twice, fresh in-memory ledger):');
  chargeFireCount = 0;
  const isoStore: LedgerStore = { ledger: {}, ledgerPath: null };
  console.log('    1st call:', chargeOnce(isoStore, 'charge:A-1001', 30));
  console.log('    2nd call:', chargeOnce(isoStore, 'charge:A-1001', 30));
  console.log('    real charges fired:', chargeFireCount, '(expected 1: retry deduped)');

  // ---- Scenario B: DRIFT — idempotency state not made durable ------------
  // Same crash, same resume, but the ledger lived only in memory (ledgerPath=null).
  // Process #2 starts with an empty ledger, so the re-run of step 1 looks like a
  // first-time charge and fires AGAIN. The answer is identical to A — the bug is
  // invisible in the output and only the real fire count exposes it.
  console.log('\n--- §DRIFT: idempotency ledger never made durable ---');
  chargeFireCount = 0;
  const b = runAcrossCrash('SCENARIO B (ledger NOT durable)', join(dir, 'b-ckpt.json'), null);
  const driftCharges = chargeFireCount;
  console.log('\n  RESULT B:');
  console.log('    answer            :', b.answer, '(identical to A — the bug hides here)');
  console.log('    real charges fired:', driftCharges, `(expected ${ORDERS.length}, got MORE: order A-1002 charged twice)`);

  console.log('\n  COMPARISON (real side effects fired):');
  console.log(`    ledger durable     : ${ORDERS.length} charges  (each order once)`);
  console.log(`    ledger drift       : ${driftCharges} charges  (+${driftCharges - ORDERS.length} duplicate from the re-run step)`);
  console.log('    lesson: checkpointing the conversation is not enough. The state that makes a retry SAFE');
  console.log('            (the idempotency ledger) is itself durable state — omit it and at-least-once');
  console.log('            execution silently becomes double side effects.');

  rmSync(dir, { recursive: true, force: true });
  console.log(`\n(cleaned up temp dir ${dir})`);
}

main();
