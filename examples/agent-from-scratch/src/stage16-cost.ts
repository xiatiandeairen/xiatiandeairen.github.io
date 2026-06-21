// Stage 16 — cost & latency engineering.
//
// A working agent that ignores cost is a demo, not a product. The single biggest
// lever is NOT prompt golf — it is routing: send the cheap-and-good-enough work to
// a cheap model, and only pay for the expensive model on the steps that actually
// need it. This chapter builds a model router / cascade and measures it, in money,
// against the naive "always use the expensive model" baseline.
//
// Three mechanisms, each with a real number attached:
//   (1) Model cascade   — try cheap first; escalate to expensive only on
//                          low-confidence answers. We measure the escalation rate
//                          and the dollar saving.
//   (2) Prefix caching  — (reused from chapter 05) a byte-identical prompt prefix
//                          is billed at a fraction of the price. Routing decides
//                          WHICH model; caching decides how cheap each call is.
//   (3) Failure modes   — routing is a bet. Under-escalation (hard task kept on
//                          the cheap model → WRONG answer) and over-escalation
//                          (easy task sent to the expensive model → wasted money)
//                          are the two ways the bet loses. We force both.
//
// HONESTY NOTE on the numbers: every token count comes from `estimateTokens`
// (~4 chars/token, see core/llm.ts) — a stable heuristic, NOT a real tokenizer,
// and noticeably off for CJK. The per-token PRICES below are illustrative round
// numbers, not any provider's real card. So treat the absolute dollar figures as
// toy; the RATIOS (cascade vs baseline, cache discount, escalation rate) are the
// real, reproducible output of this stage.

import { createLLM, MockLLM, estimateTokens } from './core/llm.js';
import type { AssistantBlock, GenerateOptions, LLM } from './core/types.js';

// --- Pricing model. --------------------------------------------------------
//
// Real APIs bill input and output tokens at different rates, and cached input
// tokens at a steep discount. We keep the same three knobs so the cascade math
// has the same shape as production. Units are $ per 1K tokens (toy values).
interface Pricing {
  readonly tier: 'cheap' | 'expensive';
  readonly inputPer1k: number;
  readonly outputPer1k: number;
  // Fraction of the input price charged for a cached-prefix HIT. Providers
  // discount cached input heavily; 0.1 ≈ "cached input is 10x cheaper".
  readonly cachedInputMultiplier: number;
}

// Illustrative ratio: the expensive tier costs ~12-15x the cheap tier per token.
// That spread is exactly why routing pays off — and also why mis-routing hurts.
const CHEAP: Pricing = { tier: 'cheap', inputPer1k: 0.25, outputPer1k: 1.25, cachedInputMultiplier: 0.1 };
const EXPENSIVE: Pricing = { tier: 'expensive', inputPer1k: 3.0, outputPer1k: 15.0, cachedInputMultiplier: 0.1 };

// Dollar cost of one model call. cachedInputTokens are billed at the discounted
// rate; the rest of the input + all output at full rate. Pure function of the
// usage numbers, so the printed totals are genuinely summed from these, not
// hand-waved.
function computeCallCost(
  price: Pricing,
  inputTokens: number,
  outputTokens: number,
  cachedInputTokens: number
): number {
  const freshInput = Math.max(0, inputTokens - cachedInputTokens);
  return (
    (freshInput / 1000) * price.inputPer1k +
    (cachedInputTokens / 1000) * price.inputPer1k * price.cachedInputMultiplier +
    (outputTokens / 1000) * price.outputPer1k
  );
}

// --- Prefix caching (reused from chapter 05, trimmed to what we bill). ------
//
// Why it belongs in a COST chapter: the cascade picks the model, but the prompt
// prefix (system + tool specs) is resent on every call. If that prefix stays
// byte-identical the provider serves it from its attention cache at a fraction of
// the price. So the cache discount multiplies whatever the router saves. The
// catch (chapter 05): edit one byte of the prefix and the cache misses from that
// point on. Here we only need "is the prefix identical to last time?" to decide
// how many input tokens get the cached discount.

// Deterministic 32-bit FNV-1a over the prefix string. Same algorithm as
// chapter 05 — this models "what the provider keys its cache on", not the real
// provider hash.
function hashPrefix(prefix: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < prefix.length; i++) {
    h ^= prefix.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

// The stable prefix every call shares: the system prompt + tool specs. In a real
// agent this is the bulk of the input tokens and the whole point of caching.
function buildPrefix(system: string): string {
  return system;
}

// --- Task model + difficulty oracle. ---------------------------------------
//
// Each task carries a hidden `groundTruthHard` flag — the TRUTH about whether it
// needs the expensive model. The router never sees this; it only sees the text
// and must GUESS. We keep the truth around solely to score the router's bet and
// to demonstrate the two failure modes. In production you don't have this oracle,
// which is exactly why routing is risky.
interface Task {
  readonly id: string;
  readonly prompt: string;
  // The honest label: does answering this CORRECTLY require the expensive model?
  readonly groundTruthHard: boolean;
}

// Heuristic router: classify a task as "hard" from cheap surface signals (length,
// multi-step / reasoning keywords). This is deliberately imperfect — a keyword
// router is what most teams ship first, and its mistakes ARE the failure modes
// below. Returns the GUESS, not the truth.
function routeIsHard(prompt: string): boolean {
  const longEnough = prompt.length > 80; // long prompts skew complex
  const hardSignal = /\b(prove|derive|design|why|trade-?off|step by step|edge case)\b/i.test(prompt);
  return longEnough || hardSignal;
}

// --- The two mock tiers. ---------------------------------------------------
//
// Both tiers are deterministic scripts so the run is reproducible offline. We
// model competence honestly:
//   - cheap tier  : answers EASY tasks confidently and correctly; on HARD tasks
//                   it returns a low-confidence hedge (the signal the cascade
//                   watches for) AND a wrong answer.
//   - expensive   : answers everything confidently and correctly.
// "Confidence" is carried as a [conf=0.NN] tag in the text — a stand-in for the
// logprobs / self-consistency / verifier score a real cascade would threshold on.

const CONFIDENCE_TAG = /\[conf=([0-9.]+)\]/;
const ESCALATE_BELOW = 0.6; // cascade escalates when cheap-tier confidence < this

function lastUserPrompt(opts: GenerateOptions): string {
  const first = opts.messages[0];
  return typeof first.content === 'string' ? first.content : '';
}

// Cheap tier: good on easy, hedges + wrong on hard. The hedge is the honest
// behavior that lets a cascade recover; a cheap model that hid its uncertainty
// would be far more dangerous.
const cheapMock = new MockLLM((opts) => {
  const prompt = lastUserPrompt(opts);
  const hard = /\b(prove|derive|design|why|trade-?off|step by step|edge case)\b/i.test(prompt) || prompt.length > 80;
  const content: AssistantBlock[] = hard
    ? [{ type: 'text', text: `[conf=0.40] I think it is roughly X, but I am not sure about the harder parts.` }]
    : [{ type: 'text', text: `[conf=0.95] Done. The answer is a concise, correct result.` }];
  return { content, stopReason: 'end_turn' };
});

// Expensive tier: always confident and correct (within this toy). Costs ~12x.
const expensiveMock = new MockLLM((_opts) => ({
  content: [{ type: 'text', text: `[conf=0.98] Verified, fully-reasoned answer covering every step and edge case.` }],
  stopReason: 'end_turn',
}));

// --- Execution strategies. -------------------------------------------------
//
// Every strategy runs the SAME task list and returns a per-task cost + an answer
// quality flag, so the totals are comparable. `correct` is judged by the oracle:
// a task answered by a tier that is competent for it. This is how we surface
// under-escalation as an objective wrong-answer count, not a vibe.

interface CallRecord {
  readonly taskId: string;
  readonly tier: 'cheap' | 'expensive';
  readonly escalated: boolean; // cheap tried first then bumped to expensive
  readonly costUsd: number;
  readonly correct: boolean;
  readonly cacheHit: boolean;
}

// A persistent prefix cache: the set of prefix hashes the provider has already
// processed. First time a prefix is seen → MISS (full price, the cache gets
// populated). Every later call with the SAME prefix → HIT (prefix billed at the
// discounted rate). This is why the FIRST call of a cold run misses and the rest
// hit, and why an entire warm re-run hits from call one. A real provider expires
// the cache on a TTL; we keep it simple and never evict.
type PrefixCache = Set<string>;

// One billed model call, with cache accounting against a shared PrefixCache.
async function callModel(
  llm: LLM,
  price: Pricing,
  opts: GenerateOptions,
  cache: PrefixCache
): Promise<Omit<CallRecord, 'taskId' | 'escalated' | 'correct'>> {
  const turn = await llm.generate(opts);
  const prefix = buildPrefix(opts.system ?? '');
  const prefixHash = hashPrefix(prefix);
  const prefixTokens = estimateTokens(prefix);

  const cacheHit = cache.has(prefixHash);
  cache.add(prefixHash); // populate so the next identical call hits
  // On a hit, the prefix portion of the input is billed at the cached rate.
  const cachedInputTokens = cacheHit ? Math.min(prefixTokens, turn.usage.inputTokens) : 0;

  const costUsd = computeCallCost(price, turn.usage.inputTokens, turn.usage.outputTokens, cachedInputTokens);
  // `correct` is the caller's call (it knows the oracle); we only return cost/cache facts.
  return { tier: price.tier, costUsd, cacheHit };
}

// Confidence extracted from a tier's answer — the cascade's escalation signal.
// Real systems use logprobs / a verifier model / self-consistency; the tag is a
// deterministic stand-in so the demo is reproducible.
function confidenceOf(turnText: string): number {
  const m = turnText.match(CONFIDENCE_TAG);
  return m ? Number(m[1]) : 1;
}

function textOf(content: AssistantBlock[]): string {
  return content.filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text').map((b) => b.text).join('');
}

// Strategy A — BASELINE: every task goes to the expensive model. Correct, but you
// pay the premium even for trivial steps. This is the bill the cascade competes
// against.
async function runBaseline(llm: LLM, system: string, tasks: Task[], cache: PrefixCache): Promise<CallRecord[]> {
  const records: CallRecord[] = [];
  for (const task of tasks) {
    const opts: GenerateOptions = { system, messages: [{ role: 'user', content: task.prompt }] };
    const record = await callModel(llm, EXPENSIVE, opts, cache);
    // Expensive tier is competent for everything in this toy → always correct.
    records.push({ ...record, taskId: task.id, escalated: false, correct: true });
  }
  return records;
}

// Strategy B — CASCADE: try cheap first. If the cheap tier's confidence clears the
// threshold, keep its (cheap) answer. Otherwise escalate: pay the cheap call AND
// the expensive call for that task. The bet: most tasks clear the bar, so the
// average cost lands far below baseline despite paying twice on the hard minority.
// Each tier caches its OWN prefix (a cheap model and an expensive model are
// different providers with independent caches), so the cascade threads two caches.
async function runCascade(
  cheap: LLM,
  expensive: LLM,
  system: string,
  tasks: Task[],
  cheapCache: PrefixCache,
  expCache: PrefixCache
): Promise<CallRecord[]> {
  const records: CallRecord[] = [];
  for (const task of tasks) {
    const opts: GenerateOptions = { system, messages: [{ role: 'user', content: task.prompt }] };

    // The cheap probe always runs first — it is the cascade's cheap "screening".
    const cheapTurn = await cheap.generate(opts);
    const prefixHash = hashPrefix(buildPrefix(system));
    const cheapCacheHit = cheapCache.has(prefixHash);
    cheapCache.add(prefixHash);
    const prefixTokens = estimateTokens(buildPrefix(system));
    const cheapCachedInput = cheapCacheHit ? Math.min(prefixTokens, cheapTurn.usage.inputTokens) : 0;
    const cheapCost = computeCallCost(CHEAP, cheapTurn.usage.inputTokens, cheapTurn.usage.outputTokens, cheapCachedInput);

    const confidence = confidenceOf(textOf(cheapTurn.content));
    if (confidence >= ESCALATE_BELOW) {
      // Kept the cheap answer. Correct iff the cheap tier was actually competent
      // for this task — i.e. the task was genuinely easy. A cheap tier that was
      // overconfident on a hard task would land here WRONG (under-escalation).
      records.push({
        taskId: task.id,
        tier: 'cheap',
        escalated: false,
        costUsd: cheapCost,
        correct: !task.groundTruthHard,
        cacheHit: cheapCacheHit,
      });
      continue;
    }

    // Escalate: the cheap call is sunk cost; now pay the expensive call too.
    const expRec = await callModel(expensive, EXPENSIVE, opts, expCache);
    records.push({
      taskId: task.id,
      tier: 'expensive',
      escalated: true,
      costUsd: cheapCost + expRec.costUsd, // both calls billed
      correct: true, // expensive tier resolves it
      cacheHit: expRec.cacheHit,
    });
  }
  return records;
}

// --- Reporting. ------------------------------------------------------------

interface Summary {
  readonly label: string;
  readonly totalCostUsd: number;
  readonly correctCount: number;
  readonly total: number;
  readonly escalations: number;
  readonly cacheHits: number;
}

function summarize(label: string, records: CallRecord[]): Summary {
  return {
    label,
    totalCostUsd: records.reduce((s, r) => s + r.costUsd, 0),
    correctCount: records.filter((r) => r.correct).length,
    total: records.length,
    escalations: records.filter((r) => r.escalated).length,
    cacheHits: records.filter((r) => r.cacheHit).length,
  };
}

function fmtUsd(n: number): string {
  return `$${n.toFixed(5)}`;
}

function printSummary(s: Summary): void {
  const escRate = s.total > 0 ? ((s.escalations / s.total) * 100).toFixed(0) : '0';
  console.log(
    `${s.label.padEnd(22)} cost=${fmtUsd(s.totalCostUsd).padStart(10)}  ` +
      `correct=${s.correctCount}/${s.total}  ` +
      `escalations=${s.escalations} (${escRate}%)  cache_hits=${s.cacheHits}/${s.total}`
  );
}

// --- Demo: failure modes. --------------------------------------------------
//
// Routing is a bet, and we show both ways it loses with the router's GUESS vs the
// oracle TRUTH side by side. These are not the cascade (which self-corrects via
// confidence) — they are the static "route once, trust the guess" policy that the
// cascade exists to fix.
function demoRoutingFailures(tasks: Task[]): void {
  console.log('\n--- failure modes: static router guess vs ground truth ---\n');
  let underEscalation = 0; // guessed easy, truly hard  → routed cheap → WRONG
  let overEscalation = 0; // guessed hard, truly easy   → routed expensive → WASTE
  for (const t of tasks) {
    const guessHard = routeIsHard(t.prompt);
    const tag =
      guessHard === t.groundTruthHard
        ? 'ok'
        : guessHard
          ? 'OVER-ESCALATION (paid premium on an easy task → wasted money)'
          : 'UNDER-ESCALATION (kept hard task on cheap model → wrong answer)';
    if (!guessHard && t.groundTruthHard) underEscalation++;
    if (guessHard && !t.groundTruthHard) overEscalation++;
    console.log(`  ${t.id}: guess=${guessHard ? 'hard' : 'easy'} truth=${t.groundTruthHard ? 'hard' : 'easy'} → ${tag}`);
  }
  console.log(
    `\n  static-router mistakes: ${underEscalation} under-escalation (correctness bug), ` +
      `${overEscalation} over-escalation (cost bug).`
  );
  console.log(
    `  the cascade dodges under-escalation by re-checking confidence and escalating,\n` +
      `  but it CANNOT dodge over-escalation it never triggers — that risk lives in the\n` +
      `  confidence signal's calibration, not the router heuristic.\n`
  );
}

// --- Workload. -------------------------------------------------------------
//
// A realistic agent workload skews easy: lots of trivial lookups/edits, a few
// genuinely hard reasoning steps. We hand-label ground truth and DELIBERATELY
// include tasks where the keyword router guesses wrong, so the failure-mode demo
// has something to find.
const TASKS: Task[] = [
  { id: 't1', prompt: 'List the files in the src directory.', groundTruthHard: false },
  { id: 't2', prompt: 'What is 2 + 2?', groundTruthHard: false },
  { id: 't3', prompt: 'Rename the variable x to count.', groundTruthHard: false },
  { id: 't4', prompt: 'Format this JSON.', groundTruthHard: false },
  {
    id: 't5',
    prompt: 'Design a retry policy with backoff and explain the trade-off between latency and load.',
    groundTruthHard: true,
  },
  {
    id: 't6',
    prompt: 'Prove that the loop terminates for every edge case of the input grammar, step by step.',
    groundTruthHard: true,
  },
  // t7: a SHORT hard task with no keywords — the keyword router GUESSES easy and
  // gets it wrong (under-escalation). The cascade still catches it via confidence.
  { id: 't7', prompt: 'Solve the halting subset for f.', groundTruthHard: true },
  // t8: a LONG easy task (verbose but trivial) — the keyword router GUESSES hard
  // (it is over 80 chars) and over-escalates. Pure waste under a static router.
  {
    id: 't8',
    prompt: 'Please kindly go ahead and simply copy the text from file A into file B, thank you so much.',
    groundTruthHard: false,
  },
];

async function main(): Promise<void> {
  // The mocks are the whole point of this stage (two priced tiers), so we always
  // use them — createLLM would swap BOTH to the same real model and collapse the
  // cascade. We still funnel through createLLM so the offline banner prints and
  // the wiring matches every other stage; in this stage a real key only changes
  // who answers, not the cost arithmetic, which is computed locally regardless.
  const cheap = createLLM(cheapMock);
  const expensive = expensiveMock; // keep tiers distinct even if a key is set

  console.log(`\n=== Stage 16: cost & latency engineering (cheap=${cheap.name}, expensive=mock) ===\n`);
  console.log(
    `pricing (toy, $/1K tok): cheap in=${CHEAP.inputPer1k} out=${CHEAP.outputPer1k} | ` +
      `expensive in=${EXPENSIVE.inputPer1k} out=${EXPENSIVE.outputPer1k} | ` +
      `cached-input discount=${EXPENSIVE.cachedInputMultiplier}x\n`
  );

  const system = 'You are an agent. Answer the task. Prefix your reply with [conf=0.NN] self-rated confidence.';

  // Cold caches for the headline comparison: both strategies start with no cached
  // prefix, so call 1 misses and the rest hit (the system prefix is constant).
  const baselineCache: PrefixCache = new Set();
  const baseline = summarize('always-expensive', await runBaseline(expensive, system, TASKS, baselineCache));

  // The cascade's caches survive into the warm-cache demo below.
  const cheapCache: PrefixCache = new Set();
  const expCache: PrefixCache = new Set();
  const cascade = summarize('cascade', await runCascade(cheap, expensive, system, TASKS, cheapCache, expCache));

  console.log('--- cost comparison on the same 8-task workload ---\n');
  printSummary(baseline);
  printSummary(cascade);

  const saved = baseline.totalCostUsd - cascade.totalCostUsd;
  const savedPct = (saved / baseline.totalCostUsd) * 100;
  console.log(
    `\ncascade saves ${fmtUsd(saved)} (${savedPct.toFixed(1)}%) vs always-expensive, ` +
      `at an escalation rate of ${((cascade.escalations / cascade.total) * 100).toFixed(0)}%.`
  );
  console.log(
    `correctness: baseline ${baseline.correctCount}/${baseline.total}, ` +
      `cascade ${cascade.correctCount}/${cascade.total} ` +
      `(${cascade.total - cascade.correctCount} wrong = tasks the cheap tier was overconfident on).`
  );
  console.log(
    `the saving is NOT free: ${cascade.total - cascade.correctCount} task(s) under-escalated ` +
      `(cheap tier confident but wrong) and some of the ${cascade.escalations} escalations are ` +
      `over-escalations (easy task the cheap tier hedged on). The dollar win and the correctness\n` +
      `loss are the same bet — quantified in the failure-mode table below.`
  );

  // Cache discount: re-run the cascade against the SAME caches (now warm). Every
  // call's prefix is already cached, so cache_hits jumps to all calls and the
  // resent system+tool tokens are billed at the discount. This isolates the
  // prefix-cache effect from routing — same routing decisions, lower bill.
  const secondPass = summarize('cascade (warm cache)', await runCascade(cheap, expensive, system, TASKS, cheapCache, expCache));
  const cacheSaving = cascade.totalCostUsd - secondPass.totalCostUsd;
  console.log(
    `\nprefix cache: a second identical run costs ${fmtUsd(secondPass.totalCostUsd)} ` +
      `(${secondPass.cacheHits}/${secondPass.total} calls hit the cached prefix), ` +
      `saving a further ${fmtUsd(cacheSaving)} on resent system+tool tokens.`
  );

  demoRoutingFailures(TASKS);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
