// Stage 09 — reflection and self-correction.
//
// A single LLM call is a one-shot guess. The cheapest way to make it better is
// not a bigger model or a longer prompt — it is a LOOP: generate, judge, and if
// the judgment fails, feed the failure back as feedback and try again. This is
// the Reflexion pattern (Shinn et al. 2023): an agent that reflects on its own
// failed attempts and retries with that reflection in context.
//
// The entire chapter hinges on one distinction that is easy to blur:
//
//   - VERIFIER: judges with a signal that exists OUTSIDE the model — a unit test,
//     a compiler, a regex, a checksum. Ground truth. Cannot be argued with.
//   - CRITIC:   judges with the model's own opinion — "does this look right?".
//     No ground truth. Can be argued with, including by itself.
//
// A retry loop driven by a VERIFIER converges: each round either passes (done)
// or produces concrete, true feedback the next attempt can act on. A retry loop
// driven by a same-model CRITIC does NOT reliably converge, and this stage demos
// two distinct ways it breaks:
//
//   1. OSCILLATION — the critic keeps flipping its preference, so the generator
//      rewrites back and forth forever and never settles (see §OSCILLATION).
//   2. SELF-PERSUASION — when the critic is the same model that produced the
//      answer, it endorses its own wrong output. The loop "passes" on round 1
//      with full confidence and ships a bug (see §SELF-PERSUASION). This is the
//      dangerous one: it looks like success.
//
// The practical takeaway the section should land: reflection is only as good as
// the signal closing the loop. Wire a real verifier (test/compiler/checker) or
// you have built a machine for laundering a guess into false confidence.
//
// Run it: `npx tsx src/stage09-reflection.ts`. Fully offline and deterministic.
// The model is a MockLLM scripted to reproduce each behavior; the VERIFIER is a
// real mechanical checker (genuine code, genuine pass/fail), so every "passed on
// round N" number below is computed, not narrated.

import { MockLLM, createLLM } from './core/llm.js';
import type { AssistantBlock, LLM, Message } from './core/types.js';

// ============================================================================
// §TASK — a task with an OBJECTIVE pass/fail signal.
//
// "Produce a slug from a title" that must satisfy four mechanical rules. A regex
// + length check decides pass/fail with zero judgment, which is exactly what
// makes it a VERIFIER and not a CRITIC. The rules are deliberately fiddly (the
// kind a model gets 90%-right on the first try) so the retry loop has something
// real to correct.
// ============================================================================

// The contract the generated slug must satisfy. Each rule maps to one line in
// verifySlug() below, so feedback can name the exact rule that failed — vague
// feedback ("make it better") is what makes critic loops oscillate; precise
// feedback ("rule 3: no uppercase") is what makes verifier loops converge.
const SLUG_RULES = [
  'rule 1: only lowercase letters, digits, and single hyphens',
  'rule 2: no leading or trailing hyphen',
  'rule 3: no two hyphens in a row',
  'rule 4: total length between 8 and 40 characters',
] as const;

// A VerifierResult is a Result type, not an exception: a failing slug is an
// expected outcome of the loop, not a program error. The loop branches on `ok`;
// `failures` carries the precise, per-rule feedback that the next attempt sees.
interface VerifierResult {
  ok: boolean;
  failures: string[];
}

// The external ground truth. This is real code with no model in it — that is the
// whole point. It can disagree with the model and the model cannot talk it out
// of the disagreement. Returns ALL broken rules at once (not just the first) so
// one retry can fix everything, rather than peeling rules off one round at a time.
function verifySlug(slug: string): VerifierResult {
  const failures: string[] = [];
  if (!/^[a-z0-9-]+$/.test(slug)) failures.push(SLUG_RULES[0]);
  if (/^-|-$/.test(slug)) failures.push(SLUG_RULES[1]);
  if (/--/.test(slug)) failures.push(SLUG_RULES[2]);
  if (slug.length < 8 || slug.length > 40) failures.push(SLUG_RULES[3]);
  return { ok: failures.length === 0, failures };
}

// ============================================================================
// §LOOP — the generator → verifier → retry loop (Reflexion).
//
// One function, parameterized by a `judge`. Swapping the judge is the entire
// experiment: pass the real verifier and the loop converges; pass a same-model
// critic and it does not. Keeping the loop identical across all three demos is
// what isolates the variable — the loop is never the problem, the SIGNAL is.
// ============================================================================

interface RoundLog {
  round: number;
  candidate: string;
  ok: boolean;
  feedback: string[];
}

interface ReflectionResult {
  finalAnswer: string;
  rounds: RoundLog[];
  // 'verified'  : a judge with ground truth accepted it — trustworthy.
  // 'accepted'  : a critic (no ground truth) accepted it — may be wrong (see
  //               §SELF-PERSUASION; the word is chosen to NOT imply correctness).
  // 'exhausted' : ran out of retries without acceptance (the honest failure).
  outcome: 'verified' | 'accepted' | 'exhausted';
}

// A judge turns a candidate into a pass/fail + feedback. The verifier and the
// critic are both Judges; that symmetry is precisely why a caller can wire the
// wrong one in without noticing — the types do not distinguish ground truth from
// opinion. Only the runtime behavior does.
type Judge = (candidate: string) => VerifierResult;

// `trustworthy` records, per judge, whether its pass actually means correct. It
// is NOT used to change control flow — the loop treats both judges identically —
// it only labels the outcome so the printout can tell "verified" from the
// look-alike "accepted". This mirrors reality: at runtime nothing forces a
// critic's "looks good" to be true; only your knowledge of the judge does.
async function reflect(
  llm: LLM,
  task: string,
  judge: Judge,
  trustworthy: boolean,
  maxRounds: number
): Promise<ReflectionResult> {
  const rounds: RoundLog[] = [];
  // The running transcript IS the reflection memory: each failed attempt plus
  // its feedback stays in `messages`, so attempt N+1 sees every prior mistake.
  // Reflexion without this accumulation is just blind resampling.
  const messages: Message[] = [{ role: 'user', content: task }];

  for (let round = 1; round <= maxRounds; round++) {
    const assistant = await llm.generate({
      system:
        'You produce a URL slug for the given title. Output ONLY the slug on a single line, nothing else.',
      messages,
    });
    messages.push({ role: 'assistant', content: assistant.content });

    const candidate = textOf(assistant.content).trim();
    const verdict = judge(candidate);
    rounds.push({ round, candidate, ok: verdict.ok, feedback: verdict.failures });

    if (verdict.ok) {
      // A pass ends the loop. Whether that pass is TRUSTWORTHY is the judge's
      // property, not the loop's — hence the outcome label is decided here, not
      // by re-checking the answer.
      return { finalAnswer: candidate, rounds, outcome: trustworthy ? 'verified' : 'accepted' };
    }

    // Failure: feed the precise reasons back as the next user turn. The model's
    // job on the next round is to read this feedback and fix exactly these rules.
    // Vague feedback here is the root cause of the oscillation demo below.
    messages.push({
      role: 'user',
      content:
        `That slug failed verification. Problems:\n` +
        verdict.failures.map((f) => `  - ${f}`).join('\n') +
        `\nProduce a corrected slug.`,
    });
  }

  // Ran out of retries. We return the honest failure marker rather than the last
  // (still-broken) candidate dressed up as an answer — a reflection loop that
  // hides non-convergence behind a confident final string is the anti-pattern
  // this whole chapter argues against.
  return { finalAnswer: '[exhausted: no candidate passed]', rounds, outcome: 'exhausted' };
}

function textOf(content: AssistantBlock[]): string {
  return content
    .filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('');
}

// ============================================================================
// §CRITIC — a judge backed by the model's OPINION, not ground truth.
//
// To demo self-persuasion honestly we need a critic whose verdict comes from a
// model rather than from verifySlug(). Below, `makeSelfCritic` builds a Judge
// that asks the SAME mock to grade its own work; the mock is scripted to always
// approve (because that is what a same-model critic empirically tends to do on
// its own confident-but-wrong output). The key property: this Judge returns
// `ok: true` for a string that verifySlug() would reject — opinion overriding
// truth. We pass trustworthy=false so the outcome is labeled 'accepted'.
// ============================================================================

function makeSelfCritic(): Judge {
  // Synchronous Judge wrapping a scripted verdict. We do not actually re-call the
  // LLM here (a critic call would just be another scripted turn); instead we
  // hard-code the behavior a same-model critic exhibits — uncritical approval —
  // so the demo is deterministic and the failure mode is unmistakable. The
  // comment, not a hidden model call, is where the honesty lives: a real critic
  // call would be non-deterministic and could occasionally catch the bug, which
  // would muddy a teaching example. See §SELF-PERSUASION printout.
  return (_candidate: string): VerifierResult => ({ ok: true, failures: [] });
}

// ============================================================================
// §MOCKS — deterministic scripts, one per demo.
//
// Each mock is a tiny state machine over turnIndex. They encode the exact answer
// trajectory we want to study; in production the real model supplies these turns
// and is far less predictable, but the LOOP and VERIFIER code above is unchanged.
// ============================================================================

// Demo 1: converges under the verifier. Round 1 violates rules 1+3 (uppercase +
// double hyphen), round 2 reads the feedback and fixes both. This is the happy
// path of reflection: a wrong-but-close first try, precise feedback, one fix.
function makeConvergingMock(): LLM {
  return new MockLLM((_opts, turnIndex) => {
    const slug =
      turnIndex === 0
        ? 'My--First-Post' // breaks rule 1 (uppercase) and rule 3 (--)
        : 'my-first-post-2026'; // satisfies all four rules
    return { content: [{ type: 'text', text: slug }], stopReason: 'end_turn' };
  });
}

// §OSCILLATION
// Demo 2: a generator with NO stable target. It alternates between two
// candidates that EACH fail (one too short, one with a trailing hyphen), never
// landing on a valid one. With a critic that gave vague, flip-flopping feedback
// this is what a real loop does: rewrite, rewrite, rewrite, no convergence. We
// drive it with the real verifier here only to SHOW the non-convergence clearly
// (the verdicts stay 'fail' every round); the cause being demonstrated is a
// generator that cannot act on feedback, which a vague critic produces.
function makeOscillatingMock(): LLM {
  return new MockLLM((_opts, turnIndex) => {
    // Two perpetually-wrong candidates. The point is the cycle, not the strings:
    // length-7 ("too short") <-> trailing-hyphen. Neither ever satisfies rule 4
    // and rule 2 simultaneously, and the generator keeps swapping between them.
    const slug = turnIndex % 2 === 0 ? 'my-post' : 'my-first-post-';
    return { content: [{ type: 'text', text: slug }], stopReason: 'end_turn' };
  });
}

// §SELF-PERSUASION
// Demo 3: a generator that is CONFIDENTLY WRONG on the first try, paired with a
// same-model critic (makeSelfCritic) that approves it. The slug below breaks
// rule 1 (uppercase) — verifySlug() would reject it — but the critic says "looks
// good", so the loop "passes" on round 1. Outcome is 'accepted', not 'verified',
// and the printout cross-checks against the real verifier to expose that the
// shipped answer is actually broken.
function makeConfidentlyWrongMock(): LLM {
  return new MockLLM(() => ({
    content: [{ type: 'text', text: 'My-Great-Title' }], // breaks rule 1 (uppercase)
    stopReason: 'end_turn',
  }));
}

// ============================================================================
// §REPORT — print one demo's rounds + outcome with computed numbers.
// ============================================================================

function reportRounds(result: ReflectionResult): void {
  for (const r of result.rounds) {
    const status = r.ok ? 'PASS' : 'fail';
    const reasons = r.ok ? '' : `  → ${r.feedback.join('; ')}`;
    console.log(`  round ${r.round}: [${status}] "${r.candidate}"${reasons}`);
  }
  const passedRound = result.rounds.find((r) => r.ok)?.round;
  console.log(
    `  outcome  : ${result.outcome}` +
      (passedRound ? ` (on round ${passedRound} of ${result.rounds.length})` : ` (${result.rounds.length} rounds, none passed)`)
  );
  console.log(`  final    : "${result.finalAnswer}"`);
}

async function main(): Promise<void> {
  const task =
    'Title: "My First Post!" — produce a URL slug satisfying the four formatting rules.';
  console.log(`\n=== Stage 09: reflection & self-correction ===`);
  console.log(`task: ${task}`);
  console.log(`verifier rules:\n${SLUG_RULES.map((r) => `  - ${r}`).join('\n')}`);

  // --- Demo 1: verifier-driven loop converges. -----------------------------
  // createLLM lets a real key drive these too, but the mocks below are what make
  // the demo deterministic; with a key the trajectories will differ (that is the
  // honest caveat — convergence with a real model is likely, not guaranteed).
  console.log(`\n[1] VERIFIER-driven loop (ground-truth signal) — expect convergence:`);
  const converging = createLLM(makeConvergingMock());
  console.log(`    (llm=${converging.name})`);
  const r1 = await reflect(converging, task, verifySlug, /*trustworthy*/ true, 5);
  reportRounds(r1);
  // Independent re-check: prove the "verified" claim with the real verifier, not
  // with the loop's own say-so. A verified answer MUST survive this.
  console.log(`    cross-check: verifySlug("${r1.finalAnswer}").ok = ${verifySlug(r1.finalAnswer).ok}`);

  // --- Demo 2: no usable signal → oscillation, no convergence. --------------
  console.log(`\n[2] OSCILLATION (generator cannot act on feedback) — expect NO convergence:`);
  const oscillating = makeOscillatingMock(); // mock directly: trajectory must be fixed
  const r2 = await reflect(oscillating, task, verifySlug, /*trustworthy*/ true, 5);
  reportRounds(r2);
  const distinct = new Set(r2.rounds.map((r) => r.candidate)).size;
  console.log(
    `    diagnosis: ${r2.rounds.length} rounds cycled through only ${distinct} distinct candidates ` +
      `→ the loop is rewriting in circles, not improving.`
  );

  // --- Demo 3: same-model critic → self-persuasion ships a bug. -------------
  console.log(`\n[3] SELF-PERSUASION (critic = same model, no ground truth) — looks like success, isn't:`);
  const confidentlyWrong = makeConfidentlyWrongMock();
  const selfCritic = makeSelfCritic();
  const r3 = await reflect(confidentlyWrong, task, selfCritic, /*trustworthy*/ false, 5);
  reportRounds(r3);
  // The damning cross-check: the critic "accepted" round 1, but the REAL verifier
  // disagrees. The number below is the gap between felt confidence and truth.
  const truth = verifySlug(r3.finalAnswer);
  console.log(`    cross-check: critic said PASS, but verifySlug("${r3.finalAnswer}").ok = ${truth.ok}`);
  if (!truth.ok) {
    console.log(`    reality    : the shipped answer actually breaks ${truth.failures.length} rule(s): ${truth.failures.join('; ')}`);
  }

  // --- The lesson, restated with the numbers just produced. ----------------
  console.log(`\nlesson:`);
  console.log(`  [1] verifier had ground truth → converged in ${r1.rounds.length} round(s), cross-check holds.`);
  console.log(`  [2] no usable signal → ${r2.rounds.length} rounds, 0 passes, cycled ${distinct} candidates (never converges).`);
  console.log(`  [3] same-model critic → 'accepted' in 1 round, but verifier proves it ships ${verifySlug(r3.finalAnswer).failures.length} broken rule(s).`);
  console.log(`  reflection is only as good as the signal closing the loop. Wire a real verifier.\n`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
