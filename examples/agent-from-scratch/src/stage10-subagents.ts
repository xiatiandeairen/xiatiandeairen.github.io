// Stage 10 — sub-agents and context isolation: spend a child's context, keep the
// parent's clean.
//
// A single agent that does deep work pays the O(T^2) transcript tax from stage01:
// every intermediate observation it generates is re-sent on EVERY subsequent
// turn. A sub-task that produces a lot of noisy middle steps (grep 200 files,
// read 30 of them, reason over each) can blow the parent's context budget even
// though the parent only ever needed the *conclusion*.
//
// The fix is a second agent with its OWN context window. The orchestrator spawns
// a sub-agent, hands it a scoped goal, lets it burn its own transcript on the
// messy investigation, and receives back ONLY the final answer. The sub-agent's
// intermediate observations never enter the parent's messages, so they never get
// re-sent. This is the mechanism behind "research" / "explore" sub-agents.
//
// This chapter shows two layers (orchestrator + sub-agent) both running the real
// agent loop against scripted MockLLMs, then does the honest token arithmetic:
// what the parent context WOULD have grown by if the investigation lived inline,
// vs what it ACTUALLY grew by with isolation. It also demos the two ways
// isolation bites back:
//   - sub-agents cannot see each other's intermediate findings (the cost of the
//     wall you put up), so they redo overlapping work;
//   - a sub-agent whose conclusion is too thin forces the parent to spawn it
//     AGAIN to ask a follow-up — round-trips that erase the savings.
//
// Run it: `npx tsx src/stage10-subagents.ts` (fully offline; no key, no network).
//
// HONESTY NOTE on the numbers: every token count comes from `estimateTokens`
// (core/llm.ts), a ~4-chars-per-token heuristic — NOT a real BPE tokenizer, and
// especially wrong for CJK. All before/after comparisons are real arithmetic on
// that same estimate (apples-to-apples), computed from the actual messages the
// loops produced, not hand-typed.

import { createLLM, MockLLM, estimateTokens } from './core/llm.js';
import type {
  AssistantBlock,
  LLM,
  Message,
  ToolSpec,
  UserBlock,
} from './core/types.js';

// ----------------------------------------------------------------------------
// A minimal agent loop, copied here on purpose.
//
// stage01 exports `runAgent`, but importing any stageNN file would execute its
// top-level main() as a side effect. So we inline a trimmed loop. It is the same
// shape as stage01's (call model → if tool_use, run tools, feed results back,
// repeat; else return), with one addition this chapter needs: it returns the
// FULL message transcript, because the whole point is to measure how big a
// context a run produced and whether that context escapes into the parent.
// ----------------------------------------------------------------------------

interface AgentRun {
  answer: string;
  // The complete conversation this loop built: user goal, assistant turns,
  // tool_result turns. Token-accounting below reads this directly so the printed
  // numbers are derived from real messages, not guessed.
  transcript: Message[];
  turns: number;
}

function textOf(content: AssistantBlock[]): string {
  return content
    .filter((b): b is Extract<AssistantBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('');
}

// Token cost of one message, mirroring how llm.ts bills the wire shapes (string
// content counts its chars; block arrays count serialized JSON, so tool_use ids
// and tool_result payloads are included, not just visible prose). Same function
// as stage05 uses — kept local to avoid the cross-stage import hazard.
function messageTokens(m: Message): number {
  if (typeof m.content === 'string') return estimateTokens(m.content);
  return m.content.reduce((n, b) => n + estimateTokens(JSON.stringify(b)), 0);
}

function transcriptTokens(messages: Message[]): number {
  return messages.reduce((n, m) => n + messageTokens(m), 0);
}

async function runLoop(
  llm: LLM,
  tools: ToolSpec[],
  toolImpls: Record<string, (input: Record<string, unknown>) => Promise<string>>,
  goal: string,
  system: string,
  maxTurns: number
): Promise<AgentRun> {
  const messages: Message[] = [{ role: 'user', content: goal }];

  for (let turn = 0; turn < maxTurns; turn++) {
    const assistant = await llm.generate({ system, messages, tools });
    messages.push({ role: 'assistant', content: assistant.content });

    if (assistant.stopReason !== 'tool_use') {
      return { answer: textOf(assistant.content), transcript: messages, turns: turn + 1 };
    }

    // Run every requested tool, feed all results back as ONE user turn.
    const results: UserBlock[] = [];
    for (const block of assistant.content) {
      if (block.type !== 'tool_use') continue;
      const impl = toolImpls[block.name];
      if (!impl) {
        results.push({ type: 'tool_result', toolUseId: block.id, content: `unknown tool: ${block.name}`, isError: true });
        continue;
      }
      results.push({ type: 'tool_result', toolUseId: block.id, content: await impl(block.input) });
    }
    messages.push({ role: 'user', content: results });
  }

  // Hard ceiling (stage01's #1 reliability knob). A sub-agent that never stops
  // must not bill forever — and crucially must not stall the parent that awaits it.
  return { answer: '[sub-agent stopped: max turns reached]', transcript: messages, turns: maxTurns };
}

// ----------------------------------------------------------------------------
// The sub-agent's tools and noisy domain.
//
// The whole reason to isolate this work is that it produces a LOT of bytes the
// parent does not care about. `grep_codebase` returns hundreds of hit lines and
// `read_file` returns whole files — exactly the kind of fat observations that, if
// left in the parent transcript, get re-sent every turn for the rest of the run.
// ----------------------------------------------------------------------------

// A fabricated repo of believable file contents, large enough that inlining the
// reads would clearly dominate a context window.
const FAKE_FILES: Record<string, string> = {
  'src/auth/session.ts': Array.from({ length: 40 }, (_, i) =>
    `  // L${i}: session token is signed with HS256 and refreshed every 15m; legacy cookie path still honored for v1 clients`
  ).join('\n'),
  'src/auth/login.ts': Array.from({ length: 35 }, (_, i) =>
    `  // L${i}: login handler validates password via bcrypt, then mints a session via session.ts mintToken()`
  ).join('\n'),
  'src/auth/legacy_cookie.ts': Array.from({ length: 50 }, (_, i) =>
    `  // L${i}: DEPRECATED v1 cookie auth; falls back to plaintext compare — this is the vulnerability`
  ).join('\n'),
};

const SUBAGENT_TOOLS: ToolSpec[] = [
  {
    name: 'grep_codebase',
    description: 'Search the codebase for a regex. Returns every matching file:line. Output can be very large.',
    inputSchema: {
      type: 'object',
      properties: { pattern: { type: 'string', description: 'regex to search for' } },
      required: ['pattern'],
    },
  },
  {
    name: 'read_file',
    description: 'Read a whole file by path. Returns the full contents.',
    inputSchema: {
      type: 'object',
      properties: { path: { type: 'string', description: 'file path to read' } },
      required: ['path'],
    },
  },
];

const SUBAGENT_TOOL_IMPLS: Record<string, (input: Record<string, unknown>) => Promise<string>> = {
  // Returns a deliberately fat result: many hit lines across the fake repo. This
  // is the "noise" we want to keep out of the parent.
  async grep_codebase(input) {
    const pattern = String(input.pattern ?? '');
    const re = new RegExp(pattern, 'i');
    const hits: string[] = [];
    for (const [path, body] of Object.entries(FAKE_FILES)) {
      body.split('\n').forEach((line, i) => {
        if (re.test(line)) hits.push(`${path}:${i + 1}: ${line.trim()}`);
      });
    }
    return hits.length ? hits.join('\n') : `no matches for /${pattern}/`;
  },
  async read_file(input) {
    const path = String(input.path ?? '');
    return FAKE_FILES[path] ?? `error: no such file ${path}`;
  },
};

// ----------------------------------------------------------------------------
// Sub-agent mock policy: a scripted 3-turn investigation that ends in a
// substantial conclusion. turn 0 greps, turn 1 reads the suspicious file, turn 2
// answers. The answer is RICH on purpose — it pre-empts the parent's likely
// follow-up (which file, which line, what fix), so the parent does not have to
// re-spawn. The thin-answer failure mode below shows the opposite.
// ----------------------------------------------------------------------------

function makeRichSubAgent(): MockLLM {
  return new MockLLM((_opts, turnIndex) => {
    if (turnIndex === 0) {
      return {
        content: [
          { type: 'text', text: 'Searching for auth-related code.' },
          { type: 'tool_use', id: 'sa_grep', name: 'grep_codebase', input: { pattern: 'auth|cookie|token|password' } },
        ],
      };
    }
    if (turnIndex === 1) {
      return {
        content: [
          { type: 'text', text: 'The legacy cookie path looks suspicious; reading it in full.' },
          { type: 'tool_use', id: 'sa_read', name: 'read_file', input: { path: 'src/auth/legacy_cookie.ts' } },
        ],
      };
    }
    // Final turn: a self-sufficient conclusion. Note it answers "where", "why",
    // and "how to fix" — the questions the parent would otherwise re-ask.
    return {
      content: [
        {
          type: 'text',
          text:
            'CONCLUSION: The auth vulnerability is in src/auth/legacy_cookie.ts (the deprecated v1 cookie path). ' +
            'It compares the cookie value with a plaintext compare instead of a constant-time check, and bypasses ' +
            'the HS256-signed session flow used everywhere else. Fix: delete the legacy cookie fallback and require ' +
            'all clients to use the signed session from session.ts mintToken(). Confidence: high (1 file, ~50 lines).',
        },
      ],
      stopReason: 'end_turn',
    };
  });
}

// A DELIBERATELY thin sub-agent for the second failure demo: it does the same
// expensive investigation but returns a one-liner with no actionable detail.
function makeThinSubAgent(): MockLLM {
  return new MockLLM((_opts, turnIndex) => {
    if (turnIndex === 0) {
      return { content: [{ type: 'tool_use', id: 'sa_grep', name: 'grep_codebase', input: { pattern: 'auth|cookie' } }] };
    }
    if (turnIndex === 1) {
      return { content: [{ type: 'tool_use', id: 'sa_read', name: 'read_file', input: { path: 'src/auth/legacy_cookie.ts' } }] };
    }
    // Too thin: technically answers "is there a vuln?" but gives the parent
    // nothing to act on, guaranteeing a follow-up round-trip.
    return { content: [{ type: 'text', text: 'Yes, there is an auth issue somewhere in the codebase.' }], stopReason: 'end_turn' };
  });
}

// ----------------------------------------------------------------------------
// The orchestrator's view of spawning a sub-agent.
//
// `spawnSubAgent` runs a child loop to completion and returns ONLY its answer
// string plus accounting. The child's transcript is captured for measurement but
// is NOT returned to the parent's message list — that omission is the entire
// isolation mechanism. Compare `subAgentContextTokens` (what the child spent,
// thrown away) against `returnedTokens` (what crosses the wall into the parent).
// ----------------------------------------------------------------------------

interface SpawnResult {
  answer: string;
  subAgentTurns: number;
  subAgentContextTokens: number; // peak tokens the child's own context reached
  returnedTokens: number; // tokens the parent actually absorbs (the answer only)
  childTranscript: Message[]; // kept for the shared-findings demo, not given to parent
}

async function spawnSubAgent(subLlm: LLM, goal: string): Promise<SpawnResult> {
  const run = await runLoop(
    subLlm,
    SUBAGENT_TOOLS,
    SUBAGENT_TOOL_IMPLS,
    goal,
    'You are a focused investigation sub-agent. Do the digging, then report a single self-contained conclusion.',
    /* maxTurns */ 6
  );

  // The child's full context (grep dumps + file reads) — this is what isolation
  // keeps OUT of the parent. We measure it to quantify the savings.
  const subAgentContextTokens = transcriptTokens(run.transcript);

  // What the parent gets: just the conclusion, wrapped as one tool_result block
  // (the orchestrator models "spawn sub-agent" as a tool call, so the answer
  // returns the same way any tool result would).
  const returnedBlock: UserBlock = { type: 'tool_result', toolUseId: 'spawn_1', content: run.answer };
  const returnedTokens = estimateTokens(JSON.stringify(returnedBlock));

  return {
    answer: run.answer,
    subAgentTurns: run.turns,
    subAgentContextTokens,
    returnedTokens,
    childTranscript: run.transcript,
  };
}

// ----------------------------------------------------------------------------
// DEMO 1 — isolation: the parent pays for the conclusion, not the investigation.
// ----------------------------------------------------------------------------

async function demoIsolation(): Promise<SpawnResult> {
  console.log('--- (1) Isolation: parent absorbs the conclusion, not the middle steps ---\n');

  const spawn = await spawnSubAgent(
    makeRichSubAgent(),
    'Find the authentication vulnerability in the codebase and report exactly where it is and how to fix it.'
  );

  console.log(`sub-agent run     : ${spawn.subAgentTurns} turns (grep → read → conclude), in its OWN context`);
  console.log(`sub-agent context : ${spawn.subAgentContextTokens} tokens at peak (grep dump + full file read live HERE)`);
  console.log(`returned to parent: ${spawn.returnedTokens} tokens (the conclusion only)`);

  // The honest comparison. If the investigation had run INLINE in the parent,
  // every message the child produced would have entered the parent's transcript
  // and been re-sent on every later parent turn. The conservative lower bound on
  // the parent's growth in that world is the child's full transcript size; the
  // real bill is larger because of the O(T^2) re-send. We report the lower bound
  // and label it as such.
  const inlineCost = spawn.subAgentContextTokens; // would-be one-time growth if inlined
  const isolatedCost = spawn.returnedTokens; // actual one-time growth with isolation
  const saved = inlineCost - isolatedCost;
  const ratio = isolatedCost / inlineCost;

  console.log('');
  console.log(`if INLINE in parent : +${inlineCost} tokens of parent context (and re-sent every later turn — O(T^2))`);
  console.log(`with ISOLATION      : +${isolatedCost} tokens of parent context (one conclusion, re-sent cheaply)`);
  console.log(`parent saved        : ${saved} tokens up front (${(ratio * 100).toFixed(0)}% of the inline cost crosses the wall; ${(1 / ratio).toFixed(1)}x leaner parent)`);
  console.log(`note                : "inline cost" is a LOWER bound — re-sending across N turns makes the real gap bigger.\n`);

  return spawn;
}

// ----------------------------------------------------------------------------
// DEMO 2 — failure mode A: the wall blocks shared intermediate findings.
//
// Isolation is not free. Two sub-agents spawned for related goals each run their
// own investigation in their own context. Sub-agent B cannot see that sub-agent A
// already grepped the codebase and already knows where the auth code lives —
// because A's intermediate findings never left A's context. So B redoes the grep
// and the read from scratch. The duplicated tokens are the price of the wall.
// ----------------------------------------------------------------------------

async function demoNoSharedFindings(firstSpawn: SpawnResult): Promise<void> {
  console.log('--- (2) Failure mode A: sub-agents cannot share intermediate findings ---\n');

  // Sub-agent B is spawned for a RELATED goal (e.g. "now also check the session
  // refresh path"). It has no access to A's transcript, so it re-greps and
  // re-reads — work A already did and threw away behind the isolation wall.
  const secondSpawn = await spawnSubAgent(
    makeRichSubAgent(),
    'Audit the session refresh path for the same class of vulnerability.'
  );

  // Quantify the duplication: find tool_result messages that appear in BOTH
  // child transcripts (the grep dump and the file read are byte-identical because
  // both agents ran the same tools on the same fake repo).
  const blocksOf = (t: Message[]): string[] =>
    t.flatMap((m) =>
      typeof m.content === 'string'
        ? []
        : m.content.filter((b) => b.type === 'tool_result').map((b) => JSON.stringify(b))
    );

  const aBlocks = new Set(blocksOf(firstSpawn.childTranscript));
  const bBlocks = blocksOf(secondSpawn.childTranscript);
  const duplicated = bBlocks.filter((b) => aBlocks.has(b));
  const duplicatedTokens = duplicated.reduce((n, b) => n + estimateTokens(b), 0);

  console.log(`sub-agent A did   : grep + read (then discarded its context behind the wall)`);
  console.log(`sub-agent B redid : ${duplicated.length} identical tool observation(s) A already had`);
  console.log(`wasted re-work    : ${duplicatedTokens} tokens of tool output B regenerated from scratch`);
  console.log(`why               : B's context cannot reach A's intermediate findings — that is the cost of isolation`);
  console.log(`mitigation        : a shared scratchpad/store (stage05 offloading) the parent passes to each child,`);
  console.log(`                    OR have the parent fold A's conclusion into B's goal. Not free either way.\n`);
}

// ----------------------------------------------------------------------------
// DEMO 3 — failure mode B: a thin conclusion forces a re-spawn round-trip.
//
// The savings from isolation assume the sub-agent returns a self-sufficient
// answer. If the conclusion is too thin, the parent must spawn the sub-agent
// AGAIN to ask the follow-up — paying a second full investigation. Two thin
// round-trips can cost more than one rich spawn.
// ----------------------------------------------------------------------------

async function demoThinConclusion(richSpawn: SpawnResult): Promise<void> {
  console.log('--- (3) Failure mode B: a too-thin conclusion triggers re-questioning ---\n');

  // First spawn: thin answer. The parent reads it and finds it unactionable.
  const thin1 = await spawnSubAgent(
    makeThinSubAgent(),
    'Find the authentication vulnerability in the codebase and report exactly where it is and how to fix it.'
  );

  // The parent's acceptance check: a usable conclusion must name a location and
  // a fix. This is the orchestrator's job — decide whether the child's answer is
  // good enough to proceed, or whether it must re-spawn. (A real orchestrator
  // would let the model judge; here we make the criterion explicit and testable.)
  const namesLocation = /\.(ts|js|py|go)\b/.test(thin1.answer) || /\bsrc\//.test(thin1.answer);
  const namesFix = /\bfix\b|\breplace\b|\bdelete\b|\brequire\b|\bremove\b/i.test(thin1.answer);
  const isActionable = namesLocation && namesFix;

  console.log(`spawn #1 (thin)   : "${thin1.answer}"`);
  console.log(`parent accepts?   : ${isActionable ? 'yes' : 'NO'} (names location=${namesLocation}, names fix=${namesFix}) → must re-spawn`);

  // Because the wall threw away the child's context, the follow-up is NOT a cheap
  // clarification — the parent must spawn a fresh investigation that repeats the
  // grep + read. We model the re-spawn with the rich agent (the parent asks a
  // sharper question this time and gets the real answer).
  const thin2 = await spawnSubAgent(
    makeRichSubAgent(),
    'Re-investigate: state the exact FILE and the exact FIX for the auth vulnerability, nothing vaguer.'
  );

  const twoThinRoundTrips = thin1.subAgentContextTokens + thin2.subAgentContextTokens;
  const oneRichSpawn = richSpawn.subAgentContextTokens;
  const overhead = twoThinRoundTrips - oneRichSpawn;

  console.log(`spawn #2 (re-ask) : ${thin2.subAgentTurns} turns, FULL re-investigation (the wall discarded #1's work)`);
  console.log('');
  console.log(`two thin spawns   : ${twoThinRoundTrips} tokens of total investigation work`);
  console.log(`one rich spawn    : ${oneRichSpawn} tokens (had it answered well the first time)`);
  console.log(`thinness overhead : ${overhead} extra tokens (${overhead > 0 ? 'a thin child is worse than no isolation savings on that task' : 'no overhead'})`);
  console.log(`lesson            : isolation pays ONLY if the conclusion is self-sufficient; reward completeness in the sub-agent's brief.\n`);
}

// ----------------------------------------------------------------------------

async function main(): Promise<void> {
  // createLLM is called only to surface the standard offline-mode banner and to
  // confirm the env policy; the two layers each drive their own scripted MockLLM
  // (a single global mock cannot play both orchestrator and sub-agent roles).
  const probe = createLLM(makeRichSubAgent());
  console.log(`\n=== Stage 10: sub-agents & context isolation (llm=${probe.name}) ===`);
  console.log('    tokens via estimateTokens (~4 chars/token heuristic, not a real tokenizer)');
  console.log('    two real agent loops (orchestrator + sub-agent); the model is scripted\n');

  const richSpawn = await demoIsolation();
  await demoNoSharedFindings(richSpawn);
  await demoThinConclusion(richSpawn);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
