// Stage 05 — context engineering: the context window is a budget, not a memory.
//
// A long-running agent re-sends its whole transcript on every turn (see
// stage01's O(T^2) cost curve), so the only resource that actually constrains a
// session is *tokens of context*. This chapter treats context as a budget and
// shows four mechanisms that buy more useful work per token:
//
//   (1) compaction  — fold old turns into a summary when nearing a budget
//   (2) offloading  — keep huge tool outputs out-of-band, leave a handle
//   (3) KV-cache    — keep a stable prefix so the model's attention cache hits
//   (4) context rot — long contexts degrade in the MIDDLE (open problem)
//
// Run it: `npm run stage05` (fully offline; no key, no network).
//
// HONESTY NOTE on the numbers below: every token count comes from `estimateTokens`
// (core/llm.ts), a ~4-chars-per-token heuristic — NOT a real BPE tokenizer, and
// especially wrong for CJK. The compaction/offloading/cache numbers are real
// arithmetic on that estimate (apples-to-apples before/after). The context-rot
// curve in (4) is a hand-written illustrative toy, NOT a measurement of any
// model; it is labelled as such where it prints.

import { estimateTokens } from './core/llm.js';
import type { Message, UserBlock } from './core/types.js';

// Token cost of one message, mirroring how llm.ts bills the wire shapes: string
// content counts its chars; block arrays count their serialized JSON (so the
// tool_use id / schema overhead is included, not just the visible text).
function messageTokens(m: Message): number {
  if (typeof m.content === 'string') return estimateTokens(m.content);
  return m.content.reduce((n, b) => n + estimateTokens(JSON.stringify(b)), 0);
}

function transcriptTokens(messages: Message[]): number {
  return messages.reduce((n, m) => n + messageTokens(m), 0);
}

// ============================================================================
// (1) COMPACTION — fold old turns into a summary near a token budget.
//
// Invariant that makes compaction safe vs lossy: the summary MUST preserve
// "live" state — unfinished tasks and key decisions the agent still needs to
// act on. Drop those and the agent gets amnesia (re-asks, re-does, contradicts
// an earlier decision). Compressing *transcript bytes* is easy; the skill is
// compressing bytes WITHOUT compressing live state. Below, the summary is built
// mechanically from explicitly-tagged decisions/TODOs so the demo is offline
// and deterministic — a real agent would ask the model to write this summary,
// but the invariant (keep live state) is identical.
// ============================================================================

interface CompactionResult {
  compacted: Message[];
  beforeTokens: number;
  afterTokens: number;
  turnsFolded: number;
}

// Build a long, realistic-ish debugging conversation that grows past a budget.
// Some turns carry load-bearing state ("DECISION:" / "TODO:") that must survive
// compaction; most are chatter that can be thrown away.
function buildLongConversation(): Message[] {
  const messages: Message[] = [
    { role: 'user', content: 'The checkout page 500s intermittently. Find the root cause.' },
  ];
  // Filler turns: the kind of back-and-forth that dominates a real transcript
  // but holds no state worth keeping verbatim once summarized.
  const chatter = [
    'Let me look at the recent deploys and the error rate timeline.',
    'I checked the logs around the spikes; nothing in the web tier stands out yet.',
    'Pulling the slow-query log for the checkout transaction path now.',
    'The p99 on the orders table climbs right before each 500 cluster.',
    'Confirming whether the connection pool is saturating under that load.',
    'Reproduced locally by replaying the traffic sample at 3x rate.',
  ];
  for (let i = 0; i < chatter.length; i++) {
    messages.push({ role: 'assistant', content: [{ type: 'text', text: chatter[i] }] });
    messages.push({ role: 'user', content: 'ok, keep going.' });
  }
  // Load-bearing turns: a decision and an open task, raised EARLY (mid-session).
  // These are exactly what a naive "drop the oldest N turns" strategy would
  // silently destroy. Placing them before the recent window is the whole point:
  // they sit inside the region that gets folded, so the summary must rescue them.
  messages.push({
    role: 'assistant',
    content: [
      {
        type: 'text',
        text: 'DECISION: root cause is connection-pool exhaustion (max=10) during the order-insert; the pool is not the query. TODO: raise pool max to 30 and add a 5s acquire timeout, then re-run the 3x replay to confirm.',
      },
    ],
  });
  // More recent chatter pushes the decision back into "old" territory, so the
  // recent-window (kept verbatim) holds only follow-up noise, not the decision.
  messages.push({ role: 'user', content: 'sounds right — go ahead and try that.' });
  messages.push({ role: 'assistant', content: [{ type: 'text', text: 'Applying the pool config change and starting the replay now.' }] });
  return messages;
}

// Keep the first user turn (the task framing) and the last `keepRecent`
// messages verbatim; fold everything in between into ONE summary turn that
// retains tagged decisions/TODOs.
function compactConversation(messages: Message[], keepRecent: number): CompactionResult {
  const beforeTokens = transcriptTokens(messages);

  // Always preserve the original task framing (messages[0]) and a recent window;
  // the middle is the compaction target.
  const head = messages[0];
  const tail = messages.slice(messages.length - keepRecent);
  const middle = messages.slice(1, messages.length - keepRecent);

  // Extract live state from the middle. This is the invariant in code: anything
  // tagged DECISION/TODO is lifted verbatim into the summary; untagged chatter
  // is collapsed to a single count. A real agent delegates this extraction to
  // the model, but must hold the same contract.
  const liveState: string[] = [];
  for (const m of middle) {
    const text =
      typeof m.content === 'string'
        ? m.content
        : m.content.map((b) => (b.type === 'text' ? b.text : '')).join(' ');
    for (const line of text.split(/(?<=\.)\s+/)) {
      if (/\b(DECISION|TODO):/.test(line)) liveState.push(line.trim());
    }
  }

  const summaryBody =
    `[compacted summary of ${middle.length} earlier turns]\n` +
    `Preserved live state (must not be lost):\n` +
    (liveState.length ? liveState.map((s) => `- ${s}`).join('\n') : '- (none)') +
    `\nRemaining ${middle.length - liveState.length} turns were investigation chatter with no open state.`;

  const summaryTurn: Message = { role: 'user', content: summaryBody };
  const compacted = [head, summaryTurn, ...tail];

  return {
    compacted,
    beforeTokens,
    afterTokens: transcriptTokens(compacted),
    turnsFolded: middle.length,
  };
}

function demoCompaction(): void {
  console.log('--- (1) Compaction: fold old turns into a summary near a budget ---\n');
  const conversation = buildLongConversation();
  // Treat this as the budget pressure point: pretend the window is small enough
  // that this transcript is "too big" and must shrink before the next turn.
  const BUDGET_TOKENS = 200;
  const before = transcriptTokens(conversation);
  console.log(`transcript    : ${conversation.length} messages, ${before} tokens (budget ${BUDGET_TOKENS})`);
  console.log(`over budget?  : ${before > BUDGET_TOKENS ? 'yes — compact before next turn' : 'no'}`);

  const r = compactConversation(conversation, /* keepRecent */ 2);
  const ratio = r.afterTokens / r.beforeTokens;
  console.log(`folded turns  : ${r.turnsFolded} (kept first task + last 2 turns verbatim)`);
  console.log(`tokens        : ${r.beforeTokens} → ${r.afterTokens}  (${(ratio * 100).toFixed(0)}% of original, ${(1 / ratio).toFixed(1)}x smaller)`);

  // Prove the invariant held: the decision + TODO survived the fold.
  const summary = r.compacted[1];
  const summaryText = typeof summary.content === 'string' ? summary.content : '';
  const keptDecision = /DECISION: root cause is connection-pool/.test(summaryText);
  const keptTodo = /TODO: raise pool max/.test(summaryText);
  console.log(`invariant     : decision preserved=${keptDecision}, open TODO preserved=${keptTodo} (else: agent amnesia)\n`);
}

// ============================================================================
// (2) OFFLOADING — keep a huge tool result out of context, leave a handle.
//
// A tool can return far more than is worth carrying every turn (a 4000-line
// directory listing, a giant file, an API dump). Stuffing it into the transcript
// taxes EVERY subsequent turn (it gets re-sent each time). Instead: write the
// full payload to an out-of-band store (here an in-memory Map standing in for the
// filesystem) and put only a handle + a short summary into context. The agent can
// re-read on demand via the handle, but doesn't pay for it until it does.
// ============================================================================

// In-memory stand-in for "files on disk". The point is that this lives OUTSIDE
// the context window, so its size does not enter the per-turn token bill.
const offloadStore = new Map<string, string>();

interface OffloadResult {
  handle: string;
  contextBlock: UserBlock;
  fullTokens: number;
  contextTokens: number;
}

// Simulate a tool that lists a huge tree, then offload the result.
function offloadLargeObservation(toolUseId: string, lineCount: number): OffloadResult {
  // Fabricate a believable multi-thousand-line `ls -R` style observation.
  const lines: string[] = [];
  for (let i = 0; i < lineCount; i++) {
    lines.push(`drwxr-xr-x  src/module_${String(i).padStart(4, '0')}/component_${i % 37}.ts`);
  }
  const fullPayload = lines.join('\n');
  const fullTokens = estimateTokens(fullPayload);

  // Stash full payload out-of-band, keyed by a stable handle the agent can cite.
  const handle = `store://ls/${toolUseId}`;
  offloadStore.set(handle, fullPayload);

  // What actually goes back into context: a handle + a cheap, computed summary.
  // The summary must be enough for the agent to decide whether to re-read.
  const summary =
    `[offloaded] ${lineCount} entries (${fullTokens} tokens) stored at ${handle}. ` +
    `Top-level dirs: ${lineCount} files under src/. Re-read with read_store("${handle}") if you need specifics.`;
  const contextBlock: UserBlock = { type: 'tool_result', toolUseId, content: summary };

  return {
    handle,
    contextBlock,
    fullTokens,
    contextTokens: estimateTokens(JSON.stringify(contextBlock)),
  };
}

function demoOffloading(): void {
  console.log('--- (2) Offloading: store huge observation, keep a handle in context ---\n');
  const r = offloadLargeObservation('call_ls_1', /* lineCount */ 4000);
  const saved = r.fullTokens - r.contextTokens;
  console.log(`tool output   : 4000-line listing = ${r.fullTokens} tokens if inlined`);
  console.log(`in context    : handle + summary  = ${r.contextTokens} tokens`);
  console.log(`saved / turn  : ${saved} tokens (and this saving recurs EVERY turn, since transcript is re-sent)`);

  // Show the data is recoverable: the agent can pull the full payload back when
  // it actually needs it, paying the cost only at that moment.
  const recovered = offloadStore.get(r.handle) ?? '';
  console.log(`recoverable?  : read_store("${r.handle}") → ${recovered.split('\n').length} lines (full fidelity on demand)\n`);
}

// ============================================================================
// (3) KV-CACHE PREFIX STABILITY — a stable prefix is a cheap prefix.
//
// During generation the model caches attention key/value tensors for the tokens
// it has already processed. On the next call, the provider can REUSE that cache
// for the longest IDENTICAL prefix and only re-process the diverging suffix.
// Cached input tokens are billed cheaper and add ~no latency. The catch: the
// cache keys on an EXACT prefix match. Reorder your tools, edit one word of the
// system prompt, or inject a timestamp at the top, and the prefix changes from
// that point on — the cache misses and you re-pay for everything after the edit.
//
// We model the cacheable prefix with a simple incremental hash over the
// serialized [system, ...tools] segments. Same segments in same order → same
// hash (HIT). Any change → different hash (MISS). The hash is a stand-in for
// "what the provider keys its cache on"; it is not the provider's real algorithm.
// ============================================================================

interface PrefixConfig {
  system: string;
  tools: { name: string; description: string }[];
}

// Deterministic, dependency-free 32-bit FNV-1a over the prefix segments. We hash
// segment-by-segment so we can also report the longest shared prefix length —
// which is what determines how much cache survives an edit.
function hashSegments(segments: string[]): string {
  let h = 0x811c9dc5;
  for (const seg of segments) {
    for (let i = 0; i < seg.length; i++) {
      h ^= seg.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    h ^= 0x0a; // segment delimiter, so [AB] and [A,B] don't collide
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function prefixSegments(cfg: PrefixConfig): string[] {
  return [cfg.system, ...cfg.tools.map((t) => JSON.stringify(t))];
}

// Longest common prefix (in segments) between two configs — everything from the
// first divergence onward is cache-invalidated.
function sharedPrefixLen(a: string[], b: string[]): number {
  let n = 0;
  while (n < a.length && n < b.length && a[n] === b[n]) n++;
  return n;
}

function demoKvCache(): void {
  console.log('--- (3) KV-cache prefix stability: same prefix → cache hit, edit → miss ---\n');

  const base: PrefixConfig = {
    system: 'You are a coding agent. Use tools precisely.',
    tools: [
      { name: 'read_file', description: 'Read a file by path.' },
      { name: 'write_file', description: 'Write content to a path.' },
      { name: 'run_tests', description: 'Run the test suite.' },
    ],
  };

  const baseSeg = prefixSegments(base);
  const baseHash = hashSegments(baseSeg);
  const baseTokens = baseSeg.reduce((n, s) => n + estimateTokens(s), 0);
  console.log(`base prefix   : system + ${base.tools.length} tools = ${baseTokens} tokens, hash=${baseHash}`);

  // Case A: identical prefix on the next turn → full cache hit.
  const identical = prefixSegments({ ...base, tools: [...base.tools] });
  const hitHash = hashSegments(identical);
  console.log(`turn 2 (same) : hash=${hitHash} → ${hitHash === baseHash ? 'HIT' : 'MISS'} — all ${baseTokens} prefix tokens served from cache (cheap, ~0 added latency)`);

  // Case B: one word edited in the system prompt → miss from segment 0.
  const editedSystem = prefixSegments({ ...base, system: base.system.replace('precisely', 'carefully') });
  const editHash = hashSegments(editedSystem);
  const editShared = sharedPrefixLen(baseSeg, editedSystem);
  console.log(`edit system   : hash=${editHash} → ${editHash === baseHash ? 'HIT' : 'MISS'} — diverges at segment ${editShared}/${baseSeg.length}; re-pay ${baseTokens} tokens (whole prefix invalid)`);

  // Case C: same tools, reordered → miss from the first moved tool onward.
  const reordered = prefixSegments({
    ...base,
    tools: [base.tools[1], base.tools[0], base.tools[2]],
  });
  const reorderHash = hashSegments(reordered);
  const reorderShared = sharedPrefixLen(baseSeg, reordered);
  const lostTokens = baseSeg.slice(reorderShared).reduce((n, s) => n + estimateTokens(s), 0);
  console.log(`swap 2 tools  : hash=${reorderHash} → ${reorderHash === baseHash ? 'HIT' : 'MISS'} — diverges at segment ${reorderShared}/${baseSeg.length}; re-pay ${lostTokens} tokens after the swap`);

  console.log(`\nlesson        : the longer the prefix you keep byte-identical, the more you cache.`);
  console.log(`                put volatile content (timestamps, retrieved docs) LAST, not in the prefix.\n`);
}

// ============================================================================
// (4) CONTEXT ROT — long contexts degrade, worst in the MIDDLE.
//
// Empirically, a model's ability to USE a fact does not stay flat as context
// grows: accuracy is high for facts near the start and end, and dips in the
// middle ("lost in the middle"). This is a frontier/open problem — mitigations
// (reranking, putting the needle last, shorter context) are active research.
//
// !!! ILLUSTRATIVE TOY — NOT A MEASUREMENT !!!
// The "retrieval score" below is a HAND-WRITTEN U-shaped function of position. It
// does NOT call any model and is NOT calibrated against any benchmark. It exists
// only to make the SHAPE of the problem concrete. Real degradation must be
// measured with a real model on a real needle-in-a-haystack benchmark; the
// numbers here are a cartoon of the phenomenon, not evidence of it.
// ============================================================================

// U-shaped recall vs relative position (0=start .. 1=end): strong at the edges,
// a deliberate trough in the middle. Hand-tuned for shape only. (See toy warning.)
function illustrativeRecallAt(relativePosition: number): number {
  const distanceFromMiddle = Math.abs(relativePosition - 0.5); // 0 at center, 0.5 at edges
  const edgeBoost = distanceFromMiddle * 2; // 0..1
  const recall = 0.45 + 0.5 * edgeBoost; // trough ~0.45 in middle, ~0.95 at edges
  return Math.min(0.99, recall);
}

function demoContextRot(): void {
  console.log('--- (4) Context rot: needle-in-haystack, recall dips in the middle ---');
  console.log('    *** ILLUSTRATIVE TOY: hand-written U-curve, NOT a real model / benchmark ***\n');

  const positions = 11; // sample the needle at 11 evenly spaced depths
  console.log('depth   recall  bar');
  for (let i = 0; i < positions; i++) {
    const rel = i / (positions - 1);
    const recall = illustrativeRecallAt(rel);
    const bar = '█'.repeat(Math.round(recall * 30));
    const depthPct = `${String(Math.round(rel * 100)).padStart(3)}%`;
    console.log(`${depthPct}    ${(recall * 100).toFixed(0).padStart(3)}%   ${bar}`);
  }
  const mid = illustrativeRecallAt(0.5);
  const edge = illustrativeRecallAt(0);
  console.log(`\nshape         : edge ~${(edge * 100).toFixed(0)}% vs middle ~${(mid * 100).toFixed(0)}% — same fact, worse recall mid-context`);
  console.log('takeaway      : if a fact must be used, place it near the END (or re-inject it); do not bury it mid-context.');
  console.log('               but VERIFY with a real model — this curve is a teaching cartoon, not data.\n');
}

// ----------------------------------------------------------------------------

function main(): void {
  console.log('\n=== Stage 05: context engineering (context as budget) ===');
  console.log('    tokens via estimateTokens (~4 chars/token heuristic, not a real tokenizer)\n');
  demoCompaction();
  demoOffloading();
  demoKvCache();
  demoContextRot();
}

main();
