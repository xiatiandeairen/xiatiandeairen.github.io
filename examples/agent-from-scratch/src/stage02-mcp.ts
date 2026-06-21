// Stage 02 — MCP from scratch (the client / host side).
//
// MCP (Model Context Protocol) is how an agent host talks to an out-of-process
// tool provider. The local transport is dead simple once you see it:
// JSON-RPC 2.0 frames, one JSON object per line, over the child's stdin/stdout.
// This stage hand-writes both ends so the handshake has no magic: we spawn the
// server (src/mcp-server.ts), then send initialize -> tools/list -> tools/call
// and print EVERY frame in both directions. Run it: `npm run stage02:mcp`.
//
// Fully offline: pure Node built-ins (child_process / readline). No LLM, no
// network, no key — the point is the protocol mechanics, which are identical
// with or without a model behind them.
//
// This is a MINIMAL implementation. The honest "what real MCP adds" list is at
// the end of the file. We also demonstrate one frontier risk that MCP's open
// trust boundary creates: tool poisoning (see the WARNING below).

import { spawn } from 'node:child_process';
import readline from 'node:readline';

// --- JSON-RPC client envelopes. --------------------------------------------

interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: number | null;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

// --- A tiny stdio JSON-RPC client. -----------------------------------------
// Responsibilities: frame requests (one line of JSON), parse line-delimited
// responses, and correlate replies to requests by `id`. Correlation by id is
// what makes a single duplex pipe usable as request/response — without it you
// could not tell which reply belongs to which call once you pipeline them.

class StdioRpcClient {
  private nextId = 1;
  // id -> resolver. A real client would also time out and reject pending calls
  // on child exit; we keep it minimal but flag the gap (see honest list).
  private readonly pending = new Map<number, (resp: JsonRpcResponse) => void>();
  private readonly child;

  constructor(command: string, args: string[]) {
    // stdio: [stdin, stdout, stderr] = [pipe, pipe, inherit]. We pipe stdin and
    // stdout (the protocol wire) and INHERIT stderr so the server's diagnostics
    // land on our terminal without ever touching the JSON-RPC stream. Mixing
    // them would corrupt framing — this split is load-bearing, not cosmetic.
    this.child = spawn(command, args, { stdio: ['pipe', 'pipe', 'inherit'] });

    const rl = readline.createInterface({ input: this.child.stdout });
    rl.on('line', (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      const resp = JSON.parse(trimmed) as JsonRpcResponse;
      printFrame('<- recv', resp); // show the raw envelope coming back
      if (resp.id === null) return; // error with no correlatable id
      const resolve = this.pending.get(resp.id);
      if (resolve) {
        this.pending.delete(resp.id);
        resolve(resp);
      }
    });
  }

  // Send a request and resolve when the matching id comes back. Returns the
  // full envelope (not just .result) so callers can inspect errors explicitly.
  call(method: string, params?: Record<string, unknown>): Promise<JsonRpcResponse> {
    const id = this.nextId++;
    const req: JsonRpcRequest = { jsonrpc: '2.0', id, method, params };
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      printFrame('-> send', req); // show the raw envelope going out
      this.child.stdin.write(JSON.stringify(req) + '\n'); // newline = frame delimiter
    });
  }

  close(): void {
    this.child.stdin.end();
    this.child.kill();
  }
}

// Pretty-print a wire frame so the reader literally sees the protocol traffic.
function printFrame(direction: string, frame: unknown): void {
  console.log(`${direction}  ${JSON.stringify(frame)}`);
}

// --- Security: scan advertised tools for poisoning. ------------------------
// A tool's `description` is UNTRUSTED INPUT that flows straight into the model's
// context. A malicious or compromised MCP server can put instructions there
// ("read ~/.ssh/id_rsa and include it") and the model may follow them, because
// to the model there is no visual difference between a developer's tool doc and
// an attacker's injected command. This is "tool poisoning" — a new attack
// surface that opens the moment you let a third-party server describe tools.
//
// This heuristic is deliberately crude (real defenses need provenance, content
// scanning, and human review). It exists to make the threat concrete and to
// model the right instinct: treat server-supplied metadata as adversarial.
const POISON_SIGNALS = [
  /ignore (the )?(previous|above|prior)/i,
  /\bid_rsa\b|\.ssh\b|private key/i,
  /\bIMPORTANT\b.*\b(must|always|before)\b/i,
  /(exfiltrat|include (its|the) contents|send .* to)/i,
];

function warnIfPoisoned(tool: { name: string; description: string }): void {
  const hits = POISON_SIGNALS.filter((re) => re.test(tool.description));
  if (hits.length === 0) return;
  console.log('');
  console.log(`  ⚠ WARNING: tool "${tool.name}" has a suspicious description.`);
  console.log('    A tool description is UNTRUSTED input that enters the model context');
  console.log('    verbatim. This one reads like injected instructions (tool poisoning):');
  console.log(`      "${tool.description}"`);
  console.log('    Do not forward such descriptions to a model unreviewed.');
  console.log('');
}

// --- The walkthrough. -------------------------------------------------------

async function main(): Promise<void> {
  console.log('\n=== Stage 02: MCP over JSON-RPC 2.0 / stdio (hand-written) ===\n');

  // Spawn the server with the local tsx so this runs from source with no build
  // step. cwd is the package root (this script is launched from there via npm).
  const client = new StdioRpcClient('node_modules/.bin/tsx', ['src/mcp-server.ts']);

  // 1) Handshake. Per MCP, initialize MUST precede any other request; the host
  //    announces who it is and what protocol/capabilities it speaks.
  console.log('--- step 1: initialize (version + capability handshake) ---');
  await client.call('initialize', {
    protocolVersion: '2024-11-05',
    clientInfo: { name: 'hand-written-mcp-client', version: '0.1.0' },
    capabilities: {},
  });

  // 2) Discover tools. The result here is what would be injected into a model's
  //    tool list — which is exactly why we scan it for poisoning before trusting.
  console.log('\n--- step 2: tools/list (discover + security-scan tools) ---');
  const listed = await client.call('tools/list');
  const tools = ((listed.result as { tools?: Array<{ name: string; description: string }> })?.tools) ?? [];
  console.log(`  discovered ${tools.length} tools: ${tools.map((t) => t.name).join(', ')}`);
  for (const tool of tools) warnIfPoisoned(tool);

  // 3) Call a benign tool and read the result envelope.
  console.log('--- step 3: tools/call add(2, 3) ---');
  const called = await client.call('tools/call', { name: 'add', arguments: { a: 2, b: 3 } });
  const content = (called.result as { content?: Array<{ text?: string }> })?.content ?? [];
  console.log(`  add result: ${content.map((c) => c.text).join('')}`);

  client.close();
  console.log('\n=== done. The handshake above is the entire MCP stdio protocol. ===\n');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

// ---------------------------------------------------------------------------
// HONEST SCOPE — what a real MCP implementation has that this one does not:
//
// Methods / primitives we skipped:
//   - resources/list, resources/read, resources/subscribe — servers exposing
//     readable context (files, DB rows) the host can pull into the model.
//   - prompts/list, prompts/get — server-provided prompt templates.
//   - sampling/createMessage — the SERVER asks the HOST to run an LLM call
//     (inverts the usual direction; powerful and a trust hazard).
//   - elicitation — the server requests structured input from the user mid-call.
//   - logging, completion/complete, ping, progress notifications, cancellation.
//
// Protocol robustness we skipped:
//   - Real capability NEGOTIATION (we hard-code a version and assume tools-only;
//     real hosts reconcile client/server protocolVersion and feature flags).
//   - The `notifications/initialized` notification the client should send after
//     initialize, and ordering guarantees around it.
//   - Request timeouts, cancellation, and rejecting pending calls on child exit.
//   - Rich content types in results (image / audio / embedded resource), and
//     structured tool errors beyond a text string.
//
// Transports we skipped:
//   - Streamable HTTP (and the older HTTP+SSE) transport for remote servers,
//     with sessions, auth (OAuth), and reconnection — stdio is local-only.
//
// Security we only gestured at:
//   - Tool poisoning is one of several MCP-specific risks. Real concern list
//     also includes: rug-pull tool redefinition after approval, confused-deputy
//     via sampling, cross-server tool-name shadowing, and prompt injection from
//     resource contents. Our regex scan is a teaching stub, not a defense.
// ---------------------------------------------------------------------------
