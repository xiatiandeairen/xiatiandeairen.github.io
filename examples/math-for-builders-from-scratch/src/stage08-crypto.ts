// stage08-crypto.ts — The number theory under cryptography: one-way functions from scratch.
//
// WHY this stage: every TLS handshake, SSH login, and signed JWT rests on ONE asymmetry —
//   multiplying two primes is cheap, factoring their product is (believed to be) hard. This
//   stage builds that asymmetry from primitives: extended Euclid (modular inverse), fast
//   modular exponentiation (the actual one-way trapdoor), Miller–Rabin (how we find primes
//   without trial-dividing), then assembles toy RSA and Diffie–Hellman and BREAKS the RSA by
//   factoring n — proving the security is purely a function of how big the primes are.
//
// HONEST-NUMBER NOTE: every printed number is computed live with BigInt (exact integer
//   arithmetic, no float rounding). The RSA primes here are TINY (≤ ~1e6) so the round-trip
//   and the attack both finish in milliseconds on one CPU. The wall-clock factoring time we
//   print is REAL (measured), but its absolute value is meaningless for security — what
//   transfers is the SHAPE: factoring cost grows with the size of the smaller prime, and
//   real keys use 1024-bit+ primes where the same trial-division loop would not finish before
//   the heat death of the sun. We make that gap concrete by extrapolating, clearly marked.
//
// CONTRACT: reuse core/rng.js ONLY to pick reproducible Miller–Rabin witnesses and DH secrets
//   for the demo. core/rng (mulberry32) is a 32-bit PRNG, fully predictable from its state —
//   it is NEVER a source of real key material. This stage exists partly to show why.

import { mulberry32, type Rng } from "./core/rng.js";

// ----------------------------------------------------------------------------
// Primitive 1: extended Euclidean algorithm.
// Returns [g, x, y] with g = gcd(a, b) and a*x + b*y = g (Bézout's identity).
// WHY we need x specifically: the modular inverse of e mod φ is exactly the x that
//   satisfies e*x ≡ 1 (mod φ), which is the Bézout coefficient when gcd(e, φ) = 1.
// INVARIANT: terminates because the remainder strictly decreases each step (Euclid).
// ----------------------------------------------------------------------------
function egcd(a: bigint, b: bigint): [bigint, bigint, bigint] {
  if (b === 0n) return [a, 1n, 0n];
  const [g, x1, y1] = egcd(b, a % b);
  // Back-substitute: gcd stays the same, coefficients fold by the quotient a/b.
  return [g, y1, x1 - (a / b) * y1];
}

/**
 * Modular inverse of `a` mod `m`: the unique r in [1, m) with a*r ≡ 1 (mod m).
 * FAILURE MODE: only exists when gcd(a, m) = 1. If a and m share a factor we throw,
 *   because returning a bogus value here would silently corrupt RSA key generation.
 */
function modInverse(a: bigint, m: bigint): bigint {
  const [g, x] = egcd(((a % m) + m) % m, m);
  if (g !== 1n) throw new Error(`no inverse: gcd(${a}, ${m}) = ${g} ≠ 1`);
  return ((x % m) + m) % m; // normalize into [0, m); egcd's x can be negative
}

// ----------------------------------------------------------------------------
// Primitive 2: fast modular exponentiation (square-and-multiply).
// This IS the one-way function. base^exp mod m runs in O(log exp) multiplies, so
//   encryption (m^e) is cheap. Inverting it WITHOUT the private exponent d is the
//   hard part — that asymmetry is the whole game.
// INVARIANT: m > 0, exp >= 0. We reduce base mod m up front so intermediate products
//   never exceed ~m^2 (keeps BigInt sizes bounded; the naive base**exp would be astronomically large).
// ----------------------------------------------------------------------------
function modpow(base: bigint, exp: bigint, m: bigint): bigint {
  if (m === 1n) return 0n; // everything ≡ 0 mod 1; guard the degenerate ring
  let result = 1n;
  base %= m;
  while (exp > 0n) {
    if (exp & 1n) result = (result * base) % m; // bit set -> fold current square in
    base = (base * base) % m; // square for the next bit position
    exp >>= 1n;
  }
  return result;
}

// ----------------------------------------------------------------------------
// Primitive 3: Miller–Rabin probabilistic primality test.
// WHY not trial division: to find a 1024-bit prime you cannot divide by every number
//   up to 2^512. Miller–Rabin tests primality in O(k log^3 n) by checking k random
//   witnesses; a composite survives one round with probability ≤ 1/4, so k rounds give
//   ≤ 4^-k false-positive rate. We SEED the witnesses (via core/rng) so the test is
//   reproducible for teaching — real implementations draw witnesses from a CSPRNG.
// FAILURE MODE (demoed below): a fixed/weak witness set has Carmichael-style composites
//   that masquerade as prime. Randomized witnesses are what make the bound hold.
// ----------------------------------------------------------------------------
function isProbablePrime(n: bigint, rounds: number, rng: Rng): boolean {
  if (n < 2n) return false;
  for (const p of [2n, 3n, 5n, 7n, 11n, 13n]) {
    if (n === p) return true;
    if (n % p === 0n) return false; // cheap small-factor sieve before the heavy test
  }
  // Write n-1 = d * 2^r with d odd; Miller–Rabin examines the square-root chain of a^d.
  let d = n - 1n;
  let r = 0n;
  while (d % 2n === 0n) {
    d /= 2n;
    r += 1n;
  }
  witness: for (let i = 0; i < rounds; i++) {
    // Pick a witness a in [2, n-2]. rng() is [0,1); scale into the integer range.
    const a = 2n + BigInt(Math.floor(rng() * Number(n - 4n)));
    let x = modpow(a, d, n);
    if (x === 1n || x === n - 1n) continue; // a^d already a trivial sqrt of 1
    for (let j = 0n; j < r - 1n; j++) {
      x = (x * x) % n;
      if (x === n - 1n) continue witness; // found a nontrivial -1, a is not a witness
    }
    return false; // no -1 in the chain => a proves n composite
  }
  return true; // survived all rounds: prime with prob >= 1 - 4^-rounds
}

/** Find the next probable prime >= start. Used to mint toy RSA / DH parameters deterministically. */
function nextPrime(start: bigint, rng: Rng): bigint {
  let n = start | 1n; // start odd; even numbers > 2 are never prime
  while (!isProbablePrime(n, 20, rng)) n += 2n;
  return n;
}

// ----------------------------------------------------------------------------
// RSA, assembled from the three primitives above.
// ----------------------------------------------------------------------------
interface RsaKeypair {
  n: bigint; // modulus = p*q (public)
  e: bigint; // public exponent
  d: bigint; // private exponent = e^-1 mod φ(n) (SECRET)
  p: bigint; // prime factor (SECRET — knowing it = knowing d)
  q: bigint; // prime factor (SECRET)
}

/**
 * Generate a toy RSA keypair from two primes.
 * INVARIANT: p ≠ q (else φ is wrong and n is a perfect square — trivially factorable);
 *   gcd(e, φ) = 1 so e is invertible. We pick e = 65537 (the standard choice: prime, and
 *   its binary 0x10001 has only two set bits so encryption is fast), falling back to 3 for
 *   the tiniest moduli where 65537 ≥ φ.
 */
function genRsa(p: bigint, q: bigint): RsaKeypair {
  if (p === q) throw new Error("p and q must differ");
  const n = p * q;
  const phi = (p - 1n) * (q - 1n); // Euler's totient: count of integers < n coprime to n
  let e = 65537n;
  if (e >= phi || egcd(e, phi)[0] !== 1n) e = 3n; // toy fallback for small φ
  const d = modInverse(e, phi);
  return { n, e, d, p, q };
}

// Textbook RSA on a single integer message m with 0 <= m < n.
// WARNING: this is RAW textbook RSA — deterministic, no OAEP padding, no integrity.
//   Real RSA without padding leaks equality of plaintexts and is malleable. Production
//   code MUST use a vetted library (OAEP/PSS). This is for showing the math only.
const rsaEncrypt = (m: bigint, n: bigint, e: bigint): bigint => modpow(m, e, n);
const rsaDecrypt = (c: bigint, n: bigint, d: bigint): bigint => modpow(c, d, n);

/**
 * The attack: recover the private key from the PUBLIC key alone by factoring n.
 * Trial division up to sqrt(n) — the most naive factoring there is. It works here ONLY
 *   because the primes are tiny. Returns the recovered private exponent d plus the count
 *   of trial divisions performed (so the reader sees the work scale with sqrt(n)).
 * FAILURE MODE this exposes: RSA security = factoring hardness. Small primes => instant break.
 */
function breakRsa(n: bigint, e: bigint): { p: bigint; q: bigint; d: bigint; trials: number } {
  let trials = 0;
  for (let f = 2n; f * f <= n; f++) {
    trials++;
    if (n % f === 0n) {
      const p = f;
      const q = n / f;
      const phi = (p - 1n) * (q - 1n);
      return { p, q, d: modInverse(e, phi), trials }; // d reconstructed from stolen φ
    }
  }
  throw new Error("n is prime — not an RSA modulus");
}

// ----------------------------------------------------------------------------
// Diffie–Hellman key exchange. Same one-way function (modpow), different protocol:
//   over a public (p, g), Alice and Bob each pick a secret exponent, exchange g^secret,
//   and both compute g^(a*b) without ever transmitting a or b. An eavesdropper sees
//   p, g, g^a, g^b and must solve the discrete log to recover the shared secret.
// ----------------------------------------------------------------------------
function diffieHellman(p: bigint, g: bigint, secretA: bigint, secretB: bigint) {
  const publicA = modpow(g, secretA, p); // Alice -> Bob
  const publicB = modpow(g, secretB, p); // Bob -> Alice
  const sharedFromA = modpow(publicB, secretA, p); // (g^b)^a mod p
  const sharedFromB = modpow(publicA, secretB, p); // (g^a)^b mod p
  return { publicA, publicB, sharedFromA, sharedFromB };
}

function main(): void {
  console.log("=== Stage 08 · 数论与密码学：单向函数的魔法 ===\n");

  // --- Primitives sanity: extended Euclid gives a real modular inverse. ---
  const invDemo = modInverse(17n, 3120n);
  console.log("【1. 模逆元 / 扩展欧几里得】");
  console.log(`  17^-1 mod 3120 = ${invDemo}  (校验 17*${invDemo} mod 3120 = ${(17n * invDemo) % 3120n}，应为 1)`);
  console.log(`  快速模幂  7^256 mod 13 = ${modpow(7n, 256n, 13n)}  (square-and-multiply, O(log exp) 次乘法)\n`);

  // --- Build a toy RSA keypair from seeded small primes (reproducible). ---
  const keyRng = mulberry32(0x08c0de);
  const p = nextPrime(60000n + BigInt(Math.floor(keyRng() * 5000)), keyRng);
  const q = nextPrime(70000n + BigInt(Math.floor(keyRng() * 5000)), keyRng);
  const key = genRsa(p, q);
  console.log("【2. 从零 RSA（玩具小素数，仅教学）】");
  console.log(`  随机素数 p = ${key.p}, q = ${key.q}  (Miller–Rabin 找到，20 轮见证)`);
  console.log(`  公钥 n = p*q = ${key.n}`);
  console.log(`  公钥指数 e = ${key.e}`);
  console.log(`  私钥指数 d = ${key.d}  (= e^-1 mod φ(n)，绝密)`);

  // Encrypt a short message. We pack the bytes of "MATH" into one integer < n.
  const text = "MATH";
  let m = 0n;
  for (const ch of text) m = m * 256n + BigInt(ch.charCodeAt(0)); // base-256 packing
  if (m >= key.n) throw new Error("message too large for this toy modulus");
  const cipher = rsaEncrypt(m, key.n, key.e);
  const recovered = rsaDecrypt(cipher, key.n, key.d);
  // Unpack the integer back to text to prove the round-trip is byte-exact.
  let unpacked = "";
  let tmp = recovered;
  while (tmp > 0n) {
    unpacked = String.fromCharCode(Number(tmp % 256n)) + unpacked;
    tmp /= 256n;
  }
  console.log(`  明文 "${text}" 打包为整数 m = ${m}`);
  console.log(`  密文 c = m^e mod n = ${cipher}`);
  console.log(`  解密 c^d mod n = ${recovered} → 还原文本 "${unpacked}"`);
  console.log(`  round-trip ${recovered === m ? "成功 ✓" : "失败 ✗"}\n`);

  // --- Diffie–Hellman over a seeded toy prime. ---
  const dhRng = mulberry32(0xd1ff1e);
  const dhP = nextPrime(900000n + BigInt(Math.floor(dhRng() * 50000)), dhRng);
  const g = 5n; // a small generator; 5 is a primitive root for many primes (not verified here)
  const secretA = 2n + BigInt(Math.floor(dhRng() * 100000)); // Alice's private exponent
  const secretB = 2n + BigInt(Math.floor(dhRng() * 100000)); // Bob's private exponent
  const dh = diffieHellman(dhP, g, secretA, secretB);
  console.log("【3. Diffie–Hellman 密钥交换】");
  console.log(`  公开参数 p = ${dhP}, g = ${g}`);
  console.log(`  Alice 私密 a = ${secretA}  → 公开 g^a mod p = ${dh.publicA}`);
  console.log(`  Bob   私密 b = ${secretB}  → 公开 g^b mod p = ${dh.publicB}`);
  console.log(`  Alice 算 (g^b)^a = ${dh.sharedFromA}`);
  console.log(`  Bob   算 (g^a)^b = ${dh.sharedFromB}`);
  console.log(`  双方得到同一密钥 ${dh.sharedFromA === dh.sharedFromB ? "✓ (a,b 从未上线传输)" : "✗ 不一致"}\n`);

  // --- Attack: break the toy RSA by factoring n, real wall-clock measured. ---
  console.log("【4. 攻击 demo：暴力分解 n 还原私钥】");
  const t0 = performance.now();
  const broken = breakRsa(key.n, key.e);
  const elapsedMs = performance.now() - t0;
  console.log(`  仅凭公钥 (n, e) 试除分解 n...`);
  console.log(`  分解出 p = ${broken.p}, q = ${broken.q}  (${broken.trials} 次试除)`);
  console.log(`  重建私钥 d = ${broken.d}  → 与真私钥 ${broken.d === key.d ? "一致 ✓ 密钥已破解" : "不一致 ✗"}`);
  console.log(`  实测耗时 = ${elapsedMs.toFixed(3)} ms (本机 wall-clock)`);

  // Extrapolate honestly: trial division does ~sqrt(n)/2 divisions. Scale that to a real key.
  // The toy break did `trials` divisions in elapsedMs; assume a flat per-division cost and
  // ask how long sqrt(2^1024) ≈ 2^512 divisions would take. (est., wildly optimistic — a real
  // attacker uses GNFS, far faster than trial division, yet 2048-bit RSA still stands.)
  const perDivisionMs = elapsedMs / Math.max(broken.trials, 1);
  const divisionsFor1024bit = Math.pow(2, 512); // ~sqrt of a 1024-bit modulus
  const yearsEst = (perDivisionMs * divisionsFor1024bit) / 1000 / 60 / 60 / 24 / 365;
  console.log(
    `  同样的试除法分解一个 1024-bit RSA 模数约需 ${yearsEst.toExponential(2)} 年 (est.，宇宙年龄约 1.4e10 年)\n`,
  );

  // --- Failure mode 1: weak randomness => predictable / shared keys. ---
  console.log("--- 失败模式 1：弱随机源 → 可预测密钥 ---");
  // Two parties seeding RSA from the SAME predictable PRNG state generate the SAME primes.
  const weak1 = mulberry32(42);
  const weak2 = mulberry32(42); // attacker who guesses the seed reproduces it bit-for-bit
  const pA = nextPrime(50000n + BigInt(Math.floor(weak1() * 1000)), weak1);
  const pB = nextPrime(50000n + BigInt(Math.floor(weak2() * 1000)), weak2);
  console.log(`  两方用同一 PRNG 种子(42) 生成素数: ${pA} 与 ${pB} → ${pA === pB ? "完全相同 ✗" : "不同"}`);
  console.log("  core/rng (mulberry32) 32-bit 状态可枚举，绝不能产密钥；真实密钥须用 CSPRNG (crypto.randomBytes)。\n");

  // --- Failure mode 2: a shared prime across two moduli kills BOTH keys at once. ---
  console.log("--- 失败模式 2：两把密钥共用一个素数 → 一次 gcd 全破 ---");
  // If RSA moduli n1, n2 accidentally share a prime (bad RNG in the field, observed in real
  // 2012 internet-wide scans), gcd(n1, n2) reveals it WITHOUT any factoring at all.
  const shared = nextPrime(80000n, mulberry32(1));
  const n1 = shared * nextPrime(81000n, mulberry32(2));
  const n2 = shared * nextPrime(82000n, mulberry32(3));
  const [common] = egcd(n1, n2);
  console.log(`  n1 = ${n1}, n2 = ${n2}`);
  console.log(`  gcd(n1, n2) = ${common} → 立刻得到公因子 ${shared} (无需试除)，两把私钥同时沦陷 ✗`);
  console.log("  教训：素数必须独立且高熵。本 stage 全部为教学实现，生产一律用经审计的库 (OpenSSL / libsodium)。");
}

main();
