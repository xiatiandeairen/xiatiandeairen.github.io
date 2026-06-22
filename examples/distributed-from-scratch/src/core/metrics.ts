// core/metrics.ts — Histogram + Stats for measured distributed numbers.
//
// Why this exists: the book's credibility rests on reporting DISTRIBUTIONS, not
// averages. "Average request latency 8ms" hides the partition-induced 2000ms
// tail that is the entire point of a failure-mode chapter. A mean is a lie of
// omission in a system with timeouts and retries; p50/p99/max are where the
// distributed behavior actually lives. So every latency/round/message-count
// series goes through a Histogram and we print percentiles.
//
// These are EXACT percentiles over a stored sample (sort + index), not the
// streaming approximations (t-digest/HDR) you'd use at production scale. The
// samples here are small (hundreds–thousands of simulated messages) so exactness
// is cheap and removes "is the estimator wrong or the system wrong?" ambiguity.
//
// Determinism note: counts and percentiles computed from simulated values are
// fully deterministic for a given seed (the values come from seeded PRNG +
// SimClock). Only `core/clock.ts::bench` produces machine-dependent numbers.

export class Histogram {
  // Raw samples, kept unsorted until a percentile is requested. We don't sort on
  // every insert (record() is hot) — we sort lazily and cache.
  private samples: number[] = [];
  private sortedCache: number[] | null = null;

  /** Record one observation (a latency in ms, a round count, etc). Invalidates
   *  the sort cache — the next percentile() re-sorts. */
  record(value: number): void {
    if (!Number.isFinite(value)) throw new Error(`Histogram.record: non-finite ${value}`);
    this.samples.push(value);
    this.sortedCache = null;
  }

  /** Number of observations. The denominator behind every other figure — print
   *  it so readers can sanity-check (e.g. "p99 over 7 samples" is not a p99). */
  count(): number {
    return this.samples.length;
  }

  private sorted(): number[] {
    if (!this.sortedCache) this.sortedCache = [...this.samples].sort((a, b) => a - b);
    return this.sortedCache;
  }

  /** Nearest-rank percentile in [0,1]. p(0)=min, p(1)=max. Nearest-rank (not
   *  interpolated) because for discrete counts (rounds, messages) an interpolated
   *  "p99 = 4.7 rounds" is nonsense; the reader wants a value that actually
   *  occurred. Empty histogram returns NaN — asking for a percentile of nothing
   *  is a caller bug we surface rather than hide as 0. */
  percentile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`percentile: p must be in [0,1], got ${p}`);
    const s = this.sorted();
    if (s.length === 0) return NaN;
    const rank = Math.ceil(p * s.length);
    const idx = Math.min(s.length - 1, Math.max(0, rank - 1));
    return s[idx];
  }

  min(): number {
    return this.sorted()[0] ?? NaN;
  }

  max(): number {
    const s = this.sorted();
    return s[s.length - 1] ?? NaN;
  }

  mean(): number {
    if (this.samples.length === 0) return NaN;
    let sum = 0;
    for (const v of this.samples) sum += v;
    return sum / this.samples.length;
  }

  /** One-line summary suitable for a metrics-table cell or quick log. */
  summary(): { count: number; min: number; p50: number; p99: number; max: number; mean: number } {
    return {
      count: this.count(),
      min: this.min(),
      p50: this.percentile(0.5),
      p99: this.percentile(0.99),
      max: this.max(),
      mean: this.mean(),
    };
  }
}

/** Stats — a bag of named monotonic counters for the discrete events a
 *  distributed run produces: messages sent/delivered/dropped, elections held,
 *  retries, term changes. Counters (not gauges) because the questions the book
 *  asks are cumulative ("how many messages did consensus cost?"). All
 *  deterministic for a fixed seed, so two readers comparing runs see the same
 *  totals — which is how a protocol regression gets caught. */
export class Stats {
  private counters = new Map<string, number>();

  /** Add `n` (default 1) to a named counter, creating it at 0 if new. */
  inc(name: string, n = 1): void {
    this.counters.set(name, (this.counters.get(name) ?? 0) + n);
  }

  /** Read a counter (0 if never incremented — a metric you forgot to bump should
   *  read 0, not throw, so partial instrumentation still prints). */
  get(name: string): number {
    return this.counters.get(name) ?? 0;
  }

  /** Snapshot all counters, insertion-ordered, for printTable. */
  snapshot(): Record<string, number> {
    return Object.fromEntries(this.counters);
  }
}
