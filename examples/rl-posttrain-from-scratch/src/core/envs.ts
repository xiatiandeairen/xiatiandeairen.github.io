// core/envs.ts — the offline, deterministic toy environments the whole book
// learns against.
//
// Why toy environments at all in an "RL from scratch" book: the lessons of RL —
// exploration vs exploitation, credit assignment, value bootstrapping, reward
// hacking — are *structural*. They appear in a 10-state gridworld exactly as in
// a 70B LLM, but in the gridworld we know the OPTIMAL answer in closed form. An
// environment whose optimum is computable is the only honest yardstick: every
// curve in the book is plotted against the true V* / best arm / optimal action,
// so "did it learn" is a measured fact, not a vibe.
//
// All envs share one interface so stages can swap them. The interface is
// classic Sutton & Barto: reset to a start state, step an action, get
// (nextState, reward, done). nStates/nActions let tabular methods size tables.
//
// Determinism contract: an env's *structure* (arm means, grid layout, optimal
// action map) is fixed at construction from the seed; per-episode randomness
// (which start state, Gaussian payoff noise) is drawn from the rng passed to
// reset/step. Same seed in, same trajectory out.

import { gaussian, sampleCategorical, type Rng } from "./rng.js";

// The uniform environment contract. Note step() takes the rng: stochastic envs
// (noisy bandit payoffs, windy gridworld slips) consume randomness AT step time,
// not at construction, so a stage that re-runs the same policy with the same
// seed gets the same rewards.
export interface Env {
  readonly nStates: number;
  readonly nActions: number;
  reset(rng: Rng): number; // returns the start state
  step(state: number, action: number, rng: Rng): StepResult;
}

export interface StepResult {
  nextState: number;
  reward: number;
  done: boolean;
}

// ---------------------------------------------------------------------------
// (a) k-armed Gaussian bandit — the simplest RL: one state, k actions, reward
//     is a noisy sample from a per-arm true mean. The point of the bandit is to
//     isolate exploration/exploitation with ZERO credit-assignment confound
//     (one step, so no future to reason about). Because we fix the true means,
//     REGRET (cumulative gap to the best arm) is exactly computable — see
//     metrics.regret. This is the canonical "is exploration working" probe.
// ---------------------------------------------------------------------------
export interface Bandit extends Env {
  readonly trueMeans: number[]; // ground truth, for regret. Do not peek in agents!
  readonly optimalArm: number;
}

export function makeBandit(nArms: number, rng: Rng): Bandit {
  // Arm means ~ N(0,1), fixed now from the construction rng. Spreading them with
  // unit variance guarantees a meaningful gap between best and rest, so a working
  // agent's regret curve visibly flattens (a tiny gap would make any agent look
  // optimal — a misleading "success").
  const trueMeans = Array.from({ length: nArms }, () => gaussian(rng));
  let optimalArm = 0;
  for (let i = 1; i < nArms; i++) if (trueMeans[i] > trueMeans[optimalArm]) optimalArm = i;
  return {
    nStates: 1,
    nActions: nArms,
    trueMeans,
    optimalArm,
    reset: () => 0, // one state; nothing to randomize
    step(_state, action, stepRng) {
      // Reward = true mean + unit Gaussian noise. The noise is why the agent
      // can't just try each arm once: a single pull is an unreliable estimate,
      // forcing repeated sampling = the explore cost regret measures.
      const reward = gaussian(stepRng, trueMeans[action], 1);
      return { nextState: 0, reward, done: true };
    },
  };
}

// ---------------------------------------------------------------------------
// (b) Gridworld — the simplest environment WITH credit assignment. Reward is
//     sparse (only at the goal / hole), so the agent must propagate value
//     backward through states it visited long before the reward. This is where
//     value iteration, bootstrapping, and discounting earn their keep. Because
//     the MDP is small and known, we can solve V* exactly (value iteration in
//     metrics/stage02) and plot the learner's gap to optimal.
//
//     Layout: row-major grid. Cells: '.' free, '#' wall (un-enterable),
//     'H' hole (terminal, negative), 'G' goal (terminal, positive), 'S' start.
//     Actions: 0=up 1=right 2=down 3=left. Optional wind pushes the agent an
//     extra row "up" with probability `windProb` — stochastic transitions that
//     make deterministic planning fail, demoing why we need expectations.
// ---------------------------------------------------------------------------
export interface GridworldOpts {
  layout?: string[]; // rows of equal length using the cell alphabet above
  stepReward?: number; // per-move cost; negative encourages short paths
  goalReward?: number;
  holeReward?: number;
  windProb?: number; // 0 = deterministic
}

export interface Gridworld extends Env {
  readonly width: number;
  readonly height: number;
  readonly start: number;
  readonly terminal: boolean[]; // terminal[s] = episode ends on entering s
  readonly cellReward: number[]; // reward for ENTERING state s
  readonly stepReward: number;
  readonly windProb: number;
  isWall(state: number): boolean;
  // Deterministic transition ignoring wind — value iteration uses this plus the
  // wind model to build the expectation. Exposed so the exact solver in stage 02
  // can enumerate transitions without re-implementing the geometry.
  intendedNext(state: number, action: number): number;
}

const DEFAULT_LAYOUT = [
  "S...",
  ".#.H",
  "...G",
];

export function makeGridworld(opts: GridworldOpts = {}): Gridworld {
  const layout = opts.layout ?? DEFAULT_LAYOUT;
  const stepReward = opts.stepReward ?? -0.04;
  const goalReward = opts.goalReward ?? 1;
  const holeReward = opts.holeReward ?? -1;
  const windProb = opts.windProb ?? 0;

  const height = layout.length;
  const width = layout[0].length;
  const nStates = width * height;
  const terminal = new Array<boolean>(nStates).fill(false);
  const cellReward = new Array<number>(nStates).fill(0);
  const wall = new Array<boolean>(nStates).fill(false);
  let start = 0;

  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const s = r * width + c;
      const ch = layout[r][c];
      if (ch === "#") wall[s] = true;
      else if (ch === "G") { terminal[s] = true; cellReward[s] = goalReward; }
      else if (ch === "H") { terminal[s] = true; cellReward[s] = holeReward; }
      else if (ch === "S") start = s;
    }
  }

  const intendedNext = (state: number, action: number): number => {
    const r = Math.floor(state / width);
    const c = state % width;
    let nr = r, nc = c;
    if (action === 0) nr--; else if (action === 1) nc++; else if (action === 2) nr++; else nc--;
    // Out-of-bounds or into a wall => stay put. This "bump" rule is why a naive
    // agent can waste moves against walls; the step cost punishes it, teaching
    // the layout. Returning `state` (not a sentinel) keeps transitions total.
    if (nr < 0 || nr >= height || nc < 0 || nc >= width) return state;
    const ns = nr * width + nc;
    if (wall[ns]) return state;
    return ns;
  };

  return {
    nStates,
    nActions: 4,
    width,
    height,
    start,
    terminal,
    cellReward,
    stepReward,
    windProb,
    isWall: (s) => wall[s],
    intendedNext,
    reset: () => start,
    step(state, action, stepRng) {
      let ns = intendedNext(state, action);
      // Wind: with windProb, after the intended move the agent is shoved one
      // extra cell up. Stochastic transition => deterministic shortest-path
      // planning is wrong; the agent must reason about expected outcomes. This
      // is the toy stand-in for "the environment is not fully controllable."
      if (windProb > 0 && stepRng() < windProb) {
        const r = Math.floor(ns / width);
        if (r > 0 && !wall[ns - width]) ns = ns - width;
      }
      // Reward for ENTERING ns = its cell reward (goal/hole) plus the per-step
      // cost. Charging step cost even on the terminal entry keeps the value
      // function consistent (the optimal path length is reflected in V*).
      const reward = cellReward[ns] + stepReward;
      return { nextState: ns, reward, done: terminal[ns] };
    },
  };
}

// ---------------------------------------------------------------------------
// (c) Contextual bandit — bridges bandit and full RL: many states (contexts),
//     each with a KNOWN optimal action, one step per episode. It is the cleanest
//     testbed for *policy gradient*: the policy is a state->action distribution,
//     and because we know optimalAction[s] we can report exact accuracy ("does
//     the learned policy pick the right action per context") alongside reward.
//     No bootstrapping needed, so REINFORCE's signal is uncontaminated by value
//     estimation error — ideal for the first policy-gradient stage.
// ---------------------------------------------------------------------------
export interface ContextualBandit extends Env {
  readonly optimalAction: number[]; // ground truth per state, for accuracy
  // Reward for taking `action` in `state`, deterministic mean (noise added in
  // step). Exposed so stages can compute the best achievable expected reward.
  rewardMean(state: number, action: number): number;
}

export function makeContextualBandit(
  nStates: number,
  nActions: number,
  rng: Rng,
  noiseStd = 0.3,
): ContextualBandit {
  // Each state has one designated optimal action with the highest mean reward.
  // Means are well-separated (optimal=1, others sampled below it) so the
  // accuracy signal is crisp: a correct policy scores ~1, a random one ~1/nActions.
  const optimalAction = Array.from({ length: nStates }, () =>
    sampleCategorical(new Array(nActions).fill(1 / nActions), rng),
  );
  // Per (state, action) base mean: optimal action = 1.0, others in [0, 0.6).
  const means: number[][] = [];
  for (let s = 0; s < nStates; s++) {
    means[s] = Array.from({ length: nActions }, () => 0.6 * rng());
    means[s][optimalAction[s]] = 1.0;
  }
  return {
    nStates,
    nActions,
    optimalAction,
    rewardMean: (s, a) => means[s][a],
    reset: (resetRng) => sampleCategorical(new Array(nStates).fill(1 / nStates), resetRng),
    step(state, action, stepRng) {
      // Gaussian noise on the reward is what makes this a *learning* problem and
      // not a lookup: a single trajectory's reward is a noisy vote, so the policy
      // gradient must average over many samples to find the signal. noiseStd
      // tunes difficulty; stages can crank it to show variance swamping signal.
      const reward = gaussian(stepRng, means[state][action], noiseStd);
      return { nextState: state, reward, done: true };
    },
  };
}
