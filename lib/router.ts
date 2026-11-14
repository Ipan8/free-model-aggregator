// Routing: which free model serves one request, and what happens when the
// provider it picked refuses.
//
// The brief never publishes a provider's per-trial budget, its failure
// schedule or its outage, so the router learns them from the answers it has
// seen: a 402 means that provider's budget is spent and it is never tried
// again; a 429 or 5xx means it refused this call, and three in a row mean
// its budget is spent too (two can still be a failure schedule: trial-0's
// Gemini refuses its 4th and 5th calls and serves its 6th). Every decision
// is a function of the brief and the providers' answers, never of the clock,
// so a replay takes the same path. One provider is never called twice at
// once: each has its own queue, so a second call can never meet a
// concurrent_requests refusal.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
} from './providers.ts';

/** How many refusals in a row mark a provider spent (a 402 does it at once). */
const REFUSALS_TO_EXHAUST = 3;

/** One provider's trial state: what the router has learned about it. */
export interface ProviderState {
  probe: Probe;
  /** Inference calls made to this provider, refusals included. */
  calls: number;
  /** Refusals since its last success. */
  consecutiveRefusals: number;
  /** True once its budget is known spent. */
  exhausted: boolean;
  /** The tail of this provider's serial queue. */
  queue: Promise<void>;
}

/** One (provider, model) that may serve a request, with its rating. */
export interface Candidate {
  state: ProviderState;
  model: ProbedModel;
  rating: number;
}

/** The router over one trial's probed providers. */
export class Router {
  readonly states: ProviderState[];
  private readonly ratings: RatingsFile | null;

  constructor(probes: readonly Probe[], ratings: RatingsFile | null) {
    this.ratings = ratings;
    this.states = probes.filter((p) => p.ok).map((probe) => ({ probe, calls: 0, consecutiveRefusals: 0, exhausted: false, queue: Promise.resolve() }));
  }

  /** The free models meeting `requires`, best rating first. */
  candidatesFor(requires: Requirement, allowAnthropicWire: boolean): Candidate[] {
    const out: Candidate[] = [];
    for (const state of this.states) {
      if (state.exhausted) continue;
      if (!allowAnthropicWire && state.probe.provider.wire === 'anthropic') continue;
      for (const model of state.probe.models) {
        const row = ratingOfModel(this.ratings, state.probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
        out.push({ state, model, rating: row?.rating_milli ?? 0 });
      }
    }
    out.sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      if (a.state.calls !== b.state.calls) return a.state.calls - b.state.calls;
      if (a.state.probe.provider.id !== b.state.probe.provider.id) return a.state.probe.provider.id < b.state.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    return out;
  }

  /** Run `work` after every earlier call to this provider has finished. */
  async serialized<T>(state: ProviderState, work: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const previous = state.queue;
    state.queue = state.queue.then(() => gate);
    await previous;
    try {
      return await work();
    } finally {
      release();
    }
  }

  /** A call answered 2xx: the provider is healthy again. */
  noteSuccess(state: ProviderState): void {
    state.calls += 1;
    state.consecutiveRefusals = 0;
  }

  /** A call refused with `status` (0 for a network error): learn from it. */
  noteRefusal(state: ProviderState, status: number): void {
    state.calls += 1;
    if (status === 402) {
      state.exhausted = true;
      return;
    }
    state.consecutiveRefusals += 1;
    if (state.consecutiveRefusals >= REFUSALS_TO_EXHAUST) state.exhausted = true;
  }
}
