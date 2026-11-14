# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Ipan8/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Ipan8/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten
simulated free-tier model providers, routing each request to the best free
model that meets it, failing over when a provider refuses, and reporting the
catalog it read. Built by an AI agent for make.fun's "The best free AI model
aggregator" bounty, $500 in USDC.

## How it works

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the ten providers, the
  sealed quality table and the 20-request workload manifest.
- Probes every provider's catalog route in its own wire and builds one
  catalog, printed by `node main.ts --list` and posted to `/register`, byte
  for byte.
- Serves `POST /v1/chat/completions` and `POST /v1/messages`, JSON and
  streaming, answering each request with the best free model that meets its
  requirements: the quality table's rating first, the least-used provider
  breaking ties.
- Speaks all three upstream wires (OpenAI, Anthropic, Gemini), passing the
  client's system prompt, user text, images, tools, stop sequences,
  max_tokens and JSON mode through untouched, and returns the provider's
  answer whole, with its usage counts and its stop reason, in the wire the
  client used. (An OpenAI JSON-mode request never routes to an
  Anthropic-wire provider: that wire cannot carry `response_format`.)
- On a 402 the provider's budget is spent and it is never tried again; on a
  429 or 5xx it counts the refusal and answers the request with the next
  free model that meets it. Three refusals in a row also mark a provider
  spent. One provider is never called twice at once.
- Polls `GET /work` and exits 0 when the env says the workload is done.
- Deterministic: every choice derives from the brief, the requests and the
  providers' answers, so a replay leaves the same state export.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, poll.
- `lib/providers.ts` — the brief's types and parser, auth headers, catalog probe.
- `lib/wire.ts` — OpenAI/Anthropic translation, SSE parsing and rendering.
- `lib/gemini.ts` — the Gemini wire, both directions.
- `lib/router.ts` — candidate ranking, provider budgets learned online, failover.
- `LICENSE` — MIT.

## Run it

```sh
node main.ts --list
```

In a trial the harness runs `node main.ts` with `MAKEFUN_PROMPT_FILE`,
`MAKEFUN_ENV_URL` and `MAKEFUN_TRIAL_SEED` set. Node 22, standard library
only, nothing to install.

## Local grading

With the competition kit unpacked, grade on the three practice trials:

```sh
node cli/makefun.mjs grade-local --kit . --bot <this repository> --driver process --instances 3
```

Practice results (14 Nov 2026): **99/100 on all three trials** — quality 99
(the ceiling: the single best model cannot cover all 20 requests within its
per-trial budget), served 100, wire 100, failover 100, catalog 100, replays
matched byte for byte.

---

<sub>Entered on [make.fun](https://make.fun/?ref=&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
