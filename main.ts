// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// This entry starts from the kit's starter and replaces its routing and its
// wires. What it does, per trial:
//
//   - reads the brief (providers, quality table, workload manifest) and
//     probes every provider's catalog route in its own wire, building one
//     catalog that `--list` prints and /register posts, byte for byte;
//   - serves POST /v1/chat/completions and POST /v1/messages, JSON and
//     streaming, answering each request with the best free model that meets
//     it: the quality table's rating first, the least-used provider breaking
//     ties, and a provider whose budget is spent never tried again;
//   - speaks all three upstream wires (OpenAI, Anthropic, Gemini), passing
//     the client's system prompt, user text, images, tools, stop sequences,
//     max_tokens and JSON mode through untouched, and returns the provider's
//     answer whole, with its usage counts and its stop reason, in the wire
//     the client used;
//   - on a 402, 429 or 5xx, counts the failure and answers the request with
//     the next free model that meets it, until one serves it;
//   - polls GET /work and exits 0 the moment the env says the workload is
//     done.
//
// Everything is a function of the brief, the requests and the providers'
// answers: no wall clock in anything sent, no unseeded randomness, so a
// replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicSseEvents,
  anthropicToOpenAI,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAISseEvents,
  openAIToAnthropicRequest,
  parseAnthropicUpstream,
  parseGeminiUpstream,
  parseOpenAIUpstream,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { anthropicToGemini, openAIToGemini } from './lib/gemini.ts';
import { Router, type Candidate, type ProviderState } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request for one candidate: its URL, headers and body. */
function upstreamFor(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate, stream: boolean): { url: string; headers: Record<string, string>; payload: Json } {
  const provider = candidate.state.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const payload = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) payload.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, payload };
  }
  const payload = clientApi === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}${stream ? '?alt=sse' : ''}`;
  return { url, headers, payload };
}

/** Parse one provider's answer body into the normalised shape, or null. */
function parseUpstream(wire: string, text: string): NormalizedAnswer | null {
  if (wire === 'openai') return parseOpenAIUpstream(text);
  if (wire === 'anthropic') return parseAnthropicUpstream(text);
  return parseGeminiUpstream(text);
}

/**
 * One try at one candidate: the provider's answer on a 2xx, else null, with
 * the refusal learned. A 200 whose body does not parse counts as a refusal:
 * it cannot serve the request.
 */
async function tryCandidate(state: ProviderState, clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate, stream: boolean): Promise<NormalizedAnswer | null> {
  const request = upstreamFor(clientApi, body, candidate, stream);
  let response: Response;
  try {
    response = await router!.serialized(state, () =>
      fetch(request.url, { method: 'POST', headers: request.headers, body: JSON.stringify(request.payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) }),
    );
  } catch {
    router!.noteRefusal(state, 0);
    return null;
  }
  if (!response.ok) {
    router!.noteRefusal(state, response.status);
    try {
      await response.arrayBuffer();
    } catch {
      // The body is gone: the refusal is already learned.
    }
    return null;
  }
  const text = await response.text();
  const answer = parseUpstream(candidate.state.probe.provider.wire, text);
  if (answer === null) {
    router!.noteRefusal(state, response.status);
    return null;
  }
  router!.noteSuccess(state);
  return answer;
}

interface Served {
  answer: NormalizedAnswer;
  model: string;
}

/**
 * Serve one client request: the best free model that meets it, failing over
 * down the ranked candidates while providers refuse. Returns null when no
 * free model serves it.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<Served | null> {
  if (router === null) return null;
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  // An OpenAI JSON-mode request must go to a wire that can carry
  // response_format: Anthropic's wire cannot, and its provider would echo
  // json_mode absent. Anthropic clients carry no JSON flag, so any wire may
  // serve their json requirement.
  const jsonMode = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = router.candidatesFor(requires, !jsonMode);
  for (const candidate of candidates) {
    if (candidate.state.exhausted) continue;
    const answer = await tryCandidate(candidate.state, clientApi, body, candidate, stream);
    if (answer !== null) return { answer, model: candidate.model.id };
  }
  return null;
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model serves this request');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(openAISseEvents(served.answer, served.model));
    return;
  }
  sendJson(res, 200, openAIJsonAnswer(served.answer, served.model));
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model serves this request');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(anthropicSseEvents(served.answer, served.model));
    return;
  }
  sendJson(res, 200, anthropicJsonAnswer(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
