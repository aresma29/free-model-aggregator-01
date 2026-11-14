// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone (nothing is installed; TypeScript
// runs directly, and a relative import needs its .ts extension). The brief is at the path in
// $MAKEFUN_PROMPT_FILE. It carries three JSON documents: the provider directory, the quality
// table, and the workload manifest. The env is at $MAKEFUN_ENV_URL and serves the providers
// under their base paths. This server reads the brief, probes each provider's catalog route in
// the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header
//      it got: x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions
//      and POST /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//      A process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
// register body's catalog must be that same document. Build it once, as this file does, and use
// the one object.
//
// HOW IT ROUTES (the score's columns, in order of weight)
//   - quality (60%): every free candidate that meets a request's requirements is ranked by the
//     quality table's rating_milli, best first. The manifest puts the heavy requests first, so
//     serving the best available model for each request as it arrives is the right greedy play:
//     a later request never outweighs an earlier one. Ties break toward the provider this
//     server has called least, which spreads the load.
//   - served (15%) + failover (10%): a provider call that answers 402, 429 or a 5xx is never
//     the end of a request. A failed candidate is tried once more only when it is strictly
//     better than the next untried one — a refusal is one call index on the provider's
//     schedule, so the next index usually answers, and a successful retry costs the same call
//     falling back would. When the next candidate ties, falling back costs the same call and
//     keeps the rating, so no retry is spent. A provider that fails twice in a row is skipped
//     while any other candidate remains: its budget is likely spent or it is in outage, and
//     hammering it only burns calls.
//   - wire (10%): every request is translated faithfully (lib/wire.ts) — system prompt, user
//     text, image, tools, max_tokens, stops and JSON mode all reach the provider exactly as
//     the client sent them, and the provider's usage comes back exactly. One rule the wires
//     force: a JSON-mode request never goes to an anthropic-wire provider, whose echo can
//     never carry JSON mode.
//   - catalog (5%): the probe reports what each listing said, with the quality table's free
//     flag and capabilities where it has a row; a listing that hides its prices leaves the row
//     short. A failed probe is retried once — catalog calls cost no budget.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register, the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the normalized request, the six wire translations, the SSE writers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
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
  anthropicModelList,
  openAIError,
  openAIModelList,
  parseAnthropicAnswer,
  parseAnthropicRequest,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  parseOpenAIRequest,
  renderAnthropic,
  renderAnthropicJson,
  renderAnthropicStream,
  renderGemini,
  renderOpenAI,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormAnswer,
  type NormRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most upstream attempts spent on one client request. */
const MAX_ATTEMPTS = 16;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Upstream infer calls made to each provider, in this trial (the env counts the same way). */
const callsMade = new Map<string, number>();
/** Consecutive failed upstream calls per provider (a success resets it). */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free (provider, model) pairs that meet one request's requirements, best rating first.
 * Ties break toward the provider called least so far, then by provider and model id — a total
 * order, so a replay makes the same choices. A JSON-mode request never lists an
 * anthropic-wire provider: its echo can never carry JSON mode, so the wire check would fail.
 * Providers that failed twice in a row are skipped while any other candidate remains.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const all: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      all.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  all.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ca = callsMade.get(a.probe.provider.id) ?? 0;
    const cb = callsMade.get(b.probe.provider.id) ?? 0;
    if (ca !== cb) return ca - cb;
    if (a.probe.provider.id !== b.probe.provider.id) {
      return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    }
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  const fresh = all.filter((c) => (consecutiveFailures.get(c.probe.provider.id) ?? 0) < 2);
  return fresh.length > 0 ? fresh : all;
}

/** The requirements of the request now arriving: the manifest's row by arrival order. */
function takeRequirements(): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return {};
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(norm: NormRequest): Requirement {
  let vision = false;
  let textLength = norm.system.length;
  for (const message of norm.messages) {
    for (const part of message.parts) {
      if (part.type === 'image' || part.type === 'imageUrl') vision = true;
      if (part.type === 'text') textLength += part.text.length;
    }
  }
  const requires: Requirement = {
    tools: norm.tools.length > 0,
    vision,
    json: norm.jsonMode,
  };
  // A long prompt needs a long-context model; ~4 characters a token is the provider's own
  // rough count. Only a fallback — the manifest is the declared source of requirements.
  if (textLength / 4 > 131072) requires.min_context = 200000;
  return requires;
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

interface UpstreamResult {
  ok: boolean;
  answer: NormAnswer | null;
  status: number;
}

/** One infer call to one candidate, in that provider's own wire. */
async function callCandidate(
  candidate: Candidate,
  norm: NormRequest,
): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (norm.stream) headers.accept = 'text/event-stream';
    body = renderOpenAI(norm, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (norm.stream) headers.accept = 'text/event-stream';
    body = renderAnthropic(norm, candidate.model.id);
  } else {
    url = norm.stream
      ? `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:streamGenerateContent`
      : `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
    headers.accept = 'application/json';
    body = renderGemini(norm);
  }
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    log(`${provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
    return { ok: false, answer: null, status: 0 };
  }
  const text = await res.text();
  if (!res.ok) {
    return { ok: false, answer: null, status: res.status };
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, norm.stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, norm.stream)
        : parseGeminiAnswer(text, norm.stream);
  return { ok: true, answer, status: res.status };
}

interface Served {
  answer: NormAnswer;
  model: string;
}

/**
 * Serve one normalized request: the best free model that meets it, with failover. A failed
 * candidate is tried once more only when it is strictly better than the next untried one:
 * retrying can then recover the top rating at the cost of one call, and when it succeeds the
 * call count is the same as falling back. When the next candidate ties (zai and cloudflare
 * both serve glm-4.7-flash at 1470), falling back costs the same call and keeps the rating, so
 * no retry is spent. Returns null when no candidate in the directory answered.
 */
async function serve(norm: NormRequest, requires: Requirement): Promise<Served | null> {
  const candidates = candidatesFor(requires, norm.jsonMode);
  let attempts = 0;
  for (let i = 0; i < candidates.length; i++) {
    const candidate = candidates[i];
    const next = candidates[i + 1];
    const retryWorthIt = next === undefined || next.rating < candidate.rating;
    const tries = retryWorthIt ? 2 : 1;
    for (let retry = 0; retry < tries && attempts < MAX_ATTEMPTS; retry++) {
      attempts += 1;
      const result = await callCandidate(candidate, norm);
      const providerId = candidate.probe.provider.id;
      if (result.ok && result.answer !== null) {
        consecutiveFailures.set(providerId, 0);
        return { answer: result.answer, model: candidate.model.id };
      }
      consecutiveFailures.set(providerId, (consecutiveFailures.get(providerId) ?? 0) + 1);
      log(
        `${providerId}/${candidate.model.id} refused (HTTP ${result.status || 'timeout'}), ` +
          `attempt ${attempts} for this request`,
      );
    }
    if (attempts >= MAX_ATTEMPTS) break;
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered on any upstream wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const requires = takeRequirements();
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const norm = parseOpenAIRequest(body);
  const served = await serve(norm, Object.keys(requires).length > 0 ? requires : requirementsFromBody(norm));
  if (served === null) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  }
  if (norm.stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    res.end(renderOpenAIStream(served.answer, served.model));
    return;
  }
  sendJson(res, 200, renderOpenAIJson(served.answer, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered on any upstream wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const requires = takeRequirements();
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const norm = parseAnthropicRequest(body);
  const served = await serve(norm, Object.keys(requires).length > 0 ? requires : requirementsFromBody(norm));
  if (served === null) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  }
  if (norm.stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    res.end(renderAnthropicStream(served.answer, served.model));
    return;
  }
  sendJson(res, 200, renderAnthropicJson(served.answer, served.model));
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
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
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
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  }
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
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
    // The same document the register body carries, byte for byte: print the one built object
    // and stop before any server starts. A process whose stdout is not flushed loses the tail.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
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
