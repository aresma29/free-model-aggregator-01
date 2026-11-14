// main.ts — the server entry: the routes, the routing, and the startup.
//
// This is the whole program: it reads the brief, probes the ten providers,
// registers the catalog it found, and serves the two APIs for the rest of
// the trial. The pieces live in lib/:
//   - lib/providers.ts reads the brief, probes each provider's catalog route
//     in its own wire and builds the catalog document;
//   - lib/wire.ts normalizes every request and every answer and translates
//     between the six client × upstream paths.
//
// Routing, in one paragraph: for each request the server lists every free
// model that meets the manifest's requirements for that request, best rating
// first, and calls them one at a time. A 402, a 429 or a 5xx never ends the
// request: the failed candidate is retried once when it is strictly better
// than the next untried candidate, and the server falls back on a tie; a
// provider is skipped after two consecutive failures while another candidate
// remains; no request makes more than sixteen upstream calls, and no call
// runs longer than ten seconds. Exactly one upstream call succeeds per
// request, and that answer — text and usage mapped exactly — is what the
// client gets back, in the client's own wire.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { randomUUID } from 'node:crypto';
import { argv, env, exit } from 'node:process';
import { buildCatalog, joinBase, probeProvider, readBrief, meetsRequirements, ratingOfModel, modelFree, capabilitiesOf, type Brief, type CatalogProvider, type DirectoryProvider, type ManifestRequest, type Probe, type ListedModel, type RatingRow, type CatalogReport } from './lib/providers.ts';
import {
  parseOpenAIRequest,
  parseAnthropicRequest,
  encodeOpenAIRequest,
  encodeAnthropicRequest,
  encodeGeminiRequest,
  parseOpenAIAnswer,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIStream,
  parseAnthropicStream,
  parseGeminiStream,
  emitOpenAIJson,
  emitOpenAIStream,
  emitAnthropicJson,
  emitAnthropicStream,
  isRecord,
  type NormalizedRequest,
  type NormalizedAnswer,
  type Wire,
} from './lib/wire.ts';

/** The knobs the server runs on: a port, a catalog, and a failure budget. */
interface ServerConfig {
  port: number;
  envUrl: string;
  catalog: CatalogReport;
  brief: Brief;
  probes: Probe[];
}

/** One free model that may serve a request, with its rating and its owner. */
interface Candidate {
  provider: DirectoryProvider;
  model: ListedModel;
  ratingMilli: number;
  row: RatingRow | null;
}

const CALL_TIMEOUT_MS = 10_000;
const MAX_ATTEMPTS = 16;

function envValue(name: string): string {
  const value = env[name];
  return typeof value === 'string' ? value.trim() : '';
}

function parsePort(): number {
  const raw = envValue('PORT');
  const parsed = raw === '' ? NaN : Number(raw);
  return Number.isFinite(parsed) && parsed > 0 && parsed < 65536 ? parsed : 3000;
}

/** The prompt file, the env URL, the port, or a --list run. */
function parseArgs(): { list: boolean; promptFile: string; envUrl: string; port: number } {
  const args = argv.slice(2);
  const list = args.includes('--list');
  let promptFile = envValue('MAKEFUN_PROMPT_FILE');
  let envUrl = envValue('MAKEFUN_ENV_URL');
  let port = parsePort();
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--prompt-file' && args[i + 1] !== undefined) promptFile = args[i + 1];
    else if (arg === '--env-url' && args[i + 1] !== undefined) envUrl = args[i + 1];
    else if (arg === '--port' && args[i + 1] !== undefined) {
      const parsed = Number(args[i + 1]);
      if (Number.isFinite(parsed) && parsed > 0 && parsed < 65536) port = parsed;
    }
  }
  return { list, promptFile, envUrl, port };
}

/** The wire a provider speaks, as the directory recorded it. */
function wireOf(provider: CatalogProvider | DirectoryProvider): Wire {
  return provider.wire === 'anthropic' || provider.wire === 'gemini' ? provider.wire : 'openai';
}

/** The headers one provider's auth wants, for a chat or a catalog call. */
function providerHeaders(provider: DirectoryProvider, extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  if (provider.key === '') return headers;
  if (provider.auth === 'bearer') headers.authorization = `Bearer ${provider.key}`;
  else if (provider.auth === 'x-api-key') headers['x-api-key'] = provider.key;
  else if (provider.auth === 'x-goog-api-key') headers['x-goog-api-key'] = provider.key;
  return headers;
}

/** A model's rating from the quality table, or zero when the table is silent. */
function ratingMilliOf(row: RatingRow | null): number {
  return row === null ? 0 : row.ratingMilli;
}

/**
 * The server's config: the brief, the probe of every provider in it, and
 * the catalog those probes make. Providers are probed in parallel — the
 * catalog probe is not a request, so the sequential rule does not apply.
 */
async function loadConfig(promptFile: string, envUrl: string, port: number): Promise<ServerConfig> {
  const brief = readBrief(promptFile);
  const probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(envUrl, provider)));
  const catalog = buildCatalog(probes, brief.ratings);
  return { port, envUrl, catalog, brief, probes };
}

/** The model list for one client wire: id, object, created, owned_by. */
function listModelsFor(catalog: CatalogReport, wire: Wire): Record<string, unknown>[] {
  const models: Record<string, unknown>[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free) continue;
      models.push({ id: model.id, object: 'model', created: 0, owned_by: provider.id });
    }
  }
  models.sort((a, b) => {
    const aId = typeof a.id === 'string' ? a.id : '';
    const bId = typeof b.id === 'string' ? b.id : '';
    return aId < bId ? -1 : aId > bId ? 1 : 0;
  });
  return models;
}

/**
 * Every free model that may serve a request: listed, free by the quality
 * table, and meeting the request's requirements. Best rating first; ties go
 * to the least-called provider, then by provider/model id — a total order,
 * so the choice is deterministic. JSON mode is never routed to an
 * anthropic-wire provider: its echo cannot carry it.
 */
function candidatesFor(config: ServerConfig, requires: ManifestRequest['requires'], jsonMode: boolean, calls: Map<string, number>): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of config.probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(config.brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (jsonMode && wireOf(probe.provider) === 'anthropic') continue;
      out.push({ provider: probe.provider, model, ratingMilli: ratingMilliOf(row), row });
    }
  }
  out.sort((a, b) => {
    if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
    const aCalls = calls.get(a.provider.id) ?? 0;
    const bCalls = calls.get(b.provider.id) ?? 0;
    if (aCalls !== bCalls) return aCalls - bCalls;
    const aKey = `${a.provider.id}/${a.model.id}`;
    const bKey = `${b.provider.id}/${b.model.id}`;
    return aKey < bKey ? -1 : aKey > bKey ? 1 : 0;
  });
  return out;
}

/** The manifest's requirements for the request now being served. */
function requirementsFor(brief: Brief, index: number, clientApi: 'openai' | 'anthropic'): ManifestRequest['requires'] {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry !== undefined && entry.api === clientApi) return entry.requires;
  return {};
}

/** One upstream call: encode the request in the provider's wire, send it, parse the answer. */
async function callCandidate(
  config: ServerConfig,
  candidate: Candidate,
  req: NormalizedRequest,
): Promise<{ ok: true; answer: NormalizedAnswer; status: number } | { ok: false; status: number; retryable: boolean }> {
  const wire = wireOf(candidate.provider);
  const body = wire === 'openai' ? encodeOpenAIRequest(req, candidate.model.id) : wire === 'anthropic' ? encodeAnthropicRequest(req, candidate.model.id) : encodeGeminiRequest(req, candidate.model.id);
  const base = joinBase(config.envUrl, candidate.provider.base);
  const url =
    wire === 'gemini'
      ? `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`
      : wire === 'anthropic'
        ? `${base}/v1/messages`
        : `${base}/v1/chat/completions`;
  const headers = providerHeaders(candidate.provider, { 'content-type': 'application/json', accept: 'application/json' });
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { ok: false, status: 0, retryable: true };
  }
  if (!res.ok) {
    const retryable = res.status === 402 || res.status === 429 || res.status >= 500;
    return { ok: false, status: res.status, retryable };
  }
  const text = await res.text();
  const contentType = res.headers.get('content-type') ?? '';
  if (contentType.includes('text/event-stream') || text.includes('event:') || text.includes('data:')) {
    const answer = wire === 'openai' ? parseOpenAIStream(text) : wire === 'anthropic' ? parseAnthropicStream(text) : parseGeminiStream(text);
    return { ok: true, answer, status: res.status };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, status: 0, retryable: true };
  }
  const answer = wire === 'openai' ? parseOpenAIAnswer(parsed) : wire === 'anthropic' ? parseAnthropicAnswer(parsed) : parseGeminiAnswer(parsed);
  return { ok: true, answer, status: res.status };
}

/**
 * The routing loop. Candidates are tried one at a time, best first, and a
 * failure never ends the request: the failed candidate is retried once when
 * it is strictly better than the next untried candidate — on a tie the server
 * falls back instead — and a provider is skipped after two consecutive
 * failures while another candidate remains. Sixteen attempts, ten seconds a
 * call, and exactly one successful upstream call per request.
 */
async function serveChat(
  config: ServerConfig,
  req: NormalizedRequest,
  clientApi: 'openai' | 'anthropic',
  requestIndex: number,
  calls: Map<string, number>,
): Promise<NormalizedAnswer> {
  const requires = requirementsFor(config.brief, requestIndex, clientApi);
  const candidates = candidatesFor(config, requires, req.jsonMode, calls);
  const failures = new Map<string, number>();
  const retried = new Set<string>();
  let attempts = 0;
  let next = 0;
  while (attempts < MAX_ATTEMPTS) {
    while (next < candidates.length && (failures.get(candidates[next].provider.id) ?? 0) >= 2) next += 1;
    if (next >= candidates.length) break;
    const candidate = candidates[next];
    const key = `${candidate.provider.id}/${candidate.model.id}`;
    attempts += 1;
    calls.set(candidate.provider.id, (calls.get(candidate.provider.id) ?? 0) + 1);
    const result = await callCandidate(config, candidate, req);
    if (result.ok) return result.answer;
    failures.set(candidate.provider.id, (failures.get(candidate.provider.id) ?? 0) + 1);
    const following = candidates[next + 1];
    const strictlyBetter = following === undefined || candidate.ratingMilli > following.ratingMilli;
    if (result.retryable && !retried.has(key) && strictlyBetter) {
      retried.add(key);
      continue;
    }
    next += 1;
  }
  throw new Error('all candidates failed');
}

/** The body of a request, as JSON, or null when it is not JSON. */
async function readBody(req: IncomingMessage): Promise<Record<string, unknown> | null> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  const text = Buffer.concat(chunks).toString('utf8');
  if (text.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(text);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, status: number, message: string): void {
  sendJson(res, status, { error: { message, type: 'server_error' } });
}

/** The model list for a client wire, from the auth it presented. */
function modelsResponse(config: ServerConfig, req: IncomingMessage): Record<string, unknown> {
  const wire: Wire = typeof req.headers['x-api-key'] === 'string' && req.headers['x-api-key'] !== '' ? 'anthropic' : 'openai';
  return { object: 'list', data: listModelsFor(config.catalog, wire) };
}

/** The whole server: routes, routing, and the startup register. */
async function main(): Promise<void> {
  const args = parseArgs();
  const config = await loadConfig(args.promptFile, args.envUrl, args.port);
  if (args.list) {
    process.stdout.write(`${JSON.stringify(config.catalog)}\n`);
    return;
  }
  const calls = new Map<string, number>();
  let requestIndex = 0;
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost');
      const path = url.pathname;
      if (req.method === 'GET' && path === '/v1/models') {
        sendJson(res, 200, modelsResponse(config, req));
        return;
      }
      if (req.method === 'GET' && path === '/health') {
        sendJson(res, 200, { status: 'ok' });
        return;
      }
      if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
        const clientApi = path === '/v1/messages' ? 'anthropic' : 'openai';
        const body = await readBody(req);
        if (body === null) {
          sendError(res, 400, 'request body is not a JSON object');
          return;
        }
        const normalized: NormalizedRequest = clientApi === 'anthropic' ? parseAnthropicRequest(body) : parseOpenAIRequest(body);
        const index = requestIndex;
        requestIndex += 1;
        const answer = await serveChat(config, normalized, clientApi, index, calls);
        const id = `chatcmpl-${randomUUID()}`;
        const model = typeof body.model === 'string' && body.model !== '' ? body.model : 'unknown';
        if (normalized.stream) {
          sendSse(res, clientApi === 'anthropic' ? emitAnthropicStream(answer, model, id) : emitOpenAIStream(answer, model, id));
        } else {
          sendJson(res, 200, clientApi === 'anthropic' ? emitAnthropicJson(answer, model, id) : emitOpenAIJson(answer, model, id));
        }
        return;
      }
      sendError(res, 404, `no route ${req.method ?? ''} ${path}`);
    } catch (err) {
      sendError(res, 502, (err as Error).message);
    }
  });
  await new Promise<void>((resolve) => {
    server.listen(config.port, () => resolve());
  });
  // The trial's register call: who this server is, on which port, with which
  // catalog. The env URL is where the register endpoint lives.
  const registerUrl = `${config.envUrl.replace(/\/+$/, '')}/register`;
  try {
    await fetch(registerUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ protocol_version: 1, port: config.port, catalog: config.catalog }),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    // A failed register is the harness's to report; the server still serves.
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`fatal: ${(err as Error).message}\n`);
  exit(1);
});
