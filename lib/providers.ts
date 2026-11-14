// lib/providers.ts — reading the trial brief and probing the provider catalogs.
//
// A trial hands this server one brief file. The brief is prose with three JSON
// documents embedded in it, each found by its `schema` member:
//   - the provider directory: ten simulated free tiers, each with an id, a wire
//     (openai, anthropic or gemini), an auth scheme, the key variable a real
//     deployment would read, the key the simulation takes, and the base path
//     under the env URL;
//   - the quality table: every model's public rating in milli-points, its
//     context length, whether it takes tools, images and JSON, and whether it
//     is free, plus the best rating per request;
//   - the workload manifest: the twenty requests the client will send, in
//     order, each with its API, its streaming flag, its weight and the
//     requirements a model must meet to serve it.
//
// This file also probes each provider's catalog route in the provider's own
// wire and folds the answers into the one catalog document the server
// registers and prints with --list. Everything here is a pure function of its
// inputs: no clock, no randomness, so the register body and the --list output
// are the same document.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** A JSON object, the only shape this file treats as structured data. */
export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One provider as the directory lists it. `base` is its path under the env URL. */
export interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  keyEnv: string;
  key: string;
}

export interface DirectoryFile {
  schema: string;
  providers: DirectoryProvider[];
}

/** One row of the quality table. */
export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; ratingMilli: number }[];
}

/** What one request needs from its model. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface ManifestRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: ManifestRequest[];
}

/** The brief as read: the directory always, the other two when present. */
export interface Brief {
  directory: DirectoryFile;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** One model as a provider's catalog route listed it. */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered. */
export interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

/** One model in the registered catalog. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What --list prints and what the register body carries: one document. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** A model's capabilities, from the quality table first, the listing second. */
export interface Capabilities {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asNumberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asBoolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** The index of the `}` that closes the object opening at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in file order. */
function documentsIn(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a standalone object: keep scanning.
    }
  }
  return found;
}

function directoryProviderOf(value: unknown): DirectoryProvider | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  const wire = value.wire === 'openai' || value.wire === 'anthropic' || value.wire === 'gemini' ? value.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme =
    value.auth === 'bearer' || value.auth === 'x-api-key' || value.auth === 'x-goog-api-key' || value.auth === 'none' ? value.auth : 'bearer';
  // The directory writes `path` (/p/<id>, joined to the env URL); a directory
  // spelling an absolute base instead may use base_path, base_url or base.
  const base = asString(value.path) !== '' ? asString(value.path) : asString(value.base_path) !== '' ? asString(value.base_path) : asString(value.base_url) !== '' ? asString(value.base_url) : asString(value.base);
  return {
    id,
    name: asString(value.name) !== '' ? asString(value.name) : id,
    wire,
    auth,
    base,
    keyEnv: asString(value.key_env),
    key: asString(value.key),
  };
}

function ratingRowOf(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = asString(value.model);
  const provider = asString(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: typeof value.rating_milli === 'number' && Number.isFinite(value.rating_milli) ? value.rating_milli : 0,
    contextLength: asNumberOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function manifestRequestOf(value: unknown): ManifestRequest | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { minContext: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`: its three documents, found by schema. */
export function readBrief(path: string): Brief {
  const empty: DirectoryFile = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') return { directory: empty, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory: empty, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null) {
    return { directory: empty, ratings: null, workload: null, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = directoryProviderOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingsFile | null =
    ratingsDoc === null
      ? null
      : {
          schema: RATINGS_SCHEMA,
          ratings: (ratingsDoc.ratings as unknown[]).flatMap((v) => {
            const r = ratingRowOf(v);
            return r === null ? [] : [r];
          }),
          best: Array.isArray(ratingsDoc.best)
            ? (ratingsDoc.best as unknown[]).flatMap((v) => {
                if (!isRecord(v) || asString(v.id) === '') return [];
                return [{ id: asString(v.id), ratingMilli: asNumberOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadFile | null =
    workloadDoc === null
      ? null
      : {
          schema: WORKLOAD_SCHEMA,
          manifest: (workloadDoc.manifest as unknown[]).flatMap((v) => {
            const w = manifestRequestOf(v);
            return w === null ? [] : [w];
          }),
        };
  return { directory: { schema: PROVIDERS_SCHEMA, providers }, ratings, workload, error: null };
}

/** The quality table's row for one provider's model, when it has one. */
export function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed
 * truth), the listing's own word where it does not, paid when neither says.
 * A paid model is never routed.
 */
export function modelFree(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, else the listing. */
export function capabilitiesOf(model: ListedModel, row: RatingRow | null): Capabilities {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. Unknown never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) return false;
  return true;
}

/** A base path joined to the env URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return prefix;
  return `${prefix}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

/** The headers one provider wants its key in. The wire never decides the header. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Whether a listing's prices are all zero: true when it prices everything at 0. */
function zeroPriced(value: Json): boolean | null {
  const pricing = isRecord(value.pricing) ? value.pricing : null;
  if (pricing === null) return null;
  const numbers = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (numbers.length === 0) return null;
  return numbers.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, value: unknown): ListedModel | null {
  if (!isRecord(value)) return null;
  const id = wire === 'gemini' ? asString(value.name).replace(/^models\//, '') : asString(value.id);
  if (id === '') return null;
  const context = asNumberOrNull(value.context_length) ?? asNumberOrNull(value.context_window) ?? asNumberOrNull(value.inputTokenLimit);
  return {
    id,
    displayName:
      asString(value.display_name) !== '' ? asString(value.display_name) : asString(value.displayName) !== '' ? asString(value.displayName) : id,
    free: asBoolOrNull(value.free) ?? zeroPriced(value),
    contextLength: context,
    tools: asBoolOrNull(value.tools),
    vision: asBoolOrNull(value.vision),
    json: asBoolOrNull(value.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire; any failure leaves it out. */
export async function probeProvider(envUrl: string, provider: DirectoryProvider, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  // A base that already ends in the version path takes no second copy of it.
  const url =
    provider.wire === 'gemini'
      ? base.endsWith('/v1beta')
        ? `${base}/models`
        : `${base}/v1beta/models`
      : base.endsWith('/v1')
        ? `${base}/models`
        : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, sorted by id within its provider.
 *
 * Its base_url is the directory's own path (/p/<id>), never the path joined to
 * the env URL: the env's origin differs between a live pass and its replay,
 * and the catalog is part of the state export a replay must reproduce byte
 * for byte.
 */
export function buildCatalog(probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  const byId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.contextLength ?? model.contextLength ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.keyEnv, wire: probe.provider.wire, models });
  }
  return { providers };
}
