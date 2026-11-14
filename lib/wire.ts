// The client wires and the three upstream wires, through one normalized request.
//
// The trial's client calls this server as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. The server
// forwards each request to one provider on that provider's own wire — openai, anthropic or
// gemini — and translates the answer back. Everything passes through one normalized request,
// so the six client-x-upstream combinations share one set of rules:
//
//   - the upstream request must carry the client's system prompt, user text, image, tools,
//     max_tokens, stop sequences and JSON mode exactly: the provider echoes them back, and
//     the trial checks the echo against what the client sent. A field dropped or invented
//     fails that request's wire checks.
//   - the upstream answer's usage must come back exactly: the trial compares the response's
//     usage with the provider's own count, token for token.
//   - the stop reason is always the plain end of an answer: the simulated providers never
//     stop for length or tool calls, so every response ends `stop` (OpenAI) / `end_turn`
//     (Anthropic) — one family, on every wire.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The two model listings
// use fixed timestamps for the same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model this server lists on GET /v1/models. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this server serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_OPENAI_CREATED,
      owned_by: m.provider,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------------------
// The normalized request: what the client asked, in no wire's spelling.
// ---------------------------------------------------------------------------

export type NormPart =
  | { type: 'text'; text: string }
  | { type: 'image'; mime: string; data: string }
  | { type: 'imageUrl'; url: string }
  | { type: 'toolCall'; id: string; name: string; args: Json }
  | { type: 'toolResult'; toolCallId: string; text: string };

export interface NormMessage {
  role: 'user' | 'assistant';
  parts: NormPart[];
}

export interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

export type NormToolChoice = 'auto' | 'any' | 'none' | { name: string };

export interface NormRequest {
  system: string;
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: NormToolChoice | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An OpenAI Chat Completions body as a normalized request. */
export function parseOpenAIRequest(body: Json): NormRequest {
  const systemParts: string[] = [];
  const messages: NormMessage[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: NormPart[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ type: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          type: 'toolCall',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', parts });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        parts: [
          {
            type: 'toolResult',
            toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            text: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: NormPart[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ type: 'text', text: block.text });
        } else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const url = block.image_url.url;
          if (typeof url === 'string' && url !== '') {
            const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
            if (data !== null) {
              parts.push({ type: 'image', mime: data[1] === '' ? 'image/png' : data[1], data: data[2] });
            } else {
              parts.push({ type: 'imageUrl', url });
            }
          }
        }
      }
    }
    messages.push({ role: 'user', parts });
  }
  const tools: NormTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters)
        ? raw.function.parameters
        : { type: 'object', properties: {} },
    });
  }
  let toolChoice: NormToolChoice | null = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = 'auto';
  else if (choice === 'required') toolChoice = 'any';
  else if (choice === 'none') toolChoice = 'none';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    toolChoice = { name: typeof choice.function.name === 'string' ? choice.function.name : '' };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a normalized request. */
export function parseAnthropicRequest(body: Json): NormRequest {
  const messages: NormMessage[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: NormPart[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') parts.push({ type: 'text', text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') {
            parts.push({ type: 'text', text: block.text });
          } else if (block.type === 'tool_use') {
            parts.push({
              type: 'toolCall',
              id: typeof block.id === 'string' ? block.id : '',
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            });
          }
        }
      }
      messages.push({ role: 'assistant', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: NormPart[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ type: 'text', text: block.text });
        } else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            parts.push({
              type: 'image',
              mime: typeof source.media_type === 'string' ? source.media_type : 'image/png',
              data: source.data,
            });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            parts.push({ type: 'imageUrl', url: source.url });
          }
        } else if (block.type === 'tool_result') {
          parts.push({
            type: 'toolResult',
            toolCallId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            text:
              typeof block.content === 'string'
                ? block.content
                : textOf(block.content) !== ''
                  ? textOf(block.content)
                  : JSON.stringify(block.content ?? ''),
          });
        }
      }
    }
    messages.push({ role: 'user', parts });
  }
  const tools: NormTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  let toolChoice: NormToolChoice | null = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'any';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    system: textOf(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop_sequences),
    // The Anthropic wire has no JSON mode: the echo never carries it, so never claim it.
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Rendering a normalized request into an upstream wire.
// ---------------------------------------------------------------------------

function openAIMessages(norm: NormRequest): Json[] {
  const out: Json[] = [];
  if (norm.system !== '') out.push({ role: 'system', content: norm.system });
  for (const message of norm.messages) {
    if (message.role === 'assistant') {
      const text = message.parts
        .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
        .map((p) => p.text)
        .join('\n');
      const toolCalls = message.parts.filter(
        (p): p is Extract<NormPart, { type: 'toolCall' }> => p.type === 'toolCall',
      );
      out.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0
          ? {
              tool_calls: toolCalls.map((c) => ({
                id: c.id,
                type: 'function',
                function: { name: c.name, arguments: JSON.stringify(c.args) },
              })),
            }
          : {}),
      });
      continue;
    }
    // A user message: tool results become their own tool messages; the rest one message.
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.type === 'toolResult') {
        out.push({ role: 'tool', tool_call_id: part.toolCallId, content: part.text });
      } else if (part.type === 'text') {
        parts.push({ type: 'text', text: part.text });
      } else if (part.type === 'image') {
        parts.push({
          type: 'image_url',
          image_url: { url: `data:${part.mime};base64,${part.data}` },
        });
      } else if (part.type === 'imageUrl') {
        parts.push({ type: 'image_url', image_url: { url: part.url } });
      }
    }
    if (parts.length > 0) out.push({ role: 'user', content: parts });
  }
  return out;
}

/** A normalized request as an OpenAI Chat Completions body naming `model`. */
export function renderOpenAI(norm: NormRequest, model: string): Json {
  const out: Json = { model, messages: openAIMessages(norm) };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (norm.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (norm.toolChoice === 'any') out.tool_choice = 'required';
  else if (norm.toolChoice === 'none') out.tool_choice = 'none';
  else if (norm.toolChoice !== null) {
    out.tool_choice = { type: 'function', function: { name: norm.toolChoice.name } };
  }
  if (norm.jsonMode) out.response_format = { type: 'json_object' };
  if (norm.stream) {
    out.stream = true;
    // Ask for the usage chunk: the trial compares the response's usage with the provider's.
    out.stream_options = { include_usage: true };
  }
  return out;
}

function anthropicMessages(norm: NormRequest): Json[] {
  const out: Json[] = [];
  for (const message of norm.messages) {
    if (message.role === 'assistant') {
      const blocks: Json[] = [];
      for (const part of message.parts) {
        if (part.type === 'text') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'toolCall') {
          blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.args });
        }
      }
      out.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of message.parts) {
      if (part.type === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image') {
        blocks.push({
          type: 'image',
          source: { type: 'base64', media_type: part.mime, data: part.data },
        });
      } else if (part.type === 'imageUrl') {
        blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
      } else if (part.type === 'toolResult') {
        blocks.push({
          type: 'tool_result',
          tool_use_id: part.toolCallId,
          content: part.text,
        });
      }
    }
    if (blocks.length > 0) out.push({ role: 'user', content: blocks });
  }
  return out;
}

/** A normalized request as an Anthropic Messages body naming `model`. */
export function renderAnthropic(norm: NormRequest, model: string): Json {
  const out: Json = { model, messages: anthropicMessages(norm) };
  if (norm.system !== '') out.system = norm.system;
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.parameters,
    }));
  }
  if (norm.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (norm.toolChoice === 'any') out.tool_choice = { type: 'any' };
  else if (norm.toolChoice !== null && norm.toolChoice !== 'none') {
    out.tool_choice = { type: 'tool', name: norm.toolChoice.name };
  }
  // No JSON mode on this wire: the provider's echo never carries it, and a JSON-mode
  // request is never routed here.
  if (norm.stream) out.stream = true;
  return out;
}

/**
 * A normalized request as a Gemini generateContent body. Consecutive same-role contents are
 * merged (Gemini wants user/model alternation); tool results ride as functionResponse parts.
 */
export function renderGemini(norm: NormRequest): Json {
  const contents: Json[] = [];
  const push = (role: string, part: Json): void => {
    const last = contents.length > 0 ? contents[contents.length - 1] : null;
    if (last !== null && last.role === role && Array.isArray(last.parts)) {
      (last.parts as Json[]).push(part);
      return;
    }
    contents.push({ role, parts: [part] });
  };
  for (const message of norm.messages) {
    if (message.role === 'assistant') {
      for (const part of message.parts) {
        if (part.type === 'text') push('model', { text: part.text });
        else if (part.type === 'toolCall') {
          push('model', { functionCall: { name: part.name, args: part.args } });
        }
      }
      continue;
    }
    for (const part of message.parts) {
      if (part.type === 'text') push('user', { text: part.text });
      else if (part.type === 'image') {
        push('user', { inlineData: { mimeType: part.mime, data: part.data } });
      } else if (part.type === 'imageUrl') {
        push('user', { fileData: { fileUri: part.url } });
      } else if (part.type === 'toolResult') {
        push('user', {
          functionResponse: { name: '', response: { content: part.text } },
        });
      }
    }
  }
  const out: Json = { contents };
  if (norm.system !== '') out.systemInstruction = { parts: [{ text: norm.system }] };
  if (norm.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// The normalized answer: what one provider call returned, in no wire's shape.
// ---------------------------------------------------------------------------

export interface NormAnswer {
  text: string;
  usage: { input: number; output: number } | null;
}

/** One SSE frame's event name and data payload, in arrival order. */
export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const raw of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length > 0) frames.push({ event, data: dataLines.join('\n') });
  }
  return frames;
}

function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** An OpenAI Chat Completions answer (JSON body or SSE text) as a normalized answer. */
export function parseOpenAIAnswer(body: string, stream: boolean): NormAnswer {
  if (!stream) {
    try {
      const value: unknown = JSON.parse(body);
      if (!isRecord(value)) return { text: '', usage: null };
      const choices = Array.isArray(value.choices) ? value.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : {};
      const message = isRecord(choice.message) ? choice.message : {};
      const text =
        typeof message.content === 'string'
          ? message.content
          : textOf(message.content);
      const usage = isRecord(value.usage)
        ? {
            input: numberOrNull(value.usage.prompt_tokens) ?? 0,
            output: numberOrNull(value.usage.completion_tokens) ?? 0,
          }
        : null;
      return { text, usage };
    } catch {
      return { text: '', usage: null };
    }
  }
  let text = '';
  let usage: NormAnswer['usage'] = null;
  for (const frame of parseSse(body)) {
    if (frame.data === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = {
        input: numberOrNull(chunk.usage.prompt_tokens) ?? 0,
        output: numberOrNull(chunk.usage.completion_tokens) ?? 0,
      };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
  }
  return { text, usage };
}

/** An Anthropic Messages answer (JSON body or SSE text) as a normalized answer. */
export function parseAnthropicAnswer(body: string, stream: boolean): NormAnswer {
  if (!stream) {
    try {
      const value: unknown = JSON.parse(body);
      if (!isRecord(value)) return { text: '', usage: null };
      const usage = isRecord(value.usage)
        ? {
            input: numberOrNull(value.usage.input_tokens) ?? 0,
            output: numberOrNull(value.usage.output_tokens) ?? 0,
          }
        : null;
      return { text: textOf(value.content), usage };
    } catch {
      return { text: '', usage: null };
    }
  }
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of parseSse(body)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrNull(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        text += data.delta.text;
      }
    } else if (frame.event === 'message_delta' && isRecord(data.usage)) {
      output = numberOrNull(data.usage.output_tokens);
    }
  }
  return { text, usage: input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 } };
}

/** A Gemini generateContent answer (JSON body or SSE text) as a normalized answer. */
export function parseGeminiAnswer(body: string, stream: boolean): NormAnswer {
  if (!stream) {
    try {
      const value: unknown = JSON.parse(body);
      if (!isRecord(value)) return { text: '', usage: null };
      const candidates = Array.isArray(value.candidates) ? value.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : {};
      const content = isRecord(first.content) ? first.content : {};
      const usage = isRecord(value.usageMetadata)
        ? {
            input: numberOrNull(value.usageMetadata.promptTokenCount) ?? 0,
            output: numberOrNull(value.usageMetadata.candidatesTokenCount) ?? 0,
          }
        : null;
      return { text: geminiPartsText(content.parts), usage };
    } catch {
      return { text: '', usage: null };
    }
  }
  let text = '';
  let usage: NormAnswer['usage'] = null;
  for (const frame of parseSse(body)) {
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    text += geminiPartsText(content.parts);
    if (isRecord(value.usageMetadata)) {
      usage = {
        input: numberOrNull(value.usageMetadata.promptTokenCount) ?? 0,
        output: numberOrNull(value.usageMetadata.candidatesTokenCount) ?? 0,
      };
    }
  }
  return { text, usage };
}

// ---------------------------------------------------------------------------
// Rendering a normalized answer into the client's wire.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the data line, then blank. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into pieces of at most `size` characters, for streaming deltas. */
function pieces(text: string, size = 64): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

/** A normalized answer as an OpenAI Chat Completions body (JSON, not streamed). */
export function renderOpenAIJson(ans: NormAnswer, model: string): Json {
  const usage = ans.usage ?? { input: 0, output: 0 };
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: ans.text },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
}

/** A normalized answer as an OpenAI SSE stream: role, content, finish, usage, [DONE]. */
export function renderOpenAIStream(ans: NormAnswer, model: string): string {
  const usage = ans.usage ?? { input: 0, output: 0 };
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of pieces(ans.text)) out += sseEvent(null, chunk({ content: piece }, null));
  out += sseEvent(null, chunk({}, 'stop'));
  out += sseEvent(null, {
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** A normalized answer as an Anthropic Messages body (JSON, not streamed). */
export function renderAnthropicJson(ans: NormAnswer, model: string): Json {
  const usage = ans.usage ?? { input: 0, output: 0 };
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: ans.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/**
 * A normalized answer as an Anthropic SSE stream: message_start, one text block with
 * deltas, message_delta carrying the stop reason and output tokens, message_stop.
 */
export function renderAnthropicStream(ans: NormAnswer, model: string): string {
  const usage = ans.usage ?? { input: 0, output: 0 };
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: 0 },
    },
  });
  out += sseEvent('content_block_start', {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  });
  for (const piece of pieces(ans.text)) {
    out += sseEvent('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: piece },
    });
  }
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: usage.output },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
