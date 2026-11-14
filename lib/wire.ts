// lib/wire.ts — the six wires, and the normalization they all pass through.
//
// A client speaks OpenAI or Anthropic. An upstream speaks OpenAI, Anthropic
// or Gemini. So there are six client × upstream paths, but they are one job
// done twice: first a client's request is normalized — its system prompt, its
// messages (text, image, tool-call and tool-result parts), its tools, its tool
// choice, its max_tokens, its temperature and top_p, its stop sequences, its
// JSON mode — into one internal form, and an upstream answer is normalized
// into one internal form (text, plus its exact usage). Then the internal form
// is translated out to the upstream's wire, and the upstream's answer is
// translated into the client's wire.
//
// Streaming is read-then-re-emit. An upstream stream is read fully, its
// tokens and its usage collected, and only then is the client stream written
// out as fresh client chunks: OpenAI gets its role, content, finish-reason
// and usage chunks and its [DONE]; Anthropic gets message_start, the content
// blocks, the deltas, message_delta and message_stop.
//
// The two rules that cost points when they slip: usage is mapped exactly on
// every wire (prompt/completion ↔ input/output ↔ promptTokenCount/
// candidatesTokenCount), and the stop reason is always the plain stop family —
// "stop" for OpenAI, "end_turn" for Anthropic, never a "length" or a model's
// own spelling.

/** One piece of a message: text, an image, a tool call, or a tool's result. */
export type NormalizedPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mimeType: string; data: string }
  | { kind: 'toolCall'; id: string; name: string; arguments: string }
  | { kind: 'toolResult'; toolCallId: string; content: string };

export interface NormalizedMessage {
  role: 'user' | 'assistant' | 'system';
  parts: NormalizedPart[];
}

export interface NormalizedTool {
  name: string;
  description: string | null;
  schema: Record<string, unknown>;
}

export type NormalizedToolChoice = { mode: 'auto' } | { mode: 'any' } | { mode: 'none' } | { mode: 'tool'; name: string };

/** A request in the one internal form every wire translates out of. */
export interface NormalizedRequest {
  system: string | null;
  messages: NormalizedMessage[];
  tools: NormalizedTool[];
  toolChoice: NormalizedToolChoice;
  maxTokens: number;
  temperature: number | null;
  topP: number | null;
  stopSequences: string[];
  jsonMode: boolean;
  stream: boolean;
}

export type Wire = 'openai' | 'anthropic' | 'gemini';

/** A usage block: the words differ per wire, the numbers do not. */
export interface Usage {
  prompt: number;
  completion: number;
}

export type StopReason = 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'error';

/** An answer in the one internal form every wire translates into. */
export interface NormalizedAnswer {
  text: string;
  usage: Usage | null;
  stopReason: StopReason;
  toolCalls: { id: string; name: string; arguments: string }[];
}

export type ToolCall = { id: string; name: string; arguments: string };

export function emptyUsage(): Usage {
  return { prompt: 0, completion: 0 };
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isArray(value: unknown): value is unknown[] {
  return Array.isArray(value);
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * A token count as one wire's spelling: output_tokens (Anthropic),
 * outputTokens (Gemini), completion_tokens (OpenAI), or a bare `output`.
 */
function tokensOut(value: unknown): number | null {
  if (!isRecord(value)) return null;
  return asNumber(value.output_tokens) ?? asNumber(value.outputTokens) ?? asNumber(value.completion_tokens) ?? asNumber(value.output);
}

/**
 * A prompt token count: input_tokens (Anthropic), promptTokens (Gemini),
 * prompt_tokens (OpenAI), or a bare `input`.
 */
function tokensIn(value: unknown): number | null {
  if (!isRecord(value)) return null;
  return asNumber(value.input_tokens) ?? asNumber(value.promptTokens) ?? asNumber(value.prompt_tokens) ?? asNumber(value.input);
}

/** A usage block in any wire's spelling, mapped exactly onto prompt/completion. */
export function usageOf(value: unknown): Usage | null {
  if (!isRecord(value)) return null;
  const prompt = tokensIn(value);
  const completion = tokensOut(value);
  if (prompt === null && completion === null) return null;
  return { prompt: prompt ?? 0, completion: completion ?? 0 };
}

export function addUsage(a: Usage | null, b: Usage | null): Usage | null {
  if (a === null) return b;
  if (b === null) return a;
  return { prompt: a.prompt + b.prompt, completion: a.completion + b.completion };
}

function toolCallOf(id: string, name: string, args: unknown): ToolCall {
  return { id, name, arguments: typeof args === 'string' ? args : JSON.stringify(args ?? {}) };
}

/** A content part list, in either wire's spelling, as normalized parts. */
function partsOf(raw: unknown): NormalizedPart[] {
  if (typeof raw === 'string') return raw === '' ? [] : [{ kind: 'text', text: raw }];
  if (!isArray(raw)) return [];
  const out: NormalizedPart[] = [];
  for (const part of raw) {
    if (typeof part === 'string') {
      if (part !== '') out.push({ kind: 'text', text: part });
      continue;
    }
    if (!isRecord(part)) continue;
    const type = asString(part.type);
    if (type === 'text') {
      const text = asString(part.text);
      if (text !== '') out.push({ kind: 'text', text });
    } else if (type === 'image' || type === 'image_url' || type === 'input_image') {
      if (isRecord(part.source) && part.source.type === 'base64') {
        out.push({ kind: 'image', mimeType: asString(part.source.media_type), data: asString(part.source.data) });
      } else if (isRecord(part.image_url)) {
        out.push({ kind: 'image', mimeType: '', data: asString(part.image_url.url) });
      }
    } else if (type === 'tool_use') {
      out.push({ kind: 'toolCall', id: asString(part.id), name: asString(part.name), arguments: typeof part.input === 'string' ? part.input : JSON.stringify(part.input ?? {}) });
    } else if (type === 'tool_result') {
      const content = typeof part.content === 'string' ? part.content : isArray(part.content) ? part.content.map((c) => (isRecord(c) ? asString(c.text) : '')).join('') : '';
      out.push({ kind: 'toolResult', toolCallId: asString(part.tool_use_id), content });
    }
  }
  return out;
}

/** An OpenAI message, its `content` string or list and its `tool_calls`. */
function openaiPartsOf(message: Record<string, unknown>): NormalizedPart[] {
  const parts = partsOf(message.content);
  if (isArray(message.tool_calls)) {
    for (const rawCall of message.tool_calls) {
      if (!isRecord(rawCall)) continue;
      const fn = isRecord(rawCall.function) ? rawCall.function : {};
      parts.push({ kind: 'toolCall', id: asString(rawCall.id), name: asString(fn.name), arguments: typeof fn.arguments === 'string' ? fn.arguments : JSON.stringify(fn.arguments ?? {}) });
    }
  }
  return parts;
}

function normalizedToolOf(raw: unknown): NormalizedTool | null {
  if (!isRecord(raw)) return null;
  if (isRecord(raw.function)) {
    return {
      name: asString(raw.function.name),
      description: asString(raw.function.description) !== '' ? asString(raw.function.description) : null,
      schema: isRecord(raw.function.parameters) ? raw.function.parameters : {},
    };
  }
  return {
    name: asString(raw.name),
    description: asString(raw.description) !== '' ? asString(raw.description) : null,
    schema: isRecord(raw.input_schema) ? raw.input_schema : {},
  };
}

function toolChoiceOf(raw: unknown): NormalizedToolChoice {
  if (isRecord(raw)) {
    const type = asString(raw.type);
    if (type === 'auto' || type === 'any') return { mode: 'auto' };
    if (type === 'none') return { mode: 'none' };
    if (type === 'tool' && asString(raw.name) !== '') return { mode: 'tool', name: asString(raw.name) };
  }
  if (raw === 'auto' || raw === 'any') return { mode: 'auto' };
  if (raw === 'none') return { mode: 'none' };
  return { mode: 'auto' };
}

function sharedNumbers(body: Record<string, unknown>): { maxTokens: number; temperature: number | null; topP: number | null } {
  const maxTokens = asNumber(body.max_tokens) ?? asNumber(body.maxTokens) ?? 4096;
  return { maxTokens, temperature: asNumber(body.temperature), topP: asNumber(body.top_p) ?? asNumber(body.topP) };
}

function stopOf(body: Record<string, unknown>): string[] {
  const raw = body.stop ?? body.stop_sequences;
  if (typeof raw === 'string') return raw === '' ? [] : [raw];
  if (isArray(raw)) return raw.filter((s): s is string => typeof s === 'string' && s !== '');
  return [];
}

/** An OpenAI client's body, normalized. `messages` arrive already unwrapped. */
export function parseOpenAIRequest(body: Record<string, unknown>): NormalizedRequest {
  const messages: NormalizedMessage[] = [];
  let system = '';
  const stop = stopOf(body);
  const rawMessages = isArray(body.messages) ? body.messages : [];
  for (const rawMessage of rawMessages) {
    if (!isRecord(rawMessage)) continue;
    const role = asString(rawMessage.role);
    if (role === 'system') {
      for (const part of partsOf(rawMessage.content)) if (part.kind === 'text') system += part.text;
      continue;
    }
    if (role === 'tool') {
      const parts: NormalizedPart[] = [];
      for (const part of partsOf(rawMessage.content)) if (part.kind === 'text') parts.push({ kind: 'toolResult', toolCallId: asString(rawMessage.tool_call_id), content: part.text });
      if (parts.length === 0) parts.push({ kind: 'toolResult', toolCallId: asString(rawMessage.tool_call_id), content: '' });
      messages.push({ role: 'user', parts });
      continue;
    }
    if (role === 'user' || role === 'assistant') messages.push({ role, parts: openaiPartsOf(rawMessage) });
  }
  if (typeof body.system === 'string' && body.system !== '') system += body.system;
  const tools: NormalizedTool[] = [];
  if (isArray(body.tools)) {
    for (const rawTool of body.tools) {
      const tool = normalizedToolOf(rawTool);
      if (tool !== null && tool.name !== '') tools.push(tool);
    }
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  const shared = sharedNumbers(body);
  return {
    system: system === '' ? null : system,
    messages,
    tools,
    toolChoice: toolChoiceOf(body.tool_choice),
    maxTokens: shared.maxTokens,
    temperature: shared.temperature,
    topP: shared.topP,
    stopSequences: stop,
    jsonMode: format !== null && asString(format.type) === 'json_object',
    stream: body.stream === true,
  };
}

/** An Anthropic client's body, normalized. */
export function parseAnthropicRequest(body: Record<string, unknown>): NormalizedRequest {
  const messages: NormalizedMessage[] = [];
  const rawMessages = isArray(body.messages) ? body.messages : [];
  for (const rawMessage of rawMessages) {
    if (!isRecord(rawMessage)) continue;
    const role = asString(rawMessage.role);
    if (role === 'user' || role === 'assistant') messages.push({ role, parts: partsOf(rawMessage.content) });
  }
  let system: string | null = null;
  if (typeof body.system === 'string') {
    system = body.system === '' ? null : body.system;
  } else if (isArray(body.system)) {
    const text = body.system.map((p) => (isRecord(p) ? asString(p.text) : typeof p === 'string' ? p : '')).join('');
    system = text === '' ? null : text;
  }
  const tools: NormalizedTool[] = [];
  if (isArray(body.tools)) {
    for (const rawTool of body.tools) {
      const tool = normalizedToolOf(rawTool);
      if (tool !== null && tool.name !== '') tools.push(tool);
    }
  }
  const shared = sharedNumbers(body);
  return {
    system,
    messages,
    tools,
    toolChoice: toolChoiceOf(body.tool_choice),
    maxTokens: shared.maxTokens,
    temperature: shared.temperature,
    topP: shared.topP,
    stopSequences: isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string' && s !== '') : [],
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** Text parts of a normalized message, joined with a newline. */
function textOf(message: NormalizedMessage): string {
  return message.parts.filter((p): p is { kind: 'text'; text: string } => p.kind === 'text').map((p) => p.text).join('\n');
}

/**
 * The normalized form as an OpenAI request body. JSON mode is sent only to a
 * wire that can carry it; `jsonMode` is false on this path by construction.
 */
export function encodeOpenAIRequest(req: NormalizedRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (req.system !== null) messages.push({ role: 'system', content: req.system });
  for (const message of req.messages) {
    const text = textOf(message);
    if (message.role === 'assistant') {
      const toolCalls = message.parts.filter((p): p is { kind: 'toolCall'; id: string; name: string; arguments: string } => p.kind === 'toolCall');
      const body: Record<string, unknown> = { role: 'assistant', content: text === '' && toolCalls.length > 0 ? null : text };
      if (toolCalls.length > 0) body.tool_calls = toolCalls.map((call, index) => ({ id: call.id === '' ? `call_${index}` : call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }));
      messages.push(body);
      continue;
    }
    const content: unknown[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') content.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') {
        if (part.data.startsWith('data:') || part.data.startsWith('http')) content.push({ type: 'image_url', image_url: { url: part.data } });
        else if (part.mimeType !== '') content.push({ type: 'image_url', image_url: { url: `data:${part.mimeType};base64,${part.data}` } });
      } else if (part.kind === 'toolResult') content.push({ type: 'text', text: part.content });
    }
    if (content.length === 0) continue;
    const toolResults = message.parts.filter((p): p is { kind: 'toolResult'; toolCallId: string; content: string } => p.kind === 'toolResult');
    if (toolResults.length > 0) {
      for (const result of toolResults) messages.push({ role: 'tool', tool_call_id: result.toolCallId === '' ? 'call_0' : result.toolCallId, content: result.content });
      if (text !== '') messages.push({ role: 'user', content: text });
    } else {
      messages.push({ role: 'user', content: content.length === 1 && content[0] && typeof content[0] === 'object' && (content[0] as Record<string, unknown>).type === 'text' ? text : content });
    }
  }
  const body: Record<string, unknown> = { model, messages };
  if (req.tools.length > 0) body.tools = req.tools.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description ?? undefined, parameters: tool.schema } }));
  if (req.toolChoice.mode === 'any') body.tool_choice = 'auto';
  else if (req.toolChoice.mode === 'none') body.tool_choice = 'none';
  else if (req.toolChoice.mode === 'tool') body.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  if (req.maxTokens !== 4096) body.max_tokens = req.maxTokens;
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.stopSequences.length > 0) body.stop = req.stopSequences;
  if (req.jsonMode) body.response_format = { type: 'json_object' };
  return body;
}

/**
 * The normalized form as an Anthropic request body. One message's content
 * blocks keep their order: a text block, the images beside it, a tool_use
 * block, then the tool_result blocks that answer it, all one message.
 * JSON mode is never sent on this wire — an Anthropic upstream cannot carry
 * it, and its echo answers whatever was sent.
 */
export function encodeAnthropicRequest(req: NormalizedRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  for (const message of req.messages) {
    const blocks: Record<string, unknown>[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') {
        if (part.text !== '') blocks.push({ type: 'text', text: part.text });
      } else if (part.kind === 'image') {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: part.mimeType === '' ? 'image/png' : part.mimeType, data: part.data } });
      } else if (part.kind === 'toolCall') {
        let input: unknown = {};
        try {
          input = JSON.parse(part.arguments);
        } catch {
          input = {};
        }
        blocks.push({ type: 'tool_use', id: part.id === '' ? 'toolu_0' : part.id, name: part.name, input });
      } else if (part.kind === 'toolResult') {
        blocks.push({ type: 'tool_result', tool_use_id: part.toolCallId === '' ? 'toolu_0' : part.toolCallId, content: part.content });
      }
    }
    if (blocks.length === 0) continue;
    messages.push({ role: message.role, content: blocks });
  }
  const body: Record<string, unknown> = { model, messages, max_tokens: req.maxTokens };
  if (req.system !== null) body.system = req.system;
  if (req.tools.length > 0) body.tools = req.tools.map((tool) => ({ name: tool.name, description: tool.description ?? undefined, input_schema: tool.schema }));
  if (req.toolChoice.mode === 'any') body.tool_choice = { type: 'auto' };
  else if (req.toolChoice.mode === 'none') body.tool_choice = { type: 'none' };
  else if (req.toolChoice.mode === 'tool') body.tool_choice = { type: 'tool', name: req.toolChoice.name };
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.stopSequences.length > 0) body.stop_sequences = req.stopSequences;
  return body;
}

/** One Gemini content part: a text part, an inline image, or a function call. */
function geminiPartOf(part: NormalizedPart): Record<string, unknown> | null {
  if (part.kind === 'text') return part.text === '' ? null : { text: part.text };
  if (part.kind === 'image') {
    return { inlineData: { mimeType: part.mimeType === '' ? 'image/png' : part.mimeType, data: part.data } };
  }
  if (part.kind === 'toolCall') {
    let args: Record<string, unknown> = {};
    try {
      const parsed: unknown = JSON.parse(part.arguments);
      if (isRecord(parsed)) args = parsed;
    } catch {
      args = {};
    }
    return { functionCall: { name: part.name, args } };
  }
  return null;
}

/** The normalized form as a Gemini generateContent body. */
export function encodeGeminiRequest(req: NormalizedRequest, model: string): Record<string, unknown> {
  const contents: Record<string, unknown>[] = [];
  if (req.system !== null) contents.push({ role: 'user', parts: [{ text: req.system }] });
  for (const message of req.messages) {
    const parts: Record<string, unknown>[] = [];
    for (const part of message.parts) {
      if (part.kind === 'toolResult') {
        parts.push({ functionResponse: { name: 'tool', response: { output: part.content } } });
        continue;
      }
      const gemini = geminiPartOf(part);
      if (gemini !== null) parts.push(gemini);
    }
    if (parts.length === 0) continue;
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const config: Record<string, unknown> = { maxOutputTokens: req.maxTokens };
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stopSequences.length > 0) config.stopSequences = req.stopSequences;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (req.tools.length > 0) {
    config.tools = [
      {
        functionDeclarations: req.tools.map((tool) => ({
          name: tool.name,
          description: tool.description ?? '',
          parameters: tool.schema,
        })),
      },
    ];
  }
  return { model, contents, generationConfig: config };
}

/** The stop reason in the one family this aggregator reports: "stop" or "length". */
function stopFamily(reason: string): StopReason {
  if (reason === 'length' || reason === 'max_tokens') return 'length';
  if (reason === 'tool_calls' || reason === 'tool_use') return 'tool_calls';
  if (reason === 'content_filter') return 'content_filter';
  return 'stop';
}

/** The text an OpenAI choice carries: its message content, or its delta. */
function openaiChoiceText(choice: Record<string, unknown>): { text: string; calls: ToolCall[] } {
  const message = isRecord(choice.message) ? choice.message : isRecord(choice.delta) ? choice.delta : null;
  if (message === null) return { text: '', calls: [] };
  const text = typeof message.content === 'string' ? message.content : '';
  const calls: ToolCall[] = [];
  if (isArray(message.tool_calls)) {
    for (const rawCall of message.tool_calls) {
      if (!isRecord(rawCall)) continue;
      const fn = isRecord(rawCall.function) ? rawCall.function : {};
      calls.push(toolCallOf(asString(rawCall.id), asString(fn.name), fn.arguments));
    }
  }
  return { text, calls };
}

/** One OpenAI response or chunk, normalized. */
export function parseOpenAIAnswer(body: unknown): NormalizedAnswer {
  if (!isRecord(body)) return { text: '', usage: null, stopReason: 'stop', toolCalls: [] };
  const choices = isArray(body.choices) ? body.choices : [];
  let text = '';
  let stopReason: StopReason = 'stop';
  const toolCalls: ToolCall[] = [];
  for (const choice of choices) {
    if (!isRecord(choice)) continue;
    const parsed = openaiChoiceText(choice);
    text += parsed.text;
    toolCalls.push(...parsed.calls);
    stopReason = stopFamily(asString(choice.finish_reason));
  }
  return { text, usage: usageOf(body.usage), stopReason, toolCalls };
}

/** An Anthropic response or content-block event, normalized. */
export function parseAnthropicAnswer(body: unknown): NormalizedAnswer {
  if (!isRecord(body)) return { text: '', usage: null, stopReason: 'stop', toolCalls: [] };
  let text = '';
  const toolCalls: ToolCall[] = [];
  const content = isArray(body.content) ? body.content : [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text') text += asString(block.text);
    else if (block.type === 'tool_use') toolCalls.push(toolCallOf(asString(block.id), asString(block.name), block.input));
  }
  const stopReason = body.stop_reason === 'tool_use' ? 'tool_calls' : stopFamily(asString(body.stop_reason));
  return { text, usage: usageOf(body.usage), stopReason, toolCalls };
}

/** A Gemini generateContent response, normalized. */
export function parseGeminiAnswer(body: unknown): NormalizedAnswer {
  if (!isRecord(body)) return { text: '', usage: null, stopReason: 'stop', toolCalls: [] };
  const candidates = isArray(body.candidates) ? body.candidates : [];
  let text = '';
  const toolCalls: ToolCall[] = [];
  let stopReason: StopReason = 'stop';
  for (const candidate of candidates) {
    if (!isRecord(candidate)) continue;
    const content = isRecord(candidate.content) ? candidate.content : null;
    const parts = content !== null && isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') text += part.text;
      else if (isRecord(part.functionCall)) toolCalls.push(toolCallOf('', asString(part.functionCall.name), part.functionCall.args));
    }
    const reason = asString(candidate.finishReason);
    if (reason === 'MAX_TOKENS') stopReason = 'length';
    else if (reason === 'SAFETY') stopReason = 'content_filter';
    else stopReason = 'stop';
  }
  return { text, usage: usageOf(body.usageMetadata), stopReason, toolCalls };
}

/** The text an Anthropic stream event contributes: its delta's text. */
function anthropicDeltaText(event: Record<string, unknown>): string {
  if (!isRecord(event.delta)) return '';
  return event.delta.type === 'text_delta' ? asString(event.delta.text) : '';
}

/** The tool use one Anthropic content-block start carries. */
function anthropicBlockCall(event: Record<string, unknown>): ToolCall | null {
  if (!isRecord(event.content_block)) return null;
  const block = event.content_block;
  if (block.type !== 'tool_use') return null;
  return toolCallOf(asString(block.id), asString(block.name), block.input);
}

/** An OpenAI event stream, re-assembled into one answer. */
export function parseOpenAIStream(text: string): NormalizedAnswer {
  let answer: NormalizedAnswer = { text: '', usage: null, stopReason: 'stop', toolCalls: [] };
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload);
    } catch {
      continue;
    }
    const part = parseOpenAIAnswer(parsed);
    answer = { text: answer.text + part.text, usage: addUsage(answer.usage, part.usage), stopReason: part.stopReason, toolCalls: [...answer.toolCalls, ...part.toolCalls] };
  }
  return answer;
}

/** An Anthropic event stream, re-assembled into one answer. */
export function parseAnthropicStream(text: string): NormalizedAnswer {
  let answer: NormalizedAnswer = { text: '', usage: null, stopReason: 'stop', toolCalls: [] };
  let currentEvent = '';
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('event:')) {
      currentEvent = trimmed.slice(6).trim();
      continue;
    }
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '') continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(parsed)) continue;
    if (currentEvent === 'message_start' || parsed.type === 'message_start') {
      if (isRecord(parsed.message)) {
        const start = parseAnthropicAnswer(parsed.message);
        answer = { ...answer, usage: addUsage(answer.usage, start.usage) };
      }
      continue;
    }
    if (currentEvent === 'content_block_delta' || parsed.type === 'content_block_delta') {
      answer = { ...answer, text: answer.text + anthropicDeltaText(parsed) };
      continue;
    }
    if (currentEvent === 'content_block_start' || parsed.type === 'content_block_start') {
      const call = anthropicBlockCall(parsed);
      if (call !== null) answer = { ...answer, toolCalls: [...answer.toolCalls, call] };
      continue;
    }
    if (currentEvent === 'message_delta' || parsed.type === 'message_delta') {
      const delta = parseAnthropicAnswer(parsed);
      // A message_delta carries its stop reason inside `delta`, not at the top.
      const deltaStop = isRecord(parsed.delta) ? asString(parsed.delta.stop_reason) : '';
      answer = { ...answer, usage: addUsage(answer.usage, delta.usage), stopReason: deltaStop !== '' ? stopFamily(deltaStop) : delta.stopReason };
      continue;
    }
  }
  return answer;
}

/** A Gemini stream: one JSON object per line, re-assembled into one answer. */
export function parseGeminiStream(text: string): NormalizedAnswer {
  let answer: NormalizedAnswer = { text: '', usage: null, stopReason: 'stop', toolCalls: [] };
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:') && !trimmed.startsWith('{')) continue;
    const payload = trimmed.startsWith('data:') ? trimmed.slice(5).trim() : trimmed;
    if (payload === '') continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload);
    } catch {
      continue;
    }
    const part = parseGeminiAnswer(parsed);
    answer = { text: answer.text + part.text, usage: addUsage(answer.usage, part.usage), stopReason: part.stopReason, toolCalls: [...answer.toolCalls, ...part.toolCalls] };
  }
  return answer;
}

/** An OpenAI response body: choices, then usage, then the fixed fields. */
export function emitOpenAIJson(answer: NormalizedAnswer, model: string, id: string): Record<string, unknown> {
  const choice: Record<string, unknown> = {
    index: 0,
    message: {
      role: 'assistant',
      content: answer.text,
      ...(answer.toolCalls.length > 0
        ? {
            tool_calls: answer.toolCalls.map((call, index) => ({ id: call.id === '' ? `call_${index}` : call.id, type: 'function', function: { name: call.name, arguments: call.arguments } })),
          }
        : {}),
    },
    finish_reason: answer.toolCalls.length > 0 ? 'tool_calls' : 'stop',
  };
  return {
    id,
    object: 'chat.completion',
    created: 0,
    model,
    choices: [choice],
    usage: { prompt_tokens: answer.usage?.prompt ?? 0, completion_tokens: answer.usage?.completion ?? 0, total_tokens: (answer.usage?.prompt ?? 0) + (answer.usage?.completion ?? 0) },
  };
}

/** An OpenAI event stream: role, content, finish-with-usage, then [DONE]. */
export function emitOpenAIStream(answer: NormalizedAnswer, model: string, id: string): string {
  const base = { id, object: 'chat.completion.chunk', created: 0, model };
  const lines: string[] = [];
  lines.push(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })}`);
  if (answer.text !== '') lines.push(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: { content: answer.text }, finish_reason: null }] })}`);
  lines.push(
    `data: ${JSON.stringify({
      ...base,
      choices: [{ index: 0, delta: {}, finish_reason: answer.toolCalls.length > 0 ? 'tool_calls' : 'stop' }],
      usage: { prompt_tokens: answer.usage?.prompt ?? 0, completion_tokens: answer.usage?.completion ?? 0, total_tokens: (answer.usage?.prompt ?? 0) + (answer.usage?.completion ?? 0) },
    })}`,
  );
  lines.push('data: [DONE]');
  return `${lines.join('\n')}\n`;
}

/** An Anthropic response body: content blocks, then usage, then the stop reason. */
export function emitAnthropicJson(answer: NormalizedAnswer, model: string, id: string): Record<string, unknown> {
  const content: Record<string, unknown>[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    content.push({ type: 'tool_use', id: call.id === '' ? 'toolu_0' : call.id, name: call.name, input });
  }
  return {
    id,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: answer.toolCalls.length > 0 ? 'tool_use' : 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.usage?.prompt ?? 0, output_tokens: answer.usage?.completion ?? 0 },
  };
}

/** An Anthropic event stream: message_start, blocks, deltas, message_delta, message_stop. */
export function emitAnthropicStream(answer: NormalizedAnswer, model: string, id: string): string {
  const lines: string[] = [];
  lines.push(
    `event: message_start\ndata: ${JSON.stringify({
      type: 'message_start',
      message: { id, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage?.prompt ?? 0, output_tokens: 0 } },
    })}`,
  );
  if (answer.text !== '') {
    lines.push(`event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })}`);
    lines.push(`event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } })}`);
    lines.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}`);
  }
  for (let i = 0; i < answer.toolCalls.length; i += 1) {
    const call = answer.toolCalls[i];
    const index = answer.text !== '' ? i + 1 : i;
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    lines.push(`event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id === '' ? 'toolu_0' : call.id, name: call.name, input } })}`);
    lines.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index })}`);
  }
  lines.push(
    `event: message_delta\ndata: ${JSON.stringify({
      type: 'message_delta',
      delta: { stop_reason: answer.toolCalls.length > 0 ? 'tool_use' : 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.usage?.completion ?? 0 },
    })}`,
  );
  lines.push('event: message_stop\ndata: {"type":"message_stop"}');
  return `${lines.join('\n')}\n`;
}
