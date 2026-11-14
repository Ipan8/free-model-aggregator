// The Gemini wire: client requests (OpenAI or Anthropic) rewritten to
// generateContent / streamGenerateContent, and Gemini answers read back.
//
// The env's Gemini provider parses exactly these fields (see the kit's
// program/env/server.mjs, geminiParsed): systemInstruction parts text, the
// contents' text and inlineData/fileData images, the functionDeclarations'
// names, and generationConfig's maxOutputTokens, stopSequences and
// responseMimeType. Everything else is carried for fidelity with the real
// service, which speaks the same shape. Nothing here reads the clock or
// randomness: a request and a provider answer always give the same bytes.

import { isRecord, type Json } from './providers.ts';

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One OpenAI message's image part as a Gemini inlineData part, or null. */
function inlineDataOf(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const match = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (match !== null) {
    return { inlineData: { mimeType: match[1] !== undefined && match[1] !== '' ? match[1] : 'image/png', data: match[2] } };
  }
  return { fileData: { mimeType: 'image/png', fileUri: url } };
}

/** One Anthropic image block as a Gemini inlineData part, or null. */
function inlineDataOfAnthropic(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  return null;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request. */
export function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemText = '';
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        parts.push({ functionCall: { name: call.function.name, args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }],
      });
      continue;
    }
    if (role !== 'user') continue;
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else {
          const image = inlineDataOf(block);
          if (image !== null) parts.push(image);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: '' }] });

  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s) => typeof s === 'string');
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use' && typeof block.name === 'string') {
            parts.push({ functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image') {
        const image = inlineDataOfAnthropic(block);
        if (image !== null) parts.push(image);
      } else if (block.type === 'tool_result') {
        const text = typeof block.content === 'string' ? block.content : textOf(block.content);
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: text } } });
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: '' }] });

  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences.filter((s) => typeof s === 'string');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini candidate's parts. */
function candidateText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  return content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const text = candidateText(candidate);
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const output = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const text = candidateText(candidate);
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const output = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: input, output_tokens: output },
  };
}
