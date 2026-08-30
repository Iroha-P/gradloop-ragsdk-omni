const MAX_TEXT_CHARS = 20000;
const MAX_FRAME_CHARS = 1024 * 1024;
const EVENT_TYPES = new Set([
  "session.queued",
  "session.queue_update",
  "session.queue_done",
  "session.created",
  "response.output.delta",
  "response.done",
  "session.closed",
  "error",
]);

function objectValue(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name} must be an object`);
  return value;
}

export function parseServerEvent(raw) {
  let value;
  try { value = JSON.parse(raw); } catch (_error) { throw new Error("invalid server event"); }
  objectValue(value, "event");
  if (!EVENT_TYPES.has(value.type)) throw new Error(`unsupported event: ${value.type || "empty"}`);
  if (value.type === "response.output.delta") {
    if (value.kind === "listen") return { type: "listen" };
    if (value.kind !== "text" && value.kind !== "audio") throw new Error("unsupported delta kind");
    if (typeof value[value.kind] !== "string") throw new Error("delta payload must be a string");
    const payload = value[value.kind];
    if (value.kind === "text" && payload.length > MAX_TEXT_CHARS) throw new Error("text delta too large");
    if (value.kind === "audio" && payload.length > MAX_FRAME_CHARS) throw new Error("audio delta too large");
    return { type: "delta", kind: value.kind, [value.kind]: payload };
  }
  if (value.type === "error") {
    const detail = value.error && typeof value.error === "object" ? value.error : value;
    return { type: "error", code: String(detail.code || "provider_error"), message: String(detail.message || "provider error") };
  }
  if (value.type === "session.queue_update") return { type: "queue", position: Number.isFinite(value.position) ? value.position : null };
  if (value.type === "session.queued") return { type: "queue", position: null };
  if (value.type === "session.queue_done") return { type: "queue_done" };
  if (value.type === "session.created") return { type: "ready" };
  if (value.type === "response.done") return { type: "done" };
  return { type: "closed" };
}

export function buildSessionInit({ instructions = "", lengthPenalty = 0 } = {}) {
  if (String(instructions).length > MAX_TEXT_CHARS) throw new Error("instructions too long");
  const payload = {};
  if (String(instructions)) payload.system_prompt = String(instructions);
  if (Number(lengthPenalty)) payload.config = { length_penalty: Number(lengthPenalty) };
  return {
    type: "session.init",
    payload,
  };
}

export function buildChatInput({ text, maxNewTokens = 64, lengthPenalty = 1.1, tts = false } = {}) {
  if (typeof text !== "string" || !text || text.length > MAX_TEXT_CHARS) throw new Error("chat text is invalid");
  return {
    type: "input.append",
    input: {
      messages: [{ role: "user", content: text }],
      streaming: true,
      generation: { max_new_tokens: Math.max(1, Math.floor(Number(maxNewTokens) || 64)), length_penalty: Number(lengthPenalty) || 1.1 },
      tts: { enabled: Boolean(tts) },
    },
  };
}

export function buildDuplexInput({ audioBase64, videoFrames = [], forceListen = false, maxSliceNums = 1 } = {}) {
  if (!Array.isArray(videoFrames)) throw new Error("video frames must be an array");
  const input = { type: "input.append", input: {} };
  if (audioBase64 != null) {
    if (typeof audioBase64 !== "string" || audioBase64.length > MAX_FRAME_CHARS) throw new Error("audio frame too large");
    input.input.audio = audioBase64;
  }
  if (videoFrames.length) {
    if (!Array.isArray(videoFrames) || videoFrames.some((frame) => typeof frame !== "string" || frame.length > MAX_FRAME_CHARS)) throw new Error("frame too large");
    input.input.video_frames = videoFrames;
  }
  if (forceListen) input.input.force_listen = true;
  if (videoFrames.length) input.input.max_slice_nums = Math.max(1, Math.floor(Number(maxSliceNums) || 1));
  return input;
}

export { MAX_FRAME_CHARS, MAX_TEXT_CHARS };
