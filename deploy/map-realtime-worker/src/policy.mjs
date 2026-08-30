export const SUPPORTED_MODES = new Set(["chat", "audio", "video"]);
export const DEFAULT_LIMITS = Object.freeze({
  maxMessageBytes: 1024 * 1024,
  messagesPerSecond: 4,
  sessionSeconds: 300,
});

export function parseMode(value) {
  const mode = String(value || "").trim().toLowerCase();
  if (!SUPPORTED_MODES.has(mode)) {
    throw new Error(`unsupported mode: ${mode || "empty"}`);
  }
  return mode;
}

export function parseAllowedOrigins(value) {
  return String(value || "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);
}

export function evaluateHandshake({
  origin,
  allowedOrigins,
  mode,
  contentLength = 0,
  limits = DEFAULT_LIMITS,
}) {
  if (!origin || !allowedOrigins.includes(origin)) {
    return { ok: false, status: 403, code: "origin_not_allowed" };
  }
  let parsedMode;
  try {
    parsedMode = parseMode(mode);
  } catch (_error) {
    return { ok: false, status: 400, code: "unsupported_mode" };
  }
  if (Number(contentLength) > limits.maxMessageBytes) {
    return { ok: false, status: 413, code: "message_too_large" };
  }
  return { ok: true, mode: parsedMode };
}

export function createRateLimiter({ limit = DEFAULT_LIMITS.messagesPerSecond, now = () => Date.now() } = {}) {
  let windowStart = now();
  let count = 0;
  return {
    allow() {
      const current = now();
      if (current - windowStart >= 1000) {
        windowStart = current;
        count = 0;
      }
      if (count >= limit) return false;
      count += 1;
      return true;
    },
  };
}
