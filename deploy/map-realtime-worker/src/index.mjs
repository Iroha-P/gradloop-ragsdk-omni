import {
  DEFAULT_LIMITS,
  createRateLimiter,
  evaluateHandshake,
  parseAllowedOrigins,
} from "./policy.mjs";

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

function config(env) {
  return {
    allowedOrigins: parseAllowedOrigins(env.ALLOWED_ORIGINS),
    upstream: String(env.MAP_REALTIME_URL || "").trim(),
    authHeader: String(env.MAP_AUTH_HEADER || "Authorization").trim() || "Authorization",
    authPrefix: String(env.MAP_AUTH_PREFIX || "Bearer").trim(),
    sessionSeconds: Math.min(
      Math.max(Number(env.MAP_SESSION_SECONDS || DEFAULT_LIMITS.sessionSeconds), 30),
      600,
    ),
  };
}

function isWebSocketUpgrade(request) {
  return request.headers.get("Upgrade")?.toLowerCase() === "websocket";
}

export function toHttpUpgradeUrl(value) {
  const url = new URL(value);
  if (url.protocol === "wss:") url.protocol = "https:";
  if (url.protocol === "ws:") url.protocol = "http:";
  if (!new Set(["https:", "http:"]).has(url.protocol)) {
    throw new Error("upstream must use ws, wss, http, or https");
  }
  return url;
}

async function websocketProxy(request, env) {
  const settings = config(env);
  const url = new URL(request.url);
  const mode = url.searchParams.get("mode");
  const result = evaluateHandshake({
    origin: request.headers.get("Origin"),
    allowedOrigins: settings.allowedOrigins,
    mode,
    contentLength: request.headers.get("Content-Length") || 0,
  });
  if (!result.ok) return json({ error: result.code }, result.status);
  if (!settings.upstream || !env.MAP_API_KEY) return json({ error: "proxy_not_configured" }, 503);
  if (!isWebSocketUpgrade(request)) return json({ error: "websocket_upgrade_required" }, 426);

  // Cloudflare's WebSocket client is a fetch() with Upgrade: websocket.
  // Normalize the config URL to the HTTP form required by that fetch.
  const upstreamUrl = toHttpUpgradeUrl(settings.upstream);
  upstreamUrl.searchParams.set("mode", result.mode);
  const headers = new Headers({ Upgrade: "websocket" });
  headers.set(settings.authHeader, settings.authPrefix ? `${settings.authPrefix} ${env.MAP_API_KEY}` : env.MAP_API_KEY);
  const upstreamResponse = await fetch(upstreamUrl, { headers });
  if (upstreamResponse.status !== 101 || !upstreamResponse.webSocket) {
    return json({ error: "upstream_unavailable" }, 502);
  }

  const pair = new WebSocketPair();
  const client = pair[0];
  const server = pair[1];
  server.accept({ allowHalfOpen: true });
  const upstream = upstreamResponse.webSocket;
  upstream.accept();
  const limiter = createRateLimiter({ limit: DEFAULT_LIMITS.messagesPerSecond });
  const closeBoth = (code = 1000) => {
    try { server.close(code, "proxy_closed"); } catch (_error) { /* already closed */ }
    try { upstream.close(code, "proxy_closed"); } catch (_error) { /* already closed */ }
  };
  server.addEventListener("message", (event) => {
    if (!limiter.allow()) return closeBoth(1008);
    if (typeof event.data === "string" && new TextEncoder().encode(event.data).byteLength > DEFAULT_LIMITS.maxMessageBytes) {
      return closeBoth(1009);
    }
    upstream.send(event.data);
  });
  upstream.addEventListener("message", (event) => server.send(event.data));
  server.addEventListener("close", () => closeBoth());
  upstream.addEventListener("close", () => closeBoth());
  return new Response(null, { status: 101, webSocket: client });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health" && request.method === "GET") {
      return json({ status: "ready", upstream: env.MAP_REALTIME_URL ? "configured" : "missing" });
    }
    if (url.pathname === "/ws" && request.method === "GET") return websocketProxy(request, env);
    return json({ error: "not_found" }, 404);
  },
};
