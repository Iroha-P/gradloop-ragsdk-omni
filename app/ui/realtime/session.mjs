import { buildSessionInit, parseServerEvent } from "./protocol.mjs";

const STATES = new Set(["idle", "connecting", "queued", "ready", "listening", "responding", "stopped", "failed"]);

export function RealtimeSession({ socketFactory, clock = globalThis, audioSink = { stop() {} }, sessionSeconds = 300 } = {}) {
  if (typeof socketFactory !== "function") throw new Error("socketFactory is required");
  let socket = null;
  let timer = null;
  let state = "idle";
  let initSent = false;
  let sessionOptions = {};
  const subscribers = new Set();

  const emit = (event = {}) => {
    for (const subscriber of subscribers) subscriber({ state, ...event });
  };
  const transition = (next, event = {}) => {
    if (!STATES.has(next)) throw new Error(`invalid state: ${next}`);
    state = next;
    emit(event);
  };
  const clearTimer = () => { if (timer != null && typeof clock.clearTimeout === "function") clock.clearTimeout(timer); timer = null; };
  const stopResources = () => { clearTimer(); audioSink.stop?.(); };
  const closeSocket = () => { try { socket?.close?.(); } catch (_error) { /* already closed */ } socket = null; };

  const handleMessage = (raw) => {
    try {
      const event = parseServerEvent(raw);
      if (event.type === "queue") transition("queued", event);
      else if (event.type === "queue_done") {
        if (state !== "queued") transition("queued", event);
        if (!initSent) {
          initSent = true;
          socket?.send(JSON.stringify(buildSessionInit(sessionOptions)));
        }
      }
      else if (event.type === "ready") transition("ready", event);
      else if (event.type === "listen") transition("listening", event);
      else if (event.type === "delta") {
        if (event.kind === "audio") audioSink.push?.(event.audio);
        transition("responding", event);
      } else if (event.type === "done") transition("ready", event);
      else if (event.type === "error") transition("failed", event);
      else transition("stopped", event);
    } catch (error) {
      transition("failed", { error: error.message });
    }
  };

  return {
    get state() { return state; },
    subscribe(listener) { subscribers.add(listener); return () => subscribers.delete(listener); },
    connect(url, options = {}) {
      if (state !== "idle" && state !== "stopped" && state !== "failed") throw new Error("session already active");
      transition("connecting");
      initSent = false;
      sessionOptions = options;
      socket = socketFactory(url);
      socket.addEventListener?.("open", () => {
        transition("queued");
        if (typeof clock.setTimeout === "function") timer = clock.setTimeout(() => this.stop("expired"), Math.max(30, sessionSeconds) * 1000);
      });
      socket.addEventListener?.("message", (event) => handleMessage(event.data));
      socket.addEventListener?.("error", () => transition("failed", { error: "socket error" }));
      socket.addEventListener?.("close", () => { if (state !== "stopped") transition("failed", { error: "socket closed" }); });
    },
    sendInput(message) {
      if (!socket || !["queued", "ready", "listening", "responding"].includes(state)) throw new Error("session is not connected");
      socket.send(JSON.stringify(message));
      if (state === "ready") transition("listening");
    },
    stop(reason = "user") {
      if (state === "stopped") return;
      if (socket && typeof socket.send === "function") {
        try { socket.send(JSON.stringify({ type: "session.close", reason })); } catch (_error) { /* closing */ }
      }
      stopResources();
      transition("stopped", { reason });
      closeSocket();
    },
  };
}
