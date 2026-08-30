import test from "node:test";
import assert from "node:assert/strict";
import { createRateLimiter, evaluateHandshake, parseMode } from "../src/policy.mjs";

test("rejects an origin outside the allowlist", () => {
  const result = evaluateHandshake({ origin: "https://evil.example", allowedOrigins: ["https://demo.example"], mode: "chat" });
  assert.equal(result.ok, false);
  assert.equal(result.status, 403);
});

test("accepts only the three documented modes", () => {
  assert.equal(parseMode("audio"), "audio");
  assert.equal(parseMode("video"), "video");
  assert.throws(() => parseMode("admin"), /unsupported mode/);
});

test("rejects oversized messages and limits burst rate", () => {
  const oversized = evaluateHandshake({
    origin: "https://demo.example",
    allowedOrigins: ["https://demo.example"],
    mode: "chat",
    contentLength: 1024 * 1024 + 1,
  });
  assert.equal(oversized.code, "message_too_large");
  let clock = 0;
  const limiter = createRateLimiter({ now: () => clock, limit: 2 });
  assert.equal(limiter.allow(), true);
  assert.equal(limiter.allow(), true);
  assert.equal(limiter.allow(), false);
  clock = 1000;
  assert.equal(limiter.allow(), true);
});
