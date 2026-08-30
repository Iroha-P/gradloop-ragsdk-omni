import test from "node:test";
import assert from "node:assert/strict";
import { toHttpUpgradeUrl } from "../src/index.mjs";

test("normalizes a wss upstream for a Cloudflare upgrade fetch", () => {
  assert.equal(
    toHttpUpgradeUrl("wss://minicpmo45.modelbest.cn/v1/realtime").toString(),
    "https://minicpmo45.modelbest.cn/v1/realtime",
  );
});

test("keeps an https upstream unchanged", () => {
  assert.equal(
    toHttpUpgradeUrl("https://provider.example/realtime").toString(),
    "https://provider.example/realtime",
  );
});
