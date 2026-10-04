import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { buildDocumentPrompt } from "../../app/ui/documents/policy.mjs";

const html = readFileSync(new URL("../../app/ui/index.html", import.meta.url), "utf8");
const flow = html.slice(html.indexOf("    function selectInputMode("), html.indexOf("    async function analyzeOmni("));

// Synthetic DOM/session doubles only: this is not a real browser acceptance test.
function harness(parser = async () => ({ text: "Synthetic public evidence", metadata: [{ kind: "pdf", pages: 1, characters: 25 }] })) {
  const nodes = new Map();
  const byId = (id) => {
    if (!nodes.has(id)) nodes.set(id, { value: "", checked: false, disabled: false, files: [], setAttribute() {} });
    return nodes.get(id);
  };
  let sessions = 0;
  const inputs = [];
  const window = {
    GradLoopDocuments: { parseDocuments: parser, buildDocumentPrompt },
    __gradloopRealtimeConfig: { enabled: true, proxy_url: "wss://worker.example/ws" },
    GradLoopRealtime: {
      buildChatInput: (value) => value,
      RealtimeSession() {
        sessions++;
        let listener;
        return { subscribe(value) { listener = value; }, connect() { listener({ state: "ready" }); }, sendInput(value) { inputs.push(value); }, stop() { listener({ state: "stopped" }); } };
      },
    },
  };
  const context = vm.createContext({ window, byId, setText: (id, value) => { byId(id).textContent = value; }, AbortController, URL, setTimeout: () => 1, clearTimeout() {} });
  vm.runInContext("let documentParseController = null, documentSession = null, documentTimer = null;\n" + flow, context);
  byId("documentTask").value = "Explain the evidence";
  return { context, byId, inputs, sessionCount: () => sessions };
}

test("document mode provides its own visible, editable analysis task", () => {
  const workspace = html.slice(html.indexOf('id="documentWorkspace"'), html.indexOf('class="realtime-panel"'));
  assert.match(workspace, /id="documentTask"/);
  assert.match(flow, /task: byId\("documentTask"\)/);
});

test("local preview never starts a model session; only explicitly confirmed sending does", async () => {
  const h = harness();
  await h.context.parseDocuments();
  assert.equal(h.sessionCount(), 0);
  assert.match(h.byId("documentPreview").value, /Synthetic/);
  await h.context.sendDocumentText();
  assert.equal(h.sessionCount(), 0);
  h.byId("documentConsent").checked = true;
  await h.context.sendDocumentText();
  assert.equal(h.sessionCount(), 1);
  assert.equal(h.inputs.length, 1);
  assert.match(h.inputs[0].text, /Explain the evidence/);
  h.context.documentTextChanged();
  assert.equal(h.byId("documentConsent").checked, false);
  assert.equal(h.byId("documentSendButton").disabled, true);
});

test("changing files cancels parsing and ignores its late result", async () => {
  let complete;
  const h = harness(() => new Promise((resolve) => { complete = resolve; }));
  const pending = h.context.parseDocuments();
  assert.equal(h.byId("documentParseButton").disabled, true);
  h.context.documentSelectionChanged();
  complete({ text: "stale text", metadata: [] });
  await pending;
  assert.equal(h.byId("documentPreview").value, "");
  assert.equal(h.byId("documentParseButton").disabled, false);
  assert.equal(h.sessionCount(), 0);
});
