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
    if (!nodes.has(id)) nodes.set(id, { value: "", checked: false, disabled: false, files: [], attributes: {}, setAttribute(name, value) { this.attributes[name] = value; } });
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
  class SyntheticDataTransfer {
    constructor() { this.files = []; this.items = { add: (file) => this.files.push(file) }; }
  }
  const context = vm.createContext({ window, byId, DataTransfer: SyntheticDataTransfer, summarizeOmniFiles() {}, setText: (id, value) => { byId(id).textContent = value; }, AbortController, URL, setTimeout: () => 1, clearTimeout() {} });
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

test("the material chooser accepts documents and automatically routes them to local parsing", () => {
  const media = html.slice(html.indexOf('id="mediaWorkspace"'), html.indexOf('id="documentWorkspace"'));
  assert.match(media, /accept="[^\"]*\.pdf[^\"]*\.docx/);
  assert.match(media, /onchange="routeMaterialSelection\(\)"/);
  for (const [name, type] of [["synthetic.pdf", "application/pdf"], ["synthetic.doc", "application/msword"], ["synthetic.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]]) {
    const h = harness();
    h.byId("omniFiles").files = [{ name, type, size: 12 }];
    h.context.routeMaterialSelection();
    assert.equal(h.byId("mediaWorkspace").hidden, true);
    assert.equal(h.byId("documentWorkspace").hidden, false);
    assert.equal(h.byId("documentFiles").files.length, 1);
    assert.equal(h.byId("documentModeButton").attributes["aria-pressed"], "true");
    assert.match(h.byId("documentStatus").textContent, /解析文档/);
    assert.equal(h.sessionCount(), 0);
  }
});

test("mixed document and media selection is rejected without sending files", () => {
  const h = harness();
  h.byId("omniFiles").files = [{ name: "synthetic.pdf", type: "application/pdf" }, { name: "synthetic.png", type: "image/png" }];
  h.context.routeMaterialSelection();
  assert.match(h.byId("omniOut").textContent, /分开选择/);
  assert.equal(h.byId("omniFiles").value, "");
  assert.equal(h.sessionCount(), 0);
});

test("local parsing has an independent module entrypoint and offline API is clearly disabled", async () => {
  const moduleBlocks = [...html.matchAll(/<script type="module">([\s\S]*?)<\/script>/g)].map((match) => match[1]);
  const documentBlock = moduleBlocks.find((block) => block.includes('from "./documents/client.mjs"'));
  assert.ok(documentBlock);
  assert.ok(!documentBlock.includes('./realtime/session.mjs'));
  assert.ok(!documentBlock.includes('./public-demo-fixtures.mjs'));
  const h = harness();
  h.byId("documentPreview").value = "Synthetic local evidence";
  h.byId("documentConsent").checked = true;
  vm.runInContext('window.__gradloopRealtimeConfig = { enabled: false }; updateDocumentSendState();', h.context);
  assert.equal(h.byId("documentSendButton").disabled, true);
  assert.match(h.byId("documentConnectionStatus").textContent, /没有接入模型 API/);
  await h.context.parseDocuments();
  assert.match(h.byId("documentPreview").value, /Synthetic/);
  assert.equal(h.sessionCount(), 0);
});

test("the training overview does not claim every local input invokes a model", () => {
  assert.ok(!html.includes("每次输入都经过模型理解"));
  assert.match(html, /本地解析不需要模型 API/);
  assert.match(html, /模型分析按运行模式启用/);
});
