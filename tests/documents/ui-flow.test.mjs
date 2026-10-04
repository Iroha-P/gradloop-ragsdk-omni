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
  const context = vm.createContext({ window, byId, isPublicPages: () => true, setDot() {}, DataTransfer: SyntheticDataTransfer, summarizeOmniFiles() {}, setText: (id, value) => { byId(id).textContent = value; }, AbortController, URL, setTimeout: () => 1, clearTimeout() {} });
  vm.runInContext("let documentParseController = null, documentSession = null, documentTimer = null, realtimeSession = null;\n" + flow, context);
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

test("public media analysis refuses the absent local backend before reading or uploading files", async () => {
  const h = harness();
  let calls = 0;
  h.context.api = () => { calls++; throw new Error("must not request a local API"); };
  h.context.FormData = class { constructor() { throw new Error("must not construct a file upload"); } };
  vm.runInContext(html.slice(html.indexOf("    async function analyzeOmni("), html.indexOf("    async function loadRealtimeConfig(")), h.context);
  h.byId("omniFiles").files = [{ name: "synthetic.png", type: "image/png", arrayBuffer() { throw new Error("must not read bytes"); } }];
  h.byId("omniPrompt").value = "Describe this synthetic image";
  await h.context.analyzeOmni();
  assert.equal(calls, 0);
  assert.match(h.byId("omniOut").textContent, /媒体文件分析需要本地后端/);
});

test("public mode rejects local API requests without making a network call", async () => {
  const h = harness();
  let calls = 0;
  h.context.fetch = () => { calls++; throw new Error("must not fetch"); };
  h.context.FormData = class {};
  vm.runInContext(html.slice(html.indexOf("    async function api("), html.indexOf("    function localDate(")), h.context);
  await assert.rejects(h.context.api("/v1/omni/analyze", { method: "POST" }), /公开静态页面不提供/);
  assert.equal(calls, 0);
});

test("public startup does not advertise verified live inference", () => {
  const startup = html.slice(html.indexOf("    function applyPublicPagesMode("), html.indexOf("    function selectTab("));
  assert.ok(!startup.includes('setDot("modelDot", true)'));
  assert.ok(!startup.includes('"LIVE MAP · MiniCPM-o"'));
  assert.match(startup, /selectInputMode\("document"\)/);
  assert.match(html, /id="mapChatConsent"/);
  assert.ok(!html.includes("仅在点击开始后访问麦克风或摄像头"));
});

function chatHarness() {
  const h = harness();
  let listener;
  let inputs = 0;
  h.context.window.GradLoopRealtime.RealtimeSession = () => ({
    subscribe(value) { listener = value; }, connect() {},
    sendInput() { inputs++; }, stop() { listener({ state: "stopped" }); },
  });
  h.context.WebSocket = class {};
  vm.runInContext("let realtimeMedia = null, realtimeTimer = null;\n" + html.slice(html.indexOf("    async function startRealtime("), html.indexOf("    function renderCitations(")), h.context);
  h.byId("realtimeMode").value = "chat";
  h.byId("mapChatText").value = "Explain this synthetic public example";
  return { ...h, emit: (event) => listener(event), inputs: () => inputs };
}

test("MAP chat requires confirmation, sends only on ready and verifies a nonempty completed reply", async () => {
  const h = chatHarness();
  await h.context.startRealtime();
  assert.equal(h.inputs(), 0);
  h.byId("mapChatConsent").checked = true;
  await h.context.startRealtime();
  assert.match(h.byId("mapConnectionBadge").textContent, /连接中/);
  h.emit({ state: "ready", type: "ready" });
  assert.equal(h.inputs(), 1);
  assert.ok(!h.byId("mapConnectionBadge").className.includes("ready"));
  h.emit({ state: "responding", type: "delta", kind: "text", text: "Synthetic reply" });
  h.emit({ state: "ready", type: "done" });
  assert.equal(h.inputs(), 1);
  assert.match(h.byId("mapConnectionBadge").textContent, /本次推理完成/);
  assert.match(h.byId("realtimeCaption").textContent, /Synthetic reply/);
});

test("an empty completed MAP response is not success and unimplemented device modes do not open a session", async () => {
  const h = chatHarness();
  h.byId("mapChatConsent").checked = true;
  await h.context.startRealtime();
  h.emit({ state: "ready", type: "ready" });
  h.emit({ state: "ready", type: "done" });
  assert.match(h.byId("mapConnectionBadge").textContent, /未完成/);
  assert.ok(!h.byId("mapConnectionBadge").className.includes("ready"));
  const audio = chatHarness();
  audio.byId("realtimeMode").value = "audio";
  await audio.context.startRealtime();
  assert.match(audio.byId("realtimeCaption").textContent, /尚未开放/);
  assert.equal(audio.inputs(), 0);
});

test("whitespace-only document replies do not claim a completed analysis", async () => {
  const h = harness();
  let listener;
  h.context.window.GradLoopRealtime.RealtimeSession = () => ({
    subscribe(value) { listener = value; }, connect() {}, sendInput() {}, stop() {},
  });
  await h.context.parseDocuments();
  h.byId("documentConsent").checked = true;
  await h.context.sendDocumentText();
  listener({ state: "ready", type: "ready" });
  listener({ state: "responding", type: "delta", kind: "text", text: " \n " });
  listener({ state: "ready", type: "done" });
  assert.match(h.byId("documentOut").textContent, /没有文字结果/);
  assert.match(h.byId("mapConnectionBadge").textContent, /未完成/);
});
