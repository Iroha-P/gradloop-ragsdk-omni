import assert from "node:assert/strict";
import test from "node:test";
import { extractDocumentData } from "../../app/ui/documents/worker.mjs";
import { parseDocuments } from "../../app/ui/documents/client.mjs";
import { LIMITS } from "../../app/ui/documents/policy.mjs";

const bufferOf = (value) => Uint8Array.from(value).buffer;

function syntheticPdf(text = "Synthetic study evidence") {
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    `<< /Length ${(`BT /F1 12 Tf 20 300 Td (${text}) Tj ET`).length} >>\nstream\nBT /F1 12 Tf 20 300 Td (${text}) Tj ET\nendstream`,
  ];
  let source = "%PDF-1.4\n";
  const offsets = [0];
  objects.forEach((object, index) => { offsets.push(source.length); source += `${index + 1} 0 obj\n${object}\nendobj\n`; });
  const xref = source.length;
  source += `xref\n0 6\n0000000000 65535 f \n${offsets.slice(1).map((offset) => String(offset).padStart(10, "0") + " 00000 n \n").join("")}trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return new TextEncoder().encode(source).buffer;
}

function syntheticDocx(text = "Synthetic document evidence") {
  const name = new TextEncoder().encode("word/document.xml");
  const xml = new TextEncoder().encode(`<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>${text}</w:t></w:r></w:p></w:body></w:document>`);
  let crc = 0xffffffff;
  for (const byte of xml) { crc ^= byte; for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0); }
  crc = (crc ^ 0xffffffff) >>> 0;
  const local = new Uint8Array(30 + name.length + xml.length);
  const l = new DataView(local.buffer);
  l.setUint32(0, 0x04034b50, true); l.setUint16(4, 20, true); l.setUint32(14, crc, true);
  l.setUint32(18, xml.length, true); l.setUint32(22, xml.length, true); l.setUint16(26, name.length, true);
  local.set(name, 30); local.set(xml, 30 + name.length);
  const central = new Uint8Array(46 + name.length);
  const c = new DataView(central.buffer);
  c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint32(16, crc, true);
  c.setUint32(20, xml.length, true); c.setUint32(24, xml.length, true); c.setUint16(28, name.length, true);
  central.set(name, 46);
  const result = new Uint8Array(local.length + central.length + 22);
  result.set(local); result.set(central, local.length);
  const e = new DataView(result.buffer, local.length + central.length);
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, 1, true); e.setUint16(10, 1, true);
  e.setUint32(12, central.length, true); e.setUint32(16, local.length, true);
  return result.buffer;
}

function syntheticDoc(text = "Synthetic legacy Word evidence") {
  const file = new Uint8Array(512 * 19);
  const header = new DataView(file.buffer, 0, 512);
  file.set([208, 207, 17, 224, 161, 177, 26, 225]);
  header.setUint16(24, 0x3e, true); header.setUint16(26, 3, true); header.setUint16(28, 0xfffe, true);
  header.setUint16(30, 9, true); header.setUint16(32, 6, true);
  header.setUint32(44, 1, true); header.setUint32(48, 1, true); header.setUint32(56, 4096, true);
  header.setUint32(60, 0xfffffffe, true); header.setUint32(68, 0xfffffffe, true);
  for (let offset = 76; offset < 512; offset += 4) header.setUint32(offset, offset === 76 ? 0 : 0xffffffff, true);
  const fat = new DataView(file.buffer, 512, 512);
  for (let index = 0; index < 128; index++) fat.setUint32(index * 4, 0xffffffff, true);
  fat.setUint32(0, 0xfffffffd, true); fat.setUint32(4, 0xfffffffe, true);
  for (let sector = 2; sector <= 17; sector++) fat.setUint32(sector * 4, sector === 9 || sector === 17 ? 0xfffffffe : sector + 1, true);
  const directory = (index, name, type, child, right, sector, size) => {
    const view = new DataView(file.buffer, 1024 + index * 128, 128);
    for (let i = 0; i < name.length; i++) view.setUint16(i * 2, name.charCodeAt(i), true);
    view.setUint16(64, (name.length + 1) * 2, true); view.setUint8(66, type); view.setUint8(67, 1);
    view.setUint32(68, 0xffffffff, true); view.setUint32(72, right, true); view.setUint32(76, child, true);
    view.setUint32(116, sector, true); view.setUint32(120, size, true);
  };
  directory(0, "Root Entry", 5, 1, 0xffffffff, 0xfffffffe, 0);
  directory(1, "WordDocument", 2, 0xffffffff, 2, 2, 4096);
  directory(2, "0Table", 2, 0xffffffff, 0xffffffff, 10, 4096);
  const word = new DataView(file.buffer, 1536, 4096);
  word.setUint16(0, 0xa5ec, true); word.setUint16(2, 0xc1, true);
  word.setUint32(0x18, 1024, true); word.setUint32(0x4c, text.length, true);
  for (let i = 0; i < text.length; i++) word.setUint16(1024 + i * 2, text.charCodeAt(i), true);
  const table = new DataView(file.buffer, 5632, 4096);
  table.setUint8(0, 2); table.setUint32(1, 16, true); table.setUint32(5, 0, true);
  table.setUint32(9, text.length, true); table.setUint32(15, 1024, true);
  return file.buffer;
}

test("actually extracts a synthetic PDF text layer", async () => {
  const result = await extractDocumentData({ buffer: syntheticPdf(), mime: "application/pdf" });
  assert.match(result.text, /Synthetic study evidence/);
  assert.equal(result.pages, 1);
});

test("actually extracts DOCX raw text without rendering embedded markup", async () => {
  const result = await extractDocumentData({ buffer: syntheticDocx("Evidence &lt;script&gt; not executed"), mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" });
  assert.match(result.text, /Evidence <script> not executed/);
  assert.equal(result.kind, "docx");
});

test("actually extracts a synthetic OLE-based DOC", async () => {
  const result = await extractDocumentData({ buffer: syntheticDoc(), mime: "application/msword" });
  assert.equal(result.text, "Synthetic legacy Word evidence");
  assert.equal(result.kind, "doc");
});

test("malformed and non-Word files fail rather than producing fabricated text", async () => {
  await assert.rejects(extractDocumentData({ buffer: bufferOf([37, 80, 68, 70, 45]), mime: "application/pdf" }));
  await assert.rejects(extractDocumentData({ buffer: bufferOf([208, 207, 17, 224, 161, 177, 26, 225]), mime: "application/msword" }));
});

test("client transmits bytes and MIME only to a local worker, never original filenames", async () => {
  let payload;
  let terminated = 0;
  const worker = { terminate() { terminated++; }, postMessage(value) { payload = value; queueMicrotask(() => this.onmessage({ data: { ok: true, result: { kind: "doc", text: "Synthetic", paragraphs: 1, pages: null, characters: 9 } } })); } };
  const result = await parseDocuments([{ name: "must-not-be-forwarded.doc", type: "application/msword", size: 8, arrayBuffer: async () => bufferOf([208, 207, 17, 224, 161, 177, 26, 225]) }], { workerFactory: () => worker });
  assert.deepEqual(Object.keys(payload).sort(), ["buffer", "mime"]);
  assert.ok(!JSON.stringify(result).includes("must-not-be-forwarded"));
  assert.equal(terminated, 1);
});

test("timeouts terminate parsing and size validation precedes any byte read", async () => {
  let read = 0;
  let terminated = 0;
  const file = { size: LIMITS.fileBytes + 1, arrayBuffer() { read++; } };
  await assert.rejects(parseDocuments([file]));
  assert.equal(read, 0);
  await assert.rejects(parseDocuments([{ size: 1, arrayBuffer: async () => bufferOf([1]) }], { timeoutMs: 5, workerFactory: () => ({ postMessage() {}, terminate() { terminated++; } }) }));
  assert.equal(terminated, 1);
});

test("cancelling local parsing terminates its worker and rejects stale results", async () => {
  const controller = new AbortController();
  let terminated = 0;
  const pending = parseDocuments([{ size: 1, arrayBuffer: async () => bufferOf([1]) }], {
    signal: controller.signal,
    workerFactory: () => ({ postMessage() { controller.abort(); }, terminate() { terminated++; } }),
  });
  await assert.rejects(pending, /取消/);
  assert.equal(terminated, 1);
});
