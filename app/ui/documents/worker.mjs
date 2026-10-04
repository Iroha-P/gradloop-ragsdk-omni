import { LIMITS, documentKind, validateZipEnvelope } from "./policy.mjs";

export async function extractDocumentData({ buffer, mime }) {
  const bytes = new Uint8Array(buffer);
  if (!bytes.length || bytes.length > LIMITS.fileBytes) throw new Error("文档大小超出限制。");
  const kind = documentKind(mime, bytes);
  let text;
  let pages = null;
  if (kind === "pdf") {
    // PDF.js 6 uses the standard typed-array hex API. Older supported browsers
    // and Node test runtimes can lack it; this equivalent polyfill handles hashes.
    if (!Uint8Array.prototype.toHex) Object.defineProperty(Uint8Array.prototype, "toHex", {
      value() { return Array.from(this, (byte) => byte.toString(16).padStart(2, "0")).join(""); },
      configurable: true, writable: true,
    });
    const pdfjs = await import("./vendor/pdf.mjs");
    pdfjs.GlobalWorkerOptions.workerSrc = new URL("./vendor/pdf.worker.mjs", import.meta.url).href;
    const loading = pdfjs.getDocument({ data: bytes, isEvalSupported: false, useSystemFonts: false, disableFontFace: true, verbosity: 0, stopAtErrors: true });
    loading.onPassword = () => { void loading.destroy(); };
    const document = await loading.promise;
    try {
      pages = document.numPages;
      if (pages > LIMITS.pages) throw new Error("PDF 超过 50 页，请先拆分。");
      const sections = [];
      for (let page = 1; page <= pages; page++) {
        const content = await (await document.getPage(page)).getTextContent();
        const line = content.items.filter((item) => typeof item.str === "string").map((item) => item.str + (item.hasEOL ? "\n" : " ")).join("").trim();
        if (line) sections.push(`［第 ${page} 页］\n${line}`);
        if (sections.join("\n\n").length > LIMITS.textChars) throw new Error("文本超过 12,000 字符，请先拆分文档。");
      }
      text = sections.join("\n\n");
      if (!text.trim()) throw new Error("PDF 没有可提取的文字层；扫描件需先进行 OCR。");
    } finally { await loading.destroy(); }
  } else {
    if (kind === "docx") validateZipEnvelope(bytes);
    const word = await import("./vendor/word.mjs");
    text = await (kind === "docx" ? word.extractDocx(buffer) : word.extractDoc(buffer));
  }
  text = String(text || "").replace(/\u0000/g, "").trim();
  if (!text || text.length > LIMITS.textChars) throw new Error("没有可用正文或正文超过 12,000 字符；请先拆分或转换文档。");
  return { kind, text, pages, paragraphs: text.split(/\n\s*\n/).filter((value) => value.trim()).length, characters: text.length };
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = async (event) => {
    try { self.postMessage({ ok: true, result: await extractDocumentData(event.data) }); }
    catch (error) {
      // Parser internals may contain document-derived strings: never forward them.
      const publicErrors = ["PDF 超过 50 页", "文本超过 12,000", "PDF 没有可提取", "文档格式与签名", "DOCX", "仅支持 PDF", "没有可用正文", "文档大小超出"];
      const message = String(error?.message || "");
      self.postMessage({ ok: false, error: publicErrors.some((term) => message.startsWith(term)) ? message : "文档解析失败：文件可能损坏、加密或使用不受支持的格式。" });
    }
  };
}
