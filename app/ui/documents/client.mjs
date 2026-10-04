import { LIMITS, validateDocumentFiles, buildDocumentPrompt } from "./policy.mjs";
export { buildDocumentPrompt };

export async function parseDocuments(files, { workerFactory = () => new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" }), timeoutMs = 20000, signal } = {}) {
  validateDocumentFiles(files);
  const results = [];
  for (const file of files) {
    if (signal?.aborted) throw new Error("文档解析已取消。");
    const buffer = await file.arrayBuffer();
    if (signal?.aborted) throw new Error("文档解析已取消。");
    const worker = workerFactory();
    const result = await new Promise((resolve, reject) => {
      let timer;
      let settled = false;
      const cancel = () => finish(reject, new Error("文档解析已取消。"));
      const finish = (callback, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        signal?.removeEventListener("abort", cancel);
        worker.terminate(); callback(value);
      };
      timer = setTimeout(() => finish(reject, new Error("文档解析超时，已停止处理；请拆分或转换文档。")), timeoutMs);
      worker.onmessage = ({ data }) => data?.ok ? finish(resolve, data.result) : finish(reject, new Error(data?.error || "文档解析失败。"));
      worker.onerror = () => finish(reject, new Error("文档解析组件未能运行；请刷新或检查部署资源。"));
      signal?.addEventListener("abort", cancel, { once: true });
      try { worker.postMessage({ buffer, mime: file.type || "" }, [buffer]); }
      catch { finish(reject, new Error("文档解析组件未能接收数据。")); }
    });
    results.push(result);
    if (results.reduce((sum, item) => sum + item.text.length + 50, 0) > LIMITS.textChars) throw new Error("合并文本超过 12,000 字符；不会静默截断，请减少文档。");
  }
  return {
    text: results.map((item, index) => `［材料 ${index + 1} · ${item.kind.toUpperCase()}］\n${item.text}`).join("\n\n"),
    metadata: results.map(({ kind, pages, paragraphs, characters }) => ({ kind, pages, paragraphs, characters })),
  };
}
