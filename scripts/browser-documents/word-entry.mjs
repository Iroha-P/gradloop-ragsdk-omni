import mammoth from "mammoth";
import WordOleExtractor from "word-extractor/lib/word-ole-extractor.js";
import BufferReader from "word-extractor/lib/buffer-reader.js";
import { Buffer } from "buffer";

export async function extractDocx(arrayBuffer) {
  const result = await mammoth.extractRawText({ arrayBuffer });
  return result.value;
}

export async function extractDoc(arrayBuffer) {
  const reader = new BufferReader(Buffer.from(arrayBuffer));
  await reader.open();
  try {
    const document = await new WordOleExtractor().extract(reader);
    return document.getBody();
  } finally {
    await reader.close();
  }
}
