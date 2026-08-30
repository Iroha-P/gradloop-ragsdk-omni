export function float32ToPcm16(samples) {
  const output = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    output[index] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return output;
}

export function pcm16ToBase64(samples) {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return globalThis.btoa ? globalThis.btoa(binary) : Buffer.from(bytes).toString("base64");
}

export function base64ToPcm16(value) {
  const binary = globalThis.atob ? globalThis.atob(value) : Buffer.from(value, "base64").toString("binary");
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return new Int16Array(bytes.buffer);
}

export function createPermissionGate({ getUserMedia } = {}) {
  let active = false;
  return {
    async start(constraints) {
      if (active) throw new Error("media capture already active");
      if (typeof getUserMedia !== "function") throw new Error("media permission is unavailable");
      active = true;
      try { return await getUserMedia(constraints); } catch (error) { active = false; throw error; }
    },
    stop() { active = false; },
  };
}
