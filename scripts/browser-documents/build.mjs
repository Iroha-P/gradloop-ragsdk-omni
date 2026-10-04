import { build } from "esbuild";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const target = resolve(root, "../../app/ui/documents/vendor");
await mkdir(target, { recursive: true });
const bundle = await build({
  entryPoints: [join(root, "word-entry.mjs")],
  outfile: join(target, "word.mjs"),
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
  legalComments: "eof",
  alias: { stream: "stream-browserify", events: "events", buffer: "buffer" },
  inject: [join(root, "globals.mjs")],
  sourcemap: false,
  metafile: true,
});
for (const name of ["pdf.mjs", "pdf.worker.mjs"]) {
  await copyFile(join(root, "node_modules/pdfjs-dist/build", name), join(target, name));
}

const lock = JSON.parse(await readFile(join(root, "package-lock.json"), "utf8"));
const notices = ["# Browser document parser third-party notices", "", "Pinned packages are installed from the npm lockfile; parsing assets are served locally, not from a CDN.", ""];
for (const [path, entry] of Object.entries(lock.packages).sort(([a], [b]) => a.localeCompare(b))) {
  if (!path.startsWith("node_modules/")) continue;
  const bundled = Object.keys(bundle.metafile.inputs).some((input) => input.replaceAll("\\", "/").startsWith(path + "/"));
  if (!bundled && path !== "node_modules/pdfjs-dist") continue;
  const packageRoot = join(root, path);
  let pkg;
  try { pkg = JSON.parse(await readFile(join(packageRoot, "package.json"), "utf8")); } catch { continue; }
  if (pkg.name === "esbuild" || pkg.name.startsWith("@esbuild/") || pkg.name.startsWith("@napi-rs/")) continue;
  let license;
  for (const name of ["LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.markdown", "LICENSE-MIT", "LICENSE-MIT.txt", "license", "license.md", "LICENCE", "LICENCE.md", "LICENCE.txt"]) {
    try { license = await readFile(join(packageRoot, name), "utf8"); break; } catch { /* other conventional filename */ }
  }
  if (!license) {
    try {
      const readme = await readFile(join(packageRoot, "README.md"), "utf8");
      const start = readme.search(/^##? License\s*$/im);
      if (start >= 0 && readme.slice(start).includes("Permission is hereby granted")) license = readme.slice(start);
    } catch { /* fail closed below */ }
  }
  if (!license) throw new Error(`Missing license notice for ${pkg.name}`);
  notices.push(`## ${pkg.name} ${entry.version}`, "", `License: ${pkg.license || "see notice"}`, "", "```text", license.trim(), "```", "");
}
await writeFile(join(target, "THIRD_PARTY_NOTICES.md"), notices.join("\n"), "utf8");
const files = {};
for (const name of ["pdf.mjs", "pdf.worker.mjs", "word.mjs", "THIRD_PARTY_NOTICES.md"]) {
  files[name] = createHash("sha256").update(await readFile(join(target, name))).digest("hex");
}
await writeFile(join(target, "provenance.json"), JSON.stringify({
  schema_version: 1,
  dependencies: { "pdfjs-dist": "6.4.299", mammoth: "1.13.0", "word-extractor": "1.0.4" },
  files,
}, null, 2) + "\n", "utf8");
console.log("Browser document assets built with pinned provenance and license notices.");
