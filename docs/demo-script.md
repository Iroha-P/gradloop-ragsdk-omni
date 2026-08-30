# 2–4 minute public demo

Use only the included synthetic examples. Before recording, close terminals,
browser tabs, notification panels, and file explorers that could reveal local
paths, accounts, documents, or credentials.

## Talk track

| Time | Screen and narration |
| --- | --- |
| 0:00–0:25 | Show the project root with `LICENSE`, `NOTICE`, and the release guide. Say: “This is a privacy-first RAG and learning-agent baseline. The public release contains no personal corpus or local model.” |
| 0:25–0:55 | Run `python scripts/public_release_scan.py`. Say: “This release gate checks for secrets, PII, private paths and links, oversized files, models, databases, and accidental local-data inclusion. It reports categories, never secret values.” |
| 0:55–1:35 | Run `docker compose --profile cpu up --build`, then open the loopback health endpoint. Say: “The CPU baseline runs as a non-root user and is reachable only from this machine by default.” |
| 1:35–2:35 | Run `python scripts/run_public_demo.py`, open the local UI, and use one synthetic question from the public example set. Point out retrieved evidence and the explicit baseline identity. The launcher overrides every private corpus path. |
| 2:35–3:15 | Show `NOTICE` and the RAGSDK note. Say: “RAGSDK is an external Huawei Ascend dependency, not bundled here. This demo does not claim an RAGSDK runtime result.” |
| 3:15–3:40 | Show the release test command and conclude: “The public baseline, safety gate, and evidence boundaries are reproducible.” |

## Screenshot checklist

- Included reference screenshot: [synthetic public Q&A](assets/demo-public-qa.png).
- Release root showing only public files: `LICENSE`, `NOTICE`, container files, and public docs.
- Successful scan summary with no file content or environment variables visible.
- Container health response at the loopback address.
- Local UI with synthetic content only.
- `NOTICE` section explaining the external RAGSDK boundary.

Do not screenshot source directories for local data, terminal history, editor
sidebars with personal files, environment variable values, or third-party
service dashboards.
