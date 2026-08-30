# Public release guide

This guide covers only a public, synthetic-data release. Do not add private
documents, local indexes, credentials, model weights, databases, or machine
paths to the release tree.

The versioned `release-manifest.json` is the authoritative publication list.
The safety scan validates that every listed path exists, stays inside the
project, is unique, and does not cover a forbidden local-work directory. Files
outside that manifest are not release candidates.

## Bootstrap in a clean environment

Use Python 3.11 or 3.12, create a virtual environment, then install the public
project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\public_release_scan.py
.\.venv\Scripts\python.exe -m pytest tests\test_release_assets.py -q
.\.venv\Scripts\python.exe -m ruff check app scripts tests
```

The scan is a release gate: exit code `0` means no configured blocker was
found; exit code `1` means at least one blocker category was found; exit code
`2` means the command could not inspect the requested tree. Its summary never
echoes matching secrets or document contents. Resolve every non-zero result
before publishing.

On the maintainer machine, an optional ignored file at
`data/local/privacy/release-denylist.txt` adds private names, handles, tenant
identifiers, and other publication blockers without committing those values.
The release builder applies the same local denylist before and after copying
the manifest-defined tree.

## Build the release archive

Do not compress or copy the working directory manually. Build an absent or
empty output directory strictly from the versioned manifest:

```powershell
.\.venv\Scripts\python.exe scripts\build_public_release.py --output <empty-output-directory>
```

The builder refuses a non-empty destination, rejects symlinks and escaped
paths, applies the declared public README rename, writes `SHA256SUMS.json`, and
runs the safety scan again against the finished release tree. Only a zero exit
code is a releasable result.

Then run the CPU baseline locally:

```powershell
.\.venv\Scripts\python.exe scripts\run_public_demo.py
.\.venv\Scripts\python.exe scripts\evaluate_public_portfolio.py
```

`run_public_demo.py` binds to the local loopback interface and overrides all
corpus, question-bank, coverage, and state paths with synthetic or isolated
public-demo locations. The baseline is explicit: it does not silently become
RAGSDK when that dependency is absent.

## Container baseline

```powershell
docker compose --profile cpu up --build
```

The published host port is loopback-only (`127.0.0.1:8000`). The container runs
as a non-root user and persists only application state to the named
`gradloop-state` volume at `/app/runtime`. It includes the 20-document synthetic
public corpus and intentionally does not mount any host data directory; mounting
personal data would create a separate local-only deployment decision.

Check both `http://127.0.0.1:8000/health` and
`http://127.0.0.1:8000/ready`, then stop with:

```powershell
docker compose --profile cpu down
```

The public aggregate evidence for the validated CPU run is
[`reports/public/docker_cpu_baseline_validation_v1.json`](../reports/public/docker_cpu_baseline_validation_v1.json).
It contains no raw question, answer, host path, account identifier, or personal
document content.

The `ragsdk-external` compose profile is an explicit dependency note, not an
integration runtime. It does not install RAGSDK, configure credentials, mount
data, or prove RAGSDK execution. Perform any real RAGSDK work separately in a
supported Linux/Ascend environment after following upstream requirements.

## Publication checklist

1. Run the release scan, release tests, and ruff from a clean checkout.
2. Confirm `LICENSE` is Apache-2.0 and retain `NOTICE` with the external
   RAGSDK attribution boundary.
3. Publish only synthetic examples and aggregate public reports.
4. Do not claim RAGSDK runtime verification from the CPU baseline container.
