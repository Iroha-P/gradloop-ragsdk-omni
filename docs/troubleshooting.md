# Release and demo troubleshooting

## The public release scan exits with 1

Treat it as a block to publishing. The output lists only finding categories to
avoid disclosing secrets. Remove the item from the release tree, move it to an
ignored local directory, or replace it with synthetic content. Do not paste the
matching file into an issue or chat. Re-run `python scripts/public_release_scan.py`.

## The scan reports local-data policy

Keep local data under an ignored local-data directory and retain the matching
ignore rule. Do not work around this by weakening the scanner or by adding a
local corpus to a container build context.

## Docker is unavailable or does not start

First, run the Python baseline with `python scripts/run_local.py`. For Docker,
confirm the daemon is running, then use `docker compose --profile cpu up --build`.
The service is published only to the loopback address; test the health endpoint
from the same machine. Do not broaden the port binding for a public demo.

If Docker Hub DNS is unavailable but AWS Public ECR is reachable, fetch the
Docker Official Image mirror without changing the project Dockerfile:

```powershell
docker pull public.ecr.aws/docker/library/python:3.11-slim
docker tag public.ecr.aws/docker/library/python:3.11-slim python:3.11-slim
docker build --pull=false -t gradloop-ragsdk-agent:cpu-baseline .
docker compose --profile cpu up -d --no-build
```

Record the resolved base digest and do not substitute an unrelated image.

## The UI has no local corpus

The clean CPU image contains 20 synthetic public documents but no personal
corpus or model artifact. If `/ready` does not report `20 documents indexed`,
rebuild the current Dockerfile and confirm that
`data/public_eval/portfolio_v1/corpus.jsonl` exists in the release tree. Never
fix this by mounting a private host directory for a public demo.

## RAGSDK is requested

The RAGSDK compose profile is documentation only. It neither installs nor
validates RAGSDK. Use a separately provisioned supported Linux/Ascend setup,
obtain the upstream dependency yourself, and keep its credentials and data out
of this project. A CPU container result is not RAGSDK runtime verification.
