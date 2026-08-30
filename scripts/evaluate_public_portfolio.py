from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "data" / "public_eval" / "portfolio_v1"
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.public_portfolio import (
    build_failed_public_portfolio_report,
    run_public_portfolio,
    scan_public_assets,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated public synthetic portfolio evaluation without models "
            "or external services."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "public" / "public_synthetic_portfolio_v1.json",
    )
    return parser.parse_args(argv)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    privacy_scan = scan_public_assets(ASSET_ROOT)
    try:
        with tempfile.TemporaryDirectory(prefix="public-portfolio-") as workdir:
            report = run_public_portfolio(ASSET_ROOT, workdir=Path(workdir))
    except (OSError, ValueError):
        report = build_failed_public_portfolio_report(privacy_scan=privacy_scan)
        _write(output, report.model_dump(mode="json"))
        print(json.dumps({"status": report.status, "failure_type": report.failure_type}))
        return 2
    _write(output, report.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "status": report.status,
                "dataset_version": report.dataset_version,
                "case_count": report.case_count,
                "unique_case_count": report.unique_case_count,
                "output": "reports/public",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
