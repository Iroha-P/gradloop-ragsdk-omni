"""Build a closed public release tree from the versioned release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import public_release_scan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify the manifest-defined public release.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _load_renames(root: Path) -> dict[str, str]:
    payload = json.loads((root / public_release_scan.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    return {str(source): str(target) for source, target in payload["rename"].items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.root.is_symlink() or args.output.is_symlink():
        print("public release build failed: symlink root or output is not allowed")
        return 2
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir() or output == root:
        print("public release build failed: invalid root or output")
        return 2
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print("public release build failed: output must be absent or empty")
        return 2

    denylist_terms, denylist_findings = public_release_scan._load_denylist(
        root / public_release_scan.DEFAULT_LOCAL_DENYLIST,
        required=False,
    )
    if denylist_findings:
        categories = ", ".join(sorted(denylist_findings))
        print(f"public release build failed: local privacy categories={categories}")
        return 1

    source_findings = public_release_scan.scan(
        root,
        manifest_name=public_release_scan.DEFAULT_MANIFEST,
        max_file_bytes=public_release_scan.LARGE_FILE_BYTES,
        denylist_terms=denylist_terms,
    )
    if source_findings:
        categories = ", ".join(sorted(source_findings))
        print(f"public release build failed: source categories={categories}")
        return 1

    files, findings = public_release_scan._manifest_files(
        root,
        public_release_scan.DEFAULT_MANIFEST,
    )
    if findings:
        categories = ", ".join(sorted(findings))
        print(f"public release build failed: manifest categories={categories}")
        return 1

    renames = _load_renames(root)
    planned: list[tuple[Path, str]] = []
    destinations: set[str] = set()
    for source in files:
        relative = source.relative_to(root).as_posix()
        if relative in public_release_scan.REQUIRED_GENERATED:
            continue
        destination = renames.get(relative, relative)
        canonical = public_release_scan._canonical_entry(destination)
        resolved_source = source.resolve()
        if (
            canonical is None
            or source.is_symlink()
            or not _inside(resolved_source, root)
            or destination in destinations
        ):
            print("public release build failed: unsafe source or destination")
            return 1
        destinations.add(destination)
        planned.append((source, destination))

    output.mkdir(parents=True, exist_ok=True)
    for source, relative in planned:
        destination = (output / relative).resolve()
        if not _inside(destination, output):
            print("public release build failed: destination escaped output")
            return 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    hashes = {
        relative: _sha256(output / relative)
        for relative in sorted(destinations)
    }
    hash_path = output / "SHA256SUMS.json"
    hash_path.write_text(
        json.dumps({"algorithm": "sha256", "files": hashes}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    release_findings = public_release_scan.scan(
        output,
        manifest_name=public_release_scan.DEFAULT_MANIFEST,
        max_file_bytes=public_release_scan.LARGE_FILE_BYTES,
        denylist_terms=denylist_terms,
    )
    if release_findings:
        categories = ", ".join(sorted(release_findings))
        print(f"public release build failed: archive categories={categories}")
        return 1
    print(f"public release build passed: files={len(destinations) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
