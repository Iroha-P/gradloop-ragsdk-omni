"""Scan the versioned public-release manifest without revealing finding values."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "release-manifest.json"
DEFAULT_LOCAL_DENYLIST = "data/local/privacy/release-denylist.txt"
LARGE_FILE_BYTES = 10 * 1024 * 1024
MODEL_SUFFIXES = {".bin", ".gguf", ".safetensors", ".pt", ".pth", ".onnx"}
DATABASE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
KEY_SUFFIXES = {".pem", ".key", ".crt"}
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".dockerignore",
    ".env",
    ".example",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {"Dockerfile", "LICENSE", "NOTICE"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUIRED_INCLUDE_ENTRIES = {
    ".dockerignore",
    "Dockerfile",
    "LICENSE",
    "NOTICE",
    "README.public.md",
    "app",
    "compose.yaml",
    "configs",
    "data/public_eval/portfolio_v1",
    "docs/demo-script.md",
    "docs/release.md",
    "docs/troubleshooting.md",
    "pyproject.toml",
    "release-manifest.json",
    "reports/public/public_synthetic_portfolio_v1.json",
    "scripts/build_public_release.py",
    "scripts/evaluate_public_portfolio.py",
    "scripts/public_release_scan.py",
    "scripts/run_public_demo.py",
    "scripts/run_local.py",
    "tests/integration/test_api_learning_loop.py",
    "tests/test_release_assets.py",
}
REQUIRED_EXPANDED_FILES = {
    "README.md",
    "app/api/main.py",
    "app/core/config.py",
    "configs/local_baseline.yaml",
    "data/public_eval/portfolio_v1/manifest.json",
    "reports/public/public_synthetic_portfolio_v1.json",
    "scripts/build_public_release.py",
    "scripts/evaluate_public_portfolio.py",
    "scripts/public_release_scan.py",
    "scripts/run_public_demo.py",
    "scripts/run_local.py",
    "tests/integration/test_api_learning_loop.py",
    "tests/test_release_assets.py",
}
PROTECTED_PREFIXES = REQUIRED_INCLUDE_ENTRIES | {
    "README.md",
    "app",
    "configs",
    "data/public_eval/portfolio_v1",
}
REQUIRED_RENAMES = {"README.public.md": "README.md"}
REQUIRED_GENERATED = {"SHA256SUMS.json"}
REQUIRED_EXCLUDES = {
    ".superpowers",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".coverage",
    "baoyan_ragsdk_agent.egg-info",
    "data/local",
    "data/private",
    "data/staging",
    "docs/superpowers",
    "交接说明.md",
    "开发总提示词.md",
}
FORBIDDEN_PREFIXES = REQUIRED_EXCLUDES | {
    ".git",
    ".env",
    "venv",
    "htmlcov",
    "models",
    "weights",
    "indexes",
    "uploads",
    "exports",
    "logs",
    "traces",
    "reports/private",
    ".dev.vars",
    ".wrangler",
    "wrangler.toml",
}

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"),
)
PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"
    r"|(?<![A-Za-z0-9_.:/-])/(?:home|Users|private|mnt)/"
)
PRIVATE_FEISHU_PATTERN = re.compile(
    r"(?i)https?://[^\s/]*(?:feishu\.cn|larksuite\.com)/(?:wiki|docx|base|drive)/[A-Za-z0-9_-]+"
)
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b")
PHONE_PATTERN = re.compile(r"(?<![A-Za-z0-9])1[3-9]\d{9}(?![A-Za-z0-9])")
RESERVED_EMAIL_DOMAINS = {"example.com", "example.org", "example.net"}
GLOBAL_CANARIES = ("C:/synthetic-canary/release-scan-fixture.txt",)

# Reviewed, reproducible OSS parser bytes contain language examples, property
# accesses and legally required public author contacts. Mask only those exact
# literals in the exact pinned file digest; a one-byte change restores all gates.
REVIEWED_VENDOR_LITERALS = {
    "app/ui/documents/vendor/pdf.worker.mjs": (
        "07ceb740d746e5d9012fcf9f27d5c7dea09f7d2f5e74f5c8109e5787551a2333",
        (
            "/" + "home" + "/",
            "password = " + "this.hasFieldFlag",
            "password: " + "this.data.password",
            "password = " + "utf8StringToString",
            "password = " + "password.subarray",
            "password: " + "pageInfo.password",
            "jm" + "@" + "kbswfq.pkbgltMlwbaof",
        ),
    ),
    "app/ui/documents/vendor/word.mjs": (
        "cd608348378469a86bfce7cd6e5a782bf9cb58ecc4da19e50f538547a25aff17",
        ("g:" + "/",),
    ),
    "app/ui/documents/vendor/THIRD_PARTY_NOTICES.md": (
        "f6c2aa9c90c4cbe476ca4e7038d8b420a37bef46686292a8363ac20ebe56643e",
        (
            "jindw" + "@" + "xidea.org",
            "nathan" + "@" + "tootallnate.net",
            "shtylman" + "@" + "gmail.com",
        ),
    ),
}

# Exact source literals used to prove redaction behavior.  These are removed only
# from the named public test/source files; arbitrary paths in any other file fail.
SOURCE_CANARIES: dict[str, tuple[str, ...]] = {
    "tests/test_release_assets.py": (
        "alice@realmail.cn",
        "C:/Users/alice/resume.docx",
        "https://acme.feishu.cn/wiki/AbCdEf012345",
        "C:/synthetic-canary/release-scan-fixture.txt",
    ),
    "tests/integration/test_boundary_gate_dev_script.py": (
        "C:/private/name",
        r"c:\\",
        "/private/",
    ),
    "tests/integration/test_hard_retrieval_dataset.py": ("d:" + chr(92) * 2, "/home/"),
    "tests/integration/test_boundary_gate_ab_script.py": (r"C:\secret",),
    "tests/integration/test_api_observability_streaming.py": ("private-answer-do-not-stream",),
    "tests/integration/test_evidence_gate_ab_script.py": (
        "C:/private/secret.txt",
        "C:/Users/private/models/qwen-secret.gguf",
        "C:/private/file",
    ),
    "tests/unit/test_boundary_gate_ab.py": (r"C:\private\runner.exe",),
    "tests/unit/test_evidence_gate_ab.py": ("C:/Users/private/models/qwen-secret.gguf",),
    "tests/unit/test_feishu_client.py": ("C:/tools/lark-cli.ps1",),
    "tests/unit/test_boundary_gate_evaluation.py": (
        r"D:\private\questions.jsonl",
        r"D:\private",
        "D:" + chr(92) * 2,
        "/home/",
    ),
    "tests/unit/test_learning_service.py": (r"D:\private\resume.pdf", r"D:\private"),
    "tests/unit/test_public_portfolio_evaluation.py": (
        r"D:\\private\\resume.docx",
        r"D:\\",
        r"C:\\Users",
    ),
    "tests/unit/test_ragsdk_matrix.py": (r"D:\\", r"C:\\Users"),
    "app/evaluation/public_portfolio.py": ("/home/", "/users/"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan the public release manifest safely.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--max-file-bytes", type=int, default=LARGE_FILE_BYTES)
    parser.add_argument(
        "--denylist-file",
        type=Path,
        help=(
            "Optional local-only file with one private identifier per line. "
            f"Defaults to {DEFAULT_LOCAL_DENYLIST} when present."
        ),
    )
    return parser.parse_args(argv)


def _canonical_entry(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or value != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        return None
    return pure.as_posix()


def _overlaps_forbidden(entry: str) -> bool:
    return any(
        entry == forbidden
        or entry.startswith(f"{forbidden}/")
        or forbidden.startswith(f"{entry}/")
        for forbidden in FORBIDDEN_PREFIXES
    )


def _is_secret_path(entry: str) -> bool:
    return any(
        entry == secret
        or entry.startswith(f"{secret}/")
        for secret in (".dev.vars", ".wrangler", "wrangler.toml")
    )


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


def _is_excluded(relative: str, excludes: set[str]) -> bool:
    path = PurePosixPath(relative)
    if path.suffix == ".pyc" or "__pycache__" in path.parts:
        return True
    return any(
        relative == excluded
        or relative.startswith(f"{excluded}/")
        or ("/" not in excluded and excluded in path.parts)
        for excluded in excludes
    )


def _has_symlink_component(path: Path, root: Path) -> bool:
    current = path
    while current != root:
        if current.is_symlink():
            return True
        if current.parent == current or (current != root and root not in current.parents):
            return True
        current = current.parent
    return root.is_symlink()


def _manifest_files(root: Path, manifest_name: str) -> tuple[list[Path], Counter[str]]:
    findings: Counter[str] = Counter()
    manifest_entry = _canonical_entry(manifest_name)
    if manifest_entry is None:
        findings["manifest_path"] += 1
        return [], findings
    manifest_path = root / manifest_entry
    if not manifest_path.is_file():
        findings["manifest_missing_path"] += 1
        return [], findings
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        findings["manifest_format"] += 1
        return [], findings
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        findings["manifest_format"] += 1
        return [], findings
    includes_raw = manifest.get("include")
    excludes_raw = manifest.get("exclude")
    rename_raw = manifest.get("rename")
    generated_raw = manifest.get("generated")
    if (
        not isinstance(includes_raw, list)
        or not isinstance(excludes_raw, list)
        or not isinstance(rename_raw, dict)
        or not isinstance(generated_raw, list)
    ):
        findings["manifest_format"] += 1
        return [], findings

    includes: list[str] = []
    excludes: set[str] = set()
    renames: dict[str, str] = {}
    generated: list[str] = []
    for raw in includes_raw:
        entry = _canonical_entry(raw)
        if entry is None:
            findings["manifest_path"] += 1
        else:
            includes.append(entry)
    for raw in excludes_raw:
        entry = _canonical_entry(raw)
        if entry is None:
            findings["manifest_path"] += 1
        else:
            excludes.add(entry)
    for raw_source, raw_target in rename_raw.items():
        source = _canonical_entry(raw_source)
        target = _canonical_entry(raw_target)
        if source is None or target is None:
            findings["manifest_path"] += 1
        else:
            renames[source] = target
    for raw in generated_raw:
        entry = _canonical_entry(raw)
        if entry is None:
            findings["manifest_path"] += 1
        else:
            generated.append(entry)
    if (
        len(includes) != len(set(includes))
        or len(excludes_raw) != len(excludes)
        or len(rename_raw) != len(renames)
        or len(set(renames.values())) != len(renames)
        or len(generated) != len(set(generated))
    ):
        findings["manifest_duplicate"] += 1
    if not REQUIRED_EXCLUDES <= excludes:
        findings["manifest_required_exclude"] += 1
    if not REQUIRED_INCLUDE_ENTRIES <= set(includes):
        findings["manifest_required_content"] += 1
    excludes_required_content = any(
        any(_paths_overlap(excluded, protected) for protected in PROTECTED_PREFIXES)
        for excluded in excludes
    )
    if excludes_required_content:
        findings["manifest_required_content"] += 1
    if any(
        source in PROTECTED_PREFIXES and REQUIRED_RENAMES.get(source) != target
        for source, target in renames.items()
    ):
        findings["manifest_required_content"] += 1
    if not REQUIRED_RENAMES.items() <= renames.items() or not REQUIRED_GENERATED <= set(generated):
        findings["manifest_required_content"] += 1
    if any(_is_secret_path(entry) for entry in includes):
        findings["secret"] += 1
    elif any(_overlaps_forbidden(entry) for entry in includes):
        findings["manifest_forbidden"] += 1
    if any(_is_secret_path(entry) for entry in [*renames.values(), *generated]):
        findings["secret"] += 1
    elif any(_overlaps_forbidden(entry) for entry in [*renames.values(), *generated]):
        findings["manifest_forbidden"] += 1
    if findings:
        return [], findings

    files: list[Path] = []
    seen: set[str] = set()
    expanded_logical: set[str] = set()
    archive_layout = any(
        not (root / source).exists() and (root / target).exists()
        for source, target in renames.items()
    )
    for entry in includes:
        target = root / entry
        logical_entry = entry
        if not target.exists() and entry in renames:
            target = root / renames[entry]
        if not target.exists():
            findings["manifest_missing_path"] += 1
            continue
        if _has_symlink_component(target, root):
            findings["manifest_symlink"] += 1
            continue
        if target.is_file():
            candidates = [target]
        else:
            descendants = sorted(target.rglob("*"))
            if any(_has_symlink_component(path, root) for path in descendants):
                findings["manifest_symlink"] += 1
                continue
            candidates = [path for path in descendants if path.is_file()]
        for candidate in candidates:
            relative = candidate.relative_to(root).as_posix()
            logical_relative = (
                logical_entry
                if target.is_file() and logical_entry in renames
                else relative
            )
            if _is_excluded(relative, excludes):
                continue
            resolved = candidate.resolve()
            if candidate.is_symlink() or (resolved != root and root not in resolved.parents):
                findings["manifest_symlink"] += 1
            elif _is_secret_path(relative):
                findings["secret"] += 1
            elif _overlaps_forbidden(relative):
                findings["manifest_forbidden"] += 1
            elif relative in seen:
                findings["manifest_duplicate"] += 1
            else:
                seen.add(relative)
                expanded_logical.add(logical_relative)
                if logical_entry in renames:
                    expanded_logical.add(renames[logical_entry])
                files.append(candidate)
    for entry in generated:
        target = root / entry
        if not target.exists():
            if archive_layout:
                findings["manifest_missing_path"] += 1
            continue
        resolved = target.resolve()
        if target.is_symlink() or (resolved != root and root not in resolved.parents):
            findings["manifest_symlink"] += 1
        elif entry in seen:
            findings["manifest_duplicate"] += 1
        else:
            seen.add(entry)
            files.append(target)
    if not REQUIRED_EXPANDED_FILES <= expanded_logical:
        findings["manifest_required_content"] += 1
    return files, findings


def _without_source_canaries(text: str, relative_path: str) -> str:
    for canary in GLOBAL_CANARIES:
        text = text.replace(canary, "<synthetic-canary>")
    canaries = list(SOURCE_CANARIES.get(relative_path, ()))
    if relative_path == "scripts/public_release_scan.py":
        canaries.extend(canary for values in SOURCE_CANARIES.values() for canary in values)
        canaries.extend(("alice@realmail.cn", "https://acme.feishu.cn/wiki/AbCdEf012345"))
    for canary in sorted(set(canaries), key=len, reverse=True):
        text = text.replace(canary, "<synthetic-canary>")
        text = text.replace(canary.replace("\\", "\\\\"), "<synthetic-canary>")
    return text


def _has_non_reserved_email(text: str) -> bool:
    for match in EMAIL_PATTERN.finditer(text):
        domain = match.group(1).lower()
        if domain not in RESERVED_EMAIL_DOMAINS and not domain.endswith(".test"):
            return True
    return False


def _load_denylist(path: Path, *, required: bool) -> tuple[tuple[str, ...], Counter[str]]:
    findings: Counter[str] = Counter()
    if not path.exists():
        if required:
            findings["denylist_unavailable"] += 1
        return (), findings
    if not path.is_file() or path.is_symlink():
        findings["denylist_unavailable"] += 1
        return (), findings
    try:
        values = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except (OSError, UnicodeError):
        findings["denylist_unavailable"] += 1
        return (), findings
    if any(len(value) < 2 or len(value) > 200 for value in values):
        findings["denylist_format"] += 1
        return (), findings
    return tuple(sorted(values, key=lambda value: (-len(value), value.casefold()))), findings


def _scan_text(
    text: str,
    relative_path: str,
    findings: Counter[str],
    *,
    denylist_terms: tuple[str, ...],
) -> None:
    reviewed = REVIEWED_VENDOR_LITERALS.get(relative_path)
    if reviewed and hashlib.sha256(text.encode("utf-8")).hexdigest() == reviewed[0]:
        for literal in reviewed[1]:
            text = text.replace(literal, "<reviewed-public-oss-literal>")
    text = _without_source_canaries(text, relative_path)
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        findings["secret"] += 1
    if PRIVATE_PATH_PATTERN.search(text):
        findings["private_absolute_path"] += 1
    if PRIVATE_FEISHU_PATTERN.search(text):
        findings["private_feishu_link"] += 1
    if _has_non_reserved_email(text) or PHONE_PATTERN.search(text):
        findings["pii"] += 1
    folded = text.casefold()
    if any(term.casefold() in folded for term in denylist_terms):
        findings["forbidden_entity"] += 1


def _png_metadata_text(data: bytes) -> str:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("invalid PNG signature")
    offset = len(PNG_SIGNATURE)
    texts: list[str] = []
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        chunk_end = payload_end + 4
        if chunk_end > len(data):
            raise ValueError("truncated PNG payload")
        payload = data[payload_start:payload_end]
        if kind == b"tEXt":
            texts.append(payload.decode("latin-1", errors="replace"))
        elif kind == b"zTXt":
            keyword, separator, remainder = payload.partition(b"\x00")
            if not separator or len(remainder) < 2 or remainder[0] != 0:
                raise ValueError("invalid PNG zTXt chunk")
            texts.append(keyword.decode("latin-1", errors="replace"))
            texts.append(zlib.decompress(remainder[1:]).decode("latin-1", errors="replace"))
        elif kind == b"iTXt":
            keyword, separator, remainder = payload.partition(b"\x00")
            if not separator or len(remainder) < 2:
                raise ValueError("invalid PNG iTXt chunk")
            compression_flag = remainder[0]
            compression_method = remainder[1]
            fields = remainder[2:].split(b"\x00", 2)
            if len(fields) != 3 or compression_flag not in {0, 1}:
                raise ValueError("invalid PNG iTXt fields")
            language, translated, value = fields
            if compression_flag == 1:
                if compression_method != 0:
                    raise ValueError("unsupported PNG iTXt compression")
                value = zlib.decompress(value)
            texts.extend(
                item.decode("utf-8", errors="replace")
                for item in (keyword, language, translated, value)
            )
        elif kind == b"eXIf":
            texts.append(payload.decode("latin-1", errors="replace"))
        if kind == b"IEND":
            saw_iend = True
            break
        offset = chunk_end
    if not saw_iend:
        raise ValueError("PNG is missing IEND")
    return "\n".join(texts)


def _read_scannable_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
        return path.read_text(encoding="utf-8", errors="replace")
    data = path.read_bytes()
    if suffix == ".png":
        return _png_metadata_text(data)
    if b"\x00" in data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def scan(
    root: Path,
    *,
    manifest_name: str,
    max_file_bytes: int,
    denylist_terms: tuple[str, ...] | None = None,
) -> Counter[str]:
    findings: Counter[str] = Counter()
    if denylist_terms is None:
        denylist_terms, denylist_findings = _load_denylist(
            root / DEFAULT_LOCAL_DENYLIST,
            required=False,
        )
        findings.update(denylist_findings)
    files, manifest_findings = _manifest_files(root, manifest_name)
    findings.update(manifest_findings)
    if findings:
        return findings
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size > max_file_bytes:
            findings["large_file"] += 1
        suffix = path.suffix.lower()
        if suffix in MODEL_SUFFIXES:
            findings["model_artifact"] += 1
        if suffix in DATABASE_SUFFIXES:
            findings["database_artifact"] += 1
        if suffix in KEY_SUFFIXES:
            findings["key_material"] += 1
        if size <= max_file_bytes:
            try:
                text = _read_scannable_text(path)
            except (OSError, ValueError, zlib.error):
                findings["invalid_binary"] += 1
                continue
            _scan_text(
                text,
                relative,
                findings,
                denylist_terms=denylist_terms,
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir() or args.max_file_bytes < 1:
        print("release scan error: invalid arguments")
        return 2
    denylist_path = (
        args.denylist_file.resolve()
        if args.denylist_file is not None
        else root / DEFAULT_LOCAL_DENYLIST
    )
    denylist_terms, denylist_findings = _load_denylist(
        denylist_path,
        required=args.denylist_file is not None,
    )
    findings = Counter(denylist_findings)
    findings.update(
        scan(
            root,
            manifest_name=args.manifest,
            max_file_bytes=args.max_file_bytes,
            denylist_terms=denylist_terms,
        )
    )
    if findings:
        summary = ", ".join(f"{category}={count}" for category, count in sorted(findings.items()))
        print(f"release scan failed: {summary}")
        return 1
    print("release scan passed: manifest is valid and no publish blockers were found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
