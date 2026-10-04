from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from scripts import public_release_scan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCANNER = PROJECT_ROOT / "scripts" / "public_release_scan.py"
BUILDER = PROJECT_ROOT / "scripts" / "build_public_release.py"
REQUIRED_INCLUDES = [
    "LICENSE",
    "NOTICE",
    "Dockerfile",
    "compose.yaml",
    ".dockerignore",
    "pyproject.toml",
    "README.public.md",
    "release-manifest.json",
    "app",
    "configs",
    "data/public_eval/portfolio_v1",
    "reports/public/public_synthetic_portfolio_v1.json",
    "scripts/public_release_scan.py",
    "scripts/run_public_demo.py",
    "scripts/build_public_release.py",
    "scripts/run_local.py",
    "scripts/evaluate_public_portfolio.py",
    "tests/test_release_assets.py",
    "tests/integration/test_api_learning_loop.py",
    "docs/release.md",
    "docs/demo-script.md",
    "docs/troubleshooting.md",
]
REQUIRED_EXCLUDES = [
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
]
ONE_KEY_INCLUDES = [
    "deploy/map-realtime-worker/src",
    "deploy/map-realtime-worker/test",
    "deploy/map-realtime-worker/wrangler.one-key.jsonc",
    "scripts/competition/map_setup_core.py",
    "scripts/competition/setup_map_demo.ps1",
    "tests/competition/test_one_key_map_setup.py",
    "docs/competition/one-key-map-setup.md",
    "reports/public/competition/one-key-setup-local-verification.json",
]


def _write_manifest(
    root: Path,
    includes: list[str],
    *,
    extra_excludes: list[str] | None = None,
) -> None:
    fixture_files = {
        "LICENSE": "safe\n",
        "NOTICE": "safe\n",
        "Dockerfile": "safe\n",
        "compose.yaml": "safe\n",
        ".dockerignore": "safe\n",
        "pyproject.toml": "safe\n",
        "README.public.md": "# Safe\n",
        "app/api/main.py": "# safe\n",
        "app/core/__init__.py": "# safe\n",
        "app/core/config.py": "# safe\n",
        "configs/local_baseline.yaml": "safe: true\n",
        "data/public_eval/portfolio_v1/manifest.json": "{}\n",
        "reports/public/public_synthetic_portfolio_v1.json": "{}\n",
        "scripts/public_release_scan.py": "# safe\n",
        "scripts/run_public_demo.py": "# safe\n",
        "scripts/build_public_release.py": "# safe\n",
        "scripts/run_local.py": "# safe\n",
        "scripts/evaluate_public_portfolio.py": "# safe\n",
        "tests/test_release_assets.py": "# safe\n",
        "tests/integration/test_api_learning_loop.py": "# safe\n",
        "docs/release.md": "safe\n",
        "docs/demo-script.md": "safe\n",
        "docs/troubleshooting.md": "safe\n",
    }
    for relative, content in fixture_files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(content, encoding="utf-8")
    manifest_includes = [*REQUIRED_INCLUDES, *includes]
    (root / "release-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "include": manifest_includes,
                "exclude": [*REQUIRED_EXCLUDES, *(extra_excludes or [])],
                "rename": {"README.public.md": "README.md"},
                "generated": ["SHA256SUMS.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _run_scan(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCANNER), "--root", str(root), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum)
    return (
        len(payload).to_bytes(4, "big")
        + kind
        + payload
        + checksum.to_bytes(4, "big")
    )


def test_release_legal_and_container_assets_are_present_and_safe() -> None:
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    notice_text = (PROJECT_ROOT / "NOTICE").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert "Apache-2.0" in notice_text
    assert "Mulan PSL v2" in notice_text
    assert "not included" in notice_text.lower()
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "mkdir -p /app/runtime" in dockerfile
    assert "chown app:app /app/runtime" in dockerfile
    assert dockerfile.index("chown app:app /app/runtime") < dockerfile.index("USER app")
    assert "COPY data/public_eval/portfolio_v1/corpus.jsonl" in dockerfile
    assert "127.0.0.1" in compose
    assert "data/local" not in compose
    assert "BAOYAN_DATABASE_PATH: runtime/learning.db" in compose
    assert "BAOYAN_AGENT_CHECKPOINT_PATH: runtime/agent_checkpoints.sqlite" in compose
    assert "BAOYAN_CORPUS_PATH: data/public_eval/portfolio_v1/corpus.jsonl" in compose
    assert "gradloop-state:/app/runtime" in compose
    assert "/var/lib/baoyan" not in compose
    assert "RAGSDK" in compose


def test_dockerignore_excludes_every_private_or_large_artifact_class() -> None:
    rules = set((PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    required = {
        ".superpowers",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "*.egg-info",
        "data/local",
        "data/private",
        "data/staging",
        "docs/superpowers",
        "交接说明.md",
        "开发总提示词.md",
        "*.bin",
        "*.gguf",
        "*.safetensors",
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
    }
    assert required <= rules


def test_release_manifest_is_versioned_complete_and_excludes_work_files() -> None:
    manifest = json.loads((PROJECT_ROOT / "release-manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    includes = set(manifest["include"])
    assert set(REQUIRED_INCLUDES) <= includes
    assert set(ONE_KEY_INCLUDES) <= includes
    assert {
        "app",
        "configs",
        "data/public_demo",
        "data/public_eval_hard_v1",
        "data/public_eval_boundary_dev_v2",
        "docs/release.md",
        "docs/demo-script.md",
        "docs/troubleshooting.md",
        "README.public.md",
        "pyproject.toml",
    } <= includes
    assert set(REQUIRED_EXCLUDES) <= set(manifest["exclude"])
    assert manifest["rename"] == {"README.public.md": "README.md"}
    assert manifest["generated"] == ["SHA256SUMS.json"]


def test_one_key_release_assets_are_secret_free_and_allowlisted() -> None:
    manifest = json.loads((PROJECT_ROOT / "release-manifest.json").read_text(encoding="utf-8"))
    includes = set(manifest["include"])
    assert set(ONE_KEY_INCLUDES) <= includes
    config = (PROJECT_ROOT / "deploy" / "map-realtime-worker" / "wrangler.one-key.jsonc").read_text(
        encoding="utf-8"
    )
    wizard = (PROJECT_ROOT / "scripts" / "competition" / "setup_map_demo.ps1").read_text(
        encoding="utf-8"
    )
    assert "MAP_API_KEY=" not in config
    assert "Write-Host $apiKey" not in wizard
    assert "--project-name" in wizard
    assert "git push" not in wizard
    assert "git clean" not in wizard


def test_release_documentation_covers_bootstrap_demo_and_troubleshooting() -> None:
    release = (PROJECT_ROOT / "docs" / "release.md").read_text(encoding="utf-8")
    demo = (PROJECT_ROOT / "docs" / "demo-script.md").read_text(encoding="utf-8")
    troubleshooting = (PROJECT_ROOT / "docs" / "troubleshooting.md").read_text(encoding="utf-8")

    assert "bootstrap" in release.lower()
    assert "public_release_scan.py" in release
    assert "release-manifest.json" in release
    assert "build_public_release.py" in release
    assert "2" in demo and "4" in demo
    assert "screenshot" in demo.lower()
    assert "RAGSDK" in troubleshooting
    assert "Docker" in troubleshooting


def test_release_documentation_commands_reference_packaged_scripts() -> None:
    import re

    manifest = json.loads((PROJECT_ROOT / "release-manifest.json").read_text(encoding="utf-8"))
    packaged = set(manifest["include"])
    documents = [
        PROJECT_ROOT / ("README.public.md" if (PROJECT_ROOT / "README.public.md").is_file() else "README.md"),
        PROJECT_ROOT / "docs" / "release.md",
        PROJECT_ROOT / "docs" / "demo-script.md",
        PROJECT_ROOT / "docs" / "troubleshooting.md",
    ]
    commands: set[str] = set()
    for document in documents:
        text = document.read_text(encoding="utf-8")
        commands.update(
            match.replace("\\", "/")
            for match in re.findall(r"scripts[\\/][A-Za-z0-9_.-]+\.py", text)
        )
    assert commands
    assert commands <= packaged


def test_default_release_manifest_scan_passes_for_real_project() -> None:
    result = _run_scan(PROJECT_ROOT)

    assert result.returncode == 0, result.stdout
    assert "passed" in result.stdout


def test_public_demo_launcher_is_fail_closed_to_synthetic_assets() -> None:
    from scripts.run_public_demo import public_demo_environment

    environment = public_demo_environment()

    assert environment["BAOYAN_PUBLIC_DEMO"] == "true"
    assert environment["BAOYAN_CORPUS_PATH"].startswith("data/public_eval/")
    assert environment["BAOYAN_QUESTION_BANK_PATH"].startswith("data/public_demo/")
    assert "data/local/private" not in str(environment)
    assert "data/local/question_bank" not in str(environment)
    assert environment["BAOYAN_GENERIC_CORPUS_PATH"].endswith(
        "no-local-generic-corpus.jsonl"
    )
    assert environment["BAOYAN_PERSONAL_CORPUS_PATH"].endswith(
        "no-local-personal-corpus.jsonl"
    )


@pytest.mark.parametrize(
    ("filename", "content", "category"),
    [
        ("person.txt", "contact=alice@realmail.cn\n", "pii"),
        ("phone.txt", "contact=" + "138" + "0013" + "8000\n", "pii"),
        ("path.txt", "source=C:/Users/alice/resume.docx\n", "private_absolute_path"),
        ("link.txt", "https://acme.feishu.cn/wiki/AbCdEf012345\n", "private_feishu_link"),
        ("weights.onnx", "synthetic", "model_artifact"),
        ("state.sqlite3", "synthetic", "database_artifact"),
        ("identity.key", "synthetic", "key_material"),
    ],
)
def test_release_scanner_rejects_each_publication_blocker(
    tmp_path: Path,
    filename: str,
    content: str,
    category: str,
) -> None:
    (tmp_path / filename).write_text(content, encoding="utf-8")
    _write_manifest(tmp_path, [filename])

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert category in result.stdout
    assert content.strip() not in result.stdout
    assert content.strip() not in result.stderr


def test_release_scanner_rejects_secret_without_echoing_it(tmp_path: Path) -> None:
    secret = "ghp_" + "0123456789abcdefghijklmnopqrstuv"
    (tmp_path / "settings.txt").write_text(f"token={secret}\n", encoding="utf-8")
    _write_manifest(tmp_path, ["settings.txt"])

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert "secret" in result.stdout
    assert secret not in result.stdout
    assert secret not in result.stderr


def test_release_scan_blocks_worker_secret_files(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [".dev.vars"])
    (tmp_path / ".dev.vars").write_text("MAP_API_KEY=x\n", encoding="utf-8")

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert "secret" in result.stdout


def test_release_scanner_does_not_treat_hex_digest_as_phone_number(
    tmp_path: Path,
) -> None:
    digest = "f13438200944d0cbb763e809b53c6842b16170133ffda84d696fd7051abc"
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": digest}),
        encoding="utf-8",
    )
    _write_manifest(tmp_path, ["manifest.json"])

    result = _run_scan(tmp_path)

    assert result.returncode == 0, result.stdout


def test_release_scanner_ignores_png_image_data_false_positive(tmp_path: Path) -> None:
    image = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IDAT", b"\x00C:/Users/synthetic/compressed-image-data")
        + _png_chunk(b"IEND", b"")
    )
    (tmp_path / "icon.png").write_bytes(image)
    _write_manifest(tmp_path, ["icon.png"])

    result = _run_scan(tmp_path)

    assert result.returncode == 0, result.stdout


def test_release_scanner_checks_png_text_metadata(tmp_path: Path) -> None:
    image = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"tEXt", b"Source\x00C:/Users/private/source.png")
        + _png_chunk(b"IEND", b"")
    )
    (tmp_path / "icon.png").write_bytes(image)
    _write_manifest(tmp_path, ["icon.png"])

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert "private_absolute_path" in result.stdout
    assert "source.png" not in result.stdout


def test_release_scanner_rejects_local_denylist_terms_without_echoing_them(
    tmp_path: Path,
) -> None:
    blocked_term = "Private Mentor Org"
    (tmp_path / "public.txt").write_text(
        f"source={blocked_term}\n",
        encoding="utf-8",
    )
    denylist = tmp_path / "private-denylist.txt"
    denylist.write_text(
        f"# local-only identifiers\n{blocked_term}\n",
        encoding="utf-8",
    )
    _write_manifest(tmp_path, ["public.txt"])

    result = _run_scan(tmp_path, "--denylist-file", str(denylist))

    assert result.returncode == 1
    assert "forbidden_entity" in result.stdout
    assert blocked_term not in result.stdout
    assert blocked_term not in result.stderr


def test_release_scanner_uses_default_local_denylist_when_present(tmp_path: Path) -> None:
    blocked_term = "Private Corpus Marker"
    (tmp_path / "public.txt").write_text(blocked_term, encoding="utf-8")
    denylist = tmp_path / "data" / "local" / "privacy" / "release-denylist.txt"
    denylist.parent.mkdir(parents=True)
    denylist.write_text(blocked_term, encoding="utf-8")
    _write_manifest(tmp_path, ["public.txt"])

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert "forbidden_entity" in result.stdout
    assert blocked_term not in result.stdout


def test_release_scanner_scans_text_between_two_and_ten_mib(tmp_path: Path) -> None:
    secret = "sk-" + "z" * 32
    payload = "x" * (3 * 1024 * 1024) + f"\ntoken={secret}\n"
    (tmp_path / "large.txt").write_text(payload, encoding="utf-8")
    _write_manifest(tmp_path, ["large.txt"])

    result = _run_scan(tmp_path, "--max-file-bytes", str(4 * 1024 * 1024))

    assert result.returncode == 1
    assert "secret" in result.stdout
    assert secret not in result.stdout


def test_release_scanner_rejects_large_file(tmp_path: Path) -> None:
    (tmp_path / "large.dat").write_bytes(b"x" * 129)
    _write_manifest(tmp_path, ["large.dat"])

    result = _run_scan(tmp_path, "--max-file-bytes", "128")

    assert result.returncode == 1
    assert "large_file" in result.stdout


def test_release_scanner_allows_reserved_and_explicit_canary_fixtures(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.txt"
    fixture.write_text(
        "private@example.com\n"
        "https://private.example.test/case\n"
        "https://private.example.test/private/case\n"
        "https://private/model?api_key=secret\n"
        "data/private/corpus.jsonl\n"
        "/v1/users/synthetic-user\n"
        "C:/synthetic-canary/release-scan-fixture.txt\n",
        encoding="utf-8",
    )
    _write_manifest(tmp_path, ["fixture.txt"])

    result = _run_scan(tmp_path)

    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    ("includes", "create_path", "category"),
    [
        (["missing.txt"], None, "manifest_missing_path"),
        (["../outside.txt"], None, "manifest_path"),
        (["data/local"], "data/local", "manifest_forbidden"),
        (["safe.txt", "safe.txt"], "safe.txt", "manifest_duplicate"),
    ],
)
def test_release_scanner_rejects_invalid_manifest_entries(
    tmp_path: Path,
    includes: list[str],
    create_path: str | None,
    category: str,
) -> None:
    if create_path:
        target = tmp_path / create_path
        if Path(create_path).suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("safe\n", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
    _write_manifest(tmp_path, includes)

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert category in result.stdout


@pytest.mark.parametrize(
    "protected_exclude",
    ["app/core", "LICENSE", "README.md", "scripts/public_release_scan.py"],
)
def test_release_scanner_rejects_excluding_required_content(
    tmp_path: Path,
    protected_exclude: str,
) -> None:
    _write_manifest(tmp_path, [], extra_excludes=[protected_exclude])

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert "manifest_required_content" in result.stdout


def test_public_release_builder_creates_closed_verifiable_archive(tmp_path: Path) -> None:
    output = tmp_path / "public-release"

    result = subprocess.run(
        [sys.executable, str(BUILDER), "--root", str(PROJECT_ROOT), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "README.md").is_file()
    assert not (output / "README.public.md").exists()
    for required in (
        "LICENSE",
        "NOTICE",
        "Dockerfile",
        "compose.yaml",
        "scripts/public_release_scan.py",
        "scripts/run_local.py",
        "scripts/evaluate_public_portfolio.py",
        "app/core/config.py",
        "data/public_eval/portfolio_v1/manifest.json",
        "reports/public/public_synthetic_portfolio_v1.json",
        "tests/test_release_assets.py",
    ):
        assert (output / required).is_file(), required
    for forbidden in (
        ".superpowers",
        "data/local",
        "data/private",
        "docs/superpowers",
        "交接说明.md",
        "开发总提示词.md",
    ):
        assert not (output / forbidden).exists(), forbidden

    scan_result = _run_scan(output)
    assert scan_result.returncode == 0, scan_result.stdout

    hashes = json.loads((output / "SHA256SUMS.json").read_text(encoding="utf-8"))
    assert hashes["algorithm"] == "sha256"
    for relative, expected in hashes["files"].items():
        import hashlib

        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected

    readme = (output / "README.md").read_text(encoding="utf-8")
    for target in __import__("re").findall(r"\[[^\]]+\]\(([^)]+)\)", readme):
        if "://" in target or target.startswith("#"):
            continue
        resolved = (output / target.split("#", 1)[0]).resolve()
        assert output.resolve() in resolved.parents or resolved == output.resolve()
        assert resolved.exists(), target

    marker = output / "do-not-overwrite.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    second = subprocess.run(
        [sys.executable, str(BUILDER), "--root", str(PROJECT_ROOT), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode != 0
    assert marker.read_text(encoding="utf-8") == "preserve\n"


def test_public_release_builder_rejects_symlink_output(tmp_path: Path) -> None:
    builder_text = BUILDER.read_text(encoding="utf-8")
    assert "args.output.is_symlink()" in builder_text

    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    try:
        linked_output.symlink_to(real_output, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this test environment")

    result = subprocess.run(
        [sys.executable, str(BUILDER), "--root", str(PROJECT_ROOT), "--output", str(linked_output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not any(real_output.iterdir())


def test_symlink_component_check_is_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    linked = root / "linked"
    linked.mkdir(parents=True)
    candidate = linked / "file.txt"
    candidate.write_text("safe\n", encoding="utf-8")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == linked or original(path))

    assert public_release_scan._has_symlink_component(candidate, root)


def test_builder_scans_source_content_before_copying(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_manifest(source, [])
    secret = "ghp_" + "q" * 32
    (source / "LICENSE").write_text(f"token={secret}\n", encoding="utf-8")
    output = tmp_path / "output"

    result = subprocess.run(
        [sys.executable, str(BUILDER), "--root", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not output.exists() or not any(output.iterdir())
    assert secret not in result.stdout + result.stderr


def test_wrangler_local_state_is_ignored() -> None:
    """Local Wrangler account/cache state must never enter a release commit."""
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".wrangler/" in ignored.splitlines()


def test_submission_copy_is_evidence_bounded() -> None:
    """Competition copy must not turn unverified outcomes into claims."""
    submission = PROJECT_ROOT / "docs" / "competition" / "submission-fields-2026-08-29.md"
    assert submission.is_file()
    text = submission.read_text(encoding="utf-8").lower()
    ledger = json.loads(
        (PROJECT_ROOT / "docs" / "competition" / "evidence-ledger.json").read_text(
            encoding="utf-8"
        )
    )
    completed = {
        key for key, value in ledger.get("evidence", {}).items() if value.get("status") == "completed"
    }
    guarded_phrases = {
        "all modes passed": "live_map_realtime_smoke",
        "production ready": "live_map_realtime_smoke",
        "zero latency": "live_map_realtime_smoke",
        "accuracy improved": "live_minicpmo_app_e2e",
    }
    for phrase, evidence_key in guarded_phrases.items():
        if phrase in text:
            assert evidence_key in completed, phrase
