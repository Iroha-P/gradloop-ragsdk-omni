from __future__ import annotations

from pathlib import Path

import pytest

from scripts.competition.build_public_site import build_public_site


def test_pages_builder_writes_only_public_runtime_values(tmp_path: Path) -> None:
    output = tmp_path / "site"

    build_public_site(output, proxy_url="wss://worker.example/ws", project_url="https://repo.example")

    config = (output / "runtime-config.js").read_text(encoding="utf-8")
    assert "wss://worker.example/ws" in config
    assert '"deployment_mode":"public_pages"' in config
    assert '"local_backend":{"enabled":false}' in config
    assert all(term not in config.casefold() for term in ("api_key", "token", "secret"))
    assert (output / "index.html").is_file()
    assert (output / "public-demo-fixtures.mjs").is_file()
    assert (output / "ui-assets" / "gradloop-icon.png").is_file()
    assert (output / "realtime" / "session.mjs").is_file()
    assert (output / ".nojekyll").is_file()
    assert "repo.example" in (output / "README.md").read_text(encoding="utf-8")


def test_pages_builder_rejects_non_wss_and_non_empty_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="wss"):
        build_public_site(tmp_path / "site", proxy_url="ws://worker.example/ws", project_url="https://repo.example")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_public_site(output, proxy_url="wss://worker.example/ws", project_url="https://repo.example")
