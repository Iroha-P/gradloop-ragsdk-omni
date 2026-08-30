from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.competition.map_setup_core import (
    OneKeyDefaults,
    build_public_plan,
    load_defaults,
    sanitize_public_summary,
    validate_defaults,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "deploy" / "map-realtime-worker" / "wrangler.one-key.jsonc"
SETUP_SCRIPT = ROOT / "scripts" / "competition" / "setup_map_demo.ps1"
GUIDE = ROOT / "docs" / "competition" / "one-key-map-setup.md"
PUBLIC_README = ROOT / "README.public.md"
DEPLOYMENT = ROOT / "DEPLOYMENT.md"


def test_committed_defaults_are_public_and_deterministic() -> None:
    defaults = load_defaults(CONFIG)
    first = build_public_plan(ROOT, ROOT / "dist" / "validation", defaults)
    second = build_public_plan(ROOT, ROOT / "dist" / "validation", defaults)
    assert first == second
    assert first["actions"] == [
        "preflight",
        "build_validation_pages",
        "cloudflare_login",
        "deploy_worker",
        "set_worker_secret",
        "build_final_pages",
        "deploy_pages",
        "health",
        "chat_smoke",
    ]
    assert "MAP_API_KEY" not in json.dumps(first)


@pytest.mark.parametrize(
    "field,value",
    [
        ("map_realtime_url", "ws://minicpmo45.modelbest.cn/v1/realtime"),
        ("map_realtime_url", "wss://user:pass@example.com/v1/realtime"),
        ("map_realtime_url", "wss://example.com/v1/realtime?mode=chat"),
        ("pages_origin", "https://example.com/app"),
        ("pages_origin", "http://example.com"),
    ],
)
def test_validate_defaults_rejects_unsafe_urls(field: str, value: str) -> None:
    defaults = load_defaults(CONFIG)
    updated = {**defaults.__dict__, field: value}
    with pytest.raises(ValueError):
        validate_defaults(OneKeyDefaults(**updated))


def test_jsonc_loader_rejects_unexpected_keys(tmp_path: Path) -> None:
    config = tmp_path / "config.jsonc"
    config.write_text(
        '{\n  // full-line comments are allowed\n  "name": "x",\n  "main": "src/index.mjs",\n'
        '  "compatibility_date": "2026-08-28",\n  "vars": {},\n'
        '  "secrets": {"required": ["MAP_API_KEY"]}\n}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_defaults(config)


def test_summary_redaction_rejects_content_credentials_and_paths() -> None:
    safe = sanitize_public_summary(
        {
            "schema_version": "one-key-map-setup.v1",
            "worker_url": "https://worker.example.workers.dev",
            "pages_url": "https://pages.example.pages.dev",
            "health_status": 200,
            "chat_smoke": "passed",
            "started_at": "2026-08-28T00:00:00Z",
            "finished_at": "2026-08-28T00:01:00Z",
            "close_reason": "completed",
        }
    )
    assert safe["chat_smoke"] == "passed"
    with pytest.raises(ValueError):
        sanitize_public_summary({"worker_url": "https://worker.example.dev?mode=chat"})
    with pytest.raises(ValueError):
        sanitize_public_summary({"headers": {"Authorization": "redacted"}})
    with pytest.raises(ValueError):
        private_path = "D" + ":\\private\\recording.wav"
        sanitize_public_summary({"close_reason": private_path})


def test_powershell_wizard_contract_is_secret_safe_and_gated() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    invoke_checked = script.split("function Invoke-Checked", 1)[1].split(
        "function ConvertTo-ProcessArgument", 1
    )[0]
    assert "[switch]$DryRun" in script
    assert "[switch]$SkipFullTests" in script
    assert "[string]$ApiKey" not in script
    assert "[string]$Secret" not in script
    assert "[string]$Token" not in script
    assert "Read-Host '本地检查已通过。若确认外部部署，请输入 DEPLOY" in script
    assert "if ($confirmation -cne 'DEPLOY')" in script
    assert "wrangler', 'secret', 'put', 'MAP_API_KEY'" in script
    assert "Write-Host $apiKey" not in script
    assert ".dev.vars" not in script
    assert "wrangler.toml" not in script
    assert "git clean" not in script
    assert "git push" not in script
    assert "& $FilePath @Arguments | Out-Host" in invoke_checked


def test_powershell_wizard_runs_all_local_gates_before_login() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    preflight = script.index("local preflight")
    login = script.index("Cloudflare browser authorization")
    for gate in (
        "focused Python tests",
        "Python lint",
        "browser module tests",
        "Worker module tests",
        "repository public release scan",
        "Git whitespace check",
        "build validation-only Pages site",
        "re-scan repository release allowlist after validation build",
    ):
        assert preflight < script.index(gate) < login


def test_powershell_wizard_disables_npx_install_prompts() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    assert "$env:npm_config_yes = 'true'" in script


def test_worker_secret_command_inherits_the_real_terminal() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    interactive = script.split("function Invoke-InteractiveChecked", 1)[1].split(
        "function ConvertTo-ProcessArgument", 1
    )[0]
    secret_call = script.split(
        "Write-Host '请在 Wrangler 隐藏输入框中粘贴 API Key", 1
    )[1].split("$finalOutput", 1)[0]
    assert "& $FilePath @Arguments" in interactive
    assert "| Out-Host" not in interactive
    assert "Invoke-InteractiveChecked" in secret_call


def test_powershell_wizard_resolves_npx_to_an_absolute_executable() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    assert "(Get-Command 'npx.cmd' -ErrorAction Stop).Source" in script
    assert "(Get-Command 'npx' -ErrorAction Stop).Source" in script


def test_captured_windows_cmd_tools_are_hosted_by_comspec() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    invoke_captured = script.split("function Invoke-Captured", 1)[1].split(
        "function Get-SafeOutput", 1
    )[0]
    assert "$info.FileName = $env:ComSpec" in invoke_captured
    assert "'/d /c call '" in invoke_captured


def test_windows_detection_does_not_depend_on_the_os_environment_variable() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    assert "$script:IsWindows = [System.Environment]::OSVersion.Platform" in script
    assert "$env:OS -eq 'Windows_NT'" not in script


def test_beginner_guide_has_one_command_three_actions_and_boundaries() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    assert "powershell -ExecutionPolicy Bypass -File scripts/competition/setup_map_demo.ps1" in guide
    assert "## 你只需要做三件事" in guide
    assert guide.count("\n1. ") == 1
    assert guide.count("\n2. ") == 1
    assert guide.count("\n3. ") == 1
    markers = (
        "DEPLOY",
        "隐藏输入框",
        "不要把 Key 发给任何人",
        "not_run",
        "失败恢复",
        "403 origin_not_allowed",
        "503 proxy_not_configured",
        "502 upstream_unavailable",
    )
    for marker in markers:
        assert marker in guide
    assert "MAP_API_KEY=" not in guide
    assert "account id" not in guide.casefold()


def test_public_docs_link_to_one_key_guide() -> None:
    assert "docs/competition/one-key-map-setup.md" in PUBLIC_README.read_text(encoding="utf-8")
    assert "docs/competition/one-key-map-setup.md" in DEPLOYMENT.read_text(encoding="utf-8")
