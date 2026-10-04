"""Build the secret-free static Pages demo from a narrow UI allowlist."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _validate_urls(proxy_url: str, project_url: str) -> None:
    proxy = urlparse(proxy_url)
    if proxy.scheme != "wss" or not proxy.netloc or proxy.username or proxy.password:
        raise ValueError("proxy_url must be a credential-free wss URL")
    project = urlparse(project_url)
    if project.scheme not in {"https", "http"} or not project.netloc:
        raise ValueError("project_url must be an http(s) URL")


def build_public_site(
    output: Path,
    *,
    proxy_url: str,
    project_url: str,
    source_root: Path = PROJECT_ROOT,
) -> Path:
    _validate_urls(proxy_url, project_url)
    output = Path(output)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("output must be absent or empty")
    source_root = Path(source_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "ui-assets").mkdir(parents=True, exist_ok=True)
    (output / "realtime").mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_root / "app/ui/index.html", output / "index.html")
    shutil.copy2(source_root / "app/ui/public-demo-fixtures.mjs", output / "public-demo-fixtures.mjs")
    shutil.copy2(source_root / "app/ui/assets/gradloop-icon.png", output / "ui-assets/gradloop-icon.png")
    for module in ("protocol.mjs", "session.mjs", "audio.mjs"):
        shutil.copy2(source_root / "app/ui/realtime" / module, output / "realtime" / module)
    document_root = source_root / "app/ui/documents"
    (output / "documents/vendor").mkdir(parents=True, exist_ok=True)
    for module in ("client.mjs", "policy.mjs", "worker.mjs"):
        shutil.copy2(document_root / module, output / "documents" / module)
    for asset in ("pdf.mjs", "pdf.worker.mjs", "word.mjs", "THIRD_PARTY_NOTICES.md", "provenance.json"):
        shutil.copy2(document_root / "vendor" / asset, output / "documents/vendor" / asset)

    index = (output / "index.html").read_text(encoding="utf-8")
    index = index.replace(
        '<script type="module">',
        '<script src="./runtime-config.js"></script>\n  <script type="module">',
        1,
    )
    (output / "index.html").write_text(index, encoding="utf-8")
    runtime = {
        "deployment_mode": "public_pages",
        "local_backend": {"enabled": False},
        "realtime": {"enabled": True, "proxy_url": proxy_url, "session_seconds": 300},
        "project_url": project_url,
        "privacy": {"public_only": True},
    }
    (output / "runtime-config.js").write_text(
        "window.__GRADLOOP_RUNTIME_CONFIG = "
        + json.dumps(runtime, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# GradLoop RAGSDK Omni\n\n"
        "静态 Pages 演示仅展示公开合成案例与受控 MAP 实时入口。完整 Agent/RAG 工作流保留在本地 FastAPI。\n\n"
        f"项目说明：{project_url}\n\n"
        "页面不包含 API Key、个人资料、私有语料或运行日志。实时服务凭据只应配置在代理的 Secret 存储中。\n",
        encoding="utf-8",
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the public GradLoop Pages site")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy-url", required=True)
    parser.add_argument("--project-url", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_public_site(args.output, proxy_url=args.proxy_url, project_url=args.project_url)
    except (FileExistsError, OSError, ValueError) as error:
        print(f"public Pages build failed: {error}")
        return 1
    print(f"public Pages build passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
