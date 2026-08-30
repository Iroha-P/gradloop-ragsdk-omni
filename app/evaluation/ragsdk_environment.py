from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from app.core.config import RagsdkSettings


class RagsdkEnvironmentTarget(StrEnum):
    OFFICIAL_ASCEND = "official_ascend"
    COMPATIBILITY_CPU = "compatibility_cpu"


class RagsdkEnvironmentStatus(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"


class RagsdkEnvironmentReport(BaseModel):
    report_version: Literal["ragsdk-environment-status-v1"] = "ragsdk-environment-status-v1"
    target: RagsdkEnvironmentTarget
    status: RagsdkEnvironmentStatus
    checks: dict[str, bool]
    missing: list[str] = Field(default_factory=list)
    runtime_claim: str = Field(pattern=r"^[a-z0-9_]+$")
    limitations: list[str]

    @model_validator(mode="after")
    def validate_status_and_missing(self) -> RagsdkEnvironmentReport:
        expected_missing = sorted(name for name, passed in self.checks.items() if not passed)
        if self.missing != expected_missing:
            raise ValueError("missing must exactly match failed environment checks")
        expected_status = (
            RagsdkEnvironmentStatus.READY
            if not expected_missing
            else RagsdkEnvironmentStatus.NOT_READY
        )
        if self.status != expected_status:
            raise ValueError("environment status must match failed checks")
        return self


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _command_available(name: str) -> bool:
    return shutil.which(name) is not None


def _path_exists(path: str) -> bool:
    return Path(path).exists()


def _endpoint_reachable(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((parsed.hostname, port), timeout=1.0):
            return True
    except (OSError, ValueError):
        return False


def _npu_healthcheck() -> bool:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = f"{result.stdout}\n{result.stderr}".lower()
    return result.returncode == 0 and "health" in output and "ok" in output


def inspect_ragsdk_environment(
    settings: RagsdkSettings,
    target: RagsdkEnvironmentTarget,
    *,
    platform_name: str | None = None,
    python_version: tuple[int, int] | None = None,
    module_available: Callable[[str], bool] = _module_available,
    command_available: Callable[[str], bool] = _command_available,
    path_exists: Callable[[str], bool] = _path_exists,
    environment: Mapping[str, str] | None = None,
    endpoint_reachable: Callable[[str], bool] = _endpoint_reachable,
    npu_healthcheck: Callable[[], bool] = _npu_healthcheck,
) -> RagsdkEnvironmentReport:
    current_platform = platform_name or platform.system()
    current_python = python_version or sys.version_info[:2]
    current_environment = environment if environment is not None else os.environ

    embedding_configured = bool(settings.embedding_url)
    reranker_configured = bool(settings.reranker_url)
    checks = {
        "linux": current_platform == "Linux",
        "python_3_11": tuple(current_python) == (3, 11),
        "mx_rag_installed": module_available("mx_rag"),
        "pymilvus_installed": module_available("pymilvus"),
        "embedding_configured": embedding_configured,
        "embedding_reachable": (
            endpoint_reachable(settings.embedding_url) if embedding_configured else False
        ),
        "reranker_configured": reranker_configured,
        "reranker_reachable": (
            endpoint_reachable(settings.reranker_url) if reranker_configured else False
        ),
    }
    if target == RagsdkEnvironmentTarget.OFFICIAL_ASCEND:
        npu_smi_available = command_available("npu-smi")
        checks.update(
            {
                "npu_smi_available": npu_smi_available,
                "npu_smi_healthy": npu_healthcheck() if npu_smi_available else False,
                "ascend_device_available": path_exists("/dev/davinci0")
                and path_exists("/dev/davinci_manager"),
                "cann_environment": any(
                    current_environment.get(name)
                    for name in ("ASCEND_HOME", "ASCEND_HOME_PATH", "ASCEND_VERSION")
                ),
                "index_sdk_environment": bool(
                    current_environment.get("MX_INDEX_INSTALL_PATH")
                ),
            }
        )

    missing = sorted(name for name, passed in checks.items() if not passed)
    status = (
        RagsdkEnvironmentStatus.READY
        if not missing
        else RagsdkEnvironmentStatus.NOT_READY
    )
    if target == RagsdkEnvironmentTarget.OFFICIAL_ASCEND:
        runtime_claim = f"official_ascend_{status.value}"
        limitations = [
            "Readiness is necessary but real indexing and retrieval must still complete.",
            "Only a completed four-mode matrix may be reported as official runtime evidence.",
        ]
    else:
        runtime_claim = (
            "compatibility_cpu_ready_not_official"
            if status == RagsdkEnvironmentStatus.READY
            else "compatibility_cpu_not_ready_not_official"
        )
        limitations = [
            "CPU compatibility evidence is not official Ascend runtime evidence.",
            "Readiness does not substitute for a completed four-mode matrix.",
        ]

    return RagsdkEnvironmentReport(
        target=target,
        status=status,
        checks=checks,
        missing=missing,
        runtime_claim=runtime_claim,
        limitations=limitations,
    )
