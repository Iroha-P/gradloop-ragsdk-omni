from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from ipaddress import ip_address
from urllib.parse import urlparse

from app.core.config import MiniCpmoSettings
from app.core.errors import BackendUnavailableError, ConfigurationError
from app.multimodal.base import MediaModality, MultimodalHealth, OmniRequest, OmniResponse

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _is_loopback(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Reject redirects so an allowlisted endpoint cannot redirect to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MiniCPMOClient:
    """Strict HTTP client for the separate Ascend MiniCPM-o inference service."""

    name = "minicpmo-ascend"

    def __init__(self, settings: MiniCpmoSettings, *, bearer_token: str | None = None):
        parsed = urlparse(settings.base_url)
        hostname = parsed.hostname or ""
        if (
            parsed.scheme not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigurationError("MiniCPM-o service URL is invalid")
        if parsed.scheme == "http" and not _is_loopback(hostname):
            raise ConfigurationError("Remote MiniCPM-o service must use HTTPS")
        allowed_hosts = {item.casefold() for item in settings.allowed_hosts}
        if hostname.casefold() not in allowed_hosts:
            raise ConfigurationError("MiniCPM-o service host is not allowlisted")
        if not _is_loopback(hostname):
            try:
                if ip_address(hostname).is_multicast:
                    raise ConfigurationError("MiniCPM-o service host is not allowed")
            except ValueError:
                pass
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")
        self._bearer_token = bearer_token.strip() if bearer_token else None
        self._opener = urllib.request.build_opener(_RejectRedirects())

    def _json_request(
        self,
        path: str,
        *,
        payload: dict | None = None,
        timeout: int | None = None,
    ) -> dict:
        body = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if payload
            else None
        )
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with self._opener.open(
                request,
                timeout=timeout or self.settings.timeout_seconds,
            ) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise ValueError("response is too large")
            result = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, urllib.error.URLError) as exc:
            raise BackendUnavailableError(
                "MiniCPM-o service is unavailable",
                details={"backend": self.name, "model": self.settings.model},
            ) from exc
        if not isinstance(result, dict):
            raise BackendUnavailableError("MiniCPM-o service returned an invalid response")
        return result

    def healthcheck(self) -> MultimodalHealth:
        try:
            result = self._json_request("/health", timeout=3)
            ready = result.get("ready") is True or result.get("status") == "ok"
            modalities = result.get("modalities", [])
            supported_values = {item.value for item in MediaModality}
            supported_modalities = (
                [item for item in modalities if item in supported_values]
                if isinstance(modalities, list)
                else []
            )
            return MultimodalHealth(
                backend=self.name,
                ready=ready,
                model=str(result.get("model") or self.settings.model),
                modalities=supported_modalities,
                message="model ready" if ready else "model is still loading",
            )
        except (BackendUnavailableError, ValueError):
            return MultimodalHealth(
                backend=self.name,
                ready=False,
                model=self.settings.model,
                modalities=[],
                message="model service unavailable",
            )

    def generate(self, request: OmniRequest) -> OmniResponse:
        started = time.perf_counter()
        attachments = [
            {
                "asset_id": item.asset_id,
                "media_type": item.media_type,
                "content_base64": base64.b64encode(item.content).decode("ascii"),
            }
            for item in request.attachments
        ]
        result = self._json_request(
            "/v1/omni/generate",
            payload={
                "model": self.settings.model,
                "prompt": request.prompt,
                "attachments": attachments,
                "max_tokens": request.max_tokens,
                "stream": False,
            },
        )
        content = str(result.get("content", "")).strip()
        if not content:
            raise BackendUnavailableError("MiniCPM-o service returned an empty answer")
        modalities = list(dict.fromkeys(item.modality for item in request.attachments))
        return OmniResponse(
            backend=self.name,
            model=str(result.get("model") or self.settings.model),
            content=content,
            modalities=modalities,
            attachment_count=len(request.attachments),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
