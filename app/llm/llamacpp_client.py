from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from app.core.config import LlamaCppSettings
from app.core.errors import BackendUnavailableError, ConfigurationError
from app.llm.base import LlmHealth, LlmResponse


class LlamaCppClient:
    """Loopback-only llama.cpp client using its OpenAI-compatible chat endpoint."""

    name = "llama.cpp"

    def __init__(self, settings: LlamaCppSettings):
        parsed = urlparse(settings.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigurationError("llama.cpp base URL must use an HTTP loopback address")
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")

    def _json_request(
        self,
        path: str,
        *,
        payload: dict | None = None,
        timeout: int | None = None,
    ) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout or self.settings.timeout_seconds,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise BackendUnavailableError(
                "Local llama.cpp service is unavailable",
                details={"backend": self.name, "model": self.settings.model},
            ) from exc
        if not isinstance(result, dict):
            raise BackendUnavailableError("llama.cpp returned an invalid response")
        return result

    def healthcheck(self) -> LlmHealth:
        try:
            result = self._json_request("/health", timeout=3)
        except BackendUnavailableError as exc:
            return LlmHealth(
                backend=self.name,
                ready=False,
                model=self.settings.model,
                message=exc.message,
            )
        ready = result.get("status") == "ok"
        return LlmHealth(
            backend=self.name,
            ready=ready,
            model=self.settings.model,
            message="model ready" if ready else "model is still loading",
        )

    def chat(self, *, system: str, user: str) -> LlmResponse:
        started = time.perf_counter()
        result = self._json_request(
            "/v1/chat/completions",
            payload={
                "model": self.settings.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"/no_think\n{user}"},
                ],
                "stream": False,
                "temperature": self.settings.temperature,
                "top_p": 0.8,
                "max_tokens": 900,
            },
        )
        choices = result.get("choices")
        message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        content = str(message.get("content", "")).strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content:
            raise BackendUnavailableError("llama.cpp returned an empty answer")
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return LlmResponse(
            backend=self.name,
            model=self.settings.model,
            content=content,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
