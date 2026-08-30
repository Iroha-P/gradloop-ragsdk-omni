from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base error carrying a stable public code and safe details."""

    code = "application_error"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(AppError):
    code = "configuration_error"
    status_code = 500


class BackendUnavailableError(AppError):
    code = "backend_unavailable"
    status_code = 503


class CapacityExceededError(AppError):
    code = "capacity_exhausted"
    status_code = 429


class UnsafeInputError(AppError):
    code = "unsafe_input"
    status_code = 400


class PayloadTooLargeError(AppError):
    code = "payload_too_large"
    status_code = 413
