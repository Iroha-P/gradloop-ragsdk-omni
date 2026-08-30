from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore

from app.core.errors import CapacityExceededError


class ExpensiveOperationGate:
    """A fail-fast, process-local limit for costly synchronous operations."""

    def __init__(self, limit: int, *, retry_after_seconds: int):
        self._semaphore = BoundedSemaphore(limit)
        self.retry_after_seconds = retry_after_seconds

    def try_acquire(self) -> None:
        if not self._semaphore.acquire(blocking=False):
            raise CapacityExceededError(
                "service is busy; retry later",
                details={
                    "retryable": True,
                    "retry_after_seconds": self.retry_after_seconds,
                },
            )

    def release(self) -> None:
        self._semaphore.release()

    @contextmanager
    def lease(self) -> Iterator[None]:
        self.try_acquire()
        try:
            yield
        finally:
            self.release()
