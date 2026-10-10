"""Celery task execution bases; no feature or producer imports."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, override

from celery import Task

from app.config import get_settings
from app.connections.redis import create_redis_client
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from app.shared.jobs.payloads import CeleryTaskPayload
    from app.utils.cache import Redis


import os

from app.shared.jobs.registry import CeleryTaskRegistry
from app.worker.observability import _celery_meters
from app.worker.reliability import (
    acquire_idempotency_lock,
    mark_idempotency_completed,
    mark_idempotency_failed_permanently,
    release_idempotency_processing_lock,
    run_with_circuit_breaker,
)

settings = get_settings()


class ResilientTask(Task):
    """Base Celery task with retries, observability, and reliability helpers."""

    abstract = True
    autoretry_for: ClassVar[tuple[type[BaseException], ...]] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    retry_backoff: ClassVar[bool] = True
    retry_backoff_max: ClassVar[int] = settings.CELERY_RETRY_BACKOFF_MAX
    retry_jitter: ClassVar[bool] = True
    retry_kwargs: ClassVar[dict[str, int]] = {"max_retries": settings.CELERY_RETRY_MAX_RETRIES}

    _redis_client: ClassVar[Redis | None] = None
    _redis_pid: ClassVar[int | None] = None

    @staticmethod
    def get_redis_client() -> Redis:
        if ResilientTask._redis_pid != os.getpid():
            ResilientTask._redis_client = None
            ResilientTask._redis_pid = os.getpid()
        if ResilientTask._redis_client is None:
            ResilientTask._redis_client = create_redis_client(settings.REDIS_URL)
        return ResilientTask._redis_client

    def acquire_idempotency_lock(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return acquire_idempotency_lock(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def mark_idempotency_completed(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        mark_idempotency_completed(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def mark_idempotency_failed_permanently(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        mark_idempotency_failed_permanently(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def release_idempotency_processing_lock(self, idempotency_key: str) -> None:
        release_idempotency_processing_lock(self.get_redis_client(), idempotency_key)

    def run_with_circuit_breaker(
        self,
        name: str,
        operation,
    ) -> Any:
        return run_with_circuit_breaker(
            self.get_redis_client(),
            name,
            operation,
            failure_threshold=settings.CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout_seconds=settings.CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        )

    @override
    def on_retry(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        _ = (args, kwargs, einfo)
        _celery_meters()[2].add(1, {"task_name": str(self.name)})
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
            error=str(exc),
        ).warning("Task scheduled for retry")

    @override
    def on_failure(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        _ = (args, kwargs, einfo)
        completed, duration, retries = _celery_meters()
        attrs = {"task_name": str(self.name), "status": "failure"}
        completed.add(1, attrs)
        duration.record(time.time() - self.request.started or time.time(), attrs)
        retries.add(1, {"task_name": str(self.name)})
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
            error=str(exc),
        ).error("Task failed")

    @override
    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
    ) -> None:
        _ = (retval, args, kwargs)
        completed, duration, _ = _celery_meters()
        attrs = {"task_name": str(self.name), "status": "success"}
        completed.add(1, attrs)
        duration.record(time.time() - self.request.started or time.time(), attrs)
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).info("Task completed successfully")


class TypedCeleryTask(Task):
    """Base Celery task that validates kwargs against a registered Pydantic model."""

    abstract = True
    _validated_payload: CeleryTaskPayload | None = None

    @property
    def validated_payload(self) -> CeleryTaskPayload:
        if self._validated_payload is None:
            msg = "validated_payload accessed before task execution"
            raise RuntimeError(msg)
        return self._validated_payload

    @override
    def before_start(self, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        super().before_start(task_id, args, kwargs)
        task_name = self.name or ""
        self._validated_payload = CeleryTaskRegistry.validate(task_name, kwargs)

    @override
    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        super().on_success(retval, task_id, args, kwargs)
        if self._validated_payload is not None:
            logger.bind(
                task=self.name, task_id=task_id, payload_type=type(self._validated_payload).__name__
            ).info("Typed task completed")
