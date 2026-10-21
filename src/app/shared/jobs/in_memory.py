"""In-memory JobBus adapters for public-interface tests.

These adapters model publication and transaction visibility only.  They do not
attempt to emulate Celery scheduling, retries, or a PostgreSQL implementation.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, override

from app.shared.jobs.bus import (
    JobBus,
    _validated_job,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from app.shared.jobs.bus import AggregateRef, TransactionScope, TypedJob
    from app.shared.jobs.payloads import CeleryTaskPayload


type FailureFactory = Callable[[], Exception]


def _raise_injected(failure: Exception | FailureFactory | None) -> None:
    if failure is None:
        return
    if isinstance(failure, Exception):
        raise failure
    raise failure()


class InMemoryTransaction(AbstractAsyncContextManager["InMemoryTransaction"]):
    """A tiny transaction handle whose staged jobs become visible on commit."""

    def __init__(self, adapter: InMemoryTransactionalAdapter) -> None:
        self._adapter = adapter
        self._staged: list[tuple[str, TypedJob[CeleryTaskPayload], AggregateRef]] = []
        self._closed = False
        self._committed = False

    def _stage(
        self,
        event_id: str,
        job: TypedJob[CeleryTaskPayload],
        aggregate: AggregateRef,
    ) -> None:
        if self._closed:
            message = "Cannot stage a job after the transaction is closed"
            raise RuntimeError(message)
        self._staged.append((event_id, job, aggregate))

    async def execute(
        self,
        _statement: object,
        _parameters: Mapping[str, object] | None = None,
        **_kwargs: object,
    ) -> object:
        """Satisfy ``TransactionScope`` without pretending to execute SQL."""
        message = "In-memory transactions do not execute SQL"
        raise NotImplementedError(message)

    async def commit(self) -> None:
        if self._closed:
            if self._committed:
                return
            message = "Cannot commit a rolled-back transaction"
            raise RuntimeError(message)
        self._adapter._commit(self)
        self._closed = True
        self._committed = True

    async def rollback(self) -> None:
        if self._closed:
            return
        self._staged.clear()
        self._closed = True

    @override
    async def __aexit__(self, *_args: object) -> None:
        if _args[1] is None:
            await self.commit()
        else:
            await self.rollback()


class InMemoryDirectAdapter:
    """Record direct publications while preserving the adapter contract."""

    def __init__(
        self,
        *,
        failure: Exception | FailureFactory | None = None,
    ) -> None:
        self.failure = failure
        self.published: list[TypedJob[CeleryTaskPayload]] = []
        self._next_id = 0

    async def publish(self, job: TypedJob[CeleryTaskPayload], **_kwargs: object) -> str:
        _raise_injected(self.failure)
        validated = _validated_job(job)
        self.published.append(validated)
        self._next_id += 1
        return f"broker-{self._next_id}"


class InMemoryTransactionalAdapter:
    """Stage jobs on fake transactions and expose them only after commit."""

    def __init__(
        self,
        *,
        failure: Exception | FailureFactory | None = None,
    ) -> None:
        self.failure = failure
        self.committed: list[tuple[TypedJob[CeleryTaskPayload], AggregateRef]] = []
        self._next_id = 0

    async def stage(
        self,
        tx: TransactionScope,
        job: TypedJob[CeleryTaskPayload],
        *,
        aggregate: AggregateRef,
    ) -> str:
        _raise_injected(self.failure)
        if not isinstance(tx, InMemoryTransaction):
            message = "InMemoryTransactionalAdapter requires InMemoryTransaction"
            raise TypeError(message)
        validated = _validated_job(job)
        self._next_id += 1
        event_id = f"outbox-{self._next_id}"
        tx._stage(event_id, validated, aggregate)
        return event_id

    def _commit(self, tx: InMemoryTransaction) -> None:
        self.committed.extend((job, aggregate) for _, job, aggregate in tx._staged)


class InMemoryJobBus(JobBus):
    """JobBus backed by the two in-memory adapters."""

    def __init__(
        self,
        *,
        direct_failure: Exception | FailureFactory | None = None,
        transactional_failure: Exception | FailureFactory | None = None,
    ) -> None:
        self.direct = InMemoryDirectAdapter(failure=direct_failure)
        self.transactional = InMemoryTransactionalAdapter(failure=transactional_failure)
        super().__init__(self.direct, self.transactional)

    def transaction(self) -> InMemoryTransaction:
        """Create a fake transaction suitable for ``async with`` tests."""
        return InMemoryTransaction(self.transactional)


__all__ = [
    "InMemoryDirectAdapter",
    "InMemoryJobBus",
    "InMemoryTransaction",
    "InMemoryTransactionalAdapter",
]
