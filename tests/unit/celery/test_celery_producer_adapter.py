"""Tests for the concrete Celery direct-publication adapter seam.

The legacy ``typed_send``/``dispatch`` direct-entry facade was removed: every
publication goes through :class:`app.shared.jobs.bus.JobBus`, whose
``enqueue`` reaches the broker through this adapter. These tests pin the
transport behavior that survives the consolidation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kombu.exceptions import OperationalError

from app.connections import celery_producer
from app.shared.jobs import EnqueueOptions, prepare_job
from app.shared.jobs.names import DOCUMENTS_INGEST
from app.shared.jobs.registry import TaskPayloadValidationError
from app.utils import ServiceUnavailableException

_PAYLOAD = {
    "document_id": "doc-1",
    "user_id": "user-1",
    "filename": "contract.pdf",
    "content_type": "application/pdf",
    "object_uri": "s3://bucket/contract.pdf",
}


class _Broker:
    def __init__(self, result: str = "broker-task-1") -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def send_task(
        self, task_name: str, *, kwargs: dict[str, object], **options: object
    ) -> SimpleNamespace:
        self.calls.append((task_name, kwargs, options))
        return SimpleNamespace(id=self.result)


@pytest.mark.asyncio
async def test_adapter_publishes_json_payload_and_catalog_routes() -> None:
    broker = _Broker()
    adapter = celery_producer.DirectPublicationAdapter(broker)  # type: ignore[arg-type]
    job = prepare_job(DOCUMENTS_INGEST, _PAYLOAD)

    task_id = await adapter.publish(
        job,
        options=EnqueueOptions(expires=30, countdown=2),
    )

    assert task_id == "broker-task-1"
    assert broker.calls == [(DOCUMENTS_INGEST, _PAYLOAD, {"expires": 30, "countdown": 2})]
    assert "queue" not in broker.calls[0][2]


@pytest.mark.asyncio
async def test_invalid_job_is_rejected_before_broker_call() -> None:
    broker = _Broker()
    adapter = celery_producer.DirectPublicationAdapter(broker)  # type: ignore[arg-type]
    job = prepare_job(DOCUMENTS_INGEST, _PAYLOAD)
    invalid = job.model_copy(update={"payload": {"document_id": "wrong"}})

    with pytest.raises(TaskPayloadValidationError):
        await adapter.publish(invalid, options=EnqueueOptions())

    assert broker.calls == []


@pytest.mark.asyncio
async def test_broker_error_is_mapped_with_cause_and_note() -> None:
    class FailingBroker(_Broker):
        def send_task(self, task_name: str, *, kwargs: dict[str, object], **options: object):
            message = "broker unavailable"
            raise OperationalError(message)

    broker = FailingBroker()
    adapter = celery_producer.DirectPublicationAdapter(broker)  # type: ignore[arg-type]
    job = prepare_job(DOCUMENTS_INGEST, _PAYLOAD)

    with pytest.raises(ServiceUnavailableException) as exc_info:
        await adapter.publish(job, options=EnqueueOptions())

    assert isinstance(exc_info.value.__cause__, OperationalError)
    assert "operation=celery_dispatch" in exc_info.value.__cause__.__notes__[0]
