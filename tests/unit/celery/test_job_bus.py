"""Public-interface tests for the caller-first JobBus seam."""

from __future__ import annotations

import subprocess  # noqa: S404 — trusted local interpreter smoke test
import sys
from typing import TYPE_CHECKING, assert_type, cast

import pytest
from pydantic import ValidationError

from app.shared.jobs import (
    AggregateRef,
    DirectEnqueueReceipt,
    DurableEnqueueReceipt,
    EnqueueOptions,
    InMemoryJobBus,
    JobBus,
    TypedJob,
    prepare_job,
    require_direct_receipt,
)
from app.shared.jobs.catalog import JOB_CATALOG, get_job
from app.shared.jobs.names import CRAWLER_CRAWL, DOCUMENTS_INGEST
from app.shared.jobs.registry import TaskPayloadValidationError, UnregisteredTaskError
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from app.shared.jobs.catalog import JobDefinition
    from app.shared.jobs.payloads import CeleryTaskPayload, CrawlerJobPayload
else:
    from app.shared.jobs.payloads import CrawlerJobPayload

_CRAWL_PAYLOAD = {
    "crawl_id": "crawl-1",
    "request_payload": {"url": "https://example.com"},
}
_DOCUMENT_PAYLOAD = {
    "document_id": "doc-1",
    "user_id": "user-1",
    "filename": "contract.pdf",
    "content_type": "application/pdf",
    "object_uri": "s3://documents/contract.pdf",
}
_AGGREGATE = AggregateRef(aggregate_type="crawl", aggregate_id="crawl-1")


def test_catalog_definition_is_the_only_job_construction_seam() -> None:
    definition = get_job(CRAWLER_CRAWL)
    assert definition is not None

    job = definition.prepare(_CRAWL_PAYLOAD)
    assert isinstance(job, TypedJob)
    assert job.task_name == CRAWLER_CRAWL
    assert job.queue == "crawler"
    assert job.wire_payload() == _CRAWL_PAYLOAD
    assert definition.make(_CRAWL_PAYLOAD) == job


def test_generic_prepare_has_static_and_runtime_payload_precision() -> None:
    definition = cast(
        "JobDefinition[CrawlerJobPayload]",
        JOB_CATALOG[CRAWLER_CRAWL],
    )
    job = definition.prepare(_CRAWL_PAYLOAD)

    assert_type(job, TypedJob[CrawlerJobPayload])
    assert type(job).model_fields["payload"].annotation is CrawlerJobPayload
    assert type(job).model_json_schema()["properties"]["payload"] == {
        "$ref": "#/$defs/CrawlerJobPayload"
    }


def test_stable_json_serialization_uses_the_catalog_payload_model() -> None:
    job = JOB_CATALOG[DOCUMENTS_INGEST].prepare(_DOCUMENT_PAYLOAD)

    assert job.wire_payload() == _DOCUMENT_PAYLOAD
    assert JOB_CATALOG[DOCUMENTS_INGEST].payload_model.model_validate(job.wire_payload()) == (
        job.payload
    )
    assert job.wire_json() == job.payload.model_dump_json()


@pytest.mark.asyncio
async def test_direct_and_durable_receipts_are_discriminated_and_ordered() -> None:
    bus = InMemoryJobBus()
    crawl = JOB_CATALOG[CRAWLER_CRAWL].prepare(_CRAWL_PAYLOAD)
    document = JOB_CATALOG[DOCUMENTS_INGEST].prepare(_DOCUMENT_PAYLOAD)

    direct = await bus.enqueue(crawl)
    async with bus.transaction() as tx:
        first = await bus.in_transaction(tx).enqueue(crawl, aggregate=_AGGREGATE)
        second = await bus.in_transaction(tx).enqueue(document, aggregate=_AGGREGATE)
        assert bus.transactional.committed == []

    assert isinstance(direct, DirectEnqueueReceipt)
    assert isinstance(first, DurableEnqueueReceipt)
    assert isinstance(second, DurableEnqueueReceipt)
    assert direct.kind == "direct"
    assert first.kind == second.kind == "durable"
    assert direct.task_id != first.event_id
    assert [job.task_name for job in bus.direct.published] == [CRAWLER_CRAWL]
    assert [job.task_name for job, _ in bus.transactional.committed] == [
        CRAWLER_CRAWL,
        DOCUMENTS_INGEST,
    ]


@pytest.mark.asyncio
async def test_transaction_rollback_discards_staged_jobs() -> None:
    bus = InMemoryJobBus()
    job = JOB_CATALOG[CRAWLER_CRAWL].prepare(_CRAWL_PAYLOAD)

    tx = bus.transaction()
    await bus.in_transaction(tx).enqueue(job, aggregate=_AGGREGATE)
    await tx.rollback()

    assert bus.transactional.committed == []


@pytest.mark.asyncio
async def test_invalid_job_is_rejected_before_direct_adapter_side_effect() -> None:
    bus = InMemoryJobBus()
    invalid = TypedJob(
        definition=JOB_CATALOG[DOCUMENTS_INGEST],
        payload=JOB_CATALOG[CRAWLER_CRAWL].payload_model(**_CRAWL_PAYLOAD),
    )

    with pytest.raises(TaskPayloadValidationError):
        await bus.enqueue(invalid)

    assert bus.direct.published == []


@pytest.mark.asyncio
async def test_invalid_job_is_rejected_before_transactional_adapter_side_effect() -> None:
    bus = InMemoryJobBus()
    invalid = TypedJob(
        definition=JOB_CATALOG[DOCUMENTS_INGEST],
        payload=JOB_CATALOG[CRAWLER_CRAWL].payload_model(**_CRAWL_PAYLOAD),
    )
    tx = bus.transaction()

    with pytest.raises(TaskPayloadValidationError):
        await bus.in_transaction(tx).enqueue(invalid, aggregate=_AGGREGATE)

    assert tx._staged == []
    await tx.rollback()


def test_unknown_name_is_rejected_before_job_construction() -> None:
    with pytest.raises(UnregisteredTaskError):
        prepare_job("tasks.unknown", {})


def test_queue_is_catalog_owned_and_caller_options_are_narrow() -> None:
    with pytest.raises(ValidationError):
        EnqueueOptions(queue="crawler")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "receipt",
    [
        "",
        "   ",
        object(),
        DurableEnqueueReceipt(event_id="event-1"),
        DirectEnqueueReceipt.model_construct(kind="durable", task_id="task-1"),
    ],
)
def test_direct_receipt_gate_rejects_empty_malformed_or_durable_ids(receipt: object) -> None:
    with pytest.raises(ServiceUnavailableException) as exc_info:
        require_direct_receipt(receipt, operation="test_receipt_gate")

    assert "operation=test_receipt_gate" in exc_info.value.__notes__
    assert "receipt_type=" in " ".join(exc_info.value.__notes__)


def test_direct_receipt_gate_accepts_legacy_and_typed_direct_ids() -> None:
    assert require_direct_receipt("task-1", operation="test") == "task-1"
    assert (
        require_direct_receipt(DirectEnqueueReceipt(task_id="task-2"), operation="test") == "task-2"
    )


@pytest.mark.asyncio
async def test_protocol_adapters_are_substitutable() -> None:
    class DirectSpy:
        async def publish(
            self,
            job: TypedJob[CeleryTaskPayload],
            *,
            options: EnqueueOptions,
        ) -> str:
            assert job.task_name == CRAWLER_CRAWL
            assert options.countdown == 2
            return "spy-task"

    class StagingSpy:
        async def stage(
            self,
            tx: object,
            job: TypedJob[CeleryTaskPayload],
            *,
            aggregate: AggregateRef,
        ) -> str:
            assert tx == "transaction"
            assert job.task_name == CRAWLER_CRAWL
            assert aggregate == _AGGREGATE
            return "spy-event"

    bus = JobBus(DirectSpy(), StagingSpy())
    job = JOB_CATALOG[CRAWLER_CRAWL].prepare(_CRAWL_PAYLOAD)

    direct = await bus.enqueue(job, EnqueueOptions(countdown=2))
    durable = await bus.in_transaction("transaction").enqueue(job, aggregate=_AGGREGATE)

    assert direct.task_id == "spy-task"
    assert durable.event_id == "spy-event"


@pytest.mark.asyncio
async def test_adapter_failures_are_injected_without_feature_results() -> None:
    job = JOB_CATALOG[CRAWLER_CRAWL].prepare(_CRAWL_PAYLOAD)
    direct_bus = InMemoryJobBus(direct_failure=RuntimeError("broker down"))

    with pytest.raises(RuntimeError, match="broker down"):
        await direct_bus.enqueue(job)
    assert direct_bus.direct.published == []

    staged_bus = InMemoryJobBus(transactional_failure=RuntimeError("database down"))
    tx = staged_bus.transaction()
    with pytest.raises(RuntimeError, match="database down"):
        await staged_bus.in_transaction(tx).enqueue(job, aggregate=_AGGREGATE)
    assert tx._staged == []
    await tx.rollback()


def test_producer_interface_import_does_not_load_workers_or_transports() -> None:
    script = """
import sys
from app.shared.jobs import InMemoryJobBus, JobBus, TypedJob
assert InMemoryJobBus and JobBus and TypedJob
assert not any(name == "app.worker" or name.startswith("app.worker.") for name in sys.modules)
assert not any(name == "celery" or name.startswith("celery.") for name in sys.modules)
assert not any(name == "sqlalchemy" or name.startswith("sqlalchemy.") for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 — literal local interpreter smoke test
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
