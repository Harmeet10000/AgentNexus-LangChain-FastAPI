"""Durable job staging adapter for search ingestion."""

from app.events.writer import stamp_outbox_payload_event_id
from app.features.search.application.ports import SearchIngestionJob, SearchTransaction
from app.shared.jobs import AggregateRef, JobBus
from app.shared.jobs.catalog import JOB_CATALOG
from app.shared.jobs.names import SEARCH_INGEST
from app.shared.jobs.payloads import SearchIngestPayload


class DurableSearchJobQueue:
    """Translate a search-owned job intent into the shared wire catalog."""

    def __init__(self, job_bus: JobBus):
        self._job_bus = job_bus

    async def stage_ingestion(
        self,
        transaction: SearchTransaction,
        *,
        job: SearchIngestionJob,
    ) -> str:
        typed_job = JOB_CATALOG[SEARCH_INGEST].prepare(SearchIngestPayload(**job.model_dump()))
        receipt = await self._job_bus.in_transaction(transaction).enqueue(
            typed_job,
            aggregate=AggregateRef(
                aggregate_type="search_document",
                aggregate_id=job.document_id,
            ),
        )
        await stamp_outbox_payload_event_id(transaction, event_id=receipt.event_id)
        return receipt.event_id
