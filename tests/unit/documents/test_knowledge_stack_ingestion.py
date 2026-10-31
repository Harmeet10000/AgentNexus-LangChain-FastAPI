from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import langextract as lx
from returns.result import Failure, Success

from app.extraction.langextract.relationships import (
    RELATIONSHIP_CONFIG_DISABLED_REASON,
    RelationshipPassResult,
)
from app.extraction.langextract.service import (
    AsyncExtractionService,
    ExtractionBudget,
    ExtractionSucceeded,
)
from app.features.documents.api.http.dto import IngestionJob
from app.features.documents.application.commands import upload_document as document_ingest
from app.features.documents.application.commands.upload_document import (
    _extract_clause_segments,
    _write_extracted_clause_episodes,
    process_document_ingestion,
)
from app.features.documents.application.queries.documents import (
    DocumentQueryService,
)
from app.features.documents.domain.errors import (
    DocumentGraphWriteError,
    DocumentIngestionCheckpointError,
)
from app.features.documents.domain.fidelity import GraphRelationshipsStatus
from app.features.documents.ingestion.chunking.classification import (
    ClassifiedDocument,
    ParsedDocument,
    PreparedChunk,
)

if TYPE_CHECKING:
    from typing import Any

    from app.extraction.langextract.service import ExtractionRequest


class _ObjectStore:
    async def get_object(self, *, key: str) -> Success[bytes]:
        del key
        return Success(b"fixture")


class _Repo:
    def __init__(self, trees: list[dict[str, Any]] | None = None) -> None:
        self.status_calls: list[dict[str, Any]] = []
        self.chunk_rows: list[dict[str, Any]] = []
        self.trees = trees or []
        self.transaction_checkpoints = 0

    async def update_document_status(self, **kwargs: Any) -> Success[None]:
        self.status_calls.append(kwargs)
        return Success(None)

    async def upsert_chunks(self, rows: object) -> Success[None]:
        self.chunk_rows = list(cast("Any", rows))
        return Success(None)

    async def fetch_structural_trees(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success(self.trees)


def _job() -> IngestionJob:
    return IngestionJob(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="tenant-1",
        filename="fixture.md",
        content_type="text/markdown",
        object_uri="s3://bucket/fixture.md",
    )


async def _run_ingestion(
    monkeypatch: Any,
    provider: object,
    *,
    graph_writer: object | None = None,
    idempotency: object | None = None,
    checkpoint_error: Exception | None = None,
) -> tuple[list[str], _Repo, object]:
    events: list[str] = []
    repo = _Repo()

    async def checkpoint() -> None:
        if checkpoint_error is not None:
            raise checkpoint_error
        repo.transaction_checkpoints += 1

    async def parse_document(**_kwargs: object) -> ParsedDocument:
        return ParsedDocument(
            title="Fixture",
            markdown="# Terms\n\nPayment is due.",
            page_count=1,
            structural_tree={"name": "Fixture", "children": [{"text": "Payment is due."}]},
        )

    async def segment_chunks(**_kwargs: object) -> tuple[list[PreparedChunk], list[object]]:
        events.append("chunk")
        return [PreparedChunk(chunk_index=0, chunk_kind="generic", content="Payment is due.")], []

    async def embed_chunks(**kwargs: object) -> list[dict[str, object]]:
        outcome = cast("Any", kwargs["extraction_outcome"])
        return [
            {
                "id": "10000000-0000-0000-0000-000000000001",
                "chunk_index": 0,
                "custom_metadata": document_ingest._extraction_metadata(outcome),
            }
        ]

    class RecordingProvider:
        def extract(self, request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            events.append("extract")
            extract = cast("Any", provider)
            return extract(request)

    monkeypatch.setattr(document_ingest, "parse_document", parse_document)
    monkeypatch.setattr(
        document_ingest,
        "classify_document",
        lambda **_kwargs: ClassifiedDocument(document_kind="generic"),
    )
    monkeypatch.setattr(document_ingest, "segment_chunks", segment_chunks)
    monkeypatch.setattr(document_ingest, "_embed_chunks", embed_chunks)

    runtime = SimpleNamespace(
        object_store=_ObjectStore(),
        repo=repo,
        graphiti=None,
        llm=object(),
        extraction=AsyncExtractionService(RecordingProvider()),
        graph_writer=graph_writer,
        idempotency=idempotency,
        transaction_checkpoint=checkpoint,
    )
    result = await process_document_ingestion(job=_job(), runtime=cast("Any", runtime))
    return events, repo, result


class _MemoryIdempotency:
    """Module-level in-memory idempotency fake for writer-phase tests."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self.values.get(key)

    async def set(self, key: str, result: object, **_kwargs: object) -> None:
        self.values[key] = result


def _writer_kwargs(
    outcome: ExtractionSucceeded,
    *,
    graph_writer: object,
    idempotency: object,
    document_id: str = "doc-1",
) -> dict[str, Any]:
    """Build the current `_write_extracted_clause_episodes` keyword surface.

    The relationship pass is represented as ran-with-zero-edges (a valid empty
    extraction), mirroring what the pipeline does when the model emits no
    relationship candidates.
    """
    return {
        "segments": _extract_clause_segments(outcome, document_id=document_id),
        "relationships": [],
        "edge_spans": {},
        "relationship_pass": RelationshipPassResult(ran=True, budget=ExtractionBudget()),
        "graph_writer": graph_writer,
        "idempotency": idempotency,
        "document_id": document_id,
        "user_id": "tenant-1",
        "jurisdiction": "India",
        "document_type": "contract",
    }


async def test_extraction_runs_once_before_chunking_and_empty_success_is_complete(
    monkeypatch: Any,
) -> None:
    events, repo, result = await _run_ingestion(monkeypatch, lambda _request: ())

    assert result.unwrap()["status"] == "completed"
    assert events == ["extract", "chunk"]
    assert repo.transaction_checkpoints == 2
    assert repo.status_calls[0]["extraction_incomplete"] is False
    assert repo.status_calls[0]["structural_tree"]["name"] == "Fixture"


async def test_extraction_failure_is_visible_as_warning_but_ingestion_completes(
    monkeypatch: Any,
) -> None:
    def fail(_request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
        message = "provider down"
        raise RuntimeError(message)

    events, repo, result = await _run_ingestion(monkeypatch, fail)

    assert result.unwrap()["status"] == "completed_with_warnings"
    assert events == ["extract", "chunk"]
    assert repo.status_calls[0]["extraction_incomplete"] is True
    metadata = repo.status_calls[0]["metadata_"]
    assert metadata["knowledge_extraction"] == "incomplete"
    assert metadata["knowledge_extraction_failure"] == "provider_error"
    assert metadata["knowledge_extraction_message"] == "LangExtract provider failed"
    assert repo.chunk_rows[0]["custom_metadata"]["knowledge_extraction"] == "incomplete"
    assert repo.chunk_rows[0]["custom_metadata"]["knowledge_extraction_failure"] == "provider_error"
    assert repo.transaction_checkpoints == 2


async def test_graph_write_failure_marks_document_failed_before_chunk_storage(
    monkeypatch: Any,
) -> None:
    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due.",
        char_interval=lx.data.CharInterval(start_pos=0, end_pos=15),
    )

    def extract(_request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
        return (lx.data.AnnotatedDocument(text="Payment is due.", extractions=[extraction]),)

    class FailingWriter:
        async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
            del clause_text, metadata
            message = "graph unavailable"
            raise RuntimeError(message)

    class Idempotency:
        async def get(self, key: str) -> None:
            del key

        async def set(self, key: str, result: object, **kwargs: object) -> None:
            del key, result, kwargs

    events, repo, result = await _run_ingestion(
        monkeypatch,
        extract,
        graph_writer=FailingWriter(),
        idempotency=Idempotency(),
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), DocumentGraphWriteError)
    assert result.failure().retryable is True
    # One provider invocation: mandatory clause writes run immediately after
    # pass-1, so the clause-write failure aborts before the optional
    # relationship pass can spend any budget. Chunking never runs.
    assert events == ["extract"]
    assert repo.status_calls[0]["status"] == "failed"
    assert repo.status_calls[0]["extraction_incomplete"] is True
    assert repo.transaction_checkpoints == 0


async def test_checkpoint_failure_is_retryable(monkeypatch: Any) -> None:
    events, _repo, result = await _run_ingestion(
        monkeypatch,
        lambda _request: (),
        checkpoint_error=RuntimeError("database unavailable"),
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), DocumentIngestionCheckpointError)
    assert result.failure().retryable is True
    assert events == ["extract"]


async def test_structural_branch_uses_repository_and_pure_navigator() -> None:
    repo = _Repo(
        trees=[
            {
                "structural_tree": {
                    "name": "Agreement",
                    "children": [{"label": "section", "name": "Payment", "text": "Due in 30 days"}],
                },
                "document_id": "document-1",
            }
        ]
    )
    service = DocumentQueryService(
        repo=cast("Any", repo), llm_factory=cast("Any", object), redis=None, graphiti=None
    )

    result = await service.navigate_structure(user_id="tenant-1", query="due", limit=1)

    assert result.unwrap() == [("document-1", "Agreement", "Payment")]


async def test_clause_extractions_use_canonical_writer_idempotently() -> None:
    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due in thirty days.",
        char_interval=lx.data.CharInterval(start_pos=10, end_pos=40),
        attributes={"clause_id": "4.1"},
    )
    outcome = ExtractionSucceeded(
        documents=(lx.data.AnnotatedDocument(text="x" * 50, extractions=[extraction]),)
    )

    class Writer:
        def __init__(self) -> None:
            self.calls = 0

        async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
            del clause_text, metadata
            self.calls += 1
            return "episode-1"

    class Idempotency:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        async def get(self, key: str) -> object | None:
            return self.values.get(key)

        async def set(self, key: str, result: object, **_kwargs: object) -> None:
            self.values[key] = result

    writer = Writer()
    idempotency = Idempotency()

    first = await _write_extracted_clause_episodes(
        **cast("Any", _writer_kwargs(outcome, graph_writer=writer, idempotency=idempotency))
    )
    second = await _write_extracted_clause_episodes(
        **cast("Any", _writer_kwargs(outcome, graph_writer=writer, idempotency=idempotency))
    )

    assert first.success is True
    assert second.success is True
    # The relationship pass ran and validly produced zero edges: a distinct,
    # truthful state — succeeded with zero writes, not a vacuous all([]).
    assert first.relationships.status is GraphRelationshipsStatus.SUCCEEDED
    assert first.relationships.relationships_attempted == 0
    assert first.relationships.relationships_succeeded == 0
    assert writer.calls == 1


async def test_clause_write_failure_is_reported_to_ingestion() -> None:
    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due in thirty days.",
        char_interval=lx.data.CharInterval(start_pos=10, end_pos=40),
        attributes={"clause_id": "4.1"},
    )
    outcome = ExtractionSucceeded(
        documents=(lx.data.AnnotatedDocument(text="x" * 50, extractions=[extraction]),)
    )

    class FailingWriter:
        async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
            del clause_text, metadata
            message = "graph unavailable"
            raise RuntimeError(message)

    class Idempotency:
        async def get(self, key: str) -> None:
            del key

        async def set(self, key: str, result: object, **kwargs: object) -> None:
            del key, result, kwargs

    result = await _write_extracted_clause_episodes(
        **cast(
            "Any",
            _writer_kwargs(outcome, graph_writer=FailingWriter(), idempotency=Idempotency()),
        )
    )

    assert result.success is False


async def test_durable_idempotency_failure_is_reported_after_graph_write() -> None:
    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due in thirty days.",
        char_interval=lx.data.CharInterval(start_pos=10, end_pos=40),
        attributes={"clause_id": "4.1"},
    )
    outcome = ExtractionSucceeded(
        documents=(lx.data.AnnotatedDocument(text="x" * 50, extractions=[extraction]),)
    )

    class Writer:
        async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
            del clause_text, metadata
            return "episode-1"

    class SaveFailingIdempotency:
        async def get(self, key: str) -> None:
            del key

        async def set(self, key: str, result: object, **kwargs: object) -> None:
            del key, result, kwargs
            message = "durable idempotency write failed"
            raise RuntimeError(message)

    result = await _write_extracted_clause_episodes(
        **cast(
            "Any",
            _writer_kwargs(outcome, graph_writer=Writer(), idempotency=SaveFailingIdempotency()),
        )
    )

    assert result.success is False


async def test_durable_idempotency_read_failure_is_reported_before_graph_write() -> None:
    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due in thirty days.",
        char_interval=lx.data.CharInterval(start_pos=10, end_pos=40),
        attributes={"clause_id": "4.1"},
    )
    outcome = ExtractionSucceeded(
        documents=(lx.data.AnnotatedDocument(text="x" * 50, extractions=[extraction]),)
    )

    class Writer:
        calls = 0

        async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
            del clause_text, metadata
            self.calls += 1
            return "episode-1"

    class ReadFailingIdempotency:
        async def get(self, key: str) -> None:
            del key
            message = "durable idempotency read failed"
            raise RuntimeError(message)

        async def set(self, key: str, result: object, **kwargs: object) -> None:
            del key, result, kwargs

    writer = Writer()
    result = await _write_extracted_clause_episodes(
        **cast(
            "Any",
            _writer_kwargs(outcome, graph_writer=writer, idempotency=ReadFailingIdempotency()),
        )
    )

    assert result.success is False
    assert writer.calls == 0


async def test_partially_configured_graph_writer_dependencies_fail_closed() -> None:
    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due.",
        char_interval=lx.data.CharInterval(start_pos=0, end_pos=15),
    )
    outcome = ExtractionSucceeded(
        documents=(lx.data.AnnotatedDocument(text="Payment is due.", extractions=[extraction]),)
    )

    partial_writer = await _write_extracted_clause_episodes(
        **cast(
            "Any",
            _writer_kwargs(outcome, graph_writer=object(), idempotency=None),
        )
    )
    assert partial_writer.success is False
    assert partial_writer.relationships.status is GraphRelationshipsStatus.NOT_ATTEMPTED

    partial_idempotency = await _write_extracted_clause_episodes(
        **cast(
            "Any",
            _writer_kwargs(outcome, graph_writer=None, idempotency=object()),
        )
    )
    assert partial_idempotency.success is False
    assert partial_idempotency.relationships.status is GraphRelationshipsStatus.NOT_ATTEMPTED


async def test_disabled_relationship_pass_reports_skipped_not_implemented() -> None:
    """A configuration-disabled pass keeps the distinct skipped state."""

    class OkWriter:
        async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
            del clause_text, metadata
            return "episode-1"

    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due.",
        char_interval=lx.data.CharInterval(start_pos=0, end_pos=15),
    )
    outcome = ExtractionSucceeded(
        documents=(lx.data.AnnotatedDocument(text="Payment is due.", extractions=[extraction]),)
    )
    kwargs = _writer_kwargs(outcome, graph_writer=OkWriter(), idempotency=_MemoryIdempotency())
    kwargs["relationship_pass"] = RelationshipPassResult(
        ran=False,
        skip_reason=RELATIONSHIP_CONFIG_DISABLED_REASON,
        budget=ExtractionBudget(),
    )

    result = await _write_extracted_clause_episodes(**cast("Any", kwargs))

    assert result.success is True
    assert result.relationships.status is GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED
    assert result.relationships.reason == RELATIONSHIP_CONFIG_DISABLED_REASON
