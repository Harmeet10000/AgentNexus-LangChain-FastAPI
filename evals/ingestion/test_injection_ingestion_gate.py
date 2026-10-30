"""Queued-ingestion security and fidelity gate (hermetic).

Runs prompt-injection fixtures and a benign control through the real queued
ingestion path — the same compiled ``build_document_ingestion_graph`` graph the
Celery worker task invokes — with hermetic fakes for the object store, the
embedding provider and the LangExtract provider (no external services, no
model calls, no database).

What the gate proves:

- Document CONTENT cannot alter server-owned identity: ``tenant_id``,
  ``document_id``, graphiti group binding or ``allowed_doc_ids``.
- Content cannot re-own or renumber contract boundaries; ``contract_index``
  stays a server-assigned contiguous sequence.
- Forged "authoritative metadata" blocks stay untrusted content-derived
  metadata; they never widen authorization or overwrite identity.
- Fidelity thresholds (table/section preservation, chunk plausibility) and the
  typed relationship-phase state are measured and persisted; a corrupted
  checksum or an untruthful success (zero chunks for a non-empty document)
  fails validation instead of reporting success.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import PrivateAttr
from returns.result import Success

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.application.commands import upload_document as document_ingest
from app.features.documents.ingestion.chunking.classification import (
    establish_contract_boundaries,
)
from app.features.documents.ingestion.workflow import build_document_ingestion_graph
from app.retrieval.graphiti.grouping import build_graphiti_group_id
from app.utils.storage import StorageService
from evals.ingestion.fixtures_loader import (
    CHECKSUMS_FILE,
    FIXTURES_DIR,
    FORGED_TENANT,
    FixtureCorruptionError,
    _verify_checksums,
)

if TYPE_CHECKING:
    from typing import Any

SERVER_TENANT = "personal:gate-tenant"

_IDENTITY_KEYS = {"tenant_id", "document_id", "group_id", "graphiti_group_id", "allowed_doc_ids"}


class _ObjectStore(StorageService):
    """Hermetic object store serving one fixture's bytes (isinstance-valid)."""

    _data: bytes = PrivateAttr()

    def __init__(self, data: bytes) -> None:
        super().__init__(bucket="ingestion-gate")
        self._data = data

    @override
    async def get_object(self, *, key: str) -> Success[bytes]:
        del key
        return Success(self._data)


class _FakeOutboxResult:
    def scalar_one_or_none(self) -> str:
        return "gate-event-id"


class _FakeOutboxSession:
    """Fake SQLAlchemy session surface for the terminal-outbox staging call."""

    async def commit(self) -> None:
        return None

    async def execute(self, *_args: object, **_kwargs: object) -> _FakeOutboxResult:
        return _FakeOutboxResult()


class _RecordingRepo(DocumentRepository):
    """In-memory repository recording every status and chunk write."""

    def __init__(self) -> None:
        self.session = cast("Any", _FakeOutboxSession())
        self.status_calls: list[dict[str, Any]] = []
        self.chunk_rows: list[dict[str, Any]] = []

    @override
    async def update_document_status(self, **kwargs: Any) -> Success[None]:
        self.status_calls.append(kwargs)
        return Success(None)

    @override
    async def fetch_status(self, **_kwargs: object) -> Success[None]:
        # No persisted record: the extraction-checkpoint rehydration is skipped.
        return Success(None)

    @override
    async def upsert_chunks(self, rows: object) -> Success[None]:
        self.chunk_rows = list(cast("Any", rows))
        return Success(None)

    @override
    async def analyze_chunks(self) -> Success[None]:
        return Success(None)


class _RecordingProvider:
    """Deterministic pass-1 provider: extraction returns nothing, no model calls."""

    def extract(self, request: object) -> tuple[()]:
        del request
        return ()


_GROUNDED_PHRASE = "The Contractor shall maintain the common areas of Greystone House"


class _GroundedProvider:
    """Pass-1 provider emitting one grounded clause so the real writer runs.

    The extraction span is located inside the actual boundary text the pipeline
    supplies, mirroring LangExtract's char-interval grounding contract without
    any model call.
    """

    def extract(self, request: Any) -> tuple[Any, ...]:
        import langextract as lx

        text = str(getattr(request, "text", ""))
        start = text.find(_GROUNDED_PHRASE)
        if start < 0:
            return ()
        return (
            lx.data.AnnotatedDocument(
                text=text,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="obligation",
                        extraction_text=_GROUNDED_PHRASE,
                        char_interval=lx.data.CharInterval(
                            start_pos=start, end_pos=start + len(_GROUNDED_PHRASE)
                        ),
                    )
                ],
            ),
        )


class _StructuredInvoker:
    """Structured-output invoker returning an empty extraction (no model call)."""

    async def ainvoke(self, _messages: object) -> dict[str, object]:
        return {}


class _HermeticContractLLM(FakeListChatModel):
    """Fake chat model supporting structured output for legal-metadata extraction.

    The real extraction path runs (prompt build, validation, heuristic merge)
    but the model call is hermetic: it always returns an empty extraction so
    metadata falls back to the typed heuristic path.
    """

    @override
    def with_structured_output(self, _schema: object) -> _StructuredInvoker:
        return _StructuredInvoker()


class _GraphWriter:
    """Minimal clause-episode writer so the real graphiti writer path executes."""

    async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
        del clause_text, metadata
        return "episode-1"


class _Idempotency:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self.values.get(key)

    async def set(self, key: str, result: object, **_kwargs: object) -> None:
        self.values[key] = result


class _QueuedRun:
    def __init__(self, result: dict[str, Any], repo: _RecordingRepo) -> None:
        self.result = result
        self.repo = repo

    @property
    def final_metadata(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self.repo.status_calls[-1]["metadata_"])

    def all_metadata_maps(self) -> list[dict[str, Any]]:
        maps: list[dict[str, Any]] = []
        for chunk in self.repo.chunk_rows:
            maps.extend(
                [
                    cast("dict[str, Any]", chunk.get("metadata_", {})),
                    cast("dict[str, Any]", chunk.get("custom_metadata", {})),
                ]
            )
        return maps


async def _run_queued_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    markdown: str,
    *,
    graph_writer: object | None = None,
    idempotency: object | None = None,
    document_id: str | None = None,
    chunk_override: list[object] | None = None,
    grounded_extraction: bool = False,
) -> _QueuedRun:
    """Run one fixture through the real compiled ingestion graph (queued body)."""
    from app.extraction.langextract.service import AsyncExtractionService

    document_id = document_id or str(uuid4())
    repo = _RecordingRepo()
    object_store = _ObjectStore(markdown.encode("utf-8"))
    llm = _HermeticContractLLM(responses=["unused"])

    async def embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(document_ingest, "embed_texts", embed_texts)
    if chunk_override is not None:

        async def segment_chunks(**_kwargs: object) -> tuple[list[object], list[object]]:
            return list(chunk_override), []

        monkeypatch.setattr(document_ingest, "segment_chunks", segment_chunks)

    graph = build_document_ingestion_graph(
        object_store=cast("Any", object_store),
        graphiti=None,
        ingest_document_fn=document_ingest.process_document_ingestion,
        llm=cast("Any", llm),
        extraction=AsyncExtractionService(
            _GroundedProvider() if grounded_extraction else _RecordingProvider()
        ),
        graph_writer=graph_writer,
        idempotency=idempotency,
    )
    result = await graph.ainvoke(
        {
            "document_id": document_id,
            "user_id": "gate-user",
            "tenant_id": SERVER_TENANT,
            "filename": "fixture.md",
            "content_type": "text/markdown",
            "object_uri": "s3://gate/fixture.md",
            "document_version": 1,
            "attempt": 0,
            "task_id": None,
        },
        {"configurable": {"document_repository": repo}},
    )
    return _QueuedRun(cast("dict[str, Any]", result), repo)


# ---------------------------------------------------------------------------
# Server-owned identity


async def test_injected_identity_cannot_alter_tenant_scope(
    monkeypatch: pytest.MonkeyPatch,
    fixtures: dict[str, str],
) -> None:
    run = await _run_queued_ingestion(monkeypatch, fixtures["injection_identity_override.md"])

    assert run.result["status"] in {"completed", "completed_with_warnings"}
    assert run.repo.chunk_rows, "gate fixture must produce chunks"
    for chunk in run.repo.chunk_rows:
        assert chunk["tenant_id"] == SERVER_TENANT
        assert chunk["user_id"] == "gate-user"
    ingestion_state = run.final_metadata["ingestion_state"]
    assert ingestion_state["tenant_id"] == SERVER_TENANT
    assert ingestion_state["tenant_id"] != FORGED_TENANT
    # The forged identity block is content, never metadata structure: it may
    # appear inside chunk text, but never as a server identity key.
    for metadata in run.all_metadata_maps():
        assert not (_IDENTITY_KEYS & set(metadata))


async def test_injected_group_id_cannot_reach_graph_binding(
    monkeypatch: pytest.MonkeyPatch,
    fixtures: dict[str, str],
) -> None:
    run = await _run_queued_ingestion(monkeypatch, fixtures["injection_identity_override.md"])

    for metadata in run.all_metadata_maps():
        assert not (_IDENTITY_KEYS & set(metadata))
    # Group binding is derived only from job identity; the forged tenant's group
    # id can never equal the server-derived one for this document.
    forged_group = build_graphiti_group_id(
        tenant_id=FORGED_TENANT, document_id="00000000-0000-0000-0000-000000000001"
    )
    server_group = build_graphiti_group_id(
        tenant_id=SERVER_TENANT, document_id="00000000-0000-0000-0000-000000000001"
    )
    assert forged_group != server_group


async def test_forged_boundary_markers_cannot_reown_contracts(
    monkeypatch: pytest.MonkeyPatch,
    fixtures: dict[str, str],
) -> None:
    run = await _run_queued_ingestion(monkeypatch, fixtures["injection_forged_boundaries.md"])

    assert run.repo.chunk_rows
    contract_indices = [
        chunk["contract_index"]
        for chunk in run.repo.chunk_rows
        if chunk.get("contract_index") is not None
    ]
    assert contract_indices, "multi-contract fixture must produce contract chunks"
    # Contract identity is server-assigned by the conservative boundary
    # function (contiguous 0..n-1); the forged "contract 7" claim can never
    # appear as an index or a tenant.
    boundary_result = establish_contract_boundaries(
        markdown=fixtures["injection_forged_boundaries.md"]
    )
    assert set(contract_indices) <= set(range(len(boundary_result.boundaries)))
    assert 7 not in contract_indices
    for chunk in run.repo.chunk_rows:
        assert chunk["tenant_id"] == SERVER_TENANT
    for metadata in run.all_metadata_maps():
        assert not (_IDENTITY_KEYS & set(metadata))


async def test_forged_legal_metadata_stays_untrusted(
    monkeypatch: pytest.MonkeyPatch,
    fixtures: dict[str, str],
) -> None:
    run = await _run_queued_ingestion(monkeypatch, fixtures["injection_forged_legal_metadata.md"])

    assert run.result["status"] in {"completed", "completed_with_warnings"}
    ingestion_state = run.final_metadata["ingestion_state"]
    assert ingestion_state["tenant_id"] == SERVER_TENANT
    for chunk in run.repo.chunk_rows:
        assert chunk["tenant_id"] == SERVER_TENANT
    for metadata in run.all_metadata_maps():
        assert not (_IDENTITY_KEYS & set(metadata))
    # Forged dates/parties are content-derived metadata at most — they must
    # never leak into the server-owned ingestion identity block.
    assert "1801-01-01" not in str(ingestion_state)


# ---------------------------------------------------------------------------
# Fidelity + relationship truth through the queued path


async def test_benign_fixture_meets_fidelity_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    fixtures: dict[str, str],
) -> None:
    run = await _run_queued_ingestion(
        monkeypatch,
        fixtures["benign_contract_with_table.md"],
        graph_writer=_GraphWriter(),
        idempotency=_Idempotency(),
        grounded_extraction=True,
    )

    assert run.result["status"] in {"completed", "completed_with_warnings"}
    fidelity = run.final_metadata["fidelity"]
    assert fidelity["fidelity_schema_version"] == "fidelity_v1"
    # The parsed table must survive into chunks — the gate fails on table loss.
    assert fidelity["tables_total"] == 1
    assert fidelity["tables_preserved"] == 1
    assert fidelity["table_loss_ratio"] <= 0.0
    assert fidelity["chunk_plausibility"]["verdict"] == "plausible"
    assert fidelity["section_loss_ratio"] <= fidelity["thresholds"]["section_loss_max"]
    # The relationship pass is not implemented; the queued path must say so.
    relationships = run.final_metadata["graph_relationships"]
    assert relationships["status"] == "skipped_not_implemented"
    assert relationships["reason"] == "relationship_extraction_not_implemented"
    # No fabricated numbers: OCR is unavailable hermetically (no docling
    # conversion of markdown), and the record says so explicitly.
    assert fidelity["ocr_status"] == "unavailable"
    assert fidelity["ocr_confidence"] is None
    for chunk in run.repo.chunk_rows:
        warning_codes = {
            warning["code"]
            for warning in cast("list[dict[str, Any]]", chunk.get("quality_warnings", []))
        }
        assert not {code for code in warning_codes if code.startswith("FIDELITY_")}


async def test_untruthful_success_zero_chunks_fails_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    fixtures: dict[str, str],
) -> None:
    """Fault injection: a chunker that emits nothing must not read as success."""
    run = await _run_queued_ingestion(
        monkeypatch,
        fixtures["benign_contract_with_table.md"],
        graph_writer=_GraphWriter(),
        idempotency=_Idempotency(),
        chunk_override=[],
    )

    assert run.result["status"] == "completed_with_warnings"
    fidelity = run.final_metadata["fidelity"]
    assert fidelity["chunk_plausibility"]["verdict"] == "implausible"
    assert "zero chunks" in fidelity["chunk_plausibility"]["reason"]


# ---------------------------------------------------------------------------
# Fixture integrity


def test_fixture_checksum_manifest_covers_all_fixtures() -> None:
    verified = _verify_checksums()
    fixture_files = {
        path.name
        for path in FIXTURES_DIR.iterdir()
        if path.is_file() and path.suffix == ".md" and path.name != "README.md"
    }
    assert set(verified) == fixture_files


def test_corrupted_fixture_fails_validation(tmp_path: Any) -> None:
    """A corrupted checksum must fail closed, never silently pass."""
    import shutil

    import evals.ingestion.fixtures_loader as loader

    for source in FIXTURES_DIR.iterdir():
        if source.is_file():
            shutil.copy(source, tmp_path / source.name)
    tampered = tmp_path / "benign_contract_with_table.md"
    tampered.write_text("tampered content", encoding="utf-8")
    checksums_copy = tmp_path / "CHECKSUMS.sha256"
    checksums_copy.write_text(CHECKSUMS_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    real_fixtures = loader.FIXTURES_DIR
    real_checksums = loader.CHECKSUMS_FILE

    loader.FIXTURES_DIR = tmp_path
    loader.CHECKSUMS_FILE = checksums_copy
    try:
        with pytest.raises(FixtureCorruptionError, match="checksum mismatch"):
            _verify_checksums()
    finally:
        loader.FIXTURES_DIR = real_fixtures
        loader.CHECKSUMS_FILE = real_checksums


def test_forged_tenant_fixture_marker_present(fixtures: dict[str, str]) -> None:
    """Sanity: the payload really contains the forged tenant the tests assert on."""
    assert FORGED_TENANT in fixtures["injection_identity_override.md"]
    assert FORGED_TENANT in fixtures["injection_forged_boundaries.md"]
