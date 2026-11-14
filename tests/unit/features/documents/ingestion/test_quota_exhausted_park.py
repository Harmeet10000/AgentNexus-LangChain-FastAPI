"""STRIDE D-3: a daily-quota 429 parks the document, it does not dead-letter.

The measured incident (2026-11-14): 12/15 corpus ingestions failed on the
Gemini free-tier cap and every one landed as an untyped
``INGESTION_UNEXPECTED_FAILURE`` retry burn. The contract pinned here:

- the ingestion node catches the typed ``EmbeddingQuotaExhaustedError`` and returns
  a non-retryable ``quota_exhausted`` result dict (the worker's fall-through
  marks idempotency completed and returns — no raise, no retry, no dead-letter);
- the repository write is the NON-terminal ``quota_exhausted`` status with the
  typed ``ingestion_error`` metadata — resumable by the next quota window's
  idempotent driver, which selects exactly the non-terminal rows;
- the read-side state surfaces as retryable ``index_not_ready``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from returns.result import Success

from app.ai.embeddings import EmbeddingQuotaExhaustedError
from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.domain.document import (
    DOCUMENT_STATUS_VALUES,
    TERMINAL_DOCUMENT_STATUSES,
    DocumentStatus,
)
from app.features.documents.ingestion.workflow import _make_ingest_document_node
from app.utils.storage import StorageService

if TYPE_CHECKING:
    from typing import Any

TENANT = "personal:tenant-1"
DOCUMENT_ID = "00000000-0000-0000-0000-000000000001"


async def _noop_commit() -> None:
    return None


class _RecordingRepo(DocumentRepository):
    """Subclasses the real repository: IngestionRuntime isinstance-checks it.

    __init__ is skipped (no session wiring is needed for the park write); the
    one overridden method is the only surface the quota path touches.
    """

    def __init__(self) -> None:
        self.status_calls: list[dict[str, Any]] = []
        # The node builds the runtime with `transaction_checkpoint=repo.session.commit`.
        self.session = SimpleNamespace(commit=_noop_commit)

    async def update_document_status(self, **kwargs: Any) -> object:
        self.status_calls.append(kwargs)
        return Success(object())


class _NoopObjectStore(StorageService):
    def __init__(self) -> None:
        pass  # deliberately unwired: isinstance is the only contract


def _node(ingest_fn: object, repo: _RecordingRepo) -> object:
    return _make_ingest_document_node(
        object_store=_NoopObjectStore(),
        graphiti=None,
        ingest_document_fn=ingest_fn,  # type: ignore[arg-type]
        llm=FakeListChatModel(responses=["unused"]),
    )


def _state() -> dict[str, Any]:
    return {
        "document_id": DOCUMENT_ID,
        "user_id": "user-1",
        "tenant_id": TENANT,
        "filename": "contract.pdf",
        "content_type": "application/pdf",
        "object_uri": "r2://bucket/abc.pdf",
        "document_version": 1,
        "attempt": 0,
    }


def _config(repo: _RecordingRepo) -> dict[str, Any]:
    return {"configurable": {"document_repository": repo}}


async def test_quota_raise_parks_the_document_non_terminally() -> None:
    repo = _RecordingRepo()

    async def _quota_boom(**_kwargs: Any) -> object:
        quota_detail = (
            "Embedding provider daily quota exhausted for model gemini-embedding-001"
        )
        raise EmbeddingQuotaExhaustedError(
            quota_detail,
            model="gemini-embedding-001",
            provider_status=429,
        )

    node = _node(_quota_boom, repo)
    result = await node(_state(), _config(repo))  # type: ignore[operator]

    assert result["status"] == "quota_exhausted"
    assert result["error_code"] == "embedding_quota_exhausted"
    assert result["error_retryable"] is False
    # The park write: non-terminal status, typed metadata, extraction honesty.
    assert len(repo.status_calls) == 1
    call = repo.status_calls[0]
    assert call["status"] == "quota_exhausted"
    ingestion_error = call["metadata_"]["ingestion_error"]
    assert ingestion_error["code"] == "embedding_quota_exhausted"
    assert ingestion_error["retryable"] is False
    assert ingestion_error["ingestion_state"]["document_id"] == DOCUMENT_ID
    assert call.get("extraction_incomplete", False) is not True


async def test_quota_status_is_non_terminal_and_typed() -> None:
    assert DocumentStatus.QUOTA_EXHAUSTED.value in DOCUMENT_STATUS_VALUES
    assert DocumentStatus.QUOTA_EXHAUSTED.value not in TERMINAL_DOCUMENT_STATUSES


async def test_evidence_state_surfaces_quota_park_as_retryable_index_not_ready() -> None:
    from app.features.documents.application.evidence.service import _ingestion_state

    state = _ingestion_state({"status": "quota_exhausted", "chunk_count": 0}, DOCUMENT_ID)
    assert state.state == "index_not_ready"
