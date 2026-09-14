"""Liveness proof that evaluation reaches the real document retrieval service.

Deselected by default through the ``requires_db`` marker. The fixture corpus is
inserted in one transaction and rolled back after the report is produced, so the
test exercises PostgreSQL's real retrieval branches without leaving rows behind.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import pytest
from returns.result import Success
from sqlalchemy import select

from app.config import get_settings
from app.connections import init_db
from app.features.documents import service as document_service
from app.features.documents.evaluation import run_live_retrieval_eval
from app.features.documents.model import UnifiedChunk, UnifiedDocument
from app.features.documents.repository import DocumentRepository
from app.features.documents.service import DocumentQueryService
from app.shared.evaluation.report import EvaluationReport, write_report
from app.shared.evaluation.schema import load_golden_set

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

    from app.shared.evaluation.schema import GoldenQuery

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

GOLDEN_PATH = Path("evals/golden/legal_retrieval_v1.jsonl")
REPORT_PATH = Path("evals/reports/baseline.json")


def _seed_rows(*, queries: list[GoldenQuery], user_id: str) -> list[object]:
    """Build the corpus named by the golden artifact itself."""
    rows: list[object] = []
    width = get_settings().EMBEDDING_DIMENSION
    for index, query in enumerate(queries):
        document_id = UUID(query.expected_document_ids[0])
        chunk_id = UUID(query.expected_chunk_ids[0])
        rows.extend(
            [
                UnifiedDocument(
                    id=document_id,
                    user_id=user_id,
                    title=f"evaluation fixture {query.document_kind}",
                    source_uri=None,
                    object_uri=f"eval://{document_id}",
                    content_hash=f"eval-{uuid4().hex}",
                    document_kind=query.document_kind,
                    status="completed",
                    jurisdiction=query.jurisdiction,
                    contract_type=None,
                    parties=[],
                    metadata_={},
                ),
                UnifiedChunk(
                    id=chunk_id,
                    document_id=document_id,
                    user_id=user_id,
                    document_version=1,
                    chunk_index=index,
                    chunk_kind=query.document_kind,
                    content=query.query,
                    preamble="",
                    locus=None,
                    clause_type=None,
                    page_no=1,
                    embedding=[0.01] * width,
                    metadata_={"jurisdiction": query.jurisdiction},
                    custom_metadata={},
                    quality_warnings=[],
                    graphiti_episode_id=None,
                    graphiti_verified=False,
                ),
            ]
        )
    return rows


async def test_live_retrieval_returns_real_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden_result = await load_golden_set(GOLDEN_PATH)
    assert isinstance(golden_result, Success)
    golden = golden_result.unwrap()
    queries = list(golden.queries)
    expected_ids = {identifier for query in queries for identifier in query.expected_chunk_ids}
    user_id = f"eval-probe-{uuid4().hex}"
    width = get_settings().EMBEDDING_DIMENSION

    async def embed_query(_text: str, **_kwargs: object) -> list[float]:
        return [0.01] * width

    def no_model_provider() -> BaseChatModel:
        message = "retrieval evaluation unexpectedly constructed an LLM"
        raise AssertionError(message)

    monkeypatch.setattr(document_service, "embed_text", embed_query)
    engine, session_local = await init_db()
    async with session_local() as session:
        transaction = await session.begin()
        try:
            session.add_all(_seed_rows(queries=queries, user_id=user_id))
            await session.flush()
            snapshot = {
                str(identifier)
                for identifier in (
                    await session.scalars(
                        select(UnifiedChunk.id).where(UnifiedChunk.user_id == user_id)
                    )
                ).all()
            }
            service = DocumentQueryService(
                repo=DocumentRepository(session),
                llm_factory=cast("Callable[[], BaseChatModel]", no_model_provider),
                redis=None,
                graphiti=None,
            )

            evaluation = await run_live_retrieval_eval(
                service=service,
                user_id=user_id,
                queries=queries,
            )
            retrieved = {
                identifier for row in evaluation.rows for identifier in row.retrieved_chunk_ids
            }
            assert retrieved, "wiring failure: live service returned no identifiers"
            assert retrieved <= snapshot
            assert expected_ids <= retrieved

            report = EvaluationReport.from_evaluation(
                evaluation,
                commit_identifier=_commit_identifier(),
                golden_set_version=golden.version,
                timestamp=datetime.now(tz=UTC),
            )
            await write_report(report, REPORT_PATH)
        finally:
            await transaction.rollback()
    await engine.dispose()


def _commit_identifier() -> str:
    """Read the current commit without spawning a subprocess."""
    head_file = Path(".git/HEAD")
    if not head_file.is_file():
        return "unknown"
    head = head_file.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref_path = Path(".git") / head.removeprefix("ref:").strip()
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()[:12]
        return "unknown"
    return head[:12] if head else "unknown"
