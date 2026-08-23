"""Unit tests for task A3 — persisted vector width derives from configuration.

A3 replaced a hard-coded ``Vector(768)`` on the chunk relation with the single
configured value. Its fourth Proof is explicitly a *stub* test rather than a data
check, and the reason is worth keeping in front of whoever edits this file: there
are **zero stored vectors anywhere in this project**. A test that read a real
column's width and compared it to real rows would be a Proof that cannot run. So
the stored width is taken from the mapped column — which is what the database
column is created from — and the *configured* width is moved out from under it.

This is also why the guard is split in two. ``stored_width_mismatch`` is pure and
returns the pair, because ``upsert_chunks`` returns ``AppResult`` and must not
raise; ``assert_stored_width_matches_configured`` raises, for the offline batch
paths that have no ``Result`` to put a failure into.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from returns.result import Failure, Success

from app.config import get_settings
from app.features.documents.model import CHUNK_EMBEDDING_DIM, UnifiedChunk
from app.features.documents.repository import DocumentRepository
from app.utils.embedding import (
    assert_stored_width_matches_configured,
    stored_width_mismatch,
    width_mismatch_detail,
)
from app.utils.exceptions import InfrastructureException


def _stored_width() -> int:
    """The width the ``chunks`` relation is created with.

    Read off the mapped column rather than from ``CHUNK_EMBEDDING_DIM``, because
    the column is what the migration emits — this is the ground truth, and
    ``test_the_named_constant_matches_the_column_it_built`` is what keeps the
    constant honest against it.
    """
    return UnifiedChunk.__table__.c.embedding.type.dim


def _settings_with_dimension(dimension: int) -> object:
    return type("StubSettings", (), {"EMBEDDING_DIMENSION": dimension})()


def _mock_session() -> MagicMock:
    session = MagicMock()
    session.execute = AsyncMock()
    return session


# `returns.Result` defines no `__bool__`, so `Success` and `Failure` are BOTH
# truthy. `assert result` / `assert not result` would pass either way and assert
# nothing at all; every check below goes through isinstance.


# --- Declared width equals configured width ---


def test_chunks_declared_width_is_configured_width() -> None:
    assert _stored_width() == get_settings().EMBEDDING_DIMENSION


def test_the_named_constant_matches_the_column_it_built() -> None:
    """``CHUNK_EMBEDDING_DIM`` is what the write guard compares against.

    If it ever drifts from the column it was used to build, the guard would be
    validating writes against a width the database does not have — the exact
    failure mode A3 exists to remove, reintroduced one layer up.
    """
    assert CHUNK_EMBEDDING_DIM == _stored_width()


# `test_search_chunks_declared_width_is_configured_width` and
# `test_the_two_relations_agree_with_each_other` stood here. Both compared `chunks.embedding`
# against `search_chunks.embedding`, and step 10 of `documents-unified-schema` deleted the second
# relation along with the rest of `app.features.search`. There is now exactly one chunk relation, so
# the cross-relation agreement they asserted is no longer a property that can drift — it is enforced
# structurally rather than by a test.
#
# Their reasoning is kept because it is the argument for *why* one relation: a query vector is
# embedded once. Two relations meant two column widths that could disagree, and a disagreement
# surfaced as an insert error at ingestion time rather than a validation error at startup — landing
# on whoever uploaded a document instead of whoever changed the configuration. Deleting the twin
# removed the failure mode; it did not merely stop testing for it.


# --- A3 Proof N6: a stored width differing from configured refuses new writes ---


async def test_stored_width_differing_from_configured_refuses_writes() -> None:
    stored = _stored_width()
    with patch("app.config.get_settings", return_value=_settings_with_dimension(stored + 256)):
        session = _mock_session()
        repository = DocumentRepository(session)
        result = await repository.upsert_chunks(
            [{"document_id": "doc-1", "chunk_index": 0, "embedding": [0.1] * stored}]
        )

    assert isinstance(result, Failure)
    error = result.failure()
    assert error.code == "EMBEDDING_WIDTH_MISMATCH"
    assert "Re-embedding" in error.message
    # Not retryable: no amount of waiting reconciles a column against a setting.
    assert error.retryable is False
    # The write is refused, not attempted-and-rolled-back.
    session.execute.assert_not_awaited()


async def test_row_wider_than_the_column_refuses_the_whole_batch() -> None:
    stored = _stored_width()
    session = _mock_session()
    repository = DocumentRepository(session)

    result = await repository.upsert_chunks(
        [
            {"document_id": "doc-1", "chunk_index": 0, "embedding": [0.1] * stored},
            {"document_id": "doc-1", "chunk_index": 1, "embedding": [0.1] * (stored * 2)},
        ]
    )

    assert isinstance(result, Failure)
    error = result.failure()
    assert error.code == "EMBEDDING_WIDTH_MISMATCH"
    assert error.details["row_index"] == 1
    assert error.retryable is False
    # No partial write: the good row at index 0 is not persisted either.
    session.execute.assert_not_awaited()


async def test_correctly_sized_rows_reach_the_database() -> None:
    stored = _stored_width()
    session = _mock_session()
    repository = DocumentRepository(session)

    result = await repository.upsert_chunks(
        [{"document_id": "doc-1", "chunk_index": 0, "embedding": [0.1] * stored}]
    )

    assert isinstance(result, Success)
    session.execute.assert_awaited_once()


async def test_rows_without_an_embedding_are_not_treated_as_mismatches() -> None:
    """A chunk may be persisted before it is embedded; ``embedding`` is nullable."""
    session = _mock_session()
    repository = DocumentRepository(session)

    result = await repository.upsert_chunks(
        [{"document_id": "doc-1", "chunk_index": 0, "embedding": None}]
    )

    assert isinstance(result, Success)
    session.execute.assert_awaited_once()


async def test_empty_batch_short_circuits_before_the_guard() -> None:
    session = _mock_session()
    repository = DocumentRepository(session)

    assert isinstance(await repository.upsert_chunks([]), Success)
    session.execute.assert_not_awaited()


# --- The two halves of the guard, directly ---


def test_predicate_returns_none_when_widths_agree() -> None:
    assert stored_width_mismatch(get_settings().EMBEDDING_DIMENSION) is None


def test_predicate_returns_both_widths_when_they_disagree() -> None:
    configured = get_settings().EMBEDDING_DIMENSION
    assert stored_width_mismatch(configured + 1) == (configured + 1, configured)


def test_raising_half_reports_the_remedy_and_is_not_retryable() -> None:
    configured = get_settings().EMBEDDING_DIMENSION
    with pytest.raises(InfrastructureException) as exc_info:
        assert_stored_width_matches_configured(configured + 1, relation="chunks.embedding")

    # `detail` on an APIException is the structured envelope
    # {"message", "error_code", "data"} — not a string. Indexing rather than
    # stringifying, so these assertions cannot pass on an incidental substring of
    # the repr.
    detail = exc_info.value.detail
    assert "Re-embedding" in detail["message"]
    assert "chunks.embedding" in detail["message"]
    assert detail["data"]["stored_dim"] == configured + 1
    assert detail["data"]["configured_dim"] == configured


def test_raising_half_is_silent_when_widths_agree() -> None:
    assert_stored_width_matches_configured(
        get_settings().EMBEDDING_DIMENSION, relation="chunks.embedding"
    )


def test_both_halves_report_the_same_diagnostic() -> None:
    """Shared phrasing, so the ``Result`` path and the raising path cannot drift."""
    configured = get_settings().EMBEDDING_DIMENSION
    shared = width_mismatch_detail(configured + 1, configured, relation="chunks.embedding")

    with pytest.raises(InfrastructureException) as exc_info:
        assert_stored_width_matches_configured(configured + 1, relation="chunks.embedding")

    assert exc_info.value.detail["message"] == shared
