"""Unit tests for task A2 — the batch embedder must never substitute a vector.

Before A2, three sites in ``rag/document_processing/embedder.py`` appended a
zero-filled vector of the configured width when a provider call failed, and a
fourth replaced a blank chunk with an empty string so the result stayed
positionally aligned. Every one of those produces a row that inserts cleanly and
ranks against nothing, so a provider outage became an invisible hole in the
corpus rather than an error. These tests pin the replacement behaviour: raise a
typed project exception, preserve the provider's own exception as ``__cause__``,
and carry the three facts a traceback does not (model, task type, text count).

The mismatch tests deliberately use a width other than the configured one to
stand in for a model of the wrong shape. That literal lives here in ``tests/``
rather than in ``src/app/``, so A2's second Proof — a repository grep for the
stale width — is unaffected.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from app.config import get_settings
from app.shared.rag.document_processing import embedder as embedder_module
from app.shared.rag.document_processing.embedder import (
    GEMINI_TASK_TYPE,
    embed_chunks,
    generate_embedding,
    generate_embeddings_batch,
    get_embedding_dimension,
)
from app.shared.rag.document_processing.models import Chunk
from app.utils.exceptions import ExternalServiceException

_MODULE = "app.shared.rag.document_processing.embedder"

# A width that is not the configured one. Read from configuration rather than
# written down, so the test cannot quietly start passing if the setting changes.
_WRONG_WIDTH = get_settings().EMBEDDING_DIMENSION * 2


def _api_error(code: int = 503) -> genai_errors.APIError:
    """A provider exception of the class the module's handlers name.

    Deliberately ``APIError`` and not ``ClientError``: ``ClientError`` is a
    *subclass*, so raising the parent skips the rate-limit handler and lands in
    the general provider handler — which is the branch that reaches
    ``_process_embeddings_individually`` and therefore the only branch that sets
    ``__cause__``.
    """
    return genai_errors.APIError(
        code,
        {"error": {"message": "upstream unavailable", "status": "UNAVAILABLE"}},
    )


def _client_returning(width: int) -> MagicMock:
    """A stub provider client whose every call returns a vector of ``width``."""
    client = MagicMock()
    client.models.embed_content.return_value = MagicMock(embedding=MagicMock(values=[0.01] * width))
    return client


def _client_raising(error: Exception) -> MagicMock:
    client = MagicMock()
    client.models.embed_content.side_effect = error
    return client


# --- The named A2 Proof: provider failure through the batch path ---


async def test_batch_provider_failure_raises_typed_exception_with_cause_and_notes() -> None:
    provider_error = _api_error()

    with (
        patch(f"{_MODULE}.genai") as mock_genai,
        patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
    ):
        mock_genai.Client.return_value = _client_raising(provider_error)
        with pytest.raises(ExternalServiceException) as exc_info:
            await generate_embeddings_batch(["a real chunk of contract text"], max_retries=1)

    exc = exc_info.value

    # `raise ... from e` at the type change, per EXCEPTION-RULES.md — the provider's
    # own exception survives rather than being swallowed and summarised.
    assert exc.__cause__ is provider_error

    notes = "\n".join(exc.__notes__)
    assert "model=" in notes
    assert f"task_type={GEMINI_TASK_TYPE}" in notes
    assert "text_count=1" in notes

    # 502, because the failure is upstream of this service and not the caller's fault.
    assert exc.status_code == 502


async def test_batch_failure_returns_no_partial_list() -> None:
    """The batch is complete or absent — never a list padded to the right length."""
    with (
        patch(f"{_MODULE}.genai") as mock_genai,
        patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
    ):
        mock_genai.Client.return_value = _client_raising(_api_error())
        with pytest.raises(ExternalServiceException):
            await generate_embeddings_batch(["one", "two", "three"], max_retries=1)


# --- Width validation: a model of the wrong shape is refused, not truncated ---


async def test_provider_returning_wrong_width_is_refused() -> None:
    with patch(f"{_MODULE}.genai") as mock_genai:
        mock_genai.Client.return_value = _client_returning(_WRONG_WIDTH)
        with pytest.raises(ExternalServiceException) as exc_info:
            await generate_embeddings_batch(["a real chunk of contract text"], max_retries=1)

    # `detail` is the structured envelope {"message", "error_code"}, not a string.
    # Indexing rather than stringifying, so the assertion cannot pass on an
    # incidental substring of the repr.
    message = exc_info.value.detail["message"]
    assert str(object=_WRONG_WIDTH) in message
    assert str(object=get_embedding_dimension()) in message


async def test_correct_width_passes_through_unchanged() -> None:
    width = get_embedding_dimension()
    with patch(f"{_MODULE}.genai") as mock_genai:
        mock_genai.Client.return_value = _client_returning(width)
        result = await generate_embeddings_batch(["a real chunk of contract text"], max_retries=1)

    assert len(result) == 1
    assert len(result[0]) == width
    # Not a zero vector: the old failure shape was degenerate in exactly this way.
    assert any(value != 0.0 for value in result[0])


# --- Blank input is a chunking defect, so it fails loudly rather than aligning ---


async def test_blank_text_in_batch_rejects_whole_batch() -> None:
    with patch(f"{_MODULE}.genai") as mock_genai:
        mock_genai.Client.return_value = _client_returning(get_embedding_dimension())
        with pytest.raises(ExternalServiceException) as exc_info:
            await generate_embeddings_batch(["real text", "   "], max_retries=1)

    assert "blank" in exc_info.value.detail["message"]
    assert "text_count=2" in "\n".join(exc_info.value.__notes__)


async def test_blank_single_text_is_refused() -> None:
    with patch(f"{_MODULE}.genai") as mock_genai:
        mock_genai.Client.return_value = _client_returning(get_embedding_dimension())
        with pytest.raises(ExternalServiceException):
            await generate_embedding("")


# --- embed_chunks annotates the batch position and re-raises the same type ---


async def test_embed_chunks_adds_batch_position_and_reraises() -> None:
    chunks = [
        Chunk(content=f"clause {index}", chunk_index=index, document_id="doc-1")
        for index in range(3)
    ]

    with (
        patch(f"{_MODULE}.genai") as mock_genai,
        patch(f"{_MODULE}.asyncio.sleep", new=AsyncMock()),
    ):
        mock_genai.Client.return_value = _client_raising(_api_error())
        with pytest.raises(ExternalServiceException) as exc_info:
            await embed_chunks(chunks, batch_size=2)

    notes = "\n".join(exc_info.value.__notes__)
    # add_note-then-bare-raise, because embed_chunks catches its own exception type.
    assert "batch=1/2" in notes
    assert "chunks_embedded_before_failure=0" in notes


async def test_embed_chunks_empty_input_is_not_an_error() -> None:
    assert await embed_chunks([]) == []


# --- The deleted dimension table must stay deleted ---


def test_model_keyed_dimension_table_is_gone() -> None:
    """``get_model_config`` claimed 1536 for every model against 768-wide columns.

    Worse than a stale literal: the configured model was absent from its map, so
    the ``.get(model, default)`` lookup could never have returned a right answer
    for the deployed configuration regardless of what its entries said.
    """
    assert not hasattr(embedder_module, "get_model_config")


def test_dimension_comes_from_configuration() -> None:
    assert get_embedding_dimension() == get_settings().EMBEDDING_DIMENSION


def test_dimension_accessor_takes_no_model_argument() -> None:
    """Width is a property of the configured corpus, not of a caller's model id."""
    with pytest.raises(TypeError):
        get_embedding_dimension("gemini-embedding-001")  # type: ignore[call-arg]
