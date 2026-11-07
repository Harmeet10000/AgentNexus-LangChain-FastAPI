"""Contextual chunk retrieval at ingest (G2 tasks 2.1).

Pins the contextual-chunk-retrieval spec at the chunking seam: preamble-fill
with document-context sentences, provider calls proportional to documents (one
structured call per document regardless of chunk count), the disabled state
explicitly marked context-free (never silently empty), and a generation failure
degrading to the context-free state with a typed warning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.config.settings import get_settings
from app.features.documents.ingestion.chunking.classification import (
    ClassifiedDocument,
    ParsedDocument,
    PreparedChunk,
)
from app.features.documents.ingestion.chunking.context_summary import (
    CONTEXT_STATE_CONTEXT_FREE,
    CONTEXT_STATE_CONTEXTUALIZED,
    CONTEXT_STATE_FAILED,
    DocumentContextSummaries,
    apply_document_context,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any


async def _chunks(count: int = 3) -> list[PreparedChunk]:
    # Hand-built structural preambles (segment_chunks merges short synthetic
    # sections, which would make the chunk count depend on the chunker's
    # peer-merging behaviour rather than the test's intent).
    return [
        PreparedChunk(
            chunk_index=index,
            chunk_kind="legal_contract",
            content=f"Section {index}. Obligations of the parties.",
            preamble=("This is a termination section from a msa document involving Acme and Beta."),
        )
        for index in range(count)
    ]


def _generate_stub(contexts: dict[int, str] | None = None) -> tuple[Callable[..., Any], list[int]]:
    calls: list[int] = []

    async def _generate(_llm: Any, _schema: type, _payload: str) -> DocumentContextSummaries:
        calls.append(1)
        return DocumentContextSummaries(
            contexts=[
                {"chunk_index": index, "context": sentence}
                for index, sentence in (contexts or {}).items()
            ]
        )

    return _generate, calls


@pytest.fixture
def _enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_CONTEXTUAL_PREAMBLE_ENABLED", "true")
    # get_settings is process-cached; the env var must reach a fresh instance.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def _disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_CONTEXTUAL_PREAMBLE_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("_enabled")
async def test_context_reaches_the_preamble_and_marks_the_chunk() -> None:
    chunks = await _chunks(2)
    generate, calls = _generate_stub(
        {0: "Chunk 0 is the termination section.", 1: "Chunk 1 is payment."}
    )

    updated, warnings = await apply_document_context(
        chunks=chunks,
        parsed=ParsedDocument(title="MSA", markdown="x", page_count=1),
        classified=ClassifiedDocument(document_kind="legal_contract"),
        llm=None,
        generate=generate,
    )

    assert not warnings
    assert calls == [1]  # one document-level call, not one per chunk
    for chunk in updated:
        assert chunk.metadata_["context_summary"]["state"] == CONTEXT_STATE_CONTEXTUALIZED
    assert updated[0].preamble.startswith("Chunk 0 is the termination section.")
    assert updated[1].preamble.startswith("Chunk 1 is payment.")
    # The structural preamble content survives behind the context sentence —
    # the generated search text (clause_type || preamble || content) therefore
    # carries both signals.
    assert "document" in updated[0].preamble.lower()


@pytest.mark.usefixtures("_enabled")
async def test_provider_calls_are_proportional_to_documents_not_chunks() -> None:
    for count in (1, 5, 9):
        chunks = await _chunks(count)
        generate, calls = _generate_stub({i: f"Context {i}" for i in range(count)})
        updated, _ = await apply_document_context(
            chunks=chunks,
            parsed=ParsedDocument(title="t", markdown="x", page_count=1),
            classified=ClassifiedDocument(document_kind="generic"),
            llm=None,
            generate=generate,
        )
        assert len(updated) == count
        assert sum(calls) == 1  # call count ∝ documents, never chunks


@pytest.mark.usefixtures("_disabled")
async def test_disabled_documents_are_explicitly_context_free() -> None:
    chunks = await _chunks(2)
    preambles = [chunk.preamble for chunk in chunks]

    updated, warnings = await apply_document_context(
        chunks=chunks,
        parsed=ParsedDocument(title="t", markdown="x", page_count=1),
        classified=ClassifiedDocument(document_kind="generic"),
        llm=None,
        generate=_generate_stub({0: "never called"})[0],
    )

    assert not warnings
    for chunk in updated:
        # The disabled state is an explicit marker, distinguishable from a run
        # that produced nothing — never a silent empty preamble.
        assert chunk.metadata_["context_summary"]["state"] == CONTEXT_STATE_CONTEXT_FREE
    assert [chunk.preamble for chunk in updated] == preambles


@pytest.mark.usefixtures("_enabled")
async def test_generation_failure_degrades_to_the_context_free_state() -> None:
    chunks = await _chunks(2)

    async def _failing(_llm: Any, _schema: type, _payload: str) -> DocumentContextSummaries:
        failure = RuntimeError("provider down")
        raise failure

    updated, warnings = await apply_document_context(
        chunks=chunks,
        parsed=ParsedDocument(title="t", markdown="x", page_count=1),
        classified=ClassifiedDocument(document_kind="generic"),
        llm=None,
        generate=_failing,
    )

    assert [w.code for w in warnings] == ["CONTEXT_SUMMARY_FAILED"]
    for chunk in updated:
        assert chunk.metadata_["context_summary"]["state"] == CONTEXT_STATE_FAILED
        assert chunk.preamble  # the structural preamble is untouched, not emptied


@pytest.mark.usefixtures("_enabled")
async def test_chunk_without_a_context_line_stays_explicitly_context_free() -> None:
    chunks = await _chunks(2)
    generate, _calls = _generate_stub({0: "Only chunk 0 was contextualized."})

    updated, _warnings = await apply_document_context(
        chunks=chunks,
        parsed=ParsedDocument(title="t", markdown="x", page_count=1),
        classified=ClassifiedDocument(document_kind="generic"),
        llm=None,
        generate=generate,
    )

    assert updated[0].metadata_["context_summary"]["state"] == CONTEXT_STATE_CONTEXTUALIZED
    assert updated[1].metadata_["context_summary"]["state"] == CONTEXT_STATE_CONTEXT_FREE


def test_flag_defaults_off(tmp_path: Path) -> None:
    """The default is off until a recorded baseline comparison supports default-on."""
    from app.config.settings import Settings

    assert (
        Settings(_env_file=tmp_path / "does-not-exist").DOCUMENTS_CONTEXTUAL_PREAMBLE_ENABLED
        is False
    )
