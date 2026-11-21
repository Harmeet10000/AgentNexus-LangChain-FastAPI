"""Validation-seam tests for legal metadata extraction.

Covers the mechanical boundary-text gate, the exactly-one feedback-carrying
retry, the honest-absence outcome on terminal failure (no heuristic merge),
and the persisted validation-failed state helpers.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from loguru import logger as loguru_logger

from app.features.documents.application.ingestion.chunking.classification import (
    ClassifiedDocument,
    ContractBoundary,
)

if TYPE_CHECKING:
    from typing import Any
from app.features.documents.application.ingestion.enrichment.legal_metadata import (
    JURISDICTION_BOUNDARY_FINDING_CODE,
    LEGAL_METADATA_VALIDATION_SEAM,
    PARTY_BOUNDARY_FINDING_CODE,
    LegalMetadataExtraction,
    enrich_legal_chunks,
    extract_legal_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.features.documents.application.ingestion.chunking.classification import PreparedChunk

_BOUNDARY = (
    "# Services Agreement\n\n"
    "This Agreement is made between Acme Corp and GlobalTech Ltd.\n\n"
    "Governing law: England and Wales.\n"
)

_GROUNDED = {
    "contract_type": "services_agreement",
    "parties": ["Acme Corp", "GlobalTech Ltd"],
    "jurisdiction": "England and Wales",
    "document_summary": "Services between Acme Corp and GlobalTech Ltd.",
}

_FABRICATED_PARTY = {
    "contract_type": "services_agreement",
    "parties": ["Forged Holdings LLC", "GlobalTech Ltd"],
    "jurisdiction": "England and Wales",
}


def _classified() -> ClassifiedDocument:
    return ClassifiedDocument(
        document_kind="legal_contract",
        jurisdiction="heuristic jurisdiction",
        contract_type="heuristic contract",
        parties=["Heuristic Party"],
        graphiti_required=True,
    )


class _ScriptedLLM:
    """Structured-output double: replays payloads, records every message list."""

    def __init__(self, payloads: Sequence[dict[str, Any] | Exception]) -> None:
        self._payloads = list(payloads)
        self.calls: list[list[Any]] = []

    def with_structured_output(self, _schema: type[LegalMetadataExtraction]) -> object:
        return self

    async def ainvoke(self, messages: list[Any]) -> object:
        self.calls.append(messages)
        payload = self._payloads[min(len(self.calls) - 1, len(self._payloads) - 1)]
        if isinstance(payload, Exception):
            raise payload
        return payload


def _captured_decision_events() -> tuple[list[dict[str, Any]], ExitStack]:
    """Capture ``validation_decision`` events emitted while the block runs."""
    events: list[dict[str, Any]] = []
    stack = ExitStack()

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] == "validation_decision":
            events.append(dict(record["extra"]))

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    stack.callback(loguru_logger.remove, sink_id)
    return events, stack


async def test_grounded_extraction_passes_with_a_single_call() -> None:
    llm = _ScriptedLLM([_GROUNDED])
    outcome = await extract_legal_metadata(llm=llm, markdown=_BOUNDARY, classified=_classified())

    assert len(llm.calls) == 1
    assert outcome.validation_failed is False
    assert outcome.metadata is not None
    assert outcome.metadata.parties == ["Acme Corp", "GlobalTech Ltd"]
    assert outcome.metadata.jurisdiction == "England and Wales"


async def test_grounded_extraction_is_not_merged_with_heuristics() -> None:
    """A validated extraction is used as-is: the heuristic merge is gone."""
    llm = _ScriptedLLM([_GROUNDED])
    outcome = await extract_legal_metadata(llm=llm, markdown=_BOUNDARY, classified=_classified())

    assert outcome.metadata is not None
    # The classified heuristics never appear in the validated record.
    assert "Heuristic Party" not in outcome.metadata.parties
    assert outcome.metadata.jurisdiction != "heuristic jurisdiction"
    assert outcome.metadata.contract_type != "heuristic contract"


async def test_fabricated_party_is_retried_once_with_structured_feedback() -> None:
    llm = _ScriptedLLM([_FABRICATED_PARTY, _GROUNDED])
    outcome = await extract_legal_metadata(llm=llm, markdown=_BOUNDARY, classified=_classified())

    assert len(llm.calls) == 2
    assert outcome.validation_failed is False
    assert outcome.metadata is not None
    assert outcome.metadata.parties == ["Acme Corp", "GlobalTech Ltd"]
    # The retry carried the finding as structured feedback, after the
    # untrusted region (server-authored, document-free).
    retry_messages = llm.calls[1]
    feedback_text = str(retry_messages[-1].content)
    assert f"[{PARTY_BOUNDARY_FINDING_CODE}]" in feedback_text
    assert "Forged Holdings LLC" in feedback_text
    assert "BEGIN UNTRUSTED DOCUMENT CONTENT" not in feedback_text


async def test_fabricated_party_twice_fails_with_honest_absence() -> None:
    llm = _ScriptedLLM([_FABRICATED_PARTY, _FABRICATED_PARTY])
    events, stack = _captured_decision_events()
    with stack:
        outcome = await extract_legal_metadata(
            llm=llm, markdown=_BOUNDARY, classified=_classified()
        )

    # Exactly one feedback-carrying retry, then terminal failure.
    assert len(llm.calls) == 2
    assert outcome.metadata is None
    assert outcome.validation_failed is True
    assert [warning.code for warning in outcome.warnings] == ["LEGAL_METADATA_VALIDATION_FAILED"]
    # One seam-named decision event, ESCALATE, for the terminal outcome.
    assert [event for event in events if event.get("seam") == LEGAL_METADATA_VALIDATION_SEAM] == [
        {
            "seam": LEGAL_METADATA_VALIDATION_SEAM,
            "decision": "escalate",
            "reason": "attempts_exhausted",
            "attempts": 2,
        }
    ]


async def test_fabricated_jurisdiction_is_caught_and_fails_honestly() -> None:
    fabricated = {**_GROUNDED, "jurisdiction": "The Atlantis Free Zone"}
    llm = _ScriptedLLM([fabricated, fabricated])
    outcome = await extract_legal_metadata(llm=llm, markdown=_BOUNDARY, classified=_classified())

    assert len(llm.calls) == 2
    assert outcome.metadata is None
    assert outcome.validation_failed is True
    feedback_text = str(llm.calls[1][-1].content)
    assert f"[{JURISDICTION_BOUNDARY_FINDING_CODE}]" in feedback_text


async def test_seam_absent_is_not_extracted_not_failed() -> None:
    """No structured-output capability: not extracted, not validation-failed."""
    outcome = await extract_legal_metadata(
        llm=object(), markdown=_BOUNDARY, classified=_classified()
    )

    assert outcome.metadata is None
    assert outcome.validation_failed is False
    assert outcome.warnings == []


async def test_provider_failure_is_validation_failed_never_heuristic_merge() -> None:
    llm = _ScriptedLLM([ValueError("transport down")])
    outcome = await extract_legal_metadata(llm=llm, markdown=_BOUNDARY, classified=_classified())

    assert outcome.metadata is None
    assert outcome.validation_failed is True
    assert [warning.code for warning in outcome.warnings] == ["LEGAL_METADATA_VALIDATION_FAILED"]


async def test_enrichment_never_runs_for_a_failed_extraction() -> None:
    """A failed extraction carries no metadata, so nothing can be merged."""
    llm = _ScriptedLLM([_FABRICATED_PARTY, _FABRICATED_PARTY])
    outcome = await extract_legal_metadata(llm=llm, markdown=_BOUNDARY, classified=_classified())

    assert outcome.metadata is None
    chunks: list[PreparedChunk] = []
    # The ingestion path only calls enrich_legal_chunks when metadata exists;
    # with the failed outcome there is nothing to enrich with.
    if outcome.metadata is not None:
        chunks = enrich_legal_chunks(
            chunks=chunks, classified=_classified(), metadata=outcome.metadata
        )
    assert chunks == []


# ---------------------------------------------------------------------------
# Persisted honest-failure state (upload_document persistence model)


def _preparation(metadata: LegalMetadataExtraction | None, *, failed: bool) -> Any:
    from app.features.documents.application.queries.documents import _LegalPreparation

    return _LegalPreparation(metadata=metadata, warnings=[], validation_failed=failed)


def _boundary(index: int) -> Any:
    return ContractBoundary(contract_index=index, markdown="x", start_char=0, end_char=1)


def test_metadata_validation_state_flags_a_single_failed_boundary() -> None:
    from app.features.documents.application.commands.upload_document import (
        _metadata_validation_state,
    )

    preparations = [(_boundary(0), _preparation(None, failed=True))]
    assert _metadata_validation_state(preparations) == {"metadata_validation_failed": True}


def test_metadata_validation_state_carries_per_contract_truth_for_multi_boundary() -> None:
    from app.features.documents.application.commands.upload_document import (
        _metadata_validation_state,
    )

    grounded = LegalMetadataExtraction(parties=["Acme Corp"])
    preparations = [
        (_boundary(0), _preparation(grounded, failed=False)),
        (_boundary(1), _preparation(None, failed=True)),
    ]
    state = _metadata_validation_state(preparations)
    assert state["metadata_validation_failed"] is True
    assert state["contracts_metadata_validation"] == [
        {"contract_index": 0, "metadata_validation_failed": False},
        {"contract_index": 1, "metadata_validation_failed": True},
    ]


def test_metadata_validation_state_is_absent_without_failures() -> None:
    from app.features.documents.application.commands.upload_document import (
        _metadata_validation_state,
    )

    grounded = LegalMetadataExtraction(parties=["Acme Corp"])
    preparations = [
        (_boundary(0), _preparation(grounded, failed=False)),
        (_boundary(1), _preparation(None, failed=False)),
    ]
    assert _metadata_validation_state(preparations) == {}


@pytest.mark.parametrize(
    ("parties", "jurisdiction", "expected_failed"),
    [
        (["Acme Corp"], "England and Wales", False),
        (["acme corp"], "England and Wales", False),
        (["Acme Corp Ltd"], "England and Wales", True),
        (["Acme Corp"], "Atlantis", True),
    ],
)
async def test_boundary_gate_is_substring_casefold_occurrence(
    parties: list[str], jurisdiction: str, expected_failed: bool
) -> None:
    payload = {**_GROUNDED, "parties": parties, "jurisdiction": jurisdiction}
    llm = _ScriptedLLM([payload])
    outcome = await extract_legal_metadata(llm=llm, markdown=_BOUNDARY, classified=_classified())

    assert outcome.validation_failed is expected_failed


# ---------------------------------------------------------------------------
# Rejected heuristics never reach searchable metadata (regression)


def _heuristic_classified() -> ClassifiedDocument:
    """Mirror classify_document's legal-contract output with heuristic fields."""
    return ClassifiedDocument(
        document_kind="legal_contract",
        jurisdiction="heuristic jurisdiction",
        contract_type="heuristic contract",
        parties=["Heuristic Party"],
        metadata_={"jurisdiction": "heuristic jurisdiction", "contract_type": "heuristic contract"},
        graphiti_required=True,
    )


def test_validation_failed_drops_heuristics_from_document_metadata() -> None:
    from app.features.documents.application.commands.upload_document import (
        _classification_metadata,
    )

    job = SimpleNamespace(content_type="text/markdown", filename="agreement.md")
    preparations = [(_boundary(0), _preparation(None, failed=True))]

    metadata = _classification_metadata(
        job=job,
        classified=_heuristic_classified(),
        legal=preparations[0][1],
        legal_preparations=preparations,
    )

    assert metadata["metadata_validation_failed"] is True
    # No rejected heuristic string survives anywhere in the record.
    assert "heuristic jurisdiction" not in str(metadata)
    assert "heuristic contract" not in str(metadata)


def test_not_extracted_keeps_heuristics_distinguishing_it_from_validation_failed() -> None:
    """The not-extracted state keeps the classifier's fields; only a terminal
    validation failure scrubs them, so consumers can tell the two apart."""
    from app.features.documents.application.commands.upload_document import (
        _classification_metadata,
    )

    job = SimpleNamespace(content_type="text/markdown", filename="agreement.md")
    preparations = [(_boundary(0), _preparation(None, failed=False))]

    metadata = _classification_metadata(
        job=job,
        classified=_heuristic_classified(),
        legal=preparations[0][1],
        legal_preparations=preparations,
    )

    assert "metadata_validation_failed" not in metadata
    assert metadata["jurisdiction"] == "heuristic jurisdiction"
    assert metadata["contract_type"] == "heuristic contract"


async def test_validation_failed_chunks_carry_no_heuristic_metadata() -> None:
    from app.features.documents.application.commands.upload_document import (
        _unvalidated_classified,
    )
    from app.features.documents.application.ingestion.chunking.classification import segment_chunks
    from app.features.documents.application.ingestion.parsing.parser import parse_document

    parsed = await parse_document(
        raw_bytes=_BOUNDARY.encode(), filename="agreement.md", content_type="text/markdown"
    )
    stripped = _unvalidated_classified(_heuristic_classified())

    chunks, _warnings = await segment_chunks(parsed=parsed, classified=stripped)
    assert chunks
    for chunk in chunks:
        assert "Heuristic Party" not in str(chunk.metadata_)
        assert "heuristic jurisdiction" not in str(chunk.metadata_)
        assert "heuristic contract" not in str(chunk.metadata_)
        assert "Heuristic Party" not in chunk.preamble
        assert "heuristic jurisdiction" not in chunk.preamble
    # The document kind still drives chunk typing and policy.
    assert all(chunk.chunk_kind == "legal_contract" for chunk in chunks)

    # Sanity: without the strip, the heuristics do reach chunk metadata and
    # the preamble — so the assertions above are load-bearing.
    unstripped, _warnings = await segment_chunks(parsed=parsed, classified=_heuristic_classified())
    assert any("Heuristic Party" in str(chunk.metadata_) for chunk in unstripped)
    assert any("Heuristic Party" in chunk.preamble for chunk in unstripped)


def test_context_pass_receives_the_stripped_classification_when_validation_failed() -> None:
    """The contextual-preamble pass is subject to the same rejected-metadata
    rule as segmentation and the status writes: on a terminal validation
    failure the summaries are generated from the stripped classification, so a
    rejected party/jurisdiction/contract_type cannot re-enter the searchable
    preamble through a context sentence."""
    from app.features.documents.application.commands.upload_document import (
        _context_classified,
    )

    context_classified = _context_classified(
        _heuristic_classified(), _preparation(None, failed=True)
    )

    assert context_classified.parties == []
    assert context_classified.jurisdiction is None
    assert context_classified.contract_type is None
    # document_kind survives — it drives chunk typing and policy.
    assert context_classified.document_kind == "legal_contract"


def test_context_pass_keeps_the_classification_without_a_failed_extraction() -> None:
    """A plain not-extracted (or validated) outcome keeps the classifier's
    fields for the context pass — only a terminal failure scrubs them."""
    from app.features.documents.application.commands.upload_document import (
        _context_classified,
    )

    context_classified = _context_classified(
        _heuristic_classified(), _preparation(None, failed=False)
    )

    assert context_classified.parties == ["Heuristic Party"]
    assert context_classified.jurisdiction == "heuristic jurisdiction"
    assert context_classified.contract_type == "heuristic contract"
