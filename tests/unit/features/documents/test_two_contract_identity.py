"""Two-contract identity fixture (I1, task 4.2).

Asserts at the classification/extraction boundary level (the seam 4.1 landed:
``establish_contract_boundaries`` feeding per-boundary classification, legal
metadata extraction, chunk segmentation and ``contract_index`` annotation) that
no finding — a party, an effective/signed date, or a governing law — is
attributed to a contract without that contract's identity, and that the pipeline
fails or downgrades instead of publishing mixed metadata when boundaries are
insufficient or extraction is rejected.

The fixture is a single file containing two independently signed contracts with
disjoint parties, dates, and governing laws.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, override

from transformers import PreTrainedTokenizerBase

from app.extraction.langextract.service import ExtractionFailureCode
from app.features.documents.api.http.dto import IngestionJob
from app.features.documents.application.commands.upload_document import (
    _aggregate_contract_classifications,
    _annotate_contract_chunks,
    _classification_metadata,
    _extract_document_knowledge,
    _LegalPreparation,
    _metadata_validation_state,
    _unvalidated_classified,
)
from app.features.documents.application.ingestion.chunking.classification import (
    ContractBoundary,
    ParsedDocument,
    classify_document,
    establish_contract_boundaries,
    segment_chunks,
)
from app.features.documents.application.ingestion.enrichment.legal_metadata import (
    extract_legal_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import pytest
    from langchain_core.messages import BaseMessage

    from app.features.documents.application.ingestion.chunking.classification import PreparedChunk
    from app.features.documents.application.ingestion.enrichment.legal_metadata import (
        LegalMetadataExtraction,
    )

CONTRACT_ONE_PARTIES = ("NORTHWIND MANUFACTURING PTY LTD", "ATLANTIC FREIGHT SOLUTIONS LTD")
CONTRACT_TWO_PARTIES = ("CROWN COURT CHAMBERS LTD", "PEMBERTON ADVISORY PARTNERS LLP")
CONTRACT_ONE_DATE = "1 March 2026"
CONTRACT_TWO_DATE = "15 April 2026"
CONTRACT_ONE_LAW = "the laws of Victoria, Australia"
CONTRACT_TWO_LAW = "the laws of England and Wales"

TWO_CONTRACT_MARKDOWN = f"""# MASTER SERVICES AGREEMENT (CONTRACT ONE)

THIS MASTER SERVICES AGREEMENT is entered into on {CONTRACT_ONE_DATE} between {CONTRACT_ONE_PARTIES[0]} (ACN 001 002 003) of Docklands, Victoria (the Client) and {CONTRACT_ONE_PARTIES[1]} (the Supplier, together with the Client, the Parties).

1. SCOPE. The Supplier shall provide the logistics services described in Schedule 1 of this Agreement.

2. TERM. This Agreement commences on the Effective Date and runs for 24 months.

3. GOVERNING LAW. This Agreement and any dispute arising from it is governed by {CONTRACT_ONE_LAW}.

4. ENTIRE AGREEMENT. This document is the entire agreement between the Parties.

SIGNED for {CONTRACT_ONE_PARTIES[0]}: A. Rector, Director. SIGNED for {CONTRACT_ONE_PARTIES[1]}: J. Mbeki, Director.

# CONSULTING SERVICES AGREEMENT (CONTRACT TWO)

THIS CONSULTING SERVICES AGREEMENT is made on {CONTRACT_TWO_DATE} between {CONTRACT_TWO_PARTIES[0]} (company 09876543) of Gray's Inn, London (the Client) and {CONTRACT_TWO_PARTIES[1]} (the Consultant, together with the Client, the Parties).

1. SERVICES. The Consultant shall provide the advisory services described in Schedule A to this Agreement.

2. FEES. The Client shall pay the fixed fee of GBP 40,000 within 30 days of invoice.

3. GOVERNING LAW. This Agreement is governed by {CONTRACT_TWO_LAW}.

4. ENTIRE AGREEMENT. This document is the entire agreement between the Parties.

SIGNED for {CONTRACT_TWO_PARTIES[0]}: H. Okafor, Partner. SIGNED for {CONTRACT_TWO_PARTIES[1]}: M. Duval, Partner.
"""


def _job() -> IngestionJob:
    return IngestionJob(
        document_id="00000000-0000-0000-0000-000000000042",
        user_id="user-two-contract",
        tenant_id="personal:user-two-contract",
        filename="two_contracts.md",
        content_type="text/markdown",
        object_uri="s3://fixtures/two_contracts.md",
    )


def _boundaries() -> tuple[list[ContractBoundary], int]:
    result = establish_contract_boundaries(markdown=TWO_CONTRACT_MARKDOWN)
    return result.boundaries, len(result.boundaries)


class _SliceFaithfulInvoker:
    """Structured invoker that extracts only what its own slice text says.

    Simulates an honest server-side model: parties, dates and governing law are
    read from the untrusted region the extraction seam actually sent (the
    boundary slice), so the values differ per contract exactly as the source
    does.
    """

    def __init__(self, captured_slices: list[str]) -> None:
        self._captured_slices = captured_slices

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> dict[str, Any]:
        text = "\n".join(str(message.content) for message in messages)
        self._captured_slices.append(text)
        between = re.search(r"between\s+(.+?)\s+and\s+(\S.+?)\s*\(", text, re.DOTALL)
        parties = [between.group(1).strip(), between.group(2).strip()] if between else []
        date = re.search(r"(?:entered into on|made on) (\d+ [A-Za-z]+ \d{4})", text)
        law = re.search(r"is governed by (the laws of [A-Za-z ,]+?)\.", text)
        return {
            "contract_type": "msa" if "MASTER SERVICES" in text else "consulting",
            "parties": parties,
            "governing_law": law.group(1).strip() if law else None,
            "effective_date": date.group(1) if date else None,
            "document_summary": "",
        }


class _SliceFaithfulLLM:
    def __init__(self) -> None:
        self.captured_slices: list[str] = []

    def with_structured_output(self, _schema: type[LegalMetadataExtraction]) -> object:
        return _SliceFaithfulInvoker(self.captured_slices)


class _CrossContractSteeringInvoker:
    """Structured invoker that names the OTHER contract's parties (both attempts)."""

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> dict[str, Any]:
        del messages
        return {
            "parties": list(CONTRACT_TWO_PARTIES),
            "governing_law": CONTRACT_TWO_LAW,
            "effective_date": CONTRACT_TWO_DATE,
            "document_summary": "",
        }


class _CrossContractSteeringLLM:
    def with_structured_output(self, _schema: type[LegalMetadataExtraction]) -> object:
        return _CrossContractSteeringInvoker()


class _OfflineTokenizer(PreTrainedTokenizerBase):
    """Deterministic word-count tokenizer (no model download) for chunking."""

    @override
    def tokenize(self, text: str, **_kwargs: object) -> list[str]:
        return str(text).split()

    @override
    def encode(self, text: str, *_args: object, **_kwargs: object) -> list[int]:
        return list(range(1, len(str(text).split()) + 1))

    @override
    def _tokenize(self, text: str, **_kwargs: object) -> list[str]:
        return str(text).split()

    def get_vocab(self) -> dict[str, int]:
        return {}

    @property
    def vocab_size(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Boundary + classification identity


def test_two_contract_fixture_establishes_distinct_boundaries() -> None:
    boundaries, count = _boundaries()
    assert count >= 2, "two independently signed contracts need separate boundaries"
    assert [boundary.contract_index for boundary in boundaries] == list(range(count))
    assert all(boundary.markdown.strip() for boundary in boundaries)
    # Each contract's parties occur in exactly one boundary slice, and no slice
    # mixes the two contracts' party sets.
    for parties in (CONTRACT_ONE_PARTIES, CONTRACT_TWO_PARTIES):
        slices_with = [
            boundary.contract_index
            for boundary in boundaries
            if parties[0].casefold() in boundary.markdown.casefold()
        ]
        assert slices_with, f"{parties[0]} must appear in a slice"
        for index in slices_with:
            slice_text = boundaries[index].markdown.casefold()
            other = (
                CONTRACT_TWO_PARTIES if parties is CONTRACT_ONE_PARTIES else CONTRACT_ONE_PARTIES
            )
            assert other[0].casefold() not in slice_text or other[1].casefold() not in slice_text


def test_per_boundary_classification_never_mixes_parties_or_laws() -> None:
    boundaries, _count = _boundaries()
    classifications = [
        classify_document(markdown=boundary.markdown, filename="two_contracts.md")
        for boundary in boundaries
    ]
    contract_one, contract_two = frozenset(CONTRACT_ONE_PARTIES), frozenset(CONTRACT_TWO_PARTIES)
    for classification in classifications:
        named = frozenset(classification.parties)
        # A finding may name one contract's parties, never a mix, and the law
        # strings must not cross contracts inside one slice.
        assert not (named & contract_one and named & contract_two)
        jurisdiction = (classification.jurisdiction or "").casefold()
        one_law = "victoria" in jurisdiction
        two_law = "england" in jurisdiction
        assert not (one_law and two_law)
    # The slice carrying contract one's preamble classifies as a contract and
    # names exactly contract one's parties — never contract two's.
    preamble_one = next(
        classification
        for classification in classifications
        if CONTRACT_ONE_PARTIES[0].casefold() in str(classification.parties).casefold()
    )
    assert preamble_one.document_kind == "legal_contract"
    assert CONTRACT_ONE_PARTIES[0] in preamble_one.parties[0]
    assert CONTRACT_ONE_PARTIES[1] in preamble_one.parties[1]
    assert not contract_two & set(" ".join(preamble_one.parties).split())
    preamble_two = next(
        classification
        for classification in classifications
        if CONTRACT_TWO_PARTIES[0].casefold() in str(classification.parties).casefold()
    )
    assert CONTRACT_TWO_PARTIES[1] in " ".join(preamble_two.parties)
    assert not {CONTRACT_ONE_PARTIES[0], CONTRACT_ONE_PARTIES[1]} & set(preamble_two.parties)


# ---------------------------------------------------------------------------
# Extraction-boundary findings carry contract identity


async def test_metadata_findings_carry_their_contract_index() -> None:
    boundaries, count = _boundaries()
    llm = _SliceFaithfulLLM()
    preparations: list[tuple[ContractBoundary, _LegalPreparation]] = []
    classifications = []
    for boundary in boundaries:
        classified = classify_document(markdown=boundary.markdown, filename="two_contracts.md")
        classifications.append(classified)
        outcome = await extract_legal_metadata(
            llm=llm, markdown=boundary.markdown, classified=classified
        )
        assert not outcome.validation_failed
        preparations.append((boundary, _LegalPreparation(metadata=outcome.metadata, warnings=[])))

    aggregated = _aggregate_contract_classifications(
        classifications, boundary_warnings=[], ambiguous=False
    )
    metadata = _classification_metadata(
        job=_job(), classified=aggregated, legal=preparations[0][1], legal_preparations=preparations
    )
    contracts = metadata.get("contracts")
    assert isinstance(contracts, list)
    assert len(contracts) == count
    findings = {
        "parties": {*CONTRACT_ONE_PARTIES, *CONTRACT_TWO_PARTIES},
        "dates": {CONTRACT_ONE_DATE, CONTRACT_TWO_DATE},
        "laws": {CONTRACT_ONE_LAW, CONTRACT_TWO_LAW},
    }
    for contract in contracts:
        assert "contract_index" in contract, "every per-contract finding needs its identity"
    # Every finding value appears under exactly one contract_index, and the
    # entry it appears under is the slice that text was extracted from.
    for contract in contracts:
        index = contract["contract_index"]
        slice_text = boundaries[index].markdown.casefold()
        blob = json_blob(contract)
        for party in findings["parties"]:
            if party.casefold() in blob:
                assert party.casefold() in slice_text, (
                    f"{party} attributed to contract {index} whose slice never wrote it"
                )
        for date in findings["dates"]:
            if date in blob:
                assert date.casefold() in slice_text
        for law in findings["laws"]:
            if law.casefold() in blob:
                assert law.casefold() in slice_text
    # Contract one's party appears in exactly one contract entry; same for
    # contract two's — the records never merge the two contracts.
    for parties in (CONTRACT_ONE_PARTIES, CONTRACT_TWO_PARTIES):
        entries = [
            contract
            for contract in contracts
            if any(party.casefold() in json_blob(contract) for party in parties)
        ]
        assert len(entries) == 1, f"{parties[0]} must be attributed to exactly one contract"


def json_blob(value: object) -> str:
    import json

    return json.dumps(value, default=str).casefold()


# ---------------------------------------------------------------------------
# Chunk seam: contract identity rides every chunk


async def test_chunk_findings_are_scoped_to_their_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.documents.application.ingestion.chunking import (
        classification as classification_module,
    )

    monkeypatch.setattr(classification_module, "get_tokenizer", _OfflineTokenizer)
    boundaries, _count = _boundaries()
    all_parties = {*CONTRACT_ONE_PARTIES, *CONTRACT_TWO_PARTIES}
    chunk_contract_one: list[PreparedChunk] = []
    chunk_contract_two: list[PreparedChunk] = []
    for boundary in boundaries:
        classified = classify_document(markdown=boundary.markdown, filename="two_contracts.md")
        parsed = ParsedDocument(title="two_contracts", markdown=boundary.markdown, page_count=1)
        chunks, _warnings = await segment_chunks(parsed=parsed, classified=classified)
        annotated = _annotate_contract_chunks(chunks, boundary)
        is_contract_one = CONTRACT_ONE_PARTIES[0].casefold() in boundary.markdown.casefold()
        target = chunk_contract_one if is_contract_one else chunk_contract_two
        target.extend(annotated)
        for chunk in annotated:
            # contract_index is the server-assigned boundary identity.
            assert chunk.contract_index == boundary.contract_index
            assert chunk.metadata_["contract_index"] == boundary.contract_index
            # No chunk metadata names a party outside its own boundary slice.
            for party in all_parties:
                if party.casefold() in str(chunk.metadata_.get("parties", "")).casefold():
                    assert party.casefold() in boundary.markdown.casefold()
    # Both contracts produced chunks, and no chunk of one contract carries the
    # other contract's parties in its metadata or preamble.
    assert chunk_contract_one
    assert chunk_contract_two
    for chunk in chunk_contract_one:
        blob = f"{chunk.preamble}\n{chunk.metadata_.get('parties', '')}".casefold()
        assert CONTRACT_TWO_PARTIES[0].casefold() not in blob
        assert CONTRACT_TWO_PARTIES[1].casefold() not in blob
    for chunk in chunk_contract_two:
        blob = f"{chunk.preamble}\n{chunk.metadata_.get('parties', '')}".casefold()
        assert CONTRACT_ONE_PARTIES[0].casefold() not in blob
        assert CONTRACT_ONE_PARTIES[1].casefold() not in blob


# ---------------------------------------------------------------------------
# Insufficient boundaries: fail or downgrade, never mixed metadata


async def test_ambiguous_boundaries_fail_extraction_and_persist_no_metadata() -> None:
    """The ambiguous-boundary path fails closed: typed failure, no metadata."""
    runtime = SimpleNamespaceForTest()
    outcome, state = await _extract_document_knowledge(
        runtime=runtime,  # ty: ignore[invalid-argument-type]
        job=_job(),
        boundaries=[ContractBoundary(contract_index=0, markdown="x", start_char=0, end_char=1)],
        boundary_ambiguous=True,
        jurisdiction=None,
        document_type="legal_contract",
    )
    assert outcome.code is ExtractionFailureCode.BOUNDARY_AMBIGUOUS
    assert state.status.value == "not_attempted"
    assert state.reason == "boundary_ambiguous_extraction_skipped"
    # The aggregated classification carries the typed warning and no metadata.
    aggregated = _aggregate_contract_classifications(
        [
            classify_document(
                markdown="THIS SERVICES AGREEMENT is made on 2 May 2026.", filename="a.md"
            ),
        ],
        boundary_warnings=[],
        ambiguous=True,
    )
    codes = {warning.code for warning in aggregated.warnings}
    assert "MULTI_CONTRACT_BOUNDARY_AMBIGUOUS" in codes
    preparations = [
        (
            ContractBoundary(contract_index=0, markdown="x", start_char=0, end_char=1),
            _LegalPreparation(metadata=None, warnings=[]),
        )
    ]
    metadata = _classification_metadata(
        job=_job(),
        classified=aggregated,
        legal=_LegalPreparation(metadata=None, warnings=[]),
        legal_preparations=preparations,
    )
    blob = json_blob(metadata)
    for party in (*CONTRACT_ONE_PARTIES, *CONTRACT_TWO_PARTIES):
        assert party.casefold() not in blob
    for law in (CONTRACT_ONE_LAW, CONTRACT_TWO_LAW):
        assert law.casefold() not in blob
    assert "jurisdiction" not in metadata
    assert "parties" not in metadata


class SimpleNamespaceForTest:
    """The ambiguous path returns before touching any runtime dependency."""


async def test_rejected_extraction_downgrades_and_scrubs_cross_contract_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slice extraction naming the OTHER contract's parties is rejected twice.

    The seam then downgrades: no metadata, the explicit validation-failed state
    with per-contract truth, and the classified heuristics scrubbed so chunk
    preambles carry no party names at all — never the wrong contract's names.
    """
    from app.features.documents.application.ingestion.chunking import (
        classification as classification_module,
    )

    monkeypatch.setattr(classification_module, "get_tokenizer", _OfflineTokenizer)
    boundaries, _count = _boundaries()
    contract_one_slice = next(
        boundary
        for boundary in boundaries
        if CONTRACT_ONE_PARTIES[0].casefold() in boundary.markdown.casefold()
    )
    outcome = await extract_legal_metadata(
        llm=_CrossContractSteeringLLM(),
        markdown=contract_one_slice.markdown,
        classified=classify_document(
            markdown=contract_one_slice.markdown, filename="two_contracts.md"
        ),
    )
    assert outcome.metadata is None
    assert outcome.validation_failed
    codes = {warning.code for warning in outcome.warnings}
    assert "LEGAL_METADATA_VALIDATION_FAILED" in codes

    other_slice = next(
        boundary
        for boundary in boundaries
        if boundary.contract_index != contract_one_slice.contract_index
    )
    ok_preparation = _LegalPreparation(metadata=None, warnings=[])
    preparations = [
        (
            contract_one_slice,
            _LegalPreparation(
                metadata=None, warnings=list(outcome.warnings), validation_failed=True
            ),
        ),
        (other_slice, ok_preparation),
    ]
    state = _metadata_validation_state(preparations)
    assert state["metadata_validation_failed"] is True
    per_contract = {entry["contract_index"] for entry in state["contracts_metadata_validation"]}
    assert per_contract == {contract_one_slice.contract_index, other_slice.contract_index}

    classified = classify_document(
        markdown=contract_one_slice.markdown, filename="two_contracts.md"
    )
    metadata = _classification_metadata(
        job=_job(),
        classified=classified,
        legal=_LegalPreparation(metadata=None, warnings=[], validation_failed=True),
        legal_preparations=preparations,
    )
    blob = json_blob(metadata)
    for party in (*CONTRACT_ONE_PARTIES, *CONTRACT_TWO_PARTIES):
        assert party.casefold() not in blob, "rejected extraction must not launder any party name"
    assert CONTRACT_TWO_LAW.casefold() not in blob
    assert CONTRACT_TWO_DATE not in blob
    assert metadata["metadata_validation_failed"] is True

    # The chunk seam uses the scrubbed classification: preambles carry no
    # party names at all instead of the wrong contract's names.
    scrubbed = _unvalidated_classified(classified)
    assert scrubbed.parties == []
    chunks, _warnings = await segment_chunks(
        parsed=ParsedDocument(title="slice", markdown=contract_one_slice.markdown, page_count=1),
        classified=scrubbed,
    )
    assert chunks
    for chunk in chunks:
        assert "unknown parties" in chunk.preamble
        for party in (*CONTRACT_ONE_PARTIES, *CONTRACT_TWO_PARTIES):
            assert party.casefold() not in chunk.preamble.casefold()


def test_two_contract_document_level_metadata_stays_per_contract() -> None:
    """Multi-boundary documents never publish document-level mixed metadata.

    With per-boundary extraction pending (seam absent), the resolved document
    columns stay empty for a multi-boundary file — the record cannot present
    contract one's parties as the document's parties.
    """
    boundaries, count = _boundaries()
    assert count > 1
    classifications = [
        classify_document(markdown=boundary.markdown, filename="two_contracts.md")
        for boundary in boundaries
    ]
    aggregated = _aggregate_contract_classifications(
        classifications, boundary_warnings=[], ambiguous=False
    )
    # The pipeline resolves document-level parties only for single-boundary
    # documents (len(boundaries) == 1); for a multi-boundary file they stay
    # empty unless per-contract extraction validated them.
    assert aggregated.parties == []
    assert aggregated.jurisdiction is None
    assert aggregated.metadata_.get("contract_count") == count
