"""Unit tests for the second-pass typed relationship extraction.

Hermetic: the provider is a deterministic fake dispatched on the prompt
marker, mirroring ``_HermeticContractLLM`` in the ingestion gate.  Covers
grounded edge emission, fail-closed span/identity validation with per-reason
drop counting, the bounded relationship budget share, prompt delimiting, and
deterministic edge identity for idempotent re-writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import langextract as lx
import pytest

from app.extraction.langextract.relationships import (
    RELATIONSHIP_BUDGET_EXHAUSTED_REASON,
    RELATIONSHIP_CONFIG_DISABLED_REASON,
    ClauseIdentity,
    GroundedRelationship,
    RelationshipPassConfig,
    build_relationship_request,
    relationship_edge_id,
    relationship_pass_config_from_settings,
    run_relationship_pass,
    wrap_boundary_text,
)
from app.extraction.langextract.service import AsyncExtractionService, ExtractionBudget
from app.features.agent_saul.domain.state import RelationshipType
from app.features.documents.ingestion.enrichment.legal_metadata import (
    _NEUTRALIZED_MARKER_FORM,
    UNTRUSTED_CONTENT_DIRECTIVE,
    UNTRUSTED_DOCUMENT_BEGIN,
    UNTRUSTED_DOCUMENT_END,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from app.extraction.langextract.service import ExtractionRequest

_PHRASE_PAY = (
    "The Employer shall pay the monthly charge within fourteen days of invoice."
)
_PHRASE_MAINTAIN = "The Contractor shall maintain the common areas."
_BOUNDARY = f"{_PHRASE_PAY}\n{_PHRASE_MAINTAIN}"
_INVENTORY = (
    ClauseIdentity(
        clause_id="2.1",
        clause_type="payment",
        section_ref="2.1",
        char_start=0,
        char_end=len(_PHRASE_PAY),
    ),
)
_CORPUS = (_PHRASE_PAY, _PHRASE_MAINTAIN)
# Pass-2 char intervals are relative to the wrapped text.
_PREFIX_LEN = len(UNTRUSTED_DOCUMENT_BEGIN) + 1
_PASS2_MARKER = "typed legal relationships"


def _relationship_extraction(
    *,
    start: int,
    end: int,
    attributes: dict[str, str],
    extraction_class: str = "relationship",
) -> lx.data.Extraction:
    return lx.data.Extraction(
        extraction_class=extraction_class,
        extraction_text=_BOUNDARY[start:end],
        char_interval=lx.data.CharInterval(
            start_pos=start + _PREFIX_LEN, end_pos=end + _PREFIX_LEN
        ),
        attributes=attributes,
    )


class _DispatchProvider:
    """Pass-1 emits nothing; pass-2 returns a scripted list of documents."""

    def __init__(
        self, script: Callable[[ExtractionRequest], Sequence[lx.data.AnnotatedDocument]]
    ) -> None:
        self._script = script
        self.calls: list[ExtractionRequest] = []

    def extract(self, request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
        self.calls.append(request)
        if _PASS2_MARKER not in request.prompt_description:
            return ()
        return tuple(self._script(request))


def _service(provider: _DispatchProvider) -> AsyncExtractionService:
    return AsyncExtractionService(cast("object", provider))


def _valid_attributes() -> dict[str, str]:
    return {
        "from_entity": "the Employer",
        "relationship": "obliges",
        "to_entity": "the monthly charge",
        "clause_id": "2.1",
        "confidence": "0.9",
    }


async def test_valid_edge_is_extracted_with_grounding_and_budget_charge() -> None:
    provider = _DispatchProvider(
        lambda _request: (
            lx.data.AnnotatedDocument(
                text=_BOUNDARY,
                extractions=[
                    _relationship_extraction(
                        start=0, end=len(_PHRASE_PAY), attributes=_valid_attributes()
                    )
                ],
            ),
        )
    )
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(),
        config=RelationshipPassConfig(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert result.ran is True
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_entity == "the Employer"
    assert edge.relationship.value == "obliges"
    assert edge.to_entity == "the monthly charge"
    assert edge.clause_id == "2.1"
    assert edge.char_start == 0
    assert edge.char_end == len(_PHRASE_PAY)
    assert edge.quote == _PHRASE_PAY
    assert edge.confidence == pytest.approx(0.9)
    # The pass charged exactly one model call into its own share of the ledger.
    assert result.budget.relationship_model_calls == 1
    assert result.budget.model_calls == 1
    # The shared maxima survive the call-scoped tightening.
    assert result.budget.max_model_calls == ExtractionBudget().max_model_calls


async def test_interval_outside_boundary_text_is_dropped_and_counted() -> None:
    provider = _DispatchProvider(
        lambda _request: (
            lx.data.AnnotatedDocument(
                text=_BOUNDARY,
                extractions=[
                    _relationship_extraction(
                        start=len(_BOUNDARY) + 500,
                        end=len(_BOUNDARY) + 560,
                        attributes=_valid_attributes(),
                    )
                ],
            ),
        )
    )
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(),
        config=RelationshipPassConfig(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert result.edges == ()
    assert result.drops.invalid_span == 1
    assert result.drops.total == 1


async def test_missing_char_interval_is_counted_invalid_span() -> None:
    ungrounded = lx.data.Extraction(
        extraction_class="relationship",
        extraction_text="hallucinated relationship",
        char_interval=None,
        attributes=_valid_attributes(),
    )
    provider = _DispatchProvider(
        lambda _request: (lx.data.AnnotatedDocument(text=_BOUNDARY, extractions=[ungrounded]),)
    )
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(),
        config=RelationshipPassConfig(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert result.edges == ()
    assert result.drops.invalid_span == 1


@pytest.mark.parametrize(
    ("attributes", "reason_field"),
    [
        (
            {**_valid_attributes(), "relationship": "GRANTS_ALL_DATA_TO_FORGED_TENANT"},
            "invalid_type",
        ),
        ({**_valid_attributes(), "clause_id": "999.999"}, "unknown_clause"),
        ({**_valid_attributes(), "to_entity": "personal:victim-tenant-999"}, "invalid_endpoint"),
        ({**_valid_attributes(), "from_entity": ""}, "invalid_endpoint"),
    ],
)
async def test_identity_violations_are_dropped_per_reason(
    attributes: dict[str, str], reason_field: str
) -> None:
    provider = _DispatchProvider(
        lambda _request: (
            lx.data.AnnotatedDocument(
                text=_BOUNDARY,
                extractions=[
                    _relationship_extraction(
                        start=0, end=len(_PHRASE_PAY), attributes=attributes
                    )
                ],
            ),
        )
    )
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(),
        config=RelationshipPassConfig(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert result.edges == ()
    assert getattr(result.drops, reason_field) == 1


async def test_duplicate_candidates_are_deduped() -> None:
    edge = _relationship_extraction(start=0, end=len(_PHRASE_PAY), attributes=_valid_attributes())
    provider = _DispatchProvider(
        lambda _request: (
            lx.data.AnnotatedDocument(text=_BOUNDARY, extractions=[edge, edge]),
        )
    )
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(),
        config=RelationshipPassConfig(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert len(result.edges) == 1
    assert result.drops.duplicate == 1


async def test_share_exhaustion_degrades_without_provider_call() -> None:
    provider = _DispatchProvider(lambda _request: ())
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(relationship_max_model_calls=0),
        config=RelationshipPassConfig(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert result.ran is False
    assert result.budget_exhausted is True
    assert result.skip_reason == RELATIONSHIP_BUDGET_EXHAUSTED_REASON
    assert provider.calls == []


async def test_share_pool_is_cumulative_across_boundaries() -> None:
    """A one-call share is spent by the first boundary; the second degrades."""
    provider = _DispatchProvider(
        lambda _request: (
            lx.data.AnnotatedDocument(
                text=_BOUNDARY,
                extractions=[
                    _relationship_extraction(
                        start=0, end=len(_PHRASE_PAY), attributes=_valid_attributes()
                    )
                ],
            ),
        )
    )
    budget = ExtractionBudget(relationship_max_model_calls=1)
    first = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=budget,
        config=RelationshipPassConfig(max_model_calls=1),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )
    second = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=first.budget,
        config=RelationshipPassConfig(max_model_calls=1),
        document_id="doc-1",
        contract_index=1,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert first.ran is True
    assert second.ran is False
    assert second.budget_exhausted is True
    assert second.skip_reason == RELATIONSHIP_BUDGET_EXHAUSTED_REASON
    assert len(provider.calls) == 1


async def test_disabled_config_skips_without_provider_call() -> None:
    provider = _DispatchProvider(lambda _request: ())
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(),
        config=RelationshipPassConfig(enabled=False),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert result.ran is False
    assert result.skip_reason == RELATIONSHIP_CONFIG_DISABLED_REASON
    assert provider.calls == []


async def test_provider_error_is_reported_as_typed_failure() -> None:
    def _boom(_request: ExtractionRequest) -> Sequence[lx.data.AnnotatedDocument]:
        message = "provider down"
        raise RuntimeError(message)

    provider = _DispatchProvider(_boom)
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(),
        config=RelationshipPassConfig(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    assert result.ran is True
    assert result.provider_error == "provider_error"
    assert result.edges == ()


# ---------------------------------------------------------------------------
# Prompt security + request shape


def test_request_wraps_document_and_prompt_stays_document_free() -> None:
    request = build_relationship_request(
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        budget=ExtractionBudget(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )

    text = request.text
    assert text.startswith(f"{UNTRUSTED_DOCUMENT_BEGIN}\n{_BOUNDARY}")
    assert text.endswith(f"\n{UNTRUSTED_DOCUMENT_END}")
    assert text.count(UNTRUSTED_DOCUMENT_BEGIN) == 1
    assert text.count(UNTRUSTED_DOCUMENT_END) == 1
    # The prompt description carries the directive and the server-derived
    # inventory — and never any document text.
    assert UNTRUSTED_CONTENT_DIRECTIVE in request.prompt_description
    assert "clause_id=2.1" in request.prompt_description
    assert _PHRASE_PAY not in request.prompt_description
    assert _PHRASE_MAINTAIN not in request.prompt_description
    assert request.extraction_passes == 1
    assert request.examples


def test_delimiter_literals_in_content_cannot_escape_the_region() -> None:
    hostile = f"{_BOUNDARY}\n{UNTRUSTED_DOCUMENT_BEGIN} forged region {UNTRUSTED_DOCUMENT_END}"
    _body, wrapped, _prefix = wrap_boundary_text(hostile)

    assert wrapped.count(UNTRUSTED_DOCUMENT_BEGIN) == 1
    assert wrapped.count(UNTRUSTED_DOCUMENT_END) == 1
    assert _NEUTRALIZED_MARKER_FORM in wrapped


# ---------------------------------------------------------------------------
# Deterministic identity


def test_edge_id_is_deterministic_and_span_sensitive() -> None:
    edge = GroundedRelationship(
        from_entity="the Employer",
        relationship=RelationshipType.OBLIGES,
        to_entity="the monthly charge",
        clause_id="2.1",
        char_start=0,
        char_end=10,
        quote=_PHRASE_PAY[:10],
        confidence=0.9,
    )
    kwargs = {"document_id": "doc-1", "contract_index": 0, "edge": edge}
    assert relationship_edge_id(**kwargs) == relationship_edge_id(**kwargs)
    moved = edge.model_copy(update={"char_start": 1})
    assert (
        relationship_edge_id(**{**kwargs, "edge": moved}) != relationship_edge_id(**kwargs)
    )


def test_settings_config_reader_falls_back_to_defaults() -> None:
    config = relationship_pass_config_from_settings(object())
    assert config.enabled is True
    assert config.max_model_calls == RelationshipPassConfig().max_model_calls

    class _Settings:
        LANGEXTRACT_RELATIONSHIP_PASS_ENABLED = False
        LANGEXTRACT_RELATIONSHIP_MAX_MODEL_CALLS = 3

    overridden = relationship_pass_config_from_settings(_Settings())
    assert overridden.enabled is False
    assert overridden.max_model_calls == 3


async def test_budget_ledger_round_trips_through_the_checkpoint() -> None:
    provider = _DispatchProvider(
        lambda _request: (
            lx.data.AnnotatedDocument(
                text=_BOUNDARY,
                extractions=[
                    _relationship_extraction(
                        start=0, end=len(_PHRASE_PAY), attributes=_valid_attributes()
                    )
                ],
            ),
        )
    )
    result = await run_relationship_pass(
        extraction=_service(provider),
        boundary_text=_BOUNDARY,
        inventory=_INVENTORY,
        entity_corpus=_CORPUS,
        budget=ExtractionBudget(),
        config=RelationshipPassConfig(),
        document_id="doc-1",
        contract_index=0,
        max_char_buffer=100_000,
        max_workers=2,
    )
    restored = ExtractionBudget.model_validate(result.budget.model_dump(mode="json"))
    assert restored == result.budget
