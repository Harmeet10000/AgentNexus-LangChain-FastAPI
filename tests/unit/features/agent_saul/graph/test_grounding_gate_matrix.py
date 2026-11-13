"""Grounding-verification gate test matrix (task 6.6).

The gate landed in 6.5; this suite is its acceptance matrix: adjacent-clause,
definition, exception, absent-span, forged-handle, non-indexed-but-canonical,
multi-contract, and Graphiti-outage cases. The invariant under every case: no
path advances a finding on a ``verified`` LLM verdict alone — every quote is
re-read from the authorized canonical chunk and every handle is mechanically
re-checked first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from returns.result import Success

from app.features.agent_saul.adapters.document_source import (
    DOCUMENT_SOURCE_CONFIG_KEY,
    CanonicalDocumentChunk,
)
from app.features.agent_saul.domain.state import (
    Citation,
    ComplianceFinding,
    ComplianceOutput,
    ComplianceStatus,
    EvidenceRef,
    GroundingStatus,
    GroundingVerificationOutput,
    LegalAgentState,
    RiskAnalysisOutput,
    RiskLabel,
    WorkflowStatus,
)
from app.features.agent_saul.graph.nodes import make_grounding_verification_node

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

pytestmark = pytest.mark.unit

_CHUNKS: tuple[CanonicalDocumentChunk, ...] = (
    CanonicalDocumentChunk(
        chunk_id="chunk-0",
        document_id="doc-a",
        document_version=1,
        chunk_index=0,
        content='"Confidential Information" shall mean any information disclosed by a party.',
        page_no=1,
        clause_type="confidentiality",
        locus="3.1",
    ),
    CanonicalDocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-a",
        document_version=1,
        chunk_index=1,
        content="The Supplier shall indemnify the Customer against all losses.",
        page_no=2,
        clause_type="indemnity",
        locus="7.1",
    ),
    CanonicalDocumentChunk(
        chunk_id="chunk-2",
        document_id="doc-a",
        document_version=1,
        chunk_index=2,
        content="Except as provided in clause 12, liability is limited to the fees paid.",
        page_no=2,
        clause_type="limitation_of_liability",
        locus="7.2",
    ),
)


class _CanonicalSource:
    async def fetch_segments(
        self, *, user_id: str, doc_ids: tuple[str, ...], tenant_id: str | None = None
    ) -> object:
        return Success(_CHUNKS)


def _ref(evidence_id: str, chunk_id: str, *, graph_indexed: bool = False) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        document_id="doc-a",
        document_version=1,
        contract_index=0,
        page_no=1,
        graph_indexed=graph_indexed,
        retrieval_method="hybrid_rrf",
        score=0.9,
        rank=1,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        quote_digest=None,
    )


def _finding(finding_id: str, citation: Citation) -> ComplianceFinding:
    return ComplianceFinding(
        finding_id=finding_id,
        clause_id="canonical:doc-a:v1:chunk:1",
        statute="Contract Act s73",
        is_compliant=True,
        explanation="supported",
        citations=[citation],
    )


def _state(
    citations: Sequence[Citation],
    refs: dict[str, EvidenceRef],
    *,
    evidence_policy: str = "mechanically_verified",
) -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "doc_id": "doc-a",
        "allowed_doc_ids": ["doc-a"],
        "tenant_id": "tenant-1",
        "evidence_checkpoint_id": "checkpoint-1",
        "evidence_registry": refs,
        "evidence_policy": evidence_policy,
        "coverage": {},
        "risk_analysis": RiskAnalysisOutput(
            findings=[], overall_label=RiskLabel.LOW, summary="assessed"
        ),
        "compliance_result": ComplianceOutput(
            findings=[
                _finding(f"finding-{number}", citation) for number, citation in enumerate(citations)
            ],
            jurisdiction="India",
            summary="assessed",
            compliance_status=ComplianceStatus.COMPLIANT,
        ),
        "segments": [],
        "extracted_entities": [],
        "relationships": [],
        "claim_ledger": [],
        "status": WorkflowStatus.VERIFYING_GROUNDING,
    }


def _config() -> dict[str, Any]:
    return {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: _CanonicalSource()}}


def _quote_issues(state: dict[str, Any]) -> list[str]:
    import asyncio

    from app.features.agent_saul.graph.nodes import _verify_quotes_against_canonical_source

    return asyncio.run(_verify_quotes_against_canonical_source(state, _config()))


def test_quote_from_the_adjacent_clause_is_rejected() -> None:
    """The indemnity quote cited against the definition chunk is not a span."""
    citation = Citation(
        claim="definition covers indemnity",
        source="evh-0",
        evidence_id="evh-0",
        confidence=0.9,
        quoted_text="The Supplier shall indemnify the Customer",
    )
    state = _state([citation], {"evh-0": _ref("evh-0", "chunk-0")})

    issues = _quote_issues(state)

    assert issues == ["Quote citation evh-0 is not a span of its canonical chunk."]


def test_quote_inside_a_definition_clause_is_accepted() -> None:
    citation = Citation(
        claim="definition is quoted exactly",
        source="evh-0",
        evidence_id="evh-0",
        confidence=0.9,
        quoted_text='"Confidential Information" shall mean any information disclosed',
    )
    state = _state([citation], {"evh-0": _ref("evh-0", "chunk-0")})

    assert _quote_issues(state) == []


def test_quote_inside_an_exception_clause_is_accepted() -> None:
    citation = Citation(
        claim="exception is quoted exactly",
        source="evh-2",
        evidence_id="evh-2",
        confidence=0.9,
        quoted_text="Except as provided in clause 12, liability is limited",
    )
    state = _state([citation], {"evh-2": _ref("evh-2", "chunk-2")})

    assert _quote_issues(state) == []


def test_absent_span_is_rejected() -> None:
    citation = Citation(
        claim="fabricated span",
        source="evh-1",
        evidence_id="evh-1",
        confidence=0.9,
        quoted_text="This sentence appears nowhere in the contract.",
    )
    state = _state([citation], {"evh-1": _ref("evh-1", "chunk-1")})

    issues = _quote_issues(state)

    assert issues == ["Quote citation evh-1 is not a span of its canonical chunk."]


def test_forged_handle_is_rejected() -> None:
    citation = Citation(
        claim="forged handle",
        source="evh-404",
        evidence_id="evh-404",
        confidence=0.9,
        quoted_text="The Supplier shall indemnify the Customer",
    )
    state = _state([citation], {"evh-1": _ref("evh-1", "chunk-1")})

    issues = _quote_issues(state)

    # With no authorized reference at all, the quote re-read cannot dereference
    # anything and fails closed on the missing canonical reference.
    assert issues == ["Quoted evidence has no authorized canonical document reference."]


def test_non_indexed_but_canonical_evidence_passes() -> None:
    """Graphiti coverage is not legal truth: canonical chunks suffice."""
    citation = Citation(
        claim="canonical quote",
        source="evh-1",
        evidence_id="evh-1",
        confidence=0.9,
        quoted_text="The Supplier shall indemnify the Customer against all losses.",
    )
    state = _state([citation], {"evh-1": _ref("evh-1", "chunk-1", graph_indexed=False)})

    assert _quote_issues(state) == []


def test_multi_contract_identity_mismatch_is_rejected() -> None:
    citation = Citation(
        claim="cross-contract citation",
        source="evh-1",
        evidence_id="evh-1",
        confidence=0.9,
        quoted_text="The Supplier shall indemnify the Customer",
        contract_index=1,
    )
    state = _state([citation], {"evh-1": _ref("evh-1", "chunk-1")})

    from app.features.agent_saul.graph.nodes import _finding_evidence_issues

    issues = _finding_evidence_issues(state, [citation], "finding-0")

    assert "Finding finding-0 cites a different contract identity." in issues


def test_graphiti_outage_degrades_the_strict_policy_but_preserves_canonical() -> None:
    """Outage visibility: graph_indexed_only fails closed; canonical survives."""
    citation = Citation(
        claim="canonical quote during outage",
        source="evh-1",
        evidence_id="evh-1",
        confidence=0.9,
        quoted_text="The Supplier shall indemnify the Customer",
    )
    canonical_policy_state = _state(
        [citation], {"evh-1": _ref("evh-1", "chunk-1", graph_indexed=False)}
    )
    assert _quote_issues(canonical_policy_state) == []

    strict_state = _state(
        [citation],
        {"evh-1": _ref("evh-1", "chunk-1", graph_indexed=False)},
        evidence_policy="graph_indexed_only",
    )
    from app.features.agent_saul.graph.nodes import _finding_evidence_issues

    issues = _finding_evidence_issues(strict_state, [citation], "finding-0")
    assert "Finding finding-0 cites evidence that is not Graphiti indexed." in issues


# --- no path advances on ``verified`` alone ---------------------------------


class _ClaimingVerifier:
    """A verifier that asserts ``verified=True`` regardless of the quotes."""

    async def ainvoke(self, _input: Any) -> Any:
        return GroundingVerificationOutput(
            verified=True,
            unverified_claims=[],
            notes="model said verified",
            citations=[
                Citation(claim="cited", source="evh-1", evidence_id="evh-1", confidence=0.95)
            ],
        )


def _wired_node_graph() -> Any:
    builder = StateGraph(LegalAgentState)
    builder.add_node(
        "grounding_verification", make_grounding_verification_node(_ClaimingVerifier())
    )
    builder.add_edge(START, "grounding_verification")
    builder.add_edge("grounding_verification", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_absent_span_blocks_a_verified_llm_verdict() -> None:
    citation = Citation(
        claim="fabricated span",
        source="evh-1",
        evidence_id="evh-1",
        confidence=0.9,
        quoted_text="This sentence appears nowhere in the contract.",
    )
    state = _state([citation], {"evh-1": _ref("evh-1", "chunk-1")})

    result = asyncio_run_node(state)
    grounding = result["grounding"]

    assert grounding.verified is False
    assert grounding.status is not GroundingStatus.VERIFIED
    assert any("not a span" in claim for claim in grounding.unverified_claims)


def asyncio_run_node(state: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    async def _run() -> dict[str, Any]:
        graph = _wired_node_graph()
        return await graph.ainvoke(
            state,
            {
                "configurable": {
                    "thread_id": "thread-matrix",
                    DOCUMENT_SOURCE_CONFIG_KEY: _CanonicalSource(),
                }
            },
        )

    return asyncio.run(_run())


def test_unsupported_finding_blocks_a_verified_llm_verdict() -> None:
    """A verified verdict cannot rescue a finding with a forged handle."""
    citation = Citation(
        claim="forged handle",
        source="evh-404",
        evidence_id="evh-404",
        confidence=0.9,
    )
    state = _state([citation], {"evh-1": _ref("evh-1", "chunk-1")})

    result = asyncio_run_node(state)
    grounding = result["grounding"]

    assert grounding.verified is False
    assert result["errors"][0].code == "GROUNDING_GATE_REJECTED"
