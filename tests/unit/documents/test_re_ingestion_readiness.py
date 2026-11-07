"""Re-ingestion readiness inspection gates (openspec ``re-ingestion-readiness``).

Requirement 1 of the capability: re-ingestion stays unexposed — no endpoint
under any API version, no model-callable tool in any role's toolset or the
registry — until every readiness requirement is satisfied by its own change.
These scans turn that requirement into a CI gate: if a surface appears, the
suite fails naming it.

Task 1.2 adds the deterministic-identity probe: same content + same pipeline
version ⇒ identical chunk identity, the stability precondition for every
historical ``EvidenceRef`` surviving a re-run.
"""

from __future__ import annotations

import pytest

from app.ai.tools.policy import ROLE_CAPABILITY_GRANTS
from app.ai.tools.registry import (
    get_tool_registry,
    register_default_tools,
    register_portfolio_traversal_tools,
)
from app.features.documents.application.tools.portfolio import make_portfolio_traversal_tools
from app.features.documents.ingestion.chunking.classification import (
    ClassifiedDocument,
    ParsedDocument,
    segment_chunks,
)

# ---------------------------------------------------------------------------
# Task 1.1 — no re-ingestion surface exists (CI-enforced inspection gate)
# ---------------------------------------------------------------------------

_REINGESTION_MARKERS = ("re-ingest", "reingest", "re-process", "reprocess")


def _all_registered_routes() -> list[str]:
    """Every HTTP path the application's API routers expose."""
    from app.api.v1 import v1_router
    from app.api.v2 import v2_router

    paths: list[str] = []
    for router in (v1_router, v2_router):
        for route in router.routes:
            path = getattr(route, "path", None)
            if isinstance(path, str):
                paths.append(path)
    return paths


def _all_registered_tool_names() -> set[str]:
    """Every tool name the authoritative registry carries (catalog seats)."""
    register_default_tools()
    register_portfolio_traversal_tools(make_portfolio_traversal_tools(None), validate=False)
    return {tool.name for tool in get_tool_registry().all()}


def test_no_api_route_exposes_re_ingestion() -> None:
    hits = [
        path
        for path in _all_registered_routes()
        for marker in _REINGESTION_MARKERS
        if marker in path.casefold()
    ]
    assert not hits, (
        "re-ingestion surface appeared on the HTTP API; the "
        "re-ingestion-readiness capability forbids exposure until its "
        f"requirements are satisfied by their own change: {hits}"
    )


def test_no_model_callable_re_ingestion_tool_exists() -> None:
    names = _all_registered_tool_names()
    hits = [name for name in names for marker in _REINGESTION_MARKERS if marker in name.casefold()]
    assert not hits, (
        "re-ingestion tool appeared in the agent tool registry; the "
        "re-ingestion-readiness capability forbids model-callable exposure: "
        f"{hits}"
    )


def test_no_policy_grant_names_a_re_ingestion_capability() -> None:
    hits = [
        (role, capability)
        for role, capabilities in ROLE_CAPABILITY_GRANTS.items()
        for capability in capabilities
        for marker in _REINGESTION_MARKERS
        if marker in capability.casefold()
    ]
    assert not hits, (
        "a role's capability grant names re-ingestion; exposure is "
        f"approval-gated and must never arrive through policy drift: {hits}"
    )


# ---------------------------------------------------------------------------
# Task 1.2 — deterministic chunk identity under the current pipeline
# ---------------------------------------------------------------------------


def _contract_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        title="Master Services Agreement",
        markdown=(
            "# Master Services Agreement\n\n"
            "## 1. Definitions\n\n"
            "1.1 ``Effective Date`` means the date of last signature.\n\n"
            "## 2. Obligations\n\n"
            "2.1 The Supplier shall deliver the services with professional care.\n\n"
            "2.2 The Customer shall pay invoices within thirty days.\n\n"
            "## 3. Indemnification\n\n"
            "3.1 The Supplier indemnifies the Customer against third-party claims.\n"
        ),
        page_count=1,
    )


async def _chunk_identity_sequence() -> list[tuple[object, ...]]:
    parsed = _contract_parsed_document()
    classified = ClassifiedDocument(document_kind="contracts", parties=[])
    chunks, _warnings = await segment_chunks(parsed=parsed, classified=classified)
    # The chunk identity the store enforces is (document_id, document_version,
    # chunk_index) — build_chunk_rows pins document_version and the upsert keys
    # on (document_id, document_version, chunk_index). What the chunking path
    # itself owes the re-run is a deterministic chunk_index/content sequence.
    return [
        (
            chunk.chunk_index,
            chunk.content,
            chunk.metadata_.get("chunk_policy"),
            chunk.preamble,
        )
        for chunk in chunks
    ]


@pytest.mark.asyncio
async def test_same_content_yields_identical_chunk_identity() -> None:
    """Same content + same pipeline version ⇒ identical chunk identity.

    The chunking path is structure-aware and deterministic (regex
    classification + policy boundaries, zero overlap), so a re-run over the
    same bytes must produce the same identity sequence — the precondition
    that keeps historical ``EvidenceRef``s resolvable after a re-ingestion.
    """
    first = await _chunk_identity_sequence()
    second = await _chunk_identity_sequence()
    assert first, "fixture produced no chunks — the probe is vacuous"
    assert first == second, (
        "chunk identity is not deterministic under the current pipeline: "
        "re-ingestion would re-map every historical EvidenceRef. The "
        "non-deterministic derivation step must be fixed or the re-ingestion "
        "re-mapping rule made explicit before exposure."
    )
