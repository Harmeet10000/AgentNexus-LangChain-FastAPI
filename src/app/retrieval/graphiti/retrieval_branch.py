"""Retrieval-shaped Graphiti read: ranked, tenant-scoped chunk identifiers.

This is the fourth fused branch's read path (``retrieval-measurement-gate``).
It is deliberately distinct from the evidence-policy filter
(``documents.queries._graphiti_filter_chunk_ids``), which stays untouched:

- the filter narrows candidates and degrades to an empty list on any failure;
- this read *ranks*: it preserves ``Graphiti.search``'s relevance order as the
  branch's ranking and, because it is an optional fused branch, a transport
  failure propagates so the ask path can fail the request naming the branch.

Tenant scoping reuses the same server-derived group-id scaffold as every other
Graphiti read: no tenant, no allowlist, or no resolvable partition means no
groups, and no groups means the branch contributes nothing (an empty ranking,
never a widened scope). A graph with no relationships reachable for the query
returns an empty ranking too — the branch contributes nothing rather than
failing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.observability.logging import logger
from app.retrieval.graphiti.client import _scoped_group_ids

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from graphiti_core.graphiti import Graphiti

# Postgres chunk identifiers embedded in episode source descriptions/bodies by
# the ingestion producer. Redeclared locally (a UUID pattern, not logic) so the
# reusable retrieval layer never imports the feature implementation.
_POSTGRES_CHUNK_ID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

DEFAULT_NUM_RESULTS = 20


async def fetch_ranked_chunk_ids(
    graphiti: Graphiti | None,
    query: str,
    *,
    allowed_doc_ids: tuple[str, ...] = (),
    tenant_id: str | None = None,
    contract_index: int | None = None,
    document_versions: Sequence[tuple[str, int, str | None]] = (),
    document_contracts: Sequence[tuple[str, int, tuple[int | None, ...], bool]] = (),
    num_results: int = DEFAULT_NUM_RESULTS,
) -> list[str]:
    """Return ranked chunk identifiers the relationship graph surfaces for a query.

    The ranking is ``Graphiti.search``'s relevance order over the authorized
    partitions, deduplicated by first appearance. Raises on transport failure —
    the enabled branch must fail the request, not silently shrink the fusion —
    and returns ``[]`` whenever no scope is resolvable or nothing matches.
    """
    if graphiti is None or not tenant_id or not allowed_doc_ids:
        return []
    group_ids = _scoped_group_ids(
        allowed_doc_ids,
        tenant_id=tenant_id,
        contract_index=contract_index,
        document_versions=document_versions,
        document_contracts=document_contracts,
    )
    # An allowlist without a resolvable partition reads nothing — never a
    # widened scope, and never an empty group list (providers may read that as
    # "all groups").
    if not group_ids:
        return []
    raw_results: list[Any] = await graphiti.search(
        query=query,
        group_ids=group_ids,
        num_results=num_results,
    )
    ranked: list[str] = []
    for result in raw_results or []:
        source_description = getattr(result, "source_description", "") or ""
        content = getattr(result, "content", "") or getattr(result, "episode_body", "") or ""
        for value in (source_description, content):
            for chunk_id in _POSTGRES_CHUNK_ID_PATTERN.findall(value):
                if chunk_id not in ranked:
                    ranked.append(chunk_id)
    logger.bind(
        service="graphiti",
        query=query[:80],
        group_ids=group_ids,
        identifiers=len(ranked),
    ).info("Graphiti fusion branch ranked chunk identifiers")
    return ranked[:num_results]
