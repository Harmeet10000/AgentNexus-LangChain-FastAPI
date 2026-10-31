"""Edge-aware context expansion over typed legal relationship edges.

The read-side twin of the pass-2 relationship producer: once
``GroundedRelationship`` edges are persisted (as ``LegalEdgeInput`` episodes
with ``char_start``/``char_end`` grounding), retrieval can expand a retrieved
chunk's context along the edges that touch it — e.g. the parties an OBLIGES
edge binds for a retrieved obligation clause.

This module is the pure expansion core and is deliberately free of provider
and domain imports: edges arrive as ``LegalEdgeRecord`` values, seeds as
``EdgeSeedChunk`` values, and the caller owns scoping.  Guarantees:

- **Deterministic**: blocks are ordered by the edge's grounded span, then by
  clause/entity identity; identical inputs always yield identical expansions.
- **Token-bounded**: per-seed budget (``max_expansion_tokens``); a block that
  does not fit is skipped, never truncated mid-span, and a global used-edge
  set guarantees one edge expands at most one seed (the first in seed order,
  i.e. the best-ranked one).
- **Span-cited**: every expansion block names the edge id and the char
  interval grounding the relationship, so a citation can be verified against
  the source boundary text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.retrieval.token_counter import CountTokens

# Entities shorter than this never match by occurrence: a one- or two-character
# entity name would match nearly every chunk and silently widen expansion.
_MIN_ENTITY_OCCURRENCE_CHARS: int = 3
_DEFAULT_MAX_BLOCKS_PER_SEED: int = 4


class LegalEdgeRecord(BaseModel):
    """One persisted typed relationship edge as the read path sees it.

    ``relationship`` stays a string rather than importing the producer's
    ``RelationshipType`` enum: the reader validates the closed vocabulary at
    the parse boundary (``graphiti.client``) and this module treats the value
    as opaque, cited text.
    """

    model_config = ConfigDict(frozen=True)

    edge_id: str
    document_id: str
    from_entity: str
    relationship: str
    to_entity: str
    clause_id: str
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    group_id: str | None = None
    # Contract/version scoping for bundled documents: an edge citing the wrong
    # contract is worse than no edge, so known mismatches never match.
    contract_index: int | None = Field(default=None, ge=0)
    document_version: int | None = Field(default=None, ge=1)


class EdgeSeedChunk(BaseModel):
    """One retrieved chunk eligible for edge expansion."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    content: str
    # Clause identities the chunk is known to carry (e.g. from chunk
    # metadata). Empty means only entity occurrence can match.
    clause_ids: tuple[str, ...] = ()
    contract_index: int | None = None
    document_version: int | None = None


class EdgeContextBlock(BaseModel):
    """One edge-grounded expansion block attached to a seed chunk."""

    model_config = ConfigDict(frozen=True)

    edge_id: str
    relationship: str
    from_entity: str
    to_entity: str
    clause_id: str
    char_start: int | None
    char_end: int | None
    text: str
    token_count: int = Field(ge=0)


class EdgeExpansion(BaseModel):
    """All edge-grounded expansion for one seed chunk."""

    model_config = ConfigDict(frozen=True)

    seed_chunk_id: str
    document_id: str
    blocks: tuple[EdgeContextBlock, ...] = ()


def format_edge_block_text(edge: LegalEdgeRecord) -> str:
    """Deterministic server-authored text for one expansion block.

    The char interval is always present in the citation surface: an edge that
    predates span grounding cites ``chars unavailable`` rather than inventing
    offsets.
    """
    if edge.char_start is not None and edge.char_end is not None:
        span = f"chars {edge.char_start}-{edge.char_end}"
    else:
        span = "chars unavailable"
    return (
        f"[grounding edge {edge.edge_id}] {edge.from_entity} "
        f"{edge.relationship.upper()} {edge.to_entity} — clause "
        f"{edge.clause_id}, grounded at {span}, confidence {edge.confidence:.2f}"
    )


def expand_along_edges(
    *,
    seeds: Sequence[EdgeSeedChunk],
    edges: Sequence[LegalEdgeRecord],
    max_expansion_tokens: int,
    count_tokens: CountTokens,
    max_blocks_per_seed: int = _DEFAULT_MAX_BLOCKS_PER_SEED,
) -> list[EdgeExpansion]:
    """Expand each seed chunk along the typed edges that touch it.

    An edge matches a seed when the edge belongs to the seed's document and
    either the edge's clause identity is among the seed's clause ids or one of
    its entity endpoints occurs in the seed content (case-insensitive; short
    entity names never match by occurrence).  Blocks are emitted in edge span
    order until the per-seed token budget or block cap is reached; an edge is
    consumed by the first seed it expands, so the best-ranked chunk wins.
    """
    if not seeds or not edges or max_expansion_tokens <= 0 or max_blocks_per_seed <= 0:
        return []
    expansions: list[EdgeExpansion] = []
    used_edge_ids: set[str] = set()
    for seed in seeds:
        matching = sorted(
            (
                edge
                for edge in edges
                if edge.edge_id not in used_edge_ids and _edge_matches_seed(edge, seed)
            ),
            key=_edge_order_key,
        )
        blocks: list[EdgeContextBlock] = []
        budget = max_expansion_tokens
        for edge in matching:
            if len(blocks) >= max_blocks_per_seed:
                break
            text = format_edge_block_text(edge)
            token_count = count_tokens(text)
            # Skip, never truncate: a block that does not fit is dropped whole
            # so no expansion ever cites a partial span, and a later, smaller
            # block may still fit.
            if token_count > budget:
                continue
            blocks.append(
                EdgeContextBlock(
                    edge_id=edge.edge_id,
                    relationship=edge.relationship,
                    from_entity=edge.from_entity,
                    to_entity=edge.to_entity,
                    clause_id=edge.clause_id,
                    char_start=edge.char_start,
                    char_end=edge.char_end,
                    text=text,
                    token_count=token_count,
                )
            )
            used_edge_ids.add(edge.edge_id)
            budget -= token_count
        if blocks:
            expansions.append(
                EdgeExpansion(
                    seed_chunk_id=seed.chunk_id,
                    document_id=seed.document_id,
                    blocks=tuple(blocks),
                )
            )
    return expansions


def _edge_matches_seed(edge: LegalEdgeRecord, seed: EdgeSeedChunk) -> bool:
    if edge.document_id != seed.document_id:
        return False
    # Bundled documents: two contracts can share an entity name.  A known
    # contract/version mismatch means the edge belongs to a different contract
    # than the seed chunk — matching it would cite one contract's relationship
    # under another's identity.  Unknown (None) on either side stays permissive
    # for legacy rows.
    if (
        edge.contract_index is not None
        and seed.contract_index is not None
        and edge.contract_index != seed.contract_index
    ):
        return False
    if (
        edge.document_version is not None
        and seed.document_version is not None
        and edge.document_version != seed.document_version
    ):
        return False
    if edge.clause_id and edge.clause_id in seed.clause_ids:
        return True
    return _entity_occurs(edge.from_entity, seed.content) or _entity_occurs(
        edge.to_entity, seed.content
    )


def _entity_occurs(entity: str, content: str) -> bool:
    if len(entity) < _MIN_ENTITY_OCCURRENCE_CHARS:
        return False
    return entity.casefold() in content.casefold()


def _edge_order_key(edge: LegalEdgeRecord) -> tuple[int, int, str, str, str, str, str]:
    """Span-first deterministic order; ungrounded edges sort last by identity."""
    return (
        edge.char_start if edge.char_start is not None else -1,
        edge.char_end if edge.char_end is not None else -1,
        edge.clause_id,
        edge.from_entity,
        edge.relationship,
        edge.to_entity,
        edge.edge_id,
    )
