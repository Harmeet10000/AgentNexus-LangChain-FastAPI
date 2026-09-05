from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import asyncer
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from graphiti_core.nodes import EpisodeType
from langchain_core.exceptions import LangChainException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from returns.result import Failure
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.langchain_layer.embeddings import EmbeddingTaskType, embed_texts
from app.shared.langchain_layer.models import serialize_to_toon
from app.shared.langgraph_layer.kb_retry import (
    TRANSIENT_EXTERNAL_TYPES,
    TransientExternalError,
    describe_failure,
    retry_immediate,
)
from app.shared.rag.document_processing.docling_enhanced import table_markdown
from app.shared.rag.graphiti.schemas import (
    GRAPHITI_EDGE_TYPE_MAP,
    GRAPHITI_EDGE_TYPES,
    GRAPHITI_ENTITY_TYPES,
)
from app.shared.result import log_expected_failure
from app.utils import logger

from .canonicalize import canonicalize_entities
from .prompts import (
    _CLASSIFY_EXTRACT_SYSTEM_PROMPT,
    _CONTEXTUALIZE_CHUNK_SYSTEM_PROMPT,
    _EXTRACT_SCHEMA_SYSTEM_PROMPT,
    _SEGMENT_DOCUMENT_SYSTEM_PROMPT,
)
from .state import (
    ClauseSegment,
    ClauseSegmentationResult,
    ContextualizedChunk,
    ContractMetadata,
    EntityExtractionResult,
    # Runtime-load-bearing, not cosmetic (the A6 situation): `from __future__
    # import annotations` makes every annotation a string, so `TC001` correctly
    # asks for a type-checking block — but langgraph calls `get_type_hints()`
    # on the dispatch function when the graph is built, which evaluates that
    # string. Confined to a type-checking block, the name is absent at runtime
    # and `build_ingestion_graph` raises `NameError` before compiling anything.
    IngestionState,  # noqa: TC001
    ParsedDocument,
    StoredChunk,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from docling_core.types.doc.document import DoclingDocument
    from redis.asyncio import Redis
    from sqlalchemy.engine.result import Result
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.sql.elements import TextClause
    from ty_extensions import Unknown

    from .errors import IngestionGraphError
    from .state import StructuredRunnable

from .errors import IngestionGraphValidationError


def _state_failure(error: IngestionGraphError) -> dict[str, object]:
    """Construct a state dict from a failure error (node boundary)."""
    return {"failure": error, "ingestion_complete": False}


def _ingestion_failure(error: IngestionGraphError) -> Failure[IngestionGraphError]:
    return Failure(error)


def _validation_failure(message: str, *, doc_id: str = "") -> Failure[IngestionGraphError]:
    return _ingestion_failure(
        IngestionGraphValidationError(
            message=message,
            details={"doc_id": doc_id} if doc_id else None,
            source="ingestion_graph",
        )
    )


def make_parse_document_node() -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def parse_document_node(state: IngestionState) -> dict[str, object]:
        if not state.raw_bytes:
            result = _validation_failure(
                "Uploaded document is empty",
                doc_id=state.doc_id,
            )
            log_expected_failure(result.failure(), operation="parse_document")
            return _state_failure(result.failure())

        parsed: ParsedDocument = await retry_immediate(
            lambda: _parse_document_with_docling(state.raw_bytes, state.filename, state.source),
            label="docling_parse_document",
        )
        return {"parsed_document": parsed}

    return parse_document_node


def make_extract_schema_node(
    schema_llm: StructuredRunnable,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def extract_schema_node(state: IngestionState) -> dict[str, object]:
        parsed: ParsedDocument | None = state.parsed_document
        if parsed is None:
            result: Failure[IngestionGraphError] = _validation_failure(
                "Parsed document is required before schema extraction",
                doc_id=state.doc_id,
            )
            log_expected_failure(result.failure(), operation="extract_schema")
            return _state_failure(result.failure())

        payload = serialize_to_toon(
            {
                "document_type": state.document_type,
                "jurisdiction_hint": state.jurisdiction,
                "source": parsed.source,
                "markdown": parsed.markdown[:40_000],
            }
        )
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=_EXTRACT_SCHEMA_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
        metadata = await retry_immediate(
            lambda: schema_llm.ainvoke(cast("list[Any]", messages)),
            label="gemini_extract_schema",
        )
        metadata: ContractMetadata = ContractMetadata.model_validate(metadata)
        if metadata.jurisdiction is None:
            metadata: ContractMetadata = metadata.model_copy(
                update={"jurisdiction": state.jurisdiction}
            )
        return {"contract_metadata": metadata}

    return extract_schema_node


def make_segment_document_node(
    segmentation_llm: StructuredRunnable,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def segment_document_node(state: IngestionState) -> dict[str, object]:
        parsed: ParsedDocument | None = state.parsed_document
        metadata: ContractMetadata | None = state.contract_metadata
        if parsed is None or metadata is None:
            result: Failure[IngestionGraphError] = _validation_failure(
                "Parsed document and metadata are required before segmentation",
                doc_id=state.doc_id,
            )
            log_expected_failure(result.failure(), operation="segment_document")
            return _state_failure(result.failure())

        payload = serialize_to_toon(
            {
                "contract_metadata": metadata.model_dump(),
                "source": parsed.source,
                "markdown": parsed.markdown[:50_000],
            }
        )
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=_SEGMENT_DOCUMENT_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
        try:
            result = await retry_immediate(
                lambda: segmentation_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_segment_document",
            )
            segments: list[ClauseSegment] = ClauseSegmentationResult.model_validate(result).segments
        # Two types, because a model failure now reaches this branch by two distinct
        # routes and only one of them existed before. A deterministic framework failure
        # is excluded from the boundary's named-transient set, so it arrives here
        # unwrapped and unretried. A genuinely transient one — a parse refusal, a quota
        # refusal, a dropped connection — is retried and, if it outlives the budget,
        # arrives as the boundary's transient type with the original reachable through
        # its cause. Catching only the first was the defect: the boundary raised only the
        # second, so this branch could not fire in production and the pipeline propagated
        # exactly where it appeared to degrade. Chaining had been proposed as the remedy
        # and cannot be one — it populates the cause, it does not change the type raised,
        # so no amount of chaining makes this `except` match.
        except (LangChainException, TransientExternalError) as exc:
            exc.add_note(f"doc_id={state.doc_id}, operation=segmentation")
            logger.bind(doc_id=state.doc_id, error=describe_failure(exc)).warning(
                "structured_segmentation_failed_using_fallback"
            )
            segments: list[ClauseSegment] = _fallback_segments(parsed.markdown)

        return {"segments": _ensure_chunk_enrichment(segments)}

    return segment_document_node


def dispatch_contextualize_chunks(state: IngestionState) -> list[Send]:
    metadata: ContractMetadata = state.contract_metadata or ContractMetadata()
    parsed: ParsedDocument = state.parsed_document or ParsedDocument(
        markdown="", title="", source=state.source
    )
    return [
        Send(
            "contextualize_chunks",
            {
                # `Send` REPLACES the state for the fanned-out invocation — the node
                # receives this dict, not `IngestionState`. So anything the node needs
                # for a diagnostic has to be carried in here; it cannot be reached from
                # the graph state. `doc_id` was missing, which made a degraded
                # contextualization attributable to a clause but not to a document, and
                # `clause_id` alone does not disambiguate under concurrent ingestion.
                "doc_id": state.doc_id,
                "segment": segment.model_dump(),
                "contract_metadata": metadata.model_dump(),
                "source": parsed.source,
            },
        )
        for segment in state.segments
    ]


def make_contextualize_chunk_node(
    contextualize_llm: StructuredRunnable,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, object]]]:
    async def contextualize_chunk_node(state: dict[str, Any]) -> dict[str, object]:
        segment: ClauseSegment = ClauseSegment.model_validate(state["segment"])
        metadata: ContractMetadata = ContractMetadata.model_validate(state["contract_metadata"])
        # Bound before the `try`, and with `.get`, for the reason A5 is about: a
        # handler must not introduce a new raise site of its own. A missing key here
        # would turn a recoverable contextualization failure into a `KeyError` that
        # replaces the original diagnostic — the precise shape A5 set out to remove.
        doc_id: str = state.get("doc_id", "")
        preamble: str = _build_preamble(segment, metadata)
        payload = serialize_to_toon(
            {
                "required_preamble": preamble,
                "segment": segment.model_dump(),
                "contract_metadata": metadata.model_dump(),
            }
        )
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=_CONTEXTUALIZE_CHUNK_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
        try:
            result = await retry_immediate(
                lambda: contextualize_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_contextualize_chunk",
            )
            chunk: ContextualizedChunk = ContextualizedChunk.model_validate(result)
        # Both routes into this branch — see the segmentation node for why the pair is
        # required and why chaining alone was not a fix.
        except (LangChainException, TransientExternalError) as exc:
            exc.add_note(
                f"doc_id={doc_id}, clause_id={segment.clause_id}, "
                f"chunk_index={segment.chunk_index}, operation=contextualize"
            )
            logger.bind(
                doc_id=doc_id,
                clause_id=segment.clause_id,
                chunk_index=segment.chunk_index,
                error=describe_failure(exc),
            ).warning("contextualize_failed_using_deterministic_preamble")
            chunk: ContextualizedChunk = ContextualizedChunk(
                clause_id=segment.clause_id,
                chunk_index=segment.chunk_index,
                clause_type=segment.clause_type,
                preamble=preamble,
                text=segment.text,
                tokens=max(1, len(f"{preamble} {segment.text}".split())),
                page_no=segment.page_no,
                chunk_faqs=segment.chunk_faqs,
                chunk_keywords=segment.chunk_keywords,
            )
        return {"contextualized_chunks": [chunk]}

    return contextualize_chunk_node


def make_classify_extract_node(
    extraction_llm: StructuredRunnable,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def classify_extract_node(state: IngestionState) -> dict[str, object]:
        metadata: ContractMetadata | None = state.contract_metadata
        if metadata is None:
            result: Failure[IngestionGraphError] = _validation_failure(
                "Contract metadata is required before entity extraction",
                doc_id=state.doc_id,
            )
            log_expected_failure(result.failure(), operation="classify_extract")
            return _state_failure(result.failure())

        payload = serialize_to_toon(
            {
                "contract_metadata": metadata.model_dump(),
                "chunks": [chunk.model_dump() for chunk in state.contextualized_chunks],
            }
        )
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=_CLASSIFY_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=payload[:80_000]),
        ]
        try:
            result = await retry_immediate(
                lambda: extraction_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_entity_extraction",
            )
            extraction: EntityExtractionResult = EntityExtractionResult.model_validate(result)
        # Both routes into this branch — see the segmentation node for why the pair is
        # required and why chaining alone was not a fix.
        except (LangChainException, TransientExternalError) as exc:
            exc.add_note(f"doc_id={state.doc_id}, operation=entity_extraction")
            logger.bind(doc_id=state.doc_id, error=describe_failure(exc)).warning(
                "entity_extraction_failed_continuing_without_entities"
            )
            extraction = EntityExtractionResult()
        return {
            "extracted_entities": extraction.entities,
            "extracted_relationships": extraction.relationships,
        }

    return classify_extract_node


def make_embed_store_node(
    db_engine: AsyncEngine,
    redis: Redis | None = None,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def embed_store_node(state: IngestionState) -> dict[str, object]:
        parsed: ParsedDocument | None = state.parsed_document
        metadata: ContractMetadata | None = state.contract_metadata
        if parsed is None or metadata is None:
            result: Failure[IngestionGraphError] = _validation_failure(
                "Parsed document and metadata are required before storage",
                doc_id=state.doc_id,
            )
            log_expected_failure(result.failure(), operation="embed_store")
            return _state_failure(result.failure())

        # ADR-2 ordering: canonicalisation precedes every graph-bound write, and
        # a refusal stops the document before a single row or episode exists.
        # There is no raw-text fallback identity.
        _canonical, refused = canonicalize_entities(state.extracted_entities)
        if refused:
            result = _refused_entities_failure(state.doc_id, refused_names(refused))
            log_expected_failure(result.failure(), operation="embed_store")
            return _state_failure(result.failure())

        async with AsyncSession(db_engine) as session, session.begin():
            parent_doc_id = await retry_immediate(
                lambda: _upsert_parent_document(session, state, parsed, metadata),
                label="postgres_upsert_parent_document",
            )
            stored_entities: dict[str, str] = await _store_entities(session, state)
            stored_relationships: list[str] = await _store_relationships(
                session, state, stored_entities
            )
            stored_chunks: list[StoredChunk] = await _store_chunks(
                session=session,
                state=state,
                parsed=parsed,
                metadata=metadata,
                parent_doc_id=parent_doc_id,
                redis=redis,
            )
            await retry_immediate(
                lambda: _force_merge_bm25(session),
                label="postgres_bm25_force_merge",
            )

        return {
            "parent_doc_id": parent_doc_id,
            "stored_clause_ids": [chunk.chunk_id for chunk in stored_chunks],
            "stored_chunks": stored_chunks,
            "stored_entity_ids": list(stored_entities.values()),
            "stored_relationship_ids": stored_relationships,
        }

    return embed_store_node


def make_graphiti_upsert_node(
    graphiti: Any,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def graphiti_upsert_node(state: IngestionState) -> dict[str, object]:
        # A terminal failure upstream means no episode is written for this
        # document — the refusal scenario requires exactly this.
        if state.failure is not None:
            return {"graphiti_episode_ids": [], "ingestion_complete": False}
        if graphiti is None:
            return {"graphiti_episode_ids": [], "ingestion_complete": True}

        # ADR-2: every episode write keys on the canonical identity. The episode
        # carries the canonical entity ids as its entity references; a refusal
        # records a terminal failure and writes no episode at all.
        canonical, refused = canonicalize_entities(state.extracted_entities)
        if refused:
            result = _refused_entities_failure(state.doc_id, refused_names(refused))
            log_expected_failure(result.failure(), operation="graphiti_upsert")
            return _state_failure(result.failure())
        canonical_entity_ids = sorted(
            record.canonical_id for record in canonical.values()
        )

        episode_ids: list[str] = []
        for chunk in state.contextualized_chunks:
            postgres_chunk_id: str | None = _stored_chunk_id(state, chunk.clause_id)
            if postgres_chunk_id is None:
                continue
            body = (
                f"{chunk.preamble}\n\n{chunk.text}\n\n"
                f"REFERENCES_CLAUSE postgres_chunk_id={postgres_chunk_id}"
            )
            source_description = json.dumps(
                {
                    "doc_id": state.doc_id,
                    "clause_id": chunk.clause_id,
                    "postgres_chunk_id": postgres_chunk_id,
                    "clause_type": chunk.clause_type.value,
                    "edge_type": "REFERENCES_CLAUSE",
                    "entities": canonical_entity_ids,
                }
            )
            episode_id: str | None = await _graphiti_add_episode(
                graphiti=graphiti,
                name=f"clause:{state.doc_id}:{chunk.clause_id}",
                body=body,
                source_description=source_description,
                group_id=state.doc_id,
            )
            if episode_id:
                episode_ids.append(episode_id)

        for event_name, event_date in _contract_events(state.contract_metadata):
            episode_id: str | None = await _graphiti_add_episode(
                graphiti=graphiti,
                name=f"{event_name}:{state.doc_id}:{event_date}",
                body=f"{event_name} for {state.doc_id} occurs on {event_date}.",
                source_description=json.dumps(
                    {"doc_id": state.doc_id, "event_type": event_name, "event_date": event_date}
                ),
                group_id=state.doc_id,
            )
            if episode_id:
                episode_ids.append(episode_id)

        return {"graphiti_episode_ids": episode_ids, "ingestion_complete": True}

    return graphiti_upsert_node


async def _parse_document_with_docling(
    raw_bytes: bytes,
    filename: str,
    source: str,
) -> ParsedDocument:
    def _sync_parse() -> ParsedDocument:
        suffix = Path(filename or "upload.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            result: ConversionResult = DocumentConverter().convert(tmp.name)
            document: DoclingDocument = result.document
            markdown = document.export_to_markdown()
            # Was a comprehension over `table.to_markdown()` guarded by
            # `hasattr(table, "to_markdown")`. `TableItem` has no such method — it is
            # `export_to_markdown` — so the guard was false for every table and this list was
            # empty for every document this node has ever parsed. The guard is what hid it: it
            # made a wrong method name look like defensive handling of an optional one.
            tables: list[str] = table_markdown(document)
            elements: list[Unknown] = []
            for item, _level in document.iterate_items():
                to_dict = getattr(item, "to_dict", None)
                if callable(to_dict):
                    elements.append(to_dict())
            return ParsedDocument(
                markdown=markdown,
                title=_extract_title(markdown, filename),
                source=source or filename or tmp.name,
                page_count=len(getattr(document, "pages", []) or []),
                tables=tables,
                elements=elements[:500],
            )

    return await asyncer.asyncify(_sync_parse)()


def _fallback_segments(markdown: str) -> list[ClauseSegment]:
    blocks: list[str | Any] = [
        block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()
    ]
    if not blocks:
        blocks = [markdown.strip()] if markdown.strip() else []
    return [
        ClauseSegment(
            clause_id=f"clause-{index + 1}",
            text=block,
            chunk_index=index,
            chunk_faqs=[f"What does clause {index + 1} cover?"],
            chunk_keywords=_keywords(block),
        )
        for index, block in enumerate(blocks[:200])
    ]


def _ensure_chunk_enrichment(segments: list[ClauseSegment]) -> list[ClauseSegment]:
    enriched: list[ClauseSegment] = []
    for index, segment in enumerate(segments):
        enriched.append(
            segment.model_copy(
                update={
                    "chunk_index": index,
                    "page_no": segment.page_no or 0,
                    "chunk_faqs": segment.chunk_faqs or [f"What does {segment.clause_id} say?"],
                    "chunk_keywords": segment.chunk_keywords or _keywords(segment.text),
                }
            )
        )
    return enriched


def _build_preamble(segment: ClauseSegment, metadata: ContractMetadata) -> str:
    party_a = metadata.party_a or (metadata.parties[0] if metadata.parties else "unknown party")
    party_b = metadata.party_b or (
        metadata.parties[1] if len(metadata.parties) > 1 else "unknown party"
    )
    effective = metadata.effective_date or "unknown date"
    return (
        f"This is {segment.clause_type.value} from {metadata.contract_name} "
        f"between {party_a} and {party_b}, effective {effective}."
    )


async def _upsert_parent_document(
    session: AsyncSession,
    state: IngestionState,
    parsed: ParsedDocument,
    metadata: ContractMetadata,
) -> str:
    # D15 (`documents`/`chunks` is the sole retrieval schema): the pipeline writes
    # the `documents` row, never the superseded parent-document relation — that
    # relation does not exist and no migration will create it. Identity is the pair
    # (user_id, content_hash), so the conflict target names
    # `uq_documents_user_content_hash`; there is no `doc_id` column to conflict on.
    # The writer supplies `id` explicitly (no database default) and `object_uri`
    # (NOT NULL, the provenance link for re-parsing). The thread scope and the
    # document summary live in `metadata_` as scalars — `documents` carries no
    # full-text body column, so `markdown` is deliberately not persisted; the chunk
    # rows carry the only text this schema keeps. Status is set on insert only: a
    # re-ingest must not regress a terminal status back to a non-terminal one.
    content_hash: str = hashlib.sha256(parsed.markdown.encode("utf-8")).hexdigest()
    query: TextClause = text(
        """
        INSERT INTO documents
            (id, user_id, title, source_uri, object_uri, content_hash,
             document_kind, status, jurisdiction, contract_type, parties,
             metadata_, updated_at)
        VALUES
            (:id, :user_id, :title, :source_uri, :object_uri, :content_hash,
             :document_kind, :status, :jurisdiction, :contract_type,
             CAST(:parties AS JSONB), CAST(:metadata_ AS JSONB), NOW())
        ON CONFLICT ON CONSTRAINT uq_documents_user_content_hash
        DO UPDATE SET
            title = EXCLUDED.title,
            source_uri = EXCLUDED.source_uri,
            object_uri = EXCLUDED.object_uri,
            document_kind = EXCLUDED.document_kind,
            jurisdiction = EXCLUDED.jurisdiction,
            contract_type = EXCLUDED.contract_type,
            parties = EXCLUDED.parties,
            metadata_ = EXCLUDED.metadata_,
            updated_at = NOW()
        RETURNING id::text
        """
    )
    row = (
        await session.execute(
            query,
            {
                "id": str(uuid4()),
                "user_id": state.user_id,
                "title": parsed.title,
                "source_uri": state.source or None,
                "object_uri": _resolve_object_uri(state, parsed),
                "content_hash": content_hash,
                "document_kind": state.document_type or "generic",
                "status": "processing",
                "jurisdiction": metadata.jurisdiction or state.jurisdiction or None,
                "contract_type": metadata.contract_type or None,
                "parties": json.dumps(list(metadata.parties)),
                "metadata_": json.dumps(
                    _contract_metadata_json(
                        metadata, parsed.source, thread_id=state.thread_id or None
                    )
                ),
            },
        )
    ).fetchone()
    if row is None:
        msg = "document upsert did not return an id"
        raise ValueError(msg)
    return str(row[0])


def _resolve_object_uri(state: IngestionState, parsed: ParsedDocument) -> str:
    """Return the non-empty provenance link `documents.object_uri` requires.

    The parsed source is preferred; the dispatch-time source and filename follow.
    The final fallback names the ingestion identity rather than an empty string —
    an empty value would claim the text came from nowhere, which is exactly what
    the NOT NULL contract forbids.
    """
    return (
        parsed.source
        or state.source
        or state.filename
        or f"ingest://{state.doc_id or 'unknown'}"
    )


async def _store_entities(
    session: AsyncSession,
    state: IngestionState,
) -> dict[str, str]:
    # D15 / Decision 10: entities live in the knowledge-graph store, not in
    # relational tables — this stage issues no SQL at all. Identities are the
    # ADR-2 canonical identities; entities the canonicaliser refused were
    # already stopped upstream, and are skipped here as a double-guard so a
    # refusal can never fabricate a raw-text endpoint downstream.
    _ = session
    canonical, _refused = canonicalize_entities(state.extracted_entities)
    return {
        entity_ref: record.canonical_id for entity_ref, record in canonical.items()
    }


async def _store_relationships(
    session: AsyncSession,
    state: IngestionState,
    entity_id_map: dict[str, str],
) -> list[str]:
    # Same fate as `_store_entities`: `relationships` does not exist and the
    # graph-episode path (`_graphiti_add_episode`) is the sole writer. This stage
    # validates that both endpoints resolved and mints the deterministic edge key
    # the episode write carries — no SQL. Unresolvable endpoints are skipped, as
    # before, because a relationship naming an entity that was refused must not
    # fabricate an endpoint.
    _ = session
    stored: list[str] = []
    for relationship in state.extracted_relationships:
        from_id: str | None = entity_id_map.get(relationship.from_entity)
        to_id: str | None = entity_id_map.get(relationship.to_entity)
        if from_id is None or to_id is None:
            continue
        stored.append(
            _graph_relationship_identity(
                relationship.type.value, from_id, to_id, relationship.clause_id
            )
        )
    return stored


def _refused_entities_failure(
    doc_id: str, names: list[str]
) -> Failure[IngestionGraphError]:
    return _validation_failure(
        f"Entity canonicalisation refused {len(names)} extracted "
        f"entit{'y' if len(names) == 1 else 'ies'} with no usable identity "
        f"({', '.join(names)}); no graph write was issued",
        doc_id=doc_id,
    )


def refused_names(refused: list[Any]) -> list[str]:
    """Surface forms behind a refusal, truncated for the failure record."""
    return [entity.name[:80] for entity in refused]


def _graph_relationship_identity(
    relation_type: str, from_id: str, to_id: str, clause_id: str | None
) -> str:
    """Deterministic edge key carried into the graph-episode path (no SQL)."""
    return f"{relation_type}:{from_id}->{to_id}:{clause_id or ''}"


async def _store_chunks(
    *,
    session: AsyncSession,
    state: IngestionState,
    parsed: ParsedDocument,
    metadata: ContractMetadata,
    parent_doc_id: str,
    redis: Redis | None,
) -> list[StoredChunk]:
    stored: list[StoredChunk] = []
    ordered = sorted(state.contextualized_chunks, key=lambda item: item.chunk_index)
    if not ordered:
        return stored

    # Bound once and reused for both the provider call and the `text` column, because those two
    # must be the same string. The column is not a copy of the chunk body — `chunk_text` is that
    # — it is a record of *what was embedded*, and a retrieval path that re-embeds it to compare
    # against the stored vector depends on that identity. Two separate expressions could drift.
    embedded_texts: list[str] = [f"{chunk.preamble}\n\n{chunk.text}" for chunk in ordered]

    # Embedded as one batch before the loop, not once per chunk inside it. The prior code made
    # one provider round-trip per chunk, so a 400-clause contract cost 400 requests where it
    # now costs four. The task type is `DOCUMENT` because these vectors are what a query is
    # later compared *against* — passing none, as the prior code did, stored query-projection
    # vectors on the document side and degraded every subsequent search without ever erroring.
    embeddings: list[list[float]] = await retry_immediate(
        lambda: embed_texts(
            embedded_texts,
            task_type=EmbeddingTaskType.DOCUMENT,
            redis=redis,
        ),
        label="gemini_embedding",
    )

    for chunk, _text_to_embed, embedding in zip(
        ordered, embedded_texts, embeddings, strict=True
    ):
        row_id = str(uuid4())
        chunk_id = row_id
        metadata_json = _chunk_metadata_json(
            metadata=metadata,
            source=parsed.source,
            page_no=chunk.page_no,
        )
        custom_metadata = {
            "source": parsed.source,
            "page_no": chunk.page_no,
            "document_summary": metadata.document_summary,
            "chunk_id": chunk_id,
            "chunk_faqs": chunk.chunk_faqs,
            "chunk_keywords": chunk.chunk_keywords,
        }
        # D15: `chunks` is the sole retrieval truth — `clauses` does not exist.
        # `clause` survives only as a `chunk_kind` value with the clause's own
        # label in `clause_type`. The upsert key is (document_id, chunk_index),
        # named via `uq_chunks_document_chunk_index`; `search_text` is generated
        # by the database and is never supplied. `updated_at` appears in both
        # the row payload and the conflict set (the D15 trap: the ORM hook does
        # not fire for a conflict-resolved upsert, so omitting either leaves a
        # column that looks maintained and is not).
        query: TextClause = text(
            """
                    INSERT INTO chunks
                        (id, document_id, user_id, chunk_index, chunk_kind,
                         content, preamble, clause_type, page_no, embedding,
                         metadata_, custom_metadata, quality_warnings, updated_at)
                    VALUES
                        (:id, :document_id, :user_id, :chunk_index, :chunk_kind,
                         :content, :preamble, :clause_type, :page_no,
                         CAST(:embedding AS vector), CAST(:metadata AS JSONB),
                         CAST(:custom_metadata AS JSONB),
                         CAST(:quality_warnings AS JSONB), NOW())
                    ON CONFLICT ON CONSTRAINT uq_chunks_document_chunk_index
                    DO UPDATE SET
                        chunk_kind = EXCLUDED.chunk_kind,
                        content = EXCLUDED.content,
                        preamble = EXCLUDED.preamble,
                        clause_type = EXCLUDED.clause_type,
                        page_no = EXCLUDED.page_no,
                        embedding = EXCLUDED.embedding,
                        metadata_ = EXCLUDED.metadata_,
                        custom_metadata = EXCLUDED.custom_metadata,
                        quality_warnings = EXCLUDED.quality_warnings,
                        updated_at = NOW()
                    RETURNING id::text
                    """
        )
        params = {
            "id": row_id,
            "document_id": parent_doc_id,
            "user_id": state.user_id,
            "chunk_index": chunk.chunk_index,
            "chunk_kind": "clause",
            "content": _naturalize_tables(chunk.text),
            "preamble": chunk.preamble,
            "clause_type": chunk.clause_type.value,
            "page_no": chunk.page_no,
            "embedding": _vector_literal(embedding),
            "metadata": json.dumps(metadata_json),
            "custom_metadata": json.dumps(custom_metadata),
            "quality_warnings": json.dumps([]),
        }
        result: Result[Any] = await retry_immediate(
            lambda query=query, params=params: session.execute(query, params),
            label="postgres_store_chunk",
        )
        row = result.fetchone()
        stored_chunk_id = str(row[0]) if row else chunk_id
        stored.append(
            StoredChunk(
                chunk_id=stored_chunk_id,
                clause_id=chunk.clause_id,
                chunk_index=chunk.chunk_index,
                clause_type=chunk.clause_type.value,
            )
        )
    return stored


async def _force_merge_bm25(session: AsyncSession) -> None:
    try:
        await session.execute(text("SELECT bm25_force_merge('chunks_bm25_idx')"))
    except Exception as exc:  # noqa: BLE001 — extension/index may be absent in local/dev DBs
        exc.add_note("operation=bm25_force_merge")
        logger.bind(error=str(exc)).warning("bm25_force_merge_skipped")


async def _graphiti_add_episode(
    *,
    graphiti: Any,
    name: str,
    body: str,
    source_description: str,
    group_id: str,
) -> str | None:
    try:
        result = await retry_immediate(
            lambda: graphiti.add_episode(
                name=name,
                episode_body=body,
                source=EpisodeType.text,
                source_description=source_description,
                reference_time=datetime.now(tz=UTC),
                group_id=group_id,
                entity_types=GRAPHITI_ENTITY_TYPES,
                edge_types=GRAPHITI_EDGE_TYPES,
                edge_type_map=GRAPHITI_EDGE_TYPE_MAP,
            ),
            label="graphiti_add_episode",
        )
        return str(getattr(result, "uuid", name))
    except (TransientExternalError, *TRANSIENT_EXTERNAL_TYPES) as exc:
        # Only the shapes retry_immediate itself calls transient may degrade the
        # graph write to None; anything else (a bug, a LangGraph pause) must
        # propagate rather than be swallowed as "graph unavailable". The write
        # still never rolls back Postgres ingestion — that property is kept.
        exc.add_note(f"name={name}, operation=graphiti_add_episode")
        logger.bind(name=name, error=str(exc)).warning("graphiti_episode_upsert_failed")
        return None


def _contract_events(metadata: ContractMetadata | None) -> list[tuple[str, str]]:
    if metadata is None:
        return []
    events = [
        ("contract_signed", metadata.contract_signed),
        ("amendment_effective", metadata.amendment_effective),
        ("expiry_date", metadata.expiry_date),
    ]
    return [(name, date) for name, date in events if date]


def _stored_chunk_id(state: IngestionState, clause_id: str) -> str | None:
    for chunk in state.stored_chunks:
        if chunk.clause_id == clause_id:
            return chunk.chunk_id
    return None


def _contract_metadata_json(
    metadata: ContractMetadata, source: str, *, thread_id: str | None = None
) -> dict[str, object]:
    # D15: `documents` has no `summary` or `thread_id` column — both live in
    # `metadata_` as scalars (`document_summary` arrives via `model_dump`).
    payload = metadata.model_dump()
    payload["source"] = source
    payload["page_no"] = 0
    if thread_id is not None:
        payload["thread_id"] = thread_id
    return payload


def _chunk_metadata_json(
    *,
    metadata: ContractMetadata,
    source: str,
    page_no: int,
) -> dict[str, object]:
    payload = _contract_metadata_json(metadata, source)
    payload["page_no"] = page_no
    payload["jurisdiction"] = metadata.jurisdiction
    payload["contract_type"] = metadata.contract_type
    payload["party_names"] = metadata.parties
    return payload


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def _extract_title(markdown: str, filename: str) -> str:
    for line in markdown.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:500]
    return Path(filename or "uploaded-document").stem[:500]


def _keywords(text_value: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text_value.lower())
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        result.append(word)
        if len(result) >= 12:
            break
    return result


def _naturalize_tables(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_table_line(line):
            table: list[str] = []
            while index < len(lines) and _is_table_line(lines[index]):
                table.append(lines[index])
                index += 1
            output.extend(_table_to_sentences(table))
            continue
        output.append(line)
        index += 1
    return "\n".join(output)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _table_to_sentences(lines: list[str]) -> list[str]:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
        if "---" not in line
    ]
    if not rows:
        return []
    headers = rows[0]
    sentences: list[str] = []
    for row_index, row in enumerate(rows[1:], start=1):
        pairs = [
            f"{headers[index]}: {value}"
            for index, value in enumerate(row)
            if index < len(headers) and value
        ]
        if pairs:
            sentences.append(f"Table row {row_index}: {'; '.join(pairs)}.")
    return sentences
