"""State and schemas for contract KB ingestion."""

from __future__ import annotations

import operator
from enum import StrEnum

# `Annotated` is imported at runtime, not under `TYPE_CHECKING`, and the suppression is
# load-bearing rather than cosmetic. `from __future__ import annotations` makes every
# annotation below a string, so `TC003` is right by the language's rules — but LangGraph
# *evaluates* those strings when it builds the graph, and
# `IngestionState.contextualized_chunks`
# is annotated `Annotated[list[ContextualizedChunk], operator.add]`. With the import confined
# to a type-checking block the name is absent at runtime and the reducer is lost.
# The state failure import below is runtime-load-bearing for the same reason:
# tools reading `IngestionState.__annotations__` resolve every name in it.
from typing import Annotated, Any, TypedDict  # noqa: TC003 - hints evaluated at graph build.

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from .errors import IngestionGraphError  # noqa: TC001 - hints evaluated at graph build.


class ClauseType(StrEnum):
    INDEMNITY = "indemnity"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    ARBITRATION = "arbitration"
    TERMINATION = "termination"
    GOVERNING_LAW = "governing_law"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT = "payment"
    IP_OWNERSHIP = "ip_ownership"
    PENALTY = "penalty"
    OBLIGATION = "obligation"
    OTHER = "other"


class EntityType(StrEnum):
    PARTY = "PARTY"
    PERSON = "PERSON"
    ORG = "ORG"
    CONTRACT = "CONTRACT"
    CLAUSE = "CLAUSE"
    OBLIGATION = "OBLIGATION"
    RIGHT_OR_PERMISSION = "RIGHT_OR_PERMISSION"
    PENALTY_CLAUSE = "PENALTY_CLAUSE"
    DATE = "DATE"
    JURISDICTION = "JURISDICTION"


class RelationType(StrEnum):
    SIGNED_BY = "SIGNED_BY"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    OBLIGATED_TO = "OBLIGATED_TO"
    GOVERNED_BY = "GOVERNED_BY"
    SUPERSEDES = "SUPERSEDES"
    REFERENCES_CLAUSE = "REFERENCES_CLAUSE"


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown: str
    title: str
    source: str
    page_count: int = 0
    tables: list[str] = Field(default_factory=list)
    elements: list[dict[str, Any]] = Field(default_factory=list)


class ContractMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_name: str = "Unknown Contract"
    contract_type: str = "unknown"
    effective_date: str | None = None
    parties: list[str] = Field(default_factory=list)
    party_a: str | None = None
    party_b: str | None = None
    contract_value: str | None = None
    jurisdiction: str | None = None
    governing_law: str | None = None
    termination_notice_days: int | None = None
    liability_cap: str | None = None
    year: int | None = None
    document_summary: str = ""
    contract_signed: str | None = None
    amendment_effective: str | None = None
    expiry_date: str | None = None


class ClauseSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause_id: str
    clause_type: ClauseType = ClauseType.OTHER
    text: str
    page_no: int = 0
    chunk_index: int
    chunk_faqs: list[str] = Field(default_factory=list)
    chunk_keywords: list[str] = Field(default_factory=list)


class ClauseSegmentationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[ClauseSegment] = Field(default_factory=list)


class ContextualizedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause_id: str
    chunk_index: int
    clause_type: ClauseType = ClauseType.OTHER
    preamble: str
    text: str
    tokens: int
    page_no: int = 0
    chunk_faqs: list[str] = Field(default_factory=list)
    chunk_keywords: list[str] = Field(default_factory=list)


class ClauseClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause_id: str
    clause_type: ClauseType
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: EntityType
    name: str
    normalized_name: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class ExtractedRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_entity: str
    to_entity: str
    type: RelationType
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    clause_id: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None


class EntityExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class StoredChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    clause_id: str
    chunk_index: int
    clause_type: str


class IngestionState(TypedDict, total=False):
    # Decision 2: no arbitrary-types permission — every channel is a plain
    # scalar, a validated model, or a container of those. Adding a channel
    # holding an arbitrary object must come with the permission *and* the
    # justification, not just the permission.
    # Channels have no defaults: readers use .get() with the documented
    # fallback, so resumed plain dicts behave exactly like fresh ones.
    doc_id: str
    user_id: str
    thread_id: str
    source: str
    filename: str
    raw_bytes: bytes
    document_type: str
    jurisdiction: str

    parsed_document: ParsedDocument | None
    contract_metadata: ContractMetadata | None
    segments: list[ClauseSegment]
    contextualized_chunks: Annotated[list[ContextualizedChunk], operator.add]
    extracted_entities: list[ExtractedEntity]
    extracted_relationships: list[ExtractedRelationship]

    parent_doc_id: str | None
    stored_clause_ids: list[str]
    stored_chunks: list[StoredChunk]
    stored_entity_ids: list[str]
    stored_relationship_ids: list[str]
    graphiti_episode_ids: list[str]
    ingestion_complete: bool
    failure: IngestionGraphError | None


StructuredRunnable = Runnable[list[Any], Any]
