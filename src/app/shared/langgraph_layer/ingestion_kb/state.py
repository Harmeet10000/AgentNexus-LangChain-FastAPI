from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

EmbeddingFunction = Callable[[str], Awaitable[list[float]]]
ExtractionRunnable = Runnable[list[BaseMessage], Any]


class ExtractedEntityType(StrEnum):
    PERSON = "PERSON"
    ORG = "ORG"
    CLAUSE = "CLAUSE"
    CONTRACT = "CONTRACT"
    OBLIGATION = "OBLIGATION"


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: ExtractedEntityType
    name: str
    normalized_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedRelationship(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    from_entity: str = Field(serialization_alias="from", validation_alias="from")
    to_entity: str = Field(serialization_alias="to", validation_alias="to")
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: str | None = None
    valid_to: str | None = None


class ExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class IngestionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = ""
    user_id: str = ""
    thread_id: str = ""
    raw_text: str = ""
    document_type: str = "unknown"
    jurisdiction: str = "India"

    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)
    extracted_relationships: list[ExtractedRelationship] = Field(default_factory=list)
    extraction_error: str | None = None

    validated_entities: list[ExtractedEntity] = Field(default_factory=list)
    validated_relationships: list[ExtractedRelationship] = Field(default_factory=list)
    dropped_entity_count: int = 0
    dropped_relationship_count: int = 0

    stored_entity_ids: list[str] = Field(default_factory=list)
    stored_clause_ids: list[str] = Field(default_factory=list)
    stored_relationship_ids: list[str] = Field(default_factory=list)
    ingestion_complete: bool = False
    error: str | None = None
