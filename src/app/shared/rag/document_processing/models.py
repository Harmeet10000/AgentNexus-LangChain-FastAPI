"""
Pydantic models for data validation and serialization.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Enums
class SearchType(StrEnum):
    """Search type enum."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class MessageRole(StrEnum):
    """Message role enum."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# Request Models
class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default=SearchType.SEMANTIC, description="Type of search")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    filters: dict[str, Any] = Field(default_factory=dict, description="Search filters")

    model_config = ConfigDict(use_enum_values=True)


# Response Models
class DocumentMetadata(BaseModel):
    """Document metadata model."""

    id: str
    title: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunk_count: int | None = None


class ChunkResult(BaseModel):
    """Chunk search result model."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_source: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is between 0 and 1."""
        return max(0.0, min(1.0, v))


class SearchResponse(BaseModel):
    """Search response model."""

    results: list[ChunkResult] = Field(default_factory=list)
    total_results: int = 0
    search_type: SearchType
    query_time_ms: float


class ToolCall(BaseModel):
    """Tool call information model."""

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    message: str
    session_id: str
    sources: list[DocumentMetadata] = Field(default_factory=list)
    tools_used: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamDelta(BaseModel):
    """Streaming response delta."""

    content: str
    delta_type: Literal["text", "tool_call", "end"] = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


# Database Models
class Document(BaseModel):
    """Document model."""

    id: str | None = None
    title: str
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Chunk(BaseModel):
    """Document chunk model."""

    id: str | None = None
    document_id: str
    content: str
    embedding: list[float] | None = None
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int | None = None
    created_at: datetime | None = None

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: list[float] | None) -> list[float] | None:
        """Validate embedding is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("Embedding cannot be empty")
        return v


class Session(BaseModel):
    """Session model."""

    id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None


class Message(BaseModel):
    """Message model."""

    id: str | None = None
    session_id: str
    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None

    model_config = ConfigDict(use_enum_values=True)


# Agent Models
class AgentDependencies(BaseModel):
    """Dependencies for the agent."""

    session_id: str
    database_url: str | None = None
    openai_api_key: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentContext(BaseModel):
    """Agent execution context."""

    session_id: str
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    search_results: list[ChunkResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Ingestion Models
class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""

    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    min_chunk_size: int = Field(default=100, ge=1)
    use_semantic_chunking: bool = True
    preserve_structure: bool = True
    max_tokens: int = 512

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class IngestionResult(BaseModel):
    """Result of document ingestion."""

    document_id: str
    title: str
    chunks_created: int
    entities_extracted: int = 0
    relationships_created: int = 0
    processing_time_ms: float
    errors: list[str] = Field(default_factory=list)


# Extraction Models
class DoclingEnhancementConfig(BaseModel):
    """Configuration for Docling enhanced features."""

    extract_tables: bool = True
    extract_code: bool = True
    extract_images: bool = True
    use_vlm_captioning: bool = True
    table_format: str = "markdown"
    generate_doctags: bool = True


class ExtractedTable(BaseModel):
    """Extracted table data."""

    table_index: int
    markdown: str
    csv: str | None = None
    html: str | None = None
    row_count: int = 0
    col_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedCodeBlock(BaseModel):
    """Extracted code block."""

    block_index: int
    code: str
    language: str
    start_line: int
    end_line: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedImage(BaseModel):
    """Extracted image with optional caption."""

    image_index: int
    image_path: str | None = None
    base64_data: str | None = None
    caption: str | None = None
    page_number: int = 1
    bounding_box: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DoclingExtractionResult(BaseModel):
    """Complete extraction result from enhanced Docling processing."""

    document_id: str
    markdown_content: str
    doctags_content: str | None = None
    tables: list[ExtractedTable] = Field(default_factory=list)
    code_blocks: list[ExtractedCodeBlock] = Field(default_factory=list)
    images: list[ExtractedImage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Knowledge Graph Models
class Entity(BaseModel):
    """Represents an extracted entity."""

    id: str | None = None
    name: str
    entity_type: str
    description: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    source_document_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Relationship(BaseModel):
    """Represents a relationship between entities."""

    id: str | None = None
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_document_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractionResult(BaseModel):
    """Result of entity extraction."""

    document_id: str
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# Tool Models
class ToolResult(BaseModel):
    """Result of tool execution."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
