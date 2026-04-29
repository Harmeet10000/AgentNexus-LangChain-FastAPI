"""Document processing utilities."""

from app.shared.document_processing.chunker import (
    ChunkingConfig,
    DocumentChunk,
    create_chunker,
)
from app.shared.document_processing.docling_enhanced import (
    BatchDoclingProcessor,
    DoclingEnhancedConverter,
    DoclingEnhancementConfig,
    DoclingExtractionResult,
    ExtractedCodeBlock,
    ExtractedImage,
    ExtractedTable,
    create_converter,
)
from app.shared.document_processing.embedder import EmbeddingGenerator, create_embedder
from app.shared.document_processing.entity_extractor import (
    Entity,
    ExtractionResult,
    GraphitiExtractor,
    Relationship,
    SimpleEntityExtractor,
    create_extractor,
)
from app.shared.document_processing.ingest import DocumentIngestionPipeline
from app.shared.document_processing.models import (
    AgentContext,
    AgentDependencies,
    Chunk,
    Document,
    IngestionConfig,
    IngestionResult,
    Message,
    MessageRole,
    SearchRequest,
    SearchResponse,
    SearchType,
    Session,
    StreamDelta,
    ToolCall,
)

__all__ = [
    # Models
    "SearchRequest",
    "SearchResponse",
    "SearchType",
    "MessageRole",
    "Document",
    "Chunk",
    "Session",
    "Message",
    "AgentDependencies",
    "AgentContext",
    "IngestionConfig",
    "IngestionResult",
    "ToolCall",
    "StreamDelta",
    # Chunker
    "ChunkingConfig",
    "DocumentChunk",
    "create_chunker",
    # Embedder
    "EmbeddingGenerator",
    "create_embedder",
    # Ingest
    "DocumentIngestionPipeline",
    # Docling Enhanced
    "DoclingEnhancementConfig",
    "DoclingEnhancedConverter",
    "DoclingExtractionResult",
    "ExtractedTable",
    "ExtractedCodeBlock",
    "ExtractedImage",
    "BatchDoclingProcessor",
    "create_converter",
    # Entity Extractor
    "Entity",
    "Relationship",
    "ExtractionResult",
    "GraphitiExtractor",
    "SimpleEntityExtractor",
    "create_extractor",
]
