"""
Docling HybridChunker implementation for intelligent document splitting.

Features:
- Token-aware chunking (uses actual tokenizer)
- Document structure preservation (headings, sections, tables)
- Semantic boundary respect (paragraphs, code blocks)
- Contextualized output (chunks include heading hierarchy)
"""

from typing import Any

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

from app.utils.logger import logger as loguru_logger

from .models import Chunk, IngestionConfig


def get_tokenizer(model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> AutoTokenizer:
    """Initialize tokenizer for token-aware chunking."""
    loguru_logger.info(f"Initializing tokenizer: {model_id}")
    return AutoTokenizer.from_pretrained(model_id)


def create_hybrid_chunker(
    tokenizer: AutoTokenizer, config: IngestionConfig
) -> HybridChunker:
    """Create HybridChunker instance."""
    loguru_logger.info(f"HybridChunker initialized (max_tokens={config.max_tokens})")
    return HybridChunker(
        tokenizer=tokenizer,
        max_tokens=config.max_tokens,
        merge_peers=True,
    )


async def chunk_document(
    content: str,
    title: str,
    source: str,
    config: IngestionConfig,
    tokenizer: AutoTokenizer,
    hybrid_chunker: HybridChunker | None = None,
    metadata: dict[str, Any] | None = None,
    docling_doc: DoclingDocument | None = None,
) -> list[Chunk]:
    """
    Chunk a document using Docling's HybridChunker or fallback.

    Args:
        content: Document content (markdown format)
        title: Document title
        source: Document source
        config: Chunking configuration
        tokenizer: Initialized tokenizer
        hybrid_chunker: Optional pre-created HybridChunker instance
        metadata: Additional metadata
        docling_doc: Optional pre-converted DoclingDocument (for efficiency)

    Returns:
        List of document chunks with contextualized content
    """
    if not content.strip():
        return []

    base_metadata = {
        "title": title,
        "source": source,
        "chunk_method": "hybrid" if docling_doc and hybrid_chunker else "simple_fallback",
        **(metadata or {}),
    }

    if docling_doc is None or hybrid_chunker is None:
        loguru_logger.warning(
            "No DoclingDocument or HybridChunker provided, using simple chunking fallback"
        )
        return _simple_fallback_chunk(content, base_metadata, config, tokenizer)

    try:
        chunk_iter = hybrid_chunker.chunk(dl_doc=docling_doc)
        chunks = list(chunk_iter)

        document_chunks = []
        for i, chunk in enumerate(chunks):
            contextualized_text = hybrid_chunker.contextualize(chunk=chunk)
            token_count = len(tokenizer.encode(contextualized_text))

            document_chunks.append(
                Chunk(
                    content=contextualized_text.strip(),
                    chunk_index=i,
                    document_id="",
                    metadata={
                        **base_metadata,
                        "total_chunks": len(chunks),
                        "token_count": token_count,
                        "has_context": True,
                    },
                    token_count=token_count,
                )
            )

        loguru_logger.info(f"Created {len(document_chunks)} chunks using HybridChunker")
        return document_chunks

    except Exception as e:
        loguru_logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
        return _simple_fallback_chunk(content, base_metadata, config, tokenizer)


def _simple_fallback_chunk(
    content: str,
    base_metadata: dict[str, Any],
    config: IngestionConfig,
    tokenizer: AutoTokenizer,
) -> list[Chunk]:
    """
    Simple fallback chunking when HybridChunker can't be used.

    Args:
        content: Content to chunk
        base_metadata: Base metadata for chunks
        config: Chunking configuration
        tokenizer: Initialized tokenizer

    Returns:
        List of document chunks
    """
    chunks = []
    chunk_size = config.chunk_size
    overlap = config.chunk_overlap

    start = 0
    chunk_index = 0

    while start < len(content):
        end = start + chunk_size

        if end >= len(content):
            chunk_text = content[start:]
        else:
            chunk_end = end
            for i in range(end, max(start + config.min_chunk_size, end - 200), -1):
                if i < len(content) and content[i] in ".!?\n":
                    chunk_end = i + 1
                    break
            chunk_text = content[start:chunk_end]
            end = chunk_end

        if chunk_text.strip():
            token_count = len(tokenizer.encode(chunk_text))
            chunks.append(
                Chunk(
                    content=chunk_text.strip(),
                    chunk_index=chunk_index,
                    document_id="",
                    metadata={
                        **base_metadata,
                        "chunk_method": "simple_fallback",
                        "total_chunks": -1,
                    },
                    token_count=token_count,
                )
            )
            chunk_index += 1

        start = end - overlap

    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)

    loguru_logger.info(f"Created {len(chunks)} chunks using simple fallback")
    return chunks


async def chunk_document_simple(
    content: str,
    title: str,
    source: str,
    config: IngestionConfig,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Simple paragraph-based chunking without semantic splitting.

    Args:
        content: Document content
        title: Document title
        source: Document source
        config: Chunking configuration
        metadata: Additional metadata

    Returns:
        List of document chunks
    """
    if not content.strip():
        return []

    import re

    base_metadata = {
        "title": title,
        "source": source,
        "chunk_method": "simple",
        **(metadata or {}),
    }

    paragraphs = re.split(r"\n\s*\n", content)
    chunks = []
    chunk_index = 0

    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

        if len(potential_chunk) <= config.chunk_size:
            current_chunk = potential_chunk
        else:
            if current_chunk:
                chunks.append(
                    Chunk(
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        document_id="",
                        metadata=base_metadata.copy(),
                    )
                )
                chunk_index += 1
            current_chunk = paragraph

    if current_chunk:
        chunks.append(
            Chunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                document_id="",
                metadata=base_metadata.copy(),
            )
        )

    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


async def initialize_chunking(
    config: IngestionConfig,
) -> tuple[AutoTokenizer, HybridChunker]:
    """
    Initialize chunking dependencies.

    Args:
        config: Chunking configuration

    Returns:
        Tuple of (tokenizer, hybrid_chunker)
    """
    if not config.use_semantic_chunking:
        raise ValueError("This function requires semantic chunking enabled")

    tokenizer = get_tokenizer()
    hybrid_chunker = create_hybrid_chunker(tokenizer, config)
    return tokenizer, hybrid_chunker
