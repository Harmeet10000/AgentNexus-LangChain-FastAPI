"""
Docling HybridChunker implementation for intelligent document splitting.

Features:
- Token-aware chunking (uses actual tokenizer)
- Document structure preservation (headings, sections, tables)
- Semantic boundary respect (paragraphs, code blocks)
- Contextualized output (chunks include heading hierarchy)

The token counter is loaded once per process, and it is not the embedding model's
--------------------------------------------------------------------------------
Two facts about the counter, both deliberate, recorded here because neither is
visible from a call site.

**It is loaded once per process.** ``get_tokenizer`` used to run the transformers
auto-class loader on every call, so every entry point that did not hoist the
result by hand paid a fresh load: a disk read once the local model cache is warm,
a network download before that. The load now happens inside ``_load_tokenizer``,
memoised for the life of the process; ``get_tokenizer`` is a normalising wrapper
over it and keeps the old signature, so no call site changes.

**It is not the counter the embedding provider uses.** Chunks are budgeted at
``IngestionConfig.max_tokens`` (512) counted by this WordPiece counter, then
embedded by the Gemini provider in ``embedder.py``. The divergence is *stated*
here rather than closed — the second of the two options Decision 3 of the
``ingestion-pipeline-unification`` design leaves open — because the installed
provider SDK exposes token counting only as a remote call, so matching the two
would put a network round trip inside every chunk-boundary decision. The margin,
computed from this repository's own two constants:

- the bound that actually applies downstream is ``embedder.py``'s, and it is a
  **character** bound, not a token bound: ``_MAX_INPUT_TOKENS * 4`` = 8192
  characters (``embedder.py:146,210``), applied by silent truncation;
- at the ~4 characters-per-token density that same guard assumes, a 512-token
  chunk is ~2048 characters, so the headroom is ~**4x**. The character-bounded
  paths are wider still: ``_simple_fallback_chunk`` cuts at ``config.chunk_size``
  (1000), ~8x;
- the margin degrades in exactly one direction. WordPiece maps an unsegmentable
  run of up to 100 characters onto a *single* unknown token, so this counter
  undercounts without bound on base64 blobs, hex digests and long URL path
  segments. A chunk of 512 such tokens can exceed 8192 characters and lose its
  tail to that truncation with no diagnostic.

Enforcing the bound belongs on the embedding side, where the constant already
lives, and to task B1, which collapses the four embedding paths into one — a
chunker cannot see which provider will embed what it emits.
"""

from functools import lru_cache
from typing import Any

from docling.chunking import HybridChunker
from docling.exceptions import BaseError as DoclingError
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.utils.logger import logger as loguru_logger

from .models import Chunk, IngestionConfig

DEFAULT_TOKENIZER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Bounded, not unbounded, because the cache key is caller-supplied: an unbounded
# cache lets a caller sweeping model ids pin an unbounded number of
# multi-megabyte counters for the life of the process. Exactly one id is passed
# anywhere in ``src/`` today, so any bound above one is spare capacity for the
# real workload; four leaves room for a comparison run, and an eviction costs a
# reload that hits the local model cache rather than the network.
_TOKENIZER_CACHE_SIZE = 4


@lru_cache(maxsize=_TOKENIZER_CACHE_SIZE)
def _load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Load the counter for ``model_id``, once per process.

    Kept separate from ``get_tokenizer`` so the memoised key is always a
    *resolved* model id. A default argument is not part of an ``lru_cache`` key:
    a call that omits the argument and a call that passes the identical default
    value hash to different entries and would each load their own copy.
    Normalising in the wrapper is what makes those two calls indistinguishable.

    The log line lives here rather than in the wrapper so that it reports a real
    load. Emitted per call it would be false for every call after the first,
    which is what it was before this became a cache.
    """
    loguru_logger.info("Loading tokenizer (first use in this process): {}", model_id)
    return AutoTokenizer.from_pretrained(model_id)


def get_tokenizer(model_id: str = DEFAULT_TOKENIZER_MODEL_ID) -> PreTrainedTokenizerBase:
    """Return the process-wide token counter for ``model_id``.

    Signature-compatible with the uncached version it replaces, so callers that
    acquire a counter per document — ``ingest_v2.py:184`` — and callers that
    hoist one by hand — ``ingest_v2.py:334`` — both get the cache without
    changing, and the hand-rolled hoist becomes redundant rather than wrong.
    """
    return _load_tokenizer(model_id)


def create_hybrid_chunker(
    tokenizer: PreTrainedTokenizerBase, config: IngestionConfig
) -> HybridChunker:
    """Create HybridChunker instance."""
    loguru_logger.info("HybridChunker initialized (max_tokens={})", config.max_tokens)
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
    tokenizer: PreTrainedTokenizerBase,
    *,
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
        document_chunks = _hybrid_chunk_documents(
            hybrid_chunker, docling_doc, tokenizer, base_metadata
        )
    except DoclingError as e:
        e.add_note("operation=hybrid_chunk")
        loguru_logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
        return _simple_fallback_chunk(content, base_metadata, config, tokenizer)

    loguru_logger.info("Created {} chunks using HybridChunker", len(document_chunks))
    return document_chunks


def _hybrid_chunk_documents(
    hybrid_chunker: HybridChunker,
    docling_doc: DoclingDocument,
    tokenizer: PreTrainedTokenizerBase,
    base_metadata: dict[str, Any],
) -> list[Chunk]:
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
    return document_chunks


def _simple_fallback_chunk(
    content: str,
    base_metadata: dict[str, Any],
    config: IngestionConfig,
    tokenizer: PreTrainedTokenizerBase,
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

    loguru_logger.info("Created {} chunks using simple fallback", len(chunks))
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
        para = paragraph.strip()
        if not para:
            continue

        potential_chunk = current_chunk + "\n\n" + para if current_chunk else para

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
) -> tuple[PreTrainedTokenizerBase, HybridChunker]:
    """
    Initialize chunking dependencies.

    Args:
        config: Chunking configuration

    Returns:
        Tuple of (tokenizer, hybrid_chunker)
    """
    if not config.use_semantic_chunking:
        msg = "This function requires semantic chunking enabled"
        raise ValueError(msg)

    tokenizer = get_tokenizer()
    hybrid_chunker = create_hybrid_chunker(tokenizer, config)
    return tokenizer, hybrid_chunker
