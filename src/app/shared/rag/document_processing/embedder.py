"""
Document embedding generation for vector search using Gemini.
"""

import asyncio
import hashlib
from datetime import datetime
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from app.utils import logger

from .models import Chunk

# Gemini embedding configuration
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_TASK_TYPE = "retrieval_document"


def get_model_config(model: str) -> dict[str, Any]:
    """Get model-specific configuration."""
    configs = {
        "gemini-embedding-001": {"dimensions": 1536, "max_tokens": 2048},
        "gemini-embedding": {"dimensions": 1536, "max_tokens": 2048},
    }
    return configs.get(model, {"dimensions": 1536, "max_tokens": 2048})


async def generate_embedding(  # noqa: RET503
    text: str,
    model: str = GEMINI_EMBEDDING_MODEL,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed
        model: Gemini embedding model to use
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Embedding vector
    """

    client = genai.Client()
    config = get_model_config(model)

    # Truncate text if too long
    if len(text) > config["max_tokens"] * 4:
        text = text[: config["max_tokens"] * 4]

    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model,
                contents=text,
                config={"task_type": GEMINI_TASK_TYPE},
            )
            return response.embedding.values

        except genai_errors.ClientError as e:
            logger.error(f"Gemini API error: {e}")
            if attempt == max_retries - 1:
                raise
            delay = retry_delay * (2**attempt)
            logger.warning(f"Rate limit hit, retrying in {delay}s")
            await asyncio.sleep(delay)

        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay)


async def generate_embeddings_batch(  # noqa: RET503
    texts: list[str],
    model: str = GEMINI_EMBEDDING_MODEL,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of texts to embed
        model: Embedding model to use
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        List of embedding vectors
    """
    from google import genai

    client = genai.Client()
    config = get_model_config(model)

    # Filter and truncate texts
    processed_texts = []
    for text in texts:
        if not text or not text.strip():
            processed_texts.append("")
            continue

        if len(text) > config["max_tokens"] * 4:
            text = text[: config["max_tokens"] * 4]

        processed_texts.append(text)

    for attempt in range(max_retries):
        try:
            # Gemini doesn't support batch embedding in the same way
            # Process individually and collect
            embeddings = []
            for text in processed_texts:
                response = client.models.embed_content(
                    model=model,
                    contents=text,
                    config={"task_type": GEMINI_TASK_TYPE},
                )
                embeddings.append(response.embedding.values)

            return embeddings

        except genai_errors.ClientError as e:
            logger.error(f"Gemini API error in batch: {e}")
            if attempt == max_retries - 1:
                return await _process_embeddings_individually(
                    processed_texts, model, retry_delay
                )
            await asyncio.sleep(retry_delay)

        except Exception as e:
            logger.error(f"Unexpected error in batch embedding: {e}")
            if attempt == max_retries - 1:
                return await _process_embeddings_individually(
                    processed_texts, model, retry_delay
                )
            await asyncio.sleep(retry_delay)


async def _process_embeddings_individually(
    texts: list[str], model: str, retry_delay: float
) -> list[list[float]]:
    """
    Process texts individually as fallback.

    Args:
        texts: List of texts to embed
        model: Embedding model to use
        retry_delay: Delay between retries

    Returns:
        List of embedding vectors
    """
    embeddings = []
    config = get_model_config(model)

    for text in texts:
        try:
            if not text or not text.strip():
                embeddings.append([0.0] * config["dimensions"])
                continue

            embedding = await generate_embedding(text, model=model)
            embeddings.append(embedding)

            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            # Use zero vector as fallback
            embeddings.append([0.0] * config["dimensions"])

    return embeddings


async def embed_chunks(
    chunks: list[Chunk],
    model: str = GEMINI_EMBEDDING_MODEL,
    batch_size: int = 100,
    progress_callback: None | callable = None,
) -> list[Chunk]:
    """
    Generate embeddings for document chunks.

    Args:
        chunks: List of document chunks
        model: Embedding model to use
        batch_size: Number of texts to process in parallel
        progress_callback: Optional callback for progress updates

    Returns:
        Chunks with embeddings added
    """
    if not chunks:
        return chunks

    config = get_model_config(model)
    logger.info(f"Generating embeddings for {len(chunks)} chunks")

    embedded_chunks = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [chunk.content for chunk in batch_chunks]

        try:
            # Generate embeddings for this batch
            embeddings = await generate_embeddings_batch(batch_texts, model=model)

            # Add embeddings to chunks
            for chunk, embedding in zip(batch_chunks, embeddings):
                # Create a new chunk with embedding
                embedded_chunk = Chunk(
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    document_id=chunk.document_id,
                    metadata={
                        **chunk.metadata,
                        "embedding_model": model,
                        "embedding_generated_at": datetime.now().isoformat(),
                    },
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
                embedded_chunks.append(embedded_chunk)

            # Progress update
            current_batch = (i // batch_size) + 1
            if progress_callback:
                progress_callback(current_batch, total_batches)

            logger.info(f"Processed batch {current_batch}/{total_batches}")

        except Exception as e:
            logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")

            # Add chunks without embeddings as fallback
            for chunk in batch_chunks:
                chunk.metadata.update(
                    {
                        "embedding_error": str(e),
                        "embedding_generated_at": datetime.now().isoformat(),
                    }
                )
                chunk.embedding = [0.0] * config["dimensions"]
                embedded_chunks.append(chunk)

    logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
    return embedded_chunks


async def embed_query(
    query: str, model: str = GEMINI_EMBEDDING_MODEL
) -> list[float]:
    """
    Generate embedding for a search query.

    Args:
        query: Search query
        model: Embedding model to use

    Returns:
        Query embedding
    """
    return await generate_embedding(query, model=model)


def get_embedding_dimension(model: str = GEMINI_EMBEDDING_MODEL) -> int:
    """
    Get the dimension of embeddings for this model.

    Args:
        model: Embedding model

    Returns:
        Embedding dimension
    """
    config = get_model_config(model)
    return config["dimensions"]


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 1000):
        """Initialize cache."""
        self.cache: dict[str, list[float]] = {}
        self.access_times: dict[str, datetime] = {}
        self.max_size = max_size

    def get(self, text: str) -> list[float] | None:
        """Get embedding from cache."""
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            self.access_times[text_hash] = datetime.now()
            return self.cache[text_hash]
        return None

    def put(self, text: str, embedding: list[float]):
        """Store embedding in cache."""
        text_hash = self._hash_text(text)

        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.access_times.keys(), key=lambda k: self.access_times[k]
            )
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[text_hash] = embedding
        self.access_times[text_hash] = datetime.now()

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode()).hexdigest()


def create_embedding_cache(max_size: int = 1000) -> EmbeddingCache:
    """Factory function to create embedding cache."""
    return EmbeddingCache(max_size)


def create_cached_embedder(
    model: str = GEMINI_EMBEDDING_MODEL, cache_max_size: int = 1000
) -> callable:
    """
    Create a cached embedding generator.

    Args:
        model: Embedding model to use
        cache_max_size: Maximum cache size

    Returns:
        Cached embedding generation function
    """
    cache = EmbeddingCache(cache_max_size)

    async def cached_generate_embedding(text: str) -> list[float]:
        cached = cache.get(text)
        if cached is not None:
            return cached

        embedding = await generate_embedding(text, model=model)
        cache.put(text, embedding)
        return embedding

    return cached_generate_embedding
