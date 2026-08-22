"""
Document embedding generation for vector search using Gemini.

Batch-only carve-out
--------------------
No request path and no ingestion stage calls into this module; its only callers
are the offline ``ingest.py`` and ``ingest_v2.py``. It must stay that way. Note
that the boundary is enforced by who *calls*, not by who imports: the package
``__init__`` re-exports ``create_embedder``, ``embed_chunks``,
``generate_embedding`` and ``generate_embeddings_batch``, and
``features/documents/classification.py:10`` imports *from the package*, so a
feature module already loads this one today. Collapsing the four embedding paths
into one is task B1's; here the job is only to stop producing invalid vectors.

Why there is no model-keyed dimension table any more
---------------------------------------------------
There was one, and every entry read 1536 against vector columns declared at 768.
The lookup made it worse than a stale literal: the configured model is
``gemini-embedding-2-preview`` (``config/settings.py:213``), which was **absent
from the table**, so ``configs.get(model, ...)`` always fell through to its 1536
default — the map could not have been right for the deployed configuration no
matter what its entries said. Width now comes from the single configured value,
``EMBEDDING_DIMENSION``, which ``config/settings.py:46`` already cross-checks
against the configured model.
"""

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime

from google import genai
from google.genai import errors as genai_errors

from app.utils import logger
from app.utils.exceptions import ExternalServiceException

from .models import Chunk

# The provider model id this module passes to the API. Deliberately not read from
# ``settings.GEMINI_EMBEDDING_MODEL`` and — as of this task — deliberately no
# longer *named* like it. The two diverge today ("gemini-embedding-001" here
# against "gemini-embedding-2-preview" in configuration), and the old name made
# that divergence read as configuration. Reconciling the model, not just the
# width, belongs to B1, which collapses the four embedding paths into one.
_PROVIDER_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_TASK_TYPE = "retrieval_document"

# Provider input bound, in tokens. Not a setting: no configured value exists for
# it, and it describes the provider endpoint rather than this project.
_MAX_INPUT_TOKENS = 2048

_SERVICE_NAME = "Gemini embeddings"


def get_embedding_dimension() -> int:
    """Width every vector this module produces must have.

    Read from configuration on each call rather than captured at import, so that
    a test overriding the setting is not defeated by module load order. The model
    argument this used to take is gone: dimension is a property of the
    application's configured corpus, not of whichever model id a caller passes.
    """
    # Lazy, mirroring utils/embedding.py:13 — this module is loaded during
    # `app.config`-adjacent import chains, and reading the setting per call also
    # keeps a test's override from losing a race with module load order.
    from app.config import get_settings

    return get_settings().EMBEDDING_DIMENSION


def _provider_failure(detail: str, *, model: str, text_count: int) -> ExternalServiceException:
    """Build the one typed failure this module raises.

    Each note carries a value the traceback alone does not: which model was
    asked, under which task type, and how many texts were in flight. The notes go
    on the *raised* exception rather than the provider's, because this changes
    exception type — ``EXCEPTION-RULES.md`` reserves ``add_note``-then-bare-raise
    for the same-type case and prescribes ``raise ... from e`` here, which is what
    every call site below does to preserve ``__cause__``.

    What this replaces is the point: these sites used to append a zero vector of
    the configured width. Such a row inserts cleanly and ranks against nothing, so
    a provider outage became an invisible hole in the corpus instead of an error
    anybody could see. (Deliberately described rather than quoted — A2's proof is
    a grep for the literal, and prose that spells it out would defeat the guard.)
    """
    exc = ExternalServiceException(service=_SERVICE_NAME, detail=detail)
    exc.add_note(f"model={model}")
    exc.add_note(f"task_type={GEMINI_TASK_TYPE}")
    exc.add_note(f"text_count={text_count}")
    return exc


def _validated_width(embedding: list[float], *, model: str, text_count: int) -> list[float]:
    """Return ``embedding`` unchanged, or raise if its width is not the configured one.

    This is what makes ``get_embedding_dimension()`` load-bearing rather than
    decorative, and it is the check the deleted dimension table pretended to be.

    It raises rather than routing through ``utils/embedding.normalize_embedding``
    deliberately. That helper truncates or zero-pads with a warning, which is
    right for a near-miss but not here: the mismatch this module can actually
    produce is a *model* mismatch, where the widths differ by a factor rather
    than a few positions. Truncating such a vector yields a row that inserts,
    ranks, and means nothing — the same invisible corruption the zero vectors
    caused, only harder to spot because the value is not obviously degenerate.
    """
    expected = get_embedding_dimension()
    actual = len(embedding)
    if actual != expected:
        msg = f"provider returned a {actual}-dimensional vector; EMBEDDING_DIMENSION is {expected}"
        raise _provider_failure(msg, model=model, text_count=text_count)
    return embedding


async def generate_embedding(  # noqa: RET503
    text: str,
    model: str = _PROVIDER_EMBEDDING_MODEL,
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

    Raises:
        ExternalServiceException: the text is empty, or the provider failed on
            every attempt. Never returns a substitute vector.
    """
    if not text or not text.strip():
        msg = "refusing to embed empty text"
        raise _provider_failure(msg, model=model, text_count=1)

    client = genai.Client()

    # Truncate text if too long
    if len(text) > _MAX_INPUT_TOKENS * 4:
        text = text[: _MAX_INPUT_TOKENS * 4]

    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model,
                contents=text,
                config={"task_type": GEMINI_TASK_TYPE},
            )
        except genai_errors.ClientError as e:
            logger.error("Gemini API error: {}", e)
            if attempt == max_retries - 1:
                raise
            delay = retry_delay * (2**attempt)
            logger.warning("Rate limit hit, retrying in {}s", delay)
            await asyncio.sleep(delay)

        except genai_errors.APIError as e:
            e.add_note(f"model={model}, operation=generate_embedding")
            logger.error(f"Unexpected error generating embedding: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay)
        else:
            return response.embedding.values


async def generate_embeddings_batch(  # noqa: RET503
    texts: list[str],
    model: str = _PROVIDER_EMBEDDING_MODEL,
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
        List of embedding vectors, positionally aligned with ``texts``.

    Raises:
        ExternalServiceException: any text is blank, or the provider failed on
            every attempt. The returned list is either complete or absent —
            there is no partial batch and no substitute vector.
    """
    client = genai.Client()

    # Blank texts used to be replaced with "" here to keep the result
    # positionally aligned, and the alignment placeholder then became a zero
    # vector downstream. Both halves of that are wrong: a blank chunk is a
    # chunking defect, and a zero-vector row is invisible rather than absent. So
    # the batch is rejected as a whole, naming how many were blank.
    blank_count = sum(1 for text in texts if not text or not text.strip())
    if blank_count:
        msg = f"refusing to embed a batch containing {blank_count} blank text(s)"
        raise _provider_failure(msg, model=model, text_count=len(texts))

    processed_texts = [
        text[: _MAX_INPUT_TOKENS * 4] if len(text) > _MAX_INPUT_TOKENS * 4 else text
        for text in texts
    ]

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
                embeddings.append(
                    _validated_width(
                        response.embedding.values, model=model, text_count=len(processed_texts)
                    )
                )

        except genai_errors.ClientError as e:
            logger.error("Gemini API error in batch: {}", e)
            if attempt == max_retries - 1:
                return await _process_embeddings_individually(processed_texts, model, retry_delay)
            await asyncio.sleep(retry_delay)

        except genai_errors.APIError as e:
            e.add_note(f"model={model}, operation=generate_embeddings_batch")
            logger.error(f"Unexpected error in batch embedding: {e}")
            if attempt == max_retries - 1:
                return await _process_embeddings_individually(processed_texts, model, retry_delay)
            await asyncio.sleep(retry_delay)
        else:
            return embeddings


async def _process_embeddings_individually(
    texts: list[str], model: str, _retry_delay: float
) -> list[list[float]]:
    """
    Process texts individually as fallback.

    Args:
        texts: List of texts to embed
        model: Embedding model to use
        retry_delay: Delay between retries

    Returns:
        List of embedding vectors, one per text.

    Raises:
        ExternalServiceException: the provider failed for any single text. The
            fallback is per-text retrying, not per-text tolerance — one
            unembeddable text fails the batch.
    """
    # No blank branch: the only caller validates the batch before reaching here,
    # and `generate_embedding` rejects a blank text on its own account. A blank
    # arriving here would be a caller defect, and it raises rather than becoming
    # a zero vector.
    embeddings = []

    for text in texts:
        try:
            embedding = await generate_embedding(text, model=model)
        except genai_errors.APIError as e:
            logger.error("Failed to embed text: {}", e)
            msg = "provider failed while embedding a batch text individually"
            raise _provider_failure(msg, model=model, text_count=len(texts)) from e
        embeddings.append(_validated_width(embedding, model=model, text_count=len(texts)))
        await asyncio.sleep(0.1)

    return embeddings


async def embed_chunks(
    chunks: list[Chunk],
    model: str = _PROVIDER_EMBEDDING_MODEL,
    batch_size: int = 100,
    progress_callback: Callable[..., None] | None = None,
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

    Raises:
        ExternalServiceException: any batch failed. Previously such a batch was
            returned with every chunk carrying a zero vector and an
            ``embedding_error`` string in its metadata — a shape indistinguishable
            from success to every caller that only checks whether a list came
            back.
    """
    if not chunks:
        return chunks

    logger.info("Generating embeddings for {} chunks", len(chunks))

    embedded_chunks = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [chunk.content for chunk in batch_chunks]
        current_batch = (i // batch_size) + 1

        try:
            embeddings = await generate_embeddings_batch(batch_texts, model=model)
            _attach_embeddings_to_chunks(batch_chunks, embeddings, model, embedded_chunks)
        except ExternalServiceException as e:
            # Same exception type, so add_note-then-bare-raise is the prescribed
            # form here; `raise ... from` is for the type change one level down.
            e.add_note(f"batch={current_batch}/{total_batches}")
            e.add_note(f"chunks_embedded_before_failure={len(embedded_chunks)}")
            logger.error("Failed to process batch {}/{}", current_batch, total_batches)
            raise

        if progress_callback:
            progress_callback(current_batch, total_batches)
        logger.info("Processed batch {}/{}", current_batch, total_batches)

    logger.info("Generated embeddings for {} chunks", len(embedded_chunks))
    return embedded_chunks


def _attach_embeddings_to_chunks(
    batch_chunks: list[Chunk],
    embeddings: list[list[float]],
    model: str,
    embedded_chunks: list[Chunk],
) -> None:
    for chunk, embedding in zip(batch_chunks, embeddings, strict=True):
        embedded_chunks.append(
            Chunk(
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                document_id=chunk.document_id,
                metadata={
                    **chunk.metadata,
                    "embedding_model": model,
                    "embedding_generated_at": datetime.now(tz=UTC).isoformat(),
                },
                token_count=chunk.token_count,
                embedding=embedding,
            )
        )


async def embed_query(query: str, model: str = _PROVIDER_EMBEDDING_MODEL) -> list[float]:
    """
    Generate embedding for a search query.

    Args:
        query: Search query
        model: Embedding model to use

    Returns:
        Query embedding
    """
    return await generate_embedding(query, model=model)


class _Embedder:
    """Object form of this module's functions, for callers that hold an embedder.

    ``embed_query`` is here because task A4's four call sites in
    ``rag_agent_advanced.py`` call it (``:201``, ``:270``). Retargeting their
    imports without it would only trade ``ModuleNotFoundError`` for
    ``AttributeError`` — still a first-call failure, still deferred, still exactly
    what A4 says is unacceptable.
    """

    embed_chunks = staticmethod(embed_chunks)
    embed_query = staticmethod(embed_query)


def create_embedder() -> _Embedder:
    return _Embedder()


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
            self.access_times[text_hash] = datetime.now(tz=UTC)
            return self.cache[text_hash]
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache."""
        text_hash = self._hash_text(text)

        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[text_hash] = embedding
        self.access_times[text_hash] = datetime.now(tz=UTC)

    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


def create_embedding_cache(max_size: int = 1000) -> EmbeddingCache:
    """Factory function to create embedding cache."""
    return EmbeddingCache(max_size)


def create_cached_embedder(
    model: str = _PROVIDER_EMBEDDING_MODEL, cache_max_size: int = 1000
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
