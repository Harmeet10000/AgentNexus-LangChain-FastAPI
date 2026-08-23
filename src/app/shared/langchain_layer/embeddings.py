"""The one embedding path: one client per process, one cache, task type on both sides.

Six paths existed before this module. Four of them are named by task B1; two are not:

1. ``features/search/embeddings.build_embedding_client`` — a fresh provider client per call.
2. ``features/documents/service`` — imported that same builder, so two features were already
   one path, and it was the only caller that declared a task type at all.
3. ``langgraph_layer/ingestion_kb/nodes._call_embedding_fn`` — duck-typed the callable through
   three candidate method names, embedded one text per call, declared no task type.
4. ``langgraph_layer/retrieval_kb/nodes._call_embedding_fn`` — **byte-identical** to (3), and
   sharing its Redis keyspace. B1 counts (3) and (4) as one path; they are two files.
5. ``rag/document_processing/embedder`` — the batch carve-out, fixed under A2/A3. Decision 15
   keeps it batch-only, so it is deliberately **not** folded in here.
6. ``langchain_layer/models.aembed_text`` / ``aembed_batch`` — dead: nothing imported them, and
   they offloaded the *synchronous* provider method to a thread while the client has native
   async methods. Deleted rather than unified, because a dead duplicate is the kind that regrows.

Three things about this module are load-bearing and easy to undo by accident.

**The task type is required, not defaulted.** Gemini embeddings are asymmetric: a vector for a
document and a vector for a query are drawn from different projections of the same model, and
comparing one against the other degrades relevance without erroring. Paths (3) and (4) passed no
task type at all, so every stored vector was asymmetric with every query vector. A default here
would silently reintroduce that, so callers must say which side they are on.

**The cache key digests the model, the task type, and the configured width along with the text.**
Paths (3) and (4) keyed on the text alone *and shared one prefix*, so the same string embedded as
a document and as a query collided on one entry. That was harmless only because neither declared
a task type — the moment one side declares it, a text-only key serves the query side a
document-side vector. Adding task types without re-keying would therefore have been worse than
leaving both wrong, which is why B1 and B2 land together.

**Normalisation happens before the cache write, not after.** ``documents/service`` wrote the raw
vector and normalised the value it returned, so a cache miss produced a width-corrected vector
and a cache hit produced the raw one. That only diverges when the provider returns an unexpected
width — but that is exactly the condition A3's guard exists for, and the two paths disagreeing
made it depend on cache warmth.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.utils.embedding import normalize_embedding
from app.utils.exceptions import InfrastructureException, ValidationException
from app.utils.json_serializer import from_json_float_list, to_float_list_str
from app.utils.logger import logger

if TYPE_CHECKING:
    from redis.asyncio import Redis


class EmbeddingTaskType(StrEnum):
    """Which side of the asymmetry a caller is on.

    The values are the provider's own task-type strings. They are spelled here once so no
    caller repeats a string literal the provider validates server-side — a typo in one
    would surface as a provider error at ingestion time rather than a name error at import.
    """

    QUERY = "RETRIEVAL_QUERY"
    DOCUMENT = "RETRIEVAL_DOCUMENT"


#: How long a cached vector stays valid. Matches what all three prior caches used, so this
#: change alters no expiry behaviour while it consolidates them.
CACHE_TTL_SECONDS = 60 * 60 * 24

#: Batch size handed to the provider's document call. B1 requires the batched form to declare
#: one rather than inherit it, so a change in the provider's default cannot silently change
#: our request shape. 100 is the provider's own default, chosen so this is a pin, not a tuning.
DOCUMENT_BATCH_SIZE = 100

#: Cache namespace. Deliberately distinct from the two prefixes it replaces, so entries written
#: by the text-only keying are orphaned rather than read back under the new contract. They
#: expire on their own within `CACHE_TTL_SECONDS`; nothing needs to delete them.
_CACHE_NAMESPACE = "embedding:v1"


@lru_cache(maxsize=1)
def get_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """Return the process-wide embedding client, constructing it at most once.

    Every prior path built a fresh client per call, which re-read settings and re-established
    provider state on each embedding — and made the per-process client B1 asks for unprovable.
    ``maxsize=1`` rather than an unbounded cache because the function takes no arguments: there
    is exactly one client, and a bound of one says so.

    ``task_type`` is a constructor field on this client as well as a per-call argument, and it
    is deliberately **not** set here: one client serves both sides of the asymmetry, so binding
    a task type at construction would either need two clients or make the per-call argument's
    precedence load-bearing. ``output_dimensionality`` is set here because it is a property of
    the deployment, not of the call.

    Call ``get_embedding_client.cache_clear()`` to force a rebuild. Tests must, because a
    singleton warmed by one test makes another's "constructed once" assertion vacuous, and a
    client built against one test's settings would otherwise leak into the next.
    """
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        api_key=settings.GEMINI_API_KEY,
        # The width this client *produces*, which must match the width the columns *store*.
        # A producer pinned to a literal against a configurable column turns a one-line config
        # change into an insert error at ingestion time instead of a validation error at startup.
        output_dimensionality=settings.EMBEDDING_DIMENSION,
    )


def _require_task_type(task_type: EmbeddingTaskType | str) -> EmbeddingTaskType:
    """Coerce and validate the task type, rejecting anything the provider would reject.

    Callers are typed to pass the enum, so this guard is for the untyped edges — a value read
    from configuration, or a plain string that looked right. Rejecting here names the offending
    value; letting it through names nothing until the provider answers.

    Raises:
        ValidationException: the value is not one of the provider's task types.
    """
    try:
        return EmbeddingTaskType(task_type)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in EmbeddingTaskType)
        msg = f"Unknown embedding task type; expected one of: {allowed}"
        raise ValidationException(
            detail=msg,
            data={"allowed": allowed},
        ) from exc


def _cache_key(text: str, *, model: str, task_type: EmbeddingTaskType, dimension: int) -> str:
    """Digest the text together with everything that changes what the vector means.

    The NUL separators are not decoration. Concatenating the parts unseparated would let two
    different triples produce one digest — a model named ``ab`` with task ``c`` and a model
    named ``a`` with task ``bc`` are different requests and must not share an entry.

    ``dimension`` is in the digest even though Decision 4 names only text, model, and task type.
    The configured width is not derivable from the model id — the same model serves several
    widths — so without it a deployment that changes ``EMBEDDING_DIMENSION`` would read back
    vectors of the previous width from a warm cache. That is the one failure this cache could
    cause that re-embedding would not fix, because the wrong-width vector looks valid.
    """
    material = f"{model}\x00{task_type.value}\x00{dimension}\x00{text}".encode()
    return f"{_CACHE_NAMESPACE}:{hashlib.sha256(material).hexdigest()}"


def _decode_cached(cached: object) -> str:
    """Return the cached payload as text, whether the client decodes responses or not.

    The prior helpers all called ``str(cached)`` directly, which is correct only because
    ``connections/redis.py`` builds its client with ``decode_responses=True``. Handed a client
    without it, ``str()`` on the returned ``bytes`` yields the *repr* — ``"b'[0.1, ...]'"`` —
    which is not JSON, and the resulting parse error names pydantic rather than the encoding.
    Decoding explicitly costs nothing on the configured path and removes that footgun.
    """
    if isinstance(cached, bytes | bytearray | memoryview):
        return bytes(cached).decode("utf-8")
    return str(cached)


async def embed_text(
    text: str,
    *,
    task_type: EmbeddingTaskType,
    redis: Redis | None = None,
) -> list[float]:
    """Embed one string, serving it from the shared cache when present.

    This calls the provider's single-text method. It does **not** wrap the text in a
    one-element batch: a batch of one costs the same request and returns a nested result the
    caller has to unwrap, and B1 asks for a single-text form that is genuinely one.

    Args:
        text: The string to embed.
        task_type: Which side of the query/document asymmetry this call is on. Required.
        redis: Shared cache. ``None`` disables caching for this call, which is what a unit
            test without a cache backend passes.

    Returns:
        The vector, at the configured width.
    """
    resolved: EmbeddingTaskType = _require_task_type(task_type)
    client: GoogleGenerativeAIEmbeddings = get_embedding_client()
    settings = get_settings()
    key: str = _cache_key(
        text,
        model=settings.GEMINI_EMBEDDING_MODEL,
        task_type=resolved,
        dimension=settings.EMBEDDING_DIMENSION,
    )

    if redis is not None:
        cached: object = await redis.get(key)
        if cached:
            # Normalised on read as well as on write. The width is already right for any entry
            # this module wrote, so this is a no-op in the ordinary case — but it is not free
            # of value: `normalize_embedding` logs when a width disagrees, which turns an entry
            # written by something else into a warning rather than a wrong-width vector.
            return normalize_embedding(from_json_float_list(_decode_cached(cached)))

    vector: list[float] = normalize_embedding(
        await client.aembed_query(text, task_type=resolved.value)
    )
    if redis is not None:
        await redis.setex(key, CACHE_TTL_SECONDS, to_float_list_str(vector))
    return vector


async def embed_texts(
    texts: list[str],
    *,
    task_type: EmbeddingTaskType,
    redis: Redis | None = None,
) -> list[list[float]]:
    """Embed several strings, serving each from the shared cache independently.

    Cache granularity is per text, not per batch. A batch keyed as a whole would miss whenever
    any one member changed, which for document ingestion is every re-run of a document with one
    edited clause.

    Args:
        texts: The strings to embed. Order is preserved in the result.
        task_type: Which side of the query/document asymmetry this call is on. Required.
        redis: Shared cache. ``None`` disables caching for this call.

    Returns:
        One vector per input, in input order, each at the configured width.
    """
    resolved: EmbeddingTaskType = _require_task_type(task_type)
    if not texts:
        return []

    client: GoogleGenerativeAIEmbeddings = get_embedding_client()
    settings = get_settings()
    keys: list[str] = [
        _cache_key(
            text,
            model=settings.GEMINI_EMBEDDING_MODEL,
            task_type=resolved,
            dimension=settings.EMBEDDING_DIMENSION,
        )
        for text in texts
    ]

    results: dict[int, list[float]] = {}
    if redis is not None:
        for index, key in enumerate(keys):
            cached: object = await redis.get(key)
            if cached:
                results[index] = normalize_embedding(from_json_float_list(_decode_cached(cached)))

    missing: list[int] = [index for index in range(len(texts)) if index not in results]
    if missing:
        fresh: list[list[float]] = await client.aembed_documents(
            [texts[index] for index in missing],
            task_type=resolved.value,
            batch_size=DOCUMENT_BATCH_SIZE,
        )
        if len(fresh) != len(missing):
            # Without this the `zip` below would pair vectors with the wrong texts from the
            # first gap onward, and every vector after it would be silently misattributed —
            # stored against a clause it does not describe, with nothing raising.
            msg = (
                f"Embedding provider returned {len(fresh)} vectors for {len(missing)} texts; "
                f"the result cannot be aligned with its inputs"
            )
            raise InfrastructureException(
                detail=msg,
                # A malformed provider response is plausibly transient, so this is the one
                # condition here worth retrying. Stated rather than defaulted, because the
                # default is `False` and silence would read as a considered `False`.
                retryable=True,
                data={"requested": len(missing), "returned": len(fresh)},
            )
        for index, vector in zip(missing, fresh, strict=True):
            normalized: list[float] = normalize_embedding(vector)
            results[index] = normalized
            if redis is not None:
                await redis.setex(keys[index], CACHE_TTL_SECONDS, to_float_list_str(normalized))

    logger.bind(
        requested=len(texts),
        from_cache=len(texts) - len(missing),
        task_type=resolved.value,
    ).debug("embedding_batch_served")
    # Indexed rather than filtered. A comprehension that dropped unfilled slots would return a
    # shorter list than it was given and misalign the caller's own zip; a missing key here is a
    # `KeyError` naming the index, which is the failure the caller can act on.
    return [results[index] for index in range(len(texts))]
