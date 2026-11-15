"""The one embedding path: one client per process, one cache, task role on both sides.

Six paths existed before this module. Four of them are named by task B1; two are not:

1. ``features/search/embeddings.build_embedding_client`` — a fresh provider client per call.
2. ``features/documents/service`` — imported that same builder, so two features were already
   one path, and it was the only caller that declared a task type at all.
3. The superseded shared ingestion graph — duck-typed the callable through three candidate
   method names, embedded one text per call, and declared no task type.
4. The retrieval graph's equivalent helper — **byte-identical** to the superseded ingestion
   helper, and sharing its Redis keyspace. B1 counted both as one path; both legacy helpers are
   now gone.
5. ``rag/docling/embedder`` — the batch carve-out, fixed under A2/A3. Decision 15
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

**The task role has two provider surfaces, and they are not interchangeable.** The
``gemini-embedding-001`` generation takes ``task_type`` as a request field. The
``gemini-embedding-2`` generation accepts the field, answers 200, and silently ignores it —
verified here as bit-identical vectors for ``RETRIEVAL_QUERY``, ``RETRIEVAL_DOCUMENT``, and
unset. Its contract is an instruction prefix in the text (``task: search result | query: ...``
for queries, ``title: none | text: ...`` for documents). Sending ``task_type`` and no prefix is
therefore the *silent* failure this module was built to prevent: no error, no log line, just
the two roles sharing one projection again. ``_provider_input`` selects the surface by model
family, and the cache key digests the resulting payload rather than the caller's string.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.observability.logging import logger
from app.utils.embedding import normalize_embedding
from app.utils.exceptions import InfrastructureException, ValidationException
from app.utils.json_serializer import from_json_float_list, to_float_list_str

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Final

    from redis.asyncio import Redis


class EmbeddingTaskType(StrEnum):
    """Which side of the asymmetry a caller is on.

    On the ``gemini-embedding-001`` generation these values are the provider's own task-type
    strings. On ``gemini-embedding-2`` they select the instruction template, because that
    generation ignores the field. Either way the role is spelled once here rather than as a
    literal at each call site, so a typo is a name error at import rather than a silent
    symmetry regression at ingestion time.
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
#:
#: Bumped to ``v2`` with the instruction-prompt contract below: vectors cached from the
#: unprefixed text are the *same model at the same width* as the prefixed ones, so the key
#: material alone would not separate them — and mixing the two spaces is silent.
_CACHE_NAMESPACE = "embedding:v2"

#: Marker for the model family that puts the task in the text instead of in ``task_type``.
#: Substring rather than a set of ids: ``gemini-embedding-2``, its ``-preview`` predecessor,
#: and any ``models/``-prefixed spelling are one contract with one mechanism.
_V2_MODEL_MARKER = "gemini-embedding-2"

#: Google's documented asymmetric-retrieval templates. Only the query side names a task; the
#: document side is a title-plus-text frame. We never carry titles, so ``title: none`` is the
#: honest document form rather than a placeholder.
_QUERY_INSTRUCTION = "task: search result | query: {content}"
_DOCUMENT_INSTRUCTION = "title: none | text: {content}"


class EmbeddingQuotaExhaustedError(Exception):
    """The embedding provider's daily quota is spent (STRIDE residual D-3).

    Raised instead of letting a provider 429 ride as a generic exception, burn
    the Celery retry budget, and dead-letter untyped — measured 2026-11-14: the
    free-tier 1000/day cap failed 12/15 corpus ingestions with no typed signal
    anywhere in the persisted state.

    ``retryable`` is ``False`` **within this process**: no seconds- or
    minutes-scale backoff clears a daily cap, so re-drive is the next quota
    window's idempotent resume (the document lands in the non-terminal
    ``quota_exhausted`` status, which the resume driver picks up), not a
    sleeping worker holding a broker slot.
    """

    error_code: Final[str] = "embedding_quota_exhausted"
    retryable: Final[bool] = False

    def __init__(self, detail: str, *, model: str, provider_status: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.model = model
        self.provider_status = provider_status


#: Seconds-scale waits for a transient provider 429. A per-minute rate limit
#: clears inside this window; a daily cap does not — and that asymmetry is the
#: classification: a 429 that names the daily cap, or one that survives both
#: probes, is exhaustion rather than a rate limit. Monkeypatched by tests.
_QUOTA_PROBE_DELAYS_SECONDS: Final[tuple[float, ...]] = (2.0, 4.0)

#: Jitter spread added to each probe wait, de-correlating concurrent ingesters
#: that hit the same cap together. Drawn from the OS CSPRNG (microsecond
#: granularity) rather than the Mersenne Twister — same convention as the
#: billing dunning ladder.
_QUOTA_PROBE_JITTER_SECONDS: Final[float] = 0.5

#: Strings in the provider's own message that name the *daily* cap specifically
#: (as opposed to a per-minute/per-request rate limit). Lowercased matching.
#: Deliberately narrow: the generic "Resource has been exhausted" message does
#: NOT qualify — it fires for per-minute limits too, and only the probe window
#: can tell those apart. When the marker is absent, classification falls to the
#: probes (a per-minute limit clears inside them; a daily cap does not).
_DAILY_QUOTA_MARKERS: Final[tuple[str, ...]] = (
    "perday",
    "per_day",
    "per day",
    "quotafailure",
)


def _provider_status_of(exc: BaseException) -> int | None:
    """Recover an HTTP-ish status from a provider exception without importing its SDK.

    The same three-shape recognition as ``app.ai.graph.retry._status_code_of`` — a
    status attribute, a numeric ``code``, a status on an attached response — kept
    local so the embedding path does not couple to the retrieval-graph retry
    module's privates. Only ``int`` values count: several libraries use ``code``
    for a string identifier, and a string would silently never match.
    """
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _names_daily_quota(exc: BaseException) -> bool:
    """True when the provider's own message names the daily cap specifically."""
    text = " ".join(str(arg) for arg in exc.args).lower()
    return any(marker in text for marker in _DAILY_QUOTA_MARKERS)


async def _quota_guarded[T](call: Callable[[], Awaitable[T]], *, model: str) -> T:
    """Run one provider embed call with the quota-aware behaviour of D-3.

    A transient 429 is probed at seconds-scale with jitter — a per-minute rate
    limit clears inside the window. A 429 that names the daily cap, or one that
    survives both probes, is quota exhaustion: raised as
    ``EmbeddingQuotaExhaustedError`` (never hours-slept in a worker), chained to the
    original provider error. Every non-429 error propagates exactly as before —
    this guard only adds classification, never changes other failure shapes.
    """
    for attempt, delay in enumerate((0.0, *_QUOTA_PROBE_DELAYS_SECONDS)):
        if delay:
            jitter_micros = secrets.randbelow(int(_QUOTA_PROBE_JITTER_SECONDS * 1_000_000) + 1)
            await asyncio.sleep(delay + jitter_micros / 1_000_000)
        try:
            return await call()
        except Exception as exc:
            if _provider_status_of(exc) != 429:
                raise
            if _names_daily_quota(exc) or attempt == len(_QUOTA_PROBE_DELAYS_SECONDS):
                detail = (
                    f"Embedding provider daily quota exhausted for model {model} "
                    f"({type(exc).__name__}: {exc})"
                )
                logger.warning(
                    "embedding_quota_exhausted",
                    model=model,
                    provider_status=_provider_status_of(exc),
                    probes_spent=attempt,
                )
                raise EmbeddingQuotaExhaustedError(
                    detail,
                    model=model,
                    provider_status=_provider_status_of(exc),
                ) from exc
    unreachable = "unreachable: probe loop must return or raise"
    raise AssertionError(unreachable)


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


def _uses_instruction_prompts(model: str) -> bool:
    """Whether this model expects the task in the text rather than in ``task_type``.

    ``gemini-embedding-001`` honours ``task_type``. The ``gemini-embedding-2`` family does
    not: it silently ignores the field and produces identical vectors for both roles, so the
    asymmetry the enum exists for would be lost with no error. Google's contract for that
    family is an instruction prefix, documented per use case.
    """
    return _V2_MODEL_MARKER in model


def _provider_input(text: str, task_type: EmbeddingTaskType, model: str) -> str:
    """Return the exact string the provider must embed for this role.

    For ``task_type`` models this is the caller's text unchanged. For instruction-prompt
    models it is the documented asymmetric-retrieval template — deliberately not a generic
    "Represent this..." string, because the family was trained on these exact frames.

    Raises:
        ValidationException: a task type was added to the enum without a v2 template, so the
            alternative is silently embedding the query side as a document.
    """
    if not _uses_instruction_prompts(model):
        return text
    if task_type is EmbeddingTaskType.QUERY:
        return _QUERY_INSTRUCTION.format(content=text)
    if task_type is EmbeddingTaskType.DOCUMENT:
        return _DOCUMENT_INSTRUCTION.format(content=text)
    msg = f"No instruction template for task type {task_type.value!r} on {model}"
    raise ValidationException(detail=msg, data={"model": model, "task_type": task_type.value})


def _cache_key(text: str, *, model: str, task_type: EmbeddingTaskType, dimension: int) -> str:
    """Digest the payload together with everything that changes what the vector means.

    The ``text`` argument is the provider payload (``_provider_input``), not necessarily the
    caller's string: for instruction-prompt models the role lives in the text, so keying the
    raw string would let one role serve the other. The ``task_type`` remains in the material
    as well — it costs nothing and keeps keys stable if a model family changes surfaces.

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

    The text handed to the provider is ``_provider_input``'s answer, not necessarily the
    caller's string: instruction-prompt models need the role in the payload, and the cache
    key digests that exact payload so the two spaces can never be read for one another.

    Args:
        text: The string to embed.
        task_type: Which side of the query/document asymmetry this call is on. Required.
        redis: Shared cache. ``None`` disables caching for this call, which is what a unit
            test without a cache backend passes.

    Returns:
        The vector, at the configured width.

    Cache reads and writes run under the configured Redis deadline and every
    cache error is a miss, never a retrieval failure: an unreachable or hung
    Redis costs a re-embed (or a skipped write-back), not the caller's request —
    and a failed write never discards an already-produced embedding.
    """
    resolved: EmbeddingTaskType = _require_task_type(task_type)
    client: GoogleGenerativeAIEmbeddings = get_embedding_client()
    settings = get_settings()
    model: str = settings.GEMINI_EMBEDDING_MODEL
    provider_text: str = _provider_input(text, resolved, model)
    key: str = _cache_key(
        provider_text,
        model=model,
        task_type=resolved,
        dimension=settings.EMBEDDING_DIMENSION,
    )

    if redis is not None:
        try:
            async with asyncio.timeout(settings.DOCUMENTS_RETRIEVAL_REDIS_TIMEOUT_SECONDS):
                cached: object = await redis.get(key)
        except Exception:  # noqa: BLE001 — degraded to a cache miss below, never surfaced
            logger.bind(operation="embedding_cache_read").warning("embedding_cache_read_failed")
            cached = None
        if cached:
            # Normalised on read as well as on write. The width is already right for any entry
            # this module wrote, so this is a no-op in the ordinary case — but it is not free
            # of value: `normalize_embedding` logs when a width disagrees, which turns an entry
            # written by something else into a warning rather than a wrong-width vector.
            return normalize_embedding(from_json_float_list(_decode_cached(cached)))

    if _uses_instruction_prompts(model):
        vector: list[float] = normalize_embedding(
            await _quota_guarded(lambda: client.aembed_query(provider_text), model=model)
        )
    else:
        vector = normalize_embedding(
            await _quota_guarded(
                lambda: client.aembed_query(provider_text, task_type=resolved.value),
                model=model,
            )
        )
    if redis is not None:
        try:
            async with asyncio.timeout(settings.DOCUMENTS_RETRIEVAL_REDIS_TIMEOUT_SECONDS):
                await redis.setex(key, CACHE_TTL_SECONDS, to_float_list_str(vector))
        except Exception:  # noqa: BLE001 — the write is skipped; the vector still returns
            logger.bind(operation="embedding_cache_write").warning("embedding_cache_write_failed")
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
    model: str = settings.GEMINI_EMBEDDING_MODEL
    provider_texts: list[str] = [_provider_input(text, resolved, model) for text in texts]
    keys: list[str] = [
        _cache_key(
            provider_text,
            model=model,
            task_type=resolved,
            dimension=settings.EMBEDDING_DIMENSION,
        )
        for provider_text in provider_texts
    ]

    results: dict[int, list[float]] = {}
    if redis is not None:
        for index, key in enumerate(keys):
            cached: object = await redis.get(key)
            if cached:
                results[index] = normalize_embedding(from_json_float_list(_decode_cached(cached)))

    missing: list[int] = [index for index in range(len(texts)) if index not in results]
    if missing:
        provider_batch: list[str] = [provider_texts[index] for index in missing]
        if _uses_instruction_prompts(model):
            fresh: list[list[float]] = await _quota_guarded(
                lambda: client.aembed_documents(
                    provider_batch,
                    batch_size=DOCUMENT_BATCH_SIZE,
                ),
                model=model,
            )
        else:
            fresh = await _quota_guarded(
                lambda: client.aembed_documents(
                    provider_batch,
                    task_type=resolved.value,
                    batch_size=DOCUMENT_BATCH_SIZE,
                ),
                model=model,
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
