"""Unit tests for tasks B1 and B2 — one embedding client, one cache, task type on both sides.

Six paths produced embeddings before this module existed. Four of them are named by B1; the
other two were a dead pair in ``langchain_layer/models.py`` and a private factory that
outlived its only callers. The tests here pin the three properties that were violated, and
each one corresponds to a defect that shipped rather than to a hypothetical.

**Why a construction spy and not an identity assertion.** ``client_a is client_b`` cannot tell
a per-process cache apart from a module-level singleton built at import time, and it says
nothing about how many provider clients were constructed. The spy counts constructions, so it
fails for the right reason.

**Why the cache tests use two Redis clients over one store.** B2 asks for a cache "visible to a
second process". A single client instance cannot show that: an in-process dict would satisfy
every assertion made through one handle. Two ``FakeRedis`` clients sharing one ``FakeServer``
is the honest analogue — separate connections, one store — so a cache that had quietly become
process-local fails ``test_the_cache_is_visible_to_a_second_process`` and nothing else.

**Why B1 and B2 are tested together.** They cannot ship apart. The two prior
``_cached_embedding`` copies keyed on the text alone *and shared one Redis prefix*, which was
survivable only because neither declared a task type — both sides got vectors from the same
projection, mutually consistent and both wrong. Declaring task types without re-keying would
serve the query side a document-side vector, which is strictly worse than the status quo.
``test_query_and_document_vectors_for_one_text_occupy_distinct_cache_entries`` is the test that
would have caught that, and it is the reason the two tasks are one commit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

if TYPE_CHECKING:
    from typing import ClassVar

from app.config import get_settings
from app.shared.langchain_layer import embeddings as embeddings_module
from app.shared.langchain_layer.embeddings import (
    DOCUMENT_BATCH_SIZE,
    EmbeddingTaskType,
    _cache_key,
    embed_text,
    embed_texts,
    get_embedding_client,
)
from app.utils.exceptions import InfrastructureException, ValidationException
from app.utils.json_serializer import from_json_float_list

_TEXT = "The Supplier shall indemnify the Customer against all third-party claims."
_DIMENSION = get_settings().EMBEDDING_DIMENSION


def _vector(seed: float = 0.1, *, length: int = _DIMENSION) -> list[float]:
    return [seed] * length


class _ClientSpy:
    """Stands in for ``GoogleGenerativeAIEmbeddings`` and records how it was used.

    Constructor kwargs are captured on the class rather than the instance because the point of
    several assertions is what happened *at construction*, and the module hands the instance
    back through a cache the test cannot reach into for a second copy.
    """

    constructions: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).constructions.append(kwargs)
        self.query_calls: list[tuple[str, object]] = []
        self.document_calls: list[tuple[list[str], object, object]] = []

    async def aembed_query(self, text: str, *, task_type: object = None, **_kw: object):
        self.query_calls.append((text, task_type))
        return _vector()

    async def aembed_documents(
        self,
        texts: list[str],
        *,
        task_type: object = None,
        batch_size: object = None,
        **_kw: object,
    ):
        self.document_calls.append((list(texts), task_type, batch_size))
        return [_vector(0.1 + index / 1000) for index in range(len(texts))]

    @classmethod
    def count(cls) -> int:
        return len(cls.constructions)


@pytest.fixture(autouse=True)
def _cold_client_cache():
    """Bracket every test with a cold client cache and an empty construction log.

    Both ends matter. A client warmed by an earlier test makes "constructed once" vacuous;
    leaving a spy cached would hand a fake provider to the rest of the session through a
    process-wide accessor.
    """
    get_embedding_client.cache_clear()
    _ClientSpy.constructions = []
    yield
    get_embedding_client.cache_clear()
    _ClientSpy.constructions = []


@pytest.fixture
def spy(monkeypatch) -> type[_ClientSpy]:
    monkeypatch.setattr(embeddings_module, "GoogleGenerativeAIEmbeddings", _ClientSpy)
    return _ClientSpy


@pytest.fixture
def shared_store() -> FakeServer:
    """One store, so tests can open more than one client against it."""
    return FakeServer()


def _client(store: FakeServer, *, decode_responses: bool = True) -> FakeRedis:
    """A cache client shaped like the one production builds.

    ``decode_responses=True`` mirrors ``connections/redis.py:48``, and it is not incidental:
    with it a ``get`` returns ``str``, without it ``bytes``. Every prior ``_cached_embedding``
    called ``str()`` on the result, which is a no-op on the former and yields a ``bytes`` repr
    on the latter. Defaulting to the production shape here keeps these tests honest about what
    runs; ``test_a_client_that_returns_bytes_is_read_correctly`` covers the other configuration.
    """
    return FakeRedis(server=store, decode_responses=decode_responses)


# ---------------------------------------------------------------------------
# B1 — one client per process, task type declared per call
# ---------------------------------------------------------------------------


async def test_the_client_is_constructed_once_per_process(spy) -> None:
    """Every prior path built a fresh client per call, re-reading settings each time."""
    await embed_text("first", task_type=EmbeddingTaskType.QUERY)
    await embed_text("second", task_type=EmbeddingTaskType.QUERY)
    await embed_texts(["third", "fourth"], task_type=EmbeddingTaskType.DOCUMENT)

    assert spy.count() == 1


async def test_the_first_call_in_a_cold_process_does_construct(spy) -> None:
    """Guards the test above from passing vacuously if the cold-cache fixture breaks."""
    assert spy.count() == 0

    await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY)

    assert spy.count() == 1


async def test_the_client_is_not_bound_to_a_task_type_at_construction(spy) -> None:
    """The property that lets one client serve both sides of the asymmetry.

    ``task_type`` is a constructor field on this provider client as well as a per-call
    argument. Binding it at construction would force either two clients or a dependence on
    which of the two wins — and the deleted ``models._build_embedding_model_gemini_full`` did
    exactly that, which is why it could never have served a query.
    """
    await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY)
    await embed_texts([_TEXT], task_type=EmbeddingTaskType.DOCUMENT)

    assert spy.count() == 1
    assert "task_type" not in spy.constructions[0]


async def test_the_constructor_receives_the_configured_width(spy) -> None:
    """A producer pinned to a literal against a configurable column fails at insert time."""
    await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY)

    assert spy.constructions[0]["output_dimensionality"] == _DIMENSION


async def test_the_task_type_reaches_the_provider_on_each_call(spy) -> None:
    """The defect in paths (3) and (4): they passed no task type at all.

    Nothing errored, because the provider defaults rather than rejecting. Query vectors were
    drawn from the document projection and compared against document vectors from the same
    one — consistent, and both wrong, which is the hardest kind of wrong to notice.
    """
    await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY)
    await embed_texts([_TEXT], task_type=EmbeddingTaskType.DOCUMENT)

    client = get_embedding_client()
    assert client.query_calls == [(_TEXT, "RETRIEVAL_QUERY")]
    assert client.document_calls[0][0] == [_TEXT]
    assert client.document_calls[0][1] == "RETRIEVAL_DOCUMENT"


async def test_a_single_text_uses_the_single_text_provider_call(spy) -> None:
    """Not a batch of one: that costs the same request and returns a nested result."""
    await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY)

    client = get_embedding_client()
    assert len(client.query_calls) == 1
    assert client.document_calls == []


async def test_the_document_batch_size_is_pinned_rather_than_inherited(spy) -> None:
    """Stated so a change in the provider's default cannot change our request shape."""
    await embed_texts([_TEXT, "another"], task_type=EmbeddingTaskType.DOCUMENT)

    client = get_embedding_client()
    assert client.document_calls[0][2] == DOCUMENT_BATCH_SIZE


def test_the_task_type_has_no_default_so_it_cannot_be_omitted() -> None:
    """A default would silently reintroduce the defect the enum exists to prevent.

    Checked on the signature as well as by calling, because a default of ``QUERY`` would make
    the call below succeed and every document ingestion quietly wrong.
    """
    import inspect

    for function in (embed_text, embed_texts):
        parameter = inspect.signature(function).parameters["task_type"]
        assert parameter.default is inspect.Parameter.empty, (
            f"{function.__name__} must not default its task type"
        )
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


async def test_omitting_the_task_type_raises_rather_than_guessing(spy) -> None:
    with pytest.raises(TypeError):
        await embed_text(_TEXT)


async def test_an_unknown_task_type_is_rejected_before_the_provider_is_called(spy) -> None:
    """Named here, or named nowhere until the provider answers."""
    with pytest.raises(ValidationException) as caught:
        await embed_text(_TEXT, task_type="SEMANTIC_SIMILARITY")

    assert "RETRIEVAL_QUERY" in str(caught.value.detail)
    assert spy.count() == 0, "the provider client must not be built for a rejected task type"


async def test_a_valid_task_type_string_is_coerced(spy) -> None:
    """The untyped edges — a value read from configuration — still work."""
    await embed_text(_TEXT, task_type="RETRIEVAL_DOCUMENT")

    assert get_embedding_client().query_calls == [(_TEXT, "RETRIEVAL_DOCUMENT")]


# ---------------------------------------------------------------------------
# B2 — one cache, keyed by everything that changes what the vector means
# ---------------------------------------------------------------------------


async def test_query_and_document_vectors_for_one_text_occupy_distinct_cache_entries(
    spy, shared_store
) -> None:
    """The test that makes B1 safe to ship, and the reason it ships with B2.

    Both prior helpers keyed on the text alone under one shared ``kb:embedding:`` prefix. Add
    task types to that and the query side reads back a document-side vector — a regression with
    no error, no log line, and a measurable drop in relevance.
    """
    redis = _client(shared_store)

    await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=redis)
    await embed_text(_TEXT, task_type=EmbeddingTaskType.DOCUMENT, redis=redis)

    keys = sorted(await redis.keys("embedding:v1:*"))
    assert len(keys) == 2, "one text embedded two ways must not share a cache entry"

    client = get_embedding_client()
    assert [call[1] for call in client.query_calls] == ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]


async def test_a_repeated_text_is_served_from_the_cache_not_the_provider(spy, shared_store) -> None:
    redis = _client(shared_store)

    first = await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=redis)
    second = await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=redis)

    assert first == second
    assert len(get_embedding_client().query_calls) == 1


async def test_the_cache_is_visible_to_a_second_process(spy, shared_store) -> None:
    """Cross-process is the whole reason this is Redis and not an in-process LRU.

    A second process starts with a cold client cache and its own connection, which is what the
    two ``cache_clear`` calls and the second ``FakeRedis`` stand for. ``embedder.py`` holds an
    in-process LRU of exactly the shape this rules out; Decision 4 rejects it for the framework
    wrapper for exactly this reason, and the project had built one anyway.
    """
    first_process_redis = _client(shared_store)
    written = await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=first_process_redis)
    assert spy.count() == 1

    # A fresh process: nothing carried over in memory, only what is in the store.
    get_embedding_client.cache_clear()
    _ClientSpy.constructions = []
    second_process_redis = _client(shared_store)

    read = await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=second_process_redis)

    assert read == written
    assert get_embedding_client().query_calls == [], (
        "the second process re-embedded, so the cache is process-local rather than shared"
    )


async def test_a_client_that_returns_bytes_is_read_correctly(spy, shared_store) -> None:
    """The configuration this module does *not* get in production, handled anyway.

    ``connections/redis.py`` sets ``decode_responses=True``, so the prior helpers' bare
    ``str(cached)`` worked. Against a client without it, ``str()`` on ``bytes`` yields the repr
    ``"b'[0.1, ...]'"`` — not JSON — and the failure surfaces as a pydantic validation error
    naming neither Redis nor the encoding. One decode removes a whole confusing failure mode.
    """
    redis = _client(shared_store, decode_responses=False)

    written = await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=redis)
    read = await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=redis)

    assert read == written
    assert len(get_embedding_client().query_calls) == 1, "the bytes payload was not read back"


async def test_the_vector_is_normalised_before_it_is_written_not_only_on_return(
    spy, shared_store, monkeypatch
) -> None:
    """``documents/service`` wrote the raw vector and normalised only what it returned.

    A miss therefore produced a width-corrected vector and a hit produced the raw one, which
    made correctness depend on cache warmth. Latent rather than active — it needs the provider
    to return an unexpected width — but that is precisely the condition the width guard exists
    for, and the two paths disagreeing is the bug regardless of how often it fires.
    """

    async def _short_vector(self, text: str, **_kw: object) -> list[float]:
        return [0.5, 0.5, 0.5]

    monkeypatch.setattr(_ClientSpy, "aembed_query", _short_vector)
    redis = _client(shared_store)

    on_miss = await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=redis)

    key = next(iter(await redis.keys("embedding:v1:*")))
    stored = from_json_float_list(str(await redis.get(key)))
    assert len(stored) == _DIMENSION, "the raw provider width was written to the cache"

    on_hit = await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY, redis=redis)
    assert on_hit == on_miss, "a hit and a miss returned different vectors for one text"


async def test_batch_cache_granularity_is_per_text_not_per_batch(spy, shared_store) -> None:
    """A batch keyed as a whole misses whenever any one member changes.

    For document ingestion that is every re-run of a document with one edited clause, which is
    the common case rather than an edge one.
    """
    redis = _client(shared_store)

    await embed_texts(["alpha", "beta"], task_type=EmbeddingTaskType.DOCUMENT, redis=redis)
    await embed_texts(["alpha", "gamma"], task_type=EmbeddingTaskType.DOCUMENT, redis=redis)

    client = get_embedding_client()
    assert [call[0] for call in client.document_calls] == [["alpha", "beta"], ["gamma"]]


async def test_batch_results_stay_aligned_with_their_inputs(spy, shared_store) -> None:
    """Order is a contract: the caller zips these against the chunks they came from."""
    redis = _client(shared_store)
    texts = ["alpha", "beta", "gamma"]

    first = await embed_texts(texts, task_type=EmbeddingTaskType.DOCUMENT, redis=redis)
    # Second pass mixes one cached and one fresh text, which is where an index-vs-filter
    # mistake in the assembly would show up as a silently shuffled result.
    second = await embed_texts(
        ["beta", "delta", "alpha"], task_type=EmbeddingTaskType.DOCUMENT, redis=redis
    )

    assert len(first) == 3
    assert second[0] == first[1]
    assert second[2] == first[0]


async def test_a_provider_returning_the_wrong_count_raises_rather_than_misaligning(
    spy, monkeypatch
) -> None:
    """Without the guard, every vector after the first gap is stored against the wrong clause."""

    async def _too_few(self, texts: list[str], **_kw: object) -> list[list[float]]:
        return [_vector()]

    monkeypatch.setattr(_ClientSpy, "aembed_documents", _too_few)

    with pytest.raises(InfrastructureException) as caught:
        await embed_texts(["alpha", "beta", "gamma"], task_type=EmbeddingTaskType.DOCUMENT)

    assert caught.value.detail["data"] == {"requested": 3, "returned": 1}


async def test_an_empty_batch_never_reaches_the_provider(spy) -> None:
    assert await embed_texts([], task_type=EmbeddingTaskType.DOCUMENT) == []
    assert spy.count() == 0


# ---------------------------------------------------------------------------
# The cache key itself — pure, so tested directly
# ---------------------------------------------------------------------------


def test_the_key_separates_the_task_types() -> None:
    query = _cache_key(_TEXT, model="m", task_type=EmbeddingTaskType.QUERY, dimension=768)
    document = _cache_key(_TEXT, model="m", task_type=EmbeddingTaskType.DOCUMENT, dimension=768)

    assert query != document


def test_the_key_separates_the_configured_widths() -> None:
    """Beyond what Decision 4 names, and deliberately so.

    The width is not derivable from the model id — one model serves several — so without it a
    deployment that changes ``EMBEDDING_DIMENSION`` reads back previous-width vectors from a
    warm cache. That is the one failure this cache could cause that re-embedding would not
    fix, because a wrong-width vector is structurally valid.
    """
    narrow = _cache_key(_TEXT, model="m", task_type=EmbeddingTaskType.QUERY, dimension=768)
    wide = _cache_key(_TEXT, model="m", task_type=EmbeddingTaskType.QUERY, dimension=1536)

    assert narrow != wide


def test_the_key_separates_the_models() -> None:
    first = _cache_key(
        _TEXT, model="gemini-embedding-001", task_type=EmbeddingTaskType.QUERY, dimension=768
    )
    second = _cache_key(
        _TEXT, model="gemini-embedding-2", task_type=EmbeddingTaskType.QUERY, dimension=768
    )

    assert first != second


def test_the_separators_stop_two_different_requests_from_colliding() -> None:
    """Bare concatenation lets the field boundaries move without changing the digest.

    Both triples below concatenate to the identical string ``mRETRIEVAL_QUERY768:x``. They are
    different requests — one is a 7-dimensional model, the other 768-dimensional — and a digest
    that cannot tell them apart would serve one deployment's vectors to the other.
    """
    shifted = _cache_key("68:x", model="m", task_type=EmbeddingTaskType.QUERY, dimension=7)
    unshifted = _cache_key(":x", model="m", task_type=EmbeddingTaskType.QUERY, dimension=768)

    assert shifted != unshifted


def test_the_namespace_is_distinct_from_the_prefixes_it_replaces() -> None:
    """Entries written under the text-only keying must be orphaned, not reinterpreted.

    They expire within the TTL on their own; nothing needs to delete them. Reading them back
    under the new contract is the failure to avoid, because their key encodes no task type.
    """
    key = _cache_key(_TEXT, model="m", task_type=EmbeddingTaskType.QUERY, dimension=768)

    assert key.startswith("embedding:v1:")
    assert not key.startswith(("kb:embedding:", "documents:embedding:"))
