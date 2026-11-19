"""STRIDE D-3: the embedding path classifies a provider 429 instead of riding it raw.

The measured incident (2026-11-14): the free-tier 1000/day cap failed 12/15
ingestions with no typed signal anywhere — a 429 propagated as a generic
exception, burned the Celery retry budget, and dead-lettered as an untyped
``INGESTION_RETRY_EXHAUSTED``. The contract pinned here:

- a transient per-minute 429 clears inside the seconds-scale probe window;
- a 429 that names the daily cap raises typed immediately (no pointless
  probing against a wall that will not move);
- a 429 that survives the probes is exhaustion, raised typed with the
  original provider error chained;
- every non-429 error propagates exactly as before — the guard adds
  classification, never changes other failure shapes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.ai.embeddings as embeddings_module
from app.ai.embeddings import (
    EmbeddingQuotaExhaustedError,
    EmbeddingTaskType,
    _provider_status_of,
    embed_text,
    embed_texts,
    get_embedding_client,
)
from app.config import get_settings

_TEXT = "The Supplier shall indemnify the Customer against all third-party claims."
_DIMENSION = get_settings().EMBEDDING_DIMENSION


class _Provider429Error(Exception):
    """A 429-shaped provider error without importing the google SDK.

    ``code`` as a plain int attribute mirrors how cloud SDKs map transport
    codes onto HTTP statuses (the recognition shape ``_provider_status_of``
    exists for).
    """

    code = 429


class _DailyQuota429Error(_Provider429Error):
    """The provider's message names the daily cap (quota id carries PerDay)."""

    def __init__(self) -> None:
        super().__init__(
            "429 Resource has been exhausted (e.g. check quota). "
            "Quota id: GenerateRequestsPerDayPerProjectPerModel."
        )


class _RateLimit429Error(_Provider429Error):
    """A per-minute rate limit: no daily marker in the message."""

    def __init__(self) -> None:
        super().__init__("429 Resource has been exhausted (e.g. check quota).")


class _QuotaSpyClient:
    """Provider stand-in whose document/query calls play a scripted failure list."""

    def __init__(self, script: list[Exception | list[list[float]]]) -> None:
        self.script = list(script)
        self.calls = 0

    def _next(self) -> object:
        self.calls += 1
        return self.script.pop(0)

    async def aembed_query(
        self, _text: str, *, task_type: object = None, **_kw: object
    ) -> list[float]:
        result = self._next()
        if isinstance(result, Exception):
            raise result
        return result

    async def aembed_documents(
        self,
        _texts: list[str],
        *,
        task_type: object = None,
        batch_size: object = None,
        **_kw: object,
    ) -> list[list[float]]:
        result = self._next()
        if isinstance(result, Exception):
            raise result
        return result


def _vector(seed: float = 0.1) -> list[float]:
    return [seed] * _DIMENSION


@pytest.fixture(autouse=True)
def _cold_client_cache():
    get_embedding_client.cache_clear()
    yield
    get_embedding_client.cache_clear()


@pytest.fixture
def fast_probes(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Zero the probe waits and record the sleeps the guard would have taken.

    The recorded list proves *how many* sleeps a scenario spends without the
    test taking real time: the daily-marker path must spend none, the probe
    path exactly one per survived attempt.
    """
    monkeypatch.setattr(embeddings_module, "_QUOTA_PROBE_DELAYS_SECONDS", (2.0, 4.0))
    sleeps: list[float] = []

    async def _record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(embeddings_module.asyncio, "sleep", _record_sleep)
    return sleeps


def _install(monkeypatch: pytest.MonkeyPatch, client: _QuotaSpyClient) -> None:
    monkeypatch.setattr(embeddings_module, "GoogleGenerativeAIEmbeddings", lambda **_kw: client)


async def test_transient_rate_limit_clears_within_the_probe_window(
    monkeypatch: pytest.MonkeyPatch, fast_probes: list[float]
) -> None:
    client = _QuotaSpyClient([_RateLimit429Error(), [_vector(0.1), _vector(0.2)]])
    _install(monkeypatch, client)

    vectors = await embed_texts(["a", "b"], task_type=EmbeddingTaskType.DOCUMENT)

    assert vectors == [_vector(0.1), _vector(0.2)]
    assert client.calls == 2
    # One probe sleep (the first wait), taken against a limit that cleared.
    assert len(fast_probes) == 1


async def test_daily_marker_raises_typed_without_probing(
    monkeypatch: pytest.MonkeyPatch, fast_probes: list[float]
) -> None:
    client = _QuotaSpyClient([_DailyQuota429Error()])
    _install(monkeypatch, client)

    with pytest.raises(EmbeddingQuotaExhaustedError) as raised:
        await embed_texts(["a"], task_type=EmbeddingTaskType.DOCUMENT)

    assert client.calls == 1  # no probe against a wall that will not move
    assert fast_probes == []
    assert raised.value.provider_status == 429
    assert "PerDay" in raised.value.detail
    assert isinstance(raised.value.__cause__, _DailyQuota429Error)


async def test_persistent_rate_limit_survives_probes_into_typed_raise(
    monkeypatch: pytest.MonkeyPatch, fast_probes: list[float]
) -> None:
    # The classification asymmetry made executable: a 429 with no daily marker
    # gets its probes, and when the limit does NOT clear, the outcome is the
    # same typed raise — never an hours-scale sleep, never a raw ride.
    client = _QuotaSpyClient([_RateLimit429Error(), _RateLimit429Error(), _RateLimit429Error()])
    _install(monkeypatch, client)

    with pytest.raises(EmbeddingQuotaExhaustedError) as raised:
        await embed_texts(["a"], task_type=EmbeddingTaskType.DOCUMENT)

    assert client.calls == 3  # initial + both probes
    assert len(fast_probes) == 2
    assert raised.value.provider_status == 429
    assert raised.value.retryable is False
    assert raised.value.error_code == "embedding_quota_exhausted"


async def test_non_429_provider_error_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch, fast_probes: list[float]
) -> None:
    class _Provider500Error(Exception):
        code = 500

    boom = _Provider500Error("internal provider error")
    client = _QuotaSpyClient([boom])
    _install(monkeypatch, client)

    with pytest.raises(_Provider500Error) as raised:
        await embed_texts(["a"], task_type=EmbeddingTaskType.DOCUMENT)

    assert raised.value is boom  # the same object, untouched
    assert fast_probes == []


async def test_query_side_raise_is_typed_too(
    monkeypatch: pytest.MonkeyPatch, fast_probes: list[float]
) -> None:
    client = _QuotaSpyClient([_DailyQuota429Error()])
    _install(monkeypatch, client)

    with pytest.raises(EmbeddingQuotaExhaustedError):
        await embed_text(_TEXT, task_type=EmbeddingTaskType.QUERY)


def test_provider_status_recovers_status_from_three_shapes() -> None:
    class _StatusAttrError(Exception):
        status_code = 429

    class _NumericCodeError(Exception):
        code = 429

    class _StringCodeError(Exception):
        code = "RESOURCE_EXHAUSTED"  # a string identifier must never match

    class _OnResponseError(Exception):
        def __init__(self) -> None:
            self.response = SimpleNamespace(status_code=429)

    assert _provider_status_of(_StatusAttrError()) == 429
    assert _provider_status_of(_NumericCodeError()) == 429
    assert _provider_status_of(_OnResponseError()) == 429
    assert _provider_status_of(_StringCodeError()) is None
    assert _provider_status_of(Exception("plain")) is None
