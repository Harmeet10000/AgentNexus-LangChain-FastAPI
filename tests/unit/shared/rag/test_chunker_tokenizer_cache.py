"""Unit tests for task B4 — the token counter is loaded once per process.

Before B4, ``rag/document_processing/chunker.py``'s accessor ran the transformers
auto-class loader on every call and logged "Initializing tokenizer" every time.
The cost is not theoretical: there is no local model cache on a fresh machine, so
the first load is a network download inside a synchronous function, and every
caller that did not hoist the result by hand paid it again.

These tests assert on a **construction spy**, not on the identity of the returned
object. Identity alone cannot distinguish a cache from a module-level singleton
that was built at import time, and it says nothing about how many times the
loader ran. The spy counts loads.

``lru_cache`` state is process-global, so a cache warmed by an earlier test makes
a later "constructed once" assertion vacuous. Every test here therefore starts
and ends with a cold cache (``_cold_tokenizer_cache``), and
``test_the_first_call_in_a_cold_process_does_construct`` exists specifically so
that a broken fixture shows up as a failure rather than as a silently vacuous
pass.

Nothing here touches the network. ``_FakeTokenizer`` is a real
``PreTrainedTokenizerBase`` subclass with no vocabulary on disk, which is what
lets it pass through Docling's own tokenizer validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizerBase

from app.shared.rag.document_processing import chunker as chunker_module
from app.shared.rag.document_processing.chunker import (
    DEFAULT_TOKENIZER_MODEL_ID,
    chunk_document,
    get_tokenizer,
    initialize_chunking,
)
from app.shared.rag.document_processing.models import IngestionConfig

_OTHER_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
_CONTENT = "Alpha beta gamma. Delta epsilon zeta."


class _FakeTokenizer(PreTrainedTokenizerBase):
    """A counter that needs neither a vocabulary file nor a network call.

    ``__len__`` is not decoration. Docling's legacy tokenizer-coercion path
    evaluates the tokenizer for *truthiness*, and truthiness on a
    ``PreTrainedTokenizerBase`` goes through ``__len__``, which the base class
    leaves unimplemented. Without this override the coercion raises, and its
    fallback would reach for the network to fetch a default counter.
    """

    def __len__(self) -> int:
        return 30522

    def encode(self, text, **_kwargs):
        # Two more than the word count, standing in for the pair of special
        # tokens a real WordPiece counter adds around a sequence.
        return list(range(len(text.split()) + 2))

    def tokenize(self, text, **_kwargs):
        return text.split()


class _ConstructionSpy:
    """Stands in for the transformers auto-class and counts real loads.

    Substituted for the module-global the loader resolves at call time, so it
    intercepts construction without the accessor knowing.
    """

    def __init__(self) -> None:
        self.model_ids: list[str] = []

    def from_pretrained(self, model_id: str) -> _FakeTokenizer:
        self.model_ids.append(model_id)
        return _FakeTokenizer()

    @property
    def count(self) -> int:
        return len(self.model_ids)


@pytest.fixture(autouse=True)
def _cold_tokenizer_cache():
    """Bracket every test with an empty cache.

    Both sides matter. Clearing on entry stops an earlier test's warm entry from
    making a "constructed once" assertion vacuous; clearing on exit stops this
    module from leaving a fake counter cached where the rest of the session can
    reach it through the process-wide accessor.
    """
    chunker_module._load_tokenizer.cache_clear()
    yield
    chunker_module._load_tokenizer.cache_clear()


@pytest.fixture
def spy(monkeypatch) -> _ConstructionSpy:
    construction_spy = _ConstructionSpy()
    monkeypatch.setattr(chunker_module, "AutoTokenizer", construction_spy)
    return construction_spy


def _config() -> IngestionConfig:
    return IngestionConfig(use_semantic_chunking=True, max_tokens=512)


async def test_two_chunking_calls_construct_the_counter_once(spy) -> None:
    """B4's first Proof, in the shape the production callers actually have.

    ``ingest_v2.py:184`` acquires a counter per document, so two documents mean
    two acquisitions; that is the amplification the cache removes. Both calls
    must still produce chunks — a cache that returned something unusable would
    otherwise satisfy the count.
    """
    config = _config()

    first = await chunk_document(
        content=_CONTENT, title="one", source="one.md", config=config, tokenizer=get_tokenizer()
    )
    second = await chunk_document(
        content=_CONTENT, title="two", source="two.md", config=config, tokenizer=get_tokenizer()
    )

    assert spy.count == 1
    assert spy.model_ids == [DEFAULT_TOKENIZER_MODEL_ID]
    assert first, "first chunking call produced no chunks"
    assert second, "second chunking call produced no chunks"
    assert first[0].token_count > 0


async def test_two_chunking_initialisations_construct_the_counter_once(spy) -> None:
    """The same guarantee, driven through production code rather than the test.

    ``initialize_chunking`` is what acquires the counter in the real path, and it
    also builds a ``HybridChunker``. The chunker is per-call state and stays
    distinct; only the counter is shared.
    """
    first_tokenizer, first_chunker = await initialize_chunking(_config())
    second_tokenizer, second_chunker = await initialize_chunking(_config())

    assert spy.count == 1
    assert first_tokenizer is second_tokenizer
    assert first_chunker is not second_chunker


async def test_the_first_call_in_a_cold_process_does_construct(spy) -> None:
    """Guards the two tests above from passing vacuously.

    If the cold-cache fixture ever stops working, "constructed once" would still
    hold while the count was zero. This pins the miss as well as the hit.
    """
    assert spy.count == 0

    get_tokenizer()

    assert spy.count == 1


def test_the_omitted_default_and_the_explicit_default_share_one_cache_entry(spy) -> None:
    """The trap that makes the obvious one-decorator version wrong.

    A default argument is not part of an ``lru_cache`` key. Decorating the public
    accessor directly would give ``get_tokenizer()`` and
    ``get_tokenizer(DEFAULT_TOKENIZER_MODEL_ID)`` separate entries and load the
    same counter twice, which is exactly the defect the cache is meant to remove.
    """
    omitted = get_tokenizer()
    explicit = get_tokenizer(DEFAULT_TOKENIZER_MODEL_ID)

    assert spy.count == 1
    assert omitted is explicit


def test_distinct_model_ids_get_their_own_counters(spy) -> None:
    """The accessor is a keyed cache, not a process singleton."""
    default_first = get_tokenizer()
    other_first = get_tokenizer(_OTHER_MODEL_ID)

    assert spy.count == 2
    assert default_first is not other_first

    assert get_tokenizer() is default_first
    assert get_tokenizer(_OTHER_MODEL_ID) is other_first
    assert spy.count == 2


def test_the_cache_is_bounded_so_an_id_sweep_cannot_grow_it_without_limit(spy) -> None:
    """Boundedness is a decision, and this is what it costs.

    The key is caller-supplied, so an unbounded cache lets a caller sweeping
    model ids pin an unbounded number of multi-megabyte counters for the life of
    the process. Asserted behaviourally — the least-recently-used entry is
    evicted and reloads — rather than by reading the decorator's parameters, so
    it would still fail if the bound were removed some other way.
    """
    maxsize = chunker_module._load_tokenizer.cache_parameters()["maxsize"]
    assert isinstance(maxsize, int), "the tokenizer cache must be bounded, not unbounded"
    assert maxsize >= 1

    swept = [f"vendor/counter-{index}" for index in range(maxsize + 1)]
    for model_id in swept:
        get_tokenizer(model_id)
    assert spy.count == maxsize + 1

    # swept[0] is the least recently used and has been evicted, so asking for it
    # again is a fresh load rather than a hit.
    get_tokenizer(swept[0])
    assert spy.count == maxsize + 2


def test_the_load_is_logged_once_rather_than_on_every_call(spy, monkeypatch) -> None:
    """A per-call line claiming initialisation would be false after the first.

    The old accessor logged on every call. The line now lives inside the memoised
    loader, so one line in the log means one real load.
    """
    recorder = MagicMock()
    monkeypatch.setattr(chunker_module, "loguru_logger", recorder)

    get_tokenizer()
    get_tokenizer()
    get_tokenizer()

    load_lines = [
        call for call in recorder.info.call_args_list if "Loading tokenizer" in str(call.args[0])
    ]
    assert spy.count == 1
    assert len(load_lines) == 1
    assert load_lines[0].args[1] == DEFAULT_TOKENIZER_MODEL_ID


def test_clearing_the_cache_makes_the_next_call_load_again(spy) -> None:
    """Proves the hygiene mechanism the fixture depends on is real.

    Without this, a ``cache_clear`` that silently stopped working would leave
    every other test in this module vacuous and nothing would report it.
    """
    get_tokenizer()
    assert spy.count == 1

    chunker_module._load_tokenizer.cache_clear()
    get_tokenizer()

    assert spy.count == 2
