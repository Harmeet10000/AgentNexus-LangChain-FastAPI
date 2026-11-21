"""The shared converter singleton: built once, serialised at ``convert``, budgets intact.

Three claims, each pinned because none is visible to lint or types.

**Why a counting stub rather than the real factory.** The real factory instantiates docling's
PDF pipeline — seconds per construction, a model download on first run, and the very memory
footprint this change exists to remove. What is under test is the parser's caching wiring, not
docling's constructor.

**Why recorded windows rather than wall-clock totals for the serialisation claim.** A duration
assertion ("two parses took about twice one parse") measures the machine as much as the code.
What actually distinguishes serialised from overlapping execution is *interleaving*: each
``convert`` call records its [start, end] window, and two windows from concurrent parses must
not overlap. That assertion holds on a loaded runner and fails against a lock-free
implementation regardless of how fast the machine is.

The stub mirrors the fakes in ``test_parser_offload_and_tables.py``: a blocking ``time.sleep``
inside ``convert`` (the real call is CPU-bound and holds its thread), a document with pages and
an empty table collection, and a result with no confidence report — every shape the parser
touches after ``convert`` returns.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

import pytest

from app.features.documents.application.ingestion.parsing import parser as parser_module
from app.features.documents.application.ingestion.parsing.parser import (
    DocumentExtractionBudgetError,
    parse_document,
)

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit

_FILENAME = "contract.pdf"
_RAW = b"%PDF-1.7 not really a pdf, the converter is faked"
_MARKDOWN = "# Master Services Agreement\n\nThe parties agree as follows."

#: Long enough that two serialised calls cannot be mistaken for overlapping ones on a noisy
#: runner, short enough that the suite does not notice.
_CONVERT_SECONDS = 0.1


class _StubDocument:
    def __init__(self, *, pages: int = 3) -> None:
        self.tables: list[object] = []
        self.pages = list(range(pages))

    def export_to_markdown(self) -> str:
        return _MARKDOWN


class _StubResult:
    def __init__(self, document: _StubDocument) -> None:
        self.document = document


class _StubConverter:
    """Records each ``convert`` window and blocks its thread for a fixed duration."""

    def __init__(self, *, convert_seconds: float = 0.0, pages: int = 3) -> None:
        self.convert_seconds = convert_seconds
        self.pages = pages
        self.convert_calls = 0
        self._windows: list[tuple[float, float]] = []
        self._windows_lock = threading.Lock()

    def convert(self, source: Any = None, **_kw: Any) -> _StubResult:
        self.convert_calls += 1
        started = time.perf_counter()
        time.sleep(self.convert_seconds)
        ended = time.perf_counter()
        with self._windows_lock:
            self._windows.append((started, ended))
        return _StubResult(_StubDocument(pages=self.pages))

    @property
    def windows(self) -> list[tuple[float, float]]:
        with self._windows_lock:
            return list(self._windows)


@pytest.fixture
def _fresh_shared_converter() -> None:
    """The cache is process-wide; a stubbed factory must not leak in or out.

    Reset before and after: before, so this module's stubbed factory is the one that
    builds the singleton no matter what an earlier test file cached; after, so no
    stub-built converter survives into a later file that expects the real factory.
    """
    parser_module._reset_shared_converter_for_tests()
    yield
    parser_module._reset_shared_converter_for_tests()


def _install_counting_factory(
    monkeypatch: pytest.MonkeyPatch,
    converter: _StubConverter,
) -> list[_StubConverter]:
    """Bind a counting stub factory at the name the parser imported."""
    constructed: list[_StubConverter] = []

    def counting_factory(**_kw: object) -> _StubConverter:
        constructed.append(converter)
        return converter

    monkeypatch.setattr(parser_module, "create_document_converter", counting_factory)
    return constructed


@pytest.mark.usefixtures("_fresh_shared_converter")
async def test_two_parses_construct_the_converter_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memory claim: one model set resident, no matter how many parses run."""
    converter = _StubConverter()
    constructed = _install_counting_factory(monkeypatch, converter)

    first = await parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf")
    second = await parse_document(
        raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"
    )

    assert len(constructed) == 1, "the factory ran per parse; the converter is not cached"
    assert converter.convert_calls == 2
    assert first.markdown == second.markdown == _MARKDOWN


@pytest.mark.usefixtures("_fresh_shared_converter")
async def test_concurrent_parses_never_overlap_convert_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thread-safety claim: ``convert`` calls are serialised across worker threads.

    Both parses run concurrently (``asyncio.gather`` + ``asyncer.asyncify``), so without the
    lock their convert windows would interleave. The windows are recorded inside the stub's
    ``convert``, under a lock of its own, so the assertion is on recorded fact, not timing.
    """
    converter = _StubConverter(convert_seconds=_CONVERT_SECONDS)
    _install_counting_factory(monkeypatch, converter)

    await asyncio.gather(
        parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"),
        parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"),
    )

    windows = sorted(converter.windows)
    assert len(windows) == 2
    assert converter.convert_calls == 2
    first_end, second_start = windows[0][1], windows[1][0]
    assert first_end <= second_start, (
        f"convert windows overlapped (first ended {first_end:.3f}, second started "
        f"{second_start:.3f}); the serialising lock is not holding"
    )


@pytest.mark.usefixtures("_fresh_shared_converter")
async def test_page_budget_is_enforced_through_the_shared_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caching must not soften the budgets: the shared converter's result is still checked."""
    converter = _StubConverter(pages=300)
    _install_counting_factory(monkeypatch, converter)

    with pytest.raises(DocumentExtractionBudgetError, match="pages; limit is 2"):
        await parse_document(
            raw_bytes=_RAW,
            filename=_FILENAME,
            content_type="application/pdf",
            max_pages=2,
        )
    assert converter.convert_calls == 1


@pytest.mark.usefixtures("_fresh_shared_converter")
async def test_output_budget_is_enforced_through_the_shared_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same claim for the byte budget on the extracted markdown."""
    converter = _StubConverter()
    _install_counting_factory(monkeypatch, converter)

    with pytest.raises(DocumentExtractionBudgetError, match="extracted output"):
        await parse_document(
            raw_bytes=_RAW,
            filename=_FILENAME,
            content_type="application/pdf",
            max_output_bytes=16,
        )


@pytest.mark.usefixtures("_fresh_shared_converter")
async def test_the_reset_helper_restores_a_cold_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second stub must be able to take over, proving the cache has no hidden state.

    This is the property the other test files rely on when they monkeypatch the factory:
    without a working reset, whichever converter got built first would be served forever.
    """
    first = _StubConverter()
    second = _StubConverter()
    constructed = _install_counting_factory(monkeypatch, first)
    await parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf")

    parser_module._reset_shared_converter_for_tests()

    def second_factory(**_kw: object) -> _StubConverter:
        constructed.append(second)
        return second

    monkeypatch.setattr(parser_module, "create_document_converter", second_factory)

    await parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf")

    assert first.convert_calls == 1
    assert second.convert_calls == 1
