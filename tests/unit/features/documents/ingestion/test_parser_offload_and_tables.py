"""B3: the parse must not block the event loop, and must not discard the tables it extracts.

Two claims, and neither is visible to lint or types — which is why this file is mandatory rather
than nice to have.

**Why the responsiveness test schedules a competing coroutine rather than timing anything.** A
wall-clock assertion ("the parse took less than N ms") measures the machine, not the code: it
passes on a fast runner with a blocking implementation and fails on a loaded one with a correct
implementation. What actually distinguishes the two is *interleaving* — whether anything else
gets to run while the parse is in flight. So the fake converter blocks on `time.sleep`, a
genuinely loop-blocking call that `asyncio.sleep` is not, and a second coroutine records that it
ran. Against the old code the parse held the loop for its whole duration and the flag was still
unset when it returned; against the offload the flag is set. The assertion is on order, not
duration, so it does not care how fast the machine is.

**Why `time.sleep` and not a `threading.Event`.** An `Event.wait()` would also block, but it
invites a test that deadlocks if the offload is ever removed rather than one that fails. A short
sleep fails cleanly in both directions.

The fakes stand in for docling rather than driving it. A real parse would need a fixture PDF, an
OCR pass and a table-structure model — seconds per test, network on first run, and a table
collection whose contents depend on a model version. What is under test here is this module's own
wiring: that the synchronous call is offloaded, and that the tables the converter reports reach
the returned model.

**Changed by the singleton converter.** There was a third test here asserting two concurrent
parses *overlap* rather than serialise; the cached converter removed that property on purpose
(docling does not document ``convert`` as thread-safe, so the calls serialise behind a lock) and
it was replaced by the both-complete assertion below. The serialisation itself is pinned by the
recorded-window test in ``test_parser_converter_singleton.py``. The responsiveness tests above
are untouched: the offload is exactly what makes the lock preferable to per-call conversion,
since the loop never blocks either way.
"""

from __future__ import annotations

import asyncio
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

#: Long enough that a blocking implementation cannot be mistaken for a fast one, short enough
#: that the suite does not notice. The competing coroutine needs only one loop iteration.
_BLOCKING_PARSE_SECONDS = 0.15


class _FakeTable:
    """A docling ``TableItem`` stand-in.

    It defines ``export_to_markdown`` and **not** ``to_markdown``, mirroring the real class. That
    asymmetry is the point: the implementation this replaced called ``to_markdown`` behind a
    ``hasattr`` guard, so a fake offering both names would have passed against the broken code and
    proved nothing.
    """

    def __init__(self, body: str) -> None:
        self.body = body
        self.export_calls: list[object] = []

    def export_to_markdown(self, doc: object = None) -> str:
        self.export_calls.append(doc)
        return self.body


class _FakeDocument:
    def __init__(self, *, tables: list[_FakeTable], pages: int = 3) -> None:
        self.tables = tables
        self.pages = list(range(pages))

    def export_to_markdown(self) -> str:
        return _MARKDOWN


class _FakeResult:
    def __init__(self, document: _FakeDocument) -> None:
        self.document = document


class _FakeConverter:
    """Blocks the calling thread, the way the real converter does."""

    def __init__(self, document: _FakeDocument, *, block_seconds: float) -> None:
        self.document = document
        self.block_seconds = block_seconds
        self.convert_calls = 0

    def convert(self, source: Any = None, **_kw: Any) -> _FakeResult:
        self.convert_calls += 1
        time.sleep(self.block_seconds)
        return _FakeResult(self.document)


@pytest.fixture
def tables() -> list[_FakeTable]:
    return [_FakeTable("| term | value |\n|---|---|\n| fee | 100 |")]


@pytest.fixture
def converter(tables: list[_FakeTable]) -> _FakeConverter:
    return _FakeConverter(_FakeDocument(tables=tables), block_seconds=_BLOCKING_PARSE_SECONDS)


@pytest.fixture
def _patched_converter(monkeypatch: pytest.MonkeyPatch, converter: _FakeConverter) -> None:
    """Substitute the converter factory the module imported.

    Patched at `parser_module.create_document_converter` — the name bound *in this module* — not
    at its definition site. `parser.py` imported the function, so rebinding the source module's
    attribute would leave this module's reference pointing at the original.

    The parser now caches the factory's product behind a process-wide singleton, so the cache is
    reset first: without that, whichever converter an earlier test file built would be served to
    every later test and this stub would never be consulted.
    """
    parser_module._reset_shared_converter_for_tests()
    monkeypatch.setattr(parser_module, "create_document_converter", lambda **_kw: converter)


@pytest.mark.usefixtures("_patched_converter")
async def test_the_event_loop_keeps_running_during_a_parse() -> None:
    """The mandatory one. A competing coroutine must run before the parse returns."""
    ran_during_parse = asyncio.Event()

    async def competitor() -> None:
        # No sleep before setting: this must get its turn purely because the parse released the
        # loop, not because it waited long enough for the parse to finish.
        ran_during_parse.set()

    task = asyncio.create_task(competitor())
    await parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf")

    assert ran_during_parse.is_set(), (
        "the competing coroutine had not run when the parse returned; "
        "converter.convert is blocking the event loop"
    )
    await task


@pytest.mark.usefixtures("_patched_converter")
async def test_two_concurrent_parses_both_complete_with_their_own_results() -> None:
    """The concurrency claim, updated for the landed singleton-converter decision.

    This used to assert the opposite — that two parses *overlap* rather than serialise. The
    singleton converter changed that deliberately: docling does not document
    ``DocumentConverter.convert`` as thread-safe, so ``convert`` calls now serialise behind a
    ``threading.Lock`` (one resident model set instead of ~3.7 GB per parse; the trade-off is
    recorded in ``parse_document``'s docstring and pinned by the window test in
    ``test_parser_converter_singleton.py``). What must still hold — and what a lock that is too
    wide, a deadlocked acquisition, or a converter shared unsafely would break — is that two
    parses gathered concurrently both complete and each return a correct, independent result.
    """
    first, second = await asyncio.gather(
        parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"),
        parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"),
    )

    for parsed in (first, second):
        assert parsed.markdown == _MARKDOWN
        assert parsed.title == "Master Services Agreement"
        assert parsed.page_count == 3


@pytest.mark.usefixtures("_patched_converter")
async def test_a_document_with_a_table_yields_a_non_empty_table_collection(
    tables: list[_FakeTable],
) -> None:
    """The other mandatory claim: extracted tables reach the returned model."""
    parsed = await parse_document(
        raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"
    )

    assert parsed.tables == [tables[0].body]


@pytest.mark.usefixtures("_patched_converter")
async def test_the_parent_document_is_passed_to_the_table_serialiser(
    tables: list[_FakeTable], converter: _FakeConverter
) -> None:
    """``export_to_markdown()`` without ``doc=`` is deprecated and takes a lossy fallback.

    Omitting it makes docling walk the cell grid directly instead of going through
    ``MarkdownDocSerializer``, so cell content that refers to something elsewhere in the document
    cannot be resolved. The call still succeeds, which is why this is asserted rather than trusted.
    """
    await parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf")

    assert tables[0].export_calls == [converter.document]


@pytest.mark.usefixtures("_patched_converter")
async def test_a_document_without_tables_yields_an_empty_collection(
    converter: _FakeConverter,
) -> None:
    """Empty because there were none — distinguishable from the old always-empty behaviour only
    in combination with the test above, which is why both exist."""
    converter.document.tables = []

    parsed = await parse_document(
        raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"
    )

    assert parsed.tables == []


@pytest.mark.usefixtures("_patched_converter")
async def test_empty_input_never_reaches_the_converter(converter: _FakeConverter) -> None:
    """The short-circuit predates this task; pinned so the offload did not move it below the
    conversion call, which would turn an empty upload into a pointless thread hop."""
    parsed = await parse_document(raw_bytes=b"", filename=_FILENAME, content_type="application/pdf")

    assert parsed.markdown == ""
    assert parsed.page_count == 0
    assert converter.convert_calls == 0


@pytest.mark.usefixtures("_patched_converter")
async def test_the_markdown_and_title_survive_the_offload() -> None:
    """A thread hop is an easy place to lose a return value; this pins that it does not."""
    parsed = await parse_document(
        raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"
    )

    assert parsed.markdown == _MARKDOWN
    assert parsed.title == "Master Services Agreement"
    assert parsed.page_count == 3


@pytest.mark.usefixtures("_patched_converter")
async def test_page_budget_rejects_documents_before_ingestion() -> None:
    with pytest.raises(DocumentExtractionBudgetError, match="pages; limit is 2"):
        await parse_document(
            raw_bytes=_RAW,
            filename=_FILENAME,
            content_type="application/pdf",
            max_pages=2,
        )


@pytest.mark.usefixtures("_patched_converter")
async def test_output_budget_rejects_expanded_markdown() -> None:
    with pytest.raises(DocumentExtractionBudgetError, match="extracted output"):
        await parse_document(
            raw_bytes=_RAW,
            filename=_FILENAME,
            content_type="application/pdf",
            max_output_bytes=16,
        )


@pytest.mark.usefixtures("_patched_converter")
async def test_wall_clock_budget_is_enforced() -> None:
    with pytest.raises(DocumentExtractionBudgetError, match="wall-clock budget"):
        await parse_document(
            raw_bytes=_RAW,
            filename=_FILENAME,
            content_type="application/pdf",
            extraction_timeout_seconds=0.01,
        )
