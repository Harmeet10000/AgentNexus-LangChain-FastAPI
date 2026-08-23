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
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import pytest

from app.features.documents import parser as parser_module
from app.features.documents.parser import parse_document

if TYPE_CHECKING:
    from collections.abc import Iterator

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
def _patched_converter(
    monkeypatch: pytest.MonkeyPatch, converter: _FakeConverter
) -> Iterator[None]:
    """Substitute the converter factory the module imported.

    Patched at `parser_module.create_document_converter` — the name bound *in this module* — not
    at its definition site. `parser.py` imported the function, so rebinding the source module's
    attribute would leave this module's reference pointing at the original.
    """
    monkeypatch.setattr(parser_module, "create_document_converter", lambda **_kw: converter)
    yield


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
async def test_two_parses_overlap_rather_than_serialising() -> None:
    """The stronger form of the same claim, and the one a caller actually feels.

    Loop responsiveness is necessary but not sufficient: an implementation that offloaded to a
    single serialising worker would pass the test above and still queue uploads behind each other.
    Two concurrent parses must take about as long as one, not twice as long.
    """
    started = time.perf_counter()
    await asyncio.gather(
        parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"),
        parse_document(raw_bytes=_RAW, filename=_FILENAME, content_type="application/pdf"),
    )
    elapsed = time.perf_counter() - started

    # Generous: the bar is "not serialised", so anything below two full parses proves overlap.
    assert elapsed < _BLOCKING_PARSE_SECONDS * 1.8, (
        f"two parses took {elapsed:.3f}s against a {_BLOCKING_PARSE_SECONDS:.3f}s parse; "
        "they serialised instead of overlapping"
    )


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
