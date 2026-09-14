from pathlib import Path

import pytest
from returns.result import Failure, Success

from app.shared.evaluation.schema import (
    GoldenSetNotFoundException,
    load_golden_set,
)


@pytest.mark.asyncio
async def test_malformed_row_returns_failure_with_row_index(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"version":"v1"}\n{"query":"missing fields"}\n', encoding="utf-8")

    result = await load_golden_set(path)

    assert isinstance(result, Failure)
    assert result.failure().row_index == 2
    assert result.failure().details == {"row_index": 2}


@pytest.mark.asyncio
async def test_absent_file_raises_typed_project_exception(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetNotFoundException):
        await load_golden_set(tmp_path / "absent.jsonl")


@pytest.mark.asyncio
async def test_valid_schema_loads(tmp_path: Path) -> None:
    path = tmp_path / "valid.jsonl"
    path.write_text(
        '{"version":"v1"}\n'
        '{"query":"q","expected_chunk_ids":["c"],"expected_document_ids":["d"],'
        '"jurisdiction":null,"document_kind":"contracts","difficulty":"easy",'
        '"notes":"seed","awaiting_sme_expansion":true}\n',
        encoding="utf-8",
    )

    result = await load_golden_set(path)

    assert isinstance(result, Success)
    assert result.unwrap().version == "v1"


@pytest.mark.asyncio
async def test_invalid_utf8_raises_typed_read_exception(tmp_path: Path) -> None:
    from app.shared.evaluation.schema import GoldenSetReadException

    path = tmp_path / "binary.jsonl"
    path.write_bytes(b'{"version":"v1"}\n\xff\xfe not utf-8 \x80\n')

    with pytest.raises(GoldenSetReadException):
        await load_golden_set(path)
