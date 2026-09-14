"""Validated schema and loader for versioned retrieval golden sets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal  # noqa: TC003

import aiofiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from returns.result import Failure, Result, Success

from app.utils.exceptions import InfrastructureException, NotFoundException

from .errors import MalformedGoldenRowError

if TYPE_CHECKING:
    from pathlib import Path


class GoldenQuery(BaseModel):
    """One human-curated retrieval expectation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    expected_chunk_ids: list[str] = Field(min_length=1)
    expected_document_ids: list[str] = Field(min_length=1)
    jurisdiction: str | None = None
    document_kind: Literal["contracts", "statutes", "judgments", "filings"]
    difficulty: Literal["easy", "medium", "hard"]
    notes: str
    awaiting_sme_expansion: bool


class GoldenSet(BaseModel):
    """A comparable, explicitly versioned collection of golden queries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    queries: list[GoldenQuery] = Field(min_length=1)


type GoldenSetLoadResult = Result[GoldenSet, MalformedGoldenRowError]


class GoldenSetNotFoundException(NotFoundException):
    """The requested golden-set artifact does not exist."""

    def __init__(self, path: Path) -> None:
        super().__init__(resource="Golden set", identifier=str(path))


class GoldenSetReadException(InfrastructureException):
    """The requested golden-set artifact exists but cannot be read."""

    def __init__(self, path: Path, cause: OSError) -> None:
        super().__init__(
            detail=f"Unable to read golden set: {path}",
            retryable=False,
            original_exc=cause,
        )


def _malformed(*, row_index: int, message: str) -> Failure[MalformedGoldenRowError]:
    return Failure(
        MalformedGoldenRowError(
            message=message,
            row_index=row_index,
            details={"row_index": row_index},
            source="golden_set_loader",
        )
    )


async def load_golden_set(path: Path) -> GoldenSetLoadResult:
    """Load metadata plus query rows from a JSON Lines golden-set artifact."""
    try:
        async with aiofiles.open(path, encoding="utf-8") as stream:
            lines = await stream.readlines()
    except FileNotFoundError as exc:
        raise GoldenSetNotFoundException(path) from exc
    except OSError as exc:
        raise GoldenSetReadException(path, exc) from exc

    if not lines:
        return _malformed(row_index=1, message="Golden set is empty")
    try:
        metadata = json.loads(lines[0])
        version = metadata["version"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return _malformed(row_index=1, message=f"Malformed golden-set metadata: {exc}")
    if not isinstance(version, str) or not version:
        return _malformed(row_index=1, message="Golden-set version must be a non-empty string")

    raw_queries: list[object] = []
    for row_index, line in enumerate(lines[1:], start=2):
        try:
            raw_queries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            return _malformed(row_index=row_index, message=f"Malformed JSON: {exc.msg}")
    try:
        queries = TypeAdapter(list[GoldenQuery]).validate_python(raw_queries)
    except ValidationError as exc:
        offset = exc.errors()[0]["loc"][0]
        row_index = int(offset) + 2 if isinstance(offset, int) else 2
        return _malformed(row_index=row_index, message=str(exc))
    if not queries:
        return _malformed(row_index=2, message="Golden set contains no query rows")
    return Success(GoldenSet(version=version, queries=queries))
