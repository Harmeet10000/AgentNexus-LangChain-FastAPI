"""Machine-readable retrieval evaluation reports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import aiofiles
from anyio import Path as AsyncPath
from pydantic import BaseModel, ConfigDict, Field

from .runner import (  # noqa: TC001 — resolved at runtime by Pydantic
    RetrievalEvaluation,
    RetrievalMetrics,
    RetrievalQueryResult,
)

if TYPE_CHECKING:
    from pathlib import Path


class JudgedMetrics(BaseModel):
    """Reserved report section for a later provider-backed judged layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judge: str
    metrics: dict[str, float] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """A durable description of one retrieval evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit_identifier: str
    timestamp: datetime
    golden_set_version: str
    rows: list[RetrievalQueryResult]
    aggregates: RetrievalMetrics
    judged: JudgedMetrics | None = None

    @classmethod
    def from_evaluation(
        cls,
        evaluation: RetrievalEvaluation,
        *,
        commit_identifier: str,
        golden_set_version: str,
        timestamp: datetime | None = None,
    ) -> EvaluationReport:
        return cls(
            commit_identifier=commit_identifier,
            timestamp=timestamp or datetime.now(tz=UTC),
            golden_set_version=golden_set_version,
            rows=evaluation.rows,
            aggregates=evaluation.aggregates,
        )


async def write_report(report: EvaluationReport, path: Path) -> None:
    """Write a stable JSON report, omitting the absent judged section."""
    await AsyncPath(path.parent).mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, mode="w", encoding="utf-8") as stream:
        await stream.write(report.model_dump_json(indent=2, exclude_none=True))
        await stream.write("\n")
