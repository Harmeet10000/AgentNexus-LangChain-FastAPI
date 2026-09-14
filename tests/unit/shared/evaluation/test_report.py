from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from app.shared.evaluation.report import EvaluationReport, write_report
from app.shared.evaluation.runner import RetrievalEvaluation, RetrievalMetrics


def _report() -> EvaluationReport:
    evaluation = RetrievalEvaluation(
        rows=[],
        aggregates=RetrievalMetrics(
            recall_at_k=0.0,
            reciprocal_rank=0.0,
            ndcg_at_k=0.0,
            precision_at_k=0.0,
        ),
    )
    return EvaluationReport.from_evaluation(
        evaluation,
        commit_identifier="abc123",
        golden_set_version="legal_retrieval_v1",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_report_json_round_trip_names_golden_set_version(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    await write_report(_report(), path)

    restored = TypeAdapter(EvaluationReport).validate_json(path.read_text(encoding="utf-8"))

    assert restored.golden_set_version == "legal_retrieval_v1"
    assert restored.commit_identifier == "abc123"


@pytest.mark.asyncio
async def test_judged_section_is_absent_by_default(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    await write_report(_report(), path)

    assert '"judged"' not in path.read_text(encoding="utf-8")
