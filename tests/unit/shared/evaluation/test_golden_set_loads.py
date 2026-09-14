from pathlib import Path

import pytest
from returns.result import Success

from app.shared.evaluation.schema import load_golden_set


@pytest.mark.asyncio
async def test_seed_golden_set_is_valid_and_covers_all_families() -> None:
    path = Path("evals/golden/legal_retrieval_v1.jsonl")

    result = await load_golden_set(path)

    assert isinstance(result, Success)
    golden_set = result.unwrap()
    assert golden_set.version == "legal_retrieval_v1"
    assert {query.document_kind for query in golden_set.queries} == {
        "contracts",
        "statutes",
        "judgments",
        "filings",
    }
    assert all(query.awaiting_sme_expansion for query in golden_set.queries)
