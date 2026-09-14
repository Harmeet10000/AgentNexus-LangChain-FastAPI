import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.shared.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.property

_identifiers = st.text(min_size=1, max_size=12)


@given(
    ranked=st.lists(_identifiers, max_size=30),
    expected=st.sets(_identifiers, max_size=15),
    first_k=st.integers(min_value=0, max_value=30),
    extra=st.integers(min_value=0, max_value=30),
)
def test_recall_is_non_decreasing(
    ranked: list[str], expected: set[str], first_k: int, extra: int
) -> None:
    assert recall_at_k(ranked, expected, first_k) <= recall_at_k(
        ranked, expected, first_k + extra
    )


@given(
    ranked=st.lists(_identifiers, max_size=30),
    expected=st.sets(_identifiers, max_size=15),
    k=st.integers(min_value=-5, max_value=35),
)
def test_every_metric_is_bounded(ranked: list[str], expected: set[str], k: int) -> None:
    scores = (
        recall_at_k(ranked, expected, k),
        reciprocal_rank(ranked, expected, k),
        ndcg_at_k(ranked, expected, k),
        precision_at_k(ranked, expected, k),
    )
    assert all(0.0 <= score <= 1.0 for score in scores)
