import pytest

from app.shared.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_rank_one_scores_one() -> None:
    ranked = ["wanted", "other"]
    expected = {"wanted"}

    assert recall_at_k(ranked, expected, 1) == pytest.approx(1.0)
    assert reciprocal_rank(ranked, expected, 1) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, expected, 1) == pytest.approx(1.0)
    assert precision_at_k(ranked, expected, 1) == pytest.approx(1.0)


def test_miss_within_cutoff_scores_zero() -> None:
    ranked = ["other", "wanted"]
    expected = {"wanted"}

    assert recall_at_k(ranked, expected, 1) == pytest.approx(0.0)
    assert reciprocal_rank(ranked, expected, 1) == pytest.approx(0.0)
    assert ndcg_at_k(ranked, expected, 1) == pytest.approx(0.0)
    assert precision_at_k(ranked, expected, 1) == pytest.approx(0.0)


def test_duplicate_relevant_identifiers_are_counted_once() -> None:
    ranked = ["wanted", "wanted", "other"]
    expected = {"wanted"}

    assert recall_at_k(ranked, expected, 3) == pytest.approx(1.0)
    assert 0.0 <= ndcg_at_k(ranked, expected, 3) <= 1.0
    assert precision_at_k(ranked, expected, 3) == pytest.approx(1 / 3)


def test_empty_expected_set_has_defined_zero_scores() -> None:
    assert recall_at_k(["anything"], set(), 1) == pytest.approx(0.0)
    assert reciprocal_rank(["anything"], set(), 1) == pytest.approx(0.0)
    assert ndcg_at_k(["anything"], set(), 1) == pytest.approx(0.0)
    assert precision_at_k(["anything"], set(), 1) == pytest.approx(0.0)
