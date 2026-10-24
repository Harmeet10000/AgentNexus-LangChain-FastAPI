"""Deterministic parser and decision tests for the Greptile required check."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "greptile_confidence_gate.py"
SPEC = importlib.util.spec_from_file_location("greptile_confidence_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

HEAD = "a" * 40
TRUSTED = {
    "author_login": "greptile-apps[bot]",
    "author_id": 165735046,
    "author_type": "Bot",
}


def comment(body: str, **author_overrides: object) -> object:
    author = TRUSTED | author_overrides
    return GATE.Comment(body, author["author_login"], author["author_id"], author["author_type"])


def valid_body(sha: str = HEAD) -> str:
    return f"Confidence Score: 5/5\nLast reviewed commit: `{sha}`"


def test_accepts_html_summary_and_exact_current_head() -> None:
    result = GATE.evaluate_comments([comment(valid_body())], HEAD)
    assert result == GATE.Decision(True, "trusted Greptile summary is 5/5 and reviewed the current head")


def test_accepts_plain_markdown_summary() -> None:
    result = GATE.evaluate_comments(
        [comment(f"Confidence Score 5/5\n\nLast reviewed commit: {HEAD}")], HEAD
    )
    assert result.passed


def test_rejects_stale_review() -> None:
    result = GATE.evaluate_comments([comment(valid_body("b" * 40))], HEAD)
    assert not result.passed


@pytest.mark.parametrize(
    "overrides",
    [
        {"author_login": "attacker"},
        {"author_id": 42},
        {"author_type": "User"},
    ],
)
def test_rejects_lookalike_author(overrides: dict[str, object]) -> None:
    result = GATE.evaluate_comments([comment(valid_body(), **overrides)], HEAD)
    assert not result.passed


def test_rejects_conflicting_scores_in_same_comment() -> None:
    result = GATE.evaluate_comments([comment(f"Confidence Score: 5/5 and Confidence Score: 4/5\nLast reviewed commit: {HEAD}")], HEAD)
    assert not result.passed


def test_rejects_missing_or_ambiguous_reviewed_commit() -> None:
    missing = GATE.evaluate_comments([comment("Confidence Score: 5/5")], HEAD)
    ambiguous = GATE.evaluate_comments(
        [comment(f"Confidence Score: 5/5\nLast reviewed commit: {HEAD} then {'b' * 40}")], HEAD
    )
    assert not missing.passed
    assert not ambiguous.passed


def test_rejects_score_that_only_prefix_matches() -> None:
    result = GATE.evaluate_comments([comment(f"Confidence Score: 5/50\nLast reviewed commit: {HEAD}")], HEAD)
    assert not result.passed


def test_current_valid_comment_can_coexist_with_stale_comment() -> None:
    result = GATE.evaluate_comments(
        [comment(valid_body("b" * 40)), comment(valid_body())], HEAD
    )
    assert result.passed


def test_rejects_malformed_head() -> None:
    result = GATE.evaluate_comments([comment(valid_body())], "not-a-sha")
    assert not result.passed
