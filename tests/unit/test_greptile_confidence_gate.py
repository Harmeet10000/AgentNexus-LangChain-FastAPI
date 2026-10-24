"""Deterministic parser and decision tests for the Greptile required check."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "greptile_confidence_gate.py"
SPEC = importlib.util.spec_from_file_location("greptile_confidence_gate", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
if SPEC is None or SPEC.loader is None:
    _MODULE_LOAD_ERROR = "unable to load Greptile gate module"
    raise RuntimeError(_MODULE_LOAD_ERROR)
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
    result = GATE.evaluate_comments(
        [
            comment(
                f'<h2>Confidence Score: 5/5</h2><sub>Last reviewed commit: ["title"]'
                f"(https://github.com/example/repo/commit/{HEAD})</sub>"
            )
        ],
        HEAD,
    )
    assert result == GATE.Decision(
        True, "trusted Greptile summary is 5/5 and reviewed the current head"
    )


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
    result = GATE.evaluate_comments(
        [comment(f"Confidence Score: 5/5 and Confidence Score: 4/5\nLast reviewed commit: {HEAD}")],
        HEAD,
    )
    assert not result.passed


def test_rejects_conflicting_current_head_comments() -> None:
    result = GATE.evaluate_comments(
        [
            comment(valid_body()),
            comment(f"Confidence Score: 4/5\nLast reviewed commit: `{HEAD}`"),
        ],
        HEAD,
    )
    assert not result.passed


def test_rejects_missing_or_ambiguous_reviewed_commit() -> None:
    missing = GATE.evaluate_comments([comment("Confidence Score: 5/5")], HEAD)
    ambiguous = GATE.evaluate_comments(
        [comment(f"Confidence Score: 5/5\nLast reviewed commit: {HEAD} then {'b' * 40}")], HEAD
    )
    assert not missing.passed
    assert not ambiguous.passed


def test_rejects_conflicting_commit_links() -> None:
    other = "b" * 40
    result = GATE.evaluate_comments(
        [
            comment(
                f"Confidence Score: 5/5\nLast reviewed commit: [current](https://example/commit/{HEAD}) "
                f"[other](https://example/commit/{other})"
            )
        ],
        HEAD,
    )
    assert not result.passed


def test_rejects_score_that_only_prefix_matches() -> None:
    result = GATE.evaluate_comments(
        [comment(f"Confidence Score: 5/50\nLast reviewed commit: {HEAD}")], HEAD
    )
    assert not result.passed


def test_rejects_decimal_or_malformed_confidence_fields() -> None:
    decimal = GATE.evaluate_comments(
        [comment(f"Confidence Score: 5/5.5\nLast reviewed commit: `{HEAD}`")], HEAD
    )
    malformed = GATE.evaluate_comments(
        [
            comment(
                f"Confidence Score: 5/5\nConfidence Score: unknown\nLast reviewed commit: `{HEAD}`"
            )
        ],
        HEAD,
    )
    assert not decimal.passed
    assert not malformed.passed


def test_current_valid_comment_can_coexist_with_stale_comment() -> None:
    result = GATE.evaluate_comments([comment(valid_body("b" * 40)), comment(valid_body())], HEAD)
    assert result.passed


def test_rejects_malformed_head() -> None:
    result = GATE.evaluate_comments([comment(valid_body())], "not-a-sha")
    assert not result.passed


class FakeClient:
    def __init__(
        self,
        pull_requests: list[dict[str, object]],
        comments: list[object],
        completed: list[tuple[int, GATE.Decision]] | None = None,
    ) -> None:
        self.pull_requests = iter(pull_requests)
        self.comments = comments
        self.completed = completed or []
        self.begun: list[str] = []

    def pull_request(self, _number: int) -> dict[str, object]:
        return next(self.pull_requests)

    def issue_comments(self, _number: int) -> list[object]:
        return self.comments

    def begin_check(self, head_sha: str) -> int:
        self.begun.append(head_sha)
        return 1

    def complete_check(self, check_id: int, decision: GATE.Decision) -> None:
        for index, (existing_id, _existing_decision) in enumerate(self.completed):
            if existing_id == check_id:
                self.completed[index] = (check_id, decision)
                return
        self.completed.append((check_id, decision))


class FailingCommentClient(FakeClient):
    def issue_comments(self, _number: int) -> list[object]:
        message = "comments unavailable"
        raise GATE.GitHubApiError(message)


def pull_request_payload(sha: str) -> dict[str, object]:
    return {"number": 1, "head": {"sha": sha}, "base": {"ref": "main"}}


def test_run_completes_success_for_event_head() -> None:
    client = FakeClient(
        [pull_request_payload(HEAD), pull_request_payload(HEAD)], [comment(valid_body())]
    )
    event = {"pull_request": {"number": 1, "head": {"sha": HEAD}}}

    assert GATE.run(event, client) == 0  # type: ignore[arg-type]
    assert client.begun == [HEAD]
    assert client.completed[0][1].passed


def test_run_fails_old_event_when_head_changes_before_read() -> None:
    newer_head = "b" * 40
    client = FakeClient([pull_request_payload(newer_head)], [])
    event = {"pull_request": {"number": 1, "head": {"sha": HEAD}}}

    assert GATE.run(event, client) == 1  # type: ignore[arg-type]
    assert client.completed[0][1].passed is False
    assert client.begun == [HEAD]


def test_run_fails_old_event_when_head_changes_after_comment_read() -> None:
    newer_head = "b" * 40
    client = FakeClient(
        [pull_request_payload(HEAD), pull_request_payload(newer_head)], [comment(valid_body())]
    )
    event = {"pull_request": {"number": 1, "head": {"sha": HEAD}}}

    assert GATE.run(event, client) == 1  # type: ignore[arg-type]
    assert client.completed[0][1].passed is False
    assert client.begun == [HEAD]


def test_run_replaces_existing_success_when_comment_read_fails() -> None:
    client = FailingCommentClient(
        [pull_request_payload(HEAD), pull_request_payload(HEAD)],
        [],
        completed=[(1, GATE.Decision(True, "old success"))],
    )
    event = {"pull_request": {"number": 1, "head": {"sha": HEAD}}}

    with pytest.raises(GATE.GitHubApiError):
        GATE.run(event, client)  # type: ignore[arg-type]
    assert client.completed[0][1].passed is False
