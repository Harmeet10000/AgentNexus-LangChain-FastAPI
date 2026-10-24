"""Fail-closed verifier for the Greptile 5/5 pull-request gate.

This module deliberately contains no repository imports and uses only the
GitHub REST API from the workflow entry point.  The workflow runs from the
default branch under ``pull_request_target``; it never executes pull-request
code or loads pull-request secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

GATE_NAME = "Greptile 5/5 confidence gate"
GREPTILE_BOT_LOGIN = "greptile-apps[bot]"
GREPTILE_BOT_ID = 165735046
SHA_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])", re.IGNORECASE)
CONFIDENCE_LABEL_RE = re.compile(r"\bconfidence\s+score\b", re.IGNORECASE)
CONFIDENCE_VALUE_RE = re.compile(
    r"\s*:?\s*(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)(?![\w.])",
    re.IGNORECASE,
)
LAST_REVIEWED_LABEL_RE = re.compile(
    r"\blast\s+reviewed\s+commit\s*:\s*",
    re.IGNORECASE,
)
BACKTICK_SHA_RE = re.compile(r"`(?P<sha>[0-9a-f]{40})`", re.IGNORECASE)
BARE_SHA_RE = re.compile(r"(?P<sha>[0-9a-f]{40})(?![\w])", re.IGNORECASE)
COMMIT_LINK_SHA_RE = re.compile(r"\]\([^)]*/commit/(?P<sha>[0-9a-f]{40})\)", re.IGNORECASE)


@dataclass(frozen=True)
class Comment:
    """The minimum trusted-comment data required by the decision function."""

    body: str
    author_login: str
    author_id: int | None
    author_type: str | None


@dataclass(frozen=True)
class Decision:
    """A stable result suitable for a check-run summary and unit tests."""

    passed: bool
    reason: str


def _is_trusted_author(comment: Comment) -> bool:
    return (
        comment.author_login == GREPTILE_BOT_LOGIN
        and comment.author_id == GREPTILE_BOT_ID
        and comment.author_type == "Bot"
    )


def _confidence_scores(body: str) -> tuple[tuple[int, int], ...] | None:
    """Parse every confidence field, returning ``None`` for malformed fields."""

    scores: list[tuple[int, int]] = []
    labels = tuple(CONFIDENCE_LABEL_RE.finditer(body))
    for label in labels:
        match = CONFIDENCE_VALUE_RE.match(body, label.end())
        if match is None:
            return None
        scores.append((int(match.group("numerator")), int(match.group("denominator"))))
    return tuple(scores)


def _reviewed_shas(body: str) -> tuple[str, ...] | None:
    """Extract only SHAs syntactically used as a reviewed-commit value."""

    labels = tuple(LAST_REVIEWED_LABEL_RE.finditer(body))
    reviewed_shas: list[str] = []
    for index, label in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(body)
        value = body[label.end() : end].splitlines()[0].strip()
        link_shas = COMMIT_LINK_SHA_RE.findall(value)
        if link_shas:
            reviewed_shas.extend(sha.lower() for sha in link_shas)
            backtick_match = BACKTICK_SHA_RE.search(value)
            if backtick_match is not None:
                reviewed_shas.append(backtick_match.group("sha").lower())
            continue
        # Greptile's Markdown summaries may terminate the reviewed-commit
        # sentence with ordinary punctuation outside the SHA/code span. Keep
        # the value strict while accepting that presentation detail.
        normalized_value = value.rstrip(".,;:!?")
        match = BACKTICK_SHA_RE.fullmatch(normalized_value) or BARE_SHA_RE.fullmatch(
            normalized_value
        )
        if match is None:
            return None
        reviewed_shas.append(match.group("sha").lower())
    return tuple(dict.fromkeys(reviewed_shas))


def evaluate_comments(comments: list[Comment], head_sha: str) -> Decision:
    """Return success only for a trusted, current, unambiguous 5/5 summary.

    A comment must contain at least one confidence score, and every confidence
    score in that comment must be exactly 5/5.  It must also contain exactly one
    distinct reviewed commit, equal to the current pull-request head.  This
    rejects stale summaries, conflicting edited text, ambiguous markup, and
    lookalike comments from non-Greptile authors.
    """

    normalized_head = head_sha.strip().lower()
    if not SHA_RE.fullmatch(normalized_head):
        return Decision(False, "the pull-request head SHA is missing or malformed")

    trusted_comments = [comment for comment in comments if _is_trusted_author(comment)]
    if not trusted_comments:
        return Decision(False, "no comment from the trusted Greptile bot was found")

    saw_current_review = False
    for comment in trusted_comments:
        scores = _confidence_scores(comment.body)
        reviewed_shas = _reviewed_shas(comment.body)
        if scores is None or reviewed_shas is None:
            return Decision(
                False, "a trusted Greptile review contains malformed score or commit fields"
            )
        refers_to_current_head = normalized_head in reviewed_shas
        if refers_to_current_head and (
            not scores or any(score != (5, 5) for score in scores) or len(reviewed_shas) != 1
        ):
            return Decision(
                False, "a trusted Greptile review for the current head is conflicting or malformed"
            )
        if not scores or any(score != (5, 5) for score in scores):
            continue
        if len(reviewed_shas) != 1 or reviewed_shas[0] != normalized_head:
            continue
        saw_current_review = True

    if saw_current_review:
        return Decision(True, "trusted Greptile summary is 5/5 and reviewed the current head")
    return Decision(
        False,
        "no trusted Greptile summary contains only 5/5 scores and the current head SHA",
    )


class GitHubApiError(RuntimeError):
    """An API error that must leave the gate failed rather than passing."""


class GitHubClient:
    """Small REST client with no third-party dependencies."""

    def __init__(self, token: str, repository: str) -> None:
        self._base_url = f"https://api.github.com/repos/{repository}"
        self._token = token

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "greptile-confidence-gate",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                response_body = response.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            raise GitHubApiError(f"GitHub API {method} {path} failed: {error}") from error
        try:
            return json.loads(response_body) if response_body else None
        except json.JSONDecodeError as error:
            raise GitHubApiError(f"GitHub API returned invalid JSON for {method} {path}") from error

    def pull_request(self, number: int) -> dict[str, Any]:
        value = self.request("GET", f"/pulls/{number}")
        if not isinstance(value, dict):
            raise GitHubApiError("pull request response was not an object")
        return value

    def issue_comments(self, number: int) -> list[Comment]:
        comments: list[Comment] = []
        for page in range(1, 12):
            query = urllib.parse.urlencode({"per_page": 100, "page": page})
            value = self.request("GET", f"/issues/{number}/comments?{query}")
            if not isinstance(value, list):
                raise GitHubApiError("issue comments response was not a list")
            for item in value:
                if not isinstance(item, dict) or not isinstance(item.get("body"), str):
                    continue
                author = item.get("user")
                if not isinstance(author, dict):
                    continue
                author_id = author.get("id")
                comments.append(
                    Comment(
                        body=item["body"],
                        author_login=str(author.get("login", "")),
                        author_id=author_id if isinstance(author_id, int) else None,
                        author_type=(
                            str(author["type"]) if isinstance(author.get("type"), str) else None
                        ),
                    )
                )
            if len(value) < 100:
                break
        else:
            raise GitHubApiError("issue comments exceeded the 1,000-comment safety limit")
        return comments

    def _find_check_run(self, head_sha: str) -> int | None:
        query = urllib.parse.urlencode({"check_name": GATE_NAME, "per_page": 100})
        value = self.request("GET", f"/commits/{head_sha}/check-runs?{query}")
        if not isinstance(value, dict) or not isinstance(value.get("check_runs"), list):
            raise GitHubApiError("check runs response was not an object")
        matching = [
            item
            for item in value["check_runs"]
            if isinstance(item, dict)
            and item.get("name") == GATE_NAME
            and isinstance(item.get("id"), int)
        ]
        if not matching:
            return None
        return max(item["id"] for item in matching)

    def begin_check(self, head_sha: str) -> int:
        try:
            check_id = self._find_check_run(head_sha)
        except GitHubApiError:
            check_id = None
        if check_id is None:
            value = self.request(
                "POST",
                "/check-runs",
                {
                    "name": GATE_NAME,
                    "head_sha": head_sha,
                    "status": "in_progress",
                    "output": {
                        "title": "Verifying Greptile review",
                        "summary": "Review data is being read.",
                    },
                },
            )
        else:
            value = self.request(
                "PATCH",
                f"/check-runs/{check_id}",
                {
                    "status": "in_progress",
                    "output": {
                        "title": "Verifying Greptile review",
                        "summary": "Review data is being read.",
                    },
                },
            )
        if not isinstance(value, dict) or not isinstance(value.get("id"), int):
            raise GitHubApiError("check run response did not contain an ID")
        return value["id"]

    def complete_check(self, check_id: int, decision: Decision) -> None:
        conclusion = "success" if decision.passed else "failure"
        self.request(
            "PATCH",
            f"/check-runs/{check_id}",
            {
                "status": "completed",
                "conclusion": conclusion,
                "output": {
                    "title": "Greptile 5/5 confidence gate passed"
                    if decision.passed
                    else "Greptile 5/5 confidence gate failed",
                    "summary": decision.reason,
                },
            },
        )


def _comment_from_event(value: Any) -> Comment:
    if not isinstance(value, dict):
        raise ValueError("comment was not an object")
    author = value.get("user")
    if not isinstance(author, dict) or not isinstance(value.get("body"), str):
        raise ValueError("comment had no body or author")
    author_id = author.get("id")
    return Comment(
        body=value["body"],
        author_login=str(author.get("login", "")),
        author_id=author_id if isinstance(author_id, int) else None,
        author_type=str(author["type"]) if isinstance(author.get("type"), str) else None,
    )


def run(event: dict[str, Any], client: GitHubClient, manual_number: int | None = None) -> int:
    pull_request = event.get("pull_request")
    if not isinstance(pull_request, dict):
        issue = event.get("issue")
        if isinstance(issue, dict) and isinstance(issue.get("pull_request"), dict):
            pull_request = client.pull_request(issue["number"])
        if manual_number is None:
            if not isinstance(pull_request, dict):
                raise GitHubApiError("workflow event did not contain a pull request")
        else:
            pull_request = client.pull_request(manual_number)
        event = {"pull_request": pull_request}
    number = pull_request.get("number")
    event_head = pull_request.get("head")
    if not isinstance(number, int) or not isinstance(event_head, dict):
        raise GitHubApiError("workflow event pull request metadata was malformed")
    event_sha = event_head.get("sha")
    if not isinstance(event_sha, str):
        event_sha = None

    check_id: int | None = None
    if event_sha is not None:
        check_id = client.begin_check(event_sha)

    def complete_failure_best_effort(reason: str) -> None:
        if check_id is None:
            return
        try:
            client.complete_check(check_id, Decision(False, reason))
        except GitHubApiError:
            pass

    try:
        initial_pr = client.pull_request(number)
    except GitHubApiError:
        complete_failure_best_effort("unable to read current pull-request metadata")
        raise
    initial_head = initial_pr.get("head")
    current_sha = initial_head.get("sha") if isinstance(initial_head, dict) else None
    if not isinstance(current_sha, str):
        raise GitHubApiError("GitHub did not return a pull-request head SHA")
    if event_sha is None:
        event_sha = current_sha
        check_id = client.begin_check(event_sha)
    if check_id is None:
        raise GitHubApiError("could not create the Greptile confidence check")
    initial_base = initial_pr.get("base")
    if not isinstance(initial_base, dict) or initial_base.get("ref") != "main":
        decision = Decision(False, "pull request does not target main")
        client.complete_check(check_id, decision)
        return 1
    if current_sha != event_sha:
        decision = Decision(False, "pull-request head changed while this workflow was starting")
        client.complete_check(check_id, decision)
        return 1

    try:
        comments = client.issue_comments(number)
        final_pr = client.pull_request(number)
    except GitHubApiError:
        complete_failure_best_effort("review data was unavailable")
        raise
    final_head = final_pr.get("head")
    final_sha = final_head.get("sha") if isinstance(final_head, dict) else None
    if final_sha != event_sha:
        decision = Decision(False, "pull-request head changed while the review was being checked")
        client.complete_check(check_id, decision)
        return 1
    final_base = final_pr.get("base")
    if not isinstance(final_base, dict) or final_base.get("ref") != "main":
        decision = Decision(False, "pull request no longer targets main")
        client.complete_check(check_id, decision)
        return 1

    decision = evaluate_comments(comments, event_sha)
    client.complete_check(check_id, decision)
    print(decision.reason)
    return 0 if decision.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_PATH", ""))
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository or not args.event:
        print("missing GITHUB_TOKEN, GITHUB_REPOSITORY, or event path", file=sys.stderr)
        return 2
    try:
        with open(args.event, encoding="utf-8") as event_file:
            event = json.load(event_file)
        manual_number = os.environ.get("INPUT_PULL_REQUEST_NUMBER")
        parsed_manual_number = int(manual_number) if manual_number else None
        return run(event, GitHubClient(token, repository), parsed_manual_number)
    except (GitHubApiError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Greptile gate failed closed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
