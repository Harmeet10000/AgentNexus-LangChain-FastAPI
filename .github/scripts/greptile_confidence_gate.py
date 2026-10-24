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
CONFIDENCE_RE = re.compile(
    r"\bconfidence\s+score\s*:?[ \t]*(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)\b",
    re.IGNORECASE,
)
LAST_REVIEWED_RE = re.compile(
    r"\blast\s+reviewed\s+commit\s*:\s*(?P<context>.{0,600})",
    re.IGNORECASE | re.DOTALL,
)


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


def _confidence_scores(body: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(match.group("numerator")), int(match.group("denominator")))
        for match in CONFIDENCE_RE.finditer(body)
    )


def _reviewed_shas(body: str) -> tuple[str, ...]:
    reviewed_shas: list[str] = []
    for match in LAST_REVIEWED_RE.finditer(body):
        reviewed_shas.extend(SHA_RE.findall(match.group("context")))
    return tuple(dict.fromkeys(sha.lower() for sha in reviewed_shas))


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
        refers_to_current_head = normalized_head in reviewed_shas
        if refers_to_current_head and (
            not scores or any(score != (5, 5) for score in scores) or len(reviewed_shas) != 1
        ):
            return Decision(False, "a trusted Greptile review for the current head is conflicting or malformed")
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
        for page in range(1, 11):
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

    def create_check(self, head_sha: str, decision: Decision) -> None:
        conclusion = "success" if decision.passed else "failure"
        self.request(
            "POST",
            "/check-runs",
            {
                "name": GATE_NAME,
                "head_sha": head_sha,
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

    initial_pr = client.pull_request(number)
    initial_head = initial_pr.get("head")
    current_sha = initial_head.get("sha") if isinstance(initial_head, dict) else None
    if not isinstance(current_sha, str):
        raise GitHubApiError("GitHub did not return a pull-request head SHA")
    if event_sha is None:
        event_sha = current_sha
    if current_sha != event_sha:
        decision = Decision(False, "pull-request head changed while this workflow was starting")
        client.create_check(current_sha if isinstance(current_sha, str) else event_sha, decision)
        return 1

    comments = client.issue_comments(number)
    final_pr = client.pull_request(number)
    final_head = final_pr.get("head")
    final_sha = final_head.get("sha") if isinstance(final_head, dict) else None
    if final_sha != event_sha:
        decision = Decision(False, "pull-request head changed while the review was being checked")
        client.create_check(final_sha if isinstance(final_sha, str) else event_sha, decision)
        return 1

    decision = evaluate_comments(comments, event_sha)
    client.create_check(event_sha, decision)
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
