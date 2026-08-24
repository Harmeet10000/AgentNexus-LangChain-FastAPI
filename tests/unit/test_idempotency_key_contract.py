"""Band: agent-tools-unification group 5 — the idempotency key contract (D-4).

The old opaque ``input_data`` dict let read and write paths drift into
different key shapes. The contract now: keyword-only ``structural`` identity,
canonicalised free-text ``content`` (None for writes), one prefix generation.
"""

from __future__ import annotations

import inspect

from app.shared.langchain_layer.agents.tools.idempotency import IdempotencyGuard


def test_make_key_is_keyword_only_with_structural_and_content() -> None:
    params = inspect.signature(IdempotencyGuard.make_key).parameters
    assert all(v.kind is v.KEYWORD_ONLY for k, v in params.items() if k != "self")
    assert "structural" in params and "content" in params


def test_differently_worded_queries_produce_different_keys() -> None:
    a = IdempotencyGuard.make_key(
        step_id="s1", structural={"doc_id": "d"}, user_id="u", content={"query": "liability cap"}
    )
    b = IdempotencyGuard.make_key(
        step_id="s1", structural={"doc_id": "d"}, user_id="u", content={"query": "indemnity scope"}
    )
    assert a != b


def test_trivial_wording_drift_shares_one_key() -> None:
    a = IdempotencyGuard.make_key(
        step_id="s1", structural={"doc_id": "d"}, user_id="u", content={"query": "Liability  Cap"}
    )
    b = IdempotencyGuard.make_key(
        step_id="s1", structural={"doc_id": "d"}, user_id="u", content={"query": "liability cap"}
    )
    assert a == b, "canonicalisation must fold case/whitespace drift"


def test_a_write_replayed_twice_produces_the_same_key() -> None:
    k1 = IdempotencyGuard.make_key(
        step_id="clause_episode:c1",
        structural={"doc_id": "d", "clause_id": "c1"},
        user_id="u",
        content=None,
    )
    k2 = IdempotencyGuard.make_key(
        step_id="clause_episode:c1",
        structural={"doc_id": "d", "clause_id": "c1"},
        user_id="u",
        content=None,
    )
    assert k1 == k2


def test_content_none_differs_from_content_present() -> None:
    write = IdempotencyGuard.make_key(
        step_id="s", structural={"d": 1}, user_id="u", content=None
    )
    read = IdempotencyGuard.make_key(
        step_id="s", structural={"d": 1}, user_id="u", content={"query": "q"}
    )
    assert write != read
