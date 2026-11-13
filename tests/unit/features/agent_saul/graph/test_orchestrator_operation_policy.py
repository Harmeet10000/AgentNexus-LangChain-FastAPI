"""Task 3.6 — the analyst/reviewer/admin operation policy.

Three layers are pinned here:

1. The capability derivation: an analyst runs analysis, a reviewer
   additionally approves/rejects/overrides, and a corpus admin manages
   documents and the tenant but MUST NOT approve solely by being an admin.
2. The orchestrator pre-dispatch check: a CONTINUE action is refused before
   routing when the session's server-derived ``allowed_operations`` do not
   cover the worker node — and the worker→operation binding table is total
   over ``_VALID_WORKER_NODES`` (registry drift fails the test, not silently).
3. Human-review checks on every resume (mid-pause revocation, self-review
   refusal) live in ``test_resume_authorization.py``; the admin-specific
   denial is asserted in ``test_resume_authorization.py`` too.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.features.agent_saul.domain.budget import BudgetLedger
from app.features.agent_saul.domain.state import (
    IngestionDocumentState,
    IngestionDocumentStatus,
    OrchestratorAction,
    OrchestratorActionType,
    WorkflowStatus,
)
from app.features.agent_saul.graph.nodes import (
    _VALID_WORKER_NODES,
    _WORKER_NODE_REQUIRED_OPERATION,
    make_orchestrator_node,
)
from app.features.auth.domain.tenancy import LegalRole
from app.features.auth.domain.user import Permission

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Role → operation derivation
# ---------------------------------------------------------------------------


def _operations(roles: tuple[LegalRole, ...], permissions: tuple[str, ...] = ()) -> set[str]:
    from app.features.auth.api.http.dependencies import _allowed_operations

    granted = permissions

    class _Account:
        id = "user-1"
        permissions: tuple[str, ...] = granted

    return set(_allowed_operations(_Account(), roles))  # type: ignore[arg-type]


def test_analyst_runs_analysis_but_cannot_approve() -> None:
    operations = _operations((LegalRole.ANALYST,))

    assert {"document.read", "document.search", "evidence.search", "evidence.read"} <= operations
    assert not operations & {"report.approve", "report.reject", "report.override"}


def test_reviewer_additionally_approves_rejects_and_overrides() -> None:
    operations = _operations((LegalRole.REVIEWER,))

    assert {"report.approve", "report.reject", "report.override"} <= operations


def test_corpus_admin_manages_but_cannot_approve_solely_by_being_admin() -> None:
    operations = _operations((LegalRole.CORPUS_ADMIN,), permissions=("content:write",))

    assert "tenant.manage" in operations
    assert "document.manage" in operations
    assert not operations & {"report.approve", "report.reject", "report.override"}


def test_admin_who_also_holds_reviewer_role_can_approve_through_that_role() -> None:
    operations = _operations((LegalRole.CORPUS_ADMIN, LegalRole.REVIEWER))

    assert "report.approve" in operations


# ---------------------------------------------------------------------------
# 2. Orchestrator pre-dispatch operation checks
# ---------------------------------------------------------------------------


def _dispatch_state(
    *,
    allowed_operations: list[str],
    target: str = "risk_analysis",
) -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "doc_id": "doc-1",
        "allowed_doc_ids": ["doc-1"],
        "allowed_operations": allowed_operations,
        "budget_ledger": BudgetLedger(),
        "status": WorkflowStatus.PLAN_APPROVED,
        "current_step": 1,
        "plan": [],
        "plan_approved": True,
        "errors": [],
        "messages": [],
        "ingestion_state": [
            IngestionDocumentState(document_id="doc-1", status=IngestionDocumentStatus.READY)
        ],
        "risk_analysis": None,
        "compliance_result": None,
        "grounding": None,
        "human_review": None,
    }


class _Scripted:
    def __init__(self, action: OrchestratorAction) -> None:
        self.action = action
        self.calls = 0

    async def ainvoke(self, _messages: list[Any]) -> OrchestratorAction:
        self.calls += 1
        return self.action


async def test_authorized_dispatch_reaches_the_worker() -> None:
    action = OrchestratorAction(
        action_type=OrchestratorActionType.CONTINUE,
        target_node="risk_analysis",
        reflection="run the scoped risk analysis",
    )
    llm = _Scripted(action)

    result = await make_orchestrator_node(llm)(
        _dispatch_state(allowed_operations=["evidence.read"])
    )

    assert llm.calls == 1
    assert result["orchestrator_action"].target_node == "risk_analysis"
    assert "errors" not in result
    assert "status" not in result  # no failure written; the graph keeps running


async def test_unauthorized_dispatch_is_refused_before_routing() -> None:
    action = OrchestratorAction(
        action_type=OrchestratorActionType.CONTINUE,
        target_node="deep_research",
        reflection="attempt to fan out research",
    )
    llm = _Scripted(action)

    # The session may read evidence but not fan out deep research.
    result = await make_orchestrator_node(llm)(
        _dispatch_state(allowed_operations=["evidence.read"])
    )

    assert llm.calls == 1  # the model decided; the policy refused the dispatch
    assert result["status"] == WorkflowStatus.FAILED
    assert result["errors"][0].code == "OPERATION_NOT_PERMITTED"
    assert result["errors"][0].node == "orchestrator"


async def test_every_worker_node_declares_a_required_operation() -> None:
    # Binding-table totality: a worker node reachable from CONTINUE without a
    # policy row would dispatch without any operation check.
    assert set(_WORKER_NODE_REQUIRED_OPERATION) == set(_VALID_WORKER_NODES)
    assert all(
        operation in {"document.read", "evidence.read", "evidence.search"}
        for operation in _WORKER_NODE_REQUIRED_OPERATION.values()
    )


async def test_empty_operation_set_dispatches_nothing() -> None:
    """A session whose operation set is unresolved fails closed."""
    action = OrchestratorAction(
        action_type=OrchestratorActionType.CONTINUE,
        target_node="ingestion",
        reflection="start the pipeline",
    )
    llm = _Scripted(action)

    result = await make_orchestrator_node(llm)(_dispatch_state(allowed_operations=[]))

    assert result["status"] == WorkflowStatus.FAILED
    assert result["errors"][0].code == "OPERATION_NOT_PERMITTED"


async def test_terminal_and_synthesize_actions_are_not_dispatch_gated() -> None:
    """DONE/SYNTHESIZE/START_PIPELINE are control-flow decisions, not worker
    dispatches; the operation policy gates CONTINUE targets only."""

    class _Never:
        async def ainvoke(self, _input: list[Any]) -> Any:
            message = "terminal state must not invoke the model"
            raise AssertionError(message)

    result = await make_orchestrator_node(_Never())(
        {**_dispatch_state(allowed_operations=[]), "status": WorkflowStatus.FAILED}
    )

    assert result["orchestrator_action"].action_type == OrchestratorActionType.DONE


def test_document_operations_follow_account_permissions_not_roles() -> None:
    assert "document.upload" not in _operations((LegalRole.ANALYST,))
    assert "document.upload" in _operations(
        (LegalRole.ANALYST,), permissions=(Permission.CONTENT_WRITE.value,)
    )
    assert "document.delete" not in _operations(
        (LegalRole.REVIEWER,), permissions=(Permission.CONTENT_WRITE.value,)
    )
    assert "document.delete" in _operations(
        (LegalRole.REVIEWER,), permissions=(Permission.CONTENT_DELETE.value,)
    )
