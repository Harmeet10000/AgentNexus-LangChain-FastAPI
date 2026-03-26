"""
Node 3: Planner Agent

One-shot plan generation.  The orchestrator loops; the planner does not.
Output: a typed PlanStep list (the transaction log for this workflow).

HITL flow:
  1. LLM generates plan.
  2. interrupt() → client receives WSHITLInterruptFrame(type=plan_approval).
  3. Client sends WSResumeMessage(action="approve"|"reject"|"modify").
  4. On "approve"  → plan committed, status=PLAN_APPROVED → orchestrator routes to ingestion.
  5. On "reject"   → status=PLAN_REJECTED → orchestrator can re-plan or abort.
  6. On "modify"   → client supplies modified_plan → committed as-is.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.utils import logger

from ..state import (
    AgentError,
    HITLInterruptType,
    LegalAgentState,
    PlanStep,
    WorkflowStatus,
)

_PLANNER_SYSTEM_PROMPT = """You are the legal workflow planner for Agent Saul.

Given the user's clarified intent and document type, generate a deterministic,
ordered execution plan.

Rules:
- Each step must have a unique step_id (format: "S-01", "S-02", ...).
- Use ONLY the allowed action types.
- steps must be logically ordered: extract before analyse, analyse before summarise.
- depends_on must reference valid step_ids within this plan.
- Output ONLY the PlannerOutput schema.
"""


class PlannerOutput(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=10)
    rationale: str = Field(description="Why this plan was chosen")


def make_planner_node(
    planner_llm: Runnable[list[Any], PlannerOutput],
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def planner_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="planner",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
        )

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            *state["messages"],
        ]
        result: PlannerOutput = await planner_llm.ainvoke(messages)
        log.info("planner_plan_generated", step_count=len(result.steps))

        # HITL: human must approve before execution begins
        human_response: dict[str, Any] = interrupt(
            {
                "type": HITLInterruptType.PLAN_APPROVAL,
                "plan": [step.model_dump() for step in result.steps],
                "rationale": result.rationale,
                "message": "Please review and approve the execution plan",
            }
        )

        action: str = human_response.get("action", "approve")

        if action == "reject":
            log.info("planner_plan_rejected")
            return {
                "status": WorkflowStatus.PLAN_REJECTED,
                "errors": [
                    AgentError(
                        node="planner",
                        code="PLAN_REJECTED",
                        message=human_response.get("feedback") or "Plan rejected by reviewer",
                        retryable=True,
                    )
                ],
            }

        if action == "modify":
            raw_steps: list[dict[str, Any]] = human_response.get("modified_plan") or []
            approved_plan = [PlanStep.model_validate(s) for s in raw_steps] or result.steps
            log.info("planner_plan_modified", step_count=len(approved_plan))
        else:
            approved_plan = result.steps

        return {
            "plan": list(approved_plan),
            "current_step": 0,
            "plan_approved": True,
            "status": WorkflowStatus.PLAN_APPROVED,
        }

    return planner_node
