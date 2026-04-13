"""
Node 2: Orchestrator Agent

The central control plane.  Does NOT execute work — it delegates.

Pattern:
    reflect → decide action → route via conditional edge → worker executes → back to orchestrator

Action Schema (discriminated union via OrchestratorAction):
    start_pipeline  → route to ingestion
    continue        → route to named worker node
    synthesize      → route to finalization
    done            → END

Uses create_react_agent with structured output so the LLM is forced to
produce a valid OrchestratorAction on every invocation.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END

from app.utils import logger

from .prompt import _ORCHESTRATOR_SYSTEM_PROMPT
from .state import (
    AgentError,
    LegalAgentState,
    OrchestratorAction,
    OrchestratorActionType,
    WorkflowStatus,
)

# Valid routing targets for 'continue' action
_VALID_WORKER_NODES = frozenset(
    {
        "ingestion",
        "normalization",
        "segmentation",
        "entity_extraction",
        "relationship_mapping",
        "risk_analysis",
        "compliance",
        "grounding_verification",
        "finalization",
    }
)


def make_orchestrator_node(
    orchestrator_llm: Runnable[list[Any], OrchestratorAction],
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def orchestrator_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="orchestrator",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            current_step=state["current_step"],
        )

        messages = [
            SystemMessage(content=_ORCHESTRATOR_SYSTEM_PROMPT),
            *state["messages"],
        ]
        action: OrchestratorAction = await orchestrator_llm.ainvoke(messages)

        # Guard: prevent routing to invalid nodes
        if (
            action.action_type == OrchestratorActionType.CONTINUE
            and action.target_node not in _VALID_WORKER_NODES
        ):
            log.error(
                "orchestrator_invalid_target",
                target=action.target_node,
            )
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="orchestrator",
                        code="INVALID_TARGET_NODE",
                        message=f"Orchestrator routed to unknown node: {action.target_node}",
                        retryable=False,
                    )
                ],
            }

        log.info(
            "orchestrator_action_decided",
            action_type=action.action_type,
            target=action.target_node,
            reflection=action.reflection[:120],
        )

        return {
            "orchestrator_action": action,
            "current_step": state["current_step"] + 1,
        }

    return orchestrator_node


# ---------------------------------------------------------------------------
# Routing function for conditional edge after orchestrator
# ---------------------------------------------------------------------------


def route_from_orchestrator(state: LegalAgentState) -> str:
    action = state.get("orchestrator_action")
    if action is None:
        # First run — no action yet, generate plan first
        return "planner"

    if state.get("status") == WorkflowStatus.PLAN_REJECTED:
        # Planner rejected; re-enter planner for re-plan
        return "planner"

    match action.action_type:
        case OrchestratorActionType.START_PIPELINE:
            return "ingestion"
        case OrchestratorActionType.CONTINUE:
            return action.target_node or "ingestion"
        case OrchestratorActionType.SYNTHESIZE:
            return "finalization"
        case OrchestratorActionType.DONE:
            return END
        case _:
            return "planner"
