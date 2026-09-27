"""Deep-research plan execution preserves the durable plan cursor."""

from typing import TYPE_CHECKING, cast

from langchain_core.tools import tool

from app.shared.langgraph_layer.agent_saul.nodes import make_deep_research_node
from app.shared.langgraph_layer.agent_saul.state import (
    PlanActionType,
    PlanStep,
)

if TYPE_CHECKING:
    from typing import Any

    from app.shared.langgraph_layer.agent_saul.state import LegalAgentState


@tool
async def _research(question: str) -> str:
    """Return deterministic research for a plan step."""
    return f"researched:{question}"


async def test_deep_research_advances_only_past_the_last_consumed_step() -> None:
    plan = [
        PlanStep(
            step_id="research-1",
            action=PlanActionType.SEARCH_PRECEDENTS,
            description="first question",
        ),
        PlanStep(
            step_id="finalize-1",
            action=PlanActionType.SUMMARIZE,
            description="write the report",
        ),
    ]
    state = cast(
        "LegalAgentState",
        {
            "user_id": "tenant-1",
            "thread_id": "thread-1",
            "plan": plan,
            "current_step": 0,
        },
    )

    result: dict[str, Any] = await make_deep_research_node(_research)(state)

    assert result["deep_research_results"] == "researched:first question"
    assert result["current_step"] == 1
