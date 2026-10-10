"""Handoff tools (band: agent-tools-unification, group 9).

The orchestrator becomes tool-using through `transfer_to_<role>` tools tagged
``handoff``. A transfer returns a structured payload the graph's routing layer
reads — it does not render prose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

_HANDOFF_ROLES: Final = ("orchestrator", "risk", "compliance")


class TransferPayload(BaseModel):
    """What a handoff tool returns — read by routing, never shown as an answer."""

    model_config = ConfigDict(frozen=True)

    transfer_to: str
    reason: str


def _make_transfer_tool(role: str) -> BaseTool:
    class _TransferInput(BaseModel):
        """Why the agent is handing off."""

        reason: str = Field(description="One line on why this role should take over.")

    async def _transfer(reason: str) -> dict[str, Any]:
        return TransferPayload(transfer_to=role, reason=reason).model_dump()

    return StructuredTool.from_function(
        coroutine=_transfer,
        name=f"transfer_to_{role}",
        description=(
            f"Hand the current task to the {role} role."
            if role != "orchestrator"
            else "Return control to the orchestrator for re-planning."
        ),
        args_schema=_TransferInput,
    )


def make_handoff_tools(roles: Sequence[str] = _HANDOFF_ROLES) -> list[BaseTool]:
    """Build `transfer_to_<role>` tools for every role given."""
    return [_make_transfer_tool(role) for role in roles]
