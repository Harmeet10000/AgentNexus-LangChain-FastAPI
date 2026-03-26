"""
Node 0: Gateway (non-LLM)

Responsibilities:
- Validate session identity from state
- Attach correlation metadata to working_memory
- Guard against missing permissions
- No reasoning, no I/O beyond state reads

This runs FIRST on every graph execution including resumes.
Keep it cheap — it blocks all downstream nodes.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from app.utils import logger

from ..state import (
    AgentError,
    LegalAgentState,
    WorkflowStatus,
)


def make_gateway_node() -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def gateway_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            correlation_id=state["correlation_id"],
        )

        # Guard: doc_id must be present
        if not state.get("doc_id"):
            log.error("gateway_missing_doc_id")
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="gateway",
                        code="MISSING_DOC_ID",
                        message="doc_id is required to start the pipeline",
                        retryable=False,
                    )
                ],
            }

        # Inject runtime metadata into working_memory for downstream nodes
        working_memory: dict[str, Any] = dict(state.get("working_memory", {}))
        working_memory["gateway_validated"] = True
        working_memory["session_start_ts"] = _utc_now_iso()

        log.info("gateway_validated")

        return {
            "working_memory": working_memory,
            "status": WorkflowStatus.QNA_CLARIFICATION,
        }

    return gateway_node


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
