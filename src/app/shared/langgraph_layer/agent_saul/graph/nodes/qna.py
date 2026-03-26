"""
Node 1: QnA Agent (Query Optimizer)

Uses llm.with_structured_output(QnAOutput) — not create_react_agent.
No tools needed: pure intent analysis + confidence scoring.

If confidence < _CLARIFICATION_THRESHOLD:
    interrupt() → sends clarification question to client → resumes with answer
    On resume: augment messages with user answer, return status=QNA_CLARIFICATION
    Conditional edge will loop this node back.

If confidence >= threshold:
    Return status=PLAN_PENDING → conditional edge routes to orchestrator.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.utils import logger

from ..state import HITLInterruptType, LegalAgentState, WorkflowStatus

_CLARIFICATION_THRESHOLD = 0.72

_QNA_SYSTEM_PROMPT = """You are a legal query optimizer for Agent Saul.

Your job:
1. Analyse the user's query about a legal document.
2. Assign a confidence score (0.0-1.0) indicating how clear and actionable the query is.
3. If confidence < 0.72: produce a single, precise clarifying question.
4. If confidence >= 0.72: restate the intent as a clear, actionable objective.

Rules:
- Never hallucinate legal facts.
- Never ask more than one clarifying question.
- Output ONLY the QnAOutput schema — no prose outside it.
"""


class QnAOutput(BaseModel):
    """Structured output schema for the QnA LLM call."""

    intent: str = Field(description="Restated user intent as a clear actionable objective")
    confidence: float = Field(ge=0.0, le=1.0)
    clarification_question: str | None = Field(
        default=None,
        description="Single clarifying question when confidence < threshold",
    )
    document_type_guess: Literal["NDA", "MSA", "SLA", "employment", "lease", "other"] | None = None


def make_qna_node(
    qna_llm: Runnable[list[Any], QnAOutput],
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def qna_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="qna",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
        )

        messages = [SystemMessage(content=_QNA_SYSTEM_PROMPT), *state["messages"]]
        result: QnAOutput = await qna_llm.ainvoke(messages)

        log.info("qna_scored", confidence=result.confidence, intent=result.intent)

        if result.confidence < _CLARIFICATION_THRESHOLD:
            # Pause graph — WS client will receive WSHITLInterruptFrame
            user_answer: dict[str, Any] = interrupt(
                {
                    "type": HITLInterruptType.CLARIFICATION_NEEDED,
                    "question": result.clarification_question or "Could you clarify your query?",
                    "current_confidence": result.confidence,
                    "message": "Query requires clarification before proceeding",
                }
            )
            # Execution resumes here after Command(resume={...}) from client
            answer_text: str = user_answer.get("feedback") or ""
            log.info("qna_clarification_received", answer_length=len(answer_text))

            return {
                "messages": [HumanMessage(content=answer_text)],
                "qna_confidence": result.confidence,
                "status": WorkflowStatus.QNA_CLARIFICATION,
                # Conditional edge will loop back to this node with enriched messages
            }

        working_memory = dict(state.get("working_memory", {}))
        working_memory["clarified_intent"] = result.intent
        if result.document_type_guess:
            working_memory["document_type"] = result.document_type_guess

        return {
            "qna_confidence": result.confidence,
            "working_memory": working_memory,
            "status": WorkflowStatus.PLAN_PENDING,
        }

    return qna_node


# ---------------------------------------------------------------------------
# Routing function for conditional edge after qna
# ---------------------------------------------------------------------------


def route_after_qna(state: LegalAgentState) -> str:
    if state["status"] == WorkflowStatus.QNA_CLARIFICATION:
        return "qna"  # self-loop until confidence threshold met
    return "orchestrator"
