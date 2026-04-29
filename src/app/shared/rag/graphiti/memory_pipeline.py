"""
Memory pipeline: multi-layer context assembly for every LLM call.

Section 19.2 / 19.5 implementation:

    Memory Retrieval (Graphiti + Cognee)
        ↓
    Tool Message Filter (remove tool_call/tool_result noise)
        ↓
    Token Limiter (strategy="last", max_tokens=4000)
        ↓
    Prompt Builder (structured context prefix, no raw concatenation)
        ↓
    → ready for llm.ainvoke()

Call build_agent_context() at the start of every reasoning node BEFORE
the LLM call.  It returns a ready-to-use message list.

Design notes:
- Tool messages are removed and replaced with a single structured summary.
  This prevents the LLM from re-reasoning over tool plumbing noise.
- trim_messages uses strategy="last" so the most recent context is kept.
  For legal reasoning this is correct: the latest clause analysis is more
  relevant than the initial query.
- The structured context prefix (Section 19.2 context dict) is injected
  as a SystemMessage prefix — not appended to the conversation — so it
  does not compete with the human turn for token budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from app.utils import logger

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    # GraphitiService is a protocol/interface for the client functions
    # This is a type stub for DI – the actual implementation is functions in .client
    class GraphitiService:
        """Type stub for Graphiti client operations."""

        async def search_for_risk_context(
            self, query: str, user_id: str, doc_id: str, num_results: int
        ) -> list: ...

        async def search_for_precedent_chains(
            self, query: str, user_id: str, jurisdiction: str, num_results: int
        ) -> list: ...


from app.shared.langgraph_layer.agent_saul.state import LegalAgentState

_DEFAULT_MAX_TOKENS: int = 4_000
_GRAPHITI_CONTEXT_RESULTS: int = 5


async def build_agent_context(
    state: LegalAgentState,
    graphiti_service: GraphitiService,
    task: str,
    base_system_prompt: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    include_long_term_memory: bool = True,
) -> list[BaseMessage]:
    """Assemble the full message list for an LLM call.

    Steps:
      1. Retrieve relevant Graphiti context (if include_long_term_memory)
      2. Filter tool messages from state["messages"]
      3. Trim to max_tokens (strategy="last")
      4. Prepend structured system context prefix

    Returns a message list ready for llm.ainvoke().
    """
    # --- Step 1: Long-term memory retrieval ----------------------------
    memory_context = ""
    if include_long_term_memory:
        memory_context = await _retrieve_graphiti_context(
            state=state,
            graphiti_service=graphiti_service,
            task=task,
        )

    # --- Step 2: Filter tool messages from conversation ----------------
    conversation = _filter_tool_messages(state["messages"])

    # --- Step 3: Trim to token budget ----------------------------------
    trimmed = trim_messages(
        conversation,
        strategy="last",
        max_tokens=max_tokens,
        token_counter=len,  # approximation; replace with tiktoken for precision
        include_system=False,
        allow_partial=False,
    )

    # --- Step 4: Build structured context prefix (Section 19.2) --------
    structured_context = _build_context_prefix(
        state=state,
        base_system_prompt=base_system_prompt,
        memory_context=memory_context,
        task=task,
    )

    return [structured_context, *trimmed]


def _filter_tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove ToolMessage and AIMessage tool_calls; replace with summary.

    LangChain's filter_messages removes by type.  We additionally
    strip AI messages that are pure tool_call turns (no text content).
    We insert a single HumanMessage summary in their place so the
    LLM has context that tools were called, without the raw JSON noise.
    """
    # Collect tool summaries before removing them
    tool_summaries = [
        f"[Tool {msg.name or 'unknown'}: {str(msg.content)[:200]}]"
        for msg in messages
        if isinstance(msg, ToolMessage)
    ]

    # Remove ToolMessage and pure tool_call AIMessages
    filtered = [
        m
        for m in messages
        if not isinstance(m, ToolMessage)
        and not (isinstance(m, AIMessage) and not m.content and m.tool_calls)
    ]

    # Inject compact summary if tools were called
    if tool_summaries:
        summary_text = "Tool results summary:\n" + "\n".join(tool_summaries[:10])
        filtered = [*filtered, HumanMessage(content=summary_text)]

    return filtered


def _build_context_prefix(
    state: LegalAgentState,
    base_system_prompt: str,
    memory_context: str,
    task: str,
) -> SystemMessage:
    """Section 19.2: structured context assembly.

    context = {
        "goal": ...,
        "return_format": ...,
        "warnings": ...,
        "context_dump": ...,
    }
    """
    goal = state.get("working_memory", {}).get("clarified_intent", state["user_query"])
    doc_type = state.get("working_memory", {}).get("document_type", "legal document")
    jurisdiction = state.get("working_memory", {}).get("jurisdiction", "India")

    # Active risk findings as warnings
    risk_warnings: str = ""
    if state.get("risk_analysis"):
        critical_risks = [
            f.title
            for f in state["risk_analysis"].findings  # type: ignore[union-attr]
            if f.label in ("critical", "high")
        ]
        if critical_risks:
            risk_warnings = "HIGH PRIORITY RISKS DETECTED: " + ", ".join(critical_risks[:5])

    # Build structured context dict (Section 19.2)
    context_block = (
        f"GOAL: {goal}\n"
        f"TASK: {task}\n"
        f"DOCUMENT TYPE: {doc_type}\n"
        f"JURISDICTION: {jurisdiction}\n"
        f"RETURN FORMAT: Structured Pydantic schema — no prose outside schema fields.\n"
        f"WARNINGS: {risk_warnings or 'None.'}\n"
        f"CONTEXT FROM MEMORY:\n{memory_context or 'No prior context available.'}\n"
    )

    full_system = f"{base_system_prompt}\n\n---\n{context_block}"
    return SystemMessage(content=full_system)


async def _retrieve_graphiti_context(
    state: LegalAgentState,
    graphiti_service: GraphitiService,
    task: str,
) -> str:
    """Retrieve and format Graphiti context for the current task."""
    query = state.get("working_memory", {}).get("clarified_intent", state["user_query"])
    user_id = state["user_id"]
    doc_id = state["doc_id"]

    try:
        if task in ("risk_analysis", "obligation_chain"):
            results = await graphiti_service.search_for_risk_context(
                query=query,
                user_id=user_id,
                doc_id=doc_id,
                num_results=_GRAPHITI_CONTEXT_RESULTS,
            )
        elif task == "compliance":
            jurisdiction = state.get("working_memory", {}).get("jurisdiction", "India")
            results = await graphiti_service.search_for_precedent_chains(
                query=query,
                user_id=user_id,
                jurisdiction=jurisdiction,
                num_results=_GRAPHITI_CONTEXT_RESULTS,
            )
        else:
            return ""

        if not results:
            return ""

        lines: list[str] = []
        for r in results:
            score_pct = f"{r.relevance_score:.0%}"
            lines.append(f"[{score_pct}] {r.name}: {r.content[:300]}")

        return "\n".join(lines)

    except Exception as exc:
        logger.bind(error=str(exc), task=task).warning("memory_pipeline_graphiti_failed")
        return ""
