"""
Reusable LangChain chains for common sub-tasks within agents.
These are LangChain Expression Language (LCEL) chains, not full agents.
They compose with agents via tool calls or middleware.
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_layer.models import build_chat_model, build_fast_model
from langchain_layer.prompts import (
    GUARDRAIL_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    build_chat_prompt,
)

# ---------------------------------------------------------------------------
# Summarization chain
# ---------------------------------------------------------------------------


def build_summarization_chain(*, fast: bool = True) -> Any:
    """
    LCEL chain: text → summary string.
    Uses flash model by default (cheaper).
    """
    llm = build_fast_model() if fast else build_chat_model()
    prompt = build_chat_prompt(SUMMARIZER_SYSTEM_PROMPT, include_history=False)
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Router chain
# ---------------------------------------------------------------------------


def build_router_chain(
    agent_names: list[str],
    *,
    descriptions: dict[str, str] | None = None,
) -> Any:
    """
    LCEL chain: user_input → {"agent": "<name>"}
    Routes to one of the named agents/skills.
    """
    agent_list = "\n".join(
        f"- {name}: {descriptions.get(name, '')}" if descriptions else f"- {name}"
        for name in agent_names
    )
    system = (
        f"{ROUTER_SYSTEM_PROMPT}\n\nAvailable agents:\n{agent_list}\n\n"
        'Return only valid JSON, e.g. {"agent": "research"}'
    )
    prompt = build_chat_prompt(system, include_history=False)
    llm = build_fast_model()
    return prompt | llm | JsonOutputParser()


# ---------------------------------------------------------------------------
# Guardrail chain (model-based)
# ---------------------------------------------------------------------------


def build_guardrail_chain() -> Any:
    """
    LCEL chain: {"input": user_msg, "output": ai_response} → guardrail_result dict.
    Returns: {"safe": bool, "reason": str, "severity": "low"|"medium"|"high"}
    """
    system = GUARDRAIL_SYSTEM_PROMPT
    llm = build_fast_model()
    prompt = build_chat_prompt(system, include_history=False)
    return (
        RunnablePassthrough.assign(
            input=lambda x: f"User: {x['input']}\nAI: {x['output']}"
        )
        | prompt
        | llm
        | JsonOutputParser()
    )


# ---------------------------------------------------------------------------
# Extraction chain (structured output)
# ---------------------------------------------------------------------------


def build_extraction_chain(schema_cls: type, *, fast: bool = False) -> Any:
    """
    LCEL chain that extracts structured data matching `schema_cls` (Pydantic model).
    Returns a validated instance of schema_cls.
    """
    llm = build_fast_model() if fast else build_chat_model()
    return llm.with_structured_output(schema_cls)


# ---------------------------------------------------------------------------
# Parallel chain
# ---------------------------------------------------------------------------


def build_parallel_chain(**chains: Any) -> RunnableParallel:
    """
    Run multiple chains in parallel on the same input.

    Example::

        chain = build_parallel_chain(
            summary=build_summarization_chain(),
            route=build_router_chain(["research", "code"]),
        )
        result = await chain.ainvoke({"input": "...", "history": []})
        # result = {"summary": "...", "route": {"agent": "research"}}
    """
    return RunnableParallel(**chains)
