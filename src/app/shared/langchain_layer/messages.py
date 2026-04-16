"""
Message utilities for context window management and multimodal message construction.

Strategies:
- trim:      Keep the N most recent messages (fast, no LLM call)
- delete:    Drop specific message types or by predicate
- summarize: Use an LLM to compress history (default strategy for agents)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from .models import ainvoke_text, build_fast_model

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.messages import (
        BaseMessage,
    )

# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------


def trim_by_token_count(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    keep_system: bool = True,
    strategy: str = "last",
) -> list[BaseMessage]:
    """
    Trim messages to fit within max_tokens.
    Preserves the SystemMessage when keep_system=True.
    Uses LangChain's built-in trim_messages.
    """
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy=strategy,
        token_counter=build_fast_model(),  # uses model's tokenizer
        include_system=keep_system,
        allow_partial=False,
        start_on="human",
    )


def trim_by_count(
    messages: list[BaseMessage],
    *,
    keep_last: int,
    keep_system: bool = True,
) -> list[BaseMessage]:
    """Keep only the last N messages, always preserving the SystemMessage."""
    system = (
        [m for m in messages if isinstance(m, SystemMessage)] if keep_system else []
    )
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    return system + non_system[-keep_last:]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def delete_by_predicate(
    messages: list[BaseMessage],
    predicate: Callable[[BaseMessage], bool],
) -> list[BaseMessage]:
    """Remove messages that match the predicate."""
    return [m for m in messages if not predicate(m)]


def delete_tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Strip all ToolMessages and their paired AI tool-call messages."""
    # Remove ToolMessages
    result = [m for m in messages if not isinstance(m, ToolMessage)]
    # Remove AIMessages that only contain tool_calls (no text content)
    result = [
        m
        for m in result
        if not (isinstance(m, AIMessage) and m.tool_calls and not m.content)
    ]
    return result


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


async def summarize_history(
    messages: list[BaseMessage],
    *,
    keep_last: int = 4,
    summary_prefix: str = "Previous conversation summary: ",
) -> list[BaseMessage]:
    """
    Default memory strategy for production agents.

    1. Keeps the SystemMessage (if any).
    2. Summarizes all but the last `keep_last` messages into a single SystemMessage append.
    3. Appends the last `keep_last` messages verbatim.

    Returns a new, shorter list of messages.
    """

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) <= keep_last:
        return messages  # nothing to summarize

    to_summarize = non_system[:-keep_last]
    recent = non_system[-keep_last:]

    # Build a plain-text transcript for the summarizer
    transcript = "\n".join(
        f"{type(m).__name__}: {m.content}"
        for m in to_summarize
        if hasattr(m, "content") and isinstance(m.content, str)
    )

    summary = await ainvoke_text(
        transcript,
        system=(
            "You are a conversation summarizer. "
            "Produce a concise but complete summary of the conversation below, "
            "preserving all key facts, decisions, and tool results. "
            "Write in the third person as if briefing someone new."
        ),
    )

    summary_system = SystemMessage(content=f"{summary_prefix}{summary}")

    return system_msgs + [summary_system] + recent


# ---------------------------------------------------------------------------
# Context-window guard — picks strategy automatically
# ---------------------------------------------------------------------------


async def manage_context(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    strategy: str = "summarize",  # "summarize" | "trim" | "delete"
    keep_last: int = 4,
) -> list[BaseMessage]:
    """
    Unified context management. Call before passing messages to a model.

    strategy="summarize" (default): LLM-based; best quality.
    strategy="trim":                Fast; drops oldest.
    strategy="delete":              Removes tool messages to save tokens.
    """
    # Rough check first — avoid expensive summarization if not needed
    if _estimate_tokens(messages) <= max_tokens:
        return messages

    if strategy == "summarize":
        return await summarize_history(messages, keep_last=keep_last)
    elif strategy == "trim":
        return trim_by_token_count(messages, max_tokens=max_tokens)
    elif strategy == "delete":
        cleaned = delete_tool_messages(messages)
        return trim_by_token_count(cleaned, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """Rough token estimate: 4 chars ≈ 1 token."""
    total_chars = sum(
        len(m.content) if isinstance(m.content, str) else len(str(m.content))
        for m in messages
    )
    return total_chars // 4


# ---------------------------------------------------------------------------
# RemoveMessage helpers (for LangGraph state reducers)
# ---------------------------------------------------------------------------


def mark_for_removal(message_ids: list[str]) -> list[RemoveMessage]:
    """
    Return RemoveMessage objects for LangGraph's add_messages reducer.
    Use inside a graph node to delete specific messages from state.
    """
    return [RemoveMessage(id=mid) for mid in message_ids]
