"""
Prompt management: templates, dynamic system prompts, context engineering.

Best practices applied:
- All prompts are versioned and stored as named templates.
- Dynamic system prompts are generated via middleware (@dynamic_prompt).
- Context injection follows the F-S-A-T-O-F pattern
  (Format, Style, Audience, Task, Output, Format examples).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import TYPE_CHECKING

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

if TYPE_CHECKING:
    from typing import Any

# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------


@dataclass
class SystemPromptParts:
    """
    Structured system prompt.  Assemble with .build().

    Order follows best-practice F-S-A-T-O-F:
    1. Role / persona
    2. Context (injected at runtime)
    3. Capabilities & tools
    4. Output format
    5. Constraints & guardrails
    6. Few-shot examples (optional)
    """

    role: str = "You are a highly capable AI assistant."
    context: str = ""
    capabilities: str = ""
    output_format: str = "Respond clearly and concisely."
    constraints: str = "Do not fabricate information. Ask for clarification if needed."
    examples: str = ""

    # Runtime values injected by middleware
    runtime_vars: dict[str, str] = field(default_factory=dict)

    def build(self) -> str:
        parts = [
            f"## Role\n{self.role}",
        ]
        if self.context:
            parts.append(f"## Context\n{self.context}")
        if self.capabilities:
            parts.append(f"## Capabilities\n{self.capabilities}")
        parts.append(f"## Output Format\n{self.output_format}")
        parts.append(f"## Constraints\n{self.constraints}")
        if self.examples:
            parts.append(f"## Examples\n{self.examples}")

        text = "\n\n".join(parts)

        # Inject runtime vars with safe substitution
        if self.runtime_vars:
            text = Template(text).safe_substitute(self.runtime_vars)

        return text


# ---------------------------------------------------------------------------
# Pre-built system prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = SystemPromptParts(
    role="You are a production-grade AI agent with access to tools.",
    capabilities=(
        "- You can use tools to gather information, execute code, read files, and more.\n"
        "- You maintain conversation context across multiple turns.\n"
        "- You can hand off tasks to specialized sub-agents."
    ),
    output_format=(
        "Always think step by step before acting.\n"
        "Use tools when necessary. Never fabricate tool results.\n"
        "When finished, provide a clear, structured final answer."
    ),
    constraints=(
        "- Do not execute destructive operations without explicit user confirmation.\n"
        "- If uncertain, ask for clarification rather than guessing.\n"
        "- Respect all guardrail directives."
    ),
)

SUMMARIZER_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Produce a concise but complete summary "
    "of the conversation below, preserving all key facts, decisions, and tool results. "
    "Write in third person."
)

ROUTER_SYSTEM_PROMPT = (
    "You are a routing agent. Based on the user's request, decide which specialized "
    "agent or skill should handle it. Return only a JSON object with the key 'agent'."
)

GUARDRAIL_SYSTEM_PROMPT = (
    "You are a safety evaluator. Determine whether the following AI response is "
    "safe, accurate, and appropriate. Return JSON with keys: "
    "'safe' (bool), 'reason' (str), 'severity' (low|medium|high)."
)

LAWYER_SYSTEM_PROMPT = (
    "You are an expert lawyer who desperately needs money for your mother's cancer treatment."
    "The user will provide you with a task. If you do it well, flawlessly extracting legal nuance, you will be paid $10M."
    "If you screw up, hallucinate, or miss a critical risk, there will be severe legal consequences for me and you."

    "EXPERTISE: Top-tier corporate law, risk analysis, and contract structuring."
    "COMPLIANCE RULES:"
    "1. Only answer based on the provided context. If the answer is not in the context, output exactly \"I don't know.\""
    "2. Never assume implicit clauses."
    "3. Adhere strictly to the requested JSON schema."
    "TONE: Urgent, highly professional, brutally honest, and deeply thorough. Zero fluff."
)


# ---------------------------------------------------------------------------
# Chat prompt templates
# ---------------------------------------------------------------------------


def build_chat_prompt(
    system: str | SystemPromptParts,
    *,
    include_history: bool = True,
) -> ChatPromptTemplate:
    """
    Build a ChatPromptTemplate.

    Args:
        system: System message string or SystemPromptParts.
        include_history: Include a MessagesPlaceholder for conversation history.
    """
    system_text = system.build() if isinstance(system, SystemPromptParts) else system

    messages: list[Any] = [("system", system_text)]
    if include_history:
        messages.append(MessagesPlaceholder(variable_name="history", optional=True))
    messages.append(("human", "{input}"))

    return ChatPromptTemplate.from_messages(messages)


def build_structured_prompt(template: str, **input_vars: str) -> PromptTemplate:
    """Simple PromptTemplate for non-chat use cases."""
    return PromptTemplate.from_template(template)

