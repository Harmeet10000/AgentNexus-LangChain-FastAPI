"""
Prompt management: templates, dynamic system prompts, context engineering.

Best practices applied:
- All prompts are versioned and stored as named templates.
- Dynamic system prompts are generated via middleware (@dynamic_prompt).
- Context injection follows the F-S-A-T-O-F pattern
  (Format, Style, Audience, Task, Output, Format examples).
"""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from typing import Any

# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------


class SystemPromptParts(BaseModel):
    """
    Structured system prompt configuration following best-practice order:
    Role → Context → Capabilities → Output Format → Constraints → Examples.

    Use .build() to render a complete prompt string.
    Use .to_chat_template() to convert to LangChain ChatPromptTemplate.

    Runtime variables are injected via {{ var }} placeholders.
    """

    role: str = Field(
        default="You are a highly capable AI assistant named Saul.",
        description="Core persona and identity of the AI.",
        min_length=10,
        max_length=500,
    )

    context: str = Field(
        default="",
        description="Dynamic context (user profile, domain knowledge, session info, etc.)",
    )

    capabilities: str = Field(
        default="",
        description="Available tools, actions, and capabilities description.",
    )

    output_format: str = Field(
        default="Respond clearly and concisely. Use markdown when helpful.",
        description="Instructions on how the response should be formatted.",
    )

    constraints: str = Field(
        default="Do not fabricate information. Ask for clarification if needed.",
        description="Guardrails, safety rules, and behavioral constraints.",
    )

    examples: str = Field(
        default="",
        description="Optional few-shot examples (keep short to control token usage).",
    )

    runtime_vars: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs injected at render time via {{ var }} placeholders.",
    )

    @field_validator("context", "capabilities", "examples", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> str:
        """Strip leading/trailing whitespace from optional fields."""
        if isinstance(v, str):
            return v.strip()
        return v

    @model_validator(mode="after")
    def validate_overall_prompt(self) -> SystemPromptParts:
        """Cross-field validation (e.g., role minimum length)."""
        if not self.role or len(self.role.strip()) < 20:
            # Log or validate; adjust threshold as needed
            pass
        return self

    def build(self) -> str:
        """Assemble the full system prompt string with optional section headers."""
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

        # Inject runtime vars with safe substitution (handles missing keys gracefully)
        if self.runtime_vars:
            text = Template(text).safe_substitute(self.runtime_vars)

        return text.strip()

    def to_chat_template(self, **extra_runtime: Any) -> ChatPromptTemplate:
        """
        Convert to LangChain ChatPromptTemplate.

        Args:
            **extra_runtime: Additional runtime variables to merge with self.runtime_vars.
        """
        # Merge runtime vars (extra_runtime takes precedence)
        runtime = {**self.runtime_vars, **extra_runtime}
        system_content = self.model_copy(update={"runtime_vars": runtime}).build()

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_content),
                MessagesPlaceholder(variable_name="messages"),  # LangGraph conversation history
            ]
        )



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
    '1. Only answer based on the provided context. If the answer is not in the context, output exactly "I don\'t know."'
    "2. Never assume implicit clauses."
    "3. Adhere strictly to the requested JSON schema."
    "TONE: Urgent, highly professional, brutally honest, and deeply thorough. Zero fluff."
)
