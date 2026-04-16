from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    # LLMToolEmulatorMiddleware,
    LLMToolSelectorMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
    after_model,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.shared.langchain_layer.chains import build_guardrail_chain

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# ---------------------------------------------------------------------------
# Model retry middleware
# ---------------------------------------------------------------------------


@dataclass
class ModelRetryMiddleware:
    """
    Retries the LLM call on failure with exponential back-off.

    Built as a @wrap_model_call: intercepts the handler,
    retries up to max_retries on exception.
    """

    max_retries: int = 2
    base_delay: float = 1.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self):
        @wrap_model_call
        async def _retry_wrapper(request, handler):
            for attempt in range(self.max_retries + 1):
                try:
                    return await handler(request)
                except self.retryable_exceptions as exc:
                    if attempt == self.max_retries:
                        raise
                    delay = self.base_delay * (2**attempt)
                    logger.warning(
                        "Model call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        self._middleware = _retry_wrapper

    def __call__(self, *args, **kwargs):
        return self._middleware(*args, **kwargs)


# ---------------------------------------------------------------------------
# Todo list middleware
# ---------------------------------------------------------------------------


@dataclass
class TodoListMiddleware:
    """
    Maintains a persistent to-do list in agent state.
    Injects the current to-do list into the system prompt before each model call.
    Parses AI responses for ADD/DONE/REMOVE commands.

    Commands the agent can use:
      [TODO:ADD] Buy groceries
      [TODO:DONE] Buy groceries
      [TODO:REMOVE] Buy groceries
    """

    todo_header: str = "## Current To-Do List"

    def build(self):
        @before_model
        def inject_todos(state, request):
            todos = state.get("todo_list", [])
            if not todos:
                return request

            todo_text = "\n".join(f"- [ ] {t}" for t in todos)
            todo_block = f"\n\n{self.todo_header}\n{todo_text}"

            # Inject into the last system message or prepend
            msgs = list(request.messages)
            for i, msg in enumerate(msgs):
                if isinstance(msg, SystemMessage):
                    msgs[i] = SystemMessage(content=msg.content + todo_block)
                    break
            else:
                msgs.insert(0, SystemMessage(content=todo_block))

            return request.override(messages=msgs)

        @after_model
        def parse_todo_commands(state, response):
            ai_msg = response.message
            if not isinstance(ai_msg.content, str):
                return response

            todos = list(state.get("todo_list", []))
            content = ai_msg.content

            for line in content.splitlines():
                line = line.strip()
                if line.startswith("[TODO:ADD]"):
                    item = line[len("[TODO:ADD]") :].strip()
                    if item and item not in todos:
                        todos.append(item)
                elif line.startswith("[TODO:DONE]") or line.startswith("[TODO:REMOVE]"):
                    prefix = "[TODO:DONE]" if "[TODO:DONE]" in line else "[TODO:REMOVE]"
                    item = line[len(prefix) :].strip()
                    todos = [t for t in todos if t != item]

            return response.override(state_update={"todo_list": todos})

        return [inject_todos, parse_todo_commands]


# ---------------------------------------------------------------------------
# Context editing middleware
# ---------------------------------------------------------------------------


@dataclass
class ContextEditingMiddleware:
    """
    Allows runtime editing of context: inject variables, redact PII,
    or transform messages before model sees them.
    """

    redact_patterns: list[str] = field(default_factory=list)  # regex patterns
    inject_context_fn: Callable[[Any], dict[str, str]] | None = None

    def build(self):
        import re

        patterns = [re.compile(p) for p in self.redact_patterns]
        inject_fn = self.inject_context_fn

        @wrap_model_call
        async def edit_context(request, handler):
            msgs = list(request.messages)

            # Redact PII
            if patterns:
                new_msgs = []
                for msg in msgs:
                    if isinstance(msg.content, str):
                        content = msg.content
                        for pat in patterns:
                            content = pat.sub("[REDACTED]", content)
                        new_msgs.append(msg.__class__(content=content))
                    else:
                        new_msgs.append(msg)
                msgs = new_msgs

            # Inject runtime context into system prompt
            if inject_fn and request.runtime and request.runtime.context:
                ctx_vars = inject_fn(request.runtime.context)
                from string import Template

                new_msgs = []
                for msg in msgs:
                    if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
                        content = Template(msg.content).safe_substitute(ctx_vars)
                        new_msgs.append(SystemMessage(content=content))
                    else:
                        new_msgs.append(msg)
                msgs = new_msgs

            return await handler(request.override(messages=msgs))

        return edit_context


# ---------------------------------------------------------------------------
# Guardrail middleware (model-based)
# ---------------------------------------------------------------------------


@dataclass
class GuardrailMiddleware:
    """
    Model-based guardrails: evaluates the AI's response before returning it.
    On violation, replaces the response with a safe fallback.
    """

    fallback_message: str = "I'm unable to provide that response due to safety guidelines."
    raise_on_violation: bool = False

    def build(self):

        guardrail_chain = build_guardrail_chain()
        fallback = self.fallback_message
        raise_on = self.raise_on_violation

        @after_model
        async def check_safety(state, response):
            ai_msg = response.message
            if not isinstance(ai_msg.content, str):
                return response

            messages = state.get("messages", [])
            last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            user_input = last_human.content if last_human else ""

            try:
                result = await guardrail_chain.ainvoke(
                    {
                        "input": user_input,
                        "output": ai_msg.content,
                    }
                )
            except Exception as exc:
                logger.error("Guardrail check failed: %s", exc)
                return response

            if not result.get("safe", True):
                logger.warning(
                    "Guardrail blocked response. Reason: %s Severity: %s",
                    result.get("reason"),
                    result.get("severity"),
                )
                if raise_on:
                    raise ValueError(f"Guardrail violation: {result.get('reason')}")

                safe_response = AIMessage(content=fallback)
                return response.override(
                    message=safe_response,
                    state_update={
                        "blocked": True,
                        "block_reason": result.get("reason"),
                    },
                )

            return response

        return check_safety


# ---------------------------------------------------------------------------
# Dynamic system prompt middleware
# ---------------------------------------------------------------------------


@dataclass
class DynamicSystemPromptMiddleware:
    """
    Generates or modifies the system prompt at runtime based on context.
    """

    prompt_fn: Callable[[Any, Any], str]  # (state, context) -> system_prompt

    def build(self):
        fn = self.prompt_fn

        @before_model
        def inject_dynamic_prompt(state, request):
            ctx = request.runtime.context if request.runtime else None
            new_system = fn(state, ctx)
            msgs = list(request.messages)

            # Replace existing system message or prepend
            has_system = False
            for i, msg in enumerate(msgs):
                if isinstance(msg, SystemMessage):
                    msgs[i] = SystemMessage(content=new_system)
                    has_system = True
                    break
            if not has_system:
                msgs.insert(0, SystemMessage(content=new_system))

            return request.override(messages=msgs)

        return inject_dynamic_prompt


# ---------------------------------------------------------------------------
# Pre-built middleware stacks
# ---------------------------------------------------------------------------


def build_default_middleware_stack(
    *,
    fast_model_name: str = "gemini-2.0-flash",
    max_tokens_before_summary: int = 4000,
    messages_to_keep: int = 8,
    enable_guardrails: bool = True,
    enable_tool_selector: bool = True,
    enable_human_loop: bool = False,
    human_loop_tools: dict[str, bool] | None = None,
) -> list[Any]:
    """
    Production-ready default middleware stack.

    Order matters — middleware runs in list order for before_model,
    and reverse order for after_model.

    Stack (before_model order):
      1. Summarization     — trim context FIRST
      2. LLMToolSelector   — reduce tool list to relevant ones
      3. DynamicPrompt     — inject runtime context
      HumanInTheLoop       — pause before dangerous tools
      Guardrails           — validate response (after_model)
    """
    stack: list[Any] = []

    # 1. Summarization (context management)
    stack.append(
        SummarizationMiddleware(
            model=fast_model_name,
            max_tokens_before_summary=max_tokens_before_summary,
            messages_to_keep=messages_to_keep,
        )
    )

    # 2. Tool selector (reduces tool-call noise)
    if enable_tool_selector:
        stack.append(LLMToolSelectorMiddleware(model=fast_model_name))

    # 3. Tool retry
    stack.append(ToolRetryMiddleware(max_retries=3, backoff_factor=1.5))

    # 4. Model retry
    stack.append(ModelRetryMiddleware(max_retries=2))

    # 5. Human in the loop
    if enable_human_loop:
        stack.append(HumanInTheLoopMiddleware(interrupt_on=human_loop_tools or {}))

    # 6. Guardrails (after_model — runs last in after-model chain)
    if enable_guardrails:
        stack.append(GuardrailMiddleware())

    return stack


def build_minimal_middleware_stack() -> list[Any]:
    """Lightweight stack for development / testing."""
    return [
        SummarizationMiddleware(
            model="gemini-2.0-flash",
            max_tokens_before_summary=8000,
            messages_to_keep=10,
        ),
        ToolRetryMiddleware(max_retries=2),
    ]
