from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, override

logger = logging.getLogger(__name__)

from langchain.agents.middleware import (  # noqa: E402
    HumanInTheLoopMiddleware,
    LLMToolSelectorMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
    after_model,
    before_model,
    wrap_model_call,
)
from langchain_core.exceptions import LangChainException  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from app.shared.langchain_layer.chains import build_guardrail_chain  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# ---------------------------------------------------------------------------
# Model retry middleware
# ---------------------------------------------------------------------------


class ModelRetryMiddleware(BaseModel):
    """
    Retries the LLM call on failure with exponential back-off.

    Built as a @wrap_model_call: intercepts the handler,
    retries up to max_retries on exception.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_retries: int = 2
    base_delay: float = 1.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)

    @override
    def model_post_init(self, __context: object) -> None:
        @wrap_model_call  # type: ignore
        async def _retry_wrapper(request: object, handler: object) -> Any:
            for attempt in range(self.max_retries + 1):
                try:
                    return await handler(request)  # type: ignore
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
            return None

        self._middleware = _retry_wrapper

    def __call__(self, *args, **kwargs):
        return self._middleware(*args, **kwargs)


# ---------------------------------------------------------------------------
# TODO: list middleware  # noqa: FIX002
# ---------------------------------------------------------------------------


class TodoListMiddleware(BaseModel):
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

    def build(self) -> list[Any]:
        @before_model  # type: ignore
        def inject_todos(state, request) -> Any:
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

        @after_model  # type: ignore
        def parse_todo_commands(state, response) -> Any:
            ai_msg = response.message
            if not isinstance(ai_msg.content, str):
                return response

            todos = list(state.get("todo_list", []))
            content = ai_msg.content

            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("[TODO:ADD]"):
                    item = stripped[len("[TODO:ADD]") :].strip()
                    if item and item not in todos:
                        todos.append(item)
                elif stripped.startswith(("[TODO:DONE]", "[TODO:REMOVE]")):
                    prefix = "[TODO:DONE]" if "[TODO:DONE]" in stripped else "[TODO:REMOVE]"
                    item = stripped[len(prefix) :].strip()
                    todos = [t for t in todos if t != item]

            return response.override(state_update={"todo_list": todos})

        return [inject_todos, parse_todo_commands]


# ---------------------------------------------------------------------------
# Context editing middleware
# ---------------------------------------------------------------------------


class ContextEditingMiddleware(BaseModel):
    """
    Allows runtime editing of context: inject variables, redact PII,
    or transform messages before model sees them.
    """

    redact_patterns: list[str] = Field(default_factory=list)  # regex patterns
    inject_context_fn: Callable[[Any], dict[str, str]] | None = None

    def build(self) -> Any:
        import re

        patterns = [re.compile(p) for p in self.redact_patterns]
        inject_fn = self.inject_context_fn

        @wrap_model_call  # type: ignore
        async def edit_context(request: object, handler: object) -> Any:
            msgs = list(request.messages)  # type: ignore

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

            return await handler(request.override(messages=msgs))  # type: ignore

        return edit_context


# ---------------------------------------------------------------------------
# Guardrail middleware (model-based)
# ---------------------------------------------------------------------------


class GuardrailMiddleware(BaseModel):
    """
    Model-based guardrails: evaluates the AI's response before returning it.
    On violation, replaces the response with a safe fallback.
    """

    fallback_message: str = "I'm unable to provide that response due to safety guidelines."
    raise_on_violation: bool = False

    def build(self) -> Any:

        guardrail_chain = build_guardrail_chain()
        fallback = self.fallback_message
        raise_on = self.raise_on_violation

        @after_model  # type: ignore
        async def check_safety(state: object, response: object) -> Any:
            ai_msg = response.message  # type: ignore
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
            except LangChainException as exc:
                exc.add_note("operation=guardrail_check")
                logger.exception("Guardrail check failed: %s")
                return response

            if not result.get("safe", True):
                logger.warning(
                    "Guardrail blocked response. Reason: %s Severity: %s",
                    result.get("reason"),
                    result.get("severity"),
                )
                if raise_on:
                    msg = f"Guardrail violation: {result.get('reason')}"
                    raise ValueError(msg)

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


class DynamicSystemPromptMiddleware(BaseModel):
    """
    Generates or modifies the system prompt at runtime based on context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_fn: Callable[[Any, Any], str]  # (state, context) -> system_prompt

    def build(self) -> Any:
        fn = self.prompt_fn

        @before_model  # type: ignore
        def inject_dynamic_prompt(state, request) -> Any:
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

    # 3. Tool retry / 4. Model retry
    stack.extend(
        [
            ToolRetryMiddleware(max_retries=3, backoff_factor=1.5),
            ModelRetryMiddleware(max_retries=2),
        ]
    )

    # 5. Human in the loop
    if enable_human_loop:
        stack.append(HumanInTheLoopMiddleware(interrupt_on=human_loop_tools or {}))  # type: ignore

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
