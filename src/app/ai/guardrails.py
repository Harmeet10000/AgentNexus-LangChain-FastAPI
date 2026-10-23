from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    LLMToolSelectorMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
    after_model,
    wrap_model_call,
)
from langchain.agents.middleware import ModelRetryMiddleware as LangChainModelRetryMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.exceptions import LangChainException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable

# ---------------------------------------------------------------------------
# Model retry middleware
# ---------------------------------------------------------------------------


class ModelRetryMiddleware(BaseModel):
    """
    Legacy local retry adapter retained for public-import compatibility.

    The production stack uses LangChain's prebuilt
    ``langchain.agents.middleware.ModelRetryMiddleware`` instead. New callers
    should prefer that implementation so retry failure semantics stay aligned
    with the agent runtime; this adapter remains available to avoid breaking
    existing imports.

    Built as a @wrap_model_call: intercepts the handler,
    retries up to max_retries on exception.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_retries: int = 2
    base_delay: float = 1.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)

    def build(self) -> Any:
        @wrap_model_call
        async def _retry_wrapper(
            request: ModelRequest[Any],
            handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
        ) -> ModelResponse[Any]:
            for attempt in range(self.max_retries + 1):
                try:
                    return await handler(request)
                except self.retryable_exceptions as exc:
                    if attempt == self.max_retries:
                        raise
                    delay = self.base_delay * (2**attempt)
                    logger.bind(
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        delay=delay,
                        error=str(exc),
                    ).warning("Model call failed, retrying")
                    await asyncio.sleep(delay)
            msg = "Model retry loop exhausted without returning"
            raise RuntimeError(msg)

        return _retry_wrapper


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
        @wrap_model_call
        async def inject_todos(
            request: ModelRequest[Any],
            handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
        ) -> ModelResponse[Any]:
            todos = request.state.get("todo_list", [])
            if not todos:
                return await handler(request)

            todo_text = "\n".join(f"- [ ] {t}" for t in todos)
            todo_block = f"\n\n{self.todo_header}\n{todo_text}"

            system_message = request.system_message
            if system_message is not None:
                system_message = SystemMessage(content=f"{system_message.content}{todo_block}")
            else:
                system_message = SystemMessage(content=todo_block)

            return await handler(request.override(system_message=system_message))

        @after_model
        def parse_todo_commands(state: dict[str, Any], _runtime: Any) -> dict[str, Any] | None:
            messages = state.get("messages", [])
            ai_msg = next(
                (message for message in reversed(messages) if isinstance(message, AIMessage)),
                None,
            )
            if ai_msg is None or not isinstance(ai_msg.content, str):
                return None

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

            return {"todo_list": todos}

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

        @wrap_model_call
        async def edit_context(
            request: ModelRequest[Any],
            handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
        ) -> ModelResponse[Any]:
            msgs = list(request.messages)

            # Redact PII
            if patterns:
                new_msgs = []
                for msg in msgs:
                    if isinstance(msg.content, str):
                        content = msg.content
                        for pat in patterns:
                            content = pat.sub("[REDACTED]", content)
                        new_msgs.append(msg.model_copy(update={"content": content}))
                    else:
                        new_msgs.append(msg)
                msgs = new_msgs

            system_message = request.system_message
            # Inject runtime context into system prompt
            if inject_fn and request.runtime and request.runtime.context:
                ctx_vars = inject_fn(request.runtime.context)
                from string import Template

                if system_message is not None and isinstance(system_message.content, str):
                    content = Template(system_message.content).safe_substitute(ctx_vars)
                    system_message = SystemMessage(content=content)

            return await handler(request.override(messages=msgs, system_message=system_message))

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
        # Keep this import local: importing the chains submodule imports the
        # langchain_layer package, whose public exports include this module.
        from app.shared.langchain_layer.chains import build_guardrail_chain

        guardrail_chain = build_guardrail_chain()
        fallback = self.fallback_message
        raise_on = self.raise_on_violation

        @wrap_model_call
        async def check_safety(
            request: ModelRequest[Any],
            handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
        ) -> ModelResponse[Any]:
            response = await handler(request)
            ai_msg = next(
                (
                    message
                    for message in reversed(response.result)
                    if isinstance(message, AIMessage)
                ),
                None,
            )
            if ai_msg is None or not isinstance(ai_msg.content, str):
                return response

            messages = request.messages
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
                logger.bind(operation="guardrail_check", error=str(exc)).exception(
                    "Guardrail check failed"
                )
                return response

            if not result.get("safe", True):
                logger.bind(reason=result.get("reason"), severity=result.get("severity")).warning(
                    "Guardrail blocked response"
                )
                if raise_on:
                    msg = f"Guardrail violation: {result.get('reason')}"
                    raise ValueError(msg)

                safe_response = AIMessage(content=fallback)
                return ModelResponse(
                    result=[safe_response],
                    structured_response=response.structured_response,
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

        @wrap_model_call
        async def inject_dynamic_prompt(
            request: ModelRequest[Any],
            handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
        ) -> ModelResponse[Any]:
            ctx = request.runtime.context if request.runtime else None
            new_system = fn(request.state, ctx)
            return await handler(request.override(system_message=SystemMessage(content=new_system)))

        return inject_dynamic_prompt


# ---------------------------------------------------------------------------
# Pre-built middleware stacks
# ---------------------------------------------------------------------------


class MiddlewareConfig(BaseModel):
    """Scalar config for the default middleware stack (frozen read model)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fast_model_name: str = "google_genai:gemini-2.0-flash"
    max_tokens_before_summary: int = 4000
    messages_to_keep: int = 8
    human_loop_tools: dict[str, bool] | None = None
    # Retrying a tool is safe only when the caller has explicitly classified it as
    # idempotent.  An empty allowlist means no tool retry middleware is installed.
    safe_tool_names: tuple[str, ...] = ()
    # Provider SDKs expose incompatible exception hierarchies.  Keep model retry
    # disabled unless a caller opts in, and let that caller supply a predicate when
    # its provider has a stronger transient-error contract than our portable default.
    enable_model_retry: bool = False
    model_retry_on: Callable[[Exception], bool] | None = None


def build_default_middleware_stack(
    *,
    config: MiddlewareConfig | None = None,
    enable_guardrails: bool = True,
    enable_tool_selector: bool = True,
    enable_human_loop: bool = False,
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
    cfg = config or MiddlewareConfig()
    fast_model_name = _provider_model_name(cfg.fast_model_name)

    # 1. Summarization (context management)
    stack.append(
        SummarizationMiddleware(
            model=fast_model_name,
            trigger=("tokens", cfg.max_tokens_before_summary),
            keep=("messages", cfg.messages_to_keep),
        )
    )

    # 2. Tool selector (reduces tool-call noise)
    if enable_tool_selector:
        stack.append(LLMToolSelectorMiddleware(model=fast_model_name))

    # 3. Tool retry is opt-in and allowlisted.  The middleware's default filter is
    # "all tools", which is unsafe for writes, sends, deletes, and other side effects.
    if cfg.safe_tool_names:
        stack.append(
            ToolRetryMiddleware(
                tools=list(cfg.safe_tool_names),
                retry_on=_is_transient_error,
                max_retries=2,
                backoff_factor=1.5,
                on_failure="error",
            )
        )

    # 4. Model retry is also opt-in.  The portable predicate intentionally covers only
    # built-in transport failures; provider-specific callers can provide a narrower,
    # provider-aware predicate through `model_retry_on`.
    if cfg.enable_model_retry:
        stack.append(
            LangChainModelRetryMiddleware(
                max_retries=2,
                retry_on=cfg.model_retry_on or _is_transient_error,
                on_failure="error",
            )
        )

    # 5. Human in the loop
    if enable_human_loop:
        stack.append(HumanInTheLoopMiddleware(interrupt_on=cfg.human_loop_tools or {}))  # ty: ignore[invalid-argument-type]

    # 6. Guardrails (after_model — runs last in after-model chain)
    if enable_guardrails:
        stack.append(GuardrailMiddleware().build())

    return stack


def build_minimal_middleware_stack() -> list[Any]:
    """Lightweight stack for development / testing."""
    return [
        SummarizationMiddleware(
            model="google_genai:gemini-2.0-flash",
            trigger=("tokens", 8000),
            keep=("messages", 10),
        ),
    ]


def _provider_model_name(model_name: str) -> str:
    """Make Gemini middleware model names explicit about their provider."""
    if ":" in model_name:
        return model_name
    if model_name.startswith("gemini-"):
        return f"google_genai:{model_name}"
    return model_name


def _is_transient_error(error: Exception) -> bool:
    """Return whether a provider-neutral transport error is safe to retry.

    Deliberately do not treat arbitrary exceptions as transient: validation errors,
    permission failures, malformed requests, and tool-side bugs are not repaired by
    repeating the operation. Provider integrations may pass a stricter predicate via
    :class:`MiddlewareConfig` when their SDK exposes a reliable transient hierarchy.
    """
    return isinstance(error, (ConnectionError, TimeoutError))
