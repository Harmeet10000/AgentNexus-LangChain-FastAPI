"""Contracts for the LangChain 1.4 middleware API used by Agent Saul."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse, Runtime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.ai import guardrails

if TYPE_CHECKING:
    from typing import Any


def _request(
    *,
    messages: list[Any] | None = None,
    system_message: SystemMessage | None = None,
    state: dict[str, Any] | None = None,
    context: Any = None,
) -> ModelRequest[Any]:
    return ModelRequest(
        model=object(),
        messages=messages or [],
        system_message=system_message,
        state=state or {"messages": messages or []},
        runtime=Runtime(context=context),
    )


@pytest.mark.asyncio
async def test_dynamic_prompt_uses_system_message_not_request_messages() -> None:
    middleware = guardrails.DynamicSystemPromptMiddleware(
        prompt_fn=lambda _state, _context: "dynamic policy"
    ).build()
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    await middleware.awrap_model_call(_request(messages=[HumanMessage(content="hello")]), handler)

    modified_request = handler.await_args.args[0]
    assert modified_request.messages == [HumanMessage(content="hello")]
    assert modified_request.system_message == SystemMessage(content="dynamic policy")


@pytest.mark.asyncio
async def test_context_editing_redacts_user_messages_and_interpolates_system_prompt() -> None:
    middleware = guardrails.ContextEditingMiddleware(
        redact_patterns=[r"secret-\d+"],
        inject_context_fn=lambda context: {"tenant": context["tenant"]},
    ).build()
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    request = _request(
        messages=[HumanMessage(content="secret-123")],
        system_message=SystemMessage(content="Tenant ${tenant}"),
        context={"tenant": "acme"},
    )
    await middleware.awrap_model_call(request, handler)

    modified_request = handler.await_args.args[0]
    assert modified_request.messages[0].content == "[REDACTED]"
    assert modified_request.system_message == SystemMessage(content="Tenant acme")


def test_todo_after_model_hook_reads_agent_state_and_returns_state_update() -> None:
    _, after_model = guardrails.TodoListMiddleware().build()

    update = after_model.after_model(
        {
            "messages": [AIMessage(content="[TODO:ADD] Verify migration")],
            "todo_list": [],
        },
        Runtime(context={}),
    )

    assert update == {"todo_list": ["Verify migration"]}


def test_default_stack_uses_current_summarization_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_calls: list[dict[str, Any]] = []

    class FakeSummarization:
        def __init__(self, **kwargs: Any) -> None:
            summary_calls.append(kwargs)

    class FakeSelector:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(guardrails, "SummarizationMiddleware", FakeSummarization)
    monkeypatch.setattr(guardrails, "LLMToolSelectorMiddleware", FakeSelector)

    guardrails.build_default_middleware_stack(
        config=guardrails.MiddlewareConfig(
            fast_model_name="gemini-2.0-flash",
            max_tokens_before_summary=1234,
            messages_to_keep=6,
        ),
        enable_guardrails=False,
        enable_tool_selector=True,
    )

    assert summary_calls == [
        {
            "model": "google_genai:gemini-2.0-flash",
            "trigger": ("tokens", 1234),
            "keep": ("messages", 6),
        }
    ]


@pytest.mark.asyncio
async def test_guardrail_replaces_unsafe_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = AsyncMock()
    chain.ainvoke.return_value = {"safe": False, "reason": "unsafe"}
    monkeypatch.setattr(
        "app.shared.langchain_layer.chains.build_guardrail_chain",
        lambda: chain,
    )
    middleware = guardrails.GuardrailMiddleware(fallback_message="blocked").build()
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="unsafe output")]))

    response = await middleware.awrap_model_call(
        _request(messages=[HumanMessage(content="request")]),
        handler,
    )

    assert response.result == [AIMessage(content="blocked")]
    chain.ainvoke.assert_awaited_once_with({"input": "request", "output": "unsafe output"})
