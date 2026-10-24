"""Contracts for the LangChain 1.4 middleware API used by Agent Saul."""

from __future__ import annotations

import inspect
from importlib.metadata import version
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, Runtime, ToolCallRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.ai import guardrails

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.messages import ToolMessage


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


@pytest.mark.asyncio
async def test_todo_state_persists_across_create_agent_calls() -> None:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="[TODO:ADD] Verify migration"),
                AIMessage(content="Migration verified"),
            ]
        )
    )
    agent = create_agent(
        model,
        middleware=guardrails.TodoListMiddleware().build(),
        state_schema=guardrails.TodoAgentState,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "todo-contract"}}

    first = await agent.ainvoke({"messages": [("user", "start")]}, config)
    second = await agent.ainvoke({"messages": [("user", "continue")]}, config)

    assert first["todo_list"] == ["Verify migration"]
    assert second["todo_list"] == ["Verify migration"]


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
            fast_model_name="gemini-3.6-flash",
            max_tokens_before_summary=1234,
            messages_to_keep=6,
        ),
        enable_guardrails=False,
        enable_tool_selector=True,
    )

    assert summary_calls == [
        {
            "model": "google_genai:gemini-3.6-flash",
            "trigger": ("tokens", 1234),
            "keep": ("messages", 6),
        }
    ]


def test_default_stack_does_not_install_broad_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guardrails, "SummarizationMiddleware", lambda **_kwargs: object())
    stack = guardrails.build_default_middleware_stack(
        enable_guardrails=False,
        enable_tool_selector=False,
    )

    assert not any(isinstance(item, ToolRetryMiddleware) for item in stack)
    assert not any(isinstance(item, ModelRetryMiddleware) for item in stack)


def test_retry_policy_is_allowlisted_and_uses_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guardrails, "SummarizationMiddleware", lambda **_kwargs: object())
    stack = guardrails.build_default_middleware_stack(
        config=guardrails.MiddlewareConfig(safe_tool_names=("safe_lookup",)),
        enable_guardrails=False,
        enable_tool_selector=False,
    )

    retry = next(item for item in stack if isinstance(item, ToolRetryMiddleware))
    assert retry._tool_filter == ["safe_lookup"]
    assert retry.on_failure == "error"
    assert retry.retry_on(ValueError("permanent")) is False
    assert retry.retry_on(TimeoutError("temporary")) is True


@pytest.mark.asyncio
async def test_unlisted_side_effect_tool_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guardrails, "SummarizationMiddleware", lambda **_kwargs: object())
    stack = guardrails.build_default_middleware_stack(
        config=guardrails.MiddlewareConfig(safe_tool_names=("safe_lookup",)),
        enable_guardrails=False,
        enable_tool_selector=False,
    )
    retry = next(item for item in stack if isinstance(item, ToolRetryMiddleware))
    request = ToolCallRequest(
        tool_call={"name": "delete_record", "args": {}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,
    )
    attempts = 0

    async def handler(_request: ToolCallRequest) -> ToolMessage:
        nonlocal attempts
        attempts += 1
        error_message = "temporary backend outage"
        raise TimeoutError(error_message)

    with pytest.raises(TimeoutError):
        await retry.awrap_tool_call(request, handler)

    assert attempts == 1


@pytest.mark.asyncio
async def test_model_retry_does_not_retry_non_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guardrails, "SummarizationMiddleware", lambda **_kwargs: object())
    stack = guardrails.build_default_middleware_stack(
        config=guardrails.MiddlewareConfig(enable_model_retry=True),
        enable_guardrails=False,
        enable_tool_selector=False,
    )
    retry = next(item for item in stack if isinstance(item, ModelRetryMiddleware))
    attempts = 0

    async def handler(_request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal attempts
        attempts += 1
        error_message = "invalid model request"
        raise ValueError(error_message)

    with pytest.raises(ValueError):
        await retry.awrap_model_call(_request(), handler)

    assert attempts == 1


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


def test_async_postgres_saver_320_contract_without_database() -> None:
    """Pin the saver API consumed by the application's pool-owning adapter."""
    from app.connections.checkpointer import setup_langgraph_checkpointer

    assert tuple(map(int, version("langgraph-checkpoint-postgres").split(".")[:2])) >= (3, 2)
    assert inspect.iscoroutinefunction(AsyncPostgresSaver.setup)
    assert inspect.isasyncgenfunction(AsyncPostgresSaver.from_conn_string.__wrapped__)

    saver_parameters = inspect.signature(AsyncPostgresSaver).parameters
    assert tuple(saver_parameters)[:2] == ("conn", "pipe")

    setup_parameters = inspect.signature(setup_langgraph_checkpointer).parameters
    assert tuple(setup_parameters) == ("conn_string",)
    assert setup_parameters["conn_string"].default is None
