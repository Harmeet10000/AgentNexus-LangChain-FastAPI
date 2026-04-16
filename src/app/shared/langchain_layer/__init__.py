"""
Production middleware for LangChain 1.0 agents.

All middleware is composable and works with create_agent's `middleware` parameter.

Built on LangChain 1.0 hooks:
- @before_model  → runs before LLM, can mutate state
- @after_model   → runs after LLM, can redirect or mutate
- @wrap_model_call → wraps the entire LLM call (request + response)
- @wrap_tool_call  → wraps individual tool execution

Usage::

    agent = create_agent(
        model,
        tools=tools,
        middleware=[
            SummarizationMiddleware(model=fast_llm, max_tokens_before_summary=4000),
            HumanInTheLoopMiddleware(interrupt_on={"delete_file": True}),
            LLMToolSelectorMiddleware(model=fast_llm),
            ToolRetryMiddleware(max_retries=3),
            ModelRetryMiddleware(max_retries=2),
            GuardrailMiddleware(),
        ],
    )
"""

from .callback import LatencyCallbackHandler, TokenUsageCallbackHandler, configure_langsmith

# ---------------------------------------------------------------------------
# Re-export built-in middleware for convenience
# ---------------------------------------------------------------------------

__all__ = [
    "ContextEditingMiddleware",
    "DynamicSystemPromptMiddleware",
    "GuardrailMiddleware",
    "HumanInTheLoopMiddleware",
    "LLMToolSelectorMiddleware",
    # "LLMToolEmulatorMiddleware",
    # Custom below:
    "ModelRetryMiddleware",
    "SummarizationMiddleware",
    "TodoListMiddleware",
    "ToolRetryMiddleware",
    "build_default_middleware_stack",
    "build_minimal_middleware_stack",
    "configure_langsmith",
]

