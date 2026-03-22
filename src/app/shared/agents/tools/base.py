"""
Tool base classes and registry.

All agent tools use structured Pydantic input/output for type safety.
The registry enables dynamic tool lookup and middleware (LLMToolSelector).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, TypeVar

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field, ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Structured base output
# ---------------------------------------------------------------------------


class ToolOutput(BaseModel):
    """Standard wrapper for all tool outputs."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any, **metadata: Any) -> ToolOutput:
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolOutput:
        return cls(success=False, error=error, metadata=metadata)

    def to_agent_string(self) -> str:
        """Return a string the agent can parse."""
        if self.success:
            return str(self.data)
        return f"ERROR: {self.error}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Central registry for all agent tools.
    Supports tag-based filtering (used by LLMToolSelectorMiddleware).
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._tags: dict[str, set[str]] = {}  # tool_name → set of tags

    def register(self, t: BaseTool, *, tags: list[str] | None = None) -> None:
        self._tools[t.name] = t
        self._tags[t.name] = set(tags or [])
        logger.debug("Registered tool: %s (tags=%s)", t.name, tags)

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def by_tags(self, *tags: str) -> list[BaseTool]:
        """Return tools matching ALL given tags."""
        return [
            t for name, t in self._tools.items()
            if set(tags).issubset(self._tags.get(name, set()))
        ]

    def by_names(self, names: list[str]) -> list[BaseTool]:
        return [self._tools[n] for n in names if n in self._tools]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def descriptions(self) -> dict[str, str]:
        return {name: t.description for name, t in self._tools.items()}


# Global registry
registry = ToolRegistry()

InjectedUserId = Annotated[str, InjectedToolArg]
InjectedAuthToken = Annotated[str, InjectedToolArg]


def format_tool_validation_error(
    error: ValidationError | ValidationErrorV1,
    *,
    args_schema: type[BaseModel],
) -> str:
    """Return a schema-first validation error message the model can retry against."""
    schema_json = args_schema.model_json_schema()
    return (
        "Invalid tool arguments. Retry with arguments that match this schema: "
        f"{schema_json}. Validation errors: {error}"
    )


def build_validation_error_handler(
    args_schema: type[BaseModel],
) -> Callable[[ValidationError | ValidationErrorV1], str]:
    """Create a stable validation error formatter bound to a specific schema."""
    return lambda error: format_tool_validation_error(error, args_schema=args_schema)


def register_tool(*tags: str) -> Callable[[BaseTool], BaseTool]:
    """
    Decorator to register a @tool-decorated function in the global registry.

    Usage::

        @register_tool("search", "web")
        @tool
        def web_search(query: str) -> str:
            ...
    """
    def decorator(t: BaseTool) -> BaseTool:
        registry.register(t, tags=list(tags))
        return t

    return decorator


# ---------------------------------------------------------------------------
# Convenience: build a StructuredTool from a Pydantic schema
# ---------------------------------------------------------------------------


def make_structured_tool(
    name: str,
    description: str,
    input_schema: type[BaseModel],
    fn: Any,
    *,
    tags: list[str] | None = None,
    return_direct: bool = False,
) -> StructuredTool:
    """
    Wrap an async function as a StructuredTool with typed input.
    Automatically registers it in the global registry.
    """
    t = StructuredTool.from_function(
        coroutine=fn,
        name=name,
        description=description,
        args_schema=input_schema,
        return_direct=return_direct,
        handle_tool_error=True,
        handle_validation_error=build_validation_error_handler(input_schema),
    )
    registry.register(t, tags=tags)
    return t
