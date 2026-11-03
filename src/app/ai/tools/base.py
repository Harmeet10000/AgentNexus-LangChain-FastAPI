"""
Tool base classes and registry.

All agent tools use structured Pydantic input/output for type safety.
The registry enables dynamic tool lookup and middleware (LLMToolSelector).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1

from app.kernel.models import StrictModel
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from langchain_core.tools import BaseTool

# ---------------------------------------------------------------------------
# Structured authority metadata (openspec: tool-capability-catalog)
#
# Authority is structured data, not a tag string: the enforced capability
# identifier, the side-effect class, and the egress classification. The
# ``capability:*`` / ``side-effect:*`` tags are derived views that must agree
# with these fields — disagreement is a registration error, not a silent drift.
# ---------------------------------------------------------------------------

#: Side-effect class of a tool's execution.
SideEffectClass = Literal["read", "write", "egress"]

_CAPABILITY_TAG_PREFIX = "capability:"
_SIDE_EFFECT_TAG_PREFIX = "side-effect:"


class ToolAuthority(StrictModel):
    """Structured authority declaration for one registered tool."""

    #: Capability identifier the executor enforces, if the tool is gated.
    capability: str | None
    #: Side-effect class, or ``None`` for an authority-free tool (e.g. handoff).
    side_effect: SideEffectClass | None
    #: True when model-chosen content leaves the trust boundary.
    egress: bool


def _authority_tag_mismatch(
    *,
    tags: set[str],
    authority: ToolAuthority,
) -> str | None:
    """Return a message when the authority tags disagree with the metadata.

    Tags are derived views: when structured metadata declares a capability or
    side-effect class, exactly the matching tag must be present; when it does
    not, no authority tag may be present.
    """
    expected: set[str] = set()
    if authority.capability is not None:
        expected.add(f"{_CAPABILITY_TAG_PREFIX}{authority.capability}")
    if authority.side_effect is not None:
        expected.add(f"{_SIDE_EFFECT_TAG_PREFIX}{authority.side_effect}")
    if authority.egress and authority.side_effect != "egress":
        return "an egress-classified tool must declare the 'egress' side-effect class"
    if authority.side_effect == "egress" and not authority.egress:
        return "the 'egress' side-effect class requires egress=True"

    declared = {
        tag for tag in tags if tag.startswith((_CAPABILITY_TAG_PREFIX, _SIDE_EFFECT_TAG_PREFIX))
    }
    if declared != expected:
        return (
            f"authority tags {sorted(declared)} disagree with structured metadata "
            f"(expected {sorted(expected)})"
        )
    return None


def _granted(runtime: Any, capability: str) -> bool:
    """True only when a server-owned runtime context carries ``capability``.

    Accepts the LangChain ``ToolRuntime`` (or any object exposing a Mapping
    ``context``) and reads its server-owned ``capabilities`` tuple. The grant
    can never come from model arguments — this helper never reads them.
    """
    context = getattr(runtime, "context", None)
    granted = context.get("capabilities", ()) if isinstance(context, Mapping) else ()
    if isinstance(granted, str):
        granted = (granted,)
    return isinstance(granted, (list, tuple, set, frozenset)) and capability in granted


def require_capability(runtime: Any, capability: str) -> None:
    """Raise :class:`PermissionError` unless the runtime context grants the capability.

    The refusal names the missing grant so the operator's next step is a policy
    grep, not a guessing game (the shell/filesystem refusal idiom).
    """
    if not _granted(runtime, capability):
        msg = (
            f"capability {capability!r} is not granted in the tool runtime context; "
            "grant it in server-side policy — it can never come from model arguments"
        )
        raise PermissionError(msg)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Central registry for all agent tools.
    Supports tag-based filtering (used by LLMToolSelectorMiddleware) and
    exposes structured authority metadata for policy evaluation and
    projection checks.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._tags: dict[str, set[str]] = {}  # tool_name → set of tags
        self._authority: dict[str, ToolAuthority] = {}  # tool_name → authority

    def register(
        self,
        t: BaseTool,
        *,
        tags: list[str] | None = None,
        capability: str | None = None,
        side_effect: SideEffectClass | None = None,
        egress: bool = False,
    ) -> None:
        """Register a tool with its derived tags and structured authority.

        Raises:
            ValueError: When the authority tags disagree with the structured
                metadata — the views must never drift from the declaration.
        """
        authority = ToolAuthority(capability=capability, side_effect=side_effect, egress=egress)
        mismatch = _authority_tag_mismatch(tags=set(tags or []), authority=authority)
        if mismatch is not None:
            msg = f"tool {t.name!r}: {mismatch}"
            raise ValueError(msg)
        self._tools[t.name] = t
        self._tags[t.name] = set(tags or [])
        self._authority[t.name] = authority
        logger.bind(tool_name=t.name, tags=tags).debug("Registered tool")

    def authority(self, name: str) -> ToolAuthority:
        """The structured authority metadata for one registered tool."""
        if name not in self._authority:
            msg = f"Tool not found: {name}"
            raise KeyError(msg)
        return self._authority[name]

    def declared_capabilities(self) -> set[str]:
        """Every capability identifier any registered tool enforces."""
        return {
            authority.capability
            for authority in self._authority.values()
            if authority.capability is not None
        }

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            msg = f"Tool not found: {name}"
            raise KeyError(msg)
        return self._tools[name]

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def by_tags(self, *tags: str) -> list[BaseTool]:
        """Return tools matching ALL given tags."""
        return [
            t for name, t in self._tools.items() if set(tags).issubset(self._tags.get(name, set()))
        ]

    def by_names(self, names: list[str]) -> list[BaseTool]:
        missing = [name for name in names if name not in self._tools]
        if missing:
            msg = f"Tool(s) not found: {', '.join(missing)}"
            raise KeyError(msg)
        return [self._tools[name] for name in names]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def tags(self) -> set[str]:
        """Every tag any registered tool carries."""
        return {tag for tags in self._tags.values() for tag in tags}

    def __len__(self) -> int:
        return len(self._tools)

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


def register_tool(
    *tags: str,
    capability: str | None = None,
    side_effect: SideEffectClass | None = None,
    egress: bool = False,
) -> Callable[[BaseTool], BaseTool]:
    """
    Decorator to register a @tool-decorated function in the global registry.

    Authority metadata is optional here; the explicit registrations in
    :mod:`app.ai.tools.registry` carry it for the default tool set. When both
    are given they must agree — disagreement is a registration error.

    Usage::

        @register_tool("search", "web", capability="web.search")
        @tool
        def web_search(query: str) -> str:
            ...
    """

    def decorator(t: BaseTool) -> BaseTool:
        registry.register(
            t, tags=list(tags), capability=capability, side_effect=side_effect, egress=egress
        )
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
