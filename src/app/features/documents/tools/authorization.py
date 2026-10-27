"""Trusted runtime scope for document evidence tools.

LangChain injects ``ToolRuntime`` and excludes it from the model-facing tool
schema.  The model can choose a query, but it cannot choose the principal,
tenant, thread, or document allowlist used to execute that query.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg


class ToolRuntimeContext(TypedDict, total=False):
    """Server-owned context propagated into Saul's nested tool agents.

    ``create_agent`` builds a separate LangGraph subgraph for each specialist.
    Its state contains only the specialist conversation, so the outer Saul
    state cannot be relied on for identity or document scope.  This narrow
    context schema is the explicit bridge for ``ToolRuntime.context``.
    """

    user_id: str
    tenant_id: str
    allowed_doc_ids: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    thread_id: str
    step_id: str
    run_id: str
    checkpoint_id: str
    document_versions: tuple[tuple[str, int, str | None], ...]
    # Server-side capability grants (PLAN-agent-tools.md §3). Default deny:
    # a system/filesystem tool executes only when its capability appears here.
    capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolAuthorizationContext:
    user_id: str
    tenant_id: str
    allowed_doc_ids: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    thread_id: str
    step_id: str
    run_id: str
    checkpoint_id: str
    document_versions: tuple[tuple[str, int, str | None], ...]
    capabilities: tuple[str, ...]
    # True only for callers that explicitly use the pre-runtime compatibility
    # arguments.  This keeps the old test/client seam isolated; production
    # graph tools must always arrive with a server-created ToolRuntime.
    legacy_compat: bool = False

    def require_document(self, document_id: str) -> None:
        if document_id not in self.allowed_doc_ids:
            message = "document is outside the authorized analysis scope"
            raise PermissionError(message)

    def require_operation(self, operation: str) -> None:
        # ``legacy_compat`` is only set by the explicit pre-runtime adapter
        # arguments.  Production ToolRuntime calls must carry the server grant
        # and therefore fail closed when it is absent.
        if not self.legacy_compat and operation not in self.allowed_operations:
            message = "operation is outside the authorized capability"
            raise PermissionError(message)


def trusted_tool_context(  # noqa: PLR0914 — normalize the complete runtime context
    runtime: Annotated[ToolRuntime | None, InjectedToolArg],
    *,
    legacy_user_id: str | None = None,
    legacy_doc_id: str | None = None,
    legacy_thread_id: str | None = None,
    legacy_step_id: str | None = None,
) -> ToolAuthorizationContext:
    """Build scope from server-created runtime state/context only."""
    if runtime is None:
        # Compatibility is deliberately explicit and narrow.  No legacy call
        # can synthesize a tenant or an allowlist; the document-id argument is
        # retained only for old callers that already supplied one.  New tool
        # invocations must use ToolRuntime and are denied without it.
        if not legacy_user_id:
            message = "server-created tool authorization context is required"
            raise PermissionError(message)
        allowed = (legacy_doc_id,) if legacy_doc_id else ()
        return ToolAuthorizationContext(
            user_id=legacy_user_id,
            tenant_id=legacy_user_id,
            allowed_doc_ids=allowed,
            allowed_operations=(),
            thread_id=legacy_thread_id or "",
            step_id=legacy_step_id or "tool",
            run_id=legacy_thread_id or "",
            checkpoint_id="",
            document_versions=(),
            capabilities=(),
            legacy_compat=True,
        )

    state = runtime.state if isinstance(runtime.state, Mapping) else {}
    context = runtime.context if isinstance(runtime.context, Mapping) else {}
    user_id = _first_non_empty(context.get("user_id"), state.get("user_id"))
    if user_id is None:
        message = "authenticated user is missing from tool runtime"
        raise PermissionError(message)
    tenant_id = _first_non_empty(context.get("tenant_id"), state.get("tenant_id"))
    if tenant_id is None:
        message = "tenant is missing from server-created tool authorization context"
        raise PermissionError(message)
    configured_docs = context.get("allowed_doc_ids", state.get("allowed_doc_ids", ()))
    if isinstance(configured_docs, str):
        allowed_doc_ids = (configured_docs,)
    elif isinstance(configured_docs, (list, tuple, set, frozenset)):
        allowed_doc_ids = tuple(str(value) for value in configured_docs)
    else:
        allowed_doc_ids = ()
    thread_id = _first_non_empty(context.get("thread_id"), state.get("thread_id")) or ""
    step_id = _first_non_empty(context.get("step_id"), state.get("current_step")) or "tool"
    run_id = _first_non_empty(context.get("run_id"), state.get("run_id")) or thread_id
    checkpoint_id = _first_non_empty(context.get("checkpoint_id"), state.get("checkpoint_id")) or ""
    raw_versions = context.get("document_versions", state.get("document_versions", ()))
    document_versions = (
        tuple(
            (str(item[0]), int(item[1]), str(item[2]) if len(item) > 2 else None)
            for item in raw_versions
            if isinstance(item, (list, tuple)) and len(item) >= 2
        )
        if isinstance(raw_versions, (list, tuple))
        else ()
    )
    # Identity and capability authority is carried by ToolRuntime.context.
    # ``runtime.state`` is model-controlled specialist state and therefore is
    # never a fallback for these grants: omission means deny, not inheritance.
    allowed_operations = _as_string_tuple(context.get("allowed_operations", ()))
    capabilities = _as_string_tuple(context.get("capabilities", ()))
    return ToolAuthorizationContext(
        user_id=user_id,
        tenant_id=tenant_id,
        allowed_doc_ids=allowed_doc_ids,
        allowed_operations=allowed_operations,
        thread_id=thread_id,
        step_id=str(step_id),
        run_id=str(run_id),
        checkpoint_id=str(checkpoint_id),
        document_versions=document_versions,
        capabilities=capabilities,
    )


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int):
            return str(value)
    return None


def _as_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if str(item))
    return ()
