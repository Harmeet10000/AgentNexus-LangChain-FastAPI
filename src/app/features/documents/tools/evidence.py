"""LangChain adapters for the scoped V1 evidence contract.

The factories keep identity and document scope out of the model-facing schema;
the Agent Saul bundle may bind them to specialist sub-agents at graph build
time when an EvidenceService is provisioned.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool

from app.features.documents.dto import EvidenceSearchRequest

from .authorization import trusted_tool_context

if TYPE_CHECKING:
    from langchain_core.tools.base import BaseTool

    from app.features.documents.service.evidence import EvidenceService


def make_search_evidence_tool(service: EvidenceService) -> BaseTool:
    @tool
    async def search_evidence(
        query: str,
        *,
        limit: int = 5,
        jurisdiction: str | None = None,
        contract_type: str | None = None,
        clause_type: str | None = None,
        document_kind: str | None = None,
        require_graph_indexed: bool = False,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> dict[str, object]:
        """Find scoped evidence and return bounded snippets plus handles."""
        authz = trusted_tool_context(runtime)
        authz.require_operation("evidence.search")
        run_id, checkpoint_id = _runtime_scopes(runtime, authz.thread_id)
        result = await service.search_evidence(
            principal_id=authz.user_id,
            tenant_id=authz.tenant_id,
            allowed_doc_ids=authz.allowed_doc_ids,
            thread_id=authz.thread_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            request=EvidenceSearchRequest(
                query=query,
                limit=limit,
                jurisdiction=jurisdiction,
                contract_type=contract_type,
                clause_type=clause_type,
                document_kind=document_kind,
                require_graph_indexed=require_graph_indexed,
            ),
        )
        return result.model_dump(mode="json")

    return search_evidence


def make_read_evidence_tool(service: EvidenceService) -> BaseTool:
    @tool
    async def read_evidence_by_id(
        evidence_id: str,
        *,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> dict[str, object]:
        """Read one exact source chunk after reauthorizing its handle."""
        authz = trusted_tool_context(runtime)
        authz.require_operation("evidence.read")
        run_id, checkpoint_id = _runtime_scopes(runtime, authz.thread_id)
        result = await service.read_evidence_by_id(
            evidence_id=evidence_id,
            principal_id=authz.user_id,
            tenant_id=authz.tenant_id,
            allowed_doc_ids=authz.allowed_doc_ids,
            thread_id=authz.thread_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
        )
        return result.model_dump(mode="json")

    return read_evidence_by_id


def make_get_ingestion_state_tool(service: EvidenceService) -> BaseTool:
    @tool
    async def get_ingestion_state(
        *,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> dict[str, object]:
        """Report readiness for every document in the server-owned scope."""
        authz = trusted_tool_context(runtime)
        authz.require_operation("ingestion.status")
        run_id, checkpoint_id = _runtime_scopes(runtime, authz.thread_id)
        result = await service.get_ingestion_state(
            principal_id=authz.user_id,
            tenant_id=authz.tenant_id,
            allowed_doc_ids=authz.allowed_doc_ids,
            thread_id=authz.thread_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
        )
        return result.model_dump(mode="json")

    return get_ingestion_state


def make_evidence_tools(service: EvidenceService) -> tuple[BaseTool, BaseTool, BaseTool]:
    """Build the V1 search/read/readiness tools without binding them to Saul."""
    return (
        make_search_evidence_tool(service),
        make_read_evidence_tool(service),
        make_get_ingestion_state_tool(service),
    )


def _runtime_scopes(runtime: ToolRuntime | None, fallback_thread_id: str) -> tuple[str, str | None]:
    if runtime is None:
        return fallback_thread_id, None
    state = runtime.state if isinstance(runtime.state, Mapping) else {}
    context = runtime.context if isinstance(runtime.context, Mapping) else {}
    run_id = _first_string(context.get("run_id"), state.get("run_id")) or fallback_thread_id
    # Checkpoint authority comes only from the server runtime context.  A
    # serialized graph-state value is evidence metadata, not an authorization
    # grant; when unavailable use the run-bound marker.
    checkpoint_id = _first_string(context.get("checkpoint_id"))
    # Bind every capability even when the LangGraph runtime does not expose a
    # concrete checkpoint in ToolRuntime.  This marker is run-scoped and cannot
    # be supplied by model arguments.
    checkpoint_id = checkpoint_id or f"run:{run_id}"
    return run_id, checkpoint_id


def _first_string(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None
