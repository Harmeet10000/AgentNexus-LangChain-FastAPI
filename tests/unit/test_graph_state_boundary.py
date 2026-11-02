"""State key-set boundary contract for every compiled graph.

``TypedDict(total=False)`` makes every channel optional, and LangGraph maps an
invocation's input onto declared channels while discarding the rest — so a
state dict whose keys are disjoint from the graph's declaration type-checks
clean and silently loses the caller's data. This suite is the seatbelt: it
compares the key set a caller actually places in an invocation's initial state
against the graph state's annotated keys, per compiled graph, from one registry.

The registry is deliberately exhaustive rather than convenient:
``test_every_compile_site_is_registered`` scans ``src/app`` for ``.compile()``
call sites and fails when one is missing from this file, so a new compiled graph
is a visible gap instead of an untested edge.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, get_type_hints

import pytest

from app.features.agent_saul.domain.state import LegalAgentState
from app.features.agent_saul.graph.deep_research import DeepResearchState
from app.features.documents.ingestion.legacy_service import map_legacy_state
from app.features.documents.ingestion.workflow import DocumentIngestionState
from app.features.documents.retrieval.graph.state import RetrievalState
from app.shared.langgraph_layer.open_deep_search.state import (
    AgentState,
    ResearcherState,
    SupervisorState,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src" / "app"
_COMPILE_RE = re.compile(r"\.compile\(")

#: Agent Saul's per-turn initial state keys, from
#: ``AgentSaulService._build_initial_state``. The graph compiles with the full
#: ``LegalAgentState`` as its input schema, so every one of these must be a
#: declared channel or LangGraph drops it before the first node.
_SAUL_INITIAL_STATE_KEYS = frozenset(
    {
        "user_id",
        "thread_id",
        "correlation_id",
        "evidence_checkpoint_id",
        "schema_version",
        "doc_id",
        "primary_doc_id",
        "allowed_doc_ids",
        "tenant_id",
        "legal_roles",
        "allowed_operations",
        "capabilities",
        "document_versions",
        "document_contracts",
        "self_review_allowed",
        "self_reviewed",
        "document_text",
        "ingestion_state",
        "messages",
        "user_query",
        "qna_confidence",
        "plan",
        "current_step",
        "plan_approved",
        "plan_rejection_feedback",
        "orchestrator_action",
        "normalized_document",
        "segments",
        "extracted_entities",
        "relationships",
        "risk_analysis",
        "compliance_result",
        "grounding",
        "human_review",
        "final_report",
        "long_term_refs",
        "working_memory",
        "status",
        "errors",
        "retry_count",
        "permissions",
        "budget_ledger",
    }
)

#: The legacy compatibility path's historical state shape. Every one of these
#: keys is undeclared by ``DocumentIngestionState`` — the original silent drop.
HISTORICAL_LEGACY_STATE_KEYS = frozenset(
    {"doc_id", "raw_bytes", "thread_id", "source", "document_type", "jurisdiction"}
)


class GraphStateContract(NamedTuple):
    """One compiled graph's input channel and the state it maps onto."""

    name: str
    #: Repo-relative path of the module that compiles the graph.
    module_path: str
    state: type
    #: Keys the application places in an invocation's initial state.
    initial_state_keys: frozenset[str]


GRAPH_STATE_CONTRACTS: tuple[GraphStateContract, ...] = (
    GraphStateContract(
        name="documents.ingestion",
        module_path="src/app/features/documents/ingestion/workflow.py",
        state=DocumentIngestionState,
        initial_state_keys=frozenset(
            {
                "document_id",
                "user_id",
                "tenant_id",
                "filename",
                "content_type",
                "object_uri",
                "document_version",
                "attempt",
                "task_id",
            }
        ),
    ),
    GraphStateContract(
        name="documents.retrieval",
        module_path="src/app/features/documents/retrieval/graph/graph.py",
        state=RetrievalState,
        initial_state_keys=frozenset(
            {
                "messages",
                "user_id",
                "tenant_id",
                "allowed_doc_ids",
                "document_versions",
                "document_contracts",
                "scope_unavailable_documents",
                "scope_bound",
                "query",
                "doc_ids_filter",
                "contract_index",
                "iteration_count",
            }
        ),
    ),
    GraphStateContract(
        name="open_deep_search.agent",
        module_path="src/app/shared/langgraph_layer/open_deep_search/graph.py",
        state=AgentState,
        initial_state_keys=frozenset({"messages"}),
    ),
    GraphStateContract(
        name="open_deep_search.supervisor",
        module_path="src/app/shared/langgraph_layer/open_deep_search/graph.py",
        state=SupervisorState,
        initial_state_keys=frozenset({"supervisor_messages", "research_brief"}),
    ),
    GraphStateContract(
        name="open_deep_search.researcher",
        module_path="src/app/shared/langgraph_layer/open_deep_search/graph.py",
        state=ResearcherState,
        initial_state_keys=frozenset({"researcher_messages", "research_topic"}),
    ),
    GraphStateContract(
        name="agent_saul.graph",
        module_path="src/app/features/agent_saul/graph/graph.py",
        state=LegalAgentState,
        initial_state_keys=_SAUL_INITIAL_STATE_KEYS,
    ),
    GraphStateContract(
        name="agent_saul.deep_research_subgraph",
        module_path="src/app/features/agent_saul/graph/deep_research.py",
        state=DeepResearchState,
        # The wrapper constructs the initial state; these are exactly its keys.
        initial_state_keys=frozenset(
            {
                "objective",
                "focus",
                "user_id",
                "tenant_id",
                "allowed_doc_ids",
                "thread_id",
                "run_id",
                "evidence_checkpoint_id",
                "queries",
                "pending_handles",
                "evidence",
                "read_notes",
                "refusals",
                "coverage",
                "steps_used",
                "max_steps",
                "max_reads",
                "search_limit",
                "status",
                "summary",
                "message",
            }
        ),
    ),
    GraphStateContract(
        name="agent_saul.reconcile_read_graph",
        module_path="src/app/features/agent_saul/application/reconcile.py",
        state=LegalAgentState,
        initial_state_keys=_SAUL_INITIAL_STATE_KEYS,
    ),
)


def undeclared_state_keys(keys: frozenset[str], state: type) -> list[str]:
    """Return the keys ``state`` does not declare, sorted for a mechanical fix."""
    return sorted(keys - set(get_type_hints(state)))


@pytest.mark.parametrize(
    "contract",
    GRAPH_STATE_CONTRACTS,
    ids=[contract.name for contract in GRAPH_STATE_CONTRACTS],
)
def test_initial_state_is_a_subset_of_the_declared_channels(
    contract: GraphStateContract,
) -> None:
    """The boundary contract: ``set(initial_state) <= set(State.__annotations__)``."""
    undeclared = undeclared_state_keys(contract.initial_state_keys, contract.state)

    assert undeclared == [], (
        f"{contract.name} sends keys its state does not declare: {undeclared}. "
        "LangGraph will discard them silently; map each key onto a declared "
        "channel or route it through invocation config."
    )


def test_every_compile_site_is_registered() -> None:
    """Every compiled graph in ``src/app`` joins the registry, by count and path."""
    compile_sites: dict[str, int] = {}
    for module in _SRC_ROOT.rglob("*.py"):
        count = 0
        for line in module.read_text(encoding="utf-8").splitlines():
            code = line.split("#", maxsplit=1)[0]
            # `re.compile(...)` is a regex, not a graph; commented-out
            # optimiser snippets are not compiled at all.
            if "re.compile(" in code:
                continue
            count += len(_COMPILE_RE.findall(code))
        if count:
            compile_sites[module.relative_to(_REPO_ROOT).as_posix()] = count

    registered: dict[str, int] = {}
    for contract in GRAPH_STATE_CONTRACTS:
        registered[contract.module_path] = registered.get(contract.module_path, 0) + 1

    assert compile_sites == registered, (
        "the compiled-graph registry is out of date: a new graph was compiled "
        "without a key-set boundary contract, or a contract points at a module "
        f"that no longer compiles a graph. Compiled: {compile_sites}. "
        f"Registered: {registered}."
    )


def test_the_historical_legacy_shape_is_disjoint_and_every_key_is_named() -> None:
    """Task 2.2: the pre-mapping legacy shape names every undeclared key.

    This is the regression scenario. Against the historical legacy service the
    assertion below reports all six keys; the mapping in task 3.1 must place
    every one of them, and a revert to the old dict fails the test naming them.
    """
    undeclared = undeclared_state_keys(HISTORICAL_LEGACY_STATE_KEYS, DocumentIngestionState)

    assert undeclared == sorted(HISTORICAL_LEGACY_STATE_KEYS)


def test_the_mapped_legacy_state_declares_every_key() -> None:
    """Task 3.1: after mapping, no undeclared key reaches ``ainvoke``."""
    legacy = {
        "doc_id": "doc-1",
        "user_id": "user-1",
        "tenant_id": "personal:user-1",
        "filename": "contract.pdf",
        "content_type": "application/pdf",
        "object_uri": "s3://bucket/documents/user-1/doc-1/hash.pdf",
        "document_version": 1,
        "attempt": 0,
        "document_type": "legal_contract",
        "thread_id": "thread-1",
        "source": "contract.pdf",
        "jurisdiction": "India",
        "raw_bytes": b"clause text",
    }

    state, configurable, unmapped = map_legacy_state(legacy)

    assert unmapped == []
    assert undeclared_state_keys(frozenset(state), DocumentIngestionState) == []
    assert state["document_id"] == "doc-1"
    assert state["document_kind"] == "legal_contract"
    assert configurable == {
        "thread_id": "thread-1",
        "source": "contract.pdf",
        "jurisdiction": "India",
    }


def test_an_unrouted_raw_bytes_key_is_named_not_dropped() -> None:
    """A key that cannot be placed is enumerated, never silently discarded."""
    legacy = {"doc_id": "doc-1", "raw_bytes": b"clause text", "unknown_extra": 1}

    state, configurable, unmapped = map_legacy_state(legacy)

    assert unmapped == ["raw_bytes", "unknown_extra"]
    assert "object_uri" not in state
    assert configurable == {}
