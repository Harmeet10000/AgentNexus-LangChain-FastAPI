"""Process-scoped graph provider wiring tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from app.features.documents.application.evidence.service import EvidenceService
from app.features.documents.ingestion import workflow as ingestion_graph
from app.lifecycle import graphs
from app.lifecycle.graphs import provide_document_ingestion_graph

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from app.features.documents.application.tools.portfolio import RelationshipEdgeProbe

pytestmark = pytest.mark.unit


def test_document_graph_provider_uses_the_canonical_model_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = object()
    compiled_graph = object()
    model_factory = MagicMock(return_value=llm)
    graph_factory = MagicMock(return_value=compiled_graph)
    monkeypatch.setattr(graphs, "build_chat_model", model_factory)
    monkeypatch.setattr(ingestion_graph, "build_document_ingestion_graph", graph_factory)
    settings = SimpleNamespace(GEMINI_FLASH_MODEL="gemini-flash")

    result = provide_document_ingestion_graph(
        settings=settings,
        object_store=object(),
        graphiti=None,
        ingest_document_fn=MagicMock(),
    )

    assert result is compiled_graph
    model_factory.assert_called_once_with(
        model_name="gemini-flash",
        temperature=0.1,
        implementation="generic",
    )
    assert graph_factory.call_args.kwargs["llm"] is llm


# ---------------------------------------------------------------------------
# Portfolio traversal binding-time gate wiring
# (openspec agent-portfolio-traversal-tools: lifespan probe + degrade rule)
# ---------------------------------------------------------------------------


class _StubGraphitiService:
    """Test double for the read-only relationship-edge fetch.

    Mirrors the real bound read's strict contract: a seeded exception only
    surfaces when the caller asks for strict mode (the binding-time probe);
    leniently the read degrades to an empty result.
    """

    def __init__(self, edges: Sequence[Any] | Exception) -> None:
        self._edges = edges
        self.calls: list[dict[str, Any]] = []

    async def fetch_relationship_edges(
        self,
        query: str,
        *,
        allowed_doc_ids: tuple[str, ...] = (),
        tenant_id: str | None = None,
        contract_index: int | None = None,
        document_versions: Sequence[tuple[str, int, str | None]] = (),
        document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
        num_results: int = 32,
        strict: bool = False,
    ) -> tuple[Any, ...]:
        self.calls.append(
            {
                "query": query,
                "allowed_doc_ids": allowed_doc_ids,
                "tenant_id": tenant_id,
                "document_versions": document_versions,
                "document_contracts": document_contracts,
                "strict": strict,
            }
        )
        if isinstance(self._edges, Exception):
            if strict:
                raise self._edges
            return ()
        return tuple(self._edges)


_ONE_TENANT_SCOPE: dict[str, Any] = {
    "tenant-1": (
        ("doc-1",),
        (("doc-1", 1, None),),
        (("doc-1", 1, (0,), False),),
    )
}


async def test_traversal_gate_probe_returns_a_completed_probe_on_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.documents.application.tools.portfolio import RelationshipEdgeVerdict

    monkeypatch.setattr(graphs, "_corpus_graph_scopes", _async_return(_ONE_TENANT_SCOPE))
    service = _StubGraphitiService([object()])

    probe = await graphs.traversal_gate_probe(service, MagicMock())  # type: ignore[arg-type]

    assert probe is not None
    assert probe.verdict is RelationshipEdgeVerdict.EDGES_PRESENT
    # The probe reused the provisioned backend with the sampled corpus scope.
    assert service.calls[0]["tenant_id"] == "tenant-1"
    assert service.calls[0]["allowed_doc_ids"] == ("doc-1",)


async def test_traversal_gate_probe_degrades_to_none_when_the_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising probe degrades to None: bindings skipped, boot unblocked.

    The probe reads strictly, so the stubbed backend failure surfaces
    instead of masquerading as a completed ``no_edges`` verdict (review
    finding B) — and the degrade-to-None handler catches it.
    """
    monkeypatch.setattr(graphs, "_corpus_graph_scopes", _async_return(_ONE_TENANT_SCOPE))
    unreachable = RuntimeError("neo4j is unreachable")
    service = _StubGraphitiService(unreachable)

    probe = await graphs.traversal_gate_probe(service, MagicMock())  # type: ignore[arg-type]

    assert probe is None
    assert service.calls[0]["strict"] is True


async def test_traversal_gate_probe_degrades_to_none_when_the_backend_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out graph read is unevaluable, never a no_edges verdict."""
    monkeypatch.setattr(graphs, "_corpus_graph_scopes", _async_return(_ONE_TENANT_SCOPE))
    service = _StubGraphitiService(TimeoutError("graph read exceeded 5s"))

    probe = await graphs.traversal_gate_probe(service, MagicMock())  # type: ignore[arg-type]

    assert probe is None


async def test_traversal_gate_probe_returns_a_completed_no_edges_verdict_from_a_genuine_empty_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read that genuinely ran and saw no edges still fails the build (D2)."""
    from app.features.documents.application.tools.portfolio import (
        RelationshipEdgeVerdict,
    )

    monkeypatch.setattr(graphs, "_corpus_graph_scopes", _async_return(_ONE_TENANT_SCOPE))
    service = _StubGraphitiService(())

    probe = await graphs.traversal_gate_probe(service, MagicMock())  # type: ignore[arg-type]

    assert probe is not None
    assert probe.verdict is RelationshipEdgeVerdict.NO_EDGES


async def test_traversal_gate_probe_skips_bindings_on_an_empty_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No documents yet: the gate is not evaluable and must not fail the boot."""
    monkeypatch.setattr(graphs, "_corpus_graph_scopes", _async_return({}))
    service = _StubGraphitiService([object()])

    probe = await graphs.traversal_gate_probe(service, MagicMock())  # type: ignore[arg-type]

    assert probe is None
    assert service.calls == []  # fail-quiet: nothing was read


async def test_saul_graph_provider_degrades_gracefully_when_the_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe error at startup: build still happens, traversal args are None."""
    unreachable = RuntimeError("neo4j is unreachable")

    async def _raising_probe(*_args: Any, **_kwargs: Any) -> RelationshipEdgeProbe:
        raise unreachable

    captured: dict[str, Any] = {}

    def _fake_build_saul_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(graphs, "traversal_gate_probe", _raising_probe)
    monkeypatch.setattr(graphs, "build_chat_model", MagicMock(return_value=object()))
    monkeypatch.setattr(graphs, "build_saul_graph", _fake_build_saul_graph)
    graphiti = _StubGraphitiService([object()])

    compiled = await graphs.provide_saul_graph(  # type: ignore[call-arg]
        settings=SimpleNamespace(
            GEMINI_PRO_MODEL="gemini-pro",
            GEMINI_FLASH_MODEL="gemini-flash",
            EVIDENCE_HANDLE_TTL_SECONDS=3600,
            COGNEE_DATASET_PREFIX="saul",
        ),
        checkpointer=object(),
        redis=MagicMock(),
        db_engine=MagicMock(),
        session_factory=MagicMock(),
        graphiti=graphiti,  # type: ignore[arg-type]
        evidence_service=EvidenceService(repo=object()),
    )

    # The boot survived the probe failure and the graph was still compiled.
    assert compiled is not None
    # Bindings are skipped: no backend and no probe reached the factory.
    assert captured["traversal_graphiti_service"] is None
    assert captured["traversal_gate_probe"] is None


async def test_saul_graph_provider_forwards_a_completed_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.documents.application.tools.portfolio import (
        RelationshipEdgeProbe,
        RelationshipEdgeVerdict,
    )

    completed = RelationshipEdgeProbe(
        verdict=RelationshipEdgeVerdict.EDGES_PRESENT,
        edges_observed=1,
        documents_probed=1,
    )

    async def _probing(*_args: Any, **_kwargs: Any) -> Any:
        return completed

    captured: dict[str, Any] = {}

    def _fake_build_saul_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(graphs, "traversal_gate_probe", _probing)
    monkeypatch.setattr(graphs, "build_chat_model", MagicMock(return_value=object()))
    monkeypatch.setattr(graphs, "build_saul_graph", _fake_build_saul_graph)
    graphiti = _StubGraphitiService([object()])

    await graphs.provide_saul_graph(  # type: ignore[call-arg]
        settings=SimpleNamespace(
            GEMINI_PRO_MODEL="gemini-pro",
            GEMINI_FLASH_MODEL="gemini-flash",
            EVIDENCE_HANDLE_TTL_SECONDS=3600,
            COGNEE_DATASET_PREFIX="saul",
        ),
        checkpointer=object(),
        redis=MagicMock(),
        db_engine=MagicMock(),
        session_factory=MagicMock(),
        graphiti=graphiti,  # type: ignore[arg-type]
        evidence_service=EvidenceService(repo=object()),
    )

    # The same provisioned backend and the completed probe reach the factory.
    assert captured["traversal_graphiti_service"] is graphiti
    assert captured["traversal_gate_probe"] is completed


async def test_saul_graph_provider_builds_without_traversal_tools_when_the_backend_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend timeout at boot (review finding B): bindings skipped, no raise.

    Before the strict probe, the timed-out read degraded to an empty result,
    the probe reported a completed ``no_edges`` verdict, and the strict gate
    in ``build_agent_registry`` failed the whole graph build — taking every
    Saul session down because the graph backend was slow. The provider must
    instead skip the portfolio bindings and still compile the graph.
    """
    monkeypatch.setattr(graphs, "_corpus_graph_scopes", _async_return(_ONE_TENANT_SCOPE))
    captured: dict[str, Any] = {}

    def _fake_build_saul_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(graphs, "build_chat_model", MagicMock(return_value=object()))
    monkeypatch.setattr(graphs, "build_saul_graph", _fake_build_saul_graph)
    graphiti = _StubGraphitiService(TimeoutError("graph read exceeded 5s"))

    compiled = await graphs.provide_saul_graph(  # type: ignore[call-arg]
        settings=SimpleNamespace(
            GEMINI_PRO_MODEL="gemini-pro",
            GEMINI_FLASH_MODEL="gemini-flash",
            EVIDENCE_HANDLE_TTL_SECONDS=3600,
            COGNEE_DATASET_PREFIX="saul",
        ),
        checkpointer=object(),
        redis=MagicMock(),
        db_engine=MagicMock(),
        session_factory=MagicMock(),
        graphiti=graphiti,  # type: ignore[arg-type]
        evidence_service=EvidenceService(repo=object()),
    )

    # The boot survived the backend timeout and the graph was still compiled.
    assert compiled is not None
    # Bindings are skipped: no backend and no probe reached the factory.
    assert captured["traversal_graphiti_service"] is None
    assert captured["traversal_gate_probe"] is None


def _async_return(value: Any) -> Any:
    async def _return(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return _return
