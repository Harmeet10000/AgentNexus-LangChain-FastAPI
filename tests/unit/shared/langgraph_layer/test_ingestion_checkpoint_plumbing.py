"""Unit tests for Band D item 5 — checkpointer plumbing and durability.

D17 keeps the application-lifespan construction commented, so every proof here
is over a construction the test itself owns: the graph is compiled with an
in-memory checkpointer (no database), without one, and the thread identity is
derived from the document identity. Nothing here provisions the shared
checkpointer or touches the commented lifespan block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import InMemorySaver

from app.shared.langgraph_layer.ingestion_kb.graph import (
    INGESTION_DURABILITY,
    build_ingestion_graph,
    ingestion_thread_config,
)

if TYPE_CHECKING:
    from typing import Any


def _build(checkpointer: Any | None) -> Any:
    return build_ingestion_graph(
        extraction_llm=object(),
        db_engine=object(),
        graphiti_service=None,
        checkpointer=checkpointer,
    )


def test_construction_accepts_a_checkpointer() -> None:
    saver = InMemorySaver()
    compiled = _build(saver)
    assert compiled.checkpointer is saver


def test_construction_without_a_checkpointer_runs_unpersisted() -> None:
    compiled = _build(None)
    assert compiled.checkpointer is None


def test_thread_identity_derives_from_the_document_identity() -> None:
    config = ingestion_thread_config(user_id="user-1", content_hash="abc123")
    assert config == {"configurable": {"thread_id": "ingestion:user-1:abc123"}}


def test_thread_identity_is_invocation_config_not_state() -> None:
    import inspect

    # The checkpoint thread identity is a pure function of the document
    # identity pair — it takes no state object, so no caller can supply it as
    # a pipeline state value. (`IngestionState.thread_id` is a different thing:
    # the session scope D15 files into `documents.metadata_`.)
    assert list(inspect.signature(ingestion_thread_config).parameters) == [
        "user_id",
        "content_hash",
    ]


def test_durability_mode_is_declared_explicitly() -> None:
    # Decision 8: persist while the next stage executes — recoverable from a
    # mid-execution crash, not only at completion.
    assert INGESTION_DURABILITY == "async"


def test_state_carries_no_arbitrary_types_permission() -> None:
    from app.shared.langgraph_layer.ingestion_kb.state import IngestionState

    assert IngestionState.model_config.get("arbitrary_types_allowed") is not True
    # The model still builds: nothing in it needed the permission.
    assert IngestionState(doc_id="doc-1").doc_id == "doc-1"
