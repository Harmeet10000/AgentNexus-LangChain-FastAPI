"""Regression tests — the ingestion graph's state schema must resolve at runtime.

`IngestionState` is a TypedDict LangGraph evaluates when it builds the graph.
`state.py` carries `from __future__ import annotations`, so every annotation in
it is a string; names used in those strings must exist at runtime or the graph
build fails. `Annotated` and the `operator.add` reducer on
`contextualized_chunks` are the load-bearing cases: with the import confined
to a type-checking block the reducer metadata is silently absent.

Three things kept the equivalent defect invisible last time, and each is worth
knowing:

* **Ruff was right and the advice was wrong.** Under `from __future__ import annotations`
  the import genuinely is typing-only by the language's rules, so `TC003` correctly asked
  for it to be moved. That rule does not know annotation consumers resolve strings at
  runtime. The suppression on the import is therefore load-bearing, not cosmetic.
* **`ty` cannot see it.** Nothing is mis-typed; the name resolves fine under type checking.
  The failure exists only at runtime, in namespace lookup.
* **The suite could not reach it.** `tests/conftest.py` replaced
  `app.shared.langgraph_layer` with a `MagicMock`, so no test had ever resolved these
  hints — or could have.

The reducer test below is the one that matters most. `Annotated[..., operator.add]` is what
makes `Send` fan-out results **accumulate** rather than overwrite, so a silent regression
here would not raise: it would quietly keep one chunk out of every N a document produces.
"""

from __future__ import annotations

import operator
import typing

from app.shared.langgraph_layer.ingestion_kb import state as state_module
from app.shared.langgraph_layer.ingestion_kb.state import (
    ContextualizedChunk,
    IngestionState,
)


def _chunk(clause_id: str, chunk_index: int) -> ContextualizedChunk:
    return ContextualizedChunk(
        clause_id=clause_id,
        chunk_index=chunk_index,
        preamble="Indemnity",
        text="The Supplier shall indemnify the Customer.",
        tokens=8,
    )


def test_the_state_schema_is_a_typed_dict() -> None:
    """Item 227: agent state is a TypedDict, never a validation model."""
    assert typing.is_typeddict(IngestionState)
    assert not hasattr(IngestionState, "model_fields")


def test_the_state_hints_resolve_fully_at_runtime() -> None:
    """Asserted directly, because a graph can compile while a channel hint stays
    unresolved — the failure then surfaces on first use, far from the import.
    """
    hints = typing.get_type_hints(IngestionState, include_extras=True)
    assert set(hints) >= {"doc_id", "segments", "contextualized_chunks", "failure"}


def test_annotated_is_available_at_runtime_not_only_to_the_type_checker() -> None:
    """Pins the mechanism, so a future `TC003` autofix fails here instead of in production.

    `ruff check --fix` would move this import back into a type-checking block, and the
    resulting breakage appears nowhere near the import.
    """
    assert getattr(state_module, "Annotated", None) is typing.Annotated


def test_the_fan_out_reducer_accumulates_rather_than_overwrites() -> None:
    """The load-bearing half of the annotation that could not resolve.

    `Send` dispatches one invocation per segment and each returns a single-element list.
    Without `operator.add` as the reducer, the last one to finish wins and every other
    chunk is silently dropped — no exception, just a document short of most of its chunks.
    """
    hints = typing.get_type_hints(IngestionState, include_extras=True)
    metadata = getattr(hints["contextualized_chunks"], "__metadata__", ())

    assert operator.add in metadata


def test_the_reducer_actually_concatenates() -> None:
    """Behavioural, not structural — the metadata being present is not the same as it working."""
    first = [_chunk("clause-1", 0)]
    second = [_chunk("clause-2", 1)]

    hints = typing.get_type_hints(IngestionState, include_extras=True)
    metadata = getattr(hints["contextualized_chunks"], "__metadata__", ())
    reducer = next(item for item in metadata)
    merged = reducer(first, second)

    assert [chunk.clause_id for chunk in merged] == ["clause-1", "clause-2"]


def test_bare_dict_is_a_usable_state_with_documented_fallbacks() -> None:
    """Channels have no defaults; readers use .get() with the documented fallback,
    so resumed plain dicts behave exactly like fresh ones."""
    state: IngestionState = {}

    assert state.get("doc_id", "") == ""
    assert state.get("segments", []) == []
    assert state.get("contextualized_chunks", []) == []
    assert state.get("parsed_document") is None
    assert state.get("failure") is None


def test_channel_names_are_closed_and_explicit() -> None:
    """Replaces `extra="forbid"`: a typo in a node's return dict is silent on a
    TypedDict, so the channel set is pinned here — adding a channel updates
    this list in the same change, deliberately."""
    expected = {
        "doc_id",
        "user_id",
        "thread_id",
        "source",
        "filename",
        "raw_bytes",
        "document_type",
        "jurisdiction",
        "parsed_document",
        "contract_metadata",
        "segments",
        "contextualized_chunks",
        "extracted_entities",
        "extracted_relationships",
        "parent_doc_id",
        "stored_clause_ids",
        "stored_chunks",
        "stored_entity_ids",
        "stored_relationship_ids",
        "graphiti_episode_ids",
        "ingestion_complete",
        "failure",
    }
    assert set(typing.get_type_hints(IngestionState)) == expected
