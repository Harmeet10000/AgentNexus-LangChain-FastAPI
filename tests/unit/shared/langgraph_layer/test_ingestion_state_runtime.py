"""Regression tests — the ingestion graph's state model must be constructible at runtime.

`IngestionState` could not be constructed at all. `state.py` carries
`from __future__ import annotations`, so every annotation in it is a string, and Pydantic
**evaluates** those strings when it builds the model. `Annotated` was imported inside an
`if TYPE_CHECKING:` block, so the name was absent at runtime and
`IngestionState.contextualized_chunks` — annotated
`Annotated[list[ContextualizedChunk], operator.add]` — could never be resolved. Every
`IngestionState(...)` raised `PydanticUserError`.

Three things kept it invisible, and each is worth knowing:

* **Ruff was right and the advice was wrong.** Under `from __future__ import annotations`
  the import genuinely is typing-only by the language's rules, so `TC003` correctly asked
  for it to be moved. That rule does not know Pydantic resolves annotations at runtime. The
  suppression on the import is therefore load-bearing, not cosmetic, and the neighbouring
  `AppError` import already carried the same one for the same reason.
* **`ty` cannot see it.** Nothing is mis-typed; the name resolves fine under type checking.
  The failure exists only at runtime, in Pydantic's namespace lookup.
* **The suite could not reach it.** `tests/conftest.py` replaced
  `app.shared.langgraph_layer` with a `MagicMock`, so no test had ever constructed this
  model — or could have.

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


def test_the_state_model_can_be_constructed() -> None:
    """The whole defect, in one line. It raised `PydanticUserError` before the fix."""
    assert IngestionState(doc_id="doc-1").doc_id == "doc-1"


def test_the_state_model_is_fully_defined() -> None:
    """Asserted directly, because construction can succeed while a field stays unresolved.

    Pydantic defers the failure to first use, so a model with an unresolvable annotation on
    a field nothing touches looks healthy until something touches it.
    """
    assert IngestionState.__pydantic_complete__ is True


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
    field = IngestionState.model_fields["contextualized_chunks"]

    assert operator.add in field.metadata


def test_the_reducer_actually_concatenates() -> None:
    """Behavioural, not structural — the metadata being present is not the same as it working."""
    first = [_chunk("clause-1", 0)]
    second = [_chunk("clause-2", 1)]

    reducer = next(item for item in IngestionState.model_fields["contextualized_chunks"].metadata)
    merged = reducer(first, second)

    assert [chunk.clause_id for chunk in merged] == ["clause-1", "clause-2"]


def test_defaults_make_a_bare_state_usable() -> None:
    """The graph's entry node builds a state before it has parsed anything."""
    state = IngestionState()

    assert state.doc_id == ""
    assert state.segments == []
    assert state.contextualized_chunks == []
    assert state.parsed_document is None


def test_extra_keys_are_forbidden() -> None:
    """`extra="forbid"` is why a typo in a node's return dict is an error, not a silent no-op."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IngestionState(doc_id="doc-1", contextualised_chunks=[])  # British spelling, deliberate
