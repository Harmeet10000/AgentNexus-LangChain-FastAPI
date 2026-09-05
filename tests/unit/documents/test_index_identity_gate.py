"""Static gate for the load-bearing schema identifiers.

`chunks_bm25_idx` is not a naming convention but a query contract: the lexical
extension's two-argument `to_bm25query` reads the named index's corpus
statistics, so an index of the right shape under a different name silently
matches nothing. The name is therefore defined once in
`features/documents/constants.py`, and every query and maintenance call naming
it is asserted equal to that definition here.

Deliberately a gate, not an interpolation: change 2's Decision 10 rejects
interpolating an identifier into query text, so the literals stay literals and
the constant is what they are compared against. Renaming the index means
editing the constant *and* every site this gate names — a rename that misses a
site fails here rather than serving empty rankings.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.features.documents import constants, repository
from app.shared.langgraph_layer.ingestion_kb import nodes as ingestion_nodes

_REPO_SOURCE = Path(repository.__file__).read_text(encoding="utf-8")
_NODES_SOURCE = Path(ingestion_nodes.__file__).read_text(encoding="utf-8")

_LITERAL_RE = re.compile(r"to_bm25query\((?::\w+|'\w[^']*'),\s*'([^']+)'\)")


def _bm25_index_literals(source: str) -> list[str]:
    return _LITERAL_RE.findall(source)


def test_every_bm25_index_literal_equals_the_single_definition() -> None:
    literals = _bm25_index_literals(_REPO_SOURCE)
    assert len(literals) == 6, literals
    assert set(literals) == {constants.CHUNKS_BM25_INDEX_NAME}
    assert constants.CHUNKS_BM25_INDEX_NAME == "chunks_bm25_idx"


def test_the_index_maintenance_call_names_the_same_index() -> None:
    assert f"bm25_force_merge('{constants.CHUNKS_BM25_INDEX_NAME}')" in _NODES_SOURCE


def test_the_chunk_upsert_key_equals_the_single_definition() -> None:
    assert "uq_chunks_document_chunk_index" in _REPO_SOURCE
    assert constants.CHUNKS_UNIQUE_CONSTRAINT_NAME == "uq_chunks_document_chunk_index"
