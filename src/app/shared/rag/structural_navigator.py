"""Pure navigation over a serialized Docling document tree."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

type NodePath = tuple[str, ...]

_WORD_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)
_HEADING_LABELS = frozenset({"section_header", "title"})
_STRUCTURAL_LABELS = frozenset({"chapter", "section", "subsection"})


@dataclass(frozen=True, slots=True)
class _Candidate:
    score: int
    order: int
    path: NodePath


def navigate_tree(
    tree: Mapping[str, object],
    query: str,
    *,
    limit: int = 5,
) -> list[NodePath]:
    """Return deterministic heading paths for nodes matching ``query``.

    ``tree`` is the JSON-compatible mapping produced by serializing a Docling
    document. Docling stores child links as JSON references (for example,
    ``#/texts/3``); directly nested child mappings are accepted as well so the
    function is independent of a particular persistence representation.

    The function performs no I/O and does not construct a parser or provider.
    Paths contain the document name followed by structural group and heading
    names. Matching paragraph text therefore resolves to its containing
    section rather than leaking the whole paragraph into a path component.
    """
    normalized_query = " ".join(query.casefold().split())
    query_terms = frozenset(_WORD_PATTERN.findall(normalized_query))
    if not query_terms or limit <= 0:
        return []

    return _TreeWalker(
        root=tree,
        normalized_query=normalized_query,
        query_terms=query_terms,
    ).navigate(limit=limit)


class _TreeWalker:
    """Stateful traversal engine kept private behind the pure public function."""

    def __init__(
        self,
        *,
        root: Mapping[str, object],
        normalized_query: str,
        query_terms: frozenset[str],
    ) -> None:
        self._root = root
        self._normalized_query = normalized_query
        self._query_terms = query_terms
        self._candidates: list[_Candidate] = []
        self._visited_references: set[str] = set()
        self._active_nodes: set[int] = set()
        self._order = 0

    def navigate(self, *, limit: int) -> list[NodePath]:
        root_path = _root_path(self._root)
        body = self._root.get("body", self._root)
        root_node, _ = _resolve_node(self._root, body)
        if root_node is not None:
            self._record_candidate(root_node, root_path)
            self._visit_children(root_node.get("children"), root_path)

        ranked = sorted(
            self._candidates,
            key=lambda candidate: (-candidate.score, candidate.order, candidate.path),
        )
        return _unique_paths(ranked, limit=limit)

    def _visit_children(self, children: object, base_path: NodePath) -> None:
        headings: list[str] = []
        for raw_child in _children(children):
            node, reference = _resolve_node(self._root, raw_child)
            if node is None or self._already_visited(reference):
                continue

            node_identity = id(node)
            if node_identity in self._active_nodes:
                continue
            self._active_nodes.add(node_identity)
            try:
                label = _string(node.get("label")).casefold()
                text = _node_text(node)
                if label in _HEADING_LABELS and text:
                    level = _heading_level(node)
                    del headings[level - 1 :]
                    headings.append(text)

                path = (*base_path, *headings)
                structural_name = _structural_name(node, label=label)
                if structural_name and structural_name not in path:
                    path = (*path, structural_name)

                self._record_candidate(node, path)
                self._visit_children(node.get("children"), path)
            finally:
                self._active_nodes.remove(node_identity)

    def _already_visited(self, reference: str | None) -> bool:
        if reference is None:
            return False
        if reference in self._visited_references:
            return True
        self._visited_references.add(reference)
        return False

    def _record_candidate(self, node: Mapping[str, object], path: NodePath) -> None:
        score = _match_score(node, self._normalized_query, self._query_terms)
        if score and path:
            self._candidates.append(_Candidate(score=score, order=self._order, path=path))
        self._order += 1


def _unique_paths(candidates: Sequence[_Candidate], *, limit: int) -> list[NodePath]:
    result: list[NodePath] = []
    seen_paths: set[NodePath] = set()
    for candidate in candidates:
        if candidate.path in seen_paths:
            continue
        seen_paths.add(candidate.path)
        result.append(candidate.path)
        if len(result) == limit:
            break
    return result


def _children(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _resolve_node(
    root: Mapping[str, object], raw_node: object
) -> tuple[Mapping[str, object] | None, str | None]:
    if not isinstance(raw_node, Mapping):
        return None, None
    node = cast("Mapping[str, object]", raw_node)
    reference = _string(node.get("$ref")) or _string(node.get("cref"))
    if not reference:
        return node, None
    resolved: object = root
    if not reference.startswith("#/"):
        return None, reference
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(resolved, Mapping):
            resolved = resolved.get(part)
        elif isinstance(resolved, Sequence) and not isinstance(resolved, (str, bytes, bytearray)):
            if not part.isdecimal() or int(part) >= len(resolved):
                return None, reference
            resolved = resolved[int(part)]
        else:
            return None, reference
    if not isinstance(resolved, Mapping):
        return None, reference
    return cast("Mapping[str, object]", resolved), reference


def _root_path(tree: Mapping[str, object]) -> NodePath:
    name = _string(tree.get("name")) or _string(tree.get("title"))
    return (name,) if name else ()


def _heading_level(node: Mapping[str, object]) -> int:
    level = node.get("level", 1)
    return max(level, 1) if isinstance(level, int) and not isinstance(level, bool) else 1


def _structural_name(node: Mapping[str, object], *, label: str) -> str:
    if label not in _STRUCTURAL_LABELS:
        return ""
    return _string(node.get("name")) or _node_text(node)


def _node_text(node: Mapping[str, object]) -> str:
    return _string(node.get("text")) or _string(node.get("title"))


def _match_score(
    node: Mapping[str, object], normalized_query: str, query_terms: frozenset[str]
) -> int:
    searchable_parts = (
        _string(node.get("text")),
        _string(node.get("title")),
        _string(node.get("name")),
        _string(node.get("summary")),
    )
    searchable = " ".join(part for part in searchable_parts if part).casefold()
    if not searchable:
        return 0
    searchable_terms = frozenset(_WORD_PATTERN.findall(searchable))
    matched_terms = query_terms & searchable_terms
    if matched_terms != query_terms:
        return 0
    phrase_bonus = len(query_terms) * 2 if normalized_query in searchable else 0
    return len(matched_terms) * 11 + phrase_bonus


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
