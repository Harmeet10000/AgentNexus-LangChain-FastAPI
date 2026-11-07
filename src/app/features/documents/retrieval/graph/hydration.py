"""Section-level context hydration for generation (G5, small-to-big).

When enabled, the generation stage hydrates the parent section of matched
clause-level chunks from the stored structural tree: the generator sees the
section text around each reranked chunk while the retrieved citation still
resolves to the matched clause exactly as without hydration (citations are
resolved from the hydrated chunk rows in ``_finalise_ask_state`` — never from
the hydrated section text).

Adoption gate: DEFAULT OFF. The spec adopts hydration only when the
measurement baseline attributes recall loss to context starvation; the
recorded attribution lives in
``evals/reports/context_starvation_legal_retrieval_v2_wave4.json``. A
regression on the baseline reverts the flag by the same evidence standard.

Hydration is additive and fail-closed: without a live repository, a resolved
tenant, or a matching section in the stored tree the generator payload is
byte-identical to the non-hydrated contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from app.features.documents.retrieval.graph.state import RetrievedChunk
    from app.retrieval.token_counter import CountTokens

#: Per-section token cap: the parent section widens generation context without
#: displacing the chunk budget the assembled context already enforces.
_MAX_SECTION_TOKENS = 1_500

_HEADING_LABELS = frozenset({"section_header", "title"})


def section_text_for_heading_path(
    tree: Mapping[str, object],
    heading_path: Sequence[str],
    *,
    count_tokens: CountTokens,
    max_tokens: int = _MAX_SECTION_TOKENS,
) -> str | None:
    """Return the parent-section text under ``heading_path``, or None.

    ``heading_path`` is the chunk's persisted ``heading_path`` metadata (the
    chunker's heading chain). ``tree`` is the serialized Docling document
    (JSON references like ``#/texts/3`` resolved in place). The section spans
    every text item whose heading chain starts with the target path — the
    parent section plus its nested subsections — truncated to ``max_tokens``
    on a paragraph boundary. A path matching no heading yields None (no
    fabrication).
    """
    target = tuple(str(part) for part in heading_path if part)
    if not target:
        return None
    collected: list[str] = []
    _walk(
        root=tree,
        node=_body(tree),
        heading_path=(),
        target=target,
        collected=collected,
        visited=set(),
    )
    if not collected:
        return None
    text = "\n\n".join(collected)
    while text and count_tokens(text) > max_tokens:
        boundary = text.rfind("\n\n", 0, int(len(text) * 0.8))
        if boundary <= 0:
            break
        text = text[:boundary]
    return text or None


def hydrate_sections_for_chunks(
    *,
    trees: Mapping[str, object],
    chunks: Sequence[RetrievedChunk],
    count_tokens: CountTokens,
) -> dict[str, str]:
    """Per-chunk parent-section text; chunks without a match hydrate nothing."""
    hydrated: dict[str, str] = {}
    for chunk in chunks:
        heading_path = (chunk.metadata_ or {}).get("heading_path")
        if not isinstance(heading_path, (list, tuple)):
            continue
        tree = trees.get(chunk.parent_doc_id)
        if not isinstance(tree, Mapping):
            continue
        section = section_text_for_heading_path(
            cast("Mapping[str, object]", tree),
            [str(part) for part in heading_path],
            count_tokens=count_tokens,
        )
        if section:
            hydrated[chunk.chunk_id] = section
    return hydrated


def _body(tree: Mapping[str, object]) -> Mapping[str, object]:
    body = tree.get("body")
    return cast("Mapping[str, object]", body) if isinstance(body, Mapping) else tree


def _walk(
    *,
    root: Mapping[str, object],
    node: object,
    heading_path: tuple[str, ...],
    target: tuple[str, ...],
    collected: list[str],
    visited: set[str],
) -> tuple[str, ...]:
    """Depth-first traversal; returns the heading path in effect afterwards.

    The heading path threads across siblings (a heading opens a section that
    stays open until the next heading of equal or higher level), mirroring the
    chunker's heading-chain metadata the chunks persist. Once the target
    heading matches, text items are collected — nested subsection headings
    that extend the target path keep the section open; a heading that leaves
    the target path closes it.
    """
    resolved = node
    if isinstance(resolved, str):
        resolved = _resolve_reference(root, resolved)
        if resolved is None:
            return heading_path
    if not isinstance(resolved, Mapping):
        return heading_path
    node_map = cast("Mapping[str, object]", resolved)

    self_ref = _string(node_map.get("self_ref"))
    if self_ref:
        if self_ref in visited:
            return heading_path
        visited.add(self_ref)

    label = _string(node_map.get("label")).casefold()
    text = _node_text(node_map)
    is_heading = label in _HEADING_LABELS and bool(text)
    current_path = heading_path
    if is_heading:
        level = node_map.get("level")
        depth = (
            level if isinstance(level, int) and not isinstance(level, bool) and level >= 1 else 1
        )
        current_path = (*heading_path[: depth - 1], text)
    elif text:
        in_target = (
            len(heading_path) >= len(target) and tuple(heading_path[: len(target)]) == target
        )
        if in_target:
            collected.append(text)

    running = current_path
    for child in _children(node_map.get("children")):
        running = _walk(
            root=root,
            node=child,
            heading_path=running,
            target=target,
            collected=collected,
            visited=visited,
        )
    # A container's internal heading chain does not leak to its following
    # siblings: the chain in effect after this subtree is the chain at the
    # node itself.
    return current_path


def _children(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _node_text(node: Mapping[str, object]) -> str:
    return _string(node.get("text")) or _string(node.get("name")) or _string(node.get("title"))


def _resolve_reference(root: Mapping[str, object], reference: str) -> Mapping[str, object] | None:
    """Resolve a ``#/collection/index`` JSON reference against the tree root."""
    if not reference.startswith("#/"):
        return None
    resolved: object = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(resolved, Mapping):
            resolved = cast("Mapping[str, object]", resolved).get(part)
        elif isinstance(resolved, Sequence) and not isinstance(resolved, (str, bytes, bytearray)):
            if not part.isdecimal() or int(part) >= len(resolved):
                return None
            resolved = resolved[int(part)]
        else:
            return None
    return cast("Mapping[str, object]", resolved) if isinstance(resolved, Mapping) else None
