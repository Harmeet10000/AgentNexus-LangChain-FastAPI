"""Deterministic offline corpus retriever for the CI retrieval harness.

The offline gate measures the golden corpus against the *recorded* corpus
artifacts: the chunk universe is re-derived from
``evals/corpus/artifacts/<SOURCE_ID>/pages.json`` exactly as documented in
``evals/golden/legal_retrieval_v2.CHUNKING.md`` (normalize rules R1-R7, split on
blank-line runs, 1-based ordinal within the page), so no database, embedding
provider, or model is involved and repeated runs on the same tree are
byte-identical.

Two complementary deterministic lexical signals are ranked per query and fused
with the production ``reciprocal_rank_fusion`` (the single RRF implementation,
reused here so the harness measures over the same fusion code the ask path
uses):

- ``bm25`` — Okapi BM25 over the chunk texts;
- ``trigram`` — character-trigram cosine similarity (fuzzy/fuzzy-typo signal,
  the offline analogue of the ask path's trigram leg).

A change that degrades the corpus, the chunk-id convention, or the golden
expectations moves these metrics, which is what the CI gate observes.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import TYPE_CHECKING

from app.retrieval.fusion import RankedResultRow, reciprocal_rank_fusion

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_TRIGRAM_PATTERN = re.compile(r"[a-z0-9]+")

_BM25_K1 = 1.5
_BM25_B = 0.75
_RRF_K = 60

_TOKEN_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "under",
        "was",
        "were",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "with",
        "within",
    ]
)


def normalize_page_text(text: str) -> str:
    """Corpus normalization rules R1-R7 (evals/corpus/README.md section 6)."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for character in "\u00a0\u2007\u2008\u2009\u200a\u200b":
        text = text.replace(character, " ")
    ligatures = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
    for ligature, expansion in ligatures.items():
        text = text.replace(ligature, expansion)
    text = text.replace("\f", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text)


def chunks_for_page(page_text: str) -> list[str]:
    """Split one page's normalized text into convention chunks (1-based order)."""
    pieces = re.split(r"\n{2,}", normalize_page_text(page_text))
    return [piece for piece in (raw.strip("\n") for raw in pieces) if piece.strip()]


class OfflineCorpusIndex:
    """The chunk universe derived from the recorded corpus artifacts."""

    def __init__(self, corpus_root: Path) -> None:
        self._chunk_ids: list[str] = []
        self._chunk_texts: dict[str, str] = {}
        self._doc_chunks: dict[str, list[str]] = {}
        self._bm25_index: dict[str, Counter[str]] = {}
        self._trigram_index: dict[str, Counter[str]] = {}
        self._build(corpus_root)
        self._avgdl = (
            sum(len(tokens) for tokens in self._bm25_index.values()) / len(self._bm25_index)
            if self._bm25_index
            else 0.0
        )
        self._doc_freq: Counter[str] = Counter()
        for tokens in self._bm25_index.values():
            self._doc_freq.update(set(tokens))

    def _build(self, corpus_root: Path) -> None:
        for pages_path in sorted(corpus_root.glob("*/pages.json")):
            source_id = pages_path.parent.name
            payload = json.loads(pages_path.read_text(encoding="utf-8"))
            for page in payload.get("pages", []):
                page_no = page["page_no"]
                for ordinal, chunk in enumerate(chunks_for_page(page.get("text", "")), start=1):
                    chunk_id = f"{source_id}:p{page_no}:c{ordinal:02d}"
                    self._chunk_ids.append(chunk_id)
                    self._chunk_texts[chunk_id] = chunk
                    self._doc_chunks.setdefault(source_id, []).append(chunk_id)
                    self._bm25_index[chunk_id] = Counter(_tokenize(chunk))
                    self._trigram_index[chunk_id] = Counter(_trigrams(chunk))

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)

    def chunk_ids_for_documents(self, document_ids: Iterable[str]) -> list[str]:
        return [cid for doc in document_ids for cid in self._doc_chunks.get(doc, [])]

    def bm25_rank(self, query: str, k: int) -> list[str]:
        """Deterministic BM25 ranking over the whole chunk universe."""
        query_tokens = [t for t in _tokenize(query) if t not in _TOKEN_STOPWORDS]
        n = len(self._bm25_index)
        scores: list[tuple[float, str]] = []
        for chunk_id, tokens in self._bm25_index.items():
            score = 0.0
            for token in query_tokens:
                tf = tokens.get(token, 0)
                if tf == 0:
                    continue
                df = self._doc_freq.get(token, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                score += idf * (
                    tf
                    * (_BM25_K1 + 1.0)
                    / (tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * len(tokens) / self._avgdl))
                )
            if score > 0.0:
                scores.append((score, chunk_id))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [chunk_id for _, chunk_id in scores[:k]]

    def trigram_rank(self, query: str, k: int) -> list[str]:
        """Deterministic character-trigram cosine ranking over the universe."""
        query_trigrams = Counter(_trigrams(query))
        if not query_trigrams:
            return []
        query_norm = math.sqrt(sum(v * v for v in query_trigrams.values()))
        scores: list[tuple[float, str]] = []
        for chunk_id, trigrams in self._trigram_index.items():
            dot = sum(count * trigrams.get(g, 0) for g, count in query_trigrams.items())
            if dot == 0:
                continue
            chunk_norm = math.sqrt(sum(v * v for v in trigrams.values()))
            scores.append((dot / (query_norm * chunk_norm), chunk_id))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [chunk_id for _, chunk_id in scores[:k]]

    def fused_rank(self, query: str, k: int) -> list[str]:
        """RRF-fuse the two lexical signals with the production fusion code."""
        return fuse_branch_ranks(self.branch_ranks(query, k), k)

    def graph_rank(self, query: str, k: int) -> list[str]:
        """Entity-anchor ranking: the offline analogue of the relationship-graph read.

        The live Graphiti branch surfaces chunks that mention the query's
        entities (parties, defined terms, clause identities) connected through
        recorded relationships. On the recorded artifacts the closest
        deterministic analogue is entity-anchor overlap: multi-word capitalized
        spans of the query (proper nouns, defined terms) matched as exact
        substrings of the chunk text. This is a measurement proxy, not the live
        graph; the harness report that cites it says so explicitly.
        """
        anchors = _entity_anchors(query)
        if not anchors:
            return []
        scores: list[tuple[float, str]] = []
        for chunk_id, text in self._chunk_texts.items():
            lower = text.lower()
            score = 0.0
            for anchor in anchors:
                if anchor in lower:
                    score += len(anchor)
            if score > 0.0:
                scores.append((score, chunk_id))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [chunk_id for _, chunk_id in scores[:k]]

    def branch_ranks(
        self, query: str, k: int, branches: Sequence[str] = ("bm25", "trigram")
    ) -> dict[str, list[str]]:
        """Per-branch ranked identifiers for the harness's attribution rows."""
        all_ranks = {
            "bm25": self.bm25_rank(query, k),
            "trigram": self.trigram_rank(query, k),
            "graph": self.graph_rank(query, k),
        }
        return {name: all_ranks[name] for name in branches if name in all_ranks}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


_ENTITY_ANCHOR_PATTERN = re.compile(r"[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+")


def _entity_anchors(query: str) -> list[str]:
    """Lowercased multi-word capitalized spans (parties, defined terms, acts)."""
    anchors = [match.group(0).lower() for match in _ENTITY_ANCHOR_PATTERN.finditer(query)]
    # Deduplicate, longest first: a longer anchor subsumes its own shorter overlaps.
    unique = list(dict.fromkeys(anchors))
    return sorted(unique, key=len, reverse=True)


def _trigrams(text: str) -> list[str]:
    joined = " ".join(_TRIGRAM_PATTERN.findall(text.lower()))
    return [joined[i : i + 3] for i in range(max(len(joined) - 2, 0))]


def fuse_branch_ranks(
    branch_ranks: dict[str, list[str]],
    k: int,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """RRF-fuse per-branch ranked identifier lists with the production fusion."""
    legs = [
        [
            RankedResultRow(chunk_id=chunk_id, score=0.0, rank=rank)
            for rank, chunk_id in enumerate(ranks, start=1)
        ]
        for ranks in branch_ranks.values()
    ]
    fused = reciprocal_rank_fusion(*legs, k=_RRF_K, limit=k, weights=weights)
    return [item.chunk_id for item in fused]
