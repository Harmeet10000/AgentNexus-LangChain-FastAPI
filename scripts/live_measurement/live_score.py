"""LIVE scoring of the golden set against the production retrieval path.

Mirrors the deterministic harness but swaps the offline index for the LIVE
index: queries are embedded via the production Gemini path, ranked by the
production vector (+bm25) search over the ingested corpus in Timescale, and
the retrieved live chunks are mapped back to the golden set's convention ids
(``SRC-XX:pNN:cNN``) by normalized text containment.

Branch modes (``LIVE_BRANCH`` env):
- ``bm25`` / ``trigram``: text legs only — no embeddings (usable during the
  Gemini quota block).
- ``vector``: embeds via Gemini; recorded PENDING when the quota is exhausted.
- ``bm25_rerank`` / ``vector_rerank``: standard fusion-then-rerank shape — the
  text/vector leg fetches its top-50 candidates, the hosted cross-encoder
  (``get_shared_reranker()``) reorders them, and the reranked top-k is mapped
  to convention ids. The comparison report
  (``evals/reports/live_rerank_comparison_legal_retrieval_v2.json``) records
  the measured rerank delta and the reranker-latency overhead.

Writes evals/reports/live_*.json. See scripts/live_measurement/RUNBOOK.md.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from returns.result import Failure  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.ai.embeddings import EmbeddingTaskType, embed_text  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.connections.postgres import get_database_url  # noqa: E402
from app.features.documents.adapters.persistence.postgres.document_repository import (  # noqa: E402
    DocumentRepository,
    build_search_filter_params,
)
from app.features.documents.retrieval.graph.state import RetrievedChunk  # noqa: E402
from app.features.documents.retrieval.reranking import reranker as reranker_module  # noqa: E402
from app.features.documents.retrieval.reranking.reranker import (  # noqa: E402
    get_shared_reranker,
)
from app.observability.degradation import RERANK_FALLBACK, record_degradation  # noqa: E402

GOLDEN = REPO / "evals" / "golden" / "legal_retrieval_v2.jsonl"
CORPUS = REPO / "evals" / "corpus" / "artifacts"
REPORTS = REPO / "evals" / "reports"
K = 10
RERANK_CANDIDATES = 50
BRANCH = __import__("os").environ.get("LIVE_BRANCH", "bm25")

# Degradation-capture tap: the HostedReranker degrades silently (fused order
# truncated) on any provider failure, which would collapse the measured rerank
# delta to zero without trace. We wrap the module-level ``record_degradation``
# the reranker calls so every RERANK_FALLBACK during a pass is visible here,
# and back off on provider failures (Cohere 429s) instead of scoring silently
# degraded queries as reranked.
DEGRADATION_EVENTS: list[dict[str, Any]] = []


def _install_degradation_tap() -> None:
    def _tapped(counter: str, *, amount: int = 1, **attributes: object) -> None:
        record_degradation(counter, amount=amount, **attributes)
        DEGRADATION_EVENTS.append({"counter": counter, "amount": amount, **attributes})

    reranker_module.record_degradation = _tapped


def _rerank_events_since(mark: int) -> list[dict[str, Any]]:
    return [e for e in DEGRADATION_EVENTS[mark:] if "rerank" in str(e.get("counter"))]


async def rerank_with_backoff(
    query: str, chunks: list[Any], *, limit: int, max_attempts: int = 3
) -> tuple[list[Any], str, list[list[str]]]:
    """Hosted rerank with honest degradation handling.

    Returns (ordered_chunks, status, attempt_reasons). ``status`` is one of
    ``ok`` / ``key_absent_fired`` / ``degraded_provider_failure_after_retries``.
    Exponential backoff (2s, 4s) between provider-failure attempts.
    """
    attempts: list[list[str]] = []
    ordered: list[Any] = []
    for attempt in range(max_attempts):
        mark = len(DEGRADATION_EVENTS)
        ordered = await get_shared_reranker().rerank(query, chunks, limit=limit)
        reasons = [
            str(e.get("reason"))
            for e in _rerank_events_since(mark)
            if e.get("counter") == RERANK_FALLBACK
        ]
        if not reasons:
            return ordered, "ok", attempts
        if "key_absent" in reasons:
            return ordered, "key_absent_fired", attempts
        attempts.append(reasons)
        if attempt < max_attempts - 1:
            await asyncio.sleep(2 ** (attempt + 1))
    return ordered, "degraded_provider_failure_after_retries", attempts


def normalize(text: str) -> str:
    """Aggressive normalization for the containment mapping."""
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def load_golden() -> list[dict[str, Any]]:
    rows = []
    for line in GOLDEN.read_text().splitlines():
        row = json.loads(line)
        if "expected_chunk_ids" in row:
            rows.append(row)
    return rows


def build_convention_index() -> dict[str, str]:
    """Convention chunk id -> normalized text, from the recorded artifacts."""
    from app.ai.evaluation.deterministic import chunks_for_page, normalize_page_text

    index: dict[str, str] = {}
    for pages_file in sorted(CORPUS.glob("*/pages.json")):
        source_id = pages_file.parent.name
        pages = json.loads(pages_file.read_text())["pages"]
        for page in pages:
            page_no = page["page_no"] if "page_no" in page else page.get("page")
            for ordinal, chunk_text in enumerate(
                chunks_for_page(normalize_page_text(page["text"])), 1
            ):
                index[f"{source_id}:p{page_no}:c{ordinal:02d}"] = normalize(chunk_text)
    return index


async def _rerank_order(
    query: str,
    ids: list[str],
    contents: dict[str, str],
    scores_by_id: dict[str, float],
    k: int,
) -> tuple[list[str], dict[str, Any]]:
    """Fusion-then-rerank: reorder the leg's top-50 by the hosted cross-encoder.

    The hydration already concatenates preamble + content, so the cross-encoder
    sees the same document text the retrieval graph would send. Returns the
    reranked chunk ids plus the rerank-stage diagnostics.
    """
    chunks = [
        RetrievedChunk(
            chunk_id=cid,
            chunk_text=contents.get(cid, ""),
            preamble="",
            clause_type="unknown",
            parent_doc_id="",
            chunk_index=0,
            metadata_={},
            custom_metadata={},
            score=scores_by_id.get(cid, 0.0),
        )
        for cid in ids
    ]
    t0 = time.perf_counter()
    ordered, status, attempts = await rerank_with_backoff(query, chunks, limit=k)
    diagnostics = {
        "rerank_ms": round((time.perf_counter() - t0) * 1000, 1),
        "rerank_status": status,
        "rerank_attempts": attempts,
    }
    return [c.chunk_id for c in ordered], diagnostics


class LiveIndex:
    """Production retrieval over the ingested corpus, plus the id mapper."""

    def __init__(self, creds: dict[str, Any]) -> None:
        self.creds = creds
        self.settings = get_settings()
        self.engine = None
        self._convention = build_convention_index()
        self._live_chunks: list[dict[str, Any]] = []
        self._repo = None
        self._session_factory = None
        self.chunks_by_source: dict[str, int] = {}

    async def connect(self) -> None:
        from app.connections.postgres import init_db

        self.engine = create_async_engine(get_database_url(), pool_pre_ping=True)
        self._session_factory = (await init_db())[1]
        self._repo = DocumentRepository(self._session_factory())
        async with self.engine.connect() as conn:
            res = await conn.execute(text("select id, content, preamble, document_id from chunks"))
            self._live_chunks = [
                {
                    "chunk_id": str(r[0]),
                    "content": r[1] or "",
                    "preamble": r[2] or "",
                    "document_id": str(r[3]),
                }
                for r in res.all()
            ]
        # Map each live chunk to its document's source via the ingestion mapping.
        mapping = json.loads((REPO / ".live-ingestion-mapping.json").read_text())
        self._doc_to_source = {m["doc_id"]: sid for sid, m in mapping.items()}
        for chunk in self._live_chunks:
            sid = self._doc_to_source.get(chunk["document_id"], "UNMAPPED")
            self.chunks_by_source[sid] = self.chunks_by_source.get(sid, 0) + 1
        print(f"live index: {len(self._live_chunks)} chunks")
        print(f"chunks by source: {json.dumps(self.chunks_by_source, sort_keys=True)}")

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()

    def map_to_convention(self, content: str) -> str | None:
        """Map a live chunk to its convention id by normalized containment."""
        target = normalize(content)
        if not target:
            return None
        best_id, best_len = None, 0
        for cid, ctext in self._convention.items():
            if (target in ctext or ctext in target) and len(ctext) > best_len:
                best_id, best_len = cid, len(ctext)
        return best_id

    async def _hydrate(self, ids: list[str]) -> dict[str, str]:
        contents: dict[str, str] = {}
        async with self.engine.connect() as conn:
            res = await conn.execute(
                text("select id::text, content, preamble from chunks where id::text = any(:ids)"),
                {"ids": ids},
            )
            for cid, content, preamble in res.all():
                contents[cid] = (preamble or "") + " " + (content or "")
        return contents

    async def _fetch_candidates(self, query: str, leg: str) -> list[dict[str, Any]]:
        """One production search leg over the live index, top-50 candidates."""
        if leg == "vector":
            embedding = await embed_text(query, task_type=EmbeddingTaskType.QUERY)
        async with self._session_factory() as session:
            repo = DocumentRepository(session)
            filter_params = build_search_filter_params(metadata_filter={})
            if leg == "bm25":
                result = await repo.bm25_search(
                    user_id=self.creds["user_id"],
                    tenant_id=self.creds.get("tenant_id"),
                    query=query,
                    candidate_limit=RERANK_CANDIDATES,
                    filter_params=filter_params,
                )
            elif leg == "trigram":
                result = await repo.trigram_search(
                    user_id=self.creds["user_id"],
                    tenant_id=self.creds.get("tenant_id"),
                    query=query,
                    candidate_limit=RERANK_CANDIDATES,
                    filter_params=filter_params,
                )
            else:
                result = await repo.vector_search(
                    user_id=self.creds["user_id"],
                    tenant_id=self.creds.get("tenant_id"),
                    embedding=embedding,
                    candidate_limit=RERANK_CANDIDATES,
                    filter_params=filter_params,
                )
        if isinstance(result, Exception):
            raise result
        if hasattr(result, "unwrap"):
            if isinstance(result, Failure):
                failure = result.failure()
                print(
                    f"{leg}_search FAILURE: {getattr(failure, 'message', failure)!r} "
                    f"details={getattr(failure, 'details', None)}"
                )
                return []
            return result.unwrap()
        return result

    async def rank(
        self, query: str, k: int = K, branch: str = BRANCH
    ) -> tuple[list[str], dict[str, Any]]:
        """Production ranking over the live index.

        branch='bm25'/'trigram' run the text-based legs (no embeddings — usable
        during the Gemini quota block); branch='vector' embeds via Gemini and is
        recorded PENDING when the quota is exhausted; branch suffix ``_rerank``
        runs the fusion-then-rerank shape: the leg's top-50 candidates are
        hydrated and reordered by the hosted cross-encoder before the
        convention-id mapping.

        Returns (mapped convention ids, diagnostics) where diagnostics carry
        leg/rerank wall-clock timings and the rerank degradation status.
        """
        leg = branch.removesuffix("_rerank")
        want_rerank = branch.endswith("_rerank")
        diagnostics: dict[str, Any] = {"rerank_status": None, "rerank_attempts": []}

        t0 = time.perf_counter()
        rows = await self._fetch_candidates(query, leg)
        diagnostics["leg_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # The search legs project chunk_id + score only; hydrate content for the
        # convention-id containment mapping (all candidates when reranking).
        depth = min(len(rows), RERANK_CANDIDATES if want_rerank else k)
        ids = [str(row.get("chunk_id")) for row in rows[:depth] if isinstance(row, dict)]
        if not ids:
            return [], diagnostics
        contents = await self._hydrate(ids)
        scores_by_id = {
            str(row.get("chunk_id")): float(row.get("score", 0.0))
            for row in rows
            if isinstance(row, dict)
        }
        if want_rerank:
            ids, rerank_diag = await _rerank_order(query, ids, contents, scores_by_id, k)
            diagnostics.update(rerank_diag)
        else:
            ids = ids[:k]

        mapped: list[str] = []
        for cid in ids:
            conv = self.map_to_convention(contents.get(cid, ""))
            if conv:
                mapped.append(conv)
        return mapped, diagnostics


def score(ranked: list[str], expected: set[str], k: int = K) -> dict[str, float]:
    top = ranked[:k]
    hits = [1 if cid in expected else 0 for cid in top]
    recall = sum(hits) / len(expected) if expected else 0.0
    rr = 0.0
    for pos, hit in enumerate(hits, 1):
        if hit:
            rr = 1.0 / pos
            break
    precision_at_5 = sum(hits[:5]) / 5
    dcg = sum(
        h / (1 + i) ** 0.65 if (h := 1 if cid in expected else 0) else 0
        for i, cid in enumerate(top)
    )
    idcg = sum(1 / (1 + i) ** 0.65 for i in range(min(len(expected), k)))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "recall@k": recall,
        "mrr": rr,
        "ndcg@k": ndcg,
        "precision@5": precision_at_5,
        "hits": sum(hits),
    }


async def run_pass(
    index: LiveIndex, golden: list[dict[str, Any]], label: str, branch: str
) -> dict[str, Any]:
    per_query = []
    started = time.perf_counter()
    for row in golden:
        t0 = time.perf_counter()
        ranked, diagnostics = await index.rank(row["query"], branch=branch)
        per_query.append(
            {
                "case_id": row.get("case_id"),
                "document_kind": row.get("document_kind"),
                "jurisdiction": row.get("jurisdiction"),
                "ranked": ranked,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                **diagnostics,
                "scores": score(ranked, set(row["expected_chunk_ids"])),
            }
        )
        print(
            f"[{label}] {row.get('case_id')}: mrr={per_query[-1]['scores']['mrr']:.3f} "
            f"status={diagnostics['rerank_status']} latency={per_query[-1]['latency_ms']}ms"
        )
    elapsed = time.perf_counter() - started

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        n = len(rows) or 1
        return {
            "recall@10": sum(r["scores"]["recall@k"] for r in rows) / n,
            "mrr": sum(r["scores"]["mrr"] for r in rows) / n,
            "ndcg@10": sum(r["scores"]["ndcg@k"] for r in rows) / n,
            "precision@5": sum(r["scores"]["precision@5"] for r in rows) / n,
        }

    overall = aggregate(per_query)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_jur: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_query:
        by_kind[r["document_kind"] or "unknown"].append(r)
        by_jur[r["jurisdiction"] or "unknown"].append(r)
    degraded = [r for r in per_query if r["rerank_status"] not in {"ok", None}]
    latencies = [r["latency_ms"] for r in per_query if r["latency_ms"] is not None]
    rerank_only = [r["rerank_ms"] for r in per_query if r.get("rerank_ms") is not None]
    report = {
        "label": label,
        "branch": branch,
        "note": (
            "vector leg PENDING — Gemini free-tier daily quota exhausted (see RUNBOOK)"
            if branch == "vector"
            else branch
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "queries": len(per_query),
        "elapsed_seconds": round(elapsed, 1),
        "overall": overall,
        "latency": {
            "avg_ms_per_query": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "max_ms_per_query": round(max(latencies), 1) if latencies else None,
            "rerank_stage_avg_ms": round(sum(rerank_only) / len(rerank_only), 1)
            if rerank_only
            else None,
            "rerank_stage_max_ms": round(max(rerank_only), 1) if rerank_only else None,
        },
        "rerank_degraded_queries": len(degraded),
        "rerank_degraded_case_ids": [r["case_id"] for r in degraded],
        "by_document_kind": {kind: aggregate(rows) for kind, rows in sorted(by_kind.items())},
        "by_jurisdiction": {jur: aggregate(rows) for jur, rows in sorted(by_jur.items())},
        "per_query": per_query,
    }
    return report


async def verify_key_present_no_counter_flip() -> dict[str, Any]:
    """Assert the degradation counter does NOT flip with the key present.

    One explicit rerank call: a relevant chunk and a distractor go in; the
    provider path must (a) record no RERANK_FALLBACK (the
    ``hosted_reranker_key_absent_degrading_to_fused_order`` branch is the
    pre-key behaviour), and (b) return provider relevance scores that reorder
    the fused (input) order when the distractor was ranked first.
    """

    def _probe(cid: str, body: str, preamble: str, parent: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=cid,
            chunk_text=body,
            preamble=preamble,
            clause_type="unknown",
            parent_doc_id=parent,
            chunk_index=0,
            metadata_={},
            custom_metadata={},
            score=0.0,
        )

    relevant = _probe(
        "probe-relevant",
        "This agreement is governed by the laws of India. The termination "
        "notice period is ninety (90) days written notice by either party.",
        "Termination and governing law",
        "probe-doc",
    )
    distractor = _probe(
        "probe-distractor",
        "The office pantry coffee machine shall be descaled monthly by the "
        "facilities vendor under the maintenance contract.",
        "Facilities maintenance",
        "probe-doc",
    )
    key_present = bool(get_settings().RERANKER_API_KEY.get_secret_value())
    mark = len(DEGRADATION_EVENTS)
    t0 = time.perf_counter()
    ordered = await get_shared_reranker().rerank(
        "What is the termination notice period for the agreement governed by Indian law?",
        [distractor, relevant],
        limit=2,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    events = _rerank_events_since(mark)
    fallback_events = [e for e in events if e.get("counter") == RERANK_FALLBACK]
    order = [c.chunk_id for c in ordered]
    scores = {c.chunk_id: round(c.score, 4) for c in ordered}
    reordered = order[:1] == ["probe-relevant"] and len(ordered) == 2
    provider_scores_applied = bool(ordered) and any(s > 0.0 for s in scores.values())
    # Behavioural assertion: with the key present the provider path must run —
    # no fallback event, provider relevance scores present, and the relevant
    # chunk ranked above the distractor regardless of the fused input order.
    passed = key_present and not fallback_events and provider_scores_applied and reordered
    return {
        "assertion": (
            "with RERANKER_API_KEY present, hosted_reranker_key_absent_"
            "degrading_to_fused_order must NOT fire; one explicit rerank call "
            "returns provider relevance scores and the relevant chunk first"
        ),
        "key_present": key_present,
        "model": get_settings().RERANKER_MODEL,
        "input_order": ["probe-distractor", "probe-relevant"],
        "returned_order": order,
        "returned_scores": scores,
        "rerank_fallback_events_during_call": fallback_events,
        "reordered_relevant_first": reordered,
        "provider_scores_applied": provider_scores_applied,
        "latency_ms": elapsed_ms,
        "passed": passed,
    }


def golden_coverage(golden: list[dict[str, Any]], index: LiveIndex) -> dict[str, Any]:
    """Partial-corpus coverage, recorded honestly.

    A query is ``uncoverable`` when none of its expected sources have any live
    chunks (quota-failed ingestions) — no retrieval branch can hit it.
    """
    ingested = {s for s, n in index.chunks_by_source.items() if n > 0 and s != "UNMAPPED"}
    fully = partially = uncoverable = 0
    for row in golden:
        expected_sources = {cid.split(":")[0] for cid in row["expected_chunk_ids"]}
        overlap = expected_sources & ingested
        if not overlap:
            uncoverable += 1
        elif overlap < expected_sources:
            partially += 1
        else:
            fully += 1
    return {
        "live_chunks_total": len(index._live_chunks),
        "chunks_by_source": dict(sorted(index.chunks_by_source.items())),
        "ingested_sources": sorted(ingested),
        "golden_queries_total": len(golden),
        "golden_queries_fully_covered": fully,
        "golden_queries_partially_covered": partially,
        "golden_queries_uncoverable": uncoverable,
        "note": (
            "PARTIAL corpus: only the sources above were ingested (the rest are "
            "quota-failed); uncoverable queries cannot be hit by any branch. "
            "Full-corpus confirmation of the rerank delta is PENDING corpus "
            "completion."
        ),
    }


def delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    keys = ("recall@10", "mrr", "ndcg@10", "precision@5")
    return {k: round(after[k] - before[k], 6) for k in keys if k in after and k in before}


def build_comparison_report(
    bm25: dict[str, Any],
    rerank: dict[str, Any],
    vector_pass: dict[str, Any] | None,
    coverage: dict[str, Any],
    counter_flip: dict[str, Any],
) -> dict[str, Any]:
    improved = worsened = unchanged = 0
    per_query: list[dict[str, Any]] = []
    base_by_id = {r["case_id"]: r for r in bm25["per_query"]}
    for r in rerank["per_query"]:
        b = base_by_id[r["case_id"]]
        d = round(r["scores"]["mrr"] - b["scores"]["mrr"], 6)
        if d > 0:
            improved += 1
        elif d < 0:
            worsened += 1
        else:
            unchanged += 1
        per_query.append(
            {
                "case_id": r["case_id"],
                "document_kind": r["document_kind"],
                "jurisdiction": r["jurisdiction"],
                "bm25_mrr": b["scores"]["mrr"],
                "rerank_mrr": r["scores"]["mrr"],
                "mrr_delta": d,
                "bm25_ranked": b["ranked"],
                "rerank_ranked": r["ranked"],
                "rerank_status": r["rerank_status"],
                "rerank_attempts": r["rerank_attempts"],
                "rerank_ms": r.get("rerank_ms"),
            }
        )

    kinds: dict[str, dict[str, float]] = {}
    for kind, agg in rerank["by_document_kind"].items():
        if kind in bm25["by_document_kind"]:
            kinds[kind] = delta(agg, bm25["by_document_kind"][kind])

    overall_delta = delta(rerank["overall"], bm25["overall"])
    better = overall_delta["mrr"] > 0 or overall_delta["precision@5"] > 0
    vector_status = "PENDING" if vector_pass is None else vector_pass.get("status", "measured")
    # Degraded rerank calls collapse to the fused order (the HostedReranker's
    # degradation contract), which biases the measured rerank delta toward
    # zero — the delta below is therefore a conservative lower bound.
    degraded_bm25 = rerank.get("rerank_degraded_queries", 0)
    degraded_vector = (vector_pass or {}).get("rerank_degraded_queries", 0)
    caveats = [
        f"PARTIAL corpus: {coverage['golden_queries_uncoverable']} of "
        f"{coverage['golden_queries_total']} golden queries reference no ingested "
        "source and cannot be hit by any branch.",
        (
            f"{degraded_bm25} of {rerank['queries']} bm25+rerank calls were "
            "rate-limited (429) even after exponential backoff and degraded to "
            "the fused order — the measured rerank delta is a conservative "
            "lower bound."
        ),
    ]
    if degraded_vector:
        caveats.append(
            f"{degraded_vector} of {(vector_pass or {}).get('queries', 0)} "
            "vector+rerank calls were rate-limited after backoff and degraded "
            "to the fused order."
        )
    decision = {
        "question": (
            "Does the hosted cross-encoder reranker improve precision@5/MRR on "
            "the available (partial) corpus?"
        ),
        "measured_verdict": (
            "reranker improves the fused order"
            if better
            else (
                "reranker does not improve the fused order on the available corpus"
                if overall_delta["mrr"] < 0 and overall_delta["precision@5"] < 0
                else "no measurable improvement on the available corpus"
            )
        ),
        "overhead_ms_per_query": round(
            (rerank["latency"]["avg_ms_per_query"] or 0)
            - (bm25["latency"]["avg_ms_per_query"] or 0),
            1,
        ),
        "vector_rerank_leg": vector_status,
        "full_corpus_confirmation": "PENDING corpus completion",
        "flag_decision": (
            "served-path flag decision stays with the owner; this report only "
            "records the measurement"
        ),
    }
    return {
        "report": "live_rerank_comparison_legal_retrieval_v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "golden": "evals/golden/legal_retrieval_v2.jsonl",
        "shape": "fuse (bm25 top-50) -> hosted cross-encoder rerank (rerank-v3.5) -> top-10",
        "corpus_coverage": coverage,
        "passes": {
            "bm25_only": bm25,
            "bm25_rerank": rerank,
            "vector_rerank": vector_pass
            or {
                "status": "PENDING",
                "reason": "Gemini embedding quota blocked — vector+rerank not measured",
            },
        },
        "deltas": {
            "rerank_minus_bm25": {
                "overall": overall_delta,
                "by_document_kind": kinds,
            },
            "vector_rerank_minus_bm25": {
                "overall": delta((vector_pass or {}).get("overall", {}), bm25["overall"])
            },
            "per_query_direction": {
                "improved_mrr": improved,
                "worsened_mrr": worsened,
                "unchanged_mrr": unchanged,
            },
        },
        "measurement_caveats": caveats,
        "latency": {
            "bm25_only_avg_ms_per_query": bm25["latency"]["avg_ms_per_query"],
            "bm25_rerank_avg_ms_per_query": rerank["latency"]["avg_ms_per_query"],
            "rerank_overhead_avg_ms_per_query": decision["overhead_ms_per_query"],
            "rerank_stage_avg_ms": rerank["latency"]["rerank_stage_avg_ms"],
            "rerank_stage_max_ms": rerank["latency"]["rerank_stage_max_ms"],
        },
        "counter_flip_verification": counter_flip,
        "decision": decision,
        "per_query": per_query,
    }


async def main() -> None:
    creds = json.loads((REPO / ".testing-credentials.json").read_text())
    golden = load_golden()
    print(f"golden queries: {len(golden)}")
    _install_degradation_tap()
    index = LiveIndex(creds)
    await index.connect()
    try:
        # Gate 1: the degradation counter must NOT flip with the key present.
        counter_flip = await verify_key_present_no_counter_flip()
        print("counter-flip verification:", json.dumps(counter_flip, indent=1))

        bm25 = await run_pass(index, golden, "bm25_only", "bm25")
        print("bm25_only:", json.dumps(bm25["overall"], indent=1))

        rerank_pass = await run_pass(index, golden, "bm25_rerank", "bm25_rerank")
        print("bm25_rerank:", json.dumps(rerank_pass["overall"], indent=1))

        # Third pass only if the Gemini quota allows even a probe.
        vector_pass: dict[str, Any] | None = None
        try:
            await embed_text("quota probe", task_type=EmbeddingTaskType.QUERY)
            vector_pass = await run_pass(index, golden, "vector_rerank", "vector_rerank")
            print("vector_rerank:", json.dumps(vector_pass["overall"], indent=1))
        except Exception as exc:  # noqa: BLE001 — quota probe failure is a recorded outcome
            print(f"vector leg unavailable (quota): {exc}")

        coverage = golden_coverage(golden, index)
        report = build_comparison_report(bm25, rerank_pass, vector_pass, coverage, counter_flip)
        (REPORTS / "live_rerank_comparison_legal_retrieval_v2.json").write_text(
            json.dumps(report, indent=1)
        )
        print("deltas:", json.dumps(report["deltas"]["rerank_minus_bm25"]["overall"], indent=1))
        print("latency:", json.dumps(report["latency"], indent=1))
        print("coverage:", json.dumps(coverage, indent=1))
    finally:
        await index.close()


if __name__ == "__main__":
    asyncio.run(main())
