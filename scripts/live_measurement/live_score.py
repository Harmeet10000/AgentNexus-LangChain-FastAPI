"""LIVE scoring of the golden set against the production retrieval path.

Mirrors the deterministic harness but swaps the offline index for the LIVE
index: queries are embedded via the production Gemini path, ranked by the
production vector (+bm25) search over the ingested corpus in Timescale, and
the retrieved live chunks are mapped back to the golden set's convention ids
(``SRC-XX:pNN:cNN``) by normalized text containment.

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

from app.ai.embeddings import EmbeddingTaskType, embed_text  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.features.documents.adapters.persistence.postgres.document_repository import (  # noqa: E402
    build_search_filter_params,
)
from returns.result import Failure  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.connections.postgres import get_database_url  # noqa: E402
from app.features.documents.adapters.persistence.postgres.document_repository import (  # noqa: E402
    DocumentRepository,
)

GOLDEN = REPO / "evals" / "golden" / "legal_retrieval_v2.jsonl"
CORPUS = REPO / "evals" / "corpus" / "artifacts"
REPORTS = REPO / "evals" / "reports"
K = 10
BRANCH = __import__("os").environ.get("LIVE_BRANCH", "bm25")


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
        print(f"live index: {len(self._live_chunks)} chunks")

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
            if target in ctext or ctext in target:
                if len(ctext) > best_len:
                    best_id, best_len = cid, len(ctext)
        return best_id

    async def rank(self, query: str, k: int = K, branch: str = "bm25") -> list[str]:
        """Production ranking over the live index.

        branch='bm25'/'trigram' run the text-based legs (no embeddings — usable
        during the Gemini quota block); branch='vector' embeds via Gemini and
        is recorded PENDING when the quota is exhausted.
        """
        if branch == "vector":
            embedding = await embed_text(query, task_type=EmbeddingTaskType.QUERY)
        async with self._session_factory() as session:
            repo = DocumentRepository(session)
            filter_params = build_search_filter_params(metadata_filter={})
            if branch == "bm25":
                result = await repo.bm25_search(
                    user_id=self.creds["user_id"],
                    tenant_id=self.creds.get("tenant_id"),
                    query=query,
                    candidate_limit=50,
                    filter_params=filter_params,
                )
            elif branch == "trigram":
                result = await repo.trigram_search(
                    user_id=self.creds["user_id"],
                    tenant_id=self.creds.get("tenant_id"),
                    query=query,
                    candidate_limit=50,
                    filter_params=filter_params,
                )
            else:
                result = await repo.vector_search(
                    user_id=self.creds["user_id"],
                    tenant_id=self.creds.get("tenant_id"),
                    embedding=embedding,
                    candidate_limit=50,
                    filter_params=filter_params,
                )
        if isinstance(result, Exception):
            raise result
        if hasattr(result, "unwrap"):
            if isinstance(result, Failure):
                failure = result.failure()
                print(
                    f"vector_search FAILURE: {getattr(failure, 'message', failure)!r} "
                    f"details={getattr(failure, 'details', None)}"
                )
                return []
            rows = result.unwrap()
        else:
            rows = result
        # The search legs project chunk_id + score only; hydrate content for the
        # convention-id containment mapping.
        ids = [str(row.get("chunk_id")) for row in rows[:k] if isinstance(row, dict)]
        if not ids:
            return []
        contents: dict[str, str] = {}
        async with self.engine.connect() as conn:
            res = await conn.execute(
                text("select id::text, content, preamble from chunks where id::text = any(:ids)"),
                {"ids": ids},
            )
            for cid, content, preamble in res.all():
                contents[cid] = (preamble or "") + " " + (content or "")
        mapped: list[str] = []
        for cid in ids:
            conv = self.map_to_convention(contents.get(cid, ""))
            if conv:
                mapped.append(conv)
        return mapped


def score(ranked: list[str], expected: set[str], k: int = K) -> dict[str, float]:
    top = ranked[:k]
    hits = [1 if cid in expected else 0 for cid in top]
    recall = sum(hits) / len(expected) if expected else 0.0
    rr = 0.0
    for pos, hit in enumerate(hits, 1):
        if hit:
            rr = 1.0 / pos
            break
    dcg = sum(
        h / (1 + i) ** 0.65 if (h := 1 if cid in expected else 0) else 0
        for i, cid in enumerate(top)
    )
    idcg = sum(1 / (1 + i) ** 0.65 for i in range(min(len(expected), k)))
    ndcg = dcg / idcg if idcg else 0.0
    return {"recall@k": recall, "mrr": rr, "ndcg@k": ndcg, "hits": sum(hits)}


async def run_pass(index: LiveIndex, golden: list[dict[str, Any]], label: str) -> dict[str, Any]:
    per_query = []
    started = time.perf_counter()
    for row in golden:
        ranked = await index.rank(row["query"], branch=BRANCH)
        per_query.append(
            {
                "case_id": row.get("case_id"),
                "document_kind": row.get("document_kind"),
                "jurisdiction": row.get("jurisdiction"),
                "ranked": ranked,
                "scores": score(ranked, set(row["expected_chunk_ids"])),
            }
        )
    elapsed = time.perf_counter() - started

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        n = len(rows) or 1
        return {
            "recall@10": sum(r["scores"]["recall@k"] for r in rows) / n,
            "mrr": sum(r["scores"]["mrr"] for r in rows) / n,
            "ndcg@10": sum(r["scores"]["ndcg@k"] for r in rows) / n,
        }

    overall = aggregate(per_query)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_jur: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_query:
        by_kind[r["document_kind"] or "unknown"].append(r)
        by_jur[r["jurisdiction"] or "unknown"].append(r)
    report = {
        "label": label,
        "branch": BRANCH,
        "note": "vector leg PENDING — Gemini free-tier daily quota exhausted (see RUNBOOK)" if BRANCH != "vector" else "vector branch",
        "generated_at": datetime.now(UTC).isoformat(),
        "queries": len(per_query),
        "elapsed_seconds": round(elapsed, 1),
        "overall": overall,
        "by_document_kind": {kind: aggregate(rows) for kind, rows in sorted(by_kind.items())},
        "by_jurisdiction": {jur: aggregate(rows) for jur, rows in sorted(by_jur.items())},
        "per_query": per_query,
    }
    return report


async def main() -> None:
    creds = json.loads((REPO / ".testing-credentials.json").read_text())
    golden = load_golden()
    print(f"golden queries: {len(golden)}")
    index = LiveIndex(creds)
    await index.connect()
    try:
        baseline = await run_pass(index, golden, "live_baseline")
        (REPORTS / "live_baseline_legal_retrieval_v2.json").write_text(
            json.dumps(baseline, indent=1)
        )
        print("baseline:", json.dumps(baseline["overall"], indent=1))
        print("by kind:", json.dumps(baseline["by_document_kind"], indent=1))
    finally:
        await index.close()


if __name__ == "__main__":
    asyncio.run(main())
