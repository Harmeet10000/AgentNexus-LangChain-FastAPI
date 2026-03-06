"""Multi-Query RAG - LangChain query diversification + merged retrieval."""

from __future__ import annotations

import psycopg2
from pgvector.psycopg2 import register_vector

from app.shared.langchain_layer.models import build_chat_model, build_embedding_model

_CONN = psycopg2.connect("dbname=rag_db")
register_vector(_CONN)
_EMBEDDINGS = build_embedding_model()
_CHAT = build_chat_model()


def _embed(text: str) -> list[float]:
    return _EMBEDDINGS.embed_query(text)


def _chunk_text(text: str, size: int = 500) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def ingest_document(text: str) -> None:
    with _CONN.cursor() as cur:
        for chunk in _chunk_text(text):
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (chunk, _embed(chunk)),
            )
    _CONN.commit()


def _generate_queries(original_query: str, *, total: int = 4) -> list[str]:
    prompt = (
        f"Generate {max(total - 1, 1)} alternate perspectives for this question.\n"
        "Return one per line.\n\n"
        f"Question: {original_query}"
    )
    raw = str(_CHAT.invoke(prompt).content)
    variants = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    return [original_query, *variants[: total - 1]]


def multi_query_search(original_query: str, *, per_query_k: int = 5) -> str:
    queries = _generate_queries(original_query)
    unique_results: set[str] = set()
    with _CONN.cursor() as cur:
        for query in queries:
            cur.execute(
                "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (_embed(query), per_query_k),
            )
            unique_results.update(row[0] for row in cur.fetchall())
    return "\n\n".join(sorted(unique_results))
