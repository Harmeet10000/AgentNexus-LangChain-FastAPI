"""Query Expansion RAG - LangChain-driven variation generation + retrieval."""

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


def expand_query(query: str, *, variants: int = 3) -> list[str]:
    prompt = (
        f"Generate {variants} alternate phrasings of this query.\n"
        "Return one query per line without numbering.\n\n"
        f"Query: {query}"
    )
    raw = str(_CHAT.invoke(prompt).content)
    generated = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    return [query, *generated[:variants]]


def search_knowledge_base(queries: list[str], *, per_query_k: int = 3) -> str:
    unique_results: set[str] = set()
    with _CONN.cursor() as cur:
        for query in queries:
            cur.execute(
                "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (_embed(query), per_query_k),
            )
            unique_results.update(row[0] for row in cur.fetchall())
    return "\n\n".join(sorted(unique_results))
