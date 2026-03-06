"""Late Chunking - full-document-aware chunk embeddings with LangChain models."""

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


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(dims)]


def late_chunk(text: str, *, chunk_size: int = 512) -> list[tuple[str, list[float]]]:
    """Approximate late chunking by blending full doc and chunk embeddings."""
    tokens = text.split()
    if not tokens:
        return []
    full_doc_embedding = _embed(text)
    results: list[tuple[str, list[float]]] = []
    for start in range(0, len(tokens), chunk_size):
        end = start + chunk_size
        chunk_text = " ".join(tokens[start:end])
        chunk_embedding = _embed(chunk_text)
        blended = _mean_pool([full_doc_embedding, chunk_embedding])
        results.append((chunk_text, blended))
    return results


def ingest_document(text: str) -> None:
    with _CONN.cursor() as cur:
        for chunk_text, embedding in late_chunk(text):
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (chunk_text, embedding),
            )
    _CONN.commit()


def search_knowledge_base(query: str, *, top_k: int = 3) -> str:
    with _CONN.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (_embed(query), top_k),
        )
        context = "\n\n".join(row[0] for row in cur.fetchall())
    return str(_CHAT.invoke(f"Question: {query}\n\nContext:\n{context}").content)
