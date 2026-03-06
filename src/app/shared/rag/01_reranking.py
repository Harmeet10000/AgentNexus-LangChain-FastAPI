"""Re-ranking RAG - Two-stage retrieval with LangChain + cross-encoder."""

from __future__ import annotations

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import CrossEncoder

from app.shared.langchain_layer.models import build_chat_model, build_embedding_model

_CONN = psycopg2.connect("dbname=rag_db")
register_vector(_CONN)
_EMBEDDINGS = build_embedding_model()
_CHAT = build_chat_model()
_RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _chunk_text(text: str, size: int = 500) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _embed(text: str) -> list[float]:
    return _EMBEDDINGS.embed_query(text)


def ingest_document(text: str) -> None:
    """Split and store document chunks with embeddings."""
    chunks = _chunk_text(text)
    with _CONN.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (chunk, _embed(chunk)),
            )
    _CONN.commit()


def search_with_reranking(query: str, *, top_k: int = 5, candidate_k: int = 20) -> str:
    """Retrieve with vectors, then rerank with a cross-encoder."""
    with _CONN.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (_embed(query), candidate_k),
        )
        candidates: list[str] = [row[0] for row in cur.fetchall()]

    if not candidates:
        return "No relevant context found."

    scores = _RERANKER.predict([(query, doc) for doc in candidates])
    ranked = sorted(zip(candidates, scores, strict=False), key=lambda item: item[1], reverse=True)
    context = "\n\n".join(doc for doc, _ in ranked[:top_k])
    prompt = f"Answer the question using only this context.\n\nQuestion: {query}\n\nContext:\n{context}"
    return str(_CHAT.invoke(prompt).content)
