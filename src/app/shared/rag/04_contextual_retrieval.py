"""Contextual Retrieval - LLM context prefixing + LangChain retrieval function."""

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


def add_context_to_chunk(document: str, chunk: str) -> str:
    prompt = (
        "Provide a short contextual prefix for this chunk.\n\n"
        f"Document preview:\n{document[:700]}\n\nChunk:\n{chunk}"
    )
    context = str(_CHAT.invoke(prompt).content).strip()
    return f"{context}\n{chunk}"


def ingest_document(text: str) -> None:
    with _CONN.cursor() as cur:
        for chunk in _chunk_text(text):
            contextualized = add_context_to_chunk(text, chunk)
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (contextualized, _embed(contextualized)),
            )
    _CONN.commit()


def search_knowledge_base(query: str, *, top_k: int = 3) -> str:
    with _CONN.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (_embed(query), top_k),
        )
        rows = [row[0] for row in cur.fetchall()]
    context = "\n\n".join(rows)
    prompt = f"Answer from the retrieved context.\n\nQuestion: {query}\n\nContext:\n{context}"
    return str(_CHAT.invoke(prompt).content)
