"""Agentic RAG - LangChain tool-calling with vector + SQL + web stubs."""

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
    chunks = _chunk_text(text)
    with _CONN.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (chunk, _embed(chunk)),
            )
    _CONN.commit()


def vector_search(query: str, *, top_k: int = 3) -> str:
    with _CONN.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (_embed(query), top_k),
        )
        rows = [row[0] for row in cur.fetchall()]
    return "\n".join(rows)


def sql_query() -> str:
    with _CONN.cursor() as cur:
        cur.execute("SELECT * FROM sales WHERE quarter='Q2'")
        return str(cur.fetchall())


def web_search(query: str) -> str:
    return f"Web results for: {query}"


def run_agentic_rag(question: str) -> str:
    """Lightweight tool routing decided by LLM output labels."""
    router_prompt = (
        "Choose one tool for the question: vector, sql, or web. "
        f"Question: {question}. Return only the tool name."
    )
    selected = str(_CHAT.invoke(router_prompt).content).strip().lower()
    if "sql" in selected:
        context = sql_query()
    elif "web" in selected:
        context = web_search(question)
    else:
        context = vector_search(question)
    prompt = f"Answer the question using this context.\n\nQuestion: {question}\n\nContext:\n{context}"
    return str(_CHAT.invoke(prompt).content)
