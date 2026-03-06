"""Fine-tuned embeddings RAG with LangChain generation for final answers."""

from __future__ import annotations

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from app.shared.langchain_layer.models import build_chat_model

_CONN = psycopg2.connect("dbname=rag_db")
register_vector(_CONN)
_CHAT = build_chat_model()
_EMBEDDING_MODEL = SentenceTransformer("./fine_tuned_model")


def prepare_training_data() -> list[tuple[str, str]]:
    return [
        ("What is EBITDA?", "financial_doc_about_ebitda.txt"),
        ("Explain capital expenditure", "capex_explanation.txt"),
    ]


def get_embedding(text: str) -> list[float]:
    return _EMBEDDING_MODEL.encode(text).tolist()


def _chunk_text(text: str, size: int = 500) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def ingest_document(text: str) -> None:
    with _CONN.cursor() as cur:
        for chunk in _chunk_text(text):
            cur.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (chunk, get_embedding(chunk)),
            )
    _CONN.commit()


def search_knowledge_base(query: str, *, top_k: int = 3) -> str:
    with _CONN.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (get_embedding(query), top_k),
        )
        context = "\n\n".join(row[0] for row in cur.fetchall())
    prompt = f"Answer using this context.\n\nQuestion: {query}\n\nContext:\n{context}"
    return str(_CHAT.invoke(prompt).content)
