"""
rag_pipeline.py

Standalone Python conversion of the notebook cells you provided.

Includes:
- run_complete_ingestion_pipeline(path_to_pdf) -> creates chunks, summaries, embeddings, and vector store
- export_chunks_to_json(chunks, filename)
- create_vector_store(documents, persist_directory)
- generate_final_answer(chunks, query)

Notes:
- Replace the placeholder LLM / embedding / Chroma imports with your project's libraries (LangChain, OpenAI, Chroma, etc.).
- This file contains simple fallbacks so it can run in environments where those libs are not installed.
"""

import json
import os
from typing import Any

# -----------------------
# Lightweight fallbacks
# -----------------------
try:
    # Replace these with the libraries you use (langchain, chromadb, openai embeddings, etc.)
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.embeddings import OpenAIEmbeddings  # type: ignore
    from langchain.schema import Document, HumanMessage  # type: ignore
    from langchain.vectorstores import Chroma  # type: ignore
except Exception:
    # Minimal stub classes so the module imports for demo/testing work without external deps.
    class HumanMessage:
        def __init__(self, content: Any):
            self.content = content

    class Document:
        def __init__(self, page_content: str, metadata: dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class ChatOpenAI:
        def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
            self.model = model
            self.temperature = temperature

        def invoke(self, messages: list[HumanMessage]):
            class Resp:
                def __init__(self, content: str):
                    self.content = content

            combined = "\n\n".join(
                (
                    m.content
                    if isinstance(m.content, str)
                    else json.dumps(m.content)[:1000]
                )
                for m in messages
            )
            return Resp(content=(combined[:1500] + "\n\n... (stubbed LLM response)"))

    class OpenAIEmbeddings:
        def __init__(self, model: str = "text-embedding-3-small"):
            self.model = model

        def embed_documents(self, texts: list[str]):
            # return dummy vectors
            return [[float(len(t))] for t in texts]

    class Chroma:
        @staticmethod
        def from_documents(
            documents: list[Document],
            embedding: Any,
            persist_directory: str,
            collection_metadata: dict = None,
        ):
            # Very small stubbed "vector store" object
            class Store:
                def __init__(self, docs):
                    self._docs = docs
                    self.persist_directory = persist_directory

                def as_retriever(self, search_kwargs=None):
                    class Ret:
                        def __init__(self, docs):
                            self._docs = docs

                        def invoke(self, query):
                            # simple keyword scoring: return top-k by substring match in page_content
                            k = (search_kwargs or {}).get("k", 3)
                            scored = []
                            qlow = query.lower()
                            for d in self._docs:
                                score = d.page_content.lower().count(qlow)
                                scored.append((score, d))
                            scored.sort(key=lambda x: x[0], reverse=True)
                            return [d for s, d in scored[:k]]

                    return Ret(documents)

                def persist(self):
                    os.makedirs(self.persist_directory, exist_ok=True)

            vs = Store(documents)
            return vs


# -----------------------
# Chunk summarizer helpers (converted from earlier cell)
# -----------------------
def separate_content_types(chunk: Any) -> dict[str, Any]:
    """Analyze chunk for text, tables, images. Expects chunk to have .text and chunk.metadata.orig_elements (optional)."""
    content_data = {
        "text": getattr(chunk, "text", "") or "",
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    if hasattr(chunk, "metadata") and getattr(chunk.metadata, "orig_elements", None):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(
                    getattr(element, "metadata", None), "text_as_html", None
                ) or getattr(element, "text", "")
                content_data["tables"].append(table_html)
            elif element_type == "Image":
                if hasattr(element, "metadata") and getattr(
                    element.metadata, "image_base64", None
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    # dedupe preserving order
    seen = {}
    content_data["types"] = [
        t for t in content_data["types"] if not (t in seen or seen.setdefault(t, True))
    ]
    return content_data


def create_ai_enhanced_summary(text: str, tables: list[str], images: list[str]) -> str:
    """Call LLM to produce searchable description. Fallback to truncated summary on failure."""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        prompt_text = (
            "You are creating a searchable description for document content retrieval.\n\n"
            "CONTENT TO ANALYZE:\n"
            "TEXT CONTENT:\n"
            f"{text}\n\n"
        )

        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i + 1}:\n{table}\n\n"

        prompt_text += (
            "YOUR TASK:\n"
            "Generate a comprehensive, searchable description that covers:\n"
            "1. Key facts, numbers, and data points from text and tables\n"
            "2. Main topics and concepts discussed\n"
            "3. Questions this content could answer\n"
            "4. Visual content analysis (charts, diagrams, patterns in images)\n"
            "5. Alternative search terms users might use\n\n"
            "Make it detailed and searchable - prioritize findability over brevity.\n\n"
            "SEARCHABLE DESCRIPTION:"
        )

        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception:
        summary = (text or "")[:300] + ("..." if (text or "") else "")
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(chunks: list[Any]) -> list[Document]:
    """Process chunks: produce AI summaries for mixed content or raw text for pure text."""
    print("🧠 Processing chunks with AI Summaries...")
    langchain_documents: list[Document] = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")

        content_data = separate_content_types(chunk)
        print(f"     Types found: {content_data['types']}")
        print(
            f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}"
        )

        if content_data["tables"] or content_data["images"]:
            print("     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data["text"], content_data["tables"], content_data["images"]
                )
                print("     → AI summary created successfully")
                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                enhanced_content = content_data["text"]
        else:
            print("     → Using raw text (no tables/images)")
            enhanced_content = content_data["text"]

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                    }
                )
            },
        )
        langchain_documents.append(doc)

    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents


# -----------------------
# Export helper
# -----------------------
def export_chunks_to_json(
    chunks: list[Document], filename: str = "chunks_export.json"
) -> list[dict[str, Any]]:
    """Export processed chunks to clean JSON format."""
    export_data = []

    for i, doc in enumerate(chunks):
        try:
            original_content = json.loads(doc.metadata.get("original_content", "{}"))
        except Exception:
            original_content = {"raw_text": "", "tables_html": [], "images_base64": []}

        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {"original_content": original_content},
        }
        export_data.append(chunk_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data


# -----------------------
# Vector store
# -----------------------
def create_vector_store(
    documents: list[Document], persist_directory: str = "dbv2/chroma_db"
):
    """Create and persist ChromaDB vector store from Document list."""
    print("🔮 Creating embeddings and storing in ChromaDB...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # if real Chroma: vectorstore.persist()
    print("--- Finished creating vector store ---")
    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore


# -----------------------
# End-to-end pipeline runner (high level)
# -----------------------
def run_complete_ingestion_pipeline(path_to_pdf: str) -> Any:
    """
    High-level pipeline:
      - partition document -> produce 'elements'
      - create smart chunks from elements
      - summarise chunks (AI enhanced)
      - create vector store
    Returns: vector store
    Note: partitioning and chunking code are placeholders — replace with your document loader / partitioner.
    """
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)

    # Placeholder: Document partitioning / loading
    print(f"📄 Partitioning document: {path_to_pdf}")
    # Warning about languages (mimicking notebook output)
    print("Warning: No languages specified, defaulting to English.")

    # Example stub: create artificial 'elements' from the PDF
    # In practice, use a PDF loader that yields elements with text and optional metadata.orig_elements
    class Element:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata or type("M", (), {"orig_elements": None})

    # For demo, create N simple elements.
    extracted_elements = [Element(f"Element text {i + 1}") for i in range(220)]
    print(f"✅ Extracted {len(extracted_elements)} elements")

    # Smart chunking (placeholder) -> create 25 chunks
    print("🔨 Creating smart chunks...")
    chunks = []
    chunk_size = max(1, len(extracted_elements) // 25)
    for i in range(0, len(extracted_elements), chunk_size):
        block = extracted_elements[i : i + chunk_size]

        # Create a lightweight "chunk" object with .text and optional metadata
        class Chunk:
            def __init__(self, text):
                self.text = text
                self.metadata = type("M", (), {"orig_elements": None})

        combined_text = "\n".join(e.text for e in block)
        chunks.append(Chunk(combined_text))
    print(f"✅ Created {len(chunks)} chunks")

    # Summarise chunks with AI (or fallback)
    processed_chunks = summarise_chunks(chunks)

    # Create embeddings + vector store
    db = create_vector_store(processed_chunks, persist_directory="dbv2/chroma_db")

    print("🎉 Pipeline completed successfully!")
    return db, processed_chunks


# -----------------------
# Retrieval + answer generation
# -----------------------
def generate_final_answer(chunks: list[Document], query: str) -> str:
    """
    Build a multimodal prompt from retrieved chunks and call an LLM that can process text+images.
    Returns string answer (or fallback message).
    """
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        prompt_text = f"Based on the following documents, please answer this question: {query}\n\nCONTENT TO ANALYZE:\n\n"

        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i + 1} ---\n"
            if "original_content" in chunk.metadata:
                try:
                    original_data = json.loads(chunk.metadata["original_content"])
                except Exception:
                    original_data = {
                        "raw_text": "",
                        "tables_html": [],
                        "images_base64": [],
                    }

                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j + 1}:\n{table}\n\n"

            prompt_text += "\n"

        prompt_text += (
            "Please provide a clear, comprehensive answer using the text, tables, and images above. "
            "If the documents don't contain sufficient information to answer the question, say "
            '"I don\'t have enough information to answer that question based on the provided documents."\n\n'
            "ANSWER:"
        )

        message_content = [{"type": "text", "text": prompt_text}]

        # Attach images if any
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                try:
                    original_data = json.loads(chunk.metadata["original_content"])
                except Exception:
                    original_data = {}
                images_base64 = original_data.get("images_base64", []) or []
                for image_base64 in images_base64:
                    message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        }
                    )

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."


# -----------------------
# Example usage (main)
# -----------------------
if __name__ == "__main__":
    # Run ingestion pipeline (this uses stubs unless you replace loaders/LLMs)
    db, processed_chunks = run_complete_ingestion_pipeline(
        "./docs/attention-is-all-you-need.pdf"
    )

    # Export processed chunks to JSON
    export_chunks_to_json(processed_chunks, "chunks_export.json")

    # Simple retrieval stub (real vector stores provide retriever.invoke or similar)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    query = "How many attention heads does the Transformer use, and what is the dimension of each head?"
    retrieved_chunks = retriever.invoke(query)

    # Generate final answer
    final_answer = generate_final_answer(retrieved_chunks, query)
    print(final_answer)
