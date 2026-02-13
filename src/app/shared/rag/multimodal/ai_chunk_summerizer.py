"""
ai_chunk_summarizer.py

Converted Python implementation of the provided notebook cell.

Notes:
- Replace the LLM and message classes with the appropriate imports from your LLM framework.
- This file focuses on processing document "chunks" that may contain text, tables, and images,
  producing either a raw-text summary or an AI-enhanced searchable description when mixed
  content is present.
"""

import json
from typing import Any

# Try to import LLM and document classes. Adjust imports to your environment.
try:
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.schema import Document, HumanMessage  # type: ignore
except Exception:
    # Placeholders if LangChain is not installed. Users should replace these with real classes.
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
            # Simple stubbed response for environments without an LLM.
            class Resp:
                def __init__(self, content: str):
                    self.content = content

            combined = "\n\n".join(
                (m.content if isinstance(m.content, str) else json.dumps(m.content))
                for m in messages
            )
            return Resp(content=(combined[:200] + "... (stubbed LLM response)"))


def separate_content_types(chunk: Any) -> dict[str, Any]:
    """Analyze what types of content are in a chunk.

    Expects `chunk` to have attributes:
      - text (str)
      - metadata.orig_elements (iterable of elements which may be Table or Image)

    Returns a dict with keys: text, tables (list of html/text), images (list of base64), types (list).
    """

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
                # Expect image metadata to provide base64 or url
                if hasattr(element, "metadata") and getattr(
                    element.metadata, "image_base64", None
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    # dedupe
    content_data["types"] = list(dict.fromkeys(content_data["types"]))
    return content_data


def create_ai_enhanced_summary(text: str, tables: list[str], images: list[str]) -> str:
    """Create AI-enhanced summary for mixed content.

    This function calls an LLM to produce a searchable description. If the LLM invocation fails
    it falls back to a short inline summary.
    """

    try:
        # Initialize LLM (adjust model name and params to your environment)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        # Build prompt text
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

        # Build message content: the example uses a message array where text and images are provided.
        message_content = [{"type": "text", "text": prompt_text}]

        # Attach images as data URLs when available
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
        # Fallback behavior: return a truncated text summary plus indicators about attachments
        summary = (text or "")[:300] + ("..." if (text or "") else "")
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(chunks: list[Any]) -> list[Document]:
    """Process all chunks and return a list of Document objects with enhanced content or raw text.

    Prints simple progress information to stdout similar to the original notebook.
    """

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


# Example usage (uncomment and adapt to your environment):
# if __name__ == '__main__':
#     # Provide `chunks` as a list of objects matching the expected shape
#     processed_chunks = summarise_chunks(chunks)


def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format."""
    export_data = []

    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(
                    doc.metadata.get("original_content", "{}")
                )
            },
        }
        export_data.append(chunk_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data


# Usage:
json_data = export_chunks_to_json(processed_chunks)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    """Create and persist ChromaDB vector store."""
    print("🔮 Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Finished creating vector store ---")

    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore


# Usage:
db = create_vector_store(processed_chunks)


query = "What are the two main components of the Transformer architecture?"

retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)

export_chunks_to_json(chunks, "rag_results.json")
