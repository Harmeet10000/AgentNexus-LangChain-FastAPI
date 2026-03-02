"""Multimodal RAG pipeline for processing documents with text, tables, and images."""

import base64
import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PIL import Image


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_ai_summary(
    raw_text: str,
    tables_html: list[str],
    images_base64: list[str],
) -> str:
    """Generate AI summary for mixed content using vision model."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""Analyze this document chunk and create a comprehensive summary:

TEXT:
{raw_text}

TABLES:
{chr(10).join(f"Table {i+1}:{chr(10)}{table}" for i, table in enumerate(tables_html))}

Create a detailed summary that:
1. Describes the main content and key information
2. Explains what the tables show (if any)
3. Describes what the images depict (if any)
4. Highlights important data points, numbers, or findings

Keep the summary detailed but concise (2-3 paragraphs)."""

    message_content: list[dict] = [{"type": "text", "text": prompt}]

    for image_base64 in images_base64:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        })

    message = HumanMessage(content=message_content)
    response = llm.invoke([message])

    return response.content


def process_docling_document(pdf_path: str | Path) -> list[Document]:
    """Process PDF using Docling and create enhanced chunks."""
    print(f"📄 Processing document: {pdf_path}")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    print("✅ Document converted successfully")
    print(f"📊 Found {len(result.document.pages)} pages")

    chunks = result.document.export_to_document_tokens(
        delim="\n\n",
        max_tokens=1000,
        image_mode="placeholder",
    )

    print(f"📦 Created {len(chunks)} chunks")
    print("🔄 Processing chunks with multimodal enhancement...")

    enhanced_docs: list[Document] = []

    for i, chunk in enumerate(chunks, 1):
        print(f"   Processing chunk {i}/{len(chunks)}")

        content_types = {item.self_ref.content_type for item in chunk.meta.doc_items}
        print(f"     Types found: {sorted(content_types)}")

        raw_text = chunk.text
        tables_html: list[str] = []
        images_base64: list[str] = []

        for item in chunk.meta.doc_items:
            if item.self_ref.content_type == "table":
                table = result.document.tables[item.self_ref.obj_ix]
                tables_html.append(table.export_to_html())
            elif item.self_ref.content_type == "image":
                pil_image = result.document.pictures[item.self_ref.obj_ix].get_image()
                images_base64.append(encode_image_to_base64(pil_image))

        print(f"     Tables: {len(tables_html)}, Images: {len(images_base64)}")

        if tables_html or images_base64:
            print("     → Creating AI summary for mixed content...")
            enhanced_text = create_ai_summary(raw_text, tables_html, images_base64)
            print("     → AI summary created successfully")
            print(f"     → Enhanced content preview: {enhanced_text[:200]}...")
        else:
            print("     → Using raw text (no tables/images)")
            enhanced_text = raw_text

        metadata = {
            "chunk_index": i - 1,
            "page_numbers": list(chunk.meta.doc_items[0].prov[0].page_no) if chunk.meta.doc_items else [],
            "content_types": sorted(content_types),
            "has_tables": len(tables_html) > 0,
            "has_images": len(images_base64) > 0,
            "original_content": json.dumps({
                "raw_text": raw_text,
                "tables_html": tables_html,
                "images_base64": images_base64,
            }),
        }

        enhanced_docs.append(Document(page_content=enhanced_text, metadata=metadata))

    print(f"✅ Processed {len(enhanced_docs)} chunks")
    return enhanced_docs


def create_vector_store(documents: Sequence[Document], persist_directory: str = "dbv2/chroma_db") -> Chroma:
    """Create and persist ChromaDB vector store."""
    print("🔮 Creating embeddings and storing in ChromaDB...")
    print("--- Creating vector store ---")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print("--- Finished creating vector store ---")
    print(f"✅ Vector store created and saved to {persist_directory}")
    return db


def run_complete_ingestion_pipeline(pdf_path: str | Path) -> Chroma:
    """Run complete multimodal RAG ingestion pipeline."""
    print("🚀 Starting Multimodal RAG Pipeline")
    print("=" * 50)

    documents = process_docling_document(pdf_path)
    db = create_vector_store(documents)

    print("🎉 Pipeline completed successfully!")
    return db


def generate_final_answer(chunks: Sequence[Document], query: str) -> str:
    """Generate final answer using multimodal content."""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""

        for i, chunk in enumerate(chunks, 1):
            prompt_text += f"--- Document {i} ---\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html, 1):
                        prompt_text += f"Table {j}:\n{table}\n\n"

            prompt_text += "\n"

        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        message_content: list[dict] = [{"type": "text", "text": prompt_text}]

        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])

                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."


def main() -> None:
    """Example usage of the multimodal RAG pipeline."""
    # Ingest document
    db = run_complete_ingestion_pipeline("./docs/attention-is-all-you-need.pdf")

    # Query the vector store
    query = "How many attention heads does the Transformer use, and what is the dimension of each head?"
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke(query)

    # Generate answer
    final_answer = generate_final_answer(chunks, query)
    print(final_answer)


if __name__ == "__main__":
    main()
