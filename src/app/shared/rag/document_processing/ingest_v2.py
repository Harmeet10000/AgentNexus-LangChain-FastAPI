"""
Document ingestion pipeline for processing documents into vector DB.

Function-based architecture with factory functions for dependency injection.
"""

import glob
import json
import os
from datetime import datetime
from typing import Any

from app.utils.logger import logger as loguru_logger

from .chunker import chunk_document, chunk_document_simple, create_hybrid_chunker, get_tokenizer
from .embedder import embed_chunks
from .models import IngestionConfig, IngestionResult


def find_document_files(documents_folder: str) -> list[str]:
    """Find all supported document files in the documents folder."""
    if not os.path.exists(documents_folder):
        loguru_logger.error(f"Documents folder not found: {documents_folder}")
        return []

    # Supported file patterns - Docling + text formats + audio
    patterns = [
        "*.md",
        "*.markdown",
        "*.txt",
        "*.pdf",
        "*.docx",
        "*.doc",
        "*.pptx",
        "*.ppt",
        "*.xlsx",
        "*.xls",
        "*.html",
        "*.htm",
        "*.mp3",
        "*.wav",
        "*.m4a",
        "*.flac",
    ]
    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                os.path.join(documents_folder, "**", pattern), recursive=True
            )
        )

    return sorted(files)


def extract_title(content: str, file_path: str) -> str:
    """Extract title from document content or filename."""
    lines = content.split("\n")
    for line in lines[:10]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()

    return os.path.splitext(os.path.basename(file_path))[0]


def extract_document_metadata(content: str, file_path: str) -> dict[str, Any]:
    """Extract metadata from document content."""
    metadata = {
        "file_path": file_path,
        "file_size": len(content),
        "ingestion_date": datetime.now().isoformat(),
    }

    # Try to extract YAML frontmatter
    if content.startswith("---"):
        try:
            import yaml

            end_marker = content.find("\n---\n", 4)
            if end_marker != -1:
                frontmatter = content[4:end_marker]
                yaml_metadata = yaml.safe_load(frontmatter)
                if isinstance(yaml_metadata, dict):
                    metadata.update(yaml_metadata)
        except ImportError:
            loguru_logger.warning("PyYAML not installed, skipping frontmatter extraction")
        except Exception as e:
            loguru_logger.warning(f"Failed to parse frontmatter: {e}")

    lines = content.split("\n")
    metadata["line_count"] = len(lines)
    metadata["word_count"] = len(content.split())

    return metadata


async def read_document(file_path: str) -> tuple[str, Any | None]:
    """
    Read document content from file - supports multiple formats via Docling.

    Returns:
        Tuple of (markdown_content, docling_document)
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    # Audio formats - transcribe with Whisper ASR
    audio_formats = [".mp3", ".wav", ".m4a", ".flac"]
    if file_ext in audio_formats:
        from .docling_enhanced import _transcribe_audio
        content = await _transcribe_audio(file_path)
        return (content, None)

    # Docling-supported formats
    docling_formats = [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".html", ".htm"]

    if file_ext in docling_formats:
        try:
            from docling.document_converter import DocumentConverter

            loguru_logger.info(
                f"Converting {file_ext} file using Docling: {os.path.basename(file_path)}"
            )

            converter = DocumentConverter()
            result = converter.convert(file_path)

            markdown_content = result.document.export_to_markdown()
            loguru_logger.info(
                f"Successfully converted {os.path.basename(file_path)} to markdown"
            )

            return (markdown_content, result.document)

        except Exception as e:
            loguru_logger.error(f"Failed to convert {file_path} with Docling: {e}")
            try:
                with open(file_path, encoding="utf-8") as f:
                    return (f.read(), None)
            except Exception:
                return (f"[Error: Could not read file {os.path.basename(file_path)}]", None)

    # Text-based formats
    else:
        try:
            with open(file_path, encoding="utf-8") as f:
                return (f.read(), None)
        except UnicodeDecodeError:
            with open(file_path, encoding="latin-1") as f:
                return (f.read(), None)


async def ingest_single_document(
    file_path: str,
    config: IngestionConfig,
    db_pool: Any = None,
    tokenizer: Any = None,
    hybrid_chunker: Any = None,
) -> IngestionResult:
    """
    Ingest a single document.

    Args:
        file_path: Path to the document file
        config: Ingestion configuration
        db_pool: Database connection pool
        tokenizer: Initialized tokenizer
        hybrid_chunker: Initialized HybridChunker

    Returns:
        Ingestion result
    """
    start_time = datetime.now()

    document_content, docling_doc = await read_document(file_path)
    document_title = extract_title(document_content, file_path)
    document_source = os.path.relpath(file_path, "documents")
    document_metadata = extract_document_metadata(document_content, file_path)

    loguru_logger.info(f"Processing document: {document_title}")

    # Chunk the document
    if config.use_semantic_chunking:
        if tokenizer is None:
            tokenizer = get_tokenizer()
        if hybrid_chunker is None:
            hybrid_chunker = create_hybrid_chunker(tokenizer, config)

        chunks = await chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            config=config,
            tokenizer=tokenizer,
            hybrid_chunker=hybrid_chunker,
            metadata=document_metadata,
            docling_doc=docling_doc,
        )
    else:
        chunks = await chunk_document_simple(
            content=document_content,
            title=document_title,
            source=document_source,
            config=config,
            metadata=document_metadata,
        )

    if not chunks:
        loguru_logger.warning(f"No chunks created for {document_title}")
        return IngestionResult(
            document_id="",
            title=document_title,
            chunks_created=0,
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            errors=["No chunks created"],
        )

    loguru_logger.info(f"Created {len(chunks)} chunks")

    # Generate embeddings
    embedded_chunks = await embed_chunks(chunks)
    loguru_logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

    # Save to database (if pool provided)
    document_id = ""
    if db_pool:
        document_id = await save_to_postgres(
            db_pool,
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata,
        )
        loguru_logger.info(f"Saved document to PostgreSQL with ID: {document_id}")

    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    return IngestionResult(
        document_id=document_id,
        title=document_title,
        chunks_created=len(chunks),
        processing_time_ms=processing_time,
    )


async def save_to_postgres(
    db_pool: Any,
    title: str,
    source: str,
    content: str,
    chunks: list,
    metadata: dict[str, Any],
) -> str:
    """Save document and chunks to PostgreSQL."""
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            # Insert document
            document_result = await conn.fetchrow(
                """
                INSERT INTO documents (title, source, content, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id::text
                """,
                title,
                source,
                content,
                json.dumps(metadata),
            )

            document_id = document_result["id"]

            # Insert chunks
            for chunk in chunks:
                embedding_data = None
                if hasattr(chunk, "embedding") and chunk.embedding:
                    embedding_data = "[" + ",".join(map(str, chunk.embedding)) + "]"

                await conn.execute(
                    """
                    INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                    VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                    """,
                    document_id,
                    chunk.content,
                    embedding_data,
                    chunk.chunk_index,
                    json.dumps(chunk.metadata),
                    chunk.token_count,
                )

            return document_id


async def ingest_documents(
    config: IngestionConfig,
    documents_folder: str = "documents",
    db_pool: Any = None,
    clean_before_ingest: bool = True,
    progress_callback: None | callable = None,
) -> list[IngestionResult]:
    """
    Ingest all documents from the documents folder.

    Args:
        config: Ingestion configuration
        documents_folder: Folder containing markdown documents
        db_pool: Optional database connection pool
        clean_before_ingest: Whether to clean existing data before ingestion
        progress_callback: Optional callback for progress updates

    Returns:
        List of ingestion results
    """
    # Clean existing data if requested
    if clean_before_ingest and db_pool:
        await clean_databases(db_pool)

    # Find all supported document files
    document_files = find_document_files(documents_folder)

    if not document_files:
        loguru_logger.warning(
            f"No supported document files found in {documents_folder}"
        )
        return []

    loguru_logger.info(f"Found {len(document_files)} document files to process")

    results = []

    # Initialize shared resources if needed
    tokenizer = None
    hybrid_chunker = None
    if config.use_semantic_chunking:
        tokenizer = get_tokenizer()
        hybrid_chunker = create_hybrid_chunker(tokenizer, config)

    for i, file_path in enumerate(document_files):
        try:
            loguru_logger.info(
                f"Processing file {i + 1}/{len(document_files)}: {file_path}"
            )

            result = await ingest_single_document(
                file_path,
                config,
                db_pool=db_pool,
                tokenizer=tokenizer,
                hybrid_chunker=hybrid_chunker,
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(document_files))

        except Exception as e:
            loguru_logger.error(f"Failed to process {file_path}: {e}")
            results.append(
                IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=[str(e)],
                )
            )

    # Log summary
    total_chunks = sum(r.chunks_created for r in results)
    total_errors = sum(len(r.errors) for r in results)

    loguru_logger.info(
        f"Ingestion complete: {len(results)} documents, {total_chunks} chunks, {total_errors} errors"
    )

    return results


async def clean_databases(db_pool: Any):
    """Clean existing data from databases."""
    loguru_logger.warning("Cleaning existing data from databases...")

    async with db_pool.acquire() as conn, conn.transaction():
        await conn.execute("DELETE FROM chunks")
        await conn.execute("DELETE FROM documents")

    loguru_logger.info("Cleaned PostgreSQL database")


def create_ingestion_pipeline(
    config: IngestionConfig,
    documents_folder: str = "documents",
    db_pool: Any = None,
) -> tuple[callable, callable, callable]:
    """
    Factory function to create ingestion pipeline functions.

    Args:
        config: Ingestion configuration
        documents_folder: Folder containing documents
        db_pool: Database connection pool

    Returns:
        Tuple of (ingest_all, ingest_one, clean) functions
    """

    async def ingest_all(progress_callback: None | callable = None) -> list[IngestionResult]:
        return await ingest_documents(
            config,
            documents_folder=documents_folder,
            db_pool=db_pool,
            progress_callback=progress_callback,
        )

    async def ingest_one(file_path: str) -> IngestionResult:
        return await ingest_single_document(file_path, config, db_pool=db_pool)

    async def clean() -> None:
        if db_pool:
            await clean_databases(db_pool)

    return ingest_all, ingest_one, clean
