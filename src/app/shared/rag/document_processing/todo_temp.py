import json
from typing import Any

from app.utils.logger import logger as loguru_logger

from .docling_enhanced import convert_document, create_converter
from .entity_extractor import extract
from .models import DoclingExtractionResult, ToolResult


async def extract_tables(
    source: str,
    format: str = "markdown",
) -> ToolResult:
    """Extract tables from document."""
    try:
        config = create_converter(
            extract_tables=True, extract_code=False, extract_images=False
        )
        result = await convert_document(source, config=config)

        tables_data = []
        for table in result.tables:
            table_dict = {
                "table_index": table.table_index,
                "row_count": table.row_count,
                "col_count": table.col_count,
            }
            if format == "markdown":
                table_dict["content"] = table.markdown
            elif format == "csv":
                table_dict["content"] = table.csv
            elif format == "html":
                table_dict["content"] = table.html

            tables_data.append(table_dict)

        return ToolResult(
            success=True,
            data={"tables": tables_data, "count": len(tables_data)},
            metadata={"source": source, "format": format},
        )

    except Exception as e:
        loguru_logger.error(f"Table extraction failed: {e}")
        return ToolResult(success=False, error=str(e))


async def extract_code_blocks(
    source: str,
) -> ToolResult:
    """Extract code blocks with language detection."""
    try:
        config = create_converter(
            extract_tables=False, extract_code=True, extract_images=False
        )
        result = await convert_document(source, config=config)

        code_data = []
        for block in result.code_blocks:
            code_data.append(
                {
                    "block_index": block.block_index,
                    "code": block.code,
                    "language": block.language,
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                }
            )

        return ToolResult(
            success=True,
            data={"code_blocks": code_data, "count": len(code_data)},
            metadata={"source": source},
        )

    except Exception as e:
        loguru_logger.error(f"Code extraction failed: {e}")
        return ToolResult(success=False, error=str(e))


async def extract_images(
    source: str,
    include_base64: bool = False,
) -> ToolResult:
    """Extract images with captions."""
    try:
        config = create_converter(
            extract_tables=False, extract_code=False, extract_images=True
        )
        result = await convert_document(source, config=config)

        images_data = []
        for image in result.images:
            image_dict = {
                "image_index": image.image_index,
                "caption": image.caption,
                "page_number": image.page_number,
                "bounding_box": image.bounding_box,
            }
            if include_base64 and image.base64_data:
                image_dict["base64"] = image.base64_data

            images_data.append(image_dict)

        return ToolResult(
            success=True,
            data={"images": images_data, "count": len(images_data)},
            metadata={"source": source},
        )

    except Exception as e:
        loguru_logger.error(f"Image extraction failed: {e}")
        return ToolResult(success=False, error=str(e))


async def extract_entities(
    text: str,
    document_id: str,
    use_graphiti: bool = False,
    neo4j_config: dict[str, Any] | None = None,
) -> ToolResult:
    """Extract entities and relationships."""
    try:
        neo4j_config = neo4j_config or {}
        result = await extract(text, document_id, use_graphiti, neo4j_config)

        entities_data = [
            {
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type,
                "description": e.description,
                "properties": e.properties,
            }
            for e in result.entities
        ]

        relationships_data = [
            {
                "id": r.id,
                "source_entity_id": r.source_entity_id,
                "target_entity_id": r.target_entity_id,
                "relationship_type": r.relationship_type,
                "properties": r.properties,
            }
            for r in result.relationships
        ]

        return ToolResult(
            success=True,
            data={
                "entities": entities_data,
                "relationships": relationships_data,
                "entity_count": len(entities_data),
                "relationship_count": len(relationships_data),
            },
            metadata={
                "document_id": document_id,
                "processing_time_ms": result.processing_time_ms,
                "extraction_method": result.metadata.get("extraction_method", "graphiti"),
            },
        )

    except Exception as e:
        loguru_logger.error(f"Entity extraction failed: {e}")
        return ToolResult(success=False, error=str(e))


async def search_rag_data(
    db_pool: Any,
    query: str,
    content_type: str | None = None,
    limit: int = 10,
) -> ToolResult:
    """Search rag_data table."""
    if db_pool is None:
        return ToolResult(success=False, error="Database pool not configured")

    try:
        async with db_pool.acquire() as conn:
            if content_type:
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, content, content_type, data, metadata, created_at
                    FROM rag_data
                    WHERE content ILIKE $1 AND content_type = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    f"%{query}%",
                    content_type,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, content, content_type, data, metadata, created_at
                    FROM rag_data
                    WHERE content ILIKE $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    f"%{query}%",
                    limit,
                )

            results = [dict(row) for row in rows]

            return ToolResult(
                success=True,
                data={"results": results, "count": len(results)},
                metadata={"query": query, "content_type": content_type},
            )

    except Exception as e:
        loguru_logger.error(f"RAG data search failed: {e}")
        return ToolResult(success=False, error=str(e))


async def process_document_full(
    source: str,
    db_pool: Any = None,
    document_id: str | None = None,
    extract_entities_flag: bool = True,
    use_graphiti: bool = False,
) -> ToolResult:
    """Full document processing pipeline."""
    try:
        result = await convert_document(source, document_id=document_id)

        if db_pool:
            await _save_to_rag_data(db_pool, result)

        entities_result = None
        if extract_entities_flag:
            entities_result = await extract(
                result.markdown_content, result.document_id, use_graphiti
            )

        return ToolResult(
            success=True,
            data={
                "document_id": result.document_id,
                "markdown_length": len(result.markdown_content),
                "table_count": len(result.tables),
                "code_block_count": len(result.code_blocks),
                "image_count": len(result.images),
                "entity_count": len(entities_result.entities) if entities_result else 0,
                "relationship_count": len(entities_result.relationships)
                if entities_result
                else 0,
            },
            metadata={
                "source": source,
                "gpu_processed": result.metadata.get("gpu_processed", False),
            },
        )

    except Exception as e:
        loguru_logger.error(f"Full document processing failed: {e}")
        return ToolResult(success=False, error=str(e))


async def _save_to_rag_data(db_pool: Any, result: DoclingExtractionResult):
    """Save extraction results to rag_data table."""
    if db_pool is None:
        return

    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                if result.markdown_content:
                    await conn.execute(
                        """
                        INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                        VALUES ($1, $2, 'markdown', $3, $4)
                        """,
                        result.document_id,
                        result.markdown_content,
                        json.dumps({"length": len(result.markdown_content)}),
                        json.dumps(result.metadata),
                    )

                if result.doctags_content:
                    await conn.execute(
                        """
                        INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                        VALUES ($1, $2, 'doctags', $3, $4)
                        """,
                        result.document_id,
                        result.doctags_content,
                        json.dumps({}),
                        json.dumps(result.metadata),
                    )

                for table in result.tables:
                    await conn.execute(
                        """
                        INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                        VALUES ($1, $2, 'table', $3, $4)
                        """,
                        result.document_id,
                        table.markdown,
                        json.dumps(
                            {
                                "table_index": table.table_index,
                                "csv": table.csv,
                                "html": table.html,
                                "row_count": table.row_count,
                                "col_count": table.col_count,
                            }
                        ),
                        json.dumps(table.metadata),
                    )

                for code in result.code_blocks:
                    await conn.execute(
                        """
                        INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                        VALUES ($1, $2, 'code', $3, $4)
                        """,
                        result.document_id,
                        code.code,
                        json.dumps(
                            {
                                "block_index": code.block_index,
                                "language": code.language,
                                "start_line": code.start_line,
                                "end_line": code.end_line,
                            }
                        ),
                        json.dumps(code.metadata),
                    )

                for image in result.images:
                    await conn.execute(
                        """
                        INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                        VALUES ($1, $2, 'image', $3, $4)
                        """,
                        result.document_id,
                        image.caption or "",
                        json.dumps(
                            {
                                "image_index": image.image_index,
                                "base64_available": bool(image.base64_data),
                                "page_number": image.page_number,
                            }
                        ),
                        json.dumps(image.metadata),
                    )

        loguru_logger.info(f"Saved extraction results: {result.document_id}")

    except Exception as e:
        loguru_logger.error(f"Failed to save to rag_data: {e}")


def create_extraction_tools(
    db_pool: Any = None,
    use_graphiti: bool = False,
    neo4j_config: dict[str, Any] | None = None,
) -> dict[str, callable]:
    """
    Factory function to create extraction tools.

    Args:
        db_pool: Database connection pool
        use_graphiti: Whether to use Graphiti
        neo4j_config: Neo4j configuration

    Returns:
        Dictionary of tool functions
    """
    neo4j_config = neo4j_config or {}

    return {
        "extract_tables": extract_tables,
        "extract_code_blocks": extract_code_blocks,
        "extract_images": extract_images,
        "extract_entities": lambda text, doc_id: extract_entities(
            text, doc_id, use_graphiti, neo4j_config
        ),
        "search_rag_data": lambda query, content_type=None, limit=10: search_rag_data(
            db_pool, query, content_type, limit
        ),
        "process_document_full": lambda source, doc_id=None: process_document_full(
            source, db_pool, doc_id, True, use_graphiti
        ),
    }





# Backward compatibility: Import from tools for old class-based code
__all__ = [
    "ToolResult",
    "extract_tables",
    "extract_code_blocks",
    "extract_images",
    "extract_entities",
    "search_rag_data",
    "process_document_full",
    "create_extraction_tools",
]

    """Collection of document extraction tools for agents."""

    def __init__(
        self,
        db_pool=None,
        use_graphiti: bool = False,
        neo4j_config: dict | None = None,
    ):
        """Initialize extraction tools."""
        self.db_pool = db_pool
        self.use_graphiti = use_graphiti
        self.neo4j_config = neo4j_config or {}
        self.converter = create_converter()
        self.entity_extractor = create_extractor(
            use_graphiti=use_graphiti,
            neo4j_uri=neo4j_config.get("uri") if neo4j_config else None,
            neo4j_user=neo4j_config.get("user") if neo4j_config else None,
            neo4j_password=neo4j_config.get("password") if neo4j_config else None,
        )

    async def extract_tables(
        self,
        source: str,
        format: str = "markdown",
    ) -> ToolResult:
        """Extract tables from document."""
        try:
            converter = create_converter(
                extract_tables=True, extract_code=False, extract_images=False
            )
            result = await converter.convert(source)

            tables_data = []
            for table in result.tables:
                table_dict = {
                    "table_index": table.table_index,
                    "row_count": table.row_count,
                    "col_count": table.col_count,
                }
                if format == "markdown":
                    table_dict["content"] = table.markdown
                elif format == "csv":
                    table_dict["content"] = table.csv
                elif format == "html":
                    table_dict["content"] = table.html

                tables_data.append(table_dict)

            return ToolResult(
                success=True,
                data={"tables": tables_data, "count": len(tables_data)},
                metadata={"source": source, "format": format},
            )

        except Exception as e:
            logger.error("Table extraction failed", error=str(e))
            return ToolResult(success=False, error=str(e))

    async def extract_code_blocks(
        self,
        source: str,
    ) -> ToolResult:
        """Extract code blocks with language detection."""
        try:
            converter = create_converter(
                extract_tables=False, extract_code=True, extract_images=False
            )
            result = await converter.convert(source)

            code_data = []
            for block in result.code_blocks:
                code_data.append(
                    {
                        "block_index": block.block_index,
                        "code": block.code,
                        "language": block.language,
                        "start_line": block.start_line,
                        "end_line": block.end_line,
                    }
                )

            return ToolResult(
                success=True,
                data={"code_blocks": code_data, "count": len(code_data)},
                metadata={"source": source},
            )

        except Exception as e:
            logger.error("Code extraction failed", error=str(e))
            return ToolResult(success=False, error=str(e))

    async def extract_images(
        self,
        source: str,
        include_base64: bool = False,
    ) -> ToolResult:
        """Extract images with captions."""
        try:
            converter = create_converter(
                extract_tables=False, extract_code=False, extract_images=True
            )
            result = await converter.convert(source)

            images_data = []
            for image in result.images:
                image_dict = {
                    "image_index": image.image_index,
                    "caption": image.caption,
                    "page_number": image.page_number,
                    "bounding_box": image.bounding_box,
                }
                if include_base64 and image.base64_data:
                    image_dict["base64"] = image.base64_data

                images_data.append(image_dict)

            return ToolResult(
                success=True,
                data={"images": images_data, "count": len(images_data)},
                metadata={"source": source},
            )

        except Exception as e:
            logger.error("Image extraction failed", error=str(e))
            return ToolResult(success=False, error=str(e))

    async def extract_entities(
        self,
        text: str,
        document_id: str,
    ) -> ToolResult:
        """Extract entities and relationships."""
        try:
            result = await self.entity_extractor.extract(text, document_id)

            entities_data = [
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "description": e.description,
                    "properties": e.properties,
                }
                for e in result.entities
            ]

            relationships_data = [
                {
                    "id": r.id,
                    "source_entity_id": r.source_entity_id,
                    "target_entity_id": r.target_entity_id,
                    "relationship_type": r.relationship_type,
                    "properties": r.properties,
                }
                for r in result.relationships
            ]

            return ToolResult(
                success=True,
                data={
                    "entities": entities_data,
                    "relationships": relationships_data,
                    "entity_count": len(entities_data),
                    "relationship_count": len(relationships_data),
                },
                metadata={
                    "document_id": document_id,
                    "processing_time_ms": result.processing_time_ms,
                    "extraction_method": result.metadata.get("extraction_method", "graphiti"),
                },
            )

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return ToolResult(success=False, error=str(e))

    async def search_rag_data(
        self,
        query: str,
        content_type: str | None = None,
        limit: int = 10,
    ) -> ToolResult:
        """Search rag_data table."""
        if self.db_pool is None:
            return ToolResult(success=False, error="Database pool not configured")

        try:
            async with self.db_pool.acquire() as conn:
                if content_type:
                    rows = await conn.fetch(
                        """
                        SELECT id, document_id, content, content_type, data, metadata, created_at
                        FROM rag_data
                        WHERE content ILIKE $1 AND content_type = $2
                        ORDER BY created_at DESC
                        LIMIT $3
                        """,
                        f"%{query}%",
                        content_type,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, document_id, content, content_type, data, metadata, created_at
                        FROM rag_data
                        WHERE content ILIKE $1
                        ORDER BY created_at DESC
                        LIMIT $2
                        """,
                        f"%{query}%",
                        limit,
                    )

                results = [dict(row) for row in rows]

                return ToolResult(
                    success=True,
                    data={"results": results, "count": len(results)},
                    metadata={"query": query, "content_type": content_type},
                )

        except Exception as e:
            logger.error("RAG data search failed", error=str(e))
            return ToolResult(success=False, error=str(e))

    async def process_document_full(
        self,
        source: str,
        document_id: str | None = None,
        extract_entities_flag: bool = True,
    ) -> ToolResult:
        """Full document processing pipeline."""
        try:
            converter = create_converter()
            result = await converter.convert(source, document_id)

            if self.db_pool:
                await self._save_to_rag_data(result)

            entities_result = None
            if extract_entities_flag:
                entities_result = await self.entity_extractor.extract(
                    result.markdown_content, result.document_id
                )

            return ToolResult(
                success=True,
                data={
                    "document_id": result.document_id,
                    "markdown_length": len(result.markdown_content),
                    "table_count": len(result.tables),
                    "code_block_count": len(result.code_blocks),
                    "image_count": len(result.images),
                    "entity_count": len(entities_result.entities) if entities_result else 0,
                    "relationship_count": len(entities_result.relationships)
                    if entities_result
                    else 0,
                },
                metadata={
                    "source": source,
                    "gpu_processed": result.metadata.get("gpu_processed", False),
                },
            )

        except Exception as e:
            logger.error("Full document processing failed", error=str(e))
            return ToolResult(success=False, error=str(e))

    async def _save_to_rag_data(self, result: DoclingExtractionResult):
        """Save extraction results to rag_data table."""
        if self.db_pool is None:
            return

        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    if result.markdown_content:
                        await conn.execute(
                            """
                            INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                            VALUES ($1, $2, 'markdown', $3, $4)
                            """,
                            result.document_id,
                            result.markdown_content,
                            json.dumps({"length": len(result.markdown_content)}),
                            json.dumps(result.metadata),
                        )

                    if result.doctags_content:
                        await conn.execute(
                            """
                            INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                            VALUES ($1, $2, 'doctags', $3, $4)
                            """,
                            result.document_id,
                            result.doctags_content,
                            json.dumps({}),
                            json.dumps(result.metadata),
                        )

                    for table in result.tables:
                        await conn.execute(
                            """
                            INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                            VALUES ($1, $2, 'table', $3, $4)
                            """,
                            result.document_id,
                            table.markdown,
                            json.dumps(
                                {
                                    "table_index": table.table_index,
                                    "csv": table.csv,
                                    "html": table.html,
                                    "row_count": table.row_count,
                                    "col_count": table.col_count,
                                }
                            ),
                            json.dumps(table.metadata),
                        )

                    for code in result.code_blocks:
                        await conn.execute(
                            """
                            INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                            VALUES ($1, $2, 'code', $3, $4)
                            """,
                            result.document_id,
                            code.code,
                            json.dumps(
                                {
                                    "block_index": code.block_index,
                                    "language": code.language,
                                    "start_line": code.start_line,
                                    "end_line": code.end_line,
                                }
                            ),
                            json.dumps(code.metadata),
                        )

                    for image in result.images:
                        await conn.execute(
                            """
                            INSERT INTO rag_data (document_id, content, content_type, data, metadata)
                            VALUES ($1, $2, 'image', $3, $4)
                            """,
                            result.document_id,
                            image.caption or "",
                            json.dumps(
                                {
                                    "image_index": image.image_index,
                                    "base64_available": bool(image.base64_data),
                                    "page_number": image.page_number,
                                }
                            ),
                            json.dumps(image.metadata),
                        )

            logger.info(
                "Saved extraction results",
                document_id=result.document_id,
            )

        except Exception as e:
            logger.error("Failed to save to rag_data", error=str(e))


def create_extraction_tools(
    db_pool=None,
    use_graphiti: bool = False,
    neo4j_config: dict | None = None,
) -> DocumentExtractionTools:
    """Factory function to create extraction tools."""
    return DocumentExtractionTools(
        db_pool=db_pool,
        use_graphiti=use_graphiti,
        neo4j_config=neo4j_config,
    )
