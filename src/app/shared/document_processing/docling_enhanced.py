"""
Enhanced Docling converter with advanced document processing features.

Features:
- Table extraction (CSV/HTML export)
- Code block enrichment (language detection)
- Image extraction
- VLM figure captioning (SmolDocling)
- DocTags export format
- Auto GPU/CPU detection
"""

import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

from app.utils.logger import logger


@dataclass
class DoclingEnhancementConfig:
    """Configuration for Docling enhanced features."""

    extract_tables: bool = True
    extract_code: bool = True
    extract_images: bool = True
    use_vlm_captioning: bool = True
    table_format: str = "markdown"
    generate_doctags: bool = True


@dataclass
class ExtractedTable:
    """Extracted table data."""

    table_index: int
    markdown: str
    csv: str | None = None
    html: str | None = None
    row_count: int = 0
    col_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedCodeBlock:
    """Extracted code block."""

    block_index: int
    code: str
    language: str
    start_line: int
    end_line: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedImage:
    """Extracted image with optional caption."""

    image_index: int
    image_path: str | None = None
    base64_data: str | None = None
    caption: str | None = None
    page_number: int = 1
    bounding_box: list | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DoclingExtractionResult:
    """Complete extraction result from enhanced Docling processing."""

    document_id: str
    markdown_content: str
    doctags_content: str | None = None
    tables: list[ExtractedTable] = field(default_factory=list)
    code_blocks: list[ExtractedCodeBlock] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "document_id": self.document_id,
            "markdown_content": self.markdown_content,
            "doctags_content": self.doctags_content,
            "tables": [
                {
                    "table_index": t.table_index,
                    "markdown": t.markdown,
                    "csv": t.csv,
                    "html": t.html,
                    "row_count": t.row_count,
                    "col_count": t.col_count,
                    "metadata": t.metadata,
                }
                for t in self.tables
            ],
            "code_blocks": [
                {
                    "block_index": c.block_index,
                    "code": c.code,
                    "language": c.language,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "metadata": c.metadata,
                }
                for c in self.code_blocks
            ],
            "images": [
                {
                    "image_index": i.image_index,
                    "image_path": i.image_path,
                    "caption": i.caption,
                    "page_number": i.page_number,
                    "bounding_box": i.bounding_box,
                    "metadata": i.metadata,
                }
                for i in self.images
            ],
            "metadata": self.metadata,
        }


class DoclingEnhancedConverter:
    """
    Enhanced Docling converter with advanced features.

    Automatically detects GPU availability and uses optimal pipeline.
    """

    def __init__(self, config: DoclingEnhancementConfig | None = None):
        """Initialize enhanced converter."""
        self.config = config or DoclingEnhancementConfig()
        self._gpu_available = self._check_gpu()
        self._converter = None
        self._initialize_converter()

    def _check_gpu(self) -> bool:
        """Check if GPU is available for accelerated processing."""
        try:
            import torch

            gpu_available = torch.cuda.is_available()
            if gpu_available:
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("No GPU detected, using CPU pipeline")
            return gpu_available
        except ImportError:
            logger.warning("PyTorch not available, using CPU pipeline")
            return False

    def _initialize_converter(self):
        """Initialize Docling converter with appropriate pipeline."""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyPipelineOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if self._gpu_available:
            logger.info("Using GPU-accelerated PDF pipeline")
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
        else:
            logger.info("Using CPU-efficient EasyPipeline")
            pipeline_options = EasyPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=None,
                    pipeline_options=pipeline_options,
                )
            }
        )

    async def convert(
        self,
        source: str,
        document_id: str | None = None,
    ) -> DoclingExtractionResult:
        """Convert document with enhanced extraction."""
        import hashlib

        from docling.document_converter import DocumentConverter

        if document_id is None:
            document_id = hashlib.md5(source.encode()).hexdigest()[:12]

        logger.info(f"Converting document: {source}")

        converter = DocumentConverter()

        try:
            result = converter.convert(source)
            doc = result.document
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            return DoclingExtractionResult(
                document_id=document_id,
                markdown_content=f"[Conversion error: {e}]",
            )

        markdown_content = doc.export_to_markdown()

        doctags_content = None
        if self.config.generate_doctags:
            try:
                doctags_content = doc.export_to_doc_tags()
            except Exception as e:
                logger.warning(f"DocTags export failed: {e}")

        tables = []
        if self.config.extract_tables:
            tables = self._extract_tables(doc)

        code_blocks = []
        if self.config.extract_code:
            code_blocks = self._extract_code_blocks(doc)

        images = []
        if self.config.extract_images:
            images = await self._extract_images(doc, source)

        metadata = {
            "source": source,
            "gpu_processed": self._gpu_available,
            "table_count": len(tables),
            "code_block_count": len(code_blocks),
            "image_count": len(images),
        }

        return DoclingExtractionResult(
            document_id=document_id,
            markdown_content=markdown_content,
            doctags_content=doctags_content,
            tables=tables,
            code_blocks=code_blocks,
            images=images,
            metadata=metadata,
        )

    def _extract_tables(self, doc) -> list[ExtractedTable]:
        """Extract tables from document."""
        tables = []

        try:
            from docling_core.types.doc import DoclingDocument

            if not isinstance(doc, DoclingDocument):
                return tables

            for idx, table in enumerate(doc.tables):
                try:
                    md_table = table.to_markdown()

                    rows = md_table.split("\n")
                    row_count = len(
                        [r for r in rows if r.strip() and not r.startswith("|---")]
                    )
                    col_count = len(rows[0].split("|")) - 2 if rows else 0

                    csv_content = self._markdown_to_csv(md_table)
                    html_content = self._markdown_to_html(md_table)

                    tables.append(
                        ExtractedTable(
                            table_index=idx,
                            markdown=md_table,
                            csv=csv_content,
                            html=html_content,
                            row_count=row_count,
                            col_count=col_count,
                            metadata={"source_table": idx},
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract table {idx}: {e}")

        except ImportError:
            logger.warning("docling_core not available for table extraction")

        logger.info(f"Extracted {len(tables)} tables")
        return tables

    def _extract_code_blocks(self, doc) -> list[ExtractedCodeBlock]:
        """Extract code blocks with language detection."""
        code_blocks = []

        try:
            from docling_core.types.doc import DoclingDocument

            if not isinstance(doc, DoclingDocument):
                return code_blocks

            for idx, item in enumerate(doc._iterate_nodes()):
                try:
                    if hasattr(item, "meta") and hasattr(item.meta, "text_type"):
                        if item.meta.text_type == "code":
                            code_text = item.text or ""
                            language = self._detect_language(item, code_text)

                            code_blocks.append(
                                ExtractedCodeBlock(
                                    block_index=idx,
                                    code=code_text,
                                    language=language,
                                    start_line=getattr(item, "start_line", 0),
                                    end_line=getattr(item, "end_line", 0),
                                    metadata={"source_block": idx},
                                )
                            )
                except Exception as e:
                    logger.warning(f"Failed to extract code block {idx}: {e}")

        except ImportError:
            logger.warning("docling_core not available for code extraction")

        if not code_blocks:
            code_blocks = self._extract_code_fallback(doc.export_to_markdown())

        logger.info(f"Extracted {len(code_blocks)} code blocks")
        return code_blocks

    def _detect_language(self, item, code_text: str) -> str:
        """Detect programming language from metadata or content."""
        if hasattr(item.meta, "language"):
            return item.meta.language

        if "def " in code_text and ":" in code_text:
            return "python"
        elif "function " in code_text or "const " in code_text or "let " in code_text:
            return "javascript"
        elif "class " in code_text and "{" in code_text:
            return "java"
        elif "#include" in code_text or "int main" in code_text:
            return "c"
        elif "package " in code_text and "func " in code_text:
            return "go"
        elif "fn " in code_text and "->" in code_text:
            return "rust"
        code_upper = code_text.upper()
        if "SELECT " in code_upper:
            return "sql"
        if "{" in code_text and ":" in code_text and "}" in code_text:
            return "json"

        return "unknown"

    def _extract_code_fallback(self, markdown: str) -> list[ExtractedCodeBlock]:
        """Fallback regex-based code extraction."""
        import re

        code_blocks = []
        pattern = r"```(\w+)?\n(.*?)```"

        for idx, match in enumerate(re.finditer(pattern, markdown, re.DOTALL)):
            language = match.group(1) or "unknown"
            code = match.group(2).strip()

            code_blocks.append(
                ExtractedCodeBlock(
                    block_index=idx,
                    code=code,
                    language=language,
                    start_line=markdown[: match.start()].count("\n"),
                    end_line=markdown[: match.end()].count("\n"),
                    metadata={"extraction_method": "regex"},
                )
            )

        return code_blocks

    async def _extract_images(self, doc, source_path: str) -> list[ExtractedImage]:
        """Extract images from document."""
        images = []

        try:
            from docling_core.types.doc import DoclingDocument

            if not isinstance(doc, DoclingDocument):
                return images

            for idx, (key, item) in enumerate(doc._iterate_artifacts()):
                try:
                    if hasattr(item, "image"):
                        image_data = item.image

                        caption = None
                        if hasattr(doc, "figures") and idx < len(doc.figures):
                            fig = doc.figures[idx]
                            caption = getattr(fig, "caption", None)

                        if self.config.use_vlm_captioning and not caption:
                            caption = await self._generate_vlm_caption(image_data)

                        images.append(
                            ExtractedImage(
                                image_index=idx,
                                image_path=key,
                                base64_data=self._encode_base64(image_data),
                                caption=caption,
                                page_number=getattr(item, "page", 1),
                                metadata={"source_image": idx},
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to extract image {idx}: {e}")

        except ImportError:
            logger.warning("docling_core not available for image extraction")

        logger.info(f"Extracted {len(images)} images")
        return images

    async def _generate_vlm_caption(self, image_data) -> str | None:
        """Generate caption using VLM (Gemini)."""
        try:
            from google import genai
            from PIL import Image

            client = genai.Client()

            if isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data))
            else:
                image = image_data

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    image,
                    "Describe this image in detail for document understanding.",
                ],
            )

            return response.text if response.text else None

        except Exception as e:
            logger.warning(f"VLM captioning failed: {e}")
            return None

    def _encode_base64(self, data: bytes) -> str:
        """Encode image data to base64."""
        import base64

        return base64.b64encode(data).decode("utf-8")

    def _markdown_to_csv(self, md_table: str) -> str:
        """Convert markdown table to CSV."""
        import csv
        from io import StringIO

        lines = [
            line
            for line in md_table.split("\n")
            if line.strip() and not line.startswith("|---")
        ]

        if not lines:
            return ""

        header = [cell.strip() for cell in lines[0].split("|")[1:-1]]
        rows = []
        for line in lines[1:]:
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            rows.append(cells)

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)

        return output.getvalue()

    def _markdown_to_html(self, md_table: str) -> str:
        """Convert markdown table to HTML."""
        lines = [
            line
            for line in md_table.split("\n")
            if line.strip() and not line.startswith("|---")
        ]

        if not lines:
            return "<table></table>"

        header = [cell.strip() for cell in lines[0].split("|")[1:-1]]
        rows = []
        for line in lines[1:]:
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            rows.append(cells)

        html = ['<table border="1">']
        html.append("<thead><tr>")
        for cell in header:
            html.append(f"<th>{cell}</th>")
        html.append("</tr></thead>")
        html.append("<tbody>")
        for row in rows:
            html.append("<tr>")
            for cell in row:
                html.append(f"<td>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody>")
        html.append("</table>")
        return "\n".join(html)


class BatchDoclingProcessor:
    """Process multiple documents efficiently with batching."""

    def __init__(
        self,
        config: DoclingEnhancementConfig | None = None,
        max_concurrent: int = 4,
    ):
        """Initialize batch processor."""
        self.config = config or DoclingEnhancementConfig()
        self.max_concurrent = max_concurrent
        self.converter = DoclingEnhancedConverter(config)

    async def process_batch(
        self,
        sources: list[str],
        progress_callback: callable | None = None,
    ) -> list[DoclingExtractionResult]:
        """Process multiple documents concurrently."""
        import asyncio

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(
            source: str, idx: int
        ) -> DoclingExtractionResult:
            async with semaphore:
                if progress_callback:
                    progress_callback(idx + 1, len(sources))
                return await self.converter.convert(source)

        tasks = [
            process_with_semaphore(source, idx) for idx, source in enumerate(sources)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {sources[idx]}: {result}")
                valid_results.append(
                    DoclingExtractionResult(
                        document_id=f"error_{idx}",
                        markdown_content=f"[Error: {result}]",
                    )
                )
            else:
                valid_results.append(result)

        return valid_results


def create_converter(
    extract_tables: bool = True,
    extract_code: bool = True,
    extract_images: bool = True,
    use_vlm: bool = True,
) -> DoclingEnhancedConverter:
    """Factory function to create enhanced converter."""
    config = DoclingEnhancementConfig(
        extract_tables=extract_tables,
        extract_code=extract_code,
        extract_images=extract_images,
        use_vlm_captioning=use_vlm,
    )
    return DoclingEnhancedConverter(config)
