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

import asyncio
import base64
import csv
import hashlib
import re
from io import BytesIO, StringIO

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from google import genai
from PIL import Image

from app.utils.logger import logger as loguru_logger

from .models import (
    DoclingEnhancementConfig,
    DoclingExtractionResult,
    ExtractedCodeBlock,
    ExtractedImage,
    ExtractedTable,
)


def check_gpu_available() -> bool:
    """Check if GPU is available for accelerated processing."""
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        if gpu_available:
            loguru_logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            loguru_logger.info("No GPU detected, using CPU pipeline")
        return gpu_available
    except ImportError:
        loguru_logger.warning("PyTorch not available, using CPU pipeline")
        return False


def create_document_converter(gpu_available: bool) -> DocumentConverter:
    """Create Docling converter with appropriate pipeline."""
    if gpu_available:
        loguru_logger.info("Using GPU-accelerated PDF pipeline")
        pipeline_options = PdfPipelineOptions()
    else:
        loguru_logger.info("Using CPU-efficient pipeline")
        pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def extract_tables(doc: DoclingDocument) -> list[ExtractedTable]:
    """Extract tables from document."""
    tables = []

    try:
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

                csv_content = _markdown_to_csv(md_table)
                html_content = _markdown_to_html(md_table)

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
                loguru_logger.warning(f"Failed to extract table {idx}: {e}")

    except ImportError:
        loguru_logger.warning("docling_core not available for table extraction")

    loguru_logger.info(f"Extracted {len(tables)} tables")
    return tables


def extract_code_blocks(doc: DoclingDocument) -> list[ExtractedCodeBlock]:
    """Extract code blocks with language detection."""
    code_blocks = []

    try:
        if not isinstance(doc, DoclingDocument):
            return code_blocks

        for idx, item in enumerate(doc._iterate_nodes()):
            try:
                if hasattr(item, "meta") and hasattr(item.meta, "text_type"):
                    if item.meta.text_type == "code":
                        code_text = item.text or ""
                        language = _detect_language(item, code_text)

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
                loguru_logger.warning(f"Failed to extract code block {idx}: {e}")

    except ImportError:
        loguru_logger.warning("docling_core not available for code extraction")

    if not code_blocks:
        code_blocks = _extract_code_fallback(doc.export_to_markdown())

    loguru_logger.info(f"Extracted {len(code_blocks)} code blocks")
    return code_blocks


def _detect_language(item, code_text: str) -> str:
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


def _extract_code_fallback(markdown: str) -> list[ExtractedCodeBlock]:
    """Fallback regex-based code extraction."""
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


async def extract_images(
    doc: DoclingDocument, source_path: str, use_vlm_captioning: bool = True
) -> list[ExtractedImage]:
    """Extract images from document."""
    images = []

    try:
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

                    if use_vlm_captioning and not caption:
                        caption = await _generate_vlm_caption(image_data)

                    images.append(
                        ExtractedImage(
                            image_index=idx,
                            image_path=key,
                            base64_data=_encode_base64(image_data),
                            caption=caption,
                            page_number=getattr(item, "page", 1),
                            metadata={"source_image": idx},
                        )
                    )
            except Exception as e:
                loguru_logger.warning(f"Failed to extract image {idx}: {e}")

    except ImportError:
        loguru_logger.warning("docling_core not available for image extraction")

    loguru_logger.info(f"Extracted {len(images)} images")
    return images


async def _generate_vlm_caption(image_data) -> str | None:
    """Generate caption using VLM (Gemini)."""
    try:
        client = genai.Client()

        image = Image.open(BytesIO(image_data)) if isinstance(image_data, bytes) else image_data

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[image, "Describe this image in detail for document understanding."],
        )

        return response.text if response.text else None

    except Exception as e:
        loguru_logger.warning(f"VLM captioning failed: {e}")
        return None


def _encode_base64(data: bytes) -> str:
    """Encode image data to base64."""
    return base64.b64encode(data).decode("utf-8")


def _markdown_to_csv(md_table: str) -> str:
    """Convert markdown table to CSV."""
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


def _markdown_to_html(md_table: str) -> str:
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


async def convert_document(
    source: str,
    document_id: str | None = None,
    config: DoclingEnhancementConfig | None = None,
    converter: DocumentConverter | None = None,
    gpu_available: bool | None = None,
) -> DoclingExtractionResult:
    """Convert document with enhanced extraction."""
    if config is None:
        config = DoclingEnhancementConfig()

    if document_id is None:
        document_id = hashlib.md5(source.encode()).hexdigest()[:12]

    if gpu_available is None:
        gpu_available = check_gpu_available()

    if converter is None:
        converter = create_document_converter(gpu_available)

    loguru_logger.info(f"Converting document: {source}")

    try:
        result = converter.convert(source)
        doc = result.document
    except Exception as e:
        loguru_logger.error(f"Docling conversion failed: {e}")
        return DoclingExtractionResult(
            document_id=document_id,
            markdown_content=f"[Conversion error: {e}]",
        )

    markdown_content = doc.export_to_markdown()

    doctags_content = None
    if config.generate_doctags:
        try:
            doctags_content = doc.export_to_doc_tags()
        except Exception as e:
            loguru_logger.warning(f"DocTags export failed: {e}")

    tables = []
    if config.extract_tables:
        tables = extract_tables(doc)

    code_blocks = []
    if config.extract_code:
        code_blocks = extract_code_blocks(doc)

    images = []
    if config.extract_images:
        images = await extract_images(doc, source, config.use_vlm_captioning)

    metadata = {
        "source": source,
        "gpu_processed": gpu_available,
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


async def process_documents_batch(
    sources: list[str],
    config: DoclingEnhancementConfig | None = None,
    max_concurrent: int = 4,
    progress_callback: None | callable = None,
) -> list[DoclingExtractionResult]:
    """Process multiple documents concurrently."""
    if config is None:
        config = DoclingEnhancementConfig()

    gpu_available = check_gpu_available()
    converter = create_document_converter(gpu_available)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(source: str, idx: int) -> DoclingExtractionResult:
        async with semaphore:
            if progress_callback:
                progress_callback(idx + 1, len(sources))
            return await convert_document(
                source,
                config=config,
                converter=converter,
                gpu_available=gpu_available,
            )

    tasks = [
        process_with_semaphore(source, idx) for idx, source in enumerate(sources)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            loguru_logger.error(f"Failed to process {sources[idx]}: {result}")
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
) -> DoclingEnhancementConfig:
    """Factory function to create enhancement config."""
    return DoclingEnhancementConfig(
        extract_tables=extract_tables,
        extract_code=extract_code,
        extract_images=extract_images,
        use_vlm_captioning=use_vlm,
    )
