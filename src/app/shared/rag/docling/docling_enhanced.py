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
from collections.abc import Awaitable, Callable
from io import BytesIO, StringIO
from typing import Any, NamedTuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import BaseError as DoclingError
from docling_core.types.doc import DoclingDocument
from google import genai
from PIL import Image

from app.utils import logger

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
            device_name = torch.cuda.get_device_name(0)
            logger.bind(device_name=device_name).info("GPU detected")
        else:
            logger.info("No GPU detected, using CPU pipeline")
    except ImportError:
        logger.warning("PyTorch not available, using CPU pipeline")
        return False
    else:
        return gpu_available


def create_document_converter(gpu_available: bool) -> DocumentConverter:
    """Create Docling converter with appropriate pipeline."""
    if gpu_available:
        logger.info("Using GPU-accelerated PDF pipeline")
        pipeline_options = PdfPipelineOptions()
    else:
        logger.info("Using CPU-efficient pipeline")
        pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def table_markdown(doc: DoclingDocument) -> list[str]:
    """Every table in ``doc`` rendered as markdown, in document order.

    One expression replacing three copies, two of which called ``table.to_markdown()`` — a method
    ``TableItem`` does not define. The real name is ``export_to_markdown``. The copies failed
    differently from the same typo: the KB ingestion node guarded the call with
    ``hasattr(table, "to_markdown")``, so the guard was false for every table and the node
    returned an empty list for every document ever parsed; ``extract_tables`` below called it
    bare, and because ``AttributeError`` is not a ``docling.exceptions.BaseError`` its ``except``
    clause never caught it. A silent nothing and an uncaught crash, from one wrong method name.

    ``doc=`` is passed rather than omitted. Without it the call logs a deprecation warning and
    takes a fallback that walks ``self.data.grid`` directly; with it the table goes through
    ``MarkdownDocSerializer``, which is the path that can resolve what a cell refers to.
    """
    return [table.export_to_markdown(doc=doc) for table in doc.tables]


def extract_tables(doc: DoclingDocument) -> list[ExtractedTable]:
    """Extract tables from document."""
    tables = []

    try:
        if not isinstance(doc, DoclingDocument):
            return tables

        for idx, table in enumerate(doc.tables):
            try:
                md_table = table.export_to_markdown(doc=doc)
                rows = md_table.split("\n")
                row_count = len([r for r in rows if r.strip() and not r.startswith("|---")])
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
            # Deliberately not catching `AttributeError`. It was the *uncaught* exception here
            # for as long as this function has existed, because the call named a method that does
            # not exist — and a clause broad enough to swallow it would have logged "failed to
            # extract table 3" for every table of every document instead of surfacing a wrong API
            # name. `ValueError`/`IndexError`/`KeyError` are what a malformed cell grid raises,
            # which is a per-table problem worth skipping; a missing attribute is a per-build
            # problem worth crashing on.
            except (DoclingError, ValueError, IndexError, KeyError) as e:
                e.add_note(f"table_index={idx}, operation=extract_table")
                logger.bind(table_index=idx, operation="extract_table", error=str(e)).opt(
                    exception=True
                ).warning("Failed to extract table")

    except ImportError:
        logger.warning("docling_core not available for table extraction")

    logger.bind(table_count=len(tables)).info("Extracted tables")
    return tables


def extract_code_blocks(doc: DoclingDocument) -> list[ExtractedCodeBlock]:
    """Extract code blocks with language detection."""
    code_blocks = []

    try:
        if not isinstance(doc, DoclingDocument):
            return code_blocks

        for idx, item in enumerate(doc._iterate_nodes()):
            try:
                if (
                    hasattr(item, "meta")
                    and hasattr(item.meta, "text_type")
                    and item.meta.text_type == "code"
                ):
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
            except DoclingError as e:
                e.add_note(f"block_index={idx}, operation=extract_code_block")
                logger.bind(block_index=idx, operation="extract_code_block", error=str(e)).opt(
                    exception=True
                ).warning("Failed to extract code block")

    except ImportError:
        logger.warning("docling_core not available for code extraction")

    if not code_blocks:
        code_blocks = _extract_code_fallback(doc.export_to_markdown())

    logger.bind(code_block_count=len(code_blocks)).info("Extracted code blocks")
    return code_blocks


def _detect_language(item, code_text: str) -> str:
    """Detect programming language from metadata or content."""
    if hasattr(item.meta, "language"):
        return item.meta.language

    if "def " in code_text and ":" in code_text:
        return "python"
    if "function " in code_text or "const " in code_text or "let " in code_text:
        return "javascript"
    if "class " in code_text and "{" in code_text:
        return "java"
    if "#include" in code_text or "int main" in code_text:
        return "c"
    if "package " in code_text and "func " in code_text:
        return "go"
    if "fn " in code_text and "->" in code_text:
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
    doc: DoclingDocument, _source_path: str, use_vlm_captioning: bool = True
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
            except DoclingError as e:
                e.add_note(f"image_index={idx}, operation=extract_image")
                logger.bind(image_index=idx, operation="extract_image", error=str(e)).opt(
                    exception=True
                ).warning("Failed to extract image")

    except ImportError:
        logger.warning("docling_core not available for image extraction")

    logger.bind(image_count=len(images)).info("Extracted images")
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

    except Exception as e:  # noqa: BLE001 — VLM API can raise varied provider errors
        e.add_note("operation=vlm_caption")
        logger.bind(operation="vlm_caption", error=str(e)).opt(exception=True).warning(
            "VLM captioning failed"
        )
        return None
    else:
        return response.text or None


def _encode_base64(data: bytes) -> str:
    """Encode image data to base64."""
    return base64.b64encode(data).decode("utf-8")


def _markdown_to_csv(md_table: str) -> str:
    """Convert markdown table to CSV."""
    lines = [line for line in md_table.split("\n") if line.strip() and not line.startswith("|---")]

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
    lines = [line for line in md_table.split("\n") if line.strip() and not line.startswith("|---")]

    if not lines:
        return "<table></table>"

    header = [cell.strip() for cell in lines[0].split("|")[1:-1]]
    rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        rows.append(cells)

    html = ['<table border="1">']
    html.append("<thead><tr>")
    html.extend(f"<th>{cell}</th>" for cell in header)
    html.extend(["</tr></thead>", "<tbody>"])
    for row in rows:
        html.append("<tr>")
        html.extend(f"<td>{cell}</td>" for cell in row)
        html.append("</tr>")
    html.extend(["</tbody>", "</table>"])
    return "\n".join(html)


class ExtractionStage(NamedTuple):
    """One `convert_document` enrichment: when it runs and how to run it.

    `enabled` reads the existing `DoclingEnhancementConfig` flags — the config
    model stays the only toggle store, there is no second flag registry.
    `run` is always async so the loop below awaits uniformly; the synchronous
    extractors are wrapped in trivial coroutines.
    """

    name: str
    enabled: Callable[[DoclingEnhancementConfig], bool]
    run: Callable[[DoclingDocument, str, DoclingEnhancementConfig], Awaitable[Any]]


async def _run_doctags_stage(
    doc: DoclingDocument, source: str, _config: DoclingEnhancementConfig
) -> str | None:
    """DocTags export; a failed export degrades to None, never to a failed convert."""
    try:
        return doc.export_to_doc_tags()
    except DoclingError as e:
        e.add_note(f"document={source}, operation=export_doctags")
        logger.bind(document=source, operation="export_doctags", error=str(e)).opt(
            exception=True
        ).warning("DocTags export failed")
        return None


async def _run_tables_stage(
    doc: DoclingDocument, _source: str, _config: DoclingEnhancementConfig
) -> list[ExtractedTable]:
    """Table extraction stage."""
    return extract_tables(doc)


async def _run_code_stage(
    doc: DoclingDocument, _source: str, _config: DoclingEnhancementConfig
) -> list[ExtractedCodeBlock]:
    """Code-block extraction stage."""
    return extract_code_blocks(doc)


async def _run_images_stage(
    doc: DoclingDocument, source: str, config: DoclingEnhancementConfig
) -> list[ExtractedImage]:
    """Image extraction stage, honouring the VLM-captioning toggle."""
    return await extract_images(doc, source, config.use_vlm_captioning)


# Ordered enrichment registry for `convert_document`. Adding a stage is a
# registration here — the gate loop below does not change.
EXTRACTION_STAGES: tuple[ExtractionStage, ...] = (
    ExtractionStage(
        name="doctags",
        enabled=lambda config: config.generate_doctags,
        run=_run_doctags_stage,
    ),
    ExtractionStage(
        name="tables", enabled=lambda config: config.extract_tables, run=_run_tables_stage
    ),
    ExtractionStage(name="code", enabled=lambda config: config.extract_code, run=_run_code_stage),
    ExtractionStage(
        name="images", enabled=lambda config: config.extract_images, run=_run_images_stage
    ),
)


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
        document_id = hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()[:12]

    if gpu_available is None:
        gpu_available = check_gpu_available()

    if converter is None:
        converter = create_document_converter(gpu_available)

    logger.bind(source=source).info("Converting document")

    try:
        result = converter.convert(source)
        doc = result.document
    except DoclingError as e:
        e.add_note(f"document={source}, operation=convert")
        logger.bind(document=source, operation="convert").exception("Docling conversion failed")
        return DoclingExtractionResult(
            document_id=document_id,
            markdown_content=f"[Conversion error: {e}]",
        )

    markdown_content = doc.export_to_markdown()

    stage_outputs: dict[str, Any] = {}
    for stage in EXTRACTION_STAGES:
        if not stage.enabled(config):
            continue
        stage_outputs[stage.name] = await stage.run(doc, source, config)

    doctags_content = stage_outputs.get("doctags")
    tables = stage_outputs.get("tables", [])
    code_blocks = stage_outputs.get("code", [])
    images = stage_outputs.get("images", [])

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
    progress_callback: Callable[..., None] | None = None,
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

    tasks = [process_with_semaphore(source, idx) for idx, source in enumerate(sources)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.bind(source=sources[idx], error=str(result)).error("Failed to process document")
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
