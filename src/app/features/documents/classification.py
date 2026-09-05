"""Document classification and segmentation helpers for the unified pipeline."""

from __future__ import annotations

import re
from typing import NamedTuple

import asyncer
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, ConfigDict, Field

from app.shared.rag.document_processing import IngestionConfig
from app.shared.rag.document_processing.chunker import (
    DEFAULT_TOKENIZER_MODEL_ID,
    create_hybrid_chunker,
    get_tokenizer,
)


class QualityWarning(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    code: str
    message: str
    severity: str


class ClassifiedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_kind: str
    jurisdiction: str | None = None
    contract_type: str | None = None
    parties: list[str] = Field(default_factory=list)
    metadata_: dict[str, object] = Field(default_factory=dict)
    graphiti_required: bool = False
    warnings: list[QualityWarning] = Field(default_factory=list)


class PreparedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: int
    chunk_kind: str
    content: str
    preamble: str = ""
    clause_type: str | None = None
    page_no: int = 0
    metadata_: dict[str, object] = Field(default_factory=dict)
    custom_metadata: dict[str, object] = Field(default_factory=dict)
    quality_warnings: list[QualityWarning] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    markdown: str
    page_count: int
    tables: list[str] = Field(default_factory=list)


def classify_document(*, markdown: str, filename: str) -> ClassifiedDocument:
    lowered = f"{filename}\n{markdown[:12000]}".lower()
    parties = _extract_parties(markdown)
    jurisdiction = _extract_jurisdiction(markdown)
    contract_type = _extract_contract_type(lowered)
    if _looks_like_policy(lowered):
        return ClassifiedDocument(
            document_kind="legal_policy",
            jurisdiction=jurisdiction,
            contract_type=contract_type,
            parties=parties,
            metadata_={"jurisdiction": jurisdiction, "contract_type": contract_type},
            graphiti_required=True,
        )
    if _looks_like_contract(lowered):
        return ClassifiedDocument(
            document_kind="legal_contract",
            jurisdiction=jurisdiction,
            contract_type=contract_type,
            parties=parties,
            metadata_={"jurisdiction": jurisdiction, "contract_type": contract_type},
            graphiti_required=True,
        )
    return ClassifiedDocument(document_kind="generic", parties=parties)


async def segment_chunks(
    *,
    parsed: ParsedDocument,
    classified: ClassifiedDocument,
) -> tuple[list[PreparedChunk], list[QualityWarning]]:
    # One structure-aware path for every document kind: the markdown is lifted
    # into a real document structure (headings + clause-bounded sections) and
    # the Docling HybridChunker decides chunk boundaries by token budget with
    # peer merging — never by blank-line pattern matching, and never truncated
    # to a fixed prefix. Legal documents additionally get clause typing; the
    # chunking itself is identical.
    return await _segment_hybrid_chunks(parsed=parsed, classified=classified)


class _StructuredItem(NamedTuple):
    kind: str  # "heading" or "text"
    text: str
    level: int = 1


class _HybridChunk(NamedTuple):
    text: str
    heading_path: tuple[str, ...]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# A paragraph that opens a new clause section. Numbered references (`12.`,
# `12.3)`, `(a)`) and named references (`Section 12`, `ARTICLE III`) start a
# section; anything else continues the current one. Deliberately textual rather
# than semantic: a missed marker merges two clauses into one section (still
# chunked, still typed), while an over-eager marker only splits a section the
# chunker's peer merging can rejoin.
_CLAUSE_START_RE = re.compile(
    r"(?i)^\s*(?:section|article|clause|schedule|exhibit|appendix|annex|part|paragraph)\b\s*\S+"
    r"|^\s*\d+(?:\.\d+)*[.)]\s+\S"
    r"|^\s*\([a-z0-9]+\)\s+\S"
)


def _structure_markdown(markdown: str) -> list[_StructuredItem]:
    """Lift markdown into heading + section items (cheap string ops, inline).

    Blank lines separate *paragraphs* — the atomic items below — but they never
    decide chunk boundaries; the chunker does that by token budget. Paragraphs
    group into one section until a heading or a clause start opens the next, so
    a multi-paragraph clause that fits the budget stays one atomic item and can
    never be split mid-clause by a merge.
    """
    items: list[_StructuredItem] = []
    paragraph_lines: list[str] = []

    def _flush_section() -> None:
        paragraph = "\n".join(paragraph_lines).strip()
        paragraph_lines.clear()
        if not paragraph:
            return
        if items and items[-1].kind == "text" and not _is_clause_start(paragraph):
            merged = f"{items[-1].text}\n\n{paragraph}"
            items[-1] = _StructuredItem(kind="text", text=merged)
        else:
            items.append(_StructuredItem(kind="text", text=paragraph))

    for line in markdown.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            _flush_section()
            items.append(
                _StructuredItem(
                    kind="heading", text=heading.group(2).strip(), level=len(heading.group(1))
                )
            )
            continue
        if not line.strip():
            _flush_section()
            continue
        paragraph_lines.append(line.rstrip())
    _flush_section()
    return [item for item in items if item.text]


def _is_clause_start(paragraph: str) -> bool:
    first_line = paragraph.splitlines()[0] if paragraph else ""
    return _CLAUSE_START_RE.match(first_line) is not None


def _run_hybrid_chunker_sync(
    *, title: str, items: list[_StructuredItem], max_tokens: int
) -> list[_HybridChunk]:
    """Build the structured document and chunk it (sync; callers offload).

    The tokenizer is the cached process-wide counter, so repeated chunking pays
    no reload; the chunker merges undersized peer sections within the token
    bound and carries each chunk's heading path in its metadata.
    """
    document = DoclingDocument(name=(title or "document")[:255])
    for item in items:
        if item.kind == "heading":
            document.add_heading(text=item.text, level=min(max(item.level, 1), 6))
        else:
            document.add_text(label=DocItemLabel.TEXT, text=item.text)
    chunker = create_hybrid_chunker(get_tokenizer(), IngestionConfig(max_tokens=max_tokens))
    chunks: list[_HybridChunk] = []
    for chunk in chunker.chunk(dl_doc=document):
        text = chunk.text.strip()
        if not text:
            continue
        # `meta` is typed as the chunker base metadata, which declares no
        # headings — read structurally rather than suppressing the checker.
        headings: tuple[str, ...] = tuple(getattr(chunk.meta, "headings", None) or ())
        chunks.append(_HybridChunk(text=text, heading_path=headings))
    return chunks


async def _segment_hybrid_chunks(
    *,
    parsed: ParsedDocument,
    classified: ClassifiedDocument,
) -> tuple[list[PreparedChunk], list[QualityWarning]]:
    items = _structure_markdown(parsed.markdown)
    warnings: list[QualityWarning] = []
    if len([item for item in items if item.kind == "text"]) <= 1:
        warnings.append(
            QualityWarning(
                stage="segment_chunks",
                code="DEGENERATE_PARSE",
                message=(
                    "Structure-aware segmentation saw one section or none "
                    f"in '{parsed.title}'."
                ),
                severity="warning",
            )
        )
    try:
        hybrid_chunks = await asyncer.asyncify(_run_hybrid_chunker_sync)(
            title=parsed.title, items=items, max_tokens=512
        )
    except Exception as exc:  # noqa: BLE001 — chunking must degrade, not fail ingestion
        exc.add_note("operation=hybrid_chunk")
        warnings.append(
            QualityWarning(
                stage="segment_chunks",
                code="HYBRID_CHUNKER_FALLBACK",
                message=(
                    "Structure-aware chunking failed "
                    f"({type(exc).__name__}); emitted atomic sections instead."
                ),
                severity="warning",
            )
        )
        hybrid_chunks = [
            _HybridChunk(text=item.text, heading_path=())
            for item in items
            if item.kind == "text"
        ]
    chunks: list[PreparedChunk] = []
    for index, hybrid_chunk in enumerate(hybrid_chunks):
        clause_type = _infer_clause_type(hybrid_chunk.text)
        preamble = _build_preamble(classified=classified, clause_type=clause_type)
        heading_path = " / ".join(hybrid_chunk.heading_path)
        if heading_path:
            preamble = f"{preamble} Section path: {heading_path}."
        chunks.append(
            PreparedChunk(
                chunk_index=index,
                chunk_kind=classified.document_kind,
                content=hybrid_chunk.text,
                preamble=preamble,
                clause_type=clause_type,
                metadata_={
                    "jurisdiction": classified.jurisdiction,
                    "contract_type": classified.contract_type,
                    "parties": classified.parties,
                    "heading_path": list(hybrid_chunk.heading_path),
                    "tokenizer": DEFAULT_TOKENIZER_MODEL_ID,
                },
                custom_metadata={"source": "hybrid"},
                quality_warnings=warnings.copy() if warnings else [],
            )
        )
    return chunks, warnings


def _looks_like_contract(text: str) -> bool:
    keywords = ("agreement", "party", "parties", "governing law", "termination", "indemn")
    return sum(keyword in text for keyword in keywords) >= 2


def _looks_like_policy(text: str) -> bool:
    keywords = ("privacy policy", "terms of service", "data retention", "cookie policy")
    return any(keyword in text for keyword in keywords)


def _extract_jurisdiction(markdown: str) -> str | None:
    match = re.search(r"govern(?:ed|ing) law[^\n:]*[:\-]?\s*([A-Za-z ,]+)", markdown, re.IGNORECASE)
    if match:
        return match.group(1).strip()[:255]
    return None


def _extract_contract_type(text: str) -> str | None:
    contract_types = {
        "nda": "nda",
        "non-disclosure": "nda",
        "msa": "msa",
        "master services": "msa",
        "employment": "employment",
        "lease": "lease",
        "privacy policy": "privacy_policy",
        "terms of service": "terms_of_service",
    }
    for needle, value in contract_types.items():
        if needle in text:
            return value
    return None


def _extract_parties(markdown: str) -> list[str]:
    matches = re.findall(r"between\s+([^\n]+?)\s+and\s+([^\n,.]+)", markdown, re.IGNORECASE)
    if not matches:
        return []
    first = matches[0]
    return [first[0].strip()[:255], first[1].strip()[:255]]


def _infer_clause_type(text: str) -> str:
    lowered = text.lower()
    mapping = {
        "termination": "termination",
        "indemn": "indemnity",
        "governing law": "governing_law",
        "confidential": "confidentiality",
        "payment": "payment",
        "liability": "limitation_of_liability",
        "arbitr": "arbitration",
    }
    for needle, clause_type in mapping.items():
        if needle in lowered:
            return clause_type
    return "other"


def _build_preamble(*, classified: ClassifiedDocument, clause_type: str) -> str:
    parties = " and ".join(classified.parties) if classified.parties else "unknown parties"
    contract_type = classified.contract_type or classified.document_kind
    return f"This is a {clause_type} section from a {contract_type} document involving {parties}."
