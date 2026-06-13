"""Document classification and segmentation helpers for the unified pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from app.shared.rag.document_processing import IngestionConfig, chunk_document_simple

if TYPE_CHECKING:
    from app.shared.rag.document_processing.models import Chunk as SharedChunk


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
    if classified.document_kind in {"legal_contract", "legal_policy"}:
        return _segment_legal_chunks(parsed=parsed, classified=classified)

    shared_chunks = await _chunk_document(
        parsed=parsed,
        classified=classified,
    )
    chunks = [
        PreparedChunk(
            chunk_index=chunk.chunk_index,
            chunk_kind=chunk.metadata.get("chunk_method", "generic"),
            content=chunk.content,
            preamble=chunk.metadata.get("preamble", ""),
            clause_type=chunk.metadata.get("clause_type"),
            page_no=int(chunk.metadata.get("page_no", 0) or 0),
            metadata_=dict(chunk.metadata),
            custom_metadata={},
            quality_warnings=[],
        )
        for chunk in shared_chunks
    ]
    return chunks, []


async def _chunk_document(
    *, parsed: ParsedDocument, classified: ClassifiedDocument
) -> list[SharedChunk]:
    config = IngestionConfig(
        chunk_size=1000,
        chunk_overlap=200,
        max_chunk_size=2000,
        min_chunk_size=100,
        use_semantic_chunking=True,
        preserve_structure=True,
        max_tokens=512,
    )
    return await chunk_document_simple(
        content=parsed.markdown,
        title=parsed.title,
        source=parsed.title,
        config=config,
        metadata={
            "document_kind": classified.document_kind,
            "jurisdiction": classified.jurisdiction,
            "contract_type": classified.contract_type,
            "parties": classified.parties,
        },
    )


def _segment_legal_chunks(
    *,
    parsed: ParsedDocument,
    classified: ClassifiedDocument,
) -> tuple[list[PreparedChunk], list[QualityWarning]]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", parsed.markdown) if block.strip()]
    warnings: list[QualityWarning] = []
    if len(blocks) <= 1:
        warnings.append(
            QualityWarning(
                stage="segment_chunks",
                code="LEGAL_FALLBACK_SEGMENTATION",
                message="Clause-aware segmentation fell back to paragraph segmentation.",
                severity="warning",
            )
        )
    chunks: list[PreparedChunk] = []
    for index, block in enumerate(blocks[:200]):
        clause_type = _infer_clause_type(block)
        preamble = _build_preamble(classified=classified, clause_type=clause_type)
        chunk_warnings = warnings.copy() if len(blocks) <= 1 else []
        chunks.append(
            PreparedChunk(
                chunk_index=index,
                chunk_kind=classified.document_kind,
                content=block,
                preamble=preamble,
                clause_type=clause_type,
                metadata_={
                    "jurisdiction": classified.jurisdiction,
                    "contract_type": classified.contract_type,
                    "parties": classified.parties,
                },
                custom_metadata={"source": "clause_aware" if len(blocks) > 1 else "fallback"},
                quality_warnings=chunk_warnings,
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
