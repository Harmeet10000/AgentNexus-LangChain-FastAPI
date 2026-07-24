"""Legal metadata extraction and chunk enrichment helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.shared.langchain_layer.models import serialize_to_toon
from app.shared.langgraph_layer.kb_retry import retry_immediate

from .classification import QualityWarning

if TYPE_CHECKING:
    from .classification import ClassifiedDocument, PreparedChunk


class StructuredOutputLLM(Protocol):
    def with_structured_output(self, schema: type[LegalMetadataExtraction]) -> object: ...


class LegalMetadataExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_type: str | None = None
    parties: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    governing_law: str | None = None
    effective_date: str | None = None
    contract_signed: str | None = None
    amendment_effective: str | None = None
    expiry_date: str | None = None
    document_summary: str = ""


async def extract_legal_metadata(
    *,
    llm: object,
    markdown: str,
    classified: ClassifiedDocument,
) -> tuple[LegalMetadataExtraction, list[QualityWarning]]:
    warnings: list[QualityWarning] = []
    if hasattr(llm, "with_structured_output"):
        structured_llm = cast("StructuredOutputLLM", llm).with_structured_output(
            LegalMetadataExtraction
        )
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(
                content=(
                    "Extract legal metadata from the document. Return only contract type, parties, "
                    "jurisdiction, governing law, effective date, contract signed date, amendment "
                    "effective date, expiry date, and document summary."
                )
            ),
            HumanMessage(
                content=serialize_to_toon(
                    {
                        "document_kind": classified.document_kind,
                        "heuristic_contract_type": classified.contract_type,
                        "heuristic_parties": classified.parties,
                        "heuristic_jurisdiction": classified.jurisdiction,
                        "markdown": markdown[:50_000],
                    }
                )
            ),
        ]
        try:
            raw = await retry_immediate(
                operation=lambda: structured_llm.ainvoke(messages),
                label="documents_extract_legal_metadata",
            )
            extracted = LegalMetadataExtraction.model_validate(raw)
            return _merge_metadata(extracted=extracted, classified=classified), warnings
        except (ValueError, TypeError):
            warnings.append(
                QualityWarning(
                    stage="classify_document",
                    code="LEGAL_METADATA_FALLBACK",
                    message="Structured legal metadata extraction fell back to heuristic parsing.",
                    severity="warning",
                )
            )

    return _heuristic_metadata(markdown=markdown, classified=classified), warnings


def enrich_legal_chunks(
    *,
    chunks: list[PreparedChunk],
    classified: ClassifiedDocument,
    metadata: LegalMetadataExtraction,
) -> list[PreparedChunk]:
    parties = metadata.parties or classified.parties
    contract_type = metadata.contract_type or classified.contract_type or classified.document_kind
    effective_date = metadata.effective_date or metadata.contract_signed or "unknown date"
    enriched: list[PreparedChunk] = []
    for chunk in chunks:
        preamble = (
            f"This is a {chunk.clause_type or 'legal'} section from a {contract_type} document between "
            f"{' and '.join(parties) if parties else 'unknown parties'}, effective {effective_date}."
        )
        metadata_map = {
            **chunk.metadata_,
            "jurisdiction": metadata.jurisdiction or classified.jurisdiction,
            "contract_type": contract_type,
            "parties": parties,
            "governing_law": metadata.governing_law,
            "effective_date": metadata.effective_date,
            "contract_signed": metadata.contract_signed,
            "amendment_effective": metadata.amendment_effective,
            "expiry_date": metadata.expiry_date,
        }
        enriched.append(chunk.model_copy(update={"preamble": preamble, "metadata_": metadata_map}))
    return enriched


def contract_event_dates(metadata: LegalMetadataExtraction) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    if metadata.contract_signed:
        events.append(("contract_signed", metadata.contract_signed))
    if metadata.amendment_effective:
        events.append(("amendment_effective", metadata.amendment_effective))
    if metadata.expiry_date:
        events.append(("expiry_date", metadata.expiry_date))
    return events


def _merge_metadata(
    *,
    extracted: LegalMetadataExtraction,
    classified: ClassifiedDocument,
) -> LegalMetadataExtraction:
    return extracted.model_copy(
        update={
            "contract_type": extracted.contract_type or classified.contract_type,
            "parties": extracted.parties or classified.parties,
            "jurisdiction": extracted.jurisdiction or classified.jurisdiction,
        }
    )


def _heuristic_metadata(
    *,
    markdown: str,
    classified: ClassifiedDocument,
) -> LegalMetadataExtraction:
    return LegalMetadataExtraction(
        contract_type=classified.contract_type,
        parties=classified.parties,
        jurisdiction=classified.jurisdiction,
        governing_law=_extract_text_after_label(markdown, label_regex=r"govern(?:ed|ing) law"),
        effective_date=_extract_text_after_label(markdown, label_regex=r"effective date"),
        contract_signed=_extract_text_after_label(markdown, label_regex=r"signed on"),
        amendment_effective=_extract_text_after_label(markdown, label_regex=r"amendment effective"),
        expiry_date=_extract_text_after_label(markdown, label_regex=r"expiry date|expiration date|expires on"),
        document_summary=_first_nonempty_paragraph(markdown),
    )


def _extract_text_after_label(markdown: str, label_regex: str) -> str | None:
    match: re.Match[str] | None = re.search(pattern=rf"{label_regex}[^\n:]*[:\-]?\s*([^\n]+)", string=markdown, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()[:255]


def _first_nonempty_paragraph(markdown: str) -> str:
    paragraphs = [
        paragraph.strip() for paragraph in re.split(r"\n\s*\n", markdown) if paragraph.strip()
    ]
    if not paragraphs:
        return ""
    return paragraphs[0][:500]
