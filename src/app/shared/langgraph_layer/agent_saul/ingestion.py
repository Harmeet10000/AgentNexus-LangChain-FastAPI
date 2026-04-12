"""
Node: Ingestion Agent

Tool-driven.  No LLM reasoning.  Retry-capable.
Loads document by doc_id, extracts text via Docling/OCR.

HITL:
  Only if OCR confidence < threshold → interrupt() asking for manual re-upload.

Tools (stubs — wire in your Docling/OCR implementations):
  - load_document_from_store(doc_id) → raw bytes
  - extract_text_with_docling(bytes) → (text, confidence)
  - extract_text_with_ocr(bytes) → (text, confidence)

This node does NOT use an LLM.  It is a deterministic I/O node.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.types import interrupt

from app.utils import logger

from ..state import (
    AgentError,
    HITLInterruptType,
    LegalAgentState,
    WorkflowStatus,
)

_OCR_CONFIDENCE_THRESHOLD = 0.85
_MAX_RETRIES = 3


def make_ingestion_node() -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def ingestion_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="ingestion",
            doc_id=state["doc_id"],
            user_id=state["user_id"],
        )
        log.info("ingestion_started")

        retry_count = state.get("retry_count", 0)
        if retry_count >= _MAX_RETRIES:
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="ingestion",
                        code="MAX_RETRIES_EXCEEDED",
                        message=f"Ingestion failed after {_MAX_RETRIES} attempts",
                        retryable=False,
                    )
                ],
            }

        # --- Step 1: Load raw document bytes --------------------------------
        # TODO: replace stub with actual document store lookup
        # raw_bytes = await load_document_from_store(state["doc_id"])
        _raw_bytes: bytes = b""  # stub

        # --- Step 2: Extract text (Docling first, OCR fallback) -------------
        # TODO: replace stubs with real Docling/OCR integrations
        # text, confidence = await extract_text_with_docling(raw_bytes)
        # if confidence < _OCR_CONFIDENCE_THRESHOLD:
        #     text, confidence = await extract_text_with_ocr(raw_bytes)
        text: str = ""  # stub
        confidence: float = 1.0  # stub

        if confidence < _OCR_CONFIDENCE_THRESHOLD:
            log.warning("ingestion_low_ocr_confidence", confidence=confidence)
            # HITL: ask user to re-upload a cleaner version
            human_response: dict[str, Any] = interrupt(
                {
                    "type": HITLInterruptType.OCR_REUPLOAD,
                    "confidence": confidence,
                    "message": (
                        f"Document OCR confidence ({confidence:.0%}) is below threshold. "
                        "Please re-upload a higher-quality scan."
                    ),
                }
            )
            # If user provides a new doc_id, update state and retry
            new_doc_id: str | None = human_response.get("new_doc_id")
            if new_doc_id:
                return {
                    "doc_id": new_doc_id,
                    "retry_count": retry_count + 1,
                    "status": WorkflowStatus.INGESTING,
                }
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="ingestion",
                        code="LOW_OCR_CONFIDENCE",
                        message=f"OCR confidence {confidence:.0%} — user declined re-upload",
                        retryable=False,
                    )
                ],
            }

        log.info("ingestion_completed", text_length=len(text), confidence=confidence)
        return {
            "document_text": text,
            "status": WorkflowStatus.NORMALIZING,
        }

    return ingestion_node
