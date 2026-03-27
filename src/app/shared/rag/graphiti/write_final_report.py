"""
Persist memory functions: write_final_report_to_memory

Called by persist_memory_node.  Not @tool — this is a direct node call,
not LLM-mediated.  The LLM never decides when to persist memory; the node
always does it unconditionally after human approval.

Memory Router logic (Section 18.8) lives here — decides what goes where:

  Graphiti:  final report → high-trust episode (group_id=user_id)
             relationships → already written by relationship_mapping node
  Cognee:    final report → episodic memory (queryable insights)
             relationships summary → procedural memory

Trust score assignment:
  human_approved=True  → trust_score=1.0 (highest; this is ground truth)
  human_approved=False → trust_score=0.3 (low; should not happen in prod)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.utils import logger

from .schemas import FinalReportEpisodeMetadata

if TYPE_CHECKING:
    from app.shared.agents.memory.cognee_client import CogneeService
    from app.shared.langgraph_layer.agent_saul.graph.state import FinalReport, LegalRelationship

    from .client import GraphitiService


class MemoryPersistResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    graphiti_report_uuid: str | None
    cognee_dataset: str | None
    success: bool
    errors: list[str]


async def write_final_report_to_memory(
    report: FinalReport,
    relationships: list[LegalRelationship],
    user_id: str,
    thread_id: str,
    human_approved: bool,
    overall_risk_label: str,
    overall_compliant: bool,
    graphiti_service: GraphitiService,
    cognee_service: CogneeService,
) -> MemoryPersistResult:
    """Memory Router: decides what to write to Graphiti vs Cognee.

    Memory routing decisions:
      Graphiti:
        - Final report → high-trust episode (trust_score=1.0 if human_approved)
          group_id=user_id → cross-document precedent queries work correctly
      Cognee:
        - Final report JSON → episodic memory (INSIGHTS search)
        - Relationships summary text → procedural memory

    Partial failures are captured without raising — persist_memory_node
    logs them but marks workflow COMPLETED (memory write failure is not
    a pipeline failure; the legal analysis is already done).
    """
    log = logger.bind(
        doc_id=report.document_id,
        user_id=user_id,
        human_approved=human_approved,
    )
    errors: list[str] = []
    graphiti_uuid: str | None = None
    cognee_dataset: str | None = None

    # ---- Graphiti: final report as high-trust episode --------------------
    try:
        metadata = FinalReportEpisodeMetadata(
            doc_id=report.document_id,
            user_id=user_id,
            thread_id=thread_id,
            overall_risk_label=overall_risk_label,
            overall_compliant=overall_compliant,
            human_approved=human_approved,
        )
        graphiti_uuid = await graphiti_service.write_final_report_episode(
            report_summary=report.summary,
            metadata=metadata,
        )
        log.info("memory_graphiti_report_written", uuid=graphiti_uuid)
    except Exception as exc:
        errors.append(f"graphiti_report: {exc}")
        log.exception("memory_graphiti_report_failed", error=str(exc))

    # ---- Cognee: episodic memory (full report JSON for INSIGHTS search) --
    try:
        report_json = report.model_dump_json()
        await cognee_service.store_final_report(
            report_json=report_json,
            user_id=user_id,
            doc_id=report.document_id,
            thread_id=thread_id,
        )
        cognee_dataset = f"{user_id}.legal_reports"
        log.info("memory_cognee_report_stored", dataset=cognee_dataset)
    except Exception as exc:
        errors.append(f"cognee_report: {exc}")
        log.exception("memory_cognee_report_failed", error=str(exc))

    # ---- Cognee: procedural memory (relationship patterns) ---------------
    if relationships:
        try:
            rel_lines = [
                f"{r.from_node} {r.relationship} {r.to_node} (clause: {r.clause_id})"
                for r in relationships
            ]
            relationships_text = (
                f"Document: {report.document_id}\n"
                f"User: {user_id}\n"
                f"Relationships:\n"
                + "\n".join(rel_lines)
            )
            await cognee_service.store_relationships(
                relationships_text=relationships_text,
                user_id=user_id,
                doc_id=report.document_id,
            )
            log.info("memory_cognee_relationships_stored")
        except Exception as exc:
            errors.append(f"cognee_relationships: {exc}")
            log.exception("memory_cognee_relationships_failed", error=str(exc))

    return MemoryPersistResult(
        graphiti_report_uuid=graphiti_uuid,
        cognee_dataset=cognee_dataset,
        success=len(errors) == 0,
        errors=errors,
    )
